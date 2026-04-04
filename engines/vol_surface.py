"""
engines/vol_surface.py
========================
Deribit volatility surface data fetching and feature extraction.

Two data sources:
    1. DVOL index — Deribit's 30-day constant-maturity IV index (like VIX).
       Free historical data via public API. Primary IV signal.
       URL: https://www.deribit.com/api/v2/public/get_historical_volatility

    2. Options chain snapshot — current live options for surface shape.
       Used for: term structure, skew (25Δ risk reversal), convexity (butterfly).
       URL: https://www.deribit.com/api/v2/public/get_book_summary_by_currency

Features extracted:
    iv_atm_7d      : ATM implied vol at ~7d expiry (from chain, or interp from DVOL)
    iv_atm_30d     : ATM implied vol at ~30d expiry (DVOL primary)
    term_slope     : (iv_30d - iv_7d) / iv_30d  — contango vs backwardation
    skew_25d       : 25-delta risk reversal (put IV - call IV) — fear indicator
    butterfly_25d  : 25-delta butterfly — fat tails / kurtosis
    vov_30d        : vol-of-vol (std of daily DVOL changes, last 30d)
    iv_pct_rank    : IV percentile rank over last 252 days (0=low, 1=high)
    iv_trend       : (iv_30d_now - iv_30d_21d_ago) / iv_30d_21d_ago
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VolSurfaceSnapshot:
    """Point-in-time vol surface features for one asset."""
    asset:          str
    timestamp:      datetime
    iv_atm_7d:      float       # ATM IV at 7d (annualized fraction, e.g. 0.75)
    iv_atm_14d:     float
    iv_atm_30d:     float       # DVOL index value
    iv_atm_60d:     float
    term_slope:     float       # (iv_30d - iv_7d) / iv_7d
    skew_25d:       float       # put25d_IV - call25d_IV (positive = fear/skew)
    butterfly_25d:  float       # (put25d + call25d)/2 - atm  (convexity)
    vov_30d:        float       # vol-of-vol
    iv_pct_rank:    float       # [0, 1] — how elevated is IV vs history
    iv_trend:       float       # IV change over last 21d


@dataclass
class VolFeatures:
    """Flattened feature vector for use in CPO."""
    asset:     str
    date:      str
    features:  np.ndarray     # shape (11,) matching VOL_SURFACE_FEATURE_NAMES


VOL_SURFACE_FEATURE_NAMES = [
    "iv_atm_7d",
    "iv_atm_30d",
    "term_slope",
    "skew_25d",
    "butterfly_25d",
    "vov_30d",
    "iv_pct_rank",
    "iv_trend",
    "iv_vs_garch_7d",     # IV_7d - GARCH_7d (set externally)
    "iv_vs_garch_30d",    # IV_30d - GARCH_30d
    "garch_persistence",  # GARCH alpha+beta (vol clustering intensity)
]


# ── Deribit API helpers ───────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None, timeout: int = 10) -> dict:
    """Call Deribit public API. Returns parsed JSON result."""
    url = f"{DERIBIT_BASE}/{endpoint}"
    if params:
        qstr = "&".join(f"{k}={v}" for k, v in params.items())
        url  = f"{url}?{qstr}"
    req  = urllib.request.Request(url, headers={"User-Agent": "praxis/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if "error" in data:
        raise ValueError(f"Deribit error: {data['error']}")
    return data["result"]


# ── DVOL historical data ──────────────────────────────────────────────────────

def fetch_dvol_history(
    asset:     str,
    start:     str,           # "2023-01-01"
    end:       str,           # "2024-12-31"
    cache_dir: Path | None = None,
) -> pd.Series:
    """
    Fetch Deribit DVOL historical index (30-day constant-maturity IV).

    Returns daily series of annualized IV (as decimal fraction, e.g. 0.75 = 75%).
    Index: UTC dates.

    Deribit returns: [[timestamp_ms, vol_percent], ...]
    vol_percent is already annualized (e.g. 75.3 means 75.3% annual vol).
    """
    currency = asset.upper()

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"dvol_{currency}_{start}_{end}.parquet"
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                return df["dvol"]
            except Exception:
                pass

    logger.info(f"  Fetching DVOL {currency} {start}→{end}...")
    result = _get("get_historical_volatility", {"currency": currency})

    # result is [[timestamp_ms, vol_pct], ...]
    rows = []
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")

    for ts_ms, vol_pct in result:
        ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
        if start_ts <= ts <= end_ts:
            rows.append({"date": ts.normalize(), "dvol": vol_pct / 100.0})

    if not rows:
        return pd.Series(dtype=float)

    df = (pd.DataFrame(rows)
            .drop_duplicates("date")
            .set_index("date")
            .sort_index())

    if cache_dir:
        df.to_parquet(cache_path)

    logger.info(f"    {currency} DVOL: {len(df)} days")
    return df["dvol"]


# ── Options chain for surface features ───────────────────────────────────────

def fetch_options_chain_snapshot(asset: str) -> pd.DataFrame | None:
    """
    Fetch current live options chain from Deribit.
    Returns DataFrame with columns: expiry, strike, kind, iv, delta, mid_price.
    Used for term structure and skew extraction.
    """
    currency = asset.upper()
    try:
        result = _get("get_book_summary_by_currency",
                      {"currency": currency, "kind": "option"})
    except Exception as e:
        logger.warning(f"  Options chain fetch failed for {currency}: {e}")
        return None

    rows = []
    now  = datetime.now(timezone.utc)

    for opt in result:
        name = opt.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) < 4:
            continue

        try:
            expiry_str = parts[1]   # e.g. "27MAR26"
            strike     = float(parts[2])
            kind       = parts[3]   # "C" or "P"
            expiry_dt  = datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=timezone.utc)
            dte        = (expiry_dt - now).days
            iv         = opt.get("mark_iv", None)
            delta      = opt.get("greeks", {}).get("delta", None) if opt.get("greeks") else None
            mid        = opt.get("mid_price", None)

            if dte > 0 and iv is not None and iv > 0:
                rows.append({
                    "instrument": name,
                    "expiry":     expiry_dt,
                    "dte":        dte,
                    "strike":     strike,
                    "kind":       kind,
                    "iv":         float(iv) / 100.0,   # to decimal
                    "delta":      float(delta) if delta else None,
                    "mid_price":  float(mid) if mid else None,
                })
        except Exception:
            continue

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values(["expiry", "kind", "strike"])


def extract_surface_features(
    chain:     pd.DataFrame,
    dvol_hist: pd.Series,
    as_of:     datetime | None = None,
    asset:     str = "BTC",
) -> VolSurfaceSnapshot | None:
    """
    Extract vol surface features from options chain + DVOL history.

    Extracts:
        - Term structure (IV at 7d, 14d, 30d, 60d)
        - Skew (25-delta risk reversal)
        - Butterfly (25-delta)
        - Vol-of-vol from DVOL history
        - IV percentile rank
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    # ── Term structure from chain ──────────────────────────────────────────
    def atm_iv_near_dte(target_dte: int, tol: int = 4) -> float:
        """Find ATM IV for expiry near target_dte days."""
        near = chain[(chain["dte"] >= target_dte - tol) &
                     (chain["dte"] <= target_dte + tol)].copy()
        if near.empty:
            # Interpolate between nearest expiries
            shorter = chain[chain["dte"] < target_dte]
            longer  = chain[chain["dte"] > target_dte]
            if shorter.empty or longer.empty:
                return np.nan
            # Use the ATM options (delta near ±0.5)
            def best_atm(df):
                calls = df[df["kind"] == "C"].copy()
                if calls.empty or calls["delta"].isna().all():
                    return df["iv"].mean()
                calls["delta_dist"] = (calls["delta"] - 0.5).abs()
                return float(calls.nsmallest(1, "delta_dist")["iv"].iloc[0])
            iv_s = best_atm(shorter[shorter["dte"] == shorter["dte"].max()])
            iv_l = best_atm(longer[longer["dte"]  == longer["dte"].min()])
            dte_s = shorter["dte"].max()
            dte_l = longer["dte"].min()
            t = (target_dte - dte_s) / (dte_l - dte_s + 1e-6)
            return iv_s + t * (iv_l - iv_s)

        calls = near[near["kind"] == "C"].copy()
        if calls.empty or calls["delta"].isna().all():
            return float(near["iv"].mean())
        calls["delta_dist"] = (calls["delta"].fillna(0.5) - 0.5).abs()
        return float(calls.nsmallest(1, "delta_dist")["iv"].iloc[0])

    iv_7d  = atm_iv_near_dte(7)
    iv_14d = atm_iv_near_dte(14)
    iv_30d = float(dvol_hist.iloc[-1]) if not dvol_hist.empty else atm_iv_near_dte(30)
    iv_60d = atm_iv_near_dte(60)

    # ── Skew and butterfly from 25-delta options ──────────────────────────
    def skew_butterfly(target_dte: int, tol: int = 4):
        near = chain[(chain["dte"] >= target_dte - tol) &
                     (chain["dte"] <= target_dte + tol)]
        if near.empty or near["delta"].isna().all():
            return 0.0, 0.0

        calls = near[near["kind"] == "C"].copy()
        puts  = near[near["kind"] == "P"].copy()

        if calls.empty or puts.empty:
            return 0.0, 0.0

        # 25-delta call: delta ≈ 0.25
        calls["d25"] = (calls["delta"].fillna(0) - 0.25).abs()
        puts["d25"]  = (puts["delta"].fillna(0) + 0.25).abs()

        c25 = calls.nsmallest(1, "d25")
        p25 = puts.nsmallest(1, "d25")

        if c25.empty or p25.empty:
            return 0.0, 0.0

        iv_c25 = float(c25["iv"].iloc[0])
        iv_p25 = float(p25["iv"].iloc[0])
        atm    = atm_iv_near_dte(target_dte, tol)

        rr  = iv_p25 - iv_c25                      # risk reversal (skew)
        fly = (iv_p25 + iv_c25) / 2 - atm          # butterfly (convexity)

        return float(rr), float(fly)

    skew_7d,  fly_7d  = skew_butterfly(7)
    skew_30d, fly_30d = skew_butterfly(30)
    skew = (skew_7d + skew_30d) / 2
    fly  = (fly_7d  + fly_30d)  / 2

    # ── Vol-of-vol from DVOL history ──────────────────────────────────────
    dvol_30d = dvol_hist.iloc[-30:] if len(dvol_hist) >= 30 else dvol_hist
    dvol_chg = dvol_30d.diff().dropna()
    vov_30d  = float(dvol_chg.std()) if len(dvol_chg) > 2 else 0.05

    # ── IV percentile rank (vs last 252 days) ─────────────────────────────
    dvol_252 = dvol_hist.iloc[-252:] if len(dvol_hist) >= 252 else dvol_hist
    iv_rank  = float((dvol_252 < iv_30d).mean()) if len(dvol_252) > 10 else 0.5

    # ── IV trend (21-day change) ──────────────────────────────────────────
    if len(dvol_hist) >= 22:
        iv_21d_ago = float(dvol_hist.iloc[-22])
        iv_trend   = (iv_30d - iv_21d_ago) / (iv_21d_ago + 1e-6)
    else:
        iv_trend = 0.0

    # ── Term slope ────────────────────────────────────────────────────────
    if np.isfinite(iv_7d) and iv_7d > 0:
        term_slope = (iv_30d - iv_7d) / (iv_7d + 1e-6)
    else:
        term_slope = 0.0

    return VolSurfaceSnapshot(
        asset         = asset,
        timestamp     = as_of,
        iv_atm_7d     = float(np.nan_to_num(iv_7d,  nan=iv_30d)),
        iv_atm_14d    = float(np.nan_to_num(iv_14d, nan=iv_30d)),
        iv_atm_30d    = float(iv_30d),
        iv_atm_60d    = float(np.nan_to_num(iv_60d, nan=iv_30d)),
        term_slope    = float(np.clip(term_slope, -1, 1)),
        skew_25d      = float(np.clip(skew, -0.5, 0.5)),
        butterfly_25d = float(np.clip(fly,  -0.2, 0.2)),
        vov_30d       = float(np.clip(vov_30d, 0, 0.5)),
        iv_pct_rank   = float(np.clip(iv_rank, 0, 1)),
        iv_trend      = float(np.clip(iv_trend, -1, 1)),
    )


# ── Rolling historical surface features ──────────────────────────────────────

def build_historical_surface_features(
    asset:     str,
    dvol_hist: pd.Series,
) -> pd.DataFrame:
    """
    Build a daily time series of vol surface features from DVOL history alone.
    This is the fallback when we can't fetch live options chains historically.

    Returns DataFrame with columns matching VOL_SURFACE_FEATURE_NAMES (partial).
    The chain-derived features (skew, butterfly) are set to 0 since we don't
    have historical options data. The DVOL-derived features are accurate.
    """
    rows = []
    dvol = dvol_hist.copy()

    for i, (date, iv_30d) in enumerate(dvol.items()):
        if i < 30:
            continue  # need at least 30 days of history

        # 30-day vol-of-vol
        dvol_30d = dvol.iloc[max(0, i-30):i]
        dvol_chg = dvol_30d.diff().dropna()
        vov      = float(dvol_chg.std()) if len(dvol_chg) > 2 else 0.05

        # Percentile rank
        dvol_252 = dvol.iloc[max(0, i-252):i]
        iv_rank  = float((dvol_252 < iv_30d).mean()) if len(dvol_252) > 10 else 0.5

        # 21-day trend
        if i >= 22:
            iv_21ago = float(dvol.iloc[i - 22])
            trend    = (iv_30d - iv_21ago) / (iv_21ago + 1e-6)
        else:
            trend = 0.0

        # 7d IV: approximate from DVOL with typical term structure shape
        # In practice: shorter-tenor IV ≈ DVOL * (1 + backwardation_factor)
        # We approximate as DVOL slightly elevated for very short tenors
        iv_7d_approx = iv_30d * 1.0  # flat assumption without chain data
        term_slope   = 0.0            # unknown without chain

        rows.append({
            "date":           date,
            "iv_atm_7d":      float(iv_7d_approx),
            "iv_atm_30d":     float(iv_30d),
            "term_slope":     float(np.clip(term_slope, -1, 1)),
            "skew_25d":       0.0,   # requires live chain
            "butterfly_25d":  0.0,
            "vov_30d":        float(np.clip(vov, 0, 0.5)),
            "iv_pct_rank":    float(np.clip(iv_rank, 0, 1)),
            "iv_trend":       float(np.clip(trend, -1, 1)),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("date")

# ── Synthetic DVOL fallback ───────────────────────────────────────────────────

def build_synthetic_dvol(
    hourly_prices: "pd.Series",
    vrp_premium:   float = 0.08,   # typical VRP: IV ~8 vol pts above RV
    window:        int   = 30,     # rolling window in days
) -> "pd.Series":
    """
    Build a synthetic DVOL series from realized vol + VRP premium.

    Used when Deribit API is unavailable. Historical crypto VRP (IV over RV)
    is typically 5-15 vol points. Default 8 pts is conservative.

    Returns daily series of annualized IV (decimal fraction, e.g. 0.75).
    """
    import numpy as np
    import pandas as pd

    daily = hourly_prices.resample("D").last().dropna()
    log_rets = np.log(daily / daily.shift(1)).dropna()

    rv_rolling = log_rets.rolling(window).std() * np.sqrt(365)
    iv_synthetic = (rv_rolling + vrp_premium).dropna()

    # Clip to reasonable range
    iv_synthetic = iv_synthetic.clip(0.20, 3.00)
    return iv_synthetic


def fetch_dvol_history_with_fallback(
    asset:         str,
    start:         str,
    end:           str,
    hourly_prices: "pd.Series | None" = None,
    cache_dir:     "Path | None" = None,
    vrp_premium:   float = 0.08,
) -> "pd.Series":
    """
    Fetch DVOL from Deribit. Falls back to synthetic IV if unavailable.

    Args:
        asset:          "BTC" or "ETH"
        start, end:     Date range strings "YYYY-MM-DD"
        hourly_prices:  Hourly spot prices (used for synthetic fallback)
        cache_dir:      Cache directory
        vrp_premium:    VRP premium for synthetic fallback (8 vol pts default)

    Returns:
        Daily series of annualized IV (decimal fraction)
    """
    import numpy as np
    import pandas as pd

    # Check for pre-downloaded file first (from scripts/download_dvol.py)
    if cache_dir:
        live_path = Path(cache_dir) / f"dvol_{asset.upper()}_live.parquet"
        if live_path.exists():
            try:
                df = pd.read_parquet(live_path)
                dvol = df["dvol"] if "dvol" in df.columns else df.iloc[:, 0]
                start_ts = pd.Timestamp(start, tz="UTC")
                end_ts   = pd.Timestamp(end,   tz="UTC")
                dvol.index = pd.DatetimeIndex([
                    ts.tz_localize("UTC") if ts.tzinfo is None else ts
                    for ts in dvol.index
                ])
                filtered = dvol[(dvol.index >= start_ts) & (dvol.index <= end_ts)]
                if not filtered.empty:
                    print(f"    {asset} DVOL: loaded from pre-downloaded file ({len(filtered)} days)")
                    return filtered
            except Exception as e:
                print(f"    {asset} pre-downloaded DVOL load failed: {e}")

    # Try live API
    try:
        dvol = fetch_dvol_history(asset, start, end, cache_dir)
        if not dvol.empty:
            return dvol
        print(f"    {asset} DVOL API returned empty — using synthetic fallback")
    except Exception as e:
        print(f"    {asset} DVOL API failed ({e}) — using synthetic fallback")

    # Synthetic fallback
    if hourly_prices is None or hourly_prices.empty:
        print(f"    {asset}: no spot data for synthetic DVOL — skipping")
        return pd.Series(dtype=float)

    print(f"    {asset}: building synthetic DVOL from RV + {vrp_premium*100:.0f}bp VRP premium")
    dvol_synthetic = build_synthetic_dvol(hourly_prices, vrp_premium=vrp_premium)

    # Filter to requested date range
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")
    dvol_synthetic.index = pd.DatetimeIndex([
        ts.tz_localize("UTC") if ts.tzinfo is None else ts
        for ts in dvol_synthetic.index
    ])
    dvol_filtered = dvol_synthetic[
        (dvol_synthetic.index >= start_ts) & (dvol_synthetic.index <= end_ts)
    ]

    # Cache the synthetic series
    if cache_dir and not dvol_filtered.empty:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"dvol_synthetic_{asset.upper()}_{start}_{end}.parquet"
        dvol_filtered.rename("dvol").to_frame().to_parquet(cache_path)

    print(f"    {asset} synthetic DVOL: {len(dvol_filtered)} days")
    return dvol_filtered
