"""
engines/funding_spread_strategy.py
====================================
Cross-venue (Binance × Bybit) funding-rate spread carry as a CPO TradingStrategy.

Position structure (delta-neutral):
    direction = sign(binance_funding_8h - bybit_funding_8h)
    if direction > 0 (binance higher):  SHORT Binance perp + LONG Bybit perp
    if direction < 0 (bybit higher):    LONG Binance perp + SHORT Bybit perp
Per 8h funding event during hold, the position receives |spread| × notional.
TC: 4 legs total (entry pair + exit pair across both venues). Default
tc_bps_per_leg=4.0 (taker baseline = 16 bps RT). For maker scenario, pass
tc_bps_per_leg=1.75 (≈7 bps RT).

Hold simulation simplification (Cycle 50 D2c thin slice):
    Cross-venue perp basis is assumed to remain synced over the hold (zero
    cross-venue basis P&L). The return is therefore: sum of (signed) spread
    payments minus TC. This is an upper bound; if D2c results are positive,
    Cycle 51 can refine with a cross-venue perp price model. Documented in
    the Cycle 50 retro.

Feature set (10 features per Option B from the Cycle 50 D2a feature
discussion):
    1. binance_8h_pct
    2. bybit_8h_pct
    3. spread_ann
    4. spread_7d_avg
    5. spread_trend
    6. spread_pct_positive_30d   (fraction with same sign as current)
    7. spread_vol_30d
    8. binance_basis_pct
    9. spot_vol_24h_ann
    10. vol_regime

bybit_basis_pct intentionally omitted per Cycle 50 D2a feature
discussion: cross-venue spread features (3-7) carry the cross-venue
information directly; binance_basis is the higher-information of the
two basis features (deeper liquidity, more meaningful price discovery);
30-min Bybit perp fetcher avoidance for thin-slice spirit. Re-add
candidate for Cycle 51 if D2c maker P>0.70 Sharpe lands in the +1.0
to +3.0 marginal zone.

Funding rates are loaded from data/crypto_data.db (populated by Cycle 50
D1c Bybit backfill + the live PraxisFundingCollector). Spot + Binance
perp bars are CCXT-fetched (same path as funding_rate_strategy.py Exp 13).

Config grid: 4 thresholds × 4 hold durations × 3 pct_positive = 48
configs/asset × 6 assets = 288 total.

Train: 2024-01-01 .. 2024-12-31. OOS: 2025-01-01 .. 2026-03-26.
(Matches Exp 13 windows for direct comparability.)
"""
from __future__ import annotations

import itertools
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "crypto_data.db"

DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]

SPREAD_FEATURES = [
    "binance_8h_pct",
    "bybit_8h_pct",
    "spread_ann",
    "spread_7d_avg",
    "spread_trend",
    "spread_pct_positive_30d",
    "spread_vol_30d",
    "binance_basis_pct",
    "spot_vol_24h_ann",
    "vol_regime",
]


# ---------------------------------------------------------------------------
# Config + model spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FundingSpreadModelSpec:
    model_id: str
    asset: str


@dataclass
class FundingSpreadConfig:
    config_id: str
    min_spread_ann_pct: float    # |spread| > this (annualized %) to enter
    hold_days: int                # 3, 7, 14, 30
    min_pct_positive: float       # fraction of last 30 spreads with same sign as current

    @staticmethod
    def param_names() -> list[str]:
        return [
            "min_spread_norm",   # / 20.0 (max grid val)
            "hold_days_norm",    # / 30.0 (max grid val)
            "min_pct_pos_norm",  # already [0, 1]
        ]

    def to_feature_vector(self) -> list[float]:
        return [
            min(self.min_spread_ann_pct / 20.0, 1.0),
            self.hold_days / 30.0,
            self.min_pct_positive,
        ]


def generate_funding_spread_param_grid() -> list[FundingSpreadConfig]:
    """4 thresholds × 4 hold durations × 3 pct_positive = 48 configs/asset."""
    configs: list[FundingSpreadConfig] = []
    for i, (ann, hold, pct_pos) in enumerate(itertools.product(
        [3.0, 5.0, 8.0, 12.0],     # min annualized spread threshold (%)
        [3, 7, 14, 30],              # hold days
        [0.5, 0.65, 0.8],            # min fraction-with-same-sign
    )):
        configs.append(FundingSpreadConfig(
            config_id=f"fs_{i:04d}",
            min_spread_ann_pct=ann,
            hold_days=hold,
            min_pct_positive=pct_pos,
        ))
    return configs


# ---------------------------------------------------------------------------
# Funding-rate loading from DB
# ---------------------------------------------------------------------------

def _load_funding_from_db(asset: str, venue: str,
                          start_iso: str, end_iso: str,
                          db_path: Path = DB_PATH) -> pd.Series:
    """Pull (asset, venue) funding rates from the funding_rates table over
    [start_iso, end_iso). Returns a chronologically-sorted pandas Series
    indexed by UTC timestamp."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp, funding_rate FROM funding_rates "
            "WHERE asset = ? AND venue = ? "
            "AND datetime >= ? AND datetime < ? "
            "ORDER BY timestamp",
            (asset, venue, start_iso, end_iso),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["timestamp", "rate"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return (df.drop_duplicates("timestamp")
              .set_index("timestamp")["rate"]
              .sort_index())


# ---------------------------------------------------------------------------
# Hold simulation
# ---------------------------------------------------------------------------

def run_funding_spread_hold(
    binance_funding: pd.Series,
    bybit_funding: pd.Series,
    hold_start: pd.Timestamp,
    hold_days: int,
    tc_bps_per_leg: float = 4.0,
    n_legs: int = 4,
) -> dict:
    """Simulate one N-day cross-venue spread carry hold.

    Direction is set at entry based on sign(binance - bybit). During hold
    the position receives direction × (binance_rate - bybit_rate) per 8h
    funding event. Cross-venue perp basis P&L is assumed zero (Cycle 50
    thin-slice).

    tc_bps_per_leg defaults to 4 bps (taker). For maker scenario pass
    tc_bps_per_leg=1.75 (≈7 bps RT). n_legs=4 because both venues have a
    long+short combo (2 legs per side × 2 entry/exit = ... wait, 1 fill
    per leg × 2 sides × 2 venues = 4 fills round trip).
    """
    hold_end = hold_start + timedelta(days=hold_days)
    tc_pct = tc_bps_per_leg * n_legs / 10000.0

    # Direction at entry: based on the most-recent funding event before hold_start
    b_pre = binance_funding[binance_funding.index <= hold_start]
    y_pre = bybit_funding[bybit_funding.index <= hold_start]
    if b_pre.empty or y_pre.empty:
        return {"net_return": 0.0, "gross_return": 0.0,
                "n_payments": 0, "profitable": False, "direction": 0}
    spread_now = float(b_pre.iloc[-1]) - float(y_pre.iloc[-1])
    direction = 1 if spread_now > 0 else -1   # 1 = short binance/long bybit

    # Funding events INSIDE the hold window
    b_in = binance_funding[(binance_funding.index > hold_start) &
                            (binance_funding.index <= hold_end)]
    y_in = bybit_funding[(bybit_funding.index > hold_start) &
                          (bybit_funding.index <= hold_end)]
    common = b_in.index.intersection(y_in.index)
    if len(common) == 0:
        return {"net_return": 0.0, "gross_return": 0.0,
                "n_payments": 0, "profitable": False, "direction": direction}
    spread_per_event = b_in.loc[common] - y_in.loc[common]
    total_spread_signed = float((direction * spread_per_event).sum())

    gross = float(np.clip(total_spread_signed, -0.99, 5.0))
    net = float(np.clip(gross - tc_pct, -0.99, 5.0))
    return {
        "net_return": net,
        "gross_return": gross,
        "n_payments": int(len(common)),
        "profitable": net > 0,
        "direction": direction,
    }


def run_funding_spread_single_day(
    binance_funding: pd.Series,
    bybit_funding: pd.Series,
    config: FundingSpreadConfig,
    tc_bps_per_leg: float = 4.0,
    as_of: pd.Timestamp | None = None,
) -> dict:
    """For one evaluation day (as_of = day_start):
    - Check entry conditions on |spread|
    - If conditions met, simulate hold_days carry
    Returns {"daily_return", "gross_return", "n_trades"}."""
    if as_of is None:
        if binance_funding.empty:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}
        as_of = binance_funding.index[0].normalize()
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")

    # Compute spread history for entry-condition checks
    b_hist = binance_funding[binance_funding.index < as_of].iloc[-90:]
    y_hist = bybit_funding[bybit_funding.index < as_of].iloc[-90:]
    if b_hist.empty or y_hist.empty:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    common = b_hist.index.intersection(y_hist.index)
    if len(common) < 3:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}
    spread_hist = b_hist.loc[common] - y_hist.loc[common]

    spread_now = float(spread_hist.iloc[-1])
    spread_now_ann_pct = spread_now * 3 * 365 * 100

    # Condition 1: |spread| above threshold
    if abs(spread_now_ann_pct) < config.min_spread_ann_pct:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    # Condition 2: sufficient sign-consistency
    sign_recent = np.sign(spread_now) if spread_now != 0 else 1
    pct_consistent = float((np.sign(spread_hist) == sign_recent).mean())
    if pct_consistent < config.min_pct_positive:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    result = run_funding_spread_hold(
        binance_funding, bybit_funding,
        hold_start=as_of,
        hold_days=config.hold_days,
        tc_bps_per_leg=tc_bps_per_leg,
    )
    if not np.isfinite(result["net_return"]):
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}
    return {
        "daily_return": float(result["net_return"]),
        "gross_return": float(result["gross_return"]),
        "n_trades": 1,
    }


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_spread_features(
    spot_bars: pd.DataFrame,
    binance_perp: pd.DataFrame | None,
    binance_funding: pd.Series,
    bybit_funding: pd.Series,
    as_of_date,
) -> np.ndarray | None:
    """Compute the 10 cross-venue spread features as of `as_of_date`.

    Returns None if any required input is insufficient (e.g. <3 funding
    events). Uses Binance spot as the basis-reference; binance_perp is
    optional (basis feature falls back to 0 if unavailable).
    """
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    hist_start = as_of - timedelta(days=35)

    # Funding history per venue (history-only; not the as_of moment itself)
    b_fund = binance_funding[(binance_funding.index >= hist_start) &
                              (binance_funding.index < as_of)]
    y_fund = bybit_funding[(bybit_funding.index >= hist_start) &
                            (bybit_funding.index < as_of)]
    if len(b_fund) < 3 or len(y_fund) < 3:
        return None

    # Per-venue current rates (8h scale, converted to % for the feature)
    b_fr_8h = float(b_fund.iloc[-1])
    y_fr_8h = float(y_fund.iloc[-1])

    # Align by timestamp for spread computation
    common_idx = b_fund.index.intersection(y_fund.index)
    if len(common_idx) < 3:
        return None
    b_common = b_fund.loc[common_idx]
    y_common = y_fund.loc[common_idx]
    spread_series = b_common - y_common   # binance - bybit, 8h-rate units

    # Annualization factor: 3 events/day × 365 days × 100 (to %)
    ANN = 3 * 365 * 100

    spread_now = float(spread_series.iloc[-1])
    spread_now_ann = spread_now * ANN

    seven_days_ago = as_of - timedelta(days=7)
    spread_7d = spread_series[spread_series.index >= seven_days_ago]
    spread_7d_ann = (float(spread_7d.mean()) * ANN
                     if len(spread_7d) > 0 else 0.0)

    spread_trend = (
        (spread_now_ann - spread_7d_ann) / (abs(spread_7d_ann) + 1e-6)
    )

    # Last 30 days (≈90 events at 8h cadence)
    spread_30 = spread_series.iloc[-90:]
    if len(spread_30) > 0:
        sign_recent = np.sign(spread_now) if spread_now != 0 else 1
        pct_pos = float((np.sign(spread_30) == sign_recent).mean())
    else:
        pct_pos = 0.5
    spread_vol = (float(spread_30.std()) * ANN
                  if len(spread_30) > 2 else 0.0)

    # Binance perp basis vs Binance spot (single basis feature)
    spot_hist = spot_bars[(spot_bars.index >= hist_start) &
                           (spot_bars.index < as_of)]
    binance_basis = 0.0
    if binance_perp is not None and len(spot_hist) >= 24:
        perp_hist = binance_perp[(binance_perp.index >= hist_start) &
                                  (binance_perp.index < as_of)]
        if not perp_hist.empty:
            merged = pd.concat([
                spot_hist["close"].rename("spot"),
                perp_hist["close"].rename("perp"),
            ], axis=1).dropna()
            if not merged.empty:
                basis_series = ((merged["perp"] - merged["spot"]) /
                                 merged["spot"])
                binance_basis = float(basis_series.iloc[-1]) * 100

    # Spot-side vol regime
    if len(spot_hist) >= 24:
        rets_24h = spot_hist["close"].pct_change().dropna().iloc[-24:]
        rets_30d = spot_hist["close"].pct_change().dropna()
        vol_24h = (float(rets_24h.std() * np.sqrt(24 * 365))
                    if len(rets_24h) > 2 else 0.01)
        vol_30d = (float(rets_30d.std() * np.sqrt(24 * 365))
                    if len(rets_30d) > 2 else 0.01)
        vol_regime = vol_24h / (vol_30d + 1e-10)
    else:
        vol_24h = 0.01
        vol_regime = 1.0

    features = np.array([
        np.clip(b_fr_8h * 100, -0.5, 0.5),       # binance_8h_pct
        np.clip(y_fr_8h * 100, -0.5, 0.5),       # bybit_8h_pct
        np.clip(spread_now_ann, -100, 100),      # spread_ann (%)
        np.clip(spread_7d_ann, -100, 100),       # spread_7d_avg
        np.clip(spread_trend, -5, 5),             # spread_trend
        pct_pos,                                   # spread_pct_positive_30d
        np.clip(spread_vol, 0, 200),              # spread_vol_30d
        np.clip(binance_basis, -2, 2),            # binance_basis_pct
        np.clip(vol_24h, 0, 5),                   # spot_vol_24h_ann
        np.clip(vol_regime, 0, 5),                # vol_regime
    ], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)


# ---------------------------------------------------------------------------
# Data fetching (CCXT) -- reuses funding_rate_strategy's helpers patterns
# ---------------------------------------------------------------------------

def _fetch_binance_spot(assets, start, end, quote="USDT", cache_dir=None):
    """Hourly spot bars from Binance via CCXT. Cached as parquet."""
    import ccxt
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.binance({"enableRateLimit": True})
    results = {}
    for asset in assets:
        symbol = f"{asset}/{quote}"
        cp = (cache_dir / f"fs_spot_{asset}_{start}_{end}.parquet"
              if cache_dir else None)
        if cp and cp.exists():
            try:
                results[asset] = pd.read_parquet(cp)
                continue
            except Exception:
                pass
        try:
            s_ms = int(datetime.strptime(start, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
            e_ms = int(datetime.strptime(end, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
            all_bars, cur = [], s_ms
            while cur < e_ms:
                bars = exchange.fetch_ohlcv(symbol, "1h", since=cur, limit=1000)
                if not bars:
                    break
                bars = [b for b in bars if b[0] < e_ms]
                if not bars:
                    break
                all_bars.extend(bars)
                last = bars[-1][0]
                if last <= cur:
                    break
                cur = last + 1
            if all_bars:
                df = pd.DataFrame(all_bars,
                                  columns=["timestamp", "open", "high",
                                           "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"],
                                                  unit="ms", utc=True)
                df = (df.drop_duplicates("timestamp")
                        .set_index("timestamp").sort_index())
                df = df[df.index < pd.Timestamp(end, tz="UTC")]
                if cp:
                    df.to_parquet(cp)
                results[asset] = df
                logger.info(f"  {asset} spot: {len(df)} bars")
        except Exception as e:
            logger.warning(f"  Spot failed {asset}: {e}")
    return results


def _fetch_binance_perp_only(assets, start, end, quote="USDT", cache_dir=None):
    """Hourly Binance perp bars only (funding comes from DB in this strategy)."""
    import ccxt
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    results = {}
    for asset in assets:
        symbol = f"{asset}/{quote}:{quote}"
        cp = (cache_dir / f"fs_perp_{asset}_{start}_{end}.parquet"
              if cache_dir else None)
        if cp and cp.exists():
            try:
                results[asset] = pd.read_parquet(cp)
                continue
            except Exception:
                pass
        try:
            s_ms = int(datetime.strptime(start, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
            e_ms = int(datetime.strptime(end, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
            all_bars, cur = [], s_ms
            while cur < e_ms:
                bars = exchange.fetch_ohlcv(symbol, "1h", since=cur, limit=1000)
                if not bars:
                    break
                bars = [b for b in bars if b[0] < e_ms]
                if not bars:
                    break
                all_bars.extend(bars)
                last = bars[-1][0]
                if last <= cur:
                    break
                cur = last + 1
            if all_bars:
                df = pd.DataFrame(all_bars,
                                  columns=["timestamp", "open", "high",
                                           "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"],
                                                  unit="ms", utc=True)
                df = (df.drop_duplicates("timestamp")
                        .set_index("timestamp").sort_index())
                df = df[df.index < pd.Timestamp(end, tz="UTC")]
                if cp:
                    df.to_parquet(cp)
                results[asset] = df
                logger.info(f"  {asset} perp: {len(df)} bars")
        except Exception as e:
            logger.warning(f"  Perp failed {asset}: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class FundingSpreadStrategy:
    """Cycle 50 D2a: cross-venue funding spread carry. CPO TradingStrategy
    interface mirrors funding_rate_strategy.FundingRateStrategy so the
    existing run_cpo.py framework can drive phase2/3/4 without changes."""

    def __init__(self, assets=None, cache_dir="data/funding_cache",
                 tc_bps=4.0, training_start="2024-01-01",
                 training_end="2024-12-31", quote="USDT"):
        # tc_bps here is the per-leg one-way TC in basis points. The hold
        # function multiplies by n_legs=4 internally (2 entry fills + 2 exit
        # fills across two venues). Taker baseline: tc_bps=4.0 (16 bps RT).
        # Maker baseline: tc_bps=1.75 (7 bps RT).
        self.assets         = assets or DEFAULT_ASSETS
        self.cache_dir      = Path(cache_dir)
        self.tc_bps         = tc_bps
        self.training_start = training_start
        self.training_end   = training_end
        self.quote          = quote

        self._models  = [FundingSpreadModelSpec(f"{a}_SPREAD", a)
                          for a in self.assets]
        self._configs = generate_funding_spread_param_grid()

        n_feat = len(self.daily_feature_names())
        print(f"  Funding Spread CPO: {len(self._models)} models "
              f"({len(self.assets)} assets)")
        print(f"  Feature mode: spread ({n_feat} features)")
        print(f"  Total configs: {len(self._configs)} (48 per model)")
        print(f"  TC: {tc_bps} bps/leg × 4 legs = {tc_bps * 4:.1f} bps RT")
        print(f"  Hold grid: 3/7/14/30 days; thresholds 3/5/8/12% ann; "
              f"pct_pos 0.5/0.65/0.8")

    # ---- CPO interface ----

    def get_models(self):     return self._models
    def get_param_grid(self): return self._configs

    def daily_feature_names(self) -> list[str]:
        return list(SPREAD_FEATURES)

    def config_param_names(self) -> list[str]:
        return FundingSpreadConfig.param_names()

    def config_to_features(self, config):
        return config.to_feature_vector()

    def compute_features(self, model, as_of_date, data):
        spot = data.get("spot", {}).get(model.asset)
        perp_data = data.get("perp", {}).get(model.asset)
        if spot is None or perp_data is None:
            return None
        return _compute_spread_features(
            spot,
            perp_data.get("perp"),
            perp_data["binance_funding"],
            perp_data["bybit_funding"],
            as_of_date,
        )

    def run_single_day(self, model, config, day, data):
        perp_data = data.get("perp", {}).get(model.asset)
        if perp_data is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}
        day_start = pd.Timestamp(day)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("UTC")
        # Slice funding series to up-to-hold-window only
        hold_end = day_start + timedelta(days=config.hold_days + 1)
        b_fund = perp_data["binance_funding"]
        y_fund = perp_data["bybit_funding"]
        b_slice = b_fund[b_fund.index <= hold_end]
        y_slice = y_fund[y_fund.index <= hold_end]
        return run_funding_spread_single_day(
            b_slice, y_slice, config,
            tc_bps_per_leg=self.tc_bps, as_of=day_start,
        )

    def run_model_year(self, model, data, param_grid):
        spot = data.get("spot", {}).get(model.asset)
        perp_data = data.get("perp", {}).get(model.asset)
        if spot is None or perp_data is None:
            return pd.DataFrame()
        b_fund = perp_data["binance_funding"]
        y_fund = perp_data["bybit_funding"]
        if b_fund.empty or y_fund.empty:
            return pd.DataFrame()

        trading_days = sorted(spot.index.normalize().unique())
        results = []
        for day_idx, day in enumerate(trading_days):
            day_start = pd.Timestamp(day)
            if day_start.tzinfo is None:
                day_start = day_start.tz_localize("UTC")
            hold_end = day_start + timedelta(days=31)  # cover the largest hold
            b_slice = b_fund[b_fund.index <= hold_end]
            y_slice = y_fund[y_fund.index <= hold_end]
            if len(b_slice) < 3 or len(y_slice) < 3:
                continue
            for config in param_grid:
                result = run_funding_spread_single_day(
                    b_slice, y_slice, config,
                    tc_bps_per_leg=self.tc_bps, as_of=day_start,
                )
                dr = result["daily_return"]
                gr = result["gross_return"]
                if not (np.isfinite(dr) and np.isfinite(gr)):
                    continue
                results.append({
                    "model_id":     model.model_id,
                    "date":         day.strftime("%Y-%m-%d"),
                    "config_id":    config.config_id,
                    "daily_return": float(dr),
                    "gross_return": float(gr),
                    "n_trades":     result["n_trades"],
                })
            if (day_idx + 1) % 30 == 0:
                print(f"    {model.model_id}: {day_idx + 1}/{len(trading_days)} days")
        return pd.DataFrame(results)

    # ---- Data fetching ----

    def _build_data_dict(self, assets, start, end):
        """Build the standard data dict for either training or OOS."""
        spot = _fetch_binance_spot(assets, start, end, self.quote, self.cache_dir)
        perp_bars = _fetch_binance_perp_only(assets, start, end, self.quote, self.cache_dir)
        perp = {}
        for a in assets:
            if a not in perp_bars and a not in spot:
                continue
            b_fund = _load_funding_from_db(a, "binance", start, end)
            y_fund = _load_funding_from_db(a, "bybit",   start, end)
            perp[a] = {
                "perp":            perp_bars.get(a),
                "binance_funding": b_fund,
                "bybit_funding":   y_fund,
            }
            logger.info(f"  {a} funding: binance={len(b_fund)} bybit={len(y_fund)}")
        return {"spot": spot, "perp": perp}

    def fetch_training_data(self, models, start, end):
        assets = list({m.asset for m in models})
        return self._build_data_dict(assets, self.training_start, self.training_end)

    def fetch_oos_data(self, models, start, end):
        assets = list({m.asset for m in models})
        return self._build_data_dict(assets, start, end)

    def fetch_warmup_daily(self, models, start, end):
        assets = list({m.asset for m in models})
        spot = _fetch_binance_spot(assets, start, end, self.quote, self.cache_dir)
        return {a: b["close"].resample("D").last().dropna() for a, b in spot.items()}

    def get_daily_prices(self, oos_data, models):
        spot = oos_data.get("spot", {})
        return {a: b["close"].resample("D").last().dropna() for a, b in spot.items()}

    def get_trading_days(self, data):
        spot = data.get("spot", {})
        if not spot:
            return []
        all_days: set = set()
        for bars in spot.values():
            all_days.update(bars.index.normalize().unique())
        return sorted(all_days)

    def prepare_warmup(self, daily_prices, warmup_daily):
        result = {}
        for asset in daily_prices:
            warm = warmup_daily.get(asset, pd.Series(dtype=float))
            curr = daily_prices[asset]
            if not warm.empty:
                combined = pd.concat([warm, curr])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                result[asset] = combined
            else:
                result[asset] = curr
        return result
