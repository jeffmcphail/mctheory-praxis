"""
engines/funding_rate_strategy.py
==================================
Funding rate carry trade as a CPO TradingStrategy.

CORRECT EVALUATION UNIT: N-day hold (not 8h).
- Enter once at start of evaluation window
- Hold for hold_days (3, 7, 14 days)
- Collect ALL funding payments during hold
- Pay TC once at entry + once at exit
- Label = +1 if total_funding > basis_drift + amortized_TC

Why N-day, not 8h:
    TC = 8bps round-trip (spot + perp, entry + exit)
    Daily funding at 10% annual = 2.74bps/day
    TC amortized over 1 day = 8bps → losing trade 100% of the time
    TC amortized over 3 days = 2.67bps ≈ break-even
    TC amortized over 7 days = 1.14bps → profitable if funding > 4% annual
    TC amortized over 14 days = 0.57bps → profitable if funding > 2% annual

CPO fit:
    - Base rate ~40-60% (carry profitable when funding > TC + basis drift)
    - RF learns: which funding environments produce sustainable carry over N days
    - Features: rate level, trend, basis stability, vol regime
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import timedelta, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX"]

FUNDING_FEATURES = [
    "funding_8h_pct",         # latest 8h rate (%)
    "funding_ann_pct",        # annualized equivalent (%)
    "funding_7d_avg_ann",     # 7-day avg annualized
    "funding_trend",          # (current - 7d_avg) / |7d_avg|
    "funding_pct_positive",   # fraction of last 30 payments positive
    "funding_volatility",     # std of last 30 payments (annualized)
    "basis_pct",              # (perp - spot) / spot × 100
    "basis_7d_avg",           # 7-day avg basis %
    "basis_trend_24h",        # basis change over last 24h
    "spot_vol_24h_ann",       # 24h realized vol (annualized)
    "vol_regime",             # today vol / 30d avg vol
]


# Regime feature names — fixed order, computed from RegimeEngine.
# 12 state integers + ~34 raw sub-features = 46 total.
# Features with NaN (e.g. L_dvol when no DVOL data) are zero-filled.
REGIME_FEATURE_NAMES = [
    # 12 regime state integers
    "regime_trend", "regime_vol_level", "regime_vol_trend",
    "regime_serial_corr", "regime_microstructure",
    "regime_funding_positioning", "regime_liquidity",
    "regime_cross_asset_corr", "regime_volume_participation",
    "regime_term_structure", "regime_dispersion", "regime_rv_iv_spread",
    # Raw sub-features (A-L)
    "A_adx_14", "A_plus_di_14", "A_minus_di_14",
    "A_ret_1d", "A_ret_7d", "A_ret_30d",
    "BC_rv_1d", "BC_rv_7d", "BC_rv_30d", "BC_vol_ratio", "BC_vol_pct_rank",
    "D_hurst", "D_vr_4", "D_vr_24", "D_serial_score",
    "E_ofi_bar_last", "E_ofi_24h", "E_amihud",
    "F_fr_8h_pct", "F_fr_ann", "F_oi_change_7d",
    "G_cs_spread", "G_amihud_ratio", "G_vol_z", "G_liq_score",
    "I_vol_ratio_24h", "I_ret_24h", "I_conviction",
    "J_fr_7d_ann", "J_fr_30d_ann", "J_fr_slope",
    "L_rv_current", "L_dvol", "L_vrp",
]


@dataclass
class FundingModelSpec:
    model_id: str
    asset: str


@dataclass
class FundingConfig:
    config_id: str
    min_funding_ann_pct: float   # minimum annualized rate to enter (e.g. 5 = 5%/yr)
    hold_days: int               # calendar days to hold position (3, 7, 14)
    min_pct_positive: float      # min fraction of recent payments that were positive

    @staticmethod
    def param_names() -> list[str]:
        return [
            "min_funding_norm",    # / 50
            "hold_days_norm",      # / 14
            "min_pct_pos_norm",    # already [0,1]
        ]

    def to_feature_vector(self) -> list[float]:
        return [
            min(self.min_funding_ann_pct / 50.0, 1.0),
            self.hold_days / 14.0,
            self.min_pct_positive,
        ]


def generate_funding_param_grid() -> list[FundingConfig]:
    """
    4 thresholds × 3 hold durations × 3 pct_positive filters = 36 configs.

    Designed so that even at 5% annualized and 14-day hold,
    the carry meaningfully exceeds TC.
    """
    configs = []
    for i, (ann, hold, pct_pos) in enumerate(itertools.product(
        [5.0, 10.0, 20.0, 40.0],    # annualized rate threshold
        [3, 7, 14],                   # hold days
        [0.5, 0.65, 0.8],             # min fraction positive
    )):
        configs.append(FundingConfig(
            config_id=f"fr_{i:04d}",
            min_funding_ann_pct=ann,
            hold_days=hold,
            min_pct_positive=pct_pos,
        ))
    return configs


def run_funding_hold(
    spot_bars:     pd.DataFrame,   # hourly spot bars (full hold window)
    perp_bars:     pd.DataFrame,   # hourly perp bars (full hold window)
    funding_rates: pd.Series,      # 8h funding rates (full hold window)
    hold_start:    pd.Timestamp,
    hold_days:     int,
    tc_bps:        float = 4.0,
) -> dict:
    """
    Simulate one N-day funding carry hold.
    Enter at hold_start, exit at hold_start + hold_days.
    Collect all funding payments between entry and exit.
    Pay TC once at entry + once at exit.

    Returns {"net_return", "gross_return", "n_payments", "profitable"}
    """
    hold_end = hold_start + timedelta(days=hold_days)
    tc_pct   = tc_bps / 10000.0   # one-way TC

    # Entry prices at hold_start
    spot_at_entry = spot_bars[spot_bars.index <= hold_start]["close"]
    perp_at_entry = perp_bars[perp_bars.index <= hold_start]["close"]
    if spot_at_entry.empty or perp_at_entry.empty:
        return {"net_return": 0.0, "gross_return": 0.0, "n_payments": 0, "profitable": False}

    spot_entry = float(spot_at_entry.iloc[-1])
    perp_entry = float(perp_at_entry.iloc[-1])
    if spot_entry <= 0 or perp_entry <= 0:
        return {"net_return": 0.0, "gross_return": 0.0, "n_payments": 0, "profitable": False}

    # Exit prices at hold_end
    spot_at_exit = spot_bars[spot_bars.index <= hold_end]["close"]
    perp_at_exit = perp_bars[perp_bars.index <= hold_end]["close"]
    if spot_at_exit.empty or perp_at_exit.empty:
        # Use last available
        spot_at_exit = spot_bars["close"]
        perp_at_exit = perp_bars["close"]

    spot_exit = float(spot_at_exit.iloc[-1])
    perp_exit = float(perp_at_exit.iloc[-1])

    # All funding payments during the hold (exclusive of entry, inclusive of exit)
    hold_funding = funding_rates[
        (funding_rates.index > hold_start) &
        (funding_rates.index <= hold_end)
    ]
    total_funding = float(hold_funding.sum()) if not hold_funding.empty else 0.0
    n_payments    = len(hold_funding)

    # P&L: long spot + short perp (delta-neutral)
    spot_ret  = (spot_exit - spot_entry) / spot_entry    # long
    perp_ret  = (perp_entry - perp_exit) / perp_entry   # short

    # gross = basis_change (≈0 for perfect hedge) + funding collected
    gross  = float(np.clip(spot_ret + perp_ret + total_funding, -0.99, 5.0))
    # net = gross - TC once at entry (tc_pct) - TC once at exit (tc_pct)
    net    = float(np.clip(gross - tc_pct * 2, -0.99, 5.0))

    return {
        "net_return":   net,
        "gross_return": gross,
        "n_payments":   n_payments,
        "profitable":   net > 0,
    }


def run_funding_single_day(
    spot_all:      pd.DataFrame,   # all spot bars (history + hold window)
    perp_all:      pd.DataFrame,   # all perp bars (history + hold window)
    funding_all:   pd.Series,      # all funding rates (history + hold window)
    config:        FundingConfig,
    tc_bps:        float = 4.0,
    as_of:         pd.Timestamp | None = None,
) -> dict:
    """
    For one evaluation day (as_of = day_start):
        - Check entry conditions using history up to as_of
        - If conditions met, simulate hold_days carry from as_of

    Returns {"daily_return", "gross_return", "n_trades"}
    Note: daily_return here is the return for the full hold period,
    attributed to the entry day. CPO labels it for the entry day.
    """
    if as_of is None:
        if spot_all.empty:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}
        as_of = spot_all.index[0].normalize()
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")

    # ── Entry conditions using history before as_of ──────────────
    fr_hist = funding_all[funding_all.index < as_of].iloc[-90:]  # last 30 days
    if fr_hist.empty:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    current_fr = float(fr_hist.iloc[-1])
    ann_rate   = current_fr * 3 * 365 * 100

    # Condition 1: rate above threshold
    if ann_rate < config.min_funding_ann_pct:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    # Condition 2: sufficient fraction of recent payments positive
    pct_pos = float((fr_hist > 0).mean())
    if pct_pos < config.min_pct_positive:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    # ── Simulate hold ─────────────────────────────────────────────
    result = run_funding_hold(
        spot_all, perp_all, funding_all,
        hold_start=as_of,
        hold_days=config.hold_days,
        tc_bps=tc_bps,
    )

    if not np.isfinite(result["net_return"]):
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    return {
        "daily_return": float(result["net_return"]),
        "gross_return": float(result["gross_return"]),
        "n_trades":     1,
    }


def _compute_funding_features(spot_bars, perp_bars, funding_rates, as_of_date):
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    hist_start = as_of - timedelta(days=35)
    fr_hist = funding_rates[
        (funding_rates.index >= hist_start) & (funding_rates.index < as_of)
    ]
    if len(fr_hist) < 3:
        return None

    current_fr  = float(fr_hist.iloc[-1])
    ann_rate    = current_fr * 3 * 365 * 100
    fr_7d       = fr_hist[fr_hist.index >= (as_of - timedelta(days=7))]
    avg_7d_ann  = float(fr_7d.mean()) * 3 * 365 * 100 if len(fr_7d) > 0 else 0.0
    trend       = (ann_rate - avg_7d_ann) / (abs(avg_7d_ann) + 1e-6)
    fr_30       = fr_hist.iloc[-90:]
    pct_pos     = float((fr_30 > 0).mean())
    fr_vol      = float(fr_30.std() * 3 * 365 * 100) if len(fr_30) > 2 else 0.0

    spot_hist = spot_bars[(spot_bars.index >= hist_start) & (spot_bars.index < as_of)]
    perp_hist = perp_bars[(perp_bars.index >= hist_start) & (perp_bars.index < as_of)]
    if len(spot_hist) < 24 or len(perp_hist) < 24:
        return None

    merged = pd.concat([
        spot_hist["close"].rename("spot"),
        perp_hist["close"].rename("perp"),
    ], axis=1).dropna()
    if len(merged) < 24:
        return None

    basis      = (merged["perp"] - merged["spot"]) / merged["spot"]
    basis_now  = float(basis.iloc[-1]) * 100
    basis_7d   = float(basis.mean()) * 100
    basis_24h  = float(basis.iloc[-1] - basis.iloc[-24]) * 100 if len(basis) >= 24 else 0.0

    rets_24h   = spot_hist["close"].pct_change().dropna().iloc[-24:]
    rets_30d   = spot_hist["close"].pct_change().dropna()
    vol_24h    = float(rets_24h.std() * np.sqrt(24 * 365)) if len(rets_24h) > 2 else 0.01
    vol_30d    = float(rets_30d.std() * np.sqrt(24 * 365)) if len(rets_30d) > 2 else 0.01
    vol_regime = vol_24h / (vol_30d + 1e-10)

    features = np.array([
        np.clip(current_fr * 100, -0.5, 0.5),
        np.clip(ann_rate, -200, 200),
        np.clip(avg_7d_ann, -200, 200),
        np.clip(trend, -5, 5),
        pct_pos,
        np.clip(fr_vol, 0, 200),
        np.clip(basis_now, -2, 2),
        np.clip(basis_7d, -2, 2),
        np.clip(basis_24h, -1, 1),
        np.clip(vol_24h, 0, 5),
        np.clip(vol_regime, 0, 5),
    ], dtype=np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)


def _compute_regime_features(regime_engine, spot_bars, funding_rates,
                             as_of_date, universe_spot=None):
    """
    Compute regime features for one asset at one point in time.

    Args:
        regime_engine: RegimeEngine instance (bars_per_day=24 for crypto)
        spot_bars: Hourly OHLCV with DatetimeIndex (UTC)
        funding_rates: Funding rate Series with DatetimeIndex (UTC)
        as_of_date: Compute using data up to this date (no lookahead)
        universe_spot: Optional dict of {asset: DataFrame} for cross-asset
                       regime classes H (correlation) and K (dispersion)

    Returns:
        np.ndarray of len(REGIME_FEATURE_NAMES) or None if insufficient data
    """
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    # Need ~35 days of hourly bars for regime computation
    hist_start = as_of - timedelta(days=40)
    ohlcv = spot_bars[(spot_bars.index >= hist_start) & (spot_bars.index < as_of)]
    if len(ohlcv) < 24 * 20:  # need at least 20 days
        return None

    # Slice funding rates up to as_of
    fr = funding_rates[(funding_rates.index >= hist_start) & (funding_rates.index < as_of)]
    fr_arr = fr.values if len(fr) > 0 else None

    # Build universe OHLCV dict (sliced to as_of) for cross-asset features
    universe = None
    if universe_spot:
        universe = {}
        for asset, bars in universe_spot.items():
            sliced = bars[(bars.index >= hist_start) & (bars.index < as_of)]
            if len(sliced) >= 24 * 20:
                universe[asset] = sliced
        if len(universe) < 2:
            universe = None  # need at least 2 assets for correlation

    state = regime_engine.compute(
        ohlcv, funding_rates=fr_arr, universe_ohlcv=universe,
    )

    # Map to fixed-order feature vector
    row = state.to_feature_row()
    features = np.array(
        [row.get(name, 0.0) for name in REGIME_FEATURE_NAMES],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)


def _fetch_perp_and_funding(assets, start, end, quote="USDT", cache_dir=None):
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
        cp = cache_dir / f"fr_perp_{asset}_{start}_{end}.parquet"    if cache_dir else None
        cf = cache_dir / f"fr_funding_{asset}_{start}_{end}.parquet" if cache_dir else None

        perp_df = None
        if cp and cp.exists():
            try: perp_df = pd.read_parquet(cp)
            except: pass

        if perp_df is None:
            try:
                s_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                e_ms = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                all_bars, cur = [], s_ms
                while cur < e_ms:
                    bars = exchange.fetch_ohlcv(symbol, "1h", since=cur, limit=1000)
                    if not bars: break
                    bars = [b for b in bars if b[0] < e_ms]
                    if not bars: break
                    all_bars.extend(bars)
                    last = bars[-1][0]
                    if last <= cur: break
                    cur = last + 1
                if all_bars:
                    perp_df = pd.DataFrame(all_bars, columns=["timestamp","open","high","low","close","volume"])
                    perp_df["timestamp"] = pd.to_datetime(perp_df["timestamp"], unit="ms", utc=True)
                    perp_df = perp_df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
                    perp_df = perp_df[perp_df.index < pd.Timestamp(end, tz="UTC")]
                    if cp: perp_df.to_parquet(cp)
                    logger.info(f"  {asset} perp: {len(perp_df)} bars")
            except Exception as e:
                logger.warning(f"  Perp failed {asset}: {e}"); continue

        if perp_df is None or perp_df.empty:
            continue

        fr_series = None
        if cf and cf.exists():
            try:
                fr_df = pd.read_parquet(cf)
                fr_series = fr_df["rate"]
            except: pass

        if fr_series is None:
            try:
                s_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                e_ms = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                all_fr, cur = [], s_ms
                while cur < e_ms:
                    recs = exchange.fetch_funding_rate_history(symbol, since=cur, limit=500)
                    if not recs: break
                    recs = [r for r in recs if r["timestamp"] < e_ms]
                    if not recs: break
                    all_fr.extend(recs)
                    last = recs[-1]["timestamp"]
                    if last <= cur: break
                    cur = last + 1
                if all_fr:
                    fr_df = pd.DataFrame([
                        {"timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                         "rate": float(r["fundingRate"])} for r in all_fr
                    ]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
                    fr_df = fr_df[fr_df.index < pd.Timestamp(end, tz="UTC")]
                    if cf: fr_df.to_parquet(cf)
                    fr_series = fr_df["rate"]
                    logger.info(f"  {asset} funding: {len(fr_series)} payments")
            except Exception as e:
                logger.warning(f"  Funding failed {asset}: {e}"); continue

        if fr_series is None or fr_series.empty:
            continue
        results[asset] = {"perp": perp_df, "funding": fr_series}

    return results


def _fetch_spot(assets, start, end, quote="USDT", cache_dir=None):
    import ccxt
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.binance({"enableRateLimit": True})
    results = {}
    for asset in assets:
        symbol = f"{asset}/{quote}"
        cp = cache_dir / f"fr_spot_{asset}_{start}_{end}.parquet" if cache_dir else None
        if cp and cp.exists():
            try: results[asset] = pd.read_parquet(cp); continue
            except: pass
        try:
            s_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            e_ms = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            all_bars, cur = [], s_ms
            while cur < e_ms:
                bars = exchange.fetch_ohlcv(symbol, "1h", since=cur, limit=1000)
                if not bars: break
                bars = [b for b in bars if b[0] < e_ms]
                if not bars: break
                all_bars.extend(bars)
                last = bars[-1][0]
                if last <= cur: break
                cur = last + 1
            if all_bars:
                df = pd.DataFrame(all_bars, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
                df = df[df.index < pd.Timestamp(end, tz="UTC")]
                if cp: df.to_parquet(cp)
                results[asset] = df
                logger.info(f"  {asset} spot: {len(df)} bars")
        except Exception as e:
            logger.warning(f"  Spot failed {asset}: {e}")
    return results


class FundingRateStrategy:
    """
    Funding rate carry trade.
    Evaluation unit: N-day hold (3, 7, 14 days).
    TC paid once per position (not per 8h cycle).

    feature_mode controls which features the RF sees:
        "funding"         — 11 hand-crafted funding/basis features (default)
        "funding+regime"  — 11 funding + 46 regime features
        "regime"          — 46 regime features only
    """

    def __init__(self, assets=None, cache_dir="data/funding_cache",
                 tc_bps=4.0, training_start="2024-01-01",
                 training_end="2024-12-31", quote="USDT",
                 feature_mode="funding"):
        self.assets         = assets or DEFAULT_ASSETS
        self.cache_dir      = Path(cache_dir)
        self.tc_bps         = tc_bps
        self.training_start = training_start
        self.training_end   = training_end
        self.quote          = quote
        self.feature_mode   = feature_mode

        self._models  = [FundingModelSpec(f"{a}_FUNDING", a) for a in self.assets]
        self._configs = generate_funding_param_grid()

        # Initialize regime engine if needed
        self._regime_engine = None
        if feature_mode in ("funding+regime", "regime"):
            from engines.regime_engine import RegimeEngine
            self._regime_engine = RegimeEngine(bars_per_day=24)

        n_feat = len(self.daily_feature_names())
        print(f"  Funding Rate CPO: {len(self._models)} models ({len(self.assets)} assets)")
        print(f"  Feature mode: {feature_mode} ({n_feat} features)")
        print(f"  Total configs: {len(self._configs)} (36 per model)")
        print(f"  TC: {tc_bps} bps one-way | Hold: 3/7/14 days | TC amortized over hold")

    def get_models(self):     return self._models
    def get_param_grid(self): return self._configs

    def daily_feature_names(self) -> list[str]:
        if self.feature_mode == "funding":
            return list(FUNDING_FEATURES)
        elif self.feature_mode == "funding+regime":
            return list(FUNDING_FEATURES) + list(REGIME_FEATURE_NAMES)
        elif self.feature_mode == "regime":
            return list(REGIME_FEATURE_NAMES)
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

    def config_param_names(self) -> list[str]:
        return FundingConfig.param_names()

    def config_to_features(self, config):
        return config.to_feature_vector()

    def compute_features(self, model, as_of_date, data):
        spot      = data.get("spot", {}).get(model.asset)
        perp_data = data.get("perp", {}).get(model.asset)
        if spot is None or perp_data is None:
            return None

        # Compute funding features (if needed)
        funding_vec = None
        if self.feature_mode in ("funding", "funding+regime"):
            funding_vec = _compute_funding_features(
                spot, perp_data["perp"], perp_data["funding"], as_of_date
            )
            if funding_vec is None and self.feature_mode == "funding":
                return None

        # Compute regime features (if needed)
        regime_vec = None
        if self._regime_engine is not None:
            # Build universe dict for cross-asset features
            universe_spot = {a: bars for a, bars in data.get("spot", {}).items()
                            if a != model.asset}
            regime_vec = _compute_regime_features(
                self._regime_engine, spot,
                perp_data["funding"], as_of_date,
                universe_spot=universe_spot if universe_spot else None,
            )
            if regime_vec is None and self.feature_mode == "regime":
                return None

        # Assemble final vector based on mode
        if self.feature_mode == "funding":
            return funding_vec
        elif self.feature_mode == "funding+regime":
            if funding_vec is None:
                funding_vec = np.zeros(len(FUNDING_FEATURES), dtype=np.float32)
            if regime_vec is None:
                regime_vec = np.zeros(len(REGIME_FEATURE_NAMES), dtype=np.float32)
            return np.concatenate([funding_vec, regime_vec])
        elif self.feature_mode == "regime":
            return regime_vec
        return None

    def run_single_day(self, model, config, day, data):
        spot      = data.get("spot", {}).get(model.asset)
        perp_data = data.get("perp", {}).get(model.asset)
        if spot is None or perp_data is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        day_start = pd.Timestamp(day)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("UTC")

        # Need bars from history through end of hold window
        hold_end = day_start + timedelta(days=config.hold_days + 1)
        spot_window    = spot[spot.index <= hold_end]
        perp_window    = perp_data["perp"][perp_data["perp"].index <= hold_end]
        funding_window = perp_data["funding"][perp_data["funding"].index <= hold_end]

        return run_funding_single_day(
            spot_window, perp_window, funding_window,
            config, self.tc_bps, as_of=day_start
        )

    def run_model_year(self, model, data, param_grid):
        spot      = data.get("spot", {}).get(model.asset)
        perp_data = data.get("perp", {}).get(model.asset)
        if spot is None or perp_data is None:
            return pd.DataFrame()

        perp    = perp_data["perp"]
        funding = perp_data["funding"]
        trading_days = sorted(spot.index.normalize().unique())
        results = []

        for day_idx, day in enumerate(trading_days):
            day_start = pd.Timestamp(day)
            if day_start.tzinfo is None:
                day_start = day_start.tz_localize("UTC")

            # Include up to 14 days ahead for the hold window
            hold_end = day_start + timedelta(days=15)
            spot_w    = spot[spot.index <= hold_end]
            perp_w    = perp[perp.index <= hold_end]
            funding_w = funding[funding.index <= hold_end]

            if len(spot_w) < 3:
                continue

            for config in param_grid:
                result = run_funding_single_day(
                    spot_w, perp_w, funding_w,
                    config, self.tc_bps, as_of=day_start
                )
                dr, gr = result["daily_return"], result["gross_return"]
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
                print(f"    {model.model_id}: {day_idx+1}/{len(trading_days)} days")

        return pd.DataFrame(results)

    def fetch_training_data(self, models, start, end):
        assets = list({m.asset for m in models})
        return {
            "spot": _fetch_spot(assets, self.training_start, self.training_end,
                                self.quote, self.cache_dir),
            "perp": _fetch_perp_and_funding(assets, self.training_start, self.training_end,
                                             self.quote, self.cache_dir),
        }

    def fetch_oos_data(self, models, start, end):
        assets = list({m.asset for m in models})
        return {
            "spot": _fetch_spot(assets, start, end, self.quote, self.cache_dir),
            "perp": _fetch_perp_and_funding(assets, start, end, self.quote, self.cache_dir),
        }

    def fetch_warmup_daily(self, models, start, end):
        assets = list({m.asset for m in models})
        spot   = _fetch_spot(assets, start, end, self.quote, self.cache_dir)
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
