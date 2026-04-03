"""
engines/momentum_strategy.py
=============================
Time-series momentum strategy suite for the CPO framework.

Momentum hypothesis: assets that have gone up over the past N hours tend
to continue going up over the next N hours (positive autocorrelation).
This is a documented risk premium in equities (Jegadeesh & Titman 1993)
and has been shown to exist in crypto on sub-weekly horizons.

Models: (asset, momentum_type)
    TSMOM_1H   — short-horizon raw momentum (1–6h lookback)
    TSMOM_4H   — medium-horizon raw momentum (4–48h lookback)
    TSMOM_DAILY — daily momentum (24–168h lookback)
    VOLSCALE    — volatility-scaled momentum (position sized by 1/realized_vol)
    DUAL        — requires both short AND long momentum to agree
    REVERSAL    — short-term reversal (1–4h lookback, fade the move)

Configs: lookback_hours × holding_hours × entry_threshold
    entry_threshold: minimum absolute return required to enter
    (filters out noise trades when recent move is tiny)

Data: CCXT/Binance hourly bars (same source as crypto_ta_strategy)

CPO fit: base_rate ~35-50% (momentum works roughly half the time per asset),
    giving the RF enough variance to learn conditional patterns.
    Trading frequency: 1-4 trades per asset per day → sufficient samples.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engines.triple_barrier import (
    BarrierConfig, standard_barrier_grid, simulate_trade,
)

logger = logging.getLogger(__name__)

# ── Universe ──────────────────────────────────────────────────────────────────

DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE"]

MOMENTUM_TYPES = [
    "TSMOM_1H",     # short-term raw momentum
    "TSMOM_4H",     # medium-term raw momentum
    "TSMOM_DAILY",  # daily momentum
    "VOLSCALE",     # vol-scaled momentum
    "DUAL",         # dual-timeframe confirmation
    "REVERSAL",     # short-term reversal (fade)
]

# Daily features fed to the RF
MOMENTUM_FEATURES = [
    "ret_1h",         # last 1h return
    "ret_4h",         # last 4h return
    "ret_24h",        # last 24h return
    "ret_7d",         # last 7-day return
    "vol_24h",        # 24h realized volatility (annualized)
    "vol_regime",     # today vol / 30-day vol (vol regime indicator)
    "volume_surge",   # today volume / 7-day avg volume
    "skew_4h",        # skewness of 4h returns over last 24h
    "autocorr_1h",    # 1h return autocorrelation (last 24 bars)
    "btc_corr_24h",   # correlation with BTC over last 24h (for non-BTC assets)
    "trend_strength", # ADX-style trend strength
    "ret_sign_persist", # fraction of last 12h with same sign as last hour
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MomModelSpec:
    model_id: str    # e.g. "BTC_TSMOM_4H"
    asset: str
    mom_type: str


@dataclass
class MomConfig:
    config_id: str
    mom_type: str
    lookback_hours: int     # hours to look back for the momentum signal
    entry_threshold: float  # min |return| over lookback to enter (filters noise)
    long_only: bool = True
    # Triple barrier exit params
    sl_pct:    float = 0.020
    tp_pct:    float = 0.060   # default 3:1 for momentum
    trail_pct: float = 0.005   # momentum default: trailing stop after TP
    t_bars:    int   = 48

    @staticmethod
    def param_names() -> list[str]:
        return [
            "lookback_norm",
            "threshold_norm",
            "long_only",
            "type_tsmom1h",
            "type_tsmom4h",
            "type_daily",
            "type_volscale",
            "type_dual",
            "type_reversal",
        ] + BarrierConfig.param_names()

    def to_feature_vector(self) -> list[float]:
        enc = {t: 0.0 for t in MOMENTUM_TYPES}
        enc[self.mom_type] = 1.0
        barrier = BarrierConfig(
            sl_pct=self.sl_pct, tp_pct=self.tp_pct,
            trail_pct=self.trail_pct, t_bars=self.t_bars,
        )
        return [
            min(self.lookback_hours / 168.0, 1.0),
            min(self.entry_threshold / 0.05, 1.0),
            float(self.long_only),
            enc["TSMOM_1H"],
            enc["TSMOM_4H"],
            enc["TSMOM_DAILY"],
            enc["VOLSCALE"],
            enc["DUAL"],
            enc["REVERSAL"],
        ] + barrier.to_feature_vector()

    def barrier(self) -> BarrierConfig:
        return BarrierConfig(
            sl_pct=self.sl_pct, tp_pct=self.tp_pct,
            trail_pct=self.trail_pct, t_bars=self.t_bars,
        )


# ── Parameter grid ────────────────────────────────────────────────────────────

def generate_momentum_param_grid() -> dict[str, list[MomConfig]]:
    """
    Generate momentum parameter grid crossed with barrier configs.

    Each signal config (lookback × threshold) is crossed with
    the standard barrier grid (72 exit combos).
    """
    barriers = standard_barrier_grid()
    grids: dict[str, list[MomConfig]] = {}
    cid_counter = [0]

    def next_id() -> str:
        cid_counter[0] += 1
        return f"mom_{cid_counter[0]:05d}"

    def cross(signal_params: list[dict], mom_type: str) -> list[MomConfig]:
        result = []
        for sig in signal_params:
            for b in barriers:
                result.append(MomConfig(
                    config_id=next_id(),
                    mom_type=mom_type,
                    sl_pct=b.sl_pct, tp_pct=b.tp_pct,
                    trail_pct=b.trail_pct, t_bars=b.t_bars,
                    **sig,
                ))
        return result

    # TSMOM_1H: short-term  (4 lookbacks × 3 thresholds × 72 barriers = 864)
    grids["TSMOM_1H"] = cross([
        {"lookback_hours": lb, "entry_threshold": thresh, "long_only": True}
        for lb in [1, 2, 4, 6]
        for thresh in [0.003, 0.007, 0.015]
    ], "TSMOM_1H")

    # TSMOM_4H: medium-term  (4 × 3 × 72 = 864)
    grids["TSMOM_4H"] = cross([
        {"lookback_hours": lb, "entry_threshold": thresh, "long_only": True}
        for lb in [4, 8, 12, 24]
        for thresh in [0.005, 0.010, 0.020]
    ], "TSMOM_4H")

    # TSMOM_DAILY: daily  (4 × 3 × 72 = 864)
    grids["TSMOM_DAILY"] = cross([
        {"lookback_hours": lb, "entry_threshold": thresh, "long_only": True}
        for lb in [24, 48, 72, 168]
        for thresh in [0.010, 0.020, 0.040]
    ], "TSMOM_DAILY")

    # VOLSCALE: vol-scaled  (3 × 2 × 72 = 432)
    grids["VOLSCALE"] = cross([
        {"lookback_hours": lb, "entry_threshold": thresh, "long_only": True}
        for lb in [4, 12, 24]
        for thresh in [0.005, 0.015]
    ], "VOLSCALE")

    # DUAL: requires both short and long to agree  (4 × 72 = 288)
    # encode long_lookback via entry_threshold field (×100)
    grids["DUAL"] = cross([
        {"lookback_hours": short_lb,
         "entry_threshold": long_lb / 100.0,   # encodes long lookback
         "long_only": True}
        for short_lb in [2, 4]
        for long_lb in [24, 72]
    ], "DUAL")

    # REVERSAL: fade the move  (3 × 3 × 72 = 648)
    grids["REVERSAL"] = cross([
        {"lookback_hours": lb, "entry_threshold": thresh, "long_only": True}
        for lb in [1, 2, 4]
        for thresh in [0.005, 0.010, 0.020]
    ], "REVERSAL")

    return grids


# ── Single-day execution ──────────────────────────────────────────────────────

def run_momentum_single_day(
    bars_hist: pd.DataFrame,
    bars_day: pd.DataFrame,
    config: MomConfig,
    tc_bps: float = 2.0,
) -> dict:
    """
    Run one momentum config on one day using the triple barrier framework.
    Entry: momentum signal exceeds threshold. Exit: SL / TP-then-trailing / vertical.
    """
    if len(bars_day) < 2:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    tc_pct  = tc_bps / 10000.0
    barrier = config.barrier()
    combined = pd.concat([bars_hist, bars_day]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    close = combined["close"].values

    equity    = 1.0
    gross_tot = 0.0
    n_trades  = 0
    pos_end   = -1   # absolute index into close[] where position ends

    n_hist = len(combined) - len(bars_day)
    lb     = config.lookback_hours
    thresh = config.entry_threshold

    for rel_i in range(len(bars_day)):
        abs_i = n_hist + rel_i

        if abs_i <= pos_end:
            continue
        if abs_i < lb:
            continue

        px = close[abs_i]
        if np.isnan(px):
            continue

        past_px    = close[abs_i - lb]
        ret_signal = (px - past_px) / (past_px + 1e-10)

        if config.mom_type == "REVERSAL":
            if abs(ret_signal) < thresh:
                continue
            direction = -1 if ret_signal > 0 else 1
            if direction == -1 and config.long_only:
                continue

        elif config.mom_type == "DUAL":
            long_lb  = int(thresh * 100)   # encoded long lookback
            if abs_i < long_lb:
                continue
            long_ret = (px - close[abs_i - long_lb]) / (close[abs_i - long_lb] + 1e-10)
            if ret_signal > 0 and long_ret > 0:
                direction = 1
            else:
                continue

        elif config.mom_type == "VOLSCALE":
            if abs_i < max(lb, 24):
                continue
            rvol = float(np.std(np.diff(np.log(close[abs_i-24:abs_i] + 1e-10))))
            rvol = max(rvol, 1e-4)
            scaled = ret_signal / (rvol * np.sqrt(24))
            if abs(scaled) <= 1.0 or ret_signal < thresh:
                continue
            direction = 1

        else:
            if ret_signal >= thresh:
                direction = 1
            elif ret_signal <= -thresh and not config.long_only:
                direction = -1
            else:
                continue

        result  = simulate_trade(close[abs_i:], direction, barrier, tc_pct)
        equity    *= (1 + result["net_return"])
        gross_tot += result["gross_return"]
        n_trades  += 1
        pos_end    = abs_i + result["exit_idx"]

    return {
        "daily_return": float(equity - 1.0),
        "gross_return": float(gross_tot),
        "n_trades":     n_trades,
    }


def _compute_momentum_features(
    bars: pd.DataFrame,
    as_of_date: str,
    btc_bars: pd.DataFrame | None = None,
) -> np.ndarray | None:
    """Compute momentum-relevant daily features from hourly bars."""
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    # Need 30+ days of history for vol regime
    hist_start = as_of - timedelta(days=35)
    window = bars[(bars.index >= hist_start) & (bars.index < as_of)]

    if len(window) < 48:
        return None

    close = window["close"]
    vol = window["volume"]

    def safe_ret(n_hours: int) -> float:
        if len(close) < n_hours + 1:
            return 0.0
        return float((close.iloc[-1] - close.iloc[-n_hours-1]) /
                     (close.iloc[-n_hours-1] + 1e-10))

    # Returns at multiple horizons
    ret_1h   = safe_ret(1)
    ret_4h   = safe_ret(4)
    ret_24h  = safe_ret(24)
    ret_7d   = safe_ret(168)

    # Realized vol (24h, annualized)
    rets_24h = close.iloc[-25:].pct_change().dropna()
    vol_24h  = float(rets_24h.std() * np.sqrt(24 * 365)) if len(rets_24h) > 2 else 0.01

    # Vol regime: today vol / 30-day rolling vol
    rets_all = close.pct_change().dropna()
    vol_30d  = float(rets_all.std()) if len(rets_all) > 5 else 1e-4
    vol_regime = float(rets_24h.std() / (vol_30d + 1e-10)) if len(rets_24h) > 2 else 1.0

    # Volume surge: last 24h volume vs 7-day average
    vol_24h_sum = float(vol.iloc[-24:].sum()) if len(vol) >= 24 else 0.0
    vol_7d_avg  = float(vol.iloc[-168:].mean()) * 24 if len(vol) >= 168 else vol_24h_sum
    volume_surge = vol_24h_sum / (vol_7d_avg + 1e-10)

    # Skewness of 4h returns over last 24h
    rets_4h_chunks = [float(close.iloc[i+4] / close.iloc[i] - 1)
                      for i in range(-28, -4, 4)
                      if i + 4 < 0 and abs(i) < len(close)]
    skew_4h = float(np.array(rets_4h_chunks).mean()) if rets_4h_chunks else 0.0

    # 1h return autocorrelation over last 24 bars
    if len(rets_24h) >= 12:
        ac = np.corrcoef(rets_24h.values[:-1], rets_24h.values[1:])[0, 1]
        autocorr_1h = float(ac) if not np.isnan(ac) else 0.0
    else:
        autocorr_1h = 0.0

    # BTC correlation (for non-BTC assets)
    btc_corr = 0.5  # default
    if btc_bars is not None and len(btc_bars) >= 48:
        btc_window = btc_bars[(btc_bars.index >= hist_start) & (btc_bars.index < as_of)]
        if len(btc_window) >= 24:
            merged = pd.concat([
                close.rename("asset"),
                btc_window["close"].rename("btc")
            ], axis=1).dropna()
            if len(merged) >= 24:
                asset_rets = merged["asset"].pct_change().dropna()
                btc_rets   = merged["btc"].pct_change().dropna()
                min_len = min(len(asset_rets), len(btc_rets))
                if min_len >= 10:
                    cc = np.corrcoef(asset_rets.values[-min_len:],
                                     btc_rets.values[-min_len:])[0, 1]
                    btc_corr = float(cc) if not np.isnan(cc) else 0.5

    # Trend strength: |ret_24h| / vol_24h_raw
    vol_24h_raw = float(rets_24h.std()) if len(rets_24h) > 2 else 1e-4
    trend_strength = abs(ret_24h) / (vol_24h_raw * np.sqrt(24) + 1e-10)

    # Sign persistence: fraction of last 12 hours with same return sign as last hour
    if len(rets_24h) >= 12:
        last_sign = np.sign(ret_1h)
        sign_persist = float(np.mean(np.sign(rets_24h.values[-12:]) == last_sign))
    else:
        sign_persist = 0.5

    features = np.array([
        np.clip(ret_1h,   -0.2, 0.2),
        np.clip(ret_4h,   -0.3, 0.3),
        np.clip(ret_24h,  -0.5, 0.5),
        np.clip(ret_7d,   -1.0, 1.0),
        np.clip(vol_24h,   0.0, 10.0),
        np.clip(vol_regime, 0.0, 5.0),
        np.clip(volume_surge, 0.0, 5.0),
        np.clip(skew_4h,  -0.05, 0.05),
        np.clip(autocorr_1h, -1.0, 1.0),
        np.clip(btc_corr,  -1.0, 1.0),
        np.clip(trend_strength, 0.0, 5.0),
        np.clip(sign_persist, 0.0, 1.0),
    ], dtype=np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)


# ── Data fetching (reuses same CCXT pattern) ──────────────────────────────────

def _fetch_hourly(
    assets: list[str],
    start: str,
    end: str,
    quote: str = "USDT",
    cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch hourly bars from Binance via CCXT. Caches to parquet."""
    import ccxt
    from datetime import datetime, timezone

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance({"enableRateLimit": True})
    results = {}

    for asset in assets:
        symbol = f"{asset}/{quote}"
        cache_key = f"mom_{asset}_1h_{start}_{end}.parquet"
        cache_path = cache_dir / cache_key if cache_dir else None

        if cache_path and cache_path.exists():
            try:
                results[asset] = pd.read_parquet(cache_path)
                logger.info(f"  Cache hit: {asset}")
                continue
            except Exception:
                pass

        logger.info(f"  Fetching {symbol} 1h {start}→{end}")
        try:
            since_ms = int(datetime.strptime(start, "%Y-%m-%d")
                          .replace(tzinfo=timezone.utc).timestamp() * 1000)
            end_ms   = int(datetime.strptime(end, "%Y-%m-%d")
                          .replace(tzinfo=timezone.utc).timestamp() * 1000)

            all_bars, cursor = [], since_ms
            while cursor < end_ms:
                bars = exchange.fetch_ohlcv(symbol, "1h", since=cursor, limit=1000)
                if not bars:
                    break
                bars = [b for b in bars if b[0] < end_ms]
                if not bars:
                    break
                all_bars.extend(bars)
                last_ts = bars[-1][0]
                if last_ts <= cursor:
                    break
                cursor = last_ts + 1

            if not all_bars:
                continue

            df = pd.DataFrame(all_bars,
                columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
            df = df[df.index < pd.Timestamp(end, tz="UTC")]

            if cache_path:
                df.to_parquet(cache_path)

            results[asset] = df
            logger.info(f"    {asset}: {len(df)} bars")

        except Exception as e:
            logger.warning(f"  Failed to fetch {symbol}: {e}")

    return results


# ── TradingStrategy implementation ────────────────────────────────────────────

class MomentumCPOStrategy:
    """
    Time-series momentum strategy suite as a CPO TradingStrategy.

    Each model = (asset, momentum_type).
    Configs = (lookback, holding_period, threshold).
    RF learns: given yesterday's price dynamics, which momentum
    variant + params will be profitable today?

    Expected CPO fit:
        - Base rate ~35-50% (momentum works about half the time)
        - Daily trades: 1-3 per model per day
        - Training samples: 36-48 configs × 365 days = 13-18K per model
        - AUC target: 0.55-0.75 (meaningful but not degenerate)
    """

    def __init__(
        self,
        assets: list[str] | None = None,
        mom_types: list[str] | None = None,
        cache_dir: str | Path = "data/momentum_cache",
        tc_bps: float = 2.0,
        training_start: str = "2024-01-01",
        training_end: str = "2024-12-31",
        quote: str = "USDT",
        long_only: bool = True,
    ):
        self.assets       = assets or DEFAULT_ASSETS
        self.mom_types    = mom_types or MOMENTUM_TYPES
        self.cache_dir    = Path(cache_dir)
        self.tc_bps       = tc_bps
        self.training_start = training_start
        self.training_end   = training_end
        self.quote        = quote
        self.long_only    = long_only

        self._models = [
            MomModelSpec(f"{asset}_{mt}", asset, mt)
            for asset, mt in itertools.product(self.assets, self.mom_types)
        ]

        self._grids = generate_momentum_param_grid()
        self._all_configs = []
        for mt in self.mom_types:
            self._all_configs.extend(self._grids.get(mt, []))

        print(f"  Momentum CPO: {len(self._models)} models "
              f"({len(self.assets)} assets × {len(self.mom_types)} types)")
        print(f"  Total configs: {len(self._all_configs)}")
        for mt in self.mom_types:
            print(f"    {mt}: {len(self._grids.get(mt,[]))} configs")

    # ── Protocol ──────────────────────────────────────────────────

    def get_models(self):
        return self._models

    def get_param_grid(self):
        return self._all_configs

    def daily_feature_names(self) -> list[str]:
        return list(MOMENTUM_FEATURES)

    def config_param_names(self) -> list[str]:
        return MomConfig.param_names()

    def config_to_features(self, config: MomConfig) -> list[float]:
        return config.to_feature_vector()

    def compute_features(self, model: MomModelSpec, as_of_date: str,
                          data: dict) -> np.ndarray | None:
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None or len(bars) < 48:
            return None
        btc_bars = hourly.get("BTC") if model.asset != "BTC" else None
        return _compute_momentum_features(bars, as_of_date, btc_bars)

    def run_single_day(self, model: MomModelSpec, config: MomConfig,
                        day: Any, data: dict) -> dict:
        if config.mom_type != model.mom_type:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        day_start = pd.Timestamp(day)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("UTC")
        day_end = day_start + timedelta(hours=24)

        bars_day = bars[(bars.index >= day_start) & (bars.index < day_end)]
        if len(bars_day) < 3:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        hist_start = day_start - timedelta(hours=200)  # ~8 days warmup
        bars_hist  = bars[(bars.index >= hist_start) & (bars.index < day_start)]

        return run_momentum_single_day(bars_hist, bars_day, config, self.tc_bps)

    def run_model_year(self, model: MomModelSpec, data: dict,
                        param_grid: list[MomConfig]) -> pd.DataFrame:
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return pd.DataFrame()

        model_configs = [c for c in param_grid if c.mom_type == model.mom_type]
        if not model_configs:
            return pd.DataFrame()

        trading_days = sorted(bars.index.normalize().unique())

        results = []
        for day_idx, day in enumerate(trading_days):
            day_start = pd.Timestamp(day)
            if day_start.tzinfo is None:
                day_start = day_start.tz_localize("UTC")
            day_end   = day_start + timedelta(hours=24)

            bars_day = bars[(bars.index >= day_start) & (bars.index < day_end)]
            if len(bars_day) < 3:
                continue

            hist_start = day_start - timedelta(hours=200)
            bars_hist  = bars[(bars.index >= hist_start) & (bars.index < day_start)]

            for config in model_configs:
                result = run_momentum_single_day(bars_hist, bars_day, config, self.tc_bps)
                results.append({
                    "model_id":     model.model_id,
                    "date":         day.strftime("%Y-%m-%d"),
                    "config_id":    config.config_id,
                    "daily_return": result["daily_return"],
                    "gross_return": result["gross_return"],
                    "n_trades":     result["n_trades"],
                })

            if (day_idx + 1) % 30 == 0:
                print(f"    {model.model_id}: {day_idx+1}/{len(trading_days)} days")

        return pd.DataFrame(results)

    def fetch_training_data(self, models, start, end) -> dict:
        assets = list({m.asset for m in models})
        # Always include BTC for correlation feature
        if "BTC" not in assets:
            assets.append("BTC")
        hourly = _fetch_hourly(assets, self.training_start, self.training_end,
                               self.quote, self.cache_dir)
        return {"hourly_data": hourly}

    def fetch_oos_data(self, models, start, end) -> dict:
        assets = list({m.asset for m in models})
        if "BTC" not in assets:
            assets.append("BTC")
        hourly = _fetch_hourly(assets, start, end, self.quote, self.cache_dir)
        return {"hourly_data": hourly}

    def fetch_warmup_daily(self, models, start, end) -> dict[str, pd.Series]:
        assets = list({m.asset for m in models})
        if "BTC" not in assets:
            assets.append("BTC")
        hourly = _fetch_hourly(assets, start, end, self.quote, self.cache_dir)
        return {a: b["close"].resample("D").last().dropna()
                for a, b in hourly.items()}

    def get_daily_prices(self, oos_data: dict, models) -> dict[str, pd.Series]:
        hourly = oos_data.get("hourly_data", {})
        return {a: b["close"].resample("D").last().dropna()
                for a, b in hourly.items()}

    def get_trading_days(self, data: dict) -> list:
        hourly = data.get("hourly_data", {})
        if not hourly:
            return []
        all_days = set()
        for bars in hourly.values():
            all_days.update(bars.index.normalize().unique())
        return sorted(all_days)

    def prepare_warmup(self, daily_prices: dict, warmup_daily: dict) -> dict:
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
