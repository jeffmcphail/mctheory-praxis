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
    holding_hours: int      # hours to hold the position
    entry_threshold: float  # min |return| over lookback to enter (filters noise)
    long_only: bool = True  # True = long-only (spot), False = long/short (perps)

    @staticmethod
    def param_names() -> list[str]:
        return [
            "lookback_norm",    # lookback_hours / 168  (normalized to [0,1] for 1w max)
            "holding_norm",     # holding_hours / 48
            "threshold_norm",   # entry_threshold / 0.05
            "long_only",        # 0/1
            "type_tsmom1h",     # one-hot
            "type_tsmom4h",
            "type_daily",
            "type_volscale",
            "type_dual",
            "type_reversal",
        ]

    def to_feature_vector(self) -> list[float]:
        enc = {t: 0.0 for t in MOMENTUM_TYPES}
        enc[self.mom_type] = 1.0
        return [
            min(self.lookback_hours / 168.0, 1.0),
            min(self.holding_hours  / 48.0,  1.0),
            min(self.entry_threshold / 0.05, 1.0),
            float(self.long_only),
            enc["TSMOM_1H"],
            enc["TSMOM_4H"],
            enc["TSMOM_DAILY"],
            enc["VOLSCALE"],
            enc["DUAL"],
            enc["REVERSAL"],
        ]


# ── Parameter grid ────────────────────────────────────────────────────────────

def generate_momentum_param_grid() -> dict[str, list[MomConfig]]:
    grids: dict[str, list[MomConfig]] = {}

    # Short-term momentum (1h bars, hold 1-4h)
    configs = []
    for i, (lb, hold, thresh) in enumerate(itertools.product(
        [1, 2, 4, 6],           # lookback hours
        [1, 2, 4],              # holding hours
        [0.003, 0.007, 0.015],  # entry threshold (0.3%, 0.7%, 1.5%)
    )):
        configs.append(MomConfig(f"tsmom1h_{i:03d}", "TSMOM_1H", lb, hold, thresh))
    grids["TSMOM_1H"] = configs   # 36 configs

    # Medium-term momentum (4h lookback, hold 4-24h)
    configs = []
    for i, (lb, hold, thresh) in enumerate(itertools.product(
        [4, 8, 12, 24],
        [4, 8, 12, 24],
        [0.005, 0.01, 0.02],
    )):
        configs.append(MomConfig(f"tsmom4h_{i:03d}", "TSMOM_4H", lb, hold, thresh))
    grids["TSMOM_4H"] = configs   # 48 configs

    # Daily momentum (24h lookback, hold 1-3 days)
    configs = []
    for i, (lb, hold, thresh) in enumerate(itertools.product(
        [24, 48, 72, 168],
        [12, 24, 48],
        [0.01, 0.02, 0.04],
    )):
        configs.append(MomConfig(f"daily_{i:03d}", "TSMOM_DAILY", lb, hold, thresh))
    grids["TSMOM_DAILY"] = configs  # 36 configs

    # Vol-scaled momentum (position = signal / realized_vol)
    configs = []
    for i, (lb, hold, thresh) in enumerate(itertools.product(
        [4, 12, 24],
        [4, 12, 24],
        [0.005, 0.015],
    )):
        configs.append(MomConfig(f"volscale_{i:03d}", "VOLSCALE", lb, hold, thresh))
    grids["VOLSCALE"] = configs   # 18 configs

    # Dual momentum (short AND long must agree)
    configs = []
    for i, (short_lb, long_lb, hold) in enumerate(itertools.product(
        [2, 4],
        [24, 72],
        [4, 12, 24],
    )):
        configs.append(MomConfig(f"dual_{i:03d}", "DUAL", short_lb, hold, 0.0,
                                  long_only=True))
        # Store long_lb in entry_threshold field (hack — reused for dual)
        configs[-1].entry_threshold = long_lb / 100.0  # encode long lookback
    grids["DUAL"] = configs   # 12 configs

    # Short-term reversal (fade the last 1-4h move)
    configs = []
    for i, (lb, hold, thresh) in enumerate(itertools.product(
        [1, 2, 4],
        [1, 2, 4],
        [0.005, 0.01, 0.02],
    )):
        configs.append(MomConfig(f"rev_{i:03d}", "REVERSAL", lb, hold, thresh))
    grids["REVERSAL"] = configs   # 27 configs

    return grids


# ── Single-day execution ──────────────────────────────────────────────────────

def run_momentum_single_day(
    bars_hist: pd.DataFrame,   # warmup history (>= 168h)
    bars_day: pd.DataFrame,    # today's hourly bars
    config: MomConfig,
    tc_bps: float = 2.0,
) -> dict:
    """
    Run one momentum config on one day's bars.

    Signal generation:
        - Compute lookback return at the start of each bar
        - If |ret| > threshold: go long (or short for reversal)
        - Hold for holding_hours, then close

    Returns {"daily_return": float, "gross_return": float, "n_trades": int}
    """
    if len(bars_day) < 2:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    tc_pct = tc_bps / 10000.0  # bps → decimal
    combined = pd.concat([bars_hist, bars_day]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    close = combined["close"]

    equity = 1.0
    n_trades = 0
    in_pos = False
    entry_px = 0.0
    hold_remaining = 0
    direction = 1  # 1=long, -1=short

    lb = config.lookback_hours
    hold = config.holding_hours
    thresh = config.entry_threshold

    for i, idx in enumerate(bars_day.index):
        loc = combined.index.get_loc(idx)
        if loc < lb:
            continue

        px = float(close.iloc[loc])

        # Close position if hold period elapsed
        if in_pos:
            hold_remaining -= 1
            if hold_remaining <= 0:
                ret = (px - entry_px) / entry_px * direction
                equity *= (1 + ret - tc_pct)
                in_pos = False

        # New signal
        if not in_pos:
            past_px = float(close.iloc[loc - lb])
            ret_signal = (px - past_px) / (past_px + 1e-10)

            if config.mom_type == "REVERSAL":
                # Fade: if went up, go short (or skip if long_only)
                if abs(ret_signal) >= thresh:
                    if ret_signal > 0 and not config.long_only:
                        direction = -1
                        entry_px = px
                        in_pos = True
                        hold_remaining = hold
                        n_trades += 1
                        equity *= (1 - tc_pct)
                    elif ret_signal < 0:
                        direction = 1
                        entry_px = px
                        in_pos = True
                        hold_remaining = hold
                        n_trades += 1
                        equity *= (1 - tc_pct)

            elif config.mom_type == "DUAL":
                # Require short AND long to agree
                long_lb = int(config.entry_threshold * 100)  # encoded above
                if loc >= long_lb:
                    long_px = float(close.iloc[loc - long_lb])
                    long_ret = (px - long_px) / (long_px + 1e-10)
                    if ret_signal > 0 and long_ret > 0:
                        direction = 1
                        entry_px = px
                        in_pos = True
                        hold_remaining = hold
                        n_trades += 1
                        equity *= (1 - tc_pct)

            elif config.mom_type == "VOLSCALE":
                # Vol-scale: signal = ret / realized_vol
                if loc >= max(lb, 24):
                    rets_window = close.iloc[loc-24:loc].pct_change().dropna()
                    rvol = float(rets_window.std()) if len(rets_window) > 2 else 1e-4
                    scaled = ret_signal / (rvol * np.sqrt(24) + 1e-10)
                    if abs(scaled) > 1.0 and ret_signal > thresh:
                        direction = 1
                        entry_px = px
                        in_pos = True
                        hold_remaining = hold
                        n_trades += 1
                        equity *= (1 - tc_pct)

            else:
                # TSMOM_1H / TSMOM_4H / TSMOM_DAILY: follow the move
                if ret_signal >= thresh:
                    direction = 1
                    entry_px = px
                    in_pos = True
                    hold_remaining = hold
                    n_trades += 1
                    equity *= (1 - tc_pct)
                elif ret_signal <= -thresh and not config.long_only:
                    direction = -1
                    entry_px = px
                    in_pos = True
                    hold_remaining = hold
                    n_trades += 1
                    equity *= (1 - tc_pct)

    # Close open position at day end
    if in_pos:
        final_px = float(bars_day["close"].iloc[-1])
        ret = (final_px - entry_px) / entry_px * direction
        equity *= (1 + ret - tc_pct)

    daily_return = equity - 1.0
    gross_return = daily_return + tc_pct * n_trades * 2 if n_trades > 0 else daily_return

    return {
        "daily_return": float(daily_return),
        "gross_return": float(gross_return),
        "n_trades":     n_trades,
    }


# ── Daily feature computation ─────────────────────────────────────────────────

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
