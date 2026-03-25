"""
engines/mcb_cpo_strategy.py
============================
Market Cipher B as a CPO TradingStrategy.

Implements the TradingStrategy protocol from cpo_core.py so the RF+Kelly
framework can conditionally select MCb strategy + params each day.

Models:   (symbol, timeframe, mcb_strategy_id)
          e.g. BTC_1h_anchor_trigger, ETH_1h_zero_line_rejection

Configs:  parameter combinations for each MCb strategy
          e.g. {os_level: -53, require_mfi: True, exit_above: 0}

Features: MCb indicator summaries from the previous day
          wt2_avg, wt2_min, mfi_pct_green, vol_regime, trend_dir, rsi_avg...

Data:     CCXT/Binance (same source as the GUI backtest, no API key needed)
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default universe ─────────────────────────────────────────────────────────

DEFAULT_ASSETS     = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE"]
DEFAULT_TIMEFRAMES = ["1h"]   # 1h is the best balance for MCb signal quality
DEFAULT_STRATEGIES = ["anchor_trigger", "zero_line_rejection",
                      "bullish_divergence", "mfi_momentum"]

# Daily feature names (must match compute_features output order)
MCB_FEATURES = [
    "wt2_avg",          # average WT2 value yesterday
    "wt2_min",          # min WT2 (depth of OS excursion)
    "wt2_max",          # max WT2 (depth of OB excursion)
    "mfi_pct_green",    # % of bars with positive MFI
    "rsi_avg",          # average RSI
    "wt_cross_count",   # number of WT crosses yesterday
    "vol_regime",       # today's 24h vol / 30-day rolling vol
    "ret_1d",           # yesterday's 24h return
    "ret_3d",           # 3-day return
    "trend_strength",   # abs(ret_3d) / vol_3d — directional strength
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MCBModelSpec:
    model_id: str       # e.g. "BTC_1h_anchor_trigger"
    asset: str          # e.g. "BTC"
    timeframe: str      # e.g. "1h"
    strategy_id: str    # e.g. "anchor_trigger"


@dataclass
class MCBConfig:
    config_id: str
    strategy_id: str
    params: dict        # strategy param dict passed to MCBStrategy.__init__

    @staticmethod
    def param_names() -> list[str]:
        """Fixed-length feature vector names for the RF."""
        return [
            "os_level_norm",      # os_level / -100 → [0,1]
            "require_mfi",        # bool → 0/1
            "reset_above_norm",   # (reset_above + 60) / 80
            "exit_above_norm",    # (exit_above + 20) / 60
            "div_lookback_norm",  # div_lookback / 20
            "mfi_threshold_norm", # (mfi_threshold + 30) / 60
            "ob_stop_norm",       # ob_stop / 100
            "strategy_enc_0",     # one-hot: anchor_trigger
            "strategy_enc_1",     # one-hot: zero_line_rejection
            "strategy_enc_2",     # one-hot: bullish_divergence
            "strategy_enc_3",     # one-hot: mfi_momentum
        ]

    def to_feature_vector(self) -> list[float]:
        p = self.params
        enc = [0.0, 0.0, 0.0, 0.0]
        idx = DEFAULT_STRATEGIES.index(self.strategy_id) if self.strategy_id in DEFAULT_STRATEGIES else 0
        enc[idx] = 1.0
        return [
            abs(p.get("os_level", -53)) / 100.0,
            float(p.get("require_green_mfi", True)),
            (p.get("reset_above", -20) + 60) / 80.0,
            (p.get("exit_above", 0) + 20) / 60.0,
            p.get("div_lookback", 5) / 20.0,
            (p.get("mfi_threshold", 0) + 30) / 60.0,
            p.get("ob_stop", 70) / 100.0,
            *enc,
        ]


# ── Parameter grid generation ─────────────────────────────────────────────────

def _grid(base: dict, **variations) -> list[dict]:
    """Generate all combinations of param variations over a base dict."""
    keys = list(variations.keys())
    results = []
    for combo in itertools.product(*variations.values()):
        p = dict(base)
        p.update(dict(zip(keys, combo)))
        results.append(p)
    return results


def generate_mcb_param_grid() -> dict[str, list[MCBConfig]]:
    """Build parameter grid for each MCb strategy."""
    grids: dict[str, list[MCBConfig]] = {}

    # Anchor & Trigger
    configs = []
    for i, p in enumerate(_grid({},
        os_level=[-53.0, -60.0, -67.0],
        require_green_mfi=[True, False],
        reset_above=[-20.0, -10.0],
        exit_above=[0.0, 10.0],
    )):
        configs.append(MCBConfig(f"anchor_{i:03d}", "anchor_trigger", p))
    grids["anchor_trigger"] = configs

    # Zero Line Rejection
    configs = []
    for i, p in enumerate(_grid({},
        entry_floor=[-30.0, -40.0, -20.0],
        entry_ceiling=[5.0, 0.0, 10.0],
        require_green_mfi=[True, False],
        ob_exit=[53.0, 60.0, 45.0],
    )):
        configs.append(MCBConfig(f"zlr_{i:03d}", "zero_line_rejection", p))
    grids["zero_line_rejection"] = configs

    # Bullish Divergence
    configs = []
    for i, p in enumerate(_grid({},
        os_level=[-40.0, -53.0, -30.0],
        div_lookback=[3, 5, 8],
        exit_level=[0.0, 10.0],
        mfi_filter=[True, False],
    )):
        configs.append(MCBConfig(f"div_{i:03d}", "bullish_divergence", p))
    grids["bullish_divergence"] = configs

    # MFI Momentum
    configs = []
    for i, p in enumerate(_grid({},
        mfi_threshold=[0.0, 5.0, -5.0],
        require_wt_bull=[True, False],
        mfi_exit_threshold=[0.0, -5.0],
        ob_stop=[70.0, 60.0, 80.0],
    )):
        configs.append(MCBConfig(f"mfi_{i:03d}", "mfi_momentum", p))
    grids["mfi_momentum"] = configs

    return grids


# ── Single-day execution ──────────────────────────────────────────────────────

def run_mcb_single_day(
    bars_hist: pd.DataFrame,     # history bars (for MCb warmup)
    bars_day: pd.DataFrame,      # today's bars (simulation window)
    config: MCBConfig,
    tc_bps: float = 10.0,        # round-trip TC in basis points
) -> dict:
    """
    Run one MCb strategy+config on one day's bars.
    Returns {"daily_return": float, "gross_return": float, "n_trades": int}
    """
    from praxis.indicators.market_cipher_b import MarketCipherB
    from engines.mcb_strategies import get_strategy

    if len(bars_day) < 3:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    # Combine history + day for MCb warmup (need ~50+ bars for stable EMA)
    combined = pd.concat([bars_hist, bars_day]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Calculate MCb indicators
    mcb = MarketCipherB()
    with_indicators = mcb.calculate(combined)

    # Extract just today's bars (post-warmup)
    day_rows = with_indicators.loc[
        with_indicators.index.isin(bars_day.index)
    ]

    if len(day_rows) < 2:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    # Apply strategy signals
    strategy = get_strategy(config.strategy_id, config.params)
    day_with_signals = strategy.generate_signals(day_rows)

    # Simulate trades (simple long-only, one position at a time)
    tc_pct = tc_bps / 100.0 / 100.0   # bps → decimal
    equity = 1.0
    in_pos = False
    entry_px = 0.0
    n_trades = 0

    for _, row in day_with_signals.iterrows():
        px = float(row["close"])

        # Exit
        if in_pos and row.get("exit_signal", False):
            gross = (px - entry_px) / entry_px
            net   = gross - tc_pct * 2
            equity *= (1 + net)
            in_pos = False

        # Entry
        if not in_pos and row.get("entry", False):
            entry_px = px * (1 + tc_pct)   # slippage/TC on entry
            in_pos = True
            n_trades += 1

    # Close any open position at day end
    if in_pos:
        px = float(day_with_signals["close"].iloc[-1])
        gross = (px - entry_px) / entry_px
        net   = gross - tc_pct
        equity *= (1 + net)

    daily_return = equity - 1.0
    gross_return = daily_return + tc_pct * 2 * n_trades if n_trades else daily_return

    return {
        "daily_return": float(daily_return),
        "gross_return": float(gross_return),
        "n_trades":     n_trades,
    }


# ── Daily feature computation ─────────────────────────────────────────────────

def _compute_mcb_features(bars: pd.DataFrame, as_of_date: str) -> np.ndarray | None:
    """Compute MCb-specific daily features from hourly bars."""
    from praxis.indicators.market_cipher_b import MarketCipherB

    as_of = pd.Timestamp(as_of_date, tz="UTC")
    hist_start = as_of - timedelta(days=35)

    window = bars[(bars.index >= hist_start) & (bars.index < as_of)]
    if len(window) < 48:   # need at least 2 days of hourly
        return None

    # Calculate MCb on history
    mcb = MarketCipherB()
    try:
        ind = mcb.calculate(window)
    except Exception:
        return None

    # Yesterday's bars
    yday_start = as_of - timedelta(days=1)
    yday = ind[(ind.index >= yday_start) & (ind.index < as_of)]
    if len(yday) < 6:
        return None

    # 30-day window for vol regime
    month_start = as_of - timedelta(days=30)
    month = window[window.index >= month_start]

    close_all = window["close"].values
    close_yday = yday["close"].values
    ret_1d = float((close_yday[-1] - close_yday[0]) / close_yday[0]) if len(close_yday) > 1 else 0.0

    # 3-day return
    three_ago = bars[(bars.index >= (as_of - timedelta(days=3))) & (bars.index < as_of)]
    ret_3d = float((three_ago["close"].iloc[-1] - three_ago["close"].iloc[0]) / three_ago["close"].iloc[0]) if len(three_ago) > 1 else 0.0

    # Volatility regime
    rets_month = np.diff(np.log(month["close"].values + 1e-10))
    vol_month  = float(np.std(rets_month)) if len(rets_month) > 5 else 1e-4
    rets_yday  = np.diff(np.log(close_yday + 1e-10))
    vol_yday   = float(np.std(rets_yday)) if len(rets_yday) > 2 else vol_month
    vol_regime = float(vol_yday / (vol_month + 1e-10))

    trend_strength = float(abs(ret_3d) / (np.std(np.diff(np.log(close_all[-72:] + 1e-10))) * np.sqrt(72) + 1e-10))

    wt2 = yday["wt2"].dropna().values
    mfi = yday["rsi_mfi"].dropna().values
    rsi = yday["rsi"].dropna().values

    features = np.array([
        float(np.mean(wt2)) if len(wt2) > 0 else 0.0,
        float(np.min(wt2))  if len(wt2) > 0 else 0.0,
        float(np.max(wt2))  if len(wt2) > 0 else 0.0,
        float(np.mean(mfi > 0)) if len(mfi) > 0 else 0.5,
        float(np.mean(rsi)) if len(rsi) > 0 else 50.0,
        float(yday["wt_cross"].sum()) if "wt_cross" in yday.columns else 0.0,
        np.clip(vol_regime, 0, 5),
        np.clip(ret_1d, -0.2, 0.2),
        np.clip(ret_3d, -0.3, 0.3),
        np.clip(trend_strength, 0, 5),
    ], dtype=np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_hourly_ccxt(
    assets: list[str],
    start: str,
    end: str,
    timeframe: str = "1h",
    quote: str = "USDT",
    cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch hourly bars from Binance via CCXT. Caches to parquet."""
    import ccxt

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance({"enableRateLimit": True})
    results = {}

    for asset in assets:
        symbol = f"{asset}/{quote}"
        cache_key = f"mcb_{asset}_{timeframe}_{start}_{end}.parquet"
        cache_path = cache_dir / cache_key if cache_dir else None

        if cache_path and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                results[asset] = df
                logger.info(f"  Cache hit: {asset} {timeframe}")
                continue
            except Exception:
                pass

        logger.info(f"  Fetching {symbol} {timeframe} {start}→{end}")
        try:
            from datetime import timezone
            from datetime import datetime as dt
            since_ms = int(dt.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            end_ms   = int(dt.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

            all_bars = []
            cursor = since_ms
            while cursor < end_ms:
                bars = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
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
                logger.warning(f"  No data for {symbol}")
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

class MCBCPOStrategy:
    """
    Market Cipher B strategy suite as a CPO TradingStrategy.

    Each model is a (symbol, timeframe, strategy_id) combination.
    The RF learns which MCb strategy + parameter set works best
    each day given yesterday's indicator readings.

    Usage:
        python scripts/run_cpo.py --strategy mcb_ta \\
            --mcb-assets BTC,ETH,SOL --mcb-strategies anchor_trigger,zero_line_rejection \\
            phase2 --start 2024-01-01 --end 2024-12-31
    """

    def __init__(
        self,
        assets: list[str] | None = None,
        timeframes: list[str] | None = None,
        strategy_ids: list[str] | None = None,
        cache_dir: str | Path = "data/mcb_cache",
        tc_bps: float = 10.0,
        training_start: str = "2024-01-01",
        training_end: str = "2024-12-31",
        quote: str = "USDT",
    ):
        self.assets       = assets or DEFAULT_ASSETS
        self.timeframes   = timeframes or DEFAULT_TIMEFRAMES
        self.strategy_ids = strategy_ids or DEFAULT_STRATEGIES
        self.cache_dir    = Path(cache_dir)
        self.tc_bps       = tc_bps
        self.training_start = training_start
        self.training_end   = training_end
        self.quote        = quote

        # Build model universe: asset × timeframe × strategy_id
        self._models = []
        for asset, tf, sid in itertools.product(self.assets, self.timeframes, self.strategy_ids):
            self._models.append(MCBModelSpec(
                model_id=f"{asset}_{tf}_{sid}",
                asset=asset,
                timeframe=tf,
                strategy_id=sid,
            ))

        # Build param grids
        self._grids = generate_mcb_param_grid()
        self._all_configs = []
        for sid in self.strategy_ids:
            self._all_configs.extend(self._grids.get(sid, []))

        print(f"  MCB CPO: {len(self._models)} models "
              f"({len(self.assets)}×{len(self.timeframes)}×{len(self.strategy_ids)})")
        print(f"  Total configs: {len(self._all_configs)}")

    # ── Protocol ──────────────────────────────────────────────────

    def get_models(self) -> list[MCBModelSpec]:
        return self._models

    def get_param_grid(self) -> list[MCBConfig]:
        return self._all_configs

    def daily_feature_names(self) -> list[str]:
        return list(MCB_FEATURES)

    def config_param_names(self) -> list[str]:
        return MCBConfig.param_names()

    def config_to_features(self, config: MCBConfig) -> list[float]:
        return config.to_feature_vector()

    def compute_features(self, model: MCBModelSpec, as_of_date: str, data: dict) -> np.ndarray | None:
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None or len(bars) < 48:
            return None
        return _compute_mcb_features(bars, as_of_date)

    def run_single_day(self, model: MCBModelSpec, config: MCBConfig,
                       day: Any, data: dict) -> dict:
        """
        Execute one model+config for one 'day' in Phase 4.
        Uses the same 3-day evaluation window as training so execution
        matches the RF's learned conditional distribution.
        """
        if config.strategy_id != model.strategy_id:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        day_start = pd.Timestamp(day)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("UTC")

        # Use same 3-day window as training
        eval_hours = 3 * 24
        window_end = day_start + timedelta(hours=eval_hours)

        tf_hours = {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24}.get(model.timeframe, 1)
        warmup_bars = 100 * tf_hours

        bars_window = bars[(bars.index >= day_start) & (bars.index < window_end)]
        if len(bars_window) < int(eval_hours * 0.5):
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        hist_start = day_start - timedelta(hours=warmup_bars)
        bars_hist = bars[(bars.index >= hist_start) & (bars.index < day_start)]

        return run_mcb_single_day(bars_hist, bars_window, config, self.tc_bps)

    def run_model_year(self, model: MCBModelSpec, data: dict,
                       param_grid: list[MCBConfig],
                       eval_days: int = 3) -> pd.DataFrame:
        """
        Run all configs for all days. Called by cpo_core run_phase2.

        Uses a rolling eval_days-day window per evaluation. Anchor & Trigger
        needs 2-3 days to complete a cycle (anchor -> reset -> trigger -> exit).
        The RF question: "will this config profit over the next eval_days days?"
        """
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return pd.DataFrame()

        model_configs = [c for c in param_grid if c.strategy_id == model.strategy_id]
        if not model_configs:
            return pd.DataFrame()

        trading_days = sorted(bars.index.normalize().unique())
        tf_hours = {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24}.get(model.timeframe, 1)
        warmup_bars = 100 * tf_hours
        eval_hours = eval_days * 24

        results = []
        for day_idx, day in enumerate(trading_days):
            day_start = pd.Timestamp(day)
            if day_start.tzinfo is None:
                day_start = day_start.tz_localize("UTC")
            window_end = day_start + timedelta(hours=eval_hours)

            bars_window = bars[(bars.index >= day_start) & (bars.index < window_end)]
            if len(bars_window) < int(eval_hours * 0.5):
                continue

            hist_start = day_start - timedelta(hours=warmup_bars)
            bars_hist = bars[(bars.index >= hist_start) & (bars.index < day_start)]

            for config in model_configs:
                result = run_mcb_single_day(bars_hist, bars_window, config, self.tc_bps)
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
        tf = self.timeframes[0]  # use first timeframe (typically 1h)
        hourly = _fetch_hourly_ccxt(assets, self.training_start, self.training_end,
                                    tf, self.quote, self.cache_dir)
        return {"hourly_data": hourly}

    def fetch_oos_data(self, models, start, end) -> dict:
        assets = list({m.asset for m in models})
        tf = self.timeframes[0]
        hourly = _fetch_hourly_ccxt(assets, start, end,
                                    tf, self.quote, self.cache_dir)
        return {"hourly_data": hourly}

    def fetch_warmup_daily(self, models, start, end) -> dict[str, pd.Series]:
        """Fetch daily closes for Kelly warmup (aggregate from hourly)."""
        assets = list({m.asset for m in models})
        tf = self.timeframes[0]
        hourly = _fetch_hourly_ccxt(assets, start, end,
                                    tf, self.quote, self.cache_dir)
        daily = {}
        for asset, bars in hourly.items():
            daily[asset] = bars["close"].resample("D").last().dropna()
        return daily

    def get_trading_days(self, data: dict) -> list:
        """Return sorted list of trading days from OOS hourly data."""
        hourly = data.get("hourly_data", {})
        if not hourly:
            return []
        # Union of all days across all assets
        all_days = set()
        for bars in hourly.values():
            all_days.update(bars.index.normalize().unique())
        return sorted(all_days)

    def get_daily_prices(self, oos_data: dict, models) -> dict[str, pd.Series]:
        hourly = oos_data.get("hourly_data", {})
        return {asset: bars["close"].resample("D").last().dropna()
                for asset, bars in hourly.items()}

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
