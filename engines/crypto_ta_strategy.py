"""
Crypto TA Strategy — Implementation of CPO TradingStrategy protocol.

8 qualitatively distinct TA models × N crypto assets × parameter grids.
Each (asset, TA_model_type) is a "model" in the CPO framework.
Each parameter set is a "config". The RF learns which config works
best under which market conditions for each model.

TA Models:
    1. RSI Mean Reversion      — buy oversold, sell overbought
    2. MACD Crossover           — trend following via signal line cross
    3. Bollinger Band Reversion — buy lower band, sell upper band
    4. EMA Crossover            — fast/slow moving average cross
    5. Stochastic Oscillator    — momentum oscillator extremes
    6. ATR Breakout             — volatility-based breakout
    7. Volume Breakout          — price move confirmed by volume surge
    8. VWAP Reversion           — revert to volume-weighted average price

Uses hourly bars (not minute) for practical crypto execution.
Trading is 24/7 — each "day" is a UTC calendar day.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engines.cpo_core import ModelSpec, ConfigSpec

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

# Default crypto universe — high liquidity, diverse sectors
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE"]

# TA model types
TA_MODEL_TYPES = [
    "RSI", "MACD", "BOLL", "EMA_CROSS",
    "STOCH", "ATR_BREAK", "VOL_BREAK", "VWAP_REV",
]


@dataclass
class CryptoModelSpec(ModelSpec):
    """One (asset, TA_model_type) combination."""
    asset: str = ""          # e.g. "BTC"
    ta_type: str = ""        # e.g. "RSI"
    quote: str = "USD"       # quote currency

    @property
    def polygon_ticker(self) -> str:
        return f"X:{self.asset}{self.quote}"


@dataclass
class TAConfig(ConfigSpec):
    """
    Generic TA parameter configuration.

    All TA models share the same struct — unused params are 0.
    This avoids needing separate config classes per model type.
    """
    ta_type: str = ""       # which model these params belong to

    # Universal params
    period: int = 14        # primary lookback period (hours)
    threshold_lo: float = 30.0   # lower threshold (RSI oversold, Stoch oversold, etc.)
    threshold_hi: float = 70.0   # upper threshold (RSI overbought, etc.)
    hold_hours: int = 24    # max holding period before forced exit

    # Model-specific params (unused fields = 0 for other models)
    fast_period: int = 0    # MACD/EMA fast period
    slow_period: int = 0    # MACD/EMA slow period
    signal_period: int = 0  # MACD signal period
    multiplier: float = 0.0 # Bollinger std dev / ATR multiplier
    vol_multiplier: float = 0.0  # Volume breakout surge threshold

    # Normalization constants for RF features
    _NORMS = {
        "period": 50.0, "threshold_lo": 100.0, "threshold_hi": 100.0,
        "hold_hours": 48.0, "fast_period": 50.0, "slow_period": 100.0,
        "signal_period": 20.0, "multiplier": 4.0, "vol_multiplier": 4.0,
    }

    def to_feature_vector(self) -> list[float]:
        return [
            self.period / self._NORMS["period"],
            self.threshold_lo / self._NORMS["threshold_lo"],
            self.threshold_hi / self._NORMS["threshold_hi"],
            self.hold_hours / self._NORMS["hold_hours"],
            self.fast_period / self._NORMS["fast_period"],
            self.slow_period / self._NORMS["slow_period"],
            self.signal_period / self._NORMS["signal_period"],
            self.multiplier / self._NORMS["multiplier"],
            self.vol_multiplier / self._NORMS["vol_multiplier"],
        ]

    @staticmethod
    def param_names() -> list[str]:
        return [
            "period", "threshold_lo", "threshold_hi", "hold_hours",
            "fast_period", "slow_period", "signal_period",
            "multiplier", "vol_multiplier",
        ]


# Daily lagged features for crypto
CRYPTO_FEATURES = [
    "ret_24h",          # 24-hour return
    "ret_7d",           # 7-day return
    "vol_24h",          # 24-hour realized volatility (hourly returns)
    "vol_7d",           # 7-day realized volatility
    "vol_ratio",        # vol_24h / vol_7d (vol regime)
    "volume_ratio",     # 24h volume / 7d avg volume (activity surge)
    "rsi_14",           # 14-period RSI on hourly
    "trend_strength",   # abs(EMA20 - EMA50) / ATR14 (trend vs chop)
    "btc_corr_7d",      # 7-day hourly correlation with BTC
    "high_low_range",   # (24h high - 24h low) / close (intraday range)
]


# ═════════════════════════════════════════════════════════════════════════════
# PARAM GRID GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_crypto_param_grid() -> dict[str, list[TAConfig]]:
    """
    Generate parameter grids per TA model type.
    Returns {ta_type: [TAConfig, ...]}.
    """
    grids = {}
    cid = 0

    # 1. RSI Mean Reversion
    configs = []
    for period in [7, 14, 21]:
        for oversold in [20, 25, 30]:
            for overbought in [70, 75, 80]:
                for hold in [4, 12, 24]:
                    configs.append(TAConfig(
                        config_id=cid, ta_type="RSI",
                        period=period, threshold_lo=oversold,
                        threshold_hi=overbought, hold_hours=hold,
                    ))
                    cid += 1
    grids["RSI"] = configs

    # 2. MACD Crossover
    configs = []
    for fast in [8, 12, 16]:
        for slow in [21, 26, 34]:
            if fast >= slow:
                continue
            for signal in [7, 9, 12]:
                for hold in [4, 12, 24]:
                    configs.append(TAConfig(
                        config_id=cid, ta_type="MACD",
                        fast_period=fast, slow_period=slow,
                        signal_period=signal, hold_hours=hold,
                    ))
                    cid += 1
    grids["MACD"] = configs

    # 3. Bollinger Band Reversion
    configs = []
    for period in [14, 20, 30]:
        for num_std in [1.5, 2.0, 2.5]:
            for hold in [4, 12, 24]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="BOLL",
                    period=period, multiplier=num_std, hold_hours=hold,
                ))
                cid += 1
    grids["BOLL"] = configs

    # 4. EMA Crossover
    configs = []
    for fast in [5, 10, 20]:
        for slow in [20, 50, 100]:
            if fast >= slow:
                continue
            for hold in [4, 12, 24, 48]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="EMA_CROSS",
                    fast_period=fast, slow_period=slow, hold_hours=hold,
                ))
                cid += 1
    grids["EMA_CROSS"] = configs

    # 5. Stochastic Oscillator
    configs = []
    for k_period in [9, 14, 21]:
        for oversold in [20, 25]:
            for overbought in [75, 80]:
                for hold in [4, 12, 24]:
                    configs.append(TAConfig(
                        config_id=cid, ta_type="STOCH",
                        period=k_period, signal_period=3,
                        threshold_lo=oversold, threshold_hi=overbought,
                        hold_hours=hold,
                    ))
                    cid += 1
    grids["STOCH"] = configs

    # 6. ATR Breakout
    configs = []
    for period in [10, 14, 20]:
        for mult in [1.5, 2.0, 3.0]:
            for hold in [4, 12, 24]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="ATR_BREAK",
                    period=period, multiplier=mult, hold_hours=hold,
                ))
                cid += 1
    grids["ATR_BREAK"] = configs

    # 7. Volume Breakout
    configs = []
    for period in [10, 20, 30]:
        for vol_mult in [1.5, 2.0, 2.5]:
            for hold in [4, 12, 24]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="VOL_BREAK",
                    period=period, vol_multiplier=vol_mult, hold_hours=hold,
                ))
                cid += 1
    grids["VOL_BREAK"] = configs

    # 8. VWAP Reversion
    configs = []
    for period in [12, 24, 48]:
        for threshold in [0.5, 1.0, 1.5]:  # % deviation from VWAP
            for hold in [4, 12, 24]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="VWAP_REV",
                    period=period, threshold_lo=threshold,
                    threshold_hi=threshold, hold_hours=hold,
                ))
                cid += 1
    grids["VWAP_REV"] = configs

    return grids


# ═════════════════════════════════════════════════════════════════════════════
# TA SIGNAL ENGINES
# ═════════════════════════════════════════════════════════════════════════════

def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(close: np.ndarray, fast: int, slow: int, signal: int
          ) -> tuple[np.ndarray, np.ndarray]:
    """MACD line and signal line."""
    s = pd.Series(close)
    ema_fast = s.ewm(span=fast, adjust=False).mean().values
    ema_slow = s.ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    return macd_line, signal_line


def _bollinger(close: np.ndarray, period: int, num_std: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: (upper, middle, lower)."""
    s = pd.Series(close)
    middle = s.rolling(period, min_periods=period).mean().values
    std = s.rolling(period, min_periods=period).std().values
    return middle + num_std * std, middle, middle - num_std * std


def _stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                k_period: int, d_period: int) -> tuple[np.ndarray, np.ndarray]:
    """%K and %D stochastic."""
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    lowest = s_low.rolling(k_period, min_periods=k_period).min().values
    highest = s_high.rolling(k_period, min_periods=k_period).max().values
    k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d = pd.Series(k).rolling(d_period, min_periods=1).mean().values
    return k, d


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int) -> np.ndarray:
    """Average True Range."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


def _vwap(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume-Weighted Average Price (rolling)."""
    s_cv = pd.Series(close * volume)
    s_v = pd.Series(volume)
    cum_cv = s_cv.rolling(period, min_periods=1).sum().values
    cum_v = s_v.rolling(period, min_periods=1).sum().values
    return cum_cv / (cum_v + 1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# INTRADAY EXECUTION — run one TA config for one day
# ═════════════════════════════════════════════════════════════════════════════

def run_ta_single_day(bars_day: pd.DataFrame, config: TAConfig,
                      bars_history: pd.DataFrame | None = None,
                      tc_bps: float = 2.0,
                      ) -> dict:
    """
    Run one TA config on one day (24h UTC) of hourly bars.

    bars_day: hourly OHLCV for the trading day
    bars_history: recent history for indicator warmup
    tc_bps: one-way transaction cost in bps (single asset)

    Returns: {daily_return, gross_return, n_trades}
    """
    if bars_history is not None and not bars_history.empty:
        full = pd.concat([bars_history, bars_day])
    else:
        full = bars_day

    close = full["close"].values
    high = full["high"].values
    low = full["low"].values
    volume = full["volume"].values

    if len(close) < 30:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    # Compute signals based on TA type
    signals = _compute_signals(config.ta_type, close, high, low, volume, config)

    # Only trade during today's window
    n_hist = len(full) - len(bars_day)
    signals_today = signals[n_hist:]
    close_today = close[n_hist:]

    if len(signals_today) < 2:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    # Trading simulation
    tc_rate = 2 * tc_bps / 10000.0  # 2 executions (entry + exit) per round trip
    position = 0
    entry_price = 0.0
    entry_idx = 0
    pnl = 0.0
    gross_pnl = 0.0
    trades = 0

    for i in range(len(signals_today)):
        sig = signals_today[i]
        price = close_today[i]

        if np.isnan(sig) or np.isnan(price):
            continue

        # Check holding period timeout
        if position != 0 and (i - entry_idx) >= config.hold_hours:
            gross = (price / entry_price - 1) * position
            cost = tc_rate
            pnl += gross - cost
            gross_pnl += gross
            trades += 1
            position = 0

        # Entry/exit
        if position == 0:
            if sig > 0:  # buy signal
                position = 1
                entry_price = price
                entry_idx = i
            elif sig < 0:  # sell/short signal
                position = -1
                entry_price = price
                entry_idx = i
        elif position == 1:
            if sig < 0:  # exit long / reverse
                gross = price / entry_price - 1
                cost = tc_rate
                pnl += gross - cost
                gross_pnl += gross
                trades += 1
                position = -1
                entry_price = price
                entry_idx = i
        elif position == -1:
            if sig > 0:  # exit short / reverse
                gross = 1 - price / entry_price
                cost = tc_rate
                pnl += gross - cost
                gross_pnl += gross
                trades += 1
                position = 1
                entry_price = price
                entry_idx = i

    # Force close at end of day
    if position != 0:
        price = close_today[-1]
        if position == 1:
            gross = price / entry_price - 1
        else:
            gross = 1 - price / entry_price
        cost = tc_rate
        pnl += gross - cost
        gross_pnl += gross
        trades += 1

    return {
        "daily_return": pnl,
        "gross_return": gross_pnl,
        "n_trades": trades,
    }


def _compute_signals(ta_type: str, close: np.ndarray, high: np.ndarray,
                     low: np.ndarray, volume: np.ndarray,
                     config: TAConfig) -> np.ndarray:
    """
    Compute trading signals for a given TA model type.
    Returns array: +1 = buy, -1 = sell, 0 = hold.
    """
    n = len(close)
    signals = np.zeros(n)

    if ta_type == "RSI":
        rsi = _rsi(close, config.period)
        for i in range(1, n):
            if rsi[i] < config.threshold_lo:
                signals[i] = 1  # oversold → buy
            elif rsi[i] > config.threshold_hi:
                signals[i] = -1  # overbought → sell

    elif ta_type == "MACD":
        macd_line, signal_line = _macd(close, config.fast_period,
                                       config.slow_period, config.signal_period)
        for i in range(1, n):
            if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                signals[i] = 1  # bullish crossover
            elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                signals[i] = -1  # bearish crossover

    elif ta_type == "BOLL":
        upper, middle, lower = _bollinger(close, config.period, config.multiplier)
        for i in range(1, n):
            if not np.isnan(lower[i]) and close[i] < lower[i]:
                signals[i] = 1  # below lower band → buy
            elif not np.isnan(upper[i]) and close[i] > upper[i]:
                signals[i] = -1  # above upper band → sell

    elif ta_type == "EMA_CROSS":
        s = pd.Series(close)
        ema_fast = s.ewm(span=config.fast_period, adjust=False).mean().values
        ema_slow = s.ewm(span=config.slow_period, adjust=False).mean().values
        for i in range(1, n):
            if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                signals[i] = 1
            elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                signals[i] = -1

    elif ta_type == "STOCH":
        k, d = _stochastic(high, low, close, config.period, config.signal_period or 3)
        for i in range(1, n):
            if not np.isnan(k[i]):
                if k[i] < config.threshold_lo and k[i-1] >= config.threshold_lo:
                    signals[i] = 1
                elif k[i] > config.threshold_hi and k[i-1] <= config.threshold_hi:
                    signals[i] = -1

    elif ta_type == "ATR_BREAK":
        atr = _atr(high, low, close, config.period)
        s = pd.Series(close)
        ma = s.rolling(config.period, min_periods=1).mean().values
        for i in range(1, n):
            if not np.isnan(atr[i]):
                if close[i] > ma[i] + config.multiplier * atr[i]:
                    signals[i] = 1  # breakout up
                elif close[i] < ma[i] - config.multiplier * atr[i]:
                    signals[i] = -1  # breakout down

    elif ta_type == "VOL_BREAK":
        s = pd.Series(close)
        ma = s.rolling(config.period, min_periods=1).mean().values
        avg_vol = pd.Series(volume).rolling(config.period, min_periods=1).mean().values
        for i in range(1, n):
            if not np.isnan(avg_vol[i]) and avg_vol[i] > 0:
                vol_surge = volume[i] / avg_vol[i]
                if vol_surge > config.vol_multiplier:
                    if close[i] > ma[i]:
                        signals[i] = 1  # up move + volume
                    elif close[i] < ma[i]:
                        signals[i] = -1  # down move + volume

    elif ta_type == "VWAP_REV":
        vwap = _vwap(close, volume, config.period)
        for i in range(1, n):
            if not np.isnan(vwap[i]) and vwap[i] > 0:
                dev_pct = (close[i] - vwap[i]) / vwap[i] * 100
                if dev_pct < -config.threshold_lo:
                    signals[i] = 1  # below VWAP → buy
                elif dev_pct > config.threshold_hi:
                    signals[i] = -1  # above VWAP → sell

    return signals


# ═════════════════════════════════════════════════════════════════════════════
# CRYPTO STRATEGY CLASS
# ═════════════════════════════════════════════════════════════════════════════

class CryptoTAStrategy:
    """
    Crypto TA strategy implementing the CPO TradingStrategy protocol.

    Each model = (asset, TA_type). Each config = parameter set for that TA type.
    RF learns: given yesterday's market conditions, which TA model + params
    have the highest probability of profiting today.
    """

    def __init__(self, api_key: str = "",
                 assets: list[str] | None = None,
                 ta_types: list[str] | None = None,
                 cache_dir: str | Path = "data/crypto_cache",
                 tc_bps: float = 2.0,
                 training_start: str = "2025-01-01",
                 training_end: str = "2025-12-31",
                 quote: str = "USDT",
                 exchange_id: str = "binance"):
        self.api_key = api_key  # not needed for CCXT public data
        self.assets = assets or DEFAULT_ASSETS
        self.ta_types = ta_types or TA_MODEL_TYPES
        self.cache_dir = Path(cache_dir)
        self.tc_bps = tc_bps
        self.training_start = training_start
        self.training_end = training_end
        self.quote = quote
        self.exchange_id = exchange_id

        # Build model universe
        self._models = []
        for asset in self.assets:
            for ta_type in self.ta_types:
                self._models.append(CryptoModelSpec(
                    model_id=f"{asset}_{ta_type}",
                    asset=asset, ta_type=ta_type, quote=quote,
                ))

        # Build param grids per ta_type
        self._grids_by_type = generate_crypto_param_grid()

        # Flatten to single config list (for CPO core)
        self._all_configs = []
        for ta_type in self.ta_types:
            self._all_configs.extend(self._grids_by_type.get(ta_type, []))

        print(f"  Crypto TA: {len(self._models)} models "
              f"({len(self.assets)} assets × {len(self.ta_types)} TA types)")
        print(f"  Configs per type: {', '.join(f'{t}={len(c)}' for t, c in self._grids_by_type.items())}")
        print(f"  Total configs: {len(self._all_configs)}")

    # ── Protocol implementation ───────────────────────────────────

    def get_models(self) -> list[CryptoModelSpec]:
        return self._models

    def get_param_grid(self) -> list[TAConfig]:
        return self._all_configs

    @staticmethod
    def model_group_fn(model_id: str) -> str:
        """Map model_id → TA type for group-level pre-filtering.
        E.g. 'BTC_MACD' → 'MACD', 'ETH_ATR_BREAK' → 'ATR_BREAK'."""
        return model_id.split("_", 1)[1]

    def daily_feature_names(self) -> list[str]:
        return list(CRYPTO_FEATURES)

    def config_param_names(self) -> list[str]:
        return TAConfig.param_names()

    def config_to_features(self, config: TAConfig) -> list[float]:
        return config.to_feature_vector()

    def compute_features(self, model: CryptoModelSpec, as_of_date: str,
                         data: dict) -> np.ndarray | None:
        """Compute crypto daily features."""
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None or len(bars) < 168:  # need 7 days of hourly
            return None

        # BTC bars for correlation
        btc_bars = hourly.get("BTC")

        return _compute_crypto_features(bars, btc_bars, as_of_date, model.asset)

    def run_single_day(self, model: CryptoModelSpec, config: TAConfig,
                       day: Any, data: dict) -> dict:
        """Execute one model+config for one day."""
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        # Get today's bars
        day_start = pd.Timestamp(day)
        day_end = day_start + timedelta(hours=24)
        bars_day = bars[(bars.index >= day_start) & (bars.index < day_end)]

        if len(bars_day) < 6:  # need at least 6 hourly bars
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        # History for indicator warmup (7 days)
        hist_start = day_start - timedelta(days=7)
        bars_hist = bars[(bars.index >= hist_start) & (bars.index < day_start)]

        return run_ta_single_day(bars_day, config, bars_hist, self.tc_bps)

    def run_model_year(self, model: CryptoModelSpec, data: dict,
                       param_grid: list[TAConfig]) -> pd.DataFrame:
        """Run all configs for all days for one model."""
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return pd.DataFrame()

        # Only use configs for this model's TA type
        model_configs = [c for c in param_grid if c.ta_type == model.ta_type]
        if not model_configs:
            return pd.DataFrame()

        # Get unique trading days
        trading_days = sorted(bars.index.normalize().unique())

        results = []
        for day_idx, day in enumerate(trading_days):
            day_start = pd.Timestamp(day)
            day_end = day_start + timedelta(hours=24)
            bars_day = bars[(bars.index >= day_start) & (bars.index < day_end)]

            if len(bars_day) < 6:
                continue

            hist_start = day_start - timedelta(days=7)
            bars_hist = bars[(bars.index >= hist_start) & (bars.index < day_start)]

            for config in model_configs:
                result = run_ta_single_day(bars_day, config, bars_hist, self.tc_bps)
                results.append({
                    "model_id": model.model_id,
                    "date": day.strftime("%Y-%m-%d"),
                    "config_id": config.config_id,
                    "daily_return": result["daily_return"],
                    "gross_return": result["gross_return"],
                    "n_trades": result["n_trades"],
                })

            if (day_idx + 1) % 50 == 0:
                print(f"    {model.model_id}: {day_idx+1}/{len(trading_days)} days")

        return pd.DataFrame(results)

    def fetch_training_data(self, models, start, end) -> dict:
        hourly = _fetch_all_hourly(
            self.assets, self.training_start, self.training_end,
            self.api_key, self.cache_dir, self.quote, self.exchange_id,
        )
        return {"hourly_data": hourly}

    def fetch_oos_data(self, models, start, end) -> dict:
        hourly = _fetch_all_hourly(
            self.assets, start, end,
            self.api_key, self.cache_dir, self.quote, self.exchange_id,
        )
        return {"hourly_data": hourly}

    def fetch_warmup_daily(self, models, start, end) -> dict[str, pd.Series]:
        """Fetch daily close for warmup (aggregate from hourly)."""
        hourly = _fetch_all_hourly(
            self.assets, start, end,
            self.api_key, self.cache_dir, self.quote, self.exchange_id,
        )
        daily = {}
        for asset, bars in hourly.items():
            d = bars["close"].resample("D").last().dropna()
            d.index = d.index.date
            daily[asset] = d
        return daily

    def get_daily_prices(self, data, models) -> dict[str, pd.Series]:
        hourly = data.get("hourly_data", {})
        daily = {}
        for asset, bars in hourly.items():
            d = bars["close"].resample("D").last().dropna()
            d.index = d.index.date
            daily[asset] = d
        return daily

    def get_trading_days(self, data) -> list:
        hourly = data.get("hourly_data", {})
        if not hourly:
            return []
        # Use first asset's data to define trading days
        first_bars = next(iter(hourly.values()))
        days = sorted(first_bars.index.normalize().unique())
        # Only keep days with 12+ hourly bars
        trading_days = []
        for day in days:
            day_start = pd.Timestamp(day)
            day_end = day_start + timedelta(hours=24)
            n = len(first_bars[(first_bars.index >= day_start) & (first_bars.index < day_end)])
            if n >= 12:
                trading_days.append(day)
        return trading_days

    def prepare_warmup(self, daily_prices, warmup_daily):
        for asset in daily_prices:
            if asset in warmup_daily:
                warmup = warmup_daily[asset]
                oos = daily_prices[asset]
                warmup_only = warmup[warmup.index < oos.index.min()]
                daily_prices[asset] = pd.concat([warmup_only, oos])
        return daily_prices


# ═════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_hourly_ccxt(asset: str, start: str, end: str,
                       cache_dir: Path, quote: str = "USDT",
                       exchange_id: str = "binance") -> pd.DataFrame:
    """
    Fetch hourly bars via CCXT (Binance by default).
    No API key needed for historical OHLCV.
    Caches to parquet per asset.
    """
    import ccxt

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{asset}_{quote}_{start}_{end}_1h_{exchange_id}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    symbol = f"{asset}/{quote}"

    # Verify symbol exists
    try:
        exchange.load_markets()
        if symbol not in exchange.markets:
            logger.warning(f"{symbol} not available on {exchange_id}")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to load markets from {exchange_id}: {e}")
        return pd.DataFrame()

    # Fetch in chunks (CCXT returns max ~1000 bars per call)
    all_bars = []
    since = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    limit = 1000  # max per request on most exchanges

    while since < end_ms:
        try:
            bars = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=limit)
            if not bars:
                break
            all_bars.extend(bars)
            # Move forward past last bar
            since = bars[-1][0] + 3600000  # +1 hour in ms
            if len(bars) < limit:
                break  # no more data
            time.sleep(exchange.rateLimit / 1000)  # respect rate limit
        except Exception as e:
            logger.warning(f"CCXT fetch error for {symbol}: {e}")
            break

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")]

    # Filter to requested range
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    if not df.empty:
        df.to_parquet(cache_path)

    return df


def _fetch_all_hourly(assets: list[str], start: str, end: str,
                      api_key: str, cache_dir: Path,
                      quote: str = "USDT",
                      exchange_id: str = "binance") -> dict[str, pd.DataFrame]:
    """Fetch hourly data for all assets via CCXT."""
    print(f"  Fetching hourly data for {len(assets)} assets: {start} -> {end}")
    print(f"  Exchange: {exchange_id}, quote: {quote}")
    data = {}
    for i, asset in enumerate(assets):
        df = _fetch_hourly_ccxt(asset, start, end, cache_dir, quote, exchange_id)
        if not df.empty:
            data[asset] = df
        if (i + 1) % 4 == 0 or i == len(assets) - 1:
            n_bars = len(df) if not df.empty else 0
            print(f"    {i+1}/{len(assets)}: {asset} ({n_bars} bars)")
    print(f"  Loaded {len(data)}/{len(assets)} assets")
    return data


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def _compute_crypto_features(bars: pd.DataFrame, btc_bars: pd.DataFrame | None,
                             as_of_date: str, asset: str) -> np.ndarray | None:
    """Compute daily features for one crypto asset as of a given date."""
    dt = pd.Timestamp(as_of_date)
    hist = bars[bars.index < dt]

    if len(hist) < 168:  # need 7 days of hourly
        return None

    close = hist["close"].values
    volume = hist["volume"].values
    high = hist["high"].values
    low = hist["low"].values

    # Returns
    ret_24h = close[-1] / close[-24] - 1 if len(close) >= 24 else 0
    ret_7d = close[-1] / close[-168] - 1 if len(close) >= 168 else 0

    # Volatility (hourly log returns)
    log_ret = np.diff(np.log(close + 1e-10))
    vol_24h = np.std(log_ret[-24:]) if len(log_ret) >= 24 else 0
    vol_7d = np.std(log_ret[-168:]) if len(log_ret) >= 168 else vol_24h
    vol_ratio = vol_24h / (vol_7d + 1e-10)

    # Volume ratio
    vol_24h_sum = np.sum(volume[-24:]) if len(volume) >= 24 else 0
    vol_7d_avg = np.mean([np.sum(volume[max(0,i-24):i])
                          for i in range(max(24, len(volume)-168), len(volume), 24)])
    volume_ratio = vol_24h_sum / (vol_7d_avg + 1e-10)

    # RSI
    rsi_vals = _rsi(close, 14)
    rsi_14 = rsi_vals[-1] if len(rsi_vals) > 0 else 50

    # Trend strength: |EMA20 - EMA50| / ATR14
    s = pd.Series(close)
    ema20 = s.ewm(span=20, adjust=False).mean().values
    ema50 = s.ewm(span=50, adjust=False).mean().values
    atr14 = _atr(high, low, close, 14)
    if len(atr14) > 0 and atr14[-1] > 0:
        trend_strength = abs(ema20[-1] - ema50[-1]) / atr14[-1]
    else:
        trend_strength = 0

    # BTC correlation (7-day hourly returns)
    btc_corr = 0.0
    if btc_bars is not None and asset != "BTC":
        btc_hist = btc_bars[btc_bars.index < dt]
        if len(btc_hist) >= 168:
            btc_ret = np.diff(np.log(btc_hist["close"].values[-168:] + 1e-10))
            asset_ret = log_ret[-167:] if len(log_ret) >= 167 else log_ret
            min_len = min(len(btc_ret), len(asset_ret))
            if min_len > 10:
                btc_corr = np.corrcoef(btc_ret[-min_len:], asset_ret[-min_len:])[0, 1]
    elif asset == "BTC":
        btc_corr = 1.0

    # High-low range
    h24 = np.max(high[-24:]) if len(high) >= 24 else high[-1]
    l24 = np.min(low[-24:]) if len(low) >= 24 else low[-1]
    high_low_range = (h24 - l24) / (close[-1] + 1e-10)

    return np.array([
        float(np.nan_to_num(ret_24h)),
        float(np.nan_to_num(ret_7d)),
        float(np.nan_to_num(vol_24h)),
        float(np.nan_to_num(vol_7d)),
        float(np.nan_to_num(vol_ratio)),
        float(np.nan_to_num(volume_ratio)),
        float(np.nan_to_num(rsi_14 / 100.0)),  # normalize to [0,1]
        float(np.nan_to_num(trend_strength)),
        float(np.nan_to_num(btc_corr)),
        float(np.nan_to_num(high_low_range)),
    ])
