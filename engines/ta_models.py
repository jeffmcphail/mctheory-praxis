"""
Shared TA Models — Signals, parameter grids, features, and execution.

Asset-class agnostic. Used by crypto_ta_strategy, futures_ta_strategy,
fx_ta_strategy, or any future TA-based strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from engines.cpo_core import ModelSpec, ConfigSpec


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

TA_MODEL_TYPES = [
    "RSI", "MACD", "BOLL", "EMA_CROSS",
    "STOCH", "ATR_BREAK", "VOL_BREAK", "VWAP_REV",
]


@dataclass
class TAModelSpec(ModelSpec):
    """One (asset, TA_model_type) combination."""
    asset: str = ""
    ta_type: str = ""
    ticker: str = ""  # actual ticker for data fetch (e.g. "ES=F", "X:BTCUSD")


@dataclass
class TAConfig(ConfigSpec):
    """Generic TA parameter configuration (shared across all asset classes)."""
    ta_type: str = ""
    period: int = 14
    threshold_lo: float = 30.0
    threshold_hi: float = 70.0
    hold_hours: int = 24
    fast_period: int = 0
    slow_period: int = 0
    signal_period: int = 0
    multiplier: float = 0.0
    vol_multiplier: float = 0.0

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


# Daily lagged features (universal across asset classes)
TA_FEATURES = [
    "ret_24h", "ret_7d",
    "vol_24h", "vol_7d", "vol_ratio",
    "volume_ratio",
    "rsi_14", "trend_strength",
    "benchmark_corr_7d",  # corr with benchmark (BTC for crypto, SPY for futures, DXY for FX)
    "high_low_range",
]


# ═════════════════════════════════════════════════════════════════════════════
# PARAM GRID GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_ta_param_grid() -> dict[str, list[TAConfig]]:
    """Generate parameter grids per TA model type. Returns {ta_type: [TAConfig]}."""
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
                        threshold_hi=overbought, hold_hours=hold))
                    cid += 1
    grids["RSI"] = configs

    # 2. MACD Crossover
    configs = []
    for fast in [8, 12, 16]:
        for slow in [21, 26, 34]:
            if fast >= slow: continue
            for signal in [7, 9, 12]:
                for hold in [4, 12, 24]:
                    configs.append(TAConfig(
                        config_id=cid, ta_type="MACD",
                        fast_period=fast, slow_period=slow,
                        signal_period=signal, hold_hours=hold))
                    cid += 1
    grids["MACD"] = configs

    # 3. Bollinger Band Reversion
    configs = []
    for period in [14, 20, 30]:
        for num_std in [1.5, 2.0, 2.5]:
            for hold in [4, 12, 24]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="BOLL",
                    period=period, multiplier=num_std, hold_hours=hold))
                cid += 1
    grids["BOLL"] = configs

    # 4. EMA Crossover
    configs = []
    for fast in [5, 10, 20]:
        for slow in [20, 50, 100]:
            if fast >= slow: continue
            for hold in [4, 12, 24, 48]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="EMA_CROSS",
                    fast_period=fast, slow_period=slow, hold_hours=hold))
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
                        hold_hours=hold))
                    cid += 1
    grids["STOCH"] = configs

    # 6. ATR Breakout
    configs = []
    for period in [10, 14, 20]:
        for mult in [1.5, 2.0, 3.0]:
            for hold in [4, 12, 24]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="ATR_BREAK",
                    period=period, multiplier=mult, hold_hours=hold))
                cid += 1
    grids["ATR_BREAK"] = configs

    # 7. Volume Breakout
    configs = []
    for period in [10, 20, 30]:
        for vol_mult in [1.5, 2.0, 2.5]:
            for hold in [4, 12, 24]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="VOL_BREAK",
                    period=period, vol_multiplier=vol_mult, hold_hours=hold))
                cid += 1
    grids["VOL_BREAK"] = configs

    # 8. VWAP Reversion
    configs = []
    for period in [12, 24, 48]:
        for threshold in [0.5, 1.0, 1.5]:
            for hold in [4, 12, 24]:
                configs.append(TAConfig(
                    config_id=cid, ta_type="VWAP_REV",
                    period=period, threshold_lo=threshold,
                    threshold_hi=threshold, hold_hours=hold))
                cid += 1
    grids["VWAP_REV"] = configs

    return grids


# ═════════════════════════════════════════════════════════════════════════════
# TA INDICATOR FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def macd(close: np.ndarray, fast: int, slow: int, signal: int):
    s = pd.Series(close)
    ema_fast = s.ewm(span=fast, adjust=False).mean().values
    ema_slow = s.ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    return macd_line, signal_line


def bollinger(close: np.ndarray, period: int, num_std: float):
    s = pd.Series(close)
    middle = s.rolling(period, min_periods=period).mean().values
    std = s.rolling(period, min_periods=period).std().values
    return middle + num_std * std, middle, middle - num_std * std


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               k_period: int, d_period: int):
    lowest = pd.Series(low).rolling(k_period, min_periods=k_period).min().values
    highest = pd.Series(high).rolling(k_period, min_periods=k_period).max().values
    k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d = pd.Series(k).rolling(d_period, min_periods=1).mean().values
    return k, d


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


def vwap(close: np.ndarray, volume: np.ndarray, period: int):
    cum_cv = pd.Series(close * volume).rolling(period, min_periods=1).sum().values
    cum_v = pd.Series(volume).rolling(period, min_periods=1).sum().values
    return cum_cv / (cum_v + 1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_signals(ta_type: str, close: np.ndarray, high: np.ndarray,
                    low: np.ndarray, volume: np.ndarray,
                    config: TAConfig) -> np.ndarray:
    """Compute trading signals. +1 = buy, -1 = sell, 0 = hold."""
    n = len(close)
    signals = np.zeros(n)

    if ta_type == "RSI":
        rsi_vals = rsi(close, config.period)
        for i in range(1, n):
            if rsi_vals[i] < config.threshold_lo:
                signals[i] = 1
            elif rsi_vals[i] > config.threshold_hi:
                signals[i] = -1

    elif ta_type == "MACD":
        macd_line, signal_line = macd(close, config.fast_period,
                                      config.slow_period, config.signal_period)
        for i in range(1, n):
            if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                signals[i] = 1
            elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                signals[i] = -1

    elif ta_type == "BOLL":
        upper, middle, lower = bollinger(close, config.period, config.multiplier)
        for i in range(1, n):
            if not np.isnan(lower[i]) and close[i] < lower[i]:
                signals[i] = 1
            elif not np.isnan(upper[i]) and close[i] > upper[i]:
                signals[i] = -1

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
        k, d = stochastic(high, low, close, config.period, config.signal_period or 3)
        for i in range(1, n):
            if not np.isnan(k[i]):
                if k[i] < config.threshold_lo and k[i-1] >= config.threshold_lo:
                    signals[i] = 1
                elif k[i] > config.threshold_hi and k[i-1] <= config.threshold_hi:
                    signals[i] = -1

    elif ta_type == "ATR_BREAK":
        atr_vals = atr(high, low, close, config.period)
        ma = pd.Series(close).rolling(config.period, min_periods=1).mean().values
        for i in range(1, n):
            if not np.isnan(atr_vals[i]):
                if close[i] > ma[i] + config.multiplier * atr_vals[i]:
                    signals[i] = 1
                elif close[i] < ma[i] - config.multiplier * atr_vals[i]:
                    signals[i] = -1

    elif ta_type == "VOL_BREAK":
        ma = pd.Series(close).rolling(config.period, min_periods=1).mean().values
        avg_vol = pd.Series(volume).rolling(config.period, min_periods=1).mean().values
        for i in range(1, n):
            if not np.isnan(avg_vol[i]) and avg_vol[i] > 0:
                vol_surge = volume[i] / avg_vol[i]
                if vol_surge > config.vol_multiplier:
                    signals[i] = 1 if close[i] > ma[i] else -1

    elif ta_type == "VWAP_REV":
        vwap_vals = vwap(close, volume, config.period)
        for i in range(1, n):
            if not np.isnan(vwap_vals[i]) and vwap_vals[i] > 0:
                dev_pct = (close[i] - vwap_vals[i]) / vwap_vals[i] * 100
                if dev_pct < -config.threshold_lo:
                    signals[i] = 1
                elif dev_pct > config.threshold_hi:
                    signals[i] = -1

    return signals


# ═════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def run_ta_single_day(bars_day: pd.DataFrame, config: TAConfig,
                      bars_history: pd.DataFrame | None = None,
                      tc_bps: float = 2.0) -> dict:
    """Run one TA config on one day of hourly bars. Asset-class agnostic."""
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

    signals = compute_signals(config.ta_type, close, high, low, volume, config)

    n_hist = len(full) - len(bars_day)
    signals_today = signals[n_hist:]
    close_today = close[n_hist:]

    if len(signals_today) < 2:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    tc_rate = 2 * tc_bps / 10000.0
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

        if position != 0 and (i - entry_idx) >= config.hold_hours:
            gross = (price / entry_price - 1) * position
            pnl += gross - tc_rate
            gross_pnl += gross
            trades += 1
            position = 0

        if position == 0:
            if sig > 0:
                position = 1; entry_price = price; entry_idx = i
            elif sig < 0:
                position = -1; entry_price = price; entry_idx = i
        elif position == 1 and sig < 0:
            gross = price / entry_price - 1
            pnl += gross - tc_rate; gross_pnl += gross; trades += 1
            position = -1; entry_price = price; entry_idx = i
        elif position == -1 and sig > 0:
            gross = 1 - price / entry_price
            pnl += gross - tc_rate; gross_pnl += gross; trades += 1
            position = 1; entry_price = price; entry_idx = i

    if position != 0:
        price = close_today[-1]
        gross = (price / entry_price - 1) if position == 1 else (1 - price / entry_price)
        pnl += gross - tc_rate; gross_pnl += gross; trades += 1

    return {"daily_return": pnl, "gross_return": gross_pnl, "n_trades": trades}


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_ta_features(bars: pd.DataFrame, benchmark_bars: pd.DataFrame | None,
                        as_of_date: str) -> np.ndarray | None:
    """Compute daily features for one asset as of a given date. Asset-class agnostic."""
    dt = pd.Timestamp(as_of_date)
    hist = bars[bars.index < dt]

    if len(hist) < 168:
        return None

    close = hist["close"].values
    volume = hist["volume"].values
    high = hist["high"].values
    low = hist["low"].values

    ret_24h = close[-1] / close[-24] - 1 if len(close) >= 24 else 0
    ret_7d = close[-1] / close[-168] - 1 if len(close) >= 168 else 0

    log_ret = np.diff(np.log(close + 1e-10))
    vol_24h = np.std(log_ret[-24:]) if len(log_ret) >= 24 else 0
    vol_7d = np.std(log_ret[-168:]) if len(log_ret) >= 168 else vol_24h
    vol_ratio = vol_24h / (vol_7d + 1e-10)

    vol_24h_sum = np.sum(volume[-24:]) if len(volume) >= 24 else 0
    vol_7d_avg = np.mean([np.sum(volume[max(0, i-24):i])
                          for i in range(max(24, len(volume)-168), len(volume), 24)])
    volume_ratio = vol_24h_sum / (vol_7d_avg + 1e-10)

    rsi_vals = rsi(close, 14)
    rsi_14 = rsi_vals[-1] if len(rsi_vals) > 0 else 50

    s = pd.Series(close)
    ema20 = s.ewm(span=20, adjust=False).mean().values
    ema50 = s.ewm(span=50, adjust=False).mean().values
    atr14 = atr(high, low, close, 14)
    trend_strength = abs(ema20[-1] - ema50[-1]) / atr14[-1] if atr14[-1] > 0 else 0

    # Benchmark correlation
    bench_corr = 0.0
    if benchmark_bars is not None:
        bench_hist = benchmark_bars[benchmark_bars.index < dt]
        if len(bench_hist) >= 168:
            bench_ret = np.diff(np.log(bench_hist["close"].values[-168:] + 1e-10))
            asset_ret = log_ret[-167:] if len(log_ret) >= 167 else log_ret
            min_len = min(len(bench_ret), len(asset_ret))
            if min_len > 10:
                bench_corr = np.corrcoef(bench_ret[-min_len:], asset_ret[-min_len:])[0, 1]

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
        float(np.nan_to_num(rsi_14 / 100.0)),
        float(np.nan_to_num(trend_strength)),
        float(np.nan_to_num(bench_corr)),
        float(np.nan_to_num(high_low_range)),
    ])
