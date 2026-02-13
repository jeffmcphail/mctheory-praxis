"""
Technical Indicators (Phase 3.3).

Pure numpy implementations matching the original analysis() outputs:
  1. zscore     — pandas_ta.zscore
  2. mfi        — talib.MFI (Money Flow Index)
  3. force_index — ta.volume.ForceIndexIndicator
  4. donchian   — ta.volatility.DonchianChannel (width band)
  5. atr        — ta.volatility.average_true_range
  6. awesome_oscillator — ta.momentum.AwesomeOscillatorIndicator
  7. adx        — ta.trend.ADXIndicator

All take numpy arrays, return numpy arrays.
Validation criterion: within 1e-6 of original on test data.
"""

from __future__ import annotations

import numpy as np


def zscore(close: np.ndarray, length: int) -> np.ndarray:
    """
    Z-Score: (price - SMA) / rolling_std.
    Matches pandas_ta.zscore(close, length=N).
    """
    result = np.full(len(close), np.nan)
    for i in range(length - 1, len(close)):
        window = close[i - length + 1:i + 1]
        mu = np.mean(window)
        sigma = np.std(window, ddof=1)
        result[i] = (close[i] - mu) / sigma if sigma > 0 else 0.0
    return result


def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        volume: np.ndarray, length: int) -> np.ndarray:
    """
    Money Flow Index.
    Matches talib.MFI(high, low, close, volume, timeperiod=length).

    MFI = 100 - 100 / (1 + positive_flow_sum / negative_flow_sum)
    Typical price = (high + low + close) / 3
    Money flow = typical_price * volume
    """
    n = len(close)
    typical = (high + low + close) / 3.0
    money_flow = typical * volume

    result = np.full(n, np.nan)

    for i in range(length, n):
        pos_flow = 0.0
        neg_flow = 0.0
        for j in range(i - length + 1, i + 1):
            if typical[j] > typical[j - 1]:
                pos_flow += money_flow[j]
            elif typical[j] < typical[j - 1]:
                neg_flow += money_flow[j]

        if neg_flow == 0:
            result[i] = 100.0
        else:
            ratio = pos_flow / neg_flow
            result[i] = 100.0 - 100.0 / (1.0 + ratio)

    return result


def force_index(close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """
    Force Index.
    Matches ta.volume.ForceIndexIndicator(close, volume, window=N).force_index().

    FI = EMA(close_change * volume, window)
    """
    n = len(close)
    raw = np.zeros(n)
    raw[1:] = (close[1:] - close[:-1]) * volume[1:]

    # EMA of raw force index
    return _ema(raw, window)


def donchian_width(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   window: int) -> np.ndarray:
    """
    Donchian Channel Width Band.
    Matches ta.volatility.DonchianChannel(...).donchian_channel_wband().

    Width = (upper - lower) / mid
    Upper = max(high, window), Lower = min(low, window)
    Mid = (upper + lower) / 2
    """
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        upper = np.max(high[i - window + 1:i + 1])
        lower = np.min(low[i - window + 1:i + 1])
        mid = (upper + lower) / 2.0
        result[i] = (upper - lower) / mid if mid != 0 else 0.0

    return result


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        window: int) -> np.ndarray:
    """
    Average True Range.
    Matches ta.volatility.average_true_range(high, low, close, window=N).

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = EMA(TR, window)  (ta library uses EMA/Wilder smoothing)
    """
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # ta library uses RMA (Wilder's smoothing = EMA with alpha=1/window)
    return _rma(tr, window)


def awesome_oscillator(high: np.ndarray, low: np.ndarray,
                       window1: int, window2: int) -> np.ndarray:
    """
    Awesome Oscillator.
    Matches ta.momentum.AwesomeOscillatorIndicator(high, low, window1, window2).

    AO = SMA(median_price, window1) - SMA(median_price, window2)
    median_price = (high + low) / 2
    """
    median_price = (high + low) / 2.0
    sma1 = _sma(median_price, window1)
    sma2 = _sma(median_price, window2)
    return sma1 - sma2


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        window: int) -> np.ndarray:
    """
    Average Directional Index.
    Matches ta.trend.ADXIndicator(high, low, close, window=N).adx().

    1. Compute +DM and -DM
    2. Smooth with RMA (Wilder's)
    3. +DI = smoothed(+DM) / ATR * 100
    4. -DI = smoothed(-DM) / ATR * 100
    5. DX = |+DI - -DI| / (+DI + -DI) * 100
    6. ADX = RMA(DX, window)
    """
    n = len(close)

    # True Range
    tr_arr = np.zeros(n)
    tr_arr[0] = high[0] - low[0]
    for i in range(1, n):
        tr_arr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Smooth with RMA (Wilder's)
    atr_smooth = _rma(tr_arr, window)
    plus_dm_smooth = _rma(plus_dm, window)
    minus_dm_smooth = _rma(minus_dm, window)

    # DI
    plus_di = np.where(atr_smooth > 0, plus_dm_smooth / atr_smooth * 100, 0.0)
    minus_di = np.where(atr_smooth > 0, minus_dm_smooth / atr_smooth * 100, 0.0)

    # DX
    di_sum = plus_di + minus_di
    with np.errstate(invalid="ignore"):
        dx = np.where(di_sum > 0, np.abs(plus_di - minus_di) / di_sum * 100, 0.0)

    # ADX = RMA of DX
    return _rma(dx, window)


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _sma(data: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average. NaN for incomplete windows."""
    result = np.full(len(data), np.nan)
    cumsum = np.cumsum(data)
    result[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:-window]])) / window
    return result


def _ema(data: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (adjust=True, matching pandas default)."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(data, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _rma(data: np.ndarray, window: int) -> np.ndarray:
    """
    Wilder's smoothing (RMA). alpha = 1/window.
    First value = SMA of first `window` values.
    """
    n = len(data)
    result = np.full(n, np.nan)

    if n < window:
        return result

    # Seed with SMA
    result[window - 1] = np.mean(data[:window])

    alpha = 1.0 / window
    for i in range(window, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result
