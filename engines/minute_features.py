"""
Minute-Frequency Feature Computation for Chan CPO.

Implements the exact feature specification from:
    Chan, Belov, Ciobanu (2021) "Conditional Parameter Optimization"
    Chan (2021) "Quantitative Trading" 2e, Chapter 7, pp. 139-147

Paper specification:
    - 8 technical indicators computed from minute bars
    - Applied to EACH of the 2 assets in a pair
    - At 7 different lookback windows: {50, 100, 200, 400, 800, 1600, 3200} min
    - Total features per row: 3 (trading params) + 8 × 2 × 7 = 115

The 8 indicators (from TA Python library):
    1. Bollinger Bands Z-score
    2. Bollinger Bands Bandwidth (paper says 8 but only names 7; bandwidth
       is the natural companion to Z-score and makes the math work)
    3. Money Flow Index (MFI)
    4. Force Index
    5. Donchian Channel (width normalized by price)
    6. Average True Range (normalized)
    7. Awesome Oscillator
    8. Average Directional Index (ADX)

IMPORTANT: The paper's lookback windows {50,100,200,400,800,1600,3200}
are the FEATURE computation windows. These are DISTINCT from the trading
parameter `lookback` grid {30,60,90,120,180,240,360,720} which controls
the strategy z-score rolling window.

Previous implementation error:
    compute_daily_features() used ~17 features from DAILY close prices.
    This module computes 112 features from MINUTE bars at 7 lookback windows.
    This is why Chan/Burgess appeared to fail entirely.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Feature lookback windows (minutes) — from Chan paper p.5 / book p.142
FEATURE_LOOKBACKS = [50, 100, 200, 400, 800, 1600, 3200]

# The 8 indicator names
INDICATOR_NAMES = [
    "bb_zscore",        # Bollinger Bands Z-score
    "bb_bandwidth",     # Bollinger Bands Bandwidth (likely 8th indicator)
    "mfi",              # Money Flow Index
    "force_index",      # Force Index
    "donchian_width",   # Donchian Channel width (normalized)
    "atr",              # Average True Range (normalized)
    "awesome_osc",      # Awesome Oscillator
    "adx",              # Average Directional Index
]

N_INDICATORS = len(INDICATOR_NAMES)       # 8
N_LOOKBACKS = len(FEATURE_LOOKBACKS)      # 7
N_ASSETS_PER_PAIR = 2                     # target + hedge
N_INDICATOR_FEATURES = N_INDICATORS * N_ASSETS_PER_PAIR * N_LOOKBACKS  # 112


def feature_column_names(target_ticker: str = "target",
                         hedge_ticker: str = "hedge") -> list[str]:
    """
    Generate ordered feature column names.

    Returns 112 names like: 'bb_zscore_target_50', 'bb_zscore_target_100', ...

    Uses generic 'target'/'hedge' labels so all pairs share the same
    112-column namespace. This is critical: ticker-specific names (like
    'bb_zscore_MPC_50') cause a 3808-column superset across 68 tickers,
    breaking OOS prediction where per-pair features only cover 112 columns.
    """
    names = []
    for indicator in INDICATOR_NAMES:
        for ticker in [target_ticker, hedge_ticker]:
            for lb in FEATURE_LOOKBACKS:
                names.append(f"{indicator}_{ticker}_{lb}")
    return names


# ═════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL INDICATOR COMPUTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _ema_series(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (vectorized)."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(series: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with leading NaN fill."""
    if len(series) < window:
        return np.full_like(series, np.nan, dtype=float)
    cumsum = np.cumsum(np.insert(series.astype(float), 0, 0))
    sma = (cumsum[window:] - cumsum[:-window]) / window
    # Pad beginning with NaN
    return np.concatenate([np.full(window - 1, np.nan), sma])


def compute_bollinger_zscore(close: np.ndarray, lookback: int) -> float:
    """
    Bollinger Bands Z-score at last bar.

    Z = (close - SMA(lookback)) / std(lookback)
    """
    if len(close) < lookback:
        return 0.0
    window = close[-lookback:]
    mu = np.mean(window)
    sigma = np.std(window, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float((close[-1] - mu) / sigma)


def compute_bollinger_bandwidth(close: np.ndarray, lookback: int) -> float:
    """
    Bollinger Bandwidth at last bar.

    BW = (upper - lower) / middle = 2 * k * std / SMA
    Using k=2 (standard Bollinger), BW = 4 * std(lookback) / SMA(lookback)
    """
    if len(close) < lookback:
        return 0.0
    window = close[-lookback:]
    mu = np.mean(window)
    sigma = np.std(window, ddof=1)
    if mu < 1e-12:
        return 0.0
    return float(4.0 * sigma / mu)


def compute_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                volume: np.ndarray, lookback: int) -> float:
    """
    Money Flow Index at last bar.

    MFI uses typical price × volume, separated into positive and negative
    flow based on whether typical price is rising or falling.
    Returns value in [0, 100].
    """
    n = len(close)
    if n < lookback + 1:
        return 50.0

    typical_price = (high + low + close) / 3.0
    raw_mf = typical_price * volume

    # Direction: compare to previous typical price
    tp_diff = np.diff(typical_price)

    # Use last `lookback` periods of flow data
    start = max(0, n - lookback - 1)
    tp_d = tp_diff[start:]
    mf = raw_mf[start + 1:]

    pos_flow = np.sum(mf[tp_d > 0])
    neg_flow = np.sum(mf[tp_d < 0])

    if neg_flow < 1e-12:
        return 100.0
    money_ratio = pos_flow / neg_flow
    mfi = 100.0 - (100.0 / (1.0 + money_ratio))
    return float(np.clip(mfi, 0, 100))


def compute_force_index(close: np.ndarray, volume: np.ndarray,
                        lookback: int) -> float:
    """
    Force Index (EMA-smoothed).

    FI = EMA(close_change × volume, lookback)
    Normalized by recent average dollar volume for scale invariance.
    """
    n = len(close)
    if n < lookback + 1:
        return 0.0

    price_change = np.diff(close)
    raw_fi = price_change * volume[1:]

    if len(raw_fi) < lookback:
        return 0.0

    fi_ema = _ema_series(raw_fi[-lookback:], lookback)
    fi_val = fi_ema[-1]

    # Normalize by average dollar volume
    avg_dv = np.mean(close[-lookback:] * volume[-lookback:])
    if avg_dv < 1e-12:
        return 0.0
    return float(fi_val / avg_dv)


def compute_donchian_width(high: np.ndarray, low: np.ndarray,
                           close: np.ndarray, lookback: int) -> float:
    """
    Donchian Channel width (normalized by price).

    Width = (highest_high - lowest_low) / close
    """
    n = len(high)
    if n < lookback:
        return 0.0

    hh = np.max(high[-lookback:])
    ll = np.min(low[-lookback:])
    mid = close[-1]
    if mid < 1e-12:
        return 0.0
    return float((hh - ll) / mid)


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                lookback: int) -> float:
    """
    Average True Range (normalized by price).

    TR = max(H-L, |H-prev_C|, |L-prev_C|)
    ATR = EMA(TR, lookback)
    Normalized: ATR / close for scale invariance.
    """
    n = len(close)
    if n < lookback + 1:
        return 0.0

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )

    if len(tr) < lookback:
        return 0.0

    atr_ema = _ema_series(tr[-lookback:], lookback)
    atr_val = atr_ema[-1]

    if close[-1] < 1e-12:
        return 0.0
    return float(atr_val / close[-1])


def compute_awesome_oscillator(high: np.ndarray, low: np.ndarray,
                               lookback: int) -> float:
    """
    Awesome Oscillator.

    AO = SMA(median_price, 5) - SMA(median_price, 34)
    Adapted: use lookback/7 as short period, lookback as long period.
    Normalized by recent median price.
    """
    n = len(high)
    if n < lookback:
        return 0.0

    median_price = (high + low) / 2.0

    short_period = max(5, lookback // 7)
    long_period = lookback

    sma_short = _sma(median_price, short_period)
    sma_long = _sma(median_price, long_period)

    ao_val = sma_short[-1] - sma_long[-1]
    if not np.isfinite(ao_val):
        return 0.0

    # Normalize by recent median price
    norm = np.mean(median_price[-lookback:])
    if norm < 1e-12:
        return 0.0
    return float(ao_val / norm)


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                lookback: int) -> float:
    """
    Average Directional Index.

    Uses Wilder's smoothing (EMA with span = lookback).
    Returns value in [0, 100].
    """
    n = len(close)
    adx_period = min(14, lookback // 3)
    if adx_period < 3:
        adx_period = 3

    if n < adx_period * 3:
        return 25.0  # neutral default

    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )

    # Directional Movement
    plus_dm = np.maximum(high[1:] - high[:-1], 0.0)
    minus_dm = np.maximum(low[:-1] - low[1:], 0.0)

    # Zero whichever is smaller
    mask_plus = plus_dm > minus_dm
    mask_minus = minus_dm > plus_dm
    plus_dm = np.where(mask_plus, plus_dm, 0.0)
    minus_dm = np.where(mask_minus, minus_dm, 0.0)

    # Smooth with EMA
    atr_s = _ema_series(tr[-lookback:], adx_period)
    plus_di = 100.0 * _ema_series(plus_dm[-lookback:], adx_period) / (atr_s + 1e-10)
    minus_di = 100.0 * _ema_series(minus_dm[-lookback:], adx_period) / (atr_s + 1e-10)

    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100.0 * di_diff / (di_sum + 1e-10)
    adx_arr = _ema_series(dx, adx_period)

    return float(np.clip(adx_arr[-1], 0, 100))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_indicators(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int,
) -> list[float]:
    """
    Compute all 8 indicators for one asset at one lookback window.

    Returns list of 8 floats in INDICATOR_NAMES order.
    """
    return [
        compute_bollinger_zscore(close, lookback),
        compute_bollinger_bandwidth(close, lookback),
        compute_mfi(high, low, close, volume, lookback),
        compute_force_index(close, volume, lookback),
        compute_donchian_width(high, low, close, lookback),
        compute_atr(high, low, close, lookback),
        compute_awesome_oscillator(high, low, lookback),
        compute_adx(high, low, close, lookback),
    ]


def compute_minute_features(
    target_ohlcv: pd.DataFrame,
    hedge_ohlcv: pd.DataFrame,
    as_of_timestamp: pd.Timestamp | None = None,
    lookback_windows: list[int] | None = None,
) -> dict[str, float] | None:
    """
    Compute the full Chan CPO minute-frequency feature vector.

    This is the correct replacement for the old compute_daily_features()
    which incorrectly used daily-frequency data.

    Args:
        target_ohlcv: Minute bars for target asset with columns
                      [open, high, low, close, volume] and DatetimeIndex.
        hedge_ohlcv: Minute bars for hedge asset (same format).
        as_of_timestamp: Compute features using data UP TO this timestamp.
                         If None, uses all available data.
        lookback_windows: Feature lookback windows in minutes.
                          Default: Chan paper spec {50,100,...,3200}.

    Returns:
        Dict of feature_name → value with 112 entries (8 indicators ×
        2 assets × 7 lookbacks), or None if insufficient data.
    """
    if lookback_windows is None:
        lookback_windows = FEATURE_LOOKBACKS

    # Slice data up to as_of_timestamp
    if as_of_timestamp is not None:
        target = target_ohlcv[target_ohlcv.index <= as_of_timestamp]
        hedge = hedge_ohlcv[hedge_ohlcv.index <= as_of_timestamp]
    else:
        target = target_ohlcv
        hedge = hedge_ohlcv

    max_lb = max(lookback_windows)
    if len(target) < max_lb + 50 or len(hedge) < max_lb + 50:
        logger.debug("Insufficient minute data: target=%d, hedge=%d, need=%d",
                     len(target), len(hedge), max_lb + 50)
        return None

    # Extract arrays — use tail to limit memory
    n_keep = max_lb + 200  # some buffer beyond max lookback
    t_o = target["open"].values[-n_keep:].astype(float)
    t_h = target["high"].values[-n_keep:].astype(float)
    t_l = target["low"].values[-n_keep:].astype(float)
    t_c = target["close"].values[-n_keep:].astype(float)
    t_v = target["volume"].values[-n_keep:].astype(float)

    h_o = hedge["open"].values[-n_keep:].astype(float)
    h_h = hedge["high"].values[-n_keep:].astype(float)
    h_l = hedge["low"].values[-n_keep:].astype(float)
    h_c = hedge["close"].values[-n_keep:].astype(float)
    h_v = hedge["volume"].values[-n_keep:].astype(float)

    # Always use generic labels — all pairs share the same 112-column namespace
    target_label = "target"
    hedge_label = "hedge"

    features = {}

    for indicator_name in INDICATOR_NAMES:
        for ticker, (o, h, l, c, v) in [
            (target_label, (t_o, t_h, t_l, t_c, t_v)),
            (hedge_label, (h_o, h_h, h_l, h_c, h_v)),
        ]:
            for lb in lookback_windows:
                col_name = f"{indicator_name}_{ticker}_{lb}"

                if indicator_name == "bb_zscore":
                    val = compute_bollinger_zscore(c, lb)
                elif indicator_name == "bb_bandwidth":
                    val = compute_bollinger_bandwidth(c, lb)
                elif indicator_name == "mfi":
                    val = compute_mfi(h, l, c, v, lb)
                elif indicator_name == "force_index":
                    val = compute_force_index(c, v, lb)
                elif indicator_name == "donchian_width":
                    val = compute_donchian_width(h, l, c, lb)
                elif indicator_name == "atr":
                    val = compute_atr(h, l, c, lb)
                elif indicator_name == "awesome_osc":
                    val = compute_awesome_oscillator(h, l, lb)
                elif indicator_name == "adx":
                    val = compute_adx(h, l, c, lb)
                else:
                    val = 0.0

                features[col_name] = float(val) if np.isfinite(val) else 0.0

    return features


def compute_daily_feature_matrix(
    target_ohlcv: pd.DataFrame,
    hedge_ohlcv: pd.DataFrame,
    trading_days: list[pd.Timestamp],
    market_close_time: str = "16:00",
    lookback_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute minute-frequency features for each trading day.

    For each day in trading_days, computes features using minute data
    up to the previous day's market close (no lookahead).

    This produces the feature matrix that, combined with trading parameters,
    forms the RF training data per the Chan CPO procedure.

    Args:
        target_ohlcv: Full minute bar history for target asset.
        hedge_ohlcv: Full minute bar history for hedge asset.
        trading_days: List of dates to compute features for.
        market_close_time: Time of market close (default "16:00" ET).
        lookback_windows: Feature lookback windows (default: Chan spec).

    Returns:
        DataFrame with index=date, columns=feature names.
        Rows where features couldn't be computed are dropped.
    """
    if lookback_windows is None:
        lookback_windows = FEATURE_LOOKBACKS

    rows = []
    dates = []

    for day in trading_days:
        # Features use data up to PREVIOUS day's close
        prev_day = day - pd.Timedelta(days=1)

        # Find the actual previous trading day
        # (skip weekends / holidays by looking back up to 5 days)
        as_of = None
        for offset in range(0, 6):
            check_date = prev_day - pd.Timedelta(days=offset)
            # Look for data on this date
            check_ts = pd.Timestamp(f"{check_date.date()} {market_close_time}")
            if check_ts.tzinfo is None and target_ohlcv.index.tz is not None:
                check_ts = check_ts.tz_localize(target_ohlcv.index.tz)

            # Check if we have data near this timestamp
            mask = target_ohlcv.index <= check_ts
            if mask.sum() > 100:
                as_of = check_ts
                break

        if as_of is None:
            continue

        feats = compute_minute_features(
            target_ohlcv, hedge_ohlcv,
            as_of_timestamp=as_of,
            lookback_windows=lookback_windows,
        )

        if feats is not None:
            rows.append(feats)
            dates.append(day)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def build_training_matrix(
    feature_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    param_configs: list[dict],
    config_param_names: list[str],
    config_normalizers: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the full CPO training matrix by crossing features with configs.

    Per Chan: each trading day generates len(param_configs) rows, each
    with the same features but different trading parameters.

    Args:
        feature_df: Daily features (N_days × 112).
        returns_df: Daily returns per config (N_days × N_configs).
        param_configs: List of config dicts.
        config_param_names: Names of config parameters.
        config_normalizers: Optional normalization constants for params.

    Returns:
        (X, y) where X has shape (N_days × N_configs, 115) and
        y has shape (N_days × N_configs,) with strategy returns.
    """
    if config_normalizers is None:
        config_normalizers = {}

    rows_X = []
    rows_y = []

    for day_idx, day in enumerate(feature_df.index):
        if day not in returns_df.index:
            continue

        feat_row = feature_df.loc[day].values

        for config_idx, config in enumerate(param_configs):
            # Normalized config parameters
            config_vals = []
            for pname in config_param_names:
                raw = config[pname]
                norm = config_normalizers.get(pname, 1.0)
                config_vals.append(raw / norm if norm != 0 else raw)

            # Full feature row: [config_params, indicator_features]
            full_row = np.concatenate([config_vals, feat_row])
            rows_X.append(full_row)

            # Return for this config on this day
            ret = returns_df.iloc[day_idx, config_idx]
            rows_y.append(ret)

    col_names = config_param_names + list(feature_df.columns)
    X = pd.DataFrame(rows_X, columns=col_names)
    y = pd.Series(rows_y, name="strategy_return")

    return X, y


# ═════════════════════════════════════════════════════════════════════════════
# CHAN_CPO.PY INTEGRATION BRIDGE
# ═════════════════════════════════════════════════════════════════════════════

def compute_pair_features_for_date(
    pair_id: str,
    target_minute_df: pd.DataFrame,
    hedge_minute_df: pd.DataFrame,
    as_of_date: str,
    market_close_hour: int = 16,
    market_close_minute: int = 0,
    lookback_windows: list[int] | None = None,
) -> dict[str, float] | None:
    """
    Bridge function: replaces chan_cpo.compute_daily_features().

    Computes minute-frequency features for a pair as of end-of-day,
    returning a flat dict compatible with chan_cpo's feature pipeline.

    Args:
        pair_id: Pair identifier (e.g., "RSG_WM").
        target_minute_df: Minute bars for target with DatetimeIndex.
        hedge_minute_df: Minute bars for hedge with DatetimeIndex.
        as_of_date: Date string "YYYY-MM-DD" — features use data
                    up to previous day's close.
        market_close_hour: Market close hour (default 16 = 4pm ET).
        market_close_minute: Market close minute (default 0).
        lookback_windows: Feature lookback windows (default: Chan spec).

    Returns:
        Dict with keys: pair_id, date, and 112 indicator features.
        None if insufficient data.
    """
    dt = pd.Timestamp(as_of_date)

    # Build as_of timestamp: previous trading day close
    # Look back up to 5 days to find actual trading data
    for offset in range(1, 6):
        prev_day = dt - pd.Timedelta(days=offset)
        as_of_ts = prev_day.replace(hour=market_close_hour,
                                     minute=market_close_minute)

        # Match timezone
        if target_minute_df.index.tz is not None:
            if as_of_ts.tzinfo is None:
                as_of_ts = as_of_ts.tz_localize(target_minute_df.index.tz)

        # Check we have data near this timestamp
        mask = target_minute_df.index <= as_of_ts
        if mask.sum() > 200:
            break
    else:
        return None

    feats = compute_minute_features(
        target_minute_df, hedge_minute_df,
        as_of_timestamp=as_of_ts,
        lookback_windows=lookback_windows,
    )

    if feats is None:
        return None

    # Add metadata
    result = {"pair_id": pair_id, "date": as_of_date}
    result.update(feats)
    return result


def get_minute_feature_names(
    target_ticker: str = "TGT",
    hedge_ticker: str = "HDG",
) -> list[str]:
    """
    Get the ordered list of feature column names.

    This replaces DailyFeatures.feature_names() for the minute-frequency
    feature set. Used by train_conditional_model to identify which
    columns in features_df are model features.
    """
    return feature_column_names(target_ticker, hedge_ticker)
