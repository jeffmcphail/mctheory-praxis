"""
ENGINE: Regime Matrix Detector — 12-class market regime classification.

Computes a compact integer state vector representing the current market regime
across 12 orthogonal dimensions. Each class has a small number of discrete
states (3–5) designed for use as CPO features.

Design spec: docs/REGIME_MATRIX.md

Regime Classes:
    A. Trend           (-2 to +2)  — ADX + multi-period return alignment
    B. Vol level       (0 to 3)    — Garman-Klass RV vs 252d percentile rank
    C. Vol trend       (-1 to +1)  — rv_1d / rv_7d ratio
    D. Serial corr     (-2 to +2)  — Hurst exponent + variance ratio
    E. Microstructure  (-1 to +1)  — Order flow imbalance from OHLCV
    F. Funding/pos     (-2 to +2)  — Annualized funding + OI change
    G. Liquidity       (0 to 3)    — Corwin-Schultz + Amihud + volume z
    H. Cross-asset     (0 to 2)    — Rolling pairwise correlation mean
    I. Volume part     (0 to 3)    — Volume ratio + price-volume sign
    J. Term structure  (-1 to +1)  — Funding term slope
    K. Dispersion      (0 to 2)    — Cross-sectional return std
    L. RV/IV spread    (-1 to +1)  — GARCH RV vs DVOL (stub without data)

Usage:
    engine = RegimeEngine()
    state = engine.compute(
        ohlcv_hourly=df,           # Required: OHLCV with DatetimeIndex
        funding_rates=fr_series,    # Optional: 8h funding rate series
        oi_series=oi_series,        # Optional: open interest series
        universe_ohlcv=dict_of_dfs, # Optional: multi-asset for H, K
        dvol_series=dvol_series,    # Optional: implied vol index for L
    )
    print(state.vector)        # [2, 1, 0, -1, 0, 1, 1, 0, 2, 0, 1, 0]
    print(state.raw_features)  # dict of ~60 raw sub-features
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

REGIME_CLASSES = list("ABCDEFGHIJKL")

REGIME_CLASS_NAMES = {
    "A": "trend",
    "B": "vol_level",
    "C": "vol_trend",
    "D": "serial_corr",
    "E": "microstructure",
    "F": "funding_positioning",
    "G": "liquidity",
    "H": "cross_asset_corr",
    "I": "volume_participation",
    "J": "term_structure",
    "K": "dispersion",
    "L": "rv_iv_spread",
}

# State ranges for each class
REGIME_STATE_RANGES = {
    "A": (-2, -1, 0, 1, 2),
    "B": (0, 1, 2, 3),
    "C": (-1, 0, 1),
    "D": (-2, -1, 0, 1, 2),
    "E": (-1, 0, 1),
    "F": (-2, -1, 0, 1, 2),
    "G": (0, 1, 2, 3),
    "H": (0, 1, 2),
    "I": (0, 1, 2, 3),
    "J": (-1, 0, 1),
    "K": (0, 1, 2),
    "L": (-1, 0, 1),
}


@dataclass
class RegimeState:
    """Complete regime state at a point in time."""
    states: dict[str, int]          # class letter → integer state
    raw_features: dict[str, float]  # all computed sub-features
    missing: list[str]              # classes that couldn't be computed

    @property
    def vector(self) -> list[int]:
        """12-element integer vector in A–L order."""
        return [self.states.get(c, 0) for c in REGIME_CLASSES]

    @property
    def named_states(self) -> dict[str, int]:
        """Human-readable state names."""
        return {REGIME_CLASS_NAMES[c]: v for c, v in self.states.items()}

    def to_feature_row(self) -> dict[str, float]:
        """Flat dict of all features suitable for DataFrame row."""
        row = {}
        for c in REGIME_CLASSES:
            row[f"regime_{REGIME_CLASS_NAMES[c]}"] = float(self.states.get(c, 0))
        row.update(self.raw_features)
        return row


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _safe_std(arr: np.ndarray) -> float:
    """Standard deviation with NaN safety."""
    if len(arr) < 2:
        return 0.0
    s = np.nanstd(arr, ddof=1)
    return float(s) if np.isfinite(s) else 0.0


def _percentile_rank(value: float, history: np.ndarray) -> float:
    """Percentile rank of value within history [0, 1]."""
    valid = history[np.isfinite(history)]
    if len(valid) < 10:
        return 0.5
    return float(np.mean(valid <= value))


def _z_score(value: float, history: np.ndarray) -> float:
    """Z-score of value relative to history."""
    valid = history[np.isfinite(history)]
    if len(valid) < 10:
        return 0.0
    mu = np.nanmean(valid)
    sigma = np.nanstd(valid, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float((value - mu) / sigma)


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


# ═════════════════════════════════════════════════════════════════════════════
# REGIME CLASS COMPUTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def compute_trend_regime(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    prev_close: np.ndarray | None = None,
) -> tuple[int, dict[str, float]]:
    """
    A. Trend Regime — ADX + multi-period return alignment.

    States: -2 (strong downtrend), -1 (mild downtrend), 0 (no trend),
            +1 (mild uptrend), +2 (strong uptrend)

    Requires at least 35 bars of daily OHLC.
    """
    n = len(close)
    if n < 35:
        return 0, {"adx_14": np.nan, "ret_1d": 0.0, "ret_7d": 0.0, "ret_30d": 0.0}

    # Use provided prev_close or construct from close shifted by 1
    if prev_close is None:
        pc = np.roll(close, 1)
        pc[0] = close[0]
    else:
        pc = prev_close

    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - pc[1:]),
            np.abs(low[1:] - pc[1:])
        )
    )

    # +DM / -DM
    plus_dm = np.maximum(high[1:] - high[:-1], 0.0)
    minus_dm = np.maximum(low[:-1] - low[1:], 0.0)

    # Zero out whichever is smaller (Wilder's rule)
    mask_plus = plus_dm > minus_dm
    mask_minus = minus_dm > plus_dm
    plus_dm = np.where(mask_plus, plus_dm, 0.0)
    minus_dm = np.where(mask_minus, minus_dm, 0.0)

    # Smoothed with EMA(14)
    atr_14 = _ema(tr, 14)
    plus_di_14 = 100.0 * _ema(plus_dm, 14) / (atr_14 + 1e-10)
    minus_di_14 = 100.0 * _ema(minus_dm, 14) / (atr_14 + 1e-10)

    # DX and ADX
    di_sum = plus_di_14 + minus_di_14
    di_diff = np.abs(plus_di_14 - minus_di_14)
    dx = 100.0 * di_diff / (di_sum + 1e-10)
    adx = _ema(dx, 14)

    adx_14 = float(adx[-1])

    # Multi-period returns (log)
    ret_1d = float(np.log(close[-1] / close[-2])) if close[-2] > 0 else 0.0
    ret_7d = float(np.log(close[-1] / close[-8])) if n >= 9 and close[-8] > 0 else 0.0
    ret_30d = float(np.log(close[-1] / close[-31])) if n >= 32 and close[-31] > 0 else 0.0

    direction = 1 if ret_7d > 0 else -1

    # State classification
    signs_aligned = (np.sign(ret_1d) == np.sign(ret_7d) == np.sign(ret_30d))
    if adx_14 > 40 and signs_aligned and ret_1d != 0:
        state = direction * 2
    elif adx_14 > 25:
        state = direction * 1
    else:
        state = 0

    raw = {
        "adx_14": adx_14,
        "plus_di_14": float(plus_di_14[-1]),
        "minus_di_14": float(minus_di_14[-1]),
        "ret_1d": ret_1d,
        "ret_7d": ret_7d,
        "ret_30d": ret_30d,
    }
    return state, raw


def _garman_klass_bars(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Garman-Klass volatility estimator per bar.
    Uses OHLC — ~4x lower variance than close-to-close.
    """
    log_hl = np.log(high / (low + 1e-10))
    log_co = np.log(close / (open_ + 1e-10))
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return gk


def compute_vol_regime(
    open_h: np.ndarray,
    high_h: np.ndarray,
    low_h: np.ndarray,
    close_h: np.ndarray,
    bars_per_day: int = 24,
) -> tuple[int, int, dict[str, float]]:
    """
    B & C. Volatility Level + Trend.

    B states: 0 (compressed), 1 (normal), 2 (elevated), 3 (spike)
    C states: -1 (compressing), 0 (stable), +1 (expanding)

    Requires hourly OHLCV bars. bars_per_day=24 for crypto, 7 for equities.
    """
    n = len(close_h)
    min_bars = bars_per_day * 30  # need 30 days minimum for rv_30d
    if n < min_bars:
        return 0, 0, {"rv_1d": np.nan, "rv_7d": np.nan, "rv_30d": np.nan,
                       "vol_pct_rank": 0.5, "vol_ratio": 1.0}

    gk = _garman_klass_bars(open_h, high_h, low_h, close_h)

    # Annualized RV at different horizons
    ann_factor = bars_per_day * 365  # crypto = 24*365, equities = 7*252
    rv_1d = float(np.sqrt(np.nanmean(gk[-bars_per_day:]) * ann_factor))
    rv_7d = float(np.sqrt(np.nanmean(gk[-bars_per_day * 7:]) * ann_factor))
    rv_30d = float(np.sqrt(np.nanmean(gk[-bars_per_day * 30:]) * ann_factor))

    # Percentile rank of rv_1d vs rolling 252-day history
    hist_len = min(n, bars_per_day * 252)
    # Compute daily RV for each of the last 252 days
    daily_rvs = []
    for i in range(min(252, n // bars_per_day)):
        end = n - i * bars_per_day
        start = end - bars_per_day
        if start < 0:
            break
        chunk = gk[start:end]
        daily_rvs.append(float(np.sqrt(np.nanmean(chunk) * ann_factor)))

    daily_rvs_arr = np.array(daily_rvs)
    vol_pct_rank = _percentile_rank(rv_1d, daily_rvs_arr)

    # B: Vol level
    if vol_pct_rank < 0.20:
        vol_level = 0  # compressed
    elif vol_pct_rank < 0.60:
        vol_level = 1  # normal
    elif vol_pct_rank < 0.85:
        vol_level = 2  # elevated
    else:
        vol_level = 3  # spike

    # C: Vol trend
    vol_ratio = rv_1d / (rv_7d + 1e-10)
    if vol_ratio > 1.25:
        vol_trend = 1   # expanding
    elif vol_ratio < 0.80:
        vol_trend = -1  # compressing
    else:
        vol_trend = 0   # stable

    raw = {
        "rv_1d": rv_1d,
        "rv_7d": rv_7d,
        "rv_30d": rv_30d,
        "vol_pct_rank": vol_pct_rank,
        "vol_ratio": vol_ratio,
    }
    return vol_level, vol_trend, raw


def _hurst_rs(log_prices: np.ndarray, min_lag: int = 8, max_lag: int = 200) -> float:
    """
    Hurst exponent via Rescaled Range (R/S) analysis.

    H > 0.5 → persistent (momentum favored)
    H < 0.5 → antipersistent (mean-reversion favored)
    H ≈ 0.5 → random walk
    """
    n = len(log_prices)
    if n < min_lag * 2:
        return 0.5  # insufficient data

    max_lag = min(max_lag, n // 2)
    if max_lag <= min_lag:
        return 0.5

    lags = np.unique(np.logspace(
        np.log10(min_lag), np.log10(max_lag), 20
    ).astype(int))
    lags = lags[lags >= min_lag]

    rs_values = []
    lag_values = []

    for lag in lags:
        # Split into non-overlapping chunks
        n_chunks = n // lag
        if n_chunks < 1:
            continue

        rs_list = []
        for c in range(n_chunks):
            chunk = log_prices[c * lag:(c + 1) * lag]
            if len(chunk) < 2:
                continue
            diffs = np.diff(chunk)
            mean_diff = np.mean(diffs)
            cumdev = np.cumsum(diffs - mean_diff)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(diffs, ddof=1)
            if S > 1e-12:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append(np.mean(rs_list))
            lag_values.append(lag)

    if len(lag_values) < 3:
        return 0.5

    # Linear regression in log-log space
    log_lags = np.log(np.array(lag_values, dtype=float))
    log_rs = np.log(np.array(rs_values, dtype=float))

    try:
        coeffs = np.polyfit(log_lags, log_rs, 1)
        H = float(coeffs[0])
        return np.clip(H, 0.0, 1.0)
    except (np.linalg.LinAlgError, ValueError):
        return 0.5


def _variance_ratio(prices: np.ndarray, q: int) -> float:
    """
    Variance ratio test.
    VR(q) = Var(q-period ret) / (q * Var(1-period ret))
    VR > 1 → momentum, VR < 1 → mean-reversion
    """
    if len(prices) < q + 2:
        return 1.0

    log_p = np.log(prices + 1e-10)
    ret_1 = np.diff(log_p)
    ret_q = log_p[q:] - log_p[:-q]

    var_1 = np.var(ret_1, ddof=1)
    var_q = np.var(ret_q, ddof=1)

    if var_1 < 1e-15:
        return 1.0

    return float(var_q / (q * var_1))


def compute_serial_corr_regime(
    close_h: np.ndarray,
    lookback_days: int = 21,
    bars_per_day: int = 24,
) -> tuple[int, dict[str, float]]:
    """
    D. Serial Correlation Regime — Hurst exponent + variance ratio.

    Most important single feature per REGIME_MATRIX.md.

    States: -2 (strong MR), -1 (mild MR), 0 (random),
            +1 (mild momentum), +2 (strong momentum)
    """
    n_bars = lookback_days * bars_per_day
    n = len(close_h)
    if n < n_bars:
        return 0, {"hurst": 0.5, "vr_4": 1.0, "vr_24": 1.0, "serial_score": 0.0}

    log_prices = np.log(close_h[-n_bars:] + 1e-10)
    hurst = _hurst_rs(log_prices)

    # Variance ratios on hourly returns
    vr_4 = _variance_ratio(close_h[-n_bars:], q=4)
    vr_24 = _variance_ratio(close_h[-n_bars:], q=bars_per_day)

    # Normalized score
    score = (hurst - 0.5) * 4.0

    if score > 1.5:
        state = 2
    elif score > 0.4:
        state = 1
    elif score < -1.5:
        state = -2
    elif score < -0.4:
        state = -1
    else:
        state = 0

    raw = {
        "hurst": hurst,
        "vr_4": vr_4,
        "vr_24": vr_24,
        "serial_score": score,
    }
    return state, raw


def compute_microstructure_regime(
    open_h: np.ndarray,
    high_h: np.ndarray,
    low_h: np.ndarray,
    close_h: np.ndarray,
    volume_h: np.ndarray,
    bars_per_day: int = 24,
) -> tuple[int, dict[str, float]]:
    """
    E. Microstructure / Order Flow Imbalance.

    Lee-Ready tick rule approximation from OHLCV (no tick data needed).

    States: -1 (sell pressure), 0 (balanced), +1 (buy pressure)
    """
    n = len(close_h)
    if n < bars_per_day:
        return 0, {"ofi_24h": 0.0, "amihud": 0.0}

    # Buy/sell volume approximation
    hl_range = high_h - low_h + 1e-10
    buy_frac = np.clip((close_h - open_h) / hl_range, 0, 1)
    buy_vol = volume_h * buy_frac
    sell_vol = volume_h - buy_vol
    ofi_bar = (buy_vol - sell_vol) / (volume_h + 1e-10)

    # 24-hour rolling mean
    window = min(bars_per_day, n)
    ofi_24h = float(np.nanmean(ofi_bar[-window:]))

    # Amihud illiquidity (daily scale)
    daily_ret = float(np.log(close_h[-1] / (close_h[-bars_per_day] + 1e-10)))
    daily_dv = float(np.sum(close_h[-bars_per_day:] * volume_h[-bars_per_day:]))
    amihud = abs(daily_ret) / (daily_dv + 1e-10)

    if ofi_24h > 0.15:
        state = 1
    elif ofi_24h < -0.15:
        state = -1
    else:
        state = 0

    raw = {
        "ofi_24h": ofi_24h,
        "ofi_bar_last": float(ofi_bar[-1]),
        "amihud": amihud,
    }
    return state, raw


def compute_funding_regime(
    funding_rates: np.ndarray,
    oi_values: np.ndarray | None = None,
) -> tuple[int, dict[str, float]]:
    """
    F. Funding / Positioning Regime.

    States: -2 (heavily short), -1 (mildly short), 0 (neutral),
            +1 (mildly long), +2 (heavily long)
    """
    if len(funding_rates) < 3:
        return 0, {"fr_ann": 0.0, "oi_change_7d": 0.0}

    # Annualize: 8h rate × 3 × 365 × 100
    fr_ann = float(funding_rates[-1]) * 3 * 365 * 100

    # OI change (7d)
    oi_change_7d = 0.0
    if oi_values is not None and len(oi_values) >= 22:
        # ~21 funding payments in 7 days (3/day × 7)
        oi_now = float(oi_values[-1])
        oi_7d = float(oi_values[-22])
        if oi_7d > 0:
            oi_change_7d = (oi_now - oi_7d) / oi_7d

    # Classification
    if fr_ann > 30 and oi_change_7d > 0.10:
        state = 2
    elif fr_ann > 10:
        state = 1
    elif fr_ann < -15 and oi_change_7d < -0.10:
        state = -2
    elif fr_ann < -5:
        state = -1
    else:
        state = 0

    raw = {
        "fr_ann": fr_ann,
        "fr_8h_pct": float(funding_rates[-1]) * 100,
        "oi_change_7d": oi_change_7d,
    }
    return state, raw


def _corwin_schultz_spread(
    high: np.ndarray,
    low: np.ndarray,
) -> float:
    """
    Corwin-Schultz (2012) bid-ask spread estimator from daily H/L.
    Returns estimated spread as fraction (e.g., 0.001 = 10 bps).
    """
    n = len(high)
    if n < 3:
        return 0.0

    spreads = []
    for t in range(1, n):
        h_t = high[t]
        l_t = low[t]
        h_t1 = high[t - 1]
        l_t1 = low[t - 1]

        if l_t <= 0 or l_t1 <= 0 or h_t <= 0 or h_t1 <= 0:
            continue

        beta = (np.log(h_t / l_t)) ** 2 + (np.log(h_t1 / l_t1)) ** 2

        h_2day = max(h_t, h_t1)
        l_2day = min(l_t, l_t1)
        if l_2day <= 0:
            continue
        gamma = (np.log(h_2day / l_2day)) ** 2

        denom = 3 - 2 * np.sqrt(2)
        if denom == 0:
            continue

        alpha_val = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        spread = max(2 * (np.exp(alpha_val) - 1) / (1 + np.exp(alpha_val)), 0.0)
        spreads.append(spread)

    return float(np.mean(spreads)) if spreads else 0.0


def compute_liquidity_regime(
    high_h: np.ndarray,
    low_h: np.ndarray,
    close_h: np.ndarray,
    volume_h: np.ndarray,
    bars_per_day: int = 24,
) -> tuple[int, dict[str, float]]:
    """
    G. Liquidity Regime — Corwin-Schultz spread + Amihud + volume z-score.

    States: 0 (abundant), 1 (normal), 2 (thin), 3 (crisis)
    """
    n = len(close_h)
    min_days = 30
    min_bars = min_days * bars_per_day

    if n < min_bars:
        return 1, {"cs_spread": 0.0, "amihud_ratio": 0.0, "vol_z": 0.0, "liq_score": 0.0}

    # Corwin-Schultz on recent 5 days of hourly bars
    # (aggregate to pseudo-daily for the estimator)
    n_days_cs = min(5, n // bars_per_day)
    daily_highs = []
    daily_lows = []
    for d in range(n_days_cs):
        end = n - d * bars_per_day
        start = end - bars_per_day
        daily_highs.append(np.max(high_h[start:end]))
        daily_lows.append(np.min(low_h[start:end]))
    daily_highs = np.array(daily_highs[::-1])
    daily_lows = np.array(daily_lows[::-1])
    cs_spread = _corwin_schultz_spread(daily_highs, daily_lows)

    # Amihud ratio (daily)
    daily_ret = np.log(close_h[-1] / (close_h[-bars_per_day] + 1e-10))
    daily_dv = np.sum(close_h[-bars_per_day:] * volume_h[-bars_per_day:])
    amihud_ratio = abs(daily_ret) / (daily_dv + 1e-10)

    # Volume z-score (current day vs 30-day mean)
    current_vol = np.sum(volume_h[-bars_per_day:])
    vol_history = []
    for d in range(min_days):
        end = n - d * bars_per_day
        start = end - bars_per_day
        if start < 0:
            break
        vol_history.append(np.sum(volume_h[start:end]))
    vol_z = _z_score(current_vol, np.array(vol_history))

    # 30-day z-scores for CS spread and Amihud
    cs_z = _z_score(cs_spread, np.array([cs_spread]))  # simplified
    amihud_z = _z_score(amihud_ratio, np.array([amihud_ratio]))  # simplified

    # Combined liquidity score (higher = more liquid)
    # Using vol_z directly — positive vol_z means higher than usual volume
    liq_score = -cs_z - amihud_z + vol_z

    # Simpler classification based on vol_z and spread
    if vol_z > 1.0 and cs_spread < 0.002:
        state = 0  # abundant
    elif vol_z > -0.5:
        state = 1  # normal
    elif vol_z > -2.0:
        state = 2  # thin
    else:
        state = 3  # crisis

    raw = {
        "cs_spread": cs_spread,
        "amihud_ratio": amihud_ratio,
        "vol_z": vol_z,
        "liq_score": liq_score,
    }
    return state, raw


def compute_cross_asset_corr_regime(
    universe_rets: dict[str, np.ndarray],
    window: int = 168,
) -> tuple[int, dict[str, float]]:
    """
    H. Cross-Asset Correlation Regime.

    States: 0 (idiosyncratic), 1 (normal), 2 (crisis corr)
    """
    assets = list(universe_rets.keys())
    if len(assets) < 2:
        return 0, {"mean_pairwise_corr": 0.0, "n_assets": len(assets)}

    # Trim all to same length (use last `window` bars)
    min_len = min(len(r) for r in universe_rets.values())
    effective_window = min(window, min_len)
    if effective_window < 20:
        return 0, {"mean_pairwise_corr": 0.0, "n_assets": len(assets)}

    rets_matrix = np.column_stack([
        universe_rets[a][-effective_window:]
        for a in assets
    ])

    # Pairwise correlations
    try:
        corr_matrix = np.corrcoef(rets_matrix, rowvar=False)
        # Upper triangle (excluding diagonal)
        n_a = len(assets)
        upper_corrs = []
        for i in range(n_a):
            for j in range(i + 1, n_a):
                c = corr_matrix[i, j]
                if np.isfinite(c):
                    upper_corrs.append(c)

        if not upper_corrs:
            return 0, {"mean_pairwise_corr": 0.0, "n_assets": len(assets)}

        mean_corr = float(np.mean(upper_corrs))
    except (np.linalg.LinAlgError, ValueError):
        return 0, {"mean_pairwise_corr": 0.0, "n_assets": len(assets)}

    if mean_corr > 0.75:
        state = 2  # crisis
    elif mean_corr > 0.45:
        state = 1  # normal
    else:
        state = 0  # idiosyncratic

    raw = {
        "mean_pairwise_corr": mean_corr,
        "n_assets": float(len(assets)),
        "max_corr": float(np.max(upper_corrs)) if upper_corrs else 0.0,
        "min_corr": float(np.min(upper_corrs)) if upper_corrs else 0.0,
    }
    return state, raw


def compute_volume_participation_regime(
    close_h: np.ndarray,
    volume_h: np.ndarray,
    bars_per_day: int = 24,
) -> tuple[int, dict[str, float]]:
    """
    I. Volume Participation Regime.

    States: 0 (dormant), 1 (normal), 2 (active), 3 (capitulation)
    """
    n = len(volume_h)
    if n < bars_per_day * 30:
        return 1, {"vol_ratio_24h": 1.0, "conviction": False}

    # Volume ratio: last 24h vs 30d mean
    vol_24h = np.sum(volume_h[-bars_per_day:])
    vol_30d_daily = []
    for d in range(30):
        end = n - d * bars_per_day
        start = end - bars_per_day
        if start < 0:
            break
        vol_30d_daily.append(np.sum(volume_h[start:end]))
    vol_30d_mean = np.mean(vol_30d_daily)
    vol_ratio = vol_24h / (vol_30d_mean + 1e-10)

    # Price-volume sign confirmation
    ret_24h = np.log(close_h[-1] / (close_h[-bars_per_day] + 1e-10))
    # Conviction: price direction aligns with above-average volume
    conviction = bool((ret_24h > 0 and vol_ratio > 1.0) or
                      (ret_24h < 0 and vol_ratio > 1.0))

    if vol_ratio < 0.5:
        state = 0  # dormant
    elif vol_ratio < 1.5 and conviction:
        state = 1  # normal
    elif vol_ratio < 3.0 and conviction:
        state = 2  # active
    elif vol_ratio >= 3.0:
        state = 3  # capitulation
    else:
        state = 1  # default normal

    raw = {
        "vol_ratio_24h": float(vol_ratio),
        "conviction": float(conviction),
        "ret_24h": float(ret_24h),
    }
    return state, raw


def compute_term_structure_regime(
    funding_rates: np.ndarray,
    bars_per_day: int = 3,  # 3 funding payments per day
) -> tuple[int, dict[str, float]]:
    """
    J. Term Structure Regime.

    Computes slope between 7-day and 30-day annualized funding.

    States: -1 (backwardation), 0 (flat), +1 (contango)
    """
    n = len(funding_rates)
    if n < bars_per_day * 30:
        return 0, {"fr_7d_ann": 0.0, "fr_30d_ann": 0.0, "fr_slope": 0.0}

    # 7-day and 30-day average, annualized
    fr_7d = float(np.nanmean(funding_rates[-bars_per_day * 7:])) * bars_per_day * 365 * 100
    fr_30d = float(np.nanmean(funding_rates[-bars_per_day * 30:])) * bars_per_day * 365 * 100

    fr_slope = (fr_30d - fr_7d) / (abs(fr_30d) + 1e-6)

    if fr_slope > 0.15:
        state = 1   # contango (long-term higher)
    elif fr_slope < -0.15:
        state = -1  # backwardation
    else:
        state = 0   # flat

    raw = {
        "fr_7d_ann": fr_7d,
        "fr_30d_ann": fr_30d,
        "fr_slope": fr_slope,
    }
    return state, raw


def compute_dispersion_regime(
    universe_rets_24h: dict[str, float],
    history_dispersion: np.ndarray | None = None,
) -> tuple[int, dict[str, float]]:
    """
    K. Cross-Sectional Dispersion (alpha environment).

    States: 0 (low), 1 (normal), 2 (high)
    """
    rets = list(universe_rets_24h.values())
    if len(rets) < 3:
        return 1, {"dispersion": 0.0, "dispersion_pct_rank": 0.5}

    dispersion = float(np.std(rets))

    # Percentile rank if history available
    if history_dispersion is not None and len(history_dispersion) > 20:
        pct_rank = _percentile_rank(dispersion, history_dispersion)
    else:
        pct_rank = 0.5

    if pct_rank > 0.70:
        state = 2  # high
    elif pct_rank > 0.30:
        state = 1  # normal
    else:
        state = 0  # low

    raw = {
        "dispersion": dispersion,
        "dispersion_pct_rank": pct_rank,
    }
    return state, raw


def compute_rv_iv_regime(
    rv_current: float,
    dvol: float | None = None,
) -> tuple[int, dict[str, float]]:
    """
    L. RV / IV Spread (VRP) — requires real options data.

    States: -1 (cheap vol / IV < RV), 0 (balanced), +1 (expensive vol / IV > RV)

    Currently a stub: returns 0 if no DVOL data provided.
    """
    if dvol is None or not np.isfinite(dvol):
        return 0, {"rv_current": rv_current, "dvol": np.nan, "vrp": np.nan}

    vrp = dvol - rv_current  # positive = IV expensive relative to RV

    if vrp > 10:    # 10% annualized
        state = 1   # expensive vol
    elif vrp < -10:
        state = -1  # cheap vol
    else:
        state = 0

    raw = {
        "rv_current": rv_current,
        "dvol": dvol,
        "vrp": vrp,
    }
    return state, raw


# ═════════════════════════════════════════════════════════════════════════════
# MAIN REGIME ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class RegimeEngine:
    """
    Computes all 12 regime states from market data.

    Input DataFrame (ohlcv_hourly) must have columns:
        open, high, low, close, volume
    with a DatetimeIndex in UTC.
    """

    def __init__(self, bars_per_day: int = 24):
        """
        Args:
            bars_per_day: 24 for crypto (24/7), 7 for US equities (9:30-16:00).
        """
        self.bars_per_day = bars_per_day

    def compute(
        self,
        ohlcv_hourly: pd.DataFrame,
        funding_rates: np.ndarray | pd.Series | None = None,
        oi_series: np.ndarray | pd.Series | None = None,
        universe_ohlcv: dict[str, pd.DataFrame] | None = None,
        dvol_series: np.ndarray | pd.Series | None = None,
    ) -> RegimeState:
        """
        Compute full 12-class regime state.

        Args:
            ohlcv_hourly: Primary asset OHLCV with DatetimeIndex.
                          Must have columns: open, high, low, close, volume.
            funding_rates: Raw 8h funding rate series (fractional, e.g. 0.0001).
            oi_series: Open interest values corresponding to funding timestamps.
            universe_ohlcv: Dict of asset→DataFrame for cross-asset features (H, K).
            dvol_series: Implied vol index (e.g., Deribit DVOL) for class L.

        Returns:
            RegimeState with states, raw_features, and missing classes.
        """
        states: dict[str, int] = {}
        raw_features: dict[str, float] = {}
        missing: list[str] = []

        # Extract arrays from DataFrame
        o = ohlcv_hourly["open"].values.astype(float)
        h = ohlcv_hourly["high"].values.astype(float)
        lo = ohlcv_hourly["low"].values.astype(float)
        c = ohlcv_hourly["close"].values.astype(float)
        v = ohlcv_hourly["volume"].values.astype(float)

        # Convert optional Series to arrays
        fr = funding_rates.values if isinstance(funding_rates, pd.Series) else funding_rates
        oi = oi_series.values if isinstance(oi_series, pd.Series) else oi_series
        dvol_arr = dvol_series.values if isinstance(dvol_series, pd.Series) else dvol_series

        # --- A. Trend (uses daily OHLC, aggregate from hourly) ---
        try:
            daily_close, daily_high, daily_low = self._aggregate_daily(c, h, lo)
            state_a, raw_a = compute_trend_regime(daily_close, daily_high, daily_low)
            states["A"] = state_a
            raw_features.update({f"A_{k}": v for k, v in raw_a.items()})
        except Exception as e:
            logger.warning("Regime A (trend) failed: %s", e)
            states["A"] = 0
            missing.append("A")

        # --- B & C. Volatility Level + Trend ---
        try:
            state_b, state_c, raw_bc = compute_vol_regime(
                o, h, lo, c, self.bars_per_day
            )
            states["B"] = state_b
            states["C"] = state_c
            raw_features.update({f"BC_{k}": v for k, v in raw_bc.items()})
        except Exception as e:
            logger.warning("Regime B/C (vol) failed: %s", e)
            states["B"] = 0
            states["C"] = 0
            missing.extend(["B", "C"])

        # --- D. Serial Correlation (most important) ---
        try:
            state_d, raw_d = compute_serial_corr_regime(
                c, lookback_days=21, bars_per_day=self.bars_per_day
            )
            states["D"] = state_d
            raw_features.update({f"D_{k}": v for k, v in raw_d.items()})
        except Exception as e:
            logger.warning("Regime D (serial corr) failed: %s", e)
            states["D"] = 0
            missing.append("D")

        # --- E. Microstructure ---
        try:
            state_e, raw_e = compute_microstructure_regime(
                o, h, lo, c, v, self.bars_per_day
            )
            states["E"] = state_e
            raw_features.update({f"E_{k}": v for k, v in raw_e.items()})
        except Exception as e:
            logger.warning("Regime E (microstructure) failed: %s", e)
            states["E"] = 0
            missing.append("E")

        # --- F. Funding / Positioning ---
        if fr is not None and len(fr) >= 3:
            try:
                state_f, raw_f = compute_funding_regime(fr, oi)
                states["F"] = state_f
                raw_features.update({f"F_{k}": v for k, v in raw_f.items()})
            except Exception as e:
                logger.warning("Regime F (funding) failed: %s", e)
                states["F"] = 0
                missing.append("F")
        else:
            states["F"] = 0
            missing.append("F")

        # --- G. Liquidity ---
        try:
            state_g, raw_g = compute_liquidity_regime(
                h, lo, c, v, self.bars_per_day
            )
            states["G"] = state_g
            raw_features.update({f"G_{k}": v for k, v in raw_g.items()})
        except Exception as e:
            logger.warning("Regime G (liquidity) failed: %s", e)
            states["G"] = 0
            missing.append("G")

        # --- H. Cross-Asset Correlation ---
        if universe_ohlcv and len(universe_ohlcv) >= 2:
            try:
                universe_rets = {}
                window = 168  # 7 days of hourly bars
                for asset, df in universe_ohlcv.items():
                    if len(df) >= window:
                        asset_close = df["close"].values.astype(float)
                        log_rets = np.diff(np.log(asset_close[-window - 1:] + 1e-10))
                        universe_rets[asset] = log_rets

                state_h, raw_h = compute_cross_asset_corr_regime(
                    universe_rets, window=window
                )
                states["H"] = state_h
                raw_features.update({f"H_{k}": v for k, v in raw_h.items()})
            except Exception as e:
                logger.warning("Regime H (cross-asset) failed: %s", e)
                states["H"] = 0
                missing.append("H")
        else:
            states["H"] = 0
            missing.append("H")

        # --- I. Volume Participation ---
        try:
            state_i, raw_i = compute_volume_participation_regime(
                c, v, self.bars_per_day
            )
            states["I"] = state_i
            raw_features.update({f"I_{k}": v for k, v in raw_i.items()})
        except Exception as e:
            logger.warning("Regime I (volume) failed: %s", e)
            states["I"] = 0
            missing.append("I")

        # --- J. Term Structure ---
        if fr is not None and len(fr) >= 90:
            try:
                state_j, raw_j = compute_term_structure_regime(fr, bars_per_day=3)
                states["J"] = state_j
                raw_features.update({f"J_{k}": v for k, v in raw_j.items()})
            except Exception as e:
                logger.warning("Regime J (term structure) failed: %s", e)
                states["J"] = 0
                missing.append("J")
        else:
            states["J"] = 0
            missing.append("J")

        # --- K. Cross-Sectional Dispersion ---
        if universe_ohlcv and len(universe_ohlcv) >= 3:
            try:
                bpd = self.bars_per_day
                rets_24h = {}
                for asset, df in universe_ohlcv.items():
                    ac = df["close"].values.astype(float)
                    if len(ac) >= bpd + 1:
                        rets_24h[asset] = float(np.log(ac[-1] / (ac[-bpd] + 1e-10)))

                state_k, raw_k = compute_dispersion_regime(rets_24h)
                states["K"] = state_k
                raw_features.update({f"K_{k}": v for k, v in raw_k.items()})
            except Exception as e:
                logger.warning("Regime K (dispersion) failed: %s", e)
                states["K"] = 0
                missing.append("K")
        else:
            states["K"] = 0
            missing.append("K")

        # --- L. RV / IV Spread (VRP) ---
        rv_current = raw_features.get("BC_rv_1d", 0.0)
        dvol_val = None
        if dvol_arr is not None and len(dvol_arr) > 0:
            dvol_val = float(dvol_arr[-1])

        try:
            state_l, raw_l = compute_rv_iv_regime(rv_current, dvol_val)
            states["L"] = state_l
            raw_features.update({f"L_{k}": v for k, v in raw_l.items()})
        except Exception as e:
            logger.warning("Regime L (VRP) failed: %s", e)
            states["L"] = 0
            missing.append("L")

        return RegimeState(states=states, raw_features=raw_features, missing=missing)

    def _aggregate_daily(
        self,
        close_h: np.ndarray,
        high_h: np.ndarray,
        low_h: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate hourly bars into daily bars."""
        bpd = self.bars_per_day
        n = len(close_h)
        n_days = n // bpd

        if n_days < 1:
            raise ValueError(f"Need at least {bpd} bars for 1 day, got {n}")

        # Trim to full days (from end, so most recent data is preserved)
        start = n - n_days * bpd

        daily_close = np.array([
            close_h[start + (d + 1) * bpd - 1]
            for d in range(n_days)
        ])
        daily_high = np.array([
            np.max(high_h[start + d * bpd: start + (d + 1) * bpd])
            for d in range(n_days)
        ])
        daily_low = np.array([
            np.min(low_h[start + d * bpd: start + (d + 1) * bpd])
            for d in range(n_days)
        ])

        return daily_close, daily_high, daily_low

    def compute_time_series(
        self,
        ohlcv_hourly: pd.DataFrame,
        funding_rates: np.ndarray | pd.Series | None = None,
        oi_series: np.ndarray | pd.Series | None = None,
        universe_ohlcv: dict[str, pd.DataFrame] | None = None,
        dvol_series: np.ndarray | pd.Series | None = None,
        step_bars: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute regime states over a rolling window through the data.

        Returns a DataFrame with one row per evaluation point, containing
        regime state integers and all raw sub-features.

        Args:
            step_bars: Step size in bars (default = bars_per_day = daily eval).
        """
        if step_bars is None:
            step_bars = self.bars_per_day

        n = len(ohlcv_hourly)
        min_bars = self.bars_per_day * 35  # need 35 days for trend ADX

        if n < min_bars:
            logger.warning("Insufficient data for time series: %d bars < %d minimum", n, min_bars)
            return pd.DataFrame()

        rows = []
        eval_points = range(min_bars, n + 1, step_bars)

        for end_idx in eval_points:
            chunk = ohlcv_hourly.iloc[:end_idx]

            # Slice funding/OI to match
            fr_slice = None
            oi_slice = None
            if funding_rates is not None:
                # Approximate: funding is 3x/day, OHLCV is 24x/day (crypto)
                n_days = end_idx // self.bars_per_day
                fr_len = min(n_days * 3, len(funding_rates) if isinstance(funding_rates, np.ndarray)
                             else len(funding_rates))
                if isinstance(funding_rates, pd.Series):
                    fr_slice = funding_rates.iloc[:fr_len]
                elif funding_rates is not None:
                    fr_slice = funding_rates[:fr_len]

                if oi_series is not None:
                    if isinstance(oi_series, pd.Series):
                        oi_slice = oi_series.iloc[:fr_len]
                    else:
                        oi_slice = oi_series[:fr_len]

            # Slice universe to match
            uni_slice = None
            if universe_ohlcv:
                uni_slice = {
                    asset: df.iloc[:end_idx]
                    for asset, df in universe_ohlcv.items()
                    if len(df) >= end_idx
                }

            state = self.compute(
                ohlcv_hourly=chunk,
                funding_rates=fr_slice,
                oi_series=oi_slice,
                universe_ohlcv=uni_slice,
                dvol_series=dvol_series,
            )

            row = state.to_feature_row()
            row["date"] = chunk.index[-1]
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.set_index("date", inplace=True)
        return df
