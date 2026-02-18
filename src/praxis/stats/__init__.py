"""
Statistical Tests (Phase 4.2).

Port of StationarityTests from statsUtilities.py plus Hurst exponent
and half-life of mean reversion. Core building blocks for Burgess
Stat Arb and cointegration analysis.

Provides:
- ADF (Augmented Dickey-Fuller)
- Johansen cointegration
- Durbin-Watson autocorrelation
- Ljung-Box autocorrelation
- Hurst exponent
- Half-life of mean reversion
- Variance ratio

All functions work on numpy arrays, no pandas dependency in core logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import linalg as LA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox


# ═══════════════════════════════════════════════════════════════════
#  Result Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ADFResult:
    """Result of Augmented Dickey-Fuller test."""
    t_statistic: float
    p_value: float
    lags_used: int
    n_observations: int
    critical_values: dict[str, float] = field(default_factory=dict)
    is_stationary: bool = False

    def at_significance(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


@dataclass
class JohansenResult:
    """Result of Johansen cointegration test."""
    max_eig_stats: np.ndarray = field(default_factory=lambda: np.array([]))
    trace_stats: np.ndarray = field(default_factory=lambda: np.array([]))
    max_eig_critical: np.ndarray = field(default_factory=lambda: np.array([]))
    trace_critical: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    n_cointegrating: int = 0  # At 95% confidence

    @property
    def is_cointegrated(self) -> bool:
        return self.n_cointegrating > 0


@dataclass
class HurstResult:
    """Result of Hurst exponent estimation."""
    hurst_exponent: float
    interpretation: str = ""

    def __post_init__(self):
        if self.hurst_exponent < 0.5:
            self.interpretation = "mean-reverting"
        elif self.hurst_exponent > 0.5:
            self.interpretation = "trending"
        else:
            self.interpretation = "random_walk"


@dataclass
class HalfLifeResult:
    """Half-life of mean reversion."""
    half_life: float
    beta: float  # AR(1) coefficient
    is_mean_reverting: bool = False

    def __post_init__(self):
        self.is_mean_reverting = 0 < self.half_life < float("inf")


@dataclass
class VarianceRatioResult:
    """Variance ratio test result."""
    ratio: float
    lag: int

    @property
    def is_mean_reverting(self) -> bool:
        return self.ratio < 1.0

    @property
    def is_trending(self) -> bool:
        return self.ratio > 1.0


@dataclass
class StationarityResult:
    """Combined stationarity test results."""
    adf: ADFResult | None = None
    johansen: JohansenResult | None = None
    durbin_watson_stat: float | None = None
    ljung_box_stat: float | None = None
    ljung_box_pvalue: float | None = None
    hurst: HurstResult | None = None
    half_life: HalfLifeResult | None = None

    @property
    def is_stationary(self) -> bool:
        """True if ADF indicates stationarity at 5%."""
        if self.adf is None:
            return False
        return self.adf.is_stationary


# ═══════════════════════════════════════════════════════════════════
#  Test Functions
# ═══════════════════════════════════════════════════════════════════

def adf_test(
    series: np.ndarray,
    significance: float = 0.05,
    autolag: str = "AIC",
) -> ADFResult:
    """
    Augmented Dickey-Fuller test for unit root.

    Port of StationarityTests.ADFTest() from statsUtilities.py.

    Args:
        series: 1D time series array.
        significance: Significance level for stationarity determination.
        autolag: Lag selection method ('AIC', 'BIC', etc.)

    Returns:
        ADFResult with test statistics.
    """
    series = np.asarray(series).ravel()
    result = adfuller(series, autolag=autolag)

    critical_values = {k: v for k, v in result[4].items()}

    return ADFResult(
        t_statistic=float(result[0]),
        p_value=float(result[1]),
        lags_used=int(result[2]),
        n_observations=int(result[3]),
        critical_values=critical_values,
        is_stationary=float(result[1]) < significance,
    )


def johansen_test(
    matrix: np.ndarray,
    det_order: int = 0,
    k_ar_diff: int = 1,
    significance: float = 0.05,
) -> JohansenResult:
    """
    Johansen cointegration test.

    Port of StationarityTests.johansenTest() from statsUtilities.py.

    Args:
        matrix: (n_obs, n_vars) matrix of time series.
        det_order: Deterministic term (-1: no det, 0: constant, 1: trend).
        k_ar_diff: Number of lag differences.
        significance: Not used directly by johansen but for counting.

    Returns:
        JohansenResult with trace and max eigenvalue statistics.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        raise ValueError("Johansen test requires at least 2 time series (matrix with 2+ columns)")

    jtest = coint_johansen(matrix, det_order, k_ar_diff)

    # Count cointegrating relationships at 95% (column index 1)
    n_coint = 0
    for i in range(len(jtest.lr1)):
        if jtest.lr1[i] > jtest.cvt[i, 1]:  # trace stat > 95% critical
            n_coint += 1

    return JohansenResult(
        max_eig_stats=jtest.lr2.copy(),
        trace_stats=jtest.lr1.copy(),
        max_eig_critical=jtest.cvm.copy(),
        trace_critical=jtest.cvt.copy(),
        eigenvectors=jtest.evec.copy(),
        eigenvalues=jtest.eig.copy(),
        n_cointegrating=n_coint,
    )


def durbin_watson_test(residuals: np.ndarray) -> float:
    """
    Durbin-Watson test for autocorrelation in residuals.

    Port of StationarityTests.durbanWatsonTest().

    Returns:
        DW statistic. Near 2 = no autocorrelation,
        near 0 = positive, near 4 = negative.
    """
    return float(durbin_watson(np.asarray(residuals).ravel()))


def ljung_box_test(
    series: np.ndarray,
    lags: int = 1,
) -> tuple[float, float]:
    """
    Ljung-Box test for autocorrelation.

    Port of StationarityTests.boxLyungTest().

    Returns:
        (test_statistic, p_value) tuple.
    """
    result = acorr_ljungbox(np.asarray(series).ravel(), lags=[lags])
    stat = float(result["lb_stat"].iloc[0])
    pval = float(result["lb_pvalue"].iloc[0])
    return stat, pval


def hurst_exponent(series: np.ndarray) -> HurstResult:
    """
    Estimate the Hurst exponent using the rescaled range (R/S) method.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending

    Args:
        series: 1D price or returns series.

    Returns:
        HurstResult with exponent and interpretation.
    """
    series = np.asarray(series).ravel()
    n = len(series)

    if n < 20:
        return HurstResult(hurst_exponent=0.5)

    # Use log returns
    returns = np.diff(np.log(np.maximum(series, 1e-10)))

    # Compute R/S for different lag sizes
    max_k = min(n // 2, 100)
    lags = range(2, max_k + 1)
    rs_values = []

    for lag in lags:
        n_chunks = len(returns) // lag
        if n_chunks == 0:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = returns[i * lag:(i + 1) * lag]
            mean_chunk = chunk.mean()
            deviations = chunk - mean_chunk
            cumdev = np.cumsum(deviations)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1) if chunk.std(ddof=1) > 0 else 1e-10
            rs_chunk.append(R / S)

        if rs_chunk:
            mean_rs = np.mean(rs_chunk)
            if mean_rs > 0:  # Skip lags where R/S collapsed to zero
                rs_values.append((lag, mean_rs))

    if len(rs_values) < 3:
        return HurstResult(hurst_exponent=0.5)

    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    # Linear regression: log(R/S) = H * log(lag) + c
    H = np.polyfit(log_lags, log_rs, 1)[0]

    return HurstResult(hurst_exponent=float(np.clip(H, 0.0, 1.0)))


def half_life(series: np.ndarray) -> HalfLifeResult:
    """
    Estimate the half-life of mean reversion using AR(1).

    Uses OLS regression: Δy_t = β * y_{t-1} + ε
    Half-life = -ln(2) / ln(1 + β)

    Args:
        series: 1D mean-reverting series (e.g., spread or residuals).

    Returns:
        HalfLifeResult with half-life in periods.
    """
    series = np.asarray(series).ravel()
    y = series[1:] - series[:-1]
    x = series[:-1]

    # OLS: y = beta * x
    x_mean = x.mean()
    x_centered = x - x_mean
    beta = np.dot(x_centered, y) / np.dot(x_centered, x_centered)

    if beta >= 0 or beta <= -1:
        # Not mean-reverting or divergent
        return HalfLifeResult(half_life=float("inf"), beta=float(beta))

    hl = -np.log(2) / np.log(1 + beta)
    return HalfLifeResult(half_life=float(hl), beta=float(beta))


def variance_ratio(
    series: np.ndarray,
    lag: int,
) -> VarianceRatioResult:
    """
    Variance ratio test.

    Port of varianceRatio() from statsUtilities.py.

    VR < 1: mean-reverting
    VR = 1: random walk
    VR > 1: trending

    Args:
        series: 1D price series.
        lag: Lag period for comparison.

    Returns:
        VarianceRatioResult.
    """
    x = np.asarray(series).ravel()
    n = len(x)

    if lag < 1 or n < lag + 2:
        return VarianceRatioResult(ratio=1.0, lag=lag)

    # VR = Var(x_{t+lag} - x_t) / (lag * Var(x_{t+1} - x_t))
    diffs_1 = np.diff(x)
    var_1 = np.var(diffs_1, ddof=1) if len(diffs_1) > 1 else 1e-10

    diffs_lag = x[lag:] - x[:-lag]
    var_lag = np.var(diffs_lag, ddof=1) if len(diffs_lag) > 1 else 1e-10

    ratio = var_lag / (lag * var_1) if var_1 > 0 else 1.0

    return VarianceRatioResult(ratio=float(ratio), lag=lag)


def variance_profile(
    series: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """
    Compute variance ratio profile across multiple lags.

    Port of varianceProfile() from statsUtilities.py.

    Returns array of VR values for lags 2..max_lag+1.
    """
    x = np.asarray(series).ravel()
    n = len(x)

    diff_var = np.var(np.diff(x), ddof=1)
    if diff_var < 1e-15:
        return np.ones(max_lag)

    profile = np.zeros(max_lag)
    for i in range(max_lag):
        lag = i + 2
        if lag >= n:
            profile[i] = 1.0
            continue
        diffs = x[lag:] - x[:-lag]
        var_lag = np.var(diffs, ddof=1) if len(diffs) > 1 else 0
        profile[i] = var_lag / (lag * diff_var) if diff_var > 0 else 1.0

    return profile


# ═══════════════════════════════════════════════════════════════════
#  Batch Test Runner
# ═══════════════════════════════════════════════════════════════════

def run_stationarity_tests(
    series: np.ndarray,
    significance: float = 0.05,
    run_hurst: bool = True,
    run_half_life: bool = True,
) -> StationarityResult:
    """
    Run a full suite of stationarity tests on a series.

    Args:
        series: 1D time series.
        significance: Significance level.
        run_hurst: Also compute Hurst exponent.
        run_half_life: Also compute half-life.

    Returns:
        StationarityResult with all test outcomes.
    """
    result = StationarityResult()
    series = np.asarray(series).ravel()

    # ADF
    result.adf = adf_test(series, significance=significance)

    # Durbin-Watson on first differences
    diffs = np.diff(series)
    if len(diffs) > 1:
        result.durbin_watson_stat = durbin_watson_test(diffs)

    # Ljung-Box
    if len(diffs) > 10:
        stat, pval = ljung_box_test(diffs, lags=1)
        result.ljung_box_stat = stat
        result.ljung_box_pvalue = pval

    # Hurst
    if run_hurst and len(series) >= 20:
        result.hurst = hurst_exponent(series)

    # Half-life
    if run_half_life and len(series) >= 10:
        result.half_life = half_life(series)

    return result
