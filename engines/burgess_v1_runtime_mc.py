"""
ENGINE 1: Burgess Statistical Arbitrage — Production Implementation.

Complete cointegration/mean-reversion pipeline:

    Phase 1 — Full Universe Scan
        Every asset as target, stepwise basket selection via vectorized
        correlation, ridge regression, ADF stationarity via statsmodels.

    Phase 2 — Monte Carlo Significance Surface
        Generate random walk universes matching real dimensions, run same
        stepwise process, build null ADF t-value distribution. Only baskets
        beating the simulated critical values are truly significant.

    Phase 3 — Variance Ratio Profile Analysis
        Multi-lag VR profiles on residuals, compared against null VR
        covariance surface using Mahalanobis distance. Independent second
        filter for mean-reversion quality.

    Phase 4 — Train/Test Split + Out-of-Sample Validation
        Chronological split. Phases 1-3 on training data only. Re-estimate
        betas on test period using basket compositions from training. Check
        persistence of cointegration relationship.

    Phase 5 — Walk-Forward Optimization
        Rolling estimation window: estimate on [t-W, t], generate signals
        on [t, t+S], roll forward. Collects OOS P&L from each window.

    Phase 6 — Backtesting
        Z-score signal generation, entry/exit execution with position
        tracking, transaction cost modeling, equity curve construction.

    Phase 7 — Performance Metrics
        Sharpe, Sortino, Calmar, max drawdown, win rate, avg hold time,
        turnover, profit factor.

Ported and productionized from statsUtilities.py (stepwise regression,
VR profiles, Monte Carlo surfaces, Mahalanobis distance) and the
skeleton cointegration engine.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance as sp_distance
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BurgessParams:
    """All tunable parameters for the full Burgess pipeline."""

    # ── Scan ────────────────────────────────────────────────────────────
    n_vars: int = 3                         # basket size (regressors per target)
    ridge_alpha: float = 0.01               # ridge penalty

    # ── Significance ────────────────────────────────────────────────────
    mc_samples: int = 5000                  # Monte Carlo random walk samples
    mc_seed: int | None = 1234              # reproducibility
    significance_levels: list[int] = field(
        default_factory=lambda: [10, 5, 1]  # percentile thresholds
    )
    significance_threshold: int = 5         # which level to use as gate

    # ── VR Profile ──────────────────────────────────────────────────────
    vr_max_lag: int = 100                   # max lag for variance profile
    vr_lags: list[int] = field(
        default_factory=lambda: [10, 25, 50, 75, 100]
    )
    vr_mahalanobis_threshold: float = 3.0   # reject if distance < threshold
                                            # (closer to null = more random-walk-like)

    # ── Train/Test ──────────────────────────────────────────────────────
    train_frac: float = 0.70                # chronological split
    min_train_obs: int = 252                # minimum training observations
    min_test_obs: int = 63                  # minimum test observations

    # ── OOS Validation ──────────────────────────────────────────────────
    oos_adf_significance: float = 0.10      # looser threshold for OOS persistence

    # ── Walk-Forward ────────────────────────────────────────────────────
    wf_estimation_window: int = 504         # ~2 years
    wf_signal_window: int = 63              # ~1 quarter
    wf_step_size: int = 21                  # ~1 month roll

    # ── Signals ─────────────────────────────────────────────────────────
    zscore_lookback: int = 63               # rolling z-score window
    entry_threshold: float = 2.0            # enter at |z| > entry
    exit_threshold: float = 0.5             # exit at |z| < exit
    stop_loss_threshold: float = 4.0        # stop if |z| > stop

    # ── Backtest ────────────────────────────────────────────────────────
    transaction_cost_bps: float = 10.0      # one-way cost in basis points
    slippage_bps: float = 5.0               # one-way slippage in basis points
    initial_capital: float = 1_000_000.0
    max_position_pct: float = 0.10          # max capital per basket

    # ── Output ──────────────────────────────────────────────────────────
    top_k: int = 20                         # how many baskets to carry forward


@dataclass
class RegressionResult:
    """Output of a single stepwise regression."""
    target_idx: int
    basket_indices: list[int]
    betas: np.ndarray           # including intercept at [0]
    residuals: np.ndarray
    r_squared: float
    adj_r_squared: float


@dataclass
class StationarityResult:
    """ADF + half-life for a residual series."""
    adf_t_value: float
    adf_p_value: float
    half_life: float
    hurst_exponent: float


@dataclass
class VRProfileResult:
    """Variance ratio profile analysis."""
    profile: np.ndarray                 # VR at each lag
    mahalanobis_distances: dict[int, float]  # per VR lag window
    is_significant: bool                # passes VR filter


@dataclass
class CandidateBasket:
    """A basket that passed in-sample stationarity."""
    target_idx: int
    basket_indices: list[int]
    regression: RegressionResult
    stationarity: StationarityResult
    vr_profile: VRProfileResult | None = None
    mc_percentile: float = 100.0        # where ADF t-value falls in MC null
    rank: int = 0


@dataclass
class OOSValidation:
    """Out-of-sample validation results for a basket."""
    basket: CandidateBasket
    oos_regression: RegressionResult
    oos_stationarity: StationarityResult
    passed: bool


@dataclass
class TradeRecord:
    """Single completed round-trip trade."""
    basket_idx: int
    entry_date_idx: int
    exit_date_idx: int
    entry_zscore: float
    exit_zscore: float
    direction: Literal["long", "short"]
    pnl: float
    pnl_pct: float
    hold_days: int
    cost: float


@dataclass
class BacktestResult:
    """Full backtest output for a single basket."""
    basket: CandidateBasket
    equity_curve: np.ndarray
    daily_returns: np.ndarray
    trades: list[TradeRecord]
    positions: np.ndarray           # +1, -1, 0 for each day
    zscore_series: np.ndarray


@dataclass
class PerformanceMetrics:
    """Summary statistics for a backtest."""
    basket_idx: int
    target_name: str
    basket_names: list[str]
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int      # days
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    avg_hold_days: float
    total_costs: float
    turnover: float                 # annual


@dataclass
class BurgessOutput:
    """Complete pipeline output."""
    # Phase 1: Scan
    all_regressions: list[RegressionResult]
    all_stationarity: list[StationarityResult]

    # Phase 2: Significance
    mc_critical_values: dict[int, float]    # {5: -4.23, 1: -4.89, ...}
    significant_baskets: list[CandidateBasket]

    # Phase 3: VR Profile
    vr_filtered_baskets: list[CandidateBasket]

    # Phase 4: OOS Validation
    oos_results: list[OOSValidation]
    oos_passed_baskets: list[CandidateBasket]

    # Phase 5-6: Backtest
    backtest_results: list[BacktestResult]

    # Phase 7: Metrics
    performance: list[PerformanceMetrics]

    # Timing
    phase_times: dict[str, float]

    # Summary
    n_assets: int
    n_obs: int
    n_train: int
    n_test: int


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: FULL UNIVERSE SCAN
# ═════════════════════════════════════════════════════════════════════════════

def corr2_coeff(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Row-wise correlation coefficient between two 2D arrays.

    Vectorized computation — no Python loops. Each row of A is correlated
    with each row of B.

    Args:
        A: (m, n) array — m row-vectors of length n.
        B: (p, n) array — p row-vectors of length n.

    Returns:
        (m, p) correlation matrix.
    """
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)
    denom = np.sqrt(np.dot(ssA[:, None], ssB[None]))
    denom = np.where(denom == 0, 1e-10, denom)
    return np.dot(A_mA, B_mB.T) / denom


def stepwise_regression(
    prices: np.ndarray,
    target_idx: int,
    n_vars: int,
    alpha: float = 0.01,
) -> RegressionResult:
    """
    Stepwise basket selection via greedy correlation maximization.

    For a given target asset, iteratively selects the asset most correlated
    with the current residual, adds it to the basket, re-regresses, and
    repeats. Uses vectorized correlation (corr2_coeff) and sklearn Ridge.

    Args:
        prices: (T, N) price matrix.
        target_idx: Column index of the target asset.
        n_vars: Number of basket members to select.
        alpha: Ridge regularization parameter.

    Returns:
        RegressionResult with target, basket indices, betas, residuals, R².
    """
    T, N = prices.shape
    y = prices[:, target_idx]
    current_residual = y.copy()

    mask = np.zeros(N, dtype=int)
    mask[target_idx] = 1
    basket_indices = []
    X_basket = np.empty((T, 0))

    for _ in range(min(n_vars, N - 1)):
        # Correlate every available asset with current residual
        correlations = np.ma.array(
            np.abs(corr2_coeff(prices.T, current_residual[None, :])).ravel(),
            mask=mask,
        )
        if correlations.count() == 0:
            break

        best_idx = int(np.argmax(correlations))
        mask[best_idx] = 1
        basket_indices.append(best_idx)

        # Rebuild X with intercept and all selected assets
        X_basket = np.column_stack([X_basket, prices[:, best_idx]])
        X_with_intercept = np.column_stack([np.ones(T), X_basket])

        # Ridge regression
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(X_with_intercept, y)
        current_residual = y - clf.predict(X_with_intercept)

    if len(basket_indices) == 0:
        # Degenerate case — no basket found
        return RegressionResult(
            target_idx=target_idx,
            basket_indices=[],
            betas=np.array([y.mean()]),
            residuals=y - y.mean(),
            r_squared=0.0,
            adj_r_squared=0.0,
        )

    # Final regression statistics
    betas = clf.coef_
    residuals = current_residual
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p = len(basket_indices)
    adj_r2 = 1 - (1 - r2) * (T - 1) / (T - p - 1) if T > p + 1 else r2

    return RegressionResult(
        target_idx=target_idx,
        basket_indices=basket_indices,
        betas=betas,
        residuals=residuals,
        r_squared=r2,
        adj_r_squared=adj_r2,
    )


def compute_stationarity(residuals: np.ndarray) -> StationarityResult:
    """
    Full stationarity test battery on a residual series.

    Uses statsmodels ADF (not hand-rolled) for proper p-values with
    automatic lag selection. Computes Hurst exponent via R/S analysis
    and half-life via OU mean-reversion speed.

    Args:
        residuals: 1D residual series.

    Returns:
        StationarityResult with ADF t-value, p-value, half-life, Hurst.
    """
    from statsmodels.tsa.stattools import adfuller

    # ADF test with automatic lag selection
    try:
        adf_result = adfuller(residuals, autolag="AIC")
        adf_t = float(adf_result[0])
        adf_p = float(adf_result[1])
    except Exception:
        adf_t, adf_p = 0.0, 1.0

    # Half-life from OU process: dy = -theta * y_lag + eps
    hl = _half_life(residuals)

    # Hurst exponent via R/S analysis
    h = _hurst_rs(residuals)

    return StationarityResult(
        adf_t_value=adf_t,
        adf_p_value=adf_p,
        half_life=hl,
        hurst_exponent=h,
    )


def _half_life(s: np.ndarray) -> float:
    """OU half-life: ln(2) / theta where dy = -theta * y_lag."""
    dy = np.diff(s)
    y_lag = s[:-1]
    X = np.column_stack([np.ones(len(dy)), y_lag])
    try:
        b, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf
    theta = -b[1]
    return np.log(2) / theta if theta > 1e-10 else np.inf


def _hurst_rs(s: np.ndarray) -> float:
    """Hurst exponent via rescaled range (R/S) analysis."""
    n = len(s)
    if n < 40:
        return 0.5
    max_sz = n // 4
    min_sz = 10
    if max_sz <= min_sz:
        return 0.5

    sizes = np.unique(np.geomspace(min_sz, max_sz, min(20, max_sz - min_sz)).astype(int))
    log_sizes, log_rs = [], []

    for sz in sizes:
        n_chunks = n // sz
        if n_chunks < 2:
            continue
        rs_values = []
        for i in range(n_chunks):
            chunk = s[i * sz : (i + 1) * sz]
            centered = chunk - chunk.mean()
            cumdev = np.cumsum(centered)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 1e-10:
                rs_values.append(R / S)
        if rs_values:
            log_sizes.append(np.log(sz))
            log_rs.append(np.log(np.mean(rs_values)))

    if len(log_sizes) < 3:
        return 0.5

    from scipy import stats as sp_stats
    slope, _, _, _, _ = sp_stats.linregress(log_sizes, log_rs)
    return float(np.clip(slope, 0.0, 1.0))


def full_universe_scan(
    prices: np.ndarray,
    params: BurgessParams,
    progress_callback=None,
) -> tuple[list[RegressionResult], list[StationarityResult]]:
    """
    Phase 1: Run stepwise regression on every asset as target.

    Args:
        prices: (T, N) price matrix (training period).
        params: Pipeline parameters.
        progress_callback: Optional callable(i, N) for progress reporting.

    Returns:
        Tuple of (regressions, stationarity_results) — one per asset.
    """
    T, N = prices.shape
    regressions = []
    stationarity_results = []

    for i in range(N):
        reg = stepwise_regression(prices, i, params.n_vars, params.ridge_alpha)
        regressions.append(reg)

        stat = compute_stationarity(reg.residuals)
        stationarity_results.append(stat)

        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, N)

    return regressions, stationarity_results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: MONTE CARLO SIGNIFICANCE SURFACE
# ═════════════════════════════════════════════════════════════════════════════

def generate_random_walk_universe(
    n_obs: int,
    n_assets: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a universe of independent random walks.

    Each path is cumsum of {-1, 0, +1} steps. Matches the dimensionality
    of the real universe for valid critical value comparison.

    Args:
        n_obs: Number of time steps.
        n_assets: Number of independent paths.
        seed: Random seed for reproducibility.

    Returns:
        (n_obs, n_assets) array of random walk paths.
    """
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 0, 1], size=(n_obs, n_assets))
    return np.cumsum(steps, axis=0)


def build_mc_null_distribution(
    n_obs: int,
    n_assets: int,
    n_vars: int,
    n_samples: int,
    ridge_alpha: float = 0.01,
    seed: int | None = None,
    progress_callback=None,
) -> np.ndarray:
    """
    Build Monte Carlo null distribution of ADF t-values.

    Generates random walk universes, runs the same stepwise regression
    process on each, collects ADF t-values. The resulting sorted array
    IS the null distribution — real baskets must beat it to be significant.

    Args:
        n_obs: Observations per simulation (match training set size).
        n_assets: Assets per simulation (match universe size).
        n_vars: Basket size (match pipeline setting).
        n_samples: Total ADF t-values to collect.
        ridge_alpha: Ridge penalty (match pipeline setting).
        seed: Random seed.
        progress_callback: Optional callable(collected, total).

    Returns:
        Sorted array of n_samples ADF t-values from the null.
    """
    from statsmodels.tsa.stattools import adfuller

    adf_values = np.zeros(n_samples)
    collected = 0
    current_seed = seed

    while collected < n_samples:
        paths = generate_random_walk_universe(n_obs, n_assets, seed=current_seed)
        current_seed = None  # only seed the first batch

        for i in range(n_assets):
            if collected >= n_samples:
                break

            reg = stepwise_regression(paths, i, n_vars, ridge_alpha)
            try:
                adf_result = adfuller(reg.residuals, autolag="AIC")
                adf_values[collected] = adf_result[0]
            except Exception:
                adf_values[collected] = 0.0
            collected += 1

            if progress_callback and collected % 500 == 0:
                progress_callback(collected, n_samples)

    adf_values.sort()
    return adf_values


def get_critical_values(
    null_distribution: np.ndarray,
    levels: list[int] = (10, 5, 1),
) -> dict[int, float]:
    """
    Extract critical values from a sorted null distribution.

    Args:
        null_distribution: Sorted array of test statistics from MC.
        levels: Percentile levels (e.g., 5 means the 5th percentile).

    Returns:
        Dict mapping level → critical value.
    """
    n = len(null_distribution)
    indices = np.arange(n)
    return {
        level: float(np.interp(n * level / 100.0 - 1, indices, null_distribution))
        for level in levels
    }


def filter_by_significance(
    regressions: list[RegressionResult],
    stationarity_results: list[StationarityResult],
    critical_values: dict[int, float],
    threshold_level: int = 5,
) -> list[CandidateBasket]:
    """
    Keep only baskets whose ADF t-value beats the MC critical value.

    This is the key step that eliminates data-mining artifacts. Standard
    ADF critical values don't account for the selection bias inherent in
    stepwise regression across a large universe. The MC surface does.

    Args:
        regressions: All regression results from Phase 1.
        stationarity_results: Corresponding stationarity results.
        critical_values: From build_mc_null_distribution.
        threshold_level: Which percentile to use as gate.

    Returns:
        List of CandidateBaskets that passed.
    """
    cutoff = critical_values[threshold_level]
    candidates = []

    for reg, stat in zip(regressions, stationarity_results):
        if len(reg.basket_indices) == 0:
            continue
        if stat.adf_t_value < cutoff:
            # Compute percentile rank within the null
            pct = _percentile_rank(stat.adf_t_value, cutoff, critical_values)
            candidates.append(CandidateBasket(
                target_idx=reg.target_idx,
                basket_indices=reg.basket_indices,
                regression=reg,
                stationarity=stat,
                mc_percentile=pct,
            ))

    # Sort by ADF t-value (most negative = strongest)
    candidates.sort(key=lambda c: c.stationarity.adf_t_value)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return candidates


def _percentile_rank(value: float, cutoff_5: float, cvs: dict) -> float:
    """Approximate percentile from critical values."""
    if 1 in cvs and value < cvs[1]:
        return 0.5
    if 5 in cvs and value < cvs[5]:
        return 2.5
    if 10 in cvs and value < cvs[10]:
        return 7.5
    return 50.0


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: VARIANCE RATIO PROFILE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def variance_profile(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute the variance ratio profile VR(τ) for τ = 2..max_lag+1.

    For a random walk, VR(τ) ≈ 1 for all τ. Mean-reverting series
    have VR(τ) < 1 (declining profile). Trending series have VR(τ) > 1.

    Args:
        x: 1D time series (residuals).
        max_lag: Maximum lag to compute.

    Returns:
        Array of length max_lag with VR values.
    """
    n = len(x)
    x_diff_var = (n - 2) * np.var(np.diff(x))
    if x_diff_var < 1e-15:
        return np.ones(max_lag)

    profile = np.zeros(max_lag)
    for i in range(2, max_lag + 2):
        shifted = x[i:] - x[:-i]
        profile[i - 2] = (n - i - 1) * np.var(shifted) / (i * x_diff_var)

    return profile


def build_vr_null_surface(
    n_obs: int,
    n_assets: int,
    n_vars: int,
    vr_lags: list[int],
    max_lag: int,
    ridge_alpha: float = 0.01,
    seed: int | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Build the null VR covariance surface from random walks.

    For each VR lag window, computes the covariance matrix and its inverse
    from the VR profiles of stepwise regression residuals on random walks.
    These define the Mahalanobis metric for significance testing.

    Args:
        n_obs: Observations (match training set).
        n_assets: Universe size (match real universe).
        n_vars: Basket size.
        vr_lags: List of lag windows to compute surfaces for.
        max_lag: Maximum lag for VR profile.
        ridge_alpha: Ridge penalty.
        seed: Random seed.

    Returns:
        Dict mapping lag → (covariance_matrix, inverse_covariance_matrix).
    """
    paths = generate_random_walk_universe(n_obs, n_assets, seed=seed)

    # Get residuals for every asset in the random walk universe
    residuals = []
    for i in range(n_assets):
        reg = stepwise_regression(paths, i, n_vars, ridge_alpha)
        residuals.append(reg.residuals)
    residuals = np.column_stack(residuals)

    # Compute VR profiles for each residual series
    vr_profiles = np.column_stack([
        variance_profile(residuals[:, i], max_lag)
        for i in range(residuals.shape[1])
    ])

    # Build covariance at each lag window
    surfaces = {}
    for lag in vr_lags:
        if lag > max_lag:
            continue
        vr_sub = vr_profiles[:lag, :]  # (lag, n_assets)
        cov = np.cov(vr_sub)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        try:
            cov_inv = LA.inv(cov)
        except LA.LinAlgError:
            cov_inv = LA.pinv(cov)
        surfaces[lag] = (cov, cov_inv)

    return surfaces


def compute_vr_significance(
    residuals: np.ndarray,
    vr_surfaces: dict[int, tuple[np.ndarray, np.ndarray]],
    max_lag: int,
) -> VRProfileResult:
    """
    Test a residual series against the null VR covariance surface.

    Computes the Mahalanobis distance of the residual's VR profile
    from the null distribution center. Higher distance = more different
    from random walk = more likely mean-reverting.

    Args:
        residuals: 1D residual series.
        vr_surfaces: From build_vr_null_surface.
        max_lag: Maximum lag for VR profile.

    Returns:
        VRProfileResult with profile, distances, significance flag.
    """
    profile = variance_profile(residuals, max_lag)
    distances = {}

    for lag, (cov, cov_inv) in vr_surfaces.items():
        if lag > len(profile):
            continue
        vr_sub = profile[:lag]
        # Center around 1.0 (random walk expectation)
        vr_centered = vr_sub - 1.0
        try:
            d = float(sp_distance.mahalanobis(
                vr_centered, np.zeros(lag), cov_inv
            ))
        except Exception:
            d = 0.0
        distances[lag] = d

    return VRProfileResult(
        profile=profile,
        mahalanobis_distances=distances,
        is_significant=True,  # set by caller
    )


def filter_by_vr_profile(
    candidates: list[CandidateBasket],
    vr_surfaces: dict[int, tuple[np.ndarray, np.ndarray]],
    params: BurgessParams,
) -> list[CandidateBasket]:
    """
    Phase 3: Filter candidates using VR profile Mahalanobis distance.

    Baskets whose residuals look like random walks (low Mahalanobis distance)
    are discarded. Only baskets that are clearly different from the null
    VR structure survive.

    Args:
        candidates: From Phase 2.
        vr_surfaces: Null VR covariance surfaces.
        params: Pipeline parameters.

    Returns:
        Filtered list of CandidateBaskets with VR profile attached.
    """
    passed = []
    for basket in candidates:
        vr = compute_vr_significance(
            basket.regression.residuals, vr_surfaces, params.vr_max_lag
        )
        basket.vr_profile = vr

        # Check if any lag window shows significant departure from null
        max_distance = max(vr.mahalanobis_distances.values()) if vr.mahalanobis_distances else 0
        vr.is_significant = max_distance >= params.vr_mahalanobis_threshold

        if vr.is_significant:
            passed.append(basket)

    return passed


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: TRAIN/TEST SPLIT + OUT-OF-SAMPLE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def split_train_test(
    prices: np.ndarray,
    train_frac: float,
    min_train: int = 252,
    min_test: int = 63,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Chronological train/test split.

    Args:
        prices: Full (T, N) price matrix.
        train_frac: Fraction for training.
        min_train: Minimum training observations.
        min_test: Minimum test observations.

    Returns:
        (train_prices, test_prices, split_index).
    """
    T = prices.shape[0]
    split = max(min_train, min(T - min_test, int(T * train_frac)))
    return prices[:split], prices[split:], split


def validate_oos(
    test_prices: np.ndarray,
    basket: CandidateBasket,
    params: BurgessParams,
) -> OOSValidation:
    """
    Re-estimate regression on test data using in-sample basket composition.

    Uses the same basket members (target + basket indices) identified in
    training, but re-fits betas on the test period. Then tests if
    cointegration persists out-of-sample.

    Args:
        test_prices: (T_test, N) price matrix.
        basket: CandidateBasket from training.
        params: Pipeline parameters.

    Returns:
        OOSValidation with regression, stationarity, pass/fail.
    """
    T = test_prices.shape[0]
    y = test_prices[:, basket.target_idx]

    if not basket.basket_indices:
        dummy_reg = RegressionResult(basket.target_idx, [], np.array([0]), y, 0, 0)
        dummy_stat = StationarityResult(0, 1.0, np.inf, 0.5)
        return OOSValidation(basket, dummy_reg, dummy_stat, False)

    X_basket = test_prices[:, basket.basket_indices]
    X_with_intercept = np.column_stack([np.ones(T), X_basket])

    clf = Ridge(alpha=params.ridge_alpha, fit_intercept=False)
    clf.fit(X_with_intercept, y)
    residuals = y - clf.predict(X_with_intercept)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p = len(basket.basket_indices)
    adj_r2 = 1 - (1 - r2) * (T - 1) / (T - p - 1) if T > p + 1 else r2

    oos_reg = RegressionResult(
        target_idx=basket.target_idx,
        basket_indices=basket.basket_indices,
        betas=clf.coef_,
        residuals=residuals,
        r_squared=r2,
        adj_r_squared=adj_r2,
    )

    oos_stat = compute_stationarity(residuals)
    passed = oos_stat.adf_p_value < params.oos_adf_significance

    return OOSValidation(
        basket=basket,
        oos_regression=oos_reg,
        oos_stationarity=oos_stat,
        passed=passed,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: WALK-FORWARD OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════

def walk_forward_windows(
    T: int,
    estimation_window: int,
    signal_window: int,
    step_size: int,
) -> list[tuple[int, int, int, int]]:
    """
    Generate walk-forward window boundaries.

    Each window is (est_start, est_end, sig_start, sig_end) where:
        - [est_start, est_end) is the estimation period
        - [sig_start, sig_end) is the signal/trading period

    Args:
        T: Total observations.
        estimation_window: Size of estimation window.
        signal_window: Size of signal window.
        step_size: How far to roll forward each step.

    Returns:
        List of (est_start, est_end, sig_start, sig_end) tuples.
    """
    windows = []
    est_start = 0
    while est_start + estimation_window + signal_window <= T:
        est_end = est_start + estimation_window
        sig_start = est_end
        sig_end = min(sig_start + signal_window, T)
        windows.append((est_start, est_end, sig_start, sig_end))
        est_start += step_size
    return windows


def walk_forward_single_basket(
    prices: np.ndarray,
    target_idx: int,
    basket_indices: list[int],
    params: BurgessParams,
) -> tuple[np.ndarray, np.ndarray, list[TradeRecord]]:
    """
    Walk-forward backtest for a single basket composition.

    For each window: estimate betas on estimation period, generate
    z-score signals on signal period, execute trades.

    Args:
        prices: Full (T, N) price matrix.
        target_idx: Target asset index.
        basket_indices: Basket member indices.
        params: Pipeline parameters.

    Returns:
        (daily_pnl, positions, trades) arrays spanning the full
        signal periods.
    """
    T = prices.shape[0]
    windows = walk_forward_windows(
        T, params.wf_estimation_window,
        params.wf_signal_window, params.wf_step_size,
    )

    if not windows:
        return np.zeros(0), np.zeros(0), []

    # Allocate full-length arrays
    all_pnl = np.zeros(T)
    all_positions = np.zeros(T)
    all_zscores = np.full(T, np.nan)
    all_trades = []

    for est_start, est_end, sig_start, sig_end in windows:
        # ── Estimation: fit betas ───────────────────────────────────
        est_prices = prices[est_start:est_end]
        y_est = est_prices[:, target_idx]
        X_est = np.column_stack([
            np.ones(est_end - est_start),
            est_prices[:, basket_indices],
        ])
        clf = Ridge(alpha=params.ridge_alpha, fit_intercept=False)
        clf.fit(X_est, y_est)

        est_residuals = y_est - clf.predict(X_est)

        # ── Signal: apply betas to signal period ────────────────────
        sig_prices = prices[sig_start:sig_end]
        y_sig = sig_prices[:, target_idx]
        X_sig = np.column_stack([
            np.ones(sig_end - sig_start),
            sig_prices[:, basket_indices],
        ])
        sig_residuals = y_sig - clf.predict(X_sig)

        # Z-scores using estimation period statistics
        # Warm up with last zscore_lookback of estimation residuals
        combined = np.concatenate([
            est_residuals[-params.zscore_lookback:],
            sig_residuals,
        ])
        lb = params.zscore_lookback
        zscores = np.zeros(len(sig_residuals))
        for i in range(len(sig_residuals)):
            window = combined[i : i + lb]
            mu, sigma = window.mean(), window.std()
            if sigma > 1e-10:
                zscores[i] = (combined[i + lb] - mu) / sigma

        all_zscores[sig_start:sig_end] = zscores

        # ── Execute trades ──────────────────────────────────────────
        position = 0  # +1 long spread, -1 short spread, 0 flat
        entry_idx = -1
        entry_z = 0.0

        for i in range(len(zscores)):
            z = zscores[i]
            global_idx = sig_start + i

            if position == 0:
                # Check for entry
                if z >= params.entry_threshold:
                    position = -1  # short the spread (sell target, buy basket)
                    entry_idx = global_idx
                    entry_z = z
                elif z <= -params.entry_threshold:
                    position = 1  # long the spread (buy target, sell basket)
                    entry_idx = global_idx
                    entry_z = z
            else:
                # Check for exit
                should_exit = False
                if position == 1 and z >= -params.exit_threshold:
                    should_exit = True
                elif position == -1 and z <= params.exit_threshold:
                    should_exit = True
                elif abs(z) >= params.stop_loss_threshold:
                    should_exit = True

                if should_exit:
                    # Record trade
                    hold_days = global_idx - entry_idx
                    if hold_days > 0 and entry_idx >= 0:
                        # P&L from spread return
                        spread_return = (
                            sig_residuals[i] - sig_residuals[i - hold_days]
                            if i >= hold_days else 0
                        )
                        cost_bps = (params.transaction_cost_bps + params.slippage_bps) * 2  # round trip
                        cost = abs(spread_return) * cost_bps / 10000

                        pnl = position * spread_return - cost
                        pnl_pct = pnl / abs(sig_residuals[max(0, i - hold_days)] + 1e-10)

                        all_trades.append(TradeRecord(
                            basket_idx=target_idx,
                            entry_date_idx=entry_idx,
                            exit_date_idx=global_idx,
                            entry_zscore=entry_z,
                            exit_zscore=z,
                            direction="long" if position == 1 else "short",
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            hold_days=hold_days,
                            cost=cost,
                        ))
                    position = 0
                    entry_idx = -1

            all_positions[global_idx] = position

            # Daily P&L from position × residual change
            if global_idx > 0 and position != 0:
                # Use log returns of the spread
                if global_idx < len(prices):
                    daily_spread_ret = sig_residuals[i] - (
                        sig_residuals[i - 1] if i > 0 else est_residuals[-1]
                    )
                    all_pnl[global_idx] = position * daily_spread_ret

    return all_pnl, all_positions, all_trades


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 6: BACKTESTING
# ═════════════════════════════════════════════════════════════════════════════

def backtest_basket(
    prices: np.ndarray,
    basket: CandidateBasket,
    params: BurgessParams,
) -> BacktestResult:
    """
    Full backtest for a single basket via walk-forward.

    Args:
        prices: Full (T, N) price matrix.
        basket: CandidateBasket to backtest.
        params: Pipeline parameters.

    Returns:
        BacktestResult with equity curve, returns, trades, positions.
    """
    daily_pnl, positions, trades = walk_forward_single_basket(
        prices,
        basket.target_idx,
        basket.basket_indices,
        params,
    )

    # Normalize P&L to capital allocation
    allocation = params.initial_capital * params.max_position_pct
    if allocation > 0:
        daily_returns = daily_pnl / allocation
    else:
        daily_returns = daily_pnl

    # Build equity curve
    equity_curve = params.initial_capital * params.max_position_pct * (
        1 + np.cumsum(daily_returns)
    )

    # Z-score series for the full period (for visualization)
    T = prices.shape[0]
    y = prices[:, basket.target_idx]
    X = np.column_stack([np.ones(T), prices[:, basket.basket_indices]])
    clf = Ridge(alpha=params.ridge_alpha, fit_intercept=False)
    clf.fit(X, y)
    full_residuals = y - clf.predict(X)

    zscores = np.zeros(T)
    lb = params.zscore_lookback
    for i in range(lb, T):
        w = full_residuals[i - lb : i]
        mu, sigma = w.mean(), w.std()
        if sigma > 1e-10:
            zscores[i] = (full_residuals[i] - mu) / sigma

    return BacktestResult(
        basket=basket,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        trades=trades,
        positions=positions,
        zscore_series=zscores,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 7: PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    bt: BacktestResult,
    asset_names: list[str],
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Compute comprehensive performance statistics from a backtest.

    Args:
        bt: BacktestResult from Phase 6.
        asset_names: List mapping column indices to ticker names.
        trading_days_per_year: Annualization factor.

    Returns:
        PerformanceMetrics dataclass.
    """
    rets = bt.daily_returns
    trades = bt.trades

    # Filter to days with actual activity
    active_rets = rets[bt.positions != 0] if np.any(bt.positions != 0) else rets

    # Annual return
    total_ret = np.sum(rets)
    n_days = max(len(rets), 1)
    years = n_days / trading_days_per_year
    annual_return = total_ret / years if years > 0 else 0.0

    # Annual volatility
    annual_vol = np.std(rets) * np.sqrt(trading_days_per_year) if len(rets) > 1 else 1e-10

    # Sharpe ratio
    sharpe = annual_return / annual_vol if annual_vol > 1e-10 else 0.0

    # Sortino ratio (downside deviation)
    neg_rets = rets[rets < 0]
    downside_std = np.std(neg_rets) * np.sqrt(trading_days_per_year) if len(neg_rets) > 1 else 1e-10
    sortino = annual_return / downside_std if downside_std > 1e-10 else 0.0

    # Max drawdown
    cum = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Max drawdown duration
    dd_duration = 0
    max_dd_duration = 0
    for i in range(len(drawdowns)):
        if drawdowns[i] < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # Calmar ratio
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_hold = np.mean([t.hold_days for t in trades])
        total_costs = sum(t.cost for t in trades)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = total_costs = 0.0

    # Turnover (approximate: 2 × n_trades / years)
    turnover = (2 * n_trades) / years if years > 0 else 0.0

    target_name = asset_names[bt.basket.target_idx] if bt.basket.target_idx < len(asset_names) else f"A{bt.basket.target_idx}"
    basket_names = [
        asset_names[i] if i < len(asset_names) else f"A{i}"
        for i in bt.basket.basket_indices
    ]

    return PerformanceMetrics(
        basket_idx=bt.basket.target_idx,
        target_name=target_name,
        basket_names=basket_names,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        n_trades=n_trades,
        avg_hold_days=float(avg_hold),
        total_costs=float(total_costs),
        turnover=turnover,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

class BurgessEngine:
    """
    Full Burgess statistical arbitrage pipeline.

    Orchestrates all seven phases in sequence. Each phase can also be
    called independently for debugging or partial runs.

    Usage:
        engine = BurgessEngine()
        output = engine.run(prices, asset_names, params)
    """

    def run(
        self,
        prices: np.ndarray,
        asset_names: list[str],
        params: BurgessParams | None = None,
        progress_callback=None,
    ) -> BurgessOutput:
        """
        Execute the full Burgess pipeline.

        Args:
            prices: (T, N) price matrix — full history.
            asset_names: List of N ticker names.
            params: Pipeline parameters (defaults if None).
            progress_callback: Optional callable(phase, message).

        Returns:
            BurgessOutput with results from all phases.
        """
        if params is None:
            params = BurgessParams()

        T, N = prices.shape
        phase_times = {}

        def _log(phase: str, msg: str):
            if progress_callback:
                progress_callback(phase, msg)
            logger.info("[%s] %s", phase, msg)

        # ── Split ───────────────────────────────────────────────────────
        train, test, split_idx = split_train_test(
            prices, params.train_frac, params.min_train_obs, params.min_test_obs
        )
        T_train, T_test = train.shape[0], test.shape[0]
        _log("SPLIT", f"Train: {T_train} obs, Test: {T_test} obs (split at {split_idx})")

        # ── Phase 1: Full Universe Scan (training data only) ────────────
        _log("PHASE1", f"Scanning {N} assets × {params.n_vars} vars...")
        t0 = time.perf_counter()

        def _scan_progress(i, n):
            _log("PHASE1", f"  {i}/{n} targets scanned")

        regressions, stationarity = full_universe_scan(train, params, _scan_progress)
        phase_times["scan"] = time.perf_counter() - t0

        n_adf_sig = sum(1 for s in stationarity if s.adf_p_value < 0.05)
        _log("PHASE1", f"  Done: {N} scanned, {n_adf_sig} raw ADF significant (p<0.05)")

        # ── Phase 2: Monte Carlo Significance Surface ───────────────────
        _log("PHASE2", f"Building MC null ({params.mc_samples} samples, "
                        f"{N} assets × {T_train} obs × {params.n_vars} vars)...")
        t0 = time.perf_counter()

        def _mc_progress(collected, total):
            _log("PHASE2", f"  {collected}/{total} ADF values collected")

        null_dist = build_mc_null_distribution(
            n_obs=T_train, n_assets=N, n_vars=params.n_vars,
            n_samples=params.mc_samples, ridge_alpha=params.ridge_alpha,
            seed=params.mc_seed, progress_callback=_mc_progress,
        )
        mc_cvs = get_critical_values(null_dist, params.significance_levels)
        phase_times["mc_surface"] = time.perf_counter() - t0

        _log("PHASE2", f"  Critical values: " +
             ", ".join(f"{k}%={v:.3f}" for k, v in mc_cvs.items()))

        significant = filter_by_significance(
            regressions, stationarity, mc_cvs, params.significance_threshold
        )
        _log("PHASE2", f"  {len(significant)} baskets passed MC significance at {params.significance_threshold}%")

        # ── Phase 3: VR Profile Analysis ────────────────────────────────
        _log("PHASE3", f"Building VR null surface (lags={params.vr_lags})...")
        t0 = time.perf_counter()

        vr_surfaces = build_vr_null_surface(
            n_obs=T_train, n_assets=N, n_vars=params.n_vars,
            vr_lags=params.vr_lags, max_lag=params.vr_max_lag,
            ridge_alpha=params.ridge_alpha, seed=params.mc_seed,
        )

        vr_filtered = filter_by_vr_profile(significant, vr_surfaces, params)
        phase_times["vr_profile"] = time.perf_counter() - t0
        _log("PHASE3", f"  {len(vr_filtered)} baskets passed VR filter "
                        f"(threshold={params.vr_mahalanobis_threshold})")

        # ── Phase 4: Out-of-Sample Validation ───────────────────────────
        _log("PHASE4", f"Validating {min(len(vr_filtered), params.top_k)} baskets OOS...")
        t0 = time.perf_counter()

        top_candidates = vr_filtered[:params.top_k]
        oos_results = [validate_oos(test, b, params) for b in top_candidates]
        oos_passed = [r for r in oos_results if r.passed]
        oos_baskets = [r.basket for r in oos_passed]
        phase_times["oos_validation"] = time.perf_counter() - t0
        _log("PHASE4", f"  {len(oos_passed)}/{len(top_candidates)} passed OOS "
                        f"(ADF p<{params.oos_adf_significance})")

        # ── Phase 5-6: Walk-Forward Backtest ────────────────────────────
        _log("PHASE5-6", f"Backtesting {len(oos_baskets)} baskets (walk-forward)...")
        t0 = time.perf_counter()

        backtest_results = []
        for basket in oos_baskets:
            bt = backtest_basket(prices, basket, params)
            backtest_results.append(bt)
        phase_times["backtest"] = time.perf_counter() - t0
        _log("PHASE5-6", f"  {len(backtest_results)} backtests complete")

        # ── Phase 7: Performance Metrics ────────────────────────────────
        _log("PHASE7", "Computing performance metrics...")
        t0 = time.perf_counter()

        performance = [
            compute_metrics(bt, asset_names)
            for bt in backtest_results
        ]
        # Sort by Sharpe ratio
        performance.sort(key=lambda p: p.sharpe_ratio, reverse=True)
        phase_times["metrics"] = time.perf_counter() - t0

        total_time = sum(phase_times.values())
        _log("DONE", f"Pipeline complete in {total_time:.1f}s")

        return BurgessOutput(
            all_regressions=regressions,
            all_stationarity=stationarity,
            mc_critical_values=mc_cvs,
            significant_baskets=significant,
            vr_filtered_baskets=vr_filtered,
            oos_results=oos_results,
            oos_passed_baskets=oos_baskets,
            backtest_results=backtest_results,
            performance=performance,
            phase_times=phase_times,
            n_assets=N,
            n_obs=T,
            n_train=T_train,
            n_test=T_test,
        )
