"""
Empirical Distribution Approximations (Phase 4.7).

Replaces Gaussian assumptions with numerical approximations to actual
return distributions. The core problem: normal distributions underestimate
tail risk, leading to heuristic corrections (half-Kelly, ad-hoc VaR
multipliers) instead of proper distribution-aware calculations.

This module provides:
1. Multiple fitting strategies (ECDF, KDE, Student-t, Cornish-Fisher,
   Gaussian mixture, EVT composite)
2. A unified interface for pdf/cdf/quantile/moments
3. Application functions: distribution-aware Kelly, VaR/CVaR
4. Diagnostic tools: tail divergence, QQ residuals, Gaussian comparison
5. Generic protocol so any model can plug in its own fitter

The key insight:
    Half-Kelly exists because Gaussian Kelly overestimates optimal sizing
    by ignoring fat tails. If you fit the actual distribution, you get
    the correct Kelly fraction directly — no arbitrary halving needed.

    Gaussian Kelly:  f* = μ / σ²
    Actual Kelly:    f* = argmax E[log(1 + f·R)]  where R ~ actual dist

    The gap between these IS the cost of the Gaussian assumption.

Usage:
    # Fit from return data
    dist = fit_distribution(returns, method="kde")

    # Use anywhere you'd assume Gaussian
    var_95 = dist.quantile(0.05)           # vs norm.ppf(0.05)*σ + μ
    cvar_95 = dist.expected_shortfall(0.05) # vs Gaussian approx
    f_star = optimal_kelly(dist)           # vs μ/σ²

    # Compare to see what Gaussian misses
    diag = gaussian_divergence(dist)
    print(diag.tail_ratio_left)  # how much fatter the left tail is
    print(diag.kelly_correction)  # actual_kelly / gaussian_kelly
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import stats as sp_stats
from scipy import optimize


# ═══════════════════════════════════════════════════════════════════
#  Core Interface
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DistributionMoments:
    """Standard and higher-order moments."""

    mean: float
    variance: float
    std: float
    skewness: float
    kurtosis: float  # excess kurtosis (0 for Gaussian)
    n_samples: int

    @property
    def is_fat_tailed(self) -> bool:
        """Excess kurtosis > 0 indicates heavier tails than Gaussian."""
        return self.kurtosis > 0.5

    @property
    def is_left_skewed(self) -> bool:
        return self.skewness < -0.25

    @property
    def gaussian_inadequacy(self) -> float:
        """
        Rough measure of how poorly Gaussian fits.
        Combines excess kurtosis and skewness into a single metric.
        0 = Gaussian is fine, higher = Gaussian is increasingly wrong.
        """
        return abs(self.skewness) + abs(self.kurtosis) / 2


class DistributionApproximation(ABC):
    """
    Abstract base for all distribution approximations.

    Every implementation provides the same interface so they're
    interchangeable anywhere a Gaussian would have been used.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Implementation name (e.g., 'kde', 'student_t')."""
        ...

    @property
    @abstractmethod
    def moments(self) -> DistributionMoments:
        """Computed moments of the distribution."""
        ...

    @abstractmethod
    def pdf(self, x: np.ndarray | float) -> np.ndarray:
        """Probability density at x."""
        ...

    @abstractmethod
    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        """Cumulative probability P(X <= x)."""
        ...

    @abstractmethod
    def quantile(self, p: np.ndarray | float) -> np.ndarray:
        """Inverse CDF: value at probability p."""
        ...

    @staticmethod
    def _maybe_scalar(result: np.ndarray, was_scalar: bool):
        """Unwrap 1-element array to scalar if input was scalar."""
        if was_scalar and result.ndim > 0 and result.size == 1:
            return float(result.flat[0])
        return result

    def survival(self, x: np.ndarray | float) -> np.ndarray | float:
        """P(X > x) = 1 - CDF(x)."""
        return 1.0 - self.cdf(x)

    def tail_probability(self, threshold: float) -> float:
        """P(X < threshold). For loss thresholds, this is the tail risk."""
        return float(np.atleast_1d(self.cdf(threshold)).flat[0])

    def expected_shortfall(self, alpha: float, n_points: int = 10000) -> float:
        """
        Expected Shortfall (CVaR): E[X | X < quantile(alpha)].

        The average loss in the worst alpha-fraction of outcomes.
        More informative than VaR because it measures HOW BAD the
        tail is, not just where it starts.

        Args:
            alpha: Tail probability (e.g., 0.05 for 5%).
            n_points: Integration resolution.

        Returns:
            Expected value conditional on being in the alpha-tail.
        """
        # Numerical integration over the left tail
        probs = np.linspace(1e-6, alpha, n_points)
        quantiles = self.quantile(probs)
        return float(np.mean(quantiles))

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Generate random samples from this distribution."""
        rng = np.random.RandomState(seed)
        u = rng.uniform(0, 1, size=n)
        return self.quantile(u)

    def log_likelihood(self, data: np.ndarray) -> float:
        """Total log-likelihood of observed data under this distribution."""
        densities = self.pdf(data)
        densities = np.clip(densities, 1e-300, None)
        return float(np.sum(np.log(densities)))


# ═══════════════════════════════════════════════════════════════════
#  Implementation 1: Empirical CDF
# ═══════════════════════════════════════════════════════════════════


class EmpiricalDistribution(DistributionApproximation):
    """
    Direct ECDF with linear interpolation.

    Pros: No assumptions, exact representation of observed data.
    Cons: No smoothing, limited extrapolation beyond observed range.
    Best for: Large samples, no need to extrapolate past observed extremes.
    """

    def __init__(self, data: np.ndarray):
        self._data = np.sort(np.asarray(data).ravel())
        self._n = len(self._data)
        self._ecdf_probs = (np.arange(1, self._n + 1)) / self._n
        self._moments = _compute_moments(self._data)

    @property
    def name(self) -> str:
        return "empirical"

    @property
    def moments(self) -> DistributionMoments:
        return self._moments

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dx = (self._data[-1] - self._data[0]) / (self._n * 2)
        r = (self._cdf_arr(x + dx) - self._cdf_arr(x - dx)) / (2 * dx)
        return self._maybe_scalar(r, scalar)

    def _cdf_arr(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._data, self._ecdf_probs, left=0.0, right=1.0)

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self._maybe_scalar(self._cdf_arr(x), scalar)

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(p) == 0
        p = np.atleast_1d(np.asarray(p, dtype=float))
        return self._maybe_scalar(np.interp(p, self._ecdf_probs, self._data), scalar)


# ═══════════════════════════════════════════════════════════════════
#  Implementation 2: Kernel Density Estimation
# ═══════════════════════════════════════════════════════════════════


class KDEDistribution(DistributionApproximation):
    """
    Gaussian KDE with bandwidth optimization.

    Pros: Smooth, nonparametric, good general-purpose approximation.
    Cons: Can over-smooth tails with default bandwidth.
    Best for: Medium-size samples, need smooth density for integration.
    """

    def __init__(self, data: np.ndarray, bw_method: str | float | None = None):
        self._data = np.asarray(data).ravel()
        self._kde = sp_stats.gaussian_kde(self._data, bw_method=bw_method)
        self._moments = _compute_moments(self._data)

        # Pre-compute CDF on a fine grid for fast quantile lookup
        lo = self._data.min() - 4 * self._data.std()
        hi = self._data.max() + 4 * self._data.std()
        self._grid = np.linspace(lo, hi, 5000)
        self._cdf_grid = np.cumsum(self._kde(self._grid))
        self._cdf_grid /= self._cdf_grid[-1]  # normalize

    @property
    def name(self) -> str:
        return "kde"

    @property
    def moments(self) -> DistributionMoments:
        return self._moments

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self._maybe_scalar(self._kde(x), scalar)

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self._maybe_scalar(np.interp(x, self._grid, self._cdf_grid, left=0.0, right=1.0), scalar)

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(p) == 0
        p = np.atleast_1d(np.asarray(p, dtype=float))
        return self._maybe_scalar(np.interp(p, self._cdf_grid, self._grid), scalar)


# ═══════════════════════════════════════════════════════════════════
#  Implementation 3: Student-t (fitted)
# ═══════════════════════════════════════════════════════════════════


class StudentTDistribution(DistributionApproximation):
    """
    Fitted Student-t distribution.

    Pros: Captures fat tails with a single parameter (df), well-understood.
    Cons: Symmetric — can't capture skewness.
    Best for: Return series with fat tails but roughly symmetric.

    The degrees of freedom parameter controls tail heaviness:
        df → ∞: converges to Gaussian
        df ~ 3-5: typical for daily equity returns
        df < 3: very heavy tails (variance may be infinite)
    """

    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data).ravel()
        # MLE fit: df, loc, scale
        self._df, self._loc, self._scale = sp_stats.t.fit(self._data)
        self._frozen = sp_stats.t(df=self._df, loc=self._loc, scale=self._scale)
        self._moments = _compute_moments(self._data)

    @property
    def name(self) -> str:
        return "student_t"

    @property
    def moments(self) -> DistributionMoments:
        return self._moments

    @property
    def df(self) -> float:
        """Degrees of freedom — lower = fatter tails."""
        return self._df

    @property
    def loc(self) -> float:
        return self._loc

    @property
    def scale(self) -> float:
        return self._scale

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        r = self._frozen.pdf(np.atleast_1d(x))
        return self._maybe_scalar(r, scalar)

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        r = self._frozen.cdf(np.atleast_1d(x))
        return self._maybe_scalar(r, scalar)

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(p) == 0
        r = self._frozen.ppf(np.atleast_1d(p))
        return self._maybe_scalar(r, scalar)


# ═══════════════════════════════════════════════════════════════════
#  Implementation 4: Cornish-Fisher Expansion
# ═══════════════════════════════════════════════════════════════════


class CornishFisherDistribution(DistributionApproximation):
    """
    Gaussian corrected by skewness and kurtosis (Cornish-Fisher expansion).

    Pros: Fast, uses only 4 moments, good for quick corrections.
    Cons: Can produce non-monotonic quantiles in extreme cases.
    Best for: Quick adjustment when you know the Gaussian is close but
    need to correct for skew/kurtosis in the tails.

    The expansion corrects Gaussian quantiles:
        z_cf = z + (z²-1)·S/6 + (z³-3z)·K/24 - (2z³-5z)·S²/36
    where S = skewness, K = excess kurtosis, z = Gaussian quantile.

    This is exactly the correction that half-Kelly is trying to
    approximate heuristically.
    """

    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data).ravel()
        self._moments = _compute_moments(self._data)
        self._mu = self._moments.mean
        self._sigma = self._moments.std
        self._skew = self._moments.skewness
        self._kurt = self._moments.kurtosis

    @property
    def name(self) -> str:
        return "cornish_fisher"

    @property
    def moments(self) -> DistributionMoments:
        return self._moments

    def _cf_correction(self, z: np.ndarray) -> np.ndarray:
        """Apply Cornish-Fisher correction to standard normal quantiles."""
        s = self._skew
        k = self._kurt
        z_cf = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )
        return z_cf

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(p) == 0
        p = np.atleast_1d(np.asarray(p, dtype=float))
        z = sp_stats.norm.ppf(p)
        z_cf = self._cf_correction(z)
        r = self._mu + self._sigma * z_cf
        return self._maybe_scalar(r, scalar)

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        probs = np.linspace(1e-6, 1 - 1e-6, 5000)
        quants = self.quantile(probs)
        r = np.interp(x, quants, probs, left=0.0, right=1.0)
        return self._maybe_scalar(r, scalar)

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dx = self._sigma / 500
        r = (np.atleast_1d(self.cdf(x + dx)) - np.atleast_1d(self.cdf(x - dx))) / (2 * dx)
        return self._maybe_scalar(r, scalar)


# ═══════════════════════════════════════════════════════════════════
#  Implementation 5: Gaussian Mixture
# ═══════════════════════════════════════════════════════════════════


class GaussianMixtureDistribution(DistributionApproximation):
    """
    Gaussian mixture model — sum of K Gaussian components.

    Pros: Can capture skewness, fat tails, AND multimodality.
    Cons: Requires choosing K, EM can find local optima.
    Best for: Regime-dependent returns (calm + crisis components),
    or any distribution that's clearly non-unimodal.

    A 2-component mixture naturally decomposes into:
        - "normal" regime (high weight, low vol)
        - "crisis" regime (low weight, high vol, possibly shifted left)
    This is often more interpretable than a single Student-t.
    """

    def __init__(self, data: np.ndarray, n_components: int = 2):
        from sklearn.mixture import GaussianMixture

        self._data = np.asarray(data).ravel()
        self._n_components = n_components
        self._moments = _compute_moments(self._data)

        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )
        gmm.fit(self._data.reshape(-1, 1))

        self._weights = gmm.weights_.ravel()
        self._means = gmm.means_.ravel()
        self._stds = np.sqrt(gmm.covariances_.ravel())
        self._components = [
            sp_stats.norm(loc=self._means[i], scale=self._stds[i])
            for i in range(n_components)
        ]

        # Pre-compute CDF grid for quantile inversion
        lo = self._data.min() - 4 * self._data.std()
        hi = self._data.max() + 4 * self._data.std()
        self._grid = np.linspace(lo, hi, 5000)
        self._cdf_grid = self.cdf(self._grid)

    @property
    def name(self) -> str:
        return "gaussian_mixture"

    @property
    def moments(self) -> DistributionMoments:
        return self._moments

    @property
    def component_params(self) -> list[dict[str, float]]:
        """Parameters of each mixture component."""
        return [
            {
                "weight": float(self._weights[i]),
                "mean": float(self._means[i]),
                "std": float(self._stds[i]),
            }
            for i in range(self._n_components)
        ]

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros_like(x)
        for i in range(self._n_components):
            result += self._weights[i] * self._components[i].pdf(x)
        return self._maybe_scalar(result, scalar)

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros_like(x)
        for i in range(self._n_components):
            result += self._weights[i] * self._components[i].cdf(x)
        return self._maybe_scalar(result, scalar)

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(p) == 0
        p = np.atleast_1d(np.asarray(p, dtype=float))
        return self._maybe_scalar(np.interp(p, self._cdf_grid, self._grid), scalar)


# ═══════════════════════════════════════════════════════════════════
#  Implementation 6: EVT Composite (GPD tails + parametric body)
# ═══════════════════════════════════════════════════════════════════


class EVTCompositeDistribution(DistributionApproximation):
    """
    Extreme Value Theory composite: GPD for tails, KDE for body.

    Pros: Best-in-class tail modeling, theoretically grounded.
    Cons: Requires choosing tail threshold, needs decent sample size.
    Best for: Risk management where tail accuracy matters most
    (VaR, CVaR, ruin probability, slippage worst-case).

    The Generalized Pareto Distribution (GPD) is the only distribution
    that's theoretically justified for modeling exceedances over a
    threshold (Pickands-Balkema-de Haan theorem). This composite
    uses GPD for both left and right tails, and KDE for the body.

    The tail_quantile parameter controls where the GPD takes over:
        0.05 = GPD handles bottom 5% and top 5%
        0.10 = GPD handles bottom 10% and top 10%
    """

    def __init__(
        self,
        data: np.ndarray,
        tail_quantile: float = 0.10,
    ):
        self._data = np.sort(np.asarray(data).ravel())
        self._n = len(self._data)
        self._moments = _compute_moments(self._data)
        self._tail_q = tail_quantile

        # Thresholds
        self._left_threshold = np.quantile(self._data, tail_quantile)
        self._right_threshold = np.quantile(self._data, 1 - tail_quantile)

        # Fit GPD to left tail (negate for standard GPD on positive exceedances)
        left_exceedances = self._left_threshold - self._data[
            self._data < self._left_threshold
        ]
        if len(left_exceedances) > 5:
            self._left_shape, _, self._left_scale = sp_stats.genpareto.fit(
                left_exceedances, floc=0
            )
            self._has_left_gpd = True
        else:
            self._has_left_gpd = False

        # Fit GPD to right tail
        right_exceedances = (
            self._data[self._data > self._right_threshold]
            - self._right_threshold
        )
        if len(right_exceedances) > 5:
            self._right_shape, _, self._right_scale = sp_stats.genpareto.fit(
                right_exceedances, floc=0
            )
            self._has_right_gpd = True
        else:
            self._has_right_gpd = False

        # KDE for the body
        body = self._data[
            (self._data >= self._left_threshold)
            & (self._data <= self._right_threshold)
        ]
        if len(body) > 10:
            self._body_kde = sp_stats.gaussian_kde(body)
        else:
            self._body_kde = sp_stats.gaussian_kde(self._data)

        # Pre-compute CDF grid for fast lookup
        lo = self._data.min() - 3 * self._data.std()
        hi = self._data.max() + 3 * self._data.std()
        self._grid = np.linspace(lo, hi, 5000)
        self._cdf_grid = np.array([self._cdf_scalar(x) for x in self._grid])

    @property
    def name(self) -> str:
        return "evt_composite"

    @property
    def moments(self) -> DistributionMoments:
        return self._moments

    @property
    def left_gpd_params(self) -> dict[str, float] | None:
        if self._has_left_gpd:
            return {"shape": self._left_shape, "scale": self._left_scale}
        return None

    @property
    def right_gpd_params(self) -> dict[str, float] | None:
        if self._has_right_gpd:
            return {"shape": self._right_shape, "scale": self._right_scale}
        return None

    def _cdf_scalar(self, x: float) -> float:
        """CDF at a single point, combining tails and body."""
        if x < self._left_threshold and self._has_left_gpd:
            # Left tail: P(X < x) = tail_q * P(exceedance > threshold - x)
            exc = self._left_threshold - x
            surv = sp_stats.genpareto.sf(exc, self._left_shape, scale=self._left_scale)
            return self._tail_q * surv
        elif x > self._right_threshold and self._has_right_gpd:
            # Right tail
            exc = x - self._right_threshold
            cdf_exc = sp_stats.genpareto.cdf(
                exc, self._right_shape, scale=self._right_scale
            )
            return (1 - self._tail_q) + self._tail_q * cdf_exc
        else:
            # Body: interpolate from ECDF for stability
            body_frac = float(
                np.searchsorted(self._data, x) / self._n
            )
            return np.clip(body_frac, self._tail_q, 1 - self._tail_q)

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        r = np.interp(x, self._grid, self._cdf_grid, left=0.0, right=1.0)
        return self._maybe_scalar(r, scalar)

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(p) == 0
        p = np.atleast_1d(np.asarray(p, dtype=float))
        r = np.interp(p, self._cdf_grid, self._grid)
        return self._maybe_scalar(r, scalar)

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dx = self._moments.std / 500
        r = np.clip(
            (np.atleast_1d(self.cdf(x + dx)) - np.atleast_1d(self.cdf(x - dx))) / (2 * dx), 0, None
        )
        return self._maybe_scalar(r, scalar)


# ═══════════════════════════════════════════════════════════════════
#  Fitting Interface
# ═══════════════════════════════════════════════════════════════════


class FitMethod(Enum):
    EMPIRICAL = "empirical"
    KDE = "kde"
    STUDENT_T = "student_t"
    CORNISH_FISHER = "cornish_fisher"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    EVT_COMPOSITE = "evt_composite"
    AUTO = "auto"


def fit_distribution(
    data: np.ndarray,
    method: str | FitMethod = FitMethod.AUTO,
    **kwargs: Any,
) -> DistributionApproximation:
    """
    Fit a distribution approximation to observed data.

    Args:
        data: 1D array of observations (returns, slippage, etc.)
        method: Fitting method. "auto" selects based on data characteristics.
        **kwargs: Method-specific parameters.

    Returns:
        DistributionApproximation fitted to the data.
    """
    if isinstance(method, str):
        method = FitMethod(method)

    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]

    if len(data) < 10:
        raise ValueError(f"Need at least 10 data points, got {len(data)}")

    if method == FitMethod.AUTO:
        return _auto_fit(data, **kwargs)
    elif method == FitMethod.EMPIRICAL:
        return EmpiricalDistribution(data)
    elif method == FitMethod.KDE:
        return KDEDistribution(data, bw_method=kwargs.get("bw_method"))
    elif method == FitMethod.STUDENT_T:
        return StudentTDistribution(data)
    elif method == FitMethod.CORNISH_FISHER:
        return CornishFisherDistribution(data)
    elif method == FitMethod.GAUSSIAN_MIXTURE:
        return GaussianMixtureDistribution(
            data, n_components=kwargs.get("n_components", 2)
        )
    elif method == FitMethod.EVT_COMPOSITE:
        return EVTCompositeDistribution(
            data, tail_quantile=kwargs.get("tail_quantile", 0.10)
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _auto_fit(data: np.ndarray, **kwargs: Any) -> DistributionApproximation:
    """
    Automatically select the best fitting method.

    Decision logic:
        n < 50     → Empirical (not enough data for parametric)
        n < 200    → Student-t (simple, robust)
        n < 500    → KDE or Student-t (based on skewness)
        n >= 500   → EVT composite (enough data for tail modeling)
        high skew  → Gaussian mixture (captures asymmetry)
    """
    n = len(data)
    moments = _compute_moments(data)

    if n < 50:
        return EmpiricalDistribution(data)

    if n < 200:
        return StudentTDistribution(data)

    # Enough data for richer models
    if abs(moments.skewness) > 0.5:
        # Significant skewness — mixture captures asymmetry better
        if n >= 500:
            return EVTCompositeDistribution(data, **kwargs)
        return GaussianMixtureDistribution(data, n_components=2)

    if n >= 500:
        return EVTCompositeDistribution(data, **kwargs)

    return KDEDistribution(data)


# ═══════════════════════════════════════════════════════════════════
#  Application: Distribution-Aware Kelly
# ═══════════════════════════════════════════════════════════════════


@dataclass
class KellyResult:
    """Result of optimal Kelly sizing computation."""

    f_star: float  # Optimal fraction
    f_gaussian: float  # What Gaussian Kelly would give
    correction_ratio: float  # f_star / f_gaussian (typically < 1)
    expected_log_growth: float  # E[log(1 + f*·R)] at optimum
    ruin_probability: float  # P(loss > 100%) at f_star
    distribution: str  # Which distribution was used


def optimal_kelly(
    dist: DistributionApproximation,
    max_fraction: float = 2.0,
    risk_free_rate: float = 0.0,
    n_samples: int = 10000,
) -> KellyResult:
    """
    Compute the true optimal Kelly fraction using the actual distribution.

    Instead of f* = μ/σ² (Gaussian assumption), we numerically solve:
        f* = argmax E[log(1 + f·R)]

    where R is drawn from the fitted distribution.

    The correction_ratio tells you exactly how much the Gaussian
    overstates optimal sizing. If it's 0.5, then half-Kelly was
    actually correct for this distribution. If it's 0.7, half-Kelly
    was too conservative. If it's 0.4, even half-Kelly was too aggressive.

    Args:
        dist: Fitted distribution of returns.
        max_fraction: Maximum fraction to search over.
        risk_free_rate: Subtracted from returns before optimization.
        n_samples: Monte Carlo integration points.

    Returns:
        KellyResult with optimal fraction, Gaussian comparison, and diagnostics.
    """
    m = dist.moments
    mu = m.mean - risk_free_rate
    sigma2 = m.variance

    # Gaussian Kelly
    f_gaussian = mu / sigma2 if sigma2 > 1e-12 else 0.0

    # Sample from the distribution for numerical optimization
    returns = dist.sample(n_samples, seed=42)
    returns = returns - risk_free_rate

    def neg_expected_log_growth(f: float) -> float:
        """Negative of E[log(1 + f·R)] — we minimize this."""
        growth = 1.0 + f * returns
        # Clamp to avoid log(0) or log(negative)
        growth = np.clip(growth, 1e-10, None)
        return -float(np.mean(np.log(growth)))

    # Optimize over [0, max_fraction]
    result = optimize.minimize_scalar(
        neg_expected_log_growth,
        bounds=(0.0, max_fraction),
        method="bounded",
    )
    f_star = result.x
    expected_log_growth = -result.fun

    # Ruin probability at f_star
    ruin_prob = float(np.mean((1.0 + f_star * returns) <= 0))

    # Correction ratio
    correction = f_star / f_gaussian if abs(f_gaussian) > 1e-12 else 1.0

    return KellyResult(
        f_star=f_star,
        f_gaussian=f_gaussian,
        correction_ratio=correction,
        expected_log_growth=expected_log_growth,
        ruin_probability=ruin_prob,
        distribution=dist.name,
    )


# ═══════════════════════════════════════════════════════════════════
#  Application: Risk Metrics
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics using actual distribution."""

    var_95: float  # Value at Risk, 5% level
    var_99: float  # Value at Risk, 1% level
    cvar_95: float  # Conditional VaR (Expected Shortfall), 5%
    cvar_99: float  # Conditional VaR, 1%
    var_95_gaussian: float  # What Gaussian would give
    var_99_gaussian: float
    tail_ratio_left: float  # actual_VaR / gaussian_VaR at 1%
    tail_ratio_right: float  # P(X > 3σ) / P_gaussian(X > 3σ)
    max_loss_99: float  # Worst 1% loss magnitude
    distribution: str


def compute_risk_metrics(dist: DistributionApproximation) -> RiskMetrics:
    """
    Compute comprehensive risk metrics using actual vs Gaussian distributions.

    The tail_ratio metrics quantify exactly how much risk the Gaussian misses:
        tail_ratio_left = 2.0 means left tail is 2x fatter than Gaussian
        tail_ratio_right = 1.5 means right tail is 1.5x fatter

    Args:
        dist: Fitted distribution.

    Returns:
        RiskMetrics comparing actual vs Gaussian assumptions.
    """
    m = dist.moments

    # Actual distribution
    var_95 = float(dist.quantile(0.05))
    var_99 = float(dist.quantile(0.01))
    cvar_95 = dist.expected_shortfall(0.05)
    cvar_99 = dist.expected_shortfall(0.01)

    # Gaussian comparison
    gaussian = sp_stats.norm(loc=m.mean, scale=m.std)
    var_95_g = float(gaussian.ppf(0.05))
    var_99_g = float(gaussian.ppf(0.01))

    # Tail ratios
    tail_left = var_99 / var_99_g if abs(var_99_g) > 1e-12 else 1.0
    # Right tail: compare probability of extreme positive moves
    three_sigma = m.mean + 3 * m.std
    p_actual_right = float(dist.survival(three_sigma))
    p_gauss_right = float(gaussian.sf(three_sigma))
    tail_right = (
        p_actual_right / p_gauss_right
        if p_gauss_right > 1e-12
        else 1.0
    )

    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        var_95_gaussian=var_95_g,
        var_99_gaussian=var_99_g,
        tail_ratio_left=tail_left,
        tail_ratio_right=tail_right,
        max_loss_99=abs(var_99),
        distribution=dist.name,
    )


# ═══════════════════════════════════════════════════════════════════
#  Diagnostics: Gaussian Divergence
# ═══════════════════════════════════════════════════════════════════


@dataclass
class GaussianDivergence:
    """Quantifies how much a distribution deviates from Gaussian."""

    excess_kurtosis: float
    skewness: float
    kl_divergence: float  # KL(actual || Gaussian)
    anderson_darling: float  # AD test statistic
    ad_p_value: float  # AD test p-value
    qq_residuals_std: float  # Std of QQ plot residuals
    tail_weight_left: float  # Mass below -2σ: actual vs Gaussian ratio
    tail_weight_right: float  # Mass above +2σ
    kelly_correction: float  # actual_kelly / gaussian_kelly
    risk_metrics: RiskMetrics
    recommended_method: str  # Which distribution method is best
    gaussian_adequate: bool  # Can you get away with Gaussian?


def gaussian_divergence(
    dist: DistributionApproximation,
    data: np.ndarray | None = None,
) -> GaussianDivergence:
    """
    Comprehensive comparison of a distribution against Gaussian.

    This is the "should I bother with a fancy distribution?" diagnostic.
    If gaussian_adequate is True, the extra complexity isn't worth it.

    Args:
        dist: Fitted distribution to evaluate.
        data: Original data (for Anderson-Darling test). Optional.

    Returns:
        GaussianDivergence with all comparison metrics.
    """
    m = dist.moments
    gaussian = sp_stats.norm(loc=m.mean, scale=m.std)

    # KL divergence (numerical)
    x_grid = np.linspace(m.mean - 5 * m.std, m.mean + 5 * m.std, 2000)
    p = dist.pdf(x_grid)
    q = gaussian.pdf(x_grid)
    # Avoid log(0)
    mask = (p > 1e-12) & (q > 1e-12)
    kl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])) * (x_grid[1] - x_grid[0]))

    # Anderson-Darling on original data
    if data is not None:
        ad_stat, ad_crit, ad_sig = sp_stats.anderson(data, "norm")
        ad_p = float(ad_sig[-1]) / 100 if ad_stat > ad_crit[-1] else 0.5
    else:
        ad_stat, ad_p = 0.0, 1.0

    # QQ residuals
    n_qq = 200
    theoretical_q = np.linspace(0.005, 0.995, n_qq)
    actual_vals = dist.quantile(theoretical_q)
    gaussian_vals = gaussian.ppf(theoretical_q)
    qq_resid_std = float(np.std(actual_vals - gaussian_vals))

    # Tail weights
    left_2s = m.mean - 2 * m.std
    right_2s = m.mean + 2 * m.std
    actual_left = float(dist.cdf(left_2s))
    gauss_left = float(gaussian.cdf(left_2s))
    actual_right = float(dist.survival(right_2s))
    gauss_right = float(gaussian.sf(right_2s))
    tw_left = actual_left / gauss_left if gauss_left > 1e-12 else 1.0
    tw_right = actual_right / gauss_right if gauss_right > 1e-12 else 1.0

    # Kelly correction
    kelly = optimal_kelly(dist)
    risk = compute_risk_metrics(dist)

    # Adequacy check
    adequate = (
        abs(m.kurtosis) < 1.0
        and abs(m.skewness) < 0.5
        and abs(kelly.correction_ratio - 1.0) < 0.15
        and tw_left < 1.5
    )

    # Method recommendation
    if adequate:
        rec = "gaussian"
    elif abs(m.skewness) < 0.3 and m.kurtosis > 1:
        rec = "student_t"
    elif abs(m.skewness) > 0.5:
        rec = "gaussian_mixture"
    elif m.n_samples >= 500:
        rec = "evt_composite"
    else:
        rec = "kde"

    return GaussianDivergence(
        excess_kurtosis=m.kurtosis,
        skewness=m.skewness,
        kl_divergence=kl,
        anderson_darling=ad_stat,
        ad_p_value=ad_p,
        qq_residuals_std=qq_resid_std,
        tail_weight_left=tw_left,
        tail_weight_right=tw_right,
        kelly_correction=kelly.correction_ratio,
        risk_metrics=risk,
        recommended_method=rec,
        gaussian_adequate=adequate,
    )


# ═══════════════════════════════════════════════════════════════════
#  QQ Plot Data (for visualization)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class QQPlotData:
    """Data for generating a QQ plot against a reference distribution."""

    theoretical_quantiles: np.ndarray
    actual_quantiles: np.ndarray
    reference_name: str
    diagonal: tuple[float, float]  # (min, max) for the 45° line


def qq_data(
    dist: DistributionApproximation,
    reference: str = "gaussian",
    n_points: int = 200,
) -> QQPlotData:
    """
    Generate QQ plot data comparing actual vs reference distribution.

    Points below the diagonal = thinner tail than reference.
    Points above the diagonal = fatter tail than reference.

    Args:
        dist: Fitted distribution.
        reference: "gaussian" or another distribution name.
        n_points: Number of quantile points.

    Returns:
        QQPlotData for plotting.
    """
    probs = np.linspace(0.005, 0.995, n_points)
    actual = dist.quantile(probs)

    if reference == "gaussian":
        m = dist.moments
        ref = sp_stats.norm(loc=m.mean, scale=m.std)
        theoretical = ref.ppf(probs)
    else:
        raise ValueError(f"Unsupported reference: {reference}")

    lo = min(float(actual.min()), float(theoretical.min()))
    hi = max(float(actual.max()), float(theoretical.max()))

    return QQPlotData(
        theoretical_quantiles=theoretical,
        actual_quantiles=actual,
        reference_name=reference,
        diagonal=(lo, hi),
    )


# ═══════════════════════════════════════════════════════════════════
#  Multi-Fit Comparison
# ═══════════════════════════════════════════════════════════════════


@dataclass
class FitComparison:
    """Side-by-side comparison of multiple distribution fits."""

    method: str
    log_likelihood: float
    aic: float  # Akaike Information Criterion
    n_params: int
    var_99: float
    cvar_99: float
    kelly_f_star: float


def compare_fits(
    data: np.ndarray,
    methods: list[str] | None = None,
) -> list[FitComparison]:
    """
    Fit multiple methods and compare on standard criteria.

    Returns results sorted by AIC (lower = better fit with parsimony).

    Args:
        data: Observed data.
        methods: List of method names. None = try all.

    Returns:
        Sorted list of FitComparison results.
    """
    if methods is None:
        methods = ["empirical", "kde", "student_t", "cornish_fisher",
                    "gaussian_mixture", "evt_composite"]

    n_params_map = {
        "empirical": 0,
        "kde": 1,  # bandwidth
        "student_t": 3,  # df, loc, scale
        "cornish_fisher": 4,  # mean, var, skew, kurt
        "gaussian_mixture": 5,  # 2 × (weight, mean, std) - 1
        "evt_composite": 6,  # 2 × GPD(shape, scale) + thresholds
    }

    results = []
    for method in methods:
        try:
            dist = fit_distribution(data, method=method)
            ll = dist.log_likelihood(data)
            k = n_params_map.get(method, 2)
            aic = 2 * k - 2 * ll

            kelly = optimal_kelly(dist)
            var_99 = float(dist.quantile(0.01))
            cvar_99 = dist.expected_shortfall(0.01)

            results.append(FitComparison(
                method=method,
                log_likelihood=ll,
                aic=aic,
                n_params=k,
                var_99=var_99,
                cvar_99=cvar_99,
                kelly_f_star=kelly.f_star,
            ))
        except Exception:
            pass

    results.sort(key=lambda r: r.aic)
    return results


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════


def _compute_moments(data: np.ndarray) -> DistributionMoments:
    """Compute standard and higher-order moments from data."""
    data = np.asarray(data).ravel()
    return DistributionMoments(
        mean=float(np.mean(data)),
        variance=float(np.var(data, ddof=1)),
        std=float(np.std(data, ddof=1)),
        skewness=float(sp_stats.skew(data)),
        kurtosis=float(sp_stats.kurtosis(data)),  # excess kurtosis
        n_samples=len(data),
    )
