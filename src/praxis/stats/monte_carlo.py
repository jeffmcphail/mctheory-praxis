"""
Monte Carlo Surface Generation (Phase 4.4).

Port of generateAdfValuesForStepwiseRegression(), getCriticalValues(),
and vrProfileCovarianceStepReg() from statsUtilities.py.

Generates custom critical values for the ADF test when applied to
stepwise regression residuals (which have different distributions
than standard ADF tables because of data mining bias).

The Monte Carlo approach:
1. Generate N random walk universes (null: no cointegration)
2. For each, run stepwise regression + ADF
3. Collect ADF t-values → sort → extract percentiles
4. These percentiles are the correct critical values

Usage:
    # Generate critical values for 500 assets, 400 obs, 3 vars
    result = generate_adf_critical_values(
        n_assets=500, n_obs=400, n_vars=3, n_samples=1000
    )
    print(result.critical_values)  # {10: -4.21, 5: -4.55, 1: -5.10}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import linalg as LA

from praxis.logger.core import PraxisLogger
from praxis.stats.regression import (
    successive_regression,
    generate_random_walk_universe,
    corr2_coeff,
)
from praxis.stats import variance_profile


# ═══════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CriticalValues:
    """Critical values at specified confidence levels."""
    values: dict[int, float] = field(default_factory=dict)
    n_samples: int = 0
    n_assets: int = 0
    n_obs: int = 0
    n_vars: int = 0

    def at(self, pct: int) -> float:
        """Get critical value at percentage confidence (e.g., 5 for 5%)."""
        return self.values.get(pct, float("nan"))

    def is_significant(self, t_statistic: float, pct: int = 5) -> bool:
        """Check if a t-statistic exceeds the critical value."""
        cv = self.at(pct)
        return t_statistic < cv  # ADF: more negative = more significant


@dataclass
class MonteCarloResult:
    """Full result of Monte Carlo critical value generation."""
    critical_values: CriticalValues
    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    elapsed_seconds: float = 0.0

    @property
    def mean_t(self) -> float:
        return float(np.mean(self.t_values)) if len(self.t_values) > 0 else 0.0

    @property
    def std_t(self) -> float:
        return float(np.std(self.t_values)) if len(self.t_values) > 0 else 0.0


@dataclass
class VRProfileResult:
    """Variance ratio profile covariance analysis result."""
    paths_matrix: np.ndarray
    residuals_matrix: np.ndarray
    vr_profiles: list[np.ndarray] = field(default_factory=list)
    covariances: list[np.ndarray] = field(default_factory=list)
    covariance_inverses: list[np.ndarray] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  Critical Value Computation
# ═══════════════════════════════════════════════════════════════════

def get_critical_values(
    statistic_array: np.ndarray,
    pct_conf: list[int] | None = None,
) -> dict[int, float]:
    """
    Extract critical values from sorted statistics via interpolation.

    Port of getCriticalValues() from statsUtilities.py.

    Args:
        statistic_array: Sorted array of test statistics.
        pct_conf: Confidence levels (e.g., [10, 5, 1]).

    Returns:
        Dict mapping confidence level → critical value.
    """
    if pct_conf is None:
        pct_conf = [10, 5, 1]

    n = len(statistic_array)
    indices = np.arange(n)
    result = {}

    for cv in pct_conf:
        idx = n * cv / 100.0 - 1
        result[cv] = float(np.interp(idx, indices, statistic_array))

    return result


def generate_adf_critical_values(
    n_assets: int,
    n_obs: int,
    n_vars: int,
    n_samples: int = 1000,
    seed: int | None = None,
    pct_conf: list[int] | None = None,
) -> MonteCarloResult:
    """
    Monte Carlo generation of ADF critical values for stepwise regression.

    Port of generateAdfValuesForStepwiseRegression() from statsUtilities.py.

    Under the null hypothesis (random walks, no cointegration), this generates
    the distribution of ADF t-statistics on stepwise regression residuals.
    This corrects for the data-mining bias introduced by selecting the most
    correlated assets.

    Args:
        n_assets: Number of assets in the universe.
        n_obs: Number of observations per asset.
        n_vars: Number of independent variables in stepwise regression.
        n_samples: Number of Monte Carlo samples.
        seed: Random seed for reproducibility.
        pct_conf: Confidence levels (default: [10, 5, 1]).

    Returns:
        MonteCarloResult with critical values and t-value distribution.
    """
    import time

    log = PraxisLogger.instance()
    if pct_conf is None:
        pct_conf = [10, 5, 1]

    t0 = time.monotonic()
    adf_t_values = np.zeros(n_samples)
    current_seed = seed
    j = 0

    while j < n_samples:
        paths = generate_random_walk_universe(
            n_steps=n_obs,
            n_paths=n_assets,
            seed=current_seed,
        )
        current_seed = None  # Only seed the first batch

        for i in range(paths.shape[1]):
            if j >= n_samples:
                break

            result = successive_regression(
                target_index=i,
                asset_matrix=paths,
                n_vars=n_vars,
                compute_stats=True,
            )

            if result.adf is not None:
                adf_t_values[j] = result.adf.t_statistic
            else:
                adf_t_values[j] = 0.0

            j += 1

    elapsed = time.monotonic() - t0

    adf_t_values.sort()
    critical_vals = get_critical_values(adf_t_values, pct_conf)

    log.info(
        f"Monte Carlo ADF: n_assets={n_assets}, n_obs={n_obs}, n_vars={n_vars}, "
        f"n_samples={n_samples}, elapsed={elapsed:.1f}s",
        tags={"monte_carlo", "stats"},
    )

    cv = CriticalValues(
        values=critical_vals,
        n_samples=n_samples,
        n_assets=n_assets,
        n_obs=n_obs,
        n_vars=n_vars,
    )

    return MonteCarloResult(
        critical_values=cv,
        t_values=adf_t_values,
        elapsed_seconds=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════
#  Batch Critical Values (multi-parameter sweep)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CriticalValueRow:
    """One row of a critical value table."""
    n_assets: int
    n_obs: int
    n_vars: int
    pct_conf: int
    critical_value: float


def generate_critical_values_batch(
    n_assets_range: list[int],
    n_obs_range: list[int],
    n_vars_range: list[int],
    n_samples: int = 1000,
    seed: int | None = None,
    pct_conf: list[int] | None = None,
) -> list[CriticalValueRow]:
    """
    Sweep across parameter combinations to build a critical value surface.

    Port of generateCriticalValuesBatch() from statsUtilities.py.

    This is the "hypersurface" that maps (N_assets, N_obs, N_vars) →
    critical values, correcting for data mining bias at each point.

    Args:
        n_assets_range: List of asset universe sizes.
        n_obs_range: List of observation counts.
        n_vars_range: List of variable counts.
        n_samples: Monte Carlo samples per combination.
        seed: Random seed.
        pct_conf: Confidence levels.

    Returns:
        List of CriticalValueRow for building a lookup table.
    """
    if pct_conf is None:
        pct_conf = [10, 5, 1]

    rows: list[CriticalValueRow] = []

    for na in n_assets_range:
        for no in n_obs_range:
            for nv in n_vars_range:
                result = generate_adf_critical_values(
                    n_assets=na,
                    n_obs=no,
                    n_vars=nv,
                    n_samples=n_samples,
                    seed=seed,
                    pct_conf=pct_conf,
                )

                for pct, cv in result.critical_values.values.items():
                    rows.append(CriticalValueRow(
                        n_assets=na,
                        n_obs=no,
                        n_vars=nv,
                        pct_conf=pct,
                        critical_value=cv,
                    ))

    return rows


# ═══════════════════════════════════════════════════════════════════
#  Variance Ratio Profile Covariance (for Mahalanobis scoring)
# ═══════════════════════════════════════════════════════════════════

def vr_profile_covariance_stepreg(
    lags: list[int],
    n_steps: int,
    n_paths: int,
    n_vars: int,
    step_set: list[int] | None = None,
    origin_range: tuple[int, int] | None = None,
    seed: int | None = None,
) -> VRProfileResult:
    """
    Generate variance ratio profile covariance for stepwise regression residuals.

    Port of vrProfileCovarianceStepReg() from statsUtilities.py.

    Used for Mahalanobis distance scoring of mean-reversion quality:
    compare a candidate basket's VR profile to the null distribution's
    covariance structure.

    Args:
        lags: List of lag values to compute covariance matrices for.
        n_steps: Number of time steps per path.
        n_paths: Number of random walk paths.
        n_vars: Number of stepwise regression variables.
        step_set: Random walk step values.
        origin_range: Optional starting value range.
        seed: Random seed.

    Returns:
        VRProfileResult with paths, residuals, VR profiles, and covariances.
    """
    max_lag = max(lags)

    paths = generate_random_walk_universe(
        n_steps=n_steps,
        n_paths=n_paths,
        step_set=step_set,
        origin_range=origin_range,
        seed=seed,
    )

    # Compute residuals for each path via stepwise regression
    residuals_list = []
    for i in range(paths.shape[1]):
        result = successive_regression(
            target_index=i,
            asset_matrix=paths,
            n_vars=n_vars,
            compute_stats=False,
        )
        if result.regression is not None:
            residuals_list.append(result.regression.residuals)

    if not residuals_list:
        return VRProfileResult(
            paths_matrix=paths,
            residuals_matrix=np.empty((n_steps, 0)),
        )

    residuals_matrix = np.column_stack(residuals_list)

    # Compute VR profiles for each residual series
    vr_profile_matrix = np.column_stack([
        variance_profile(residuals_matrix[:, i], max_lag)
        for i in range(residuals_matrix.shape[1])
    ])

    # Compute covariance at each lag level
    vr_profiles = []
    covariances = []
    cov_inverses = []

    for lag in sorted(lags):
        vr_sub = vr_profile_matrix[:lag]
        vr_profiles.append(vr_sub)

        cov = np.cov(vr_sub)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        covariances.append(cov)

        try:
            cov_inv = LA.inv(cov)
        except LA.LinAlgError:
            cov_inv = np.eye(cov.shape[0])
        cov_inverses.append(cov_inv)

    return VRProfileResult(
        paths_matrix=paths,
        residuals_matrix=residuals_matrix,
        vr_profiles=vr_profiles,
        covariances=covariances,
        covariance_inverses=cov_inverses,
    )


def mahalanobis_distance(
    u: np.ndarray,
    v: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """
    Mahalanobis distance between vectors u and v.

    Port of mahalanobis() from statsUtilities.py.

    Args:
        u: First vector.
        v: Second vector (often the mean/centroid).
        cov_inv: Inverse covariance matrix.

    Returns:
        Mahalanobis distance scalar.
    """
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    diff = u - v
    return float(np.sqrt(diff @ cov_inv @ diff))
