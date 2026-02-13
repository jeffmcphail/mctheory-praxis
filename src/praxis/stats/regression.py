"""
Successive Regression (Phase 4.3).

Port of generateStepwiseRegressionEff() and getRidgeRegression() from
statsUtilities.py. Core of the Burgess Stat Arb candidate generation:

Given a universe of N assets, for each target asset:
1. Find the most correlated asset → regress → get residuals
2. Find the next most correlated with residuals → add to regression
3. Repeat until numVars assets selected
4. Run ADF on final residuals → stationary = mean-reverting basket

Usage:
    from praxis.stats.regression import successive_regression
    indices, result = successive_regression(target_idx=0, asset_matrix=M, n_vars=3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

from praxis.stats import adf_test, ADFResult


# ═══════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RegressionResult:
    """Result of a Ridge regression. Port of TestStatistics."""
    beta: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    X: np.ndarray = field(default_factory=lambda: np.array([]))
    X_no_intercept: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class StepwiseResult:
    """Result of successive/stepwise regression."""
    target_index: int
    selected_indices: list[int] = field(default_factory=list)
    regression: RegressionResult | None = None
    adf: ADFResult | None = None

    @property
    def is_stationary(self) -> bool:
        if self.adf is None:
            return False
        return self.adf.is_stationary


# ═══════════════════════════════════════════════════════════════════
#  Core Functions
# ═══════════════════════════════════════════════════════════════════

def corr2_coeff(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Fast rowwise correlation coefficient.

    Port of corr2_coeff() from statsUtilities.py.
    Computes correlations between rows of A and rows of B.

    Args:
        A: (m, n) matrix — each row is a variable.
        B: (p, n) matrix — each row is a variable.

    Returns:
        (m, p) correlation matrix.
    """
    A_m = A - A.mean(axis=1, keepdims=True)
    B_m = B - B.mean(axis=1, keepdims=True)

    ssA = (A_m ** 2).sum(axis=1)
    ssB = (B_m ** 2).sum(axis=1)

    denom = np.sqrt(np.outer(ssA, ssB))
    # Avoid division by zero
    denom = np.where(denom > 0, denom, 1e-15)

    return A_m @ B_m.T / denom


def ridge_regression(
    y: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.01,
    fit_intercept: bool = False,
    add_intercept_feature: bool = True,
    use_ridge_cv: bool = False,
) -> RegressionResult:
    """
    Ridge regression with optional intercept and cross-validation.

    Port of getRidgeRegression() from statsUtilities.py.

    Args:
        y: Target array (n_obs,).
        X: Feature matrix (n_obs, n_features).
        alpha: Regularization strength.
        fit_intercept: Whether sklearn fits intercept internally.
        add_intercept_feature: Add column of 1s to X.
        use_ridge_cv: Use RidgeCV for alpha selection.

    Returns:
        RegressionResult with beta, residuals, R².
    """
    y = np.asarray(y).ravel()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(y)
    p = X.shape[1]
    X_no_intercept = X.copy()

    if add_intercept_feature:
        X = np.insert(X, 0, np.ones(n), axis=1)

    if use_ridge_cv:
        clf = RidgeCV(alphas=np.logspace(-6, 6, 13))
    else:
        clf = Ridge(alpha=alpha, fit_intercept=fit_intercept)

    clf.fit(X, y)
    residuals = y - clf.predict(X)

    y_centered = y - y.mean()
    ss_res = residuals.dot(residuals)
    ss_tot = y_centered.dot(y_centered)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

    return RegressionResult(
        beta=clf.coef_.copy(),
        residuals=residuals.copy(),
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        y=y.copy(),
        X=X.copy(),
        X_no_intercept=X_no_intercept.copy(),
    )


def successive_regression(
    target_index: int,
    asset_matrix: np.ndarray,
    n_vars: int = 3,
    compute_stats: bool = True,
    significance: float = 0.05,
) -> StepwiseResult:
    """
    Successive (stepwise) regression to find mean-reverting baskets.

    Port of generateStepwiseRegressionEff() from statsUtilities.py.

    Algorithm:
    1. Start with target asset's price series
    2. Find most correlated asset → Ridge regress → get residuals
    3. Repeat on residuals, masking already-selected assets
    4. After n_vars steps, run ADF on final residuals

    Args:
        target_index: Column index of target asset in asset_matrix.
        asset_matrix: (n_obs, n_assets) matrix of price series.
        n_vars: Number of independent variables to select.
        compute_stats: Run ADF on final residuals.
        significance: ADF significance level.

    Returns:
        StepwiseResult with selected indices and test results.
    """
    asset_matrix = np.asarray(asset_matrix)
    if asset_matrix.ndim != 2:
        raise ValueError("asset_matrix must be 2D (n_obs, n_assets)")

    n_obs, n_assets = asset_matrix.shape
    if target_index < 0 or target_index >= n_assets:
        raise ValueError(f"target_index {target_index} out of range [0, {n_assets})")

    target = asset_matrix[:, target_index].copy()
    current_target = target.copy()

    # Mask: 1 = already selected/target, 0 = available
    mask = np.zeros(n_assets, dtype=int)
    mask[target_index] = 1

    selected_indices: list[int] = []
    independent_matrix = np.empty((n_obs, 0))
    regression = None

    for _ in range(min(n_vars, n_assets - 1)):
        # Correlation of each asset with current residuals
        correlations = np.ma.array(
            np.abs(corr2_coeff(asset_matrix.T, current_target[None, :])),
            mask=mask,
        )

        if correlations.count() == 0:
            break

        best_idx = int(np.argmax(correlations))
        mask[best_idx] = 1
        selected_indices.append(best_idx)

        independent_matrix = np.column_stack([
            independent_matrix,
            asset_matrix[:, best_idx],
        ])

        regression = ridge_regression(target, independent_matrix)
        current_target = regression.residuals.copy()

    result = StepwiseResult(
        target_index=target_index,
        selected_indices=selected_indices,
        regression=regression,
    )

    # ADF test on final residuals
    if compute_stats:
        if regression is not None and len(regression.residuals) > 10:
            result.adf = adf_test(
                regression.residuals,
                significance=significance,
            )
        elif n_vars == 0:
            result.adf = adf_test(target, significance=significance)

    return result


def generate_random_walk_universe(
    n_steps: int,
    n_paths: int,
    step_set: list[int] | None = None,
    origin_range: tuple[int, int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a matrix of random walk paths.

    Port of generateRandomWalkUniverse() from statsUtilities.py.

    Args:
        n_steps: Number of time steps.
        n_paths: Number of independent paths.
        step_set: Possible step values (default: [-1, 0, 1]).
        origin_range: (min, max) for starting values. None = start at 0.
        seed: Random seed.

    Returns:
        (n_steps, n_paths) matrix of random walk paths.
    """
    if step_set is None:
        step_set = [-1, 0, 1]

    rng = np.random.RandomState(seed)
    paths = np.zeros((n_steps, n_paths))

    for j in range(n_paths):
        if origin_range is not None:
            start = rng.randint(origin_range[0], origin_range[1])
            steps = rng.choice(step_set, size=n_steps - 1)
            paths[:, j] = np.concatenate([[start], steps]).cumsum()
        else:
            steps = rng.choice(step_set, size=n_steps)
            paths[:, j] = steps.cumsum()

    return paths
