"""
Markowitz Portfolio Optimization (Phase 4.5 dependency).

Weight optimization for stat arb baskets. Given a set of candidate
assets with covariance structure, find optimal weights for:
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Risk parity

Uses analytical solutions where possible, scipy.optimize for
constrained cases.

Usage:
    result = markowitz_optimize(returns_matrix, method="min_variance")
    print(result.weights, result.expected_return, result.volatility)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import linalg as LA


@dataclass
class PortfolioResult:
    """Result of portfolio optimization."""
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    method: str = ""
    n_assets: int = 0
    converged: bool = True

    @property
    def is_valid(self) -> bool:
        return len(self.weights) > 0 and self.converged


def covariance_matrix(
    returns: np.ndarray,
    shrinkage: str = "none",
    shrinkage_target: float = 0.5,
) -> np.ndarray:
    """
    Compute covariance matrix with optional shrinkage.

    Args:
        returns: (n_obs, n_assets) matrix of returns.
        shrinkage: 'none', 'ledoit_wolf', or 'constant'.
        shrinkage_target: Shrinkage intensity for 'constant' method.

    Returns:
        (n_assets, n_assets) covariance matrix.
    """
    cov = np.cov(returns, rowvar=False)

    if shrinkage == "ledoit_wolf":
        # Simplified Ledoit-Wolf: shrink toward diagonal
        n = cov.shape[0]
        target = np.diag(np.diag(cov))
        # Simple shrinkage intensity
        delta = shrinkage_target
        cov = (1 - delta) * cov + delta * target

    elif shrinkage == "constant":
        n = cov.shape[0]
        target = np.diag(np.diag(cov))
        cov = (1 - shrinkage_target) * cov + shrinkage_target * target

    return cov


def min_variance_portfolio(
    cov: np.ndarray,
    long_only: bool = False,
    max_weight: float = 1.0,
) -> PortfolioResult:
    """
    Minimum variance portfolio.

    Analytical solution: w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)

    Args:
        cov: (n, n) covariance matrix.
        long_only: Constrain weights ≥ 0.
        max_weight: Maximum weight per asset.

    Returns:
        PortfolioResult with optimal weights.
    """
    n = cov.shape[0]

    if long_only:
        return _constrained_min_variance(cov, max_weight)

    try:
        cov_inv = LA.inv(cov)
    except LA.LinAlgError:
        cov_inv = LA.pinv(cov)

    ones = np.ones(n)
    w = cov_inv @ ones
    w = w / w.sum()

    vol = float(np.sqrt(w @ cov @ w))

    return PortfolioResult(
        weights=w,
        volatility=vol,
        method="min_variance",
        n_assets=n,
    )


def _constrained_min_variance(
    cov: np.ndarray,
    max_weight: float = 1.0,
) -> PortfolioResult:
    """Constrained minimum variance using iterative projection."""
    from scipy.optimize import minimize

    n = cov.shape[0]
    x0 = np.ones(n) / n

    def objective(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, max_weight)] * n

    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    w = result.x
    vol = float(np.sqrt(w @ cov @ w))

    return PortfolioResult(
        weights=w,
        volatility=vol,
        method="min_variance_constrained",
        n_assets=n,
        converged=result.success,
    )


def max_sharpe_portfolio(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0,
    long_only: bool = False,
    max_weight: float = 1.0,
) -> PortfolioResult:
    """
    Maximum Sharpe ratio portfolio.

    Analytical: w ∝ Σ⁻¹ (μ - rf)
    Constrained: scipy optimize.

    Args:
        expected_returns: (n,) vector of expected returns.
        cov: (n, n) covariance matrix.
        risk_free_rate: Risk-free rate.
        long_only: Constrain weights ≥ 0.
        max_weight: Maximum weight per asset.

    Returns:
        PortfolioResult with optimal weights.
    """
    n = len(expected_returns)
    excess = expected_returns - risk_free_rate

    if long_only:
        return _constrained_max_sharpe(excess, cov, risk_free_rate, max_weight)

    try:
        cov_inv = LA.inv(cov)
    except LA.LinAlgError:
        cov_inv = LA.pinv(cov)

    w = cov_inv @ excess
    w_sum = w.sum()
    if abs(w_sum) > 1e-12:
        w = w / w_sum
    else:
        w = np.ones(n) / n

    ret = float(w @ expected_returns)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0

    return PortfolioResult(
        weights=w,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        method="max_sharpe",
        n_assets=n,
    )


def _constrained_max_sharpe(
    excess_returns: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    max_weight: float,
) -> PortfolioResult:
    """Constrained max Sharpe using scipy."""
    from scipy.optimize import minimize

    n = len(excess_returns)
    x0 = np.ones(n) / n

    def neg_sharpe(w):
        ret = w @ excess_returns
        vol = np.sqrt(w @ cov @ w)
        return -(ret / vol) if vol > 1e-12 else 0.0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, max_weight)] * n

    result = minimize(
        neg_sharpe, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    w = result.x
    ret = float(w @ (excess_returns + risk_free_rate))
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0

    return PortfolioResult(
        weights=w,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        method="max_sharpe_constrained",
        n_assets=n,
        converged=result.success,
    )


def equal_weight_portfolio(
    n_assets: int,
    cov: np.ndarray | None = None,
    expected_returns: np.ndarray | None = None,
) -> PortfolioResult:
    """Simple 1/N portfolio."""
    w = np.ones(n_assets) / n_assets

    vol = 0.0
    ret = 0.0
    if cov is not None:
        vol = float(np.sqrt(w @ cov @ w))
    if expected_returns is not None:
        ret = float(w @ expected_returns)

    return PortfolioResult(
        weights=w,
        expected_return=ret,
        volatility=vol,
        method="equal_weight",
        n_assets=n_assets,
    )


def markowitz_optimize(
    returns: np.ndarray,
    method: str = "min_variance",
    risk_free_rate: float = 0.0,
    long_only: bool = False,
    max_weight: float = 1.0,
    shrinkage: str = "none",
) -> PortfolioResult:
    """
    High-level Markowitz optimization.

    Args:
        returns: (n_obs, n_assets) return matrix.
        method: 'min_variance', 'max_sharpe', or 'equal_weight'.
        risk_free_rate: For Sharpe computation.
        long_only: Constrain weights ≥ 0.
        max_weight: Maximum weight per asset.
        shrinkage: Covariance shrinkage method.

    Returns:
        PortfolioResult.
    """
    returns = np.asarray(returns)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    n_assets = returns.shape[1]
    cov = covariance_matrix(returns, shrinkage=shrinkage)
    mu = returns.mean(axis=0)

    if method == "min_variance":
        result = min_variance_portfolio(cov, long_only=long_only, max_weight=max_weight)
        result.expected_return = float(result.weights @ mu)
    elif method == "max_sharpe":
        result = max_sharpe_portfolio(
            mu, cov, risk_free_rate=risk_free_rate,
            long_only=long_only, max_weight=max_weight,
        )
    elif method == "equal_weight":
        result = equal_weight_portfolio(n_assets, cov=cov, expected_returns=mu)
    else:
        raise ValueError(f"Unknown method: {method}")

    if result.volatility > 0:
        result.sharpe_ratio = (result.expected_return - risk_free_rate) / result.volatility

    return result
