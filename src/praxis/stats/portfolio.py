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


# ═══════════════════════════════════════════════════════════════════
#  Kelly-Markowitz Unified Portfolio Optimization
# ═══════════════════════════════════════════════════════════════════
#
#  The beautiful result (Chan 2014, Thorp 2006):
#
#    Maximize E[log(1 + w'R)]
#    ≈ w'μ - ½ w'Σw        (second-order Taylor)
#    ∂/∂w = μ - Σw = 0
#    ∴ w* = Σ⁻¹μ
#
#  This is the Markowitz tangency portfolio (max Sharpe direction)
#  SCALED by the Kelly optimal leverage. One formula, both answers:
#    - Direction: relative weights (same as tangency)
#    - Scale: weights DON'T sum to 1 — their sum IS optimal leverage
#
#  Portfolio Sharpe: S = √(μ'Σ⁻¹μ)
#  Optimal growth:  g = r + S²/2
#  Optimal leverage: K = sum(w*) = 1'Σ⁻¹μ
#
#  For a "model of models" portfolio where each "asset" is a
#  trading model's backtest returns, this gives us the exact
#  capital allocation across models that maximizes long-term
#  compound growth — not just the best Sharpe.
# ═══════════════════════════════════════════════════════════════════


@dataclass
class KellyResult:
    """Result of Kelly-Markowitz unified optimization."""

    # Core weights
    kelly_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    markowitz_weights: np.ndarray = field(default_factory=lambda: np.array([]))

    # Kelly specifics
    optimal_leverage: float = 0.0  # sum of kelly weights
    optimal_growth_rate: float = 0.0  # g = r + S²/2
    portfolio_sharpe: float = 0.0  # S = √(μ'Σ⁻¹μ)
    risk_free_rate: float = 0.0

    # Fat-tail correction (from distribution module)
    gaussian_kelly_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    fat_tail_correction: float = 1.0  # actual/gaussian ratio
    distribution_method: str = "gaussian"

    # Diagnostics
    n_models: int = 0
    model_names: list[str] = field(default_factory=list)
    condition_number: float = 0.0  # of covariance matrix
    method: str = ""

    @property
    def is_well_conditioned(self) -> bool:
        """Condition number < 100 is generally safe for inversion."""
        return self.condition_number < 100

    @property
    def half_kelly_weights(self) -> np.ndarray:
        """The conventional conservative approximation."""
        return self.kelly_weights / 2

    @property
    def corrected_kelly_weights(self) -> np.ndarray:
        """Kelly weights adjusted for fat tails (the right answer)."""
        return self.kelly_weights * self.fat_tail_correction

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Kelly-Markowitz Portfolio ({self.method})",
            f"  Models: {self.n_models}",
            f"  Portfolio Sharpe: {self.portfolio_sharpe:.4f}",
            f"  Optimal Leverage: {self.optimal_leverage:.2f}x",
            f"  Growth Rate: {self.optimal_growth_rate:.4%}",
            f"  Condition Number: {self.condition_number:.1f}",
        ]
        if self.fat_tail_correction < 0.99:
            lines.append(f"  Fat-tail correction: {self.fat_tail_correction:.3f}")
            lines.append(f"  (Half-Kelly would be: 0.500)")

        for i in range(self.n_models):
            name = self.model_names[i] if i < len(self.model_names) else f"Model_{i}"
            kw = self.kelly_weights[i] if i < len(self.kelly_weights) else 0
            mw = self.markowitz_weights[i] if i < len(self.markowitz_weights) else 0
            cw = self.corrected_kelly_weights[i] if i < len(self.corrected_kelly_weights) else 0
            lines.append(f"  {name:20s}: kelly={kw:+.4f}  markowitz={mw:.4f}  corrected={cw:+.4f}")

        return "\n".join(lines)


def kelly_portfolio(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0,
    model_names: list[str] | None = None,
    shrinkage: str = "none",
    shrinkage_target: float = 0.5,
) -> KellyResult:
    """
    Kelly-Markowitz unified portfolio optimization.

    The unconstrained solution: w* = Σ⁻¹(μ - r)

    This gives BOTH optimal direction (which models to allocate to)
    AND optimal scale (how much total leverage to use).

    Weights do NOT sum to 1. Their sum is the optimal leverage.
    To get Markowitz-normalized weights, divide by the sum.

    Args:
        expected_returns: (n,) mean returns per model/asset.
        cov: (n,n) covariance matrix of returns.
        risk_free_rate: Risk-free rate (same frequency as returns).
        model_names: Optional names for each model/asset.
        shrinkage: Covariance shrinkage method.
        shrinkage_target: Shrinkage intensity.

    Returns:
        KellyResult with both Kelly and Markowitz weights.
    """
    n = len(expected_returns)
    excess = expected_returns - risk_free_rate

    # Apply shrinkage if requested
    if shrinkage != "none":
        cov = covariance_matrix(
            np.eye(n),  # dummy, we already have cov
            shrinkage=shrinkage,
            shrinkage_target=shrinkage_target,
        )

    # Condition number check
    cond = float(np.linalg.cond(cov))

    # Invert covariance
    try:
        cov_inv = LA.inv(cov)
    except LA.LinAlgError:
        cov_inv = LA.pinv(cov)

    # === THE FORMULA ===
    kelly_w = cov_inv @ excess  # w* = Σ⁻¹μ

    # Markowitz normalized (sum to 1)
    w_sum = kelly_w.sum()
    if abs(w_sum) > 1e-12:
        markowitz_w = kelly_w / w_sum
    else:
        markowitz_w = np.ones(n) / n

    # Portfolio Sharpe: S = √(μ'Σ⁻¹μ)
    S_squared = float(excess @ cov_inv @ excess)
    S = float(np.sqrt(max(S_squared, 0)))

    # Optimal growth rate: g = r + S²/2
    g = risk_free_rate + S_squared / 2

    # Optimal leverage
    K = float(w_sum)

    names = model_names or [f"Model_{i}" for i in range(n)]

    return KellyResult(
        kelly_weights=kelly_w,
        markowitz_weights=markowitz_w,
        gaussian_kelly_weights=kelly_w.copy(),
        optimal_leverage=K,
        optimal_growth_rate=g,
        portfolio_sharpe=S,
        risk_free_rate=risk_free_rate,
        n_models=n,
        model_names=names,
        condition_number=cond,
        method="kelly_markowitz_gaussian",
    )


def kelly_portfolio_fat_tailed(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    model_names: list[str] | None = None,
    distribution_method: str = "auto",
    shrinkage: str = "none",
    shrinkage_target: float = 0.5,
) -> KellyResult:
    """
    Kelly-Markowitz with fat-tail correction via distribution module.

    Instead of the Gaussian approximation E[log(1+w'R)] ≈ w'μ - ½w'Σw,
    we numerically optimize E[log(1+w'R)] using the actual fitted
    distribution of portfolio returns.

    The correction ratio tells you exactly how wrong half-Kelly is:
        correction = 0.50 → half-Kelly was actually right
        correction = 0.35 → half-Kelly is still too aggressive
        correction = 0.70 → half-Kelly was too conservative

    Args:
        returns: (n_obs, n_models) matrix of historical returns.
        risk_free_rate: Risk-free rate.
        model_names: Optional names.
        distribution_method: Method for distribution fitting ('auto', 'student_t', etc.)
        shrinkage: Covariance shrinkage.
        shrinkage_target: Shrinkage intensity.

    Returns:
        KellyResult with both Gaussian and fat-tail-corrected weights.
    """
    from praxis.stats.distribution import fit_distribution, optimal_kelly

    n_obs, n_models = returns.shape
    mu = returns.mean(axis=0) - risk_free_rate
    cov_mat = np.cov(returns, rowvar=False)

    if shrinkage != "none":
        cov_mat = covariance_matrix(returns, shrinkage=shrinkage, shrinkage_target=shrinkage_target)

    # Step 1: Gaussian Kelly-Markowitz (analytical)
    gaussian_result = kelly_portfolio(
        returns.mean(axis=0), cov_mat, risk_free_rate, model_names,
    )

    # Step 2: Compute portfolio returns at Gaussian Kelly weights
    # (to measure the actual distribution of the OPTIMAL portfolio)
    gaussian_w = gaussian_result.kelly_weights
    portfolio_returns = returns @ gaussian_w  # weighted returns at Kelly weights

    # Step 3: Fit actual distribution of portfolio returns
    dist = fit_distribution(portfolio_returns, method=distribution_method)

    # Step 4: Find the fat-tail-corrected optimal Kelly for the portfolio
    kelly_actual = optimal_kelly(dist, risk_free_rate=risk_free_rate)

    # The correction ratio: how much to scale the Gaussian Kelly
    correction = kelly_actual.correction_ratio

    # Step 5: Apply correction to the weight vector
    # The direction (Markowitz) stays the same — only the scale changes
    corrected_kelly_w = gaussian_w * correction

    names = model_names or [f"Model_{i}" for i in range(n_models)]

    return KellyResult(
        kelly_weights=corrected_kelly_w,
        markowitz_weights=gaussian_result.markowitz_weights,
        gaussian_kelly_weights=gaussian_w,
        optimal_leverage=float(corrected_kelly_w.sum()),
        optimal_growth_rate=kelly_actual.expected_log_growth,
        portfolio_sharpe=gaussian_result.portfolio_sharpe,
        risk_free_rate=risk_free_rate,
        fat_tail_correction=correction,
        distribution_method=dist.name,
        n_models=n_models,
        model_names=names,
        condition_number=gaussian_result.condition_number,
        method="kelly_markowitz_fat_tailed",
    )


def model_of_models(
    backtest_returns: dict[str, np.ndarray],
    risk_free_rate: float = 0.0,
    use_fat_tails: bool = True,
    distribution_method: str = "auto",
    shrinkage: str = "none",
    shrinkage_target: float = 0.5,
    min_observations: int = 50,
) -> KellyResult:
    """
    Optimal capital allocation across trading models.

    Takes the backtest return series from each model and computes
    the Kelly-Markowitz optimal weight vector: how much capital
    to allocate to each model to maximize long-term compound growth.

    This is the "portfolio of models" optimizer. Each model is treated
    as an asset whose return series comes from its backtest. The
    optimization uses the same parameters that will be used for
    live trading — so what you optimize is what you trade.

    Args:
        backtest_returns: {model_name: returns_array} for each model.
            All arrays must have the same length (aligned dates).
        risk_free_rate: Risk-free rate (same frequency as returns).
        use_fat_tails: Apply fat-tail correction (recommended).
        distribution_method: Distribution fitting method.
        shrinkage: Covariance shrinkage ('none', 'ledoit_wolf', 'constant').
        shrinkage_target: Shrinkage intensity.
        min_observations: Minimum required observations.

    Returns:
        KellyResult with optimal allocation across models.

    Example:
        # After backtesting several models
        returns = {
            "burgess_stat_arb": burgess_daily_returns,
            "momentum_sma": sma_daily_returns,
            "pairs_kalman": kalman_daily_returns,
        }

        result = model_of_models(returns)
        print(result.summary())

        # Use corrected_kelly_weights for live trading
        for name, weight in zip(result.model_names, result.corrected_kelly_weights):
            print(f"  {name}: allocate {weight:.1%} of capital")
    """
    names = list(backtest_returns.keys())
    arrays = list(backtest_returns.values())

    # Validate alignment
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        arrays = [a[-min_len:] for a in arrays]  # align to shortest from end

    n_obs = len(arrays[0])
    if n_obs < min_observations:
        raise ValueError(
            f"Need at least {min_observations} observations, got {n_obs}"
        )

    # Stack into matrix: (n_obs, n_models)
    returns_matrix = np.column_stack(arrays)

    if use_fat_tails:
        return kelly_portfolio_fat_tailed(
            returns_matrix,
            risk_free_rate=risk_free_rate,
            model_names=names,
            distribution_method=distribution_method,
            shrinkage=shrinkage,
            shrinkage_target=shrinkage_target,
        )
    else:
        mu = returns_matrix.mean(axis=0)
        cov_mat = covariance_matrix(
            returns_matrix, shrinkage=shrinkage, shrinkage_target=shrinkage_target,
        )
        return kelly_portfolio(
            mu, cov_mat, risk_free_rate, names,
        )
