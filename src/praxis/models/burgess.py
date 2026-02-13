"""
Burgess Statistical Arbitrage Model (Phase 4.5, §14.3).

The full pipeline:
1. Generate candidates via successive regression
2. Run statistical tests (ADF, Johansen, Hurst, half-life)
3. Build Monte Carlo critical value surfaces (data-mining correction)
4. Rank and select top baskets by adjusted p-value
5. Optimize basket weights via Markowitz
6. Generate z-score trading signals
7. Execute single-leg

This module provides both:
- BurgessStatArb class (programmatic API)
- build_burgess_workflow() (workflow DAG builder for §10 executor)

Usage:
    model = BurgessStatArb(config)
    result = model.run(price_data)
    print(result.selected_baskets)
    print(result.signals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from praxis.logger.core import PraxisLogger
from praxis.stats import (
    adf_test,
    johansen_test,
    hurst_exponent,
    half_life,
    run_stationarity_tests,
    ADFResult,
    HurstResult,
    HalfLifeResult,
)
from praxis.stats.regression import (
    successive_regression,
    ridge_regression,
    StepwiseResult,
    RegressionResult,
)
from praxis.stats.monte_carlo import (
    generate_adf_critical_values,
    CriticalValues,
)
from praxis.stats.portfolio import (
    markowitz_optimize,
    PortfolioResult,
)
from praxis.workflow import (
    WorkflowExecutor,
    WorkflowStep,
    WorkflowFunctionRegistry,
)


# ═══════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CandidateBasket:
    """A candidate mean-reverting basket from successive regression."""
    target_index: int
    partner_indices: list[int] = field(default_factory=list)
    adf_t_statistic: float = 0.0
    adf_p_value: float = 1.0
    adjusted_p_value: float = 1.0
    hurst: float = 0.5
    half_life_periods: float = float("inf")
    r_squared: float = 0.0
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    is_stationary: bool = False
    rank: int = 0

    @property
    def all_indices(self) -> list[int]:
        return [self.target_index] + self.partner_indices

    @property
    def n_legs(self) -> int:
        return len(self.all_indices)


@dataclass
class BurgessConfig:
    """Configuration for Burgess Stat Arb pipeline."""
    # Candidate generation
    n_per_basket: int = 3
    max_candidates: int = 0          # 0 = scan all targets

    # Statistical tests
    significance: float = 0.05
    min_half_life: float = 1.0
    max_half_life: float = 252.0     # 1 year
    max_hurst: float = 0.5

    # Monte Carlo
    mc_enabled: bool = True
    mc_n_samples: int = 1000
    mc_seed: int | None = 42

    # Selection
    top_k: int = 20
    rank_by: str = "adjusted_pvalue"  # or "adf_t", "hurst", "half_life"

    # Weight optimization
    optimization_method: str = "min_variance"
    long_only: bool = False
    max_weight: float = 1.0
    shrinkage: str = "none"

    # Signals
    zscore_lookback: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5


@dataclass
class BurgessResult:
    """Result of running the full Burgess pipeline."""
    # Candidates
    n_scanned: int = 0
    n_candidates: int = 0
    candidates: list[CandidateBasket] = field(default_factory=list)

    # Selected
    selected_baskets: list[CandidateBasket] = field(default_factory=list)

    # MC correction
    critical_values: CriticalValues | None = None

    # Portfolio weights
    portfolio_results: list[PortfolioResult] = field(default_factory=list)

    # Performance
    elapsed_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════
#  Pipeline Steps
# ═══════════════════════════════════════════════════════════════════

def generate_candidates(
    price_matrix: np.ndarray,
    n_per_basket: int = 3,
    max_candidates: int = 0,
    significance: float = 0.05,
) -> list[CandidateBasket]:
    """
    Step 1: Scan universe via successive regression.

    For each target asset, find the n_per_basket most correlated
    partners, regress, and test residuals for stationarity.

    Args:
        price_matrix: (n_obs, n_assets) price matrix.
        n_per_basket: Number of partners per basket.
        max_candidates: Max targets to scan (0 = all).
        significance: ADF significance level.

    Returns:
        List of CandidateBasket with test results.
    """
    n_assets = price_matrix.shape[1]
    targets = range(n_assets) if max_candidates == 0 else range(min(max_candidates, n_assets))
    candidates = []

    for target_idx in targets:
        result = successive_regression(
            target_index=target_idx,
            asset_matrix=price_matrix,
            n_vars=n_per_basket,
            compute_stats=True,
            significance=significance,
        )

        basket = CandidateBasket(
            target_index=target_idx,
            partner_indices=result.selected_indices,
        )

        if result.adf is not None:
            basket.adf_t_statistic = result.adf.t_statistic
            basket.adf_p_value = result.adf.p_value
            basket.is_stationary = result.adf.is_stationary

        if result.regression is not None:
            basket.r_squared = result.regression.r_squared
            basket.residuals = result.regression.residuals.copy()

            # Hurst and half-life on residuals
            if len(result.regression.residuals) > 20:
                h = hurst_exponent(result.regression.residuals)
                basket.hurst = h.hurst_exponent

                hl = half_life(result.regression.residuals)
                basket.half_life_periods = hl.half_life

        candidates.append(basket)

    return candidates


def apply_mc_correction(
    candidates: list[CandidateBasket],
    critical_values: CriticalValues,
) -> list[CandidateBasket]:
    """
    Step 2: Adjust p-values using Monte Carlo critical values.

    The standard ADF tables are wrong for stepwise regression residuals
    because of data-mining bias. The MC critical values correct for this.

    Args:
        candidates: From generate_candidates().
        critical_values: From generate_adf_critical_values().

    Returns:
        Same candidates with adjusted_p_value set.
    """
    # Approximate adjusted p-value by interpolating between MC critical values
    cv = critical_values.values  # {10: -X, 5: -Y, 1: -Z}
    sorted_pcts = sorted(cv.keys(), reverse=True)  # [10, 5, 1]

    for basket in candidates:
        t = basket.adf_t_statistic

        # Find where t falls in the critical value scale
        if not sorted_pcts:
            basket.adjusted_p_value = basket.adf_p_value
            continue

        # More negative t = more significant
        if t <= cv[sorted_pcts[-1]]:
            # More extreme than most stringent
            basket.adjusted_p_value = sorted_pcts[-1] / 100.0 * 0.5
        elif t >= cv[sorted_pcts[0]]:
            # Less extreme than least stringent
            basket.adjusted_p_value = sorted_pcts[0] / 100.0 * 2.0
        else:
            # Interpolate
            for i in range(len(sorted_pcts) - 1):
                pct_high = sorted_pcts[i]
                pct_low = sorted_pcts[i + 1]
                if cv[pct_high] <= t <= cv[pct_low] or cv[pct_low] <= t <= cv[pct_high]:
                    # Linear interpolation in t-stat space
                    frac = (t - cv[pct_high]) / (cv[pct_low] - cv[pct_high]) if cv[pct_low] != cv[pct_high] else 0.5
                    basket.adjusted_p_value = (pct_high + frac * (pct_low - pct_high)) / 100.0
                    break
            else:
                basket.adjusted_p_value = basket.adf_p_value

    return candidates


def filter_and_rank(
    candidates: list[CandidateBasket],
    config: BurgessConfig,
) -> list[CandidateBasket]:
    """
    Step 3: Filter by quality criteria and rank.

    Filters:
    - Must be stationary (adjusted p-value < significance)
    - Half-life within bounds
    - Hurst < threshold

    Ranking: by adjusted_pvalue (ascending = most significant first),
    or by adf_t (most negative first), etc.
    """
    filtered = []
    for b in candidates:
        p = b.adjusted_p_value if b.adjusted_p_value < 1.0 else b.adf_p_value
        if p >= config.significance:
            continue
        if not (config.min_half_life <= b.half_life_periods <= config.max_half_life):
            continue
        if b.hurst >= config.max_hurst:
            continue
        filtered.append(b)

    # Rank
    if config.rank_by == "adjusted_pvalue":
        filtered.sort(key=lambda b: b.adjusted_p_value)
    elif config.rank_by == "adf_t":
        filtered.sort(key=lambda b: b.adf_t_statistic)
    elif config.rank_by == "hurst":
        filtered.sort(key=lambda b: b.hurst)
    elif config.rank_by == "half_life":
        filtered.sort(key=lambda b: b.half_life_periods)

    for i, b in enumerate(filtered):
        b.rank = i + 1

    return filtered[:config.top_k]


def optimize_basket_weights(
    price_matrix: np.ndarray,
    baskets: list[CandidateBasket],
    method: str = "min_variance",
    long_only: bool = False,
    max_weight: float = 1.0,
    shrinkage: str = "none",
) -> list[PortfolioResult]:
    """
    Step 4: Optimize weights for each selected basket.

    Computes returns for basket assets and runs Markowitz.
    """
    results = []
    for basket in baskets:
        indices = basket.all_indices
        basket_prices = price_matrix[:, indices]

        # Compute returns
        returns = np.diff(basket_prices, axis=0) / basket_prices[:-1]

        if returns.shape[0] < returns.shape[1] + 2:
            # Not enough data
            results.append(PortfolioResult(
                weights=np.ones(len(indices)) / len(indices),
                method="equal_weight_fallback",
                n_assets=len(indices),
            ))
            continue

        pr = markowitz_optimize(
            returns,
            method=method,
            long_only=long_only,
            max_weight=max_weight,
            shrinkage=shrinkage,
        )
        basket.weights = pr.weights.copy()
        results.append(pr)

    return results


def generate_basket_signals(
    price_matrix: np.ndarray,
    basket: CandidateBasket,
    lookback: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Step 5: Generate z-score signals for a basket.

    Computes the weighted spread, then z-score, then positions.

    Returns dict with: spread, zscore, positions
    """
    indices = basket.all_indices
    weights = basket.weights if len(basket.weights) == len(indices) else np.ones(len(indices)) / len(indices)

    # Weighted spread: target - sum(weight_i * partner_i)
    spread = price_matrix[:, basket.target_index].copy()
    for i, partner_idx in enumerate(basket.partner_indices):
        w_idx = i + 1 if i + 1 < len(weights) else 0
        spread -= weights[w_idx] * price_matrix[:, partner_idx]

    n = len(spread)

    # Rolling mean and std
    zscore = np.zeros(n)
    for i in range(lookback, n):
        window = spread[i - lookback:i]
        mu = window.mean()
        sigma = window.std()
        if sigma > 1e-10:
            zscore[i] = (spread[i] - mu) / sigma

    # Positions
    positions = np.zeros(n)
    positions[zscore >= entry_threshold] = -1    # Short spread
    positions[zscore <= -entry_threshold] = 1    # Long spread
    positions[zscore <= exit_threshold] = np.where(
        positions[zscore <= exit_threshold] == -1, 0,
        positions[zscore <= exit_threshold]
    )
    positions[zscore >= -exit_threshold] = np.where(
        positions[zscore >= -exit_threshold] == 1, 0,
        positions[zscore >= -exit_threshold]
    )

    return {
        "spread": spread,
        "zscore": zscore,
        "positions": positions,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════

class BurgessStatArb:
    """
    Full Burgess Statistical Arbitrage pipeline.

    Chains: candidates → MC correction → filter/rank → optimize → signals.
    """

    def __init__(self, config: BurgessConfig | None = None):
        self._config = config or BurgessConfig()
        self._log = PraxisLogger.instance()

    @property
    def config(self) -> BurgessConfig:
        return self._config

    def run(self, price_matrix: np.ndarray) -> BurgessResult:
        """
        Execute the full pipeline.

        Args:
            price_matrix: (n_obs, n_assets) price matrix.

        Returns:
            BurgessResult with candidates, selected baskets, weights.
        """
        import time
        t0 = time.monotonic()
        cfg = self._config
        result = BurgessResult()

        # Step 1: Generate candidates
        self._log.info(
            f"Burgess: scanning {price_matrix.shape[1]} assets, "
            f"n_per_basket={cfg.n_per_basket}",
            tags={"burgess"},
        )
        candidates = generate_candidates(
            price_matrix,
            n_per_basket=cfg.n_per_basket,
            max_candidates=cfg.max_candidates,
            significance=cfg.significance,
        )
        result.n_scanned = len(candidates)
        result.candidates = candidates

        # Step 2: Monte Carlo correction
        if cfg.mc_enabled:
            n_assets = price_matrix.shape[1]
            n_obs = price_matrix.shape[0]
            mc = generate_adf_critical_values(
                n_assets=n_assets,
                n_obs=n_obs,
                n_vars=cfg.n_per_basket,
                n_samples=cfg.mc_n_samples,
                seed=cfg.mc_seed,
            )
            result.critical_values = mc.critical_values
            candidates = apply_mc_correction(candidates, mc.critical_values)

        # Step 3: Filter and rank
        selected = filter_and_rank(candidates, cfg)
        result.selected_baskets = selected
        result.n_candidates = len(selected)

        # Step 4: Optimize weights
        if selected:
            result.portfolio_results = optimize_basket_weights(
                price_matrix, selected,
                method=cfg.optimization_method,
                long_only=cfg.long_only,
                max_weight=cfg.max_weight,
                shrinkage=cfg.shrinkage,
            )

        result.elapsed_seconds = time.monotonic() - t0

        self._log.info(
            f"Burgess: {result.n_scanned} scanned → "
            f"{result.n_candidates} selected in {result.elapsed_seconds:.1f}s",
            tags={"burgess"},
        )

        return result


# ═══════════════════════════════════════════════════════════════════
#  Workflow DAG Builder (§10 integration)
# ═══════════════════════════════════════════════════════════════════

def build_burgess_workflow(
    config: BurgessConfig,
    price_matrix: np.ndarray,
) -> WorkflowExecutor:
    """
    Build a §10 workflow DAG for the Burgess pipeline.

    Steps:
      generate_candidates → mc_correction → filter_rank → optimize → signals

    This is the declarative version suitable for the workflow executor.
    """
    reg = WorkflowFunctionRegistry()
    cfg = config

    # Register pipeline functions with captured config/data
    reg.register("generate_candidates", lambda: generate_candidates(
        price_matrix,
        n_per_basket=cfg.n_per_basket,
        max_candidates=cfg.max_candidates,
        significance=cfg.significance,
    ))

    reg.register("mc_correction", lambda candidates=None, **kw: (
        apply_mc_correction(
            candidates,
            generate_adf_critical_values(
                n_assets=price_matrix.shape[1],
                n_obs=price_matrix.shape[0],
                n_vars=cfg.n_per_basket,
                n_samples=cfg.mc_n_samples,
                seed=cfg.mc_seed,
            ).critical_values,
        ) if candidates else []
    ))

    reg.register("filter_rank", lambda candidates=None, **kw: (
        filter_and_rank(candidates or [], cfg)
    ))

    reg.register("optimize_weights", lambda baskets=None, **kw: (
        optimize_basket_weights(
            price_matrix, baskets or [],
            method=cfg.optimization_method,
            long_only=cfg.long_only,
            max_weight=cfg.max_weight,
        )
    ))

    # Build DAG
    wf = WorkflowExecutor(registry=reg)

    wf.add_step(WorkflowStep(
        id="generate_candidates",
        function="generate_candidates",
    ))

    if cfg.mc_enabled:
        wf.add_step(WorkflowStep(
            id="mc_correction",
            function="mc_correction",
            params={"candidates": "generate_candidates.output"},
            depends_on=["generate_candidates"],
        ))
        rank_dep = "mc_correction"
    else:
        rank_dep = "generate_candidates"

    wf.add_step(WorkflowStep(
        id="filter_rank",
        function="filter_rank",
        params={"candidates": f"{rank_dep}.output"},
        depends_on=[rank_dep],
    ))

    wf.add_step(WorkflowStep(
        id="optimize_weights",
        function="optimize_weights",
        params={"baskets": "filter_rank.output"},
        depends_on=["filter_rank"],
    ))

    return wf
