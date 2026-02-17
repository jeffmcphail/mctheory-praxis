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


# ═══════════════════════════════════════════════════════════════════
#  Composite Statistical Score
# ═══════════════════════════════════════════════════════════════════
#
#  Normalizes all test statistics to a common [0, 1] scale via
#  percentile rank against the null surface, then combines them
#  with configurable weights.
#
#  The key insight: the empirical p-value from the MC surface IS
#  the natural normalization. A p-value of 0.01 means "only 1% of
#  null (data-mined) residuals were this extreme." Converting to
#  a score: score = 1 - p_value, so p=0.01 → score=0.99.
#
#  This works regardless of the raw statistic's scale (ADF t-values
#  are negative, Hurst is [0,1], Mahalanobis is positive, etc.)
#  because we're comparing against each test's own null distribution.
# ═══════════════════════════════════════════════════════════════════


@dataclass
class StatScore:
    """Single test statistic with its raw value and normalized score."""
    name: str
    raw_value: float
    p_value: float          # Empirical p-value from MC surface
    score: float            # 1 - p_value → [0, 1], higher = more significant
    tail: str = "left"      # "left" or "right" — which tail indicates significance

    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


@dataclass
class CompositeStatReport:
    """
    Full statistical assessment of a candidate basket.

    Contains individual test scores, the weighted composite,
    and optionally the raw VR profile artifact for LLM analysis.
    """
    individual_scores: dict[str, StatScore] = field(default_factory=dict)
    composite_score: float = 0.0
    weights_used: dict[str, float] = field(default_factory=dict)

    # Optional: VR profile data for qualitative analysis
    vr_profile: np.ndarray | None = None
    vr_artifact: Any = None  # ProfileArtifact if available

    @property
    def n_significant(self) -> int:
        """Number of tests significant at 5%."""
        return sum(1 for s in self.individual_scores.values() if s.p_value < 0.05)

    @property
    def n_tests(self) -> int:
        return len(self.individual_scores)

    def summary(self) -> str:
        """Human-readable summary for logging or LLM context."""
        lines = [f"Composite Score: {self.composite_score:.4f} ({self.n_significant}/{self.n_tests} significant at 5%)"]
        for name, sc in sorted(self.individual_scores.items(), key=lambda x: -x[1].score):
            sig = "✓" if sc.p_value < 0.05 else "·"
            lines.append(f"  {sig} {name:25s}: raw={sc.raw_value:10.4f}  p={sc.p_value:.4f}  score={sc.score:.4f}  weight={self.weights_used.get(name, 0):.3f}")
        return "\n".join(lines)


@dataclass
class ScoreWeights:
    """
    Configurable weights for combining test statistics.

    Defaults to equal weighting. In practice, optimize these
    per (asset_class, timescale, universe_size, history_length)
    via backtesting.

    Weights are normalized to sum to 1.0 at score computation time,
    so absolute values don't matter — only ratios.
    """
    adf_t: float = 1.0
    hurst: float = 1.0
    half_life: float = 1.0
    johansen_trace: float = 1.0
    vr_eigen1_proj: float = 0.5      # Level shift — less informative
    vr_eigen2_proj: float = 2.0      # Curvature — the star performer
    vr_eigen3_proj: float = 0.5      # Higher-order — noisy
    vr_mahalanobis: float = 1.5      # Joint multivariate deviation

    def as_dict(self) -> dict[str, float]:
        return {
            "adf_t": self.adf_t,
            "hurst": self.hurst,
            "half_life": self.half_life,
            "johansen_trace": self.johansen_trace,
            "vr_eigen1_proj": self.vr_eigen1_proj,
            "vr_eigen2_proj": self.vr_eigen2_proj,
            "vr_eigen3_proj": self.vr_eigen3_proj,
            "vr_mahalanobis": self.vr_mahalanobis,
        }

    def normalized(self) -> dict[str, float]:
        """Weights normalized to sum to 1."""
        d = self.as_dict()
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total > 0 else d


# ═══════════════════════════════════════════════════════════════════
#  Percentile Rank Computation
# ═══════════════════════════════════════════════════════════════════

# Test metadata: which tail indicates significance
_TEST_TAIL = {
    "adf_t": "left",           # More negative = more stationary
    "hurst": "left",           # Lower = more mean-reverting
    "half_life": "left",       # Shorter = faster reversion
    "johansen_trace": "right", # Higher = more cointegrated
    "vr_eigen1_proj": "left",  # More negative = stronger level shift
    "vr_eigen2_proj": "left",  # More negative = stronger curvature (MR signature)
    "vr_eigen3_proj": "left",  # More negative = stronger higher-order mode
    "vr_mahalanobis": "right", # Higher = more anomalous vs null
}


def _p_value_from_surface(
    raw_value: float,
    critical_values: dict[int, float],
    tail: str = "left",
) -> float:
    """
    Compute empirical p-value by interpolating between surface percentiles.

    The surface stores critical values at [1, 5, 10, 90, 95, 99] percentiles.
    We interpolate to estimate where the raw value falls in the null CDF.

    For left-tail tests (ADF, Hurst): p-value = percentile rank (lower = more extreme)
    For right-tail tests (Johansen, Mahalanobis): p-value = 1 - percentile rank
    """
    if not critical_values:
        return 0.5  # No surface data — uninformative

    # Build sorted (percentile, cv_value) pairs
    sorted_pcts = sorted(critical_values.keys())
    sorted_cvs = [critical_values[p] for p in sorted_pcts]
    pcts_frac = [p / 100.0 for p in sorted_pcts]  # Convert to [0, 1]

    if tail == "left":
        # For left-tail: raw value ≤ cv at percentile p means percentile rank ≈ p
        # Lower percentile = more extreme = smaller p-value
        if raw_value <= sorted_cvs[0]:
            # More extreme than the 1st percentile
            p_value = pcts_frac[0] * 0.5  # Extrapolate: ~0.005
        elif raw_value >= sorted_cvs[-1]:
            # Less extreme than the 99th percentile — not significant
            p_value = pcts_frac[-1] + (1 - pcts_frac[-1]) * 0.5
        else:
            # Interpolate
            p_value = float(np.interp(raw_value, sorted_cvs, pcts_frac))

    elif tail == "right":
        # For right-tail: raw value ≥ cv at percentile p means p-value ≈ 1-p
        # Higher percentile = more extreme = smaller p-value
        if raw_value >= sorted_cvs[-1]:
            # More extreme than 99th percentile
            p_value = (1 - pcts_frac[-1]) * 0.5  # ~0.005
        elif raw_value <= sorted_cvs[0]:
            # Less extreme than 1st percentile — not significant
            p_value = 1 - pcts_frac[0] * 0.5
        else:
            percentile = float(np.interp(raw_value, sorted_cvs, pcts_frac))
            p_value = 1.0 - percentile
    else:
        p_value = 0.5

    return max(min(p_value, 1.0), 0.0)


def compute_composite_score(
    residuals: np.ndarray,
    composite_surface,  # CompositeSurface instance
    surface_req,        # MultiSurfaceRequirement
    n_assets: int,
    n_obs: int,
    n_vars: int,
    weights: ScoreWeights | None = None,
) -> CompositeStatReport:
    """
    Compute the full normalized composite score for a candidate basket.

    1. Extract all raw test statistics from residuals
    2. Look up surface critical values for each test
    3. Compute empirical p-value via interpolation
    4. Convert to score = 1 - p_value
    5. Weighted combination

    Args:
        residuals: 1D residual series from stepwise regression.
        composite_surface: CompositeSurface with computed surfaces.
        surface_req: MultiSurfaceRequirement defining the grid.
        n_assets: Universe size (for surface lookup).
        n_obs: Number of observations.
        n_vars: Number of regression variables.
        weights: ScoreWeights for combination. None = equal weights.

    Returns:
        CompositeStatReport with individual and composite scores.
    """
    from praxis.stats.surface import (
        SurfaceRegistry,
        _PROFILE_REGISTRY,
        _register_multi_builtins,
    )
    _register_multi_builtins()

    w = weights or ScoreWeights()
    w_norm = w.normalized()
    query_params = dict(n_assets=n_assets, n_obs=n_obs, n_vars=n_vars)

    report = CompositeStatReport()
    report.weights_used = w_norm

    # ── Scalar tests ─────────────────────────────────────────
    for ext_name in surface_req.scalar_extractors:
        try:
            extractor = SurfaceRegistry.get_extractor(ext_name)
            raw_value = extractor.extract(residuals)
        except Exception:
            continue

        try:
            cvs = composite_surface.query_scalar(surface_req, ext_name, **query_params)
        except Exception:
            cvs = {}

        tail = _TEST_TAIL.get(ext_name, "left")
        p_val = _p_value_from_surface(raw_value, cvs, tail)
        score = 1.0 - p_val

        report.individual_scores[ext_name] = StatScore(
            name=ext_name,
            raw_value=raw_value,
            p_value=p_val,
            score=score,
            tail=tail,
        )

    # ── Profile-derived tests ────────────────────────────────
    for coll_name in surface_req.profile_collectors:
        coll = _PROFILE_REGISTRY.get(coll_name)
        if coll is None:
            continue

        merged_params = {**query_params, **surface_req.profile_params}
        profile = coll.collect(residuals, merged_params)
        if profile is None:
            continue

        report.vr_profile = profile

        artifact = composite_surface.query_artifact(surface_req, coll_name, **query_params)
        if artifact is None:
            continue

        report.vr_artifact = artifact

        # Compute derived statistics from profile
        candidate_stats = artifact.all_projections(profile)

        for stat_name, raw_value in candidate_stats.items():
            full_name = f"vr_{stat_name}"
            try:
                cvs = composite_surface.query_scalar(surface_req, full_name, **query_params)
            except Exception:
                cvs = {}

            tail = _TEST_TAIL.get(full_name, "left")
            p_val = _p_value_from_surface(raw_value, cvs, tail)
            score = 1.0 - p_val

            report.individual_scores[full_name] = StatScore(
                name=full_name,
                raw_value=raw_value,
                p_value=p_val,
                score=score,
                tail=tail,
            )

    # ── Weighted composite ───────────────────────────────────
    total_weight = 0.0
    weighted_sum = 0.0
    for name, sc in report.individual_scores.items():
        wt = w_norm.get(name, 0.0)
        if wt > 0:
            weighted_sum += wt * sc.score
            total_weight += wt

    report.composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return report


# ═══════════════════════════════════════════════════════════════════
#  Enhanced CandidateBasket
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EnhancedCandidateBasket(CandidateBasket):
    """
    CandidateBasket with composite scoring and LLM analysis fields.

    Extends the base with:
    - Full CompositeStatReport (all tests, normalized scores)
    - LLM qualitative analysis results
    - Override flags from qualitative review
    """
    stat_report: CompositeStatReport | None = None
    composite_score: float = 0.0

    # LLM qualitative analysis
    llm_recommendation: str = ""          # "include", "exclude", "flag_for_review"
    llm_confidence: float = 0.0           # [0, 1]
    llm_reasoning: str = ""
    llm_suggested_horizon: str = ""       # e.g., "3-5 days", "2-4 weeks"
    llm_risk_flags: list[str] = field(default_factory=list)
    llm_override: bool = False            # True if LLM recommends changing rank

    @property
    def final_score(self) -> float:
        """Score used for final ranking after LLM review."""
        if self.llm_override and self.llm_recommendation == "exclude":
            return -1.0  # Demoted
        if self.llm_override and self.llm_recommendation == "include":
            return self.composite_score + 0.1  # Boosted slightly
        return self.composite_score


# ═══════════════════════════════════════════════════════════════════
#  Enhanced Pipeline with Composite Scoring
# ═══════════════════════════════════════════════════════════════════


def score_and_rank_candidates(
    candidates: list[CandidateBasket],
    price_matrix: np.ndarray,
    composite_surface,
    surface_req,
    weights: ScoreWeights | None = None,
    top_k: int = 20,
    min_composite: float = 0.0,
    min_significant_tests: int = 0,
) -> list[EnhancedCandidateBasket]:
    """
    Replacement for filter_and_rank() that uses composite scoring.

    Instead of filtering by individual thresholds (significance, max_hurst,
    half_life bounds) and ranking by a single statistic, this:

    1. Computes ALL test statistics per candidate
    2. Normalizes via surface lookup → [0, 1] per test
    3. Combines with configurable weights → composite score
    4. Ranks by composite score
    5. Optionally filters by min_composite or min_significant_tests

    Args:
        candidates: Raw candidates from generate_candidates().
        price_matrix: Original price data (for universe context).
        composite_surface: CompositeSurface with precomputed surfaces.
        surface_req: MultiSurfaceRequirement.
        weights: ScoreWeights (None = default weights).
        top_k: Maximum baskets to return.
        min_composite: Minimum composite score to include.
        min_significant_tests: Minimum number of individually significant tests.

    Returns:
        Ranked list of EnhancedCandidateBasket, best first.
    """
    n_assets = price_matrix.shape[1]
    n_obs = price_matrix.shape[0]
    enhanced = []

    for basket in candidates:
        if len(basket.residuals) < 30:
            continue

        # Compute full composite score
        report = compute_composite_score(
            residuals=basket.residuals,
            composite_surface=composite_surface,
            surface_req=surface_req,
            n_assets=n_assets,
            n_obs=n_obs,
            n_vars=len(basket.partner_indices),
            weights=weights,
        )

        # Create enhanced basket
        eb = EnhancedCandidateBasket(
            target_index=basket.target_index,
            partner_indices=basket.partner_indices,
            adf_t_statistic=basket.adf_t_statistic,
            adf_p_value=basket.adf_p_value,
            adjusted_p_value=basket.adjusted_p_value,
            hurst=basket.hurst,
            half_life_periods=basket.half_life_periods,
            r_squared=basket.r_squared,
            weights=basket.weights,
            residuals=basket.residuals,
            is_stationary=basket.is_stationary,
            stat_report=report,
            composite_score=report.composite_score,
        )

        # Filter
        if report.composite_score < min_composite:
            continue
        if report.n_significant < min_significant_tests:
            continue

        enhanced.append(eb)

    # Rank by composite score (descending)
    enhanced.sort(key=lambda b: -b.composite_score)

    for i, b in enumerate(enhanced):
        b.rank = i + 1

    return enhanced[:top_k]


# ═══════════════════════════════════════════════════════════════════
#  LLM Qualitative Analysis
# ═══════════════════════════════════════════════════════════════════


@dataclass
class QualitativeAnalysisRequest:
    """
    Structured request for the Research Praxis Agent to review
    quantitative results with qualitative judgment.
    """
    # Quantitative context
    baskets: list[EnhancedCandidateBasket] = field(default_factory=list)
    asset_names: list[str] | None = None
    asset_class: str = "equity"
    timescale: str = "daily"
    universe_size: int = 0
    history_length: int = 0

    # What we want the LLM to evaluate
    top_k_quantitative: int = 10
    near_miss_count: int = 5    # Baskets ranked just below top_k

    # Optional context
    market_regime: str = ""     # "bull", "bear", "sideways", "volatile"
    recent_events: str = ""     # Free text about recent news
    trading_constraints: str = ""  # e.g., "long-only", "max 5 day hold"

    def build_prompt(self) -> str:
        """
        Build the structured prompt for the Research Praxis Agent.

        Provides all quantitative data plus context, asks for
        qualitative assessment with structured output.
        """
        sections = []

        sections.append("# Quantitative Trading Model Review Request\n")
        sections.append(
            "You are a senior quantitative trader reviewing candidate "
            "mean-reversion baskets identified by the Burgess statistical "
            "arbitrage model. Your job is to apply qualitative judgment "
            "that the quantitative scoring cannot capture.\n"
        )

        # Context
        sections.append("## Market Context")
        sections.append(f"- Asset class: {self.asset_class}")
        sections.append(f"- Timescale: {self.timescale}")
        sections.append(f"- Universe size: {self.universe_size} assets")
        sections.append(f"- History length: {self.history_length} observations")
        if self.market_regime:
            sections.append(f"- Current regime: {self.market_regime}")
        if self.recent_events:
            sections.append(f"- Recent events: {self.recent_events}")
        if self.trading_constraints:
            sections.append(f"- Constraints: {self.trading_constraints}")
        sections.append("")

        # Top quantitative picks
        sections.append(f"## Top {self.top_k_quantitative} Quantitative Picks\n")
        for i, b in enumerate(self.baskets[:self.top_k_quantitative]):
            name = self._basket_name(b)
            sections.append(f"### Basket #{b.rank}: {name}")
            sections.append(f"Composite score: {b.composite_score:.4f}")
            if b.stat_report:
                sections.append(b.stat_report.summary())

            # VR profile shape description
            if b.stat_report and b.stat_report.vr_profile is not None:
                vr = b.stat_report.vr_profile
                sections.append(f"VR profile: starts {vr[0]:.3f} at lag 2, "
                                f"{'sags to' if vr[-1] < vr[0] else 'rises to'} "
                                f"{vr[-1]:.3f} at max lag. "
                                f"Min VR={vr.min():.3f} at lag {np.argmin(vr)+2}.")
            sections.append("")

        # Near misses
        near_start = self.top_k_quantitative
        near_end = near_start + self.near_miss_count
        near_misses = self.baskets[near_start:near_end]
        if near_misses:
            sections.append(f"\n## Near Misses (ranked #{near_start+1}-{near_end})\n")
            for b in near_misses:
                name = self._basket_name(b)
                sections.append(f"### Basket #{b.rank}: {name}")
                sections.append(f"Composite score: {b.composite_score:.4f}")
                if b.stat_report:
                    sections.append(b.stat_report.summary())
                if b.stat_report and b.stat_report.vr_profile is not None:
                    vr = b.stat_report.vr_profile
                    sections.append(f"VR profile min={vr.min():.3f} at lag {np.argmin(vr)+2}")
                sections.append("")

        # Instructions
        sections.append("\n## Your Analysis\n")
        sections.append(
            "For each basket in the top picks AND the near misses, provide:\n\n"
            "1. **recommendation**: 'include' | 'exclude' | 'flag_for_review'\n"
            "2. **confidence**: 0.0-1.0 in your recommendation\n"
            "3. **reasoning**: Brief explanation (2-3 sentences)\n"
            "4. **suggested_horizon**: Optimal holding period based on VR profile shape\n"
            "5. **risk_flags**: Any concerns (earnings, regime change, liquidity, etc.)\n"
            "6. **override**: true if you recommend changing this basket's rank\n\n"
            "Pay special attention to:\n"
            "- VR profiles that show extreme mean-reversion at a specific timescale "
            "(even if overall composite is mediocre)\n"
            "- Baskets where individual tests strongly disagree (some very significant, "
            "others not at all)\n"
            "- Assets with known upcoming catalysts (earnings, splits, regulatory)\n"
            "- Near misses that might be better than top picks under specific conditions\n"
            "- Any basket where the half-life suggests a different trading horizon than "
            "the default\n\n"
            "Respond in JSON format:\n"
            "```json\n"
            "[\n"
            '  {"basket_rank": 1, "recommendation": "include", "confidence": 0.85, '
            '"reasoning": "...", "suggested_horizon": "3-5 days", '
            '"risk_flags": ["earnings in 2 days"], "override": false},\n'
            "  ...\n"
            "]\n"
            "```"
        )

        return "\n".join(sections)

    def _basket_name(self, basket: EnhancedCandidateBasket) -> str:
        """Generate a human-readable basket name."""
        if self.asset_names:
            indices = basket.all_indices
            names = [self.asset_names[i] if i < len(self.asset_names)
                     else f"Asset_{i}" for i in indices]
            return " + ".join(names)
        return f"Target_{basket.target_index} + Partners_{basket.partner_indices}"


def apply_llm_analysis(
    baskets: list[EnhancedCandidateBasket],
    llm_response: list[dict[str, Any]],
) -> list[EnhancedCandidateBasket]:
    """
    Apply LLM qualitative analysis results to enhanced baskets.

    Args:
        baskets: Ranked EnhancedCandidateBaskets.
        llm_response: Parsed JSON from LLM (list of dicts with
                      basket_rank, recommendation, confidence, etc.)

    Returns:
        Same baskets with LLM fields populated, re-sorted by final_score.
    """
    # Index by rank for matching
    by_rank = {b.rank: b for b in baskets}

    for entry in llm_response:
        rank = entry.get("basket_rank")
        if rank not in by_rank:
            continue

        b = by_rank[rank]
        b.llm_recommendation = entry.get("recommendation", "")
        b.llm_confidence = entry.get("confidence", 0.0)
        b.llm_reasoning = entry.get("reasoning", "")
        b.llm_suggested_horizon = entry.get("suggested_horizon", "")
        b.llm_risk_flags = entry.get("risk_flags", [])
        b.llm_override = entry.get("override", False)

    # Re-sort by final_score (which accounts for overrides)
    baskets.sort(key=lambda b: -b.final_score)

    # Re-rank
    for i, b in enumerate(baskets):
        b.rank = i + 1

    return baskets
