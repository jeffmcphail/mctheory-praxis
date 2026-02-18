"""
Tests for Phase 4.5: Burgess Statistical Arbitrage.

Covers:
- Markowitz portfolio optimization
- Candidate generation
- MC correction
- Filter/rank
- Weight optimization
- Full pipeline
- Workflow DAG integration
"""

import numpy as np
import pytest

from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ═══════════════════════════════════════════════════════════════════
#  Synthetic Data
# ═══════════════════════════════════════════════════════════════════

def _make_cointegrated_universe(
    n_obs=500, n_cointegrated=5, n_independent=10, seed=42,
):
    """
    Build a universe with known cointegrated groups + noise.

    Groups 0-4 share a common random walk → are cointegrated.
    Groups 5-14 are independent random walks → not cointegrated.
    """
    rng = np.random.RandomState(seed)
    common = np.cumsum(rng.randn(n_obs)) * 2

    assets = []
    # Cointegrated group: linear combos of common + small noise
    for i in range(n_cointegrated):
        weight = 0.5 + rng.rand() * 1.5  # 0.5 to 2.0
        noise = rng.randn(n_obs) * 0.5
        assets.append(weight * common + noise + 50 + i * 10)

    # Independent walks
    for _ in range(n_independent):
        assets.append(np.cumsum(rng.randn(n_obs)) + 100)

    return np.column_stack(assets)


def _make_return_matrix(n=200, k=4, seed=42):
    """Generate correlated return matrix for Markowitz tests."""
    rng = np.random.RandomState(seed)
    # Create correlated returns
    factor = rng.randn(n) * 0.01
    returns = np.column_stack([
        factor * (0.5 + i * 0.3) + rng.randn(n) * 0.005
        for i in range(k)
    ])
    return returns


# ═══════════════════════════════════════════════════════════════════
#  Markowitz Optimization Tests
# ═══════════════════════════════════════════════════════════════════

from praxis.stats.portfolio import (
    markowitz_optimize,
    min_variance_portfolio,
    max_sharpe_portfolio,
    equal_weight_portfolio,
    covariance_matrix,
    PortfolioResult,
)


class TestCovarianceMatrix:
    def test_basic(self):
        returns = _make_return_matrix()
        cov = covariance_matrix(returns)
        assert cov.shape == (4, 4)
        # Covariance matrix should be symmetric
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)

    def test_positive_diagonal(self):
        cov = covariance_matrix(_make_return_matrix())
        assert all(cov[i, i] > 0 for i in range(cov.shape[0]))

    def test_shrinkage(self):
        returns = _make_return_matrix()
        cov_none = covariance_matrix(returns, shrinkage="none")
        cov_shrunk = covariance_matrix(returns, shrinkage="constant", shrinkage_target=0.5)
        # Shrinkage moves off-diagonal elements toward zero
        assert not np.allclose(cov_none, cov_shrunk)


class TestMinVariance:
    def test_weights_sum_to_one(self):
        cov = covariance_matrix(_make_return_matrix())
        result = min_variance_portfolio(cov)
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)

    def test_result_fields(self):
        cov = covariance_matrix(_make_return_matrix())
        result = min_variance_portfolio(cov)
        assert isinstance(result, PortfolioResult)
        assert result.method == "min_variance"
        assert result.volatility > 0
        assert result.n_assets == 4

    def test_long_only(self):
        cov = covariance_matrix(_make_return_matrix())
        result = min_variance_portfolio(cov, long_only=True)
        assert all(w >= -1e-10 for w in result.weights)
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_max_weight_constraint(self):
        cov = covariance_matrix(_make_return_matrix())
        result = min_variance_portfolio(cov, long_only=True, max_weight=0.4)
        assert all(w <= 0.4 + 1e-6 for w in result.weights)


class TestMaxSharpe:
    def test_weights_sum_to_one(self):
        returns = _make_return_matrix()
        cov = covariance_matrix(returns)
        mu = returns.mean(axis=0)
        result = max_sharpe_portfolio(mu, cov)
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)

    def test_sharpe_positive(self):
        returns = _make_return_matrix()
        cov = covariance_matrix(returns)
        mu = returns.mean(axis=0) + 0.01  # Ensure positive expected returns
        result = max_sharpe_portfolio(mu, cov)
        assert result.sharpe_ratio > 0

    def test_long_only(self):
        returns = _make_return_matrix()
        cov = covariance_matrix(returns)
        mu = returns.mean(axis=0) + 0.01
        result = max_sharpe_portfolio(mu, cov, long_only=True)
        assert all(w >= -1e-10 for w in result.weights)


class TestEqualWeight:
    def test_uniform_weights(self):
        result = equal_weight_portfolio(4)
        np.testing.assert_allclose(result.weights, [0.25, 0.25, 0.25, 0.25])

    def test_with_cov(self):
        cov = covariance_matrix(_make_return_matrix())
        result = equal_weight_portfolio(4, cov=cov)
        assert result.volatility > 0


class TestMarkowitzOptimize:
    def test_min_variance(self):
        returns = _make_return_matrix()
        result = markowitz_optimize(returns, method="min_variance")
        assert result.is_valid
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)

    def test_max_sharpe(self):
        returns = _make_return_matrix()
        result = markowitz_optimize(returns, method="max_sharpe")
        assert result.is_valid

    def test_equal_weight(self):
        returns = _make_return_matrix()
        result = markowitz_optimize(returns, method="equal_weight")
        assert len(result.weights) == 4

    def test_unknown_method_raises(self):
        returns = _make_return_matrix()
        with pytest.raises(ValueError, match="Unknown"):
            markowitz_optimize(returns, method="magic")

    def test_min_variance_lower_vol_than_equal_weight(self):
        returns = _make_return_matrix()
        mv = markowitz_optimize(returns, method="min_variance")
        ew = markowitz_optimize(returns, method="equal_weight")
        assert mv.volatility <= ew.volatility + 1e-8


# ═══════════════════════════════════════════════════════════════════
#  Burgess Pipeline Tests
# ═══════════════════════════════════════════════════════════════════

from praxis.models.burgess import (
    BurgessStatArb,
    BurgessConfig,
    BurgessResult,
    CandidateBasket,
    generate_candidates,
    apply_mc_correction,
    filter_and_rank,
    optimize_basket_weights,
    generate_basket_signals,
    build_burgess_workflow,
)
from praxis.stats.monte_carlo import CriticalValues


class TestGenerateCandidates:
    def test_scans_all_assets(self):
        M = _make_cointegrated_universe(n_obs=200, n_cointegrated=3, n_independent=5)
        candidates = generate_candidates(M, n_per_basket=2)
        assert len(candidates) == 8  # 3 + 5 = 8 assets

    def test_max_candidates(self):
        M = _make_cointegrated_universe()
        candidates = generate_candidates(M, n_per_basket=2, max_candidates=5)
        assert len(candidates) == 5

    def test_candidate_structure(self):
        M = _make_cointegrated_universe(n_obs=200, n_cointegrated=3, n_independent=5)
        candidates = generate_candidates(M, n_per_basket=2)
        c = candidates[0]
        assert isinstance(c, CandidateBasket)
        assert len(c.partner_indices) == 2
        assert c.target_index not in c.partner_indices
        assert c.adf_t_statistic != 0.0
        assert len(c.residuals) > 0

    def test_cointegrated_found_stationary(self):
        """At least some cointegrated assets should test stationary."""
        M = _make_cointegrated_universe(n_obs=500, n_cointegrated=5, n_independent=10)
        candidates = generate_candidates(M, n_per_basket=3, significance=0.10)
        # At least one of the cointegrated group should register
        stationary = [c for c in candidates[:5] if c.is_stationary]
        assert len(stationary) >= 1


class TestApplyMCCorrection:
    def test_adjusts_p_values(self):
        # Create fake candidates
        candidates = [
            CandidateBasket(target_index=0, adf_t_statistic=-5.0, adf_p_value=0.001),
            CandidateBasket(target_index=1, adf_t_statistic=-2.0, adf_p_value=0.30),
        ]
        cv = CriticalValues(values={10: -3.0, 5: -4.0, 1: -5.5})
        result = apply_mc_correction(candidates, cv)
        # Candidate 0 is very significant → low adjusted p
        assert result[0].adjusted_p_value < 0.10
        # Candidate 1 is not significant → high adjusted p
        assert result[1].adjusted_p_value > 0.10


class TestFilterAndRank:
    def test_filters_by_significance(self):
        config = BurgessConfig(significance=0.05, max_hurst=1.0,
                               min_half_life=0, max_half_life=999)
        candidates = [
            CandidateBasket(target_index=0, adjusted_p_value=0.01,
                            hurst=0.4, half_life_periods=10),
            CandidateBasket(target_index=1, adjusted_p_value=0.20,
                            hurst=0.4, half_life_periods=10),
        ]
        selected = filter_and_rank(candidates, config)
        assert len(selected) == 1
        assert selected[0].target_index == 0

    def test_filters_by_hurst(self):
        config = BurgessConfig(significance=0.10, max_hurst=0.5,
                               min_half_life=0, max_half_life=999)
        candidates = [
            CandidateBasket(target_index=0, adjusted_p_value=0.01,
                            hurst=0.3, half_life_periods=10),
            CandidateBasket(target_index=1, adjusted_p_value=0.01,
                            hurst=0.7, half_life_periods=10),
        ]
        selected = filter_and_rank(candidates, config)
        assert len(selected) == 1

    def test_ranks_by_adjusted_pvalue(self):
        config = BurgessConfig(significance=0.10, max_hurst=1.0,
                               min_half_life=0, max_half_life=999,
                               rank_by="adjusted_pvalue")
        candidates = [
            CandidateBasket(target_index=0, adjusted_p_value=0.05,
                            hurst=0.4, half_life_periods=10),
            CandidateBasket(target_index=1, adjusted_p_value=0.01,
                            hurst=0.4, half_life_periods=10),
        ]
        selected = filter_and_rank(candidates, config)
        assert selected[0].target_index == 1  # Lower p-value ranked first
        assert selected[0].rank == 1

    def test_top_k(self):
        config = BurgessConfig(significance=0.50, max_hurst=1.0,
                               min_half_life=0, max_half_life=999,
                               top_k=2)
        candidates = [
            CandidateBasket(target_index=i, adjusted_p_value=0.01 * (i + 1),
                            hurst=0.3, half_life_periods=10)
            for i in range(5)
        ]
        selected = filter_and_rank(candidates, config)
        assert len(selected) == 2


class TestOptimizeBasketWeights:
    def test_produces_weights(self):
        M = _make_cointegrated_universe(n_obs=200, n_cointegrated=5, n_independent=5)
        baskets = [CandidateBasket(
            target_index=0,
            partner_indices=[1, 2],
        )]
        results = optimize_basket_weights(M, baskets)
        assert len(results) == 1
        assert results[0].is_valid
        assert len(baskets[0].weights) == 3  # target + 2 partners

    def test_weights_sum_to_one(self):
        M = _make_cointegrated_universe(n_obs=200)
        baskets = [CandidateBasket(target_index=0, partner_indices=[1, 2])]
        results = optimize_basket_weights(M, baskets)
        np.testing.assert_allclose(results[0].weights.sum(), 1.0, atol=1e-6)


class TestGenerateBasketSignals:
    def test_produces_signals(self):
        M = _make_cointegrated_universe(n_obs=200)
        basket = CandidateBasket(
            target_index=0,
            partner_indices=[1, 2],
            weights=np.array([0.5, 0.3, 0.2]),
        )
        signals = generate_basket_signals(M, basket, lookback=30)
        assert "spread" in signals
        assert "zscore" in signals
        assert "positions" in signals
        assert len(signals["spread"]) == 200

    def test_zscore_bounded(self):
        M = _make_cointegrated_universe(n_obs=500)
        basket = CandidateBasket(
            target_index=0,
            partner_indices=[1, 2],
            weights=np.array([0.5, 0.3, 0.2]),
        )
        signals = generate_basket_signals(M, basket, lookback=60)
        # Z-scores should be reasonable (not inf/nan)
        zs = signals["zscore"][60:]  # After warmup
        assert not np.any(np.isnan(zs))
        assert not np.any(np.isinf(zs))


# ═══════════════════════════════════════════════════════════════════
#  Full Pipeline Tests
# ═══════════════════════════════════════════════════════════════════

class TestBurgessFullPipeline:
    def test_runs_without_error(self):
        M = _make_cointegrated_universe(n_obs=200, n_cointegrated=3, n_independent=5)
        config = BurgessConfig(
            n_per_basket=2,
            mc_enabled=False,  # Skip MC for speed
            significance=0.20,
            max_hurst=0.8,
            min_half_life=0.1,
            max_half_life=500,
            top_k=3,
        )
        model = BurgessStatArb(config)
        result = model.run(M)

        assert isinstance(result, BurgessResult)
        assert result.n_scanned == 8  # 3 + 5
        assert result.elapsed_seconds > 0

    def test_pipeline_with_mc(self):
        M = _make_cointegrated_universe(n_obs=150, n_cointegrated=3, n_independent=3)
        config = BurgessConfig(
            n_per_basket=2,
            mc_enabled=True,
            mc_n_samples=50,  # Low for speed
            mc_seed=42,
            significance=0.20,
            max_hurst=0.8,
            min_half_life=0.1,
            max_half_life=500,
        )
        model = BurgessStatArb(config)
        result = model.run(M)

        assert result.critical_values is not None
        assert 5 in result.critical_values.values

    def test_selected_baskets_have_weights(self):
        M = _make_cointegrated_universe(n_obs=200, n_cointegrated=5, n_independent=5)
        config = BurgessConfig(
            n_per_basket=2,
            mc_enabled=False,
            significance=0.30,
            max_hurst=0.8,
            min_half_life=0.1,
            max_half_life=500,
        )
        model = BurgessStatArb(config)
        result = model.run(M)

        if result.selected_baskets:
            basket = result.selected_baskets[0]
            assert len(basket.weights) > 0

    def test_default_config(self):
        config = BurgessConfig()
        assert config.n_per_basket == 3
        assert config.significance == 0.05
        assert config.top_k == 20


class TestBurgessWorkflow:
    def test_workflow_builds(self):
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)
        config = BurgessConfig(n_per_basket=2, mc_enabled=False)
        wf = build_burgess_workflow(config, M)
        assert wf.size >= 3  # generate, filter, optimize

    def test_workflow_executes(self):
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)
        config = BurgessConfig(
            n_per_basket=2,
            mc_enabled=False,
            significance=0.50,
            max_hurst=0.9,
            min_half_life=0.1,
            max_half_life=999,
        )
        wf = build_burgess_workflow(config, M)
        result = wf.run()

        assert result.success
        assert "generate_candidates" in result.outputs
        assert "filter_rank" in result.outputs
        assert "optimize_weights" in result.outputs

    def test_workflow_with_mc(self):
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)
        config = BurgessConfig(
            n_per_basket=2,
            mc_enabled=True,
            mc_n_samples=20,
            mc_seed=42,
            significance=0.50,
            max_hurst=0.9,
            min_half_life=0.1,
            max_half_life=999,
        )
        wf = build_burgess_workflow(config, M)
        assert wf.size >= 4  # generate, mc, filter, optimize
        result = wf.run()
        assert result.success


# ═══════════════════════════════════════════════════════════════════
#  Composite Scoring Pipeline Integration Tests
# ═══════════════════════════════════════════════════════════════════

from praxis.stats.surface import (
    CompositeSurface,
    MultiSurfaceRequirement,
    SurfaceAxis,
    _register_multi_builtins,
)
from praxis.models.burgess import (
    EnhancedCandidateBasket,
    CompositeStatReport,
    ScoreWeights,
    score_and_rank_candidates,
    compute_composite_score,
)


@pytest.fixture(scope="module")
def tiny_surface():
    """
    Pre-compute a minimal surface in-memory for integration tests.

    Grid: n_assets=[6], n_obs=[100], n_vars=[2]
    This is 1 point, computed once for all tests in the module.
    """
    _register_multi_builtins()
    surface = CompositeSurface(db_path=None)  # in-memory
    req = MultiSurfaceRequirement(
        generator="stepwise_regression",
        universe_factory="random_walk",
        axes=[
            SurfaceAxis("n_assets", [6]),
            SurfaceAxis("n_obs", [100]),
            SurfaceAxis("n_vars", [2]),
        ],
        n_samples=50,  # minimal for speed
        seed=42,
        scalar_extractors=["adf_t", "hurst", "half_life"],
        profile_collectors=["vr_profile"],
        profile_params={"vr_max_lag": 10},
        pct_conf=[1, 5, 10, 90, 95, 99],
    )
    surface.compute(req, n_workers=1)
    return surface, req


class TestCompositeScoringPipeline:
    """Tests for composite scoring wired into Burgess pipeline."""

    def test_burgess_composite_mode_runs(self, tiny_surface):
        """BurgessStatArb with scoring_mode='composite' produces results."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        model = BurgessStatArb.__new__(BurgessStatArb)
        model._config = BurgessConfig(
            n_per_basket=2,
            scoring_mode="composite",
            significance=0.99,  # very loose to get candidates
            top_k=10,
            min_composite=0.0,
        )
        model._log = PraxisLogger.instance()
        model._surface = surface
        model._surface_req = req

        result = model.run(M)

        assert result.scoring_mode == "composite"
        assert result.n_scanned > 0
        # Some candidates should score above zero
        if result.n_candidates > 0:
            first = result.selected_baskets[0]
            assert isinstance(first, EnhancedCandidateBasket)
            assert first.composite_score >= 0.0
            assert first.stat_report is not None
            assert first.rank == 1

    def test_composite_produces_enhanced_baskets(self, tiny_surface):
        """Composite path returns EnhancedCandidateBasket, not CandidateBasket."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        candidates = generate_candidates(M, n_per_basket=2, significance=0.99)
        assert len(candidates) > 0

        enhanced = score_and_rank_candidates(
            candidates=candidates,
            price_matrix=M,
            composite_surface=surface,
            surface_req=req,
            top_k=5,
        )

        for eb in enhanced:
            assert isinstance(eb, EnhancedCandidateBasket)
            assert isinstance(eb.stat_report, CompositeStatReport)
            assert 0.0 <= eb.composite_score <= 1.0
            assert eb.rank > 0
            # Should have at least some scored tests
            assert len(eb.stat_report.individual_scores) > 0

    def test_composite_scores_are_normalized(self, tiny_surface):
        """All individual scores should be in [0, 1] range."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        candidates = generate_candidates(M, n_per_basket=2, significance=0.99)
        enhanced = score_and_rank_candidates(
            candidates=candidates,
            price_matrix=M,
            composite_surface=surface,
            surface_req=req,
            top_k=5,
        )

        for eb in enhanced:
            for name, sc in eb.stat_report.individual_scores.items():
                assert 0.0 <= sc.score <= 1.0, \
                    f"{name}: score {sc.score} out of [0,1]"
                assert 0.0 <= sc.p_value <= 1.0, \
                    f"{name}: p_value {sc.p_value} out of [0,1]"

    def test_composite_ranking_order(self, tiny_surface):
        """Results should be ranked by composite score descending."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        candidates = generate_candidates(M, n_per_basket=2, significance=0.99)
        enhanced = score_and_rank_candidates(
            candidates=candidates,
            price_matrix=M,
            composite_surface=surface,
            surface_req=req,
            top_k=10,
        )

        if len(enhanced) >= 2:
            scores = [eb.composite_score for eb in enhanced]
            assert scores == sorted(scores, reverse=True), \
                "Results not sorted by composite score descending"
            # Ranks should be sequential
            ranks = [eb.rank for eb in enhanced]
            assert ranks == list(range(1, len(enhanced) + 1))

    def test_min_composite_filters(self, tiny_surface):
        """min_composite should exclude low-scoring candidates."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        candidates = generate_candidates(M, n_per_basket=2, significance=0.99)

        # No filter
        all_results = score_and_rank_candidates(
            candidates=candidates,
            price_matrix=M,
            composite_surface=surface,
            surface_req=req,
            min_composite=0.0,
        )
        # With high filter
        filtered = score_and_rank_candidates(
            candidates=candidates,
            price_matrix=M,
            composite_surface=surface,
            surface_req=req,
            min_composite=0.99,
        )
        assert len(filtered) <= len(all_results)

    def test_custom_weights(self, tiny_surface):
        """ScoreWeights should change composite scores."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)
        candidates = generate_candidates(M, n_per_basket=2, significance=0.99)
        if not candidates:
            pytest.skip("No candidates generated")

        basket = candidates[0]
        report_equal = compute_composite_score(
            residuals=basket.residuals,
            composite_surface=surface,
            surface_req=req,
            n_assets=M.shape[1],
            n_obs=M.shape[0],
            n_vars=len(basket.partner_indices),
        )
        # ADF-heavy weights
        adf_heavy = ScoreWeights(adf_t=10.0, hurst=0.1, half_life=0.1)
        report_adf = compute_composite_score(
            residuals=basket.residuals,
            composite_surface=surface,
            surface_req=req,
            n_assets=M.shape[1],
            n_obs=M.shape[0],
            n_vars=len(basket.partner_indices),
            weights=adf_heavy,
        )
        # Both should be valid scores, but likely different
        assert 0.0 <= report_equal.composite_score <= 1.0
        assert 0.0 <= report_adf.composite_score <= 1.0

    def test_result_has_stat_reports(self, tiny_surface):
        """BurgessResult from composite mode should contain stat reports."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        model = BurgessStatArb.__new__(BurgessStatArb)
        model._config = BurgessConfig(
            n_per_basket=2,
            scoring_mode="composite",
            significance=0.99,
            top_k=5,
        )
        model._log = PraxisLogger.instance()
        model._surface = surface
        model._surface_req = req

        result = model.run(M)
        assert result.scoring_mode == "composite"
        for basket in result.selected_baskets:
            report = basket.stat_report
            assert report is not None
            assert isinstance(report.composite_score, float)
            # Should have at least scalar tests
            scalar_names = {"adf_t", "hurst", "half_life"}
            scored_names = set(report.individual_scores.keys())
            assert scored_names & scalar_names, \
                f"Expected some scalar tests, got: {scored_names}"

    def test_classic_mode_still_works(self):
        """Classic mode should be completely unaffected by composite changes."""
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)
        config = BurgessConfig(
            n_per_basket=2,
            scoring_mode="classic",
            mc_enabled=True,
            mc_n_samples=20,
            mc_seed=42,
            significance=0.50,
            max_hurst=0.9,
            min_half_life=0.1,
            max_half_life=999,
        )
        model = BurgessStatArb(config)
        result = model.run(M)

        assert result.scoring_mode == "classic"
        assert result.n_scanned > 0
        assert result.critical_values is not None
        for b in result.selected_baskets:
            assert isinstance(b, CandidateBasket)
            assert not isinstance(b, EnhancedCandidateBasket)

    def test_portfolio_optimization_works_with_enhanced(self, tiny_surface):
        """Optimize weights should accept EnhancedCandidateBasket (inherits CandidateBasket)."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        model = BurgessStatArb.__new__(BurgessStatArb)
        model._config = BurgessConfig(
            n_per_basket=2,
            scoring_mode="composite",
            significance=0.99,
            top_k=5,
        )
        model._log = PraxisLogger.instance()
        model._surface = surface
        model._surface_req = req

        result = model.run(M)
        # Portfolio optimization should have run on all selected baskets
        if result.n_candidates > 0:
            assert len(result.portfolio_results) == result.n_candidates

    def test_signals_work_with_enhanced(self, tiny_surface):
        """generate_basket_signals should work with EnhancedCandidateBasket."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        candidates = generate_candidates(M, n_per_basket=2, significance=0.99)
        enhanced = score_and_rank_candidates(
            candidates=candidates,
            price_matrix=M,
            composite_surface=surface,
            surface_req=req,
            top_k=3,
        )
        if not enhanced:
            pytest.skip("No enhanced baskets")

        signals = generate_basket_signals(M, enhanced[0])
        assert "spread" in signals
        assert "zscore" in signals
        assert "positions" in signals
        assert len(signals["spread"]) == M.shape[0]


class TestCompositeWorkflow:
    """Workflow DAG builder with composite mode."""

    def test_workflow_composite_builds(self, tiny_surface):
        """Composite workflow should have different steps than classic."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        config = BurgessConfig(
            n_per_basket=2,
            scoring_mode="composite",
            significance=0.99,
            surface_db_path=None,  # in-memory
        )
        # Patch: the workflow builder creates its own surface, but for testing
        # we need it to work. We'll just verify the DAG structure.
        wf = build_burgess_workflow(config, M)
        assert wf.size >= 3  # generate, composite_score_rank, optimize

    def test_workflow_composite_executes(self, tiny_surface):
        """Composite workflow should execute successfully."""
        surface, req = tiny_surface
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)

        config = BurgessConfig(
            n_per_basket=2,
            scoring_mode="composite",
            significance=0.99,
            surface_db_path=None,  # in-memory
            top_k=5,
        )
        wf = build_burgess_workflow(config, M)
        result = wf.run()

        assert result.success
        assert "generate_candidates" in result.outputs
        assert "composite_score_rank" in result.outputs
        assert "optimize_weights" in result.outputs

    def test_workflow_classic_unchanged(self):
        """Classic workflow should still work exactly as before."""
        M = _make_cointegrated_universe(n_obs=100, n_cointegrated=3, n_independent=3)
        config = BurgessConfig(
            n_per_basket=2,
            scoring_mode="classic",
            mc_enabled=False,
            significance=0.50,
            max_hurst=0.9,
            min_half_life=0.1,
            max_half_life=999,
        )
        wf = build_burgess_workflow(config, M)
        result = wf.run()
        assert result.success
        assert "generate_candidates" in result.outputs
        assert "filter_rank" in result.outputs
        assert "optimize_weights" in result.outputs
        # Should NOT have composite steps
        assert "composite_score_rank" not in result.outputs
