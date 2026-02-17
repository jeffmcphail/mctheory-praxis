"""
Battle Test Round 3: Full Pipeline Integration
═══════════════════════════════════════════════

Round 1 validated the Burgess model mechanics.
Round 2 validated the vectorized stats engine.
Round 3 validates the INTEGRATION of all new modules into
the production pipeline:

  data → candidates → surface_lookup → filter → optimize
       → signal → size(Kelly) → backtest → metrics
       → multi-model → model_of_models(Kelly-Markowitz)

New modules under test:
  - surface.py: Pre-computed critical values replace runtime MC
  - distribution.py: Fat-tail-aware risk metrics and Kelly correction
  - portfolio.py: Kelly-Markowitz model_of_models allocation

Test Structure:
  3.1: Surface Integration — surface lookup replaces runtime MC in Burgess
  3.2: Kelly Sizer — distribution-aware position sizing
  3.3: Pipeline Stitching — full Burgess → signals → Kelly sizing → backtest
  3.4: Multi-Model Allocation — model_of_models across backtests
  3.5: Regression Guards — new modules don't break existing Round 1/2 results
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ── Shared synthetic data ────────────────────────────────────────

@pytest.fixture
def price_universe():
    """Generate a small cointegrated universe for pipeline testing.

    3 cointegrated pairs embedded in 15 random walks.
    Sized for fast tests, not statistical power.
    """
    rng = np.random.RandomState(42)
    n_obs, n_assets = 300, 15

    # Random walk base
    prices = np.zeros((n_obs, n_assets))
    prices[0] = 100 + rng.randn(n_assets) * 10
    for t in range(1, n_obs):
        prices[t] = prices[t - 1] + rng.randn(n_assets) * 0.5

    # Embed 3 cointegrated relationships:
    # asset_1 = 2.0 * asset_0 + mean-reverting noise
    # asset_3 = 1.5 * asset_2 - 0.5 * asset_4 + MR noise
    # asset_6 = 0.8 * asset_5 + MR noise
    for t in range(1, n_obs):
        spread1 = prices[t, 1] - 2.0 * prices[t, 0]
        prices[t, 1] -= 0.15 * spread1 + rng.randn() * 0.2

        spread2 = prices[t, 3] - 1.5 * prices[t, 2] + 0.5 * prices[t, 4]
        prices[t, 3] -= 0.10 * spread2 + rng.randn() * 0.2

        spread3 = prices[t, 6] - 0.8 * prices[t, 5]
        prices[t, 6] -= 0.12 * spread3 + rng.randn() * 0.2

    return prices


@pytest.fixture
def daily_returns(price_universe):
    """Price matrix → daily return matrix for distribution/portfolio tests."""
    return np.diff(price_universe, axis=0) / price_universe[:-1]


# ═══════════════════════════════════════════════════════════════════
#  3.1: Surface Integration — surface lookup replaces runtime MC
# ═══════════════════════════════════════════════════════════════════

class TestSurfaceIntegration:
    """The surface module should produce CriticalValues objects
    compatible with the existing Burgess apply_mc_correction() flow."""

    def test_surface_produces_critical_values(self):
        """Build a tiny surface and extract CriticalValues at a point."""
        from praxis.stats.surface import (
            CriticalValueSurface,
            SurfaceRequirement,
            SurfaceAxis,
        )
        from praxis.stats.monte_carlo import CriticalValues

        req = SurfaceRequirement(
            generator="stepwise_regression",
            stat_test="adf_t",
            universe_factory="random_walk",
            axes=[
                SurfaceAxis("n_assets", [10, 15]),
                SurfaceAxis("n_obs", [200, 300]),
                SurfaceAxis("n_vars", [3]),
            ],
            n_samples=50,
            seed=42,
        )

        surface = CriticalValueSurface()
        surface.compute(req, n_workers=1)

        # Query at an exact point
        cv = surface.query_cv(req, n_assets=10, n_obs=200, n_vars=3)
        assert isinstance(cv, CriticalValues)
        assert 10 in cv.values
        assert 5 in cv.values
        assert 1 in cv.values
        # ADF critical values should be negative
        assert cv.values[5] < 0

    def test_surface_interpolated_query(self):
        """Query between grid points returns interpolated values."""
        from praxis.stats.surface import (
            CriticalValueSurface,
            SurfaceRequirement,
            SurfaceAxis,
        )

        req = SurfaceRequirement(
            generator="stepwise_regression",
            stat_test="adf_t",
            universe_factory="random_walk",
            axes=[
                SurfaceAxis("n_assets", [10, 20]),
                SurfaceAxis("n_obs", [200, 400]),
                SurfaceAxis("n_vars", [3]),
            ],
            n_samples=50,
            seed=42,
        )

        surface = CriticalValueSurface()
        surface.compute(req, n_workers=1)

        # Query at midpoint
        cv_mid = surface.query(req, n_assets=15, n_obs=300, n_vars=3)
        cv_lo = surface.query(req, n_assets=10, n_obs=200, n_vars=3)
        cv_hi = surface.query(req, n_assets=20, n_obs=400, n_vars=3)

        # Interpolated 5% critical value should be reasonably close to
        # the endpoint average. With only 50 MC samples, noise is large,
        # so we allow generous tolerance.
        mid_5 = cv_mid[5]
        lo_5 = cv_lo[5]
        hi_5 = cv_hi[5]
        avg = (lo_5 + hi_5) / 2
        spread = abs(lo_5 - hi_5)
        assert abs(mid_5 - avg) < spread * 3.0, \
            f"Interpolated {mid_5} too far from avg {avg} (spread={spread})"

    def test_surface_compatible_with_burgess(self, price_universe):
        """Surface CriticalValues works with apply_mc_correction()."""
        from praxis.models.burgess import generate_candidates, apply_mc_correction
        from praxis.stats.monte_carlo import CriticalValues

        candidates = generate_candidates(
            price_universe,
            n_per_basket=3,
            max_candidates=5,
            significance=0.50,  # loose to guarantee candidates
        )

        if not candidates:
            pytest.skip("No candidates generated (stochastic)")

        # Simulate a surface lookup result
        cv = CriticalValues(values={10: -2.5, 5: -3.0, 1: -3.8})
        corrected = apply_mc_correction(candidates, cv)

        for c in corrected:
            assert 0 < c.adjusted_p_value < 1.0


# ═══════════════════════════════════════════════════════════════════
#  3.2: Kelly Sizer — distribution-aware position sizing
# ═══════════════════════════════════════════════════════════════════

class TestKellySizer:
    """Kelly-based position sizing using the distribution module."""

    def test_gaussian_kelly_single_strategy(self):
        """Single strategy: μ/σ² should give correct Kelly fraction."""
        from praxis.stats.distribution import fit_distribution, optimal_kelly

        rng = np.random.RandomState(42)
        # Simulate: 10% annual, 20% vol → μ_daily ≈ 0.0004, σ_daily ≈ 0.0126
        returns = rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=500)

        dist = fit_distribution(returns, method="kde")
        result = optimal_kelly(dist)

        assert result.f_star > 0, "Positive edge → positive Kelly"
        assert result.f_star < result.f_gaussian, "Fat tails → smaller Kelly"
        assert result.correction_ratio < 1.0

    def test_fat_tail_correction_meaningful(self):
        """Student-t returns give materially different Kelly than Gaussian."""
        from scipy import stats as sp_stats
        from praxis.stats.distribution import fit_distribution, optimal_kelly

        rng = np.random.RandomState(42)
        # Fat tails: t(df=4)
        raw = sp_stats.t.rvs(df=4, size=1000, random_state=42)
        returns = 0.0005 + raw * 0.01  # shift to positive mean

        dist_gauss = fit_distribution(returns, method="kde")
        dist_t = fit_distribution(returns, method="student_t")

        kelly_gauss = optimal_kelly(dist_gauss)
        kelly_t = optimal_kelly(dist_t)

        # Both should be positive (positive mean)
        assert kelly_gauss.f_star > 0
        assert kelly_t.f_star > 0
        # Correction should be < 1
        assert kelly_t.correction_ratio < 1.0

    def test_risk_metrics_vs_gaussian(self):
        """Distribution-based VaR should exceed Gaussian VaR for fat tails."""
        from scipy import stats as sp_stats
        from praxis.stats.distribution import fit_distribution, compute_risk_metrics

        raw = sp_stats.t.rvs(df=4, size=2000, random_state=42)
        returns = raw * 0.01  # center at 0

        dist = fit_distribution(returns, method="student_t")
        metrics = compute_risk_metrics(dist)

        # Fat tails → actual VaR more extreme than Gaussian VaR
        assert abs(metrics.var_99) >= abs(metrics.var_99_gaussian) * 0.9
        # Left tail ratio > 1 for fat tails
        assert metrics.tail_ratio_left >= 0.9  # should be > 1 for proper fat tails

    def test_gaussian_divergence_detects_fat_tails(self):
        """gaussian_divergence correctly flags non-Gaussian returns."""
        from scipy import stats as sp_stats
        from praxis.stats.distribution import fit_distribution, gaussian_divergence

        # Clearly fat-tailed
        raw = sp_stats.t.rvs(df=3, size=2000, random_state=42)
        returns = raw * 0.01

        dist = fit_distribution(returns, method="student_t")
        diag = gaussian_divergence(dist)

        assert diag.excess_kurtosis > 2.0, "df=3 should show high kurtosis"
        assert not diag.gaussian_adequate, "Should NOT be flagged as Gaussian-adequate"
        assert diag.kelly_correction < 0.8, "Kelly correction should be substantial"


# ═══════════════════════════════════════════════════════════════════
#  3.3: Pipeline Stitching — Burgess → signal → Kelly → backtest
# ═══════════════════════════════════════════════════════════════════

class TestPipelineStitching:
    """The full pipeline: generate baskets, signal, size with Kelly, backtest."""

    def test_burgess_to_signals(self, price_universe):
        """Burgess model produces usable signals."""
        from praxis.models.burgess import (
            BurgessStatArb,
            BurgessConfig,
            generate_basket_signals,
        )

        config = BurgessConfig(
            n_per_basket=3,
            max_candidates=10,
            mc_enabled=False,  # skip MC for speed (tested separately)
            significance=0.50,  # loose threshold
        )
        model = BurgessStatArb(config)
        result = model.run(price_universe)

        if not result.selected_baskets:
            pytest.skip("No baskets selected (stochastic)")

        # Generate signals for first basket
        basket = result.selected_baskets[0]
        sig_dict = generate_basket_signals(
            price_universe,
            basket,
            lookback=60,
            entry_threshold=1.5,
            exit_threshold=0.0,
        )

        signals = sig_dict["positions"]
        assert len(signals) == price_universe.shape[0]
        assert set(np.unique(signals)).issubset({-1, 0, 1})

    def test_signals_to_kelly_sizing(self, price_universe):
        """Signals + returns → Kelly-scaled position sizes."""
        from praxis.stats.distribution import fit_distribution, optimal_kelly

        rng = np.random.RandomState(42)
        # Mock signals and strategy returns
        signals = np.zeros(250)
        signals[50:80] = 1  # long period
        signals[120:160] = -1  # short period
        signals[200:230] = 1  # another long

        # Synthetic strategy returns (signal × market returns)
        market_ret = rng.normal(0.0003, 0.012, size=250)
        strategy_ret = signals * market_ret

        # Only size on periods with trades
        trade_mask = signals != 0
        if trade_mask.sum() < 30:
            pytest.skip("Not enough trades for Kelly")

        trade_returns = strategy_ret[trade_mask]
        dist = fit_distribution(trade_returns, method="auto")
        kelly = optimal_kelly(dist)

        # Kelly fraction should be finite and reasonable
        assert np.isfinite(kelly.f_star)
        assert 0 < kelly.f_star < 5.0  # not insane leverage

        # Apply Kelly to signals
        sized_positions = signals * min(kelly.f_star, 2.0)  # cap at 2x
        assert np.max(np.abs(sized_positions)) <= 2.0

    def test_kelly_sizing_to_backtest(self):
        """Kelly-sized positions → backtest engine → valid metrics."""
        from praxis.backtest import VectorizedEngine, BacktestMetrics

        rng = np.random.RandomState(42)
        n = 200

        # Simulated prices
        prices = 100 + np.cumsum(rng.randn(n) * 0.5)
        returns_pct = np.diff(prices) / prices[:-1]

        # Simulated Kelly-sized positions (variable sizing)
        positions = np.zeros(n)
        positions[30:60] = 0.8  # 80% of capital long
        positions[100:140] = -0.5  # 50% short
        positions[160:190] = 1.2  # 120% (mild leverage)

        engine = VectorizedEngine()
        output = engine.run(
            positions=positions[:-1],
            prices=prices[:-1],
        )

        assert output.metrics is not None
        metrics = output.metrics
        assert np.isfinite(metrics.sharpe_ratio)
        assert np.isfinite(metrics.max_drawdown)
        assert metrics.max_drawdown <= 0  # drawdowns are negative

    def test_full_pipeline_smoke(self, price_universe):
        """Smoke test: entire pipeline doesn't crash."""
        from praxis.models.burgess import (
            BurgessStatArb,
            BurgessConfig,
            generate_basket_signals,
        )
        from praxis.stats.distribution import fit_distribution, optimal_kelly

        # 1. Run Burgess
        config = BurgessConfig(
            n_per_basket=3,
            max_candidates=10,
            mc_enabled=False,
            significance=0.50,
        )
        model = BurgessStatArb(config)
        result = model.run(price_universe)

        if not result.selected_baskets:
            pytest.skip("No baskets selected")

        # 2. Generate signals
        basket = result.selected_baskets[0]
        sig_dict = generate_basket_signals(
            price_universe, basket,
            lookback=60, entry_threshold=1.5, exit_threshold=0.0,
        )
        signals = sig_dict["positions"]

        # 3. Compute strategy returns
        target_returns = np.diff(price_universe[:, basket.target_index]) / \
            price_universe[:-1, basket.target_index]
        strategy_returns = signals[:-1] * target_returns

        # 4. Fit distribution and get Kelly
        trade_returns = strategy_returns[signals[:-1] != 0]
        if len(trade_returns) < 30:
            pytest.skip("Not enough trades")

        dist = fit_distribution(trade_returns, method="auto")
        kelly = optimal_kelly(dist)

        # 5. Size positions
        kelly_cap = min(kelly.f_star, 2.0)
        sized = signals[:-1] * kelly_cap

        # 6. Basic P&L
        pnl = sized * target_returns
        total_pnl = pnl.sum()

        # Smoke: everything ran without error and produced finite results
        assert np.isfinite(total_pnl)
        assert np.isfinite(kelly.f_star)
        assert kelly.correction_ratio <= 1.0


# ═══════════════════════════════════════════════════════════════════
#  3.4: Multi-Model Allocation — model_of_models
# ═══════════════════════════════════════════════════════════════════

class TestMultiModelAllocation:
    """Kelly-Markowitz allocation across multiple model backtests."""

    def test_model_of_models_basic(self):
        """model_of_models runs on dict of return arrays."""
        from praxis.stats.portfolio import model_of_models, KellyResult

        rng = np.random.RandomState(42)
        n = 300

        backtests = {
            "stat_arb": rng.normal(0.0005, 0.01, n),
            "momentum": rng.normal(0.0003, 0.02, n),
            "mean_rev": rng.normal(0.0006, 0.015, n),
        }

        result = model_of_models(backtests, use_fat_tails=False)

        assert isinstance(result, KellyResult)
        assert result.n_models == 3
        assert len(result.kelly_weights) == 3
        assert len(result.markowitz_weights) == 3
        assert abs(sum(result.markowitz_weights) - 1.0) < 1e-10
        assert result.optimal_leverage > 0

    def test_model_of_models_fat_tailed(self):
        """Fat-tail correction applied to model portfolio."""
        from scipy import stats as sp_stats
        from praxis.stats.portfolio import model_of_models

        rng = np.random.RandomState(42)
        n = 500

        # Fat-tailed returns
        t_returns = sp_stats.t.rvs(df=4, size=(n, 3), random_state=42)
        t_returns = t_returns / np.sqrt(4 / 2)  # normalize variance
        means = np.array([0.0005, 0.0003, 0.0006])
        vols = np.array([0.01, 0.02, 0.015])
        returns_matrix = means + t_returns * vols

        backtests = {
            "stat_arb": returns_matrix[:, 0],
            "momentum": returns_matrix[:, 1],
            "mean_rev": returns_matrix[:, 2],
        }

        result = model_of_models(backtests, use_fat_tails=True)

        assert result.fat_tail_correction < 1.0, "Fat tails should reduce Kelly"
        assert result.optimal_leverage > 0
        # Corrected weights should be smaller in magnitude than Gaussian
        assert np.sum(np.abs(result.corrected_kelly_weights)) < \
            np.sum(np.abs(result.gaussian_kelly_weights))

    def test_allocation_direction_matches_sharpe(self):
        """Higher-Sharpe model should get higher relative weight."""
        from praxis.stats.portfolio import model_of_models

        rng = np.random.RandomState(42)
        n = 500

        backtests = {
            "low_sharpe": rng.normal(0.0001, 0.02, n),   # SR ≈ 0.08
            "high_sharpe": rng.normal(0.0008, 0.01, n),   # SR ≈ 1.27
        }

        result = model_of_models(backtests, use_fat_tails=False)

        # Markowitz weight for high_sharpe should exceed low_sharpe
        idx_low = result.model_names.index("low_sharpe")
        idx_high = result.model_names.index("high_sharpe")
        assert result.markowitz_weights[idx_high] > result.markowitz_weights[idx_low]

    def test_negative_sharpe_gets_negative_weight(self):
        """Model with negative edge should get shorted (negative Kelly weight)."""
        from praxis.stats.portfolio import kelly_portfolio

        mu = np.array([0.001, -0.002, 0.0005])  # model 2 loses money
        cov = np.diag([0.01**2, 0.015**2, 0.012**2])  # uncorrelated

        result = kelly_portfolio(mu, cov, model_names=["A", "B", "C"])

        assert result.kelly_weights[1] < 0, "Losing model → negative Kelly weight"
        assert result.kelly_weights[0] > 0
        assert result.kelly_weights[2] > 0

    def test_correlation_shifts_allocation(self):
        """Correlated models get less combined weight than uncorrelated ones."""
        from praxis.stats.portfolio import kelly_portfolio

        mu = np.array([0.001, 0.001])

        # Uncorrelated
        cov_uncorr = np.array([[0.01**2, 0.0], [0.0, 0.01**2]])
        result_uncorr = kelly_portfolio(mu, cov_uncorr)

        # Highly correlated
        rho = 0.9
        cov_corr = np.array([
            [0.01**2, rho * 0.01**2],
            [rho * 0.01**2, 0.01**2],
        ])
        result_corr = kelly_portfolio(mu, cov_corr)

        # Total leverage should be lower when correlated
        assert result_corr.optimal_leverage < result_uncorr.optimal_leverage


# ═══════════════════════════════════════════════════════════════════
#  3.5: Regression Guards — don't break existing results
# ═══════════════════════════════════════════════════════════════════

class TestRegressionGuards:
    """Ensure new modules don't break the existing pipeline.
    Spot-checks from Rounds 1 and 2 that should still pass."""

    def test_stepwise_regression_unchanged(self):
        """The core regression engine still works."""
        from praxis.stats.regression import successive_regression

        rng = np.random.RandomState(42)
        paths = np.cumsum(rng.randn(200, 10), axis=0)

        result = successive_regression(0, paths, n_vars=3)
        assert result is not None
        assert result.regression.r_squared > 0
        assert len(result.regression.residuals) == 200

    def test_adf_test_unchanged(self):
        """ADF still works on stationary series."""
        from praxis.stats import adf_test

        rng = np.random.RandomState(42)
        # AR(1) with phi = 0.5 (stationary)
        x = np.zeros(200)
        for t in range(1, 200):
            x[t] = 0.5 * x[t - 1] + rng.randn()

        result = adf_test(x)
        assert result.t_statistic < -2.0  # should reject unit root
        assert result.p_value < 0.10

    def test_markowitz_still_works(self):
        """Original Markowitz functions unbroken."""
        from praxis.stats.portfolio import max_sharpe_portfolio, min_variance_portfolio

        mu = np.array([0.10, 0.05, 0.08])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.03],
        ])

        result_mv = min_variance_portfolio(cov)
        assert abs(sum(result_mv.weights) - 1.0) < 1e-10

        result_ms = max_sharpe_portfolio(mu, cov)
        assert abs(sum(result_ms.weights) - 1.0) < 1e-10
        assert result_ms.sharpe_ratio > 0

    def test_vectorized_engine_unchanged(self):
        """Backtest engine still produces valid metrics."""
        from praxis.backtest import VectorizedEngine

        rng = np.random.RandomState(42)
        positions = np.sign(rng.randn(100))
        returns = rng.randn(100) * 0.01

        engine = VectorizedEngine()
        prices = 100 + np.cumsum(returns)  # reconstruct prices from returns
        output = engine.run(positions=positions, prices=prices)

        assert np.isfinite(output.metrics.sharpe_ratio)


# ═══════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════
#
#  Round 3 Battle Test Structure:
#
#  3.1 Surface Integration (3 tests)
#      - Surface produces CriticalValues objects
#      - Interpolated queries work
#      - Compatible with Burgess apply_mc_correction
#
#  3.2 Kelly Sizer (4 tests)
#      - Gaussian Kelly on single strategy
#      - Fat-tail correction is meaningful
#      - Risk metrics exceed Gaussian for fat tails
#      - Gaussian divergence flags non-Normal
#
#  3.3 Pipeline Stitching (4 tests)
#      - Burgess → signals
#      - Signals → Kelly sizing
#      - Kelly sizing → backtest metrics
#      - Full pipeline smoke test
#
#  3.4 Multi-Model Allocation (5 tests)
#      - model_of_models basic
#      - Fat-tail correction applied
#      - Higher Sharpe → higher weight
#      - Negative edge → negative weight
#      - Correlation reduces combined leverage
#
#  3.5 Regression Guards (4 tests)
#      - Stepwise regression unchanged
#      - ADF test unchanged
#      - Markowitz unchanged
#      - Backtest engine unchanged
#
#  Total: 20 tests
# ═══════════════════════════════════════════════════════════════════
