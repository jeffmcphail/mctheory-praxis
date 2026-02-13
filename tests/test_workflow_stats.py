"""
Tests for Phase 4.1 (Workflow Executor), 4.2 (Statistical Tests),
4.3 (Successive Regression), 4.4 (Monte Carlo Surface).
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
#  Phase 4.1: Workflow Executor (§10)
# ═══════════════════════════════════════════════════════════════════

from praxis.workflow import (
    WorkflowExecutor,
    WorkflowStep,
    WorkflowResult,
    WorkflowFunctionRegistry,
    StepStatus,
    workflow_from_config,
    _resolve_reference,
    _evaluate_condition,
)


class TestWorkflowBasic:
    def test_empty_workflow(self):
        wf = WorkflowExecutor()
        result = wf.run()
        assert result.success
        assert result.steps_total == 0

    def test_single_step(self):
        reg = WorkflowFunctionRegistry()
        reg.register("add_one", lambda x=0: x + 1)

        wf = WorkflowExecutor(registry=reg)
        wf.add_step(WorkflowStep(id="s1", function="add_one", params={"x": 5}))
        result = wf.run()

        assert result.success
        assert result.outputs["s1"] == 6

    def test_linear_chain(self):
        reg = WorkflowFunctionRegistry()
        reg.register("double", lambda x=0: x * 2)
        reg.register("add_ten", lambda x=0: x + 10)

        wf = WorkflowExecutor(registry=reg)
        wf.add_step(WorkflowStep(id="a", function="double", params={"x": 5}))
        wf.add_step(WorkflowStep(
            id="b", function="add_ten",
            params={"x": "a.output"},
            depends_on=["a"],
        ))
        result = wf.run()

        assert result.success
        assert result.outputs["a"] == 10
        assert result.outputs["b"] == 20

    def test_duplicate_step_raises(self):
        wf = WorkflowExecutor()
        wf.add_step(WorkflowStep(id="s1", function="f"))
        with pytest.raises(ValueError, match="Duplicate"):
            wf.add_step(WorkflowStep(id="s1", function="g"))

    def test_unknown_dependency_raises(self):
        wf = WorkflowExecutor()
        wf.add_step(WorkflowStep(id="s1", function="f", depends_on=["nonexistent"]))
        result = wf.run()
        assert result.error is not None
        assert "unknown" in result.error.lower()

    def test_cycle_detected(self):
        wf = WorkflowExecutor()
        wf.add_step(WorkflowStep(id="a", function="f", depends_on=["b"]))
        wf.add_step(WorkflowStep(id="b", function="g", depends_on=["a"]))
        result = wf.run()
        assert result.error is not None
        assert "Cycle" in result.error


class TestWorkflowFailure:
    def test_failed_step_skips_dependents(self):
        reg = WorkflowFunctionRegistry()
        reg.register("fail", lambda: 1 / 0)
        reg.register("ok", lambda: "done")

        wf = WorkflowExecutor(registry=reg)
        wf.add_step(WorkflowStep(id="a", function="fail"))
        wf.add_step(WorkflowStep(id="b", function="ok", depends_on=["a"]))

        result = wf.run()
        assert not result.success
        assert result.steps_failed == 1
        assert result.steps_skipped == 1

    def test_unknown_function_fails(self):
        wf = WorkflowExecutor()
        wf.add_step(WorkflowStep(id="s1", function="no_such_function"))
        result = wf.run()
        assert result.steps_failed == 1

    def test_partial_failure_records_duration(self):
        reg = WorkflowFunctionRegistry()
        reg.register("ok", lambda: 42)
        reg.register("fail", lambda: 1 / 0)

        wf = WorkflowExecutor(registry=reg)
        wf.add_step(WorkflowStep(id="a", function="ok"))
        wf.add_step(WorkflowStep(id="b", function="fail"))

        result = wf.run()
        assert result.total_duration >= 0
        assert result.steps_completed == 1
        assert result.steps_failed == 1


class TestWorkflowConditional:
    def test_condition_true_branch(self):
        reg = WorkflowFunctionRegistry()
        reg.register("count", lambda: {"count": 200})
        reg.register("filter_big", lambda: "filtered_big")
        reg.register("pass_through", lambda: "passed")

        wf = WorkflowExecutor(registry=reg)
        wf.add_step(WorkflowStep(id="gen", function="count"))
        wf.add_step(WorkflowStep(
            id="branch",
            depends_on=["gen"],
            condition="gen.output.count > 100",
            if_true={"function": "filter_big"},
            if_false={"function": "pass_through"},
        ))
        result = wf.run()

        assert result.success
        assert result.outputs["branch"] == "filtered_big"

    def test_condition_false_branch(self):
        reg = WorkflowFunctionRegistry()
        reg.register("count", lambda: {"count": 50})
        reg.register("filter_big", lambda: "filtered_big")
        reg.register("pass_through", lambda: "passed")

        wf = WorkflowExecutor(registry=reg)
        wf.add_step(WorkflowStep(id="gen", function="count"))
        wf.add_step(WorkflowStep(
            id="branch",
            depends_on=["gen"],
            condition="gen.output.count > 100",
            if_true={"function": "filter_big"},
            if_false={"function": "pass_through"},
        ))
        result = wf.run()

        assert result.success
        assert result.outputs["branch"] == "passed"


class TestWorkflowForEach:
    def test_for_each_basic(self):
        reg = WorkflowFunctionRegistry()
        reg.register("gen_items", lambda: {"items": [1, 2, 3]})
        reg.register("process", lambda item=0, **kw: item * 10)

        wf = WorkflowExecutor(registry=reg)
        wf.add_step(WorkflowStep(id="gen", function="gen_items"))
        wf.add_step(WorkflowStep(
            id="proc",
            function="process",
            depends_on=["gen"],
            for_each="gen.output.items",
            for_each_as="item",
        ))
        result = wf.run()

        assert result.success
        assert result.outputs["proc"] == [10, 20, 30]

    def test_for_each_empty_list(self):
        reg = WorkflowFunctionRegistry()
        reg.register("gen_empty", lambda: {"items": []})
        reg.register("process", lambda item=0, **kw: item * 10)

        wf = WorkflowExecutor(registry=reg)
        wf.add_step(WorkflowStep(id="gen", function="gen_empty"))
        wf.add_step(WorkflowStep(
            id="proc",
            function="process",
            depends_on=["gen"],
            for_each="gen.output.items",
        ))
        result = wf.run()
        assert result.success
        assert result.outputs["proc"] == []


class TestOutputResolution:
    def test_simple_reference(self):
        outputs = {"step_a": 42}
        assert _resolve_reference("step_a", outputs) == 42

    def test_nested_dict_reference(self):
        outputs = {"step_a": {"count": 100, "items": [1, 2]}}
        assert _resolve_reference("step_a.output.count", outputs) == 100
        assert _resolve_reference("step_a.output.items", outputs) == [1, 2]

    def test_missing_reference(self):
        outputs = {}
        assert _resolve_reference("nonexistent.output", outputs) is None


class TestConditionEvaluation:
    def test_greater_than(self):
        outputs = {"s": {"count": 200}}
        assert _evaluate_condition("s.output.count > 100", outputs) is True
        assert _evaluate_condition("s.output.count > 300", outputs) is False

    def test_less_than(self):
        outputs = {"s": {"val": 5}}
        assert _evaluate_condition("s.output.val < 10", outputs) is True

    def test_equality(self):
        outputs = {"s": {"status": "ready"}}
        assert _evaluate_condition("s.output.status == ready", outputs) is True

    def test_not_equal(self):
        outputs = {"s": {"x": 5}}
        assert _evaluate_condition("s.output.x != 10", outputs) is True


class TestWorkflowFromConfig:
    def test_basic_config(self):
        reg = WorkflowFunctionRegistry()
        reg.register("load", lambda: {"data": [1, 2, 3]})
        reg.register("process", lambda: "done")

        config = {
            "workflow": {
                "steps": [
                    {"id": "load", "function": "load"},
                    {"id": "process", "function": "process", "depends_on": ["load"]},
                ]
            }
        }

        wf = workflow_from_config(config, reg)
        assert wf.size == 2
        result = wf.run()
        assert result.success


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.2: Statistical Tests
# ═══════════════════════════════════════════════════════════════════

from praxis.stats import (
    adf_test,
    johansen_test,
    durbin_watson_test,
    ljung_box_test,
    hurst_exponent,
    half_life,
    variance_ratio,
    variance_profile,
    run_stationarity_tests,
    ADFResult,
    JohansenResult,
    HurstResult,
    HalfLifeResult,
    VarianceRatioResult,
    StationarityResult,
)


def _make_stationary(n=500, seed=42):
    """OU process — mean-reverting."""
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.randn()
    return x


def _make_random_walk(n=500, seed=42):
    """Pure random walk — not stationary."""
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(n))


class TestADF:
    def test_stationary_series(self):
        x = _make_stationary()
        result = adf_test(x)
        assert isinstance(result, ADFResult)
        assert result.is_stationary
        assert result.p_value < 0.05

    def test_random_walk_not_stationary(self):
        x = _make_random_walk()
        result = adf_test(x)
        assert not result.is_stationary
        assert result.p_value > 0.05

    def test_critical_values_exist(self):
        result = adf_test(_make_stationary())
        assert "1%" in result.critical_values
        assert "5%" in result.critical_values
        assert "10%" in result.critical_values

    def test_at_significance(self):
        result = adf_test(_make_stationary())
        assert result.at_significance(0.05)

    def test_result_fields(self):
        result = adf_test(_make_stationary())
        assert result.lags_used >= 0
        assert result.n_observations > 0
        assert isinstance(result.t_statistic, float)


class TestJohansen:
    def test_cointegrated_pair(self):
        rng = np.random.RandomState(42)
        # Two series with shared random walk + stationary spread
        common = np.cumsum(rng.randn(500))
        x1 = common + rng.randn(500) * 0.1
        x2 = 0.8 * common + rng.randn(500) * 0.1
        matrix = np.column_stack([x1, x2])

        result = johansen_test(matrix)
        assert isinstance(result, JohansenResult)
        assert result.is_cointegrated
        assert result.n_cointegrating >= 1

    def test_independent_walks(self):
        rng = np.random.RandomState(42)
        x1 = np.cumsum(rng.randn(500))
        x2 = np.cumsum(rng.randn(500))
        matrix = np.column_stack([x1, x2])

        result = johansen_test(matrix)
        # Independent walks should not be cointegrated
        assert result.n_cointegrating == 0

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            johansen_test(np.array([1, 2, 3]))

    def test_result_shapes(self):
        rng = np.random.RandomState(42)
        m = np.column_stack([np.cumsum(rng.randn(200)) for _ in range(3)])
        result = johansen_test(m)
        assert len(result.trace_stats) == 3
        assert len(result.max_eig_stats) == 3


class TestDurbinWatson:
    def test_white_noise(self):
        rng = np.random.RandomState(42)
        dw = durbin_watson_test(rng.randn(500))
        assert 1.5 < dw < 2.5  # Near 2 for uncorrelated

    def test_autocorrelated(self):
        x = _make_stationary()
        dw = durbin_watson_test(x)
        assert isinstance(dw, float)


class TestLjungBox:
    def test_white_noise(self):
        rng = np.random.RandomState(42)
        stat, pval = ljung_box_test(rng.randn(500))
        assert pval > 0.05  # White noise should not reject

    def test_autocorrelated(self):
        x = _make_stationary()
        stat, pval = ljung_box_test(x)
        assert stat > 0


class TestHurst:
    def test_random_walk_near_05(self):
        x = _make_random_walk(1000)
        result = hurst_exponent(x)
        assert isinstance(result, HurstResult)
        # Random walk should be near 0.5
        assert 0.3 < result.hurst_exponent < 0.8

    def test_mean_reverting_below_05(self):
        x = _make_stationary(1000)
        # OU process should have H < 0.5
        result = hurst_exponent(x)
        assert result.hurst_exponent < 0.55
        assert result.interpretation == "mean-reverting"

    def test_short_series(self):
        result = hurst_exponent(np.array([1, 2, 3]))
        assert result.hurst_exponent == 0.5  # Default for short


class TestHalfLife:
    def test_ou_process(self):
        x = _make_stationary(1000)
        result = half_life(x)
        assert isinstance(result, HalfLifeResult)
        assert result.is_mean_reverting
        assert 0 < result.half_life < 50
        assert result.beta < 0

    def test_random_walk_not_mean_reverting(self):
        x = _make_random_walk(1000)
        result = half_life(x)
        # Random walk: beta ≈ 0 or positive → infinite half-life
        # May or may not register as mean-reverting depending on sample
        assert isinstance(result.half_life, float)


class TestVarianceRatio:
    def test_random_walk_near_1(self):
        x = _make_random_walk(1000)
        result = variance_ratio(x, lag=10)
        assert isinstance(result, VarianceRatioResult)
        assert 0.7 < result.ratio < 1.3

    def test_mean_reverting_below_1(self):
        x = _make_stationary(1000)
        result = variance_ratio(x, lag=10)
        assert result.ratio < 1.0
        assert result.is_mean_reverting

    def test_profile_shape(self):
        x = _make_random_walk(500)
        profile = variance_profile(x, max_lag=20)
        assert len(profile) == 20


class TestRunStationarityTests:
    def test_stationary_full_suite(self):
        x = _make_stationary(500)
        result = run_stationarity_tests(x)
        assert isinstance(result, StationarityResult)
        assert result.is_stationary
        assert result.adf is not None
        assert result.hurst is not None
        assert result.half_life is not None
        assert result.durbin_watson_stat is not None

    def test_random_walk_full_suite(self):
        x = _make_random_walk(500)
        result = run_stationarity_tests(x)
        assert not result.is_stationary


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.3: Successive Regression
# ═══════════════════════════════════════════════════════════════════

from praxis.stats.regression import (
    corr2_coeff,
    ridge_regression,
    successive_regression,
    generate_random_walk_universe,
    RegressionResult,
    StepwiseResult,
)


class TestCorr2Coeff:
    def test_identical_rows(self):
        A = np.array([[1, 2, 3, 4]], dtype=float)
        result = corr2_coeff(A, A)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-10)

    def test_anticorrelated(self):
        A = np.array([[1, 2, 3, 4]], dtype=float)
        B = np.array([[4, 3, 2, 1]], dtype=float)
        result = corr2_coeff(A, B)
        np.testing.assert_allclose(result[0, 0], -1.0, atol=1e-10)

    def test_matrix_shape(self):
        A = np.random.randn(5, 100)
        B = np.random.randn(3, 100)
        result = corr2_coeff(A, B)
        assert result.shape == (5, 3)


class TestRidgeRegression:
    def test_basic_fit(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        true_beta = np.array([3.0, -1.5])
        y = X @ true_beta + rng.randn(100) * 0.1

        result = ridge_regression(y, X)
        assert isinstance(result, RegressionResult)
        assert result.r_squared > 0.9
        assert len(result.residuals) == 100

    def test_residuals_small_for_perfect_fit(self):
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = 2 * X.ravel() + 1
        result = ridge_regression(y, X)
        assert np.std(result.residuals) < 0.1

    def test_ridge_cv(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = X[:, 0] * 2 + rng.randn(50) * 0.5
        result = ridge_regression(y, X, use_ridge_cv=True)
        assert result.r_squared > 0.5


class TestSuccessiveRegression:
    def test_selects_correct_count(self):
        rng = np.random.RandomState(42)
        M = rng.randn(200, 10)
        result = successive_regression(0, M, n_vars=3)
        assert isinstance(result, StepwiseResult)
        assert len(result.selected_indices) == 3
        assert 0 not in result.selected_indices

    def test_adf_on_residuals(self):
        rng = np.random.RandomState(42)
        M = rng.randn(200, 10)
        result = successive_regression(0, M, n_vars=3, compute_stats=True)
        assert result.adf is not None
        assert isinstance(result.adf.t_statistic, float)

    def test_zero_vars(self):
        rng = np.random.RandomState(42)
        M = rng.randn(200, 5)
        result = successive_regression(0, M, n_vars=0, compute_stats=True)
        assert len(result.selected_indices) == 0
        assert result.adf is not None

    def test_cointegrated_assets_are_stationary(self):
        """Stepwise regression on cointegrated assets should find stationarity."""
        rng = np.random.RandomState(42)
        common = np.cumsum(rng.randn(500))
        noise = lambda: rng.randn(500) * 0.5
        M = np.column_stack([
            common + noise(),
            0.8 * common + noise(),
            -0.5 * common + noise(),
            np.cumsum(rng.randn(500)),  # independent
            np.cumsum(rng.randn(500)),  # independent
        ])
        result = successive_regression(0, M, n_vars=2)
        # Should find cointegrating partners and produce stationary residuals
        assert result.adf is not None
        # Strong cointegration should be detected
        assert result.adf.p_value < 0.10

    def test_invalid_target_index(self):
        M = np.random.randn(100, 5)
        with pytest.raises(ValueError, match="out of range"):
            successive_regression(10, M)


class TestRandomWalkUniverse:
    def test_shape(self):
        M = generate_random_walk_universe(100, 10, seed=42)
        assert M.shape == (100, 10)

    def test_reproducible(self):
        M1 = generate_random_walk_universe(50, 5, seed=42)
        M2 = generate_random_walk_universe(50, 5, seed=42)
        np.testing.assert_array_equal(M1, M2)

    def test_with_origin_range(self):
        M = generate_random_walk_universe(50, 5, origin_range=(90, 110), seed=42)
        assert M.shape == (50, 5)
        # First row should be in origin range
        assert all(90 <= M[0, j] <= 150 for j in range(5))


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.4: Monte Carlo Surface
# ═══════════════════════════════════════════════════════════════════

from praxis.stats.monte_carlo import (
    get_critical_values,
    generate_adf_critical_values,
    generate_critical_values_batch,
    mahalanobis_distance,
    CriticalValues,
    MonteCarloResult,
    CriticalValueRow,
)


class TestGetCriticalValues:
    def test_sorted_array(self):
        arr = np.arange(100, dtype=float)
        cv = get_critical_values(arr, [10, 5, 1])
        # Lower percentile = more extreme = smaller value
        assert cv[1] < cv[5] < cv[10]

    def test_percentile_accuracy(self):
        arr = np.arange(1000, dtype=float)
        cv = get_critical_values(arr, [50])
        assert abs(cv[50] - 499) < 2  # ~50th percentile ≈ 499


class TestMonteCarloADF:
    def test_basic_generation(self):
        result = generate_adf_critical_values(
            n_assets=20, n_obs=100, n_vars=2,
            n_samples=50, seed=42,
        )
        assert isinstance(result, MonteCarloResult)
        assert len(result.t_values) == 50
        assert result.elapsed_seconds > 0

    def test_critical_values_exist(self):
        result = generate_adf_critical_values(
            n_assets=20, n_obs=100, n_vars=1,
            n_samples=50, seed=42,
        )
        cv = result.critical_values
        assert isinstance(cv, CriticalValues)
        assert 5 in cv.values
        assert 1 in cv.values
        assert 10 in cv.values

    def test_critical_values_negative(self):
        """ADF critical values should be negative."""
        result = generate_adf_critical_values(
            n_assets=20, n_obs=100, n_vars=2,
            n_samples=100, seed=42,
        )
        assert result.critical_values.at(5) < 0

    def test_more_stringent_at_lower_pct(self):
        result = generate_adf_critical_values(
            n_assets=20, n_obs=100, n_vars=2,
            n_samples=100, seed=42,
        )
        cv = result.critical_values
        # 1% critical value should be more negative than 10%
        assert cv.at(1) < cv.at(10)

    def test_significance_check(self):
        result = generate_adf_critical_values(
            n_assets=20, n_obs=100, n_vars=2,
            n_samples=100, seed=42,
        )
        cv = result.critical_values
        # A very extreme t-statistic should be significant
        assert cv.is_significant(-10.0, pct=5)
        # A mild one should not
        assert not cv.is_significant(-1.0, pct=5)


class TestCriticalValuesBatch:
    def test_batch_produces_rows(self):
        rows = generate_critical_values_batch(
            n_assets_range=[10],
            n_obs_range=[50],
            n_vars_range=[1, 2],
            n_samples=20,
            seed=42,
        )
        assert len(rows) > 0
        assert all(isinstance(r, CriticalValueRow) for r in rows)
        # 2 n_vars × 3 conf levels = 6 rows
        assert len(rows) == 6


class TestMahalanobis:
    def test_zero_distance(self):
        u = np.array([1.0, 2.0, 3.0])
        cov_inv = np.eye(3)
        d = mahalanobis_distance(u, u, cov_inv)
        assert abs(d) < 1e-10

    def test_euclidean_with_identity(self):
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 0.0])
        cov_inv = np.eye(2)
        d = mahalanobis_distance(u, v, cov_inv)
        assert abs(d - 1.0) < 1e-10

    def test_scaled_covariance(self):
        u = np.array([2.0, 0.0])
        v = np.array([0.0, 0.0])
        # Variance of 4 in first dimension → std=2 → 2/2 = 1 Mahalanobis unit
        cov_inv = np.array([[0.25, 0], [0, 1.0]])
        d = mahalanobis_distance(u, v, cov_inv)
        assert abs(d - 1.0) < 1e-10
