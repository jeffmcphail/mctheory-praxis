"""
Tests for Phase 3.11 (User Code Templates), 3.12 (Cross-Source Data Quality),
3.13 (Declarative Scheduling), 3.14 (Tick DAG Engine).
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from praxis.logger.core import PraxisLogger
from praxis.registry import FunctionRegistry


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.11: User Code Templates (§16.2-16.6)
# ═══════════════════════════════════════════════════════════════════

from praxis.templates import (
    SignalTemplate,
    SizingTemplate,
    ExecutionAdapterTemplate,
    DataSourceTemplate,
    UserCodeLoader,
    ValidationResult,
    LoadResult,
    validate_interface,
    validate_sandbox,
    validate_no_conflicts,
    TEMPLATE_INTERFACES,
)


class _GoodSignal(SignalTemplate):
    """A valid user signal implementation."""
    def generate(self, prices, params):
        return prices


class _BadSignal(SignalTemplate):
    """Missing generate method — still abstract."""
    pass  # generate is abstract, but Python won't error until instantiation


class _GoodSizer(SizingTemplate):
    def size(self, signals, params):
        return signals


class _GoodExecution(ExecutionAdapterTemplate):
    def connect(self):
        return True

    def submit_order(self, order):
        return "order_123"


class _GoodDataSource(DataSourceTemplate):
    def fetch(self, tickers, start, end):
        return {}


class TestTemplateInterfaces:
    """§16.3: Template contracts."""

    def test_signal_interface_exists(self):
        assert "signals" in TEMPLATE_INTERFACES
        assert TEMPLATE_INTERFACES["signals"] is SignalTemplate

    def test_all_four_categories(self):
        assert set(TEMPLATE_INTERFACES.keys()) == {
            "signals", "sizing", "execution", "data_sources"
        }

    def test_good_signal_instantiates(self):
        s = _GoodSignal()
        assert s.generate([1, 2, 3], {}) == [1, 2, 3]

    def test_good_sizer_instantiates(self):
        s = _GoodSizer()
        assert s.size([1], {}) == [1]

    def test_good_execution_instantiates(self):
        e = _GoodExecution()
        assert e.connect() is True
        assert e.submit_order({}) == "order_123"

    def test_good_datasource_instantiates(self):
        d = _GoodDataSource()
        assert d.fetch([], "", "") == {}


class TestInterfaceValidation:
    """§16.5 step 1: Interface compliance."""

    def test_good_signal_passes(self):
        errors = validate_interface(_GoodSignal, "signals")
        assert errors == []

    def test_good_sizer_passes(self):
        errors = validate_interface(_GoodSizer, "sizing")
        assert errors == []

    def test_unknown_category_passes(self):
        """Unknown categories have no interface check."""
        errors = validate_interface(_GoodSignal, "unknown_category")
        assert errors == []


class TestSandboxValidation:
    """§16.6: Sandbox rules."""

    def test_execution_exempt_from_network(self):
        errors = validate_sandbox("some.module", "execution")
        assert errors == []

    def test_nonexistent_module_no_crash(self):
        errors = validate_sandbox("nonexistent.module", "signals")
        assert errors == []  # Can't read source, skip check


class TestConflictDetection:
    """§16.5 step 4: Conflict detection."""

    def test_no_conflict_for_new_name(self):
        registry = FunctionRegistry.instance()
        errors = validate_no_conflicts("totally_new_signal", "signals", registry)
        assert errors == []


class TestUserCodeLoader:
    """§16.4: Registry integration."""

    def test_no_registry_file(self):
        loader = UserCodeLoader("/nonexistent/path")
        result = loader.load_and_validate()
        assert isinstance(result, LoadResult)
        assert result.all_passed
        assert result.registered == 0

    def test_empty_registry_file(self):
        with tempfile.TemporaryDirectory() as td:
            reg_path = Path(td) / "registry.yaml"
            reg_path.write_text("{}")
            loader = UserCodeLoader(td)
            result = loader.load_and_validate()
            assert result.registered == 0

    def test_missing_module_fails(self):
        with tempfile.TemporaryDirectory() as td:
            reg_path = Path(td) / "registry.yaml"
            reg_path.write_text(
                "signals:\n"
                "  bad_signal:\n"
                "    module: nonexistent.module\n"
                "    class: BadClass\n"
            )
            loader = UserCodeLoader(td)
            result = loader.load_and_validate()
            assert result.failed == 1
            assert "Cannot import" in result.validations[0].errors[0]

    def test_missing_module_key_fails(self):
        with tempfile.TemporaryDirectory() as td:
            reg_path = Path(td) / "registry.yaml"
            reg_path.write_text(
                "signals:\n"
                "  bad_signal:\n"
                "    class: SomeClass\n"
            )
            loader = UserCodeLoader(td)
            result = loader.load_and_validate()
            assert result.failed == 1
            assert "Missing 'module'" in result.validations[0].errors[0]

    def test_load_result_structure(self):
        result = LoadResult()
        assert result.all_passed
        assert result.registered == 0
        assert result.failed == 0

    def test_validation_result_structure(self):
        vr = ValidationResult(name="test", category="signals", passed=True)
        assert vr.name == "test"
        assert vr.errors == []


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.12: Cross-Source Data Quality
# ═══════════════════════════════════════════════════════════════════

from praxis.quality.cross_source import (
    cross_source_check,
    detect_stale_source,
    CrossSourceResult,
    FieldComparison,
)


def _make_matching_prices(n=100, seed=42):
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    return {
        "close": close,
        "open": close + rng.randn(n) * 0.1,
        "high": close + np.abs(rng.randn(n) * 0.3),
        "low": close - np.abs(rng.randn(n) * 0.3),
        "volume": (rng.rand(n) * 1e6).astype(np.float64),
    }


class TestCrossSourceBasic:
    def test_identical_sources_pass(self):
        prices = _make_matching_prices()
        result = cross_source_check(prices, prices, "GLD", "yf", "poly")
        assert result.passed
        assert result.n_overlapping == 100

    def test_result_structure(self):
        prices = _make_matching_prices()
        result = cross_source_check(prices, prices, "GLD", "yf", "poly")
        assert isinstance(result, CrossSourceResult)
        assert result.ticker == "GLD"
        assert result.source_a == "yf"
        assert result.source_b == "poly"
        assert len(result.fields) == 5  # close, open, high, low, volume

    def test_all_fields_compared(self):
        prices = _make_matching_prices()
        result = cross_source_check(prices, prices)
        field_names = [f.field for f in result.fields]
        assert "close" in field_names
        assert "open" in field_names
        assert "volume" in field_names

    def test_field_comparison_structure(self):
        prices = _make_matching_prices()
        result = cross_source_check(prices, prices)
        fc = result.fields[0]
        assert isinstance(fc, FieldComparison)
        assert fc.n_compared == 100
        assert fc.n_mismatches == 0
        assert fc.max_diff == 0.0


class TestCrossSourceDivergence:
    def test_small_diff_within_tolerance(self):
        prices_a = _make_matching_prices()
        prices_b = {k: v + 0.001 if k != "volume" else v
                     for k, v in prices_a.items()}
        result = cross_source_check(prices_a, prices_b, tolerance_pct=0.01)
        # 0.001 on ~100 price = ~0.001% — well within 1%
        close_fc = next(f for f in result.fields if f.field == "close")
        assert close_fc.passed

    def test_large_diff_fails(self):
        prices_a = _make_matching_prices()
        prices_b = {k: v * 1.05 if k != "volume" else v
                     for k, v in prices_a.items()}  # 5% off
        result = cross_source_check(prices_a, prices_b, tolerance_pct=0.01)
        assert not result.passed
        assert len(result.issues) > 0

    def test_different_lengths(self):
        prices_a = _make_matching_prices(100)
        prices_b = _make_matching_prices(80)
        result = cross_source_check(prices_a, prices_b)
        assert result.n_overlapping == 80
        # Should note bar count mismatch
        assert any("Bar count mismatch" in i for i in result.issues)

    def test_empty_source_fails(self):
        prices_a = _make_matching_prices()
        prices_b = {"close": np.array([]), "open": np.array([])}
        result = cross_source_check(prices_a, prices_b)
        assert not result.passed

    def test_custom_fields(self):
        prices = _make_matching_prices()
        result = cross_source_check(prices, prices, fields=["close"])
        assert len(result.fields) == 1
        assert result.fields[0].field == "close"


class TestStaleDetection:
    def test_no_stale_data(self):
        prices = _make_matching_prices()
        issues = detect_stale_source(prices, "yfinance")
        assert issues == []

    def test_stale_data_detected(self):
        close = np.array([100.0] * 20 + [101.0])  # 20 identical
        prices = {"close": close}
        issues = detect_stale_source(prices, "yfinance", max_constant_bars=10)
        assert len(issues) == 1
        assert "stale" in issues[0].lower()

    def test_short_constant_ok(self):
        close = np.array([100.0] * 5 + [101.0])
        prices = {"close": close}
        issues = detect_stale_source(prices, "yfinance", max_constant_bars=10)
        assert issues == []


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.13: Declarative Scheduling (§19.8)
# ═══════════════════════════════════════════════════════════════════

from praxis.scheduler.declarative import (
    SchedulingDeclaration,
    parse_scheduling_declaration,
    infer_schedules,
)
from praxis.config import ModelConfig


def _make_model_config(name="test_model", model_type="SingleAssetModel"):
    """Create a minimal ModelConfig for testing."""
    yaml_str = f"""
model:
  name: {name}
  type: {model_type}
  universe:
    tickers: [SPY]
signal:
  method: sma_crossover
  params:
    fast_period: 10
    slow_period: 20
"""
    return ModelConfig.from_yaml_string(yaml_str)


class TestSchedulingDeclaration:
    def test_default_values(self):
        decl = SchedulingDeclaration()
        assert decl.state == "active"
        assert decl.priority == 50
        assert decl.data_frequency == "daily"
        assert decl.signal_schedule == "market_open"

    def test_parse_minimal_config(self):
        config = _make_model_config()
        decl = parse_scheduling_declaration(config)
        assert decl.state == "active"
        assert decl.data_sources == ["yfinance"]


class TestInferSchedules:
    def test_infers_data_and_signal(self):
        config = _make_model_config()
        schedules = infer_schedules(config)
        # Should get at least data_load + compute_signal + execute
        names = [s.name for s in schedules]
        assert any("data_load" in n for n in names)
        assert any("compute_signal" in n for n in names)

    def test_schedule_names_include_model(self):
        config = _make_model_config("my_model")
        schedules = infer_schedules(config)
        assert all("my_model" in s.name for s in schedules)

    def test_schedules_have_cron(self):
        config = _make_model_config()
        schedules = infer_schedules(config)
        for s in schedules:
            assert s.cron is not None
            assert len(s.cron) > 0

    def test_schedules_are_enabled(self):
        config = _make_model_config()
        schedules = infer_schedules(config)
        assert all(s.enabled for s in schedules)


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.14: Tick DAG Engine (§19.4-19.5)
# ═══════════════════════════════════════════════════════════════════

from praxis.scheduler.declarative import (
    TickDAG,
    DAGNode,
    DAGEdge,
    DAGExecutionResult,
    NodeStatus,
    build_tick_dag,
)


class TestTickDAGBasic:
    def test_empty_dag(self):
        dag = TickDAG()
        assert dag.size == 0
        assert dag.topological_order() == []

    def test_add_node(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="n1", node_type="data_load"))
        assert dag.size == 1

    def test_add_edge(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="n1", node_type="data_load"))
        dag.add_node(DAGNode(id="n2", node_type="compute"))
        dag.add_edge("n1", "n2")
        assert len(dag.edges) == 1

    def test_duplicate_node_raises(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="n1", node_type="data_load"))
        with pytest.raises(ValueError, match="Duplicate"):
            dag.add_node(DAGNode(id="n1", node_type="compute"))

    def test_edge_unknown_source_raises(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="n1", node_type="data_load"))
        with pytest.raises(ValueError, match="Unknown source"):
            dag.add_edge("nonexistent", "n1")

    def test_edge_unknown_target_raises(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="n1", node_type="data_load"))
        with pytest.raises(ValueError, match="Unknown target"):
            dag.add_edge("n1", "nonexistent")


class TestTopologicalOrder:
    def test_linear_chain(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="data_load"))
        dag.add_node(DAGNode(id="b", node_type="validate"))
        dag.add_node(DAGNode(id="c", node_type="compute"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        order = dag.topological_order()
        assert order == ["a", "b", "c"]

    def test_diamond_dag(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="load", node_type="data_load"))
        dag.add_node(DAGNode(id="val_a", node_type="validate"))
        dag.add_node(DAGNode(id="val_b", node_type="validate"))
        dag.add_node(DAGNode(id="merge", node_type="compute"))
        dag.add_edge("load", "val_a")
        dag.add_edge("load", "val_b")
        dag.add_edge("val_a", "merge")
        dag.add_edge("val_b", "merge")
        order = dag.topological_order()
        assert order[0] == "load"
        assert order[-1] == "merge"
        assert set(order[1:3]) == {"val_a", "val_b"}

    def test_cycle_detected(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="compute"))
        dag.add_node(DAGNode(id="b", node_type="compute"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(ValueError, match="Cycle"):
            dag.topological_order()

    def test_independent_nodes(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="data_load"))
        dag.add_node(DAGNode(id="b", node_type="data_load"))
        dag.add_node(DAGNode(id="c", node_type="data_load"))
        order = dag.topological_order()
        assert set(order) == {"a", "b", "c"}


class TestDAGDependencies:
    def test_get_dependencies(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="data_load"))
        dag.add_node(DAGNode(id="b", node_type="validate"))
        dag.add_node(DAGNode(id="c", node_type="compute"))
        dag.add_edge("a", "c")
        dag.add_edge("b", "c")
        deps = dag.get_dependencies("c")
        assert set(deps) == {"a", "b"}

    def test_get_dependents(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="data_load"))
        dag.add_node(DAGNode(id="b", node_type="validate"))
        dag.add_node(DAGNode(id="c", node_type="compute"))
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dependents = dag.get_dependents("a")
        assert set(dependents) == {"b", "c"}


class TestDAGExecution:
    def test_execute_empty_dag(self):
        dag = TickDAG()
        result = dag.execute()
        assert isinstance(result, DAGExecutionResult)
        assert result.completed == 0

    def test_execute_all_succeed(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="data_load"))
        dag.add_node(DAGNode(id="b", node_type="compute"))
        dag.add_edge("a", "b")
        result = dag.execute()
        assert result.completed == 2
        assert result.failed == 0
        assert result.all_succeeded

    def test_execute_with_executor(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="data_load"))
        results_collected = []

        def executor(node):
            results_collected.append(node.id)
            return f"done_{node.id}"

        result = dag.execute(executor=executor)
        assert result.completed == 1
        assert "a" in results_collected

    def test_execute_failure_skips_dependents(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="load", node_type="data_load"))
        dag.add_node(DAGNode(id="compute", node_type="compute"))
        dag.add_edge("load", "compute")

        def failing_executor(node):
            if node.id == "load":
                raise RuntimeError("Load failed!")
            return "ok"

        result = dag.execute(executor=failing_executor)
        assert result.failed == 1
        assert result.skipped == 1
        assert not result.all_succeeded

    def test_execute_records_duration(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="compute"))
        result = dag.execute()
        assert result.total_duration >= 0

    def test_node_status_tracking(self):
        dag = TickDAG()
        dag.add_node(DAGNode(id="a", node_type="compute"))
        result = dag.execute()
        node = result.nodes[0]
        assert node.status == NodeStatus.COMPLETED
        assert node.duration_seconds >= 0


class TestBuildTickDAG:
    def test_single_model(self):
        models = [{"name": "model_a", "tickers": ["SPY"], "source": "yfinance"}]
        dag = build_tick_dag(models)
        assert dag.size == 4  # load, validate, compute, execute
        order = dag.topological_order()
        assert order[0] == "load_model_a"
        assert order[-1] == "execute_model_a"

    def test_two_models_separate(self):
        models = [
            {"name": "model_a", "tickers": ["SPY"], "source": "yfinance"},
            {"name": "model_b", "tickers": ["QQQ"], "source": "yfinance"},
        ]
        dag = build_tick_dag(models)
        assert dag.size == 8  # 4 per model

    def test_merged_data_loads(self):
        """Two models with same tickers + source → merged load node."""
        models = [
            {"name": "model_a", "tickers": ["SPY"], "source": "yfinance"},
            {"name": "model_b", "tickers": ["SPY"], "source": "yfinance"},
        ]
        dag = build_tick_dag(models, merge_data_loads=True)
        # load is shared: 1 load + 2*(validate + compute + execute) = 7
        assert dag.size == 7
        load_nodes = [n for n in dag.nodes if n.node_type == "data_load"]
        assert len(load_nodes) == 1

    def test_no_merge_different_sources(self):
        """Different sources → separate loads even for same tickers."""
        models = [
            {"name": "model_a", "tickers": ["SPY"], "source": "yfinance"},
            {"name": "model_b", "tickers": ["SPY"], "source": "polygon"},
        ]
        dag = build_tick_dag(models, merge_data_loads=True)
        load_nodes = [n for n in dag.nodes if n.node_type == "data_load"]
        assert len(load_nodes) == 2

    def test_merge_disabled(self):
        models = [
            {"name": "model_a", "tickers": ["SPY"], "source": "yfinance"},
            {"name": "model_b", "tickers": ["SPY"], "source": "yfinance"},
        ]
        dag = build_tick_dag(models, merge_data_loads=False)
        assert dag.size == 8  # No sharing

    def test_dag_is_acyclic(self):
        models = [
            {"name": "m1", "tickers": ["SPY"], "source": "yf"},
            {"name": "m2", "tickers": ["QQQ"], "source": "yf"},
        ]
        dag = build_tick_dag(models)
        order = dag.topological_order()  # Would raise on cycle
        assert len(order) == dag.size

    def test_end_to_end_execute(self):
        """Build DAG from models and execute it."""
        models = [
            {"name": "chan_cpo", "tickers": ["GLD", "GDX"], "source": "yfinance"},
        ]
        dag = build_tick_dag(models)
        executed = []

        def executor(node):
            executed.append(node.id)
            return "ok"

        result = dag.execute(executor=executor)
        assert result.all_succeeded
        assert len(executed) == 4
        # Load must execute before validate
        assert executed.index("load_chan_cpo") < executed.index("validate_chan_cpo")
