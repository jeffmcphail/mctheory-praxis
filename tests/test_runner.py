"""
Tests for Config → Code Router (Phase 1.6).

Covers:
- SimpleExecutor: signal → sizing pipeline
- Executor dispatch by model type
- PraxisRunner: YAML → result, config → result
- Error handling: missing signal, unsupported model type
- CLI entry point
- Integration: end-to-end YAML config → execution
"""

import numpy as np
import polars as pl
import pytest

from praxis.config import ModelConfig, ModelType
from praxis.executor import (
    SimpleExecutor,
    ExecutionResult,
    get_executor,
)
from praxis.logger.core import PraxisLogger
from praxis.registry import FunctionRegistry
from praxis.registry.defaults import register_defaults
from praxis.runner import PraxisRunner, run_cli


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_singletons():
    FunctionRegistry.reset()
    PraxisLogger.reset()
    yield
    FunctionRegistry.reset()
    PraxisLogger.reset()


@pytest.fixture
def registry():
    reg = FunctionRegistry.instance()
    register_defaults(reg)
    return reg


@pytest.fixture
def price_data():
    np.random.seed(42)
    n = 200
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pl.DataFrame({"close": closes})


@pytest.fixture
def sma_config():
    return ModelConfig.from_dict({
        "model": {"name": "test_sma", "type": "SingleAssetModel", "version": "v1.0"},
        "signal": {"method": "sma_crossover", "fast_period": 10, "slow_period": 50},
        "sizing": {"method": "fixed_fraction", "fraction": 0.5},
    })


@pytest.fixture
def ema_config():
    return ModelConfig.from_dict({
        "model": {"name": "test_ema", "type": "SingleAssetModel"},
        "signal": {"method": "ema_crossover", "fast_period": 12, "slow_period": 26},
        "sizing": {"method": "fixed_fraction", "fraction": 1.0},
    })


# ═══════════════════════════════════════════════════════════════════
#  SimpleExecutor
# ═══════════════════════════════════════════════════════════════════

class TestSimpleExecutor:
    def test_sma_execution(self, registry, sma_config, price_data):
        executor = SimpleExecutor(registry=registry)
        result = executor.execute(sma_config, price_data)

        assert result.success is True
        assert result.error is None
        assert result.signals is not None
        assert result.positions is not None
        assert len(result.signals) == len(price_data)
        assert len(result.positions) == len(price_data)

    def test_ema_execution(self, registry, ema_config, price_data):
        executor = SimpleExecutor(registry=registry)
        result = executor.execute(ema_config, price_data)

        assert result.success is True
        assert len(result.signals) == len(price_data)

    def test_positions_respect_sizing(self, registry, sma_config, price_data):
        executor = SimpleExecutor(registry=registry)
        result = executor.execute(sma_config, price_data)

        # fraction=0.5, so positions should be in {-0.5, 0, 0.5}
        unique = set(result.positions.to_list())
        # Filter out None/null
        unique.discard(None)
        assert unique.issubset({-0.5, 0.0, 0.5})

    def test_default_sizing_when_none(self, registry, price_data):
        """When no sizing section, defaults to fixed_fraction 100%."""
        config = ModelConfig.from_dict({
            "model": {"name": "no_sizing", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover", "fast_period": 10, "slow_period": 50},
        })
        executor = SimpleExecutor(registry=registry)
        result = executor.execute(config, price_data)

        assert result.success is True
        unique = set(result.positions.to_list())
        unique.discard(None)
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_config_preserved_in_result(self, registry, sma_config, price_data):
        executor = SimpleExecutor(registry=registry)
        result = executor.execute(sma_config, price_data)
        assert result.config is sma_config
        assert result.config.model.name == "test_sma"

    def test_prices_preserved_in_result(self, registry, sma_config, price_data):
        executor = SimpleExecutor(registry=registry)
        result = executor.execute(sma_config, price_data)
        assert result.prices is price_data

    def test_missing_signal_method_returns_error(self, registry, price_data):
        config = ModelConfig.from_dict({
            "model": {"name": "bad", "type": "SingleAssetModel"},
            "signal": {"method": "nonexistent_signal"},
        })
        executor = SimpleExecutor(registry=registry)
        result = executor.execute(config, price_data)

        assert result.success is False
        assert result.error is not None
        assert "nonexistent_signal" in result.error


# ═══════════════════════════════════════════════════════════════════
#  Executor Dispatch
# ═══════════════════════════════════════════════════════════════════

class TestExecutorDispatch:
    def test_single_asset_routes_to_simple(self, registry, sma_config):
        executor = get_executor(sma_config, registry)
        assert isinstance(executor, SimpleExecutor)

    def test_cpo_model_dispatches(self, registry):
        config = ModelConfig.from_dict({
            "model": {"name": "cpo_test", "type": "CPOModel"},
        })
        executor = get_executor(config, registry)
        assert type(executor).__name__ == "CPOExecutor"

    def test_pair_model_not_yet_supported(self, registry):
        config = ModelConfig.from_dict({
            "model": {"name": "pair_test", "type": "PairModel"},
        })
        with pytest.raises(ValueError, match="No executor"):
            get_executor(config, registry)


# ═══════════════════════════════════════════════════════════════════
#  PraxisRunner
# ═══════════════════════════════════════════════════════════════════

class TestPraxisRunner:
    def test_run_config(self, sma_config, price_data):
        runner = PraxisRunner()
        result = runner.run_config(sma_config, price_data)

        assert result.success is True
        assert result.signals is not None
        assert result.positions is not None

    def test_run_yaml(self, tmp_path, price_data):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("""
model:
  name: yaml_test
  type: SingleAssetModel
signal:
  method: sma_crossover
  fast_period: 10
  slow_period: 50
sizing:
  method: fixed_fraction
  fraction: 1.0
""")
        runner = PraxisRunner()
        result = runner.run_yaml(yaml_file, price_data)

        assert result.success is True
        assert result.config.model.name == "yaml_test"

    def test_run_ema(self, ema_config, price_data):
        runner = PraxisRunner()
        result = runner.run_config(ema_config, price_data)
        assert result.success is True

    def test_runner_registers_defaults(self):
        """Runner should auto-register defaults."""
        runner = PraxisRunner()
        reg = FunctionRegistry.instance()
        assert reg.has("signals", "sma_crossover")
        assert reg.has("sizing", "fixed_fraction")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

class TestCLI:
    def test_no_args_returns_1(self):
        assert run_cli([]) == 1

    def test_missing_file_returns_1(self):
        assert run_cli(["nonexistent.yaml"]) == 1

    def test_no_prices_returns_1(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("model:\n  name: test\n  type: SingleAssetModel\nsignal:\n  method: sma_crossover\n")
        assert run_cli([str(yaml_file)]) == 1

    def test_full_cli(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("""
model:
  name: cli_test
  type: SingleAssetModel
signal:
  method: sma_crossover
  fast_period: 10
  slow_period: 50
""")
        np.random.seed(42)
        prices_file = tmp_path / "prices.csv"
        closes = 100 + np.cumsum(np.random.randn(200) * 0.5)
        pl.DataFrame({"close": closes}).write_csv(str(prices_file))

        result = run_cli([str(yaml_file), "--prices", str(prices_file)])
        assert result == 0


# ═══════════════════════════════════════════════════════════════════
#  Integration: Full YAML → Execution
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    SMA_YAML = """
model:
  name: sma_crossover
  type: SingleAssetModel
  version: v1.0
signal:
  method: sma_crossover
  fast_period: 10
  slow_period: 50
sizing:
  method: fixed_fraction
  fraction: 0.5
backtest:
  engine: vectorized
"""

    def test_end_to_end_sma(self, price_data):
        config = ModelConfig.from_yaml_string(self.SMA_YAML)
        runner = PraxisRunner()
        result = runner.run_config(config, price_data)

        assert result.success is True
        assert result.config.model.name == "sma_crossover"
        assert result.config.model.version == "v1.0"
        assert result.config.config_hash  # Non-empty hash
        assert len(result.signals) == 200
        assert len(result.positions) == 200
        assert result.positions.max() <= 0.5
        assert result.positions.min() >= -0.5

    def test_end_to_end_ema(self, price_data):
        config = ModelConfig.from_yaml_string("""
model:
  name: ema_test
  type: SingleAssetModel
signal:
  method: ema_crossover
  fast_period: 12
  slow_period: 26
""")
        runner = PraxisRunner()
        result = runner.run_config(config, price_data)

        assert result.success is True
        assert len(result.signals) == 200
