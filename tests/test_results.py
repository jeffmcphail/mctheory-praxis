"""
Tests for BacktestResult storage (Phase 1.8) + Ephemeral mode (Phase 1.9).

Covers:
- Save backtest results to fact_backtest_run
- Load results by BPK
- List runs with model name filter
- Results STRUCT round-trip (all 13 metrics)
- Params JSON snapshot preserved
- rpt_backtest_summary view (pre-joined)
- Multiple runs for same model
- Ephemeral mode: full pipeline without temporal views
- PraxisRunner with database storage
"""

import json
import numpy as np
import polars as pl
import pytest

from praxis.backtest import VectorizedEngine, BacktestOutput
from praxis.config import ModelConfig
from praxis.datastore.database import PraxisDatabase
from praxis.datastore.results import BacktestResultStore
from praxis.executor import SimpleExecutor
from praxis.logger.core import PraxisLogger
from praxis.registry import FunctionRegistry
from praxis.registry.defaults import register_defaults
from praxis.runner import PraxisRunner


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
def db():
    """Persistent in-memory database with all tables + views."""
    database = PraxisDatabase(":memory:")
    database.initialize()
    return database


@pytest.fixture
def ephemeral_db():
    """Ephemeral database: tables only, no views."""
    db = PraxisDatabase.ephemeral()
    db.initialize()
    return db


@pytest.fixture
def store(db):
    return BacktestResultStore(db.connection)


@pytest.fixture
def sma_config():
    return ModelConfig.from_dict({
        "model": {"name": "test_sma", "type": "SingleAssetModel", "version": "v1.0"},
        "signal": {"method": "sma_crossover", "fast_period": 10, "slow_period": 50},
        "sizing": {"method": "fixed_fraction", "fraction": 1.0},
        "backtest": {"engine": "vectorized"},
    })


@pytest.fixture
def bt_output():
    """Run a real backtest to get a BacktestOutput."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    positions = np.ones(200)
    engine = VectorizedEngine()
    return engine.run(positions, prices)


# ═══════════════════════════════════════════════════════════════════
#  Save + Load
# ═══════════════════════════════════════════════════════════════════

class TestSaveLoad:
    def test_save_returns_keys(self, store, sma_config, bt_output):
        keys = store.save(sma_config, bt_output)
        assert "run_hist_id" in keys
        assert "run_base_id" in keys
        assert "run_bpk" in keys
        assert keys["run_bpk"].startswith("test_sma|")

    def test_load_by_bpk(self, store, sma_config, bt_output):
        keys = store.save(sma_config, bt_output)
        loaded = store.load(keys["run_bpk"])

        assert loaded is not None
        assert loaded["run_bpk"] == keys["run_bpk"]
        assert loaded["run_mode"] == "backtest"

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("nonexistent|2024-01-01") is None

    def test_count(self, store, sma_config, bt_output):
        assert store.count() == 0
        store.save(sma_config, bt_output)
        assert store.count() == 1
        store.save(sma_config, bt_output)
        assert store.count() == 2


# ═══════════════════════════════════════════════════════════════════
#  Results STRUCT Round-Trip
# ═══════════════════════════════════════════════════════════════════

class TestResultsRoundTrip:
    def test_all_13_metrics_stored(self, store, sma_config, bt_output):
        keys = store.save(sma_config, bt_output)
        loaded = store.load(keys["run_bpk"])

        results = loaded["results"]
        assert "total_return" in results
        assert "annualized_return" in results
        assert "sharpe_ratio" in results
        assert "sortino_ratio" in results
        assert "max_drawdown" in results
        assert "max_drawdown_duration_days" in results
        assert "win_rate" in results
        assert "profit_factor" in results
        assert "total_trades" in results
        assert "avg_trade_return" in results
        assert "avg_holding_days" in results
        assert "calmar_ratio" in results
        assert "volatility" in results

    def test_metrics_values_match(self, store, sma_config, bt_output):
        keys = store.save(sma_config, bt_output)
        loaded = store.load(keys["run_bpk"])

        results = loaded["results"]
        m = bt_output.metrics

        assert abs(results["total_return"] - m.total_return) < 1e-6
        assert abs(results["sharpe_ratio"] - m.sharpe_ratio) < 1e-6
        assert abs(results["max_drawdown"] - m.max_drawdown) < 1e-6
        assert results["total_trades"] == m.total_trades

    def test_execution_metadata(self, store, sma_config, bt_output):
        keys = store.save(sma_config, bt_output)
        loaded = store.load(keys["run_bpk"])

        assert loaded["engine"] == "vectorized"
        assert loaded["bar_count"] == bt_output.bar_count
        assert loaded["duration_seconds"] >= 0


# ═══════════════════════════════════════════════════════════════════
#  Params JSON Snapshot
# ═══════════════════════════════════════════════════════════════════

class TestParamsSnapshot:
    def test_params_preserved(self, store, sma_config, bt_output):
        keys = store.save(sma_config, bt_output)
        loaded = store.load(keys["run_bpk"])

        params = json.loads(loaded["params"]) if isinstance(loaded["params"], str) else loaded["params"]
        assert params["model"]["name"] == "test_sma"
        assert params["signal"]["method"] == "sma_crossover"
        assert params["signal"]["fast_period"] == 10

    def test_config_hash_reproducible(self, store, sma_config, bt_output):
        """Same config should produce same hash."""
        keys = store.save(sma_config, bt_output)
        loaded = store.load(keys["run_bpk"])
        params = json.loads(loaded["params"]) if isinstance(loaded["params"], str) else loaded["params"]

        # Reconstruct config from stored params
        reconstructed = ModelConfig.from_dict(params)
        assert reconstructed.config_hash == sma_config.config_hash


# ═══════════════════════════════════════════════════════════════════
#  List Runs
# ═══════════════════════════════════════════════════════════════════

class TestListRuns:
    def test_list_all(self, store, sma_config, bt_output):
        store.save(sma_config, bt_output)
        store.save(sma_config, bt_output)

        runs = store.list_runs()
        assert len(runs) == 2
        assert "total_return" in runs[0]
        assert "sharpe_ratio" in runs[0]

    def test_list_by_model_name(self, store, bt_output):
        config_a = ModelConfig.from_dict({
            "model": {"name": "model_a", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover"},
        })
        config_b = ModelConfig.from_dict({
            "model": {"name": "model_b", "type": "SingleAssetModel"},
            "signal": {"method": "ema_crossover"},
        })

        store.save(config_a, bt_output)
        store.save(config_b, bt_output)
        store.save(config_a, bt_output)

        runs_a = store.list_runs(model_name="model_a")
        runs_b = store.list_runs(model_name="model_b")

        assert len(runs_a) == 2
        assert len(runs_b) == 1

    def test_list_empty(self, store):
        runs = store.list_runs()
        assert runs == []


# ═══════════════════════════════════════════════════════════════════
#  rpt_backtest_summary View
# ═══════════════════════════════════════════════════════════════════

class TestReportView:
    def test_rpt_backtest_summary(self, db, sma_config, bt_output):
        """Pre-joined view: backtest + model definition."""
        conn = db.connection

        # Insert a model definition first
        from praxis.datastore.keys import EntityKeys
        from datetime import datetime, timezone

        model_keys = EntityKeys.create(sma_config.bpk)
        conn.execute("""
            INSERT INTO fact_model_definition (
                model_def_hist_id, model_def_base_id, model_def_bpk,
                model_name, model_type
            ) VALUES ($1, $2, $3, $4, $5)
        """, [
            model_keys.hist_id, model_keys.base_id, model_keys.bpk,
            sma_config.model.name, sma_config.model.type.value,
        ])

        # Save backtest result with FK
        store = BacktestResultStore(conn)
        store.save(
            sma_config, bt_output,
            model_def_base_id=model_keys.base_id,
            model_def_hist_id=model_keys.hist_id,
        )

        # Query the pre-joined report view
        rows = conn.execute("""
            SELECT
                model_name, model_type,
                sharpe_ratio,
                total_return,
                max_drawdown,
                total_trades
            FROM rpt_backtest_summary
        """).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "test_sma"
        assert rows[0][1] == "SingleAssetModel"


# ═══════════════════════════════════════════════════════════════════
#  Phase 1.9: Ephemeral Mode
# ═══════════════════════════════════════════════════════════════════

class TestEphemeralMode:
    def test_ephemeral_has_tables(self, ephemeral_db):
        tables = ephemeral_db.tables()
        assert "fact_backtest_run" in tables
        assert "fact_model_definition" in tables

    def test_ephemeral_no_views(self, ephemeral_db):
        views = ephemeral_db.views()
        assert len(views) == 0

    def test_ephemeral_save_load(self, ephemeral_db, sma_config, bt_output):
        """Backtest results work in ephemeral mode."""
        store = BacktestResultStore(ephemeral_db.connection)
        keys = store.save(sma_config, bt_output, platform_mode="ephemeral")
        loaded = store.load(keys["run_bpk"])

        assert loaded is not None
        assert loaded["platform_mode"] == "ephemeral"
        assert loaded["results"]["total_trades"] == bt_output.metrics.total_trades

    def test_ephemeral_full_pipeline(self, ephemeral_db):
        """End-to-end: YAML → signal → sizing → backtest → store in ephemeral DB."""
        reg = FunctionRegistry.instance()
        register_defaults(reg)

        config = ModelConfig.from_yaml_string("""
model:
  name: ephemeral_sma
  type: SingleAssetModel
signal:
  method: sma_crossover
  fast_period: 10
  slow_period: 50
sizing:
  method: fixed_fraction
  fraction: 1.0
""")
        np.random.seed(42)
        prices = pl.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(200) * 0.5),
        })

        # Run pipeline
        executor = SimpleExecutor(registry=reg)
        result = executor.execute(config, prices)
        assert result.success

        # Store in ephemeral DB
        store = BacktestResultStore(ephemeral_db.connection)
        engine = VectorizedEngine()
        bt_output = engine.run(result.positions, prices["close"])
        keys = store.save(config, bt_output, platform_mode="ephemeral")

        # Verify stored
        assert store.count() == 1
        loaded = store.load(keys["run_bpk"])
        assert loaded["results"]["total_trades"] > 0

    def test_ephemeral_status(self, ephemeral_db):
        status = ephemeral_db.status()
        assert status["mode"] == "ephemeral"
        assert status["initialized"] is True
