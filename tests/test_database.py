"""
Tests for DuckDB initialization (Phase 1.3).

Covers:
- Table creation (dim_security, fact_model_definition, fact_backtest_run, fact_log)
- View creation (vew_, vt2_, rpt_)
- Ephemeral mode (no views)
- Schema validation (columns, types)
- Insert + query through views
- Temporal view correctness (vew_ returns latest, vt2_ derives start/end dates)
- Logger database adapter wiring
- fact_model_definition STRUCT insert/query (validated by Spike 2)
- fact_backtest_run results STRUCT
- rpt_backtest_summary join
"""

from datetime import datetime, timezone, timedelta

import duckdb
import pytest

from praxis.datastore.database import PraxisDatabase
from praxis.datastore.keys import generate_base_id, EntityKeys
from praxis.logger.core import PraxisLogger
from praxis.logger.adapters import DatabaseAdapter, AgentAdapter
from praxis.logger.records import LogLevel


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_logger():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


@pytest.fixture
def db():
    """Persistent-mode in-memory database (has views)."""
    # Use in-memory but pretend it's persistent to get views
    database = PraxisDatabase.__new__(PraxisDatabase)
    database._path = None
    database._mode = "persistent"
    database._conn = duckdb.connect(":memory:")
    database._initialized = False
    database._log = PraxisLogger.instance()
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def ephemeral_db():
    """Ephemeral-mode database (no views)."""
    database = PraxisDatabase.ephemeral()
    database.initialize()
    yield database
    database.close()


# ═══════════════════════════════════════════════════════════════════
#  Table Creation
# ═══════════════════════════════════════════════════════════════════

class TestTableCreation:
    def test_all_tables_created(self, db):
        tables = db.tables()
        assert "dim_security" in tables
        assert "fact_model_definition" in tables
        assert "fact_backtest_run" in tables
        assert "fact_log" in tables

    def test_all_views_created_persistent(self, db):
        views = db.views()
        assert "vew_security" in views
        assert "vew_model_definition" in views
        assert "vt2_security" in views
        assert "vt2_model_definition" in views
        assert "rpt_backtest_summary" in views

    def test_no_views_in_ephemeral(self, ephemeral_db):
        views = ephemeral_db.views()
        assert len(views) == 0

    def test_tables_exist_in_ephemeral(self, ephemeral_db):
        tables = ephemeral_db.tables()
        assert "dim_security" in tables
        assert "fact_model_definition" in tables

    def test_initialize_returns_counts(self, db):
        # Already initialized in fixture; re-init is idempotent (IF NOT EXISTS)
        result = db.initialize()
        assert result["tables"] == 10
        assert result["views"] == 5

    def test_mode_property(self, db, ephemeral_db):
        assert db.mode == "persistent"
        assert ephemeral_db.mode == "ephemeral"


# ═══════════════════════════════════════════════════════════════════
#  Schema Validation
# ═══════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    def test_dim_security_columns(self, db):
        cols = db.table_columns("dim_security")
        col_names = [c["name"] for c in cols]
        assert "security_hist_id" in col_names
        assert "security_base_id" in col_names
        assert "security_bpk" in col_names
        assert "sec_type" in col_names
        assert "ticker" in col_names
        assert "isin" in col_names

    def test_fact_model_definition_has_structs(self, db):
        cols = db.table_columns("fact_model_definition")
        col_names = [c["name"] for c in cols]
        assert "construction_params" in col_names
        assert "signal_params" in col_names
        assert "entry_params" in col_names
        assert "exit_params" in col_names
        assert "sizing_params" in col_names
        assert "cpo_params" in col_names
        assert "backtest_params" in col_names
        assert "risk_params" in col_names

    def test_fact_backtest_run_columns(self, db):
        cols = db.table_columns("fact_backtest_run")
        col_names = [c["name"] for c in cols]
        assert "run_hist_id" in col_names
        assert "model_def_base_id" in col_names
        assert "run_timestamp" in col_names
        assert "results" in col_names
        assert "params" in col_names

    def test_fact_log_columns(self, db):
        cols = db.table_columns("fact_log")
        col_names = [c["name"] for c in cols]
        assert "log_id" in col_names
        assert "log_timestamp" in col_names
        assert "level" in col_names
        assert "tags" in col_names
        assert "context" in col_names


# ═══════════════════════════════════════════════════════════════════
#  dim_security: Insert + View Queries
# ═══════════════════════════════════════════════════════════════════

class TestDimSecurity:
    def _insert_security(self, db, bpk, ticker, name, ts):
        keys = EntityKeys.create(bpk, timestamp=ts)
        db.connection.execute(
            """INSERT INTO dim_security
               (security_hist_id, security_base_id, security_bpk,
                sec_type, ticker, name, currency, exchange, status)
               VALUES (?, ?, ?, 'EQUITY', ?, ?, 'USD', 'NASDAQ', 'ACTIVE')""",
            [keys.hist_id, keys.base_id, keys.bpk, ticker, name],
        )
        return keys

    def test_insert_and_vew_returns_latest(self, db):
        ts1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 15, 14, 0, 0, tzinfo=timezone.utc)

        bpk = "EQUITY|TICKER|AAPL"
        k1 = self._insert_security(db, bpk, "AAPL", "Apple Inc v1", ts1)
        # Update: name change
        k2 = EntityKeys.new_version(bpk, k1.base_id, timestamp=ts2)
        db.connection.execute(
            """INSERT INTO dim_security
               (security_hist_id, security_base_id, security_bpk,
                sec_type, ticker, name, currency, exchange, status)
               VALUES (?, ?, ?, 'EQUITY', 'AAPL', 'Apple Inc v2', 'USD', 'NASDAQ', 'ACTIVE')""",
            [k2.hist_id, k2.base_id, k2.bpk],
        )

        # vew_ should return latest version only
        result = db.connection.execute(
            "SELECT name FROM vew_security WHERE security_base_id = ?",
            [k1.base_id],
        ).fetchone()
        assert result[0] == "Apple Inc v2"

    def test_vt2_derives_start_end_dates(self, db):
        bpk = "EQUITY|TICKER|MSFT"
        ts1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 7, 1, 14, 0, 0, tzinfo=timezone.utc)

        k1 = self._insert_security(db, bpk, "MSFT", "Microsoft v1", ts1)
        k2 = EntityKeys.new_version(bpk, k1.base_id, timestamp=ts2)
        db.connection.execute(
            """INSERT INTO dim_security
               (security_hist_id, security_base_id, security_bpk,
                sec_type, ticker, name, currency, exchange, status)
               VALUES (?, ?, ?, 'EQUITY', 'MSFT', 'Microsoft v2', 'USD', 'NASDAQ', 'ACTIVE')""",
            [k2.hist_id, k2.base_id, k2.bpk],
        )

        rows = db.connection.execute(
            """SELECT name, start_date, end_date
               FROM vt2_security
               WHERE security_base_id = ?
               ORDER BY start_date""",
            [k1.base_id],
        ).fetchall()

        assert len(rows) == 2
        # v1: start=2024-01-15, end=2024-06-30 (day before v2 start)
        assert rows[0][0] == "Microsoft v1"
        assert str(rows[0][1]) == "2024-01-15"
        assert str(rows[0][2]) == "2024-06-30"
        # v2: start=2024-07-01, end=9999-12-31
        assert rows[1][0] == "Microsoft v2"
        assert str(rows[1][1]) == "2024-07-01"
        assert str(rows[1][2]) == "9999-12-31"

    def test_vt2_multiple_updates_same_day_last_wins(self, db):
        """§2.3: Multiple updates same day → latest hist_id within that day wins."""
        bpk = "EQUITY|TICKER|GOOG"
        ts1 = datetime(2024, 3, 10, 9, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 3, 10, 15, 0, 0, tzinfo=timezone.utc)  # Same day!

        self._insert_security(db, bpk, "GOOG", "First update", ts1)
        k1 = EntityKeys.create(bpk, timestamp=ts1)
        k2 = EntityKeys.new_version(bpk, k1.base_id, timestamp=ts2)
        db.connection.execute(
            """INSERT INTO dim_security
               (security_hist_id, security_base_id, security_bpk,
                sec_type, ticker, name, currency, exchange, status)
               VALUES (?, ?, ?, 'EQUITY', 'GOOG', 'Second update same day', 'USD', 'NASDAQ', 'ACTIVE')""",
            [k2.hist_id, k2.base_id, k2.bpk],
        )

        rows = db.connection.execute(
            "SELECT name FROM vt2_security WHERE security_base_id = ?",
            [k1.base_id],
        ).fetchall()
        # Only one row per day — the later timestamp wins
        assert len(rows) == 1
        assert rows[0][0] == "Second update same day"


# ═══════════════════════════════════════════════════════════════════
#  fact_model_definition: STRUCT insert + query
# ═══════════════════════════════════════════════════════════════════

class TestFactModelDefinition:
    def test_insert_sma_config(self, db):
        """Insert a minimal SMA crossover model definition."""
        keys = EntityKeys.create("sma_crossover|v1.0")
        db.connection.execute(
            """INSERT INTO fact_model_definition (
                model_def_hist_id, model_def_base_id, model_def_bpk,
                model_name, model_type, model_version,
                signal_params, sizing_params, backtest_params,
                source_yaml
            ) VALUES (
                ?, ?, ?,
                'sma_crossover', 'SingleAssetModel', 'v1.0',
                {method: 'sma_crossover', params: '{"fast": 10, "slow": 50}'::JSON,
                 lookback: 50, threshold: NULL, confirmation: NULL, filters: NULL, composite: NULL},
                {method: 'fixed_fraction', fixed_fraction: 1.0,
                 max_position_pct: 1.0, kelly: NULL, volatility: NULL, risk_parity: NULL},
                {engine: 'vectorized', reconciliation_tolerance: NULL,
                 costs: NULL, slippage: NULL, fills: NULL, data: NULL, validation: NULL},
                'model:\n  name: sma_crossover\n  type: SingleAssetModel'
            )""",
            [keys.hist_id, keys.base_id, keys.bpk],
        )

        # Query through vew_
        result = db.connection.execute(
            "SELECT model_name, model_type, signal_params.method "
            "FROM vew_model_definition WHERE model_name = 'sma_crossover'"
        ).fetchone()
        assert result[0] == "sma_crossover"
        assert result[1] == "SingleAssetModel"
        assert result[2] == "sma_crossover"

    def test_insert_with_nested_structs(self, db):
        """Validate nested STRUCT access (confirmed by Spike 2)."""
        keys = EntityKeys.create("stat_arb_test|v1.0")
        db.connection.execute(
            """INSERT INTO fact_model_definition (
                model_def_hist_id, model_def_base_id, model_def_bpk,
                model_name, model_type, model_version,
                entry_params, exit_params
            ) VALUES (
                ?, ?, ?,
                'stat_arb_test', 'PairModel', 'v1.0',
                {method: 'threshold', long_threshold: -2.0, short_threshold: 2.0,
                 order_type: 'MARKET', limit_offset_pct: NULL, time_in_force: 'DAY',
                 max_entry_attempts: 3,
                 scale_in: {enabled: true, max_entries: 3, scale_factor: 0.5, min_interval: '1d'}},
                {method: 'threshold',
                 take_profit: {method: 'fixed', target: 0.05, "trailing": false, trail_pct: NULL},
                 stop_loss: {method: 'fixed', level: -0.03, "trailing": true, trail_pct: 0.02},
                 time_exit: {max_holding_days: 20, max_calendar_days: 30},
                 signal_exit: {method: 'mean_reversion', threshold: 0.0}}
            )""",
            [keys.hist_id, keys.base_id, keys.bpk],
        )

        # Query nested fields
        result = db.connection.execute(
            """SELECT entry_params.scale_in.max_entries,
                      exit_params.stop_loss.trail_pct,
                      exit_params.time_exit.max_holding_days
               FROM vew_model_definition
               WHERE model_name = 'stat_arb_test'"""
        ).fetchone()
        assert result[0] == 3
        assert result[1] == 0.02
        assert result[2] == 20

    def test_model_versioning_through_vt2(self, db):
        """§6.13: New version inserts new hist_id, vt2_ shows history."""
        bpk = "versioned_model|v1.0"
        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)

        k1 = EntityKeys.create(bpk, timestamp=ts1)
        db.connection.execute(
            """INSERT INTO fact_model_definition (
                model_def_hist_id, model_def_base_id, model_def_bpk,
                model_name, model_type, model_version
            ) VALUES (?, ?, ?, 'versioned_model', 'SingleAssetModel', 'v1.0')""",
            [k1.hist_id, k1.base_id, k1.bpk],
        )

        # v2 same base_id, different hist_id
        k2 = EntityKeys.new_version(bpk, k1.base_id, timestamp=ts2)
        db.connection.execute(
            """INSERT INTO fact_model_definition (
                model_def_hist_id, model_def_base_id, model_def_bpk,
                model_name, model_type, model_version
            ) VALUES (?, ?, ?, 'versioned_model', 'SingleAssetModel', 'v2.0')""",
            [k2.hist_id, k2.base_id, k2.bpk],
        )

        # vew_ returns latest only
        vew_result = db.connection.execute(
            "SELECT model_version FROM vew_model_definition WHERE model_name = 'versioned_model'"
        ).fetchall()
        assert len(vew_result) == 1
        assert vew_result[0][0] == "v2.0"

        # vt2_ returns both versions with start/end dates
        vt2_result = db.connection.execute(
            """SELECT model_version, start_date, end_date
               FROM vt2_model_definition
               WHERE model_def_base_id = ?
               ORDER BY start_date""",
            [k1.base_id],
        ).fetchall()
        assert len(vt2_result) == 2
        assert vt2_result[0][0] == "v1.0"
        assert str(vt2_result[0][2]) == "2024-03-14"  # Day before v2
        assert vt2_result[1][0] == "v2.0"
        assert str(vt2_result[1][2]) == "9999-12-31"  # Current


# ═══════════════════════════════════════════════════════════════════
#  fact_backtest_run: Results STRUCT + rpt_ view
# ═══════════════════════════════════════════════════════════════════

class TestFactBacktestRun:
    def _insert_model_and_run(self, db):
        """Helper: insert a model and a backtest run, return keys."""
        # Model
        model_keys = EntityKeys.create("sma_crossover|v1.0")
        db.connection.execute(
            """INSERT INTO fact_model_definition (
                model_def_hist_id, model_def_base_id, model_def_bpk,
                model_name, model_type, model_version
            ) VALUES (?, ?, ?, 'sma_crossover', 'SingleAssetModel', 'v1.0')""",
            [model_keys.hist_id, model_keys.base_id, model_keys.bpk],
        )

        # Backtest run
        run_ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        run_keys = EntityKeys.create(
            f"sma_crossover|{run_ts.isoformat()}", timestamp=run_ts
        )
        db.connection.execute(
            """INSERT INTO fact_backtest_run (
                run_hist_id, run_base_id, run_bpk,
                model_def_base_id, model_def_hist_id,
                run_timestamp, run_mode, platform_mode,
                start_date, end_date,
                results, duration_seconds, bar_count, engine
            ) VALUES (
                ?, ?, ?,
                ?, ?,
                ?, 'backtest', 'persistent',
                '2023-01-01', '2024-01-01',
                {total_return: 0.15, annualized_return: 0.15,
                 sharpe_ratio: 1.2, sortino_ratio: 1.5,
                 max_drawdown: -0.08, max_drawdown_duration_days: 45,
                 win_rate: 0.55, profit_factor: 1.8,
                 total_trades: 42, avg_trade_return: 0.003,
                 avg_holding_days: 5.2, calmar_ratio: 1.875,
                 volatility: 0.12},
                2.5, 252, 'vectorized'
            )""",
            [
                run_keys.hist_id, run_keys.base_id, run_keys.bpk,
                model_keys.base_id, model_keys.hist_id,
                run_ts,
            ],
        )
        return model_keys, run_keys

    def test_insert_and_query_results_struct(self, db):
        model_keys, run_keys = self._insert_model_and_run(db)
        result = db.connection.execute(
            """SELECT results.sharpe_ratio, results.total_return,
                      results.max_drawdown, results.total_trades
               FROM fact_backtest_run
               WHERE run_base_id = ?""",
            [run_keys.base_id],
        ).fetchone()
        assert abs(result[0] - 1.2) < 0.001
        assert abs(result[1] - 0.15) < 0.001
        assert abs(result[2] - (-0.08)) < 0.001
        assert result[3] == 42

    def test_rpt_backtest_summary_join(self, db):
        """§2.4: rpt_ view pre-joins fact with dimension."""
        model_keys, run_keys = self._insert_model_and_run(db)
        result = db.connection.execute(
            """SELECT model_name, model_type, sharpe_ratio, total_return,
                      max_drawdown, total_trades, engine
               FROM rpt_backtest_summary
               WHERE model_name = 'sma_crossover'"""
        ).fetchone()
        assert result[0] == "sma_crossover"
        assert result[1] == "SingleAssetModel"
        assert abs(result[2] - 1.2) < 0.001
        assert result[6] == "vectorized"


# ═══════════════════════════════════════════════════════════════════
#  fact_log: Direct insert + logger wiring
# ═══════════════════════════════════════════════════════════════════

class TestFactLog:
    def test_direct_insert(self, db):
        db.connection.execute(
            """INSERT INTO fact_log (log_id, log_timestamp, level, level_name, message, tags)
               VALUES (1, CURRENT_TIMESTAMP, 20, 'INFO', 'test message', ['datastore'])"""
        )
        result = db.connection.execute(
            "SELECT message, tags FROM fact_log WHERE log_id = 1"
        ).fetchone()
        assert result[0] == "test message"
        assert "datastore" in result[1]

    def test_logger_wire_and_write(self, db):
        """Wire logger DatabaseAdapter to fact_log and verify logs land."""
        log = PraxisLogger.instance()
        db_adapter = DatabaseAdapter(buffer_size=10)
        log.add_adapter(db_adapter)
        log.current_level = LogLevel.INFO

        # Wire
        result = db.wire_logger()
        assert result is True

        # Log something
        log.info("wired test message", tags={"datastore.init"})
        db_adapter.flush()

        # Verify in fact_log — wire_logger itself logs, so query by message
        row = db.connection.execute(
            "SELECT message, level, level_name FROM fact_log WHERE message = 'wired test message'"
        ).fetchone()
        assert row is not None
        assert row[0] == "wired test message"
        assert row[1] == 20
        assert row[2] == "INFO"

    def test_logger_wire_no_adapter(self, db):
        """wire_logger returns False when no DatabaseAdapter configured."""
        PraxisLogger.instance()  # No database adapter added
        assert db.wire_logger() is False


# ═══════════════════════════════════════════════════════════════════
#  Introspection
# ═══════════════════════════════════════════════════════════════════

class TestIntrospection:
    def test_status(self, db):
        status = db.status()
        assert status["mode"] == "persistent"
        assert status["initialized"] is True
        assert "dim_security" in status["tables"]
        assert "vew_security" in status["views"]

    def test_row_count_empty(self, db):
        assert db.row_count("dim_security") == 0

    def test_row_count_after_insert(self, db):
        keys = EntityKeys.create("EQUITY|TICKER|TEST")
        db.connection.execute(
            """INSERT INTO dim_security
               (security_hist_id, security_base_id, security_bpk, sec_type, ticker)
               VALUES (?, ?, ?, 'EQUITY', 'TEST')""",
            [keys.hist_id, keys.base_id, keys.bpk],
        )
        assert db.row_count("dim_security") == 1


# ═══════════════════════════════════════════════════════════════════
#  Ephemeral Mode
# ═══════════════════════════════════════════════════════════════════

class TestEphemeralMode:
    def test_tables_work_without_views(self, ephemeral_db):
        keys = EntityKeys.create("EQUITY|TICKER|TSLA")
        ephemeral_db.connection.execute(
            """INSERT INTO dim_security
               (security_hist_id, security_base_id, security_bpk, sec_type, ticker, name)
               VALUES (?, ?, ?, 'EQUITY', 'TSLA', 'Tesla')""",
            [keys.hist_id, keys.base_id, keys.bpk],
        )
        # Query table directly (no views in ephemeral)
        result = ephemeral_db.connection.execute(
            "SELECT name FROM dim_security WHERE ticker = 'TSLA'"
        ).fetchone()
        assert result[0] == "Tesla"

    def test_ephemeral_factory(self):
        db = PraxisDatabase.ephemeral()
        assert db.mode == "ephemeral"
        db.initialize()
        assert db.is_initialized
        db.close()
