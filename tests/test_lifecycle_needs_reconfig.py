"""
Tests for Model Lifecycle (2.12), Needs Filter (2.6), Logger Reconfig (2.14).
"""

import time
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest

from praxis.datastore.calendar import Calendar
from praxis.datastore.database import PraxisDatabase
from praxis.datastore.keys import EntityKeys
from praxis.datastore.lifecycle import ModelLifecycle, VALID_TRANSITIONS
from praxis.datastore.needs_filter import NeedsFilter, NeedsResult
from praxis.logger.adapters import AgentAdapter
from praxis.logger.core import PraxisLogger
from praxis.logger.reconfig import LoggerReconfig
from praxis.security import SecurityMaster


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


@pytest.fixture
def db():
    database = PraxisDatabase(":memory:")
    database.initialize()
    return database


# ═══════════════════════════════════════════════════════════════════
#  Model Lifecycle (Phase 2.12)
# ═══════════════════════════════════════════════════════════════════

class TestModelLifecycleCreate:
    def test_create_draft(self, db):
        lc = ModelLifecycle(db.connection)
        base_id = lc.create("model_alpha")
        state = lc.get_state("model_alpha")
        assert state is not None
        assert state["state"] == "DRAFT"
        assert state["execution_mode"] == "backtest_only"

    def test_create_with_params(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("model_beta", data_frequency="intraday_1min",
                   priority=10, execution_mode="live")
        state = lc.get_state("model_beta")
        assert state["data_frequency"] == "intraday_1min"
        assert state["priority"] == 10
        assert state["execution_mode"] == "live"

    def test_count(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        lc.create("m2")
        assert lc.count() == 2


class TestModelLifecycleTransitions:
    def test_draft_to_active(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        ok = lc.activate("m1", reason="passed validation")
        assert ok is True
        assert lc.get_state("m1")["state"] == "ACTIVE"

    def test_active_to_paused(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        lc.activate("m1")
        ok = lc.pause("m1", reason="reviewing")
        assert ok is True
        assert lc.get_state("m1")["state"] == "PAUSED"

    def test_paused_to_active(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        lc.activate("m1")
        lc.pause("m1")
        ok = lc.activate("m1", reason="resume")
        assert ok is True
        assert lc.get_state("m1")["state"] == "ACTIVE"

    def test_active_to_retired(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        lc.activate("m1")
        ok = lc.retire("m1", reason="replaced by v2")
        assert ok is True
        assert lc.get_state("m1")["state"] == "RETIRED"

    def test_retired_is_terminal(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        lc.activate("m1")
        lc.retire("m1")
        ok = lc.activate("m1")
        assert ok is False
        assert lc.get_state("m1")["state"] == "RETIRED"

    def test_invalid_transition_rejected(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        # Can't go DRAFT → PAUSED
        ok = lc.pause("m1")
        assert ok is False
        assert lc.get_state("m1")["state"] == "DRAFT"

    def test_backtesting_cycle(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        ok = lc.start_backtest("m1")
        assert ok is True
        assert lc.get_state("m1")["state"] == "BACKTESTING"

        ok = lc.complete_backtest("m1")
        assert ok is True
        assert lc.get_state("m1")["state"] == "DRAFT"

    def test_error_and_recovery(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        lc.activate("m1")
        lc.mark_error("m1", reason="repeated failures")
        assert lc.get_state("m1")["state"] == "ERROR"

        lc.activate("m1", reason="fix deployed")
        assert lc.get_state("m1")["state"] == "ACTIVE"

    def test_invalid_state_raises(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        with pytest.raises(ValueError, match="Invalid state"):
            lc.transition("m1", "FLYING")

    def test_nonexistent_model_raises(self, db):
        lc = ModelLifecycle(db.connection)
        with pytest.raises(ValueError, match="not found"):
            lc.transition("nonexistent", "ACTIVE")


class TestModelLifecycleAudit:
    def test_full_history(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        time.sleep(0.01)
        lc.activate("m1", reason="go live")
        time.sleep(0.01)
        lc.pause("m1", reason="review")
        time.sleep(0.01)
        lc.activate("m1", reason="resume")

        history = lc.get_history("m1")
        assert len(history) == 4
        states = [h["state"] for h in history]
        assert states == ["DRAFT", "ACTIVE", "PAUSED", "ACTIVE"]

    def test_changed_by_tracked(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1", changed_by="jeff")
        time.sleep(0.01)
        lc.activate("m1", changed_by="scheduler")

        history = lc.get_history("m1")
        assert history[0]["changed_by"] == "jeff"
        assert history[1]["changed_by"] == "scheduler"

    def test_params_preserved_across_transitions(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1", priority=10, data_frequency="daily",
                   execution_mode="live")
        lc.activate("m1")

        state = lc.get_state("m1")
        assert state["priority"] == 10
        assert state["data_frequency"] == "daily"
        assert state["execution_mode"] == "live"


class TestModelLifecycleQueries:
    def test_get_active_models(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        lc.activate("m1")
        lc.create("m2")
        lc.create("m3")
        lc.activate("m3")
        lc.pause("m3")

        active = lc.get_active_models()
        assert len(active) == 1
        assert active[0]["model_state_bpk"] == "m1"

    def test_count_by_state(self, db):
        lc = ModelLifecycle(db.connection)
        lc.create("m1")
        lc.create("m2")
        lc.activate("m2")
        lc.create("m3")
        lc.activate("m3")
        lc.retire("m3")

        assert lc.count("DRAFT") == 1
        assert lc.count("ACTIVE") == 1
        assert lc.count("RETIRED") == 1


# ═══════════════════════════════════════════════════════════════════
#  Needs Filter (Phase 2.6)
# ═══════════════════════════════════════════════════════════════════

class TestNeedsFilterBasic:
    def test_all_needed_when_empty(self, db):
        nf = NeedsFilter(db.connection)
        # Populate calendar
        cal = Calendar(db.connection)
        cal.populate(2024, 2024)

        master = SecurityMaster(db.connection)
        base_id = master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="test")

        result = nf.compute_price_needs([base_id], "2024-01-02", "2024-01-05")
        assert result.total_required > 0
        assert result.already_loaded == 0
        assert result.delta == result.total_required

    def test_nothing_needed_when_all_loaded(self, db):
        nf = NeedsFilter(db.connection)
        cal = Calendar(db.connection)
        cal.populate(2024, 2024)

        master = SecurityMaster(db.connection)
        base_id = master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="test")

        # Insert prices for the range
        bdays = cal.business_days("2024-01-02", "2024-01-05")
        for d in bdays:
            keys = EntityKeys.create(f"{base_id}|{d}")
            db.connection.execute("""
                INSERT INTO fact_price_daily (
                    price_hist_id, price_base_id, price_bpk,
                    security_base_id, trade_date, close, source
                ) VALUES ($1, $2, $3, $4, $5, 100.0, 'test')
            """, [keys.hist_id, keys.base_id, keys.bpk, base_id, d])

        result = nf.compute_price_needs([base_id], "2024-01-02", "2024-01-05")
        assert result.delta == 0
        assert result.savings_pct > 0

    def test_partial_load_computes_delta(self, db):
        nf = NeedsFilter(db.connection)
        cal = Calendar(db.connection)
        cal.populate(2024, 2024)

        master = SecurityMaster(db.connection)
        base_id = master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="test")

        # Load only first 2 business days
        bdays = cal.business_days("2024-01-02", "2024-01-31")
        for d in bdays[:2]:
            keys = EntityKeys.create(f"{base_id}|{d}")
            db.connection.execute("""
                INSERT INTO fact_price_daily (
                    price_hist_id, price_base_id, price_bpk,
                    security_base_id, trade_date, close, source
                ) VALUES ($1, $2, $3, $4, $5, 100.0, 'test')
            """, [keys.hist_id, keys.base_id, keys.bpk, base_id, d])

        result = nf.compute_price_needs([base_id], "2024-01-02", "2024-01-31")
        assert result.already_loaded == 2
        assert result.delta == result.total_required - 2


class TestNeedsFilterCrossModel:
    def test_cross_model_deduplication(self, db):
        """§19.3: Model B doesn't reload what Model A already loaded."""
        nf = NeedsFilter(db.connection)
        cal = Calendar(db.connection)
        cal.populate(2024, 2024)

        master = SecurityMaster(db.connection)
        aapl = master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="test")
        msft = master.match_or_create("EQUITY", {"TICKER": "MSFT"}, source="test")

        # Model A loads AAPL for Jan 2-5
        bdays = cal.business_days("2024-01-02", "2024-01-05")
        for d in bdays:
            keys = EntityKeys.create(f"{aapl}|{d}")
            db.connection.execute("""
                INSERT INTO fact_price_daily (
                    price_hist_id, price_base_id, price_bpk,
                    security_base_id, trade_date, close, source
                ) VALUES ($1, $2, $3, $4, $5, 100.0, 'test')
            """, [keys.hist_id, keys.base_id, keys.bpk, aapl, d])

        # Model B needs both AAPL and MSFT for Jan 2-5
        result = nf.compute_price_needs([aapl, msft], "2024-01-02", "2024-01-05")

        # AAPL already loaded, only MSFT needed
        assert result.already_loaded == len(bdays)
        assert result.delta == len(bdays)  # Only MSFT dates

    def test_empty_securities_list(self, db):
        nf = NeedsFilter(db.connection)
        result = nf.compute_price_needs([], "2024-01-01", "2024-01-31")
        assert result.total_required == 0
        assert result.delta == 0


class TestNeedsResult:
    def test_savings_pct(self):
        r = NeedsResult(total_required=100, already_loaded=75, delta=25)
        assert r.savings_pct == 75.0

    def test_savings_pct_zero_required(self):
        r = NeedsResult()
        assert r.savings_pct == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Logger Reconfig (Phase 2.14)
# ═══════════════════════════════════════════════════════════════════

class TestLoggerReconfigTags:
    def test_add_tag(self):
        rc = LoggerReconfig()
        rc.add_tag("data_pipeline")
        assert "data_pipeline" in rc.list_tags()

    def test_remove_tag(self):
        rc = LoggerReconfig()
        rc.add_tag("data_pipeline")
        rc.remove_tag("data_pipeline")
        assert "data_pipeline" not in rc.list_tags()

    def test_set_tag_level(self):
        rc = LoggerReconfig()
        rc.set_tag_level("security_resolve", "DEBUG")
        status = rc.status()
        assert "security_resolve" in status["tag_levels"]


class TestLoggerReconfigLevel:
    def test_set_global_level(self):
        rc = LoggerReconfig()
        rc.set_level("DEBUG")
        assert rc.get_level() == 10

    def test_set_adapter_level(self):
        log = PraxisLogger.instance()
        log.configure_defaults()
        rc = LoggerReconfig(log)
        rc.set_adapter_level("terminal", "WARNING")
        status = rc.status()
        terminal = [a for a in status["adapters"] if a["name"] == "terminal"]
        assert len(terminal) == 1
        assert terminal[0]["level"] == 30


class TestLoggerReconfigStatus:
    def test_status_structure(self):
        log = PraxisLogger.instance()
        log.configure_defaults()
        rc = LoggerReconfig(log)
        status = rc.status()

        assert "global_level" in status
        assert "active_tags" in status
        assert "adapters" in status
        assert isinstance(status["adapters"], list)

    def test_status_shows_adapters(self):
        log = PraxisLogger.instance()
        log.configure_defaults()
        rc = LoggerReconfig(log)
        status = rc.status()

        names = [a["name"] for a in status["adapters"]]
        assert "terminal" in names


class TestLoggerReconfigAgent:
    def test_agent_buffer_read(self):
        log = PraxisLogger.instance()
        log.configure_defaults()

        agent = AgentAdapter(name="agent", min_level=10)
        log.add_adapter(agent)
        log.activate_tag("test_tag")

        log.info("test message", tags={"test_tag"})
        log.info("another message", tags={"test_tag"})

        rc = LoggerReconfig(log)
        buffer = rc.get_agent_buffer(n=10)
        assert len(buffer) >= 2
        assert buffer[-1]["message"] == "another message"

    def test_agent_buffer_clear(self):
        log = PraxisLogger.instance()
        log.configure_defaults()

        agent = AgentAdapter(name="agent", min_level=10)
        log.add_adapter(agent)
        log.activate_tag("test_tag")
        log.info("test", tags={"test_tag"})

        rc = LoggerReconfig(log)
        rc.clear_agent_buffer()
        assert len(rc.get_agent_buffer()) == 0

    def test_agent_buffer_empty_without_adapter(self):
        rc = LoggerReconfig()
        assert rc.get_agent_buffer() == []

    def test_agent_buffer_filter_by_tags(self):
        log = PraxisLogger.instance()
        log.configure_defaults()

        agent = AgentAdapter(name="agent", min_level=10)
        log.add_adapter(agent)
        log.activate_tag("alpha")
        log.activate_tag("beta")

        log.info("alpha msg", tags={"alpha"})
        log.info("beta msg", tags={"beta"})

        rc = LoggerReconfig(log)
        alpha = rc.get_agent_buffer(tags={"alpha"})
        assert len(alpha) == 1
        assert alpha[0]["message"] == "alpha msg"
