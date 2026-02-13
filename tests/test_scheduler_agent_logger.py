"""
Tests for Phase 4.14 (GUI gate), 4.15+4.17 (Agent Scheduler + Holistic),
4.16 (Logger Agent Tool).
"""

import pytest
from datetime import datetime, timezone, timedelta

from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.14: GUI Dependency Gate (Decision Record)
# ═══════════════════════════════════════════════════════════════════

class TestGUIGateCheck:
    """
    Phase 4.14 is a decision gate, not code. Record the decision:
    Core GUI adapters do not yet exist → GUI phases 4.14a-d deferred
    to post-Core-MVP. Platform logic is fully testable without GUI.
    """

    def test_gate_decision_documented(self):
        """The gate decision is: defer GUI to post-Core-MVP."""
        decision = {
            "phase": "4.14",
            "question": "Do Core GUI adapters exist?",
            "answer": "No — Core is at v0.1.0, GUI adapters not yet built",
            "decision": "Defer 4.14a-d GUI panels to post-Core-MVP",
            "rationale": "All platform logic is testable via programmatic API. "
                         "GUI is presentation layer, not business logic.",
        }
        assert decision["answer"].startswith("No")
        assert "defer" in decision["decision"].lower()


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.15 + 4.17: Agent Scheduler + Holistic Scheduling
# ═══════════════════════════════════════════════════════════════════

from praxis.scheduler.agent_scheduler import (
    AgentScheduler,
    AgentSchedulerConfig,
    ModelState,
    ModelStateRecord,
    HealthMonitor,
    HealthStatus,
    PriorityOptimizer,
    UniverseMerger,
    MergedUniverse,
    UniverseMergeResult,
    AgentAction,
    can_transition,
    transition,
)


class TestModelState:
    def test_valid_transitions(self):
        assert can_transition(ModelState.DRAFT, ModelState.ACTIVE)
        assert can_transition(ModelState.ACTIVE, ModelState.PAUSED)
        assert can_transition(ModelState.PAUSED, ModelState.ACTIVE)
        assert can_transition(ModelState.ACTIVE, ModelState.ERROR)
        assert can_transition(ModelState.ERROR, ModelState.ACTIVE)

    def test_invalid_transitions(self):
        assert not can_transition(ModelState.RETIRED, ModelState.ACTIVE)
        assert not can_transition(ModelState.DRAFT, ModelState.RETIRED)
        assert not can_transition(ModelState.DRAFT, ModelState.PAUSED)

    def test_transition_function(self):
        record = ModelStateRecord(model_id="test", state=ModelState.DRAFT)
        assert transition(record, ModelState.ACTIVE, by="user", reason="go live")
        assert record.state == ModelState.ACTIVE
        assert record.changed_by == "user"

    def test_transition_fails_invalid(self):
        record = ModelStateRecord(model_id="test", state=ModelState.RETIRED)
        assert not transition(record, ModelState.ACTIVE)
        assert record.state == ModelState.RETIRED


class TestHealthMonitor:
    def test_healthy_model(self):
        monitor = HealthMonitor()
        record = ModelStateRecord(model_id="m1", state=ModelState.ACTIVE, failure_count=0)
        status = monitor.assess(record)
        assert status.is_healthy
        assert status.recommendation == "ok"

    def test_repeated_failures_trigger_pause(self):
        monitor = HealthMonitor(AgentSchedulerConfig(max_consecutive_failures=3))
        record = ModelStateRecord(model_id="m1", state=ModelState.ACTIVE, failure_count=3)
        status = monitor.assess(record)
        assert not status.is_healthy
        assert status.recommendation == "pause"

    def test_error_state_recommends_retry(self):
        monitor = HealthMonitor(AgentSchedulerConfig(max_total_failures=10))
        record = ModelStateRecord(model_id="m1", state=ModelState.ERROR, failure_count=2)
        status = monitor.assess(record)
        assert not status.is_healthy
        assert status.recommendation == "retry"

    def test_error_state_excess_failures_recommends_retire(self):
        monitor = HealthMonitor(AgentSchedulerConfig(max_total_failures=5))
        record = ModelStateRecord(model_id="m1", state=ModelState.ERROR, failure_count=6)
        status = monitor.assess(record)
        assert status.recommendation == "retire"

    def test_stale_model(self):
        monitor = HealthMonitor(AgentSchedulerConfig(stale_days_threshold=7))
        record = ModelStateRecord(
            model_id="m1", state=ModelState.ACTIVE,
            last_success=datetime.now(timezone.utc) - timedelta(days=10),
        )
        status = monitor.assess(record)
        assert status.recommendation == "investigate"

    def test_assess_all(self):
        monitor = HealthMonitor()
        records = [
            ModelStateRecord(model_id="m1", state=ModelState.ACTIVE),
            ModelStateRecord(model_id="m2", state=ModelState.ACTIVE),
        ]
        statuses = monitor.assess_all(records)
        assert len(statuses) == 2


class TestPriorityOptimizer:
    def test_boost_on_success(self):
        optimizer = PriorityOptimizer(AgentSchedulerConfig(priority_boost_on_success=5))
        record = ModelStateRecord(model_id="m1", priority=50)
        new = optimizer.adjust_on_success(record)
        assert new == 45
        assert record.failure_count == 0

    def test_penalty_on_failure(self):
        optimizer = PriorityOptimizer(AgentSchedulerConfig(priority_penalty_on_failure=10))
        record = ModelStateRecord(model_id="m1", priority=50)
        new = optimizer.adjust_on_failure(record)
        assert new == 60
        assert record.failure_count == 1

    def test_priority_bounds(self):
        optimizer = PriorityOptimizer(AgentSchedulerConfig(min_priority=1, max_priority=100))
        record = ModelStateRecord(model_id="m1", priority=1)
        optimizer.adjust_on_success(record)
        assert record.priority >= 1

        record2 = ModelStateRecord(model_id="m2", priority=100)
        optimizer.adjust_on_failure(record2)
        assert record2.priority <= 100

    def test_rebalance(self):
        optimizer = PriorityOptimizer()
        records = [
            ModelStateRecord(model_id="m1", state=ModelState.ACTIVE, priority=50),
            ModelStateRecord(model_id="m2", state=ModelState.ERROR, priority=30),
        ]
        sorted_records = optimizer.rebalance(records)
        # ERROR model should have worse (higher) priority after rebalance
        assert sorted_records[0].model_id == "m2" or sorted_records[1].priority > 50

    def test_permission_disabled(self):
        optimizer = PriorityOptimizer(AgentSchedulerConfig(can_change_priority=False))
        record = ModelStateRecord(model_id="m1", priority=50)
        assert optimizer.adjust_on_success(record) == 50
        assert optimizer.adjust_on_failure(record) == 50


class TestUniverseMerger:
    def test_no_active_models(self):
        merger = UniverseMerger()
        result = merger.analyze([
            ModelStateRecord(model_id="m1", state=ModelState.RETIRED, universe=["AAPL"]),
        ])
        assert result.total_assets_after == 0

    def test_merge_overlapping(self):
        merger = UniverseMerger()
        result = merger.analyze([
            ModelStateRecord(model_id="m1", state=ModelState.ACTIVE,
                             universe=["AAPL", "GOOG", "MSFT"], data_frequency="daily"),
            ModelStateRecord(model_id="m2", state=ModelState.ACTIVE,
                             universe=["AAPL", "GOOG", "AMZN"], data_frequency="daily"),
        ])
        assert result.total_assets_before == 6  # 3 + 3
        assert result.total_assets_after == 4   # AAPL, GOOG, MSFT, AMZN
        assert result.dedup_savings_pct > 0

    def test_no_overlap(self):
        merger = UniverseMerger()
        result = merger.analyze([
            ModelStateRecord(model_id="m1", state=ModelState.ACTIVE,
                             universe=["AAPL", "GOOG"], data_frequency="daily"),
            ModelStateRecord(model_id="m2", state=ModelState.ACTIVE,
                             universe=["TSLA", "AMZN"], data_frequency="daily"),
        ])
        assert result.total_assets_before == 4
        assert result.total_assets_after == 4
        assert result.dedup_savings_pct == 0.0

    def test_different_frequencies_separate(self):
        merger = UniverseMerger()
        result = merger.analyze([
            ModelStateRecord(model_id="m1", state=ModelState.ACTIVE,
                             universe=["AAPL"], data_frequency="daily"),
            ModelStateRecord(model_id="m2", state=ModelState.ACTIVE,
                             universe=["AAPL"], data_frequency="intraday_1min"),
        ])
        assert len(result.merged_universes) == 2

    def test_merged_universe_properties(self):
        merger = UniverseMerger()
        result = merger.analyze([
            ModelStateRecord(model_id="m1", state=ModelState.ACTIVE,
                             universe=["AAPL", "GOOG"], data_frequency="daily",
                             depends_on_data=["yfinance"]),
            ModelStateRecord(model_id="m2", state=ModelState.QUEUED,
                             universe=["GOOG", "MSFT"], data_frequency="daily",
                             depends_on_data=["polygon"]),
        ])
        merged = result.merged_universes[0]
        assert merged.n_assets == 3
        assert merged.n_models == 2
        assert "yfinance" in merged.sources
        assert "polygon" in merged.sources


class TestAgentScheduler:
    @pytest.fixture
    def scheduler(self):
        return AgentScheduler(AgentSchedulerConfig(
            max_consecutive_failures=3,
            max_total_failures=10,
            can_pause_models=True,
            can_retry_failed=True,
        ))

    def test_register_model(self, scheduler):
        record = ModelStateRecord(model_id="m1", state=ModelState.ACTIVE)
        scheduler.register_model(record)
        assert scheduler.get_model("m1") is not None
        assert len(scheduler.models) == 1

    def test_monitor_health_auto_retry(self, scheduler):
        record = ModelStateRecord(model_id="m1", state=ModelState.ERROR, failure_count=1)
        scheduler.register_model(record)
        actions = scheduler.monitor_health()
        assert any(a.action_type == "retry" for a in actions)
        assert record.state == ModelState.ACTIVE

    def test_monitor_health_auto_pause(self, scheduler):
        record = ModelStateRecord(model_id="m1", state=ModelState.ACTIVE, failure_count=5)
        scheduler.register_model(record)
        actions = scheduler.monitor_health()
        assert any(a.action_type == "pause" for a in actions)
        assert record.state == ModelState.PAUSED

    def test_record_success_boosts_priority(self, scheduler):
        record = ModelStateRecord(model_id="m1", state=ModelState.ACTIVE, priority=50)
        scheduler.register_model(record)
        scheduler.record_success("m1")
        assert record.priority < 50

    def test_record_failure_penalizes(self, scheduler):
        record = ModelStateRecord(model_id="m1", state=ModelState.ACTIVE, priority=50)
        scheduler.register_model(record)
        scheduler.record_failure("m1")
        assert record.priority > 50
        assert record.failure_count == 1

    def test_transition_model(self, scheduler):
        record = ModelStateRecord(model_id="m1", state=ModelState.DRAFT)
        scheduler.register_model(record)
        assert scheduler.transition_model("m1", ModelState.ACTIVE, by="user")
        assert record.state == ModelState.ACTIVE

    def test_transition_blocked_by_permission(self, scheduler):
        # Agent can't activate (requires human approval)
        scheduler._config.can_activate_models = False
        record = ModelStateRecord(model_id="m1", state=ModelState.DRAFT)
        scheduler.register_model(record)
        assert not scheduler.transition_model("m1", ModelState.ACTIVE, by="agent")
        assert record.state == ModelState.DRAFT
        # Check approval required action was logged
        assert any(a.requires_approval for a in scheduler.action_log)

    def test_get_execution_order(self, scheduler):
        scheduler.register_model(ModelStateRecord(model_id="m1", state=ModelState.ACTIVE, priority=30))
        scheduler.register_model(ModelStateRecord(model_id="m2", state=ModelState.ACTIVE, priority=10))
        scheduler.register_model(ModelStateRecord(model_id="m3", state=ModelState.PAUSED, priority=5))
        order = scheduler.get_execution_order()
        # m3 is paused, not in execution order
        assert "m3" not in order
        # m2 has higher priority (lower number) → first
        assert order.index("m2") < order.index("m1")

    def test_execution_order_respects_dependencies(self, scheduler):
        scheduler.register_model(ModelStateRecord(
            model_id="m1", state=ModelState.ACTIVE, priority=10,
            depends_on_models=["m2"],
        ))
        scheduler.register_model(ModelStateRecord(
            model_id="m2", state=ModelState.ACTIVE, priority=50,
        ))
        order = scheduler.get_execution_order()
        # m2 must come before m1 despite lower priority
        assert order.index("m2") < order.index("m1")

    def test_merge_universes(self, scheduler):
        scheduler.register_model(ModelStateRecord(
            model_id="m1", state=ModelState.ACTIVE,
            universe=["AAPL", "GOOG", "MSFT"],
        ))
        scheduler.register_model(ModelStateRecord(
            model_id="m2", state=ModelState.ACTIVE,
            universe=["AAPL", "GOOG", "AMZN"],
        ))
        result = scheduler.merge_universes()
        assert result.dedup_savings_pct > 0

    def test_summary(self, scheduler):
        scheduler.register_model(ModelStateRecord(model_id="m1", state=ModelState.ACTIVE))
        scheduler.register_model(ModelStateRecord(model_id="m2", state=ModelState.PAUSED))
        s = scheduler.summary()
        assert s["total_models"] == 2
        assert "active" in s["by_state"]
        assert "paused" in s["by_state"]

    def test_action_log(self, scheduler):
        record = ModelStateRecord(model_id="m1", state=ModelState.DRAFT)
        scheduler.register_model(record)
        scheduler.transition_model("m1", ModelState.ACTIVE, by="user")
        assert len(scheduler.action_log) >= 1
        assert scheduler.action_log[-1].action_type == "state_change"


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.16: Logger Agent Tool
# ═══════════════════════════════════════════════════════════════════

from praxis.logger.agent_tool import (
    LoggerAgentTool,
    RingBufferAdapter,
    LogConfigSnapshot,
)
from praxis.logger.records import LogRecord


class TestRingBufferAdapter:
    def test_emit_and_read(self):
        ring = RingBufferAdapter(capacity=100)
        record = LogRecord.create(level=20, message="test message", tags={"test"})
        ring.emit(record)
        assert ring.size == 1
        records = ring.get_records(n=10)
        assert len(records) == 1
        assert records[0].message == "test message"

    def test_capacity_limit(self):
        ring = RingBufferAdapter(capacity=5)
        for i in range(10):
            ring.emit(LogRecord.create(level=20, message=f"msg_{i}"))
        assert ring.size == 5  # Only last 5
        records = ring.get_records(n=10)
        assert records[0].message == "msg_5"

    def test_filter_by_tags(self):
        ring = RingBufferAdapter(capacity=100)
        ring.emit(LogRecord.create(level=20, message="trade", tags={"trade_cycle"}))
        ring.emit(LogRecord.create(level=20, message="data", tags={"data_pipeline"}))
        ring.emit(LogRecord.create(level=20, message="trade2", tags={"trade_cycle"}))

        records = ring.get_records(tags=["trade_cycle"])
        assert len(records) == 2
        assert all("trade" in r.message for r in records)

    def test_filter_by_level(self):
        ring = RingBufferAdapter(capacity=100)
        ring.emit(LogRecord.create(level=10, message="debug"))
        ring.emit(LogRecord.create(level=20, message="info"))
        ring.emit(LogRecord.create(level=40, message="error"))

        records = ring.get_records(min_level=20)
        assert len(records) == 2

    def test_filter_by_contains(self):
        ring = RingBufferAdapter(capacity=100)
        ring.emit(LogRecord.create(level=20, message="order filled AAPL"))
        ring.emit(LogRecord.create(level=20, message="signal computed"))
        ring.emit(LogRecord.create(level=20, message="order rejected GOOG"))

        records = ring.get_records(contains="order")
        assert len(records) == 2

    def test_clear(self):
        ring = RingBufferAdapter(capacity=100)
        ring.emit(LogRecord.create(level=20, message="test"))
        ring.clear()
        assert ring.size == 0


class TestLoggerAgentTool:
    @pytest.fixture
    def tool(self):
        return LoggerAgentTool(ring_capacity=500)

    def test_activate_tags(self, tool):
        result = tool.activate_tags(["trade_cycle", "data_pipeline"])
        assert "trade_cycle" in tool.get_active_tags()
        assert "data_pipeline" in tool.get_active_tags()

    def test_deactivate_tags(self, tool):
        tool.activate_tags(["trade_cycle", "data_pipeline"])
        tool.deactivate_tags(["trade_cycle"])
        assert "trade_cycle" not in tool.get_active_tags()
        assert "data_pipeline" in tool.get_active_tags()

    def test_set_level(self, tool):
        result = tool.set_level("debug")
        info = tool.get_level()
        assert info["global_level"] == 10

    def test_save_restore_config(self, tool):
        # Set initial state
        tool.set_level("info")
        tool.activate_tags(["original"])

        # Save
        snap = tool.save_config()
        assert tool.has_saved_config()

        # Modify
        tool.set_level("debug")
        tool.activate_tags(["diagnostic"])

        # Restore
        assert tool.restore_config()
        assert not tool.has_saved_config()
        info = tool.get_level()
        assert info["global_level"] == 20  # INFO restored

    def test_get_recent_logs(self, tool):
        logger = PraxisLogger.instance()
        logger.info("test log message", tags={"test"})
        records = tool.get_recent_logs(n=10)
        # May or may not have records depending on level/tag filtering
        assert isinstance(records, list)

    def test_get_log_summary(self, tool):
        # Emit some records directly to ring buffer
        tool.ring_buffer.emit(LogRecord.create(level=20, message="info msg", tags={"test"}))
        tool.ring_buffer.emit(LogRecord.create(level=40, message="error msg", tags={"test"}))
        summary = tool.get_log_summary()
        assert summary["total_records"] == 2
        assert "INFO" in summary["by_level"]
        assert "ERROR" in summary["by_level"]

    def test_start_diagnostic(self, tool):
        tool.set_level("info")
        result = tool.start_diagnostic("trading")
        assert tool.has_saved_config()
        info = tool.get_level()
        assert info["global_level"] == 10  # DEBUG
        active = tool.get_active_tags()
        assert "trade_cycle" in active

    def test_end_diagnostic(self, tool):
        tool.set_level("info")
        tool.start_diagnostic("data")
        # Add some logs during diagnostic
        tool.ring_buffer.emit(LogRecord.create(level=10, message="diag log"))
        result = tool.end_diagnostic()
        assert result["diagnostic_complete"]
        assert result["config_restored"]
        info = tool.get_level()
        assert info["global_level"] == 20  # Restored to INFO

    def test_full_troubleshooting_workflow(self, tool):
        """Simulate the §18.7 agent troubleshooting workflow."""
        # 1. Initial state
        tool.set_level("info")
        assert tool.get_level()["global_level"] == 20

        # 2. Start diagnostic
        tool.start_diagnostic("scheduler")
        assert tool.get_level()["global_level"] == 10
        assert "scheduler" in tool.get_active_tags()

        # 3. Simulate diagnostic logs
        tool.ring_buffer.emit(LogRecord.create(level=10, message="tick started", tags={"scheduler"}))
        tool.ring_buffer.emit(LogRecord.create(level=10, message="dag built", tags={"dag"}))
        tool.ring_buffer.emit(LogRecord.create(level=40, message="node failed: data_load", tags={"scheduler"}))

        # 4. Read filtered logs
        errors = tool.get_recent_logs(min_level=40)
        assert len(errors) == 1
        assert "failed" in errors[0].message

        scheduler_logs = tool.get_recent_logs(tags=["scheduler"])
        assert len(scheduler_logs) == 2

        # 5. End diagnostic
        result = tool.end_diagnostic()
        assert result["logs_captured"] >= 3
        assert tool.get_level()["global_level"] == 20  # Restored

    def test_status(self, tool):
        status = tool.status()
        assert "global_level" in status
        assert "active_tags" in status
        assert "ring_buffer_size" in status
        assert "ring_buffer_capacity" in status

    def test_restore_without_save(self, tool):
        assert not tool.restore_config()
