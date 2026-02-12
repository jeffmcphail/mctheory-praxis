"""
Tests for Diagnostic Logger (§18).

Covers:
- LogRecord and LogLevel
- Formatters (compact, detailed, JSON)
- Adapters (terminal, file, database, agent)
- Routing matrix
- PraxisLogger singleton
- Tag system (scope tags, trace tags, auto-activate)
- Configuration from dict
- Runtime reconfiguration
"""

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

import pytest

from praxis.logger.records import LogRecord, LogLevel, level_name
from praxis.logger.formatters import CompactFormatter, DetailedFormatter, JsonFormatter
from praxis.logger.adapters import (
    TerminalAdapter,
    FileAdapter,
    DatabaseAdapter,
    AgentAdapter,
)
from praxis.logger.routing import RoutingMatrix, RoutingRule
from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  LogLevel
# ═══════════════════════════════════════════════════════════════════

class TestLogLevel:
    def test_standard_levels_ordered(self):
        assert LogLevel.TRACE < LogLevel.DEBUG < LogLevel.INFO < LogLevel.WARNING
        assert LogLevel.WARNING < LogLevel.ERROR < LogLevel.CRITICAL

    def test_custom_levels_interleaved(self):
        """§18.9: Custom levels slot between standard ones."""
        assert LogLevel.TRACE < LogLevel.DEBUG
        assert LogLevel.VERBOSE > LogLevel.DEBUG
        assert LogLevel.VERBOSE < LogLevel.INFO
        assert LogLevel.NOTICE > LogLevel.INFO
        assert LogLevel.NOTICE < LogLevel.WARNING
        assert LogLevel.ALERT > LogLevel.ERROR
        assert LogLevel.ALERT < LogLevel.CRITICAL

    def test_python_compatible_values(self):
        """§18.2: Python-compatible numeric values."""
        assert LogLevel.DEBUG == 10
        assert LogLevel.INFO == 20
        assert LogLevel.WARNING == 30
        assert LogLevel.ERROR == 40
        assert LogLevel.CRITICAL == 50

    def test_from_name_case_insensitive(self):
        assert LogLevel.from_name("debug") == LogLevel.DEBUG
        assert LogLevel.from_name("DEBUG") == LogLevel.DEBUG
        assert LogLevel.from_name("Info") == LogLevel.INFO

    def test_from_name_invalid(self):
        with pytest.raises(ValueError, match="Unknown log level"):
            LogLevel.from_name("nonexistent")

    def test_from_value_int(self):
        assert LogLevel.from_value(20) == LogLevel.INFO
        assert LogLevel.from_value(50) == LogLevel.CRITICAL

    def test_from_value_string(self):
        assert LogLevel.from_value("warning") == LogLevel.WARNING

    def test_level_name_function(self):
        assert level_name(20) == "INFO"
        assert level_name(50) == "CRITICAL"
        assert level_name(999) == "999"  # Fallback for unknown


# ═══════════════════════════════════════════════════════════════════
#  LogRecord
# ═══════════════════════════════════════════════════════════════════

class TestLogRecord:
    def test_create_basic(self):
        record = LogRecord.create(LogLevel.INFO, "test message")
        assert record.level == 20
        assert record.level_name == "INFO"
        assert record.message == "test message"
        assert record.tags == frozenset()
        assert record.source == "system"

    def test_create_with_tags(self):
        record = LogRecord.create(
            LogLevel.DEBUG, "matched",
            tags={"datastore.security_master", "data_pipeline"}
        )
        assert "datastore.security_master" in record.tags
        assert "data_pipeline" in record.tags

    def test_create_with_context(self):
        record = LogRecord.create(
            LogLevel.DEBUG, "Z-score computed",
            tags={"compute.signals"},
            module="signals",
            function="zscore",
            zscore=2.1,
        )
        assert record.context["module"] == "signals"
        assert record.context["zscore"] == 2.1

    def test_immutable(self):
        record = LogRecord.create(LogLevel.INFO, "test")
        with pytest.raises(AttributeError):
            record.message = "changed"

    def test_timestamp_is_utc(self):
        record = LogRecord.create(LogLevel.INFO, "test")
        assert record.timestamp.tzinfo is not None

    def test_source_and_session(self):
        record = LogRecord.create(
            LogLevel.INFO, "test",
            source="agent",
            session_id="backtest-001"
        )
        assert record.source == "agent"
        assert record.session_id == "backtest-001"


# ═══════════════════════════════════════════════════════════════════
#  Formatters
# ═══════════════════════════════════════════════════════════════════

class TestCompactFormatter:
    def test_basic_format(self):
        ts = datetime(2026, 2, 12, 14, 32, 5, tzinfo=timezone.utc)
        record = LogRecord(
            timestamp=ts, level=20, level_name="INFO",
            message="Security matched"
        )
        result = CompactFormatter().format(record)
        assert "14:32:05" in result
        assert "INFO" in result
        assert "Security matched" in result

    def test_level_right_aligned(self):
        record = LogRecord.create(LogLevel.DEBUG, "test")
        result = CompactFormatter().format(record)
        # DEBUG should be right-aligned in 8 chars
        assert "   DEBUG" in result


class TestDetailedFormatter:
    def test_with_tags_and_context(self):
        ts = datetime(2026, 2, 12, 14, 32, 5, 123456, tzinfo=timezone.utc)
        record = LogRecord(
            timestamp=ts, level=20, level_name="INFO",
            message="Fetching GLD",
            tags=frozenset({"datastore.loader", "data_pipeline"}),
            context={"module": "loader", "function": "fetch"},
        )
        result = DetailedFormatter().format(record)
        assert "2026-02-12 14:32:05.123456" in result
        assert "data_pipeline" in result
        assert "datastore.loader" in result
        assert "loader.fetch:" in result
        assert "Fetching GLD" in result

    def test_extra_context_appended(self):
        record = LogRecord.create(
            LogLevel.DEBUG, "computed",
            tags={"compute"},
            module="signals",
            zscore=2.1234,
        )
        result = DetailedFormatter().format(record)
        assert "zscore=2.1234" in result

    def test_no_tags(self):
        record = LogRecord.create(LogLevel.INFO, "plain message")
        result = DetailedFormatter().format(record)
        assert "[-]" in result  # empty tags placeholder


class TestJsonFormatter:
    def test_valid_json(self):
        record = LogRecord.create(
            LogLevel.INFO, "test",
            tags={"trade_cycle"},
            module="execution",
        )
        result = JsonFormatter().format(record)
        parsed = json.loads(result)
        assert parsed["level"] == 20
        assert parsed["level_name"] == "INFO"
        assert parsed["message"] == "test"
        assert "trade_cycle" in parsed["tags"]
        assert parsed["context"]["module"] == "execution"

    def test_serializes_complex_types(self):
        record = LogRecord.create(
            LogLevel.INFO, "test",
            data=[1, 2, 3],
            nested={"a": {"b": 1}},
        )
        result = JsonFormatter().format(record)
        parsed = json.loads(result)
        assert parsed["context"]["data"] == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════
#  Adapters
# ═══════════════════════════════════════════════════════════════════

class TestTerminalAdapter:
    def test_emit_to_stdout(self, capsys):
        adapter = TerminalAdapter(color=False)
        record = LogRecord.create(LogLevel.INFO, "hello terminal")
        adapter.emit(record)
        captured = capsys.readouterr()
        assert "hello terminal" in captured.out

    def test_error_to_stderr(self, capsys):
        adapter = TerminalAdapter(color=False)
        record = LogRecord.create(LogLevel.ERROR, "bad thing")
        adapter.emit(record)
        captured = capsys.readouterr()
        assert "bad thing" in captured.err

    def test_color_codes_applied(self):
        adapter = TerminalAdapter(color=True)
        record = LogRecord.create(LogLevel.WARNING, "caution")
        with patch("sys.stdout", new_callable=io.StringIO) as mock:
            adapter.emit(record)
            output = mock.getvalue()
            assert "\033[33m" in output  # Yellow for WARNING
            assert "\033[0m" in output   # Reset


class TestFileAdapter:
    def test_writes_to_file(self, tmp_path):
        log_path = tmp_path / "test.log"
        adapter = FileAdapter(path=log_path, rotation="none")
        record = LogRecord.create(LogLevel.INFO, "file test")
        adapter.emit(record)
        adapter.flush()
        adapter.close()
        content = log_path.read_text()
        assert "file test" in content

    def test_daily_rotation(self, tmp_path):
        log_path = tmp_path / "praxis.log"
        adapter = FileAdapter(path=log_path, rotation="daily")

        ts1 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 1, 16, 10, 0, 0, tzinfo=timezone.utc)

        r1 = LogRecord(
            timestamp=ts1, level=20, level_name="INFO", message="day1"
        )
        r2 = LogRecord(
            timestamp=ts2, level=20, level_name="INFO", message="day2"
        )

        adapter.emit(r1)
        adapter.flush()
        adapter.emit(r2)
        adapter.flush()
        adapter.close()

        files = list(tmp_path.glob("praxis_*.log"))
        assert len(files) == 2

    def test_creates_directory(self, tmp_path):
        log_path = tmp_path / "nested" / "dir" / "test.log"
        adapter = FileAdapter(path=log_path, rotation="none")
        record = LogRecord.create(LogLevel.INFO, "nested test")
        adapter.emit(record)
        adapter.close()
        assert log_path.exists()


class TestDatabaseAdapter:
    def test_buffers_without_connection(self):
        adapter = DatabaseAdapter(buffer_size=100)
        record = LogRecord.create(LogLevel.INFO, "buffered")
        adapter.emit(record)
        assert adapter.buffered_count == 1

    def test_buffer_not_lost_without_connection(self):
        adapter = DatabaseAdapter(buffer_size=100)
        for i in range(5):
            adapter.emit(LogRecord.create(LogLevel.INFO, f"msg {i}"))
        adapter.flush()  # No connection → buffer stays
        assert adapter.buffered_count == 5

    def test_backtest_throttle(self):
        adapter = DatabaseAdapter(backtest_throttle=30)
        assert adapter.effective_min_level == 20  # Normal: INFO
        adapter.set_backtest_mode(True)
        assert adapter.effective_min_level == 30  # Backtest: WARNING


class TestAgentAdapter:
    def test_ring_buffer(self):
        adapter = AgentAdapter(ring_buffer_size=5)
        for i in range(10):
            adapter.emit(LogRecord.create(LogLevel.INFO, f"msg {i}"))
        assert adapter.count == 5
        recent = adapter.get_recent(n=3)
        assert len(recent) == 3
        assert recent[-1].message == "msg 9"

    def test_filter_by_tags(self):
        adapter = AgentAdapter(ring_buffer_size=100)
        adapter.emit(LogRecord.create(LogLevel.INFO, "trade", tags={"trade_cycle"}))
        adapter.emit(LogRecord.create(LogLevel.INFO, "data", tags={"data_pipeline"}))
        adapter.emit(LogRecord.create(LogLevel.INFO, "both", tags={"trade_cycle", "data_pipeline"}))

        trade_only = adapter.get_recent(tags={"trade_cycle"})
        assert len(trade_only) == 2  # "trade" and "both"

    def test_clear(self):
        adapter = AgentAdapter()
        adapter.emit(LogRecord.create(LogLevel.INFO, "test"))
        assert adapter.count == 1
        adapter.clear()
        assert adapter.count == 0


# ═══════════════════════════════════════════════════════════════════
#  Routing Matrix
# ═══════════════════════════════════════════════════════════════════

class TestRoutingMatrix:
    def test_default_routes_by_adapter_min_level(self):
        routing = RoutingMatrix()
        record = LogRecord.create(LogLevel.INFO, "test")  # level 20
        assert routing.should_route(record, "terminal", 20)  # INFO >= INFO
        assert not routing.should_route(record, "terminal", 30)  # INFO < WARNING

    def test_critical_override(self):
        """§18.6: CRITICAL goes everywhere regardless of adapter min_level."""
        routing = RoutingMatrix(critical_override=True)
        record = LogRecord.create(LogLevel.CRITICAL, "crash")
        assert routing.should_route(record, "terminal", 50)  # CRITICAL >= CRITICAL
        assert routing.should_route(record, "terminal", 999)  # Override!

    def test_critical_override_disabled(self):
        routing = RoutingMatrix(critical_override=False)
        record = LogRecord.create(LogLevel.CRITICAL, "crash")
        assert not routing.should_route(record, "terminal", 999)  # No override

    def test_tag_routes(self):
        """§18.6: Tag-specific routing to specific adapters."""
        routing = RoutingMatrix()
        routing.add_tag_route("trade_cycle", ["terminal", "database", "agent"])

        record = LogRecord.create(
            LogLevel.DEBUG, "signal fired",  # DEBUG (10) — below most adapter min_levels
            tags={"compute.signals", "trade_cycle"}
        )

        # trade_cycle tag routes to terminal even though level is DEBUG
        assert routing.should_route(record, "terminal", 20)  # Would fail without tag route
        assert routing.should_route(record, "database", 20)
        assert routing.should_route(record, "agent", 20)
        # logfile not in tag route, and DEBUG < logfile min_level
        assert not routing.should_route(record, "logfile", 20)

    def test_get_target_adapters(self):
        routing = RoutingMatrix()
        routing.add_tag_route("trade_cycle", ["terminal", "database"])

        adapters = {"terminal": 20, "logfile": 10, "database": 20}
        record = LogRecord.create(LogLevel.DEBUG, "test", tags={"trade_cycle"})

        targets = routing.get_target_adapters(record, adapters)
        assert "terminal" in targets  # Via tag route
        assert "database" in targets  # Via tag route
        assert "logfile" in targets   # DEBUG(10) >= logfile min(10)

    def test_describe(self):
        routing = RoutingMatrix()
        routing.add_tag_route("trade_cycle", ["terminal", "database"])
        desc = routing.describe()
        assert desc["critical_override"] is True
        assert "trade_cycle" in desc["tag_routes"]


# ═══════════════════════════════════════════════════════════════════
#  PraxisLogger Singleton
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_logger():
    """Reset singleton before and after each test."""
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


class TestPraxisLoggerSingleton:
    def test_singleton_identity(self):
        a = PraxisLogger.instance()
        b = PraxisLogger.instance()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = PraxisLogger.instance()
        PraxisLogger.reset()
        b = PraxisLogger.instance()
        assert a is not b

    def test_thread_safe_singleton(self):
        """Verify singleton is thread-safe."""
        instances = []

        def get_instance():
            instances.append(PraxisLogger.instance())

        threads = [threading.Thread(target=get_instance) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)


class TestPraxisLoggerEmission:
    def test_nanosecond_gate_no_adapters(self):
        """With no adapters, _should_emit returns False immediately."""
        log = PraxisLogger.instance()
        assert not log._should_emit(LogLevel.CRITICAL, set())

    def test_global_level_gate(self):
        log = PraxisLogger.instance()
        log.add_adapter(TerminalAdapter(color=False))
        log.current_level = LogLevel.WARNING
        assert not log._should_emit(LogLevel.INFO, set())
        assert log._should_emit(LogLevel.WARNING, set())
        assert log._should_emit(LogLevel.ERROR, set())

    def test_active_tag_bypasses_level(self):
        """§18.3: Explicitly activated tag emits regardless of global level."""
        log = PraxisLogger.instance()
        log.add_adapter(TerminalAdapter(color=False))
        log.current_level = LogLevel.ERROR  # Only ERROR+
        log.activate_tag("trade_cycle")

        # DEBUG with activated tag should still emit
        assert log._should_emit(LogLevel.DEBUG, {"trade_cycle"})
        # DEBUG without the tag should not
        assert not log._should_emit(LogLevel.DEBUG, {"something_else"})

    def test_tag_auto_activate_threshold(self):
        """§18.4: Tag auto-activates when global level reaches its threshold."""
        log = PraxisLogger.instance()
        log.add_adapter(TerminalAdapter(color=False))
        log.set_tag_level("cpo_cycle", LogLevel.DEBUG)  # Activates at DEBUG

        log.current_level = LogLevel.INFO  # Global is INFO
        # cpo_cycle threshold is DEBUG(10), current_level is INFO(20) >= 10
        assert log._should_emit(LogLevel.TRACE, {"cpo_cycle"})  # Below INFO but tag auto-activates

        log.current_level = LogLevel.WARNING  # Raise global
        # cpo_cycle threshold 10 <= WARNING(30) → still auto-active
        assert log._should_emit(LogLevel.TRACE, {"cpo_cycle"})

    def test_emit_routes_to_adapter(self, capsys):
        log = PraxisLogger.instance()
        log.add_adapter(TerminalAdapter(color=False))
        log.current_level = LogLevel.INFO
        log.info("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_emit_with_context(self, capsys):
        log = PraxisLogger.instance()
        adapter = AgentAdapter()
        log.add_adapter(adapter)
        log.current_level = LogLevel.DEBUG
        log.debug("computed", tags={"compute"}, module="signals", zscore=2.1)
        records = adapter.get_recent()
        assert len(records) == 1
        assert records[0].context["zscore"] == 2.1

    def test_adapter_failure_does_not_crash(self):
        """Adapter exceptions are swallowed — logging must never crash the caller."""
        log = PraxisLogger.instance()

        class BrokenAdapter(TerminalAdapter):
            def emit(self, record):
                raise RuntimeError("adapter exploded")

        log.add_adapter(BrokenAdapter(name="broken", color=False))
        log.current_level = LogLevel.INFO
        # This should NOT raise
        log.info("this should not crash")


class TestPraxisLoggerConvenienceMethods:
    def test_all_levels_emit(self):
        log = PraxisLogger.instance()
        agent = AgentAdapter(min_level=1)
        log.add_adapter(agent)
        log.current_level = LogLevel.TRACE

        log.trace("t")
        log.debug("d")
        log.verbose("v")
        log.info("i")
        log.notice("n")
        log.warning("w")
        log.error("e")
        log.alert("a")
        log.critical("c")

        records = agent.get_recent(n=100)
        assert len(records) == 9
        levels = [r.level for r in records]
        assert levels == [5, 10, 15, 20, 25, 30, 40, 45, 50]


class TestPraxisLoggerConfiguration:
    def test_configure_from_dict(self):
        """§18.6: Full YAML-style configuration."""
        log = PraxisLogger.instance()
        log.configure({
            "current_level": "INFO",
            "adapters": {
                "terminal": {"type": "terminal", "min_level": 20, "color": False},
                "logfile": {"type": "file", "min_level": 10, "path": "/tmp/test_praxis.log"},
                "agent": {"type": "agent", "min_level": 10, "ring_buffer_size": 500},
            },
            "tag_levels": {
                "trade_cycle": 20,
                "cpo_cycle": 10,
            },
            "routing": {
                "critical_override": True,
                "tag_routes": {
                    "trade_cycle": ["terminal", "agent"],
                },
            },
        })

        assert log.current_level == 20
        assert "terminal" in log._adapters
        assert "logfile" in log._adapters
        assert "agent" in log._adapters
        assert log._tag_levels["trade_cycle"] == 20
        assert log._tag_levels["cpo_cycle"] == 10
        assert log._routing.critical_override is True

    def test_configure_defaults(self, capsys):
        log = PraxisLogger.instance()
        log.configure_defaults()
        log.info("defaults work")
        captured = capsys.readouterr()
        assert "defaults work" in captured.out


class TestPraxisLoggerRuntimeReconfiguration:
    def test_activate_deactivate_tag(self):
        log = PraxisLogger.instance()
        log.activate_tag("trade_cycle")
        assert "trade_cycle" in log.active_tags
        log.deactivate_tag("trade_cycle")
        assert "trade_cycle" not in log.active_tags

    def test_set_level_by_name(self):
        log = PraxisLogger.instance()
        log.current_level = "debug"
        assert log.current_level == 10

    def test_set_adapter_level(self):
        log = PraxisLogger.instance()
        log.add_adapter(TerminalAdapter(name="terminal", min_level=20, color=False))
        log.set_adapter_level("terminal", "debug")
        assert log._adapters["terminal"].min_level == 10

    def test_set_adapter_level_unknown(self):
        log = PraxisLogger.instance()
        with pytest.raises(ValueError, match="Unknown adapter"):
            log.set_adapter_level("nonexistent", 10)


class TestPraxisLoggerStatus:
    def test_status_structure(self):
        log = PraxisLogger.instance()
        log.configure({
            "current_level": "INFO",
            "adapters": {
                "terminal": {"type": "terminal", "min_level": 20, "color": False},
            },
            "tag_levels": {"trade_cycle": 20},
        })
        log.activate_tag("compute.signals")

        status = log.status()
        assert status["current_level"] == 20
        assert status["current_level_name"] == "INFO"
        assert "compute.signals" in status["active_tags"]
        assert "terminal" in status["adapters"]
        assert status["adapters"]["terminal"]["type"] == "TerminalAdapter"
        assert "trade_cycle" in status["tag_levels"]


class TestPraxisLoggerSessionId:
    def test_session_id_propagated(self):
        log = PraxisLogger.instance()
        agent = AgentAdapter()
        log.add_adapter(agent)
        log.current_level = LogLevel.INFO
        log.session_id = "backtest-42"
        log.info("session test")
        records = agent.get_recent()
        assert records[0].session_id == "backtest-42"


class TestPraxisLoggerTagRouting:
    def test_trade_cycle_trace_tag(self):
        """
        §18.3: Activating trade_cycle illuminates the entire path
        from signal to settlement.
        """
        log = PraxisLogger.instance()
        agent = AgentAdapter(min_level=1)
        log.add_adapter(agent)
        log.current_level = LogLevel.INFO
        log.activate_tag("trade_cycle")

        # These are all DEBUG but should emit because trade_cycle is active
        log.debug("Entry signal fired", tags={"compute.signals", "trade_cycle"})
        log.debug("Position sized", tags={"compute.sizing", "trade_cycle"})
        log.debug("Order submitted", tags={"execution.order", "trade_cycle"})
        log.debug("Fill confirmed", tags={"execution.fill", "trade_cycle"})
        log.debug("P&L attributed", tags={"compute.pnl", "trade_cycle"})

        records = agent.get_recent()
        assert len(records) == 5
        messages = [r.message for r in records]
        assert "Entry signal fired" in messages
        assert "P&L attributed" in messages

    def test_scope_tag_without_activation(self):
        """Scope tags without activation respect global level."""
        log = PraxisLogger.instance()
        agent = AgentAdapter(min_level=1)
        log.add_adapter(agent)
        log.current_level = LogLevel.INFO

        log.debug("deep detail", tags={"compute.signals.zscore_spread"})
        records = agent.get_recent()
        assert len(records) == 0  # DEBUG below INFO, tag not active


class TestPraxisLoggerFlushAndClose:
    def test_flush_all_adapters(self, tmp_path):
        log = PraxisLogger.instance()
        file_path = tmp_path / "flush_test.log"
        log.add_adapter(FileAdapter(path=file_path, rotation="none"))
        log.current_level = LogLevel.INFO
        log.info("flush me")
        log.flush()
        content = file_path.read_text()
        assert "flush me" in content

    def test_close_cleans_up(self, tmp_path):
        log = PraxisLogger.instance()
        file_path = tmp_path / "close_test.log"
        adapter = FileAdapter(path=file_path, rotation="none")
        log.add_adapter(adapter)
        log.current_level = LogLevel.INFO
        log.info("close me")
        log.close()
        assert adapter._file is None


class TestPraxisLoggerRemoveAdapter:
    def test_remove_existing(self):
        log = PraxisLogger.instance()
        log.add_adapter(TerminalAdapter(color=False))
        removed = log.remove_adapter("terminal")
        assert removed is not None
        assert "terminal" not in log._adapters

    def test_remove_nonexistent(self):
        log = PraxisLogger.instance()
        removed = log.remove_adapter("ghost")
        assert removed is None
