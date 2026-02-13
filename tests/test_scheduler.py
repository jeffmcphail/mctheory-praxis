"""
Tests for Basic Explicit Scheduler (Phase 2.13).

Covers:
- Cron expression parsing (*, ranges, lists, steps)
- Schedule matching against datetime
- Schedule management (add, remove, enable, disable)
- Tick engine: fire due schedules, dedup within same minute
- Handler registration and execution
- YAML-style configuration
- Milestone 2 criterion: cron fires data load at configured time
"""

from datetime import datetime, timezone

import pytest

from praxis.logger.core import PraxisLogger
from praxis.scheduler import (
    PraxisScheduler,
    Schedule,
    TickResult,
    cron_matches,
    parse_cron_field,
)


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ═══════════════════════════════════════════════════════════════════
#  Cron Parsing
# ═══════════════════════════════════════════════════════════════════

class TestCronParsing:
    def test_star(self):
        assert parse_cron_field("*", 0, 59) == set(range(0, 60))

    def test_single_value(self):
        assert parse_cron_field("5", 0, 59) == {5}

    def test_range(self):
        assert parse_cron_field("1-5", 0, 59) == {1, 2, 3, 4, 5}

    def test_list(self):
        assert parse_cron_field("1,3,5", 0, 59) == {1, 3, 5}

    def test_step(self):
        assert parse_cron_field("*/15", 0, 59) == {0, 15, 30, 45}

    def test_range_step(self):
        assert parse_cron_field("0-30/10", 0, 59) == {0, 10, 20, 30}

    def test_day_range(self):
        # Monday-Friday in cron: 1-5
        assert parse_cron_field("1-5", 0, 7) == {1, 2, 3, 4, 5}


# ═══════════════════════════════════════════════════════════════════
#  Cron Matching
# ═══════════════════════════════════════════════════════════════════

class TestCronMatching:
    def test_every_minute(self):
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert cron_matches("* * * * *", dt) is True

    def test_specific_time(self):
        dt = datetime(2024, 6, 15, 6, 0, 0, tzinfo=timezone.utc)  # Saturday
        assert cron_matches("0 6 * * *", dt) is True

    def test_weekdays_only(self):
        # 2024-06-17 is Monday
        monday = datetime(2024, 6, 17, 6, 0, 0, tzinfo=timezone.utc)
        assert cron_matches("0 6 * * 1-5", monday) is True

        # 2024-06-15 is Saturday
        saturday = datetime(2024, 6, 15, 6, 0, 0, tzinfo=timezone.utc)
        assert cron_matches("0 6 * * 1-5", saturday) is False

    def test_specific_day_of_week(self):
        # 2024-06-21 is Friday (cron: 5)
        friday = datetime(2024, 6, 21, 18, 0, 0, tzinfo=timezone.utc)
        assert cron_matches("0 18 * * 5", friday) is True

    def test_wrong_hour(self):
        dt = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert cron_matches("0 6 * * *", dt) is False

    def test_wrong_minute(self):
        dt = datetime(2024, 6, 15, 6, 15, 0, tzinfo=timezone.utc)
        assert cron_matches("0 6 * * *", dt) is False

    def test_invalid_cron(self):
        dt = datetime(2024, 6, 15, 6, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="5 fields"):
            cron_matches("0 6 *", dt)

    def test_every_15_minutes(self):
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert cron_matches("*/15 * * * *", dt) is True

        dt2 = datetime(2024, 6, 15, 10, 7, 0, tzinfo=timezone.utc)
        assert cron_matches("*/15 * * * *", dt2) is False


# ═══════════════════════════════════════════════════════════════════
#  Schedule Management
# ═══════════════════════════════════════════════════════════════════

class TestScheduleManagement:
    def test_add_schedule(self):
        sched = PraxisScheduler()
        s = sched.add_schedule("test", "0 6 * * 1-5", "data_load")
        assert s.name == "test"
        assert s.cron == "0 6 * * 1-5"
        assert s.action == "data_load"
        assert s.enabled is True

    def test_add_with_params(self):
        sched = PraxisScheduler()
        s = sched.add_schedule("test", "0 6 * * *", "data_load", {"source": "yfinance"})
        assert s.params["source"] == "yfinance"

    def test_remove_schedule(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "0 6 * * *", "data_load")
        assert sched.remove_schedule("test") is True
        assert sched.get_schedule("test") is None

    def test_remove_nonexistent(self):
        sched = PraxisScheduler()
        assert sched.remove_schedule("nope") is False

    def test_enable_disable(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "0 6 * * *", "data_load")
        sched.disable_schedule("test")
        assert sched.get_schedule("test").enabled is False

        sched.enable_schedule("test")
        assert sched.get_schedule("test").enabled is True

    def test_list_schedules(self):
        sched = PraxisScheduler()
        sched.add_schedule("a", "0 6 * * *", "load")
        sched.add_schedule("b", "0 18 * * 5", "backtest")
        assert len(sched.list_schedules()) == 2

    def test_invalid_cron_rejected(self):
        sched = PraxisScheduler()
        with pytest.raises(ValueError):
            sched.add_schedule("bad", "invalid", "load")


# ═══════════════════════════════════════════════════════════════════
#  Tick Engine
# ═══════════════════════════════════════════════════════════════════

class TestTickEngine:
    def test_tick_fires_due_schedule(self):
        sched = PraxisScheduler()
        sched.add_schedule("morning_load", "0 6 * * 1-5", "data_load")

        # Monday 06:00
        at = datetime(2024, 6, 17, 6, 0, 0, tzinfo=timezone.utc)
        result = sched.tick(at=at)

        assert result.schedules_checked == 1
        assert result.schedules_due == 1
        assert "morning_load" in result.executed

    def test_tick_skips_not_due(self):
        sched = PraxisScheduler()
        sched.add_schedule("morning_load", "0 6 * * 1-5", "data_load")

        # Monday 10:00 — not due
        at = datetime(2024, 6, 17, 10, 0, 0, tzinfo=timezone.utc)
        result = sched.tick(at=at)

        assert result.schedules_due == 0
        assert len(result.executed) == 0

    def test_tick_skips_disabled(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "* * * * *", "load")
        sched.disable_schedule("test")

        result = sched.tick()
        assert result.schedules_checked == 0

    def test_tick_dedup_within_minute(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "0 6 * * *", "load")

        at = datetime(2024, 6, 17, 6, 0, 0, tzinfo=timezone.utc)
        r1 = sched.tick(at=at)
        assert len(r1.executed) == 1

        # Same minute — should not fire again
        at2 = datetime(2024, 6, 17, 6, 0, 30, tzinfo=timezone.utc)
        r2 = sched.tick(at=at2)
        assert len(r2.executed) == 0

    def test_tick_fires_again_next_match(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "0 * * * *", "load")

        at1 = datetime(2024, 6, 17, 6, 0, 0, tzinfo=timezone.utc)
        sched.tick(at=at1)

        # Next hour
        at2 = datetime(2024, 6, 17, 7, 0, 0, tzinfo=timezone.utc)
        r2 = sched.tick(at=at2)
        assert len(r2.executed) == 1


# ═══════════════════════════════════════════════════════════════════
#  Handler Execution
# ═══════════════════════════════════════════════════════════════════

class TestHandlers:
    def test_handler_called_with_params(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "* * * * *", "data_load", {"source": "yfinance"})

        calls = []
        sched.register_handler("data_load", lambda params: calls.append(params))

        sched.tick()
        assert len(calls) == 1
        assert calls[0]["source"] == "yfinance"

    def test_handler_error_captured(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "* * * * *", "fail_action")
        sched.register_handler("fail_action", lambda p: 1 / 0)

        result = sched.tick()
        assert len(result.errors) == 1
        assert "division by zero" in result.errors[0]

    def test_run_count_incremented(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "0 * * * *", "load")

        at = datetime(2024, 6, 17, 6, 0, 0, tzinfo=timezone.utc)
        sched.tick(at=at)

        s = sched.get_schedule("test")
        assert s.run_count == 1


# ═══════════════════════════════════════════════════════════════════
#  YAML Configuration
# ═══════════════════════════════════════════════════════════════════

class TestConfiguration:
    def test_configure_from_dict(self):
        sched = PraxisScheduler()
        sched.configure({
            "scheduler": {
                "tick_interval_seconds": 30,
                "explicit_schedules": [
                    {
                        "name": "daily_load",
                        "cron": "0 6 * * 1-5",
                        "action": "data_load",
                        "params": {"source": "yfinance"},
                    },
                    {
                        "name": "weekly_backtest",
                        "cron": "0 18 * * 5",
                        "action": "backtest",
                    },
                ],
            },
        })

        assert sched._tick_interval == 30
        assert len(sched.list_schedules()) == 2

    def test_get_due_schedules(self):
        sched = PraxisScheduler()
        sched.add_schedule("a", "0 6 * * *", "load")
        sched.add_schedule("b", "0 18 * * *", "backtest")

        at = datetime(2024, 6, 17, 6, 0, 0, tzinfo=timezone.utc)
        due = sched.get_due_schedules(at)
        assert len(due) == 1
        assert due[0].name == "a"


# ═══════════════════════════════════════════════════════════════════
#  Milestone 2 Criterion
# ═══════════════════════════════════════════════════════════════════

class TestMilestone2Scheduler:
    def test_cron_triggers_data_load_at_configured_time(self):
        """
        Milestone 2 pass:
        'cron: "0 6 * * 1-5"' triggers data load at configured time
        """
        sched = PraxisScheduler()
        sched.configure({
            "scheduler": {
                "explicit_schedules": [{
                    "name": "daily_equity_load",
                    "cron": "0 6 * * 1-5",
                    "action": "data_load",
                    "params": {"sources": ["yfinance"], "universe": "sp500"},
                }],
            },
        })

        loads_executed = []
        sched.register_handler("data_load", lambda p: loads_executed.append(p))

        # Wednesday 06:00 UTC — should fire
        wednesday_6am = datetime(2024, 6, 19, 6, 0, 0, tzinfo=timezone.utc)
        result = sched.tick(at=wednesday_6am)
        assert result.schedules_executed == 1
        assert loads_executed[0]["sources"] == ["yfinance"]

        # Saturday 06:00 UTC — should NOT fire (weekend)
        saturday_6am = datetime(2024, 6, 22, 6, 0, 0, tzinfo=timezone.utc)
        result = sched.tick(at=saturday_6am)
        assert result.schedules_executed == 0

        # Wednesday 10:00 UTC — should NOT fire (wrong hour)
        wednesday_10am = datetime(2024, 6, 19, 10, 0, 0, tzinfo=timezone.utc)
        result = sched.tick(at=wednesday_10am)
        assert result.schedules_executed == 0


# ═══════════════════════════════════════════════════════════════════
#  Status
# ═══════════════════════════════════════════════════════════════════

class TestStatus:
    def test_status_structure(self):
        sched = PraxisScheduler()
        sched.add_schedule("test", "0 6 * * *", "load")
        status = sched.status()

        assert status["running"] is False
        assert status["schedules"] == 1
        assert len(status["schedule_details"]) == 1
        assert status["schedule_details"][0]["name"] == "test"
