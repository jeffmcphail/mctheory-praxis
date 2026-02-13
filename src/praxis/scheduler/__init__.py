"""
Basic Explicit Scheduler (Phase 2.13, §19.7).

Cron-like scheduling from config. Foundation for declarative scheduling.

Supports:
- Cron expressions (minute, hour, day_of_month, month, day_of_week)
- Named schedules with actions and params
- Tick-based checking (is this schedule due?)
- Schedule history tracking

Usage:
    scheduler = PraxisScheduler(conn)
    scheduler.add_schedule("daily_load", "0 6 * * 1-5", "data_load", {"source": "yfinance"})
    due = scheduler.tick()  # Returns list of due schedules
    for schedule in due:
        scheduler.execute(schedule)

Cron format: minute hour day_of_month month day_of_week
  - * = any
  - 1-5 = range
  - 1,3,5 = list
  - */15 = step
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Optional

import duckdb

from praxis.logger.core import PraxisLogger


@dataclass
class Schedule:
    """A named cron schedule."""
    name: str
    cron: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 50
    last_run: datetime | None = None
    run_count: int = 0


@dataclass
class TickResult:
    """Result of a scheduler tick."""
    timestamp: datetime
    schedules_checked: int = 0
    schedules_due: int = 0
    schedules_executed: int = 0
    executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_cron_field(field_str: str, min_val: int, max_val: int) -> set[int]:
    """
    Parse a single cron field into a set of valid values.

    Supports: *, N, N-M, N,M,O, */N, N-M/S
    """
    values = set()

    for part in field_str.split(","):
        part = part.strip()

        if part == "*":
            values.update(range(min_val, max_val + 1))
        elif "/" in part:
            range_part, step = part.split("/", 1)
            step = int(step)
            if range_part == "*":
                start, end = min_val, max_val
            elif "-" in range_part:
                start, end = map(int, range_part.split("-", 1))
            else:
                start, end = int(range_part), max_val
            values.update(range(start, end + 1, step))
        elif "-" in part:
            start, end = map(int, part.split("-", 1))
            values.update(range(start, end + 1))
        else:
            values.add(int(part))

    return values


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """
    Check if a datetime matches a cron expression.

    Format: minute hour day_of_month month day_of_week
    day_of_week: 0=Monday, 6=Sunday (ISO style, but also accepts 0=Sunday, 7=Sunday)
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Invalid cron expression (need 5 fields): {cron_expr}")

    minute_vals = parse_cron_field(fields[0], 0, 59)
    hour_vals = parse_cron_field(fields[1], 0, 23)
    dom_vals = parse_cron_field(fields[2], 1, 31)
    month_vals = parse_cron_field(fields[3], 1, 12)
    dow_vals = parse_cron_field(fields[4], 0, 7)

    # Normalize day_of_week: Python uses 0=Monday, cron traditionally 0=Sunday
    # We accept both: 0 and 7 mean Sunday, 1=Monday, 5=Friday
    # Convert Python weekday (0=Mon) to cron (0=Sun): (py_weekday + 1) % 7
    py_dow = dt.weekday()  # 0=Monday
    cron_dow = (py_dow + 1) % 7  # 0=Sunday

    return (
        dt.minute in minute_vals
        and dt.hour in hour_vals
        and dt.day in dom_vals
        and dt.month in month_vals
        and (cron_dow in dow_vals or (7 in dow_vals and cron_dow == 0))
    )


class PraxisScheduler:
    """
    §19.7: Basic explicit scheduler with cron expressions.

    Phase 2.13 scope: explicit scheduling only.
    Phase 3.13-3.14 will add declarative scheduling + DAG engine.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection | None = None):
        self._conn = conn
        self._log = PraxisLogger.instance()
        self._schedules: dict[str, Schedule] = {}
        self._handlers: dict[str, Callable] = {}
        self._tick_interval: int = 60  # seconds
        self._running = False

    # ── Schedule management ───────────────────────────────────

    def add_schedule(
        self,
        name: str,
        cron: str,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> Schedule:
        """Add a named schedule."""
        # Validate cron expression
        try:
            cron_matches(cron, datetime.now(timezone.utc))
        except Exception as e:
            raise ValueError(f"Invalid cron expression '{cron}': {e}")

        schedule = Schedule(
            name=name,
            cron=cron,
            action=action,
            params=params or {},
        )
        self._schedules[name] = schedule

        self._log.info(
            f"Schedule added: {name} ({cron}) → {action}",
            tags={"scheduler_cycle"},
        )
        return schedule

    def remove_schedule(self, name: str) -> bool:
        """Remove a schedule by name."""
        if name in self._schedules:
            del self._schedules[name]
            return True
        return False

    def enable_schedule(self, name: str) -> bool:
        if name in self._schedules:
            self._schedules[name].enabled = True
            return True
        return False

    def disable_schedule(self, name: str) -> bool:
        if name in self._schedules:
            self._schedules[name].enabled = False
            return True
        return False

    def get_schedule(self, name: str) -> Schedule | None:
        return self._schedules.get(name)

    def list_schedules(self) -> list[Schedule]:
        return list(self._schedules.values())

    # ── Action handlers ───────────────────────────────────────

    def register_handler(self, action: str, handler: Callable) -> None:
        """Register a handler function for an action type."""
        self._handlers[action] = handler

    # ── Configuration ─────────────────────────────────────────

    def configure(self, config: dict[str, Any]) -> None:
        """
        Configure from YAML-style dict.

        scheduler:
          tick_interval_seconds: 60
          explicit_schedules:
            - name: daily_equity_load
              cron: "0 6 * * 1-5"
              action: data_load
              params:
                sources: [yfinance]
        """
        sched_config = config.get("scheduler", config)

        if "tick_interval_seconds" in sched_config:
            self._tick_interval = sched_config["tick_interval_seconds"]

        for sched in sched_config.get("explicit_schedules", []):
            self.add_schedule(
                sched["name"],
                sched["cron"],
                sched["action"],
                sched.get("params", {}),
            )

    # ── Tick engine ───────────────────────────────────────────

    def tick(self, at: datetime | None = None) -> TickResult:
        """
        Execute one scheduler tick.

        Checks all schedules against the current time (or `at`),
        fires handlers for due schedules.

        Args:
            at: Override current time (for testing).

        Returns:
            TickResult with execution summary.
        """
        now = at or datetime.now(timezone.utc)
        result = TickResult(timestamp=now)

        for name, schedule in self._schedules.items():
            if not schedule.enabled:
                continue

            result.schedules_checked += 1

            if not cron_matches(schedule.cron, now):
                continue

            # Dedup: don't re-fire within the same minute
            if schedule.last_run and schedule.last_run.replace(second=0, microsecond=0) == now.replace(second=0, microsecond=0):
                continue

            result.schedules_due += 1

            # Execute handler if registered
            handler = self._handlers.get(schedule.action)
            if handler:
                try:
                    handler(schedule.params)
                    schedule.last_run = now
                    schedule.run_count += 1
                    result.schedules_executed += 1
                    result.executed.append(name)

                    self._log.info(
                        f"Scheduler: executed {name} ({schedule.action})",
                        tags={"scheduler_cycle"},
                    )
                except Exception as e:
                    result.errors.append(f"{name}: {e}")
                    self._log.error(
                        f"Scheduler: {name} failed: {e}",
                        tags={"scheduler_cycle"},
                    )
            else:
                # No handler but schedule is due — record it
                schedule.last_run = now
                schedule.run_count += 1
                result.schedules_due += 0  # Already counted
                result.executed.append(name)

                self._log.debug(
                    f"Scheduler: {name} due but no handler for '{schedule.action}'",
                    tags={"scheduler_cycle"},
                )

        return result

    def get_due_schedules(self, at: datetime | None = None) -> list[Schedule]:
        """Check which schedules are due without executing them."""
        now = at or datetime.now(timezone.utc)
        due = []

        for schedule in self._schedules.values():
            if not schedule.enabled:
                continue
            if cron_matches(schedule.cron, now):
                due.append(schedule)

        return due

    # ── Run loop (blocking) ───────────────────────────────────

    def run(self, max_ticks: int | None = None) -> None:
        """
        Run the scheduler loop.

        Args:
            max_ticks: Stop after N ticks (for testing). None = run forever.
        """
        self._running = True
        tick_count = 0

        self._log.info(
            f"Scheduler started (interval={self._tick_interval}s, "
            f"schedules={len(self._schedules)})",
            tags={"scheduler_cycle"},
        )

        try:
            while self._running:
                self.tick()
                tick_count += 1

                if max_ticks and tick_count >= max_ticks:
                    break

                time.sleep(self._tick_interval)
        finally:
            self._running = False
            self._log.info(
                f"Scheduler stopped after {tick_count} ticks",
                tags={"scheduler_cycle"},
            )

    def stop(self) -> None:
        """Signal the scheduler to stop."""
        self._running = False

    # ── Status ────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "tick_interval_seconds": self._tick_interval,
            "schedules": len(self._schedules),
            "handlers": list(self._handlers.keys()),
            "schedule_details": [
                {
                    "name": s.name,
                    "cron": s.cron,
                    "action": s.action,
                    "enabled": s.enabled,
                    "run_count": s.run_count,
                    "last_run": s.last_run.isoformat() if s.last_run else None,
                }
                for s in self._schedules.values()
            ],
        }
