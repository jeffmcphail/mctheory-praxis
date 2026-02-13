"""
Logger Agent Tool Integration (Phase 4.16, §18.7).

Provides the agent with runtime log control capabilities:
- Activate/deactivate trace tags
- Adjust verbosity levels (global and per-adapter)
- Read recent logs from ring buffer (filtered by tag/level)
- Diagnostic snapshots for troubleshooting workflows
- Save/restore log configuration for scoped debugging

Usage:
    tool = LoggerAgentTool()
    tool.activate_tags(["trade_cycle", "data_pipeline"])
    tool.set_level("debug")
    logs = tool.get_recent_logs(n=50, tags=["trade_cycle"])
    tool.restore_config()  # Restore original settings
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from praxis.logger.core import PraxisLogger
from praxis.logger.records import LogRecord, LogLevel, level_name
from praxis.logger.adapters import LogAdapter


# ═══════════════════════════════════════════════════════════════════
#  Ring Buffer Adapter
# ═══════════════════════════════════════════════════════════════════

class RingBufferAdapter(LogAdapter):
    """
    In-memory ring buffer adapter for agent log reading.

    Stores the last N log records in a circular buffer.
    The agent reads from this buffer for diagnostics.
    """

    def __init__(self, name: str = "agent_ring", capacity: int = 1000, min_level: int = 10):
        super().__init__(name=name, min_level=min_level)
        self._buffer: deque[LogRecord] = deque(maxlen=capacity)
        self._capacity = capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return len(self._buffer)

    def emit(self, record: LogRecord) -> None:
        self._buffer.append(record)

    def get_records(
        self,
        n: int = 100,
        tags: list[str] | None = None,
        min_level: int | None = None,
        contains: str | None = None,
    ) -> list[LogRecord]:
        """
        Read recent records with optional filters.

        Args:
            n: Max records to return.
            tags: Only records containing at least one of these tags.
            min_level: Only records at or above this level.
            contains: Only records whose message contains this substring.
        """
        result = []
        tag_set = set(tags) if tags else None

        for record in reversed(self._buffer):
            if len(result) >= n:
                break

            if min_level is not None and record.level < min_level:
                continue

            if tag_set is not None:
                record_tags = record.tags if isinstance(record.tags, set) else set(record.tags or [])
                if not tag_set.intersection(record_tags):
                    continue

            if contains is not None and contains not in record.message:
                continue

            result.append(record)

        result.reverse()
        return result

    def clear(self) -> None:
        """Clear the ring buffer."""
        self._buffer.clear()


# ═══════════════════════════════════════════════════════════════════
#  Config Snapshot
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LogConfigSnapshot:
    """Snapshot of logger configuration for save/restore."""
    level: int = 20
    active_tags: set[str] = field(default_factory=set)
    tag_levels: dict[str, int] = field(default_factory=dict)
    adapter_levels: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════════
#  Logger Agent Tool
# ═══════════════════════════════════════════════════════════════════

class LoggerAgentTool:
    """
    Agent tool for runtime log reconfiguration (§18.7).

    Provides the agent with a controlled interface to the logger,
    including save/restore for scoped debugging sessions.
    """

    def __init__(self, ring_capacity: int = 1000):
        self._logger = PraxisLogger.instance()
        self._ring = RingBufferAdapter(capacity=ring_capacity)
        self._saved_config: LogConfigSnapshot | None = None

        # Install ring buffer adapter if not already present
        if self._logger.get_adapter("agent_ring") is None:
            self._logger.add_adapter(self._ring)

    @property
    def ring_buffer(self) -> RingBufferAdapter:
        return self._ring

    # ── Tag Control ───────────────────────────────────────────

    def activate_tags(self, tags: list[str]) -> str:
        """Activate trace tags for diagnostic visibility."""
        for tag in tags:
            self._logger.activate_tag(tag)
        active = self._logger.active_tags
        return f"Activated {tags}. Active tags: {sorted(active)}"

    def deactivate_tags(self, tags: list[str]) -> str:
        """Deactivate trace tags."""
        for tag in tags:
            self._logger.deactivate_tag(tag)
        active = self._logger.active_tags
        return f"Deactivated {tags}. Active tags: {sorted(active)}"

    def get_active_tags(self) -> list[str]:
        """Get currently active tags."""
        return sorted(self._logger.active_tags)

    # ── Level Control ─────────────────────────────────────────

    def set_level(self, level: int | str) -> str:
        """Set global log level."""
        old = self._logger.current_level
        self._logger.current_level = level
        new = self._logger.current_level
        return f"Level changed: {level_name(old)} ({old}) → {level_name(new)} ({new})"

    def set_adapter_level(self, adapter_name: str, level: int | str) -> str:
        """Set minimum level for a specific adapter."""
        self._logger.set_adapter_level(adapter_name, level)
        return f"Adapter '{adapter_name}' level set to {level}"

    def get_level(self) -> dict[str, Any]:
        """Get current level info."""
        lvl = self._logger.current_level
        return {
            "global_level": lvl,
            "global_level_name": level_name(lvl),
        }

    # ── Log Reading ───────────────────────────────────────────

    def get_recent_logs(
        self,
        n: int = 100,
        tags: list[str] | None = None,
        min_level: int | None = None,
        contains: str | None = None,
    ) -> list[LogRecord]:
        """Read recent logs from ring buffer with filters."""
        return self._ring.get_records(n=n, tags=tags, min_level=min_level, contains=contains)

    def get_log_summary(self, n: int = 100) -> dict[str, Any]:
        """Get summary statistics of recent logs."""
        records = self._ring.get_records(n=n)
        by_level: dict[str, int] = {}
        all_tags: dict[str, int] = {}

        for r in records:
            name = level_name(r.level)
            by_level[name] = by_level.get(name, 0) + 1
            for tag in (r.tags if isinstance(r.tags, set) else set(r.tags or [])):
                all_tags[tag] = all_tags.get(tag, 0) + 1

        return {
            "total_records": len(records),
            "buffer_size": self._ring.size,
            "buffer_capacity": self._ring.capacity,
            "by_level": by_level,
            "top_tags": dict(sorted(all_tags.items(), key=lambda x: -x[1])[:10]),
        }

    # ── Config Save/Restore ───────────────────────────────────

    def save_config(self) -> LogConfigSnapshot:
        """
        Save current logger configuration.

        Call before making diagnostic changes so you can restore later.
        """
        self._saved_config = LogConfigSnapshot(
            level=self._logger.current_level,
            active_tags=set(self._logger.active_tags),
            tag_levels=dict(self._logger.tag_levels),
        )
        return self._saved_config

    def restore_config(self) -> bool:
        """
        Restore previously saved configuration.

        Returns True if restoration succeeded.
        """
        if self._saved_config is None:
            return False

        snap = self._saved_config
        self._logger.current_level = snap.level

        # Restore tags: deactivate all current, activate saved
        for tag in list(self._logger.active_tags):
            self._logger.deactivate_tag(tag)
        for tag in snap.active_tags:
            self._logger.activate_tag(tag)

        # Restore tag levels
        for tag, level in snap.tag_levels.items():
            self._logger.set_tag_level(tag, level)

        self._saved_config = None
        return True

    def has_saved_config(self) -> bool:
        """Check if there's a saved config to restore."""
        return self._saved_config is not None

    # ── Troubleshooting Workflow ──────────────────────────────

    def start_diagnostic(self, domain: str) -> str:
        """
        Start a scoped diagnostic session.

        Saves config, activates relevant tags, lowers level to DEBUG.

        Args:
            domain: Problem domain — "trading", "data", "scheduler", "backtest"
        """
        self.save_config()
        self._ring.clear()

        # Domain-specific tag activation
        domain_tags = {
            "trading": ["trade_cycle", "execution", "risk_check", "paper"],
            "data": ["data_pipeline", "data_load", "data_quality", "validation"],
            "scheduler": ["scheduler", "tick", "dag", "lifecycle"],
            "backtest": ["backtest", "signals", "sizing", "fill_model"],
            "agent": ["agent", "discovery", "autonomous"],
        }

        tags = domain_tags.get(domain, [domain])
        self.activate_tags(tags)
        self.set_level("debug")

        return f"Diagnostic started for '{domain}'. Tags: {tags}. Level: DEBUG. Ring buffer cleared."

    def end_diagnostic(self) -> dict[str, Any]:
        """
        End diagnostic session. Restore config, return summary.
        """
        summary = self.get_log_summary()
        self.restore_config()
        return {
            "diagnostic_complete": True,
            "logs_captured": summary["total_records"],
            "by_level": summary["by_level"],
            "config_restored": True,
        }

    # ── Status ────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Get full logger status."""
        return {
            "global_level": level_name(self._logger.current_level),
            "active_tags": sorted(self._logger.active_tags),
            "ring_buffer_size": self._ring.size,
            "ring_buffer_capacity": self._ring.capacity,
            "has_saved_config": self.has_saved_config(),
        }
