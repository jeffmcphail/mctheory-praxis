"""
Logger Runtime Reconfiguration (Phase 2.14).

Provides runtime control over logging without restart:
- Add/remove active tags
- Change log level (global, per-adapter, per-tag)
- Get status overview
- Access agent ring buffer

Usage:
    reconfig = LoggerReconfig()
    reconfig.add_tag("data_pipeline")
    reconfig.set_level("DEBUG")
    status = reconfig.status()
    recent = reconfig.get_agent_buffer(n=50)
"""

from __future__ import annotations

from typing import Any

from praxis.logger.core import PraxisLogger
from praxis.logger.adapters import AgentAdapter


class LoggerReconfig:
    """Runtime reconfiguration interface for PraxisLogger."""

    def __init__(self, logger: PraxisLogger | None = None):
        self._log = logger or PraxisLogger.instance()

    # ── Tag management ────────────────────────────────────────

    def add_tag(self, tag: str) -> None:
        """Activate a logging tag at runtime."""
        self._log.activate_tag(tag)

    def remove_tag(self, tag: str) -> None:
        """Deactivate a logging tag at runtime."""
        self._log.deactivate_tag(tag)

    def set_tag_level(self, tag: str, level: str | int) -> None:
        """Set per-tag threshold level."""
        self._log.set_tag_level(tag, level)

    def list_tags(self) -> list[str]:
        """Get all active tags."""
        return sorted(self._log.active_tags)

    # ── Level management ──────────────────────────────────────

    def set_level(self, level: str | int) -> None:
        """Set global log level."""
        self._log.current_level = level

    def set_adapter_level(self, adapter_name: str, level: str | int) -> None:
        """Set level for a specific adapter."""
        self._log.set_adapter_level(adapter_name, level)

    def get_level(self) -> int:
        """Get current global level."""
        return self._log.current_level

    # ── Status ────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """
        Get complete logger status overview.

        Returns:
            {
                "global_level": int,
                "active_tags": [...],
                "tag_levels": {...},
                "adapters": [{"name": ..., "type": ..., "level": ...}],
                "session_id": ...,
                "agent_buffer_size": int,
            }
        """
        adapters = []
        for name, adapter in self._log._adapters.items():
            info = {
                "name": name,
                "type": type(adapter).__name__,
                "level": adapter.min_level,
            }
            if isinstance(adapter, AgentAdapter):
                info["buffer_count"] = adapter.count
                info["buffer_max"] = adapter._buffer.maxlen
            adapters.append(info)

        return {
            "global_level": self._log.current_level,
            "active_tags": sorted(self._log.active_tags),
            "tag_levels": dict(self._log.tag_levels),
            "adapters": adapters,
            "session_id": self._log.session_id,
        }

    # ── Agent ring buffer access ──────────────────────────────

    def get_agent_buffer(
        self,
        n: int = 100,
        tags: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Read from agent ring buffer.

        Returns formatted log records from the agent's diagnostic window.
        """
        agent = self._log.get_adapter("agent")
        if agent is None or not isinstance(agent, AgentAdapter):
            return []

        records = agent.get_recent(n, tags)
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "level": r.level_name,
                "message": r.message,
                "tags": sorted(r.tags) if r.tags else [],
                "context": r.context,
            }
            for r in records
        ]

    def clear_agent_buffer(self) -> None:
        """Clear the agent ring buffer."""
        agent = self._log.get_adapter("agent")
        if agent is not None and isinstance(agent, AgentAdapter):
            agent.clear()
