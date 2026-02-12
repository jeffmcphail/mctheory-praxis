"""
PraxisLogger: Singleton diagnostic instrumentation system.

§18.1: Not a logging library — a diagnostic instrumentation system.
§18.2: One instance, multiple output adapters, configurable routing matrix.

The nanosecond gate: most log calls exit at _should_emit() with no work done.
Log statements are placed at development time and activated at troubleshooting time.
"""

import threading
from datetime import datetime, timezone
from typing import Any, Optional

from praxis.logger.records import LogRecord, LogLevel, level_name
from praxis.logger.adapters import (
    LogAdapter,
    TerminalAdapter,
    FileAdapter,
    DatabaseAdapter,
    AgentAdapter,
)
from praxis.logger.routing import RoutingMatrix
from praxis.logger.formatters import (
    LogFormatter,
    CompactFormatter,
    DetailedFormatter,
    JsonFormatter,
)


class PraxisLogger:
    """
    §18.2: Singleton logger with routing matrix.

    Usage:
        log = PraxisLogger.instance()
        log.info("Security matched", tags={"datastore.security_master.match"})
        log.debug("Z-score computed", tags={"compute.signals", "trade_cycle"}, zscore=2.1)
    """

    _instance: Optional["PraxisLogger"] = None
    _lock = threading.Lock()

    # Re-export levels for convenience: PraxisLogger.DEBUG, etc.
    TRACE = LogLevel.TRACE
    DEBUG = LogLevel.DEBUG
    VERBOSE = LogLevel.VERBOSE
    INFO = LogLevel.INFO
    NOTICE = LogLevel.NOTICE
    WARNING = LogLevel.WARNING
    ERROR = LogLevel.ERROR
    ALERT = LogLevel.ALERT
    CRITICAL = LogLevel.CRITICAL

    def __init__(self) -> None:
        self._adapters: dict[str, LogAdapter] = {}
        self._routing = RoutingMatrix()
        self._active_tags: set[str] = set()
        self._tag_levels: dict[str, int] = {}  # tag → auto-activate threshold
        self._current_level: int = LogLevel.INFO
        self._session_id: str | None = None
        self._emit_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "PraxisLogger":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset singleton. For testing only — not for production use.
        Closes all adapters before resetting.
        """
        with cls._lock:
            if cls._instance is not None:
                for adapter in cls._instance._adapters.values():
                    adapter.close()
                cls._instance = None

    # ── Configuration ─────────────────────────────────────────────

    def configure(self, config: dict) -> None:
        """
        Configure logger from a dict (parsed YAML).

        Expected structure matches §18.6:
            current_level: 20
            adapters:
                terminal: {type: terminal, min_level: 20, color: true}
                logfile:  {type: file, path: logs/praxis.log, min_level: 10}
                database: {type: database, min_level: 20, buffer_size: 1000}
                agent:    {type: agent, min_level: 10, ring_buffer_size: 10000}
            tag_levels:
                trade_cycle: 20
                data_pipeline: 20
                cpo_cycle: 10
            routing:
                critical_override: true
                tag_routes:
                    trade_cycle: [terminal, database, agent]
                    data_pipeline: [database, agent]
        """
        # Global level
        if "current_level" in config:
            self._current_level = _resolve_level(config["current_level"])

        # Adapters
        for name, adapter_cfg in config.get("adapters", {}).items():
            adapter = _build_adapter(name, adapter_cfg)
            self._adapters[name] = adapter

        # Tag levels (auto-activate thresholds) §18.4
        for tag, threshold in config.get("tag_levels", {}).items():
            self._tag_levels[tag] = _resolve_level(threshold)

        # Routing §18.6
        routing_cfg = config.get("routing", {})
        self._routing.critical_override = routing_cfg.get("critical_override", True)
        for tag, adapter_names in routing_cfg.get("tag_routes", {}).items():
            self._routing.add_tag_route(tag, adapter_names)

    def configure_defaults(self) -> None:
        """
        Set up a sensible default configuration for development.
        Terminal at INFO, no file/database adapters.
        """
        self._current_level = LogLevel.INFO
        self._adapters["terminal"] = TerminalAdapter(color=True)
        self._routing.critical_override = True

    # ── Adapter Management ────────────────────────────────────────

    def add_adapter(self, adapter: LogAdapter) -> None:
        """Add or replace an adapter."""
        self._adapters[adapter.name] = adapter

    def remove_adapter(self, name: str) -> LogAdapter | None:
        """Remove an adapter by name. Returns it (for closing) or None."""
        adapter = self._adapters.pop(name, None)
        if adapter:
            adapter.close()
        return adapter

    def get_adapter(self, name: str) -> LogAdapter | None:
        """Get an adapter by name."""
        return self._adapters.get(name)

    # ── Tag Management (§18.3, §18.7) ────────────────────────────

    def activate_tag(self, tag: str) -> None:
        """§18.7: Explicitly activate a tag for diagnostic output."""
        self._active_tags.add(tag)

    def deactivate_tag(self, tag: str) -> None:
        """§18.7: Deactivate a previously activated tag."""
        self._active_tags.discard(tag)

    def set_tag_level(self, tag: str, threshold: int | str) -> None:
        """§18.4: Set auto-activate threshold for a tag."""
        self._tag_levels[tag] = _resolve_level(threshold)

    @property
    def active_tags(self) -> frozenset[str]:
        return frozenset(self._active_tags)

    @property
    def tag_levels(self) -> dict[str, int]:
        return dict(self._tag_levels)

    # ── Level Management (§18.7) ──────────────────────────────────

    @property
    def current_level(self) -> int:
        return self._current_level

    @current_level.setter
    def current_level(self, value: int | str) -> None:
        self._current_level = _resolve_level(value)

    def set_adapter_level(self, adapter_name: str, level: int | str) -> None:
        """§18.7: Change an adapter's min_level at runtime."""
        adapter = self._adapters.get(adapter_name)
        if adapter is None:
            raise ValueError(f"Unknown adapter '{adapter_name}'")
        adapter.min_level = _resolve_level(level)

    # ── Session Management ────────────────────────────────────────

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._session_id = value

    # ── Core Logging (§18.2) ──────────────────────────────────────

    def log(
        self,
        level: int,
        message: str,
        tags: set[str] | None = None,
        source: str = "system",
        **context: Any,
    ) -> None:
        """
        §18.2: Core logging method.

        A log statement executes its adapters only if:
        1. level >= current_level (global verbosity floor), OR
        2. Any tag in `tags` is in active_tags, OR
        3. Any tag's auto-activate threshold <= current_level

        If none are true, returns immediately (nanosecond cost).
        """
        tags = tags or set()

        # The nanosecond gate
        if not self._should_emit(level, tags):
            return

        record = LogRecord.create(
            level=level,
            message=message,
            tags=tags,
            source=source,
            session_id=self._session_id,
            **context,
        )

        # Route to appropriate adapters
        with self._emit_lock:
            for adapter_name, adapter in self._adapters.items():
                effective_level = adapter.min_level
                # Database adapter has backtest throttling
                if isinstance(adapter, DatabaseAdapter):
                    effective_level = adapter.effective_min_level

                if self._routing.should_route(record, adapter_name, effective_level):
                    try:
                        adapter.emit(record)
                    except Exception:
                        # Never let adapter failure crash the caller
                        pass

    def _should_emit(self, level: int, tags: set[str]) -> bool:
        """
        §18.2: The nanosecond gate. Most calls exit here.

        Returns True if this log statement should produce output.
        """
        # No adapters → nothing to do
        if not self._adapters:
            return False

        # 1. Global verbosity floor
        if level >= self._current_level:
            return True

        # 2. Explicitly activated tags
        if tags and (tags & self._active_tags):
            return True

        # 3. Tag auto-activate thresholds
        for tag in tags:
            threshold = self._tag_levels.get(tag)
            if threshold is not None and threshold <= self._current_level:
                return True

        return False

    # ── Convenience Methods ───────────────────────────────────────

    def trace(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.TRACE, message, tags, **ctx)

    def debug(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.DEBUG, message, tags, **ctx)

    def verbose(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.VERBOSE, message, tags, **ctx)

    def info(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.INFO, message, tags, **ctx)

    def notice(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.NOTICE, message, tags, **ctx)

    def warning(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.WARNING, message, tags, **ctx)

    def error(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.ERROR, message, tags, **ctx)

    def alert(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.ALERT, message, tags, **ctx)

    def critical(self, message: str, tags: set[str] | None = None, **ctx: Any) -> None:
        self.log(LogLevel.CRITICAL, message, tags, **ctx)

    # ── Status (§18.7) ────────────────────────────────────────────

    def status(self) -> dict:
        """
        §18.7: praxis logger status

        Returns current logger state for display.
        """
        return {
            "current_level": self._current_level,
            "current_level_name": level_name(self._current_level),
            "active_tags": sorted(self._active_tags),
            "adapters": {
                name: {
                    "type": type(adapter).__name__,
                    "min_level": adapter.min_level,
                    "min_level_name": level_name(adapter.min_level),
                }
                for name, adapter in self._adapters.items()
            },
            "tag_levels": {
                tag: {"threshold": lvl, "threshold_name": level_name(lvl)}
                for tag, lvl in sorted(self._tag_levels.items())
            },
            "routing": self._routing.describe(),
        }

    # ── Cleanup ───────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush all adapters."""
        for adapter in self._adapters.values():
            adapter.flush()

    def close(self) -> None:
        """Close all adapters. Call during shutdown."""
        for adapter in self._adapters.values():
            adapter.close()


# ── Helpers ───────────────────────────────────────────────────────────

def _resolve_level(value: int | str) -> int:
    """Convert level name or int to numeric level."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return LogLevel.from_name(value).value
    raise TypeError(f"Expected int or str for level, got {type(value).__name__}")


def _build_adapter(name: str, cfg: dict) -> LogAdapter:
    """Build an adapter from config dict."""
    adapter_type = cfg.get("type", name)
    min_level = _resolve_level(cfg.get("min_level", LogLevel.INFO))

    # Resolve formatter
    formatter = None
    fmt_name = cfg.get("formatter")
    if fmt_name == "compact":
        formatter = CompactFormatter()
    elif fmt_name == "detailed":
        formatter = DetailedFormatter()
    elif fmt_name == "json" or fmt_name is True:
        formatter = JsonFormatter()

    if adapter_type == "terminal":
        return TerminalAdapter(
            name=name,
            min_level=min_level,
            formatter=formatter,
            color=cfg.get("color", True),
        )
    elif adapter_type == "file":
        return FileAdapter(
            name=name,
            min_level=min_level,
            formatter=formatter,
            path=cfg.get("path", "logs/praxis.log"),
            rotation=cfg.get("rotation", "daily"),
            retention_days=cfg.get("retention_days", 30),
        )
    elif adapter_type == "database":
        return DatabaseAdapter(
            name=name,
            min_level=min_level,
            formatter=formatter,
            buffer_size=cfg.get("buffer_size", 1000),
            flush_interval_seconds=cfg.get("flush_interval_seconds", 5.0),
            backtest_throttle=_resolve_level(cfg.get("backtest_throttle", LogLevel.WARNING)),
        )
    elif adapter_type == "agent":
        return AgentAdapter(
            name=name,
            min_level=min_level,
            formatter=formatter,
            ring_buffer_size=cfg.get("ring_buffer_size", 10000),
        )
    else:
        raise ValueError(f"Unknown adapter type '{adapter_type}' for adapter '{name}'")
