"""
Log adapters (output destinations).

§18.5: One logger, multiple adapters. Each adapter receives log records
according to the routing matrix.

Phase 1.2 scope: Terminal + File + Database (buffered) + Agent (ring buffer).
Database adapter buffers records and writes when flush() is called or
buffer is full. DuckDB connection is injected later (Phase 1.3).
"""

import sys
import os
import threading
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from praxis.logger.records import LogRecord
from praxis.logger.formatters import (
    LogFormatter,
    CompactFormatter,
    DetailedFormatter,
    JsonFormatter,
)


class LogAdapter(ABC):
    """Base adapter. Receives formatted log records."""

    def __init__(self, name: str, min_level: int = 20, formatter: LogFormatter | None = None):
        self.name = name
        self.min_level = min_level
        self._formatter = formatter

    @property
    def formatter(self) -> LogFormatter:
        if self._formatter is None:
            self._formatter = self._default_formatter()
        return self._formatter

    @formatter.setter
    def formatter(self, value: LogFormatter) -> None:
        self._formatter = value

    def _default_formatter(self) -> LogFormatter:
        """Subclass-specific default."""
        return CompactFormatter()

    @abstractmethod
    def emit(self, record: LogRecord) -> None:
        """Write a log record. Called only after routing filter passes."""
        ...

    def flush(self) -> None:
        """Flush any buffered records. Override in buffered adapters."""
        pass

    def close(self) -> None:
        """Cleanup. Override if adapter holds resources."""
        self.flush()


class TerminalAdapter(LogAdapter):
    """
    §18.5: Writes to stdout/stderr with ANSI color coding.
    ERROR+ goes to stderr, everything else to stdout.
    """

    COLORS = {
        5: "\033[90m",       # TRACE: gray
        10: "\033[36m",      # DEBUG: cyan
        15: "\033[96m",      # VERBOSE: bright cyan
        20: "\033[37m",      # INFO: white/default
        25: "\033[97m",      # NOTICE: bright white
        30: "\033[33m",      # WARNING: yellow
        40: "\033[31m",      # ERROR: red
        45: "\033[91m",      # ALERT: bright red
        50: "\033[1;91m",    # CRITICAL: bold bright red
    }
    RESET = "\033[0m"

    def __init__(
        self,
        name: str = "terminal",
        min_level: int = 20,
        formatter: LogFormatter | None = None,
        color: bool = True,
    ):
        super().__init__(name, min_level, formatter)
        self.color = color

    def _default_formatter(self) -> LogFormatter:
        return CompactFormatter()

    def emit(self, record: LogRecord) -> None:
        formatted = self.formatter.format(record)
        if self.color:
            color = self._get_color(record.level)
            formatted = f"{color}{formatted}{self.RESET}"
        stream = sys.stderr if record.level >= 40 else sys.stdout
        print(formatted, file=stream, flush=True)

    def _get_color(self, level: int) -> str:
        """Get ANSI color for level, falling back to nearest lower level."""
        if level in self.COLORS:
            return self.COLORS[level]
        # Find nearest lower standard level
        for threshold in sorted(self.COLORS.keys(), reverse=True):
            if level >= threshold:
                return self.COLORS[threshold]
        return ""


class FileAdapter(LogAdapter):
    """
    §18.5: Writes to rotating .log files.
    Simple daily rotation: when date changes, opens new file.
    """

    def __init__(
        self,
        name: str = "logfile",
        min_level: int = 10,
        formatter: LogFormatter | None = None,
        path: str | Path = "logs/praxis.log",
        rotation: str = "daily",
        retention_days: int = 30,
    ):
        super().__init__(name, min_level, formatter)
        self.base_path = Path(path)
        self.rotation = rotation
        self.retention_days = retention_days
        self._current_date: Optional[str] = None
        self._file = None
        self._lock = threading.Lock()

    def _default_formatter(self) -> LogFormatter:
        return DetailedFormatter()

    def _ensure_file(self, record_date: datetime) -> None:
        """Open or rotate file as needed."""
        date_str = record_date.strftime("%Y-%m-%d")
        if self._current_date == date_str and self._file is not None:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._current_date == date_str and self._file is not None:
                return

            if self._file is not None:
                self._file.close()

            self.base_path.parent.mkdir(parents=True, exist_ok=True)

            if self.rotation == "daily":
                stem = self.base_path.stem
                suffix = self.base_path.suffix or ".log"
                file_path = self.base_path.parent / f"{stem}_{date_str}{suffix}"
            else:
                file_path = self.base_path

            self._file = open(file_path, "a", encoding="utf-8")
            self._current_date = date_str

    def emit(self, record: LogRecord) -> None:
        self._ensure_file(record.timestamp)
        formatted = self.formatter.format(record)
        with self._lock:
            if self._file:
                self._file.write(formatted + "\n")

    def flush(self) -> None:
        with self._lock:
            if self._file:
                self._file.flush()

    def close(self) -> None:
        self.flush()
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
                self._current_date = None

    def cleanup_old_files(self) -> int:
        """Remove log files older than retention_days. Returns count removed."""
        if not self.base_path.parent.exists():
            return 0

        cutoff = datetime.now(timezone.utc).timestamp() - (self.retention_days * 86400)
        removed = 0
        stem = self.base_path.stem
        suffix = self.base_path.suffix or ".log"

        for f in self.base_path.parent.glob(f"{stem}_*{suffix}"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1

        return removed


class DatabaseAdapter(LogAdapter):
    """
    §18.5: Writes to fact_log in DuckDB.
    Batch-buffered with configurable flush interval to avoid performance
    impact during hot paths.

    Phase 1.2: Buffer records. DuckDB connection injected via connect().
    Phase 1.3: fact_log table created, connection wired up.
    """

    def __init__(
        self,
        name: str = "database",
        min_level: int = 20,
        formatter: LogFormatter | None = None,
        buffer_size: int = 1000,
        flush_interval_seconds: float = 5.0,
        backtest_throttle: int = 30,
    ):
        super().__init__(name, min_level, formatter)
        self.buffer_size = buffer_size
        self.flush_interval_seconds = flush_interval_seconds
        self.backtest_throttle = backtest_throttle
        self._buffer: list[LogRecord] = []
        self._connection = None  # DuckDB connection, injected later
        self._lock = threading.Lock()
        self._in_backtest = False
        self._seq = 0  # Monotonic log_id sequence

    def _default_formatter(self) -> LogFormatter:
        return JsonFormatter()

    @property
    def effective_min_level(self) -> int:
        """During backtests, throttle to backtest_throttle level."""
        return self.backtest_throttle if self._in_backtest else self.min_level

    def set_backtest_mode(self, active: bool) -> None:
        """Toggle backtest throttling. §18.6: WARNING+ during backtests."""
        self._in_backtest = active

    def connect(self, connection) -> None:
        """Inject DuckDB connection. Called during Phase 1.3 initialization."""
        self._connection = connection
        self.flush()  # Write any buffered records

    def emit(self, record: LogRecord) -> None:
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.buffer_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush buffer to database. Must hold self._lock."""
        if not self._buffer or self._connection is None:
            return

        try:
            for record in self._buffer:
                self._seq += 1
                self._connection.execute(
                    """
                    INSERT INTO fact_log (
                        log_id, log_timestamp, level, level_name, message, tags,
                        context, source, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?::JSON, ?, ?)
                    """,
                    [
                        self._seq,
                        record.timestamp,
                        record.level,
                        record.level_name,
                        record.message,
                        sorted(record.tags),
                        self.formatter.format(record) if record.context else "{}",
                        record.source,
                        record.session_id,
                    ],
                )
            self._buffer.clear()
        except Exception:
            # Don't lose records on DB failure — keep in buffer
            pass

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)

    def close(self) -> None:
        self.flush()


class AgentAdapter(LogAdapter):
    """
    §18.5: Ring buffer of last N entries.
    The agent's dedicated diagnostic window. Does not grow unbounded.
    """

    def __init__(
        self,
        name: str = "agent",
        min_level: int = 10,
        formatter: LogFormatter | None = None,
        ring_buffer_size: int = 10000,
    ):
        super().__init__(name, min_level, formatter)
        self._buffer: deque[LogRecord] = deque(maxlen=ring_buffer_size)
        self._lock = threading.Lock()

    def _default_formatter(self) -> LogFormatter:
        return DetailedFormatter()

    def emit(self, record: LogRecord) -> None:
        with self._lock:
            self._buffer.append(record)

    def get_recent(self, n: int = 100, tags: set[str] | None = None) -> list[LogRecord]:
        """
        Get recent log records, optionally filtered by tags.
        §18.7: Agent reads from ring buffer during troubleshooting.
        """
        with self._lock:
            records = list(self._buffer)

        if tags:
            records = [r for r in records if r.tags & tags]

        return records[-n:]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def count(self) -> int:
        return len(self._buffer)
