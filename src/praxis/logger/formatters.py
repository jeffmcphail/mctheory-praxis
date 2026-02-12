"""
Log formatters.

§18.10: Each adapter can use a different formatter.
  - compact:  "{timestamp:%H:%M:%S} [{level_name:>8}] {message}"
  - detailed: "{timestamp} [{level_name}] [{tags}] {module}.{function}: {message}"
  - json:     Structured JSON for machine parsing
"""

import json
from abc import ABC, abstractmethod
from typing import Any

from praxis.logger.records import LogRecord


class LogFormatter(ABC):
    """Base formatter. Transforms LogRecord → string."""

    @abstractmethod
    def format(self, record: LogRecord) -> str: ...


class CompactFormatter(LogFormatter):
    """
    Compact single-line format for terminal display.
    Example: 14:32:05 [    INFO] Security matched
    """

    def format(self, record: LogRecord) -> str:
        ts = record.timestamp.strftime("%H:%M:%S")
        return f"{ts} [{record.level_name:>8}] {record.message}"


class DetailedFormatter(LogFormatter):
    """
    Detailed format with tags and context for file/agent adapters.
    Example: 2026-02-12 14:32:05.123456 [    INFO] [datastore.loader,data_pipeline] loader.fetch: Fetching GLD
    """

    def format(self, record: LogRecord) -> str:
        ts = record.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
        tags_str = ",".join(sorted(record.tags)) if record.tags else "-"
        module = record.context.get("module", "")
        function = record.context.get("function", "")
        loc = f"{module}.{function}" if module and function else module or function or ""

        parts = [f"{ts} [{record.level_name:>8}] [{tags_str}]"]
        if loc:
            parts.append(f"{loc}:")
        parts.append(record.message)

        # Append extra context (excluding module/function which are in the location)
        extras = {
            k: v for k, v in record.context.items()
            if k not in ("module", "function") and v is not None
        }
        if extras:
            extras_str = " ".join(f"{k}={_format_value(v)}" for k, v in extras.items())
            parts.append(f"| {extras_str}")

        return " ".join(parts)


class JsonFormatter(LogFormatter):
    """
    Structured JSON for database adapter and machine parsing.
    One JSON object per line.
    """

    def format(self, record: LogRecord) -> str:
        obj: dict[str, Any] = {
            "timestamp": record.timestamp.isoformat(),
            "level": record.level,
            "level_name": record.level_name,
            "message": record.message,
            "tags": sorted(record.tags) if record.tags else [],
            "source": record.source,
        }
        if record.session_id:
            obj["session_id"] = record.session_id
        if record.context:
            obj["context"] = {
                k: _serialize_value(v) for k, v in record.context.items()
            }
        return json.dumps(obj, default=str)


def _format_value(v: Any) -> str:
    """Format a context value for detailed display."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _serialize_value(v: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialize_value(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _serialize_value(val) for k, val in v.items()}
    return str(v)
