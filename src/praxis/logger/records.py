"""
Log records and level definitions.

§18.2: Standard levels (Python-compatible values).
§18.9: Custom levels slot into the standard hierarchy.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any


class LogLevel(IntEnum):
    """Standard log levels, Python-compatible numeric values."""
    TRACE = 5        # §18.9: Below DEBUG — extreme detail
    DEBUG = 10
    VERBOSE = 15     # §18.9: Between DEBUG and INFO
    INFO = 20
    NOTICE = 25      # §18.9: Between INFO and WARNING
    WARNING = 30
    ERROR = 40
    ALERT = 45       # §18.9: Between ERROR and CRITICAL
    CRITICAL = 50

    @classmethod
    def from_name(cls, name: str) -> "LogLevel":
        """Resolve level from string name, case-insensitive."""
        name_upper = name.upper()
        try:
            return cls[name_upper]
        except KeyError:
            raise ValueError(
                f"Unknown log level '{name}'. "
                f"Valid levels: {', '.join(m.name for m in cls)}"
            )

    @classmethod
    def from_value(cls, value: int | str) -> "LogLevel":
        """Resolve level from int or string."""
        if isinstance(value, str):
            return cls.from_name(value)
        if isinstance(value, int):
            # Try exact match first
            for member in cls:
                if member.value == value:
                    return member
            # Accept arbitrary int as custom level
            raise ValueError(
                f"No standard level with value {value}. "
                f"Valid values: {', '.join(f'{m.name}={m.value}' for m in cls)}"
            )
        raise TypeError(f"Expected int or str, got {type(value).__name__}")


# Map for display: level int → name string
LEVEL_NAMES: dict[int, str] = {member.value: member.name for member in LogLevel}


def level_name(level: int) -> str:
    """Get display name for a level value. Falls back to numeric string."""
    return LEVEL_NAMES.get(level, str(level))


@dataclass(frozen=True)
class LogRecord:
    """
    Immutable log record. Created by PraxisLogger.log(), routed to adapters.

    §18.2: Contains timestamp, level, message, tags, and arbitrary context.
    §18.8: Context includes module, function, model_name, security_bpk, etc.
    """
    timestamp: datetime
    level: int
    level_name: str
    message: str
    tags: frozenset[str] = field(default_factory=frozenset)
    context: dict[str, Any] = field(default_factory=dict)
    source: str = "system"       # 'user', 'agent', 'system', 'scheduler'
    session_id: str | None = None

    @classmethod
    def create(
        cls,
        level: int,
        message: str,
        tags: set[str] | frozenset[str] | None = None,
        source: str = "system",
        session_id: str | None = None,
        **context: Any,
    ) -> "LogRecord":
        """Factory method with auto-timestamp and level name resolution."""
        return cls(
            timestamp=datetime.now(timezone.utc),
            level=level,
            level_name=level_name(level),
            message=message,
            tags=frozenset(tags) if tags else frozenset(),
            context=context,
            source=source,
            session_id=session_id,
        )
