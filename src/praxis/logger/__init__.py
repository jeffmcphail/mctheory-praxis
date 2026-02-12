"""
McTheory Praxis Diagnostic Logger (§18)

Not a logging library — a diagnostic instrumentation system.
Singleton with routing matrix, tag-as-trace architecture.
"""

from praxis.logger.core import PraxisLogger
from praxis.logger.records import LogRecord, LogLevel
from praxis.logger.adapters import (
    LogAdapter,
    TerminalAdapter,
    FileAdapter,
    DatabaseAdapter,
    AgentAdapter,
)
from praxis.logger.routing import RoutingMatrix, RoutingRule
from praxis.logger.formatters import LogFormatter, CompactFormatter, DetailedFormatter, JsonFormatter

__all__ = [
    "PraxisLogger",
    "LogRecord",
    "LogLevel",
    "LogAdapter",
    "TerminalAdapter",
    "FileAdapter",
    "DatabaseAdapter",
    "AgentAdapter",
    "RoutingMatrix",
    "RoutingRule",
    "LogFormatter",
    "CompactFormatter",
    "DetailedFormatter",
    "JsonFormatter",
]
