"""
Data Quality Framework.

Layer 1: Source validation (Phase 2.8) — in data module
Layer 2: Cross-source reconciliation (Phase 3.12) — this package
"""
from praxis.quality.cross_source import (
    CrossSourceResult,
    FieldComparison,
    cross_source_check,
    detect_stale_source,
)

__all__ = [
    "CrossSourceResult",
    "FieldComparison",
    "cross_source_check",
    "detect_stale_source",
]
