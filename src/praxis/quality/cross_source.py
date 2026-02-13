"""
Cross-Source Data Quality (Phase 3.12, §2.8 Layer 2).

Compare prices from two data sources for the same security to
detect discrepancies, stale data, and source reliability issues.

Usage:
    result = cross_source_check(yf_prices, poly_prices, tolerance=0.01)
    if not result.passed:
        for issue in result.issues:
            print(f"  {issue.field}: diff={issue.max_diff:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from praxis.logger.core import PraxisLogger


@dataclass
class FieldComparison:
    """Comparison result for one OHLCV field."""
    field: str
    n_compared: int
    n_mismatches: int
    max_diff: float
    mean_diff: float
    max_pct_diff: float
    mean_pct_diff: float
    passed: bool


@dataclass
class CrossSourceResult:
    """Result of comparing two data sources."""
    ticker: str
    source_a: str
    source_b: str
    n_bars: int
    n_overlapping: int
    fields: list[FieldComparison] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    passed: bool = True

    def add_field(self, fc: FieldComparison):
        self.fields.append(fc)
        if not fc.passed:
            self.passed = False
            self.issues.append(
                f"{fc.field}: {fc.n_mismatches}/{fc.n_compared} mismatches, "
                f"max_pct={fc.max_pct_diff:.4%}"
            )


def cross_source_check(
    prices_a: dict[str, np.ndarray],
    prices_b: dict[str, np.ndarray],
    ticker: str = "",
    source_a: str = "source_a",
    source_b: str = "source_b",
    tolerance_pct: float = 0.01,
    fields: list[str] | None = None,
) -> CrossSourceResult:
    """
    Compare OHLCV prices from two sources.

    Args:
        prices_a: Dict of field → numpy array from source A.
        prices_b: Dict of field → numpy array from source B.
        ticker: Ticker symbol for reporting.
        source_a: Name of source A.
        source_b: Name of source B.
        tolerance_pct: Max allowed percentage difference (0.01 = 1%).
        fields: Which fields to compare (default: close, open, high, low, volume).

    Returns:
        CrossSourceResult with per-field comparisons.
    """
    log = PraxisLogger.instance()

    if fields is None:
        fields = ["close", "open", "high", "low", "volume"]

    # Determine overlap length
    n_a = min(len(v) for v in prices_a.values()) if prices_a else 0
    n_b = min(len(v) for v in prices_b.values()) if prices_b else 0
    n_overlap = min(n_a, n_b)

    result = CrossSourceResult(
        ticker=ticker,
        source_a=source_a,
        source_b=source_b,
        n_bars=max(n_a, n_b),
        n_overlapping=n_overlap,
    )

    if n_overlap == 0:
        result.passed = False
        result.issues.append("No overlapping bars between sources")
        return result

    for fld in fields:
        if fld not in prices_a or fld not in prices_b:
            continue

        a = np.asarray(prices_a[fld][:n_overlap], dtype=np.float64)
        b = np.asarray(prices_b[fld][:n_overlap], dtype=np.float64)

        # Absolute difference
        abs_diff = np.abs(a - b)

        # Percentage difference (relative to source A, avoiding div/0)
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = np.where(np.abs(a) > 1e-10, np.abs(a), 1.0)
            pct_diff = abs_diff / denom

        # Volume uses absolute threshold instead of percentage
        if fld == "volume":
            # Volume: allow 5% difference
            vol_tol = 0.05
            mismatches = int(np.sum(pct_diff > vol_tol))
            fc = FieldComparison(
                field=fld,
                n_compared=n_overlap,
                n_mismatches=mismatches,
                max_diff=float(np.max(abs_diff)),
                mean_diff=float(np.mean(abs_diff)),
                max_pct_diff=float(np.max(pct_diff)),
                mean_pct_diff=float(np.mean(pct_diff)),
                passed=mismatches == 0,
            )
        else:
            mismatches = int(np.sum(pct_diff > tolerance_pct))
            fc = FieldComparison(
                field=fld,
                n_compared=n_overlap,
                n_mismatches=mismatches,
                max_diff=float(np.max(abs_diff)),
                mean_diff=float(np.mean(abs_diff)),
                max_pct_diff=float(np.max(pct_diff)),
                mean_pct_diff=float(np.mean(pct_diff)),
                passed=mismatches == 0,
            )

        result.add_field(fc)

    # Staleness check: if one source has fewer bars
    if n_a != n_b:
        shorter = source_a if n_a < n_b else source_b
        diff = abs(n_a - n_b)
        result.issues.append(
            f"Bar count mismatch: {source_a}={n_a}, {source_b}={n_b} "
            f"({shorter} is {diff} bars behind)"
        )

    log.info(
        f"Cross-source check {ticker}: {source_a} vs {source_b}, "
        f"{n_overlap} bars, {'PASS' if result.passed else 'FAIL'}",
        tags={"data_quality", "cross_source"},
    )

    return result


def detect_stale_source(
    prices: dict[str, np.ndarray],
    source_name: str,
    max_constant_bars: int = 10,
) -> list[str]:
    """
    Detect if a source has stale/frozen data.

    Checks for runs of identical close prices that exceed threshold.
    """
    issues = []
    close = prices.get("close")
    if close is None or len(close) < 2:
        return issues

    close = np.asarray(close, dtype=np.float64)
    diffs = np.diff(close)
    zero_runs = 0
    max_run = 0

    for d in diffs:
        if d == 0:
            zero_runs += 1
            max_run = max(max_run, zero_runs)
        else:
            zero_runs = 0

    if max_run >= max_constant_bars:
        issues.append(
            f"{source_name}: {max_run} consecutive identical close prices "
            f"(stale data suspected)"
        )

    return issues
