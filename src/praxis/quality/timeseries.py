"""
Time-Series Quality (Phase 4.11, §4 Layer 3).

Daily quality checks for price/volume time series:
- Gap detection: missing trading days
- Stale data: price unchanged for N days
- Outlier detection: returns > N standard deviations
- Corporate action: split/dividend detection via price jumps
- Completeness scoring: overall quality score 0-1

Usage:
    checker = TimeSeriesQuality()
    result = checker.check(prices, dates, calendar=nyse_dates)
    print(result.quality_score, result.gaps, result.outliers)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GapInfo:
    """A detected gap in the time series."""
    start_date: date | int      # Start of gap (date or index)
    end_date: date | int        # End of gap
    missing_days: int           # Number of missing trading days
    severity: str = "warning"   # warning, error


@dataclass
class OutlierInfo:
    """A detected outlier."""
    index: int
    date: date | int | None = None
    value: float = 0.0
    return_pct: float = 0.0
    z_score: float = 0.0
    severity: str = "warning"


@dataclass
class StaleInfo:
    """A stale data period."""
    start_index: int = 0
    end_index: int = 0
    n_unchanged: int = 0
    value: float = 0.0
    severity: str = "warning"


@dataclass
class CorporateActionInfo:
    """A suspected corporate action."""
    index: int = 0
    date: date | int | None = None
    price_before: float = 0.0
    price_after: float = 0.0
    ratio: float = 0.0
    action_type: str = ""      # "split", "reverse_split", "dividend"
    confidence: str = "medium"  # low, medium, high


@dataclass
class TimeSeriesQualityResult:
    """Result of time-series quality checks."""
    n_observations: int = 0
    gaps: list[GapInfo] = field(default_factory=list)
    outliers: list[OutlierInfo] = field(default_factory=list)
    stale_periods: list[StaleInfo] = field(default_factory=list)
    corporate_actions: list[CorporateActionInfo] = field(default_factory=list)

    # Scores
    gap_score: float = 1.0         # 1.0 = no gaps
    outlier_score: float = 1.0     # 1.0 = no outliers
    stale_score: float = 1.0       # 1.0 = no stale data
    quality_score: float = 1.0     # Composite 0-1

    # Counts for fact_data_quality
    gaps_detected: int = 0
    outliers_detected: int = 0
    stale_detected: int = 0

    @property
    def is_clean(self) -> bool:
        return self.quality_score >= 0.95

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "quality_score": round(self.quality_score, 4),
            "gaps": self.gaps_detected,
            "outliers": self.outliers_detected,
            "stale_periods": self.stale_detected,
            "corporate_actions": len(self.corporate_actions),
        }


# ═══════════════════════════════════════════════════════════════════
#  Checker
# ═══════════════════════════════════════════════════════════════════

class TimeSeriesQuality:
    """
    Time-series quality checker (Layer 3).

    Runs gap, stale, outlier, and corporate action checks.
    """

    def __init__(
        self,
        outlier_std: float = 5.0,
        stale_threshold: int = 5,
        gap_severity_threshold: int = 3,
        split_ratio_tolerance: float = 0.02,
        volatility_lookback: int = 60,
    ):
        self.outlier_std = outlier_std
        self.stale_threshold = stale_threshold
        self.gap_severity_threshold = gap_severity_threshold
        self.split_ratio_tolerance = split_ratio_tolerance
        self.volatility_lookback = volatility_lookback

    def check(
        self,
        prices: np.ndarray,
        dates: list[date] | np.ndarray | None = None,
        calendar: set[date] | None = None,
        volumes: np.ndarray | None = None,
    ) -> TimeSeriesQualityResult:
        """
        Run all time-series quality checks.

        Args:
            prices: 1D price array.
            dates: Optional date array aligned with prices.
            calendar: Optional set of expected trading dates.
            volumes: Optional volume array for additional checks.

        Returns:
            TimeSeriesQualityResult with all findings.
        """
        prices = np.asarray(prices, dtype=float).ravel()
        result = TimeSeriesQualityResult(n_observations=len(prices))

        if len(prices) < 2:
            return result

        # Run checks
        result.gaps = self._detect_gaps(prices, dates, calendar)
        result.gaps_detected = len(result.gaps)

        result.outliers = self._detect_outliers(prices, dates)
        result.outliers_detected = len(result.outliers)

        result.stale_periods = self._detect_stale(prices, volumes)
        result.stale_detected = len(result.stale_periods)

        result.corporate_actions = self._detect_corporate_actions(prices, dates)

        # Compute scores
        n = len(prices)
        result.gap_score = max(0, 1.0 - result.gaps_detected * 0.05)
        result.outlier_score = max(0, 1.0 - result.outliers_detected / max(n, 1) * 10)
        result.stale_score = max(0, 1.0 - sum(s.n_unchanged for s in result.stale_periods) / max(n, 1))

        # Composite: weighted average
        result.quality_score = (
            0.4 * result.gap_score +
            0.3 * result.outlier_score +
            0.3 * result.stale_score
        )

        return result

    def _detect_gaps(
        self,
        prices: np.ndarray,
        dates: list[date] | np.ndarray | None,
        calendar: set[date] | None,
    ) -> list[GapInfo]:
        """Detect missing trading days."""
        gaps = []

        if dates is None or calendar is None:
            return gaps

        date_set = set(dates) if not isinstance(dates, set) else dates
        sorted_dates = sorted(dates)

        if len(sorted_dates) < 2:
            return gaps

        # Find calendar dates within our range that are missing
        min_d = sorted_dates[0]
        max_d = sorted_dates[-1]
        expected = {d for d in calendar if min_d <= d <= max_d}
        missing = sorted(expected - date_set)

        if not missing:
            return gaps

        # Group consecutive missing dates
        current_start = missing[0]
        current_end = missing[0]
        count = 1

        for d in missing[1:]:
            # Check if next calendar day (simplified: within 4 days)
            delta = (d - current_end).days
            if delta <= 4:
                current_end = d
                count += 1
            else:
                severity = "error" if count >= self.gap_severity_threshold else "warning"
                gaps.append(GapInfo(
                    start_date=current_start,
                    end_date=current_end,
                    missing_days=count,
                    severity=severity,
                ))
                current_start = d
                current_end = d
                count = 1

        # Final group
        severity = "error" if count >= self.gap_severity_threshold else "warning"
        gaps.append(GapInfo(
            start_date=current_start,
            end_date=current_end,
            missing_days=count,
            severity=severity,
        ))

        return gaps

    def _detect_outliers(
        self,
        prices: np.ndarray,
        dates: list[date] | np.ndarray | None,
    ) -> list[OutlierInfo]:
        """Detect returns exceeding N standard deviations."""
        outliers = []
        n = len(prices)

        if n < self.volatility_lookback + 1:
            # Not enough data for rolling vol
            returns = np.diff(prices) / np.maximum(prices[:-1], 1e-10)
            if len(returns) < 10:
                return outliers
            mu = np.mean(returns)
            sigma = np.std(returns)
            if sigma < 1e-10:
                return outliers

            for i in range(len(returns)):
                z = (returns[i] - mu) / sigma
                if abs(z) > self.outlier_std:
                    outliers.append(OutlierInfo(
                        index=i + 1,
                        date=dates[i + 1] if dates is not None and i + 1 < len(dates) else None,
                        value=prices[i + 1],
                        return_pct=returns[i],
                        z_score=z,
                        severity="error" if abs(z) > self.outlier_std * 1.5 else "warning",
                    ))
            return outliers

        # Rolling window outlier detection
        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-10)
        for i in range(self.volatility_lookback, len(returns)):
            window = returns[i - self.volatility_lookback:i]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma < 1e-10:
                continue
            z = (returns[i] - mu) / sigma
            if abs(z) > self.outlier_std:
                outliers.append(OutlierInfo(
                    index=i + 1,
                    date=dates[i + 1] if dates is not None and i + 1 < len(dates) else None,
                    value=prices[i + 1],
                    return_pct=returns[i],
                    z_score=z,
                    severity="error" if abs(z) > self.outlier_std * 1.5 else "warning",
                ))

        return outliers

    def _detect_stale(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None,
    ) -> list[StaleInfo]:
        """Detect periods where price is unchanged."""
        stale = []
        n = len(prices)
        i = 0

        while i < n:
            # Count consecutive unchanged prices
            j = i + 1
            while j < n and abs(prices[j] - prices[i]) < 1e-10:
                j += 1

            run_length = j - i
            if run_length >= self.stale_threshold:
                # Check if volume is also zero (more suspicious)
                severity = "warning"
                if volumes is not None:
                    vol_slice = volumes[i:j]
                    if np.all(vol_slice < 1e-10):
                        severity = "error"

                stale.append(StaleInfo(
                    start_index=i,
                    end_index=j - 1,
                    n_unchanged=run_length,
                    value=prices[i],
                    severity=severity,
                ))
            i = j

        return stale

    def _detect_corporate_actions(
        self,
        prices: np.ndarray,
        dates: list[date] | np.ndarray | None,
    ) -> list[CorporateActionInfo]:
        """Detect suspected stock splits and reverse splits via price ratios."""
        actions = []
        common_ratios = [2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 1/3, 0.25, 0.2, 0.1]

        for i in range(1, len(prices)):
            if prices[i - 1] <= 0:
                continue

            ratio = prices[i] / prices[i - 1]

            # Skip normal moves
            if 0.7 < ratio < 1.4:
                continue

            # Check against common split ratios
            for cr in common_ratios:
                if abs(ratio - cr) / cr < self.split_ratio_tolerance:
                    if cr > 1:
                        action_type = "reverse_split"
                    else:
                        action_type = "split"

                    actions.append(CorporateActionInfo(
                        index=i,
                        date=dates[i] if dates is not None and i < len(dates) else None,
                        price_before=prices[i - 1],
                        price_after=prices[i],
                        ratio=ratio,
                        action_type=action_type,
                        confidence="high",
                    ))
                    break
            else:
                # Large move but not a clean ratio — could be special dividend or data error
                if ratio < 0.7 or ratio > 1.5:
                    actions.append(CorporateActionInfo(
                        index=i,
                        date=dates[i] if dates is not None and i < len(dates) else None,
                        price_before=prices[i - 1],
                        price_after=prices[i],
                        ratio=ratio,
                        action_type="unknown_large_move",
                        confidence="low",
                    ))

        return actions
