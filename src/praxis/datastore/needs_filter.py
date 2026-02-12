"""
Needs Filter — Idempotent Loading (Phase 2.6, §19.3).

Core principle: Never reload data you already have validated for the current period.

Queries ldr_*_hist and fact_price_daily to determine the delta—
what securities × dates actually need loading vs what's already present.

Usage:
    nf = NeedsFilter(conn)
    needs = nf.compute_price_needs(
        securities=[base_id_1, base_id_2],
        start="2024-01-01",
        end="2024-03-31",
    )
    # needs = [(base_id_1, date(2024, 3, 15)), ...] only missing dates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import duckdb

from praxis.logger.core import PraxisLogger


@dataclass
class DataNeed:
    """A single needed data point."""
    security_base_id: int
    trade_date: date
    source: str | None = None


@dataclass
class NeedsResult:
    """Result of needs computation."""
    total_required: int = 0
    already_loaded: int = 0
    delta: int = 0
    needs: list[DataNeed] = field(default_factory=list)

    @property
    def savings_pct(self) -> float:
        if self.total_required == 0:
            return 0.0
        return round(self.already_loaded / self.total_required * 100, 1)


class NeedsFilter:
    """
    §19.3: Idempotent loading — compute delta before fetching.

    Prevents star-schema bloat from phantom _hist_id versions
    of identical data.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._log = PraxisLogger.instance()

    def compute_price_needs(
        self,
        securities: list[int],
        start: str | date,
        end: str | date,
        source: str | None = None,
    ) -> NeedsResult:
        """
        Compute which (security, date) pairs are missing from fact_price_daily.

        Args:
            securities: List of security_base_id values.
            start: Start date of required range.
            end: End date of required range.

        Returns:
            NeedsResult with delta (only what needs loading).
        """
        result = NeedsResult()

        if not securities:
            return result

        # Get all business days in range from dim_calendar
        # Fall back to weekdays if calendar not populated
        try:
            bday_rows = self._conn.execute("""
                SELECT calendar_date FROM dim_calendar
                WHERE is_business_day = TRUE
                  AND calendar_date >= $1::DATE
                  AND calendar_date <= $2::DATE
                ORDER BY calendar_date
            """, [str(start), str(end)]).fetchall()
            bdays = [r[0] for r in bday_rows]
        except Exception:
            # Calendar not populated — generate weekdays
            from datetime import timedelta
            d = date.fromisoformat(str(start)) if isinstance(start, str) else start
            e = date.fromisoformat(str(end)) if isinstance(end, str) else end
            bdays = []
            while d <= e:
                if d.weekday() < 5:
                    bdays.append(d)
                d += timedelta(days=1)

        if not bdays:
            return result

        # Build required set: all (security, date) pairs
        required = set()
        for sec_id in securities:
            for d in bdays:
                required.add((sec_id, d))

        result.total_required = len(required)

        # Query what's already in fact_price_daily
        placeholders = ",".join(str(s) for s in securities)
        already = self._conn.execute(f"""
            SELECT security_base_id, trade_date
            FROM fact_price_daily
            WHERE security_base_id IN ({placeholders})
              AND trade_date >= $1::DATE
              AND trade_date <= $2::DATE
        """, [str(start), str(end)]).fetchall()

        loaded = {(r[0], r[1]) for r in already}
        result.already_loaded = len(loaded & required)

        # Compute delta
        missing = required - loaded
        result.delta = len(missing)
        result.needs = [
            DataNeed(security_base_id=sec_id, trade_date=d)
            for sec_id, d in sorted(missing)
        ]

        if result.already_loaded > 0:
            self._log.info(
                f"NeedsFilter: {result.already_loaded}/{result.total_required} "
                f"already loaded ({result.savings_pct}% savings), "
                f"delta={result.delta}",
                tags={"data_pipeline"},
            )

        return result

    def compute_loader_needs(
        self,
        securities: list[int],
        start: str | date,
        end: str | date,
    ) -> NeedsResult:
        """
        Check what's already been loaded in ldr_yfinance_hist
        (even if not yet in fact_price_daily).
        """
        result = self.compute_price_needs(securities, start, end)

        if not result.needs:
            return result

        # Also check ldr_yfinance_hist for in-progress loads
        sec_ids = list({n.security_base_id for n in result.needs})
        placeholders = ",".join(str(s) for s in sec_ids)

        try:
            hist_rows = self._conn.execute(f"""
                SELECT security_base_id, date
                FROM ldr_yfinance_hist
                WHERE security_base_id IN ({placeholders})
                  AND date >= $1::DATE
                  AND date <= $2::DATE
                  AND process_status IN ('matched', 'created')
            """, [str(start), str(end)]).fetchall()

            hist_loaded = {(r[0], r[1]) for r in hist_rows}
            remaining = [
                n for n in result.needs
                if (n.security_base_id, n.trade_date) not in hist_loaded
            ]

            additional_savings = len(result.needs) - len(remaining)
            result.already_loaded += additional_savings
            result.delta = len(remaining)
            result.needs = remaining
        except Exception:
            pass  # ldr_yfinance_hist may not exist

        return result
