"""
Calendar infrastructure (Phase 2.7).

Populates dim_calendar with business day information.
Used for ved_ expansion, gap detection, and scheduling.

Usage:
    cal = Calendar(conn)
    cal.populate(2020, 2026, exchange="NYSE")
    bdays = cal.business_days("2024-01-01", "2024-01-31")
    is_bd = cal.is_business_day("2024-01-15")
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import duckdb

from praxis.datastore.keys import EntityKeys


# ── US Market Holidays (simplified, extensible) ───────────────────

def _us_market_holidays(year: int) -> dict[date, str]:
    """
    US market holidays for a given year.
    Covers NYSE/NASDAQ observed holidays.
    """
    holidays = {}

    # New Year's Day (Jan 1, or observed)
    nyd = date(year, 1, 1)
    if nyd.weekday() == 6:  # Sunday → Monday
        holidays[date(year, 1, 2)] = "New Year's Day (observed)"
    elif nyd.weekday() == 5:  # Saturday → previous Friday
        holidays[date(year - 1, 12, 31)] = "New Year's Day (observed)"
    else:
        holidays[nyd] = "New Year's Day"

    # MLK Day (3rd Monday of January)
    d = date(year, 1, 1)
    mondays = 0
    while mondays < 3:
        d += timedelta(days=1)
        if d.weekday() == 0:
            mondays += 1
    holidays[d] = "MLK Day"

    # Presidents' Day (3rd Monday of February)
    d = date(year, 2, 1)
    mondays = 0
    while mondays < 3:
        if d.weekday() == 0:
            mondays += 1
        if mondays < 3:
            d += timedelta(days=1)
    holidays[d] = "Presidents' Day"

    # Good Friday (approximate — 2 days before Easter)
    easter = _easter(year)
    holidays[easter - timedelta(days=2)] = "Good Friday"

    # Memorial Day (last Monday of May)
    d = date(year, 5, 31)
    while d.weekday() != 0:
        d -= timedelta(days=1)
    holidays[d] = "Memorial Day"

    # Juneteenth (June 19)
    jt = date(year, 6, 19)
    if jt.weekday() == 6:
        holidays[date(year, 6, 20)] = "Juneteenth (observed)"
    elif jt.weekday() == 5:
        holidays[date(year, 6, 18)] = "Juneteenth (observed)"
    else:
        holidays[jt] = "Juneteenth"

    # Independence Day (July 4)
    july4 = date(year, 7, 4)
    if july4.weekday() == 6:
        holidays[date(year, 7, 5)] = "Independence Day (observed)"
    elif july4.weekday() == 5:
        holidays[date(year, 7, 3)] = "Independence Day (observed)"
    else:
        holidays[july4] = "Independence Day"

    # Labor Day (1st Monday of September)
    d = date(year, 9, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    holidays[d] = "Labor Day"

    # Thanksgiving (4th Thursday of November)
    d = date(year, 11, 1)
    thursdays = 0
    while thursdays < 4:
        if d.weekday() == 3:
            thursdays += 1
        if thursdays < 4:
            d += timedelta(days=1)
    holidays[d] = "Thanksgiving"

    # Christmas (Dec 25)
    xmas = date(year, 12, 25)
    if xmas.weekday() == 6:
        holidays[date(year, 12, 26)] = "Christmas (observed)"
    elif xmas.weekday() == 5:
        holidays[date(year, 12, 24)] = "Christmas (observed)"
    else:
        holidays[xmas] = "Christmas"

    return holidays


def _easter(year: int) -> date:
    """Computus algorithm for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class Calendar:
    """Business day calendar backed by dim_calendar."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn

    def populate(
        self,
        start_year: int,
        end_year: int,
        exchange: str = "NYSE",
    ) -> int:
        """
        Populate dim_calendar for a year range.

        Returns number of rows inserted.
        """
        rows = 0
        for year in range(start_year, end_year + 1):
            holidays = _us_market_holidays(year)
            d = date(year, 1, 1)
            end = date(year, 12, 31)

            while d <= end:
                is_weekend = d.weekday() >= 5
                is_holiday = d in holidays
                is_business = not is_weekend and not is_holiday

                self._conn.execute("""
                    INSERT OR REPLACE INTO dim_calendar (
                        calendar_date, exchange_code,
                        year, quarter, month, day_of_week, day_name,
                        is_business_day, is_holiday, holiday_name
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, [
                    d, exchange,
                    d.year,
                    (d.month - 1) // 3 + 1,
                    d.month,
                    d.weekday(),
                    DAY_NAMES[d.weekday()],
                    is_business,
                    is_holiday,
                    holidays.get(d),
                ])
                rows += 1
                d += timedelta(days=1)

        return rows

    def business_days(
        self,
        start: str | date,
        end: str | date,
        exchange: str = "NYSE",
    ) -> list[date]:
        """Get business days in a date range."""
        rows = self._conn.execute("""
            SELECT calendar_date FROM dim_calendar
            WHERE exchange_code = $1
              AND calendar_date >= $2::DATE
              AND calendar_date <= $3::DATE
              AND is_business_day = TRUE
            ORDER BY calendar_date
        """, [exchange, str(start), str(end)]).fetchall()
        return [r[0] for r in rows]

    def is_business_day(
        self,
        d: str | date,
        exchange: str = "NYSE",
    ) -> bool:
        """Check if a date is a business day."""
        row = self._conn.execute("""
            SELECT is_business_day FROM dim_calendar
            WHERE exchange_code = $1 AND calendar_date = $2::DATE
        """, [exchange, str(d)]).fetchone()
        return row[0] if row else False

    def next_business_day(
        self,
        d: str | date,
        exchange: str = "NYSE",
    ) -> date | None:
        """Get the next business day after a given date."""
        row = self._conn.execute("""
            SELECT calendar_date FROM dim_calendar
            WHERE exchange_code = $1
              AND calendar_date > $2::DATE
              AND is_business_day = TRUE
            ORDER BY calendar_date LIMIT 1
        """, [exchange, str(d)]).fetchone()
        return row[0] if row else None

    def prev_business_day(
        self,
        d: str | date,
        exchange: str = "NYSE",
    ) -> date | None:
        """Get the previous business day before a given date."""
        row = self._conn.execute("""
            SELECT calendar_date FROM dim_calendar
            WHERE exchange_code = $1
              AND calendar_date < $2::DATE
              AND is_business_day = TRUE
            ORDER BY calendar_date DESC LIMIT 1
        """, [exchange, str(d)]).fetchone()
        return row[0] if row else None

    def holidays(
        self,
        year: int,
        exchange: str = "NYSE",
    ) -> list[tuple[date, str]]:
        """Get all holidays for a year."""
        rows = self._conn.execute("""
            SELECT calendar_date, holiday_name FROM dim_calendar
            WHERE exchange_code = $1 AND year = $2 AND is_holiday = TRUE
            ORDER BY calendar_date
        """, [exchange, year]).fetchall()
        return [(r[0], r[1]) for r in rows]

    def count(self, exchange: str = "NYSE") -> int:
        """Total calendar entries."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM dim_calendar WHERE exchange_code = $1",
            [exchange],
        ).fetchone()[0]
