"""
Index membership providers — survivorship-bias-free universe construction.

Provides point-in-time index membership by reconstructing historical
constituents from Wikipedia's change history. The primary interface is
members(as_of), which returns the exact list of tickers that were in
the index on any given date.

Key design principles:
    - Point-in-time is the default. Current membership is just members(date.today()).
    - Wikipedia is fetched once, history is reconstructed once, everything is cached.
    - Ticker normalization is configurable (BRK.B → BRK-B, etc.).
    - The fetch_callback() method returns a callable compatible with
      DataTable.set_fetch_callback() for seamless DataStore integration.
    - Extensible: subclass IndexMembershipProvider for Russell 2000, FTSE, etc.

Example:
    >>> sp500 = SP500MembershipProvider()
    >>> today = sp500.members()                       # current constituents
    >>> historical = sp500.members("2020-03-15")      # who was in on that date
    >>> all_ever = sp500.all_constituents()            # every ticker ever in the index

DataStore integration:
    >>> provider = SP500MembershipProvider()
    >>> table = ds.get_table("sp500_members")
    >>> table.set_fetch_callback(provider.fetch_callback())
    >>> # Now has_row_filtered("universe_id", "SP500", fill_missing=True) works
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import date, timedelta
from typing import Any, Callable

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Ticker normalization
# ═════════════════════════════════════════════════════════════════════════════

# Standard overrides for tickers that differ between sources.
# Wikipedia uses dots (BRK.B), yfinance expects dashes (BRK-B).
# Add entries here for any systematic mappings.
DEFAULT_TICKER_OVERRIDES: dict[str, str] = {
    # Class shares
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    # Historical renames that Wikipedia sometimes gets wrong
}


def normalize_ticker(
    raw: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """
    Normalize a ticker symbol to a canonical form.

    Applies explicit overrides first, then the standard dot-to-dash
    transformation that aligns Wikipedia symbols with yfinance.

    Args:
        raw: Raw ticker string from source.
        overrides: Optional mapping of raw → canonical. Takes precedence
            over the default dot-to-dash rule.

    Returns:
        Normalized ticker string.
    """
    merged = {**DEFAULT_TICKER_OVERRIDES, **(overrides or {})}
    stripped = raw.strip()
    if stripped in merged:
        return merged[stripped]
    return stripped.replace(".", "-")


# ═════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═════════════════════════════════════════════════════════════════════════════

class IndexMembershipProvider(ABC):
    """
    Abstract base for index membership providers.

    Subclasses implement _load_history() to populate the interval-keyed
    membership dictionary. All point-in-time logic, caching, and DataStore
    integration is handled here.
    """

    def __init__(
        self,
        index_id: str,
        ticker_overrides: dict[str, str] | None = None,
    ):
        self._index_id = index_id
        self._ticker_overrides = ticker_overrides or {}

        # Lazy-loaded cache. Populated on first access.
        # Keys are (start_date, end_date) tuples (inclusive).
        # Values are sorted lists of tickers valid during that interval.
        self._history: OrderedDict[tuple[date, date], list[str]] | None = None
        self._all_constituents: list[str] | None = None
        self._current: list[str] | None = None

    # ── Public interface ────────────────────────────────────────────────

    @property
    def index_id(self) -> str:
        """Canonical identifier for this index (e.g. 'SP500')."""
        return self._index_id

    def members(self, as_of: str | date | None = None) -> list[str]:
        """
        Get index constituents as of a specific date.

        This is the primary interface. Returns the exact set of tickers
        that were members of the index on the given date, eliminating
        survivorship bias for backtesting.

        Args:
            as_of: Date to query membership for. Accepts 'YYYY-MM-DD'
                string or date object. Defaults to today (current membership).

        Returns:
            Sorted list of ticker symbols.

        Raises:
            ValueError: If as_of date is outside the range of available history.
        """
        self._ensure_loaded()

        if as_of is None:
            return list(self._current)

        query_date = _parse_date(as_of)

        for (start, end), tickers in self._history.items():
            if start <= query_date <= end:
                return list(tickers)

        # Date is outside all known intervals
        intervals = list(self._history.keys())
        earliest = intervals[-1][0] if intervals else None
        latest = intervals[0][1] if intervals else None
        raise ValueError(
            f"Date {query_date} is outside available history "
            f"({earliest} to {latest})"
        )

    def all_constituents(self) -> list[str]:
        """
        Every ticker that has ever been a member of this index.

        Useful for pre-fetching historical data across the full universe
        when running backtests that span multiple reconstitution periods.

        Returns:
            Sorted list of all tickers ever included.
        """
        self._ensure_loaded()
        return list(self._all_constituents)

    def history(self) -> OrderedDict[tuple[date, date], list[str]]:
        """
        Full membership history as interval-keyed dictionary.

        Keys are (start_date, end_date) tuples defining when each
        membership set was valid. Ordered newest-first.

        Returns:
            OrderedDict mapping date intervals to membership lists.
        """
        self._ensure_loaded()
        return OrderedDict(self._history)

    @property
    def n_current(self) -> int:
        """Number of current constituents."""
        self._ensure_loaded()
        return len(self._current)

    @property
    def n_all_time(self) -> int:
        """Number of tickers ever in the index."""
        self._ensure_loaded()
        return len(self._all_constituents)

    @property
    def n_periods(self) -> int:
        """Number of distinct reconstitution periods."""
        self._ensure_loaded()
        return len(self._history)

    # ── DataStore integration ───────────────────────────────────────────

    def fetch_callback(self) -> Callable[[str, str, Any], pl.DataFrame | None]:
        """
        Return a callable compatible with DataTable.set_fetch_callback().

        The callback responds to queries on a table with a 'universe_id'
        column, returning a DataFrame of (universe_id, security_id) rows.

        Supports an extended filter_val format for point-in-time queries:
            "SP500"              → current membership
            "SP500:2020-03-15"   → membership as of that date

        Returns:
            Callback function matching the DataTable fetch protocol.
        """

        def _callback(
            table_name: str, filter_col: str, filter_val: Any,
        ) -> pl.DataFrame | None:
            # Parse optional as-of date from "SP500:2023-01-15" format
            val_str = str(filter_val)
            if ":" in val_str:
                parts = val_str.split(":", 1)
                universe_id, as_of = parts[0], parts[1]
            else:
                universe_id, as_of = val_str, None

            if universe_id != self._index_id:
                return None

            try:
                tickers = self.members(as_of)
            except (ValueError, RuntimeError) as e:
                logger.warning("Membership fetch failed for %s: %s", filter_val, e)
                return None

            return pl.DataFrame({
                "universe_id": [universe_id] * len(tickers),
                "security_id": tickers,
            })

        return _callback

    # ── Internal ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load history on first access."""
        if self._history is not None:
            return

        logger.info("Loading %s membership history...", self._index_id)
        self._current, self._history = self._load_history()

        # Derive all-time constituent list
        all_tickers: set[str] = set()
        for tickers in self._history.values():
            all_tickers.update(tickers)
        self._all_constituents = sorted(all_tickers)

        logger.info(
            "%s loaded: %d current, %d all-time, %d periods",
            self._index_id, len(self._current),
            len(self._all_constituents), len(self._history),
        )

    @abstractmethod
    def _load_history(
        self,
    ) -> tuple[list[str], OrderedDict[tuple[date, date], list[str]]]:
        """
        Load the full membership history.

        Returns:
            Tuple of (current_members, historical_dict) where historical_dict
            is an OrderedDict keyed by (start_date, end_date) intervals,
            ordered newest-first.
        """
        ...

    def _normalize(self, raw: str) -> str:
        """Normalize a single ticker."""
        return normalize_ticker(raw, self._ticker_overrides)


# ═════════════════════════════════════════════════════════════════════════════
# S&P 500
# ═════════════════════════════════════════════════════════════════════════════

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Ordered by preference. lxml is fastest, bs4 uses html.parser from stdlib.
_HTML_FLAVORS = ["lxml", "bs4", "html5lib"]

# Wikipedia blocks requests without a User-Agent header.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class SP500MembershipProvider(IndexMembershipProvider):
    """
    S&P 500 membership provider using Wikipedia change history.

    Parses two tables from the Wikipedia S&P 500 page:
        Table 0: Current constituents (symbol, name, sector, ...)
        Table 1: Historical changes (date, ticker added, ticker removed)

    Reconstructs point-in-time membership by walking backwards from
    today through every change event, reversing each add/remove to
    recover the membership set at each point in history.

    Args:
        ticker_overrides: Optional mapping for non-standard ticker symbols.
        url: Wikipedia URL override (for testing or alternative mirrors).
    """

    def __init__(
        self,
        ticker_overrides: dict[str, str] | None = None,
        url: str = WIKIPEDIA_SP500_URL,
    ):
        super().__init__(index_id="SP500", ticker_overrides=ticker_overrides)
        self._url = url

    def _load_history(
        self,
    ) -> tuple[list[str], OrderedDict[tuple[date, date], list[str]]]:
        """
        Parse Wikipedia and reconstruct full S&P 500 history.

        Algorithm:
            1. Parse Table 0 for current constituents.
            2. Parse Table 1 for change events (date, added, removed).
            3. Starting from today's membership, walk backwards day by day.
            4. At each change date: record the current set, then reverse
               the change (add back removed tickers, remove added tickers).
            5. Result: an interval-keyed dict of membership sets covering
               every reconstitution period back to the earliest Wikipedia entry.
        """
        tables = self._parse_wikipedia()
        current_table, changes_table = tables[0], tables[1]

        # ── Extract current constituents ────────────────────────────────
        current = sorted(
            self._normalize(sym) for sym in current_table["Symbol"].tolist()
        )

        # ── Parse change history ────────────────────────────────────────
        changes = self._parse_changes(changes_table)

        # ── Walk backwards to reconstruct history ───────────────────────
        history = self._reconstruct_history(current, changes)

        return current, history

    def _parse_wikipedia(self) -> list[pd.DataFrame]:
        """
        Fetch and parse the Wikipedia S&P 500 tables.

        Fetches the raw HTML with a proper User-Agent (Wikipedia returns
        403 without one), then tries multiple HTML parsers.

        Returns:
            List of DataFrames (table 0 = current, table 1 = changes).

        Raises:
            RuntimeError: If the page can't be fetched or no parser works.
        """
        import io
        from urllib.request import Request, urlopen

        # Fetch HTML with User-Agent to avoid 403
        try:
            req = Request(self._url, headers={"User-Agent": _USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch {self._url}: {e}. "
                f"Check your network connection."
            ) from e

        # Parse with available flavor
        errors: list[str] = []
        for flavor in _HTML_FLAVORS:
            try:
                tables = pd.read_html(io.StringIO(html), flavor=flavor)
                logger.info("Parsed Wikipedia with %s", flavor)
                return tables
            except ImportError:
                errors.append(f"{flavor}: not installed")
            except Exception as e:
                errors.append(f"{flavor}: {e}")

        msg = (
            f"Fetched {self._url} but no parser worked. "
            f"Tried: {'; '.join(errors)}. "
            f"Install a parser: pip install lxml  (or: pip install beautifulsoup4)"
        )
        raise RuntimeError(msg)

    def _parse_changes(self, changes_table: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and normalize the change history from Wikipedia Table 1.

        Handles the multi-level column headers that Wikipedia produces and
        normalizes dates, tickers, and missing values.

        Returns:
            DataFrame with columns ['date', 'added', 'removed'], indexed by date.
        """
        df = changes_table.fillna("")

        # Wikipedia produces multi-level column tuples like ('Date', 'Date'),
        # ('Added', 'Ticker'), ('Removed', 'Ticker'). Try multi-level first.
        try:
            df = df[[
                ("Date", "Date"),
                ("Added", "Ticker"),
                ("Removed", "Ticker"),
            ]]
        except (KeyError, TypeError):
            # Flat columns — find by substring match
            cols = df.columns.tolist()
            date_col = next(c for c in cols if "date" in str(c).lower())
            add_col = next(c for c in cols if "added" in str(c).lower())
            rem_col = next(c for c in cols if "removed" in str(c).lower())
            df = df[[date_col, add_col, rem_col]]

        df.columns = ["date", "added", "removed"]
        df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
        df = df.dropna(subset=["date"])

        # Normalize tickers
        for col in ["added", "removed"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .apply(lambda x: self._normalize(x) if x and x != "nan" else "")
            )

        df.set_index("date", inplace=True)
        return df

    def _reconstruct_history(
        self,
        current: list[str],
        changes: pd.DataFrame,
    ) -> OrderedDict[tuple[date, date], list[str]]:
        """
        Reconstruct point-in-time membership from change events.

        Event-driven approach: iterates over change dates (not every day),
        recording the membership set valid between consecutive change events.

        At each change date, the current working set is recorded for the
        interval [change_date, next_change_date). Then the change is reversed:
        tickers that were added are removed (they weren't there before), and
        tickers that were removed are added back.

        Args:
            current: Today's membership list.
            changes: DataFrame of change events, indexed by Timestamp.

        Returns:
            OrderedDict of (start_date, end_date) → membership lists,
            newest-first. Intervals are inclusive on both ends.
        """
        history: OrderedDict[tuple[date, date], list[str]] = OrderedDict()
        working = list(current)

        tomorrow = date.today() + timedelta(days=1)
        earliest_possible = date(1900, 1, 1)

        # Get unique change dates, sorted newest-first
        change_dates = sorted(changes.index.unique(), reverse=True)

        if not change_dates:
            # No changes at all — single interval covering all time
            history[(earliest_possible, tomorrow)] = sorted(set(working))
            return history

        # Track the end of the current interval
        # Semantics: change takes effect at start of day. So on change_date,
        # the NEW set applies. The OLD set was valid through change_date - 1.
        interval_end = tomorrow

        for change_ts in change_dates:
            change_date = change_ts.date()

            # Record the working set for [change_date, interval_end]
            history[(change_date, interval_end)] = sorted(set(working))
            interval_end = change_date - timedelta(days=1)

            # Get all changes on this date (could be multiple)
            rows = changes.loc[change_ts]
            if isinstance(rows, pd.Series):
                rows = rows.to_frame().T

            # Reverse each change to recover the pre-change state
            for _, row in rows.iterrows():
                removed_ticker = str(row.get("removed", "")).strip()
                added_ticker = str(row.get("added", "")).strip()

                # Ticker was removed on this date → add it back (it was there before)
                if removed_ticker and removed_ticker not in ("", "nan"):
                    normalized = self._normalize(removed_ticker)
                    if normalized not in working:
                        working.append(normalized)

                # Ticker was added on this date → remove it (it wasn't there before)
                if added_ticker and added_ticker not in ("", "nan"):
                    normalized = self._normalize(added_ticker)
                    if normalized in working:
                        working.remove(normalized)

            working = sorted(set(working))

        # Final interval: from earliest_possible to the oldest change date
        history[(earliest_possible, interval_end)] = sorted(set(working))

        return history


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════

def _parse_date(value: str | date) -> date:
    """Parse a string or date into a date object."""
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()
