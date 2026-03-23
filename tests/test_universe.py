"""
Tests for market_data.universe — index membership providers.

Tests cover:
    - Ticker normalization (overrides, dot-to-dash, whitespace)
    - SP500MembershipProvider with mocked Wikipedia HTML
    - Point-in-time membership queries
    - History reconstruction correctness
    - DataStore fetch_callback protocol
    - Edge cases (unknown dates, empty changes, duplicate tickers)
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import polars as pl
import pytest

from market_data.universe import (
    normalize_ticker,
    DEFAULT_TICKER_OVERRIDES,
    IndexMembershipProvider,
    SP500MembershipProvider,
)


# ═════════════════════════════════════════════════════════════════════════════
# Ticker normalization
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalizeTicker:

    def test_dot_to_dash(self):
        assert normalize_ticker("BRK.B") == "BRK-B"

    def test_default_override(self):
        assert normalize_ticker("BF.B") == "BF-B"

    def test_custom_override_takes_precedence(self):
        assert normalize_ticker("OLD", overrides={"OLD": "NEW"}) == "NEW"

    def test_no_change_needed(self):
        assert normalize_ticker("AAPL") == "AAPL"

    def test_whitespace_stripped(self):
        assert normalize_ticker("  MSFT  ") == "MSFT"

    def test_multiple_dots(self):
        assert normalize_ticker("BRK.A.X") == "BRK-A-X"

    def test_empty_string(self):
        assert normalize_ticker("") == ""


# ═════════════════════════════════════════════════════════════════════════════
# Mock Wikipedia data
# ═════════════════════════════════════════════════════════════════════════════

def _make_current_table() -> pd.DataFrame:
    """Simulate Wikipedia Table 0: current constituents."""
    return pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK.B"],
    })


def _make_changes_table() -> pd.DataFrame:
    """
    Simulate Wikipedia Table 1: change history.

    Timeline:
        2024-06-15: Added META, Removed INTC
        2024-01-10: Added AMZN, Removed GE
        2023-07-01: Added GOOGL, Removed IBM

    So walking backward from current {AAPL, MSFT, GOOGL, AMZN, META, BRK.B}:
        After 2024-06-15: remove META, add INTC → {AAPL, MSFT, GOOGL, AMZN, BRK-B, INTC}
        After 2024-01-10: remove AMZN, add GE   → {AAPL, MSFT, GOOGL, BRK-B, GE, INTC}
        After 2023-07-01: remove GOOGL, add IBM  → {AAPL, MSFT, BRK-B, GE, IBM, INTC}
    """
    return pd.DataFrame({
        ("Date", "Date"): [
            "June 15, 2024",
            "January 10, 2024",
            "July 1, 2023",
        ],
        ("Added", "Ticker"): ["META", "AMZN", "GOOGL"],
        ("Removed", "Ticker"): ["INTC", "GE", "IBM"],
    })


def _mock_parse_wikipedia(self) -> list[pd.DataFrame]:
    """Replacement for SP500MembershipProvider._parse_wikipedia()."""
    return [_make_current_table(), _make_changes_table()]


# ═════════════════════════════════════════════════════════════════════════════
# SP500MembershipProvider
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def provider():
    """Create a provider with mocked Wikipedia data."""
    p = SP500MembershipProvider()
    p._parse_wikipedia = lambda: _mock_parse_wikipedia(p)
    return p


class TestSP500MembershipProvider:

    def test_current_members(self, provider):
        """members() with no date returns current constituents."""
        current = provider.members()
        assert set(current) == {"AAPL", "AMZN", "BRK-B", "GOOGL", "META", "MSFT"}

    def test_ticker_normalization(self, provider):
        """BRK.B in Wikipedia is normalized to BRK-B."""
        current = provider.members()
        assert "BRK-B" in current
        assert "BRK.B" not in current

    def test_point_in_time_after_all_changes(self, provider):
        """Query after the last change returns current membership."""
        members = provider.members("2024-12-01")
        assert set(members) == {"AAPL", "AMZN", "BRK-B", "GOOGL", "META", "MSFT"}

    def test_point_in_time_before_meta_added(self, provider):
        """Before META was added (2024-06-15), INTC should be present."""
        members = provider.members("2024-06-01")
        assert "META" not in members
        assert "INTC" in members
        assert set(members) == {"AAPL", "AMZN", "BRK-B", "GOOGL", "INTC", "MSFT"}

    def test_point_in_time_before_amzn_added(self, provider):
        """Before AMZN was added (2024-01-10), GE should be present."""
        members = provider.members("2023-12-01")
        assert "AMZN" not in members
        assert "GE" in members
        assert set(members) == {"AAPL", "BRK-B", "GE", "GOOGL", "INTC", "MSFT"}

    def test_point_in_time_before_googl_added(self, provider):
        """Before GOOGL was added (2023-07-01), IBM should be present."""
        members = provider.members("2023-01-01")
        assert "GOOGL" not in members
        assert "IBM" in members
        assert set(members) == {"AAPL", "BRK-B", "GE", "IBM", "INTC", "MSFT"}

    def test_all_constituents(self, provider):
        """all_constituents() returns every ticker that ever appeared."""
        all_tickers = provider.all_constituents()
        expected = {"AAPL", "AMZN", "BRK-B", "GE", "GOOGL", "IBM",
                    "INTC", "META", "MSFT"}
        assert set(all_tickers) == expected

    def test_history_periods(self, provider):
        """History should have 4 periods (3 changes + 1 earliest)."""
        history = provider.history()
        assert len(history) == 4

    def test_history_intervals_non_overlapping(self, provider):
        """Intervals should not overlap."""
        intervals = list(provider.history().keys())
        for i in range(len(intervals) - 1):
            # Each interval's start should be <= previous interval's start
            # (since ordered newest-first)
            assert intervals[i][0] >= intervals[i + 1][1]

    def test_n_current(self, provider):
        assert provider.n_current == 6

    def test_n_all_time(self, provider):
        assert provider.n_all_time == 9

    def test_n_periods(self, provider):
        assert provider.n_periods == 4

    def test_members_sorted(self, provider):
        """Output should always be sorted."""
        for as_of in [None, "2024-12-01", "2023-01-01"]:
            members = provider.members(as_of)
            assert members == sorted(members)

    def test_caching(self, provider):
        """Second call should use cached data, not re-parse."""
        _ = provider.members()
        history_ref = provider._history

        _ = provider.members("2023-06-01")
        assert provider._history is history_ref  # same object, not rebuilt

    def test_date_object_accepted(self, provider):
        """members() accepts date objects, not just strings."""
        members = provider.members(date(2024, 6, 1))
        assert "META" not in members
        assert "INTC" in members


# ═════════════════════════════════════════════════════════════════════════════
# DataStore fetch_callback protocol
# ═════════════════════════════════════════════════════════════════════════════

class TestFetchCallback:

    def test_callback_returns_dataframe(self, provider):
        """Callback returns proper (universe_id, security_id) DataFrame."""
        cb = provider.fetch_callback()
        result = cb("sp500_members", "universe_id", "SP500")

        assert result is not None
        assert isinstance(result, pl.DataFrame)
        assert "universe_id" in result.columns
        assert "security_id" in result.columns
        assert (result["universe_id"] == "SP500").all()
        assert len(result) == 6

    def test_callback_with_as_of_date(self, provider):
        """Callback supports 'SP500:2023-01-01' format."""
        cb = provider.fetch_callback()
        result = cb("sp500_members", "universe_id", "SP500:2023-01-01")

        assert result is not None
        tickers = set(result["security_id"].to_list())
        assert "IBM" in tickers
        assert "META" not in tickers

    def test_callback_wrong_universe(self, provider):
        """Callback returns None for unknown index."""
        cb = provider.fetch_callback()
        assert cb("sp500_members", "universe_id", "RUSSELL2000") is None

    def test_callback_current_membership(self, provider):
        """Bare 'SP500' returns current membership."""
        cb = provider.fetch_callback()
        result = cb("sp500_members", "universe_id", "SP500")

        tickers = set(result["security_id"].to_list())
        assert tickers == {"AAPL", "AMZN", "BRK-B", "GOOGL", "META", "MSFT"}


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_custom_overrides(self):
        """Custom ticker overrides are applied."""
        p = SP500MembershipProvider(ticker_overrides={"GOOGL": "GOOG"})
        p._parse_wikipedia = lambda: _mock_parse_wikipedia(p)

        members = p.members()
        assert "GOOG" in members
        assert "GOOGL" not in members

    def test_no_changes(self):
        """Provider works with empty change history."""
        p = SP500MembershipProvider()

        def mock_parse():
            current = pd.DataFrame({"Symbol": ["A", "B", "C"]})
            changes = pd.DataFrame({
                ("Date", "Date"): pd.Series(dtype=str),
                ("Added", "Ticker"): pd.Series(dtype=str),
                ("Removed", "Ticker"): pd.Series(dtype=str),
            })
            return [current, changes]

        p._parse_wikipedia = mock_parse

        assert p.members() == ["A", "B", "C"]
        # With no changes, there's just one period (the earliest)
        # or zero periods if no change dates exist
        assert p.n_periods <= 1

    def test_same_day_multiple_changes(self):
        """Multiple changes on the same day are all applied."""
        p = SP500MembershipProvider()

        def mock_parse():
            current = pd.DataFrame({"Symbol": ["A", "B", "C", "D"]})
            changes = pd.DataFrame({
                ("Date", "Date"): [
                    "March 1, 2024",
                    "March 1, 2024",  # two changes same day
                ],
                ("Added", "Ticker"): ["C", "D"],
                ("Removed", "Ticker"): ["X", "Y"],
            })
            return [current, changes]

        p._parse_wikipedia = mock_parse

        # Before March 1: C and D removed, X and Y added back
        members = p.members("2024-02-15")
        assert "C" not in members
        assert "D" not in members
        assert "X" in members
        assert "Y" in members


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
