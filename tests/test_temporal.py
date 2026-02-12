"""
Bi-temporal validation test suite (Phase 2.9).

Proves:
- Insert security → update security → AS-IS sees update, AS-WAS sees original
- vt2_ views correctly derive start_date/end_date from hist_id sequence
- Multiple versions tracked with temporal boundaries
- AS-IS always returns latest; AS-WAS reproduces historical state
- Compare utility detects field-level differences
- Model definitions also support temporal queries
- Cross-verification: SecurityMaster enrichment visible in temporal queries

Milestone 2 criterion:
  "Update GLD's exchange field. AS-IS sees update.
   AS-WAS at original date sees old value."
"""

import time
from datetime import date, datetime, timedelta, timezone

import duckdb
import pytest

from praxis.datastore.database import PraxisDatabase
from praxis.datastore.keys import EntityKeys
from praxis.datastore.temporal import TemporalQuery
from praxis.logger.core import PraxisLogger
from praxis.security import SecurityMaster


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_singletons():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


@pytest.fixture
def db():
    database = PraxisDatabase(":memory:")
    database.initialize()
    return database


@pytest.fixture
def tq(db):
    return TemporalQuery(db.connection)


@pytest.fixture
def master(db):
    return SecurityMaster(db.connection)


def _insert_security_version(
    conn, base_id: int, bpk: str, sec_type: str = "EQUITY",
    ticker: str = None, isin: str = None, exchange: str = None,
    name: str = None, sector: str = None, country: str = None,
    hist_date: str = None,
):
    """Insert a security version with controlled hist_id date."""
    if hist_date:
        # Use a specific date for the hist_id (for temporal boundary testing)
        ts = datetime.strptime(hist_date, "%Y-%m-%d").replace(
            hour=12, minute=0, second=0, tzinfo=timezone.utc
        )
    else:
        ts = datetime.now(timezone.utc)

    conn.execute("""
        INSERT INTO dim_security (
            security_hist_id, security_base_id, security_bpk,
            sec_type, ticker, isin, exchange, name, sector, country,
            status, created_by
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'ACTIVE', 'test')
    """, [ts, base_id, bpk, sec_type, ticker, isin, exchange, name, sector, country])

    return ts


# ═══════════════════════════════════════════════════════════════════
#  vt2_ View Fundamentals
# ═══════════════════════════════════════════════════════════════════

class TestVt2ViewBasics:
    def test_single_version_has_open_end_date(self, db, tq):
        """Single version → start_date = creation date, end_date = 9999-12-31."""
        ts = _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-01-15",
        )
        versions = tq.security_all_versions(1)
        assert len(versions) == 1
        assert versions[0]["start_date"] == date(2024, 1, 15)
        assert versions[0]["end_date"] == date(9999, 12, 31)

    def test_two_versions_boundary(self, db, tq):
        """Two versions → v1 ends day before v2 starts."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-01-15",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", hist_date="2024-03-10",
        )
        versions = tq.security_all_versions(1)
        assert len(versions) == 2

        v1, v2 = versions
        assert v1["start_date"] == date(2024, 1, 15)
        assert v1["end_date"] == date(2024, 3, 9)   # day before v2
        assert v2["start_date"] == date(2024, 3, 10)
        assert v2["end_date"] == date(9999, 12, 31)  # open-ended

    def test_three_versions_chain(self, db, tq):
        """Three versions form a contiguous chain."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-01-01",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", hist_date="2024-06-01",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", sector="Technology",
            hist_date="2024-09-01",
        )
        versions = tq.security_all_versions(1)
        assert len(versions) == 3

        assert versions[0]["end_date"] == date(2024, 5, 31)
        assert versions[1]["start_date"] == date(2024, 6, 1)
        assert versions[1]["end_date"] == date(2024, 8, 31)
        assert versions[2]["start_date"] == date(2024, 9, 1)
        assert versions[2]["end_date"] == date(9999, 12, 31)

    def test_multiple_updates_same_day_deduped(self, db, tq):
        """Multiple updates on same day → only latest kept per day."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", name="Apple v1", hist_date="2024-01-15",
        )
        # Slightly later same day (we'll use a different time)
        conn = db.connection
        ts2 = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        conn.execute("""
            INSERT INTO dim_security (
                security_hist_id, security_base_id, security_bpk,
                sec_type, ticker, name, status, created_by
            ) VALUES ($1, $2, $3, 'EQUITY', 'AAPL', 'Apple v2', 'ACTIVE', 'test')
        """, [ts2, 1, "EQUITY|TICKER|AAPL"])

        versions = tq.security_all_versions(1)
        # vt2_ view takes latest per day, so only 1 version
        assert len(versions) == 1
        assert versions[0]["name"] == "Apple v2"  # Latest wins


# ═══════════════════════════════════════════════════════════════════
#  AS-IS Queries
# ═══════════════════════════════════════════════════════════════════

class TestAsIs:
    def test_as_is_returns_latest(self, db, tq):
        """AS-IS always returns the most current version."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NYSE", hist_date="2024-01-15",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", hist_date="2024-06-01",
        )

        result = tq.security_as_is(1)
        assert result is not None
        assert result["exchange"] == "NASDAQ"

    def test_as_is_picks_up_corrections(self, db, tq):
        """AS-IS shows corrected data."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", sector="Consumer", hist_date="2024-01-01",
        )
        # Correction: sector was wrong
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", sector="Technology", hist_date="2024-01-02",
        )

        result = tq.security_as_is(1)
        assert result["sector"] == "Technology"

    def test_as_is_nonexistent(self, tq):
        assert tq.security_as_is(999999) is None


# ═══════════════════════════════════════════════════════════════════
#  AS-WAS Queries
# ═══════════════════════════════════════════════════════════════════

class TestAsWas:
    def test_as_was_sees_original(self, db, tq):
        """AS-WAS at original date sees the old value."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NYSE", hist_date="2024-01-15",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", hist_date="2024-06-01",
        )

        # AS-WAS at Feb 2024 → should see NYSE (original version)
        result = tq.security_as_was(1, "2024-02-15")
        assert result is not None
        assert result["exchange"] == "NYSE"

    def test_as_was_sees_update_after_update_date(self, db, tq):
        """AS-WAS after the update date sees the updated value."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NYSE", hist_date="2024-01-15",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", hist_date="2024-06-01",
        )

        # AS-WAS at July 2024 → should see NASDAQ (updated version)
        result = tq.security_as_was(1, "2024-07-01")
        assert result is not None
        assert result["exchange"] == "NASDAQ"

    def test_as_was_before_creation(self, db, tq):
        """AS-WAS before the security existed → None."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-06-01",
        )
        result = tq.security_as_was(1, "2024-01-01")
        assert result is None

    def test_as_was_exact_boundary(self, db, tq):
        """AS-WAS on exact start_date of v2 → sees v2."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NYSE", hist_date="2024-01-15",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", hist_date="2024-06-01",
        )

        # On the exact update date → v2
        result = tq.security_as_was(1, "2024-06-01")
        assert result["exchange"] == "NASDAQ"

        # Day before → v1
        result = tq.security_as_was(1, "2024-05-31")
        assert result["exchange"] == "NYSE"


# ═══════════════════════════════════════════════════════════════════
#  Milestone 2 Criterion: GLD Exchange Correction
# ═══════════════════════════════════════════════════════════════════

class TestMilestone2GLD:
    def test_gld_exchange_correction(self, db, tq):
        """
        Milestone 2 pass criterion:
        "Update GLD's exchange field. AS-IS sees update.
         AS-WAS at original date sees old value."
        """
        # Original: GLD created with exchange = "ARCA"
        _insert_security_version(
            db.connection, base_id=42, bpk="ETF|TICKER|GLD",
            sec_type="ETF", ticker="GLD", exchange="ARCA",
            name="SPDR Gold Shares",
            hist_date="2024-01-10",
        )

        # Backtest ran on 2024-03-15 using GLD data
        backtest_date = "2024-03-15"

        # Later correction: exchange should be "NYSE_ARCA"
        _insert_security_version(
            db.connection, base_id=42, bpk="ETF|TICKER|GLD",
            sec_type="ETF", ticker="GLD", exchange="NYSE_ARCA",
            name="SPDR Gold Shares",
            hist_date="2024-07-01",
        )

        # AS-IS: sees corrected exchange
        as_is = tq.security_as_is(42)
        assert as_is["exchange"] == "NYSE_ARCA"

        # AS-WAS at backtest run date: sees original exchange
        as_was = tq.security_as_was(42, backtest_date)
        assert as_was["exchange"] == "ARCA"

        # Compare shows the difference
        diff = tq.compare_security(42, backtest_date)
        assert diff["changed"] is True
        assert "exchange" in diff["differences"]
        assert diff["differences"]["exchange"]["old"] == "ARCA"
        assert diff["differences"]["exchange"]["new"] == "NYSE_ARCA"


# ═══════════════════════════════════════════════════════════════════
#  Compare Utility
# ═══════════════════════════════════════════════════════════════════

class TestCompare:
    def test_no_change_detected(self, db, tq):
        """When no update happened, compare shows no differences."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ",
            hist_date="2024-01-15",
        )
        # Query after creation but before any update → same version
        diff = tq.compare_security(1, "2024-06-01")
        assert diff["changed"] is False
        assert diff["differences"] == {}

    def test_multiple_fields_changed(self, db, tq):
        """Compare detects changes across multiple fields."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NYSE", sector="Consumer",
            hist_date="2024-01-01",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", sector="Technology",
            hist_date="2024-06-01",
        )

        diff = tq.compare_security(1, "2024-03-01")
        assert diff["changed"] is True
        assert "exchange" in diff["differences"]
        assert "sector" in diff["differences"]

    def test_compare_nonexistent_as_was(self, db, tq):
        """Compare when AS-WAS returns None → still works."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-06-01",
        )
        diff = tq.compare_security(1, "2024-01-01")
        assert diff["as_was"] is None


# ═══════════════════════════════════════════════════════════════════
#  SecurityMaster + Temporal Integration
# ═══════════════════════════════════════════════════════════════════

class TestSecurityMasterTemporal:
    def test_enrichment_creates_temporal_version(self, db, master, tq):
        """SecurityMaster enrichment → new version visible in temporal queries."""
        # Create with ticker only
        base_id = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="yfinance",
        )

        v1_versions = tq.security_all_versions(base_id)
        assert len(v1_versions) == 1
        assert v1_versions[0]["isin"] is None

        time.sleep(0.01)

        # Enrich with ISIN
        master.match_or_create(
            "EQUITY", {"TICKER": "AAPL", "ISIN": "US0378331005"},
            source="polygon",
        )

        v2_versions = tq.security_all_versions(base_id)
        # May be 1 or 2 depending on whether same-day dedup applies
        latest = v2_versions[-1]
        assert latest["isin"] == "US0378331005"

    def test_as_is_after_enrichment(self, db, master, tq):
        """AS-IS picks up enriched data."""
        base_id = master.match_or_create(
            "EQUITY", {"TICKER": "MSFT"}, source="yfinance",
        )
        time.sleep(0.01)
        master.match_or_create(
            "EQUITY", {"TICKER": "MSFT"},
            sector="Technology", country="US",
            source="classification_provider",
        )

        result = tq.security_as_is(base_id)
        assert result["sector"] == "Technology"
        assert result["country"] == "US"


# ═══════════════════════════════════════════════════════════════════
#  vew_ (Current State) vs vt2_ (Point-in-Time)
# ═══════════════════════════════════════════════════════════════════

class TestVewVsVt2:
    def test_vew_always_latest(self, db):
        """vew_security always returns latest version only."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NYSE", hist_date="2024-01-15",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", hist_date="2024-06-01",
        )

        rows = db.connection.execute(
            "SELECT exchange FROM vew_security WHERE security_base_id = 1"
        ).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "NASDAQ"

    def test_vt2_returns_all_versions(self, db):
        """vt2_security returns all temporal versions."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NYSE", hist_date="2024-01-15",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", exchange="NASDAQ", hist_date="2024-06-01",
        )

        rows = db.connection.execute(
            "SELECT exchange FROM vt2_security WHERE security_base_id = 1 ORDER BY start_date"
        ).fetchall()

        assert len(rows) == 2
        assert rows[0][0] == "NYSE"
        assert rows[1][0] == "NASDAQ"

    def test_dim_raw_has_all_records(self, db):
        """Raw dim_security has every insert."""
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-01-15",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-06-01",
        )
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-09-01",
        )

        count = db.connection.execute(
            "SELECT COUNT(*) FROM dim_security WHERE security_base_id = 1"
        ).fetchone()[0]
        assert count == 3


# ═══════════════════════════════════════════════════════════════════
#  Date Input Formats
# ═══════════════════════════════════════════════════════════════════

class TestDateFormats:
    def test_string_date(self, db, tq):
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-01-15",
        )
        result = tq.security_as_was(1, "2024-06-01")
        assert result is not None

    def test_date_object(self, db, tq):
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-01-15",
        )
        result = tq.security_as_was(1, date(2024, 6, 1))
        assert result is not None

    def test_datetime_object(self, db, tq):
        _insert_security_version(
            db.connection, base_id=1, bpk="EQUITY|TICKER|AAPL",
            ticker="AAPL", hist_date="2024-01-15",
        )
        result = tq.security_as_was(1, datetime(2024, 6, 1, 12, 0, 0))
        assert result is not None
