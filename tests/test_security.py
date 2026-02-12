"""
Tests for Security Master (Phase 2.1/2.2).

Covers:
- SecIdType hierarchy definitions and BPK generation
- match_or_create: new security creation
- match_or_create: match existing by any identifier in hierarchy
- match_or_create: populate missing identifiers on match
- Multiple data sources resolve to same security_base_id
- Conflict detection: sec_type mismatch → conflict_queue
- Audit trail: dim_security_identifier_audit populated
- Temporal versioning: update creates new hist_id
- AS-IS / AS-WAS queries after update
- Ephemeral mode compatibility
"""

import time

import duckdb
import pytest

from praxis.datastore.database import PraxisDatabase
from praxis.datastore.keys import EntityKeys
from praxis.logger.core import PraxisLogger
from praxis.security import SecurityMaster
from praxis.security.hierarchy import (
    SECID_HIERARCHY,
    SECID_TO_COLUMN,
    SecIdType,
    SecType,
    get_preferred_bpk,
)


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
def master(db):
    return SecurityMaster(db.connection)


# ═══════════════════════════════════════════════════════════════════
#  SecIdType Hierarchy
# ═══════════════════════════════════════════════════════════════════

class TestHierarchy:
    def test_all_sec_types_have_hierarchy(self):
        for sec_type in SecType:
            assert sec_type in SECID_HIERARCHY
            assert len(SECID_HIERARCHY[sec_type]) > 0

    def test_equity_hierarchy_order(self):
        h = SECID_HIERARCHY[SecType.EQUITY]
        assert h[0] == SecIdType.ISIN
        assert h[1] == SecIdType.CUSIP
        assert SecIdType.TICKER in h

    def test_crypto_hierarchy(self):
        h = SECID_HIERARCHY[SecType.CRYPTO]
        assert h[0] == SecIdType.SYMBOL
        assert SecIdType.CONTRACT_ADDRESS in h

    def test_all_secid_types_have_column_mapping(self):
        for secid_type in SecIdType:
            assert secid_type in SECID_TO_COLUMN


class TestBPKGeneration:
    def test_isin_preferred_over_ticker(self):
        bpk, secid = get_preferred_bpk(
            "EQUITY", {"TICKER": "AAPL", "ISIN": "US0378331005"}
        )
        assert bpk == "EQUITY|ISIN|US0378331005"
        assert secid == SecIdType.ISIN

    def test_cusip_preferred_over_ticker(self):
        bpk, secid = get_preferred_bpk(
            "EQUITY", {"TICKER": "AAPL", "CUSIP": "037833100"}
        )
        assert bpk == "EQUITY|CUSIP|037833100"
        assert secid == SecIdType.CUSIP

    def test_ticker_only(self):
        bpk, secid = get_preferred_bpk("EQUITY", {"TICKER": "AAPL"})
        assert bpk == "EQUITY|TICKER|AAPL"
        assert secid == SecIdType.TICKER

    def test_crypto_symbol(self):
        bpk, secid = get_preferred_bpk("CRYPTO", {"SYMBOL": "BTC-USD"})
        assert bpk == "CRYPTO|SYMBOL|BTC-USD"

    def test_fx_pair(self):
        bpk, secid = get_preferred_bpk("FX", {"ISO_PAIR": "EUR/USD"})
        assert bpk == "FX|ISO_PAIR|EUR/USD"

    def test_unknown_sec_type_raises(self):
        with pytest.raises(ValueError, match="WIDGET"):
            get_preferred_bpk("WIDGET", {"TICKER": "X"})

    def test_no_valid_identifier_raises(self):
        with pytest.raises(ValueError, match="No valid identifier"):
            get_preferred_bpk("EQUITY", {"UNKNOWN_ID": "123"})

    def test_case_insensitive_identifiers(self):
        bpk, _ = get_preferred_bpk("EQUITY", {"ticker": "AAPL"})
        assert bpk == "EQUITY|TICKER|AAPL"

    def test_string_sec_type(self):
        bpk, _ = get_preferred_bpk("ETF", {"TICKER": "SPY"})
        assert bpk == "ETF|TICKER|SPY"


# ═══════════════════════════════════════════════════════════════════
#  Match-or-Create: New Security
# ═══════════════════════════════════════════════════════════════════

class TestCreateNew:
    def test_create_equity_by_ticker(self, master):
        base_id = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"},
            name="Apple Inc.", source="yfinance",
        )
        assert isinstance(base_id, int)
        assert base_id != 0
        assert master.count() == 1

    def test_create_equity_by_isin(self, master):
        base_id = master.match_or_create(
            "EQUITY", {"ISIN": "US0378331005", "TICKER": "AAPL"},
            source="polygon",
        )
        assert base_id != 0

        sec = master.get(base_id)
        assert sec["isin"] == "US0378331005"
        assert sec["ticker"] == "AAPL"
        assert sec["sec_type"] == "EQUITY"

    def test_create_crypto(self, master):
        base_id = master.match_or_create(
            "CRYPTO", {"SYMBOL": "BTC-USD"},
            name="Bitcoin", source="binance",
        )
        sec = master.get(base_id)
        assert sec["symbol"] == "BTC-USD"
        assert sec["sec_type"] == "CRYPTO"

    def test_create_etf(self, master):
        base_id = master.match_or_create(
            "ETF", {"TICKER": "SPY", "ISIN": "US78462F1030"},
            name="SPDR S&P 500", source="yfinance",
        )
        sec = master.get(base_id)
        assert sec["isin"] == "US78462F1030"
        assert sec["ticker"] == "SPY"

    def test_create_with_descriptive_fields(self, master):
        base_id = master.match_or_create(
            "EQUITY", {"TICKER": "MSFT"},
            name="Microsoft Corp",
            currency="USD",
            exchange="NASDAQ",
            sector="Technology",
            industry="Software",
            country="US",
            source="manual",
        )
        sec = master.get(base_id)
        assert sec["name"] == "Microsoft Corp"
        assert sec["currency"] == "USD"
        assert sec["exchange"] == "NASDAQ"
        assert sec["sector"] == "Technology"


# ═══════════════════════════════════════════════════════════════════
#  Match-or-Create: Match Existing
# ═══════════════════════════════════════════════════════════════════

class TestMatchExisting:
    def test_same_ticker_returns_same_base_id(self, master):
        """Two calls with same ticker → same security."""
        id1 = master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="yfinance")
        id2 = master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="polygon")
        assert id1 == id2
        assert master.count() == 1

    def test_match_by_isin_when_created_by_ticker(self, master):
        """Create by ticker, then match by ISIN after enrichment."""
        id1 = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"},
            source="yfinance",
        )
        # Second source provides ISIN for same ticker
        id2 = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL", "ISIN": "US0378331005"},
            source="polygon",
        )
        assert id1 == id2

        # Now ISIN should be populated
        sec = master.get(id1)
        assert sec["isin"] == "US0378331005"

    def test_match_by_different_identifier(self, master):
        """Create by ISIN, match by CUSIP → same if CUSIP added."""
        id1 = master.match_or_create(
            "EQUITY", {"ISIN": "US0378331005", "CUSIP": "037833100"},
            source="source_a",
        )
        # Match by CUSIP alone (also in hierarchy)
        # Note: This should match because ISIN is checked first in hierarchy,
        # but the CUSIP column was populated during creation
        id2 = master.match_or_create(
            "EQUITY", {"CUSIP": "037833100"},
            source="source_b",
        )
        assert id1 == id2
        assert master.count() == 1


# ═══════════════════════════════════════════════════════════════════
#  Cross-Source Resolution (Milestone 2 criterion)
# ═══════════════════════════════════════════════════════════════════

class TestCrossSourceResolution:
    def test_yfinance_and_polygon_resolve_same(self, master):
        """
        Milestone 2: Load GLD from yfinance AND Polygon →
        same security_base_id for both.
        """
        # yfinance provides ticker only
        id_yf = master.match_or_create(
            "ETF", {"TICKER": "GLD"},
            name="SPDR Gold Shares",
            source="yfinance",
        )

        # Polygon provides ticker + CUSIP
        id_poly = master.match_or_create(
            "ETF", {"TICKER": "GLD", "CUSIP": "78463V107"},
            name="SPDR Gold Shares",
            source="polygon",
        )

        assert id_yf == id_poly
        assert master.count() == 1

        # CUSIP should now be populated
        sec = master.get(id_yf)
        assert sec["cusip"] == "78463V107"

    def test_three_sources_same_security(self, master):
        """Three different sources, each with different identifiers."""
        id1 = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="yfinance",
        )
        time.sleep(0.01)
        id2 = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL", "ISIN": "US0378331005"},
            source="polygon",
        )
        time.sleep(0.01)
        id3 = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL", "CUSIP": "037833100", "SEDOL": "2046251"},
            source="bloomberg",
        )

        assert id1 == id2 == id3
        assert master.count() == 1

        sec = master.get(id1)
        assert sec["isin"] == "US0378331005"
        assert sec["cusip"] == "037833100"
        assert sec["sedol"] == "2046251"
        assert sec["ticker"] == "AAPL"


# ═══════════════════════════════════════════════════════════════════
#  Identifier Enrichment (new hist_id versions)
# ═══════════════════════════════════════════════════════════════════

class TestEnrichment:
    def test_update_creates_new_version(self, db, master):
        base_id = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="yfinance",
        )

        # Count versions before enrichment
        count_before = db.connection.execute("""
            SELECT COUNT(*) FROM dim_security
            WHERE security_base_id = $1
        """, [base_id]).fetchone()[0]

        time.sleep(0.01)

        # Enrich with ISIN
        master.match_or_create(
            "EQUITY", {"TICKER": "AAPL", "ISIN": "US0378331005"},
            source="polygon",
        )

        count_after = db.connection.execute("""
            SELECT COUNT(*) FROM dim_security
            WHERE security_base_id = $1
        """, [base_id]).fetchone()[0]

        assert count_after == count_before + 1

    def test_no_update_when_nothing_new(self, db, master):
        """If all identifiers already known, no new version created."""
        master.match_or_create(
            "EQUITY", {"TICKER": "AAPL", "ISIN": "US0378331005"},
            source="source_a",
        )

        count_before = db.connection.execute(
            "SELECT COUNT(*) FROM dim_security"
        ).fetchone()[0]

        time.sleep(0.01)

        # Same identifiers again
        master.match_or_create(
            "EQUITY", {"TICKER": "AAPL", "ISIN": "US0378331005"},
            source="source_b",
        )

        count_after = db.connection.execute(
            "SELECT COUNT(*) FROM dim_security"
        ).fetchone()[0]

        assert count_after == count_before  # No new version

    def test_descriptive_field_enrichment(self, master):
        base_id = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="yfinance",
        )
        sec = master.get(base_id)
        assert sec["sector"] is None

        time.sleep(0.01)

        master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"},
            sector="Technology", industry="Consumer Electronics",
            source="classification_provider",
        )

        sec = master.get(base_id)
        assert sec["sector"] == "Technology"
        assert sec["industry"] == "Consumer Electronics"


# ═══════════════════════════════════════════════════════════════════
#  Conflict Queue
# ═══════════════════════════════════════════════════════════════════

class TestConflicts:
    def test_sectype_mismatch_queues_conflict(self, master):
        """Same ticker, different sec_type → conflict."""
        master.match_or_create("EQUITY", {"TICKER": "GLD"}, source="a")

        # Same ticker but different sec_type
        master.match_or_create("ETF", {"TICKER": "GLD"}, source="b")

        conflicts = master.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["conflict_type"] == "sectype_mismatch"

    def test_conflict_still_returns_base_id(self, master):
        """Conflict is flagged but matching still works."""
        id1 = master.match_or_create("EQUITY", {"TICKER": "GLD"}, source="a")
        id2 = master.match_or_create("ETF", {"TICKER": "GLD"}, source="b")

        # Returns existing base_id despite conflict
        assert id1 == id2


# ═══════════════════════════════════════════════════════════════════
#  Audit Trail
# ═══════════════════════════════════════════════════════════════════

class TestAudit:
    def test_audit_records_created(self, db, master):
        master.match_or_create(
            "EQUITY", {"TICKER": "AAPL", "ISIN": "US0378331005"},
            source="polygon",
        )
        count = db.connection.execute(
            "SELECT COUNT(*) FROM dim_security_identifier_audit"
        ).fetchone()[0]
        assert count == 2  # TICKER + ISIN

    def test_audit_tracks_source(self, db, master):
        master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="yfinance",
        )
        time.sleep(0.01)
        master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="polygon",
        )

        rows = db.connection.execute("""
            SELECT source FROM dim_security_identifier_audit
            WHERE secid_type = 'TICKER' AND secid_value = 'AAPL'
            ORDER BY audit_hist_id
        """).fetchall()

        sources = [r[0] for r in rows]
        assert "yfinance" in sources
        assert "polygon" in sources


# ═══════════════════════════════════════════════════════════════════
#  Lookup (read-only)
# ═══════════════════════════════════════════════════════════════════

class TestLookup:
    def test_lookup_existing(self, master):
        base_id = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="test",
        )
        found = master.lookup("EQUITY", {"TICKER": "AAPL"})
        assert found == base_id

    def test_lookup_missing(self, master):
        found = master.lookup("EQUITY", {"TICKER": "NONEXIST"})
        assert found is None

    def test_lookup_does_not_create(self, master):
        master.lookup("EQUITY", {"TICKER": "NEW"})
        assert master.count() == 0


# ═══════════════════════════════════════════════════════════════════
#  Ephemeral Mode
# ═══════════════════════════════════════════════════════════════════

class TestEphemeral:
    def test_works_without_views(self):
        """SecurityMaster falls back to raw tables in ephemeral mode."""
        db = PraxisDatabase.ephemeral()
        db.initialize()
        master = SecurityMaster(db.connection)

        base_id = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="test",
        )
        assert base_id != 0

        sec = master.get(base_id)
        assert sec is not None
        assert sec["ticker"] == "AAPL"

        # Match works too
        id2 = master.match_or_create(
            "EQUITY", {"TICKER": "AAPL"}, source="test2",
        )
        assert id2 == base_id
