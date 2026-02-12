"""
Tests for YFinance Data Loader (Phase 2.3) + Price Storage (Phase 2.11).

Covers:
- Loader lifecycle: fetch → working → match → archive → prices
- Security matching: creates new + matches existing
- Working table safety: no wipe until all terminal
- Archive: ldr_yfinance_hist populated
- fact_price_daily storage and retrieval
- Cross-source: same security from loader + manual match
- Batch tracking and lineage
- Mock yfinance (no network)
"""

import time
from datetime import date, datetime, timezone
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from praxis.datastore.database import PraxisDatabase
from praxis.datastore.keys import EntityKeys
from praxis.loaders import YFinanceLoader, LoadResult
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
def master(db):
    return SecurityMaster(db.connection)


@pytest.fixture
def loader(db, master):
    return YFinanceLoader(db.connection, master)


def _mock_yf_data(symbol: str, n: int = 100, start_date: str = "2023-01-03"):
    """Create mock yfinance-style pandas DataFrame."""
    dates = pd.date_range(start_date, periods=n, freq="B")
    np.random.seed(hash(symbol) % 2**31)
    base = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open": base * 0.998,
        "High": base * 1.005,
        "Low": base * 0.995,
        "Close": base,
        "Adj Close": base * 0.999,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


# ═══════════════════════════════════════════════════════════════════
#  Full Pipeline (mocked yfinance)
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    @patch("praxis.loaders.yf")
    def test_single_symbol_load(self, mock_yf, loader, db):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=50)

        result = loader.load(["AAPL"], start="2023-01-01", end="2023-03-31")

        assert isinstance(result, LoadResult)
        assert result.rows_fetched == 50
        assert result.symbols_loaded == 1
        assert result.rows_stored == 50
        assert len(result.errors) == 0

    @patch("praxis.loaders.yf")
    def test_multiple_symbols_load(self, mock_yf, loader, db):
        mock_yf.download.side_effect = [
            _mock_yf_data("AAPL", n=30),
            _mock_yf_data("MSFT", n=30),
        ]

        result = loader.load(["AAPL", "MSFT"], start="2023-01-01")

        assert result.symbols_loaded == 2
        assert result.rows_fetched == 60
        assert result.rows_stored == 60

    @patch("praxis.loaders.yf")
    def test_string_symbol_accepted(self, mock_yf, loader):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=10)
        result = loader.load("AAPL")
        assert result.symbols_loaded == 1

    @patch("praxis.loaders.yf")
    def test_batch_id_generated(self, mock_yf, loader):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=5)
        result = loader.load(["AAPL"])
        assert result.batch_id.startswith("yf_")

    @patch("praxis.loaders.yf")
    def test_empty_response(self, mock_yf, loader):
        mock_yf.download.return_value = pd.DataFrame()
        result = loader.load(["INVALID"])
        assert result.rows_fetched == 0
        assert len(result.errors) > 0


# ═══════════════════════════════════════════════════════════════════
#  Security Matching
# ═══════════════════════════════════════════════════════════════════

class TestSecurityMatching:
    @patch("praxis.loaders.yf")
    def test_creates_new_security(self, mock_yf, loader, master):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=10)
        result = loader.load(["AAPL"])

        assert result.rows_created == 10
        assert master.count() == 1

        sec = master.lookup("EQUITY", {"TICKER": "AAPL"})
        assert sec is not None

    @patch("praxis.loaders.yf")
    def test_matches_existing_security(self, mock_yf, loader, master):
        # Pre-create security
        master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="manual")

        mock_yf.download.return_value = _mock_yf_data("AAPL", n=10)
        result = loader.load(["AAPL"])

        assert result.rows_matched == 10
        assert result.rows_created == 0
        assert master.count() == 1  # No new security created

    @patch("praxis.loaders.yf")
    def test_mix_new_and_existing(self, mock_yf, loader, master):
        master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="manual")

        mock_yf.download.side_effect = [
            _mock_yf_data("AAPL", n=10),
            _mock_yf_data("MSFT", n=10),
        ]
        result = loader.load(["AAPL", "MSFT"])

        assert result.rows_matched == 10   # AAPL
        assert result.rows_created == 10   # MSFT
        assert master.count() == 2


# ═══════════════════════════════════════════════════════════════════
#  Working Table Safety
# ═══════════════════════════════════════════════════════════════════

class TestWorkingTableSafety:
    @patch("praxis.loaders.yf")
    def test_working_table_wiped_after_success(self, mock_yf, loader, db):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=10)
        result = loader.load(["AAPL"])

        # Working table should be empty after successful batch
        count = db.connection.execute(
            "SELECT COUNT(*) FROM ldr_yfinance"
        ).fetchone()[0]
        assert count == 0

    @patch("praxis.loaders.yf")
    def test_archive_populated(self, mock_yf, loader, db):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=10)
        result = loader.load(["AAPL"])

        # Archive should have all rows
        count = db.connection.execute(
            "SELECT COUNT(*) FROM ldr_yfinance_hist WHERE batch_id = $1",
            [result.batch_id],
        ).fetchone()[0]
        assert count == 10

    @patch("praxis.loaders.yf")
    def test_archive_tracks_process_status(self, mock_yf, loader, db):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=5)
        result = loader.load(["AAPL"])

        statuses = db.connection.execute("""
            SELECT DISTINCT process_status FROM ldr_yfinance_hist
            WHERE batch_id = $1
        """, [result.batch_id]).fetchall()

        assert len(statuses) == 1
        assert statuses[0][0] in ("matched", "created")


# ═══════════════════════════════════════════════════════════════════
#  fact_price_daily
# ═══════════════════════════════════════════════════════════════════

class TestPriceStorage:
    @patch("praxis.loaders.yf")
    def test_prices_stored(self, mock_yf, loader, db):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=20)
        result = loader.load(["AAPL"])

        count = db.connection.execute(
            "SELECT COUNT(*) FROM fact_price_daily"
        ).fetchone()[0]
        assert count == 20

    @patch("praxis.loaders.yf")
    def test_prices_linked_to_security(self, mock_yf, loader, db, master):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=10)
        loader.load(["AAPL"])

        sec_id = master.lookup("EQUITY", {"TICKER": "AAPL"})
        prices = db.connection.execute("""
            SELECT COUNT(*) FROM fact_price_daily
            WHERE security_base_id = $1
        """, [sec_id]).fetchone()[0]

        assert prices == 10

    @patch("praxis.loaders.yf")
    def test_get_prices_as_polars(self, mock_yf, loader, master):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=15)
        loader.load(["AAPL"])

        sec_id = master.lookup("EQUITY", {"TICKER": "AAPL"})
        df = loader.get_prices(sec_id)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 15
        assert "close" in df.columns
        assert "date" in df.columns

    @patch("praxis.loaders.yf")
    def test_prices_source_tracking(self, mock_yf, loader, db):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=5)
        result = loader.load(["AAPL"])

        row = db.connection.execute("""
            SELECT source, batch_id, created_by FROM fact_price_daily LIMIT 1
        """).fetchone()

        assert row[0] == "yfinance"
        assert row[1] == result.batch_id
        assert row[2] == "yfinance_loader"


# ═══════════════════════════════════════════════════════════════════
#  Lineage and Batch Tracking
# ═══════════════════════════════════════════════════════════════════

class TestLineage:
    @patch("praxis.loaders.yf")
    def test_multiple_batches_tracked(self, mock_yf, loader, db):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=5)
        r1 = loader.load(["AAPL"])

        mock_yf.download.return_value = _mock_yf_data("AAPL", n=5, start_date="2023-04-01")
        r2 = loader.load(["AAPL"])

        assert r1.batch_id != r2.batch_id

        batches = db.connection.execute("""
            SELECT DISTINCT batch_id FROM ldr_yfinance_hist
        """).fetchall()
        assert len(batches) == 2

    @patch("praxis.loaders.yf")
    def test_lineage_trace(self, mock_yf, loader, db, master):
        """Trace a security back to its loader origin."""
        mock_yf.download.return_value = _mock_yf_data("GLD", n=5)
        loader.load(["GLD"], sec_type="ETF")

        sec_id = master.lookup("ETF", {"TICKER": "GLD"})

        # Find earliest loader record for this security
        origin = db.connection.execute("""
            SELECT symbol, batch_id, load_timestamp
            FROM ldr_yfinance_hist
            WHERE security_base_id = $1
            ORDER BY load_timestamp ASC
            LIMIT 1
        """, [sec_id]).fetchone()

        assert origin[0] == "GLD"


# ═══════════════════════════════════════════════════════════════════
#  ETF / Crypto sec_type
# ═══════════════════════════════════════════════════════════════════

class TestSecTypes:
    @patch("praxis.loaders.yf")
    def test_etf_loading(self, mock_yf, loader, master):
        mock_yf.download.return_value = _mock_yf_data("SPY", n=10)
        result = loader.load(["SPY"], sec_type="ETF")

        assert result.symbols_loaded == 1
        sec = master.get(master.lookup("ETF", {"TICKER": "SPY"}))
        assert sec["sec_type"] == "ETF"

    @patch("praxis.loaders.yf")
    def test_crypto_loading(self, mock_yf, loader, master):
        mock_yf.download.return_value = _mock_yf_data("BTC-USD", n=10)
        result = loader.load(["BTC-USD"], sec_type="CRYPTO")

        # Crypto uses SYMBOL, not TICKER
        # The loader passes TICKER but security master maps it
        assert result.symbols_loaded == 1


# ═══════════════════════════════════════════════════════════════════
#  LoadResult
# ═══════════════════════════════════════════════════════════════════

class TestLoadResult:
    @patch("praxis.loaders.yf")
    def test_duration_tracked(self, mock_yf, loader):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=5)
        result = loader.load(["AAPL"])
        assert result.duration_seconds > 0

    @patch("praxis.loaders.yf")
    def test_result_fields_complete(self, mock_yf, loader):
        mock_yf.download.return_value = _mock_yf_data("AAPL", n=5)
        result = loader.load(["AAPL"])

        assert result.batch_id is not None
        assert result.symbols_requested == ["AAPL"]
        assert result.symbols_loaded >= 0
        assert result.rows_fetched >= 0
        assert result.rows_stored >= 0
