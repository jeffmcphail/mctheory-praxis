"""
Tests for Vendor Raw Capture System.

Validates:
- VendorCaptureAdapter writes structured capture files
- vendor_capture() emits records only when tag is active
- Capture file parser round-trips correctly
- Session management (start/end)
- Integration with existing data sources (DEX/CEX connectors emit captures)
"""

import os
import tempfile
from pathlib import Path

import pytest

from praxis.logger.core import PraxisLogger
from praxis.logger.vendor_capture import (
    VendorCaptureAdapter,
    vendor_capture,
    parse_capture_file,
    get_payload_for_ticker,
    start_capture_session,
    end_capture_session,
    CAPTURE_TAG,
    CALL_START,
    PAYLOAD_DELIM,
    CALL_END,
)


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


@pytest.fixture
def capture_dir(tmp_path):
    return tmp_path / "captures"


# ═══════════════════════════════════════════════════════════════════
#  VendorCaptureAdapter
# ═══════════════════════════════════════════════════════════════════

class TestVendorCaptureAdapter:
    def test_lazy_file_creation(self, capture_dir):
        adapter = VendorCaptureAdapter(capture_dir=capture_dir)
        # File not created until first write
        assert not adapter.file_path.exists()

    def test_write_capture(self, capture_dir):
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir, session_id="test001")
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        vendor_capture(
            vendor="yfinance",
            endpoint="download",
            ticker="AAPL",
            params={"start": "2023-01-01", "end": "2024-01-01"},
            raw_payload="Date,Open,High,Low,Close\n2023-01-03,130.28,130.90,124.17,125.07",
        )

        assert adapter.file_path.exists()
        assert adapter.call_count == 1

        content = adapter.file_path.read_text(encoding="utf-8")
        assert "vendor: yfinance" in content
        assert "endpoint: download" in content
        assert "ticker: AAPL" in content
        assert "130.28" in content
        assert CALL_START in content
        assert CALL_END in content

    def test_multiple_captures(self, capture_dir):
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir)
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        for ticker in ["AAPL", "MSFT", "GOOG"]:
            vendor_capture(
                vendor="yfinance", endpoint="download",
                ticker=ticker, raw_payload=f"data_for_{ticker}",
            )

        assert adapter.call_count == 3

    def test_raw_payload_unmodified(self, capture_dir):
        """The payload between delimiters must be exactly what was passed in."""
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir, session_id="raw_test")
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        raw = "col1,col2,col3\n1.123456789,2.987654321,3.111111111\nspecial chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        vendor_capture(vendor="test", endpoint="test", raw_payload=raw)

        content = adapter.file_path.read_text(encoding="utf-8")
        # The raw payload appears verbatim between delimiters
        assert raw in content

    def test_session_header(self, capture_dir):
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir, session_id="hdr_test")
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        vendor_capture(vendor="x", endpoint="y", raw_payload="z")

        content = adapter.file_path.read_text(encoding="utf-8")
        assert "# Vendor Raw Capture Session: hdr_test" in content

    def test_close_writes_footer(self, capture_dir):
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir)
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        vendor_capture(vendor="x", endpoint="y", raw_payload="z")
        adapter.close()

        content = adapter.file_path.read_text(encoding="utf-8")
        assert "# Session ended:" in content
        assert "# Total calls captured: 1" in content


# ═══════════════════════════════════════════════════════════════════
#  Dormancy (tag gating)
# ═══════════════════════════════════════════════════════════════════

class TestTagGating:
    def test_no_capture_without_tag(self, capture_dir):
        """When tag is not active, adapter receives nothing."""
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir)
        log.add_adapter(adapter)
        # NOT activating the tag

        vendor_capture(vendor="yfinance", endpoint="download", raw_payload="should not appear")

        assert adapter.call_count == 0
        assert not adapter.file_path.exists()

    def test_activate_then_deactivate(self, capture_dir):
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir)
        log.add_adapter(adapter)

        # Activate → capture works
        log.activate_tag(CAPTURE_TAG)
        vendor_capture(vendor="a", endpoint="b", raw_payload="captured")
        assert adapter.call_count == 1

        # Deactivate → capture stops
        log.deactivate_tag(CAPTURE_TAG)
        vendor_capture(vendor="a", endpoint="b", raw_payload="not captured")
        assert adapter.call_count == 1


# ═══════════════════════════════════════════════════════════════════
#  Capture File Parser
# ═══════════════════════════════════════════════════════════════════

class TestCaptureParser:
    def test_round_trip(self, capture_dir):
        """Write captures then parse them back — data integrity check."""
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir, session_id="parse_test")
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        vendor_capture(
            vendor="yfinance", endpoint="download", ticker="AAPL",
            params={"start": "2023-01-01", "end": "2024-01-01"},
            raw_payload="Date,Close\n2023-01-03,125.07\n2023-01-04,126.36",
        )
        vendor_capture(
            vendor="polygon", endpoint="aggs", ticker="MSFT",
            params={"timespan": "day"},
            raw_payload='{"results":[{"c":250.1,"h":252.0}]}',
        )

        adapter.close()

        records = parse_capture_file(adapter.file_path)
        assert len(records) == 2

        assert records[0]["vendor"] == "yfinance"
        assert records[0]["ticker"] == "AAPL"
        assert "125.07" in records[0]["raw_payload"]
        assert records[0]["params"]["start"] == "2023-01-01"

        assert records[1]["vendor"] == "polygon"
        assert records[1]["ticker"] == "MSFT"
        assert '"results"' in records[1]["raw_payload"]

    def test_get_payload_for_ticker(self, capture_dir):
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir, session_id="ticker_test")
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        vendor_capture(vendor="yfinance", endpoint="download", ticker="AAPL", raw_payload="aapl_data")
        vendor_capture(vendor="yfinance", endpoint="download", ticker="GOOG", raw_payload="goog_data")
        adapter.close()

        records = parse_capture_file(adapter.file_path)
        assert get_payload_for_ticker(records, "yfinance", "AAPL") == "aapl_data"
        assert get_payload_for_ticker(records, "yfinance", "GOOG") == "goog_data"
        assert get_payload_for_ticker(records, "yfinance", "MISSING") is None

    def test_parse_nonexistent_file(self):
        records = parse_capture_file("/tmp/does_not_exist.txt")
        assert records == []

    def test_multiline_payload(self, capture_dir):
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir, session_id="multiline")
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        payload = "line1\nline2\nline3\nline4 with special chars: <>&\"\nline5"
        vendor_capture(vendor="test", endpoint="test", raw_payload=payload)
        adapter.close()

        records = parse_capture_file(adapter.file_path)
        assert len(records) == 1
        assert records[0]["raw_payload"] == payload

    def test_call_number_tracking(self, capture_dir):
        log = PraxisLogger.instance()
        adapter = VendorCaptureAdapter(capture_dir=capture_dir, session_id="callnum")
        log.add_adapter(adapter)
        log.activate_tag(CAPTURE_TAG)

        for i in range(5):
            vendor_capture(vendor="v", endpoint="e", raw_payload=f"call_{i}")
        adapter.close()

        records = parse_capture_file(adapter.file_path)
        assert len(records) == 5
        assert records[0]["call_number"] == 1
        assert records[4]["call_number"] == 5


# ═══════════════════════════════════════════════════════════════════
#  Session Management
# ═══════════════════════════════════════════════════════════════════

class TestSessionManagement:
    def test_start_end_session(self, capture_dir):
        adapter = start_capture_session(capture_dir=capture_dir, session_id="session01")
        assert CAPTURE_TAG in PraxisLogger.instance().active_tags

        vendor_capture(vendor="yfinance", endpoint="download", raw_payload="test data")

        summary = end_capture_session()
        assert summary["calls_captured"] == 1
        assert summary["session_id"] == "session01"
        assert CAPTURE_TAG not in PraxisLogger.instance().active_tags

    def test_session_file_persists(self, capture_dir):
        adapter = start_capture_session(capture_dir=capture_dir, session_id="persist")
        vendor_capture(vendor="test", endpoint="test", raw_payload="persistent data")
        end_capture_session()

        # File should exist and be parseable
        records = parse_capture_file(capture_dir / "capture_persist.txt")
        assert len(records) == 1


# ═══════════════════════════════════════════════════════════════════
#  Integration: DEX/CEX connectors emit captures
# ═══════════════════════════════════════════════════════════════════

from praxis.onchain.connectors import (
    UniswapV2Source,
    BinanceSource,
)


class TestConnectorCapture:
    def test_dex_fetch_captured(self, capture_dir):
        adapter = start_capture_session(capture_dir=capture_dir, session_id="dex")

        source = UniswapV2Source()
        source.set_prices({"WETH/USDC": 3000.0})
        source.fetch(["WETH/USDC"], "", "")

        summary = end_capture_session()
        assert summary["calls_captured"] == 1

        records = parse_capture_file(capture_dir / "capture_dex.txt")
        assert records[0]["vendor"] == "uniswap_v2"
        assert "3000" in records[0]["raw_payload"]

    def test_cex_fetch_captured(self, capture_dir):
        adapter = start_capture_session(capture_dir=capture_dir, session_id="cex")

        source = BinanceSource()
        source.set_prices({"ETHUSDT": (3000.0, 3001.0)})
        source.fetch(["ETHUSDT"], "", "")

        summary = end_capture_session()
        assert summary["calls_captured"] == 1

        records = parse_capture_file(capture_dir / "capture_cex.txt")
        assert records[0]["vendor"] == "binance"
        assert "3000" in records[0]["raw_payload"]

    def test_no_capture_without_session(self, capture_dir):
        """Without an active session, fetch works but doesn't capture."""
        source = UniswapV2Source()
        source.set_prices({"WETH/USDC": 3000.0})
        source.fetch(["WETH/USDC"], "", "")
        # No crash, no file created
        assert not list(capture_dir.glob("*.txt"))
