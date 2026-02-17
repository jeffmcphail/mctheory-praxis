"""
Vendor Raw Capture Adapter (§18 diagnostic instrumentation).

Dormant instrumentation that activates via the tag system to capture
raw, unparsed vendor responses. Zero overhead when inactive — just the
nanosecond tag-check that §18 promises.

Two components:
1. VendorCaptureAdapter — LogAdapter that writes structured capture files
2. vendor_capture() — helper to emit raw capture records at call sites

Activation:
    log = PraxisLogger.instance()
    log.activate_tag("vendor_raw_capture")

Capture file format (one per session):
    ╔══VENDOR_CALL══════════════════════════════════════════╗
    ║ vendor: yfinance                                      ║
    ║ endpoint: download                                    ║
    ║ ticker: AAPL                                          ║
    ║ timestamp: 2026-02-13T16:30:00.123456Z               ║
    ║ params: start=2023-01-01 end=2024-01-01 interval=1d  ║
    ╠══RAW_PAYLOAD═════════════════════════════════════════╣
    Date,Open,High,Low,Close,Adj Close,Volume
    2023-01-03,130.28,130.90,124.17,125.07,124.40,112117500
    ...
    ╚══END_CALL═════════════════════════════════════════════╝

Usage at vendor call sites:
    from praxis.logger.vendor_capture import vendor_capture

    raw_text = response.text  # Before ANY parsing
    vendor_capture(
        vendor="yfinance",
        endpoint="download",
        ticker="AAPL",
        params={"start": "2023-01-01", "end": "2024-01-01"},
        raw_payload=raw_text,
    )
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from praxis.logger.adapters import LogAdapter
from praxis.logger.records import LogRecord, LogLevel
from praxis.logger.formatters import LogFormatter, CompactFormatter


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

CAPTURE_TAG = "vendor_raw_capture"

CALL_START = "+==VENDOR_CALL======================================================"
PAYLOAD_DELIM = "|==RAW_PAYLOAD======================================================"
CALL_END = "+==END_CALL========================================================"

DEFAULT_CAPTURE_DIR = "logs/vendor_captures"


# ═══════════════════════════════════════════════════════════════════
#  Capture Adapter
# ═══════════════════════════════════════════════════════════════════

class VendorCaptureAdapter(LogAdapter):
    """
    Log adapter that writes raw vendor data to session capture files.

    Only processes records tagged with 'vendor_raw_capture'.
    One file per session, named with session start timestamp.

    The adapter writes raw text with no modification — the payload
    between delimiters is exactly what the vendor returned.
    """

    def __init__(
        self,
        capture_dir: str | Path = DEFAULT_CAPTURE_DIR,
        session_id: str | None = None,
    ):
        super().__init__(
            name="vendor_capture",
            min_level=LogLevel.TRACE,  # Accept everything — tag gates it
        )
        self._capture_dir = Path(capture_dir)
        self._session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._file_path = self._capture_dir / f"capture_{self._session_id}.txt"
        self._lock = threading.Lock()
        self._call_count = 0
        self._is_open = False
        self._file = None

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def session_id(self) -> str:
        return self._session_id

    def _ensure_open(self) -> None:
        """Lazy-open the capture file on first write."""
        if not self._is_open:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self._file_path, "a", encoding="utf-8")
            # Write session header
            self._file.write(f"# Vendor Raw Capture Session: {self._session_id}\n")
            self._file.write(f"# Started: {datetime.now(timezone.utc).isoformat()}\n")
            self._file.write(f"# Format: raw vendor responses, no parsing applied\n")
            self._file.write("\n")
            self._file.flush()
            self._is_open = True

    def emit(self, record: LogRecord) -> None:
        """
        Write a vendor capture record.

        Only processes records that have the capture metadata
        (vendor, raw_payload) in their kwargs.
        """
        # Only process vendor capture records
        if not record.context.get("_is_vendor_capture"):
            return

        with self._lock:
            self._ensure_open()
            self._call_count += 1

            vendor = record.context.get("vendor", "unknown")
            endpoint = record.context.get("endpoint", "unknown")
            ticker = record.context.get("ticker", "")
            params = record.context.get("params", {})
            raw_payload = record.context.get("raw_payload", "")
            call_timestamp = record.timestamp.isoformat()

            # Write structured capture block
            self._file.write(f"{CALL_START}\n")
            self._file.write(f"vendor: {vendor}\n")
            self._file.write(f"endpoint: {endpoint}\n")
            if ticker:
                self._file.write(f"ticker: {ticker}\n")
            self._file.write(f"timestamp: {call_timestamp}\n")
            if params:
                param_str = " ".join(f"{k}={v}" for k, v in params.items())
                self._file.write(f"params: {param_str}\n")
            self._file.write(f"call_number: {self._call_count}\n")
            self._file.write(f"{PAYLOAD_DELIM}\n")
            self._file.write(raw_payload)
            if raw_payload and not raw_payload.endswith("\n"):
                self._file.write("\n")
            self._file.write(f"{CALL_END}\n\n")
            self._file.flush()

    def flush(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file and not self._file.closed:
                self._file.write(f"# Session ended: {datetime.now(timezone.utc).isoformat()}\n")
                self._file.write(f"# Total calls captured: {self._call_count}\n")
                self._file.close()
            self._is_open = False


# ═══════════════════════════════════════════════════════════════════
#  Capture Helper
# ═══════════════════════════════════════════════════════════════════

def vendor_capture(
    vendor: str,
    endpoint: str,
    raw_payload: str,
    ticker: str = "",
    params: dict[str, Any] | None = None,
) -> None:
    """
    Emit a vendor raw capture record.

    This is the function called at every vendor data retrieval point.
    When the 'vendor_raw_capture' tag is not active, the logger's
    tag-check exits in nanoseconds — zero overhead.

    Args:
        vendor: Provider name ("yfinance", "polygon", "binance", etc.)
        endpoint: API endpoint or method name ("download", "aggs", "ticker")
        raw_payload: The raw, unparsed text from the vendor. NO modification.
        ticker: Ticker/symbol being fetched (optional, for filtering).
        params: Dict of call parameters (start, end, interval, etc.)
    """
    from praxis.logger.core import PraxisLogger

    log = PraxisLogger.instance()

    log.debug(
        f"[vendor_capture] {vendor}.{endpoint}" + (f" {ticker}" if ticker else ""),
        tags={CAPTURE_TAG},
        _is_vendor_capture=True,
        vendor=vendor,
        endpoint=endpoint,
        ticker=ticker,
        params=params or {},
        raw_payload=raw_payload,
    )


# ═══════════════════════════════════════════════════════════════════
#  Capture File Parser (for replay/reconciliation)
# ═══════════════════════════════════════════════════════════════════

def parse_capture_file(path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a vendor capture file back into structured records.

    Returns list of dicts with keys:
        vendor, endpoint, ticker, timestamp, params, call_number, raw_payload

    This is what Claude (or any consumer) uses to replay vendor data.
    """
    path = Path(path)
    if not path.exists():
        return []

    records = []
    current: dict[str, Any] | None = None
    in_payload = False
    payload_lines: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.rstrip("\n")

            if line_stripped.startswith(CALL_START):
                current = {}
                in_payload = False
                payload_lines = []
                continue

            if line_stripped.startswith(PAYLOAD_DELIM):
                in_payload = True
                continue

            if line_stripped.startswith(CALL_END):
                if current is not None:
                    current["raw_payload"] = "\n".join(payload_lines)
                    records.append(current)
                current = None
                in_payload = False
                payload_lines = []
                continue

            if in_payload:
                payload_lines.append(line_stripped)
            elif current is not None:
                # Parse header fields
                if ": " in line_stripped:
                    key, _, value = line_stripped.partition(": ")
                    key = key.strip()
                    value = value.strip()
                    if key == "params":
                        # Parse "k1=v1 k2=v2" format
                        params = {}
                        for pair in value.split():
                            if "=" in pair:
                                pk, _, pv = pair.partition("=")
                                params[pk] = pv
                        current["params"] = params
                    elif key == "call_number":
                        current["call_number"] = int(value)
                    else:
                        current[key] = value

    return records


def get_payload_for_ticker(
    records: list[dict[str, Any]],
    vendor: str,
    ticker: str,
) -> str | None:
    """
    Extract raw payload for a specific vendor+ticker from parsed records.

    Returns the raw_payload string, or None if not found.
    """
    for record in records:
        if record.get("vendor") == vendor and record.get("ticker") == ticker:
            return record.get("raw_payload")
    return None


# ═══════════════════════════════════════════════════════════════════
#  Session Management
# ═══════════════════════════════════════════════════════════════════

def start_capture_session(
    capture_dir: str | Path = DEFAULT_CAPTURE_DIR,
    session_id: str | None = None,
) -> VendorCaptureAdapter:
    """
    Start a vendor capture session.

    Installs the VendorCaptureAdapter and activates the tag.
    Returns the adapter for inspection/closing.

    Usage:
        adapter = start_capture_session()
        # ... run data operations ...
        end_capture_session()
        print(f"Captured {adapter.call_count} calls to {adapter.file_path}")
    """
    from praxis.logger.core import PraxisLogger

    log = PraxisLogger.instance()
    adapter = VendorCaptureAdapter(capture_dir=capture_dir, session_id=session_id)
    log.add_adapter(adapter)
    log.activate_tag(CAPTURE_TAG)
    return adapter


def end_capture_session() -> dict[str, Any]:
    """
    End a vendor capture session.

    Deactivates the tag and removes the adapter.
    Returns session summary.
    """
    from praxis.logger.core import PraxisLogger

    log = PraxisLogger.instance()
    log.deactivate_tag(CAPTURE_TAG)

    adapter = log.get_adapter("vendor_capture")
    summary = {
        "calls_captured": 0,
        "file_path": None,
        "session_id": None,
    }

    if adapter and isinstance(adapter, VendorCaptureAdapter):
        summary["calls_captured"] = adapter.call_count
        summary["file_path"] = str(adapter.file_path)
        summary["session_id"] = adapter.session_id

    log.remove_adapter("vendor_capture")
    return summary
