#!/usr/bin/env python3
"""
Vendor Raw Capture Script.

Run from project root:
    python scripts/capture_vendor_data.py

Produces: logs/vendor_captures/capture_<session_id>.txt
Upload that file to Claude for reconciliation testing.
"""

import sys
import os
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from praxis.logger.core import PraxisLogger
from praxis.logger.vendor_capture import start_capture_session, end_capture_session

# ── Config ────────────────────────────────────────────────
SESSION_ID = "battle_test_001"

# What to fetch — edit these for each test run
TICKERS = ["AAPL", "MSFT", "GLD", "GDX"]
START = "2023-01-01"
END = "2024-01-01"
# ──────────────────────────────────────────────────────────

def main():
    # Initialize logger
    log = PraxisLogger.instance()
    log.configure_defaults()

    print(f"Starting vendor capture session: {SESSION_ID}")
    print(f"Tickers: {TICKERS}")
    print(f"Range: {START} → {END}")
    print()

    # Start capture
    adapter = start_capture_session(session_id=SESSION_ID)

    # Fetch data
    from praxis.data import fetch_prices

    try:
        prices = fetch_prices(TICKERS, start=START, end=END)
        print(f"Fetched {len(prices)} total rows")
        print(f"Columns: {prices.columns}")
        print()
        print(prices.head(5))
    except Exception as e:
        print(f"Fetch error: {e}")
        print("Make sure yfinance is installed: pip install yfinance")

    # End capture
    summary = end_capture_session()
    print()
    print("=" * 60)
    print(f"Capture complete!")
    print(f"  Calls captured: {summary['calls_captured']}")
    print(f"  File: {summary['file_path']}")
    print(f"  Session: {summary['session_id']}")
    print()
    print(f"Upload this file to Claude:")
    print(f"  {summary['file_path']}")

if __name__ == "__main__":
    main()
