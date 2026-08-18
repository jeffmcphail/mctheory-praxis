"""
scripts/cycle57_hl_price_backfill.py
====================================
Cycle 57 — extend Hyperliquid PERP price history backward to 2024-01-01.

ccxt's hyperliquid.fetchOHLCV returns only the most-recent ~5000 candles
(ignores `since`), so the Cycle 57 price collector only captured ~7 months of
HL perp price. This backfills the rest via the RAW public candleSnapshot
endpoint with BACKWARD pagination (endTime walked back until 2024-01-01).

READ-ONLY public data, no creds/orders/keys. Writes into the SAME isolated
`prices` table (venue='hyperliquid', market='perp'); INSERT OR IGNORE dedups
against the ~5000 already present. Run AFTER the price collector finishes (one
writer at a time on the SQLite file).
"""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESEARCH_DB = REPO / "data" / "defi_funding_research.db"
HL_INFO = "https://api.hyperliquid.xyz/info"
HOUR_MS = 3_600_000

START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
NOW_MS = int(datetime.now(timezone.utc).timestamp() * 1000)
ASSETS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "DOGE", "LINK", "UNI", "LTC"]


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00")


def candles(coin: str, start_ms: int, end_ms: int) -> list:
    body = json.dumps({"type": "candleSnapshot", "req": {
        "coin": coin, "interval": "1h",
        "startTime": start_ms, "endTime": end_ms}}).encode()
    req = urllib.request.Request(HL_INFO, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


def main():
    conn = sqlite3.connect(str(RESEARCH_DB))
    conn.execute("""CREATE TABLE IF NOT EXISTS prices (
        asset TEXT NOT NULL, venue TEXT NOT NULL, market TEXT NOT NULL,
        timestamp INTEGER NOT NULL, datetime TEXT NOT NULL, close REAL NOT NULL,
        PRIMARY KEY (asset, venue, market, timestamp))""")
    conn.commit()
    print(f"[hl-backfill] window {iso(START_MS)} .. {iso(NOW_MS)}")
    for a in ASSETS:
        end = NOW_MS
        pages = total = 0
        try:
            while end > START_MS and pages < 40:
                pages += 1
                start = max(START_MS, end - 5000 * HOUR_MS)
                for attempt in range(4):
                    try:
                        data = candles(a, start, end)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(1.5 * (attempt + 1))
                if not data:
                    break
                rows = []
                oldest = end
                for c in data:
                    t = int(c["t"]); th = (t // HOUR_MS) * HOUR_MS
                    oldest = min(oldest, t)
                    rows.append((a, "hyperliquid", "perp", th, iso(th),
                                 float(c["c"])))
                conn.executemany(
                    "INSERT OR IGNORE INTO prices "
                    "(asset, venue, market, timestamp, datetime, close) "
                    "VALUES (?,?,?,?,?,?)", rows)
                conn.commit()
                total += len(data)
                if oldest <= start + HOUR_MS or oldest >= end:
                    end = oldest - HOUR_MS
                else:
                    end = oldest - HOUR_MS
                time.sleep(0.2)
            n, lo, hi = conn.execute(
                "SELECT COUNT(*), MIN(datetime), MAX(datetime) FROM prices "
                "WHERE asset=? AND venue='hyperliquid' AND market='perp'",
                (a,)).fetchone()
            print(f"  HL {a:<6} api={total:<7} db_rows={n:<7} {lo} .. {hi}")
        except Exception as e:
            print(f"  HL {a}: ERROR {e!r}")
    conn.close()


if __name__ == "__main__":
    main()
