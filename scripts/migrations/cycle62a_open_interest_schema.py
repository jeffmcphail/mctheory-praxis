"""
Cycle 62A T2 -- open_interest table.

Repairs regime class F, which Cycle 61 T4 showed silently collapsing from five
declared states to three whenever no OI series was supplied -- which was every
production caller. Cycle 62A T5 made that degradation announce itself; this
table is what lets it stop being degraded at all.

URGENCY (why the table exists before the analysis)
--------------------------------------------------
The Cycle 61 T4 venue probe established hard history walls:
    Binance -- 30 days (startTime rejected beyond that)
    Bybit   -- 200 rows (since at -200d/-400d/-730d all return the same rows)
Nothing before those walls is retrievable at any granularity, from either
venue, ever. Every day without collection is a permanent hole in a series that
cannot be reconstructed.

RULE 35 CONFORMANCE
-------------------
`timestamp` is INTEGER ms-since-epoch UTC and part of the compound primary key
(asset, venue, timestamp) -- the same shape funding_rates was migrated to in
Cycle 50, deliberately, so the two join cleanly. `datetime` is the derived
ISO-8601 (+00:00) read cache.

`open_interest` is in base units (contracts/coins as the venue reports them);
`open_interest_value` is the quote-currency notional. Venues differ in which
they populate, so both are nullable and the collector records which arrived.

Idempotent: re-running on an already-created table prints "already exists"
and exits 0.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"

EXPECTED_COLS = {
    "asset", "venue", "timestamp", "datetime", "open_interest",
    "open_interest_value", "symbol", "source", "ingested_at",
}
EXPECTED_PK = ["asset", "timestamp", "venue"]

DDL = """
    CREATE TABLE open_interest (
        asset               TEXT    NOT NULL,
        venue               TEXT    NOT NULL,
        timestamp           INTEGER NOT NULL,
        datetime            TEXT    NOT NULL,
        open_interest       REAL,
        open_interest_value REAL,
        symbol              TEXT,
        source              TEXT    NOT NULL,
        ingested_at         INTEGER NOT NULL,
        PRIMARY KEY (asset, venue, timestamp)
    )
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_open_interest_ts ON open_interest(timestamp)",
]


def detect_state(conn) -> str:
    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='open_interest'"
    ).fetchone() is None:
        return "missing"
    info = list(conn.execute("PRAGMA table_info(open_interest)"))
    cols = {r[1] for r in info}
    pk = sorted(r[1] for r in info if r[5] > 0)
    if EXPECTED_COLS.issubset(cols) and pk == EXPECTED_PK:
        return "ready"
    return "unknown"


def main() -> int:
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        state = detect_state(conn)
        if state == "ready":
            print("[migrate] open_interest already exists with the expected "
                  "schema. Exiting cleanly.")
            return 0
        if state == "unknown":
            cols = [(r[1], r[2]) for r in
                    conn.execute("PRAGMA table_info(open_interest)")]
            print(f"[migrate] ERROR: open_interest exists with unexpected "
                  f"schema: {cols}", file=sys.stderr)
            return 3
        assert state == "missing"

        print("[migrate] Creating open_interest table...")
        conn.execute(DDL)
        for idx in INDEXES:
            conn.execute(idx)
        conn.commit()

        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-create state = {post}", file=sys.stderr)
            return 4
        print("[migrate] open_interest created OK.")
        for r in conn.execute("PRAGMA table_info(open_interest)"):
            print(f"  {r[1]:<20} {r[2]:<8} notnull={r[3]} pk={r[5]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
