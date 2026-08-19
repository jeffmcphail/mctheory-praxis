"""
Cycle 62A T1 -- liquidations table (Binance forced-order stream).

Cycle 61 could not compute a false-positive rate for forced-trade scenario A2
because a forced liquidation and a large discretionary market order are
identical in the `trades` tape. Binance publishes forced orders on the
!forceOrder@arr WebSocket stream, which converts A2 from inference to
observation. This table is where that stream lands.

RULE 35 CONFORMANCE
-------------------
`timestamp` is INTEGER ms-since-epoch UTC and is part of the primary key.
`datetime` is the derived ISO-8601 (+00:00) read cache. Binance sends ms
natively (field `T`, order trade time), so no unit conversion is applied --
Rule 35.4's conversion foot-gun does not arise here, and the collector
asserts the value is in a sane ms range rather than trusting it.

PRIMARY KEY / APPEND-ONLY
------------------------
PK is the natural event key (venue, symbol, timestamp, side, price, quantity).
The table is append-only in the sense that matters: rows are only ever
INSERTed, never UPDATEd or DELETEd. The PK exists so that a reconnect which
overlaps an already-captured window is idempotent (INSERT OR IGNORE) rather
than double-counting.

The tradeoff, stated rather than hidden: two genuinely distinct liquidations
sharing the same venue+symbol+millisecond+side+price+quantity would collapse
to one row. Binance throttles the all-market stream to at most one order per
second per symbol, so the stream is already a sample at that granularity, and
two such events would be indistinguishable to every downstream analysis
anyway. The collector counts suppressed duplicates and reports them, so the
collapse is visible rather than silent.

`venue` is in the PK from day one so a second venue can be added without a
schema migration -- the Cycle 50 funding_rates lesson.

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
    "venue", "symbol", "timestamp", "datetime", "side", "price", "quantity",
    "quote_qty", "order_status", "order_type", "avg_price", "event_time",
    "ingested_at",
}
EXPECTED_PK = ["price", "quantity", "side", "symbol", "timestamp", "venue"]

DDL = """
    CREATE TABLE liquidations (
        venue        TEXT    NOT NULL,
        symbol       TEXT    NOT NULL,
        timestamp    INTEGER NOT NULL,
        datetime     TEXT    NOT NULL,
        side         TEXT    NOT NULL,
        price        REAL    NOT NULL,
        quantity     REAL    NOT NULL,
        quote_qty    REAL,
        order_status TEXT,
        order_type   TEXT,
        avg_price    REAL,
        event_time   INTEGER,
        ingested_at  INTEGER NOT NULL,
        PRIMARY KEY (venue, symbol, timestamp, side, price, quantity)
    )
"""

INDEXES = [
    # Time-range scans across all symbols: the A2 clustering question.
    "CREATE INDEX IF NOT EXISTS idx_liquidations_ts ON liquidations(timestamp)",
    # Per-symbol history.
    "CREATE INDEX IF NOT EXISTS idx_liquidations_symbol_ts "
    "ON liquidations(symbol, timestamp)",
]


def detect_state(conn) -> str:
    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='liquidations'"
    ).fetchone() is None:
        return "missing"
    info = list(conn.execute("PRAGMA table_info(liquidations)"))
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
            print("[migrate] liquidations already exists with the expected "
                  "schema. Exiting cleanly.")
            return 0
        if state == "unknown":
            cols = [(r[1], r[2]) for r in
                    conn.execute("PRAGMA table_info(liquidations)")]
            pk = sorted(r[1] for r in
                        conn.execute("PRAGMA table_info(liquidations)")
                        if r[5] > 0)
            print(f"[migrate] ERROR: liquidations exists with unexpected "
                  f"schema: cols={cols} pk={pk}", file=sys.stderr)
            return 3
        assert state == "missing"

        print("[migrate] Creating liquidations table...")
        conn.execute(DDL)
        for idx in INDEXES:
            conn.execute(idx)
        conn.commit()

        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-create state = {post}", file=sys.stderr)
            return 4
        print("[migrate] liquidations created OK.")
        for r in conn.execute("PRAGMA table_info(liquidations)"):
            print(f"  {r[1]:<13} {r[2]:<8} notnull={r[3]} pk={r[5]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
