"""
Cycle 62A -- collector_gaps table (shared by T1 and T2).

The Cycle 62A brief requires the liquidation stream to log gaps explicitly
rather than silently resuming: a gap that leaves no trace is indistinguishable
from a quiet market. A log line is not a trace -- logs rotate, and nobody
joins a .log file against a time series. A row is a trace.

This table is deliberately collector-agnostic so every future collector can
record the same thing in the same place. Any analysis over a collected series
can LEFT JOIN it to establish whether the feed was actually up during a window
before concluding the market was quiet.

RULE 35 CONFORMANCE
-------------------
`timestamp` (INTEGER ms UTC) is the gap START and part of the PK
(collector, venue, timestamp). `gap_end` is the resumption instant;
`gap_seconds` is a derived convenience.

An open gap (collector died and never resumed) is recorded with gap_end NULL
and closed on the next successful connect, so a crash leaves a trace too.

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
    "collector", "venue", "timestamp", "datetime", "gap_end", "gap_seconds",
    "reason", "detected_at",
}
EXPECTED_PK = ["collector", "timestamp", "venue"]

DDL = """
    CREATE TABLE collector_gaps (
        collector   TEXT    NOT NULL,
        venue       TEXT    NOT NULL,
        timestamp   INTEGER NOT NULL,
        datetime    TEXT    NOT NULL,
        gap_end     INTEGER,
        gap_seconds REAL,
        reason      TEXT,
        detected_at INTEGER NOT NULL,
        PRIMARY KEY (collector, venue, timestamp)
    )
"""


def detect_state(conn) -> str:
    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='collector_gaps'"
    ).fetchone() is None:
        return "missing"
    info = list(conn.execute("PRAGMA table_info(collector_gaps)"))
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
            print("[migrate] collector_gaps already exists with the expected "
                  "schema. Exiting cleanly.")
            return 0
        if state == "unknown":
            cols = [(r[1], r[2]) for r in
                    conn.execute("PRAGMA table_info(collector_gaps)")]
            print(f"[migrate] ERROR: collector_gaps exists with unexpected "
                  f"schema: {cols}", file=sys.stderr)
            return 3
        assert state == "missing"

        print("[migrate] Creating collector_gaps table...")
        conn.execute(DDL)
        conn.commit()

        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-create state = {post}", file=sys.stderr)
            return 4
        print("[migrate] collector_gaps created OK.")
        for r in conn.execute("PRAGMA table_info(collector_gaps)"):
            print(f"  {r[1]:<12} {r[2]:<8} notnull={r[3]} pk={r[5]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
