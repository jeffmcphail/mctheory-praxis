"""
Cycle 54c -- trading_sessions table (Engine 7 first-class sessions).

Creates a NEW table. A session is a real entity with a session_id; every
booked paper row (paper_trades + paper_position_exits, both gaining a NOT
NULL session_id in cycle54a/b) is attributable to one. Created by the GUI
controllers (live/replay) and by funding_executor.main()'s ambient session
(scheduled + CLI).

Columns beyond the brief's confirmed set:
  harness_db_path  -- R3 isolation: a paper_replay session's booked rows live
                      in an isolated per-session harness DB, NOT the main DB;
                      this records where, so the analyze view can find them.
                      NULL for paper_live (rows are in the main DB).
  pnl_rollup_json  -- guardrail F: the P&L rollup persisted at completion, so
                      a session stays reviewable even after its (cleanable)
                      harness DB is removed.

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
    "session_id", "created_at", "started_at", "ended_at", "status", "mode",
    "trigger_source", "config_json", "executor_version", "replay_start",
    "replay_end", "harness_db_path", "pnl_rollup_json", "notes",
}

DDL = """
    CREATE TABLE trading_sessions (
        session_id       TEXT    PRIMARY KEY,
        created_at       TEXT    NOT NULL,
        started_at       TEXT,
        ended_at         TEXT,
        status           TEXT    NOT NULL,
        mode             TEXT    NOT NULL,
        trigger_source   TEXT    NOT NULL,
        config_json      TEXT    NOT NULL,
        executor_version TEXT    NOT NULL,
        replay_start     TEXT,
        replay_end       TEXT,
        harness_db_path  TEXT,
        pnl_rollup_json  TEXT,
        notes            TEXT
    )
"""


def detect_state(conn) -> str:
    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='trading_sessions'"
    ).fetchone() is None:
        return "missing"
    info = list(conn.execute("PRAGMA table_info(trading_sessions)"))
    cols = {r[1] for r in info}
    pk = sorted(r[1] for r in info if r[5] > 0)
    if EXPECTED_COLS.issubset(cols) and pk == ["session_id"]:
        return "ready"
    return "unknown"


def main() -> int:
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        state = detect_state(conn)
        if state == "ready":
            print("[migrate] trading_sessions already exists with the expected "
                  "schema. Exiting cleanly.")
            return 0
        if state == "unknown":
            cols = [(r[1], r[2]) for r in
                    conn.execute("PRAGMA table_info(trading_sessions)")]
            print(f"[migrate] ERROR: trading_sessions exists with unexpected "
                  f"schema: {cols}", file=sys.stderr)
            return 3
        assert state == "missing"

        print("[migrate] Creating trading_sessions table...")
        conn.execute(DDL)
        conn.commit()

        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-create state = {post}", file=sys.stderr)
            return 4
        print("[migrate] trading_sessions created OK.")
        for r in conn.execute("PRAGMA table_info(trading_sessions)"):
            print(f"  {r[1]:<18} {r[2]:<8} notnull={r[3]} pk={r[5]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
