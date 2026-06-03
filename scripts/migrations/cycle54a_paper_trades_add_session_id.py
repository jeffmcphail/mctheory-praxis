"""
Cycle 54a -- paper_trades.session_id (Engine 7 session attribution).

R2 decision (Cycle 54 RECON): session_id is REQUIRED (NOT NULL) going
forward, so every booked paper row is attributable to a trading_sessions
row. SQLite cannot `ADD COLUMN ... NOT NULL` without a DEFAULT, and a ''
sentinel would defeat the contract (a forgotten session_id would silently
become '' instead of failing loud). Because paper_trades is at 0 rows, the
contract-true path is a 0-row REBUILD: recreate the table afresh with
session_id TEXT NOT NULL appended.

GUARDED: the rebuild proceeds ONLY if the table has the known pre-Cycle-54
schema AND COUNT(*) == 0. If any rows exist, it ABORTS (non-zero exit)
rather than silently dropping booked trades -- surface + decide manually.

Idempotent: once session_id (NOT NULL) is present, re-running exits 0.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"

# The pre-Cycle-54 column set (post-cycle52a, which added hold_days).
BASE_COLS = {
    "asset", "signal_timestamp", "signal_datetime", "funding_alert_alerted_at",
    "decided_at", "decision", "skip_reason", "intended_direction",
    "intended_size_usd", "p_profitable", "gate_threshold", "risk_checks_json",
    "executor_version", "hold_days",
}

NEW_DDL = """
    CREATE TABLE paper_trades (
        asset                    TEXT    NOT NULL,
        signal_timestamp         INTEGER NOT NULL,
        signal_datetime          TEXT    NOT NULL,
        funding_alert_alerted_at TEXT    NOT NULL,
        decided_at               TEXT    NOT NULL,
        decision                 TEXT    NOT NULL,
        skip_reason              TEXT,
        intended_direction       TEXT,
        intended_size_usd        REAL,
        p_profitable             REAL    NOT NULL,
        gate_threshold           REAL    NOT NULL,
        risk_checks_json         TEXT    NOT NULL,
        executor_version         TEXT    NOT NULL,
        hold_days                INTEGER,
        session_id               TEXT    NOT NULL,
        PRIMARY KEY (asset, signal_timestamp)
    )
"""


def detect_state(conn) -> str:
    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'"
    ).fetchone() is None:
        return "table_missing"
    info = list(conn.execute("PRAGMA table_info(paper_trades)"))
    cols = {r[1] for r in info}
    pk = sorted(r[1] for r in info if r[5] > 0)
    pk_ok = pk == ["asset", "signal_timestamp"]
    if "session_id" in cols and BASE_COLS.issubset(cols) and pk_ok:
        return "ready"
    if BASE_COLS.issubset(cols) and "session_id" not in cols and pk_ok:
        n = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        return "rebuildable" if n == 0 else "has_rows"
    return "unknown"


def main() -> int:
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        state = detect_state(conn)
        if state == "ready":
            print("[migrate] paper_trades.session_id already present (NOT NULL). "
                  "Exiting cleanly.")
            return 0
        if state == "table_missing":
            print("[migrate] ERROR: paper_trades does not exist; run cycle51 "
                  "first.", file=sys.stderr)
            return 2
        if state == "unknown":
            cols = [(r[1], r[2]) for r in conn.execute("PRAGMA table_info(paper_trades)")]
            print(f"[migrate] ERROR: unexpected paper_trades schema: {cols}",
                  file=sys.stderr)
            return 3
        if state == "has_rows":
            n = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
            print(f"[migrate] ABORT: paper_trades has {n} row(s); a 0-row rebuild "
                  f"would drop booked trades. Backfill session_id + tighten "
                  f"manually, or snapshot first.", file=sys.stderr)
            return 5
        assert state == "rebuildable"

        print("[migrate] Rebuilding paper_trades (0 rows) -> session_id TEXT NOT NULL ...")
        conn.execute("BEGIN")
        conn.execute("ALTER TABLE paper_trades RENAME TO paper_trades_old_c54a")
        conn.execute(NEW_DDL)
        conn.execute("DROP TABLE paper_trades_old_c54a")
        conn.commit()

        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-rebuild state = {post}", file=sys.stderr)
            return 4
        print("[migrate] paper_trades rebuilt OK.")
        for r in conn.execute("PRAGMA table_info(paper_trades)"):
            print(f"  {r[1]:<26} {r[2]:<8} notnull={r[3]} pk={r[5]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
