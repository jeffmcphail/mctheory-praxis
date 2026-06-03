"""
Cycle 54b -- paper_position_exits.session_id (Engine 7 session attribution).

Companion to cycle54a; same rationale + mechanics. R2 decision: session_id
REQUIRED (NOT NULL). SQLite can't ADD a NOT NULL column without a sentinel
default, so at 0 rows we do a guarded 0-row REBUILD appending
session_id TEXT NOT NULL.

GUARDED: rebuild only if known pre-Cycle-54 schema AND COUNT(*) == 0; else
ABORT (never drop booked exits silently).

Idempotent: once session_id (NOT NULL) is present, re-running exits 0.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"

BASE_COLS = {
    "asset", "signal_timestamp", "entry_decided_at", "exit_decided_at",
    "exit_timestamp", "exit_datetime", "hold_days", "funding_events_count",
    "funding_payments_usd", "tc_entry_usd", "tc_exit_usd", "net_pnl_usd",
    "notional_usd", "direction", "executor_version",
}

NEW_DDL = """
    CREATE TABLE paper_position_exits (
        asset                TEXT    NOT NULL,
        signal_timestamp     INTEGER NOT NULL,
        entry_decided_at     TEXT    NOT NULL,
        exit_decided_at      TEXT    NOT NULL,
        exit_timestamp       INTEGER NOT NULL,
        exit_datetime        TEXT    NOT NULL,
        hold_days            INTEGER NOT NULL,
        funding_events_count INTEGER NOT NULL,
        funding_payments_usd REAL    NOT NULL,
        tc_entry_usd         REAL    NOT NULL,
        tc_exit_usd          REAL    NOT NULL,
        net_pnl_usd          REAL    NOT NULL,
        notional_usd         REAL    NOT NULL,
        direction            TEXT    NOT NULL,
        executor_version     TEXT    NOT NULL,
        session_id           TEXT    NOT NULL,
        PRIMARY KEY (asset, signal_timestamp)
    )
"""


def detect_state(conn) -> str:
    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='paper_position_exits'"
    ).fetchone() is None:
        return "table_missing"
    info = list(conn.execute("PRAGMA table_info(paper_position_exits)"))
    cols = {r[1] for r in info}
    pk = sorted(r[1] for r in info if r[5] > 0)
    pk_ok = pk == ["asset", "signal_timestamp"]
    if "session_id" in cols and BASE_COLS.issubset(cols) and pk_ok:
        return "ready"
    if BASE_COLS.issubset(cols) and "session_id" not in cols and pk_ok:
        n = conn.execute("SELECT COUNT(*) FROM paper_position_exits").fetchone()[0]
        return "rebuildable" if n == 0 else "has_rows"
    return "unknown"


def main() -> int:
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        state = detect_state(conn)
        if state == "ready":
            print("[migrate] paper_position_exits.session_id already present "
                  "(NOT NULL). Exiting cleanly.")
            return 0
        if state == "table_missing":
            print("[migrate] ERROR: paper_position_exits does not exist; run "
                  "cycle52b first.", file=sys.stderr)
            return 2
        if state == "unknown":
            cols = [(r[1], r[2]) for r in
                    conn.execute("PRAGMA table_info(paper_position_exits)")]
            print(f"[migrate] ERROR: unexpected paper_position_exits schema: "
                  f"{cols}", file=sys.stderr)
            return 3
        if state == "has_rows":
            n = conn.execute("SELECT COUNT(*) FROM paper_position_exits").fetchone()[0]
            print(f"[migrate] ABORT: paper_position_exits has {n} row(s); a 0-row "
                  f"rebuild would drop booked exits. Backfill manually or "
                  f"snapshot first.", file=sys.stderr)
            return 5
        assert state == "rebuildable"

        print("[migrate] Rebuilding paper_position_exits (0 rows) -> "
              "session_id TEXT NOT NULL ...")
        conn.execute("BEGIN")
        conn.execute("ALTER TABLE paper_position_exits "
                     "RENAME TO paper_position_exits_old_c54b")
        conn.execute(NEW_DDL)
        conn.execute("DROP TABLE paper_position_exits_old_c54b")
        conn.commit()

        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-rebuild state = {post}", file=sys.stderr)
            return 4
        print("[migrate] paper_position_exits rebuilt OK.")
        for r in conn.execute("PRAGMA table_info(paper_position_exits)"):
            print(f"  {r[1]:<26} {r[2]:<8} notnull={r[3]} pk={r[5]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
