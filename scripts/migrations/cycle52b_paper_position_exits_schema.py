"""
Cycle 52b -- paper_position_exits schema (Engine 7 position lifecycle).

Creates a NEW table that joins to paper_trades on (asset, signal_timestamp).
One row per closed paper position; together with paper_trades.decision='enter'
rows, the pair gives an append-only entry/exit ledger. paper_trades itself
is never mutated post-write -- Schema-Option-B from Cycle 52 RECON.

A position is OPEN if:
    EXISTS (asset, signal_timestamp) IN paper_trades WHERE decision='enter'
    AND NOT EXISTS (asset, signal_timestamp) IN paper_position_exits

The executor's run_once() exit-reconciliation loop computes whether a
position has reached its hold-period end (wall-clock >= signal_timestamp
+ hold_days*86400000) and, if so, inserts a paper_position_exits row.

Sign convention (carried from atlas Exp 13 run_funding_hold):
  Direction = long spot + short perp -> we are SHORT perp.
  Funding rate POSITIVE -> longs pay shorts -> we RECEIVE.
  funding_payments_usd = sum(funding_rate_i * notional_usd) over events
  where signal_timestamp < event.timestamp <= target_exit_timestamp.
  Exclusive at entry, inclusive at exit (atlas L173-177 semantics).
  net_pnl_usd = funding_payments_usd - tc_entry_usd - tc_exit_usd.

Idempotent: re-running on an already-created table prints "Already exists"
and exits 0. INSERT OR IGNORE on the PK in the executor protects against
double-exits within the same window.

Schema:
    CREATE TABLE paper_position_exits (
        asset                  TEXT    NOT NULL,
        signal_timestamp       INTEGER NOT NULL,  -- FK-style ref to paper_trades
        entry_decided_at       TEXT    NOT NULL,  -- copied at exit time for forensic
        exit_decided_at        TEXT    NOT NULL,  -- ISO+00:00 wall clock of the exit-reconcile loop run
        exit_timestamp         INTEGER NOT NULL,  -- ms; target_exit_timestamp = signal_ts + hold_days*86400000
        exit_datetime          TEXT    NOT NULL,  -- ISO+00:00
        hold_days              INTEGER NOT NULL,  -- copied from paper_trades.hold_days
        funding_events_count   INTEGER NOT NULL,  -- COUNT(funding_rates rows that contributed)
        funding_payments_usd   REAL    NOT NULL,  -- SUM(rate * notional) over those events
        tc_entry_usd           REAL    NOT NULL,  -- 4 bps * notional (Cycle 52 baseline)
        tc_exit_usd            REAL    NOT NULL,  -- 4 bps * notional
        net_pnl_usd            REAL    NOT NULL,  -- funding_payments - tc_entry - tc_exit
        notional_usd           REAL    NOT NULL,  -- copied from paper_trades.intended_size_usd
        direction              TEXT    NOT NULL,  -- copied from paper_trades.intended_direction
        executor_version       TEXT    NOT NULL,  -- 'cycle52-lifecycle' (blame trail)
        PRIMARY KEY (asset, signal_timestamp)
    )

Acceptance:
  - Table exists with the exact schema above
  - PK enforced (rejects duplicate inserts via INSERT OR IGNORE)
  - Idempotent re-run
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def detect_state(conn) -> str:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_position_exits'"
    )
    if cur.fetchone() is None:
        return "missing"
    cur = conn.execute("PRAGMA table_info(paper_position_exits)")
    cols = [(r[1], r[2], r[5]) for r in cur.fetchall()]
    col_names = {c[0] for c in cols}
    pk_cols = sorted([c[0] for c in cols if c[2] > 0])
    expected_cols = {
        "asset", "signal_timestamp", "entry_decided_at", "exit_decided_at",
        "exit_timestamp", "exit_datetime", "hold_days", "funding_events_count",
        "funding_payments_usd", "tc_entry_usd", "tc_exit_usd", "net_pnl_usd",
        "notional_usd", "direction", "executor_version",
    }
    if expected_cols.issubset(col_names) and pk_cols == ["asset", "signal_timestamp"]:
        return "ready"
    return "unknown"


def main() -> int:
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        state = detect_state(conn)
        if state == "ready":
            print("[migrate] paper_position_exits already exists with the expected schema. Exiting cleanly.")
            return 0
        if state == "unknown":
            cur = conn.execute("PRAGMA table_info(paper_position_exits)")
            cols = [(r[1], r[2], r[5]) for r in cur.fetchall()]
            print(f"[migrate] ERROR: paper_position_exits exists with unexpected schema: {cols}",
                  file=sys.stderr)
            return 3
        assert state == "missing"

        print("[migrate] Creating paper_position_exits table...")
        conn.execute("""
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
                PRIMARY KEY (asset, signal_timestamp)
            )
        """)
        conn.commit()
        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-create state = {post}", file=sys.stderr)
            return 4
        print("[migrate] paper_position_exits created OK.")
        cur = conn.execute("PRAGMA table_info(paper_position_exits)")
        cols = [(r[1], r[2], r[3], r[5]) for r in cur.fetchall()]
        print(f"[migrate] Columns ({len(cols)}):")
        for c in cols:
            print(f"  {c[0]:<26} {c[1]:<10} notnull={c[2]} pk={c[3]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
