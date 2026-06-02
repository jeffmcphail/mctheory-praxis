"""
Cycle 52a -- paper_trades.hold_days addition (Engine 7 position lifecycle).

Adds a nullable `hold_days INTEGER` column to paper_trades. Captures the
RF-selected hold value (from funding_signals.hold_days) at decision time
so the entry record carries hold_days as part of its decision context.
Decouples paper_trades' audit trail from funding_signals' lifetime.

Sourcing rule (enforced by the executor, not this migration):
  - Default: JOIN-resolved from funding_signals on (asset, timestamp).
  - Override: --force-hold-days N CLI flag (for Cycle 53 backtests).
  - JOIN miss + no override: decision='skip', skip_reason='hold_days_unknown',
    hold_days column = NULL. No fallback default.

ALTER TABLE ADD COLUMN is safe here:
  - paper_trades currently has 0 rows (verified pre-migration); nothing
    to backfill.
  - hold_days is nullable (existing rows would be NULL if any).
  - SQLite ALTER TABLE ADD COLUMN doesn't rewrite the table; it just
    extends the schema.

Idempotent: re-running on a paper_trades that already has hold_days
exits cleanly.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def detect_state(conn) -> str:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'"
    )
    if cur.fetchone() is None:
        return "table_missing"
    cur = conn.execute("PRAGMA table_info(paper_trades)")
    col_names = {r[1] for r in cur.fetchall()}
    if "hold_days" in col_names:
        return "ready"
    expected_pre = {
        "asset", "signal_timestamp", "signal_datetime",
        "funding_alert_alerted_at", "decided_at", "decision",
        "skip_reason", "intended_direction", "intended_size_usd",
        "p_profitable", "gate_threshold", "risk_checks_json",
        "executor_version",
    }
    if expected_pre.issubset(col_names):
        return "missing_column"
    return "unknown"


def main() -> int:
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        state = detect_state(conn)
        if state == "ready":
            print("[migrate] paper_trades.hold_days already present. Exiting cleanly.")
            return 0
        if state == "table_missing":
            print("[migrate] ERROR: paper_trades does not exist; run cycle51_paper_trades_schema.py first.",
                  file=sys.stderr)
            return 2
        if state == "unknown":
            cur = conn.execute("PRAGMA table_info(paper_trades)")
            print(f"[migrate] ERROR: paper_trades has unexpected schema: "
                  f"{[r[1] for r in cur.fetchall()]}", file=sys.stderr)
            return 3
        assert state == "missing_column"

        # Sanity: pre-migration row count should be 0 (Cycle 51 had no real
        # entries; smoke tests cleaned up). If there are real rows already,
        # they will get NULL hold_days, which is fine for new column adds but
        # worth surfacing.
        n = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        print(f"[migrate] paper_trades row count pre-migration: {n}")
        if n > 0:
            print(f"[migrate] note: {n} existing rows will have hold_days=NULL.")

        print("[migrate] Adding hold_days INTEGER (nullable) ...")
        conn.execute("ALTER TABLE paper_trades ADD COLUMN hold_days INTEGER")
        conn.commit()

        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-alter state = {post}", file=sys.stderr)
            return 4
        print("[migrate] paper_trades.hold_days added OK.")
        cur = conn.execute("PRAGMA table_info(paper_trades)")
        cols = [(r[1], r[2], r[3], r[5]) for r in cur.fetchall()]
        print(f"[migrate] Columns ({len(cols)}):")
        for c in cols:
            print(f"  {c[0]:<26} {c[1]:<10} notnull={c[2]} pk={c[3]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
