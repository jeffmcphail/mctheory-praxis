"""
Cycle 53 -- funding_signals.min_pct_positive addition (Engine 7 config-gate).

D8 fix context (Cycle 53 D7 finding): the live monitor gates alerts on
argmax-P > gate ALONE, omitting the per-config hard thresholds
(min_funding_ann + min_pct_positive) that atlas Exp 13's run_funding_single_day
applies. The decomposition measured the resulting leak at 3.3% of signals at
gate 0.70 (SOL 33%) and 17% at gate 0.50 -- trades atlas zeroed but the live
system would book. The fix re-applies the config gate at BOTH the monitor
(alert emission) and the executor (11th risk check). For the executor to
JOIN-resolve the gate at decision time, funding_signals must carry all four
fields {ann_rate, pct_positive, min_funding_ann, min_pct_positive}.

Pre-fix funding_signals (Cycle 41 schema) already has three of the four:
  min_funding_ann, ann_rate, pct_positive
It is MISSING min_pct_positive (the argmax config's pct-positive threshold).
This migration adds it.

ALTER TABLE ADD COLUMN is safe here:
  - min_pct_positive is nullable; existing rows (written pre-fix) get NULL.
    The executor treats a NULL config field as "config not verifiable" ->
    config_thresholds_met=False -> skip (conservative; no real money on an
    unverifiable strategy definition).
  - SQLite ADD COLUMN does not rewrite the table.

Idempotent: re-running on a funding_signals that already has the column
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
        "SELECT name FROM sqlite_master WHERE type='table' AND name='funding_signals'"
    )
    if cur.fetchone() is None:
        return "table_missing"
    cur = conn.execute("PRAGMA table_info(funding_signals)")
    col_names = {r[1] for r in cur.fetchall()}
    if "min_pct_positive" in col_names:
        return "ready"
    # Require the three pre-existing config fields so we only ADD onto the
    # known Cycle 41 schema.
    required_pre = {"min_funding_ann", "ann_rate", "pct_positive",
                    "best_config_id", "p_profitable"}
    if required_pre.issubset(col_names):
        return "missing_column"
    return "unknown"


def main() -> int:
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        state = detect_state(conn)
        if state == "ready":
            print("[migrate] funding_signals.min_pct_positive already present. "
                  "Exiting cleanly.")
            return 0
        if state == "table_missing":
            print("[migrate] ERROR: funding_signals does not exist; run the "
                  "collector init_db first.", file=sys.stderr)
            return 2
        if state == "unknown":
            cur = conn.execute("PRAGMA table_info(funding_signals)")
            print(f"[migrate] ERROR: funding_signals has unexpected schema: "
                  f"{[r[1] for r in cur.fetchall()]}", file=sys.stderr)
            return 3
        assert state == "missing_column"

        n = conn.execute("SELECT COUNT(*) FROM funding_signals").fetchone()[0]
        print(f"[migrate] funding_signals row count pre-migration: {n}")
        if n > 0:
            print(f"[migrate] note: {n} existing row(s) will have "
                  f"min_pct_positive=NULL (treated as config-not-verifiable "
                  f"by the executor's 11th check).")

        print("[migrate] Adding min_pct_positive REAL (nullable) ...")
        conn.execute("ALTER TABLE funding_signals ADD COLUMN min_pct_positive REAL")
        conn.commit()

        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-alter state = {post}", file=sys.stderr)
            return 4
        print("[migrate] funding_signals.min_pct_positive added OK.")
        cur = conn.execute("PRAGMA table_info(funding_signals)")
        cols = [(r[1], r[2]) for r in cur.fetchall()]
        print(f"[migrate] Columns ({len(cols)}): "
              f"{', '.join(c[0] for c in cols)}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
