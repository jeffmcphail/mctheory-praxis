"""
Cycle 51 -- paper_trades schema (Engine 7 paper-trading executor scaffold).

Creates a NEW table; not a recreate-table migration. The Rule-35-conforming
compound PK on (asset, signal_timestamp) mirrors funding_alerts so each
(asset, funding-window) gets exactly one decision row. INSERT OR IGNORE
in the executor makes re-runs idempotent.

Idempotent: re-running on an already-created table prints "Already exists"
and exits 0.

Schema:
    CREATE TABLE paper_trades (
        asset                     TEXT NOT NULL,
        signal_timestamp          INTEGER NOT NULL,   -- ms epoch; matches funding_alerts PK
        signal_datetime           TEXT NOT NULL,      -- ISO+00:00; matches funding_alerts.datetime
        funding_alert_alerted_at  TEXT NOT NULL,      -- ISO+00:00 from funding_alerts (Cycle 51 refinement)
        decided_at                TEXT NOT NULL,      -- ISO+00:00; executor's decision wall-clock
        decision                  TEXT NOT NULL,      -- 'enter' or 'skip'
        skip_reason               TEXT,               -- NULL when decision='enter'; '; '-joined reasons otherwise
        intended_direction        TEXT,               -- 'long_spot_short_perp' (Exp 13) or NULL on skip
        intended_size_usd         REAL,               -- per-leg notional; 0 on skip
        p_profitable              REAL NOT NULL,      -- from funding_alerts
        gate_threshold            REAL NOT NULL,      -- from funding_alerts (typically 0.70)
        risk_checks_json          TEXT NOT NULL,      -- JSON dict of all 9 risk-check outcomes
        executor_version          TEXT NOT NULL,      -- 'cycle51-paper-scaffold' (blame trail)
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
        "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'"
    )
    if cur.fetchone() is None:
        return "missing"
    cur = conn.execute("PRAGMA table_info(paper_trades)")
    cols = [(r[1], r[2], r[5]) for r in cur.fetchall()]
    col_names = {c[0] for c in cols}
    pk_cols = sorted([c[0] for c in cols if c[2] > 0])
    expected_cols = {
        "asset", "signal_timestamp", "signal_datetime", "funding_alert_alerted_at",
        "decided_at", "decision", "skip_reason", "intended_direction",
        "intended_size_usd", "p_profitable", "gate_threshold",
        "risk_checks_json", "executor_version",
    }
    if expected_cols.issubset(col_names) and pk_cols == ["asset", "signal_timestamp"]:
        return "ready"
    return "unknown"


def main():
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        state = detect_state(conn)
        if state == "ready":
            print("[migrate] paper_trades already exists with the expected schema. Exiting cleanly.")
            return 0
        if state == "unknown":
            cur = conn.execute("PRAGMA table_info(paper_trades)")
            cols = [(r[1], r[2], r[5]) for r in cur.fetchall()]
            print(f"[migrate] ERROR: paper_trades exists with unexpected schema: {cols}", file=sys.stderr)
            return 3
        assert state == "missing"
        print("[migrate] Creating paper_trades table...")
        conn.execute("""
            CREATE TABLE paper_trades (
                asset                    TEXT NOT NULL,
                signal_timestamp         INTEGER NOT NULL,
                signal_datetime          TEXT NOT NULL,
                funding_alert_alerted_at TEXT NOT NULL,
                decided_at               TEXT NOT NULL,
                decision                 TEXT NOT NULL,
                skip_reason              TEXT,
                intended_direction       TEXT,
                intended_size_usd        REAL,
                p_profitable             REAL NOT NULL,
                gate_threshold           REAL NOT NULL,
                risk_checks_json         TEXT NOT NULL,
                executor_version         TEXT NOT NULL,
                PRIMARY KEY (asset, signal_timestamp)
            )
        """)
        conn.commit()
        # Verify
        post = detect_state(conn)
        if post != "ready":
            print(f"[migrate] ERROR: post-create state = {post}", file=sys.stderr)
            return 4
        print("[migrate] paper_trades created OK.")
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
