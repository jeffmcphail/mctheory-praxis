"""
Cycle 34 -- info_bars table schema migration.

Creates `info_bars` in data/crypto_data.db. Rule 35 conforming
(ms timestamps + ISO datetimes + natural-key PK).

Single generic table parameterized by (asset, bar_type,
threshold_value). New thresholds = new rows, not new tables.

Idempotent: detects already-applied state and exits cleanly.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS info_bars (
    asset            TEXT    NOT NULL,
    bar_type         TEXT    NOT NULL,
    threshold_value  REAL    NOT NULL,
    bar_index        INTEGER NOT NULL,
    start_timestamp  INTEGER NOT NULL,
    end_timestamp    INTEGER NOT NULL,
    start_datetime   TEXT    NOT NULL,
    end_datetime     TEXT    NOT NULL,
    open             REAL    NOT NULL,
    high             REAL    NOT NULL,
    low              REAL    NOT NULL,
    close            REAL    NOT NULL,
    base_volume      REAL    NOT NULL,
    quote_volume     REAL    NOT NULL,
    tick_count       INTEGER NOT NULL,
    buy_quote        REAL    NOT NULL,
    sell_quote       REAL    NOT NULL,
    imbalance_quote  REAL    NOT NULL,
    PRIMARY KEY (asset, bar_type, threshold_value, bar_index)
)
"""

INDEX_LOOKUP_SQL = """
CREATE INDEX IF NOT EXISTS idx_info_bars_lookup
    ON info_bars (asset, bar_type, threshold_value, end_timestamp)
"""

INDEX_END_TS_SQL = """
CREATE INDEX IF NOT EXISTS idx_info_bars_end_ts
    ON info_bars (end_timestamp)
"""


def main() -> int:
    print(f"[cycle34] Opening {DB_PATH}")
    if not DB_PATH.exists():
        print(f"[cycle34] ABORT: {DB_PATH} doesn't exist.",
              file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Pre-state: does the table exist?
        existing = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='info_bars'"
        ).fetchone()

        if existing:
            cols = conn.execute(
                "PRAGMA table_info(info_bars)"
            ).fetchall()
            print(f"[cycle34] info_bars already exists with "
                  f"{len(cols)} columns. Confirming indexes...")
            conn.execute(INDEX_LOOKUP_SQL)
            conn.execute(INDEX_END_TS_SQL)
            conn.commit()
            print(f"[cycle34] Indexes ensured. Exiting cleanly.")
            return 0

        # Create table + indexes
        print(f"[cycle34] Creating info_bars table...")
        conn.execute("BEGIN")
        try:
            conn.execute(SCHEMA_SQL)
            conn.execute(INDEX_LOOKUP_SQL)
            conn.execute(INDEX_END_TS_SQL)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Verification
        cols = conn.execute(
            "PRAGMA table_info(info_bars)"
        ).fetchall()
        idxs = conn.execute(
            "PRAGMA index_list(info_bars)"
        ).fetchall()
        print(f"[cycle34] Post-state: {len(cols)} columns, "
              f"{len(idxs)} indexes.")
        print()
        print("=" * 60)
        print("[cycle34] INFO_BARS TABLE CREATED")
        print("  Columns:", ", ".join(c[1] for c in cols))
        print("  Indexes:", ", ".join(i[1] for i in idxs))
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
