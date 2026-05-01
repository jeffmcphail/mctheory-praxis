"""
Cycle 19 -- market_data table migration to Rule 35 standard.

Brief: claude/handoffs/BRIEF_market_data_migration.md (Task 1).

OLD schema (empty table; never had a working scheduled collector):
    CREATE TABLE market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        date TEXT NOT NULL,                 -- 'YYYY-MM-DD'
        market_cap REAL,
        total_volume REAL,
        circulating_supply REAL,
        total_supply REAL,
        ath REAL,
        ath_change_pct REAL,
        btc_dominance REAL,
        UNIQUE(asset, date)
    )

NEW schema (Rule 35 conforming):
    CREATE TABLE market_data (
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,         -- UTC milliseconds (UTC midnight)
        date TEXT NOT NULL,                 -- 'YYYY-MM-DD' (derived)
        market_cap REAL,
        total_volume REAL,
        circulating_supply REAL,
        total_supply REAL,
        ath REAL,
        ath_change_pct REAL,
        btc_dominance REAL,
        PRIMARY KEY (asset, timestamp)
    )

Idempotent: re-running on an already-migrated table prints "Already migrated"
and exits 0.

Empty-table schema-only migration: no rows to preserve. Still follows the
Cycle 17/18 transactional recipe for safety.

Usage:
    python scripts/migrations/cycle19_market_data_to_v2.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def get_columns(conn, table):
    """Return list of (name, type, pk_index) tuples from PRAGMA table_info."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    # PRAGMA columns: cid, name, type, notnull, dflt_value, pk
    return [(r[1], r[2], r[5]) for r in rows]


def detect_schema(conn):
    """Return 'old', 'new', 'missing', or 'unknown'."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'"
    )
    if cur.fetchone() is None:
        return "missing"

    cols = get_columns(conn, "market_data")
    col_names = {c[0] for c in cols}
    pk_cols = sorted([c[0] for c in cols if c[2] > 0])

    if "id" in col_names and pk_cols == ["id"] and "timestamp" not in col_names:
        return "old"
    if (
        "id" not in col_names
        and "timestamp" in col_names
        and pk_cols == ["asset", "timestamp"]
    ):
        return "new"
    return "unknown"


def main():
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Step 1: detect schema, idempotent guard
        schema = detect_schema(conn)
        if schema == "missing":
            print("[migrate] ERROR: market_data table does not exist", file=sys.stderr)
            return 2
        if schema == "new":
            print("[migrate] Already migrated -- market_data has new schema. Exiting cleanly.")
            return 0
        if schema == "unknown":
            cols = get_columns(conn, "market_data")
            print(f"[migrate] ERROR: market_data has unexpected schema: {cols}", file=sys.stderr)
            return 3

        assert schema == "old"
        print("[migrate] Detected OLD schema. Proceeding.")

        # Pre-migration snapshot (expected: empty)
        old_count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        old_cols = get_columns(conn, "market_data")
        print(f"[migrate] Pre-migration: rows={old_count}")
        print(f"[migrate] Pre-migration columns: {old_cols}")

        # Step 2-3: build new table, schema-only
        print("[migrate] Creating market_data_new with target schema...")
        conn.execute("BEGIN")
        try:
            conn.execute("DROP TABLE IF EXISTS market_data_new")
            conn.execute(
                """
                CREATE TABLE market_data_new (
                    asset TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    market_cap REAL,
                    total_volume REAL,
                    circulating_supply REAL,
                    total_supply REAL,
                    ath REAL,
                    ath_change_pct REAL,
                    btc_dominance REAL,
                    PRIMARY KEY (asset, timestamp)
                )
                """
            )

            # Copy rows if any (defense in depth -- expected to be 0)
            conn.execute(
                """
                INSERT INTO market_data_new
                    (asset, timestamp, date, market_cap, total_volume,
                     circulating_supply, total_supply, ath, ath_change_pct,
                     btc_dominance)
                SELECT asset,
                       CAST(strftime('%s', date || ' 00:00:00') AS INTEGER) * 1000,
                       date, market_cap, total_volume, circulating_supply,
                       total_supply, ath, ath_change_pct, btc_dominance
                FROM market_data
                """
            )

            new_count = conn.execute("SELECT COUNT(*) FROM market_data_new").fetchone()[0]
            print(f"[migrate] Post-copy rows in market_data_new: {new_count}")
            if new_count != old_count:
                raise RuntimeError(f"Row count mismatch: old={old_count}, new={new_count}")

            print("[migrate] Dropping old market_data and renaming market_data_new...")
            conn.execute("DROP TABLE market_data")
            conn.execute("ALTER TABLE market_data_new RENAME TO market_data")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Post-migration sanity check
        post_count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        post_cols = get_columns(conn, "market_data")
        post_schema = detect_schema(conn)
        print()
        print("=" * 60)
        print("[migrate] MIGRATION COMPLETE")
        print(f"  schema after migration: {post_schema}")
        print(f"  rows: {old_count} -> {post_count}")
        print(f"  columns: {post_cols}")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
