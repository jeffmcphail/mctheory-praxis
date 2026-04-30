"""
Cycle 17 -- fear_greed table migration to Rule 35 standard.

Brief: claude/handoffs/BRIEF_temporal_standard_pilot.md (Task 2).

OLD schema:
    CREATE TABLE fear_greed (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,        -- UTC seconds
        date TEXT NOT NULL,                 -- YYYY-MM-DD
        value INTEGER,
        classification TEXT,
        UNIQUE(timestamp)
    )

NEW schema (Rule 35 conforming):
    CREATE TABLE fear_greed (
        timestamp INTEGER PRIMARY KEY,     -- UTC milliseconds
        date TEXT NOT NULL,                 -- YYYY-MM-DD (UTC midnight, derived)
        value INTEGER,
        classification TEXT
    )

Idempotent: re-running on an already-migrated table prints "Already migrated"
and exits 0.

Usage:
    python scripts/migrations/cycle17_fear_greed_to_v2.py

Acceptance criteria:
- Pre/post row counts match (901 expected)
- Latest UTC moment unchanged within 1-second precision
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def get_columns(conn, table):
    """Return list of (name, type, pk) tuples from PRAGMA table_info."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    # PRAGMA columns: cid, name, type, notnull, dflt_value, pk
    return [(r[1], r[2], r[5]) for r in rows]


def detect_schema(conn):
    """Return 'old', 'new', or 'missing'."""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fear_greed'")
    if cur.fetchone() is None:
        return "missing"

    cols = get_columns(conn, "fear_greed")
    col_names = {c[0] for c in cols}
    pk_cols = [c[0] for c in cols if c[2] > 0]

    if "id" in col_names and pk_cols == ["id"]:
        return "old"
    if "id" not in col_names and pk_cols == ["timestamp"]:
        return "new"
    return "unknown"


def fetch_latest(conn, table):
    """Return (timestamp, date, value, classification) for the row with the largest timestamp."""
    cur = conn.execute(
        f"SELECT timestamp, date, value, classification FROM {table} ORDER BY timestamp DESC LIMIT 1"
    )
    return cur.fetchone()


def main():
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Step 1+2: detect schema
        schema = detect_schema(conn)
        if schema == "missing":
            print("[migrate] ERROR: fear_greed table does not exist", file=sys.stderr)
            return 2
        if schema == "new":
            print("[migrate] Already migrated -- fear_greed has new schema. Exiting cleanly.")
            return 0
        if schema == "unknown":
            cols = get_columns(conn, "fear_greed")
            print(f"[migrate] ERROR: fear_greed has unexpected schema: {cols}", file=sys.stderr)
            return 3

        assert schema == "old"
        print("[migrate] Detected OLD schema. Proceeding.")

        # Pre-migration snapshot
        old_count = conn.execute("SELECT COUNT(*) FROM fear_greed").fetchone()[0]
        old_latest = fetch_latest(conn, "fear_greed")
        old_min = conn.execute("SELECT MIN(timestamp) FROM fear_greed").fetchone()[0]
        old_max = conn.execute("SELECT MAX(timestamp) FROM fear_greed").fetchone()[0]
        print(f"[migrate] Pre-migration: rows={old_count}, ts_min={old_min}, ts_max={old_max}")
        print(f"[migrate] Pre-migration latest row: {old_latest}")

        old_latest_utc = datetime.fromtimestamp(old_latest[0], tz=timezone.utc)
        print(f"[migrate] Pre-migration latest UTC: {old_latest_utc.isoformat()}")

        # Step 4-5: build new table, copy with seconds*1000
        print("[migrate] Creating fear_greed_new and copying rows (timestamp *= 1000)...")
        conn.execute("BEGIN")
        try:
            conn.execute("DROP TABLE IF EXISTS fear_greed_new")
            conn.execute(
                """
                CREATE TABLE fear_greed_new (
                    timestamp INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    value INTEGER,
                    classification TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO fear_greed_new (timestamp, date, value, classification)
                SELECT timestamp * 1000, date, value, classification FROM fear_greed
                """
            )

            # Step 6: row-count verification
            new_count = conn.execute("SELECT COUNT(*) FROM fear_greed_new").fetchone()[0]
            print(f"[migrate] Post-copy rows in fear_greed_new: {new_count}")
            if new_count != old_count:
                raise RuntimeError(f"Row count mismatch: old={old_count}, new={new_count}")

            # Step 7: latest UTC moment cross-check
            new_latest = fetch_latest(conn, "fear_greed_new")
            print(f"[migrate] Post-copy latest row: {new_latest}")
            new_latest_ms = new_latest[0]
            new_latest_utc = datetime.fromtimestamp(new_latest_ms / 1000, tz=timezone.utc)
            print(f"[migrate] Post-copy latest UTC: {new_latest_utc.isoformat()}")

            delta_s = abs(new_latest_ms / 1000 - old_latest[0])
            if delta_s > 1.0:
                raise RuntimeError(
                    f"Latest UTC moment drift > 1s: old={old_latest[0]}, new_ms={new_latest_ms}"
                )

            # Date-string cross-check: derive date from ms timestamp and compare
            derived_date = new_latest_utc.strftime("%Y-%m-%d")
            if derived_date != new_latest[1]:
                raise RuntimeError(
                    f"Date text mismatch: derived={derived_date}, stored={new_latest[1]}"
                )

            # Step 8-9: drop old, rename new
            print("[migrate] Dropping old fear_greed and renaming fear_greed_new...")
            conn.execute("DROP TABLE fear_greed")
            conn.execute("ALTER TABLE fear_greed_new RENAME TO fear_greed")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Step 10: post-migration sanity check on the renamed table
        post_count = conn.execute("SELECT COUNT(*) FROM fear_greed").fetchone()[0]
        post_latest = fetch_latest(conn, "fear_greed")
        post_latest_utc = datetime.fromtimestamp(post_latest[0] / 1000, tz=timezone.utc)
        post_schema = detect_schema(conn)
        print()
        print("=" * 60)
        print("[migrate] MIGRATION COMPLETE")
        print(f"  schema after migration: {post_schema}")
        print(f"  rows: {old_count} -> {post_count}")
        print(f"  latest pre:  ts={old_latest[0]} (s)   UTC={old_latest_utc.isoformat()}")
        print(f"  latest post: ts={post_latest[0]} (ms) UTC={post_latest_utc.isoformat()}")
        print(f"  delta from pre: {abs(post_latest[0] / 1000 - old_latest[0])} seconds")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
