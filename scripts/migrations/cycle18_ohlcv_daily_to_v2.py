"""
Cycle 18 -- ohlcv_daily table migration to Rule 35 standard.

Brief: claude/handoffs/BRIEF_ohlcv_daily_migration.md (Task 2).

OLD schema:
    CREATE TABLE ohlcv_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,        -- UTC seconds
        date TEXT NOT NULL,                 -- 'YYYY-MM-DD'
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        UNIQUE(asset, timestamp)
    )

NEW schema (Rule 35 conforming):
    CREATE TABLE ohlcv_daily (
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,        -- UTC milliseconds
        date TEXT NOT NULL,                 -- 'YYYY-MM-DD' (UTC midnight)
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        PRIMARY KEY (asset, timestamp)
    )

Idempotent: re-running on an already-migrated table prints "Already migrated"
and exits 0.

Usage:
    python scripts/migrations/cycle18_ohlcv_daily_to_v2.py

Acceptance criteria:
- Pre/post row counts match (1,802 expected: 901 BTC + 901 ETH)
- Latest UTC moment unchanged within 1-second precision
- Compound PK (asset, timestamp); no id column
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def get_columns(conn, table):
    """Return list of (name, type, pk_index) tuples from PRAGMA table_info."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    # PRAGMA columns: cid, name, type, notnull, dflt_value, pk
    # pk is 0 for non-PK columns, otherwise the 1-based position in the PK
    return [(r[1], r[2], r[5]) for r in rows]


def detect_schema(conn):
    """Return 'old', 'new', or 'missing'."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_daily'"
    )
    if cur.fetchone() is None:
        return "missing"

    cols = get_columns(conn, "ohlcv_daily")
    col_names = {c[0] for c in cols}
    pk_cols = sorted([c[0] for c in cols if c[2] > 0])

    if "id" in col_names and pk_cols == ["id"]:
        return "old"
    if "id" not in col_names and pk_cols == ["asset", "timestamp"]:
        return "new"
    return "unknown"


def fetch_latest(conn, table):
    """Return (asset, timestamp, date, open, high, low, close, volume) for the
    row with the largest timestamp (ties broken by asset alphabetically)."""
    cur = conn.execute(
        f"SELECT asset, timestamp, date, open, high, low, close, volume "
        f"FROM {table} ORDER BY timestamp DESC, asset ASC LIMIT 1"
    )
    return cur.fetchone()


def main():
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Step 1+2: detect schema, idempotent guard
        schema = detect_schema(conn)
        if schema == "missing":
            print("[migrate] ERROR: ohlcv_daily table does not exist", file=sys.stderr)
            return 2
        if schema == "new":
            print("[migrate] Already migrated -- ohlcv_daily has new schema. Exiting cleanly.")
            return 0
        if schema == "unknown":
            cols = get_columns(conn, "ohlcv_daily")
            print(f"[migrate] ERROR: ohlcv_daily has unexpected schema: {cols}", file=sys.stderr)
            return 3

        assert schema == "old"
        print("[migrate] Detected OLD schema. Proceeding.")

        # Pre-migration snapshot
        old_count = conn.execute("SELECT COUNT(*) FROM ohlcv_daily").fetchone()[0]
        old_min = conn.execute("SELECT MIN(timestamp) FROM ohlcv_daily").fetchone()[0]
        old_max = conn.execute("SELECT MAX(timestamp) FROM ohlcv_daily").fetchone()[0]
        old_per_asset = conn.execute(
            "SELECT asset, COUNT(*) FROM ohlcv_daily GROUP BY asset ORDER BY asset"
        ).fetchall()
        old_latest = fetch_latest(conn, "ohlcv_daily")
        old_latest_utc = datetime.fromtimestamp(old_latest[1], tz=timezone.utc)
        print(f"[migrate] Pre-migration: rows={old_count}, ts_min={old_min}, ts_max={old_max}")
        print(f"[migrate] Pre-migration per-asset: {old_per_asset}")
        print(f"[migrate] Pre-migration latest row: {old_latest}")
        print(f"[migrate] Pre-migration latest UTC: {old_latest_utc.isoformat()}")

        # Step 4-5: build new table, copy with seconds*1000
        print("[migrate] Creating ohlcv_daily_new and copying rows (timestamp *= 1000)...")
        conn.execute("BEGIN")
        try:
            conn.execute("DROP TABLE IF EXISTS ohlcv_daily_new")
            conn.execute(
                """
                CREATE TABLE ohlcv_daily_new (
                    asset TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL,
                    PRIMARY KEY (asset, timestamp)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO ohlcv_daily_new
                    (asset, timestamp, date, open, high, low, close, volume)
                SELECT asset, timestamp * 1000, date, open, high, low, close, volume
                FROM ohlcv_daily
                """
            )

            # Step 6: row-count verification (overall and per-asset)
            new_count = conn.execute("SELECT COUNT(*) FROM ohlcv_daily_new").fetchone()[0]
            new_per_asset = conn.execute(
                "SELECT asset, COUNT(*) FROM ohlcv_daily_new GROUP BY asset ORDER BY asset"
            ).fetchall()
            print(f"[migrate] Post-copy rows in ohlcv_daily_new: {new_count}")
            print(f"[migrate] Post-copy per-asset: {new_per_asset}")
            if new_count != old_count:
                raise RuntimeError(f"Row count mismatch: old={old_count}, new={new_count}")
            if new_per_asset != old_per_asset:
                raise RuntimeError(
                    f"Per-asset count mismatch: old={old_per_asset}, new={new_per_asset}"
                )

            # Step 7: latest UTC moment cross-check
            new_latest = fetch_latest(conn, "ohlcv_daily_new")
            print(f"[migrate] Post-copy latest row: {new_latest}")
            new_latest_ms = new_latest[1]
            new_latest_utc = datetime.fromtimestamp(new_latest_ms / 1000, tz=timezone.utc)
            print(f"[migrate] Post-copy latest UTC: {new_latest_utc.isoformat()}")

            delta_s = abs(new_latest_ms / 1000 - old_latest[1])
            if delta_s > 1.0:
                raise RuntimeError(
                    f"Latest UTC moment drift > 1s: old={old_latest[1]}, new_ms={new_latest_ms}"
                )

            # Date-string cross-check: derive date from ms timestamp and compare
            derived_date = new_latest_utc.strftime("%Y-%m-%d")
            if derived_date != new_latest[2]:
                raise RuntimeError(
                    f"Date text mismatch: derived={derived_date}, stored={new_latest[2]}"
                )

            # OHLCV value cross-check: latest row values preserved exactly
            for i, label in enumerate(["open", "high", "low", "close", "volume"], start=3):
                if old_latest[i] != new_latest[i]:
                    raise RuntimeError(
                        f"{label} mismatch on latest row: "
                        f"old={old_latest[i]}, new={new_latest[i]}"
                    )

            # Step 8-9: drop old, rename new
            print("[migrate] Dropping old ohlcv_daily and renaming ohlcv_daily_new...")
            conn.execute("DROP TABLE ohlcv_daily")
            conn.execute("ALTER TABLE ohlcv_daily_new RENAME TO ohlcv_daily")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Step 10: post-migration sanity check on the renamed table
        post_count = conn.execute("SELECT COUNT(*) FROM ohlcv_daily").fetchone()[0]
        post_latest = fetch_latest(conn, "ohlcv_daily")
        post_latest_utc = datetime.fromtimestamp(post_latest[1] / 1000, tz=timezone.utc)
        post_schema = detect_schema(conn)
        print()
        print("=" * 60)
        print("[migrate] MIGRATION COMPLETE")
        print(f"  schema after migration: {post_schema}")
        print(f"  rows: {old_count} -> {post_count}")
        print(f"  latest pre:  asset={old_latest[0]} ts={old_latest[1]} (s)  UTC={old_latest_utc.isoformat()}")
        print(f"  latest post: asset={post_latest[0]} ts={post_latest[1]} (ms) UTC={post_latest_utc.isoformat()}")
        print(f"  delta from pre: {abs(post_latest[1] / 1000 - old_latest[1])} seconds")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
