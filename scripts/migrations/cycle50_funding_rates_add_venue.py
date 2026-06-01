"""
Cycle 50 D1a -- funding_rates schema: add `venue` column to PK.

Adapted from scripts/migrations/cycle21_funding_rates_to_v2.py.

OLD schema (post Cycle 21 Rule-35 migration):
    CREATE TABLE funding_rates (
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,      -- UTC ms
        datetime TEXT NOT NULL,          -- ISO+00:00
        funding_rate REAL,
        PRIMARY KEY (asset, timestamp)
    )

NEW schema (Cycle 50 cross-venue extension):
    CREATE TABLE funding_rates (
        asset TEXT NOT NULL,
        venue TEXT NOT NULL,             -- NEW; backfills existing rows with 'binance'
        timestamp INTEGER NOT NULL,
        datetime TEXT NOT NULL,
        funding_rate REAL,
        PRIMARY KEY (asset, venue, timestamp)
    )

Idempotent: re-running on an already-migrated table prints
"Already migrated" and exits 0.

Acceptance:
  - Pre/post row counts match exactly
  - Every existing row's venue = 'binance' post-migration
  - Per-asset counts unchanged (since only Binance rows exist pre-migration)
  - Latest row's (timestamp, asset, funding_rate) preserved
  - PK (asset, venue, timestamp) enforced on the new table
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def get_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [(r[1], r[2], r[5]) for r in cur.fetchall()]


def detect_schema(conn):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='funding_rates'"
    )
    if cur.fetchone() is None:
        return "missing"
    cols = get_columns(conn, "funding_rates")
    col_names = {c[0] for c in cols}
    pk_cols = sorted([c[0] for c in cols if c[2] > 0])
    if "venue" in col_names and pk_cols == ["asset", "timestamp", "venue"]:
        return "new"
    if "venue" not in col_names and pk_cols == ["asset", "timestamp"]:
        return "old"
    return "unknown"


def fetch_latest(conn, table):
    cur = conn.execute(
        f"SELECT asset, timestamp, datetime, funding_rate "
        f"FROM {table} ORDER BY timestamp DESC, asset ASC LIMIT 1"
    )
    return cur.fetchone()


def main():
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        schema = detect_schema(conn)
        if schema == "missing":
            print("[migrate] ERROR: funding_rates table does not exist", file=sys.stderr)
            return 2
        if schema == "new":
            print("[migrate] Already migrated -- funding_rates has venue PK. Exiting cleanly.")
            return 0
        if schema == "unknown":
            cols = get_columns(conn, "funding_rates")
            print(f"[migrate] ERROR: funding_rates has unexpected schema: {cols}", file=sys.stderr)
            return 3

        assert schema == "old"
        print("[migrate] Detected OLD schema (no venue column). Proceeding.")

        old_count = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
        old_per_asset = conn.execute(
            "SELECT asset, COUNT(*) FROM funding_rates GROUP BY asset ORDER BY asset"
        ).fetchall()
        old_latest = fetch_latest(conn, "funding_rates")
        old_min = conn.execute("SELECT MIN(timestamp) FROM funding_rates").fetchone()[0]
        old_max = conn.execute("SELECT MAX(timestamp) FROM funding_rates").fetchone()[0]
        print(f"[migrate] Pre-migration: rows={old_count}, ts_range={old_min}..{old_max}")
        print(f"[migrate] Pre-migration per-asset: {old_per_asset}")
        print(f"[migrate] Pre-migration latest: {old_latest}")

        print("[migrate] Creating funding_rates_new with venue PK; copying rows with venue='binance'...")
        conn.execute("BEGIN")
        try:
            conn.execute("""
                CREATE TABLE funding_rates_new (
                    asset TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    funding_rate REAL,
                    PRIMARY KEY (asset, venue, timestamp)
                )
            """)
            conn.execute("""
                INSERT INTO funding_rates_new (asset, venue, timestamp, datetime, funding_rate)
                SELECT asset, 'binance' AS venue, timestamp, datetime, funding_rate
                FROM funding_rates
            """)
            new_count = conn.execute("SELECT COUNT(*) FROM funding_rates_new").fetchone()[0]
            if new_count != old_count:
                raise RuntimeError(f"Row count mismatch: old={old_count} new={new_count}")
            new_per_asset = conn.execute(
                "SELECT asset, COUNT(*) FROM funding_rates_new GROUP BY asset ORDER BY asset"
            ).fetchall()
            if new_per_asset != old_per_asset:
                raise RuntimeError(f"Per-asset mismatch: old={old_per_asset} new={new_per_asset}")
            venues = conn.execute(
                "SELECT DISTINCT venue FROM funding_rates_new"
            ).fetchall()
            if venues != [("binance",)]:
                raise RuntimeError(f"Venue values not all 'binance': {venues}")
            conn.execute("DROP TABLE funding_rates")
            conn.execute("ALTER TABLE funding_rates_new RENAME TO funding_rates")
            conn.execute("COMMIT")
            print(f"[migrate] Commit OK. {new_count} rows now in funding_rates with venue PK.")
        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"[migrate] ROLLBACK due to: {e}", file=sys.stderr)
            return 4

        # Post-migration verification (outside the transaction)
        post_count = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
        post_per_asset = conn.execute(
            "SELECT asset, COUNT(*) FROM funding_rates GROUP BY asset ORDER BY asset"
        ).fetchall()
        post_per_venue = conn.execute(
            "SELECT venue, COUNT(*) FROM funding_rates GROUP BY venue ORDER BY venue"
        ).fetchall()
        post_latest = fetch_latest(conn, "funding_rates")
        post_cols = get_columns(conn, "funding_rates")
        post_pk = sorted([c[0] for c in post_cols if c[2] > 0])
        print()
        print(f"[migrate] Post-migration: rows={post_count}")
        print(f"[migrate] Post-migration per-asset: {post_per_asset}")
        print(f"[migrate] Post-migration per-venue: {post_per_venue}")
        print(f"[migrate] Post-migration latest: {post_latest}")
        print(f"[migrate] Post-migration columns: {post_cols}")
        print(f"[migrate] Post-migration PK: {post_pk}")
        assert post_count == old_count, "row count drift"
        assert post_per_asset == old_per_asset, "per-asset drift"
        assert post_pk == ["asset", "timestamp", "venue"], "PK shape wrong"
        print()
        print("[migrate] All assertions OK. Migration complete.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
