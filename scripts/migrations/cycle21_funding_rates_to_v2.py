"""
Cycle 21 -- funding_rates table migration to Rule 35 standard.

Adapted from scripts/migrations/cycle20_ohlcv_4h_to_v2.py.

OLD schema:
    CREATE TABLE funding_rates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,        -- UTC seconds
        datetime TEXT NOT NULL,             -- 'YYYY-MM-DD HH:MM:SS' (naive)
        funding_rate REAL,
        UNIQUE(asset, timestamp)
    )

NEW schema (Rule 35 conforming):
    CREATE TABLE funding_rates (
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,        -- UTC milliseconds
        datetime TEXT NOT NULL,             -- 'YYYY-MM-DDTHH:MM:SS+00:00' (ISO)
        funding_rate REAL,
        PRIMARY KEY (asset, timestamp)
    )

Same recipe as cycle 20 (ohlcv_4h):
  (1) datetime is re-derived from `timestamp` via SQLite strftime
      (defense-in-depth, not a copy of the old naive text).
  (2) timestamp seconds -> milliseconds via `* 1000`.
  (3) Drop `id` AUTOINCREMENT, compound PK on `(asset, timestamp)`.

Idempotent: re-running on an already-migrated table prints "Already migrated"
and exits 0.

Acceptance:
  - Pre/post row counts match (~2,212 expected; growth possible since brief)
  - Pre/post per-asset counts match
  - Latest UTC moment unchanged within 1-second precision
  - New datetime renders ISO with +00:00 offset
  - funding_rate values preserved on latest row
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
    rows = cur.fetchall()
    return [(r[1], r[2], r[5]) for r in rows]


def detect_schema(conn):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='funding_rates'"
    )
    if cur.fetchone() is None:
        return "missing"

    cols = get_columns(conn, "funding_rates")
    col_names = {c[0] for c in cols}
    pk_cols = sorted([c[0] for c in cols if c[2] > 0])

    if "id" in col_names and pk_cols == ["id"]:
        return "old"
    if "id" not in col_names and pk_cols == ["asset", "timestamp"]:
        return "new"
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
            print("[migrate] Already migrated -- funding_rates has new schema. Exiting cleanly.")
            return 0
        if schema == "unknown":
            cols = get_columns(conn, "funding_rates")
            print(f"[migrate] ERROR: funding_rates has unexpected schema: {cols}", file=sys.stderr)
            return 3

        assert schema == "old"
        print("[migrate] Detected OLD schema. Proceeding.")

        old_count = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
        old_min = conn.execute("SELECT MIN(timestamp) FROM funding_rates").fetchone()[0]
        old_max = conn.execute("SELECT MAX(timestamp) FROM funding_rates").fetchone()[0]
        old_per_asset = conn.execute(
            "SELECT asset, COUNT(*) FROM funding_rates GROUP BY asset ORDER BY asset"
        ).fetchall()
        old_latest = fetch_latest(conn, "funding_rates")
        old_latest_utc = datetime.fromtimestamp(old_latest[1], tz=timezone.utc)
        print(f"[migrate] Pre-migration: rows={old_count}, ts_min={old_min}, ts_max={old_max}")
        print(f"[migrate] Pre-migration per-asset: {old_per_asset}")
        print(f"[migrate] Pre-migration latest row: {old_latest}")
        print(f"[migrate] Pre-migration latest UTC: {old_latest_utc.isoformat()}")
        print(f"[migrate] Pre-migration latest datetime text: {old_latest[2]!r}")

        print("[migrate] Creating funding_rates_new and copying rows (timestamp *= 1000, datetime ISO+00:00)...")
        conn.execute("BEGIN")
        try:
            conn.execute("DROP TABLE IF EXISTS funding_rates_new")
            conn.execute(
                """
                CREATE TABLE funding_rates_new (
                    asset TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    funding_rate REAL,
                    PRIMARY KEY (asset, timestamp)
                )
                """
            )
            # Defense in depth: datetime is re-derived from `timestamp` (seconds)
            # via SQLite strftime, not copied from the old (naive) text.
            conn.execute(
                """
                INSERT INTO funding_rates_new
                    (asset, timestamp, datetime, funding_rate)
                SELECT asset,
                       timestamp * 1000,
                       strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch'),
                       funding_rate
                FROM funding_rates
                """
            )

            new_count = conn.execute("SELECT COUNT(*) FROM funding_rates_new").fetchone()[0]
            new_per_asset = conn.execute(
                "SELECT asset, COUNT(*) FROM funding_rates_new GROUP BY asset ORDER BY asset"
            ).fetchall()
            print(f"[migrate] Post-copy rows in funding_rates_new: {new_count}")
            print(f"[migrate] Post-copy per-asset: {new_per_asset}")
            if new_count != old_count:
                raise RuntimeError(f"Row count mismatch: old={old_count}, new={new_count}")
            if new_per_asset != old_per_asset:
                raise RuntimeError(
                    f"Per-asset count mismatch: old={old_per_asset}, new={new_per_asset}"
                )

            new_latest = fetch_latest(conn, "funding_rates_new")
            print(f"[migrate] Post-copy latest row: {new_latest}")
            new_latest_ms = new_latest[1]
            new_latest_utc = datetime.fromtimestamp(new_latest_ms / 1000, tz=timezone.utc)
            print(f"[migrate] Post-copy latest UTC: {new_latest_utc.isoformat()}")
            print(f"[migrate] Post-copy latest datetime text: {new_latest[2]!r}")

            delta_s = abs(new_latest_ms / 1000 - old_latest[1])
            if delta_s > 1.0:
                raise RuntimeError(
                    f"Latest UTC moment drift > 1s: old={old_latest[1]}, new_ms={new_latest_ms}"
                )

            # Datetime cross-check: derive expected ISO and confirm match
            expected_dt = new_latest_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            if expected_dt != new_latest[2]:
                raise RuntimeError(
                    f"Datetime ISO mismatch: derived={expected_dt!r}, stored={new_latest[2]!r}"
                )
            if "+00:00" not in new_latest[2]:
                raise RuntimeError(
                    f"Datetime missing +00:00 offset: {new_latest[2]!r}"
                )

            # funding_rate value cross-check on latest row
            if old_latest[3] != new_latest[3]:
                raise RuntimeError(
                    f"funding_rate mismatch on latest row: "
                    f"old={old_latest[3]}, new={new_latest[3]}"
                )

            # timestamp arithmetic cross-check on latest row
            if new_latest_ms != old_latest[1] * 1000:
                raise RuntimeError(
                    f"Latest timestamp not exactly old*1000: "
                    f"old_s={old_latest[1]}, new_ms={new_latest_ms}"
                )

            print("[migrate] Dropping old funding_rates and renaming funding_rates_new...")
            conn.execute("DROP TABLE funding_rates")
            conn.execute("ALTER TABLE funding_rates_new RENAME TO funding_rates")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        post_count = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
        post_latest = fetch_latest(conn, "funding_rates")
        post_latest_utc = datetime.fromtimestamp(post_latest[1] / 1000, tz=timezone.utc)
        post_schema = detect_schema(conn)
        print()
        print("=" * 60)
        print("[migrate] MIGRATION COMPLETE")
        print(f"  schema after migration: {post_schema}")
        print(f"  rows: {old_count} -> {post_count}")
        print(f"  latest pre:  asset={old_latest[0]} ts={old_latest[1]} (s)  UTC={old_latest_utc.isoformat()}  dt={old_latest[2]!r}")
        print(f"  latest post: asset={post_latest[0]} ts={post_latest[1]} (ms) UTC={post_latest_utc.isoformat()}  dt={post_latest[2]!r}")
        print(f"  delta from pre: {abs(post_latest[1] / 1000 - old_latest[1])} seconds")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
