"""
Cycle 20 -- ohlcv_4h table migration to Rule 35 standard.

Adapted from scripts/migrations/cycle18_ohlcv_daily_to_v2.py.

OLD schema:
    CREATE TABLE ohlcv_4h (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,        -- UTC seconds
        datetime TEXT NOT NULL,             -- 'YYYY-MM-DD HH:MM:SS' (naive)
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        UNIQUE(asset, timestamp)
    )

NEW schema (Rule 35 conforming):
    CREATE TABLE ohlcv_4h (
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,        -- UTC milliseconds
        datetime TEXT NOT NULL,             -- 'YYYY-MM-DDTHH:MM:SS+00:00' (ISO)
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        PRIMARY KEY (asset, timestamp)
    )

Two changes from cycle 18 (ohlcv_daily):
  (1) datetime column needs format rewrite -- naive 'space' separator ->
      ISO 'T' separator with explicit '+00:00' offset. New datetime is
      derived from `timestamp * 1000` rather than from the existing
      datetime text, so it's a defense-in-depth re-render.
  (2) Schema notes claimed datetime was already +00:00; verified empirically
      it is NOT (sample row reads '2026-05-02 04:00:00').

Idempotent: re-running on an already-migrated table prints "Already migrated"
and exits 0.

Acceptance:
  - Pre/post row counts match (~10,830 expected: 5,415 BTC + 5,415 ETH)
  - Latest UTC moment unchanged within 1-second precision
  - New datetime renders ISO with +00:00 offset
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
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_4h'"
    )
    if cur.fetchone() is None:
        return "missing"

    cols = get_columns(conn, "ohlcv_4h")
    col_names = {c[0] for c in cols}
    pk_cols = sorted([c[0] for c in cols if c[2] > 0])

    if "id" in col_names and pk_cols == ["id"]:
        return "old"
    if "id" not in col_names and pk_cols == ["asset", "timestamp"]:
        return "new"
    return "unknown"


def fetch_latest(conn, table):
    cur = conn.execute(
        f"SELECT asset, timestamp, datetime, open, high, low, close, volume "
        f"FROM {table} ORDER BY timestamp DESC, asset ASC LIMIT 1"
    )
    return cur.fetchone()


def main():
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        schema = detect_schema(conn)
        if schema == "missing":
            print("[migrate] ERROR: ohlcv_4h table does not exist", file=sys.stderr)
            return 2
        if schema == "new":
            print("[migrate] Already migrated -- ohlcv_4h has new schema. Exiting cleanly.")
            return 0
        if schema == "unknown":
            cols = get_columns(conn, "ohlcv_4h")
            print(f"[migrate] ERROR: ohlcv_4h has unexpected schema: {cols}", file=sys.stderr)
            return 3

        assert schema == "old"
        print("[migrate] Detected OLD schema. Proceeding.")

        old_count = conn.execute("SELECT COUNT(*) FROM ohlcv_4h").fetchone()[0]
        old_min = conn.execute("SELECT MIN(timestamp) FROM ohlcv_4h").fetchone()[0]
        old_max = conn.execute("SELECT MAX(timestamp) FROM ohlcv_4h").fetchone()[0]
        old_per_asset = conn.execute(
            "SELECT asset, COUNT(*) FROM ohlcv_4h GROUP BY asset ORDER BY asset"
        ).fetchall()
        old_latest = fetch_latest(conn, "ohlcv_4h")
        old_latest_utc = datetime.fromtimestamp(old_latest[1], tz=timezone.utc)
        print(f"[migrate] Pre-migration: rows={old_count}, ts_min={old_min}, ts_max={old_max}")
        print(f"[migrate] Pre-migration per-asset: {old_per_asset}")
        print(f"[migrate] Pre-migration latest row: {old_latest}")
        print(f"[migrate] Pre-migration latest UTC: {old_latest_utc.isoformat()}")
        print(f"[migrate] Pre-migration latest datetime text: {old_latest[2]!r}")

        print("[migrate] Creating ohlcv_4h_new and copying rows (timestamp *= 1000, datetime ISO+00:00)...")
        conn.execute("BEGIN")
        try:
            conn.execute("DROP TABLE IF EXISTS ohlcv_4h_new")
            conn.execute(
                """
                CREATE TABLE ohlcv_4h_new (
                    asset TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL,
                    PRIMARY KEY (asset, timestamp)
                )
                """
            )
            # Defense in depth: datetime is re-derived from `timestamp * 1000`
            # via SQLite strftime, not copied from the old (naive) text.
            conn.execute(
                """
                INSERT INTO ohlcv_4h_new
                    (asset, timestamp, datetime, open, high, low, close, volume)
                SELECT asset,
                       timestamp * 1000,
                       strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch'),
                       open, high, low, close, volume
                FROM ohlcv_4h
                """
            )

            new_count = conn.execute("SELECT COUNT(*) FROM ohlcv_4h_new").fetchone()[0]
            new_per_asset = conn.execute(
                "SELECT asset, COUNT(*) FROM ohlcv_4h_new GROUP BY asset ORDER BY asset"
            ).fetchall()
            print(f"[migrate] Post-copy rows in ohlcv_4h_new: {new_count}")
            print(f"[migrate] Post-copy per-asset: {new_per_asset}")
            if new_count != old_count:
                raise RuntimeError(f"Row count mismatch: old={old_count}, new={new_count}")
            if new_per_asset != old_per_asset:
                raise RuntimeError(
                    f"Per-asset count mismatch: old={old_per_asset}, new={new_per_asset}"
                )

            new_latest = fetch_latest(conn, "ohlcv_4h_new")
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

            # OHLCV value cross-check on latest row
            for i, label in enumerate(["open", "high", "low", "close", "volume"], start=3):
                if old_latest[i] != new_latest[i]:
                    raise RuntimeError(
                        f"{label} mismatch on latest row: "
                        f"old={old_latest[i]}, new={new_latest[i]}"
                    )

            print("[migrate] Dropping old ohlcv_4h and renaming ohlcv_4h_new...")
            conn.execute("DROP TABLE ohlcv_4h")
            conn.execute("ALTER TABLE ohlcv_4h_new RENAME TO ohlcv_4h")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        post_count = conn.execute("SELECT COUNT(*) FROM ohlcv_4h").fetchone()[0]
        post_latest = fetch_latest(conn, "ohlcv_4h")
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
