"""
Cycle 22 -- ohlcv_1m table migration to Rule 35 standard.

Adapted from scripts/migrations/cycle20_ohlcv_4h_to_v2.py. Same recipe;
the only meaningful difference is row volume (~530k rows, ~250x larger
than any prior simple-pattern cycle), so this script also prints
wall-clock time for the INSERT-SELECT step as a performance datapoint.

OLD schema:
    CREATE TABLE ohlcv_1m (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,        -- UTC seconds
        datetime TEXT NOT NULL,             -- 'YYYY-MM-DD HH:MM:SS' (naive)
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        UNIQUE(asset, timestamp)
    )

NEW schema (Rule 35 conforming):
    CREATE TABLE ohlcv_1m (
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,        -- UTC milliseconds
        datetime TEXT NOT NULL,             -- 'YYYY-MM-DDTHH:MM:SS+00:00' (ISO)
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        PRIMARY KEY (asset, timestamp)
    )

Same recipe as cycles 18/20/21:
  (1) datetime is re-derived from `timestamp` via SQLite strftime
      (defense-in-depth, not a copy of the old naive text).
  (2) timestamp seconds -> milliseconds via `* 1000`.
  (3) Drop `id` AUTOINCREMENT, compound PK on `(asset, timestamp)`.

Idempotent: re-running on an already-migrated table prints "Already migrated"
and exits 0.

Acceptance:
  - Pre/post row counts match (~530k expected, exact value preserved)
  - Pre/post per-asset counts match exactly
  - Latest UTC moment unchanged within 1-second precision
  - New datetime renders ISO with +00:00 offset
  - OHLCV values byte-identical on latest row
  - INSERT-SELECT wall-clock time recorded
"""

from __future__ import annotations

import sqlite3
import sys
import time
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
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_1m'"
    )
    if cur.fetchone() is None:
        return "missing"

    cols = get_columns(conn, "ohlcv_1m")
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


def fetch_oldest(conn, table, asset):
    cur = conn.execute(
        f"SELECT asset, timestamp, datetime, open, high, low, close, volume "
        f"FROM {table} WHERE asset = ? ORDER BY timestamp ASC LIMIT 1",
        (asset,),
    )
    return cur.fetchone()


def main():
    print(f"[migrate] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        schema = detect_schema(conn)
        if schema == "missing":
            print("[migrate] ERROR: ohlcv_1m table does not exist", file=sys.stderr)
            return 2
        if schema == "new":
            print("[migrate] Already migrated -- ohlcv_1m has new schema. Exiting cleanly.")
            return 0
        if schema == "unknown":
            cols = get_columns(conn, "ohlcv_1m")
            print(f"[migrate] ERROR: ohlcv_1m has unexpected schema: {cols}", file=sys.stderr)
            return 3

        assert schema == "old"
        print("[migrate] Detected OLD schema. Proceeding.")

        old_count = conn.execute("SELECT COUNT(*) FROM ohlcv_1m").fetchone()[0]
        old_min = conn.execute("SELECT MIN(timestamp) FROM ohlcv_1m").fetchone()[0]
        old_max = conn.execute("SELECT MAX(timestamp) FROM ohlcv_1m").fetchone()[0]
        old_per_asset = conn.execute(
            "SELECT asset, COUNT(*) FROM ohlcv_1m GROUP BY asset ORDER BY asset"
        ).fetchall()
        old_latest = fetch_latest(conn, "ohlcv_1m")
        old_latest_utc = datetime.fromtimestamp(old_latest[1], tz=timezone.utc)
        old_oldest_btc = fetch_oldest(conn, "ohlcv_1m", "BTC")
        old_oldest_eth = fetch_oldest(conn, "ohlcv_1m", "ETH")
        print(f"[migrate] Pre-migration: rows={old_count}, ts_min={old_min}, ts_max={old_max}")
        print(f"[migrate] Pre-migration per-asset: {old_per_asset}")
        print(f"[migrate] Pre-migration latest row: {old_latest}")
        print(f"[migrate] Pre-migration latest UTC: {old_latest_utc.isoformat()}")
        print(f"[migrate] Pre-migration latest datetime text: {old_latest[2]!r}")
        print(f"[migrate] Pre-migration oldest BTC: {old_oldest_btc}")
        print(f"[migrate] Pre-migration oldest ETH: {old_oldest_eth}")

        print("[migrate] Creating ohlcv_1m_new and copying rows (timestamp *= 1000, datetime ISO+00:00)...")
        t_total_start = time.perf_counter()
        conn.execute("BEGIN")
        try:
            conn.execute("DROP TABLE IF EXISTS ohlcv_1m_new")
            conn.execute(
                """
                CREATE TABLE ohlcv_1m_new (
                    asset TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL,
                    PRIMARY KEY (asset, timestamp)
                )
                """
            )
            # Defense in depth: datetime is re-derived from `timestamp` (seconds)
            # via SQLite strftime, not copied from the old (naive) text.
            t_insert_start = time.perf_counter()
            conn.execute(
                """
                INSERT INTO ohlcv_1m_new
                    (asset, timestamp, datetime, open, high, low, close, volume)
                SELECT asset,
                       timestamp * 1000,
                       strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch'),
                       open, high, low, close, volume
                FROM ohlcv_1m
                """
            )
            t_insert_elapsed = time.perf_counter() - t_insert_start
            print(f"[migrate] INSERT-SELECT wall-clock: {t_insert_elapsed:.3f} s")

            new_count = conn.execute("SELECT COUNT(*) FROM ohlcv_1m_new").fetchone()[0]
            new_per_asset = conn.execute(
                "SELECT asset, COUNT(*) FROM ohlcv_1m_new GROUP BY asset ORDER BY asset"
            ).fetchall()
            print(f"[migrate] Post-copy rows in ohlcv_1m_new: {new_count}")
            print(f"[migrate] Post-copy per-asset: {new_per_asset}")
            if new_count != old_count:
                raise RuntimeError(f"Row count mismatch: old={old_count}, new={new_count}")
            if new_per_asset != old_per_asset:
                raise RuntimeError(
                    f"Per-asset count mismatch: old={old_per_asset}, new={new_per_asset}"
                )

            new_latest = fetch_latest(conn, "ohlcv_1m_new")
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

            # OHLCV value cross-check on latest row (open=3, high=4, low=5, close=6, volume=7)
            for i, label in enumerate(["open", "high", "low", "close", "volume"], start=3):
                if old_latest[i] != new_latest[i]:
                    raise RuntimeError(
                        f"{label} mismatch on latest row: "
                        f"old={old_latest[i]}, new={new_latest[i]}"
                    )

            # Timestamp arithmetic cross-check on latest row
            if new_latest_ms != old_latest[1] * 1000:
                raise RuntimeError(
                    f"Latest timestamp not exactly old*1000: "
                    f"old_s={old_latest[1]}, new_ms={new_latest_ms}"
                )

            # Spot-check oldest BTC + ETH rows
            new_oldest_btc = fetch_oldest(conn, "ohlcv_1m_new", "BTC")
            new_oldest_eth = fetch_oldest(conn, "ohlcv_1m_new", "ETH")
            for label, old_row, new_row in [
                ("oldest BTC", old_oldest_btc, new_oldest_btc),
                ("oldest ETH", old_oldest_eth, new_oldest_eth),
            ]:
                if new_row[1] != old_row[1] * 1000:
                    raise RuntimeError(
                        f"{label} ts mismatch: old_s={old_row[1]}, new_ms={new_row[1]}"
                    )
                for i, fld in enumerate(["open", "high", "low", "close", "volume"], start=3):
                    if old_row[i] != new_row[i]:
                        raise RuntimeError(
                            f"{label} {fld} mismatch: old={old_row[i]}, new={new_row[i]}"
                        )
                if "+00:00" not in new_row[2]:
                    raise RuntimeError(
                        f"{label} datetime missing +00:00: {new_row[2]!r}"
                    )

            print("[migrate] Dropping old ohlcv_1m and renaming ohlcv_1m_new...")
            t_drop_start = time.perf_counter()
            conn.execute("DROP TABLE ohlcv_1m")
            conn.execute("ALTER TABLE ohlcv_1m_new RENAME TO ohlcv_1m")
            t_drop_elapsed = time.perf_counter() - t_drop_start
            print(f"[migrate] DROP+RENAME wall-clock: {t_drop_elapsed:.3f} s")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        t_total_elapsed = time.perf_counter() - t_total_start
        print(f"[migrate] Total transaction wall-clock: {t_total_elapsed:.3f} s")

        post_count = conn.execute("SELECT COUNT(*) FROM ohlcv_1m").fetchone()[0]
        post_latest = fetch_latest(conn, "ohlcv_1m")
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
        print(f"  INSERT-SELECT: {t_insert_elapsed:.3f} s")
        print(f"  DROP+RENAME:   {t_drop_elapsed:.3f} s")
        print(f"  total txn:     {t_total_elapsed:.3f} s")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
