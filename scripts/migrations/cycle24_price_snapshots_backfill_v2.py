"""
Cycle 24 Phase 2 -- backfill price_snapshots_v2 from legacy.

Pure-SQL INSERT ... SELECT. Unlike Cycle 23 (which had to derive ms from
an ISO datetime via julianday/ROUND), this migration is a clean
multiply: legacy.timestamp is already integer seconds, so v2.timestamp =
legacy.timestamp * 1000 is exact. No off-by-1ms risk.

Pre-step (per Cycle 23 lesson): ensure (slug, timestamp) indexes exist
on BOTH tables before running the backfill. The NOT EXISTS subquery
is O(n^2) without them. Both indexes are also created by init_db()
in engines/live_collector.py, so this is defense-in-depth.

Idempotent: if no legacy rows are missing from v2 (NOT EXISTS subquery
returns zero), prints "Already backfilled" and exits 0.

`datetime` column: derived in pure SQL via
strftime('%Y-%m-%dT%H:%M:%S+00:00', l.timestamp, 'unixepoch'). This
yields the same wire format as the dual-write writer's Python-side
fromtimestamp().strftime() because both round to the second (legacy
data has no sub-second info to preserve).
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "live_collector.db"


def main():
    print(f"[backfill] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Sanity: both tables must exist
        for tbl in ("price_snapshots", "price_snapshots_v2"):
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (tbl,),
            )
            if cur.fetchone() is None:
                print(f"[backfill] ERROR: table {tbl} missing", file=sys.stderr)
                return 2

        # Defense-in-depth: ensure (slug, timestamp) indexes exist before
        # the NOT EXISTS subquery runs. Both should already be present
        # from init_db(), but CREATE INDEX IF NOT EXISTS is idempotent.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ps_slug_ts "
            "ON price_snapshots(slug, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_psv2_slug_ts "
            "ON price_snapshots_v2(slug, timestamp DESC)"
        )
        conn.commit()

        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots"
        ).fetchone()[0]
        v2_count_pre = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots_v2"
        ).fetchone()[0]
        count_missing = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots l "
            "WHERE NOT EXISTS (SELECT 1 FROM price_snapshots_v2 v "
            "WHERE v.slug = l.slug AND v.timestamp = l.timestamp * 1000)"
        ).fetchone()[0]

        print(f"[backfill] Pre-state: legacy={legacy_count}, v2={v2_count_pre}, "
              f"missing_in_v2={count_missing}")

        if count_missing == 0:
            print("[backfill] Already backfilled -- every legacy "
                  "(slug, timestamp) is present in v2 as "
                  "(slug, timestamp*1000). Exiting cleanly.")
            return 0

        # Pure-SQL INSERT-SELECT. Single statement, holds the writer
        # lock briefly. The dual-write writer's INSERT OR IGNORE
        # protects against the race window.
        backfill_sql = """
            INSERT OR IGNORE INTO price_snapshots_v2 (
                slug, timestamp, datetime, yes_mid, yes_bid, yes_ask, spread
            )
            SELECT
                l.slug,
                l.timestamp * 1000 AS ts_ms,
                strftime('%Y-%m-%dT%H:%M:%S+00:00', l.timestamp, 'unixepoch') AS dt,
                l.yes_mid, l.yes_bid, l.yes_ask, l.spread
            FROM price_snapshots l
            WHERE NOT EXISTS (
                SELECT 1 FROM price_snapshots_v2 v
                WHERE v.slug = l.slug AND v.timestamp = l.timestamp * 1000
            )
        """

        t0 = time.perf_counter()
        cur = conn.execute(backfill_sql)
        inserted = cur.rowcount
        conn.commit()
        t_elapsed = time.perf_counter() - t0
        print(f"[backfill] INSERT-SELECT wall-clock: {t_elapsed:.3f} s")
        print(f"[backfill] Inserted {inserted} rows")

        # Post-state verification
        legacy_post = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots"
        ).fetchone()[0]
        v2_post = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots_v2"
        ).fetchone()[0]
        still_missing = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots l "
            "WHERE NOT EXISTS (SELECT 1 FROM price_snapshots_v2 v "
            "WHERE v.slug = l.slug AND v.timestamp = l.timestamp * 1000)"
        ).fetchone()[0]

        print()
        print("=" * 60)
        print("[backfill] PHASE 2 COMPLETE")
        print(f"  legacy: {legacy_count} -> {legacy_post} (any growth from "
              f"live writer during backfill)")
        print(f"  v2:     {v2_count_pre} -> {v2_post}")
        print(f"  missing post-backfill: {still_missing} (must be 0)")
        print(f"  wall-clock: {t_elapsed:.3f} s")
        print("=" * 60)

        if still_missing != 0:
            print(f"[backfill] FAIL: {still_missing} rows still missing "
                  f"from v2 after backfill", file=sys.stderr)
            return 5
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
