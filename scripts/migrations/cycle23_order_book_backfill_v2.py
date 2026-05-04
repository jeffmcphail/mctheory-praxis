"""
Cycle 23 Phase 2 -- backfill order_book_snapshots_v2 from legacy.

Pure-SQL INSERT ... SELECT. SQLite's julianday() correctly parses the
ISO+microsecond+offset datetime format used in order_book_snapshots
(verified via empirical test: julianday('2026-05-04T19:35:51.647000+00:00')
yields 1777923351647 ms, matching Python's int(dt.timestamp() * 1000)).

The first-cut Python row-by-row implementation hung under contention
with the live PraxisOrderBookCollector that writes every 10s; SQL-only
backfill avoids that by running as a single statement.

Idempotent: if no legacy rows are missing from v2 (NOT EXISTS subquery
returns zero), prints "Already backfilled" and exits 0.

ms derivation:
    CAST(ROUND((julianday(datetime) - 2440587.5) * 86400000) AS INTEGER)
where 2440587.5 is the julian day of the Unix epoch and 86400000 is
ms-per-day. ROUND() is required because SQLite's julianday returns a
double; for datetimes with .NNN-precision fractional seconds, the
product (julianday - epoch) * 86400000 lands ~1 ULP below the integer
about half the time, and CAST AS INTEGER truncates toward zero.
ROUND() rounds to nearest, matching Python's
int(datetime.fromisoformat(dt).timestamp() * 1000).
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def main():
    print(f"[backfill] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Sanity: both tables must exist
        for tbl in ("order_book_snapshots", "order_book_snapshots_v2"):
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (tbl,),
            )
            if cur.fetchone() is None:
                print(f"[backfill] ERROR: table {tbl} missing", file=sys.stderr)
                return 2

        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots"
        ).fetchone()[0]
        v2_count_pre = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots_v2"
        ).fetchone()[0]
        count_missing = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots l "
            "WHERE NOT EXISTS (SELECT 1 FROM order_book_snapshots_v2 v "
            "WHERE v.asset = l.asset AND v.datetime = l.datetime)"
        ).fetchone()[0]

        print(f"[backfill] Pre-state: legacy={legacy_count}, v2={v2_count_pre}, "
              f"missing_in_v2={count_missing}")

        if count_missing == 0:
            print("[backfill] Already backfilled -- every legacy "
                  "(asset, datetime) is present in v2. Exiting cleanly.")
            return 0

        # Pure-SQL INSERT-SELECT. This runs as a single SQLite statement,
        # holds the writer lock briefly, and avoids per-row Python overhead.
        # The dual-write writer's INSERT OR IGNORE protects us from races.
        backfill_sql = """
            INSERT OR IGNORE INTO order_book_snapshots_v2 (
                asset, timestamp, datetime, mid_price, best_bid, best_ask,
                spread, spread_bps,
                bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3,
                bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6,
                bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9,
                bid_price_10, bid_vol_10,
                ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3,
                ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6,
                ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9,
                ask_price_10, ask_vol_10,
                bid_volume_top10, ask_volume_top10, order_imbalance_top10
            )
            SELECT
                l.asset,
                CAST(ROUND((julianday(l.datetime) - 2440587.5) * 86400000) AS INTEGER) AS ts_ms,
                l.datetime, l.mid_price, l.best_bid, l.best_ask,
                l.spread, l.spread_bps,
                l.bid_price_1, l.bid_vol_1, l.bid_price_2, l.bid_vol_2, l.bid_price_3, l.bid_vol_3,
                l.bid_price_4, l.bid_vol_4, l.bid_price_5, l.bid_vol_5, l.bid_price_6, l.bid_vol_6,
                l.bid_price_7, l.bid_vol_7, l.bid_price_8, l.bid_vol_8, l.bid_price_9, l.bid_vol_9,
                l.bid_price_10, l.bid_vol_10,
                l.ask_price_1, l.ask_vol_1, l.ask_price_2, l.ask_vol_2, l.ask_price_3, l.ask_vol_3,
                l.ask_price_4, l.ask_vol_4, l.ask_price_5, l.ask_vol_5, l.ask_price_6, l.ask_vol_6,
                l.ask_price_7, l.ask_vol_7, l.ask_price_8, l.ask_vol_8, l.ask_price_9, l.ask_vol_9,
                l.ask_price_10, l.ask_vol_10,
                l.bid_volume_top10, l.ask_volume_top10, l.order_imbalance_top10
            FROM order_book_snapshots l
            WHERE NOT EXISTS (
                SELECT 1 FROM order_book_snapshots_v2 v
                WHERE v.asset = l.asset AND v.datetime = l.datetime
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
            "SELECT COUNT(*) FROM order_book_snapshots"
        ).fetchone()[0]
        v2_post = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots_v2"
        ).fetchone()[0]
        still_missing = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots l "
            "WHERE NOT EXISTS (SELECT 1 FROM order_book_snapshots_v2 v "
            "WHERE v.asset = l.asset AND v.datetime = l.datetime)"
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
