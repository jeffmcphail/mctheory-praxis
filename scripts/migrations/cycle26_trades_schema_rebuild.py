"""
Cycle 26 -- trades schema rebuild (one-shot, NOT dual-write).

v2 (2026-05-06): fixes the CREATE INDEX namespace collision that
caused v1 to abort mid-transaction. SQLite indexes are namespaced
per-database, not per-table -- v1 tried to CREATE INDEX
idx_trades_asset_timestamp on trades_v2 while the old trades
table's index of the same name still existed, causing
sqlite3.OperationalError. v2 reorders: DROP old trades (which
drops its associated index too), ALTER RENAME _v2 -> trades, THEN
CREATE INDEX on the renamed table.

v1 transaction structure (broken):
  CREATE trades_v2 -> INSERT _v2 SELECT FROM trades ->
  CREATE INDEX on _v2 [FAILS: name collision] -> ROLLBACK

v2 transaction structure:
  CREATE trades_v2 -> INSERT _v2 SELECT FROM trades ->
  DROP trades [also drops its index] ->
  ALTER trades_v2 RENAME TO trades ->
  CREATE INDEX idx_trades_asset_timestamp ON trades

Both versions wrap everything in BEGIN/COMMIT so any failure
rolls back atomically. v1's rollback was confirmed to work
(post-failure schema was unchanged from pre-script).

Removes the synthetic `id INTEGER PRIMARY KEY AUTOINCREMENT` column
from `trades` and promotes the existing `UNIQUE(asset, trade_id)`
constraint to be the compound PRIMARY KEY. No data transformation;
this is a structure-only change.

PRE-CONDITION (required before running):

1. PraxisTradesCollector must be DISABLED (Disable-ScheduledTask)
   AND the long-lived collect-trades-loop processes must be killed
   (Stop-Process). Disabling the scheduled task alone is NOT
   sufficient -- the long-lived loop processes survive the disable
   until their --duration expires. Verify both via:
       Get-Process python | where CommandLine -like "*collect-trades*"
   should return zero results.

2. The trades writer's init_db() CREATE TABLE has been updated
   separately and committed in step 1 (a1c1638).

PERFORMANCE EXPECTATION (8.8M rows):

- v1 measurement: 19.8s for the bulk INSERT, ~447k rows/sec
- v2 has the same INSERT step; total wall-clock should be under
  25s for the full transaction.

Rollback: BEGIN/ROLLBACK in either version. Idempotent: detects
already-rebuilt state via "no `id` column" and exits cleanly. Also
detects a v1-style mid-transaction rollback leftover (no _v2 in
state because the rollback successfully cleaned up).

Run from the repo root (with PraxisTradesCollector disabled AND
loop processes killed):

    python scripts/migrations/cycle26_trades_schema_rebuild.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def main() -> int:
    print(f"[rebuild] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # ---------------------------------------------------------
        # Pre-flight 1: live trades schema
        # ---------------------------------------------------------
        cols = conn.execute(
            "PRAGMA table_info(trades)"
        ).fetchall()
        if not cols:
            print(
                "[rebuild] ABORT: trades table doesn't exist.",
                file=sys.stderr,
            )
            return 2

        col_names = [c[1] for c in cols]
        has_id = "id" in col_names
        if not has_id:
            print(
                "[rebuild] Already rebuilt -- trades has no `id` column. "
                "Exiting cleanly.",
            )
            return 0

        # Verify all expected columns are present
        expected_cols = {
            "id", "asset", "trade_id", "timestamp", "datetime",
            "price", "amount", "quote_amount", "is_buyer_maker", "side",
        }
        actual_cols = set(col_names)
        if not expected_cols.issubset(actual_cols):
            missing = expected_cols - actual_cols
            print(
                f"[rebuild] ABORT: trades is missing expected columns: "
                f"{missing}. Schema may have drifted since this script "
                f"was written; investigate before proceeding.",
                file=sys.stderr,
            )
            return 2

        # ---------------------------------------------------------
        # Pre-flight 2: no _v2 leftover from a previous attempt
        # ---------------------------------------------------------
        v2_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='trades_v2'"
            ).fetchone()
        )
        if v2_exists:
            print(
                "[rebuild] ABORT: trades_v2 already exists, suggesting "
                "a previous rebuild attempt was interrupted. "
                "Investigate before proceeding (manually drop "
                "trades_v2 if confirmed safe).",
                file=sys.stderr,
            )
            return 3

        # ---------------------------------------------------------
        # Pre-flight 3: writer is not actively writing
        # ---------------------------------------------------------
        latest_ts = conn.execute(
            "SELECT MAX(timestamp) FROM trades"
        ).fetchone()[0]
        if latest_ts is not None:
            now_ms = int(time.time() * 1000)
            age_s = (now_ms - latest_ts) / 1000
            if age_s < 60:
                print(
                    f"[rebuild] ABORT: latest trade was {age_s:.0f}s ago. "
                    f"PraxisTradesCollector or a collect-trades-loop "
                    f"process appears to still be active. Disable the "
                    f"scheduled task AND kill all long-lived loop "
                    f"processes (Stop-Process), then verify via:\n"
                    f"  Get-Process python | where CommandLine -like "
                    f"\"*collect-trades*\"\n"
                    f"returns zero results before re-running.",
                    file=sys.stderr,
                )
                return 4
            print(
                f"  trades last write: {age_s:.0f}s ago "
                f"(writer appears disabled, OK to proceed)"
            )

        # ---------------------------------------------------------
        # Pre-state inventory
        # ---------------------------------------------------------
        live_count = conn.execute(
            "SELECT COUNT(*) FROM trades"
        ).fetchone()[0]
        print(f"[rebuild] Pre-state:")
        print(f"  trades:    {live_count:,} rows")
        print(f"  has `id`:  yes (will be removed)")
        print(f"  natural key: (asset, trade_id) -- currently UNIQUE, "
              f"will become PRIMARY KEY")

        # ---------------------------------------------------------
        # The rebuild transaction (v2: index after rename)
        # ---------------------------------------------------------
        print()
        print("[rebuild] Starting atomic rebuild transaction (v2)...")
        t_start = time.time()

        conn.execute("BEGIN")
        try:
            # 1. Create new table with the post-Cycle-26 schema.
            #    NOTE: no index on _v2 yet -- index name would
            #    collide with the existing one on `trades`. We
            #    create the index AFTER the DROP+RENAME below.
            print("  [1/5] Creating trades_v2 with new schema...")
            conn.execute("""
                CREATE TABLE trades_v2 (
                    asset TEXT NOT NULL,
                    trade_id INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    quote_amount REAL NOT NULL,
                    is_buyer_maker INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    PRIMARY KEY (asset, trade_id)
                )
            """)

            # 2. Copy all rows (omitting `id`).
            print(f"  [2/5] Copying {live_count:,} rows from trades to "
                  f"trades_v2 (typically ~20s)...")
            t_copy_start = time.time()
            conn.execute("""
                INSERT INTO trades_v2 (
                    asset, trade_id, timestamp, datetime,
                    price, amount, quote_amount, is_buyer_maker, side
                )
                SELECT
                    asset, trade_id, timestamp, datetime,
                    price, amount, quote_amount, is_buyer_maker, side
                FROM trades
            """)
            t_copy_s = time.time() - t_copy_start
            v2_count = conn.execute(
                "SELECT COUNT(*) FROM trades_v2"
            ).fetchone()[0]
            print(f"        copied {v2_count:,} rows in {t_copy_s:.1f}s "
                  f"({v2_count / t_copy_s:.0f} rows/s)")

            if v2_count != live_count:
                raise RuntimeError(
                    f"Row count mismatch: trades has {live_count}, "
                    f"trades_v2 has {v2_count} -- aborting"
                )

            # 3. Drop old trades (drops its index automatically too).
            print("  [3/5] Dropping old trades (and its index)...")
            conn.execute("DROP TABLE trades")

            # 4. Rename _v2 to trades.
            print("  [4/5] Renaming trades_v2 -> trades...")
            conn.execute("ALTER TABLE trades_v2 RENAME TO trades")

            # 5. Create the index on the now-renamed table. The name
            #    is now free because the old table (with its index)
            #    is gone.
            print("  [5/5] Creating idx_trades_asset_timestamp on "
                  "trades...")
            conn.execute("""
                CREATE INDEX idx_trades_asset_timestamp
                    ON trades(asset, timestamp DESC)
            """)

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        t_total_s = time.time() - t_start
        print(f"[rebuild] Transaction committed in {t_total_s:.1f}s "
              f"total wall-clock.")

        # ---------------------------------------------------------
        # Post-state verification
        # ---------------------------------------------------------
        post_cols = conn.execute(
            "PRAGMA table_info(trades)"
        ).fetchall()
        post_col_names = [c[1] for c in post_cols]
        post_has_id = "id" in post_col_names

        if post_has_id:
            print(
                "[rebuild] FAIL: post-rebuild, trades still has `id` "
                "column. Something went wrong.",
                file=sys.stderr,
            )
            return 5

        # Confirm compound PK is in place
        pk_cols = sorted(c[1] for c in post_cols if c[5] > 0)
        if pk_cols != ["asset", "trade_id"]:
            print(
                f"[rebuild] FAIL: expected PK (asset, trade_id), got "
                f"{pk_cols}.",
                file=sys.stderr,
            )
            return 6

        post_count = conn.execute(
            "SELECT COUNT(*) FROM trades"
        ).fetchone()[0]
        if post_count != live_count:
            print(
                f"[rebuild] FAIL: post-rebuild row count "
                f"{post_count:,} != pre-rebuild {live_count:,}.",
                file=sys.stderr,
            )
            return 7

        # Verify the index landed
        idx_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='index' AND name='idx_trades_asset_timestamp'"
            ).fetchone()
        )
        if not idx_exists:
            print(
                "[rebuild] WARN: idx_trades_asset_timestamp index missing "
                "post-rebuild. Schema is OK but query performance for "
                "(asset, timestamp DESC) range queries will be poor "
                "until the index is recreated manually.",
                file=sys.stderr,
            )

        print()
        print("=" * 60)
        print("[rebuild] CYCLE 26 SCHEMA REBUILD COMPLETE (v2)")
        print(f"  trades: {live_count:,} -> {post_count:,} rows "
              f"(unchanged)")
        print(f"  schema: id PK -> compound (asset, trade_id) PK")
        print(f"  indexes: idx_trades_asset_timestamp created")
        print(f"  total time: {t_total_s:.1f}s")
        print("=" * 60)
        print()
        print("Next steps (USER):")
        print("  1. Re-enable PraxisTradesCollector:")
        print("     Enable-ScheduledTask -TaskName 'PraxisTradesCollector'")
        print("  2. Wait ~70s for the next fire to confirm writes work.")
        print("  3. Verify via get_collector_health that trades is")
        print("     reporting fresh data with is_stale=false.")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
