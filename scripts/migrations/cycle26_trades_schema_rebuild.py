"""
Cycle 26 -- trades schema rebuild (one-shot, NOT dual-write).

Removes the synthetic `id INTEGER PRIMARY KEY AUTOINCREMENT` column
from `trades` and promotes the existing `UNIQUE(asset, trade_id)`
constraint to be the compound PRIMARY KEY. No data transformation;
this is a structure-only change.

Why one-shot instead of dual-write (which Cycles 23-25 used):

1. The trades table is ALREADY Rule 35 compliant for column types:
   `timestamp INTEGER NOT NULL` (ms), `datetime TEXT NOT NULL`. The
   ONLY non-conforming aspect is the synthetic `id` PK. No data
   transformation is needed -- rows copy 1:1 minus the id.

2. The writer (collect_recent_trades) doesn't specify `id` in its
   INSERT. Removing the id column requires NO writer change (only
   the init_db() CREATE TABLE statement needs updating).

3. PraxisTradesCollector is a scheduled task (every 60s, NOT long-
   lived). We can briefly disable it, rebuild atomically, and
   re-enable. No dual-write window is needed because there is no
   data semantic gap between old and new shapes.

This script is the "Phase 4" equivalent for a structure-only
change: one atomic transaction that creates the new table,
copies all rows, drops old, renames new. The CREATE TABLE in
init_db() (separate writer-collapse commit) updates the
on-disk schema definition to match.

PRE-CONDITION (required before running):

1. PraxisTradesCollector must be DISABLED (Disable-ScheduledTask)
   so no writes happen during the rebuild. The script does NOT
   automate this -- the user runs it manually before invoking
   this script and re-enables after verification.

2. The trades writer's init_db() CREATE TABLE must have been
   updated separately and committed (so a fresh process picks up
   the new schema definition). The init_db() is idempotent: if it
   sees a table named `trades` already exists, it doesn't try to
   re-create it. So the on-disk init_db() change is purely for
   future fresh-DB initializations -- existing DBs go through this
   migration script.

PERFORMANCE EXPECTATION (8.7M rows):

- INSERT INTO trades_v2 SELECT FROM trades: ~30-60s (SQLite pure-
  SQL bulk insert is fast, but constructing PK + index entries is
  the bottleneck).
- DROP trades + ALTER RENAME: instant.
- Re-CREATE INDEX statements on trades_v2: applies BEFORE the
  rename, in the same transaction, so they cover the new table
  immediately.

Rollback: if anything in the transaction fails, BEGIN/ROLLBACK
restores everything to pre-script state. The script is idempotent:
re-running on a fully-rebuilt state detects the already-converted
schema and exits cleanly.

Run from the repo root (with PraxisTradesCollector disabled):

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
        # Sample latest trade timestamp; if it's <60s old, the writer
        # is still active. The user is supposed to have disabled
        # PraxisTradesCollector before running this; this guard
        # catches forgotten disables.
        latest_ts = conn.execute(
            "SELECT MAX(timestamp) FROM trades"
        ).fetchone()[0]
        if latest_ts is not None:
            now_ms = int(time.time() * 1000)
            age_s = (now_ms - latest_ts) / 1000
            if age_s < 60:
                print(
                    f"[rebuild] ABORT: latest trade was {age_s:.0f}s ago. "
                    f"PraxisTradesCollector appears to still be active. "
                    f"Disable it (Disable-ScheduledTask -TaskName "
                    f"'PraxisTradesCollector') before running this "
                    f"script. The script needs an exclusive write window "
                    f"to rebuild the table atomically.",
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
        # The rebuild transaction
        # ---------------------------------------------------------
        print()
        print("[rebuild] Starting atomic rebuild transaction...")
        t_start = time.time()

        conn.execute("BEGIN")
        try:
            # 1. Create new table with the post-Cycle-26 schema:
            #    no `id`, compound PK on (asset, trade_id).
            print("  [1/4] Creating trades_v2 with new schema...")
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
            print(f"  [2/4] Copying {live_count:,} rows from trades to "
                  f"trades_v2 (this is the slow step; ~30-60s)...")
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

            # 3. Add the same indexes that exist on the old table.
            #    The compound PK (asset, trade_id) covers the
            #    idx_trades_asset_tradeid index (asset DESC ordering
            #    is irrelevant for PK presence). Keep idx_trades_
            #    asset_timestamp explicitly since it's a different
            #    column ordering.
            print("  [3/4] Adding idx_trades_asset_timestamp on "
                  "trades_v2...")
            conn.execute("""
                CREATE INDEX idx_trades_asset_timestamp
                    ON trades_v2(asset, timestamp DESC)
            """)

            # 4. Drop old + rename new. ALTER RENAME is atomic in
            #    SQLite within a transaction.
            print("  [4/4] Dropping old trades and renaming "
                  "trades_v2 -> trades...")
            conn.execute("DROP TABLE trades")
            conn.execute("ALTER TABLE trades_v2 RENAME TO trades")

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

        print()
        print("=" * 60)
        print("[rebuild] CYCLE 26 SCHEMA REBUILD COMPLETE")
        print(f"  trades: {live_count:,} -> {post_count:,} rows "
              f"(unchanged)")
        print(f"  schema: id PK -> compound (asset, trade_id) PK")
        print(f"  indexes: idx_trades_asset_timestamp restored")
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
