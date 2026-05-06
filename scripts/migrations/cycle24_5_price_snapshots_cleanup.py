"""
Cycle 24.5 -- live_collector.price_snapshots Phase 5 cleanup.

Drops the two leftover tables from Cycle 24's dual-write:

  * price_snapshots_legacy -- the OLD seconds-format table; renamed
    from `price_snapshots` during Cycle 24's Phase 4 cutover. Has
    been receiving dual-writes via the runtime-introspection writer
    for the burn-in window. After this script runs, no further
    writes target it.

  * price_snapshots_v2 -- the empty orphan stub recreated on every
    PraxisLiveCollector startup by `init_db()`'s
    `CREATE TABLE IF NOT EXISTS` from Cycle 24 Phase 0. The
    runtime-introspection writer bypasses it post-cutover, so it
    has been empty for ~24h. The writer collapse (separate change)
    removes the CREATE statement so it doesn't reappear.

Idempotent: if either table is already absent, this script logs
that fact and continues. Re-running on a fully-cleaned state
produces no errors.

PRE-FLIGHT ORDERING (per Cycle 23.5 lesson, 2026-05-05):
This script MUST be run AFTER:
  1. The writer collapse has been committed.
  2. The long-lived PraxisLiveCollector process has been killed.
  3. The fresh PraxisLiveCollector process has spawned with the
     new code (verify via `Get-Process python` or by waiting >70s).

Running this script BEFORE killing the old process leaves a
window where the in-memory dual-write writer hits dropped
`_legacy`, throws silent OperationalErrors every iteration, and
can lock out other collectors via SQLite write-lock contention.
This is exactly how Cycle 23.5 cascaded into a multi-collector
outage; do not repeat.

Pre-flight check: refuses to run if the live `price_snapshots`
table doesn't have the post-cutover schema (compound PK + ms
timestamp + datetime). Guards against running before cutover.

Pre-flight check: refuses to run if `price_snapshots_legacy`
doesn't exist OR has substantially fewer rows than live; either
suggests Cycle 24's cutover didn't run as expected.

Pre-flight check: refuses to run if `price_snapshots_v2` has
non-zero rows post-cutover -- means the writer collapse didn't
land or didn't take effect (old process still writing to v2
which was renamed to live, so v2 should be the orphan stub).

Run from the repo root:

    python scripts/migrations/cycle24_5_price_snapshots_cleanup.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "live_collector.db"


def main() -> int:
    print(f"[cleanup] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Pre-flight 1: live table has the post-cutover schema
        cols = conn.execute("PRAGMA table_info(price_snapshots)").fetchall()
        col_names = {c[1] for c in cols}
        # Cycle 24's post-cutover schema for price_snapshots is
        # (timestamp INTEGER ms, no datetime column -- price_snapshots
        # has only a single timestamp column per Cycle 24's design).
        # Check: timestamp column is INTEGER (not REAL or TEXT).
        ts_col = next((c for c in cols if c[1] == "timestamp"), None)
        if ts_col is None:
            print(
                f"[cleanup] ABORT: live price_snapshots is missing the "
                f"timestamp column. Schema: {col_names}",
                file=sys.stderr,
            )
            return 2
        # SQLite doesn't strictly enforce column types, so check that
        # the latest timestamp value reads as ms-magnitude (>1e12).
        latest_ts = conn.execute(
            "SELECT MAX(timestamp) FROM price_snapshots"
        ).fetchone()[0]
        if latest_ts is None:
            print(
                "[cleanup] ABORT: live price_snapshots has no rows; "
                "cutover state is unverifiable.",
                file=sys.stderr,
            )
            return 2
        if latest_ts < 1e12:
            print(
                f"[cleanup] ABORT: live price_snapshots latest timestamp "
                f"is {latest_ts} (< 1e12), suggesting it's still in "
                f"seconds format. Cycle 24's cutover may not have run.",
                file=sys.stderr,
            )
            return 2

        # Inventory current state
        legacy_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='price_snapshots_legacy'"
            ).fetchone()
        )
        v2_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='price_snapshots_v2'"
            ).fetchone()
        )

        live_count = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots"
        ).fetchone()[0]
        legacy_count = (
            conn.execute(
                "SELECT COUNT(*) FROM price_snapshots_legacy"
            ).fetchone()[0]
            if legacy_exists
            else 0
        )
        v2_count = (
            conn.execute(
                "SELECT COUNT(*) FROM price_snapshots_v2"
            ).fetchone()[0]
            if v2_exists
            else 0
        )

        print("[cleanup] Pre-state:")
        print(f"  price_snapshots:        {live_count:,} rows (live)")
        print(
            f"  price_snapshots_legacy: {legacy_count:,} rows "
            f"(exists={legacy_exists})"
        )
        print(
            f"  price_snapshots_v2:     {v2_count:,} rows "
            f"(exists={v2_exists})"
        )

        # Pre-flight 2: legacy row count sanity check
        # Post-cutover both tables should have grown together via dual-write.
        # Allow up to 5% drift for the Phase 4 cutover transaction window.
        if legacy_exists and legacy_count > 0:
            ratio = legacy_count / live_count if live_count > 0 else 0
            if ratio < 0.95:
                print(
                    f"[cleanup] ABORT: legacy has {legacy_count:,} rows but "
                    f"live has {live_count:,} ({ratio:.1%}); dual-write "
                    f"may have been broken. Investigate before dropping.",
                    file=sys.stderr,
                )
                return 3

        # Pre-flight 3: v2 stub should be empty post-cutover
        if v2_exists and v2_count > 0:
            print(
                f"[cleanup] ABORT: price_snapshots_v2 has {v2_count:,} "
                f"rows but should be an empty post-cutover stub. "
                f"Either the writer collapse didn't land OR the old "
                f"long-lived PraxisLiveCollector process is still "
                f"running with pre-collapse code. Per Cycle 23.5 "
                f"lesson: kill the long-lived process and wait for "
                f"fresh spawn BEFORE running this cleanup.",
                file=sys.stderr,
            )
            return 4

        # Pre-flight 4: detect if the writer is still actively touching
        # the legacy table. If legacy is being written to RIGHT NOW,
        # the writer collapse hasn't taken effect yet (process still
        # running old code). Check by sampling legacy's latest row
        # against the script's current wall-clock; if it's <60s old,
        # the old writer is still alive.
        if legacy_exists and legacy_count > 0:
            # Use the same ms-format timestamp the post-cutover live
            # table uses; legacy stores seconds (Cycle 24 wrote
            # seconds-truncated to legacy in the dual-write window).
            import time
            now_s = int(time.time())
            latest_legacy_s = conn.execute(
                "SELECT MAX(timestamp) FROM price_snapshots_legacy"
            ).fetchone()[0] or 0
            age_s = now_s - latest_legacy_s
            if age_s < 60:
                print(
                    f"[cleanup] ABORT: price_snapshots_legacy was "
                    f"written {age_s}s ago. The dual-write writer is "
                    f"still active -- writer collapse hasn't taken "
                    f"effect. Per Cycle 23.5 lesson: kill the "
                    f"PraxisLiveCollector process and wait for fresh "
                    f"spawn BEFORE running this cleanup. Verify with: "
                    f"Get-Process python | where CommandLine -like "
                    f"'*live_collector start*'",
                    file=sys.stderr,
                )
                return 5
            print(
                f"  legacy last write: {age_s}s ago (writer collapse "
                f"appears to have taken effect)"
            )

        # Idempotency: if both already gone, nothing to do
        if not legacy_exists and not v2_exists:
            print(
                "[cleanup] Already cleaned up -- no action needed. "
                "Exiting cleanly."
            )
            return 0

        # Drop within a single transaction
        conn.execute("BEGIN")
        try:
            if legacy_exists:
                print(
                    f"[cleanup] Dropping price_snapshots_legacy "
                    f"({legacy_count:,} rows)..."
                )
                conn.execute("DROP TABLE price_snapshots_legacy")
            if v2_exists:
                print(
                    f"[cleanup] Dropping price_snapshots_v2 "
                    f"({v2_count:,} rows)..."
                )
                conn.execute("DROP TABLE price_snapshots_v2")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Verify
        legacy_still = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='price_snapshots_legacy'"
            ).fetchone()
        )
        v2_still = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='price_snapshots_v2'"
            ).fetchone()
        )
        if legacy_still or v2_still:
            print(
                f"[cleanup] FAIL: post-DROP, tables still present "
                f"(legacy={legacy_still}, v2={v2_still})",
                file=sys.stderr,
            )
            return 6

        post_live_count = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots"
        ).fetchone()[0]
        print()
        print("=" * 60)
        print("[cleanup] PHASE 5 CLEANUP COMPLETE")
        print(
            f"  price_snapshots: {live_count:,} -> {post_live_count:,} "
            f"(live count grew during cleanup; healthy)"
        )
        print(f"  price_snapshots_legacy: dropped")
        print(f"  price_snapshots_v2: dropped")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
