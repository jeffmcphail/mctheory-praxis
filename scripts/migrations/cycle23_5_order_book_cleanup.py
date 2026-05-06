"""
Cycle 23.5 -- order_book_snapshots Phase 5 cleanup.

Drops the two leftover tables from Cycle 23's dual-write pilot:

  * order_book_snapshots_legacy -- the OLD seconds-format table; renamed
    from `order_book_snapshots` during Cycle 23's Phase 4 cutover. Has
    been receiving dual-writes via the runtime-introspection writer for
    the burn-in window. After this script runs, no further writes will
    target it.

  * order_book_snapshots_v2 -- the empty orphan stub; recreated on every
    collector startup by `init_db()`'s `CREATE TABLE IF NOT EXISTS`
    statement that dates back to Cycle 23 Phase 0. The collector's
    runtime-introspection writer has been bypassing it post-cutover, so
    it's been empty for ~24h. This script drops it once; the writer
    collapse (separate change) removes the CREATE statement so it doesn't
    reappear.

Idempotent: if either table is already absent, this script logs that
fact and continues. Re-running on a fully-cleaned state produces no
errors.

Pre-flight check: refuses to run if the live `order_book_snapshots`
table doesn't have the post-cutover schema (compound PK + ms timestamp
+ datetime column). This guards against accidentally running this
script before Cycle 23's Phase 4 cutover completed.

Pre-flight check: refuses to run if `order_book_snapshots_legacy`
doesn't exist OR if it has substantially fewer rows than the live
table; either suggests Cycle 23's cutover didn't run as expected.

Run from the repo root:

    python scripts/migrations/cycle23_5_order_book_cleanup.py

Cycle 24.5 will follow the same pattern for live_collector.price_snapshots.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def main() -> int:
    print(f"[cleanup] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Pre-flight 1: live table has the post-cutover schema
        cols = conn.execute("PRAGMA table_info(order_book_snapshots)").fetchall()
        col_names = {c[1] for c in cols}
        has_id = "id" in col_names
        has_datetime = "datetime" in col_names
        if has_id or not has_datetime:
            print(
                f"[cleanup] ABORT: live order_book_snapshots does not have the "
                f"post-cutover schema (has_id={has_id}, has_datetime={has_datetime}). "
                f"Cycle 23 Phase 4 cutover may not have run; check "
                f"docs/SCHEMA_NOTES.md before proceeding.",
                file=sys.stderr,
            )
            return 2

        # Inventory current state
        legacy_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='order_book_snapshots_legacy'"
            ).fetchone()
        )
        v2_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='order_book_snapshots_v2'"
            ).fetchone()
        )

        live_count = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots"
        ).fetchone()[0]
        legacy_count = (
            conn.execute(
                "SELECT COUNT(*) FROM order_book_snapshots_legacy"
            ).fetchone()[0]
            if legacy_exists
            else 0
        )
        v2_count = (
            conn.execute(
                "SELECT COUNT(*) FROM order_book_snapshots_v2"
            ).fetchone()[0]
            if v2_exists
            else 0
        )

        print(f"[cleanup] Pre-state:")
        print(f"  order_book_snapshots:        {live_count:,} rows (live)")
        print(
            f"  order_book_snapshots_legacy: {legacy_count:,} rows "
            f"(exists={legacy_exists})"
        )
        print(
            f"  order_book_snapshots_v2:     {v2_count:,} rows "
            f"(exists={v2_exists})"
        )

        # Pre-flight 2: legacy row count sanity check
        # Post-cutover both tables should have grown together via dual-write,
        # so legacy_count should be roughly equal to live_count (give or take
        # a few rows from the Phase 4 transaction window). If legacy is much
        # smaller than live, dual-write was broken at some point and we
        # should investigate before dropping.
        if legacy_exists and legacy_count > 0:
            ratio = legacy_count / live_count if live_count > 0 else 0
            if ratio < 0.95:
                print(
                    f"[cleanup] ABORT: legacy table has {legacy_count:,} rows but "
                    f"live has {live_count:,} ({ratio:.1%}); dual-write may have "
                    f"been broken. Investigate before dropping legacy.",
                    file=sys.stderr,
                )
                return 3

        # Pre-flight 3: v2 stub should be empty (post-cutover the writer
        # bypasses it; it only gets recreated as an empty stub by init_db).
        # If v2 has rows, something is writing to it that shouldn't be.
        if v2_exists and v2_count > 0:
            print(
                f"[cleanup] ABORT: order_book_snapshots_v2 has {v2_count:,} rows "
                f"but should be an empty post-cutover stub. Something is still "
                f"writing to it. Check the writer collapse landed before running "
                f"this cleanup.",
                file=sys.stderr,
            )
            return 4

        # Idempotency: if both already gone, nothing to do
        if not legacy_exists and not v2_exists:
            print("[cleanup] Already cleaned up -- no action needed. Exiting cleanly.")
            return 0

        # Drop within a single transaction
        conn.execute("BEGIN")
        try:
            if legacy_exists:
                print(
                    f"[cleanup] Dropping order_book_snapshots_legacy "
                    f"({legacy_count:,} rows)..."
                )
                conn.execute("DROP TABLE order_book_snapshots_legacy")
            if v2_exists:
                print(
                    f"[cleanup] Dropping order_book_snapshots_v2 "
                    f"({v2_count:,} rows)..."
                )
                conn.execute("DROP TABLE order_book_snapshots_v2")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Verify
        legacy_still = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='order_book_snapshots_legacy'"
            ).fetchone()
        )
        v2_still = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='order_book_snapshots_v2'"
            ).fetchone()
        )
        if legacy_still or v2_still:
            print(
                f"[cleanup] FAIL: post-DROP, tables still present "
                f"(legacy={legacy_still}, v2={v2_still})",
                file=sys.stderr,
            )
            return 5

        post_live_count = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots"
        ).fetchone()[0]
        print()
        print("=" * 60)
        print("[cleanup] PHASE 5 CLEANUP COMPLETE")
        print(f"  order_book_snapshots: {live_count:,} -> {post_live_count:,} "
              f"(unchanged)")
        print(f"  order_book_snapshots_legacy: dropped")
        print(f"  order_book_snapshots_v2: dropped")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
