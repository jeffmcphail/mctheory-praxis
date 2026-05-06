"""
Cycle 25.5 -- smart_money.position_snapshots Phase 5 cleanup.

Drops the two leftover tables from Cycle 25's dual-write:

  * position_snapshots_legacy -- the OLD TEXT-timestamp table with
    synthetic `id` PK; renamed from `position_snapshots` during
    Cycle 25's Phase 4 cutover. Has been receiving dual-writes via
    the runtime-introspection writer for the burn-in window. After
    this script runs, no further writes target it.

  * position_snapshots_v2 -- the empty orphan stub recreated on
    every PraxisSmartMoney scheduled invocation by `init_db()`'s
    `CREATE TABLE IF NOT EXISTS` from Cycle 25 Phase 0. The
    runtime-introspection writer bypasses it post-cutover, so it
    has been empty for ~30h+. The writer collapse (separate change)
    removes the CREATE statement so it doesn't reappear.

Idempotent: if either table is already absent, this script logs
that fact and continues. Re-running on a fully-cleaned state
produces no errors.

PROCESS PATTERN -- DIFFERENT from Cycles 23.5 and 24.5:

PraxisSmartMoney is a SCHEDULED task (every 6h) that runs
`python -m engines.smart_money snapshot` and exits. NOT a long-
lived process. This means the Cycle 23.5 lock-contention failure
mode does NOT apply: there is no in-memory writer holding open
SQLite handles to dropped tables. The next scheduled fire spawns
a fresh process that loads the post-collapse code automatically.

The Cycle 24.5 ordering trick (writer-collapse-FIRST, kill, wait,
cleanup) is therefore optional here. The natural ordering is:
1. Commit writer collapse.
2. Run cleanup script (any time before the next scheduled fire).
3. Next scheduled fire automatically uses new code.

The pre-flight #4 guard (legacy age check) is retained from
Cycle 24.5 as defense-in-depth -- if for any reason a process
were still actively writing to legacy, the script would refuse.

TIMING NOTE: PraxisSmartMoney fires every 6h at :24 of the hour.
Current burn-in window is far exceeded. Run this script at any
point between scheduled fires; it completes in well under a
second. The 60s legacy-age threshold in pre-flight #4 means the
script can be run as soon as ~70s after a scheduled fire ends.

Pre-flight check 1: refuses to run if the live `position_snapshots`
table doesn't have the post-cutover schema (compound PK, INTEGER
timestamp, datetime column).

Pre-flight check 2: refuses to run if `position_snapshots_legacy`
has substantially fewer rows than live; suggests Cycle 25's
cutover didn't run as expected.

Pre-flight check 3: refuses to run if `position_snapshots_v2`
has non-zero rows -- means the writer collapse didn't land.

Pre-flight check 4: refuses to run if legacy was written within
60s -- defense-in-depth against a still-active dual-writer.

Run from the repo root:

    python scripts/migrations/cycle25_5_position_snapshots_cleanup.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "smart_money.db"


def main() -> int:
    print(f"[cleanup] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Pre-flight 1: live table has the post-cutover schema
        cols = conn.execute(
            "PRAGMA table_info(position_snapshots)"
        ).fetchall()
        col_names = {c[1] for c in cols}
        has_id = "id" in col_names
        has_datetime = "datetime" in col_names
        has_timestamp = "timestamp" in col_names
        if has_id or not has_datetime or not has_timestamp:
            print(
                f"[cleanup] ABORT: live position_snapshots does not have "
                f"the post-cutover schema (has_id={has_id}, "
                f"has_datetime={has_datetime}, has_timestamp={has_timestamp}). "
                f"Cycle 25 Phase 4 cutover may not have run; check "
                f"docs/SCHEMA_NOTES.md before proceeding.",
                file=sys.stderr,
            )
            return 2

        # Verify timestamp is INTEGER ms (not TEXT ISO from pre-cutover)
        latest_ts = conn.execute(
            "SELECT MAX(timestamp) FROM position_snapshots"
        ).fetchone()[0]
        if latest_ts is None:
            print(
                "[cleanup] ABORT: live position_snapshots has no rows; "
                "cutover state is unverifiable.",
                file=sys.stderr,
            )
            return 2
        # Post-cutover timestamp is INTEGER ms (>1e12). Pre-cutover was
        # TEXT ISO (would fail integer comparison or be a string).
        try:
            ts_int = int(latest_ts)
            if ts_int < 1e12:
                print(
                    f"[cleanup] ABORT: live position_snapshots latest "
                    f"timestamp is {ts_int} (< 1e12), suggesting it's "
                    f"still in seconds format. Cycle 25's cutover may "
                    f"not have completed.",
                    file=sys.stderr,
                )
                return 2
        except (TypeError, ValueError):
            print(
                f"[cleanup] ABORT: live position_snapshots timestamp is "
                f"not numeric: {latest_ts!r}. Cutover may not have run.",
                file=sys.stderr,
            )
            return 2

        # Inventory current state
        legacy_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='position_snapshots_legacy'"
            ).fetchone()
        )
        v2_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='position_snapshots_v2'"
            ).fetchone()
        )

        live_count = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots"
        ).fetchone()[0]
        legacy_count = (
            conn.execute(
                "SELECT COUNT(*) FROM position_snapshots_legacy"
            ).fetchone()[0]
            if legacy_exists
            else 0
        )
        v2_count = (
            conn.execute(
                "SELECT COUNT(*) FROM position_snapshots_v2"
            ).fetchone()[0]
            if v2_exists
            else 0
        )

        print("[cleanup] Pre-state:")
        print(f"  position_snapshots:        {live_count:,} rows (live)")
        print(
            f"  position_snapshots_legacy: {legacy_count:,} rows "
            f"(exists={legacy_exists})"
        )
        print(
            f"  position_snapshots_v2:     {v2_count:,} rows "
            f"(exists={v2_exists})"
        )

        # Pre-flight 2: legacy row count sanity check
        # Position snapshots write ~50-100 rows per 6h cycle, all with
        # the same snapshot_id. Total row count grows in ~50-row steps.
        # Allow up to 5% drift for the Phase 4 cutover transaction
        # window plus any in-flight 6h-cycle interruptions.
        if legacy_exists and legacy_count > 0:
            ratio = legacy_count / live_count if live_count > 0 else 0
            if ratio < 0.95:
                print(
                    f"[cleanup] ABORT: legacy has {legacy_count:,} rows "
                    f"but live has {live_count:,} ({ratio:.1%}); "
                    f"dual-write may have been broken. Investigate "
                    f"before dropping.",
                    file=sys.stderr,
                )
                return 3

        # Pre-flight 3: v2 stub should be empty post-cutover
        if v2_exists and v2_count > 0:
            print(
                f"[cleanup] ABORT: position_snapshots_v2 has "
                f"{v2_count:,} rows but should be an empty post-cutover "
                f"stub. The writer collapse didn't land.",
                file=sys.stderr,
            )
            return 4

        # Pre-flight 4: legacy age guard (defense-in-depth from Cycle 24.5)
        # PraxisSmartMoney is scheduled, not long-lived, so a stale
        # in-memory writer shouldn't be possible -- but the guard is
        # cheap and catches edge cases (e.g. someone manually invoked
        # `engines.smart_money loop` and forgot to kill it).
        #
        # Legacy stores ISO TEXT timestamps (Cycle 25 dual-write writes
        # the same ISO string to legacy and the new ms timestamp +
        # datetime to v2). Parse it as ISO and compare to current time.
        if legacy_exists and legacy_count > 0:
            latest_legacy_iso = conn.execute(
                "SELECT MAX(timestamp) FROM position_snapshots_legacy"
            ).fetchone()[0]
            if latest_legacy_iso:
                try:
                    latest_legacy_dt = datetime.fromisoformat(
                        latest_legacy_iso
                    )
                    if latest_legacy_dt.tzinfo is None:
                        latest_legacy_dt = latest_legacy_dt.replace(
                            tzinfo=timezone.utc
                        )
                    now_dt = datetime.now(timezone.utc)
                    age_s = (now_dt - latest_legacy_dt).total_seconds()
                    if age_s < 60:
                        print(
                            f"[cleanup] ABORT: position_snapshots_legacy "
                            f"was written {age_s:.0f}s ago. A dual-write "
                            f"writer is still active. Verify no "
                            f"`engines.smart_money loop` process is "
                            f"running, or wait for the next 6h scheduled "
                            f"cycle to confirm new code is live.",
                            file=sys.stderr,
                        )
                        return 5
                    print(
                        f"  legacy last write: {age_s:.0f}s ago "
                        f"(writer collapse appears to have taken effect)"
                    )
                except (ValueError, TypeError) as e:
                    print(
                        f"[cleanup] WARN: could not parse legacy latest "
                        f"timestamp {latest_legacy_iso!r}: {e}. "
                        f"Skipping age check.",
                        file=sys.stderr,
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
                    f"[cleanup] Dropping position_snapshots_legacy "
                    f"({legacy_count:,} rows)..."
                )
                conn.execute("DROP TABLE position_snapshots_legacy")
            if v2_exists:
                print(
                    f"[cleanup] Dropping position_snapshots_v2 "
                    f"({v2_count:,} rows)..."
                )
                conn.execute("DROP TABLE position_snapshots_v2")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Verify
        legacy_still = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='position_snapshots_legacy'"
            ).fetchone()
        )
        v2_still = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='position_snapshots_v2'"
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
            "SELECT COUNT(*) FROM position_snapshots"
        ).fetchone()[0]
        print()
        print("=" * 60)
        print("[cleanup] PHASE 5 CLEANUP COMPLETE")
        print(
            f"  position_snapshots: {live_count:,} -> {post_live_count:,} "
            f"(unchanged; PraxisSmartMoney is scheduled, not long-lived, "
            f"so the live count won't grow until the next 6h fire)"
        )
        print(f"  position_snapshots_legacy: dropped")
        print(f"  position_snapshots_v2: dropped")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
