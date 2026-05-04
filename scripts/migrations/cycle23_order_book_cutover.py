"""
Cycle 23 Phase 4 -- atomic cutover of order_book_snapshots.

Single-transaction RENAME pair:

    BEGIN;
    ALTER TABLE order_book_snapshots RENAME TO order_book_snapshots_legacy;
    ALTER TABLE order_book_snapshots_v2 RENAME TO order_book_snapshots;
    COMMIT;

Both DDL statements complete or none do (SQLite single-transaction
DDL is genuinely atomic).

Pre-conditions checked before cutover:
  - Both `order_book_snapshots` AND `order_book_snapshots_v2` exist
  - Phase 3 verification was passed (this script does NOT re-verify;
    run cycle23_order_book_verify.py first)

Idempotent post-conditions (re-run on already-cut-over state):
  - If `order_book_snapshots_legacy` exists AND
    `order_book_snapshots_v2` does NOT exist AND
    `order_book_snapshots` exists with the new (no-id) schema,
    print "Already cut over" and exit 0.

Rollback (if needed manually after this script): in a single
transaction, RENAME order_book_snapshots back to
order_book_snapshots_v2, then RENAME order_book_snapshots_legacy
back to order_book_snapshots. The data shape is preserved either
way; only the names move.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def table_exists(conn, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def has_id_column(conn, name: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({name})")
    return any(r[1] == "id" for r in cur.fetchall())


def main():
    print(f"[cutover] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        legacy_renamed_exists = table_exists(conn, "order_book_snapshots_legacy")
        v2_exists = table_exists(conn, "order_book_snapshots_v2")
        live_exists = table_exists(conn, "order_book_snapshots")

        # Idempotent shortcut: already cut over
        if legacy_renamed_exists and not v2_exists and live_exists:
            if not has_id_column(conn, "order_book_snapshots"):
                print("[cutover] Already cut over -- "
                      "order_book_snapshots_legacy exists, _v2 does not, "
                      "and live table has new schema. Exiting cleanly.")
                return 0
            else:
                print("[cutover] WARN: legacy renamed table exists but live "
                      "order_book_snapshots still has `id` column; state "
                      "is inconsistent. Aborting.", file=sys.stderr)
                return 4

        # Pre-conditions for forward cutover
        if not live_exists:
            print("[cutover] ERROR: order_book_snapshots does not exist",
                  file=sys.stderr)
            return 2
        if not v2_exists:
            print("[cutover] ERROR: order_book_snapshots_v2 does not exist; "
                  "Phase 0 must have created it. Did init_db run?",
                  file=sys.stderr)
            return 2
        if legacy_renamed_exists:
            print("[cutover] ERROR: order_book_snapshots_legacy already "
                  "exists -- partial prior cutover state? Aborting.",
                  file=sys.stderr)
            return 3

        # Sanity: live should still have the OLD schema (with `id`); v2
        # should be the NEW schema (no `id`). If either is wrong, abort.
        if not has_id_column(conn, "order_book_snapshots"):
            print("[cutover] ERROR: order_book_snapshots has no `id` column "
                  "-- already migrated? Inconsistent state. Aborting.",
                  file=sys.stderr)
            return 3
        if has_id_column(conn, "order_book_snapshots_v2"):
            print("[cutover] ERROR: order_book_snapshots_v2 has `id` column "
                  "-- v2 schema is wrong. Aborting.", file=sys.stderr)
            return 3

        # Pre-cutover counts
        n_live = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots"
        ).fetchone()[0]
        n_v2 = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots_v2"
        ).fetchone()[0]
        print(f"[cutover] Pre-cutover: legacy live = {n_live}, v2 = {n_v2}")
        if n_v2 < n_live:
            print(f"[cutover] WARN: v2 ({n_v2}) < legacy ({n_live}). "
                  f"Phase 2 backfill may not have completed. Aborting.",
                  file=sys.stderr)
            return 5

        # Atomic rename pair
        print("[cutover] Executing atomic RENAME pair...")
        t0 = time.perf_counter()
        conn.execute("BEGIN")
        try:
            conn.execute(
                "ALTER TABLE order_book_snapshots "
                "RENAME TO order_book_snapshots_legacy"
            )
            conn.execute(
                "ALTER TABLE order_book_snapshots_v2 "
                "RENAME TO order_book_snapshots"
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        t_elapsed = time.perf_counter() - t0
        print(f"[cutover] RENAME pair wall-clock: {t_elapsed:.3f} s")

        # Post-cutover verification
        new_legacy = table_exists(conn, "order_book_snapshots_legacy")
        new_live = table_exists(conn, "order_book_snapshots")
        new_v2 = table_exists(conn, "order_book_snapshots_v2")
        new_has_id = has_id_column(conn, "order_book_snapshots")
        new_legacy_has_id = (
            has_id_column(conn, "order_book_snapshots_legacy")
            if new_legacy else False
        )
        n_live_post = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots"
        ).fetchone()[0]
        n_legacy_post = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots_legacy"
        ).fetchone()[0]

        print()
        print("=" * 60)
        print("[cutover] PHASE 4 COMPLETE")
        print(f"  order_book_snapshots_legacy exists:    {new_legacy}")
        print(f"  order_book_snapshots_v2 exists:        {new_v2} "
              f"(should be False)")
        print(f"  order_book_snapshots exists:           {new_live}")
        print(f"  live table has `id` column:            {new_has_id} "
              f"(should be False)")
        print(f"  legacy renamed table has `id` column:  {new_legacy_has_id} "
              f"(should be True)")
        print(f"  live (new) row count:  {n_live_post}")
        print(f"  legacy renamed count:  {n_legacy_post}")
        print(f"  RENAME pair wall-clock: {t_elapsed:.3f} s")
        print("=" * 60)

        if new_v2:
            print("[cutover] FAIL: _v2 still exists after cutover",
                  file=sys.stderr)
            return 6
        if new_has_id:
            print("[cutover] FAIL: live table still has `id` column "
                  "after cutover", file=sys.stderr)
            return 6
        if not new_legacy_has_id:
            print("[cutover] FAIL: legacy renamed table is missing "
                  "`id` column", file=sys.stderr)
            return 6

        print("[cutover] Cutover verified. Live order_book_snapshots is "
              "Rule 35 conforming. Writer continues dual-writing for "
              "Cycle 23.5 burn-in window.")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
