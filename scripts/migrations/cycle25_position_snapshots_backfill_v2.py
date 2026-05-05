"""
Cycle 25 Phase 2 -- backfill position_snapshots_v2 from legacy.

Pure-SQL INSERT ... SELECT. The legacy `timestamp` column holds a
microsecond-precise ISO string (e.g.
"2026-04-30T02:24:09.613460+00:00"); the v2 schema splits this into
two columns:

  - v2.timestamp  INTEGER ms (derived via julianday/ROUND)
  - v2.datetime   TEXT (verbatim copy of legacy.timestamp)

Pre-step (per Cycle 23 lesson): create indexes on the natural-key
columns BEFORE the NOT EXISTS subquery runs. The compound PK on v2
already covers (snapshot_id, wallet, market_slug, outcome); legacy
has no such PK so we add a defense-in-depth index.

ms derivation:
    CAST(ROUND((julianday(timestamp) - 2440587.5) * 86400000) AS INTEGER)
Same formula as Cycle 23's order_book_snapshots backfill. ROUND() is
required because SQLite's julianday returns a double; for datetimes
with .NNN-precision fractional seconds (Cycle 23's input shape), the
product can land ~1 ULP below the integer about half the time, which
`CAST AS INTEGER` would truncate. ROUND-to-nearest fixes ULP underflow
for ms-precision sources.

NOTE for Cycle 25's microsecond-precision source: ROUND introduces a
new disagreement vs the dual-write writer's `int(time.time() * 1000)`
convention -- when the microsecond fraction is >= 500us, ROUND rounds
the v2.timestamp UP by 1 ms while the writer truncates. This produces
~50% rate of +1ms drift on backfilled rows vs Python's
int(datetime.fromisoformat(dt).timestamp() * 1000). The verify script
tolerates +/-1ms drift on Check 5 to acknowledge this. The drift is
harmless for this table (readers key on snapshot_id, not timestamp;
ms is a lossy quantization of microseconds anyway).

For future microsecond-source migrations, consider whether to (a)
keep ROUND and accept the 1ms drift (chosen here -- minimal risk,
established formula), or (b) switch to TRUNC + epsilon to match the
writer convention exactly. Either is defensible; consistency across
the recipe matters more than the specific choice.

NULL handling on the natural key: per Brief, market_slug and outcome
are nullable in the schema. Empirical pre-state (Cycle 25 brief-time):
zero NULLs in either column across all 65k rows -- everything is
empty-string `market_slug` plus a filled `outcome`. The COALESCE
clauses in the NOT EXISTS subquery are still included as defense in
depth in case future writes introduce NULLs.

Idempotent: if no legacy rows are missing from v2, prints
"Already backfilled" and exits 0.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "smart_money.db"


def main():
    print(f"[backfill] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Sanity: both tables must exist
        for tbl in ("position_snapshots", "position_snapshots_v2"):
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (tbl,),
            )
            if cur.fetchone() is None:
                print(f"[backfill] ERROR: table {tbl} missing", file=sys.stderr)
                return 2

        # Defense-in-depth indexes on legacy. v2's compound PK already
        # covers the lookup columns. Both are idempotent.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pos_legacy_natural "
            "ON position_snapshots(snapshot_id, wallet, market_slug, outcome)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pos_legacy_ts "
            "ON position_snapshots(timestamp)"
        )
        conn.commit()

        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots"
        ).fetchone()[0]
        v2_count_pre = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots_v2"
        ).fetchone()[0]
        count_missing = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots l "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM position_snapshots_v2 v "
            "  WHERE v.snapshot_id = l.snapshot_id "
            "    AND v.wallet = l.wallet "
            "    AND COALESCE(v.market_slug, '') = COALESCE(l.market_slug, '') "
            "    AND COALESCE(v.outcome, '') = COALESCE(l.outcome, '')"
            ")"
        ).fetchone()[0]

        print(
            f"[backfill] Pre-state: legacy={legacy_count}, v2={v2_count_pre}, "
            f"missing_in_v2={count_missing}"
        )

        if count_missing == 0:
            print(
                "[backfill] Already backfilled -- every legacy "
                "(snapshot_id, wallet, market_slug, outcome) is present "
                "in v2. Exiting cleanly."
            )
            return 0

        # Pure-SQL INSERT-SELECT. INSERT OR IGNORE protects against the
        # race window with the dual-write writer (which uses
        # INSERT OR REPLACE on the same natural key).
        backfill_sql = """
            INSERT OR IGNORE INTO position_snapshots_v2 (
                snapshot_id, timestamp, datetime,
                wallet, market_slug, market_title, outcome,
                size, avg_price, current_price, value_usd, pnl_usd
            )
            SELECT
                l.snapshot_id,
                CAST(ROUND((julianday(l.timestamp) - 2440587.5) * 86400000)
                     AS INTEGER) AS ts_ms,
                l.timestamp AS dt,
                l.wallet, l.market_slug, l.market_title, l.outcome,
                l.size, l.avg_price, l.current_price, l.value_usd, l.pnl_usd
            FROM position_snapshots l
            WHERE NOT EXISTS (
                SELECT 1 FROM position_snapshots_v2 v
                WHERE v.snapshot_id = l.snapshot_id
                  AND v.wallet = l.wallet
                  AND COALESCE(v.market_slug, '') = COALESCE(l.market_slug, '')
                  AND COALESCE(v.outcome, '') = COALESCE(l.outcome, '')
            )
        """

        t0 = time.perf_counter()
        cur = conn.execute(backfill_sql)
        inserted = cur.rowcount
        conn.commit()
        t_elapsed = time.perf_counter() - t0
        print(f"[backfill] INSERT-SELECT wall-clock: {t_elapsed:.3f} s")
        print(f"[backfill] Inserted {inserted} rows")

        legacy_post = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots"
        ).fetchone()[0]
        v2_post = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots_v2"
        ).fetchone()[0]
        still_missing = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots l "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM position_snapshots_v2 v "
            "  WHERE v.snapshot_id = l.snapshot_id "
            "    AND v.wallet = l.wallet "
            "    AND COALESCE(v.market_slug, '') = COALESCE(l.market_slug, '') "
            "    AND COALESCE(v.outcome, '') = COALESCE(l.outcome, '')"
            ")"
        ).fetchone()[0]

        print()
        print("=" * 60)
        print("[backfill] PHASE 2 COMPLETE")
        print(
            f"  legacy: {legacy_count} -> {legacy_post} "
            f"(any growth from live writer during backfill)"
        )
        print(f"  v2:     {v2_count_pre} -> {v2_post}")
        print(f"  missing post-backfill: {still_missing} (must be 0)")
        print(f"  wall-clock: {t_elapsed:.3f} s")
        print("=" * 60)

        if still_missing != 0:
            print(
                f"[backfill] FAIL: {still_missing} rows still missing "
                f"from v2 after backfill",
                file=sys.stderr,
            )
            return 5
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
