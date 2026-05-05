"""
Cycle 24 Phase 3 -- verify legacy + v2 consistency before cutover.

ABORTs (non-zero exit) on any failure. The Brief is explicit: if Phase
3 fails, do not proceed to Phase 4 cutover.

Checks:

1. v2 row count >= legacy row count. v2 may be larger because the
   dual-write writer captures fresh time.time() per insert, so two
   live snapshots in the same wallclock second are deduplicated by
   legacy's UNIQUE(slug, timestamp) but accepted by v2 (which has
   ms precision). v2 < legacy is impossible after Phase 2.

2. Every legacy (slug, timestamp) has a matching v2 row at
   (slug, timestamp * 1000). No multiple matches in v2 -- compound
   PK already enforces this, but verify anyway.

3. Sample 100 random legacy rows and verify byte-identity with their
   v2 partners. Backfilled rows: v2.timestamp == legacy.timestamp *
   1000 exactly. Dual-write rows: v2.timestamp is within 0-999ms of
   legacy.timestamp * 1000 (sub-second drift from independent
   time.time() calls in the writer). Other columns (yes_mid, yes_bid,
   yes_ask, spread) byte-identical.

Exit codes:
  0 = all checks passed
  2 = total-row check failed
  3 = (slug, timestamp) coverage check failed
  4 = byte-identity sample check failed
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "live_collector.db"

COLS_AFTER_TS = ["yes_mid", "yes_bid", "yes_ask", "spread"]


def main():
    print(f"[verify] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        # ----- Check 1: total rows -----
        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots"
        ).fetchone()[0]
        v2_count = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots_v2"
        ).fetchone()[0]
        print(f"[verify] Check 1: total rows -- legacy={legacy_count}, "
              f"v2={v2_count}")
        if v2_count < legacy_count:
            print(f"[verify] FAIL: v2 ({v2_count}) < legacy ({legacy_count})",
                  file=sys.stderr)
            return 2

        # ----- Check 2: every legacy (slug, timestamp) -> v2 -----
        # Backfill key is (slug, timestamp*1000); dual-write key is
        # (slug, ms_close_to_sec*1000). Either way, every legacy row
        # should map to an _exact_ (slug, timestamp*1000) v2 row
        # because the backfill INSERT is exact.
        missing = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots l "
            "WHERE NOT EXISTS (SELECT 1 FROM price_snapshots_v2 v "
            "WHERE v.slug = l.slug AND v.timestamp = l.timestamp * 1000)"
        ).fetchone()[0]
        # Multi-match check (compound PK already enforces but verify):
        dup_v2 = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT slug, timestamp, COUNT(*) c "
            "  FROM price_snapshots_v2 "
            "  GROUP BY slug, timestamp HAVING c > 1"
            ")"
        ).fetchone()[0]
        print(f"[verify] Check 2: legacy rows missing from v2 = {missing}; "
              f"v2 (slug, timestamp) duplicates = {dup_v2}")
        if missing != 0 or dup_v2 != 0:
            print(f"[verify] FAIL: coverage check (missing={missing}, "
                  f"dup={dup_v2})", file=sys.stderr)
            return 3

        # ----- Check 3: byte-identity sample -----
        # 100 random legacy rows. For each, look up the matching v2 row
        # at (slug, timestamp*1000). Verify yes_mid, yes_bid, yes_ask,
        # spread byte-identical. Timestamp relation: v2.ts ==
        # legacy.ts * 1000 exactly (backfill) OR v2.ts within
        # legacy.ts*1000 .. legacy.ts*1000 + 999 (dual-write -- but
        # the dual-write rows ALSO have an exact-match row from backfill
        # at v2.ts == legacy.ts*1000 because the backfill INSERTed it
        # using legacy.ts -- so the JOIN by ts*1000 always finds a row).
        sample = conn.execute(
            "SELECT id, slug, timestamp, yes_mid, yes_bid, yes_ask, spread "
            "FROM price_snapshots ORDER BY RANDOM() LIMIT 100"
        ).fetchall()

        mismatches = []
        for l in sample:
            v = conn.execute(
                "SELECT slug, timestamp, yes_mid, yes_bid, yes_ask, spread, "
                "       datetime "
                "FROM price_snapshots_v2 "
                "WHERE slug = ? AND timestamp = ?",
                (l["slug"], l["timestamp"] * 1000),
            ).fetchone()
            if v is None:
                mismatches.append({
                    "lid": l["id"],
                    "reason": "no v2 row at (slug, timestamp*1000)",
                    "slug": l["slug"],
                    "legacy_ts": l["timestamp"],
                })
                continue

            # All other columns byte-identical
            mismatch_col = None
            for c in COLS_AFTER_TS:
                if l[c] != v[c]:
                    mismatch_col = c
                    break
            if mismatch_col:
                mismatches.append({
                    "lid": l["id"],
                    "reason": f"{mismatch_col} mismatch",
                    "legacy": l[mismatch_col],
                    "v2": v[mismatch_col],
                    "slug": l["slug"],
                })
                continue

        print(f"[verify] Check 3: sampled {len(sample)} legacy rows "
              f"-> {len(mismatches)} mismatches")
        if mismatches:
            for m in mismatches[:5]:
                print(f"  {m}")
            print(f"[verify] FAIL: byte-identity sample "
                  f"({len(mismatches)} mismatches)", file=sys.stderr)
            return 4

        # ----- Summary -----
        print()
        print("=" * 60)
        print("[verify] PHASE 3 COMPLETE -- all checks passed")
        print(f"  legacy rows:                 {legacy_count}")
        print(f"  v2 rows:                     {v2_count}")
        print(f"  legacy rows missing from v2: {missing}")
        print(f"  v2 (slug, timestamp) dups:   {dup_v2}")
        print(f"  byte-identity samples:       {len(sample)}/{len(sample)} passed")
        print("=" * 60)
        print("[verify] OK to proceed to Phase 4 cutover.")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
