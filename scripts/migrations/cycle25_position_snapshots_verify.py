"""
Cycle 25 Phase 3 -- verify legacy + v2 consistency before cutover.

ABORTs (non-zero exit) on any failure. The Brief is explicit: if Phase
3 fails, do not proceed to Phase 4 cutover.

Checks:

1. v2 row count >= legacy row count.

2. Every legacy (snapshot_id, wallet, market_slug, outcome) has a
   matching v2 row, COALESCE-aware on the nullable columns. Zero
   missing.

3. Zero (snapshot_id, wallet, market_slug, outcome) duplicates in v2.
   Compound PK already enforces this; verify anyway.

4. Sample 100 random legacy rows: for each, the matching v2 row's
   shared columns (market_title, size, avg_price, current_price,
   value_usd, pnl_usd) are byte-identical. Also verify v2.datetime ==
   legacy.timestamp (TEXT-byte-identical -- v2.datetime is the
   renamed legacy column).

5. Sample 10 random v2 rows: verify the round-trip
   int(datetime.fromisoformat(v2.datetime).timestamp() * 1000) is
   within +/- 1 ms of v2.timestamp. The +/- 1 ms tolerance
   acknowledges that two derivation paths exist for v2.timestamp:

     (a) Phase 2 backfill: SQLite julianday/ROUND.
         For microsecond-precision datetimes (.NNNNNN), ROUND of the
         float (julianday - 2440587.5) * 86400000 rounds the
         sub-millisecond fraction to nearest, which produces a +1 ms
         disagreement vs Python's int(... * 1000) when the
         microsecond fraction is >= 500us (~50% of rows).

     (b) Dual-write rows: writer captures int(time.time() * 1000),
         which truncates the sub-millisecond fraction. For these
         rows the round-trip vs v2.datetime is typically exact (0 ms
         drift) because `now_iso` and `now_ms` are computed within
         microseconds of each other.

   The 1 ms drift is harmless for this table -- ms precision is what
   it is, and readers key on snapshot_id, not v2.timestamp.

Exit codes:
  0 = all checks passed
  2 = total-row check failed
  3 = coverage / dup check failed
  4 = byte-identity sample check failed
  5 = ms round-trip sample check failed
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "smart_money.db"

COLS_TO_COMPARE = [
    "market_title",
    "size",
    "avg_price",
    "current_price",
    "value_usd",
    "pnl_usd",
]


def main():
    print(f"[verify] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        # ----- Check 1: total rows -----
        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots"
        ).fetchone()[0]
        v2_count = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots_v2"
        ).fetchone()[0]
        print(
            f"[verify] Check 1: total rows -- legacy={legacy_count}, "
            f"v2={v2_count}"
        )
        if v2_count < legacy_count:
            print(
                f"[verify] FAIL: v2 ({v2_count}) < legacy ({legacy_count})",
                file=sys.stderr,
            )
            return 2

        # ----- Check 2: coverage + dups -----
        missing = conn.execute(
            "SELECT COUNT(*) FROM position_snapshots l "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM position_snapshots_v2 v "
            "  WHERE v.snapshot_id = l.snapshot_id "
            "    AND v.wallet = l.wallet "
            "    AND COALESCE(v.market_slug, '') = COALESCE(l.market_slug, '') "
            "    AND COALESCE(v.outcome, '') = COALESCE(l.outcome, '')"
            ")"
        ).fetchone()[0]
        dup_v2 = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT snapshot_id, wallet, "
            "         COALESCE(market_slug, '') AS ms, "
            "         COALESCE(outcome, '') AS oc, "
            "         COUNT(*) c "
            "  FROM position_snapshots_v2 "
            "  GROUP BY snapshot_id, wallet, ms, oc HAVING c > 1"
            ")"
        ).fetchone()[0]
        print(
            f"[verify] Check 2/3: legacy rows missing from v2 = {missing}; "
            f"v2 natural-key duplicates = {dup_v2}"
        )
        if missing != 0 or dup_v2 != 0:
            print(
                f"[verify] FAIL: coverage check "
                f"(missing={missing}, dup={dup_v2})",
                file=sys.stderr,
            )
            return 3

        # ----- Check 4: byte-identity sample -----
        sample = conn.execute(
            "SELECT id, snapshot_id, timestamp AS legacy_ts, wallet, "
            "       market_slug, market_title, outcome, size, avg_price, "
            "       current_price, value_usd, pnl_usd "
            "FROM position_snapshots ORDER BY RANDOM() LIMIT 100"
        ).fetchall()

        mismatches = []
        for l in sample:
            v = conn.execute(
                "SELECT snapshot_id, timestamp AS ts_ms, datetime, "
                "       wallet, market_slug, market_title, outcome, "
                "       size, avg_price, current_price, value_usd, pnl_usd "
                "FROM position_snapshots_v2 "
                "WHERE snapshot_id = ? AND wallet = ? "
                "  AND COALESCE(market_slug, '') = COALESCE(?, '') "
                "  AND COALESCE(outcome, '') = COALESCE(?, '')",
                (l["snapshot_id"], l["wallet"], l["market_slug"], l["outcome"]),
            ).fetchone()
            if v is None:
                mismatches.append({
                    "lid": l["id"],
                    "reason": "no v2 row at natural key",
                    "snapshot_id": l["snapshot_id"],
                    "wallet": l["wallet"],
                })
                continue

            # v2.datetime should equal legacy.timestamp byte-for-byte
            if v["datetime"] != l["legacy_ts"]:
                mismatches.append({
                    "lid": l["id"],
                    "reason": "datetime mismatch",
                    "legacy_ts": l["legacy_ts"],
                    "v2_dt": v["datetime"],
                })
                continue

            # All shared columns byte-identical
            mismatch_col = None
            for c in COLS_TO_COMPARE:
                if l[c] != v[c]:
                    mismatch_col = c
                    break
            if mismatch_col:
                mismatches.append({
                    "lid": l["id"],
                    "reason": f"{mismatch_col} mismatch",
                    "legacy": l[mismatch_col],
                    "v2": v[mismatch_col],
                })
                continue

        print(
            f"[verify] Check 4: sampled {len(sample)} legacy rows "
            f"-> {len(mismatches)} mismatches"
        )
        if mismatches:
            for m in mismatches[:5]:
                print(f"  {m}")
            print(
                f"[verify] FAIL: byte-identity sample "
                f"({len(mismatches)} mismatches)",
                file=sys.stderr,
            )
            return 4

        # ----- Check 5: datetime <-> ms round-trip -----
        rt_sample = conn.execute(
            "SELECT timestamp AS ts_ms, datetime "
            "FROM position_snapshots_v2 ORDER BY RANDOM() LIMIT 10"
        ).fetchall()
        rt_mismatches = []
        rt_within_tol = 0
        for r in rt_sample:
            py_ms = int(datetime.fromisoformat(r["datetime"]).timestamp() * 1000)
            delta = r["ts_ms"] - py_ms
            if abs(delta) > 1:
                rt_mismatches.append({
                    "datetime": r["datetime"],
                    "v2_ts_ms": r["ts_ms"],
                    "py_ms": py_ms,
                    "delta": delta,
                })
            elif delta != 0:
                rt_within_tol += 1

        rt_exact = len(rt_sample) - rt_within_tol - len(rt_mismatches)
        print(
            f"[verify] Check 5: round-trip sampled {len(rt_sample)} v2 rows "
            f"-> {rt_exact} exact, {rt_within_tol} within +/-1ms tolerance, "
            f"{len(rt_mismatches)} out of tolerance"
        )
        if rt_mismatches:
            for m in rt_mismatches[:5]:
                print(f"  {m}")
            print(
                f"[verify] FAIL: ms round-trip "
                f"({len(rt_mismatches)} out-of-tolerance mismatches)",
                file=sys.stderr,
            )
            return 5

        # ----- Summary -----
        print()
        print("=" * 60)
        print("[verify] PHASE 3 COMPLETE -- all checks passed")
        print(f"  legacy rows:                 {legacy_count}")
        print(f"  v2 rows:                     {v2_count}")
        print(f"  legacy rows missing from v2: {missing}")
        print(f"  v2 natural-key dups:         {dup_v2}")
        print(
            f"  byte-identity samples:       "
            f"{len(sample) - len(mismatches)}/{len(sample)} passed"
        )
        print(
            f"  ms round-trip samples:       "
            f"{len(rt_sample) - len(rt_mismatches)}/{len(rt_sample)} "
            f"within +/-1ms tolerance"
        )
        print("=" * 60)
        print("[verify] OK to proceed to Phase 4 cutover.")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
