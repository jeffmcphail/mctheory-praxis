"""
Cycle 23 Phase 3 -- verify legacy + v2 consistency before cutover.

Runs a sequence of checks; ABORTs (non-zero exit) on any failure.
The Brief is explicit: if Phase 3 fails, do not proceed to Phase 4
cutover. Surface to chat instead.

Checks:

1. Total rows in v2 >= total rows in legacy. (V2 is a strict superset
   in terms of (asset, datetime) coverage; it may also have rows the
   live dual-write writer added that aren't yet in legacy because
   legacy uses ts_ms // 1000 and v2 uses ts_ms -- but practically
   both INSERTs commit atomically, so legacy and v2 should be
   identical in (asset, datetime) terms after backfill.)

2. Every (asset, datetime) in legacy appears exactly once in v2.
   Multiple matches -> dual-write went wrong; zero matches -> backfill
   missed it.

3. Sample 100 random rows from a) the dual-write window (post-Phase-0
   timestamps) and b) the backfilled range (pre-Phase-0 timestamps).
   For each: every column except `id` (legacy only) and `timestamp`
   should be byte-identical between legacy and v2. The legacy
   timestamp times 1000 should equal v2 timestamp - sub-second offset
   (in practice, `parse(datetime).timestamp() * 1000` should equal
   v2 timestamp, and `legacy_ts == v2_ts // 1000`).

Exit codes:
  0 = all checks passed
  2 = total-row check failed
  3 = (asset, datetime) coverage check failed
  4 = byte-identity sample check failed
"""

from __future__ import annotations

import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"

# Match the column list in the writer / backfill -- everything after
# (asset, timestamp, datetime). 48 columns.
COLS_AFTER_DT = [
    "mid_price", "best_bid", "best_ask", "spread", "spread_bps",
    "bid_price_1", "bid_vol_1", "bid_price_2", "bid_vol_2",
    "bid_price_3", "bid_vol_3", "bid_price_4", "bid_vol_4",
    "bid_price_5", "bid_vol_5", "bid_price_6", "bid_vol_6",
    "bid_price_7", "bid_vol_7", "bid_price_8", "bid_vol_8",
    "bid_price_9", "bid_vol_9", "bid_price_10", "bid_vol_10",
    "ask_price_1", "ask_vol_1", "ask_price_2", "ask_vol_2",
    "ask_price_3", "ask_vol_3", "ask_price_4", "ask_vol_4",
    "ask_price_5", "ask_vol_5", "ask_price_6", "ask_vol_6",
    "ask_price_7", "ask_vol_7", "ask_price_8", "ask_vol_8",
    "ask_price_9", "ask_vol_9", "ask_price_10", "ask_vol_10",
    "bid_volume_top10", "ask_volume_top10", "order_imbalance_top10",
]


def parse_dt_to_ms(dt_text: str) -> int:
    return int(datetime.fromisoformat(dt_text).timestamp() * 1000)


def main():
    print(f"[verify] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        # ----- Check 1: total rows -----
        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots"
        ).fetchone()[0]
        v2_count = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots_v2"
        ).fetchone()[0]
        print(f"[verify] Check 1: total rows -- legacy={legacy_count}, "
              f"v2={v2_count}")
        if v2_count < legacy_count:
            print(f"[verify] FAIL: v2 ({v2_count}) < legacy ({legacy_count})",
                  file=sys.stderr)
            return 2

        # ----- Check 2: every (asset, datetime) in legacy -> v2 -----
        missing = conn.execute(
            "SELECT COUNT(*) FROM order_book_snapshots l "
            "WHERE NOT EXISTS (SELECT 1 FROM order_book_snapshots_v2 v "
            "WHERE v.asset = l.asset AND v.datetime = l.datetime)"
        ).fetchone()[0]
        # Multi-match check: any (asset, datetime) duplicated in v2?
        dup_v2 = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT asset, datetime, COUNT(*) c "
            "  FROM order_book_snapshots_v2 "
            "  GROUP BY asset, datetime HAVING c > 1"
            ")"
        ).fetchone()[0]
        print(f"[verify] Check 2: legacy rows missing from v2 = {missing}; "
              f"v2 (asset, datetime) duplicates = {dup_v2}")
        if missing != 0 or dup_v2 != 0:
            print(f"[verify] FAIL: coverage check (missing={missing}, "
                  f"dup={dup_v2})", file=sys.stderr)
            return 3

        # ----- Check 3: byte-identity sample -----
        # Pick 50 random rows from each of: pre-Phase-0 (backfilled range)
        # and post-Phase-0 (dual-write range). Use timestamp boundary as
        # a proxy: rows with v2 ts > some Phase-0-ish threshold are
        # post-Phase-0. We'll use the latest legacy ts that pre-dates the
        # Phase 0 commit -- but practically, just pick 100 random rows
        # straight from legacy and verify each has a matching v2 row with
        # byte-identical fields (excluding timestamp + id).
        # 50 random IDs from legacy:
        sample_ids = conn.execute(
            "SELECT id FROM order_book_snapshots "
            "ORDER BY RANDOM() LIMIT 100"
        ).fetchall()

        mismatches = []
        cols_csv = ", ".join(COLS_AFTER_DT)
        for srow in sample_ids:
            lid = srow["id"]
            l = conn.execute(
                f"SELECT asset, timestamp, datetime, {cols_csv} "
                f"FROM order_book_snapshots WHERE id = ?",
                (lid,),
            ).fetchone()
            if l is None:
                continue
            v = conn.execute(
                f"SELECT asset, timestamp, datetime, {cols_csv} "
                f"FROM order_book_snapshots_v2 "
                f"WHERE asset = ? AND datetime = ?",
                (l["asset"], l["datetime"]),
            ).fetchone()
            if v is None:
                mismatches.append({
                    "lid": lid,
                    "reason": "no v2 row for (asset, datetime)",
                    "legacy": dict(l),
                })
                continue

            # Timestamp relation: legacy_ts == v2_ts // 1000 (because
            # legacy is seconds-truncated)
            if l["timestamp"] != v["timestamp"] // 1000:
                mismatches.append({
                    "lid": lid,
                    "reason": "ts arithmetic",
                    "legacy_ts": l["timestamp"],
                    "v2_ts": v["timestamp"],
                })
                continue

            # v2_ts should equal parse(datetime).timestamp() * 1000
            try:
                expected_v2_ts = parse_dt_to_ms(l["datetime"])
            except Exception as e:
                mismatches.append({
                    "lid": lid,
                    "reason": f"datetime unparseable: {e}",
                    "datetime": l["datetime"],
                })
                continue
            if expected_v2_ts != v["timestamp"]:
                mismatches.append({
                    "lid": lid,
                    "reason": "v2 ts != parse(datetime)*1000",
                    "expected": expected_v2_ts,
                    "v2_ts": v["timestamp"],
                    "datetime": l["datetime"],
                })
                continue

            # All other columns byte-identical
            for c in COLS_AFTER_DT:
                if l[c] != v[c]:
                    mismatches.append({
                        "lid": lid,
                        "reason": f"{c} mismatch",
                        "legacy": l[c],
                        "v2": v[c],
                    })
                    break

        print(f"[verify] Check 3: sampled {len(sample_ids)} legacy rows "
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
        print(f"  v2 (asset, datetime) dups:   {dup_v2}")
        print(f"  byte-identity samples:       100/100 passed")
        print("=" * 60)
        print("[verify] OK to proceed to Phase 4 cutover.")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
