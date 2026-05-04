"""
Cycle 21.5 -- funding_rates writer alignment hotfix dedup script.

Cycle 21 migrated funding_rates to ms by multiplying legacy seconds * 1000,
producing seconds-aligned `.000`-ms timestamps. The post-Cycle-21 writer in
`collect_funding_rates` then started storing Binance's `fundingTime` raw,
which carries sub-second jitter (e.g., `1777795200003`). The compound PK
on `(asset, timestamp)` does NOT collapse `.000` and `.NNN` into one row,
so each new funding event accumulated a duplicate row.

Empirical: at Brief-write time funding_rates had 2,240 rows, 2,214 distinct
`(asset, datetime)` events, 26 duplicate rows -- one `.000`-ms legacy row
plus one `.NNN`-ms post-migration row per duplicated event, across 13 events
x 2 assets (BTC + ETH).

Funding-rate values are byte-identical across the duplicate pair (same
Binance event), so the dedup is provably lossless. This script keeps the
`.000`-aligned row and deletes the jittered one. The companion writer fix
(this same cycle) prevents future occurrences by truncating `fundingTime`
to seconds-aligned ms before INSERT.

Idempotent: re-running on an already-deduped table prints "Already deduped"
and exits 0.

No backup is created (lossless dedup, ~26 rows, transactional). If a backup
is wanted, copy `data/crypto_data.db` -> `data/crypto_data.db.cycle21_5_backup`
before running.

Acceptance:
  - Post-dedup row count == COUNT(DISTINCT asset || '|' || datetime)
  - Per-asset count drops by exactly the number of duplicate groups
  - All `funding_rates.timestamp % 1000 == 0` post-dedup
  - Funding-rate values byte-identical across the dup pair (verified
    pre-DELETE; would raise if any pair disagrees)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def count_dupes(conn) -> int:
    return conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT asset || '|' || datetime) "
        "FROM funding_rates"
    ).fetchone()[0]


def per_asset_counts(conn):
    return conn.execute(
        "SELECT asset, COUNT(*) FROM funding_rates GROUP BY asset ORDER BY asset"
    ).fetchall()


def find_dup_groups(conn):
    return conn.execute(
        """
        SELECT asset, datetime, COUNT(*) AS n
        FROM funding_rates
        GROUP BY asset, datetime
        HAVING COUNT(*) > 1
        ORDER BY datetime, asset
        """
    ).fetchall()


def main():
    print(f"[dedup] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        pre_total = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
        pre_distinct = conn.execute(
            "SELECT COUNT(DISTINCT asset || '|' || datetime) FROM funding_rates"
        ).fetchone()[0]
        pre_dupes = pre_total - pre_distinct
        pre_per_asset = per_asset_counts(conn)
        pre_jittered = conn.execute(
            "SELECT COUNT(*) FROM funding_rates WHERE timestamp % 1000 != 0"
        ).fetchone()[0]

        print(f"[dedup] Pre-state: total={pre_total}, distinct={pre_distinct}, "
              f"dupes={pre_dupes}, jittered_ts={pre_jittered}")
        print(f"[dedup] Pre-state per-asset: {pre_per_asset}")

        if pre_dupes == 0:
            print("[dedup] Already deduped -- no duplicate (asset, datetime) groups found. "
                  "Exiting cleanly.")
            return 0

        groups = find_dup_groups(conn)
        print(f"[dedup] Found {len(groups)} duplicate (asset, datetime) groups")
        print(f"[dedup] First few: {groups[:4]}")

        # Verify funding_rate byte-identity within each duplicate group BEFORE
        # touching anything. This is the lossless-dedup guarantee.
        mismatches = []
        for asset, dt, _n in groups:
            rates = conn.execute(
                "SELECT DISTINCT funding_rate FROM funding_rates "
                "WHERE asset = ? AND datetime = ?",
                (asset, dt),
            ).fetchall()
            if len(rates) > 1:
                mismatches.append((asset, dt, [r[0] for r in rates]))

        if mismatches:
            print(f"[dedup] ERROR: funding_rate values disagree within "
                  f"{len(mismatches)} duplicate group(s). Aborting -- dedup "
                  f"would NOT be lossless. Sample: {mismatches[:3]}",
                  file=sys.stderr)
            return 4

        print("[dedup] All duplicate groups have byte-identical funding_rate "
              "values. Dedup is lossless. Proceeding.")

        # Delete jittered rows: keep the seconds-aligned row, drop the rest.
        # `timestamp % 1000 != 0` identifies post-Cycle-21 writer rows.
        conn.execute("BEGIN")
        try:
            cur = conn.execute(
                "DELETE FROM funding_rates WHERE timestamp % 1000 != 0"
            )
            deleted = cur.rowcount
            print(f"[dedup] DELETE FROM funding_rates WHERE timestamp % 1000 != 0 "
                  f"-> {deleted} rows removed")

            # Verify post-state inside the transaction; rollback if anything off.
            post_total = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
            post_distinct = conn.execute(
                "SELECT COUNT(DISTINCT asset || '|' || datetime) FROM funding_rates"
            ).fetchone()[0]
            post_jittered = conn.execute(
                "SELECT COUNT(*) FROM funding_rates WHERE timestamp % 1000 != 0"
            ).fetchone()[0]

            if post_total != post_distinct:
                raise RuntimeError(
                    f"Post-DELETE total ({post_total}) != distinct ({post_distinct}); "
                    f"more dupes than expected, ROLLBACK"
                )
            if post_jittered != 0:
                raise RuntimeError(
                    f"Post-DELETE jittered_ts ({post_jittered}) != 0; ROLLBACK"
                )
            if deleted != pre_dupes:
                raise RuntimeError(
                    f"Deleted row count ({deleted}) != pre_dupes ({pre_dupes}); ROLLBACK"
                )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        post_total = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
        post_distinct = conn.execute(
            "SELECT COUNT(DISTINCT asset || '|' || datetime) FROM funding_rates"
        ).fetchone()[0]
        post_per_asset = per_asset_counts(conn)
        post_jittered = conn.execute(
            "SELECT COUNT(*) FROM funding_rates WHERE timestamp % 1000 != 0"
        ).fetchone()[0]

        print()
        print("=" * 60)
        print("[dedup] HOTFIX COMPLETE")
        print(f"  rows: {pre_total} -> {post_total} (deleted {pre_total - post_total})")
        print(f"  distinct (asset|datetime): {pre_distinct} -> {post_distinct}")
        print(f"  jittered_ts: {pre_jittered} -> {post_jittered}")
        print(f"  per-asset pre:  {pre_per_asset}")
        print(f"  per-asset post: {post_per_asset}")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
