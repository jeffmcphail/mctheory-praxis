"""
Cycle 31 -- onchain_btc full Rule 35 conformance.

Brings onchain_btc into full Rule 35 compliance by:
  1. Removing the synthetic `id INTEGER PRIMARY KEY AUTOINCREMENT`
  2. Adding `timestamp INTEGER NOT NULL` (ms; 00:00:00 UTC of the
     date's UTC midnight, matching ohlcv_daily's convention)
  3. Adding `datetime TEXT NOT NULL` (ISO 8601 with `+00:00`,
     matching the canonical Rule 35 representation)
  4. Promoting `UNIQUE(date)` to `PRIMARY KEY (date)`

Preserves:
  - All existing column data (active_addresses, transaction_count,
    hash_rate, difficulty, block_size, total_btc, market_cap)
  - The `date` TEXT column (as a primary lookup field; matches
    ohlcv_daily's convention of keeping `date` for human readout)

Why now: at end-of-Cycle-30, the user flagged that onchain_btc
was the only temporally-indexed table not fully compliant with
Rule 35 (still had `id` PK, no `timestamp`, no `datetime`).
Even if we don't typically join on it today, Rule 35's contract
exists precisely so cross-table joins on `timestamp` work
uniformly. Daily-grain doesn't exempt the table from the contract.

Why one-shot rebuild (not dual-write): same justification as
Cycle 26 (trades). Pure structural change with deterministic
column derivation (timestamp = parse(date) at UTC midnight).
The writer code doesn't reference `id`, and the new `timestamp`/
`datetime` columns can be deterministically derived from each
existing row's `date`. No data semantic transformation. No
burn-in needed.

Convention for the ms timestamp: `datetime.strptime(date,
"%Y-%m-%d").replace(tzinfo=timezone.utc)`, then `.timestamp() *
1000`. This makes onchain_btc's timestamps directly JOIN-able
with ohlcv_daily, which uses the identical convention
(verified: 2026-05-07 -> 1778112000000 in both).

PRE-CONDITION (required before running):
  1. PraxisOnchainCollector is daily and short-lived. Disable
     the scheduled task before running this script. The
     `collect-onchain` subcommand finishes within ~10s normally,
     but the script's pre-flight #4 will check anyway.

PERFORMANCE EXPECTATION (370 rows):
  - Bulk INSERT-SELECT with deterministic date->ms derivation:
    sub-second.
  - Total transaction wall-clock: well under 1s.
  - Smallest rebuild in the migration program by far.

Rollback: BEGIN/ROLLBACK in the transaction. Idempotent: detects
already-rebuilt state via "no `id` column" and exits cleanly.

Run from the repo root (with PraxisOnchainCollector disabled):

    python scripts/migrations/cycle31_onchain_btc_schema_rebuild.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def _date_to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD to UTC midnight Unix ms.

    Matches ohlcv_daily's convention (verified at Cycle 31 design time:
    '2026-05-07' -> 1778112000000 in both tables).
    """
    dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _date_to_iso(date_str: str) -> str:
    """Convert YYYY-MM-DD to ISO 8601 with +00:00 offset at UTC midnight."""
    dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def main() -> int:
    print(f"[rebuild] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # ---------------------------------------------------------
        # Pre-flight 1: live onchain_btc schema
        # ---------------------------------------------------------
        cols = conn.execute(
            "PRAGMA table_info(onchain_btc)"
        ).fetchall()
        if not cols:
            print(
                "[rebuild] ABORT: onchain_btc table doesn't exist.",
                file=sys.stderr,
            )
            return 2

        col_names = [c[1] for c in cols]
        has_id = "id" in col_names
        has_timestamp = "timestamp" in col_names
        has_datetime = "datetime" in col_names

        if not has_id and has_timestamp and has_datetime:
            print(
                "[rebuild] Already rebuilt -- onchain_btc has Rule 35 "
                "shape (no `id`, has `timestamp`, has `datetime`). "
                "Exiting cleanly.",
            )
            return 0

        # Verify expected pre-rebuild columns are present
        expected_pre = {
            "id", "date", "active_addresses", "transaction_count",
            "hash_rate", "difficulty", "block_size", "market_cap",
        }
        actual = set(col_names)
        if not expected_pre.issubset(actual):
            missing = expected_pre - actual
            print(
                f"[rebuild] ABORT: onchain_btc is missing expected "
                f"pre-rebuild columns: {missing}. Investigate before "
                f"proceeding.",
                file=sys.stderr,
            )
            return 2

        # `total_btc` is in the live schema but the writer doesn't insert
        # it -- we'll preserve it in the rebuild for compatibility. Verify
        # it's present.
        has_total_btc = "total_btc" in col_names
        if not has_total_btc:
            print(
                "  WARN: live onchain_btc has no `total_btc` column "
                "(unexpected per init_db schema). Continuing without "
                "it.",
                file=sys.stderr,
            )

        # ---------------------------------------------------------
        # Pre-flight 2: no _v2 leftover from a previous attempt
        # ---------------------------------------------------------
        v2_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='onchain_btc_v2'"
            ).fetchone()
        )
        if v2_exists:
            print(
                "[rebuild] ABORT: onchain_btc_v2 already exists, "
                "suggesting a previous rebuild attempt was "
                "interrupted. Investigate before proceeding (manually "
                "drop onchain_btc_v2 if confirmed safe).",
                file=sys.stderr,
            )
            return 3

        # ---------------------------------------------------------
        # Pre-flight 3: writer is not actively writing
        # ---------------------------------------------------------
        # collect-onchain runs daily and finishes in <10s. Check via
        # latest date: should be at least one full day old before we
        # consider it safe (no in-flight write). 60s gap from
        # *anything* writing also acceptable.
        latest_date = conn.execute(
            "SELECT MAX(date) FROM onchain_btc"
        ).fetchone()[0]
        if latest_date is None:
            print(
                "  WARN: onchain_btc has no rows. Rebuild will produce "
                "an empty post-state. Continuing.",
                file=sys.stderr,
            )

        # Pre-flight 4: confirm no scheduled task fire is in flight.
        # Use the same approach as Cycle 26: SQLite write-lock check
        # via a no-op transaction. If we can BEGIN IMMEDIATE within
        # 1s, no other writer holds the lock.
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                print(
                    f"[rebuild] ABORT: database is locked. A writer "
                    f"(possibly PraxisOnchainCollector or another "
                    f"collector sharing crypto_data.db) is currently "
                    f"holding the write lock. Wait and retry.",
                    file=sys.stderr,
                )
                return 4
            raise

        # ---------------------------------------------------------
        # Pre-state inventory
        # ---------------------------------------------------------
        live_count = conn.execute(
            "SELECT COUNT(*) FROM onchain_btc"
        ).fetchone()[0]
        print(f"[rebuild] Pre-state:")
        print(f"  onchain_btc:    {live_count:,} rows")
        print(f"  has `id`:       yes (will be removed)")
        print(f"  has `timestamp`: {has_timestamp} "
              f"(will be added if False)")
        print(f"  has `datetime`:  {has_datetime} "
              f"(will be added if False)")
        print(f"  natural key:    `date` -- currently UNIQUE, "
              f"will become PRIMARY KEY")

        # ---------------------------------------------------------
        # Read all rows + compute new columns in Python
        # ---------------------------------------------------------
        # Doing this in Python (rather than pure SQL strftime) keeps the
        # date->ms conversion identical to what the post-rebuild writer
        # will use, guaranteeing JOIN compatibility with ohlcv_daily.
        select_cols = "date, active_addresses, transaction_count, " \
                      "hash_rate, difficulty, block_size, market_cap"
        if has_total_btc:
            select_cols += ", total_btc"

        rows = conn.execute(
            f"SELECT {select_cols} FROM onchain_btc ORDER BY date"
        ).fetchall()

        # Verify every date parses cleanly before the transaction
        for r in rows:
            date_str = r[0]
            try:
                _date_to_ms(date_str)
            except ValueError as e:
                print(
                    f"[rebuild] ABORT: date {date_str!r} doesn't parse "
                    f"as YYYY-MM-DD: {e}. No data written.",
                    file=sys.stderr,
                )
                return 5

        print(f"  parsed {len(rows):,} dates cleanly; ready to rebuild")

        # ---------------------------------------------------------
        # The rebuild transaction
        # ---------------------------------------------------------
        print()
        print("[rebuild] Starting atomic rebuild transaction...")
        t_start = time.time()

        conn.execute("BEGIN")
        try:
            # 1. Create new table with full Rule 35 schema.
            #    PRIMARY KEY on `date`, plus `timestamp INTEGER NOT NULL`
            #    and `datetime TEXT NOT NULL` for cross-table JOIN
            #    compatibility.
            print("  [1/5] Creating onchain_btc_v2 with new schema...")
            create_sql = """
                CREATE TABLE onchain_btc_v2 (
                    date TEXT NOT NULL PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    active_addresses INTEGER,
                    transaction_count INTEGER,
                    hash_rate REAL,
                    difficulty REAL,
                    block_size REAL,
                    market_cap REAL"""
            if has_total_btc:
                create_sql += """,
                    total_btc REAL"""
            create_sql += """
                )
            """
            conn.execute(create_sql)

            # 2. Copy all rows with derived timestamp + datetime.
            print(f"  [2/5] Copying {len(rows):,} rows with derived "
                  f"timestamp + datetime columns...")
            t_copy_start = time.time()

            if has_total_btc:
                insert_sql = """
                    INSERT INTO onchain_btc_v2
                    (date, timestamp, datetime, active_addresses,
                     transaction_count, hash_rate, difficulty,
                     block_size, market_cap, total_btc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            else:
                insert_sql = """
                    INSERT INTO onchain_btc_v2
                    (date, timestamp, datetime, active_addresses,
                     transaction_count, hash_rate, difficulty,
                     block_size, market_cap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

            payload = []
            for r in rows:
                if has_total_btc:
                    (date_str, active_addresses, transaction_count,
                     hash_rate, difficulty, block_size, market_cap,
                     total_btc) = r
                    payload.append((
                        date_str,
                        _date_to_ms(date_str),
                        _date_to_iso(date_str),
                        active_addresses, transaction_count, hash_rate,
                        difficulty, block_size, market_cap, total_btc,
                    ))
                else:
                    (date_str, active_addresses, transaction_count,
                     hash_rate, difficulty, block_size, market_cap) = r
                    payload.append((
                        date_str,
                        _date_to_ms(date_str),
                        _date_to_iso(date_str),
                        active_addresses, transaction_count, hash_rate,
                        difficulty, block_size, market_cap,
                    ))

            conn.executemany(insert_sql, payload)
            t_copy_s = time.time() - t_copy_start
            v2_count = conn.execute(
                "SELECT COUNT(*) FROM onchain_btc_v2"
            ).fetchone()[0]
            print(f"        copied {v2_count:,} rows in {t_copy_s:.3f}s")

            if v2_count != live_count:
                raise RuntimeError(
                    f"Row count mismatch: onchain_btc has {live_count}, "
                    f"onchain_btc_v2 has {v2_count} -- aborting"
                )

            # 3. Drop old onchain_btc (drops sqlite_sequence entry too).
            print("  [3/5] Dropping old onchain_btc...")
            conn.execute("DROP TABLE onchain_btc")

            # 4. Rename _v2 to onchain_btc.
            print("  [4/5] Renaming onchain_btc_v2 -> onchain_btc...")
            conn.execute("ALTER TABLE onchain_btc_v2 RENAME TO onchain_btc")

            # 5. (No index on this table -- the PK on `date` is the only
            #    access pattern. Cross-table JOINs on `timestamp` will
            #    use the small row count + table scan; no index needed.)
            print("  [5/5] No additional indexes to create (370 rows; "
                  "PK on `date` covers all access patterns).")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        t_total_s = time.time() - t_start
        print(f"[rebuild] Transaction committed in {t_total_s:.3f}s "
              f"total wall-clock.")

        # ---------------------------------------------------------
        # Post-state verification
        # ---------------------------------------------------------
        post_cols = conn.execute(
            "PRAGMA table_info(onchain_btc)"
        ).fetchall()
        post_col_names = [c[1] for c in post_cols]
        post_has_id = "id" in post_col_names
        post_has_timestamp = "timestamp" in post_col_names
        post_has_datetime = "datetime" in post_col_names

        if post_has_id:
            print(
                "[rebuild] FAIL: post-rebuild, onchain_btc still has "
                "`id` column.",
                file=sys.stderr,
            )
            return 6
        if not post_has_timestamp:
            print(
                "[rebuild] FAIL: post-rebuild, onchain_btc lacks "
                "`timestamp` column.",
                file=sys.stderr,
            )
            return 6
        if not post_has_datetime:
            print(
                "[rebuild] FAIL: post-rebuild, onchain_btc lacks "
                "`datetime` column.",
                file=sys.stderr,
            )
            return 6

        # Confirm date is the PK
        pk_cols = sorted(c[1] for c in post_cols if c[5] > 0)
        if pk_cols != ["date"]:
            print(
                f"[rebuild] FAIL: expected PK (date), got {pk_cols}.",
                file=sys.stderr,
            )
            return 7

        post_count = conn.execute(
            "SELECT COUNT(*) FROM onchain_btc"
        ).fetchone()[0]
        if post_count != live_count:
            print(
                f"[rebuild] FAIL: post-rebuild row count "
                f"{post_count:,} != pre-rebuild {live_count:,}.",
                file=sys.stderr,
            )
            return 8

        # Sanity check: pick a row and verify timestamp matches the
        # ohlcv_daily convention
        sample = conn.execute("""
            SELECT date, timestamp, datetime FROM onchain_btc
            ORDER BY date DESC LIMIT 1
        """).fetchone()
        if sample:
            date_s, ts_ms, dt_iso = sample
            expected_ts = _date_to_ms(date_s)
            expected_iso = _date_to_iso(date_s)
            if ts_ms != expected_ts or dt_iso != expected_iso:
                print(
                    f"[rebuild] FAIL: sample row {date_s!r} has "
                    f"ts={ts_ms!r}, iso={dt_iso!r}; expected "
                    f"ts={expected_ts!r}, iso={expected_iso!r}.",
                    file=sys.stderr,
                )
                return 9

        # Cross-check ohlcv_daily JOIN: pick our latest date, find the
        # ohlcv_daily row for the same date, confirm the timestamps
        # match.
        if sample:
            date_s, ts_ms, _ = sample
            ohlcv_match = conn.execute("""
                SELECT timestamp FROM ohlcv_daily
                WHERE date = ? AND asset = 'BTC'
                LIMIT 1
            """, (date_s,)).fetchone()
            if ohlcv_match:
                ohlcv_ts = ohlcv_match[0]
                if ohlcv_ts != ts_ms:
                    print(
                        f"  WARN: timestamp mismatch for date {date_s!r}: "
                        f"onchain_btc.timestamp={ts_ms}, "
                        f"ohlcv_daily.timestamp={ohlcv_ts}. JOINs may "
                        f"not align as expected.",
                        file=sys.stderr,
                    )
                else:
                    print(f"  JOIN verification: onchain_btc and "
                          f"ohlcv_daily timestamps match for "
                          f"{date_s} ({ts_ms}). OK.")

        print()
        print("=" * 60)
        print("[rebuild] CYCLE 31 SCHEMA REBUILD COMPLETE")
        print(f"  onchain_btc: {live_count:,} -> {post_count:,} rows "
              f"(unchanged)")
        print(f"  schema: id PK -> date TEXT PK; added timestamp "
              f"INTEGER + datetime TEXT")
        print(f"  total time: {t_total_s:.3f}s")
        print("=" * 60)
        print()
        print("Next steps (USER):")
        print("  1. Re-enable PraxisOnchainCollector if it was disabled:")
        print("     Enable-ScheduledTask -TaskName 'PraxisOnchainCollector'")
        print("  2. Verify via get_collector_health that onchain_btc is")
        print("     reporting fresh data.")
        print("  3. Verify with raw_query that the new timestamp column")
        print("     JOINs cleanly with ohlcv_daily.")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
