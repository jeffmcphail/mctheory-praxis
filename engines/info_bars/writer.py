"""Info Bars writer: trade-table -> info_bars table.

Two public entrypoints:

- backfill_slice(...) -- one-shot scan of trades for a single
  (asset, bar_type, threshold_value). Used by the backfill script
  and by tests.

- live_update(...) -- enumerates DISTINCT (asset, bar_type,
  threshold_value) already present in info_bars and incrementally
  appends any newly-closed bars since each slice's
  MAX(end_timestamp). Used by the scheduled collector.

Trade-order convention: trades are pulled with
`ORDER BY timestamp ASC, trade_id ASC`. Same-ms ties resolve by
trade_id; the trades writer assigns monotonic trade_ids from
exchange feed order, so this is the correct tape order.

Idempotency: bar PK is (asset, bar_type, threshold_value, bar_index).
Re-inserting an existing (slice, bar_index) hits INSERT OR IGNORE
and is silently skipped. Re-running backfill on the same range
produces the same bars (deterministic given trade order).

Late-trade safety lag: live_update excludes trades with
timestamp >= now_ms - safety_lag_seconds*1000. Backfill does NOT
apply the lag (historical scan; late arrivals aren't a concern).

Exit-code policy (per memory #12): the __main__ block exits 0 if
all slices ran cleanly (regardless of new-bar counts) and
non-zero if any slice attempted writes but got 0 due to transient
errors.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from engines.info_bars.bars import Trade, build_for


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "data" / "crypto_data.db"

TRADE_CHUNK_SIZE = 100_000


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _max_end_ts(conn: sqlite3.Connection, asset: str,
                bar_type: str, threshold_value: float) -> Optional[int]:
    row = conn.execute(
        """
        SELECT MAX(end_timestamp) AS m, MAX(bar_index) AS bi
        FROM info_bars
        WHERE asset = ? AND bar_type = ? AND threshold_value = ?
        """,
        (asset, bar_type, threshold_value),
    ).fetchone()
    return (row["m"], row["bi"])


def _iter_trades(conn: sqlite3.Connection, asset: str,
                 start_exclusive_ms: Optional[int],
                 end_inclusive_ms: Optional[int]):
    """Yield Trade tuples in (timestamp ASC, trade_id ASC) order.

    start_exclusive_ms: only trades with timestamp > this (None = no lower bound)
    end_inclusive_ms: only trades with timestamp <= this  (None = no upper bound)
    """
    where = ["asset = ?"]
    params: list = [asset]
    if start_exclusive_ms is not None:
        where.append("timestamp > ?")
        params.append(start_exclusive_ms)
    if end_inclusive_ms is not None:
        where.append("timestamp <= ?")
        params.append(end_inclusive_ms)
    sql = (
        "SELECT trade_id, timestamp, price, amount, quote_amount, side "
        "FROM trades WHERE " + " AND ".join(where) +
        " ORDER BY timestamp ASC, trade_id ASC"
    )
    cur = conn.execute(sql, params)
    while True:
        rows = cur.fetchmany(TRADE_CHUNK_SIZE)
        if not rows:
            return
        for r in rows:
            yield Trade(
                trade_id=r["trade_id"],
                timestamp_ms=r["timestamp"],
                price=r["price"],
                amount=r["amount"],
                quote_amount=r["quote_amount"],
                side=r["side"],
            )


def _insert_bars(conn: sqlite3.Connection, bars: list, start_index: int) -> int:
    """Insert closed bars with monotonically increasing bar_index starting
    at start_index. Uses INSERT OR IGNORE so PK conflicts (re-runs) are
    silently skipped. Returns the rowcount inserted."""
    if not bars:
        return 0
    rows = []
    for offset, b in enumerate(bars):
        rows.append((
            b.asset, b.bar_type, b.threshold_value, start_index + offset,
            b.start_timestamp, b.end_timestamp,
            b.start_datetime, b.end_datetime,
            b.open, b.high, b.low, b.close,
            b.base_volume, b.quote_volume, b.tick_count,
            b.buy_quote, b.sell_quote, b.imbalance_quote,
        ))
    cur = conn.executemany(
        """
        INSERT OR IGNORE INTO info_bars (
            asset, bar_type, threshold_value, bar_index,
            start_timestamp, end_timestamp, start_datetime, end_datetime,
            open, high, low, close,
            base_volume, quote_volume, tick_count,
            buy_quote, sell_quote, imbalance_quote
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return cur.rowcount


def backfill_slice(
    db_path: str | Path,
    asset: str,
    bar_type: str,
    threshold_value: float,
    start_ts_ms: Optional[int] = None,
    end_ts_ms: Optional[int] = None,
    validate_only: bool = False,
) -> dict:
    """Scan trades for one slice; INSERT closed bars.

    start_ts_ms / end_ts_ms: optional inclusive bounds on trade
    timestamps. If start_ts_ms is None, resumes from the slice's
    last persisted end_timestamp (exclusive). If end_ts_ms is None,
    no upper bound.

    validate_only: if True, scan and count but do NOT INSERT.

    Returns a summary dict.
    """
    t0 = time.time()
    conn = _connect(db_path)
    try:
        existing_max_ts, existing_max_idx = _max_end_ts(
            conn, asset, bar_type, threshold_value)
        if start_ts_ms is None:
            start_exclusive = existing_max_ts  # None or int
        else:
            start_exclusive = start_ts_ms - 1

        next_bar_index = (existing_max_idx + 1) if existing_max_idx is not None else 0

        builder = build_for(bar_type, asset, threshold_value)
        trades_processed = 0
        closed_bars: list = []
        first_bar_start: Optional[int] = None
        last_bar_end: Optional[int] = None
        max_seen_ts: Optional[int] = None

        for trade in _iter_trades(conn, asset, start_exclusive, end_ts_ms):
            trades_processed += 1
            max_seen_ts = trade.timestamp_ms
            bar = builder.push(trade)
            if bar is not None:
                closed_bars.append(bar)
                if first_bar_start is None:
                    first_bar_start = bar.start_timestamp
                last_bar_end = bar.end_timestamp

        if validate_only:
            inserted = 0
        else:
            try:
                conn.execute("BEGIN")
                inserted = _insert_bars(conn, closed_bars, next_bar_index)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        elapsed_s = time.time() - t0
        return {
            "slice": (asset, bar_type, threshold_value),
            "validate_only": validate_only,
            "trades_processed": trades_processed,
            "closed_bars": len(closed_bars),
            "inserted": inserted,
            "first_bar_start": first_bar_start,
            "last_bar_end": last_bar_end,
            "max_trade_ts": max_seen_ts,
            "next_bar_index": next_bar_index,
            "elapsed_s": round(elapsed_s, 2),
        }
    finally:
        conn.close()


def _distinct_slices(conn: sqlite3.Connection) -> list:
    rows = conn.execute(
        """
        SELECT DISTINCT asset, bar_type, threshold_value
        FROM info_bars
        ORDER BY asset, bar_type, threshold_value
        """
    ).fetchall()
    return [(r["asset"], r["bar_type"], r["threshold_value"]) for r in rows]


def live_update(
    db_path: str | Path = DEFAULT_DB,
    safety_lag_seconds: int = 30,
) -> dict:
    """For every DISTINCT slice in info_bars, append newly-closed bars.

    Reads trades in (last_end_ts, now - safety_lag) into the slice's
    builder, INSERTs any newly-closed bars. Skips slices that have
    zero rows in info_bars (not yet backfilled).

    Returns a summary dict with per-slice results and overall counts.
    """
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    upper_bound_ms = now_ms - safety_lag_seconds * 1000

    conn = _connect(db_path)
    try:
        slices = _distinct_slices(conn)
    finally:
        conn.close()

    if not slices:
        return {
            "slices_processed": 0,
            "slices_with_new_bars": 0,
            "total_new_bars": 0,
            "errors": [],
            "per_slice": [],
            "safety_lag_seconds": safety_lag_seconds,
            "upper_bound_ms": upper_bound_ms,
        }

    per_slice = []
    errors: list = []
    total_new = 0
    slices_with_new = 0

    for asset, bar_type, threshold_value in slices:
        try:
            summary = backfill_slice(
                db_path=db_path,
                asset=asset,
                bar_type=bar_type,
                threshold_value=threshold_value,
                start_ts_ms=None,           # resume from last end_ts
                end_ts_ms=upper_bound_ms,   # respect safety lag
                validate_only=False,
            )
            per_slice.append(summary)
            n_new = summary["inserted"]
            total_new += n_new
            if n_new > 0:
                slices_with_new += 1
            # Honest-exit-code condition: expected bars (closed) but
            # wrote 0 -> error. The closed_bars count is the number
            # the builder produced; if that's positive and inserted
            # is zero we have a silent failure.
            if summary["closed_bars"] > 0 and summary["inserted"] == 0:
                errors.append({
                    "slice": (asset, bar_type, threshold_value),
                    "error": (
                        f"expected {summary['closed_bars']} new bars "
                        f"but inserted 0"),
                })
        except Exception as e:
            errors.append({
                "slice": (asset, bar_type, threshold_value),
                "error": f"{type(e).__name__}: {e}",
            })

    return {
        "slices_processed": len(slices),
        "slices_with_new_bars": slices_with_new,
        "total_new_bars": total_new,
        "errors": errors,
        "per_slice": per_slice,
        "safety_lag_seconds": safety_lag_seconds,
        "upper_bound_ms": upper_bound_ms,
    }


def _print_live_summary(summary: dict) -> None:
    print(
        f"[info_bars.live] slices={summary['slices_processed']} "
        f"with_new={summary['slices_with_new_bars']} "
        f"total_new_bars={summary['total_new_bars']} "
        f"errors={len(summary['errors'])} "
        f"safety_lag={summary['safety_lag_seconds']}s"
    )
    for s in summary["per_slice"]:
        asset, bt, th = s["slice"]
        print(
            f"  {asset:<4} {bt:<6} th={th:<12g} "
            f"trades={s['trades_processed']:>8} "
            f"new_bars={s['inserted']:>4} "
            f"elapsed={s['elapsed_s']:>5}s"
        )
    for e in summary["errors"]:
        asset, bt, th = e["slice"]
        print(f"  ERR  {asset} {bt} th={th}: {e['error']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Info Bars writer -- live collector for scheduled task")
    parser.add_argument("--live", action="store_true",
                        help="Run live_update against the trades table.")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="Path to crypto_data.db")
    parser.add_argument("--safety-lag-seconds", type=int, default=30,
                        help="Exclude trades newer than now - this (default 30).")
    args = parser.parse_args(argv)

    if not args.live:
        parser.error("must specify --live (no other modes wired into main)")

    summary = live_update(
        db_path=args.db,
        safety_lag_seconds=args.safety_lag_seconds,
    )
    _print_live_summary(summary)

    # Honest exit code: 0 if all slices ran cleanly (regardless of
    # new-bar count); non-zero if any slice had an error or a
    # silent zero-insert with expected bars > 0.
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
