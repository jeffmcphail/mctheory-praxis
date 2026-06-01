#!/usr/bin/env python3
"""
scripts/backfill_bybit_funding.py
==================================
One-shot historical backfill of Bybit linear-perp funding rates into the
canonical `funding_rates` table in `data/crypto_data.db`, written with
venue='bybit' (per Cycle 50 D1a schema extension).

Used by Cycle 50 (D1c) to seed Bybit history for the cross-venue
funding-spread strategy on the 6-asset atlas Exp 13 universe.
Safe to re-run; idempotent via the (asset, venue, timestamp) PK and
INSERT OR IGNORE.

Endpoint
--------
GET https://api.bybit.com/v5/market/funding/history
  category=linear, symbol={ASSET}USDT, endTime=<cursor>, limit=200

Returns newest-first; we paginate backward via the endTime cursor,
collect chronologically client-side, then INSERT in order.

Schema target (Cycle 50 / Rule 35 conforming)
---------------------------------------------
    funding_rates (
        asset TEXT NOT NULL,
        venue TEXT NOT NULL,           -- new in Cycle 50; PK component
        timestamp INTEGER NOT NULL,    -- ms since epoch, seconds-aligned
        datetime TEXT NOT NULL,        -- ISO+00:00
        funding_rate REAL,
        PRIMARY KEY (asset, venue, timestamp)
    )

Sub-second jitter in Bybit's reported fundingRateTimestamp is
truncated to seconds-aligned ms (matching the Cycle 21.5 hotfix for
the Binance collector path), so the seconds-aligned ms representation
is the same canonical thing across venues.

Usage
-----
    # Default: SOL,XRP,ADA,AVAX,BTC,ETH from 2023-01-01 through today
    python scripts/backfill_bybit_funding.py

    # Override window
    python scripts/backfill_bybit_funding.py \\
        --assets BTC,ETH --start 2024-01-01 --end 2026-06-01

Notes
-----
- python-dotenv loaded per memory #4 even though no auth is required
  for the public funding-history endpoint.
- Validation is on by default per memory #5; --no-validate to skip.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT      = Path(__file__).resolve().parent.parent
DB_PATH        = REPO_ROOT / "data" / "crypto_data.db"
ENDPOINT       = "https://api.bybit.com/v5/market/funding/history"
SLEEP_SEC      = 0.6
LIMIT          = 200
VENUE          = "bybit"
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
EXP13_OOS_START = "2025-01-01"
EXP13_OOS_END   = "2026-03-27"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def iso_to_ms(iso_date: str) -> int:
    return int(datetime.fromisoformat(iso_date)
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def ms_to_iso(ts_ms: int) -> str:
    return (datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%S+00:00"))


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def schema_check(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='funding_rates'")
    if not cur.fetchone():
        raise RuntimeError(
            "funding_rates table not found; init via "
            "`python -m engines.crypto_data_collector status` first."
        )
    cur.execute("PRAGMA table_info(funding_rates)")
    cols = {(r[1], r[2]) for r in cur.fetchall()}
    expected = {
        ("asset", "TEXT"), ("venue", "TEXT"),
        ("timestamp", "INTEGER"), ("datetime", "TEXT"),
        ("funding_rate", "REAL"),
    }
    if not expected.issubset(cols):
        raise RuntimeError(
            f"funding_rates schema mismatch.\n  expected superset: {expected}\n"
            f"  actual cols: {cols}\n  (Did you run the Cycle 50 migration?)"
        )
    print("  Schema check OK  "
          "(asset+venue+timestamp PK, Cycle 50 schema present)")


# ---------------------------------------------------------------------------
# Pagination loop (backward via endTime cursor)
# ---------------------------------------------------------------------------

def fetch_bybit_window(asset: str, start_ms: int, end_ms: int,
                      verbose: bool) -> list[dict]:
    symbol = f"{asset}USDT"
    out: list[dict] = []
    cursor_end = end_ms
    pages = 0
    MAX_PAGES = 250  # 250 pages * 200/page = 50000 events; far above any window we'd request
    while pages < MAX_PAGES:
        pages += 1
        try:
            r = requests.get(
                ENDPOINT,
                params={"category": "linear", "symbol": symbol,
                        "endTime": cursor_end, "limit": LIMIT},
                timeout=20,
            )
            r.raise_for_status()
            j = r.json()
        except requests.exceptions.RequestException as e:
            print(f"    [page {pages}] HTTP error: {e}; retrying after 5s...")
            time.sleep(5)
            r = requests.get(
                ENDPOINT,
                params={"category": "linear", "symbol": symbol,
                        "endTime": cursor_end, "limit": LIMIT},
                timeout=20,
            )
            r.raise_for_status()
            j = r.json()
        rows = j.get("result", {}).get("list", [])
        if not rows:
            if verbose:
                print(f"    [page {pages}] empty -- end of data")
            break
        # rows newest-first
        first_ts = int(rows[0]["fundingRateTimestamp"])
        last_ts  = int(rows[-1]["fundingRateTimestamp"])
        # keep rows that fall within our window
        kept = 0
        for row in rows:
            ts = int(row["fundingRateTimestamp"])
            if ts < start_ms or ts > end_ms:
                continue
            out.append({
                "timestamp": ts,
                "fundingRate": float(row["fundingRate"]),
            })
            kept += 1
        if verbose:
            print(f"    [page {pages:3d}] {len(rows):3d} rows kept={kept:3d} "
                  f"cursor: {ms_to_iso(last_ts)[:16]} .. "
                  f"{ms_to_iso(first_ts)[:16]}", flush=True)
        if last_ts <= start_ms:
            break
        cursor_end = last_ts - 1
        time.sleep(SLEEP_SEC)
    # Deduplicate + sort chronologically
    seen = set()
    deduped = []
    for r in sorted(out, key=lambda x: x["timestamp"]):
        if r["timestamp"] in seen:
            continue
        seen.add(r["timestamp"])
        deduped.append(r)
    return deduped


# ---------------------------------------------------------------------------
# Insert (idempotent)
# ---------------------------------------------------------------------------

def insert_events(conn: sqlite3.Connection, asset: str,
                  events: list[dict]) -> tuple[int, int, int]:
    cur = conn.cursor()
    attempted = inserted = 0
    for e in events:
        ts_raw = int(e["timestamp"])
        ts     = (ts_raw // 1000) * 1000   # Cycle 21.5 alignment
        dt     = ms_to_iso(ts)
        rate   = float(e["fundingRate"])
        cur.execute(
            "INSERT OR IGNORE INTO funding_rates "
            "(asset, venue, timestamp, datetime, funding_rate) "
            "VALUES (?, ?, ?, ?, ?)",
            (asset, VENUE, ts, dt, rate),
        )
        attempted += 1
        if cur.rowcount == 1:
            inserted += 1
    conn.commit()
    return attempted, inserted, attempted - inserted


# ---------------------------------------------------------------------------
# Post-fetch validation
# ---------------------------------------------------------------------------

def report_validation(conn: sqlite3.Connection, assets: list[str],
                      oos_start: str, oos_end: str) -> None:
    print(f"\n=== VALIDATION ===")
    print(f"OOS window for stats (atlas Exp 13 primary): "
          f"[{oos_start}, {oos_end})\n")
    cur = conn.cursor()
    for asset in assets:
        cur.execute(
            "SELECT COUNT(*), MIN(datetime), MAX(datetime) "
            "FROM funding_rates WHERE asset=? AND venue=?",
            (asset, VENUE),
        )
        total, mn, mx = cur.fetchone()
        cur.execute(
            "SELECT COUNT(DISTINCT date(datetime)) FROM funding_rates "
            "WHERE asset=? AND venue=?", (asset, VENUE),
        )
        ndays = cur.fetchone()[0]
        cur.execute(
            "SELECT date(datetime), COUNT(*) FROM funding_rates "
            "WHERE asset=? AND venue=? GROUP BY date(datetime) "
            "HAVING COUNT(*) != 3 ORDER BY date(datetime)",
            (asset, VENUE),
        )
        bad = cur.fetchall()
        cur.execute(
            "SELECT COUNT(*), AVG(funding_rate), "
            "       SUM(CASE WHEN funding_rate > 0 THEN 1 ELSE 0 END) "
            "         * 1.0 / COUNT(*) "
            "FROM funding_rates WHERE asset=? AND venue=? "
            "AND datetime >= ? AND datetime < ?",
            (asset, VENUE, oos_start, oos_end),
        )
        n_oos, mean_oos, pos_oos = cur.fetchone()
        ann = (mean_oos * 3 * 365 * 100) if mean_oos else 0.0
        print(f"  {asset}:")
        print(f"    rows: {total:>5}  distinct days: {ndays:>4}  "
              f"range: {mn} .. {mx}")
        pos_str = f"{pos_oos:.3f}" if pos_oos is not None else "n/a"
        print(f"    OOS-window: n={n_oos:>5}  ann_mean={ann:+7.2f}%  "
              f"pos_share={pos_str}")
        if bad:
            print(f"    !! {len(bad)} days with != 3 events:")
            for d, c in bad[:10]:
                print(f"       {d}: {c}")
            if len(bad) > 10:
                print(f"       ... and {len(bad) - 10} more")
        else:
            print(f"    OK  zero gap-days inside covered range")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", default=",".join(DEFAULT_ASSETS),
                   help=f"Comma-separated assets (default: {','.join(DEFAULT_ASSETS)})")
    p.add_argument("--start", default="2023-01-01",
                   help="ISO start date inclusive (default 2023-01-01)")
    p.add_argument("--end", default=None,
                   help="ISO end date exclusive (default: today UTC)")
    p.add_argument("--db", default=str(DB_PATH),
                   help=f"SQLite DB path (default {DB_PATH})")
    p.add_argument("--validate", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--verbose",  action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--oos-start", default=EXP13_OOS_START)
    p.add_argument("--oos-end",   default=EXP13_OOS_END)
    args = p.parse_args()

    assets    = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    start_iso = args.start
    end_iso   = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_ms  = iso_to_ms(start_iso)
    end_ms    = iso_to_ms(end_iso)

    print(f"=== Cycle 50 D1c: Bybit funding-rate backfill ===")
    print(f"  assets: {assets}")
    print(f"  range:  [{start_iso}, {end_iso})")
    print(f"  db:     {args.db}")
    print(f"  validate={args.validate}  verbose={args.verbose}")
    print()

    conn = sqlite3.connect(args.db)
    if args.validate:
        schema_check(conn)

    per_asset = {}
    t_start = time.time()
    for asset in assets:
        print(f"\n--- {asset} ---")
        t0 = time.time()
        try:
            events = fetch_bybit_window(asset, start_ms, end_ms, args.verbose)
        except Exception as e:
            print(f"  ERROR fetching {asset}: {e}")
            per_asset[asset] = {"fetched": 0, "inserted": 0, "skipped": 0,
                                "elapsed": time.time() - t0, "error": str(e)}
            continue
        attempted, inserted, dupes = insert_events(conn, asset, events)
        dt = time.time() - t0
        per_asset[asset] = {"fetched": len(events), "inserted": inserted,
                            "skipped": dupes, "elapsed": dt}
        print(f"  -> fetched={len(events)}  inserted={inserted}  "
              f"skipped_duplicate={dupes}  ({dt:.1f}s)")
        time.sleep(SLEEP_SEC)

    print(f"\n=== FETCH COMPLETE in {time.time() - t_start:.1f}s ===")
    print(f"  Per-asset summary:")
    for asset in assets:
        s = per_asset[asset]
        if "error" in s:
            print(f"    {asset:<6} ERROR: {s['error']}")
        else:
            print(f"    {asset:<6} fetched={s['fetched']:>5}  "
                  f"inserted={s['inserted']:>5}  "
                  f"skipped_dup={s['skipped']:>5}  ({s['elapsed']:.1f}s)")

    if args.validate:
        report_validation(conn, assets, args.oos_start, args.oos_end)

    conn.close()
    print("\n=== D1c DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
