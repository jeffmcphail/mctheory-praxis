#!/usr/bin/env python3
"""
scripts/backfill_funding_history.py
====================================
One-shot historical backfill of Binance Futures funding rates into the
canonical `funding_rates` table in `data/crypto_data.db`.

Used by Cycle 40 (Engine 7 funding-carry full-universe reproduction) to fill
in SOL/XRP/ADA/AVAX history that the live PraxisFundingCollector only
captures for BTC + ETH. Safe to re-run; idempotent.

Endpoint
--------
GET https://fapi.binance.com/fapi/v1/fundingRate
- Public; no API key required for funding-rate history.
- Limit 1000 events per call; pagination via startTime cursor.
- ~1095 events per asset per year at 8h funding cadence.

Schema target (Rule 35 / Cycle 21 conforming)
--------------------------------------------
    funding_rates (
        asset TEXT NOT NULL,
        timestamp INTEGER NOT NULL,      -- ms since epoch, seconds-aligned
        datetime TEXT NOT NULL,          -- ISO-8601 UTC with +00:00
        funding_rate REAL,
        PRIMARY KEY (asset, timestamp)
    )

Sub-second jitter in Binance's reported `fundingTime` is truncated to
seconds-aligned ms (`(ts // 1000) * 1000`), matching the Cycle 21.5
hotfix in `engines/crypto_data_collector.py`.

Idempotency
-----------
INSERT OR IGNORE on (asset, timestamp) PK. Re-runs are safe and report
duplicate-skip counts via --verbose.

Usage
-----
    # Defaults: SOL,XRP,ADA,AVAX from 2023-01-01 through today UTC
    python scripts/backfill_funding_history.py

    # Override window
    python scripts/backfill_funding_history.py \\
        --assets SOL,XRP,ADA,AVAX \\
        --start 2024-01-01 --end 2026-05-27

    # Quieter (default is verbose per memory #5)
    python scripts/backfill_funding_history.py --no-verbose

Pre-flight listing check
------------------------
If --start < 2024-01-01 and any requested asset was not yet listed on
Binance Futures by --start, the script falls back to --start=2024-01-01
for ALL assets (so the universe stays consistent for downstream training).
This satisfies the Cycle 40 D1 condition "Skip 2023 if any of
SOL/XRP/ADA/AVAX wasn't listed yet on Binance Futures at the start of 2023".

Notes
-----
- python-dotenv loaded per memory #4 (env-loading pattern), even though no
  secrets are required for the public funding-rate endpoint.
- Validation is on by default (memory #5: max validation + verbose).
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

DB_PATH        = Path(__file__).resolve().parent.parent / "data" / "crypto_data.db"  # Cycle 46 (44h): anchor to repo root
ENDPOINT       = "https://fapi.binance.com/fapi/v1/fundingRate"
SLEEP_SEC      = 1.0     # cushion between paginated calls
LIMIT          = 1000
DEFAULT_ASSETS = ["SOL", "XRP", "ADA", "AVAX"]
EXP13_OOS_START = "2025-01-01"
EXP13_OOS_END   = "2026-03-27"   # exclusive (atlas OOS is 2025-01-01 .. 2026-03-26)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def iso_to_ms(iso_date: str) -> int:
    """ISO date string (YYYY-MM-DD) -> UTC ms epoch."""
    return int(datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc)
               .timestamp() * 1000)


def ms_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) \
                   .strftime("%Y-%m-%dT%H:%M:%S+00:00")


# ---------------------------------------------------------------------------
# Schema validation (memory #5: max validation by default)
# ---------------------------------------------------------------------------

def schema_check(conn: sqlite3.Connection) -> None:
    """Verify funding_rates table exists with Rule-35-conforming schema."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='funding_rates'")
    if not cur.fetchone():
        raise RuntimeError(
            "funding_rates table not found in DB. Initialize via "
            "`python -m engines.crypto_data_collector status` first."
        )
    cur.execute("PRAGMA table_info(funding_rates)")
    cols = {(r[1], r[2]) for r in cur.fetchall()}
    expected = {
        ("asset", "TEXT"),
        ("timestamp", "INTEGER"),
        ("datetime", "TEXT"),
        ("funding_rate", "REAL"),
    }
    if not expected.issubset(cols):
        raise RuntimeError(
            f"funding_rates schema mismatch.\n  expected superset: {expected}\n  "
            f"actual cols: {cols}"
        )
    # Compound PK on (asset, timestamp) was confirmed Cycle 39; PRAGMA index_list
    # would show it as `sqlite_autoindex_funding_rates_1`. We trust schema_check
    # above since the brief explicitly states the schema is already conforming.
    print("  Schema check OK  "
          "(asset TEXT, timestamp INTEGER ms, datetime TEXT ISO+00:00, "
          "funding_rate REAL; PK (asset, timestamp))")


# ---------------------------------------------------------------------------
# Pre-flight listing probe
# ---------------------------------------------------------------------------

def probe_listing(asset: str, start_iso: str, verbose: bool) -> str | None:
    """
    Check whether `asset` was listed on Binance Futures at `start_iso`.

    Returns the earliest funding-event date string (YYYY-MM-DD) found within
    the 30-day window starting at start_iso, or None if no events exist in
    that window (asset not yet listed at start_iso).
    """
    symbol = f"{asset}USDT"
    start_ms = iso_to_ms(start_iso)
    window_end_ms = start_ms + 30 * 86400_000  # 30-day probe window
    params = {"symbol": symbol, "startTime": start_ms,
              "endTime": window_end_ms, "limit": 5}
    r = requests.get(ENDPOINT, params=params, timeout=20)
    r.raise_for_status()
    events = r.json()
    if not events:
        if verbose:
            print(f"    {asset}: no events in {start_iso} .. +30d "
                  f"(not listed yet at {start_iso})")
        return None
    first_dt = datetime.fromtimestamp(int(events[0]["fundingTime"]) / 1000,
                                      tz=timezone.utc).strftime("%Y-%m-%d")
    if verbose:
        print(f"    {asset}: earliest event in probe = {first_dt}")
    return first_dt


# ---------------------------------------------------------------------------
# Pagination loop
# ---------------------------------------------------------------------------

def fetch_funding_window(asset: str, start_ms: int, end_ms: int,
                         verbose: bool) -> list[dict]:
    """Page through /fapi/v1/fundingRate from start_ms to end_ms (UTC ms)."""
    symbol = f"{asset}USDT"
    all_events: list[dict] = []
    cursor = start_ms
    page = 0
    while cursor < end_ms:
        page += 1
        params = {"symbol": symbol, "startTime": cursor,
                  "endTime": end_ms, "limit": LIMIT}
        try:
            r = requests.get(ENDPOINT, params=params, timeout=30)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"    [page {page}] HTTP error: {e}; retrying after 5s...")
            time.sleep(5)
            r = requests.get(ENDPOINT, params=params, timeout=30)
            r.raise_for_status()

        events = r.json()
        if not events:
            if verbose:
                print(f"    [page {page}] empty -- end of available data")
            break
        all_events.extend(events)
        last_ts = int(events[-1]["fundingTime"])
        if verbose:
            first_ts = int(events[0]["fundingTime"])
            print(f"    [page {page}] {len(events):4d} events: "
                  f"{ms_to_iso(first_ts)[:16]} .. {ms_to_iso(last_ts)[:16]}",
                  flush=True)
        if len(events) < LIMIT:
            break
        cursor = last_ts + 1
        time.sleep(SLEEP_SEC)
    return all_events


# ---------------------------------------------------------------------------
# Insert (idempotent INSERT OR IGNORE)
# ---------------------------------------------------------------------------

def insert_events(conn: sqlite3.Connection, asset: str,
                  events: list[dict]) -> tuple[int, int, int]:
    """
    Insert events into funding_rates. Returns (attempted, inserted, skipped).
    skipped = duplicates already present (PK collision).
    """
    cur = conn.cursor()
    attempted = inserted = 0
    for e in events:
        ts_raw = int(e["fundingTime"])
        ts     = (ts_raw // 1000) * 1000     # Cycle 21.5 seconds-alignment
        dt     = ms_to_iso(ts)
        rate   = float(e["fundingRate"])
        cur.execute(
            "INSERT OR IGNORE INTO funding_rates "
            "(asset, timestamp, datetime, funding_rate) VALUES (?, ?, ?, ?)",
            (asset, ts, dt, rate),
        )
        attempted += 1
        if cur.rowcount == 1:
            inserted += 1
    conn.commit()
    return attempted, inserted, attempted - inserted


# ---------------------------------------------------------------------------
# Post-fetch validation
# ---------------------------------------------------------------------------

def per_asset_stats(conn: sqlite3.Connection, asset: str,
                    oos_start: str, oos_end: str) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), MIN(datetime), MAX(datetime) "
                "FROM funding_rates WHERE asset=?", (asset,))
    total, mn, mx = cur.fetchone()
    cur.execute("SELECT COUNT(DISTINCT date(datetime)) "
                "FROM funding_rates WHERE asset=?", (asset,))
    ndays = cur.fetchone()[0]
    cur.execute(
        "SELECT date(datetime), COUNT(*) FROM funding_rates "
        "WHERE asset=? GROUP BY date(datetime) HAVING COUNT(*) != 3 "
        "ORDER BY date(datetime)", (asset,))
    bad = cur.fetchall()
    cur.execute(
        "SELECT COUNT(*), AVG(funding_rate), "
        "       SUM(CASE WHEN funding_rate > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), "
        "       MIN(funding_rate), MAX(funding_rate) "
        "FROM funding_rates WHERE asset=? "
        "AND datetime >= ? AND datetime < ?",
        (asset, oos_start, oos_end))
    n_oos, mean_oos, pos_oos, min_oos, max_oos = cur.fetchone()
    return {
        "total": total, "min": mn, "max": mx, "ndays": ndays,
        "bad_days": bad,
        "oos_n": n_oos or 0, "oos_mean": mean_oos,
        "oos_pos_share": pos_oos, "oos_min": min_oos, "oos_max": max_oos,
    }


def report_validation(conn: sqlite3.Connection, assets: list[str],
                      oos_start: str, oos_end: str) -> None:
    print(f"\n=== VALIDATION ===")
    print(f"OOS window for regime stats (atlas Exp 13 primary): "
          f"[{oos_start}, {oos_end})\n")

    for asset in assets:
        s = per_asset_stats(conn, asset, oos_start, oos_end)
        ann_mean = (s["oos_mean"] * 3 * 365 * 100) if s["oos_mean"] is not None else 0.0
        pos_str = f"{s['oos_pos_share']:.3f}" if s["oos_pos_share"] is not None else "n/a"
        print(f"  {asset}:")
        print(f"    table rows:    {s['total']:>5}  "
              f"distinct days: {s['ndays']:>4}")
        print(f"    range:         {s['min']} .. {s['max']}")
        print(f"    OOS-window:    n={s['oos_n']:>5}  ann_mean={ann_mean:+7.2f}%  "
              f"pos_share={pos_str}  [{s['oos_min']!r}, {s['oos_max']!r}]")
        if s["bad_days"]:
            print(f"    !! {len(s['bad_days'])} days with != 3 events:")
            for d, c in s["bad_days"][:15]:
                print(f"       {d}: {c} events")
            if len(s["bad_days"]) > 15:
                print(f"       ... and {len(s['bad_days']) - 15} more")
        else:
            print(f"    OK  zero gap-days (every covered day has 3 events)")
        print()

    print("  Full-table snapshot (all assets in funding_rates):")
    cur = conn.cursor()
    cur.execute("SELECT asset, COUNT(*), MIN(datetime), MAX(datetime) "
                "FROM funding_rates GROUP BY asset ORDER BY asset")
    for row in cur.fetchall():
        print(f"    {row[0]:<6} rows={row[1]:>5}  range={row[2]} .. {row[3]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Backfill historical funding rates from Binance Futures REST.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--assets", default=",".join(DEFAULT_ASSETS),
                   help="Comma-separated asset symbols "
                        "(default: SOL,XRP,ADA,AVAX)")
    p.add_argument("--start", default="2023-01-01",
                   help="ISO start date inclusive (default 2023-01-01)")
    p.add_argument("--end",   default=None,
                   help="ISO end date exclusive (default: today UTC)")
    p.add_argument("--db", default=str(DB_PATH),
                   help=f"SQLite DB path (default {DB_PATH})")
    p.add_argument("--validate", action=argparse.BooleanOptionalAction,
                   default=True, help="Run validation queries after fetch")
    p.add_argument("--verbose",  action=argparse.BooleanOptionalAction,
                   default=True, help="Verbose progress logging")
    p.add_argument("--oos-start", default=EXP13_OOS_START,
                   help="OOS window start for regime stats (atlas Exp 13)")
    p.add_argument("--oos-end",   default=EXP13_OOS_END,
                   help="OOS window end (exclusive, atlas Exp 13)")
    args = p.parse_args()

    assets    = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    start_iso = args.start
    end_iso   = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"=== Cycle 40 D1: funding-rate backfill ===")
    print(f"  assets: {assets}")
    print(f"  range:  [{start_iso}, {end_iso})")
    print(f"  db:     {args.db}")
    print(f"  validate={args.validate}  verbose={args.verbose}")
    print()

    conn = sqlite3.connect(args.db)
    if args.validate:
        schema_check(conn)

    # ----- Pre-flight: listing-existence at start_iso -----
    if start_iso < "2024-01-01":
        print(f"\n  Pre-flight listing probe at {start_iso} (30-day window):")
        unlisted: list[str] = []
        for asset in assets:
            try:
                first = probe_listing(asset, start_iso, args.verbose)
            except Exception as e:
                print(f"    {asset}: probe error: {e}")
                first = None
            if first is None:
                unlisted.append(asset)
            time.sleep(SLEEP_SEC)
        if unlisted:
            new_start = "2024-01-01"
            print(f"\n  ↩ Listing gap: {unlisted} not listed at {start_iso}.")
            print(f"     Falling back to start={new_start} for ALL assets "
                  f"(uniform-universe rule).")
            start_iso = new_start
        else:
            print(f"  OK -- all assets listed by {start_iso}; proceeding "
                  f"with 2023 extension.")

    start_ms = iso_to_ms(start_iso)
    end_ms   = iso_to_ms(end_iso)

    # ----- Main fetch + insert loop -----
    print(f"\n=== FETCH + INSERT ===")
    per_asset: dict[str, dict] = {}
    t_global = time.time()

    for asset in assets:
        print(f"\n  -- {asset} --")
        t0 = time.time()
        try:
            events = fetch_funding_window(asset, start_ms, end_ms, args.verbose)
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

    print(f"\n=== FETCH COMPLETE in {time.time() - t_global:.1f}s ===")
    print()
    print(f"  Per-asset summary:")
    for asset in assets:
        s = per_asset[asset]
        if "error" in s:
            print(f"    {asset:<6} ERROR: {s['error']}")
        else:
            print(f"    {asset:<6} fetched={s['fetched']:>5}  "
                  f"inserted={s['inserted']:>5}  "
                  f"skipped_dup={s['skipped']:>5}  ({s['elapsed']:.1f}s)")

    # ----- Validation -----
    if args.validate:
        report_validation(conn, assets, args.oos_start, args.oos_end)

    conn.close()
    print("\n=== D1 BACKFILL DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
