"""
engines/unlock_market_data_collector.py -- market_data for the unlock universe.

Cycle 62A T3, collection half. `engines/unlock_universe.py` decides WHICH
assets; this collects them.

WHY IT IS A SEPARATE COLLECTOR
------------------------------
`crypto_data_collector.collect_market_data` keys off the hard-coded
SUPPORTED_ASSETS dict, which several scheduled services iterate with
`--asset all`. Widening that dict to 25 unlock assets would have widened the
market_data, OHLCV and funding services along with it -- a 4x CoinGecko call
rate and a behaviour change to collectors the brief says to leave alone. This
writes the SAME `market_data` table with the same columns and the same Rule 35
timestamps, driven by config/unlock_universe.json instead.

VERIFICATION IS PART OF COLLECTION, NOT A SEPARATE PASS
-------------------------------------------------------
The brief requires that every added asset expose BOTH circulating_supply and
total_supply, verified per asset rather than assumed. That check happens here,
against /coins/{id} -- the exact endpoint and the exact response the collector
stores from. A separate verify pass would double an already rate-limited API
budget to check a different response than the one being written, which is
weaker evidence for twice the cost.

An asset missing either field is NOT stored and is reported as failing F1.
Storing it with a 0 or a synthesised value is how a supply series becomes
quietly meaningless -- the Cycle 57 basis-blind failure mode.

Usage:
    python -m engines.unlock_market_data_collector collect
    python -m engines.unlock_market_data_collector report
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

from engines.collector_common import now_ms, open_db, setup_logging
from engines.unlock_universe import _get, load_universe

logger = logging.getLogger("collector.unlock_market_data")

INSERT_SQL = """
    INSERT OR REPLACE INTO market_data
        (asset, timestamp, date, market_cap, total_volume,
         circulating_supply, total_supply, ath, ath_change_pct, btc_dominance)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def utc_midnight_ms() -> tuple[int, str]:
    """market_data is a daily series: Rule 35.1 says date-only data converts to
    midnight-UTC milliseconds. Matches what collect_market_data already does."""
    d = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                           microsecond=0)
    return int(d.timestamp() * 1000), d.strftime("%Y-%m-%d")


def cmd_collect(args) -> int:
    universe = load_universe()
    assets = universe["assets"]
    ts_ms, date_str = utc_midnight_ms()

    print("=" * 74)
    print("UNLOCK UNIVERSE -- MARKET DATA COLLECTION")
    print("=" * 74)
    print("  universe generated: %s" % universe["generated_at"])
    print("  assets            : %d" % len(assets))
    print("  as-of (UTC midnight): %s  (%d)" % (date_str, ts_ms))
    print("  supply fields are verified per asset against the SAME response")
    print("  that gets stored -- an asset missing either is NOT stored.")
    print("")

    conn = open_db()
    stored, rejected, errored = [], [], []
    try:
        for a in assets:
            asset, cg_id = a["asset"], a["coingecko_id"]
            try:
                d = _get("coins/%s" % cg_id, {
                    "localization": "false", "tickers": "false",
                    "community_data": "false", "developer_data": "false",
                })
            except Exception as e:
                logger.warning("%s lookup failed: %s", asset, e)
                print("  [ERR ] %-8s %s" % (asset, str(e)[:60]))
                errored.append(asset)
                time.sleep(args.sleep)
                continue

            md = d.get("market_data") or {}
            circ = md.get("circulating_supply")
            total = md.get("total_supply")

            # The brief's hard requirement, checked not assumed.
            if not circ or not total or circ <= 0 or total <= 0:
                print("  [FAIL] %-8s circ=%-14s total=%-14s -- does not serve "
                      "F1, NOT stored" % (asset, circ, total))
                rejected.append(asset)
                time.sleep(args.sleep)
                continue

            conn.execute(INSERT_SQL, (
                asset, ts_ms, date_str,
                (md.get("market_cap") or {}).get("usd", 0),
                (md.get("total_volume") or {}).get("usd", 0),
                circ, total,
                (md.get("ath") or {}).get("usd", 0),
                (md.get("ath_change_percentage") or {}).get("usd", 0),
                None,   # btc_dominance is a global, owned by the base collector
            ))
            print("  [OK  ] %-8s circ=%-16.0f total=%-16.0f float=%.3f"
                  % (asset, circ, total, circ / total))
            stored.append(asset)
            time.sleep(args.sleep)

        print("")
        print("  stored   : %d  %s" % (len(stored), " ".join(stored)))
        if rejected:
            print("  REJECTED : %d  %s (missing a supply field)"
                  % (len(rejected), " ".join(rejected)))
        if errored:
            print("  errored  : %d  %s" % (len(errored), " ".join(errored)))

        n_assets = conn.execute(
            "SELECT COUNT(DISTINCT asset) FROM market_data").fetchone()[0]
        print("  distinct assets now in market_data: %d" % n_assets)

        if not stored:
            print("\n[FAIL] no asset was stored.", file=sys.stderr)
            return 1
        if len(stored) < args.min_stored:
            print("\n[FAIL] stored %d assets, below the required %d. F1 needs "
                  "an unlock-bearing universe, and a short one is reported "
                  "rather than quietly accepted."
                  % (len(stored), args.min_stored), file=sys.stderr)
            return 2

        print("\n[OK] %d unlock-bearing assets stored, every one carrying both "
              "circulating_supply and total_supply." % len(stored))
        return 0
    finally:
        conn.commit()
        conn.close()


def cmd_report(args) -> int:
    conn = open_db()
    try:
        print("=" * 74)
        print("MARKET_DATA -- COVERAGE")
        print("=" * 74)
        total = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        n = conn.execute(
            "SELECT COUNT(DISTINCT asset) FROM market_data").fetchone()[0]
        print("  rows: %d across %d assets\n" % (total, n))
        print("  %-8s %-6s %-12s %-12s %s"
              % ("asset", "rows", "first", "last", "latest float"))
        for asset, cnt, lo, hi in conn.execute(
            "SELECT asset, COUNT(*), MIN(date), MAX(date) FROM market_data "
            "GROUP BY asset ORDER BY asset"
        ):
            row = conn.execute(
                "SELECT circulating_supply, total_supply FROM market_data "
                "WHERE asset=? ORDER BY timestamp DESC LIMIT 1",
                (asset,)).fetchone()
            ratio = ("%.3f" % (row[0] / row[1])) if row and row[0] and row[1] else "n/a"
            print("  %-8s %-6d %-12s %-12s %s" % (asset, cnt, lo, hi, ratio))
        return 0
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(
        description="market_data collection for the unlock universe.")
    p.add_argument("--verbose", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--sleep", type=float, default=6.5,
                   help="Seconds between CoinGecko calls (free tier; "
                        "default 6.5)")
    subs = p.add_subparsers(dest="command", required=True)

    c = subs.add_parser("collect", help="Collect + verify the unlock universe")
    c.add_argument("--min-stored", type=int, default=20,
                   help="Fail if fewer than this many assets stored "
                        "(default 20, the brief's floor)")

    subs.add_parser("report", help="market_data coverage")

    args = p.parse_args()
    setup_logging(args.verbose)

    if args.command == "collect":
        return cmd_collect(args)
    if args.command == "report":
        return cmd_report(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
