"""
engines/open_interest_collector.py -- Binance + Bybit open interest.

WHY THIS EXISTS
---------------
Regime class F declares five states but reaches only three without an open
interest series: the +/-2 states require abs(oi_change_7d) > 0.10 and
oi_change_7d pins to 0.0 when OI is absent. Cycle 61 T4 found that; Cycle 62A
T5 made the degradation announce itself. This collector is what makes it stop
being degraded.

WHY IT IS URGENT
----------------
Both venues wall off history, and the walls were measured, not assumed:

    Binance  temporal wall, ~30 days. startTime beyond it returns
             -1130 "parameter 'startTime' is invalid" at EVERY granularity
             (1h, 4h and 1d all fail identically at -400d).

    Bybit    ROW-COUNT wall, 200 rows -- not a date wall. Granularity
             therefore trades against reach, which is the useful part:
                 1h -> 200 rows ->   8.3 days
                 4h -> 200 rows ->  33.2 days
                 1d -> 200 rows -> 199.0 days (back to 2026-02-01)
             Cycle 61 saw "the same 200 rows from 2026-02-01" because it
             probed at daily granularity. The floor is real, but within it
             far more history is reachable than a single-granularity probe
             suggests.

Nothing before those walls is retrievable from either venue at any
granularity, ever. Every day without collection is a permanent hole. Hence:
seed once, as deep as each wall allows, then forward-only.

CADENCE: HOURLY -- AND WHY
--------------------------
The feature this feeds is oi_change_7d: OI now against OI ~7 days ago. Hourly
sampling gives 168 points across that window, roughly 24x finer than the
feature can use, so it is not the binding constraint on anything. It is chosen
over something coarser because:

  1. It matches Binance's finest history granularity, so seeded rows and
     live rows are the same kind of measurement rather than two regimes
     stitched together.
  2. It leaves headroom for any later intraday use without a second
     migration.
  3. It is cheap: 6 assets x 2 venues x 24 = 288 rows/day.

Anything finer buys nothing for a 7-day feature; anything coarser would make
the seed and the live series inhomogeneous.

NOTIONAL
--------
Binance's *history* endpoint returns both sumOpenInterest (base units) and
sumOpenInterestValue (quote notional); its current-value endpoint returns only
the former. Bybit returns base units only, at either endpoint. Both the seed
and the live poll therefore go through the history endpoint, so the two paths
produce identical fields and notional is captured wherever the venue offers
it. open_interest_value stays NULL where it does not -- recorded as absent
rather than back-filled with a guess.

EXIT CODES
    0  healthy
    1  hard failure (no venue reachable, DB error)
    2  ran, but wrote fewer rows than expected -- the silent-zero case

Usage:
    python -m engines.open_interest_collector collect
    python -m engines.open_interest_collector seed
    python -m engines.open_interest_collector report
    python -m engines.open_interest_collector validate
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import ccxt

from engines.collector_common import (
    assert_ms, ms_to_iso, now_ms, open_db, setup_logging,
)

logger = logging.getLogger("collector.open_interest")

COLLECTOR = "open_interest"

# The funding_rates universe: these are the assets whose funding series an
# oi_change_7d feature would actually join against.
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
DEFAULT_VENUES = ["binance", "bybit"]

# Seed plan per venue: (timeframe, days_back). Ordered coarse -> fine so the
# finer pass overwrites nothing but fills in denser points near the present.
SEED_PLAN = {
    # Binance's wall is temporal, so 1h for the full 30 days is both the
    # deepest and the densest option available. Paginated in 500-row pages.
    "binance": [("1h", 30)],
    # Bybit's wall is 200 rows, so each granularity buys a different reach.
    # All three are collected: 1d for depth, 4h for the middle, 1h for
    # recency.
    "bybit": [("1d", 199), ("4h", 33), ("1h", 8)],
}

INSERT_SQL = """
    INSERT OR IGNORE INTO open_interest
        (asset, venue, timestamp, datetime, open_interest,
         open_interest_value, symbol, source, ingested_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

DAY_MS = 86_400_000


def make_exchange(venue: str):
    cls = getattr(ccxt, venue)
    return cls({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def perp_symbol(asset: str) -> str:
    return "%s/USDT:USDT" % asset


def _extract(entry: dict) -> tuple[float | None, float | None]:
    """Pull (base_amount, quote_notional) out of a ccxt OI entry.

    ccxt's unified openInterestValue is None on both venues here, so the raw
    `info` payload is consulted -- but only for keys the venue documents, and
    a missing notional stays None rather than being synthesised from price.
    """
    amount = entry.get("openInterestAmount")
    value = entry.get("openInterestValue")
    info = entry.get("info") or {}

    if amount is None:
        for k in ("sumOpenInterest", "openInterest"):
            if info.get(k) is not None:
                amount = float(info[k])
                break
    if value is None:
        for k in ("sumOpenInterestValue", "openInterestValue"):
            if info.get(k) is not None:
                value = float(info[k])
                break

    return (float(amount) if amount is not None else None,
            float(value) if value is not None else None)


def store(conn, asset: str, venue: str, symbol: str, source: str,
          entries: list) -> int:
    """Insert OI entries; return the number of rows actually written."""
    rows = []
    ingested = now_ms()
    for e in entries:
        ts = e.get("timestamp")
        if ts is None:
            continue
        try:
            ts = assert_ms(ts, "open_interest.timestamp")
        except ValueError as err:
            logger.warning("%s/%s rejected entry: %s", asset, venue, err)
            continue
        amount, value = _extract(e)
        if amount is None:
            continue
        rows.append((asset, venue, ts, ms_to_iso(ts), amount, value, symbol,
                     source, ingested))
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(INSERT_SQL, rows)
    return conn.total_changes - before


def fetch_history(ex, symbol: str, timeframe: str, since: int | None,
                  limit: int = 500) -> list:
    return ex.fetch_open_interest_history(symbol, timeframe, since=since,
                                          limit=limit)


# ---------------------------------------------------------------- collect ---

def cmd_collect(args) -> int:
    """One poll per asset x venue. Scheduled hourly."""
    conn = open_db()
    written_total = 0
    attempted = 0
    failures = []
    try:
        for venue in args.venues:
            try:
                ex = make_exchange(venue)
            except Exception as e:
                logger.error("venue %s unavailable: %s", venue, e)
                failures.append("%s (exchange init)" % venue)
                continue
            for asset in args.assets:
                attempted += 1
                symbol = perp_symbol(asset)
                try:
                    # limit is small: the hourly poll only needs the most
                    # recent closed buckets. A few extra give free repair of
                    # any bucket a missed run left behind.
                    entries = fetch_history(ex, symbol, args.timeframe,
                                            since=None, limit=args.poll_limit)
                    n = store(conn, asset, venue, symbol,
                              "%s:%s" % (venue, args.timeframe), entries)
                    written_total += n
                    logger.info("%-8s %-5s %s -> %d new row(s) of %d fetched",
                                venue, asset, args.timeframe, n, len(entries))
                except Exception as e:
                    logger.warning("%s/%s failed: %s: %s", venue, asset,
                                   type(e).__name__, e)
                    failures.append("%s/%s" % (venue, asset))
                time.sleep(args.sleep)

        print("")
        print("=" * 66)
        print("OPEN INTEREST COLLECT")
        print("=" * 66)
        print("  venues x assets  : %d" % attempted)
        print("  rows written     : %d" % written_total)
        print("  failures         : %d %s"
              % (len(failures), ("(" + ", ".join(failures[:8]) + ")")
                 if failures else ""))

        if attempted and len(failures) == attempted:
            print("\n[FAIL] every venue/asset poll failed.", file=sys.stderr)
            return 1

        # The silent-zero guard, expressed as STALENESS rather than row count.
        #
        # A row count is the wrong test for this collector: immediately after a
        # seed, or on a re-run inside the same hourly bucket, 0 new rows is the
        # correct and healthy outcome -- the table is simply already current.
        # What actually separates "ran cleanly, nothing new" from "the feed is
        # dead" is whether the newest stored observation has kept up with the
        # clock. So that is what decides the exit code.
        limit_ms = int(args.max_staleness_hours * 3600_000)
        placeholders = ",".join("?" * len(args.venues))
        stale = []
        for venue, asset, newest in conn.execute(
            "SELECT venue, asset, MAX(timestamp) FROM open_interest "
            "WHERE venue IN (%s) GROUP BY venue, asset" % placeholders,
            args.venues
        ):
            if now_ms() - newest > limit_ms:
                stale.append((venue, asset, (now_ms() - newest) / 3600_000))

        missing = [
            (v, a) for v in args.venues for a in args.assets
            if conn.execute(
                "SELECT COUNT(*) FROM open_interest WHERE venue=? AND asset=?",
                (v, a)).fetchone()[0] == 0
        ]

        print("  stale series     : %d (older than %gh)"
              % (len(stale), args.max_staleness_hours))
        for venue, asset, age_h in stale:
            print("      %-8s %-5s newest observation is %.1fh old"
                  % (venue, asset, age_h))
        if missing:
            print("  series with no rows: %s"
                  % ", ".join("%s/%s" % m for m in missing))

        if stale or missing:
            print("\n[FAIL] %d stale and %d empty series. A dead feed and an "
                  "already-current table both write 0 rows, so freshness -- "
                  "not the row count -- is what decides the exit code."
                  % (len(stale), len(missing)), file=sys.stderr)
            return 2

        print("\n[OK] wrote %d new row(s); all %d series current within %gh."
              % (written_total, attempted, args.max_staleness_hours))
        return 0
    finally:
        conn.close()


# ------------------------------------------------------------------- seed ---

def cmd_seed(args) -> int:
    """Backfill as deep as each venue's wall allows. Run once."""
    conn = open_db()
    summary = []
    try:
        for venue in args.venues:
            plan = SEED_PLAN.get(venue, [(args.timeframe, 30)])
            try:
                ex = make_exchange(venue)
            except Exception as e:
                logger.error("venue %s unavailable: %s", venue, e)
                continue
            for timeframe, days in plan:
                for asset in args.assets:
                    symbol = perp_symbol(asset)
                    src = "%s:%s" % (venue, timeframe)
                    total = 0
                    try:
                        since = now_ms() - days * DAY_MS
                        if venue == "binance":
                            # Temporal wall: page forward from it.
                            cursor = since
                            seen_last = None
                            while True:
                                batch = fetch_history(ex, symbol, timeframe,
                                                      since=cursor, limit=500)
                                if not batch:
                                    break
                                total += store(conn, asset, venue, symbol,
                                               src, batch)
                                last = batch[-1]["timestamp"]
                                if seen_last is not None and last <= seen_last:
                                    break
                                seen_last = last
                                if len(batch) < 500 or last >= now_ms() - 3600_000:
                                    break
                                cursor = last + 1
                                time.sleep(args.sleep)
                        else:
                            # Row-count wall: one request already returns the
                            # deepest 200 rows available at this granularity.
                            batch = fetch_history(ex, symbol, timeframe,
                                                  since=since, limit=200)
                            total += store(conn, asset, venue, symbol, src,
                                           batch)
                    except Exception as e:
                        logger.warning("seed %s/%s %s failed: %s: %s", venue,
                                       asset, timeframe, type(e).__name__, e)
                    logger.info("seeded %-8s %-5s %-3s -> %d rows",
                                venue, asset, timeframe, total)
                    summary.append((venue, asset, timeframe, total))
                    time.sleep(args.sleep)

        print("")
        print("=" * 66)
        print("OPEN INTEREST SEED")
        print("=" * 66)
        for venue, asset, tf, n in summary:
            print("  %-8s %-5s %-3s  %6d rows" % (venue, asset, tf, n))
        print("  %-8s %-5s %-3s  %6d rows TOTAL"
              % ("", "", "", sum(s[3] for s in summary)))

        print("\n  -- SEED BOUNDARY (where backfill ends, live capture begins) --")
        for venue, floor, ceiling, n in conn.execute(
            "SELECT venue, MIN(timestamp), MAX(timestamp), COUNT(*) "
            "FROM open_interest GROUP BY venue ORDER BY venue"
        ):
            print("    %-8s %s -> %s  (%d rows)"
                  % (venue, ms_to_iso(floor), ms_to_iso(ceiling), n))
        return 0
    finally:
        conn.close()


# ----------------------------------------------------------------- report ---

def cmd_report(args) -> int:
    conn = open_db()
    try:
        print("=" * 66)
        print("OPEN INTEREST -- COVERAGE")
        print("=" * 66)
        total = conn.execute("SELECT COUNT(*) FROM open_interest").fetchone()[0]
        print("  total rows: %d" % total)
        if not total:
            print("  (empty)")
            return 0

        print("\n  -- per venue/asset --")
        print("    %-8s %-5s %-7s %-18s %-18s %s"
              % ("venue", "asset", "rows", "first", "last", "notional"))
        for venue, asset, n, lo, hi, nv in conn.execute(
            "SELECT venue, asset, COUNT(*), MIN(timestamp), MAX(timestamp), "
            "SUM(open_interest_value IS NOT NULL) FROM open_interest "
            "GROUP BY venue, asset ORDER BY venue, asset"
        ):
            print("    %-8s %-5s %-7d %-18s %-18s %d/%d"
                  % (venue, asset, n, ms_to_iso(lo)[:16], ms_to_iso(hi)[:16],
                     nv, n))

        print("\n  -- rows by source granularity --")
        for source, n, lo in conn.execute(
            "SELECT source, COUNT(*), MIN(timestamp) FROM open_interest "
            "GROUP BY source ORDER BY source"
        ):
            print("    %-14s %7d rows, reaching back to %s"
                  % (source, n, ms_to_iso(lo)[:16]))

        # The number the retro has to carry.
        print("\n  -- SEED BOUNDARY --")
        for venue, lo in conn.execute(
            "SELECT venue, MIN(timestamp) FROM open_interest GROUP BY venue"
        ):
            print("    %-8s earliest observation: %s" % (venue, ms_to_iso(lo)))
        return 0
    finally:
        conn.close()


# --------------------------------------------------------------- validate ---

def cmd_validate(args) -> int:
    conn = open_db()
    failures = []
    try:
        print("=" * 66)
        print("OPEN INTEREST -- VALIDATION")
        print("=" * 66)

        def check(name, sql, want_zero=True):
            n = conn.execute(sql).fetchone()[0]
            ok = (n == 0) if want_zero else (n > 0)
            print("  [%s] %s: %d" % ("OK  " if ok else "FAIL", name, n))
            if not ok:
                failures.append(name)
            return n

        check("rows present", "SELECT COUNT(*) FROM open_interest",
              want_zero=False)
        check("timestamps outside epoch-ms range",
              "SELECT COUNT(*) FROM open_interest WHERE timestamp < 1577836800000 "
              "OR timestamp > 4102444800000")
        check("timestamps in the future",
              "SELECT COUNT(*) FROM open_interest WHERE timestamp > %d"
              % (now_ms() + 3600_000))
        check("non-positive open_interest",
              "SELECT COUNT(*) FROM open_interest WHERE open_interest <= 0")
        check("venue not set",
              "SELECT COUNT(*) FROM open_interest "
              "WHERE venue IS NULL OR venue = ''")
        check("source not set",
              "SELECT COUNT(*) FROM open_interest "
              "WHERE source IS NULL OR source = ''")

        sample = list(conn.execute(
            "SELECT timestamp, datetime FROM open_interest "
            "ORDER BY RANDOM() LIMIT 200"))
        bad = [(t, d) for t, d in sample if ms_to_iso(t) != d]
        ok = not bad
        print("  [%s] datetime cache matches timestamp (sampled %d): "
              "%d mismatches"
              % ("OK  " if ok else "FAIL", len(sample), len(bad)))
        if bad:
            failures.append("datetime cache")

        # A 7d OI change needs >= 7 days of span to be computable at all.
        print("\n  -- oi_change_7d computability per venue/asset --")
        for venue, asset, lo, hi, n in conn.execute(
            "SELECT venue, asset, MIN(timestamp), MAX(timestamp), COUNT(*) "
            "FROM open_interest GROUP BY venue, asset ORDER BY venue, asset"
        ):
            span_d = (hi - lo) / DAY_MS
            ok = span_d >= 7.0
            print("    [%s] %-8s %-5s span=%.1fd rows=%d"
                  % ("OK  " if ok else "THIN", venue, asset, span_d, n))

        print("")
        if failures:
            print("[FAIL] %d check(s) failed: %s"
                  % (len(failures), ", ".join(failures)), file=sys.stderr)
            return 1
        print("[OK] all validation checks passed.")
        return 0
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(description="Open interest collector.")
    p.add_argument("--verbose", type=int, default=3, choices=[0, 1, 2, 3],
                   help="0=ERROR 1=WARNING 2=INFO 3=DEBUG (default 3, maximum)")
    p.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS,
                   help="Assets to poll (default: the funding_rates universe)")
    p.add_argument("--venues", nargs="+", default=DEFAULT_VENUES,
                   choices=["binance", "bybit"])
    p.add_argument("--timeframe", default="1h",
                   help="Sampling granularity (default 1h; see module "
                        "docstring for why hourly)")
    p.add_argument("--sleep", type=float, default=0.35,
                   help="Seconds between venue calls (default 0.35)")

    subs = p.add_subparsers(dest="command", required=True)

    pc = subs.add_parser("collect", help="Poll current OI (scheduled hourly)")
    pc.add_argument("--poll-limit", type=int, default=6,
                    help="Recent buckets fetched per poll; >1 self-repairs a "
                         "missed run (default 6)")
    pc.add_argument("--max-staleness-hours", type=float, default=3.0,
                    help="A series whose newest observation is older than this "
                         "fails the run (default 3.0). This, not the row "
                         "count, is the silent-zero guard -- see cmd_collect.")

    subs.add_parser("seed", help="One-time backfill to each venue's wall")
    subs.add_parser("report", help="Coverage and seed boundary")
    subs.add_parser("validate", help="Structural checks on stored rows")

    args = p.parse_args()
    setup_logging(args.verbose)
    args.assets = [a.upper() for a in args.assets]

    if args.command == "collect":
        return cmd_collect(args)
    if args.command == "seed":
        return cmd_seed(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "validate":
        return cmd_validate(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
