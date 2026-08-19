"""
engines/liquidation_collector.py -- Binance forced-order (liquidation) stream.

WHY THIS EXISTS
---------------
Cycle 61 could not compute a false-positive rate for forced-trade scenario A2,
because a forced liquidation and a large discretionary market order are
identical in the `trades` tape. The Cycle 61 detector found real, clustered,
cross-asset-correlated sub-minute flow bursts (dispersion index 3.79-8.51,
cross-asset lift 8-12x) but with no book-stress signature and BELOW-chance
concentration on large-move days (lift 0.29 BTC / 0.47 ETH). Whether those
bursts are liquidations was unanswerable by construction.

Binance publishes forced orders on the !forceOrder@arr all-symbol WebSocket
stream. That converts A2 from inference to observation: the ground truth
arrives labelled.

WHAT IT GUARANTEES
------------------
1. Rule 35 storage: INTEGER ms UTC `timestamp` in the PK, derived ISO-8601
   `datetime` cache, every incoming timestamp range-checked (not trusted).
2. Append-only: rows are INSERTed, never UPDATEd or DELETEd. INSERT OR IGNORE
   makes a reconnect that overlaps a captured window idempotent instead of
   double-counting; suppressed duplicates are COUNTED and reported, so the
   suppression is visible rather than silent.
3. Gaps leave a trace. Every disconnect writes a `collector_gaps` row -- open
   on disconnect, closed on reconnect. A gap that leaves no trace is
   indistinguishable from a quiet market, and that table is how a later
   analysis tells those two apart.
4. Honest exit codes. A run that writes 0 rows in a window where rows were
   expected exits NON-ZERO. "Ran cleanly with N rows" and "ran cleanly with 0
   rows when N>0 was expected" are different outcomes and must not share an
   exit code.

EXIT CODES
    0  healthy -- wrote at least the expected number of rows
    1  hard failure -- never connected, or a DB/config error
    2  connected and ran, but wrote fewer rows than expected (the silent-zero
       case the brief requires be made loud)

SINGLE WRITER
-------------
One instance at a time. The scheduled .bat runs a bounded --duration and exits
before the next trigger, the same handoff pattern trades_collector uses to
dodge the MultipleInstances IgnoreNew race (Cycles 7-8, 10).

BLOCKED AS OF CYCLE 62A -- READ BEFORE SCHEDULING
-------------------------------------------------
This collector is correct and tested, but it currently receives NOTHING from
this host. Binance's futures WebSocket endpoint (fstream.binance.com) accepts
the connection and then never sends a frame. Measured, not inferred:

    Binance SPOT   wss://stream.binance.com:9443  btcusdt@aggTrade
                   -> 347 frames / 20s          DATA FLOWS
    Binance PERP   wss://fstream.binance.com     btcusdt@aggTrade
                   -> 0 frames / 20s            SILENT (handshake 101, open)
    Binance PERP   !markPrice@arr@1s (pushes unconditionally every second)
                   -> 0 frames / 20s            SILENT
    Binance PERP   !forceOrder@arr
                   -> 0 frames / 119s           SILENT
    Bybit   PERP   publicTrade.BTCUSDT
                   -> 271 frames / 20s          DATA FLOWS
    Hyperliquid    trades BTC
                   -> 96 frames / 20s           DATA FLOWS

Every fstream stream completes the WebSocket upgrade (HTTP 101), leaves the
socket open (state=1, no close code), and delivers zero bytes. Binance futures
REST (fapi.binance.com) works normally from the same host and process, which
is how the T2 open-interest collector still functions. Reproduced identically
inside and outside the tool sandbox, so it is not a harness artefact.

The signature -- upgrade accepted, stream silent, one venue's derivatives only
-- is consistent with Binance geo-restricting derivatives STREAMING, matching
the Canadian CEX-perp constraint Cycle 56 established. There is no same-venue
fallback: Binance withdrew the public GET /fapi/v1/allForceOrders REST
endpoint, so forced orders are stream-only.

Bybit's allLiquidation topic IS reachable from this host and carries the same
event class (129 liquidation events in 300s across BTC/ETH/SOL, against a
31,763-event publicTrade control on the same connection). That is a VENUE
SUBSTITUTION, not a drop-in: Bybit liquidations are not Binance liquidations,
and adopting them changes what scenario A2 is measured against. Deliberately
left as a decision rather than made silently -- the Cycle 57 basis-blind P&L
is what silent substitution costs.

Usage:
    python -m engines.liquidation_collector collect --duration 3550
    python -m engines.liquidation_collector report --hours 24
    python -m engines.liquidation_collector validate
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter

from engines.collector_common import (
    assert_ms, ms_to_iso, now_ms, open_db, record_gap, setup_logging,
)

logger = logging.getLogger("collector.liquidations")

VENUE = "binance"
COLLECTOR = "liquidations"
STREAM_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"

# Backoff schedule for reconnects (seconds), capped. Binance closes the
# connection every 24h by design, so reconnecting is routine, not exceptional.
BACKOFF_BASE = 1.0
BACKOFF_MAX = 60.0

# Rows expected per connected hour before a run is considered suspicious.
# Binance liquidates far more than this across all symbols -- the default is
# deliberately near-zero so the check fires only on a genuinely dead feed,
# never on a merely quiet one.
DEFAULT_MIN_ROWS_PER_HOUR = 1.0
# Runs shorter than this are too brief to draw any conclusion from.
DEFAULT_MIN_WINDOW_SECONDS = 300

INSERT_SQL = """
    INSERT OR IGNORE INTO liquidations
        (venue, symbol, timestamp, datetime, side, price, quantity,
         quote_qty, order_status, order_type, avg_price, event_time,
         ingested_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def parse_force_order(msg: dict) -> dict | None:
    """Binance forceOrder payload -> a storable row dict, or None if unusable.

    Payload shape (futures):
        {"e":"forceOrder","E":<event ms>,"o":{
            "s":symbol,"S":side,"o":type,"f":TIF,"q":orig qty,"p":price,
            "ap":avg price,"X":status,"l":last filled,"z":filled accum,
            "T":<trade ms>}}
    """
    o = msg.get("o")
    if not isinstance(o, dict):
        return None

    symbol = o.get("s")
    side = o.get("S")
    if not symbol or not side:
        return None

    # `T` is the order trade time and is the event's real instant; `E` is when
    # Binance emitted it. The former is canonical, the latter kept for latency
    # forensics. Both are range-checked -- Rule 35.4 says do not trust a feed.
    ts = assert_ms(o.get("T") or msg.get("E"), "liquidation.timestamp")
    event_time = msg.get("E")
    try:
        event_time = assert_ms(event_time, "liquidation.event_time")
    except ValueError:
        event_time = None

    price = float(o.get("p") or 0.0)
    quantity = float(o.get("q") or 0.0)
    avg_price = float(o.get("ap") or 0.0) or None
    filled = float(o.get("z") or 0.0)

    if price <= 0 or quantity <= 0:
        return None

    # Notional: prefer what actually filled at the price it actually filled
    # at; fall back to the order's own price x size.
    if avg_price and filled > 0:
        quote_qty = avg_price * filled
    else:
        quote_qty = price * quantity

    return {
        "venue": VENUE,
        "symbol": symbol,
        "timestamp": ts,
        "datetime": ms_to_iso(ts),
        "side": side,
        "price": price,
        "quantity": quantity,
        "quote_qty": quote_qty,
        "order_status": o.get("X"),
        "order_type": o.get("o"),
        "avg_price": avg_price,
        "event_time": event_time,
    }


def insert_rows(conn, rows):
    """Insert rows; return (written, suppressed_duplicates)."""
    if not rows:
        return 0, 0
    ingested = now_ms()
    payload = [
        (r["venue"], r["symbol"], r["timestamp"], r["datetime"], r["side"],
         r["price"], r["quantity"], r["quote_qty"], r["order_status"],
         r["order_type"], r["avg_price"], r["event_time"], ingested)
        for r in rows
    ]
    cur = conn.cursor()
    before = conn.total_changes
    cur.executemany(INSERT_SQL, payload)
    written = conn.total_changes - before
    return written, len(rows) - written


async def stream_once(conn, stats, deadline, batch_size):
    """One connected session. Returns a reason string when it ends."""
    from websockets.asyncio.client import connect

    buf = []

    def flush():
        if not buf:
            return
        written, dupes = insert_rows(conn, buf)
        stats["written"] += written
        stats["duplicates"] += dupes
        logger.info("flushed %d events (written=%d dupes=%d, run total=%d)",
                    len(buf), written, dupes, stats["written"])
        buf.clear()

    async with connect(STREAM_URL, ping_interval=20, ping_timeout=20,
                       max_queue=4096) as ws:
        connected_at = now_ms()
        stats["connects"] += 1
        logger.info("connected to %s", STREAM_URL)

        # A disconnect we already recorded now has an end: close it.
        if stats.get("open_gap_start") is not None:
            record_gap(conn, COLLECTOR, VENUE, stats["open_gap_start"],
                       connected_at, stats.get("open_gap_reason", "reconnect"))
            stats["open_gap_start"] = None
            stats["gaps_closed"] += 1

        try:
            while True:
                if deadline is not None:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        return "duration reached"
                    timeout = min(remaining, 30.0)
                else:
                    timeout = 30.0

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    # Idle tick, not an error: flush what we have and re-check
                    # the deadline. The stream is genuinely bursty.
                    flush()
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("undecodable frame skipped: %.120s", raw)
                    stats["bad_frames"] += 1
                    continue

                # The all-market stream is documented as an array feed but
                # delivers single objects in practice. Handle both rather
                # than depending on which.
                events = msg if isinstance(msg, list) else [msg]
                for ev in events:
                    if not isinstance(ev, dict) or ev.get("e") != "forceOrder":
                        continue
                    stats["received"] += 1
                    try:
                        row = parse_force_order(ev)
                    except ValueError as e:
                        logger.warning("rejected event: %s", e)
                        stats["rejected"] += 1
                        continue
                    if row is None:
                        stats["rejected"] += 1
                        continue
                    buf.append(row)
                    stats["symbols"][row["symbol"]] += 1
                    logger.debug("%s %s %s qty=%s px=%s notional=%.0f",
                                 row["datetime"], row["symbol"], row["side"],
                                 row["quantity"], row["price"],
                                 row["quote_qty"])

                if len(buf) >= batch_size:
                    flush()
        finally:
            flush()
            stats["connected_seconds"] += (now_ms() - connected_at) / 1000.0


async def run_collect(args):
    conn = open_db()
    stats = {
        "received": 0, "written": 0, "duplicates": 0, "rejected": 0,
        "bad_frames": 0, "connects": 0, "gaps_opened": 0, "gaps_closed": 0,
        "connected_seconds": 0.0, "symbols": Counter(),
        "open_gap_start": None, "open_gap_reason": None,
    }

    loop = asyncio.get_running_loop()
    deadline = (loop.time() + args.duration) if args.duration else None
    started_ms = now_ms()
    backoff = BACKOFF_BASE

    try:
        while True:
            if deadline is not None and loop.time() >= deadline:
                break
            try:
                reason = await stream_once(conn, stats, deadline,
                                           args.batch_size)
                backoff = BACKOFF_BASE
                if reason == "duration reached":
                    break
                # A clean close that was not our deadline is still a gap.
                stats["open_gap_start"] = now_ms()
                stats["open_gap_reason"] = reason
                record_gap(conn, COLLECTOR, VENUE, stats["open_gap_start"],
                           None, reason)
                stats["gaps_opened"] += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                reason = "%s: %s" % (type(e).__name__, e)
                logger.warning("stream error: %s", reason)
                stats["open_gap_start"] = now_ms()
                stats["open_gap_reason"] = reason
                record_gap(conn, COLLECTOR, VENUE, stats["open_gap_start"],
                           None, reason)
                stats["gaps_opened"] += 1

            if deadline is not None and loop.time() >= deadline:
                break
            logger.warning("reconnecting in %.1fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
    finally:
        # A still-open gap at exit is left open ON PURPOSE: the collector is
        # down and the feed really is unobserved until something restarts it.
        conn.close()

    return report_run(args, stats, started_ms)


def report_run(args, stats, started_ms):
    elapsed = (now_ms() - started_ms) / 1000.0
    connected = stats["connected_seconds"]
    n_written = stats["written"]

    print("")
    print("=" * 66)
    print("LIQUIDATION COLLECTOR RUN")
    print("=" * 66)
    print("  window             : %s -> %s"
          % (ms_to_iso(started_ms), ms_to_iso(now_ms())))
    print("  elapsed / connected: %.1fs / %.1fs" % (elapsed, connected))
    print("  connects           : %d" % stats["connects"])
    print("  gaps opened/closed : %d / %d"
          % (stats["gaps_opened"], stats["gaps_closed"]))
    print("  events received    : %d" % stats["received"])
    print("  rows written       : %d" % n_written)
    print("  duplicates ignored : %d" % stats["duplicates"])
    print("  rejected / bad     : %d / %d"
          % (stats["rejected"], stats["bad_frames"]))
    if stats["symbols"]:
        top = stats["symbols"].most_common(10)
        print("  distinct symbols   : %d" % len(stats["symbols"]))
        print("  top symbols        : "
              + ", ".join("%s=%d" % (s, n) for s, n in top))

    if stats["connects"] == 0:
        print("\n[FAIL] never established a connection.", file=sys.stderr)
        return 1

    # The silent-zero check. Expectation scales with time actually CONNECTED,
    # not wall-clock: a run that spent its window in backoff is not evidence
    # of a quiet market.
    if connected < args.min_window_seconds:
        print("\n[OK] ran %.0fs connected (below the %ds window needed to "
              "judge row counts); wrote %d rows."
              % (connected, args.min_window_seconds, n_written))
        return 0

    expected = max(1, int(args.min_rows_per_hour * connected / 3600.0))
    if n_written < expected:
        print("\n[FAIL] wrote %d rows in %.0fs connected; expected at least "
              "%d. A dead feed and a quiet market look identical from the row "
              "count alone, so this exits non-zero rather than reporting "
              "success." % (n_written, connected, expected), file=sys.stderr)
        return 2

    print("\n[OK] wrote %d rows (expected >= %d)." % (n_written, expected))
    return 0


# ---------------------------------------------------------------- report ---

def cmd_report(args):
    conn = open_db()
    try:
        cutoff = now_ms() - int(args.hours * 3600 * 1000)
        total = conn.execute(
            "SELECT COUNT(*) FROM liquidations WHERE timestamp >= ?",
            (cutoff,)).fetchone()[0]

        print("=" * 66)
        print("LIQUIDATIONS -- last %gh" % args.hours)
        print("=" * 66)

        span = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM liquidations"
        ).fetchone()
        if span[2]:
            print("  table span   : %s -> %s"
                  % (ms_to_iso(span[0]), ms_to_iso(span[1])))
        print("  table rows   : %d" % span[2])
        print("  window rows  : %d" % total)
        if not total:
            print("\n  (no rows in window)")
            return 0

        print("\n  -- side split --")
        for side, n, notional in conn.execute(
            "SELECT side, COUNT(*), SUM(quote_qty) FROM liquidations "
            "WHERE timestamp >= ? GROUP BY side ORDER BY 2 DESC", (cutoff,)
        ):
            print("    %-5s %7d  notional=%15.0f" % (side, n, notional or 0))

        print("\n  -- symbol distribution (top 15) --")
        nsym = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM liquidations "
            "WHERE timestamp >= ?", (cutoff,)).fetchone()[0]
        print("    distinct symbols: %d" % nsym)
        for sym, n, notional in conn.execute(
            "SELECT symbol, COUNT(*), SUM(quote_qty) FROM liquidations "
            "WHERE timestamp >= ? GROUP BY symbol ORDER BY 2 DESC LIMIT 15",
            (cutoff,)
        ):
            print("    %-14s %7d  notional=%15.0f" % (sym, n, notional or 0))

        print("\n  -- size distribution (USD notional) --")
        vals = [r[0] for r in conn.execute(
            "SELECT quote_qty FROM liquidations WHERE timestamp >= ? "
            "AND quote_qty IS NOT NULL ORDER BY quote_qty", (cutoff,))]
        if vals:
            def pct(p):
                return vals[min(len(vals) - 1, int(len(vals) * p))]
            print("    count  : %d" % len(vals))
            print("    sum    : %.0f" % sum(vals))
            print("    min    : %.2f" % vals[0])
            print("    p25    : %.2f" % pct(.25))
            print("    median : %.2f" % pct(.50))
            print("    p75    : %.2f" % pct(.75))
            print("    p95    : %.2f" % pct(.95))
            print("    p99    : %.2f" % pct(.99))
            print("    max    : %.2f" % vals[-1])

        gaps = list(conn.execute(
            "SELECT timestamp, gap_end, gap_seconds, reason "
            "FROM collector_gaps WHERE collector=? AND timestamp >= ? "
            "ORDER BY timestamp", (COLLECTOR, cutoff)))
        print("\n  -- collection gaps in window: %d --" % len(gaps))
        for ts, end, secs, reason in gaps[:20]:
            end_s = ms_to_iso(end) if end else "STILL OPEN"
            secs_s = ("%.1fs" % secs) if secs is not None else "open"
            print("    %s -> %s (%s) %s" % (ms_to_iso(ts), end_s, secs_s, reason))
        if len(gaps) > 20:
            print("    ... and %d more" % (len(gaps) - 20))
        return 0
    finally:
        conn.close()


# -------------------------------------------------------------- validate ---

def cmd_validate(args):
    """Structural checks on what is stored. Exits non-zero on any failure."""
    conn = open_db()
    failures = []
    try:
        print("=" * 66)
        print("LIQUIDATIONS -- VALIDATION")
        print("=" * 66)

        def check(name, sql, want_zero=True):
            n = conn.execute(sql).fetchone()[0]
            ok = (n == 0) if want_zero else (n > 0)
            print("  [%s] %s: %d" % ("OK  " if ok else "FAIL", name, n))
            if not ok:
                failures.append(name)
            return n

        check("rows present", "SELECT COUNT(*) FROM liquidations",
              want_zero=False)
        check("timestamps outside epoch-ms range",
              "SELECT COUNT(*) FROM liquidations WHERE timestamp < 1577836800000 "
              "OR timestamp > 4102444800000")
        check("timestamps in the future",
              "SELECT COUNT(*) FROM liquidations WHERE timestamp > %d"
              % (now_ms() + 60000))
        check("non-positive price",
              "SELECT COUNT(*) FROM liquidations WHERE price <= 0")
        check("non-positive quantity",
              "SELECT COUNT(*) FROM liquidations WHERE quantity <= 0")
        check("side outside BUY/SELL",
              "SELECT COUNT(*) FROM liquidations "
              "WHERE side NOT IN ('BUY','SELL')")
        check("datetime missing",
              "SELECT COUNT(*) FROM liquidations "
              "WHERE datetime IS NULL OR datetime = ''")
        check("venue not set",
              "SELECT COUNT(*) FROM liquidations "
              "WHERE venue IS NULL OR venue = ''")

        # Rule 35.3: datetime is a derived cache, so it must actually derive.
        sample = list(conn.execute(
            "SELECT timestamp, datetime FROM liquidations "
            "ORDER BY RANDOM() LIMIT 200"))
        bad = [(t, d) for t, d in sample if ms_to_iso(t) != d]
        ok = not bad
        print("  [%s] datetime cache matches timestamp (sampled %d): "
              "%d mismatches"
              % ("OK  " if ok else "FAIL", len(sample), len(bad)))
        if bad:
            failures.append("datetime cache")
            for t, d in bad[:5]:
                print("        %d stored=%s expected=%s" % (t, d, ms_to_iso(t)))

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
    p = argparse.ArgumentParser(
        description="Binance forced-order (liquidation) stream collector.")
    p.add_argument("--verbose", type=int, default=3, choices=[0, 1, 2, 3],
                   help="0=ERROR 1=WARNING 2=INFO 3=DEBUG (default 3, maximum)")
    subs = p.add_subparsers(dest="command", required=True)

    c = subs.add_parser("collect", help="Stream forced orders into the DB")
    c.add_argument("--duration", type=int, default=None,
                   help="Seconds to run before exiting (default: forever). "
                        "The scheduled task uses 3550 against an hourly "
                        "trigger so invocations never overlap.")
    c.add_argument("--batch-size", type=int, default=50,
                   help="Events buffered before a DB flush (default 50)")
    c.add_argument("--min-rows-per-hour", type=float,
                   default=DEFAULT_MIN_ROWS_PER_HOUR,
                   help="Below this rate the run exits non-zero (default %g)"
                        % DEFAULT_MIN_ROWS_PER_HOUR)
    c.add_argument("--min-window-seconds", type=int,
                   default=DEFAULT_MIN_WINDOW_SECONDS,
                   help="Connected seconds required before the row-count "
                        "check applies (default %d)"
                        % DEFAULT_MIN_WINDOW_SECONDS)

    r = subs.add_parser("report", help="Capture statistics over a window")
    r.add_argument("--hours", type=float, default=24.0)

    subs.add_parser("validate", help="Structural checks on stored rows")

    args = p.parse_args()
    setup_logging(args.verbose)

    if args.command == "collect":
        return asyncio.run(run_collect(args))
    if args.command == "report":
        return cmd_report(args)
    if args.command == "validate":
        return cmd_validate(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
