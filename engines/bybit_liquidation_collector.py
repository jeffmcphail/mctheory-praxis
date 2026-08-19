"""
engines/bybit_liquidation_collector.py -- Bybit allLiquidation stream (T1 venue).

WHY BYBIT, AND WHY NOW
----------------------
Cycle 61 could not compute a false-positive rate for forced-trade scenario A2
because a forced liquidation and a large discretionary market order are
identical in the `trades` tape. Labelled liquidations are what settle it.

Binance was the intended source and is unreachable from this host (see
engines/liquidation_collector.py, which stays in place, unscheduled). The
decisive point is not that Binance is blocked -- it is that NO BACKFILL EXISTS
ON ANY PATH:

  - Binance withdrew the public GET /fapi/v1/allForceOrders REST endpoint, so
    forced orders are stream-only.
  - data.binance.vision has NO USD-margined liquidation dataset at all (0
    keys), and its coin-margined one stops at 2024-10-14 while our `trades`
    coverage starts 2026-04-29 -- zero overlap, a ~18.5 month gap.

So this series can only ever be built forward, and every hour not recorded is
lost permanently. That asymmetry -- a reversible venue choice against an
irreversible data loss -- is why Bybit is adopted now rather than waiting for
Binance to clear.

THIS IS A VENUE SUBSTITUTION. COUNTS DO NOT TRANSFER.
-----------------------------------------------------
Bybit liquidations are not Binance liquidations. Three separate reasons, all
of which push event counts around, and none of which cancel:

  1. PERP MARKET SHARE. Binance carries substantially more perpetual open
     interest and volume than Bybit. A given market-wide cascade produces a
     different number of events, on different symbols, at different sizes, on
     each venue.

  2. LIQUIDATION ENGINE. Maintenance-margin tiers, ADL policy, partial-
     liquidation rules and insurance-fund behaviour all differ. The same
     trader in the same position is force-closed at a different price, in a
     different number of pieces, on the two venues.

  3. STREAM THROTTLE -- MEASURED, NOT ASSUMED. Binance's `!forceOrder@arr`
     publishes at most ONE order per symbol per second (the public archive
     inherits the same cap; Cycle 62A measured max distinct events per
     symbol-second = 1 in every archive file). Bybit's `allLiquidation`
     applies no such cap: a 180s probe on this host saw up to 18 distinct
     events in a single BTCUSDT second. Bybit is closer to a complete record,
     which is a data-quality gain, and simultaneously makes its raw event
     counts incomparable to any Binance-derived figure by a factor that
     varies with how bursty the moment was.

Consequence, stated plainly so it is not rediscovered later: any Binance-based
prior on liquidation event RATES -- thresholds, "events per minute" cutoffs,
burst-size percentiles -- does not carry over. Rates must be re-estimated on
Bybit data. What DOES carry over is the event CLASS: these are genuine forced
closures, which is the labelling scenario A2 needs. The original warning
against silent venue substitution stands; this module is that warning made
explicit rather than dropped now that we are proceeding anyway.

TWO FIELD-LEVEL TRAPS, HANDLED AT THE EDGE
-------------------------------------------
  SIDE IS INVERTED. Binance reports the ORDER side (SELL => a long was
  liquidated); Bybit reports the POSITION side (BUY => a long was liquidated).
  Same letters, opposite meaning, both legal values. Translation happens in
  engines/liquidation_common.normalise_side and the venue's own token is kept
  in `side_raw` so the translation stays auditable.

  PRICE IS A BANKRUPTCY PRICE. Bybit's `p` is the price at which the position's
  margin reaches zero, not a price anything traded at. Binance sends an actual
  average fill price. `quote_qty` from Bybit is therefore an approximation with
  a known directional bias, recorded as price_basis='bankruptcy'.

Bybit also sends no order status, order type or average fill price, so those
columns stay NULL for Bybit rows -- absent, not guessed.

COVERAGE IS A CHOSEN SUBSET
---------------------------
Binance offered one all-market topic. Bybit's `allLiquidation` is PER SYMBOL,
so coverage is whatever we subscribe to -- by default the six assets the
funding and open-interest series already cover, so liquidations join against
them without a universe mismatch. Symbols outside that list are simply not
observed, which is a coverage boundary, not a quiet market.

WHY A CONTROL TOPIC
-------------------
Cycle 62A lost time to this: an early Bybit probe subscribed to 13 topics in
one request, received `success: true`, and then received nothing -- Bybit ACKs
and silently drops a batch over the per-request arg limit. A success ACK is not
evidence of a working subscription. So this collector subscribes in small
batches AND keeps a low-rate control topic (a 1-minute kline) that MUST
produce data. The control is what separates "connection alive, market quiet"
from "subscription silently dead", which a liquidation count alone cannot do.

EXIT CODES
    0  healthy -- connected, control topic delivered, feed observed
    1  hard failure -- never connected, or a DB/config error
    2  connected but the feed cannot be trusted: the control topic delivered
       nothing (silently dead subscription), or liquidations were zero over a
       window long enough that zero is itself suspicious

SINGLE WRITER
-------------
One instance at a time. The scheduled .bat runs a bounded --duration and exits
before the next trigger, the same handoff pattern trades_collector uses to
dodge the MultipleInstances IgnoreNew race (Cycles 7-8, 10).

Usage:
    python -m engines.bybit_liquidation_collector collect --duration 3550
    python -m engines.bybit_liquidation_collector report --hours 24
    python -m engines.bybit_liquidation_collector validate
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
from engines.liquidation_common import (
    PRICE_BASIS, cmd_report, cmd_validate, normalise_side,
)

logger = logging.getLogger("collector.liquidations.bybit")

VENUE = "bybit"
COLLECTOR = "liquidations"
SOURCE = "stream:bybit"
STREAM_URL = "wss://stream.bybit.com/v5/public/linear"

# The funding / open-interest universe, so liquidations join against the series
# that already exist rather than introducing a second, mismatched asset list.
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]

# Bybit ACKs a subscribe request that exceeds its per-request arg limit and
# then silently delivers nothing for it. Five is well inside the limit and is
# the batch size proven to work from this host.
SUBSCRIBE_BATCH = 5

# Bybit drops a connection that has not been pinged within 20s.
PING_INTERVAL = 20.0

BACKOFF_BASE = 1.0
BACKOFF_MAX = 60.0

# Runs shorter than this are too brief to draw any conclusion from.
DEFAULT_MIN_WINDOW_SECONDS = 300
# Liquidations are genuinely bursty and a quiet hour is real, so the
# liquidation-count floor is deliberately near-zero: it fires only on a feed
# that is dead, never on a market that is calm. The control topic, not this
# number, is what actually proves the subscription is alive.
DEFAULT_MIN_ROWS_PER_HOUR = 1.0

INSERT_SQL = """
    INSERT OR IGNORE INTO liquidations
        (venue, symbol, timestamp, datetime, side, price, quantity,
         quote_qty, order_status, order_type, avg_price, event_time,
         ingested_at, source, side_raw, price_basis)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def bybit_symbol(asset: str) -> str:
    """Asset ticker -> Bybit USDT-linear perpetual symbol."""
    return "%sUSDT" % asset.upper()


def parse_liquidation(d: dict, event_ts=None) -> dict | None:
    """One allLiquidation data element -> a storable row dict, or None.

    Payload element (Bybit v5):
        {"T": <ms>, "s": symbol, "S": "Buy"|"Sell", "v": size, "p": price}

    `S` is the POSITION side and is inverted relative to Binance's order side;
    normalise_side does the translation and the raw token is preserved.
    `p` is the BANKRUPTCY price, not a fill price.
    """
    symbol = d.get("s")
    raw_side = d.get("S")
    if not symbol or not raw_side:
        return None

    # Rule 35.4: range-check rather than trust the feed.
    ts = assert_ms(d.get("T") if d.get("T") is not None else event_ts,
                   "liquidation.timestamp")

    price = float(d.get("p") or 0.0)
    quantity = float(d.get("v") or 0.0)
    if price <= 0 or quantity <= 0:
        return None

    # Raises on an unrecognised token -- a liquidation whose direction cannot
    # be established would be counted on the wrong side of every later test.
    side = normalise_side(VENUE, raw_side)

    # Bybit gives no fill price, so this is bankruptcy price x size: an
    # approximation with a known directional bias, flagged as such by
    # price_basis rather than presented as a measured notional.
    quote_qty = price * quantity

    return {
        "venue": VENUE,
        "symbol": symbol,
        "timestamp": ts,
        "datetime": ms_to_iso(ts),
        "side": side,
        "side_raw": str(raw_side),
        "price": price,
        "quantity": quantity,
        "quote_qty": quote_qty,
        # Bybit sends none of these. Left NULL: absent, not guessed.
        "order_status": None,
        "order_type": None,
        "avg_price": None,
        "event_time": event_ts,
        "price_basis": PRICE_BASIS[VENUE],
    }


def insert_rows(conn, rows):
    """Insert rows; return (written, suppressed_duplicates)."""
    if not rows:
        return 0, 0
    ingested = now_ms()
    payload = [
        (r["venue"], r["symbol"], r["timestamp"], r["datetime"], r["side"],
         r["price"], r["quantity"], r["quote_qty"], r["order_status"],
         r["order_type"], r["avg_price"], r["event_time"], ingested, SOURCE,
         r["side_raw"], r["price_basis"])
        for r in rows
    ]
    cur = conn.cursor()
    before = conn.total_changes
    cur.executemany(INSERT_SQL, payload)
    written = conn.total_changes - before
    return written, len(rows) - written


async def stream_once(conn, stats, deadline, batch_size, symbols, control_topic):
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

    topics = ["allLiquidation.%s" % s for s in symbols]

    async with connect(STREAM_URL, ping_interval=None, max_queue=8192) as ws:
        connected_at = now_ms()
        stats["connects"] += 1
        logger.info("connected to %s", STREAM_URL)

        # A disconnect we already recorded now has an end: close it.
        if stats.get("open_gap_start") is not None:
            record_gap(conn, COLLECTOR, VENUE, stats["open_gap_start"],
                       connected_at, stats.get("open_gap_reason", "reconnect"))
            stats["open_gap_start"] = None
            stats["gaps_closed"] += 1

        # Small batches: an oversized args list is ACKed and silently dropped.
        for i in range(0, len(topics), SUBSCRIBE_BATCH):
            chunk = topics[i:i + SUBSCRIBE_BATCH]
            await ws.send(json.dumps({"op": "subscribe", "args": chunk}))
            stats["subscribes_sent"] += 1
            logger.info("subscribe -> %s", ", ".join(chunk))
        # The control topic proves the market-data path is live. A pong only
        # proves the socket is; the Cycle 62A trap was a live socket with a
        # silently dead subscription.
        await ws.send(json.dumps({"op": "subscribe", "args": [control_topic]}))
        stats["subscribes_sent"] += 1
        logger.info("subscribe -> %s (liveness control)", control_topic)

        async def pinger():
            while True:
                await asyncio.sleep(PING_INTERVAL)
                try:
                    await ws.send(json.dumps({"op": "ping"}))
                except Exception:
                    return

        ping_task = asyncio.create_task(pinger())
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
                    # Idle tick, not an error: the feed is genuinely bursty.
                    flush()
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("undecodable frame skipped: %.120s", raw)
                    stats["bad_frames"] += 1
                    continue
                if not isinstance(msg, dict):
                    stats["bad_frames"] += 1
                    continue

                # Control-plane replies: subscribe ACKs and pongs.
                op = msg.get("op")
                if op or "success" in msg:
                    if op == "subscribe":
                        stats["subscribe_acks"] += 1
                        if msg.get("success") is False:
                            logger.error("subscribe REJECTED: %s", raw[:200])
                            stats["subscribe_failures"] += 1
                    continue

                topic = msg.get("topic") or ""
                if topic == control_topic:
                    stats["control_frames"] += 1
                    continue
                if not topic.startswith("allLiquidation"):
                    continue

                event_ts = msg.get("ts")
                try:
                    event_ts = assert_ms(event_ts, "liquidation.event_time")
                except ValueError:
                    event_ts = None

                data = msg.get("data")
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    stats["bad_frames"] += 1
                    continue

                for d in data:
                    if not isinstance(d, dict):
                        continue
                    stats["received"] += 1
                    try:
                        row = parse_liquidation(d, event_ts)
                    except ValueError as e:
                        logger.warning("rejected event: %s", e)
                        stats["rejected"] += 1
                        continue
                    if row is None:
                        stats["rejected"] += 1
                        continue
                    buf.append(row)
                    stats["symbols"][row["symbol"]] += 1
                    stats["sides"][row["side"]] += 1
                    logger.debug("%s %s %s(raw=%s) qty=%s px=%s notional=%.0f",
                                 row["datetime"], row["symbol"], row["side"],
                                 row["side_raw"], row["quantity"],
                                 row["price"], row["quote_qty"])

                if len(buf) >= batch_size:
                    flush()
        finally:
            ping_task.cancel()
            flush()
            stats["connected_seconds"] += (now_ms() - connected_at) / 1000.0


async def run_collect(args):
    conn = open_db()
    symbols = [bybit_symbol(a) for a in args.assets]
    control_topic = "kline.1.%s" % symbols[0]

    stats = {
        "received": 0, "written": 0, "duplicates": 0, "rejected": 0,
        "bad_frames": 0, "connects": 0, "gaps_opened": 0, "gaps_closed": 0,
        "connected_seconds": 0.0, "symbols": Counter(), "sides": Counter(),
        "control_frames": 0, "subscribes_sent": 0, "subscribe_acks": 0,
        "subscribe_failures": 0,
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
                                           args.batch_size, symbols,
                                           control_topic)
                backoff = BACKOFF_BASE
                if reason == "duration reached":
                    break
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

    return report_run(args, stats, started_ms, symbols, control_topic)


def report_run(args, stats, started_ms, symbols, control_topic):
    elapsed = (now_ms() - started_ms) / 1000.0
    connected = stats["connected_seconds"]
    n_written = stats["written"]

    print("")
    print("=" * 66)
    print("BYBIT LIQUIDATION COLLECTOR RUN")
    print("=" * 66)
    print("  window             : %s -> %s"
          % (ms_to_iso(started_ms), ms_to_iso(now_ms())))
    print("  elapsed / connected: %.1fs / %.1fs" % (elapsed, connected))
    print("  symbols subscribed : %d (%s)" % (len(symbols), ", ".join(symbols)))
    print("  connects           : %d" % stats["connects"])
    print("  subscribes sent/ack: %d / %d"
          % (stats["subscribes_sent"], stats["subscribe_acks"]))
    print("  control frames     : %d (%s)"
          % (stats["control_frames"], control_topic))
    print("  gaps opened/closed : %d / %d"
          % (stats["gaps_opened"], stats["gaps_closed"]))
    print("  events received    : %d" % stats["received"])
    print("  rows written       : %d" % n_written)
    print("  duplicates ignored : %d" % stats["duplicates"])
    print("  rejected / bad     : %d / %d"
          % (stats["rejected"], stats["bad_frames"]))
    if stats["sides"]:
        print("  side split (order) : "
              + ", ".join("%s=%d" % (s, n)
                          for s, n in sorted(stats["sides"].items())))
    if stats["symbols"]:
        top = stats["symbols"].most_common(10)
        print("  distinct symbols   : %d" % len(stats["symbols"]))
        print("  top symbols        : "
              + ", ".join("%s=%d" % (s, n) for s, n in top))

    if stats["connects"] == 0:
        print("\n[FAIL] never established a connection.", file=sys.stderr)
        return 1

    if stats["subscribe_failures"]:
        print("\n[FAIL] %d subscribe request(s) were rejected."
              % stats["subscribe_failures"], file=sys.stderr)
        return 2

    if connected < args.min_window_seconds:
        print("\n[OK] ran %.0fs connected (below the %ds window needed to "
              "judge feed health); wrote %d rows."
              % (connected, args.min_window_seconds, n_written))
        return 0

    # The check that a row count cannot make. A live socket with a silently
    # dropped subscription looks exactly like a quiet market from the
    # liquidation count alone; the control topic tells them apart.
    if stats["control_frames"] == 0:
        print("\n[FAIL] connected %.0fs and the control topic %s delivered "
              "NOTHING. The socket was open but the market-data subscription "
              "was not live, so 'no liquidations' here is not evidence of a "
              "quiet market -- it is an unobserved window."
              % (connected, control_topic), file=sys.stderr)
        return 2

    expected = max(1, int(args.min_rows_per_hour * connected / 3600.0))
    if n_written < expected:
        print("\n[FAIL] wrote %d rows in %.0fs connected; expected at least "
              "%d. The control topic WAS live (%d frames), so the "
              "subscription path works and this is a genuinely empty "
              "liquidation window -- flagged rather than reported as success "
              "so it is looked at."
              % (n_written, connected, expected, stats["control_frames"]),
              file=sys.stderr)
        return 2

    print("\n[OK] wrote %d rows (expected >= %d); control topic live "
          "(%d frames)." % (n_written, expected, stats["control_frames"]))
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Bybit allLiquidation stream collector (T1 venue). "
                    "Counts are NOT comparable to Binance-based priors -- see "
                    "the module docstring.")
    p.add_argument("--verbose", type=int, default=3, choices=[0, 1, 2, 3],
                   help="0=ERROR 1=WARNING 2=INFO 3=DEBUG (default 3, maximum)")
    subs = p.add_subparsers(dest="command", required=True)

    c = subs.add_parser("collect", help="Stream liquidations into the DB")
    c.add_argument("--duration", type=int, default=None,
                   help="Seconds to run before exiting (default: forever). "
                        "The scheduled task uses 3550 against an hourly "
                        "trigger so invocations never overlap.")
    c.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS,
                   help="Assets to subscribe (default: the funding/OI "
                        "universe). Bybit has no all-market liquidation "
                        "topic, so coverage is exactly this list.")
    c.add_argument("--batch-size", type=int, default=50,
                   help="Events buffered before a DB flush (default 50)")
    c.add_argument("--min-rows-per-hour", type=float,
                   default=DEFAULT_MIN_ROWS_PER_HOUR,
                   help="Below this rate the run exits non-zero (default %g)"
                        % DEFAULT_MIN_ROWS_PER_HOUR)
    c.add_argument("--min-window-seconds", type=int,
                   default=DEFAULT_MIN_WINDOW_SECONDS,
                   help="Connected seconds required before health checks "
                        "apply (default %d)" % DEFAULT_MIN_WINDOW_SECONDS)

    r = subs.add_parser("report", help="Capture statistics over a window")
    r.add_argument("--hours", type=float, default=24.0)
    r.add_argument("--venue", default=None,
                   help="Restrict to one venue (counts are not poolable)")

    subs.add_parser("validate", help="Structural checks on stored rows")

    args = p.parse_args()
    setup_logging(args.verbose)

    if args.command == "collect":
        return asyncio.run(run_collect(args))
    if args.command == "report":
        args.collector = COLLECTOR
        return cmd_report(args)
    if args.command == "validate":
        return cmd_validate(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
