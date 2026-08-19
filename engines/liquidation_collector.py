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

SUPERSEDED AS THE T1 VENUE -- BYBIT COLLECTS INSTEAD
-----------------------------------------------------
engines/bybit_liquidation_collector.py is the ACTIVE T1 collector. This module
stays in place, correct and tested, but UNSCHEDULED, for whenever fstream
becomes reachable. It was not deleted because if the block ever lifts, Binance
becomes the better source (larger perp share, and the venue every published
liquidation prior is built on) and Bybit becomes the cross-venue check.

Do not read Bybit rows and Binance rows as the same measurement -- perp market
share, liquidation engine and stream throttle all differ. The comparison is
spelled out in the Bybit module's docstring; the short version is that Binance
throttles to one event per symbol per second and Bybit does not, so event RATES
do not transfer in either direction.

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

WHERE THE BLOCK IS -- SETTLED, AND NOT WHERE CYCLE 62A FIRST GUESSED
--------------------------------------------------------------------
Cycle 62A's first reading was "a local middlebox, not a regional restriction",
reasoning that geo-blocking would take REST and WS together. That reasoning was
wrong, and the follow-up measurement says so plainly.

The fstream endpoint ANSWERS CONTROL MESSAGES and then withholds only market
data:

    -> {"method":"LIST_SUBSCRIPTIONS","id":1}
    <- {"result":[],"id":1}                     server replies
    -> {"method":"SUBSCRIBE","params":["btcusdt@aggTrade"],"id":2}
    <- {"result":null,"id":2}                   subscribe ACCEPTED
    then: zero market-data frames, socket open, no close code.

The identical exchange on spot returns the same two replies AND starts
delivering aggTrade frames immediately. So the futures server is alive,
parsing our JSON, and selectively sending everything except the data.

Nothing on the path could do that. The three local candidates were checked and
all are absent: Windows Firewall has NO enabled outbound Block rules (and a
firewall block would refuse the connection, not ACK a subscribe); the only
security product installed is Windows Defender, which does not intercept TLS;
and no proxy is configured (WinHTTP direct, WinINET disabled, no PAC). Most
decisively, fstream/fapi/stream all present a genuine DigiCert "GeoTrust TLS
RSA CA G1" chain for *.binance.com -- there is no interception, so nothing
on the path can read a WebSocket frame, let alone tell a subscribe ACK from an
aggTrade and drop one of them.

The suppression therefore happens inside the TLS tunnel, at the application
layer, which only Binance can do. It is SERVER-SIDE -- an entitlement or
regional restriction on futures market data for this egress IP.

What follows from that: no local change fixes this. Firewall rules, AV
settings and proxy configuration are all dead ends. Only a different egress
(VPN, different network) would test it, and that is an infrastructure decision,
not a code one.

Bybit's allLiquidation topic IS reachable from this host and carries the same
event class (129 liquidation events in 300s across BTC/ETH/SOL, against a
31,763-event publicTrade control on the same connection). That is a VENUE
SUBSTITUTION, not a drop-in: Bybit liquidations are not Binance liquidations,
and adopting them changes what scenario A2 is measured against.

ADOPTED (Cycle 62A, revised) once the archive path was enumerated and closed.
The substitution was made explicitly, not silently -- the Cycle 57 basis-blind
P&L is what silent substitution costs -- and the warning above was carried
forward into the Bybit collector, its .bat and the retro rather than dropped
now that we are proceeding anyway. The deciding argument was asymmetry: the
venue choice is reversible, the data loss is not.

THE ARCHIVE PATH IS ALSO CLOSED (Cycle 62A, measured)
------------------------------------------------------
data.binance.vision was checked as a same-venue, historical alternative to the
stream. It cannot serve this purpose either:

  data/futures/um/daily/liquidationSnapshot/   DOES NOT EXIST -- 0 keys.
      There is no USD-margined liquidation dataset in the public archive.
  data/futures/cm/daily/liquidationSnapshot/   EXISTS -- 118 COIN-margined
      symbols (BTCUSD_PERP and quarterlies), but the whole dataset stops at
      2024-10-14. Binance discontinued it.

Our `trades` coverage begins 2026-04-29, so the overlap with the archive is
ZERO -- a gap of roughly 18.5 months. No cross-check of the Cycle 61 A2
flow-burst detector against archive labels is possible at any date.

No liquidation dataset exists anywhere else in the bucket (cm/monthly,
spot/daily and option/daily were all enumerated).

Two properties of the archive were measured anyway, because they bound any
future use of it:

  1. IT IS A SAMPLED LOWER BOUND. Across every file checked, the maximum
     number of DISTINCT liquidations in any one second was exactly 1 -- the
     archive inherits the documented "largest order per symbol per 1000ms"
     stream throttle. Event counts and volumes taken from it are a floor,
     never a complete record.
  2. EVERY ROW IS WRITTEN TWICE. Duplication factor was exactly 2.00 in every
     file (BTCUSD_PERP 2024-10-14: 100 rows / 50 unique; 2024-09-02: 112 / 56;
     ETHUSD_PERP 2024-08-05: 1146 / 573). A naive row count DOUBLES the true
     event count.

Those two errors run in opposite directions, which is why the `source` column
exists: a count is only interpretable next to the provenance that produced it.

Archive layout, for whenever it matters (10 columns, header row present):
    time,side,order_type,time_in_force,original_quantity,price,
    average_price,order_status,last_fill_quantity,accumulated_fill_quantity
`time` is epoch MILLISECONDS (1728866962145 -> 2024-10-14), not microseconds.

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
from engines.liquidation_common import (
    PRICE_BASIS, cmd_report, cmd_validate, normalise_side,
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

# Provenance travels with the row. The archive path is a sampled, duplicated
# view of the same event class (Cycle 62A: <=1 distinct event per symbol per
# second, every row written twice), so counts are only interpretable next to
# the source that produced them.
SOURCE = "stream:binance"

INSERT_SQL = """
    INSERT OR IGNORE INTO liquidations
        (venue, symbol, timestamp, datetime, side, price, quantity,
         quote_qty, order_status, order_type, avg_price, event_time,
         ingested_at, source, side_raw, price_basis)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        # Binance already reports the ORDER side, so this is the identity
        # translation -- but it goes through the same shared map as Bybit's
        # inverted one, so a venue can never quietly bypass the convention.
        "side": normalise_side(VENUE, side),
        "side_raw": str(side),
        "datetime": ms_to_iso(ts),
        "price": price,
        "quantity": quantity,
        "quote_qty": quote_qty,
        "order_status": o.get("X"),
        "order_type": o.get("o"),
        "avg_price": avg_price,
        "event_time": event_time,
        # avg fill price x filled qty where available, so a real executed
        # notional -- unlike Bybit's bankruptcy-price approximation.
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


# Report and validate are shared with the Bybit collector so the two
# venues can never drift into different definitions of the same table.
# See engines/liquidation_common.py -- in particular the side convention,
# which is inverted between these two venues.

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
