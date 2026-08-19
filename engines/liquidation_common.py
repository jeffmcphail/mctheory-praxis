"""
engines/liquidation_common.py -- what the two liquidation collectors must agree on.

There are now two venues writing into one `liquidations` table, and they do
NOT speak the same language. This module is the single place where the
translation lives, so the two collectors cannot drift apart quietly.

THE SIDE CONVENTION -- THE ONE THAT BITES
-----------------------------------------
Binance `forceOrder.S` is the ORDER side. Force-closing a long sends a market
SELL into the book, so on Binance:

    SELL => a LONG was liquidated
    BUY  => a SHORT was liquidated

Bybit `allLiquidation.S` is documented as the POSITION side -- "Position side.
Buy,Sell. When you receive a `Buy` update, this means that a long position has
been liquidated" -- so on Bybit:

    BUY  => a LONG was liquidated
    SELL => a SHORT was liquidated

Same two letters, opposite meanings. Copying Bybit's `S` straight into the
`side` column would invert every Bybit row relative to every Binance row, and
no validator, type check or unit test would notice: both values are legal
members of {BUY, SELL}. It is the exact shape of the Cycle 57 basis-blind P&L
and the Cycle 53 unverified-strategy failures -- a field that is present,
well-typed, and wrong.

So `side` carries ONE convention for all venues: the Binance ORDER side. That
one is chosen not because Binance is privileged but because the column already
means that, the Cycle 61 A2 detector already compares against it, and
rewriting stored history to a new convention is a worse risk than translating
at the edge.

`side_raw` keeps whatever the venue actually sent, so the translation is
auditable after the fact and reversible if a venue changes its docs.

VERIFIED AGAINST PRICE, NOT AGAINST THE DOCS
--------------------------------------------
Bybit's `S` is a widely mis-documented field, so the mapping above was checked
against live tape rather than trusted: force-closing a long sends a market
SELL, which ticks price DOWN in the seconds around the event, and force-closing
a short ticks it UP. See outputs/forced_trade/t1_bybit_side_convention.json for
the measurement and the Cycle 62A retro for the numbers.
"""
from __future__ import annotations

import sys

from engines.collector_common import ms_to_iso, now_ms, open_db

# Canonical stored convention: the side of the LIQUIDATION ORDER.
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

# What each venue's raw field means, mapped to the canonical order side.
# Keyed by venue, then by the venue's own uppercased token.
SIDE_MAP = {
    # Binance already reports the order side -- identity.
    "binance": {"BUY": SIDE_BUY, "SELL": SIDE_SELL},
    # Bybit reports the POSITION side -- inverted.
    #   position Buy  = a long  was liquidated = the order that closed it SOLD
    #   position Sell = a short was liquidated = the order that closed it BOUGHT
    "bybit": {"BUY": SIDE_SELL, "SELL": SIDE_BUY},
}

# Which kind of price each venue's `price` field is. Bybit publishes the
# BANKRUPTCY price -- the price at which margin hits zero, not a traded price
# -- so a Bybit quote_qty is an approximation with a known directional bias,
# never a measured notional.
PRICE_BASIS = {
    "binance": "executed",     # avg fill price x filled qty where available
    "bybit": "bankruptcy",
}


def normalise_side(venue: str, raw) -> str:
    """Translate a venue's side token into the canonical ORDER side.

    Raises on anything unrecognised. A liquidation whose direction cannot be
    established is worse than no row at all -- it is a row that will be counted
    on the wrong side of every subsequent test.
    """
    if raw is None:
        raise ValueError("%s: side missing" % venue)
    table = SIDE_MAP.get(venue)
    if table is None:
        raise ValueError("no side convention registered for venue %r -- add "
                         "one to SIDE_MAP before collecting from it" % venue)
    token = str(raw).strip().upper()
    try:
        return table[token]
    except KeyError:
        raise ValueError("%s: unrecognised side %r (expected one of %s)"
                         % (venue, raw, sorted(table)))


# --------------------------------------------------------------- report ---

def cmd_report(args) -> int:
    """Window statistics over the liquidations table, optionally per venue."""
    conn = open_db()
    try:
        cutoff = now_ms() - int(args.hours * 3600 * 1000)
        venue = getattr(args, "venue", None)
        where = "timestamp >= ?"
        params = [cutoff]
        if venue:
            where += " AND venue = ?"
            params.append(venue)

        print("=" * 66)
        print("LIQUIDATIONS -- last %gh%s"
              % (args.hours, (" [venue=%s]" % venue) if venue else ""))
        print("=" * 66)

        span = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM liquidations"
        ).fetchone()
        if span[2]:
            print("  table span   : %s -> %s"
                  % (ms_to_iso(span[0]), ms_to_iso(span[1])))
        print("  table rows   : %d" % span[2])

        # Venue mix first, and loudly: these counts are NOT poolable. Different
        # perp market share, different liquidation engine, different stream
        # throttle. Summing them produces a number that means nothing.
        rows = list(conn.execute(
            "SELECT venue, source, COUNT(*), MIN(timestamp), MAX(timestamp) "
            "FROM liquidations GROUP BY venue, source ORDER BY 3 DESC"))
        if rows:
            print("\n  -- provenance (counts are NOT comparable across venues) --")
            for v, src, n, lo, hi in rows:
                print("    %-9s %-18s %8d  %s -> %s"
                      % (v, src or "(null)", n, ms_to_iso(lo), ms_to_iso(hi)))

        total = conn.execute(
            "SELECT COUNT(*) FROM liquidations WHERE " + where, params
        ).fetchone()[0]
        print("\n  window rows  : %d" % total)
        if not total:
            print("\n  (no rows in window)")
            return 0

        print("\n  -- side split (canonical ORDER side; SELL = long liquidated) --")
        for side, n, notional in conn.execute(
            "SELECT side, COUNT(*), SUM(quote_qty) FROM liquidations "
            "WHERE " + where + " GROUP BY side ORDER BY 2 DESC", params
        ):
            print("    %-5s %7d  notional=%15.0f" % (side, n, notional or 0))

        print("\n  -- symbol distribution (top 15) --")
        nsym = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM liquidations WHERE " + where,
            params).fetchone()[0]
        print("    distinct symbols: %d" % nsym)
        for sym, n, notional in conn.execute(
            "SELECT symbol, COUNT(*), SUM(quote_qty) FROM liquidations "
            "WHERE " + where + " GROUP BY symbol ORDER BY 2 DESC LIMIT 15",
            params
        ):
            print("    %-14s %7d  notional=%15.0f" % (sym, n, notional or 0))

        print("\n  -- size distribution (USD notional) --")
        vals = [r[0] for r in conn.execute(
            "SELECT quote_qty FROM liquidations WHERE " + where +
            " AND quote_qty IS NOT NULL ORDER BY quote_qty", params)]
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

        collector = getattr(args, "collector", None)
        gsql = ("SELECT timestamp, gap_end, gap_seconds, reason, collector "
                "FROM collector_gaps WHERE timestamp >= ?")
        gparams = [cutoff]
        if collector:
            gsql += " AND collector = ?"
            gparams.append(collector)
        gaps = list(conn.execute(gsql + " ORDER BY timestamp", gparams))
        print("\n  -- collection gaps in window: %d --" % len(gaps))
        for ts, end, secs, reason, coll in gaps[:20]:
            end_s = ms_to_iso(end) if end else "STILL OPEN"
            secs_s = ("%.1fs" % secs) if secs is not None else "open"
            print("    [%s] %s -> %s (%s) %s"
                  % (coll, ms_to_iso(ts), end_s, secs_s, reason))
        if len(gaps) > 20:
            print("    ... and %d more" % (len(gaps) - 20))
        return 0
    finally:
        conn.close()


# ------------------------------------------------------------- validate ---

def cmd_validate(args) -> int:
    """Structural checks on stored rows. Exits non-zero on any failure."""
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
        check("side outside canonical BUY/SELL",
              "SELECT COUNT(*) FROM liquidations "
              "WHERE side NOT IN ('BUY','SELL')")
        check("datetime missing",
              "SELECT COUNT(*) FROM liquidations "
              "WHERE datetime IS NULL OR datetime = ''")
        check("venue not set",
              "SELECT COUNT(*) FROM liquidations "
              "WHERE venue IS NULL OR venue = ''")
        check("source not set (provenance is mandatory)",
              "SELECT COUNT(*) FROM liquidations "
              "WHERE source IS NULL OR source = ''")

        # Every venue in the table must have a registered side convention,
        # otherwise `side` is not comparable across the rows it contains.
        unknown = [v for (v,) in conn.execute(
            "SELECT DISTINCT venue FROM liquidations") if v not in SIDE_MAP]
        ok = not unknown
        print("  [%s] every stored venue has a side convention: %s"
              % ("OK  " if ok else "FAIL", unknown or "yes"))
        if unknown:
            failures.append("unregistered venue side convention")

        # A venue whose raw token differs from the canonical one MUST have
        # preserved the original, or the translation is unauditable.
        n = conn.execute(
            "SELECT COUNT(*) FROM liquidations "
            "WHERE venue != 'binance' AND (side_raw IS NULL OR side_raw = '')"
        ).fetchone()[0]
        ok = n == 0
        print("  [%s] translated rows keep side_raw: %d missing"
              % ("OK  " if ok else "FAIL", n))
        if n:
            failures.append("side_raw missing")

        # And the translation must actually match the declared map, row by row.
        bad_map = 0
        for venue, raw, side, n in conn.execute(
            "SELECT venue, side_raw, side, COUNT(*) FROM liquidations "
            "WHERE side_raw IS NOT NULL GROUP BY venue, side_raw, side"
        ):
            try:
                expect = normalise_side(venue, raw)
            except ValueError:
                bad_map += n
                continue
            if expect != side:
                print("        %s raw=%s stored=%s expected=%s (%d rows)"
                      % (venue, raw, side, expect, n))
                bad_map += n
        ok = bad_map == 0
        print("  [%s] stored side matches the declared convention: %d bad rows"
              % ("OK  " if ok else "FAIL", bad_map))
        if bad_map:
            failures.append("side translation mismatch")

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
