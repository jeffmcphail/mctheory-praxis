"""
engines/ohlcv_universe_collector.py -- multi-asset OHLCV (Cycle 62A T4).

WHY THIS EXISTS
---------------
Regime class K (cross-sectional dispersion) needs at least three universe
assets to compute. Every OHLCV table in crypto_data.db held exactly BTC and
ETH, so K was uncomputable and appeared in RegimeState.missing on 100% of
evaluations -- Cycle 61 T5. Not degraded, not noisy: structurally absent, on
every regime vector Praxis has ever produced.

Two assets is one pair, and a "cross-sectional dispersion" over one pair is
not a dispersion. Three is the floor, and the floor is what the engine checks.

SCOPE, AND WHY IT DOES NOT DEPEND ON T3
---------------------------------------
This collector reads Binance via ccxt REST. The T3 unlock universe comes from
CoinGecko. They are different providers with different failure modes, and K's
repair must not be hostage to T3's: if CoinGecko cannot serve the unlock
universe, K still has to become computable.

So the asset list is layered:
  1. BASE -- the funding_rates universe (BTC ETH SOL XRP ADA AVAX). Six assets,
     all long-listed on Binance, sufficient on their own to clear K's floor of
     three. This layer alone repairs K.
  2. UNLOCK -- whatever config/unlock_universe.json holds, if it exists AND
     the asset actually lists on Binance. Purely additive; a missing or empty
     config downgrades to layer 1 with a warning rather than failing.

Every symbol is resolved against Binance's live market list before use, so an
asset that does not trade there is skipped loudly instead of producing an
empty series that later reads as a quiet market.

Writes into the EXISTING ohlcv_daily and ohlcv_4h tables -- same schema, same
Rule 35 ms timestamps, same (asset, timestamp) primary key. No new table, no
change to the existing collectors, which keep their BTC/ETH scope.

Usage:
    python -m engines.ohlcv_universe_collector collect --days 400
    python -m engines.ohlcv_universe_collector verify-k
    python -m engines.ohlcv_universe_collector report
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import ccxt

from engines.collector_common import assert_ms, ms_to_iso, now_ms, open_db, setup_logging

logger = logging.getLogger("collector.ohlcv_universe")

# Layer 1: sufficient on its own to make K computable.
BASE_ASSETS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]

TIMEFRAMES = {
    "1d": ("ohlcv_daily", "date", "%Y-%m-%d"),
    "4h": ("ohlcv_4h", "datetime", None),
}

DAY_MS = 86_400_000


def resolve_symbols(ex, assets: list) -> tuple[dict, list]:
    """Map ASSET -> a Binance spot symbol that actually exists.

    Resolved against the live market list rather than assumed, because a
    hard-coded 'FOO/USDT' that does not list returns an empty series, and an
    empty series is indistinguishable from a dead one downstream.
    """
    markets = ex.load_markets()
    resolved, unlisted = {}, []
    for a in assets:
        for quote in ("USDT", "USDC", "FDUSD"):
            sym = "%s/%s" % (a, quote)
            m = markets.get(sym)
            if m and m.get("spot") and m.get("active"):
                resolved[a] = sym
                break
        else:
            unlisted.append(a)
    return resolved, unlisted


def load_unlock_assets() -> list:
    """Layer 2, optional. A missing config is a warning, never a failure."""
    try:
        from engines.unlock_universe import universe_assets
        return [a for a, _cg in universe_assets()]
    except FileNotFoundError:
        logger.warning("no unlock universe config yet -- collecting the base "
                       "universe only. K's repair does not depend on it.")
        return []
    except Exception as e:
        logger.warning("unlock universe unreadable (%s) -- base universe only", e)
        return []


def fetch_ohlcv(ex, symbol: str, timeframe: str, days: int) -> list:
    since = now_ms() - days * DAY_MS
    out, cursor = [], since
    step = DAY_MS if timeframe == "1d" else DAY_MS // 6
    while True:
        try:
            candles = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        except Exception as e:
            logger.warning("%s %s fetch error: %s", symbol, timeframe, e)
            break
        if not candles:
            break
        out.extend(candles)
        last = candles[-1][0]
        if len(candles) < 1000 or last >= now_ms() - step:
            break
        cursor = last + step
        time.sleep(0.25)
    return out


def store(conn, asset: str, timeframe: str, candles: list) -> int:
    table, text_col, fmt = TIMEFRAMES[timeframe]
    rows = []
    for ts, o, h, lo, c, v in candles:
        try:
            ts = assert_ms(ts, "%s.timestamp" % table)
        except ValueError as e:
            logger.warning("%s rejected candle: %s", asset, e)
            continue
        if fmt:
            text = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(fmt)
        else:
            text = ms_to_iso(ts)
        rows.append((asset, ts, text, o, h, lo, c, v))
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO %s (asset, timestamp, %s, open, high, low, "
        "close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)" % (table, text_col),
        rows)
    return conn.total_changes - before


def cmd_collect(args) -> int:
    assets = list(BASE_ASSETS)
    unlock = load_unlock_assets() if args.include_unlock else []
    for a in unlock:
        if a not in assets:
            assets.append(a)

    ex = ccxt.binance({"enableRateLimit": True})
    resolved, unlisted = resolve_symbols(ex, assets)

    print("=" * 70)
    print("MULTI-ASSET OHLCV COLLECTION")
    print("=" * 70)
    print("  base universe   : %s" % " ".join(BASE_ASSETS))
    print("  unlock universe : %s"
          % (" ".join(unlock) if unlock else "(none loaded)"))
    print("  resolved on Binance: %d of %d" % (len(resolved), len(assets)))
    if unlisted:
        print("  NOT LISTED (skipped, not silently empty): %s"
              % " ".join(unlisted))
    print("")

    conn = open_db()
    total = 0
    try:
        for timeframe in args.timeframes:
            days = args.days if timeframe == "1d" else min(args.days, args.days_4h)
            print("  -- %s (%d days) --" % (timeframe, days))
            for asset, symbol in resolved.items():
                candles = fetch_ohlcv(ex, symbol, timeframe, days)
                n = store(conn, asset, timeframe, candles)
                total += n
                print("    %-8s %-12s %5d fetched -> %5d new"
                      % (asset, symbol, len(candles), n))
                time.sleep(args.sleep)

        print("\n  rows written: %d" % total)
        if not resolved:
            print("\n[FAIL] no asset resolved to a Binance symbol.",
                  file=sys.stderr)
            return 1

        n_daily = conn.execute(
            "SELECT COUNT(DISTINCT asset) FROM ohlcv_daily").fetchone()[0]
        print("  distinct assets now in ohlcv_daily: %d" % n_daily)
        if n_daily < 3:
            print("\n[FAIL] fewer than 3 assets in ohlcv_daily; regime class K "
                  "remains uncomputable.", file=sys.stderr)
            return 2
        print("\n[OK] ohlcv_daily holds %d assets (K needs >= 3)." % n_daily)
        return 0
    finally:
        conn.close()


def cmd_verify_k(args) -> int:
    """Confirm K at the ACTING layer -- the engine, not the schema.

    The brief is explicit that a row count is not evidence: K has to be shown
    returning a state and leaving RegimeState.missing.
    """
    import numpy as np
    import pandas as pd
    from engines.regime_engine import RegimeEngine, compute_dispersion_regime

    conn = open_db()
    try:
        assets = [r[0] for r in conn.execute(
            "SELECT asset FROM ohlcv_daily GROUP BY asset "
            "HAVING COUNT(*) >= 60 ORDER BY asset")]
        print("=" * 70)
        print("REGIME CLASS K -- ACTING-LAYER VERIFICATION")
        print("=" * 70)
        print("  assets with >= 60 daily bars: %d  %s"
              % (len(assets), " ".join(assets)))

        if len(assets) < 3:
            print("\n[FAIL] K needs >= 3 assets; have %d." % len(assets),
                  file=sys.stderr)
            return 2

        universe = {}
        for a in assets:
            rows = list(conn.execute(
                "SELECT timestamp, open, high, low, close, volume "
                "FROM ohlcv_daily WHERE asset=? ORDER BY timestamp", (a,)))
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high",
                                             "low", "close", "volume"])
            df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            universe[a] = df[["open", "high", "low", "close", "volume"]]

        # 1. The dispersion function itself, on real returns.
        rets = {a: float(np.log(df["close"].iloc[-1] / df["close"].iloc[-2]))
                for a, df in universe.items()}
        state_k, raw_k = compute_dispersion_regime(rets)
        print("\n  -- compute_dispersion_regime on %d real 24h returns --"
              % len(rets))
        for a, r in sorted(rets.items(), key=lambda kv: kv[1]):
            print("      %-8s %+.4f" % (a, r))
        print("      state=%d  raw=%s"
              % (state_k, {k: round(v, 6) for k, v in raw_k.items()}))

        # 2. The full engine: K must carry a state AND be absent from missing.
        primary = universe[assets[0]]
        hourly = primary.resample("1h").ffill()
        engine = RegimeEngine(bars_per_day=24)
        st = engine.compute(ohlcv_hourly=hourly, universe_ohlcv=universe)

        print("\n  -- RegimeEngine.compute with a %d-asset universe --"
              % len(universe))
        print("      K state        : %s" % st.states.get("K"))
        print("      K in missing   : %s" % ("K" in st.missing))
        print("      missing classes: %s" % (st.missing or "(none)"))
        print("      K raw features : %s"
              % {k: round(v, 6) for k, v in st.raw_features.items()
                 if k.startswith("K_")})

        ok = ("K" in st.states) and ("K" not in st.missing)
        print("")
        if not ok:
            print("[FAIL] K still absent from the acting layer.", file=sys.stderr)
            return 2
        print("[OK] K is computable and no longer reported missing.")
        return 0
    finally:
        conn.close()


def cmd_report(args) -> int:
    conn = open_db()
    try:
        for table in ("ohlcv_daily", "ohlcv_4h"):
            print("=" * 70)
            print(table)
            print("=" * 70)
            for asset, n, lo, hi in conn.execute(
                "SELECT asset, COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM %s GROUP BY asset ORDER BY asset" % table
            ):
                print("  %-8s %6d rows  %s -> %s"
                      % (asset, n, ms_to_iso(lo)[:10], ms_to_iso(hi)[:10]))
            n_assets = conn.execute(
                "SELECT COUNT(DISTINCT asset) FROM %s" % table).fetchone()[0]
            print("  distinct assets: %d\n" % n_assets)
        return 0
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(description="Multi-asset OHLCV collector.")
    p.add_argument("--verbose", type=int, default=3, choices=[0, 1, 2, 3])
    subs = p.add_subparsers(dest="command", required=True)

    c = subs.add_parser("collect", help="Fetch OHLCV for the layered universe")
    c.add_argument("--days", type=int, default=400,
                   help="Daily-bar history to fetch (default 400)")
    c.add_argument("--days-4h", type=int, default=180,
                   help="4h-bar history, capped separately (default 180)")
    c.add_argument("--timeframes", nargs="+", default=["1d", "4h"],
                   choices=["1d", "4h"])
    c.add_argument("--sleep", type=float, default=0.3)
    c.add_argument("--include-unlock", action="store_true", default=True,
                   help="Add the T3 unlock universe when available (default on)")
    c.add_argument("--base-only", dest="include_unlock", action="store_false",
                   help="Base universe only; K's repair needs nothing more")

    subs.add_parser("verify-k", help="Prove K computes at the acting layer")
    subs.add_parser("report", help="Per-asset OHLCV coverage")

    args = p.parse_args()
    setup_logging(args.verbose)

    if args.command == "collect":
        return cmd_collect(args)
    if args.command == "verify-k":
        return cmd_verify_k(args)
    if args.command == "report":
        return cmd_report(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
