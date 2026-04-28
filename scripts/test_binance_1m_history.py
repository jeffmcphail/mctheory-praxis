"""One-off test: how far back does Binance serve 1-min (and 5-min) OHLCV?

Probes the live API at several `since` offsets without modifying the
production collector. Prints a table Chat can use to decide whether
Option C (dataset extension) is blocked.

Bonus probes per BRIEF_praxis_intrabar_data_extension.md open questions:
  - 730-day (2-year) probe for 1-min data
  - Parallel 5-min probe at the same offsets, in case 5-min retention
    differs from 1-min retention

Usage:
    python scripts/test_binance_1m_history.py
"""
import sys
from datetime import datetime, timezone, timedelta

import ccxt
from dotenv import load_dotenv

load_dotenv()

PROBE_DAYS = [60, 90, 180, 365, 730]
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = ["1m", "5m"]


def probe(exchange, symbol, timeframe, days_back):
    """Return (status, oldest_dt_str, n_candles, error) for one probe."""
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000
    )
    requested_dt = datetime.fromtimestamp(
        since_ms / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")

    try:
        candles = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=1000
        )
    except Exception as e:
        return ("ERROR", requested_dt, 0, f"{type(e).__name__}: {e}")

    if not candles:
        return ("EMPTY", requested_dt, 0, "")

    oldest_ts = candles[0][0] / 1000
    oldest_dt = datetime.fromtimestamp(
        oldest_ts, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    requested_ts = since_ms / 1000
    drift_seconds = oldest_ts - requested_ts
    if drift_seconds > 86400:
        status = "CLAMPED"
    else:
        status = "OK"
    return (status, oldest_dt, len(candles), "")


def main():
    exchange = ccxt.binance({"enableRateLimit": True})

    print("=" * 80)
    print("BINANCE 1-MIN / 5-MIN HISTORICAL RETENTION PROBE")
    print(f"Run at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    results = []
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\n  --- {symbol}  timeframe={tf} ---")
            for days in PROBE_DAYS:
                status, oldest_dt, n, err = probe(exchange, symbol, tf, days)
                results.append({
                    "symbol": symbol, "tf": tf, "days_back": days,
                    "status": status, "oldest": oldest_dt, "n": n, "error": err,
                })
                req_dt = datetime.fromtimestamp(
                    (datetime.now(timezone.utc) - timedelta(days=days)).timestamp(),
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d %H:%M")
                print(f"    probe {days:>4d}d  status={status:<8s}  "
                      f"requested={req_dt}  oldest_returned={oldest_dt}  "
                      f"n={n}" + (f"  err={err}" if err else ""))

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"  {'symbol':<10s} {'tf':<4s} {'days':>5s}  {'status':<8s}  "
          f"{'oldest returned':<22s}  {'n':>5s}")
    print("  " + "-" * 72)
    for r in results:
        print(f"  {r['symbol']:<10s} {r['tf']:<4s} {r['days_back']:>5d}  "
              f"{r['status']:<8s}  {r['oldest']:<22s}  {r['n']:>5d}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT (1-min only -- Option C gate)")
    print("=" * 80)
    verdict_ok_90 = all(
        r["status"] == "OK" and r["n"] > 0
        for r in results
        if r["tf"] == "1m" and r["days_back"] == 90
    )
    if verdict_ok_90:
        print("  BTC and ETH served >=90 days of 1-min data. "
              "Proceed to modify collector cap and backfill.")
    else:
        print("  Binance did NOT serve >=90 days of 1-min data cleanly. "
              "Do NOT modify collector. ZIP loader path needed for Option C.")

    return results


if __name__ == "__main__":
    main()
