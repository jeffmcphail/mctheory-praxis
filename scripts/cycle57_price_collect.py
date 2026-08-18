"""
scripts/cycle57_price_collect.py
================================
Cycle 57 — READ-ONLY public PRICE collection for the basis-inclusive Sharpe.

Collects hourly close prices into the ISOLATED research DB
(data/defi_funding_research.db, a new `prices` table) — NOT production
crypto_data.db (Cycle 54/56 isolation lesson):

  - hyperliquid PERP  ({asset}/USDC:USDC)     -- the short leg
  - coinbase   SPOT   ({asset}/USD)           -- the realistic Canada-legal hedge leg
  - binance    SPOT   ({asset}/USDT)          -- deep reliable cross-check + same-venue anchor
  - binance    PERP   ({asset}/USDT:USDT)     -- same-venue CEX anchor (vs binance spot)

Window 2024-01-01 -> now, hourly. Funding already collected (Cycle 56 HL hourly
in this DB; binance/bybit 8h in production, read-only). Join on (asset, timestamp).

NO credentials, NO orders, NO keys. All public market data. Idempotent
(INSERT OR IGNORE on (asset, venue, market, timestamp)).

Why Coinbase (not Kraken) for the hedge leg: Kraken's public OHLC endpoint is
capped at ~720 most-recent candles and cannot return 2.5y of hourly history;
Coinbase Advanced paginates via since+limit. Binance spot is collected too as a
deep cross-check (major-CEX spot prices track within a few bps, so it is a valid
proxy for hedge-venue spot when judging basis volatility).
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt

REPO = Path(__file__).resolve().parent.parent
RESEARCH_DB = REPO / "data" / "defi_funding_research.db"

START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
NOW_MS = int(datetime.now(timezone.utc).timestamp() * 1000)

MAJORS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
SHORTLIST = ["DOGE", "LINK", "UNI", "LTC"]
ASSETS = MAJORS + SHORTLIST

# (ccxt_id, market, symbol_fmt, page_limit)
SOURCES = [
    ("hyperliquid", "perp", "{a}/USDC:USDC", 5000),
    ("binance",     "spot", "{a}/USDT",      1000),
    ("binance",     "perp", "{a}/USDT:USDT", 1000),
    ("coinbase",    "spot", "{a}/USD",       300),
]


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(RESEARCH_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            asset     TEXT    NOT NULL,
            venue     TEXT    NOT NULL,
            market    TEXT    NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime  TEXT    NOT NULL,
            close     REAL    NOT NULL,
            PRIMARY KEY (asset, venue, market, timestamp)
        )
    """)
    conn.commit()
    return conn


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00")


def make_exchange(cid: str):
    opts = {"enableRateLimit": True}
    if cid == "binance":
        # default spot client; perp uses a separate future-typed client
        return ccxt.binance(opts)
    return getattr(ccxt, cid)(opts)


def fetch_all(ex, symbol: str, limit: int, pause: float = 0.25) -> list:
    out, since, guard = [], START_MS, 0
    while since < NOW_MS and guard < 400:
        guard += 1
        for attempt in range(4):
            try:
                batch = ex.fetch_ohlcv(symbol, "1h", since=since, limit=limit)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(1.0 * (attempt + 1))
        if not batch:
            break
        out.extend(batch)
        last = batch[-1][0]
        if last <= since:
            break
        since = last + 1
        time.sleep(pause)
        if len(batch) < limit // 2:  # near the live edge
            break
    return out


def main():
    print(f"[cycle57] research DB: {RESEARCH_DB}")
    print(f"[cycle57] window {iso(START_MS)} .. {iso(NOW_MS)}")
    conn = init_db()
    # one perp-typed binance client reused
    binance_perp = ccxt.binance({"enableRateLimit": True,
                                 "options": {"defaultType": "future"}})
    try:
        for cid, market, fmt, limit in SOURCES:
            print(f"=== {cid} {market} ===")
            ex = binance_perp if (cid == "binance" and market == "perp") \
                else make_exchange(cid)
            try:
                ex.load_markets()
            except Exception as e:
                print(f"  load_markets failed: {e!r}")
            for a in ASSETS:
                sym = fmt.format(a=a)
                try:
                    if sym not in ex.markets:
                        # try USDC quote fallback for coinbase
                        alt = f"{a}/USDC" if cid == "coinbase" else sym
                        if alt in ex.markets:
                            sym = alt
                        else:
                            print(f"  {cid:<11}{market:<5}{a:<6} NOT LISTED")
                            continue
                    bars = fetch_all(ex, sym, limit)
                    rows = [(a, cid, market, int(b[0]), iso(int(b[0])),
                             float(b[4])) for b in bars if b[4] is not None]
                    conn.executemany(
                        "INSERT OR IGNORE INTO prices "
                        "(asset, venue, market, timestamp, datetime, close) "
                        "VALUES (?,?,?,?,?,?)", rows)
                    conn.commit()
                    n, lo, hi = conn.execute(
                        "SELECT COUNT(*), MIN(datetime), MAX(datetime) FROM prices "
                        "WHERE asset=? AND venue=? AND market=?",
                        (a, cid, market)).fetchone()
                    print(f"  {cid:<11}{market:<5}{a:<6} sym={sym:<14} "
                          f"rows={n:<6} {lo} .. {hi}")
                except Exception as e:
                    print(f"  {cid:<11}{market:<5}{a:<6} ERROR {e!r}")
        print("=== DONE ===")
        for row in conn.execute(
                "SELECT venue, market, COUNT(DISTINCT asset), COUNT(*) "
                "FROM prices GROUP BY venue, market"):
            print(" ", row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
