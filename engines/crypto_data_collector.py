"""
engines/crypto_data_collector.py -- Extended Crypto Data Collection

Collects 2+ years of data across multiple dimensions for the
LSTM + Quantamental crypto prediction system.

Data sources:
  1. OHLCV: CCXT/Binance (daily + 4h candles, 2+ years)
  2. Fear & Greed Index: alternative.me (daily history)
  3. Funding Rates: CCXT/Binance (perpetual futures)
  4. Market Cap / Dominance: CoinGecko
  5. On-Chain: blockchain.info (BTC) -- active addresses, tx volume, hash rate
  6. Exchange Flows: CoinGecko exchange reserves
  7. Social Sentiment: LunarCrush-style metrics (when available)

All stored in SQLite for feature engineering.

Usage:
    python -m engines.crypto_data_collector collect-all --asset BTC --days 900
    python -m engines.crypto_data_collector collect-ohlcv --asset BTC --days 900
    python -m engines.crypto_data_collector collect-ohlcv-4h --asset BTC --days 180
    python -m engines.crypto_data_collector collect-fear-greed --days 900
    python -m engines.crypto_data_collector collect-funding --asset BTC --days 365
    python -m engines.crypto_data_collector collect-onchain --days 365
    python -m engines.crypto_data_collector status
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

DB_PATH = Path("data/crypto_data.db")

SUPPORTED_ASSETS = {
    "BTC": {"coingecko_id": "bitcoin", "symbol": "BTC/USDT", "perp": "BTC/USDT:USDT"},
    "ETH": {"coingecko_id": "ethereum", "symbol": "ETH/USDT", "perp": "ETH/USDT:USDT"},
    "SOL": {"coingecko_id": "solana", "symbol": "SOL/USDT", "perp": "SOL/USDT:USDT"},
}


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            UNIQUE(asset, timestamp)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_4h (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            UNIQUE(asset, timestamp)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_1m (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            UNIQUE(asset, timestamp)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fear_greed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            date TEXT NOT NULL,
            value INTEGER,
            classification TEXT,
            UNIQUE(timestamp)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            funding_rate REAL,
            UNIQUE(asset, timestamp)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS onchain_btc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            active_addresses INTEGER,
            transaction_count INTEGER,
            hash_rate REAL,
            difficulty REAL,
            block_size REAL,
            total_btc REAL,
            market_cap REAL,
            UNIQUE(date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            date TEXT NOT NULL,
            market_cap REAL,
            total_volume REAL,
            circulating_supply REAL,
            total_supply REAL,
            ath REAL,
            ath_change_pct REAL,
            btc_dominance REAL,
            UNIQUE(asset, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS order_book_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            mid_price REAL NOT NULL,
            best_bid REAL NOT NULL,
            best_ask REAL NOT NULL,
            spread REAL NOT NULL,
            spread_bps REAL NOT NULL,
            bid_price_1 REAL, bid_vol_1 REAL,
            bid_price_2 REAL, bid_vol_2 REAL,
            bid_price_3 REAL, bid_vol_3 REAL,
            bid_price_4 REAL, bid_vol_4 REAL,
            bid_price_5 REAL, bid_vol_5 REAL,
            bid_price_6 REAL, bid_vol_6 REAL,
            bid_price_7 REAL, bid_vol_7 REAL,
            bid_price_8 REAL, bid_vol_8 REAL,
            bid_price_9 REAL, bid_vol_9 REAL,
            bid_price_10 REAL, bid_vol_10 REAL,
            ask_price_1 REAL, ask_vol_1 REAL,
            ask_price_2 REAL, ask_vol_2 REAL,
            ask_price_3 REAL, ask_vol_3 REAL,
            ask_price_4 REAL, ask_vol_4 REAL,
            ask_price_5 REAL, ask_vol_5 REAL,
            ask_price_6 REAL, ask_vol_6 REAL,
            ask_price_7 REAL, ask_vol_7 REAL,
            ask_price_8 REAL, ask_vol_8 REAL,
            ask_price_9 REAL, ask_vol_9 REAL,
            ask_price_10 REAL, ask_vol_10 REAL,
            bid_volume_top10 REAL NOT NULL,
            ask_volume_top10 REAL NOT NULL,
            order_imbalance_top10 REAL NOT NULL,
            UNIQUE(asset, timestamp)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ob_asset_timestamp
            ON order_book_snapshots(asset, timestamp DESC)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            trade_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            quote_amount REAL NOT NULL,
            is_buyer_maker INTEGER NOT NULL,
            side TEXT NOT NULL,
            UNIQUE(asset, trade_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_asset_timestamp
            ON trades(asset, timestamp DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_asset_tradeid
            ON trades(asset, trade_id DESC)
    """)

    conn.commit()
    return conn


# ===================================================================
# DATA COLLECTORS
# ===================================================================

def collect_ohlcv_daily(asset, days, conn):
    """Collect daily OHLCV via CCXT."""
    print(f"\n  Collecting daily OHLCV for {asset} ({days} days)...")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    except ImportError:
        print(f"    pip install ccxt --break-system-packages")
        return

    symbol = SUPPORTED_ASSETS[asset]["symbol"]
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_candles = []
    fetch_since = since

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, "1d", since=fetch_since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            print(f"    Fetched {len(all_candles)} candles so far "
                  f"(latest: {datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
            fetch_since = candles[-1][0] + 86400000
            if len(candles) < 1000:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"    CCXT error: {e}")
            break

    stored = 0
    for c in all_candles:
        ts = int(c[0] / 1000)
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ohlcv_daily
                (asset, timestamp, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (asset, ts, date, c[1], c[2], c[3], c[4], c[5]))
            stored += 1
        except Exception:
            pass

    conn.commit()
    print(f"    Stored {stored} daily candles")


def collect_ohlcv_4h(asset, days, conn):
    """Collect 4-hour OHLCV for higher-resolution features."""
    print(f"\n  Collecting 4h OHLCV for {asset} ({days} days)...")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    except ImportError:
        print(f"    pip install ccxt")
        return

    symbol = SUPPORTED_ASSETS[asset]["symbol"]
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_candles = []
    fetch_since = since

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, "4h", since=fetch_since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            if len(all_candles) % 5000 == 0:
                print(f"    {len(all_candles)} candles fetched...")
            fetch_since = candles[-1][0] + 14400000  # 4h in ms
            if len(candles) < 1000:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"    CCXT error: {e}")
            break

    stored = 0
    for c in all_candles:
        ts = int(c[0] / 1000)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ohlcv_4h
                (asset, timestamp, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (asset, ts, dt, c[1], c[2], c[3], c[4], c[5]))
            stored += 1
        except Exception:
            pass

    conn.commit()
    print(f"    Stored {stored} 4h candles")


def collect_ohlcv_1m(asset, days, conn):
    """Collect 1-minute OHLCV for high-frequency mean-reversion models.

    Binance actually serves 1-min klines at least 730 days back (verified
    2026-04-22 via scripts/test_binance_1m_history.py). The prior ~30-day
    retention assumption was incorrect. Lookback now capped at 180 days to
    balance coverage with fetch time (180d = ~259K candles per asset).
    For a one-time bigger backfill, bump the cap locally.
    """
    print(f"\n  Collecting 1m OHLCV for {asset} ({days} days)...")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    except ImportError:
        print(f"    pip install ccxt")
        return

    symbol = SUPPORTED_ASSETS[asset]["symbol"]
    since = int((datetime.now(timezone.utc) - timedelta(days=min(days, 180))).timestamp() * 1000)

    all_candles = []
    fetch_since = since
    batch = 0

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, "1m", since=fetch_since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            batch += 1
            if batch % 50 == 0:
                latest = datetime.fromtimestamp(
                    candles[-1][0] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M")
                print(f"    {len(all_candles):>7d} candles fetched "
                      f"(latest: {latest}, batch {batch})")
            fetch_since = candles[-1][0] + 60000  # 1 min in ms
            if len(candles) < 1000:
                break
            time.sleep(0.3)  # Slightly more conservative rate limiting
        except Exception as e:
            print(f"    CCXT error at batch {batch}: {e}")
            time.sleep(2)
            break

    print(f"    Total fetched: {len(all_candles)} candles")

    stored = 0
    for c in all_candles:
        ts = int(c[0] / 1000)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ohlcv_1m
                (asset, timestamp, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (asset, ts, dt, c[1], c[2], c[3], c[4], c[5]))
            stored += 1
        except Exception:
            pass

    conn.commit()
    days_covered = len(all_candles) / 1440 if all_candles else 0
    print(f"    Stored {stored} 1m candles ({days_covered:.1f} days)")


def collect_fear_greed(days, conn):
    """Collect historical Fear & Greed Index."""
    print(f"\n  Collecting Fear & Greed Index ({days} days)...")

    try:
        r = requests.get(f"https://api.alternative.me/fng/",
                         params={"limit": days, "format": "json"},
                         timeout=15)
        data = r.json().get("data", [])

        stored = 0
        for d in data:
            ts = int(d.get("timestamp", 0))
            date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO fear_greed
                    (timestamp, date, value, classification)
                    VALUES (?, ?, ?, ?)
                """, (ts, date, int(d.get("value", 50)),
                      d.get("value_classification", "Neutral")))
                stored += 1
            except Exception:
                pass

        conn.commit()
        print(f"    Stored {stored} days of Fear & Greed data")
        if data:
            print(f"    Range: {data[-1].get('timestamp', '?')} to {data[0].get('timestamp', '?')}")
            print(f"    Latest: {data[0].get('value', '?')} ({data[0].get('value_classification', '?')})")

    except Exception as e:
        print(f"    Error: {e}")


def collect_funding_rates(asset, days, conn):
    """Collect funding rate history from Binance."""
    print(f"\n  Collecting funding rates for {asset} ({days} days)...")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    except ImportError:
        print(f"    pip install ccxt")
        return

    symbol = SUPPORTED_ASSETS[asset]["perp"]
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_rates = []
    fetch_since = since

    while True:
        try:
            rates = exchange.fetch_funding_rate_history(symbol, since=fetch_since, limit=1000)
            if not rates:
                break
            all_rates.extend(rates)
            if len(all_rates) % 3000 == 0:
                print(f"    {len(all_rates)} rates fetched...")
            fetch_since = rates[-1]["timestamp"] + 1
            if len(rates) < 1000:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"    Funding rate error: {e}")
            break

    stored = 0
    for r in all_rates:
        ts = int(r["timestamp"] / 1000)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            conn.execute("""
                INSERT OR REPLACE INTO funding_rates
                (asset, timestamp, datetime, funding_rate)
                VALUES (?, ?, ?, ?)
            """, (asset, ts, dt, r.get("fundingRate", 0)))
            stored += 1
        except Exception:
            pass

    conn.commit()
    print(f"    Stored {stored} funding rate observations")


def collect_onchain_btc(days, conn):
    """Collect BTC on-chain metrics from blockchain.info."""
    print(f"\n  Collecting BTC on-chain data ({days} days)...")

    metrics = {
        "n-unique-addresses": "active_addresses",
        "n-transactions": "transaction_count",
        "hash-rate": "hash_rate",
        "difficulty": "difficulty",
        "avg-block-size": "block_size",
        "market-cap": "market_cap",
    }

    timespan = f"{days}days"
    all_data = {}

    for api_name, col_name in metrics.items():
        try:
            r = requests.get(f"https://api.blockchain.info/charts/{api_name}",
                             params={"timespan": timespan, "format": "json",
                                     "rollingAverage": "24hours"},
                             timeout=30)

            if r.status_code == 200:
                data = r.json()
                values = data.get("values", [])
                for v in values:
                    ts = v.get("x", 0)
                    date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                    if date not in all_data:
                        all_data[date] = {}
                    all_data[date][col_name] = v.get("y", 0)
                print(f"    {col_name}: {len(values)} data points")
            else:
                print(f"    {col_name}: HTTP {r.status_code}")

            time.sleep(1)  # Rate limit

        except Exception as e:
            print(f"    {col_name}: error {e}")

    # Store
    stored = 0
    for date, metrics_data in all_data.items():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO onchain_btc
                (date, active_addresses, transaction_count, hash_rate,
                 difficulty, block_size, market_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (date,
                  metrics_data.get("active_addresses", 0),
                  metrics_data.get("transaction_count", 0),
                  metrics_data.get("hash_rate", 0),
                  metrics_data.get("difficulty", 0),
                  metrics_data.get("block_size", 0),
                  metrics_data.get("market_cap", 0)))
            stored += 1
        except Exception:
            pass

    conn.commit()
    print(f"    Stored {stored} days of on-chain data")


def collect_market_data(asset, conn):
    """Collect current market data from CoinGecko."""
    print(f"\n  Collecting market data for {asset}...")

    cg_id = SUPPORTED_ASSETS[asset]["coingecko_id"]

    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}", params={
            "localization": "false", "tickers": "false",
            "community_data": "false", "developer_data": "false",
        }, timeout=15)

        if r.status_code == 200:
            data = r.json()
            md = data.get("market_data", {})
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            conn.execute("""
                INSERT OR REPLACE INTO market_data
                (asset, date, market_cap, total_volume, circulating_supply,
                 total_supply, ath, ath_change_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (asset, today,
                  md.get("market_cap", {}).get("usd", 0),
                  md.get("total_volume", {}).get("usd", 0),
                  md.get("circulating_supply", 0),
                  md.get("total_supply", 0),
                  md.get("ath", {}).get("usd", 0),
                  md.get("ath_change_percentage", {}).get("usd", 0)))
            conn.commit()
            print(f"    Price: ${md.get('current_price', {}).get('usd', 0):,.2f}")
            print(f"    Market cap: ${md.get('market_cap', {}).get('usd', 0):,.0f}")
            print(f"    ATH distance: {md.get('ath_change_percentage', {}).get('usd', 0):.1f}%")
        else:
            print(f"    CoinGecko HTTP {r.status_code}")

    except Exception as e:
        print(f"    Error: {e}")


# ===================================================================
# ORDER BOOK SNAPSHOTS
# ===================================================================

OB_SYMBOL_MAP = {"BTC": "BTC/USDT", "ETH": "ETH/USDT"}


def collect_order_book_snapshot(asset, exchange, conn):
    """Fetch current Binance order book for `asset` and insert one row.

    Returns (rows_inserted, error_msg). Never raises; transient API failures
    are reported to the caller so the outer loop can continue.
    """
    if asset not in OB_SYMBOL_MAP:
        return (0, f"unsupported asset {asset}")

    symbol = OB_SYMBOL_MAP[asset]

    try:
        ob = exchange.fetch_order_book(symbol, limit=10)
    except Exception as e:
        return (0, f"fetch_order_book: {type(e).__name__}: {e}")

    ts_ms = ob.get("timestamp") or int(time.time() * 1000)
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

    bids = list(ob.get("bids", []))[:10]
    asks = list(ob.get("asks", []))[:10]

    if not bids or not asks:
        return (0, "empty order book")

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    spread_bps = (spread / mid) * 10000 if mid else 0

    while len(bids) < 10:
        bids.append([0.0, 0.0])
    while len(asks) < 10:
        asks.append([0.0, 0.0])

    bid_top10 = sum(b[1] for b in bids)
    ask_top10 = sum(a[1] for a in asks)
    denom = bid_top10 + ask_top10
    imbalance = (bid_top10 - ask_top10) / denom if denom > 0 else 0

    bid_flat = [v for b in bids for v in b]
    ask_flat = [v for a in asks for v in a]

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO order_book_snapshots (
                asset, timestamp, datetime, mid_price, best_bid, best_ask,
                spread, spread_bps,
                bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3,
                bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6,
                bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9,
                bid_price_10, bid_vol_10,
                ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3,
                ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6,
                ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9,
                ask_price_10, ask_vol_10,
                bid_volume_top10, ask_volume_top10, order_imbalance_top10
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?)
        """, [
            asset, ts_ms // 1000, dt, mid, best_bid, best_ask, spread, spread_bps,
            *bid_flat, *ask_flat,
            bid_top10, ask_top10, imbalance
        ])
        conn.commit()
        return (cursor.rowcount, None)
    except Exception as e:
        return (0, f"insert: {type(e).__name__}: {e}")


def cmd_collect_order_book(args):
    """One-shot: collect one snapshot per requested asset."""
    assets = [a.upper() for a in args.assets]
    print(f"\n  ORDER BOOK SNAPSHOT -- {', '.join(assets)}")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    except ImportError:
        print("    pip install ccxt")
        return

    conn = init_db()
    for asset in assets:
        rows, err = collect_order_book_snapshot(asset, exchange, conn)
        if err:
            print(f"    {asset}: skipped -- {err}")
        else:
            print(f"    {asset}: inserted {rows} row(s)")
    conn.close()


def cmd_collect_order_book_loop(args):
    """Continuous polling loop. Meant for the scheduled task bat script.

    Runs until --duration seconds elapse (default: forever). Handles
    KeyboardInterrupt / SIGTERM cleanly by exiting the loop.
    """
    assets = [a.upper() for a in args.assets]
    interval = max(1, int(args.interval))
    duration = args.duration

    print(f"  Order book loop: assets={assets} interval={interval}s "
          f"duration={'forever' if duration is None else f'{duration}s'}")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    except ImportError:
        print("    pip install ccxt")
        return

    conn = init_db()
    t0 = time.time()
    totals = {a: 0 for a in assets}
    errors = {a: 0 for a in assets}
    iterations = 0

    try:
        while True:
            loop_start = time.time()
            iterations += 1
            for asset in assets:
                rows, err = collect_order_book_snapshot(asset, exchange, conn)
                if err:
                    errors[asset] += 1
                else:
                    totals[asset] += rows

            if iterations % 6 == 0:
                elapsed = int(time.time() - t0)
                summary = " ".join(
                    f"{a}={totals[a]}r/{errors[a]}e" for a in assets)
                print(f"    [{elapsed:>5d}s iter={iterations}] {summary}",
                      flush=True)

            if duration is not None and (time.time() - t0) >= duration:
                break

            sleep_for = interval - (time.time() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n  Interrupted by user -- exiting cleanly")

    elapsed = int(time.time() - t0)
    print(f"\n  Loop finished: {iterations} iterations over {elapsed}s")
    for a in assets:
        print(f"    {a}: {totals[a]} rows inserted, {errors[a]} errors")
    conn.close()


# ===================================================================
# TRADE FLOW (aggressor-side tagged trades)
# ===================================================================

TRADES_SYMBOL_MAP = {"BTC": "BTC/USDT", "ETH": "ETH/USDT"}


def get_latest_trade_id(asset, conn):
    """Return most recent trade_id stored for asset, or None if none exist."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT MAX(trade_id) FROM trades WHERE asset = ?", (asset,))
    r = cursor.fetchone()
    return r[0] if r and r[0] is not None else None


def collect_recent_trades(asset, exchange, conn, last_trade_id=None):
    """Fetch recent trades for `asset` and insert rows.

    If last_trade_id is None, fetches the most recent 1000 trades.
    Otherwise uses Binance's fromId param to cursor forward.

    Returns (rows_inserted, latest_trade_id, error_msg).
    """
    if asset not in TRADES_SYMBOL_MAP:
        return (0, last_trade_id, f"unsupported asset {asset}")
    symbol = TRADES_SYMBOL_MAP[asset]

    params = {}
    if last_trade_id is not None:
        params["fromId"] = last_trade_id + 1

    try:
        trades = exchange.fetch_trades(symbol, limit=1000, params=params)
    except Exception as e:
        return (0, last_trade_id,
                f"fetch_trades: {type(e).__name__}: {e}")

    if not trades:
        return (0, last_trade_id, None)

    cursor = conn.cursor()
    inserted = 0
    max_id = last_trade_id if last_trade_id is not None else 0

    for tr in trades:
        try:
            trade_id = int(tr["id"])
            ts = int(tr["timestamp"])
            dt = tr["datetime"]
            price = float(tr["price"])
            amount = float(tr["amount"])
            quote_amount = price * amount
            side = tr.get("side") or "buy"
            # Binance 'isBuyerMaker' = True  => seller hit the bid (side='sell')
            # Binance 'isBuyerMaker' = False => buyer hit the ask (side='buy')
            is_buyer_maker = 1 if side == "sell" else 0
        except (KeyError, ValueError, TypeError):
            continue

        cursor.execute("""
            INSERT OR IGNORE INTO trades (
                asset, trade_id, timestamp, datetime, price, amount,
                quote_amount, is_buyer_maker, side
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (asset, trade_id, ts, dt, price, amount,
              quote_amount, is_buyer_maker, side))

        if cursor.rowcount > 0:
            inserted += 1
        if trade_id > max_id:
            max_id = trade_id

    conn.commit()
    return (inserted, max_id, None)


def cmd_collect_trades(args):
    """One-shot: pull the latest batch of trades per asset (up to 1000 each)."""
    assets = [a.upper() for a in args.assets]
    print(f"\n  TRADES SNAPSHOT -- {', '.join(assets)}")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    except ImportError:
        print("    pip install ccxt")
        return

    conn = init_db()
    for asset in assets:
        last_id = get_latest_trade_id(asset, conn)
        rows, max_id, err = collect_recent_trades(
            asset, exchange, conn, last_trade_id=last_id)
        if err:
            print(f"    {asset}: skipped -- {err}")
        else:
            print(f"    {asset}: inserted {rows} row(s); "
                  f"max_trade_id={max_id}")
    conn.close()


def cmd_collect_trades_loop(args):
    """Continuous loop with adaptive sleep.

    Normal interval between batches = args.interval. But if a batch comes back
    saturated (rows >= 1000, meaning we may have missed trades between calls),
    we immediately refetch without sleeping to catch up. Sleep to interval
    only when no asset saturated in the last iteration.
    """
    assets = [a.upper() for a in args.assets]
    interval = max(1, int(args.interval))
    duration = args.duration

    print(f"  Trades loop: assets={assets} interval={interval}s "
          f"duration={'forever' if duration is None else f'{duration}s'}")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    except ImportError:
        print("    pip install ccxt")
        return

    conn = init_db()

    last_ids = {a: get_latest_trade_id(a, conn) for a in assets}
    for a in assets:
        start = last_ids[a] if last_ids[a] is not None else "(latest)"
        print(f"    {a}: starting from trade_id > {start}", flush=True)

    t0 = time.time()
    totals = {a: 0 for a in assets}
    errors = {a: 0 for a in assets}
    iterations = 0

    try:
        while True:
            iterations += 1
            any_saturated = False
            for asset in assets:
                rows, max_id, err = collect_recent_trades(
                    asset, exchange, conn, last_trade_id=last_ids[asset])
                if err:
                    errors[asset] += 1
                    print(f"    [{asset}] iter {iterations} ERROR: {err}",
                          flush=True)
                    continue
                totals[asset] += rows
                if rows >= 1000:
                    any_saturated = True
                last_ids[asset] = max_id

            if iterations % 6 == 0:
                elapsed = int(time.time() - t0)
                summary = " ".join(
                    f"{a}={totals[a]}t/{errors[a]}e" for a in assets)
                print(f"    [{elapsed:>5d}s iter={iterations}] {summary}",
                      flush=True)

            if duration is not None and (time.time() - t0) >= duration:
                break

            # Adaptive: skip sleep if we hit 1000-cap to catch up fast
            if not any_saturated:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Interrupted by user -- exiting cleanly")

    elapsed = int(time.time() - t0)
    print(f"\n  Loop finished: {iterations} iterations over {elapsed}s")
    for a in assets:
        print(f"    {a}: {totals[a]} trades inserted, {errors[a]} errors")
    conn.close()


def cmd_detect_whales(args):
    """Run whale detection on the trades table for one asset."""
    from engines.whale_detector import (
        detect_single_trade_whales,
        detect_windowed_whales,
        summarize_whales,
    )

    asset = args.asset.upper()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        single = detect_single_trade_whales(
            asset, conn, lookback_minutes=args.lookback)
        windowed = detect_windowed_whales(
            asset, conn,
            lookback_minutes=args.lookback,
            window_seconds=args.window_seconds)
        summarize_whales(single, windowed, asset)
    finally:
        conn.close()


# ===================================================================
# COMMANDS
# ===================================================================

def cmd_collect_all(args):
    """Collect all data for an asset."""
    asset = args.asset.upper()
    days = getattr(args, "days", 900)

    if asset not in SUPPORTED_ASSETS:
        print(f"  Unsupported: {asset}. Use: {', '.join(SUPPORTED_ASSETS.keys())}")
        return

    conn = init_db()

    print(f"\n{'='*70}")
    print(f"  FULL DATA COLLECTION -- {asset}")
    print(f"  Days: {days} | Target: {days/365:.1f} years")
    print(f"{'='*70}")

    collect_ohlcv_daily(asset, days, conn)
    collect_ohlcv_4h(asset, min(days, 180), conn)  # 4h only last 6 months
    collect_ohlcv_1m(asset, min(days, 30), conn)   # 1m only last 30 days (Binance limit)
    collect_fear_greed(days, conn)
    collect_funding_rates(asset, min(days, 365), conn)

    if asset == "BTC":
        collect_onchain_btc(min(days, 365), conn)

    collect_market_data(asset, conn)

    # Status summary
    cmd_status_internal(conn)
    conn.close()

    print(f"\n{'='*70}")


def cmd_status(args):
    """Show data collection status."""
    conn = init_db()
    cmd_status_internal(conn)
    conn.close()


def cmd_status_internal(conn):
    """Internal status display."""
    print(f"\n  DATA COLLECTION STATUS:")
    print(f"  {'-'*50}")

    tables = {
        "ohlcv_daily": ("asset", "date"),
        "ohlcv_4h": ("asset", "datetime"),
        "ohlcv_1m": ("asset", "datetime"),
        "fear_greed": (None, "date"),
        "funding_rates": ("asset", "datetime"),
        "onchain_btc": (None, "date"),
        "market_data": ("asset", "date"),
        "order_book_snapshots": ("asset", "datetime"),
        "trades": ("asset", "datetime"),
    }

    for table, (asset_col, date_col) in tables.items():
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count > 0:
                if asset_col:
                    assets = conn.execute(
                        f"SELECT DISTINCT {asset_col} FROM {table}"
                    ).fetchall()
                    asset_str = ", ".join(a[0] for a in assets)
                else:
                    asset_str = "-"

                min_date = conn.execute(
                    f"SELECT MIN({date_col}) FROM {table}").fetchone()[0]
                max_date = conn.execute(
                    f"SELECT MAX({date_col}) FROM {table}").fetchone()[0]

                print(f"    {table:<20s} {count:>8d} rows | "
                      f"{min_date} to {max_date} | {asset_str}")
            else:
                print(f"    {table:<20s}        0 rows")
        except Exception:
            print(f"    {table:<20s}    (not created)")


def main():
    parser = argparse.ArgumentParser(description="Extended Crypto Data Collector")
    subs = parser.add_subparsers(dest="command")

    p_all = subs.add_parser("collect-all", help="Collect everything")
    p_all.add_argument("--asset", required=True)
    p_all.add_argument("--days", type=int, default=900)

    p_ohlcv = subs.add_parser("collect-ohlcv", help="Daily OHLCV")
    p_ohlcv.add_argument("--asset", required=True)
    p_ohlcv.add_argument("--days", type=int, default=900)

    p_4h = subs.add_parser("collect-ohlcv-4h", help="4h OHLCV")
    p_4h.add_argument("--asset", required=True)
    p_4h.add_argument("--days", type=int, default=180)

    p_1m = subs.add_parser("collect-ohlcv-1m", help="1-minute OHLCV (max ~30 days from Binance)")
    p_1m.add_argument("--asset", required=True)
    p_1m.add_argument("--days", type=int, default=30)

    p_fg = subs.add_parser("collect-fear-greed", help="Fear & Greed history")
    p_fg.add_argument("--days", type=int, default=900)

    p_fr = subs.add_parser("collect-funding", help="Funding rates")
    p_fr.add_argument("--asset", required=True)
    p_fr.add_argument("--days", type=int, default=365)

    p_oc = subs.add_parser("collect-onchain", help="BTC on-chain")
    p_oc.add_argument("--days", type=int, default=365)

    p_ob = subs.add_parser("collect-order-book",
                           help="Collect one order book snapshot per asset")
    p_ob.add_argument("--assets", nargs="+", default=["BTC", "ETH"])

    p_obc = subs.add_parser("collect-order-book-loop",
                            help="Run continuous order book collection")
    p_obc.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    p_obc.add_argument("--interval", type=int, default=10,
                       help="Seconds between snapshots (default 10)")
    p_obc.add_argument("--duration", type=int, default=None,
                       help="Total seconds before exit (default: forever)")

    p_t = subs.add_parser("collect-trades",
                          help="Collect latest batch of trades per asset")
    p_t.add_argument("--assets", nargs="+", default=["BTC", "ETH"])

    p_tl = subs.add_parser("collect-trades-loop",
                           help="Run continuous trade collection")
    p_tl.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    p_tl.add_argument("--interval", type=int, default=30,
                      help="Seconds between fetch batches (default 30)")
    p_tl.add_argument("--duration", type=int, default=None,
                      help="Total seconds before exit (default: forever)")

    p_dw = subs.add_parser("detect-whales",
                           help="Detect whale trades/windows from trades table")
    p_dw.add_argument("--asset", required=True, choices=["BTC", "ETH"])
    p_dw.add_argument("--lookback", type=int, default=60,
                      help="Lookback in minutes (default 60)")
    p_dw.add_argument("--window-seconds", dest="window_seconds",
                      type=int, default=30,
                      help="Aggregation window for windowed detection "
                           "(default 30)")

    subs.add_parser("status", help="Collection status")

    args = parser.parse_args()

    dispatch = {
        "collect-all": cmd_collect_all,
        "collect-ohlcv": lambda a: collect_ohlcv_daily(a.asset.upper(), a.days, init_db()),
        "collect-ohlcv-4h": lambda a: collect_ohlcv_4h(a.asset.upper(), a.days, init_db()),
        "collect-ohlcv-1m": lambda a: collect_ohlcv_1m(a.asset.upper(), a.days, init_db()),
        "collect-fear-greed": lambda a: collect_fear_greed(a.days, init_db()),
        "collect-funding": lambda a: collect_funding_rates(a.asset.upper(), a.days, init_db()),
        "collect-onchain": lambda a: collect_onchain_btc(a.days, init_db()),
        "collect-order-book": cmd_collect_order_book,
        "collect-order-book-loop": cmd_collect_order_book_loop,
        "collect-trades": cmd_collect_trades,
        "collect-trades-loop": cmd_collect_trades_loop,
        "detect-whales": cmd_detect_whales,
        "status": cmd_status,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
