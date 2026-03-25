"""
gui/mcb_studio/backend/data_feed.py
=====================================
Fetches OHLCV data from Binance via CCXT (no API key required).
"""

from datetime import datetime, timezone
import pandas as pd
import ccxt

SUPPORTED_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT",
]

SUPPORTED_INTERVALS = {
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

_exchange: ccxt.binance | None = None


def _get_exchange() -> ccxt.binance:
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({"enableRateLimit": True})
    return _exchange


def fetch_ohlcv(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Binance. Paginates automatically.
    Returns DataFrame with lowercase OHLCV columns, DatetimeIndex (UTC).
    """
    tf = SUPPORTED_INTERVALS.get(interval)
    if not tf:
        raise ValueError(f"Unsupported interval: {interval}")

    exchange = _get_exchange()

    since_ms = int(datetime.strptime(start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(datetime.strptime(end,   "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_bars = []
    cursor = since_ms
    while cursor < end_ms:
        bars = exchange.fetch_ohlcv(symbol, tf, since=cursor, limit=1000)
        if not bars:
            break
        bars = [b for b in bars if b[0] < end_ms]
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

    if not all_bars:
        raise ValueError(f"No data returned for {symbol} {interval} {start}→{end}")

    df = pd.DataFrame(all_bars,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df = df[df.index < pd.Timestamp(end, tz="UTC")]

    if df.empty:
        raise ValueError(f"Empty dataframe for {symbol} {interval}")

    return df
