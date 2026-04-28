"""
Whale trade detector -- Praxis v8.2.2

First consumer of engines.microstructure_utils. Detects institutional-size
activity in the `trades` table populated by PraxisTradesCollector.

Two detection modes:

1. Single-trade whales: individual trades where quote_amount >= always_floor.
   "Block trade" whales -- one entity deploying large capital in one market order.

2. Windowed-aggregate whales: rolling windows where total invested passes the
   tiered threshold test. Catches sustained accumulation that doesn't manifest
   as a single block but IS institutional in aggregate.

Default thresholds are engineered guesses scaled from options-market whale
conventions; expect to re-tune after first data review.
"""

import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from engines.microstructure_utils import tiered_threshold_detector


DEFAULT_THRESHOLDS = {
    "BTC": {
        "single_trade_always": 500_000,    # $500k single market order = always whale
        "single_trade_never":    1_000,    # below $1k = ignore even if statistical
        "window_always":       2_000_000,  # $2M in the window = always whale window
        "window_never":           10_000,  # below $10k in window = never whale
    },
    "ETH": {
        "single_trade_always": 200_000,
        "single_trade_never":      500,
        "window_always":         800_000,
        "window_never":            5_000,
    },
}


def detect_single_trade_whales(asset, conn, lookback_minutes=60, thresholds=None):
    """Find individual trades exceeding the single_trade_always threshold.

    Returns a DataFrame sorted by quote_amount descending (biggest first),
    columns: trade_id, timestamp, datetime, price, amount, quote_amount, side.
    """
    t = thresholds or DEFAULT_THRESHOLDS.get(asset, {})
    always = t.get("single_trade_always", 500_000)

    cutoff_ms = int(
        (datetime.now(tz=timezone.utc).timestamp() - lookback_minutes * 60) * 1000
    )

    query = """
        SELECT trade_id, timestamp, datetime, price, amount, quote_amount, side
        FROM trades
        WHERE asset = ?
          AND timestamp >= ?
          AND quote_amount >= ?
        ORDER BY quote_amount DESC
    """
    return pd.read_sql(query, conn, params=(asset, cutoff_ms, always))


def detect_windowed_whales(asset, conn, lookback_minutes=60,
                           window_seconds=30, thresholds=None):
    """Aggregate trades into time windows, detect whale windows via tiered threshold.

    Returns DataFrame (possibly empty) columns: window_start, window_end,
    trade_count, total_invested, buy_invested, sell_invested,
    aggressor_imbalance, detection_reason.
    """
    t = thresholds or DEFAULT_THRESHOLDS.get(asset, {})
    always = t.get("window_always", 2_000_000)
    never = t.get("window_never", 10_000)

    cutoff_ms = int(
        (datetime.now(tz=timezone.utc).timestamp() - lookback_minutes * 60) * 1000
    )

    query = """
        SELECT timestamp, quote_amount, side
        FROM trades
        WHERE asset = ? AND timestamp >= ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, conn, params=(asset, cutoff_ms))
    if df.empty:
        return pd.DataFrame()

    # Bucket trades into windows of window_seconds
    bucket_ms = window_seconds * 1000
    df["window_start_ms"] = (df["timestamp"] // bucket_ms) * bucket_ms

    # Aggregate per window.
    # NOTE: the buy_invested / sell_invested lambdas below rely on df's
    # original index being intact through groupby so df.loc[x.index, "side"]
    # selects the matching rows. If anyone adds df.reset_index() upstream
    # of this groupby, the lambdas break.
    windows = df.groupby("window_start_ms").agg(
        trade_count=("quote_amount", "count"),
        total_invested=("quote_amount", "sum"),
        buy_invested=(
            "quote_amount",
            lambda x: x[df.loc[x.index, "side"] == "buy"].sum(),
        ),
        sell_invested=(
            "quote_amount",
            lambda x: x[df.loc[x.index, "side"] == "sell"].sum(),
        ),
    ).reset_index()

    total_safe = windows["total_invested"].where(
        windows["total_invested"] > 0, 1
    )
    windows["aggressor_imbalance"] = (
        (windows["buy_invested"] - windows["sell_invested"]) / total_safe
    )

    windows["window_start"] = pd.to_datetime(
        windows["window_start_ms"], unit="ms", utc=True
    )
    windows["window_end"] = (
        windows["window_start"] + pd.Timedelta(seconds=window_seconds)
    )

    flags = tiered_threshold_detector(
        windows["total_invested"].to_numpy(),
        always_floor=always,
        never_floor=never,
    )
    windows["is_whale"] = flags

    windows["detection_reason"] = np.where(
        windows["total_invested"] >= always, "always_floor",
        np.where(flags, "statistical", "not_whale"),
    )

    whales = windows[windows["is_whale"]].copy()
    whales = whales.sort_values("total_invested", ascending=False)
    return whales[[
        "window_start", "window_end", "trade_count", "total_invested",
        "buy_invested", "sell_invested", "aggressor_imbalance",
        "detection_reason",
    ]]


def summarize_whales(single_df, window_df, asset):
    """Print a human-readable summary of whale detections."""
    print(f"\n=== Whale Detection Summary: {asset} ===")

    print(f"\nSingle-trade whales: {len(single_df)}")
    if len(single_df) > 0:
        total_single = single_df["quote_amount"].sum()
        buy_count = int((single_df["side"] == "buy").sum())
        sell_count = int((single_df["side"] == "sell").sum())
        print(f"  Total dollar volume: ${total_single:,.0f}")
        print(f"  Buy-initiated: {buy_count}, Sell-initiated: {sell_count}")
        print(f"  Top 5:")
        for _, row in single_df.head(5).iterrows():
            print(f"    {row['datetime']}  {row['side']:4s}  "
                  f"${row['quote_amount']:>12,.0f}  @ ${row['price']:>10,.2f}")

    print(f"\nWindowed whale events: {len(window_df)}")
    if len(window_df) > 0:
        print(f"  Top 5:")
        for _, row in window_df.head(5).iterrows():
            imb_pct = row["aggressor_imbalance"] * 100
            print(f"    {row['window_start'].strftime('%H:%M:%S')}  "
                  f"${row['total_invested']:>12,.0f}  "
                  f"{row['trade_count']:>4} trades  "
                  f"imb={imb_pct:+6.1f}%  ({row['detection_reason']})")
