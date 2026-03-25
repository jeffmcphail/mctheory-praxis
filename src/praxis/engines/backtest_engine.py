"""
backtest_engine.py
==================
Simulates trades from strategy signals and streams bar-by-bar frames
via an async generator for the WebSocket handler.

Trade model:
  - Long-only (spot crypto)
  - Entry at close of entry bar
  - Exit at close of exit bar
  - 1 position at a time (no pyramiding)
  - No transaction costs in v1 (clearly noted in stats)

Each yielded frame is a dict ready for JSON serialization.
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import AsyncGenerator

import numpy as np
import pandas as pd

from engines.mcb_strategies.base import MCBStrategy


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_bar: int
    entry_time: str
    entry_price: float
    exit_bar: int | None = None
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def pnl_pct(self) -> float | None:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price * 100.0

    @property
    def pnl_r(self) -> float | None:
        """Return in R (multiple of entry price movement)."""
        return self.pnl_pct

    def to_dict(self) -> dict:
        return {
            "entry_time":  self.entry_time,
            "entry_price": round(self.entry_price, 4),
            "exit_time":   self.exit_time,
            "exit_price":  round(self.exit_price, 4) if self.exit_price else None,
            "exit_reason": self.exit_reason,
            "pnl_pct":     round(self.pnl_pct, 3) if self.pnl_pct is not None else None,
            "is_open":     self.is_open,
        }


# ---------------------------------------------------------------------------
# Running stats tracker
# ---------------------------------------------------------------------------

class StatsTracker:
    def __init__(self):
        self.equity: float = 100.0          # Start at 100 (index)
        self.peak_equity: float = 100.0
        self.trades: list[Trade] = []
        self.daily_returns: list[float] = []

    def close_trade(self, trade: Trade):
        pct = trade.pnl_pct or 0.0
        multiplier = 1.0 + pct / 100.0
        self.equity *= multiplier
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_returns.append(pct / 100.0)

    def open_pnl_pct(self, current_price: float) -> float | None:
        open_trade = next((t for t in self.trades if t.is_open), None)
        if open_trade is None:
            return None
        return (current_price - open_trade.entry_price) / open_trade.entry_price * 100.0

    @property
    def completed_trades(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_open]

    @property
    def total_return_pct(self) -> float:
        return self.equity - 100.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.daily_returns:
            return 0.0
        eq = 100.0
        peak = 100.0
        max_dd = 0.0
        for r in self.daily_returns:
            eq *= (1 + r)
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100.0
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def win_rate(self) -> float:
        ct = self.completed_trades
        if not ct:
            return 0.0
        wins = sum(1 for t in ct if (t.pnl_pct or 0) > 0)
        return wins / len(ct) * 100.0

    @property
    def sharpe(self) -> float:
        if len(self.daily_returns) < 5:
            return 0.0
        arr = np.array(self.daily_returns)
        std = arr.std()
        if std == 0:
            return 0.0
        # Annualize assuming ~365 "trading periods" per year for crypto
        return float(arr.mean() / std * math.sqrt(365))

    @property
    def avg_win_pct(self) -> float:
        wins = [t.pnl_pct for t in self.completed_trades if (t.pnl_pct or 0) > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss_pct(self) -> float:
        losses = [t.pnl_pct for t in self.completed_trades if (t.pnl_pct or 0) <= 0]
        return float(np.mean(losses)) if losses else 0.0

    def snapshot(self, current_price: float | None = None) -> dict:
        ct = self.completed_trades
        open_pnl = self.open_pnl_pct(current_price) if current_price else None
        return {
            "total_return_pct": round(self.total_return_pct, 2),
            "equity":           round(self.equity, 2),
            "total_trades":     len(ct),
            "open_trades":      len(self.trades) - len(ct),
            "win_rate":         round(self.win_rate, 1),
            "sharpe":           round(self.sharpe, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "avg_win_pct":      round(self.avg_win_pct, 2),
            "avg_loss_pct":     round(self.avg_loss_pct, 2),
            "open_pnl_pct":     round(open_pnl, 2) if open_pnl is not None else None,
            "in_position":      any(t.is_open for t in self.trades),
        }


# ---------------------------------------------------------------------------
# Main streaming generator
# ---------------------------------------------------------------------------

def _ts_str(idx) -> str:
    """Convert pandas Timestamp index value to ISO string."""
    ts = pd.Timestamp(idx)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.isoformat()


def _unix(idx) -> int:
    """Convert index to Unix timestamp (seconds) for Lightweight Charts."""
    ts = pd.Timestamp(idx)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


async def run_backtest_streaming(
    df: pd.DataFrame,
    strategy: MCBStrategy,
    replay_delay_ms: int = 30,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields one frame per bar.

    Parameters
    ----------
    df              : MCb-calculated DataFrame (output of MarketCipherB.calculate)
    strategy        : Instantiated MCBStrategy
    replay_delay_ms : ms to sleep between frames (0 = as fast as possible)

    Yields
    ------
    dict frames with type "bar", "trade_event", or "final_stats"
    """
    # Apply strategy signals
    df = strategy.generate_signals(df)

    stats = StatsTracker()
    open_trade: Trade | None = None
    delay = replay_delay_ms / 1000.0

    total_bars = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        # ---- Trade simulation ----
        trade_event = None
        close_px = float(row["close"])

        # Exit check (before entry — no same-bar reversal)
        if open_trade is not None and row["exit_signal"]:
            open_trade.exit_bar   = i
            open_trade.exit_time  = _ts_str(idx)
            open_trade.exit_price = close_px
            open_trade.exit_reason = str(row.get("signal_label", "EXIT"))
            stats.close_trade(open_trade)
            trade_event = {
                "type":       "EXIT",
                "price":      close_px,
                "pnl_pct":    round(open_trade.pnl_pct or 0, 3),
                "entry_time": open_trade.entry_time,
                "trade":      open_trade.to_dict(),
            }
            open_trade = None

        # Entry check
        if open_trade is None and row["entry"]:
            open_trade = Trade(
                entry_bar=i,
                entry_time=_ts_str(idx),
                entry_price=close_px,
            )
            stats.trades.append(open_trade)
            trade_event = {
                "type":  "ENTRY",
                "price": close_px,
            }

        # ---- Build frame ----
        frame = {
            "type": "bar",
            "i":    i,
            "total": total_bars,
            "time": _unix(idx),
            "time_str": _ts_str(idx),

            # OHLCV
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "close":  close_px,
            "volume": float(row["volume"]),

            # MCb oscillator values
            "wt1":      _safe_float(row.get("wt1")),
            "wt2":      _safe_float(row.get("wt2")),
            "rsi_mfi":  _safe_float(row.get("rsi_mfi")),
            "rsi":      _safe_float(row.get("rsi")),
            "stoch_k":  _safe_float(row.get("stoch_k")),
            "stoch_color": int(row.get("stoch_color", 0)),

            # Signals
            "buy_dot":   bool(row.get("buy_dot", False)),
            "sell_dot":  bool(row.get("sell_dot", False)),
            "gold_dot":  bool(row.get("gold_dot", False)),
            "bull_div":  bool(row.get("bull_div", False)),
            "bear_div":  bool(row.get("bear_div", False)),

            # Strategy signals
            "entry":        bool(row.get("entry", False)),
            "exit_signal":  bool(row.get("exit_signal", False)),
            "signal_label": str(row.get("signal_label", "")),

            # Trade event (null if none this bar)
            "trade_event": trade_event,

            # Running stats
            "stats": stats.snapshot(close_px),
        }

        yield frame

        if delay > 0:
            await asyncio.sleep(delay)

    # ---- Close any open trade at last close ----
    if open_trade is not None:
        last_idx = df.index[-1]
        last_close = float(df["close"].iloc[-1])
        open_trade.exit_bar   = total_bars - 1
        open_trade.exit_time  = _ts_str(last_idx)
        open_trade.exit_price = last_close
        open_trade.exit_reason = "END OF DATA"
        stats.close_trade(open_trade)

    # ---- Final stats ----
    final = {
        "type": "final_stats",
        "stats": stats.snapshot(),
        "trades": [t.to_dict() for t in stats.trades],
    }
    yield final


def _safe_float(val) -> float | None:
    """Return float or None (handles NaN from pandas)."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None
