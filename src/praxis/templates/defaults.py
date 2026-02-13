"""
User Code Template Defaults (Phase 4.12).

Concrete implementations of template ABCs from praxis.templates:
- SimulatedExecutionAdapter: paper/backtest execution
- YFinanceDataSource: Yahoo Finance data (when yfinance available)
- CSVDataSource: load from CSV files
- StaticUniverse: fixed instrument list construction
- CointegrationConstruction: Burgess-style successive regression construction
- MomentumSignal: simple momentum signal generator

These serve as:
1. Working defaults for quick starts
2. Reference implementations for users writing custom versions
3. Building blocks for the `praxis run` pipeline

Usage:
    from praxis.templates.defaults import CSVDataSource, MomentumSignal
    source = CSVDataSource()
    data = source.fetch(["AAPL", "GOOG"], "2023-01-01", "2024-01-01")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from praxis.templates import (
    SignalTemplate,
    SizingTemplate,
    ExecutionAdapterTemplate,
    DataSourceTemplate,
)


# ═══════════════════════════════════════════════════════════════════
#  Execution Adapters
# ═══════════════════════════════════════════════════════════════════

class SimulatedExecutionAdapter(ExecutionAdapterTemplate):
    """
    Simulated execution adapter for backtesting and paper trading.

    Orders are filled instantly at the given price with optional
    slippage and commission.
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
    ):
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self._connected = False
        self._orders: list[dict] = []
        self._order_counter = 0

    def connect(self) -> bool:
        self._connected = True
        return True

    def submit_order(self, order: dict) -> str:
        self._order_counter += 1
        order_id = f"sim_{self._order_counter:06d}"

        # Simulate fill
        price = order.get("price", 0)
        qty = order.get("quantity", 0)
        side = order.get("side", "buy")

        slippage = price * self.slippage_bps / 10_000
        if side == "buy":
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        commission = self.commission_per_share * abs(qty)

        self._orders.append({
            "order_id": order_id,
            "fill_price": fill_price,
            "commission": commission,
            **order,
        })
        return order_id

    @property
    def orders(self) -> list[dict]:
        return list(self._orders)


class LoggingExecutionAdapter(ExecutionAdapterTemplate):
    """
    Logging adapter that records all orders without executing.

    Useful for signal validation and dry runs.
    """

    def __init__(self):
        self._log: list[dict] = []

    def connect(self) -> bool:
        return True

    def submit_order(self, order: dict) -> str:
        order_id = f"log_{len(self._log):06d}"
        self._log.append({"order_id": order_id, **order})
        return order_id

    @property
    def log(self) -> list[dict]:
        return list(self._log)


# ═══════════════════════════════════════════════════════════════════
#  Data Sources
# ═══════════════════════════════════════════════════════════════════

class CSVDataSource(DataSourceTemplate):
    """
    Load price data from CSV files.

    Expected CSV format: date,open,high,low,close,volume
    One file per ticker: {data_dir}/{ticker}.csv
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        """
        Fetch data from CSV files.

        Returns dict of ticker → numpy array (date, OHLCV).
        """
        result = {}
        for ticker in tickers:
            path = self.data_dir / f"{ticker}.csv"
            if not path.exists():
                continue
            try:
                import csv
                rows = []
                with open(path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        d = row.get("date", row.get("Date", ""))
                        if start <= d <= end:
                            rows.append({
                                "date": d,
                                "open": float(row.get("open", row.get("Open", 0))),
                                "high": float(row.get("high", row.get("High", 0))),
                                "low": float(row.get("low", row.get("Low", 0))),
                                "close": float(row.get("close", row.get("Close", 0))),
                                "volume": float(row.get("volume", row.get("Volume", 0))),
                            })
                result[ticker] = rows
            except Exception:
                continue
        return result


class InMemoryDataSource(DataSourceTemplate):
    """
    In-memory data source for testing.

    Pre-loaded with numpy arrays.
    """

    def __init__(self):
        self._data: dict[str, np.ndarray] = {}

    def load(self, ticker: str, prices: np.ndarray) -> None:
        """Load price array for a ticker."""
        self._data[ticker] = np.asarray(prices)

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        return {t: self._data[t] for t in tickers if t in self._data}


class RandomWalkDataSource(DataSourceTemplate):
    """
    Generate random walk prices for testing/simulation.
    """

    def __init__(self, n_obs: int = 500, seed: int | None = 42):
        self.n_obs = n_obs
        self.seed = seed

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        rng = np.random.RandomState(self.seed)
        result = {}
        for ticker in tickers:
            returns = rng.randn(self.n_obs) * 0.02
            prices = 100 * np.exp(np.cumsum(returns))
            result[ticker] = prices
        return result


# ═══════════════════════════════════════════════════════════════════
#  Signal Generators
# ═══════════════════════════════════════════════════════════════════

class MomentumSignal(SignalTemplate):
    """
    Simple momentum signal: long if price > SMA, short if below.

    params:
        lookback (int): SMA lookback period. Default 20.
    """

    def generate(self, prices: Any, params: dict) -> Any:
        prices = np.asarray(prices, dtype=float).ravel()
        lookback = params.get("lookback", 20)
        n = len(prices)
        signals = np.zeros(n)

        for i in range(lookback, n):
            sma = np.mean(prices[i - lookback:i])
            if prices[i] > sma:
                signals[i] = 1.0
            elif prices[i] < sma:
                signals[i] = -1.0

        return signals


class MeanReversionSignal(SignalTemplate):
    """
    Z-score mean reversion signal.

    params:
        lookback (int): Z-score lookback. Default 60.
        entry (float): Entry z-score threshold. Default 2.0.
        exit (float): Exit z-score threshold. Default 0.5.
    """

    def generate(self, prices: Any, params: dict) -> Any:
        prices = np.asarray(prices, dtype=float).ravel()
        lookback = params.get("lookback", 60)
        entry = params.get("entry", 2.0)
        exit_t = params.get("exit", 0.5)
        n = len(prices)
        signals = np.zeros(n)

        for i in range(lookback, n):
            window = prices[i - lookback:i]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma < 1e-10:
                continue
            z = (prices[i] - mu) / sigma
            if z >= entry:
                signals[i] = -1.0  # Short
            elif z <= -entry:
                signals[i] = 1.0   # Long
            elif abs(z) <= exit_t:
                signals[i] = 0.0   # Exit

        return signals


class CrossoverSignal(SignalTemplate):
    """
    Dual SMA crossover signal.

    params:
        fast (int): Fast SMA period. Default 10.
        slow (int): Slow SMA period. Default 30.
    """

    def generate(self, prices: Any, params: dict) -> Any:
        prices = np.asarray(prices, dtype=float).ravel()
        fast_p = params.get("fast", 10)
        slow_p = params.get("slow", 30)
        n = len(prices)
        signals = np.zeros(n)

        for i in range(slow_p, n):
            fast_sma = np.mean(prices[i - fast_p:i])
            slow_sma = np.mean(prices[i - slow_p:i])
            if fast_sma > slow_sma:
                signals[i] = 1.0
            elif fast_sma < slow_sma:
                signals[i] = -1.0

        return signals


# ═══════════════════════════════════════════════════════════════════
#  Sizing
# ═══════════════════════════════════════════════════════════════════

class FixedFractionSizing(SizingTemplate):
    """
    Fixed fraction of equity per trade.

    params:
        fraction (float): Fraction of equity. Default 0.02.
    """

    def size(self, signals: Any, params: dict) -> Any:
        signals = np.asarray(signals, dtype=float).ravel()
        fraction = params.get("fraction", 0.02)
        return signals * fraction


class VolTargetSizing(SizingTemplate):
    """
    Volatility-targeted sizing.

    params:
        target_vol (float): Target annualized vol. Default 0.15.
        vol_lookback (int): Lookback for vol estimation. Default 20.
    """

    def size(self, signals: Any, params: dict) -> Any:
        signals = np.asarray(signals, dtype=float).ravel()
        target_vol = params.get("target_vol", 0.15)
        vol_lookback = params.get("vol_lookback", 20)
        prices = params.get("prices", None)

        if prices is None:
            return signals * 0.02

        prices = np.asarray(prices, dtype=float).ravel()
        n = len(signals)
        sized = np.zeros(n)

        for i in range(vol_lookback + 1, n):
            returns = np.diff(prices[i - vol_lookback:i]) / prices[i - vol_lookback:i - 1]
            vol = np.std(returns) * np.sqrt(252)
            if vol > 1e-10:
                scalar = target_vol / vol
                sized[i] = signals[i] * min(scalar, 2.0)  # Cap at 2x

        return sized
