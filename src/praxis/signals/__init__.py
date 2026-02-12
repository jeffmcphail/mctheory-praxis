"""
Signal generators.

Base class + Phase 1 implementations:
- SMACrossover (§7.2): Simple moving average crossover
- EMACrossover (§7.2): Exponential moving average crossover

Signals consume price data and produce a signal Series:
  +1 = long, -1 = short, 0 = neutral/flat

All signals follow the same interface so the registry can
resolve them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl

from praxis.logger.core import PraxisLogger
from praxis.logger.records import LogLevel


class Signal(ABC):
    """
    Base signal generator.

    Subclasses implement generate() which takes a DataFrame of price data
    and signal params, returning a signal Series of {-1, 0, +1}.
    """

    @abstractmethod
    def generate(self, prices: pl.DataFrame, params: dict[str, Any]) -> pl.Series:
        """
        Generate trading signals from price data.

        Args:
            prices: DataFrame with at minimum a 'close' column.
            params: Signal parameters from config (fast_period, slow_period, etc.)

        Returns:
            pl.Series of int values: +1 (long), -1 (short), 0 (flat).
            Length matches input prices.
        """
        ...

    @property
    def name(self) -> str:
        return type(self).__name__


class SMACrossover(Signal):
    """
    Simple Moving Average Crossover.

    Signal logic:
    - fast_sma > slow_sma → +1 (long)
    - fast_sma < slow_sma → -1 (short)
    - equal or NaN → 0 (flat)

    Required params: fast_period (int), slow_period (int)
    Optional: price_column (str, default 'close')
    """

    def generate(self, prices: pl.DataFrame, params: dict[str, Any]) -> pl.Series:
        log = PraxisLogger.instance()

        fast = params.get("fast_period", 10)
        slow = params.get("slow_period", 50)
        col = params.get("price_column", "close")

        log.debug(
            f"SMACrossover: fast={fast}, slow={slow}, bars={len(prices)}",
            tags={"compute.signals", "compute.signals.sma_crossover"},
            module="signals.trend", function="SMACrossover.generate",
            fast_period=fast, slow_period=slow,
        )

        if fast >= slow:
            log.warning(
                f"SMACrossover: fast_period ({fast}) >= slow_period ({slow})",
                tags={"compute.signals"},
            )

        price = prices[col]

        fast_sma = price.rolling_mean(fast)
        slow_sma = price.rolling_mean(slow)

        # Signal: fast > slow → +1, fast < slow → -1, else 0
        # Use select() to materialize the Expr into a Series
        signal = prices.select(
            pl.when(fast_sma > slow_sma).then(1)
              .when(fast_sma < slow_sma).then(-1)
              .otherwise(0)
              .alias("signal")
        )["signal"]

        log.debug(
            f"SMACrossover: generated {(signal == 1).sum()} long, "
            f"{(signal == -1).sum()} short, {(signal == 0).sum()} flat",
            tags={"compute.signals", "compute.signals.sma_crossover"},
        )

        return signal


class EMACrossover(Signal):
    """
    Exponential Moving Average Crossover.

    Same logic as SMA but uses EMA for faster response.

    Required params: fast_period (int), slow_period (int)
    Optional: price_column (str, default 'close')
    """

    def generate(self, prices: pl.DataFrame, params: dict[str, Any]) -> pl.Series:
        log = PraxisLogger.instance()

        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        col = params.get("price_column", "close")

        log.debug(
            f"EMACrossover: fast={fast}, slow={slow}, bars={len(prices)}",
            tags={"compute.signals", "compute.signals.ema_crossover"},
            module="signals.trend", function="EMACrossover.generate",
            fast_period=fast, slow_period=slow,
        )

        price = prices[col]

        fast_ema = price.ewm_mean(span=fast, ignore_nulls=True)
        slow_ema = price.ewm_mean(span=slow, ignore_nulls=True)

        signal = prices.select(
            pl.when(fast_ema > slow_ema).then(1)
              .when(fast_ema < slow_ema).then(-1)
              .otherwise(0)
              .alias("signal")
        )["signal"]

        log.debug(
            f"EMACrossover: generated {(signal == 1).sum()} long, "
            f"{(signal == -1).sum()} short, {(signal == 0).sum()} flat",
            tags={"compute.signals", "compute.signals.ema_crossover"},
        )

        return signal
