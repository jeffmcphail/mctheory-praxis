"""
Position Sizing (Phase 3.2).

VolatilityTarget: Scale position sizes to target a specific annualized volatility.
EqualWeight: Simple equal-weight allocation.

Usage:
    sizer = VolatilityTarget()
    weights = sizer.size(prices, signal, params)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl


class Sizer(ABC):
    """Base position sizer."""

    @abstractmethod
    def size(self, prices: pl.DataFrame, signal: pl.Series, params: dict[str, Any]) -> pl.Series:
        """
        Compute position weights from signal.

        Args:
            prices: Price data with at minimum 'close'.
            signal: Signal series {-1, 0, +1}.
            params: Sizing parameters.

        Returns:
            pl.Series of position weights (fractional, can be > 1 for leverage).
        """
        ...

    @property
    def name(self) -> str:
        return type(self).__name__


class VolatilityTarget(Sizer):
    """
    §6.6: Scale position size to target annualized volatility.

    weight = target_vol / (realized_vol * sqrt(periods_per_year)) * signal

    Params:
        target_vol: float (e.g., 0.10 for 10% annualized)
        vol_lookback: int (rolling window for realized vol)
        periods_per_year: int (252 for daily, 252*6.5*60 for 1-min intraday)
        price_column: str (default 'close')
        max_leverage: float (cap, default 3.0)
    """

    def size(self, prices: pl.DataFrame, signal: pl.Series, params: dict[str, Any]) -> pl.Series:
        target_vol = params.get("target_vol", 0.10)
        vol_lookback = params.get("vol_lookback", 60)
        periods = params.get("periods_per_year", 252)
        col = params.get("price_column", "close")
        max_lev = params.get("max_leverage", 3.0)

        close = prices[col].to_numpy().astype(np.float64)
        sig = signal.to_numpy().astype(np.float64)

        # Returns
        returns = np.diff(close) / close[:-1]
        returns = np.insert(returns, 0, 0.0)

        # Rolling realized vol (annualized)
        real_vol = np.full(len(returns), np.nan)
        for i in range(vol_lookback, len(returns)):
            real_vol[i] = np.std(returns[i - vol_lookback:i], ddof=1) * np.sqrt(periods)

        # Scale
        weights = np.where(
            (real_vol > 0) & ~np.isnan(real_vol),
            np.clip(target_vol / real_vol, -max_lev, max_lev) * sig,
            sig,  # Fallback: no scaling if vol unavailable
        )

        return pl.Series("weight", weights)


class EqualWeight(Sizer):
    """Simple equal-weight: signal * fixed_weight."""

    def size(self, prices: pl.DataFrame, signal: pl.Series, params: dict[str, Any]) -> pl.Series:
        weight = params.get("weight", 1.0)
        return signal.cast(pl.Float64) * weight
