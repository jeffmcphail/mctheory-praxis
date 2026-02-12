"""
Position sizing functions.

Base class + Phase 1 implementation:
- FixedFraction: Allocate a fixed fraction of capital per trade.

Sizers consume signal + portfolio state and produce position sizes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl

from praxis.logger.core import PraxisLogger
from praxis.logger.records import LogLevel


class Sizer(ABC):
    """
    Base position sizer.

    Subclasses implement size() which takes signals and params,
    returning a Series of position sizes (fraction of capital).
    """

    @abstractmethod
    def size(self, signals: pl.Series, params: dict[str, Any]) -> pl.Series:
        """
        Compute position sizes from signals.

        Args:
            signals: Series of {-1, 0, +1} trading signals.
            params: Sizing parameters from config.

        Returns:
            pl.Series of float position sizes.
            +fraction = long, -fraction = short, 0 = flat.
        """
        ...

    @property
    def name(self) -> str:
        return type(self).__name__


class FixedFraction(Sizer):
    """
    Fixed Fraction Sizing.

    Allocates a constant fraction of capital for every signal.
    Signal * fraction * direction.

    Required params: fraction (float, 0.0 to 1.0)
    Optional: max_position_pct (float, caps absolute position)
    """

    def size(self, signals: pl.Series, params: dict[str, Any]) -> pl.Series:
        log = PraxisLogger.instance()

        fraction = params.get("fraction", 1.0)
        max_pct = params.get("max_position_pct", 1.0)

        log.debug(
            f"FixedFraction: fraction={fraction}, max_pct={max_pct}",
            tags={"compute.sizing"},
            module="sizing.fixed", function="FixedFraction.size",
            fraction=fraction, max_position_pct=max_pct,
        )

        if not 0.0 <= fraction <= 1.0:
            log.warning(
                f"FixedFraction: fraction {fraction} outside [0, 1]",
                tags={"compute.sizing"},
            )

        # Position = signal direction * fraction, clamped to max
        positions = (signals.cast(pl.Float64) * fraction).alias("position")

        # Clamp to max_position_pct
        positions = positions.clip(-max_pct, max_pct)

        long_count = (positions > 0).sum()
        short_count = (positions < 0).sum()
        log.debug(
            f"FixedFraction: {long_count} long, {short_count} short positions",
            tags={"compute.sizing"},
        )

        return positions
