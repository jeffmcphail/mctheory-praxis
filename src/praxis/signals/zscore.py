"""
Z-Score Spread Signals (Phase 3.1).

ZScoreSpread: The core Chan CPO signal.
  spread = close_A - weight * close_B
  zscore = (spread - EMA(spread)) / EMA_std(spread)
  Entry: zscore >= threshold → short, zscore <= -threshold → long
  Exit: zscore <= exit_threshold → close short, zscore >= -exit_threshold → close long
  Positions carry forward (ffill) until exit signal.

BollingerZScore: Bollinger Band z-score on a single series.

These MUST match the original pair_trade_gld_gdx() output exactly
for the 5% validation criterion (Milestone 3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from praxis.signals import Signal
from praxis.logger.core import PraxisLogger


class ZScoreSpread(Signal):
    """
    Z-Score of a weighted spread — the Chan CPO pair trade signal.

    Matches original code:
        spread = close_GLD - gdx_weight * close_GDX
        spread_ema_mean = spread.ewm(span=lookback).mean()
        spread_ema_std = spread.ewm(span=lookback).std()
        zscore = (spread - spread_ema_mean) / spread_ema_std

    Entry/exit with position carry-forward:
        zscore >= entry → short (-1)
        zscore <= -entry → long (+1)
        zscore <= exit_threshold → close short (0)
        zscore >= -exit_threshold → close long (0)
        ffill positions

    Required params:
        weight: float (gdx_weight in original)
        lookback: int (EMA span)
        entry_threshold: float
        exit_threshold_fraction: float (exit = fraction * entry)

    Required columns:
        close_a, close_b (or custom via params)
    """

    def generate(self, prices: pl.DataFrame, params: dict[str, Any]) -> pl.Series:
        log = PraxisLogger.instance()

        weight = params["weight"]
        lookback = params["lookback"]
        entry_threshold = params["entry_threshold"]
        exit_frac = params.get("exit_threshold_fraction", -0.6)
        exit_threshold = exit_frac * entry_threshold
        col_a = params.get("close_a", "close_a")
        col_b = params.get("close_b", "close_b")

        log.debug(
            f"ZScoreSpread: weight={weight}, lookback={lookback}, "
            f"entry={entry_threshold}, exit_frac={exit_frac}",
            tags={"compute.signals"},
        )

        a = prices[col_a].to_numpy().astype(np.float64)
        b = prices[col_b].to_numpy().astype(np.float64)

        # Spread
        spread = a - weight * b

        # EMA mean and std (matching pandas ewm(span=lookback, adjust=False))
        ema_mean = _ewm_mean(spread, lookback)
        ema_std = _ewm_std(spread, lookback)

        # Z-score
        with np.errstate(invalid="ignore"):
            zscore = np.where(ema_std > 0, (spread - ema_mean) / ema_std, 0.0)

        # Position signals (matching original exactly)
        n = len(zscore)
        positions_long = np.zeros(n)
        positions_short = np.zeros(n)

        # Set entry/exit signals
        positions_short[zscore >= entry_threshold] = -1
        positions_long[zscore <= -entry_threshold] = 1
        positions_short[zscore <= exit_threshold] = 0  # Exit short
        positions_long[zscore >= -exit_threshold] = 0   # Exit long

        # Forward fill (carry positions until exit)
        positions_long = _ffill_signal(positions_long, zscore, entry_threshold, exit_threshold, side="long")
        positions_short = _ffill_signal(positions_short, zscore, entry_threshold, exit_threshold, side="short")

        # Combined position
        signal = (positions_long + positions_short).astype(np.int32)

        log.debug(
            f"ZScoreSpread: {(signal == 1).sum()} long, "
            f"{(signal == -1).sum()} short, {(signal == 0).sum()} flat",
            tags={"compute.signals"},
        )

        return pl.Series("signal", signal)

    def compute_zscore(self, prices: pl.DataFrame, params: dict[str, Any]) -> pl.Series:
        """Return raw z-score (for feature generation / analysis)."""
        weight = params["weight"]
        lookback = params["lookback"]
        col_a = params.get("close_a", "close_a")
        col_b = params.get("close_b", "close_b")

        a = prices[col_a].to_numpy().astype(np.float64)
        b = prices[col_b].to_numpy().astype(np.float64)

        spread = a - weight * b
        ema_mean = _ewm_mean(spread, lookback)
        ema_std = _ewm_std(spread, lookback)
        with np.errstate(invalid="ignore"):
            zscore = np.where(ema_std > 0, (spread - ema_mean) / ema_std, np.nan)

        return pl.Series("zscore", zscore)


class BollingerZScore(Signal):
    """
    Bollinger Band Z-Score on a single price series.

    zscore = (price - SMA(price, period)) / rolling_std(price, period)

    +1 when zscore < -threshold (mean reversion buy)
    -1 when zscore > +threshold (mean reversion sell)
     0 otherwise
    """

    def generate(self, prices: pl.DataFrame, params: dict[str, Any]) -> pl.Series:
        period = params.get("period", 20)
        threshold = params.get("threshold", 2.0)
        col = params.get("price_column", "close")

        price = prices[col].to_numpy().astype(np.float64)

        sma = _rolling_mean(price, period)
        std = _rolling_std(price, period)

        with np.errstate(invalid="ignore"):
            zscore = np.where(std > 0, (price - sma) / std, 0.0)

        signal = np.where(zscore < -threshold, 1,
                 np.where(zscore > threshold, -1, 0)).astype(np.int32)

        return pl.Series("signal", signal)


# ═══════════════════════════════════════════════════════════════════
#  Numpy helpers (matching pandas exactly)
# ═══════════════════════════════════════════════════════════════════

def _ewm_mean(data: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential weighted mean matching pandas ewm(span=N, adjust=False).

    alpha = 2 / (span + 1)
    """
    alpha = 2.0 / (span + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _ewm_std(data: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential weighted std matching pandas ewm(span=N, adjust=False).std().
    """
    alpha = 2.0 / (span + 1)
    n = len(data)
    mean = _ewm_mean(data, span)
    result = np.zeros(n)

    # Recursive variance: var_t = (1-alpha) * (var_{t-1} + alpha * (x_t - mean_{t-1})^2)
    var = 0.0
    for i in range(1, n):
        diff = data[i] - mean[i - 1]
        var = (1 - alpha) * (var + alpha * diff * diff)
        result[i] = np.sqrt(var) if var > 0 else 0.0

    return result


def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean (NaN for incomplete windows)."""
    result = np.full_like(data, np.nan)
    for i in range(window - 1, len(data)):
        result[i] = np.mean(data[i - window + 1:i + 1])
    return result


def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling std (NaN for incomplete windows)."""
    result = np.full_like(data, np.nan)
    for i in range(window - 1, len(data)):
        result[i] = np.std(data[i - window + 1:i + 1], ddof=1)
    return result


def _ffill_signal(
    initial: np.ndarray,
    zscore: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    side: str,
) -> np.ndarray:
    """
    Forward-fill position signals matching pandas ffill behavior.

    This replicates the original code where positions are set at
    entry/exit points, then ffill carries them forward.
    """
    result = np.zeros(len(initial))
    pos = 0.0

    for i in range(len(initial)):
        z = zscore[i]
        if np.isnan(z):
            result[i] = pos
            continue

        if side == "long":
            if z <= -entry_threshold:
                pos = 1.0
            elif z >= -exit_threshold:
                pos = 0.0
        else:  # short
            if z >= entry_threshold:
                pos = -1.0
            elif z <= exit_threshold:
                pos = 0.0

        result[i] = pos

    return result
