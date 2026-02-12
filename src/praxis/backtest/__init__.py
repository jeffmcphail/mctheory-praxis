"""
Vectorized Backtest Engine (§9.1).

Numpy/Polars on full series. No fill simulation, no order book.
Suitable for research, parameter sweeps, and screening.

Pipeline:
    positions (from executor) + prices → returns → equity curve → metrics

Trade-on-next-bar: positions[t] determined by signal[t], but the
RETURN earned is prices[t+1]/prices[t]. This avoids look-ahead bias.

Metrics match the results STRUCT in fact_backtest_run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from praxis.logger.core import PraxisLogger


@dataclass
class BacktestMetrics:
    """
    Matches fact_backtest_run.results STRUCT.
    All metrics computed from equity curve and trade list.
    """
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    avg_holding_days: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_return": self.avg_trade_return,
            "avg_holding_days": self.avg_holding_days,
            "calmar_ratio": self.calmar_ratio,
            "volatility": self.volatility,
        }


@dataclass
class BacktestOutput:
    """Full backtest output: metrics + time series for analysis."""
    metrics: BacktestMetrics
    equity_curve: np.ndarray          # Cumulative equity (starts at 1.0)
    daily_returns: np.ndarray         # Strategy daily returns
    positions: np.ndarray             # Position series used
    trades: list[dict[str, Any]]      # Individual trade records
    bar_count: int = 0
    duration_seconds: float = 0.0


class VectorizedEngine:
    """
    §9.1: Vectorized backtest engine.

    Takes positions + prices → computes returns, equity curve, metrics.

    Assumptions:
    - Trade-on-next-bar: position[t] earns return from t→t+1
    - No transaction costs (Phase 2 adds slippage/commission)
    - Fully invested at position size (no cash drag modeling)
    - Daily frequency (annualization factor = 252)
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, annualization_factor: int = 252):
        self._ann_factor = annualization_factor
        self._log = PraxisLogger.instance()

    def run(
        self,
        positions: pl.Series | np.ndarray,
        prices: pl.Series | np.ndarray,
        initial_capital: float = 1.0,
    ) -> BacktestOutput:
        """
        Execute vectorized backtest.

        Args:
            positions: Position sizes per bar (+1 long, -1 short, 0 flat, fractional ok).
            prices: Price series (close prices).
            initial_capital: Starting capital (default 1.0 for returns-based).

        Returns:
            BacktestOutput with metrics, equity curve, trades.
        """
        import time
        start_time = time.monotonic()

        self._log.info(
            "VectorizedEngine: starting backtest",
            tags={"backtest", "trade_cycle"},
            bars=len(prices),
        )

        # ── Convert to numpy ──────────────────────────────────────
        pos = self._to_numpy(positions)
        px = self._to_numpy(prices)

        if len(pos) != len(px):
            raise ValueError(
                f"Position length ({len(pos)}) != price length ({len(px)})"
            )

        n = len(px)
        if n < 2:
            raise ValueError(f"Need at least 2 bars, got {n}")

        # ── Price returns ─────────────────────────────────────────
        # returns[t] = (px[t] - px[t-1]) / px[t-1]
        price_returns = np.diff(px) / px[:-1]  # length n-1

        self._log.debug(
            f"Price returns: n={len(price_returns)}, "
            f"mean={np.nanmean(price_returns):.6f}, "
            f"std={np.nanstd(price_returns):.6f}",
            tags={"backtest"},
        )

        # ── Strategy returns ──────────────────────────────────────
        # Trade-on-next-bar: pos[t] earns price_returns[t+1]
        # So strategy_returns[t] = pos[t-1] * price_returns[t]
        # pos[:-1] aligned with price_returns[:]
        # Result length: n-1
        strat_returns = pos[:-1] * price_returns

        # Replace NaN with 0 (from warmup period nulls)
        strat_returns = np.nan_to_num(strat_returns, nan=0.0)

        self._log.debug(
            f"Strategy returns: mean={np.mean(strat_returns):.6f}, "
            f"std={np.std(strat_returns):.6f}",
            tags={"backtest"},
        )

        # ── Equity curve ──────────────────────────────────────────
        equity = initial_capital * np.cumprod(1.0 + strat_returns)
        # Prepend initial capital
        equity = np.concatenate([[initial_capital], equity])

        # ── Metrics ───────────────────────────────────────────────
        trades = self._extract_trades(pos, px, strat_returns)
        metrics = self._compute_metrics(strat_returns, equity, trades)

        duration = time.monotonic() - start_time

        self._log.info(
            f"VectorizedEngine: complete — "
            f"return={metrics.total_return:.2%}, "
            f"sharpe={metrics.sharpe_ratio:.2f}, "
            f"drawdown={metrics.max_drawdown:.2%}, "
            f"trades={metrics.total_trades}",
            tags={"backtest", "trade_cycle"},
            total_return=metrics.total_return,
            sharpe=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            trades=metrics.total_trades,
        )

        return BacktestOutput(
            metrics=metrics,
            equity_curve=equity,
            daily_returns=strat_returns,
            positions=pos,
            trades=trades,
            bar_count=n,
            duration_seconds=duration,
        )

    # ── Metrics Computation ───────────────────────────────────────

    def _compute_metrics(
        self,
        returns: np.ndarray,
        equity: np.ndarray,
        trades: list[dict],
    ) -> BacktestMetrics:
        """Compute all metrics from returns and equity curve."""
        n = len(returns)
        if n == 0:
            return BacktestMetrics()

        # Total return
        total_return = equity[-1] / equity[0] - 1.0

        # Annualized return
        years = n / self._ann_factor
        if years > 0 and equity[-1] > 0:
            annualized_return = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0
        else:
            annualized_return = 0.0

        # Volatility (annualized)
        daily_vol = np.std(returns, ddof=1) if n > 1 else 0.0
        volatility = daily_vol * np.sqrt(self._ann_factor)

        # Sharpe ratio (assumes rf = 0)
        mean_daily = np.mean(returns)
        sharpe = (mean_daily / daily_vol * np.sqrt(self._ann_factor)) if daily_vol > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside = returns[returns < 0]
        downside_vol = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
        sortino = (mean_daily / downside_vol * np.sqrt(self._ann_factor)) if downside_vol > 0 else 0.0

        # Drawdown
        max_dd, max_dd_duration = self._compute_drawdown(equity)

        # Calmar ratio
        calmar = abs(annualized_return / max_dd) if max_dd != 0 else 0.0

        # Trade metrics
        total_trades = len(trades)
        if total_trades > 0:
            trade_returns = [t["return_pct"] for t in trades]
            winners = [r for r in trade_returns if r > 0]
            losers = [r for r in trade_returns if r <= 0]

            win_rate = len(winners) / total_trades
            avg_trade_return = np.mean(trade_returns)
            avg_holding = np.mean([t["duration_bars"] for t in trades])

            gross_profit = sum(winners) if winners else 0.0
            gross_loss = abs(sum(losers)) if losers else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        else:
            win_rate = 0.0
            avg_trade_return = 0.0
            avg_holding = 0.0
            profit_factor = 0.0

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            avg_holding_days=avg_holding,
            calmar_ratio=calmar,
            volatility=volatility,
        )

    def _compute_drawdown(self, equity: np.ndarray) -> tuple[float, int]:
        """
        Compute max drawdown and duration.

        Returns (max_drawdown_pct, max_duration_bars).
        max_drawdown is negative (e.g., -0.15 for 15% drawdown).
        """
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max

        max_dd = np.min(drawdowns)  # Most negative = worst drawdown

        # Duration: longest consecutive period below previous peak
        underwater = drawdowns < 0
        max_duration = 0
        current_duration = 0
        for uw in underwater:
            if uw:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, max_duration

    # ── Trade Extraction ──────────────────────────────────────────

    def _extract_trades(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        returns: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Extract individual trades from position changes.

        A trade starts when position changes from 0 to non-zero (or sign flip),
        and ends when position returns to 0 or flips sign.
        """
        trades = []
        n = len(positions)
        in_trade = False
        entry_bar = 0
        entry_price = 0.0
        trade_direction = 0
        cum_return = 0.0

        for i in range(n):
            current_pos = positions[i]
            current_sign = np.sign(current_pos) if not np.isnan(current_pos) else 0

            if not in_trade:
                # Look for entry
                if current_sign != 0:
                    in_trade = True
                    entry_bar = i
                    entry_price = prices[i]
                    trade_direction = int(current_sign)
                    cum_return = 0.0
            else:
                # Accumulate return for this bar
                if i - 1 < len(returns):
                    cum_return += returns[i - 1] if i > 0 else 0.0

                # Check for exit: position goes to 0 or flips sign
                if current_sign != trade_direction:
                    exit_price = prices[i]
                    trades.append({
                        "entry_bar": entry_bar,
                        "exit_bar": i,
                        "duration_bars": i - entry_bar,
                        "direction": trade_direction,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": cum_return,
                    })

                    # If flipped to opposite side, start new trade immediately
                    if current_sign != 0:
                        entry_bar = i
                        entry_price = prices[i]
                        trade_direction = int(current_sign)
                        cum_return = 0.0
                    else:
                        in_trade = False

        # Close any open trade at end
        if in_trade:
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": n - 1,
                "duration_bars": n - 1 - entry_bar,
                "direction": trade_direction,
                "entry_price": entry_price,
                "exit_price": prices[-1],
                "return_pct": cum_return,
            })

        self._log.debug(
            f"Extracted {len(trades)} trades",
            tags={"backtest"},
        )

        return trades

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(data: pl.Series | np.ndarray) -> np.ndarray:
        if isinstance(data, pl.Series):
            return data.to_numpy().astype(np.float64)
        return np.asarray(data, dtype=np.float64)
