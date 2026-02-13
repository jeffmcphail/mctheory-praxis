"""
Event-Driven Backtest Engine (Phase 3.9, §9.2).

Bar-by-bar processing with explicit fill simulation.
Processes each bar sequentially, maintaining state between bars.

Key differences from VectorizedEngine:
- Explicit order → fill → position update cycle
- Fill price modeling (open of next bar, or configurable)
- Cash tracking and partial fills possible
- Latency: signal at bar T, fill at bar T+1

Usage:
    engine = EventDrivenEngine()
    output = engine.run(signal_fn, prices, params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import polars as pl

from praxis.backtest import BacktestMetrics, BacktestOutput
from praxis.logger.core import PraxisLogger


@dataclass
class Order:
    """A pending order."""
    bar_index: int
    target_position: float
    order_type: str = "market"  # "market", "limit"


@dataclass
class Fill:
    """An executed fill."""
    bar_index: int
    fill_price: float
    quantity: float  # Signed: positive=buy, negative=sell
    commission: float = 0.0


@dataclass
class EngineState:
    """Mutable state maintained across bars."""
    position: float = 0.0
    cash: float = 1.0
    equity: float = 1.0
    pending_order: Order | None = None
    fills: list[Fill] = field(default_factory=list)
    positions_history: list[float] = field(default_factory=list)
    equity_history: list[float] = field(default_factory=list)
    cash_history: list[float] = field(default_factory=list)


class EventDrivenEngine:
    """
    §9.2: Event-driven backtest engine.

    Bar-by-bar processing:
    1. On each bar: receive signal → create order
    2. On next bar open: fill order at fill price
    3. Mark-to-market position at close
    4. Record equity

    Fill model: fill at next bar's open price (default) or close.
    """

    def __init__(
        self,
        annualization_factor: int = 252,
        fill_on: str = "next_open",  # "next_open" or "close"
        commission_per_trade: float = 0.0,
    ):
        self._ann_factor = annualization_factor
        self._fill_on = fill_on
        self._commission = commission_per_trade
        self._log = PraxisLogger.instance()

    def run(
        self,
        positions: pl.Series | np.ndarray,
        prices: pl.Series | np.ndarray,
        initial_capital: float = 1.0,
        open_prices: pl.Series | np.ndarray | None = None,
    ) -> BacktestOutput:
        """
        Run event-driven backtest.

        Args:
            positions: Target position per bar (+1, -1, 0, fractional).
            prices: Close prices.
            initial_capital: Starting equity.
            open_prices: Open prices for fill simulation (defaults to close).

        Returns:
            BacktestOutput matching VectorizedEngine interface.
        """
        import time
        start_time = time.monotonic()

        pos_arr = self._to_numpy(positions)
        close = self._to_numpy(prices)
        n = len(close)

        if open_prices is not None:
            opens = self._to_numpy(open_prices)
        else:
            opens = close.copy()

        state = EngineState(cash=initial_capital, equity=initial_capital)

        self._log.debug(
            f"EventDrivenEngine: {n} bars, fill_on={self._fill_on}",
            tags={"backtest"},
        )

        for i in range(n):
            # ── Step 1: Fill pending order ────────────────────
            if state.pending_order is not None:
                fill_price = opens[i] if self._fill_on == "next_open" else close[i]
                self._fill_order(state, fill_price, i)

            # ── Step 2: Mark-to-market ────────────────────────
            state.equity = state.cash + state.position * close[i]

            # ── Step 3: Generate order from signal ────────────
            target = float(pos_arr[i])
            if target != state.position:
                state.pending_order = Order(bar_index=i, target_position=target)

            # ── Record state ──────────────────────────────────
            state.positions_history.append(state.position)
            state.equity_history.append(state.equity)
            state.cash_history.append(state.cash)

        # ── Compute metrics ───────────────────────────────────
        equity = np.array(state.equity_history)
        daily_returns = np.zeros(n)
        if n > 1:
            daily_returns[1:] = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1.0)

        metrics = self._compute_metrics(
            equity, daily_returns, state.fills, state.positions_history
        )

        duration = time.monotonic() - start_time

        return BacktestOutput(
            metrics=metrics,
            equity_curve=equity,
            daily_returns=daily_returns,
            positions=np.array(state.positions_history),
            trades=self._fills_to_trades(state.fills),
            bar_count=n,
            duration_seconds=duration,
        )

    def _fill_order(self, state: EngineState, fill_price: float, bar_idx: int) -> None:
        """Execute pending order at fill price."""
        order = state.pending_order
        if order is None:
            return

        quantity = order.target_position - state.position
        cost = quantity * fill_price
        commission = abs(quantity) * self._commission

        state.cash -= cost + commission
        state.position = order.target_position
        state.pending_order = None

        state.fills.append(Fill(
            bar_index=bar_idx,
            fill_price=fill_price,
            quantity=quantity,
            commission=commission,
        ))

    def _compute_metrics(
        self,
        equity: np.ndarray,
        daily_returns: np.ndarray,
        fills: list[Fill],
        positions: list[float],
    ) -> BacktestMetrics:
        """Compute metrics matching VectorizedEngine output."""
        metrics = BacktestMetrics()

        if len(equity) < 2:
            return metrics

        # Total return
        metrics.total_return = float((equity[-1] / equity[0]) - 1)

        # Annualized return
        n_bars = len(equity)
        years = n_bars / self._ann_factor
        if years > 0 and equity[0] > 0:
            metrics.annualized_return = float(
                (equity[-1] / equity[0]) ** (1 / years) - 1
            )

        # Volatility
        valid_returns = daily_returns[1:]  # Skip first zero
        if len(valid_returns) > 1:
            metrics.volatility = float(np.std(valid_returns, ddof=1) * np.sqrt(self._ann_factor))

            # Sharpe
            mean_ret = float(np.mean(valid_returns))
            std_ret = float(np.std(valid_returns, ddof=1))
            if std_ret > 0:
                metrics.sharpe_ratio = float(
                    mean_ret / std_ret * np.sqrt(self._ann_factor)
                )

            # Sortino
            downside = valid_returns[valid_returns < 0]
            if len(downside) > 0:
                down_std = float(np.std(downside, ddof=1))
                if down_std > 0:
                    metrics.sortino_ratio = float(
                        mean_ret / down_std * np.sqrt(self._ann_factor)
                    )

        # Drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        metrics.max_drawdown = float(np.min(dd))

        # Calmar
        if metrics.max_drawdown < 0:
            metrics.calmar_ratio = float(
                metrics.annualized_return / abs(metrics.max_drawdown)
            )

        # Trades
        metrics.total_trades = len(fills)

        # Win rate from fills
        if len(fills) >= 2:
            trade_returns = []
            for j in range(1, len(fills)):
                if fills[j - 1].quantity != 0:
                    ret = (fills[j].fill_price - fills[j - 1].fill_price) / fills[j - 1].fill_price
                    if fills[j - 1].quantity < 0:
                        ret = -ret
                    trade_returns.append(ret)
            if trade_returns:
                wins = sum(1 for r in trade_returns if r > 0)
                metrics.win_rate = wins / len(trade_returns)
                gross_profit = sum(r for r in trade_returns if r > 0)
                gross_loss = abs(sum(r for r in trade_returns if r < 0))
                if gross_loss > 0:
                    metrics.profit_factor = gross_profit / gross_loss
                if trade_returns:
                    metrics.avg_trade_return = float(np.mean(trade_returns))

        return metrics

    def _fills_to_trades(self, fills: list[Fill]) -> list[dict[str, Any]]:
        """Convert fills to trade dicts for BacktestOutput."""
        return [
            {
                "bar": f.bar_index,
                "price": f.fill_price,
                "quantity": f.quantity,
                "commission": f.commission,
            }
            for f in fills
        ]

    @staticmethod
    def _to_numpy(data: pl.Series | np.ndarray) -> np.ndarray:
        if isinstance(data, pl.Series):
            return data.to_numpy().astype(np.float64)
        return np.asarray(data, dtype=np.float64)
