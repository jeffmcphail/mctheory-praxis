"""
Paper Trading Mode (Phase 4.9).

Simulated execution against live or replayed price feeds. Orders are
"filled" at market prices with configurable slippage and latency.
Positions, P&L, and fills are tracked in the LiveStore.

This bridges backtesting and live trading: same signal logic,
same risk checks, but no real money at risk.

Usage:
    engine = PaperTradingEngine(store=LiveStore.duckdb())
    engine.submit_order(order)
    engine.on_price_update({"AAPL": 185.50, "GOOG": 142.30})
    print(engine.positions, engine.nav)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from praxis.live import (
    LiveStore,
    DuckDBLiveStore,
    LiveStoreBackend,
    LiveOrder,
    LiveFill,
    LivePosition,
    OrderSide,
    OrderStatus,
    OrderType,
)
from praxis.risk import (
    RiskManager,
    RiskConfig,
    PortfolioState,
    ProposedOrder,
    RiskAction,
)
from praxis.logger.core import PraxisLogger


@dataclass
class PaperConfig:
    """Paper trading configuration."""
    initial_cash: float = 100_000.0
    slippage_bps: float = 5.0            # 5 bps slippage
    commission_per_share: float = 0.005  # $0.005/share
    latency_ms: int = 0                  # Simulated latency
    risk_config: RiskConfig | None = None
    model_id: str = "paper_default"


@dataclass
class PaperState:
    """Current state of the paper trading engine."""
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)       # asset → quantity
    avg_prices: dict[str, float] = field(default_factory=dict)      # asset → avg entry price
    last_prices: dict[str, float] = field(default_factory=dict)     # asset → last known price
    realized_pnl: float = 0.0
    total_orders: int = 0
    total_fills: int = 0
    bar_count: int = 0

    @property
    def nav(self) -> float:
        """Net asset value = cash + market value of positions."""
        mv = sum(
            qty * self.last_prices.get(asset, self.avg_prices.get(asset, 0))
            for asset, qty in self.positions.items()
        )
        return self.cash + mv

    @property
    def gross_exposure(self) -> float:
        return sum(
            abs(qty) * self.last_prices.get(asset, self.avg_prices.get(asset, 0))
            for asset, qty in self.positions.items()
        )

    @property
    def position_count(self) -> int:
        return sum(1 for q in self.positions.values() if abs(q) > 1e-10)


class PaperTradingEngine:
    """
    Paper trading execution engine.

    Accepts orders, simulates fills with slippage/commission,
    tracks positions and P&L in a LiveStore.
    """

    def __init__(
        self,
        config: PaperConfig | None = None,
        store: LiveStoreBackend | None = None,
    ):
        self._config = config or PaperConfig()
        self._store = store or LiveStore.duckdb()
        self._state = PaperState(cash=self._config.initial_cash)
        self._risk = RiskManager(self._config.risk_config) if self._config.risk_config else None
        self._log = PraxisLogger.instance()
        self._pending_orders: list[LiveOrder] = []
        self._peak_nav = self._config.initial_cash

    @property
    def config(self) -> PaperConfig:
        return self._config

    @property
    def state(self) -> PaperState:
        return self._state

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._state.positions)

    @property
    def cash(self) -> float:
        return self._state.cash

    @property
    def nav(self) -> float:
        return self._state.nav

    @property
    def peak_nav(self) -> float:
        return self._peak_nav

    def submit_order(self, order: LiveOrder) -> str:
        """
        Submit an order for execution.

        Market orders fill immediately on next price update.
        Limit/stop orders queue until conditions met.
        """
        if not order.model_id:
            order.model_id = self._config.model_id

        # Risk check
        if self._risk is not None:
            portfolio_state = self._build_portfolio_state()
            proposed = ProposedOrder(
                asset=order.asset,
                side=1 if order.side == OrderSide.BUY else -1,
                size_dollars=order.quantity * self._state.last_prices.get(order.asset, 100),
            )
            check = self._risk.check_order(proposed, portfolio_state)
            if check.action == RiskAction.REJECT:
                order.status = OrderStatus.REJECTED
                self._store.insert_order(order)
                self._log.warning(
                    f"Paper: order rejected — {'; '.join(check.reasons)}",
                    tags={"paper"},
                )
                return order.order_id

        order.status = OrderStatus.SUBMITTED
        self._store.insert_order(order)
        self._pending_orders.append(order)
        self._state.total_orders += 1

        self._log.info(
            f"Paper: submitted {order.side.value} {order.quantity} {order.asset}",
            tags={"paper"},
        )
        return order.order_id

    def on_price_update(self, prices: dict[str, float]) -> list[LiveFill]:
        """
        Process a price update tick.

        Updates last prices, attempts to fill pending orders,
        updates positions and P&L.

        Args:
            prices: Dict of asset → current price.

        Returns:
            List of fills generated this tick.
        """
        self._state.last_prices.update(prices)
        self._state.bar_count += 1
        fills = []

        # Try to fill pending orders
        remaining = []
        for order in self._pending_orders:
            price = prices.get(order.asset)
            if price is None:
                remaining.append(order)
                continue

            should_fill = False

            if order.order_type == OrderType.MARKET:
                should_fill = True
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= (order.limit_price or float("inf")):
                    should_fill = True
                elif order.side == OrderSide.SELL and price >= (order.limit_price or 0):
                    should_fill = True
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= (order.stop_price or float("inf")):
                    should_fill = True
                elif order.side == OrderSide.SELL and price <= (order.stop_price or 0):
                    should_fill = True

            if should_fill:
                fill = self._execute_fill(order, price)
                fills.append(fill)
            else:
                remaining.append(order)

        self._pending_orders = remaining

        # Update peak NAV
        current_nav = self._state.nav
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav

        return fills

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for i, order in enumerate(self._pending_orders):
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self._store.update_order_status(order_id, OrderStatus.CANCELLED)
                self._pending_orders.pop(i)
                return True
        return False

    def close_position(self, asset: str) -> LiveFill | None:
        """Close an entire position at market."""
        qty = self._state.positions.get(asset, 0)
        if abs(qty) < 1e-10:
            return None

        side = OrderSide.SELL if qty > 0 else OrderSide.BUY
        order = LiveOrder(
            model_id=self._config.model_id,
            asset=asset,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(qty),
        )
        self.submit_order(order)

        # Immediately fill if we have a price
        price = self._state.last_prices.get(asset)
        if price is not None:
            return self.on_price_update({asset: price})[0] if self._pending_orders else None
        return None

    def get_fills(self, model_id: str | None = None) -> list[LiveFill]:
        """Get fills from store."""
        return self._store.get_fills(model_id=model_id or self._config.model_id)

    def get_orders(self, status: OrderStatus | None = None) -> list[LiveOrder]:
        """Get orders from store."""
        return self._store.get_orders(model_id=self._config.model_id, status=status)

    def summary(self) -> dict[str, Any]:
        """Get a summary of current paper trading state."""
        return {
            "nav": self.nav,
            "cash": self.cash,
            "peak_nav": self.peak_nav,
            "drawdown": 1 - self.nav / self.peak_nav if self.peak_nav > 0 else 0,
            "positions": dict(self._state.positions),
            "position_count": self._state.position_count,
            "realized_pnl": self._state.realized_pnl,
            "total_orders": self._state.total_orders,
            "total_fills": self._state.total_fills,
            "bar_count": self._state.bar_count,
        }

    # ── Internal ──────────────────────────────────────────────

    def _execute_fill(self, order: LiveOrder, market_price: float) -> LiveFill:
        """Execute a fill with slippage and commission."""
        # Apply slippage
        slippage_mult = self._config.slippage_bps / 10_000
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1 + slippage_mult)
        else:
            fill_price = market_price * (1 - slippage_mult)

        commission = self._config.commission_per_share * order.quantity

        fill = LiveFill(
            order_id=order.order_id,
            model_id=order.model_id,
            asset=order.asset,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            venue="paper",
        )

        # Update order status
        order.status = OrderStatus.FILLED
        self._store.update_order_status(order.order_id, OrderStatus.FILLED)

        # Record fill
        self._store.insert_fill(fill)
        self._state.total_fills += 1

        # Update position
        self._update_position(fill)

        self._log.info(
            f"Paper: filled {fill.side.value} {fill.quantity} {fill.asset} "
            f"@ {fill.price:.2f} (commission: {fill.commission:.2f})",
            tags={"paper"},
        )

        return fill

    def _update_position(self, fill: LiveFill) -> None:
        """Update cash and positions after a fill."""
        asset = fill.asset
        current_qty = self._state.positions.get(asset, 0)
        current_avg = self._state.avg_prices.get(asset, 0)

        cost = fill.price * fill.quantity + fill.commission

        if fill.side == OrderSide.BUY:
            # Buying: increase position, decrease cash
            new_qty = current_qty + fill.quantity
            if new_qty != 0:
                self._state.avg_prices[asset] = (
                    (current_avg * current_qty + fill.price * fill.quantity) / new_qty
                )
            self._state.cash -= cost
            self._state.positions[asset] = new_qty

        else:
            # Selling: decrease position, increase cash
            # Realize P&L on the sold quantity
            if current_qty > 0:
                realized = (fill.price - current_avg) * min(fill.quantity, current_qty)
                self._state.realized_pnl += realized

            new_qty = current_qty - fill.quantity
            self._state.cash += fill.price * fill.quantity - fill.commission
            self._state.positions[asset] = new_qty

            # Remove flat positions
            if abs(new_qty) < 1e-10:
                self._state.positions.pop(asset, None)
                self._state.avg_prices.pop(asset, None)

        # Sync to store
        pos = LivePosition(
            model_id=fill.model_id,
            asset=asset,
            quantity=self._state.positions.get(asset, 0),
            avg_entry_price=self._state.avg_prices.get(asset, 0),
            current_price=self._state.last_prices.get(asset, fill.price),
            realized_pnl=self._state.realized_pnl,
        )
        self._store.upsert_position(pos)

    def _build_portfolio_state(self) -> PortfolioState:
        """Build PortfolioState for risk checks."""
        position_dollars = {}
        for asset, qty in self._state.positions.items():
            price = self._state.last_prices.get(asset, self._state.avg_prices.get(asset, 100))
            position_dollars[asset] = qty * price

        return PortfolioState(
            nav=self.nav,
            peak_nav=self.peak_nav,
            positions=position_dollars,
            current_bar=self._state.bar_count,
        )
