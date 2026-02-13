"""
Risk Management Framework (Phase 4.7).

Pre-trade and post-trade risk checks:
- Position limits (per-asset, per-sector, total)
- Drawdown monitoring and circuit breakers
- Exposure limits (gross, net, sector)
- Stop-loss enforcement
- Sizing constraint validation

The risk manager is a pipeline gate: signals pass through risk checks
before becoming orders. Any violation blocks or reduces the signal.

Usage:
    rm = RiskManager(config)
    checked = rm.check_order(order, portfolio_state)
    if checked.approved:
        execute(checked.order)
    else:
        log(checked.rejection_reason)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════

class RiskAction(str, Enum):
    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"
    CLOSE = "close"          # Force close position


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Position limits
    max_position_pct: float = 0.10       # Max single position as % of NAV
    max_sector_pct: float = 0.30         # Max sector exposure as % of NAV
    max_correlated_pct: float = 0.25     # Max correlated group exposure

    # Exposure limits
    max_gross_exposure: float = 2.0      # Max gross exposure as multiple of NAV
    max_net_exposure: float = 1.0        # Max net exposure as multiple of NAV

    # Drawdown
    max_drawdown_pct: float = 0.20       # Circuit breaker drawdown threshold
    drawdown_reduce_pct: float = 0.10    # Reduce sizing at this drawdown

    # Stop-loss
    stop_loss_pct: float = 0.05          # Per-position stop loss
    stop_loss_atr_mult: float | None = None  # ATR-based stop

    # Trading limits
    max_orders_per_bar: int = 10
    max_turnover_daily: float = 0.50     # Max daily turnover as % of NAV
    cooldown_bars: int = 0               # Min bars between trades per asset


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk checks."""
    nav: float = 100_000.0
    peak_nav: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)    # asset → size in $
    sectors: dict[str, str] = field(default_factory=dict)         # asset → sector
    daily_turnover: float = 0.0
    orders_this_bar: int = 0
    last_trade_bar: dict[str, int] = field(default_factory=dict)  # asset → last bar#
    current_bar: int = 0

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_nav <= 0:
            return 0.0
        return 1.0 - self.nav / self.peak_nav

    @property
    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(v for v in self.positions.values())

    @property
    def gross_leverage(self) -> float:
        return self.gross_exposure / self.nav if self.nav > 0 else 0.0

    @property
    def net_leverage(self) -> float:
        return self.net_exposure / self.nav if self.nav > 0 else 0.0

    def sector_exposure(self, sector: str) -> float:
        """Total absolute exposure in a sector."""
        return sum(
            abs(v) for asset, v in self.positions.items()
            if self.sectors.get(asset) == sector
        )


@dataclass
class ProposedOrder:
    """An order to be risk-checked."""
    asset: str
    side: int          # +1 buy, -1 sell
    size_dollars: float
    order_type: str = "market"
    sector: str = ""


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    action: RiskAction = RiskAction.APPROVE
    original_size: float = 0.0
    approved_size: float = 0.0
    reasons: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.action in (RiskAction.APPROVE, RiskAction.REDUCE)

    @property
    def rejected(self) -> bool:
        return self.action == RiskAction.REJECT


@dataclass
class DrawdownState:
    """Drawdown monitoring state."""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    peak_nav: float = 0.0
    trough_nav: float = 0.0
    is_circuit_broken: bool = False
    is_reduced: bool = False


# ═══════════════════════════════════════════════════════════════════
#  Risk Manager
# ═══════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Pre-trade and post-trade risk management.

    Validates proposed orders against position limits, exposure limits,
    drawdown thresholds, and trading constraints.
    """

    def __init__(self, config: RiskConfig | None = None):
        self._config = config or RiskConfig()

    @property
    def config(self) -> RiskConfig:
        return self._config

    def check_order(
        self,
        order: ProposedOrder,
        state: PortfolioState,
    ) -> RiskCheckResult:
        """
        Run all pre-trade risk checks on a proposed order.

        Returns RiskCheckResult with action (approve/reduce/reject)
        and detailed reasons.
        """
        result = RiskCheckResult(
            original_size=order.size_dollars,
            approved_size=order.size_dollars,
        )

        checks = [
            self._check_drawdown_circuit_breaker,
            self._check_position_limit,
            self._check_sector_limit,
            self._check_gross_exposure,
            self._check_net_exposure,
            self._check_orders_per_bar,
            self._check_daily_turnover,
            self._check_cooldown,
        ]

        for check in checks:
            check(order, state, result)
            if result.action == RiskAction.REJECT:
                break

        return result

    def check_drawdown(self, state: PortfolioState) -> DrawdownState:
        """Monitor drawdown state for circuit breaker."""
        dd = DrawdownState(
            current_drawdown=state.drawdown,
            peak_nav=state.peak_nav,
            trough_nav=state.nav,
        )

        if state.drawdown > dd.max_drawdown:
            dd.max_drawdown = state.drawdown

        if state.drawdown >= self._config.max_drawdown_pct:
            dd.is_circuit_broken = True
        elif state.drawdown >= self._config.drawdown_reduce_pct:
            dd.is_reduced = True

        return dd

    def check_stop_loss(
        self,
        asset: str,
        entry_price: float,
        current_price: float,
        side: int = 1,
        atr: float | None = None,
    ) -> bool:
        """
        Check if a position has hit its stop-loss.

        Returns True if stop-loss is triggered.
        """
        if entry_price <= 0:
            return False

        pnl_pct = side * (current_price - entry_price) / entry_price

        # Percentage-based stop
        if pnl_pct <= -self._config.stop_loss_pct:
            return True

        # ATR-based stop
        if self._config.stop_loss_atr_mult is not None and atr is not None:
            atr_stop = self._config.stop_loss_atr_mult * atr / entry_price
            if pnl_pct <= -atr_stop:
                return True

        return False

    def compute_sizing_adjustment(self, state: PortfolioState) -> float:
        """
        Compute a sizing multiplier based on current risk state.

        Returns 1.0 (full size), 0.5 (reduced), or 0.0 (halted).
        """
        dd = state.drawdown

        if dd >= self._config.max_drawdown_pct:
            return 0.0
        elif dd >= self._config.drawdown_reduce_pct:
            # Linear reduction between reduce and max
            range_pct = self._config.max_drawdown_pct - self._config.drawdown_reduce_pct
            if range_pct > 0:
                reduction = (dd - self._config.drawdown_reduce_pct) / range_pct
                return max(0.0, 1.0 - reduction)
            return 0.5

        return 1.0

    # ── Individual Checks ─────────────────────────────────────

    def _check_drawdown_circuit_breaker(
        self, order: ProposedOrder, state: PortfolioState, result: RiskCheckResult,
    ) -> None:
        dd = state.drawdown
        if dd >= self._config.max_drawdown_pct:
            result.action = RiskAction.REJECT
            result.approved_size = 0.0
            result.reasons.append(
                f"Circuit breaker: drawdown {dd:.1%} >= {self._config.max_drawdown_pct:.1%}"
            )
            result.checks_failed.append("drawdown_circuit_breaker")
        else:
            result.checks_passed.append("drawdown_circuit_breaker")

    def _check_position_limit(
        self, order: ProposedOrder, state: PortfolioState, result: RiskCheckResult,
    ) -> None:
        current = abs(state.positions.get(order.asset, 0))
        proposed = current + order.size_dollars
        limit = self._config.max_position_pct * state.nav

        if proposed > limit:
            allowed = max(0, limit - current)
            if allowed <= 0:
                result.action = RiskAction.REJECT
                result.approved_size = 0.0
                result.reasons.append(
                    f"Position limit: {order.asset} would be ${proposed:,.0f} "
                    f"(limit ${limit:,.0f})"
                )
                result.checks_failed.append("position_limit")
            else:
                result.action = RiskAction.REDUCE
                result.approved_size = min(result.approved_size, allowed)
                result.reasons.append(
                    f"Position reduced: {order.asset} capped at ${allowed:,.0f}"
                )
                result.checks_failed.append("position_limit_reduced")
        else:
            result.checks_passed.append("position_limit")

    def _check_sector_limit(
        self, order: ProposedOrder, state: PortfolioState, result: RiskCheckResult,
    ) -> None:
        sector = order.sector or state.sectors.get(order.asset, "")
        if not sector:
            result.checks_passed.append("sector_limit")
            return

        current_exposure = state.sector_exposure(sector)
        proposed = current_exposure + order.size_dollars
        limit = self._config.max_sector_pct * state.nav

        if proposed > limit:
            result.action = RiskAction.REDUCE
            allowed = max(0, limit - current_exposure)
            result.approved_size = min(result.approved_size, allowed)
            result.reasons.append(
                f"Sector limit: {sector} would be ${proposed:,.0f} (limit ${limit:,.0f})"
            )
            result.checks_failed.append("sector_limit")
        else:
            result.checks_passed.append("sector_limit")

    def _check_gross_exposure(
        self, order: ProposedOrder, state: PortfolioState, result: RiskCheckResult,
    ) -> None:
        proposed = state.gross_exposure + order.size_dollars
        limit = self._config.max_gross_exposure * state.nav

        if proposed > limit:
            result.action = RiskAction.REJECT
            result.approved_size = 0.0
            result.reasons.append(
                f"Gross exposure: ${proposed:,.0f} > {self._config.max_gross_exposure:.1f}x NAV"
            )
            result.checks_failed.append("gross_exposure")
        else:
            result.checks_passed.append("gross_exposure")

    def _check_net_exposure(
        self, order: ProposedOrder, state: PortfolioState, result: RiskCheckResult,
    ) -> None:
        direction = order.side * order.size_dollars
        proposed_net = abs(state.net_exposure + direction)
        limit = self._config.max_net_exposure * state.nav

        if proposed_net > limit:
            result.action = RiskAction.REDUCE
            result.reasons.append(
                f"Net exposure: would be ${proposed_net:,.0f} > {self._config.max_net_exposure:.1f}x NAV"
            )
            result.checks_failed.append("net_exposure")
        else:
            result.checks_passed.append("net_exposure")

    def _check_orders_per_bar(
        self, order: ProposedOrder, state: PortfolioState, result: RiskCheckResult,
    ) -> None:
        if state.orders_this_bar >= self._config.max_orders_per_bar:
            result.action = RiskAction.REJECT
            result.approved_size = 0.0
            result.reasons.append(
                f"Order limit: {state.orders_this_bar} >= {self._config.max_orders_per_bar}"
            )
            result.checks_failed.append("orders_per_bar")
        else:
            result.checks_passed.append("orders_per_bar")

    def _check_daily_turnover(
        self, order: ProposedOrder, state: PortfolioState, result: RiskCheckResult,
    ) -> None:
        proposed = state.daily_turnover + order.size_dollars
        limit = self._config.max_turnover_daily * state.nav

        if proposed > limit:
            result.action = RiskAction.REJECT
            result.approved_size = 0.0
            result.reasons.append(
                f"Turnover limit: ${proposed:,.0f} > ${limit:,.0f}"
            )
            result.checks_failed.append("daily_turnover")
        else:
            result.checks_passed.append("daily_turnover")

    def _check_cooldown(
        self, order: ProposedOrder, state: PortfolioState, result: RiskCheckResult,
    ) -> None:
        if self._config.cooldown_bars <= 0:
            result.checks_passed.append("cooldown")
            return

        last = state.last_trade_bar.get(order.asset, -999)
        bars_since = state.current_bar - last

        if bars_since < self._config.cooldown_bars:
            result.action = RiskAction.REJECT
            result.approved_size = 0.0
            result.reasons.append(
                f"Cooldown: {order.asset} traded {bars_since} bars ago "
                f"(min {self._config.cooldown_bars})"
            )
            result.checks_failed.append("cooldown")
        else:
            result.checks_passed.append("cooldown")
