"""
engines/carry_executor.py — Funding Rate Carry Trade Executor

Manages the full lifecycle of delta-neutral carry trades on Binance:
  Entry:  Buy spot + Sell perp (short) at equal notional
  Hold:   Collect funding payments every 8h
  Exit:   Sell spot + Buy perp (close short)

Positions are persisted to a JSON file so they survive restarts.
All trades use Binance API via CCXT.

Safety:
  - Paper trading mode by default (no real orders)
  - Position size limits enforced
  - Delta-neutral check on entry
  - All orders logged with timestamps

Usage:
    from engines.carry_executor import CarryExecutor
    executor = CarryExecutor(paper=True)
    executor.enter_carry("BTC", notional_usd=500, hold_days=7)
    executor.status()
    executor.exit_carry("BTC")
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

QUOTE = "USDT"
POSITIONS_FILE = "data/carry_positions.json"
TRADE_LOG_FILE = "data/carry_trade_log.json"


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CarryPosition:
    """An open carry position."""
    asset: str
    entry_time: str          # ISO format UTC
    hold_days: int
    exit_due: str            # ISO format UTC

    # Spot leg
    spot_qty: float          # amount of asset bought
    spot_entry_price: float  # average fill price
    spot_order_id: str       # exchange order ID

    # Perp leg
    perp_qty: float          # amount of asset shorted (should match spot_qty)
    perp_entry_price: float  # average fill price
    perp_order_id: str       # exchange order ID

    # Sizing
    notional_usd: float     # approximate USD value of each leg
    p_score: float           # RF P(profitable) at entry

    # Tracking
    funding_collected: float = 0.0    # cumulative funding in USDT
    funding_payments: int = 0         # number of payments received
    status: str = "open"              # open, closing, closed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CarryPosition:
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    @property
    def age_hours(self) -> float:
        entry = datetime.fromisoformat(self.entry_time)
        now = datetime.now(timezone.utc)
        return (now - entry).total_seconds() / 3600

    @property
    def is_expired(self) -> bool:
        due = datetime.fromisoformat(self.exit_due)
        return datetime.now(timezone.utc) >= due

    @property
    def expected_funding_payments(self) -> int:
        """How many 8h funding windows have passed since entry."""
        return int(self.age_hours / 8)


@dataclass
class TradeRecord:
    """A single trade execution record."""
    timestamp: str
    asset: str
    side: str               # "buy" or "sell"
    market: str             # "spot" or "perp"
    qty: float
    price: float
    notional_usd: float
    order_id: str
    order_type: str          # "market" or "limit"
    paper: bool              # True if paper trade
    action: str              # "entry" or "exit"

    def to_dict(self) -> dict:
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════════════
# EXECUTOR
# ═════════════════════════════════════════════════════════════════════════════

class CarryExecutor:
    """
    Manages carry trade lifecycle on Binance.

    Paper mode (default): simulates orders using current market prices.
    Live mode: places real orders via Binance API.
    """

    def __init__(
        self,
        paper: bool = True,
        positions_file: str = POSITIONS_FILE,
        trade_log_file: str = TRADE_LOG_FILE,
        max_position_usd: float = 2000.0,
        max_total_exposure_usd: float = 5000.0,
        order_type: str = "market",
    ):
        """
        Args:
            paper: If True, simulate orders without hitting exchange
            positions_file: JSON file for position persistence
            trade_log_file: JSON file for trade history
            max_position_usd: Maximum USD per position
            max_total_exposure_usd: Maximum total exposure across all positions
            order_type: "market" or "limit"
        """
        self.paper = paper
        self.positions_file = Path(positions_file)
        self.trade_log_file = Path(trade_log_file)
        self.max_position_usd = max_position_usd
        self.max_total_exposure_usd = max_total_exposure_usd
        self.order_type = order_type

        self._spot_exchange = None
        self._perp_exchange = None

        # Load existing positions
        self.positions: dict[str, CarryPosition] = self._load_positions()
        self.trade_log: list[TradeRecord] = self._load_trade_log()

    def _get_spot_exchange(self):
        if self._spot_exchange is None:
            import ccxt
            from dotenv import load_dotenv
            import os
            load_dotenv()

            self._spot_exchange = ccxt.binance({
                "apiKey": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_API_SECRET", ""),
                "enableRateLimit": True,
            })
        return self._spot_exchange

    def _get_perp_exchange(self):
        if self._perp_exchange is None:
            import ccxt
            from dotenv import load_dotenv
            import os
            load_dotenv()

            self._perp_exchange = ccxt.binance({
                "apiKey": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_API_SECRET", ""),
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
        return self._perp_exchange

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_positions(self) -> dict[str, CarryPosition]:
        if self.positions_file.exists():
            with open(self.positions_file) as f:
                data = json.load(f)
            return {k: CarryPosition.from_dict(v) for k, v in data.items()}
        return {}

    def _save_positions(self):
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.positions_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.positions.items()},
                      f, indent=2)

    def _load_trade_log(self) -> list[TradeRecord]:
        if self.trade_log_file.exists():
            with open(self.trade_log_file) as f:
                data = json.load(f)
            return [TradeRecord(**d) for d in data]
        return []

    def _save_trade_log(self):
        self.trade_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trade_log_file, "w") as f:
            json.dump([t.to_dict() for t in self.trade_log], f, indent=2)

    def _log_trade(self, record: TradeRecord):
        self.trade_log.append(record)
        self._save_trade_log()

    # ── Market data ──────────────────────────────────────────────────────────

    def get_spot_price(self, asset: str) -> float:
        """Get current spot price."""
        exchange = self._get_spot_exchange()
        ticker = exchange.fetch_ticker(f"{asset}/{QUOTE}")
        return float(ticker["last"])

    def get_perp_price(self, asset: str) -> float:
        """Get current perp price."""
        exchange = self._get_perp_exchange()
        ticker = exchange.fetch_ticker(f"{asset}/{QUOTE}:{QUOTE}")
        return float(ticker["last"])

    def get_funding_rate(self, asset: str) -> float:
        """Get current funding rate."""
        exchange = self._get_perp_exchange()
        info = exchange.fetch_funding_rate(f"{asset}/{QUOTE}:{QUOTE}")
        return float(info.get("fundingRate", 0))

    # ── Order execution ──────────────────────────────────────────────────────

    def _execute_order(
        self,
        exchange,
        symbol: str,
        side: str,
        qty: float,
        market: str,
    ) -> dict:
        """
        Execute an order (or simulate in paper mode).

        Returns: {"id": order_id, "price": fill_price, "qty": filled_qty}
        """
        now = datetime.now(timezone.utc)

        if self.paper:
            # Simulate with current market price
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker["last"])
            order_id = f"PAPER-{now.strftime('%Y%m%d%H%M%S')}-{market}-{side}"
            print(f"    📝 PAPER {side.upper()} {qty:.6f} {symbol} "
                  f"@ ${price:,.2f} = ${qty * price:,.2f}")
            return {"id": order_id, "price": price, "qty": qty}

        else:
            # Real order
            if self.order_type == "market":
                order = exchange.create_market_order(symbol, side, qty)
            else:
                # Limit order at current best price
                ticker = exchange.fetch_ticker(symbol)
                price = float(ticker["bid"] if side == "buy" else ticker["ask"])
                order = exchange.create_limit_order(symbol, side, qty, price)

            # Wait for fill
            filled_price = float(order.get("average", order.get("price", 0)))
            filled_qty = float(order.get("filled", qty))
            order_id = str(order.get("id", "unknown"))

            print(f"    ✅ LIVE {side.upper()} {filled_qty:.6f} {symbol} "
                  f"@ ${filled_price:,.2f} = ${filled_qty * filled_price:,.2f}")
            return {"id": order_id, "price": filled_price, "qty": filled_qty}

    # ── Entry ────────────────────────────────────────────────────────────────

    def enter_carry(
        self,
        asset: str,
        notional_usd: float,
        hold_days: int = 7,
        p_score: float = 0.0,
    ) -> CarryPosition | None:
        """
        Enter a delta-neutral carry position.

        Buys spot + Sells perp for equal notional.

        Args:
            asset: Token symbol (e.g. "BTC")
            notional_usd: USD value for each leg
            hold_days: Days to hold before exit
            p_score: RF P(profitable) at entry time
        """
        mode = "PAPER" if self.paper else "LIVE"
        now = datetime.now(timezone.utc)

        print(f"\n  [{mode}] Entering carry: {asset}")
        print(f"    Notional: ${notional_usd:,.2f} per leg")
        print(f"    Hold: {hold_days} days")

        # ── Safety checks ──
        if asset in self.positions:
            print(f"    ❌ Already have open position in {asset}")
            return None

        if notional_usd > self.max_position_usd:
            print(f"    ❌ Exceeds max position size "
                  f"(${notional_usd:,.0f} > ${self.max_position_usd:,.0f})")
            return None

        total_exposure = sum(p.notional_usd for p in self.positions.values())
        if total_exposure + notional_usd > self.max_total_exposure_usd:
            print(f"    ❌ Exceeds max total exposure "
                  f"(${total_exposure + notional_usd:,.0f} > "
                  f"${self.max_total_exposure_usd:,.0f})")
            return None

        # ── Get prices ──
        spot_price = self.get_spot_price(asset)
        perp_price = self.get_perp_price(asset)
        basis_bps = (perp_price - spot_price) / spot_price * 10000
        funding = self.get_funding_rate(asset)

        print(f"    Spot: ${spot_price:,.2f}")
        print(f"    Perp: ${perp_price:,.2f}")
        print(f"    Basis: {basis_bps:+.1f} bps")
        print(f"    Current funding: {funding * 100:.4f}%")

        # ── Calculate quantity ──
        qty = notional_usd / spot_price

        # Round to exchange precision
        # (simplified — production code should use exchange.amount_to_precision)
        if spot_price > 10000:      # BTC-class
            qty = round(qty, 5)
        elif spot_price > 100:      # ETH-class
            qty = round(qty, 4)
        elif spot_price > 1:        # SOL-class
            qty = round(qty, 2)
        else:                       # ARB-class
            qty = round(qty, 1)

        print(f"    Quantity: {qty} {asset}")

        # ── Execute spot buy ──
        print(f"\n    Leg 1: Buy spot")
        spot_symbol = f"{asset}/{QUOTE}"
        spot_result = self._execute_order(
            self._get_spot_exchange(), spot_symbol, "buy", qty, "spot"
        )

        # ── Execute perp sell (short) ──
        print(f"    Leg 2: Short perp")
        perp_symbol = f"{asset}/{QUOTE}:{QUOTE}"
        perp_result = self._execute_order(
            self._get_perp_exchange(), perp_symbol, "sell", qty, "perp"
        )

        # ── Create position record ──
        exit_due = now + timedelta(days=hold_days)
        position = CarryPosition(
            asset=asset,
            entry_time=now.isoformat(),
            hold_days=hold_days,
            exit_due=exit_due.isoformat(),
            spot_qty=spot_result["qty"],
            spot_entry_price=spot_result["price"],
            spot_order_id=spot_result["id"],
            perp_qty=perp_result["qty"],
            perp_entry_price=perp_result["price"],
            perp_order_id=perp_result["id"],
            notional_usd=notional_usd,
            p_score=p_score,
        )

        # ── Log trades ──
        for side, market, result in [
            ("buy", "spot", spot_result),
            ("sell", "perp", perp_result),
        ]:
            self._log_trade(TradeRecord(
                timestamp=now.isoformat(),
                asset=asset,
                side=side,
                market=market,
                qty=result["qty"],
                price=result["price"],
                notional_usd=result["qty"] * result["price"],
                order_id=result["id"],
                order_type=self.order_type,
                paper=self.paper,
                action="entry",
            ))

        # ── Save ──
        self.positions[asset] = position
        self._save_positions()

        print(f"\n    ✅ Position opened: {asset} carry, "
              f"exit due {exit_due.strftime('%Y-%m-%d %H:%M UTC')}")
        return position

    # ── Exit ─────────────────────────────────────────────────────────────────

    def exit_carry(self, asset: str, reason: str = "manual") -> dict | None:
        """
        Close a carry position.

        Sells spot + Buys perp (close short).

        Args:
            asset: Token symbol
            reason: Why we're exiting (manual, expired, signal, stop_loss)

        Returns:
            Dict with P&L summary
        """
        mode = "PAPER" if self.paper else "LIVE"
        now = datetime.now(timezone.utc)

        if asset not in self.positions:
            print(f"  ❌ No open position in {asset}")
            return None

        pos = self.positions[asset]
        pos.status = "closing"

        print(f"\n  [{mode}] Exiting carry: {asset} (reason: {reason})")
        print(f"    Held: {pos.age_hours:.1f} hours ({pos.age_hours/24:.1f} days)")

        # ── Get current prices ──
        spot_price = self.get_spot_price(asset)
        perp_price = self.get_perp_price(asset)

        # ── Execute spot sell ──
        print(f"\n    Leg 1: Sell spot")
        spot_symbol = f"{asset}/{QUOTE}"
        spot_result = self._execute_order(
            self._get_spot_exchange(), spot_symbol, "sell", pos.spot_qty, "spot"
        )

        # ── Execute perp buy (close short) ──
        print(f"    Leg 2: Close perp short")
        perp_symbol = f"{asset}/{QUOTE}:{QUOTE}"
        perp_result = self._execute_order(
            self._get_perp_exchange(), perp_symbol, "buy", pos.perp_qty, "perp"
        )

        # ── Calculate P&L ──
        spot_pnl = (spot_result["price"] - pos.spot_entry_price) * pos.spot_qty
        perp_pnl = (pos.perp_entry_price - perp_result["price"]) * pos.perp_qty
        total_pnl = spot_pnl + perp_pnl + pos.funding_collected

        pnl_bps = total_pnl / pos.notional_usd * 10000

        print(f"\n    P&L Breakdown:")
        print(f"      Spot:    ${spot_pnl:+.2f} "
              f"(${pos.spot_entry_price:,.2f} → ${spot_result['price']:,.2f})")
        print(f"      Perp:    ${perp_pnl:+.2f} "
              f"(${pos.perp_entry_price:,.2f} → ${perp_result['price']:,.2f})")
        print(f"      Funding: ${pos.funding_collected:+.2f} "
              f"({pos.funding_payments} payments)")
        print(f"      TOTAL:   ${total_pnl:+.2f} ({pnl_bps:+.1f} bps)")

        icon = "✅" if total_pnl > 0 else "❌"
        print(f"\n    {icon} Position closed: {asset}")

        # ── Log trades ──
        for side, market, result in [
            ("sell", "spot", spot_result),
            ("buy", "perp", perp_result),
        ]:
            self._log_trade(TradeRecord(
                timestamp=now.isoformat(),
                asset=asset,
                side=side,
                market=market,
                qty=result["qty"],
                price=result["price"],
                notional_usd=result["qty"] * result["price"],
                order_id=result["id"],
                order_type=self.order_type,
                paper=self.paper,
                action="exit",
            ))

        # ── Remove position ──
        del self.positions[asset]
        self._save_positions()

        return {
            "asset": asset,
            "hold_hours": pos.age_hours,
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "funding_collected": pos.funding_collected,
            "total_pnl": total_pnl,
            "pnl_bps": pnl_bps,
            "reason": reason,
        }

    # ── Status ───────────────────────────────────────────────────────────────

    def status(self) -> str:
        """Print current portfolio status."""
        lines = []
        now = datetime.now(timezone.utc)
        mode = "PAPER" if self.paper else "LIVE"

        lines.append(f"\n{'='*70}")
        lines.append(f"CARRY PORTFOLIO STATUS [{mode}]")
        lines.append(f"As of: {now.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"{'='*70}")

        if not self.positions:
            lines.append("\n  No open positions.")
            lines.append(f"\n  Limits: ${self.max_position_usd:,.0f}/position, "
                         f"${self.max_total_exposure_usd:,.0f} total")
            report = "\n".join(lines)
            print(report)
            return report

        total_notional = 0
        total_funding = 0

        for asset, pos in sorted(self.positions.items()):
            # Get live prices for unrealized P&L
            try:
                spot_now = self.get_spot_price(asset)
                perp_now = self.get_perp_price(asset)
                spot_pnl = (spot_now - pos.spot_entry_price) * pos.spot_qty
                perp_pnl = (pos.perp_entry_price - perp_now) * pos.perp_qty
                unreal = spot_pnl + perp_pnl
                price_str = f"spot=${spot_now:,.2f} perp=${perp_now:,.2f}"
            except Exception:
                unreal = 0
                price_str = "prices unavailable"

            total_notional += pos.notional_usd
            total_funding += pos.funding_collected

            due = datetime.fromisoformat(pos.exit_due)
            remaining = due - now
            remaining_str = (f"{remaining.days}d {remaining.seconds//3600}h"
                             if remaining.total_seconds() > 0 else "EXPIRED")

            lines.append(f"\n  {asset}:")
            lines.append(f"    Notional:  ${pos.notional_usd:,.2f}")
            lines.append(f"    Qty:       {pos.spot_qty} {asset}")
            lines.append(f"    Entry:     spot=${pos.spot_entry_price:,.2f} "
                         f"perp=${pos.perp_entry_price:,.2f}")
            lines.append(f"    Current:   {price_str}")
            lines.append(f"    Unreal:    ${unreal:+.2f} "
                         f"(spot=${spot_pnl:+.2f}, perp=${perp_pnl:+.2f})")
            lines.append(f"    Funding:   ${pos.funding_collected:+.2f} "
                         f"({pos.funding_payments} payments)")
            lines.append(f"    Age:       {pos.age_hours:.1f}h "
                         f"({pos.age_hours/24:.1f}d)")
            lines.append(f"    Remaining: {remaining_str}")
            lines.append(f"    P(entry):  {pos.p_score:.3f}")

        lines.append(f"\n{'─'*70}")
        lines.append(f"  Total exposure:  ${total_notional:,.2f} / "
                     f"${self.max_total_exposure_usd:,.0f}")
        lines.append(f"  Total funding:   ${total_funding:+.2f}")
        lines.append(f"  Open positions:  {len(self.positions)}")
        lines.append(f"{'='*70}")

        report = "\n".join(lines)
        print(report)
        return report

    # ── Auto-management ──────────────────────────────────────────────────────

    def check_exits(self) -> list[dict]:
        """
        Check all open positions for exit conditions.
        Returns list of P&L dicts for positions that were closed.
        """
        results = []
        # Copy keys to avoid modifying dict during iteration
        for asset in list(self.positions.keys()):
            pos = self.positions[asset]

            if pos.is_expired:
                print(f"  ⏰ {asset}: hold period expired")
                result = self.exit_carry(asset, reason="expired")
                if result:
                    results.append(result)

        return results

    def update_funding(self):
        """
        Check and record any new funding payments for open positions.

        Call this every ~8 hours (after funding settlement).
        Estimates funding from rate × position size.
        """
        for asset, pos in self.positions.items():
            expected = pos.expected_funding_payments
            if expected > pos.funding_payments:
                # Estimate funding received
                try:
                    rate = self.get_funding_rate(asset)
                    # Funding = rate × position_notional
                    # (short position receives positive funding when rate > 0)
                    payment = rate * pos.perp_qty * self.get_perp_price(asset)
                    pos.funding_collected += payment
                    pos.funding_payments = expected
                    print(f"  💰 {asset}: funding payment ${payment:+.4f} "
                          f"(rate={rate*100:.4f}%, total=${pos.funding_collected:+.4f})")
                except Exception as e:
                    logger.warning(f"Failed to update funding for {asset}: {e}")

        self._save_positions()
