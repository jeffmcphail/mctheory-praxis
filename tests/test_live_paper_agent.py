"""
Tests for Phase 4.8 (Live Store), 4.9 (Paper Trading), 4.10 (Agent Autonomy).
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.8: Live Store
# ═══════════════════════════════════════════════════════════════════

from praxis.live import (
    LiveStore,
    DuckDBLiveStore,
    LiveOrder,
    LiveFill,
    LivePosition,
    OrderSide,
    OrderStatus,
    OrderType,
)


class TestLiveStoreFactory:
    def test_duckdb_factory(self):
        store = LiveStore.duckdb()
        assert isinstance(store, DuckDBLiveStore)

    def test_postgres_factory(self):
        from praxis.live import PostgresLiveStore
        store = LiveStore.postgres("postgresql://localhost/test")
        assert isinstance(store, PostgresLiveStore)


class TestLiveOrder:
    def test_auto_id(self):
        o = LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100)
        assert o.order_id.startswith("ord_")

    def test_fields(self):
        o = LiveOrder(
            model_id="burgess_v1", asset="AAPL",
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=100, limit_price=185.50,
        )
        assert o.model_id == "burgess_v1"
        assert o.status == OrderStatus.PENDING


class TestLiveFill:
    def test_auto_id(self):
        f = LiveFill(order_id="ord_123", asset="AAPL", price=185.0, quantity=100)
        assert f.fill_id.startswith("fill_")


class TestLivePosition:
    def test_market_value(self):
        p = LivePosition(quantity=100, current_price=185.0)
        assert p.market_value == 18_500.0

    def test_side(self):
        assert LivePosition(quantity=100).side == "long"
        assert LivePosition(quantity=-50).side == "short"
        assert LivePosition(quantity=0).side == "flat"


class TestDuckDBLiveStore:
    @pytest.fixture
    def store(self):
        return LiveStore.duckdb()

    def test_insert_and_get_order(self, store):
        order = LiveOrder(
            model_id="test", asset="AAPL",
            side=OrderSide.BUY, quantity=100,
        )
        oid = store.insert_order(order)
        retrieved = store.get_order(oid)
        assert retrieved is not None
        assert retrieved.asset == "AAPL"
        assert retrieved.quantity == 100

    def test_update_order_status(self, store):
        order = LiveOrder(model_id="test", asset="AAPL", side=OrderSide.BUY, quantity=50)
        oid = store.insert_order(order)
        store.update_order_status(oid, OrderStatus.FILLED)
        retrieved = store.get_order(oid)
        assert retrieved.status == OrderStatus.FILLED

    def test_get_orders_by_model(self, store):
        store.insert_order(LiveOrder(model_id="m1", asset="AAPL", side=OrderSide.BUY, quantity=10))
        store.insert_order(LiveOrder(model_id="m2", asset="GOOG", side=OrderSide.SELL, quantity=20))
        orders = store.get_orders(model_id="m1")
        assert len(orders) == 1
        assert orders[0].model_id == "m1"

    def test_get_orders_by_status(self, store):
        o1 = LiveOrder(model_id="m1", asset="AAPL", side=OrderSide.BUY, quantity=10)
        store.insert_order(o1)
        store.update_order_status(o1.order_id, OrderStatus.FILLED)
        o2 = LiveOrder(model_id="m1", asset="GOOG", side=OrderSide.BUY, quantity=20)
        store.insert_order(o2)

        filled = store.get_orders(status=OrderStatus.FILLED)
        assert len(filled) == 1
        pending = store.get_orders(status=OrderStatus.PENDING)
        assert len(pending) == 1

    def test_insert_and_get_fill(self, store):
        order = LiveOrder(model_id="test", asset="AAPL", side=OrderSide.BUY, quantity=100)
        oid = store.insert_order(order)
        fill = LiveFill(
            order_id=oid, model_id="test", asset="AAPL",
            side=OrderSide.BUY, quantity=100, price=185.0,
        )
        fid = store.insert_fill(fill)
        fills = store.get_fills(order_id=oid)
        assert len(fills) == 1
        assert fills[0].price == 185.0

    def test_upsert_position(self, store):
        pos = LivePosition(
            model_id="test", asset="AAPL",
            quantity=100, avg_entry_price=185.0,
        )
        store.upsert_position(pos)
        positions = store.get_positions(model_id="test")
        assert len(positions) == 1
        assert positions[0].quantity == 100

        # Update
        pos.quantity = 200
        store.upsert_position(pos)
        positions = store.get_positions(model_id="test")
        assert len(positions) == 1
        assert positions[0].quantity == 200

    def test_multiple_positions(self, store):
        store.upsert_position(LivePosition(model_id="m1", asset="AAPL", quantity=100, avg_entry_price=185))
        store.upsert_position(LivePosition(model_id="m1", asset="GOOG", quantity=50, avg_entry_price=142))
        store.upsert_position(LivePosition(model_id="m2", asset="AAPL", quantity=30, avg_entry_price=186))

        m1 = store.get_positions(model_id="m1")
        assert len(m1) == 2
        m2 = store.get_positions(model_id="m2")
        assert len(m2) == 1

    def test_get_nonexistent_order(self, store):
        assert store.get_order("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.9: Paper Trading
# ═══════════════════════════════════════════════════════════════════

from praxis.live.paper import PaperTradingEngine, PaperConfig, PaperState
from praxis.risk import RiskConfig


class TestPaperState:
    def test_initial_nav(self):
        s = PaperState(cash=100_000)
        assert s.nav == 100_000

    def test_nav_with_positions(self):
        s = PaperState(
            cash=50_000,
            positions={"AAPL": 100},
            last_prices={"AAPL": 200.0},
        )
        assert s.nav == 50_000 + 100 * 200

    def test_position_count(self):
        s = PaperState(positions={"AAPL": 100, "GOOG": 0, "MSFT": -50})
        assert s.position_count == 2


class TestPaperTradingEngine:
    @pytest.fixture
    def engine(self):
        return PaperTradingEngine(PaperConfig(initial_cash=100_000))

    def test_initial_state(self, engine):
        assert engine.nav == 100_000
        assert engine.cash == 100_000
        assert engine.positions == {}

    def test_market_buy(self, engine):
        order = LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_order(order)
        fills = engine.on_price_update({"AAPL": 185.0})
        assert len(fills) == 1
        assert fills[0].asset == "AAPL"
        assert engine.positions.get("AAPL") == 100
        assert engine.cash < 100_000  # Paid for shares + slippage + commission

    def test_market_sell(self, engine):
        # Buy first
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        engine.on_price_update({"AAPL": 185.0})

        # Sell
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.SELL, quantity=100))
        fills = engine.on_price_update({"AAPL": 190.0})
        assert len(fills) == 1
        assert "AAPL" not in engine.positions  # Flat

    def test_slippage_applied(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        fills = engine.on_price_update({"AAPL": 100.0})
        # 5 bps slippage → fill at 100.05
        assert fills[0].price > 100.0

    def test_commission_applied(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        fills = engine.on_price_update({"AAPL": 100.0})
        assert fills[0].commission > 0

    def test_limit_buy_not_triggered(self, engine):
        order = LiveOrder(
            asset="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, limit_price=180.0,
        )
        engine.submit_order(order)
        fills = engine.on_price_update({"AAPL": 185.0})
        assert len(fills) == 0  # Price too high

    def test_limit_buy_triggered(self, engine):
        order = LiveOrder(
            asset="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, limit_price=180.0,
        )
        engine.submit_order(order)
        fills = engine.on_price_update({"AAPL": 179.0})
        assert len(fills) == 1

    def test_stop_buy_triggered(self, engine):
        order = LiveOrder(
            asset="AAPL", side=OrderSide.BUY,
            order_type=OrderType.STOP, quantity=100, stop_price=190.0,
        )
        engine.submit_order(order)
        fills = engine.on_price_update({"AAPL": 185.0})
        assert len(fills) == 0
        fills = engine.on_price_update({"AAPL": 191.0})
        assert len(fills) == 1

    def test_cancel_order(self, engine):
        order = LiveOrder(
            asset="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, limit_price=170.0,
        )
        oid = engine.submit_order(order)
        assert engine.cancel_order(oid)
        fills = engine.on_price_update({"AAPL": 160.0})
        assert len(fills) == 0  # Cancelled, won't fill

    def test_close_position(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        engine.on_price_update({"AAPL": 185.0})
        assert "AAPL" in engine.positions
        engine.close_position("AAPL")
        assert "AAPL" not in engine.positions

    def test_nav_tracks(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        engine.on_price_update({"AAPL": 100.0})
        nav_after_buy = engine.nav

        engine.on_price_update({"AAPL": 110.0})
        assert engine.nav > nav_after_buy  # Profit

        engine.on_price_update({"AAPL": 90.0})
        assert engine.nav < nav_after_buy  # Loss

    def test_peak_nav_tracks(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        engine.on_price_update({"AAPL": 100.0})
        engine.on_price_update({"AAPL": 120.0})
        peak = engine.peak_nav
        engine.on_price_update({"AAPL": 90.0})
        assert engine.peak_nav == peak  # Doesn't decrease

    def test_realized_pnl(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        engine.on_price_update({"AAPL": 100.0})
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.SELL, quantity=100))
        engine.on_price_update({"AAPL": 110.0})
        assert engine.state.realized_pnl > 0

    def test_summary(self, engine):
        s = engine.summary()
        assert "nav" in s
        assert "cash" in s
        assert "positions" in s
        assert s["bar_count"] == 0

    def test_fills_in_store(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        engine.on_price_update({"AAPL": 185.0})
        fills = engine.get_fills()
        assert len(fills) == 1

    def test_orders_in_store(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        engine.on_price_update({"AAPL": 185.0})
        orders = engine.get_orders()
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.FILLED

    def test_multi_asset(self, engine):
        engine.submit_order(LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=100))
        engine.submit_order(LiveOrder(asset="GOOG", side=OrderSide.BUY, quantity=50))
        fills = engine.on_price_update({"AAPL": 185.0, "GOOG": 142.0})
        assert len(fills) == 2
        assert engine.state.position_count == 2


class TestPaperWithRisk:
    def test_risk_rejection(self):
        config = PaperConfig(
            initial_cash=100_000,
            risk_config=RiskConfig(max_gross_exposure=0.5),  # 50K limit
        )
        engine = PaperTradingEngine(config)
        # Pre-seed a price
        engine.on_price_update({"AAPL": 100.0})

        # Buy $60K (600 shares × $100) exceeds 50% gross exposure
        order = LiveOrder(asset="AAPL", side=OrderSide.BUY, quantity=600)
        engine.submit_order(order)

        orders = engine.get_orders(status=OrderStatus.REJECTED)
        assert len(orders) == 1


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.10: Agent Autonomy
# ═══════════════════════════════════════════════════════════════════

from praxis.agent import (
    AutonomousAgent,
    AgentConfig,
    AgentPhase,
    DiscoveryResult,
    BacktestCandidate,
    BacktestOutcome,
    AgentReport,
    CycleResult,
)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.discovery_enabled
        assert cfg.backtest_enabled
        assert cfg.report_enabled
        assert cfg.max_concurrent_backtests == 4

    def test_custom(self):
        cfg = AgentConfig(
            sharpe_alert_threshold=3.0,
            top_n_report=5,
        )
        assert cfg.sharpe_alert_threshold == 3.0


class TestAgentReport:
    def test_format_empty(self):
        report = AgentReport()
        text = report.format_text()
        assert "Autonomous Agent Report" in text

    def test_format_with_results(self):
        report = AgentReport(
            n_discovered=10,
            n_backtested=5,
            n_exceptional=2,
            top_results=[
                BacktestOutcome(candidate_id="test_1", sharpe_ratio=2.5, total_return=0.15, max_drawdown=0.08, n_trades=50),
            ],
            alerts=["High Sharpe: test_1 = 2.50"],
        )
        text = report.format_text()
        assert "test_1" in text
        assert "2.50" in text
        assert "Alerts" in text


class TestAutonomousAgent:
    def test_idle_state(self):
        agent = AutonomousAgent()
        assert agent.phase == AgentPhase.IDLE

    def test_simple_cycle(self):
        """Run a cycle with mock discovery and backtest."""
        agent = AutonomousAgent(AgentConfig(
            sharpe_alert_threshold=1.5,
            min_sharpe_for_report=0.5,
        ))

        # Mock discovery
        def mock_discover(**kwargs):
            return DiscoveryResult(
                source="mock",
                method="scan",
                candidates=[
                    {"id": "cand_1", "config": {"n_per_basket": 2}},
                    {"id": "cand_2", "config": {"n_per_basket": 3}},
                ],
            )
        agent.register_discovery(mock_discover)

        # Mock backtest
        def mock_backtest(candidate, **kwargs):
            sharpe = 2.0 if candidate.candidate_id == "cand_1" else 0.8
            return BacktestOutcome(
                candidate_id=candidate.candidate_id,
                sharpe_ratio=sharpe,
                total_return=0.15 if sharpe > 1 else 0.05,
                max_drawdown=0.10,
                n_trades=30,
            )
        agent.register_backtest(mock_backtest)

        result = agent.run_overnight_cycle()
        assert result.success
        assert agent.phase == AgentPhase.COMPLETE
        assert result.report.n_discovered == 2
        assert result.report.n_backtested == 2
        assert result.report.n_exceptional == 2  # Both > 0.5
        assert len(result.report.top_results) == 2
        # cand_1 has higher Sharpe → ranked first
        assert result.report.top_results[0].candidate_id == "cand_1"

    def test_alerts_generated(self):
        agent = AutonomousAgent(AgentConfig(
            sharpe_alert_threshold=1.5,
            drawdown_alert_threshold=0.15,
        ))
        agent.register_discovery(lambda **kw: DiscoveryResult(
            source="m", candidates=[{"id": "a"}],
        ))
        agent.register_backtest(lambda c, **kw: BacktestOutcome(
            candidate_id=c.candidate_id,
            sharpe_ratio=3.0,
            max_drawdown=0.20,
        ))

        result = agent.run_overnight_cycle()
        assert any("High Sharpe" in a for a in result.report.alerts)
        assert any("High Drawdown" in a for a in result.report.alerts)

    def test_discovery_disabled(self):
        agent = AutonomousAgent(AgentConfig(discovery_enabled=False, backtest_enabled=False))
        result = agent.run_overnight_cycle()
        assert result.success
        assert result.report.n_discovered == 0

    def test_no_backtest_fn(self):
        """If no backtest registered, backtesting phase is skipped."""
        agent = AutonomousAgent()
        agent.register_discovery(lambda **kw: DiscoveryResult(
            source="m", candidates=[{"id": "a"}],
        ))
        result = agent.run_overnight_cycle()
        assert result.success
        assert result.report.n_backtested == 0

    def test_failed_discovery_handled(self):
        agent = AutonomousAgent()

        def bad_discover(**kwargs):
            raise ValueError("source unavailable")
        agent.register_discovery(bad_discover)

        result = agent.run_overnight_cycle()
        assert result.success  # Continues despite discovery failure
        assert result.discoveries[0].source == "error"

    def test_failed_backtest_handled(self):
        agent = AutonomousAgent()
        agent.register_discovery(lambda **kw: DiscoveryResult(
            source="m", candidates=[{"id": "a"}],
        ))

        def bad_backtest(c, **kw):
            raise RuntimeError("out of memory")
        agent.register_backtest(bad_backtest)

        result = agent.run_overnight_cycle()
        assert result.success
        assert result.backtests[0].success is False
        assert "out of memory" in result.backtests[0].error

    def test_custom_delivery(self):
        delivered = []

        agent = AutonomousAgent(AgentConfig(
            delivery_methods=["custom"],
        ))
        agent.register_delivery("custom", lambda report: delivered.append(report))
        agent.register_discovery(lambda **kw: DiscoveryResult(source="m", candidates=[]))

        result = agent.run_overnight_cycle()
        assert len(delivered) == 1
        assert isinstance(delivered[0], AgentReport)

    def test_compute_budget(self):
        """Agent stops backtesting when compute budget exhausted."""
        import time as _time

        agent = AutonomousAgent(AgentConfig(max_compute_seconds=0.1))
        agent.register_discovery(lambda **kw: DiscoveryResult(
            source="m",
            candidates=[{"id": f"c_{i}"} for i in range(100)],
        ))

        def slow_backtest(c, **kw):
            _time.sleep(0.05)
            return BacktestOutcome(candidate_id=c.candidate_id, sharpe_ratio=1.0)
        agent.register_backtest(slow_backtest)

        result = agent.run_overnight_cycle()
        # Should have stopped before processing all 100
        assert result.report.n_backtested < 100

    def test_phases_completed_tracking(self):
        agent = AutonomousAgent()
        agent.register_discovery(lambda **kw: DiscoveryResult(source="m", candidates=[]))
        result = agent.run_overnight_cycle()
        assert "discovery" in result.report.phases_completed
        assert "construction" in result.report.phases_completed
        assert "reporting" in result.report.phases_completed
        assert "delivery" in result.report.phases_completed
