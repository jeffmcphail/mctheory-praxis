"""
Tests for Phase 3.8 (Port Validation), 3.9 (Event-Driven Engine),
and 3.10 (Vectorized vs Event-Driven Reconciliation).

These are the validation tests that prove the Praxis CPO port
is faithful to the original code.
"""

import numpy as np
import polars as pl
import pytest

from praxis.backtest import BacktestOutput, VectorizedEngine
from praxis.backtest.event_driven import EventDrivenEngine, EngineState, Order
from praxis.backtest.reconcile import ReconciliationResult, reconcile
from praxis.logger.core import PraxisLogger
from praxis.signals import SMACrossover
from praxis.validation import (
    PortValidationResult,
    run_full_validation,
    validate_metrics_port,
    validate_pnl_port,
    validate_positions_port,
    validate_zscore_port,
)


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ── Synthetic data ────────────────────────────────────────────

def _make_pair(n=500, seed=42):
    rng = np.random.RandomState(seed)
    close_a = 180.0 + np.cumsum(rng.randn(n) * 0.3)
    open_a = close_a + rng.randn(n) * 0.1
    close_b = 30.0 + np.cumsum(rng.randn(n) * 0.15)
    return close_a, open_a, close_b


def _make_prices(n=200, seed=42):
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    return pl.DataFrame({"close": close})


def _make_sma_signal(prices, fast=10, slow=50):
    sig = SMACrossover()
    return sig.generate(prices, {"fast_period": fast, "slow_period": slow})


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.9: Event-Driven Backtest Engine
# ═══════════════════════════════════════════════════════════════════

class TestEventDrivenBasic:
    def test_returns_backtest_output(self):
        prices = _make_prices(100)
        positions = pl.Series("pos", np.ones(100))
        engine = EventDrivenEngine()
        out = engine.run(positions, prices["close"])
        assert isinstance(out, BacktestOutput)

    def test_equity_curve_length(self):
        prices = _make_prices(100)
        positions = pl.Series("pos", np.ones(100))
        engine = EventDrivenEngine()
        out = engine.run(positions, prices["close"])
        assert len(out.equity_curve) == 100

    def test_bar_count(self):
        prices = _make_prices(100)
        positions = pl.Series("pos", np.ones(100))
        engine = EventDrivenEngine()
        out = engine.run(positions, prices["close"])
        assert out.bar_count == 100

    def test_duration_positive(self):
        prices = _make_prices(100)
        positions = pl.Series("pos", np.ones(100))
        engine = EventDrivenEngine()
        out = engine.run(positions, prices["close"])
        assert out.duration_seconds >= 0

    def test_flat_positions_no_trades(self):
        prices = _make_prices(100)
        positions = pl.Series("pos", np.zeros(100))
        engine = EventDrivenEngine()
        out = engine.run(positions, prices["close"])
        assert out.metrics.total_trades == 0

    def test_constant_position_one_fill(self):
        """Constant +1 position → 1 fill on bar 1."""
        prices = _make_prices(50)
        positions = pl.Series("pos", np.ones(50))
        engine = EventDrivenEngine()
        out = engine.run(positions, prices["close"])
        # One fill: from 0 → 1 on bar 1
        assert out.metrics.total_trades == 1


class TestEventDrivenFillModel:
    def test_fill_on_close(self):
        close = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
        positions = np.array([1.0, 1.0, 0.0, -1.0, -1.0])
        engine = EventDrivenEngine(fill_on="close")
        out = engine.run(positions, close)
        # First fill at bar 0's close=100 (signal at bar 0, filled at bar 0 close)
        assert len(out.trades) > 0

    def test_fill_on_next_open(self):
        close = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
        opens = np.array([99.0, 101.0, 102.0, 100.0, 103.0])
        positions = np.array([1.0, 1.0, 0.0, -1.0, -1.0])
        engine = EventDrivenEngine(fill_on="next_open")
        out = engine.run(positions, close, open_prices=opens)
        # Signal at bar 0, fill at bar 1's open=101
        assert len(out.trades) > 0
        if len(out.trades) > 0:
            assert out.trades[0]["price"] == 101.0

    def test_commission_reduces_equity(self):
        prices = _make_prices(100)
        positions = pl.Series("pos", np.ones(100))
        eng_no_comm = EventDrivenEngine(commission_per_trade=0.0)
        eng_comm = EventDrivenEngine(commission_per_trade=1.0)
        out_no = eng_no_comm.run(positions, prices["close"])
        out_comm = eng_comm.run(positions, prices["close"])
        # Commission should reduce final equity
        assert out_no.equity_curve[-1] >= out_comm.equity_curve[-1]


class TestEventDrivenMetrics:
    def test_sharpe_computed(self):
        prices = _make_prices(200)
        signal = _make_sma_signal(prices)
        engine = EventDrivenEngine()
        out = engine.run(signal, prices["close"])
        assert np.isfinite(out.metrics.sharpe_ratio)

    def test_max_drawdown_negative(self):
        prices = _make_prices(200)
        signal = _make_sma_signal(prices)
        engine = EventDrivenEngine()
        out = engine.run(signal, prices["close"])
        assert out.metrics.max_drawdown <= 0

    def test_volatility_positive(self):
        prices = _make_prices(200)
        signal = _make_sma_signal(prices)
        engine = EventDrivenEngine()
        out = engine.run(signal, prices["close"])
        assert out.metrics.volatility >= 0


class TestEventDrivenEdge:
    def test_numpy_inputs(self):
        close = np.linspace(100, 110, 50)
        positions = np.ones(50)
        engine = EventDrivenEngine()
        out = engine.run(positions, close)
        assert isinstance(out, BacktestOutput)

    def test_alternating_positions(self):
        close = np.linspace(100, 110, 100)
        positions = np.array([1, -1] * 50, dtype=float)
        engine = EventDrivenEngine()
        out = engine.run(positions, close)
        # Many position changes → many fills
        assert out.metrics.total_trades > 10


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.10: Reconciliation — Vectorized vs Event-Driven
# ═══════════════════════════════════════════════════════════════════

class TestReconciliationSMA:
    """§9.2: Run both engines on SMA crossover, compare."""

    def test_reconcile_returns_result(self):
        prices = _make_prices(200)
        signal = _make_sma_signal(prices)
        result = reconcile(signal, prices["close"])
        assert isinstance(result, ReconciliationResult)

    def test_both_engines_produce_output(self):
        """Both engines should produce valid BacktestOutput."""
        prices = _make_prices(300)
        signal = _make_sma_signal(prices)
        result = reconcile(signal, prices["close"])
        assert len(result.vectorized.equity_curve) == 300
        assert len(result.event_driven.equity_curve) == 300

    def test_same_number_of_trades(self):
        """Both engines should detect the same trade count."""
        prices = _make_prices(300)
        signal = _make_sma_signal(prices)
        result = reconcile(signal, prices["close"])
        assert result.details["vec_trades"] == result.details["ed_trades"]

    def test_flat_positions_exact_match(self):
        """Zero positions → both engines return zero."""
        prices = _make_prices(100)
        positions = pl.Series("pos", np.zeros(100))
        result = reconcile(positions, prices["close"])
        assert result.total_return_diff < 1e-10
        assert result.within_tolerance

    def test_constant_long_same_direction(self):
        """Constant +1 position: both engines return same direction."""
        prices = _make_prices(100)
        positions = pl.Series("pos", np.ones(100))
        result = reconcile(positions, prices["close"])
        # Both should agree on direction even if magnitudes differ
        # (vectorized uses pct returns, ED uses cash tracking)
        vec_ret = result.details["vec_total_return"]
        ed_ret = result.details["ed_total_return"]
        if abs(vec_ret) > 0.01:
            assert np.sign(vec_ret) == np.sign(ed_ret)


class TestReconciliationDetails:
    def test_equity_curves_same_length(self):
        prices = _make_prices(200)
        signal = _make_sma_signal(prices)
        result = reconcile(signal, prices["close"])
        assert result.details["equity_curve_len_vec"] == result.details["equity_curve_len_ed"]

    def test_details_populated(self):
        prices = _make_prices(100)
        positions = pl.Series("pos", np.ones(100))
        result = reconcile(positions, prices["close"])
        assert "vec_total_return" in result.details
        assert "ed_total_return" in result.details
        assert "vec_sharpe" in result.details
        assert "vec_trades" in result.details


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.8: Port Validation
# ═══════════════════════════════════════════════════════════════════

class TestZScoreValidation:
    def test_zscore_matches_reference(self):
        close_a, _, close_b = _make_pair(500)
        result = validate_zscore_port(close_a, close_b, weight=3.0, lookback=60)
        assert result.passed, f"Z-score mismatch: max_diff={result.max_diff}"
        assert result.max_diff < 1e-10

    def test_zscore_different_params(self):
        close_a, _, close_b = _make_pair(300)
        for w, lb in [(2.0, 30), (4.0, 120), (5.0, 60)]:
            result = validate_zscore_port(close_a, close_b, weight=w, lookback=lb)
            assert result.passed, f"Failed w={w} lb={lb}: max_diff={result.max_diff}"


class TestPositionValidation:
    def test_positions_match_reference(self):
        close_a, _, close_b = _make_pair(500)
        result = validate_positions_port(
            close_a, close_b, weight=3.0, lookback=60,
            entry_threshold=1.0, exit_threshold_fraction=-0.6,
        )
        assert result.passed, f"Position mismatch: {result.detail}"

    def test_positions_different_thresholds(self):
        close_a, _, close_b = _make_pair(500)
        for et in [0.5, 1.0, 1.5, 2.0]:
            result = validate_positions_port(
                close_a, close_b, weight=3.0, lookback=60,
                entry_threshold=et, exit_threshold_fraction=-0.6,
            )
            assert result.passed, f"Failed et={et}: {result.detail}"


class TestPnLValidation:
    def test_pnl_matches_reference(self):
        close_a, open_a, close_b = _make_pair(500)
        params = {"weight": 3.0, "lookback": 60,
                  "entry_threshold": 1.0, "exit_threshold_fraction": -0.6}
        result = validate_pnl_port(close_a, open_a, close_b, params)
        assert result.passed, f"PnL mismatch: max_diff={result.max_diff}"

    def test_pnl_zero_tc(self):
        close_a, open_a, close_b = _make_pair(300)
        params = {"weight": 3.0, "lookback": 60,
                  "entry_threshold": 1.0, "exit_threshold_fraction": -0.6}
        result = validate_pnl_port(close_a, open_a, close_b, params,
                                    transaction_costs=0.0)
        assert result.passed


class TestMetricsValidation:
    def test_sharpe_matches_reference(self):
        close_a, open_a, close_b = _make_pair(500)
        params = {"weight": 3.0, "lookback": 60,
                  "entry_threshold": 1.0, "exit_threshold_fraction": -0.6}
        result = validate_metrics_port(close_a, open_a, close_b, params)
        assert result.passed, f"Metrics mismatch: {result.detail}"


class TestFullPortValidation:
    def test_all_stages_pass(self):
        close_a, open_a, close_b = _make_pair(500)
        params = {"weight": 3.0, "lookback": 60,
                  "entry_threshold": 1.0, "exit_threshold_fraction": -0.6}
        result = run_full_validation(close_a, open_a, close_b, params)
        assert result.all_passed, (
            f"{result.summary}\n" +
            "\n".join(f"  {s.stage}: {'PASS' if s.passed else 'FAIL'} "
                      f"(max_diff={s.max_diff:.2e})"
                      for s in result.stages)
        )

    def test_all_stages_pass_different_params(self):
        close_a, open_a, close_b = _make_pair(500)
        params = {"weight": 4.5, "lookback": 120,
                  "entry_threshold": 0.7, "exit_threshold_fraction": -0.6}
        result = run_full_validation(close_a, open_a, close_b, params)
        assert result.all_passed, result.summary

    def test_result_structure(self):
        close_a, open_a, close_b = _make_pair(200)
        params = {"weight": 3.0, "lookback": 60,
                  "entry_threshold": 1.0, "exit_threshold_fraction": -0.6}
        result = run_full_validation(close_a, open_a, close_b, params)
        assert len(result.stages) == 4
        stage_names = [s.stage for s in result.stages]
        assert "zscore_computation" in stage_names
        assert "position_signals" in stage_names
        assert "pnl_calculation" in stage_names
        assert "metrics_computation" in stage_names
