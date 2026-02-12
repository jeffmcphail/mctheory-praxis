"""
Tests for Vectorized Backtest Engine (Phase 1.7).

Covers:
- Basic returns calculation (trade-on-next-bar)
- Equity curve correctness
- All BacktestMetrics fields
- Known-answer tests (deterministic data)
- Edge cases (flat positions, single trade, all long, all short)
- Drawdown computation
- Trade extraction (entries, exits, direction, duration)
- Sharpe/Sortino/Calmar ratios
- Polars Series and numpy array inputs
- Integration with executor pipeline
"""

import numpy as np
import polars as pl
import pytest

from praxis.backtest import VectorizedEngine, BacktestMetrics, BacktestOutput
from praxis.config import ModelConfig
from praxis.executor import SimpleExecutor
from praxis.logger.core import PraxisLogger
from praxis.registry import FunctionRegistry
from praxis.registry.defaults import register_defaults


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_singletons():
    FunctionRegistry.reset()
    PraxisLogger.reset()
    yield
    FunctionRegistry.reset()
    PraxisLogger.reset()


@pytest.fixture
def engine():
    return VectorizedEngine()


# ═══════════════════════════════════════════════════════════════════
#  Known-Answer Tests
# ═══════════════════════════════════════════════════════════════════

class TestKnownAnswers:
    """Deterministic tests with hand-calculated expected values."""

    def test_constant_long_linear_prices(self, engine):
        """
        Long +1 with prices [100, 101, 102, 103, 104]
        Returns: [1%, ~0.99%, ~0.98%, ~0.97%]
        Strategy earns all of them.
        Total return: 104/100 - 1 = 4%
        """
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        positions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = engine.run(positions, prices)

        assert abs(result.metrics.total_return - 0.04) < 0.001
        assert result.metrics.total_trades >= 1
        assert result.equity_curve[0] == 1.0
        assert abs(result.equity_curve[-1] - 1.04) < 0.001

    def test_constant_short_rising_prices(self, engine):
        """
        Short -1 with rising prices → lose money.
        Price: 100→104 = 4% up, so short loses ~4%.
        """
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        positions = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])

        result = engine.run(positions, prices)

        assert result.metrics.total_return < 0
        assert abs(result.metrics.total_return - (-0.04)) < 0.002

    def test_flat_position_zero_return(self, engine):
        """All flat → zero return, no trades."""
        prices = np.array([100.0, 105.0, 95.0, 110.0, 90.0])
        positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = engine.run(positions, prices)

        assert result.metrics.total_return == 0.0
        assert result.metrics.total_trades == 0
        assert result.metrics.sharpe_ratio == 0.0
        # Equity flat at 1.0
        assert all(e == 1.0 for e in result.equity_curve)

    def test_single_bar_trade(self, engine):
        """
        Long for one bar then flat.
        prices: [100, 110, 105, 100]
        pos:    [  1,   0,   0,   0]
        Returns bar 0→1: pos[0]*10% = 10%
        """
        prices = np.array([100.0, 110.0, 105.0, 100.0])
        positions = np.array([1.0, 0.0, 0.0, 0.0])

        result = engine.run(positions, prices)

        assert abs(result.metrics.total_return - 0.10) < 0.001

    def test_half_position(self, engine):
        """
        0.5 position with 10% price move → 5% return.
        """
        prices = np.array([100.0, 110.0])
        positions = np.array([0.5, 0.5])

        result = engine.run(positions, prices)

        assert abs(result.metrics.total_return - 0.05) < 0.001


# ═══════════════════════════════════════════════════════════════════
#  Equity Curve
# ═══════════════════════════════════════════════════════════════════

class TestEquityCurve:
    def test_starts_at_initial_capital(self, engine):
        prices = np.array([100.0, 101.0, 102.0])
        positions = np.array([1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        assert result.equity_curve[0] == 1.0

    def test_custom_initial_capital(self, engine):
        prices = np.array([100.0, 101.0, 102.0])
        positions = np.array([1.0, 1.0, 1.0])

        result = engine.run(positions, prices, initial_capital=10000.0)
        assert result.equity_curve[0] == 10000.0
        assert result.equity_curve[-1] > 10000.0

    def test_equity_length(self, engine):
        """Equity curve has n+1 points? No — n points (initial + n-1 returns = n)."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        positions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        # n prices → n-1 returns → equity has 1 (initial) + n-1 = n points
        assert len(result.equity_curve) == len(prices)

    def test_equity_monotonic_for_constant_uptrend(self, engine):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        positions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        # All positive returns → equity strictly increasing
        for i in range(1, len(result.equity_curve)):
            assert result.equity_curve[i] > result.equity_curve[i - 1]


# ═══════════════════════════════════════════════════════════════════
#  Drawdown
# ═══════════════════════════════════════════════════════════════════

class TestDrawdown:
    def test_no_drawdown_for_rising_equity(self, engine):
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        positions = np.array([1.0, 1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        assert result.metrics.max_drawdown == 0.0
        assert result.metrics.max_drawdown_duration_days == 0

    def test_drawdown_for_loss(self, engine):
        """
        Equity: 1.0 → 1.1 → 0.99 → 1.05
        Peak at 1.1, trough at 0.99 → dd = (0.99-1.1)/1.1 = -10%
        """
        prices = np.array([100.0, 110.0, 90.0, 105.0])
        positions = np.array([1.0, 1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        assert result.metrics.max_drawdown < 0
        # Rough check: significant drawdown from 110→90
        assert result.metrics.max_drawdown < -0.10

    def test_drawdown_duration(self, engine):
        """Drawdown duration = bars spent below peak."""
        # Long position: up, down, down, recovery
        prices = np.array([100.0, 110.0, 100.0, 95.0, 115.0])
        positions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        assert result.metrics.max_drawdown_duration_days >= 2


# ═══════════════════════════════════════════════════════════════════
#  Risk Ratios
# ═══════════════════════════════════════════════════════════════════

class TestRiskRatios:
    def test_sharpe_positive_for_good_strategy(self, engine):
        # Steady uptrend → positive Sharpe
        np.random.seed(42)
        n = 252
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + 0.001))
        positions = np.ones(n)

        result = engine.run(positions, prices)
        assert result.metrics.sharpe_ratio > 0

    def test_sharpe_negative_for_bad_strategy(self, engine):
        # Wrong direction: short a rising market
        np.random.seed(42)
        n = 252
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + 0.002))
        positions = -np.ones(n)  # Short

        result = engine.run(positions, prices)
        assert result.metrics.sharpe_ratio < 0

    def test_sortino_higher_than_sharpe_for_positive_skew(self, engine):
        """With positive skew (few large gains, small losses), Sortino > Sharpe."""
        np.random.seed(42)
        n = 252
        # Uptrend with occasional small dips (positive skew)
        raw = np.random.randn(n) * 0.01 + 0.002
        prices = 100 * np.exp(np.cumsum(raw))
        positions = np.ones(n)

        result = engine.run(positions, prices)
        # Both should be positive, Sortino >= Sharpe due to fewer downside obs
        assert result.metrics.sharpe_ratio > 0
        assert result.metrics.sortino_ratio > 0
        assert result.metrics.sortino_ratio >= result.metrics.sharpe_ratio

    def test_calmar_ratio(self, engine):
        np.random.seed(42)
        n = 252
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + 0.001))
        positions = np.ones(n)

        result = engine.run(positions, prices)
        if result.metrics.max_drawdown != 0:
            expected_calmar = abs(
                result.metrics.annualized_return / result.metrics.max_drawdown
            )
            assert abs(result.metrics.calmar_ratio - expected_calmar) < 0.01

    def test_volatility_annualized(self, engine):
        np.random.seed(42)
        n = 252
        daily_vol = 0.01
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * daily_vol))
        positions = np.ones(n)

        result = engine.run(positions, prices)
        # Annualized vol should be roughly daily_vol * sqrt(252)
        expected_ann_vol = daily_vol * np.sqrt(252)
        assert abs(result.metrics.volatility - expected_ann_vol) < expected_ann_vol * 0.5


# ═══════════════════════════════════════════════════════════════════
#  Trade Extraction
# ═══════════════════════════════════════════════════════════════════

class TestTradeExtraction:
    def test_single_long_trade(self, engine):
        positions = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        prices = np.array([100.0, 100.0, 105.0, 110.0, 108.0])

        result = engine.run(positions, prices)
        assert result.metrics.total_trades == 1
        assert result.trades[0]["direction"] == 1
        assert result.trades[0]["entry_bar"] == 1

    def test_single_short_trade(self, engine):
        positions = np.array([0.0, -1.0, -1.0, 0.0])
        prices = np.array([100.0, 100.0, 95.0, 90.0])

        result = engine.run(positions, prices)
        assert result.metrics.total_trades == 1
        assert result.trades[0]["direction"] == -1

    def test_multiple_trades(self, engine):
        # Long, flat, short, flat
        positions = np.array([1.0, 1.0, 0.0, -1.0, -1.0, 0.0])
        prices = np.array([100.0, 105.0, 103.0, 103.0, 98.0, 100.0])

        result = engine.run(positions, prices)
        assert result.metrics.total_trades >= 2

    def test_trade_direction_flip(self, engine):
        """Long → short with no flat bar = two trades."""
        positions = np.array([1.0, 1.0, -1.0, -1.0])
        prices = np.array([100.0, 105.0, 103.0, 100.0])

        result = engine.run(positions, prices)
        assert result.metrics.total_trades == 2
        assert result.trades[0]["direction"] == 1
        assert result.trades[1]["direction"] == -1

    def test_win_rate(self, engine):
        """Two trades: one winner, one loser → 50% win rate."""
        # Trade 1: long 100→110 = win
        # Trade 2: long 110→100 = loss
        positions = np.array([1.0, 0.0, 1.0, 0.0])
        prices = np.array([100.0, 110.0, 110.0, 100.0])

        result = engine.run(positions, prices)
        assert result.metrics.total_trades == 2
        assert abs(result.metrics.win_rate - 0.5) < 0.01

    def test_no_trades_when_flat(self, engine):
        positions = np.array([0.0, 0.0, 0.0, 0.0])
        prices = np.array([100.0, 105.0, 95.0, 100.0])

        result = engine.run(positions, prices)
        assert result.metrics.total_trades == 0
        assert result.metrics.win_rate == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Input Types
# ═══════════════════════════════════════════════════════════════════

class TestInputTypes:
    def test_polars_series_input(self, engine):
        prices = pl.Series([100.0, 101.0, 102.0, 103.0])
        positions = pl.Series([1.0, 1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        assert result.metrics.total_return > 0

    def test_numpy_array_input(self, engine):
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        positions = np.array([1.0, 1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        assert result.metrics.total_return > 0

    def test_length_mismatch_raises(self, engine):
        prices = np.array([100.0, 101.0, 102.0])
        positions = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="length"):
            engine.run(positions, prices)

    def test_too_few_bars_raises(self, engine):
        prices = np.array([100.0])
        positions = np.array([1.0])

        with pytest.raises(ValueError, match="at least 2"):
            engine.run(positions, prices)


# ═══════════════════════════════════════════════════════════════════
#  BacktestOutput
# ═══════════════════════════════════════════════════════════════════

class TestBacktestOutput:
    def test_output_fields(self, engine):
        prices = np.array([100.0, 101.0, 102.0])
        positions = np.array([1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        assert isinstance(result, BacktestOutput)
        assert isinstance(result.metrics, BacktestMetrics)
        assert isinstance(result.equity_curve, np.ndarray)
        assert isinstance(result.daily_returns, np.ndarray)
        assert isinstance(result.trades, list)
        assert result.bar_count == 3
        assert result.duration_seconds >= 0

    def test_metrics_to_dict(self, engine):
        prices = np.array([100.0, 101.0, 102.0])
        positions = np.array([1.0, 1.0, 1.0])

        result = engine.run(positions, prices)
        d = result.metrics.to_dict()
        assert "total_return" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d
        assert "total_trades" in d
        assert len(d) == 13


# ═══════════════════════════════════════════════════════════════════
#  Integration: Executor → Backtest
# ═══════════════════════════════════════════════════════════════════

class TestExecutorIntegration:
    def test_simple_executor_produces_metrics(self):
        reg = FunctionRegistry.instance()
        register_defaults(reg)

        config = ModelConfig.from_dict({
            "model": {"name": "test_sma", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover", "fast_period": 10, "slow_period": 50},
            "sizing": {"method": "fixed_fraction", "fraction": 1.0},
        })

        np.random.seed(42)
        prices = pl.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(200) * 0.5),
        })

        executor = SimpleExecutor(registry=reg)
        result = executor.execute(config, prices)

        assert result.success is True
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "total_trades" in result.metrics
        assert isinstance(result.metrics["total_return"], float)

    def test_end_to_end_yaml(self):
        from praxis.runner import PraxisRunner

        config = ModelConfig.from_yaml_string("""
model:
  name: backtest_e2e
  type: SingleAssetModel
  version: v1.0
signal:
  method: sma_crossover
  fast_period: 10
  slow_period: 50
sizing:
  method: fixed_fraction
  fraction: 0.5
backtest:
  engine: vectorized
""")
        np.random.seed(42)
        prices = pl.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(252) * 0.5),
        })

        runner = PraxisRunner()
        result = runner.run_config(config, prices)

        assert result.success is True
        assert result.metrics["total_trades"] > 0
        assert isinstance(result.metrics["sharpe_ratio"], float)
        assert result.metrics["max_drawdown"] <= 0  # Drawdown is non-positive
