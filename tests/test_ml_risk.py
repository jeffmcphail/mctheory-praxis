"""
Tests for Phase 4.6 (ML Momentum) and Phase 4.7 (Risk Management).
"""

import numpy as np
import pytest

from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


def _trending_prices(n=500, seed=42):
    """Uptrending price series with volatility."""
    rng = np.random.RandomState(seed)
    returns = 0.001 + rng.randn(n) * 0.02
    return 100 * np.exp(np.cumsum(returns))


def _mean_reverting_prices(n=500, seed=42):
    """Mean-reverting around 100."""
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + rng.randn()
    return 100 + x


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.6: Triple Barrier Labeling
# ═══════════════════════════════════════════════════════════════════

from praxis.models.ml_momentum import (
    triple_barrier_label,
    TripleBarrierResult,
    TripleBarrierEvent,
    compute_sample_uniqueness,
    compute_sample_weights,
    PurgedKFold,
    compute_momentum_features,
    build_feature_matrix,
)


class TestTripleBarrier:
    def test_basic_labeling(self):
        prices = _trending_prices()
        result = triple_barrier_label(prices, pt_level=0.05, sl_level=0.05, max_bars=30)
        assert isinstance(result, TripleBarrierResult)
        assert result.n_events > 0
        assert len(result.labels) == result.n_events

    def test_label_values(self):
        prices = _trending_prices()
        result = triple_barrier_label(prices, pt_level=0.03, sl_level=0.03, max_bars=20)
        # Labels should be -1, 0, or +1
        assert set(result.labels).issubset({-1, 0, 1})

    def test_profit_take_hit(self):
        # Price jumps up 10% immediately
        prices = np.array([100, 112, 113, 114, 115], dtype=float)
        result = triple_barrier_label(prices, pt_level=0.05, sl_level=0.05, max_bars=10)
        assert result.events[0].barrier_hit == "profit_take"
        assert result.events[0].label == 1

    def test_stop_loss_hit(self):
        # Price drops 10% immediately
        prices = np.array([100, 88, 87, 86, 85], dtype=float)
        result = triple_barrier_label(prices, pt_level=0.05, sl_level=0.05, max_bars=10)
        assert result.events[0].barrier_hit == "stop_loss"
        assert result.events[0].label == -1

    def test_vertical_barrier(self):
        # Price stays flat → hits vertical
        prices = np.full(20, 100.0)
        result = triple_barrier_label(prices, pt_level=0.10, sl_level=0.10, max_bars=5)
        assert result.events[0].barrier_hit == "vertical"

    def test_entry_exit_indices(self):
        prices = _trending_prices(100)
        result = triple_barrier_label(prices, max_bars=10)
        assert all(result.exit_indices >= result.entry_indices)
        assert all(result.exit_indices - result.entry_indices <= 10)

    def test_returns_reasonable(self):
        prices = _trending_prices()
        result = triple_barrier_label(prices, pt_level=0.05, sl_level=0.05, max_bars=20)
        # Returns should be bounded
        assert all(abs(r) < 0.5 for r in result.returns)

    def test_label_distribution(self):
        prices = _trending_prices()
        result = triple_barrier_label(prices, pt_level=0.03, sl_level=0.03, max_bars=20)
        dist = result.label_distribution
        assert isinstance(dist, dict)
        assert sum(dist.values()) == result.n_events

    def test_min_return_filter(self):
        prices = np.full(20, 100.0)  # Flat prices
        result = triple_barrier_label(prices, pt_level=0.5, sl_level=0.5,
                                      max_bars=5, min_return=0.01)
        # All labels should be 0 since return is ~0
        assert all(l == 0 for l in result.labels)

    def test_custom_side(self):
        prices = _trending_prices(50)
        sides = np.ones(50)
        result = triple_barrier_label(prices, side=sides, max_bars=10)
        assert all(s == 1 for s in result.sides)


class TestSampleUniqueness:
    def test_non_overlapping(self):
        # Non-overlapping samples → uniqueness = 1.0
        entry = np.array([0, 10, 20])
        exit = np.array([5, 15, 25])
        u = compute_sample_uniqueness(entry, exit, n_bars=30)
        np.testing.assert_allclose(u, [1.0, 1.0, 1.0], atol=1e-10)

    def test_fully_overlapping(self):
        # All samples cover same period
        entry = np.array([0, 0, 0])
        exit = np.array([10, 10, 10])
        u = compute_sample_uniqueness(entry, exit, n_bars=20)
        # Each bar has 3 concurrent → uniqueness = 1/3
        np.testing.assert_allclose(u, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_partial_overlap(self):
        entry = np.array([0, 5])
        exit = np.array([10, 15])
        u = compute_sample_uniqueness(entry, exit, n_bars=20)
        # Sample 0: bars 0-10, 5 unique (0-4) + 6 shared (5-10) → avg(1/1,...,1/2,...)
        assert 0 < u[0] < 1
        assert 0 < u[1] < 1


class TestSampleWeights:
    def test_sums_to_one(self):
        entry = np.array([0, 10, 20, 30, 40])
        exit = np.array([5, 15, 25, 35, 45])
        w = compute_sample_weights(entry, exit, n_bars=50)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)

    def test_return_weighting(self):
        entry = np.array([0, 10])
        exit = np.array([5, 15])
        returns = np.array([0.10, 0.01])
        w = compute_sample_weights(entry, exit, n_bars=20, returns=returns)
        # Larger return should get more weight
        assert w[0] > w[1]

    def test_decay_weighting(self):
        entry = np.array([0, 10, 20, 30])
        exit = np.array([5, 15, 25, 35])
        w = compute_sample_weights(entry, exit, n_bars=40, decay_halflife=10)
        # More recent samples should get more weight
        assert w[-1] > w[0]


class TestPurgedKFold:
    def test_n_splits(self):
        cv = PurgedKFold(n_splits=5)
        X = np.random.randn(100, 3)
        entry = np.arange(100)
        exit = np.arange(100) + 5
        exit = np.minimum(exit, 104)
        splits = list(cv.split(X, entry, exit))
        assert len(splits) == 5

    def test_no_overlap_train_test(self):
        cv = PurgedKFold(n_splits=3, embargo_pct=0.0)
        X = np.random.randn(60, 3)
        entry = np.arange(60)
        exit = np.arange(60) + 2
        exit = np.minimum(exit, 61)

        for train_idx, test_idx in cv.split(X, entry, exit):
            # No index should appear in both train and test
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_purging_removes_overlapping(self):
        cv = PurgedKFold(n_splits=2, embargo_pct=0.0)
        n = 100
        X = np.random.randn(n, 3)
        # Long overlapping label periods
        entry = np.arange(n)
        exit = np.minimum(entry + 20, n - 1)

        for train_idx, test_idx in cv.split(X, entry, exit):
            # Training samples should not overlap with test period
            test_min = entry[test_idx].min()
            test_max = exit[test_idx].max()
            for i in train_idx:
                sample_start = entry[i]
                sample_end = exit[i]
                # Should not overlap
                assert sample_end < test_min or sample_start > test_max

    def test_embargo(self):
        cv = PurgedKFold(n_splits=2, embargo_pct=0.10)
        n = 100
        X = np.random.randn(n, 3)
        entry = np.arange(n)
        exit = np.arange(n)  # point-in-time labels

        for train_idx, test_idx in cv.split(X, entry, exit):
            test_max = exit[test_idx].max()
            embargo_end = test_max + int(n * 0.10)
            # No training sample should start within embargo period
            for i in train_idx:
                if entry[i] > test_max:
                    assert entry[i] >= embargo_end

    def test_all_samples_covered(self):
        cv = PurgedKFold(n_splits=3, embargo_pct=0.0)
        n = 60
        X = np.random.randn(n, 3)
        entry = np.arange(n)
        exit = np.arange(n)

        test_sets = set()
        for _, test_idx in cv.split(X, entry, exit):
            test_sets.update(test_idx.tolist())
        # All samples should appear in some test set
        assert len(test_sets) == n


class TestMomentumFeatures:
    def test_computes_features(self):
        prices = _trending_prices()
        features = compute_momentum_features(prices)
        assert "log_return_1" in features
        assert "return_5" in features
        assert "volatility_21" in features
        assert "rsi_14" in features

    def test_feature_lengths(self):
        prices = _trending_prices(200)
        features = compute_momentum_features(prices)
        for name, arr in features.items():
            assert len(arr) == 200, f"Feature {name} has wrong length"

    def test_custom_windows(self):
        prices = _trending_prices()
        features = compute_momentum_features(prices, windows=[3, 7])
        assert "return_3" in features
        assert "return_7" in features
        assert "return_5" not in features

    def test_rsi_bounded(self):
        prices = _trending_prices()
        features = compute_momentum_features(prices)
        rsi = features["rsi_14"][20:]  # After warmup
        assert all(0 <= r <= 100 for r in rsi)


class TestBuildFeatureMatrix:
    def test_basic(self):
        prices = _trending_prices()
        features = compute_momentum_features(prices)
        X, names = build_feature_matrix(features, start_idx=63)
        assert X.shape[0] == 500 - 63
        assert X.shape[1] == len(names)
        assert len(names) > 0


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.7: Risk Management
# ═══════════════════════════════════════════════════════════════════

from praxis.risk import (
    RiskManager,
    RiskConfig,
    PortfolioState,
    ProposedOrder,
    RiskCheckResult,
    RiskAction,
    DrawdownState,
)


class TestRiskManagerBasic:
    def test_approve_small_order(self):
        rm = RiskManager(RiskConfig(max_position_pct=0.10))
        state = PortfolioState(nav=100_000)
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=5_000)
        result = rm.check_order(order, state)
        assert result.approved
        assert result.approved_size == 5_000

    def test_reject_oversized_order(self):
        rm = RiskManager(RiskConfig(max_position_pct=0.10))
        state = PortfolioState(nav=100_000)
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=20_000)
        result = rm.check_order(order, state)
        # Should be reduced or rejected
        assert result.approved_size <= 10_000


class TestPositionLimit:
    def test_within_limit(self):
        rm = RiskManager(RiskConfig(max_position_pct=0.20))
        state = PortfolioState(nav=100_000)
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=15_000)
        result = rm.check_order(order, state)
        assert result.approved

    def test_exceeds_with_existing(self):
        rm = RiskManager(RiskConfig(max_position_pct=0.10))
        state = PortfolioState(
            nav=100_000,
            positions={"AAPL": 8_000},
        )
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=5_000)
        result = rm.check_order(order, state)
        assert result.approved_size <= 2_000 + 1  # Only 2K room left


class TestSectorLimit:
    def test_within_sector_limit(self):
        rm = RiskManager(RiskConfig(max_sector_pct=0.30))
        state = PortfolioState(
            nav=100_000,
            positions={"AAPL": 10_000},
            sectors={"AAPL": "tech", "MSFT": "tech"},
        )
        order = ProposedOrder(asset="MSFT", side=1, size_dollars=15_000, sector="tech")
        result = rm.check_order(order, state)
        assert result.approved

    def test_exceeds_sector_limit(self):
        rm = RiskManager(RiskConfig(max_sector_pct=0.20))
        state = PortfolioState(
            nav=100_000,
            positions={"AAPL": 15_000},
            sectors={"AAPL": "tech", "MSFT": "tech"},
        )
        order = ProposedOrder(asset="MSFT", side=1, size_dollars=10_000, sector="tech")
        result = rm.check_order(order, state)
        assert result.approved_size < 10_000


class TestGrossExposure:
    def test_within_limit(self):
        rm = RiskManager(RiskConfig(max_gross_exposure=2.0))
        state = PortfolioState(
            nav=100_000,
            positions={"AAPL": 50_000, "GOOG": -30_000},
        )
        order = ProposedOrder(asset="MSFT", side=1, size_dollars=50_000)
        result = rm.check_order(order, state)
        assert result.approved

    def test_exceeds_limit(self):
        rm = RiskManager(RiskConfig(max_gross_exposure=1.5))
        state = PortfolioState(
            nav=100_000,
            positions={"AAPL": 100_000, "GOOG": 40_000},
        )
        order = ProposedOrder(asset="MSFT", side=1, size_dollars=20_000)
        result = rm.check_order(order, state)
        assert result.rejected


class TestDrawdown:
    def test_no_drawdown(self):
        rm = RiskManager()
        state = PortfolioState(nav=100_000, peak_nav=100_000)
        dd = rm.check_drawdown(state)
        assert dd.current_drawdown == 0.0
        assert not dd.is_circuit_broken
        assert not dd.is_reduced

    def test_circuit_breaker(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=0.20))
        state = PortfolioState(nav=75_000, peak_nav=100_000)
        dd = rm.check_drawdown(state)
        assert dd.current_drawdown == 0.25
        assert dd.is_circuit_broken

    def test_circuit_breaker_blocks_orders(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=0.15))
        state = PortfolioState(nav=80_000, peak_nav=100_000)
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=1_000)
        result = rm.check_order(order, state)
        assert result.rejected

    def test_reduced_sizing(self):
        rm = RiskManager(RiskConfig(
            drawdown_reduce_pct=0.05,
            max_drawdown_pct=0.20,
        ))
        state = PortfolioState(nav=90_000, peak_nav=100_000)
        dd = rm.check_drawdown(state)
        assert dd.is_reduced
        assert not dd.is_circuit_broken


class TestStopLoss:
    def test_no_stop(self):
        rm = RiskManager(RiskConfig(stop_loss_pct=0.05))
        assert not rm.check_stop_loss("AAPL", 100, 98, side=1)

    def test_stop_triggered(self):
        rm = RiskManager(RiskConfig(stop_loss_pct=0.05))
        assert rm.check_stop_loss("AAPL", 100, 94, side=1)

    def test_short_side(self):
        rm = RiskManager(RiskConfig(stop_loss_pct=0.05))
        # Short position: entry at 100, price goes to 106 = -6% loss
        assert rm.check_stop_loss("AAPL", 100, 106, side=-1)

    def test_atr_stop(self):
        rm = RiskManager(RiskConfig(stop_loss_pct=0.50, stop_loss_atr_mult=2.0))
        # ATR=2, entry=100, 2*ATR stop = 4% → trigger at 96
        assert rm.check_stop_loss("AAPL", 100, 95, side=1, atr=2.0)
        assert not rm.check_stop_loss("AAPL", 100, 97, side=1, atr=2.0)


class TestSizingAdjustment:
    def test_full_size_no_drawdown(self):
        rm = RiskManager(RiskConfig(drawdown_reduce_pct=0.10, max_drawdown_pct=0.20))
        state = PortfolioState(nav=100_000, peak_nav=100_000)
        assert rm.compute_sizing_adjustment(state) == 1.0

    def test_reduced_in_drawdown(self):
        rm = RiskManager(RiskConfig(drawdown_reduce_pct=0.10, max_drawdown_pct=0.20))
        state = PortfolioState(nav=85_000, peak_nav=100_000)  # 15% DD
        adj = rm.compute_sizing_adjustment(state)
        assert 0 < adj < 1

    def test_zero_at_max_drawdown(self):
        rm = RiskManager(RiskConfig(drawdown_reduce_pct=0.10, max_drawdown_pct=0.20))
        state = PortfolioState(nav=80_000, peak_nav=100_000)  # 20% DD
        assert rm.compute_sizing_adjustment(state) < 1e-10


class TestOrderLimits:
    def test_orders_per_bar(self):
        rm = RiskManager(RiskConfig(max_orders_per_bar=5))
        state = PortfolioState(nav=100_000, orders_this_bar=5)
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=1_000)
        result = rm.check_order(order, state)
        assert result.rejected

    def test_daily_turnover(self):
        rm = RiskManager(RiskConfig(max_turnover_daily=0.50))
        state = PortfolioState(nav=100_000, daily_turnover=45_000)
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=10_000)
        result = rm.check_order(order, state)
        assert result.rejected

    def test_cooldown(self):
        rm = RiskManager(RiskConfig(cooldown_bars=5))
        state = PortfolioState(
            nav=100_000,
            current_bar=10,
            last_trade_bar={"AAPL": 8},
        )
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=1_000)
        result = rm.check_order(order, state)
        assert result.rejected
        assert "Cooldown" in result.reasons[0]

    def test_cooldown_passed(self):
        rm = RiskManager(RiskConfig(cooldown_bars=5))
        state = PortfolioState(
            nav=100_000,
            current_bar=20,
            last_trade_bar={"AAPL": 8},
        )
        order = ProposedOrder(asset="AAPL", side=1, size_dollars=1_000)
        result = rm.check_order(order, state)
        assert result.approved


class TestPortfolioState:
    def test_drawdown(self):
        s = PortfolioState(nav=80_000, peak_nav=100_000)
        assert abs(s.drawdown - 0.20) < 1e-10

    def test_gross_exposure(self):
        s = PortfolioState(positions={"A": 50_000, "B": -30_000})
        assert s.gross_exposure == 80_000

    def test_net_exposure(self):
        s = PortfolioState(positions={"A": 50_000, "B": -30_000})
        assert s.net_exposure == 20_000

    def test_leverage(self):
        s = PortfolioState(nav=100_000, positions={"A": 150_000})
        assert s.gross_leverage == 1.5

    def test_sector_exposure(self):
        s = PortfolioState(
            positions={"A": 10_000, "B": 20_000, "C": 5_000},
            sectors={"A": "tech", "B": "tech", "C": "fin"},
        )
        assert s.sector_exposure("tech") == 30_000
        assert s.sector_exposure("fin") == 5_000
