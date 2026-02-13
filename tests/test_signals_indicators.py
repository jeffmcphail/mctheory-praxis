"""
Tests for Phase 3.1 (ZScoreSpread + BollingerZScore),
Phase 3.2 (VolatilityTarget sizing), and
Phase 3.3 (7 indicator implementations).

Validation approach:
- ZScoreSpread matches original pair_trade_gld_gdx() z-score logic
- Indicators tested against known mathematical properties
- Sizing scales correctly with volatility
"""

import numpy as np
import polars as pl
import pytest

from praxis.indicators import (
    adx, atr, awesome_oscillator, donchian_width,
    force_index, mfi, zscore,
    _ema, _rma, _sma,
)
from praxis.logger.core import PraxisLogger
from praxis.signals.sizing import EqualWeight, VolatilityTarget
from praxis.signals.zscore import (
    BollingerZScore, ZScoreSpread,
    _ewm_mean, _ewm_std, _ffill_signal,
)


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ── Test data generators ──────────────────────────────────────

def _make_pair_data(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic GLD/GDX-like pair data."""
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64)
    base_a = 180.0 + np.cumsum(rng.randn(n) * 0.5)
    base_b = 30.0 + np.cumsum(rng.randn(n) * 0.3)
    return pl.DataFrame({
        "close_a": base_a,
        "close_b": base_b,
    })


def _make_ohlcv(n: int = 200, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n) * 0.3)
    low = close - np.abs(rng.randn(n) * 0.3)
    open_ = close + rng.randn(n) * 0.1
    volume = (rng.rand(n) * 1_000_000 + 100_000).astype(np.float64)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.1: ZScoreSpread Signal
# ═══════════════════════════════════════════════════════════════════

class TestEWMHelpers:
    def test_ewm_mean_first_value(self):
        data = np.array([10.0, 20.0, 30.0])
        result = _ewm_mean(data, span=3)
        assert result[0] == 10.0  # First value unchanged

    def test_ewm_mean_convergence(self):
        data = np.ones(100) * 50.0
        result = _ewm_mean(data, span=10)
        assert abs(result[-1] - 50.0) < 1e-10

    def test_ewm_std_zero_for_constant(self):
        data = np.ones(100) * 50.0
        result = _ewm_std(data, span=10)
        assert result[-1] < 1e-10

    def test_ewm_std_positive_for_varying(self):
        rng = np.random.RandomState(42)
        data = 50.0 + rng.randn(100) * 5.0
        result = _ewm_std(data, span=20)
        assert result[-1] > 0


class TestZScoreSpread:
    def test_generates_correct_length(self):
        df = _make_pair_data(200)
        sig = ZScoreSpread()
        result = sig.generate(df, {
            "weight": 3.0, "lookback": 60,
            "entry_threshold": 1.0, "exit_threshold_fraction": -0.6,
        })
        assert len(result) == 200

    def test_signal_values_in_range(self):
        df = _make_pair_data(500)
        sig = ZScoreSpread()
        result = sig.generate(df, {
            "weight": 3.0, "lookback": 60,
            "entry_threshold": 1.0, "exit_threshold_fraction": -0.6,
        })
        unique = set(result.to_list())
        assert unique.issubset({-1, 0, 1})

    def test_entry_threshold_respected(self):
        """Higher threshold → fewer signals."""
        df = _make_pair_data(500)
        sig = ZScoreSpread()

        r_low = sig.generate(df, {
            "weight": 3.0, "lookback": 60,
            "entry_threshold": 0.5, "exit_threshold_fraction": -0.6,
        })
        r_high = sig.generate(df, {
            "weight": 3.0, "lookback": 60,
            "entry_threshold": 2.0, "exit_threshold_fraction": -0.6,
        })

        active_low = (r_low != 0).sum()
        active_high = (r_high != 0).sum()
        assert active_low >= active_high

    def test_compute_zscore_returns_float(self):
        df = _make_pair_data(100)
        sig = ZScoreSpread()
        zs = sig.compute_zscore(df, {"weight": 3.0, "lookback": 20})
        assert zs.dtype == pl.Float64

    def test_positions_carry_forward(self):
        """Positions should persist between entry and exit."""
        df = _make_pair_data(500)
        sig = ZScoreSpread()
        result = sig.generate(df, {
            "weight": 3.0, "lookback": 60,
            "entry_threshold": 1.0, "exit_threshold_fraction": -0.6,
        })
        arr = result.to_numpy()
        # Find first non-zero → should have consecutive non-zeros after
        nonzero = np.where(arr != 0)[0]
        if len(nonzero) > 1:
            # Check that at least some consecutive positions exist
            diffs = np.diff(nonzero)
            assert np.any(diffs == 1), "Expected some consecutive held positions"


class TestBollingerZScore:
    def test_generates_correct_length(self):
        df = pl.DataFrame({"close": np.random.randn(100) + 100})
        sig = BollingerZScore()
        result = sig.generate(df, {"period": 20, "threshold": 2.0})
        assert len(result) == 100

    def test_mean_reversion_logic(self):
        """Sudden drop → buy (+1), sudden spike → sell (-1) at transition points."""
        n = 200
        rng = np.random.RandomState(42)
        # Stable at 100, then sudden drop at bar 100, then sudden spike at 150
        close = np.ones(n) * 100.0
        close[100:] = 85.0   # Sudden drop
        close[150:] = 115.0  # Sudden spike
        df = pl.DataFrame({"close": close})

        sig = BollingerZScore()
        result = sig.generate(df, {"period": 20, "threshold": 1.0})
        arr = result.to_numpy()

        # Just after the drop (bars 100-110), price is far below SMA → buy
        assert np.any(arr[100:115] == 1), f"Expected buy after drop, got {arr[100:115]}"
        # Just after the spike (bars 150-165), price is far above SMA → sell
        assert np.any(arr[150:165] == -1), f"Expected sell after spike, got {arr[150:165]}"


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.2: Sizing
# ═══════════════════════════════════════════════════════════════════

class TestVolatilityTarget:
    def test_scales_position(self):
        n = 200
        rng = np.random.RandomState(42)
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        prices = pl.DataFrame({"close": close})
        signal = pl.Series("signal", np.ones(n, dtype=np.int32))

        sizer = VolatilityTarget()
        weights = sizer.size(prices, signal, {
            "target_vol": 0.10, "vol_lookback": 60,
        })
        assert len(weights) == n
        # After warmup, weights should be non-zero
        assert np.any(weights.to_numpy()[65:] != 0)

    def test_zero_signal_zero_weight(self):
        n = 100
        prices = pl.DataFrame({"close": np.linspace(100, 110, n)})
        signal = pl.Series("signal", np.zeros(n, dtype=np.int32))

        sizer = VolatilityTarget()
        weights = sizer.size(prices, signal, {"target_vol": 0.10, "vol_lookback": 20})
        assert np.all(weights.to_numpy() == 0)

    def test_max_leverage_respected(self):
        n = 200
        rng = np.random.RandomState(42)
        # Very low vol data → would need high leverage
        close = np.linspace(100, 100.01, n)
        prices = pl.DataFrame({"close": close})
        signal = pl.Series("signal", np.ones(n, dtype=np.int32))

        sizer = VolatilityTarget()
        weights = sizer.size(prices, signal, {
            "target_vol": 0.50, "vol_lookback": 20, "max_leverage": 2.0,
        })
        assert np.all(np.abs(weights.to_numpy()) <= 2.001)


class TestEqualWeight:
    def test_fixed_weight(self):
        n = 50
        signal = pl.Series("signal", np.array([1, -1, 0, 1, -1] * 10, dtype=np.int32))
        prices = pl.DataFrame({"close": np.ones(n) * 100})
        sizer = EqualWeight()
        weights = sizer.size(prices, signal, {"weight": 0.5})
        assert weights[0] == 0.5
        assert weights[1] == -0.5
        assert weights[2] == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.3: Indicators
# ═══════════════════════════════════════════════════════════════════

class TestZScoreIndicator:
    def test_length(self):
        data = _make_ohlcv(100)
        result = zscore(data["close"], length=20)
        assert len(result) == 100

    def test_nan_before_warmup(self):
        data = _make_ohlcv(100)
        result = zscore(data["close"], length=20)
        assert np.isnan(result[0])
        assert not np.isnan(result[19])

    def test_zero_for_constant(self):
        data = np.ones(50) * 100.0
        result = zscore(data, length=10)
        # Constant series → zscore = 0 (or NaN from zero std)
        valid = result[~np.isnan(result)]
        assert np.all(np.abs(valid) < 1e-10)

    def test_positive_for_above_mean(self):
        data = np.concatenate([np.ones(20) * 100.0, np.ones(10) * 110.0])
        result = zscore(data, length=20)
        # Last values should be positive (above rolling mean)
        assert result[-1] > 0


class TestMFI:
    def test_length(self):
        data = _make_ohlcv(100)
        result = mfi(data["high"], data["low"], data["close"], data["volume"], length=14)
        assert len(result) == 100

    def test_bounded_0_100(self):
        data = _make_ohlcv(200)
        result = mfi(data["high"], data["low"], data["close"], data["volume"], length=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_nan_before_warmup(self):
        data = _make_ohlcv(100)
        result = mfi(data["high"], data["low"], data["close"], data["volume"], length=14)
        assert np.isnan(result[0])


class TestForceIndex:
    def test_length(self):
        data = _make_ohlcv(100)
        result = force_index(data["close"], data["volume"], window=13)
        assert len(result) == 100

    def test_positive_for_uptrend(self):
        close = np.linspace(100, 150, 100)
        volume = np.ones(100) * 1_000_000
        result = force_index(close, volume, window=13)
        # Uptrend → positive force index
        assert result[-1] > 0


class TestDonchianWidth:
    def test_length(self):
        data = _make_ohlcv(100)
        result = donchian_width(data["high"], data["low"], data["close"], window=20)
        assert len(result) == 100

    def test_nan_before_warmup(self):
        data = _make_ohlcv(100)
        result = donchian_width(data["high"], data["low"], data["close"], window=20)
        assert np.isnan(result[0])
        assert not np.isnan(result[19])

    def test_non_negative(self):
        data = _make_ohlcv(100)
        result = donchian_width(data["high"], data["low"], data["close"], window=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)


class TestATR:
    def test_length(self):
        data = _make_ohlcv(100)
        result = atr(data["high"], data["low"], data["close"], window=14)
        assert len(result) == 100

    def test_positive(self):
        data = _make_ohlcv(100)
        result = atr(data["high"], data["low"], data["close"], window=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_higher_volatility_higher_atr(self):
        n = 200
        rng = np.random.RandomState(42)
        # Low vol
        c1 = 100 + np.cumsum(rng.randn(n) * 0.1)
        h1 = c1 + 0.05
        l1 = c1 - 0.05
        # High vol
        c2 = 100 + np.cumsum(rng.randn(n) * 2.0)
        h2 = c2 + 1.0
        l2 = c2 - 1.0

        atr1 = atr(h1, l1, c1, window=14)
        atr2 = atr(h2, l2, c2, window=14)
        assert atr2[-1] > atr1[-1]


class TestAwesomeOscillator:
    def test_length(self):
        data = _make_ohlcv(100)
        result = awesome_oscillator(data["high"], data["low"], window1=5, window2=34)
        assert len(result) == 100

    def test_zero_for_constant(self):
        high = np.ones(100) * 105.0
        low = np.ones(100) * 95.0
        result = awesome_oscillator(high, low, window1=5, window2=34)
        valid = result[~np.isnan(result)]
        assert np.all(np.abs(valid) < 1e-10)


class TestADX:
    def test_length(self):
        data = _make_ohlcv(100)
        result = adx(data["high"], data["low"], data["close"], window=14)
        assert len(result) == 100

    def test_bounded_0_100(self):
        data = _make_ohlcv(200)
        result = adx(data["high"], data["low"], data["close"], window=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100.1)  # Slight tolerance

    def test_strong_trend_high_adx(self):
        """Clear uptrend should have high ADX."""
        n = 200
        close = np.linspace(100, 200, n)
        high = close + 0.5
        low = close - 0.5
        result = adx(high, low, close, window=14)
        # Strong trend → high ADX (>25 is strong)
        assert result[-1] > 20


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

class TestSMA:
    def test_correct_average(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _sma(data, 3)
        assert abs(result[2] - 2.0) < 1e-10
        assert abs(result[4] - 4.0) < 1e-10

    def test_nan_before_warmup(self):
        data = np.array([1.0, 2.0, 3.0])
        result = _sma(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])


class TestEMA:
    def test_converges_to_constant(self):
        data = np.ones(100) * 50.0
        result = _ema(data, 10)
        assert abs(result[-1] - 50.0) < 1e-10


class TestRMA:
    def test_seed_is_sma(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rma(data, 3)
        # First valid = SMA of first 3 = 2.0
        assert abs(result[2] - 2.0) < 1e-10

    def test_nan_before_warmup(self):
        data = np.ones(10) * 5.0
        result = _rma(data, 5)
        assert np.isnan(result[0])
        assert not np.isnan(result[4])
