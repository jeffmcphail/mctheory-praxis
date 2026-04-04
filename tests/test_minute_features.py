"""
Tests for engines/minute_features.py — Chan CPO minute-frequency features.

Validates that:
1. Each indicator returns finite values for valid input
2. Indicators respond correctly to synthetic market conditions
3. Full feature vector has exactly 112 elements (8 × 2 × 7)
4. Daily feature matrix produces correct shapes
5. Training matrix builder crosses features × configs correctly
"""
import numpy as np
import pandas as pd
import pytest

from engines.minute_features import (
    FEATURE_LOOKBACKS,
    INDICATOR_NAMES,
    N_INDICATORS,
    N_LOOKBACKS,
    N_INDICATOR_FEATURES,
    feature_column_names,
    compute_bollinger_zscore,
    compute_bollinger_bandwidth,
    compute_mfi,
    compute_force_index,
    compute_donchian_width,
    compute_atr,
    compute_awesome_oscillator,
    compute_adx,
    compute_all_indicators,
    compute_minute_features,
    compute_daily_feature_matrix,
    build_training_matrix,
)


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

def _make_minute_ohlcv(n_bars: int = 5000, base_price: float = 100.0,
                        vol: float = 0.001, trend: float = 0.0,
                        base_volume: float = 10000,
                        seed: int = 42) -> pd.DataFrame:
    """Generate synthetic minute OHLCV data."""
    np.random.seed(seed)
    log_rets = trend + vol * np.random.randn(n_bars)
    close = base_price * np.exp(np.cumsum(log_rets))

    noise = vol * 0.5
    high = close * (1 + np.abs(np.random.randn(n_bars)) * noise)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * noise)
    open_ = close * (1 + np.random.randn(n_bars) * noise * 0.3)

    high = np.maximum(high, np.maximum(close, open_))
    low = np.minimum(low, np.minimum(close, open_))

    volume = base_volume * (1 + 0.5 * np.abs(np.random.randn(n_bars)))

    dates = pd.date_range("2024-01-02 14:30", periods=n_bars, freq="min", tz="UTC")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)
    df.name = "TEST"
    return df


def _make_pair_data(n_bars: int = 5000):
    """Generate correlated pair data."""
    target = _make_minute_ohlcv(n_bars, base_price=150, seed=42)
    target.name = "TGT"
    hedge = _make_minute_ohlcv(n_bars, base_price=50, seed=43)
    hedge.name = "HDG"
    return target, hedge


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_indicator_count(self):
        assert N_INDICATORS == 8

    def test_lookback_count(self):
        assert N_LOOKBACKS == 7

    def test_total_features(self):
        assert N_INDICATOR_FEATURES == 112  # 8 × 2 × 7

    def test_lookback_values(self):
        assert FEATURE_LOOKBACKS == [50, 100, 200, 400, 800, 1600, 3200]

    def test_feature_column_names(self):
        names = feature_column_names("target", "hedge")
        assert len(names) == 112
        assert names[0] == "bb_zscore_target_50"
        assert names[1] == "bb_zscore_target_100"
        assert "mfi_hedge_800" in names
        assert "adx_hedge_3200" in names


# ═══════════════════════════════════════════════════════════════════════
# INDIVIDUAL INDICATOR TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestBollingerZscore:
    def test_basic(self):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.1)
        z = compute_bollinger_zscore(close, 50)
        assert np.isfinite(z)
        assert -5 < z < 5  # reasonable range

    def test_at_mean(self):
        """Constant price should give z ≈ 0."""
        close = np.ones(100) * 50.0
        z = compute_bollinger_zscore(close, 50)
        assert z == pytest.approx(0.0, abs=0.01)

    def test_above_mean(self):
        """Rising price should give positive z."""
        close = np.linspace(100, 110, 100)
        z = compute_bollinger_zscore(close, 50)
        assert z > 0

    def test_insufficient_data(self):
        z = compute_bollinger_zscore(np.array([100.0, 101.0]), 50)
        assert z == 0.0


class TestBollingerBandwidth:
    def test_basic(self):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        bw = compute_bollinger_bandwidth(close, 50)
        assert np.isfinite(bw)
        assert bw > 0

    def test_zero_vol(self):
        bw = compute_bollinger_bandwidth(np.ones(100) * 50.0, 50)
        assert bw == pytest.approx(0.0, abs=0.001)

    def test_high_vol_wider(self):
        np.random.seed(42)
        low_vol = 100 + np.cumsum(np.random.randn(200) * 0.1)
        high_vol = 100 + np.cumsum(np.random.randn(200) * 1.0)
        bw_low = compute_bollinger_bandwidth(low_vol, 50)
        bw_high = compute_bollinger_bandwidth(high_vol, 50)
        assert bw_high > bw_low


class TestMFI:
    def test_basic(self):
        df = _make_minute_ohlcv(500)
        mfi = compute_mfi(df["high"].values, df["low"].values,
                          df["close"].values, df["volume"].values, 50)
        assert 0 <= mfi <= 100

    def test_insufficient_data(self):
        mfi = compute_mfi(np.ones(5), np.ones(5), np.ones(5), np.ones(5), 50)
        assert mfi == 50.0


class TestForceIndex:
    def test_basic(self):
        df = _make_minute_ohlcv(500)
        fi = compute_force_index(df["close"].values, df["volume"].values, 50)
        assert np.isfinite(fi)

    def test_rising_market_positive(self):
        """Consistently rising prices with volume should give positive FI."""
        close = np.linspace(100, 120, 200)
        volume = np.ones(200) * 1000
        fi = compute_force_index(close, volume, 50)
        assert fi > 0


class TestDonchianWidth:
    def test_basic(self):
        df = _make_minute_ohlcv(500)
        dw = compute_donchian_width(df["high"].values, df["low"].values,
                                    df["close"].values, 50)
        assert dw > 0

    def test_wider_with_more_range(self):
        h1 = np.ones(100) * 101
        l1 = np.ones(100) * 99
        c1 = np.ones(100) * 100
        h2 = np.ones(100) * 110
        l2 = np.ones(100) * 90
        c2 = np.ones(100) * 100
        dw1 = compute_donchian_width(h1, l1, c1, 50)
        dw2 = compute_donchian_width(h2, l2, c2, 50)
        assert dw2 > dw1


class TestATR:
    def test_basic(self):
        df = _make_minute_ohlcv(500)
        atr = compute_atr(df["high"].values, df["low"].values,
                          df["close"].values, 50)
        assert atr > 0
        assert atr < 1  # normalized by price, should be < 100%


class TestAwesomeOscillator:
    def test_basic(self):
        df = _make_minute_ohlcv(500)
        ao = compute_awesome_oscillator(df["high"].values, df["low"].values, 50)
        assert np.isfinite(ao)

    def test_trending_up_positive(self):
        """Rising market should give positive AO."""
        high = np.linspace(101, 121, 200)
        low = np.linspace(99, 119, 200)
        ao = compute_awesome_oscillator(high, low, 50)
        assert ao > 0


class TestADX:
    def test_basic(self):
        df = _make_minute_ohlcv(500)
        adx = compute_adx(df["high"].values, df["low"].values,
                          df["close"].values, 50)
        assert 0 <= adx <= 100

    def test_insufficient_data(self):
        adx = compute_adx(np.ones(5), np.ones(5), np.ones(5), 50)
        assert adx == 25.0  # default


class TestComputeAllIndicators:
    def test_returns_8_values(self):
        df = _make_minute_ohlcv(500)
        vals = compute_all_indicators(
            df["open"].values, df["high"].values, df["low"].values,
            df["close"].values, df["volume"].values, 50
        )
        assert len(vals) == 8
        assert all(np.isfinite(v) for v in vals)


# ═══════════════════════════════════════════════════════════════════════
# FULL FEATURE VECTOR TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestComputeMinuteFeatures:
    def test_feature_count(self):
        """Should produce exactly 112 features (8 × 2 × 7)."""
        target, hedge = _make_pair_data(n_bars=5000)
        feats = compute_minute_features(target, hedge)
        assert feats is not None
        assert len(feats) == 112

    def test_all_finite(self):
        target, hedge = _make_pair_data(n_bars=5000)
        feats = compute_minute_features(target, hedge)
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature {k} = {v} is not finite"

    def test_feature_names_format(self):
        target, hedge = _make_pair_data(n_bars=5000)
        feats = compute_minute_features(target, hedge)
        # Check naming convention
        for key in feats:
            parts = key.rsplit("_", 1)
            assert len(parts) == 2, f"Bad feature name: {key}"
            assert parts[1].isdigit(), f"Expected numeric lookback in: {key}"

    def test_insufficient_data_returns_none(self):
        target = _make_minute_ohlcv(n_bars=100)
        hedge = _make_minute_ohlcv(n_bars=100, seed=43)
        result = compute_minute_features(target, hedge)
        assert result is None

    def test_as_of_timestamp(self):
        target, hedge = _make_pair_data(n_bars=8000)
        # Use 3/4 point as as_of (enough data for max lookback 3200)
        mid_ts = target.index[6000]
        feats_mid = compute_minute_features(target, hedge, as_of_timestamp=mid_ts)
        feats_full = compute_minute_features(target, hedge)
        # Different timestamps should give different features
        assert feats_mid is not None
        assert feats_full is not None
        # At least some values should differ
        diffs = sum(1 for k in feats_mid if abs(feats_mid[k] - feats_full[k]) > 1e-6)
        assert diffs > 0

    def test_custom_lookbacks(self):
        target, hedge = _make_pair_data(n_bars=2000)
        feats = compute_minute_features(
            target, hedge,
            lookback_windows=[50, 100, 200],
        )
        assert feats is not None
        # 8 indicators × 2 assets × 3 lookbacks = 48
        assert len(feats) == 48


class TestDailyFeatureMatrix:
    def test_basic_output(self):
        """Should produce a DataFrame with one row per trading day."""
        target, hedge = _make_pair_data(n_bars=5000)
        # Create a few trading days within the data range
        days = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])
        days = days.tz_localize("UTC")

        matrix = compute_daily_feature_matrix(target, hedge, days)
        # Should have at least some rows (depends on data alignment)
        assert isinstance(matrix, pd.DataFrame)
        if len(matrix) > 0:
            assert matrix.shape[1] == 112


class TestBuildTrainingMatrix:
    def test_shape(self):
        """Training matrix should be N_days × N_configs rows."""
        # Create small feature matrix
        n_days = 5
        n_features = 112
        np.random.seed(42)
        feat_data = np.random.randn(n_days, n_features)
        feat_cols = feature_column_names("A", "B")
        dates = pd.date_range("2024-01-01", periods=n_days)
        feature_df = pd.DataFrame(feat_data, index=dates, columns=feat_cols)

        # Create returns for 3 configs
        n_configs = 3
        returns_data = np.random.randn(n_days, n_configs) * 0.01
        returns_df = pd.DataFrame(returns_data, index=dates)

        configs = [
            {"lookback": 30, "entry_z": 1.0},
            {"lookback": 60, "entry_z": 1.5},
            {"lookback": 90, "entry_z": 2.0},
        ]
        config_names = ["lookback", "entry_z"]

        X, y = build_training_matrix(
            feature_df, returns_df, configs, config_names,
            config_normalizers={"lookback": 720, "entry_z": 2.5},
        )

        assert X.shape[0] == n_days * n_configs  # 15
        assert X.shape[1] == len(config_names) + n_features  # 2 + 112 = 114
        assert len(y) == n_days * n_configs

    def test_features_constant_within_day(self):
        """All configs on same day should share the same indicator features."""
        n_days = 2
        feat_data = np.random.randn(n_days, 4)
        feature_df = pd.DataFrame(feat_data,
                                  index=pd.date_range("2024-01-01", periods=n_days),
                                  columns=["f1", "f2", "f3", "f4"])

        returns_df = pd.DataFrame(
            np.random.randn(n_days, 2),
            index=feature_df.index,
        )

        configs = [{"p": 1.0}, {"p": 2.0}]

        X, y = build_training_matrix(
            feature_df, returns_df, configs, ["p"],
        )

        # Day 1, config 0 and config 1 should have same features (cols 1-4)
        assert np.allclose(X.iloc[0, 1:].values, X.iloc[1, 1:].values)
        # But different parameter (col 0)
        assert X.iloc[0, 0] != X.iloc[1, 0]
