"""
Tests for engines/regime_engine.py — 12-class regime matrix detector.

Tests each regime class independently with synthetic data designed to
trigger specific states, then tests the full engine integration.
"""
import numpy as np
import pandas as pd
import pytest

from engines.regime_engine import (
    RegimeEngine,
    RegimeState,
    REGIME_CLASSES,
    REGIME_CLASS_NAMES,
    REGIME_STATE_RANGES,
    compute_trend_regime,
    compute_vol_regime,
    compute_serial_corr_regime,
    compute_microstructure_regime,
    compute_funding_regime,
    compute_liquidity_regime,
    compute_cross_asset_corr_regime,
    compute_volume_participation_regime,
    compute_term_structure_regime,
    compute_dispersion_regime,
    compute_rv_iv_regime,
    _hurst_rs,
    _variance_ratio,
    _garman_klass_bars,
    _corwin_schultz_spread,
    _percentile_rank,
    _z_score,
)


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES — synthetic data generators
# ═══════════════════════════════════════════════════════════════════════

def _make_ohlcv_hourly(n_days: int = 60, bars_per_day: int = 24,
                        trend: float = 0.0, vol: float = 0.02,
                        base_price: float = 100.0,
                        base_volume: float = 1000.0) -> pd.DataFrame:
    """Generate synthetic hourly OHLCV data."""
    n_bars = n_days * bars_per_day
    np.random.seed(42)

    # Generate close prices with trend + noise
    log_returns = trend / bars_per_day + vol / np.sqrt(bars_per_day) * np.random.randn(n_bars)
    log_prices = np.cumsum(log_returns)
    close = base_price * np.exp(log_prices)

    # Generate OHLC from close
    noise = vol / np.sqrt(bars_per_day) * 0.5
    high = close * (1 + np.abs(np.random.randn(n_bars)) * noise)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * noise)
    open_ = close * (1 + np.random.randn(n_bars) * noise * 0.3)

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(close, open_))
    low = np.minimum(low, np.minimum(close, open_))

    volume = base_volume * (1 + 0.5 * np.random.randn(n_bars))
    volume = np.maximum(volume, 10)

    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def _make_trending_data(direction: int = 1, strength: str = "strong") -> pd.DataFrame:
    """Generate data with a clear trend."""
    trend_rate = direction * (0.005 if strength == "strong" else 0.002)
    return _make_ohlcv_hourly(n_days=60, trend=trend_rate, vol=0.01)


def _make_choppy_data() -> pd.DataFrame:
    """Generate data with no clear trend (mean-reverting)."""
    df = _make_ohlcv_hourly(n_days=60, trend=0.0, vol=0.02)
    # Add mean-reversion by smoothing
    close = df["close"].values
    ma = pd.Series(close).rolling(48).mean().bfill().values
    # Pull close toward MA
    close = 0.7 * close + 0.3 * ma
    df["close"] = close
    df["high"] = np.maximum(df["high"].values, close)
    df["low"] = np.minimum(df["low"].values, close)
    return df


def _make_high_vol_data() -> pd.DataFrame:
    """Generate data with high volatility."""
    return _make_ohlcv_hourly(n_days=60, vol=0.10, trend=0.0)


def _make_low_vol_data() -> pd.DataFrame:
    """Generate data with very low volatility."""
    return _make_ohlcv_hourly(n_days=60, vol=0.002, trend=0.0)


def _make_funding_rates(n_payments: int = 270, mean_rate: float = 0.0001) -> np.ndarray:
    """Generate synthetic 8h funding rate series (3/day × 90 days)."""
    np.random.seed(42)
    return mean_rate + 0.00005 * np.random.randn(n_payments)


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_percentile_rank(self):
        history = np.arange(100, dtype=float)
        assert _percentile_rank(50.0, history) == pytest.approx(0.51, abs=0.01)
        assert _percentile_rank(0.0, history) == pytest.approx(0.01, abs=0.01)
        assert _percentile_rank(99.0, history) == pytest.approx(1.0, abs=0.01)

    def test_percentile_rank_insufficient_data(self):
        assert _percentile_rank(5.0, np.array([1, 2, 3])) == 0.5

    def test_z_score(self):
        history = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 11.0, 12.0, 10.0, 11.0, 12.0])
        z = _z_score(15.0, history)
        assert z > 2.0  # well above mean

    def test_z_score_insufficient_data(self):
        assert _z_score(5.0, np.array([1, 2, 3])) == 0.0

    def test_garman_klass_bars(self):
        o = np.array([100.0, 101.0, 102.0])
        h = np.array([102.0, 103.0, 104.0])
        lo = np.array([99.0, 100.0, 101.0])
        c = np.array([101.0, 102.0, 103.0])
        gk = _garman_klass_bars(o, h, lo, c)
        assert len(gk) == 3
        assert all(np.isfinite(gk))
        assert all(gk >= 0)  # variance is non-negative (approximately)

    def test_hurst_random_walk(self):
        """Random walk should have Hurst ≈ 0.5."""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(2000)) + 100
        h = _hurst_rs(prices)
        assert 0.35 < h < 0.65, f"Random walk Hurst = {h}, expected ~0.5"

    def test_hurst_trending(self):
        """Strongly trending series should have Hurst > 0.5."""
        np.random.seed(42)
        prices = np.cumsum(0.1 + 0.05 * np.random.randn(2000))
        h = _hurst_rs(prices)
        assert h > 0.5, f"Trending Hurst = {h}, expected > 0.5"

    def test_hurst_insufficient_data(self):
        assert _hurst_rs(np.array([1.0, 2.0, 3.0])) == 0.5

    def test_variance_ratio_random_walk(self):
        """Random walk should have VR ≈ 1.0."""
        np.random.seed(42)
        prices = np.exp(np.cumsum(np.random.randn(1000) * 0.01))
        vr = _variance_ratio(prices, q=10)
        assert 0.7 < vr < 1.3, f"Random walk VR = {vr}, expected ~1.0"

    def test_variance_ratio_insufficient_data(self):
        assert _variance_ratio(np.array([1.0, 2.0, 3.0]), q=10) == 1.0


# ═══════════════════════════════════════════════════════════════════════
# INDIVIDUAL REGIME CLASS TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestTrendRegime:
    def test_no_trend(self):
        df = _make_choppy_data()
        close = df["close"].values
        # Aggregate to daily
        daily_c = close[::24]
        daily_h = np.array([np.max(close[i*24:(i+1)*24]) for i in range(len(daily_c))])
        daily_l = np.array([np.min(close[i*24:(i+1)*24]) for i in range(len(daily_c))])
        state, raw = compute_trend_regime(daily_c, daily_h, daily_l)
        assert state in REGIME_STATE_RANGES["A"]  # valid state
        assert "adx_14" in raw

    def test_insufficient_data(self):
        state, raw = compute_trend_regime(
            np.array([100.0] * 10),
            np.array([101.0] * 10),
            np.array([99.0] * 10),
        )
        assert state == 0
        assert np.isnan(raw["adx_14"])

    def test_raw_features_present(self):
        df = _make_ohlcv_hourly(n_days=60)
        daily_c = df["close"].values[::24]
        daily_h = np.array([np.max(df["high"].values[i*24:(i+1)*24]) for i in range(len(daily_c))])
        daily_l = np.array([np.min(df["low"].values[i*24:(i+1)*24]) for i in range(len(daily_c))])
        state, raw = compute_trend_regime(daily_c, daily_h, daily_l)
        assert "adx_14" in raw
        assert "ret_1d" in raw
        assert "ret_7d" in raw
        assert "ret_30d" in raw
        assert state in REGIME_STATE_RANGES["A"]


class TestVolRegime:
    def test_normal_vol(self):
        df = _make_ohlcv_hourly(n_days=60, vol=0.02)
        state_b, state_c, raw = compute_vol_regime(
            df["open"].values, df["high"].values,
            df["low"].values, df["close"].values,
        )
        assert state_b in REGIME_STATE_RANGES["B"]
        assert state_c in REGIME_STATE_RANGES["C"]
        assert "rv_1d" in raw
        assert raw["rv_1d"] > 0

    def test_high_vol_spike(self):
        # Create data where recent vol is much higher than historical
        df = _make_ohlcv_hourly(n_days=60, vol=0.01)
        # Inject high vol in last day
        idx = len(df) - 24
        df.iloc[idx:, df.columns.get_loc("high")] *= 1.1
        df.iloc[idx:, df.columns.get_loc("low")] *= 0.9
        state_b, state_c, raw = compute_vol_regime(
            df["open"].values, df["high"].values,
            df["low"].values, df["close"].values,
        )
        # Recent vol should be elevated or spike
        assert state_b >= 2 or state_c == 1  # elevated/spike or expanding

    def test_insufficient_data(self):
        state_b, state_c, raw = compute_vol_regime(
            np.ones(10), np.ones(10) * 1.01,
            np.ones(10) * 0.99, np.ones(10),
        )
        assert state_b == 0
        assert state_c == 0


class TestSerialCorrRegime:
    def test_random_walk(self):
        np.random.seed(42)
        n = 21 * 24
        close = np.exp(np.cumsum(np.random.randn(n) * 0.01)) * 100
        state, raw = compute_serial_corr_regime(close)
        assert abs(state) <= 1  # should be near zero
        assert "hurst" in raw
        assert 0.3 < raw["hurst"] < 0.7

    def test_insufficient_data(self):
        state, raw = compute_serial_corr_regime(np.ones(10))
        assert state == 0
        assert raw["hurst"] == 0.5

    def test_state_in_range(self):
        df = _make_ohlcv_hourly(n_days=30)
        state, raw = compute_serial_corr_regime(df["close"].values)
        assert state in REGIME_STATE_RANGES["D"]


class TestMicrostructureRegime:
    def test_balanced_market(self):
        df = _make_ohlcv_hourly(n_days=5)
        state, raw = compute_microstructure_regime(
            df["open"].values, df["high"].values,
            df["low"].values, df["close"].values,
            df["volume"].values,
        )
        assert state in REGIME_STATE_RANGES["E"]
        assert "ofi_24h" in raw

    def test_buy_pressure(self):
        """When close >> open consistently, OFI should be positive."""
        n = 48
        open_ = np.ones(n) * 100
        close = np.ones(n) * 103  # consistently closing higher
        high = np.ones(n) * 104
        low = np.ones(n) * 99
        volume = np.ones(n) * 1000
        state, raw = compute_microstructure_regime(open_, high, low, close, volume)
        assert raw["ofi_24h"] > 0
        assert state >= 0  # should be balanced or buy

    def test_sell_pressure(self):
        """When close << open consistently, OFI should be negative."""
        n = 48
        open_ = np.ones(n) * 103
        close = np.ones(n) * 100
        high = np.ones(n) * 104
        low = np.ones(n) * 99
        volume = np.ones(n) * 1000
        state, raw = compute_microstructure_regime(open_, high, low, close, volume)
        assert raw["ofi_24h"] < 0
        assert state <= 0


class TestFundingRegime:
    def test_neutral_funding(self):
        rates = np.zeros(100) + 0.00001  # very small positive
        state, raw = compute_funding_regime(rates)
        assert state == 0
        assert "fr_ann" in raw

    def test_heavily_long(self):
        rates = np.ones(100) * 0.001  # very high positive funding
        oi = np.linspace(1000, 1200, 100)  # OI increasing
        state, raw = compute_funding_regime(rates, oi)
        assert state >= 1  # long positioning

    def test_heavily_short(self):
        rates = np.ones(100) * -0.001  # very negative funding
        oi = np.linspace(1000, 800, 100)  # OI decreasing
        state, raw = compute_funding_regime(rates, oi)
        assert state <= -1  # short positioning

    def test_insufficient_data(self):
        state, raw = compute_funding_regime(np.array([0.0001]))
        assert state == 0


class TestLiquidityRegime:
    def test_normal_liquidity(self):
        df = _make_ohlcv_hourly(n_days=60)
        state, raw = compute_liquidity_regime(
            df["high"].values, df["low"].values,
            df["close"].values, df["volume"].values,
        )
        assert state in REGIME_STATE_RANGES["G"]
        assert "cs_spread" in raw
        assert "vol_z" in raw

    def test_corwin_schultz_positive(self):
        """CS spread should be non-negative."""
        np.random.seed(42)
        h = 100 + np.cumsum(np.random.randn(20) * 0.5) + 2
        lo = h - np.abs(np.random.randn(20)) * 3 - 1
        spread = _corwin_schultz_spread(h, lo)
        assert spread >= 0


class TestCrossAssetCorrRegime:
    def test_uncorrelated_assets(self):
        np.random.seed(42)
        rets = {
            "A": np.random.randn(200),
            "B": np.random.randn(200),
            "C": np.random.randn(200),
        }
        state, raw = compute_cross_asset_corr_regime(rets, window=168)
        assert state == 0  # should be idiosyncratic
        assert abs(raw["mean_pairwise_corr"]) < 0.3

    def test_highly_correlated(self):
        np.random.seed(42)
        base = np.random.randn(200)
        rets = {
            "A": base + 0.1 * np.random.randn(200),
            "B": base + 0.1 * np.random.randn(200),
            "C": base + 0.1 * np.random.randn(200),
        }
        state, raw = compute_cross_asset_corr_regime(rets, window=168)
        assert state >= 1  # should be normal or crisis
        assert raw["mean_pairwise_corr"] > 0.5

    def test_single_asset(self):
        state, raw = compute_cross_asset_corr_regime({"A": np.random.randn(200)})
        assert state == 0


class TestVolumeParticipationRegime:
    def test_normal_volume(self):
        df = _make_ohlcv_hourly(n_days=60)
        state, raw = compute_volume_participation_regime(
            df["close"].values, df["volume"].values,
        )
        assert state in REGIME_STATE_RANGES["I"]
        assert "vol_ratio_24h" in raw

    def test_capitulation_volume(self):
        """Very high volume should trigger capitulation state."""
        df = _make_ohlcv_hourly(n_days=60, base_volume=1000)
        # Spike volume in last day
        vol = df["volume"].values.copy()
        vol[-24:] = 5000  # 5x normal
        state, raw = compute_volume_participation_regime(
            df["close"].values, vol,
        )
        assert raw["vol_ratio_24h"] > 3.0


class TestTermStructureRegime:
    def test_flat_term_structure(self):
        rates = np.ones(270) * 0.0001  # constant funding
        state, raw = compute_term_structure_regime(rates)
        assert state == 0
        assert abs(raw["fr_slope"]) < 0.15

    def test_contango(self):
        """30d rate much higher than 7d → contango."""
        rates = np.ones(270) * 0.0001
        # Make older rates higher (30d avg will be higher)
        rates[:180] = 0.0003  # older rates higher
        rates[180:] = 0.0001  # recent rates lower
        state, raw = compute_term_structure_regime(rates)
        # 30d avg includes the high rates, 7d avg is lower
        assert raw["fr_30d_ann"] > raw["fr_7d_ann"]

    def test_insufficient_data(self):
        state, raw = compute_term_structure_regime(np.ones(10) * 0.0001)
        assert state == 0


class TestDispersionRegime:
    def test_low_dispersion(self):
        rets = {"A": 0.01, "B": 0.011, "C": 0.009, "D": 0.01}
        state, raw = compute_dispersion_regime(rets)
        assert state in REGIME_STATE_RANGES["K"]
        assert raw["dispersion"] < 0.01

    def test_high_dispersion(self):
        rets = {"A": 0.10, "B": -0.15, "C": 0.05, "D": -0.20}
        state, raw = compute_dispersion_regime(rets)
        assert raw["dispersion"] > 0.10


class TestRvIvRegime:
    def test_no_dvol(self):
        state, raw = compute_rv_iv_regime(0.5)
        assert state == 0
        assert np.isnan(raw["dvol"])

    def test_expensive_vol(self):
        state, raw = compute_rv_iv_regime(rv_current=40.0, dvol=60.0)
        assert state == 1  # IV > RV → expensive

    def test_cheap_vol(self):
        state, raw = compute_rv_iv_regime(rv_current=60.0, dvol=40.0)
        assert state == -1  # IV < RV → cheap

    def test_balanced(self):
        state, raw = compute_rv_iv_regime(rv_current=50.0, dvol=52.0)
        assert state == 0  # within ±10%


# ═══════════════════════════════════════════════════════════════════════
# FULL ENGINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestRegimeEngine:
    def test_basic_compute(self):
        """Engine should compute all 12 states from OHLCV alone."""
        engine = RegimeEngine(bars_per_day=24)
        df = _make_ohlcv_hourly(n_days=60)
        state = engine.compute(ohlcv_hourly=df)

        assert isinstance(state, RegimeState)
        assert len(state.vector) == 12
        # A-E, G, I should be computed (no funding/universe needed)
        for c in ["A", "B", "C", "D", "E", "G", "I"]:
            assert c in state.states
        # F, H, J, K should be missing (no funding/universe data)
        for c in ["F", "H", "J", "K"]:
            assert c in state.missing

    def test_with_funding_data(self):
        """Engine should compute F and J when funding data provided."""
        engine = RegimeEngine(bars_per_day=24)
        df = _make_ohlcv_hourly(n_days=60)
        fr = _make_funding_rates(n_payments=270)

        state = engine.compute(ohlcv_hourly=df, funding_rates=fr)
        assert "F" in state.states
        assert "J" in state.states
        assert "F" not in state.missing
        assert "J" not in state.missing

    def test_with_universe_data(self):
        """Engine should compute H and K when universe data provided."""
        engine = RegimeEngine(bars_per_day=24)
        df = _make_ohlcv_hourly(n_days=60)
        universe = {
            "BTC": _make_ohlcv_hourly(n_days=60, base_price=50000),
            "ETH": _make_ohlcv_hourly(n_days=60, base_price=3000),
            "SOL": _make_ohlcv_hourly(n_days=60, base_price=100),
        }

        state = engine.compute(ohlcv_hourly=df, universe_ohlcv=universe)
        assert "H" in state.states
        assert "K" in state.states

    def test_vector_length(self):
        engine = RegimeEngine()
        df = _make_ohlcv_hourly(n_days=60)
        state = engine.compute(ohlcv_hourly=df)
        assert len(state.vector) == 12

    def test_named_states(self):
        engine = RegimeEngine()
        df = _make_ohlcv_hourly(n_days=60)
        state = engine.compute(ohlcv_hourly=df)
        named = state.named_states
        assert "trend" in named
        assert "vol_level" in named
        assert "serial_corr" in named

    def test_to_feature_row(self):
        engine = RegimeEngine()
        df = _make_ohlcv_hourly(n_days=60)
        state = engine.compute(ohlcv_hourly=df)
        row = state.to_feature_row()
        assert "regime_trend" in row
        assert "regime_vol_level" in row
        assert "regime_serial_corr" in row
        # Raw features should also be present
        assert any(k.startswith("D_") for k in row)
        assert any(k.startswith("BC_") for k in row)

    def test_all_states_in_valid_ranges(self):
        """Every computed state must be within its defined range."""
        engine = RegimeEngine()
        df = _make_ohlcv_hourly(n_days=60)
        fr = _make_funding_rates()
        universe = {
            "BTC": _make_ohlcv_hourly(n_days=60, base_price=50000),
            "ETH": _make_ohlcv_hourly(n_days=60, base_price=3000),
            "SOL": _make_ohlcv_hourly(n_days=60, base_price=100),
        }
        state = engine.compute(
            ohlcv_hourly=df, funding_rates=fr, universe_ohlcv=universe
        )
        for cls, val in state.states.items():
            assert val in REGIME_STATE_RANGES[cls], (
                f"Class {cls} state {val} not in {REGIME_STATE_RANGES[cls]}"
            )

    def test_raw_features_all_finite(self):
        """All raw features should be finite (no NaN leaking)."""
        engine = RegimeEngine()
        df = _make_ohlcv_hourly(n_days=60)
        state = engine.compute(ohlcv_hourly=df)
        for k, v in state.raw_features.items():
            if "dvol" not in k and "vrp" not in k:  # L is expected NaN without data
                assert np.isfinite(v), f"Feature {k} = {v} is not finite"

    def test_equities_bars_per_day(self):
        """Engine should work with equities schedule (7 bars/day)."""
        engine = RegimeEngine(bars_per_day=7)
        df = _make_ohlcv_hourly(n_days=120, bars_per_day=7)
        state = engine.compute(ohlcv_hourly=df)
        assert isinstance(state, RegimeState)
        assert len(state.vector) == 12


class TestRegimeEngineTimeSeries:
    def test_time_series_output(self):
        """Time series should produce a DataFrame with regime columns."""
        engine = RegimeEngine(bars_per_day=24)
        df = _make_ohlcv_hourly(n_days=60)
        ts = engine.compute_time_series(ohlcv_hourly=df)
        assert isinstance(ts, pd.DataFrame)
        assert len(ts) > 0
        assert "regime_trend" in ts.columns
        assert "regime_serial_corr" in ts.columns

    def test_time_series_has_raw_features(self):
        engine = RegimeEngine(bars_per_day=24)
        df = _make_ohlcv_hourly(n_days=60)
        ts = engine.compute_time_series(ohlcv_hourly=df)
        assert any(c.startswith("D_") for c in ts.columns)

    def test_time_series_insufficient_data(self):
        engine = RegimeEngine(bars_per_day=24)
        df = _make_ohlcv_hourly(n_days=5)  # too short
        ts = engine.compute_time_series(ohlcv_hourly=df)
        assert len(ts) == 0

    def test_time_series_step_size(self):
        engine = RegimeEngine(bars_per_day=24)
        df = _make_ohlcv_hourly(n_days=60)
        ts_daily = engine.compute_time_series(ohlcv_hourly=df, step_bars=24)
        ts_2day = engine.compute_time_series(ohlcv_hourly=df, step_bars=48)
        assert len(ts_daily) > len(ts_2day)


class TestRegimeConstants:
    def test_all_classes_defined(self):
        assert len(REGIME_CLASSES) == 12
        assert REGIME_CLASSES == list("ABCDEFGHIJKL")

    def test_all_classes_have_names(self):
        for c in REGIME_CLASSES:
            assert c in REGIME_CLASS_NAMES

    def test_all_classes_have_state_ranges(self):
        for c in REGIME_CLASSES:
            assert c in REGIME_STATE_RANGES
            assert len(REGIME_STATE_RANGES[c]) >= 3
