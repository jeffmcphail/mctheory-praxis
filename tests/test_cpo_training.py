"""
Tests for engines/cpo_training.py — CPO training and feature integration.

Tests the feature computation modes, training, prediction, and ablation.
"""
import numpy as np
import pandas as pd
import pytest

from engines.pairs_trading import PairSpec, ParamConfig, generate_param_grid
from engines.cpo_training import (
    _parse_mode,
    compute_features_v2,
    get_feature_columns,
    train_v2,
    predict_v2,
    _make_config_only_features,
    VALID_MODES,
)


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

def _make_pair() -> PairSpec:
    return PairSpec(
        pair_id="A_B", target="A", hedge="B",
        hedge_ratio=1.5, adf_t=-3.5, adf_p=0.01,
        hurst=0.35, half_life=10, composite_score=0.8,
    )


def _make_minute_data(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.001))
    noise = 0.0005
    high = close * (1 + np.abs(np.random.randn(n_bars)) * noise)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * noise)
    open_ = close * (1 + np.random.randn(n_bars) * noise * 0.3)
    high = np.maximum(high, np.maximum(close, open_))
    low = np.minimum(low, np.minimum(close, open_))
    volume = 10000 * (1 + 0.5 * np.abs(np.random.randn(n_bars)))
    dates = pd.date_range("2024-01-02 14:30", periods=n_bars, freq="min", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


def _make_returns_df(pair_id: str = "A_B", n_days: int = 20,
                     n_configs: int = 5) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    start = pd.Timestamp("2024-01-03")
    for d in range(n_days):
        day = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for c in range(n_configs):
            rows.append({
                "pair_id": pair_id,
                "date": day,
                "config_id": c,
                "daily_return": np.random.randn() * 0.01,
                "gross_return": np.random.randn() * 0.01,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# PARSE MODE
# ═══════════════════════════════════════════════════════════════════════

class TestParseMode:
    def test_chan(self):
        assert _parse_mode("chan") == ("chan", None)

    def test_regime(self):
        assert _parse_mode("regime") == ("regime", None)

    def test_hybrid(self):
        assert _parse_mode("hybrid") == ("hybrid", None)

    def test_ablation_single(self):
        assert _parse_mode("ablation:D") == ("ablation", "D")

    def test_ablation_multiple(self):
        assert _parse_mode("ablation:BCD") == ("ablation", "BCD")

    def test_case_insensitive(self):
        assert _parse_mode("Chan") == ("chan", None)
        assert _parse_mode("REGIME") == ("regime", None)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

class TestComputeFeaturesV2:
    def test_chan_mode_feature_count(self):
        """Chan mode should produce 112 features per row."""
        pair = _make_pair()
        minute_data = {
            "A": _make_minute_data(5000, seed=42),
            "B": _make_minute_data(5000, seed=43),
        }
        days = ["2024-01-05", "2024-01-06"]
        df = compute_features_v2([pair], minute_data, days, mode="chan")

        if len(df) > 0:
            feat_cols = get_feature_columns(df)
            assert len(feat_cols) == 112

    def test_regime_mode_has_regime_columns(self):
        """Regime mode should have regime_ prefixed columns."""
        pair = _make_pair()
        minute_data = {
            "A": _make_minute_data(5000, seed=42),
            "B": _make_minute_data(5000, seed=43),
        }
        days = ["2024-01-05"]
        df = compute_features_v2([pair], minute_data, days, mode="regime",
                                 bars_per_day=24)
        if len(df) > 0:
            feat_cols = get_feature_columns(df)
            regime_cols = [c for c in feat_cols if c.startswith("regime_")]
            assert len(regime_cols) > 0

    def test_ablation_filters_classes(self):
        """Ablation mode should only include features from specified class."""
        pair = _make_pair()
        minute_data = {
            "A": _make_minute_data(5000, seed=42),
            "B": _make_minute_data(5000, seed=43),
        }
        days = ["2024-01-05"]

        # Full regime
        df_full = compute_features_v2([pair], minute_data, days, mode="regime",
                                       bars_per_day=24)
        # Ablation: only class D
        df_d = compute_features_v2([pair], minute_data, days, mode="ablation:D",
                                    bars_per_day=24)

        if len(df_full) > 0 and len(df_d) > 0:
            full_cols = get_feature_columns(df_full)
            d_cols = get_feature_columns(df_d)
            assert len(d_cols) < len(full_cols)
            # All D columns should have D_ prefix or regime_serial_corr
            for c in d_cols:
                assert c.startswith("D_") or c == "regime_serial_corr", \
                    f"Unexpected column in ablation:D: {c}"

    def test_empty_minute_data(self):
        pair = _make_pair()
        df = compute_features_v2([pair], {}, ["2024-01-05"], mode="chan")
        assert len(df) == 0


class TestGetFeatureColumns:
    def test_excludes_pair_id_and_date(self):
        df = pd.DataFrame({
            "pair_id": ["A_B"],
            "date": ["2024-01-01"],
            "feat1": [1.0],
            "feat2": [2.0],
        })
        assert get_feature_columns(df) == ["feat1", "feat2"]


# ═══════════════════════════════════════════════════════════════════════
# TRAINING AND PREDICTION
# ═══════════════════════════════════════════════════════════════════════

class TestTrainV2:
    def test_basic_training(self):
        """Should train a model with synthetic data."""
        returns_df = _make_returns_df(n_days=50, n_configs=5)

        # Create feature DataFrame matching the returns days
        days = sorted(returns_df["date"].unique())
        np.random.seed(42)
        feat_rows = []
        for day in days:
            feat_rows.append({
                "pair_id": "A_B",
                "date": day,
                "f1": np.random.randn(),
                "f2": np.random.randn(),
                "f3": np.random.randn(),
            })
        features_df = pd.DataFrame(feat_rows)

        # Small param grid for testing
        grid = [
            ParamConfig(config_id=c, lookback_minutes=390,
                        entry_z=1.5, exit_z=0.25, stop_z=3.0)
            for c in range(5)
        ]

        model = train_v2(features_df, returns_df, "A_B", grid)
        assert model["model"] is not None
        assert 0.0 <= model["train_score"] <= 1.0
        assert model["n_features"] == 3
        assert model["n_samples"] > 0

    def test_no_data_returns_error(self):
        returns_df = _make_returns_df()
        features_df = pd.DataFrame({"pair_id": [], "date": [], "f1": []})
        model = train_v2(features_df, returns_df, "A_B")
        assert model["model"] is None


class TestPredictV2:
    def test_basic_prediction(self):
        """Should return a config, probability, and expected return."""
        returns_df = _make_returns_df(n_days=50, n_configs=5)
        days = sorted(returns_df["date"].unique())
        np.random.seed(42)
        feat_rows = [{"pair_id": "A_B", "date": d,
                      "f1": np.random.randn(), "f2": np.random.randn()}
                     for d in days]
        features_df = pd.DataFrame(feat_rows)

        grid = [ParamConfig(config_id=c, lookback_minutes=390,
                            entry_z=1.5, exit_z=0.25, stop_z=3.0)
                for c in range(5)]

        model = train_v2(features_df, returns_df, "A_B", grid)
        assert model["model"] is not None

        test_features = np.array([0.5, -0.3])
        config, prob, e_ret = predict_v2(model, test_features, grid)

        assert isinstance(config, ParamConfig)
        assert 0.0 <= prob <= 1.0
        assert np.isfinite(e_ret)

    def test_no_model_returns_default(self):
        grid = [ParamConfig(config_id=0, lookback_minutes=390,
                            entry_z=1.5, exit_z=0.25, stop_z=3.0)]
        config, prob, e_ret = predict_v2({"model": None}, np.array([1.0]), grid)
        assert prob == 0.0


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

class TestMakeConfigOnlyFeatures:
    def test_has_all_pairs_and_dates(self):
        returns_df = _make_returns_df(n_days=10)
        result = _make_config_only_features(returns_df)
        assert "pair_id" in result.columns
        assert "date" in result.columns
        assert "dummy_const" in result.columns
        assert result["pair_id"].unique().tolist() == ["A_B"]
        assert len(result) == 10  # one row per day
