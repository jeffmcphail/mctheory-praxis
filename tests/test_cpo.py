"""
Tests for CPO Pipeline (Phase 3.4 / 3.5 / 3.6).

3.4: Single-leg execution (SignalBasket)
3.5: CPO training data generator
3.6: CPO predictor (RandomForest)
"""

import numpy as np
import polars as pl
import pytest

from praxis.cpo import (
    CPOPredictor,
    CPOPrediction,
    CPOTrainingResult,
    SingleLegResult,
    execute_single_leg,
    generate_training_data,
)
from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ── Synthetic data ────────────────────────────────────────────

def _make_pair_arrays(n: int = 500, seed: int = 42):
    """Generate synthetic GLD/GDX-like OHLCV arrays."""
    rng = np.random.RandomState(seed)
    close_a = 180.0 + np.cumsum(rng.randn(n) * 0.3)
    open_a = close_a + rng.randn(n) * 0.1
    close_b = 30.0 + np.cumsum(rng.randn(n) * 0.15)
    return close_a, open_a, close_b


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.4: Single-Leg Execution
# ═══════════════════════════════════════════════════════════════════

class TestSingleLegBasic:
    def test_returns_result(self):
        close_a, open_a, close_b = _make_pair_arrays(200)
        params = {"weight": 3.0, "lookback": 60, "entry_threshold": 1.0,
                  "exit_threshold_fraction": -0.6}
        result = execute_single_leg(close_a, open_a, close_b, params)
        assert isinstance(result, SingleLegResult)

    def test_sharpe_is_finite(self):
        close_a, open_a, close_b = _make_pair_arrays(500)
        params = {"weight": 3.0, "lookback": 60, "entry_threshold": 1.0,
                  "exit_threshold_fraction": -0.6}
        result = execute_single_leg(close_a, open_a, close_b, params)
        assert np.isfinite(result.sharpe_ratio)

    def test_positions_correct_length(self):
        close_a, open_a, close_b = _make_pair_arrays(200)
        params = {"weight": 3.0, "lookback": 60, "entry_threshold": 1.0,
                  "exit_threshold_fraction": -0.6}
        result = execute_single_leg(close_a, open_a, close_b, params)
        assert len(result.positions) == 200
        assert len(result.pnl) == 200

    def test_positions_in_range(self):
        close_a, open_a, close_b = _make_pair_arrays(300)
        params = {"weight": 3.0, "lookback": 60, "entry_threshold": 1.0,
                  "exit_threshold_fraction": -0.6}
        result = execute_single_leg(close_a, open_a, close_b, params)
        unique = set(result.positions.astype(int))
        assert unique.issubset({-1, 0, 1})


class TestSingleLegParams:
    def test_higher_threshold_fewer_trades(self):
        close_a, open_a, close_b = _make_pair_arrays(500)
        r_low = execute_single_leg(close_a, open_a, close_b,
                                    {"weight": 3.0, "lookback": 60,
                                     "entry_threshold": 0.5, "exit_threshold_fraction": -0.6})
        r_high = execute_single_leg(close_a, open_a, close_b,
                                     {"weight": 3.0, "lookback": 60,
                                      "entry_threshold": 2.0, "exit_threshold_fraction": -0.6})
        assert r_low.num_trades >= r_high.num_trades

    def test_different_weights_different_results(self):
        close_a, open_a, close_b = _make_pair_arrays(500)
        r1 = execute_single_leg(close_a, open_a, close_b,
                                 {"weight": 2.0, "lookback": 60,
                                  "entry_threshold": 1.0, "exit_threshold_fraction": -0.6})
        r2 = execute_single_leg(close_a, open_a, close_b,
                                 {"weight": 5.0, "lookback": 60,
                                  "entry_threshold": 1.0, "exit_threshold_fraction": -0.6})
        # Different weights should produce different Sharpes
        assert r1.sharpe_ratio != r2.sharpe_ratio

    def test_transaction_costs_reduce_return(self):
        close_a, open_a, close_b = _make_pair_arrays(500)
        params = {"weight": 3.0, "lookback": 60, "entry_threshold": 0.5,
                  "exit_threshold_fraction": -0.6}
        r_no_tc = execute_single_leg(close_a, open_a, close_b, params,
                                      transaction_costs=0.0)
        r_with_tc = execute_single_leg(close_a, open_a, close_b, params,
                                        transaction_costs=0.01)
        assert r_no_tc.daily_return >= r_with_tc.daily_return


class TestSingleLegEdge:
    def test_very_high_threshold_no_trades(self):
        close_a, open_a, close_b = _make_pair_arrays(200)
        params = {"weight": 3.0, "lookback": 60, "entry_threshold": 100.0,
                  "exit_threshold_fraction": -0.6}
        result = execute_single_leg(close_a, open_a, close_b, params)
        assert result.num_trades == 0
        assert np.all(result.positions == 0)

    def test_short_data(self):
        close_a, open_a, close_b = _make_pair_arrays(20)
        params = {"weight": 3.0, "lookback": 10, "entry_threshold": 1.0,
                  "exit_threshold_fraction": -0.6}
        result = execute_single_leg(close_a, open_a, close_b, params)
        assert isinstance(result, SingleLegResult)


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.5: CPO Training Data Generator
# ═══════════════════════════════════════════════════════════════════

class TestTrainingDataBasic:
    def test_generates_rows(self):
        close_a, open_a, close_b = _make_pair_arrays(500)
        dates = np.array([100, 200, 300, 400])
        param_grid = {
            "weights": [3.0, 4.0],
            "entry_thresholds": [0.5, 1.0],
            "lookbacks": [60],
        }
        result = generate_training_data(
            close_a, open_a, close_b, dates, param_grid
        )
        assert isinstance(result, CPOTrainingResult)
        # 4 dates × 2 weights × 2 entries × 1 lookback = 16
        assert result.count == 16

    def test_to_polars(self):
        close_a, open_a, close_b = _make_pair_arrays(300)
        dates = np.array([100, 200])
        param_grid = {
            "weights": [3.0],
            "entry_thresholds": [1.0],
            "lookbacks": [60],
        }
        result = generate_training_data(
            close_a, open_a, close_b, dates, param_grid
        )
        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "sharpe_ratio" in df.columns
        assert "weight" in df.columns
        assert len(df) == result.count

    def test_different_params_different_sharpes(self):
        close_a, open_a, close_b = _make_pair_arrays(500)
        dates = np.array([300])
        param_grid = {
            "weights": [2.0, 5.0],
            "entry_thresholds": [0.3, 2.0],
            "lookbacks": [60],
        }
        result = generate_training_data(
            close_a, open_a, close_b, dates, param_grid
        )
        df = result.to_polars()
        # Different params should not all produce same Sharpe
        unique_sharpes = df["sharpe_ratio"].n_unique()
        assert unique_sharpes > 1


class TestTrainingDataEdge:
    def test_empty_dates(self):
        close_a, open_a, close_b = _make_pair_arrays(100)
        result = generate_training_data(
            close_a, open_a, close_b, np.array([]), {"weights": [3.0]}
        )
        assert result.count == 0

    def test_skips_short_windows(self):
        """Dates where lookback exceeds available data should be skipped."""
        close_a, open_a, close_b = _make_pair_arrays(50)
        dates = np.array([5])  # Only 5 bars before
        param_grid = {
            "weights": [3.0],
            "entry_thresholds": [1.0],
            "lookbacks": [100],  # Needs 100 bars
        }
        result = generate_training_data(
            close_a, open_a, close_b, dates, param_grid
        )
        # Should still produce a result (start=0 to date=5, only 6 bars)
        # but execute_single_leg handles short data gracefully
        assert isinstance(result, CPOTrainingResult)


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.6: CPO Predictor
# ═══════════════════════════════════════════════════════════════════

class TestCPOPredictorFit:
    def _make_training_df(self, n: int = 200):
        rng = np.random.RandomState(42)
        return pl.DataFrame({
            "weight": rng.choice([2.0, 3.0, 4.0, 5.0], n),
            "entry_threshold": rng.choice([0.5, 1.0, 1.5, 2.0], n),
            "lookback": rng.choice([30, 60, 90, 120], n).astype(float),
            "sharpe_ratio": rng.randn(n) * 0.5 + 0.3,
        })

    def test_fit_returns_metrics(self):
        df = self._make_training_df()
        pred = CPOPredictor(n_estimators=10, random_state=42)
        metrics = pred.fit(df)
        assert "mse_train" in metrics
        assert "mse_test" in metrics
        assert "r2_train" in metrics

    def test_fit_mse_positive(self):
        df = self._make_training_df()
        pred = CPOPredictor(n_estimators=10, random_state=42)
        metrics = pred.fit(df)
        assert metrics["mse_train"] >= 0
        assert metrics["mse_test"] >= 0

    def test_fit_too_few_rows_raises(self):
        df = pl.DataFrame({
            "weight": [3.0],
            "entry_threshold": [1.0],
            "lookback": [60.0],
            "sharpe_ratio": [0.5],
        })
        pred = CPOPredictor(n_estimators=10)
        with pytest.raises(ValueError, match="Too few"):
            pred.fit(df)


class TestCPOPredictorPredict:
    def _fitted_predictor(self):
        rng = np.random.RandomState(42)
        n = 200
        df = pl.DataFrame({
            "weight": rng.choice([2.0, 3.0, 4.0, 5.0], n),
            "entry_threshold": rng.choice([0.5, 1.0, 1.5, 2.0], n),
            "lookback": rng.choice([30, 60, 90, 120], n).astype(float),
            "sharpe_ratio": rng.randn(n) * 0.5 + 0.3,
        })
        pred = CPOPredictor(n_estimators=50, random_state=42)
        pred.fit(df)
        return pred

    def test_predict_best_params(self):
        pred = self._fitted_predictor()
        candidates = pl.DataFrame({
            "weight": [2.0, 3.0, 4.0, 5.0],
            "entry_threshold": [0.5, 1.0, 1.5, 2.0],
            "lookback": [30.0, 60.0, 90.0, 120.0],
        })
        result = pred.predict_best_params(candidates)
        assert isinstance(result, CPOPrediction)
        assert "weight" in result.predicted_params
        assert np.isfinite(result.predicted_sharpe)

    def test_predict_sharpe_single(self):
        pred = self._fitted_predictor()
        sharpe = pred.predict_sharpe({"weight": 3.0, "entry_threshold": 1.0, "lookback": 60.0})
        assert np.isfinite(sharpe)

    def test_predict_without_fit_raises(self):
        pred = CPOPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            pred.predict_sharpe({"weight": 3.0, "entry_threshold": 1.0, "lookback": 60.0})

    def test_feature_importances(self):
        pred = self._fitted_predictor()
        imp = pred.feature_importances
        assert imp is not None
        assert len(imp) == 3
        assert sum(imp.values()) == pytest.approx(1.0, abs=1e-6)


class TestCPOPredictorCustomFeatures:
    def test_fit_with_indicator_features(self):
        """CPO should work with indicator columns as features."""
        rng = np.random.RandomState(42)
        n = 200
        df = pl.DataFrame({
            "weight": rng.choice([2.0, 3.0, 4.0, 5.0], n),
            "entry_threshold": rng.choice([0.5, 1.0, 1.5, 2.0], n),
            "lookback": rng.choice([30, 60, 90, 120], n).astype(float),
            "zscore_a_50": rng.randn(n),
            "mfi_a_14": rng.rand(n) * 100,
            "atr_a_14": rng.rand(n) * 2,
            "sharpe_ratio": rng.randn(n) * 0.5 + 0.3,
        })

        pred = CPOPredictor(n_estimators=10, random_state=42)
        metrics = pred.fit(df, feature_columns=[
            "weight", "entry_threshold", "lookback",
            "zscore_a_50", "mfi_a_14", "atr_a_14",
        ])

        assert metrics["r2_train"] > 0  # Should explain some variance
        imp = pred.feature_importances
        assert "zscore_a_50" in imp


# ═══════════════════════════════════════════════════════════════════
#  Integration: Full CPO Pipeline
# ═══════════════════════════════════════════════════════════════════

class TestCPOFullPipeline:
    def test_generate_then_predict(self):
        """
        End-to-end: generate training data → fit predictor → predict best params.
        """
        close_a, open_a, close_b = _make_pair_arrays(500)

        # Step 1: Generate training data
        dates = np.arange(100, 400, 50)  # 6 dates
        param_grid = {
            "weights": [2.5, 3.0, 4.0],
            "entry_thresholds": [0.5, 1.0],
            "lookbacks": [30, 60],
        }
        training = generate_training_data(
            close_a, open_a, close_b, dates, param_grid
        )
        assert training.count > 0

        # Step 2: Fit predictor
        df = training.to_polars()
        pred = CPOPredictor(n_estimators=50, random_state=42)
        metrics = pred.fit(df)
        assert metrics["mse_train"] >= 0

        # Step 3: Predict best params
        candidates = pl.DataFrame({
            "weight": [2.5, 3.0, 3.5, 4.0, 4.5],
            "entry_threshold": [0.3, 0.5, 0.7, 1.0, 1.5],
            "lookback": [30.0, 60.0, 90.0, 120.0, 180.0],
        })
        prediction = pred.predict_best_params(candidates)

        assert isinstance(prediction, CPOPrediction)
        assert "weight" in prediction.predicted_params
        assert np.isfinite(prediction.predicted_sharpe)

        # Step 4: Execute with predicted params
        best = prediction.predicted_params
        best["exit_threshold_fraction"] = -0.6
        result = execute_single_leg(
            close_a[400:],
            open_a[400:],
            close_b[400:],
            {k: (int(v) if k == "lookback" else v) for k, v in best.items()},
        )
        assert isinstance(result, SingleLegResult)
        assert np.isfinite(result.sharpe_ratio)
