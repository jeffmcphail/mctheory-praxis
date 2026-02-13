"""
Chan CPO Pipeline (Phase 3.4 / 3.5 / 3.6).

§5.2 SignalBasket: Compute signal from full basket, trade only one leg.
CPO training data: Loop date × params → Sharpe (port of run_pair_trade_gld_gdx_history).
CPO predictor: RandomForest on indicators + params → predict optimal Sharpe.

Usage:
    cpo = CPOPipeline()
    # Training: generate Sharpe landscape
    training_data = cpo.generate_training_data(pair_df, param_grid, dates)
    # Prediction: pick best params for next period
    best_params = cpo.predict_best_params(training_data, indicator_features)
    # Execution: run with predicted params
    result = cpo.execute_single_period(pair_df, best_params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from praxis.logger.core import PraxisLogger
from praxis.signals.zscore import ZScoreSpread, _ewm_mean, _ewm_std


# ═══════════════════════════════════════════════════════════════════
#  §5.2: SignalBasket — Single-Leg Execution
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SingleLegResult:
    """Result of a single-period single-leg backtest."""
    daily_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    num_trades: int
    positions: np.ndarray
    pnl: np.ndarray


def execute_single_leg(
    close_a: np.ndarray,
    open_a: np.ndarray,
    close_b: np.ndarray,
    params: dict[str, Any],
    transaction_costs: float = 0.0005,
    periods_per_year: float = 252 * 6.5 * 60,
) -> SingleLegResult:
    """
    §5.2 single-leg execution: exact port of pair_trade_gld_gdx().

    Compute z-score spread signal from (A, B) pair.
    Trade only leg A. P&L = position * (close_A - open_A) / open_A.

    Args:
        close_a: Close prices of traded asset (GLD).
        open_a: Open prices of traded asset (GLD).
        close_b: Close prices of signal asset (GDX).
        params: {weight, lookback, entry_threshold, exit_threshold_fraction}
        transaction_costs: Per-trade cost fraction.
        periods_per_year: Annualization factor.

    Returns:
        SingleLegResult with Sharpe, return, vol, etc.
    """
    weight = params["weight"]
    lookback = params["lookback"]
    entry_threshold = params["entry_threshold"]
    exit_frac = params.get("exit_threshold_fraction", -0.6)
    exit_threshold = exit_frac * entry_threshold

    n = len(close_a)

    # ── Spread z-score ────────────────────────────────────────
    spread = close_a - weight * close_b
    ema_mean = _ewm_mean(spread, lookback)
    ema_std = _ewm_std(spread, lookback)
    with np.errstate(invalid="ignore"):
        zs = np.where(ema_std > 0, (spread - ema_mean) / ema_std, 0.0)

    # ── Position signals (matching original exactly) ──────────
    pos_long = np.zeros(n)
    pos_short = np.zeros(n)

    pos_short[zs >= entry_threshold] = -1
    pos_long[zs <= -entry_threshold] = 1
    pos_short[zs <= exit_threshold] = 0
    pos_long[zs >= -exit_threshold] = 0

    # Forward-fill (pandas ffill equivalent)
    # The original code uses DataFrame.ffill() which carries forward
    # ALL columns including position columns
    for i in range(1, n):
        if pos_long[i] == 0 and zs[i] < -entry_threshold:
            pos_long[i] = 1
        elif pos_long[i] == 0 and pos_long[i - 1] == 1 and zs[i] < -exit_threshold:
            pos_long[i] = 1

        if pos_short[i] == 0 and zs[i] > entry_threshold:
            pos_short[i] = -1
        elif pos_short[i] == 0 and pos_short[i - 1] == -1 and zs[i] > exit_threshold:
            pos_short[i] = -1

    positions = pos_long + pos_short

    # ── Trade counting ────────────────────────────────────────
    pos_diff = np.diff(positions, prepend=0)
    num_long_trades = int(np.sum(np.diff(pos_long, prepend=0) == 1))
    num_short_trades = int(np.sum(np.diff(pos_short, prepend=0) == -1))
    num_trades = num_long_trades + num_short_trades

    # ── P&L: single-leg on A ─────────────────────────────────
    # Original: period_return = (close_GLD - open_GLD) / open_GLD
    with np.errstate(invalid="ignore", divide="ignore"):
        period_return = np.where(open_a != 0, (close_a - open_a) / open_a, 0.0)

    # Shifted positions × returns (original uses positions.shift())
    shifted_pos = np.zeros(n)
    shifted_pos[1:] = positions[:-1]

    pnl = shifted_pos * period_return
    tc = np.abs(pos_diff) * transaction_costs
    pnl_tc = pnl - tc

    # ── Metrics (matching original exactly) ───────────────────
    pnl_slice = pnl_tc[1:]  # Skip first bar
    daily_return = float(np.sum(pnl_slice))
    # Liquidation cost if holding at end
    if positions[-1] != 0:
        daily_return -= transaction_costs

    annualized_return = float((1 + daily_return) ** 252 - 1)

    std_pnl = float(np.std(pnl_slice)) if len(pnl_slice) > 1 else 0.0
    volatility = float(np.sqrt(periods_per_year) * std_pnl)

    if std_pnl > 0:
        sharpe_ratio = float(np.sqrt(periods_per_year) * np.mean(pnl_slice) / std_pnl)
    else:
        sharpe_ratio = 0.0

    return SingleLegResult(
        daily_return=daily_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        num_trades=num_trades,
        positions=positions,
        pnl=pnl_tc,
    )


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.5: CPO Training Data Generator
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CPOTrainingRow:
    """One row of CPO training data: date × params → performance."""
    date: str
    weight: float
    entry_threshold: float
    lookback: int
    sharpe_ratio: float
    daily_return: float
    annualized_return: float
    volatility: float
    num_trades: int


@dataclass
class CPOTrainingResult:
    """Complete training dataset."""
    rows: list[CPOTrainingRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame for ML consumption."""
        if not self.rows:
            return pl.DataFrame()
        return pl.DataFrame([
            {
                "date": r.date,
                "weight": r.weight,
                "entry_threshold": r.entry_threshold,
                "lookback": r.lookback,
                "sharpe_ratio": r.sharpe_ratio,
                "daily_return": r.daily_return,
                "annualized_return": r.annualized_return,
                "volatility": r.volatility,
                "num_trades": r.num_trades,
            }
            for r in self.rows
        ])

    @property
    def count(self) -> int:
        return len(self.rows)


def generate_training_data(
    close_a: np.ndarray,
    open_a: np.ndarray,
    close_b: np.ndarray,
    dates: np.ndarray,
    param_grid: dict[str, list],
    exit_threshold_fraction: float = -0.6,
    transaction_costs: float = 0.0005,
    periods_per_year: float = 252 * 6.5 * 60,
) -> CPOTrainingResult:
    """
    Port of run_pair_trade_gld_gdx_history().

    Loop: for each date × weight × entry_threshold × lookback:
        Run single-leg execution for that day's data window.
        Record Sharpe ratio and other metrics.

    Args:
        close_a: Full close array for asset A.
        open_a: Full open array for asset A.
        close_b: Full close array for asset B.
        dates: Array of unique dates (indices into the arrays).
        param_grid: {
            "weights": [2, 2.5, 3, ...],
            "entry_thresholds": [0.2, 0.3, ...],
            "lookbacks": [30, 60, ...]
        }
        exit_threshold_fraction: Fraction for exit threshold.

    Returns:
        CPOTrainingResult with all combinations.
    """
    log = PraxisLogger.instance()
    result = CPOTrainingResult()

    weights = param_grid.get("weights", [3.0])
    entries = param_grid.get("entry_thresholds", [1.0])
    lookbacks = param_grid.get("lookbacks", [60])

    total = len(dates) * len(weights) * len(entries) * len(lookbacks)
    log.info(
        f"CPO training: {len(dates)} dates × "
        f"{len(weights)} weights × {len(entries)} entries × "
        f"{len(lookbacks)} lookbacks = {total} combinations",
        tags={"compute.cpo"},
    )

    for i, date_idx in enumerate(dates):
        for w in weights:
            for et in entries:
                for lb in lookbacks:
                    try:
                        # Window: lookback bars before date through date
                        start = max(0, date_idx - lb)
                        end = date_idx + 1

                        if end - start < 10:  # Too little data
                            continue

                        params = {
                            "weight": w,
                            "lookback": lb,
                            "entry_threshold": et,
                            "exit_threshold_fraction": exit_threshold_fraction,
                        }

                        slr = execute_single_leg(
                            close_a[start:end],
                            open_a[start:end],
                            close_b[start:end],
                            params,
                            transaction_costs=transaction_costs,
                            periods_per_year=periods_per_year,
                        )

                        result.rows.append(CPOTrainingRow(
                            date=str(date_idx),
                            weight=w,
                            entry_threshold=et,
                            lookback=lb,
                            sharpe_ratio=slr.sharpe_ratio,
                            daily_return=slr.daily_return,
                            annualized_return=slr.annualized_return,
                            volatility=slr.volatility,
                            num_trades=slr.num_trades,
                        ))
                    except Exception as e:
                        result.errors.append(f"date={date_idx} w={w} et={et} lb={lb}: {e}")

        if (i + 1) % 10 == 0:
            log.debug(
                f"CPO training: processed {i + 1}/{len(dates)} dates, "
                f"{result.count} rows so far",
                tags={"compute.cpo"},
            )

    log.info(
        f"CPO training complete: {result.count} rows, {len(result.errors)} errors",
        tags={"compute.cpo"},
    )
    return result


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.6: CPO Predictor (RandomForest)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CPOPrediction:
    """Prediction result from CPO predictor."""
    predicted_params: dict[str, float]
    predicted_sharpe: float
    model_score: float  # In-sample R² or MSE


class CPOPredictor:
    """
    Port of generate_pair_trade_gld_gdx_predictor().

    Uses RandomForest to predict Sharpe ratio from:
    - Parameters (weight, entry_threshold, lookback)
    - Technical indicators (optional feature columns)

    Expanding window: train on all data up to date T, predict T+1.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        target_field: str = "sharpe_ratio",
    ):
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._target_field = target_field
        self._model = None
        self._feature_cols: list[str] = []
        self._log = PraxisLogger.instance()

    def fit(
        self,
        training_data: pl.DataFrame,
        feature_columns: list[str] | None = None,
        train_fraction: float = 0.8,
    ) -> dict[str, float]:
        """
        Fit RandomForest on training data.

        Args:
            training_data: DataFrame with params + indicators + target.
            feature_columns: Column names to use as features.
                Default: ["weight", "entry_threshold", "lookback"]
            train_fraction: Train/test split ratio.

        Returns:
            {"mse_train": ..., "mse_test": ..., "r2_train": ...}
        """
        from sklearn.ensemble import RandomForestRegressor

        if feature_columns:
            self._feature_cols = feature_columns
        else:
            self._feature_cols = ["weight", "entry_threshold", "lookback"]

        # Filter valid rows
        df = training_data.filter(
            pl.col(self._target_field).is_not_null()
            & pl.col(self._target_field).is_not_nan()
        )

        if len(df) < 10:
            raise ValueError(f"Too few valid training rows: {len(df)}")

        n = len(df)
        split = int(n * train_fraction)

        X_train = df[:split].select(self._feature_cols).to_numpy()
        y_train = df[:split][self._target_field].to_numpy()
        X_test = df[split:].select(self._feature_cols).to_numpy()
        y_test = df[split:][self._target_field].to_numpy()

        self._model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            random_state=self._random_state,
        )
        self._model.fit(X_train, y_train)

        # Metrics
        pred_train = self._model.predict(X_train)
        pred_test = self._model.predict(X_test)

        mse_train = float(np.mean((pred_train - y_train) ** 2))
        mse_test = float(np.mean((pred_test - y_test) ** 2)) if len(y_test) > 0 else 0.0

        ss_res = np.sum((y_train - pred_train) ** 2)
        ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        self._log.info(
            f"CPO predictor fit: {n} rows, {len(self._feature_cols)} features, "
            f"MSE_train={mse_train:.6f}, MSE_test={mse_test:.6f}, R²={r2:.4f}",
            tags={"compute.cpo"},
        )

        return {"mse_train": mse_train, "mse_test": mse_test, "r2_train": r2}

    def predict_best_params(
        self,
        candidates: pl.DataFrame,
    ) -> CPOPrediction:
        """
        Predict Sharpe for each candidate param set, return the best.

        Args:
            candidates: DataFrame with feature columns (param combinations).

        Returns:
            CPOPrediction with best predicted params and Sharpe.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = candidates.select(self._feature_cols).to_numpy()
        predictions = self._model.predict(X)

        best_idx = int(np.argmax(predictions))
        best_row = candidates[best_idx]

        params = {}
        for col in self._feature_cols:
            val = best_row[col].item()
            if isinstance(val, (list, pl.Series)):
                val = val[0] if len(val) > 0 else 0
            params[col] = float(val)

        return CPOPrediction(
            predicted_params=params,
            predicted_sharpe=float(predictions[best_idx]),
            model_score=0.0,
        )

    def predict_sharpe(self, params: dict[str, float]) -> float:
        """Predict Sharpe for a single parameter set."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.array([[params.get(c, 0.0) for c in self._feature_cols]])
        return float(self._model.predict(X)[0])

    @property
    def feature_importances(self) -> dict[str, float] | None:
        """Get feature importances from fitted model."""
        if self._model is None:
            return None
        return dict(zip(self._feature_cols, self._model.feature_importances_))
