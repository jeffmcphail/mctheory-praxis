"""
Model executors (§7.3).

Executors orchestrate the config → signal → sizing → backtest pipeline.
Each model type routes to a specific executor:

    SingleAssetModel → SimpleExecutor
    CompositeModel   → CompositeExecutor (Phase 2)
    CPOModel         → CPOExecutor (Phase 3)
    MLModel          → MLExecutor (Phase 3)

Phase 1.6 implements SimpleExecutor only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from praxis.config import ModelConfig, ModelType
from praxis.backtest import VectorizedEngine
from praxis.logger.core import PraxisLogger
from praxis.registry import FunctionRegistry
from praxis.registry.defaults import register_defaults


@dataclass
class ExecutionResult:
    """
    Result of an executor run.

    Contains the raw signals, positions, and any metrics computed
    by the backtest engine (Phase 1.7 will populate metrics).
    """
    config: ModelConfig
    signals: Optional[pl.Series] = None
    positions: Optional[pl.Series] = None
    prices: Optional[pl.DataFrame] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class Executor(ABC):
    """Base executor interface."""

    @abstractmethod
    def execute(self, config: ModelConfig, prices: pl.DataFrame) -> ExecutionResult:
        """
        Execute the model pipeline.

        Args:
            config: Validated ModelConfig.
            prices: Price DataFrame with at minimum 'close' column.

        Returns:
            ExecutionResult with signals, positions, and metrics.
        """
        ...


class SimpleExecutor(Executor):
    """
    §7.3: SingleAssetModel executor.

    Pipeline:
    1. Resolve signal function from registry
    2. Generate signals from price data
    3. Resolve sizing function from registry
    4. Compute position sizes
    5. (Phase 1.7: Run backtest engine)
    """

    def __init__(self, registry: FunctionRegistry | None = None):
        self._registry = registry or FunctionRegistry.instance()
        self._log = PraxisLogger.instance()

    def execute(self, config: ModelConfig, prices: pl.DataFrame) -> ExecutionResult:
        self._log.info(
            f"SimpleExecutor: running '{config.model.name}'",
            tags={"executor", "trade_cycle"},
            model_name=config.model.name,
            model_type=config.model.type.value,
            bars=len(prices),
        )

        try:
            # ── 1. Resolve signal function ────────────────────────
            signal_method = config.signal.method
            signal_fn = self._registry.resolve("signals", signal_method)

            self._log.debug(
                f"Resolved signal: {signal_method} → {type(signal_fn).__name__}",
                tags={"executor", "trade_cycle"},
            )

            # ── 2. Generate signals ───────────────────────────────
            signal_params = config.signal.model_dump(exclude_none=True)
            signals = signal_fn.generate(prices, signal_params)

            self._log.info(
                f"Signals generated: {(signals == 1).sum()} long, "
                f"{(signals == -1).sum()} short, {(signals == 0).sum()} flat",
                tags={"executor", "trade_cycle"},
            )

            # ── 3. Resolve sizing function ────────────────────────
            if config.sizing:
                sizing_method = config.sizing.method
                sizer_fn = self._registry.resolve("sizing", sizing_method)
                sizing_params = config.sizing.model_dump(exclude_none=True)
            else:
                # Default: fixed_fraction at 100%
                sizer_fn = self._registry.resolve("sizing", "fixed_fraction")
                sizing_params = {"fraction": 1.0}

            self._log.debug(
                f"Resolved sizer: {sizing_params.get('method', 'fixed_fraction')} "
                f"→ {type(sizer_fn).__name__}",
                tags={"executor", "trade_cycle"},
            )

            # ── 4. Compute positions ──────────────────────────────
            positions = sizer_fn.size(signals, sizing_params)

            self._log.info(
                f"Positions computed: "
                f"{(positions > 0).sum()} long, "
                f"{(positions < 0).sum()} short, "
                f"{(positions == 0).sum()} flat",
                tags={"executor", "trade_cycle"},
            )

            # ── 5. Run backtest engine ─────────────────────────────
            engine = VectorizedEngine()
            bt_output = engine.run(positions, prices["close"])

            return ExecutionResult(
                config=config,
                signals=signals,
                positions=positions,
                prices=prices,
                metrics=bt_output.metrics.to_dict(),
                success=True,
            )

        except Exception as e:
            self._log.error(
                f"SimpleExecutor failed: {e}",
                tags={"executor", "trade_cycle"},
                error=str(e),
            )
            return ExecutionResult(
                config=config,
                success=False,
                error=str(e),
            )


class CPOExecutor(Executor):
    """
    §7.3: CPOModel executor (Phase 3.7).

    Pipeline:
    1. Generate training data: sweep params over historical windows
    2. Fit CPO predictor (RandomForest)
    3. Predict best params for current period
    4. Execute single-leg with predicted params
    5. Backtest and return metrics
    """

    def __init__(self, registry: FunctionRegistry | None = None):
        self._registry = registry or FunctionRegistry.instance()
        self._log = PraxisLogger.instance()

    def execute(self, config: ModelConfig, prices: pl.DataFrame) -> ExecutionResult:
        self._log.info(
            f"CPOExecutor: running '{config.model.name}'",
            tags={"executor", "trade_cycle", "compute.cpo"},
        )

        try:
            from praxis.cpo import (
                CPOPredictor, execute_single_leg, generate_training_data,
            )

            cpo_cfg = config.cpo
            if cpo_cfg is None:
                return ExecutionResult(
                    config=config, success=False,
                    error="CPOModel requires 'cpo' config section",
                )

            # Extract arrays
            close_a = prices[config.cpo.features.get("close_a", "close")].to_numpy().astype(float)
            open_a = prices[config.cpo.features.get("open_a", "open")].to_numpy().astype(float) if "open" in prices.columns else close_a.copy()
            close_b_col = config.cpo.features.get("close_b", None)
            if close_b_col and close_b_col in prices.columns:
                close_b = prices[close_b_col].to_numpy().astype(float)
            else:
                close_b = np.zeros_like(close_a)

            # Param grid from config
            param_grid = {}
            if cpo_cfg.parameter_grid:
                for pg in cpo_cfg.parameter_grid:
                    param_grid.update(pg)

            # Generate training data over first portion
            n = len(close_a)
            train_end = int(n * 0.8)
            dates = np.arange(100, train_end, 50)

            training = generate_training_data(
                close_a[:train_end], open_a[:train_end], close_b[:train_end],
                dates, param_grid,
            )

            if training.count < 10:
                return ExecutionResult(
                    config=config, success=False,
                    error=f"Insufficient training data: {training.count} rows",
                )

            # Fit predictor
            df = training.to_polars()
            predictor = CPOPredictor(
                n_estimators=cpo_cfg.model.get("n_estimators", 100) if cpo_cfg.model else 100,
                random_state=42,
            )
            metrics = predictor.fit(df)

            # Predict best params
            candidates = df.select(["weight", "entry_threshold", "lookback"]).unique()
            prediction = predictor.predict_best_params(candidates)

            # Execute on test portion
            best = prediction.predicted_params
            best["exit_threshold_fraction"] = -0.6
            slr = execute_single_leg(
                close_a[train_end:], open_a[train_end:], close_b[train_end:],
                {k: (int(v) if k == "lookback" else v) for k, v in best.items()},
            )

            return ExecutionResult(
                config=config,
                signals=pl.Series("signal", slr.positions.astype(int)),
                positions=pl.Series("positions", slr.positions),
                prices=prices,
                metrics={
                    "sharpe_ratio": slr.sharpe_ratio,
                    "daily_return": slr.daily_return,
                    "annualized_return": slr.annualized_return,
                    "volatility": slr.volatility,
                    "num_trades": slr.num_trades,
                    "predicted_sharpe": prediction.predicted_sharpe,
                    "predicted_params": prediction.predicted_params,
                    "training_rows": training.count,
                    "mse_test": metrics.get("mse_test", 0),
                },
                success=True,
            )

        except Exception as e:
            self._log.error(
                f"CPOExecutor failed: {e}",
                tags={"executor", "trade_cycle", "compute.cpo"},
            )
            return ExecutionResult(config=config, success=False, error=str(e))


# ═══════════════════════════════════════════════════════════════════
#  Executor Dispatch (§7.3 step 3)
# ═══════════════════════════════════════════════════════════════════

_EXECUTOR_MAP: dict[ModelType, type[Executor]] = {
    ModelType.SINGLE_ASSET: SimpleExecutor,
    ModelType.CPO: CPOExecutor,
    # Phase 3+:
    # ModelType.PAIR: PairExecutor,
    # ModelType.COMPOSITE: CompositeExecutor,
    # ModelType.ML: MLExecutor,
}


def get_executor(config: ModelConfig, registry: FunctionRegistry | None = None) -> Executor:
    """
    §7.3: Route model_type to the correct executor.

    Raises ValueError for unsupported model types.
    """
    model_type = config.model.type
    executor_cls = _EXECUTOR_MAP.get(model_type)

    if executor_cls is None:
        supported = ", ".join(t.value for t in _EXECUTOR_MAP)
        raise ValueError(
            f"No executor for model type '{model_type.value}'. "
            f"Supported: {supported}"
        )

    return executor_cls(registry=registry)
