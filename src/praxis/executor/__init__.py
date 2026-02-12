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

import polars as pl

from praxis.config import ModelConfig, ModelType
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

            # ── 5. Backtest engine (Phase 1.7) ────────────────────
            # Placeholder: metrics populated by backtest engine later
            return ExecutionResult(
                config=config,
                signals=signals,
                positions=positions,
                prices=prices,
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


# ═══════════════════════════════════════════════════════════════════
#  Executor Dispatch (§7.3 step 3)
# ═══════════════════════════════════════════════════════════════════

_EXECUTOR_MAP: dict[ModelType, type[Executor]] = {
    ModelType.SINGLE_ASSET: SimpleExecutor,
    # Phase 2+:
    # ModelType.PAIR: PairExecutor,
    # ModelType.COMPOSITE: CompositeExecutor,
    # ModelType.CPO: CPOExecutor,
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
