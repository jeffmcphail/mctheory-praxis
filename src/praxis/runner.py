"""
Praxis CLI runner (§7.3).

Entry point: `praxis run config.yaml`

Pipeline:
1. Parse YAML → dict
2. Validate against schema (Pydantic ModelConfig)
3. Initialize infrastructure (logger, database, registry)
4. Determine model_type → route to executor
5. Execute pipeline

Phase 1.6: CLI skeleton with SingleAssetModel routing.
Phase 1.7: Backtest engine integration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import polars as pl

from praxis.config import ModelConfig, PlatformMode
from praxis.executor import ExecutionResult, get_executor
from praxis.logger.core import PraxisLogger
from praxis.registry import FunctionRegistry
from praxis.registry.defaults import register_defaults


class PraxisRunner:
    """
    Orchestrates the full `praxis run` pipeline.

    Usage:
        runner = PraxisRunner()
        result = runner.run_yaml("sma_crossover.yaml", prices=df)
        result = runner.run_config(config, prices=df)
    """

    def __init__(
        self,
        registry: FunctionRegistry | None = None,
    ):
        self._log = PraxisLogger.instance()
        self._registry = registry or FunctionRegistry.instance()

        # Ensure defaults are registered
        register_defaults(self._registry)

    def run_yaml(
        self,
        yaml_path: str | Path,
        prices: pl.DataFrame,
    ) -> ExecutionResult:
        """
        §7.3 steps 1-5: Full pipeline from YAML file.

        Args:
            yaml_path: Path to model config YAML.
            prices: Price DataFrame.

        Returns:
            ExecutionResult with signals, positions, metrics.
        """
        self._log.info(
            f"Loading config: {yaml_path}",
            tags={"runner", "trade_cycle"},
        )

        # Step 1-2: Parse + validate
        config = ModelConfig.from_yaml(yaml_path)
        return self.run_config(config, prices)

    def run_config(
        self,
        config: ModelConfig,
        prices: pl.DataFrame,
    ) -> ExecutionResult:
        """
        Run from a pre-validated ModelConfig.

        Args:
            config: Validated ModelConfig.
            prices: Price DataFrame.

        Returns:
            ExecutionResult with signals, positions, metrics.
        """
        self._log.info(
            f"Running model: {config.model.name} ({config.model.type.value})",
            tags={"runner", "trade_cycle"},
            model_name=config.model.name,
            model_type=config.model.type.value,
            version=config.model.version,
            config_hash=config.config_hash[:12],
        )

        # Step 3: model_type → executor
        executor = get_executor(config, self._registry)

        self._log.debug(
            f"Executor: {type(executor).__name__}",
            tags={"runner", "trade_cycle"},
        )

        # Step 4-5: Execute
        result = executor.execute(config, prices)

        if result.success:
            self._log.info(
                f"Model '{config.model.name}' completed successfully",
                tags={"runner", "trade_cycle"},
            )
        else:
            self._log.error(
                f"Model '{config.model.name}' failed: {result.error}",
                tags={"runner", "trade_cycle"},
            )

        return result


def run_cli(args: list[str] | None = None) -> int:
    """
    CLI entry point for `praxis run <config.yaml>`.

    Returns exit code (0 = success, 1 = error).
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: praxis run <config.yaml> [--prices <prices.csv>]")
        return 1

    yaml_path = Path(args[0])
    if not yaml_path.exists():
        print(f"Error: Config file not found: {yaml_path}")
        return 1

    # Price data source
    prices_path = None
    if "--prices" in args:
        idx = args.index("--prices")
        if idx + 1 < len(args):
            prices_path = Path(args[idx + 1])

    if prices_path and prices_path.exists():
        prices = pl.read_csv(prices_path)
    else:
        print("Error: Price data required. Use --prices <file.csv>")
        return 1

    runner = PraxisRunner()
    result = runner.run_config(
        ModelConfig.from_yaml(yaml_path),
        prices,
    )

    if result.success:
        print(f"✓ Model '{result.config.model.name}' completed")
        if result.metrics:
            for k, v in result.metrics.items():
                print(f"  {k}: {v}")
        return 0
    else:
        print(f"✗ Model '{result.config.model.name}' failed: {result.error}")
        return 1
