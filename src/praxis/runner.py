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
from praxis.data import fetch_prices, generate_synthetic_prices, PriceData
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

    def run_with_fetch(
        self,
        config: ModelConfig,
        tickers: str | list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> ExecutionResult:
        """
        Run with automatic data fetching.

        Args:
            config: Validated ModelConfig.
            tickers: Ticker(s) to fetch.
            start: Start date (ISO string).
            end: End date (ISO string).

        Returns:
            ExecutionResult with signals, positions, metrics.
        """
        prices = fetch_prices(tickers, start=start, end=end)
        return self.run_config(config, prices)


def run_cli(args: list[str] | None = None) -> int:
    """
    CLI entry point for `praxis run <config.yaml>`.

    Usage:
        praxis run config.yaml --prices data.csv
        praxis run config.yaml --ticker AAPL --start 2023-01-01
        praxis run config.yaml --synthetic 252

    Returns exit code (0 = success, 1 = error).
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: praxis run <config.yaml> [--prices <file.csv>] "
              "[--ticker <SYM>] [--synthetic <bars>]")
        return 1

    yaml_path = Path(args[0])
    if not yaml_path.exists():
        print(f"Error: Config file not found: {yaml_path}")
        return 1

    config = ModelConfig.from_yaml(yaml_path)

    # Determine data source
    prices = None

    if "--prices" in args:
        idx = args.index("--prices")
        if idx + 1 < len(args):
            prices_path = Path(args[idx + 1])
            if prices_path.exists():
                prices = pl.read_csv(str(prices_path))
            else:
                print(f"Error: Prices file not found: {prices_path}")
                return 1

    elif "--ticker" in args:
        idx = args.index("--ticker")
        if idx + 1 < len(args):
            ticker = args[idx + 1]
            start = None
            end = None
            if "--start" in args:
                start = args[args.index("--start") + 1]
            if "--end" in args:
                end = args[args.index("--end") + 1]
            prices = fetch_prices(ticker, start=start, end=end)

    elif "--synthetic" in args:
        idx = args.index("--synthetic")
        n_bars = 252
        if idx + 1 < len(args):
            try:
                n_bars = int(args[idx + 1])
            except ValueError:
                pass
        prices = generate_synthetic_prices(n_bars=n_bars, seed=42)

    if prices is None:
        print("Error: Data source required. Use --prices, --ticker, or --synthetic")
        return 1

    runner = PraxisRunner()
    result = runner.run_config(config, prices)

    if result.success:
        print(f"✓ Model '{result.config.model.name}' completed")
        if result.metrics:
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        return 0
    else:
        print(f"✗ Model '{result.config.model.name}' failed: {result.error}")
        return 1
