"""
End-to-End Flash Loan Atomic Arbitrage Pipeline (Phase 5.13).

Integrates all Phase 5 components into a runnable pipeline:
  signal detection → opportunity scanning → provider selection
  → gas check → simulation → MEV protection → execution → tracking

This is the `praxis run flash_loan_atomic_arb.yaml` implementation.

Usage:
    pipeline = FlashArbPipeline(config)
    pipeline.run_once()          # Single scan + execute cycle
    pipeline.run_loop(max_ticks=100)  # Continuous operation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from praxis.logger.core import PraxisLogger
from praxis.onchain import (
    OnChainAtomicExecutor,
    FlashLoanAtomicExecutor,
    OnChainConfig,
    ArbitrageParams,
    FlashLoanParams,
    SimulationResult,
    ExecutionResult,
    ExecutionStatus,
)
from praxis.onchain.connectors import (
    GasOracle,
    GasOracleConfig,
    GasEstimate,
    MEVProtector,
    MEVConfig,
    scan_opportunities,
    ArbitrageOpportunity,
    DEXQuote,
)
from praxis.onchain.crypto import (
    CryptoCointegrationAnalyzer,
    CryptoCointegrationConfig,
    CryptoSignal,
    FlashLoanOrchestrator,
    OrchestratorConfig,
)
from praxis.onchain.flash_providers import (
    FlashLoanProviderSelector,
)


# ═══════════════════════════════════════════════════════════════════
#  Pipeline Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FlashArbPipelineConfig:
    """Full pipeline configuration — maps to flash_loan_atomic_arb.yaml."""
    # Chain
    chain: str = "arbitrum"
    network: str = "mainnet"
    rpc_url: str = ""

    # Flash loan
    provider: str = "balancer"
    fallback_provider: str = "aave_v3"
    borrow_token: str = "USDC"
    borrow_amount: float = 1_000_000.0

    # Signal
    zscore_window: int = 30
    z_score_entry: float = 2.0
    z_score_exit: float = 0.5
    min_correlation: float = 0.7

    # Execution
    profit_threshold_bps: int = 10
    max_gas_price_gwei: float = 1.0     # Arbitrum
    gas_limit: int = 800_000
    slippage_tolerance_bps: int = 50
    simulation_before_execution: bool = True

    # Risk / Budget
    min_profit_usd: float = 50.0
    max_gas_spend_per_day_usd: float = 100.0
    max_failed_attempts_per_hour: int = 500
    cooldown_success_seconds: float = 30.0
    cooldown_failure_seconds: float = 5.0

    # MEV
    use_flashbots: bool = True

    # Timing
    tick_interval_seconds: float = 10.0


# ═══════════════════════════════════════════════════════════════════
#  Pipeline State
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PipelineTickResult:
    """Result of a single pipeline tick."""
    tick_number: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    opportunities_found: int = 0
    signals_generated: int = 0
    simulations_run: int = 0
    executions_attempted: int = 0
    executions_succeeded: int = 0
    profit_usd: float = 0.0
    gas_cost_usd: float = 0.0
    skipped_reason: str = ""
    details: list[str] = field(default_factory=list)

    @property
    def net_profit(self) -> float:
        return self.profit_usd - self.gas_cost_usd


@dataclass
class PipelineState:
    """Cumulative pipeline state."""
    total_ticks: int = 0
    total_opportunities: int = 0
    total_signals: int = 0
    total_executions: int = 0
    total_successes: int = 0
    total_profit_usd: float = 0.0
    total_gas_usd: float = 0.0
    is_running: bool = False
    start_time: datetime | None = None

    @property
    def net_profit(self) -> float:
        return self.total_profit_usd - self.total_gas_usd

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_successes / self.total_executions


# ═══════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════

class FlashArbPipeline:
    """
    Full flash loan atomic arbitrage pipeline (§14.6, Phase 5.13).

    Orchestrates:
    1. CryptoCointegrationAnalyzer — signal detection
    2. Opportunity scanner — cross-venue price comparison
    3. FlashLoanProviderSelector — optimal provider
    4. GasOracle — profitability gate
    5. FlashLoanAtomicExecutor — simulation + execution
    6. MEVProtector — private mempool submission
    7. FlashLoanOrchestrator — state tracking + budget management
    """

    def __init__(self, config: FlashArbPipelineConfig | None = None):
        self._config = config or FlashArbPipelineConfig()
        self._log = PraxisLogger.instance()
        self._state = PipelineState()

        # Initialize components
        self._analyzer = CryptoCointegrationAnalyzer(CryptoCointegrationConfig(
            zscore_window=self._config.zscore_window,
            z_score_entry=self._config.z_score_entry,
            z_score_exit=self._config.z_score_exit,
            min_correlation=self._config.min_correlation,
        ))

        self._gas = GasOracle(GasOracleConfig(
            max_gas_price_gwei=self._config.max_gas_price_gwei,
            default_gas_limit=self._config.gas_limit,
        ))

        self._executor = FlashLoanAtomicExecutor(OnChainConfig(
            chain=self._config.chain,
            network=self._config.network,
            rpc_url=self._config.rpc_url,
            profit_threshold_bps=self._config.profit_threshold_bps,
            max_gas_price_gwei=self._config.max_gas_price_gwei,
            gas_limit=self._config.gas_limit,
            simulation_before_execution=self._config.simulation_before_execution,
        ))

        self._provider_selector = FlashLoanProviderSelector()

        self._orchestrator = FlashLoanOrchestrator(OrchestratorConfig(
            min_profit_usd=self._config.min_profit_usd,
            max_gas_spend_per_day_usd=self._config.max_gas_spend_per_day_usd,
            max_failed_attempts_per_hour=self._config.max_failed_attempts_per_hour,
            borrow_amount_usd=self._config.borrow_amount,
        ))

        self._mev = MEVProtector(MEVConfig(
            use_private_mempool=self._config.use_flashbots,
        ))

        # Data feeds (set externally)
        self._price_feeds: dict[str, np.ndarray] = {}
        self._dex_quotes: dict[str, DEXQuote] = {}

    @property
    def config(self) -> FlashArbPipelineConfig:
        return self._config

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def analyzer(self) -> CryptoCointegrationAnalyzer:
        return self._analyzer

    @property
    def executor(self) -> FlashLoanAtomicExecutor:
        return self._executor

    @property
    def orchestrator(self) -> FlashLoanOrchestrator:
        return self._orchestrator

    @property
    def gas_oracle(self) -> GasOracle:
        return self._gas

    @property
    def mev_protector(self) -> MEVProtector:
        return self._mev

    @property
    def provider_selector(self) -> FlashLoanProviderSelector:
        return self._provider_selector

    # ── Data Feeds ────────────────────────────────────────────

    def set_price_feeds(self, feeds: dict[str, np.ndarray]) -> None:
        """Set historical price feeds for cointegration analysis."""
        self._price_feeds = feeds

    def set_dex_quotes(self, quotes: dict[str, DEXQuote]) -> None:
        """Set current DEX quotes for opportunity scanning."""
        self._dex_quotes = quotes

    # ── Pipeline Steps ────────────────────────────────────────

    def step_detect_signals(self) -> list[CryptoSignal]:
        """Step 1: Detect cointegration signals from price feeds."""
        signals = []
        assets = list(self._price_feeds.keys())

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                signal = self._analyzer.compute_signal(
                    self._price_feeds[assets[i]],
                    self._price_feeds[assets[j]],
                    asset_a=assets[i],
                    asset_b=assets[j],
                )
                if signal.should_trade:
                    signals.append(signal)

        return signals

    def step_scan_opportunities(self) -> list[ArbitrageOpportunity]:
        """Step 2: Scan DEX quotes for cross-venue opportunities."""
        if not self._dex_quotes:
            return []
        return scan_opportunities(
            self._dex_quotes,
            min_profit_bps=self._config.profit_threshold_bps,
        )

    def step_check_gas(self) -> GasEstimate:
        """Step 3: Check current gas conditions."""
        return self._gas.estimate(self._config.gas_limit)

    def step_select_provider(self) -> dict[str, Any]:
        """Step 4: Select optimal flash loan provider."""
        return self._provider_selector.select(
            token=self._config.borrow_token,
            amount=self._config.borrow_amount,
            preferred=self._config.provider,
        )

    # ── Main Tick ─────────────────────────────────────────────

    def run_once(self) -> PipelineTickResult:
        """
        Execute one full pipeline tick.

        1. Detect signals
        2. Scan opportunities
        3. Check gas
        4. For each viable opportunity: simulate → execute
        """
        self._state.total_ticks += 1
        result = PipelineTickResult(tick_number=self._state.total_ticks)

        # Check if orchestrator allows trading
        if not self._orchestrator.state.can_trade:
            result.skipped_reason = "Orchestrator paused"
            return result

        # Step 1: Signals
        signals = self.step_detect_signals()
        result.signals_generated = len(signals)
        self._state.total_signals += len(signals)

        # Step 2: Opportunities
        opportunities = self.step_scan_opportunities()
        result.opportunities_found = len(opportunities)
        self._state.total_opportunities += len(opportunities)

        # Step 3: Gas
        gas = self.step_check_gas()
        if not gas.is_acceptable:
            result.skipped_reason = f"Gas too high: {gas.total_gwei:.1f} gwei"
            return result

        # Step 4: Provider
        provider = self.step_select_provider()
        if provider.get("provider") is None:
            result.skipped_reason = "No flash loan provider available"
            return result

        # Step 5: Evaluate and execute signals
        for signal in signals:
            evaluation = self._orchestrator.evaluate_opportunity(
                signal, gas_cost_usd=gas.estimated_cost_usd,
                flash_fee_usd=provider.get("fee", 0),
            )

            if not evaluation.get("go"):
                result.details.append(f"Skip {signal.pair}: {evaluation.get('reason')}")
                continue

            result.simulations_run += 1

            # Execute
            exec_result = self._orchestrator.execute_opportunity(
                signal, gas_cost_usd=gas.estimated_cost_usd,
            )

            result.executions_attempted += 1
            self._state.total_executions += 1

            if exec_result.get("success"):
                profit = exec_result.get("profit", 0)
                result.executions_succeeded += 1
                result.profit_usd += profit
                result.gas_cost_usd += gas.estimated_cost_usd
                self._state.total_successes += 1
                self._state.total_profit_usd += profit
                self._state.total_gas_usd += gas.estimated_cost_usd
                result.details.append(f"SUCCESS {signal.pair}: +${profit:.2f}")
            else:
                result.gas_cost_usd += gas.estimated_cost_usd
                self._state.total_gas_usd += gas.estimated_cost_usd
                result.details.append(
                    f"FAIL {signal.pair}: {exec_result.get('error', 'unknown')}"
                )

        return result

    def run_loop(self, max_ticks: int = 10) -> list[PipelineTickResult]:
        """
        Run the pipeline for multiple ticks.

        In production this would run continuously with sleep intervals.
        For testing, runs synchronously for max_ticks iterations.
        """
        self._state.is_running = True
        self._state.start_time = datetime.now(timezone.utc)
        results = []

        for _ in range(max_ticks):
            if not self._state.is_running:
                break
            tick_result = self.run_once()
            results.append(tick_result)

        self._state.is_running = False
        return results

    def stop(self) -> None:
        """Stop the pipeline."""
        self._state.is_running = False

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Get full pipeline summary."""
        return {
            "pipeline": "flash_loan_atomic_arb",
            "chain": self._config.chain,
            "provider": self._config.provider,
            "borrow_amount": self._config.borrow_amount,
            "total_ticks": self._state.total_ticks,
            "total_signals": self._state.total_signals,
            "total_opportunities": self._state.total_opportunities,
            "total_executions": self._state.total_executions,
            "success_rate": f"{self._state.success_rate:.1%}",
            "net_profit_usd": f"${self._state.net_profit:.2f}",
            "total_gas_usd": f"${self._state.total_gas_usd:.2f}",
            "orchestrator": self._orchestrator.summary(),
            "executor": self._executor.flash_stats,
            "mev": self._mev.stats,
        }
