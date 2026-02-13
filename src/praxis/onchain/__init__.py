"""
On-Chain Execution Layer (Phase 5.1 + 5.3, §14.5-14.6).

Python-side interfaces for smart contract interaction:
- Contract ABI/interface definitions
- OnChainAtomicExecutor: profit-or-revert atomic execution
- FlashLoanAtomicExecutor: zero-capital flash-loan-funded execution
- SimulationResult / ExecutionResult types
- Gas estimation and profit threshold validation

The actual Solidity contracts are deployed separately. This module
provides the Python orchestration layer.

Usage:
    executor = OnChainAtomicExecutor(config)
    sim = executor.simulate(params)
    if sim.is_profitable:
        result = executor.execute(params)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Contract ABIs (minimal for interface definition)
# ═══════════════════════════════════════════════════════════════════

FLASH_ARBITRAGE_ABI = [
    {
        "name": "executeExchangeArbitrage",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "tokenIn", "type": "address"},
            {"name": "tokenOut", "type": "address"},
            {"name": "amountIn", "type": "uint256"},
            {"name": "router1", "type": "address"},
            {"name": "router2", "type": "address"},
            {"name": "profitThresholdBps", "type": "uint256"},
        ],
        "outputs": [{"name": "profit", "type": "uint256"}],
    },
    {
        "name": "simulateExchangeArbitrage",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "tokenIn", "type": "address"},
            {"name": "tokenOut", "type": "address"},
            {"name": "amountIn", "type": "uint256"},
            {"name": "router1", "type": "address"},
            {"name": "router2", "type": "address"},
        ],
        "outputs": [
            {"name": "expectedProfit", "type": "uint256"},
            {"name": "profitBps", "type": "uint256"},
        ],
    },
    {
        "name": "executeFlashLoanArbitrage",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "flashLoanProvider", "type": "address"},
            {"name": "borrowToken", "type": "address"},
            {"name": "borrowAmount", "type": "uint256"},
            {"name": "router1", "type": "address"},
            {"name": "router2", "type": "address"},
            {"name": "tokenMid", "type": "address"},
            {"name": "profitThresholdBps", "type": "uint256"},
        ],
        "outputs": [{"name": "profit", "type": "uint256"}],
    },
    {
        "name": "owner",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    },
    {
        "name": "withdrawToken",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "token", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [],
    },
]

# Well-known addresses
KNOWN_ROUTERS = {
    "uniswap_v2": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
    "uniswap_v3": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
}

KNOWN_FLASH_PROVIDERS = {
    "balancer": "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
    "aave_v3": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
}

KNOWN_TOKENS = {
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
}


# ═══════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════

class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    REVERTED = "reverted"
    FAILED = "failed"
    SIMULATED = "simulated"


@dataclass
class ArbitrageParams:
    """Parameters for an atomic arbitrage execution."""
    token_in: str = ""           # Address or symbol
    token_out: str = ""
    amount_in: float = 0.0       # In token units
    amount_in_wei: int = 0       # In wei
    router1: str = ""            # Buy venue
    router2: str = ""            # Sell venue
    profit_threshold_bps: int = 30
    max_gas_price_gwei: float = 100.0
    gas_limit: int = 500_000
    simulation_first: bool = True


@dataclass
class FlashLoanParams:
    """Parameters for flash-loan-funded arbitrage."""
    provider: str = "balancer"          # balancer, aave_v3
    fallback_provider: str = "aave_v3"
    borrow_token: str = "USDC"
    borrow_amount: float = 1_000_000    # In token units
    borrow_amount_wei: int = 0
    fee_bps: float = 0.0                # 0 for Balancer, 5 for Aave V3
    arb_params: ArbitrageParams = field(default_factory=ArbitrageParams)


@dataclass
class SimulationResult:
    """Result of a contract simulation (view function, no gas)."""
    expected_profit: float = 0.0
    expected_profit_bps: int = 0
    gas_estimate: int = 0
    gas_cost_eth: float = 0.0
    gas_cost_usd: float = 0.0
    net_profit: float = 0.0
    is_profitable: bool = False
    route: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def profit_after_gas(self) -> float:
        return self.expected_profit - self.gas_cost_usd


@dataclass
class ExecutionResult:
    """Result of an on-chain execution."""
    status: ExecutionStatus = ExecutionStatus.FAILED
    tx_hash: str = ""
    profit: float = 0.0
    profit_bps: int = 0
    gas_used: int = 0
    gas_price_gwei: float = 0.0
    gas_cost_eth: float = 0.0
    gas_cost_usd: float = 0.0
    block_number: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str = ""
    revert_reason: str = ""
    attempts: int = 1

    @property
    def net_profit(self) -> float:
        return self.profit - self.gas_cost_usd

    @property
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS


# ═══════════════════════════════════════════════════════════════════
#  On-Chain Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OnChainConfig:
    """Configuration for on-chain execution."""
    chain: str = "ethereum"
    network: str = "mainnet"
    rpc_url: str = ""
    contract_address: str = ""
    private_key: str = ""           # For signing transactions
    profit_threshold_bps: int = 30
    max_gas_price_gwei: float = 100.0
    gas_limit: int = 500_000
    slippage_tolerance_bps: int = 50
    simulation_before_execution: bool = True
    supported_routers: dict[str, str] = field(default_factory=lambda: dict(KNOWN_ROUTERS))
    eth_price_usd: float = 3000.0   # For gas cost calculation


# ═══════════════════════════════════════════════════════════════════
#  Executors
# ═══════════════════════════════════════════════════════════════════

class OnChainAtomicExecutor:
    """
    Atomic arbitrage executor (§14.5).

    Profit-or-revert semantics: trade either completes profitably
    or the entire transaction reverts. Gas is the only cost on failure.

    In test/simulation mode, no RPC calls are made.
    """

    def __init__(self, config: OnChainConfig | None = None):
        self._config = config or OnChainConfig()
        self._log = PraxisLogger.instance()
        self._execution_count = 0
        self._total_profit = 0.0
        self._total_gas_spent = 0.0
        self._simulation_fn = None   # Pluggable for testing
        self._execution_fn = None

    @property
    def config(self) -> OnChainConfig:
        return self._config

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "executions": self._execution_count,
            "total_profit_usd": self._total_profit,
            "total_gas_usd": self._total_gas_spent,
            "net_profit_usd": self._total_profit - self._total_gas_spent,
        }

    def set_simulation_fn(self, fn) -> None:
        """Set a custom simulation function (for testing)."""
        self._simulation_fn = fn

    def set_execution_fn(self, fn) -> None:
        """Set a custom execution function (for testing)."""
        self._execution_fn = fn

    def resolve_address(self, name: str) -> str:
        """Resolve a token/router name to address."""
        if name.startswith("0x"):
            return name
        addr = KNOWN_TOKENS.get(name) or KNOWN_ROUTERS.get(name)
        if addr:
            return addr
        return self._config.supported_routers.get(name, name)

    def simulate(self, params: ArbitrageParams) -> SimulationResult:
        """
        Simulate arbitrage via contract view function (no gas).

        In test mode, uses the pluggable simulation function.
        """
        if self._simulation_fn:
            return self._simulation_fn(params)

        # Default simulation: estimate based on params
        gas_cost_eth = params.gas_limit * params.max_gas_price_gwei * 1e-9
        gas_cost_usd = gas_cost_eth * self._config.eth_price_usd

        return SimulationResult(
            expected_profit=0.0,
            expected_profit_bps=0,
            gas_estimate=params.gas_limit,
            gas_cost_eth=gas_cost_eth,
            gas_cost_usd=gas_cost_usd,
            is_profitable=False,
            route=f"{params.router1} → {params.router2}",
        )

    def execute(self, params: ArbitrageParams) -> ExecutionResult:
        """
        Execute atomic arbitrage on-chain.

        1. Check gas price vs max
        2. Optionally simulate first
        3. Submit transaction
        4. Track result
        """
        # Gas check
        if params.max_gas_price_gwei > self._config.max_gas_price_gwei:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Gas price {params.max_gas_price_gwei} > max {self._config.max_gas_price_gwei}",
            )

        # Simulate first if configured
        if self._config.simulation_before_execution:
            sim = self.simulate(params)
            if not sim.is_profitable:
                return ExecutionResult(
                    status=ExecutionStatus.SIMULATED,
                    error="Simulation showed unprofitable",
                    gas_cost_usd=0.0,  # No gas spent on view call
                )

        # Execute
        if self._execution_fn:
            result = self._execution_fn(params)
        else:
            # Default: no RPC available
            gas_cost_eth = params.gas_limit * params.max_gas_price_gwei * 1e-9
            gas_cost_usd = gas_cost_eth * self._config.eth_price_usd
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                error="No RPC connection configured",
                gas_cost_eth=gas_cost_eth,
                gas_cost_usd=gas_cost_usd,
            )

        # Track stats
        self._execution_count += 1
        if result.is_success:
            self._total_profit += result.profit
        self._total_gas_spent += result.gas_cost_usd

        return result


class FlashLoanAtomicExecutor(OnChainAtomicExecutor):
    """
    Flash-loan-funded atomic arbitrage (§14.6).

    Extends OnChainAtomicExecutor with zero-capital execution:
    Borrow → arb → repay → profit, all in one atomic transaction.
    """

    def __init__(
        self,
        config: OnChainConfig | None = None,
        default_provider: str = "balancer",
    ):
        super().__init__(config)
        self._default_provider = default_provider
        self._flash_execution_fn = None
        self._failed_attempts = 0
        self._successful_attempts = 0

    @property
    def flash_stats(self) -> dict[str, Any]:
        base = self.stats
        base.update({
            "failed_attempts": self._failed_attempts,
            "successful_attempts": self._successful_attempts,
            "success_rate": (
                self._successful_attempts / max(1, self._failed_attempts + self._successful_attempts)
            ),
            "default_provider": self._default_provider,
        })
        return base

    def set_flash_execution_fn(self, fn) -> None:
        """Set custom flash loan execution function (for testing)."""
        self._flash_execution_fn = fn

    def get_provider_address(self, provider: str) -> str:
        """Resolve flash loan provider name to address."""
        return KNOWN_FLASH_PROVIDERS.get(provider, provider)

    def get_provider_fee_bps(self, provider: str) -> float:
        """Get flash loan fee for a provider."""
        fees = {
            "balancer": 0.0,
            "aave_v3": 5.0,    # 0.05%
            "dodo": 0.0,
        }
        return fees.get(provider, 0.0)

    def estimate_flash_loan_cost(self, params: FlashLoanParams) -> float:
        """Estimate flash loan fee in USD."""
        fee_bps = params.fee_bps or self.get_provider_fee_bps(params.provider)
        return params.borrow_amount * fee_bps / 10_000

    def execute_with_flash_loan(self, params: FlashLoanParams) -> ExecutionResult:
        """
        Execute flash-loan-funded atomic arbitrage.

        Flow:
        1. Borrow from flash loan provider
        2. Execute arbitrage legs
        3. Repay loan + fee
        4. Keep profit

        Entire transaction reverts if unprofitable.
        """
        # Validate
        flash_fee = self.estimate_flash_loan_cost(params)
        arb = params.arb_params

        # Simulate first
        if self._config.simulation_before_execution:
            sim = self.simulate(arb)
            # Must cover flash loan fee + gas + profit threshold
            min_profit = flash_fee + sim.gas_cost_usd
            if sim.expected_profit < min_profit:
                self._failed_attempts += 1
                return ExecutionResult(
                    status=ExecutionStatus.SIMULATED,
                    error=f"Expected profit ${sim.expected_profit:.2f} < min ${min_profit:.2f}",
                )

        # Execute
        if self._flash_execution_fn:
            result = self._flash_execution_fn(params)
        else:
            gas_cost_eth = arb.gas_limit * arb.max_gas_price_gwei * 1e-9
            gas_cost_usd = gas_cost_eth * self._config.eth_price_usd
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                error="No RPC connection configured",
                gas_cost_eth=gas_cost_eth,
                gas_cost_usd=gas_cost_usd,
            )

        # Track
        self._execution_count += 1
        if result.is_success:
            self._successful_attempts += 1
            self._total_profit += result.profit
        else:
            self._failed_attempts += 1
        self._total_gas_spent += result.gas_cost_usd

        return result
