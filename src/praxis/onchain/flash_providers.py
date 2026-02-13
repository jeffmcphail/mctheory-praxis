"""
Flash Loan Provider Interfaces (Phase 5.7 + 5.8).

Python-side ABI definitions and interaction helpers for:
- Balancer Vault IFlashLoanRecipient (0% fee)
- Aave V3 Pool IFlashLoanReceiver (0.05% fee)
- Provider selection and fallback logic

The Solidity contract implements these interfaces. This module
provides the Python orchestration to construct and validate
the flash loan parameters before on-chain execution.

Usage:
    provider = FlashLoanProviderSelector()
    best = provider.select(borrow_amount=1_000_000, token="USDC")
    tx_data = best.build_flash_loan_tx(params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ═══════════════════════════════════════════════════════════════════
#  Balancer Flash Loan (Phase 5.7)
# ═══════════════════════════════════════════════════════════════════

BALANCER_VAULT_ABI = [
    {
        "name": "flashLoan",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "recipient", "type": "address"},
            {"name": "tokens", "type": "address[]"},
            {"name": "amounts", "type": "uint256[]"},
            {"name": "userData", "type": "bytes"},
        ],
        "outputs": [],
    },
]

# IFlashLoanRecipient — our contract must implement this
BALANCER_RECIPIENT_ABI = [
    {
        "name": "receiveFlashLoan",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "tokens", "type": "address[]"},
            {"name": "amounts", "type": "uint256[]"},
            {"name": "feeAmounts", "type": "uint256[]"},
            {"name": "userData", "type": "bytes"},
        ],
        "outputs": [],
    },
]


@dataclass
class BalancerFlashLoanConfig:
    """Configuration for Balancer flash loans."""
    vault_address: str = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"
    fee_bps: float = 0.0       # Balancer: zero fee
    max_tokens_per_loan: int = 5
    supported_tokens: list[str] = field(default_factory=lambda: [
        "USDC", "USDT", "DAI", "WETH", "WBTC",
    ])


class BalancerFlashLoan:
    """
    Balancer Vault flash loan interface.

    Zero-fee flash loans via IFlashLoanRecipient callback.
    Our contract receives tokens, executes arb, repays exact amount.
    """

    PROVIDER = "balancer"
    FEE_BPS = 0.0

    def __init__(self, config: BalancerFlashLoanConfig | None = None):
        self._config = config or BalancerFlashLoanConfig()

    @property
    def config(self) -> BalancerFlashLoanConfig:
        return self._config

    @property
    def fee_bps(self) -> float:
        return self.FEE_BPS

    def compute_fee(self, amount: float) -> float:
        """Compute flash loan fee. Always 0 for Balancer."""
        return 0.0

    def compute_repayment(self, amount: float) -> float:
        """Compute total repayment (principal + fee)."""
        return amount  # No fee

    def validate_loan(self, token: str, amount: float) -> dict[str, Any]:
        """Validate flash loan parameters."""
        issues = []

        if token not in self._config.supported_tokens:
            issues.append(f"Token {token} not in supported list")

        if amount <= 0:
            issues.append("Amount must be positive")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "provider": self.PROVIDER,
            "fee": self.compute_fee(amount),
            "repayment": self.compute_repayment(amount),
        }

    def build_flash_loan_calldata(
        self,
        recipient: str,
        token_address: str,
        amount_wei: int,
        user_data: bytes = b"",
    ) -> dict[str, Any]:
        """
        Build calldata for Balancer Vault.flashLoan().

        Returns dict representing the transaction parameters.
        """
        return {
            "to": self._config.vault_address,
            "function": "flashLoan",
            "args": {
                "recipient": recipient,
                "tokens": [token_address],
                "amounts": [amount_wei],
                "userData": user_data.hex() if isinstance(user_data, bytes) else user_data,
            },
        }


# ═══════════════════════════════════════════════════════════════════
#  Aave V3 Flash Loan (Phase 5.8)
# ═══════════════════════════════════════════════════════════════════

AAVE_POOL_ABI = [
    {
        "name": "flashLoan",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "receiverAddress", "type": "address"},
            {"name": "assets", "type": "address[]"},
            {"name": "amounts", "type": "uint256[]"},
            {"name": "interestRateModes", "type": "uint256[]"},
            {"name": "onBehalfOf", "type": "address"},
            {"name": "params", "type": "bytes"},
            {"name": "referralCode", "type": "uint16"},
        ],
        "outputs": [],
    },
    {
        "name": "flashLoanSimple",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "receiverAddress", "type": "address"},
            {"name": "asset", "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "params", "type": "bytes"},
            {"name": "referralCode", "type": "uint16"},
        ],
        "outputs": [],
    },
]

# IFlashLoanReceiver — our contract must implement this
AAVE_RECEIVER_ABI = [
    {
        "name": "executeOperation",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "assets", "type": "address[]"},
            {"name": "amounts", "type": "uint256[]"},
            {"name": "premiums", "type": "uint256[]"},
            {"name": "initiator", "type": "address"},
            {"name": "params", "type": "bytes"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
]


@dataclass
class AaveFlashLoanConfig:
    """Configuration for Aave V3 flash loans."""
    pool_address: str = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
    fee_bps: float = 5.0       # 0.05%
    referral_code: int = 0
    use_simple: bool = True     # flashLoanSimple for single-asset
    supported_tokens: list[str] = field(default_factory=lambda: [
        "USDC", "USDT", "DAI", "WETH", "WBTC", "AAVE", "LINK",
    ])


class AaveFlashLoan:
    """
    Aave V3 Pool flash loan interface.

    0.05% fee flash loans via IFlashLoanReceiver callback.
    Fallback provider when Balancer lacks liquidity.
    """

    PROVIDER = "aave_v3"
    FEE_BPS = 5.0  # 0.05%

    def __init__(self, config: AaveFlashLoanConfig | None = None):
        self._config = config or AaveFlashLoanConfig()

    @property
    def config(self) -> AaveFlashLoanConfig:
        return self._config

    @property
    def fee_bps(self) -> float:
        return self._config.fee_bps

    def compute_fee(self, amount: float) -> float:
        """Compute flash loan fee."""
        return amount * self._config.fee_bps / 10_000

    def compute_repayment(self, amount: float) -> float:
        """Compute total repayment (principal + fee)."""
        return amount + self.compute_fee(amount)

    def validate_loan(self, token: str, amount: float) -> dict[str, Any]:
        """Validate flash loan parameters."""
        issues = []

        if token not in self._config.supported_tokens:
            issues.append(f"Token {token} not in supported list")

        if amount <= 0:
            issues.append("Amount must be positive")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "provider": self.PROVIDER,
            "fee": self.compute_fee(amount),
            "repayment": self.compute_repayment(amount),
        }

    def build_flash_loan_calldata(
        self,
        receiver: str,
        token_address: str,
        amount_wei: int,
        params: bytes = b"",
    ) -> dict[str, Any]:
        """Build calldata for Aave V3 flashLoanSimple()."""
        if self._config.use_simple:
            return {
                "to": self._config.pool_address,
                "function": "flashLoanSimple",
                "args": {
                    "receiverAddress": receiver,
                    "asset": token_address,
                    "amount": amount_wei,
                    "params": params.hex() if isinstance(params, bytes) else params,
                    "referralCode": self._config.referral_code,
                },
            }
        else:
            return {
                "to": self._config.pool_address,
                "function": "flashLoan",
                "args": {
                    "receiverAddress": receiver,
                    "assets": [token_address],
                    "amounts": [amount_wei],
                    "interestRateModes": [0],  # 0 = no debt
                    "onBehalfOf": receiver,
                    "params": params.hex() if isinstance(params, bytes) else params,
                    "referralCode": self._config.referral_code,
                },
            }


# ═══════════════════════════════════════════════════════════════════
#  Provider Selector
# ═══════════════════════════════════════════════════════════════════

class FlashLoanProviderSelector:
    """
    Select optimal flash loan provider.

    Priority: lowest fee → most liquidity → fallback.
    Default order: Balancer (0%) → Aave V3 (0.05%).
    """

    def __init__(self):
        self._providers: dict[str, BalancerFlashLoan | AaveFlashLoan] = {
            "balancer": BalancerFlashLoan(),
            "aave_v3": AaveFlashLoan(),
        }
        self._preference_order = ["balancer", "aave_v3"]

    @property
    def providers(self) -> list[str]:
        return list(self._providers.keys())

    def add_provider(self, name: str, provider: BalancerFlashLoan | AaveFlashLoan) -> None:
        self._providers[name] = provider
        if name not in self._preference_order:
            self._preference_order.append(name)

    def select(
        self,
        token: str,
        amount: float,
        preferred: str | None = None,
    ) -> dict[str, Any]:
        """
        Select best flash loan provider.

        Returns selection result with provider, fee, and validation.
        """
        # Try preferred first
        if preferred and preferred in self._providers:
            provider = self._providers[preferred]
            validation = provider.validate_loan(token, amount)
            if validation["valid"]:
                return {
                    "provider": preferred,
                    "fee": validation["fee"],
                    "repayment": validation["repayment"],
                    "instance": provider,
                }

        # Try in preference order
        for name in self._preference_order:
            provider = self._providers[name]
            validation = provider.validate_loan(token, amount)
            if validation["valid"]:
                return {
                    "provider": name,
                    "fee": validation["fee"],
                    "repayment": validation["repayment"],
                    "instance": provider,
                }

        return {
            "provider": None,
            "error": f"No provider supports {token} for amount {amount}",
        }

    def compare_providers(
        self,
        token: str,
        amount: float,
    ) -> list[dict[str, Any]]:
        """Compare all providers for a given loan."""
        comparisons = []
        for name, provider in self._providers.items():
            validation = provider.validate_loan(token, amount)
            comparisons.append({
                "provider": name,
                "fee_bps": provider.fee_bps,
                "fee_usd": validation["fee"],
                "repayment": validation["repayment"],
                "valid": validation["valid"],
                "issues": validation.get("issues", []),
            })
        comparisons.sort(key=lambda x: x["fee_usd"])
        return comparisons
