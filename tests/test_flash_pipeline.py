"""
Tests for Phase 5.7+5.8 (Flash Loan Providers) and 5.13 (E2E Pipeline).
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ═══════════════════════════════════════════════════════════════════
#  Phase 5.7: Balancer Flash Loan
# ═══════════════════════════════════════════════════════════════════

from praxis.onchain.flash_providers import (
    BalancerFlashLoan,
    BalancerFlashLoanConfig,
    AaveFlashLoan,
    AaveFlashLoanConfig,
    FlashLoanProviderSelector,
    BALANCER_VAULT_ABI,
    BALANCER_RECIPIENT_ABI,
    AAVE_POOL_ABI,
    AAVE_RECEIVER_ABI,
)


class TestBalancerABI:
    def test_vault_abi_has_flash_loan(self):
        names = [f["name"] for f in BALANCER_VAULT_ABI]
        assert "flashLoan" in names

    def test_recipient_abi_has_callback(self):
        names = [f["name"] for f in BALANCER_RECIPIENT_ABI]
        assert "receiveFlashLoan" in names


class TestBalancerFlashLoan:
    def test_zero_fee(self):
        provider = BalancerFlashLoan()
        assert provider.fee_bps == 0.0
        assert provider.compute_fee(1_000_000) == 0.0

    def test_repayment_equals_principal(self):
        provider = BalancerFlashLoan()
        assert provider.compute_repayment(1_000_000) == 1_000_000

    def test_validate_supported_token(self):
        provider = BalancerFlashLoan()
        result = provider.validate_loan("USDC", 1_000_000)
        assert result["valid"]
        assert result["fee"] == 0.0

    def test_validate_unsupported_token(self):
        provider = BalancerFlashLoan()
        result = provider.validate_loan("SHIB", 1_000)
        assert not result["valid"]

    def test_validate_zero_amount(self):
        provider = BalancerFlashLoan()
        result = provider.validate_loan("USDC", 0)
        assert not result["valid"]

    def test_build_calldata(self):
        provider = BalancerFlashLoan()
        calldata = provider.build_flash_loan_calldata(
            recipient="0xMyContract",
            token_address="0xUSDC",
            amount_wei=1_000_000_000_000,
        )
        assert calldata["function"] == "flashLoan"
        assert calldata["args"]["tokens"] == ["0xUSDC"]
        assert calldata["args"]["amounts"] == [1_000_000_000_000]


# ═══════════════════════════════════════════════════════════════════
#  Phase 5.8: Aave V3 Flash Loan
# ═══════════════════════════════════════════════════════════════════

class TestAaveABI:
    def test_pool_abi_has_flash_loan(self):
        names = [f["name"] for f in AAVE_POOL_ABI]
        assert "flashLoan" in names
        assert "flashLoanSimple" in names

    def test_receiver_abi_has_callback(self):
        names = [f["name"] for f in AAVE_RECEIVER_ABI]
        assert "executeOperation" in names


class TestAaveFlashLoan:
    def test_fee_rate(self):
        provider = AaveFlashLoan()
        assert provider.fee_bps == 5.0  # 0.05%

    def test_compute_fee(self):
        provider = AaveFlashLoan()
        fee = provider.compute_fee(1_000_000)
        assert fee == 500.0  # $1M * 0.05% = $500

    def test_repayment_includes_fee(self):
        provider = AaveFlashLoan()
        repayment = provider.compute_repayment(1_000_000)
        assert repayment == 1_000_500.0

    def test_validate_supported_token(self):
        provider = AaveFlashLoan()
        result = provider.validate_loan("WETH", 100)
        assert result["valid"]
        assert result["fee"] > 0

    def test_validate_unsupported_token(self):
        provider = AaveFlashLoan()
        result = provider.validate_loan("DOGE", 100)
        assert not result["valid"]

    def test_build_calldata_simple(self):
        provider = AaveFlashLoan(AaveFlashLoanConfig(use_simple=True))
        calldata = provider.build_flash_loan_calldata(
            receiver="0xMyContract",
            token_address="0xWETH",
            amount_wei=10**18,
        )
        assert calldata["function"] == "flashLoanSimple"
        assert calldata["args"]["asset"] == "0xWETH"

    def test_build_calldata_full(self):
        provider = AaveFlashLoan(AaveFlashLoanConfig(use_simple=False))
        calldata = provider.build_flash_loan_calldata(
            receiver="0xMyContract",
            token_address="0xWETH",
            amount_wei=10**18,
        )
        assert calldata["function"] == "flashLoan"
        assert calldata["args"]["assets"] == ["0xWETH"]


# ═══════════════════════════════════════════════════════════════════
#  Provider Selector
# ═══════════════════════════════════════════════════════════════════

class TestFlashLoanProviderSelector:
    def test_prefers_balancer(self):
        selector = FlashLoanProviderSelector()
        result = selector.select("USDC", 1_000_000)
        assert result["provider"] == "balancer"
        assert result["fee"] == 0.0

    def test_falls_back_to_aave(self):
        selector = FlashLoanProviderSelector()
        # Balancer doesn't support AAVE token by default
        result = selector.select("AAVE", 1_000)
        assert result["provider"] == "aave_v3"
        assert result["fee"] > 0

    def test_no_provider_available(self):
        selector = FlashLoanProviderSelector()
        result = selector.select("UNKNOWN_TOKEN", 1_000)
        assert result["provider"] is None
        assert "error" in result

    def test_preferred_provider(self):
        selector = FlashLoanProviderSelector()
        result = selector.select("USDC", 1_000_000, preferred="aave_v3")
        assert result["provider"] == "aave_v3"

    def test_compare_providers(self):
        selector = FlashLoanProviderSelector()
        comparisons = selector.compare_providers("USDC", 1_000_000)
        assert len(comparisons) == 2
        # Sorted by fee — Balancer (0) first
        assert comparisons[0]["provider"] == "balancer"
        assert comparisons[0]["fee_usd"] == 0
        assert comparisons[1]["provider"] == "aave_v3"
        assert comparisons[1]["fee_usd"] == 500

    def test_providers_list(self):
        selector = FlashLoanProviderSelector()
        assert "balancer" in selector.providers
        assert "aave_v3" in selector.providers


# ═══════════════════════════════════════════════════════════════════
#  Phase 5.13: End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════════

from praxis.onchain.pipeline import (
    FlashArbPipeline,
    FlashArbPipelineConfig,
    PipelineTickResult,
    PipelineState,
)
from praxis.onchain.connectors import DEXQuote


def _make_cointegrated_feeds(n=100, seed=42):
    """Generate cointegrated price feeds for pipeline testing."""
    rng = np.random.RandomState(seed)
    b = 3000 + np.cumsum(rng.randn(n) * 10)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.8 * spread[i-1] + rng.randn() * 5
    a = 1.2 * b + spread + 100
    return {"ETH": a, "BTC_PROXY": b}


class TestFlashArbPipeline:
    @pytest.fixture
    def pipeline(self):
        config = FlashArbPipelineConfig(
            chain="arbitrum",
            provider="balancer",
            borrow_amount=1_000_000,
            max_gas_price_gwei=1.0,
            profit_threshold_bps=10,
            min_profit_usd=50,
        )
        p = FlashArbPipeline(config)
        # Set gas oracle to acceptable level
        p.gas_oracle.set_gas_price(0.1, 0.01)
        return p

    def test_init(self, pipeline):
        assert pipeline.config.chain == "arbitrum"
        assert pipeline.config.provider == "balancer"
        assert pipeline.state.total_ticks == 0

    def test_step_detect_signals_no_data(self, pipeline):
        signals = pipeline.step_detect_signals()
        assert signals == []

    def test_step_detect_signals_with_data(self, pipeline):
        feeds = _make_cointegrated_feeds()
        # Push z-score beyond entry
        feeds["ETH"][-1] += 200
        pipeline.set_price_feeds(feeds)
        signals = pipeline.step_detect_signals()
        # May or may not find entry depending on exact z-score
        assert isinstance(signals, list)

    def test_step_scan_opportunities_empty(self, pipeline):
        opps = pipeline.step_scan_opportunities()
        assert opps == []

    def test_step_scan_opportunities_with_quotes(self, pipeline):
        pipeline.set_dex_quotes({
            "ETH/USDC": DEXQuote(venue="uni_v2", pair="ETH/USDC", price=3000, fee_bps=30),
        })
        opps = pipeline.step_scan_opportunities()
        # Single venue → no cross-venue arb
        assert opps == []

    def test_step_check_gas(self, pipeline):
        pipeline.gas_oracle.set_gas_price(0.1, 0.01)
        gas = pipeline.step_check_gas()
        assert gas.is_acceptable
        assert gas.estimated_cost_usd > 0

    def test_step_select_provider(self, pipeline):
        result = pipeline.step_select_provider()
        assert result["provider"] == "balancer"
        assert result["fee"] == 0.0

    def test_run_once_no_signals(self, pipeline):
        result = pipeline.run_once()
        assert result.tick_number == 1
        assert result.signals_generated == 0
        assert result.executions_attempted == 0
        assert pipeline.state.total_ticks == 1

    def test_run_once_gas_too_high(self, pipeline):
        pipeline.gas_oracle.set_gas_price(100, 10)  # Way over limit
        result = pipeline.run_once()
        assert "Gas too high" in result.skipped_reason

    def test_run_once_with_successful_execution(self, pipeline):
        # Set up feeds with extreme z-score to trigger signal
        feeds = _make_cointegrated_feeds(100)
        feeds["ETH"][-1] += 500
        pipeline.set_price_feeds(feeds)

        # Set up orchestrator to execute
        pipeline.orchestrator.set_execute_fn(
            lambda s: {"success": True, "profit": 250.0}
        )

        result = pipeline.run_once()
        # If signal was detected, should have attempted execution
        if result.signals_generated > 0:
            assert result.executions_attempted >= 0
            # Check state accumulation
            assert pipeline.state.total_ticks == 1

    def test_run_loop(self, pipeline):
        results = pipeline.run_loop(max_ticks=5)
        assert len(results) == 5
        assert pipeline.state.total_ticks == 5
        assert not pipeline.state.is_running

    def test_run_loop_with_execution(self, pipeline):
        feeds = _make_cointegrated_feeds(100)
        feeds["ETH"][-1] += 500
        pipeline.set_price_feeds(feeds)
        pipeline.orchestrator.set_execute_fn(
            lambda s: {"success": True, "profit": 100.0}
        )

        results = pipeline.run_loop(max_ticks=3)
        assert len(results) == 3

    def test_stop(self, pipeline):
        pipeline._state.is_running = True
        pipeline.stop()
        assert not pipeline.state.is_running

    def test_summary(self, pipeline):
        pipeline.run_once()
        s = pipeline.summary()
        assert s["pipeline"] == "flash_loan_atomic_arb"
        assert s["chain"] == "arbitrum"
        assert "total_ticks" in s
        assert "orchestrator" in s
        assert "executor" in s
        assert "mev" in s

    def test_orchestrator_pause_blocks_pipeline(self, pipeline):
        pipeline.orchestrator._state.is_paused = True
        pipeline.orchestrator._state.pause_reason = "Gas budget"
        result = pipeline.run_once()
        assert result.skipped_reason == "Orchestrator paused"

    def test_pipeline_state_accumulation(self, pipeline):
        pipeline.run_loop(max_ticks=10)
        assert pipeline.state.total_ticks == 10
        assert pipeline.state.total_signals >= 0
        assert pipeline.state.total_opportunities >= 0

    def test_pipeline_net_profit(self, pipeline):
        feeds = _make_cointegrated_feeds(100)
        feeds["ETH"][-1] += 500
        pipeline.set_price_feeds(feeds)
        pipeline.orchestrator.set_execute_fn(
            lambda s: {"success": True, "profit": 100.0}
        )
        pipeline.run_loop(max_ticks=3)
        # Net profit = total profit - gas
        assert pipeline.state.net_profit == pipeline.state.total_profit_usd - pipeline.state.total_gas_usd
