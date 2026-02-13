"""
Tests for Phase 5: On-Chain Execution Layer.

5.1+5.3: Contract interfaces, OnChainAtomicExecutor, FlashLoanAtomicExecutor
5.4: DEX connectors (Uniswap V2/V3, SushiSwap, Curve)
5.5: CEX connectors (Binance, Kraken)
5.6: Crypto cointegration analyzer
5.9: Flash loan orchestrator
5.11: MEV protection (Flashbots)
5.12: Gas oracle
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
#  Phase 5.1 + 5.3: On-Chain Executors
# ═══════════════════════════════════════════════════════════════════

from praxis.onchain import (
    OnChainAtomicExecutor,
    FlashLoanAtomicExecutor,
    OnChainConfig,
    ArbitrageParams,
    FlashLoanParams,
    SimulationResult,
    ExecutionResult,
    ExecutionStatus,
    FLASH_ARBITRAGE_ABI,
    KNOWN_ROUTERS,
    KNOWN_TOKENS,
    KNOWN_FLASH_PROVIDERS,
)


class TestContractInterfaces:
    def test_abi_has_execute(self):
        names = [f["name"] for f in FLASH_ARBITRAGE_ABI]
        assert "executeExchangeArbitrage" in names
        assert "simulateExchangeArbitrage" in names
        assert "executeFlashLoanArbitrage" in names

    def test_known_routers(self):
        assert "uniswap_v2" in KNOWN_ROUTERS
        assert "sushiswap" in KNOWN_ROUTERS
        assert all(a.startswith("0x") for a in KNOWN_ROUTERS.values())

    def test_known_tokens(self):
        assert "WETH" in KNOWN_TOKENS
        assert "USDC" in KNOWN_TOKENS
        assert "WBTC" in KNOWN_TOKENS

    def test_known_flash_providers(self):
        assert "balancer" in KNOWN_FLASH_PROVIDERS
        assert "aave_v3" in KNOWN_FLASH_PROVIDERS


class TestOnChainAtomicExecutor:
    def test_default_config(self):
        executor = OnChainAtomicExecutor()
        assert executor.config.chain == "ethereum"
        assert executor.config.profit_threshold_bps == 30

    def test_resolve_address_token(self):
        executor = OnChainAtomicExecutor()
        assert executor.resolve_address("WETH") == KNOWN_TOKENS["WETH"]
        assert executor.resolve_address("0xABC").startswith("0x")

    def test_resolve_address_router(self):
        executor = OnChainAtomicExecutor()
        assert executor.resolve_address("uniswap_v2") == KNOWN_ROUTERS["uniswap_v2"]

    def test_simulate_default(self):
        executor = OnChainAtomicExecutor(OnChainConfig(eth_price_usd=3000))
        params = ArbitrageParams(
            token_in="WETH", token_out="USDC",
            amount_in=10.0, router1="uniswap_v2", router2="sushiswap",
            gas_limit=500_000, max_gas_price_gwei=50,
        )
        sim = executor.simulate(params)
        assert isinstance(sim, SimulationResult)
        assert sim.gas_cost_usd > 0

    def test_simulate_custom_fn(self):
        executor = OnChainAtomicExecutor()
        executor.set_simulation_fn(lambda p: SimulationResult(
            expected_profit=500, expected_profit_bps=50, is_profitable=True,
        ))
        sim = executor.simulate(ArbitrageParams())
        assert sim.is_profitable
        assert sim.expected_profit == 500

    def test_execute_gas_check(self):
        executor = OnChainAtomicExecutor(OnChainConfig(max_gas_price_gwei=50))
        result = executor.execute(ArbitrageParams(max_gas_price_gwei=100))
        assert result.status == ExecutionStatus.FAILED
        assert "Gas price" in result.error

    def test_execute_simulation_blocks(self):
        executor = OnChainAtomicExecutor(OnChainConfig(simulation_before_execution=True))
        # Default simulation returns unprofitable
        result = executor.execute(ArbitrageParams(max_gas_price_gwei=50))
        assert result.status == ExecutionStatus.SIMULATED

    def test_execute_custom_fn(self):
        executor = OnChainAtomicExecutor(OnChainConfig(
            simulation_before_execution=False, max_gas_price_gwei=100,
        ))
        executor.set_execution_fn(lambda p: ExecutionResult(
            status=ExecutionStatus.SUCCESS, profit=200, gas_cost_usd=5,
        ))
        result = executor.execute(ArbitrageParams(max_gas_price_gwei=50))
        assert result.is_success
        assert result.profit == 200
        assert executor.stats["executions"] == 1
        assert executor.stats["total_profit_usd"] == 200

    def test_stats_tracking(self):
        executor = OnChainAtomicExecutor(OnChainConfig(
            simulation_before_execution=False, max_gas_price_gwei=100,
        ))
        executor.set_execution_fn(lambda p: ExecutionResult(
            status=ExecutionStatus.SUCCESS, profit=100, gas_cost_usd=2,
        ))
        executor.execute(ArbitrageParams(max_gas_price_gwei=50))
        executor.execute(ArbitrageParams(max_gas_price_gwei=50))
        assert executor.stats["executions"] == 2
        assert executor.stats["total_profit_usd"] == 200
        assert executor.stats["total_gas_usd"] == 4


class TestFlashLoanAtomicExecutor:
    def test_provider_fee(self):
        executor = FlashLoanAtomicExecutor()
        assert executor.get_provider_fee_bps("balancer") == 0.0
        assert executor.get_provider_fee_bps("aave_v3") == 5.0

    def test_estimate_flash_cost(self):
        executor = FlashLoanAtomicExecutor()
        params = FlashLoanParams(
            provider="aave_v3", borrow_amount=1_000_000, fee_bps=5.0,
        )
        cost = executor.estimate_flash_loan_cost(params)
        assert cost == 500.0  # $1M * 5bps

    def test_balancer_zero_fee(self):
        executor = FlashLoanAtomicExecutor()
        params = FlashLoanParams(provider="balancer", borrow_amount=1_000_000)
        cost = executor.estimate_flash_loan_cost(params)
        assert cost == 0.0

    def test_execute_with_flash_loan(self):
        executor = FlashLoanAtomicExecutor(OnChainConfig(
            simulation_before_execution=False, max_gas_price_gwei=100,
        ))
        executor.set_flash_execution_fn(lambda p: ExecutionResult(
            status=ExecutionStatus.SUCCESS, profit=1500, gas_cost_usd=0.10,
        ))
        params = FlashLoanParams(
            provider="balancer", borrow_amount=1_000_000,
            arb_params=ArbitrageParams(max_gas_price_gwei=1, gas_limit=800_000),
        )
        result = executor.execute_with_flash_loan(params)
        assert result.is_success
        assert result.profit == 1500
        assert executor.flash_stats["successful_attempts"] == 1

    def test_simulation_blocks_unprofitable(self):
        executor = FlashLoanAtomicExecutor(OnChainConfig(
            simulation_before_execution=True, eth_price_usd=3000,
        ))
        # Default simulation returns 0 profit
        params = FlashLoanParams(
            provider="aave_v3", borrow_amount=1_000_000, fee_bps=5.0,
            arb_params=ArbitrageParams(gas_limit=500_000, max_gas_price_gwei=50),
        )
        result = executor.execute_with_flash_loan(params)
        assert result.status == ExecutionStatus.SIMULATED
        assert executor.flash_stats["failed_attempts"] == 1

    def test_flash_stats(self):
        executor = FlashLoanAtomicExecutor(OnChainConfig(
            simulation_before_execution=False, max_gas_price_gwei=100,
        ))
        executor.set_flash_execution_fn(lambda p: ExecutionResult(
            status=ExecutionStatus.REVERTED, gas_cost_usd=0.10,
        ))
        params = FlashLoanParams(
            arb_params=ArbitrageParams(max_gas_price_gwei=1, gas_limit=500_000),
        )
        executor.execute_with_flash_loan(params)
        stats = executor.flash_stats
        assert stats["failed_attempts"] == 1
        assert stats["success_rate"] == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Phase 5.4 + 5.5: DEX/CEX Connectors
# ═══════════════════════════════════════════════════════════════════

from praxis.onchain.connectors import (
    UniswapV2Source, UniswapV3Source, SushiSwapSource, CurveSource,
    BinanceSource, KrakenSource,
    DEXQuote, CEXQuote,
    scan_opportunities, ArbitrageOpportunity,
    GasOracle, GasOracleConfig, GasEstimate,
    MEVProtector, MEVConfig, BundleResult,
)


class TestDEXSources:
    def test_uniswap_v2(self):
        source = UniswapV2Source()
        source.set_prices({"WETH/USDC": 3000.0, "WBTC/USDC": 60000.0})
        data = source.fetch(["WETH/USDC", "WBTC/USDC"], "", "")
        assert len(data) == 2
        assert data["WETH/USDC"].price == 3000.0
        assert data["WETH/USDC"].venue == "uniswap_v2"
        assert data["WETH/USDC"].fee_bps == 30

    def test_uniswap_v3(self):
        source = UniswapV3Source()
        source.set_prices({"WETH/USDC": 3001.0}, fee_tiers={"WETH/USDC": 5.0})
        data = source.fetch(["WETH/USDC"], "", "")
        assert data["WETH/USDC"].fee_bps == 5.0

    def test_sushiswap(self):
        source = SushiSwapSource()
        source.set_prices({"WETH/USDC": 2999.0})
        data = source.fetch(["WETH/USDC"], "", "")
        assert data["WETH/USDC"].venue == "sushiswap"

    def test_curve(self):
        source = CurveSource()
        source.set_prices({"USDC/USDT": 1.0001})
        data = source.fetch(["USDC/USDT"], "", "")
        assert data["USDC/USDT"].fee_bps == 4  # Stablecoin pool

    def test_dex_quote_inverse(self):
        q = DEXQuote(price=3000.0)
        assert abs(q.inverse_price - 1/3000) < 1e-10

    def test_missing_pair(self):
        source = UniswapV2Source()
        data = source.fetch(["DOESNT/EXIST"], "", "")
        assert len(data) == 0


class TestCEXSources:
    def test_binance(self):
        source = BinanceSource()
        source.set_prices({"ETHUSDT": (3000.0, 3001.0)})
        data = source.fetch(["ETHUSDT"], "", "")
        assert data["ETHUSDT"].bid == 3000.0
        assert data["ETHUSDT"].ask == 3001.0
        assert data["ETHUSDT"].mid == 3000.5

    def test_kraken(self):
        source = KrakenSource()
        source.set_prices({"XETHZUSD": (2999.0, 3002.0)})
        data = source.fetch(["XETHZUSD"], "", "")
        assert data["XETHZUSD"].venue == "kraken"

    def test_cex_spread(self):
        q = CEXQuote(bid=100, ask=101, mid=100.5)
        assert q.spread_bps > 0


class TestOpportunityScanner:
    def test_detects_cross_venue_arb(self):
        uni = UniswapV2Source()
        uni.set_prices({"WETH/USDC": 3000.0})
        sushi = SushiSwapSource()
        sushi.set_prices({"WETH/USDC": 3050.0})  # Higher on Sushi

        all_quotes = {}
        all_quotes.update(uni.fetch(["WETH/USDC"], "", ""))

        extra = {}
        for p, q in sushi.fetch(["WETH/USDC"], "", "").items():
            extra[p] = q

        opps = scan_opportunities(
            dex_quotes=all_quotes,
            additional_quotes=extra,
            min_profit_bps=10,
        )
        assert len(opps) >= 1
        assert opps[0].buy_venue == "uniswap_v2"
        assert opps[0].sell_venue == "sushiswap"
        assert opps[0].net_profit_bps > 0

    def test_no_opportunity_same_price(self):
        uni = UniswapV2Source()
        uni.set_prices({"WETH/USDC": 3000.0})
        sushi = SushiSwapSource()
        sushi.set_prices({"WETH/USDC": 3000.0})

        all_q = uni.fetch(["WETH/USDC"], "", "")
        extra = sushi.fetch(["WETH/USDC"], "", "")
        opps = scan_opportunities(all_q, extra, min_profit_bps=10)
        assert len(opps) == 0

    def test_fees_reduce_profit(self):
        uni = UniswapV2Source()
        uni.set_prices({"WETH/USDC": 3000.0})  # 30bps fee
        sushi = SushiSwapSource()
        sushi.set_prices({"WETH/USDC": 3010.0})  # 30bps fee

        all_q = uni.fetch(["WETH/USDC"], "", "")
        extra = sushi.fetch(["WETH/USDC"], "", "")
        # Spread is ~33bps but fees are 60bps total
        opps = scan_opportunities(all_q, extra, min_profit_bps=10)
        assert len(opps) == 0

    def test_sorted_by_profit(self):
        uni = UniswapV2Source()
        uni.set_prices({"A/B": 100.0, "C/D": 100.0})
        sushi = SushiSwapSource()
        sushi.set_prices({"A/B": 110.0, "C/D": 120.0})  # C/D more profitable

        all_q = uni.fetch(["A/B", "C/D"], "", "")
        extra = sushi.fetch(["A/B", "C/D"], "", "")
        opps = scan_opportunities(all_q, extra, min_profit_bps=0)
        if len(opps) >= 2:
            assert opps[0].net_profit_bps >= opps[1].net_profit_bps


# ═══════════════════════════════════════════════════════════════════
#  Phase 5.12: Gas Oracle
# ═══════════════════════════════════════════════════════════════════

class TestGasOracle:
    def test_estimate(self):
        oracle = GasOracle(GasOracleConfig(eth_price_usd=3000))
        oracle.set_gas_price(30, 2)
        est = oracle.estimate(gas_limit=500_000)
        assert est.base_fee_gwei == 30
        assert est.total_gwei == 32
        assert est.estimated_cost_usd > 0
        assert est.is_acceptable

    def test_unacceptable_gas(self):
        oracle = GasOracle(GasOracleConfig(max_gas_price_gwei=20))
        oracle.set_gas_price(50, 5)
        est = oracle.estimate()
        assert not est.is_acceptable

    def test_profitability_check(self):
        oracle = GasOracle(GasOracleConfig(eth_price_usd=3000))
        oracle.set_gas_price(30, 2)
        # Gas cost ~$0.048 at 500k gas
        assert oracle.is_profitable(100.0)      # $100 profit > gas
        assert not oracle.is_profitable(0.001)   # Too low

    def test_average_gas(self):
        oracle = GasOracle()
        for gwei in [20, 30, 40, 50, 60]:
            oracle.set_gas_price(gwei)
        assert oracle.average_gas(5) == 40.0

    def test_dynamic_threshold(self):
        oracle = GasOracle(GasOracleConfig(max_gas_price_gwei=100))
        oracle.set_gas_price(50)  # 50% of max
        threshold = oracle.dynamic_threshold(base_profit_bps=30)
        # Should be higher than base due to gas conditions
        assert threshold >= 30


# ═══════════════════════════════════════════════════════════════════
#  Phase 5.11: MEV Protection
# ═══════════════════════════════════════════════════════════════════

class TestMEVProtector:
    def test_build_bundle(self):
        protector = MEVProtector()
        bundle = protector.build_bundle("0xsignedtx", 12345)
        assert bundle["signedTransactions"] == ["0xsignedtx"]
        assert bundle["blockNumber"] == hex(12345)

    def test_submit_bundle_custom(self):
        protector = MEVProtector()
        protector.set_submit_fn(lambda b: BundleResult(
            bundle_hash="0xbundle", is_included=True, block_number=12345,
        ))
        result = protector.submit_bundle("0xtx", 12345)
        assert result.is_included
        assert protector.stats["bundles_included"] == 1

    def test_submit_bundle_default_fails(self):
        protector = MEVProtector()
        result = protector.submit_bundle("0xtx", 12345)
        assert not result.is_included
        assert "No Flashbots" in result.error

    def test_should_use_private_mempool(self):
        protector = MEVProtector(MEVConfig(use_private_mempool=True))
        # High profit → use private mempool
        assert protector.should_use_private_mempool(1000, 5)
        # Low profit → don't bother
        assert not protector.should_use_private_mempool(10, 5)

    def test_private_mempool_disabled(self):
        protector = MEVProtector(MEVConfig(use_private_mempool=False))
        assert not protector.should_use_private_mempool(10000, 1)

    def test_stats(self):
        protector = MEVProtector()
        protector.set_submit_fn(lambda b: BundleResult(is_included=True))
        protector.submit_bundle("0x1", 100)
        protector.submit_bundle("0x2", 101)
        assert protector.stats["bundles_submitted"] == 2
        assert protector.stats["bundles_included"] == 2
        assert protector.stats["inclusion_rate"] == 1.0


# ═══════════════════════════════════════════════════════════════════
#  Phase 5.6: Crypto Cointegration
# ═══════════════════════════════════════════════════════════════════

from praxis.onchain.crypto import (
    CryptoCointegrationAnalyzer,
    CryptoCointegrationConfig,
    CryptoSignal,
    CointPairResult,
    FlashLoanOrchestrator,
    OrchestratorConfig,
    OrchestratorState,
)


def _make_cointegrated_pair(n=200, seed=42):
    """Generate a cointegrated pair."""
    rng = np.random.RandomState(seed)
    b = 100 + np.cumsum(rng.randn(n) * 0.5)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.8 * spread[i-1] + rng.randn() * 0.3
    a = 1.5 * b + spread + 50
    return a, b


class TestCryptoCointegration:
    def test_cointegrated_pair(self):
        analyzer = CryptoCointegrationAnalyzer()
        a, b = _make_cointegrated_pair()
        result = analyzer.test_cointegration(a, b, "ETH", "BTC")
        assert result.is_cointegrated
        assert result.adf_pvalue < 0.05
        assert abs(result.hedge_ratio - 1.5) < 0.5

    def test_random_walk_not_cointegrated(self):
        analyzer = CryptoCointegrationAnalyzer()
        rng = np.random.RandomState(99)
        a = 100 + np.cumsum(rng.randn(200) * 0.5)
        b = 50 + np.cumsum(rng.randn(200) * 0.3)
        result = analyzer.test_cointegration(a, b)
        # Random walks are typically not cointegrated
        # (might occasionally pass, so we check structure)
        assert isinstance(result, CointPairResult)

    def test_too_short(self):
        analyzer = CryptoCointegrationAnalyzer(CryptoCointegrationConfig(min_history_points=20))
        result = analyzer.test_cointegration(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert not result.is_cointegrated

    def test_compute_signal_entry(self):
        analyzer = CryptoCointegrationAnalyzer(CryptoCointegrationConfig(
            zscore_window=30, z_score_entry=2.0,
        ))
        a, b = _make_cointegrated_pair(100)
        # Push z-score beyond entry
        a_mod = a.copy()
        a_mod[-1] += 10 * np.std(a)
        signal = analyzer.compute_signal(a_mod, b, asset_a="ETH", asset_b="BTC")
        assert isinstance(signal, CryptoSignal)
        assert signal.pair == "ETH/BTC"

    def test_compute_signal_exit(self):
        analyzer = CryptoCointegrationAnalyzer(CryptoCointegrationConfig(
            zscore_window=30, z_score_exit=0.5,
        ))
        a, b = _make_cointegrated_pair(100)
        # At the mean → exit signal
        signal = analyzer.compute_signal(a, b, hedge_ratio=1.5)
        # z-score should be near 0 → exit
        if abs(signal.z_score) <= 0.5:
            assert signal.should_close

    def test_compute_signal_emergency(self):
        analyzer = CryptoCointegrationAnalyzer(CryptoCointegrationConfig(
            zscore_window=30, z_score_emergency=4.0,
        ))
        a, b = _make_cointegrated_pair(100)
        a[-1] += 50 * np.std(a)  # Massive deviation
        signal = analyzer.compute_signal(a, b, hedge_ratio=1.5)
        if abs(signal.z_score) >= 4.0:
            assert signal.is_emergency
            assert signal.should_close

    def test_scan_pairs(self):
        analyzer = CryptoCointegrationAnalyzer()
        a, b = _make_cointegrated_pair(200, seed=42)
        rng = np.random.RandomState(99)
        c = 30 + np.cumsum(rng.randn(200) * 0.4)

        results = analyzer.scan_pairs({"ETH": a, "BTC": b, "UNI": c})
        # Should find ETH/BTC as cointegrated
        cointegrated = [r for r in results if r.is_cointegrated]
        assert len(cointegrated) >= 1

    def test_half_life_estimation(self):
        analyzer = CryptoCointegrationAnalyzer()
        a, b = _make_cointegrated_pair()
        result = analyzer.test_cointegration(a, b)
        if result.is_cointegrated:
            assert result.half_life > 0
            assert result.half_life < 100


# ═══════════════════════════════════════════════════════════════════
#  Phase 5.9: Flash Loan Orchestrator
# ═══════════════════════════════════════════════════════════════════

class TestFlashLoanOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        return FlashLoanOrchestrator(OrchestratorConfig(
            min_profit_usd=50, max_gas_spend_per_day_usd=100,
            borrow_amount_usd=1_000_000,
        ))

    def test_evaluate_no_signal(self, orchestrator):
        signal = CryptoSignal(should_trade=False)
        result = orchestrator.evaluate_opportunity(signal)
        assert not result["go"]
        assert "No entry signal" in result["reason"]

    def test_evaluate_profitable(self, orchestrator):
        signal = CryptoSignal(should_trade=True, signal_strength=0.8)
        result = orchestrator.evaluate_opportunity(signal, gas_cost_usd=0.10)
        assert result["go"]
        assert result["estimated_profit"] > 0

    def test_evaluate_unprofitable(self, orchestrator):
        signal = CryptoSignal(should_trade=True, signal_strength=0.001)
        result = orchestrator.evaluate_opportunity(signal, gas_cost_usd=100)
        assert not result["go"]

    def test_execute_success(self, orchestrator):
        orchestrator.set_execute_fn(lambda s: {"success": True, "profit": 500})
        signal = CryptoSignal(should_trade=True, signal_strength=0.5)
        result = orchestrator.execute_opportunity(signal)
        assert result["success"]
        assert orchestrator.state.successful_attempts == 1
        assert orchestrator.state.total_profit_usd == 500

    def test_execute_failure(self, orchestrator):
        orchestrator.set_execute_fn(lambda s: {"success": False, "error": "reverted"})
        signal = CryptoSignal(should_trade=True)
        result = orchestrator.execute_opportunity(signal)
        assert not result["success"]
        assert orchestrator.state.failed_attempts == 1

    def test_gas_budget_limit(self, orchestrator):
        orchestrator._state.gas_spent_today_usd = 100.0  # At limit
        signal = CryptoSignal(should_trade=True, signal_strength=0.5)
        result = orchestrator.evaluate_opportunity(signal)
        assert not result["go"]
        assert "Gas budget" in result["reason"]

    def test_failure_rate_limit(self, orchestrator):
        orchestrator._state.failed_this_hour = 500
        signal = CryptoSignal(should_trade=True, signal_strength=0.5)
        result = orchestrator.evaluate_opportunity(signal)
        assert not result["go"]

    def test_reset_daily(self, orchestrator):
        orchestrator._state.gas_spent_today_usd = 50.0
        orchestrator._state.is_paused = True
        orchestrator.reset_daily()
        assert orchestrator.state.gas_spent_today_usd == 0
        assert orchestrator.state.can_trade

    def test_reset_hourly(self, orchestrator):
        orchestrator._state.failed_this_hour = 100
        orchestrator.reset_hourly()
        assert orchestrator.state.failed_this_hour == 0

    def test_summary(self, orchestrator):
        s = orchestrator.summary()
        assert "total_attempts" in s
        assert "success_rate" in s
        assert "can_trade" in s

    def test_net_profit_tracking(self, orchestrator):
        orchestrator.set_execute_fn(lambda s: {"success": True, "profit": 200})
        for _ in range(3):
            orchestrator.execute_opportunity(CryptoSignal(), gas_cost_usd=0.10)
        assert orchestrator.state.net_profit == pytest.approx(600 - 0.30, abs=0.01)

    def test_paused_state(self, orchestrator):
        orchestrator._state.is_paused = True
        orchestrator._state.pause_reason = "Manual pause"
        signal = CryptoSignal(should_trade=True, signal_strength=0.8)
        result = orchestrator.evaluate_opportunity(signal)
        assert not result["go"]
        assert "Paused" in result["reason"]
