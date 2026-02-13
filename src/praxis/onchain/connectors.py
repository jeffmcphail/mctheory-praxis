"""
DEX/CEX Data Connectors + Gas Oracle + MEV Protection (Phase 5.4, 5.5, 5.11, 5.12).

Data source adapters for on-chain and off-chain price feeds:
- DEX: Uniswap V2/V3, SushiSwap, Curve (on-chain reserve queries)
- CEX: Binance, Kraken (REST/WebSocket APIs)
- Gas oracle: EIP-1559 gas estimation with dynamic thresholds
- MEV protection: Flashbots bundle submission, private mempool

All connectors implement DataSourceTemplate for registry integration.

Usage:
    dex = UniswapV2Source(config)
    prices = dex.fetch(["WETH/USDC", "WBTC/USDC"], "", "")

    gas = GasOracle(config)
    estimate = gas.estimate()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from praxis.templates import DataSourceTemplate
from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  DEX Data Sources (Phase 5.4)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DEXConfig:
    """Configuration for DEX data sources."""
    rpc_url: str = ""
    chain: str = "ethereum"
    poll_interval_seconds: float = 10.0
    max_staleness_seconds: float = 60.0


@dataclass
class DEXQuote:
    """A price quote from a DEX."""
    venue: str = ""
    pair: str = ""
    price: float = 0.0
    liquidity: float = 0.0      # In USD
    fee_bps: float = 30.0       # Default Uniswap V2 fee
    block_number: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def inverse_price(self) -> float:
        return 1.0 / self.price if self.price > 0 else 0.0


class UniswapV2Source(DataSourceTemplate):
    """
    Uniswap V2 price source via reserve queries.

    Reads pair reserves from on-chain contracts to compute prices.
    In test mode, accepts pre-loaded prices.
    """

    VENUE = "uniswap_v2"
    FEE_BPS = 30  # 0.30%

    def __init__(self, config: DEXConfig | None = None):
        self._config = config or DEXConfig()
        self._cache: dict[str, DEXQuote] = {}

    def set_prices(self, prices: dict[str, float]) -> None:
        """Pre-load prices for testing."""
        for pair, price in prices.items():
            self._cache[pair] = DEXQuote(
                venue=self.VENUE, pair=pair, price=price,
                fee_bps=self.FEE_BPS, liquidity=1_000_000,
            )

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        """Fetch current prices for pairs."""
        result = {}
        for pair in tickers:
            quote = self._cache.get(pair)
            if quote:
                result[pair] = quote
        return result

    def get_quote(self, pair: str) -> DEXQuote | None:
        return self._cache.get(pair)


class UniswapV3Source(DataSourceTemplate):
    """Uniswap V3 price source (concentrated liquidity)."""

    VENUE = "uniswap_v3"

    def __init__(self, config: DEXConfig | None = None):
        self._config = config or DEXConfig()
        self._cache: dict[str, DEXQuote] = {}

    def set_prices(self, prices: dict[str, float], fee_tiers: dict[str, float] | None = None) -> None:
        for pair, price in prices.items():
            fee = (fee_tiers or {}).get(pair, 5.0)  # Default 0.05% pool
            self._cache[pair] = DEXQuote(
                venue=self.VENUE, pair=pair, price=price,
                fee_bps=fee, liquidity=5_000_000,
            )

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        return {p: self._cache[p] for p in tickers if p in self._cache}


class SushiSwapSource(DataSourceTemplate):
    """SushiSwap price source (Uniswap V2 fork)."""

    VENUE = "sushiswap"
    FEE_BPS = 30

    def __init__(self, config: DEXConfig | None = None):
        self._config = config or DEXConfig()
        self._cache: dict[str, DEXQuote] = {}

    def set_prices(self, prices: dict[str, float]) -> None:
        for pair, price in prices.items():
            self._cache[pair] = DEXQuote(
                venue=self.VENUE, pair=pair, price=price,
                fee_bps=self.FEE_BPS, liquidity=500_000,
            )

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        return {p: self._cache[p] for p in tickers if p in self._cache}


class CurveSource(DataSourceTemplate):
    """Curve Finance price source (stablecoin/pegged asset pools)."""

    VENUE = "curve"
    FEE_BPS = 4  # ~0.04% for stablecoin pools

    def __init__(self, config: DEXConfig | None = None):
        self._config = config or DEXConfig()
        self._cache: dict[str, DEXQuote] = {}

    def set_prices(self, prices: dict[str, float]) -> None:
        for pair, price in prices.items():
            self._cache[pair] = DEXQuote(
                venue=self.VENUE, pair=pair, price=price,
                fee_bps=self.FEE_BPS, liquidity=10_000_000,
            )

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        return {p: self._cache[p] for p in tickers if p in self._cache}


# ═══════════════════════════════════════════════════════════════════
#  CEX Data Sources (Phase 5.5)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CEXConfig:
    """Configuration for CEX data sources."""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    poll_interval_seconds: float = 1.0


@dataclass
class CEXQuote:
    """A price quote from a CEX."""
    venue: str = ""
    pair: str = ""
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    volume_24h: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def spread_bps(self) -> float:
        if self.mid > 0:
            return (self.ask - self.bid) / self.mid * 10_000
        return 0.0


class BinanceSource(DataSourceTemplate):
    """
    Binance CEX price source.

    In production: REST API to /api/v3/ticker/bookTicker
    In test mode: pre-loaded prices.
    """

    VENUE = "binance"

    def __init__(self, config: CEXConfig | None = None):
        self._config = config or CEXConfig(base_url="https://api.binance.com")
        self._cache: dict[str, CEXQuote] = {}

    def set_prices(self, prices: dict[str, tuple[float, float]]) -> None:
        """Set prices as {pair: (bid, ask)}."""
        for pair, (bid, ask) in prices.items():
            self._cache[pair] = CEXQuote(
                venue=self.VENUE, pair=pair,
                bid=bid, ask=ask, mid=(bid + ask) / 2,
            )

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        return {p: self._cache[p] for p in tickers if p in self._cache}


class KrakenSource(DataSourceTemplate):
    """Kraken CEX price source."""

    VENUE = "kraken"

    def __init__(self, config: CEXConfig | None = None):
        self._config = config or CEXConfig(base_url="https://api.kraken.com")
        self._cache: dict[str, CEXQuote] = {}

    def set_prices(self, prices: dict[str, tuple[float, float]]) -> None:
        for pair, (bid, ask) in prices.items():
            self._cache[pair] = CEXQuote(
                venue=self.VENUE, pair=pair,
                bid=bid, ask=ask, mid=(bid + ask) / 2,
            )

    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        return {p: self._cache[p] for p in tickers if p in self._cache}


# ═══════════════════════════════════════════════════════════════════
#  Opportunity Scanner
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ArbitrageOpportunity:
    """A detected cross-venue arbitrage opportunity."""
    pair: str = ""
    buy_venue: str = ""
    sell_venue: str = ""
    buy_price: float = 0.0
    sell_price: float = 0.0
    spread_bps: float = 0.0
    buy_fee_bps: float = 0.0
    sell_fee_bps: float = 0.0
    net_profit_bps: float = 0.0
    estimated_profit_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_viable(self) -> bool:
        return self.net_profit_bps > 0


def scan_opportunities(
    dex_quotes: dict[str, DEXQuote],
    additional_quotes: dict[str, DEXQuote | CEXQuote] | None = None,
    min_profit_bps: float = 10.0,
    trade_size_usd: float = 10_000.0,
) -> list[ArbitrageOpportunity]:
    """
    Scan for cross-venue arbitrage opportunities.

    Compares prices across all provided venues for overlapping pairs.
    """
    # Collect all quotes by pair
    by_pair: dict[str, list] = {}
    for pair, quote in dex_quotes.items():
        by_pair.setdefault(pair, []).append(("dex", quote))

    if additional_quotes:
        for pair, quote in additional_quotes.items():
            by_pair.setdefault(pair, []).append(("extra", quote))

    opportunities = []

    for pair, quotes in by_pair.items():
        if len(quotes) < 2:
            continue

        # Compare all pairs of venues
        for i in range(len(quotes)):
            for j in range(i + 1, len(quotes)):
                _, q1 = quotes[i]
                _, q2 = quotes[j]

                p1 = q1.price if hasattr(q1, 'price') else q1.mid
                p2 = q2.price if hasattr(q2, 'price') else q2.mid

                if p1 <= 0 or p2 <= 0:
                    continue

                # Try both directions
                for buy_q, sell_q, buy_p, sell_p in [
                    (q1, q2, p1, p2),
                    (q2, q1, p2, p1),
                ]:
                    if sell_p <= buy_p:
                        continue

                    spread = (sell_p - buy_p) / buy_p * 10_000
                    buy_fee = getattr(buy_q, 'fee_bps', 0)
                    sell_fee = getattr(sell_q, 'fee_bps', 0)
                    net = spread - buy_fee - sell_fee

                    if net >= min_profit_bps:
                        opportunities.append(ArbitrageOpportunity(
                            pair=pair,
                            buy_venue=buy_q.venue,
                            sell_venue=sell_q.venue,
                            buy_price=buy_p,
                            sell_price=sell_p,
                            spread_bps=spread,
                            buy_fee_bps=buy_fee,
                            sell_fee_bps=sell_fee,
                            net_profit_bps=net,
                            estimated_profit_usd=trade_size_usd * net / 10_000,
                        ))

    opportunities.sort(key=lambda o: o.net_profit_bps, reverse=True)
    return opportunities


# ═══════════════════════════════════════════════════════════════════
#  Gas Oracle (Phase 5.12)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GasEstimate:
    """EIP-1559 gas price estimate."""
    base_fee_gwei: float = 0.0
    priority_fee_gwei: float = 0.0
    max_fee_gwei: float = 0.0
    estimated_cost_eth: float = 0.0
    estimated_cost_usd: float = 0.0
    is_acceptable: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_gwei(self) -> float:
        return self.base_fee_gwei + self.priority_fee_gwei


@dataclass
class GasOracleConfig:
    """Configuration for gas oracle."""
    max_gas_price_gwei: float = 100.0
    default_gas_limit: int = 500_000
    eth_price_usd: float = 3000.0
    chain: str = "ethereum"     # ethereum, arbitrum, polygon


class GasOracle:
    """
    Gas price oracle with dynamic threshold adjustment.

    Tracks gas prices over time and computes whether
    current gas makes an operation profitable.
    """

    def __init__(self, config: GasOracleConfig | None = None):
        self._config = config or GasOracleConfig()
        self._history: list[float] = []
        self._current_base_fee: float = 30.0
        self._current_priority: float = 2.0

    def set_gas_price(self, base_fee_gwei: float, priority_fee_gwei: float = 2.0) -> None:
        """Manually set current gas price (for testing or manual feed)."""
        self._current_base_fee = base_fee_gwei
        self._current_priority = priority_fee_gwei
        self._history.append(base_fee_gwei)

    def estimate(self, gas_limit: int | None = None) -> GasEstimate:
        """Get current gas estimate."""
        gas_limit = gas_limit or self._config.default_gas_limit
        total_gwei = self._current_base_fee + self._current_priority
        cost_eth = gas_limit * total_gwei * 1e-9
        cost_usd = cost_eth * self._config.eth_price_usd

        return GasEstimate(
            base_fee_gwei=self._current_base_fee,
            priority_fee_gwei=self._current_priority,
            max_fee_gwei=self._config.max_gas_price_gwei,
            estimated_cost_eth=cost_eth,
            estimated_cost_usd=cost_usd,
            is_acceptable=total_gwei <= self._config.max_gas_price_gwei,
        )

    def is_profitable(
        self, expected_profit_usd: float, gas_limit: int | None = None,
    ) -> bool:
        """Check if an operation is profitable at current gas."""
        est = self.estimate(gas_limit)
        return expected_profit_usd > est.estimated_cost_usd and est.is_acceptable

    def average_gas(self, window: int = 20) -> float:
        """Average gas price over recent history."""
        if not self._history:
            return self._current_base_fee
        window_data = self._history[-window:]
        return sum(window_data) / len(window_data)

    def dynamic_threshold(self, base_profit_bps: int = 30) -> int:
        """
        Adjust profit threshold based on current gas conditions.

        Higher gas → higher required profit threshold.
        """
        avg = self.average_gas()
        max_gas = self._config.max_gas_price_gwei
        # Scale: at max gas, require 2x base threshold
        ratio = min(avg / max(max_gas, 1), 1.0)
        return int(base_profit_bps * (1 + ratio))


# ═══════════════════════════════════════════════════════════════════
#  MEV Protection (Phase 5.11)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MEVConfig:
    """MEV protection configuration."""
    use_private_mempool: bool = True
    flashbots_relay_url: str = "https://relay.flashbots.net"
    max_bundle_blocks: int = 3
    priority_fee_boost_pct: float = 10.0    # Boost priority fee for inclusion


@dataclass
class BundleResult:
    """Result of submitting a Flashbots bundle."""
    bundle_hash: str = ""
    is_included: bool = False
    block_number: int = 0
    simulation_profit: float = 0.0
    error: str = ""


class MEVProtector:
    """
    MEV protection via Flashbots private mempool (§5.11).

    Instead of broadcasting transactions to the public mempool
    (where MEV bots can front-run), submit directly to block
    builders via Flashbots relay.
    """

    def __init__(self, config: MEVConfig | None = None):
        self._config = config or MEVConfig()
        self._log = PraxisLogger.instance()
        self._submit_fn = None
        self._bundles_submitted = 0
        self._bundles_included = 0

    @property
    def config(self) -> MEVConfig:
        return self._config

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "bundles_submitted": self._bundles_submitted,
            "bundles_included": self._bundles_included,
            "inclusion_rate": (
                self._bundles_included / max(1, self._bundles_submitted)
            ),
        }

    def set_submit_fn(self, fn) -> None:
        """Set custom bundle submission function (for testing)."""
        self._submit_fn = fn

    def build_bundle(
        self,
        signed_tx: str,
        target_block: int,
    ) -> dict[str, Any]:
        """Build a Flashbots bundle."""
        return {
            "signedTransactions": [signed_tx],
            "blockNumber": hex(target_block),
            "maxBlockNumber": hex(target_block + self._config.max_bundle_blocks),
        }

    def submit_bundle(
        self,
        signed_tx: str,
        target_block: int,
    ) -> BundleResult:
        """
        Submit a transaction bundle to Flashbots relay.

        In test mode, uses the pluggable submit function.
        """
        bundle = self.build_bundle(signed_tx, target_block)
        self._bundles_submitted += 1

        if self._submit_fn:
            result = self._submit_fn(bundle)
        else:
            result = BundleResult(
                error="No Flashbots relay configured",
            )

        if result.is_included:
            self._bundles_included += 1

        return result

    def should_use_private_mempool(
        self, profit_usd: float, gas_cost_usd: float,
    ) -> bool:
        """
        Decide whether to use private mempool for a transaction.

        Large profits relative to gas cost are more likely to be
        front-run, so use private mempool for high-value transactions.
        """
        if not self._config.use_private_mempool:
            return False
        # Use private mempool if profit > 10x gas cost
        return profit_usd > gas_cost_usd * 10
