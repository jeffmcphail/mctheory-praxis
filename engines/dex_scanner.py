"""
engines/dex_scanner.py — Read-Only Cross-DEX Arbitrage Scanner

Monitors token pairs across multiple DEX venues on Arbitrum L2.
Detects momentary price differences that would be profitable
via flash loan atomic execution.

This is a READ-ONLY data collection tool. It does not execute trades.
Purpose: answer "how often do atomic arb opportunities appear, at what
size, and on which pairs?" to validate the flash loan stat arb thesis.

Usage:
    from engines.dex_scanner import DexScanner
    scanner = DexScanner(rpc_url="https://arb1.arbitrum.io/rpc")
    scanner.discover_pools()
    opps = scanner.scan_once()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN REGISTRY — Arbitrum One
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Token:
    symbol: str
    address: str
    decimals: int

# Major tokens on Arbitrum One
ARBITRUM_TOKENS = {
    "WETH":  Token("WETH",  "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1", 18),
    "USDC":  Token("USDC",  "0xaf88d065e77c8cC2239327C5EDb3A432268e5831", 6),
    "USDCe": Token("USDCe", "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8", 6),
    "USDT":  Token("USDT",  "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9", 6),
    "WBTC":  Token("WBTC",  "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f", 8),
    "ARB":   Token("ARB",   "0x912CE59144191C1204E64559FE8253a0e49E6548", 18),
    "GMX":   Token("GMX",   "0xfc5A1A6EB076a2C7aD06eD22C90d7E710E35ad0a", 18),
    "LINK":  Token("LINK",  "0xf97f4df75117a78c1A5a0DBb814Af92458539FB4", 18),
    "UNI":   Token("UNI",   "0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0", 18),
    "DAI":   Token("DAI",   "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", 18),
    "PENDLE":Token("PENDLE","0x0c880f6761F1af8d9Aa9C466984b80DAb9a8c9e8", 18),
}

# Default scan universe — highest liquidity pairs
DEFAULT_TOKENS = ["WETH", "WBTC", "USDC", "USDCe", "USDT", "ARB", "GMX", "LINK"]


# ═════════════════════════════════════════════════════════════════════════════
# DEX VENUE REGISTRY — Arbitrum One
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DexVenue:
    name: str
    factory: str
    fee_tiers: list[int]  # in hundredths of a bps (Uni V3 convention)
    # 100 = 0.01%, 500 = 0.05%, 3000 = 0.30%, 10000 = 1.00%

VENUES = {
    "uniswap_v3": DexVenue(
        "Uniswap V3",
        "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        [100, 500, 3000, 10000],
    ),
    "sushiswap_v3": DexVenue(
        "SushiSwap V3",
        "0x1af415a1EbA07a4986a52B6f2e7dE7003D82231e",
        [100, 500, 3000, 10000],
    ),
    # Pancakeswap V3 on Arbitrum
    "pancakeswap_v3": DexVenue(
        "PancakeSwap V3",
        "0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865",
        [100, 500, 2500, 10000],
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# ABI FRAGMENTS — minimal for read-only scanning
# ═════════════════════════════════════════════════════════════════════════════

FACTORY_ABI = json.loads("""[
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "uint24", "name": "fee", "type": "uint24"}
        ],
        "name": "getPool",
        "outputs": [{"internalType": "address", "name": "pool", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]""")

POOL_ABI = json.loads("""[
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
            {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
            {"internalType": "bool", "name": "unlocked", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "liquidity",
        "outputs": [{"internalType": "uint128", "name": "", "type": "uint128"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]""")

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PoolInfo:
    """A discovered DEX pool."""
    venue: str           # e.g. "uniswap_v3"
    address: str         # pool contract address
    token0: str          # token0 symbol
    token1: str          # token1 symbol
    token0_addr: str
    token1_addr: str
    token0_decimals: int
    token1_decimals: int
    fee_tier: int        # in hundredths of bps (500 = 0.05%)
    fee_pct: float       # as percentage (0.05)

@dataclass
class PriceReading:
    """A price observation from one pool."""
    timestamp: datetime
    pool: PoolInfo
    sqrt_price_x96: int
    tick: int
    price: float         # token1 per token0 (decimal-adjusted)
    liquidity: int       # current in-range liquidity

@dataclass
class ArbOpportunity:
    """A detected cross-venue arbitrage opportunity."""
    timestamp: datetime
    token0: str
    token1: str
    buy_venue: str       # cheaper venue (buy here)
    sell_venue: str      # more expensive venue (sell here)
    buy_price: float
    sell_price: float
    spread_bps: float    # price difference in basis points
    buy_fee_bps: float   # fee on buy leg
    sell_fee_bps: float  # fee on sell leg
    net_spread_bps: float  # spread - buy_fee - sell_fee
    buy_liquidity: int
    sell_liquidity: int
    estimated_profit_bps: float  # net spread - estimated gas


# ═════════════════════════════════════════════════════════════════════════════
# PRICE MATH
# ═════════════════════════════════════════════════════════════════════════════

def sqrt_price_x96_to_price(sqrt_price_x96: int,
                             token0_decimals: int,
                             token1_decimals: int) -> float:
    """
    Convert Uniswap V3 sqrtPriceX96 to human-readable price.

    sqrtPriceX96 = sqrt(token1/token0) * 2^96
    price = (sqrtPriceX96 / 2^96)^2 * 10^(token0_decimals - token1_decimals)

    Returns: token1 per token0 (e.g., USDC per WETH = ~3500)
    """
    price_raw = (sqrt_price_x96 / (2 ** 96)) ** 2
    decimal_adjustment = 10 ** (token0_decimals - token1_decimals)
    return price_raw * decimal_adjustment


# ═════════════════════════════════════════════════════════════════════════════
# SCANNER
# ═════════════════════════════════════════════════════════════════════════════

class DexScanner:
    """
    Read-only cross-DEX arbitrage scanner for Arbitrum.

    Discovers pools, reads prices, detects opportunities.
    Does NOT execute trades.
    """

    def __init__(self, rpc_url: str = "https://arb1.arbitrum.io/rpc",
                 tokens: list[str] | None = None,
                 venues: list[str] | None = None,
                 gas_cost_bps: float = 0.5,
                 flash_loan_fee_bps: float = 0.0,
                 min_spread_bps: float = 1.0):
        """
        Args:
            rpc_url: Arbitrum RPC endpoint
            tokens: Token symbols to scan (default: DEFAULT_TOKENS)
            venues: DEX venue keys to scan (default: all)
            gas_cost_bps: Estimated gas cost in bps of trade size
                          (Arbitrum ~$0.05-0.50 per tx, ~0.5 bps on $10k trade)
            flash_loan_fee_bps: Flash loan fee (0 for Balancer, 5 for Aave)
            min_spread_bps: Minimum net spread to log as opportunity
        """
        from web3 import Web3

        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.token_symbols = tokens or DEFAULT_TOKENS
        self.venue_keys = venues or list(VENUES.keys())
        self.gas_cost_bps = gas_cost_bps
        self.flash_loan_fee_bps = flash_loan_fee_bps
        self.min_spread_bps = min_spread_bps

        # Resolve tokens
        self.tokens = {}
        for sym in self.token_symbols:
            if sym in ARBITRUM_TOKENS:
                self.tokens[sym] = ARBITRUM_TOKENS[sym]
            else:
                logger.warning(f"Unknown token: {sym}")

        # Build address → symbol lookup
        self._addr_to_symbol = {
            t.address.lower(): sym for sym, t in self.tokens.items()
        }

        # Discovered pools
        self.pools: list[PoolInfo] = []

        # Price history for cointegration analysis
        self.price_history: list[dict] = []
        self.opportunity_log: list[ArbOpportunity] = []

        # Staleness detection: track previous scan's spread per venue-pair
        self._prev_spreads: dict[str, float] = {}

    def connect(self) -> bool:
        """Test RPC connection."""
        try:
            block = self.w3.eth.block_number
            chain_id = self.w3.eth.chain_id
            print(f"  Connected: chain_id={chain_id}, block={block}")
            if chain_id != 42161:
                print(f"  WARNING: Expected Arbitrum (42161), got {chain_id}")
            return True
        except Exception as e:
            print(f"  Connection failed: {e}")
            return False

    def discover_pools(self) -> int:
        """
        Discover all pools for token pairs across all venues.

        For N tokens and V venues with F fee tiers each,
        this makes N*(N-1)/2 * V * F RPC calls.
        """
        from web3 import Web3

        self.pools = []
        token_list = list(self.tokens.values())
        n_checked = 0
        n_found = 0

        for venue_key in self.venue_keys:
            venue = VENUES[venue_key]
            factory = self.w3.eth.contract(
                address=Web3.to_checksum_address(venue.factory),
                abi=FACTORY_ABI,
            )

            for i in range(len(token_list)):
                for j in range(i + 1, len(token_list)):
                    t0 = token_list[i]
                    t1 = token_list[j]

                    for fee in venue.fee_tiers:
                        n_checked += 1
                        try:
                            pool_addr = factory.functions.getPool(
                                Web3.to_checksum_address(t0.address),
                                Web3.to_checksum_address(t1.address),
                                fee,
                            ).call()

                            if pool_addr == ZERO_ADDRESS:
                                continue

                            # Determine token ordering (pool's token0/token1)
                            pool_contract = self.w3.eth.contract(
                                address=Web3.to_checksum_address(pool_addr),
                                abi=POOL_ABI,
                            )
                            actual_t0 = pool_contract.functions.token0().call().lower()

                            if actual_t0 == t0.address.lower():
                                tok0, tok1 = t0, t1
                            else:
                                tok0, tok1 = t1, t0

                            pool = PoolInfo(
                                venue=venue_key,
                                address=pool_addr,
                                token0=tok0.symbol,
                                token1=tok1.symbol,
                                token0_addr=tok0.address,
                                token1_addr=tok1.address,
                                token0_decimals=tok0.decimals,
                                token1_decimals=tok1.decimals,
                                fee_tier=fee,
                                fee_pct=fee / 10000,
                            )
                            self.pools.append(pool)
                            n_found += 1

                        except Exception as e:
                            logger.debug(f"Pool query failed: {venue_key} "
                                        f"{t0.symbol}/{t1.symbol} fee={fee}: {e}")

                        # Rate limiting
                        if n_checked % 50 == 0:
                            time.sleep(0.5)

        print(f"  Pool discovery: {n_found} pools found "
              f"({n_checked} checked across {len(self.venue_keys)} venues)")

        # Group by canonical pair for summary
        pair_venues = {}
        for p in self.pools:
            tokens_sorted = sorted([p.token0, p.token1])
            pair_key = f"{tokens_sorted[0]}/{tokens_sorted[1]}"
            if pair_key not in pair_venues:
                pair_venues[pair_key] = set()
            pair_venues[pair_key].add(p.venue)

        multi_venue = {k: v for k, v in pair_venues.items() if len(v) >= 2}
        print(f"  Pairs on 2+ venues (arb-eligible): {len(multi_venue)}")
        for pair, venues in sorted(multi_venue.items()):
            print(f"    {pair}: {', '.join(sorted(venues))}")

        return n_found

    def read_pool_price(self, pool: PoolInfo) -> PriceReading | None:
        """Read current price from a single pool via slot0."""
        from web3 import Web3

        try:
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(pool.address),
                abi=POOL_ABI,
            )
            slot0 = contract.functions.slot0().call()
            sqrt_price = slot0[0]
            tick = slot0[1]

            if sqrt_price == 0:
                return None

            price = sqrt_price_x96_to_price(
                sqrt_price, pool.token0_decimals, pool.token1_decimals
            )

            liq = contract.functions.liquidity().call()

            # Filter: skip pools with zero in-range liquidity (dead/empty pools)
            if liq == 0:
                return None

            # Sanity: price must be a reasonable positive finite number
            if not (np.isfinite(price) and 1e-12 < price < 1e12):
                return None

            return PriceReading(
                timestamp=datetime.now(timezone.utc),
                pool=pool,
                sqrt_price_x96=sqrt_price,
                tick=tick,
                price=price,
                liquidity=liq,
            )
        except Exception as e:
            logger.debug(f"Price read failed {pool.venue} "
                        f"{pool.token0}/{pool.token1}: {e}")
            return None

    def scan_once(self) -> list[ArbOpportunity]:
        """
        Read all pool prices and detect cross-venue opportunities.

        Returns list of opportunities where net spread > min_spread_bps.
        """
        now = datetime.now(timezone.utc)

        # Read all prices
        readings: list[PriceReading] = []
        for pool in self.pools:
            reading = self.read_pool_price(pool)
            if reading is not None:
                readings.append(reading)

        if not readings:
            return []

        # Store in history
        for r in readings:
            self.price_history.append({
                "timestamp": r.timestamp.isoformat(),
                "venue": r.pool.venue,
                "token0": r.pool.token0,
                "token1": r.pool.token1,
                "fee_tier": r.pool.fee_tier,
                "price": r.price,
                "liquidity": r.liquidity,
                "tick": r.tick,
            })

        # Group readings by CANONICAL token pair (sorted alphabetically)
        # This ensures pools with opposite token ordering are compared correctly
        pair_readings: dict[str, list[tuple[PriceReading, float]]] = {}
        for r in readings:
            # Canonical key: alphabetically sorted
            tokens_sorted = sorted([r.pool.token0, r.pool.token1])
            key = f"{tokens_sorted[0]}/{tokens_sorted[1]}"

            # Normalize price to canonical direction: always tokens_sorted[1] per tokens_sorted[0]
            if r.pool.token0 == tokens_sorted[0]:
                # Pool ordering matches canonical — price is already correct
                canonical_price = r.price
            else:
                # Pool ordering is inverted — flip the price
                canonical_price = 1.0 / r.price if r.price > 0 else 0.0

            if key not in pair_readings:
                pair_readings[key] = []
            pair_readings[key].append((r, canonical_price))

        # Compare prices across venues for each pair
        opportunities = []
        current_spreads: dict[str, float] = {}  # for staleness detection
        for pair_key, pair_r in pair_readings.items():
            if len(pair_r) < 2:
                continue

            # Compare every pair of readings
            for i in range(len(pair_r)):
                for j in range(i + 1, len(pair_r)):
                    r1, cp1 = pair_r[i]
                    r2, cp2 = pair_r[j]

                    # Skip same-venue comparisons (different fee tiers)
                    if r1.pool.venue == r2.pool.venue:
                        continue

                    if cp1 <= 0 or cp2 <= 0:
                        continue

                    # Determine buy/sell using canonical prices
                    if cp1 < cp2:
                        buy_r, buy_cp = r1, cp1
                        sell_r, sell_cp = r2, cp2
                    else:
                        buy_r, buy_cp = r2, cp2
                        sell_r, sell_cp = r1, cp1

                    spread_bps = (sell_cp - buy_cp) / buy_cp * 10000

                    # Sanity: skip obviously broken comparisons
                    if spread_bps > 500:  # >5% spread is almost certainly a data issue
                        continue

                    buy_fee_bps = buy_r.pool.fee_tier / 100
                    sell_fee_bps = sell_r.pool.fee_tier / 100
                    net_spread = spread_bps - buy_fee_bps - sell_fee_bps
                    est_profit = net_spread - self.gas_cost_bps - self.flash_loan_fee_bps

                    if net_spread >= self.min_spread_bps:
                        # Staleness detection: if spread is identical to previous
                        # scan (within 0.01 bps), this is a dead pool — skip it
                        stale_key = f"{pair_key}:{buy_r.pool.venue}:{sell_r.pool.venue}"
                        prev = self._prev_spreads.get(stale_key)
                        is_stale = prev is not None and abs(spread_bps - prev) < 0.01
                        current_spreads[stale_key] = spread_bps

                        if is_stale:
                            continue  # same spread as last scan = dead pool

                        tokens = pair_key.split("/")
                        opp = ArbOpportunity(
                            timestamp=now,
                            token0=tokens[0],
                            token1=tokens[1],
                            buy_venue=buy_r.pool.venue,
                            sell_venue=sell_r.pool.venue,
                            buy_price=buy_cp,
                            sell_price=sell_cp,
                            spread_bps=spread_bps,
                            buy_fee_bps=buy_fee_bps,
                            sell_fee_bps=sell_fee_bps,
                            net_spread_bps=net_spread,
                            buy_liquidity=buy_r.liquidity,
                            sell_liquidity=sell_r.liquidity,
                            estimated_profit_bps=est_profit,
                        )
                        opportunities.append(opp)
                        self.opportunity_log.append(opp)

        # Update staleness tracker for next scan
        self._prev_spreads = current_spreads

        return opportunities

    def scan_loop(self, interval_seconds: float = 10.0,
                  duration_minutes: float = 60.0,
                  output_dir: str | Path = "output/dex_scanner"):
        """
        Continuous scanning loop with periodic output.

        Args:
            interval_seconds: Time between scans
            duration_minutes: Total scan duration
            output_dir: Where to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        end_time = time.time() + duration_minutes * 60
        scan_count = 0
        total_opps = 0

        print(f"\n{'='*70}")
        print(f"DEX SCANNER — Continuous Mode")
        print(f"  Pools: {len(self.pools)}")
        print(f"  Interval: {interval_seconds}s")
        print(f"  Duration: {duration_minutes} min")
        print(f"  Min spread: {self.min_spread_bps} bps")
        print(f"  Gas estimate: {self.gas_cost_bps} bps")
        print(f"{'='*70}\n")

        try:
            while time.time() < end_time:
                t0 = time.time()
                opps = self.scan_once()
                elapsed = time.time() - t0
                scan_count += 1
                total_opps += len(opps)

                if opps:
                    for opp in opps:
                        print(f"  [{opp.timestamp.strftime('%H:%M:%S')}] "
                              f"{opp.token0}/{opp.token1}: "
                              f"{opp.buy_venue}→{opp.sell_venue} "
                              f"spread={opp.spread_bps:+.1f}bps "
                              f"net={opp.net_spread_bps:+.1f}bps "
                              f"est_profit={opp.estimated_profit_bps:+.1f}bps")

                if scan_count % 30 == 0:
                    print(f"  --- Scan #{scan_count}: {total_opps} opps total, "
                          f"{len(self.price_history)} price readings ---")

                # Save periodically (every 5 minutes)
                if scan_count % (300 // max(int(interval_seconds), 1)) == 0:
                    self._save_results(output_dir)

                # Wait for next scan
                wait = max(0, interval_seconds - elapsed)
                if wait > 0:
                    time.sleep(wait)

        except KeyboardInterrupt:
            print("\n  Scanner stopped by user.")

        # Final save
        self._save_results(output_dir)
        self._print_summary()

    def _save_results(self, output_dir: Path):
        """Save price history and opportunity log to disk."""
        if self.price_history:
            ph = pd.DataFrame(self.price_history)
            # Cast liquidity to float — uint128 overflows pyarrow int64
            if "liquidity" in ph.columns:
                ph["liquidity"] = ph["liquidity"].astype(float)
            ph.to_parquet(output_dir / "price_history.parquet", index=False)

        if self.opportunity_log:
            rows = [{
                "timestamp": o.timestamp.isoformat(),
                "token0": o.token0, "token1": o.token1,
                "buy_venue": o.buy_venue, "sell_venue": o.sell_venue,
                "buy_price": o.buy_price, "sell_price": o.sell_price,
                "spread_bps": o.spread_bps,
                "buy_fee_bps": o.buy_fee_bps, "sell_fee_bps": o.sell_fee_bps,
                "net_spread_bps": o.net_spread_bps,
                "estimated_profit_bps": o.estimated_profit_bps,
                "buy_liquidity": float(o.buy_liquidity),
                "sell_liquidity": float(o.sell_liquidity),
            } for o in self.opportunity_log]
            pd.DataFrame(rows).to_csv(output_dir / "opportunities.csv", index=False)

    def _print_summary(self):
        """Print summary statistics."""
        print(f"\n{'='*70}")
        print("SCAN SUMMARY")
        print(f"{'='*70}")
        print(f"  Price readings: {len(self.price_history)}")
        print(f"  Opportunities:  {len(self.opportunity_log)}")

        if not self.opportunity_log:
            print("  No opportunities detected.")
            return

        df = pd.DataFrame([{
            "token0": o.token0, "token1": o.token1,
            "buy_venue": o.buy_venue, "sell_venue": o.sell_venue,
            "spread_bps": o.spread_bps,
            "net_spread_bps": o.net_spread_bps,
            "estimated_profit_bps": o.estimated_profit_bps,
        } for o in self.opportunity_log])

        print(f"\n  By pair:")
        for pair, grp in df.groupby(["token0", "token1"]):
            print(f"    {pair[0]}/{pair[1]}: {len(grp)} opps, "
                  f"avg_spread={grp['spread_bps'].mean():.1f}bps, "
                  f"avg_net={grp['net_spread_bps'].mean():.1f}bps")

        print(f"\n  By venue pair:")
        for vp, grp in df.groupby(["buy_venue", "sell_venue"]):
            print(f"    {vp[0]}→{vp[1]}: {len(grp)} opps, "
                  f"avg_spread={grp['spread_bps'].mean():.1f}bps")

        profitable = df[df["estimated_profit_bps"] > 0]
        print(f"\n  Estimated profitable (after gas+fees): {len(profitable)}/{len(df)}")
        if not profitable.empty:
            print(f"    Avg profit: {profitable['estimated_profit_bps'].mean():.1f} bps")
            print(f"    Max profit: {profitable['estimated_profit_bps'].max():.1f} bps")


# ═════════════════════════════════════════════════════════════════════════════
# COINTEGRATION ANALYSIS ON COLLECTED DATA
# ═════════════════════════════════════════════════════════════════════════════

def analyze_cointegration(price_history_path: str | Path,
                          min_observations: int = 500) -> pd.DataFrame:
    """
    Run cointegration analysis on collected price history.

    Identifies which token pairs across venues have mean-reverting
    spread relationships — these are the best candidates for
    flash loan stat arb.

    Args:
        price_history_path: Path to price_history.parquet from scanner
        min_observations: Minimum price readings per pool to analyze

    Returns:
        DataFrame with cointegration statistics per pair-venue combo
    """
    from statsmodels.tsa.stattools import adfuller

    df = pd.read_parquet(price_history_path)
    if df.empty:
        print("No data to analyze.")
        return pd.DataFrame()

    # Group by (token pair, venue+fee) → time series
    df["pool_key"] = df["venue"] + "_" + df["fee_tier"].astype(str)
    df["pair"] = df["token0"] + "/" + df["token1"]

    results = []

    for pair in df["pair"].unique():
        pair_df = df[df["pair"] == pair]
        pool_keys = pair_df["pool_key"].unique()

        if len(pool_keys) < 2:
            continue

        # For each pair of pools, compute spread and test stationarity
        for i in range(len(pool_keys)):
            for j in range(i + 1, len(pool_keys)):
                pk1, pk2 = pool_keys[i], pool_keys[j]
                s1 = pair_df[pair_df["pool_key"] == pk1].set_index("timestamp")["price"]
                s2 = pair_df[pair_df["pool_key"] == pk2].set_index("timestamp")["price"]

                # Align on timestamp
                merged = pd.concat([s1.rename("p1"), s2.rename("p2")], axis=1).dropna()
                if len(merged) < min_observations:
                    continue

                # Spread = log(p1) - log(p2)
                spread = np.log(merged["p1"]) - np.log(merged["p2"])

                # ADF test
                try:
                    adf_result = adfuller(spread.values, maxlag=20)
                    adf_t, adf_p = adf_result[0], adf_result[1]
                except Exception:
                    adf_t, adf_p = 0.0, 1.0

                # Half-life of mean reversion
                spread_arr = spread.values
                spread_lag = spread_arr[:-1]
                spread_diff = np.diff(spread_arr)
                if len(spread_lag) > 10 and np.std(spread_lag) > 1e-10:
                    beta = np.polyfit(spread_lag, spread_diff, 1)[0]
                    half_life = -np.log(2) / beta if beta < 0 else np.inf
                else:
                    half_life = np.inf

                results.append({
                    "pair": pair,
                    "pool1": pk1,
                    "pool2": pk2,
                    "n_obs": len(merged),
                    "spread_mean": float(spread.mean()),
                    "spread_std": float(spread.std()),
                    "adf_t": float(adf_t),
                    "adf_p": float(adf_p),
                    "half_life": float(half_life),
                    "is_cointegrated": adf_p < 0.05,
                })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("adf_p")
        print(f"\nCointegration Analysis: {len(result_df)} pair-venue combos tested")
        coint = result_df[result_df["is_cointegrated"]]
        print(f"  Cointegrated (p < 0.05): {len(coint)}")
        if not coint.empty:
            for _, row in coint.iterrows():
                print(f"    {row['pair']}: {row['pool1']} ↔ {row['pool2']} "
                      f"ADF t={row['adf_t']:.2f} p={row['adf_p']:.4f} "
                      f"HL={row['half_life']:.1f} obs")

    return result_df
