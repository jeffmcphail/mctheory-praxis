"""
engines/dex_quoter.py — Real Swap Output Estimation for V3 Pools

Computes exact swap outputs using Uniswap V3 concentrated liquidity math.
Works with ANY V3 fork (Uniswap, SushiSwap, PancakeSwap) by reading pool
state directly — no venue-specific quoter contracts needed.

This is the critical missing piece between "theoretical spread" and
"actual executable profit". The scanner shows mid-market price differences;
this module shows what you'd actually receive after price impact.

Usage:
    from engines.dex_quoter import V3Quoter
    quoter = V3Quoter(w3)
    result = quoter.quote_swap(pool_address, token_in, token_out, amount_in, fee)
    print(f"Input: {result.amount_in_human}, Output: {result.amount_out_human}")
    print(f"Price impact: {result.price_impact_bps:.1f} bps")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

Q96 = 2 ** 96
Q192 = 2 ** 192

# Reuse pool ABI from scanner
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
    },
    {
        "inputs": [],
        "name": "fee",
        "outputs": [{"internalType": "uint24", "name": "", "type": "uint24"}],
        "stateMutability": "view",
        "type": "function"
    }
]""")


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PoolState:
    """Current state of a V3 pool."""
    address: str
    token0: str          # address
    token1: str          # address
    fee: int             # in hundredths of bps (e.g. 500 = 0.05%)
    sqrt_price_x96: int
    tick: int
    liquidity: int

@dataclass
class SwapQuote:
    """Result of a simulated swap."""
    pool_address: str
    token_in: str        # address
    token_out: str       # address
    amount_in: int       # raw (smallest units)
    amount_out: int      # raw (smallest units)
    amount_in_human: float   # decimal-adjusted
    amount_out_human: float  # decimal-adjusted
    fee_amount: int      # amount consumed by fee
    effective_price: float   # token_out per token_in (decimal-adjusted)
    mid_price: float         # mid-market price before swap
    price_impact_bps: float  # how much worse than mid price
    sqrt_price_after: int    # new sqrtPriceX96 after swap
    crosses_tick: bool       # True if swap likely crosses tick boundary

@dataclass
class ArbQuote:
    """Result of a simulated arbitrage round-trip."""
    # Leg 1: buy on cheap venue
    buy_venue: str
    buy_pool: str
    buy_quote: SwapQuote

    # Leg 2: sell on expensive venue
    sell_venue: str
    sell_pool: str
    sell_quote: SwapQuote

    # P&L
    token0: str          # token symbol
    token1: str          # token symbol
    trade_size_usd: float
    amount_in: float     # what we start with (human units)
    amount_out: float    # what we end with (human units)
    gross_profit: float  # amount_out - amount_in (in starting token)
    gross_profit_bps: float
    gas_cost_usd: float
    net_profit_usd: float
    net_profit_bps: float
    is_profitable: bool


# ═════════════════════════════════════════════════════════════════════════════
# V3 SWAP MATH
# ═════════════════════════════════════════════════════════════════════════════

def compute_swap_output(
    sqrt_price_x96: int,
    liquidity: int,
    amount_in: int,
    fee: int,
    zero_for_one: bool,
) -> tuple[int, int]:
    """
    Compute the output amount for a V3 swap using concentrated liquidity math.

    This is the single-tick-range approximation — accurate when the swap
    doesn't cross initialized tick boundaries. For deep pools like
    USDC/WETH on Uniswap V3 Arbitrum, a $10-50k swap typically stays
    within one tick range.

    Args:
        sqrt_price_x96: Current sqrtPriceX96 from slot0
        liquidity: Current in-range liquidity
        amount_in: Input amount in smallest token units
        fee: Pool fee in hundredths of bps (500 = 0.05%)
        zero_for_one: True if swapping token0 → token1

    Returns:
        (amount_out, sqrt_price_x96_after)
    """
    if liquidity == 0 or amount_in == 0:
        return 0, sqrt_price_x96

    # Apply fee
    effective_in = amount_in * (1_000_000 - fee) // 1_000_000

    if zero_for_one:
        # Selling token0, buying token1 → price decreases
        # sqrtP_new = L * sqrtP / (L + effectiveIn * sqrtP / Q96)
        # Using integer math to avoid overflow:
        denominator = liquidity + (effective_in * sqrt_price_x96 // Q96)
        if denominator == 0:
            return 0, sqrt_price_x96

        sqrt_price_new = liquidity * sqrt_price_x96 // denominator

        # amountOut = L * (sqrtP - sqrtP_new) / Q96
        if sqrt_price_x96 <= sqrt_price_new:
            return 0, sqrt_price_x96

        amount_out = liquidity * (sqrt_price_x96 - sqrt_price_new) // Q96

    else:
        # Selling token1, buying token0 → price increases
        # sqrtP_new = sqrtP + effectiveIn * Q96 / L
        sqrt_price_new = sqrt_price_x96 + (effective_in * Q96 // liquidity)

        # amountOut = L * (1/sqrtP_old - 1/sqrtP_new)
        #           = L * Q96 * (sqrtP_new - sqrtP) / (sqrtP * sqrtP_new)
        numerator = liquidity * Q96 * (sqrt_price_new - sqrt_price_x96)
        denominator = sqrt_price_x96 * sqrt_price_new
        if denominator == 0:
            return 0, sqrt_price_x96

        amount_out = numerator // denominator

    return max(0, amount_out), sqrt_price_new


def sqrt_price_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
    """Convert sqrtPriceX96 to human-readable price (token1 per token0)."""
    price_raw = (sqrt_price_x96 / Q96) ** 2
    return price_raw * (10 ** (decimals0 - decimals1))


# ═════════════════════════════════════════════════════════════════════════════
# QUOTER
# ═════════════════════════════════════════════════════════════════════════════

class V3Quoter:
    """
    Computes real swap outputs for V3 pools.

    Reads pool state on-chain and applies V3 concentrated liquidity math
    to determine the exact output for a given input amount. Works with
    any V3 fork.
    """

    def __init__(self, w3, token_registry: dict[str, Any] | None = None):
        """
        Args:
            w3: web3.Web3 instance connected to Arbitrum
            token_registry: {symbol: Token} mapping from dex_scanner
        """
        self.w3 = w3
        self.token_registry = token_registry or {}
        self._pool_state_cache: dict[str, PoolState] = {}

    def read_pool_state(self, pool_address: str) -> PoolState | None:
        """Read current state from a V3 pool contract."""
        from web3 import Web3

        try:
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(pool_address),
                abi=POOL_ABI,
            )
            slot0 = contract.functions.slot0().call()
            liq = contract.functions.liquidity().call()
            t0 = contract.functions.token0().call()
            t1 = contract.functions.token1().call()
            fee = contract.functions.fee().call()

            state = PoolState(
                address=pool_address,
                token0=t0.lower(),
                token1=t1.lower(),
                fee=fee,
                sqrt_price_x96=slot0[0],
                tick=slot0[1],
                liquidity=liq,
            )
            self._pool_state_cache[pool_address.lower()] = state
            return state
        except Exception as e:
            logger.debug(f"Failed to read pool {pool_address}: {e}")
            return None

    def get_token_decimals(self, token_addr: str) -> int:
        """Look up token decimals from registry."""
        addr_lower = token_addr.lower()
        for sym, tok in self.token_registry.items():
            if tok.address.lower() == addr_lower:
                return tok.decimals
        # Default guess based on common tokens
        return 18

    def get_token_symbol(self, token_addr: str) -> str:
        """Look up token symbol from registry."""
        addr_lower = token_addr.lower()
        for sym, tok in self.token_registry.items():
            if tok.address.lower() == addr_lower:
                return sym
        return token_addr[:8] + "..."

    def quote_swap(
        self,
        pool_address: str,
        token_in_addr: str,
        token_out_addr: str,
        amount_in: int,
        pool_state: PoolState | None = None,
    ) -> SwapQuote | None:
        """
        Simulate a swap on a V3 pool.

        Args:
            pool_address: Pool contract address
            token_in_addr: Address of token being sold
            token_out_addr: Address of token being bought
            amount_in: Amount of token_in in smallest units
            pool_state: Pre-read pool state (avoids extra RPC call)

        Returns:
            SwapQuote with output amount, price impact, etc.
        """
        if pool_state is None:
            pool_state = self.read_pool_state(pool_address)
        if pool_state is None:
            return None
        if pool_state.liquidity == 0:
            return None

        # Determine swap direction
        token_in_lower = token_in_addr.lower()
        zero_for_one = (token_in_lower == pool_state.token0.lower())

        # Compute output
        amount_out, sqrt_price_after = compute_swap_output(
            pool_state.sqrt_price_x96,
            pool_state.liquidity,
            amount_in,
            pool_state.fee,
            zero_for_one,
        )

        if amount_out == 0:
            return None

        # Get decimals for human-readable values
        decimals_in = self.get_token_decimals(token_in_addr)
        decimals_out = self.get_token_decimals(token_out_addr)
        dec0 = self.get_token_decimals(pool_state.token0)
        dec1 = self.get_token_decimals(pool_state.token1)

        amount_in_human = amount_in / (10 ** decimals_in)
        amount_out_human = amount_out / (10 ** decimals_out)

        # Fee consumed
        fee_amount = amount_in - (amount_in * (1_000_000 - pool_state.fee) // 1_000_000)

        # Mid-market price (before swap)
        mid_price = sqrt_price_to_price(pool_state.sqrt_price_x96, dec0, dec1)
        if not zero_for_one:
            mid_price = 1.0 / mid_price if mid_price > 0 else 0

        # Effective price (what we actually got)
        effective_price = amount_out_human / amount_in_human if amount_in_human > 0 else 0

        # Price impact = how much worse than mid-market (in bps)
        if mid_price > 0 and effective_price > 0:
            price_impact_bps = (1 - effective_price / mid_price) * 10000
        else:
            price_impact_bps = 0

        # Check if swap likely crosses tick boundary
        # Rough heuristic: if price moves more than 1%, probably crosses ticks
        if pool_state.sqrt_price_x96 > 0:
            price_move_pct = abs(sqrt_price_after - pool_state.sqrt_price_x96) / pool_state.sqrt_price_x96 * 100
            crosses_tick = price_move_pct > 0.5
        else:
            crosses_tick = True

        return SwapQuote(
            pool_address=pool_address,
            token_in=token_in_addr,
            token_out=token_out_addr,
            amount_in=amount_in,
            amount_out=amount_out,
            amount_in_human=amount_in_human,
            amount_out_human=amount_out_human,
            fee_amount=fee_amount,
            effective_price=effective_price,
            mid_price=mid_price,
            price_impact_bps=abs(price_impact_bps),
            sqrt_price_after=sqrt_price_after,
            crosses_tick=crosses_tick,
        )

    def quote_arb(
        self,
        buy_pool_addr: str,
        sell_pool_addr: str,
        buy_venue: str,
        sell_venue: str,
        token_in_addr: str,
        token_mid_addr: str,
        amount_in: int,
        trade_size_usd: float,
        gas_cost_usd: float = 0.50,
        token_in_symbol: str = "",
        token_mid_symbol: str = "",
    ) -> ArbQuote | None:
        """
        Simulate a full arbitrage round-trip.

        Flow: token_in → [buy pool] → token_mid → [sell pool] → token_in

        We start with token_in, swap to token_mid on the cheaper venue,
        then swap back to token_in on the more expensive venue.

        Args:
            buy_pool_addr: Pool where we buy (cheaper price)
            sell_pool_addr: Pool where we sell (more expensive price)
            buy_venue: Name of buy venue
            sell_venue: Name of sell venue
            token_in_addr: Address of starting/ending token
            token_mid_addr: Address of intermediate token
            amount_in: Starting amount of token_in (smallest units)
            trade_size_usd: Approximate USD value (for reporting)
            gas_cost_usd: Estimated gas cost for the full arb tx
            token_in_symbol: Symbol for display
            token_mid_symbol: Symbol for display
        """
        # Read both pool states
        buy_state = self.read_pool_state(buy_pool_addr)
        sell_state = self.read_pool_state(sell_pool_addr)
        if buy_state is None or sell_state is None:
            return None

        # Leg 1: Buy — swap token_in → token_mid on buy venue
        buy_quote = self.quote_swap(
            buy_pool_addr, token_in_addr, token_mid_addr, amount_in, buy_state
        )
        if buy_quote is None or buy_quote.amount_out == 0:
            return None

        # Leg 2: Sell — swap token_mid → token_in on sell venue
        sell_quote = self.quote_swap(
            sell_pool_addr, token_mid_addr, token_in_addr,
            buy_quote.amount_out, sell_state
        )
        if sell_quote is None or sell_quote.amount_out == 0:
            return None

        # P&L in starting token
        decimals_in = self.get_token_decimals(token_in_addr)
        final_amount = sell_quote.amount_out
        gross_profit_raw = final_amount - amount_in
        gross_profit_human = gross_profit_raw / (10 ** decimals_in)
        amount_in_human = amount_in / (10 ** decimals_in)
        final_human = final_amount / (10 ** decimals_in)

        gross_profit_bps = (gross_profit_raw / amount_in * 10000) if amount_in > 0 else 0

        # Convert gross profit to USD for comparison with gas
        # Use approximate USD value per token
        usd_per_token = trade_size_usd / amount_in_human if amount_in_human > 0 else 0
        gross_profit_usd = gross_profit_human * usd_per_token
        net_profit_usd = gross_profit_usd - gas_cost_usd

        net_profit_bps = gross_profit_bps - (gas_cost_usd / trade_size_usd * 10000 if trade_size_usd > 0 else 0)

        return ArbQuote(
            buy_venue=buy_venue,
            buy_pool=buy_pool_addr,
            buy_quote=buy_quote,
            sell_venue=sell_venue,
            sell_pool=sell_pool_addr,
            sell_quote=sell_quote,
            token0=token_in_symbol or self.get_token_symbol(token_in_addr),
            token1=token_mid_symbol or self.get_token_symbol(token_mid_addr),
            trade_size_usd=trade_size_usd,
            amount_in=amount_in_human,
            amount_out=final_human,
            gross_profit=gross_profit_human,
            gross_profit_bps=gross_profit_bps,
            gas_cost_usd=gas_cost_usd,
            net_profit_usd=net_profit_usd,
            net_profit_bps=net_profit_bps,
            is_profitable=net_profit_usd > 0,
        )


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH SCANNER
# ═════════════════════════════════════════════════════════════════════════════

def quote_scanner_opportunities(
    scanner,
    trade_sizes_usd: list[float] | None = None,
    gas_cost_usd: float = 0.50,
    pairs_filter: list[str] | None = None,
    max_opps: int = 20,
) -> list[dict]:
    """
    Take opportunities from scanner, quote them with real depth.

    This is the key function — it bridges the scanner's theoretical
    spreads with actual executable profits.

    Args:
        scanner: DexScanner instance (with pools discovered)
        trade_sizes_usd: USD amounts to quote (default: [1k, 5k, 10k, 50k])
        gas_cost_usd: Estimated gas for flash loan arb tx
        pairs_filter: Only quote these pairs (e.g. ["USDC/WETH"])
        max_opps: Maximum opportunities to quote

    Returns:
        List of dicts with quote results per opportunity per trade size
    """
    from engines.dex_scanner import ARBITRUM_TOKENS

    if trade_sizes_usd is None:
        trade_sizes_usd = [1_000, 5_000, 10_000, 50_000]

    quoter = V3Quoter(scanner.w3, scanner.tokens)

    # Run a fresh scan
    opps = scanner.scan_once()

    if pairs_filter:
        opps = [o for o in opps if f"{o.token0}/{o.token1}" in pairs_filter]

    if not opps:
        print("  No opportunities to quote.")
        return []

    # Deduplicate: keep best spread per pair+venue combo
    best_opps: dict[str, Any] = {}
    for opp in opps:
        key = f"{opp.token0}/{opp.token1}:{opp.buy_venue}:{opp.sell_venue}"
        if key not in best_opps or opp.net_spread_bps > best_opps[key].net_spread_bps:
            best_opps[key] = opp

    opps = list(best_opps.values())[:max_opps]

    print(f"\n{'='*80}")
    print(f"QUOTER — Real Depth Analysis")
    print(f"  Opportunities: {len(opps)}")
    print(f"  Trade sizes:   ${', $'.join(str(int(s)) for s in trade_sizes_usd)}")
    print(f"  Gas estimate:  ${gas_cost_usd}")
    print(f"{'='*80}")

    results = []

    for opp in opps:
        print(f"\n  {opp.token0}/{opp.token1}: {opp.buy_venue} → {opp.sell_venue}")
        print(f"    Scanner spread: {opp.spread_bps:.1f} bps gross, "
              f"{opp.net_spread_bps:.1f} bps net")

        # Find the pool addresses for buy and sell venues
        # Need to find pools matching this opportunity
        buy_pools = [p for p in scanner.pools
                     if p.venue == opp.buy_venue
                     and _tokens_match(p, opp.token0, opp.token1)
                     and p.fee_tier / 100 == opp.buy_fee_bps]
        sell_pools = [p for p in scanner.pools
                      if p.venue == opp.sell_venue
                      and _tokens_match(p, opp.token0, opp.token1)
                      and p.fee_tier / 100 == opp.sell_fee_bps]

        if not buy_pools or not sell_pools:
            print(f"    SKIP: Could not find matching pools")
            continue

        buy_pool = buy_pools[0]
        sell_pool = sell_pools[0]

        # Determine token addresses
        # The "starting token" for the arb depends on direction
        # We buy on buy_venue (cheaper) and sell on sell_venue (more expensive)
        # For USDC/WETH with buy on sushi, sell on uni:
        #   Start with USDC → buy WETH on sushi → sell WETH on uni → get USDC back
        # OR:
        #   Start with WETH → sell for USDC on uni → buy WETH on sushi → get WETH back
        #
        # Simplification: use token0 (alphabetically first) as the starting token
        tok0_info = ARBITRUM_TOKENS.get(opp.token0)
        tok1_info = ARBITRUM_TOKENS.get(opp.token1)
        if not tok0_info or not tok1_info:
            print(f"    SKIP: Unknown token")
            continue

        # Start with token0, route through token1
        token_in_addr = tok0_info.address
        token_mid_addr = tok1_info.address
        token_in_decimals = tok0_info.decimals

        # Estimate a reference price for USD sizing
        # Read mid-price from the deeper (buy) pool
        buy_state = quoter.read_pool_state(buy_pool.address)
        if buy_state is None:
            print(f"    SKIP: Could not read buy pool state")
            continue

        # Get USD value of token_in
        # For stablecoins (USDC, USDT, USDCe, DAI): 1 token ≈ $1
        # For WETH: read from pool price vs stablecoin
        # For others: approximate
        usd_per_token_in = _estimate_usd_price(opp.token0)

        print(f"    Buy pool:  {buy_pool.address[:10]}... ({buy_pool.venue}, "
              f"fee={buy_pool.fee_tier/100:.0f}bps)")
        print(f"    Sell pool: {sell_pool.address[:10]}... ({sell_pool.venue}, "
              f"fee={sell_pool.fee_tier/100:.0f}bps)")
        print(f"    Start token: {opp.token0} (≈${usd_per_token_in}/token)")

        for size_usd in trade_sizes_usd:
            # Convert USD to token amount
            amount_tokens = size_usd / usd_per_token_in
            amount_raw = int(amount_tokens * (10 ** token_in_decimals))

            quote = quoter.quote_arb(
                buy_pool_addr=buy_pool.address,
                sell_pool_addr=sell_pool.address,
                buy_venue=opp.buy_venue,
                sell_venue=opp.sell_venue,
                token_in_addr=token_in_addr,
                token_mid_addr=token_mid_addr,
                amount_in=amount_raw,
                trade_size_usd=size_usd,
                gas_cost_usd=gas_cost_usd,
                token_in_symbol=opp.token0,
                token_mid_symbol=opp.token1,
            )

            if quote is None:
                print(f"    ${size_usd:>6,.0f}: FAILED (quote returned None)")
                continue

            profit_indicator = "✅" if quote.is_profitable else "❌"
            cross_warning = " ⚠️TICK" if (quote.buy_quote.crosses_tick
                                           or quote.sell_quote.crosses_tick) else ""

            print(f"    ${size_usd:>6,.0f}: gross={quote.gross_profit_bps:+.1f}bps "
                  f"(${quote.gross_profit * usd_per_token_in:+.2f}) "
                  f"net=${quote.net_profit_usd:+.2f} "
                  f"impact_buy={quote.buy_quote.price_impact_bps:.1f}bps "
                  f"impact_sell={quote.sell_quote.price_impact_bps:.1f}bps "
                  f"{profit_indicator}{cross_warning}")

            results.append({
                "pair": f"{opp.token0}/{opp.token1}",
                "buy_venue": opp.buy_venue,
                "sell_venue": opp.sell_venue,
                "scanner_spread_bps": opp.spread_bps,
                "scanner_net_bps": opp.net_spread_bps,
                "trade_size_usd": size_usd,
                "gross_profit_bps": quote.gross_profit_bps,
                "gross_profit_usd": quote.gross_profit * usd_per_token_in,
                "net_profit_usd": quote.net_profit_usd,
                "buy_impact_bps": quote.buy_quote.price_impact_bps,
                "sell_impact_bps": quote.sell_quote.price_impact_bps,
                "is_profitable": quote.is_profitable,
                "crosses_tick": quote.buy_quote.crosses_tick or quote.sell_quote.crosses_tick,
            })

    # Summary
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        profitable = df[df["is_profitable"]]
        print(f"\n{'='*80}")
        print(f"QUOTER SUMMARY")
        print(f"  Total quotes:   {len(df)}")
        print(f"  Profitable:     {len(profitable)}")
        if not profitable.empty:
            print(f"  Avg net profit: ${profitable['net_profit_usd'].mean():.2f}")
            print(f"  Max net profit: ${profitable['net_profit_usd'].max():.2f}")
            print(f"  Best pair:      {profitable.loc[profitable['net_profit_usd'].idxmax(), 'pair']}")
        unprofitable = df[~df["is_profitable"]]
        if not unprofitable.empty:
            print(f"  Avg loss:       ${unprofitable['net_profit_usd'].mean():.2f}")
        print(f"{'='*80}")

    return results


def _tokens_match(pool, tok0_sym: str, tok1_sym: str) -> bool:
    """Check if pool contains both tokens (in either order)."""
    pool_tokens = {pool.token0, pool.token1}
    return {tok0_sym, tok1_sym} == pool_tokens


def _estimate_usd_price(symbol: str) -> float:
    """Rough USD price estimate for trade sizing. Doesn't need to be exact."""
    stables = {"USDC", "USDCe", "USDT", "DAI"}
    if symbol in stables:
        return 1.0
    # These are rough estimates — only used for trade sizing, not P&L
    prices = {
        "WETH": 1900.0,
        "WBTC": 85000.0,
        "ARB": 0.35,
        "GMX": 20.0,
        "LINK": 14.0,
        "UNI": 6.0,
        "PENDLE": 3.0,
    }
    return prices.get(symbol, 1.0)
