"""
engines/market_maker.py — Polymarket Market Making Bot

Provides liquidity on both sides of prediction markets, earning the spread.
Target: 1-3% monthly (13-45% APR) with low drawdown.

Architecture:
    1. SCAN    — Find markets suitable for market making (liquid, stable, wide spread)
    2. QUOTE   — Place limit orders on both YES and NO sides
    3. MANAGE  — Track inventory, adjust quotes, manage risk
    4. REPORT  — P&L tracking and performance analysis

Key principles:
    - Never hold more than MAX_INVENTORY on either side
    - Widen spreads when volatility spikes
    - Pull all quotes before major news events
    - Only make markets where we have NO directional view

Usage:
    python -m engines.market_maker scan                     # Find MM opportunities
    python -m engines.market_maker scan --min-spread 0.03   # Markets with 3%+ spread
    python -m engines.market_maker quote --slug <slug>      # Show proposed quotes
    python -m engines.market_maker run --slug <slug>        # Live market making (dry run)
    python -m engines.market_maker run --slug <slug> --execute  # Live execution
"""
import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DB_PATH = "data/market_maker.db"

# Contract addresses (Polygon)
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"

# Minimal ABIs for the functions we need
CTF_ABI = json.loads("""[
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "partition", "type": "uint256[]"},
            {"name": "amount", "type": "uint256"}
        ],
        "name": "mergePositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "id", "type": "uint256"}
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]""")

# Risk parameters
MAX_INVENTORY_PCT = 0.30       # Max 30% of capital on one side
MAX_POSITION_USD = 50          # Max position per market
MIN_SPREAD = 0.02              # Don't make markets with < 2% spread
DEFAULT_QUOTE_SIZE = 10        # $10 per side
REQUOTE_INTERVAL = 30          # Re-check every 30 seconds
PULL_BEFORE_NEWS_MINS = 5      # Pull quotes 5 min before known events


def init_db():
    """Initialize SQLite database for MM tracking."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mm_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            market_slug TEXT,
            side TEXT,
            direction TEXT,
            price REAL,
            size REAL,
            order_id TEXT,
            status TEXT DEFAULT 'open'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mm_inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            market_slug TEXT,
            yes_shares REAL DEFAULT 0,
            no_shares REAL DEFAULT 0,
            cost_basis REAL DEFAULT 0,
            realized_pnl REAL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mm_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            market_slug TEXT,
            mid_price REAL,
            bid REAL,
            ask REAL,
            spread REAL,
            our_bid REAL,
            our_ask REAL,
            our_spread REAL,
            tick_size REAL,
            yes_inventory REAL,
            no_inventory REAL,
            -- Features for ML spread optimization
            realized_vol REAL,
            flow_rate REAL,
            price_momentum REAL,
            spread_pct REAL,
            time_of_day REAL,
            inventory_skew REAL,
            round_trips_this_session INTEGER
        )
    """)
    conn.commit()
    return conn


def get_clob_client():
    """Get a read-only CLOB client."""
    from py_clob_client.client import ClobClient
    return ClobClient("https://clob.polymarket.com")


def get_auth_client():
    """Get authenticated CLOB client for order placement."""
    from dotenv import load_dotenv
    load_dotenv()
    from py_clob_client.client import ClobClient
    client = ClobClient(
        "https://clob.polymarket.com",
        key=os.getenv("POLYMARKET_PRIVATE_KEY"),
        chain_id=137, signature_type=0
    )
    client.set_api_creds(client.derive_api_key())
    return client


def merge_positions_onchain(condition_id, yes_token_id, no_token_id, dry_run=False):
    """Execute on-chain merge: 1 YES + 1 NO → 1 USDC.
    
    Burns matching YES+NO token pairs via the CTF contract's mergePositions(),
    returning USDC.e directly to wallet. No need to wait for resolution.
    
    Returns: (merged_count, usdc_received) or (0, 0) if nothing to merge.
    """
    from dotenv import load_dotenv
    load_dotenv()
    from web3 import Web3

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("    ❌ No POLYMARKET_PRIVATE_KEY in .env")
        return 0, 0

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    if not w3.is_connected():
        print(f"    ❌ Cannot connect to {POLYGON_RPC}")
        return 0, 0

    account = w3.eth.account.from_key(pk)
    wallet = account.address

    # Get CTF contract
    ctf = w3.eth.contract(
        address=Web3.to_checksum_address(CTF_ADDRESS),
        abi=CTF_ABI
    )

    # Check on-chain balances of YES and NO tokens
    yes_balance = ctf.functions.balanceOf(wallet, int(yes_token_id)).call()
    no_balance = ctf.functions.balanceOf(wallet, int(no_token_id)).call()

    # Minimum of the two = number of full sets we can merge
    merge_amount = min(yes_balance, no_balance)

    if merge_amount == 0:
        return 0, 0

    # USDC.e has 6 decimals, CTF tokens match this
    merge_shares = merge_amount / 1e6
    usdc_back = merge_shares  # 1 share of each = $1 USDC

    print(f"    📦 On-chain balances: YES={yes_balance} ({yes_balance/1e6:.2f} shares)  "
          f"NO={no_balance} ({no_balance/1e6:.2f} shares)")
    print(f"    🔄 Mergeable: {merge_amount} ({merge_shares:.2f} shares) → ${usdc_back:.2f} USDC")

    if dry_run:
        print(f"    (dry run — not executing)")
        return merge_shares, usdc_back

    # Execute merge
    try:
        # Build the transaction
        parent_collection_id = bytes(32)  # bytes32(0)
        partition = [1, 2]  # Binary: YES=1, NO=2

        tx = ctf.functions.mergePositions(
            Web3.to_checksum_address(USDC_E_ADDRESS),
            parent_collection_id,
            bytes.fromhex(condition_id[2:]) if condition_id.startswith("0x") else bytes.fromhex(condition_id),
            partition,
            merge_amount
        ).build_transaction({
            "from": wallet,
            "nonce": w3.eth.get_transaction_count(wallet),
            "gas": 300000,
            "gasPrice": w3.eth.gas_price,
            "chainId": 137,
        })

        # Sign and send
        signed = w3.eth.account.sign_transaction(tx, pk)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"    📤 Tx sent: {tx_hash.hex()[:20]}...")

        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        if receipt.status == 1:
            print(f"    ✅ Merge successful! {merge_shares:.2f} pairs → ${usdc_back:.2f} USDC")
            print(f"    Gas used: {receipt.gasUsed}  "
                  f"(${receipt.gasUsed * receipt.effectiveGasPrice / 1e18:.4f} POL)")
            return merge_shares, usdc_back
        else:
            print(f"    ❌ Merge transaction reverted!")
            return 0, 0

    except Exception as e:
        print(f"    ❌ Merge error: {e}")
        return 0, 0


def analyze_market_for_mm(event):
    """Analyze a market's suitability for market making."""
    markets = event.get("markets", [])
    if not markets:
        return None

    results = []
    clob = get_clob_client()

    for m in markets:
        question = m.get("question", m.get("groupItemTitle", ""))
        slug = m.get("slug", "")
        volume = float(m.get("volume", 0))
        end_date = (m.get("endDate") or "")[:10]

        try:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue

        if not token_ids or len(token_ids) < 2:
            continue

        yes_token = token_ids[0]

        # Get CLOB pricing
        try:
            mid_resp = clob.get_midpoint(yes_token)
            mid = float(mid_resp.get("mid", 0)) if isinstance(mid_resp, dict) else float(mid_resp)
        except Exception:
            mid = 0

        try:
            buy_resp = clob.get_price(yes_token, "BUY")
            best_bid = float(buy_resp.get("price", 0)) if isinstance(buy_resp, dict) else float(buy_resp)
        except Exception:
            best_bid = 0

        try:
            sell_resp = clob.get_price(yes_token, "SELL")
            best_ask = float(sell_resp.get("price", 0)) if isinstance(sell_resp, dict) else float(sell_resp)
        except Exception:
            best_ask = 0

        spread = best_ask - best_bid if (best_ask > 0 and best_bid > 0) else 0

        # Note: get_order_book() returns stale depth (known bug #180)
        # We skip depth-based scoring — use volume as liquidity proxy instead
        depth_bid = depth_ask = 0

        # Score for MM suitability
        # Good MM markets: mid near 50%, wide spread, decent volume, far resolution
        score = 0

        # Prefer mid near 50% (lower resolution risk)
        if 0.30 <= mid <= 0.70:
            score += 30
        elif 0.20 <= mid <= 0.80:
            score += 15

        # Prefer wider spreads (more profit per round trip)
        if spread >= 0.05:
            score += 25
        elif spread >= 0.03:
            score += 15
        elif spread >= 0.02:
            score += 5

        # Prefer decent volume (orders will fill)
        if volume >= 100000:
            score += 20
        elif volume >= 10000:
            score += 10

        # Prefer further resolution dates (more time to earn)
        if end_date:
            try:
                resolve = datetime.strptime(end_date, "%Y-%m-%d")
                days_left = (resolve - datetime.now()).days
                if days_left > 60:
                    score += 15
                elif days_left > 14:
                    score += 10
                elif days_left > 3:
                    score += 5
            except ValueError:
                pass

        # Volume-based competition proxy (replaces stale depth data)
        if volume < 50000:
            score += 10  # Less competition

        # Prefer 3-decimal tick size (more granular quoting)
        tick = float(m.get("orderPriceMinTickSize", 0.01))
        if tick <= 0.001:
            score += 15  # 3 decimal = dynamic spread optimization possible
        elif tick <= 0.01:
            score += 5

        results.append({
            "question": question,
            "slug": slug,
            "event_slug": event.get("slug", ""),
            "volume": volume,
            "end_date": end_date,
            "mid": mid,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread / mid * 100 if mid > 0 else 0,
            "tick_size": tick,
            "depth_bid": depth_bid,
            "depth_ask": depth_ask,
            "yes_token": yes_token,
            "no_token": token_ids[1] if len(token_ids) > 1 else "",
            "score": score,
        })

    return results


def cmd_scan(args):
    """Scan for market making opportunities."""
    min_spread = getattr(args, "min_spread", 0.02)
    min_volume = getattr(args, "min_volume", 5000)
    limit = getattr(args, "limit", 30)

    print(f"\n{'='*120}")
    print(f"MARKET MAKER OPPORTUNITY SCANNER — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Min spread: {min_spread:.0%}  |  Min volume: ${min_volume:,.0f}")
    print(f"{'='*120}")

    print(f"\n  Fetching markets...")
    all_results = []

    tags = ["politics", "crypto", "finance", "geopolitics", "economy",
            "tech", "sports", "culture", "science"]

    seen_slugs = set()
    for tag in tags:
        try:
            r = requests.get(f"{GAMMA_API}/events", params={
                "tag_slug": tag, "limit": "30", "active": "true",
                "closed": "false", "order": "volume", "ascending": "false",
            }, timeout=10)
            for event in r.json():
                eid = event.get("id", "")
                if eid in seen_slugs:
                    continue
                seen_slugs.add(eid)

                results = analyze_market_for_mm(event)
                if results:
                    for res in results:
                        if res["spread"] >= min_spread and res["volume"] >= min_volume:
                            all_results.append(res)

            time.sleep(0.3)
        except Exception:
            continue

    all_results.sort(key=lambda r: r["score"], reverse=True)
    all_results = all_results[:limit]

    print(f"  Found {len(all_results)} MM opportunities\n")

    if all_results:
        print(f"  {'#':<3s} {'Market':<55s} {'Mid':>5s} {'Bid':>5s} {'Ask':>5s} "
              f"{'Sprd':>5s} {'S%':>5s} {'Tick':>5s} {'Vol':>10s} {'Score':>5s} {'Resolve'}")
        print(f"  {'─'*120}")

        for i, r in enumerate(all_results, 1):
            q = r["question"][:54]
            print(f"  {i:<3d} {q:<55s} {r['mid']:>4.0%} {r['best_bid']:>4.0%} "
                  f"{r['best_ask']:>4.0%} {r['spread']:>4.2f} "
                  f"{r['spread_pct']:>4.1f}% {r['tick_size']:>5.3f} ${r['volume']:>9,.0f} "
                  f"{r['score']:>5d} {r['end_date']}")

        # Show top 3 with proposed quotes
        print(f"\n{'─'*120}")
        print(f"  TOP OPPORTUNITIES — Proposed Quotes")
        print(f"{'─'*120}")

        for r in all_results[:5]:
            mid = r["mid"]
            spread = r["spread"]

            # Our quotes: tighten the existing spread by ~30%
            # We want to be inside the current best bid/ask
            our_half_spread = max(spread * 0.35, 0.01)
            our_bid = round(mid - our_half_spread, 2)
            our_ask = round(mid + our_half_spread, 2)
            our_spread = our_ask - our_bid
            profit_per_round_trip = our_spread
            trades_for_1pct = int(0.01 * DEFAULT_QUOTE_SIZE * 2 / profit_per_round_trip) if profit_per_round_trip > 0 else 999

            print(f"\n  📊 {r['question'][:70]}")
            print(f"     Market:  Mid={mid:.1%}  Bid={r['best_bid']:.1%}  Ask={r['best_ask']:.1%}  "
                  f"Spread={spread:.2f} ({r['spread_pct']:.1f}%)")
            print(f"     Our Quotes:  Bid={our_bid:.2f}  Ask={our_ask:.2f}  "
                  f"Spread={our_spread:.2f}")
            print(f"     Profit/RT: ${profit_per_round_trip:.3f} per share  "
                  f"(${profit_per_round_trip * DEFAULT_QUOTE_SIZE:.2f} per ${DEFAULT_QUOTE_SIZE} lot)")
            print(f"     Round trips for 1% return on ${DEFAULT_QUOTE_SIZE*2} capital: {trades_for_1pct}")
            print(f"     Score:   {r['score']}  |  Resolve: {r['end_date']}")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/mm_opportunities.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(all_results),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved: data/mm_opportunities.json")
    print(f"{'='*120}")


def cmd_quote(args):
    """Show proposed quotes for a specific market."""
    slug = args.slug
    print(f"\n  Fetching market: {slug}...")

    clob = get_clob_client()

    r = requests.get(f"{GAMMA_API}/markets", params={"slug": slug})
    markets = r.json()
    if not markets:
        # Try as event slug
        r = requests.get(f"{GAMMA_API}/events", params={"slug": slug})
        events = r.json()
        if events:
            markets = events[0].get("markets", [])

    if not markets:
        print(f"  ❌ Market not found")
        return

    m = markets[0] if isinstance(markets, list) else markets
    question = m.get("question", "")
    token_ids = json.loads(m.get("clobTokenIds", "[]"))

    if not token_ids:
        print(f"  ❌ No token IDs found")
        return

    yes_token = token_ids[0]
    no_token = token_ids[1] if len(token_ids) > 1 else ""

    # Get live CLOB data
    mid = buy = sell = 0
    volume = float(m.get("volume", 0))
    try:
        mid_r = clob.get_midpoint(yes_token)
        mid = float(mid_r.get("mid", 0)) if isinstance(mid_r, dict) else float(mid_r)
        buy_r = clob.get_price(yes_token, "BUY")
        buy = float(buy_r.get("price", 0)) if isinstance(buy_r, dict) else float(buy_r)
        sell_r = clob.get_price(yes_token, "SELL")
        sell = float(sell_r.get("price", 0)) if isinstance(sell_r, dict) else float(sell_r)
    except Exception as e:
        print(f"  Price error: {e}")

    spread = sell - buy
    quote_size = DEFAULT_QUOTE_SIZE
    tick = float(m.get("orderPriceMinTickSize", 0.01))
    tick_dec = len(str(tick).rstrip('0').split('.')[-1]) if '.' in str(tick) else 0

    # Calculate our quotes (tick-size aware)
    our_half = max(spread * 0.35, tick)
    our_bid = round((mid - our_half) / tick) * tick
    our_ask = round((mid + our_half) / tick) * tick
    our_spread = our_ask - our_bid

    print(f"\n{'='*80}")
    print(f"  MARKET MAKER QUOTES: {question[:65]}")
    print(f"{'='*80}")
    print(f"\n  Current Market:")
    print(f"    Mid:        {mid:.3f} ({mid:.1%})")
    print(f"    Best Bid:   {buy:.3f}")
    print(f"    Best Ask:   {sell:.3f}")
    print(f"    Spread:     {spread:.3f} ({spread/mid*100:.1f}%)" if mid > 0 else "")
    print(f"    Tick size:  {tick} ({int(1/tick)} price levels)")
    print(f"\n  Our Proposed Quotes (${quote_size} per side):")
    print(f"    YES BID:  {our_bid:.{tick_dec}f}  ×  {quote_size/our_bid:.0f} shares" if our_bid > 0 else "")
    print(f"    YES ASK:  {our_ask:.{tick_dec}f}  ×  {quote_size/our_ask:.0f} shares" if our_ask > 0 else "")
    print(f"    Spread:   {our_spread:.{tick_dec}f} ({our_spread/mid*100:.1f}%)" if mid > 0 else "")
    print(f"\n  Economics:")
    print(f"    Profit per round trip:    ${our_spread:.4f}/share")
    print(f"    Profit per ${quote_size} lot RT:  ${our_spread * (quote_size/our_bid):.3f}" if our_bid > 0 else "")
    print(f"    Capital required:         ${quote_size * 2:.0f}")

    # Order book depth — NOTE: get_order_book() returns stale data (known issue #180)
    # Use get_price() for live pricing. Show what we can.
    print(f"\n  Order Book Note:")
    print(f"    ⚠ get_order_book() returns stale depth (known Polymarket bug #180)")
    print(f"    Live prices from get_price() are accurate (shown above)")

    # Try to get NO side pricing too
    if no_token:
        try:
            no_mid_r = clob.get_midpoint(no_token)
            no_mid = float(no_mid_r.get("mid", 0)) if isinstance(no_mid_r, dict) else float(no_mid_r)
            no_buy_r = clob.get_price(no_token, "BUY")
            no_buy = float(no_buy_r.get("price", 0)) if isinstance(no_buy_r, dict) else float(no_buy_r)
            no_sell_r = clob.get_price(no_token, "SELL")
            no_sell = float(no_sell_r.get("price", 0)) if isinstance(no_sell_r, dict) else float(no_sell_r)
            print(f"\n  NO Side Pricing:")
            print(f"    Mid: {no_mid:.3f}  |  Bid: {no_buy:.3f}  |  Ask: {no_sell:.3f}")
            print(f"    YES+NO mid: {mid + no_mid:.3f} (should be ~1.00)")
        except Exception:
            pass

    # Estimate monthly return
    if our_spread > 0 and volume > 0:
        # Conservative: assume we capture 0.1% of daily volume as round trips
        daily_vol_est = volume / 30  # Rough daily volume
        daily_rts_est = daily_vol_est * 0.001 / (quote_size * 2)
        daily_profit_est = daily_rts_est * our_spread * (quote_size / our_bid)
        monthly_return = daily_profit_est * 30 / (quote_size * 2) * 100

        print(f"\n  Return Estimate (conservative):")
        print(f"    Est daily volume:  ${daily_vol_est:,.0f}")
        print(f"    Est daily RTs:     {daily_rts_est:.1f}")
        print(f"    Est daily profit:  ${daily_profit_est:.2f}")
        print(f"    Est monthly return: {monthly_return:.1f}% on ${quote_size*2} capital")

    print(f"\n{'='*80}")


def cmd_run(args):
    """Run the market maker with fill-chase state machine.

    CRITICAL: Each loop cancels our orders FIRST, then reads the market.
    This ensures we see the REAL spread, not our own quotes reflected back.
    """
    slug = args.slug
    execute = getattr(args, "execute", False)
    quote_size = getattr(args, "size", DEFAULT_QUOTE_SIZE)
    duration = getattr(args, "minutes", 30)

    mode = "🔴 LIVE" if execute else "🟡 DRY RUN"

    print(f"\n{'='*100}")
    print(f"  MARKET MAKER — {mode} (Fill-Chase Mode)")
    print(f"  Market: {slug}  |  Quote size: ${quote_size}/side  |  Duration: {duration}m")
    print(f"{'='*100}")

    clob = get_clob_client()
    auth_client = get_auth_client() if execute else None
    conn = init_db()

    r = requests.get(f"{GAMMA_API}/markets", params={"slug": slug})
    markets = r.json()
    if not markets:
        print(f"  ❌ Market not found")
        return

    m = markets[0]
    question = m.get("question", "")
    token_ids = json.loads(m.get("clobTokenIds", "[]"))
    yes_token = token_ids[0]
    no_token = token_ids[1] if len(token_ids) > 1 else ""
    condition_id = m.get("conditionId", "")

    if not no_token:
        print(f"  ❌ No NO token found")
        return

    tick_size = float(m.get("orderPriceMinTickSize", 0.01))
    tick_dec = len(str(tick_size).rstrip('0').split('.')[-1]) if '.' in str(tick_size) else 0

    print(f"  Market:    {question[:70]}")
    print(f"  YES token: {yes_token[:24]}...")
    print(f"  NO token:  {no_token[:24]}...")
    print(f"  Condition: {condition_id[:24]}..." if condition_id else "")
    print(f"  Tick size: {tick_size} ({tick_dec} decimals, {int(1/tick_size)} price levels)")
    print(f"  Strategy:  BUY YES @ bid  +  BUY NO @ (1-ask)  →  merge for $1.00")

    # State
    yes_inv = no_inv = 0.0
    yes_cost = no_cost = 0.0
    realized_pnl = 0.0
    round_trips = 0
    yes_oid = no_oid = None
    prev_yes_m = prev_no_m = 0.0

    state = "FLAT"
    chase_ticks = chase_loops = 0
    chase_entry_mid = 0.0
    mid = 0.0

    CHASE_INTERVAL = 2
    MAX_CHASE_TICKS = 5
    ADVERSE_PCT = 0.03
    MAX_CHASE_SECS = 300

    mid_history = []
    spread_history = []

    total_loops = int(duration * 60 / REQUOTE_INTERVAL)

    print(f"\n  Loop: cancel→check fills→read market→quote→post")
    print(f"  Chase: tighten every {CHASE_INTERVAL} loops, max {MAX_CHASE_TICKS} ticks, "
          f"cut on {ADVERSE_PCT:.0%} adverse or {MAX_CHASE_SECS}s timeout")
    print(f"\n  {'Time':<10s} {'State':<8s} {'Mid':>5s} {'MktSp':>5s} {'Prft':>5s} "
          f"{'YesBd':>6s} {'NoBd':>6s} "
          f"{'YInv':>5s} {'NInv':>5s} {'RTs':>4s} {'PnL':>8s}")
    print(f"  {'─'*90}")

    # ==== STARTUP: Cancel ALL existing orders to start clean ====
    if execute and auth_client:
        try:
            existing = auth_client.get_orders()
            if existing:
                print(f"\n  ⚠ Found {len(existing)} existing orders — cancelling all...")
                for o in existing:
                    oid = o.get("id", "")
                    matched = float(o.get("size_matched", 0))
                    if matched > 0:
                        print(f"    ⚠ Order {oid[:16]}... has {matched} matched (may need attention)")
                    try:
                        auth_client.cancel(order_id=oid)
                    except Exception:
                        pass
                print(f"    Cancelled {len(existing)} orphaned orders.")
                time.sleep(0.5)
        except Exception as e:
            print(f"  Startup cleanup error: {e}")

    for loop in range(total_loops):
        try:
            now = datetime.now(timezone.utc)

            # ==== STEP 1: CHECK FILLS then CANCEL our orders ====
            if execute and auth_client:
                # Check fills before cancelling
                orders = []
                try:
                    orders = auth_client.get_orders()
                    for o in orders:
                        oid = o.get("id", "")
                        matched = float(o.get("size_matched", 0))
                        price = float(o.get("price", 0))

                        if oid == yes_oid and matched > prev_yes_m:
                            nf = matched - prev_yes_m
                            yes_inv += nf
                            yes_cost += nf * price
                            prev_yes_m = matched
                            print(f"    ✅ YES filled: +{nf:.1f} @ {price:.3f}")
                            if state in ("FLAT", "QUOTING"):
                                state = "CHASE_NO"
                                chase_ticks = chase_loops = 0
                                chase_entry_mid = mid
                                print(f"    → CHASE_NO")

                        elif oid == no_oid and matched > prev_no_m:
                            nf = matched - prev_no_m
                            no_inv += nf
                            no_cost += nf * price
                            prev_no_m = matched
                            print(f"    ✅ NO filled:  +{nf:.1f} @ {price:.3f}")
                            if state in ("FLAT", "QUOTING"):
                                state = "CHASE_YS"
                                chase_ticks = chase_loops = 0
                                chase_entry_mid = mid
                                print(f"    → CHASE_YS")
                except Exception:
                    pass

                # Cancel ALL orders from the list we just fetched
                try:
                    for ao in orders:
                        try:
                            auth_client.cancel(order_id=ao.get("id", ""))
                        except Exception:
                            pass
                    yes_oid = None
                    no_oid = None
                except Exception:
                    pass

                time.sleep(0.5)

                # Merge check
                mg = min(yes_inv, no_inv)
                if mg >= 1.0:
                    ayc = yes_cost / yes_inv if yes_inv > 0 else 0
                    anc = no_cost / no_inv if no_inv > 0 else 0
                    ms, _ = merge_positions_onchain(condition_id, yes_token, no_token)
                    if ms > 0:
                        mp = ms * (1.0 - ayc - anc)
                        realized_pnl += mp
                        round_trips += int(ms)
                        yes_inv = max(yes_inv - ms, 0)
                        no_inv = max(no_inv - ms, 0)
                        yes_cost = max(yes_cost - ms * ayc, 0)
                        no_cost = max(no_cost - ms * anc, 0)
                        print(f"    💰 MERGED {ms:.1f} → +${mp:.4f}")
                        state = "FLAT"
                        chase_ticks = chase_loops = 0

                # Adverse / timeout
                if state.startswith("CHASE"):
                    chase_loops += 1
                    adv = 0
                    if state == "CHASE_NO" and chase_entry_mid > 0:
                        adv = (chase_entry_mid - mid) / chase_entry_mid
                    elif state == "CHASE_YS" and chase_entry_mid > 0:
                        adv = (mid - chase_entry_mid) / chase_entry_mid

                    if adv >= ADVERSE_PCT:
                        print(f"    ⚠ Adverse {adv:.1%} — CUT")
                        state = "CUTTING"
                    elif chase_ticks >= MAX_CHASE_TICKS:
                        print(f"    ⚠ Max chase — CUT")
                        state = "CUTTING"
                    elif chase_loops * REQUOTE_INTERVAL >= MAX_CHASE_SECS:
                        print(f"    ⚠ Timeout — CUT")
                        state = "CUTTING"

                if state == "CUTTING":
                    from py_clob_client.clob_types import OrderArgs
                    from py_clob_client.order_builder.constants import BUY
                    if yes_inv > no_inv:
                        cp = round(((1 - mid) + tick_size * 2) / tick_size) * tick_size
                        cp = min(cp, 0.99)
                        cs = round(yes_inv - no_inv, 2)
                        if cs >= 1:
                            try:
                                print(f"    🔪 BUY {cs:.1f} NO @ {cp:.3f}")
                                o = auth_client.create_order(OrderArgs(price=cp, size=cs, side=BUY, token_id=no_token))
                                r = auth_client.post_order(o)
                                no_oid = r.get("orderID", "")
                                prev_no_m = 0.0
                            except Exception as e:
                                print(f"    CUT err: {e}")
                    elif no_inv > yes_inv:
                        cp = round((mid + tick_size * 2) / tick_size) * tick_size
                        cp = min(cp, 0.99)
                        cs = round(no_inv - yes_inv, 2)
                        if cs >= 1:
                            try:
                                print(f"    🔪 BUY {cs:.1f} YES @ {cp:.3f}")
                                o = auth_client.create_order(OrderArgs(price=cp, size=cs, side=BUY, token_id=yes_token))
                                r = auth_client.post_order(o)
                                yes_oid = r.get("orderID", "")
                                prev_yes_m = 0.0
                            except Exception as e:
                                print(f"    CUT err: {e}")
                    state = "FLAT"
                    chase_ticks = chase_loops = 0

            # ==== STEP 2: READ CLEAN MARKET ====
            mid = mkt_bid = mkt_ask = 0
            try:
                mr = clob.get_midpoint(yes_token)
                mid = float(mr.get("mid", 0)) if isinstance(mr, dict) else float(mr)
                br = clob.get_price(yes_token, "BUY")
                mkt_bid = float(br.get("price", 0)) if isinstance(br, dict) else float(br)
                sr = clob.get_price(yes_token, "SELL")
                mkt_ask = float(sr.get("price", 0)) if isinstance(sr, dict) else float(sr)
            except Exception:
                pass

            if mid == 0:
                print(f"  {now.strftime('%H:%M:%S'):<10s} {'—':<8s} No price data")
                time.sleep(REQUOTE_INTERVAL)
                continue

            spread = mkt_ask - mkt_bid

            # ==== STEP 3: FEATURES ====
            mid_history.append((int(now.timestamp()), mid))
            spread_history.append((int(now.timestamp()), spread))
            if len(mid_history) > 300:
                mid_history = mid_history[-300:]
                spread_history = spread_history[-300:]

            realized_vol = 0
            if len(mid_history) >= 10:
                rm = [h[1] for h in mid_history[-30:]]
                rets = [(rm[i] - rm[i-1]) / rm[i-1] for i in range(1, len(rm)) if rm[i-1] > 0]
                if rets:
                    mr = sum(rets) / len(rets)
                    realized_vol = (sum((r - mr)**2 for r in rets) / len(rets)) ** 0.5

            price_momentum = 0
            if len(mid_history) >= 10:
                om = mid_history[-10][1]
                if om > 0:
                    price_momentum = (mid - om) / om

            inv_skew = (yes_inv - no_inv) / max(quote_size, 1)

            # ==== STEP 4: CALCULATE QUOTES ====
            our_yes_bid = our_no_bid = 0

            if state in ("FLAT", "QUOTING"):
                if spread < tick_size * 2:
                    print(f"  {now.strftime('%H:%M:%S'):<10s} {'WAIT':<8s} {mid:>4.0%} {spread:>.3f} "
                          f"— spread < {tick_size*2:.3f}")
                    time.sleep(REQUOTE_INTERVAL)
                    continue

                # 1 tick inside the REAL market
                iyb = mkt_bid + tick_size
                inb = (1 - mkt_ask) + tick_size

                # Pull back for vol/mom
                vt = min(int(realized_vol * 3000), 3)
                mt = min(int(abs(price_momentum) * 200), 2)
                sk = inv_skew * tick_size * 5

                our_yes_bid = round((iyb - vt * tick_size - mt * tick_size - sk) / tick_size) * tick_size
                our_no_bid = round((inb - vt * tick_size - mt * tick_size + sk) / tick_size) * tick_size

                # Never above mid
                our_yes_bid = min(our_yes_bid, round((mid - tick_size) / tick_size) * tick_size)
                our_no_bid = min(our_no_bid, round(((1 - mid) - tick_size) / tick_size) * tick_size)
                state = "QUOTING"

            elif state == "CHASE_NO":
                bnp = round(((1 - mid) + tick_size) / tick_size) * tick_size
                if chase_loops > 0 and chase_loops % CHASE_INTERVAL == 0 and chase_ticks < MAX_CHASE_TICKS:
                    chase_ticks += 1
                    print(f"    📈 Tightening NO +{chase_ticks}")
                our_no_bid = min(round((bnp + chase_ticks * tick_size) / tick_size) * tick_size, 0.99)

            elif state == "CHASE_YS":
                byp = round((mkt_bid + tick_size) / tick_size) * tick_size
                if chase_loops > 0 and chase_loops % CHASE_INTERVAL == 0 and chase_ticks < MAX_CHASE_TICKS:
                    chase_ticks += 1
                    print(f"    📈 Tightening YES +{chase_ticks}")
                our_yes_bid = min(round((byp + chase_ticks * tick_size) / tick_size) * tick_size, 0.99)

            our_yes_bid = max(our_yes_bid, 0) if our_yes_bid else 0
            our_no_bid = max(our_no_bid, 0) if our_no_bid else 0
            pc = (our_yes_bid + our_no_bid) if (our_yes_bid > 0 and our_no_bid > 0) else 0
            pp = (1.0 - pc) if pc > 0 else 0

            ybs = round(quote_size / our_yes_bid, 2) if our_yes_bid > 0 else 0
            nbs = round(quote_size / our_no_bid, 2) if our_no_bid > 0 else 0

            if state == "QUOTING":
                mx = quote_size * MAX_INVENTORY_PCT * 10
                if yes_inv >= mx: ybs = 0
                if no_inv >= mx: nbs = 0

            # ==== STEP 5: LOG + DISPLAY ====
            conn.execute("""
                INSERT INTO mm_snapshots
                (timestamp, market_slug, mid_price, bid, ask, spread,
                 our_bid, our_ask, our_spread, tick_size,
                 yes_inventory, no_inventory,
                 realized_vol, flow_rate, price_momentum,
                 spread_pct, time_of_day, inventory_skew,
                 round_trips_this_session)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (now.isoformat(), slug, mid, mkt_bid, mkt_ask, spread,
                  our_yes_bid, our_no_bid, pp, tick_size,
                  yes_inv, no_inv,
                  realized_vol, 0, price_momentum,
                  spread / mid * 100 if mid > 0 else 0,
                  now.hour + now.minute / 60.0, inv_skew,
                  round_trips))
            conn.commit()

            iv = yes_inv * mid + no_inv * (1 - mid)
            tp = realized_pnl + iv - yes_cost - no_cost
            ps = f"+${tp:.2f}" if tp >= 0 else f"-${abs(tp):.2f}"

            ybs2 = f"{our_yes_bid:>5.3f}" if our_yes_bid > 0 else f"{'—':>5s}"
            nbs2 = f"{our_no_bid:>5.3f}" if our_no_bid > 0 else f"{'—':>5s}"
            pps = f"{pp:>.3f}" if pp > 0 else f"{'—':>5s}"

            print(f"  {now.strftime('%H:%M:%S'):<10s} {state[:7]:<8s} {mid:>4.0%} {spread:>.3f} "
                  f"{pps:>5s} {ybs2} {nbs2} "
                  f"{yes_inv:>5.1f} {no_inv:>5.1f} {round_trips:>4d} {ps:>8s}")

            # ==== STEP 6: PLACE NEW ORDERS ====
            if execute and auth_client:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY

                if ybs >= 5 and our_yes_bid > 0:
                    try:
                        o = auth_client.create_order(
                            OrderArgs(price=our_yes_bid, size=ybs, side=BUY, token_id=yes_token))
                        r = auth_client.post_order(o, OrderType.GTC)
                        yes_oid = r.get("orderID", "")
                        if not yes_oid:
                            print(f"    ⚠ YES post returned no orderID: {r}")
                        else:
                            print(f"    📤 YES bid posted: {our_yes_bid:.3f} × {ybs:.0f}")
                        prev_yes_m = 0.0
                    except Exception as e:
                        print(f"    YES err: {e}")
                else:
                    if our_yes_bid > 0:
                        print(f"    ⚠ YES skipped: size={ybs:.1f} < 5 min")

                if nbs >= 5 and our_no_bid > 0:
                    try:
                        o = auth_client.create_order(
                            OrderArgs(price=our_no_bid, size=nbs, side=BUY, token_id=no_token))
                        r = auth_client.post_order(o, OrderType.GTC)
                        no_oid = r.get("orderID", "")
                        if not no_oid:
                            print(f"    ⚠ NO post returned no orderID: {r}")
                        else:
                            print(f"    📤 NO bid posted:  {our_no_bid:.3f} × {nbs:.0f}")
                        prev_no_m = 0.0
                    except Exception as e:
                        print(f"    NO err: {e}")
                else:
                    if our_no_bid > 0:
                        print(f"    ⚠ NO skipped: size={nbs:.1f} < 5 min")

        except KeyboardInterrupt:
            print(f"\n  Stopping...")
            break
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(REQUOTE_INTERVAL)

    # Summary
    ay = yes_cost / yes_inv if yes_inv > 0 else 0
    an = no_cost / no_inv if no_inv > 0 else 0
    iv = yes_inv * mid + no_inv * (1 - mid)

    print(f"\n{'─'*90}")
    print(f"  SESSION SUMMARY")
    print(f"{'─'*90}")
    print(f"  Duration:       {duration}m  |  Tick: {tick_size}")
    print(f"  Round trips:    {round_trips}")
    print(f"  Realized P&L:   ${realized_pnl:.4f}")
    print(f"  YES inv:        {yes_inv:.1f} (avg {ay:.3f})")
    print(f"  NO inv:         {no_inv:.1f} (avg {an:.3f})")
    print(f"  Inv value:      ${iv:.2f}")
    print(f"  Total cost:     ${yes_cost + no_cost:.2f}")
    tp = realized_pnl + iv - yes_cost - no_cost
    print(f"  Total P&L:      ${tp:.4f}")

    if mid_history:
        ms = [h[1] for h in mid_history]
        ss = [h[1] for h in spread_history]
        print(f"\n  Features: {len(ms)} samples, mid {min(ms):.3f}-{max(ms):.3f}, "
              f"spread {min(ss):.3f}-{max(ss):.3f} avg {sum(ss)/len(ss):.3f}")

    if execute and auth_client:
        try:
            if yes_oid: auth_client.cancel(order_id=yes_oid)
            if no_oid: auth_client.cancel(order_id=no_oid)
            print(f"  Cancelled remaining orders.")
        except Exception:
            pass

    conn.close()
    print(f"{'='*100}")


def cmd_flatten(args):
    """Flatten all inventory — dump positions at market to free capital.
    
    Cancels all open orders, then sells remaining inventory at best
    available price. Reports the exit cost vs. holding value.
    """
    slug = args.slug
    execute = getattr(args, "execute", False)
    mode = "🔴 LIVE FLATTEN" if execute else "🟡 DRY RUN (preview)"

    print(f"\n{'='*90}")
    print(f"  MARKET MAKER — {mode}")
    print(f"  Market: {slug}")
    print(f"{'='*90}")

    clob = get_clob_client()
    auth_client = get_auth_client() if execute else None

    # Resolve market
    r = requests.get(f"{GAMMA_API}/markets", params={"slug": slug})
    markets = r.json()
    if not markets:
        print(f"  ❌ Market not found")
        return

    m = markets[0]
    question = m.get("question", "")
    token_ids = json.loads(m.get("clobTokenIds", "[]"))
    yes_token = token_ids[0]
    no_token = token_ids[1] if len(token_ids) > 1 else ""

    print(f"  Market: {question[:70]}")

    # Get current prices
    mid = buy = sell = 0
    try:
        mid_r = clob.get_midpoint(yes_token)
        mid = float(mid_r.get("mid", 0)) if isinstance(mid_r, dict) else float(mid_r)
        buy_r = clob.get_price(yes_token, "BUY")
        buy = float(buy_r.get("price", 0)) if isinstance(buy_r, dict) else float(buy_r)
        sell_r = clob.get_price(yes_token, "SELL")
        sell = float(sell_r.get("price", 0)) if isinstance(sell_r, dict) else float(sell_r)
    except Exception as e:
        print(f"  Price error: {e}")

    spread = sell - buy if sell > 0 and buy > 0 else 0

    print(f"\n  Current Market:")
    print(f"    Mid: {mid:.3f}  |  Bid: {buy:.3f}  |  Ask: {sell:.3f}  |  Spread: {spread:.3f}")

    # Check what we actually hold via the Data API
    from dotenv import load_dotenv
    load_dotenv()
    wallet = os.getenv("POLYMARKET_WALLET", "")
    if not wallet:
        # Derive from private key
        try:
            from eth_account import Account
            pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            if pk:
                wallet = Account.from_key(pk).address
        except Exception:
            pass

    yes_shares = no_shares = 0
    yes_avg_price = no_avg_price = 0

    if wallet:
        try:
            r = requests.get(f"https://data-api.polymarket.com/positions",
                             params={"user": wallet.lower()}, timeout=10)
            positions = r.json()
            for pos in positions:
                asset = pos.get("asset", "")
                if asset == yes_token:
                    yes_shares = float(pos.get("size", 0))
                    yes_avg_price = float(pos.get("avgPrice", 0))
                elif asset == no_token:
                    no_shares = float(pos.get("size", 0))
                    no_avg_price = float(pos.get("avgPrice", 0))
        except Exception as e:
            print(f"  Position lookup error: {e}")

    if yes_shares == 0 and no_shares == 0:
        print(f"\n  ✅ No inventory to flatten. Position is already flat.")

        # Still check for open orders to cancel
        if execute and auth_client:
            try:
                orders = auth_client.get_orders()
                market_orders = [o for o in orders
                                 if o.get("asset_id", "") in [yes_token, no_token]]
                if market_orders:
                    for o in market_orders:
                        auth_client.cancel(order_id=o["id"])
                    print(f"  Cancelled {len(market_orders)} open orders.")
                else:
                    print(f"  No open orders to cancel.")
            except Exception as e:
                print(f"  Order check error: {e}")
        return

    # Calculate exit costs
    print(f"\n  Current Inventory:")

    total_cost = 0
    total_proceeds = 0
    exit_actions = []

    if yes_shares > 0:
        cost_basis = yes_shares * yes_avg_price
        # Selling YES shares = hitting the bid
        exit_price = buy  # Best bid is what we'd get
        proceeds = yes_shares * exit_price
        exit_cost = cost_basis - proceeds
        mark_value = yes_shares * mid

        print(f"    YES: {yes_shares:.1f} shares @ avg {yes_avg_price:.3f} "
              f"(cost ${cost_basis:.2f})")
        print(f"      Mark-to-market:  ${mark_value:.2f} (at mid {mid:.3f})")
        print(f"      Exit at bid:     ${proceeds:.2f} (at bid {buy:.3f})")
        print(f"      Exit cost:       ${exit_cost:.2f} "
              f"({exit_cost/cost_basis*100:.1f}% of cost)" if cost_basis > 0 else "")

        total_cost += cost_basis
        total_proceeds += proceeds
        exit_actions.append(("SELL", "YES", yes_token, yes_shares, exit_price))

    if no_shares > 0:
        cost_basis = no_shares * no_avg_price
        # Selling NO shares = effectively selling at (1 - ask) on YES side
        no_bid = 1 - sell if sell > 0 else 0
        exit_price = no_bid
        proceeds = no_shares * exit_price
        exit_cost = cost_basis - proceeds
        mark_value = no_shares * (1 - mid)

        print(f"    NO:  {no_shares:.1f} shares @ avg {no_avg_price:.3f} "
              f"(cost ${cost_basis:.2f})")
        print(f"      Mark-to-market:  ${mark_value:.2f} (at 1-mid {1-mid:.3f})")
        print(f"      Exit at bid:     ${proceeds:.2f} (at NO bid {exit_price:.3f})")
        print(f"      Exit cost:       ${exit_cost:.2f} "
              f"({exit_cost/cost_basis*100:.1f}% of cost)" if cost_basis > 0 else "")

        total_cost += cost_basis
        total_proceeds += proceeds
        exit_actions.append(("SELL", "NO", no_token, no_shares, exit_price))

    total_exit_cost = total_cost - total_proceeds

    print(f"\n  FLATTEN SUMMARY:")
    print(f"    Total cost basis:   ${total_cost:.2f}")
    print(f"    Total proceeds:     ${total_proceeds:.2f}")
    print(f"    Exit cost:          ${total_exit_cost:.2f} "
          f"({total_exit_cost/total_cost*100:.1f}% of cost)" if total_cost > 0 else "")
    print(f"    Capital freed:      ${total_proceeds:.2f}")

    if not execute:
        print(f"\n  Add --execute to flatten for real.")
        return

    # Execute flatten
    print(f"\n  Executing flatten...")

    # Step 1: Cancel all open orders
    try:
        orders = auth_client.get_orders()
        market_orders = [o for o in orders
                         if o.get("asset_id", "") in [yes_token, no_token]]
        if market_orders:
            for o in market_orders:
                auth_client.cancel(order_id=o["id"])
            print(f"  Cancelled {len(market_orders)} open orders.")
    except Exception as e:
        print(f"  Order cancel error: {e}")

    time.sleep(0.5)

    # Step 2: Market sell all inventory
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import SELL as SELL_SIDE

    for side_label, outcome, token_id, shares, price in exit_actions:
        try:
            print(f"  Selling {shares:.1f} {outcome} shares at market...")
            mo = MarketOrderArgs(
                token_id=token_id,
                amount=round(shares * price, 2),  # Amount in USDC
                side=SELL_SIDE,
                order_type=OrderType.FOK
            )
            signed = auth_client.create_market_order(mo)
            result = auth_client.post_order(signed, OrderType.FOK)
            status = result.get("status", "?")
            print(f"    {outcome}: {status}")
        except Exception as e:
            print(f"    {outcome} sell error: {e}")
            # Fallback: try limit order at bid
            try:
                from py_clob_client.clob_types import OrderArgs
                print(f"    Trying limit sell at {price:.2f}...")
                order = auth_client.create_order(
                    OrderArgs(price=price, size=round(shares, 2),
                              side=SELL_SIDE, token_id=token_id))
                result = auth_client.post_order(order)
                print(f"    {outcome}: {result.get('status', '?')}")
            except Exception as e2:
                print(f"    {outcome} limit sell error: {e2}")

        time.sleep(0.5)

    print(f"\n  ✅ Flatten complete. Capital freed: ~${total_proceeds:.2f}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Market Maker")
    parser.add_argument("--min-volume", type=float, default=5000)
    subs = parser.add_subparsers(dest="command", required=True)

    p_scan = subs.add_parser("scan", help="Scan for MM opportunities")
    p_scan.add_argument("--min-spread", type=float, default=0.02)
    p_scan.add_argument("--limit", type=int, default=30)

    p_quote = subs.add_parser("quote", help="Show proposed quotes")
    p_quote.add_argument("--slug", type=str, required=True)

    p_run = subs.add_parser("run", help="Run market maker")
    p_run.add_argument("--slug", type=str, required=True)
    p_run.add_argument("--execute", action="store_true")
    p_run.add_argument("--size", type=float, default=DEFAULT_QUOTE_SIZE)
    p_run.add_argument("--minutes", type=int, default=30)

    p_flat = subs.add_parser("flatten", help="Flatten inventory — dump at market")
    p_flat.add_argument("--slug", type=str, required=True)
    p_flat.add_argument("--execute", action="store_true")

    args = parser.parse_args()
    {"scan": cmd_scan, "quote": cmd_quote, "run": cmd_run,
     "flatten": cmd_flatten}[args.command](args)


if __name__ == "__main__":
    main()
