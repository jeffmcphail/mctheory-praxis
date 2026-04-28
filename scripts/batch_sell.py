"""
scripts/batch_sell.py -- Sell multiple underperforming positions

Reads your portfolio, identifies positions to dump, and sells them.
Uses the actuarial engine's logic to find positions below risk-free rate.

Usage:
    python -m scripts.batch_sell --list                    # Show what would be sold
    python -m scripts.batch_sell --weather                 # Sell all weather bets
    python -m scripts.batch_sell --below-rf                # Sell everything below risk-free rate
    python -m scripts.batch_sell --slugs "slug1,slug2"     # Sell specific markets
    python -m scripts.batch_sell --weather --execute       # Actually execute sells
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
load_dotenv()

DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
CLOB_HOST = "clob.polymarket.com"

POLYMARKET_FEE = 0.02
RISK_FREE_RATE = 0.045


def get_wallet():
    from web3 import Web3
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("  No POLYMARKET_PRIVATE_KEY in .env")
        sys.exit(1)
    w3 = Web3()
    return w3.eth.account.from_key(pk), pk


def get_positions():
    acct, _ = get_wallet()
    r = requests.get(f"{DATA_API}/positions",
                     params={"user": acct.address}, timeout=15)
    return r.json()


def create_sell_order(token_id, size, price, pk):
    """Create and submit a sell order via CLOB API."""
    from py_clob_client.client import ClobClient

    chain_id = 137
    client = ClobClient(
        CLOB_API,
        key=pk,
        chain_id=chain_id,
    )

    # Create a market sell (limit at slightly below current price for fill)
    sell_price = round(max(0.01, price - 0.02), 2)  # Sell 2c below mid for fast fill

    try:
        order = client.create_and_post_order(
            token_id=token_id,
            side="SELL",
            size=size,
            price=sell_price,
        )
        return order
    except Exception as e:
        return {"error": str(e)}


def cmd_list(positions, filter_fn, label):
    """List positions that match the filter."""
    matches = [p for p in positions if filter_fn(p)]

    if not matches:
        print(f"\n  No positions match filter: {label}")
        return matches

    total_value = 0
    print(f"\n  Positions to sell ({label}):")
    print(f"  {'Market':<50s} {'Side':<4s} {'Size':>6s} {'Price':>6s} {'Value':>7s}")
    print(f"  {'-'*80}")

    for p in matches:
        title = str(p.get("title", p.get("market", {}).get("question", "?")))[:49]
        outcome = p.get("outcome", "?")
        size = float(p.get("size", 0) or 0)
        cur_price = float(p.get("curPrice", p.get("currentPrice", 0)) or 0)
        value = size * cur_price

        total_value += value
        print(f"  {title:<50s} {outcome:<4s} {size:>6.1f} {cur_price:>5.0%} ${value:>6.0f}")

    print(f"  {'-'*80}")
    print(f"  Total value to recover: ${total_value:.0f}")
    print(f"  (Actual recovery will be slightly less due to spread)")

    return matches


def is_weather(p):
    title = str(p.get("title", p.get("market", {}).get("question", ""))).lower()
    return "temperature" in title or "weather" in title


def is_below_rf(p):
    """Position earning below risk-free rate."""
    title = str(p.get("title", p.get("market", {}).get("question", ""))).lower()
    # Weather bets + resolved/near-resolved with low value
    if "temperature" in title or "weather" in title:
        return True
    # Monthly CPI that already resolved
    if "monthly inflation" in title and "march" in title:
        return True
    # 76ers (season ending)
    if "76ers" in title or "philadelphia" in title:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch sell underperforming positions")
    parser.add_argument("--list", action="store_true", help="List only, don't sell")
    parser.add_argument("--weather", action="store_true", help="Sell all weather bets")
    parser.add_argument("--below-rf", action="store_true", help="Sell below risk-free rate")
    parser.add_argument("--slugs", type=str, help="Comma-separated slugs to sell")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute sells (without this, just shows what would happen)")
    parser.add_argument("--min-value", type=float, default=1.0,
                        help="Skip positions worth less than this (default $1)")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  BATCH POSITION SELL")
    print(f"{'='*80}")

    positions = get_positions()
    if not positions:
        print("  No positions found.")
        return

    print(f"  Found {len(positions)} positions")

    # Determine filter
    if args.weather:
        matches = cmd_list(positions, is_weather, "weather bets")
    elif args.below_rf:
        matches = cmd_list(positions, is_below_rf, "below risk-free rate")
    elif args.slugs:
        slug_set = set(args.slugs.split(","))
        matches = cmd_list(positions,
                           lambda p: p.get("slug", "") in slug_set,
                           f"slugs: {args.slugs}")
    else:
        # Default: show everything
        matches = cmd_list(positions, lambda p: True, "all positions")
        print(f"\n  Use --weather, --below-rf, or --slugs to filter")
        print(f"  Add --execute to actually sell")
        return

    if not matches:
        return

    if not args.execute:
        print(f"\n  DRY RUN -- add --execute to actually sell")
        print(f"  Example: python -m scripts.batch_sell --weather --execute")
        return

    # Execute sells
    _, pk = get_wallet()
    print(f"\n  Executing sells...")

    sold = 0
    failed = 0
    total_recovered = 0

    for p in matches:
        title = str(p.get("title", p.get("market", {}).get("question", "?")))[:40]
        size = float(p.get("size", 0) or 0)
        cur_price = float(p.get("curPrice", p.get("currentPrice", 0)) or 0)
        value = size * cur_price

        if value < args.min_value:
            print(f"  SKIP {title} (value ${value:.2f} < min ${args.min_value})")
            continue

        # Get token ID
        token_id = p.get("tokenId", p.get("token_id", ""))
        if not token_id:
            # Try to extract from asset field
            asset = p.get("asset", "")
            if asset:
                token_id = asset
            else:
                print(f"  SKIP {title} (no token ID)")
                failed += 1
                continue

        print(f"  SELLING {title}...")
        print(f"    Size: {size:.1f} | Price: {cur_price:.2f} | Value: ${value:.0f}")

        result = create_sell_order(token_id, size, cur_price, pk)

        if isinstance(result, dict) and "error" in result:
            print(f"    FAILED: {result['error']}")
            failed += 1
        else:
            print(f"    OK: {result}")
            sold += 1
            total_recovered += value

        time.sleep(1)  # Rate limit

    print(f"\n  Results: {sold} sold, {failed} failed")
    print(f"  Estimated recovery: ${total_recovered:.0f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
