"""
scripts/sell_position.py — Sell an existing position on Polymarket CLOB.

Usage:
    python -m scripts.sell_position --slug will-annual-inflation-increase-by-34-in-march --side YES --price 0.32
    python -m scripts.sell_position --slug will-annual-inflation-increase-by-34-in-march --side YES --price 0.32 --execute
"""
import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
load_dotenv()

GAMMA_API = "https://gamma-api.polymarket.com"


def main():
    parser = argparse.ArgumentParser(description="Sell a Polymarket position")
    parser.add_argument("--slug", required=False, default="")
    parser.add_argument("--token-id", required=False, default="", help="Direct token ID (bypasses slug)")
    parser.add_argument("--side", required=True, choices=["YES", "NO"])
    parser.add_argument("--price", type=float, required=True, help="Limit price to sell at")
    parser.add_argument("--shares", type=float, default=0, help="Shares to sell (0=all)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("  ❌ No POLYMARKET_PRIVATE_KEY"); return

    # Get token ID either from slug or directly
    token_id = args.token_id
    question = "(direct token ID)"
    tick_size = 0.01
    neg_risk = False

    if not token_id:
        if not args.slug:
            print("  ❌ Must provide --slug or --token-id"); return

        r = requests.get(f"{GAMMA_API}/markets", params={"slug": args.slug}).json()
        if not r:
            print(f"  ❌ Market not found: {args.slug}"); return
        m = r[0]
        question = m.get("question", "")
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if len(token_ids) < 2:
            print(f"  ❌ No token IDs"); return
        yes_token = token_ids[0]
        no_token = token_ids[1]
        token_id = yes_token if args.side == "YES" else no_token
        tick_size = float(m.get("orderPriceMinTickSize", 0.01) or 0.01)
        neg_risk = m.get("negRisk", False)

    print(f"\n{'='*70}")
    print(f"  SELL POSITION")
    print(f"{'='*70}")
    print(f"  Market:   {question}")
    print(f"  Side:     {args.side}")
    print(f"  Token:    {token_id[:30]}...")
    print(f"  NegRisk:  {neg_risk}")
    print(f"  Tick:     {tick_size}")

    # Check on-chain balance
    from web3 import Web3
    CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    CTF_ABI = json.loads('[{"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]')

    w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))
    wallet = w3.eth.account.from_key(pk).address
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=CTF_ABI)

    balance_raw = ctf.functions.balanceOf(wallet, int(token_id)).call()
    balance = balance_raw / 1e6
    print(f"  Balance:  {balance:.6f} {args.side} shares")

    if balance <= 0:
        print(f"  ❌ No shares to sell"); return

    sell_shares = args.shares if args.shares > 0 else balance
    sell_shares = min(sell_shares, balance)
    proceeds = sell_shares * args.price

    print(f"  Selling:  {sell_shares:.1f} shares @ {args.price:.3f}")
    print(f"  Expected: ${proceeds:.2f}")

    if not args.execute:
        print(f"\n  Dry run. Add --execute to sell.")
        return

    # Initialize CLOB client
    chain_id = 137
    clob = ClobClient(
        "https://clob.polymarket.com",
        key=pk,
        chain_id=chain_id,
    )

    # Get API creds
    try:
        clob.set_api_creds(clob.create_or_derive_api_creds())
    except Exception:
        clob.set_api_creds(clob.derive_api_key())

    print(f"\n  Posting SELL order...")

    try:
        order_args = OrderArgs(
            price=args.price,
            size=sell_shares,
            side=SELL,
            token_id=token_id,
        )

        signed = clob.create_and_post_order(order_args)
        print(f"  📤 Order posted: {signed}")

        # Wait and check
        time.sleep(2)
        orders = clob.get_orders()
        open_orders = [o for o in orders if o.get("status") == "LIVE"]
        print(f"  Open orders: {len(open_orders)}")

        for o in open_orders:
            oid = o.get("id", "?")
            side = o.get("side", "?")
            price = o.get("price", "?")
            size = o.get("original_size", "?")
            filled = o.get("size_matched", "0")
            print(f"    {oid[:12]}... {side} {size} @ {price} (filled: {filled})")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
