"""
scripts/sell_inflation34.py — Sell the ≥3.4% annual inflation YES position.

Self-contained: finds the token ID from your positions, verifies balance,
and sells at specified price. No copy/paste needed.

Usage:
    python -m scripts.sell_inflation34              # Dry run
    python -m scripts.sell_inflation34 --execute    # Sell
    python -m scripts.sell_inflation34 --price 0.30 --execute  # Sell at 30c
"""
import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", type=float, default=0.31)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from web3 import Web3

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("  No POLYMARKET_PRIVATE_KEY"); return

    w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))
    wallet = w3.eth.account.from_key(pk).address

    # Step 1: Find the position
    print(f"\n{'='*70}")
    print(f"  SELL ≥3.4% ANNUAL INFLATION POSITION")
    print(f"{'='*70}")
    print(f"  Finding position...")

    r = requests.get("https://data-api.polymarket.com/positions",
                     params={"user": wallet}, timeout=10).json()

    token_id = None
    size = 0
    for p in r:
        title = p.get("title", p.get("market", {}).get("question", "?"))
        if "3.4" in str(title):
            token_id = p.get("asset", "")
            size = float(p.get("size", 0))
            side = p.get("outcome", "?")
            print(f"  Found: {title}")
            print(f"  Side:  {side}")
            print(f"  Size:  {size:.1f} shares")
            print(f"  Token: {token_id}")
            break

    if not token_id:
        print("  Position not found!")
        return

    # Step 2: Verify on-chain balance
    CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    CTF_ABI = json.loads('[{"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]')
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=CTF_ABI)

    balance_raw = ctf.functions.balanceOf(wallet, int(token_id)).call()
    balance = balance_raw / 1e6
    print(f"  On-chain: {balance:.1f} shares")

    if balance <= 0:
        print("  No shares on-chain!")
        return

    proceeds = balance * args.price
    print(f"\n  Sell {balance:.1f} shares @ {args.price:.3f} = ~${proceeds:.2f}")

    if not args.execute:
        print(f"\n  DRY RUN. Add --execute to sell.")
        return

    # Step 3: Sell via CLOB
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import SELL

    clob = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137)
    try:
        clob.set_api_creds(clob.create_or_derive_api_creds())
    except Exception:
        clob.set_api_creds(clob.derive_api_key())

    print(f"\n  Posting SELL order...")
    try:
        order_args = OrderArgs(
            price=args.price,
            size=balance,
            side=SELL,
            token_id=token_id,
        )
        result = clob.create_and_post_order(order_args)
        print(f"  Order posted: {result}")

        time.sleep(2)

        # Check if filled
        orders = clob.get_orders()
        live = [o for o in orders if o.get("status") == "LIVE"]
        matched = [o for o in orders if float(o.get("size_matched", 0)) > 0]

        if live:
            print(f"\n  {len(live)} order(s) still open:")
            for o in live:
                print(f"    {o.get('side','')} {o.get('original_size','')} "
                      f"@ {o.get('price','')} (filled: {o.get('size_matched','0')})")
        else:
            print(f"  Order filled or not found in open orders.")

        # Check new balance
        time.sleep(1)
        new_balance = ctf.functions.balanceOf(wallet, int(token_id)).call() / 1e6
        print(f"\n  Balance after: {new_balance:.1f} (was {balance:.1f})")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
