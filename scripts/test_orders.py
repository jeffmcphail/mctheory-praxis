"""
scripts/test_orders.py — Diagnostic: post a tiny order and verify it's visible.

Usage:
    python -m scripts.test_orders
"""
import os
import sys
import time
import json

from dotenv import load_dotenv
load_dotenv()


def main():
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("❌ No key")
        sys.exit(1)

    client = ClobClient(
        "https://clob.polymarket.com",
        key=pk, chain_id=137, signature_type=0
    )
    client.set_api_creds(client.derive_api_key())

    # Use the Fed rate cut YES token
    import requests
    r = requests.get("https://gamma-api.polymarket.com/markets",
                      params={"slug": "fed-rate-cut-by-october-2026-meeting-199-747"})
    m = r.json()[0]
    token_ids = json.loads(m.get("clobTokenIds", "[]"))
    yes_token = token_ids[0]

    print(f"Token: {yes_token[:30]}...")

    # Step 1: Check current orders
    print(f"\n--- Before posting ---")
    orders_before = client.get_orders()
    print(f"get_orders() type: {type(orders_before)}")
    print(f"get_orders() value: {orders_before}")

    # Step 2: Post a tiny order far from market
    print(f"\n--- Posting test order ---")
    try:
        order = client.create_order(
            OrderArgs(price=0.100, size=5.0, side=BUY, token_id=yes_token))
        print(f"create_order returned: {type(order)}")
        print(f"  {order}")

        result = client.post_order(order, OrderType.GTC)
        print(f"post_order returned: {type(result)}")
        print(f"  {result}")

        order_id = result.get("orderID", "") if isinstance(result, dict) else ""
        print(f"  orderID: '{order_id}'")

    except Exception as e:
        print(f"  Post error: {e}")
        return

    # Step 3: Wait and query
    print(f"\n--- Checking 1 second after posting ---")
    time.sleep(1)
    orders_after = client.get_orders()
    print(f"get_orders() type: {type(orders_after)}")
    print(f"get_orders() count: {len(orders_after) if isinstance(orders_after, list) else 'N/A'}")
    if isinstance(orders_after, list):
        for o in orders_after:
            print(f"  {o}")

    # Step 4: Try get_order with the specific ID
    if order_id:
        print(f"\n--- Trying get_order('{order_id[:20]}...') ---")
        try:
            single = client.get_order(order_id)
            print(f"  Result: {single}")
        except Exception as e:
            print(f"  Error: {e}")

    # Step 5: Cancel
    if order_id:
        print(f"\n--- Cancelling ---")
        try:
            client.cancel(order_id=order_id)
            print(f"  Cancelled OK")
        except Exception as e:
            print(f"  Cancel error: {e}")

    # Step 6: Verify cancelled
    time.sleep(1)
    orders_final = client.get_orders()
    print(f"\n--- After cancel ---")
    print(f"get_orders() count: {len(orders_final) if isinstance(orders_final, list) else 'N/A'}")


if __name__ == "__main__":
    main()
