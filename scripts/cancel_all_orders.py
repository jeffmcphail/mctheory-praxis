"""
scripts/cancel_all_orders.py — Find and cancel ALL open Polymarket orders.

Usage:
    python -m scripts.cancel_all_orders              # Show open orders (dry run)
    python -m scripts.cancel_all_orders --execute    # Cancel everything
"""
import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Cancel all open Polymarket orders")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from py_clob_client.client import ClobClient

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("❌ POLYMARKET_PRIVATE_KEY not set in .env")
        sys.exit(1)

    client = ClobClient(
        "https://clob.polymarket.com",
        key=pk, chain_id=137, signature_type=0
    )
    client.set_api_creds(client.derive_api_key())

    print(f"Fetching open orders...")

    try:
        orders = client.get_orders()
    except Exception as e:
        print(f"❌ Error fetching orders: {e}")
        sys.exit(1)

    # Show raw response first
    print(f"\nRaw get_orders() returned: {type(orders)}")
    if isinstance(orders, list):
        print(f"  Count: {len(orders)}")
        for i, o in enumerate(orders[:5]):
            print(f"  [{i}] {o}")
    elif isinstance(orders, dict):
        print(f"  Keys: {list(orders.keys())}")
        print(f"  Raw: {str(orders)[:500]}")
    else:
        print(f"  Value: {orders}")

    # Try without filtering
    if not orders:
        print(f"\n  Empty response. Trying alternative approaches...")
        
        # Try get_order with no params
        try:
            r2 = client.get_orders()
            print(f"  get_orders() again: {r2}")
        except Exception as e:
            print(f"  Error: {e}")

        print(f"\n✅ No orders found via API.")
        return

    # Show all orders regardless of status
    all_orders = orders if isinstance(orders, list) else []

    if not all_orders:
        print(f"\n✅ No orders found.")
        return

    print(f"\nFound {len(all_orders)} order(s):\n")
    print(f"  {'#':<4s} {'Status':<10s} {'Side':<5s} {'Price':>7s} {'Size':>8s} {'Matched':>8s} {'Outcome':<6s} {'Token':>24s}")
    print(f"  {'─'*85}")

    for i, o in enumerate(all_orders, 1):
        status = o.get("status", "?")
        side = o.get("side", "?")
        price = o.get("price", "?")
        size = o.get("original_size", o.get("size", "?"))
        matched = o.get("size_matched", "0")
        outcome = o.get("outcome", "?")
        token = o.get("asset_id", "")[:22]
        print(f"  {i:<4d} {status:<10s} {side:<5s} {price:>7s} {size:>8s} {matched:>8s} {outcome:<6s} {token:>24s}")

    if not args.execute:
        print(f"\n  Dry run. Add --execute to cancel all.")
        return

    print(f"\n  Cancelling {len(all_orders)} orders...")
    cancelled = 0
    for o in all_orders:
        oid = o.get("id", "")
        status = o.get("status", "")
        try:
            client.cancel(order_id=oid)
            cancelled += 1
            print(f"  ✅ Cancelled {oid[:20]}... ({status})")
        except Exception as e:
            print(f"  ❌ Failed {oid[:20]}...: {e}")

    print(f"\n  Cancelled {cancelled}/{len(all_orders)} orders.")

    # Also try cancel_all if available
    try:
        client.cancel_all()
        print(f"  Also called cancel_all() as safety net.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
