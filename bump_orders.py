"""
bump_orders.py — Cancel unfilled orders and resubmit at higher price.

Cancels orders that haven't filled and resubmits with a price bump
to cross the spread and get immediate fills.

Usage:
    python bump_orders.py                  # Show what would be bumped
    python bump_orders.py --execute        # Cancel and resubmit
    python bump_orders.py --bump 0.02      # Bump by 2 cents (default: 1 cent)
    python bump_orders.py --only-favorites # Only bump favorites
    python bump_orders.py --only-unfilled  # Only fully unfilled (skip partials)
"""
import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


def get_client():
    from py_clob_client.client import ClobClient
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=0)
    client.set_api_creds(client.derive_api_key())
    return client


def main():
    parser = argparse.ArgumentParser(description="Bump unfilled orders")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--bump", type=float, default=0.01,
                        help="Price bump in dollars (default: $0.01)")
    parser.add_argument("--only-favorites", action="store_true",
                        help="Only bump favorite trades")
    parser.add_argument("--only-unfilled", action="store_true",
                        help="Only bump fully unfilled orders (skip partials)")
    parser.add_argument("--min-unfilled-pct", type=float, default=0.50,
                        help="Min unfilled percentage to bump (default: 50%%)")
    args = parser.parse_args()

    print(f"\n{'='*85}")
    print(f"ORDER BUMP — Cancel & Resubmit at Higher Price")
    print(f"  Bump: +${args.bump:.3f} per share")
    print(f"  Mode: {'🔴 LIVE' if args.execute else '📋 DRY RUN'}")
    print(f"{'='*85}")

    client = get_client()

    # Get current orders
    try:
        orders = client.get_orders()
        if not isinstance(orders, list):
            orders = []
    except Exception as e:
        print(f"  ❌ Could not fetch orders: {e}")
        return

    print(f"\n  Found {len(orders)} open orders")

    # Load trade logs to identify trade types
    trade_types = {}
    for path, ttype in [("data/weather_trades.json", "weather"),
                         ("data/favorites_trades.json", "favorites"),
                         ("data/longshot_trades.json", "longshot")]:
        if os.path.exists(path):
            with open(path) as f:
                for t in json.load(f):
                    oid = t.get("order_id", "")
                    if not oid:
                        r = t.get("result", {})
                        if isinstance(r, dict):
                            oid = r.get("orderID", "")
                    if oid:
                        trade_types[oid] = {
                            "type": ttype,
                            "name": t.get("name", t.get("bucket", t.get("question", "?"))),
                        }

    # Filter orders to bump
    to_bump = []
    for o in orders:
        oid = o.get("id", o.get("orderID", ""))
        status = o.get("status", "")
        orig_size = float(o.get("original_size", o.get("size", 0)))
        matched = float(o.get("size_matched", 0))
        price = float(o.get("price", 0))
        remaining = orig_size - matched
        fill_pct = matched / orig_size if orig_size > 0 else 0
        unfilled_pct = 1 - fill_pct

        if status != "LIVE":
            continue

        # Skip if mostly filled
        if unfilled_pct < args.min_unfilled_pct:
            continue

        if args.only_unfilled and matched > 0:
            continue

        info = trade_types.get(oid, {"type": "unknown", "name": "?"})

        if args.only_favorites and info["type"] != "favorites":
            continue

        new_price = min(round(price + args.bump, 3), 0.99)
        new_size = round(remaining * price / new_price, 2)  # Adjust size to keep same spend
        cost_now = remaining * price
        cost_new = new_size * new_price

        to_bump.append({
            "order_id": oid,
            "name": info["name"][:40],
            "type": info["type"],
            "old_price": price,
            "new_price": new_price,
            "orig_size": orig_size,
            "matched": matched,
            "remaining": remaining,
            "new_size": new_size,
            "fill_pct": fill_pct,
            "token_id": o.get("asset_id", o.get("token_id", "")),
            "side": o.get("side", "BUY"),
        })

    if not to_bump:
        print(f"\n  No orders to bump.")
        return

    # Display plan
    print(f"\n  Orders to bump: {len(to_bump)}")
    print(f"\n  {'Name':<40s} {'Type':<10s} {'Old$':>6s} {'New$':>6s} "
          f"{'Remain':>8s} {'NewSize':>8s} {'Filled':>6s}")
    print(f"  {'-'*85}")

    for b in to_bump:
        print(f"  {b['name']:<40s} {b['type']:<10s} ${b['old_price']:>5.3f} ${b['new_price']:>5.3f} "
              f"{b['remaining']:>7.1f}sh {b['new_size']:>7.1f}sh {b['fill_pct']:>5.0%}")

    if not args.execute:
        print(f"\n  Add --execute to cancel and resubmit.")
        return

    confirm = input(f"\n  Cancel {len(to_bump)} orders and resubmit at higher prices? (YES/no): ")
    if confirm != "YES":
        print("  Cancelled.")
        return

    # Execute: cancel then resubmit each
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    succeeded = 0
    failed = 0

    for b in to_bump:
        name = b["name"][:35]
        print(f"\n  [{name}] Cancelling...")

        # Cancel
        try:
            cancel_result = client.cancel(order_id=b["order_id"])
            print(f"    Cancelled: {cancel_result}")
        except Exception as e:
            # Try alternate cancel method
            try:
                cancel_result = client.cancel_order(b["order_id"])
                print(f"    Cancelled: {cancel_result}")
            except Exception as e2:
                print(f"    ❌ Cancel failed: {e2}")
                failed += 1
                continue

        time.sleep(0.5)

        # Resubmit at new price
        print(f"    Resubmitting at ${b['new_price']:.3f}...")
        try:
            order_args = OrderArgs(
                price=round(b["new_price"], 2),
                size=b["new_size"],
                side=BUY,
                token_id=b["token_id"],
            )
            signed = client.create_order(order_args)
            result = client.post_order(signed)
            status = result.get("status", "?")
            new_oid = result.get("orderID", "?")[:16]
            print(f"    ✅ {status} (ID: {new_oid}...)")
            succeeded += 1
        except Exception as e:
            print(f"    ❌ Resubmit failed: {str(e)[:100]}")
            failed += 1

        time.sleep(0.5)

    print(f"\n{'='*85}")
    print(f"  RESULTS: {succeeded} bumped, {failed} failed")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
