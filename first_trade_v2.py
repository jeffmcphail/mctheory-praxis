"""
first_trade_v2.py — Place first Polymarket weather trade.

Uses derive_api_key() directly — no separate setup step needed.

Usage:
    python first_trade_v2.py check
    python first_trade_v2.py find
    python first_trade_v2.py trade --amount 5
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
    """Get authenticated CLOB client using derived credentials."""
    from py_clob_client.client import ClobClient

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("❌ POLYMARKET_PRIVATE_KEY not in .env")
        sys.exit(1)

    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID)
    client.set_api_creds(client.derive_api_key())
    return client


def cmd_check(args):
    """Check connection and balance."""
    print("\n  Connecting to Polymarket CLOB...")
    client = get_client()

    try:
        # Try to get open orders (proves auth works)
        orders = client.get_orders()
        print(f"  Open orders: {len(orders) if isinstance(orders, list) else orders}")
    except Exception as e:
        print(f"  Orders check: {e}")

    try:
        bal = client.get_balance_allowance()
        print(f"  Balance/Allowance: {bal}")
    except Exception as e:
        print(f"  Balance check: {e}")

    print(f"\n  ✅ Connection check complete")


def cmd_find(args):
    """Find weather markets with edge."""
    import requests

    print("\n  Searching for weather markets...")

    # Try multiple LA slugs
    event = None
    for slug in [
        "highest-temperature-in-los-angeles-on-april-6-2026",
        "highest-temperature-in-la-on-april-6-2026",
        "highest-temperature-in-los-angeles-on-april-7-2026",
    ]:
        r = requests.get("https://gamma-api.polymarket.com/events",
                         params={"slug": slug})
        events = r.json()
        if events:
            event = events[0]
            break

    if not event:
        # Broader search
        r = requests.get("https://gamma-api.polymarket.com/events", params={
            "tag_slug": "weather", "limit": "100", "order": "startDate",
            "ascending": "false", "active": "true", "closed": "false",
        })
        all_evts = r.json()
        la = [e for e in all_evts if "los angeles" in e.get("title", "").lower()]
        if la:
            event = la[0]

    if not event:
        print("  ❌ No LA weather market found")
        return

    print(f"\n  {event['title']}")
    markets = event.get("markets", [])
    print(f"  Buckets: {len(markets)}\n")

    target_market = None
    for m in markets:
        title = m.get("groupItemTitle", m.get("question", "?"))
        prices = json.loads(m.get("outcomePrices", "[0,0]"))
        yes_price = float(prices[0])
        volume = float(m.get("volume", 0))
        token_ids = m.get("clobTokenIds", "")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except:
                token_ids = []

        marker = ""
        if "76" in title and "77" in title:
            marker = " ← TARGET (46% edge)"
            target_market = m

        print(f"  {title:<20s} YES={yes_price:.2f}  vol=${volume:>8,.0f}{marker}")
        if marker and token_ids:
            print(f"    YES token: {token_ids[0]}")
            if len(token_ids) > 1:
                print(f"    NO token:  {token_ids[1]}")

    # Save for trading
    os.makedirs("data", exist_ok=True)
    with open("data/target_market.json", "w") as f:
        json.dump({"event": event, "target": target_market}, f, indent=2)
    print(f"\n  Saved: data/target_market.json")


def cmd_trade(args):
    """Place a trade."""
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    if not os.path.exists("data/target_market.json"):
        print("  ❌ Run 'python first_trade_v2.py find' first")
        return

    with open("data/target_market.json") as f:
        data = json.load(f)

    event = data["event"]
    target = data.get("target")

    if not target:
        print("  ❌ No target market found. Run 'find' again.")
        return

    title = target.get("groupItemTitle", target.get("question", "?"))
    prices = json.loads(target.get("outcomePrices", "[0,0]"))
    yes_price = float(prices[0])
    token_ids = target.get("clobTokenIds", "")
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)

    yes_token_id = token_ids[0] if token_ids else None
    if not yes_token_id:
        print("  ❌ No token ID")
        return

    amount = args.amount
    size = round(amount / yes_price, 2)

    print(f"\n{'='*60}")
    print(f"  TRADE DETAILS")
    print(f"{'='*60}")
    print(f"  Event:   {event['title']}")
    print(f"  Bucket:  {title}")
    print(f"  Action:  BUY YES")
    print(f"  Price:   ${yes_price:.2f}")
    print(f"  Spend:   ${amount:.2f}")
    print(f"  Shares:  {size:.1f}")
    print(f"  Payout:  ${size:.2f} if correct (${size - amount:.2f} profit)")
    print(f"{'='*60}")

    confirm = input(f"\n  Type 'YES' to execute: ")
    if confirm != "YES":
        print("  Cancelled.")
        return

    print(f"\n  Placing order...")
    client = get_client()

    try:
        order_args = OrderArgs(
            price=yes_price,
            size=size,
            side=BUY,
            token_id=yes_token_id,
        )
        signed_order = client.create_order(order_args)
        result = client.post_order(signed_order)

        print(f"\n  ✅ ORDER PLACED!")
        print(f"  {json.dumps(result, indent=2)}")

        # Log
        trade = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event["title"],
            "bucket": title,
            "side": "BUY_YES",
            "price": yes_price,
            "amount": amount,
            "size": size,
            "result": result,
        }
        log_path = "data/weather_trades.json"
        trades = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                trades = json.load(f)
        trades.append(trade)
        with open(log_path, "w") as f:
            json.dump(trades, f, indent=2)
        print(f"  Logged: {log_path}")

    except Exception as e:
        print(f"\n  ❌ Order failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("check")
    subs.add_parser("find")
    p = subs.add_parser("trade")
    p.add_argument("--amount", type=float, default=5.0)
    args = parser.parse_args()
    {"check": cmd_check, "find": cmd_find, "trade": cmd_trade}[args.command](args)

if __name__ == "__main__":
    main()
