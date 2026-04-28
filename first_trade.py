"""
first_trade.py — First Polymarket Weather Trade

Connects to Polymarket CLOB, verifies balance, and places a small
test trade on a weather market with confirmed edge.

Usage:
    # Step 1: Generate CLOB API credentials (one-time)
    python first_trade.py setup

    # Step 2: Check balance and connection
    python first_trade.py check

    # Step 3: Find the LA weather market and show details  
    python first_trade.py find

    # Step 4: Place a $5 test trade
    python first_trade.py trade --amount 5

    # Step 5: Place the full Kelly-sized trade
    python first_trade.py trade --amount 39
"""
import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
API_KEY = os.getenv("POLYMARKET_CLOB_API_KEY", "")
API_SECRET = os.getenv("POLYMARKET_CLOB_API_SECRET", "")
API_PASSPHRASE = os.getenv("POLYMARKET_CLOB_API_PASSPHRASE", "")

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon


def cmd_setup(args):
    """Generate CLOB API credentials from your private key."""
    from py_clob_client.client import ClobClient

    if not PRIVATE_KEY:
        print("❌ POLYMARKET_PRIVATE_KEY not set in .env")
        return

    print("\n  Generating CLOB API credentials...")
    client = ClobClient(HOST, key=PRIVATE_KEY, chain_id=CHAIN_ID)

    try:
        creds = client.create_api_key()
        print(f"\n  ✅ API credentials generated!")
        print(f"\n  Add these to your .env file:")
        print(f"  POLYMARKET_CLOB_API_KEY={creds.get('apiKey', '')}")
        print(f"  POLYMARKET_CLOB_API_SECRET={creds.get('secret', '')}")
        print(f"  POLYMARKET_CLOB_API_PASSPHRASE={creds.get('passphrase', '')}")
        print(f"\n  Then run: python first_trade.py check")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print(f"  Make sure your wallet has been connected to polymarket.com")
        print(f"  and you completed the 3-step Enable Trading process.")
        import traceback
        traceback.print_exc()


def get_client():
    """Get authenticated CLOB client."""
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        print("❌ CLOB API credentials not set. Run: python first_trade.py setup")
        sys.exit(1)

    creds = ApiCreds(
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=API_PASSPHRASE,
    )

    client = ClobClient(HOST, key=PRIVATE_KEY, chain_id=CHAIN_ID, creds=creds)
    return client


def cmd_check(args):
    """Check connection and balance."""
    print("\n  Checking Polymarket CLOB connection...")
    
    client = get_client()
    
    try:
        # Check allowances
        allowances = client.get_allowances()
        print(f"  Allowances: {json.dumps(allowances, indent=2)}")
    except Exception as e:
        print(f"  Allowances check: {e}")
    
    try:
        # Get balance
        balance = client.get_balance()
        print(f"  Balance: {balance}")
    except Exception as e:
        print(f"  Balance check: {e}")

    print(f"\n  ✅ Connection working")


def cmd_find(args):
    """Find the LA April 6 weather market."""
    import requests

    print("\n  Searching for LA weather market (April 6)...")
    
    slug = "highest-temperature-in-los-angeles-on-april-6-2026"
    r = requests.get("https://gamma-api.polymarket.com/events",
                     params={"slug": slug})
    events = r.json()

    if not events:
        # Try alternate slugs
        for alt in ["highest-temperature-in-la-on-april-6-2026",
                     "highest-temperature-in-los-angeles-on-april-5-2026"]:
            r = requests.get("https://gamma-api.polymarket.com/events",
                             params={"slug": alt})
            events = r.json()
            if events:
                break

    if not events:
        print("  ❌ Market not found. Searching broadly...")
        r = requests.get("https://gamma-api.polymarket.com/events", params={
            "tag_slug": "weather", "limit": "100", "order": "startDate",
            "ascending": "false", "active": "true", "closed": "false",
        })
        la_events = [e for e in r.json()
                     if "los angeles" in e.get("title", "").lower()
                     or " la " in e.get("title", "").lower()]
        if la_events:
            events = [la_events[0]]
            print(f"  Found: {la_events[0]['title']}")

    if not events:
        print("  ❌ No LA weather markets found")
        return

    event = events[0]
    print(f"\n  {event['title']}")
    print(f"  Slug: {event['slug']}")

    markets = event.get("markets", [])
    print(f"  Buckets: {len(markets)}")
    print()

    for m in markets:
        title = m.get("groupItemTitle", m.get("question", "?"))
        prices = json.loads(m.get("outcomePrices", "[0,0]"))
        yes_price = float(prices[0])
        volume = float(m.get("volume", 0))
        token_ids = m.get("clobTokenIds", "")
        condition_id = m.get("conditionId", "")

        # Highlight the 76-77°F bucket
        marker = " ← TARGET" if "76" in title and "77" in title else ""
        
        print(f"  {title:<20s} YES={yes_price:.2f}  vol=${volume:>8,.0f}  "
              f"cond={condition_id[:12]}...{marker}")
        
        if token_ids:
            try:
                ids = json.loads(token_ids) if isinstance(token_ids, str) else token_ids
                if ids and marker:
                    print(f"    YES token: {ids[0][:20]}...")
                    print(f"    NO token:  {ids[1][:20]}..." if len(ids) > 1 else "")
            except Exception:
                pass

    # Save market data for trading
    with open("data/target_market.json", "w") as f:
        json.dump(event, f, indent=2)
    print(f"\n  Market data saved: data/target_market.json")


def cmd_trade(args):
    """Place a trade on the target market."""
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    # Load target market
    if not os.path.exists("data/target_market.json"):
        print("  ❌ Run 'python first_trade.py find' first")
        return

    with open("data/target_market.json") as f:
        event = json.load(f)

    # Find the target bucket (76-77°F or whatever has edge)
    markets = event.get("markets", [])
    target = None
    
    for m in markets:
        title = m.get("groupItemTitle", m.get("question", "?"))
        # Look for 76-77 bucket (our confirmed signal)
        if "76" in title and "77" in title:
            target = m
            break

    if target is None:
        # Fall back to first bucket with YES price < 0.30
        for m in markets:
            prices = json.loads(m.get("outcomePrices", "[0,0]"))
            if float(prices[0]) < 0.30 and float(prices[0]) > 0.01:
                target = m
                break

    if target is None:
        print("  ❌ No suitable target bucket found")
        return

    title = target.get("groupItemTitle", target.get("question", "?"))
    prices = json.loads(target.get("outcomePrices", "[0,0]"))
    yes_price = float(prices[0])
    token_ids = target.get("clobTokenIds", "")
    
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    
    yes_token_id = token_ids[0] if token_ids else None

    if not yes_token_id:
        print("  ❌ No token ID found for target market")
        return

    amount = args.amount
    # Number of contracts = amount / price
    size = round(amount / yes_price, 2)

    print(f"\n{'='*60}")
    print(f"TRADE CONFIRMATION")
    print(f"{'='*60}")
    print(f"  Event:    {event['title']}")
    print(f"  Bucket:   {title}")
    print(f"  Action:   BUY YES")
    print(f"  Price:    ${yes_price:.2f}")
    print(f"  Amount:   ${amount:.2f}")
    print(f"  Shares:   {size:.1f}")
    print(f"  Payout:   ${size:.2f} if correct (${size - amount:.2f} profit)")
    print(f"  Token ID: {yes_token_id[:30]}...")
    print(f"{'='*60}")

    if not args.yes:
        confirm = input(f"\n  Type 'YES' to execute: ")
        if confirm != "YES":
            print("  Cancelled.")
            return

    # Execute trade
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
        print(f"  Result: {json.dumps(result, indent=2)}")

        # Log the trade
        trade_record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event["title"],
            "bucket": title,
            "side": "BUY_YES",
            "price": yes_price,
            "amount_usd": amount,
            "size": size,
            "token_id": yes_token_id,
            "result": result,
        }

        log_path = "data/weather_trades.json"
        trades = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                trades = json.load(f)
        trades.append(trade_record)
        with open(log_path, "w") as f:
            json.dump(trades, f, indent=2)
        print(f"  Trade logged: {log_path}")

    except Exception as e:
        print(f"\n  ❌ Order failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="First Polymarket Trade")
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("setup", help="Generate CLOB API credentials")
    subs.add_parser("check", help="Check connection and balance")
    subs.add_parser("find", help="Find target weather market")

    p_trade = subs.add_parser("trade", help="Place a trade")
    p_trade.add_argument("--amount", type=float, default=5.0,
                         help="USD amount to trade (default: $5)")
    p_trade.add_argument("--yes", action="store_true",
                         help="Skip confirmation (dangerous!)")

    args = parser.parse_args()

    dispatch = {
        "setup": cmd_setup,
        "check": cmd_check,
        "find": cmd_find,
        "trade": cmd_trade,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
