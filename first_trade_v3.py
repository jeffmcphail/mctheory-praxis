"""First trade v3 — with correct proxy wallet signature type."""
import json
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
funder = os.getenv("POLYMARKET_API_ADDRESS", "")


def get_client(sig_type=1):
    """Get CLOB client with proxy signature type."""
    from py_clob_client.client import ClobClient
    client = ClobClient(
        HOST, key=pk, chain_id=CHAIN_ID,
        signature_type=sig_type, funder=funder,
    )
    client.set_api_creds(client.derive_api_key())
    return client


def cmd_trade(amount=5.0):
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    if not os.path.exists("data/target_market.json"):
        print("  Run: python first_trade_v2.py find")
        return

    with open("data/target_market.json") as f:
        data = json.load(f)

    event = data["event"]
    target = data.get("target")
    if not target:
        print("  No target found")
        return

    title = target.get("groupItemTitle", "?")
    prices = json.loads(target.get("outcomePrices", "[0,0]"))
    yes_price = float(prices[0])
    token_ids = target.get("clobTokenIds", "")
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    yes_token_id = token_ids[0]

    size = round(amount / yes_price, 2)

    print(f"\n  {event['title']}")
    print(f"  Bucket: {title}")
    print(f"  BUY YES @ ${yes_price:.2f}, ${amount:.2f} spend, {size:.1f} shares")

    # Try sig_type=1 (POLY_PROXY) first, then 0 (EOA)
    for sig_type, name in [(1, "POLY_PROXY"), (0, "EOA"), (2, "GNOSIS_SAFE")]:
        print(f"\n  Trying signature type {sig_type} ({name})...")
        try:
            client = get_client(sig_type)
            order_args = OrderArgs(
                price=yes_price,
                size=size,
                side=BUY,
                token_id=yes_token_id,
            )
            signed_order = client.create_order(order_args)
            result = client.post_order(signed_order)
            print(f"\n  ✅ ORDER PLACED with {name}!")
            print(f"  {json.dumps(result, indent=2)}")

            # Log
            os.makedirs("data", exist_ok=True)
            log_path = "data/weather_trades.json"
            trades = []
            if os.path.exists(log_path):
                with open(log_path) as f:
                    trades = json.load(f)
            trades.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event["title"],
                "bucket": title,
                "side": "BUY_YES",
                "price": yes_price,
                "amount": amount,
                "size": size,
                "sig_type": name,
                "result": result,
            })
            with open(log_path, "w") as f:
                json.dump(trades, f, indent=2)
            return

        except Exception as e:
            err = str(e)
            if "balance" in err.lower():
                print(f"    Balance error: {err[:100]}")
            else:
                print(f"    Failed: {err[:100]}")

    print(f"\n  ❌ All signature types failed.")
    print(f"  The USDC may need to be deposited into the Polymarket proxy wallet.")
    print(f"  Try depositing through polymarket.com with VPN (just need 10 seconds).")


if __name__ == "__main__":
    amt = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    confirm = input(f"\n  Place ${amt:.2f} trade on LA weather? (YES/no): ")
    if confirm == "YES":
        cmd_trade(amt)
    else:
        print("  Cancelled.")
