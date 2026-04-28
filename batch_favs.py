"""Execute curated favorite trades on specific markets."""
import json
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()
import requests

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

# Target trades — slug, search term for bucket, side, budget
TARGETS = [
    {
        "name": "March CPI >= 0.8%",
        "search": "monthly inflation increase by 0.8",
        "side": "YES",
        "budget": 150.0,
    },
    {
        "name": "76ers > 43.5 wins",
        "search": "Philadelphia 76ers win more than 43.5",
        "side": "YES",
        "budget": 50.0,
    },
    {
        "name": "OKC Thunder #1 seed",
        "search": "Oklahoma City Thunder finish as the #1 seed",
        "side": "YES",
        "budget": 50.0,
    },
]


def get_client():
    from py_clob_client.client import ClobClient
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=0)
    client.set_api_creds(client.derive_api_key())
    return client


def find_market(search_term):
    """Search Polymarket for a specific market."""
    # Try multiple tag categories
    for tag in ["economy", "politics", "sports", "finance", "crypto"]:
        for offset in range(0, 200, 100):
            r = requests.get("https://gamma-api.polymarket.com/events", params={
                "tag_slug": tag, "limit": "100", "offset": str(offset),
                "active": "true", "closed": "false",
            })
            events = r.json()
            for event in events:
                for m in event.get("markets", []):
                    q = m.get("question", m.get("groupItemTitle", ""))
                    if search_term.lower() in q.lower():
                        prices = json.loads(m.get("outcomePrices", "[0,0]"))
                        token_ids = m.get("clobTokenIds", "")
                        if isinstance(token_ids, str):
                            token_ids = json.loads(token_ids)
                        return {
                            "question": q,
                            "yes_price": float(prices[0]),
                            "no_price": float(prices[1]) if len(prices) > 1 else 1 - float(prices[0]),
                            "yes_token": token_ids[0] if token_ids else None,
                            "no_token": token_ids[1] if len(token_ids) > 1 else None,
                            "slug": event.get("slug", ""),
                            "volume": float(m.get("volume", 0)),
                        }
            if len(events) < 100:
                break
            time.sleep(0.3)
    return None


def main():
    execute = "--execute" in sys.argv
    
    print(f"\n{'='*80}")
    print(f"CURATED FAVORITES EXECUTION")
    print(f"  Mode: {'🔴 LIVE' if execute else '📋 DRY RUN'}")
    print(f"{'='*80}")
    
    # Find all target markets
    trades = []
    total = 0
    
    for t in TARGETS:
        print(f"\n  Searching: {t['name']}...")
        market = find_market(t["search"])
        
        if not market:
            print(f"    ❌ Not found: {t['search']}")
            continue
        
        price = market["yes_price"] if t["side"] == "YES" else market["no_price"]
        token_id = market["yes_token"] if t["side"] == "YES" else market["no_token"]
        size = round(t["budget"] / price, 2)
        profit = size * (1.0 - price) * 0.98  # After 2% fee on winnings
        
        print(f"    ✅ Found: {market['question'][:70]}")
        print(f"    {t['side']} @ ${price:.3f}  |  ${t['budget']:.0f} spend  |  "
              f"{size:.0f} shares  |  ${profit:.2f} profit if correct")
        
        trades.append({
            **t,
            "market": market,
            "price": price,
            "token_id": token_id,
            "size": size,
            "profit": profit,
        })
        total += t["budget"]
    
    total_profit = sum(t["profit"] for t in trades)
    
    print(f"\n  {'─'*75}")
    print(f"  TOTAL: {len(trades)} trades, ${total:.2f} spend, "
          f"${total_profit:.2f} expected profit ({total_profit/total*100:.1f}%)")
    
    if not execute:
        print(f"\n  Add --execute to place trades.")
        return
    
    confirm = input(f"\n  Type 'YES' to place {len(trades)} trades for ${total:.2f}: ")
    if confirm != "YES":
        print("  Cancelled.")
        return
    
    # Execute
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY
    
    client = get_client()
    results = []
    
    for i, t in enumerate(trades, 1):
        print(f"\n  [{i}/{len(trades)}] {t['name']}...")
        
        if not t["token_id"]:
            print(f"    ❌ No token ID")
            continue
        
        try:
            order_args = OrderArgs(
                price=round(t["price"], 2),
                size=t["size"],
                side=BUY,
                token_id=t["token_id"],
            )
            signed = client.create_order(order_args)
            result = client.post_order(signed)
            
            status = result.get("status", "?")
            oid = result.get("orderID", "?")[:16]
            print(f"    ✅ {status} (ID: {oid}...)")
            
            results.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "name": t["name"],
                "side": t["side"],
                "price": t["price"],
                "spend": t["budget"],
                "size": t["size"],
                "order_id": result.get("orderID", ""),
                "status": status,
            })
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    ❌ {str(e)[:120]}")
            results.append({"name": t["name"], "error": str(e)[:120], "status": "FAILED"})
            time.sleep(1)
    
    # Save
    os.makedirs("data", exist_ok=True)
    log_path = "data/favorites_trades.json"
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    existing.extend(results)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    
    succeeded = sum(1 for r in results if r.get("status") != "FAILED")
    print(f"\n{'='*80}")
    print(f"  RESULTS: {succeeded}/{len(trades)} succeeded")
    print(f"  Logged: {log_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
