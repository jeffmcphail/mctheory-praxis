"""Probe Gamma API to find the right way to look up markets by condition ID or token ID."""
import json, requests

# From the CLOB orders, we know:
# - asset_id (token ID): long number like "745354076704..."
# - market (condition ID): hex like "0x..."

# Get a real order to test with
import os
from dotenv import load_dotenv
load_dotenv()
from py_clob_client.client import ClobClient

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
client = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137, signature_type=0)
client.set_api_creds(client.derive_api_key())

orders = client.get_orders()
print(f"Got {len(orders)} orders\n")

# Pick an order to test with
for o in orders[:3]:
    asset_id = o.get("asset_id", "")
    market_id = o.get("market", "")
    print(f"Order: {o.get('id', '')[:20]}")
    print(f"  asset_id (token): {asset_id[:30]}...")
    print(f"  market (cond_id): {market_id[:30]}...")
    print(f"  price: {o.get('price')}, side: {o.get('side')}")
    print()

# Test different Gamma API endpoints with first order
if orders:
    asset_id = orders[0].get("asset_id", "")
    market_id = orders[0].get("market", "")
    
    endpoints = [
        ("GET /markets?conditionID=X", f"https://gamma-api.polymarket.com/markets?conditionID={market_id}"),
        ("GET /markets?condition_id=X", f"https://gamma-api.polymarket.com/markets?condition_id={market_id}"),
        ("GET /markets?conditionId=X", f"https://gamma-api.polymarket.com/markets?conditionId={market_id}"),
        ("GET /markets/{conditionID}", f"https://gamma-api.polymarket.com/markets/{market_id}"),
        ("GET /markets?clob_token_ids=X", f"https://gamma-api.polymarket.com/markets?clob_token_ids={asset_id}"),
        ("GET /markets?token_id=X", f"https://gamma-api.polymarket.com/markets?token_id={asset_id}"),
        ("GET /markets?id=X", f"https://gamma-api.polymarket.com/markets?id={market_id}"),
        ("GET /events?id=X", f"https://gamma-api.polymarket.com/events?id={market_id}"),
    ]
    
    print("=" * 80)
    print("TESTING GAMMA API ENDPOINTS")
    print("=" * 80)
    
    for label, url in endpoints:
        try:
            r = requests.get(url, timeout=5)
            data = r.json() if r.ok else None
            
            if data:
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    q = first.get("question", first.get("title", first.get("groupItemTitle", "NO TITLE")))
                    print(f"\n  ✅ {label}")
                    print(f"     Status: {r.status_code}, Items: {len(data)}")
                    print(f"     Title: {q[:70]}")
                    print(f"     Keys: {list(first.keys())[:10]}")
                elif isinstance(data, dict):
                    q = data.get("question", data.get("title", data.get("groupItemTitle", "NO TITLE")))
                    print(f"\n  ✅ {label}")
                    print(f"     Status: {r.status_code}, Type: dict")
                    print(f"     Title: {q[:70]}")
                    print(f"     Keys: {list(data.keys())[:10]}")
                else:
                    print(f"\n  ⚠ {label} — {r.status_code}, empty: {data}")
            else:
                print(f"\n  ❌ {label} — {r.status_code}: {r.text[:80]}")
        except Exception as e:
            print(f"\n  ❌ {label} — Error: {str(e)[:60]}")

print("\n\nDone.")
