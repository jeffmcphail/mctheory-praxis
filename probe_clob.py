"""Probe CLOB API for all available position/trade/order methods."""
import os, json
from dotenv import load_dotenv
load_dotenv()
from py_clob_client.client import ClobClient

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
client = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137, signature_type=0)
client.set_api_creds(client.derive_api_key())

# List all methods that might give us position/trade data
methods = [m for m in dir(client) if not m.startswith("_") and callable(getattr(client, m, None))]
relevant = [m for m in methods if any(w in m.lower() for w in 
    ["order", "trade", "position", "balance", "fill", "history", "market"])]
print("Relevant client methods:")
for m in sorted(relevant):
    print(f"  {m}")

# Try each to see what returns data
print("\n--- get_orders() ---")
try:
    orders = client.get_orders()
    print(f"  Type: {type(orders)}, Count: {len(orders) if isinstance(orders, list) else '?'}")
    if isinstance(orders, list) and orders:
        print(f"  First order keys: {list(orders[0].keys())}")
        print(f"  Statuses: {set(o.get('status') for o in orders)}")
except Exception as e:
    print(f"  Error: {e}")

print("\n--- get_trades() ---")
try:
    trades = client.get_trades()
    print(f"  Type: {type(trades)}, Count: {len(trades) if isinstance(trades, list) else '?'}")
    if isinstance(trades, list) and trades:
        print(f"  First trade keys: {list(trades[0].keys())}")
        print(f"  First: {json.dumps(trades[0], indent=2)[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# Try get_orders with different params
print("\n--- get_orders (trying params) ---")
for param_set in [
    {"state": "ALL"},
    {"state": "MATCHED"},
    {"market": ""},
]:
    try:
        result = client.get_orders(**param_set)
        count = len(result) if isinstance(result, list) else "?"
        print(f"  get_orders({param_set}): {count} results")
    except Exception as e:
        print(f"  get_orders({param_set}): {str(e)[:80]}")

# Try getting specific order by ID (one of our matched ones)
print("\n--- get_order (specific) ---")
matched_ids = [
    "0x1e81af0ad78fb9",  # OKC bumped
    "0xb38f323d5cf9f0",  # CPI bumped
]
for oid in matched_ids:
    try:
        result = client.get_order(oid)
        print(f"  get_order({oid[:16]}): {result}")
    except Exception as e:
        print(f"  get_order({oid[:16]}): {str(e)[:80]}")

# Check if there's a REST endpoint directly
print("\n--- Direct API calls ---")
import requests
addr = os.getenv("POLYMARKET_API_ADDRESS", "")
headers = {"Authorization": f"Bearer {client.creds.api_key}"}

for endpoint in [
    f"https://clob.polymarket.com/orders?maker_address={addr}",
    f"https://clob.polymarket.com/trades?maker_address={addr}",
    f"https://clob.polymarket.com/positions?maker_address={addr}",
    f"https://clob.polymarket.com/balances/{addr}",
]:
    try:
        r = requests.get(endpoint, headers=headers, timeout=5)
        data = r.json() if r.ok else r.text[:100]
        count = len(data) if isinstance(data, list) else "?"
        print(f"  {endpoint.split('.com/')[1][:40]}: {r.status_code} ({count} items)")
        if isinstance(data, list) and data:
            print(f"    Keys: {list(data[0].keys()) if isinstance(data[0], dict) else '?'}")
    except Exception as e:
        print(f"  {endpoint.split('.com/')[1][:40]}: {str(e)[:60]}")

# Try data API (different base URL)
print("\n--- Data API ---")
for endpoint in [
    f"https://data-api.polymarket.com/positions?user={addr}",
    f"https://data-api.polymarket.com/trades?user={addr}",
]:
    try:
        r = requests.get(endpoint, timeout=5)
        data = r.json() if r.ok else r.text[:100]
        count = len(data) if isinstance(data, list) else "?"
        print(f"  {endpoint.split('.com/')[1][:40]}: {r.status_code} ({count} items)")
        if isinstance(data, list) and data:
            print(f"    Keys: {list(data[0].keys()) if isinstance(data[0], dict) else '?'}")
            print(f"    First: {json.dumps(data[0])[:300]}")
    except Exception as e:
        print(f"  {endpoint.split('.com/')[1][:40]}: {str(e)[:60]}")
