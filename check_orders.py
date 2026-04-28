"""Check current orders and positions on Polymarket."""
import os, json
from dotenv import load_dotenv
load_dotenv()
from py_clob_client.client import ClobClient

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
c = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137, signature_type=0)
c.set_api_creds(c.derive_api_key())

print("=== OPEN ORDERS ===")
try:
    orders = c.get_orders()
    if isinstance(orders, list):
        print(f"  {len(orders)} orders")
        for o in orders:
            print(f"  Status: {o.get('status')}  Side: {o.get('side')}  "
                  f"Price: {o.get('price')}  Size: {o.get('original_size')}  "
                  f"Matched: {o.get('size_matched')}")
    else:
        print(f"  {orders}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== TRADES/FILLS ===")
try:
    trades = c.get_trades()
    if isinstance(trades, list):
        print(f"  {len(trades)} trades")
        for t in trades[:5]:
            print(f"  {json.dumps(t, indent=2)[:200]}")
    else:
        print(f"  {trades}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== POSITIONS ===")
try:
    # Try different method names
    for method in ["get_positions", "get_all_positions"]:
        if hasattr(c, method):
            pos = getattr(c, method)()
            print(f"  {method}: {pos}")
            break
    else:
        print("  No position method found")
except Exception as e:
    print(f"  Error: {e}")
