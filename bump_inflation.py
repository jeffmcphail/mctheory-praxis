"""Bump two pending inflation orders and cancel OKC."""
import os, json, sys, time
from dotenv import load_dotenv; load_dotenv()
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY

client = ClobClient("https://clob.polymarket.com", key=os.getenv("POLYMARKET_PRIVATE_KEY"), chain_id=137, signature_type=0)
client.set_api_creds(client.derive_api_key())

orders = client.get_orders()
print(f"Found {len(orders)} open orders\n")

# Identify orders by price to match our targets
for o in orders:
    price = float(o.get("price", 0))
    size = float(o.get("original_size", 0))
    matched = float(o.get("size_matched", 0))
    remaining = size - matched
    oid = o.get("id", "")
    asset = o.get("asset_id", "")[:20]
    
    action = None
    new_price = None
    
    # Inflation >3.5% — price 0.90, ~167 shares
    if abs(price - 0.90) < 0.005 and abs(size - 166.7) < 5 and matched == 0:
        action = "BUMP"
        new_price = 0.91
        label = "Inflation >3.5% 2026"
    # Annual CPI >=3.4% — price 0.49, ~202 shares  
    elif abs(price - 0.49) < 0.005 and abs(size - 202.0) < 5 and matched == 0:
        action = "BUMP"
        new_price = 0.51
        label = "Annual CPI >=3.4% March"
    # OKC #1 seed — price ~0.976, small fill
    elif abs(price - 0.976) < 0.01 and remaining > 40:
        action = "CANCEL"
        label = "OKC #1 seed (freeing capital)"
    else:
        continue
    
    print(f"  {label}")
    print(f"    Order: {oid[:20]}... Price: {price} Size: {size:.1f} Matched: {matched:.1f}")
    
    if action == "CANCEL":
        if "--execute" not in sys.argv:
            print(f"    → Would CANCEL (frees ${remaining * price:.2f})")
            continue
        result = client.cancel(order_id=oid)
        print(f"    → CANCELLED: {result}")
        
    elif action == "BUMP":
        new_size = round(remaining * price / new_price, 2)
        print(f"    → Bump {price:.3f} → {new_price:.3f} ({new_size:.1f} shares)")
        
        if "--execute" not in sys.argv:
            continue
            
        # Cancel
        result = client.cancel(order_id=oid)
        print(f"    Cancelled: {result}")
        time.sleep(0.5)
        
        # Resubmit
        token_id = o.get("asset_id", "")
        order_args = OrderArgs(price=new_price, size=new_size, side=BUY, token_id=token_id)
        signed = client.create_order(order_args)
        result = client.post_order(signed)
        print(f"    Resubmitted: {result.get('status')} (ID: {result.get('orderID', '?')[:16]}...)")
        time.sleep(0.5)

if "--execute" not in sys.argv:
    print(f"\n  Add --execute to go live.")
