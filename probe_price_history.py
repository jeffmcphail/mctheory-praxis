"""Probe Polymarket price history API endpoints."""
import json, os, requests, time
from dotenv import load_dotenv; load_dotenv()
from py_clob_client.client import ClobClient

client = ClobClient("https://clob.polymarket.com", 
                     key=os.getenv("POLYMARKET_PRIVATE_KEY"), 
                     chain_id=137, signature_type=0)
client.set_api_creds(client.derive_api_key())

# Get a known market to test with
print("=== Finding test markets ===")
r = requests.get("https://gamma-api.polymarket.com/events",
                 params={"slug": "march-inflation-us-monthly"})
event = r.json()[0]
test_market = None
for m in event["markets"]:
    if "0.8" in m.get("question", ""):
        test_market = m
        break

if test_market:
    cid = test_market["conditionId"]
    token_ids = json.loads(test_market["clobTokenIds"])
    print(f"Market: {test_market['question'][:60]}")
    print(f"Condition ID: {cid}")
    print(f"Token IDs: {[t[:20]+'...' for t in token_ids]}")

    # Test 1: CLOB API price history
    print(f"\n=== Test 1: CLOB /prices-history ===")
    for fidelity in [1, 5, 60, 1440]:
        try:
            url = f"https://clob.polymarket.com/prices-history"
            params = {"market": cid, "interval": "all", "fidelity": fidelity}
            r = requests.get(url, params=params, timeout=10)
            if r.ok:
                data = r.json()
                if isinstance(data, dict) and "history" in data:
                    hist = data["history"]
                    print(f"  fidelity={fidelity}: {len(hist)} points")
                    if hist:
                        print(f"    First: {hist[0]}")
                        print(f"    Last:  {hist[-1]}")
                elif isinstance(data, list):
                    print(f"  fidelity={fidelity}: {len(data)} points (list)")
                    if data:
                        print(f"    First: {data[0]}")
                        print(f"    Last:  {data[-1]}")
                else:
                    print(f"  fidelity={fidelity}: {type(data)} — keys: {list(data.keys()) if isinstance(data, dict) else '?'}")
            else:
                print(f"  fidelity={fidelity}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  fidelity={fidelity}: Error: {e}")
        time.sleep(0.3)

    # Test 2: Try with token_id instead
    print(f"\n=== Test 2: CLOB /prices-history with token_id ===")
    try:
        r = requests.get("https://clob.polymarket.com/prices-history",
                         params={"market": token_ids[0], "interval": "all", "fidelity": 60},
                         timeout=10)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and "history" in data:
                print(f"  {len(data['history'])} points")
                if data["history"]:
                    print(f"    First: {data['history'][0]}")
                    print(f"    Last:  {data['history'][-1]}")
            else:
                print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else len(data)}")
        else:
            print(f"  HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 3: Try the client's built-in method
    print(f"\n=== Test 3: Client get_last_trade_price ===")
    try:
        price = client.get_last_trade_price(token_ids[0])
        print(f"  Last trade price: {price}")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n=== Test 4: Client get_last_trades_prices ===")
    try:
        prices = client.get_last_trades_prices([{"token_id": t} for t in token_ids])
        print(f"  Result: {prices}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 5: Try timeseries endpoint
    print(f"\n=== Test 5: CLOB /timeseries ===")
    for endpoint in ["/timeseries", "/time-series", "/candles"]:
        try:
            r = requests.get(f"https://clob.polymarket.com{endpoint}",
                             params={"market": cid, "interval": "1h"},
                             timeout=5)
            print(f"  {endpoint}: HTTP {r.status_code}")
            if r.ok:
                data = r.json()
                print(f"    {type(data)}: {str(data)[:200]}")
        except Exception as e:
            print(f"  {endpoint}: {e}")

    # Test 6: Gamma API timeseries 
    print(f"\n=== Test 6: Gamma API timeseries ===")
    try:
        r = requests.get("https://gamma-api.polymarket.com/markets",
                         params={"clob_token_ids": token_ids[0]})
        if r.ok:
            markets = r.json()
            if markets:
                m = markets[0]
                # Check for any timeseries/history fields
                for key in m.keys():
                    if any(w in key.lower() for w in ["price", "history", "time", "series"]):
                        val = m[key]
                        print(f"  {key}: {str(val)[:100]}")
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone.")
