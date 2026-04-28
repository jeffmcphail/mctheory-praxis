"""
scripts/find_parent.py -- Find the correct parentCollectionId for redemption

Approach: Look up the market by CLOB token ID, get the event-level data,
and extract the negRiskMarketId which IS the parentCollectionId.
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

CTF_ABI = json.loads("""[
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"collectionId","type":"bytes32"}],"name":"getPositionId","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSet","type":"uint256"}],"name":"getCollectionId","outputs":[{"name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],"name":"payoutDenominator","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"},{"name":"index","type":"uint256"}],"name":"payoutNumerators","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"}
]""")


def find_market_by_token(token_id):
    """Find market data by searching CLOB token ID."""
    # Method 1: Gamma API by CLOB token
    print(f"    Searching Gamma by clobTokenIds...")
    try:
        r = requests.get(f"{GAMMA_API}/markets", params={
            "clobTokenIds": token_id,
        }, timeout=10)
        data = r.json()
        if data and len(data) > 0:
            return data[0]
    except Exception as e:
        print(f"    Gamma search failed: {e}")

    # Method 2: Search CLOB API
    print(f"    Searching CLOB API...")
    try:
        r = requests.get(f"{CLOB_API}/markets/{token_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    return None


def find_event_by_market(market_data):
    """Find the parent event for a market."""
    event_slug = market_data.get("eventSlug", "")
    if not event_slug:
        # Try condition_id based lookup
        cond = market_data.get("conditionId", "")
        if cond:
            try:
                r = requests.get(f"{GAMMA_API}/events", params={
                    "closed": "true", "limit": 200,
                }, timeout=15)
                events = r.json()
                for e in events:
                    for m in e.get("markets", []):
                        if m.get("conditionId") == cond:
                            return e
            except Exception:
                pass
        return None

    try:
        r = requests.get(f"{GAMMA_API}/events", params={
            "slug": event_slug,
        }, timeout=10)
        events = r.json()
        if events:
            return events[0]
    except Exception:
        pass

    return None


def main():
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("No POLYMARKET_PRIVATE_KEY"); sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    acct = w3.eth.account.from_key(pk)
    wallet = acct.address
    ctf = w3.eth.contract(address=w3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)

    print(f"\n{'='*80}")
    print(f"  PARENT COLLECTION ID FINDER")
    print(f"  Wallet: {wallet}")
    print(f"{'='*80}")

    # Fetch positions
    r = requests.get("https://data-api.polymarket.com/positions",
                     params={"user": wallet.lower()}, timeout=15)
    positions = r.json()

    for pos in positions:
        asset_id = pos.get("asset", "")
        cond_id = pos.get("conditionId", "")
        outcome = pos.get("outcome", "")
        size = float(pos.get("size", 0))
        title = pos.get("title", "")[:55]

        if size <= 0 or not cond_id or not asset_id:
            continue

        # Check balance
        try:
            balance = ctf.functions.balanceOf(wallet, int(asset_id)).call()
        except Exception:
            balance = 0
        if balance == 0:
            continue

        # Check if resolved and we won
        cid_bytes = bytes.fromhex(cond_id[2:] if cond_id.startswith("0x") else cond_id)
        try:
            denom = ctf.functions.payoutDenominator(cid_bytes).call()
        except Exception:
            denom = 0
        if denom == 0:
            continue

        our_idx = 0 if outcome == "Yes" else 1
        try:
            our_num = ctf.functions.payoutNumerators(cid_bytes, our_idx).call()
        except Exception:
            our_num = 0
        if our_num == 0:
            continue

        print(f"\n  {'─'*70}")
        print(f"  {title}")
        print(f"  Asset ID: {asset_id}")
        print(f"  Cond ID:  {cond_id}")
        print(f"  Balance:  {balance/1e6:.6f} | Payout: {our_num}/{denom}")

        # Step 1: Find the market by token ID
        print(f"\n  Step 1: Find market by token ID")
        market = find_market_by_token(asset_id)

        if market:
            print(f"    Found market: {market.get('question', '?')[:50]}")
            print(f"    negRisk: {market.get('negRisk', '?')}")
            print(f"    negRiskMarketId: {market.get('negRiskMarketId', 'N/A')}")
            print(f"    negRiskRequestId: {market.get('negRiskRequestId', 'N/A')}")
            print(f"    eventSlug: {market.get('eventSlug', 'N/A')}")
            print(f"    conditionId from Gamma: {market.get('conditionId', 'N/A')}")

            # All available fields
            print(f"    All fields: {sorted(market.keys())}")

            neg_risk_market_id = market.get("negRiskMarketId", "")

            if neg_risk_market_id:
                # This IS the parentCollectionId
                print(f"\n  Step 2: Testing negRiskMarketId as parentCollectionId")
                parent_hex = neg_risk_market_id
                if not parent_hex.startswith("0x"):
                    parent_hex = "0x" + parent_hex
                parent_bytes = bytes.fromhex(parent_hex[2:].zfill(64))

                idx = 1 if outcome == "Yes" else 2
                try:
                    coll = ctf.functions.getCollectionId(
                        parent_bytes, cid_bytes, idx).call()
                    pid = ctf.functions.getPositionId(
                        w3.to_checksum_address(USDC_E), coll).call()
                    print(f"    Computed positionId: {pid}")
                    print(f"    Actual asset_id:    {int(asset_id)}")
                    if pid == int(asset_id):
                        print(f"    >>> MATCH! parentCollectionId = {parent_hex}")
                        print(f"    >>> Use this to redeem!")
                    else:
                        print(f"    No match. Trying other indexSets...")
                        for try_idx in [1, 2, 3]:
                            try:
                                coll = ctf.functions.getCollectionId(
                                    parent_bytes, cid_bytes, try_idx).call()
                                pid = ctf.functions.getPositionId(
                                    w3.to_checksum_address(USDC_E), coll).call()
                                if pid == int(asset_id):
                                    print(f"    >>> MATCH with indexSet={try_idx}!")
                                    break
                            except Exception:
                                pass
                except Exception as e:
                    print(f"    Error: {e}")
            else:
                # No negRiskMarketId -- try to get from event level
                print(f"\n  Step 2: No negRiskMarketId, checking event level")
                event = find_event_by_market(market)
                if event:
                    print(f"    Event: {event.get('title', '?')[:50]}")
                    print(f"    Event negRisk: {event.get('negRisk', '?')}")
                    print(f"    Event negRiskMarketId: "
                          f"{event.get('negRiskMarketId', 'N/A')}")
                    # Check all market-level fields
                    for m in event.get("markets", [])[:2]:
                        print(f"    Sub-market: {m.get('question', '?')[:40]}")
                        print(f"      negRiskMarketId: {m.get('negRiskMarketId', 'N/A')}")
                        print(f"      negRiskRequestId: {m.get('negRiskRequestId', 'N/A')}")
                else:
                    print(f"    Event not found")

                # Last resort: try to look at our original trade tx
                print(f"\n  Step 3: Checking Polygonscan for token transfers")
                try:
                    api_key = os.getenv("POLYGONSCAN_API_KEY", "")
                    params = {
                        "module": "account",
                        "action": "token1155tx",
                        "address": wallet,
                        "sort": "desc",
                        "page": 1,
                        "offset": 50,
                    }
                    if api_key:
                        params["apikey"] = api_key
                    r = requests.get("https://api.polygonscan.com/api",
                                     params=params, timeout=15)
                    txs = r.json().get("result", [])
                    if isinstance(txs, list):
                        # Find transfers of our token
                        for tx in txs:
                            if tx.get("tokenID") == asset_id:
                                print(f"    Found transfer tx: {tx.get('hash', '?')}")
                                print(f"    From: {tx.get('from', '?')}")
                                print(f"    To: {tx.get('to', '?')}")
                                print(f"    Block: {tx.get('blockNumber', '?')}")
                                print(f"    Decode this tx on Polygonscan to find "
                                      f"parentCollectionId")
                                break
                        else:
                            print(f"    No matching token transfers found in recent history")
                    else:
                        print(f"    Polygonscan response: {str(txs)[:100]}")
                except Exception as e:
                    print(f"    Polygonscan error: {e}")
        else:
            print(f"    Market not found in Gamma API")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
