"""
scripts/find_parent.py -- Find parentCollectionId from trade history

Strategy: fetch your trade history from the Data API, find the tx hash
where you acquired the tokens, decode the on-chain calldata to extract
the parentCollectionId used during position creation.
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEGRISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"

CTF_ABI = json.loads("""[
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"collectionId","type":"bytes32"}],"name":"getPositionId","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSet","type":"uint256"}],"name":"getCollectionId","outputs":[{"name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],"name":"payoutDenominator","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"},{"name":"index","type":"uint256"}],"name":"payoutNumerators","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"}
]""")


def main():
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("No POLYMARKET_PRIVATE_KEY"); sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    acct = w3.eth.account.from_key(pk)
    wallet = acct.address
    ctf = w3.eth.contract(address=w3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)

    print(f"\n{'='*80}")
    print(f"  PARENT COLLECTION ID FINDER v2")
    print(f"  Wallet: {wallet}")
    print(f"{'='*80}")

    # Get positions
    r = requests.get("https://data-api.polymarket.com/positions",
                     params={"user": wallet.lower()}, timeout=15)
    positions = r.json()

    # Get ALL trade history
    print(f"\n  Fetching trade history...")
    all_trades = []
    try:
        r = requests.get("https://data-api.polymarket.com/trades", params={
            "user": wallet.lower(),
            "limit": 100,
        }, timeout=15)
        all_trades = r.json()
        print(f"  Found {len(all_trades)} trades")
        if all_trades and isinstance(all_trades, list) and len(all_trades) > 0:
            print(f"  Trade fields: {sorted(all_trades[0].keys())}")
            # Show first trade for debugging
            t = all_trades[0]
            print(f"  Sample trade:")
            for k in sorted(t.keys()):
                v = str(t[k])[:60]
                print(f"    {k}: {v}")
    except Exception as e:
        print(f"  Failed to fetch trades: {e}")

    # For each redeemable position, find the trade tx
    for pos in positions:
        asset_id = pos.get("asset", "")
        cond_id = pos.get("conditionId", "")
        outcome = pos.get("outcome", "")
        size = float(pos.get("size", 0))
        title = pos.get("title", "")[:55]

        if size <= 0 or not cond_id or not asset_id:
            continue

        # Only check redeemable positions
        cid_bytes = bytes.fromhex(cond_id[2:] if cond_id.startswith("0x") else cond_id)
        try:
            balance = ctf.functions.balanceOf(wallet, int(asset_id)).call()
            denom = ctf.functions.payoutDenominator(cid_bytes).call()
            our_idx = 0 if outcome == "Yes" else 1
            our_num = ctf.functions.payoutNumerators(cid_bytes, our_idx).call()
        except Exception:
            continue

        if balance == 0 or denom == 0 or our_num == 0:
            continue

        print(f"\n  {'='*70}")
        print(f"  {title}")
        print(f"  Asset: {asset_id[:30]}...")
        print(f"  CondID: {cond_id}")
        print(f"  Balance: {balance/1e6:.6f} | Payout: {our_num}/{denom}")

        # Find matching trade
        matching_trades = [t for t in all_trades
                          if t.get("asset", "") == asset_id or
                          t.get("tokenId", "") == asset_id]

        if matching_trades:
            print(f"\n  Found {len(matching_trades)} matching trade(s)")
            for t in matching_trades[:3]:
                tx_hash = t.get("transactionHash", t.get("txHash", ""))
                print(f"    Tx: {tx_hash}")
                print(f"    Side: {t.get('side', '?')} | Price: {t.get('price', '?')} | "
                      f"Size: {t.get('size', '?')}")

                # Decode the transaction
                if tx_hash:
                    try:
                        tx = w3.eth.get_transaction(tx_hash)
                        print(f"    To: {tx.to}")
                        print(f"    Input selector: {tx.input.hex()[:10]}")
                        print(f"    Input length: {len(tx.input.hex())} chars")

                        # Check internal transactions / trace
                        # The Exchange contract calls CTF.splitPosition internally
                        # We need to see what parentCollectionId it used

                        # For now, check the tx receipt for events
                        receipt = w3.eth.get_transaction_receipt(tx_hash)
                        print(f"    Logs: {len(receipt.logs)}")

                        for li, log in enumerate(receipt.logs):
                            # Look for CTF events
                            if log.address.lower() == CTF_ADDRESS.lower():
                                topic0 = log.topics[0].hex() if log.topics else ""
                                # PositionSplit = keccak256("PositionSplit(...)") 
                                # PositionsMerge = ...
                                # TransferSingle = c3d58168...
                                print(f"    CTF Log {li}: topic0={topic0[:20]}...")
                                # The data field might contain the parentCollectionId
                                if log.data:
                                    data_hex = log.data.hex()
                                    print(f"    Data ({len(data_hex)} chars): "
                                          f"{data_hex[:128]}...")

                    except Exception as e:
                        print(f"    Tx decode error: {e}")
        else:
            print(f"\n  No matching trades found in Data API")
            print(f"  Trying by conditionId...")
            cond_trades = [t for t in all_trades
                          if t.get("conditionId", "") == cond_id]
            if cond_trades:
                print(f"  Found {len(cond_trades)} trades by conditionId")
                for t in cond_trades[:2]:
                    print(f"    {json.dumps(t, indent=2)[:300]}")
            else:
                print(f"  No trades found by conditionId either")
                print(f"  Listing all unique assets in trades:")
                assets = set(t.get("asset", "")[:20] for t in all_trades)
                for a in sorted(assets):
                    print(f"    {a}...")

        # ALSO: Try bruteforce with the questionID approach
        # In Polymarket, questionID is often used as part of the parent
        print(f"\n  Trying questionID-based parent derivation...")
        slug = pos.get("slug", "")
        if slug:
            try:
                r = requests.get(f"https://gamma-api.polymarket.com/markets",
                                 params={"slug": slug}, timeout=5)
                gdata = r.json()
                if gdata and len(gdata) > 0:
                    mkt = gdata[0]
                    qid = mkt.get("questionID", "")
                    neg_risk_req = mkt.get("negRiskRequestID", "")
                    print(f"    questionID: {qid}")
                    print(f"    negRiskRequestID: {neg_risk_req}")

                    # Try questionID as parent
                    if qid:
                        qid_hex = qid if qid.startswith("0x") else "0x" + qid
                        try:
                            parent_bytes = bytes.fromhex(qid_hex[2:].zfill(64))
                            for idx in [1, 2]:
                                coll = ctf.functions.getCollectionId(
                                    parent_bytes, cid_bytes, idx).call()
                                pid = ctf.functions.getPositionId(
                                    w3.to_checksum_address(USDC_E), coll).call()
                                if pid == int(asset_id):
                                    print(f"    >>> MATCH! parent=questionID, indexSet={idx}")
                                    print(f"    >>> parentCollectionId = {qid_hex}")
                                    break
                            else:
                                print(f"    questionID as parent: no match")
                        except Exception as e:
                            print(f"    questionID test error: {e}")

                    # Try negRiskRequestID
                    if neg_risk_req:
                        req_hex = neg_risk_req if neg_risk_req.startswith("0x") else "0x" + neg_risk_req
                        try:
                            parent_bytes = bytes.fromhex(req_hex[2:].zfill(64))
                            for idx in [1, 2]:
                                coll = ctf.functions.getCollectionId(
                                    parent_bytes, cid_bytes, idx).call()
                                pid = ctf.functions.getPositionId(
                                    w3.to_checksum_address(USDC_E), coll).call()
                                if pid == int(asset_id):
                                    print(f"    >>> MATCH! parent=negRiskRequestID, indexSet={idx}")
                                    print(f"    >>> parentCollectionId = {req_hex}")
                                    break
                            else:
                                print(f"    negRiskRequestID as parent: no match")
                        except Exception as e:
                            print(f"    negRiskRequestID test error: {e}")
            except Exception as e:
                print(f"    Gamma lookup error: {e}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
