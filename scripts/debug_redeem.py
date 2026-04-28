"""
scripts/debug_redeem.py -- Diagnose why redemption fails

The CTF contract computes positionId from:
  collectionId = getCollectionId(parentCollectionId, conditionId, indexSet)
  positionId = getPositionId(collateralToken, collectionId)

If positionId doesn't match asset_id, the redemption finds zero balance.
This script computes expected vs actual and finds what's wrong.
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEGRISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"

# Extended CTF ABI with position/collection helpers
CTF_ABI = json.loads("""[
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"collectionId","type":"bytes32"}],"name":"getPositionId","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSet","type":"uint256"}],"name":"getCollectionId","outputs":[{"name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],"name":"getOutcomeSlotCount","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],"name":"payoutDenominator","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"},{"name":"index","type":"uint256"}],"name":"payoutNumerators","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
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
    print(f"  REDEMPTION DEBUGGER")
    print(f"  Wallet: {wallet}")
    print(f"{'='*80}")

    # Fetch positions
    r = requests.get("https://data-api.polymarket.com/positions",
                     params={"user": wallet.lower()}, timeout=15)
    positions = r.json()

    # Filter to redeemable only
    for pos in positions:
        asset_id = pos.get("asset", "")
        cond_id = pos.get("conditionId", "")
        outcome = pos.get("outcome", "")
        size = float(pos.get("size", 0))
        title = pos.get("title", "")[:50]

        if size <= 0 or not cond_id or not asset_id:
            continue

        # Check on-chain balance
        try:
            balance = ctf.functions.balanceOf(wallet, int(asset_id)).call()
        except Exception:
            balance = 0

        if balance == 0:
            continue

        # Check if resolved
        cid_bytes = bytes.fromhex(cond_id[2:] if cond_id.startswith("0x") else cond_id)
        try:
            denom = ctf.functions.payoutDenominator(cid_bytes).call()
        except Exception:
            denom = 0

        if denom == 0:
            continue  # Not resolved yet

        # Check our payout
        our_idx = 0 if outcome == "Yes" else 1
        try:
            our_num = ctf.functions.payoutNumerators(cid_bytes, our_idx).call()
        except Exception:
            our_num = 0

        if our_num == 0:
            continue  # We lost

        print(f"\n  {'─'*70}")
        print(f"  {title}")
        print(f"  Outcome: {outcome} | Balance: {balance/1e6:.6f} | "
              f"Payout: {our_num}/{denom}")
        print(f"  Asset ID:     {asset_id}")
        print(f"  Condition ID: {cond_id}")

        # ── COMPUTE EXPECTED POSITION IDS ──
        print(f"\n  Computing expected positionIds with parentCollectionId = 0x00...")

        zero_parent = bytes(32)
        index_set_yes = 1  # Binary: YES = indexSet 1
        index_set_no = 2   # Binary: NO = indexSet 2

        # Get collection ID for YES with zero parent
        try:
            coll_yes = ctf.functions.getCollectionId(
                zero_parent, cid_bytes, index_set_yes).call()
            pos_id_yes = ctf.functions.getPositionId(
                w3.to_checksum_address(USDC_E), coll_yes).call()
            print(f"  positionId(parent=0, YES): {pos_id_yes}")
        except Exception as e:
            print(f"  positionId(parent=0, YES): ERROR {e}")
            pos_id_yes = None

        try:
            coll_no = ctf.functions.getCollectionId(
                zero_parent, cid_bytes, index_set_no).call()
            pos_id_no = ctf.functions.getPositionId(
                w3.to_checksum_address(USDC_E), coll_no).call()
            print(f"  positionId(parent=0, NO):  {pos_id_no}")
        except Exception as e:
            print(f"  positionId(parent=0, NO):  ERROR {e}")
            pos_id_no = None

        actual_id = int(asset_id)
        print(f"  Actual asset_id:          {actual_id}")

        if pos_id_yes == actual_id:
            print(f"  --> MATCH on YES with parent=0x00")
            print(f"      Redemption SHOULD work with indexSets=[1]")
            correct_index_set = [1]
        elif pos_id_no == actual_id:
            print(f"  --> MATCH on NO with parent=0x00")
            print(f"      Redemption SHOULD work with indexSets=[2]")
            correct_index_set = [2]
        else:
            print(f"  --> NO MATCH with parent=0x00!")
            print(f"      This means parentCollectionId is NOT zero.")
            print(f"      Searching for correct parent...")

            # Try to find parent from transfer events
            # The token was minted via splitPosition or through the exchange
            # Let's check if the NegRisk adapter has info
            found_parent = False

            # Check outcome slot count
            try:
                slot_count = ctf.functions.getOutcomeSlotCount(cid_bytes).call()
                print(f"      Outcome slot count: {slot_count}")
            except Exception:
                slot_count = 0

            # Try common parentCollectionId patterns
            # Method: check balance at computed positionIds with different parents
            # The NegRisk system uses a deterministic parent derived from the event

            # Let's try to get the parent from the Gamma API event level
            slug = pos.get("slug", "")
            if slug:
                try:
                    gr = requests.get("https://gamma-api.polymarket.com/markets",
                                      params={"slug": slug}, timeout=5)
                    gdata = gr.json()
                    if gdata and len(gdata) > 0:
                        event_slug = gdata[0].get("eventSlug", "")
                        neg_risk = gdata[0].get("negRisk", False)
                        neg_risk_market_id = gdata[0].get("negRiskMarketId", "")
                        neg_risk_request_id = gdata[0].get("negRiskRequestId", "")
                        print(f"      Gamma: eventSlug={event_slug}")
                        print(f"      Gamma: negRisk={neg_risk}")
                        print(f"      Gamma: negRiskMarketId={neg_risk_market_id}")
                        print(f"      Gamma: negRiskRequestId={neg_risk_request_id}")

                        # If there's a negRiskMarketId, try it as parent
                        if neg_risk_market_id:
                            try:
                                parent_bytes = bytes.fromhex(
                                    neg_risk_market_id[2:]
                                    if neg_risk_market_id.startswith("0x")
                                    else neg_risk_market_id)
                                if len(parent_bytes) == 32:
                                    coll = ctf.functions.getCollectionId(
                                        parent_bytes, cid_bytes, index_set_yes).call()
                                    pid = ctf.functions.getPositionId(
                                        w3.to_checksum_address(USDC_E), coll).call()
                                    if pid == actual_id:
                                        print(f"      FOUND! parent = negRiskMarketId")
                                        found_parent = True
                            except Exception as e:
                                print(f"      negRiskMarketId as parent: {e}")
                except Exception as e:
                    print(f"      Gamma lookup error: {e}")

            if not found_parent:
                # Brute force: check the mint event for this token
                print(f"      Checking mint events for token {asset_id[:20]}...")
                try:
                    # TransferSingle topic for ERC-1155
                    transfer_topic = w3.keccak(
                        text="TransferSingle(address,address,address,uint256,uint256)")
                    # Filter for mints TO our wallet of this token
                    # This is expensive but we only need the first one
                    logs = w3.eth.get_logs({
                        "address": w3.to_checksum_address(CTF_ADDRESS),
                        "topics": [transfer_topic.hex()],
                        "fromBlock": "latest",
                    })
                    print(f"      Found {len(logs)} recent transfer events")
                except Exception as e:
                    print(f"      Event scan failed: {e}")
                    print(f"      Need archival RPC for full history")

            if not found_parent:
                print(f"\n      MANUAL FIX: Check the Polymarket Exchange contract")
                print(f"      Your tokens at asset_id {asset_id[:20]}... were created with")
                print(f"      a specific parentCollectionId during the trade.")
                print(f"      Look up the original trade tx on Polygonscan to find it.")

        # ── TRY SINGLE INDEX SET ──
        # Maybe [1,2] is wrong and we need just [1] or [2]
        if pos_id_yes == actual_id or pos_id_no == actual_id:
            print(f"\n  Checking balance at each indexSet position:")
            for idx in [1, 2]:
                try:
                    coll = ctf.functions.getCollectionId(
                        zero_parent, cid_bytes, idx).call()
                    pid = ctf.functions.getPositionId(
                        w3.to_checksum_address(USDC_E), coll).call()
                    bal = ctf.functions.balanceOf(wallet, pid).call()
                    print(f"    indexSet={idx}: positionId={pid}, balance={bal/1e6:.6f}")
                except Exception as e:
                    print(f"    indexSet={idx}: ERROR {e}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
