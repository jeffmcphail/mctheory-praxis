"""
scripts/redeem_negrisk.py -- Redeem NegRisk-wrapped positions through the Adapter

The issue: Polymarket's NegRisk Exchange wraps CTF tokens with different IDs.
Calling CTF.redeemPositions directly finds zero balance.
Solution: call NegRiskAdapter.redeemPositions instead.

Usage:
    python -m scripts.redeem_negrisk                    # Dry run
    python -m scripts.redeem_negrisk --execute          # Execute
"""
import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEGRISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
NEGRISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"

# Same function signature as CTF -- selector 0x01864fcf
REDEEM_ABI = json.loads("""[
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"}
]""")

CTF_ABI = json.loads("""[
    {"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"},{"name":"index","type":"uint256"}],"name":"payoutNumerators","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],"name":"payoutDenominator","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")

USDC_ABI = json.loads("""[
    {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("No POLYMARKET_PRIVATE_KEY"); sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    acct = w3.eth.account.from_key(pk)
    wallet = acct.address

    ctf = w3.eth.contract(address=w3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)
    adapter = w3.eth.contract(address=w3.to_checksum_address(NEGRISK_ADAPTER), abi=REDEEM_ABI)
    usdc_c = w3.eth.contract(address=w3.to_checksum_address(USDC_E), abi=USDC_ABI)

    usdc_before = usdc_c.functions.balanceOf(wallet).call()

    print(f"\n{'='*80}")
    print(f"  NEGRISK ADAPTER REDEMPTION")
    print(f"  Wallet:  {wallet}")
    print(f"  USDC.e:  ${usdc_before/1e6:.6f}")
    print(f"  Adapter: {NEGRISK_ADAPTER}")
    print(f"{'='*80}")

    # Fetch positions
    r = requests.get("https://data-api.polymarket.com/positions",
                     params={"user": wallet.lower()}, timeout=15)
    positions = r.json()

    redeemable = []

    for pos in positions:
        asset_id = pos.get("asset", "")
        cond_id = pos.get("conditionId", "")
        outcome = pos.get("outcome", "")
        size = float(pos.get("size", 0))
        title = pos.get("title", "")[:55]

        if size <= 0 or not cond_id:
            continue

        cid_bytes = bytes.fromhex(cond_id[2:] if cond_id.startswith("0x") else cond_id)

        # Check on-chain balance
        try:
            balance = ctf.functions.balanceOf(wallet, int(asset_id)).call()
        except Exception:
            balance = 0

        if balance == 0:
            continue

        # Check resolution
        try:
            denom = ctf.functions.payoutDenominator(cid_bytes).call()
            our_idx = 0 if outcome == "Yes" else 1
            our_num = ctf.functions.payoutNumerators(cid_bytes, our_idx).call()
        except Exception:
            continue

        if denom == 0 or our_num == 0:
            continue

        expected = (our_num / denom) * (balance / 1e6)

        print(f"\n  REDEEMABLE: {title}")
        print(f"    Outcome: {outcome} | Balance: {balance/1e6:.6f}")
        print(f"    Payout: {our_num}/{denom} = ${expected:.6f}")
        print(f"    ConditionId: {cond_id}")
        print(f"    Asset ID: {asset_id[:30]}...")

        redeemable.append({
            "title": title,
            "cond_id": cond_id,
            "cid_bytes": cid_bytes,
            "outcome": outcome,
            "our_idx": our_idx,
            "balance": balance,
            "expected": expected,
            "asset_id": asset_id,
        })

    if not redeemable:
        print(f"\n  Nothing to redeem.")
        return

    total = sum(r["expected"] for r in redeemable)
    print(f"\n  Total redeemable: ${total:.2f}")

    if not args.execute:
        print(f"\n  Dry run. Add --execute to redeem via NegRisk Adapter.")
        return

    # Execute through NegRisk Adapter
    print(f"\n  {'='*60}")
    print(f"  EXECUTING via NegRisk Adapter")
    print(f"  {'='*60}")

    for r in redeemable:
        print(f"\n  -- {r['title']} --")

        usdc_pre = usdc_c.functions.balanceOf(wallet).call()
        print(f"  USDC.e before: ${usdc_pre/1e6:.6f}")

        # Try 1: CTF direct with WRAPPED collateral token (NegRisk system)
        # The NegRisk system uses a wrapped collateral, not USDC.e directly
        WRAPPED_COLLATERAL = "0x3a3bd7bb9528e159577f7c2e685cc81a765002e2"
        print(f"  Attempt 1: CTF.redeemPositions(collateral=WRAPPED, parent=0x0, indexSets=[1,2])")

        ctf_redeem = w3.eth.contract(
            address=w3.to_checksum_address(CTF_ADDRESS), abi=REDEEM_ABI)
        try:
            tx = ctf_redeem.functions.redeemPositions(
                w3.to_checksum_address(WRAPPED_COLLATERAL),
                bytes(32),  # parentCollectionId = 0x0
                r["cid_bytes"],
                [1, 2],
            ).build_transaction({
                "from": wallet,
                "nonce": w3.eth.get_transaction_count(wallet),
                "gas": 500000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })

            signed = acct.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  Tx: {tx_hash.hex()}")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            print(f"  Status: {receipt.status} | Gas: {receipt.gasUsed} | Logs: {len(receipt.logs)}")

            # Check if we got wrapped tokens (not USDC directly)
            # Also check USDC in case the system auto-unwraps
            time.sleep(2)
            usdc_post = usdc_c.functions.balanceOf(wallet).call()
            delta = (usdc_post - usdc_pre) / 1e6
            print(f"  USDC.e after:  ${usdc_post/1e6:.6f} (delta: ${delta:.6f})")

            # Check wrapped token balance
            try:
                wrapped_c = w3.eth.contract(
                    address=w3.to_checksum_address(WRAPPED_COLLATERAL), abi=USDC_ABI)
                wrapped_bal = wrapped_c.functions.balanceOf(wallet).call()
                print(f"  Wrapped token balance: {wrapped_bal}")
                if wrapped_bal > 0:
                    print(f"  Got wrapped tokens! Need to unwrap to USDC.e")
                    print(f"  Check NegRisk Adapter for unwrap function")
            except Exception as e:
                print(f"  Wrapped token check: {e}")

            if delta > 0.001:
                print(f"  SUCCESS! Received ${delta:.6f} USDC.e")
                continue
            elif receipt.status == 1 and len(receipt.logs) > 1:
                print(f"  Tx succeeded with events -- may have received wrapped tokens")
                print(f"  Check tx: https://polygonscan.com/tx/{tx_hash.hex()}")
                continue
            else:
                print(f"  No USDC from Attempt 1")

        except Exception as e:
            print(f"  Attempt 1 error: {e}")

        # Try 2: NegRisk Adapter with USDC.e (in case adapter handles unwrap)
        print(f"\n  Attempt 2: Adapter.redeemPositions(USDC, parent=0x0, indexSets=[1,2])")

        try:
            tx = adapter.functions.redeemPositions(
                w3.to_checksum_address(USDC_E),
                bytes(32),  # parentCollectionId = 0x0
                r["cid_bytes"],
                [1, 2],
            ).build_transaction({
                "from": wallet,
                "nonce": w3.eth.get_transaction_count(wallet),
                "gas": 500000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })

            signed = acct.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  Tx: {tx_hash.hex()}")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            print(f"  Status: {receipt.status} | Gas: {receipt.gasUsed}")
            print(f"  Logs: {len(receipt.logs)}")

            time.sleep(2)  # Wait for state to settle

            usdc_post = usdc_c.functions.balanceOf(wallet).call()
            delta = (usdc_post - usdc_pre) / 1e6
            print(f"  USDC.e after:  ${usdc_post/1e6:.6f}")
            print(f"  Delta:         ${delta:.6f}")

            if delta > 0.001:
                print(f"  SUCCESS! Received ${delta:.6f}")
                continue
            else:
                print(f"  No USDC received from Attempt 1")

        except Exception as e:
            print(f"  Attempt 1 failed: {e}")

        # Try 2: Adapter with indexSets=[1] only (just our side)
        print(f"\n  Attempt 2: Adapter.redeemPositions(parent=0x0, indexSets=[{1 if r['outcome']=='Yes' else 2}])")

        idx_set = [1] if r["outcome"] == "Yes" else [2]
        try:
            tx = adapter.functions.redeemPositions(
                w3.to_checksum_address(USDC_E),
                bytes(32),
                r["cid_bytes"],
                idx_set,
            ).build_transaction({
                "from": wallet,
                "nonce": w3.eth.get_transaction_count(wallet),
                "gas": 500000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })

            signed = acct.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  Tx: {tx_hash.hex()}")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            print(f"  Status: {receipt.status} | Gas: {receipt.gasUsed}")

            time.sleep(2)

            usdc_post = usdc_c.functions.balanceOf(wallet).call()
            delta = (usdc_post - usdc_pre) / 1e6
            print(f"  USDC.e after:  ${usdc_post/1e6:.6f}")
            print(f"  Delta:         ${delta:.6f}")

            if delta > 0.001:
                print(f"  SUCCESS! Received ${delta:.6f}")
                continue
            else:
                print(f"  No USDC received from Attempt 2")

        except Exception as e:
            print(f"  Attempt 2 failed: {e}")

        # Try 3: CTF direct with indexSets=[1] only
        print(f"\n  Attempt 3: CTF.redeemPositions(parent=0x0, indexSets=[{1 if r['outcome']=='Yes' else 2}])")

        ctf_redeem = w3.eth.contract(
            address=w3.to_checksum_address(CTF_ADDRESS), abi=REDEEM_ABI)
        try:
            tx = ctf_redeem.functions.redeemPositions(
                w3.to_checksum_address(USDC_E),
                bytes(32),
                r["cid_bytes"],
                idx_set,
            ).build_transaction({
                "from": wallet,
                "nonce": w3.eth.get_transaction_count(wallet),
                "gas": 300000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })

            signed = acct.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  Tx: {tx_hash.hex()}")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            print(f"  Status: {receipt.status} | Gas: {receipt.gasUsed}")

            time.sleep(2)

            usdc_post = usdc_c.functions.balanceOf(wallet).call()
            delta = (usdc_post - usdc_pre) / 1e6
            print(f"  USDC.e after:  ${usdc_post/1e6:.6f}")
            print(f"  Delta:         ${delta:.6f}")

            if delta > 0.001:
                print(f"  SUCCESS! Received ${delta:.6f}")
            else:
                print(f"  All attempts failed for this position.")
                print(f"  Check: https://polygonscan.com/address/{wallet}#tokentxnsErc1155")

        except Exception as e:
            print(f"  Attempt 3 failed: {e}")

    # Final balance
    usdc_final = usdc_c.functions.balanceOf(wallet).call()
    total_delta = (usdc_final - usdc_before) / 1e6
    print(f"\n  {'='*60}")
    print(f"  FINAL USDC.e: ${usdc_final/1e6:.6f} (delta: ${total_delta:.6f})")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()
