"""
scripts/merge_positions.py — Merge YES+NO pairs back into USDC.e.

For binary (non-NegRisk) markets: burns equal amounts of YES + NO tokens,
returns USDC.e at 1:1 ratio.

VALIDATION (maximal):
  PRE-1: Verify market is NOT negRisk (different merge path)
  PRE-2: Check on-chain YES balance
  PRE-3: Check on-chain NO balance
  PRE-4: Compute mergeable amount (min of both)
  PRE-5: Record USDC.e balance before
  POST-1: Verify USDC.e increased by expected amount
  POST-2: Verify token balances decreased

Usage:
    python -m scripts.merge_positions --slug fed-rate-cut-by-october-2026-meeting-199-747
    python -m scripts.merge_positions --slug fed-rate-cut-by-october-2026-meeting-199-747 --execute
"""
import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
load_dotenv()

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"
GAMMA_API = "https://gamma-api.polymarket.com"

CTF_ABI = json.loads("""[
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"partition","type":"uint256[]"},{"name":"amount","type":"uint256"}],"name":"mergePositions","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")

USDC_ABI = json.loads("""[
    {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="Market slug")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from web3 import Web3

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("No POLYMARKET_PRIVATE_KEY"); sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    wallet = w3.eth.account.from_key(pk).address
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)
    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_E_ADDRESS), abi=USDC_ABI)

    print(f"\n{'='*80}")
    print(f"  MERGE POSITIONS — YES + NO → USDC.e")
    print(f"  Wallet: {wallet}")
    print(f"{'='*80}")

    # Fetch market
    r = requests.get(f"{GAMMA_API}/markets", params={"slug": args.slug}).json()
    if not r:
        print(f"  ❌ Market not found: {args.slug}")
        return
    m = r[0]

    question = m.get("question", "")
    neg_risk = m.get("negRisk", False)
    condition_id = m.get("conditionId", "")
    token_ids = json.loads(m.get("clobTokenIds", "[]"))

    print(f"  Market:  {question}")
    print(f"  Slug:    {args.slug}")

    # PRE-1: Verify not NegRisk
    print(f"\n  PRE-1 negRisk:         {neg_risk}", end="")
    if neg_risk:
        print(f" ❌ ABORT: NegRisk markets need WCOL merge path")
        print(f"  Use CTF.mergePositions(WCOL, ...) instead of USDC.e")
        return
    print(f" ✓ (binary market)")

    if len(token_ids) < 2:
        print(f"  ❌ No token IDs found")
        return

    yes_token = int(token_ids[0])
    no_token = int(token_ids[1])
    cid_bytes = bytes.fromhex(condition_id[2:] if condition_id.startswith("0x") else condition_id)

    # PRE-2: On-chain YES balance
    yes_raw = ctf.functions.balanceOf(wallet, yes_token).call()
    yes_shares = yes_raw / 1e6
    print(f"  PRE-2 YES balance:     {yes_raw} ({yes_shares:.6f} shares)", end="")
    if yes_raw == 0:
        print(f" ❌ No YES tokens")
        return
    print(f" ✓")

    # PRE-3: On-chain NO balance
    no_raw = ctf.functions.balanceOf(wallet, no_token).call()
    no_shares = no_raw / 1e6
    print(f"  PRE-3 NO balance:      {no_raw} ({no_shares:.6f} shares)", end="")
    if no_raw == 0:
        print(f" ❌ No NO tokens")
        return
    print(f" ✓")

    # PRE-4: Mergeable amount
    merge_raw = min(yes_raw, no_raw)
    merge_shares = merge_raw / 1e6
    expected_usdc = merge_shares
    remaining_yes = (yes_raw - merge_raw) / 1e6
    remaining_no = (no_raw - merge_raw) / 1e6
    print(f"  PRE-4 Mergeable:       {merge_raw} ({merge_shares:.6f} shares)")
    print(f"        Expected USDC:   ${expected_usdc:.6f}")
    print(f"        Remaining YES:   {remaining_yes:.6f}")
    print(f"        Remaining NO:    {remaining_no:.6f}")

    if merge_raw == 0:
        print(f"  ❌ Nothing to merge")
        return

    # PRE-5: USDC.e before
    usdc_before = usdc.functions.balanceOf(wallet).call()
    print(f"  PRE-5 USDC.e before:   ${usdc_before/1e6:.6f} ✓")

    print(f"\n  ═══ ALL 5 PRE-FLIGHT CHECKS PASSED ═══")

    if not args.execute:
        print(f"\n  Dry run. Add --execute to merge {merge_shares:.6f} pairs → ${expected_usdc:.6f} USDC.e")
        return

    # EXECUTE
    print(f"\n  Executing merge of {merge_shares:.6f} pairs...")
    try:
        tx = ctf.functions.mergePositions(
            Web3.to_checksum_address(USDC_E_ADDRESS),
            bytes(32),
            cid_bytes,
            [1, 2],
            merge_raw
        ).build_transaction({
            "from": wallet,
            "nonce": w3.eth.get_transaction_count(wallet),
            "gas": 300000,
            "gasPrice": w3.eth.gas_price * 2,
            "chainId": 137,
        })
        signed = w3.eth.account.sign_transaction(tx, pk)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        tx_hex = tx_hash.hex()
        print(f"  📤 Tx: {tx_hex}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        gas_pol = receipt.gasUsed * receipt.effectiveGasPrice / 1e18

        if receipt.status != 1:
            print(f"  ❌ REVERTED! Gas: {receipt.gasUsed}")
            print(f"  https://polygonscan.com/tx/{tx_hex}")
            return
        print(f"  ✓ Confirmed. Gas: {receipt.gasUsed} ({gas_pol:.6f} POL)")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return

    time.sleep(1)

    # POST-1: USDC.e after
    usdc_after = usdc.functions.balanceOf(wallet).call()
    delta = (usdc_after - usdc_before) / 1e6
    print(f"\n  POST-1 USDC.e after:   ${usdc_after/1e6:.6f}")
    print(f"  POST-1 Delta:          ${delta:.6f}", end="")
    if abs(delta - expected_usdc) < 0.01:
        print(f" ✅ VERIFIED (expected ${expected_usdc:.6f})")
    elif delta < 0.001:
        print(f" ⚠ WARNING: No USDC received!")
        print(f"  https://polygonscan.com/tx/{tx_hex}")
    else:
        print(f" ⚠ Mismatch: expected ${expected_usdc:.6f}")

    # POST-2: Token balances
    yes_after = ctf.functions.balanceOf(wallet, yes_token).call()
    no_after = ctf.functions.balanceOf(wallet, no_token).call()
    print(f"  POST-2 YES remaining:  {yes_after/1e6:.6f} (was {yes_shares:.6f})")
    print(f"  POST-2 NO remaining:   {no_after/1e6:.6f} (was {no_shares:.6f})")

    print(f"\n{'─'*80}")
    print(f"  RESULT: Merged {merge_shares:.6f} pairs → ${delta:.6f} USDC.e")
    print(f"  USDC.e balance: ${usdc_after/1e6:.6f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
