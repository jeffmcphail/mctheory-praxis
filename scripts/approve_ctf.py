"""
scripts/approve_ctf.py — Approve CTF Exchange to transfer your conditional tokens.

One-time setup: allows the CTF Exchange contract to move your YES/NO tokens
so you can sell positions and merge YES+NO pairs back into USDC.

Without this approval, you'll get:
  "not enough balance / allowance: the allowance is not enough"

Usage:
    python -m scripts.approve_ctf             # Dry run (show what would happen)
    python -m scripts.approve_ctf --execute   # Execute the approval transaction
"""
import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()


# Contract addresses (Polygon)
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"       # Conditional Tokens
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"      # CTF Exchange (spender)
NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # NegRisk Exchange
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"   # NegRisk Adapter
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"

# Minimal ABI for setApprovalForAll and isApprovedForAll
ERC1155_ABI = [
    {
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"}
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "operator", "type": "address"}
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]


def main():
    parser = argparse.ArgumentParser(description="Approve CTF Exchange for token transfers")
    parser.add_argument("--execute", action="store_true", help="Actually send the transaction")
    args = parser.parse_args()

    from web3 import Web3

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("❌ POLYMARKET_PRIVATE_KEY not set in .env")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    if not w3.is_connected():
        print(f"❌ Cannot connect to {POLYGON_RPC}")
        sys.exit(1)

    account = w3.eth.account.from_key(pk)
    wallet = account.address
    print(f"Wallet: {wallet}")

    # Get CTF contract
    ctf = w3.eth.contract(
        address=Web3.to_checksum_address(CTF_CONTRACT),
        abi=ERC1155_ABI
    )

    # Check current approval status
    approvals_needed = []

    for name, spender in [
        ("CTF Exchange", CTF_EXCHANGE),
        ("NegRisk Exchange", NEG_RISK_EXCHANGE),
        ("NegRisk Adapter", NEG_RISK_ADAPTER),
    ]:
        is_approved = ctf.functions.isApprovedForAll(
            wallet, Web3.to_checksum_address(spender)
        ).call()
        status = "✅ Approved" if is_approved else "❌ Not approved"
        print(f"  {name}: {status}")
        if not is_approved:
            approvals_needed.append((name, spender))

    if not approvals_needed:
        print("\n✅ All approvals already set! You can sell and merge tokens.")
        return

    print(f"\n{len(approvals_needed)} approval(s) needed.")

    if not args.execute:
        print("\nDry run — add --execute to send transactions.")
        print("Each approval costs ~0.005 POL in gas.")
        return

    # Execute approvals
    for name, spender in approvals_needed:
        print(f"\n  Approving {name} ({spender[:10]}...)...")
        try:
            tx = ctf.functions.setApprovalForAll(
                Web3.to_checksum_address(spender),
                True
            ).build_transaction({
                "from": wallet,
                "nonce": w3.eth.get_transaction_count(wallet),
                "gas": 100000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })

            signed = w3.eth.account.sign_transaction(tx, pk)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  Tx sent: {tx_hash.hex()[:24]}...")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            if receipt.status == 1:
                gas_cost = receipt.gasUsed * receipt.effectiveGasPrice / 1e18
                print(f"  ✅ {name} approved! Gas: {receipt.gasUsed} ({gas_cost:.6f} POL)")
            else:
                print(f"  ❌ Transaction reverted!")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Verify
    print(f"\nVerifying...")
    for name, spender in [
        ("CTF Exchange", CTF_EXCHANGE),
        ("NegRisk Exchange", NEG_RISK_EXCHANGE),
        ("NegRisk Adapter", NEG_RISK_ADAPTER),
    ]:
        is_approved = ctf.functions.isApprovedForAll(
            wallet, Web3.to_checksum_address(spender)
        ).call()
        status = "✅" if is_approved else "❌"
        print(f"  {status} {name}")

    print("\nDone! You can now sell tokens and merge YES+NO pairs.")


if __name__ == "__main__":
    main()
