"""
scripts/decode_parent.py -- Extract parentCollectionId from trade transactions

The PositionSplit event has parentCollectionId as an indexed topic:
  event PositionSplit(
    address indexed stakeholder,
    address collateralToken,
    bytes32 indexed parentCollectionId,
    bytes32 indexed conditionId,
    uint[] partition,
    uint amount
  )

topic[0] = event signature hash
topic[1] = stakeholder (address)
topic[2] = parentCollectionId  <-- WHAT WE NEED
topic[3] = conditionId
"""
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"

CTF_ABI = json.loads("""[
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"collectionId","type":"bytes32"}],"name":"getPositionId","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSet","type":"uint256"}],"name":"getCollectionId","outputs":[{"name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"}
]""")

# Known event signatures
POSITION_SPLIT = Web3.keccak(text="PositionSplit(address,address,bytes32,bytes32,uint256[],uint256)").hex()
POSITION_MERGE = Web3.keccak(text="PositionsMerge(address,address,bytes32,bytes32,uint256[],uint256)").hex()
TRANSFER_SINGLE = Web3.keccak(text="TransferSingle(address,address,address,uint256,uint256)").hex()
TRANSFER_BATCH = Web3.keccak(text="TransferBatch(address,address,address,uint256[],uint256[])").hex()
PAYOUT_REDEMPTION = Web3.keccak(text="PayoutRedemption(address,address,bytes32,bytes32,uint256[],uint256)").hex()

EVENT_NAMES = {
    POSITION_SPLIT: "PositionSplit",
    POSITION_MERGE: "PositionsMerge",
    TRANSFER_SINGLE: "TransferSingle",
    TRANSFER_BATCH: "TransferBatch",
    PAYOUT_REDEMPTION: "PayoutRedemption",
}

# Transactions to decode
TXS = {
    "CPI monthly": "0x8751652ad5f3e3e47094e46d555db60bd11a7fba95bd02a3f34d138f0b85f6dc",
    "Thunder": "0x4884f7e5e3f921899148b82f72f2613ac5c18e19aaa69e7b78bc6e3e74bb6f0d",
    "76ers (worked)": "0xe7e6a72989e1245f09362282d906333ffe2430715b89c7f47a9a079acd376238",
}


def main():
    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    ctf = w3.eth.contract(address=w3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    wallet = w3.eth.account.from_key(pk).address if pk else "?"

    print(f"\n{'='*80}")
    print(f"  PARENT COLLECTION ID DECODER")
    print(f"  Wallet: {wallet}")
    print(f"{'='*80}")

    found_parents = {}

    for label, tx_hash in TXS.items():
        print(f"\n  {'='*70}")
        print(f"  {label}: {tx_hash[:20]}...")

        try:
            receipt = w3.eth.get_transaction_receipt(tx_hash)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        print(f"  Total logs: {len(receipt.logs)}")

        # Decode ALL CTF logs with full topics
        for li, log in enumerate(receipt.logs):
            if log.address.lower() != CTF_ADDRESS.lower():
                continue

            topic0 = log.topics[0].hex() if log.topics else ""
            event_name = EVENT_NAMES.get(topic0, f"Unknown({topic0[:16]}...)")

            print(f"\n    Log {li}: {event_name}")
            for ti, t in enumerate(log.topics):
                print(f"      topic[{ti}]: {t.hex()}")

            if log.data:
                data_hex = log.data.hex()
                # Show first 256 chars of data in 64-char chunks (32 bytes each)
                for chunk_i in range(0, min(len(data_hex), 512), 64):
                    chunk = data_hex[chunk_i:chunk_i+64]
                    print(f"      data[{chunk_i//64}]: {chunk}")

            # Extract parentCollectionId from PositionSplit
            if event_name == "PositionSplit" and len(log.topics) >= 4:
                parent = log.topics[2].hex()
                cond = log.topics[3].hex()
                print(f"\n      >>> parentCollectionId = 0x{parent}")
                print(f"      >>> conditionId        = 0x{cond}")

                if parent != "0" * 64:
                    print(f"      >>> NON-ZERO PARENT FOUND!")
                    found_parents[label] = {
                        "parent": "0x" + parent,
                        "condition": "0x" + cond,
                    }

            # Extract from PayoutRedemption too
            if event_name == "PayoutRedemption" and len(log.topics) >= 3:
                # PayoutRedemption(address indexed redeemer, address collateralToken,
                #   bytes32 indexed parentCollectionId, bytes32 indexed conditionId,
                #   uint256[] indexSets, uint256 payout)
                if len(log.topics) >= 4:
                    parent = log.topics[2].hex()
                    cond = log.topics[3].hex()
                    print(f"\n      >>> Redemption parentCollectionId = 0x{parent}")
                    print(f"      >>> Redemption conditionId        = 0x{cond}")

    # Summary and verification
    print(f"\n\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")

    if found_parents:
        for label, info in found_parents.items():
            print(f"\n  {label}:")
            print(f"    parentCollectionId: {info['parent']}")
            print(f"    conditionId:        {info['condition']}")

            # Verify: compute positionId with this parent
            parent_bytes = bytes.fromhex(info['parent'][2:])
            cond_bytes = bytes.fromhex(info['condition'][2:])

            for idx in [1, 2]:
                try:
                    coll = ctf.functions.getCollectionId(
                        parent_bytes, cond_bytes, idx).call()
                    pid = ctf.functions.getPositionId(
                        w3.to_checksum_address(USDC_E), coll).call()
                    bal = ctf.functions.balanceOf(wallet, pid).call()
                    print(f"    indexSet={idx}: positionId={str(pid)[:30]}... "
                          f"balance={bal/1e6:.6f}")
                    if bal > 0:
                        print(f"    >>> THIS IS THE CORRECT REDEMPTION PATH!")
                        print(f"    >>> redeemPositions(USDC, {info['parent']}, "
                              f"{info['condition']}, [{idx}])")
                except Exception as e:
                    print(f"    indexSet={idx}: error {e}")
    else:
        print(f"  No PositionSplit events found in CTF logs.")
        print(f"  The positions may have been created via a different mechanism.")
        print(f"  Check if the Exchange contract (0xB768...) wraps the CTF calls.")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
