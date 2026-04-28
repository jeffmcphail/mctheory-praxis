"""Check what happened in Polymarket redemption txs using web3 RPC."""
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))

txs = [
    ("CPI monthly (failed)", "0x289e921b303a4502f2d52ddb8806e5df39ab3164188c913a46ea59870e02dcfc"),
    ("Thunder (failed)", "0x53cbbb6e2beaf69b9b1a510295c9479ae786d6b6cd2cb141618e02fbaf922aa6"),
    ("CPI prev attempt", "0x69022246da6019665a20dcc703f695cac9e53d5d4b50be40a966b96794c364ff"),
    ("76ers (worked)", "0xe7e6a72989e1245f09362282d906333ffe2430715b89c7f47a9a079acd376238"),
]

for label, tx_hash in txs:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {tx_hash}")

    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    print(f"  Status: {receipt.status} ({'SUCCESS' if receipt.status == 1 else 'REVERTED'})")
    print(f"  Gas used: {receipt.gasUsed}")
    print(f"  Logs: {len(receipt.logs)}")

    for i, log in enumerate(receipt.logs):
        print(f"    Log {i}: from {log.address}")
        for ti, t in enumerate(log.topics[:3]):
            print(f"      topic[{ti}]: {t.hex()}")
        if log.data and len(log.data) > 2:
            print(f"      data: {log.data.hex()[:80]}...")

    if len(receipt.logs) == 0:
        print(f"  --> NO EVENTS: redeemPositions was a no-op")
        print(f"      The function ran but found nothing to redeem.")
        print(f"      Likely: wrong parentCollectionId or indexSets")

        # Also check the tx input to see what was called
        try:
            tx = w3.eth.get_transaction(tx_hash)
            input_data = tx.input.hex()
            print(f"      Function selector: {input_data[:10]}")
            print(f"      Input length: {len(input_data)} chars")
            # redeemPositions selector = 0x01864fcf
            if input_data[:10] == "0x01864fcf":
                print(f"      Confirmed: redeemPositions call")
                # Decode: collateralToken(32) + parentCollectionId(32) + conditionId(32) + offset + length + indexSets
                if len(input_data) > 10:
                    params = input_data[10:]
                    collateral = "0x" + params[24:64]
                    parent = "0x" + params[64:128]
                    condition = "0x" + params[128:192]
                    print(f"      collateralToken:    {collateral}")
                    print(f"      parentCollectionId: {parent}")
                    print(f"      conditionId:        {condition}")
                    if parent == "0x" + "0" * 64:
                        print(f"      parentCollectionId is ZERO -- standard binary market")
                    else:
                        print(f"      parentCollectionId is NON-ZERO -- NegRisk sub-market!")
        except Exception as e:
            print(f"      Could not decode tx: {e}")
    else:
        print(f"  --> EVENTS EMITTED: redemption did something")
