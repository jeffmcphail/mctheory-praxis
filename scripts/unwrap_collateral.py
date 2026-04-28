"""
scripts/unwrap_collateral.py -- Convert wrapped NegRisk collateral back to USDC.e

The wrapped token at 0x3a3bd7bb... is redeemable 1:1 for USDC.e.
Need to find the correct unwrap mechanism.
"""
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

WRAPPED_TOKEN = "0x3a3bd7bb9528e159577f7c2e685cc81a765002e2"
NEGRISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"

# Try common ERC-20 + withdraw/redeem ABI
TOKEN_ABI = json.loads("""[
    {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"amount","type":"uint256"}],"name":"withdraw","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"to","type":"address"},{"name":"amount","type":"uint256"}],"name":"withdraw","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"amount","type":"uint256"}],"name":"redeem","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"shares","type":"uint256"}],"name":"burn","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"to","type":"address"},{"name":"amount","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"}
]""")

USDC_ABI = json.loads("""[
    {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")


def check_function_exists(w3, address, selector):
    """Check if a function selector exists on a contract."""
    try:
        result = w3.eth.call({
            "to": address,
            "data": selector + "0" * 56,  # Pad to 32 bytes
        })
        return True
    except Exception:
        return False


def main():
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("No POLYMARKET_PRIVATE_KEY"); sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    acct = w3.eth.account.from_key(pk)
    wallet = acct.address

    wrapped = w3.eth.contract(address=w3.to_checksum_address(WRAPPED_TOKEN), abi=TOKEN_ABI)
    usdc_c = w3.eth.contract(address=w3.to_checksum_address(USDC_E), abi=USDC_ABI)

    print(f"\n{'='*70}")
    print(f"  UNWRAP NEGRISK COLLATERAL")
    print(f"  Wallet: {wallet}")
    print(f"{'='*70}")

    # Step 1: Check wrapped token info
    print(f"\n  Step 1: Inspect wrapped token at {WRAPPED_TOKEN}")
    try:
        name = wrapped.functions.name().call()
        print(f"    Name: {name}")
    except Exception:
        print(f"    Name: (not available)")

    try:
        symbol = wrapped.functions.symbol().call()
        print(f"    Symbol: {symbol}")
    except Exception:
        print(f"    Symbol: (not available)")

    try:
        decimals = wrapped.functions.decimals().call()
        print(f"    Decimals: {decimals}")
    except Exception:
        decimals = 6
        print(f"    Decimals: (assuming 6)")

    balance = wrapped.functions.balanceOf(wallet).call()
    print(f"    Your balance: {balance} ({balance / 10**decimals:.6f})")

    usdc_before = usdc_c.functions.balanceOf(wallet).call()
    print(f"    USDC.e before: ${usdc_before/1e6:.6f}")

    if balance == 0:
        print(f"\n  No wrapped tokens to unwrap.")
        return

    # Step 2: Probe for unwrap functions
    print(f"\n  Step 2: Probing for unwrap functions...")

    # Check known function selectors
    selectors = {
        "withdraw(uint256)": "0x2e1a7d4d",
        "withdraw(address,uint256)": "0xf3fef3a3",
        "redeem(uint256)": "0xdb006a75",
        "burn(uint256)": "0x42966c68",
        "convertToUSDC(uint256)": "0x",  # Unknown
    }

    for fname, sel in selectors.items():
        if sel == "0x":
            continue
        exists = check_function_exists(w3, w3.to_checksum_address(WRAPPED_TOKEN), sel)
        print(f"    {fname}: {'EXISTS' if exists else 'no'}")

    # Step 3: Try withdraw(uint256)
    print(f"\n  Step 3: Attempting withdraw({balance})...")
    try:
        tx = wrapped.functions.withdraw(balance).build_transaction({
            "from": wallet,
            "nonce": w3.eth.get_transaction_count(wallet),
            "gas": 200000,
            "gasPrice": w3.eth.gas_price,
            "chainId": 137,
        })

        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"    Tx: {tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        print(f"    Status: {receipt.status} | Gas: {receipt.gasUsed} | Logs: {len(receipt.logs)}")

        time.sleep(2)

        usdc_after = usdc_c.functions.balanceOf(wallet).call()
        wrapped_after = wrapped.functions.balanceOf(wallet).call()
        delta = (usdc_after - usdc_before) / 1e6

        print(f"    USDC.e after:   ${usdc_after/1e6:.6f} (delta: ${delta:.6f})")
        print(f"    Wrapped after:  {wrapped_after}")

        if delta > 0.001:
            print(f"\n    SUCCESS! Unwrapped ${delta:.6f} USDC.e")
            return
        elif receipt.status == 0:
            print(f"    withdraw(uint256) reverted")
        else:
            print(f"    withdraw succeeded but no USDC delta")

    except Exception as e:
        print(f"    withdraw(uint256) error: {e}")

    # Step 4: Try approve + adapter interaction
    print(f"\n  Step 4: Try approve wrapped to Adapter, then Adapter withdraw...")
    try:
        # Approve adapter to spend our wrapped tokens
        tx = wrapped.functions.approve(
            w3.to_checksum_address(NEGRISK_ADAPTER), balance
        ).build_transaction({
            "from": wallet,
            "nonce": w3.eth.get_transaction_count(wallet),
            "gas": 100000,
            "gasPrice": w3.eth.gas_price,
            "chainId": 137,
        })
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        print(f"    Approve tx: {tx_hash.hex()} status={receipt.status}")

        if receipt.status == 1:
            # Try calling adapter with withdraw-like selectors
            # NegRisk Adapter might have a specific unwrap function
            # Try raw call with withdraw selector
            for sel_name, sel_hex in [
                ("withdraw(uint256)", "0x2e1a7d4d"),
                ("redeem(uint256)", "0xdb006a75"),
            ]:
                try:
                    data = bytes.fromhex(sel_hex[2:]) + balance.to_bytes(32, "big")
                    tx = {
                        "from": wallet,
                        "to": w3.to_checksum_address(NEGRISK_ADAPTER),
                        "data": data,
                        "nonce": w3.eth.get_transaction_count(wallet),
                        "gas": 300000,
                        "gasPrice": w3.eth.gas_price,
                        "chainId": 137,
                    }
                    signed = acct.sign_transaction(tx)
                    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                    print(f"    Adapter.{sel_name}: status={receipt.status} "
                          f"gas={receipt.gasUsed}")

                    if receipt.status == 1:
                        time.sleep(2)
                        usdc_after = usdc_c.functions.balanceOf(wallet).call()
                        delta = (usdc_after - usdc_before) / 1e6
                        if delta > 0.001:
                            print(f"    SUCCESS via Adapter.{sel_name}! "
                                  f"Got ${delta:.6f} USDC.e")
                            return
                except Exception as e:
                    print(f"    Adapter.{sel_name}: {e}")

    except Exception as e:
        print(f"    Approve/adapter error: {e}")

    # Step 5: Try just transferring wrapped tokens to adapter (some unwrap on receive)
    print(f"\n  Step 5: Transfer wrapped tokens to Adapter (some contracts auto-unwrap)...")
    wrapped_bal = wrapped.functions.balanceOf(wallet).call()
    if wrapped_bal > 0:
        try:
            tx = wrapped.functions.transfer(
                w3.to_checksum_address(NEGRISK_ADAPTER), wrapped_bal
            ).build_transaction({
                "from": wallet,
                "nonce": w3.eth.get_transaction_count(wallet),
                "gas": 200000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })
            signed = acct.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            print(f"    Transfer tx: {tx_hash.hex()} status={receipt.status}")

            time.sleep(2)
            usdc_after = usdc_c.functions.balanceOf(wallet).call()
            delta = (usdc_after - usdc_before) / 1e6
            print(f"    USDC.e after: ${usdc_after/1e6:.6f} (delta: ${delta:.6f})")

            if delta > 0.001:
                print(f"    SUCCESS! Auto-unwrap worked!")
            else:
                print(f"    No auto-unwrap. Tokens sent to adapter.")
                print(f"    WARNING: May need to recover from adapter")
        except Exception as e:
            print(f"    Transfer error: {e}")

    print(f"\n  Final USDC.e: ${usdc_c.functions.balanceOf(wallet).call()/1e6:.6f}")
    print(f"  Final wrapped: {wrapped.functions.balanceOf(wallet).call()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
