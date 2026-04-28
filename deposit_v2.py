"""Deposit USDC into Polymarket — with correct signature type."""
import os
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
funder = os.getenv("POLYMARKET_API_ADDRESS", "")

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

# Signature types: 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE
print("Connecting with POLY_PROXY signature type...")
client = ClobClient(
    HOST,
    key=pk,
    chain_id=CHAIN_ID,
    signature_type=1,  # POLY_PROXY
    funder=funder,
)
client.set_api_creds(client.derive_api_key())

print(f"  Funder (your wallet): {funder}")

# Check balance
print("\nChecking balance...")
try:
    bal = client.get_balance_allowance()
    print(f"  Balance/Allowance: {bal}")
except Exception as e:
    print(f"  Error: {e}")

# Try EOA signature type instead
print("\nTrying EOA signature type...")
client2 = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=0)
client2.set_api_creds(client2.derive_api_key())

try:
    bal = client2.get_balance_allowance()
    print(f"  Balance/Allowance: {bal}")
except Exception as e:
    print(f"  Error: {e}")

# Try update_balance_allowance with proxy
print("\nApproving USDC (proxy)...")
try:
    result = client.update_balance_allowance(allowance=2**128)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {e}")

# Try update_balance_allowance with EOA
print("\nApproving USDC (EOA)...")
try:
    result = client2.update_balance_allowance(allowance=2**128)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {e}")

# Re-check
print("\nRe-checking balance (proxy)...")
try:
    bal = client.get_balance_allowance()
    print(f"  Balance/Allowance: {bal}")
except Exception as e:
    print(f"  Error: {e}")
