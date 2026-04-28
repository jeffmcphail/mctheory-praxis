"""Deposit USDC from MetaMask wallet into Polymarket exchange."""
import os
import json
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

print("Connecting to Polymarket...")
client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID)
client.set_api_creds(client.derive_api_key())

# Check current allowances
print("\nStep 1: Checking allowances...")
try:
    allowances = client.get_balance_allowance()
    print(f"  Balance/Allowance: {allowances}")
except Exception as e:
    print(f"  {e}")

# Set max allowances (approve USDC spending)
print("\nStep 2: Setting allowances (approve USDC for exchange)...")
try:
    result = client.set_allowances()
    print(f"  Allowances set: {result}")
except Exception as e:
    print(f"  set_allowances: {e}")

# Try to update balance allowance
print("\nStep 3: Updating balance allowance...")
try:
    result = client.update_balance_allowance()
    print(f"  Balance allowance updated: {result}")
except Exception as e:
    print(f"  update_balance_allowance: {e}")

# Check available methods for depositing
print("\nAvailable client methods related to balance/deposit:")
methods = [m for m in dir(client) if not m.startswith("_") and 
           any(w in m.lower() for w in ["balance", "deposit", "fund", "allow", "approv", "transfer"])]
for m in methods:
    print(f"  {m}")

# Re-check balance
print("\nStep 4: Re-checking balance...")
try:
    allowances = client.get_balance_allowance()
    print(f"  Balance/Allowance: {allowances}")
except Exception as e:
    print(f"  {e}")
