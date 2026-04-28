"""Derive CLOB API credentials from private key."""
import os
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
client = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137)

creds = client.derive_api_key()
print(f"Type: {type(creds)}")
print(f"Attributes: {[a for a in dir(creds) if not a.startswith('_')]}")

# Try common attribute names
for attr in ["api_key", "apiKey", "key", "api_secret", "secret", "api_passphrase", "passphrase"]:
    val = getattr(creds, attr, None)
    if val:
        print(f"  {attr} = {val}")
