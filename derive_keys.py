"""Derive CLOB API credentials from private key (no server call needed)."""
import os
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
if not pk:
    print("❌ POLYMARKET_PRIVATE_KEY not in .env")
    exit(1)

print("Deriving CLOB API credentials...")
client = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137)

try:
    creds = client.derive_api_key()
    print(f"\n✅ Credentials derived!")
    print(f"\nAdd to .env:")
    print(f"POLYMARKET_CLOB_API_KEY={creds.get('apiKey', creds.get('api_key', ''))}")
    print(f"POLYMARKET_CLOB_API_SECRET={creds.get('secret', '')}")
    print(f"POLYMARKET_CLOB_API_PASSPHRASE={creds.get('passphrase', '')}")
except Exception as e:
    print(f"derive_api_key failed: {e}")
    print("\nTrying create_or_derive...")
    try:
        creds = client.create_or_derive_api_creds()
        print(f"\n✅ Credentials created/derived!")
        print(f"\nAdd to .env:")
        print(f"POLYMARKET_CLOB_API_KEY={creds.get('apiKey', creds.get('api_key', ''))}")
        print(f"POLYMARKET_CLOB_API_SECRET={creds.get('secret', '')}")  
        print(f"POLYMARKET_CLOB_API_PASSPHRASE={creds.get('passphrase', '')}")
    except Exception as e2:
        print(f"create_or_derive also failed: {e2}")
        import traceback
        traceback.print_exc()
