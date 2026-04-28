"""Find the token ID for a position by searching your portfolio."""
import os
import requests
from dotenv import load_dotenv
from web3 import Web3
load_dotenv()

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
w3 = Web3()
wallet = w3.eth.account.from_key(pk).address

DATA_API = "https://data-api.polymarket.com"

# Get all positions
r = requests.get(f"{DATA_API}/positions", params={"user": wallet}, timeout=10)
positions = r.json()

print(f"Found {len(positions)} positions\n")
for p in positions:
    title = p.get("title", p.get("market", {}).get("question", "?"))
    asset = p.get("asset", "")
    side = p.get("outcome", "?")
    size = float(p.get("size", 0))
    slug = p.get("market", {}).get("slug", "") if isinstance(p.get("market"), dict) else ""
    cid = p.get("market", {}).get("conditionId", "") if isinstance(p.get("market"), dict) else ""

    # Show all, highlight inflation
    if "3.4" in title:
        print(f"  *** {title} ***")
        print(f"    Side: {side} | Size: {size:.1f}")
        print(f"    FULL TOKEN: {asset}")
        # Write ready-to-run batch files
        with open("sell_dryrun.bat", "w") as f:
            f.write(f"python -m scripts.sell_position --token-id {asset} --side YES --price 0.31\n")
        with open("sell_execute.bat", "w") as f:
            f.write(f"python -m scripts.sell_position --token-id {asset} --side YES --price 0.31 --execute\n")
        print(f"\n    >>> Run:  .\\sell_dryrun.bat    (to preview)")
        print(f"    >>> Then: .\\sell_execute.bat   (to sell)")
    else:
        print(f"  {title[:65]}")
        print(f"    Side: {side} | Size: {size:.1f} | Token: {asset[:40]}...")
    if slug:
        print(f"    Slug: {slug}")
    if cid:
        print(f"    ConditionId: {cid[:40]}...")
    print()
