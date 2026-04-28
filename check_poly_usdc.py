"""Check which USDC token Polymarket uses and our balances."""
import os
from dotenv import load_dotenv
load_dotenv()
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))
pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
if not pk.startswith("0x"):
    pk = "0x" + pk
account = w3.eth.account.from_key(pk)

ERC20_ABI = [
    {"constant":True,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
    {"constant":True,"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"},
]

# Both USDC tokens on Polygon
USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
USDC_BRIDGED = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e

# All Polymarket exchange contracts
EXCHANGES = {
    "CTF Exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "Neg Risk Exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "Neg Risk Adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
}

print(f"Wallet: {account.address}\n")

for name, addr in [("USDC native", USDC_NATIVE), ("USDC.e bridged", USDC_BRIDGED)]:
    token = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)
    bal = token.functions.balanceOf(account.address).call()
    print(f"{name} ({addr[:10]}...): ${bal/1e6:.2f}")
    
    for ex_name, ex_addr in EXCHANGES.items():
        allow = token.functions.allowance(
            account.address, Web3.to_checksum_address(ex_addr)
        ).call()
        status = "✅ approved" if allow > 1e12 else f"❌ {allow/1e6:.2f}"
        print(f"  → {ex_name}: {status}")
    print()

# Check the CLOB API for which token it expects
print("Checking CLOB API collateral info...")
import requests
r = requests.get("https://clob.polymarket.com/collateral")
if r.ok:
    print(f"  Collateral: {r.json()}")
else:
    print(f"  {r.status_code}: {r.text[:200]}")

# Also check neg-risk collateral
r2 = requests.get("https://clob.polymarket.com/neg-risk/collateral") 
if r2.ok:
    print(f"  Neg-risk collateral: {r2.json()}")
else:
    print(f"  {r2.status_code}: {r2.text[:200]}")
