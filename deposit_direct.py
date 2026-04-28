"""Transfer USDC into Polymarket proxy wallet directly via web3."""
import os
from dotenv import load_dotenv
load_dotenv()
from web3 import Web3

RPC_URL = "https://polygon-bor-rpc.publicnode.com"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
if not pk.startswith("0x"):
    pk = "0x" + pk

account = w3.eth.account.from_key(pk)
print(f"Wallet: {account.address}")

# USDC on Polygon
USDC = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

# Polymarket CTF Exchange — this is where deposits go
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# But actually, the proxy wallet address is derived from your account
# Let's check what Polymarket sees as your proxy
# The proxy factory is at a known address

# First, let's check if Polymarket has a getProxyWallet function
# or if we can find the proxy address from the API

import requests

# Check the CLOB API for our proxy address
from py_clob_client.client import ClobClient
client = ClobClient("https://clob.polymarket.com", key=pk.replace("0x",""), chain_id=137)
creds = client.derive_api_key()
client.set_api_creds(creds)

# The proxy wallet address might be available
print(f"\nLooking for proxy wallet...")

# Try getting the proxy from the Polymarket API
try:
    # Check if there's a proxy wallet registered
    r = requests.get(
        "https://clob.polymarket.com/get-address",
        headers={"Authorization": f"Bearer {creds.api_key}"}
    )
    print(f"  get-address: {r.status_code} {r.text[:200]}")
except Exception as e:
    print(f"  get-address failed: {e}")

# The Polymarket proxy factory contract
PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"

PROXY_FACTORY_ABI = [
    {"inputs":[{"name":"_addr","type":"address"}],"name":"getProxy","outputs":[{"name":"","type":"address"}],"stateMutability":"view","type":"function"},
]

try:
    factory = w3.eth.contract(
        address=Web3.to_checksum_address(PROXY_FACTORY),
        abi=PROXY_FACTORY_ABI
    )
    proxy_addr = factory.functions.getProxy(account.address).call()
    print(f"  Proxy wallet: {proxy_addr}")
    
    # Check USDC balance in proxy
    ERC20_ABI = [
        {"constant":True,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
        {"constant":False,"inputs":[{"name":"to","type":"address"},{"name":"amount","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"type":"function"},
    ]
    
    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC), abi=ERC20_ABI)
    
    eoa_bal = usdc.functions.balanceOf(account.address).call()
    proxy_bal = usdc.functions.balanceOf(Web3.to_checksum_address(proxy_addr)).call()
    
    print(f"\n  EOA balance:   ${eoa_bal / 1e6:.2f}")
    print(f"  Proxy balance: ${proxy_bal / 1e6:.2f}")
    
    if proxy_bal > 0:
        print(f"\n  ✅ Proxy wallet already has USDC! Try trading again.")
    elif eoa_bal > 0:
        # Transfer USDC from EOA to proxy wallet
        amount = eoa_bal - 1_000_000  # Leave $1 buffer
        print(f"\n  Transferring ${amount/1e6:.2f} USDC to proxy wallet...")
        
        confirm = input(f"  Type 'YES' to transfer: ")
        if confirm != "YES":
            print("  Cancelled.")
            exit()
        
        tx = usdc.functions.transfer(
            Web3.to_checksum_address(proxy_addr),
            amount
        ).build_transaction({
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 100000,
            "gasPrice": w3.eth.gas_price,
            "chainId": 137,
        })
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  TX: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        print(f"  Status: {'✅ Success' if receipt['status'] == 1 else '❌ Failed'}")
        
        # Verify
        new_proxy_bal = usdc.functions.balanceOf(Web3.to_checksum_address(proxy_addr)).call()
        print(f"\n  Proxy balance now: ${new_proxy_bal / 1e6:.2f}")
        print(f"\n  Now try: python first_trade_v3.py")
    
except Exception as e:
    print(f"  Proxy lookup failed: {e}")
    import traceback
    traceback.print_exc()
