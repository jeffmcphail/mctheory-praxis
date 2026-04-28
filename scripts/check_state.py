"""Check all balances right now."""
import json
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))

WALLET = "0x55D36A902C7D2080230ce39F5921cC90D42c5EFa"
WCOL = "0x3a3bd7bb9528e159577f7c2e685cc81a765002e2"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

abi = json.loads('[{"inputs":[{"name":"a","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]')

usdc = w3.eth.contract(address=w3.to_checksum_address(USDC_E), abi=abi)
wcol = w3.eth.contract(address=w3.to_checksum_address(WCOL), abi=abi)

print("Current state:")
print(f"  Your USDC.e:     ${usdc.functions.balanceOf(w3.to_checksum_address(WALLET)).call()/1e6:.6f}")
print(f"  Your WCOL:       {wcol.functions.balanceOf(w3.to_checksum_address(WALLET)).call()/1e6:.6f}")
print(f"  Adapter WCOL:    {wcol.functions.balanceOf(w3.to_checksum_address(ADAPTER)).call()/1e6:.6f}")
print(f"  Adapter USDC.e:  ${usdc.functions.balanceOf(w3.to_checksum_address(ADAPTER)).call()/1e6:.6f}")

try:
    ts = wcol.functions.totalSupply().call()
    print(f"  WCOL totalSupply: {ts/1e6:.6f}")
except:
    print(f"  WCOL totalSupply: (not available)")

# Check your POL balance too
pol = w3.eth.get_balance(w3.to_checksum_address(WALLET))
print(f"  Your POL:        {pol/1e18:.4f}")

# Check native USDC (not bridged) in case
USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
try:
    usdc_native = w3.eth.contract(address=w3.to_checksum_address(USDC_NATIVE), abi=abi)
    print(f"  Your USDC (native): ${usdc_native.functions.balanceOf(w3.to_checksum_address(WALLET)).call()/1e6:.6f}")
except:
    pass
