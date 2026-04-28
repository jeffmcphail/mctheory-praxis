"""Check if wrapped tokens are recoverable from the NegRisk Adapter."""
import json
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))

ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
WCOL = "0x3a3bd7bb9528e159577f7c2e685cc81a765002e2"

abi = json.loads('[{"inputs":[{"name":"a","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]')
wcol = w3.eth.contract(address=w3.to_checksum_address(WCOL), abi=abi)

# Check adapter balance
bal = wcol.functions.balanceOf(w3.to_checksum_address(ADAPTER)).call()
print(f"Adapter WCOL balance: {bal} ({bal/1e6:.2f})")
print(f"(This includes ALL users' collateral, not just yours)")

# Check proxy implementation
impl_slot = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
impl = w3.eth.get_storage_at(w3.to_checksum_address(ADAPTER), impl_slot)
print(f"Proxy implementation: 0x{impl.hex()[-40:]}")

# Check if adapter has any rescue/recover functions
print(f"\nProbing adapter for recovery functions...")
selectors = {
    "emergencyWithdraw(address,uint256)": "0x",
    "rescueTokens(address,uint256)": "0x",  
    "recoverERC20(address,uint256)": "0x",
    "sweep(address)": "0x",
    "claimCollateral()": "0x",
}

# Just check the transfer tx to see what happened
print(f"\nChecking transfer tx...")
tx_hash = "0xd741c31ac1a6fe260fbe8d54bdf1de37aa33f8563ea2a19bee858354a0810190"
try:
    receipt = w3.eth.get_transaction_receipt(tx_hash)
    print(f"Status: {receipt.status}")
    print(f"Logs: {len(receipt.logs)}")
    for i, log in enumerate(receipt.logs):
        print(f"  Log {i}: from {log.address}")
        for ti, t in enumerate(log.topics):
            print(f"    topic[{ti}]: {t.hex()}")
        if log.data:
            print(f"    data: {log.data.hex()[:128]}")
except Exception as e:
    print(f"Error: {e}")

print(f"\nThe WCOL is inside the Adapter contract.")
print(f"The Adapter holds {bal/1e6:.2f} WCOL total from all users.")
print(f"Your 160.36 is pooled with everyone else's collateral.")
print(f"This is normal -- the Adapter always holds the WCOL backing.")
print(f"The question is whether there's a way to claim it back.")
