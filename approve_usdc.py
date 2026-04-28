"""Approve USDC for Polymarket exchange using web3 directly."""
import os
from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

# Polygon RPC (public)
RPC_URL = "https://polygon-bor-rpc.publicnode.com"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
address = os.getenv("POLYMARKET_API_ADDRESS", "")

if not pk.startswith("0x"):
    pk = "0x" + pk

account = w3.eth.account.from_key(pk)
print(f"Wallet: {account.address}")
print(f"Connected: {w3.is_connected()}")

# USDC on Polygon (native)
USDC_NATIVE = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
# USDC.e (bridged) 
USDC_BRIDGED = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# Polymarket Exchange contracts
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# ERC20 ABI (just balanceOf and approve)
ERC20_ABI = [
    {"constant":True,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
    {"constant":False,"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},
    {"constant":True,"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"},
]

# Check both USDC variants
for name, addr in [("USDC (native)", USDC_NATIVE), ("USDC.e (bridged)", USDC_BRIDGED)]:
    token = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)
    bal = token.functions.balanceOf(account.address).call()
    decimals = 6  # USDC has 6 decimals
    print(f"\n{name}: ${bal / 10**decimals:.2f} ({bal} raw)")
    
    if bal > 0:
        print(f"  Found USDC! Checking allowances...")
        
        for ex_name, ex_addr in [
            ("CTF Exchange", CTF_EXCHANGE),
            ("Neg Risk Exchange", NEG_RISK_EXCHANGE),
            ("Neg Risk Adapter", NEG_RISK_ADAPTER),
        ]:
            allowance = token.functions.allowance(
                account.address,
                Web3.to_checksum_address(ex_addr)
            ).call()
            print(f"  Allowance for {ex_name}: {allowance / 10**decimals:.2f}")
            
            if allowance < bal:
                print(f"  → Approving {ex_name}...")
                max_uint = 2**256 - 1
                tx = token.functions.approve(
                    Web3.to_checksum_address(ex_addr),
                    max_uint
                ).build_transaction({
                    "from": account.address,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "gas": 100000,
                    "gasPrice": w3.eth.gas_price,
                    "chainId": 137,
                })
                signed = account.sign_transaction(tx)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                print(f"  → TX: {tx_hash.hex()}")
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                print(f"  → Status: {'✅ Success' if receipt['status'] == 1 else '❌ Failed'}")
                # Update nonce for next tx
                
# Check POL balance for gas
pol_bal = w3.eth.get_balance(account.address)
print(f"\nPOL balance (for gas): {w3.from_wei(pol_bal, 'ether'):.4f} POL")
if pol_bal < w3.to_wei(0.01, 'ether'):
    print("⚠️ Low POL — may need gas for approval transactions")

print("\nDone! Try trading again:")
print("  python first_trade_v2.py trade --amount 5")
