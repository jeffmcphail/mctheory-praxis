"""
engines/flash_executor.py — Flash Loan Execution Bridge

Connects the Python flash scanner to the PraxisMEV Solidity contract.
When the scanner detects a profitable opportunity, this module:
  1. Validates the opportunity is still live
  2. Calculates exact parameters for the smart contract call
  3. Submits the transaction via web3.py
  4. Monitors the result and logs P&L

Also includes ABI discovery tools for the Polymarket CTF contracts,
which are needed to wire up the smart contract's internal logic.

Usage:
    python -m engines.flash_executor discover-abi          # Discover CTF ABIs
    python -m engines.flash_executor simulate               # Simulate on fork
    python -m engines.flash_executor execute --opp-id 123   # Execute a logged opportunity
    python -m engines.flash_executor status                  # Contract state
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

try:
    from web3 import Web3
    from web3.middleware import ExtraDataToPOAMiddleware
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

RPC_URL = os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")
PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")

# Contract addresses (Polygon mainnet)
ADDRESSES = {
    "USDC_E": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "CTF": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
    "NEG_RISK_CTF": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "NEG_RISK_ADAPTER": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
    "AAVE_POOL_PROVIDER": "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb",
    "AAVE_POOL": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
    "POLYMARKET_EXCHANGE": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "NEG_RISK_EXCHANGE": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
}

# Deployment info (set after deploying PraxisMEV.sol)
DEPLOYMENT_PATH = Path("contracts/deployments/polygon.json")
FLASH_DB = Path("data/flash_scanner.db")

# ═══════════════════════════════════════════════════════
# WEB3 SETUP
# ═══════════════════════════════════════════════════════

def get_web3():
    """Initialize Web3 connection to Polygon."""
    if not HAS_WEB3:
        print("  ❌ pip install web3 --break-system-packages")
        return None

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    # Polygon is a POA chain — need this middleware
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    if not w3.is_connected():
        print(f"  ❌ Cannot connect to {RPC_URL}")
        return None

    return w3


def get_contract(w3):
    """Load the deployed PraxisMEV contract."""
    if not DEPLOYMENT_PATH.exists():
        print(f"  ❌ No deployment found at {DEPLOYMENT_PATH}")
        print(f"  Deploy first: cd contracts && npx hardhat run scripts/deploy.js --network polygon")
        return None

    with open(DEPLOYMENT_PATH) as f:
        deployment = json.load(f)

    address = deployment.get("address")
    if not address:
        print(f"  ❌ No address in deployment file")
        return None

    # Load ABI from Hardhat artifacts
    abi_path = Path("contracts/artifacts/src/PraxisMEV.sol/PraxisMEV.json")
    if not abi_path.exists():
        print(f"  ❌ No ABI found. Run: cd contracts && npx hardhat compile")
        return None

    with open(abi_path) as f:
        artifact = json.load(f)

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(address),
        abi=artifact["abi"],
    )

    return contract


# ═══════════════════════════════════════════════════════
# ABI DISCOVERY
# ═══════════════════════════════════════════════════════

def cmd_discover_abi(args):
    """Discover and document the CTF contract ABIs.

    These are needed to wire up the PraxisMEV contract's internal logic
    for buying/redeeming outcome tokens.
    """
    w3 = get_web3()
    if not w3:
        return

    print(f"\n{'='*90}")
    print(f"  CTF CONTRACT ABI DISCOVERY")
    print(f"  Inspecting Polymarket contracts on Polygon")
    print(f"{'='*90}")

    # Known function signatures for CTF contracts
    # These are from the Gnosis Conditional Token Framework
    KNOWN_CTF_FUNCTIONS = {
        # ConditionalTokens (CTF) — ERC-1155 outcome tokens
        "splitPosition": "0x72ce4275",       # Split collateral into outcome tokens
        "mergePositions": "0x65c4ce7a",      # Merge outcome tokens back to collateral
        "redeemPositions": "0x01864fcf",     # Redeem winning positions after resolution
        "getCollectionId": "0x2e7ba6ef",     # Get collection ID for a condition
        "getPositionId": "0x39d21d3b",       # Get position ID (token ID)
        "getOutcomeSlotCount": "0xd42dc0c2", # Number of outcomes in a condition
        "payoutDenominator": "0xdd34de67",   # Payout denominator for a condition
        "balanceOf": "0x00fdd58e",           # ERC-1155 balance check
    }

    # Check which functions exist on the CTF contract
    print(f"\n  CTF Contract ({ADDRESSES['CTF']}):")
    print(f"  {'Function':<25s} {'Selector':<12s} {'Exists'}")
    print(f"  {'─'*50}")

    for name, selector in KNOWN_CTF_FUNCTIONS.items():
        # Check if the contract has this function by calling with empty args
        # (will revert but tells us the function exists)
        try:
            result = w3.eth.call({
                "to": Web3.to_checksum_address(ADDRESSES["CTF"]),
                "data": selector,
            })
            print(f"  {name:<25s} {selector:<12s} ✅")
        except Exception as e:
            err = str(e)
            if "revert" in err.lower() or "invalid" in err.lower():
                # Function exists but reverted (expected with no args)
                print(f"  {name:<25s} {selector:<12s} ✅ (reverts)")
            elif "execution" in err.lower():
                print(f"  {name:<25s} {selector:<12s} ✅ (exec error)")
            else:
                print(f"  {name:<25s} {selector:<12s} ❌ ({err[:30]})")

    # Polymarket Exchange contract (handles order matching)
    print(f"\n  Polymarket Exchange ({ADDRESSES['POLYMARKET_EXCHANGE']}):")
    EXCHANGE_FUNCTIONS = {
        "fillOrder": "0x64a3bc15",
        "matchOrders": "0x88ec79fb",
        "cancelOrder": "0xfa00e6c1",
    }

    for name, selector in EXCHANGE_FUNCTIONS.items():
        try:
            w3.eth.call({
                "to": Web3.to_checksum_address(ADDRESSES["POLYMARKET_EXCHANGE"]),
                "data": selector,
            })
            print(f"  {name:<25s} {selector:<12s} ✅")
        except Exception as e:
            err = str(e)
            if "revert" in err or "execution" in err:
                print(f"  {name:<25s} {selector:<12s} ✅ (exists)")
            else:
                print(f"  {name:<25s} {selector:<12s} ❌")

    # NegRisk Adapter
    print(f"\n  NegRisk Adapter ({ADDRESSES['NEG_RISK_ADAPTER']}):")
    ADAPTER_FUNCTIONS = {
        "splitPosition": "0x72ce4275",
        "mergePositions": "0x65c4ce7a",
        "redeemPositions": "0x01864fcf",
        "getConditionId": "0xb741d5e2",
    }

    for name, selector in ADAPTER_FUNCTIONS.items():
        try:
            w3.eth.call({
                "to": Web3.to_checksum_address(ADDRESSES["NEG_RISK_ADAPTER"]),
                "data": selector,
            })
            print(f"  {name:<25s} {selector:<12s} ✅")
        except Exception as e:
            err = str(e)
            if "revert" in err or "execution" in err:
                print(f"  {name:<25s} {selector:<12s} ✅ (exists)")
            else:
                print(f"  {name:<25s} {selector:<12s} ❌")

    # Fetch verified ABI from Polygonscan if available
    print(f"\n  Fetching verified ABIs from Polygonscan...")
    polygonscan_key = os.getenv("POLYGONSCAN_API_KEY", "")

    for name, addr in [("CTF", ADDRESSES["CTF"]),
                       ("NEG_RISK_ADAPTER", ADDRESSES["NEG_RISK_ADAPTER"])]:
        if polygonscan_key:
            try:
                r = requests.get("https://api.polygonscan.com/api", params={
                    "module": "contract",
                    "action": "getabi",
                    "address": addr,
                    "apikey": polygonscan_key,
                }, timeout=10)

                data = r.json()
                if data.get("status") == "1":
                    abi = json.loads(data["result"])
                    abi_path = Path(f"contracts/abis/{name}.json")
                    abi_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(abi_path, "w") as f:
                        json.dump(abi, f, indent=2)
                    funcs = [f["name"] for f in abi if f.get("type") == "function"]
                    print(f"    ✅ {name}: {len(funcs)} functions → {abi_path}")
                    for fn in funcs[:10]:
                        print(f"       - {fn}")
                    if len(funcs) > 10:
                        print(f"       ... and {len(funcs) - 10} more")
                else:
                    print(f"    ⚠️ {name}: {data.get('message', 'not verified')}")
            except Exception as e:
                print(f"    ❌ {name}: {e}")
        else:
            print(f"    ⚠️ Set POLYGONSCAN_API_KEY in .env for auto-ABI fetch")
            break

    # Summary
    print(f"\n{'─'*90}")
    print(f"  SUMMARY")
    print(f"{'─'*90}")
    print(f"  The key contract interaction chain for NegRisk arb:")
    print(f"    1. USDC.e.approve(NegRiskAdapter, amount)")
    print(f"    2. NegRiskAdapter.splitPosition(conditionId, amount)")
    print(f"       → Gives us all N outcome tokens")
    print(f"    3. After holding tokens: NegRiskAdapter.mergePositions()")
    print(f"       → Burns outcome tokens, returns USDC.e")
    print(f"")
    print(f"  For flash loan arb (sum < K):")
    print(f"    1. Flash loan USDC.e from Aave")
    print(f"    2. Buy each underpriced outcome on CLOB (off-chain!)")
    print(f"    3. Once holding all N outcomes → mergePositions on-chain")
    print(f"    4. Repay flash loan from merged USDC.e")
    print(f"")
    print(f"  ⚠️ CRITICAL: Step 2 (CLOB buy) is OFF-CHAIN and cannot be")
    print(f"  executed atomically within a flash loan transaction.")
    print(f"  This means pure flash loan arb on Polymarket requires")
    print(f"  on-chain liquidity (DEX) or pre-positioned tokens.")
    print(f"")
    print(f"  Alternative: splitPosition atomically creates all outcomes")
    print(f"  from USDC.e. If we can sell overpriced outcomes on-chain,")
    print(f"  that IS atomic and flash-loan-compatible.")
    print(f"\n{'='*90}")


# ═══════════════════════════════════════════════════════
# CONTRACT INTERACTION
# ═══════════════════════════════════════════════════════

def cmd_status(args):
    """Check deployed contract status."""
    w3 = get_web3()
    if not w3:
        return

    contract = get_contract(w3)
    if not contract:
        return

    address = contract.address

    print(f"\n{'='*70}")
    print(f"  PRAXIS MEV CONTRACT STATUS")
    print(f"{'='*70}")
    print(f"  Address:      {address}")
    print(f"  Network:      Polygon (chain {w3.eth.chain_id})")
    print(f"  Block:        {w3.eth.block_number}")

    # Contract state
    owner = contract.functions.owner().call()
    killed = contract.functions.killed().call()
    total_profit = contract.functions.totalProfit().call()
    total_trades = contract.functions.totalTrades().call()

    print(f"  Owner:        {owner}")
    print(f"  Kill switch:  {'🔴 ACTIVE' if killed else '🟢 OFF'}")
    print(f"  Total profit: {total_profit / 1e6:.2f} USDC.e")
    print(f"  Total trades: {total_trades}")

    # Balances
    usdc_abi = json.loads('[{"inputs":[{"name":"account","type":"address"}],'
                          '"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],'
                          '"stateMutability":"view","type":"function"}]')
    usdc = w3.eth.contract(
        address=Web3.to_checksum_address(ADDRESSES["USDC_E"]),
        abi=usdc_abi,
    )
    usdc_balance = usdc.functions.balanceOf(address).call()
    pol_balance = w3.eth.get_balance(address)

    print(f"\n  Balances:")
    print(f"    USDC.e:  ${usdc_balance / 1e6:.2f}")
    print(f"    POL:     {w3.from_wei(pol_balance, 'ether'):.4f}")

    # Wallet balances (deployer)
    wallet = w3.eth.account.from_key(PRIVATE_KEY).address if PRIVATE_KEY else "?"
    if PRIVATE_KEY:
        wallet_usdc = usdc.functions.balanceOf(wallet).call()
        wallet_pol = w3.eth.get_balance(wallet)
        print(f"\n  Wallet ({wallet[:10]}...):")
        print(f"    USDC.e:  ${wallet_usdc / 1e6:.2f}")
        print(f"    POL:     {w3.from_wei(wallet_pol, 'ether'):.4f}")

    print(f"\n{'='*70}")


def cmd_simulate(args):
    """Simulate a flash loan execution on a forked network.

    This uses Hardhat's fork mode to simulate without real money.
    """
    print(f"\n{'='*70}")
    print(f"  FLASH LOAN SIMULATION")
    print(f"{'='*70}")
    print(f"  To simulate, run the Hardhat tests on a Polygon fork:")
    print(f"")
    print(f"  cd contracts")
    print(f"  npm install")
    print(f"  npx hardhat test")
    print(f"")
    print(f"  This forks Polygon mainnet locally and runs the test suite,")
    print(f"  including flash loan execution with real Aave pool state.")
    print(f"  No real money is used.")
    print(f"\n{'='*70}")


def cmd_execute(args):
    """Execute a flash loan for a specific logged opportunity."""
    opp_id = getattr(args, "opp_id", None)

    if not opp_id:
        print("  ❌ Provide --opp-id from flash_scanner.db")
        return

    w3 = get_web3()
    if not w3:
        return

    contract = get_contract(w3)
    if not contract:
        return

    # Load opportunity from DB
    if not FLASH_DB.exists():
        print(f"  ❌ No flash scanner DB. Run: python -m engines.flash_scanner scan")
        return

    conn = sqlite3.connect(str(FLASH_DB))
    conn.row_factory = sqlite3.Row
    opp = conn.execute(
        "SELECT * FROM flash_opportunities WHERE id=?", (opp_id,)).fetchone()

    if not opp:
        print(f"  ❌ Opportunity {opp_id} not found")
        conn.close()
        return

    if not opp["executable"]:
        print(f"  ⚠️ Opportunity {opp_id} was not flagged as executable")
        print(f"  Reason: {opp['reason']}")
        conn.close()
        return

    print(f"\n{'='*70}")
    print(f"  EXECUTING FLASH LOAN — Opportunity #{opp_id}")
    print(f"{'='*70}")
    print(f"  Type:      {opp['opp_type']}")
    print(f"  Event:     {opp['event_title']}")
    print(f"  Deviation: {opp['deviation_pct']:.2f}%")
    print(f"  Net profit: ${opp['net_profit_usd_per_1k']:.2f} per $1K")
    print(f"\n  ⚠️ LIVE EXECUTION — Real money will be used!")
    print(f"  Type 'yes' to confirm: ", end="")

    if input().strip().lower() != "yes":
        print("  Aborted.")
        conn.close()
        return

    # TODO: Build transaction based on opportunity type
    # This requires the CTF ABI to be wired in to the smart contract
    print(f"\n  ❌ Execution not yet wired — need CTF ABI integration.")
    print(f"  Run: python -m engines.flash_executor discover-abi")
    print(f"  Then update PraxisMEV.sol with actual CTF calls.")

    conn.close()
    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Flash Loan Execution Bridge")
    subs = parser.add_subparsers(dest="command")

    subs.add_parser("discover-abi", help="Discover CTF contract ABIs")
    subs.add_parser("simulate", help="Simulate on fork")
    subs.add_parser("status", help="Contract status")

    p_exec = subs.add_parser("execute", help="Execute flash loan")
    p_exec.add_argument("--opp-id", type=int, required=True)

    args = parser.parse_args()

    if args.command == "discover-abi":
        cmd_discover_abi(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "execute":
        cmd_execute(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
