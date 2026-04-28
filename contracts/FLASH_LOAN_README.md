# Flash Loan MEV System — Setup & Architecture

## Quick Start

### Phase 1: Scanner (Python only — ready now)
```powershell
# Full scan for flash-loan-profitable opportunities
python -m engines.flash_scanner scan

# Continuous monitoring (every 5 min)
python -m engines.flash_scanner monitor --interval 300

# View historical results
python -m engines.flash_scanner stats
```

### Phase 2: Smart Contract (requires Node.js + Hardhat)
```powershell
# One-time setup
cd contracts
npm install

# Compile the contract
npx hardhat compile

# Test on Polygon fork (no real money)
npx hardhat test

# Deploy to Polygon mainnet
npx hardhat run scripts/deploy.js --network polygon
```

### Phase 3: Python↔Contract Bridge
```powershell
# Discover CTF contract ABIs (needed to wire up strategies)
python -m engines.flash_executor discover-abi

# Check deployed contract status
python -m engines.flash_executor status

# Simulate execution
python -m engines.flash_executor simulate
```

### Phase 4: DEX Exploration (research phase)
```powershell
python -m engines.flash_scanner dex-check
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1: flash_scanner.py (Python)                          │
│  Continuously scans for opportunities:                       │
│  • Binary merge arb (YES+NO < $1)                            │
│  • NegRisk sum arb (sum < K after fees)                      │
│  • DEX liquidity check                                       │
│  Calculates exact profitability after ALL costs               │
│  Logs to data/flash_scanner.db                                │
└──────────────┬───────────────────────────────────────────────┘
               │ When executable opportunity found
               ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 3: flash_executor.py (Python → web3.py)               │
│  Validates opportunity is still live                          │
│  Builds transaction parameters                                │
│  Calls PraxisMEV contract via web3.py                         │
│  Monitors result, logs P&L                                    │
└──────────────┬───────────────────────────────────────────────┘
               │ contract.executeNegRiskArb() or .executeMergeArb()
               ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 2: PraxisMEV.sol (Solidity on Polygon)                │
│  1. Borrows USDC.e from Aave V3 flash loan                   │
│  2. Executes arb strategy (buy outcomes, redeem)              │
│  3. Repays loan + 0.05% fee                                  │
│  4. Sends profit to owner                                     │
│  5. If unprofitable → entire tx reverts (lose nothing)        │
└──────────────────────────────────────────────────────────────┘
```

---

## Cost Model

| Fee                    | Amount     | Notes                              |
|------------------------|------------|-------------------------------------|
| Aave flash loan        | 0.05%      | Of borrowed amount                  |
| Polymarket winner fee  | 2.00%      | On winning side at settlement       |
| Polygon gas            | ~$0.01-0.05| Per transaction                     |
| **Breakeven deviation**| **~2.05%** | Sum must deviate >2.05% from K      |

For a $10,000 flash loan on a NegRisk market with 3% sum deviation:
- Gross profit: $10,000 × 0.03 / sum = ~$300
- Aave fee: $10,000 × 0.0005 = $5
- Poly fee: $10,000 × 0.02 = $200
- Gas: $0.03
- **Net profit: ~$95 per trade**

---

## .env additions needed

```env
# Add to your existing .env file:
POLYGON_RPC_URL=https://polygon-bor-rpc.publicnode.com
POLYGONSCAN_API_KEY=your_key_here  # Optional, for ABI auto-fetch
```

---

## Phase 4: DEX Exploration & ERC-1155 Wrapper

### The Problem
CTF outcome tokens are ERC-1155 (multi-token standard), not ERC-20.
Standard DEXes (QuickSwap, Uniswap) only support ERC-20 tokens.
This means CTF tokens can't be traded on DEXes directly.

### The Solution: ERC-1155 → ERC-20 Wrapper
A wrapper contract that:
1. Accepts ERC-1155 CTF tokens (deposit)
2. Mints corresponding ERC-20 tokens 1:1
3. ERC-20 tokens can be traded on DEXes
4. Redeem ERC-20 → unwrap → get ERC-1155 back

This is the same pattern used by:
- Wrapped CryptoPunks (WPUNKS)
- Wrapped ERC-721 → ERC-20 for DeFi

### Why This Matters
If CTF tokens trade on a DEX via wrapper:
- **DEX↔CLOB arbitrage becomes possible**
- Flash loan → buy cheap on DEX → sell expensive on CLOB
- Or: flash loan → buy cheap on CLOB (off-chain) → sell on DEX
- The DEX provides on-chain liquidity for atomic execution

### Implementation Sketch

```solidity
// CTFWrapper.sol — Wraps ERC-1155 CTF tokens as ERC-20
contract CTFWrapper is ERC20 {
    IERC1155 public ctf;
    uint256 public tokenId;

    constructor(address _ctf, uint256 _tokenId, string memory name, string memory symbol)
        ERC20(name, symbol)
    {
        ctf = IERC1155(_ctf);
        tokenId = _tokenId;
    }

    // Deposit ERC-1155 → mint ERC-20
    function wrap(uint256 amount) external {
        ctf.safeTransferFrom(msg.sender, address(this), tokenId, amount, "");
        _mint(msg.sender, amount);
    }

    // Burn ERC-20 → withdraw ERC-1155
    function unwrap(uint256 amount) external {
        _burn(msg.sender, amount);
        ctf.safeTransferFrom(address(this), msg.sender, tokenId, amount, "");
    }
}
```

### Phase 4 TODO
1. Check if any wrapper contracts already exist on Polygon for CTF tokens
2. If not, deploy CTFWrapper for high-volume markets
3. Seed DEX liquidity (provide wrapped tokens + USDC.e on QuickSwap)
4. Build the DEX↔CLOB arbitrage scanner
5. Wire into PraxisMEV.sol for atomic flash loan execution

### Existing Wrapper Research
- Gnosis has an official "Wrapped1155Factory" contract
- Check if Polymarket or community has deployed wrappers
- Search Polygonscan for contracts interacting with CTF + DEX routers

---

## Critical Design Insight

**The CLOB is off-chain. Flash loans are on-chain.**

This creates a fundamental tension:
- Flash loans require atomic execution (everything in one tx)
- CLOB orders are matched off-chain by the Polymarket operator
- You CAN'T buy tokens on the CLOB from within a flash loan

**What CAN be done atomically:**
1. `splitPosition()` — mint all outcome tokens from USDC.e (on-chain)
2. `mergePositions()` — burn outcome tokens for USDC.e (on-chain)
3. DEX trades — if wrapped CTF tokens exist on QuickSwap

**The atomic arb path:**
1. Flash loan USDC.e
2. `splitPosition()` → get all N outcome tokens
3. Sell OVERPRICED outcomes on DEX (if liquidity exists)
4. `mergePositions()` with remaining tokens
5. Repay flash loan
6. Keep profit

This is the reverse of what we originally designed — instead of buying
underpriced outcomes, we MINT all outcomes and SELL the overpriced ones.
The profit comes from selling at above-fair-value on the DEX while
the redemption value is fixed at $1/set.

---

## File Inventory

```
contracts/
├── src/
│   └── PraxisMEV.sol              # Flash loan smart contract
├── scripts/
│   └── deploy.js                   # Deployment script
├── test/
│   └── PraxisMEV.test.js          # Fork-based tests
├── hardhat.config.js               # Hardhat configuration
├── package.json                    # Node.js dependencies
└── FLASH_LOAN_README.md            # This file

engines/
├── flash_scanner.py                # Phase 1: Opportunity scanner
├── flash_executor.py               # Phase 3: Python↔Contract bridge
├── mev_scanner.py                  # Original MEV scanner (Phase 1 MEV)
└── mev_executor.py                 # Spike fade executor (Phase 2 MEV)
```
