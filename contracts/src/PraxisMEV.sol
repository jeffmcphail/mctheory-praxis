// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "@aave/core-v3/contracts/interfaces/IPoolAddressesProvider.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC1155/IERC1155.sol";
import "@openzeppelin/contracts/token/ERC1155/utils/ERC1155Holder.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title PraxisMEV
 * @notice Flash loan-powered MEV execution on Polymarket (Polygon)
 * @dev Borrows USDC.e from Aave V3, executes arb strategies on Polymarket
 *      CTF contracts, repays loan + fee, keeps profit.
 *
 * Strategies:
 *   1. NegRisk Sum Arb: Buy all outcomes in underpriced multi-outcome market,
 *      redeem complete sets for $1 each.
 *   2. Merge Arb: Buy YES+NO in binary market where sum < $1, merge for $1.
 *   3. Leveraged Fade: Buy outcome tokens with flash-loaned capital,
 *      hold position (non-atomic, requires collateral).
 *
 * Safety:
 *   - Owner-only execution (no front-running by others)
 *   - Minimum profit check (reverts if unprofitable)
 *   - Emergency withdraw function
 *   - Kill switch
 */
contract PraxisMEV is FlashLoanSimpleReceiverBase, ERC1155Holder, Ownable, ReentrancyGuard {

    // ═══════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════

    // Polymarket contract addresses (Polygon mainnet)
    address public constant USDC_E = 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174;
    address public constant CTF = 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045;
    address public constant NEG_RISK_CTF = 0xC5d563A36AE78145C45a50134d48A1215220f80a;
    address public constant NEG_RISK_ADAPTER = 0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296;

    // Kill switch
    bool public killed = false;

    // Strategy being executed (set before flash loan, read in callback)
    enum Strategy { NONE, NEGRISK_SUM, MERGE_ARB, LEVERAGED_FADE }
    Strategy private _currentStrategy;

    // Params passed to flash loan callback
    struct NegRiskParams {
        bytes32 conditionId;
        uint256[] tokenIds;
        uint256[] amounts;
        uint256 minProfit;
    }

    struct MergeParams {
        uint256 yesTokenId;
        uint256 noTokenId;
        uint256 amount;
        uint256 minProfit;
    }

    struct FadeParams {
        uint256 tokenId;
        uint256 amount;
    }

    // Encoded params for current execution
    bytes private _currentParams;

    // Tracking
    uint256 public totalProfit;
    uint256 public totalTrades;

    // ═══════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════

    event TradeExecuted(
        Strategy strategy,
        uint256 borrowed,
        uint256 profit,
        uint256 timestamp
    );

    event KillSwitchToggled(bool active);
    event EmergencyWithdraw(address token, uint256 amount);

    // ═══════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════

    constructor(
        address _addressProvider
    )
        FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider))
        Ownable()
    {}

    // ═══════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════

    modifier notKilled() {
        require(!killed, "Kill switch active");
        _;
    }

    // ═══════════════════════════════════════════════════
    // STRATEGY 1: NEGRISK SUM ARBITRAGE
    // ═══════════════════════════════════════════════════

    /**
     * @notice Execute NegRisk sum arbitrage via flash loan
     * @dev Borrows USDC.e → buys all outcome tokens → redeems complete sets
     *      → repays loan → keeps profit
     * @param conditionId The CTF condition ID for the NegRisk event
     * @param tokenIds Array of outcome token IDs to buy
     * @param amounts Array of USDC amounts to spend on each outcome
     * @param totalBorrow Total USDC.e to borrow from Aave
     * @param minProfit Minimum profit in USDC.e (reverts if not met)
     */
    function executeNegRiskArb(
        bytes32 conditionId,
        uint256[] calldata tokenIds,
        uint256[] calldata amounts,
        uint256 totalBorrow,
        uint256 minProfit
    ) external onlyOwner notKilled nonReentrant {
        require(tokenIds.length == amounts.length, "Array length mismatch");
        require(tokenIds.length >= 2, "Need at least 2 outcomes");

        _currentStrategy = Strategy.NEGRISK_SUM;
        _currentParams = abi.encode(NegRiskParams({
            conditionId: conditionId,
            tokenIds: tokenIds,
            amounts: amounts,
            minProfit: minProfit
        }));

        // Request flash loan
        POOL.flashLoanSimple(
            address(this),    // receiver
            USDC_E,           // asset
            totalBorrow,      // amount
            "",               // params (we use storage instead)
            0                 // referral code
        );

        _currentStrategy = Strategy.NONE;
        _currentParams = "";
    }

    // ═══════════════════════════════════════════════════
    // STRATEGY 2: BINARY MERGE ARBITRAGE
    // ═══════════════════════════════════════════════════

    /**
     * @notice Execute binary merge arbitrage via flash loan
     * @dev Borrows USDC.e → buys YES+NO tokens → merges for $1 each
     *      → repays loan → keeps profit
     * @param yesTokenId YES outcome token ID
     * @param noTokenId NO outcome token ID
     * @param amount Number of complete sets to buy
     * @param totalBorrow Total USDC.e to borrow from Aave
     * @param minProfit Minimum profit in USDC.e
     */
    function executeMergeArb(
        uint256 yesTokenId,
        uint256 noTokenId,
        uint256 amount,
        uint256 totalBorrow,
        uint256 minProfit
    ) external onlyOwner notKilled nonReentrant {
        _currentStrategy = Strategy.MERGE_ARB;
        _currentParams = abi.encode(MergeParams({
            yesTokenId: yesTokenId,
            noTokenId: noTokenId,
            amount: amount,
            minProfit: minProfit
        }));

        POOL.flashLoanSimple(
            address(this),
            USDC_E,
            totalBorrow,
            "",
            0
        );

        _currentStrategy = Strategy.NONE;
        _currentParams = "";
    }

    // ═══════════════════════════════════════════════════
    // STRATEGY 3: LEVERAGED FADE (NON-ATOMIC)
    // ═══════════════════════════════════════════════════

    /**
     * @notice Buy outcome tokens with flash-loaned capital
     * @dev NOT ATOMIC -- tokens stay in contract. Owner must call
     *      exitPosition() later to sell and repay.
     *      Requires contract to hold collateral for the Aave position.
     *
     * NOTE: This strategy requires the contract to become an Aave borrower
     * (supply collateral, borrow against it). More complex than simple
     * flash loans. Deferred to Phase 3+.
     */
    // function executeLeveragedFade(...) -- TODO Phase 3+

    // ═══════════════════════════════════════════════════
    // FLASH LOAN CALLBACK
    // ═══════════════════════════════════════════════════

    /**
     * @notice Called by Aave after flash loan funds are received
     * @dev This is where the actual arbitrage logic runs
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata /* params */
    ) external override returns (bool) {
        require(msg.sender == address(POOL), "Caller must be Aave Pool");
        require(initiator == address(this), "Initiator must be this contract");

        uint256 balanceBefore = IERC20(USDC_E).balanceOf(address(this));

        if (_currentStrategy == Strategy.NEGRISK_SUM) {
            _executeNegRiskLogic();
        } else if (_currentStrategy == Strategy.MERGE_ARB) {
            _executeMergeLogic();
        } else {
            revert("Unknown strategy");
        }

        uint256 balanceAfter = IERC20(USDC_E).balanceOf(address(this));

        // Verify profit
        uint256 totalOwed = amount + premium;
        require(balanceAfter >= totalOwed, "Insufficient funds to repay");

        uint256 profit = balanceAfter - totalOwed;

        // Check minimum profit
        if (_currentStrategy == Strategy.NEGRISK_SUM) {
            NegRiskParams memory p = abi.decode(_currentParams, (NegRiskParams));
            require(profit >= p.minProfit, "Profit below minimum");
        } else if (_currentStrategy == Strategy.MERGE_ARB) {
            MergeParams memory p = abi.decode(_currentParams, (MergeParams));
            require(profit >= p.minProfit, "Profit below minimum");
        }

        // Approve Aave to pull repayment
        IERC20(USDC_E).approve(address(POOL), totalOwed);

        // Track
        totalProfit += profit;
        totalTrades += 1;

        emit TradeExecuted(_currentStrategy, amount, profit, block.timestamp);

        return true;
    }

    // ═══════════════════════════════════════════════════
    // INTERNAL STRATEGY LOGIC
    // ═══════════════════════════════════════════════════

    function _executeNegRiskLogic() internal {
        NegRiskParams memory p = abi.decode(_currentParams, (NegRiskParams));

        // Step 1: Approve USDC.e for the NegRisk CTF Exchange
        uint256 totalSpend = 0;
        for (uint i = 0; i < p.amounts.length; i++) {
            totalSpend += p.amounts[i];
        }
        IERC20(USDC_E).approve(NEG_RISK_CTF, totalSpend);

        // Step 2: Buy each outcome token
        // NOTE: This interacts with the NegRisk CTF Exchange contract
        // The exact function signature depends on the Polymarket CTF ABI
        // which we'll wire up in Phase 3 after ABI inspection
        //
        // For each outcome:
        //   INegRiskCTFExchange(NEG_RISK_CTF).buyOutcome(
        //       p.conditionId, p.tokenIds[i], p.amounts[i]
        //   );

        // Step 3: Redeem complete sets
        // Once we hold all N outcomes, we can redeem them for USDC.e
        //   ICTF(CTF).redeemPositions(
        //       USDC_E, p.conditionId, amounts
        //   );

        // PLACEHOLDER: The actual CTF interaction ABI needs to be
        // extracted from the deployed contracts. See flash_executor.py
        // for the ABI discovery logic.
        revert("NegRisk logic not yet wired -- need CTF Exchange ABI");
    }

    function _executeMergeLogic() internal {
        MergeParams memory p = abi.decode(_currentParams, (MergeParams));

        // Step 1: Buy YES tokens
        // Step 2: Buy NO tokens
        // Step 3: Merge YES+NO → USDC.e
        //   ICTF(CTF).mergePositions(
        //       USDC_E, conditionId, partition, amount
        //   );

        // PLACEHOLDER: Same as above -- need CTF ABI
        revert("Merge logic not yet wired -- need CTF Exchange ABI");
    }

    // ═══════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════

    /**
     * @notice Toggle kill switch
     */
    function setKillSwitch(bool _active) external onlyOwner {
        killed = _active;
        emit KillSwitchToggled(_active);
    }

    /**
     * @notice Emergency withdraw any token from the contract
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        if (token == address(0)) {
            // Withdraw native token (POL/MATIC)
            (bool success, ) = owner().call{value: amount}("");
            require(success, "ETH transfer failed");
        } else {
            IERC20(token).transfer(owner(), amount);
        }
        emit EmergencyWithdraw(token, amount);
    }

    /**
     * @notice Emergency withdraw ERC-1155 tokens (CTF outcome tokens)
     */
    function emergencyWithdraw1155(
        address token,
        uint256 tokenId,
        uint256 amount
    ) external onlyOwner {
        IERC1155(token).safeTransferFrom(address(this), owner(), tokenId, amount, "");
    }

    /**
     * @notice Withdraw accumulated profits
     */
    function withdrawProfit() external onlyOwner {
        uint256 balance = IERC20(USDC_E).balanceOf(address(this));
        require(balance > 0, "No balance");
        IERC20(USDC_E).transfer(owner(), balance);
    }

    // Allow contract to receive native tokens
    receive() external payable {}

    // Required for ERC1155 support (inherited from ERC1155Holder)
    function supportsInterface(bytes4 interfaceId)
        public view virtual override(ERC1155Receiver)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
