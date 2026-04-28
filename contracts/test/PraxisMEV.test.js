/**
 * PraxisMEV Contract Tests
 *
 * Runs against a forked Polygon mainnet to test with real Aave state.
 *
 * Usage:
 *   npx hardhat test                    # Run all tests on fork
 *   npx hardhat test --grep "deploy"    # Run specific test
 */
const { expect } = require("chai");
const { ethers } = require("hardhat");

// Polygon mainnet addresses
const ADDRESSES = {
  AAVE_POOL_PROVIDER: "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb",
  AAVE_POOL: "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
  USDC_E: "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
  CTF: "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
  NEG_RISK_CTF: "0xC5d563A36AE78145C45a50134d48A1215220f80a",
  // A known USDC.e holder on Polygon (for impersonation in tests)
  USDC_WHALE: "0xe7804c37c13166fF0b37F5aE0BB07A3aEbb6e245",
};

describe("PraxisMEV", function () {
  let contract;
  let owner;
  let usdc;

  beforeEach(async function () {
    [owner] = await ethers.getSigners();

    // Deploy PraxisMEV
    const PraxisMEV = await ethers.getContractFactory("PraxisMEV");
    contract = await PraxisMEV.deploy(ADDRESSES.AAVE_POOL_PROVIDER);
    await contract.waitForDeployment();

    // Get USDC.e contract reference
    usdc = await ethers.getContractAt(
      "@openzeppelin/contracts/token/ERC20/IERC20.sol:IERC20",
      ADDRESSES.USDC_E
    );
  });

  describe("Deployment", function () {
    it("Should deploy with correct owner", async function () {
      expect(await contract.owner()).to.equal(owner.address);
    });

    it("Should start with kill switch off", async function () {
      expect(await contract.killed()).to.equal(false);
    });

    it("Should have zero profit tracked", async function () {
      expect(await contract.totalProfit()).to.equal(0);
      expect(await contract.totalTrades()).to.equal(0);
    });
  });

  describe("Kill Switch", function () {
    it("Should allow owner to toggle kill switch", async function () {
      await contract.setKillSwitch(true);
      expect(await contract.killed()).to.equal(true);

      await contract.setKillSwitch(false);
      expect(await contract.killed()).to.equal(false);
    });

    it("Should prevent non-owner from toggling", async function () {
      const [, other] = await ethers.getSigners();
      await expect(
        contract.connect(other).setKillSwitch(true)
      ).to.be.revertedWith("Ownable: caller is not the owner");
    });
  });

  describe("Emergency Withdraw", function () {
    it("Should allow owner to withdraw USDC.e", async function () {
      // Fund the contract (impersonate a USDC whale)
      const whale = await ethers.getImpersonatedSigner(ADDRESSES.USDC_WHALE);

      // Give whale some POL for gas
      await owner.sendTransaction({
        to: ADDRESSES.USDC_WHALE,
        value: ethers.parseEther("1"),
      });

      const contractAddr = await contract.getAddress();
      const fundAmount = 1000000; // 1 USDC.e (6 decimals)

      // Check whale has USDC
      const whaleBalance = await usdc.balanceOf(ADDRESSES.USDC_WHALE);
      if (whaleBalance < fundAmount) {
        console.log("  Whale has insufficient USDC.e, skipping");
        this.skip();
      }

      // Fund contract
      await usdc.connect(whale).transfer(contractAddr, fundAmount);

      const contractBalance = await usdc.balanceOf(contractAddr);
      expect(contractBalance).to.equal(fundAmount);

      // Withdraw
      await contract.emergencyWithdraw(ADDRESSES.USDC_E, fundAmount);
      const afterBalance = await usdc.balanceOf(contractAddr);
      expect(afterBalance).to.equal(0);
    });
  });

  describe("Flash Loan Basic", function () {
    it("Should revert NegRisk arb (placeholder logic)", async function () {
      // Fund contract with USDC for flash loan fee
      const whale = await ethers.getImpersonatedSigner(ADDRESSES.USDC_WHALE);
      await owner.sendTransaction({
        to: ADDRESSES.USDC_WHALE,
        value: ethers.parseEther("1"),
      });

      const contractAddr = await contract.getAddress();
      const whaleBalance = await usdc.balanceOf(ADDRESSES.USDC_WHALE);
      if (whaleBalance < 10000000) {
        console.log("  Whale has insufficient USDC.e, skipping");
        this.skip();
      }

      await usdc.connect(whale).transfer(contractAddr, 10000000); // $10

      // Try to execute NegRisk arb — should revert with placeholder message
      const dummyCondition = ethers.encodeBytes32String("test");
      await expect(
        contract.executeNegRiskArb(
          dummyCondition,
          [1, 2],        // tokenIds
          [500000, 500000], // amounts
          1000000,       // borrow $1
          100            // min profit
        )
      ).to.be.reverted; // Will revert inside executeOperation
    });

    it("Should block execution when kill switch is active", async function () {
      await contract.setKillSwitch(true);

      const dummyCondition = ethers.encodeBytes32String("test");
      await expect(
        contract.executeNegRiskArb(
          dummyCondition,
          [1, 2],
          [500000, 500000],
          1000000,
          100
        )
      ).to.be.revertedWith("Kill switch active");
    });
  });
});
