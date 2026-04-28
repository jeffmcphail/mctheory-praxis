/**
 * Deploy PraxisMEV contract to Polygon
 *
 * Usage:
 *   npx hardhat run scripts/deploy.js --network hardhat   # Test on fork
 *   npx hardhat run scripts/deploy.js --network polygon   # Deploy to mainnet
 */
const hre = require("hardhat");

async function main() {
  // Aave V3 PoolAddressesProvider on Polygon
  const AAVE_POOL_PROVIDER = "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb";

  console.log("═══════════════════════════════════════════════════");
  console.log("  Deploying PraxisMEV to", hre.network.name);
  console.log("═══════════════════════════════════════════════════");

  const [deployer] = await hre.ethers.getSigners();
  console.log("  Deployer:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("  Balance:", hre.ethers.formatEther(balance), "POL");

  // Deploy
  const PraxisMEV = await hre.ethers.getContractFactory("PraxisMEV");
  const contract = await PraxisMEV.deploy(AAVE_POOL_PROVIDER);
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log("\n  ✅ PraxisMEV deployed to:", address);

  // Verify state
  const owner = await contract.owner();
  const killed = await contract.killed();
  console.log("  Owner:", owner);
  console.log("  Kill switch:", killed);

  // Save deployment info
  const deployInfo = {
    network: hre.network.name,
    address: address,
    deployer: deployer.address,
    aavePoolProvider: AAVE_POOL_PROVIDER,
    timestamp: new Date().toISOString(),
    blockNumber: await hre.ethers.provider.getBlockNumber(),
  };

  const fs = require("fs");
  const deployPath = `./deployments/${hre.network.name}.json`;
  fs.mkdirSync("./deployments", { recursive: true });
  fs.writeFileSync(deployPath, JSON.stringify(deployInfo, null, 2));
  console.log("  Deployment info saved to:", deployPath);

  // Fund contract with a small amount of USDC.e for flash loan fees
  console.log("\n  NOTE: Send some USDC.e to the contract address to cover");
  console.log("  flash loan fees (0.05% of borrowed amount).");
  console.log("  For $10K flash loans, you need ~$5 USDC.e in the contract.");

  console.log("\n═══════════════════════════════════════════════════");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
