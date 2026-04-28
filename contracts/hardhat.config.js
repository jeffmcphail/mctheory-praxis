require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config({ path: "../.env" }); // Read from Praxis root .env

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.10",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    // Polygon mainnet
    polygon: {
      url: process.env.POLYGON_RPC_URL || "https://polygon-bor-rpc.publicnode.com",
      accounts: process.env.POLYMARKET_PRIVATE_KEY
        ? [process.env.POLYMARKET_PRIVATE_KEY]
        : [],
      chainId: 137,
      gasPrice: 50000000000, // 50 gwei (Polygon is cheap)
    },
    // Polygon fork for testing (simulates mainnet state locally)
    // Polygon fork for testing (needs archival RPC)
    // Free options: Alchemy, Infura, or QuickNode free tier
    // Set POLYGON_RPC_URL in .env to your archival RPC
    // e.g. POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
    hardhat: {
      forking: {
        url: process.env.POLYGON_RPC_URL || "https://polygon-mainnet.g.alchemy.com/v2/demo",
        blockNumber: 70000000,
        enabled: !!process.env.POLYGON_RPC_URL,  // Only fork if archival RPC is set
      },
      chainId: 137,
    },
  },
  paths: {
    sources: "./src",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
  },
};
