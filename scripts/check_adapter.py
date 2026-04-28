from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://polygon-bor-rpc.publicnode.com'))
ADAPTER = '0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296'
code = w3.eth.get_code(Web3.to_checksum_address(ADAPTER)).hex()

funcs = [
    'redeemPositions(address,bytes32,bytes32,uint256[])',
    'redeemPositions(bytes32,uint256[])',
    'redeemPositions(bytes32,bytes32,uint256[])',
    'redeemPositions(uint256,bytes32,uint256[])',
    'redeem(bytes32,uint256[])',
    'splitPosition(address,bytes32,bytes32,uint256[],uint256)',
    'mergePositions(address,bytes32,bytes32,uint256[],uint256)',
    'convertPositions(bytes32,bytes32,uint256,uint256)',
    'getConditionId(bytes32)',
    'getDetermined(bytes32)',
    'getMarketData(bytes32)',
    'reportOutcome(bytes32,bool)',
    'prepareMarket(uint256,bytes)',
    'prepareQuestion(bytes32,bytes)',
]
for f in funcs:
    sel = Web3.keccak(text=f)[:4].hex()
    found = sel in code
    tag = "FOUND" if found else "     "
    print(f'{tag} {sel} {f}')
