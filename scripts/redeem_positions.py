"""
scripts/redeem_positions.py — Redeem resolved Polymarket positions for USDC.

VALIDATION (maximal by default — 6 pre-flight, 2 post-flight checks):
  PRE-1:  Expected payout > $0.001
  PRE-2:  Re-read payoutDenominator right before execution (must be > 0)
  PRE-3:  Re-read our payoutNumerator right before execution (must be > 0)
  PRE-4:  Re-read on-chain token balance (must be > 0)
  PRE-5:  Record USDC.e balance before
  PRE-6:  Recompute expected payout with rechecked values

  POST-1: Verify USDC.e balance increased by expected amount
  POST-2: Verify token balance is 0 (burned)

Usage:
    python -m scripts.redeem_positions                    # Scan all positions (dry run)
    python -m scripts.redeem_positions --verbose          # Detailed payout info
    python -m scripts.redeem_positions --execute          # Redeem with full validation
"""
import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
load_dotenv()

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEGRISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"

CTF_ABI = json.loads("""[
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"},{"name":"index","type":"uint256"}],"name":"payoutNumerators","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],"name":"payoutDenominator","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")

NEGRISK_ABI = json.loads("""[
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"}
]""")

USDC_ABI = json.loads("""[
    {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from web3 import Web3

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("No POLYMARKET_PRIVATE_KEY"); sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    if not w3.is_connected():
        print("Cannot connect to RPC"); sys.exit(1)

    acct = w3.eth.account.from_key(pk)
    wallet = acct.address
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)
    neg_adapter = w3.eth.contract(address=Web3.to_checksum_address(NEGRISK_ADAPTER), abi=NEGRISK_ABI)
    usdc_c = w3.eth.contract(address=Web3.to_checksum_address(USDC_E_ADDRESS), abi=USDC_ABI)

    usdc_bal = usdc_c.functions.balanceOf(wallet).call()
    print(f"{'='*90}")
    print(f"  POLYMARKET POSITION REDEEMER")
    print(f"  Wallet:  {wallet}")
    print(f"  USDC.e:  ${usdc_bal/1e6:.6f} (raw: {usdc_bal})")
    print(f"{'='*90}")

    print(f"\n  Fetching positions...")
    r = requests.get("https://data-api.polymarket.com/positions",
                     params={"user": wallet.lower()}, timeout=15)
    positions = r.json()
    print(f"  Found {len(positions)} position(s)\n")

    redeemable = []
    worthless = []
    active = []

    print(f"  {'#':<4s} {'Market':<50s} {'Side':<5s} {'Shares':>8s} {'OnChain':>10s} "
          f"{'Denom':>6s} {'Payout':>8s} {'Status':<14s}")
    print(f"  {'─'*115}")

    for i, pos in enumerate(positions, 1):
        asset_id = pos.get("asset", "")
        cond_id = pos.get("conditionId", pos.get("condition_id", ""))
        outcome = pos.get("outcome", "")
        size = float(pos.get("size", 0))
        avg_price = float(pos.get("avgPrice", 0))
        title = pos.get("title", pos.get("market", ""))[:49]
        cur_price = float(pos.get("curPrice", pos.get("price", 0)))

        # Detect NegRisk markets by checking Gamma API
        market_data = pos.get("market", {}) if isinstance(pos.get("market"), dict) else {}
        is_negrisk = (pos.get("negRisk", False) or
                      market_data.get("negRisk", False) or
                      pos.get("neg_risk", False))

        # If not detected from position data, check Gamma API
        if not is_negrisk:
            slug = pos.get("slug", pos.get("market_slug", ""))
            if not slug and isinstance(pos.get("market"), dict):
                slug = pos["market"].get("slug", "")

            if args.verbose:
                print(f"       NegRisk check: slug='{slug}' condId='{cond_id[:16]}...' "
                      f"keys={list(pos.keys())[:8]}")

            # Try by conditionId (most reliable for NegRisk sub-markets)
            if cond_id:
                try:
                    gr = requests.get(f"https://gamma-api.polymarket.com/markets",
                                      params={"conditionId": cond_id}, timeout=5)
                    gdata = gr.json()
                    if gdata and isinstance(gdata, list) and len(gdata) > 0:
                        is_negrisk = gdata[0].get("negRisk", False)
                        if args.verbose:
                            print(f"       Gamma lookup by conditionId: negRisk={is_negrisk}")
                    elif args.verbose:
                        print(f"       Gamma lookup by conditionId: no results")
                except Exception as e:
                    if args.verbose:
                        print(f"       Gamma lookup failed: {e}")

            # Fallback: try by slug
            if not is_negrisk and slug:
                try:
                    gr = requests.get(f"https://gamma-api.polymarket.com/markets",
                                      params={"slug": slug}, timeout=5)
                    gdata = gr.json()
                    if gdata and isinstance(gdata, list) and len(gdata) > 0:
                        is_negrisk = gdata[0].get("negRisk", False)
                        if args.verbose:
                            print(f"       Gamma lookup by slug: negRisk={is_negrisk}")
                except Exception:
                    pass

        if size <= 0:
            continue

        if args.verbose and is_negrisk:
            print(f"       ** NegRisk market detected -- will use adapter for redemption **")

        # On-chain balance
        oc_raw = 0
        try:
            oc_raw = ctf.functions.balanceOf(wallet, int(asset_id)).call()
        except Exception:
            pass
        oc_shares = oc_raw / 1e6

        # Payout data
        p_yes = p_no = denom = 0
        cid_bytes = None
        if cond_id:
            cid_hex = cond_id[2:] if cond_id.startswith("0x") else cond_id
            cid_bytes = bytes.fromhex(cid_hex)
            try:
                p_yes = ctf.functions.payoutNumerators(cid_bytes, 0).call()
                p_no = ctf.functions.payoutNumerators(cid_bytes, 1).call()
            except Exception:
                pass
            try:
                denom = ctf.functions.payoutDenominator(cid_bytes).call()
            except Exception:
                pass

        resolved = denom > 0
        our_num = p_yes if outcome == "Yes" else p_no
        our_idx = 0 if outcome == "Yes" else 1
        expected = (our_num / denom * oc_shares) if (resolved and denom > 0 and oc_raw > 0) else 0

        # Categorize
        if resolved and expected > 0.001:
            status = "REDEEMABLE"
            redeemable.append({
                "title": title, "asset_id": asset_id, "cond_id": cond_id,
                "cid_bytes": cid_bytes, "outcome": outcome, "our_idx": our_idx,
                "size": size, "oc_shares": oc_shares, "oc_raw": oc_raw,
                "expected": expected, "p_yes": p_yes, "p_no": p_no,
                "our_num": our_num, "denom": denom,
                "avg_price": avg_price, "cost": size * avg_price,
                "is_negrisk": is_negrisk,
            })
        elif resolved and our_num == 0:
            status = "LOST"
            worthless.append({"title": title, "cost": size * avg_price})
        elif not resolved and (p_yes > 0 or p_no > 0):
            status = "SETTLING"
            active.append({"title": title})
        else:
            status = "ACTIVE"
            active.append({"title": title})

        icons = {"REDEEMABLE": "✅", "LOST": "💀", "SETTLING": "⏳", "ACTIVE": ""}
        st_str = f"{icons.get(status,'')} {status}"
        oc_str = f"{oc_shares:.1f}" if oc_shares > 0 else "—"
        exp_str = f"${expected:.2f}" if expected > 0.001 else "—"

        print(f"  {i:<4d} {title:<50s} {outcome:<5s} {size:>7.1f} {oc_str:>10s} "
              f"{denom:>6d} {exp_str:>8s} {st_str:<14s}")

        if args.verbose and cid_bytes:
            print(f"       Numerators: YES={p_yes} NO={p_no}  Denom={denom}")
            print(f"       Our ratio:  {our_num}/{denom} = {our_num/denom:.6f}" if denom > 0 else "       NOT RESOLVED")
            print(f"       On-chain:   {oc_raw}")

    # Summary
    print(f"\n{'─'*90}")
    print(f"  Active: {len(active)}  |  Lost: {len(worthless)} (${sum(w['cost'] for w in worthless):.2f})  "
          f"|  Redeemable: {len(redeemable)}")

    if redeemable:
        total_exp = sum(r["expected"] for r in redeemable)
        total_cost = sum(r["cost"] for r in redeemable)
        for r in redeemable:
            print(f"\n    ✅ {r['title']}")
            print(f"       {r['oc_shares']:.1f} {r['outcome']} × {r['our_num']}/{r['denom']} "
                  f"= ${r['expected']:.6f}")
            print(f"       Cost: ${r['cost']:.2f}  Profit: ${r['expected']-r['cost']:.2f}")
        print(f"\n    Total redeemable: ${total_exp:.2f}")
    else:
        print(f"\n  Nothing to redeem. Markets may not be settled on-chain yet.")
        print(f"  (payoutDenominator = 0 means oracle hasn't called reportPayouts)")
        return

    if not args.execute:
        print(f"\n  Dry run. Add --execute to redeem.")
        return

    # ═════════════ EXECUTE WITH MAXIMAL VALIDATION ═════════════
    print(f"\n  {'═'*60}")
    print(f"  EXECUTING REDEMPTIONS — 6 pre-flight + 2 post-flight checks")
    print(f"  {'═'*60}")

    for r in redeemable:
        cb = r["cid_bytes"]
        print(f"\n  ── {r['title']} ──")

        # PRE-1: Expected payout
        print(f"  PRE-1 Expected payout:    ${r['expected']:.6f}", end="")
        if r["expected"] < 0.001:
            print(f" ❌ ABORT: too low"); continue
        print(f" ✓")

        # PRE-2: Recheck payoutDenominator
        try:
            d2 = ctf.functions.payoutDenominator(cb).call()
        except Exception as e:
            print(f"  PRE-2 PayoutDenominator:  ❌ ABORT: {e}"); continue
        print(f"  PRE-2 PayoutDenominator:  {d2}", end="")
        if d2 == 0:
            print(f" ❌ ABORT: not resolved on-chain"); continue
        print(f" ✓")

        # PRE-3: Recheck our numerator
        try:
            n2 = ctf.functions.payoutNumerators(cb, r["our_idx"]).call()
        except Exception as e:
            print(f"  PRE-3 Our numerator:      ❌ ABORT: {e}"); continue
        print(f"  PRE-3 Our numerator:      {n2}", end="")
        if n2 == 0:
            print(f" ❌ ABORT: we LOST"); continue
        print(f" ✓")

        # PRE-4: Recheck token balance
        try:
            b2 = ctf.functions.balanceOf(wallet, int(r["asset_id"])).call()
        except Exception as e:
            print(f"  PRE-4 Token balance:      ❌ ABORT: {e}"); continue
        print(f"  PRE-4 Token balance:      {b2} ({b2/1e6:.6f})", end="")
        if b2 == 0:
            print(f" ❌ ABORT: no tokens"); continue
        print(f" ✓")

        # PRE-5: USDC.e before
        try:
            u_before = usdc_c.functions.balanceOf(wallet).call()
        except Exception as e:
            print(f"  PRE-5 USDC.e before:      ❌ ABORT: {e}"); continue
        print(f"  PRE-5 USDC.e before:      ${u_before/1e6:.6f} ✓")

        # PRE-6: Recompute
        recomp = (n2 / d2) * (b2 / 1e6)
        print(f"  PRE-6 Recomputed payout:  ${recomp:.6f}", end="")
        if recomp < 0.001:
            print(f" ❌ ABORT: too low"); continue
        print(f" ✓")

        print(f"  ═══ ALL 6 PRE-FLIGHT CHECKS PASSED ═══")

        # EXECUTE
        try:
            from web3 import Web3 as W3

            if r.get("is_negrisk"):
                # NegRisk: redeem through adapter contract
                print(f"  NegRisk market detected -- using NegRisk Adapter")
                print(f"  Adapter: {NEGRISK_ADAPTER}")
                tx = neg_adapter.functions.redeemPositions(
                    W3.to_checksum_address(USDC_E_ADDRESS), bytes(32), cb, [1, 2]
                ).build_transaction({
                    "from": wallet,
                    "nonce": w3.eth.get_transaction_count(wallet),
                    "gas": 500000, "gasPrice": w3.eth.gas_price, "chainId": 137,
                })
            else:
                # Regular: redeem through CTF directly
                print(f"  Regular market -- using CTF direct redemption")
                tx = ctf.functions.redeemPositions(
                    W3.to_checksum_address(USDC_E_ADDRESS), bytes(32), cb, [1, 2]
                ).build_transaction({
                    "from": wallet,
                    "nonce": w3.eth.get_transaction_count(wallet),
                    "gas": 300000, "gasPrice": w3.eth.gas_price, "chainId": 137,
                })
            signed = w3.eth.account.sign_transaction(tx, pk)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hex = tx_hash.hex()
            print(f"  📤 Tx: {tx_hex}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            gas_pol = receipt.gasUsed * receipt.effectiveGasPrice / 1e18

            if receipt.status != 1:
                print(f"  ❌ REVERTED! Gas: {receipt.gasUsed}")
                print(f"  https://polygonscan.com/tx/{tx_hex}")
                continue

            print(f"  ✓ Confirmed. Gas: {receipt.gasUsed} ({gas_pol:.6f} POL)")
        except Exception as e:
            print(f"  ❌ Tx error: {e}"); continue

        time.sleep(1)

        # POST-1: USDC.e after
        try:
            u_after = usdc_c.functions.balanceOf(wallet).call()
            delta = (u_after - u_before) / 1e6
            print(f"  POST-1 USDC.e after:     ${u_after/1e6:.6f}")
            print(f"  POST-1 Delta:            ${delta:.6f}", end="")
            if delta < 0.001:
                print(f" ⚠ WARNING: No USDC received!")
                print(f"         Expected ${recomp:.6f} — got ${delta:.6f}")
                print(f"         https://polygonscan.com/tx/{tx_hex}")
            elif abs(delta - recomp) > 0.01:
                print(f" ⚠ Mismatch: expected ${recomp:.6f}")
            else:
                print(f" ✅ VERIFIED")
        except Exception as e:
            print(f"  POST-1 ⚠ Cannot verify: {e}")

        # POST-2: Token balance
        try:
            b_after = ctf.functions.balanceOf(wallet, int(r["asset_id"])).call()
            print(f"  POST-2 Tokens remaining: {b_after}", end="")
            if b_after > 0:
                print(f" ⚠ NOT fully burned")
            else:
                print(f" ✅ Burned")
        except Exception as e:
            print(f"  POST-2 ⚠ Cannot verify: {e}")

    # Final
    time.sleep(1)
    final = usdc_c.functions.balanceOf(wallet).call() / 1e6
    print(f"\n{'─'*90}")
    print(f"  FINAL USDC.e: ${final:.6f}  (was ${usdc_bal/1e6:.6f}, delta ${final-usdc_bal/1e6:.6f})")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
