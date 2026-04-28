"""
batch_longshots.py — Execute curated longshot bias trades.

Reads signals from data/longshot_signals.json, curates the best
favorites and longshot fades, and places trades on Polymarket.

Usage:
    python batch_longshots.py                          # Dry run
    python batch_longshots.py --execute                # Live
    python batch_longshots.py --fav-budget 320 --ls-budget 80
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import requests

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

# Categories to AVOID for favorites (too unpredictable)
RISKY_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "crypto", "price of", "token", "fdv", "market cap",
    "strike", "war", "invade", "military", "nuclear",
]


def is_risky(question):
    """Check if a market question involves risky/unpredictable topics."""
    q = question.lower()
    return any(kw in q for kw in RISKY_KEYWORDS)


def get_client():
    from py_clob_client.client import ClobClient
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=0)
    client.set_api_creds(client.derive_api_key())
    return client


def find_token_id_for_side(slug, question, side="YES"):
    """Find token ID for a specific side of a market."""
    r = requests.get("https://gamma-api.polymarket.com/events",
                     params={"slug": slug})
    events = r.json()
    if not events:
        return None, None

    event = events[0]
    for m in event.get("markets", []):
        q = m.get("question", m.get("groupItemTitle", ""))
        if q[:40] == question[:40]:  # Fuzzy match
            token_ids = m.get("clobTokenIds", "")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            prices = json.loads(m.get("outcomePrices", "[0,0]"))
            
            if side == "YES":
                return token_ids[0] if token_ids else None, float(prices[0])
            else:  # NO
                return token_ids[1] if len(token_ids) > 1 else None, float(prices[1]) if len(prices) > 1 else 1 - float(prices[0])
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Longshot Bias Batch Trader")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fav-budget", type=float, default=320.0,
                        help="Budget for favorites (default: $320)")
    parser.add_argument("--ls-budget", type=float, default=80.0,
                        help="Budget for longshot fades (default: $80)")
    parser.add_argument("--max-per-trade", type=float, default=50.0,
                        help="Max per single trade")
    args = parser.parse_args()

    if not os.path.exists("data/longshot_signals.json"):
        print("❌ Run scan_longshots.py first")
        return

    with open("data/longshot_signals.json") as f:
        signals = json.load(f)

    # === CURATE FAVORITES ===
    favorites = [s for s in signals
                 if s.get("type") == "FAVORITE_BACK"
                 and s.get("yes_price", 0) >= 0.93
                 and s.get("volume", 0) >= 2000
                 and not is_risky(s.get("question", ""))]
    
    # Sort by volume (highest first — most liquid)
    favorites.sort(key=lambda s: s.get("volume", 0), reverse=True)
    
    # Select favorites within budget
    fav_selected = []
    fav_total = 0
    for s in favorites:
        spend = min(args.max_per_trade, s.get("yes_price", 0.95) * 50)  # ~50 contracts
        if fav_total + spend > args.fav_budget:
            continue
        s["spend"] = round(spend, 2)
        s["num_contracts"] = round(spend / s.get("yes_price", 0.95), 1)
        s["side"] = "YES"
        fav_selected.append(s)
        fav_total += spend

    # === CURATE LONGSHOT FADES ===
    longshots = [s for s in signals
                 if s.get("type") == "LONGSHOT_FADE"
                 and s.get("yes_price", 1) <= 0.10
                 and s.get("volume", 0) >= 5000
                 and not is_risky(s.get("question", ""))]
    
    # Sort by YES price ascending (cheapest longshots = biggest bias)
    longshots.sort(key=lambda s: (s.get("yes_price", 1), -s.get("volume", 0)))
    
    # Select longshots within budget — BUY NO on these
    ls_selected = []
    ls_total = 0
    for s in longshots:
        no_price = 1.0 - s.get("yes_price", 0.10)
        spend = min(args.max_per_trade, no_price * 20)  # ~20 contracts
        if ls_total + spend > args.ls_budget:
            continue
        s["spend"] = round(spend, 2)
        s["no_price"] = no_price
        s["num_contracts"] = round(spend / no_price, 1)
        s["side"] = "NO"
        ls_selected.append(s)
        ls_total += spend

    total_spend = fav_total + ls_total
    
    # === DISPLAY PLAN ===
    print(f"\n{'='*85}")
    print(f"LONGSHOT BIAS TRADE PLAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Mode: {'🔴 LIVE' if args.execute else '📋 DRY RUN'}")
    print(f"  Favorites budget: ${args.fav_budget:.0f}  |  Longshots budget: ${args.ls_budget:.0f}")
    print(f"{'='*85}")

    print(f"\n  FAVORITE BACKS (BUY YES) — {len(fav_selected)} trades, ${fav_total:.2f}")
    print(f"  {'-'*80}")
    
    for i, s in enumerate(fav_selected, 1):
        q = s.get("question", "?")[:55]
        print(f"  {i:<3d} {q:<55s} YES={s['yes_price']:.0%} ${s['spend']:>6.1f} "
              f"~{s['num_contracts']:.0f}sh  vol=${s['volume']:>10,.0f}")

    fav_profit = sum(s["num_contracts"] * (1 - s["yes_price"]) * 0.98
                     for s in fav_selected)
    print(f"  Expected profit if all hit: ${fav_profit:.2f}")

    print(f"\n  LONGSHOT FADES (BUY NO) — {len(ls_selected)} trades, ${ls_total:.2f}")
    print(f"  {'-'*80}")
    
    for i, s in enumerate(ls_selected, 1):
        q = s.get("question", "?")[:55]
        print(f"  {i:<3d} {q:<55s} YES={s['yes_price']:.0%} NO={s['no_price']:.0%} "
              f"${s['spend']:>5.1f} ~{s['num_contracts']:.0f}sh  "
              f"vol=${s['volume']:>9,.0f}")

    ls_profit = sum(s["num_contracts"] * s["yes_price"] * 0.98
                    for s in ls_selected)
    print(f"  Expected profit if all hit: ${ls_profit:.2f}")

    print(f"\n  {'─'*80}")
    print(f"  TOTAL: {len(fav_selected) + len(ls_selected)} trades, "
          f"${total_spend:.2f} spend")
    print(f"  Expected profit: ${fav_profit + ls_profit:.2f} "
          f"({(fav_profit + ls_profit) / total_spend * 100:.1f}%)")

    if not args.execute:
        print(f"\n  Add --execute to place trades.")
        return

    confirm = input(f"\n  Type 'YES' to place {len(fav_selected) + len(ls_selected)} "
                    f"trades for ${total_spend:.2f}: ")
    if confirm != "YES":
        print("  Cancelled.")
        return

    # === EXECUTE ===
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    client = get_client()
    results = []
    succeeded = 0
    failed = 0
    all_trades = fav_selected + ls_selected

    for i, t in enumerate(all_trades, 1):
        side_label = t["side"]
        q = t.get("question", "?")[:50]
        print(f"\n  [{i}/{len(all_trades)}] {side_label} — {q}...")

        # Find token ID
        token_id, current_price = find_token_id_for_side(
            t["slug"], t["question"], side_label
        )
        
        if not token_id:
            print(f"    ❌ Token not found")
            failed += 1
            continue

        price = current_price if current_price > 0.01 else (
            t["yes_price"] if side_label == "YES" else t["no_price"]
        )
        size = round(t["spend"] / price, 2)

        print(f"    {side_label} @ ${price:.3f}  Size: {size:.1f}  Spend: ${t['spend']:.2f}")

        try:
            order_args = OrderArgs(
                price=round(price, 2),
                size=size,
                side=BUY,
                token_id=token_id,
            )
            signed_order = client.create_order(order_args)
            result = client.post_order(signed_order)

            status = result.get("status", "?")
            oid = result.get("orderID", "?")[:16]
            print(f"    ✅ {status} (ID: {oid}...)")

            results.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "question": t.get("question", ""),
                "side": side_label,
                "type": t.get("type", ""),
                "price": price,
                "spend": t["spend"],
                "size": size,
                "order_id": result.get("orderID", ""),
                "status": status,
            })
            succeeded += 1
            time.sleep(0.5)

        except Exception as e:
            err = str(e)[:120]
            print(f"    ❌ {err}")
            failed += 1
            results.append({
                "question": t.get("question", ""),
                "side": side_label,
                "error": err,
                "status": "FAILED",
            })
            time.sleep(1)

    print(f"\n{'='*85}")
    print(f"  RESULTS: {succeeded} succeeded, {failed} failed")
    print(f"{'='*85}")

    # Save
    log_path = "data/longshot_trades.json"
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    existing.extend(results)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"  Logged: {log_path}")


if __name__ == "__main__":
    main()
