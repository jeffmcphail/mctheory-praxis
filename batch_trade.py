"""
batch_trade.py — Place multiple weather trades from scanner signals.

Reads signals from data/weather_signals.json, filters to high-conviction
BUY YES opportunities, and places orders on Polymarket.

Usage:
    python batch_trade.py                    # Show plan, don't trade
    python batch_trade.py --execute          # Place all trades
    python batch_trade.py --min-edge 0.20    # Only edges >= 20%
    python batch_trade.py --max-spend 200    # Cap total spend
"""
import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

import requests

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


def get_client():
    from py_clob_client.client import ClobClient
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=0)
    client.set_api_creds(client.derive_api_key())
    return client


def find_token_id(slug, bucket_title):
    """Fetch market event and find the YES token ID for a specific bucket."""
    r = requests.get("https://gamma-api.polymarket.com/events", params={"slug": slug})
    events = r.json()
    if not events:
        return None, None

    event = events[0]
    for m in event.get("markets", []):
        title = m.get("groupItemTitle", m.get("question", ""))
        if title == bucket_title:
            token_ids = m.get("clobTokenIds", "")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            prices = json.loads(m.get("outcomePrices", "[0,0]"))
            current_price = float(prices[0])
            return token_ids[0] if token_ids else None, current_price

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Batch weather trades")
    parser.add_argument("--execute", action="store_true", help="Actually place trades")
    parser.add_argument("--min-edge", type=float, default=0.20, help="Min edge (default: 20%%)")
    parser.add_argument("--max-spend", type=float, default=400.0, help="Max total spend")
    parser.add_argument("--max-per-trade", type=float, default=75.0, help="Max per trade")
    parser.add_argument("--direction", default="BUY YES", help="Filter direction")
    args = parser.parse_args()

    # Load signals
    if not os.path.exists("data/weather_signals.json"):
        print("❌ Run scan_all_weather.py first")
        return

    with open("data/weather_signals.json") as f:
        all_signals = json.load(f)

    # Filter to high-conviction BUY YES signals
    trades = [s for s in all_signals
              if s["direction"] == args.direction
              and s["edge"] >= args.min_edge
              and s["position"] >= 5.0]  # Skip tiny positions

    # Sort by edge descending
    trades.sort(key=lambda s: s["edge"], reverse=True)

    # Cap total spend
    selected = []
    total_spend = 0
    for t in trades:
        spend = min(t["position"], args.max_per_trade)
        if total_spend + spend > args.max_spend:
            continue
        t["spend"] = spend
        selected.append(t)
        total_spend += spend

    print(f"\n{'='*80}")
    print(f"BATCH TRADE PLAN — {len(selected)} trades, ${total_spend:,.2f} total")
    print(f"  Min edge: {args.min_edge:.0%}")
    print(f"  Max spend: ${args.max_spend:,.0f}")
    print(f"  Mode: {'🔴 LIVE EXECUTION' if args.execute else '📋 DRY RUN (add --execute to trade)'}")
    print(f"{'='*80}\n")

    print(f"  {'#':<3s} {'City':<15s} {'Date':<12s} {'Bucket':<18s} "
          f"{'Mkt':>5s} {'Model':>6s} {'Edge':>6s} {'Spend':>7s}")
    print(f"  {'-'*75}")

    for i, t in enumerate(selected, 1):
        print(f"  {i:<3d} {t['city']:<15s} {t['date']:<12s} {t['bucket']:<18s} "
              f"{t['market_prob']:>4.0%} {t['model_prob']:>5.0%} "
              f"{t['edge']:>+5.0%} ${t['spend']:>6.1f}")

    total_ev = sum(t["edge"] * t["spend"] for t in selected)
    print(f"\n  Total spend: ${total_spend:,.2f}")
    print(f"  Expected edge value: ${total_ev:+,.2f}")
    print(f"  Max possible payout: ${sum(t['spend']/t['market_prob'] for t in selected):,.0f}")

    if not args.execute:
        print(f"\n  Add --execute to place these trades.")
        return

    # Confirm
    confirm = input(f"\n  Type 'YES' to place {len(selected)} trades for ${total_spend:.2f}: ")
    if confirm != "YES":
        print("  Cancelled.")
        return

    # Execute trades
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    client = get_client()
    results = []
    succeeded = 0
    failed = 0

    for i, t in enumerate(selected, 1):
        print(f"\n  [{i}/{len(selected)}] {t['city']} {t['date']} {t['bucket']}...")

        # Find token ID
        token_id, current_price = find_token_id(t["slug"], t["bucket"])
        if not token_id:
            print(f"    ❌ Token ID not found for {t['slug']}")
            failed += 1
            continue

        # Use current live price (may have changed since scan)
        price = current_price if current_price > 0.01 else t["market_prob"]
        
        # Recalculate size based on current price
        size = round(t["spend"] / price, 2)

        print(f"    Price: ${price:.3f}  Size: {size:.1f} shares  Spend: ${t['spend']:.2f}")

        try:
            order_args = OrderArgs(
                price=round(price, 2),  # CLOB requires 2 decimal places
                size=size,
                side=BUY,
                token_id=token_id,
            )
            signed_order = client.create_order(order_args)
            result = client.post_order(signed_order)

            status = result.get("status", "?")
            order_id = result.get("orderID", "?")[:16]
            print(f"    ✅ {status} (ID: {order_id}...)")

            results.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "city": t["city"],
                "date": t["date"],
                "bucket": t["bucket"],
                "price": price,
                "spend": t["spend"],
                "size": size,
                "edge": t["edge"],
                "model_prob": t["model_prob"],
                "order_id": result.get("orderID", ""),
                "status": status,
            })
            succeeded += 1

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            err = str(e)[:100]
            print(f"    ❌ {err}")
            failed += 1
            results.append({
                "city": t["city"], "date": t["date"], "bucket": t["bucket"],
                "error": err, "status": "FAILED",
            })
            time.sleep(1)

    # Summary
    print(f"\n{'='*80}")
    print(f"  RESULTS: {succeeded} succeeded, {failed} failed")
    print(f"  Total deployed: ${sum(r.get('spend', 0) for r in results if r.get('status') != 'FAILED'):,.2f}")
    print(f"{'='*80}")

    # Save trade log
    os.makedirs("data", exist_ok=True)
    log_path = "data/weather_trades.json"
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    existing.extend(results)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Trades logged: {log_path}")


if __name__ == "__main__":
    main()
