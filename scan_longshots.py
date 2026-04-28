"""
scan_longshots.py — Longshot Bias Scanner for Polymarket

Based on Whelan 2026 CEPR finding: contracts <10c win far less than
their price implies. Strategy: BUY NO on overpriced longshots.

Also scans for underpriced favorites (>90c) where BUY YES is +EV.

Usage:
    python scan_longshots.py                     # Scan all categories
    python scan_longshots.py --category politics  # Filter category
    python scan_longshots.py --min-volume 5000    # Min volume filter
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
import requests

# Polymarket fee: 2% on winnings
POLY_FEE_RATE = 0.02

# Whelan bias model: empirical win rates vs implied probability
# For contracts <10c, they win ~40% less often than price implies
# For contracts >90c, they win slightly MORE than price implies
# These are conservative estimates from the paper
def adjusted_win_rate(market_price):
    """
    Estimate true probability given market price, incorporating
    the favorite-longshot bias from Whelan 2026.
    
    Returns estimated true probability of YES.
    """
    p = market_price
    if p <= 0.05:
        return p * 0.30   # 70% overpriced
    elif p <= 0.10:
        return p * 0.50   # 50% overpriced
    elif p <= 0.20:
        return p * 0.70   # 30% overpriced
    elif p <= 0.30:
        return p * 0.85   # 15% overpriced
    elif p >= 0.95:
        return min(p * 1.03, 0.99)  # 3% underpriced
    elif p >= 0.90:
        return min(p * 1.02, 0.99)  # 2% underpriced
    elif p >= 0.80:
        return p * 1.01   # 1% underpriced
    else:
        return p  # Mid-range: no systematic bias


def expected_return_buy_no(yes_price, fee_rate=POLY_FEE_RATE):
    """Expected return from buying NO on a longshot YES contract."""
    no_price = 1.0 - yes_price
    true_yes_prob = adjusted_win_rate(yes_price)
    true_no_prob = 1.0 - true_yes_prob
    
    # Payout if NO wins: $1.00 minus fee on winnings
    payout = 1.0 - fee_rate * (1.0 - no_price)
    
    # Expected value
    ev = true_no_prob * payout - no_price
    return_pct = ev / no_price if no_price > 0 else 0
    
    return {
        "direction": "BUY NO",
        "cost": no_price,
        "true_prob": true_no_prob,
        "ev": ev,
        "return_pct": return_pct,
    }


def expected_return_buy_yes(yes_price, fee_rate=POLY_FEE_RATE):
    """Expected return from buying YES on an underpriced favorite."""
    true_yes_prob = adjusted_win_rate(yes_price)
    
    # Payout if YES wins: $1.00 minus fee on winnings
    payout = 1.0 - fee_rate * (1.0 - yes_price)
    
    # Expected value
    ev = true_yes_prob * payout - yes_price
    return_pct = ev / yes_price if yes_price > 0 else 0
    
    return {
        "direction": "BUY YES",
        "cost": yes_price,
        "true_prob": true_yes_prob,
        "ev": ev,
        "return_pct": return_pct,
    }


def fetch_all_markets(categories=None, min_volume=1000):
    """Fetch active markets from Polymarket across all categories."""
    print(f"  Fetching markets from Polymarket...")
    
    all_events = []
    seen_ids = set()
    
    # Fetch by different tag slugs to cover all categories
    tags = [
        "politics", "sports", "crypto", "finance", "tech",
        "culture", "geopolitics", "economy", "elections",
        "entertainment", "science", "business",
    ]
    
    if categories:
        tags = [t for t in tags if t in categories]
    
    for tag in tags:
        for offset in range(0, 300, 100):
            try:
                r = requests.get("https://gamma-api.polymarket.com/events", params={
                    "tag_slug": tag,
                    "limit": "100",
                    "offset": str(offset),
                    "order": "volume",
                    "ascending": "false",
                    "active": "true",
                    "closed": "false",
                })
                batch = r.json()
                if not batch:
                    break
                for e in batch:
                    eid = e.get("id", "")
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        all_events.append(e)
                if len(batch) < 100:
                    break
                time.sleep(0.3)
            except Exception as e:
                print(f"    Error fetching {tag}: {e}")
                break
    
    print(f"  Fetched {len(all_events)} events")
    
    # Extract individual markets (binary YES/NO contracts)
    markets = []
    for event in all_events:
        event_title = event.get("title", "")
        event_slug = event.get("slug", "")
        event_markets = event.get("markets", [])
        
        # Skip weather (we handle separately)
        if "temperature" in event_title.lower() or "weather" in event_title.lower():
            continue
        
        for m in event_markets:
            try:
                prices = json.loads(m.get("outcomePrices", "[0,0]"))
                yes_price = float(prices[0])
                volume = float(m.get("volume", 0))
                
                if volume < min_volume:
                    continue
                if yes_price <= 0.01 or yes_price >= 0.99:
                    continue
                
                question = m.get("question", m.get("groupItemTitle", "?"))
                
                markets.append({
                    "question": question[:80],
                    "event_title": event_title[:60],
                    "yes_price": yes_price,
                    "volume": volume,
                    "slug": event_slug,
                    "token_ids": m.get("clobTokenIds", ""),
                    "end_date": m.get("endDate", ""),
                    "condition_id": m.get("conditionId", ""),
                })
            except (json.JSONDecodeError, ValueError, IndexError):
                continue
    
    return markets


def main():
    parser = argparse.ArgumentParser(description="Longshot Bias Scanner")
    parser.add_argument("--min-volume", type=float, default=5000,
                        help="Min volume (default: $5k)")
    parser.add_argument("--min-return", type=float, default=0.02,
                        help="Min expected return (default: 2%%)")
    parser.add_argument("--category", type=str, nargs="*", default=None,
                        help="Filter categories")
    parser.add_argument("--bankroll", type=float, default=85.0,
                        help="Available bankroll")
    args = parser.parse_args()

    print(f"\n{'='*85}")
    print(f"LONGSHOT BIAS SCANNER — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Based on Whelan 2026 CEPR: 300k+ contracts, favorite-longshot bias")
    print(f"  Min volume: ${args.min_volume:,.0f}  |  Min return: {args.min_return:.0%}")
    print(f"{'='*85}")
    
    markets = fetch_all_markets(args.category, args.min_volume)
    print(f"  {len(markets)} markets above volume threshold")
    
    # Evaluate each market for longshot bias opportunity
    opportunities = []
    
    for m in markets:
        yes_price = m["yes_price"]
        
        # Check longshot side (YES < 15c): BUY NO
        if yes_price < 0.15:
            result = expected_return_buy_no(yes_price)
            if result["return_pct"] >= args.min_return:
                opportunities.append({
                    **m,
                    **result,
                    "type": "LONGSHOT_FADE",
                })
        
        # Check favorite side (YES > 88c): BUY YES
        if yes_price > 0.88:
            result = expected_return_buy_yes(yes_price)
            if result["return_pct"] >= args.min_return:
                opportunities.append({
                    **m,
                    **result,
                    "type": "FAVORITE_BACK",
                })
    
    # Sort by expected return
    opportunities.sort(key=lambda o: o["return_pct"], reverse=True)
    
    # Separate by type
    longshots = [o for o in opportunities if o["type"] == "LONGSHOT_FADE"]
    favorites = [o for o in opportunities if o["type"] == "FAVORITE_BACK"]
    
    # Display longshots
    print(f"\n{'='*85}")
    print(f"LONGSHOT FADES (BUY NO) — {len(longshots)} opportunities")
    print(f"  These contracts are overpriced per favorite-longshot bias")
    print(f"{'='*85}\n")
    
    if longshots:
        print(f"  {'Question':<55s} {'YES$':>5s} {'NO$':>5s} {'E[R]':>6s} {'Vol':>10s}")
        print(f"  {'-'*85}")
        for o in longshots[:30]:
            print(f"  {o['question']:<55s} {o['yes_price']:>4.0%} {o['cost']:>4.0%} "
                  f"{o['return_pct']:>+5.1%} ${o['volume']:>9,.0f}")
    else:
        print(f"  No longshot fade opportunities found above {args.min_return:.0%} threshold")
    
    # Display favorites
    print(f"\n{'='*85}")
    print(f"FAVORITE BACKS (BUY YES) — {len(favorites)} opportunities")
    print(f"  These favorites are slightly underpriced per the bias")
    print(f"{'='*85}\n")
    
    if favorites:
        print(f"  {'Question':<55s} {'YES$':>5s} {'E[R]':>6s} {'Vol':>10s}")
        print(f"  {'-'*85}")
        for o in favorites[:30]:
            print(f"  {o['question']:<55s} {o['yes_price']:>4.0%} "
                  f"{o['return_pct']:>+5.1%} ${o['volume']:>9,.0f}")
    else:
        print(f"  No favorite back opportunities found above {args.min_return:.0%} threshold")
    
    # Summary
    total = len(longshots) + len(favorites)
    print(f"\n{'='*85}")
    print(f"SUMMARY")
    print(f"  Longshot fades: {len(longshots)}")
    print(f"  Favorite backs: {len(favorites)}")
    print(f"  Total opportunities: {total}")
    if opportunities:
        avg_return = sum(o["return_pct"] for o in opportunities) / len(opportunities)
        print(f"  Average expected return: {avg_return:+.1%}")
    print(f"{'='*85}")
    
    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/longshot_signals.json", "w") as f:
        json.dump(opportunities, f, indent=2, default=str)
    print(f"  Saved: data/longshot_signals.json")


if __name__ == "__main__":
    main()
