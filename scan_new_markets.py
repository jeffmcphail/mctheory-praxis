"""
scan_new_markets.py — New Market Lifecycle Scanner

Finds recently created Polymarket markets (last 24-48 hours) and
identifies potential mispricings in early-stage markets where
initial prices haven't been calibrated by the crowd yet.

Key insight: new markets have thin order books and prices set by
1-2 early traders. Information-advantage traders who evaluate
quickly can capture the mispricing before the crowd corrects.

Usage:
    python scan_new_markets.py                       # Last 24h
    python scan_new_markets.py --hours 48            # Last 48h
    python scan_new_markets.py --min-volume 100      # Include very thin markets
    python scan_new_markets.py --category politics   # Filter category
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

GAMMA_API = "https://gamma-api.polymarket.com"


def fetch_new_markets(hours=24, min_volume=100):
    """Fetch markets created within the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_str = cutoff.isoformat()

    all_markets = []
    seen = set()

    # Fetch recent events sorted by creation date
    for offset in range(0, 500, 100):
        try:
            r = requests.get(f"{GAMMA_API}/events", params={
                "limit": "100",
                "offset": str(offset),
                "order": "startDate",
                "ascending": "false",
                "active": "true",
                "closed": "false",
            }, timeout=10)
            batch = r.json()
            if not batch:
                break

            for event in batch:
                created = event.get("createdAt", "")
                if created and created < cutoff_str:
                    continue  # Too old

                for m in event.get("markets", []):
                    cid = m.get("conditionId", "")
                    if cid in seen:
                        continue
                    seen.add(cid)

                    try:
                        prices = json.loads(m.get("outcomePrices", "[0,0]"))
                        yes_price = float(prices[0])
                        volume = float(m.get("volume", 0))
                        liquidity = float(m.get("liquidity", 0))

                        if yes_price <= 0.01 or yes_price >= 0.99:
                            continue

                        all_markets.append({
                            "condition_id": cid,
                            "question": m.get("question", m.get("groupItemTitle", "")),
                            "event_title": event.get("title", ""),
                            "slug": event.get("slug", ""),
                            "yes_price": yes_price,
                            "volume": volume,
                            "liquidity": liquidity,
                            "created_at": created,
                            "end_date": m.get("endDate", "")[:10],
                            "token_ids": m.get("clobTokenIds", ""),
                        })
                    except (json.JSONDecodeError, ValueError):
                        continue

            if len(batch) < 100:
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error: {e}")
            break

    # Filter by volume
    if min_volume > 0:
        all_markets = [m for m in all_markets if m["volume"] >= min_volume]

    return all_markets


def assess_opportunity(market):
    """
    Assess whether a new market might be mispriced.
    
    Heuristics for potential mispricing:
    1. Low volume (<$5k) = price not well-calibrated
    2. Low liquidity = thin order book, easy to move
    3. Price near 50% = maximum uncertainty (good for informed traders)
    4. Short resolution = faster capital turnover
    """
    score = 0
    reasons = []

    volume = market["volume"]
    liquidity = market["liquidity"]
    yes_price = market["yes_price"]
    end_date = market.get("end_date", "")

    # Low volume = less calibrated
    if volume < 1000:
        score += 3
        reasons.append("very low vol")
    elif volume < 5000:
        score += 2
        reasons.append("low vol")
    elif volume < 20000:
        score += 1
        reasons.append("moderate vol")

    # Low liquidity = thin book
    if liquidity < 1000:
        score += 2
        reasons.append("thin book")
    elif liquidity < 5000:
        score += 1
        reasons.append("light liquidity")

    # Price near extremes (more likely mispriced — longshot bias)
    if yes_price < 0.10 or yes_price > 0.90:
        score += 1
        reasons.append("extreme price")

    # Price near 50% = max uncertainty
    if 0.35 < yes_price < 0.65:
        score += 1
        reasons.append("high uncertainty")

    # Short resolution = better capital efficiency
    if end_date:
        try:
            end = datetime.fromisoformat(end_date + "T00:00:00+00:00")
            days = (end - datetime.now(timezone.utc)).days
            if days <= 3:
                score += 2
                reasons.append(f"{days}d resolve")
            elif days <= 7:
                score += 1
                reasons.append(f"{days}d resolve")
        except ValueError:
            pass

    market["opportunity_score"] = score
    market["reasons"] = ", ".join(reasons)
    return market


def main():
    parser = argparse.ArgumentParser(description="New Market Scanner")
    parser.add_argument("--hours", type=int, default=24,
                        help="Look back N hours (default: 24)")
    parser.add_argument("--min-volume", type=float, default=100,
                        help="Min volume (default: $100)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by keyword in title")
    parser.add_argument("--top", type=int, default=30,
                        help="Show top N results")
    args = parser.parse_args()

    print(f"\n{'='*105}")
    print(f"NEW MARKET SCANNER — Markets created in last {args.hours} hours")
    print(f"  Min volume: ${args.min_volume:,.0f}")
    print(f"{'='*105}")

    markets = fetch_new_markets(args.hours, args.min_volume)

    if args.category:
        markets = [m for m in markets
                   if args.category.lower() in m.get("question", "").lower()
                   or args.category.lower() in m.get("event_title", "").lower()]

    print(f"\n  Found {len(markets)} new markets")

    # Assess each market
    for m in markets:
        assess_opportunity(m)

    # Sort by opportunity score
    markets.sort(key=lambda m: m["opportunity_score"], reverse=True)

    # Display
    print(f"\n{'─'*105}")
    print(f"  TOP OPPORTUNITIES (ranked by mispricing potential)")
    print(f"{'─'*105}")
    print(f"  {'#':<3s} {'Score':>5s} {'Question':<50s} {'YES':>5s} "
          f"{'Vol':>9s} {'Liq':>8s} {'Resolve':<11s} {'Signals'}")
    print(f"  {'-'*103}")

    for i, m in enumerate(markets[:args.top], 1):
        q = m["question"][:49]
        score = m["opportunity_score"]

        # Score indicator
        if score >= 5:
            indicator = "🔴"
        elif score >= 3:
            indicator = "🟡"
        else:
            indicator = "⚪"

        print(f"  {i:<3d} {indicator} {score:>2d}  {q:<50s} {m['yes_price']:>4.0%} "
              f"${m['volume']:>8,.0f} ${m['liquidity']:>7,.0f} "
              f"{m['end_date']:<11s} {m['reasons']}")

    # Summary by score tier
    high = [m for m in markets if m["opportunity_score"] >= 5]
    med = [m for m in markets if 3 <= m["opportunity_score"] < 5]
    low = [m for m in markets if m["opportunity_score"] < 3]

    print(f"\n  Summary:")
    print(f"    🔴 High potential:   {len(high)} markets (score >= 5)")
    print(f"    🟡 Medium potential: {len(med)} markets (score 3-4)")
    print(f"    ⚪ Low potential:    {len(low)} markets (score < 3)")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/new_market_signals.json", "w") as f:
        json.dump(markets[:50], f, indent=2, default=str)
    print(f"\n  Saved: data/new_market_signals.json")

    print(f"\n  💡 Next step: Review high-score markets manually.")
    print(f"     Use your domain knowledge to assess if the price is wrong.")
    print(f"     New markets are most mispriced in their first 6-12 hours.")
    print(f"{'='*105}")


if __name__ == "__main__":
    main()
