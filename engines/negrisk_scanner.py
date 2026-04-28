"""
engines/negrisk_scanner.py — NegRisk Market Rebalancing Scanner

Scans multi-outcome Polymarket events where bucket probabilities must
sum to 100%. Finds sum violations and identifies mispriced complementary
probability space for rebalancing trades.

Key insight: In multi-outcome markets (e.g., "What will inflation be?"),
retail flow concentrates on 1-2 favorites, leaving the complementary
space thinly traded. When bucket probabilities don't sum to 100%,
there's a structural arbitrage.

Usage:
    python -m engines.negrisk_scanner scan                    # Find all violations
    python -m engines.negrisk_scanner scan --min-deviation 5  # Only >5% deviations
    python -m engines.negrisk_scanner detail --slug "march-inflation-us-annual"
    python -m engines.negrisk_scanner history --slug "march-inflation-us-annual"
"""
import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def fetch_multi_outcome_events(min_volume=5000, negrisk_only=True):
    """Fetch events with multiple markets (multi-outcome)."""
    events = []
    seen = set()

    tags = [
        "economy", "finance", "politics", "geopolitics", "crypto",
        "sports", "tech", "elections", "culture", "science",
        "business", "iran", "trump",
    ]

    for tag in tags:
        for offset in range(0, 500, 100):
            try:
                r = requests.get(f"{GAMMA_API}/events", params={
                    "tag_slug": tag, "limit": "100", "offset": str(offset),
                    "active": "true", "closed": "false",
                    "order": "volume", "ascending": "false",
                }, timeout=10)
                batch = r.json()
                if not batch:
                    break
                for event in batch:
                    eid = event.get("id", "")
                    if eid in seen:
                        continue
                    seen.add(eid)

                    markets = event.get("markets", [])
                    if len(markets) < 3:  # Need 3+ outcomes to be interesting
                        continue

                    # CRITICAL: Only include actual NegRisk events where
                    # outcomes are mutually exclusive (exactly one wins).
                    # Independent markets (Trump visits, esports props) are NOT NegRisk.
                    neg_risk = event.get("negRisk", False)
                    
                    # Also check if event has negRiskMarketID (another indicator)
                    has_neg_risk_id = bool(event.get("negRiskMarketID", ""))
                    
                    # Filter: must be flagged as NegRisk by Polymarket
                    if negrisk_only and not neg_risk and not has_neg_risk_id:
                        continue

                    total_vol = sum(float(m.get("volume", 0)) for m in markets)
                    if total_vol < min_volume:
                        continue

                    events.append(event)

                if len(batch) < 100:
                    break
                time.sleep(0.2)
            except Exception:
                break

    return events


def analyze_event(event):
    """Analyze probability sums for a multi-outcome event."""
    markets = event.get("markets", [])
    title = event.get("title", "?")
    slug = event.get("slug", "")
    neg_risk = event.get("negRisk", False)

    buckets = []
    total_yes = 0
    total_volume = 0

    for m in markets:
        try:
            prices = json.loads(m.get("outcomePrices", "[0,0]"))
            yes_price = float(prices[0])
            no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
            volume = float(m.get("volume", 0))
            question = m.get("question", m.get("groupItemTitle", ""))
            token_ids = m.get("clobTokenIds", "")
            if isinstance(token_ids, str):
                try:
                    token_ids = json.loads(token_ids) if token_ids else []
                except json.JSONDecodeError:
                    token_ids = []

            buckets.append({
                "question": question,
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": volume,
                "condition_id": m.get("conditionId", ""),
                "token_ids": token_ids,
                "end_date": (m.get("endDate") or "")[:10],
            })
            total_yes += yes_price
            total_volume += volume
        except (json.JSONDecodeError, ValueError):
            continue

    if not buckets:
        return None

    # The sum of all YES prices should be ~1.0 for mutually exclusive outcomes
    # Deviation from 1.0 = potential arbitrage
    deviation = total_yes - 1.0
    deviation_pct = deviation * 100

    # Find overpriced and underpriced buckets
    # In a properly priced NegRisk market, buying NO on all outcomes
    # should cost ~$1.00 total. If it costs less, you profit.
    total_no_cost = sum(b["no_price"] for b in buckets)
    no_arb = len(buckets) - 1 - total_yes  # Cost of buying NO on everything - guaranteed payout

    return {
        "title": title,
        "slug": slug,
        "neg_risk": neg_risk,
        "num_outcomes": len(buckets),
        "total_yes": total_yes,
        "deviation": deviation,
        "deviation_pct": deviation_pct,
        "total_no_cost": total_no_cost,
        "total_volume": total_volume,
        "buckets": sorted(buckets, key=lambda b: b["yes_price"], reverse=True),
        "end_date": buckets[0].get("end_date", "") if buckets else "",
    }


def fetch_price_history(token_id, fidelity=60):
    """Fetch price history for a specific token."""
    try:
        r = requests.get(f"{CLOB_API}/prices-history", params={
            "market": token_id,
            "interval": "all",
            "fidelity": fidelity,
        }, timeout=10)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and "history" in data:
                return data["history"]
            elif isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def cmd_scan(args):
    """Scan all multi-outcome markets for probability sum violations."""
    min_dev = getattr(args, "min_deviation", 3.0)
    min_vol = getattr(args, "min_volume", 5000)
    max_dev = getattr(args, "max_deviation", 50.0)
    min_liquid = getattr(args, "min_liquid_outcomes", 3)

    negrisk_only = not getattr(args, "all", False)

    print(f"\n{'='*120}")
    print(f"NEGRISK REBALANCING SCANNER — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Min deviation: {min_dev}%  |  Max deviation: {max_dev}%  |  "
          f"Min liquid outcomes: {min_liquid}  |  "
          f"{'NegRisk only' if negrisk_only else 'ALL events'}")
    print(f"{'='*120}")

    print(f"\n  Fetching multi-outcome events...")
    events = fetch_multi_outcome_events(min_vol, negrisk_only=negrisk_only)
    print(f"  Found {len(events)} events with 3+ outcomes")

    results = []
    for event in events:
        analysis = analyze_event(event)
        if not analysis:
            continue
        if abs(analysis["deviation_pct"]) < min_dev:
            continue
        if abs(analysis["deviation_pct"]) > max_dev:
            continue

        # Count outcomes with actual volume (not dead markets)
        liquid_outcomes = sum(1 for b in analysis["buckets"] if b["volume"] > 0)
        # Count outcomes with non-zero, non-50% prices (real price discovery)
        priced_outcomes = sum(1 for b in analysis["buckets"]
                             if 0.001 < b["yes_price"] < 0.999)
        analysis["liquid_outcomes"] = liquid_outcomes
        analysis["priced_outcomes"] = priced_outcomes
        avg_vol = analysis["total_volume"] / max(analysis["num_outcomes"], 1)
        analysis["avg_vol_per_outcome"] = avg_vol

        if liquid_outcomes < min_liquid:
            continue

        results.append(analysis)

    results.sort(key=lambda r: r["total_volume"], reverse=True)

    print(f"\n  {len(results)} events with {min_dev}% <= deviation <= {max_dev}% "
          f"and {min_liquid}+ liquid outcomes (sorted by volume)")

    if results:
        print(f"\n{'─'*120}")
        print(f"  PROBABILITY SUM VIOLATIONS")
        print(f"{'─'*120}")
        print(f"  {'#':<3s} {'Event':<55s} {'#Out':>4s} {'Liq':>4s} {'Sum%':>6s} {'Dev%':>7s} "
              f"{'Volume':>12s} {'Resolve':<11s} {'Signal'}")
        print(f"  {'─'*118}")

        for i, r in enumerate(results[:30], 1):
            title = r["title"][:54]
            dev = r["deviation_pct"]

            # Signal interpretation
            if dev > 3:
                signal = f"OVERPRICED +{dev:.1f}% → sell YES basket"
            elif dev < -3:
                signal = f"UNDERPRICED {dev:.1f}% → buy YES basket"
            else:
                signal = f"minor {dev:+.1f}%"

            # Color indicator
            if abs(dev) >= 10:
                indicator = "🔴"
            elif abs(dev) >= 5:
                indicator = "🟡"
            else:
                indicator = "⚪"

            print(f"  {i:<3d} {title:<55s} {r['num_outcomes']:>4d} {r['liquid_outcomes']:>4d} {r['total_yes']*100:>5.1f}% "
                  f"{dev:>+6.1f}% ${r['total_volume']:>11,.0f} {r['end_date']:<11s} "
                  f"{indicator} {signal}")

    # Show detailed breakdown of top violations
    if results:
        print(f"\n{'─'*120}")
        print(f"  TOP VIOLATIONS — BUCKET DETAIL")
        print(f"{'─'*120}")

        for r in results[:5]:
            print(f"\n  📊 {r['title']}")
            print(f"     Sum: {r['total_yes']*100:.1f}%  |  Deviation: {r['deviation_pct']:+.1f}%  |  "
                  f"Volume: ${r['total_volume']:,.0f}  |  Outcomes: {r['num_outcomes']}")
            print(f"     {'Outcome':<50s} {'YES':>6s} {'NO':>6s} {'Vol':>10s}")
            print(f"     {'-'*75}")

            for b in r["buckets"]:
                q = b["question"][:49]
                print(f"     {q:<50s} {b['yes_price']:>5.1%} {b['no_price']:>5.1%} "
                      f"${b['volume']:>9,.0f}")

            # Actionable trade suggestion
            if r["deviation_pct"] > 3:
                print(f"\n     💡 ACTION: Sum > 100% by {r['deviation_pct']:.1f}%. "
                      f"Buy NO on ALL outcomes. Cost ≈ ${r['total_no_cost']:.2f} per set. "
                      f"Guaranteed payout: ${r['num_outcomes']-1:.2f}. "
                      f"Profit: ${r['num_outcomes']-1-r['total_no_cost']:.2f} per set "
                      f"({((r['num_outcomes']-1-r['total_no_cost'])/r['total_no_cost'])*100:.1f}%)")
            elif r["deviation_pct"] < -3:
                print(f"\n     💡 ACTION: Sum < 100% by {abs(r['deviation_pct']):.1f}%. "
                      f"Buy YES on ALL outcomes. Cost ≈ ${r['total_yes']:.2f} per set. "
                      f"Guaranteed payout: $1.00. "
                      f"Profit: ${1.0 - r['total_yes']:.2f} per set "
                      f"({((1.0 - r['total_yes'])/r['total_yes'])*100:.1f}%)")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/negrisk_scan.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "events_scanned": len(events),
            "violations": len(results),
            "results": results[:50],
        }, f, indent=2, default=str)
    print(f"\n  Saved: data/negrisk_scan.json")
    print(f"{'='*120}")


def cmd_detail(args):
    """Show detailed breakdown for a specific event."""
    slug = args.slug
    print(f"\n  Fetching event: {slug}...")

    r = requests.get(f"{GAMMA_API}/events", params={"slug": slug})
    events = r.json()
    if not events:
        print(f"  ❌ Event not found: {slug}")
        return

    analysis = analyze_event(events[0])
    if not analysis:
        print(f"  ❌ Could not analyze event")
        return

    print(f"\n{'='*100}")
    print(f"  {analysis['title']}")
    print(f"  NegRisk: {analysis['neg_risk']}  |  Outcomes: {analysis['num_outcomes']}  |  "
          f"Volume: ${analysis['total_volume']:,.0f}")
    print(f"  Probability Sum: {analysis['total_yes']*100:.2f}%  |  "
          f"Deviation: {analysis['deviation_pct']:+.2f}%")
    print(f"{'='*100}")

    print(f"\n  {'#':<3s} {'Outcome':<60s} {'YES':>7s} {'NO':>7s} {'Volume':>12s}")
    print(f"  {'-'*92}")

    for i, b in enumerate(analysis["buckets"], 1):
        q = b["question"][:59]
        print(f"  {i:<3d} {q:<60s} {b['yes_price']:>6.1%} {b['no_price']:>6.1%} "
              f"${b['volume']:>11,.0f}")

    print(f"  {'-'*92}")
    print(f"  {'':3s} {'TOTAL':<60s} {analysis['total_yes']:>6.1%}")

    # Check if there's a riskless profit
    if analysis["deviation_pct"] > 1:
        print(f"\n  ⚠ OVERPRICED: Sum exceeds 100% by {analysis['deviation_pct']:.2f}%")
        print(f"    Strategy: Buy NO on all {analysis['num_outcomes']} outcomes")
        no_cost = analysis["total_no_cost"]
        payout = analysis["num_outcomes"] - 1
        print(f"    Cost per set: ${no_cost:.4f}")
        print(f"    Payout: ${payout:.2f}")
        if no_cost < payout:
            profit = payout - no_cost
            print(f"    Profit per set: ${profit:.4f} ({profit/no_cost*100:.2f}%)")
        else:
            print(f"    ❌ No profit after costs (need fees < deviation)")
    elif analysis["deviation_pct"] < -1:
        print(f"\n  ⚠ UNDERPRICED: Sum below 100% by {abs(analysis['deviation_pct']):.2f}%")
        print(f"    Strategy: Buy YES on all {analysis['num_outcomes']} outcomes")
        yes_cost = analysis["total_yes"]
        print(f"    Cost per set: ${yes_cost:.4f}")
        print(f"    Payout: $1.00 (one outcome must win)")
        profit = 1.0 - yes_cost
        print(f"    Profit per set: ${profit:.4f} ({profit/yes_cost*100:.2f}%)")
    else:
        print(f"\n  ✅ Properly priced (deviation within ±1%)")


def cmd_history(args):
    """Show price history and sum deviation over time for a multi-outcome event."""
    slug = args.slug
    print(f"\n  Fetching event: {slug}...")

    r = requests.get(f"{GAMMA_API}/events", params={"slug": slug})
    events = r.json()
    if not events:
        print(f"  ❌ Event not found")
        return

    event = events[0]
    markets = event.get("markets", [])
    print(f"  {event.get('title', '?')} — {len(markets)} outcomes")

    # Fetch price history for each outcome
    print(f"  Fetching price histories...")
    histories = {}
    for m in markets:
        token_ids = m.get("clobTokenIds", "")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids) if token_ids else []
            except json.JSONDecodeError:
                continue

        if token_ids:
            q_full = m.get("question", m.get("groupItemTitle", ""))
            # Use last distinctive part of question for display key
            # Strip common prefix to get the unique part
            q_short = q_full[-30:] if len(q_full) > 30 else q_full
            q_key = f"{len(histories)}_{q_short}"
            hist = fetch_price_history(token_ids[0], fidelity=3600)  # Hourly
            if hist:
                histories[q_key] = hist
                print(f"    {q_short}: {len(hist)} points")
            time.sleep(0.3)

    if len(histories) < 2:
        print(f"  Not enough history data")
        return

    # Align timestamps and compute sums
    # Collect all unique timestamps
    all_times = set()
    for hist in histories.values():
        for point in hist:
            all_times.add(point["t"])

    all_times = sorted(all_times)

    print(f"\n  Computing probability sums over time...")
    print(f"\n  {'Time':<20s} {'Sum':>7s} {'Dev':>7s} | Bucket prices...")
    print(f"  {'-'*80}")

    # For each timestamp, find the closest price for each outcome
    last_prices = {q: 0 for q in histories}
    price_lookup = {}
    for q, hist in histories.items():
        for point in hist:
            price_lookup[(q, point["t"])] = point["p"]

    sample_times = all_times[::max(1, len(all_times)//30)]  # Sample ~30 points

    for t in sample_times:
        # Update prices from history
        for q, hist in histories.items():
            if (q, t) in price_lookup:
                last_prices[q] = price_lookup[(q, t)]
            else:
                # Find closest earlier timestamp
                closest = None
                for point in hist:
                    if point["t"] <= t:
                        closest = point["p"]
                if closest is not None:
                    last_prices[q] = closest

        total = sum(last_prices.values())
        dev = (total - 1.0) * 100

        ts = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        bucket_str = " ".join(f"{p:.0%}" for p in last_prices.values())

        indicator = "🔴" if abs(dev) > 5 else "🟡" if abs(dev) > 2 else ""
        print(f"  {ts:<20s} {total:>6.1%} {dev:>+6.1f}% {indicator} | {bucket_str}")

    print(f"\n  Current sum: {sum(last_prices.values())*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="NegRisk Scanner")
    parser.add_argument("--min-volume", type=float, default=5000)
    subs = parser.add_subparsers(dest="command", required=True)

    p_scan = subs.add_parser("scan", help="Scan for probability sum violations")
    p_scan.add_argument("--min-deviation", type=float, default=3.0,
                        help="Minimum deviation percentage (default: 3%%)")
    p_scan.add_argument("--max-deviation", type=float, default=50.0,
                        help="Maximum deviation percentage (default: 50%%)")
    p_scan.add_argument("--min-liquid-outcomes", type=int, default=3,
                        help="Min outcomes with volume > $0 (default: 3)")
    p_scan.add_argument("--all", action="store_true",
                        help="Include non-NegRisk events (independent outcomes)")

    p_detail = subs.add_parser("detail", help="Show detail for specific event")
    p_detail.add_argument("--slug", type=str, required=True)

    p_hist = subs.add_parser("history", help="Show sum deviation over time")
    p_hist.add_argument("--slug", type=str, required=True)

    args = parser.parse_args()
    {"scan": cmd_scan, "detail": cmd_detail, "history": cmd_history}[args.command](args)


if __name__ == "__main__":
    main()
