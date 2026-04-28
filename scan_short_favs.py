"""Scan for short-dated favorites resolving within 7 days."""
import json
import os
import time
from datetime import datetime, timezone, timedelta
import requests

POLY_FEE = 0.02
NOW = datetime.now(timezone.utc)
CUTOFF = NOW + timedelta(days=7)  # Resolve by April 12

RISKY_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "crypto", "price of", "token", "fdv", "market cap",
    "strike", "war", "invade", "military", "nuclear",
    "temperature", "weather", "highest temp",
]

def is_risky(q):
    return any(kw in q.lower() for kw in RISKY_KEYWORDS)

print(f"\n{'='*90}")
print(f"SHORT-DATED FAVORITES SCANNER — resolving by {CUTOFF.strftime('%Y-%m-%d')}")
print(f"{'='*90}")

all_events = []
seen = set()

for tag in ["politics", "sports", "crypto", "finance", "tech", "culture",
            "geopolitics", "economy", "elections", "entertainment", "science",
            "business", "esports"]:
    for offset in range(0, 300, 100):
        try:
            r = requests.get("https://gamma-api.polymarket.com/events", params={
                "tag_slug": tag, "limit": "100", "offset": str(offset),
                "order": "volume", "ascending": "false",
                "active": "true", "closed": "false",
            })
            batch = r.json()
            if not batch:
                break
            for e in batch:
                eid = e.get("id", "")
                if eid not in seen:
                    seen.add(eid)
                    all_events.append(e)
            if len(batch) < 100:
                break
            time.sleep(0.3)
        except:
            break

print(f"  Fetched {len(all_events)} events")

# Extract markets with end dates within 7 days
opportunities = []
for event in all_events:
    for m in event.get("markets", []):
        try:
            end_str = m.get("endDate", "")
            if not end_str:
                continue
            end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            if end_date > CUTOFF:
                continue  # Too far out
            if end_date < NOW:
                continue  # Already expired

            prices = json.loads(m.get("outcomePrices", "[0,0]"))
            yes_price = float(prices[0])
            volume = float(m.get("volume", 0))
            question = m.get("question", m.get("groupItemTitle", "?"))

            if volume < 1000:
                continue
            if is_risky(question):
                continue

            days_to_resolve = (end_date - NOW).total_seconds() / 86400

            # Favorites: YES >= 90%
            if yes_price >= 0.90 and yes_price <= 0.98:
                profit_per_contract = (1.0 - yes_price) * (1 - POLY_FEE)
                return_pct = profit_per_contract / yes_price
                annualized = (1 + return_pct) ** (365 / max(days_to_resolve, 1)) - 1

                opportunities.append({
                    "question": question[:70],
                    "yes_price": yes_price,
                    "volume": volume,
                    "end_date": end_str[:10],
                    "days": round(days_to_resolve, 1),
                    "return_pct": return_pct,
                    "annualized": annualized,
                    "profit_per_dollar": profit_per_contract / yes_price,
                    "slug": event.get("slug", ""),
                    "event_title": event.get("title", "")[:60],
                    "token_ids": m.get("clobTokenIds", ""),
                })

            # Also check: longshots resolving soon (YES <= 10%)
            if yes_price <= 0.10 and yes_price >= 0.02:
                no_price = 1.0 - yes_price
                profit_per_contract = yes_price * (1 - POLY_FEE)
                return_pct = profit_per_contract / no_price
                annualized = (1 + return_pct) ** (365 / max(days_to_resolve, 1)) - 1

                opportunities.append({
                    "question": question[:70],
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "volume": volume,
                    "end_date": end_str[:10],
                    "days": round(days_to_resolve, 1),
                    "return_pct": return_pct,
                    "annualized": annualized,
                    "type": "LONGSHOT_FADE",
                    "slug": event.get("slug", ""),
                    "event_title": event.get("title", "")[:60],
                    "token_ids": m.get("clobTokenIds", ""),
                })

        except (json.JSONDecodeError, ValueError, KeyError):
            continue

# Separate favorites and longshots
favs = [o for o in opportunities if o.get("type") != "LONGSHOT_FADE"]
longs = [o for o in opportunities if o.get("type") == "LONGSHOT_FADE"]

# Sort by annualized return
favs.sort(key=lambda o: o["annualized"], reverse=True)
longs.sort(key=lambda o: o["annualized"], reverse=True)

print(f"\n  {len(favs)} short-dated favorites, {len(longs)} short-dated longshots")

print(f"\n{'='*90}")
print(f"SHORT-DATED FAVORITES (BUY YES) — resolving within 7 days")
print(f"{'='*90}\n")
print(f"  {'Question':<50s} {'YES':>5s} {'Days':>5s} {'Ret':>6s} {'Ann.':>8s} {'Vol':>10s}")
print(f"  {'-'*90}")

for o in favs[:40]:
    print(f"  {o['question']:<50s} {o['yes_price']:>4.0%} {o['days']:>5.1f} "
          f"{o['return_pct']:>+5.1%} {o['annualized']:>+7.0%} ${o['volume']:>9,.0f}")

if longs:
    print(f"\n{'='*90}")
    print(f"SHORT-DATED LONGSHOT FADES (BUY NO) — resolving within 7 days")
    print(f"{'='*90}\n")
    print(f"  {'Question':<50s} {'YES':>5s} {'Days':>5s} {'Ret':>6s} {'Ann.':>8s} {'Vol':>10s}")
    print(f"  {'-'*90}")

    for o in longs[:20]:
        print(f"  {o['question']:<50s} {o['yes_price']:>4.0%} {o['days']:>5.1f} "
              f"{o['return_pct']:>+5.1%} {o['annualized']:>+7.0%} ${o['volume']:>9,.0f}")

# Save
os.makedirs("data", exist_ok=True)
with open("data/short_dated_signals.json", "w") as f:
    json.dump({"favorites": favs, "longshots": longs}, f, indent=2, default=str)
print(f"\n  Saved: data/short_dated_signals.json")
