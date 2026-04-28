"""Find the exact inflation and Fed rate cut markets on Polymarket."""
import json, requests, time

GAMMA = "https://gamma-api.polymarket.com"

search_terms = [
    "inflation",
    "CPI",
    "Fed rate",
    "interest rate",
    "rate cut",
]

found = {}
for tag in ["economy", "finance", "politics", "geopolitics"]:
    for offset in range(0, 300, 100):
        try:
            r = requests.get(f"{GAMMA}/events", params={
                "tag_slug": tag, "limit": "100", "offset": str(offset),
                "active": "true", "closed": "false",
                "order": "volume", "ascending": "false",
            })
            for event in r.json():
                title = event.get("title", "")
                slug = event.get("slug", "")
                t_lower = title.lower()
                
                if not any(term in t_lower for term in [
                    "inflation", "cpi", "rate cut", "interest rate", "fed",
                    "recession", "gdp", "unemployment"
                ]):
                    continue
                
                if slug in found:
                    continue
                found[slug] = True
                
                print(f"\n  EVENT: {title}")
                print(f"  Slug: {slug}")
                
                for m in event.get("markets", []):
                    q = m.get("question", m.get("groupItemTitle", ""))
                    prices = json.loads(m.get("outcomePrices", "[0,0]"))
                    vol = float(m.get("volume", 0))
                    end = m.get("endDate", "")[:10]
                    cid = m.get("conditionId", "")[:16]
                    
                    if vol < 500:
                        continue
                    
                    yes_price = float(prices[0])
                    print(f"    {q[:75]}")
                    print(f"      YES={yes_price:.1%}  vol=${vol:,.0f}  end={end}")
            
            if len(r.json()) < 100:
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error: {e}")
            break

print("\n\nDone.")
