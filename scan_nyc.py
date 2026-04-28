"""Fetch NYC weather market from Polymarket and compare with our forecast."""
import requests
import json
import sys
sys.path.insert(0, ".")

# Fetch NYC April 5 market
slug = "highest-temperature-in-nyc-on-april-5-2026"
r = requests.get("https://gamma-api.polymarket.com/events", params={"slug": slug})
events = r.json()

if not events:
    # Try April 6
    slug = "highest-temperature-in-nyc-on-april-6-2026"
    r = requests.get("https://gamma-api.polymarket.com/events", params={"slug": slug})
    events = r.json()

if not events:
    print("No NYC weather markets found for April 5 or 6")
    # List what's available
    r2 = requests.get("https://gamma-api.polymarket.com/events", params={
        "tag_slug": "weather", "limit": "20", "order": "startDate",
        "ascending": "false", "active": "true", "closed": "false"
    })
    nyc = [e for e in r2.json() if "nyc" in e.get("slug", "").lower()]
    print(f"\nAvailable NYC markets:")
    for e in nyc[:5]:
        print(f"  {e['slug']}")
    sys.exit(0)

event = events[0]
print(f"\n{'='*70}")
print(f"{event['title']}")
print(f"{'='*70}")

markets = event.get("markets", [])
print(f"  Buckets: {len(markets)}")
print()

# Parse and display each bucket
buckets = []
for m in markets:
    title = m.get("groupItemTitle", m.get("question", "?"))
    prices = json.loads(m.get("outcomePrices", "[0,0]"))
    yes_price = float(prices[0])
    no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
    volume = float(m.get("volume", 0))
    
    buckets.append({
        "title": title,
        "yes_price": yes_price,
        "no_price": no_price,
        "volume": volume,
        "condition_id": m.get("conditionId", ""),
        "token_id": m.get("clobTokenIds", ""),
    })

# Sort by yes_price descending
buckets.sort(key=lambda b: b["yes_price"], reverse=True)

print(f"  {'Bucket':<20s} {'Market%':>8s} {'Yes$':>6s} {'No$':>6s} {'Volume':>10s}")
print(f"  {'-'*55}")
for b in buckets:
    bar = "█" * int(b["yes_price"] * 20) + "░" * (20 - int(b["yes_price"] * 20))
    print(f"  {b['title']:<20s} {b['yes_price']:>7.0%} {b['yes_price']:>5.2f} "
          f"{b['no_price']:>5.2f} ${b['volume']:>9,.0f}")

# Now compare with our forecast
print(f"\n{'='*70}")
print("FORECAST COMPARISON")
print(f"{'='*70}")

try:
    from engines.weather_forecaster import WeatherForecaster
    forecaster = WeatherForecaster()
    
    # Determine date from slug
    date_str = slug.replace("highest-temperature-in-nyc-on-", "").replace("-2026", "")
    parts = date_str.split("-")
    month_map = {"january":"01","february":"02","march":"03","april":"04",
                 "may":"05","june":"06","july":"07","august":"08",
                 "september":"09","october":"10","november":"11","december":"12"}
    month = month_map.get(parts[0], "04")
    day = parts[1] if len(parts) > 1 else "05"
    target_date = f"2026-{month}-{day.zfill(2)}"
    
    results = forecaster.scan_thresholds("new_york", target_date)
    if results:
        print(f"\n  GFS Ensemble for NYC {target_date}:")
        print(f"  Mean high: {results[0].ensemble_mean:.1f}°F")
        print(f"  Range: {results[0].ensemble_min:.1f} - {results[0].ensemble_max:.1f}°F")
        print(f"  Std: {results[0].ensemble_std:.1f}°F")
        print()
        
        # Compare each bucket
        print(f"  {'Bucket':<20s} {'Market':>7s} {'Model':>7s} {'Edge':>7s} {'Signal'}")
        print(f"  {'-'*60}")
        
        for b in buckets:
            title = b["title"]
            market_prob = b["yes_price"]
            
            # Parse temperature range from title
            # Formats: "52-53°F", "67°F or below", "68-69°F"
            import re
            match = re.search(r'(\d+)-(\d+)', title)
            match_below = re.search(r'(\d+).+below', title)
            match_above = re.search(r'(\d+).+above|higher', title)
            
            if match:
                low_f = float(match.group(1))
                high_f = float(match.group(2))
                # P(bucket) = P(>= low) - P(>= high+1)
                p_above_low = sum(1 for v in results[0].ensemble_values if v >= low_f) / len(results[0].ensemble_values)
                p_above_high = sum(1 for v in results[0].ensemble_values if v >= high_f + 1) / len(results[0].ensemble_values)
                model_prob = p_above_low - p_above_high
            elif match_below:
                threshold = float(match_below.group(1))
                model_prob = sum(1 for v in results[0].ensemble_values if v <= threshold) / len(results[0].ensemble_values)
            elif match_above:
                threshold = float(match_above.group(1))
                model_prob = sum(1 for v in results[0].ensemble_values if v >= threshold) / len(results[0].ensemble_values)
            else:
                model_prob = None
            
            if model_prob is not None:
                edge = model_prob - market_prob
                signal = ""
                if abs(edge) >= 0.08:
                    if edge > 0:
                        signal = "🎯 BUY YES"
                    else:
                        signal = "🎯 BUY NO"
                print(f"  {title:<20s} {market_prob:>6.0%} {model_prob:>6.0%} "
                      f"{edge:>+6.0%}  {signal}")
            else:
                print(f"  {title:<20s} {market_prob:>6.0%}     ?       ?")
    else:
        print("  No forecast data available")
except Exception as e:
    print(f"  Forecast comparison failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
