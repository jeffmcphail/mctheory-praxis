"""
scan_all_weather.py — Full Polymarket Weather Market Scanner

Scans ALL active temperature markets on Polymarket, compares each
bucket against GFS ensemble forecast, and ranks opportunities by edge.

Usage:
    python scan_all_weather.py
    python scan_all_weather.py --min-edge 0.10
    python scan_all_weather.py --city nyc
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, ".")
from engines.weather_forecaster import WeatherForecaster, CITIES

# Map Polymarket city names to our forecaster city keys
POLY_CITY_MAP = {
    "nyc": "new_york",
    "new york": "new_york",
    "chicago": "chicago",
    "miami": "miami",
    "los angeles": "los_angeles",
    "la": "los_angeles",
    "denver": "denver",
    "london": "london",
    "toronto": None,       # Need to add to forecaster
    "seoul": None,
    "hong kong": None,
    "shanghai": None,
    "tokyo": None,
    "paris": None,
    "beijing": None,
    "singapore": None,
    "wellington": None,
    "atlanta": None,       # Need to add
    "seattle": None,       # Need to add
    "dallas": None,        # Need to add
    "tel aviv": None,
    "buenos aires": None,
    "munich": None,
    "milan": None,
    "ankara": None,
    "taipei": None,
    "lucknow": None,
}

# Additional cities we can add dynamically
EXTRA_CITIES = {
    "toronto": {"lat": 43.6532, "lon": -79.3832, "tz": -4},
    "seoul": {"lat": 37.5665, "lon": 126.9780, "tz": 9},
    "hong kong": {"lat": 22.3193, "lon": 114.1694, "tz": 8},
    "shanghai": {"lat": 31.2304, "lon": 121.4737, "tz": 8},
    "tokyo": {"lat": 35.6762, "lon": 139.6503, "tz": 9},
    "paris": {"lat": 48.8566, "lon": 2.3522, "tz": 2},
    "beijing": {"lat": 39.9042, "lon": 116.4074, "tz": 8},
    "singapore": {"lat": 1.3521, "lon": 103.8198, "tz": 8},
    "wellington": {"lat": -41.2865, "lon": 174.7762, "tz": 12},
    "atlanta": {"lat": 33.7490, "lon": -84.3880, "tz": -4},
    "seattle": {"lat": 47.6062, "lon": -122.3321, "tz": -7},
    "dallas": {"lat": 32.7767, "lon": -96.7970, "tz": -5},
    "tel aviv": {"lat": 32.0853, "lon": 34.7818, "tz": 3},
    "buenos aires": {"lat": -34.6037, "lon": -58.3816, "tz": -3},
    "munich": {"lat": 48.1351, "lon": 11.5820, "tz": 2},
    "milan": {"lat": 45.4642, "lon": 9.1900, "tz": 2},
    "ankara": {"lat": 39.9334, "lon": 32.8597, "tz": 3},
    "taipei": {"lat": 25.0330, "lon": 121.5654, "tz": 8},
    "lucknow": {"lat": 26.8467, "lon": 80.9462, "tz": 5},
}


def parse_city_from_slug(slug):
    """Extract city name from Polymarket slug."""
    # "highest-temperature-in-nyc-on-april-5-2026"
    match = re.search(r'highest-temperature-in-(.+?)-on-', slug)
    if match:
        city_raw = match.group(1).replace("-", " ")
        return city_raw
    return None


def parse_date_from_slug(slug):
    """Extract date from slug."""
    # "highest-temperature-in-nyc-on-april-5-2026"
    match = re.search(r'-on-([a-z]+)-(\d+)-(\d{4})', slug)
    if match:
        month_map = {"january": "01", "february": "02", "march": "03",
                     "april": "04", "may": "05", "june": "06",
                     "july": "07", "august": "08", "september": "09",
                     "october": "10", "november": "11", "december": "12"}
        month = month_map.get(match.group(1), "01")
        day = match.group(2).zfill(2)
        year = match.group(3)
        return f"{year}-{month}-{day}"
    return None


def parse_bucket_temp(title, is_celsius=False):
    """
    Parse temperature range from bucket title.
    Returns (low, high, type) where type is 'range', 'below', or 'above'.
    
    Examples:
      "52-53°F" -> (52, 53, 'range')
      "67°F or below" -> (None, 67, 'below')  
      "86°F or higher" -> (86, None, 'above')
      "14°C" -> (14, 14, 'exact') [for °C markets with single values]
    """
    # "X°F or below" / "X°F or lower"
    match = re.search(r'(\d+)°[FC]\s+or\s+(?:below|lower)', title, re.IGNORECASE)
    if match:
        return (None, float(match.group(1)), "below")
    
    # "X°F or higher" / "X°F or above"  
    match = re.search(r'(\d+)°[FC]\s+or\s+(?:higher|above)', title, re.IGNORECASE)
    if match:
        return (float(match.group(1)), None, "above")
    
    # "X-Y°F" range
    match = re.search(r'(\d+)-(\d+)°[FC]', title)
    if match:
        return (float(match.group(1)), float(match.group(2)), "range")
    
    # "X°C" exact (common in international markets)
    match = re.search(r'(\d+)°C', title)
    if match:
        val = float(match.group(1))
        return (val, val, "exact_c")
    
    # "X°F" exact
    match = re.search(r'(\d+)°F', title)
    if match:
        val = float(match.group(1))
        return (val, val, "exact_f")
    
    return None


def c_to_f(c):
    """Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def f_to_c(f):
    """Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9


def compute_bucket_probability(ensemble_highs, bucket_info, is_celsius=False):
    """
    Compute model probability for a temperature bucket.
    
    ensemble_highs: array of daily high temps in °F from GFS
    bucket_info: (low, high, type) from parse_bucket_temp
    is_celsius: if True, bucket thresholds are in °C
    """
    if bucket_info is None:
        return None
    
    low, high, btype = bucket_info
    highs = ensemble_highs
    n = len(highs)
    
    # Handle exact types FIRST (before is_celsius conversion)
    if btype == "exact_c":
        # "19°C" means high rounds to 19°C, i.e., falls in [18.5, 19.5) °C
        target_c = low  # original value in °C (not yet converted)
        low_f = c_to_f(target_c - 0.5)
        high_f = c_to_f(target_c + 0.5)
        return sum(1 for v in highs if low_f <= v < high_f) / n
    elif btype == "exact_f":
        # "52°F" means high rounds to 52°F, i.e., falls in [51.5, 52.5) °F
        target_f = low
        return sum(1 for v in highs if target_f - 0.5 <= v < target_f + 0.5) / n
    
    # For range/below/above types, convert °C thresholds to °F
    if is_celsius:
        if low is not None:
            low = c_to_f(low)
        if high is not None:
            high = c_to_f(high)
    
    if btype == "below":
        # P(high <= threshold)
        return sum(1 for v in highs if v <= high) / n
    elif btype == "above":
        # P(high >= threshold)
        return sum(1 for v in highs if v >= low) / n
    elif btype == "range":
        # P(low <= high_temp <= high)
        return sum(1 for v in highs if low <= v <= high) / n
    
    return None


def fetch_ensemble_for_city(forecaster, city_raw, target_date):
    """Get ensemble highs for a city, adding it dynamically if needed."""
    # Check if city is in our standard config
    city_key = POLY_CITY_MAP.get(city_raw)
    
    if city_key and city_key in CITIES:
        results = forecaster.scan_thresholds(city_key, target_date)
        if results:
            return results[0].ensemble_values, False  # values in °F, not celsius
        return None, False
    
    # Try dynamic city
    extra = EXTRA_CITIES.get(city_raw)
    if extra is None:
        return None, False
    
    # Add temporarily to CITIES
    from engines.weather_forecaster import CityConfig
    temp_key = city_raw.replace(" ", "_")
    CITIES[temp_key] = CityConfig(
        name=temp_key,
        display_name=city_raw.title(),
        latitude=extra["lat"],
        longitude=extra["lon"],
        kalshi_ticker_prefix="",
        resolution_station="",
        tz_offset_hours=extra["tz"],
    )
    
    results = forecaster.scan_thresholds(temp_key, target_date)
    if results:
        return results[0].ensemble_values, False
    return None, False


def main():
    parser = argparse.ArgumentParser(description="Scan all Polymarket weather markets")
    parser.add_argument("--min-edge", type=float, default=0.08,
                        help="Minimum edge to flag (default: 8%%)")
    parser.add_argument("--city", type=str, default=None,
                        help="Filter to specific city (e.g., 'nyc')")
    parser.add_argument("--bankroll", type=float, default=500.0,
                        help="Bankroll for Kelly sizing")
    args = parser.parse_args()

    forecaster = WeatherForecaster()
    
    print(f"\n{'='*80}")
    print(f"POLYMARKET WEATHER SCANNER — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Min edge: {args.min_edge:.0%}")
    print(f"  Bankroll: ${args.bankroll:,.0f}")
    print(f"{'='*80}")
    
    # Fetch all active weather temperature markets (paginate through multiple tags)
    print(f"\n  Fetching weather markets from Polymarket...")
    all_events = []
    seen_ids = set()
    
    for tag in ["weather", "temperature", "forecast"]:
        for offset in range(0, 500, 100):
            r = requests.get("https://gamma-api.polymarket.com/events", params={
                "tag_slug": tag,
                "limit": "100",
                "offset": str(offset),
                "order": "startDate",
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
            time.sleep(0.3)  # rate limit
    
    # Filter to "highest temperature" markets
    temp_events = [e for e in all_events if "highest temperature" in e.get("title", "").lower()]
    
    # Optional city filter
    if args.city:
        temp_events = [e for e in temp_events if args.city.lower() in e.get("slug", "").lower()]
    
    print(f"  Found {len(temp_events)} temperature markets")
    
    # Process each event
    all_signals = []
    cities_processed = set()
    cities_failed = set()
    
    for event in temp_events:
        slug = event.get("slug", "")
        title = event.get("title", "")
        city_raw = parse_city_from_slug(slug)
        target_date = parse_date_from_slug(slug)
        
        if not city_raw or not target_date:
            continue
        
        # Check if market uses °C or °F
        markets = event.get("markets", [])
        if not markets:
            continue
        
        sample_title = markets[0].get("groupItemTitle", "")
        is_celsius = "°C" in sample_title or "°c" in sample_title
        
        # Get ensemble forecast
        ensemble_key = f"{city_raw}_{target_date}"
        if city_raw in cities_failed:
            continue
            
        ensemble_highs, _ = fetch_ensemble_for_city(forecaster, city_raw, target_date)
        
        if ensemble_highs is None:
            if city_raw not in cities_failed:
                cities_failed.add(city_raw)
            continue
        
        cities_processed.add(city_raw)
        
        # Compare each bucket
        for m in markets:
            bucket_title = m.get("groupItemTitle", m.get("question", "?"))
            prices = json.loads(m.get("outcomePrices", "[0,0]"))
            market_prob = float(prices[0])
            volume = float(m.get("volume", 0))
            
            if market_prob <= 0.01 or market_prob >= 0.99:
                continue  # Skip extreme prices
            
            bucket_info = parse_bucket_temp(bucket_title, is_celsius)
            model_prob = compute_bucket_probability(ensemble_highs, bucket_info, is_celsius)
            
            if model_prob is None:
                continue
            
            edge = model_prob - market_prob
            
            # Kelly sizing
            if abs(edge) >= args.min_edge:
                if edge > 0:
                    direction = "BUY YES"
                    buy_price = market_prob
                else:
                    direction = "BUY NO"
                    buy_price = 1 - market_prob
                    
                win_prob = model_prob if edge > 0 else (1 - model_prob)
                if buy_price > 0 and buy_price < 1:
                    odds = (1 - buy_price) / buy_price
                    kelly = max(0, (win_prob * odds - (1 - win_prob)) / odds)
                    kelly_frac = kelly * 0.15
                    position = min(kelly_frac * args.bankroll, 100.0)
                else:
                    kelly = 0
                    position = 0
                
                all_signals.append({
                    "city": city_raw,
                    "date": target_date,
                    "bucket": bucket_title,
                    "market_prob": market_prob,
                    "model_prob": model_prob,
                    "edge": edge,
                    "direction": direction,
                    "kelly": kelly,
                    "position": position,
                    "volume": volume,
                    "slug": slug,
                    "condition_id": m.get("conditionId", ""),
                })
    
    # Sort by absolute edge
    all_signals.sort(key=lambda s: abs(s["edge"]), reverse=True)
    
    # Display results
    print(f"\n  Cities scanned: {len(cities_processed)} ({', '.join(sorted(cities_processed))})")
    if cities_failed:
        print(f"  Cities without forecast data: {len(cities_failed)} ({', '.join(sorted(cities_failed))})")
    
    tradeable = [s for s in all_signals if abs(s["edge"]) >= args.min_edge]
    
    print(f"\n{'='*80}")
    print(f"TRADEABLE SIGNALS — {len(tradeable)} opportunities (edge >= {args.min_edge:.0%})")
    print(f"{'='*80}")
    
    if tradeable:
        print(f"\n  {'City':<14s} {'Date':<12s} {'Bucket':<18s} {'Mkt':>5s} {'Model':>6s} "
              f"{'Edge':>6s} {'Dir':<8s} {'$Size':>6s} {'Vol':>8s}")
        print(f"  {'-'*88}")
        
        for s in tradeable:
            print(f"  {s['city']:<14s} {s['date']:<12s} {s['bucket']:<18s} "
                  f"{s['market_prob']:>4.0%} {s['model_prob']:>5.0%} "
                  f"{s['edge']:>+5.0%} {s['direction']:<8s} "
                  f"${s['position']:>5.1f} ${s['volume']:>7,.0f}")
        
        total_position = sum(s["position"] for s in tradeable)
        total_ev = sum(s["edge"] * s["position"] for s in tradeable)
        
        print(f"\n  {'─'*88}")
        print(f"  Total signals: {len(tradeable)}")
        print(f"  Total position size: ${total_position:,.2f}")
        print(f"  Estimated total edge: ${total_ev:+,.2f}")
    else:
        print(f"\n  No signals above {args.min_edge:.0%} edge threshold.")
        print(f"  Closest opportunities:")
        for s in all_signals[:10]:
            print(f"    {s['city']:<14s} {s['date']:<12s} {s['bucket']:<18s} "
                  f"mkt={s['market_prob']:.0%} model={s['model_prob']:.0%} "
                  f"edge={s['edge']:+.0%}")
    
    # Save signals
    output_path = "data/weather_signals.json"
    import os
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_signals, f, indent=2)
    print(f"\n  All signals saved: {output_path}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
