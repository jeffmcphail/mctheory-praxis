"""
scripts/run_weather_bot.py — Weather Prediction Market Bot CLI

Scans weather prediction markets, compares against GFS ensemble
forecasts, and generates/executes trading signals.

Usage:
    # Forecast only — see what the model predicts (no Kalshi needed)
    python scripts/run_weather_bot.py forecast --city new_york --date 2026-04-05

    # Forecast all cities for tomorrow
    python scripts/run_weather_bot.py forecast --all

    # Scan Kalshi markets + generate signals
    python scripts/run_weather_bot.py scan

    # Paper trade — execute signals without real money
    python scripts/run_weather_bot.py trade --paper

    # Monitor loop — scan every 5 min, paper trade
    python scripts/run_weather_bot.py monitor --duration 24 --interval 300

Environment:
    KALSHI_EMAIL     — Kalshi account email (for trading)
    KALSHI_PASSWORD  — Kalshi account password (for trading)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def get_tomorrow() -> str:
    """Get tomorrow's date string."""
    return (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")


def cmd_forecast(args):
    """Run weather forecast without market data."""
    from engines.weather_forecaster import WeatherForecaster, CITIES

    forecaster = WeatherForecaster()
    target_date = args.date or get_tomorrow()

    print(f"\n{'='*70}")
    print(f"WEATHER FORECAST — {target_date}")
    print(f"{'='*70}")

    if args.all:
        cities = list(CITIES.keys())
    else:
        cities = [args.city]

    for city_name in cities:
        city = CITIES.get(city_name)
        if not city:
            print(f"\n  ❌ Unknown city: {city_name}")
            continue

        print(f"\n  {city.display_name} ({target_date}):")

        results = forecaster.scan_thresholds(city_name, target_date)
        if not results:
            print(f"    No forecast data available")
            continue

        # Find the mean high
        if results:
            mean_high = results[0].ensemble_mean
            print(f"    Ensemble mean high: {mean_high:.1f}°F")
            print(f"    Range: {results[0].ensemble_min:.1f}°F - "
                  f"{results[0].ensemble_max:.1f}°F")
            print(f"    Std dev: {results[0].ensemble_std:.1f}°F")
            print(f"    Members: {results[0].n_members}")
            print()

        for r in results:
            bar = "█" * int(r.model_probability * 20) + \
                  "░" * (20 - int(r.model_probability * 20))
            print(f"    P(high >= {r.threshold:3.0f}°F) = "
                  f"{r.model_probability:5.0%} [{bar}] "
                  f"{r.confidence}")


def cmd_scan(args):
    """Scan markets and generate signals."""
    from engines.weather_forecaster import WeatherForecaster
    from engines.prediction_scanner import KalshiScanner

    forecaster = WeatherForecaster()
    scanner = KalshiScanner(
        demo=not args.live,
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        max_position=args.max_position,
    )

    mode = "LIVE" if args.live else "DEMO"
    print(f"\n{'='*70}")
    print(f"WEATHER BOT — Market Scan [{mode}]")
    print(f"  Bankroll: ${args.bankroll:,.0f}")
    print(f"  Min edge: {args.min_edge:.0%}")
    print(f"  Max position: ${args.max_position:.0f}")
    print(f"{'='*70}")

    # Discover markets
    print(f"\n  Discovering weather markets on Kalshi...")
    markets = scanner.discover_weather_markets()

    if not markets:
        print(f"  No weather markets found.")
        print(f"  (Make sure you have Kalshi API access configured)")
        return

    print(f"\n  Found {len(markets)} weather markets")

    # Generate signals
    print(f"\n  Computing forecasts and generating signals...")
    signals = scanner.generate_signals(markets, forecaster)

    # Display results
    print(f"\n{'='*70}")
    print(f"SIGNALS — {len(signals)} markets analyzed")
    print(f"{'='*70}")

    tradeable = [s for s in signals if s.is_tradeable]
    non_tradeable = [s for s in signals if not s.is_tradeable]

    if tradeable:
        print(f"\n  🎯 TRADEABLE SIGNALS ({len(tradeable)}):\n")
        for s in tradeable:
            print(f"  {s.summary()}")
            print(f"      Market: {s.market.ticker}")
            print(f"      Spread: bid={s.market.yes_bid:.2f} "
                  f"ask={s.market.yes_ask:.2f} "
                  f"({s.market.spread_pct:.1f}%)")
            print(f"      Kelly: {s.kelly_full:.1%} full → "
                  f"${s.position_size_usd:.2f}")
            print(f"      EV: ${s.expected_value:+.2f}")
            print()
    else:
        print(f"\n  No tradeable signals (edge < {args.min_edge:.0%})")

    if non_tradeable and args.verbose:
        print(f"\n  ─ MONITORING ({len(non_tradeable)}):\n")
        for s in non_tradeable[:10]:  # show top 10
            print(f"  {s.summary()}")

    # Save signals
    scanner.save_signals(signals)
    print(f"\n  Signals saved: data/weather_signals.json")

    # Summary
    print(f"\n{'─'*70}")
    total_ev = sum(s.expected_value for s in tradeable)
    total_size = sum(s.position_size_usd for s in tradeable)
    print(f"  Tradeable: {len(tradeable)} signals")
    print(f"  Total position: ${total_size:,.2f}")
    print(f"  Total expected value: ${total_ev:+,.2f}")
    print(f"{'='*70}")


def cmd_monitor(args):
    """Continuous monitoring loop."""
    from engines.weather_forecaster import WeatherForecaster
    from engines.prediction_scanner import KalshiScanner

    forecaster = WeatherForecaster()
    scanner = KalshiScanner(
        demo=not args.live,
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        max_position=args.max_position,
    )

    end_time = time.time() + args.duration * 3600
    cycle = 0

    mode = "LIVE" if args.live else "PAPER"
    print(f"\n{'='*70}")
    print(f"WEATHER BOT — Monitor [{mode}]")
    print(f"  Duration: {args.duration}h")
    print(f"  Interval: {args.interval}s")
    print(f"  Bankroll: ${args.bankroll:,.0f}")
    print(f"{'='*70}")

    try:
        while time.time() < end_time:
            cycle += 1
            now = datetime.now(timezone.utc)
            print(f"\n{'─'*70}")
            print(f"  Cycle #{cycle} — {now.strftime('%Y-%m-%d %H:%M UTC')}")

            markets = scanner.discover_weather_markets()
            if markets:
                signals = scanner.generate_signals(markets, forecaster)
                tradeable = [s for s in signals if s.is_tradeable]

                if tradeable:
                    print(f"\n  🎯 {len(tradeable)} TRADEABLE SIGNALS:")
                    for s in tradeable:
                        print(f"    {s.summary()}")
                else:
                    print(f"  No tradeable signals ({len(signals)} markets scanned)")
            else:
                print(f"  No markets found")

            # Sleep until next cycle
            wait = max(10, args.interval)
            next_scan = now + timedelta(seconds=wait)
            print(f"  Next scan: {next_scan.strftime('%H:%M:%S UTC')}")
            time.sleep(wait)

    except KeyboardInterrupt:
        print(f"\n  Monitor stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Weather Prediction Market Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common args
    parser.add_argument("--live", action="store_true",
                        help="Use live Kalshi API (default: demo)")
    parser.add_argument("--bankroll", type=float, default=500.0,
                        help="Total trading capital (default: $500)")
    parser.add_argument("--min-edge", type=float, default=0.08,
                        help="Minimum edge to trade (default: 8%%)")
    parser.add_argument("--max-position", type=float, default=100.0,
                        help="Maximum position per trade (default: $100)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all signals including non-tradeable")

    subs = parser.add_subparsers(dest="command", required=True)

    # forecast
    p_fc = subs.add_parser("forecast",
                           help="Weather forecast (no market data needed)")
    p_fc.add_argument("--city", type=str, default="new_york",
                      help="City to forecast (default: new_york)")
    p_fc.add_argument("--date", type=str, default=None,
                      help="Target date YYYY-MM-DD (default: tomorrow)")
    p_fc.add_argument("--all", action="store_true",
                      help="Forecast all configured cities")

    # scan
    subs.add_parser("scan", help="Scan markets and generate signals")

    # monitor
    p_mn = subs.add_parser("monitor", help="Continuous monitoring loop")
    p_mn.add_argument("--duration", type=float, default=24.0,
                      help="Duration in hours (default: 24)")
    p_mn.add_argument("--interval", type=float, default=300.0,
                      help="Seconds between scans (default: 300)")

    args = parser.parse_args()
    t0 = time.time()

    dispatch = {
        "forecast": cmd_forecast,
        "scan": cmd_scan,
        "monitor": cmd_monitor,
    }
    dispatch[args.command](args)

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
