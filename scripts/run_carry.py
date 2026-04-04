"""
scripts/run_carry.py — Funding Rate Carry Strategy CLI

Integrates the funding monitor (signal generation) with the carry
executor (trade execution) for end-to-end carry trade management.

Usage:
    # Check current signals and portfolio status
    python scripts/run_carry.py status

    # Paper trade: enter carry on a specific asset
    python scripts/run_carry.py enter --asset ETH --notional 500 --hold 7

    # Paper trade: exit a position
    python scripts/run_carry.py exit --asset ETH

    # Auto mode: monitor signals, enter/exit automatically (paper)
    python scripts/run_carry.py auto --duration 24

    # Go live (real money!)
    python scripts/run_carry.py enter --asset ETH --notional 500 --hold 7 --live

    # Full auto with custom parameters
    python scripts/run_carry.py auto --duration 168 --gate 0.70 --notional 500 \\
        --max-positions 3 --max-total 2000

Environment:
    BINANCE_API_KEY     — Binance API key (required for --live)
    BINANCE_API_SECRET  — Binance API secret (required for --live)
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


def cmd_status(args):
    """Show current signals and portfolio status."""
    from engines.carry_executor import CarryExecutor
    from scripts.funding_monitor import (
        fetch_live_data, compute_live_features, run_inference,
        format_report, DEFAULT_ASSETS,
    )

    executor = CarryExecutor(
        paper=not args.live,
        max_position_usd=args.max_notional,
        max_total_exposure_usd=args.max_total,
    )

    # Portfolio status
    executor.status()

    # Current signals
    assets = args.assets.split(",") if args.assets else DEFAULT_ASSETS
    print(f"\n  Fetching signals for {assets}...")
    data = fetch_live_data(assets, cache_dir=args.cache_dir)
    features = compute_live_features(data, assets)
    signals = run_inference(features, args.models, assets, gate=args.gate)
    report = format_report(signals, args.gate, datetime.now(timezone.utc))
    print(report)


def cmd_enter(args):
    """Manually enter a carry position."""
    from engines.carry_executor import CarryExecutor

    executor = CarryExecutor(
        paper=not args.live,
        max_position_usd=args.max_notional,
        max_total_exposure_usd=args.max_total,
    )

    mode = "LIVE" if args.live else "PAPER"
    print(f"\n{'='*70}")
    print(f"CARRY ENTRY [{mode}]")
    print(f"{'='*70}")

    if args.live:
        print(f"\n  ⚠️  LIVE MODE — Real money will be used!")
        print(f"  Asset:    {args.asset}")
        print(f"  Notional: ${args.notional:,.2f}")
        print(f"  Hold:     {args.hold} days")
        confirm = input(f"\n  Type 'YES' to confirm: ")
        if confirm != "YES":
            print("  Cancelled.")
            return

    position = executor.enter_carry(
        asset=args.asset,
        notional_usd=args.notional,
        hold_days=args.hold,
        p_score=args.p_score,
    )

    if position:
        executor.status()


def cmd_exit(args):
    """Manually exit a carry position."""
    from engines.carry_executor import CarryExecutor

    executor = CarryExecutor(
        paper=not args.live,
        max_position_usd=args.max_notional,
        max_total_exposure_usd=args.max_total,
    )

    mode = "LIVE" if args.live else "PAPER"

    if args.live:
        print(f"\n  ⚠️  LIVE MODE — Closing real position!")
        print(f"  Asset: {args.asset}")
        confirm = input(f"  Type 'YES' to confirm: ")
        if confirm != "YES":
            print("  Cancelled.")
            return

    result = executor.exit_carry(args.asset, reason="manual")
    if result:
        executor.status()


def cmd_auto(args):
    """
    Automated carry management loop.

    Every 8 hours (at Binance funding windows):
    1. Update funding payments for open positions
    2. Check for expired positions and exit them
    3. Run signal monitor
    4. Enter new positions for assets above gate

    Between windows: sleep.
    """
    from engines.carry_executor import CarryExecutor
    from scripts.funding_monitor import (
        fetch_live_data, compute_live_features, run_inference,
        format_report, next_funding_window, DEFAULT_ASSETS,
    )

    mode = "LIVE" if args.live else "PAPER"
    assets = args.assets.split(",") if args.assets else DEFAULT_ASSETS

    executor = CarryExecutor(
        paper=not args.live,
        max_position_usd=args.notional,
        max_total_exposure_usd=args.max_total,
    )

    end_time = time.time() + args.duration * 3600
    cycle_count = 0

    print(f"\n{'='*70}")
    print(f"CARRY AUTO MODE [{mode}]")
    print(f"  Assets:     {', '.join(assets)}")
    print(f"  Gate:       P > {args.gate}")
    print(f"  Notional:   ${args.notional:,.0f} per position")
    print(f"  Max total:  ${args.max_total:,.0f}")
    print(f"  Duration:   {args.duration}h")
    print(f"  Hold:       {args.hold} days (RF may override)")
    print(f"{'='*70}")

    if args.live:
        print(f"\n  ⚠️  LIVE MODE — Real money will be used!")
        confirm = input(f"  Type 'YES' to start: ")
        if confirm != "YES":
            print("  Cancelled.")
            return

    try:
        while time.time() < end_time:
            cycle_count += 1
            now = datetime.now(timezone.utc)
            print(f"\n{'─'*70}")
            print(f"  Cycle #{cycle_count} — {now.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"{'─'*70}")

            # 1. Update funding for open positions
            if executor.positions:
                print(f"\n  Updating funding payments...")
                executor.update_funding()

            # 2. Check for exits
            if executor.positions:
                print(f"\n  Checking exit conditions...")
                exits = executor.check_exits()
                for ex in exits:
                    print(f"    Closed {ex['asset']}: ${ex['total_pnl']:+.2f}")

            # 3. Fetch signals
            print(f"\n  Fetching signals...")
            data = fetch_live_data(assets, cache_dir=args.cache_dir)
            features = compute_live_features(data, assets)
            signals = run_inference(features, args.models, assets, gate=args.gate)

            # Print report
            report = format_report(signals, args.gate, now)
            print(report)

            # 4. Enter new positions for assets above gate
            active_signals = [s for s in signals if s["above_gate"]]
            for sig in active_signals:
                asset = sig["asset"]
                if asset in executor.positions:
                    continue  # already have a position

                print(f"\n  🎯 Signal: {asset} P={sig['p_profitable']:.3f} > "
                      f"gate {args.gate}")
                hold = sig.get("hold_days", args.hold)

                executor.enter_carry(
                    asset=asset,
                    notional_usd=args.notional,
                    hold_days=hold,
                    p_score=sig["p_profitable"],
                )

            # Status
            executor.status()

            # 5. Sleep until next funding window
            next_window = next_funding_window(datetime.now(timezone.utc))
            sleep_seconds = (next_window - datetime.now(timezone.utc)).total_seconds()
            sleep_seconds = max(60, min(sleep_seconds, 8 * 3600))  # clamp

            print(f"\n  Next check: {next_window.strftime('%H:%M UTC')} "
                  f"(sleeping {sleep_seconds/60:.0f} min)")
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\n  Auto mode stopped by user.")

    # Final status
    executor.status()

    # Summary
    if executor.trade_log:
        entries = [t for t in executor.trade_log if t.action == "entry"]
        exits = [t for t in executor.trade_log if t.action == "exit"]
        print(f"\n  Session summary: {len(entries)//2} entries, "
              f"{len(exits)//2} exits, "
              f"{len(executor.positions)} still open")


def main():
    parser = argparse.ArgumentParser(
        description="Funding Rate Carry Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common args
    parser.add_argument("--live", action="store_true",
                        help="Use real money (default: paper trading)")
    parser.add_argument("--models", type=str,
                        default="output/funding_rate/cpo/phase3_models_funding.joblib",
                        help="Path to trained RF models")
    parser.add_argument("--gate", type=float, default=0.70,
                        help="P(profitable) gate threshold")
    parser.add_argument("--assets", type=str, default=None,
                        help="Comma-separated assets (default: BTC,ETH,SOL,XRP,ADA,AVAX)")
    parser.add_argument("--cache-dir", type=str, default="data/funding_cache",
                        help="Cache directory for market data")
    parser.add_argument("--max-notional", type=float, default=2000.0,
                        help="Max USD per position")
    parser.add_argument("--max-total", type=float, default=5000.0,
                        help="Max total exposure USD")

    subs = parser.add_subparsers(dest="command", required=True)

    # status
    subs.add_parser("status", help="Show signals and portfolio")

    # enter
    p_e = subs.add_parser("enter", help="Enter a carry position")
    p_e.add_argument("--asset", type=str, required=True,
                     help="Asset to trade (e.g. BTC)")
    p_e.add_argument("--notional", type=float, required=True,
                     help="USD per leg")
    p_e.add_argument("--hold", type=int, default=7,
                     help="Hold period in days")
    p_e.add_argument("--p-score", type=float, default=0.0,
                     help="RF P(profitable) at entry")

    # exit
    p_x = subs.add_parser("exit", help="Exit a carry position")
    p_x.add_argument("--asset", type=str, required=True,
                     help="Asset to close")

    # auto
    p_a = subs.add_parser("auto", help="Automated carry management")
    p_a.add_argument("--duration", type=float, default=168.0,
                     help="Duration in hours (default: 168 = 1 week)")
    p_a.add_argument("--notional", type=float, default=500.0,
                     help="USD per position")
    p_a.add_argument("--hold", type=int, default=7,
                     help="Default hold period if RF doesn't specify")

    args = parser.parse_args()
    t0 = time.time()

    dispatch = {
        "status": cmd_status,
        "enter": cmd_enter,
        "exit": cmd_exit,
        "auto": cmd_auto,
    }
    dispatch[args.command](args)

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
