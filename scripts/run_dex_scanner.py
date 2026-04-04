"""
scripts/run_dex_scanner.py — Cross-DEX Arbitrage Scanner CLI

Read-only scanner that monitors price differences across DEX venues
on Arbitrum L2. Collects data to validate the flash loan stat arb thesis.

Usage:
    # Quick scan (single pass, see what pools exist)
    python scripts/run_dex_scanner.py discover

    # Continuous scanning for 1 hour
    python scripts/run_dex_scanner.py scan --duration 60

    # Continuous scanning with custom tokens and interval
    python scripts/run_dex_scanner.py scan --tokens WETH,WBTC,USDC,ARB --interval 5 --duration 120

    # Analyze collected data for cointegration
    python scripts/run_dex_scanner.py analyze

    # Full pipeline: discover → scan → analyze
    python scripts/run_dex_scanner.py full --duration 60

Environment:
    ARBITRUM_RPC_URL — Arbitrum RPC endpoint (default: https://arb1.arbitrum.io/rpc)
                       For better rate limits, use Alchemy/Infura/QuickNode
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_discover(args):
    """Discover pools across DEX venues."""
    from engines.dex_scanner import DexScanner

    tokens = args.tokens.split(",") if args.tokens else None
    scanner = DexScanner(
        rpc_url=args.rpc_url,
        tokens=tokens,
        min_spread_bps=args.min_spread,
    )

    print(f"\n  RPC: {args.rpc_url}")
    if not scanner.connect():
        return

    t0 = time.time()
    n_pools = scanner.discover_pools()
    print(f"\n  Discovery took {time.time() - t0:.1f}s")

    # Do a single price scan to show current state
    if n_pools > 0:
        print(f"\n  Running single price scan...")
        opps = scanner.scan_once()
        if opps:
            print(f"\n  Opportunities found: {len(opps)}")
            for opp in opps[:20]:
                print(f"    {opp.token0}/{opp.token1}: "
                      f"{opp.buy_venue}→{opp.sell_venue} "
                      f"spread={opp.spread_bps:+.1f}bps "
                      f"net={opp.net_spread_bps:+.1f}bps "
                      f"est_profit={opp.estimated_profit_bps:+.1f}bps")
        else:
            print(f"\n  No opportunities above {args.min_spread} bps threshold.")
            print(f"  This is expected — arb opportunities are fleeting.")
            print(f"  Use 'scan' mode to monitor continuously.")


def cmd_scan(args):
    """Run continuous scanning loop."""
    from engines.dex_scanner import DexScanner

    tokens = args.tokens.split(",") if args.tokens else None
    scanner = DexScanner(
        rpc_url=args.rpc_url,
        tokens=tokens,
        gas_cost_bps=args.gas_cost,
        flash_loan_fee_bps=args.flash_fee,
        min_spread_bps=args.min_spread,
    )

    print(f"\n  RPC: {args.rpc_url}")
    if not scanner.connect():
        return

    print(f"\n  Discovering pools...")
    n_pools = scanner.discover_pools()
    if n_pools == 0:
        print("  No pools found. Check RPC connection and token list.")
        return

    output_dir = Path(args.output_dir)
    scanner.scan_loop(
        interval_seconds=args.interval,
        duration_minutes=args.duration,
        output_dir=output_dir,
    )


def cmd_analyze(args):
    """Analyze collected price data for cointegration."""
    from engines.dex_scanner import analyze_cointegration

    history_path = Path(args.output_dir) / "price_history.parquet"
    if not history_path.exists():
        print(f"  ERROR: No price history found at {history_path}")
        print(f"  Run 'scan' first to collect data.")
        return

    results = analyze_cointegration(
        history_path,
        min_observations=args.min_obs,
    )

    if not results.empty:
        output_path = Path(args.output_dir) / "cointegration_results.csv"
        results.to_csv(output_path, index=False)
        print(f"\n  Results saved: {output_path}")


def cmd_full(args):
    """Full pipeline: discover → scan → analyze."""
    cmd_scan(args)
    print(f"\n{'='*70}")
    print("Running cointegration analysis on collected data...")
    print(f"{'='*70}")
    cmd_analyze(args)


def cmd_quote(args):
    """Quote real executable depth for current opportunities."""
    from engines.dex_scanner import DexScanner
    from engines.dex_quoter import quote_scanner_opportunities

    tokens = args.tokens.split(",") if args.tokens else None
    scanner = DexScanner(
        rpc_url=args.rpc_url,
        tokens=tokens,
        min_spread_bps=args.min_spread,
    )

    print(f"\n  RPC: {args.rpc_url}")
    if not scanner.connect():
        return

    print(f"\n  Discovering pools...")
    n_pools = scanner.discover_pools()
    if n_pools == 0:
        print("  No pools found.")
        return

    # Parse trade sizes
    sizes = [float(s) for s in args.sizes.split(",")]

    # Parse pair filter
    pairs = args.pairs.split(",") if args.pairs else None

    results = quote_scanner_opportunities(
        scanner,
        trade_sizes_usd=sizes,
        gas_cost_usd=args.gas_cost_usd,
        pairs_filter=pairs,
        max_opps=args.max_opps,
    )

    # Save results
    if results:
        import pandas as pd
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_dir / "quotes.csv", index=False)
        print(f"\n  Results saved: {output_dir / 'quotes.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-DEX Arbitrage Scanner (Arbitrum L2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common args
    parser.add_argument("--rpc-url", type=str,
                        default=os.getenv("ARBITRUM_RPC_URL",
                                          "https://arb1.arbitrum.io/rpc"),
                        help="Arbitrum RPC endpoint")
    parser.add_argument("--tokens", type=str, default=None,
                        help="Comma-separated token symbols (default: major tokens)")
    parser.add_argument("--output-dir", type=str, default="output/dex_scanner",
                        help="Output directory for results")
    parser.add_argument("--min-spread", type=float, default=1.0,
                        help="Minimum net spread in bps to log as opportunity")
    parser.add_argument("--gas-cost", type=float, default=0.5,
                        help="Estimated gas cost in bps of trade size")
    parser.add_argument("--flash-fee", type=float, default=0.0,
                        help="Flash loan fee in bps (0 for Balancer)")

    subs = parser.add_subparsers(dest="command", required=True)

    # discover
    subs.add_parser("discover", help="Discover pools and do single price scan")

    # scan
    p_s = subs.add_parser("scan", help="Continuous scanning loop")
    p_s.add_argument("--interval", type=float, default=10.0,
                     help="Seconds between scans")
    p_s.add_argument("--duration", type=float, default=60.0,
                     help="Total scan duration in minutes")

    # analyze
    p_a = subs.add_parser("analyze", help="Analyze collected data")
    p_a.add_argument("--min-obs", type=int, default=100,
                     help="Min observations per pool for cointegration analysis")

    # full
    p_f = subs.add_parser("full", help="Discover → scan → analyze")
    p_f.add_argument("--interval", type=float, default=10.0)
    p_f.add_argument("--duration", type=float, default=60.0)
    p_f.add_argument("--min-obs", type=int, default=100)

    # quote
    p_q = subs.add_parser("quote", help="Quote real depth for opportunities")
    p_q.add_argument("--sizes", type=str, default="1000,5000,10000,50000",
                     help="Comma-separated trade sizes in USD")
    p_q.add_argument("--pairs", type=str, default=None,
                     help="Filter to specific pairs (e.g. USDC/WETH,USDCe/WETH)")
    p_q.add_argument("--gas-cost-usd", type=float, default=0.50,
                     help="Estimated gas cost in USD for arb tx")
    p_q.add_argument("--max-opps", type=int, default=20,
                     help="Max opportunities to quote")

    args = parser.parse_args()
    t0 = time.time()

    dispatch = {
        "discover": cmd_discover,
        "scan": cmd_scan,
        "analyze": cmd_analyze,
        "full": cmd_full,
        "quote": cmd_quote,
    }
    dispatch[args.command](args)

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
