"""
scripts/run_momentum.py — Momentum Signal Detector CLI

Detect short-term momentum signals for flash loan looping strategy.

Usage:
    # Backtest on recent 24h of BTC data
    python scripts/run_momentum.py backtest --symbol BTC/USDT --hours 24

    # Backtest multiple assets
    python scripts/run_momentum.py backtest --symbols BTC/USDT,ETH/USDT,ARB/USDT

    # Live paper trading monitor (watch signals in real time)
    python scripts/run_momentum.py monitor --symbols BTC/USDT,ETH/USDT --duration 60

    # Single snapshot of current signals
    python scripts/run_momentum.py scan --symbols BTC/USDT,ETH/USDT,SOL/USDT,ARB/USDT

    # Tune parameters
    python scripts/run_momentum.py backtest --symbol BTC/USDT --hours 48 \\
        --tp 40 --sl 25 --timeout 20 --min-score 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_symbols(args) -> list[str]:
    """Parse symbol arguments."""
    if hasattr(args, "symbols") and args.symbols:
        return args.symbols.split(",")
    if hasattr(args, "symbol") and args.symbol:
        return [args.symbol]
    return ["BTC/USDT", "ETH/USDT"]


def make_params(args):
    """Build SignalParams from CLI args."""
    from engines.momentum_signals import SignalParams

    params = SignalParams()
    if hasattr(args, "tp") and args.tp is not None:
        params.take_profit_bps = args.tp
    if hasattr(args, "sl") and args.sl is not None:
        params.stop_loss_bps = args.sl
    if hasattr(args, "timeout") and args.timeout is not None:
        params.max_hold_minutes = args.timeout
    if hasattr(args, "min_score") and args.min_score is not None:
        params.min_composite_score = args.min_score
    if hasattr(args, "vol_spike") and args.vol_spike is not None:
        params.vol_spike_threshold = args.vol_spike
    if hasattr(args, "velocity") and args.velocity is not None:
        params.velocity_threshold_bps = args.velocity
    if hasattr(args, "consec") and args.consec is not None:
        params.consec_min = args.consec
    return params


def cmd_scan(args):
    """Single snapshot of current signals across assets."""
    from engines.momentum_signals import MomentumDetector

    symbols = parse_symbols(args)
    params = make_params(args)
    detector = MomentumDetector(params, args.exchange)

    print(f"\n{'='*80}")
    print(f"MOMENTUM SCAN — {len(symbols)} assets")
    print(f"{'='*80}")

    signals = detector.scan_assets(symbols, lookback_minutes=120)

    for sig in signals:
        actionable = "  ★ ACTIONABLE" if sig.is_actionable else ""
        print(f"\n  {sig.summary()}{actionable}")

        # Price context
        chg_5m = (sig.current_price - sig.price_5m_ago) / sig.price_5m_ago * 10000
        chg_15m = (sig.current_price - sig.price_15m_ago) / sig.price_15m_ago * 10000
        print(f"    Δ5min={chg_5m:+.1f}bps  Δ15min={chg_15m:+.1f}bps")

        # Individual signals
        for s in sig.signals:
            dir_icon = {"1": "▲", "-1": "▼", "0": "─"}[str(s.direction)]
            strength_bar = "█" * int(s.strength * 5) + "░" * (5 - int(s.strength * 5))
            print(f"    {dir_icon} {s.name:15s} [{strength_bar}] {s.description}")


def cmd_backtest(args):
    """Backtest signals on historical data."""
    from engines.momentum_signals import backtest_signals

    symbols = parse_symbols(args)
    params = make_params(args)

    print(f"\n{'='*80}")
    print(f"MOMENTUM BACKTEST")
    print(f"  Assets: {', '.join(symbols)}")
    print(f"  Period: {args.hours}h")
    print(f"  TP={params.take_profit_bps}bps  SL={params.stop_loss_bps}bps  "
          f"timeout={params.max_hold_minutes}min  min_score={params.min_composite_score}")
    print(f"{'='*80}")

    all_trades = []
    for symbol in symbols:
        trades = backtest_signals(
            symbol=symbol,
            lookback_hours=args.hours,
            params=params,
            exchange=args.exchange,
        )
        all_trades.extend(trades)

    # Combined summary
    if len(symbols) > 1 and all_trades:
        print(f"\n{'='*80}")
        print(f"COMBINED RESULTS ({len(symbols)} assets)")
        from engines.momentum_signals import _print_trade_summary
        _print_trade_summary(all_trades)

    # Save results
    if all_trades:
        import pandas as pd
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_trades).to_csv(output_dir / "backtest_trades.csv",
                                        index=False)
        print(f"\n  Trades saved: {output_dir / 'backtest_trades.csv'}")


def cmd_monitor(args):
    """Live paper trading monitor."""
    from engines.momentum_signals import MomentumDetector

    symbols = parse_symbols(args)
    params = make_params(args)
    detector = MomentumDetector(params, args.exchange)

    trade_log = detector.monitor_loop(
        symbols=symbols,
        interval_seconds=args.interval,
        duration_minutes=args.duration,
        lookback_minutes=120,
    )

    # Save results
    if trade_log:
        import pandas as pd
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trade_log).to_csv(output_dir / "paper_trades.csv",
                                       index=False)
        print(f"\n  Trades saved: {output_dir / 'paper_trades.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Momentum Signal Detector for Flash Loan Looping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common args
    parser.add_argument("--exchange", type=str, default="binance",
                        help="Exchange for market data (default: binance)")
    parser.add_argument("--output-dir", type=str, default="output/momentum",
                        help="Output directory for results")

    # Signal tuning
    parser.add_argument("--tp", type=float, default=None,
                        help="Take profit in bps (default: 50)")
    parser.add_argument("--sl", type=float, default=None,
                        help="Stop loss in bps (default: 30)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max hold time in minutes (default: 30)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Min composite score to trigger (default: 0.6)")
    parser.add_argument("--vol-spike", type=float, default=None,
                        help="Volume spike threshold (default: 2.5)")
    parser.add_argument("--velocity", type=float, default=None,
                        help="Price velocity threshold in bps (default: 15)")
    parser.add_argument("--consec", type=int, default=None,
                        help="Min consecutive candles (default: 3)")

    subs = parser.add_subparsers(dest="command", required=True)

    # scan
    p_sc = subs.add_parser("scan", help="Single snapshot of current signals")
    p_sc.add_argument("--symbols", type=str,
                      default="BTC/USDT,ETH/USDT,SOL/USDT,ARB/USDT",
                      help="Comma-separated trading pairs")

    # backtest
    p_bt = subs.add_parser("backtest", help="Backtest on historical data")
    p_bt.add_argument("--symbol", type=str, default=None,
                      help="Single symbol to backtest")
    p_bt.add_argument("--symbols", type=str, default=None,
                      help="Comma-separated symbols to backtest")
    p_bt.add_argument("--hours", type=int, default=24,
                      help="Hours of history to backtest on")

    # monitor
    p_mn = subs.add_parser("monitor",
                           help="Live paper trading monitor")
    p_mn.add_argument("--symbols", type=str,
                      default="BTC/USDT,ETH/USDT",
                      help="Comma-separated trading pairs")
    p_mn.add_argument("--interval", type=float, default=60.0,
                      help="Seconds between scans (default: 60)")
    p_mn.add_argument("--duration", type=float, default=60.0,
                      help="Monitor duration in minutes (default: 60)")

    args = parser.parse_args()
    t0 = time.time()

    dispatch = {
        "scan": cmd_scan,
        "backtest": cmd_backtest,
        "monitor": cmd_monitor,
    }
    dispatch[args.command](args)

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
