"""
Cycle 34 -- one-shot info_bars backfill.

Scans crypto_data.trades for each (asset, bar_type, threshold)
slice in the configured set, builds closed bars, INSERTs them into
info_bars. Idempotent: PK conflict skips bars already present.

Default initial slice set is set per BRIEF_info_bars_v0_1.md:
- BTC + ETH x dollar bars at $1M and $5M
- BTC + ETH x volume bars at asset-appropriate base amounts
  (BTC: 100, 500 ; ETH: 1000, 5000)
- BTC + ETH x vib bars at $500k expected imbalance
- BTC + ETH x vrb bars at $500k expected run

Flags:
    --asset {BTC|ETH|all}      default all
    --bar-type {dollar|volume|vib|vrb|all}  default all
    --threshold-set <name>     default 'default' (only 'default' wired)
    --validate                 dry-run: counts only, no INSERTs
    --verbose                  per-trade-chunk logging in writer (unused)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from engines.info_bars.writer import backfill_slice  # noqa: E402

DEFAULT_DB = REPO / "data" / "crypto_data.db"


DEFAULT_THRESHOLDS = {
    "dollar": {
        "BTC": [1_000_000.0, 5_000_000.0],
        "ETH": [1_000_000.0, 5_000_000.0],
    },
    "volume": {
        "BTC": [100.0, 500.0],
        "ETH": [1000.0, 5000.0],
    },
    "vib": {
        "BTC": [500_000.0],
        "ETH": [500_000.0],
    },
    "vrb": {
        "BTC": [500_000.0],
        "ETH": [500_000.0],
    },
}


def _slices(asset_filter: str, bar_type_filter: str) -> list:
    out = []
    bar_types = (["dollar", "volume", "vib", "vrb"]
                 if bar_type_filter == "all" else [bar_type_filter])
    assets = ["BTC", "ETH"] if asset_filter == "all" else [asset_filter]
    for bt in bar_types:
        per_asset = DEFAULT_THRESHOLDS.get(bt, {})
        for asset in assets:
            for th in per_asset.get(asset, []):
                out.append((asset, bt, th))
    return out


def _fmt_th(v: float) -> str:
    if v >= 1_000_000 and v % 1_000_000 == 0:
        return f"${int(v // 1_000_000)}M"
    if v >= 1_000 and v % 1_000 == 0:
        return f"${int(v // 1_000)}k"
    return f"{v:g}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", choices=["BTC", "ETH", "all"], default="all")
    parser.add_argument("--bar-type",
                        choices=["dollar", "volume", "vib", "vrb", "all"],
                        default="all")
    parser.add_argument("--threshold-set", default="default",
                        help="only 'default' is wired in v0.1")
    parser.add_argument("--validate", action="store_true",
                        help="dry-run: count closed bars, do not INSERT")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args(argv)

    if args.threshold_set != "default":
        parser.error(f"unknown threshold-set {args.threshold_set!r}; "
                     f"only 'default' is wired in v0.1")

    slices = _slices(args.asset, args.bar_type)
    print(f"[backfill] slices to process: {len(slices)} "
          f"(validate_only={args.validate})")
    print(f"[backfill] db: {args.db}")
    print()

    header = f"  {'ASSET':<5} {'BAR':<7} {'THRESHOLD':<10} "
    header += f"{'TRADES':>12} {'BARS':>8} {'INSERTED':>9} "
    header += f"{'FIRST':>14} {'LAST':>14} {'ELAPSED':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_bars = 0
    total_inserted = 0
    total_trades = 0
    t_start = time.time()
    sub_50_warnings = []

    for asset, bar_type, threshold_value in slices:
        summary = backfill_slice(
            db_path=args.db,
            asset=asset,
            bar_type=bar_type,
            threshold_value=threshold_value,
            start_ts_ms=None,
            end_ts_ms=None,
            validate_only=args.validate,
        )
        n_bars = summary["closed_bars"]
        n_ins = summary["inserted"]
        n_tr = summary["trades_processed"]
        first = summary["first_bar_start"]
        last = summary["last_bar_end"]
        elapsed = summary["elapsed_s"]
        total_bars += n_bars
        total_inserted += n_ins
        total_trades += n_tr

        print(f"  {asset:<5} {bar_type:<7} {_fmt_th(threshold_value):<10} "
              f"{n_tr:>12,} {n_bars:>8,} {n_ins:>9,} "
              f"{str(first or '-'):>14} {str(last or '-'):>14} "
              f"{elapsed:>7.2f}s")

        if n_bars < 50:
            sub_50_warnings.append(
                f"{asset} {bar_type} th={_fmt_th(threshold_value)}: "
                f"only {n_bars} closed bars over the trade window")

    elapsed_total = time.time() - t_start
    print()
    print(f"[backfill] TOTAL: bars={total_bars:,} "
          f"inserted={total_inserted:,} trades_scanned={total_trades:,} "
          f"elapsed={elapsed_total:.2f}s")

    if sub_50_warnings:
        print()
        print("[backfill] WARNING: thresholds yielding <50 bars:")
        for w in sub_50_warnings:
            print(f"  - {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
