"""
Runner: Generic CPO Pipeline.

Usage:
    # Pairs strategy (Burgess → Chan CPO)
    python scripts/run_cpo.py --strategy pairs --pairs-json output/burgess/burgess_pairs.json \
        phase2 --start 2025-01-01 --end 2025-12-31
    python scripts/run_cpo.py --strategy pairs --pairs-json output/burgess/burgess_pairs.json phase3
    python scripts/run_cpo.py --strategy pairs --pairs-json output/burgess/burgess_pairs.json \
        phase4 --start 2026-01-01

    # Future: crypto TA strategy
    python scripts/run_cpo.py --strategy crypto_ta --config crypto_config.yaml phase2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.cpo_core import run_phase2, run_phase3, run_phase4


def build_strategy(args):
    """Factory: build the appropriate strategy from CLI args."""
    api_key = os.getenv("POLYGON_API_KEY", "")

    if args.strategy == "pairs":
        if not api_key:
            raise RuntimeError("POLYGON_API_KEY not set in .env (required for pairs strategy)")
        from engines.pairs_strategy import PairsStrategy
        return PairsStrategy(
            pairs_json=args.pairs_json,
            api_key=api_key,
            cache_dir=args.cache_dir,
            top_n=args.top_n,
            tc_bps=args.tc_bps,
            training_start=args.training_start,
            training_end=args.training_end,
        )
    elif args.strategy == "crypto_ta":
        from engines.crypto_ta_strategy import CryptoTAStrategy
        assets = args.assets.split(",") if args.assets else None
        return CryptoTAStrategy(
            assets=assets,
            cache_dir=args.cache_dir,
            tc_bps=args.tc_bps,
            training_start=args.training_start,
            training_end=args.training_end,
        )
    elif args.strategy in ("futures_ta", "fx_ta", "universal_ta"):
        from engines.universal_ta_strategy import UniversalTAStrategy
        # Map strategy name to asset class
        ac_map = {"futures_ta": "futures", "fx_ta": "fx", "universal_ta": args.asset_class}
        ac = ac_map[args.strategy]
        assets = args.assets.split(",") if args.assets else None
        return UniversalTAStrategy(
            asset_class=ac,
            assets=assets,
            cache_dir=args.cache_dir,
            training_start=args.training_start,
            training_end=args.training_end,
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}. "
                         f"Available: pairs, crypto_ta, futures_ta, fx_ta, universal_ta")


def cmd_phase2(args):
    strategy = build_strategy(args)
    output_dir = Path(args.output_dir) / "cpo"
    returns_df, features_df = run_phase2(strategy, output_dir)
    print(f"\n  Phase 2 complete:")
    print(f"    Returns: {len(returns_df)} rows")
    print(f"    Features: {len(features_df)} rows")


def cmd_phase3(args):
    strategy = build_strategy(args)
    output_dir = Path(args.output_dir) / "cpo"

    returns_df = pd.read_parquet(output_dir / "phase2_returns.parquet")
    features_df = pd.read_parquet(output_dir / "phase2_features.parquet")
    print(f"  Loaded Phase 2 data: {len(returns_df)} returns, {len(features_df)} features")

    # Use strategy's model_group_fn if available (for pre-filtering)
    group_fn = getattr(strategy, 'model_group_fn', None)
    models = run_phase3(strategy, returns_df, features_df, output_dir,
                        model_group_fn=group_fn)

    n_trained = sum(1 for m in models.values() if m.get("model"))
    print(f"\n  Phase 3 complete: {n_trained} models trained")

    try:
        import joblib
        joblib.dump(models, output_dir / "phase3_models.joblib")
        print(f"  Models saved: {output_dir / 'phase3_models.joblib'}")
    except ImportError:
        print("  WARNING: joblib not available")


def cmd_phase4(args):
    strategy = build_strategy(args)
    output_dir = Path(args.output_dir) / "cpo"

    try:
        import joblib
        models = joblib.load(output_dir / "phase3_models.joblib")
        for mid, m in models.items():
            if m.get("model") is not None:
                m["model"].n_jobs = 1
        n_loaded = sum(1 for m in models.values() if m.get("model"))
        print(f"  Loaded {n_loaded} models")
    except (ImportError, FileNotFoundError):
        print("  ERROR: phase3_models.joblib not found. Run phase3 first.")
        return

    # Fetch warmup daily data
    warmup_daily = None
    oos_start = pd.Timestamp(args.oos_start)
    warmup_start = oos_start - timedelta(days=120)
    print(f"\n  Fetching warmup: {warmup_start.strftime('%Y-%m-%d')} -> {args.oos_start}")
    try:
        warmup_daily = strategy.fetch_warmup_daily(
            strategy.get_models(),
            warmup_start.strftime("%Y-%m-%d"),
            args.oos_start,
        )
        avg_days = np.mean([len(v) for v in warmup_daily.values()])
        print(f"  Warmup: {len(warmup_daily)} tickers, ~{avg_days:.0f} days each")
    except Exception as e:
        print(f"  WARNING: warmup failed ({e})")

    # Fetch OOS data
    oos_end = args.oos_end or datetime.now().strftime("%Y-%m-%d")
    oos_data = strategy.fetch_oos_data(strategy.get_models(), args.oos_start, oos_end)

    # Merge warmup into daily prices
    if warmup_daily:
        daily_prices = strategy.get_daily_prices(oos_data, strategy.get_models())
        daily_prices = strategy.prepare_warmup(daily_prices, warmup_daily)
        oos_data["daily_prices"] = daily_prices

    pnl_df = run_phase4(
        strategy, models, oos_data, output_dir,
        max_leverage=args.max_leverage,
        max_weight_per_model=args.max_weight,
        prob_threshold=args.prob_threshold,
        min_lift=args.min_lift,
        allocation_mode=args.alloc_mode,
        corr_threshold=args.corr_threshold,
        warmup_daily=warmup_daily,
    )

    # Final report
    if not pnl_df.empty:
        rets = pnl_df["portfolio_return"].values
        cum = np.cumsum(rets)
        n_days = len(rets)
        daily_mean = np.mean(rets)
        daily_std = np.std(rets) + 1e-10
        sr = daily_mean / daily_std * np.sqrt(252)
        ann_ret = daily_mean * 252
        ann_vol = daily_std * np.sqrt(252)
        max_dd = np.min(cum - np.maximum.accumulate(cum))
        win_days = (rets > 0).sum()

        print(f"\n{'='*70}")
        print("FINAL OOS PORTFOLIO RESULTS")
        print(f"{'='*70}")
        print(f"  Period:          {args.oos_start} -> {oos_end}")
        print(f"  Trading days:    {n_days}")
        print(f"  Cumulative ret:  {cum[-1]:+.4f}  ({cum[-1]*100:+.2f}%)")
        print(f"  Ann. return:     {ann_ret:+.4f}  ({ann_ret*100:+.1f}%)")
        print(f"  Ann. volatility: {ann_vol:.4f}   ({ann_vol*100:.1f}%)")
        print(f"  Sharpe ratio:    {sr:+.4f}")
        print(f"  Max drawdown:    {max_dd:+.4f}  ({max_dd*100:+.2f}%)")
        print(f"  Win days:        {win_days}/{n_days} ({win_days/n_days:.1%})")
        print(f"  Avg models/day:  {pnl_df['n_models_active'].mean():.1f}")
        print(f"  Max models/day:  {pnl_df['n_models_active'].max()}")


def cmd_full(args):
    t0 = time.perf_counter()
    cmd_phase2(args)
    cmd_phase3(args)
    args.oos_start = args.oos_start or "2026-01-01"
    cmd_phase4(args)
    print(f"\n  Total time: {time.perf_counter() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="CPO Pipeline — Generic Runner")
    parser.add_argument("--strategy", required=True,
                        choices=["pairs", "crypto_ta", "futures_ta", "fx_ta", "universal_ta"],
                        help="Trading strategy to use")
    parser.add_argument("--asset-class", default="crypto",
                        choices=["crypto", "futures", "fx"],
                        help="Asset class (universal_ta strategy only)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", default="data/minute_cache")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--tc-bps", type=float, default=2.0)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=0.05,
                        help="Max weight per model (default 0.05 = 5%%)")
    parser.add_argument("--prob-threshold", type=float, default=0.50,
                        help="P(profitable) gate threshold (default 0.50)")
    parser.add_argument("--min-lift", type=float, default=0.0,
                        help="Lift-based gate: P must exceed base_rate + min_lift. "
                             "Overrides prob-threshold when > 0. (default 0.0 = use prob-threshold)")
    parser.add_argument("--alloc-mode", default="equal_weight",
                        choices=["equal_weight", "kelly"],
                        help="Allocation mode: equal_weight (crypto/short-lived) or kelly (pairs/persistent)")
    parser.add_argument("--corr-threshold", type=float, default=0.85,
                        help="Correlation dedup threshold (kelly mode only)")
    parser.add_argument("--training-start", default="2025-01-01")
    parser.add_argument("--training-end", default="2025-12-31")

    # Strategy-specific args
    parser.add_argument("--pairs-json", type=str, default=None,
                        help="Path to Burgess pairs JSON (pairs strategy)")
    parser.add_argument("--assets", type=str, default=None,
                        help="Comma-separated asset names to trade (default: all in asset class)")

    subparsers = parser.add_subparsers(dest="phase", required=True)

    p2 = subparsers.add_parser("phase2", help="Parameter grid search")
    p2.add_argument("--start", dest="training_start_override", default=None)
    p2.add_argument("--end", dest="training_end_override", default=None)

    p3 = subparsers.add_parser("phase3", help="Train RF models")

    p4 = subparsers.add_parser("phase4", help="OOS portfolio trading")
    p4.add_argument("--start", dest="oos_start", default="2026-01-01")
    p4.add_argument("--end", dest="oos_end", default=None)

    pf = subparsers.add_parser("full", help="Run all phases")
    pf.add_argument("--oos-start", default="2026-01-01")
    pf.add_argument("--oos-end", default=None)

    args = parser.parse_args()

    # Override training dates if provided
    if hasattr(args, 'training_start_override') and args.training_start_override:
        args.training_start = args.training_start_override
    if hasattr(args, 'training_end_override') and args.training_end_override:
        args.training_end = args.training_end_override

    if args.strategy == "pairs" and not args.pairs_json:
        parser.error("--pairs-json required for pairs strategy")
    if args.strategy == "crypto_ta":
        if args.output_dir is None:
            args.output_dir = Path("output/crypto_ta")
    elif args.strategy == "futures_ta":
        if args.output_dir is None:
            args.output_dir = Path("output/futures_ta")
    elif args.strategy == "fx_ta":
        if args.output_dir is None:
            args.output_dir = Path("output/fx_ta")
    elif args.strategy == "universal_ta":
        if args.output_dir is None:
            args.output_dir = Path(f"output/{args.asset_class}_ta")
    elif args.output_dir is None:
        args.output_dir = Path("output/burgess")

    if args.phase == "phase2":
        cmd_phase2(args)
    elif args.phase == "phase3":
        cmd_phase3(args)
    elif args.phase == "phase4":
        cmd_phase4(args)
    elif args.phase == "full":
        cmd_full(args)


if __name__ == "__main__":
    main()
