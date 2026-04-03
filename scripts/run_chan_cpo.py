"""
Burgess → Chan CPO Pipeline (unified).

Phases:
    phase1    — Pair discovery via Burgess cointegration (n=2)
    phase2    — Strategy grid search on minute bars → returns labels
    features  — Compute minute-frequency features (112 per Chan paper)
    train     — Train RF classifiers on features × returns
    pipeline  — features + train in one shot (most common)
    oos       — OOS portfolio trading with Kelly allocation
    ablation  — Regime ablation: which market dimensions predict profitability?
    diagnose  — Trace OOS per-model results for debugging

Usage:
    # Phase 1: Pair discovery (run once)
    python scripts/run_chan_cpo.py phase1 --universe sp500

    # Phase 2: Grid search on minute data (generates returns labels)
    python scripts/run_chan_cpo.py phase2 --pairs-json output/burgess/burgess_pairs.json

    # Compute features + train RF (the common case after Phase 2)
    python scripts/run_chan_cpo.py pipeline --pairs-json output/burgess/burgess_pairs.json

    # OOS trading
    python scripts/run_chan_cpo.py oos --pairs-json output/burgess/burgess_pairs.json

    # Regime ablation
    python scripts/run_chan_cpo.py ablation --pairs-json output/burgess/burgess_pairs.json
"""
from __future__ import annotations

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.pairs_trading import (
    PairSpec, ParamConfig,
    generate_param_grid, load_pairs_from_burgess,
    fetch_all_minute_data,
    run_phase2_returns,
    construct_minute_spread,
    run_intraday_single_day,
    compute_kelly_allocation,
)
from engines.cpo_training import (
    compute_features_v2,
    get_feature_columns,
    train_v2,
    predict_v2,
    run_phase2b_features,
    run_phase3_v2,
    run_ablation_experiment,
)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: PAIR DISCOVERY VIA BURGESS
# ═════════════════════════════════════════════════════════════════════════════

def cmd_phase1(args):
    """Run Burgess with n_vars=2 to discover cointegrated pairs."""
    from engines.burgess import BurgessParams, BurgessEngine

    print(f"{'='*70}")
    print("PHASE 1: Pair Discovery (Burgess n=2)")
    print(f"{'='*70}")

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    if args.universe == "sp500":
        tickers = _fetch_sp500()
    elif args.tickers:
        tickers = args.tickers
    else:
        raise ValueError("Provide --universe sp500 or --tickers")

    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=args.years * 365)
    start = start_dt.strftime("%Y-%m-%d")

    print(f"  Universe: {len(tickers)} tickers")
    print(f"  Period: {start} -> {end_date} ({args.years} years)")

    if not args.no_liquidity_filter:
        tickers = _liquidity_filter(tickers, api_key, args.min_adv, args.min_price)

    prices, valid_tickers, dates = _fetch_daily_polygon(tickers, start, end_date, api_key)
    print(f"\n  Loaded: {prices.shape[0]} days x {prices.shape[1]} assets")

    surface_path = Path(args.surface)
    if not surface_path.exists():
        raise FileNotFoundError(f"Surface DB not found: {surface_path}")

    params = BurgessParams(
        surface_db_path=str(surface_path),
        n_vars=1, train_frac=args.train_frac,
        wf_estimation_window=args.est_window,
        wf_signal_window=args.sig_window,
        wf_step_size=args.step_size,
        entry_threshold=args.entry_z, exit_threshold=args.exit_z,
        stop_loss_threshold=args.stop_z,
        transaction_cost_bps=args.tc_bps, slippage_bps=args.slippage_bps,
    )

    engine = BurgessEngine()
    output = engine.run(prices, valid_tickers, params, lambda p, m: print(f"  [{p}] {m}"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_burgess_results(output, valid_tickers, output_dir)

    pairs_data = _extract_top_pairs(output, valid_tickers, args.top_n)
    pairs_path = output_dir / "burgess_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs_data, f, indent=2)
    print(f"\n  Pairs saved: {pairs_path} ({len(pairs_data['ranked_baskets'])} pairs)")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: STRATEGY GRID SEARCH ON MINUTE BARS
# ═════════════════════════════════════════════════════════════════════════════

def cmd_phase2(args):
    """Run parameter grid search on minute data. Generates returns labels."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    print(f"  Loaded {len(pairs)} pairs from {args.pairs_json}")

    param_grid = generate_param_grid()
    print(f"  Parameter grid: {len(param_grid)} configs")

    cache_dir = Path(args.cache_dir)
    minute_data = fetch_all_minute_data(
        pairs, args.train_start, args.train_end, api_key, cache_dir
    )

    output_dir = Path(args.output_dir) / "chan_cpo"
    # run_phase2_returns generates both returns and old features —
    # we only use the returns. The old features parquet is ignored.
    returns_df, _ = run_phase2_returns(pairs, minute_data, param_grid, output_dir)

    print(f"\n  Phase 2 complete:")
    print(f"    Returns: {len(returns_df)} rows ({returns_df['pair_id'].nunique()} pairs × "
          f"{returns_df['config_id'].nunique()} configs × "
          f"{returns_df['date'].nunique()} days)")
    print(f"    Next: python scripts/run_chan_cpo.py pipeline "
          f"--pairs-json {args.pairs_json}")


# ═════════════════════════════════════════════════════════════════════════════
# FEATURES + TRAIN + PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def cmd_features(args):
    """Compute minute-frequency features from cached minute data."""
    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    returns_df = _load_returns(args.output_dir)
    minute_data = _load_minute_cache(pairs, args.train_start, args.train_end, args.cache_dir)
    if not minute_data:
        return

    output_dir = Path(args.output_dir) / "chan_cpo"
    features_df = run_phase2b_features(
        pairs, minute_data, returns_df, output_dir, mode=args.mode,
        bars_per_day=args.bars_per_day,
    )
    print(f"\n  Done. {len(features_df)} feature rows computed.")


def cmd_train(args):
    """Train RF models from pre-computed features."""
    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    returns_df = _load_returns(args.output_dir)

    output_dir = Path(args.output_dir) / "chan_cpo"
    suffix = args.mode.replace(":", "_")
    features_path = output_dir / f"phase2b_features_{suffix}.parquet"

    if not features_path.exists():
        print(f"ERROR: Features not found: {features_path}")
        print(f"  Run: python scripts/run_chan_cpo.py features --pairs-json {args.pairs_json}")
        return

    features_df = pd.read_parquet(features_path)
    print(f"  Loaded features: {features_df.shape}")

    models = run_phase3_v2(pairs, returns_df, features_df, output_dir, mode=args.mode)
    _save_models(models, output_dir, args.mode)


def cmd_pipeline(args):
    """Compute features + train RF in one shot."""
    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    returns_df = _load_returns(args.output_dir)
    minute_data = _load_minute_cache(pairs, args.train_start, args.train_end, args.cache_dir)
    if not minute_data:
        return

    output_dir = Path(args.output_dir) / "chan_cpo"

    features_df = run_phase2b_features(
        pairs, minute_data, returns_df, output_dir, mode=args.mode,
        bars_per_day=args.bars_per_day,
    )
    models = run_phase3_v2(pairs, returns_df, features_df, output_dir, mode=args.mode)
    _save_models(models, output_dir, args.mode)

    n_trained = sum(1 for m in models.values() if m.get("model"))
    aucs = [m["train_score"] for m in models.values() if m.get("model")]
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE (mode={args.mode})")
    print(f"  Models trained: {n_trained}/{len(pairs)}")
    if aucs:
        print(f"  AUC range: [{min(aucs):.4f}, {max(aucs):.4f}]")
        print(f"  AUC mean:  {np.mean(aucs):.4f}")
    print(f"{'='*70}")


# ═════════════════════════════════════════════════════════════════════════════
# OOS PORTFOLIO TRADING — with correct notional_capital + spread_history
# ═════════════════════════════════════════════════════════════════════════════

def cmd_oos(args):
    """Run OOS portfolio trading with Kelly allocation."""
    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    param_grid = generate_param_grid()

    output_dir = Path(args.output_dir) / "chan_cpo"
    models = _load_models(output_dir, args.mode)
    if models is None:
        return

    minute_data = _load_minute_cache(pairs, args.oos_start, args.oos_end, args.cache_dir)
    if not minute_data:
        return

    feature_cols = None
    for m in models.values():
        if m.get("model"):
            feature_cols = m["feature_cols"]
            break
    if not feature_cols:
        print("ERROR: No trained models found")
        return

    # Build trading days
    all_days = sorted(set().union(*(
        set(minute_data[t].index.normalize().unique())
        for t in minute_data
    )))
    trading_days = []
    for day in all_days:
        if day.weekday() >= 5:
            continue
        has_bars = any(
            len(minute_data[t][(minute_data[t].index >= day.replace(hour=9, minute=30)) &
                                (minute_data[t].index <= day.replace(hour=16, minute=0))]) >= 100
            for t in list(minute_data.keys())[:5]
        )
        if has_bars:
            trading_days.append(day)

    print(f"\n{'='*70}")
    print(f"OOS TRADING (mode={args.mode})")
    print(f"  Trading days: {len(trading_days)}")
    print(f"  Pairs: {len(pairs)}, Feature cols: {len(feature_cols)}")
    print(f"  Prob threshold: {args.prob_threshold}")
    print(f"{'='*70}")

    returns_history = []
    portfolio_pnl = []
    lookback_days = 10  # match Phase 2 training (run_pair_year uses 10)

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")
        day_start = day.replace(hour=9, minute=30)
        day_end = day.replace(hour=16, minute=0)

        # Step 1: Compute features and predict for each pair
        predictions = []
        for pair in pairs:
            model = models.get(pair.pair_id, {})
            if model.get("model") is None:
                continue

            feat_row = compute_features_v2(
                [pair], minute_data, [day_str],
                mode=args.mode, bars_per_day=args.bars_per_day,
            )
            if feat_row.empty:
                continue

            feat_vec = np.zeros(len(feature_cols))
            for i, col in enumerate(feature_cols):
                if col in feat_row.columns:
                    val = feat_row[col].values[0]
                    feat_vec[i] = float(val) if np.isfinite(val) else 0.0
            if np.count_nonzero(feat_vec) == 0:
                continue

            config, p_profitable, expected_ret = predict_v2(
                model, feat_vec, param_grid
            )
            predictions.append({
                "pair_id": pair.pair_id,
                "pair": pair,
                "config": config,
                "p_profitable": p_profitable,
                "expected_return": expected_ret,
            })

        if day_idx < 15 or day_idx % 10 == 0:
            n_above = sum(1 for p in predictions if p["p_profitable"] > args.prob_threshold)
            print(f"  Day {day_idx+1} ({day_str}): {len(predictions)} predictions, "
                  f"{n_above} above {args.prob_threshold}")

        if not predictions:
            continue

        # Step 2: Kelly allocation
        returns_hist_df = pd.DataFrame(returns_history) if returns_history else pd.DataFrame()
        allocation = compute_kelly_allocation(
            predictions, returns_hist_df,
            max_leverage=args.max_leverage,
            prob_threshold=args.prob_threshold,
        )

        # Step 3: Execute trades with correct notional and spread warmup
        day_pnl = 0.0
        n_executed = 0

        for pred in predictions:
            pid = pred["pair_id"]
            weight = allocation.get(pid, 0)
            if weight < 0.0001:
                continue

            pair = pred["pair"]
            config = pred["config"]
            target_bars = minute_data.get(pair.target)
            hedge_bars = minute_data.get(pair.hedge)
            if target_bars is None or hedge_bars is None:
                continue

            spread_df = construct_minute_spread(target_bars, hedge_bars, pair.hedge_ratio)
            spread = spread_df["spread"]
            target_close = spread_df["target_close"]
            hedge_close = spread_df["hedge_close"]

            spread_day = spread[(spread.index >= day_start) & (spread.index <= day_end)]
            if len(spread_day) < 30:
                continue

            # Notional capital from opening prices
            # Matches run_pair_year: target_open + |HR| * hedge_open
            tgt_day = target_close[(target_close.index >= day_start) & (target_close.index <= day_end)]
            hdg_day = hedge_close[(hedge_close.index >= day_start) & (hedge_close.index <= day_end)]
            if len(tgt_day) > 0 and len(hdg_day) > 0:
                notional = float(tgt_day.iloc[0]) + abs(pair.hedge_ratio) * float(hdg_day.iloc[0])
            else:
                notional = 1.0

            # Spread history for z-score warmup
            # Matches run_pair_year: previous lookback_days of minute spread
            hist_start = day - timedelta(days=lookback_days * 2)
            spread_hist = spread[(spread.index >= hist_start) & (spread.index < day_start)]

            result = run_intraday_single_day(
                spread_day, config,
                spread_history=spread_hist,
                notional_capital=notional,
            )
            ret = result["daily_return"]
            day_pnl += weight * ret
            n_executed += 1

            returns_history.append({
                "pair_id": pid, "date": day_str, "daily_return": ret,
            })

        portfolio_pnl.append({
            "date": day_str,
            "portfolio_return": day_pnl,
            "n_models_active": n_executed,
            "total_weight": sum(allocation.values()),
        })

    pnl_df = pd.DataFrame(portfolio_pnl)
    if pnl_df.empty:
        print("\n  No trading activity.")
        return

    suffix = args.mode.replace(":", "_")
    pnl_df.to_parquet(output_dir / f"phase4_{suffix}_portfolio.parquet")

    rets = pnl_df["portfolio_return"].values
    cum = np.cumsum(rets)
    daily_std = np.std(rets) + 1e-10
    sr = np.mean(rets) / daily_std * np.sqrt(252)
    max_dd = np.min(cum - np.maximum.accumulate(cum))
    win_days = (rets > 0).sum()

    print(f"\n{'='*70}")
    print(f"OOS RESULTS (mode={args.mode})")
    print(f"{'='*70}")
    print(f"  Trading days:    {len(rets)}")
    print(f"  Cumulative ret:  {cum[-1]*100:+.2f}%")
    print(f"  Sharpe ratio:    {sr:+.4f}")
    print(f"  Max drawdown:    {max_dd*100:+.2f}%")
    print(f"  Win days:        {win_days}/{len(rets)} ({win_days/len(rets):.1%})")
    print(f"  Avg models/day:  {pnl_df['n_models_active'].mean():.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# ABLATION
# ═════════════════════════════════════════════════════════════════════════════

def cmd_ablation(args):
    """Run regime ablation experiment."""
    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    returns_df = _load_returns(args.output_dir)
    minute_data = _load_minute_cache(pairs, args.train_start, args.train_end, args.cache_dir)
    if not minute_data:
        return

    output_dir = Path(args.output_dir) / "chan_cpo"
    run_ablation_experiment(
        pairs, minute_data, returns_df, output_dir,
        regime_classes=args.regime_classes,
        bars_per_day=args.bars_per_day,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSE
# ═════════════════════════════════════════════════════════════════════════════

def cmd_diagnose(args):
    """Trace OOS results per-model for debugging."""
    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    param_grid = generate_param_grid()
    output_dir = Path(args.output_dir) / "chan_cpo"

    models = _load_models(output_dir, args.mode)
    if models is None:
        return

    minute_data = _load_minute_cache(pairs, args.oos_start, args.oos_end, args.cache_dir)
    if not minute_data:
        return

    feature_cols = None
    for m in models.values():
        if m.get("model"):
            feature_cols = m["feature_cols"]
            break

    all_days = sorted(set().union(*(
        set(minute_data[t].index.normalize().unique()) for t in minute_data
    )))
    trading_days = [d for d in all_days if d.weekday() < 5]

    print(f"\nTrading days: {len(trading_days)}, Features: {len(feature_cols)}")

    lookback_days = 10
    days_with_predictions = 0
    rows = []

    print(f"\n{'='*90}")
    print(f"{'Day':>4} {'Date':>12} {'Pair':>12} {'LB':>5} {'EntZ':>5} "
          f"{'Trades':>6} {'Gross%':>8} {'TC%':>8} {'Net%':>8} {'P(prof)':>8}")
    print(f"{'='*90}")

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")
        day_start = day.replace(hour=9, minute=30)
        day_end = day.replace(hour=16, minute=0)

        preds = []
        for pair in pairs:
            model = models.get(pair.pair_id, {})
            if model.get("model") is None:
                continue
            feat_row = compute_features_v2(
                [pair], minute_data, [day_str],
                mode=args.mode, bars_per_day=args.bars_per_day,
            )
            if feat_row.empty:
                continue
            feat_vec = np.zeros(len(feature_cols))
            for i, col in enumerate(feature_cols):
                if col in feat_row.columns:
                    val = feat_row[col].values[0]
                    feat_vec[i] = float(val) if np.isfinite(val) else 0.0
            if np.count_nonzero(feat_vec) == 0:
                continue
            config, p_prof, _ = predict_v2(model, feat_vec, param_grid)
            preds.append({"pair": pair, "config": config, "p_profitable": p_prof})

        if not preds:
            continue

        days_with_predictions += 1
        if days_with_predictions > args.max_days:
            break

        for pred in preds:
            pair = pred["pair"]
            config = pred["config"]
            p_prof = pred["p_profitable"]

            target_bars = minute_data.get(pair.target)
            hedge_bars = minute_data.get(pair.hedge)
            if target_bars is None or hedge_bars is None:
                continue

            spread_df = construct_minute_spread(target_bars, hedge_bars, pair.hedge_ratio)
            spread = spread_df["spread"]
            target_close = spread_df["target_close"]
            hedge_close = spread_df["hedge_close"]

            spread_day = spread[(spread.index >= day_start) & (spread.index <= day_end)]
            if len(spread_day) < 30:
                continue

            tgt_day = target_close[(target_close.index >= day_start) & (target_close.index <= day_end)]
            hdg_day = hedge_close[(hedge_close.index >= day_start) & (hedge_close.index <= day_end)]
            notional = (float(tgt_day.iloc[0]) + abs(pair.hedge_ratio) * float(hdg_day.iloc[0])
                        if len(tgt_day) > 0 and len(hdg_day) > 0 else 1.0)

            hist_start = day - timedelta(days=lookback_days * 2)
            spread_hist = spread[(spread.index >= hist_start) & (spread.index < day_start)]

            result = run_intraday_single_day(
                spread_day, config,
                spread_history=spread_hist,
                notional_capital=notional,
            )

            gross_pct = result["gross_return"] * 100
            net_pct = result["daily_return"] * 100
            tc_pct = (result["total_costs"] / notional) * 100

            rows.append({
                "date": day_str, "pair_id": pair.pair_id,
                "lookback": config.lookback_minutes, "entry_z": config.entry_z,
                "n_trades": result["n_trades"],
                "n_spread_bars": len(spread_day), "n_hist_bars": len(spread_hist),
                "notional": notional,
                "gross_pct": gross_pct, "tc_pct": tc_pct, "net_pct": net_pct,
                "p_profitable": p_prof,
            })

            if result["n_trades"] > 0 or p_prof > 0.60:
                print(f"{day_idx+1:>4} {day_str:>12} {pair.pair_id:>12} "
                      f"{config.lookback_minutes:>5} {config.entry_z:>5.1f} "
                      f"{result['n_trades']:>6} {gross_pct:>+8.4f} "
                      f"{tc_pct:>8.4f} {net_pct:>+8.4f} {p_prof:>8.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo data.")
        return

    traded = df[df["n_trades"] > 0]
    print(f"\n{'='*90}")
    print(f"SUMMARY: {df['date'].nunique()} days, {len(df)} predictions")
    print(f"  Traded: {len(traded)}/{len(df)} ({len(traded)/len(df):.1%})")
    if not traded.empty:
        print(f"  Avg gross:  {traded['gross_pct'].mean():+.4f}%")
        print(f"  Avg TC:     {traded['tc_pct'].mean():.4f}%")
        print(f"  Avg net:    {traded['net_pct'].mean():+.4f}%")
        print(f"  Gross > 0:  {(traded['gross_pct'] > 0).sum()}/{len(traded)} "
              f"({(traded['gross_pct'] > 0).mean():.1%})")

    print(f"\nConfig selection by lookback:")
    for lb, grp in df.groupby("lookback"):
        print(f"  LB={lb:>5}: {len(grp):>4}x, trades={grp['n_trades'].mean():.1f}, "
              f"hist_bars={grp['n_hist_bars'].mean():.0f}, "
              f"gross={grp['gross_pct'].mean():+.4f}%, net={grp['net_pct'].mean():+.4f}%")

    detail_path = output_dir / "oos_diagnostic.csv"
    df.to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _load_returns(output_dir: str | Path) -> pd.DataFrame:
    path = Path(output_dir) / "chan_cpo" / "phase2_returns.parquet"
    if not path.exists():
        print(f"ERROR: Phase 2 returns not found: {path}")
        print(f"  Run: python scripts/run_chan_cpo.py phase2 --pairs-json <pairs.json>")
        sys.exit(1)
    df = pd.read_parquet(path)
    print(f"  Returns: {len(df)} rows ({df['pair_id'].nunique()} pairs × "
          f"{df['config_id'].nunique()} configs × {df['date'].nunique()} days)")
    return df


def _load_minute_cache(pairs, start, end, cache_dir) -> dict[str, pd.DataFrame]:
    tickers = sorted(set(t for p in pairs for t in [p.target, p.hedge]))
    cache_path = Path(cache_dir)
    print(f"  Minute cache: {cache_path} ({len(tickers)} tickers, {start} → {end})")

    data = {}
    missing = []
    for ticker in tickers:
        fp = cache_path / f"{ticker}_{start}_{end}_1min.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("US/Eastern").tz_localize(None)
            elif hasattr(df.index, 'tz_localize'):
                try:
                    df.index = (pd.DatetimeIndex(df.index).tz_localize("UTC")
                                .tz_convert("US/Eastern").tz_localize(None))
                except TypeError:
                    pass
            data[ticker] = df
        else:
            missing.append(ticker)

    print(f"  Loaded: {len(data)}/{len(tickers)} tickers")
    if missing:
        print(f"  MISSING: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
        print(f"  → Run phase2 to fetch from Polygon first")
        return {}
    return data


def _save_models(models, output_dir, mode):
    try:
        import joblib
        suffix = mode.replace(":", "_")
        path = output_dir / f"phase3_{suffix}_models.joblib"
        joblib.dump(models, path)
        print(f"  Models saved: {path}")
    except ImportError:
        print("  WARNING: joblib not available")


def _load_models(output_dir, mode) -> dict | None:
    try:
        import joblib
        suffix = mode.replace(":", "_")
        path = output_dir / f"phase3_{suffix}_models.joblib"
        models = joblib.load(path)
        n = sum(1 for m in models.values() if m.get("model"))
        print(f"  Loaded {n} models from {path}")
        return models
    except (ImportError, FileNotFoundError) as e:
        print(f"ERROR: Cannot load models: {e}")
        print(f"  Run: python scripts/run_chan_cpo.py pipeline ...")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# DATA HELPERS (Phase 1/2)
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_sp500() -> list[str]:
    print("  Fetching S&P 500 constituents...")
    import urllib.request
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    from io import StringIO
    tables = pd.read_html(StringIO(html))
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"  Found {len(tickers)} constituents")
    return tickers


def _liquidity_filter(tickers, api_key, min_adv, min_price):
    import requests
    print(f"\n  Liquidity filter: ADV >= {min_adv:,.0f}, price >= ${min_price}")
    passed, unavail, drop_v, drop_p = [], 0, 0, 0
    for i, ticker in enumerate(tickers):
        try:
            url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                   f"{(datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')}/"
                   f"{datetime.now().strftime('%Y-%m-%d')}"
                   f"?adjusted=true&sort=desc&limit=63&apiKey={api_key}")
            resp = requests.get(url, timeout=10)
            results = resp.json().get("results", [])
            if not results: unavail += 1; continue
            if np.mean([r["v"] for r in results]) < min_adv: drop_v += 1
            elif np.mean([r["c"] for r in results]) < min_price: drop_p += 1
            else: passed.append(ticker)
        except Exception: unavail += 1
        if (i + 1) % 100 == 0: print(f"    {i+1}/{len(tickers)} ({len(passed)} pass)")
        time.sleep(0.12)
    print(f"  Passed: {len(passed)}, dropped: vol={drop_v} price={drop_p} unavail={unavail}")
    return passed


def _fetch_daily_polygon(tickers, start, end, api_key):
    import requests
    print(f"\n  Polygon: fetching {len(tickers)} tickers...")
    all_series = {}
    for i, ticker in enumerate(tickers):
        try:
            url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                   f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")
            resp = requests.get(url, timeout=15)
            results = resp.json().get("results", [])
            if results:
                dates = [pd.Timestamp(r["t"], unit="ms").date() for r in results]
                all_series[ticker] = pd.Series([r["c"] for r in results], index=dates, name=ticker)
        except Exception: pass
        if (i + 1) % 50 == 0: print(f"    {i+1}/{len(tickers)} loaded")
        time.sleep(0.12)
    if not all_series: raise RuntimeError("No price data from Polygon")
    df = pd.DataFrame(all_series)
    valid = [t for t in tickers if t in df.columns and df[t].isna().mean() <= 0.10]
    df = df[valid].ffill().dropna()
    print(f"  Loaded: {df.shape[0]} days × {df.shape[1]} assets")
    return df.values.astype(np.float64), valid, [str(d) for d in df.index]


def _extract_top_pairs(output, asset_names, top_n=50):
    return {
        "timestamp": datetime.now().isoformat(),
        "n_vars": 2, "n_assets": output.n_assets, "n_obs": output.n_obs,
        "n_ranked": min(top_n, len(output.ranked_baskets)),
        "score_weights": output.score_weights,
        "ranked_baskets": [{
            "rank": b.rank, "target": asset_names[b.target_idx],
            "basket": [asset_names[i] for i in b.basket_indices],
            "hedge_ratio": b.regression.betas.tolist(),
            "composite_score": b.composite_score,
            "score_components": b.score_components,
            "adf_t": b.stationarity.adf_t_value, "adf_p": b.stationarity.adf_p_value,
            "hurst": b.stationarity.hurst_exponent, "half_life": b.stationarity.half_life,
        } for b in output.ranked_baskets[:top_n]],
    }


def _save_burgess_results(output, asset_names, output_dir):
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_assets": output.n_assets, "n_obs": output.n_obs,
        "n_ranked": len(output.ranked_baskets),
        "phase_times": output.phase_times,
        "ranked_baskets": [{
            "rank": b.rank, "target": asset_names[b.target_idx],
            "basket": [asset_names[i] for i in b.basket_indices],
            "hedge_ratio": b.regression.betas.tolist(),
            "composite_score": b.composite_score,
            "adf_t": b.stationarity.adf_t_value, "hurst": b.stationarity.hurst_exponent,
        } for b in output.ranked_baskets],
    }
    with open(output_dir / "burgess_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Burgess → Chan CPO Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--pairs-json", type=str, required=True)
        p.add_argument("--output-dir", type=str, default="output/burgess")
        p.add_argument("--cache-dir", type=str, default="data/minute_cache")
        p.add_argument("--top-n", type=int, default=50)
        p.add_argument("--train-start", type=str, default="2025-01-01")
        p.add_argument("--train-end", type=str, default="2025-12-31")
        p.add_argument("--bars-per-day", type=int, default=7)

    def add_mode(p):
        p.add_argument("--mode", type=str, default="chan", choices=["chan", "regime", "hybrid"])

    # phase1
    p1 = subs.add_parser("phase1", help="Pair discovery via Burgess")
    p1.add_argument("--output-dir", type=str, default="output/burgess")
    p1.add_argument("--top-n", type=int, default=50)
    g = p1.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers", nargs="+"); g.add_argument("--universe", choices=["sp500"])
    p1.add_argument("--years", type=int, default=2)
    p1.add_argument("--end-date", type=str, default="2024-12-31")
    p1.add_argument("--train-frac", type=float, default=0.667)
    p1.add_argument("--min-adv", type=float, default=500_000)
    p1.add_argument("--min-price", type=float, default=10.0)
    p1.add_argument("--no-liquidity-filter", action="store_true")
    p1.add_argument("--surface", type=str, default="data/surfaces.duckdb")
    p1.add_argument("--est-window", type=int, default=252)
    p1.add_argument("--sig-window", type=int, default=63)
    p1.add_argument("--step-size", type=int, default=21)
    p1.add_argument("--entry-z", type=float, default=2.0)
    p1.add_argument("--exit-z", type=float, default=0.5)
    p1.add_argument("--stop-z", type=float, default=4.0)
    p1.add_argument("--tc-bps", type=float, default=5.0)
    p1.add_argument("--slippage-bps", type=float, default=2.0)

    # phase2
    p2 = subs.add_parser("phase2", help="Grid search on minute data")
    p2.add_argument("--pairs-json", type=str, required=True)
    p2.add_argument("--output-dir", type=str, default="output/burgess")
    p2.add_argument("--cache-dir", type=str, default="data/minute_cache")
    p2.add_argument("--top-n", type=int, default=50)
    p2.add_argument("--train-start", type=str, default="2025-01-01")
    p2.add_argument("--train-end", type=str, default="2025-12-31")

    # features / train / pipeline
    for name, hlp in [("features", "Compute features"), ("train", "Train RF"), ("pipeline", "Features + train")]:
        p = subs.add_parser(name, help=hlp); add_common(p); add_mode(p)

    # oos
    p_o = subs.add_parser("oos", help="OOS portfolio trading")
    add_common(p_o); add_mode(p_o)
    p_o.add_argument("--oos-start", type=str, default="2026-01-01")
    p_o.add_argument("--oos-end", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    p_o.add_argument("--max-leverage", type=float, default=2.0)
    p_o.add_argument("--prob-threshold", type=float, default=0.65)

    # ablation
    p_a = subs.add_parser("ablation", help="Regime ablation experiment")
    add_common(p_a)
    p_a.add_argument("--regime-classes", type=str, default="ABCDEFGHIJK")

    # diagnose
    p_d = subs.add_parser("diagnose", help="Trace OOS per-model results")
    add_common(p_d); add_mode(p_d)
    p_d.add_argument("--oos-start", type=str, default="2026-01-01")
    p_d.add_argument("--oos-end", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    p_d.add_argument("--max-days", type=int, default=15)

    args = parser.parse_args()
    t0 = time.perf_counter()

    dispatch = {
        "phase1": cmd_phase1, "phase2": cmd_phase2,
        "features": cmd_features, "train": cmd_train,
        "pipeline": cmd_pipeline, "oos": cmd_oos,
        "ablation": cmd_ablation, "diagnose": cmd_diagnose,
    }
    dispatch[args.command](args)
    print(f"\n  Elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
