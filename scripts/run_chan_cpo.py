"""
Runner: Burgess → Chan CPO Pipeline.

Usage:
    # Phase 1: Pair discovery (run once)
    python scripts/run_chan_cpo.py phase1 --universe sp500 --years 2 --end-date 2024-12-31

    # Phase 2: Param grid on minute data (training year)
    python scripts/run_chan_cpo.py phase2 --pairs-json output/burgess/burgess_pairs.json \
        --start 2025-01-01 --end 2025-12-31

    # Phase 3: Train Random Forests
    python scripts/run_chan_cpo.py phase3 --pairs-json output/burgess/burgess_pairs.json

    # Phase 4: OOS portfolio trading
    python scripts/run_chan_cpo.py phase4 --pairs-json output/burgess/burgess_pairs.json \
        --start 2026-01-01 --end 2026-03-01

    # Full pipeline (all phases)
    python scripts/run_chan_cpo.py full --universe sp500
"""
from __future__ import annotations

import os
os.environ["PYTHONWARNINGS"] = "ignore"  # propagates to joblib workers

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.chan_cpo import (
    PairSpec, ParamConfig,
    generate_param_grid, load_pairs_from_burgess,
    fetch_all_minute_data,
    run_phase2_training, run_phase3_training, run_phase4_oos,
)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: PAIR DISCOVERY VIA BURGESS (n=2)
# ═════════════════════════════════════════════════════════════════════════════

def run_phase1(args):
    """
    Run Burgess with n_vars=2 to discover cointegrated pairs.
    Saves top 50 pairs with hedge ratios.
    """
    from engines.burgess import BurgessParams, BurgessEngine

    print("=" * 70)
    print("PHASE 1: Pair Discovery (Burgess n=2)")
    print("=" * 70)

    # ── Resolve tickers ───────────────────────────────────────────────
    if args.universe == "sp500":
        tickers = _fetch_sp500()
    elif args.tickers:
        tickers = args.tickers
    else:
        raise ValueError("Provide --universe sp500 or --tickers")

    # ── Date range ────────────────────────────────────────────────────
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=args.years * 365)
    start = start_dt.strftime("%Y-%m-%d")

    print(f"  Universe: {len(tickers)} tickers")
    print(f"  Period: {start} -> {end_date} ({args.years} years)")
    print(f"  Basket size: n=2 (pairs)")

    # ── Liquidity filter ──────────────────────────────────────────────
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    if not args.no_liquidity_filter:
        tickers = _liquidity_filter(tickers, api_key, args.min_adv, args.min_price)

    # ── Fetch daily data ──────────────────────────────────────────────
    prices, valid_tickers, dates = _fetch_daily_polygon(
        tickers, start, end_date, api_key
    )

    print(f"\n  Loaded: {prices.shape[0]} days x {prices.shape[1]} assets")

    # ── Run Burgess with n_vars=2 ─────────────────────────────────────
    surface_path = Path(args.surface)
    if not surface_path.exists():
        raise FileNotFoundError(f"Surface DB not found: {surface_path}")

    params = BurgessParams(
        surface_db_path=str(surface_path),
        n_vars=1,  # PAIRS: target + 1 hedge = 2 total
        train_frac=args.train_frac,
        wf_estimation_window=args.est_window,
        wf_signal_window=args.sig_window,
        wf_step_size=args.step_size,
        entry_threshold=args.entry_z,
        exit_threshold=args.exit_z,
        stop_loss_threshold=args.stop_z,
        transaction_cost_bps=args.tc_bps,
        slippage_bps=args.slippage_bps,
    )

    def _progress(phase, msg):
        print(f"  [{phase}] {msg}")

    engine = BurgessEngine()
    output = engine.run(prices, valid_tickers, params, _progress)

    # ── Save results ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full Burgess output
    _save_burgess_results(output, valid_tickers, output_dir)

    # Pairs-specific output for Phase 2+
    pairs_data = _extract_top_pairs(output, valid_tickers, args.top_n)
    pairs_path = output_dir / "burgess_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs_data, f, indent=2)
    print(f"\n  Pairs saved: {pairs_path} ({len(pairs_data['ranked_baskets'])} pairs)")

    return pairs_path


def _extract_top_pairs(output, asset_names, top_n=50):
    """Extract top pairs with hedge ratios for Chan CPO."""
    ranked = output.ranked_baskets[:top_n]
    baskets = []
    for b in ranked:
        baskets.append({
            "rank": b.rank,
            "target": asset_names[b.target_idx],
            "basket": [asset_names[i] for i in b.basket_indices],
            "hedge_ratio": b.regression.betas.tolist(),
            "composite_score": b.composite_score,
            "score_components": b.score_components,
            "adf_t": b.stationarity.adf_t_value,
            "adf_p": b.stationarity.adf_p_value,
            "hurst": b.stationarity.hurst_exponent,
            "half_life": b.stationarity.half_life,
        })
    return {
        "timestamp": datetime.now().isoformat(),
        "n_vars": 2,
        "n_assets": output.n_assets,
        "n_obs": output.n_obs,
        "n_ranked": len(baskets),
        "score_weights": output.score_weights,
        "ranked_baskets": baskets,
    }


def _save_burgess_results(output, asset_names, output_dir):
    """Save full Burgess output (same as run_burgess.py)."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_assets": output.n_assets, "n_obs": output.n_obs,
        "n_train": output.n_train, "n_test": output.n_test,
        "n_ranked": len(output.ranked_baskets),
        "n_oos_passed": len(output.oos_passed_baskets),
        "n_backtested": len(output.backtest_results),
        "phase_times": output.phase_times,
        "score_weights": output.score_weights,
        "ranked_baskets": [
            {"rank": b.rank, "target": asset_names[b.target_idx],
             "basket": [asset_names[i] for i in b.basket_indices],
             "hedge_ratio": b.regression.betas.tolist(),
             "composite_score": b.composite_score,
             "score_components": b.score_components,
             "adf_t": b.stationarity.adf_t_value,
             "adf_p": b.stationarity.adf_p_value,
             "hurst": b.stationarity.hurst_exponent,
             "half_life": b.stationarity.half_life}
            for b in output.ranked_baskets
        ],
        "performance": [
            {"target": p.target_name, "basket": p.basket_names,
             "sharpe": p.sharpe_ratio, "sortino": p.sortino_ratio,
             "calmar": p.calmar_ratio, "annual_return": p.annual_return,
             "annual_vol": p.annual_volatility, "max_drawdown": p.max_drawdown,
             "win_rate": p.win_rate, "profit_factor": p.profit_factor,
             "n_trades": p.n_trades, "avg_hold_days": p.avg_hold_days,
             "composite_score": p.composite_score,
             "score_components": p.score_components}
            for p in output.performance
        ],
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = output_dir / f"burgess_n2_{ts}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Full Burgess results: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2-4 RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def run_phase2(args):
    """Run parameter grid search on minute data."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    print(f"  Loaded {len(pairs)} pairs from {args.pairs_json}")

    param_grid = generate_param_grid()
    print(f"  Parameter grid: {len(param_grid)} configs")

    cache_dir = Path(args.cache_dir)
    minute_data = fetch_all_minute_data(
        pairs, args.start, args.end, api_key, cache_dir
    )

    output_dir = Path(args.output_dir) / "chan_cpo"
    returns_df, features_df = run_phase2_training(
        pairs, minute_data, param_grid, output_dir
    )

    print(f"\n  Phase 2 complete:")
    print(f"    Returns: {len(returns_df)} rows ({returns_df['pair_id'].nunique()} pairs x "
          f"{returns_df['config_id'].nunique()} configs x "
          f"{returns_df['date'].nunique()} days)")
    print(f"    Features: {len(features_df)} rows")


def run_phase3(args):
    """Train Random Forest models."""
    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)

    output_dir = Path(args.output_dir) / "chan_cpo"
    returns_df = pd.read_parquet(output_dir / "phase2_returns.parquet")
    features_df = pd.read_parquet(output_dir / "phase2_features.parquet")

    print(f"  Loaded Phase 2 data: {len(returns_df)} returns, {len(features_df)} features")

    models = run_phase3_training(pairs, returns_df, features_df, output_dir)

    # Save model summary
    summary = {}
    for pid, m in models.items():
        summary[pid] = {
            "auc": m.get("train_score"),
            "base_rate": m.get("base_rate"),
            "mean_win": m.get("mean_win"),
            "mean_loss": m.get("mean_loss"),
            "n_samples": m.get("n_samples"),
            "feature_importance": m.get("feature_importance"),
            "error": m.get("error"),
        }
    with open(output_dir / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Phase 3 complete: {sum(1 for m in models.values() if m.get('model'))} models trained")

    # Persist models via joblib
    try:
        import joblib
        joblib.dump(models, output_dir / "phase3_models.joblib")
        print(f"  Models saved: {output_dir / 'phase3_models.joblib'}")
    except ImportError:
        print("  WARNING: joblib not available, models not persisted to disk")


def run_phase4(args):
    """Run OOS portfolio trading."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    pairs = load_pairs_from_burgess(args.pairs_json, args.top_n)
    param_grid = generate_param_grid()

    output_dir = Path(args.output_dir) / "chan_cpo"

    # Load models
    try:
        import joblib
        models = joblib.load(output_dir / "phase3_models.joblib")
        # Patch n_jobs=1 on all loaded RF models to suppress sklearn warnings
        for pid, m in models.items():
            if m.get("model") is not None:
                m["model"].n_jobs = 1
            for cid, rf in m.get("config_models", {}).items():
                if hasattr(rf, 'n_jobs'):
                    rf.n_jobs = 1
        print(f"  Loaded {sum(1 for m in models.values() if m.get('model'))} models")
    except (ImportError, FileNotFoundError):
        print("  ERROR: phase3_models.joblib not found. Run phase3 first.")
        return

    # Fetch warmup daily data (60 days before OOS start for feature computation)
    warmup_daily = None
    warmup_days = 60
    oos_start = pd.Timestamp(args.start)
    warmup_start = oos_start - timedelta(days=warmup_days * 2)  # buffer for weekends
    print(f"\n  Fetching warmup daily data: {warmup_start.strftime('%Y-%m-%d')} -> {args.start}")
    try:
        tickers = set()
        for p in pairs:
            tickers.add(p.target)
            tickers.add(p.hedge)
        warmup_daily = _fetch_warmup_daily(
            sorted(tickers), warmup_start.strftime("%Y-%m-%d"), args.start, api_key
        )
        print(f"  Warmup: {len(warmup_daily)} tickers, "
              f"~{np.mean([len(v) for v in warmup_daily.values()]):.0f} days each")
    except Exception as e:
        print(f"  WARNING: warmup fetch failed ({e}), trading will start after day 25")

    # Fetch OOS minute data
    cache_dir = Path(args.cache_dir)
    minute_data = fetch_all_minute_data(
        pairs, args.start, args.end, api_key, cache_dir
    )

    pnl_df = run_phase4_oos(
        pairs, models, minute_data, param_grid, output_dir,
        max_leverage=args.max_leverage,
        warmup_daily=warmup_daily,
    )

    # Final report
    if not pnl_df.empty:
        rets = pnl_df["portfolio_return"].values
        n_days = len(rets)
        cum = np.cumsum(rets)
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
        print(f"  Period:          {args.start} -> {args.end}")
        print(f"  Trading days:    {n_days}")
        print(f"  Cumulative ret:  {cum[-1]:+.4f}  ({cum[-1]*100:+.2f}%)")
        print(f"  Ann. return:     {ann_ret:+.4f}  ({ann_ret*100:+.1f}%)")
        print(f"  Ann. volatility: {ann_vol:.4f}   ({ann_vol*100:.1f}%)")
        print(f"  Sharpe ratio:    {sr:+.4f}")
        print(f"  Max drawdown:    {max_dd:+.4f}  ({max_dd*100:+.2f}%)")
        print(f"  Win days:        {win_days}/{n_days} ({win_days/n_days:.1%})")
        print(f"  Avg models/day:  {pnl_df['n_models_active'].mean():.1f}")
        print(f"  Max models/day:  {pnl_df['n_models_active'].max()}")


def run_full(args):
    """Run all 4 phases sequentially."""
    t_total = time.perf_counter()

    # Phase 1
    pairs_path = run_phase1(args)
    args.pairs_json = str(pairs_path)

    # Phase 2
    args.start = "2025-01-01"
    args.end = "2025-12-31"
    run_phase2(args)

    # Phase 3
    run_phase3(args)

    # Phase 4
    args.start = "2026-01-01"
    args.end = datetime.now().strftime("%Y-%m-%d")
    run_phase4(args)

    elapsed = time.perf_counter() - t_total
    print(f"\n  Total pipeline time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


# ═════════════════════════════════════════════════════════════════════════════
# DATA HELPERS (shared with run_burgess.py)
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_sp500() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    print("  Fetching S&P 500 constituents from Wikipedia...")
    try:
        import urllib.request
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8")
        from io import StringIO
        tables = pd.read_html(StringIO(html))
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch S&P 500: {e}")
    print(f"  Found {len(tickers)} constituents")
    return tickers


def _liquidity_filter(tickers, api_key, min_adv, min_price):
    """Filter tickers by ADV and price using Polygon."""
    import requests
    print(f"\n  Liquidity filter: ADV >= {min_adv:,.0f} shares, price >= ${min_price}")
    print(f"  Checking {len(tickers)} tickers (last 63 trading days)...")

    passed = []
    unavailable = 0
    dropped_vol = 0
    dropped_price = 0

    for i, ticker in enumerate(tickers):
        try:
            url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                   f"{(datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')}/"
                   f"{datetime.now().strftime('%Y-%m-%d')}"
                   f"?adjusted=true&sort=desc&limit=63&apiKey={api_key}")
            resp = requests.get(url, timeout=10)
            data = resp.json()
            results = data.get("results", [])
            if not results:
                unavailable += 1
                continue
            avg_vol = np.mean([r["v"] for r in results])
            avg_price = np.mean([r["c"] for r in results])
            if avg_vol < min_adv:
                dropped_vol += 1
            elif avg_price < min_price:
                dropped_price += 1
            else:
                passed.append(ticker)
        except Exception:
            unavailable += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(tickers)} checked ({len(passed)} passing)...")
        time.sleep(0.12)  # rate limit

    print(f"  Liquidity filter results:")
    print(f"    Passed:         {len(passed)}")
    print(f"    Dropped (vol):  {dropped_vol}")
    print(f"    Dropped (price):{dropped_price}")
    print(f"    Unavailable:    {unavailable}")
    return passed


def _fetch_warmup_daily(tickers, start, end, api_key):
    """Fetch daily close prices for warmup period. Returns {ticker: Series}."""
    import requests
    daily = {}
    for i, ticker in enumerate(tickers):
        try:
            url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                   f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")
            resp = requests.get(url, timeout=15)
            data = resp.json()
            results = data.get("results", [])
            if results:
                dates = [pd.Timestamp(r["t"], unit="ms").date() for r in results]
                closes = [r["c"] for r in results]
                daily[ticker] = pd.Series(closes, index=dates, name=ticker)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"    Warmup: {i+1}/{len(tickers)} loaded")
        time.sleep(0.12)
    return daily


def _fetch_daily_polygon(tickers, start, end, api_key):
    """Fetch daily close prices from Polygon."""
    import requests
    print(f"\n  Polygon: fetching {len(tickers)} tickers...")

    all_series = {}
    for i, ticker in enumerate(tickers):
        try:
            url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                   f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")
            resp = requests.get(url, timeout=15)
            data = resp.json()
            results = data.get("results", [])
            if results:
                dates_t = [pd.Timestamp(r["t"], unit="ms").date() for r in results]
                closes = [r["c"] for r in results]
                all_series[ticker] = pd.Series(closes, index=dates_t, name=ticker)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  Polygon: {i+1}/{len(tickers)} loaded")
        time.sleep(0.12)

    print(f"  Polygon: {len(all_series)}/{len(tickers)} loaded")

    if not all_series:
        raise RuntimeError("No price data from Polygon")

    df = pd.DataFrame(all_series)
    print(f"  Polygon date range: {df.index.min()} -> {df.index.max()} ({len(df)} raw rows)")

    # Per-ticker coverage check
    valid_tickers = [t for t in tickers if t in df.columns]
    keep = []
    for t in valid_tickers:
        nan_frac = df[t].isna().mean()
        if nan_frac <= 0.10:
            keep.append(t)
        else:
            print(f"  WARNING: Dropping {t} ({nan_frac:.1%} missing)")
    valid_tickers = keep

    df = df[valid_tickers].ffill().dropna()
    dates = [str(d) for d in df.index]
    prices = df.values.astype(np.float64)
    print(f"  After ffill: {prices.shape[0]} days x {prices.shape[1]} assets")
    return prices, valid_tickers, dates


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Burgess → Chan CPO Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    # ── Shared args ───────────────────────────────────────────────────
    def add_common_args(p):
        p.add_argument("--output-dir", type=Path, default=Path("output/burgess"))
        p.add_argument("--cache-dir", type=str, default="data/minute_cache")
        p.add_argument("--top-n", type=int, default=50)

    # ── Phase 1 ───────────────────────────────────────────────────────
    p1 = subparsers.add_parser("phase1", help="Pair discovery via Burgess (n=2)")
    add_common_args(p1)
    data_group = p1.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--tickers", nargs="+")
    data_group.add_argument("--universe", choices=["sp500"])
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

    # ── Phase 2 ───────────────────────────────────────────────────────
    p2 = subparsers.add_parser("phase2", help="Param grid on minute data (training)")
    add_common_args(p2)
    p2.add_argument("--pairs-json", type=str, required=True)
    p2.add_argument("--start", type=str, default="2025-01-01")
    p2.add_argument("--end", type=str, default="2025-12-31")

    # ── Phase 3 ───────────────────────────────────────────────────────
    p3 = subparsers.add_parser("phase3", help="Train Random Forest models")
    add_common_args(p3)
    p3.add_argument("--pairs-json", type=str, required=True)

    # ── Phase 4 ───────────────────────────────────────────────────────
    p4 = subparsers.add_parser("phase4", help="OOS portfolio trading (Kelly)")
    add_common_args(p4)
    p4.add_argument("--pairs-json", type=str, required=True)
    p4.add_argument("--start", type=str, default="2026-01-01")
    p4.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    p4.add_argument("--max-leverage", type=float, default=2.0)

    # ── Full pipeline ─────────────────────────────────────────────────
    pf = subparsers.add_parser("full", help="Run all 4 phases")
    add_common_args(pf)
    data_group_f = pf.add_mutually_exclusive_group(required=True)
    data_group_f.add_argument("--tickers", nargs="+")
    data_group_f.add_argument("--universe", choices=["sp500"])
    pf.add_argument("--years", type=int, default=2)
    pf.add_argument("--end-date", type=str, default="2024-12-31")
    pf.add_argument("--train-frac", type=float, default=0.667)
    pf.add_argument("--min-adv", type=float, default=500_000)
    pf.add_argument("--min-price", type=float, default=10.0)
    pf.add_argument("--no-liquidity-filter", action="store_true")
    pf.add_argument("--surface", type=str, default="data/surfaces.duckdb")
    pf.add_argument("--est-window", type=int, default=252)
    pf.add_argument("--sig-window", type=int, default=63)
    pf.add_argument("--step-size", type=int, default=21)
    pf.add_argument("--entry-z", type=float, default=2.0)
    pf.add_argument("--exit-z", type=float, default=0.5)
    pf.add_argument("--stop-z", type=float, default=4.0)
    pf.add_argument("--tc-bps", type=float, default=5.0)
    pf.add_argument("--slippage-bps", type=float, default=2.0)
    pf.add_argument("--max-leverage", type=float, default=2.0)

    args = parser.parse_args()

    if args.phase == "phase1":
        run_phase1(args)
    elif args.phase == "phase2":
        run_phase2(args)
    elif args.phase == "phase3":
        run_phase3(args)
    elif args.phase == "phase4":
        run_phase4(args)
    elif args.phase == "full":
        run_full(args)


if __name__ == "__main__":
    main()
