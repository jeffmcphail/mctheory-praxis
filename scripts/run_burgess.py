#!/usr/bin/env python3
"""
run_burgess.py — CLI runner for the Burgess statistical arbitrage engine.

Data: Polygon.io (requires POLYGON_API_KEY in .env).

Usage:
    # Calibration run: S&P 500, 5yr ending 2yr ago, top 100, analyze scores
    python scripts/run_burgess.py --universe sp500 --years 5 --end-date 2024-03-01 --top-k 100

    # Holdout validation: same universe, next 2yr period
    python scripts/run_burgess.py --universe sp500 --years 2 --end-date 2026-03-01 --top-k 100

    # Custom tickers
    python scripts/run_burgess.py --tickers SPY XLF XLK XLE XLV XLI XLP XLY XLU XLB

    # Quick scan only
    python scripts/run_burgess.py --universe sp500 --scan-only --end-date 2024-03-01
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═════════════════════════════════════════════════════════════════════════════

def fetch_sp500_tickers():
    import requests
    import pandas as pd
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print("  Fetching S&P 500 constituents from Wikipedia...")
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"  Found {len(tickers)} constituents")
    return tickers


def liquidity_filter(tickers, start, end, min_adv=500_000, min_price=5.0, lookback_days=63):
    import os
    print(f"\n  Liquidity filter: ADV >= {min_adv:,.0f} shares, price >= ${min_price:.0f}")
    print(f"  Checking {len(tickers)} tickers (last {lookback_days} trading days)...")

    api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    passed, dropped_vol, dropped_price, failed = [], [], [], []

    if api_key:
        import requests as req_lib
        for i, ticker in enumerate(tickers):
            try:
                url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
                       f"/{start}/{end}?adjusted=true&sort=desc&limit={lookback_days}"
                       f"&apiKey={api_key}")
                resp = req_lib.get(url, timeout=15)
                data = resp.json()
                if data.get("resultsCount", 0) == 0 or "results" not in data:
                    failed.append(ticker); continue
                bars = data["results"]
                avg_vol = np.mean([b["v"] for b in bars])
                avg_price = np.mean([b["c"] for b in bars])
                if avg_vol < min_adv: dropped_vol.append((ticker, avg_vol))
                elif avg_price < min_price: dropped_price.append((ticker, avg_price))
                else: passed.append(ticker)
            except Exception:
                failed.append(ticker)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(tickers)} checked ({len(passed)} passing)...")
        if failed:
            print(f"  Polygon: {len(failed)} unavailable")
    else:
        raise RuntimeError("No POLYGON_API_KEY — set it in .env")

    print(f"  Liquidity filter results:")
    print(f"    Passed:         {len(passed)}")
    print(f"    Dropped (vol):  {len(dropped_vol)}")
    print(f"    Dropped (price):{len(dropped_price)}")
    if failed: print(f"    Unavailable:    {len(failed)}")
    return passed



def _fetch_polygon(tickers, start, end):
    import os
    import pandas as pd
    api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not api_key: return None
    try:
        import requests
    except ImportError: return None
    print(f"  Polygon: fetching {len(tickers)} tickers...")
    series, failed = {}, []
    for ticker in tickers:
        url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
               f"/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")
        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()
            if data.get("resultsCount", 0) == 0 or "results" not in data:
                failed.append(ticker); continue
            bars = data["results"]
            dates = pd.to_datetime([b["t"] for b in bars], unit="ms").normalize()
            closes = [b["c"] for b in bars]
            s = pd.Series(closes, index=dates, name=ticker, dtype=np.float64)
            s.index = s.index.date  # normalize to pure date for cross-source alignment
            s = s[~s.index.duplicated(keep="last")]
            series[ticker] = s
        except Exception as e:
            print(f"  WARNING: Polygon failed for {ticker}: {e}")
            failed.append(ticker)
    if failed: print(f"  Polygon: {len(failed)} tickers failed: {', '.join(failed)}")
    return series if series else None



def fetch_prices(tickers, years=5, end_date=None):
    import pandas as pd
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    print(f"Fetching {len(tickers)} tickers")
    print(f"  Period: {start} -> {end}")

    all_series = {}
    polygon_data = _fetch_polygon(tickers, start, end)
    if not polygon_data:
        raise RuntimeError("Polygon fetch failed — check POLYGON_API_KEY in .env")
    all_series.update(polygon_data)
    got = set(all_series.keys())
    missing = [t for t in tickers if t not in got]
    print(f"  Polygon: {len(got)}/{len(tickers)} loaded")
    if missing:
        print(f"  Unavailable: {', '.join(missing[:20])}")

    if not all_series: raise RuntimeError("No price data from Polygon")

    df = pd.DataFrame(all_series)
    print(f"  Polygon date range: {df.index.min()} -> {df.index.max()} ({len(df)} raw rows)")

    # Per-ticker: check coverage, drop if too sparse
    valid_tickers = [t for t in tickers if t in df.columns]
    keep = []
    for t in valid_tickers:
        nan_frac = df[t].isna().mean()
        if nan_frac <= 0.10: keep.append(t)
        else: print(f"  WARNING: Dropping {t} ({nan_frac:.1%} missing)")
    valid_tickers = keep

    # Forward-fill small gaps, drop leading NaN rows
    df = df[valid_tickers].ffill().dropna()
    dates = [str(d) for d in df.index]
    prices = df.values.astype(np.float64)
    print(f"  After ffill: {prices.shape[0]} days x {prices.shape[1]} assets")
    return prices, valid_tickers, dates


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def print_progress(phase, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {phase}: {msg}")


def print_results(output, asset_names):
    print("\n" + "=" * 78)
    print("BURGESS ENGINE — RESULTS SUMMARY")
    print("=" * 78)

    print(f"\n  Universe:      {output.n_assets} assets, {output.n_obs} observations")
    print(f"  Train/Test:    {output.n_train} / {output.n_test} obs")
    print(f"  Ranked:        {len(output.ranked_baskets)} baskets scored")
    print(f"  OOS pass:      {len(output.oos_passed_baskets)}")
    print(f"  Backtested:    {len(output.backtest_results)}")

    print(f"\n  Phase Timing:")
    for phase, elapsed in output.phase_times.items():
        print(f"    {phase:20s} {elapsed:8.2f}s")
    print(f"    {'TOTAL':20s} {sum(output.phase_times.values()):8.2f}s")

    print(f"\n  Score Weights: {output.score_weights}")

    # Top ranked
    if output.ranked_baskets:
        print(f"\n  Top 15 Ranked Baskets (before OOS):")
        print(f"  {'Rk':>3s}  {'Target':>8s}  {'Basket':<25s}  {'Score':>7s}  "
              f"{'ADF':>7s}  {'eigen2':>7s}  {'Hurst':>7s}  {'HL':>7s}  {'Maha':>7s}")
        print(f"  {'---':>3s}  {'--------':>8s}  {'-'*25}  {'-------':>7s}  "
              f"{'-------':>7s}  {'-------':>7s}  {'-------':>7s}  {'-------':>7s}  {'-------':>7s}")
        for b in output.ranked_baskets[:15]:
            tgt = asset_names[b.target_idx] if b.target_idx < len(asset_names) else f"A{b.target_idx}"
            bsk = ", ".join(asset_names[i] if i < len(asset_names) else f"A{i}" for i in b.basket_indices)
            if len(bsk) > 25: bsk = bsk[:22] + "..."
            sc = b.score_components
            print(f"  {b.rank:3d}  {tgt:>8s}  {bsk:<25s}  {b.composite_score:7.4f}  "
                  f"{sc.get('adf_t',0):7.4f}  {sc.get('vr_eigen2_proj',0):7.4f}  "
                  f"{sc.get('hurst',0):7.4f}  {sc.get('half_life',0):7.4f}  "
                  f"{sc.get('vr_mahalanobis',0):7.4f}")

    # Performance
    if output.performance:
        print(f"\n  Backtest Performance (top 20 by Sharpe, single-asset trading):")
        print(f"  {'Target':>8s}  {'Basket':<20s}  {'Sharpe':>7s}  {'Ann.Ret':>8s}  {'MaxDD':>7s}  "
              f"{'WR':>5s}  {'#Tr':>4s}  {'AvgH':>5s}  {'PF':>6s}  {'Score':>7s}")
        print(f"  {'--------':>8s}  {'-'*20}  {'-------':>7s}  {'--------':>8s}  {'-------':>7s}  "
              f"{'-----':>5s}  {'----':>4s}  {'-----':>5s}  {'------':>6s}  {'-------':>7s}")
        for p in output.performance[:20]:
            bsk = ", ".join(p.basket_names)
            if len(bsk) > 20: bsk = bsk[:17] + "..."
            print(f"  {p.target_name:>8s}  {bsk:<20s}  {p.sharpe_ratio:7.3f}  "
                  f"{p.annual_return:7.2%}  {p.max_drawdown:7.2%}  "
                  f"{p.win_rate:5.1%}  {p.n_trades:4d}  {p.avg_hold_days:4.0f}d  "
                  f"{p.profit_factor:6.2f}  {p.composite_score:7.4f}")

        # Score component correlation with Sharpe
        if len(output.performance) >= 10:
            print(f"\n  Score Component Correlation with Sharpe Ratio:")
            sharpes = np.array([p.sharpe_ratio for p in output.performance])
            for comp_name in output.score_weights.keys():
                comp_vals = np.array([p.score_components.get(comp_name, 0) for p in output.performance])
                if np.std(comp_vals) > 1e-10 and np.std(sharpes) > 1e-10:
                    corr = np.corrcoef(comp_vals, sharpes)[0, 1]
                    print(f"    {comp_name:20s}  r = {corr:+.4f}")

    print("\n" + "=" * 78)


def save_results(output, asset_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
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
             "max_dd_duration": p.max_drawdown_duration,
             "win_rate": p.win_rate, "avg_win": p.avg_win, "avg_loss": p.avg_loss,
             "profit_factor": p.profit_factor, "n_trades": p.n_trades,
             "avg_hold_days": p.avg_hold_days, "total_costs": p.total_costs,
             "turnover": p.turnover,
             "composite_score": p.composite_score,
             "score_components": p.score_components}
            for p in output.performance
        ],
    }
    out_path = output_dir / f"burgess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Burgess Statistical Arbitrage Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--tickers", nargs="+", help="Ticker symbols")
    data_group.add_argument("--ticker-file", type=Path, help="File with tickers")
    data_group.add_argument("--universe", choices=["sp500"], help="Built-in universe")

    parser.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--train-frac", type=float, default=0.667, help="Train fraction (default: 0.667 = 2yr/1yr)")

    parser.add_argument("--min-adv", type=float, default=500_000, help="Min avg daily volume")
    parser.add_argument("--min-price", type=float, default=5.0, help="Min avg close price")
    parser.add_argument("--no-liquidity-filter", action="store_true")

    parser.add_argument("--surface", type=str, default="data/surfaces.duckdb")
    parser.add_argument("--n-vars", type=int, default=3, help="Basket size (default: 3)")
    parser.add_argument("--top-k", type=int, default=100, help="Top baskets to backtest (default: 100)")

    parser.add_argument("--est-window", type=int, default=504)
    parser.add_argument("--sig-window", type=int, default=63)
    parser.add_argument("--step-size", type=int, default=21)

    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stop-z", type=float, default=4.0)

    parser.add_argument("--tc-bps", type=float, default=10.0, help="One-way transaction cost bps")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="One-way slippage bps")

    parser.add_argument("--output-dir", type=Path, default=Path("output/burgess"))
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()

    # Load .env
    from dotenv import load_dotenv
    load_dotenv()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")
    elif not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Resolve tickers
    if args.universe == "sp500":
        tickers = fetch_sp500_tickers()
    elif args.ticker_file:
        tickers = [l.strip().upper() for l in args.ticker_file.read_text().splitlines()
                    if l.strip() and not l.strip().startswith("#")]
    else:
        tickers = [t.upper() for t in args.tickers]

    if len(tickers) < 3:
        print("ERROR: Need >= 3 tickers."); sys.exit(1)

    # Liquidity filter
    if not args.no_liquidity_filter and len(tickers) > 20:
        end = args.end_date or datetime.now().strftime("%Y-%m-%d")
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
        tickers = liquidity_filter(tickers, start, end, min_adv=args.min_adv, min_price=args.min_price)
        if len(tickers) < 3:
            print(f"ERROR: Only {len(tickers)} survived filter."); sys.exit(1)

    # Surface check
    surface_path = Path(args.surface)
    if not surface_path.exists():
        print(f"ERROR: Surface not found: {surface_path}"); sys.exit(1)

    print("=" * 78)
    print("BURGESS ENGINE — STARTING")
    print("=" * 78)

    try:
        prices, valid_tickers, dates = fetch_prices(tickers, args.years, args.end_date)
    except Exception as e:
        print(f"ERROR fetching prices: {e}"); sys.exit(1)

    if prices.shape[1] < 3:
        print(f"ERROR: Only {prices.shape[1]} valid tickers."); sys.exit(1)

    from engines.burgess import BurgessParams, BurgessEngine

    params = BurgessParams(
        surface_db_path=str(surface_path),
        n_vars=args.n_vars,
        top_k=args.top_k,
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

    print(f"\n  Surface:    {surface_path} ({surface_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  Top-K:      {params.top_k}")
    print(f"  Basket:     {params.n_vars} vars, ridge a={params.ridge_alpha}")
    print(f"  WF windows: est={params.wf_estimation_window}d, sig={params.wf_signal_window}d, step={params.wf_step_size}d")
    print(f"  Signals:    entry={params.entry_threshold}, exit={params.exit_threshold}, stop={params.stop_loss_threshold}")
    print(f"  Costs:      tc={params.transaction_cost_bps}bps + slip={params.slippage_bps}bps = {params.transaction_cost_bps+params.slippage_bps}bps one-way (1 leg)")
    print(f"  Weights:    {params.score_weights}")
    print()

    callback = print_progress if not args.quiet else None

    if args.scan_only:
        from engines.burgess import full_universe_scan, surface_rank, split_train_test, burgess_production_requirement
        from praxis.stats.surface import CompositeSurface
        train, test, split_idx = split_train_test(prices, params.train_frac, params.min_train_obs, params.min_test_obs)
        print(f"  SCAN-ONLY mode: Train {train.shape[0]} obs\n")
        regressions, stationarity = full_universe_scan(train, params, callback)
        surface = CompositeSurface(db_path=params.surface_db_path)
        req = burgess_production_requirement()
        ranked = surface_rank(regressions, stationarity, surface, req,
                              n_assets=train.shape[1], n_obs=train.shape[0], params=params)
        print(f"\n  Top 20 by composite score:")
        for b in ranked[:20]:
            tgt = valid_tickers[b.target_idx]
            bsk = ", ".join(valid_tickers[i] for i in b.basket_indices)
            sc = b.score_components
            print(f"    #{b.rank:3d}  {tgt} ~ [{bsk}]  score={b.composite_score:.4f}  "
                  f"adf={sc.get('adf_t',0):.3f}  e2={sc.get('vr_eigen2_proj',0):.3f}  "
                  f"H={b.stationarity.hurst_exponent:.3f}")
    else:
        engine = BurgessEngine()
        output = engine.run(prices, valid_tickers, params, callback)
        print_results(output, valid_tickers)
        save_results(output, valid_tickers, args.output_dir)


if __name__ == "__main__":
    main()
