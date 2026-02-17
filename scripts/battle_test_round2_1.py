#!/usr/bin/env python3
"""
Battle Test Suite — Round 2.1: CPO + Burgess Stress Test
==========================================================

Round 2 was a smoke test. This is the real thing.

Key changes from Round 2:
- 3-year date range (750+ bars) instead of 1 year (250)
- CPO: 5×5×3 param grid over 30+ training dates = 2250+ rows
- Burgess: relaxed thresholds to force selections through MC correction
- Dense single-leg sweep: 5×7×3 = 105 param combos
- All outputs fingerprinted for Claude reconciliation

RUN:
    cd mctheory_praxis
    python scripts/battle_test_round2_1.py

ESTIMATED TIME: 60-120 seconds

GIVE BACK TO CLAUDE:
    Upload battle_results/round2_1/ directory.
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "battle_results" / "round2_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "results_round2_1.json"
ERRORS_LOG = OUTPUT_DIR / "errors.log"

# ── Dependency check ──────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["numpy", "pandas", "polars", "pyarrow", "yfinance",
                "duckdb", "pydantic", "sklearn", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import numpy as np
import pandas as pd
import polars as pl

results = {
    "meta": {
        "script": "battle_test_round2_1.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "polars_version": pl.__version__,
    },
    "tests": {},
}
error_lines = []

def run_test(name, fn):
    print(f"  [{name}] ... ", end="", flush=True)
    t0 = time.monotonic()
    try:
        result = fn()
        elapsed = time.monotonic() - t0
        results["tests"][name] = {
            "status": "PASS", "elapsed_seconds": round(elapsed, 3), "result": result,
        }
        print(f"PASS ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc()
        results["tests"][name] = {
            "status": "FAIL", "elapsed_seconds": round(elapsed, 3), "error": str(e),
        }
        error_lines.append(f"=== {name} ===\n{tb}\n")
        print(f"FAIL ({elapsed:.1f}s): {e}")


# ==============================================================
#  SHARED: 3-year universe with vendor capture
# ==============================================================

UNIVERSE_TICKERS = ["AAPL", "MSFT", "GLD", "GDX", "SPY", "QQQ",
                    "TLT", "SLV", "XLE", "XLF", "IWM", "EEM"]
UNIVERSE_START = "2021-01-01"
UNIVERSE_END   = "2024-01-01"

_universe = None

def get_universe():
    global _universe
    if _universe is not None:
        return _universe

    from praxis.logger.core import PraxisLogger
    from praxis.logger.vendor_capture import start_capture_session, end_capture_session
    from praxis.data import fetch_prices

    PraxisLogger.reset()
    log = PraxisLogger.instance()
    log.configure_defaults()

    adapter = start_capture_session(capture_dir=OUTPUT_DIR, session_id="round2_1")

    _universe = {}
    for ticker in UNIVERSE_TICKERS:
        try:
            df = fetch_prices(ticker, start=UNIVERSE_START, end=UNIVERSE_END)
            _universe[ticker] = df
        except Exception as e:
            print(f"    WARN: {ticker} failed: {e}")

    summary = end_capture_session()
    print(f"    Captured {summary['calls_captured']} calls, tickers: {list(_universe.keys())}")
    min_rows = min(len(v) for v in _universe.values()) if _universe else 0
    print(f"    Min rows per ticker: {min_rows}")
    return _universe


def get_pair(ticker_a="GLD", ticker_b="GDX"):
    u = get_universe()
    a, b = u[ticker_a], u[ticker_b]
    n = min(len(a), len(b))
    return (
        a["close"].to_numpy()[:n].astype(float),
        a["open"].to_numpy()[:n].astype(float) if "open" in a.columns else a["close"].to_numpy()[:n].astype(float),
        b["close"].to_numpy()[:n].astype(float),
        n,
    )


def build_matrix():
    u = get_universe()
    tickers = [t for t in UNIVERSE_TICKERS if t in u]
    n = min(len(u[t]) for t in tickers)
    matrix = np.column_stack([u[t]["close"].to_numpy()[:n].astype(float) for t in tickers])
    return matrix, tickers, n


# ==============================================================
#  A: CPO SINGLE-LEG — Dense Sweep (105 combos)
# ==============================================================

def test_A1_dense_sweep():
    """105-combo param sweep on 3-year GLD/GDX."""
    from praxis.cpo import execute_single_leg

    close_a, open_a, close_b, n = get_pair("GLD", "GDX")
    sweep = {}
    weights = [1.5, 2.0, 2.5, 3.0, 5.0]
    entries = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    lookbacks = [30, 60, 120]

    for w in weights:
        for e in entries:
            for lb in lookbacks:
                params = {"weight": w, "lookback": lb, "entry_threshold": e,
                          "exit_threshold_fraction": -0.6}
                r = execute_single_leg(close_a, open_a, close_b, params)
                key = f"w{w}_e{e}_lb{lb}"
                sweep[key] = {
                    "sharpe": round(float(r.sharpe_ratio), 6),
                    "trades": int(r.num_trades),
                    "return": round(float(r.annualized_return), 6),
                    "vol": round(float(r.volatility), 6),
                }

    # Summary stats
    sharpes = [v["sharpe"] for v in sweep.values() if np.isfinite(v["sharpe"])]
    trades = [v["trades"] for v in sweep.values()]

    return {
        "n_bars": n, "n_combos": len(sweep),
        "sharpe_mean": round(float(np.mean(sharpes)), 6),
        "sharpe_std": round(float(np.std(sharpes)), 6),
        "sharpe_min": round(float(np.min(sharpes)), 6),
        "sharpe_max": round(float(np.max(sharpes)), 6),
        "sharpe_median": round(float(np.median(sharpes)), 6),
        "trades_mean": round(float(np.mean(trades)), 2),
        "trades_max": int(np.max(trades)),
        "zero_trade_combos": sum(1 for t in trades if t == 0),
        "sweep": sweep,
    }


# ==============================================================
#  B: CPO TRAINING — Large Grid
# ==============================================================

def test_B1_large_training():
    """CPO training: 5×5×3=75 params × 30+ dates = 2000+ rows."""
    from praxis.cpo import generate_training_data

    close_a, open_a, close_b, n = get_pair("GLD", "GDX")

    # Dense date spacing: every 20 bars from bar 120 onward
    dates = np.arange(120, n - 20, 20)

    param_grid = {
        "weights": [1.5, 2.0, 3.0, 4.0, 5.0],
        "entry_thresholds": [0.25, 0.5, 1.0, 1.5, 2.0],
        "lookbacks": [30, 60, 120],
    }

    training = generate_training_data(
        close_a, open_a, close_b, dates, param_grid,
        transaction_costs=0.0005,
    )

    df = training.to_polars()

    # Per-param summary
    param_summary = {}
    for w in [1.5, 2.0, 3.0, 4.0, 5.0]:
        subset = df.filter(pl.col("weight") == w)
        if len(subset) > 0:
            param_summary[f"w{w}"] = {
                "rows": len(subset),
                "sharpe_mean": round(float(subset["sharpe_ratio"].mean()), 6),
                "sharpe_std": round(float(subset["sharpe_ratio"].std()), 6),
            }

    return {
        "n_bars": n,
        "n_dates": len(dates),
        "grid_size": 75,
        "total_rows": training.count,
        "errors": len(training.errors),
        "sharpe_mean": round(float(df["sharpe_ratio"].mean()), 6) if len(df) > 0 else None,
        "sharpe_std": round(float(df["sharpe_ratio"].std()), 6) if len(df) > 0 else None,
        "sharpe_p25": round(float(df["sharpe_ratio"].quantile(0.25)), 6) if len(df) > 0 else None,
        "sharpe_p50": round(float(df["sharpe_ratio"].quantile(0.50)), 6) if len(df) > 0 else None,
        "sharpe_p75": round(float(df["sharpe_ratio"].quantile(0.75)), 6) if len(df) > 0 else None,
        "num_trades_mean": round(float(df["num_trades"].mean()), 2) if len(df) > 0 else None,
        "param_summary": param_summary,
    }


# ==============================================================
#  C: CPO PREDICTOR — Proper Fit + Out-of-Sample
# ==============================================================

def test_C1_predictor_proper():
    """CPO predictor on 2000+ training rows with OOS validation."""
    from praxis.cpo import generate_training_data, CPOPredictor

    close_a, open_a, close_b, n = get_pair("GLD", "GDX")
    dates = np.arange(120, n - 20, 15)

    param_grid = {
        "weights": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "entry_thresholds": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        "lookbacks": [30, 60, 90, 120],
    }

    training = generate_training_data(close_a, open_a, close_b, dates, param_grid)
    df = training.to_polars()

    predictor = CPOPredictor(n_estimators=200, random_state=42)
    fit_metrics = predictor.fit(df, train_fraction=0.7)

    candidates = df.select(["weight", "entry_threshold", "lookback"]).unique()
    prediction = predictor.predict_best_params(candidates)

    # Validate prediction by running single-leg with predicted params
    from praxis.cpo import execute_single_leg
    best = prediction.predicted_params.copy()
    best["exit_threshold_fraction"] = -0.6
    actual = execute_single_leg(close_a, open_a, close_b, best)

    return {
        "training_rows": training.count,
        "mse_train": round(float(fit_metrics["mse_train"]), 6),
        "mse_test": round(float(fit_metrics["mse_test"]), 6),
        "r2_train": round(float(fit_metrics["r2_train"]), 6),
        "predicted_weight": round(float(prediction.predicted_params["weight"]), 4),
        "predicted_entry": round(float(prediction.predicted_params["entry_threshold"]), 4),
        "predicted_lookback": round(float(prediction.predicted_params["lookback"]), 4),
        "predicted_sharpe": round(float(prediction.predicted_sharpe), 6),
        "actual_sharpe_with_predicted_params": round(float(actual.sharpe_ratio), 6),
        "actual_trades": int(actual.num_trades),
        "actual_return": round(float(actual.annualized_return), 6),
    }


# ==============================================================
#  D: BURGESS — Relaxed to Force Selections
# ==============================================================

def test_D1_burgess_candidates_3yr():
    """Burgess candidates on 3-year 12-asset universe."""
    from praxis.models.burgess import generate_candidates

    matrix, tickers, n_obs = build_matrix()
    candidates = generate_candidates(matrix, n_per_basket=3, significance=0.10)

    # Detailed stats on all candidates
    stats = []
    for c in candidates:
        stats.append({
            "target": tickers[c.target_index],
            "partners": [tickers[i] for i in c.partner_indices],
            "adf_t": round(float(c.adf_t_statistic), 6),
            "pvalue": round(float(c.adf_p_value), 6),
            "hurst": round(float(c.hurst), 6) if np.isfinite(c.hurst) else None,
            "half_life": round(float(c.half_life_periods), 4) if np.isfinite(c.half_life_periods) else None,
            "r_squared": round(float(c.r_squared), 6),
            "is_stationary": c.is_stationary,
        })

    n_stationary = sum(1 for c in candidates if c.is_stationary)
    n_good_hurst = sum(1 for c in candidates if c.hurst < 0.5)

    return {
        "n_assets": len(tickers), "n_obs": n_obs, "tickers": tickers,
        "n_candidates": len(candidates),
        "n_stationary": n_stationary,
        "n_good_hurst": n_good_hurst,
        "all_candidates": stats,
    }


def test_D2_burgess_pipeline_relaxed():
    """Burgess full pipeline with relaxed thresholds."""
    from praxis.models.burgess import BurgessStatArb, BurgessConfig

    matrix, tickers, n_obs = build_matrix()

    config = BurgessConfig(
        n_per_basket=3,
        significance=0.20,       # Very relaxed
        mc_enabled=True,
        mc_n_samples=500,
        mc_seed=42,
        top_k=10,
        min_half_life=1.0,
        max_half_life=500.0,     # Relaxed
        max_hurst=0.55,          # Relaxed
        optimization_method="min_variance",
    )

    engine = BurgessStatArb(config)
    result = engine.run(matrix)

    selected_info = []
    for b in result.selected_baskets:
        selected_info.append({
            "target": tickers[b.target_index],
            "partners": [tickers[i] for i in b.partner_indices],
            "adf_t": round(float(b.adf_t_statistic), 6),
            "adj_pvalue": round(float(b.adjusted_p_value), 6),
            "hurst": round(float(b.hurst), 6),
            "half_life": round(float(b.half_life_periods), 4),
            "rank": b.rank,
        })

    portfolio_info = []
    for p in result.portfolio_results:
        portfolio_info.append({
            "weights": [round(float(w), 6) for w in p.weights],
            "volatility": round(float(p.volatility), 6),
            "expected_return": round(float(p.expected_return), 6),
            "sharpe": round(float(p.sharpe_ratio), 6),
            "method": p.method,
        })

    return {
        "n_assets": len(tickers), "n_obs": n_obs,
        "n_scanned": result.n_scanned,
        "n_selected": result.n_candidates,
        "has_mc_cv": result.critical_values is not None,
        "n_portfolios": len(result.portfolio_results),
        "elapsed": round(result.elapsed_seconds, 3),
        "selected": selected_info,
        "portfolios": portfolio_info,
    }


# ==============================================================
#  E: PORTFOLIO — All Methods
# ==============================================================

def test_E1_portfolio_all_methods():
    """Min-var, max-Sharpe, equal-weight on 12-asset universe."""
    from praxis.stats.portfolio import (
        min_variance_portfolio, max_sharpe_portfolio,
        equal_weight_portfolio, covariance_matrix,
    )

    matrix, tickers, n_obs = build_matrix()
    returns = np.diff(np.log(matrix), axis=0)
    cov = covariance_matrix(returns)
    mu = np.mean(returns, axis=0) * 252  # Annualized

    mv = min_variance_portfolio(cov)
    ms = max_sharpe_portfolio(mu, cov, risk_free_rate=0.0)
    ew = equal_weight_portfolio(len(tickers))

    def portfolio_stats(pr, label):
        return {
            "method": label,
            "weights": {t: round(float(w), 6) for t, w in zip(tickers, pr.weights)},
            "weights_sum": round(float(np.sum(pr.weights)), 6),
            "volatility": round(float(pr.volatility), 6),
            "expected_return": round(float(pr.expected_return), 6),
            "sharpe": round(float(pr.sharpe_ratio), 6),
            "max_weight": round(float(np.max(np.abs(pr.weights))), 6),
            "n_nonzero": int(np.sum(np.abs(pr.weights) > 0.01)),
        }

    return {
        "n_assets": len(tickers), "n_obs": n_obs,
        "cov_shape": list(cov.shape),
        "cov_condition_number": round(float(np.linalg.cond(cov)), 2),
        "min_variance": portfolio_stats(mv, "min_variance"),
        "max_sharpe": portfolio_stats(ms, "max_sharpe"),
        "equal_weight": portfolio_stats(ew, "equal_weight"),
    }


# ==============================================================
#  F: REGRESSION — Full Stepwise + Ridge
# ==============================================================

def test_F1_full_stepwise():
    """Stepwise regression for all 12 targets, 3 partners each."""
    from praxis.stats.regression import successive_regression

    matrix, tickers, n_obs = build_matrix()
    all_results = []

    for idx in range(len(tickers)):
        r = successive_regression(target_index=idx, asset_matrix=matrix, n_vars=3, compute_stats=True)
        entry = {
            "target": tickers[idx],
            "partners": [tickers[i] for i in r.selected_indices],
            "is_stationary": r.is_stationary,
        }
        if r.regression:
            entry["r_squared"] = round(float(r.regression.r_squared), 6)
            entry["residual_std"] = round(float(np.std(r.regression.residuals)), 6)
        if r.adf:
            entry["adf_t"] = round(float(r.adf.t_statistic), 6)
            entry["adf_p"] = round(float(r.adf.p_value), 6)
        all_results.append(entry)

    return {
        "n_assets": len(tickers), "n_obs": n_obs,
        "n_stationary": sum(1 for r in all_results if r.get("is_stationary", False)),
        "results": all_results,
    }


def test_F2_ridge_spy():
    """Ridge regression: predict SPY from all other assets."""
    from praxis.stats.regression import ridge_regression

    matrix, tickers, n_obs = build_matrix()
    spy_idx = tickers.index("SPY")
    y = matrix[:, spy_idx]
    X = np.delete(matrix, spy_idx, axis=1)
    others = [t for i, t in enumerate(tickers) if i != spy_idx]

    # Multiple alpha values
    alpha_results = {}
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        r = ridge_regression(y, X, alpha=alpha)
        alpha_results[f"alpha_{alpha}"] = {
            "r_squared": round(float(r.r_squared), 6),
            "adj_r_squared": round(float(r.adj_r_squared), 6),
            "residual_std": round(float(np.std(r.residuals)), 6),
            "beta_max": round(float(np.max(np.abs(r.beta))), 6),
        }

    # Best alpha
    best_r = ridge_regression(y, X, alpha=1.0)
    betas = {t: round(float(b), 6) for t, b in zip(others, best_r.beta[1:])}

    return {
        "target": "SPY", "features": others, "n_obs": n_obs,
        "alpha_sweep": alpha_results,
        "best_betas": betas,
        "best_r2": round(float(best_r.r_squared), 6),
    }


# ==============================================================
#  G: BACKTEST — Captured Universe Data
# ==============================================================

def test_G1_sma_3yr():
    """SMA crossover on 3-year captured AAPL."""
    from praxis.config import ModelConfig, ModelIdentity
    from praxis.runner import PraxisRunner
    from praxis.logger.core import PraxisLogger

    PraxisLogger.reset()
    PraxisLogger.instance().configure_defaults()

    prices = get_universe()["AAPL"]

    config = ModelConfig(
        model=ModelIdentity(name="r2_sma_3yr", version="v1.0"),
        signal={"method": "sma_crossover", "fast_period": 10, "slow_period": 30},
        sizing={"method": "fixed_fraction", "fraction": 0.1},
        backtest={"initial_capital": 100_000, "commission_bps": 10},
    )

    runner = PraxisRunner()
    result = runner.run_config(config, prices)
    m = result.metrics

    return {
        "data_rows": len(prices),
        "total_return": round(float(m["total_return"]), 6),
        "sharpe_ratio": round(float(m["sharpe_ratio"]), 6),
        "max_drawdown": round(float(m["max_drawdown"]), 6),
        "total_trades": int(m["total_trades"]),
        "win_rate": round(float(m["win_rate"]), 6),
        "annualized_return": round(float(m["annualized_return"]), 6),
        "volatility": round(float(m["volatility"]), 6),
    }


def test_G2_multi_ticker_backtest():
    """SMA crossover on all 12 tickers — compare metrics."""
    from praxis.config import ModelConfig, ModelIdentity
    from praxis.runner import PraxisRunner
    from praxis.logger.core import PraxisLogger

    u = get_universe()
    ticker_results = {}

    for ticker in UNIVERSE_TICKERS:
        if ticker not in u:
            continue
        PraxisLogger.reset()
        PraxisLogger.instance().configure_defaults()

        prices = u[ticker]
        config = ModelConfig(
            model=ModelIdentity(name=f"r2_{ticker}_sma", version="v1.0"),
            signal={"method": "sma_crossover", "fast_period": 20, "slow_period": 50},
            sizing={"method": "fixed_fraction", "fraction": 1.0},
        )

        runner = PraxisRunner()
        result = runner.run_config(config, prices)
        m = result.metrics

        ticker_results[ticker] = {
            "rows": len(prices),
            "total_return": round(float(m["total_return"]), 6),
            "sharpe": round(float(m["sharpe_ratio"]), 6),
            "max_dd": round(float(m["max_drawdown"]), 6),
            "trades": int(m["total_trades"]),
        }

    return {
        "n_tickers": len(ticker_results),
        "tickers": ticker_results,
    }


# ==============================================================
#  MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("  McTheory Praxis — Battle Test Round 2.1 (Stress)")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Universe: {len(UNIVERSE_TICKERS)} tickers, {UNIVERSE_START} to {UNIVERSE_END}")
    print("=" * 60)
    print()

    print("SETUP: Fetching 3-year 12-asset universe...")
    get_universe()
    print()

    print("PHASE A: CPO Dense Sweep (105 combos)")
    run_test("A1_dense_sweep", test_A1_dense_sweep)
    print()

    print("PHASE B: CPO Training (2000+ rows)")
    run_test("B1_large_training", test_B1_large_training)
    print()

    print("PHASE C: CPO Predictor (RF on 2000+ rows)")
    run_test("C1_predictor_proper", test_C1_predictor_proper)
    print()

    print("PHASE D: Burgess Stat Arb (3-year, relaxed)")
    run_test("D1_candidates_3yr", test_D1_burgess_candidates_3yr)
    run_test("D2_pipeline_relaxed", test_D2_burgess_pipeline_relaxed)
    print()

    print("PHASE E: Portfolio Optimization (all methods)")
    run_test("E1_portfolio_all_methods", test_E1_portfolio_all_methods)
    print()

    print("PHASE F: Regression (stepwise + ridge)")
    run_test("F1_full_stepwise", test_F1_full_stepwise)
    run_test("F2_ridge_spy", test_F2_ridge_spy)
    print()

    print("PHASE G: Backtests (3-year captured)")
    run_test("G1_sma_3yr", test_G1_sma_3yr)
    run_test("G2_multi_ticker", test_G2_multi_ticker_backtest)
    print()

    # ── Summary ───────────────────────────────────────────────
    total = len(results["tests"])
    passed = sum(1 for t in results["tests"].values() if t["status"] == "PASS")
    failed = total - passed

    results["summary"] = {
        "total": total, "passed": passed, "failed": failed,
        "pass_rate": f"{passed/total*100:.0f}%" if total > 0 else "N/A",
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if error_lines:
        with open(ERRORS_LOG, "w") as f:
            f.writelines(error_lines)

    total_time = sum(t["elapsed_seconds"] for t in results["tests"].values())
    print("=" * 60)
    print(f"  RESULTS: {passed}/{total} passed  ({total_time:.1f}s total compute)")
    if failed:
        for name, info in results["tests"].items():
            if info["status"] == "FAIL":
                print(f"    ✗ {name}: {info['error'][:120]}")
    captures = list(OUTPUT_DIR.glob("capture_*.txt"))
    print(f"\n  Output: {OUTPUT_DIR}")
    if captures:
        for c in captures:
            size_kb = c.stat().st_size / 1024
            print(f"    {c.name} ({size_kb:.0f} KB)")
    print(f"\n  Upload battle_results/round2_1/ to Claude.")
    print("=" * 60)


if __name__ == "__main__":
    main()
