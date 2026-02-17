#!/usr/bin/env python3
"""
Battle Test Suite — Round 2: CPO + Burgess + Portfolio + Regression
====================================================================

PREREQUISITES:
    Round 1 must have been run first (capture files needed).

RUN:
    cd mctheory_praxis
    python scripts/battle_test_round2.py

OUTPUT:
    battle_results/round2/
        capture_round2.txt          — raw vendor data for new tickers
        results_round2.json         — structured test results
        errors.log                  — any errors encountered

GIVE BACK TO CLAUDE:
    Upload the entire battle_results/round2/ directory.
"""

import sys
import os
import json
import time
import hashlib
import traceback
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "battle_results" / "round2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROUND1_DIR = PROJECT_ROOT / "battle_results" / "round1"

ERRORS_LOG = OUTPUT_DIR / "errors.log"
RESULTS_FILE = OUTPUT_DIR / "results_round2.json"

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
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import numpy as np
import pandas as pd
import polars as pl

results = {
    "meta": {
        "script": "battle_test_round2.py",
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
        print(f"FAIL: {e}")


# ==============================================================
#  SHARED: Capture broader universe + load Round 1 data
# ==============================================================

# Extra tickers for Burgess multi-asset universe
UNIVERSE_TICKERS = ["AAPL", "MSFT", "GLD", "GDX", "SPY", "QQQ",
                    "TLT", "SLV", "XLE", "XLF", "IWM", "EEM"]
UNIVERSE_START = "2023-01-01"
UNIVERSE_END = "2024-01-01"

_universe_prices = None  # Lazy cached

def get_universe_prices():
    """Fetch and cache full universe — captured for Claude replay."""
    global _universe_prices
    if _universe_prices is not None:
        return _universe_prices

    from praxis.logger.core import PraxisLogger
    from praxis.logger.vendor_capture import start_capture_session, end_capture_session
    from praxis.data import fetch_prices

    PraxisLogger.reset()
    log = PraxisLogger.instance()
    log.configure_defaults()

    adapter = start_capture_session(capture_dir=OUTPUT_DIR, session_id="round2")

    _universe_prices = {}
    for ticker in UNIVERSE_TICKERS:
        try:
            df = fetch_prices(ticker, start=UNIVERSE_START, end=UNIVERSE_END)
            _universe_prices[ticker] = df
        except Exception as e:
            print(f"    Warning: failed to fetch {ticker}: {e}")

    summary = end_capture_session()
    print(f"    Captured {summary['calls_captured']} vendor calls")
    return _universe_prices


def get_pair_arrays(ticker_a="GLD", ticker_b="GDX"):
    """Get OHLCV arrays for a pair from universe."""
    u = get_universe_prices()
    a, b = u[ticker_a], u[ticker_b]
    min_len = min(len(a), len(b))
    close_a = a["close"].to_numpy()[:min_len].astype(float)
    open_a = a["open"].to_numpy()[:min_len].astype(float) if "open" in a.columns else close_a.copy()
    close_b = b["close"].to_numpy()[:min_len].astype(float)
    return close_a, open_a, close_b, min_len


def build_price_matrix():
    """Build (n_obs, n_assets) price matrix from universe."""
    u = get_universe_prices()
    tickers = [t for t in UNIVERSE_TICKERS if t in u]
    min_len = min(len(u[t]) for t in tickers)
    matrix = np.column_stack([u[t]["close"].to_numpy()[:min_len].astype(float) for t in tickers])
    return matrix, tickers, min_len


# ==============================================================
#  PHASE A: CPO Single-Leg Execution
# ==============================================================

def test_A1_single_leg_gld_gdx():
    """Single-leg execution on real GLD/GDX data."""
    from praxis.cpo import execute_single_leg

    close_a, open_a, close_b, n = get_pair_arrays("GLD", "GDX")

    params = {"weight": 3.0, "lookback": 60, "entry_threshold": 1.0,
              "exit_threshold_fraction": -0.6}
    result = execute_single_leg(close_a, open_a, close_b, params)

    return {
        "n_bars": n,
        "sharpe_ratio": round(float(result.sharpe_ratio), 6),
        "daily_return": round(float(result.daily_return), 8),
        "annualized_return": round(float(result.annualized_return), 6),
        "volatility": round(float(result.volatility), 6),
        "num_trades": int(result.num_trades),
        "positions_unique": sorted(set(result.positions.astype(int).tolist())),
        "positions_long_pct": round(float((result.positions > 0).sum() / n), 4),
        "positions_short_pct": round(float((result.positions < 0).sum() / n), 4),
        "pnl_sum": round(float(result.pnl.sum()), 8),
        "pnl_mean": round(float(result.pnl.mean()), 8),
    }


def test_A2_single_leg_param_sweep():
    """Sweep weight × entry_threshold — verify Sharpe varies monotonically-ish."""
    from praxis.cpo import execute_single_leg

    close_a, open_a, close_b, n = get_pair_arrays("GLD", "GDX")
    sweep = {}

    for weight in [2.0, 3.0, 5.0]:
        for entry in [0.5, 1.0, 2.0]:
            params = {"weight": weight, "lookback": 60, "entry_threshold": entry,
                      "exit_threshold_fraction": -0.6}
            r = execute_single_leg(close_a, open_a, close_b, params)
            key = f"w{weight}_e{entry}"
            sweep[key] = {
                "sharpe": round(float(r.sharpe_ratio), 6),
                "trades": int(r.num_trades),
                "return": round(float(r.annualized_return), 6),
            }

    return {"n_bars": n, "sweep": sweep}


# ==============================================================
#  PHASE B: CPO Training Data Generation
# ==============================================================

def test_B1_training_data():
    """Generate CPO training data over real GLD/GDX — full param grid."""
    from praxis.cpo import generate_training_data

    close_a, open_a, close_b, n = get_pair_arrays("GLD", "GDX")
    dates = np.arange(100, n - 10, 50)  # Every 50 bars from bar 100

    param_grid = {
        "weights": [2.0, 3.0, 5.0],
        "entry_thresholds": [0.5, 1.0, 2.0],
        "lookbacks": [30, 60],
    }

    training = generate_training_data(
        close_a, open_a, close_b, dates, param_grid,
        transaction_costs=0.0005,
    )

    df = training.to_polars()

    return {
        "total_rows": training.count,
        "errors": len(training.errors),
        "n_dates": len(dates),
        "grid_size": 3 * 3 * 2,  # weights × entries × lookbacks
        "expected_max": len(dates) * 18,
        "columns": df.columns if len(df) > 0 else [],
        "sharpe_mean": round(float(df["sharpe_ratio"].mean()), 6) if len(df) > 0 else None,
        "sharpe_std": round(float(df["sharpe_ratio"].std()), 6) if len(df) > 0 else None,
        "sharpe_min": round(float(df["sharpe_ratio"].min()), 6) if len(df) > 0 else None,
        "sharpe_max": round(float(df["sharpe_ratio"].max()), 6) if len(df) > 0 else None,
        "num_trades_mean": round(float(df["num_trades"].mean()), 2) if len(df) > 0 else None,
    }


# ==============================================================
#  PHASE C: CPO Predictor (RandomForest)
# ==============================================================

def test_C1_cpo_predictor():
    """Fit CPO predictor on real training data, predict best params."""
    from praxis.cpo import generate_training_data, CPOPredictor

    close_a, open_a, close_b, n = get_pair_arrays("GLD", "GDX")
    dates = np.arange(100, n - 10, 30)

    param_grid = {
        "weights": [2.0, 2.5, 3.0, 4.0, 5.0],
        "entry_thresholds": [0.5, 0.75, 1.0, 1.5, 2.0],
        "lookbacks": [30, 60, 90],
    }

    training = generate_training_data(close_a, open_a, close_b, dates, param_grid)
    df = training.to_polars()

    predictor = CPOPredictor(n_estimators=100, random_state=42)
    fit_metrics = predictor.fit(df, train_fraction=0.8)

    # Predict best from candidate grid
    candidates = df.select(["weight", "entry_threshold", "lookback"]).unique()
    prediction = predictor.predict_best_params(candidates)

    return {
        "training_rows": training.count,
        "mse_train": round(float(fit_metrics["mse_train"]), 8),
        "mse_test": round(float(fit_metrics["mse_test"]), 8),
        "r2_train": round(float(fit_metrics["r2_train"]), 6),
        "predicted_weight": round(float(prediction.predicted_params["weight"]), 4),
        "predicted_entry": round(float(prediction.predicted_params["entry_threshold"]), 4),
        "predicted_lookback": round(float(prediction.predicted_params["lookback"]), 4),
        "predicted_sharpe": round(float(prediction.predicted_sharpe), 6),
        "model_score": round(float(prediction.model_score), 6),
    }


# ==============================================================
#  PHASE D: Burgess Stat Arb Pipeline
# ==============================================================

def test_D1_burgess_candidates():
    """Generate cointegration candidates from real 12-asset universe."""
    from praxis.models.burgess import generate_candidates

    matrix, tickers, n_obs = build_price_matrix()

    candidates = generate_candidates(
        matrix,
        n_per_basket=3,
        max_candidates=0,  # scan all
        significance=0.10,  # relaxed for real data
    )

    return {
        "n_assets": len(tickers),
        "n_obs": n_obs,
        "tickers": tickers,
        "n_candidates": len(candidates),
        "top_5": [
            {
                "indices": c.all_indices,
                "adf_t": round(float(c.adf_t_statistic), 6),
                "pvalue": round(float(c.adf_p_value), 6),
                "half_life": round(float(c.half_life_periods), 4) if c.half_life_periods and np.isfinite(c.half_life_periods) else None,
                "hurst": round(float(c.hurst), 6) if c.hurst and np.isfinite(c.hurst) else None,
            }
            for c in candidates[:5]
        ] if candidates else [],
    }


def test_D2_burgess_full_pipeline():
    """Full Burgess pipeline: candidates → MC → filter → optimize."""
    from praxis.models.burgess import BurgessStatArb, BurgessConfig

    matrix, tickers, n_obs = build_price_matrix()

    config = BurgessConfig(
        n_per_basket=3,
        significance=0.10,
        mc_enabled=True,
        mc_n_samples=200,  # Keep low for speed
        mc_seed=42,
        top_k=10,
        min_half_life=1.0,
        max_half_life=252.0,
        max_hurst=0.6,  # Relaxed for real data
        optimization_method="min_variance",
    )

    engine = BurgessStatArb(config)
    result = engine.run(matrix)

    return {
        "n_assets": len(tickers),
        "n_obs": n_obs,
        "n_scanned": result.n_scanned,
        "n_selected": result.n_candidates,
        "has_critical_values": result.critical_values is not None,
        "n_portfolio_results": len(result.portfolio_results),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "selected_top3": [
            {
                "indices": b.all_indices,
                "adf_t": round(float(b.adf_t_statistic), 6),
                "pvalue": round(float(b.adf_p_value), 6),
            }
            for b in result.selected_baskets[:3]
        ] if result.selected_baskets else [],
        "portfolio_top3": [
            {
                "weights": [round(float(w), 6) for w in p.weights],
                "expected_return": round(float(p.expected_return), 6) if p.expected_return else None,
                "volatility": round(float(p.volatility), 6) if p.volatility else None,
            }
            for p in result.portfolio_results[:3]
        ] if result.portfolio_results else [],
    }


# ==============================================================
#  PHASE E: Portfolio Optimization
# ==============================================================

def test_E1_min_variance():
    """Min-variance portfolio on real 12-asset universe."""
    from praxis.stats.portfolio import min_variance_portfolio, covariance_matrix

    matrix, tickers, n_obs = build_price_matrix()
    # Convert prices to returns
    returns = np.diff(np.log(matrix), axis=0)

    cov = covariance_matrix(returns)
    result = min_variance_portfolio(cov)

    return {
        "n_assets": len(tickers),
        "tickers": tickers,
        "weights": [round(float(w), 6) for w in result.weights],
        "weights_sum": round(float(np.sum(result.weights)), 6),
        "volatility": round(float(result.volatility), 6),
        "expected_return": round(float(result.expected_return), 6) if result.expected_return else None,
    }


def test_E2_max_sharpe():
    """Max-Sharpe portfolio on real universe."""
    from praxis.stats.portfolio import max_sharpe_portfolio, covariance_matrix

    matrix, tickers, n_obs = build_price_matrix()
    returns = np.diff(np.log(matrix), axis=0)

    cov = covariance_matrix(returns)
    mu = np.mean(returns, axis=0)
    result = max_sharpe_portfolio(mu, cov, risk_free_rate=0.0)

    return {
        "n_assets": len(tickers),
        "weights": [round(float(w), 6) for w in result.weights],
        "weights_sum": round(float(np.sum(result.weights)), 6),
        "volatility": round(float(result.volatility), 6),
        "expected_return": round(float(result.expected_return), 6) if result.expected_return else None,
        "sharpe_ratio": round(float(result.sharpe_ratio), 6) if result.sharpe_ratio else None,
    }


# ==============================================================
#  PHASE F: Regression (Ridge + Successive)
# ==============================================================

def test_F1_ridge_regression():
    """Ridge regression on real universe returns."""
    from praxis.stats.regression import ridge_regression

    matrix, tickers, n_obs = build_price_matrix()
    returns = np.diff(np.log(matrix), axis=0)

    # Predict SPY returns from other assets
    spy_idx = tickers.index("SPY") if "SPY" in tickers else 0
    y = returns[:, spy_idx]
    X = np.delete(returns, spy_idx, axis=1)
    other_tickers = [t for i, t in enumerate(tickers) if i != spy_idx]

    result = ridge_regression(y, X, alpha=1.0)

    return {
        "target": tickers[spy_idx],
        "features": other_tickers,
        "beta_values": {t: round(float(c), 6) for t, c in zip(other_tickers, result.beta[1:])},
        "intercept_beta": round(float(result.beta[0]), 8),
        "r_squared": round(float(result.r_squared), 6),
        "adj_r_squared": round(float(result.adj_r_squared), 6),
        "residual_std": round(float(np.std(result.residuals)), 8),
    }


def test_F2_successive_regression():
    """Successive regression (Burgess-style stepwise) on real universe."""
    from praxis.stats.regression import successive_regression

    matrix, tickers, n_obs = build_price_matrix()

    # Run stepwise for first 4 targets, finding 3 partners each
    step_results = []
    for target_idx in range(min(4, len(tickers))):
        result = successive_regression(
            target_index=target_idx,
            asset_matrix=matrix,
            n_vars=3,
            compute_stats=True,
        )
        entry = {
            "target": tickers[target_idx],
            "selected_indices": result.selected_indices,
            "selected_tickers": [tickers[i] for i in result.selected_indices],
            "is_stationary": result.is_stationary,
        }
        if result.regression:
            entry["r_squared"] = round(float(result.regression.r_squared), 6)
        if result.adf:
            entry["adf_t"] = round(float(result.adf.t_statistic), 6)
            entry["adf_pvalue"] = round(float(result.adf.p_value), 6)
        step_results.append(entry)

    return {
        "n_assets": len(tickers),
        "n_targets_tested": len(step_results),
        "n_vars": 3,
        "results": step_results,
    }


# ==============================================================
#  PHASE G: Backtest Pipeline with Captured Data (Round 1 fixes)
# ==============================================================

def test_G1_sma_backtest_captured():
    """SMA crossover on captured AAPL — uses Round 2 capture."""
    from praxis.config import ModelConfig, ModelIdentity
    from praxis.runner import PraxisRunner
    from praxis.logger.core import PraxisLogger

    PraxisLogger.reset()
    PraxisLogger.instance().configure_defaults()

    u = get_universe_prices()
    prices = u["AAPL"]

    config = ModelConfig(
        model=ModelIdentity(name="round2_sma_captured", version="v1.0"),
        signal={"method": "sma_crossover", "fast_period": 10, "slow_period": 30},
        sizing={"method": "fixed_fraction", "fraction": 0.1},
        backtest={"initial_capital": 100_000, "commission_bps": 10},
    )

    runner = PraxisRunner()
    result = runner.run_config(config, prices)
    m = result.metrics

    return {
        "total_return": round(float(m["total_return"]), 6),
        "sharpe_ratio": round(float(m["sharpe_ratio"]), 6),
        "max_drawdown": round(float(m["max_drawdown"]), 6),
        "total_trades": int(m["total_trades"]),
        "win_rate": round(float(m["win_rate"]), 6),
        "data_rows": len(prices),
    }


def test_G2_event_vs_vectorized_captured():
    """Event vs vectorized backtest on captured AAPL."""
    from praxis.config import ModelConfig, ModelIdentity
    from praxis.runner import PraxisRunner
    from praxis.backtest.event_driven import EventDrivenEngine
    from praxis.logger.core import PraxisLogger

    PraxisLogger.reset()
    PraxisLogger.instance().configure_defaults()

    u = get_universe_prices()
    prices = u["AAPL"]
    close = prices["close"].to_numpy()

    config = ModelConfig(
        model=ModelIdentity(name="round2_event_cmp", version="v1.0"),
        signal={"method": "sma_crossover", "fast_period": 10, "slow_period": 30},
        sizing={"method": "fixed_fraction", "fraction": 1.0},
        backtest={"initial_capital": 100_000, "commission_bps": 0},
    )

    runner = PraxisRunner()
    vec_result = runner.run_config(config, prices)

    positions = vec_result.positions.to_numpy() if vec_result.positions is not None else np.zeros(len(close))
    engine = EventDrivenEngine(commission_per_trade=0.0)
    ed_output = engine.run(positions, close, initial_capital=100_000.0)

    return {
        "vectorized": {
            "total_return": round(float(vec_result.metrics["total_return"]), 6),
            "total_trades": int(vec_result.metrics["total_trades"]),
            "sharpe": round(float(vec_result.metrics["sharpe_ratio"]), 6),
        },
        "event_driven": {
            "total_return": round(float(ed_output.metrics.total_return), 6),
            "total_trades": int(ed_output.metrics.total_trades),
            "sharpe": round(float(ed_output.metrics.sharpe_ratio), 6),
        },
        "delta_return": round(abs(float(vec_result.metrics["total_return"]) - float(ed_output.metrics.total_return)), 8),
    }


# ==============================================================
#  MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("  McTheory Praxis — Battle Test Round 2")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    print()

    # Pre-fetch universe (captured)
    print("SETUP: Fetching 12-asset universe with vendor capture...")
    get_universe_prices()
    print()

    print("PHASE A: CPO Single-Leg Execution")
    run_test("A1_single_leg_gld_gdx", test_A1_single_leg_gld_gdx)
    run_test("A2_single_leg_sweep", test_A2_single_leg_param_sweep)
    print()

    print("PHASE B: CPO Training Data")
    run_test("B1_training_data", test_B1_training_data)
    print()

    print("PHASE C: CPO Predictor")
    run_test("C1_cpo_predictor", test_C1_cpo_predictor)
    print()

    print("PHASE D: Burgess Stat Arb")
    run_test("D1_burgess_candidates", test_D1_burgess_candidates)
    run_test("D2_burgess_full_pipeline", test_D2_burgess_full_pipeline)
    print()

    print("PHASE E: Portfolio Optimization")
    run_test("E1_min_variance", test_E1_min_variance)
    run_test("E2_max_sharpe", test_E2_max_sharpe)
    print()

    print("PHASE F: Regression")
    run_test("F1_ridge_regression", test_F1_ridge_regression)
    run_test("F2_successive_regression", test_F2_successive_regression)
    print()

    print("PHASE G: Backtest (Captured Data)")
    run_test("G1_sma_backtest_captured", test_G1_sma_backtest_captured)
    run_test("G2_event_vs_vectorized", test_G2_event_vs_vectorized_captured)
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

    print("=" * 60)
    print(f"  RESULTS: {passed}/{total} passed")
    if failed:
        for name, info in results["tests"].items():
            if info["status"] == "FAIL":
                print(f"    ✗ {name}: {info['error'][:100]}")
    print(f"\n  Output: {OUTPUT_DIR}")
    captures = list(OUTPUT_DIR.glob("capture_*.txt"))
    if captures:
        print(f"  Captures: {len(captures)} files")
        for c in captures:
            print(f"    {c}")
    print(f"\n  Upload battle_results/round2/ to Claude.")
    print("=" * 60)


if __name__ == "__main__":
    main()
