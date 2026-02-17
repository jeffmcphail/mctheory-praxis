#!/usr/bin/env python3
"""
Battle Test Suite — Round 1: Data Fetch + Basic Pipeline
=========================================================

RUN:
    cd mctheory_praxis
    python scripts/battle_test_round1.py

OUTPUT:
    battle_results/round1/  (entire directory)
        capture_round1.txt          — raw vendor data for Claude replay
        results_round1.json         — structured test results for comparison
        errors.log                  — any errors encountered

GIVE BACK TO CLAUDE:
    Zip and upload the entire battle_results/round1/ directory.
    Or just upload capture_round1.txt and results_round1.json.
"""

import sys
import os
import json
import time
import hashlib
import traceback
from pathlib import Path
from datetime import datetime, timezone

# ── Ensure src on path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "battle_results" / "round1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERRORS_LOG = OUTPUT_DIR / "errors.log"
RESULTS_FILE = OUTPUT_DIR / "results_round1.json"

# ── Dependency check ──────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["numpy", "pandas", "polars", "pyarrow", "yfinance", "duckdb", "pydantic"]:
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

# ── Test harness ──────────────────────────────────────────────
results = {
    "meta": {
        "script": "battle_test_round1.py",
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
    """Run a test function, record pass/fail and result."""
    print(f"  [{name}] ... ", end="", flush=True)
    t0 = time.monotonic()
    try:
        result = fn()
        elapsed = time.monotonic() - t0
        results["tests"][name] = {
            "status": "PASS",
            "elapsed_seconds": round(elapsed, 3),
            "result": result,
        }
        print(f"PASS ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc()
        results["tests"][name] = {
            "status": "FAIL",
            "elapsed_seconds": round(elapsed, 3),
            "error": str(e),
        }
        error_lines.append(f"=== {name} ===\n{tb}\n")
        print(f"FAIL: {e}")


# ==============================================================
#  PHASE A: Vendor Data Capture
# ==============================================================

def test_A1_capture_yfinance_equities():
    """Fetch AAPL, MSFT, GLD, GDX — 1 year daily."""
    from praxis.logger.core import PraxisLogger
    from praxis.logger.vendor_capture import start_capture_session, end_capture_session
    from praxis.data import fetch_prices

    log = PraxisLogger.instance()
    log.configure_defaults()

    adapter = start_capture_session(
        capture_dir=OUTPUT_DIR,
        session_id="round1",
    )

    tickers = ["AAPL", "MSFT", "GLD", "GDX"]
    prices = fetch_prices(tickers, start="2023-01-01", end="2024-01-01")

    summary = end_capture_session()

    # Compute fingerprints for reconciliation
    ticker_results = {}
    if "ticker" in prices.columns:
        for t in tickers:
            chunk = prices.filter(pl.col("ticker") == t)
            ticker_results[t] = {
                "rows": len(chunk),
                "first_date": str(chunk["date"].min()),
                "last_date": str(chunk["date"].max()),
                "first_close": round(float(chunk["close"][0]), 6),
                "last_close": round(float(chunk["close"][-1]), 6),
                "mean_close": round(float(chunk["close"].mean()), 6),
                "sum_volume": int(chunk["volume"].sum()),
            }
    else:
        # Single ticker result
        ticker_results[tickers[0]] = {
            "rows": len(prices),
            "first_date": str(prices["date"].min()),
            "last_date": str(prices["date"].max()),
            "first_close": round(float(prices["close"][0]), 6),
            "last_close": round(float(prices["close"][-1]), 6),
            "mean_close": round(float(prices["close"].mean()), 6),
            "sum_volume": int(prices["volume"].sum()),
        }

    return {
        "calls_captured": summary["calls_captured"],
        "capture_file": summary["file_path"],
        "total_rows": len(prices),
        "columns": prices.columns,
        "tickers": ticker_results,
    }


def test_A2_capture_crypto_proxies():
    """Fetch BTC-USD, ETH-USD — 6 months daily."""
    from praxis.logger.core import PraxisLogger
    from praxis.logger.vendor_capture import start_capture_session, end_capture_session
    from praxis.data import fetch_prices

    PraxisLogger.reset()
    log = PraxisLogger.instance()
    log.configure_defaults()

    adapter = start_capture_session(
        capture_dir=OUTPUT_DIR,
        session_id="round1_crypto",
    )

    tickers = ["BTC-USD", "ETH-USD"]
    prices = fetch_prices(tickers, start="2024-01-01", end="2024-07-01")

    summary = end_capture_session()

    ticker_results = {}
    if "ticker" in prices.columns:
        for t in tickers:
            chunk = prices.filter(pl.col("ticker") == t)
            ticker_results[t] = {
                "rows": len(chunk),
                "first_close": round(float(chunk["close"][0]), 6),
                "last_close": round(float(chunk["close"][-1]), 6),
                "mean_close": round(float(chunk["close"].mean()), 6),
            }

    return {
        "calls_captured": summary["calls_captured"],
        "total_rows": len(prices),
        "tickers": ticker_results,
    }


# ==============================================================
#  PHASE B: Config → Runner → Backtest (SMA Crossover)
# ==============================================================

def test_B1_sma_crossover_backtest():
    """Full praxis run: SMA crossover on AAPL."""
    from praxis.logger.core import PraxisLogger
    from praxis.data import fetch_prices
    from praxis.config import ModelConfig, ModelIdentity
    from praxis.runner import PraxisRunner

    PraxisLogger.reset()
    log = PraxisLogger.instance()
    log.configure_defaults()

    prices = fetch_prices("AAPL", start="2022-01-01", end="2024-01-01")

    config = ModelConfig(
        model=ModelIdentity(name="battle_test_sma", version="v1.0"),
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
        "annualized_return": round(float(m["annualized_return"]), 6),
        "data_rows": len(prices),
        "first_date": str(prices["date"].min()),
        "last_date": str(prices["date"].max()),
    }


def test_B2_ema_crossover_backtest():
    """EMA crossover on MSFT — different signal, same pipeline."""
    from praxis.logger.core import PraxisLogger
    from praxis.data import fetch_prices
    from praxis.config import ModelConfig, ModelIdentity
    from praxis.runner import PraxisRunner

    PraxisLogger.reset()
    log = PraxisLogger.instance()
    log.configure_defaults()

    prices = fetch_prices("MSFT", start="2022-01-01", end="2024-01-01")

    config = ModelConfig(
        model=ModelIdentity(name="battle_test_ema", version="v1.0"),
        signal={"method": "ema_crossover", "fast_period": 12, "slow_period": 26},
        sizing={"method": "fixed_fraction", "fraction": 0.2},
        backtest={"initial_capital": 100_000, "commission_bps": 5},
    )

    runner = PraxisRunner()
    result = runner.run_config(config, prices)

    m = result.metrics
    return {
        "total_return": round(float(m["total_return"]), 6),
        "sharpe_ratio": round(float(m["sharpe_ratio"]), 6),
        "max_drawdown": round(float(m["max_drawdown"]), 6),
        "total_trades": int(m["total_trades"]),
        "annualized_return": round(float(m["annualized_return"]), 6),
    }


# ==============================================================
#  PHASE C: Signal & Indicator Computations
# ==============================================================

def test_C1_zscore_signal():
    """Z-score spread signal on GLD/GDX pair."""
    from praxis.logger.core import PraxisLogger
    from praxis.data import fetch_prices
    from praxis.registry import FunctionRegistry

    PraxisLogger.reset()
    PraxisLogger.instance().configure_defaults()

    gld = fetch_prices("GLD", start="2023-01-01", end="2024-01-01")
    gdx = fetch_prices("GDX", start="2023-01-01", end="2024-01-01")

    # Align dates
    gld_close = gld["close"].to_numpy()
    gdx_close = gdx["close"].to_numpy()
    min_len = min(len(gld_close), len(gdx_close))
    gld_close = gld_close[:min_len]
    gdx_close = gdx_close[:min_len]

    # Compute z-score spread manually for validation
    ratio = gld_close / gdx_close
    window = 60
    rolling_mean = pd.Series(ratio).rolling(window).mean().to_numpy()
    rolling_std = pd.Series(ratio).rolling(window).std().to_numpy()
    zscore = (ratio - rolling_mean) / rolling_std

    # Remove NaN
    valid = ~np.isnan(zscore)
    zscore_valid = zscore[valid]

    return {
        "pair": "GLD/GDX",
        "data_points": int(min_len),
        "zscore_valid_points": int(len(zscore_valid)),
        "zscore_mean": round(float(np.mean(zscore_valid)), 6),
        "zscore_std": round(float(np.std(zscore_valid)), 6),
        "zscore_min": round(float(np.min(zscore_valid)), 6),
        "zscore_max": round(float(np.max(zscore_valid)), 6),
        "zscore_first_5": [round(float(x), 6) for x in zscore_valid[:5]],
        "zscore_last_5": [round(float(x), 6) for x in zscore_valid[-5:]],
        "ratio_mean": round(float(np.mean(ratio)), 6),
        "ratio_std": round(float(np.std(ratio)), 6),
        "gld_first_close": round(float(gld_close[0]), 6),
        "gld_last_close": round(float(gld_close[-1]), 6),
        "gdx_first_close": round(float(gdx_close[0]), 6),
        "gdx_last_close": round(float(gdx_close[-1]), 6),
    }


def test_C2_indicators():
    """Compute 7 indicators on AAPL — fingerprint each."""
    from praxis.logger.core import PraxisLogger
    from praxis.data import fetch_prices
    from praxis.indicators import mfi, force_index, atr, _sma, _ema

    PraxisLogger.reset()
    PraxisLogger.instance().configure_defaults()

    prices = fetch_prices("AAPL", start="2023-01-01", end="2024-01-01")
    close = prices["close"].to_numpy()
    high = prices["high"].to_numpy()
    low = prices["low"].to_numpy()
    volume = prices["volume"].to_numpy().astype(float)

    indicator_results = {}

    # SMA
    vals = _sma(close, window=20)
    valid = vals[~np.isnan(vals)]
    indicator_results["sma_20"] = {"valid_points": len(valid), "mean": round(float(np.mean(valid)), 6), "last": round(float(valid[-1]), 6)}

    # EMA
    vals = _ema(close, span=20)
    valid = vals[~np.isnan(vals)]
    indicator_results["ema_20"] = {"valid_points": len(valid), "mean": round(float(np.mean(valid)), 6), "last": round(float(valid[-1]), 6)}

    # Bollinger Bands
    bb_mid = _sma(close, window=20)
    bb_std = pd.Series(close).rolling(20).std().to_numpy()
    upper = bb_mid + 2.0 * bb_std
    lower = bb_mid - 2.0 * bb_std
    valid_upper = upper[~np.isnan(upper)]
    indicator_results["bollinger"] = {
        "valid_points": len(valid_upper),
        "upper_last": round(float(valid_upper[-1]), 6),
        "lower_last": round(float(lower[~np.isnan(lower)][-1]), 6),
        "mid_last": round(float(bb_mid[~np.isnan(bb_mid)][-1]), 6),
    }

    # RSI (manual)
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gains).rolling(14).mean().to_numpy()
    avg_loss = pd.Series(losses).rolling(14).mean().to_numpy()
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_vals = 100 - 100 / (1 + rs)
    valid = rsi_vals[~np.isnan(rsi_vals)]
    indicator_results["rsi_14"] = {"valid_points": len(valid), "mean": round(float(np.mean(valid)), 6), "last": round(float(valid[-1]), 6)}

    # ATR
    vals = atr(high, low, close, window=14)
    valid = vals[~np.isnan(vals)]
    indicator_results["atr_14"] = {"valid_points": len(valid), "mean": round(float(np.mean(valid)), 6), "last": round(float(valid[-1]), 6)}

    # MFI
    vals = mfi(high, low, close, volume, length=14)
    valid = vals[~np.isnan(vals)]
    indicator_results["mfi_14"] = {"valid_points": len(valid), "mean": round(float(np.mean(valid)), 6), "last": round(float(valid[-1]), 6)}

    # Force Index
    vals = force_index(close, volume, window=13)
    valid = vals[~np.isnan(vals)]
    indicator_results["force_index_13"] = {"valid_points": len(valid), "mean": round(float(np.mean(valid)), 2), "last": round(float(valid[-1]), 2)}

    return {
        "ticker": "AAPL",
        "data_points": len(close),
        "indicators": indicator_results,
    }


# ==============================================================
#  PHASE D: Statistical Tests (ADF, Hurst, Half-life)
# ==============================================================

def test_D1_stat_tests_gld_gdx():
    """Run ADF, Hurst, half-life on GLD/GDX spread."""
    from praxis.logger.core import PraxisLogger
    from praxis.data import fetch_prices
    from praxis.stats import adf_test, hurst_exponent, half_life

    PraxisLogger.reset()
    PraxisLogger.instance().configure_defaults()

    gld = fetch_prices("GLD", start="2023-01-01", end="2024-01-01")
    gdx = fetch_prices("GDX", start="2023-01-01", end="2024-01-01")

    gld_close = gld["close"].to_numpy()
    gdx_close = gdx["close"].to_numpy()
    min_len = min(len(gld_close), len(gdx_close))
    gld_close = gld_close[:min_len]
    gdx_close = gdx_close[:min_len]

    # OLS hedge ratio
    beta = np.polyfit(gdx_close, gld_close, 1)[0]
    spread = gld_close - beta * gdx_close

    adf_result = adf_test(spread)
    hurst = hurst_exponent(spread)
    hl = half_life(spread)

    return {
        "pair": "GLD/GDX",
        "hedge_ratio": round(float(beta), 6),
        "spread_mean": round(float(np.mean(spread)), 6),
        "spread_std": round(float(np.std(spread)), 6),
        "adf_t_statistic": round(float(adf_result.t_statistic), 6),
        "adf_pvalue": round(float(adf_result.p_value), 6),
        "adf_is_stationary": bool(adf_result.is_stationary),
        "hurst_exponent": round(float(hurst.hurst_exponent), 6),
        "hurst_interpretation": hurst.interpretation,
        "half_life": round(float(hl.half_life), 6),
        "half_life_is_mean_reverting": bool(hl.is_mean_reverting),
    }


# ==============================================================
#  PHASE E: Event-Driven Backtest
# ==============================================================

def test_E1_event_driven_sma():
    """Event-driven backtest on AAPL — compare to vectorized."""
    from praxis.logger.core import PraxisLogger
    from praxis.data import fetch_prices
    from praxis.config import ModelConfig, ModelIdentity
    from praxis.runner import PraxisRunner
    from praxis.backtest.event_driven import EventDrivenEngine

    PraxisLogger.reset()
    PraxisLogger.instance().configure_defaults()

    prices = fetch_prices("AAPL", start="2023-01-01", end="2024-01-01")
    close = prices["close"].to_numpy()

    # Vectorized path (via runner)
    config = ModelConfig(
        model=ModelIdentity(name="event_vs_vector", version="v1.0"),
        signal={"method": "sma_crossover", "fast_period": 10, "slow_period": 30},
        sizing={"method": "fixed_fraction", "fraction": 1.0},
        backtest={"initial_capital": 100_000, "commission_bps": 0},
    )

    runner = PraxisRunner()
    vec_result = runner.run_config(config, prices)

    # Event-driven path (using same positions from runner)
    positions = vec_result.positions.to_numpy() if vec_result.positions is not None else np.zeros(len(close))
    engine = EventDrivenEngine(commission_per_trade=0.0)
    ed_output = engine.run(positions, close, initial_capital=100_000.0)

    return {
        "vectorized": {
            "total_return": round(float(vec_result.metrics["total_return"]), 6),
            "total_trades": int(vec_result.metrics["total_trades"]),
            "sharpe_ratio": round(float(vec_result.metrics["sharpe_ratio"]), 6),
        },
        "event_driven": {
            "total_return": round(float(ed_output.metrics.total_return), 6),
            "total_trades": int(ed_output.metrics.total_trades),
            "sharpe_ratio": round(float(ed_output.metrics.sharpe_ratio), 6),
        },
        "delta_return": round(abs(float(vec_result.metrics["total_return"]) - float(ed_output.metrics.total_return)), 8),
    }


# ==============================================================
#  PHASE F: Crypto Cointegration
# ==============================================================

def test_F1_crypto_cointegration():
    """Cointegration test on BTC/ETH."""
    from praxis.logger.core import PraxisLogger
    from praxis.data import fetch_prices
    from praxis.onchain.crypto import CryptoCointegrationAnalyzer, CryptoCointegrationConfig

    PraxisLogger.reset()
    PraxisLogger.instance().configure_defaults()

    btc = fetch_prices("BTC-USD", start="2024-01-01", end="2024-07-01")
    eth = fetch_prices("ETH-USD", start="2024-01-01", end="2024-07-01")

    btc_close = btc["close"].to_numpy()
    eth_close = eth["close"].to_numpy()
    min_len = min(len(btc_close), len(eth_close))
    btc_close = btc_close[:min_len]
    eth_close = eth_close[:min_len]

    analyzer = CryptoCointegrationAnalyzer(CryptoCointegrationConfig(
        zscore_window=30,
        min_correlation=0.5,
    ))

    coint = analyzer.test_cointegration(btc_close, eth_close)
    signal = analyzer.compute_signal(btc_close, eth_close, asset_a="BTC", asset_b="ETH")

    return {
        "pair": "BTC/ETH",
        "data_points": int(min_len),
        "is_cointegrated": bool(coint.is_cointegrated),
        "adf_pvalue": round(float(coint.adf_pvalue), 6),
        "correlation": round(float(coint.correlation), 6),
        "hedge_ratio": round(float(coint.hedge_ratio), 6),
        "half_life": round(float(coint.half_life), 6) if coint.half_life else None,
        "signal_zscore": round(float(signal.z_score), 6),
        "signal_should_trade": bool(signal.should_trade),
        "signal_direction": signal.direction if hasattr(signal, 'direction') else None,
    }


# ==============================================================
#  MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("  McTheory Praxis — Battle Test Round 1")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    print()

    # Phase A: Vendor data capture
    print("PHASE A: Vendor Data Capture")
    run_test("A1_yfinance_equities", test_A1_capture_yfinance_equities)
    run_test("A2_crypto_proxies", test_A2_capture_crypto_proxies)
    print()

    # Phase B: Config → Runner → Backtest
    print("PHASE B: Backtest Pipeline")
    run_test("B1_sma_crossover", test_B1_sma_crossover_backtest)
    run_test("B2_ema_crossover", test_B2_ema_crossover_backtest)
    print()

    # Phase C: Signals & Indicators
    print("PHASE C: Signals & Indicators")
    run_test("C1_zscore_gld_gdx", test_C1_zscore_signal)
    run_test("C2_indicators_aapl", test_C2_indicators)
    print()

    # Phase D: Statistical Tests
    print("PHASE D: Statistical Tests")
    run_test("D1_stat_tests_gld_gdx", test_D1_stat_tests_gld_gdx)
    print()

    # Phase E: Event-Driven Backtest
    print("PHASE E: Event-Driven Engine")
    run_test("E1_event_vs_vectorized", test_E1_event_driven_sma)
    print()

    # Phase F: Crypto Cointegration
    print("PHASE F: Crypto Cointegration")
    run_test("F1_btc_eth_cointegration", test_F1_crypto_cointegration)
    print()

    # ── Write results ─────────────────────────────────────────
    # Summary
    total = len(results["tests"])
    passed = sum(1 for t in results["tests"].values() if t["status"] == "PASS")
    failed = total - passed

    results["summary"] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed/total*100:.0f}%" if total > 0 else "N/A",
    }

    # Write JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Write errors
    if error_lines:
        with open(ERRORS_LOG, "w") as f:
            f.writelines(error_lines)

    # ── Print summary ─────────────────────────────────────────
    print("=" * 60)
    print(f"  RESULTS: {passed}/{total} passed")
    if failed:
        print(f"  FAILURES: {failed}")
        for name, info in results["tests"].items():
            if info["status"] == "FAIL":
                print(f"    - {name}: {info['error']}")
    print()
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Results: {RESULTS_FILE}")
    if error_lines:
        print(f"  Errors: {ERRORS_LOG}")
    print()

    # List capture files
    captures = list(OUTPUT_DIR.glob("capture_*.txt"))
    if captures:
        print(f"  Capture files ({len(captures)}):")
        for c in captures:
            print(f"    {c}")

    print()
    print("  Upload the entire battle_results/round1/ directory to Claude.")
    print("=" * 60)


if __name__ == "__main__":
    main()
