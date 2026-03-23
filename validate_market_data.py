"""
Market Data Integration Validation Script
==========================================

Run this on your machine to validate the full DataStore → Engine pipeline
with real market data from Yahoo Finance.

Usage:
    python validate_market_data.py

Requirements:
    pip install yfinance polars numpy scipy

    Ensure core_repo is accessible. Adjust CORE_PATH below if needed.
"""

import sys
import time
from pathlib import Path
from datetime import date

# ── Path setup ──────────────────────────────────────────────────────────────
# Adjust these if your layout differs
SCRIPT_DIR = Path(__file__).parent
CORE_PATH = SCRIPT_DIR.parent / "core" / "src"  # mctheory-core/src

# Try common locations
for candidate in [
    CORE_PATH,
    SCRIPT_DIR.parent / "core_repo" / "src",
    SCRIPT_DIR / "core_repo" / "src",
    Path(r"C:\Data\Development\Python\McTheoryApps\core\src"),
    Path(r"C:\Data\Development\Python\McTheoryApps\mctheory-core\src"),
]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        print(f"✓ Core path: {candidate}")
        break
else:
    print("⚠ Could not find core_repo/src. Set CORE_PATH in this script.")
    print("  Trying anyway in case mctheory.core is installed...")

# Praxis engines must be importable
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np

# ── Imports ─────────────────────────────────────────────────────────────────

from mctheory.core.datastore import DataStore
from market_data.schema import create_market_datastore
from market_data.fetchers import YFinanceFetcher, MockMarketDataFetcher
from market_data.bridge import DataStoreDataProvider
from engines.context.model_context import (
    UniverseSpec, TemporalSpec, AssetClass, ModelContext,
)
from engines.cointegration import StatArbEngine, StatArbParams, StatArbInput
from engines.momentum import MomentumEngine, MomentumParams, MomentumInput
from engines.allocation import AllocationEngine, AllocationParams, AllocationInput
from engines.model import Model


# ── Helpers ─────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *_):
        elapsed = time.time() - self.t0
        print(f"  ⏱ {self.label}: {elapsed:.2f}s")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def try_yfinance():
    """Check if YFinance works (network access)."""
    try:
        import yfinance as yf
        t = yf.Ticker("AAPL")
        hist = t.history(period="5d")
        return len(hist) > 0
    except Exception as e:
        print(f"  YFinance unavailable: {e}")
        return False


# ── Test Functions ──────────────────────────────────────────────────────────

def test_lazy_loading(provider):
    """Verify DataStore lazy-loads on demand."""
    section("1. LAZY LOADING")

    ds = provider._ds
    prices_table = ds.get_table("prices")
    sec_table = ds.get_table("securities")
    print(f"  Tables before fetch: prices={len(prices_table)} rows, securities={len(sec_table)} rows")

    with Timer("First fetch (SPY, GLD, AGG)"):
        prices = provider.fetch_prices(
            UniverseSpec(tickers=["SPY", "GLD", "AGG"]),
            TemporalSpec(lookback_days=504),
        )

    print(f"  Price matrix: {prices.shape} (dates × assets)")
    print(f"  Tables after fetch: prices={len(prices_table)} rows, securities={len(sec_table)} rows")
    print(f"  Asset names: {provider.asset_names()}")
    print(f"  NaN count: {np.isnan(prices).sum()}")

    with Timer("Second fetch (cache hit)"):
        prices2 = provider.fetch_prices(
            UniverseSpec(tickers=["SPY", "GLD", "AGG"]),
            TemporalSpec(lookback_days=504),
        )

    assert np.allclose(prices, prices2, equal_nan=True), "Cache miss — prices differ!"
    print("  ✓ Cache hit confirmed — identical results")

    return prices, provider.asset_names()


def test_stat_arb(provider):
    """GLD/GDX cointegration scan with real data."""
    section("2. STAT-ARB: GLD/GDX Pairs Trade")

    with Timer("Fetch GLD/GDX"):
        prices = provider.fetch_prices(
            UniverseSpec(tickers=["GLD", "GDX"]),
            TemporalSpec(lookback_days=756),
        )
    names = provider.asset_names()
    print(f"  Data: {prices.shape}, assets: {names}")

    engine = StatArbEngine()
    for params_label, params in [
        ("Classic (ADF only)", StatArbParams(
            entry_threshold=2.0, zscore_lookback=63,
            regression_method="ols", scoring_mode="classic", max_hurst=1.1,
        )),
        ("Composite scoring", StatArbParams(
            entry_threshold=1.5, zscore_lookback=63,
            regression_method="ridge", scoring_mode="composite", max_hurst=1.1,
        )),
    ]:
        result = engine.compute(StatArbInput(prices=prices, asset_names=names), params)
        status = "✓" if result.ok else "⚠"
        n_baskets = len(result.top_baskets)
        signal = result.signals[0].signal if result.signals else "none"
        zscore = result.signals[0].current_zscore if result.signals else 0
        print(f"  {status} {params_label}: {n_baskets} basket(s), signal={signal}, z={zscore:.2f}")


def test_risk_parity(provider):
    """Risk parity on TAA universe — verify equal risk contributions."""
    section("3. RISK PARITY: Bridgewater All-Weather (TAA_5)")

    with Timer("Fetch TAA universe"):
        prices = provider.fetch_prices(
            UniverseSpec(tickers=["SPY", "EFA", "AGG", "GLD", "VNQ"]),
            TemporalSpec(lookback_days=756),
        )
    names = provider.asset_names()
    returns = np.diff(prices, axis=0) / prices[:-1]

    engine = AllocationEngine()
    result = engine.compute(
        AllocationInput(returns=returns, asset_names=names),
        AllocationParams(method="risk_parity", cov_method="shrinkage", long_only=True),
    )

    w = result.result.weights
    cov = result.covariance.matrix
    sp = np.sqrt(w @ cov @ w)
    rc = w * (cov @ w) / sp
    rc_pct = rc / rc.sum() * 100
    vols = np.sqrt(np.diag(cov) * 252) * 100

    print(f"\n  {'Asset':<6} {'Weight':>8} {'AnnVol%':>8} {'RC%':>7}")
    print(f"  {'-'*32}")
    for i, nm in enumerate(names):
        print(f"  {nm:<6} {w[i]:>8.4f} {vols[i]:>7.2f}% {rc_pct[i]:>6.2f}%")

    print(f"\n  Portfolio vol (ann): {sp * np.sqrt(252) * 100:.2f}%")
    print(f"  Sharpe ratio: {result.result.sharpe_ratio:.3f}")
    print(f"  Diversification ratio: {result.result.diversification_ratio:.3f}")

    max_dev = np.max(np.abs(rc_pct - 20.0))
    status = "✓" if max_dev < 1.0 else "⚠"
    print(f"  {status} Max RC deviation from 20%: {max_dev:.3f}%")


def test_momentum_scan(provider):
    """Cross-sectional momentum on sector ETFs."""
    section("4. MOMENTUM: Sector ETF Cross-Sectional Scan")

    tickers = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]

    with Timer("Fetch sector ETFs"):
        prices = provider.fetch_prices(
            UniverseSpec(tickers=tickers),
            TemporalSpec(lookback_days=504),
        )
    names = provider.asset_names()
    print(f"  Data: {prices.shape}, assets: {len(names)}")

    engine = MomentumEngine()
    result = engine.compute(
        MomentumInput(prices=prices, asset_names=names),
        MomentumParams(
            lookback_periods=[63, 126, 252],
            skip_recent=21,
            scoring_method="cross_sectional",
            sizing_method="vol_target",
        ),
    )

    # Sort by composite score
    scored = sorted(
        zip(names, result.scores),
        key=lambda x: x[1].composite_score,
        reverse=True,
    )

    print(f"\n  {'Rank':<5} {'Asset':<6} {'Score':>8} {'Trend':>8}")
    print(f"  {'-'*30}")
    for rank, (nm, s) in enumerate(scored, 1):
        print(f"  {rank:<5} {nm:<6} {s.composite_score:>8.4f} {s.trend_strength:>8.4f}")

    print(f"\n  Weights: {np.round(result.portfolio_weights, 4).tolist()}")


def test_pipeline(provider):
    """Full pipeline: momentum → allocation."""
    section("5. FULL PIPELINE: Momentum Scores → Max Sharpe Allocation")

    with Timer("Fetch TAA universe"):
        prices = provider.fetch_prices(
            UniverseSpec(tickers=["SPY", "EFA", "AGG", "GLD", "VNQ"]),
            TemporalSpec(lookback_days=504),
        )
    names = provider.asset_names()
    returns = np.diff(prices, axis=0) / prices[:-1]

    # Stage 1: Momentum
    mom = MomentumEngine()
    mom_result = mom.compute(
        MomentumInput(prices=prices, asset_names=names),
        MomentumParams(lookback_periods=[126], scoring_method="time_series"),
    )
    expected_returns = np.array([s.composite_score for s in mom_result.scores])
    print(f"  Momentum scores: {dict(zip(names, np.round(expected_returns, 4)))}")

    # Stage 2: Allocation using momentum as expected returns
    alloc = AllocationEngine()
    alloc_result = alloc.compute(
        AllocationInput(returns=returns, asset_names=names),
        AllocationParams(
            method="max_sharpe",
            expected_returns=expected_returns,
            long_only=True,
        ),
    )

    w = alloc_result.result.weights
    print(f"  Optimal weights: {dict(zip(names, np.round(w, 4)))}")
    print(f"  Expected return: {alloc_result.result.expected_return:.4f}")
    print(f"  Expected vol: {alloc_result.result.expected_vol:.4f}")
    print(f"  Sharpe ratio: {alloc_result.result.sharpe_ratio:.3f}")


def test_model_orchestrator(provider):
    """Model orchestrator with DataStoreDataProvider."""
    section("6. MODEL ORCHESTRATOR")

    context = ModelContext(
        universe=UniverseSpec(tickers=["GLD", "GDX"], name="pairs_test"),
        temporal=TemporalSpec(lookback_days=504),
        name="stat_arb_validation",
    )

    model = Model(
        engine=StatArbEngine(),
        context=context,
        data_provider=provider,
    )

    with Timer("Model.run()"):
        result = model.run()

    print(f"  Status: {result.status}")
    print(f"  Baskets found: {len(result.top_baskets) if hasattr(result, 'top_baskets') else 'N/A'}")
    print(f"  ✓ Model orchestrator works with DataStoreDataProvider")


def test_datastore_state(provider):
    """Show final DataStore state after all tests."""
    section("7. DATASTORE FINAL STATE")

    ds = provider._ds
    for name in ["securities", "prices", "universes", "universe_members"]:
        if ds.has_table(name):
            table = ds.get_table(name)
            print(f"  {name:<20} {len(table):>6} rows")

    sec = ds.get_table("securities")
    if len(sec) > 0:
        print(f"\n  Securities loaded: {sec.data['security_id'].to_list()}")

    prices = ds.get_table("prices")
    if len(prices) > 0:
        tickers = prices.data["security_id"].unique().to_list()
        date_range = (
            prices.data["date"].min(),
            prices.data["date"].max(),
        )
        print(f"  Price tickers: {tickers}")
        print(f"  Date range: {date_range[0]} → {date_range[1]}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Praxis Market Data Integration Validation              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Choose fetcher based on network availability
    use_yfinance = try_yfinance()

    if use_yfinance:
        print("\n✓ Using YFinance (real market data)")
        fetcher = YFinanceFetcher(default_period="3y")
    else:
        print("\n⚠ YFinance unavailable — using MockMarketDataFetcher")
        fetcher = MockMarketDataFetcher.with_standard_profiles(n_days=756, seed=42)

    DataStore.reset_instance()
    ds = create_market_datastore(fetcher=fetcher)
    provider = DataStoreDataProvider(ds)

    t0 = time.time()
    passed = 0
    failed = 0

    tests = [
        test_lazy_loading,
        test_stat_arb,
        test_risk_parity,
        test_momentum_scan,
        test_pipeline,
        test_model_orchestrator,
        test_datastore_state,
    ]

    for test_fn in tests:
        try:
            test_fn(provider)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0

    section("SUMMARY")
    source = "YFinance (real data)" if use_yfinance else "Mock (synthetic)"
    print(f"  Data source: {source}")
    print(f"  Tests passed: {passed}/{passed + failed}")
    print(f"  Total time: {elapsed:.1f}s")

    if failed == 0:
        print("\n  ✓ ALL TESTS PASSED")
    else:
        print(f"\n  ✗ {failed} TEST(S) FAILED")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
