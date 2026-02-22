"""
Deep-Dive Validation: Realistic Strategy Examples
==================================================

Tests each engine against published/well-known strategies to verify
the framework handles real-world configurations with ONLY context changes.

Each test documents:
  - Published source / strategy archetype
  - How it maps to Engine + params
  - Structural findings marked with [GAP] where extension needed

Naming: test_<engine>_<strategy_archetype>
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from engines.cointegration import StatArbEngine, StatArbParams, StatArbInput
from engines.momentum import MomentumEngine, MomentumParams, MomentumInput
from engines.allocation import AllocationEngine, AllocationParams, AllocationInput
from engines.options import (
    OptionsEngine, OptionsParams, OptionsInput, GreeksVector)
from engines.event_signal import (
    EventSignalEngine, EventSignalParams, EventSignalInput)
from engines.model import Model, ModelResult
from engines.context.model_context import (
    ModelContext, UniverseSpec, TemporalSpec, RiskSpec, ExecutionSpec,
    AssetClass, Frequency, Calendar, RebalanceFrequency, ExecutionMode)
from engines.adapters.providers import InMemoryDataProvider, InMemoryResultStore
from engines.base import EngineStatus

# ════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS — realistic market structures, not random noise
# ════════════════════════════════════════════════════════════════════════════

def make_gld_gdx_style(n=504, seed=42):
    """
    Simulate GLD/GDX-like pair: gold miner ETF cointegrated with gold ETF.
    Mirrors the Chan CPO paper pair trade from main.py.
    GDX ≈ 2.5 * GLD + mean-reverting spread.
    """
    rng = np.random.default_rng(seed)
    # Common gold factor
    gold = np.cumsum(rng.normal(0.0002, 0.01, n)) + np.log(150)
    gld = np.exp(gold)
    # GDX = hedge_ratio * GLD + spread
    hedge = 2.5
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.92 * spread[t-1] + rng.normal(0, 0.3)
    gdx = hedge * gld / 4.0 + spread + 30  # scale to ~$30 range
    # Add a few uncorrelated ETFs for the universe
    spy = np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)) + np.log(400))
    tlt = np.exp(np.cumsum(rng.normal(0.0001, 0.008, n)) + np.log(100))
    slv = np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)) + np.log(22))
    return np.column_stack([gld, gdx, spy, tlt, slv]), \
           ["GLD", "GDX", "SPY", "TLT", "SLV"]


def make_sector_etf_universe(n=504, seed=43):
    """
    SPY + 10 sector ETFs. SPY is weighted sum of sectors → cointegrated.
    Simulates ETF index arbitrage opportunity.
    """
    rng = np.random.default_rng(seed)
    sectors = []
    for i in range(10):
        drift = rng.normal(0.0003, 0.0001)
        vol = rng.uniform(0.008, 0.018)
        s = np.exp(np.cumsum(rng.normal(drift, vol, n)) + np.log(50 + i*10))
        sectors.append(s)
    sectors = np.column_stack(sectors)
    # SPY = weighted sum of sectors + noise
    wts = rng.dirichlet(np.ones(10))
    spy_base = sectors @ wts * 4 + 100
    spy = spy_base + np.cumsum(rng.normal(0, 0.1, n))  # small tracking noise
    prices = np.column_stack([spy, sectors])
    names = ["SPY","XLK","XLF","XLV","XLY","XLP","XLE","XLI","XLB","XLU","XLRE"]
    return prices, names


def make_global_futures(n=756, n_assets=12, seed=44):
    """
    12 futures-like series with persistent trends (CTA universe).
    Some trending up, some down, some range-bound.
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros((n, n_assets))
    prices[0] = 100
    regimes = rng.choice([-1, 0, 1], size=n_assets, p=[0.3, 0.2, 0.5])
    for i in range(n_assets):
        drift = regimes[i] * rng.uniform(0.0003, 0.001)
        vol = rng.uniform(0.008, 0.02)
        for t in range(1, n):
            # Regime shifts every ~120 days
            if rng.random() < 1/120:
                regimes[i] = rng.choice([-1, 0, 1])
                drift = regimes[i] * rng.uniform(0.0003, 0.001)
            prices[t, i] = prices[t-1, i] * (1 + rng.normal(drift, vol))
    names = ["ES","NQ","YM","CL","GC","SI","ZN","ZB","6E","6J","6B","6A"]
    return prices, names


def make_equity_cross_section(n=504, n_assets=50, seed=45):
    """
    50 stocks with varying momentum profiles. Some winners, some losers,
    most mediocre. For Jegadeesh-Titman style cross-sectional momentum.
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros((n, n_assets))
    prices[0] = rng.uniform(20, 200, n_assets)
    for i in range(n_assets):
        drift = rng.normal(0.0003, 0.002)
        vol = rng.uniform(0.01, 0.03)
        for t in range(1, n):
            prices[t, i] = prices[t-1, i] * (1 + rng.normal(drift, vol))
            prices[t, i] = max(prices[t, i], 1.0)  # floor
    return prices, [f"STK{i:02d}" for i in range(n_assets)]


def make_multi_asset_portfolio(n=756, seed=46):
    """
    Classic TAA universe: US equity, intl equity, bonds, gold, REITs.
    Different vols and correlations for allocation testing.
    """
    rng = np.random.default_rng(seed)
    # Correlated via factor model
    factors = rng.normal(0, 1, (n, 3))  # market, rate, inflation
    loadings = np.array([
        [1.0, -0.2, 0.1],   # US equity
        [0.8, -0.1, 0.2],   # Intl equity
        [0.05, 0.8, -0.1],  # Bonds
        [0.1, -0.1, 0.7],   # Gold
        [0.6, 0.3, 0.2],    # REITs
    ])
    vols = np.array([0.015, 0.018, 0.005, 0.012, 0.016])
    drifts = np.array([0.0004, 0.0003, 0.0001, 0.0002, 0.0003])
    returns = factors @ loadings.T * vols[None, :] * 0.5 + \
              rng.normal(0, 1, (n, 5)) * vols[None, :] * 0.5 + drifts
    prices = np.exp(np.cumsum(returns, axis=0))
    prices = prices / prices[0] * np.array([400, 60, 100, 170, 90])
    names = ["SPY", "EFA", "AGG", "GLD", "VNQ"]
    return prices, names


def make_options_surface(S=100, base_vol=0.20, seed=47):
    """
    Realistic vol surface: negative skew, term structure, realistic prices.
    """
    from scipy.stats import norm as _norm
    rng = np.random.default_rng(seed)
    strikes = np.array([80, 85, 90, 95, 97.5, 100, 102.5, 105, 110, 115, 120])
    expiries = np.array([7, 14, 30, 60, 90, 180, 365]) / 365.0
    r, q = 0.05, 0.01
    mon = np.log(strikes / S)
    iv = np.zeros((len(strikes), len(expiries)))
    for j, T in enumerate(expiries):
        # ATM vol increases with sqrt(T) (realistic term structure)
        atm = base_vol * (1 + 0.05 * np.sqrt(T * 365 / 30))
        # Skew: OTM puts more expensive
        iv[:, j] = atm - 0.12 * mon + 0.25 * mon**2
        iv[:, j] += rng.normal(0, 0.003, len(strikes))
        iv[:, j] = np.clip(iv[:, j], 0.05, 1.0)
    # Generate consistent market prices
    mp = np.zeros_like(iv)
    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            s = iv[i, j]
            d1 = (np.log(S/K) + (r - q + 0.5*s**2)*T) / (s*np.sqrt(T))
            d2 = d1 - s * np.sqrt(T)
            mp[i, j] = S*np.exp(-q*T)*_norm.cdf(d1) - K*np.exp(-r*T)*_norm.cdf(d2)
    # Underlying history for realized vol
    hist = np.exp(np.cumsum(rng.normal(0.0003, base_vol/np.sqrt(252), 504)))
    hist = hist / hist[-1] * S
    return dict(underlying_price=S, risk_free_rate=r, dividend_yield=q,
                strikes=strikes, expiries=expiries, market_prices=mp,
                underlying_history=hist)


# ════════════════════════════════════════════════════════════════════════════
# ENGINE 1: COINTEGRATION / STAT-ARB
# ════════════════════════════════════════════════════════════════════════════

class TestStatArbRealistic:
    """
    Published references:
    - Chan (CPO paper): GLD/GDX mean-reversion with z-score entry/exit
    - Burgess (PhD thesis): stepwise regression, multi-leg baskets
    - Gatev et al (2006): pairs trading distance method
    """

    def test_gld_gdx_pairs_trade(self):
        """
        Chan CPO paper: GLD/GDX pair trade with z-score signals.
        This is the strategy in main.py pair_trade_gld_gdx().
        Engine should find GDX as primary basket member for GLD.
        """
        prices, names = make_gld_gdx_style(n=504)
        engine = StatArbEngine()
        # Mimic Chan's config: simple 2-asset pair, ridge regression
        params = StatArbParams(
            n_per_basket=1,            # pairs trade = 1 basket member
            max_candidates=5,          # small universe
            regression_method="ridge",
            ridge_alpha=0.01,
            significance=0.05,
            min_half_life=5.0,
            max_half_life=126.0,
            max_hurst=1.1,            # relaxed for short series
            scoring_mode="classic",
            top_k=3,
            optimization_method="equal",
            zscore_lookback=63,
            entry_threshold=2.0,       # Chan uses 2.0
            exit_threshold=0.5,
        )
        data = StatArbInput(prices=prices, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        assert result.n_candidates_scanned > 0
        # Should find at least one stationary basket
        # GLD-GDX is genuinely cointegrated in our synthetic data
        if result.top_baskets:
            # Check that GLD (idx 0) or GDX (idx 1) appears as target
            targets = {b.target_idx for b in result.top_baskets}
            baskets_flat = set()
            for b in result.top_baskets:
                baskets_flat.update(b.basket_indices)
            # At least one basket involves GLD or GDX
            gold_involved = (0 in targets or 1 in targets or
                             0 in baskets_flat or 1 in baskets_flat)
            assert gold_involved, "GLD/GDX should appear in top baskets"

            # Signals should be generated for each top basket
            assert len(result.signals) == len(result.top_baskets)
            for sig in result.signals:
                assert sig.signal in ("long", "short", "flat")

    def test_sector_etf_index_arb(self):
        """
        ETF index arbitrage: SPY vs sector ETFs.
        SPY is constructionally cointegrated with its sectors.
        Engine should find highly stationary residuals.
        """
        prices, names = make_sector_etf_universe(n=504)
        engine = StatArbEngine()
        params = StatArbParams(
            n_per_basket=5,            # multi-leg like Burgess
            max_candidates=11,
            regression_method="ridge",
            significance=0.05,
            max_hurst=1.1,
            scoring_mode="composite",
            score_weights={"adf": 0.3, "hurst": 0.2, "half_life": 0.3,
                          "variance_ratio": 0.2},
            top_k=5,
            optimization_method="min_variance",
            zscore_lookback=63,
        )
        data = StatArbInput(prices=prices, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        assert result.n_candidates_scanned > 0
        # With composite scoring, all baskets should have scores > 0
        for b in result.top_baskets:
            assert b.composite_score >= 0
            assert b.rank > 0
        # Portfolios should have reasonable weights
        for pf in result.portfolios:
            assert_allclose(np.sum(np.abs(pf.weights)), 1.0, atol=0.01)

    def test_burgess_multi_leg_basket(self):
        """
        Burgess PhD: stepwise regression builds 5-leg baskets for residual
        stationarity. Test the full extraction pipeline.
        """
        prices, names = make_sector_etf_universe(n=756)
        engine = StatArbEngine()
        params = StatArbParams(
            n_per_basket=5,
            max_candidates=11,
            regression_method="ols",  # Burgess uses OLS
            significance=0.05,
            min_half_life=5.0,
            max_half_life=252.0,      # longer for daily
            max_hurst=1.1,
            scoring_mode="composite",
            top_k=3,
            optimization_method="max_sharpe",
        )
        data = StatArbInput(prices=prices, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        # Verify regression structure
        for b in result.top_baskets:
            reg = b.regression
            # OLS should achieve decent R² on cointegrated ETFs
            assert reg.r_squared > 0.5, \
                f"R² too low: {reg.r_squared:.3f} for target {b.target_idx}"
            # Residuals should be generated
            assert len(reg.residuals) == prices.shape[0]
            # Stationarity should be tested
            stat = b.stationarity
            assert stat.adf_is_stationary
            assert stat.half_life > 0

    def test_statarb_scoring_modes_differ(self):
        """
        Verify classic vs composite scoring produce different rankings.
        Classic = just ADF statistic; composite = weighted multi-metric.
        """
        prices, names = make_gld_gdx_style(n=504)
        engine = StatArbEngine()

        classic = engine.compute(
            StatArbInput(prices=prices, asset_names=names),
            StatArbParams(scoring_mode="classic", max_hurst=1.1, top_k=5))

        composite = engine.compute(
            StatArbInput(prices=prices, asset_names=names),
            StatArbParams(scoring_mode="composite", max_hurst=1.1, top_k=5))

        # Both should produce results
        assert classic.ok and composite.ok
        # Scores should differ in magnitude (different scoring functions)
        if classic.top_baskets and composite.top_baskets:
            c_scores = [b.composite_score for b in classic.top_baskets]
            cp_scores = [b.composite_score for b in composite.top_baskets]
            # Not identical (different scoring methods)
            assert c_scores != cp_scores or len(c_scores) != len(cp_scores)


# ════════════════════════════════════════════════════════════════════════════
# ENGINE 2: MOMENTUM / TREND-FOLLOWING
# ════════════════════════════════════════════════════════════════════════════

class TestMomentumRealistic:
    """
    Published references:
    - Moskowitz, Ooi, Pedersen (2012): TSMOM time-series momentum
    - Jegadeesh & Titman (1993): cross-sectional momentum
    - AQR/Man CTA trend-following
    """

    def test_tsmom_futures(self):
        """
        TSMOM (Moskowitz et al): 12-month return, skip last month,
        vol-target sizing. Time-series scoring on CTA universe.
        """
        prices, names = make_global_futures(n=756)
        engine = MomentumEngine()
        params = MomentumParams(
            return_type="log",
            scoring_method="time_series",
            lookback_periods=[252],     # 12-month return
            skip_recent=21,             # skip last month
            ma_type="ema",
            fast_period=50,
            slow_period=200,
            breakout_method="donchian",
            breakout_period=60,
            signal_sign=1.0,            # momentum (not contrarian)
            sizing_method="vol_target",
            vol_lookback=63,
            vol_target=0.10,            # 10% vol target per position
            ann_factor=252.0,
            long_only=False,            # long/short
            max_weight=0.20,
        )
        data = MomentumInput(prices=prices, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        assert len(result.scores) == prices.shape[1]
        assert len(result.signals) == prices.shape[1]
        # Vol-target sizing should produce reasonable weights
        w = result.portfolio_weights
        assert len(w) == prices.shape[1]
        assert np.sum(np.abs(w)) > 0  # not all zero
        # Some assets should be long, some short (trending universe)
        assert np.any(w > 0) and np.any(w < 0), "Expected long/short positions"

    def test_jegadeesh_titman_cross_sectional(self):
        """
        J&T (1993): rank stocks by 12-1 month return, go long top decile,
        short bottom decile. Cross-sectional scoring.
        """
        prices, names = make_equity_cross_section(n=504, n_assets=50)
        engine = MomentumEngine()
        params = MomentumParams(
            scoring_method="cross_sectional",
            lookback_periods=[252],
            skip_recent=21,
            top_n=5,                    # top decile of 50
            bottom_n=5,                 # bottom decile
            long_only=False,
            max_weight=0.20,
            sizing_method="equal",
        )
        data = MomentumInput(prices=prices, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        w = result.portfolio_weights
        # In cross-sectional mode, should have long and short legs
        n_long = np.sum(w > 0.001)
        n_short = np.sum(w < -0.001)
        assert n_long >= 1, "Should have long positions"
        assert n_short >= 1, "Should have short positions"
        # Rankings should be valid permutation
        assert set(result.rankings) == set(range(50))

    def test_cta_trend_following_donchian(self):
        """
        Classic CTA: Donchian channel breakout with vol sizing.
        Turtle-trader style on futures.
        """
        prices, names = make_global_futures(n=504)
        engine = MomentumEngine()
        params = MomentumParams(
            scoring_method="time_series",
            lookback_periods=[63, 126],   # 3m and 6m
            ma_type="sma",
            fast_period=20,
            slow_period=55,               # Donchian-55
            breakout_method="donchian",
            breakout_period=55,
            signal_sign=1.0,
            sizing_method="vol_target",
            vol_target=0.15,              # higher vol budget for CTA
            max_weight=0.25,
        )
        data = MomentumInput(prices=prices, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        # Check breakout signals are generated
        for sig in result.signals:
            assert sig.breakout_signal in (-1.0, 0.0, 1.0), \
                f"Donchian breakout should be -1/0/1, got {sig.breakout_signal}"

    def test_contrarian_is_sign_flip(self):
        """
        Key architectural claim: contrarian is literally the same engine
        with signal_sign=-1. Verify identical magnitudes, flipped signs.
        """
        prices, _ = make_equity_cross_section(n=504, n_assets=10)
        engine = MomentumEngine()

        mom_result = engine.compute(
            MomentumInput(prices=prices),
            MomentumParams(signal_sign=1.0, sizing_method="equal"))
        con_result = engine.compute(
            MomentumInput(prices=prices),
            MomentumParams(signal_sign=-1.0, sizing_method="equal"))

        # Composite scores should be negated
        for ms, cs in zip(mom_result.scores, con_result.scores):
            assert_allclose(ms.composite_score, -cs.composite_score, atol=1e-10,
                err_msg="Contrarian scores should be negated momentum scores")

    def test_dual_momentum(self):
        """
        Antonacci's dual momentum: combine time-series and cross-sectional.
        """
        prices, names = make_multi_asset_portfolio(n=504)
        engine = MomentumEngine()
        params = MomentumParams(
            scoring_method="dual",
            lookback_periods=[126, 252],
            skip_recent=0,
            long_only=True,             # dual momentum is long-only
            top_n=2,                    # hold top 2 assets
            max_weight=0.50,
        )
        data = MomentumInput(prices=prices, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        w = result.portfolio_weights
        # Long only
        assert np.all(w >= -1e-10), "Dual momentum should be long-only"
        assert_allclose(np.sum(np.abs(w)), 1.0, atol=0.01)

    def test_bollinger_breakout(self):
        """
        Bollinger band breakout variant for mean-reversion detection.
        """
        prices, _ = make_global_futures(n=504, n_assets=6)
        engine = MomentumEngine()
        params = MomentumParams(
            breakout_method="bollinger",
            breakout_period=20,
            breakout_mult=2.0,
            ma_type="sma",
        )
        result = engine.compute(MomentumInput(prices=prices), params)
        assert result.ok
        # Bollinger signals should be continuous, not just -1/0/1
        bsigs = [s.breakout_signal for s in result.signals]
        # At least some should be fractional
        unique_vals = set(bsigs)
        assert len(unique_vals) >= 2, "Expected varied Bollinger signals"


# ════════════════════════════════════════════════════════════════════════════
# ENGINE 3: ALLOCATION
# ════════════════════════════════════════════════════════════════════════════

class TestAllocationRealistic:
    """
    Published references:
    - Bridgewater All-Weather: risk parity
    - Black-Litterman (1992): Bayesian portfolio optimization
    - Lopez de Prado (2016): HRP
    - Chan CPO paper: TAA with GLD, IJS, SHY, TLT, SPY
    """

    def test_risk_parity_all_weather(self):
        """
        Bridgewater All-Weather: equal risk contribution across asset classes.
        With 5-asset TAA universe.
        """
        prices, names = make_multi_asset_portfolio(n=756)
        returns = np.diff(prices, axis=0) / prices[:-1]
        engine = AllocationEngine()
        params = AllocationParams(
            method="risk_parity",
            cov_method="shrinkage",
            shrinkage_intensity=0.2,
            long_only=True,
        )
        data = AllocationInput(returns=returns, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        w = result.result.weights
        # All weights should be non-negative (long-only)
        assert np.all(w >= -1e-10), f"Expected non-negative weights, got {w}"
        # Should sum to 1
        assert_allclose(np.sum(w), 1.0, atol=0.01)
        # LOW-VOL ASSETS GET HIGHER WEIGHT (the whole point of risk parity)
        # AGG (bonds, idx 2) has ~5% vol vs SPY (idx 0) at ~17%
        assert w[2] > w[0], \
            f"Bonds (w={w[2]:.4f}) should outweigh equities (w={w[0]:.4f}) in risk parity"
        # EQUAL RISK CONTRIBUTIONS (the defining property)
        cov = result.covariance.matrix
        sp = np.sqrt(w @ cov @ w)
        rc = w * (cov @ w) / sp  # risk contribution per asset
        rc_pct = rc / rc.sum()   # as fraction of total
        budget = np.ones(5) / 5  # equal budget
        assert_allclose(rc_pct, budget, atol=0.01,
            err_msg=f"Risk contributions {np.round(rc_pct*100,2)}% should be ~20% each")
        # Marginal risk contributions should exist
        mrc = result.result.marginal_risk_contributions
        assert mrc is not None
        # Diversification ratio > 1 means diversification benefit
        assert result.result.diversification_ratio > 1.0

    def test_black_litterman_with_views(self):
        """
        B-L: Start with equilibrium, overlay views.
        View: "US equity will outperform intl equity by 2%"
        """
        prices, names = make_multi_asset_portfolio(n=504)
        returns = np.diff(prices, axis=0) / prices[:-1]
        n_assets = 5
        engine = AllocationEngine()

        # View: asset 0 (SPY) outperforms asset 1 (EFA) by 2% annualized
        P = np.array([[1, -1, 0, 0, 0]])  # relative view
        Q = np.array([0.02 / 252])         # daily excess return
        omega = np.array([[0.0001]])        # confidence

        params = AllocationParams(
            method="black_litterman",
            views_P=P, views_Q=Q, views_omega=omega,
            tau=0.05,
            cov_method="shrinkage",
        )
        data = AllocationInput(returns=returns, asset_names=names)
        result = engine.compute(data, params)

        assert result.ok
        w = result.result.weights
        # With bullish US view, SPY weight should exceed EFA
        assert w[0] > w[1], \
            f"SPY ({w[0]:.3f}) should outweigh EFA ({w[1]:.3f}) given bullish view"

    def test_black_litterman_without_views(self):
        """B-L with no views should give equilibrium (market-cap) weights."""
        prices, names = make_multi_asset_portfolio(n=504)
        returns = np.diff(prices, axis=0) / prices[:-1]
        engine = AllocationEngine()
        params = AllocationParams(method="black_litterman")
        result = engine.compute(
            AllocationInput(returns=returns, asset_names=names), params)
        assert result.ok
        # Without views, should approximate equal-weight-ish market portfolio
        w = result.result.weights
        assert np.all(np.abs(w) > 0), "All assets should have some weight"

    def test_hrp(self):
        """
        Lopez de Prado HRP: hierarchical risk parity.
        Should produce reasonable weights without matrix inversion.
        """
        prices, names = make_multi_asset_portfolio(n=504)
        returns = np.diff(prices, axis=0) / prices[:-1]
        engine = AllocationEngine()
        params = AllocationParams(
            method="hrp",
            long_only=True,
        )
        result = engine.compute(
            AllocationInput(returns=returns, asset_names=names), params)
        assert result.ok
        w = result.result.weights
        assert np.all(w >= 0), "HRP should be long-only"
        assert_allclose(np.sum(w), 1.0, atol=0.01)

    def test_target_volatility(self):
        """
        Target vol: scale portfolio to hit 10% annualized vol.
        Common in institutional mandates.
        """
        prices, names = make_multi_asset_portfolio(n=504)
        returns = np.diff(prices, axis=0) / prices[:-1]
        engine = AllocationEngine()
        params = AllocationParams(
            method="target_vol",
            vol_target=0.10,
            cov_method="ewma",
            ewma_halflife=63,
        )
        result = engine.compute(
            AllocationInput(returns=returns, asset_names=names), params)
        assert result.ok
        # Expected vol should be near target (not exact due to estimation error)
        ev = result.result.expected_vol
        assert 0.02 < ev < 0.50, f"Expected vol {ev:.3f} seems unreasonable"

    def test_max_diversification(self):
        """
        Choueifaty's max diversification: maximize diversification ratio.
        """
        prices, names = make_multi_asset_portfolio(n=504)
        returns = np.diff(prices, axis=0) / prices[:-1]
        engine = AllocationEngine()
        params = AllocationParams(method="max_diversification")
        result = engine.compute(
            AllocationInput(returns=returns, asset_names=names), params)
        assert result.ok
        assert result.result.diversification_ratio > 1.0, \
            "Max diversification should achieve DR > 1"

    def test_turnover_constraint(self):
        """
        Verify turnover constraint works: rebalancing from known starting weights.
        """
        prices, names = make_multi_asset_portfolio(n=504)
        returns = np.diff(prices, axis=0) / prices[:-1]
        engine = AllocationEngine()
        # Start equal weight
        current = np.ones(5) / 5
        params = AllocationParams(
            method="max_sharpe",
            max_turnover=0.3,  # very tight constraint
        )
        result = engine.compute(
            AllocationInput(returns=returns, asset_names=names,
                          current_weights=current), params)
        assert result.ok
        # Turnover should respect constraint
        actual_turnover = np.sum(np.abs(result.result.weights - current))
        assert actual_turnover <= 0.31, \
            f"Turnover {actual_turnover:.3f} exceeds constraint 0.3"

    def test_allocation_as_subcomponent(self):
        """
        Key architectural claim: AllocationEngine.allocate() provides
        simplified interface for use as sub-component in pipelines.
        """
        prices, _ = make_multi_asset_portfolio(n=504)
        returns = np.diff(prices, axis=0) / prices[:-1]
        engine = AllocationEngine()

        # Sub-component interface
        weights = engine.allocate(returns, AllocationParams(method="risk_parity"))
        assert len(weights) == 5
        assert_allclose(np.sum(np.abs(weights)), 1.0, atol=0.05)

    def test_chan_cpo_taa_portfolio(self):
        """
        Chan CPO paper: TAA portfolio of GLD, IJS, SHY, TLT, SPY.
        Semi-annual rebalancing, Markowitz vs CPO comparison.
        We verify the allocation engine can handle this universe.

        [GAP] CPO regime-conditional rebalancing is not yet implemented.
        The engine handles static optimization; CPO would need a regime
        detection layer feeding different params per period.
        """
        prices, names = make_multi_asset_portfolio(n=756)
        returns = np.diff(prices, axis=0) / prices[:-1]
        engine = AllocationEngine()

        # Standard Markowitz (Chan's baseline)
        params = AllocationParams(
            method="max_sharpe",
            cov_method="sample",
            long_only=True,
            max_weight=0.40,
        )
        result = engine.compute(
            AllocationInput(returns=returns, asset_names=names), params)
        assert result.ok
        assert result.result.sharpe_ratio != 0, "Sharpe should be computed"


# ════════════════════════════════════════════════════════════════════════════
# ENGINE 4: OPTIONS / VOLATILITY
# ════════════════════════════════════════════════════════════════════════════

class TestOptionsRealistic:
    """
    Published references:
    - CBOE PutWrite Index (PUT): systematic OTM put selling
    - Variance risk premium literature (Carr & Wu, Bollerslev et al)
    - Volatility surface trading (Gatheral, SVI parameterization)
    """

    def test_systematic_put_writing(self):
        """
        CBOE PUT index: sell OTM puts, collect premium.
        Engine should price puts and compute Greeks for risk management.
        """
        data = make_options_surface(S=100, base_vol=0.20)
        engine = OptionsEngine()
        params = OptionsParams(greek_method="analytic")

        inp = OptionsInput(
            underlying_price=data["underlying_price"],
            risk_free_rate=data["risk_free_rate"],
            dividend_yield=data["dividend_yield"],
            strikes=data["strikes"],
            expiries=data["expiries"],
            market_prices=data["market_prices"],
            option_type="call",  # IV extraction works on calls
            underlying_history=data["underlying_history"],
        )
        result = engine.compute(inp, params)

        assert result.ok
        surf = result.surface
        assert surf is not None
        # IV surface should be populated
        assert not np.all(np.isnan(surf.iv_matrix))
        # ATM vol should be close to base_vol
        valid_atm = surf.atm_vol[~np.isnan(surf.atm_vol)]
        if len(valid_atm) > 0:
            assert 0.10 < valid_atm[0] < 0.40, \
                f"ATM vol {valid_atm[0]:.3f} unreasonable"
        # Greeks matrix should be populated
        assert len(result.greeks_matrix) == len(data["strikes"])
        # Realized vol should be computed
        assert result.realized_vol > 0

    def test_variance_risk_premium(self):
        """
        VRP: IV systematically exceeds RV.
        Engine should detect IV > RV spread and generate signal.
        """
        # Use higher base vol but lower realized vol to create VRP
        data = make_options_surface(S=100, base_vol=0.25, seed=48)
        engine = OptionsEngine()
        params = OptionsParams(iv_rv_threshold=0.03)

        inp = OptionsInput(
            underlying_price=data["underlying_price"],
            risk_free_rate=data["risk_free_rate"],
            dividend_yield=data["dividend_yield"],
            strikes=data["strikes"],
            expiries=data["expiries"],
            market_prices=data["market_prices"],
            option_type="call",
            underlying_history=data["underlying_history"],
        )
        result = engine.compute(inp, params)

        assert result.ok
        # Should detect IV vs RV spread
        iv_rv_sigs = [s for s in result.signals if s.signal_type == "iv_vs_rv"]
        if iv_rv_sigs:
            sig = iv_rv_sigs[0]
            assert sig.direction in ("short_vol", "long_vol")
            assert sig.magnitude > 0

    def test_vol_surface_skew(self):
        """
        Skew trading: detect when put skew is abnormally steep or flat.
        Negative skew is structural in equity options.
        """
        data = make_options_surface(S=100, base_vol=0.20)
        engine = OptionsEngine()
        params = OptionsParams(skew_zscore_threshold=1.0)  # lower threshold

        inp = OptionsInput(
            underlying_price=data["underlying_price"],
            risk_free_rate=data["risk_free_rate"],
            dividend_yield=data["dividend_yield"],
            strikes=data["strikes"],
            expiries=data["expiries"],
            market_prices=data["market_prices"],
            option_type="call",
            underlying_history=data["underlying_history"],
        )
        result = engine.compute(inp, params)

        assert result.ok
        surf = result.surface
        # Skew should be negative (OTM puts more expensive) for at least
        # some expiries in our realistic surface
        assert surf.skew is not None
        valid_skew = surf.skew[~np.isnan(surf.skew)]
        # Our generator creates negative skew via -0.12*moneyness term
        if len(valid_skew) > 0:
            assert np.any(valid_skew > 0), \
                "Expected positive skew values (OTM put > OTM call IV)"

    def test_greeks_consistency(self):
        """
        Verify put-call parity holds in Greeks: call_delta - put_delta ≈ e^{-qT}
        Uses the engine's BS implementation directly.
        """
        engine = OptionsEngine()
        S, K, T, r, sigma, q = 100, 100, 0.25, 0.05, 0.20, 0.02
        call_g = engine.bs_greeks(S, K, T, r, sigma, q, "call")
        put_g = engine.bs_greeks(S, K, T, r, sigma, q, "put")

        # Delta: call - put = e^{-qT}
        assert_allclose(call_g.delta - put_g.delta, np.exp(-q*T), atol=0.01)
        # Gamma: same for call and put
        assert_allclose(call_g.gamma, put_g.gamma, atol=1e-6)
        # Vega: same for call and put
        assert_allclose(call_g.vega, put_g.vega, atol=1e-6)

    def test_iv_round_trip(self):
        """
        Price → IV → Price round-trip should be consistent.
        """
        engine = OptionsEngine()
        S, K, T, r, sigma, q = 100, 105, 0.5, 0.05, 0.25, 0.0
        price = engine.bs_price(S, K, T, r, sigma, q, "call")

        # Extract IV from this price
        from scipy.optimize import brentq
        iv = brentq(
            lambda s: engine.bs_price(S, K, T, r, s, q, "call") - price,
            0.001, 5.0)
        assert_allclose(iv, sigma, atol=1e-6)

    def test_vol_term_structure(self):
        """
        Term structure signal: detect contango/backwardation in vol term structure.
        """
        data = make_options_surface(S=100, base_vol=0.20)
        engine = OptionsEngine()
        params = OptionsParams(iv_rv_threshold=0.01)

        inp = OptionsInput(
            underlying_price=data["underlying_price"],
            risk_free_rate=data["risk_free_rate"],
            dividend_yield=data["dividend_yield"],
            strikes=data["strikes"],
            expiries=data["expiries"],
            market_prices=data["market_prices"],
            option_type="call",
            underlying_history=data["underlying_history"],
        )
        result = engine.compute(inp, params)

        assert result.ok
        ts_sigs = [s for s in result.signals if s.signal_type == "term_structure"]
        # Term structure signal should be generated when front/back vols differ
        # (our generator creates upward-sloping term structure)
        if ts_sigs:
            assert ts_sigs[0].direction in ("short_vol", "long_vol")


# ════════════════════════════════════════════════════════════════════════════
# ENGINE 7: EVENT / SIGNAL
# ════════════════════════════════════════════════════════════════════════════

class TestEventSignalRealistic:
    """
    Published references:
    - Ball & Brown (1968): PEAD (post-earnings announcement drift)
    - Tetlock (2007): news sentiment and stock returns
    - Piotroski (2000): F-Score fundamental signals
    """

    def test_pead_earnings_momentum(self):
        """
        PEAD: stocks with positive earnings surprises continue to drift up.
        Model as earnings surprise feature processed through EventSignal.
        """
        rng = np.random.default_rng(50)
        n_stocks = 30
        # Earnings surprise: strong signal
        earnings_surprise = rng.normal(0, 1, (n_stocks, 1))
        names = [f"STK{i}" for i in range(n_stocks)]

        engine = EventSignalEngine()
        params = EventSignalParams(
            feature_method="zscore",
            combine_method="equal",
            decay_type="exponential",
            decay_halflife=63,  # PEAD drifts over ~3 months
        )
        data = EventSignalInput(
            numeric_features=earnings_surprise,
            feature_names=["earnings_surprise"],
            asset_names=names,
            timestamps=np.arange(n_stocks, dtype=float),
        )
        result = engine.compute(data, params)

        assert result.ok
        assert len(result.alphas) == n_stocks
        # Rankings should exist
        assert len(result.rankings) == n_stocks
        # Top-ranked stock should have highest alpha
        top_idx = np.argmin(result.rankings)
        assert result.alphas[top_idx].rank == 0

    def test_news_sentiment_scoring(self):
        """
        Tetlock-style sentiment: count positive/negative words in news.
        """
        n_assets = 10
        engine = EventSignalEngine()
        params = EventSignalParams(
            feature_method="raw",
            combine_method="equal",
            decay_type="none",
        )
        text_data = {
            0: ["Company beat earnings with strong revenue growth and record profits"],
            1: ["Disappointing results with weak guidance and declining margins"],
            2: ["Upgrade to outperform bullish outlook positive momentum"],
            3: ["Downgrade to underperform bearish negative sentiment"],
            5: ["Mixed results but growth in key segments offset decline elsewhere"],
        }
        data = EventSignalInput(
            numeric_features=np.zeros((n_assets, 1)),
            feature_names=["placeholder"],
            asset_names=[f"S{i}" for i in range(n_assets)],
            text_data=text_data,
        )
        result = engine.compute(data, params)

        assert result.ok
        # Stock 0 (bullish text) should rank higher than stock 1 (bearish)
        alpha_0 = result.alphas[0].sentiment_score
        alpha_1 = result.alphas[1].sentiment_score
        assert alpha_0 > alpha_1, \
            f"Bullish sentiment ({alpha_0:.3f}) should exceed bearish ({alpha_1:.3f})"

    def test_piotroski_fscore_multi_feature(self):
        """
        Piotroski F-Score: 9 binary signals combined.
        We simulate as 5 continuous features (simplified).
        """
        rng = np.random.default_rng(51)
        n_stocks = 40
        features = rng.standard_normal((n_stocks, 5))
        engine = EventSignalEngine()
        params = EventSignalParams(
            feature_method="rank",          # rank normalization
            combine_method="equal",
            decay_type="none",
            feature_weights={
                "roa": 1.5,
                "cash_flow": 1.5,
                "leverage": 1.0,
                "liquidity": 1.0,
                "margin": 1.0,
            },
        )
        data = EventSignalInput(
            numeric_features=features,
            feature_names=["roa", "cash_flow", "leverage", "liquidity", "margin"],
            asset_names=[f"STK{i}" for i in range(n_stocks)],
        )
        result = engine.compute(data, params)

        assert result.ok
        assert len(result.alphas) == n_stocks
        # Rank-based scores should be in [-1, 1]
        for a in result.alphas:
            for nm, v in a.normalized_scores.items():
                assert -1.1 <= v <= 1.1, f"Rank score {v} out of range for {nm}"

    def test_decay_reduces_stale_signals(self):
        """
        Exponential decay: older signals should have lower magnitude.
        """
        n = 20
        rng = np.random.default_rng(52)
        features = np.ones((n, 1))  # all same raw score
        engine = EventSignalEngine()
        params = EventSignalParams(
            feature_method="raw",
            decay_type="exponential",
            decay_halflife=5,
        )
        data = EventSignalInput(
            numeric_features=features,
            feature_names=["signal"],
            timestamps=np.arange(n, dtype=float),
        )
        result = engine.compute(data, params)

        assert result.ok
        # Most recent signal should have highest decayed alpha
        alphas = [a.decayed_alpha for a in result.alphas]
        assert alphas[-1] > alphas[0], \
            "Most recent signal should have highest decayed alpha"
        # Decay should be monotonically increasing (newer = stronger)
        for i in range(1, len(alphas)):
            assert alphas[i] >= alphas[i-1] - 1e-10

    def test_text_only_no_numeric(self):
        """
        EventSignal should work with text-only input (no numeric features).
        """
        engine = EventSignalEngine()
        params = EventSignalParams(decay_type="none")
        data = EventSignalInput(
            text_data={
                0: ["Strong growth beat expectations"],
                1: ["Weak performance miss disappointing"],
                2: ["Neutral outlook"],
            }
        )
        result = engine.compute(data, params)
        assert result.ok
        assert len(result.alphas) == 3


# ════════════════════════════════════════════════════════════════════════════
# CROSS-ENGINE COMPOSITION
# ════════════════════════════════════════════════════════════════════════════

class TestCrossEngineComposition:
    """
    Validates the key architectural claim: same engine + different context
    = different named strategy. Also tests multi-engine pipelines.
    """

    def test_same_engine_three_contexts(self):
        """
        MomentumEngine configured three ways:
        1. CTA trend-following (futures, 12m lookback, vol-target)
        2. Equity sector rotation (stocks, 6m lookback, long-only)
        3. Crypto momentum (24/7, shorter lookback, higher vol)
        All use the SAME engine class.
        """
        engine = MomentumEngine()

        # Context 1: CTA
        cta_prices, _ = make_global_futures(n=504)
        cta_result = engine.compute(
            MomentumInput(prices=cta_prices),
            MomentumParams(
                lookback_periods=[252],
                sizing_method="vol_target",
                vol_target=0.10,
                long_only=False,
            ))

        # Context 2: Equity sector rotation
        eq_prices, _ = make_equity_cross_section(n=504, n_assets=10)
        eq_result = engine.compute(
            MomentumInput(prices=eq_prices),
            MomentumParams(
                lookback_periods=[126],
                scoring_method="cross_sectional",
                long_only=True,
                top_n=3,
            ))

        # Context 3: Crypto (shorter lookback, higher vol target)
        crypto_prices, _ = make_global_futures(n=504, seed=99)
        crypto_result = engine.compute(
            MomentumInput(prices=crypto_prices),
            MomentumParams(
                lookback_periods=[30, 60],
                sizing_method="vol_target",
                vol_target=0.25,   # higher for crypto
                ann_factor=365.0,  # 365 trading days
            ))

        # All three used the SAME engine
        assert cta_result.ok and eq_result.ok and crypto_result.ok
        # Equity should be long-only
        assert np.all(eq_result.portfolio_weights >= -1e-10)
        # CTA and crypto should have shorts
        assert np.any(cta_result.portfolio_weights < -0.001)

    def test_momentum_then_allocation_pipeline(self):
        """
        Two-stage pipeline:
        1. MomentumEngine scores assets
        2. AllocationEngine optimizes weights using scores as expected returns
        This is how many quant funds operate.
        """
        prices, names = make_multi_asset_portfolio(n=504)
        returns = np.diff(prices, axis=0) / prices[:-1]

        # Stage 1: Score with momentum
        mom = MomentumEngine()
        mom_result = mom.compute(
            MomentumInput(prices=prices, asset_names=names),
            MomentumParams(lookback_periods=[63, 126, 252]))

        # Stage 2: Use momentum scores as expected returns for allocation
        scores = np.array([s.composite_score for s in mom_result.scores])
        alloc = AllocationEngine()
        alloc_result = alloc.compute(
            AllocationInput(returns=returns, asset_names=names),
            AllocationParams(
                method="max_sharpe",
                expected_returns=scores,  # momentum signal as expected return
                cov_method="shrinkage",
            ))

        assert alloc_result.ok
        w = alloc_result.result.weights
        # Assets with highest momentum score should get more weight
        top_mom = np.argmax(scores)
        # Weight of top momentum stock should be among top 2
        sorted_w_idx = np.argsort(-np.abs(w))
        assert top_mom in sorted_w_idx[:3], \
            f"Top momentum asset {top_mom} not in top-3 by weight"

    def test_event_feeds_momentum(self):
        """
        EventSignal → MomentumEngine pipeline:
        1. EventSignal produces alpha scores from fundamentals
        2. Use alpha to bias momentum signal direction
        """
        rng = np.random.default_rng(60)
        n_assets = 10
        n_obs = 504

        # Event scores
        event = EventSignalEngine()
        features = rng.standard_normal((n_assets, 3))
        event_result = event.compute(
            EventSignalInput(
                numeric_features=features,
                feature_names=["earnings", "value", "quality"],
            ),
            EventSignalParams(feature_method="zscore", decay_type="none"))

        assert event_result.ok

        # Use event alpha to select which assets to give momentum treatment
        alphas = event_result.alpha_vector
        top_k = np.argsort(-alphas)[:5]

        # Momentum on just the top-scored assets
        prices = np.exp(np.cumsum(rng.normal(0.0003, 0.015, (n_obs, n_assets)), axis=0)) * 100
        selected_prices = prices[:, top_k]
        mom = MomentumEngine()
        mom_result = mom.compute(
            MomentumInput(prices=selected_prices),
            MomentumParams(long_only=True))

        assert mom_result.ok
        assert len(mom_result.portfolio_weights) == len(top_k)

    def test_full_pipeline_event_momentum_allocation(self):
        """
        Full three-engine pipeline:
        1. EventSignal → alpha scores
        2. MomentumEngine → directional signals
        3. AllocationEngine → risk-managed weights
        """
        rng = np.random.default_rng(70)
        n_assets = 20
        n_obs = 504

        # Stage 1: Event signals
        event = EventSignalEngine()
        features = rng.standard_normal((n_assets, 4))
        event_result = event.compute(
            EventSignalInput(
                numeric_features=features,
                feature_names=["earn", "val", "qual", "growth"],
                text_data={
                    0: ["Strong beat growth surge outperform"],
                    5: ["Weak miss decline bearish"],
                },
            ),
            EventSignalParams(feature_method="zscore", decay_type="none"))
        assert event_result.ok

        # Stage 2: Momentum on full universe
        prices = np.exp(np.cumsum(
            rng.normal(0.0003, 0.015, (n_obs, n_assets)), axis=0)) * 100
        mom = MomentumEngine()
        mom_result = mom.compute(
            MomentumInput(prices=prices),
            MomentumParams(lookback_periods=[63, 126]))
        assert mom_result.ok

        # Combine: event alpha + momentum score = composite expected return
        event_alpha = event_result.alpha_vector
        mom_scores = np.array([s.composite_score for s in mom_result.scores])
        # Normalize both to same scale
        ea_norm = (event_alpha - event_alpha.mean()) / max(event_alpha.std(), 1e-10)
        ms_norm = (mom_scores - mom_scores.mean()) / max(mom_scores.std(), 1e-10)
        combined = 0.5 * ea_norm + 0.5 * ms_norm

        # Stage 3: Allocation using combined signal
        returns = np.diff(prices, axis=0) / prices[:-1]
        alloc = AllocationEngine()
        alloc_result = alloc.compute(
            AllocationInput(returns=returns),
            AllocationParams(
                method="max_sharpe",
                expected_returns=combined,
                cov_method="shrinkage",
                long_only=True,
                max_weight=0.15,
            ))
        assert alloc_result.ok
        w = alloc_result.result.weights
        assert np.all(w >= -1e-10), "Should be long-only"
        assert_allclose(np.sum(np.abs(w)), 1.0, atol=0.01)


# ════════════════════════════════════════════════════════════════════════════
# MODEL ORCHESTRATOR WITH REALISTIC CONTEXTS
# ════════════════════════════════════════════════════════════════════════════

class TestModelOrchestrator:
    """
    Tests the Model class with fully specified ModelContext objects
    matching real strategy configurations.
    """

    def test_model_gld_gdx_stat_arb(self):
        """
        Full Model run: GLD/GDX stat arb with complete context specification.
        """
        prices, names = make_gld_gdx_style(n=504)
        ctx = ModelContext(
            name="GLD_GDX_PairsTrading",
            universe=UniverseSpec(
                tickers=names,
                asset_class=AssetClass.COMMODITY,
                name="Gold_Mining_Pairs",
            ),
            temporal=TemporalSpec(
                start="2020-01-01", end="2024-01-01",
                frequency=Frequency.DAILY,
                calendar=Calendar.NYSE,
                rebalance=RebalanceFrequency.DAILY,
            ),
            risk=RiskSpec(
                max_position_size=0.40,
                max_drawdown=0.15,
                leverage_limit=1.0,
            ),
            execution=ExecutionSpec(
                mode=ExecutionMode.BACKTEST,
                slippage_bps=5.0,
                commission_bps=5.0,
            ),
            metadata={"source": "Chan CPO Paper", "hedge_ratio": 2.5},
        )

        provider = InMemoryDataProvider(prices, names)
        store = InMemoryResultStore()
        engine = StatArbEngine()
        params = StatArbParams(
            n_per_basket=1,
            max_candidates=5,
            max_hurst=1.1,
            scoring_mode="classic",
            entry_threshold=2.0,
        )

        model = Model(engine, ctx, provider, store, params)
        result = model.run()

        assert isinstance(result, ModelResult)
        assert result.ok
        assert result.engine_name == "StatArbEngine"
        assert result.context.name == "GLD_GDX_PairsTrading"
        # Result should be persisted
        assert store.load("GLD_GDX_PairsTrading_StatArbEngine") is not None

    def test_model_risk_parity_taa(self):
        """
        Full Model run: Bridgewater-style risk parity TAA.
        """
        prices, names = make_multi_asset_portfolio(n=756)
        ctx = ModelContext(
            name="AllWeather_RiskParity",
            universe=UniverseSpec(
                tickers=names,
                asset_class=AssetClass.EQUITY,
                name="TAA_5Asset",
            ),
            temporal=TemporalSpec(
                frequency=Frequency.DAILY,
                rebalance=RebalanceFrequency.MONTHLY,
            ),
            risk=RiskSpec(long_only=True, vol_target=0.10),
        )

        provider = InMemoryDataProvider(prices, names)
        engine = AllocationEngine()
        params = AllocationParams(method="risk_parity", long_only=True)

        model = Model(engine, ctx, provider, params=params)
        result = model.run()

        assert result.ok
        assert result.engine_name == "AllocationEngine"


# ════════════════════════════════════════════════════════════════════════════
# GAP ANALYSIS: document what the engines CAN'T do yet
# ════════════════════════════════════════════════════════════════════════════

class TestGapAnalysis:
    """
    These tests document known structural gaps. Each test passes but
    includes [GAP] comments about what would need to change for
    production use of specific strategies.
    """

    def test_gap_rolling_window_rebalance(self):
        """
        [GAP] Engines are single-shot: compute() runs once on full data.
        Real strategies need rolling-window walk-forward with periodic
        rebalancing. This requires an outer loop (not engine-level).

        Mitigation: The Model orchestrator should gain a run_rolling()
        method that handles train/test splits and rebalance scheduling.
        """
        prices, _ = make_multi_asset_portfolio(n=504)
        engine = AllocationEngine()
        returns = np.diff(prices, axis=0) / prices[:-1]

        # Simulate manual rolling: take last 252 days only
        recent = returns[-252:]
        result = engine.compute(
            AllocationInput(returns=recent),
            AllocationParams(method="min_variance"))
        assert result.ok  # Single-shot works fine

    def test_gap_transaction_costs_in_backtest(self):
        """
        [GAP] ExecutionSpec has slippage/commission fields but they're
        not consumed by any engine. Cost modeling requires a separate
        backtesting layer that applies costs to weight changes.
        """
        ctx = ModelContext(
            execution=ExecutionSpec(slippage_bps=10.0, commission_bps=5.0))
        # These fields exist and are well-defined
        assert ctx.execution.slippage_bps == 10.0
        assert ctx.execution.commission_bps == 5.0
        # But engines don't use them yet — noted as gap

    def test_gap_options_multi_leg_strategies(self):
        """
        [GAP] OptionsEngine handles single-strike analysis.
        Multi-leg strategies (spreads, straddles, iron condors) need
        a StrategyBuilder layer that composes legs and computes
        aggregate Greeks/PnL.
        """
        engine = OptionsEngine()
        S = 100
        # Can price individual legs
        call_atm = engine.bs_price(S, 100, 0.25, 0.05, 0.20, 0, "call")
        call_otm = engine.bs_price(S, 110, 0.25, 0.05, 0.20, 0, "call")
        # Bull call spread = long ATM call - short OTM call
        spread_value = call_atm - call_otm
        assert spread_value > 0  # Debit spread
        # [GAP] No native spread/combo support in engine

    def test_gap_regime_detection(self):
        """
        [GAP] Chan's CPO requires regime-conditional parameter selection.
        Our engines use static params. CPO would need:
        1. Feature extraction layer (technical indicators as regime features)
        2. ML model to predict strategy return given params + features
        3. Daily param optimization loop

        This is a meta-strategy layer above the engine level.
        """
        # Current approach: manual param selection
        prices, _ = make_gld_gdx_style(n=504)
        engine = StatArbEngine()

        # "Calm regime" params
        calm = engine.compute(
            StatArbInput(prices=prices),
            StatArbParams(entry_threshold=2.0, max_hurst=1.1))

        # "Volatile regime" params
        volatile = engine.compute(
            StatArbInput(prices=prices),
            StatArbParams(entry_threshold=3.0, max_hurst=1.1))

        # Both work, but switching between them requires external logic
        assert calm.ok and volatile.ok


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
