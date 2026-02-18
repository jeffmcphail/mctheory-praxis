"""Comprehensive tests for all 5 engine implementations + orchestrator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from engines.base import EngineValidationError, EngineStatus, ModelEngine, TimeSeriesEngine, SurfaceEngine
from engines.cointegration import StatArbEngine, StatArbParams, StatArbInput
from engines.momentum import MomentumEngine, MomentumParams, MomentumInput
from engines.allocation import AllocationEngine, AllocationParams, AllocationInput
from engines.options import OptionsEngine, OptionsParams, OptionsInput, GreeksVector
from engines.event_signal import EventSignalEngine, EventSignalParams, EventSignalInput
from engines.model import Model, ModelResult
from engines.context.model_context import ModelContext, UniverseSpec, TemporalSpec, AssetClass, Calendar
from engines.adapters.providers import InMemoryDataProvider, InMemoryResultStore
from tests.test_data_generators import *

# ═══ ENGINE 1: COINTEGRATION ═══════════════════════════════════════════════
class TestStatArbEngine:
    def setup_method(self):
        self.engine = StatArbEngine()
        self.prices = generate_cointegrated_series(504, 8)

    def test_default_params(self):
        assert isinstance(self.engine.default_params(), StatArbParams)

    def test_validate_valid(self):
        self.engine.validate_input(self.prices, self.engine.default_params())

    def test_validate_rejects_short(self):
        with pytest.raises(EngineValidationError): self.engine.validate_input(self.prices[:20], self.engine.default_params())

    def test_validate_rejects_nan(self):
        bad = np.full_like(self.prices, np.nan)
        with pytest.raises(EngineValidationError): self.engine.validate_input(bad, self.engine.default_params())

    def test_compute_basic(self):
        r = self.engine.compute(self.prices, StatArbParams(max_candidates=8, n_per_basket=3, top_k=3))
        assert r.ok; assert r.n_candidates_scanned > 0

    def test_compute_with_wrapper(self):
        d = StatArbInput(prices=self.prices, asset_names=[f"E{i}" for i in range(8)])
        r = self.engine.compute(d, StatArbParams(max_candidates=8, n_per_basket=3, top_k=3))
        assert r.ok

    def test_cointegrated_finds_candidates(self):
        p = generate_cointegrated_series(504, 8, mr_speed=0.15, noise=0.003)
        r = self.engine.compute(p, StatArbParams(max_candidates=8, n_per_basket=3, max_hurst=1.1, max_half_life=504, min_half_life=0.5, significance=0.2))
        assert r.n_passed_stationarity > 0

    def test_scoring_modes(self):
        for mode in ["classic", "composite"]:
            r = self.engine.compute(self.prices, StatArbParams(max_candidates=8, n_per_basket=3, scoring_mode=mode, max_hurst=1.1, max_half_life=252, min_half_life=1))
            assert isinstance(r.candidates, list)

    def test_optimization_methods(self):
        for m in ["equal", "min_variance", "max_sharpe"]:
            r = self.engine.compute(self.prices, StatArbParams(max_candidates=8, n_per_basket=3, top_k=2, optimization_method=m, max_hurst=1.1, max_half_life=252, min_half_life=1))
            if r.portfolios:
                assert abs(np.sum(np.abs(r.portfolios[0].weights)) - 1.0) < 0.01

    def test_signal_generation(self):
        r = self.engine.compute(self.prices, StatArbParams(max_candidates=8, n_per_basket=3, top_k=2, max_hurst=1.1, max_half_life=252, min_half_life=1))
        for s in r.signals: assert s.signal in ("long","short","flat")

    def test_adf_ou_process(self):
        rng = np.random.default_rng(42); ou = np.zeros(500)
        for t in range(1,500): ou[t] = ou[t-1]*0.95 + rng.normal(0,0.1)
        _, p = StatArbEngine._adf(ou)
        assert p < 0.1

    def test_half_life(self):
        rng = np.random.default_rng(42); ou = np.zeros(1000)
        for t in range(1,1000): ou[t] = ou[t-1]*0.95 + rng.normal(0,0.1)
        hl = StatArbEngine._half_life(ou)
        assert 5 < hl < 30

    def test_variance_ratio_rw(self):
        rng = np.random.default_rng(42); rw = np.cumsum(rng.normal(0,1,1000))
        vr = StatArbEngine._vr(rw, 10)
        assert 0.7 < vr < 1.3

# ═══ ENGINE 2: MOMENTUM ════════════════════════════════════════════════════
class TestMomentumEngine:
    def setup_method(self):
        self.engine = MomentumEngine()
        self.prices = generate_trending_series(504, 10)

    def test_compute_basic(self):
        r = self.engine.compute(self.prices, MomentumParams())
        assert r.ok; assert len(r.scores)==10; assert r.portfolio_weights.shape==(10,)

    def test_contrarian_sign_flip(self):
        mr = self.engine.compute(self.prices, MomentumParams(signal_sign=1.0))
        cr = self.engine.compute(self.prices, MomentumParams(signal_sign=-1.0))
        for ms, cs in zip(mr.scores, cr.scores):
            if ms.composite_score != 0:
                assert ms.composite_score * cs.composite_score <= 0

    def test_cross_sectional(self):
        r = self.engine.compute(self.prices, MomentumParams(scoring_method="cross_sectional", top_n=3, bottom_n=3))
        assert set(r.rankings) == set(range(10))

    def test_ma_types(self):
        for ma in ["sma","ema","dema","hull"]:
            assert self.engine.compute(self.prices, MomentumParams(ma_type=ma)).ok

    def test_breakout_methods(self):
        for bk in ["donchian","bollinger"]:
            assert self.engine.compute(self.prices, MomentumParams(breakout_method=bk)).ok

    def test_sizing_methods(self):
        for sz in ["equal","inverse_vol","vol_target"]:
            assert self.engine.compute(self.prices, MomentumParams(sizing_method=sz)).ok

    def test_long_only(self):
        r = self.engine.compute(self.prices, MomentumParams(long_only=True))
        assert np.all(r.portfolio_weights >= -1e-10)

    def test_weights_sum_one(self):
        r = self.engine.compute(self.prices, MomentumParams())
        t = np.sum(np.abs(r.portfolio_weights))
        if t > 0: assert abs(t-1) < 0.01

# ═══ ENGINE 3: ALLOCATION ══════════════════════════════════════════════════
class TestAllocationEngine:
    def setup_method(self):
        self.engine = AllocationEngine()
        self.prices = generate_random_walks(504, 6)

    def test_compute_from_prices(self):
        r = self.engine.compute(self.prices, AllocationParams())
        assert r.ok; assert r.result.weights.shape == (6,)

    def test_all_methods(self):
        for m in ["equal_weight","min_variance","max_sharpe","risk_parity","target_vol","hrp","inverse_vol","max_diversification"]:
            r = self.engine.compute(self.prices, AllocationParams(method=m))
            assert r.ok, f"Failed: {m}"

    def test_black_litterman(self):
        P = np.zeros((1,6)); P[0,0]=1; P[0,1]=-1
        r = self.engine.compute(self.prices, AllocationParams(method="black_litterman", views_P=P, views_Q=np.array([0.02])))
        assert r.ok; assert r.result.weights[0] > r.result.weights[1]

    def test_long_only(self):
        r = self.engine.compute(self.prices, AllocationParams(method="min_variance", long_only=True))
        assert np.all(r.result.weights >= -1e-10)

    def test_risk_parity(self):
        r = self.engine.compute(self.prices, AllocationParams(method="risk_parity"))
        assert r.result.diversification_ratio >= 0.9

    def test_cov_methods(self):
        for m in ["sample","shrinkage","ewma"]:
            r = self.engine.compute(self.prices, AllocationParams(cov_method=m))
            assert r.covariance.matrix.shape == (6,6)

    def test_sub_component(self):
        ret = np.diff(self.prices, axis=0)/self.prices[:-1]
        w = self.engine.allocate(ret)
        assert w.shape == (6,); assert abs(np.sum(np.abs(w))-1) < 0.05

    def test_turnover_constraint(self):
        ret = np.diff(self.prices, axis=0)/self.prices[:-1]
        d = AllocationInput(returns=ret, current_weights=np.ones(6)/6)
        r = self.engine.compute(d, AllocationParams(method="max_sharpe", max_turnover=0.2))
        assert r.turnover <= 0.21

# ═══ ENGINE 4: OPTIONS ═════════════════════════════════════════════════════
class TestOptionsEngine:
    def setup_method(self):
        self.engine = OptionsEngine()
        self.data = generate_options_data()

    def _inp(self):
        d = self.data
        return OptionsInput(underlying_price=d["underlying_price"], risk_free_rate=d["risk_free_rate"],
            dividend_yield=d["dividend_yield"], strikes=d["strikes"], expiries=d["expiries"],
            market_prices=d["market_prices"], underlying_history=d["underlying_history"])

    def test_validate(self):
        self.engine.validate_input(self._inp(), OptionsParams())

    def test_validate_rejects_bad_price(self):
        i = self._inp(); i.underlying_price = -1
        with pytest.raises(EngineValidationError): self.engine.validate_input(i, OptionsParams())

    def test_bs_call(self):
        p = OptionsEngine.bs_price(100,100,1,.05,.20)
        assert 8 < p < 15

    def test_put_call_parity(self):
        S,K,T,r,s,q = 100,100,1,.05,.20,0
        c = OptionsEngine.bs_price(S,K,T,r,s,q,"call")
        p = OptionsEngine.bs_price(S,K,T,r,s,q,"put")
        assert abs((c-p) - (S*np.exp(-q*T)-K*np.exp(-r*T))) < 0.01

    def test_greeks(self):
        g = OptionsEngine.bs_greeks(100,100,0.25,0.05,0.20)
        assert 0.4 < g.delta < 0.7; assert g.gamma > 0; assert g.vega > 0; assert g.theta < 0

    def test_greeks_deep_itm(self):
        assert OptionsEngine.bs_greeks(150,100,0.25,0.05,0.20).delta > 0.9

    def test_greeks_deep_otm(self):
        assert OptionsEngine.bs_greeks(50,100,0.25,0.05,0.20).delta < 0.1

    def test_compute_full(self):
        r = self.engine.compute(self._inp(), OptionsParams())
        assert r.ok; assert r.surface.iv_matrix.shape == (11,4); assert r.realized_vol > 0

    def test_iv_roundtrip(self):
        r = self.engine.compute(self._inp(), OptionsParams())
        v = ~np.isnan(r.surface.iv_matrix) & ~np.isnan(self.data["iv_matrix"])
        if v.sum() > 0:
            assert np.max(np.abs(r.surface.iv_matrix[v] - self.data["iv_matrix"][v])) < 0.01

    def test_vol_signals(self):
        r = self.engine.compute(self._inp(), OptionsParams(iv_rv_threshold=0.001))
        assert isinstance(r.signals, list)

    def test_greeks_matrix(self):
        r = self.engine.compute(self._inp(), OptionsParams())
        assert len(r.greeks_matrix) == 11; assert len(r.greeks_matrix[0]) == 4

# ═══ ENGINE 7: EVENT/SIGNAL ════════════════════════════════════════════════
class TestEventSignalEngine:
    def setup_method(self):
        self.engine = EventSignalEngine()
        self.data = generate_event_data(20, 5)

    def _inp(self):
        d = self.data
        return EventSignalInput(numeric_features=d["numeric_features"],
            feature_names=d["feature_names"], asset_names=d["asset_names"], text_data=d["text_data"])

    def test_compute(self):
        r = self.engine.compute(self._inp(), EventSignalParams())
        assert r.ok; assert len(r.alphas)==20; assert r.alpha_vector.shape==(20,)

    def test_feature_methods(self):
        for m in ["zscore","rank","percentile","raw"]:
            assert self.engine.compute(self._inp(), EventSignalParams(feature_method=m)).ok

    def test_ranking(self):
        r = self.engine.compute(self._inp(), EventSignalParams())
        assert set(a.rank for a in r.alphas) == set(range(20))

    def test_decay(self):
        i = self._inp(); i.timestamps = np.arange(20, dtype=float)
        r = self.engine.compute(i, EventSignalParams(decay_type="exponential", decay_halflife=5))
        for a in r.alphas: assert abs(a.decayed_alpha) <= abs(a.composite_alpha) + 1e-10

    def test_numeric_only(self):
        assert self.engine.compute(EventSignalInput(numeric_features=np.random.randn(10,3), feature_names=["a","b","c"]), EventSignalParams()).ok

    def test_text_only(self):
        assert self.engine.compute(EventSignalInput(text_data={0:["strong growth beat"],1:["weak decline miss"]}), EventSignalParams()).ok

# ═══ MODEL ORCHESTRATOR ════════════════════════════════════════════════════
class TestModel:
    def test_stat_arb_model(self):
        p = generate_cointegrated_series(504, 8)
        m = Model(StatArbEngine(), ModelContext(name="ETF StatArb", universe=UniverseSpec(tickers=[f"E{i}" for i in range(8)])),
            InMemoryDataProvider(p), params=StatArbParams(max_candidates=8, n_per_basket=3, top_k=3))
        r = m.run(); assert r.engine_name == "StatArbEngine"

    def test_momentum_model(self):
        p = generate_trending_series(504, 10)
        r = Model(MomentumEngine(), ModelContext(name="CTA"), InMemoryDataProvider(p)).run()
        assert r.ok

    def test_allocation_model(self):
        p = generate_random_walks(504, 6)
        r = Model(AllocationEngine(), ModelContext(name="RiskParity"), InMemoryDataProvider(p),
            params=AllocationParams(method="risk_parity")).run()
        assert r.ok

    def test_persistence(self):
        store = InMemoryResultStore()
        Model(AllocationEngine(), ModelContext(name="T"), InMemoryDataProvider(generate_random_walks(504,6)), result_store=store).run()
        assert store.exists("T_AllocationEngine")

    def test_same_engine_diff_context(self):
        e = MomentumEngine(); p = generate_trending_series(504,5); dp = InMemoryDataProvider(p)
        r1 = Model(e, ModelContext(name="CTA", universe=UniverseSpec(asset_class=AssetClass.FUTURES)), dp).run()
        r2 = Model(e, ModelContext(name="Equity Mom", universe=UniverseSpec(asset_class=AssetClass.EQUITY)), dp).run()
        r3 = Model(e, ModelContext(name="Crypto Mom", universe=UniverseSpec(asset_class=AssetClass.CRYPTO)), dp).run()
        assert r1.engine_name == r2.engine_name == r3.engine_name == "MomentumEngine"
        assert {r1.context.name, r2.context.name, r3.context.name} == {"CTA","Equity Mom","Crypto Mom"}

# ═══ TYPE HIERARCHY ════════════════════════════════════════════════════════
class TestHierarchy:
    def test_inheritance(self):
        assert issubclass(StatArbEngine, TimeSeriesEngine)
        assert issubclass(MomentumEngine, TimeSeriesEngine)
        assert issubclass(AllocationEngine, TimeSeriesEngine)
        assert issubclass(OptionsEngine, SurfaceEngine)
        assert issubclass(EventSignalEngine, TimeSeriesEngine)
        for c in [StatArbEngine, MomentumEngine, AllocationEngine, OptionsEngine, EventSignalEngine]:
            assert issubclass(c, ModelEngine)

# ═══ COMPOSITION ═══════════════════════════════════════════════════════════
class TestComposition:
    def test_momentum_then_allocation(self):
        p = generate_trending_series(504, 10)
        mr = MomentumEngine().compute(p, MomentumParams())
        ret = np.diff(p, axis=0)/p[:-1]
        top = np.argsort(mr.rankings)[:5]
        ar = AllocationEngine().compute(AllocationInput(returns=ret[:,top]), AllocationParams(method="risk_parity"))
        assert ar.ok; assert ar.result.weights.shape == (5,)

    def test_event_then_momentum(self):
        d = generate_event_data(20, 5); p = generate_trending_series(504, 20)
        er = EventSignalEngine().compute(EventSignalInput(numeric_features=d["numeric_features"], feature_names=d["feature_names"]), EventSignalParams())
        mr = MomentumEngine().compute(p, MomentumParams())
        combined = mr.portfolio_weights * er.alpha_vector
        assert combined.shape == (20,)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
