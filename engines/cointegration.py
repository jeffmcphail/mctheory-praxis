"""ENGINE 1: Cointegration/Mean-Reversion (Burgess). Pure math on residual series."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from scipy import stats as sp_stats
from engines.base import (
    TimeSeriesEngine, EngineParams, EngineInput, EngineOutput, EngineStatus, EngineValidationError)

@dataclass(frozen=True)
class StatArbParams(EngineParams):
    n_per_basket: int = 5; max_candidates: int = 50
    regression_method: Literal["ols","ridge"] = "ridge"; ridge_alpha: float = 0.01
    significance: float = 0.05; min_half_life: float = 5.0; max_half_life: float = 126.0
    max_hurst: float = 0.5; vr_lags: list[int] = field(default_factory=lambda: [2,5,10,20,50])
    scoring_mode: Literal["classic","composite"] = "classic"
    score_weights: dict[str,float] = field(default_factory=lambda: {"adf":.25,"hurst":.25,"half_life":.25,"variance_ratio":.25})
    top_k: int = 10
    optimization_method: Literal["equal","min_variance","max_sharpe"] = "min_variance"
    long_only: bool = False; max_weight: float = 0.40; shrinkage: float = 0.1
    zscore_lookback: int = 63; entry_threshold: float = 2.0; exit_threshold: float = 0.5

@dataclass
class StatArbInput(EngineInput):
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    asset_names: list[str] = field(default_factory=list)
    def shape_summary(self): return f"prices={self.prices.shape}"

@dataclass
class RegressionResult:
    target_idx: int; basket_indices: list[int]; betas: np.ndarray
    residuals: np.ndarray; r_squared: float; adj_r_squared: float

@dataclass
class StationarityResult:
    adf_statistic: float; adf_pvalue: float; adf_is_stationary: bool
    hurst_exponent: float; half_life: float; variance_ratios: dict[int,float]

@dataclass
class CandidateBasket:
    target_idx: int; basket_indices: list[int]; regression: RegressionResult
    stationarity: StationarityResult; composite_score: float = 0.0; rank: int = 0

@dataclass
class PortfolioWeights:
    basket_indices: list[int]; weights: np.ndarray; method: str = ""

@dataclass
class TradingSignal:
    basket_indices: list[int]; zscore_series: np.ndarray; current_zscore: float
    signal: Literal["long","short","flat"]; entry_level: float; exit_level: float

@dataclass
class StatArbOutput(EngineOutput):
    candidates: list[CandidateBasket] = field(default_factory=list)
    top_baskets: list[CandidateBasket] = field(default_factory=list)
    portfolios: list[PortfolioWeights] = field(default_factory=list)
    signals: list[TradingSignal] = field(default_factory=list)
    n_candidates_scanned: int = 0; n_passed_stationarity: int = 0

class StatArbEngine(TimeSeriesEngine):
    def default_params(self): return StatArbParams()
    def validate_input(self, data, params):
        p = data.prices if isinstance(data, StatArbInput) else data
        self._validate_price_matrix(p, min_obs=60)
    def build_input(self, prices):
        return StatArbInput(prices=prices, asset_names=[f"A{i}" for i in range(prices.shape[1])])

    def compute(self, data, params: StatArbParams) -> StatArbOutput:
        if isinstance(data, np.ndarray): data = self.build_input(data)
        self.validate_input(data, params)
        prices = data.prices; n_obs, n_assets = prices.shape
        candidates = self._scan(prices, params)
        tested = []
        for reg in candidates:
            s = self._test_stat(reg.residuals, params)
            b = CandidateBasket(reg.target_idx, reg.basket_indices, reg, s)
            if s.adf_is_stationary and s.hurst_exponent < params.max_hurst and params.min_half_life <= s.half_life <= params.max_half_life:
                tested.append(b)
        self._score(tested, params)
        top = sorted(tested, key=lambda c: c.composite_score, reverse=True)[:params.top_k]
        for i, b in enumerate(top): b.rank = i + 1
        portfolios = [self._opt_weights(prices, b, params) for b in top]
        signals = [self._gen_signal(prices, b, params) for b in top]
        return StatArbOutput(status=EngineStatus.SUCCESS if top else EngineStatus.PARTIAL,
            candidates=tested, top_baskets=top, portfolios=portfolios, signals=signals,
            n_candidates_scanned=len(candidates), n_passed_stationarity=len(tested),
            diagnostics={"n_assets":n_assets,"n_obs":n_obs})

    def _scan(self, prices, params):
        results = []
        for t in range(min(params.max_candidates, prices.shape[1])):
            r = self._stepwise(prices, t, params.n_per_basket, params)
            if r: results.append(r)
        return results

    def _stepwise(self, prices, tidx, nvars, params):
        y = prices[:, tidx]; avail = [i for i in range(prices.shape[1]) if i != tidx]
        sel = []; best_r2 = -np.inf
        for _ in range(min(nvars, len(avail))):
            bi, br2 = -1, -np.inf
            for idx in avail:
                r2 = self._quick_r2(y, prices[:, sel + [idx]])
                if r2 > br2: br2, bi = r2, idx
            if bi == -1 or br2 <= best_r2: break
            sel.append(bi); avail.remove(bi); best_r2 = br2
        if not sel: return None
        return self._regress(y, prices[:, sel], tidx, sel, params)

    def _regress(self, y, X, tidx, bidx, params):
        n, p = X.shape; Xa = np.column_stack([np.ones(n), X])
        if params.regression_method == "ridge":
            lam = params.ridge_alpha * np.eye(Xa.shape[1]); lam[0,0] = 0
            b = np.linalg.solve(Xa.T @ Xa + lam, Xa.T @ y)
        else:
            b, *_ = np.linalg.lstsq(Xa, y, rcond=None)
        res = y - Xa @ b; ssr = np.sum(res**2); sst = np.sum((y-y.mean())**2)
        r2 = 1 - ssr/sst if sst > 0 else 0
        adj = 1 - (1-r2)*(n-1)/(n-p-1) if n > p+1 else r2
        return RegressionResult(tidx, bidx, b, res, r2, adj)

    @staticmethod
    def _quick_r2(y, X):
        Xa = np.column_stack([np.ones(len(y)), X])
        try: b, *_ = np.linalg.lstsq(Xa, y, rcond=None)
        except: return -np.inf
        return 1 - np.sum((y - Xa @ b)**2) / max(np.sum((y - y.mean())**2), 1e-10)

    def _test_stat(self, resid, params):
        adf_s, adf_p = self._adf(resid)
        h = self._hurst(resid); hl = self._half_life(resid)
        vr = {lag: self._vr(resid, lag) for lag in params.vr_lags if lag < len(resid)}
        return StationarityResult(adf_s, adf_p, adf_p < params.significance, h, hl, vr)

    @staticmethod
    def _adf(s):
        dy = np.diff(s); yl = s[:-1]; n = len(dy)
        X = np.column_stack([np.ones(n), yl])
        try: b, *_ = np.linalg.lstsq(X, dy, rcond=None)
        except: return 0.0, 1.0
        r = dy - X @ b
        se = np.sqrt(np.sum(r**2)/max(n-2,1)) / np.sqrt(max(np.sum((yl-yl.mean())**2),1e-10))
        t = b[1]/se if se > 0 else 0
        if t < -3.43: p = 0.005
        elif t < -2.86: p = 0.03
        elif t < -2.57: p = 0.07
        elif t < -1.94: p = 0.15
        else: p = min(1.0, 0.5 + 0.1*(t+1.62))
        return t, p

    @staticmethod
    def _hurst(s):
        n = len(s)
        if n < 20: return 0.5
        max_sz = n // 4; min_sz = 8
        if max_sz <= min_sz: return 0.5
        sizes = np.unique(np.geomspace(min_sz, max_sz, min(20, max_sz-min_sz)).astype(int))
        lsz, lrs = [], []
        for sz in sizes:
            nc = n // sz
            if nc < 2: continue
            rs = []
            for i in range(nc):
                c = s[i*sz:(i+1)*sz]; m = c - c.mean(); cd = np.cumsum(m)
                R = cd.max() - cd.min(); S = c.std(ddof=1)
                if S > 1e-10: rs.append(R/S)
            if rs: lsz.append(np.log(sz)); lrs.append(np.log(np.mean(rs)))
        if len(lsz) < 3: return 0.5
        slope, *_ = sp_stats.linregress(lsz, lrs)
        return float(np.clip(slope, 0.0, 1.0))

    @staticmethod
    def _half_life(s):
        dy = np.diff(s); yl = s[:-1]
        X = np.column_stack([np.ones(len(dy)), yl])
        try: b, *_ = np.linalg.lstsq(X, dy, rcond=None)
        except: return np.inf
        theta = -b[1]
        return np.log(2)/theta if theta > 0 else np.inf

    @staticmethod
    def _vr(s, lag):
        r1 = np.diff(s); rq = s[lag:] - s[:-lag]
        v1 = np.var(r1, ddof=1); vq = np.var(rq, ddof=1)
        return float(vq/(lag*v1)) if v1 > 0 else 1.0

    def _score(self, cands, params):
        if not cands: return
        if params.scoring_mode == "classic":
            for c in cands: c.composite_score = -c.stationarity.adf_statistic
            return
        adf = np.array([c.stationarity.adf_statistic for c in cands])
        hu = np.array([c.stationarity.hurst_exponent for c in cands])
        hl = np.array([c.stationarity.half_life for c in cands])
        vr = np.array([np.mean(list(c.stationarity.variance_ratios.values())) if c.stationarity.variance_ratios else 1 for c in cands])
        def norm(v): r=v.max()-v.min(); return (v.max()-v)/r if r>0 else np.full(len(v),.5)
        w = params.score_weights
        comp = norm(adf)*w.get("adf",.25) + norm(hu)*w.get("hurst",.25) + norm(np.abs(hl-(params.min_half_life+params.max_half_life)/2))*w.get("half_life",.25) + norm(vr)*w.get("variance_ratio",.25)
        for i, c in enumerate(cands): c.composite_score = float(comp[i])

    def _opt_weights(self, prices, basket, params):
        idx = [basket.target_idx] + basket.basket_indices
        ret = np.diff(prices[:, idx], axis=0) / prices[:-1, idx]; n = ret.shape[1]
        if params.optimization_method == "equal": w = np.ones(n)/n
        elif params.optimization_method == "min_variance":
            cov = np.cov(ret, rowvar=False)
            cov_s = (1-params.shrinkage)*cov + params.shrinkage*np.diag(np.diag(cov))
            try: inv = np.linalg.inv(cov_s); o = np.ones(n); w = inv@o/(o@inv@o)
            except: w = np.ones(n)/n
        else:
            mu = ret.mean(axis=0); cov = np.cov(ret, rowvar=False)
            cov_s = (1-params.shrinkage)*cov + params.shrinkage*np.diag(np.diag(cov))
            try: w = np.linalg.inv(cov_s)@mu; w = w/max(np.sum(np.abs(w)),1e-10)
            except: w = np.ones(n)/n
        if params.long_only: w = np.maximum(w, 0)
        w = np.clip(w, -params.max_weight, params.max_weight)
        t = np.sum(np.abs(w))
        if t > 0: w = w/t
        return PortfolioWeights(idx, w, params.optimization_method)

    def _gen_signal(self, prices, basket, params):
        res = basket.regression.residuals; lb = min(params.zscore_lookback, len(res)-1)
        idx = [basket.target_idx] + basket.basket_indices
        if lb < 10: return TradingSignal(idx, np.zeros(len(res)), 0, "flat", params.entry_threshold, params.exit_threshold)
        zs = np.zeros(len(res))
        for i in range(lb, len(res)):
            w = res[i-lb:i]; mu, sig = w.mean(), w.std()
            if sig > 0: zs[i] = (res[i]-mu)/sig
        cz = zs[-1]
        sig = "short" if cz >= params.entry_threshold else "long" if cz <= -params.entry_threshold else "flat"
        return TradingSignal(idx, zs, float(cz), sig, params.entry_threshold, params.exit_threshold)
