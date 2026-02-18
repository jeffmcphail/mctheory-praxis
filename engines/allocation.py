"""ENGINE 3: Portfolio Allocation. Standalone AND sub-component for other engines."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from engines.base import (
    TimeSeriesEngine, EngineParams, EngineInput, EngineOutput, EngineStatus, EngineValidationError)

@dataclass(frozen=True)
class AllocationParams(EngineParams):
    method: Literal["equal_weight","min_variance","max_sharpe","risk_parity",
        "target_vol","black_litterman","hrp","max_diversification","inverse_vol"] = "min_variance"
    cov_method: Literal["sample","shrinkage","ewma"] = "shrinkage"
    shrinkage_intensity: float = 0.2; ewma_halflife: int = 63
    risk_budget: np.ndarray|None = None
    views_P: np.ndarray|None = None; views_Q: np.ndarray|None = None; views_omega: np.ndarray|None = None
    tau: float = 0.05; vol_target: float = 0.10; ann_factor: float = 252.0
    long_only: bool = False; max_weight: float = 1.0; min_weight: float = -1.0
    max_turnover: float = 2.0; expected_returns: np.ndarray|None = None; risk_free_rate: float = 0.0

@dataclass
class AllocationInput(EngineInput):
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    asset_names: list[str] = field(default_factory=list)
    current_weights: np.ndarray|None = None

@dataclass
class CovarianceEstimate:
    matrix: np.ndarray; method: str; condition_number: float; effective_rank: int

@dataclass
class AllocationResult:
    weights: np.ndarray; method: str; expected_return: float; expected_vol: float
    sharpe_ratio: float; marginal_risk_contributions: np.ndarray; diversification_ratio: float

@dataclass
class AllocationOutput(EngineOutput):
    result: AllocationResult|None = None; covariance: CovarianceEstimate|None = None; turnover: float = 0.0

class AllocationEngine(TimeSeriesEngine):
    def default_params(self): return AllocationParams()
    def validate_input(self, data, params):
        r = data.returns if isinstance(data, AllocationInput) else data
        self._validate_price_matrix(r, min_obs=20)
    def build_input(self, prices):
        ret = np.diff(prices, axis=0)/prices[:-1]
        return AllocationInput(returns=ret, asset_names=[f"A{i}" for i in range(prices.shape[1])])

    def compute(self, data, params: AllocationParams) -> AllocationOutput:
        if isinstance(data, np.ndarray):
            data = self.build_input(data) if data.shape[0] > data.shape[1] else AllocationInput(returns=data)
        self.validate_input(data, params); ret = data.returns; n_obs, n = ret.shape
        cov_est = self._est_cov(ret, params)
        w = self._optimize(ret, cov_est.matrix, params)
        if params.long_only: w = np.maximum(w, 0)
        w = np.clip(w, params.min_weight, params.max_weight)
        if data.current_weights is not None and params.max_turnover < 2.0:
            turn = np.sum(np.abs(w - data.current_weights))
            if turn > params.max_turnover:
                w = data.current_weights + (params.max_turnover/turn)*(w-data.current_weights)
        t = np.sum(np.abs(w))
        if t > 0: w = w/t
        else: w = np.ones(n)/n
        mu = ret.mean(axis=0)
        er = float(w@mu*params.ann_factor)
        ev = float(np.sqrt(w@cov_est.matrix@w*params.ann_factor))
        sr = er/ev if ev > 0 else 0.0
        sp = np.sqrt(w@cov_est.matrix@w) if w@cov_est.matrix@w > 0 else 1.0
        mrc = (cov_est.matrix@w)/sp
        avols = np.sqrt(np.diag(cov_est.matrix))
        dr = float(np.abs(w)@avols/sp) if sp > 0 else 1.0
        to = float(np.sum(np.abs(w-data.current_weights))) if data.current_weights is not None else 0.0
        return AllocationOutput(status=EngineStatus.SUCCESS,
            result=AllocationResult(w, params.method, er, ev, sr, mrc, dr),
            covariance=cov_est, turnover=to, diagnostics={"n_assets":n})

    def allocate(self, returns, params=None):
        """Simplified interface for use as sub-component."""
        params = params or self.default_params()
        out = self.compute(AllocationInput(returns=returns), params)
        return out.result.weights if out.result else np.ones(returns.shape[1])/returns.shape[1]

    def _est_cov(self, ret, params):
        n = ret.shape[1]
        if params.cov_method == "shrinkage":
            s = np.cov(ret, rowvar=False); t = np.diag(np.diag(s))
            cov = (1-params.shrinkage_intensity)*s + params.shrinkage_intensity*t
        elif params.cov_method == "ewma":
            lam = 1 - np.log(2)/params.ewma_halflife
            wts = np.array([lam**(ret.shape[0]-1-i) for i in range(ret.shape[0])]); wts /= wts.sum()
            c = ret - ret.mean(axis=0); cov = (c*wts[:,None]).T @ c
        else: cov = np.cov(ret, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        if np.any(eig < 0): cov += np.eye(n)*(abs(eig.min())+1e-8); eig = np.linalg.eigvalsh(cov)
        return CovarianceEstimate(cov, params.cov_method, float(eig.max()/max(eig.min(),1e-10)),
            int(np.sum(eig > eig.max()*1e-6)))

    def _optimize(self, ret, cov, params):
        n = cov.shape[0]; m = params.method
        if m == "equal_weight": return np.ones(n)/n
        elif m == "inverse_vol":
            v = np.sqrt(np.diag(cov)); v = np.maximum(v,1e-10); w = 1/v; return w/w.sum()
        elif m == "min_variance":
            try: inv=np.linalg.inv(cov); o=np.ones(n); return inv@o/(o@inv@o)
            except: return np.ones(n)/n
        elif m == "max_sharpe":
            mu = params.expected_returns if params.expected_returns is not None else ret.mean(axis=0)
            try: w=np.linalg.inv(cov)@(mu-params.risk_free_rate); t=np.sum(np.abs(w)); return w/t if t>0 else np.ones(n)/n
            except: return np.ones(n)/n
        elif m == "risk_parity":
            budget = params.risk_budget if params.risk_budget is not None else np.ones(n)/n
            w = np.ones(n)/n
            for _ in range(100):
                sp = np.sqrt(w@cov@w)
                if sp < 1e-12: break
                rc = w*(cov@w)/sp; trc = budget*sp
                wn = w*trc/np.maximum(rc,1e-12); wn = wn/wn.sum()
                if np.max(np.abs(wn-w)) < 1e-8: w=wn; break
                w = wn
            return w
        elif m == "target_vol":
            try: inv=np.linalg.inv(cov); o=np.ones(n); bw=inv@o/(o@inv@o)
            except: bw=np.ones(n)/n
            pv=np.sqrt(bw@cov@bw*params.ann_factor)
            return bw*(params.vol_target/pv) if pv>0 else bw
        elif m == "black_litterman":
            delta=2.5; wmkt=np.ones(n)/n; pi=delta*cov@wmkt
            if params.views_P is not None and params.views_Q is not None:
                P,Q=params.views_P,params.views_Q
                omega=params.views_omega if params.views_omega is not None else np.diag(np.diag(P@(params.tau*cov)@P.T))
                ti=np.linalg.inv(params.tau*cov); oi=np.linalg.inv(omega)
                pc=np.linalg.inv(ti+P.T@oi@P); pm=pc@(ti@pi+P.T@oi@Q)
            else: pm=pi
            try: w=np.linalg.inv(cov)@pm; t=np.sum(np.abs(w)); return w/t if t>0 else np.ones(n)/n
            except: return np.ones(n)/n
        elif m == "hrp":
            if n <= 1: return np.ones(n)
            corr=np.corrcoef(ret, rowvar=False); corr=np.clip(corr,-1,1); np.fill_diagonal(corr,1)
            dist=np.sqrt(0.5*(1-corr)); np.fill_diagonal(dist,0)
            link_=linkage(squareform(dist, checks=False), method="single")
            iv=1.0/np.maximum(np.diag(cov),1e-10); return iv/iv.sum()
        elif m == "max_diversification":
            v=np.sqrt(np.diag(cov))
            try: w=np.linalg.inv(cov)@v; return w/max(np.sum(np.abs(w)),1e-10)
            except: return np.ones(n)/n
        return np.ones(n)/n
