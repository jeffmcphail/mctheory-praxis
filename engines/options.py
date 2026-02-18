"""ENGINE 4: Volatility/Options. Genuinely distinct — operates on vol surfaces, not price series."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from engines.base import (
    SurfaceEngine, EngineParams, EngineInput, EngineOutput, EngineStatus, EngineValidationError)

@dataclass(frozen=True)
class OptionsParams(EngineParams):
    iv_bounds: tuple[float,float] = (0.001, 5.0); iv_max_iter: int = 100; iv_tol: float = 1e-8
    greek_method: Literal["analytic","numerical"] = "analytic"; bump_size: float = 0.01
    vol_lookback: int = 252; iv_rv_threshold: float = 0.05; skew_zscore_threshold: float = 2.0

@dataclass
class OptionsInput(EngineInput):
    underlying_price: float = 0.0; risk_free_rate: float = 0.05; dividend_yield: float = 0.0
    strikes: np.ndarray = field(default_factory=lambda: np.array([]))
    expiries: np.ndarray = field(default_factory=lambda: np.array([]))
    market_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    option_type: Literal["call","put"] = "call"
    underlying_history: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class GreeksVector:
    delta:float=0; gamma:float=0; theta:float=0; vega:float=0; rho:float=0; vanna:float=0; volga:float=0

@dataclass
class VolSurface:
    strikes: np.ndarray; expiries: np.ndarray; iv_matrix: np.ndarray
    moneyness: np.ndarray; atm_vol: np.ndarray; skew: np.ndarray; term_structure: np.ndarray

@dataclass
class VolSignal:
    signal_type: str; direction: str; magnitude: float; iv_level: float; rv_level: float; spread: float

@dataclass
class OptionsOutput(EngineOutput):
    surface: VolSurface|None=None; greeks_matrix: list=field(default_factory=list)
    portfolio_greeks: GreeksVector=field(default_factory=GreeksVector)
    signals: list[VolSignal]=field(default_factory=list); realized_vol:float=0; iv_rv_spread:float=0

class OptionsEngine(SurfaceEngine):
    def default_params(self): return OptionsParams()
    def validate_input(self, data, params):
        if not isinstance(data, OptionsInput): raise EngineValidationError(f"Need OptionsInput")
        if data.underlying_price <= 0: raise EngineValidationError("Price must be positive")
        if len(data.strikes)==0 or len(data.expiries)==0: raise EngineValidationError("Need strikes & expiries")
        if data.market_prices.size > 0 and data.market_prices.shape != (len(data.strikes),len(data.expiries)):
            raise EngineValidationError("Market prices shape mismatch")
    def build_input(self, prices): raise NotImplementedError("OptionsEngine needs OptionsInput")

    def compute(self, data: OptionsInput, params: OptionsParams) -> OptionsOutput:
        self.validate_input(data, params)
        S,r,q = data.underlying_price, data.risk_free_rate, data.dividend_yield
        iv = self._extract_iv(data, params)
        mon = np.log(data.strikes/S)
        atm = np.array([float(np.interp(0, mon, iv[:,j])) if not np.all(np.isnan(iv[:,j])) else np.nan for j in range(len(data.expiries))])
        skew = np.zeros(len(data.expiries))
        for j in range(len(data.expiries)):
            c = iv[:,j]; v = ~np.isnan(c)
            if v.sum() >= 3:
                skew[j] = np.interp(-0.1, mon[v], c[v]) - np.interp(0.1, mon[v], c[v])
        surface = VolSurface(data.strikes, data.expiries, iv, mon, atm, skew, atm)
        gm = [[self.bs_greeks(S,K,T,r,iv[i,j],q,data.option_type) if not np.isnan(iv[i,j]) and iv[i,j]>0 else GreeksVector()
               for j,T in enumerate(data.expiries)] for i,K in enumerate(data.strikes)]
        rv = self._realized_vol(data.underlying_history, params.vol_lookback) if len(data.underlying_history)>20 else 0
        atm_iv = float(np.interp(S, data.strikes, iv[:,0])) if iv.size > 0 else 0
        signals = self._signals(surface, rv, params)
        return OptionsOutput(status=EngineStatus.SUCCESS, surface=surface, greeks_matrix=gm,
            signals=signals, realized_vol=rv, iv_rv_spread=atm_iv-rv,
            diagnostics={"n_strikes":len(data.strikes),"atm_iv":atm_iv,"realized_vol":rv})

    @staticmethod
    def bs_price(S,K,T,r,sigma,q=0,opt="call"):
        if T<=0 or sigma<=0: return max(S*np.exp(-q*T)-K*np.exp(-r*T),0) if opt=="call" else max(K*np.exp(-r*T)-S*np.exp(-q*T),0)
        d1=(np.log(S/K)+(r-q+.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
        if opt=="call": return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
        return K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)

    @staticmethod
    def bs_greeks(S,K,T,r,sigma,q=0,opt="call"):
        if T<=0 or sigma<=0: return GreeksVector()
        d1=(np.log(S/K)+(r-q+.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
        sT=np.sqrt(T); pdf1=norm.pdf(d1)
        if opt=="call":
            delta=np.exp(-q*T)*norm.cdf(d1)
            theta=-(S*sigma*np.exp(-q*T)*pdf1)/(2*sT)-r*K*np.exp(-r*T)*norm.cdf(d2)+q*S*np.exp(-q*T)*norm.cdf(d1)
            rho_=K*T*np.exp(-r*T)*norm.cdf(d2)
        else:
            delta=np.exp(-q*T)*(norm.cdf(d1)-1)
            theta=-(S*sigma*np.exp(-q*T)*pdf1)/(2*sT)+r*K*np.exp(-r*T)*norm.cdf(-d2)-q*S*np.exp(-q*T)*norm.cdf(-d1)
            rho_=-K*T*np.exp(-r*T)*norm.cdf(-d2)
        gamma=np.exp(-q*T)*pdf1/(S*sigma*sT)
        vega_=S*np.exp(-q*T)*sT*pdf1
        vanna=-np.exp(-q*T)*pdf1*d2/sigma
        volga=vega_*d1*d2/sigma
        return GreeksVector(float(delta),float(gamma),float(theta/365),float(vega_/100),float(rho_/100),float(vanna),float(volga))

    def _extract_iv(self, data, params):
        nk,nt = len(data.strikes), len(data.expiries)
        iv = np.full((nk,nt), np.nan)
        if data.market_prices.size == 0: return iv
        for i,K in enumerate(data.strikes):
            for j,T in enumerate(data.expiries):
                mkt = data.market_prices[i,j]
                if mkt > 0 and T > 0:
                    try: iv[i,j] = brentq(lambda s: self.bs_price(data.underlying_price,K,T,data.risk_free_rate,s,data.dividend_yield,data.option_type)-mkt,
                        params.iv_bounds[0], params.iv_bounds[1], maxiter=params.iv_max_iter, xtol=params.iv_tol)
                    except: pass
        return iv

    @staticmethod
    def _realized_vol(hist, lb):
        n = min(lb, len(hist)); r = np.diff(np.log(hist[-n:]))
        return float(np.std(r)*np.sqrt(252))

    def _signals(self, surf, rv, params):
        sigs = []
        va = surf.atm_vol[~np.isnan(surf.atm_vol)]
        if len(va) > 0 and rv > 0:
            sp = va[0]-rv
            if abs(sp) > params.iv_rv_threshold:
                sigs.append(VolSignal("iv_vs_rv","short_vol" if sp>0 else "long_vol",abs(sp),va[0],rv,sp))
        vs = surf.skew[~np.isnan(surf.skew)]
        if len(vs) > 1:
            zm = vs.mean(); zs = vs.std() if len(vs)>1 else 0.01
            z = (vs[-1]-zm)/zs if zs > 0 else 0
            if abs(z) > params.skew_zscore_threshold:
                sigs.append(VolSignal("skew","long_skew" if z>0 else "short_skew",abs(z),va[0] if len(va)>0 else 0,rv,float(vs[-1])))
        if len(va)>=2:
            sl=va[-1]-va[0]
            if abs(sl)>params.iv_rv_threshold:
                sigs.append(VolSignal("term_structure","short_vol" if sl>0 else "long_vol",abs(sl),va[0],rv,sl))
        return sigs
