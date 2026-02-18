"""ENGINE 2: Momentum/Trend-Following. Contrarian is same engine, sign flip."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from scipy import stats as sp_stats
from engines.base import (
    TimeSeriesEngine, EngineParams, EngineInput, EngineOutput, EngineStatus, EngineValidationError)

@dataclass(frozen=True)
class MomentumParams(EngineParams):
    return_type: Literal["simple","log"] = "simple"
    scoring_method: Literal["time_series","cross_sectional","dual"] = "time_series"
    lookback_periods: list[int] = field(default_factory=lambda: [63,126,252])
    skip_recent: int = 21
    ma_type: Literal["sma","ema","dema","hull"] = "ema"
    fast_period: int = 12; slow_period: int = 26
    breakout_method: Literal["donchian","bollinger","atr"] = "donchian"
    breakout_period: int = 20; breakout_mult: float = 2.0
    adx_period: int = 14; min_adx: float = 20.0
    signal_sign: float = 1.0  # 1=momentum, -1=contrarian
    combine_method: Literal["average","vote","strongest"] = "average"
    sizing_method: Literal["equal","inverse_vol","vol_target"] = "inverse_vol"
    vol_lookback: int = 63; vol_target: float = 0.10; ann_factor: float = 252.0
    top_n: int = 10; bottom_n: int = 10; long_only: bool = False; max_weight: float = 0.20

@dataclass
class MomentumInput(EngineInput):
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    asset_names: list[str] = field(default_factory=list)
    def shape_summary(self): return f"prices={self.prices.shape}"

@dataclass
class MomentumScore:
    asset_idx: int; scores_by_lookback: dict[int,float]; composite_score: float
    trend_strength: float; current_vol: float

@dataclass
class MomentumSignal:
    asset_idx: int; signal_value: float; position_weight: float
    ma_crossover: float; breakout_signal: float

@dataclass
class MomentumOutput(EngineOutput):
    scores: list[MomentumScore] = field(default_factory=list)
    signals: list[MomentumSignal] = field(default_factory=list)
    portfolio_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    rankings: np.ndarray = field(default_factory=lambda: np.array([]))

class MomentumEngine(TimeSeriesEngine):
    def default_params(self): return MomentumParams()
    def validate_input(self, data, params):
        p = data.prices if isinstance(data, MomentumInput) else data
        self._validate_price_matrix(p, min_obs=30)
    def build_input(self, prices):
        return MomentumInput(prices=prices, asset_names=[f"A{i}" for i in range(prices.shape[1])])

    def compute(self, data, params: MomentumParams) -> MomentumOutput:
        if isinstance(data, np.ndarray): data = self.build_input(data)
        self.validate_input(data, params)
        prices = data.prices; n_obs, n_assets = prices.shape
        rets = np.diff(np.log(prices), axis=0) if params.return_type == "log" else np.diff(prices, axis=0)/prices[:-1]
        scores = [self._score_asset(prices[:,i], rets[:,i], i, params) for i in range(n_assets)]
        composites = np.array([s.composite_score for s in scores])
        rankings = np.argsort(np.argsort(-composites))
        signals = [self._gen_signal(prices[:,i], rets[:,i], scores[i], i, params) for i in range(n_assets)]
        weights = self._portfolio(scores, signals, rankings, params)
        return MomentumOutput(status=EngineStatus.SUCCESS, scores=scores, signals=signals,
            portfolio_weights=weights, rankings=rankings, diagnostics={"n_assets":n_assets})

    def _score_asset(self, price, ret, idx, params):
        sbl = {}
        for lb in params.lookback_periods:
            if lb + params.skip_recent > len(ret): sbl[lb] = 0.0; continue
            end = len(ret) - params.skip_recent; start = max(0, end-lb)
            sbl[lb] = float(np.prod(1+ret[start:end])-1) * params.signal_sign
        comp = np.mean(list(sbl.values())) if sbl else 0.0
        # Trend strength (regression slope)
        p = min(params.adx_period, len(price))
        x = np.arange(p); recent = price[-p:]
        slope, *_ = sp_stats.linregress(x, recent) if p > 2 else (0,)
        ts = abs(slope/recent.mean())*100*np.sqrt(p) if recent.mean() > 0 else 0.0
        vol = float(np.std(ret[-params.vol_lookback:])*np.sqrt(params.ann_factor)) if len(ret) >= params.vol_lookback else 0.15
        return MomentumScore(idx, sbl, comp, ts, vol)

    def _gen_signal(self, price, ret, score, idx, params):
        fast = self._ma(price, params.fast_period, params.ma_type)
        slow = self._ma(price, params.slow_period, params.ma_type)
        mac = (fast[-1]-slow[-1])/slow[-1] if slow[-1] > 0 else 0.0
        brk = self._breakout(price, params)
        raw = (mac*100 + brk + score.composite_score*10)/3 * params.signal_sign
        if score.trend_strength < params.min_adx: raw *= 0.5
        if params.sizing_method == "vol_target" and score.current_vol > 0:
            w = params.vol_target / score.current_vol
        elif params.sizing_method == "inverse_vol" and score.current_vol > 0:
            w = 1.0 / score.current_vol
        else: w = 1.0
        w *= np.sign(raw) * min(abs(raw), 1.0)
        w = np.clip(w, -params.max_weight, params.max_weight)
        return MomentumSignal(idx, float(raw), float(w), float(mac), float(brk))

    @staticmethod
    def _ma(prices, period, ma_type):
        n = len(prices)
        if n < period: return prices.copy()
        if ma_type == "sma":
            m = np.convolve(prices, np.ones(period)/period, mode='full')[:n]; m[:period-1]=prices[:period-1]; return m
        elif ma_type == "ema":
            a = 2.0/(period+1); e = np.zeros(n); e[0]=prices[0]
            for i in range(1,n): e[i]=a*prices[i]+(1-a)*e[i-1]
            return e
        elif ma_type == "dema":
            a = 2.0/(period+1); e1=np.zeros(n); e2=np.zeros(n); e1[0]=e2[0]=prices[0]
            for i in range(1,n): e1[i]=a*prices[i]+(1-a)*e1[i-1]; e2[i]=a*e1[i]+(1-a)*e2[i-1]
            return 2*e1-e2
        elif ma_type == "hull":
            h = MomentumEngine._ma(prices, period//2, "sma"); f = MomentumEngine._ma(prices, period, "sma")
            return MomentumEngine._ma(2*h-f, max(int(np.sqrt(period)),1), "sma")
        return prices.copy()

    @staticmethod
    def _breakout(prices, params):
        p = params.breakout_period
        if len(prices) < p: return 0.0
        r = prices[-p:]; c = prices[-1]
        if params.breakout_method == "donchian":
            return 1.0 if c >= r.max() else -1.0 if c <= r.min() else 0.0
        elif params.breakout_method == "bollinger":
            mu, sig = r.mean(), r.std()
            u = mu+params.breakout_mult*sig; l = mu-params.breakout_mult*sig
            return 1.0 if c >= u else -1.0 if c <= l else (c-mu)/(params.breakout_mult*sig) if sig > 0 else 0.0
        return 0.0

    def _portfolio(self, scores, signals, rankings, params):
        n = len(scores); w = np.zeros(n)
        if params.scoring_method == "cross_sectional":
            for i in range(n):
                if rankings[i] < params.top_n: w[i] = max(signals[i].position_weight, 1.0/params.top_n)
                elif not params.long_only and rankings[i] >= n-params.bottom_n: w[i] = min(signals[i].position_weight, -1.0/params.bottom_n)
        else:
            for i in range(n): w[i] = signals[i].position_weight
        if params.long_only: w = np.maximum(w, 0)
        t = np.sum(np.abs(w))
        if t > 0: w = w/t
        w = np.clip(w, -params.max_weight, params.max_weight)
        t = np.sum(np.abs(w))
        if t > 0: w = w/t
        return w
