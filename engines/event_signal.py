"""ENGINE 7: Event/Signal. Feeder into Engines 2 (Momentum) and 3 (Allocation)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from engines.base import (
    TimeSeriesEngine, EngineParams, EngineInput, EngineOutput, EngineStatus, EngineValidationError)

@dataclass(frozen=True)
class EventSignalParams(EngineParams):
    feature_method: Literal["rank","zscore","percentile","raw"] = "zscore"
    winsorize_pct: float = 0.025
    combine_method: Literal["equal","ic_weighted"] = "equal"
    feature_weights: dict[str,float] = field(default_factory=dict)
    decay_type: Literal["none","exponential","linear"] = "exponential"
    decay_halflife: int = 21
    positive_words: list[str] = field(default_factory=lambda:["beat","exceeded","strong","growth","upgrade","outperform","positive","bullish","surge","record"])
    negative_words: list[str] = field(default_factory=lambda:["miss","disappointing","weak","decline","downgrade","underperform","negative","bearish","plunge","loss"])

@dataclass
class EventSignalInput(EngineInput):
    numeric_features: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: list[str] = field(default_factory=list)
    asset_names: list[str] = field(default_factory=list)
    text_data: dict[int,list[str]] = field(default_factory=dict)
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class AlphaScore:
    asset_idx:int; raw_scores:dict[str,float]; normalized_scores:dict[str,float]
    composite_alpha:float; decayed_alpha:float; rank:int=0; sentiment_score:float=0

@dataclass
class EventSignalOutput(EngineOutput):
    alphas: list[AlphaScore] = field(default_factory=list)
    alpha_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    rankings: np.ndarray = field(default_factory=lambda: np.array([]))

class EventSignalEngine(TimeSeriesEngine):
    def default_params(self): return EventSignalParams()
    def validate_input(self, data, params):
        if not isinstance(data, EventSignalInput): raise EngineValidationError("Need EventSignalInput")
        if data.numeric_features.size == 0 and not data.text_data: raise EngineValidationError("Need features or text")
    def build_input(self, prices): raise NotImplementedError("EventSignalEngine needs EventSignalInput")

    def compute(self, data: EventSignalInput, params: EventSignalParams) -> EventSignalOutput:
        self.validate_input(data, params)
        na = data.numeric_features.shape[0] if data.numeric_features.size > 0 else max(data.text_data.keys(), default=-1)+1
        if na == 0: return EventSignalOutput(status=EngineStatus.INSUFFICIENT_DATA)
        # Numeric
        nscores = {}
        if data.numeric_features.size > 0:
            names = data.feature_names or [f"f{i}" for i in range(data.numeric_features.shape[1])]
            for j, nm in enumerate(names):
                col = data.numeric_features[:,j].astype(float)
                v = col[~np.isnan(col)]
                if len(v) > 0:
                    lo, hi = np.percentile(v, params.winsorize_pct*100), np.percentile(v, (1-params.winsorize_pct)*100)
                    col = np.clip(col, lo, hi)
                if params.feature_method == "zscore":
                    mu, sig = np.nanmean(col), np.nanstd(col)
                    col = (col-mu)/sig if sig > 0 else np.zeros(na)
                elif params.feature_method == "rank":
                    col = np.argsort(np.argsort(col)).astype(float)/max(na-1,1)*2-1
                elif params.feature_method == "percentile":
                    col = np.argsort(np.argsort(col)).astype(float)/max(na-1,1)
                nscores[nm] = col
        # Text
        tscores = {}
        for idx, texts in data.text_data.items():
            total = 0
            for t in texts:
                words = t.lower().split()
                total += (sum(1 for w in words if w in params.positive_words) - sum(1 for w in words if w in params.negative_words))/max(len(words),1)
            tscores[idx] = total/max(len(texts),1)
        # Combine
        alphas = []
        for i in range(na):
            raw, norm_ = {}, {}
            for nm, arr in nscores.items():
                raw[nm] = float(arr[i]) if i < len(arr) else 0; norm_[nm] = raw[nm]
            if i in tscores: raw["sentiment"] = tscores[i]; norm_["sentiment"] = tscores[i]
            w = params.feature_weights or {k:1 for k in norm_}
            tw = sum(w.get(k,1) for k in norm_)
            comp = sum(v*w.get(k,1) for k,v in norm_.items())/tw if tw > 0 else 0
            dec = comp
            if params.decay_type != "none" and data.timestamps.size > 0 and i < len(data.timestamps):
                age = float(len(data.timestamps)-1-i)
                if params.decay_type == "exponential": dec = comp*np.exp(-np.log(2)*age/params.decay_halflife)
                elif params.decay_type == "linear": dec = comp*max(0,1-age/(2*params.decay_halflife))
            alphas.append(AlphaScore(i, raw, norm_, comp, dec, sentiment_score=tscores.get(i,0)))
        composites = np.array([a.decayed_alpha for a in alphas])
        rankings = np.argsort(np.argsort(-composites))
        for i, a in enumerate(alphas): a.rank = int(rankings[i])
        return EventSignalOutput(status=EngineStatus.SUCCESS, alphas=alphas, alpha_vector=composites, rankings=rankings)
