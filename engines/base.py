"""Engine base classes: ModelEngine ABC and three intermediate bases."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import numpy as np

class EngineStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"

@dataclass(frozen=True)
class EngineParams:
    def to_dict(self) -> dict: return asdict(self)

@dataclass
class EngineInput:
    def shape_summary(self) -> str: return "(abstract)"

@dataclass
class EngineOutput:
    status: EngineStatus = EngineStatus.SUCCESS
    diagnostics: dict[str, Any] = field(default_factory=dict)
    @property
    def ok(self) -> bool: return self.status in (EngineStatus.SUCCESS, EngineStatus.PARTIAL)

class EngineValidationError(ValueError): pass

class ModelEngine(ABC):
    """Pure computation. No I/O. No business logic. Stateless."""
    @abstractmethod
    def compute(self, data: EngineInput, params: EngineParams) -> EngineOutput: ...
    @abstractmethod
    def validate_input(self, data: EngineInput, params: EngineParams) -> None: ...
    @abstractmethod
    def default_params(self) -> EngineParams: ...
    def __repr__(self): return f"{self.__class__.__name__}()"

class TimeSeriesEngine(ModelEngine):
    """Engines operating on (n_obs, n_assets) price/return matrices."""
    def _validate_price_matrix(self, prices: np.ndarray, min_obs: int = 30) -> None:
        if not isinstance(prices, np.ndarray):
            raise EngineValidationError(f"Expected ndarray, got {type(prices).__name__}")
        if prices.ndim == 1: prices = prices.reshape(-1, 1)
        if prices.ndim != 2:
            raise EngineValidationError(f"Expected 2D, got {prices.ndim}D shape {prices.shape}")
        if prices.shape[0] < min_obs:
            raise EngineValidationError(f"Need at least {min_obs} observations, got {prices.shape[0]}")
        if np.isnan(prices).sum() / prices.size > 0.1:
            raise EngineValidationError(f"Too many NaN values: {np.isnan(prices).sum()/prices.size:.1%}")

class SurfaceEngine(ModelEngine):
    """Engines operating on 2D+ surfaces (strike x expiry, etc.)."""
    pass

class StateEngine(ModelEngine):
    """Engines operating on evolving state. Deferred (Engines 5 & 6)."""
    pass
