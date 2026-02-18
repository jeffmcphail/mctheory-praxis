"""I/O adapters: DataProvider/ResultStore protocols + in-memory for testing."""
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np
from engines.context.model_context import UniverseSpec, TemporalSpec

@runtime_checkable
class DataProvider(Protocol):
    def fetch_prices(self, universe: UniverseSpec, temporal: TemporalSpec) -> np.ndarray: ...
    def asset_names(self) -> list[str]: ...

@runtime_checkable
class ResultStore(Protocol):
    def save(self, key: str, result: Any) -> None: ...
    def load(self, key: str) -> Any: ...
    def exists(self, key: str) -> bool: ...

class InMemoryDataProvider:
    def __init__(self, prices: np.ndarray, asset_names_list: list[str] | None = None):
        self._prices = prices
        self._names = asset_names_list or [f"ASSET_{i}" for i in range(prices.shape[1])]
    def fetch_prices(self, universe, temporal) -> np.ndarray: return self._prices.copy()
    def asset_names(self) -> list[str]: return self._names.copy()

class InMemoryResultStore:
    def __init__(self): self._store: dict[str, Any] = {}
    def save(self, key, result): self._store[key] = result
    def load(self, key): return self._store[key]
    def exists(self, key) -> bool: return key in self._store
