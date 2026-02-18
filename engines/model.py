"""Model orchestrator: thin glue composing Engine + Context + I/O."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from engines.base import ModelEngine, EngineParams, EngineOutput, EngineStatus
from engines.context.model_context import ModelContext
from engines.adapters.providers import DataProvider, ResultStore

@dataclass
class ModelResult:
    engine_output: EngineOutput
    context: ModelContext
    engine_name: str = ""
    run_metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def ok(self) -> bool: return self.engine_output.ok
    @property
    def status(self) -> EngineStatus: return self.engine_output.status

class Model:
    """Composes Engine + Context + I/O. Zero math, zero I/O — only orchestration."""
    def __init__(self, engine: ModelEngine, context: ModelContext,
                 data_provider: DataProvider, result_store: ResultStore | None = None,
                 params: EngineParams | None = None):
        self.engine = engine
        self.context = context
        self.data_provider = data_provider
        self.result_store = result_store
        self._params = params

    @property
    def params(self) -> EngineParams:
        return self._params if self._params is not None else self.engine.default_params()

    def run(self) -> ModelResult:
        prices = self.data_provider.fetch_prices(self.context.universe, self.context.temporal)
        engine_input = self.engine.build_input(prices) if hasattr(self.engine, 'build_input') else prices
        params = self.params
        self.engine.validate_input(engine_input, params)
        engine_output = self.engine.compute(engine_input, params)
        result = ModelResult(engine_output=engine_output, context=self.context,
                             engine_name=self.engine.__class__.__name__)
        if self.result_store is not None:
            self.result_store.save(f"{self.context.name}_{self.engine.__class__.__name__}", result)
        return result

    def __repr__(self):
        return f"Model(engine={self.engine}, strategy='{self.context.name}')"
