"""
Pydantic configuration schemas for McTheory Praxis.

§6: Model Parameterization — mirrors DuckDB STRUCTs as validated Python models.
§18.6: Logger configuration.

Design principle: SMA-required fields are the only required fields.
Everything else is Optional with sensible defaults. A 15-line YAML
for SMA crossover must validate. A 200-line YAML for Burgess stat arb
must also validate.

Usage:
    config = ModelConfig.from_yaml("sma_crossover.yaml")
    config.validate()
    config.to_dict()  # → ready for DuckDB insert
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════

class ModelType(str, Enum):
    SINGLE_ASSET = "SingleAssetModel"
    PAIR = "PairModel"
    COMPOSITE = "CompositeModel"
    CPO = "CPOModel"
    ML = "MLModel"


class PlatformMode(str, Enum):
    FULL = "full"
    EPHEMERAL = "ephemeral"


class BacktestEngine(str, Enum):
    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"
    BOTH = "both"


# ═══════════════════════════════════════════════════════════════════
#  §6.2 Construction Params
# ═══════════════════════════════════════════════════════════════════

class UniverseFilters(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_adv: Optional[float] = None
    min_market_cap: Optional[float] = None
    exclude_types: Optional[list[str]] = None
    sectors: Optional[list[str]] = None
    exclude_sectors: Optional[list[str]] = None


class UniverseConfig(BaseModel):
    method: str = "static"  # 'static', 'index_constituents', 'filtered', 'custom'
    instruments: Optional[list[str]] = None
    index_name: Optional[str] = None
    filters: Optional[UniverseFilters] = None
    refresh_frequency: Optional[str] = None


class DataSourceConfig(BaseModel):
    provider: str = "yfinance"
    resolution: str = "daily"
    fields: Optional[list[str]] = None
    adjustments: Optional[str] = None
    lookback_days: Optional[int] = None
    api_params: Optional[dict[str, Any]] = None


class CointegrationConfig(BaseModel):
    test: Optional[str] = None
    p_value_threshold: Optional[float] = None
    lookback_days: Optional[int] = None
    min_half_life: Optional[int] = None
    max_half_life: Optional[int] = None


class CorrelationConfig(BaseModel):
    min_correlation: Optional[float] = None
    method: Optional[str] = None
    lookback_days: Optional[int] = None


class PairSelectionFilters(BaseModel):
    same_sector: Optional[bool] = None
    same_country: Optional[bool] = None
    min_adv_ratio: Optional[float] = None
    max_market_cap_ratio: Optional[float] = None


class PairSelectionConfig(BaseModel):
    method: Optional[str] = None
    cointegration: Optional[CointegrationConfig] = None
    correlation: Optional[CorrelationConfig] = None
    filters: Optional[PairSelectionFilters] = None
    max_pairs: Optional[int] = None
    rebalance_frequency: Optional[str] = None


class SpreadConfig(BaseModel):
    method: Optional[str] = None
    hedge_ratio: Optional[dict[str, Any]] = None
    normalization: Optional[str] = None
    mean_lookback: Optional[int] = None
    std_lookback: Optional[int] = None


class ConstructionParams(BaseModel):
    universe: Optional[UniverseConfig] = None
    data_source: Optional[DataSourceConfig] = None
    pair_selection: Optional[PairSelectionConfig] = None
    spread: Optional[SpreadConfig] = None


# ═══════════════════════════════════════════════════════════════════
#  §6.3 Signal Params
# ═══════════════════════════════════════════════════════════════════

class SignalParams(BaseModel):
    method: str  # Required: 'sma_crossover', 'ema_crossover', 'zscore', etc.

    # SMA/EMA specific
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    indicator: Optional[str] = None

    # Threshold-based
    threshold_long: Optional[float] = None
    threshold_short: Optional[float] = None
    threshold_exit: Optional[float] = None

    # General
    lookback: Optional[int] = None
    zscore_window: Optional[int] = None
    half_life_method: Optional[str] = None

    # ML/CPO
    model_ref: Optional[str] = None
    prediction_threshold: Optional[float] = None

    # Composite
    composite_method: Optional[str] = None
    component_signals: Optional[list[str]] = None
    component_weights: Optional[list[float]] = None

    # Meta-labeling
    meta_label_enabled: Optional[bool] = None
    primary_signal_ref: Optional[str] = None

    # Escape hatch
    custom_function: Optional[str] = None
    custom_params: Optional[dict[str, Any]] = None

    # Legacy/flexible params dict
    params: Optional[dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════
#  §6.4 Entry Params
# ═══════════════════════════════════════════════════════════════════

class RegimeCondition(BaseModel):
    condition: Optional[str] = None
    threshold_long: Optional[float] = None
    threshold_short: Optional[float] = None


class RegimeConditions(BaseModel):
    high_vol: Optional[RegimeCondition] = None
    low_vol: Optional[RegimeCondition] = None


class EntryParams(BaseModel):
    method: str = "threshold"
    long_threshold: Optional[float] = None
    short_threshold: Optional[float] = None
    confirmation_periods: Optional[int] = None
    confirmation_pct: Optional[float] = None
    entry_window_start: Optional[str] = None
    entry_window_end: Optional[str] = None
    entry_days: Optional[list[str]] = None
    regime_field: Optional[str] = None
    regime_conditions: Optional[RegimeConditions] = None
    entry_order_type: Optional[str] = None
    limit_offset_bps: Optional[float] = None
    custom_function: Optional[str] = None
    custom_params: Optional[dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════
#  §6.5 Exit Params
# ═══════════════════════════════════════════════════════════════════

class TripleBarrierConfig(BaseModel):
    pt_level: Optional[float] = None
    sl_level: Optional[float] = None
    max_days: Optional[int] = None
    vertical_first: Optional[bool] = None


class ExitParams(BaseModel):
    method: str = "threshold"

    # Profit target
    profit_target_pct: Optional[float] = None
    profit_target_atr_mult: Optional[float] = None

    # Stop loss
    stop_loss_pct: Optional[float] = None
    stop_loss_atr_mult: Optional[float] = None
    stop_loss_zscore: Optional[float] = None

    # Trailing
    trailing_enabled: Optional[bool] = None
    trailing_activation_pct: Optional[float] = None
    trailing_distance_pct: Optional[float] = None

    # Time-based
    max_holding_bars: Optional[int] = None
    max_holding_days: Optional[int] = None
    decay_exit_enabled: Optional[bool] = None

    # Mean reversion
    mean_reversion_level: Optional[float] = None
    mean_reversion_partial: Optional[bool] = None

    # Invalidation
    invalidation_enabled: Optional[bool] = None
    invalidation_test: Optional[str] = None
    invalidation_threshold: Optional[float] = None

    # AFML triple barrier
    triple_barrier: Optional[TripleBarrierConfig] = None

    custom_function: Optional[str] = None
    custom_params: Optional[dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════
#  §6.6 Sizing Params
# ═══════════════════════════════════════════════════════════════════

class RegimeSizingCondition(BaseModel):
    condition: Optional[str] = None
    method: Optional[str] = None
    target_vol: Optional[float] = None
    max_position_pct: Optional[float] = None


class RegimeSizing(BaseModel):
    high_vol: Optional[RegimeSizingCondition] = None
    low_vol: Optional[RegimeSizingCondition] = None


class SizingParams(BaseModel):
    method: str = "fixed_fraction"  # Required

    # Fixed fraction
    fraction: Optional[float] = None

    # Volatility targeting
    target_vol: Optional[float] = None
    vol_lookback: Optional[int] = None
    vol_method: Optional[str] = None
    vol_floor: Optional[float] = None
    vol_cap: Optional[float] = None

    # Kelly
    kelly_fraction: Optional[float] = None
    kelly_distribution: Optional[str] = None
    kelly_df: Optional[float] = None
    kelly_lookback: Optional[int] = None

    # Risk parity
    risk_contribution_target: Optional[str] = None
    custom_risk_weights: Optional[list[dict[str, Any]]] = None

    # Position limits
    max_position_pct: Optional[float] = None
    max_sector_pct: Optional[float] = None
    max_correlated_pct: Optional[float] = None
    max_drawdown_cut: Optional[float] = None

    # Regime
    regime_sizing_enabled: Optional[bool] = None
    regime_sizing: Optional[RegimeSizing] = None

    # Forecast scaling (Carver)
    forecast_scalar: Optional[float] = None
    forecast_cap: Optional[float] = None
    forecast_floor: Optional[float] = None

    custom_function: Optional[str] = None
    custom_params: Optional[dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════
#  §6.7 Single-Leg Params
# ═══════════════════════════════════════════════════════════════════

class SingleLegParams(BaseModel):
    enabled: Optional[bool] = None
    target_selection: Optional[dict[str, Any]] = None
    signal_basket: Optional[dict[str, Any]] = None
    lag_analysis: Optional[dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════
#  §6.8 CPO Params
# ═══════════════════════════════════════════════════════════════════

class CPOParams(BaseModel):
    enabled: bool = False
    search_method: Optional[str] = None
    parameter_grid: Optional[list[dict[str, Any]]] = None
    max_evaluations: Optional[int] = None
    features: Optional[dict[str, Any]] = None
    model: Optional[dict[str, Any]] = None
    prediction: Optional[dict[str, Any]] = None
    fallback: Optional[dict[str, Any]] = None
    baseline: Optional[dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════
#  §6.9 Backtest Params
# ═══════════════════════════════════════════════════════════════════

class CostsConfig(BaseModel):
    commission_per_share: Optional[float] = None
    commission_min: Optional[float] = None
    commission_pct: Optional[float] = None
    sec_fee_pct: Optional[float] = None
    exchange_fee_per_contract: Optional[float] = None


class SlippageConfig(BaseModel):
    method: str = "fixed"
    fixed_bps: Optional[float] = None
    volume_participation: Optional[float] = None
    market_impact_coef: Optional[float] = None


class FillsConfig(BaseModel):
    method: str = "immediate"
    partial_fills: Optional[bool] = None
    fill_ratio: Optional[float] = None


class BacktestDataConfig(BaseModel):
    survivorship_bias_free: Optional[bool] = None
    point_in_time: Optional[bool] = None
    corporate_actions: Optional[str] = None


class ValidationConfig(BaseModel):
    walk_forward: Optional[bool] = None
    n_splits: Optional[int] = None
    train_pct: Optional[float] = None
    embargo_days: Optional[int] = None


class BacktestParams(BaseModel):
    engine: str = "vectorized"
    reconciliation_tolerance: Optional[float] = None
    costs: Optional[CostsConfig] = None
    slippage: Optional[SlippageConfig] = None
    fills: Optional[FillsConfig] = None
    data: Optional[BacktestDataConfig] = None
    validation: Optional[ValidationConfig] = None


# ═══════════════════════════════════════════════════════════════════
#  §6.10 Risk Params
# ═══════════════════════════════════════════════════════════════════

class RiskParams(BaseModel):
    position_limits: Optional[dict[str, Any]] = None
    greeks: Optional[dict[str, Any]] = None
    var: Optional[dict[str, Any]] = None
    stress_tests: Optional[dict[str, Any]] = None
    drawdown: Optional[dict[str, Any]] = None
    correlation: Optional[dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════
#  Workflow Params
# ═══════════════════════════════════════════════════════════════════

class WorkflowStep(BaseModel):
    id: str
    function: str
    params: Optional[dict[str, Any]] = None
    depends_on: Optional[list[str]] = None
    condition: Optional[str] = None
    for_each: Optional[str] = None
    as_var: Optional[str] = None
    parallel: Optional[bool] = None


class WorkflowParams(BaseModel):
    enabled: bool = False
    steps: Optional[list[WorkflowStep]] = None


# ═══════════════════════════════════════════════════════════════════
#  §18.6 Logger Config
# ═══════════════════════════════════════════════════════════════════

class LogAdapterConfig(BaseModel):
    type: str
    min_level: int | str = 20
    color: Optional[bool] = None              # terminal
    path: Optional[str] = None                # file
    rotation: Optional[str] = None            # file
    retention_days: Optional[int] = None      # file
    buffer_size: Optional[int] = None         # database
    flush_interval_seconds: Optional[float] = None  # database
    backtest_throttle: Optional[int | str] = None   # database
    ring_buffer_size: Optional[int] = None    # agent
    formatter: Optional[str] = None           # any


class LogRoutingConfig(BaseModel):
    critical_override: bool = True
    tag_routes: Optional[dict[str, list[str]]] = None


class LoggerConfig(BaseModel):
    current_level: int | str = 20  # INFO
    adapters: Optional[dict[str, LogAdapterConfig]] = None
    tag_levels: Optional[dict[str, int | str]] = None
    routing: Optional[LogRoutingConfig] = None


# ═══════════════════════════════════════════════════════════════════
#  §1.5 Platform Config
# ═══════════════════════════════════════════════════════════════════

class PlatformConfig(BaseModel):
    mode: PlatformMode = PlatformMode.EPHEMERAL
    data_source: Optional[str] = None
    persistence: Optional[str] = None
    db_path: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
#  Top-Level Model Config
# ═══════════════════════════════════════════════════════════════════

class ModelConfig(BaseModel):
    """
    Top-level configuration for a Praxis model.

    This is what a user writes in YAML and what gets validated,
    hashed, and stored in fact_model_definition.

    Minimal SMA example (passes validation):
        model:
          name: sma_crossover
          type: SingleAssetModel
          version: v1.0
        signal:
          method: sma_crossover
          fast_period: 10
          slow_period: 50
        sizing:
          method: fixed_fraction
          fraction: 1.0
    """

    # ── Identity (required) ───────────────────────────────────────
    model: ModelIdentity

    # ── Parameter sections (§6.2-6.10) ────────────────────────────
    construction: Optional[ConstructionParams] = None
    signal: Optional[SignalParams] = None
    entry: Optional[EntryParams] = None
    exit: Optional[ExitParams] = None
    sizing: Optional[SizingParams] = None
    single_leg: Optional[SingleLegParams] = None
    cpo: Optional[CPOParams] = None
    backtest: Optional[BacktestParams] = None
    risk: Optional[RiskParams] = None
    workflow: Optional[WorkflowParams] = None

    # ── Platform & Logger ─────────────────────────────────────────
    platform: Optional[PlatformConfig] = None
    logger: Optional[LoggerConfig] = None

    # ── Metadata (auto-generated) ─────────────────────────────────
    source_yaml: Optional[str] = Field(None, exclude=True)

    @model_validator(mode="after")
    def validate_signal_required(self) -> "ModelConfig":
        """SMA needs at minimum: signal.method."""
        if self.model.type == ModelType.SINGLE_ASSET and self.signal is None:
            raise ValueError(
                f"SingleAssetModel requires 'signal' section"
            )
        return self

    @property
    def config_hash(self) -> str:
        """SHA256 of the config dict (excluding source_yaml and metadata)."""
        import json
        d = self.model_dump(exclude={"source_yaml"}, exclude_none=True)
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def bpk(self) -> str:
        """Business primary key: model_name|version."""
        version = self.model.version or "v0"
        return f"{self.model.name}|{version}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load and validate from a YAML file."""
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        config = cls.model_validate(data)
        config.source_yaml = raw
        return config

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "ModelConfig":
        """Load and validate from a YAML string."""
        data = yaml.safe_load(yaml_string)
        config = cls.model_validate(data)
        config.source_yaml = yaml_string
        return config

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Load and validate from a dict."""
        return cls.model_validate(data)

    def to_dict(self, exclude_none: bool = True) -> dict:
        """Export as dict, suitable for DuckDB insert or serialization."""
        return self.model_dump(exclude_none=exclude_none)


class ModelIdentity(BaseModel):
    """Model identity section — always required."""
    name: str
    type: ModelType = ModelType.SINGLE_ASSET
    version: Optional[str] = None
