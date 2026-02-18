"""Business context: everything circumstantial about how/where/when to run."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class AssetClass(Enum):
    EQUITY = "equity"; FUTURES = "futures"; OPTIONS = "options"
    FX = "fx"; CRYPTO = "crypto"; FIXED_INCOME = "fixed_income"; COMMODITY = "commodity"

class Frequency(Enum):
    TICK = "tick"; MINUTE_1 = "1m"; MINUTE_5 = "5m"; HOUR_1 = "1h"
    DAILY = "1d"; WEEKLY = "1w"; MONTHLY = "1M"

class RebalanceFrequency(Enum):
    CONTINUOUS = "continuous"; DAILY = "daily"; WEEKLY = "weekly"; MONTHLY = "monthly"

class ExecutionMode(Enum):
    BACKTEST = "backtest"; PAPER = "paper"; LIVE = "live"

class Calendar(Enum):
    NYSE = "NYSE"; LSE = "LSE"; CME = "CME"; CRYPTO_247 = "24/7"

@dataclass
class UniverseSpec:
    tickers: list[str] = field(default_factory=list)
    asset_class: AssetClass = AssetClass.EQUITY
    sector_filters: list[str] = field(default_factory=list)
    market_cap_range: tuple[float, float] | None = None
    exclusions: list[str] = field(default_factory=list)
    name: str = ""
    @property
    def n_assets(self) -> int: return len(self.tickers)

@dataclass
class TemporalSpec:
    start: str | None = None; end: str | None = None
    frequency: Frequency = Frequency.DAILY
    lookback_days: int = 504
    calendar: Calendar = Calendar.NYSE
    rebalance: RebalanceFrequency = RebalanceFrequency.DAILY

@dataclass
class RiskSpec:
    max_position_size: float = 1.0; max_sector_exposure: float = 1.0
    max_drawdown: float = 0.25; vol_target: float | None = None
    leverage_limit: float = 1.0; long_only: bool = False

@dataclass
class ExecutionSpec:
    mode: ExecutionMode = ExecutionMode.BACKTEST
    slippage_bps: float = 5.0; commission_bps: float = 5.0

@dataclass
class ModelContext:
    universe: UniverseSpec = field(default_factory=UniverseSpec)
    temporal: TemporalSpec = field(default_factory=TemporalSpec)
    risk: RiskSpec = field(default_factory=RiskSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
