"""
engines/xsec_reversal — Cross-sectional reversal in crypto, liquidity-tiered (Cycle 60).

Tests the pre-registered hypothesis that short-horizon cross-sectional reversal
in crypto is REAL BUT LIQUIDITY-TIERED: absent in the liquid majors (where the
2026 Fayez post-mortem found net Sharpe -3.22 on 10 perps), present in the
illiquid tail (where Kakushadze/Yu and the size-and-illiquidity factor
literature locate it), and bounded by capacity.

The load-bearing design decision is SURVIVORSHIP-BIAS-FREE universe
construction: Binance's REST API and the official `fetch-all-trading-pairs.sh`
helper return only CURRENTLY LISTED symbols. Building the universe that way
silently deletes every delisted coin — and in the illiquid tier, the delisted
coins are precisely the ones that went to zero. That would MANUFACTURE the
exact edge this experiment is trying to detect.

The fix: enumerate symbols from the data.binance.vision S3 bucket listing,
which retains archives for delisted pairs (Binance's own docs use ADABKRW — a
long-delisted pair — as their download example).

See claude/handoffs/CYCLE60_XSEC_REVERSAL_PREREG.md for the pre-registration.

Phase 1 (this cycle): data layer + universe + signal + cost + backtest engine,
verified on synthetic panels. NO real-data verdict until the collector has run.
"""

from .archive import (
    KlineSchema,
    ArchiveClient,
    parse_kline_csv,
    DEFAULT_KLINE_SCHEMA,
)
from .universe import (
    UniverseSpec,
    TierSpec,
    build_point_in_time_universe,
    assign_tiers,
)
from .costs import (
    CostSpec,
    corwin_schultz_spread,
    abdi_ranaldo_spread,
    estimate_spread_bps,
)
from .backtest import (
    SignalSpec,
    BacktestSpec,
    BacktestResult,
    compute_formation_returns,
    residualize,
    build_positions,
    run_backtest,
    capacity_analysis,
)

__all__ = [
    "KlineSchema", "ArchiveClient", "parse_kline_csv", "DEFAULT_KLINE_SCHEMA",
    "UniverseSpec", "TierSpec", "build_point_in_time_universe", "assign_tiers",
    "CostSpec", "corwin_schultz_spread", "abdi_ranaldo_spread", "estimate_spread_bps",
    "SignalSpec", "BacktestSpec", "BacktestResult",
    "compute_formation_returns", "residualize", "build_positions",
    "run_backtest", "capacity_analysis",
]
