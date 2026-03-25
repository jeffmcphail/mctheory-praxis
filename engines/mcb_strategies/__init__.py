"""
engines/mcb_strategies/__init__.py
====================================
Registry of all MCb backtest strategies.
Sits alongside cpo_core.py, crypto_ta_strategy.py etc. in engines/.
"""

from .base import MCBStrategy
from .anchor_trigger import AnchorTriggerStrategy
from .zero_line_rejection import ZeroLineRejectionStrategy
from .bullish_divergence import BullishDivergenceStrategy
from .mfi_momentum import MFIMomentumStrategy

_REGISTRY: dict[str, type[MCBStrategy]] = {
    cls.id: cls
    for cls in [
        AnchorTriggerStrategy,
        ZeroLineRejectionStrategy,
        BullishDivergenceStrategy,
        MFIMomentumStrategy,
    ]
}


def list_strategies() -> list[dict]:
    """Return metadata for all strategies (for the kickoff form)."""
    return [cls.to_info() for cls in _REGISTRY.values()]


def get_strategy(strategy_id: str, params: dict | None = None) -> MCBStrategy:
    """Instantiate a strategy by ID with optional param overrides."""
    cls = _REGISTRY.get(strategy_id)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{strategy_id}'. Available: {list(_REGISTRY)}"
        )
    return cls(params or {})
