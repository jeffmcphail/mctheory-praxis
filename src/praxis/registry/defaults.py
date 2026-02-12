"""
Built-in function registry configuration.

Registers all Phase 1 functions. Import this module to populate
the registry with defaults.

Usage:
    from praxis.registry.defaults import register_defaults
    register_defaults()
"""

from praxis.registry import FunctionRegistry


# §7.2: Registry configuration matching spec
BUILTIN_REGISTRY = {
    "signals": {
        "sma_crossover": {"module": "praxis.signals.trend", "class": "SMACrossover"},
        "ema_crossover": {"module": "praxis.signals.trend", "class": "EMACrossover"},
    },
    "sizing": {
        "fixed_fraction": {"module": "praxis.sizing.fixed", "class": "FixedFraction"},
    },
}


def register_defaults(registry: FunctionRegistry | None = None) -> int:
    """
    Register all built-in functions.

    Args:
        registry: Registry to populate. Defaults to singleton.

    Returns count of functions registered.
    """
    if registry is None:
        registry = FunctionRegistry.instance()
    return registry.register_from_config(BUILTIN_REGISTRY)
