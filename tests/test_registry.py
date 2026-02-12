"""
Tests for Function Registry + Signals + Sizing (Phase 1.5).

Covers:
- FunctionRegistry singleton
- Registration (string path, class, instance, config dict)
- Resolution (lazy import, caching)
- Error cases (missing function, bad module)
- Introspection (list, describe)
- SMACrossover signal generation
- EMACrossover signal generation
- FixedFraction sizing
- Integration: registry → signal → sizing pipeline
- Built-in defaults registration
"""

from typing import Any

import numpy as np
import polars as pl
import pytest

from praxis.registry import FunctionRegistry
from praxis.registry.defaults import register_defaults, BUILTIN_REGISTRY
from praxis.signals import Signal, SMACrossover, EMACrossover
from praxis.sizing import Sizer, FixedFraction
from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_singletons():
    FunctionRegistry.reset()
    PraxisLogger.reset()
    yield
    FunctionRegistry.reset()
    PraxisLogger.reset()


@pytest.fixture
def registry():
    reg = FunctionRegistry.instance()
    register_defaults(reg)
    return reg


@pytest.fixture
def price_data():
    """Synthetic price data with a clear trend: up then down."""
    np.random.seed(42)
    n = 200
    # Create a price series: uptrend (0-100), downtrend (100-200)
    trend_up = np.cumsum(np.random.randn(100) * 0.5 + 0.3) + 100
    trend_down = np.cumsum(np.random.randn(100) * 0.5 - 0.3) + trend_up[-1]
    closes = np.concatenate([trend_up, trend_down])
    return pl.DataFrame({"close": closes})


@pytest.fixture
def flat_data():
    """Flat/noisy price data for edge cases."""
    np.random.seed(99)
    closes = 100 + np.random.randn(100) * 0.01  # Nearly flat
    return pl.DataFrame({"close": closes})


# ═══════════════════════════════════════════════════════════════════
#  FunctionRegistry Singleton
# ═══════════════════════════════════════════════════════════════════

class TestRegistrySingleton:
    def test_singleton_identity(self):
        a = FunctionRegistry.instance()
        b = FunctionRegistry.instance()
        assert a is b

    def test_reset_creates_new(self):
        a = FunctionRegistry.instance()
        FunctionRegistry.reset()
        b = FunctionRegistry.instance()
        assert a is not b


# ═══════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_register_string_path(self):
        reg = FunctionRegistry.instance()
        reg.register("signals", "sma_crossover", "praxis.signals.trend", "SMACrossover")
        assert reg.has("signals", "sma_crossover")

    def test_register_class(self):
        reg = FunctionRegistry.instance()
        reg.register_class("signals", "sma_test", SMACrossover)
        result = reg.resolve("signals", "sma_test")
        assert isinstance(result, SMACrossover)

    def test_register_instance(self):
        reg = FunctionRegistry.instance()
        sma = SMACrossover()
        reg.register_instance("signals", "sma_direct", sma)
        result = reg.resolve("signals", "sma_direct")
        assert result is sma

    def test_register_from_config(self):
        reg = FunctionRegistry.instance()
        count = reg.register_from_config(BUILTIN_REGISTRY)
        assert count == 3  # sma_crossover, ema_crossover, fixed_fraction

    def test_re_register_clears_cache(self):
        reg = FunctionRegistry.instance()
        reg.register_class("signals", "test", SMACrossover)
        first = reg.resolve("signals", "test")
        reg.register_class("signals", "test", EMACrossover)
        second = reg.resolve("signals", "test")
        assert isinstance(first, SMACrossover)
        assert isinstance(second, EMACrossover)


# ═══════════════════════════════════════════════════════════════════
#  Resolution
# ═══════════════════════════════════════════════════════════════════

class TestResolution:
    def test_resolve_lazy_import(self, registry):
        signal = registry.resolve("signals", "sma_crossover")
        assert isinstance(signal, SMACrossover)

    def test_resolve_caches_instance(self, registry):
        a = registry.resolve("signals", "sma_crossover")
        b = registry.resolve("signals", "sma_crossover")
        assert a is b

    def test_resolve_different_categories(self, registry):
        signal = registry.resolve("signals", "sma_crossover")
        sizer = registry.resolve("sizing", "fixed_fraction")
        assert isinstance(signal, Signal)
        assert isinstance(sizer, Sizer)

    def test_resolve_missing_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="not found.*signals"):
            registry.resolve("signals", "nonexistent")

    def test_resolve_missing_shows_available(self, registry):
        with pytest.raises(KeyError, match="sma_crossover"):
            registry.resolve("signals", "nonexistent")

    def test_resolve_empty_category(self):
        reg = FunctionRegistry.instance()
        with pytest.raises(KeyError, match="none"):
            reg.resolve("empty_category", "something")


# ═══════════════════════════════════════════════════════════════════
#  Introspection
# ═══════════════════════════════════════════════════════════════════

class TestIntrospection:
    def test_list_category(self, registry):
        signals = registry.list_category("signals")
        assert "sma_crossover" in signals
        assert "ema_crossover" in signals

    def test_list_categories(self, registry):
        cats = registry.list_categories()
        assert "signals" in cats
        assert "sizing" in cats

    def test_describe(self, registry):
        desc = registry.describe()
        assert "signals" in desc
        assert "sma_crossover" in desc["signals"]
        assert "sizing" in desc
        assert "fixed_fraction" in desc["sizing"]

    def test_count(self, registry):
        assert registry.count == 3

    def test_has(self, registry):
        assert registry.has("signals", "sma_crossover")
        assert not registry.has("signals", "nonexistent")


# ═══════════════════════════════════════════════════════════════════
#  SMACrossover Signal
# ═══════════════════════════════════════════════════════════════════

class TestSMACrossover:
    def test_generate_basic(self, price_data):
        sma = SMACrossover()
        signals = sma.generate(price_data, {"fast_period": 10, "slow_period": 50})
        assert len(signals) == len(price_data)
        assert signals.dtype == pl.Int32 or signals.dtype == pl.Int64 or signals.dtype == pl.Int8

    def test_signal_values_in_range(self, price_data):
        sma = SMACrossover()
        signals = sma.generate(price_data, {"fast_period": 10, "slow_period": 50})
        unique = set(signals.to_list())
        assert unique.issubset({-1, 0, 1, None})

    def test_uptrend_produces_longs(self):
        """Clear uptrend should eventually produce long signals."""
        prices = pl.DataFrame({"close": list(range(1, 101))})
        sma = SMACrossover()
        signals = sma.generate(prices, {"fast_period": 5, "slow_period": 20})
        # After warmup, uptrend should be mostly long
        late_signals = signals[50:]
        assert (late_signals == 1).sum() > (late_signals == -1).sum()

    def test_downtrend_produces_shorts(self):
        """Clear downtrend should eventually produce short signals."""
        prices = pl.DataFrame({"close": list(range(100, 0, -1))})
        sma = SMACrossover()
        signals = sma.generate(prices, {"fast_period": 5, "slow_period": 20})
        late_signals = signals[50:]
        assert (late_signals == -1).sum() > (late_signals == 1).sum()

    def test_warmup_period_is_null_or_zero(self, price_data):
        """First slow_period bars should be 0 or null (insufficient data)."""
        sma = SMACrossover()
        signals = sma.generate(price_data, {"fast_period": 10, "slow_period": 50})
        # First 49 bars don't have enough data for slow SMA
        early = signals[:49]
        assert (early == 0).sum() + early.null_count() == 49

    def test_custom_price_column(self):
        prices = pl.DataFrame({"adj_close": list(range(1, 101))})
        sma = SMACrossover()
        signals = sma.generate(
            prices, {"fast_period": 5, "slow_period": 20, "price_column": "adj_close"}
        )
        assert len(signals) == 100

    def test_default_params(self, price_data):
        """Default fast=10, slow=50."""
        sma = SMACrossover()
        signals = sma.generate(price_data, {})
        assert len(signals) == len(price_data)

    def test_name_property(self):
        assert SMACrossover().name == "SMACrossover"


# ═══════════════════════════════════════════════════════════════════
#  EMACrossover Signal
# ═══════════════════════════════════════════════════════════════════

class TestEMACrossover:
    def test_generate_basic(self, price_data):
        ema = EMACrossover()
        signals = ema.generate(price_data, {"fast_period": 12, "slow_period": 26})
        assert len(signals) == len(price_data)

    def test_ema_responds_faster_than_sma(self):
        """EMA should detect trend changes earlier than SMA."""
        # Pure uptrend after initial flat - EMA should go long first
        flat = [50.0] * 30
        ramp = [50.0 + i * 1.0 for i in range(1, 71)]  # Steady climb
        prices = pl.DataFrame({"close": flat + ramp})

        sma = SMACrossover()
        ema = EMACrossover()
        params = {"fast_period": 5, "slow_period": 20}

        sma_signals = sma.generate(prices, params)
        ema_signals = ema.generate(prices, params)

        # Find first long signal after the ramp starts (bar 30)
        def first_long_after(signals, start):
            for i in range(start, len(signals)):
                if signals[i] == 1:
                    return i
            return len(signals)

        sma_first = first_long_after(sma_signals, 30)
        ema_first = first_long_after(ema_signals, 30)

        # EMA should go long no later than SMA
        assert ema_first <= sma_first

    def test_default_params(self, price_data):
        """Default fast=12, slow=26."""
        ema = EMACrossover()
        signals = ema.generate(price_data, {})
        assert len(signals) == len(price_data)

    def test_name_property(self):
        assert EMACrossover().name == "EMACrossover"


# ═══════════════════════════════════════════════════════════════════
#  FixedFraction Sizing
# ═══════════════════════════════════════════════════════════════════

class TestFixedFraction:
    def test_full_fraction(self):
        signals = pl.Series([1, -1, 0, 1, -1])
        sizer = FixedFraction()
        positions = sizer.size(signals, {"fraction": 1.0})
        assert positions.to_list() == [1.0, -1.0, 0.0, 1.0, -1.0]

    def test_half_fraction(self):
        signals = pl.Series([1, -1, 0, 1])
        sizer = FixedFraction()
        positions = sizer.size(signals, {"fraction": 0.5})
        assert positions.to_list() == [0.5, -0.5, 0.0, 0.5]

    def test_max_position_clamp(self):
        signals = pl.Series([1, -1, 0])
        sizer = FixedFraction()
        positions = sizer.size(signals, {"fraction": 1.0, "max_position_pct": 0.25})
        assert positions.to_list() == [0.25, -0.25, 0.0]

    def test_zero_fraction(self):
        signals = pl.Series([1, -1, 1])
        sizer = FixedFraction()
        positions = sizer.size(signals, {"fraction": 0.0})
        assert positions.to_list() == [0.0, 0.0, 0.0]

    def test_default_fraction(self):
        signals = pl.Series([1, -1])
        sizer = FixedFraction()
        positions = sizer.size(signals, {})
        assert positions.to_list() == [1.0, -1.0]

    def test_name_property(self):
        assert FixedFraction().name == "FixedFraction"


# ═══════════════════════════════════════════════════════════════════
#  Integration: Registry → Signal → Sizing Pipeline
# ═══════════════════════════════════════════════════════════════════

class TestPipeline:
    def test_registry_to_signal_to_sizing(self, registry, price_data):
        """End-to-end: resolve signal + sizer from registry, run pipeline."""
        # Resolve from registry
        signal_fn = registry.resolve("signals", "sma_crossover")
        sizer_fn = registry.resolve("sizing", "fixed_fraction")

        # Generate signals
        signals = signal_fn.generate(
            price_data, {"fast_period": 10, "slow_period": 50}
        )

        # Size positions
        positions = sizer_fn.size(signals, {"fraction": 0.5, "max_position_pct": 0.5})

        # Validate pipeline output
        assert len(positions) == len(price_data)
        assert positions.max() <= 0.5
        assert positions.min() >= -0.5

    def test_ema_pipeline(self, registry, price_data):
        signal_fn = registry.resolve("signals", "ema_crossover")
        sizer_fn = registry.resolve("sizing", "fixed_fraction")

        signals = signal_fn.generate(
            price_data, {"fast_period": 12, "slow_period": 26}
        )
        positions = sizer_fn.size(signals, {"fraction": 1.0})

        assert len(positions) == len(price_data)
        unique_pos = set(positions.to_list())
        assert unique_pos.issubset({-1.0, 0.0, 1.0, None})


# ═══════════════════════════════════════════════════════════════════
#  Built-in Defaults
# ═══════════════════════════════════════════════════════════════════

class TestDefaults:
    def test_register_defaults(self):
        reg = FunctionRegistry.instance()
        count = register_defaults(reg)
        assert count == 3
        assert reg.has("signals", "sma_crossover")
        assert reg.has("signals", "ema_crossover")
        assert reg.has("sizing", "fixed_fraction")

    def test_register_defaults_idempotent(self):
        reg = FunctionRegistry.instance()
        register_defaults(reg)
        register_defaults(reg)
        assert reg.count == 3  # No duplicates
