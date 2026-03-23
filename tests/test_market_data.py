"""
Tests for market_data layer: schema, fetchers, bridge, end-to-end with engines.

Tests verify:
- Schema creation and table registration
- MockMarketDataFetcher produces correct data shapes and types
- Lazy loading: data fetched on demand via has_row(fill_missing=True)
- DataStoreDataProvider produces correct numpy arrays for engines
- Universe resolution (explicit tickers, named universes)
- End-to-end: DataStore → Provider → Engine → Results
- Caching: second fetch hits cache, no re-fetch
"""

import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# ── Path setup so we can import both core.datastore and engines ──
CORE_SRC = Path(__file__).parent.parent.parent / "core_repo" / "src"
sys.path.insert(0, str(CORE_SRC))

from mctheory.core.datastore import DataStore, DataTable, DataView

from market_data.schema import create_market_datastore, SCHEMA_TABLES
from market_data.fetchers import MockMarketDataFetcher
from market_data.bridge import DataStoreDataProvider
from engines.context.model_context import (
    UniverseSpec, TemporalSpec, AssetClass, Frequency,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_datastore():
    """Reset DataStore singleton between tests."""
    DataStore.reset_instance()
    yield
    DataStore.reset_instance()


@pytest.fixture
def fetcher():
    return MockMarketDataFetcher.with_standard_profiles(n_days=504, seed=42)


@pytest.fixture
def ds(fetcher):
    return create_market_datastore(fetcher=fetcher)


@pytest.fixture
def provider(ds):
    return DataStoreDataProvider(ds)


# ═════════════════════════════════════════════════════════════════════════════
# Schema Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestSchema:
    def test_tables_registered(self, ds):
        for name in SCHEMA_TABLES:
            assert ds.has_table(name), f"Table '{name}' not registered"

    def test_tables_empty_initially(self, ds):
        for name in SCHEMA_TABLES:
            table = ds.get_table(name)
            assert len(table) == 0, f"Table '{name}' should be empty, has {len(table)} rows"

    def test_table_schemas_correct(self, ds):
        for name, defn in SCHEMA_TABLES.items():
            table = ds.get_table(name)
            for col, dtype in defn["schema"].items():
                assert col in table.columns, f"Missing column '{col}' in {name}"

    def test_primary_keys_set(self, ds):
        for name, defn in SCHEMA_TABLES.items():
            table = ds.get_table(name)
            assert table.primary_key == defn["primary_key"]

    def test_create_without_fetcher(self):
        ds = create_market_datastore(fetcher=None)
        assert ds.has_table("prices")
        # No fetcher → has_row won't auto-populate
        table = ds.get_table("prices")
        assert not table.has_row_filtered("security_id", "AAPL", fill_missing=True)


# ═════════════════════════════════════════════════════════════════════════════
# Fetcher Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMockFetcher:
    def test_fetch_security(self, fetcher):
        df = fetcher("securities", "security_id", "SPY")
        assert len(df) == 1
        assert df["security_id"][0] == "SPY"
        assert df["asset_class"][0] == "equity"
        assert df["name"][0] == "SPDR S&P 500"

    def test_fetch_security_unknown(self, fetcher):
        df = fetcher("securities", "security_id", "ZZZZ")
        assert len(df) == 1
        assert df["security_id"][0] == "ZZZZ"
        assert df["asset_class"][0] == "equity"  # default

    def test_fetch_prices_shape(self, fetcher):
        df = fetcher("prices", "security_id", "SPY")
        assert len(df) == 504
        assert set(df.columns) == {
            "security_id", "date", "open", "high", "low",
            "close", "volume", "adj_close",
        }

    def test_fetch_prices_types(self, fetcher):
        df = fetcher("prices", "security_id", "SPY")
        assert df["date"].dtype == pl.Date
        assert df["close"].dtype == pl.Float64
        assert df["security_id"].dtype == pl.Utf8

    def test_fetch_prices_positive(self, fetcher):
        df = fetcher("prices", "security_id", "SPY")
        assert (df["close"] > 0).all()
        assert (df["high"] >= df["low"]).all()

    def test_bond_lower_vol(self, fetcher):
        """Fixed income should have lower vol than equity."""
        spy = fetcher("prices", "security_id", "SPY")
        agg = fetcher("prices", "security_id", "AGG")
        spy_vol = np.std(np.diff(np.log(spy["close"].to_numpy())))
        agg_vol = np.std(np.diff(np.log(agg["close"].to_numpy())))
        assert agg_vol < spy_vol, "Bond vol should be lower than equity vol"

    def test_fetch_universe_members(self, fetcher):
        df = fetcher("universe_members", "universe_id", "TAA_5")
        assert len(df) == 5
        assert set(df["security_id"].to_list()) == {"SPY", "EFA", "AGG", "GLD", "VNQ"}

    def test_fetch_unknown_universe(self, fetcher):
        df = fetcher("universe_members", "universe_id", "NONEXISTENT")
        assert df is None

    def test_fetch_unknown_table(self, fetcher):
        result = fetcher("nonexistent_table", "id", "X")
        assert result is None

    def test_cache_consistency(self, fetcher):
        """Same ticker should return identical data on second call."""
        df1 = fetcher("prices", "security_id", "GLD")
        df2 = fetcher("prices", "security_id", "GLD")
        assert df1.equals(df2)

    def test_different_tickers_different_data(self, fetcher):
        """Different tickers should produce different price series."""
        spy = fetcher("prices", "security_id", "SPY")
        gld = fetcher("prices", "security_id", "GLD")
        assert not np.allclose(
            spy["close"].to_numpy(), gld["close"].to_numpy()
        ), "Different tickers should have different prices"


# ═════════════════════════════════════════════════════════════════════════════
# Lazy Loading / DataStore Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestLazyLoading:
    def test_has_row_triggers_fetch(self, ds):
        """has_row(fill_missing=True) should populate the table."""
        table = ds.get_table("securities")
        assert len(table) == 0

        result = table.has_row_filtered("security_id", "SPY", fill_missing=True)
        assert result is True
        assert len(table) == 1
        assert table.data["security_id"][0] == "SPY"

    def test_prices_lazy_load(self, ds):
        """Prices table should populate on demand."""
        prices = ds.get_table("prices")
        assert len(prices) == 0

        prices.has_row_filtered("security_id", "AAPL", fill_missing=True)
        assert len(prices) == 504  # n_days from fixture

    def test_multiple_tickers_accumulate(self, ds):
        """Loading multiple tickers should accumulate in the same table."""
        prices = ds.get_table("prices")

        prices.has_row_filtered("security_id", "SPY", fill_missing=True)
        assert len(prices) == 504

        prices.has_row_filtered("security_id", "GLD", fill_missing=True)
        assert len(prices) == 1008  # 504 * 2

    def test_second_fetch_is_cache_hit(self, ds):
        """Second has_row for same ticker should not re-fetch."""
        prices = ds.get_table("prices")

        prices.has_row_filtered("security_id", "SPY", fill_missing=True)
        count_after_first = len(prices)

        # Second call — should be a cache hit, no new rows
        prices.has_row_filtered("security_id", "SPY", fill_missing=True)
        assert len(prices) == count_after_first

    def test_universe_members_lazy_load(self, ds):
        members = ds.get_table("universe_members")
        assert len(members) == 0

        members.has_row_filtered("universe_id", "TAA_5", fill_missing=True)
        assert len(members) == 5


# ═════════════════════════════════════════════════════════════════════════════
# Bridge (DataStoreDataProvider) Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBridge:
    def test_fetch_prices_shape(self, provider):
        universe = UniverseSpec(tickers=["SPY", "GLD", "AGG"])
        temporal = TemporalSpec(lookback_days=504)
        prices = provider.fetch_prices(universe, temporal)

        assert prices.ndim == 2
        assert prices.shape[1] == 3
        assert prices.shape[0] > 400  # most of 504 days should have data

    def test_fetch_prices_no_nan(self, provider):
        universe = UniverseSpec(tickers=["SPY"])
        temporal = TemporalSpec(lookback_days=252)
        prices = provider.fetch_prices(universe, temporal)
        assert not np.isnan(prices).any()

    def test_asset_names_match(self, provider):
        universe = UniverseSpec(tickers=["SPY", "EFA", "AGG"])
        temporal = TemporalSpec(lookback_days=252)
        provider.fetch_prices(universe, temporal)

        names = provider.asset_names()
        assert names == ["SPY", "EFA", "AGG"]

    def test_universe_by_name(self, provider):
        universe = UniverseSpec(name="TAA_5")
        temporal = TemporalSpec(lookback_days=504)
        prices = provider.fetch_prices(universe, temporal)

        assert prices.shape[1] == 5
        names = provider.asset_names()
        assert set(names) == {"SPY", "EFA", "AGG", "GLD", "VNQ"}

    def test_exclusions(self, provider):
        universe = UniverseSpec(name="TAA_5", exclusions=["GLD", "VNQ"])
        temporal = TemporalSpec(lookback_days=252)
        prices = provider.fetch_prices(universe, temporal)

        assert prices.shape[1] == 3
        assert "GLD" not in provider.asset_names()
        assert "VNQ" not in provider.asset_names()

    def test_empty_universe_raises(self, provider):
        with pytest.raises(ValueError, match="must have either"):
            provider.fetch_prices(
                UniverseSpec(tickers=[]),
                TemporalSpec(lookback_days=252),
            )

    def test_no_tickers_or_name_raises(self, provider):
        with pytest.raises(ValueError, match="must have either"):
            provider.fetch_prices(
                UniverseSpec(),
                TemporalSpec(lookback_days=252),
            )

    def test_fetch_ohlcv(self, provider):
        df = provider.fetch_ohlcv(["SPY", "GLD"])
        assert isinstance(df, pl.DataFrame)
        assert "open" in df.columns
        assert "high" in df.columns
        assert len(df["security_id"].unique()) == 2

    def test_get_security_info(self, provider):
        info = provider.get_security_info("SPY")
        assert info["security_id"] == "SPY"
        assert info["name"] == "SPDR S&P 500"

    def test_caching_across_calls(self, provider):
        """Second fetch_prices should use cached data."""
        universe = UniverseSpec(tickers=["SPY"])
        temporal = TemporalSpec(lookback_days=252)

        p1 = provider.fetch_prices(universe, temporal)
        p2 = provider.fetch_prices(universe, temporal)
        np.testing.assert_array_equal(p1, p2)


# ═════════════════════════════════════════════════════════════════════════════
# End-to-End: DataStore → Provider → Engine → Results
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Full pipeline tests: live data from DataStore through Praxis engines."""

    def test_stat_arb_pairs(self, provider):
        """GLD/GDX cointegration scan via DataStore-sourced data."""
        from engines.cointegration import StatArbEngine, StatArbParams, StatArbInput

        universe = UniverseSpec(tickers=["GLD", "GDX"])
        temporal = TemporalSpec(lookback_days=504)
        prices = provider.fetch_prices(universe, temporal)
        names = provider.asset_names()

        engine = StatArbEngine()
        result = engine.compute(
            StatArbInput(prices=prices, asset_names=names),
            StatArbParams(
                entry_threshold=1.5,
                zscore_lookback=63,
                regression_method="ols",
                max_hurst=1.1,          # lenient for synthetic data
                scoring_mode="classic",
            ),
        )
        assert result.ok or result.status.value == "partial"
        # With cointegrated mock data, we should find the pair
        # but if stationarity tests are borderline, at least verify the engine ran
        assert result.diagnostics.get("n_assets") == 2

    def test_risk_parity_taa(self, provider):
        """Risk parity on TAA universe via DataStore-sourced data."""
        from engines.allocation import AllocationEngine, AllocationParams, AllocationInput

        universe = UniverseSpec(name="TAA_5")
        temporal = TemporalSpec(lookback_days=504)
        prices = provider.fetch_prices(universe, temporal)
        names = provider.asset_names()

        returns = np.diff(prices, axis=0) / prices[:-1]

        engine = AllocationEngine()
        result = engine.compute(
            AllocationInput(returns=returns, asset_names=names),
            AllocationParams(method="risk_parity", cov_method="shrinkage", long_only=True),
        )
        assert result.ok
        w = result.result.weights
        assert abs(w.sum() - 1.0) < 0.01

        # Verify equal risk contributions (Spinu solver)
        cov = result.covariance.matrix
        sp = np.sqrt(w @ cov @ w)
        rc = w * (cov @ w) / sp
        rc_pct = rc / rc.sum()
        np.testing.assert_allclose(rc_pct, np.ones(5) / 5, atol=0.01)

    def test_momentum_sector_scan(self, provider):
        """Cross-sectional momentum on sector ETFs."""
        from engines.momentum import MomentumEngine, MomentumParams, MomentumInput

        universe = UniverseSpec(name="SECTOR_ETFS")
        temporal = TemporalSpec(lookback_days=504)
        prices = provider.fetch_prices(universe, temporal)
        names = provider.asset_names()

        engine = MomentumEngine()
        result = engine.compute(
            MomentumInput(prices=prices, asset_names=names),
            MomentumParams(
                lookback_periods=[252],
                skip_recent=21,
                scoring_method="cross_sectional",
                sizing_method="vol_target",
            ),
        )
        assert result.ok
        assert len(result.scores) == len(names)

    def test_full_pipeline_momentum_to_allocation(self, provider):
        """Momentum scores feed into allocation — full pipeline."""
        from engines.momentum import MomentumEngine, MomentumParams, MomentumInput
        from engines.allocation import AllocationEngine, AllocationParams, AllocationInput

        universe = UniverseSpec(name="TAA_5")
        temporal = TemporalSpec(lookback_days=504)
        prices = provider.fetch_prices(universe, temporal)
        names = provider.asset_names()

        # Stage 1: Momentum scores as expected returns
        mom = MomentumEngine()
        mom_result = mom.compute(
            MomentumInput(prices=prices, asset_names=names),
            MomentumParams(lookback_periods=[126], scoring_method="time_series"),
        )
        assert mom_result.ok

        # Stage 2: Allocation using momentum as expected returns
        returns = np.diff(prices, axis=0) / prices[:-1]
        expected_returns = np.array([s.composite_score for s in mom_result.scores])

        alloc = AllocationEngine()
        alloc_result = alloc.compute(
            AllocationInput(returns=returns, asset_names=names),
            AllocationParams(
                method="max_sharpe",
                expected_returns=expected_returns,
                long_only=True,
            ),
        )
        assert alloc_result.ok
        assert abs(alloc_result.result.weights.sum() - 1.0) < 0.05

    def test_model_orchestrator_with_datastore(self, provider):
        """Model orchestrator using DataStoreDataProvider."""
        from engines.model import Model
        from engines.cointegration import StatArbEngine
        from engines.context.model_context import ModelContext

        universe = UniverseSpec(tickers=["GLD", "GDX"], name="pairs_test")
        temporal = TemporalSpec(lookback_days=504)
        context = ModelContext(universe=universe, temporal=temporal, name="stat_arb_test")

        model = Model(
            engine=StatArbEngine(),
            context=context,
            data_provider=provider,
        )
        result = model.run()
        assert result is not None

    def test_sector_etf_stat_arb_universe(self, provider):
        """Stat-arb scan across 11 sector ETFs — realistic universe size."""
        from engines.cointegration import StatArbEngine, StatArbParams, StatArbInput

        universe = UniverseSpec(name="SECTOR_ETFS")
        temporal = TemporalSpec(lookback_days=504)
        prices = provider.fetch_prices(universe, temporal)
        names = provider.asset_names()

        engine = StatArbEngine()
        result = engine.compute(
            StatArbInput(prices=prices, asset_names=names),
            StatArbParams(
                max_candidates=10,
                n_per_basket=2,
                scoring_mode="composite",
            ),
        )
        assert result.ok
        # With 11 assets, C(11,2) = 55 pairs to scan
        assert result.diagnostics.get("n_assets") == len(names)

    def test_global_futures_momentum(self, provider):
        """CTA-style trend following on global futures universe."""
        from engines.momentum import MomentumEngine, MomentumParams, MomentumInput

        universe = UniverseSpec(name="GLOBAL_FUTURES", asset_class=AssetClass.FUTURES)
        temporal = TemporalSpec(lookback_days=504)
        prices = provider.fetch_prices(universe, temporal)
        names = provider.asset_names()

        engine = MomentumEngine()
        result = engine.compute(
            MomentumInput(prices=prices, asset_names=names),
            MomentumParams(
                lookback_periods=[252],
                breakout_method="donchian",
                breakout_period=55,
                sizing_method="vol_target",
                scoring_method="time_series",
            ),
        )
        assert result.ok
        # Each asset should have a score
        assert len(result.scores) == len(names)
