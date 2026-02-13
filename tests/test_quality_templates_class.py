"""
Tests for Phase 4.11 (Time-Series Quality), 4.12 (Template Defaults), 4.13 (Classification Mappings).
"""

import numpy as np
import pytest
from datetime import date, timedelta

from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.11: Time-Series Quality
# ═══════════════════════════════════════════════════════════════════

from praxis.quality.timeseries import (
    TimeSeriesQuality,
    TimeSeriesQualityResult,
    GapInfo,
    OutlierInfo,
    StaleInfo,
    CorporateActionInfo,
)


def _make_clean_prices(n=252, seed=42):
    rng = np.random.RandomState(seed)
    returns = rng.randn(n) * 0.01
    return 100 * np.exp(np.cumsum(returns))


def _make_trading_dates(n=252, start=date(2024, 1, 2)):
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


class TestGapDetection:
    def test_no_gaps(self):
        checker = TimeSeriesQuality()
        prices = _make_clean_prices()
        dates = _make_trading_dates(len(prices))
        calendar = set(dates)
        result = checker.check(prices, dates, calendar)
        assert result.gaps_detected == 0

    def test_missing_days(self):
        checker = TimeSeriesQuality()
        all_dates = _make_trading_dates(100)
        # Remove 5 consecutive days
        kept_dates = all_dates[:20] + all_dates[25:]
        prices = _make_clean_prices(len(kept_dates))
        calendar = set(all_dates)
        result = checker.check(prices, kept_dates, calendar)
        assert result.gaps_detected >= 1
        assert result.gaps[0].missing_days == 5

    def test_no_calendar_skips_gaps(self):
        checker = TimeSeriesQuality()
        prices = _make_clean_prices()
        result = checker.check(prices)
        assert result.gaps_detected == 0

    def test_gap_severity(self):
        checker = TimeSeriesQuality(gap_severity_threshold=3)
        all_dates = _make_trading_dates(50)
        # Remove 4 days → error severity
        kept = all_dates[:10] + all_dates[14:]
        prices = _make_clean_prices(len(kept))
        result = checker.check(prices, kept, set(all_dates))
        assert any(g.severity == "error" for g in result.gaps)


class TestOutlierDetection:
    def test_no_outliers_clean(self):
        checker = TimeSeriesQuality(outlier_std=5.0)
        prices = _make_clean_prices(200)
        result = checker.check(prices)
        # Clean random walk should have very few outliers
        assert result.outliers_detected <= 2

    def test_injected_outlier(self):
        checker = TimeSeriesQuality(outlier_std=4.0, volatility_lookback=30)
        prices = _make_clean_prices(200)
        # Inject a huge spike
        prices[100] = prices[99] * 1.50  # +50% in one day
        result = checker.check(prices)
        assert result.outliers_detected >= 1
        outlier_indices = [o.index for o in result.outliers]
        assert 100 in outlier_indices

    def test_short_series(self):
        checker = TimeSeriesQuality()
        prices = np.array([100, 101, 102])
        result = checker.check(prices)
        assert isinstance(result, TimeSeriesQualityResult)


class TestStaleDetection:
    def test_no_stale_normal(self):
        checker = TimeSeriesQuality(stale_threshold=5)
        prices = _make_clean_prices()
        result = checker.check(prices)
        assert result.stale_detected == 0

    def test_stale_detected(self):
        checker = TimeSeriesQuality(stale_threshold=3)
        prices = _make_clean_prices(50)
        # Insert flat period
        prices[20:28] = 105.0
        result = checker.check(prices)
        assert result.stale_detected >= 1
        assert result.stale_periods[0].n_unchanged >= 3

    def test_stale_with_zero_volume(self):
        checker = TimeSeriesQuality(stale_threshold=3)
        prices = np.ones(20) * 100.0
        volumes = np.zeros(20)
        result = checker.check(prices, volumes=volumes)
        assert result.stale_detected >= 1
        assert result.stale_periods[0].severity == "error"


class TestCorporateActionDetection:
    def test_2_for_1_split(self):
        checker = TimeSeriesQuality()
        prices = np.array([100, 101, 102, 50.5, 51, 52], dtype=float)
        result = checker.check(prices)
        assert len(result.corporate_actions) >= 1
        action = result.corporate_actions[0]
        assert action.action_type == "split"
        assert action.confidence == "high"

    def test_reverse_split(self):
        checker = TimeSeriesQuality()
        prices = np.array([10, 10.5, 11, 33.0, 33.5], dtype=float)
        result = checker.check(prices)
        assert len(result.corporate_actions) >= 1
        assert result.corporate_actions[0].action_type == "reverse_split"

    def test_no_action_normal(self):
        checker = TimeSeriesQuality()
        prices = _make_clean_prices(100)
        result = checker.check(prices)
        # Normal random walk shouldn't trigger many corporate actions
        high_conf = [a for a in result.corporate_actions if a.confidence == "high"]
        assert len(high_conf) == 0


class TestQualityScore:
    def test_clean_series(self):
        checker = TimeSeriesQuality()
        prices = _make_clean_prices()
        result = checker.check(prices)
        assert result.quality_score >= 0.9

    def test_degraded_series(self):
        checker = TimeSeriesQuality(stale_threshold=3, outlier_std=3.0)
        prices = _make_clean_prices(200)
        prices[50:65] = 100.0  # Stale
        prices[100] = prices[99] * 2.0  # Outlier
        result = checker.check(prices)
        assert result.quality_score < 0.95

    def test_is_clean_property(self):
        checker = TimeSeriesQuality()
        prices = _make_clean_prices()
        result = checker.check(prices)
        assert result.is_clean == (result.quality_score >= 0.95)

    def test_summary_dict(self):
        checker = TimeSeriesQuality()
        result = checker.check(_make_clean_prices())
        s = result.summary
        assert "quality_score" in s
        assert "gaps" in s
        assert "outliers" in s


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.12: Template Defaults
# ═══════════════════════════════════════════════════════════════════

from praxis.templates.defaults import (
    SimulatedExecutionAdapter,
    LoggingExecutionAdapter,
    InMemoryDataSource,
    RandomWalkDataSource,
    MomentumSignal,
    MeanReversionSignal,
    CrossoverSignal,
    FixedFractionSizing,
    VolTargetSizing,
)


class TestSimulatedExecution:
    def test_connect(self):
        adapter = SimulatedExecutionAdapter()
        assert adapter.connect()

    def test_submit_order(self):
        adapter = SimulatedExecutionAdapter(slippage_bps=10)
        oid = adapter.submit_order({"asset": "AAPL", "side": "buy", "quantity": 100, "price": 185.0})
        assert oid.startswith("sim_")
        assert len(adapter.orders) == 1
        # Slippage applied
        assert adapter.orders[0]["fill_price"] > 185.0

    def test_sell_slippage(self):
        adapter = SimulatedExecutionAdapter(slippage_bps=10)
        adapter.submit_order({"asset": "AAPL", "side": "sell", "quantity": 50, "price": 185.0})
        assert adapter.orders[0]["fill_price"] < 185.0

    def test_commission(self):
        adapter = SimulatedExecutionAdapter(commission_per_share=0.01)
        adapter.submit_order({"asset": "AAPL", "side": "buy", "quantity": 100, "price": 100.0})
        assert adapter.orders[0]["commission"] == 1.0


class TestLoggingExecution:
    def test_logs_orders(self):
        adapter = LoggingExecutionAdapter()
        assert adapter.connect()
        oid = adapter.submit_order({"asset": "GOOG", "quantity": 50})
        assert oid.startswith("log_")
        assert len(adapter.log) == 1


class TestInMemoryDataSource:
    def test_load_and_fetch(self):
        source = InMemoryDataSource()
        prices = np.array([100, 101, 102, 103])
        source.load("AAPL", prices)
        data = source.fetch(["AAPL", "GOOG"], "2024-01-01", "2024-12-31")
        assert "AAPL" in data
        assert "GOOG" not in data
        np.testing.assert_array_equal(data["AAPL"], prices)


class TestRandomWalkDataSource:
    def test_generates_data(self):
        source = RandomWalkDataSource(n_obs=100, seed=42)
        data = source.fetch(["A", "B", "C"], "", "")
        assert len(data) == 3
        assert len(data["A"]) == 100

    def test_reproducible(self):
        s1 = RandomWalkDataSource(seed=123)
        s2 = RandomWalkDataSource(seed=123)
        d1 = s1.fetch(["X"], "", "")
        d2 = s2.fetch(["X"], "", "")
        np.testing.assert_array_equal(d1["X"], d2["X"])


class TestMomentumSignal:
    def test_generates_signals(self):
        sig = MomentumSignal()
        prices = _make_clean_prices(200)
        signals = sig.generate(prices, {"lookback": 20})
        assert len(signals) == 200
        # First 20 should be 0 (warmup)
        assert all(signals[:20] == 0)
        # Should have some +1 and -1
        assert 1.0 in signals
        assert -1.0 in signals


class TestMeanReversionSignal:
    def test_generates_signals(self):
        sig = MeanReversionSignal()
        # Mean-reverting series should generate signals
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.randn(300)) * 0.5
        signals = sig.generate(prices, {"lookback": 60, "entry": 2.0})
        assert len(signals) == 300


class TestCrossoverSignal:
    def test_generates_signals(self):
        sig = CrossoverSignal()
        prices = _make_clean_prices(200)
        signals = sig.generate(prices, {"fast": 10, "slow": 30})
        assert len(signals) == 200
        assert all(signals[:30] == 0)


class TestFixedFractionSizing:
    def test_scales_signals(self):
        sizer = FixedFractionSizing()
        signals = np.array([1, -1, 0, 1, -1], dtype=float)
        sized = sizer.size(signals, {"fraction": 0.05})
        np.testing.assert_allclose(sized, [0.05, -0.05, 0, 0.05, -0.05])


class TestVolTargetSizing:
    def test_without_prices(self):
        sizer = VolTargetSizing()
        signals = np.array([1, 0, -1], dtype=float)
        sized = sizer.size(signals, {"target_vol": 0.15})
        # Falls back to flat scaling
        assert len(sized) == 3

    def test_with_prices(self):
        sizer = VolTargetSizing()
        prices = _make_clean_prices(100)
        signals = np.zeros(100)
        signals[50:] = 1.0
        sized = sizer.size(signals, {"target_vol": 0.15, "vol_lookback": 20, "prices": prices})
        assert len(sized) == 100
        # After warmup, sized should be non-zero where signals are non-zero
        assert any(sized[50:] != 0)


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.13: Classification Mappings
# ═══════════════════════════════════════════════════════════════════

from praxis.classification import (
    cfi_to_core,
    sp_rating_to_tier,
    moody_to_tier,
    sp_to_moody_equivalent,
    msci_market_to_tier,
    msci_esg_to_tier,
    get_markets_by_tier,
    get_markets_by_region,
    compare_credit_ratings,
    get_all_mappings,
    CoreClassification,
    CreditTier,
    MarketTier,
    ESGTier,
)


class TestCFIMappings:
    def test_equity_common(self):
        result = cfi_to_core("ESVUFR")
        assert result is not None
        assert result.asset_class == "EQUITY"
        assert result.instrument_type == "COMMON_STOCK"

    def test_debt_corporate(self):
        result = cfi_to_core("DBFUFR")
        assert result is not None
        assert result.asset_class == "FIXED_INCOME"
        assert result.instrument_type == "CORPORATE_BOND"

    def test_etf(self):
        result = cfi_to_core("EFXXXX")
        assert result is not None
        assert result.instrument_type == "ETF"

    def test_option(self):
        result = cfi_to_core("OCXXXX")
        assert result is not None
        assert result.asset_class == "DERIVATIVE"

    def test_future(self):
        result = cfi_to_core("FFXXXX")
        assert result is not None
        assert result.asset_class == "DERIVATIVE"

    def test_fallback_single_char(self):
        result = cfi_to_core("EXXXXX")
        assert result is not None
        assert result.asset_class == "EQUITY"

    def test_invalid(self):
        assert cfi_to_core("") is None
        assert cfi_to_core("Z") is None


class TestSPRatings:
    def test_investment_grade(self):
        for rating in ["AAA", "AA+", "A", "BBB-"]:
            tier = sp_rating_to_tier(rating)
            assert tier is not None
            assert tier.is_investment_grade

    def test_high_yield(self):
        for rating in ["BB+", "BB", "B-"]:
            tier = sp_rating_to_tier(rating)
            assert tier is not None
            assert not tier.is_investment_grade
            assert tier.tier == "HIGH_YIELD"

    def test_distressed(self):
        tier = sp_rating_to_tier("CCC")
        assert tier is not None
        assert tier.tier == "DISTRESSED"

    def test_ordering(self):
        aaa = sp_rating_to_tier("AAA")
        bb = sp_rating_to_tier("BB")
        assert aaa.numeric_rank < bb.numeric_rank

    def test_invalid(self):
        assert sp_rating_to_tier("XYZ") is None


class TestMoodyRatings:
    def test_investment_grade(self):
        tier = moody_to_tier("Aaa")
        assert tier.is_investment_grade
        assert tier.numeric_rank == 1

    def test_high_yield(self):
        tier = moody_to_tier("Ba1")
        assert not tier.is_investment_grade

    def test_sp_to_moody(self):
        equiv = sp_to_moody_equivalent("AAA")
        assert equiv == "Aaa"
        equiv = sp_to_moody_equivalent("BB+")
        assert equiv == "Ba1"


class TestMSCIMarkets:
    def test_developed(self):
        tier = msci_market_to_tier("US")
        assert tier.tier == "DEVELOPED"
        assert tier.region == "AMERICAS"

    def test_emerging(self):
        tier = msci_market_to_tier("CN")
        assert tier.tier == "EMERGING"

    def test_frontier(self):
        tier = msci_market_to_tier("VN")
        assert tier.tier == "FRONTIER"

    def test_by_tier(self):
        developed = get_markets_by_tier("DEVELOPED")
        assert "US" in developed
        assert "CN" not in developed

    def test_by_region(self):
        apac = get_markets_by_region("ASIA_PACIFIC")
        assert "JP" in apac
        assert "US" not in apac

    def test_invalid(self):
        assert msci_market_to_tier("ZZ") is None


class TestMSCIESG:
    def test_leader(self):
        tier = msci_esg_to_tier("AAA")
        assert tier.tier == "LEADER"
        assert tier.numeric == 10.0

    def test_laggard(self):
        tier = msci_esg_to_tier("CCC")
        assert tier.tier == "LAGGARD"

    def test_average(self):
        tier = msci_esg_to_tier("BBB")
        assert tier.tier == "AVERAGE"


class TestCreditComparison:
    def test_aligned(self):
        result = compare_credit_ratings(sp_rating="AAA", moody_rating="Aaa")
        assert result["ig_agreement"] is True
        assert result["notch_difference"] == 0

    def test_split_rated(self):
        result = compare_credit_ratings(sp_rating="BBB-", moody_rating="Ba1")
        assert result["split_rated"] is True

    def test_partial(self):
        result = compare_credit_ratings(sp_rating="AA+")
        assert result["moody_tier"] is None


class TestGetAllMappings:
    def test_returns_all_systems(self):
        m = get_all_mappings()
        assert "cfi" in m
        assert "sp_ratings" in m
        assert "moody_ratings" in m
        assert "msci_markets" in m
        assert "msci_esg" in m
        assert len(m["sp_ratings"]) == 22
        assert len(m["msci_markets"]) > 50
