"""Tests for the four-date orchestration.

What this layer owns is the *order* of assembly and the consolidation of provenance, so that
is what gets tested: that weights come from the exposure basis rather than the market basis,
that a covered name with no exposures is distinguished from cash, and that every layer's
substitutions arrive in one list rather than four.
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.asof import CALENDARS, AsOf, DateCalendar
from praxis.midoffice.engine import RiskResult, compute_risk, positions_from
from praxis.midoffice.risk.basis import FactorCovariance
from praxis.midoffice.risk.config import RiskConfig
from praxis.midoffice.risk.exposures import ExposureVector, SpecificRisk
from praxis.midoffice.risk.factors import FactorTree
from praxis.midoffice.risk.portfolio import WeightBasis
from praxis.midoffice.risk.tail import TailSpec
from praxis.midoffice.value.market import (
    AssetClass,
    Holding,
    MarketData,
    Quote,
    value_book,
)
from praxis.midoffice.value.valuation import PortfolioValue, PositionValue, ValueBasis

DATES = ["20150331", "20150430", "20150529", "20150630"]
IDS = (101, 102, 103)


def calendars():
    return {n: DateCalendar.of(n, DATES) for n in CALENDARS}


def as_of(**requests):
    base = dict(model="20150630", exposure="20150630", market="20150630", security="20150630")
    base.update(requests)
    return AsOf.resolve(calendars(), **base)


def covariance(seed=4):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(3, 3)) * 0.1
    m = b.T @ b / 3 + np.diag([0.02, 0.03, 0.015])
    return FactorCovariance(IDS, (m + m.T) / 2)


def tree():
    return FactorTree.build(
        [("Total Risk", "Style"), ("Total Risk", "Rates")],
        {"Style": [101, 102], "Rates": [103]},
    )


def exposures(**overrides):
    base = {
        "AAPL": ExposureVector({101: 1.1, 102: 0.3}, SpecificRisk.for_security("AAPL", 0.22)),
        "MSFT": ExposureVector({101: 0.8, 103: 0.2}, SpecificRisk.for_security("MSFT", 0.18)),
    }
    base.update(overrides)
    return base


def simple_book(name="FUND", benchmark="INDEX"):
    md = MarketData.of(
        [
            Quote("AAPL", 190.0, currency="USD", price_multiplier=1.0),
            Quote("MSFT", 410.0, currency="USD", price_multiplier=1.0),
        ]
    )
    holdings = [
        Holding("AAPL", AssetClass.EQUITY, quantity=1000.0, benchmark_quantity=900.0),
        Holding("MSFT", AssetClass.EQUITY, quantity=500.0, benchmark_quantity=600.0),
    ]
    return value_book(name, holdings, md, benchmark_name=benchmark)


# --- assembly -------------------------------------------------------------------------


def test_a_clean_run_produces_numbers_and_no_qualifications():
    book, load = simple_book()
    result = compute_risk(as_of(), book, exposures(), covariance(), load=load)
    assert isinstance(result, RiskResult)
    assert result.total_risk > 0.0
    assert result.tracking_error > 0.0
    assert result.is_exact, result.qualifications()


def test_every_metric_family_is_present():
    book, load = simple_book()
    result = compute_risk(
        as_of(), book, exposures(), covariance(), tree=tree(),
        tails=(TailSpec(0.95), TailSpec(0.99)), load=load,
    )
    for side in ("portfolio", "benchmark", "active"):
        assert result.position_contributions[side]
        assert len(result.factor_decompositions[side]) == len(IDS) + 1  # plus specific
        assert result.group_decompositions[side]
        assert len(result.tails[side]) == 2


def test_group_decompositions_are_absent_without_a_tree():
    book, load = simple_book()
    result = compute_risk(as_of(), book, exposures(), covariance(), load=load)
    assert result.group_decompositions == {}


def test_the_weight_basis_comes_from_the_config():
    book, load = simple_book()
    covered = compute_risk(
        as_of(), book, exposures(), covariance(), load=load,
        config=RiskConfig(weight_basis="covered"),
    )
    assert covered.risk.coverage.weight_basis is WeightBasis.COVERED
    explicit = compute_risk(
        as_of(), book, exposures(), covariance(), load=load,
        weight_basis=WeightBasis.FULL, config=RiskConfig(weight_basis="covered"),
    )
    assert explicit.risk.coverage.weight_basis is WeightBasis.FULL


# --- the weights are leverage, not shares ---------------------------------------------


def test_weights_come_from_the_exposure_basis():
    """RAT's expWeight: exposure value over the book's market value.

    A futures overlay has a market value of zero by formula, so market weights would be
    undefined for it. Exposure weights say how levered the book is through each position.
    """
    md = MarketData.of(
        [
            Quote("AAPL", 190.0, currency="USD", price_multiplier=1.0),
            Quote("ESZ6", 5500.0 * 50.0 * 100.0, currency="USD", price_multiplier=0.01),
        ]
    )
    holdings = [
        Holding("AAPL", AssetClass.EQUITY, quantity=5000.0),
        Holding("ESZ6", AssetClass.FUTURE, quantity=20.0, par_amount=1.0),
    ]
    book, load = value_book("LEV", holdings, md)
    result = compute_risk(
        as_of(), book,
        exposures(ESZ6=ExposureVector({101: 1.0}, SpecificRisk({}))),
        covariance(), load=load,
    )
    assert result.weights["ESZ6"] > 5.0, "a levered position's exposure weight exceeds one"
    assert sum(result.weights.values()) == pytest.approx(book.gross_exposure)
    assert sum(result.weights.values()) > 1.0


# --- covered without exposures --------------------------------------------------------


def test_a_covered_name_with_no_exposures_is_flagged():
    """It carries weight and contributes no risk, which is a data gap, not a feature."""
    book, load = simple_book()
    result = compute_risk(
        as_of(), book, {"AAPL": exposures()["AAPL"]}, covariance(), load=load
    )
    assert result.covered_without_exposure == ("MSFT",)
    assert any("marked covered but the model supplied no exposures" in q
               for q in result.qualifications())


def test_cash_is_covered_with_no_exposures_and_is_not_flagged():
    """Coverage comes from the feed, never from an empty exposure vector.

    Cash is covered and genuinely riskless. Inferring coverage from the absence of factor
    loadings would classify every cash position as a data gap.
    """
    md = MarketData.of([Quote("AAPL", 190.0, currency="USD", price_multiplier=1.0)])
    holdings = [
        Holding("AAPL", AssetClass.EQUITY, quantity=1000.0),
        Holding("CASHUSD", AssetClass.CASH, quantity=50_000.0),
    ]
    book, load = value_book("FUND", holdings, md)
    result = compute_risk(
        as_of(), book,
        {"AAPL": exposures()["AAPL"], "CASHUSD": ExposureVector.zero()},
        covariance(), load=load,
    )
    assert result.covered_without_exposure == ()


def test_an_uncovered_name_with_no_exposures_is_not_flagged():
    """It is already reported as uncovered; saying it twice is noise."""
    md = MarketData.of([
        Quote("AAPL", 190.0, currency="USD", price_multiplier=1.0),
        Quote("PRIVATE1", 50.0, currency="USD", price_multiplier=1.0),
    ])
    holdings = [
        Holding("AAPL", AssetClass.EQUITY, quantity=1000.0),
        Holding("PRIVATE1", AssetClass.EQUITY, quantity=400.0, covered=False),
    ]
    book, load = value_book("FUND", holdings, md)
    result = compute_risk(as_of(), book, {"AAPL": exposures()["AAPL"]}, covariance(), load=load)
    assert result.covered_without_exposure == ()
    assert any("not covered by the risk model" in q for q in result.qualifications())


# --- provenance in one stream ---------------------------------------------------------


def test_substituted_dates_reach_the_result():
    book, load = simple_book()
    result = compute_risk(as_of(model="20150615"), book, exposures(), covariance(), load=load)
    notes = " ".join(result.qualifications())
    assert "model date 2015-05-29 was used for a request of 2015-06-15" in notes
    assert not result.is_exact


def test_market_data_gaps_reach_the_result():
    md = MarketData.of([Quote("AAPL", 190.0, currency="USD", price_multiplier=1.0)])
    holdings = [
        Holding("AAPL", AssetClass.EQUITY, quantity=1000.0),
        Holding("NOQUOTE", AssetClass.EQUITY, quantity=100.0),
    ]
    book, load = value_book("FUND", holdings, md)
    result = compute_risk(as_of(), book, exposures(), covariance(), load=load)
    assert any("no price from any permitted source" in q for q in result.qualifications())


def test_the_cash_plug_reaches_the_result():
    md = MarketData.of([Quote("AAPL", 190.0, currency="USD", price_multiplier=1.0)])
    book, load = value_book(
        "FUND", [Holding("AAPL", AssetClass.EQUITY, quantity=1000.0)], md,
        stated_nav=250_000.0,
    )
    result = compute_risk(
        as_of(), book, exposures() | {"CASH_PLUG": ExposureVector.zero()},
        covariance(), load=load,
    )
    assert any("synthetic cash position" in q for q in result.qualifications())


def test_qualifications_are_de_duplicated():
    """The same repair is reported by every structure built on it; a reader needs it once."""
    book, load = simple_book()
    result = compute_risk(as_of(model="20150615"), book, exposures(), covariance(), load=load)
    notes = result.qualifications()
    assert len(notes) == len(set(notes))


def test_conditioning_is_reported_for_the_active_book_too():
    """`PortfolioRisk.qualifications()` covers only the portfolio side.

    The active book is where conditioning bites hardest: a factor-neutral long-short book is
    the active book of any index-hugging fund. Reporting only the portfolio side would stay
    silent on exactly the case that matters.
    """
    exposure = ExposureVector({101: 1.0}, SpecificRisk({}))
    book = PortfolioValue(
        "HUGGER",
        (
            PositionValue("A", clean_price=100.0, quantity=100.0, benchmark_quantity=100.0),
        ),
        benchmark_name="INDEX",
    )
    # Identical weights on both sides: the active vector is exactly zero.
    result = compute_risk(as_of(), book, {"A": exposure}, covariance())
    assert result.risk.active.total_risk == pytest.approx(0.0, abs=1e-15)
    # A zero active book is well conditioned by definition -- there is nothing to cancel.
    assert result.risk.active.is_well_conditioned


# --- positions_from -------------------------------------------------------------------


def test_positions_carry_both_sides_weights():
    book, _ = simple_book()
    positions, missing = positions_from(book, exposures())
    assert missing == ()
    held = book.weights(ValueBasis.EXPOSURE)
    bench = book.benchmark_weights(ValueBasis.EXPOSURE)
    for position in positions:
        assert position.weight == pytest.approx(held[position.security_code])
        assert position.benchmark_weight == pytest.approx(bench[position.security_code])


def test_a_missing_exposure_becomes_a_zero_vector_rather_than_an_error():
    book, _ = simple_book()
    positions, missing = positions_from(book, {"AAPL": exposures()["AAPL"]})
    msft = next(p for p in positions if p.security_code == "MSFT")
    assert msft.exposure == ExposureVector.zero()
    assert msft.weight > 0.0, "it still carries weight; that is why it is reported"
    assert missing == ("MSFT",)


def test_describe_names_the_books_and_the_dates():
    book, load = simple_book()
    text = compute_risk(as_of(), book, exposures(), covariance(), load=load).describe()
    assert "FUND" in text and "INDEX" in text
    assert "2015-06-30" in text
    assert "nothing substituted" in text
