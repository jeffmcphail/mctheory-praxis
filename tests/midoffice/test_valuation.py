"""Tests for the value layer.

The load-bearing test here is `test_exposure_weight_is_notional_over_nav_not_over_notional`.
RAT's `expWeight` divides an exposure value by the book's *market* value, and implementing it
from the name instead of from the source gives `exposure / total_exposure`. That mistake is
invisible on a cash-equity book -- where market and exposure value are the same number, both
formulas agree exactly -- and wrong by the leverage factor the moment a future appears. So
there is a test that computes the wrong formula alongside the right one and asserts they
disagree, which is the only way to keep the distinction from being quietly "simplified" later.
"""

from __future__ import annotations

import pytest

from praxis.midoffice.value.valuation import (
    CASH_PLUG,
    Accrual,
    Coverage,
    PortfolioValue,
    PositionValue,
    ValueBasis,
)


def equity(code="AAPL", qty=1000.0, bench=1000.0, price=190.0, covered=True):
    return PositionValue(
        code, clean_price=price, quantity=qty, benchmark_quantity=bench, covered=covered
    )


def bond(code="T 4.5 30", qty=5000.0, bench=5000.0):
    return PositionValue(
        code, clean_price=98.5, accrued_interest=1.25, quantity=qty, benchmark_quantity=bench
    )


def future(code="ESZ6", contracts=20.0, bench=0.0, index=5500.0, size=50.0, mark=12.5):
    """A future: tiny market value, large notional. The case the axes exist for."""
    return PositionValue(
        code,
        clean_price=mark,
        exposure_price=index * size,
        quantity=contracts,
        benchmark_quantity=bench,
    )


# --- the per-unit formula -------------------------------------------------------------


def test_a_cash_equity_has_the_same_value_on_both_bases():
    p = equity()
    for accrual in Accrual:
        assert p.value(ValueBasis.MARKET, accrual) == p.value(ValueBasis.EXPOSURE, accrual)
    assert not p.is_leveraged


def test_a_future_has_a_market_value_unrelated_to_its_exposure():
    p = future()
    assert p.value(ValueBasis.MARKET, Accrual.DIRTY) == pytest.approx(250.0)
    assert p.value(ValueBasis.EXPOSURE, Accrual.DIRTY) == pytest.approx(5_500_000.0)
    assert p.is_leveraged


def test_dirty_less_clean_is_exactly_the_accrued_interest():
    p = bond()
    for basis in ValueBasis:
        difference = p.value(basis, Accrual.DIRTY) - p.value(basis, Accrual.CLEAN)
        assert difference == pytest.approx(p.accrued_interest * p.quantity)


def test_fx_scales_every_axis_identically():
    local = PositionValue("X", clean_price=100.0, accrued_interest=2.0,
                          exposure_price=500.0, quantity=10.0)
    foreign = PositionValue("X", clean_price=100.0, accrued_interest=2.0,
                            exposure_price=500.0, quantity=10.0, fx_rate=1.35)
    for basis in ValueBasis:
        for accrual in Accrual:
            assert foreign.value(basis, accrual) == pytest.approx(
                local.value(basis, accrual) * 1.35
            )


def test_exposure_price_defaults_to_the_market_price():
    """So the unleveraged case needs no input and cannot be got wrong by omission."""
    p = PositionValue("Y", clean_price=42.0, quantity=1.0)
    assert p.resolved_exposure_price == 42.0


def test_a_non_positive_fx_rate_is_refused():
    """Zero would silently zero the position; negative would flip its sign."""
    for rate in (0.0, -1.35):
        with pytest.raises(ValueError, match="fx_rate"):
            PositionValue("Z", clean_price=1.0, fx_rate=rate)


def test_an_unpriced_position_is_flagged_rather_than_dropped():
    assert not PositionValue("NOPRICE", clean_price=0.0, quantity=100.0).is_priced
    assert equity().is_priced
    # A bond quoted at zero clean but carrying accrued is still priced.
    assert PositionValue("ACCR", clean_price=0.0, accrued_interest=0.5).is_priced


# --- the weight, and the trap ---------------------------------------------------------


def test_exposure_weight_is_notional_over_nav_not_over_notional():
    """The asymmetry, asserted against the plausible wrong formula.

    RAT: expWeight = RiskExpValue / portRiskData.RiskMarketValue (RiskData.cs:99).

    A book with $950,250 of net asset value holding one S&P future with $5.5m of notional has
    an exposure weight of 5.788 on that future -- it is levered nearly six times through it.
    The wrong formula divides by total notional instead and reports 0.853, which looks like a
    perfectly ordinary weight on a position that happens to be most of the book.

    Expected values are derived from the fixture rather than written in by hand. The first
    draft of this test hard-coded them from arithmetic done in my head, got the NAV wrong, and
    failed against correct code -- which is the same mistake in miniature as asserting what a
    run produced instead of what must be true.
    """
    book = PortfolioValue("FUND", (equity(qty=5000.0, bench=0.0), future()))
    nav = book.total(ValueBasis.MARKET)
    notional = book.total(ValueBasis.EXPOSURE)
    future_notional = 20.0 * 5500.0 * 50.0

    correct = book.weights(ValueBasis.EXPOSURE)["ESZ6"]
    naive = future_notional / notional

    assert correct == pytest.approx(future_notional / nav)
    assert correct > 5.0, "a levered position's exposure weight exceeds one"
    assert naive < 1.0, "the wrong formula cannot report leverage at all"
    assert correct / naive == pytest.approx(notional / nav, rel=1e-12)
    assert correct != pytest.approx(naive, rel=0.1), "the two formulas must differ here"


def test_the_two_weight_formulas_agree_on_a_cash_only_book():
    """Which is why a cash-equity test suite cannot catch the wrong one."""
    book = PortfolioValue("CASH", (equity("A", 1000.0, 0.0), equity("B", 500.0, 0.0, 80.0)))
    notional = book.total(ValueBasis.EXPOSURE)
    for code, weight in book.weights(ValueBasis.EXPOSURE).items():
        position = next(p for p in book.positions if p.security_code == code)
        assert weight == pytest.approx(
            position.value(ValueBasis.EXPOSURE, Accrual.DIRTY) / notional
        )


def test_exposure_weights_sum_to_one_only_without_leverage():
    cash = PortfolioValue("CASH", (equity("A", 1000.0, 0.0), equity("B", 500.0, 0.0, 80.0)))
    assert sum(cash.weights(ValueBasis.EXPOSURE).values()) == pytest.approx(1.0)
    assert cash.gross_exposure == pytest.approx(1.0)

    levered = PortfolioValue("LEV", (equity(qty=5000.0, bench=0.0), future()))
    assert levered.gross_exposure > 1.0
    assert sum(levered.weights(ValueBasis.EXPOSURE).values()) == pytest.approx(
        levered.gross_exposure
    )


def test_market_weights_do_sum_to_one():
    """The market basis is a share of NAV, so it sums to one whatever the leverage."""
    book = PortfolioValue("LEV", (equity(qty=5000.0, bench=0.0), future()))
    assert sum(book.weights(ValueBasis.MARKET).values()) == pytest.approx(1.0)


def test_each_side_divides_by_its_own_denominator():
    """Otherwise the difference of the two is not an active weight."""
    book = PortfolioValue("FUND", (equity(qty=1000.0, bench=2000.0), bond()))
    held = book.weights(ValueBasis.MARKET)
    bench = book.benchmark_weights(ValueBasis.MARKET)
    assert sum(held.values()) == pytest.approx(1.0)
    assert sum(bench.values()) == pytest.approx(1.0)
    active = book.active_weights(ValueBasis.MARKET)
    assert sum(active.values()) == pytest.approx(0.0, abs=1e-12)
    for code in held:
        assert active[code] == pytest.approx(held[code] - bench[code])


# --- the denominator fallback ---------------------------------------------------------


def test_a_zero_market_value_book_falls_back_to_exposure():
    """A pure futures overlay: market value ~0, notional large. RAT: Portfolio.cs:835."""
    overlay = PortfolioValue("OVERLAY", (future(mark=0.0),))
    assert overlay.total(ValueBasis.MARKET) == 0.0
    assert overlay.denominator() == pytest.approx(5_500_000.0)
    assert overlay.weights(ValueBasis.EXPOSURE)["ESZ6"] == pytest.approx(1.0)


def test_the_fallback_is_reported():
    overlay = PortfolioValue("OVERLAY", (future(mark=0.0),))
    notes = overlay.notes()
    assert notes.market_total_fell_back_to_exposure
    assert any("fraction of notional" in q for q in notes.qualifications())


def test_an_empty_book_does_not_claim_a_fallback():
    notes = PortfolioValue("EMPTY", ()).notes()
    assert not notes.market_total_fell_back_to_exposure
    assert PortfolioValue("EMPTY", ()).weights() == {}


# --- the cash plug --------------------------------------------------------------------


def test_the_plug_makes_the_book_tie_to_its_stated_nav():
    book = PortfolioValue.with_cash_plug("FUND", (equity(), bond()), stated_nav=1_000_000.0)
    assert book.total(ValueBasis.MARKET) == pytest.approx(1_000_000.0)
    assert CASH_PLUG in {p.security_code for p in book.positions}


def test_the_plug_is_a_covered_riskless_position_not_a_hidden_adjustment():
    """It belongs in a holdings listing, which is where a reader can see it."""
    book = PortfolioValue.with_cash_plug("FUND", (equity(),), stated_nav=1_000_000.0)
    plug = next(p for p in book.positions if p.security_code == CASH_PLUG)
    assert plug.covered, "cash is covered -- genuinely riskless, not unmodelled"
    assert not plug.is_leveraged
    assert plug.value(ValueBasis.MARKET, Accrual.DIRTY) == pytest.approx(
        1_000_000.0 - 190_000.0
    )


def test_the_plug_is_reported():
    book = PortfolioValue.with_cash_plug("FUND", (equity(),), stated_nav=1_000_000.0)
    assert book.notes().cash_plug == pytest.approx(810_000.0)
    assert any("synthetic cash" in q for q in book.notes().qualifications())


def test_no_plug_is_inserted_for_float_noise():
    exact = 190_000.0
    book = PortfolioValue.with_cash_plug("FUND", (equity(),), stated_nav=exact + 0.001)
    assert CASH_PLUG not in {p.security_code for p in book.positions}
    assert book.notes().cash_plug == 0.0


def test_a_plug_sized_gap_is_inserted():
    book = PortfolioValue.with_cash_plug("FUND", (equity(),), stated_nav=190_000.0 + 1.00)
    assert CASH_PLUG in {p.security_code for p in book.positions}


def test_the_plug_enters_the_weight_denominator():
    """It is cash the book holds, so it dilutes every other weight. Stated, not hidden."""
    without = PortfolioValue("FUND", (equity(),))
    with_plug = PortfolioValue.with_cash_plug("FUND", (equity(),), stated_nav=1_000_000.0)
    assert without.weights(ValueBasis.MARKET)["AAPL"] == pytest.approx(1.0)
    assert with_plug.weights(ValueBasis.MARKET)["AAPL"] == pytest.approx(0.19)


# --- coverage and provenance ----------------------------------------------------------


def test_the_coverage_axis_selects_which_positions_count():
    book = PortfolioValue(
        "FUND", (equity(), equity("PRIVATE1", 400.0, 0.0, 50.0, covered=False))
    )
    assert book.total(ValueBasis.MARKET, coverage=Coverage.ALL) == pytest.approx(210_000.0)
    assert book.total(ValueBasis.MARKET, coverage=Coverage.COVERED) == pytest.approx(190_000.0)
    assert "PRIVATE1" not in book.weights(coverage=Coverage.COVERED)


def test_uncovered_value_is_reported():
    book = PortfolioValue(
        "FUND", (equity(), equity("PRIVATE1", 400.0, 0.0, 50.0, covered=False))
    )
    notes = book.notes()
    assert notes.n_uncovered == 1
    assert notes.uncovered_market_value == pytest.approx(20_000.0)


def test_a_clean_book_carries_no_qualifications():
    book = PortfolioValue("FUND", (equity(), equity("MSFT", 500.0, 500.0, 410.0)))
    assert book.notes().is_clean, book.notes().qualifications()


def test_leverage_is_reported_even_when_nothing_else_is_wrong():
    book = PortfolioValue("FUND", (equity(qty=5000.0, bench=0.0), future()))
    notes = book.notes()
    assert notes.n_leveraged == 1
    assert any("will not sum to one" in q for q in notes.qualifications())


def test_a_nav_gap_is_reported_when_no_plug_was_applied():
    book = PortfolioValue("FUND", (equity(),), stated_nav=250_000.0)
    assert book.notes().nav_gap == pytest.approx(60_000.0)
    assert any("stated NAV" in q for q in book.notes().qualifications())


# --- accrual convention ---------------------------------------------------------------


def test_the_accrual_convention_is_a_frozen_field_not_a_mutable_flag():
    """RAT memoises this choice on first read, so a later change serves a stale value.

    A frozen field cannot go stale. The override exists for a report that shows both bases.
    """
    dirty = PortfolioValue("FUND", (bond(),), accrual=Accrual.DIRTY)
    clean = PortfolioValue("FUND", (bond(),), accrual=Accrual.CLEAN)
    assert dirty.total(ValueBasis.MARKET) > clean.total(ValueBasis.MARKET)
    assert dirty.total(ValueBasis.MARKET, accrual=Accrual.CLEAN) == pytest.approx(
        clean.total(ValueBasis.MARKET)
    )
    with pytest.raises(Exception):
        dirty.accrual = Accrual.CLEAN  # type: ignore[misc]


def test_duplicate_securities_are_refused():
    with pytest.raises(ValueError, match="duplicate security"):
        PortfolioValue("FUND", (equity(), equity()))
