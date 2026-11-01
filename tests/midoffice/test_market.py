"""Tests for market data and the holdings-to-values loader.

The load-bearing test is `test_fixed_income_is_quoted_as_a_percent_of_par`. A bond quoted at
98.5 against a million of par is worth 985,000, not 98.5 -- the `.01 * par_amount` factor is
the difference, and getting it wrong is a hundredfold error that no equity test can see,
because equities carry neither factor.

`test_the_price_multiplier_table_matches_the_overrides_in_rat` pins the authoritative set.
RAT's documentation comment lists a formula per asset class and disagrees with its own code in
two rows, so the multipliers are taken from the `priceMultiplier` overrides instead.
"""

from __future__ import annotations

import pytest

from praxis.midoffice.value.market import (
    DEFAULT_SOURCE,
    AssetClass,
    Holding,
    MarketData,
    MissingPrice,
    MissingRate,
    Quote,
    rule_for,
    value_book,
    value_holding,
)
from praxis.midoffice.value.valuation import Accrual, ValueBasis

USD_ONLY = {("USD", "USD"): 1.0}


def market(*quotes: Quote, rates=None, numeraire="USD", default_source=DEFAULT_SOURCE):
    return MarketData.of(
        quotes, rates=rates or {}, numeraire=numeraire, default_source=default_source
    )


def valued(holding, md, **kwargs):
    position, source, missing_accrued = value_holding(holding, md, **kwargs)
    return position


# --- the asset-class formulas ---------------------------------------------------------


def test_the_price_multiplier_table_matches_the_overrides_in_rat():
    """Exactly four classes override priceMultiplier to .01; the rest inherit 1.

    Taken from the overrides in FISecurity, FIFutureSecurity, IRSWPSecurity and CDXSecurity,
    not from the documentation comment in RATEngine.cs -- which omits the .01 for interest rate
    swaps and asserts one for credit default swaps that CDSSecurity does not override.
    """
    percent_of_par = {
        c for c in AssetClass if rule_for(c).price_multiplier == 0.01
    }
    assert percent_of_par == {
        AssetClass.FIXED_INCOME,
        AssetClass.FUTURE,
        AssetClass.INTEREST_RATE_SWAP,
        AssetClass.INDEX_CDS,
    }
    assert rule_for(AssetClass.CREDIT_DEFAULT_SWAP).price_multiplier == 1.0, (
        "the stale comment claims .01 here; CDSSecurity has no override"
    )
    assert rule_for(AssetClass.INTEREST_RATE_SWAP).price_multiplier == 0.01, (
        "the stale comment omits .01 here; IRSWPSecurity does override it"
    )


def test_only_the_future_has_a_zero_market_value():
    """FIFutureSecurity.getPriceNumeraire is literally `return 0.0;`."""
    zero_mv = {c for c in AssetClass if rule_for(c).market_value_is_zero}
    assert zero_mv == {AssetClass.FUTURE}


def test_equity_is_price_times_fx_times_quantity():
    md = market(Quote("AAPL", 190.0, currency="USD"))
    p = valued(Holding("AAPL", AssetClass.EQUITY, quantity=1000.0), md)
    assert p.value(ValueBasis.MARKET, Accrual.DIRTY) == pytest.approx(190_000.0)
    assert p.value(ValueBasis.EXPOSURE, Accrual.DIRTY) == pytest.approx(190_000.0)


def test_fixed_income_is_quoted_as_a_percent_of_par():
    """The hundredfold trap.

    98.5 against a million of par is 985,000. Without the .01 * par_amount factor a quantity
    of one bond would be valued at 98.50.
    """
    md = market(Quote("T 4.5 30", 98.5, currency="USD"))
    p = valued(
        Holding("T 4.5 30", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0),
        md,
    )
    assert p.value(ValueBasis.MARKET, Accrual.CLEAN) == pytest.approx(985_000.0)
    assert p.value(ValueBasis.MARKET, Accrual.CLEAN) != pytest.approx(98.5)


def test_accrued_interest_scales_with_par_too():
    md = market(Quote("T 4.5 30", 98.5, accrued_interest=1.25, currency="USD"))
    p = valued(
        Holding("T 4.5 30", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0),
        md,
    )
    assert p.value(ValueBasis.MARKET, Accrual.DIRTY) == pytest.approx(997_500.0)
    difference = p.value(ValueBasis.MARKET, Accrual.DIRTY) - p.value(
        ValueBasis.MARKET, Accrual.CLEAN
    )
    assert difference == pytest.approx(12_500.0), "1.25 percent of a million of par"


def test_a_future_has_no_market_value_and_a_notional_exposure():
    notional_quote = 5500.0 * 50.0 * 100.0  # quoted as a percent of par, par 1
    md = market(Quote("ESZ6", notional_quote, currency="USD"))
    p = valued(
        Holding("ESZ6", AssetClass.FUTURE, quantity=20.0, par_amount=1.0), md
    )
    assert p.value(ValueBasis.MARKET, Accrual.DIRTY) == 0.0
    assert p.value(ValueBasis.EXPOSURE, Accrual.DIRTY) == pytest.approx(5_500_000.0)
    assert p.is_leveraged


def test_cash_needs_no_price_at_all():
    """RAT values cash as fx_rate * position: the quantity is the amount."""
    p = valued(Holding("CASHUSD", AssetClass.CASH, quantity=50_000.0), market())
    assert p.value(ValueBasis.MARKET, Accrual.DIRTY) == pytest.approx(50_000.0)
    assert p.is_priced


def test_a_non_positive_par_amount_is_refused_for_every_class():
    """Par always multiplies the price now, so zero is never meaningful.

    It used to be gated on the asset class, which meant an equity could carry par 0 harmlessly.
    Applying it universally is simpler -- an equity's par defaults to 1.0 and is inert -- but it
    makes a zero a real error everywhere rather than only where it was read.
    """
    for asset_class in (AssetClass.FIXED_INCOME, AssetClass.EQUITY, AssetClass.FUTURE):
        for bad in (0.0, -1000.0):
            with pytest.raises(ValueError, match="par_amount must be positive"):
                Holding("X", asset_class, quantity=1.0, par_amount=bad)
    assert Holding("AAPL", AssetClass.EQUITY).par_amount == 1.0


# --- the price multiplier as data -----------------------------------------------------


def test_a_feed_stated_multiplier_overrides_the_class_default():
    md = market(Quote("B", 98.5, currency="USD", price_multiplier=0.01))
    p = valued(
        Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0), md
    )
    assert p.value(ValueBasis.MARKET, Accrual.CLEAN) == pytest.approx(985_000.0)


def test_a_pre_normalised_feed_can_state_a_multiplier_of_one():
    """The case that was inexpressible before.

    A query that normalises prices itself delivers 0.985. Applying the class's hardcoded .01 on
    top gives 9,850 -- a hundredfold error with no symptom. Stating the basis in the data fixes
    it, which is the whole reason the multiplier moved out of the class table.
    """
    md = market(Quote("B", 0.985, currency="USD", price_multiplier=1.0))
    p = valued(
        Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0), md
    )
    assert p.value(ValueBasis.MARKET, Accrual.CLEAN) == pytest.approx(985_000.0)


def test_none_is_not_the_same_as_one():
    """None falls back to the class default; 1.0 asserts the price is already per unit."""
    silent = market(Quote("B", 98.5, currency="USD"))
    stated = market(Quote("B", 98.5, currency="USD", price_multiplier=1.0))
    holding = Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0)
    assert valued(holding, silent).value(ValueBasis.MARKET, Accrual.CLEAN) == pytest.approx(
        985_000.0
    )
    assert valued(holding, stated).value(ValueBasis.MARKET, Accrual.CLEAN) == pytest.approx(
        98_500_000.0
    )


def test_a_silent_feed_behaves_exactly_as_rat_did():
    """Which is what keeps value parity possible: RAT hardcoded these per class."""
    md = market(Quote("B", 98.5, currency="USD"))
    p = valued(
        Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0), md
    )
    assert p.value(ValueBasis.MARKET, Accrual.CLEAN) == pytest.approx(
        98.5 * 0.01 * 1_000_000.0
    )


def test_an_inferred_multiplier_is_reported():
    """The feed has not confirmed its own basis, so the assumption is on the result."""
    md = market(Quote("B", 98.5, currency="USD"))
    _, notes = value_book(
        "FUND",
        [Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0)],
        md,
    )
    assert notes.inferred_multiplier == ("B",)
    assert any("carried no price multiplier" in q for q in notes.qualifications())


def test_a_stated_multiplier_is_not_reported_as_inferred():
    md = market(Quote("B", 98.5, currency="USD", price_multiplier=0.01))
    _, notes = value_book(
        "FUND",
        [Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0)],
        md,
    )
    assert notes.inferred_multiplier == ()


def test_an_equity_with_a_unit_multiplier_is_not_reported():
    """Only a class whose default is not 1.0 is worth flagging as assumed."""
    md = market(Quote("AAPL", 190.0, currency="USD"))
    _, notes = value_book(
        "FUND", [Holding("AAPL", AssetClass.EQUITY, quantity=1000.0)], md
    )
    assert notes.inferred_multiplier == ()


def test_a_stated_unit_multiplier_leaves_the_basis_check_inactive():
    """A price the feed says is per unit has no par basis to be wrong about."""
    md = market(Quote("B", 0.985, currency="USD", price_multiplier=1.0))
    _, notes = value_book(
        "FUND",
        [Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0)],
        md,
    )
    assert notes.par_quoted == 0
    assert notes.par_basis_suspects == ()


# --- exchange rates -------------------------------------------------------------------


def test_a_rate_is_used_directly_when_held_that_way_round():
    md = market(rates={("USD", "GBP"): 1.26})
    assert md.rate_to_numeraire("GBP") == pytest.approx(1.26)


def test_a_rate_is_inverted_when_only_the_reverse_is_held():
    md = market(rates={("GBP", "USD"): 1.0 / 1.26})
    assert md.rate_to_numeraire("GBP") == pytest.approx(1.26)


def test_the_numeraire_against_itself_is_one_without_a_rate():
    assert market().rate_to_numeraire("USD") == 1.0
    assert market().rate_to_numeraire("") == 1.0


def test_a_missing_rate_raises_rather_than_defaulting_to_one():
    """Defaulting to 1.0 would value a foreign holding at its local price."""
    with pytest.raises(MissingRate, match="either direction"):
        market().rate_to_numeraire("JPY")


def test_a_zero_reverse_rate_is_not_inverted():
    """RAT guards this explicitly rather than dividing by zero."""
    md = market(rates={("GBP", "USD"): 0.0})
    with pytest.raises(MissingRate, match="is zero"):
        md.rate_to_numeraire("GBP")


def test_fx_is_applied_to_a_foreign_quote():
    md = market(Quote("BP LN", 4.80, currency="GBP"), rates={("USD", "GBP"): 1.26})
    p = valued(Holding("BP LN", AssetClass.EQUITY, quantity=20_000.0), md)
    assert p.value(ValueBasis.MARKET, Accrual.DIRTY) == pytest.approx(120_960.0)


def test_the_quotes_currency_is_used_not_the_holdings():
    """RAT converts with priceRecord.currency, which matters for a foreign listing."""
    md = market(Quote("X", 100.0, currency="GBP"), rates={("USD", "GBP"): 1.26})
    p = valued(Holding("X", AssetClass.EQUITY, quantity=1.0), md)
    assert p.fx_rate == pytest.approx(1.26)


# --- price sources --------------------------------------------------------------------


def test_the_default_source_answers_when_no_sources_are_named():
    md = market(Quote("AAPL", 190.0, currency="USD"))
    assert md.quote("AAPL").source == DEFAULT_SOURCE


def test_sources_are_tried_in_order():
    md = market(
        Quote("AAPL", 1.0, currency="USD", source="SECONDARY"),
        Quote("AAPL", 2.0, currency="USD", source="PRIMARY"),
    )
    assert md.quote("AAPL", ("PRIMARY", "SECONDARY")).price == 2.0
    assert md.quote("AAPL", ("SECONDARY", "PRIMARY")).price == 1.0


def test_the_default_source_is_the_last_resort():
    md = market(
        Quote("AAPL", 190.0, currency="USD"),
        Quote("MSFT", 410.0, currency="USD", source="PRIMARY"),
    )
    assert md.quote("AAPL", ("PRIMARY",)).source == DEFAULT_SOURCE


def test_a_fallback_source_is_reported():
    """Two funds priced from different providers show different risk, so the number says so."""
    md = market(Quote("AAPL", 190.0, currency="USD", source="SECONDARY"))
    _, notes = value_book(
        "FUND",
        [Holding("AAPL", AssetClass.EQUITY, quantity=1.0,
                 price_sources=("PRIMARY", "SECONDARY"))],
        md,
    )
    assert notes.fallback_sources == {"AAPL": "SECONDARY"}
    assert any("fallback provider" in q for q in notes.qualifications())


def test_the_primary_source_answering_is_not_a_fallback():
    md = market(Quote("AAPL", 190.0, currency="USD", source="PRIMARY"))
    _, notes = value_book(
        "FUND",
        [Holding("AAPL", AssetClass.EQUITY, quantity=1.0,
                 price_sources=("PRIMARY", "SECONDARY"))],
        md,
    )
    assert notes.fallback_sources == {}


def test_two_quotes_from_one_source_for_one_security_are_refused():
    with pytest.raises(ValueError, match="two quotes"):
        MarketData.of([Quote("AAPL", 1.0), Quote("AAPL", 2.0)])


# --- FILL_MISSING_PRICES --------------------------------------------------------------


def test_a_missing_price_is_zeroed_and_flagged_by_default():
    """RAT's FILL_MISSING_PRICES: the alternative was failing the whole run."""
    book, notes = value_book(
        "FUND", [Holding("NOQUOTE", AssetClass.EQUITY, quantity=100.0)], market()
    )
    assert book.total(ValueBasis.MARKET) == 0.0
    assert notes.unpriced == ("NOQUOTE",)
    assert any("no price from any permitted source" in q for q in notes.qualifications())


def test_a_missing_price_can_be_made_fatal_instead():
    with pytest.raises(MissingPrice, match="NOQUOTE"):
        value_holding(
            Holding("NOQUOTE", AssetClass.EQUITY, quantity=100.0),
            market(),
            fill_missing_prices=False,
        )


def test_an_unpriced_position_is_kept_in_the_book_not_dropped():
    """It is in the position count and in a holdings listing, contributing nothing."""
    book, _ = value_book(
        "FUND",
        [
            Holding("AAPL", AssetClass.EQUITY, quantity=1000.0),
            Holding("NOQUOTE", AssetClass.EQUITY, quantity=100.0),
        ],
        market(Quote("AAPL", 190.0, currency="USD")),
    )
    assert len(book.positions) == 2
    assert "NOQUOTE" in {p.security_code for p in book.positions}
    assert book.notes().unpriced == ("NOQUOTE",)


def test_cash_is_never_counted_as_unpriced():
    _, notes = value_book(
        "FUND", [Holding("CASHUSD", AssetClass.CASH, quantity=1.0)], market()
    )
    assert notes.unpriced == ()
    assert notes.priced == 1


# --- accrued interest records ---------------------------------------------------------


def test_an_absent_accrued_record_makes_dirty_equal_clean():
    md = market(Quote("B", 99.0, currency="USD"))
    p = valued(Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1000.0), md)
    assert p.value(ValueBasis.MARKET, Accrual.DIRTY) == p.value(
        ValueBasis.MARKET, Accrual.CLEAN
    )


def test_a_bond_with_no_accrued_record_is_flagged():
    md = market(Quote("B", 99.0, currency="USD"))
    _, notes = value_book(
        "FUND", [Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1000.0)], md
    )
    assert notes.missing_accrued == ("B",)


def test_classes_that_legitimately_have_no_accrued_are_not_flagged():
    """A future, an equity and cash all have none; flagging them buries the bond that matters."""
    md = market(
        Quote("ESZ6", 100.0, currency="USD"), Quote("AAPL", 190.0, currency="USD")
    )
    _, notes = value_book(
        "FUND",
        [
            Holding("ESZ6", AssetClass.FUTURE, quantity=1.0, par_amount=1.0),
            Holding("AAPL", AssetClass.EQUITY, quantity=1.0),
            Holding("CASHUSD", AssetClass.CASH, quantity=1.0),
        ],
        md,
    )
    assert notes.missing_accrued == ()


def test_a_zero_accrued_record_is_not_the_same_as_an_absent_one():
    md = market(Quote("B", 99.0, accrued_interest=0.0, currency="USD"))
    _, notes = value_book(
        "FUND", [Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1000.0)], md
    )
    assert notes.missing_accrued == (), "a record of zero is a record"


# --- the whole book -------------------------------------------------------------------


def test_a_mixed_book_totals_correctly_on_both_bases():
    md = market(
        Quote("AAPL", 190.0, currency="USD"),
        Quote("T 4.5 30", 98.5, accrued_interest=1.25, currency="USD"),
        Quote("ESZ6", 5500.0 * 50.0 * 100.0, currency="USD"),
    )
    holdings = [
        Holding("AAPL", AssetClass.EQUITY, quantity=1000.0),
        Holding("T 4.5 30", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0),
        Holding("ESZ6", AssetClass.FUTURE, quantity=20.0, par_amount=1.0),
        Holding("CASHUSD", AssetClass.CASH, quantity=50_000.0),
    ]
    book, _ = value_book("FUND", holdings, md)
    assert book.total(ValueBasis.MARKET) == pytest.approx(190_000 + 997_500 + 0 + 50_000)
    assert book.total(ValueBasis.EXPOSURE) == pytest.approx(
        190_000 + 997_500 + 5_500_000 + 50_000
    )
    assert book.gross_exposure > 5.0


def test_a_stated_nav_inserts_the_cash_plug_through_the_loader():
    md = market(Quote("AAPL", 190.0, currency="USD"))
    book, _ = value_book(
        "FUND",
        [Holding("AAPL", AssetClass.EQUITY, quantity=1000.0)],
        md,
        stated_nav=250_000.0,
    )
    assert book.total(ValueBasis.MARKET) == pytest.approx(250_000.0)
    assert book.notes().cash_plug == pytest.approx(60_000.0)


def test_the_accrual_convention_reaches_the_loaded_book():
    md = market(Quote("B", 99.0, accrued_interest=1.0, currency="USD"))
    holdings = [Holding("B", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1000.0)]
    dirty, _ = value_book("F", holdings, md, accrual=Accrual.DIRTY)
    clean, _ = value_book("F", holdings, md, accrual=Accrual.CLEAN)
    assert dirty.total(ValueBasis.MARKET) == pytest.approx(1000.0)
    assert clean.total(ValueBasis.MARKET) == pytest.approx(990.0)


def test_benchmark_quantities_survive_the_load():
    md = market(Quote("AAPL", 190.0, currency="USD"))
    book, _ = value_book(
        "FUND",
        [Holding("AAPL", AssetClass.EQUITY, quantity=1000.0, benchmark_quantity=1100.0)],
        md,
    )
    assert book.benchmark_total(ValueBasis.MARKET) == pytest.approx(209_000.0)


def test_coverage_survives_the_load():
    md = market(Quote("P", 50.0, currency="USD"))
    book, _ = value_book(
        "FUND", [Holding("P", AssetClass.EQUITY, quantity=400.0, covered=False)], md
    )
    assert book.notes().n_uncovered == 1


# --- the par-basis check at ingestion -------------------------------------------------


def bonds(prices, accrued=0.5):
    md = MarketData.of(
        Quote(f"B{i}", p, accrued_interest=accrued, currency="USD")
        for i, p in enumerate(prices)
    )
    holdings = [
        Holding(f"B{i}", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0)
        for i in range(len(prices))
    ]
    return value_book("FUND", holdings, md)


def test_a_whole_feed_on_the_wrong_basis_is_named_as_a_basis_mismatch():
    """The hundredfold error, caught at ingestion rather than in the output.

    RAT's per-100 normalisation lived in the extraction query, not in its code, so nothing in
    the engine stated the basis the `.01` multiplier relied on. A replacement feed delivering
    fractions produces values a hundred times too small with no other symptom.
    """
    _, notes = bonds([0.985, 0.991, 1.002, 0.978])
    assert len(notes.par_basis_suspects) == 4
    assert notes.par_quoted == 4
    note = " ".join(notes.qualifications())
    assert "fractions rather than per 100" in note
    assert "extraction query" in note


def test_one_cheap_bond_among_many_reads_as_distress_not_a_mismatch():
    _, notes = bonds([1.5, 99.0, 101.2, 98.75, 100.0])
    assert notes.par_basis_suspects == ("B0",)
    note = " ".join(notes.qualifications())
    assert "genuine distress" in note
    assert "extraction query" not in note


def test_a_healthy_bond_feed_raises_nothing():
    _, notes = bonds([99.0, 101.2, 98.75])
    assert notes.par_basis_suspects == ()
    assert notes.par_quoted == 3
    assert not any("percent of par priced below" in q for q in notes.qualifications())


def test_a_cheap_equity_is_not_a_basis_suspect():
    """An equity has no par basis, so a low price carries no information about scaling."""
    md = market(Quote("PENNY", 0.42, currency="USD"))
    _, notes = value_book(
        "FUND", [Holding("PENNY", AssetClass.EQUITY, quantity=1000.0)], md
    )
    assert notes.par_basis_suspects == ()
    assert notes.par_quoted == 0


def test_an_unpriced_bond_is_not_a_basis_suspect():
    """It has no price to be on the wrong basis; it is already reported as unpriced."""
    book, notes = value_book(
        "FUND",
        [Holding("NOQUOTE", AssetClass.FIXED_INCOME, quantity=1.0, par_amount=1_000_000.0)],
        market(),
    )
    assert notes.unpriced == ("NOQUOTE",)
    assert notes.par_basis_suspects == ()
    assert notes.par_quoted == 0


def test_futures_are_checked_on_the_same_basis_as_bonds():
    """A future's exposure uses the same .01 * par factor, so the same mismatch applies."""
    md = market(Quote("ESZ6", 0.55, currency="USD"))
    _, notes = value_book(
        "FUND", [Holding("ESZ6", AssetClass.FUTURE, quantity=20.0, par_amount=1.0)], md
    )
    assert notes.par_basis_suspects == ("ESZ6",)
