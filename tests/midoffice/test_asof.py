"""Tests for as-of date resolution.

Two properties carry most of the weight. First, that a substitution is *recorded*: RAT returns
a bare `yyyyMMdd` string from `getLatestDate`, so a model six months stale and one from
yesterday are indistinguishable to everything downstream. Second, that ordering is enforced in
code rather than assumed: RAT's `Tn` family indexes `DataTable.Select()` with no sort argument,
so `T0` is the latest date only if a query in a configuration file sorted descending.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from praxis.midoffice.asof import (
    CALENDARS,
    EXPOSURE,
    LATEST,
    MARKET,
    MODEL,
    SECURITY,
    AsOf,
    DateCalendar,
    EmptyCalendar,
    NoAvailableDate,
    ResolvedDate,
    parse_request,
)

MONTH_ENDS = ["20150331", "20150430", "20150529", "20150630"]


def calendar(name=MODEL, dates=None) -> DateCalendar:
    return DateCalendar.of(name, dates or MONTH_ENDS)


def four_calendars(**overrides) -> dict[str, DateCalendar]:
    base = {name: calendar(name) for name in CALENDARS}
    base.update(overrides)
    return base


# --- ordering is enforced, not assumed ------------------------------------------------


@pytest.mark.parametrize(
    "order",
    [
        ["20150331", "20150430", "20150529", "20150630"],  # ascending
        ["20150630", "20150529", "20150430", "20150331"],  # descending
        ["20150430", "20150630", "20150331", "20150529"],  # arbitrary
    ],
)
def test_the_calendar_sorts_whatever_order_it_is_given(order):
    """The property RAT leaves to a config file's ORDER BY.

    `getLatestDate(tableName, int t)` indexes `Select()` with no sort argument, and the query
    that fills the table comes from `TABLE_MODEL_DATES_QUERY`. Drop the ORDER BY there and T0
    becomes the oldest date, with nothing in the code to notice.
    """
    cal = DateCalendar.of(MODEL, order)
    assert cal.dates == (
        date(2015, 6, 30), date(2015, 5, 29), date(2015, 4, 30), date(2015, 3, 31)
    )
    assert cal.resolve("T0").actual == date(2015, 6, 30)
    assert cal.latest == date(2015, 6, 30)
    assert cal.earliest == date(2015, 3, 31)


def test_duplicate_dates_are_refused():
    """A duplicate makes every Tn offset past it ambiguous."""
    with pytest.raises(ValueError, match="duplicate date"):
        DateCalendar.of(MODEL, MONTH_ENDS + ["20150430"])


def test_an_empty_calendar_refuses_to_answer():
    empty = DateCalendar(MODEL, ())
    for call in (lambda: empty.resolve("T0"), lambda: empty.latest, lambda: empty.earliest):
        with pytest.raises(EmptyCalendar, match="no dates"):
            call()


def test_a_calendar_cannot_hold_a_request():
    with pytest.raises(ValueError, match="literal dates"):
        DateCalendar.of(MODEL, ["20150630", "T1"])


# --- the three request forms ----------------------------------------------------------


def test_latest_and_t0_agree():
    cal = calendar()
    for request in (None, "", LATEST, "T0", "t0"):
        resolved = cal.resolve(request)
        assert resolved.actual == date(2015, 6, 30)
        assert not resolved.substituted, request


def test_tn_counts_back_through_the_calendar():
    cal = calendar()
    assert cal.resolve("T1").actual == date(2015, 5, 29)
    assert cal.resolve("T3").actual == date(2015, 3, 31)
    assert cal.resolve("T2").offset == 2


def test_an_offset_past_the_start_is_a_message_not_an_index_error():
    """RAT checks `rows.Length == 0` and then indexes `rows[t]`."""
    with pytest.raises(NoAvailableDate, match="past its start"):
        calendar().resolve("T4")


def test_an_exact_date_is_not_a_substitution():
    resolved = calendar().resolve("20150529")
    assert resolved.actual == date(2015, 5, 29)
    assert not resolved.substituted
    assert resolved.gap_days == 0


def test_a_date_with_no_data_resolves_backwards_and_records_the_gap():
    """The behaviour RAT has and the record it does not keep."""
    resolved = calendar().resolve("20150615")
    assert resolved.actual == date(2015, 5, 29)
    assert resolved.substituted
    assert resolved.gap_days == 17
    assert resolved.requested == "2015-06-15"
    assert "was used for a request of" in resolved.describe()


def test_a_request_beyond_the_latest_date_resolves_back_to_it():
    resolved = calendar().resolve("20150701")
    assert resolved.actual == date(2015, 6, 30)
    assert resolved.substituted
    assert resolved.gap_days == 1


def test_a_request_before_the_calendar_starts_is_refused():
    """Resolving forwards would silently use data from after the date asked about."""
    with pytest.raises(NoAvailableDate, match="on or before"):
        calendar().resolve("20150101")


def test_staleness_is_distinguished_from_mere_substitution():
    cal = calendar()
    weekend = cal.resolve("20150702")           # 2 days past a month end
    quarter = cal.resolve("20150628")           # 30 days back to 2015-05-29
    assert weekend.substituted and not weekend.is_stale
    assert quarter.substituted and quarter.is_stale


# --- parsing --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, LATEST),
        ("", LATEST),
        ("latest", LATEST),
        ("T0", "T0"),
        ("t7", "T7"),
        ("20150630", date(2015, 6, 30)),
        ("2015-06-30", date(2015, 6, 30)),
        (date(2015, 6, 30), date(2015, 6, 30)),
        (datetime(2015, 6, 30, 17, 30), date(2015, 6, 30)),
    ],
)
def test_accepted_request_forms(value, expected):
    assert parse_request(value) == expected


@pytest.mark.parametrize("value", ["20150631", "20150230", "99999999"])
def test_a_structurally_valid_but_impossible_date_is_named_as_such(value):
    with pytest.raises(ValueError, match="not a real date"):
        parse_request(value)


@pytest.mark.parametrize("value", ["last tuesday", "T", "TX", "30/06/2015", "0630"])
def test_an_unreadable_request_is_refused_rather_than_defaulted(value):
    """RAT hands an unparseable string to a Select filter, where it becomes a malformed
    expression error a long way from the typo that caused it."""
    with pytest.raises(ValueError, match="cannot read"):
        parse_request(value)


# --- the four-date set ----------------------------------------------------------------


def test_all_four_dates_resolve_independently():
    resolved = AsOf.resolve(
        four_calendars(),
        model="20150630",
        exposure="20150615",
        market="T0",
        security=None,
    )
    assert resolved.model.actual == date(2015, 6, 30)
    assert resolved.exposure.actual == date(2015, 5, 29)
    assert resolved.market.actual == date(2015, 6, 30)
    assert resolved.security.actual == date(2015, 6, 30)
    assert [d.calendar for d in resolved] == list(CALENDARS)


def test_a_missing_calendar_is_refused_rather_than_defaulted():
    partial = {MODEL: calendar(MODEL), MARKET: calendar(MARKET)}
    with pytest.raises(KeyError, match="no calendar supplied"):
        AsOf.resolve(partial, model="T0")


def test_an_exact_four_date_run_reports_no_qualifications():
    resolved = AsOf.resolve(four_calendars(), model="20150630", exposure="20150630",
                            market="20150630", security="20150630")
    assert resolved.is_exact
    assert resolved.substitutions() == ()
    assert resolved.qualifications() == ()


def test_substitutions_are_listed_with_the_calendar_that_made_them():
    resolved = AsOf.resolve(four_calendars(), model="20150615", exposure="20150630",
                            market="20150630", security="20150610")
    substituted = {d.calendar for d in resolved.substitutions()}
    assert substituted == {MODEL, SECURITY}
    assert not resolved.is_exact
    notes = " ".join(resolved.qualifications())
    assert MODEL in notes and SECURITY in notes
    assert EXPOSURE not in notes


def test_staleness_is_called_out_separately_in_the_qualifications():
    resolved = AsOf.resolve(four_calendars(), model="20150628", exposure="20150630",
                            market="20150630", security="20150630")
    notes = resolved.qualifications()
    assert any("long weekend" in n for n in notes)


def test_the_key_is_built_from_the_actual_dates_not_the_requested_ones():
    """A cache keyed on the request would serve one run's numbers for another's dates."""
    asked = AsOf.resolve(four_calendars(), model="20150615", exposure="20150615",
                         market="20150615", security="20150615")
    exact = AsOf.resolve(four_calendars(), model="20150529", exposure="20150529",
                         market="20150529", security="20150529")
    assert asked.key() == exact.key() == "20150529|20150529|20150529|20150529"


def test_different_dates_give_different_keys():
    a = AsOf.resolve(four_calendars(), model="20150630", exposure="20150630",
                     market="20150630", security="20150630")
    b = AsOf.resolve(four_calendars(), model="20150529", exposure="20150630",
                     market="20150630", security="20150630")
    assert a.key() != b.key()


def test_describe_names_every_date():
    resolved = AsOf.resolve(four_calendars(), model="T0", exposure="T1", market="T2",
                            security="T3")
    text = resolved.describe()
    for name in CALENDARS:
        assert name in text


def test_calendars_may_have_different_dates_available():
    """The reason the four are resolved separately rather than together.

    A model is published monthly, prices daily. Asking for one date across both gives a month
    end from one and the day itself from the other, and that is correct.
    """
    calendars = four_calendars(
        market=DateCalendar.of(MARKET, ["20150615", "20150616", "20150617"])
    )
    resolved = AsOf.resolve(calendars, model="20150617", exposure="20150617",
                            market="20150617", security="20150617")
    assert resolved.model.actual == date(2015, 5, 29)
    assert resolved.market.actual == date(2015, 6, 17)
    assert resolved.model.substituted
    assert not resolved.market.substituted
