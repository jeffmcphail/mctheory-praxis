"""Tests for the panel on securities: the first-count search, prices stitched across a change
of ticker, a reorganisation one column, total returns on the prices' split basis, share counts
carried with splits applied and stale samples read past, and the guards on what was built."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from praxis.midoffice.refdata import securities
from praxis.midoffice.refdata.identity import Record
from praxis.midoffice.refdata.polygon import Bar, Dividend, Split
from praxis.midoffice.refdata.rebuild import Span, SplitCheck

DAYS = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40))


def day(i: int) -> dt.date:
    return DAYS[i]


def span(key, ticker, lo, hi, *samples, member_days=1, previous=None):
    return Span(key, f"name {key}", ticker, day(lo), day(hi), member_days,
                tuple(day(i) for i in samples), previous)


def test_the_first_count_is_found_by_bisection_and_carries_from_its_own_date():
    """A spin-off's first records carry no count; its second sample does. Bisection between the
    two finds the first record with one in a handful of lookups -- the counts appearing on day 7
    and staying -- and a span with no count on any sample has nothing to search."""
    first = span("F", "SPUN", 0, 30, 0, 20, 30)
    asked = []

    def lookup(ticker, on):
        asked.append(on)
        i = DAYS.index(on)
        return Record(ticker, on, "F", "0001", "Spun", "CS"), (500.0 if i >= 7 else None)

    found = securities.search_first_count(first, list(DAYS[:21]), (day(20), 510.0), lookup,
                                          lambda r: r.key == "F")
    assert found.found == (day(7), 500.0) and found.bracket == (day(0), day(20))
    assert len(found.probes) == len(asked) <= 5 and day(7) in asked
    assert "the first record with a count is 2024-01-08's" in found.detail
    none = securities.search_first_count(first, list(DAYS[:21]), None, lookup, lambda r: True)
    assert none.found is None and none.bracket is None and "nothing to search" in none.detail


def test_the_search_stops_on_a_record_naming_another_security():
    first = span("F", "SPUN", 0, 30, 0, 20)

    def lookup(ticker, on):
        return Record(ticker, on, "OTHER", "0009", "Other", "CS"), None

    found = securities.search_first_count(first, list(DAYS[:21]), (day(20), 5.0), lookup,
                                          lambda r: r.key == "F")
    assert found.found is None and "names Other (OTHER), not name F (F)" in found.failure


# A small panel of three securities over 40 days, each rule's case in it:
#  - F1 traded as OLD to day 9 and NEW from day 10, a two-for-one under NEW on day 15, so NEW's
#    adjusted closes before then are half its raw ones and OLD's are its raw ones;
#  - a reorganisation, P under RE to day 19 and S under RE from day 20, one column under S;
#  - a splice: SPL is another security's to day 9, Y's from day 20, and only Y is a member.

RAW = {
    "OLD": {i: 10.0 + i for i in range(0, 10)},
    "NEW": {i: (40.0 + i if i < 15 else 20.0 + i) for i in range(10, 25)},
    "RE": {i: 5.0 + 0.1 * i for i in range(0, 40)},
    "SPL": {i: 50.0 + i for i in list(range(0, 10)) + list(range(20, 40))},
    "SPY": {i: 400.0 + i for i in range(0, 40)},
}
FACTOR = {"NEW": lambda i: 0.5 if i < 15 else 1.0}


def bars(ticker, basis):
    out = []
    for i, close in sorted(RAW[ticker].items()):
        scale = FACTOR.get(ticker, lambda i: 1.0)(i) if basis == "adj" else 1.0
        volume = 1000.0 / scale
        out.append(Bar(day(i), close * scale, close * scale, close * scale, close * scale,
                       volume, basis == "adj"))
    return tuple(out)


def two_for_one(stale=()):
    return SplitCheck("F1", "name F1", Split(day(15), 1.0, 2.0, "NEW"), (day(10), 100.0),
                      (day(16), 100.0), True, "follow", "split", False, "a split",
                      tuple(stale), (day(19), 201.0))


def small_panel(splits=None, samples=None, member_days=None, dividends=None, overrides=()):
    spans = [span("F1", "OLD", 0, 9, 0, 9), span("F1", "NEW", 10, 24, 10, 16, 19, 24),
             span("P", "RE", 0, 19, 0, 19), span("S", "RE", 20, 39, 20, 30, 39),
             span("Y", "SPL", 20, 39, 20, 39)]
    columns = securities.columns_of(spans, {"P": "S"})
    samples = samples if samples is not None else {
        "F1": [(day(0), "OLD", 100.0), (day(10), "NEW", 100.0), (day(16), "NEW", 100.0),
               (day(19), "NEW", 201.0)],
        "P": [(day(0), "RE", 50.0)],
        "S": [(day(30), "RE", 54.0)],
        "Y": [(day(20), "SPL", 7.0)],
    }
    paid = dividends if dividends is not None else {"NEW": [
        Dividend(day(12), 1.0, "USD", "CD", 4, None, "NEW"),
        Dividend(day(12), 1.0, "USD", "CD", 4, None, "NEW"),
        Dividend(day(10), 0.5, "USD", "CD", 4, None, "NEW"),
    ]}
    market = (bars("SPY", "adj"), bars("SPY", "raw"),
              [Dividend(day(5), 2.0, "USD", "CD", 4, None, "SPY")])
    members = member_days if member_days is not None else {
        "F1": [(day(5), day(25))], "P": [(day(0), day(19))], "S": [(day(20), day(35))],
        "Y": [(day(25), day(39))]}
    return securities.build_panel(
        DAYS, columns, bars, splits if splits is not None else [two_for_one([(day(16), 100.0)])],
        samples, paid, members, market, overrides)


def test_the_columns_merge_a_reorganisation_under_its_successor_and_keep_a_splice_apart():
    panel = small_panel()
    assert [(c.label, c.key, c.keys) for c in panel.columns] == [
        ("NEW", "F1", ("F1",)), ("RE", "S", ("P", "S")), ("SPL", "Y", ("Y",))]
    spl = 2
    assert not np.isfinite(panel.prices[:10, spl]).any()
    assert panel.prices[20, spl] == RAW["SPL"][20]


def test_prices_are_stitched_across_a_change_of_ticker_by_the_raw_closes():
    """OLD's adjusted closes and NEW's are on different bases: NEW's split halves its history.
    The return across the change is the raw closes', and OLD's run is scaled to continue
    NEW's, whose own adjusted closes the column keeps exactly."""
    panel = small_panel()
    p = panel.prices[:, 0]
    (bridge,) = panel.bridges
    assert (bridge.before, bridge.after, bridge.ratio) == ("OLD", "NEW", 1.0)
    assert bridge.scale == pytest.approx(0.5)
    assert p[10] / p[9] == pytest.approx(RAW["NEW"][10] / RAW["OLD"][9])
    assert p[5] / p[4] == pytest.approx(RAW["OLD"][5] / RAW["OLD"][4])
    assert p[15] / p[14] == pytest.approx((RAW["NEW"][15] * 2) / RAW["NEW"][14])
    assert p[20] == RAW["NEW"][20] and p[12] == RAW["NEW"][12] * 0.5
    assert panel.raw_prices[12, 0] == RAW["NEW"][12] and panel.volumes[12, 0] == 1000.0
    assert panel.adjusted_volumes[12, 0] == pytest.approx(2000.0)
    assert panel.adjusted_volumes[3, 0] == pytest.approx(1000.0 / 0.5)
    assert list(panel.tickers[8:12, 0]) == ["OLD", "OLD", "NEW", "NEW"]


def test_total_return_reinvests_each_distribution_on_the_prices_basis():
    """A dollar on day 12, when NEW's adjusted close is half its raw one, is half a dollar on
    the prices' basis; TEL's case, the same record twice, counts once; one on the column's
    first priced day under NEW is reinvested there, being no column's first price."""
    panel = small_panel()
    p, total = panel.prices[:, 0], panel.total_prices[:, 0]
    assert total[12] / total[11] == pytest.approx((p[12] + 1.0 * 0.5) / p[11])
    assert total[10] / total[9] == pytest.approx((p[10] + 0.5 * 0.5) / p[9])
    assert total[13] / total[12] == pytest.approx(p[13] / p[12])
    assert len(panel.duplicates) == 1 and panel.distributions == 2
    assert panel.market_total_returns[5] == pytest.approx((405.0 + 2.0) / 404.0 - 1.0)
    assert panel.market_returns[5] == pytest.approx(405.0 / 404.0 - 1.0)
    assert panel.market_total_returns[6] == panel.market_returns[6]


def test_counts_carry_forward_with_the_split_applied_and_the_stale_sample_read_past():
    """F1 counts 100 from day 0; the two-for-one makes it 200 on day 15; day 16's sample, still
    100, is stale and read past; day 19's 201 resets it. S carries P's 50 across the
    reorganisation until its own first count, 54 on day 30. Y's count starts at its first."""
    panel = small_panel()
    f1 = panel.shares[:, 0]
    assert list(f1[[0, 14, 15, 16, 18, 19, 24]]) == [100.0, 100.0, 200.0, 200.0, 200.0, 201.0,
                                                     201.0]
    assert not np.isfinite(f1[25:]).any()
    assert panel.applied == [("F1", day(15), 2.0)]
    assert panel.stale_read_past == [("F1", day(16), 100.0)]
    re = panel.shares[:, 1]
    assert re[19] == 50.0 and re[25] == 50.0 and re[30] == 54.0
    (carry,) = [c for c in panel.carries if c.key == "S"]
    assert carry.reorganisation and carry.first == day(20) and carry.until == day(30)
    assert panel.shares[20, 2] == 7.0 and not np.isfinite(panel.shares[19, 2])


def test_the_split_guard_is_exact():
    """The panel as built passes. Had the stale sample reset the count, the days after the
    split would carry a pre-split count; had the split not been applied, the execution date's
    count would not be the count before it times the ratio: both fail."""
    samples = {"F1": [(day(0), "OLD", 100.0), (day(10), "NEW", 100.0),
                      (day(16), "NEW", 100.0), (day(19), "NEW", 201.0)]}
    good = small_panel()
    assert securities.split_guard(good, [two_for_one([(day(16), 100.0)])], samples).passed
    unread = small_panel(splits=[two_for_one()])
    guard = securities.split_guard(unread, [two_for_one([(day(16), 100.0)])], samples)
    assert not guard.passed and "carries 100, a count from before its split" in guard.failures[0]
    unapplied = small_panel(splits=[])
    guard = securities.split_guard(unapplied, [two_for_one([(day(16), 100.0)])], samples)
    assert any("not 100 times 2" in f for f in guard.failures)


def test_the_identity_guards_hold_on_the_built_panel():
    """Each (ticker, date) feeds one column; no column takes two tickers on one date; every
    member-day is inside its column's spans, but a day the table maps after the last trade;
    every bar of a span is in its column and nothing else is."""
    panel = small_panel()
    guards = securities.identity_guards(panel, bars, {("F1", day(25))})
    assert all(g.passed for g in guards), [g for g in guards if not g.passed]
    strict = securities.identity_guards(panel, bars)
    (failed,) = [g for g in strict if not g.passed]
    assert failed.failures == (f"NEW {day(25)}",)


def test_the_weighted_check_and_the_count_moves_flag_what_the_samples_cannot_show():
    """Weighted counts are a check, never a source: met against the count the panel carries,
    S's carried 50 against a weighted 70 is flagged, F1's own 201 against 200 is not. A move
    no split explains is flagged: F1's from 100 to 200 is the split's and is not."""
    panel = small_panel()
    met = securities.weighted_check(panel, {"S": [(day(25), "RE", 70.0, None)],
                                            "F1": [(day(19), "NEW", 200.0, 201.0)]})
    by_key = {w.key: w for w in met}
    assert by_key["S"].flagged() and not by_key["S"].own and by_key["S"].ratio == 1.4
    assert not by_key["F1"].flagged() and by_key["F1"].own
    moves = securities.count_moves(panel, [two_for_one([(day(16), 100.0)])])
    assert [(m.label, m.day) for m in moves] == []
    no_split = securities.count_moves(panel, [])
    assert [(m.label, m.day, m.ratio) for m in no_split] == [("NEW", day(15), 2.0)]


def test_a_first_count_the_next_sample_does_not_confirm_is_passed_over():
    """GE Vernova's case: its first records carry GE's count, four times its own, until Polygon
    corrects them. The next sample does not confirm that first count, so the search goes on to
    the first record it does confirm, and the days before that one have no count. A first count
    the next sample confirms within 10% stands; confirm=False keeps the first, midoffice08's
    rule."""
    first = span("F", "SPUN", 0, 30, 0, 20, 30)

    def records(early, later):
        def lookup(ticker, on):
            i = DAYS.index(on)
            count = None if i < 3 else early if i < 9 else later
            return Record(ticker, on, "F", "0001", "Spun", "CS"), count
        return lookup

    def search(lookup, **kw):
        return securities.search_first_count(first, list(DAYS[:21]), (day(20), 1000.0), lookup,
                                             lambda r: r.key == "F", **kw)

    passed = search(records(4000.0, 1010.0))
    assert passed.unconfirmed == (day(3), 4000.0) and passed.found == (day(9), 1010.0)
    assert passed.next_count == 1000.0
    assert f"unconfirmed; the first the next sample confirms is {day(9)}'s" in passed.detail
    kept = search(records(4000.0, 1010.0), confirm=False)
    assert kept.found == (day(3), 4000.0) and kept.unconfirmed is None
    near = search(records(920.0, 1010.0))
    assert near.found == (day(3), 920.0) and near.unconfirmed is None and "confirmed" in near.detail


def test_an_isolated_bad_sample_is_removed_and_alternation_steps_and_splits_are_left():
    """DGX's case: one sample a tenth of the four either side, which agree with each other --
    not carried. Two wrong samples in a row, samples alternating between two counts, a real
    step the later samples share, a sample without two on each side, and one with a split
    between its neighbours: none of them is removed, the rule unable to tell which count is
    true, or the split's to explain."""
    def column(key):
        return securities.Column(key, (key,), key, (span(key, key, 0, 39),))

    def seq(*pairs):
        return [(day(i), "T", float(c)) for i, c in pairs]

    samples = {
        "ERR": seq((0, 100), (5, 101), (10, 10), (15, 102), (20, 100)),
        "TWO": seq((0, 100), (5, 101), (10, 10), (15, 11), (20, 100), (25, 100)),
        "ALT": seq((0, 100), (5, 150), (10, 100), (15, 150), (20, 100)),
        "STEP": seq((0, 100), (5, 100), (10, 150), (15, 150), (20, 150)),
        "SPLIT": seq((0, 100), (5, 100), (10, 50), (15, 100), (20, 100)),
        "EDGE": seq((0, 10), (5, 100), (10, 100), (15, 100), (20, 100)),
    }
    split = SplitCheck("SPLIT", "SPLIT", Split(day(12), 1.0, 2.0, "SPLIT"), None, None, True,
                       None, "split", False, "a split")
    columns = [column(k) for k in samples]
    (repair,) = securities.isolated_errors(columns, samples, [split])
    assert (repair.key, repair.day, repair.count) == ("ERR", day(10), 10.0)
    assert repair.neighbours == ((day(0), 100.0), (day(5), 101.0), (day(15), 102.0),
                                 (day(20), 100.0))
    unsplit = securities.isolated_errors(columns, samples, [])
    assert {(r.key, r.day) for r in unsplit} == {("ERR", day(10)), ("SPLIT", day(10))}


def test_an_override_carries_its_count_or_none_and_reads_no_sample_inside():
    """HON's case and SKYD's: a ruling carries a count over its range, or leaves the range with
    none, and the samples inside it are not read; the split before it still applies, a count
    carried into a span is closed where the ruling starts, and each override used is listed
    with its days, member-days and the samples it did not read."""
    carry = securities.Override("F1", "NEW", day(16), day(20), "carry", 150.0, "evidence", "r")
    none = securities.Override("S", "RE", day(25), day(39), "none", None, "evidence", "r")
    panel = small_panel(overrides=[carry, none])
    f1, re = panel.shares[:, 0], panel.shares[:, 1]
    assert list(f1[[15, 16, 19, 20, 21, 24]]) == [200.0, 150.0, 150.0, 150.0, 150.0, 150.0]
    assert list(re[[19, 24]]) == [50.0, 50.0] and not np.isfinite(re[25:]).any()
    (into,) = [c for c in panel.carries if c.key == "S"]
    assert into.first == day(20) and into.until == day(25)
    assert [(o.key, days, member, unread) for o, days, member, unread in panel.overridden] == [
        ("F1", 5, 5, 2), ("S", 15, 11, 1)]


def test_the_overrides_file_is_read_with_its_rules(tmp_path):
    """Each entry a key, a range, a carry with its count or none without one, and evidence; a
    carry without a count, a none with one, a range backwards, no evidence, or two entries
    overlapping for one security are refused."""
    import json

    def write(*entries):
        path = tmp_path / "overrides.json"
        path.write_text(json.dumps({"entries": list(entries)}), encoding="utf-8")
        return path

    good = {"key": "K", "label": "KK", "first": "2026-10-12", "last": "2026-11-03",
            "action": "carry", "count": 635675701, "evidence": "a contaminated record",
            "ruled": "midoffice08"}
    (read,) = securities.read_overrides(write(good))
    assert (read.key, read.label, read.first, read.action, read.count) == (
        "K", "KK", dt.date(2026, 10, 12), "carry", 635675701.0)
    for bad, why in (({**good, "count": None}, "needs a count"),
                     ({**good, "action": "none"}, "only a carry has one"),
                     ({**good, "action": "halve"}, "not 'halve'"),
                     ({**good, "last": "2026-10-01"}, "ends before it starts"),
                     ({**good, "evidence": ""}, "needs its evidence")):
        with pytest.raises(ValueError, match=why):
            securities.read_overrides(write(bad))
    with pytest.raises(ValueError, match="overlap"):
        securities.read_overrides(
            write(good, {**good, "first": "2026-10-20", "last": "2026-12-01"}))
    path = Path(__file__).resolve().parents[2] / "config" / "sp500_share_count_overrides.json"
    assert {o.label for o in securities.read_overrides(path)} == {"HON", "SKYD"}
