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
    # A spin-off's child, in no column: trading from day 12.
    "KID": {i: 3.0 + 0.1 * i for i in range(12, 40)},
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


def small_panel(splits=None, samples=None, member_days=None, dividends=None, overrides=(),
                spin_offs=(), bars_of=None):
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
        DAYS, columns, bars_of or bars,
        splits if splits is not None else [two_for_one([(day(16), 100.0)])],
        samples, paid, members, market, overrides, spin_offs)


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


def test_a_sample_its_own_weighted_count_confirms_is_spared_and_still_reported():
    """PCG's case (midoffice09, ruling 1): an isolated sample whose own record's weighted count
    agrees with it within the tolerance is two fields of one record agreeing -- no error. It is
    still found and reported, spared; one the weighted count does not confirm is removed."""
    def column(key):
        return securities.Column(key, (key,), key, (span(key, key, 0, 39),))

    def seq(*pairs):
        return [(day(i), "T", float(c)) for i, c in pairs]

    samples = {k: seq((0, 100), (5, 101), (10, 10), (15, 102), (20, 100))
               for k in ("OWN", "OTHER", "NONE")}
    weighted = {("OWN", day(10)): 10.4, ("OTHER", day(10)): 99.0}
    repairs = securities.isolated_errors([column(k) for k in samples], samples, [],
                                         weighted=weighted)
    by_key = {r.key: r for r in repairs}
    assert set(by_key) == {"OWN", "OTHER", "NONE"}
    assert by_key["OWN"].spared and by_key["OWN"].weighted == 10.4
    assert not by_key["OTHER"].spared and by_key["OTHER"].weighted == 99.0
    assert not by_key["NONE"].spared and by_key["NONE"].weighted is None


def test_a_span_off_an_exchange_ends_but_for_the_days_the_index_still_held_it():
    """Midoffice09's ruling 2: from the first sample whose record's market is not "stocks" the
    span goes; it ends on its last sample on an exchange, or runs on through the member-days
    after it -- trading days only -- so no member-day changes. A span wholly off one with no
    member-day goes; a listed span, or one with no records, is kept as it is."""
    calendar = tuple(d for d in DAYS if d.weekday() < 5)
    spans = [
        span("LEFT", "LFT", 0, 39, 0, 10, 20, 30),     # off from day 20; held to day 24
        span("GONE", "GNE", 5, 39, 5, 15),             # off from its first sample
        span("STAY", "STY", 0, 39, 0, 10, 20),         # listed throughout
        span("BARE", "BAR", 0, 39, 0, 10),             # no records at all
    ]
    markets = {("LFT", day(0)): ("stocks", "XNAS"), ("LFT", day(10)): ("stocks", "XNAS"),
               ("LFT", day(20)): ("otc", "OTC Link"), ("LFT", day(30)): ("otc", "OTC Link"),
               ("GNE", day(5)): ("otc", "OTC Link"), ("GNE", day(15)): ("otc", "OTC Link"),
               ("STY", day(0)): ("stocks", "XNYS"), ("STY", day(10)): ("stocks", "XNYS"),
               ("STY", day(20)): ("stocks", "XNYS")}
    member_days = {"LEFT": [(day(0), day(25))], "GONE": [(day(0), day(4))],
                   "STAY": [(day(0), day(39))]}
    kept, ended = securities.listed_spans(
        spans, lambda ticker, d: markets.get((ticker, d)), member_days, calendar)
    assert [s.key for s in kept] == ["LEFT", "STAY", "BARE"]
    left, gone = ended
    assert (left.span.key, left.off_exchange, left.market, left.exchange) == (
        "LEFT", day(20), "otc", "OTC Link")
    assert left.last_listed == day(10)
    # Days 11 to 25 are the member-days after it; the weekends among them are not.
    assert left.held == tuple(d for d in DAYS[11:26] if d.weekday() < 5)
    assert left.kept_to == max(left.held) and kept[0].last == left.kept_to
    assert kept[0].samples == (day(0), day(10), day(20))
    assert (gone.span.key, gone.last_listed, gone.kept_to, gone.held) == ("GONE", None, None, ())
    assert kept[1] == spans[2] and kept[2] == spans[3]


# --------------------------------------------------------------------- spin-offs (midoffice14)

def spin_off(ex, *, child="KID", ratio=0.5, record=None, split=1.0, child_key=None,
             child_count=None, retained=0.0, held=""):
    return securities.SpinOff("F1", "NEW", "name F1", child, child_key, ratio,
                              record or day(ex - 1), day(ex), day(ex), split, child_count,
                              (), "evidence", "ruled", retained=retained, held=held)


def price_only(on):
    return SplitCheck("F1", "name F1", Split(day(on), 1000.0, 1033.0, "NEW"), (day(10), 100.0),
                      (day(16), 100.0), False, "unchanged", "price-only", False, "price-only")


def test_a_spin_off_ruling_carries_the_childs_shares_in_the_parents_total_return():
    """CTVA's case: on the ex-date the parent's total return is its raw close plus the child's
    shares at the child's close, plus the day's cash, over its raw close the day before; a
    price-only event the cache records there does not enter it; the stitched prices keep their
    own step, and the days after keep theirs."""
    plain = small_panel()
    panel = small_panel(splits=[two_for_one([(day(16), 100.0)]), price_only(12)],
                        spin_offs=[spin_off(12)])
    p, total = panel.prices[:, 0], panel.total_prices[:, 0]
    assert total[12] / total[11] == pytest.approx((52.0 + 1.0 + 0.5 * 4.2) / 51.0)
    assert np.array_equal(p, plain.prices[:, 0], equal_nan=True)
    assert total[13] / total[12] == pytest.approx(p[13] / p[12])
    assert total[11] == plain.total_prices[11, 0]
    (use,) = panel.spin_offs
    assert (use.key, use.previous, use.day, use.child_day, use.splits) == (
        "F1", day(11), day(12), day(12), 1.0)
    assert (use.raw_before, use.raw_after, use.child_close, use.cash) == pytest.approx(
        (51.0, 52.0, 4.2, 1.0))
    assert use.adjusted == pytest.approx((26.0 + 0.5) / 25.5)
    assert use.step == pytest.approx(55.1 / 51.0)
    assert panel.distributions == plain.distributions


def test_a_ruling_on_a_split_day_takes_the_split_as_the_adjusted_closes_do():
    """The ratio is the child's shares an old share: on the two-for-one's day the parent's new
    shares count twice, as the adjusted closes' step has them."""
    panel = small_panel(spin_offs=[spin_off(15)])
    (use,) = panel.spin_offs
    assert use.splits == 2.0 and use.adjusted == pytest.approx(35.0 / 27.0)
    assert use.step == pytest.approx((2.0 * 35.0 + 0.5 * 4.5) / 54.0)


def test_a_rulings_split_is_applied_to_the_count_as_the_gates_are_when_its_samples_follow():
    """HON's case: a split the parent made with its distribution and the cache does not record,
    checked by the parent's own samples, which must follow it, and applied on the ex-date; the
    split guard holds; it enters the day's total return; counts that do not follow it refuse."""
    samples = {"F1": [(day(0), "OLD", 100.0), (day(10), "NEW", 100.0), (day(16), "NEW", 100.0),
                      (day(19), "NEW", 201.0), (day(24), "NEW", 100.0)],
               "P": [(day(0), "RE", 50.0)], "S": [(day(30), "RE", 54.0)],
               "Y": [(day(20), "SPL", 7.0)]}
    ruling = spin_off(20, split=0.5)
    (check,) = securities.spin_off_splits([ruling], samples)
    assert (check.key, check.split.execution_date, check.split.ratio) == ("F1", day(20), 0.5)
    assert (check.kind, check.counts, check.before, check.followed) == (
        "split", "follow", (day(19), 201.0), (day(24), 100.0))
    assert "the cache does not record" in check.detail and check.prices is None
    splits = [two_for_one([(day(16), 100.0)]), check]
    panel = small_panel(splits=splits, samples=samples, spin_offs=[ruling])
    f1 = panel.shares[:, 0]
    assert list(f1[[19, 20, 23, 24]]) == [201.0, 100.5, 100.5, 100.0]
    assert ("F1", day(20), 0.5) in panel.applied
    assert securities.split_guard(panel, splits, samples).passed
    assert panel.total_prices[20, 0] / panel.total_prices[19, 0] == pytest.approx(
        (0.5 * 40.0 + 0.5 * 5.0) / 39.0)
    unchanged = {**samples, "F1": samples["F1"][:-1] + [(day(24), "NEW", 201.0)]}
    with pytest.raises(ValueError, match="not one its counts follow"):
        securities.spin_off_splits([ruling], unchanged)


def test_a_child_without_a_count_takes_its_parents_on_the_record_date_until_its_own():
    """Vylor's case: the parent's carried count on the record date times the ratio, from the
    child's first day on or after the ex-date until a sample of its own carries one -- never
    back -- and the seeding is listed; a child no column holds is refused."""
    samples = {"F1": [(day(0), "OLD", 100.0), (day(10), "NEW", 100.0), (day(16), "NEW", 100.0),
                      (day(19), "NEW", 201.0)],
               "P": [(day(0), "RE", 50.0)], "S": [(day(30), "RE", 54.0)],
               "Y": [(day(30), "SPL", 7.0)]}
    ruling = spin_off(20, child="SPL", child_key="Y", record=day(19), child_count="parent")
    panel = small_panel(samples=samples, spin_offs=[ruling])
    y = panel.shares[:, 2]
    assert not np.isfinite(y[:20]).any()
    assert list(y[[20, 29, 30, 39]]) == [100.5, 100.5, 7.0, 7.0]
    (seed,) = panel.seeded
    assert (seed.key, seed.first, seed.count, seed.parent_count, seed.until) == (
        "Y", day(20), 100.5, 201.0, day(30))
    assert (seed.days, seed.member_days) == (10, 5)
    assert panel.total_prices[20, 0] / panel.total_prices[19, 0] == pytest.approx(
        (40.0 + 0.5 * 70.0) / 39.0)
    with pytest.raises(ValueError, match="as columns"):
        small_panel(samples=samples, spin_offs=[spin_off(20, child="SPL", child_key="NONE",
                                                         child_count="parent")])


def test_the_spin_off_file_is_read_with_its_rules(tmp_path):
    """Each ruling the parent's key, the child, a positive ratio and split, its dates in order,
    a seeded count only with the child's key, and evidence; one ruling a parent and ex-date.
    The repo's file reads: CTVA's with Vylor's count seeded, HON's with its reverse split and the
    count override it supersedes."""
    import json

    def write(*entries):
        path = tmp_path / "spin_offs.json"
        path.write_text(json.dumps({"entries": list(entries)}), encoding="utf-8")
        return path

    good = {"parent": "K", "label": "KK", "name": "the parent", "child": "C", "child_key": "F",
            "ratio": 1.0, "record": "2026-09-24", "distributed": "2026-10-01",
            "ex_date": "2026-10-01", "child_count": "parent", "evidence": "the record",
            "ruled": "midoffice14"}
    (read,) = securities.read_spin_offs(write(good))
    assert (read.parent, read.child, read.ratio, read.split, read.ex_date, read.child_count) == (
        "K", "C", 1.0, 1.0, dt.date(2026, 10, 1), "parent")
    for bad, why in (({**good, "ratio": 0}, "positive"),
                     ({**good, "child_count": "half"}, "not 'half'"),
                     ({**good, "child_key": None}, "needs the child's key"),
                     ({**good, "ex_date": "2026-09-01"}, "backwards"),
                     ({**good, "evidence": ""}, "needs its evidence")):
        with pytest.raises(ValueError, match=why):
            securities.read_spin_offs(write(bad))
    with pytest.raises(ValueError, match="two rulings"):
        securities.read_spin_offs(write(good, good))
    path = Path(__file__).resolve().parents[2] / "config" / "sp500_spin_off_overrides.json"
    rulings = {(r.label, r.ex_date): r for r in securities.read_spin_offs(path)}
    ctva = rulings[("CTVA", dt.date(2026, 10, 1))]
    assert (ctva.child, ctva.ratio, ctva.record, ctva.child_count) == (
        "VYLR", 1.0, dt.date(2026, 9, 24), "parent")
    hon = rulings[("HON", dt.date(2026, 6, 29))]
    assert (hon.child, hon.ratio, hon.split) == ("HONA", 0.5, 0.5)
    assert hon.supersedes == (("BBG000H556T9", dt.date(2026, 10, 12), dt.date(2026, 11, 3)),)
    assert all(r.evidence and r.ruled for r in rulings.values())


def test_the_first_day_check_lists_the_members_gaps_and_what_explains_them():
    """On a security's first day in the panel, a member whose raw price opened below the
    market's by more than the threshold is listed, and marked by a ruling, a split event or a
    distribution that day where one explains it."""
    import dataclasses

    def gapped(ticker, basis):
        rows = bars(ticker, basis)
        if ticker != "NEW" or basis != "raw":
            return rows
        return tuple(dataclasses.replace(b, open=30.0) if b.date == day(20) else b for b in rows)

    splits = [two_for_one([(day(16), 100.0)])]
    panel = small_panel(bars_of=gapped)
    (first,) = securities.first_days(panel, bars("SPY", "raw"), splits, {})
    expected = 30.0 / 39.0 - 1.0 - (420.0 / 419.0 - 1.0)
    assert (first.key, first.label, first.day) == ("Y", "SPL", day(20))
    ((key, label, gap, why),) = first.gaps
    assert (key, label, why) == ("F1", "NEW", "") and gap == pytest.approx(expected)
    (ruled,) = securities.first_days(panel, bars("SPY", "raw"), splits, {}, [spin_off(20)])
    assert ruled.gaps[0][3] == "ruling"
    on_day = SplitCheck("F1", "name F1", Split(day(20), 1.0, 2.0, "NEW"), None, None, True,
                        None, "split", False, "a split")
    (split,) = securities.first_days(panel, bars("SPY", "raw"), splits + [on_day], {})
    assert split.gaps[0][3] == "split"
    cash = {"NEW": [Dividend(day(20), 0.5, "USD", "SC", 0, None, "NEW")]}
    (paid,) = securities.first_days(panel, bars("SPY", "raw"), splits, cash)
    assert paid.gaps[0][3] == "distribution"


# ------------------------------------------------- midoffice14's rulings, revised (midoffice15)


def test_a_seeded_count_is_grossed_up_by_the_share_the_parent_kept():
    """FedEx Freight's case (midoffice15's ruling 3): the parent kept part of the child, so the
    shares it distributed are the rest of the child's count -- the seeded count is the parent's
    on the record date times the ratio over 1 - retained, to the nearest share, the child's total
    outstanding as every other count is; the day's total return is the ruling's as before."""
    samples = {"F1": [(day(0), "OLD", 100.0), (day(10), "NEW", 100.0), (day(16), "NEW", 100.0),
                      (day(19), "NEW", 201.0)],
               "P": [(day(0), "RE", 50.0)], "S": [(day(30), "RE", 54.0)],
               "Y": [(day(30), "SPL", 7.0)]}
    whole = spin_off(20, child="SPL", child_key="Y", record=day(19), child_count="parent")
    kept = spin_off(20, child="SPL", child_key="Y", record=day(19), child_count="parent",
                    retained=0.2)
    plain = small_panel(samples=samples, spin_offs=[whole])
    panel = small_panel(samples=samples, spin_offs=[kept])
    y = panel.shares[:, 2]
    # 201 x 0.5 = 100.5 distributed, over the 0.8 not kept: 125.625, to the share 126.
    assert list(y[[20, 29, 30, 39]]) == [126.0, 126.0, 7.0, 7.0]
    assert not np.isfinite(y[:20]).any()
    (seed,) = panel.seeded
    assert (seed.count, seed.parent_count, seed.first, seed.until) == (
        126.0, 201.0, day(20), day(30))
    assert np.array_equal(panel.total_prices, plain.total_prices, equal_nan=True)
    assert np.array_equal(panel.shares[:, :2], plain.shares[:, :2], equal_nan=True)


def test_a_held_ruling_is_read_with_its_reason_and_never_applied(tmp_path):
    """S&P Global's case (midoffice15's ruling 1): a ruling held until it is understood is read
    with why, and the build and the split check refuse to apply it; the reader's other new rule,
    a retained share at least 0 and under 1 and only on a seeded count. The repo's file reads:
    SPGI's ruling held, FDXF's count grossed up by FedEx's 19.9%, CTVA's and HON's as before."""
    import json

    def write(*entries):
        path = tmp_path / "spin_offs.json"
        path.write_text(json.dumps({"entries": list(entries)}), encoding="utf-8")
        return path

    good = {"parent": "K", "label": "KK", "name": "the parent", "child": "C", "child_key": "F",
            "ratio": 1.0, "record": "2026-09-24", "distributed": "2026-10-01",
            "ex_date": "2026-10-01", "child_count": "parent", "retained": 0.199,
            "evidence": "the record", "ruled": "midoffice15"}
    (read,) = securities.read_spin_offs(write(good))
    assert (read.retained, read.held) == (0.199, "")
    (held,) = securities.read_spin_offs(write({**good, "held": "no gap on either day"}))
    assert held.held == "no gap on either day"
    for bad, why in (({**good, "retained": 1.0}, "under 1"),
                     ({**good, "retained": -0.1}, "under 1"),
                     ({**good, "child_count": None}, "seeds none"),
                     ({**good, "held": True}, "in words")):
        with pytest.raises(ValueError, match=why):
            securities.read_spin_offs(write(bad))
    with pytest.raises(ValueError, match="held spin-off ruling is not applied"):
        small_panel(spin_offs=[spin_off(12, held="not understood")])
    with pytest.raises(ValueError, match="held spin-off ruling is not applied"):
        securities.spin_off_splits([spin_off(20, split=0.5, held="not understood")], {})
    path = Path(__file__).resolve().parents[2] / "config" / "sp500_spin_off_overrides.json"
    rulings = {(r.label, r.ex_date): r for r in securities.read_spin_offs(path)}
    spgi = rulings[("SPGI", dt.date(2026, 7, 1))]
    assert spgi.held and "no gap" in spgi.held
    fdx = rulings[("FDX", dt.date(2026, 6, 1))]
    assert (fdx.child, fdx.child_count, fdx.retained) == ("FDXF", "parent", 0.199)
    assert [r.label for r in rulings.values() if r.held] == ["SPGI"]
    assert [r.label for r in rulings.values() if r.retained] == ["FDX"]
    assert rulings[("CTVA", dt.date(2026, 10, 1))].retained == 0.0
