"""Tests for the identity table: who a ticker is on each date it has a price, and which security
each index member is on each date it is one.

Every case is one the S&P 500 panel has, made small: a ticker taken over by another company
with no gap in the prices (DOC), one a holding company reissues (XOM), one a member had only
from its rename (BALL), events that contradict the panel's own prices (GOOGL's), a member
neither source can name (DAY), and one before the start of Polygon's record of it (SW).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from praxis.midoffice.refdata import identity as idn
from praxis.midoffice.refdata import sp500
from praxis.midoffice.refdata.identity import History, Record, Tenure

N = 60
DATES = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(N))


def day(i: int) -> dt.date:
    return DATES[i]


def held(*spans: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(N, dtype=bool)
    for lo, hi in spans:
        mask[lo:hi + 1] = True
    return mask


def rec(ticker: str, i: int, figi: str, kind: str = "CS", name: str | None = None) -> Record:
    return Record(ticker, day(i), figi, f"C{figi}", name or figi, kind)


def hist(figi: str, *steps: tuple[dt.date, str], name: str | None = None) -> History:
    return idn.history_from_events(figi, {
        "composite_figi": figi, "cik": f"C{figi}", "name": name or figi,
        "events": [{"type": "ticker_change", "date": on.isoformat(),
                    "ticker_change": {"ticker": t}} for on, t in steps],
    })


def build(panel, records, histories, extra=(), found_as=None):
    """Segment every ticker of `panel` ({ticker: held mask}) and build the table."""
    by_key = idn.histories_by_key(histories)
    identities = {t: idn.segment_ticker(t, DATES, mask, records.get(t, ()), by_key)
                  for t, mask in panel.items()}
    rows, conflicts, tenures = idn.build_rows(DATES, identities, records, by_key, extra,
                                              found_as)
    return identities, rows, conflicts, tenures


EARLY = dt.date(2020, 1, 1)


def test_ticker_events_are_tenures_and_only_a_change_is_a_known_start():
    """The first event is where Polygon's record of a security starts -- Super Micro's is its
    2024 split -- so only a change from another ticker dates when a ticker began."""
    history = hist("F1", (day(10), "AAA"), (day(10), "AAA"), (day(30), "BBB"), (day(30), "BBB"))
    assert [(t.ticker, t.start, t.end, t.known_start) for t in history.tenures] == [
        ("AAA", day(10), day(29), False),
        ("BBB", day(30), None, True),
    ]
    assert history.key == "F1"
    assert idn.security_key(None, "0000123") == "CIK0000123"
    with pytest.raises(ValueError, match="identified"):
        idn.security_key(None, None, None)


def test_one_security_under_one_ticker_throughout_is_one_row_and_every_cell_its():
    panel = {"AAA": held((0, N - 1))}
    records = {"AAA": (rec("AAA", 0, "F1"), rec("AAA", 30, "F1"), rec("AAA", N - 1, "F1"))}
    identities, rows, conflicts, _ = build(panel, records, {})
    assert [(r.ticker, r.first, r.last, r.held, r.source) for r in rows] == [
        ("AAA", day(0), day(N - 1), N, "panel")
    ]
    assert not identities["AAA"].unassigned and not conflicts
    assert idn.guard_cells(DATES, panel, rows).passed


def test_a_ticker_taken_over_with_no_gap_in_the_prices_is_dated_by_the_newcomers_events():
    """DOC: Physicians Realty until 2024-03-01, Healthpeak -- PEAK until then -- from
    2024-03-04, and the prices run on unbroken. The records bracket the change by a year; the
    newcomer's change from its old ticker dates it to the day. A splice: the newcomer traded
    under another ticker before taking this one."""
    panel = {"DOC": held((0, N - 1))}
    records = {"DOC": (rec("DOC", 0, "OLD"), rec("DOC", 20, "OLD"),
                       rec("DOC", 40, "NEW"), rec("DOC", N - 1, "NEW"))}
    histories = {"DOC": hist("NEW", (EARLY, "PEAK"), (day(30), "DOC"))}
    identities, rows, _, _ = build(panel, records, histories)
    owner = identities["DOC"].owner
    assert set(owner[:30]) == {"OLD"} and set(owner[30:]) == {"NEW"}
    (boundary,) = identities["DOC"].boundaries
    assert boundary.kind == "splice" and boundary.first_after == day(30) and boundary.pinned
    assert [(r.key, r.ticker, r.first, r.last, r.source) for r in rows] == [
        ("NEW", "PEAK", day(0), day(29), "events"),
        ("NEW", "DOC", day(30), day(N - 1), "panel"),
        ("OLD", "DOC", day(0), day(29), "panel"),
    ]
    assert idn.guard_cells(DATES, panel, rows).passed
    assert idn.guard_one_ticker(rows).passed and idn.guard_ticker_overlap(rows).passed


def test_without_events_a_change_is_dated_by_a_gap_in_the_prices_or_left_unassigned():
    """Two records naming two securities and nothing else: a gap in the prices dates the
    change; with no gap nothing does, and the dates between stay unassigned -- reported, and
    the guard fails on them -- rather than guessed."""
    records = {"T": (rec("T", 0, "X"), rec("T", 50, "Y", kind="ETF"))}
    gapped = {"T": held((0, 14), (25, N - 1))}
    identities, rows, _, _ = build(gapped, records, {})
    owner = identities["T"].owner
    assert set(owner[:15]) == {"X"} and set(owner[25:]) == {"Y"}
    assert identities["T"].boundaries[0].kind == "splice"
    assert "a gap in the prices" in identities["T"].boundaries[0].evidence
    assert idn.guard_cells(DATES, gapped, rows).passed
    unbroken = {"T": held((0, N - 1))}
    identities, rows, _, _ = build(unbroken, records, {})
    (left,) = identities["T"].unassigned
    assert (left[0], left[1], left[2]) == (day(1), day(49), 49)
    assert not identities["T"].boundaries[0].pinned
    guard = idn.guard_cells(DATES, unbroken, rows)
    assert not guard.passed and len(guard.failures) == 49


def test_a_reorganisation_is_a_new_identifier_and_the_old_one_is_the_member_before_it():
    """XOM: a holding company's new FIGI from 2026-07-02, the ticker, type and prices running
    on. The index carries the member through it, so before the new identifier's first day the
    member is the old one, under the same ticker."""
    panel = {"XOM": held((0, N - 1))}
    records = {"XOM": (rec("XOM", 0, "OLD"), rec("XOM", 40, "OLD"), rec("XOM", N - 1, "NEW"))}
    histories = {"XOM": hist("NEW", (day(50), "XOM"))}
    identities, rows, _, tenures = build(panel, records, histories)
    (boundary,) = identities["XOM"].boundaries
    assert boundary.kind == "reorganisation" and boundary.first_after == day(50)
    members = [frozenset({"XOM"})] * N
    mapping = idn.map_members(DATES, members, rows, histories, tenures, set(panel))
    assert mapping.mapped[("XOM", day(10))] == ("OLD", "XOM", "predecessor")
    assert mapping.mapped[("XOM", day(55))] == ("NEW", "XOM", "panel")
    assert idn.guard_members(DATES, members, mapping).passed


def test_a_member_the_panel_has_only_from_its_rename_traded_under_its_old_ticker_before():
    """BALL: the panel asked for BALL, which Ball traded under from 2022-05-10; before, it was
    BLL. The member maps to Ball as BLL on those dates, and the panel has no price for it."""
    panel = {"BALL": held((20, N - 1))}
    records = {"BALL": (rec("BALL", 40, "B"), rec("BALL", N - 1, "B"))}
    histories = {"BALL": hist("B", (EARLY, "BLL"), (day(20), "BALL"))}
    _, rows, _, tenures = build(panel, records, histories)
    assert [(r.ticker, r.first, r.last) for r in rows] == [
        ("BLL", day(0), day(19)), ("BALL", day(20), day(N - 1))
    ]
    members = [frozenset({"BALL"})] * N
    mapping = idn.map_members(DATES, members, rows, histories, tenures, set(panel))
    assert mapping.mapped[("BALL", day(5))] == ("B", "BLL", "events")
    cut = idn.truncations(DATES, mapping, set(panel))
    assert cut == {"BALL": {("B", "BLL"): [(day(0), day(19), 20)]}}
    renames = idn.renames_from(tenures, rows, {"B"})
    assert renames == (sp500.Rename(day(20), "BLL", "BALL", since=EARLY),)


def test_events_that_contradict_the_panels_prices_are_cut_and_the_panel_believed():
    """GOOGL's events name it GOOG since 2015; the panel holds its prices under GOOGL with
    records naming it. The panel's evidence is taken, and the disagreement reported. And a
    ticker the security is found by today, where its events end elsewhere -- Casey's, CASYV
    from 2010 -- is presumed, not evidenced."""
    panel = {"GOOGL": held((0, N - 1))}
    records = {"GOOGL": (rec("GOOGL", 0, "A"), rec("GOOGL", N - 1, "A"))}
    histories = {"GOOGL": hist("A", (EARLY, "GOOG")),
                 "CASY": hist("C", (EARLY, "CASY"), (dt.date(2021, 1, 1), "CASYV"))}
    _, rows, conflicts, _ = build(panel, records, histories, extra=("C",),
                                  found_as={"C": "CASY"})
    assert [(r.key, r.ticker, r.source) for r in rows] == [
        ("A", "GOOGL", "panel"), ("C", "CASY", "presumed")
    ]
    notes = " ".join(c.detail for c in conflicts)
    assert "GOOG" in notes and "the panel holds its prices under GOOGL" in notes
    assert "CASYV" in notes and "presumed CASY" in notes


def test_a_member_neither_source_can_name_is_named_never_guessed():
    """DAY: no events and not in the panel -- missing. SW: the panel's prices under it begin
    with its security's first event, so before then the member traded under another ticker or
    was another security -- unverified. Both accounted for, the guard passes, nothing mapped."""
    panel = {"SW": held((30, N - 1))}
    records = {"SW": (rec("SW", 40, "S"), rec("SW", N - 1, "S"))}
    histories = {"SW": hist("S", (day(30), "SW")), "DAY": None}
    identities, rows, _, tenures = build(panel, records, histories)
    members = [frozenset({"SW", "DAY"})] * N
    mapping = idn.map_members(DATES, members, rows, histories, tenures, set(panel))
    assert mapping.named[("DAY", day(0))].startswith("missing")
    assert mapping.named[("SW", day(10))].startswith("unverified")
    assert mapping.mapped[("SW", day(30))] == ("S", "SW", "panel")
    assert not identities["SW"].unassigned
    guard = idn.guard_members(DATES, members, mapping)
    assert guard.passed and guard.checked == 2 * N
    assert not any(k[0] == "DAY" for k in mapping.mapped)


def test_a_security_stopped_before_the_index_removed_it_is_still_the_member():
    """ATVI's last price is 2023-10-12 and the index removed it on 2023-10-18: the member on
    the days between is the same security, after its last price; one halted between two spans
    of its own prices is too."""
    panel = {"AT": held((0, 39)), "SB": held((0, 19), (30, N - 1))}
    records = {"AT": (rec("AT", 0, "A"), rec("AT", 39, "A")),
               "SB": (rec("SB", 0, "B"), rec("SB", N - 1, "B"))}
    _, rows, _, tenures = build(panel, records, {})
    members = [frozenset({"AT", "SB"}) if i < 45 else frozenset({"SB"}) for i in range(N)]
    mapping = idn.map_members(DATES, members, rows, {}, tenures, set(panel))
    assert mapping.mapped[("AT", day(42))] == ("A", "AT", "after its last price")
    assert mapping.mapped[("SB", day(25))] == ("B", "SB", "inside a gap in its prices")
    assert idn.guard_members(DATES, members, mapping).passed


def test_the_guards_catch_a_cell_with_two_owners_and_a_security_under_two_tickers():
    def row(key, ticker, lo, hi, held_days=1):
        return idn.Row(key, key, None, key, "CS", ticker, day(lo), day(hi), held_days, "panel",
                       "", "")

    panel = {"T": held((0, 9))}
    two_owners = [row("X", "T", 0, 9), row("Y", "T", 5, 9)]
    guard = idn.guard_cells(DATES, panel, two_owners)
    assert not guard.passed and len(guard.failures) == 5
    assert not idn.guard_ticker_overlap(two_owners).passed
    two_tickers = [row("X", "T", 0, 9), row("X", "U", 8, 20)]
    assert not idn.guard_one_ticker(two_tickers).passed
    mapping = idn.MemberMap({("T", day(0)): ("X", "T", "panel")},
                            {("T", day(0)): "missing: twice"}, {})
    guard = idn.guard_members(DATES, [frozenset({"T"})] + [frozenset()] * (N - 1), mapping)
    assert not guard.passed and "2 times" in guard.failures[0]


def test_renames_the_table_already_records_or_that_lead_off_todays_list_are_not_read_through():
    """EchoStar's SATS -> ECHO is read through: ECHO is in today's list. WLTW -> WTW is not:
    the table records it as a removal and an addition the same day. Nor is GPS -> GAP: The Gap
    left the index before it became GAP, and its removal names it GPS."""
    renames = (
        sp500.Rename(dt.date(2026, 6, 24), "SATS", "ECHO"),
        sp500.Rename(dt.date(2022, 1, 10), "WLTW", "WTW"),
        sp500.Rename(dt.date(2024, 8, 22), "GPS", "GAP"),
    )
    changes = (sp500.Change(dt.date(2022, 1, 10), added=("WTW",), removed=("WLTW",)),)
    kept, skipped = idn.applicable_renames(renames, {"ECHO", "WTW"}, changes)
    assert kept == (renames[0],)
    assert [r.old for r, _ in skipped] == ["WLTW", "GPS"]
    assert "swap" in skipped[0][1] and "GAP is not in today's list" in skipped[1][1]
    assert sp500.todays_ticker("SATS", dt.date(2026, 3, 23), kept) == "ECHO"
    assert sp500.todays_ticker("SATS", dt.date(2026, 7, 1), kept) == "SATS"


def test_phantoms_are_compared_in_todays_names():
    """ECHO is a member from the start as membership() gives it; read through EchoStar's
    rename it joins as SATS on day 30, so before then it is a phantom. GPS left on day 10 and
    became GAP on day 50: read through, it is called GAP on days 0..9 -- the same member under
    today's name, not a phantom and not a second one."""
    renames = (sp500.Rename(day(45), "SATS", "ECHO"), sp500.Rename(day(50), "GPS", "GAP"))
    given = [frozenset({"ECHO", "GPS"}) if i < 10 else frozenset({"ECHO"}) for i in range(N)]
    read = [frozenset({"GAP"}) if i < 10 else frozenset({"ECHO"}) if i >= 30 else frozenset()
            for i in range(N)]
    extra, lacking = idn.phantoms(DATES, given, read, renames)
    assert extra == {"ECHO": [(day(0), day(29), 30)]}
    assert lacking == {}


def test_the_fetch_plan_samples_share_counts_inside_each_span_under_its_own_ticker():
    """A count sampled under BALL before Ball traded as BALL is someone else's, and one
    carried across a change of identity is the BNY fund's 24.1 million shares on BNY Mellon.
    So every span is sampled on its own first and last day, and on the security's grid inside
    it, each under the span's ticker; samples already on disk cost nothing."""
    panel = {"BALL": held((20, N - 1))}
    records = {"BALL": (rec("BALL", 40, "B"), rec("BALL", N - 1, "B"))}
    histories = {"BALL": hist("B", (EARLY, "BLL"), (day(20), "BALL"))}
    _, rows, _, _ = build(panel, records, histories)
    plan = idn.fetch_plan(DATES, rows, {("B", "BLL"): 20, ("B", "BALL"): 40}, ["B"],
                          {("BALL", day(40)), ("BALL", day(N - 1))}, every=25)
    assert [(p.ticker, p.samples) for p in plan] == [
        ("BLL", (day(0), day(19))),
        ("BALL", (day(20), day(25), day(50), day(N - 1))),
    ]
    assert [p.new_samples for p in plan] == [2, 3]
    assert [p.member_days for p in plan] == [20, 40]


def test_the_cache_readers_tell_none_found_from_never_asked(tmp_path):
    (tmp_path / "ref-AAA-2024-01-01.json").write_text(json.dumps({"results": {
        "composite_figi": "F1", "cik": "0001", "name": "Aaa", "type": "CS",
        "list_date": "1999-01-01"}}))
    (tmp_path / "events-AAA.json").write_text(json.dumps({"status": "OK", "results": {
        "composite_figi": "F1", "cik": "0001", "name": "Aaa",
        "events": [{"type": "ticker_change", "date": "1999-01-01",
                    "ticker_change": {"ticker": "AAA"}}]}}))
    (tmp_path / "events-GONE.json").write_text(json.dumps({"status": "NOT_FOUND"}))
    records = idn.read_records(tmp_path, ["AAA", "BBB"])
    assert records["BBB"] == ()
    assert records["AAA"][0] == Record("AAA", dt.date(2024, 1, 1), "F1", "0001", "Aaa", "CS",
                                       dt.date(1999, 1, 1))
    histories = idn.read_histories(tmp_path, ["AAA", "GONE", "NEVER"])
    assert histories["GONE"] is None and "NEVER" not in histories
    assert histories["AAA"].tenures == (Tenure("AAA", dt.date(1999, 1, 1), None, False),)
