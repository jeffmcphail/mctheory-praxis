"""Tests for point-in-time S&P 500 constituents.

The headline case is `test_single_change_date_is_not_iterated_character_by_character`.
The predecessor implementation re-queried the changes table by date inside the walk loop,
where pandas returns a DataFrame for a date with several changes and a Series for a date
with one -- so single-change dates iterated the ticker string and appended 'W', 'H', 'R'
instead of 'WHR'.  Every other test here passes against that implementation; only this one
fails.  A test suite that exercises only multi-change dates is the reason it survived.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from praxis.midoffice.refdata import sp500
from praxis.midoffice.refdata.sp500 import (
    Change,
    CoverageError,
    Issue,
    Snapshot,
    StructureError,
    all_constituents,
    intervals,
    membership,
    parse_tables,
    validate,
)

UTC = dt.timezone.utc


def snap(current, changes, fetched="2026-10-16T12:00:00+00:00") -> Snapshot:
    return sp500._finalise(
        Snapshot(
            source_url="test://",
            fetched_at=dt.datetime.fromisoformat(fetched),
            current=tuple(current),
            changes=tuple(sorted(changes, key=lambda c: c.date, reverse=True)),
        )
    )


def constituents_df(symbols):
    return pd.DataFrame(
        {
            "Symbol": list(symbols),
            "Security": [f"{s} Inc" for s in symbols],
            "GICS Sector": ["Industrials"] * len(symbols),
        }
    )


def changes_df(rows):
    """rows: (date_str, added_ticker, removed_ticker) -- Wikipedia's MultiIndex shape."""
    cols = pd.MultiIndex.from_tuples(
        [
            ("Date", "Date"),
            ("Added", "Ticker"),
            ("Added", "Security"),
            ("Removed", "Ticker"),
            ("Removed", "Security"),
            ("Reason", "Reason"),
        ]
    )
    data = [[d, a, f"{a} Inc" if a else "", r, f"{r} Inc" if r else "", "n/a"] for d, a, r in rows]
    return pd.DataFrame(data, columns=cols)


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------


def test_single_change_date_is_not_iterated_character_by_character():
    """A date with exactly one change must yield one ticker, not one per character.

    Two single-change dates on purpose: the earlier one establishes coverage so the
    membership assertion is answerable, and both exercise the failing path.
    """
    tables = [
        constituents_df(["AAA", "SMCI"]),
        changes_df(
            [
                ("March 18, 2024", "SMCI", "WHR"),
                ("January 5, 2020", "AAA", "OLD"),
            ]
        ),
    ]
    _, changes = parse_tables(tables)

    assert len(changes) == 2
    recent = changes[0]
    assert recent.date == dt.date(2024, 3, 18)
    assert recent.removed == ("WHR",)
    assert recent.added == ("SMCI",)
    # The failure mode, stated explicitly so the test documents it.
    assert recent.removed != ("W", "H", "R")

    s = snap(["AAA", "SMCI"], changes)
    before = membership(s, dt.date(2024, 3, 17))
    assert "WHR" in before
    assert not any(len(t) == 1 for t in before), f"single-character tickers leaked in: {before}"


def test_multi_change_date_groups_into_one_entry():
    tables = [
        constituents_df(["KKR", "CRWD", "AAA"]),
        changes_df(
            [
                ("June 24, 2024", "KKR", "RHI"),
                ("June 24, 2024", "CRWD", "COMM"),
            ]
        ),
    ]
    _, changes = parse_tables(tables)
    assert len(changes) == 1, "two rows on the same date must group into one Change"
    assert changes[0].added == ("CRWD", "KKR")
    assert changes[0].removed == ("COMM", "RHI")


# ---------------------------------------------------------------------------
# The backward walk
# ---------------------------------------------------------------------------


def test_name_added_then_later_removed_is_resurrected_in_its_window():
    """The property most free approaches get wrong."""
    s = snap(
        current=["AAA", "BBB", "ANCHOR"],
        changes=[
            Change(dt.date(2024, 6, 1), added=("BBB",), removed=("TEMP",)),
            Change(dt.date(2022, 1, 5), added=("TEMP",), removed=("OLD",)),
            Change(dt.date(2020, 1, 1), added=("ANCHOR",), removed=("OLDER",)),
        ],
    )
    assert "TEMP" not in membership(s, dt.date(2026, 1, 1))
    assert "TEMP" in membership(s, dt.date(2023, 1, 1)), "a name added then dropped must reappear"
    assert "TEMP" not in membership(s, dt.date(2021, 12, 1))
    assert "OLD" in membership(s, dt.date(2022, 1, 4))
    assert "OLD" not in membership(s, dt.date(2022, 1, 5))


def test_change_date_belongs_to_the_post_change_interval():
    """A change effective on d means d already carries the new membership."""
    s = snap(
        current=["AAA", "NEW"],
        changes=[
            Change(dt.date(2025, 3, 10), added=("NEW",), removed=("GONE",)),
            Change(dt.date(2020, 1, 1), added=("AAA",), removed=("ANCIENT",)),
        ],
    )
    assert "NEW" in membership(s, dt.date(2025, 3, 10))
    assert "GONE" not in membership(s, dt.date(2025, 3, 10))
    assert "NEW" not in membership(s, dt.date(2025, 3, 9))
    assert "GONE" in membership(s, dt.date(2025, 3, 9))


def test_intervals_are_contiguous_and_half_open():
    s = snap(
        current=["AAA", "BBB", "CCC"],
        changes=[
            Change(dt.date(2025, 3, 10), added=("CCC",), removed=("X1",)),
            Change(dt.date(2024, 7, 1), added=("BBB",), removed=("X2",)),
            Change(dt.date(2023, 2, 2), added=("AAA",), removed=("X3",)),
        ],
    )
    ivs, issues = intervals(s)
    assert [iv.start for iv in ivs] == [
        dt.date(2023, 2, 2),
        dt.date(2024, 7, 1),
        dt.date(2025, 3, 10),
    ]
    for a, b in zip(ivs, ivs[1:]):
        assert a.end == b.start, "no gaps and no overlaps"
    assert ivs[-1].end == s.fetched_at.date() + dt.timedelta(days=1)
    assert not issues


def test_fetch_date_itself_is_covered():
    s = snap(current=["AAA"], changes=[Change(dt.date(2025, 1, 1), added=("AAA",))])
    assert membership(s, s.fetched_at.date()) == ("AAA",)


# ---------------------------------------------------------------------------
# Coverage -- refuse rather than extrapolate
# ---------------------------------------------------------------------------


def test_date_before_earliest_change_is_refused_not_guessed():
    s = snap(
        current=["AAA", "BBB"],
        changes=[Change(dt.date(2020, 5, 1), added=("BBB",), removed=("OLD",))],
    )
    assert s.coverage_start == dt.date(2020, 5, 1)
    with pytest.raises(CoverageError) as e:
        membership(s, dt.date(1995, 6, 1))
    assert "coverage" in str(e.value).lower()


def test_date_after_fetch_is_refused():
    s = snap(current=["AAA"], changes=[Change(dt.date(2025, 1, 1), added=("AAA",))])
    with pytest.raises(CoverageError):
        membership(s, dt.date(2027, 1, 1))


# ---------------------------------------------------------------------------
# Parsing robustness
# ---------------------------------------------------------------------------


def test_tables_are_located_by_signature_not_position():
    """An unrelated table inserted first must not break parsing."""
    noise = pd.DataFrame({"Rank": [1, 2], "Note": ["a", "b"]})
    tables = [
        noise,
        constituents_df(["AAA", "BBB"]),
        changes_df([("March 18, 2024", "AAA", "ZZZ")]),
    ]
    current, changes = parse_tables(tables)
    assert current == ("AAA", "BBB")
    assert changes[0].removed == ("ZZZ",)


def test_missing_changes_table_raises_structure_error():
    with pytest.raises(StructureError, match="changes table"):
        parse_tables([constituents_df(["AAA"])])


def test_unparseable_date_format_raises_rather_than_silently_dropping():
    rows = [(f"18/03/202{i % 10}", f"A{i}", f"B{i}") for i in range(20)]
    with pytest.raises(StructureError, match="date format"):
        parse_tables([constituents_df(["AAA"]), changes_df(rows)])


def test_overrides_and_footnotes_are_applied():
    tables = [
        constituents_df(["BRK.B", "AAA[1]"]),
        changes_df([("March 18, 2024", "OLDSYM", "ZZZ")]),
    ]
    current, changes = parse_tables(tables, overrides={"OLDSYM": "NEWSYM"})
    assert "AAA" in current and "BRK.B" in current
    assert changes[0].added == ("NEWSYM",)


# ---------------------------------------------------------------------------
# Anomalies are recorded, not swallowed
# ---------------------------------------------------------------------------


def test_inconsistent_source_table_is_reported_not_silently_absorbed():
    s = snap(
        current=["AAA"],
        changes=[Change(dt.date(2025, 1, 1), added=("NEVER_HERE",), removed=())],
    )
    _, issues = intervals(s)
    codes = {i.code for i in issues}
    assert "added_absent" in codes
    assert all(i.severity == "warning" for i in issues)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def big_snapshot(extra_history=()):
    current = [f"S{i:03d}" for i in range(500)]
    changes = [
        Change(dt.date(2020, 1, 1) + dt.timedelta(days=30 * i),
               added=(f"S{i:03d}",), removed=(f"R{i:03d}",))
        for i in range(60)
    ]
    changes.extend(extra_history)
    return snap(current, changes)


def test_clean_snapshot_validates_without_errors():
    issues = validate(big_snapshot())
    errors = [i for i in issues if i.severity == "error"]
    assert not errors, errors


def test_legitimate_single_letter_tickers_do_not_trip_validation():
    """The real index carries eleven one-letter tickers today (A, C, D, F, J, L, O, P, Q, T, V).

    An earlier draft of this module errored above twelve one-character names across history.
    That threshold fires on healthy data, so it was replaced: the length heuristic is now a
    warning keyed on removal-only appearances, and the error-level control is interval size.
    """
    real_singles = list("ACDFJLOPQTV")
    current = real_singles + [f"S{i:03d}" for i in range(490)]
    changes = [
        Change(dt.date(2020, 1, 1) + dt.timedelta(days=30 * i),
               added=(f"S{i:03d}",), removed=(f"R{i:03d}",))
        for i in range(60)
    ]
    issues = validate(snap(current, changes))
    assert not [i for i in issues if i.severity == "error"], \
        "legitimate one-letter tickers must not produce an error"


def test_split_ticker_is_caught_by_same_date_multiplicity():
    """The character-iteration signature: a ticker's letters removed together on one date."""
    extra = [Change(dt.date(2019, 6, 1), removed=("W", "H", "R"))]
    issues = validate(big_snapshot(extra_history=extra))
    flagged = [i for i in issues if i.code == "split_ticker_suspected"]
    assert flagged and flagged[0].severity == "error"
    assert "2019-06-01" in flagged[0].detail


def test_lone_single_letter_removal_is_not_flagged():
    """K, M, Q, R, S, T and X were all genuinely removed from the index, each alone."""
    extra = [
        Change(dt.date(2025, 12, 11), removed=("K",)),
        Change(dt.date(2020, 4, 6), removed=("M", "RTN")),
        Change(dt.date(2017, 6, 19), removed=("R", "YHOO", "TDC", "MJN")),
    ]
    issues = validate(big_snapshot(extra_history=extra))
    assert not [i for i in issues if i.code == "split_ticker_suspected"]


def test_malformed_ticker_in_changes_table_is_an_error():
    """Template markup leaks a trailing pipe into some live cells ("ALLE |", "JCP |")."""
    issues = validate(big_snapshot(extra_history=[Change(dt.date(2019, 5, 1), removed=("JCP |",))]))
    assert any(i.code == "malformed_tickers_history" and i.severity == "error" for i in issues)


def test_trailing_pipe_is_stripped_from_tickers():
    tables = [
        real_shaped_constituents_df(["AAA", "BBB"]),
        changes_df([("December 2, 2013", "ALLE |", "JCP |")]),
    ]
    _, changes = parse_tables(tables)
    assert changes[0].added == ("ALLE",)
    assert changes[0].removed == ("JCP",)


def test_oversized_interval_is_caught():
    bloat = [
        Change(dt.date(2019, 1, 1) + dt.timedelta(days=i), removed=(f"EXTRA{i:03d}",))
        for i in range(40)
    ]
    issues = validate(big_snapshot(extra_history=bloat))
    assert any(i.code == "interval_oversized" and i.severity == "error" for i in issues)


def test_implausible_constituent_count_is_an_error():
    issues = validate(snap(["AAA", "BBB"], [Change(dt.date(2025, 1, 1), added=("AAA",))]))
    assert any(i.code == "size_implausible" for i in issues)


# ---------------------------------------------------------------------------
# Snapshot identity and round-trip
# ---------------------------------------------------------------------------


def test_content_hash_is_stable_and_survives_round_trip(tmp_path):
    s = big_snapshot()
    path = sp500.save(s, tmp_path)
    back = sp500.load(path)
    assert back.content_hash == s.content_hash
    assert back.current == s.current
    assert back.changes == s.changes
    assert membership(back, dt.date(2024, 1, 15)) == membership(s, dt.date(2024, 1, 15))


def test_content_hash_ignores_fetch_time_but_tracks_content():
    a = snap(["AAA", "BBB"], [Change(dt.date(2025, 1, 1), added=("BBB",))],
             fetched="2026-10-16T12:00:00+00:00")
    b = snap(["AAA", "BBB"], [Change(dt.date(2025, 1, 1), added=("BBB",))],
             fetched="2026-10-17T12:00:00+00:00")
    c = snap(["AAA", "CCC"], [Change(dt.date(2025, 1, 1), added=("CCC",))],
             fetched="2026-10-16T12:00:00+00:00")
    assert a.content_hash == b.content_hash, "same content fetched later is the same content"
    assert a.content_hash != c.content_hash, "different content must hash differently"


def test_load_latest_picks_the_newest_by_fetch_time(tmp_path):
    old = snap(["AAA"], [Change(dt.date(2025, 1, 1), added=("AAA",))],
               fetched="2026-01-01T00:00:00+00:00")
    new = snap(["AAA", "BBB"], [Change(dt.date(2025, 1, 1), added=("AAA",))],
               fetched="2026-06-01T00:00:00+00:00")
    sp500.save(old, tmp_path)
    sp500.save(new, tmp_path)
    assert sp500.load_latest(tmp_path).fetched_at == new.fetched_at


def test_query_path_does_not_import_pandas(tmp_path, monkeypatch):
    """membership() must work from a stored snapshot with no network and no pandas."""
    s = big_snapshot()
    path = sp500.save(s, tmp_path)
    raw = json.loads(path.read_text())
    rebuilt = Snapshot.from_dict(raw)
    assert len(membership(rebuilt, dt.date(2023, 6, 1))) > 0


def test_all_constituents_spans_the_history():
    s = big_snapshot()
    everything = all_constituents(s)
    assert any(t.startswith("R") for t in everything), "removed names must appear in history"
    assert len(everything) > len(s.current)


def test_reversed_state_below_the_earliest_change_is_not_served():
    """Reversing the earliest change yields a membership whose validity window is unknowable.

    The snapshot cannot say how far back it held, because earlier changes may exist and be
    unrecorded.  So coverage stops at the earliest recorded change rather than serving a
    state with no lower bound.  Three tests in this file originally assumed otherwise.
    """
    s = snap(
        current=["AAA", "BBB"],
        changes=[Change(dt.date(2024, 1, 10), added=("BBB",), removed=("PRIOR",))],
    )
    assert s.coverage_start == dt.date(2024, 1, 10)
    assert membership(s, dt.date(2024, 1, 10)) == ("AAA", "BBB")
    with pytest.raises(CoverageError):
        membership(s, dt.date(2024, 1, 9))


# ---------------------------------------------------------------------------
# Fixtures taken from the live page, 2026-10-16
# ---------------------------------------------------------------------------

REAL_CONSTITUENT_COLUMNS = [
    "Symbol", "Security", "GICS Sector", "GICS Sub-Industry",
    "Headquarters Location", "Date added", "CIK",
]

# Every one-letter ticker in the index on 2026-10-16. All legitimate.
REAL_SINGLE_LETTER_TICKERS = list("ACDFJLOPQTV")


def real_shaped_constituents_df(symbols):
    """The constituents table with the column set the live page actually carries."""
    return pd.DataFrame(
        {
            "Symbol": list(symbols),
            "Security": [f"{s} Inc" for s in symbols],
            "GICS Sector": ["Industrials"] * len(symbols),
            "GICS Sub-Industry": ["Widgets"] * len(symbols),
            "Headquarters Location": ["Toronto, Ontario"] * len(symbols),
            "Date added": ["1999-01-01"] * len(symbols),
            "CIK": [1234567] * len(symbols),
        }
    )


def test_live_page_column_set_is_located():
    """Guards against drift in the constituents table's real column names."""
    tables = [
        real_shaped_constituents_df(["AAPL", "BRK.B", "A"]),
        changes_df([("March 18, 2024", "AAPL", "WHR")]),
    ]
    current, changes = parse_tables(tables)
    assert current == ("A", "AAPL", "BRK.B")
    assert changes[0].removed == ("WHR",)


def test_class_shares_and_real_single_letters_survive_parsing():
    symbols = REAL_SINGLE_LETTER_TICKERS + ["BRK.B", "BF.B", "AAPL"]
    tables = [
        real_shaped_constituents_df(symbols),
        changes_df([("March 18, 2024", "AAPL", "WHR")]),
    ]
    current, _ = parse_tables(tables)
    assert set(REAL_SINGLE_LETTER_TICKERS) <= set(current)
    assert "BRK.B" in current and "BF.B" in current
    assert all(sp500.TICKER_RE.match(t) for t in current)


def test_changes_tables_are_pooled_across_sources():
    """Wikipedia serves constituents and changes from two different articles."""
    tables = [
        real_shaped_constituents_df(["AAA", "BBB"]),
        changes_df([("March 18, 2024", "AAA", "WHR")]),      # as if from article one
        changes_df([("January 5, 2020", "BBB", "OLD")]),     # as if from article two
    ]
    _, changes = parse_tables(tables)
    assert len(changes) == 2
    assert {c.date for c in changes} == {dt.date(2024, 3, 18), dt.date(2020, 1, 5)}


def test_missing_changes_source_names_the_likely_cause():
    """The real failure encountered on 2026-10-16, when the table moved articles."""
    with pytest.raises(StructureError) as e:
        parse_tables([real_shaped_constituents_df(["AAA"])])
    msg = str(e.value)
    assert "no changes table" in msg
    assert "DEFAULT_SOURCES" in msg, "the message must point at the fix"
