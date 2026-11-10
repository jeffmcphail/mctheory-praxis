"""Who each ticker is: an identity table for a ticker-keyed price panel, from what is on disk.

The S&P 500 panel names a security by a ticker string -- the one membership() returns, today's
for a current member -- and asked Polygon for that string's prices across the whole window. A
ticker is not a security. Over 2022-01-03..2026-11-03 the panel's META is the Roundhill Ball
Metaverse ETF until 2022-01-28 and Meta Platforms from 2022-06-09; its DOC is Physicians Realty
until 2024-03-01 and Healthpeak from 2024-03-04, with no gap in the prices to show it; its BALL
starts on 2022-05-10 because Ball traded as BLL before. This module works out, for every date a
ticker has a price, whose price it is, and for every index member on every date, which
security it is -- or says it cannot.

**A security is a share class**, identified by its composite FIGI where the records carry one,
and otherwise by its CIK. The CIK names a company, not a share class -- GOOGL and GOOG are one
CIK and two FIGIs, and both are members -- so it is the cross-check, never the key where a FIGI
exists. A FIGI that changes under one ticker is a new security even where the index carries it
on unbroken: a holding-company reorganisation (XOM on 2026-07-02, BLK on 2024-10-02) issues
one, and the table records it as a reorganisation rather than pretend the identifier did not
change. The index member runs on through it.

**Three sources, cheapest first, none of them complete.**

- The reference records the panel fetch cached: for each ticker, at each of its sample dates,
  the composite FIGI, CIK, name, type and list date of whoever then traded under it. Direct,
  but six dates a ticker, so a change between two of them is bracketed, not dated.
- Polygon's ticker events (`PolygonClient.ticker_events`): for a listed security, each ticker
  it has traded under and the date each began. Dated, but listed securities only; and the first
  event is where Polygon's record of the security starts, sometimes a split or a redomicile
  rather than the day the ticker began -- Super Micro's is its split of 2024-10-01 -- so only a
  change from one ticker to another is a known start. The list can also be short (Fiserv's
  stops at FI, though it trades as FISV again from 2025-11-11) or give a share class's root
  (BF for BF.B, BRK for BRK.B, GOOG for GOOGL).
- The panel's own prices. Each ticker's were asked for over the whole window, so a date
  without one is a date nobody traded under it, past a halt of under `HALT_DAYS`.

Where the events and the panel's records and prices disagree, the panel's are taken -- they
are its own evidence of who traded under the tickers it holds -- and the disagreement is
reported (`Conflict`).

**What is pinned and what is not.** Between two records of one ticker naming different
securities, the change is dated by the later security's events (the day it began under the
ticker), by the earlier one's (the day it left), or by a gap in the prices; failing all three,
the dates between are left unassigned and reported, never guessed. Before a ticker's first
record and after its last, its prices go to that record's security as far as they run without
a gap and the security's events allow.

**The rebuild's evidence.** The table is built from the panel fetch's own records. The
rebuild's fetch then prices every member security's spans and samples its share count inside
them, and what it found corrects the table (`start_at_first_price`, `drop_refuted`,
`reorganise`, `confirm_span`): a span starts at its first price under its ticker; a span with
no price, or one every record under its ticker refutes, goes; a reorganisation is dated by the
identifier the table keys on; and a span the reference endpoint confirms at both ends joins
its security. Each correction is recorded with why (`Correction`).

Pure functions over in-memory inputs, and three readers of the client's cache. No network.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .sp500 import Rename

HALT_DAYS = 5
"""Fewer missing trading days than this inside one ticker's prices is a halt: it neither dates
a change of hands nor breaks a run of prices. A longer gap is evidence only between two records
naming different securities -- on the S&P 500 panel the changes leave gaps of 71 trading days
(BNY), 90 (META) and 658 (INFO) -- and is no change at all between two naming the same one, as
Signature Bank's 11-day suspension of March 2023 is not."""


# ----------------------------------------------------------------------------- evidence


def security_key(figi: str | None, cik: str | None, name: str | None = None) -> str:
    """The share-class identifier: the composite FIGI where there is one, else the CIK, else
    the name. Two share classes of one company without FIGIs would share a key; the guards
    would then find one security under two tickers on one date, and say so."""
    if figi:
        return str(figi)
    if cik:
        return f"CIK{cik}"
    if name:
        return f"NAME:{name}"
    raise ValueError("a security needs a FIGI, a CIK or a name to be identified")


@dataclass(frozen=True)
class Record:
    """One cached reference record: who traded under `ticker` on `on`, as Polygon has it."""

    ticker: str
    on: dt.date
    figi: str | None
    cik: str | None
    name: str | None
    kind: str | None
    """Polygon's `type`: CS for common stock, ETF, FUND, and others."""
    list_date: dt.date | None = None
    share_class: str | None = None
    """The share-class FIGI, which every composite FIGI of one share class shares: reported
    beside a composite FIGI the table does not expect, and not yet a key."""

    @property
    def key(self) -> str:
        return security_key(self.figi, self.cik, self.name)

    @property
    def identified(self) -> bool:
        """Whether it carries a FIGI or a CIK. One with neither -- Everest Group's under EG on
        its first day there, a name and a list date and nothing else -- is no evidence of
        another security: its key would be the name, which no other record carries."""
        return bool(self.figi or self.cik)


@dataclass(frozen=True)
class Tenure:
    """A ticker a security traded under, from `start` to `end` inclusive (None: it still does)."""

    ticker: str
    start: dt.date
    end: dt.date | None
    known_start: bool
    """True where `start` is a change from another ticker; False for the first event, the start
    of Polygon's record of the security, before which it may have traded under this ticker or
    not at all."""
    presumed: bool = False
    """The ticker a security is found under today, where its events end with a change away
    from it and never record the return: presumed, not evidenced."""


@dataclass(frozen=True)
class History:
    """A security's ticker events, as `PolygonClient.ticker_events` gives them."""

    asked: str
    figi: str | None
    cik: str | None
    name: str | None
    tenures: tuple[Tenure, ...]

    @property
    def key(self) -> str:
        return security_key(self.figi, self.cik, self.name)


def history_from_events(asked: str, results: Mapping) -> History:
    """A `History` from the endpoint's `results`: its ticker changes in date order, a repeat of
    the ticker before it dropped, each tenure ending the day before the next begins."""
    events = sorted(
        (dt.date.fromisoformat(str(e["date"])[:10]), str(e["ticker_change"]["ticker"]))
        for e in results.get("events", ())
        if e.get("type") == "ticker_change" and (e.get("ticker_change") or {}).get("ticker")
    )
    steps: list[tuple[dt.date, str]] = []
    for on, ticker in events:
        if not steps or steps[-1][1] != ticker:
            steps.append((on, ticker))
    tenures = tuple(
        Tenure(
            ticker=ticker,
            start=on,
            end=steps[k + 1][0] - dt.timedelta(days=1) if k + 1 < len(steps) else None,
            known_start=k > 0,
        )
        for k, (on, ticker) in enumerate(steps)
    )
    return History(
        asked=asked,
        figi=results.get("composite_figi") or None,
        cik=results.get("cik") or None,
        name=results.get("name") or None,
        tenures=tenures,
    )


def _date(text) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(text)[:10]) if text else None
    except ValueError:
        return None


def read_records(
    cache: Path | str, tickers: Iterable[str], on: Iterable[dt.date] | None = None
) -> dict[str, tuple[Record, ...]]:
    """Every cached reference record of each ticker, oldest first -- `ref-<ticker>-<date>.json`,
    as `PolygonClient.ticker_details` writes them. No requests.

    `on`, if given, keeps the records on those dates alone: the panel fetch's own sample
    dates, so that the records a later fetch caches beside them -- the rebuild's share samples,
    which check the table -- do not become the evidence it is built from."""
    cache = Path(cache)
    dates = None if on is None else set(on)
    out: dict[str, tuple[Record, ...]] = {}
    for ticker in tickers:
        prefix = f"ref-{ticker}-"
        found = []
        for path in cache.glob(f"{prefix}*.json"):
            on_day = _date(path.stem[len(prefix):])
            if on_day is None or (dates is not None and on_day not in dates):
                continue
            results = json.loads(path.read_text(encoding="utf-8")).get("results") or {}
            if not results:
                continue
            found.append(Record(
                ticker=ticker,
                on=on_day,
                figi=results.get("composite_figi") or None,
                cik=results.get("cik") or None,
                name=results.get("name") or None,
                kind=results.get("type") or None,
                list_date=_date(results.get("list_date")),
                share_class=results.get("share_class_figi") or None,
            ))
        out[ticker] = tuple(sorted(found, key=lambda r: r.on))
    return out


def read_histories(cache: Path | str, ids: Iterable[str]) -> dict[str, History | None]:
    """The cached ticker events of each id: a `History`, or None where Polygon had none -- a
    delisted security. An id never asked is absent, which is not the same as None."""
    cache = Path(cache)
    out: dict[str, History | None] = {}
    for identifier in ids:
        path = cache / f"events-{identifier}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("status", "")).upper() == "NOT_FOUND":
            out[identifier] = None
        else:
            out[identifier] = history_from_events(identifier, payload.get("results") or {})
    return out


def histories_by_key(histories: Mapping[str, History | None]) -> dict[str, History]:
    """One history per security, however many ids found it -- by two tickers, or by a ticker
    and a FIGI -- their events pooled."""
    pooled: dict[str, set[tuple[dt.date, str]]] = {}
    base: dict[str, History] = {}
    for history in histories.values():
        if history is None:
            continue
        base.setdefault(history.key, history)
        pooled.setdefault(history.key, set()).update(
            (t.start, t.ticker) for t in history.tenures
        )
    return {
        key: history_from_events(base[key].asked, {
            "composite_figi": base[key].figi,
            "cik": base[key].cik,
            "name": base[key].name,
            "events": [
                {"type": "ticker_change", "date": on.isoformat(), "ticker_change": {"ticker": t}}
                for on, t in sorted(steps)
            ],
        })
        for key, steps in pooled.items()
    }


# ------------------------------------------------------------- one ticker's prices, by owner


@dataclass(frozen=True)
class Segment:
    """A run of one ticker's prices that one security traded under, unbroken but for halts."""

    ticker: str
    key: str
    first: dt.date
    last: dt.date
    held: int
    starts: str
    ends: str


@dataclass(frozen=True)
class Boundary:
    """Where one ticker passes from one security to another, between two of its records."""

    ticker: str
    before: str
    before_name: str | None
    after: str
    after_name: str | None
    last_before: dt.date | None
    first_after: dt.date | None
    kind: str
    """"splice": an unrelated security under the same ticker -- one that traded under another
    ticker before taking this one, or one of another type, or after a gap. "reorganisation": a
    new identifier for the same holding -- the later security has no earlier ticker, and the
    prices run on unbroken, of the same type."""
    evidence: str
    pinned: bool


@dataclass(frozen=True)
class TickerIdentity:
    """Whose prices one ticker holds: per date, a security key, or None where none is."""

    ticker: str
    owner: tuple[str | None, ...]
    segments: tuple[Segment, ...]
    boundaries: tuple[Boundary, ...]
    unassigned: tuple[tuple[dt.date, dt.date, int, str], ...]
    """Runs of priced dates no evidence assigns: first, last, how many, and why."""


def _tenure_starting(
    history: History | None, ticker: str, after: dt.date, until: dt.date
) -> Tenure | None:
    """The tenure under `ticker` that starts in (after, until], the last if several."""
    if history is None:
        return None
    found = [t for t in history.tenures if t.ticker == ticker and after < t.start <= until]
    return found[-1] if found else None


def _tenure_ending(
    history: History | None, ticker: str, after: dt.date, before: dt.date
) -> Tenure | None:
    """The tenure under `ticker` that ends in [after, before), the first if several."""
    if history is None:
        return None
    found = [
        t for t in history.tenures
        if t.ticker == ticker and t.end is not None and after <= t.end < before
    ]
    return found[0] if found else None


def _run_around(held: np.ndarray, i: int) -> tuple[int, int] | None:
    """The run of priced positions, joined across halts, containing position `i`."""
    where = np.flatnonzero(held)
    if not where.size:
        return None
    runs = [[int(where[0]), int(where[0])]]
    for j in where[1:]:
        if j - runs[-1][1] - 1 < HALT_DAYS:
            runs[-1][1] = int(j)
        else:
            runs.append([int(j), int(j)])
    for lo, hi in runs:
        if lo <= i <= hi:
            return lo, hi
    return None


def segment_ticker(
    ticker: str,
    dates: Sequence[dt.date],
    held: np.ndarray,
    records: Sequence[Record],
    histories: Mapping[str, History],
) -> TickerIdentity:
    """Assign each date on which `ticker` has a price to the security that traded under it.

    `held` is a boolean per date of `dates`; `records` are the ticker's own; `histories` are by
    security key. The rules are the module docstring's.
    """
    held = np.asarray(held, dtype=bool)
    n = len(dates)
    at = {day: i for i, day in enumerate(dates)}
    owner: list[str | None] = [None] * n
    unassigned: list[tuple[dt.date, dt.date, int, str]] = []
    boundaries: list[Boundary] = []
    starts: dict[int, str] = {}
    ends: dict[int, str] = {}

    def leave(lo: int, hi: int, why: str) -> None:
        cells = [i for i in range(lo, hi + 1) if held[i] and owner[i] is None]
        if cells:
            unassigned.append((dates[cells[0]], dates[cells[-1]], len(cells), why))

    # Anchors: consecutive records naming one security, as [key, first, last, records]. A
    # record with neither a FIGI nor a CIK names nobody (`Record.identified`).
    anchors: list[list] = []
    for record in sorted(records, key=lambda r: r.on):
        if not record.identified:
            continue
        if record.on not in at:
            raise ValueError(f"{ticker}'s record of {record.on} is not on a panel date")
        i = at[record.on]
        if anchors and anchors[-1][0] == record.key:
            anchors[-1][2] = i
            anchors[-1][3].append(record)
        else:
            anchors.append([record.key, i, i, [record]])
    if not anchors:
        leave(0, n - 1, "no reference record names anyone under the ticker")
        return TickerIdentity(ticker, tuple(owner), (), (), tuple(unassigned))
    for key, lo, hi, _ in anchors:
        for i in range(lo, hi + 1):
            if held[i]:
                owner[i] = key

    # Between two anchors: date the change.
    for (a_key, _, a, a_records), (b_key, b, _, b_records) in zip(anchors, anchors[1:]):
        a_name, b_name = a_records[-1].name or a_key, b_records[0].name or b_key
        arrive = _tenure_starting(histories.get(b_key), ticker, dates[a], dates[b])
        depart = _tenure_ending(histories.get(a_key), ticker, dates[a], dates[b])
        priced = [i for i in range(a, b + 1) if held[i]]
        gap = None
        for x, y in zip(priced, priced[1:]):
            if y - x - 1 >= HALT_DAYS and (gap is None or y - x > gap[1] - gap[0]):
                gap = (x, y)
        between = [i for i in range(a + 1, b) if held[i]]
        evidence: list[str] = []
        if arrive is not None:
            evidence.append(
                f"{b_name} began trading as {ticker} on {arrive.start} (events: "
                + ("a change from another ticker)" if arrive.known_start
                   else "the first of its record)")
            )
        if depart is not None:
            evidence.append(f"{a_name} traded as another ticker from "
                            f"{depart.end + dt.timedelta(days=1)} (events)")
        for i in between:
            if arrive is not None:
                if dates[i] >= arrive.start:
                    owner[i] = b_key
                elif depart is None or dates[i] <= depart.end:
                    owner[i] = a_key
            elif depart is not None:
                if dates[i] <= depart.end:
                    owner[i] = a_key
                elif gap is not None and i >= gap[1]:
                    owner[i] = b_key
            elif gap is not None:
                owner[i] = a_key if i <= gap[0] else b_key
        if gap is not None and arrive is None and depart is None:
            evidence.append(f"a gap in the prices from {dates[gap[0]]} to {dates[gap[1]]}")
        pinned = arrive is not None or depart is not None or gap is not None
        if not pinned:
            evidence.append(f"nothing dates the change between the records of {dates[a]} and "
                            f"{dates[b]}")
        leave(a + 1, b - 1, f"between {a_name} under {ticker} and {b_name}: "
                            + "; ".join(evidence))
        last_before = max((i for i in priced if owner[i] == a_key), default=None)
        first_after = min((i for i in priced if owner[i] == b_key), default=None)
        contiguous = (
            last_before is not None and first_after is not None
            and first_after - last_before - 1 < HALT_DAYS
        )
        kinds = {a_records[-1].kind, b_records[0].kind}
        kind = (
            "splice"
            if (arrive is not None and arrive.known_start) or len(kinds) > 1 or not contiguous
            else "reorganisation"
        )
        boundaries.append(Boundary(
            ticker, a_key, a_records[-1].name, b_key, b_records[0].name,
            dates[last_before] if last_before is not None else None,
            dates[first_after] if first_after is not None else None,
            kind, "; ".join(evidence), pinned,
        ))
        if first_after is not None:
            starts[first_after] = "; ".join(evidence)
        if last_before is not None:
            ends[last_before] = "; ".join(evidence)

    # Before the first record: the first record's security, from the day its events say it
    # began under the ticker, or else as far back as the prices run unbroken.
    head_key, head, _, head_records = anchors[0]
    arrive = _tenure_starting(histories.get(head_key), ticker, dates[0], dates[head])
    run = _run_around(held, head)
    for i in range(head):
        if not held[i]:
            continue
        if arrive is not None:
            if dates[i] >= arrive.start:
                owner[i] = head_key
        elif run is not None and i >= run[0]:
            owner[i] = head_key
    if arrive is not None:
        first = min((i for i in range(head + 1) if held[i] and owner[i] == head_key),
                    default=None)
        if first is not None and first > 0:
            starts[first] = (f"{head_records[0].name or head_key} began trading as {ticker} on "
                             f"{arrive.start} (events: "
                             + ("a change from another ticker)" if arrive.known_start
                                else "the first of its record)"))
        leave(0, head - 1, f"before {head_records[0].name or head_key} began trading as "
                           f"{ticker} on {arrive.start}, and no record names who did")
    else:
        leave(0, head - 1, f"prices before the first record, {dates[head]}, with a gap between")

    # After the last record: the last record's security, to the day its events say it left
    # the ticker, or else as far as the prices run unbroken.
    tail_key, _, tail, tail_records = anchors[-1]
    depart = _tenure_ending(histories.get(tail_key), ticker, dates[tail], dates[-1])
    run = _run_around(held, tail)
    for i in range(tail + 1, n):
        if not held[i]:
            continue
        if depart is not None:
            if dates[i] <= depart.end:
                owner[i] = tail_key
        elif run is not None and i <= run[1]:
            owner[i] = tail_key
    if depart is not None:
        last = max((i for i in range(tail, n) if held[i] and owner[i] == tail_key),
                   default=None)
        if last is not None and last < n - 1:
            ends[last] = (f"{tail_records[-1].name or tail_key} traded as another ticker from "
                          f"{depart.end + dt.timedelta(days=1)} (events)")
        leave(tail + 1, n - 1, f"after {tail_records[-1].name or tail_key} left {ticker} on "
                               f"{depart.end}, and no record names who traded under it")
    else:
        leave(tail + 1, n - 1, f"prices after the last record, {dates[tail]}, with a gap "
                               f"between")

    # Segments: runs of one owner, broken by a halt-length gap or another's price.
    segments: list[list] = []
    for i in range(n):
        if not held[i] or owner[i] is None:
            continue
        if segments and segments[-1][0] == owner[i] and i - segments[-1][2] - 1 < HALT_DAYS \
                and all(not held[j] for j in range(segments[-1][2] + 1, i)):
            segments[-1][2] = i
            segments[-1][3] += 1
        else:
            segments.append([owner[i], i, i, 1])
    return TickerIdentity(
        ticker=ticker,
        owner=tuple(owner),
        segments=tuple(
            Segment(
                ticker, key, dates[lo], dates[hi], count,
                starts.get(lo, "the window's first day" if lo == 0
                           else "its first price under the ticker"),
                ends.get(hi, "the window's last day" if hi == n - 1
                         else "its last price under the ticker"),
            )
            for key, lo, hi, count in segments
        ),
        boundaries=tuple(boundaries),
        unassigned=tuple(unassigned),
    )


# ------------------------------------------------------------------ securities and their rows


@dataclass(frozen=True)
class Row:
    """One security under one ticker over one span of the window: the identity table's row."""

    key: str
    figi: str | None
    cik: str | None
    name: str | None
    kind: str | None
    ticker: str
    first: dt.date
    last: dt.date
    held: int
    """Dates of the span on which the panel holds this security's price under the ticker."""
    source: str
    """"panel" -- the panel's prices under the ticker, named by its records -- "events", or
    "presumed" (`Tenure.presumed`)."""
    starts: str
    ends: str


@dataclass(frozen=True)
class Conflict:
    """Where Polygon's events and the panel's records and prices disagree; the panel's taken."""

    key: str
    name: str | None
    detail: str


def _trading_span(dates: Sequence[dt.date], tenure: Tenure) -> tuple[int, int] | None:
    """The first and last positions of `dates` inside the tenure, or None if none are."""
    inside = [
        i for i, day in enumerate(dates)
        if day >= tenure.start and (tenure.end is None or day <= tenure.end)
    ]
    return (inside[0], inside[-1]) if inside else None


def credible_tenures(
    history: History | None,
    held_under: Mapping[str, set[int]],
    dates: Sequence[dt.date],
    found_as: str | None = None,
) -> tuple[list[Tenure], list[str]]:
    """A security's tenures from its events, each cut short where the panel holds the
    security's prices under another ticker -- the GOOG that GOOGL's events name, Fiserv's FI
    after its return to FISV. `held_under` is the security's priced positions per panel
    ticker.

    `found_as` is the ticker the events were found by, which the security holds today. Where
    the events end with a change away from it and the panel does not hold it under that ticker
    -- Casey's, whose end with CASYV from 2010-08-27 though CASY finds it -- the return is in
    neither source, and the last tenure is replaced by one under `found_as`, presumed.

    Returns the tenures kept and a note of each cut."""
    if history is None:
        return [], []
    held_at: dict[int, str] = {}
    for ticker, cells in held_under.items():
        for i in cells:
            held_at[i] = ticker
    kept: list[Tenure] = []
    notes: list[str] = []
    tenures = list(history.tenures)
    if (found_as is not None and tenures and tenures[-1].ticker != found_as
            and found_as not in held_under):
        last = tenures.pop()
        notes.append(
            f"its events end with {last.ticker} from {last.start}, yet {found_as} finds it "
            f"today; when it began trading as {found_as} again is in neither source, and it is "
            f"presumed {found_as} from {last.start}"
        )
        tenures.append(Tenure(found_as, last.start, None, False, presumed=True))
    for tenure in tenures:
        clash = next(
            (i for i, day in enumerate(dates)
             if day >= tenure.start and (tenure.end is None or day <= tenure.end)
             and held_at.get(i, tenure.ticker) != tenure.ticker),
            None,
        )
        if clash is None:
            kept.append(tenure)
            continue
        end = dates[clash] - dt.timedelta(days=1)
        notes.append(
            f"its events have it trading as {tenure.ticker} from {tenure.start}"
            + (f" to {tenure.end}" if tenure.end else "")
            + f", but the panel holds its prices under {held_at[clash]} from {dates[clash]}; "
            + (f"the tenure is taken to end on {end}" if end >= tenure.start
               else "the tenure is dropped")
        )
        if end >= tenure.start:
            kept.append(Tenure(tenure.ticker, tenure.start, end, tenure.known_start,
                               tenure.presumed))
    # A tenure inside the window holding no trading day is not one: Signature Bank's S217 of
    # 2023-11-18 and -19, a weekend, its prices running on under SBNY. It is dropped, and the
    # tenure before it runs on to the one after.
    out: list[Tenure] = []
    for tenure in kept:
        empty = (
            tenure.start >= dates[0] and tenure.end is not None and tenure.end <= dates[-1]
            and _trading_span(dates, tenure) is None
        )
        if empty:
            notes.append(f"its events have it trading as {tenure.ticker} from {tenure.start} to "
                         f"{tenure.end}, which holds no trading day; the tenure is dropped")
            continue
        out.append(tenure)
    for k, tenure in enumerate(out):
        follow = out[k + 1].start - dt.timedelta(days=1) if k + 1 < len(out) else None
        if tenure.end is not None and (follow is None or tenure.end < follow) and any(
            t not in out and t.start == tenure.end + dt.timedelta(days=1) for t in kept
        ):
            out[k] = Tenure(tenure.ticker, tenure.start, follow, tenure.known_start,
                            tenure.presumed)
    return out, notes


def build_rows(
    dates: Sequence[dt.date],
    identities: Mapping[str, TickerIdentity],
    records: Mapping[str, Sequence[Record]],
    histories: Mapping[str, History],
    extra_keys: Iterable[str] = (),
    found_as: Mapping[str, str] | None = None,
) -> tuple[list[Row], list[Conflict], dict[str, list[Tenure]]]:
    """The identity table: per security, its spans under each ticker within the window.

    Spans the panel prices come from its segments; the rest from the security's credible
    tenures, clipped to the window. A tenure under one of the panel's own tickers adds nothing
    the panel's prices do not show: the panel asked for that ticker across the window, so off
    its own prices nobody traded under it, this security included. `extra_keys` are
    securities the panel holds nothing of -- members it lacks -- whose rows come from their
    events alone. `found_as` is, per security, the ticker its events were found by, if any
    (see `credible_tenures`). Returns the rows, the conflicts, and each security's credible
    tenures."""
    panel_tickers = set(identities)
    at = {day: i for i, day in enumerate(dates)}
    held_under: dict[str, dict[str, set[int]]] = {}
    segments_of: dict[str, list[Segment]] = {}
    for ticker, identity in identities.items():
        for i, key in enumerate(identity.owner):
            if key is not None:
                held_under.setdefault(key, {}).setdefault(ticker, set()).add(i)
        for segment in identity.segments:
            segments_of.setdefault(segment.key, []).append(segment)
    described: dict[str, Record] = {}
    for record in sorted((r for rs in records.values() for r in rs), key=lambda r: r.on):
        described[record.key] = record  # the latest record's name and type
    rows: list[Row] = []
    conflicts: list[Conflict] = []
    tenures_of: dict[str, list[Tenure]] = {}
    for key in dict.fromkeys([*segments_of, *extra_keys]):
        history = histories.get(key)
        record = described.get(key)
        source = record if record is not None else history
        name = getattr(source, "name", None)
        figi = getattr(source, "figi", None)
        cik = getattr(source, "cik", None)
        kind = record.kind if record is not None else None
        cells = held_under.get(key, {})
        tenures, notes = credible_tenures(history, cells, dates, (found_as or {}).get(key))
        tenures_of[key] = tenures
        conflicts.extend(Conflict(key, name, note) for note in notes)
        claimed: set[int] = set()
        for segment in sorted(segments_of.get(key, ()), key=lambda s: s.first):
            rows.append(Row(key, figi, cik, name, kind, segment.ticker, segment.first,
                            segment.last, segment.held, "panel", segment.starts, segment.ends))
            claimed.update(range(at[segment.first], at[segment.last] + 1))
        for tenure in tenures:
            if tenure.ticker in panel_tickers:
                continue
            within = _trading_span(dates, tenure)
            if within is None:
                continue
            first, last = within
            runs: list[list[int]] = []
            for i in range(first, last + 1):
                if i in claimed:
                    continue
                if runs and i == runs[-1][1] + 1:
                    runs[-1][1] = i
                else:
                    runs.append([i, i])
            for lo, hi in runs:
                if lo > first:
                    began = "the day after its prices under another ticker end"
                elif tenure.start < dates[0]:
                    began = f"the window's first day (as {tenure.ticker} from {tenure.start})"
                elif tenure.known_start:
                    began = f"began trading as {tenure.ticker} on {tenure.start} (events)"
                else:
                    began = (f"the first of its events, {tenure.start}: the start of Polygon's "
                             f"record of it, not necessarily of the ticker")
                if hi < last:
                    ended = "the day before its prices under another ticker begin"
                elif tenure.end is None or tenure.end >= dates[-1]:
                    ended = "the window's last day"
                else:
                    ended = (f"traded as another ticker from "
                             f"{tenure.end + dt.timedelta(days=1)} (events)")
                if tenure.presumed:
                    began = (f"presumed: {tenure.ticker} finds it today, and its events' last "
                             f"change, on {tenure.start}, was away from it")
                rows.append(Row(key, figi, cik, name, kind, tenure.ticker, dates[lo],
                                dates[hi], 0, "presumed" if tenure.presumed else "events",
                                began, ended))
    rows.sort(key=lambda r: (r.key, r.first, r.ticker))
    return rows, conflicts, tenures_of


# ----------------------------------------------------------------------------------- guards


@dataclass(frozen=True)
class GuardResult:
    name: str
    checked: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def guard_cells(
    dates: Sequence[dt.date], held: Mapping[str, np.ndarray], rows: Sequence[Row]
) -> GuardResult:
    """Every (ticker, date) the panel holds a price for maps to exactly one security: exactly
    one row of the table, under that ticker, spans the date."""
    by_ticker: dict[str, list[Row]] = {}
    for row in rows:
        by_ticker.setdefault(row.ticker, []).append(row)
    failures: list[str] = []
    checked = 0
    for ticker, mask in held.items():
        mine = by_ticker.get(ticker, [])
        for i in np.flatnonzero(mask):
            day = dates[int(i)]
            checked += 1
            owners = sorted({r.key for r in mine if r.first <= day <= r.last})
            if len(owners) != 1:
                failures.append(f"{ticker} on {day}: {len(owners)} securities"
                                + (f" ({', '.join(owners)})" if owners else ""))
    return GuardResult("every priced (ticker, date) maps to exactly one security", checked,
                       tuple(failures))


def guard_one_ticker(rows: Sequence[Row]) -> GuardResult:
    """No security is held under two tickers on one date: no two of a security's rows -- the
    panel's or its events' -- overlap."""
    by_key: dict[str, list[Row]] = {}
    for row in rows:
        by_key.setdefault(row.key, []).append(row)
    failures: list[str] = []
    checked = 0
    for key, mine in by_key.items():
        mine = sorted(mine, key=lambda r: r.first)
        checked += len(mine)
        for x, y in zip(mine, mine[1:]):
            if y.first <= x.last:
                failures.append(
                    f"{x.name or key} ({key}) as {x.ticker} {x.first}..{x.last} and as "
                    f"{y.ticker} {y.first}..{y.last}"
                    + (", both priced by the panel" if x.held and y.held else "")
                )
    return GuardResult("no security under two tickers on one date", checked, tuple(failures))


def guard_ticker_overlap(rows: Sequence[Row]) -> GuardResult:
    """No ticker names two securities on one date, priced or not: the table's own consistency,
    beyond the priced cells the first guard covers."""
    by_ticker: dict[str, list[Row]] = {}
    for row in rows:
        by_ticker.setdefault(row.ticker, []).append(row)
    failures: list[str] = []
    checked = 0
    for ticker, mine in by_ticker.items():
        mine = sorted(mine, key=lambda r: r.first)
        checked += len(mine)
        for x, y in zip(mine, mine[1:]):
            if y.first <= x.last and x.key != y.key:
                failures.append(f"{ticker}: {x.name or x.key} {x.first}..{x.last} and "
                                f"{y.name or y.key} {y.first}..{y.last}")
    return GuardResult("no ticker names two securities on one date", checked, tuple(failures))


@dataclass(frozen=True)
class MemberMap:
    """What each member ticker is on each date it is a member: a security, or why not."""

    mapped: dict[tuple[str, dt.date], tuple[str, str, str]]
    """(ticker, date) -> (security key, the ticker it traded under that day, how): "panel",
    "events" or "presumed" for the row that spans the date; "predecessor" for a reorganisation's
    old identifier; "after its last price" for a security that stopped trading before the
    index removed it; "inside a gap in its prices" for one suspended between two spans."""
    named: dict[tuple[str, dt.date], str]
    """(ticker, date) -> why no security: "unverified: ..." where the run's security is known
    but not on this date, "missing: ..." where no security is."""
    anchors: dict[tuple[str, dt.date, dt.date], str | None]
    """(member ticker, its run's first and last dates) -> the security it names, or None."""


def map_members(
    dates: Sequence[dt.date],
    members: Sequence[frozenset[str]],
    rows: Sequence[Row],
    histories_by_ticker: Mapping[str, History | None],
    tenures_of: Mapping[str, Sequence[Tenure]],
    panel_tickers: set[str],
    links: Sequence["Link"] = (),
) -> MemberMap:
    """Per member ticker and run of membership, the security it names -- whoever traded under
    it on the run's last date -- and then, per date of the run, that security's ticker that day.

    The run's last date is where the ticker means the member: for a current member membership()
    reads today's list, for a removed one the change table's removal, each in that day's
    ticker. For a ticker the panel lacks, Polygon's events name whoever holds it now: the
    member, for a run to the window's end; for an earlier run, if its events have it under the
    ticker then, or start later than the run's end; not if it took the ticker over afterwards.

    Where the security has no row on a date of its run, a reorganisation's predecessor under the
    same ticker -- trading under it to within a halt of the successor's first day -- is the
    member; so is the security itself after its last price, or in a gap between two spans of
    it; so is a predecessor `links` name -- a merger's, which traded under another ticker, as
    WestRock did before Smurfit Westrock. Otherwise the date is named. 'unverified': the run's
    security is known but not on the
    date -- before the first of its events, the start of Polygon's record of it, or before the
    panel's prices under the ticker begin -- so the member may be it under another ticker, or
    another security. 'missing': neither source has the member at all.
    """
    by_ticker: dict[str, list[Row]] = {}
    by_key: dict[str, list[Row]] = {}
    for row in rows:
        by_ticker.setdefault(row.ticker, []).append(row)
        by_key.setdefault(row.key, []).append(row)
    for group in (*by_ticker.values(), *by_key.values()):
        group.sort(key=lambda r: r.first)
    at = {day: i for i, day in enumerate(dates)}
    linked: dict[str, list[Link]] = {}
    for link in links:
        linked.setdefault(link.successor, []).append(link)
    mapped: dict[tuple[str, dt.date], tuple[str, str, str]] = {}
    named: dict[tuple[str, dt.date], str] = {}
    anchors: dict[tuple[str, dt.date, dt.date], str | None] = {}
    for ticker in sorted({t for day in members for t in day}):
        runs: list[list[int]] = []
        for i, day in enumerate(members):
            if ticker in day:
                if runs and runs[-1][-1] == i - 1:
                    runs[-1].append(i)
                else:
                    runs.append([i])
        for run in runs:
            last = dates[run[-1]]
            under = by_ticker.get(ticker, [])
            spanning = [r for r in under if r.first <= last <= r.last]
            earlier = [r for r in under if r.last < last]
            row = spanning[0] if spanning else (earlier[-1] if earlier else None)
            key = row.key if row is not None else None
            history = histories_by_ticker.get(ticker)
            reused = None
            if key is None and ticker not in panel_tickers and history is not None:
                held = [t for t in tenures_of.get(history.key, ()) if t.ticker == ticker]
                if run[-1] == len(dates) - 1 or any(
                    t.start <= last and (t.end is None or t.end >= last) for t in held
                ) or (held and not held[0].known_start and held[0].start > last):
                    key = history.key
                elif held:
                    reused = held[0].start
            anchors[(ticker, dates[run[0]], last)] = key
            mine = by_key.get(key, []) if key is not None else []
            successor = [r for r in mine if r.ticker == ticker]
            tenures = tenures_of.get(key, ()) if key is not None else ()
            first = tenures[0] if tenures else None
            for i in run:
                day = dates[i]
                if key is None:
                    named[(ticker, day)] = (
                        f"missing: whoever holds the ticker now took it on {reused}, after the "
                        f"membership ended, and the panel does not hold it"
                        if reused is not None else
                        "missing: Polygon has no events for the ticker and the panel does not "
                        "hold it -- delisted, and not among the panel's securities"
                        if ticker not in panel_tickers else
                        "missing: no record names anyone under the ticker by the run's end"
                    )
                    continue
                spans = [r for r in mine if r.first <= day <= r.last]
                if spans:
                    mapped[(ticker, day)] = (key, spans[0].ticker, spans[0].source)
                    continue
                if mine and day > mine[-1].last:
                    mapped[(ticker, day)] = (key, mine[-1].ticker, "after its last price")
                    continue
                if any(r.last < day for r in mine) and any(r.first > day for r in mine):
                    # Between two spans of its own: halted, or between one ticker's last price
                    # and the next's first (First Republic, from FRC to FRCB).
                    covering = [t for t in tenures if t.start <= day
                                and (t.end is None or day <= t.end)]
                    previous = [r for r in mine if r.last < day][-1]
                    mapped[(ticker, day)] = (
                        key, covering[0].ticker if covering else previous.ticker,
                        "inside a gap in its prices",
                    )
                    continue
                if successor:
                    joined = successor[0].first
                    prior = [r for r in under if r.key != key and r.first <= day <= r.last
                             and r.last < joined and 0 < at[joined] - at[r.last] <= HALT_DAYS]
                    if prior:
                        mapped[(ticker, day)] = (prior[0].key, ticker, "predecessor")
                        continue
                before = [r for link in linked.get(key, ()) if day <= link.until
                          for r in by_key.get(link.predecessor, ()) if r.first <= day <= r.last]
                if before:
                    mapped[(ticker, day)] = (before[0].key, before[0].ticker, "predecessor")
                    continue
                label = mine[0].name if mine else key
                if (ticker in panel_tickers and successor and day < successor[0].first
                        and (first is None or first.ticker == ticker)):
                    named[(ticker, day)] = (
                        f"unverified: the panel's {ticker} prices begin on {successor[0].first}; "
                        f"before then {label} traded under another ticker or the member was "
                        f"another security, and neither source says which"
                    )
                elif first is not None and day < first.start and not first.known_start:
                    named[(ticker, day)] = (
                        f"unverified: before {first.start}, the first of Polygon's events for "
                        f"{label}, the start of its record there; the member may be it, under "
                        f"{first.ticker} or another ticker, or another security"
                    )
                else:
                    named[(ticker, day)] = f"missing: {label} has no ticker on this date"
    return MemberMap(mapped, named, anchors)


def guard_members(
    dates: Sequence[dt.date], members: Sequence[frozenset[str]], mapping: MemberMap
) -> GuardResult:
    """Every member membership() returns on every date of the window maps to a security or is
    named as missing: between them they account for every (member, date) exactly once."""
    failures: list[str] = []
    checked = 0
    for i, day in enumerate(dates):
        for ticker in sorted(members[i]):
            checked += 1
            hit = int((ticker, day) in mapping.mapped) + int((ticker, day) in mapping.named)
            if hit != 1:
                failures.append(f"{ticker} on {day}: accounted for {hit} times")
    return GuardResult("every member on every date maps to a security or is named", checked,
                       tuple(failures))


# ---------------------------------------------------------------------------------- renames


def renames_from(
    tenures_of: Mapping[str, Sequence[Tenure]], rows: Sequence[Row], keys: Iterable[str]
) -> tuple[Rename, ...]:
    """The dated ticker changes of the securities in `keys`, for `sp500.membership(...,
    renames=...)`: each change from one ticker to another in their credible events, and each
    one the panel's own prices and records show -- Fiserv's return to FISV, which its events
    omit -- or the reference endpoint's records do. `since` is the date the security is first
    known under the old ticker, so a row of the changes table naming that ticker earlier --
    another company's -- is left alone."""
    out: dict[tuple[dt.date, str, str], Rename] = {}
    by_key: dict[str, list[Row]] = {}
    for row in rows:
        by_key.setdefault(row.key, []).append(row)
    for key in keys:
        tenures = list(tenures_of.get(key, ()))
        for before, after in zip(tenures, tenures[1:]):
            if after.known_start and before.ticker != after.ticker:
                out[(after.start, before.ticker, after.ticker)] = Rename(
                    after.start, before.ticker, after.ticker, since=before.start
                )
        mine = sorted(by_key.get(key, ()), key=lambda r: r.first)
        for before, after in zip(mine, mine[1:]):
            if before.ticker == after.ticker or after.source not in ("panel", "reference"):
                continue
            if any((r.old, r.new) == (before.ticker, after.ticker) for r in out.values()):
                continue
            out[(after.first, before.ticker, after.ticker)] = Rename(
                after.first, before.ticker, after.ticker, since=before.first
            )
    return tuple(sorted(out.values(), key=lambda r: (r.date, r.old)))


# ----------------------------------------------------------------------------------- findings


def date_runs(
    dates: Sequence[dt.date], positions: Iterable[int]
) -> list[tuple[dt.date, dt.date, int]]:
    """Consecutive positions of `dates`, as (first date, last date, how many)."""
    runs: list[list[int]] = []
    for i in sorted(set(positions)):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
            runs[-1][2] += 1
        else:
            runs.append([i, i, 1])
    return [(dates[a], dates[b], n) for a, b, n in runs]


def foreign_prices(
    dates: Sequence[dt.date],
    identities: Mapping[str, TickerIdentity],
    mapping: MemberMap,
) -> dict[str, dict[str, list[tuple[dt.date, dt.date, int]]]]:
    """A splice's cost: per member ticker the panel holds, the dates it is an index member and
    the panel's price under it is another security's than the member's -- by that security."""
    at = {day: i for i, day in enumerate(dates)}
    found: dict[str, dict[str, list[int]]] = {}
    for (ticker, day), (key, _, _) in mapping.mapped.items():
        identity = identities.get(ticker)
        if identity is None:
            continue
        owner = identity.owner[at[day]]
        if owner is not None and owner != key:
            found.setdefault(ticker, {}).setdefault(owner, []).append(at[day])
    return {
        ticker: {owner: date_runs(dates, cells) for owner, cells in by_owner.items()}
        for ticker, by_owner in found.items()
    }


def truncations(
    dates: Sequence[dt.date],
    mapping: MemberMap,
    panel_tickers: set[str],
    exclude: set[tuple[str, dt.date]] = frozenset(),
) -> dict[str, dict[tuple[str, str], list[tuple[dt.date, dt.date, int]]]]:
    """Per member ticker the panel holds, the dates the member traded under another ticker --
    so the panel, which asked for this one, has no price for it -- by (security, ticker).
    `exclude` are (ticker, date) pairs not to count: membership that is itself wrong."""
    at = {day: i for i, day in enumerate(dates)}
    found: dict[str, dict[tuple[str, str], list[int]]] = {}
    for (ticker, day), (key, traded, how) in mapping.mapped.items():
        if (ticker, day) in exclude:
            continue
        if ticker in panel_tickers and traded != ticker and how != "predecessor":
            found.setdefault(ticker, {}).setdefault((key, traded), []).append(at[day])
    return {
        ticker: {pair: date_runs(dates, cells) for pair, cells in by_pair.items()}
        for ticker, by_pair in found.items()
    }


def duplicates(
    dates: Sequence[dt.date], members: Sequence[frozenset[str]], mapping: MemberMap
) -> dict[str, list[tuple[tuple[str, ...], tuple[dt.date, dt.date, int]]]]:
    """Per security, the dates two or more member tickers name it at once, by which tickers."""
    found: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for i, day in enumerate(dates):
        by_key: dict[str, list[str]] = {}
        for ticker in members[i]:
            hit = mapping.mapped.get((ticker, day))
            if hit is not None:
                by_key.setdefault(hit[0], []).append(ticker)
        for key, tickers in by_key.items():
            if len(tickers) > 1:
                found.setdefault((key, tuple(sorted(tickers))), []).append(i)
    out: dict[str, list[tuple[tuple[str, ...], tuple[dt.date, dt.date, int]]]] = {}
    for (key, tickers), cells in found.items():
        for run in date_runs(dates, cells):
            out.setdefault(key, []).append((tickers, run))
    return out


def phantoms(
    dates: Sequence[dt.date],
    members: Sequence[frozenset[str]],
    read_through: Sequence[frozenset[str]],
    renames: Sequence[Rename],
) -> tuple[dict[str, list[tuple[dt.date, dt.date, int]]],
           dict[str, list[tuple[dt.date, dt.date, int]]]]:
    """Membership as membership() gives it against membership with the changes table read
    through `renames`, compared in today's names, since the second calls a renamed member by its
    new ticker: per ticker, the dates it is a member only as given -- a phantom, as ECHO is from
    1976 when EchoStar joined as SATS in 2026 -- and the dates a ticker is a member only as read
    through."""
    from .sp500 import todays_ticker

    olds = {r.old for r in renames}
    extra: dict[str, list[int]] = {}
    lacking: dict[str, list[int]] = {}
    for i, day in enumerate(dates):
        today = {t: (todays_ticker(t, day, renames) if t in olds else t) for t in members[i]}
        for ticker, name in today.items():
            if name not in read_through[i]:
                extra.setdefault(ticker, []).append(i)
        for ticker in read_through[i] - set(today.values()):
            lacking.setdefault(ticker, []).append(i)
    return (
        {t: date_runs(dates, cells) for t, cells in extra.items()},
        {t: date_runs(dates, cells) for t, cells in lacking.items()},
    )


def swap_candidates(
    on: dt.date,
    removed: Iterable[str],
    dates: Sequence[dt.date],
    rows: Sequence[Row],
) -> list[tuple[str, str, str | None, dt.date]]:
    """For a member the sources cannot identify, added on `on`: each ticker the change table
    removes the same day whose security's last price under it is within a halt of `on` -- a
    rename the table recorded as a removal and an addition, as it did CDAY's to DAY. Returns
    (ticker, key, name, last price); a candidate, never a resolution."""
    at = {day: i for i, day in enumerate(dates)}
    if on not in at:
        later = [day for day in dates if day >= on]
        if not later:
            return []
        on = later[0]
    out = []
    for ticker in removed:
        mine = [r for r in rows if r.ticker == ticker and r.held and r.last < on]
        if not mine:
            continue
        last = max(mine, key=lambda r: r.last)
        if 0 < at[on] - at[last.last] <= HALT_DAYS:
            out.append((ticker, last.key, last.name, last.last))
    return out


@dataclass(frozen=True)
class PlanRow:
    """One span of the rebuild's fetch: a member security under one ticker."""

    key: str
    name: str | None
    ticker: str
    first: dt.date
    last: dt.date
    held: int
    """Dates of the span the panel already prices under this ticker."""
    member_days: int
    samples: tuple[dt.date, ...]
    """Share-count sample dates inside the span: its first and last day, and every
    `every`-th trading day of the security's own first-to-last span that falls in it."""
    new_samples: int
    """Of `samples`, those with no reference record on disk under this ticker."""
    previous: str | None = None
    """The security the table has under the same ticker before the span, if any. On the
    span's first day a record naming it is the change day's, still carrying the identifier the
    ticker had: Coherent, Inc.'s FIGI on II-VI's first day as COHR."""


def fetch_plan(
    dates: Sequence[dt.date],
    rows: Sequence[Row],
    member_days: Mapping[tuple[str, str], int],
    keys: Iterable[str],
    cached_refs: set[tuple[str, dt.date]],
    every: int = 63,
    origins: Mapping[str, dt.date] | None = None,
) -> list[PlanRow]:
    """What a rebuild must fetch to price `keys` -- the securities that are index members in
    the window -- through the window under the tickers they actually traded under.

    Per security and row of the identity table: the span, and share-count samples taken inside
    the security's own spans only -- each span's first and last day, so a count is never
    carried across a change of identity or ticker, and a grid of every `every` trading days
    across the security's first-to-last span -- each asked under the ticker it traded under on
    the day. `member_days` counts membership per (security, ticker). Samples already on disk
    under that ticker cost nothing.

    The grid runs through the security's first day, or through `origins[key]` where given, in
    both directions: a table corrected after a fetch keeps the grid it was sampled on -- a span
    moved a week later, to its first price, would otherwise move every sample after it."""
    at = {day: i for i, day in enumerate(dates)}
    by_key: dict[str, list[Row]] = {}
    by_ticker: dict[str, list[Row]] = {}
    for row in rows:
        by_key.setdefault(row.key, []).append(row)
        by_ticker.setdefault(row.ticker, []).append(row)
    plan: list[PlanRow] = []
    for key in keys:
        mine = sorted(by_key.get(key, ()), key=lambda r: r.first)
        if not mine:
            continue
        first, last = at[mine[0].first], at[mine[-1].last]
        origin = at.get((origins or {}).get(key), first)
        grid = {i for i in range(first, last + 1) if (i - origin) % every == 0}
        for row in mine:
            lo, hi = at[row.first], at[row.last]
            chosen = sorted({lo, hi} | {i for i in grid if lo <= i <= hi})
            samples = tuple(dates[i] for i in chosen)
            earlier = [r for r in by_ticker.get(row.ticker, ())
                       if r.key != key and r.last < row.first]
            plan.append(PlanRow(
                key=key, name=row.name, ticker=row.ticker, first=row.first, last=row.last,
                held=row.held, member_days=member_days.get((key, row.ticker), 0),
                samples=samples,
                new_samples=sum((row.ticker, day) not in cached_refs for day in samples),
                previous=max(earlier, key=lambda r: r.last).key if earlier else None,
            ))
    return plan


def applicable_renames(
    renames: Sequence[Rename], current: set[str], changes: Sequence
) -> tuple[tuple[Rename, ...], tuple[tuple[Rename, str], ...]]:
    """Of `renames`, those the changes table should be read through, and why each other is not.

    The walk starts from today's list, so a rename helps it only where the security still
    trades, in that list, under the name the rename leads to: ECHO, once SATS. A security that
    left the index before or soon after its rename -- GPS, The Gap since 2024; FBHS, removed
    under that name four days after it became FBIN -- is named by its removal, and reading its
    earlier rows in a later name splits it in two. Nor is a rename the table already records as
    a removal and an addition within a halt of its date, as it does WLTW's to WTW: read through
    as well, it is counted twice. `current` is the list as membership() reads it on its last
    day; `changes` are the snapshot's."""
    kept: list[Rename] = []
    skipped: list[tuple[Rename, str]] = []
    for rename in renames:
        final = _final_name(rename, renames)
        if final not in current:
            skipped.append((rename, f"{final} is not in today's list"))
            continue
        swap = next(
            (c for c in changes
             if abs((c.date - rename.date).days) <= HALT_DAYS and rename.old in c.removed
             and rename.new in c.added),
            None,
        )
        if swap is not None:
            skipped.append((rename, f"the changes table records it as a swap on {swap.date}"))
            continue
        kept.append(rename)
    return tuple(kept), tuple(skipped)


def _final_name(rename: Rename, renames: Sequence[Rename]) -> str:
    """Where a rename's chain ends: the name the security trades under today."""
    from .sp500 import todays_ticker

    return todays_ticker(rename.old, rename.date - dt.timedelta(days=1), renames)


# --------------------------------------------------------- the reference endpoint's word


@dataclass(frozen=True)
class Hypothesis:
    """A candidate for member-days neither source could name, from the public record, put to
    the reference endpoint: nothing enters the table on its say-so, only on the endpoint's."""

    member: str
    """The member ticker whose named member-days it is for."""
    ask: tuple[str, ...]
    """The tickers to ask the endpoint under, on those dates: the member's own, or the one
    the hypothesis says the member traded under."""
    statement: str
    """As it was put: "CDAY for DAY", "Bunge Limited for BG"."""
    same_as: str | None = None
    """For "the same security under another name": the ticker whose security in the table
    the endpoint's must be -- CDAY's, for DAY."""
    called: str | None = None
    """For a predecessor named by company, a name the endpoint's security must carry."""


@dataclass(frozen=True)
class Link:
    """The member before its successor, under another identity: a merger's predecessor that
    traded under another ticker (WestRock, before Smurfit Westrock), or a reorganisation's."""

    successor: str
    predecessor: str
    until: dt.date
    evidence: str
    kind: str = "merger"
    """"merger": two securities, the member one and then the other (WestRock, then Smurfit
    Westrock). "reorganisation": one security under a new identifier, its prices one series
    (Bunge Limited, then Bunge Global SA, under BG)."""


@dataclass(frozen=True)
class Resolution:
    """What the reference endpoint said to one hypothesis."""

    hypothesis: Hypothesis
    named: tuple[dt.date, dt.date, int] | None
    """The named member-days it was for -- first, last, how many -- or None if none were."""
    rows: tuple[Row, ...]
    """What the endpoint confirms and the table takes: who traded under each ticker asked,
    and over which trading days."""
    link: Link | None
    confirmed: bool
    """Whether the hypothesis, as stated, is what the endpoint says."""
    covered: int
    """Of the named member-days, those the rows and link taken account for."""
    verdict: str
    asked: int
    """Reference lookups it took, cached or not."""


def _settled(found: Sequence[Row], dates: Sequence[dt.date]) -> list[Row]:
    """One security's confirmed rows under two tickers, where they overlap: the later
    ticker's row takes the overlap, the earlier ending the day before it begins."""
    at = {day: i for i, day in enumerate(dates)}
    out: list[Row] = []
    for key in dict.fromkeys(r.key for r in found):
        group = sorted((r for r in found if r.key == key), key=lambda r: (r.first, r.last))
        for k, row in enumerate(group):
            later = [r for r in group[k + 1:] if r.ticker != row.ticker and r.first <= row.last]
            if later and at[later[0].first] > at[row.first]:
                stop = dates[at[later[0].first] - 1]
                row = Row(row.key, row.figi, row.cik, row.name, row.kind, row.ticker, row.first,
                          stop, 0, row.source, row.starts,
                          f"{row.ends}; to the day before it is {later[0].ticker}, "
                          f"{later[0].first}")
            out.append(row)
    return out


def _trimmed(
    taken: Sequence[Row], by_key: Mapping[str, Sequence[Row]], dates: Sequence[dt.date]
) -> list[Row]:
    """Confirmed rows less the trading days their security already has a row on: Super
    Micro's SMCI from the window's start, against the events' SMCI from its 2024 split."""
    out: list[Row] = []
    for row in taken:
        held = {
            i for other in by_key.get(row.key, ())
            for i, day in enumerate(dates) if other.first <= day <= other.last
        }
        span = [i for i, day in enumerate(dates) if row.first <= day <= row.last and i not in held]
        runs: list[list[int]] = []
        for i in span:
            if runs and i == runs[-1][1] + 1:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        for lo, hi in runs:
            out.append(Row(
                row.key, row.figi, row.cik, row.name, row.kind, row.ticker, dates[lo], dates[hi],
                0, row.source,
                row.starts if dates[lo] == row.first else
                f"{row.starts}; from the day after its other row ends",
                row.ends if dates[hi] == row.last else
                f"{row.ends}; to the day before its other row begins",
            ))
    return out


def resolve_named(
    dates: Sequence[dt.date],
    mapping: MemberMap,
    rows: Sequence[Row],
    hypotheses: Sequence[Hypothesis],
    lookup: Callable[[str, dt.date], Record | None],
) -> list[Resolution]:
    """Put each hypothesis to the reference endpoint, through `lookup` -- the record under a
    ticker on a date, or None where it has none -- and keep only what it confirms.

    Each ticker asked is looked up on the first and last of the member's named dates. For each
    security it names there, the trading days it traded under that ticker are found by a
    binary search on the endpoint's own answers, out to the window's ends -- the whole window,
    so a joiner keeps its history before joining and a leaver its record after. What the
    security is to the member decides what is taken:

    - the member's run names no security at all (DAY, KVUE, WBD): the endpoint's, asked under
      the member's own ticker, is the member;
    - it is the run's security (Corpay's FIGI under FLT; Super Micro's under SMCI before its
      split): the same security, under another ticker or on earlier dates;
    - it is another: a predecessor (WestRock for Smurfit Westrock, Paramount for Skydance,
      Bunge Limited for Bunge Global), taken only if it traded to within a halt of the
      successor's first day.

    The hypothesis is judged apart: "CDAY for DAY" holds only if DAY's security is CDAY's in
    the table. Rows the endpoint confirms enter even when the hypothesis does not -- WBD's own
    security, if not DISCA's -- and named member-days nothing accounts for stay named.
    """
    at = {day: i for i, day in enumerate(dates)}
    n = len(dates) - 1
    by_key: dict[str, list[Row]] = {}
    by_ticker: dict[str, list[Row]] = {}
    for row in rows:
        by_key.setdefault(row.key, []).append(row)
        by_ticker.setdefault(row.ticker, []).append(row)
    out: list[Resolution] = []
    for hypothesis in hypotheses:
        days = sorted(day for (ticker, day) in mapping.named if ticker == hypothesis.member)
        if not days:
            out.append(Resolution(hypothesis, None, (), None, False, 0,
                                  "nothing under the ticker is named", 0))
            continue
        first, last = days[0], days[-1]
        anchor = next((key for (ticker, a, b), key in mapping.anchors.items()
                       if ticker == hypothesis.member and a <= first <= b), None)
        answers: dict[tuple[str, int], Record | None] = {}

        def ask(ticker: str, i: int) -> Record | None:
            if (ticker, i) not in answers:
                answers[(ticker, i)] = lookup(ticker, dates[i])
            return answers[(ticker, i)]

        def names(ticker: str, i: int, key: str) -> bool:
            record = ask(ticker, i)
            return record is not None and record.key == key

        def edge(ticker: str, key: str, inside: int, outside: int) -> int:
            """The position nearest `outside` at which `ticker` still names `key`, between a
            position where it does and one where it does not."""
            while abs(outside - inside) > 1:
                middle = (inside + outside) // 2
                if names(ticker, middle, key):
                    inside = middle
                else:
                    outside = middle
            return inside

        found: list[Row] = []
        for ticker in hypothesis.ask:
            probes = (at[first], at[last])
            seen = {}
            for i in probes:
                record = ask(ticker, i)
                if record is not None:
                    seen.setdefault(record.key, (record, []))[1].append(i)
            for key, (record, hits) in seen.items():
                low, high = min(hits), max(hits)
                if low == 0 or names(ticker, 0, key):
                    start = 0
                else:
                    outside = max([i for i in probes if i < low] or [0])
                    start = edge(ticker, key, low, outside)
                if high == n or names(ticker, n, key):
                    stop = n
                else:
                    outside = min([i for i in probes if i > high] or [n])
                    stop = edge(ticker, key, high, outside)
                note = (f"reference: {ticker} names {record.name} ({key}) from {dates[start]} "
                        f"to {dates[stop]}")
                found.append(Row(
                    key, record.figi, record.cik, record.name, record.kind, ticker,
                    dates[start], dates[stop], 0, "reference",
                    note + ("" if start == 0 else f"; not on {dates[start - 1]}"),
                    note + ("" if stop == n else f"; not on {dates[stop + 1]}"),
                ))
        # On the day a ticker changes the endpoint can answer under both, old and new: the new
        # one takes that day, as it does in the events (VIAC to PARA on 2022-02-17).
        found = _settled(found, dates)
        taken: list[Row] = []
        link = None
        mine = sorted(by_key.get(anchor, ()), key=lambda r: r.first) if anchor else []
        for key in dict.fromkeys(r.key for r in found):
            group = [r for r in found if r.key == key]
            if anchor is None or key == anchor:
                taken.extend(group)
                continue
            if not mine:
                continue
            # A predecessor ends the day before its successor's first, where the endpoint's
            # records run on to it or a halt past it -- Paramount under PARA on 2025-08-07,
            # Skydance's first day -- and not otherwise.
            start = at[mine[0].first]
            if at[max(r.last for r in group)] - start > HALT_DAYS:
                continue
            cut = [r for r in group if at[r.first] < start]
            cut = [r if at[r.last] < start else Row(
                r.key, r.figi, r.cik, r.name, r.kind, r.ticker, r.first, dates[start - 1], 0,
                r.source, r.starts, f"{r.ends}; to the day before {mine[0].name}'s first, "
                f"{mine[0].first}") for r in cut]
            if not cut:
                continue
            latest = max(cut, key=lambda r: r.last)
            last_of = latest.last
            if 0 < start - at[last_of] <= HALT_DAYS:
                taken.extend(cut)
                link = Link(anchor, key, last_of,
                            f"{latest.name} traded as {latest.ticker} to {last_of}, within a "
                            f"halt of {mine[0].name}'s first day under {mine[0].ticker}, "
                            f"{mine[0].first}")
        taken = _trimmed(taken, by_key, dates)
        covered = sum(
            1 for day in days
            if any(r.first <= day <= r.last for r in taken)
            or (anchor is None and any(r.last < day for r in taken))
        )
        securities = sorted({r.key for r in found})
        listed = ", ".join(securities) or "nothing"
        if hypothesis.same_as is not None:
            expected = sorted({r.key for r in by_ticker.get(hypothesis.same_as, ())})
            confirmed = bool(securities) and set(securities) <= set(expected)
            verdict = (
                f"confirmed: the endpoint names {listed}, {hypothesis.same_as}'s security in "
                f"the table" if confirmed else
                f"not confirmed: the endpoint names {listed}, and {hypothesis.same_as}'s "
                f"security in the table is {', '.join(expected) or 'not there'}"
            )
        elif hypothesis.called is not None:
            confirmed = bool(found) and all(
                hypothesis.called.lower() in (r.name or "").lower() for r in found
            )
            called = ", ".join(sorted({r.name or r.key for r in found})) or "nothing"
            verdict = ("confirmed" if confirmed else "not confirmed") + (
                f": the endpoint names {called}"
            )
        else:
            confirmed = bool(taken) and covered == len(days)
            verdict = ("confirmed: " if confirmed else "not confirmed: ") + (
                "; ".join(f"{r.ticker} names {r.name} from {r.first} to {r.last}"
                          for r in found) or "the endpoint names nothing"
            )
        out.append(Resolution(
            hypothesis, (first, last, len(days)), tuple(taken), link, confirmed, covered,
            verdict, len(answers),
        ))
    return out


# --------------------------------------------------------------- the rebuild's evidence


def read_answers(cache: Path | str, ticker: str) -> list[tuple[dt.date, Record | None]]:
    """Every reference answer on disk under a ticker, oldest first: a `Record`, or None where
    the endpoint had none -- the rebuild's share samples among them. No requests."""
    prefix = f"ref-{ticker}-"
    out: list[tuple[dt.date, Record | None]] = []
    for path in Path(cache).glob(f"{prefix}*.json"):
        on = _date(path.stem[len(prefix):])
        if on is None:
            continue
        results = json.loads(path.read_text(encoding="utf-8")).get("results") or {}
        out.append((on, Record(
            ticker=ticker, on=on, figi=results.get("composite_figi") or None,
            cik=results.get("cik") or None, name=results.get("name") or None,
            kind=results.get("type") or None, list_date=_date(results.get("list_date")),
            share_class=results.get("share_class_figi") or None,
        ) if results else None))
    return sorted(out, key=lambda pair: pair[0])


@dataclass(frozen=True)
class Correction:
    """One change the rebuild's evidence makes to the table, and why."""

    rule: str
    """"first price": a span starts at its first price under its ticker, not at the first of
    its security's events; "unpriced": a span with no price under its ticker is no span;
    "refuted": every record under its ticker inside a span names another security, and none
    its own; "reorganisation": a new identifier under one ticker, the boundary dated by the
    identifier the table keys on; "confirmed": a span the reference endpoint names at both
    ends."""
    key: str
    ticker: str
    removed: tuple[Row, ...]
    added: tuple[Row, ...]
    detail: str
    asked: int = 0
    """Reference lookups it took, cached or not."""
    link: Link | None = None


def start_at_first_price(
    rows: Sequence[Row], priced: Callable[[str], Sequence[dt.date] | None]
) -> tuple[list[Row], list[Correction]]:
    """Every span the panel's prices did not draw starts at its first price under its ticker.

    A span from Polygon's events starts at the first of them, which for a spin-off is its
    listing date: GE HealthCare's at 2022-12-15, its first regular-way price on 2023-01-04, and
    no reference record before then to say who traded. A span with no price under its ticker
    at all is dropped -- a tenure Polygon records and nobody traded under, First Republic's
    S600 and Constellation's CEGV. `priced(ticker)` is the ticker's trading days, ascending, or
    None where its prices were not fetched, which leaves its rows as they are; the panel's own
    rows begin at a price already."""
    out: list[Row] = []
    corrections: list[Correction] = []
    for row in rows:
        days = None if row.source == "panel" else priced(row.ticker)
        if days is None:
            out.append(row)
            continue
        inside = [day for day in days if row.first <= day <= row.last]
        label = row.name or row.key
        if not inside:
            corrections.append(Correction(
                "unpriced", row.key, row.ticker, (row,), (),
                f"{label} under {row.ticker} {row.first}..{row.last}: no price under the ticker "
                f"inside the span, which began at {row.starts}",
            ))
            continue
        if inside[0] > row.first:
            moved = replace(row, first=inside[0],
                            starts=f"its first price under {row.ticker}, {inside[0]} (before it, "
                                   f"{row.starts})")
            corrections.append(Correction(
                "first price", row.key, row.ticker, (row,), (moved,),
                f"{label} under {row.ticker}: from {row.first} to {inside[0]}, its first price "
                f"there",
            ))
            row = moved
        out.append(row)
    return out, corrections


def drop_refuted(
    rows: Sequence[Row],
    answers_of: Callable[[str], Sequence[tuple[dt.date, Record | None]]],
) -> tuple[list[Row], list[Correction]]:
    """Every span the records refute is dropped: one the panel's prices did not draw, under
    whose ticker every record inside it names another identified security, and none its own.
    Polygon's events give CRH plc the ticker CRHC from 2021-09-23, and each record under CRHC
    there names Cohn Robbins Holdings, a SPAC, or nobody. `answers_of(ticker)` is every answer
    on disk under the ticker (`read_answers`)."""
    out: list[Row] = []
    corrections: list[Correction] = []
    for row in rows:
        if row.source == "panel":
            out.append(row)
            continue
        inside = [(day, r) for day, r in answers_of(row.ticker) if row.first <= day <= row.last]
        own = [day for day, r in inside if r is not None and r.identified and r.key == row.key]
        other = [(day, r) for day, r in inside
                 if r is not None and r.identified and r.key != row.key]
        if own or not other:
            out.append(row)
            continue
        named: dict[str, list] = {}
        for day, record in other:
            entry = named.setdefault(record.key, [record.name, day, day, 0])
            entry[2], entry[3] = day, entry[3] + 1
        empty = sum(1 for _, r in inside if r is None)
        corrections.append(Correction(
            "refuted", row.key, row.ticker, (row,), (),
            f"{row.name or row.key} under {row.ticker} {row.first}..{row.last}, from "
            f"{row.starts}: " + "; ".join(
                f"{n} record(s) name {name} ({key}), {a}..{b}"
                for key, (name, a, b, n) in named.items())
            + (f"; {empty} answer(s) no record" if empty else "") + "; none names it",
        ))
    return out, corrections


@dataclass(frozen=True)
class Reorganisation:
    """A new identifier under one ticker, put to the reference endpoint to be dated. The
    boundary falls between the last day its records name `before` and the first they name
    `after`: the identifier the table keys on decides, not the name, which can change first --
    Bunge Global SA is named under Bunge Limited's CIK from 2023-11-01 to 2023-11-07."""

    ticker: str
    before: str
    after: str
    statement: str


def reorganise(
    dates: Sequence[dt.date],
    rows: Sequence[Row],
    spec: Reorganisation,
    lookup: Callable[[str, dt.date], Record | None],
    known: Sequence[tuple[dt.date, Record | None]] = (),
) -> tuple[list[Row], Correction]:
    """Date `spec`'s boundary by bisection on the endpoint's answers under its ticker, between
    the latest day `known` -- the answers on disk -- name `before` and the earliest after it
    they name `after`. `before`'s span under the ticker then ends at the boundary, and starts
    with `after`'s if the table has none for it (Vylor's first FIGI); `after`'s starts the next
    day. One security, its prices one series: a `Link` of kind "reorganisation" joins the two.
    `held` is kept as it was, the spans being of tickers the panel does not hold.

    Raises ValueError where the answers cannot bracket the boundary, or a day inside the
    bracket names neither."""
    at = {day: i for i, day in enumerate(dates)}
    answers: dict[int, Record | None] = {at[day]: r for day, r in known if day in at}
    asked = 0

    def names(i: int, key: str) -> bool:
        record = answers.get(i)
        return record is not None and record.identified and record.key == key

    lo = max((i for i in answers if names(i, spec.before)), default=None)
    hi = None if lo is None else min(
        (i for i in answers if i > lo and names(i, spec.after)), default=None)
    if lo is None or hi is None:
        raise ValueError(f"{spec.statement}: no answer under {spec.ticker} names {spec.before} "
                         f"and a later one {spec.after}, so the boundary cannot be bracketed")
    while hi - lo > 1:
        middle = (lo + hi) // 2
        answers[middle] = lookup(spec.ticker, dates[middle])
        asked += 1
        if names(middle, spec.before):
            lo = middle
        elif names(middle, spec.after):
            hi = middle
        else:
            found = answers[middle]
            raise ValueError(f"{spec.statement}: {spec.ticker} on {dates[middle]} names "
                             f"{found.key if found is not None else 'nobody'}, neither identifier")
    last_before, first_after = dates[lo], dates[hi]
    evidence = (f"{spec.ticker} names {spec.before} to {last_before} and {spec.after} from "
                f"{first_after} (the reference endpoint)")
    after_rows = [r for r in rows if r.key == spec.after and r.ticker == spec.ticker
                  and r.first <= first_after <= r.last]
    if len(after_rows) != 1:
        raise ValueError(f"{spec.statement}: {len(after_rows)} rows of {spec.after} under "
                         f"{spec.ticker} span {first_after}")
    successor = after_rows[0]
    before_rows = [r for r in rows if r.key == spec.before and r.ticker == spec.ticker
                   and r.first <= last_before]
    removed, added = [successor], []
    if before_rows:
        predecessor = max(before_rows, key=lambda r: r.first)
        removed.append(predecessor)
        added.append(replace(predecessor, last=last_before,
                             ends=f"the reorganisation: {evidence}"))
    else:
        record = answers[lo]
        added.append(Row(spec.before, record.figi, record.cik, record.name, record.kind,
                         spec.ticker, successor.first, last_before, 0, "reference",
                         successor.starts, f"the reorganisation: {evidence}"))
    added.append(replace(successor, first=first_after, starts=f"the reorganisation: {evidence}"))
    kept = [r for r in rows if not any(r is gone for gone in removed)]
    link = Link(spec.after, spec.before, last_before, evidence, kind="reorganisation")
    return sorted([*kept, *added], key=lambda r: (r.key, r.first, r.ticker)), Correction(
        "reorganisation", spec.after, spec.ticker, tuple(removed), tuple(added),
        f"{spec.statement}: {evidence}", asked, link,
    )


@dataclass(frozen=True)
class Confirmation:
    """A span the table lacks, put to the reference endpoint at both ends: a member security's
    history under an earlier ticker its events do not record (II-VI's, as IIVI, before it took
    COHR as Coherent Corp.), or who held a ticker before a member security took it (Coherent,
    Inc., under COHR)."""

    ticker: str
    first: dt.date
    last: dt.date
    statement: str
    expect: str | None = None
    """The security the endpoint must name, where the statement says which."""
    last_price: dt.date | None = None
    """The span's last price, where the endpoint names the security on a last day it did not
    trade: Coherent, Inc. under COHR on 2022-07-01, the day II-VI's purchase of it closed, its
    last price on 2022-06-30. None: the span's last day."""


def confirm_span(
    dates: Sequence[dt.date],
    rows: Sequence[Row],
    spec: Confirmation,
    lookup: Callable[[str, dt.date], Record | None],
    priced: Callable[[str], Sequence[dt.date] | None],
) -> tuple[list[Row], Correction]:
    """Add `spec`'s span if the endpoint names one identified security under its ticker on its
    first and its last day -- `spec.expect`, where given -- and the ticker's prices run from
    the first to its last price, `spec.last_price` or else the last day, without a gap longer
    than a halt, so nobody else traded between, and none after it inside the span. Raises
    ValueError otherwise, saying why; and where the span would put its security under two
    tickers on one date, or its ticker under two securities."""
    at = {day: i for i, day in enumerate(dates)}
    ends = [lookup(spec.ticker, spec.first), lookup(spec.ticker, spec.last)]
    keys = {r.key if r is not None and r.identified else None for r in ends}
    if None in keys or len(keys) != 1 or (spec.expect is not None and keys != {spec.expect}):
        named = ", ".join(f"{day}: {r.name} ({r.key})" if r is not None else f"{day}: nobody"
                          for day, r in zip((spec.first, spec.last), ends))
        raise ValueError(f"{spec.statement}: not confirmed -- {spec.ticker} names {named}")
    record = ends[1]
    days = priced(spec.ticker) or ()
    final = spec.last_price or spec.last
    inside = [at[day] for day in days if spec.first <= day <= spec.last and day in at]
    if not spec.first <= final <= spec.last or not inside or dates[inside[0]] != spec.first \
            or dates[inside[-1]] != final \
            or any(b - a - 1 >= HALT_DAYS for a, b in zip(inside, inside[1:])):
        raise ValueError(f"{spec.statement}: {spec.ticker}'s prices do not run unbroken from "
                         f"{spec.first} to {final}"
                         + (f", and none after it to {spec.last}" if final != spec.last else ""))
    clash = [r for r in rows if r.first <= spec.last and spec.first <= r.last
             and (r.key == record.key or r.ticker == spec.ticker)]
    if clash:
        raise ValueError(f"{spec.statement}: the span overlaps " + "; ".join(
            f"{r.name or r.key} as {r.ticker} {r.first}..{r.last}" for r in clash))
    between = ("its prices unbroken between" if final == spec.last else
               f"its prices unbroken from the first to its last price, on {final}")
    note = (f"reference: {spec.ticker} names {record.name} ({record.key}) on {spec.first} and "
            f"{spec.last}, {between}")
    row = Row(record.key, record.figi, record.cik, record.name, record.kind, spec.ticker,
              spec.first, spec.last, 0, "reference", note, note)
    return sorted([*rows, row], key=lambda r: (r.key, r.first, r.ticker)), Correction(
        "confirmed", record.key, spec.ticker, (), (row,), f"{spec.statement}: {note}", 2,
    )
