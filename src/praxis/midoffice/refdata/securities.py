"""The S&P 500 panel on securities, built from the rebuild's fetch by the identity table.

The old panel's columns are tickers -- the index on its first day, each asked for its prices
over the whole window -- and a ticker is not a security (`identity`). This panel's columns are
the securities the identity table finds among the index's members on some date of the window,
each a share class keyed by composite FIGI, else CIK. What each column holds, and from where:

**Prices, stitched.** A column takes, on each trading day, the bars of the ticker its security
traded under that day, from the plan's spans (`rebuild.read_plan`). A splice -- one ticker, two
unrelated securities, as BNY, DOC and META were -- is two columns, each cut to its own spans. A
reorganisation -- a new identifier under one ticker, the prices unbroken (BLK, XOM, Bunge,
Vylor) -- is one column with one return series, keyed by its successor, the predecessor
merged in. Within a run of one ticker's bars the column's price is that ticker's adjusted close,
times one scale for the run: Polygon adjusts each ticker's history for that ticker's events, so
two tickers' adjusted closes need not share a basis. Across a change of ticker the return is
the raw close's -- the first day's under the new ticker over the last day's under the old --
times the ratio of any split event of the security's between the two, and each earlier run's
scale makes its prices continue the later run's: the last run's prices are its ticker's
adjusted closes exactly (`Bridge`).

**Total returns.** The adjusted bars carry split events alone, a spin-off among them, and no
distributions. A total-return index adds each cash distribution under the security's tickers
inside its spans -- regular and special alike -- on its ex-date, put on the prices' split basis
by the ex-date's ratio of the column's price to its raw close: (P_t + D_t) / P_s for the
previous priced day s. An exact duplicate -- every field of Polygon's record but its id, TEL's
of 2024-11-22 -- is kept once. The market series, SPY, has both returns the same way.

**Volumes.** The traded ones: the raw bars' share volume, on the basis of the traded price and
of a point-in-time share count, so turnover and dollar volume are what changed hands. The
adjusted bars' volumes are kept beside them, put on the stitched prices' basis, for the
comparison with the old panel, which stored those.

**Share counts.** Per security, from its share samples -- each span's first and last day and a
63-day grid, inside its own spans (`identity.fetch_plan`) -- and, where its first span starts
without one, the first record with a count that a search of its first quarter finds
(`search_first_count`). Each count is carried forward to the next, never back: the days before
a security's first count have none. A split the gate classified as one (`rebuild.check_splits`)
multiplies the carried count on its execution date; a sample on or after it still at the
pre-split count is stale and does not reset it, and the first that follows the ratio does. A
price-only adjustment is not applied. A count carries across a security's own spans and across
a reorganisation, the successor's first sample with a count resetting it: Carnival's 1.1054
arrives with its first, OKE's and XOM's successors never carry one, so their predecessors'
counts run 1:1 to the window's end, flagged (`Carry`).

**Membership**, the identity table's: membership() read through the renames and mapped to
securities, a reorganisation's member-days under its successor's column.

**Guards on what was built.** Each (ticker, date) price feeds one column; no column takes two
tickers on one date; every member-day is a column's, on a day inside its spans; every bar under
a span's ticker inside the span is in its column. The split guard, exact: no day on or after a
split's execution date carries a pre-split count -- any count the column carried before it, or
a stale sample's -- and its execution date carries the count before it times the ratio, unless
a sample that follows the ratio resets it there (`split_guard`). And a standing check, not a
source: wherever a record carries Polygon's weighted share count, its ratio to the count the
panel carries that day, flagged beyond 10% (`weighted_check`).

Pure functions over in-memory inputs; `scripts/build_sp500_securities.py` reads the cache.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .identity import Record
from .polygon import Bar, Dividend, Split
from .rebuild import LARGEST_TERM, TOLERANCE, Span, SplitCheck, classify

Bars = Callable[[str, str], Sequence[Bar]]
"""A ticker's daily bars, "adj" or "raw": read as needed, one column's at a time."""


# ----------------------------------------------------------------------------- columns


@dataclass(frozen=True)
class Column:
    """One security of the panel: its spans, oldest first, with any reorganisation's
    predecessors merged in under the successor's key."""

    key: str
    keys: tuple[str, ...]
    """Every plan key the column holds, oldest first: a reorganisation's predecessor, then its
    successor, whose key is the column's."""
    name: str
    spans: tuple[Span, ...]

    @property
    def label(self) -> str:
        """The ticker of its last span: what the security is called at the window's end, or
        when it left."""
        return self.spans[-1].ticker


def columns_of(spans: Sequence[Span], reorganisations: Mapping[str, str]) -> list[Column]:
    """The panel's columns from the plan's spans: one per security, a reorganisation's
    predecessor merged into its successor (`reorganisations`: predecessor key to successor
    key). Chains are followed to their end. Sorted by label, then key."""
    def last(key: str) -> str:
        seen = set()
        while key in reorganisations:
            if key in seen:
                raise ValueError(f"the reorganisations loop at {key}")
            seen.add(key)
            key = reorganisations[key]
        return key

    grouped: dict[str, list[Span]] = {}
    for span in spans:
        grouped.setdefault(last(span.key), []).append(span)
    out = []
    for key, mine in grouped.items():
        mine.sort(key=lambda s: (s.first, s.ticker))
        keys = tuple(dict.fromkeys(s.key for s in mine))
        if keys[-1] != key:
            raise ValueError(f"the column {key}'s last span is {keys[-1]}'s, not its own")
        out.append(Column(key, keys, mine[-1].name, tuple(mine)))
    return sorted(out, key=lambda c: (c.label, c.key))


# ----------------------------------------------------------------------------- the search


@dataclass(frozen=True)
class FirstCount:
    """The search of a span that starts without a share count for its first record with one
    that the span's next sample confirms."""

    span: Span
    bracket: tuple[dt.date, dt.date] | None
    """The span's first sample, without a count, and its first sample with one -- its next
    sample, once a record with a count is found between; None where no sample in the span
    carries one -- nothing to search between."""
    found: tuple[dt.date, float] | None
    """The record the count carries from, its date and count: the first with a count that the
    next sample confirms within the tolerance."""
    probes: tuple[tuple[dt.date, float | None, str | None], ...] = ()
    """Each record asked for, in order: its date, its count, the security it names."""
    failure: str | None = None
    """A probe naming another security than the span's: the search stops on it."""
    unconfirmed: tuple[dt.date, float] | None = None
    """The first record with a count, where the next sample does not confirm it: a figure
    Polygon carries until it corrects the record, never carried here (midoffice08, ruling 1).
    The days before `found` have no count."""
    next_count: float | None = None
    """The next sample's count, which confirms a first count or does not."""

    @property
    def detail(self) -> str:
        if self.failure:
            return self.failure
        if self.bracket is None:
            return (f"{self.span.ticker}: no sample in {self.span.first}..{self.span.last} "
                    f"carries a share-class count, so there is nothing to search between")
        a, b = self.bracket
        assert self.found is not None and self.next_count
        first = self.unconfirmed or self.found
        text = (f"{self.span.ticker}: between {a}, without a count, and {b}, {self.next_count:,.0f}"
                f", the first record with a count is {first[0]}'s, {first[1]:,.0f}, "
                f"{first[1] / self.next_count:.3f} of the next sample's")
        if self.unconfirmed is None:
            return text + f", confirmed; in {len(self.probes)} lookup(s)"
        return (text + f", unconfirmed; the first the next sample confirms is "
                f"{self.found[0]}'s, {self.found[1]:,.0f}; in {len(self.probes)} lookup(s)")


def search_first_count(
    span: Span,
    days: Sequence[dt.date],
    bracket: tuple[dt.date, float] | None,
    lookup: Callable[[str, dt.date], tuple[Record | None, float | None]],
    names: Callable[[Record], bool],
    *,
    confirm: bool = True,
    tolerance: float = TOLERANCE,
) -> FirstCount:
    """The first record with a share count under the span's ticker that the span's next sample
    confirms, by bisection of the trading days `days` between the span's first sample, which
    has none, and `bracket`, its first sample with one (date and count).

    The first bisection finds the first record with any count, assuming that a record once
    carrying a count goes on carrying one. A first count stands only if the next sample --
    `bracket` -- confirms it within `tolerance`; otherwise a second bisection, between it and
    the next sample, finds the first record that the next sample confirms, assuming that one
    confirmed stays confirmed, and the days before that one have no count (midoffice08, ruling
    1). `confirm=False` keeps the first record with any count, midoffice08's rule.

    `lookup(ticker, day)` is the reference endpoint's record on the day and its share-class
    count. Every record asked for must name the span's security (`names`); one naming another
    stops the search. Each lookup is one request at most: the client caches."""
    if bracket is None:
        return FirstCount(span, None, None)
    first = span.samples[0]
    end = (first, bracket[0])
    probes: list[tuple[dt.date, float | None, str | None]] = []

    def bisect(lo: int, hi: int, accepts) -> tuple[int, tuple[dt.date, float], str | None]:
        """The first index after `lo`, up to `hi` -- the next sample, which accepts -- whose
        record's count `accepts`; that record; or a failure."""
        found = (bracket[0], float(bracket[1]))
        while hi - lo > 1:
            mid = (lo + hi) // 2
            record, count = lookup(span.ticker, days[mid])
            probes.append((days[mid], count, record.key if record is not None else None))
            if record is not None and record.identified and not names(record):
                return hi, found, (f"{span.ticker} on {days[mid]} names {record.name} "
                                   f"({record.key}), not {span.name} ({span.key}): the search "
                                   f"stops")
            if count and accepts(float(count)):
                hi, found = mid, (days[mid], float(count))
            else:
                lo = mid
        return hi, found, None

    top = days.index(bracket[0])
    at, found, failure = bisect(days.index(first), top, lambda count: True)
    if failure is not None:
        return FirstCount(span, end, None, tuple(probes), failure, next_count=bracket[1])
    if not confirm or abs(found[1] / bracket[1] - 1.0) <= tolerance:
        return FirstCount(span, end, found, tuple(probes), next_count=bracket[1])
    unconfirmed = found
    _, found, failure = bisect(at, top, lambda count: abs(count / bracket[1] - 1.0) <= tolerance)
    if failure is not None:
        return FirstCount(span, end, None, tuple(probes), failure, unconfirmed, bracket[1])
    return FirstCount(span, end, found, tuple(probes), None, unconfirmed, bracket[1])


# ----------------------------------------------------------------------------- repairs


@dataclass(frozen=True)
class Repair:
    """A sample the isolated-error rule finds (midoffice08, ruling 2): its two samples before
    and its two after agree with each other within the tolerance, it differs from each of them
    beyond it, and no split the gate applies falls between them. It is not carried, and the count
    before it carries through -- unless its own record's weighted count confirms it within the
    tolerance: two fields of one record agreeing, it is spared (midoffice09, ruling 1)."""

    key: str
    """Its column's key."""
    sample_key: str
    """The plan key whose sample it is: the column's, or a reorganisation's predecessor's."""
    label: str
    ticker: str
    day: dt.date
    count: float
    neighbours: tuple[tuple[dt.date, float], ...]
    """The two samples before it and the two after: date and count."""
    weighted: float | None = None
    """Its own record's weighted share count, where the record carries one."""
    spared: bool = False
    """Whether that weighted count confirms the sample, within the tolerance: not an error."""


def isolated_errors(
    columns: Sequence[Column],
    samples: Mapping[str, Sequence[tuple[dt.date, str, float]]],
    splits: Sequence[SplitCheck],
    tolerance: float = TOLERANCE,
    weighted: Mapping[tuple[str, dt.date], float] | None = None,
) -> list[Repair]:
    """Every sample the isolated-error rule finds, per column over its keys' samples in date
    order -- the stale ones, read past by the lag rule, are none of them. `weighted[(plan key,
    day)]` is a sample's own record's weighted count: where it confirms the sample within the
    tolerance, the sample is spared, two independent fields of one record agreeing -- the same
    day's record, so nothing looks ahead (midoffice09, ruling 1). The rest are removed.

    One pass over the samples as they are, so the rule reads only its own case: where two
    samples in a row are wrong, or the samples alternate between two counts, as CEG's do, the
    neighbours of each disagree and nothing is removed -- the rule cannot tell which count is
    the true one -- and the count-move report still flags them. A real change, a merger's
    issuance or a unification, is a step the later samples share, and is adopted."""
    out = []
    for column in columns:
        mine = [c for c in splits if c.key in column.keys and c.applies]
        stale = {(day, float(count)) for c in mine for day, count in c.stale}
        split_days = [c.split.execution_date for c in mine]
        sequence = sorted(
            (day, ticker, float(count), key) for key in column.keys
            for day, ticker, count in samples.get(key, ())
            if count and (day, float(count)) not in stale
        )
        for k in range(2, len(sequence) - 2):
            around = [sequence[k - 2], sequence[k - 1], sequence[k + 1], sequence[k + 2]]
            values = [count for _, _, count, _ in around]
            if max(values) / min(values) - 1.0 > tolerance:
                continue
            if any(around[0][0] < day <= around[-1][0] for day in split_days):
                continue
            day, ticker, count, key = sequence[k]
            if all(abs(count / value - 1.0) > tolerance for value in values):
                own = (weighted or {}).get((key, day))
                out.append(Repair(column.key, key, column.label, ticker, day, count,
                                  tuple((d, c) for d, _, c, _ in around), own,
                                  own is not None and abs(own / count - 1.0) <= tolerance))
    return out


@dataclass(frozen=True)
class Delisting:
    """A span the reference records place off an exchange (midoffice09, ruling 2): from its
    first sample whose record's `market` is not "stocks" -- OTC Link's, after a failure -- it is
    not an instrument the model is for. The span ends on its last sample the records place on an
    exchange, or goes whole if it has none; but it runs on through any day the index still held
    the security, so no member-day changes, and that day is reported."""

    span: Span
    off_exchange: dt.date
    """The span's first sample whose record places it off an exchange."""
    market: str
    exchange: str | None
    """The record's `market` and `primary_exchange` there."""
    last_listed: dt.date | None
    """The span's last sample before it on an exchange; None: none in the span."""
    kept_to: dt.date | None
    """The span's new last day; None: the span goes."""
    held: tuple[dt.date, ...] = ()
    """Member-days after `last_listed`, which the span keeps: the trading days of the index's
    runs of membership, where a calendar is given, else every day of them."""


def listed_spans(
    spans: Sequence[Span],
    market_of: Callable[[str, dt.date], tuple[str, str | None] | None],
    member_days: Mapping[str, Sequence[tuple[dt.date, dt.date]]],
    calendar: Sequence[dt.date] | None = None,
) -> tuple[list[Span], list[Delisting]]:
    """The spans with each one's off-exchange days removed (`Delisting`). `market_of(ticker,
    day)` is the cached record's `market` and `primary_exchange`, or None where there is no
    record; a span with no sample off an exchange is kept as it is. `member_days[key]` is each
    plan key's runs of membership, (first, last), and `calendar` the panel's trading days, which
    a run's member-days are. A leaver that stays listed keeps its span."""
    trading = None if calendar is None else frozenset(calendar)
    out: list[Span] = []
    ended: list[Delisting] = []
    for span in spans:
        markets = [(day, market_of(span.ticker, day)) for day in span.samples]
        off = next(((day, m) for day, m in markets if m is not None and m[0] != "stocks"), None)
        if off is None:
            out.append(span)
            continue
        listed = [day for day, m in markets if day < off[0] and m is not None
                  and m[0] == "stocks"]
        last_listed = max(listed) if listed else None
        held = sorted(
            day for first, last in member_days.get(span.key, ())
            for day in map(dt.date.fromordinal, range(first.toordinal(), last.toordinal() + 1))
            if span.first <= day <= span.last
            and (last_listed is None or day > last_listed)
            and (trading is None or day in trading)
        )
        ends = [day for day in (last_listed, held[-1] if held else None) if day is not None]
        kept_to = max(ends) if ends else None
        ended.append(Delisting(span, off[0], off[1][0], off[1][1], last_listed, kept_to,
                               tuple(held)))
        if kept_to is not None:
            out.append(Span(span.key, span.name, span.ticker, span.first, kept_to,
                            span.member_days, tuple(d for d in span.samples if d <= kept_to),
                            span.previous))
    return out, ended


@dataclass(frozen=True)
class Override:
    """A ruling on a security's share counts, where its records are wrong and no general rule
    can tell (`read_overrides`). Over `first` to `last`, "carry" sets the carried count to
    `count` on the range's first day, and "none" leaves the range with none. Either way the
    samples inside the range are not read, and a split the gate applies inside it still
    multiplies a carried count. A "none" range carries nothing out: the next sample sets the
    count."""

    key: str
    label: str
    first: dt.date
    last: dt.date
    action: str
    count: float | None
    evidence: str
    ruled: str


def read_overrides(path: Path | str) -> list[Override]:
    """The overrides file, `config/sp500_share_count_overrides.json`: rulings, not vendor data,
    so in the repo rather than in `data/`. Each entry a security's key, a date range, what the
    build does there, and the evidence (midoffice08, ruling 3). A new anomaly goes in the file
    with its evidence, not into a rule written for one case."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
    out = []
    for entry in entries:
        action, count = entry["action"], entry.get("count")
        if action not in ("carry", "none"):
            raise ValueError(f"{entry['key']}: an override carries a count or none, not "
                             f"{action!r}")
        if (action == "carry") != (count is not None):
            raise ValueError(f"{entry['key']}: a carry needs a count, and only a carry has one")
        first, last = dt.date.fromisoformat(entry["first"]), dt.date.fromisoformat(entry["last"])
        if last < first:
            raise ValueError(f"{entry['key']}: its range ends before it starts")
        if not entry.get("evidence"):
            raise ValueError(f"{entry['key']}: an override needs its evidence")
        out.append(Override(entry["key"], entry.get("label", entry["key"]), first, last, action,
                            None if count is None else float(count), entry["evidence"],
                            entry.get("ruled", "")))
    keys = [(o.key, o.first, o.last) for o in out]
    for o in out:
        if any(k == o.key and a <= o.last and o.first <= b and (a, b) != (o.first, o.last)
               for k, a, b in keys):
            raise ValueError(f"{o.key}: two overrides overlap")
    return out


@dataclass(frozen=True)
class SpinOff:
    """A ruling on a distribution of another company's shares that the cache records wrongly or
    not at all (`read_spin_offs`; midoffice14). On `ex_date` the parent's total return carries
    `ratio` of the child's shares a parent share, valued at the child's raw close that day -- or
    its first after, where it had none -- with any cash distribution the same day:

        (w (C_t + D_t) + k V) / C_s - 1,

    C the parent's raw closes on the ex-date and its previous priced day s, D its cash that day,
    V the child's close and w the ratios of the splits the build applies between s and t. The
    stitched prices -- Polygon's adjusted closes -- are left as they are: the ruling is the total
    return's. `split`, where the parent split with the distribution and the cache records no
    event for it, is applied to its carried count on the ex-date as a split the gate classified
    (`spin_off_splits`), and so enters w. `child_count` "parent": where the child's records carry
    no count, the parent's carried count on the record date times the ratio, from the child's
    first day on or after the ex-date until a sample of its own carries one. `supersedes`: count
    overrides this ruling replaces, each (key, first, last).

    `retained`, for a seeded count: the share of the child the parent kept, so the child's count
    is the shares distributed over 1 - retained, to the nearest share -- every count in the panel
    is the total outstanding (FedEx's 19.9% of FedEx Freight, midoffice15's ruling 3). `held`: a
    ruling held out until it is understood, with why -- read and listed, never applied (S&P
    Global's Mobility, whose ruled ex-date the data show no gap on, midoffice15's ruling 1)."""

    parent: str
    label: str
    name: str
    child: str
    child_key: str | None
    """The child's composite FIGI, where Polygon's record has one: its column's key, or a key it
    holds, for `child_count`; checked against the child's record where the cache has one."""
    ratio: float
    record: dt.date
    distributed: dt.date
    ex_date: dt.date
    split: float = 1.0
    child_count: str | None = None
    supersedes: tuple[tuple[str, dt.date, dt.date], ...] = ()
    evidence: str = ""
    ruled: str = ""
    retained: float = 0.0
    held: str = ""


def read_spin_offs(path: Path | str) -> list[SpinOff]:
    """The spin-off overrides file, `config/sp500_spin_off_overrides.json`: rulings with their
    evidence, as the count rulings are carried (midoffice14's brief). Each entry the parent's key,
    label and name, the child's ticker and key, the ratio, the record, distribution and ex-dates,
    any split the parent made with it, whether the child's count is seeded from the parent's and
    the share of the child the parent kept, the count overrides it supersedes, the evidence, and
    whether it is held (midoffice15). A held ruling is read with the rest: the build lists it and
    applies only the others."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
    out = []
    for entry in entries:
        where = f"{entry.get('label', entry['parent'])} {entry.get('ex_date')}"
        ratio, split = float(entry["ratio"]), float(entry.get("split", 1.0))
        if not ratio > 0.0 or not split > 0.0:
            raise ValueError(f"{where}: a ratio and a split are positive")
        child_count = entry.get("child_count")
        if child_count not in (None, "parent"):
            raise ValueError(f"{where}: a child's count is seeded from the parent's or not, not "
                             f"{child_count!r}")
        if child_count and not entry.get("child_key"):
            raise ValueError(f"{where}: a seeded count needs the child's key")
        retained = float(entry.get("retained", 0.0))
        if not 0.0 <= retained < 1.0:
            raise ValueError(f"{where}: the share of the child the parent kept is at least 0 and "
                             f"under 1, not {retained!r}")
        if retained and not child_count:
            raise ValueError(f"{where}: a retained share grosses up a seeded count, and this "
                             f"ruling seeds none")
        held = entry.get("held", "")
        if not isinstance(held, str):
            raise ValueError(f"{where}: a held ruling says why it is held, in words")
        record, distributed, ex = (dt.date.fromisoformat(entry[k])
                                   for k in ("record", "distributed", "ex_date"))
        if distributed < record or ex < record:
            raise ValueError(f"{where}: its dates run backwards")
        if not entry.get("evidence"):
            raise ValueError(f"{where}: a ruling needs its evidence")
        supersedes = tuple((s["key"], dt.date.fromisoformat(s["first"]),
                            dt.date.fromisoformat(s["last"]))
                           for s in entry.get("supersedes", ()))
        out.append(SpinOff(entry["parent"], entry.get("label", entry["parent"]),
                           entry.get("name", ""), entry["child"], entry.get("child_key"), ratio,
                           record, distributed, ex, split, child_count, supersedes,
                           entry["evidence"], entry.get("ruled", ""), retained=retained,
                           held=held))
    seen = set()
    for s in out:
        if (s.parent, s.ex_date) in seen:
            raise ValueError(f"{s.label}: two rulings on {s.ex_date}")
        seen.add((s.parent, s.ex_date))
    return out


def _refuse_held(spin_offs: Sequence[SpinOff]) -> None:
    """A held ruling is the build's to list and leave out, never to apply."""
    held = [f"{s.label} {s.ex_date}" for s in spin_offs if s.held]
    if held:
        raise ValueError(f"a held spin-off ruling is not applied: {', '.join(held)}")


def spin_off_splits(
    spin_offs: Sequence[SpinOff],
    samples: Mapping[str, Sequence[tuple[dt.date, str, float]]],
    tolerance: float = TOLERANCE,
    largest_term: int = LARGEST_TERM,
) -> list[SplitCheck]:
    """Each split a spin-off ruling says the parent made with its distribution, and the cache
    does not record, as a split event the gate would classify: tested against the parent's own
    share samples (`rebuild.classify`), which must follow it, and not against the raw price,
    which the distribution in the same minute offsets -- Honeywell's 1-for-2 reverse split of
    2026-06-29, one minute after its Aerospace distribution. Raises if the counts do not follow
    the ratio, and on a held ruling, which is not applied."""
    _refuse_held(spin_offs)
    out = []
    for s in spin_offs:
        if s.split == 1.0:
            continue
        fraction = Fraction(s.split).limit_denominator(1000)
        split = Split(s.ex_date, float(fraction.denominator), float(fraction.numerator), s.label)
        counts = sorted((day, float(count)) for day, _, count in samples.get(s.parent, ())
                        if count)
        before = next((c for c in reversed(counts) if c[0] < s.ex_date), None)
        later = [c for c in counts if c[0] >= s.ex_date]
        found = classify(split, before, later, None, tolerance=tolerance,
                         largest_term=largest_term)
        if found["kind"] != "split" or found["counts"] != "follow":
            raise ValueError(f"{s.label} {s.ex_date}: the ruling's split of {s.split:g} is not "
                             f"one its counts follow: {found['detail']}")
        detail = (f"{found['detail']}; a spin-off ruling's split the cache does not record, its "
                  f"price not tested -- the distribution the same day offsets the gap")
        out.append(SplitCheck(s.parent, s.name, split, before, found["after"], found["whole"],
                              found["counts"], found["kind"], False, detail, found["stale"],
                              found["followed"], None, None))
    return out


@dataclass(frozen=True)
class SpinOffUse:
    """A spin-off ruling as the build applied it: the parent's ex-date total return, from the
    adjusted closes and from the ruling."""

    ruling: SpinOff
    key: str
    previous: dt.date
    day: dt.date
    raw_before: float
    raw_after: float
    child_close: float
    child_day: dt.date
    """The child's close the value is taken at: the ex-date's, or its first after."""
    cash: float
    splits: float
    """w: the ratios of the splits the build applies between the two days."""
    adjusted: float
    """The day's total-return step from the adjusted closes, as built without the ruling."""
    step: float
    """The ruling's: (w (C_t + D_t) + k V) / C_s."""


@dataclass(frozen=True)
class Seeded:
    """A child's count seeded from its parent's: the parent's carried count on the record date
    times the ratio -- over 1 - retained, to the nearest share, where the parent kept part of
    the child -- from `first` until its own first count, `until` (None: to the window's end)."""

    ruling: SpinOff
    key: str
    first: dt.date
    count: float
    parent_count: float
    until: dt.date | None
    days: int
    member_days: int


# ----------------------------------------------------------------------------- the build


@dataclass(frozen=True)
class Bridge:
    """A change of ticker inside a column: the return across it, from the raw closes."""

    key: str
    before: str
    after: str
    last_before: dt.date
    first_after: dt.date
    raw_before: float
    raw_after: float
    ratio: float
    """The product of the ratios of the security's split events between the two days, after
    the first and up to the second: 1 where there are none."""
    scale: float
    """The earlier run's scale on its ticker's adjusted closes, so its prices continue the
    later run's."""

    @property
    def relative(self) -> float:
        return self.raw_after * self.ratio / self.raw_before


@dataclass(frozen=True)
class Carry:
    """A count carried into a span from before it -- across a reorganisation, or from the
    security's own earlier span -- until a sample with a count resets it."""

    key: str
    ticker: str
    first: dt.date
    count: float
    until: dt.date | None
    """The day a sample of its own resets it; None: to the span's end."""
    reorganisation: bool


@dataclass
class Panel:
    """The panel and what building it found. Arrays are dates by columns."""

    dates: tuple[dt.date, ...]
    columns: tuple[Column, ...]
    prices: np.ndarray
    """The stitched adjusted closes: price returns."""
    total_prices: np.ndarray
    """A total-return index: the stitched closes with each distribution reinvested."""
    raw_prices: np.ndarray
    volumes: np.ndarray
    """Traded: the raw bars' share volume."""
    adjusted_volumes: np.ndarray
    """The adjusted bars' volume on the stitched prices' basis: the old panel's kind."""
    shares: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    tickers: np.ndarray
    """The ticker each cell's price is under; "" outside the column's spans."""
    membership: np.ndarray
    market_returns: np.ndarray
    market_total_returns: np.ndarray
    bridges: list[Bridge] = field(default_factory=list)
    carries: list[Carry] = field(default_factory=list)
    applied: list[tuple[str, dt.date, float]] = field(default_factory=list)
    """The splits applied to the carried counts: column key, execution date, ratio."""
    stale_read_past: list[tuple[str, dt.date, float]] = field(default_factory=list)
    duplicates: list[Dividend] = field(default_factory=list)
    """Distributions dropped as exact duplicates of one kept."""
    distributions: int = 0
    """Distributions reinvested in the columns' total-return indices."""
    unpriced_ex_dates: list[tuple[str, Dividend, dt.date]] = field(default_factory=list)
    """Distributions whose ex-date has no price in their column: put on the basis of, and
    reinvested at, the first priced day after it."""
    before_first_price: list[tuple[str, Dividend]] = field(default_factory=list)
    """Distributions on or before a column's first price, which no return can carry."""
    market_distributions: int = 0
    unapplied: list[tuple[str, dt.date, str]] = field(default_factory=list)
    """Splits the gate classified as splits whose execution date found no count to apply to:
    column key, date, why."""
    overridden: list[tuple[Override, int, int, int]] = field(default_factory=list)
    """Each override used: it, the days of its range inside the column's spans, the member-days
    among them, and the samples inside it not read."""
    spin_offs: list[SpinOffUse] = field(default_factory=list)
    """Each spin-off ruling applied to its parent's total return."""
    seeded: list[Seeded] = field(default_factory=list)
    """Each child's count seeded from its parent's."""
    opens: np.ndarray | None = None
    """The raw opens, under each cell's ticker: for the first-day check, not written."""


def _by_date(bars: Sequence[Bar]) -> dict[dt.date, Bar]:
    return {bar.date: bar for bar in bars}


def _exact_duplicates(events: Sequence[Dividend]) -> tuple[list[Dividend], list[Dividend]]:
    """The distributions with each exact duplicate kept once -- equal in every field, Polygon's
    id aside -- and the ones dropped."""
    kept, dropped, seen = [], [], set()
    for event in events:
        if event in seen:
            dropped.append(event)
            continue
        seen.add(event)
        kept.append(event)
    return kept, dropped


def build_panel(
    dates: Sequence[dt.date],
    columns: Sequence[Column],
    bars: Bars,
    splits: Sequence[SplitCheck],
    samples: Mapping[str, Sequence[tuple[dt.date, str, float]]],
    dividends: Mapping[str, Sequence[Dividend]],
    member_days: Mapping[str, Sequence[tuple[dt.date, dt.date]]],
    market: tuple[Sequence[Bar], Sequence[Bar], Sequence[Dividend]],
    overrides: Sequence[Override] = (),
    spin_offs: Sequence[SpinOff] = (),
) -> Panel:
    """Assemble the panel. `bars(ticker, "adj" | "raw")` is a ticker's daily bars;
    `splits` the gate's classification of every split event inside the spans, with any split a
    spin-off ruling adds (`spin_off_splits`); `samples[key]` each plan key's share counts, (date,
    ticker, count), the searched first counts among them; `dividends[ticker]` each ticker's
    distributions; `member_days[key]` each plan key's runs of membership, (first, last), read
    through the renames; `market` the market series' adjusted and raw bars and its
    distributions; `overrides` the rulings on counts (`Override`); `spin_offs` the rulings on
    distributions of shares (`SpinOff`), none of them held. `dates` are the trading calendar, the
    market's."""
    _refuse_held(spin_offs)
    dates = tuple(dates)
    n, m = len(dates), len(columns)
    at = {day: i for i, day in enumerate(dates)}
    shape = (n, m)

    def blank() -> np.ndarray:
        return np.full(shape, np.nan)

    prices, total, raw_prices = blank(), blank(), blank()
    volumes, adjusted_volumes, shares, highs, lows = blank(), blank(), blank(), blank(), blank()
    opens = blank()
    tickers = np.full(shape, "", dtype=object)
    membership = np.zeros(shape, dtype=bool)
    out = Panel(dates, tuple(columns), prices, total, raw_prices, volumes, adjusted_volumes,
                shares, highs, lows, tickers, membership, np.full(n, np.nan),
                np.full(n, np.nan), opens=opens)
    for ruling in spin_offs:
        if ruling.ex_date not in at or ruling.record not in at:
            raise ValueError(f"{ruling.label}: the spin-off ruling's ex-date {ruling.ex_date} "
                             f"and record date {ruling.record} must be trading days")
    by_key: dict[str, list[SplitCheck]] = {}
    for check in splits:
        by_key.setdefault(check.key, []).append(check)

    for j, column in enumerate(columns):
        # Each day's ticker, from the column's spans; they must not overlap.
        span_of = np.full(n, -1, dtype=int)
        for k, span in enumerate(column.spans):
            for i in range(at[span.first], at[span.last] + 1):
                if span_of[i] >= 0:
                    raise ValueError(f"{column.label} ({column.key}): its spans overlap on "
                                     f"{dates[i]}")
                span_of[i] = k
        adj = {s.ticker: _by_date(bars(s.ticker, "adj")) for s in column.spans}
        raw = {s.ticker: _by_date(bars(s.ticker, "raw")) for s in column.spans}
        # Runs of one ticker's priced days: (ticker, [day indices]).
        runs: list[tuple[str, list[int]]] = []
        for i in np.flatnonzero(span_of >= 0):
            ticker = column.spans[span_of[i]].ticker
            tickers[i, j] = ticker
            bar = adj[ticker].get(dates[i])
            if bar is None:
                continue
            if raw[ticker].get(dates[i]) is None:
                raise ValueError(f"{column.label}: an adjusted bar and no raw one under "
                                 f"{ticker} on {dates[i]}")
            if not runs or runs[-1][0] != ticker:
                runs.append((ticker, []))
            runs[-1][1].append(int(i))
        events = sorted((c for key in column.keys for c in by_key.get(key, ())),
                        key=lambda c: c.split.execution_date)
        # Each run's scale, from the last back: the bridge from raw closes.
        scales = [1.0] * len(runs)
        for k in range(len(runs) - 2, -1, -1):
            (before, earlier), (after, later) = runs[k], runs[k + 1]
            s, t = earlier[-1], later[0]
            ratio = 1.0
            for check in events:
                if dates[s] < check.split.execution_date <= dates[t]:
                    ratio *= check.split.ratio
            raw_s, raw_t = raw[before][dates[s]].close, raw[after][dates[t]].close
            if raw_s <= 0.0 or raw_t <= 0.0:
                raise ValueError(f"{column.label}: no positive raw close to bridge {before} on "
                                 f"{dates[s]} to {after} on {dates[t]}")
            relative = raw_t * ratio / raw_s
            scales[k] = (adj[after][dates[t]].close * scales[k + 1]
                         / (adj[before][dates[s]].close * relative))
            out.bridges.append(Bridge(column.key, before, after, dates[s], dates[t], raw_s,
                                      raw_t, ratio, scales[k]))
        for (ticker, days), scale in zip(runs, scales):
            for i in days:
                a, r = adj[ticker][dates[i]], raw[ticker][dates[i]]
                prices[i, j] = a.close * scale
                adjusted_volumes[i, j] = a.volume / scale
                raw_prices[i, j] = r.close
                opens[i, j] = r.open
                volumes[i, j] = r.volume
                highs[i, j] = r.high
                lows[i, j] = r.low

        # Total return: each distribution inside the column's spans, on its ex-date.
        paid = [d for span in column.spans for d in dividends.get(span.ticker, ())
                if span.first <= d.ex_date <= span.last]
        paid, dropped = _exact_duplicates(sorted(paid, key=lambda d: (d.ex_date, d.ticker)))
        out.duplicates.extend(dropped)
        priced = [i for _, days in runs for i in days]
        due: dict[int, list[Dividend]] = {}
        for event in paid:
            later = [i for i in priced if dates[i] >= event.ex_date]
            if not priced or event.ex_date <= dates[priced[0]] or not later:
                out.before_first_price.append((column.key, event))
                continue
            due.setdefault(later[0], []).append(event)
            if dates[later[0]] != event.ex_date:
                out.unpriced_ex_dates.append((column.key, event, dates[later[0]]))
        # A spin-off ruling's ex-date carries the child's shares in the total return, from the
        # raw closes (`SpinOff`); the stitched prices keep the adjusted closes' own step.
        spun = {at[r.ex_date]: r for r in spin_offs if r.parent in column.keys}
        if priced:
            total[priced[0], j] = prices[priced[0], j]
            for s, t in zip(priced, priced[1:]):
                cash = sum(d.cash_amount for d in due.get(t, ()))
                reinvested = cash * prices[t, j] / raw_prices[t, j]
                ruling = spun.pop(t, None)
                if ruling is None:
                    total[t, j] = total[s, j] * (prices[t, j] + reinvested) / prices[s, j]
                else:
                    step = (prices[t, j] + reinvested) / prices[s, j]
                    w = 1.0
                    for check in events:
                        if check.applies and dates[s] < check.split.execution_date <= dates[t]:
                            w *= check.split.ratio
                    child = [b for b in bars(ruling.child, "raw") if b.date >= ruling.ex_date]
                    if not child or child[0].close <= 0.0:
                        raise ValueError(f"{column.label}: no close of {ruling.child} on or "
                                         f"after its ex-date {ruling.ex_date}")
                    ruled = ((w * (raw_prices[t, j] + cash) + ruling.ratio * child[0].close)
                             / raw_prices[s, j])
                    out.spin_offs.append(SpinOffUse(
                        ruling, column.key, dates[s], dates[t], float(raw_prices[s, j]),
                        float(raw_prices[t, j]), float(child[0].close), child[0].date,
                        float(cash), w, float(step), float(ruled)))
                    total[t, j] = total[s, j] * ruled
                out.distributions += len(due.get(t, ()))
        if spun:
            raise ValueError(f"{column.label}: a spin-off ruling's ex-date "
                             f"{', '.join(str(dates[t]) for t in spun)} is not a priced day "
                             f"after the column's first")

        # Share counts: carried forward from each sample, splits applied on their dates.
        applied = [c for c in events if c.applies]
        stale = {(day, count) for c in applied for day, count in c.stale}
        by_day: dict[dt.date, list[tuple[str, float]]] = {}
        for key in column.keys:
            for day, ticker, count in samples.get(key, ()):
                if count:
                    by_day.setdefault(day, []).append((ticker, float(count)))
        splits_on: dict[dt.date, list[SplitCheck]] = {}
        for check in applied:
            splits_on.setdefault(check.split.execution_date, []).append(check)
        assigned = np.flatnonzero(span_of >= 0)
        if assigned.size == 0:
            raise ValueError(f"{column.label} ({column.key}): no day of the window in its spans")
        carried = np.nan
        # A count carried into the span being walked, not yet reset by a sample of its own.
        pending: Carry | None = None
        mine = [o for o in overrides if o.key in column.keys]
        unread = {o: 0 for o in mine}
        for i in range(int(assigned[0]), int(assigned[-1]) + 1):
            day = dates[i]
            for check in splits_on.get(day, ()):
                if np.isfinite(carried):
                    carried *= check.split.ratio
                    out.applied.append((column.key, day, check.split.ratio))
                else:
                    out.unapplied.append((column.key, day, "no count carried to it"))
            active = next((o for o in mine if o.first <= day <= o.last), None)
            if active is not None and (i == assigned[0]
                                       or not active.first <= dates[i - 1] <= active.last):
                # The ruling governs from the range's first day: a carry into it ends there.
                carried = active.count if active.action == "carry" else np.nan
                if pending is not None and pending.first != day:
                    out.carries.append(Carry(pending.key, pending.ticker, pending.first,
                                             pending.count, day, pending.reorganisation))
                pending = None
            starts = span_of[i] >= 0 and (i == assigned[0] or span_of[i - 1] != span_of[i])
            if starts and np.isfinite(carried) and active is None:
                span = column.spans[span_of[i]]
                before = column.spans[span_of[i] - 1] if span_of[i] > 0 else None
                pending = Carry(column.key, span.ticker, day, float(carried), None,
                                before is not None and before.key != span.key)
            for _ticker, count in by_day.get(day, ()):
                if active is not None:
                    unread[active] += 1
                    continue
                if (day, count) in stale:
                    out.stale_read_past.append((column.key, day, count))
                    continue
                carried = count
                if pending is not None and pending.first != day:
                    out.carries.append(Carry(pending.key, pending.ticker, pending.first,
                                             pending.count, day, pending.reorganisation))
                pending = None
            if span_of[i] >= 0:
                shares[i, j] = carried
            ends = span_of[i] >= 0 and (i + 1 >= n or span_of[i + 1] != span_of[i])
            if ends and pending is not None:
                out.carries.append(pending)  # to the span's end
                pending = None

        # Membership: the column's keys' member-days.
        for key in column.keys:
            for first, last in member_days.get(key, ()):
                for i in range(at[first], at[last] + 1):
                    if membership[i, j]:
                        raise ValueError(f"{column.label}: a member twice on {dates[i]}")
                    membership[i, j] = True
        for o in mine:
            inside = [i for i in assigned if o.first <= dates[i] <= o.last]
            out.overridden.append((o, len(inside), int(membership[inside, j].sum()), unread[o]))

    # A child whose records carry no count: its parent's on the record date at the ratio, from
    # its first day on or after the ex-date until a sample of its own carries one, never back --
    # over 1 - retained where the parent kept part of the child, to the nearest share: the
    # child's total outstanding, as every other count is (midoffice15's ruling 3).
    index = {key: j for j, c in enumerate(columns) for key in c.keys}
    for ruling in spin_offs:
        if ruling.child_count != "parent":
            continue
        if ruling.child_key not in index or ruling.parent not in index:
            raise ValueError(f"{ruling.label}: a seeded count needs the parent and the child, "
                             f"{ruling.child} ({ruling.child_key}), as columns")
        jp, jc = index[ruling.parent], index[ruling.child_key]
        base = shares[at[ruling.record], jp]
        if not np.isfinite(base):
            raise ValueError(f"{ruling.label}: no count carried on the record date "
                             f"{ruling.record} to seed {ruling.child}'s")
        inside = tickers[:, jc] != ""
        days = [i for i in range(at[ruling.ex_date], n) if inside[i]]
        if not days:
            raise ValueError(f"{ruling.child}: no day in its spans on or after {ruling.ex_date}")
        count, filled, i = float(base) * ruling.ratio, [], days[0]
        if ruling.retained:
            count = float(round(count / (1.0 - ruling.retained)))
        while i < n and not np.isfinite(shares[i, jc]):
            if inside[i]:
                shares[i, jc] = count
                filled.append(i)
            i += 1
        out.seeded.append(Seeded(ruling, columns[jc].key, dates[days[0]], count, float(base),
                                 dates[i] if i < n else None, len(filled),
                                 int(membership[filled, jc].sum())))

    # The market series, both returns; a distribution on its ex-date, or the first trading day
    # after it.
    adjusted, unadjusted, paid = market
    adj, raw = _by_date(adjusted), _by_date(unadjusted)
    market_due: dict[int, list[Dividend]] = {}
    for event in _exact_duplicates(sorted(paid, key=lambda d: d.ex_date))[0]:
        later = [i for i in range(n) if dates[i] >= event.ex_date]
        if later and event.ex_date > dates[0]:
            market_due.setdefault(later[0], []).append(event)
    for i in range(1, n):
        a, b = adj.get(dates[i - 1]), adj.get(dates[i])
        if a is None or b is None or a.close <= 0.0:
            continue
        out.market_returns[i] = b.close / a.close - 1.0
        cash = sum(d.cash_amount for d in market_due.get(i, ()))
        if cash and dates[i] not in raw:
            raise ValueError(f"the market series pays on {dates[i]} and has no raw bar there")
        basis = b.close / raw[dates[i]].close if cash else 1.0
        out.market_distributions += len(market_due.get(i, ()))
        out.market_total_returns[i] = (b.close + cash * basis) / a.close - 1.0
    return out


# ----------------------------------------------------------------------------- guards


@dataclass(frozen=True)
class Guard:
    name: str
    checked: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def identity_guards(
    panel: Panel, bars: Bars, outside: set[tuple[str, dt.date]] = frozenset()
) -> list[Guard]:
    """The identity table's guards, on the built panel: each (ticker, date) price feeds one
    column, and no column takes two tickers on one date -- the cells say whose ticker each is,
    one per cell; every member-day is a column's on a day inside its spans, but those the table
    maps outside them, `outside`, (plan key, date) -- a member the index carried a day or two
    past its last trade; every bar under a span's ticker inside the span is in its column, and
    nothing else is."""
    n, m = panel.tickers.shape
    owner: dict[tuple[str, int], int] = {}
    clashes = []
    for j in range(m):
        for i in np.flatnonzero(panel.tickers[:, j] != ""):
            cell = (panel.tickers[i, j], int(i))
            if cell in owner and owner[cell] != j:
                clashes.append(f"{cell[0]} on {panel.dates[i]}: {panel.columns[owner[cell]].key}"
                               f" and {panel.columns[j].key}")
            owner[cell] = j
    unspanned = [(int(i), int(j)) for i, j in np.argwhere(panel.membership & (panel.tickers == ""))
                 if not any((key, panel.dates[i]) in outside for key in panel.columns[j].keys)]
    at = {day: i for i, day in enumerate(panel.dates)}
    missing, extra, checked = [], [], 0
    for j, column in enumerate(panel.columns):
        expected = set()
        for span in column.spans:
            for bar in bars(span.ticker, "adj"):
                if span.first <= bar.date <= span.last and bar.date in at:
                    expected.add(at[bar.date])
        priced = set(int(i) for i in np.flatnonzero(np.isfinite(panel.prices[:, j])))
        checked += len(expected)
        missing += [f"{column.label} {panel.tickers[i, j]} {panel.dates[i]}: a bar, no price"
                    for i in sorted(expected - priced)]
        extra += [f"{column.label} {panel.tickers[i, j]} {panel.dates[i]}: a price, no bar"
                  for i in sorted(priced - expected)]
    overlaps = []
    for column in panel.columns:
        ordered = sorted(column.spans, key=lambda s: s.first)
        overlaps += [f"{column.label}: {a.ticker} to {a.last}, {b.ticker} from {b.first}"
                     for a, b in zip(ordered, ordered[1:]) if b.first <= a.last]
    return [
        Guard("each (ticker, date) price feeds one column", len(owner), tuple(clashes)),
        Guard("no column takes two tickers on one date: its spans do not overlap",
              sum(len(c.spans) for c in panel.columns), tuple(overlaps)),
        Guard("every member-day is a column's, on a day inside its spans or one the table maps "
              "outside them", int(panel.membership.sum()),
              tuple(f"{panel.columns[j].label} {panel.dates[i]}" for i, j in unspanned)),
        Guard("every bar under a span's ticker inside the span is its column's, and only those",
              checked, tuple(missing + extra)),
    ]


def split_guard(panel: Panel, splits: Sequence[SplitCheck],
                samples: Mapping[str, Sequence[tuple[dt.date, str, float]]]) -> Guard:
    """Exact: for every split the gate classified as one, no day of its column on or after the
    execution date -- to the column's next split -- carries a pre-split count: a value the
    column carried on a day before it, since its last split, or a stale sample's. And the
    execution date's count is the count before it times the ratio, exactly, unless a sample
    there that follows the ratio resets it. Read from the built counts, not from the code that
    built them."""
    index = {key: j for j, c in enumerate(panel.columns) for key in c.keys}
    at = {day: i for i, day in enumerate(panel.dates)}
    by_column: dict[int, list[SplitCheck]] = {}
    for check in splits:
        if check.applies and check.key in index:
            by_column.setdefault(index[check.key], []).append(check)
    failures, checked, members = [], 0, 0
    for j, checks in by_column.items():
        column = panel.columns[j]
        counts = panel.shares[:, j]
        checks.sort(key=lambda c: c.split.execution_date)
        start = 0
        for k, check in enumerate(checks):
            e = at.get(check.split.execution_date)
            if e is None:
                continue
            stop = (at.get(checks[k + 1].split.execution_date, len(panel.dates))
                    if k + 1 < len(checks) else len(panel.dates))
            before = counts[start:e]
            pre = {float(v) for v in before[np.isfinite(before)]}
            pre |= {float(c) for _, c in check.stale}
            for i in range(e, stop):
                if np.isfinite(counts[i]) and panel.tickers[i, j] != "":
                    checked += 1
                    members += bool(panel.membership[i, j])
                    if float(counts[i]) in pre:
                        failures.append(f"{column.label} on {panel.dates[i]} carries "
                                        f"{counts[i]:,.0f}, a count from before its split of "
                                        f"{check.split.execution_date}")
            prior = [v for v in before if np.isfinite(v)]
            if prior and np.isfinite(counts[e]):
                expected = prior[-1] * check.split.ratio
                fresh = {float(c) for key in column.keys for day, _, c in samples.get(key, ())
                         if day == check.split.execution_date and c
                         and (day, c) not in set(check.stale)}
                if float(counts[e]) != expected and float(counts[e]) not in fresh:
                    failures.append(f"{column.label} on {check.split.execution_date}: "
                                    f"{counts[e]:,.0f}, not {prior[-1]:,.0f} times "
                                    f"{check.split.ratio:g}")
            start = e
    return Guard(f"no day on or after a split carries a pre-split count, and the split's day "
                 f"carries the count before it times the ratio ({members:,} of the days "
                 f"checked member-days)", checked, tuple(failures))


@dataclass(frozen=True)
class Move:
    """A day a column's carried count moves beyond the tolerance with no split to explain it."""

    key: str
    label: str
    day: dt.date
    before: float
    after: float

    @property
    def ratio(self) -> float:
        return self.after / self.before


def count_moves(panel: Panel, splits: Sequence[SplitCheck],
                tolerance: float = TOLERANCE) -> list[Move]:
    """A standing check on the built counts: every day a column's count moves from the one
    before it by more than `tolerance`, other than by a split the gate applies that day --
    the gate tests the counts across each split event, and nothing else tests them between.
    Such a move is issuance or a buyback past a tenth of the shares, a merger, a split no event
    records, or a record carrying another figure than the shares in existence."""
    index = {key: j for j, c in enumerate(panel.columns) for key in c.keys}
    split_days = {(index[c.key], c.split.execution_date) for c in splits
                  if c.applies and c.key in index}
    out = []
    for j, column in enumerate(panel.columns):
        counts = panel.shares[:, j]
        days = np.flatnonzero(np.isfinite(counts))
        for s, t in zip(days, days[1:]):
            if counts[t] == counts[s] or (j, panel.dates[t]) in split_days:
                continue
            if abs(counts[t] / counts[s] - 1.0) > tolerance:
                out.append(Move(column.key, column.label, panel.dates[t], float(counts[s]),
                                float(counts[t])))
    return out


@dataclass(frozen=True)
class Weighted:
    """One record's weighted share count against the count the panel carries that day."""

    key: str
    label: str
    ticker: str
    day: dt.date
    weighted: float
    carried: float
    own: bool
    """Whether the day's count is the record's own share-class count; else carried from an
    earlier one -- a reorganisation's, a split's, past a stale sample, or a record without
    one."""

    @property
    def ratio(self) -> float:
        return self.weighted / self.carried

    def flagged(self, tolerance: float = TOLERANCE) -> bool:
        return abs(self.ratio - 1.0) > tolerance


def weighted_check(
    panel: Panel,
    weighted: Mapping[str, Sequence[tuple[dt.date, str, float, float | None]]],
) -> list[Weighted]:
    """Wherever a record carries Polygon's weighted share count and the panel carries a count
    that day: the two, and whether the day's count is that record's own share-class count.
    `weighted[key]` is each plan key's records, (date, ticker, weighted count, share-class
    count). A standing check, never a source: the weighted count is the earnings-per-share
    denominator, an average over a period, not the shares in existence on the day."""
    index = {key: j for j, c in enumerate(panel.columns) for key in c.keys}
    at = {day: i for i, day in enumerate(panel.dates)}
    out = []
    for key, records in weighted.items():
        j = index.get(key)
        if j is None:
            continue
        for day, ticker, value, own in records:
            i = at.get(day)
            if i is None or not value or not np.isfinite(panel.shares[i, j]):
                continue
            carried = float(panel.shares[i, j])
            out.append(Weighted(panel.columns[j].key, panel.columns[j].label, ticker, day,
                                float(value), carried, bool(own) and float(own) == carried))
    return sorted(out, key=lambda w: (w.label, w.day))


@dataclass(frozen=True)
class FirstDay:
    """A security's first priced day in the panel, and the members whose raw price opened
    furthest below the market's that day: where a child's first day is its parent's ex-date,
    the parent's gap is the distribution's value, and a gap no record or ruling explains is a
    spin-off the total return may not carry (midoffice14)."""

    key: str
    label: str
    day: dt.date
    gaps: tuple[tuple[str, str, float, str], ...]
    """Each member's key, label, opening gap less the market's, and what explains it: "ruling"
    (a spin-off ruling's ex-date), "split" (a split event the build read), "distribution" (a
    cash one), or "" for nothing on record."""


def first_days(
    panel: Panel,
    market_raw: Sequence[Bar],
    splits: Sequence[SplitCheck],
    dividends: Mapping[str, Sequence[Dividend]],
    spin_offs: Sequence[SpinOff] = (),
    threshold: float = 0.015,
    most: int = 3,
) -> list[FirstDay]:
    """A standing check: on each column's first priced day after the window's first, the
    members other than it whose raw opening gap -- the open over the previous raw close, less
    the market's -- is below `-threshold`, the `most` lowest, each with what explains it. The
    previous close across a change of ticker is the old ticker's, as the stitched prices bridge
    it."""
    if panel.opens is None:
        raise ValueError("the first-day check needs the panel's raw opens")
    market = _by_date(market_raw)
    split_days = {(c.key, c.split.execution_date) for c in splits}
    ruled = {(r.parent, r.ex_date) for r in spin_offs}
    out = []
    for j, column in enumerate(panel.columns):
        priced = np.flatnonzero(np.isfinite(panel.raw_prices[:, j]))
        if not priced.size or priced[0] == 0:
            continue
        i = int(priced[0])
        day, before = panel.dates[i], panel.dates[i - 1]
        if day not in market or before not in market or market[before].close <= 0.0:
            continue
        reference = market[day].open / market[before].close - 1.0
        rows = []
        for k in np.flatnonzero(panel.membership[i]):
            if k == j or not (np.isfinite(panel.opens[i, k])
                              and np.isfinite(panel.raw_prices[i - 1, k])):
                continue
            gap = float(panel.opens[i, k] / panel.raw_prices[i - 1, k] - 1.0 - reference)
            if gap >= -threshold:
                continue
            keys = panel.columns[k].keys
            ticker = str(panel.tickers[i, k])
            why = ("ruling" if any((key, day) in ruled for key in keys)
                   else "split" if any((key, day) in split_days for key in keys)
                   else "distribution" if any(d.ex_date == day
                                              for d in dividends.get(ticker, ()))
                   else "")
            rows.append((panel.columns[k].key, panel.columns[k].label, gap, why))
        rows.sort(key=lambda row: row[2])
        out.append(FirstDay(column.key, column.label, day, tuple(rows[:most])))
    return sorted(out, key=lambda f: (f.day, f.label))
