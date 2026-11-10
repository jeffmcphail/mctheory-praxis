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
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .identity import Record
from .polygon import Bar, Dividend
from .rebuild import TOLERANCE, Span, SplitCheck

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
    """The search of a span that starts without a share count for its first record with one."""

    span: Span
    bracket: tuple[dt.date, dt.date] | None
    """The span's first sample, without a count, and its first sample with one; None where no
    sample in the span carries one -- nothing to search between."""
    found: tuple[dt.date, float] | None
    """The first record with a count: its date, from which the count carries, and the count."""
    probes: tuple[tuple[dt.date, float | None, str | None], ...] = ()
    """Each record asked for, in order: its date, its count, the security it names."""
    failure: str | None = None
    """A probe naming another security than the span's: the search stops on it."""

    @property
    def detail(self) -> str:
        if self.failure:
            return self.failure
        if self.bracket is None:
            return (f"{self.span.ticker}: no sample in {self.span.first}..{self.span.last} "
                    f"carries a share-class count, so there is nothing to search between")
        a, b = self.bracket
        assert self.found is not None
        return (f"{self.span.ticker}: between {a}, without a count, and {b}, with one, the "
                f"first record with a count is {self.found[0]}'s, {self.found[1]:,.0f}, in "
                f"{len(self.probes)} lookup(s)")


def search_first_count(
    span: Span,
    days: Sequence[dt.date],
    bracket: tuple[dt.date, float] | None,
    lookup: Callable[[str, dt.date], tuple[Record | None, float | None]],
    names: Callable[[Record], bool],
) -> FirstCount:
    """The first record with a share count under the span's ticker, by bisection of the
    trading days `days` between the span's first sample, which has none, and `bracket`, its
    first sample with one (date and count), assuming that a record once carrying a count goes
    on carrying one. `lookup(ticker, day)` is the reference endpoint's record on the day and
    its share-class count. Every record asked for must name the span's security (`names`); one
    naming another stops the search. Each lookup is one request at most: the client caches."""
    if bracket is None:
        return FirstCount(span, None, None)
    first = span.samples[0]
    lo = days.index(first)
    hi = days.index(bracket[0])
    found = bracket
    probes: list[tuple[dt.date, float | None, str | None]] = []
    while hi - lo > 1:
        mid = (lo + hi) // 2
        record, count = lookup(span.ticker, days[mid])
        probes.append((days[mid], count, record.key if record is not None else None))
        if record is not None and record.identified and not names(record):
            return FirstCount(span, (first, bracket[0]), None, tuple(probes),
                              f"{span.ticker} on {days[mid]} names {record.name} "
                              f"({record.key}), not {span.name} ({span.key}): the search stops")
        if count:
            hi, found = mid, (days[mid], float(count))
        else:
            lo = mid
    return FirstCount(span, (first, bracket[0]), found, tuple(probes))


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
) -> Panel:
    """Assemble the panel. `bars(ticker, "adj" | "raw")` is a ticker's daily bars;
    `splits` the gate's classification of every split event inside the spans; `samples[key]`
    each plan key's share counts, (date, ticker, count), the searched first counts among them;
    `dividends[ticker]` each ticker's distributions; `member_days[key]` each plan key's runs of
    membership, (first, last), read through the renames; `market` the market series' adjusted
    and raw bars and its distributions. `dates` are the trading calendar, the market's."""
    dates = tuple(dates)
    n, m = len(dates), len(columns)
    at = {day: i for i, day in enumerate(dates)}
    shape = (n, m)

    def blank() -> np.ndarray:
        return np.full(shape, np.nan)

    prices, total, raw_prices = blank(), blank(), blank()
    volumes, adjusted_volumes, shares, highs, lows = blank(), blank(), blank(), blank(), blank()
    tickers = np.full(shape, "", dtype=object)
    membership = np.zeros(shape, dtype=bool)
    out = Panel(dates, tuple(columns), prices, total, raw_prices, volumes, adjusted_volumes,
                shares, highs, lows, tickers, membership, np.full(n, np.nan),
                np.full(n, np.nan))
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
        if priced:
            total[priced[0], j] = prices[priced[0], j]
            for s, t in zip(priced, priced[1:]):
                cash = sum(d.cash_amount for d in due.get(t, ()))
                reinvested = cash * prices[t, j] / raw_prices[t, j]
                total[t, j] = total[s, j] * (prices[t, j] + reinvested) / prices[s, j]
                out.distributions += len(due.get(t, ()))

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
        for i in range(int(assigned[0]), int(assigned[-1]) + 1):
            day = dates[i]
            for check in splits_on.get(day, ()):
                if np.isfinite(carried):
                    carried *= check.split.ratio
                    out.applied.append((column.key, day, check.split.ratio))
                else:
                    out.unapplied.append((column.key, day, "no count carried to it"))
            starts = span_of[i] >= 0 and (i == assigned[0] or span_of[i - 1] != span_of[i])
            if starts and np.isfinite(carried):
                span = column.spans[span_of[i]]
                before = column.spans[span_of[i] - 1] if span_of[i] > 0 else None
                pending = Carry(column.key, span.ticker, day, float(carried), None,
                                before is not None and before.key != span.key)
            for _ticker, count in by_day.get(day, ()):
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
