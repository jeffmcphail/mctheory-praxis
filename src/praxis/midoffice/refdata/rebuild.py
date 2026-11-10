"""The S&P 500 panel rebuilt on securities: the fetch the identity table plans, and the checks
that must pass before anything is built from it.

`identity.fetch_plan` says, for every security that is an index member somewhere in the
window, each ticker it traded under, over which trading days, and when to sample its share
count -- inside its own spans only, each span's first and last day among them. `fetch` asks
Polygon for all of it through `PolygonClient`'s cache, so an interrupted run resumes where it
stopped and a finished one is offline: each ticker's prices over the window, adjusted and not;
the reference record at every sample; each ticker's split events; and each ticker's cash
distributions, from which the build makes total returns -- the adjusted prices carry none. The
market series is fetched as a ticker is, since the market model regresses on its total return.

Two checks stand between the fetch and the build, and either failing stops it:

- **Identity.** The first share sample of every span is a reference record under the span's
  ticker on its first day. It must name the security the table expects there (`check_identities`).
  A ticker the events gave for a span -- CRHC for CRH, the presumed CASY -- is believed only once
  its record agrees.
- **Splits.** Every split event a security's tickers record inside its spans is classified by
  two tests (`check_splits`): whether its ratio is one of whole numbers, and whether the share
  counts either side move by it, within 10%. A whole-number ratio the counts follow is a split,
  applied to the carried count; a fractional one they leave unchanged is a price-only
  adjustment -- Polygon records a spin-off so, to adjust the parent's price history -- which the
  adjusted prices already carry. The two tests disagreeing is a contradiction, and stops the
  build; DuPont's are reported, not stopped on. Where the counts cannot test an event, the
  ratio alone classifies it, and it is reported as untested.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .identity import Record
from .polygon import Dividend, PolygonClient, PolygonError, PolygonNotFoundError, Split


@dataclass(frozen=True)
class Span:
    """One line of the fetch plan: a security under one ticker over one span of the window."""

    key: str
    name: str
    ticker: str
    first: dt.date
    last: dt.date
    member_days: int
    samples: tuple[dt.date, ...]


def read_plan(path: Path | str) -> list[Span]:
    """The fetch plan `scripts/build_sp500_identity.py` writes (`*_fetch.csv`)."""
    out = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(Span(
                key=row["key"], name=row["name"], ticker=row["ticker"],
                first=dt.date.fromisoformat(row["first"]),
                last=dt.date.fromisoformat(row["last"]),
                member_days=int(row["member_days"]),
                samples=tuple(dt.date.fromisoformat(d) for d in row["sample_dates"].split()),
            ))
    return out


def record_of(ticker: str, on: dt.date, results: Mapping | None) -> Record | None:
    """A reference endpoint answer as an identity `Record`, or None where there was none."""
    if not results:
        return None
    list_date = results.get("list_date")
    return Record(
        ticker=ticker, on=on,
        figi=results.get("composite_figi") or None,
        cik=results.get("cik") or None,
        name=results.get("name") or None,
        kind=results.get("type") or None,
        list_date=dt.date.fromisoformat(str(list_date)[:10]) if list_date else None,
    )


@dataclass
class Fetched:
    """What the fetch brought back, and what it could not."""

    bars: dict[tuple[str, str], int] = field(default_factory=dict)
    """Daily bars on disk per (ticker, "adj" or "raw"), counted rather than held: the build
    reads them back from the cache, and holding every ticker's cost the fetch about 560 MB."""
    splits: dict[str, tuple[Split, ...]] = field(default_factory=dict)
    dividends: dict[str, tuple[Dividend, ...]] = field(default_factory=dict)
    """Each ticker's cash distributions, the market series' among them."""
    records: dict[tuple[str, dt.date], Record | None] = field(default_factory=dict)
    shares: dict[tuple[str, dt.date], float | None] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)
    """(what was asked, why it failed): a request that failed for any reason but "not found",
    which is an answer. A re-run asks for these again."""
    requests: int = 0
    """Requests that went over the network this run; the rest were read from the cache."""


def fetch(
    client: PolygonClient,
    spans: Sequence[Span],
    start: dt.date,
    end: dt.date,
    *,
    market: str | None = None,
    cached: Callable[[str], bool] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Fetched:
    """Everything the plan needs, through the client's cache.

    `start` and `end` are the window every ticker's prices are asked over -- the panel fetch's
    own, so the 506 tickers it holds are read back rather than asked again. `market` is the
    market series' ticker, SPY, asked for as every ticker is: prices, split events and
    distributions. `cached(key)` says whether a cache key is on disk, to count the requests that
    went over the network.
    """
    out = Fetched()
    tickers = list(dict.fromkeys(span.ticker for span in spans))
    if market is not None and market not in tickers:
        tickers.append(market)
    samples = list(dict.fromkeys((span.ticker, day) for span in spans for day in span.samples))
    total = 4 * len(tickers) + len(samples)
    done = 0

    def tick(what: str) -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(done, total, what)

    def counted(key: str) -> None:
        if cached is not None and not cached(key):
            out.requests += 1

    for ticker in tickers:
        for adjusted, tag in ((True, "adj"), (False, "raw")):
            counted(f"bars-{ticker}-{start}-{end}-{tag}")
            try:
                bars = client.daily_bars(ticker, start, end, adjusted=adjusted)
                out.bars[(ticker, tag)] = len(bars)
            except PolygonError as error:
                out.failures.append((f"{ticker} prices ({tag})", str(error)))
            tick(ticker)
        counted(f"splits-{ticker}")
        try:
            out.splits[ticker] = client.splits(ticker)
        except PolygonNotFoundError:
            out.splits[ticker] = ()
        except PolygonError as error:
            out.failures.append((f"{ticker} splits", str(error)))
        tick(ticker)
        counted(f"dividends-{ticker}")
        try:
            out.dividends[ticker] = client.dividends(ticker)
        except PolygonNotFoundError:
            out.dividends[ticker] = ()
        except PolygonError as error:
            out.failures.append((f"{ticker} dividends", str(error)))
        tick(ticker)
    for ticker, day in samples:
        counted(f"ref-{ticker}-{day}")
        try:
            results = client.ticker_details(ticker, day)
        except PolygonNotFoundError:
            results = None
        except PolygonError as error:
            out.failures.append((f"{ticker} record on {day}", str(error)))
            tick(ticker)
            continue
        record = record_of(ticker, day, results)
        out.records[(ticker, day)] = record
        value = (results or {}).get("share_class_shares_outstanding")
        out.shares[(ticker, day)] = float(value) if value else None
        tick(ticker)
    return out


@dataclass(frozen=True)
class IdentityCheck:
    """A span's first share sample, set against the security the table expects there."""

    span: Span
    record: Record | None
    passed: bool
    detail: str


def check_identities(spans: Sequence[Span], fetched: Fetched) -> list[IdentityCheck]:
    """Per span, whether its first share sample -- the reference record under its ticker on its
    first day -- names the security the table expects. No record at all is a failure too: the
    span's identity is then the table's word alone."""
    out = []
    for span in spans:
        day = span.samples[0]
        if (span.ticker, day) not in fetched.records:
            out.append(IdentityCheck(span, None, False, f"{span.ticker} on {day}: not fetched"))
            continue
        record = fetched.records[(span.ticker, day)]
        if record is None:
            out.append(IdentityCheck(span, None, False,
                                     f"{span.ticker} on {day}: no reference record"))
        elif record.key != span.key:
            out.append(IdentityCheck(
                span, record, False,
                f"{span.ticker} on {day} names {record.name} ({record.key}), not "
                f"{span.name} ({span.key})",
            ))
        else:
            out.append(IdentityCheck(span, record, True,
                                     f"{span.ticker} on {day} names {record.name}"))
    return out


TOLERANCE = 0.10
"""How near a share count must come to what a split predicts, or to no change at all, to have
followed it or stayed unchanged: 10%, a quarter's buybacks and issuance being far less."""

LARGEST_TERM = 100
"""The largest term a whole-number ratio has in lowest terms. A split's are small -- 2-for-1,
3-for-2, 50-for-1, 1-for-5 -- and Polygon writes a price-only adjustment in thousandths or
hundredths, 1,000 to 1,281 or 100 to 239, whole numbers in form that in lowest terms run to
hundreds."""


@dataclass(frozen=True)
class SplitCheck:
    """One split event of a security, classified by two tests: whether its ratio is one of
    whole numbers, as a split's is, and whether the security's share counts either side of it
    move by it."""

    key: str
    name: str
    split: Split
    before: tuple[dt.date, float] | None
    after: tuple[dt.date, float] | None
    whole: bool
    """No term of the ratio in lowest terms is over the largest term allowed."""
    counts: str | None
    """What the counts did across it: "follow" -- moved by the ratio, within the tolerance;
    "unchanged" -- within it of no change; "either" -- within it of both, so the counts cannot
    tell; "neither"; or None, a side having no sample."""
    kind: str
    """"split" or "price-only" where the two tests agree, or by the ratio alone where the
    counts cannot test it; "contradiction" where the tests disagree."""
    exempt: bool
    detail: str

    @property
    def tested(self) -> bool:
        return self.counts in ("follow", "unchanged", "neither")

    @property
    def applies(self) -> bool:
        """Whether the build applies it to the carried share count: a split's ratio does, a
        price-only adjustment's does not -- the adjusted prices carry it -- and a contradiction
        is not applied either: the samples rule there."""
        return self.kind == "split"


def is_whole(split: Split, largest_term: int = LARGEST_TERM) -> bool:
    """Whether a split's ratio is one of whole numbers, no term over `largest_term` in lowest
    terms: 3-for-2 is, and so is 2,000-for-1,000; 1,000-for-1,281 is not."""
    ratio = Fraction(str(split.split_to)) / Fraction(str(split.split_from))
    return max(ratio.numerator, ratio.denominator) <= largest_term


def classify(
    split: Split,
    before: tuple[dt.date, float] | None,
    after: tuple[dt.date, float] | None,
    *,
    tolerance: float = TOLERANCE,
    largest_term: int = LARGEST_TERM,
) -> tuple[bool, str | None, str, str]:
    """(whole, counts, kind, detail) for one split event and the share counts either side.

    A whole-number ratio the counts follow is a split; a fractional one they leave within the
    tolerance of unchanged is a price-only adjustment, a spin-off or the like. Any other
    pairing -- a whole-number ratio the counts do not follow, a fractional one they follow or
    move by neither it nor nothing -- is a contradiction. Where a side has no sample, or the
    counts sit within the tolerance of both the ratio and no change, the counts cannot test it,
    and the ratio alone decides."""
    whole = is_whole(split, largest_term)
    by_ratio = "split" if whole else "price-only"
    form = (f"{split.ticker} {split.split_from:g}-to-{split.split_to:g} on "
            f"{split.execution_date} ({split.ratio:.4g}, "
            f"{'whole numbers' if whole else 'fractional'})")
    if before is None or after is None:
        side = "before" if before is None else "after"
        return whole, None, by_ratio, (f"{form}: untested, no share sample {side} it; "
                                       f"{by_ratio} by the ratio alone")
    moved = after[1] / before[1]
    follow = abs(moved / split.ratio - 1) <= tolerance
    unchanged = abs(moved - 1) <= tolerance
    counts = ("either" if follow and unchanged else "follow" if follow
              else "unchanged" if unchanged else "neither")
    form += (f": {before[1]:,.0f} on {before[0]}, {after[1]:,.0f} on {after[0]}, "
             f"{moved:.3f} times")
    if counts == "either":
        return whole, counts, by_ratio, (
            f"{form}; untested, within {tolerance:.0%} of both the ratio and no change; "
            f"{by_ratio} by the ratio alone")
    if whole and counts == "follow":
        return whole, counts, "split", f"{form}; a split, the counts follow it"
    if not whole and counts == "unchanged":
        return whole, counts, "price-only", (
            f"{form}; price-only, the counts within {tolerance:.0%} of unchanged")
    did = {"follow": "follow it", "unchanged": "stay unchanged",
           "neither": "neither follow it nor stay unchanged"}[counts]
    return whole, counts, "contradiction", (
        f"{form}; contradiction, a {'whole-number' if whole else 'fractional'} ratio and the "
        f"counts {did}")


def check_splits(
    spans: Sequence[Span],
    fetched: Fetched,
    *,
    exempt: Iterable[str] = ("DD",),
    tolerance: float = TOLERANCE,
    largest_term: int = LARGEST_TERM,
) -> list[SplitCheck]:
    """Every split event recorded under a ticker inside one of its security's spans, classified
    (`classify`) against the security's share count: the last sample before the execution date
    and the first after it. The samples are the security's own, under the tickers it traded
    under, so a count is never another security's. A contradiction under a ticker in `exempt`
    is reported but does not count against the gate: DuPont's."""
    exempt = set(exempt)
    by_key: dict[str, list[Span]] = {}
    for span in spans:
        by_key.setdefault(span.key, []).append(span)
    out = []
    for key, mine in by_key.items():
        counts = sorted(
            (day, fetched.shares.get((span.ticker, day)))
            for span in mine for day in span.samples
            if fetched.shares.get((span.ticker, day))
        )
        for span in mine:
            for split in fetched.splits.get(span.ticker, ()):
                if not span.first <= split.execution_date <= span.last:
                    continue
                before = next((c for c in reversed(counts) if c[0] < split.execution_date),
                              None)
                after = next((c for c in counts if c[0] > split.execution_date), None)
                whole, moved, kind, detail = classify(
                    split, before, after, tolerance=tolerance, largest_term=largest_term)
                out.append(SplitCheck(key, span.name, split, before, after, whole, moved, kind,
                                      split.ticker in exempt, detail))
    return sorted(out, key=lambda c: (c.split.execution_date, c.split.ticker))


@dataclass(frozen=True)
class Gate:
    """Whether the build may go ahead: every identity check passed, and no split event outside
    the exempt ones is a contradiction."""

    identity_failures: tuple[IdentityCheck, ...]
    contradictions: tuple[SplitCheck, ...]
    untested: tuple[SplitCheck, ...]

    @property
    def passed(self) -> bool:
        return not self.identity_failures and not self.contradictions


def gate(identities: Sequence[IdentityCheck], splits: Sequence[SplitCheck]) -> Gate:
    return Gate(
        identity_failures=tuple(c for c in identities if not c.passed),
        contradictions=tuple(c for c in splits if c.kind == "contradiction" and not c.exempt),
        untested=tuple(c for c in splits if not c.tested),
    )
