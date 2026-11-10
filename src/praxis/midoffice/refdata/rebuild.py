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
  its record agrees. A record with no identifier is no evidence, and nor is the change day's
  naming the ticker's previous holder: the next sample decides.
- **Splits.** Every split event a security's tickers record inside its spans is classified by
  its ratio -- whole numbers or not -- and two tests (`check_splits`): whether the share counts
  move by it, within 10%, and whether the raw price gaps by it on its execution date, within
  the same. A whole-number ratio the counts follow is a split, applied to the carried count on
  its execution date; samples after it still at the pre-split count are a filing's figure from
  before, stale, and the first that follows the ratio resets the count. A fractional ratio the
  counts leave unchanged is a price-only adjustment -- Polygon records a spin-off so, to adjust
  the parent's price history -- which the adjusted prices already carry. A test disagreeing is
  a contradiction, and stops the build, as does a split whose raw price does not gap; DuPont's
  are reported, not stopped on. Where neither test can speak, the ratio alone classifies the
  event, and it is reported as untested.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .identity import Record
from .polygon import Bar, Dividend, PolygonClient, PolygonError, PolygonNotFoundError, Split


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
    previous: str | None = None
    """The security the table has under the ticker before the span (`identity.PlanRow`)."""


def read_plan(path: Path | str) -> list[Span]:
    """The fetch plan `scripts/build_sp500_identity.py` writes (`*_fetch.csv`); a corrected
    plan's carries each span's previous holder too."""
    out = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(Span(
                key=row["key"], name=row["name"], ticker=row["ticker"],
                first=dt.date.fromisoformat(row["first"]),
                last=dt.date.fromisoformat(row["last"]),
                member_days=int(row["member_days"]),
                samples=tuple(dt.date.fromisoformat(d) for d in row["sample_dates"].split()),
                previous=row.get("previous") or None,
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
        share_class=results.get("share_class_figi") or None,
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
    """A span's first share sample that carries evidence, set against the security the table
    expects there."""

    span: Span
    record: Record | None
    passed: bool
    detail: str
    skipped: tuple[tuple[dt.date, str], ...] = ()
    """Samples read past before the one that decides, each with why."""


def check_identities(spans: Sequence[Span], fetched: Fetched) -> list[IdentityCheck]:
    """Per span, whether its first share sample -- the reference record under its ticker on its
    first day -- names the security the table expects. No record at all is a failure: the
    span's identity is then the table's word alone.

    Two samples are no evidence, and the next one decides: a record with neither a FIGI nor a
    CIK (`Record.identified`), which names nobody; and on the span's first day, a record naming
    the security that held the ticker before (`Span.previous`) -- the change day's, still
    carrying the identifier the ticker had. A failure says whether the record's share-class
    FIGI is one the expected security's own records carry."""
    share_classes: dict[str, set[str]] = {}
    for span in spans:
        for day in span.samples:
            record = fetched.records.get((span.ticker, day))
            if record is not None and record.identified and record.key == span.key \
                    and record.share_class:
                share_classes.setdefault(span.key, set()).add(record.share_class)
    out = []
    for span in spans:
        skipped: list[tuple[dt.date, str]] = []
        check = None
        for day in span.samples:
            if (span.ticker, day) not in fetched.records:
                check = IdentityCheck(span, None, False, f"{span.ticker} on {day}: not fetched")
                break
            record = fetched.records[(span.ticker, day)]
            if record is None:
                check = IdentityCheck(span, None, False,
                                      f"{span.ticker} on {day}: no reference record")
                break
            if not record.identified:
                skipped.append((day, f"{record.name or 'its record'} with neither a FIGI nor "
                                     f"a CIK, no evidence"))
                continue
            if day == span.first and span.previous is not None and record.key == span.previous:
                skipped.append((day, f"the change day's record, naming {record.key}, the "
                                     f"ticker's previous holder"))
                continue
            if record.key == span.key:
                check = IdentityCheck(span, record, True,
                                      f"{span.ticker} on {day} names {record.name}")
                break
            expected = share_classes.get(span.key, set())
            share = record.share_class
            note = ("" if not share else
                    f"; its share-class FIGI, {share}, is the expected security's"
                    if share in expected else
                    f"; its share-class FIGI, {share}, is not the expected security's "
                    f"({', '.join(sorted(expected)) or 'none on its records'})")
            check = IdentityCheck(span, record, False,
                                  f"{span.ticker} on {day} names {record.name} ({record.key}), "
                                  f"not {span.name} ({span.key}){note}")
            break
        if check is None:
            check = IdentityCheck(span, None, False,
                                  f"{span.ticker}: no sample carries an identifier")
        if skipped:
            check = IdentityCheck(check.span, check.record, check.passed, check.detail + (
                "; read past " + "; ".join(f"{day}, {why}" for day, why in skipped)),
                tuple(skipped))
        out.append(check)
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
    """One split event of a security, classified by its ratio -- whole numbers, as a split's
    are, or not -- and by two tests of what it did: whether the security's share counts moved
    by it, and whether its raw price gapped by it on the execution date."""

    key: str
    name: str
    split: Split
    before: tuple[dt.date, float] | None
    """The last share sample before the execution date."""
    after: tuple[dt.date, float] | None
    """The first on or after it."""
    whole: bool
    """No term of the ratio in lowest terms is over the largest term allowed."""
    counts: str | None
    """What the counts did: "follow" -- a sample on or after the execution date moved by the
    ratio, within the tolerance, every sample between still at the pre-split count; "stale" --
    a whole-number ratio's samples after it all still at the pre-split count, to the span's
    end; "unchanged" -- a fractional ratio's first sample after it within the tolerance of no
    change; "either" -- within it of both the ratio and no change, so the counts cannot tell;
    "neither"; or None, a side having no sample."""
    kind: str
    """"split" or "price-only" where the tests agree with the ratio, or by the ratio alone
    where they cannot speak; "contradiction" where one disagrees."""
    exempt: bool
    detail: str
    stale: tuple[tuple[dt.date, float], ...] = ()
    """Samples on or after a split still at the pre-split count: a filing's figure from before
    it, which does not reset the carried count."""
    followed: tuple[dt.date, float] | None = None
    """The first sample that follows the ratio, which resets it."""
    gap: tuple[dt.date, float, dt.date, float] | None = None
    """The raw close on the last trading day before the execution date and the raw open on the
    first on or after it, each with its date."""
    prices: str | None = None
    """What the raw price did: "gap" -- opened within the tolerance of the close before over the
    ratio; "no gap" -- within it of the close; "either"; "neither"; or None, a side having no
    price."""

    @property
    def tested(self) -> bool:
        """Whether a test beside the ratio confirms its kind: the counts following a split or its
        price gapping by it; the counts unchanged across a price-only adjustment. A
        contradiction is a test speaking."""
        if self.kind == "split":
            return self.counts == "follow" or self.prices == "gap"
        if self.kind == "price-only":
            return self.counts == "unchanged"
        return True

    @property
    def applies(self) -> bool:
        """Whether the build applies it to the carried share count: a split's ratio does, a
        price-only adjustment's does not -- the adjusted prices carry it -- and a contradiction
        stops the build."""
        return self.kind == "split"


def is_whole(split: Split, largest_term: int = LARGEST_TERM) -> bool:
    """Whether a split's ratio is one of whole numbers, no term over `largest_term` in lowest
    terms: 3-for-2 is, and so is 2,000-for-1,000; 1,000-for-1,281 is not."""
    ratio = Fraction(str(split.split_to)) / Fraction(str(split.split_from))
    return max(ratio.numerator, ratio.denominator) <= largest_term


def _near(moved: float, ratio: float, tolerance: float) -> tuple[bool, bool]:
    """Whether `moved` is within the tolerance of `ratio`, and of one."""
    return abs(moved / ratio - 1) <= tolerance, abs(moved - 1) <= tolerance


def classify(
    split: Split,
    before: tuple[dt.date, float] | None,
    later: Sequence[tuple[dt.date, float]],
    gap: tuple[dt.date, float, dt.date, float] | None = None,
    *,
    tolerance: float = TOLERANCE,
    largest_term: int = LARGEST_TERM,
) -> dict:
    """The fields of one split event's `SplitCheck` beyond its identity: whole, counts, stale,
    followed, after, prices, kind and detail.

    `before` is the last share sample before the execution date; `later` the samples on or
    after it, in date order, to the span's end or the security's next whole-number event; `gap`
    the raw close before and the raw open on the day (`raw_gap`).

    **The counts.** A whole-number ratio is followed if a sample moves by it, every sample
    before that one still at the pre-split count -- a filing's figure from before the split,
    stale, which does not reset the carried count: Polygon's exact counts, from 2025 Q3, lag a
    split by up to five months. Samples stale to the span's end leave the counts unable to
    speak. A fractional ratio is read at its first sample after, which must stay unchanged.

    **The price.** A split's raw price opens at the close before over the ratio, within the
    tolerance, on its execution date; one that does not gap stops the gate. A price-only
    adjustment's gap is reported, and does not decide it.

    **The kind.** A whole-number ratio the counts follow or the price gaps by, and that neither
    test contradicts, is a split; a fractional one the counts leave unchanged is price-only.
    Counts that move by neither the ratio nor nothing, or follow a fractional ratio, and a split
    whose price does not gap, are contradictions. Where neither test can speak, the ratio
    alone decides."""
    whole = is_whole(split, largest_term)
    form = (f"{split.ticker} {split.split_from:g}-to-{split.split_to:g} on "
            f"{split.execution_date} ({split.ratio:.4g}, "
            f"{'whole numbers' if whole else 'fractional'})")
    counts, stale, followed, decisive = None, [], None, None
    after = later[0] if later else None
    if before is not None and later:
        if whole:
            for day, count in later:
                follow, unchanged = _near(count / before[1], split.ratio, tolerance)
                decisive = (day, count)
                if follow and unchanged:
                    counts = "either"
                    break
                if follow:
                    counts, followed = "follow", (day, count)
                    break
                if not unchanged:
                    counts = "neither"
                    break
                stale.append((day, count))
            else:
                counts, decisive = "stale", None
        else:
            follow, unchanged = _near(after[1] / before[1], split.ratio, tolerance)
            counts = ("either" if follow and unchanged else "follow" if follow
                      else "unchanged" if unchanged else "neither")
    prices = None
    if gap is not None:
        follow, unchanged = _near(gap[3] / gap[1], 1 / split.ratio, tolerance)
        prices = ("either" if follow and unchanged else "gap" if follow
                  else "no gap" if unchanged else "neither")

    # What each test saw, in words.
    if before is None or not later:
        said = f"no share sample {'before' if before is None else 'on or after'} it"
    else:
        said = f"{before[1]:,.0f} on {before[0]}"
        if stale:
            said += (f", then {len(stale)} sample(s) still at it, to {stale[-1][0]} "
                     f"({stale[-1][1]:,.0f})")
        shown = decisive if whole else after
        if counts == "stale":
            said += ", to the span's end"
        elif shown is not None:
            said += f", {shown[1]:,.0f} on {shown[0]}, {shown[1] / before[1]:.3f} times"
    if gap is None:
        priced = "no raw price on a side"
    else:
        priced = (f"{gap[1]:,.2f} at the close of {gap[0]}, {gap[3]:,.2f} at the open of "
                  f"{gap[2]}, {gap[3] / gap[1]:.3f} times against {1 / split.ratio:.3f}")
    form += f": counts {said}; raw price {priced}"

    def out(kind: str, why: str) -> dict:
        return {"whole": whole, "counts": counts, "stale": tuple(stale), "followed": followed,
                "after": after, "prices": prices, "kind": kind, "detail": f"{form}; {why}"}

    tol = f"{tolerance:.0%}"
    if whole:
        if prices in ("no gap", "neither"):
            return out("contradiction", "contradiction, a whole-number ratio whose raw price "
                                        f"does not gap by it within {tol}")
        if counts == "neither":
            return out("contradiction", "contradiction, a whole-number ratio and the counts "
                                        "neither follow it nor stay unchanged")
        if counts == "follow":
            return out("split", "a split, the counts follow it"
                       + (f" after {len(stale)} stale sample(s)" if stale else "")
                       + (" and the price gaps by it" if prices == "gap" else ""))
        if prices == "gap":
            cannot = {"stale": "stale to the span's end", "either": f"within {tol} of both",
                      None: "without a sample on a side"}[counts]
            return out("split", f"a split, the price gaps by it; the counts {cannot}")
        return out("split", "untested, neither the counts nor the price can tell; a split by "
                            "the ratio alone")
    if counts == "unchanged":
        return out("price-only", f"price-only, the counts within {tol} of unchanged"
                   + ("; the price gaps by it" if prices == "gap" else
                      f"; the raw price: {prices or 'not read'}"))
    if counts in ("follow", "neither"):
        did = "follow it" if counts == "follow" else "neither follow it nor stay unchanged"
        return out("contradiction", f"contradiction, a fractional ratio and the counts {did}")
    return out("price-only", (f"untested, within {tol} of both the ratio and no change"
                              if counts == "either" else "untested, no share sample on a side")
               + "; price-only by the ratio alone")


def check_splits(
    spans: Sequence[Span],
    fetched: Fetched,
    *,
    gaps: Callable[[str, dt.date], tuple[dt.date, float, dt.date, float] | None] | None = None,
    exempt: Iterable[str] = ("DD",),
    tolerance: float = TOLERANCE,
    largest_term: int = LARGEST_TERM,
) -> list[SplitCheck]:
    """Every split event recorded under a ticker inside one of its security's spans, classified
    (`classify`) against the security's share counts -- the last sample before the execution
    date, and those on or after it up to the security's next whole-number event -- and its raw
    prices, `gaps(ticker, execution date)`. The samples are the security's own, under the
    tickers it traded under, so a count is never another security's. A contradiction under a
    ticker in `exempt` is reported but does not count against the gate: DuPont's."""
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
        events = sorted(
            ((split, span) for span in mine for split in fetched.splits.get(span.ticker, ())
             if span.first <= split.execution_date <= span.last),
            key=lambda pair: pair[0].execution_date,
        )
        wholes = [split.execution_date for split, _ in events if is_whole(split, largest_term)]
        for split, span in events:
            on = split.execution_date
            before = next((c for c in reversed(counts) if c[0] < on), None)
            stop = min((day for day in wholes if day > on), default=None)
            later = [c for c in counts if c[0] >= on and (stop is None or c[0] < stop)]
            gap = None if gaps is None else gaps(split.ticker, on)
            found = classify(split, before, later, gap, tolerance=tolerance,
                             largest_term=largest_term)
            out.append(SplitCheck(
                key, span.name, split, before, found["after"], found["whole"], found["counts"],
                found["kind"], split.ticker in exempt, found["detail"], found["stale"],
                found["followed"], gap, found["prices"],
            ))
    return sorted(out, key=lambda c: (c.split.execution_date, c.split.ticker))


def raw_gap(bars: Sequence[Bar], on: dt.date) -> tuple[dt.date, float, dt.date, float] | None:
    """From a ticker's raw bars, ascending: the close on the last trading day before `on` and
    the open on the first on or after it, each with its date; None where either is missing."""
    before = [b for b in bars if b.date < on]
    after = [b for b in bars if b.date >= on]
    if not before or not after or before[-1].close <= 0.0:
        return None
    return before[-1].date, before[-1].close, after[0].date, after[0].open


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
