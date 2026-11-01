"""As-of dating: what was asked for, what was actually used, and the gap between them.

A risk computation runs on four dates -- model, exposure, market and security -- and each is
*requested* rather than given.  The data has the dates it has, so a request for 30 June is
answered with the most recent date on or before it.  That substitution is routine and correct.
What is not correct is losing it.

RAT resolves all four through `RATEngine.getLatestDate` and returns a bare `yyyyMMdd` string:

    DataRow[] rows = this.dbInfo.Tables[tableName].Select("date <= #" + latestDate + "#",
                                                          "date desc");
    return ((DateTime) rows[0]["date"]).ToString("yyyyMMdd");

The caller receives a date and cannot tell whether it is the one it asked for.  A model six
months stale and a model from yesterday are the same string to everything downstream, and the
risk numbers built on either look equally plausible.  This module returns the request, the
answer and the distance between them, so a run can be reconstructed rather than merely
repeated.

Two further things RAT leaves to chance and this does not.

**Ordering is enforced here, not assumed.**  `getLatestDate(tableName, int t)` -- the form
that answers `null` and the `T0`/`Tn` notation -- calls `Select()` with no sort argument and
indexes the result.  `DataTable.Select()` returns load order, so `T0` is the latest date only
if the query that filled the table sorted descending; and those queries live in parameters
(`TABLE_MODEL_DATES_QUERY` and friends), not in code.  The date form of the same function
passes `"date desc"` explicitly, so the author knew ordering mattered in one branch.  Drop the
`ORDER BY` from a config file and every `Tn` run silently uses the wrong day.

**An offset past the end of the calendar is a message, not an IndexOutOfRangeException.**
RAT checks `rows.Length == 0` and then indexes `rows[t]`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Mapping, Sequence

MODEL = "model"
EXPOSURE = "exposure"
MARKET = "market"
SECURITY = "security"

CALENDARS = (MODEL, EXPOSURE, MARKET, SECURITY)

_OFFSET = re.compile(r"^T(\d+)$", re.IGNORECASE)
_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")

LATEST = "latest"
"""The request meaning "whatever the most recent available date is"."""


class NoAvailableDate(LookupError):
    """Nothing in the calendar satisfies the request."""


class EmptyCalendar(ValueError):
    """A calendar with no dates cannot answer anything."""


def parse_request(requested: str | date | None) -> str | date:
    """Normalise a request to either a `date`, an offset string, or `LATEST`.

    Accepts what RAT accepts -- `None`, `yyyyMMdd`, `Tn` -- plus a real `date`, and refuses
    anything else rather than falling through to a default. RAT's resolution treats an
    unparseable string as a date literal and hands it to a `Select` filter, where it becomes a
    malformed-expression error a long way from the typo that caused it.
    """
    if requested is None or requested == "":
        return LATEST
    if isinstance(requested, datetime):
        return requested.date()
    if isinstance(requested, date):
        return requested
    text = str(requested).strip()
    if text.lower() == LATEST:
        return LATEST
    match = _OFFSET.match(text)
    if match:
        return f"T{int(match.group(1))}"
    match = _COMPACT.match(text)
    if match:
        year, month, day = (int(g) for g in match.groups())
        try:
            return date(year, month, day)
        except ValueError as exc:
            raise ValueError(f"{requested!r} is not a real date: {exc}") from exc
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"cannot read {requested!r} as a date request; expected yyyyMMdd, an ISO date, "
            f"'Tn' for n business days back, or 'latest'"
        ) from None


@dataclass(frozen=True)
class ResolvedDate:
    """One date: what was asked for, what was used, and how far apart they are."""

    calendar: str
    requested: str
    actual: date
    gap_days: int = 0
    offset: int | None = None

    @property
    def substituted(self) -> bool:
        """True when an explicit date was asked for and a different one was used.

        Asking for `latest` or `Tn` and receiving it is not a substitution: the request named
        a position in the calendar rather than a day, so there is nothing to have missed.
        """
        return self.gap_days > 0

    @property
    def is_stale(self) -> bool:
        """More than a long weekend behind the request. A judgement, deliberately crude."""
        return self.gap_days > 4

    def describe(self) -> str:
        if not self.substituted:
            return f"{self.calendar} date {self.actual:%Y-%m-%d}"
        return (
            f"{self.calendar} date {self.actual:%Y-%m-%d} was used for a request of "
            f"{self.requested} ({self.gap_days} day(s) earlier)"
        )


@dataclass(frozen=True)
class DateCalendar:
    """The dates one kind of data is available for, sorted newest first at construction.

    Sorting here rather than trusting the caller is the point: the whole `Tn` family indexes
    into this sequence, so an unsorted calendar makes `T0` mean an arbitrary date. RAT holds
    that guarantee in a configuration file's `ORDER BY` and never checks it.
    """

    name: str
    dates: tuple[date, ...]

    def __post_init__(self) -> None:
        parsed: list[date] = []
        for d in self.dates:
            parsed.append(d.date() if isinstance(d, datetime) else d)
        if len(set(parsed)) != len(parsed):
            dupes = sorted({d for d in parsed if parsed.count(d) > 1})
            raise ValueError(
                f"{self.name} calendar has duplicate date(s) {[str(d) for d in dupes[:5]]}; "
                f"a duplicate makes the Tn offsets ambiguous"
            )
        object.__setattr__(self, "dates", tuple(sorted(parsed, reverse=True)))

    @classmethod
    def of(cls, name: str, dates: Iterable[str | date]) -> "DateCalendar":
        """Build from anything `parse_request` reads as a literal date."""
        out: list[date] = []
        for d in dates:
            parsed = parse_request(d)
            if not isinstance(parsed, date):
                raise ValueError(
                    f"{name} calendar cannot contain {d!r}: a calendar holds literal dates, "
                    f"not requests like {parsed!r}"
                )
            out.append(parsed)
        return cls(name=name, dates=tuple(out))

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def latest(self) -> date:
        self._require_dates()
        return self.dates[0]

    @property
    def earliest(self) -> date:
        self._require_dates()
        return self.dates[-1]

    def _require_dates(self) -> None:
        if not self.dates:
            raise EmptyCalendar(f"the {self.name} calendar has no dates")

    def resolve(self, requested: str | date | None = None) -> ResolvedDate:
        """Answer a request, recording what was asked and what was used."""
        self._require_dates()
        parsed = parse_request(requested)

        if parsed == LATEST:
            return ResolvedDate(self.name, LATEST, self.dates[0], 0, offset=0)

        if isinstance(parsed, str):  # "Tn"
            n = int(parsed[1:])
            if n >= len(self.dates):
                raise NoAvailableDate(
                    f"the {self.name} calendar holds {len(self.dates)} date(s) "
                    f"({self.earliest:%Y-%m-%d} to {self.latest:%Y-%m-%d}), so {parsed} is "
                    f"past its start"
                )
            return ResolvedDate(self.name, parsed, self.dates[n], 0, offset=n)

        on_or_before = [d for d in self.dates if d <= parsed]
        if not on_or_before:
            raise NoAvailableDate(
                f"the {self.name} calendar has no date on or before "
                f"{parsed:%Y-%m-%d}; its earliest is {self.earliest:%Y-%m-%d}"
            )
        actual = on_or_before[0]
        return ResolvedDate(
            calendar=self.name,
            requested=f"{parsed:%Y-%m-%d}",
            actual=actual,
            gap_days=(parsed - actual).days,
            offset=self.dates.index(actual),
        )


@dataclass(frozen=True)
class AsOf:
    """The four dates a risk computation runs on, each resolved and recorded.

    The four-date interface is the engine's real signature, correcting the roadmap's earlier
    single-date one: a model estimated on one date can be applied to exposures from another,
    priced on a third, against a security master from a fourth. RAT keys its caches on pairs
    of them (`getFundGroupKey`, `getRiskModelKey`, joined with a pipe), which is the clearest
    statement in the source that they are genuinely independent.
    """

    model: ResolvedDate
    exposure: ResolvedDate
    market: ResolvedDate
    security: ResolvedDate

    @classmethod
    def resolve(
        cls,
        calendars: Mapping[str, DateCalendar],
        *,
        model: str | date | None = None,
        exposure: str | date | None = None,
        market: str | date | None = None,
        security: str | date | None = None,
    ) -> "AsOf":
        missing = [c for c in CALENDARS if c not in calendars]
        if missing:
            raise KeyError(
                f"no calendar supplied for {missing}; all four of {list(CALENDARS)} are "
                f"needed because a run can mix dates across them"
            )
        requests = {
            MODEL: model,
            EXPOSURE: exposure,
            MARKET: market,
            SECURITY: security,
        }
        return cls(**{name: calendars[name].resolve(requests[name]) for name in CALENDARS})

    def __iter__(self):
        return iter((self.model, self.exposure, self.market, self.security))

    @property
    def is_exact(self) -> bool:
        return not any(d.substituted for d in self)

    def substitutions(self) -> tuple[ResolvedDate, ...]:
        return tuple(d for d in self if d.substituted)

    def qualifications(self) -> tuple[str, ...]:
        """What a reader should know about the dates these numbers were computed on."""
        notes = [d.describe() for d in self.substitutions()]
        stale = [d for d in self if d.is_stale]
        if stale:
            notes.append(
                f"{len(stale)} date(s) fell more than a long weekend behind the request: "
                + ", ".join(f"{d.calendar} by {d.gap_days} days" for d in stale)
            )
        return tuple(notes)

    def key(self) -> str:
        """A content key for these four dates, for caching or for naming an output.

        RAT builds two pipe-joined pairs. One key over all four is the same idea without the
        implication that the pairs are independent of each other.
        """
        return "|".join(f"{d.actual:%Y%m%d}" for d in self)

    def describe(self) -> str:
        return ", ".join(f"{d.calendar}={d.actual:%Y-%m-%d}" for d in self)
