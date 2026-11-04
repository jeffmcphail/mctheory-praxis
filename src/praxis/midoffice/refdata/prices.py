"""Assembling a `SecurityPanel` from Polygon, with the awkward parts made explicit.

Three things here are decisions rather than plumbing, and each costs something if taken the
other way.

**Two price fetches per security.**  Returns want split-adjusted prices; market
capitalisation and LPRI want what the stock actually traded at.  One series cannot serve
both, so this fetches both and the panel carries both.  See `descriptors.SecurityPanel` for
what mixing them costs.

**Shares outstanding is sampled, not fetched daily.**  Polygon serves it from a
point-in-time reference endpoint, one request per ticker per date, so a daily series over
five years for five hundred names would be well over half a million requests.  It is also
*reported* quarterly, so a daily series would be a quarterly series at higher resolution and
with no more information.  This samples on a grid and carries each value forward to the next
sample.  Carrying forward is correct -- the count genuinely did not change between filings.
Carrying *backward*, before the first sample, would be invention, so those days stay missing.

**A ticker that fails is recorded, not dropped.**  The code this replaces wrapped its fetch
in `except Exception: pass`, which turns a rate limit or a delisting into a security that is
quietly absent from every cross-section.  `LoadReport.failed` names each one and why.

The index supplies the trading calendar.  Using the union of whatever days the securities
happen to have would let one illiquid name introduce a date the market did not trade, and
every other security would show a missing return there.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .polygon import Bar, PolygonClient, PolygonError
from ..model.descriptors import SecurityPanel
from ..model.panel import Frequency

MARKET_PROXY = "SPY"
"""The S&P 500 ETF, used as the market series rather than the index itself.

Partly necessity: `I:SPX` needs an indices subscription, and a stocks plan returns 403 for it.

But it is also the better choice, which is worth saying so nobody "fixes" it later.  The
securities' returns here come from split- *and dividend-* adjusted prices, so they are total
returns.  `I:SPX` is a price index: it excludes dividends entirely.  Using it as the market
series would pair total-return securities with a price-return market, leaving roughly one and
a half to two percent a year of market return unexplained and biasing every beta estimated
against it.  SPY's adjusted series is a total return, so the two sides match.

`SP500_INDEX` is kept for anyone with an indices subscription who wants the index proper.
"""

SP500_INDEX = "I:SPX"
QUARTERLY_TRADING_DAYS = 63
"""A quarter, in trading days. The natural grid for a quantity reported quarterly."""


@dataclass(frozen=True)
class PanelRequest:
    """What to fetch. `estimate_requests` prices it before anything is spent."""

    start: dt.date
    end: dt.date
    tickers: tuple[str, ...]
    index: str = MARKET_PROXY
    shares_every: int = QUARTERLY_TRADING_DAYS
    fetch_unadjusted: bool = True

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("no tickers requested")
        if self.end < self.start:
            raise ValueError(f"end {self.end} precedes start {self.start}")
        if self.shares_every < 1:
            raise ValueError("shares_every must be at least one trading day")
        object.__setattr__(self, "tickers", tuple(dict.fromkeys(self.tickers)))


@dataclass(frozen=True)
class LoadReport:
    """What was fetched, what was not, and what that costs the model downstream."""

    requested: int = 0
    loaded: tuple[str, ...] = ()
    failed: Mapping[str, str] = field(default_factory=dict)
    trading_days: int = 0
    shares_sample_dates: tuple[dt.date, ...] = ()
    no_shares: tuple[str, ...] = ()
    partial_history: tuple[str, ...] = ()
    reference_gaps: Mapping[str, int] = field(default_factory=dict)
    requests_made: int = 0
    unadjusted: bool = True

    def qualifications(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.failed:
            # The reason matters more than the name: a delisting is expected and benign,
            # an auth failure is a plan boundary, and a parse failure is a defect. Reporting
            # only the ticker leaves the reader unable to tell which they have.
            shown = "; ".join(
                f"{name} ({self.failed[name].splitlines()[0][:80]})"
                for name in sorted(self.failed)[:3]
            )
            more = f" and {len(self.failed) - 3} more" if len(self.failed) > 3 else ""
            out.append(
                f"{len(self.failed)} of {self.requested} ticker(s) could not be loaded "
                f"and are absent from every cross-section: {shown}{more}"
            )
        if self.reference_gaps:
            worst = sorted(self.reference_gaps, key=self.reference_gaps.get, reverse=True)
            out.append(
                f"{len(self.reference_gaps)} ticker(s) had a shares-outstanding lookup "
                f"fail at one or more sample dates and kept their price history anyway, "
                f"which is usually a delisting inside the window: "
                + ", ".join(f"{t} ({self.reference_gaps[t]})" for t in worst[:5])
            )
        if self.no_shares:
            out.append(
                f"{len(self.no_shares)} ticker(s) reported no shares outstanding at any "
                f"sample date, so they carry no Size or Trading Activity exposure: "
                f"{', '.join(self.no_shares[:5])}"
            )
        if self.partial_history:
            out.append(
                f"{len(self.partial_history)} ticker(s) have returns for less than half "
                f"the window, usually a listing or delisting inside it: "
                f"{', '.join(self.partial_history[:5])}"
            )
        if not self.unadjusted:
            out.append(
                "unadjusted prices were not fetched, so LNCAP, LPRI and VLVR are on the "
                "split-adjusted basis and understate any security that split"
            )
        if self.trading_days and self.shares_sample_dates:
            span = self.trading_days / max(len(self.shares_sample_dates), 1)
            if span > 70:
                out.append(
                    f"shares outstanding was sampled every {span:.0f} trading days, wider "
                    f"than the quarterly cadence it is reported on, so some filings were "
                    f"stepped over"
                )
        return tuple(out)

    def describe(self) -> str:
        return (
            f"{len(self.loaded)} of {self.requested} securities over "
            f"{self.trading_days} trading day(s), {len(self.shares_sample_dates)} shares "
            f"sample(s), {self.requests_made} request(s)"
        )


def _estimate_trading_days(request: "PanelRequest") -> int:
    return int((request.end - request.start).days * 252 / 365) or 1


def estimate_requests(request: PanelRequest, trading_days: int | None = None) -> int:
    """How many API calls `build_security_panel` will make. Worth knowing beforehand.

    Five hundred names over five years is around ten thousand requests: fine on an
    unlimited-call plan, and not fine where calls are metered.
    """
    days = trading_days if trading_days is not None else _estimate_trading_days(request)
    # Mirror the builder's own grid rather than approximating it: it samples on
    # range(0, n, every) and appends the final day when the stride misses it. Computing
    # ceil(n/every) instead understated a 506-name run by 506 requests, which is exactly
    # the kind of error a cost estimate must not make.
    samples = len(_sample_slots(days, request.shares_every))
    per_ticker = (2 if request.fetch_unadjusted else 1) + samples
    return 1 + len(request.tickers) * per_ticker


def _sample_slots(n_days: int, every: int) -> list[int]:
    """Which day-indices get a shares-outstanding lookup. One definition, two callers."""
    if n_days <= 0:
        return []
    slots = list(range(0, n_days, every))
    if slots[-1] != n_days - 1:
        slots.append(n_days - 1)
    return slots


def _series(
    bars: Sequence[Bar], index: Mapping[dt.date, int], n: int, field_name: str
) -> np.ndarray:
    out = np.full(n, np.nan)
    for bar in bars:
        slot = index.get(bar.date)
        if slot is not None:
            out[slot] = getattr(bar, field_name)
    return out


def _carry_forward(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    last = np.nan
    for i in range(out.size):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    return out


def build_security_panel(
    client: PolygonClient,
    request: PanelRequest,
    *,
    on_progress=None,
) -> tuple[SecurityPanel, np.ndarray, LoadReport]:
    """Fetch and assemble. Returns the panel, the index return series, and the report.

    `on_progress` is called with `(done, total, ticker)` after each security, because five
    hundred names is a long wait with nothing on screen otherwise.
    """
    requests_made = 0
    index_bars = client.index_bars(request.index, request.start, request.end)
    requests_made += 1
    if not index_bars:
        raise PolygonError(
            f"the market series {request.index} returned no bars between {request.start} "
            f"and {request.end}; without it there is no trading calendar to build a panel on"
        )
    dates = tuple(bar.date for bar in index_bars)
    slot_of = {day: i for i, day in enumerate(dates)}
    n_days = len(dates)

    index_close = np.array([bar.close for bar in index_bars], dtype=float)
    market_returns = np.full(n_days, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        market_returns[1:] = np.where(
            index_close[:-1] > 0.0, index_close[1:] / index_close[:-1] - 1.0, np.nan
        )

    sample_slots = _sample_slots(n_days, request.shares_every)
    sample_dates = tuple(dates[i] for i in sample_slots)

    loaded: list[str] = []
    failed: dict[str, str] = {}
    no_shares: list[str] = []
    partial: list[str] = []
    reference_gaps: dict[str, int] = {}
    columns: list[dict[str, np.ndarray]] = []

    for position, ticker in enumerate(request.tickers, start=1):
        try:
            adjusted = client.daily_bars(
                ticker, request.start, request.end, adjusted=True
            )
            requests_made += 1
            if not adjusted:
                failed[ticker] = "no bars returned for the requested window"
                continue
            raw = adjusted
            if request.fetch_unadjusted:
                raw = client.daily_bars(
                    ticker, request.start, request.end, adjusted=False
                )
                requests_made += 1

            # A reference lookup that fails at one date must not discard the price history.
            # A company delisted mid-window -- acquired, merged, taken private -- has no
            # ticker record at the later sample dates, and letting that fail the whole
            # security would drop exactly the names point-in-time membership exists to keep.
            # Survivorship bias does not care which layer reintroduces it.
            shares = np.full(n_days, np.nan)
            reference_errors = 0
            for slot in sample_slots:
                try:
                    value = client.shares_outstanding(ticker, dates[slot])
                except PolygonError:
                    reference_errors += 1
                    value = None
                requests_made += 1
                if value:
                    shares[slot] = value
            shares = _carry_forward(shares)
            if not np.isfinite(shares).any():
                no_shares.append(ticker)
            if reference_errors:
                reference_gaps[ticker] = reference_errors

            close = _series(adjusted, slot_of, n_days, "close")
            if np.count_nonzero(np.isfinite(close)) < n_days // 2:
                partial.append(ticker)

            columns.append(
                {
                    "prices": close,
                    "unadjusted": _series(raw, slot_of, n_days, "close"),
                    "volumes": _series(adjusted, slot_of, n_days, "volume"),
                    "shares": shares,
                    "highs": _series(raw, slot_of, n_days, "high"),
                    "lows": _series(raw, slot_of, n_days, "low"),
                }
            )
            loaded.append(ticker)
        except PolygonError as error:
            failed[ticker] = str(error)
        if on_progress is not None:
            on_progress(position, len(request.tickers), ticker)

    if not loaded:
        raise PolygonError(
            f"every one of {len(request.tickers)} ticker(s) failed; first reason: "
            f"{next(iter(failed.values()), 'none recorded')}"
        )

    def stack(key: str) -> np.ndarray:
        return np.column_stack([column[key] for column in columns])

    panel = SecurityPanel(
        dates=dates,
        securities=tuple(loaded),
        prices=stack("prices"),
        unadjusted_prices=stack("unadjusted") if request.fetch_unadjusted else None,
        volumes=stack("volumes"),
        shares_outstanding=stack("shares"),
        highs=stack("highs"),
        lows=stack("lows"),
        frequency=Frequency.DAILY,
    )
    report = LoadReport(
        requested=len(request.tickers),
        loaded=tuple(loaded),
        failed=failed,
        trading_days=n_days,
        shares_sample_dates=sample_dates,
        no_shares=tuple(no_shares),
        partial_history=tuple(partial),
        reference_gaps=reference_gaps,
        requests_made=requests_made,
        unadjusted=request.fetch_unadjusted,
    )
    return panel, market_returns, report
