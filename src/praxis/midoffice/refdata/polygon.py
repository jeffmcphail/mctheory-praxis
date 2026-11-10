"""A Polygon.io client, consolidated out of the ad-hoc calls in `engines/`.

Four things this fixes relative to the inline versions it replaces.

**Failures are raised, not swallowed.**  `engines/pairs_strategy.py` wraps its fetch in
`except Exception: pass`, so a rate limit, an auth failure or a typo in a ticker produces a
silently absent series rather than an error.  In a risk panel that is the worst possible
outcome: the model runs, the security is quietly missing from every cross-section, and
nothing says why.  Here a bad status raises `PolygonError` and names the ticker.

**The API key never reaches an exception, a log line or a cache path.**  The key is a query
parameter, so any naive error message that includes the URL leaks it into whatever collects
logs.  `_redact` strips it from everything that escapes this module.

**Adjusted and unadjusted are separate calls, deliberately.**  Returns need adjusted prices,
so a split is not a -75% day.  Market capitalisation needs *unadjusted* prices, because it is
paired with point-in-time shares outstanding, and an adjusted price against a pre-split share
count undercounts by the split factor.  Polygon's own `market_cap` field shows what that
mistake looks like: at 2020-06-30 it reports AAPL at $5.99tn, which is the unadjusted
pre-split price of $364.80 multiplied by the *post-split* share count of 16.4bn.  Four times
the true figure.  Mixing the two bases is easy and silent, so this client makes you say which
one you want.

**The transport is injectable.**  Every test in this repo runs without a network, so the
HTTP layer is a Protocol and the tests use a recorded fake.

Why this lives in `refdata/` and not in `praxis.loaders`: that package's `__init__` is a
511-line YFinance-to-DuckDB ingestion module, so importing anything beneath it pulls in
duckdb, PraxisLogger and SecurityMaster.  This is an HTTP client with no database in it and
no business acquiring one through its own import path.  The two are also different kinds of
thing -- `loaders` moves vendor data into working and fact tables, this just speaks to an
API -- and a client that happens to feed a loader is not itself one.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

BASE = "https://api.polygon.io"

AUTHORISED_HISTORY_NOTE = (
    "Polygon returns NOT_AUTHORIZED rather than an empty result for dates outside the "
    "plan's history window, so a short panel is a subscription limit and not a gap in the "
    "market"
)


class PolygonError(RuntimeError):
    """A request failed, or returned a status this client will not interpret."""


class PolygonNotFoundError(PolygonError):
    """Polygon knows nothing under the id asked for: an HTTP 404, or a NOT_FOUND status.

    A PolygonError still, so every caller that catches those keeps working. Separate because
    for one endpoint it is an answer rather than a failure: the ticker events endpoint covers
    listed securities only, and answers 404 "No events found for given ID" for a delisted one.
    """


_KEY_PARAM = re.compile(r"([?&]apiKey=)[^&\s]*", re.IGNORECASE)


def _redact(text: str) -> str:
    """Remove the API key from anything that escapes this module.

    Matches the parameter in either position.  The first version of this split on "&" and
    redacted any part beginning with "apiKey=", which worked only because this client always
    appends the key last -- and would have leaked it from any URL where apiKey came first,
    which is how a human writes one by hand.  A test of the retry path caught it.
    """
    return _KEY_PARAM.sub(r"\1<redacted>", text)


def _not_found_detail(error: urllib.error.HTTPError) -> str:
    """The server's own message on a 404, when its body is Polygon's JSON -- "No events found
    for given ID" says something "HTTP 404" does not -- redacted like everything else."""
    try:
        body = json.loads(error.read().decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - no body, or not JSON: the status line is all there is
        return ""
    message = body.get("message") or body.get("error") if isinstance(body, dict) else None
    return f": {_redact(str(message))}" if message else ""


class Transport(Protocol):
    """Anything that can turn a URL into parsed JSON."""

    def get_json(self, url: str) -> Mapping: ...


@dataclass(frozen=True)
class HttpTransport:
    """The real one. Retries on transient failures with a bounded backoff."""

    timeout: float = 30.0
    retries: int = 3
    backoff: float = 1.5

    def get_json(self, url: str) -> Mapping:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                # 429 and 5xx are worth retrying; a 401 or 404 will not improve.
                if error.code == 404:
                    raise PolygonNotFoundError(
                        f"HTTP 404 for {_redact(url)}{_not_found_detail(error)}"
                    ) from None
                if error.code not in (429, 500, 502, 503, 504):
                    raise PolygonError(
                        f"HTTP {error.code} for {_redact(url)}"
                    ) from None
                last = error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last = error
            if attempt < self.retries - 1:
                time.sleep(self.backoff**attempt)
        raise PolygonError(
            f"{self.retries} attempt(s) failed for {_redact(url)}: {type(last).__name__}"
        ) from None


@dataclass(frozen=True)
class Bar:
    """One daily bar. `adjusted` records which basis it is on, because it matters."""

    date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted: bool
    vwap: float | None = None
    transactions: int | None = None


@dataclass(frozen=True)
class Split:
    """One split event: on `execution_date`, `split_from` shares become `split_to`."""

    execution_date: dt.date
    split_from: float
    split_to: float
    ticker: str

    @property
    def ratio(self) -> float:
        """What a share count multiplies by on the execution date: 10 for a ten-for-one."""
        return self.split_to / self.split_from


@dataclass(frozen=True)
class Dividend:
    """One cash distribution under a ticker: `cash_amount` per share, in `currency`, to the
    holders of a share before `ex_date`, the first day it trades without it. `kind` is
    Polygon's `dividend_type`: "CD" a regular dividend, "SC" a special one, "LT" and "ST"
    capital gains; `frequency` a year's count of them, 0 for a one-off."""

    ex_date: dt.date
    cash_amount: float
    currency: str | None
    kind: str | None
    frequency: int | None
    pay_date: dt.date | None
    ticker: str


@dataclass
class PolygonClient:
    """Rate-limited, optionally disk-cached access to the endpoints this build needs."""

    api_key: str
    transport: Transport = field(default_factory=HttpTransport)
    min_interval: float = 0.0
    cache_dir: Path | None = None
    _last_call: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise PolygonError(
                "no API key; set POLYGON_API_KEY in .env (not POLYGONSCAN_API_KEY or "
                "POLYGON_RPC, which are the blockchain)"
            )
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ plumbing

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.cache_dir / f"{safe}.json"

    def _get(self, path: str, params: Mapping[str, object], cache_key: str) -> Mapping:
        """One request, or its cached answer. A not-found answer is cached too and raised
        again on reading: a ticker absent on a date stays absent, and a resumed fetch must not
        ask for it twice."""
        cached = self._cache_path(cache_key)
        if cached is not None and cached.exists():
            payload = json.loads(cached.read_text(encoding="utf-8"))
            if str(payload.get("status", "")).upper() == "NOT_FOUND":
                raise PolygonNotFoundError(
                    f"{cache_key}: {payload.get('message') or 'not found'} (cached)"
                )
            return payload

        query = "".join(f"{k}={v}&" for k, v in params.items())
        url = f"{BASE}{path}?{query}apiKey={self.api_key}"

        if self.min_interval > 0.0:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        try:
            payload = self.transport.get_json(url)
        except PolygonNotFoundError as error:
            self._last_call = time.monotonic()
            self._remember_not_found(cached, str(error))
            raise
        self._last_call = time.monotonic()

        status = str(payload.get("status", "")).upper()
        if status == "NOT_FOUND":
            error = PolygonNotFoundError(
                f"status 'NOT_FOUND' for {_redact(url)}: "
                f"{payload.get('error') or payload.get('message') or 'no detail given'}"
            )
            self._remember_not_found(cached, str(error))
            raise error
        if status == "NOT_AUTHORIZED":
            raise PolygonError(
                f"NOT_AUTHORIZED for {_redact(url)} -- {AUTHORISED_HISTORY_NOTE}"
            )
        if status not in ("OK", "DELAYED", ""):
            raise PolygonError(
                f"status {status!r} for {_redact(url)}: "
                f"{payload.get('error') or payload.get('message') or 'no detail given'}"
            )
        if cached is not None:
            cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def has_cached(self, cache_key: str) -> bool:
        """Whether an answer under this cache key is on disk, found or not found."""
        path = self._cache_path(cache_key)
        return path is not None and path.exists()

    @staticmethod
    def _remember_not_found(cached: Path | None, message: str) -> None:
        # The message has been through _redact already; nothing else of the URL is kept.
        if cached is not None:
            cached.write_text(
                json.dumps({"status": "NOT_FOUND", "message": message}), encoding="utf-8"
            )

    # ------------------------------------------------------------------ endpoints

    def daily_bars(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
        *,
        adjusted: bool = True,
    ) -> tuple[Bar, ...]:
        """Daily OHLCV. `adjusted` is explicit because the two bases are not interchangeable.

        Use `adjusted=True` for returns and `adjusted=False` for anything multiplied by a
        point-in-time share count -- see the module docstring for what mixing them costs.
        """
        payload = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000},
            cache_key=f"bars-{ticker}-{start}-{end}-{'adj' if adjusted else 'raw'}",
        )
        rows = payload.get("results") or []
        bars = [
            Bar(
                date=dt.datetime.fromtimestamp(
                    row["t"] / 1000.0, tz=dt.timezone.utc
                ).date(),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row.get("v", 0.0)),
                adjusted=adjusted,
                vwap=float(row["vw"]) if "vw" in row else None,
                transactions=int(row["n"]) if "n" in row else None,
            )
            for row in rows
        ]
        # Polygon returns ascending when asked, but a panel that silently depends on that is
        # a panel that breaks quietly when it does not.
        return tuple(sorted(bars, key=lambda b: b.date))

    def ticker_details(self, ticker: str, on: dt.date) -> Mapping:
        """The reference record as of a date: shares outstanding, SIC code, the lot.

        One cache entry serves every caller, so asking for an industry classification after
        a panel fetch costs nothing -- the response is already on disk.
        """
        payload = self._get(
            f"/v3/reference/tickers/{ticker}",
            {"date": on.isoformat()},
            cache_key=f"ref-{ticker}-{on}",
        )
        return payload.get("results") or {}

    def ticker_events(self, identifier: str) -> Mapping | None:
        """Every ticker a listed security has traded under, with the date each began:
        `/vX/reference/tickers/{id}/events`, by ticker, CUSIP or composite FIGI.

        The result carries the security's `name`, `composite_figi` and `cik`, and `events`,
        each `{"type": "ticker_change", "date": ..., "ticker_change": {"ticker": ...}}`. A
        ticker resolves to whoever holds it now, not to whoever held it on some earlier date.

        None where Polygon has no events for the id, which is its answer for a delisted
        security -- the endpoint covers listed ones only. That answer is cached as well as a
        found one, so a rebuild asks each id once and every later run is offline.
        """
        try:
            payload = self._get(
                f"/vX/reference/tickers/{identifier}/events", {},
                cache_key=f"events-{identifier}",
            )
        except PolygonNotFoundError:
            return None
        return payload.get("results") or {}

    def splits(self, ticker: str) -> tuple["Split", ...]:
        """Every split event Polygon records under a ticker, by execution date:
        `/v3/reference/splits?ticker=...`. For a split, a share count multiplies by
        `Split.ratio` on the execution date, and the traded price divides by it.

        Not every event is a split. Polygon records a spin-off here too, as a fractional ratio
        that adjusts the parent's price history -- DuPont's 2.39 for Qnity, GE's 1.281 for GE
        HealthCare -- and the parent's share count does not move across it.
        `rebuild.check_splits` tells the two apart.
        """
        payload = self._get(
            "/v3/reference/splits", {"ticker": ticker, "limit": 1000},
            cache_key=f"splits-{ticker}",
        )
        if payload.get("next_url"):
            raise PolygonError(f"more than 1000 splits under {ticker}: paging is not built")
        return tuple(sorted(
            (
                Split(
                    execution_date=dt.date.fromisoformat(str(row["execution_date"])[:10]),
                    split_from=float(row["split_from"]),
                    split_to=float(row["split_to"]),
                    ticker=str(row.get("ticker") or ticker),
                )
                for row in payload.get("results") or []
            ),
            key=lambda s: s.execution_date,
        ))

    def dividends(self, ticker: str) -> tuple["Dividend", ...]:
        """Every cash distribution Polygon records under a ticker, by ex-dividend date --
        regular, special and capital gains alike: `/v3/reference/dividends?ticker=...`.

        The adjusted bars carry none of them. They are adjusted for split events alone, so
        their returns are price returns, and a total return is built from these. A ticker is
        not a security: the events under one ticker can be several securities'.
        """
        payload = self._get(
            "/v3/reference/dividends", {"ticker": ticker, "limit": 1000},
            cache_key=f"dividends-{ticker}",
        )
        if payload.get("next_url"):
            raise PolygonError(f"more than 1000 dividends under {ticker}: paging is not built")

        def day(value) -> dt.date | None:
            return dt.date.fromisoformat(str(value)[:10]) if value else None

        return tuple(sorted(
            (
                Dividend(
                    ex_date=dt.date.fromisoformat(str(row["ex_dividend_date"])[:10]),
                    cash_amount=float(row["cash_amount"]),
                    currency=row.get("currency") or None,
                    kind=row.get("dividend_type") or None,
                    frequency=(int(row["frequency"]) if row.get("frequency") is not None
                               else None),
                    pay_date=day(row.get("pay_date")),
                    ticker=str(row.get("ticker") or ticker),
                )
                for row in payload.get("results") or []
            ),
            key=lambda d: d.ex_date,
        ))

    def cached_reference_dates(self, ticker: str) -> tuple[dt.date, ...]:
        """Dates with a reference record already on disk, newest first.

        A delisted security has no record at the end of a window but plenty before it, and
        those responses are already cached from the panel fetch. Reading them back costs a
        directory listing; guessing at dates instead would put thousands of requests on the
        wire to rediscover what is sitting on disk.
        """
        if self.cache_dir is None:
            return ()
        found: list[dt.date] = []
        prefix = f"ref-{ticker}-"
        for path in self.cache_dir.glob(f"{prefix}*.json"):
            try:
                found.append(dt.date.fromisoformat(path.stem[len(prefix):]))
            except ValueError:
                continue
        return tuple(sorted(found, reverse=True))

    def sic_code_any(self, ticker: str, preferred: dt.date) -> tuple[str | None, dt.date | None]:
        """The SIC code at `preferred`, else the most recent cached date that has one.

        Returns the date it came from, because a classification carried forward from before
        a delisting is a different claim from a current one and the caller should be able
        to say so.
        """
        candidates = [preferred] + [
            d for d in self.cached_reference_dates(ticker) if d != preferred
        ]
        for on in candidates:
            try:
                code = self.sic_code(ticker, on)
            except PolygonError:
                continue
            if code:
                return code, on
        return None, None

    def sic_code(self, ticker: str, on: dt.date) -> str | None:
        """The four-digit SIC code, as a string so leading zeros survive."""
        value = self.ticker_details(ticker, on).get("sic_code")
        return str(value).strip().zfill(4) if value else None

    def shares_outstanding(self, ticker: str, on: dt.date) -> float | None:
        """`share_class_shares_outstanding` as of a date, or None if not reported.

        Deliberately not `weighted_shares_outstanding`, which is the weighted average used
        as an earnings-per-share denominator rather than the count of shares in existence.
        The two converge for most companies and diverge exactly where it matters -- around
        issuance, buybacks and splits.
        """
        value = self.ticker_details(ticker, on).get("share_class_shares_outstanding")
        return float(value) if value else None

    def market_bars(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> tuple[Bar, ...]:
        """Daily bars for whatever stands in for the market: an ETF or a true index.

        A stocks subscription answers 403 for an index symbol like `I:SPX`, so an ETF such
        as SPY is usually what this gets. Its adjusted bars are a price return, split-adjusted
        only, as every ticker's are; its total return needs its distributions (`dividends`).
        Indices carry no volume; ETFs do, and it is ignored either way.
        """
        return self.daily_bars(symbol, start, end, adjusted=True)

    # Retained so existing callers keep working; prefer market_bars.
    index_bars = market_bars


def read_api_key(env_path: Path | str = ".env", variable: str = "POLYGON_API_KEY") -> str:
    """Read the key from a .env file without pulling in a dotenv dependency.

    Named explicitly rather than scanning for anything key-shaped, because this repo's .env
    also holds POLYGON_RPC and POLYGONSCAN_API_KEY, which belong to the Polygon *blockchain*
    and will authenticate against nothing here.
    """
    path = Path(env_path)
    if not path.exists():
        raise PolygonError(f"no {path} to read {variable} from")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{variable}=") and not stripped.startswith("#"):
            value = stripped.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise PolygonError(f"{variable} not found in {path}")
