"""The Polygon client and the panel it assembles, tested without touching a network.

The transport is a Protocol precisely so these can run offline, and the fake here serves
canned payloads shaped like Polygon's. Three concerns dominate:

* the API key must not escape into an exception, a log line or a cache path;
* a ticker that fails must be *recorded*, because the code this replaces swallowed failures
  and produced panels with silently absent securities;
* adjusted and unadjusted prices must stay separate all the way through, since mixing them
  is the error Polygon's own market_cap field makes at a split.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error

import numpy as np
import pytest

from praxis.midoffice.refdata.polygon import (
    Bar,
    HttpTransport,
    PolygonClient,
    PolygonError,
    PolygonNotFoundError,
    _redact,
    read_api_key,
)
from praxis.midoffice.model.panel import Frequency
from praxis.midoffice.refdata.prices import (
    LoadReport,
    PanelRequest,
    build_security_panel,
    estimate_requests,
)

KEY = "secretkey0123456789abcdef012345"


def _ms(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


class FakeTransport:
    """Serves canned payloads and records every URL it was asked for."""

    def __init__(self, days=5, tickers=("AAA", "BBB"), shares=1_000_000.0, fail=()):
        self.calls: list[str] = []
        self.days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(days))
        self.tickers = tickers
        self.shares = shares
        self.fail = set(fail)

    def get_json(self, url: str):
        self.calls.append(url)
        if "/v3/reference/tickers/" in url:
            ticker = url.split("/v3/reference/tickers/")[1].split("?")[0]
            if ticker in self.fail:
                return {"status": "NOT_FOUND", "error": "unknown ticker"}
            return {
                "status": "OK",
                "results": {
                    "ticker": ticker,
                    "share_class_shares_outstanding": self.shares,
                    "weighted_shares_outstanding": self.shares * 4.0,
                    "market_cap": 1.0,
                },
            }
        ticker = url.split("/ticker/")[1].split("/range")[0]
        if ticker in self.fail:
            return {"status": "NOT_AUTHORIZED"}
        adjusted = "adjusted=true" in url
        scale = 1.0 if adjusted else 4.0
        return {
            "status": "OK",
            "results": [
                {
                    "t": _ms(day),
                    "o": 100.0 * scale,
                    "h": 105.0 * scale,
                    "l": 95.0 * scale,
                    "c": (100.0 + i) * scale,
                    "v": 10_000.0,
                }
                for i, day in enumerate(self.days)
            ],
        }


def _client(**kw) -> PolygonClient:
    return PolygonClient(api_key=KEY, transport=FakeTransport(**kw))


# ------------------------------------------------------------------ the key never escapes


def test_the_api_key_is_redacted_from_anything_that_escapes():
    url = f"https://api.polygon.io/v2/x?adjusted=true&apiKey={KEY}&sort=asc"
    out = _redact(url)
    assert KEY not in out
    assert "apiKey=<redacted>" in out
    assert "adjusted=true" in out and "sort=asc" in out


def test_a_failing_request_does_not_leak_the_key_into_the_exception():
    class Boom:
        def get_json(self, url):
            return {"status": "ERROR", "error": "nope"}

    client = PolygonClient(api_key=KEY, transport=Boom())
    with pytest.raises(PolygonError) as caught:
        client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert KEY not in str(caught.value)
    assert "<redacted>" in str(caught.value)


def test_the_cache_path_does_not_contain_the_key(tmp_path):
    client = PolygonClient(api_key=KEY, transport=FakeTransport(), cache_dir=tmp_path)
    client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    for path in tmp_path.iterdir():
        assert KEY not in path.name


def test_an_empty_key_is_refused_with_the_right_variable_named():
    with pytest.raises(PolygonError, match="POLYGONSCAN_API_KEY or"):
        PolygonClient(api_key="   ")


# ------------------------------------------------------------------------ status handling


def test_not_authorized_explains_that_it_is_a_subscription_limit():
    """The distinction that matters: a short panel is a plan boundary, not a quiet market."""
    client = _client(fail=("AAA",))
    with pytest.raises(PolygonError, match="subscription limit"):
        client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5))


def test_an_unrecognised_status_raises_rather_than_returning_nothing():
    client = _client(fail=("AAA",))
    with pytest.raises(PolygonError, match="NOT_FOUND"):
        client.shares_outstanding("AAA", dt.date(2024, 1, 3))


def test_a_client_error_is_not_retried_but_a_rate_limit_is():
    """429 and 5xx are worth another go; a 401 or 404 will never improve."""
    import urllib.request

    attempts = {"n": 0}

    def failing_with(code):
        def fake_urlopen(url, timeout=None):
            attempts["n"] += 1
            raise urllib.error.HTTPError(url, code, "boom", {}, None)

        return fake_urlopen

    original = urllib.request.urlopen
    try:
        urllib.request.urlopen = failing_with(401)
        attempts["n"] = 0
        with pytest.raises(PolygonError, match="HTTP 401"):
            HttpTransport(retries=3, backoff=0.0).get_json("https://x/y?apiKey=k")
        assert attempts["n"] == 1

        urllib.request.urlopen = failing_with(429)
        attempts["n"] = 0
        with pytest.raises(PolygonError, match="3 attempt"):
            HttpTransport(retries=3, backoff=0.0).get_json("https://x/y?apiKey=k")
        assert attempts["n"] == 3
    finally:
        urllib.request.urlopen = original


def test_the_retry_message_does_not_leak_the_key_either():
    import urllib.request

    original = urllib.request.urlopen
    try:
        def always_timeout(url, timeout=None):
            raise TimeoutError("slow")

        urllib.request.urlopen = always_timeout
        with pytest.raises(PolygonError) as caught:
            HttpTransport(retries=2, backoff=0.0).get_json(
                f"https://x/y?apiKey={KEY}"
            )
        assert KEY not in str(caught.value)
    finally:
        urllib.request.urlopen = original


# ------------------------------------------------------------------------- parsing bars


def test_bars_parse_and_carry_which_basis_they_are_on():
    client = _client()
    adjusted = client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    raw = client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5), adjusted=False)
    assert all(bar.adjusted for bar in adjusted)
    assert not any(bar.adjusted for bar in raw)
    # The fake prices the unadjusted series four times higher, as a split would.
    assert raw[0].close == pytest.approx(adjusted[0].close * 4.0)


def test_bars_come_back_in_date_order_even_if_the_payload_is_not():
    class Shuffled(FakeTransport):
        def get_json(self, url):
            payload = super().get_json(url)
            if "results" in payload and isinstance(payload["results"], list):
                payload["results"] = list(reversed(payload["results"]))
            return payload

    client = PolygonClient(api_key=KEY, transport=Shuffled())
    bars = client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert list(bars) == sorted(bars, key=lambda b: b.date)


def test_the_adjusted_flag_reaches_the_url():
    transport = FakeTransport()
    client = PolygonClient(api_key=KEY, transport=transport)
    client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5), adjusted=False)
    assert "adjusted=false" in transport.calls[-1]


def test_shares_outstanding_reads_share_class_not_weighted():
    """weighted_shares_outstanding is an earnings-per-share denominator, and the two
    diverge exactly where it matters -- issuance, buybacks and splits."""
    client = _client(shares=1_234_000.0)
    value = client.shares_outstanding("AAA", dt.date(2024, 1, 3))
    assert value == pytest.approx(1_234_000.0)  # not 4x that


def test_absent_shares_outstanding_comes_back_as_none():
    class NoShares(FakeTransport):
        def get_json(self, url):
            payload = super().get_json(url)
            if "/v3/reference/" in url:
                payload["results"].pop("share_class_shares_outstanding")
            return payload

    client = PolygonClient(api_key=KEY, transport=NoShares())
    assert client.shares_outstanding("AAA", dt.date(2024, 1, 3)) is None


def test_the_cache_stops_a_second_identical_request(tmp_path):
    transport = FakeTransport()
    client = PolygonClient(api_key=KEY, transport=transport, cache_dir=tmp_path)
    client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    first = len(transport.calls)
    client.daily_bars("AAA", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert len(transport.calls) == first


# ------------------------------------------------------------------------ reading the key


def test_the_key_is_read_by_name_not_by_shape(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nPOLYGONSCAN_API_KEY=blockchain_one\n"
        f"POLYGON_API_KEY={KEY}\nPOLYGON_RPC=https://rpc\n",
        encoding="utf-8",
    )
    assert read_api_key(env) == KEY


def test_a_quoted_value_and_a_missing_variable_are_both_handled(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f'POLYGON_API_KEY="{KEY}"\n', encoding="utf-8")
    assert read_api_key(env) == KEY
    with pytest.raises(PolygonError, match="not found"):
        read_api_key(env, variable="NOPE_API_KEY")
    with pytest.raises(PolygonError, match="no .* to read"):
        read_api_key(tmp_path / "absent.env")


# --------------------------------------------------------------------- the panel builder


def _request(**kw) -> PanelRequest:
    kw.setdefault("start", dt.date(2024, 1, 1))
    kw.setdefault("end", dt.date(2024, 1, 5))
    kw.setdefault("tickers", ("AAA", "BBB"))
    kw.setdefault("shares_every", 2)
    return PanelRequest(**kw)


def test_the_request_validates_and_de_duplicates():
    assert _request(tickers=("AAA", "BBB", "AAA")).tickers == ("AAA", "BBB")
    with pytest.raises(ValueError, match="no tickers"):
        _request(tickers=())
    with pytest.raises(ValueError, match="precedes start"):
        PanelRequest(dt.date(2024, 6, 1), dt.date(2024, 1, 1), ("AAA",))
    with pytest.raises(ValueError, match="at least one trading day"):
        _request(shares_every=0)


def test_the_request_count_is_knowable_before_anything_is_spent():
    """Five hundred names over five years is ten thousand calls. Worth seeing first."""
    big = PanelRequest(
        dt.date(2022, 1, 1), dt.date(2026, 11, 1), tuple(f"T{i}" for i in range(500))
    )
    assert 10_000 < estimate_requests(big) < 12_000
    cheaper = PanelRequest(
        dt.date(2022, 1, 1),
        dt.date(2026, 11, 1),
        tuple(f"T{i}" for i in range(500)),
        shares_every=252,
    )
    assert estimate_requests(cheaper) < estimate_requests(big) / 2


def test_the_panel_carries_both_price_bases():
    panel, market, report = build_security_panel(_client(), _request())
    assert panel.has_unadjusted
    assert panel.frequency is Frequency.DAILY
    assert np.allclose(panel.unadjusted_prices, panel.prices * 4.0)
    assert report.unadjusted


def test_declining_the_unadjusted_fetch_is_reported_not_silent():
    panel, _, report = build_security_panel(
        _client(), _request(fetch_unadjusted=False)
    )
    assert not panel.has_unadjusted
    assert any("split-adjusted basis" in q for q in report.qualifications())


def test_the_index_supplies_the_trading_calendar():
    """Not the union of the securities' days: one illiquid name would otherwise introduce
    a date the market never traded, and every other security would show a gap there."""
    client = _client(days=7)
    panel, market, report = build_security_panel(client, _request())
    assert len(panel.dates) == 7
    assert report.trading_days == 7
    assert np.isnan(market[0])  # no return on the first day
    assert np.isfinite(market[1:]).all()


def test_a_failing_ticker_is_recorded_rather_than_quietly_dropped():
    """The defect in the code this replaces: `except Exception: pass` turns a rate limit
    into a security silently missing from every cross-section."""
    client = _client(fail=("BBB",))
    panel, _, report = build_security_panel(
        client, _request(tickers=("AAA", "BBB", "CCC"))
    )
    assert "BBB" not in panel.securities
    assert "BBB" in report.failed
    assert "NOT_AUTHORIZED" in report.failed["BBB"]
    assert any("could not be loaded" in q for q in report.qualifications())


def test_every_ticker_failing_raises_rather_than_returning_an_empty_panel():
    client = _client(fail=("AAA", "BBB"))
    with pytest.raises(PolygonError, match="every one of"):
        build_security_panel(client, _request())


def test_shares_are_carried_forward_but_never_backward():
    """Carrying forward is true -- the count did not change between filings. Carrying
    backward would be invention."""
    class LateShares(FakeTransport):
        def get_json(self, url):
            payload = super().get_json(url)
            if "/v3/reference/" in url and "date=2024-01-01" in url:
                payload["results"]["share_class_shares_outstanding"] = None
            return payload

    client = PolygonClient(api_key=KEY, transport=LateShares(days=5))
    panel, _, _ = build_security_panel(client, _request(shares_every=2))
    column = panel.shares_outstanding[:, 0]
    assert np.isnan(column[0])             # before the first sample: left missing
    assert np.isfinite(column[2:]).all()   # from the first sample on: carried forward


def test_a_security_with_no_shares_at_all_is_named():
    class NoShares(FakeTransport):
        def get_json(self, url):
            payload = super().get_json(url)
            if "/v3/reference/" in url:
                payload["results"]["share_class_shares_outstanding"] = None
            return payload

    client = PolygonClient(api_key=KEY, transport=NoShares())
    _, _, report = build_security_panel(client, _request())
    assert set(report.no_shares) == {"AAA", "BBB"}
    assert any("no Size or Trading Activity" in q for q in report.qualifications())


def test_a_security_with_half_a_history_is_flagged():
    class Short(FakeTransport):
        def get_json(self, url):
            payload = super().get_json(url)
            if "/ticker/BBB/" in url and isinstance(payload.get("results"), list):
                payload["results"] = payload["results"][:1]
            return payload

    client = PolygonClient(api_key=KEY, transport=Short(days=8))
    _, _, report = build_security_panel(client, _request())
    assert "BBB" in report.partial_history
    assert any("listing or delisting" in q for q in report.qualifications())


def test_a_sparse_shares_grid_is_called_out():
    report = LoadReport(trading_days=1260, shares_sample_dates=tuple([dt.date(2024, 1, 1)] * 5))
    assert any("stepped over" in q for q in report.qualifications())


def test_the_report_describes_what_it_did():
    _, _, report = build_security_panel(_client(), _request())
    text = report.describe()
    assert "2 of 2 securities" in text
    assert "request(s)" in text
    assert report.requests_made > len(report.loaded)


def test_the_assembled_panel_feeds_the_descriptors():
    """End to end from a fetch to exposures, with no network anywhere in it."""
    from praxis.midoffice.model.descriptors import compute_descriptors

    client = _client(days=40, tickers=("AAA", "BBB"))
    panel, market, _ = build_security_panel(
        client, _request(end=dt.date(2024, 2, 9), shares_every=10)
    )
    out, skipped = compute_descriptors(
        panel, market_returns=market, names=("LNCAP", "LPRI", "STOM")
    )
    assert set(out.names) == {"LNCAP", "LPRI", "STOM"}
    assert "LNCAP" not in skipped and "LPRI" not in skipped
    assert np.isfinite(out.values[-1, :, out.names.index("LNCAP")]).all()


# ------------------------------------------------- the market series: SPY, not the index


def test_the_default_market_series_is_the_etf_not_the_index():
    """Two reasons, and the second outlives the first.

    A stocks subscription answers 403 for `I:SPX`. But even with an indices subscription SPY
    is the right series here: these securities' returns come from dividend-adjusted prices,
    so they are total returns, while `I:SPX` is a price index. Pairing total-return
    securities with a price-return market leaves roughly 1.5-2% a year unexplained and
    biases every beta estimated against it.
    """
    from praxis.midoffice.refdata.prices import MARKET_PROXY, SP500_INDEX

    assert MARKET_PROXY == "SPY"
    assert SP500_INDEX == "I:SPX"
    assert _request().index == MARKET_PROXY


def test_the_market_series_can_be_overridden_for_an_indices_subscription():
    assert _request(index="I:SPX").index == "I:SPX"


def test_market_bars_is_the_name_and_index_bars_still_works():
    client = _client()
    by_new = client.market_bars("SPY", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    by_old = client.index_bars("SPY", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert by_new == by_old
    assert all(bar.adjusted for bar in by_new)


def test_a_market_series_that_cannot_be_fetched_says_so_rather_than_failing_obscurely():
    client = _client(fail=("SPY",))
    with pytest.raises(PolygonError, match="NOT_AUTHORIZED"):
        build_security_panel(client, _request())


def test_an_empty_market_series_names_the_calendar_as_the_casualty():
    class NoIndex(FakeTransport):
        def get_json(self, url):
            payload = super().get_json(url)
            if "/ticker/SPY/" in url and isinstance(payload.get("results"), list):
                payload["results"] = []
            return payload

    client = PolygonClient(api_key=KEY, transport=NoIndex())
    with pytest.raises(PolygonError, match="no trading calendar"):
        build_security_panel(client, _request())


# ------------------------------------- a delisting must not become survivorship bias


class DelistedMidWindow(FakeTransport):
    """Bars exist for the whole window; the reference record has gone.

    What an acquisition looks like through this API: the company's price history remains,
    but `/v3/reference/tickers/{t}?date=` has nothing to say about a date after the ticker
    stopped existing.
    """

    def __init__(self, gone="BBB", from_date=dt.date(2024, 1, 3), **kw):
        super().__init__(**kw)
        self.gone = gone
        self.from_date = from_date

    def get_json(self, url: str):
        if "/v3/reference/tickers/" in url and f"/{self.gone}?" in url:
            on = dt.date.fromisoformat(url.split("date=")[1].split("&")[0])
            if on >= self.from_date:
                self.calls.append(url)
                return {"status": "NOT_FOUND", "error": "ticker not found on that date"}
        return super().get_json(url)


def test_a_delisted_security_keeps_its_price_history():
    """The whole point of point-in-time membership is to retain companies that left the
    index -- and they left precisely because something happened to them. Letting a reference
    lookup fail the security would reintroduce survivorship bias one layer down, where the
    snapshot module cannot see it.
    """
    client = PolygonClient(api_key=KEY, transport=DelistedMidWindow(days=6))
    panel, _, report = build_security_panel(client, _request(shares_every=2))

    assert "BBB" in panel.securities, "a delisting discarded the price history"
    assert "BBB" not in report.failed
    column = panel.prices[:, panel.securities.index("BBB")]
    assert np.isfinite(column).all()


def test_the_failed_reference_dates_are_reported_rather_than_absorbed():
    client = PolygonClient(api_key=KEY, transport=DelistedMidWindow(days=6))
    _, _, report = build_security_panel(client, _request(shares_every=2))
    assert report.reference_gaps.get("BBB", 0) >= 1
    assert any("kept their price history anyway" in q for q in report.qualifications())


def test_shares_before_the_delisting_still_carry_forward():
    """The samples that did come back are still good, and the count did not change just
    because the ticker later stopped existing."""
    client = PolygonClient(api_key=KEY, transport=DelistedMidWindow(days=6))
    panel, _, _ = build_security_panel(client, _request(shares_every=2))
    column = panel.shares_outstanding[:, panel.securities.index("BBB")]
    assert np.isfinite(column).any()


def test_a_failure_reason_reaches_the_report_not_just_the_name():
    """A delisting is benign, an auth failure is a plan boundary, a parse failure is a
    defect. The ticker alone does not say which."""
    client = _client(fail=("BBB",))
    _, _, report = build_security_panel(client, _request(tickers=("AAA", "BBB")))
    note = next(q for q in report.qualifications() if "could not be loaded" in q)
    assert "BBB" in note
    assert "NOT_AUTHORIZED" in note


def test_the_estimate_matches_what_the_builder_actually_spends():
    """A cost estimate that undercounts is worse than none. This one said 3,543 for a
    506-name run that spent 4,049, because it computed ceil(n/every) while the builder
    also appends the final day as an extra sample."""
    from praxis.midoffice.refdata.prices import _sample_slots

    transport = FakeTransport(days=9, tickers=("AAA", "BBB"))
    client = PolygonClient(api_key=KEY, transport=transport)
    request = _request(tickers=("AAA", "BBB"), shares_every=4)
    _, _, report = build_security_panel(client, request)
    assert estimate_requests(request, trading_days=9) == report.requests_made


def test_the_sample_grid_always_includes_the_final_day():
    from praxis.midoffice.refdata.prices import _sample_slots

    assert _sample_slots(9, 4) == [0, 4, 8]
    assert _sample_slots(10, 4) == [0, 4, 8, 9]   # stride misses the end; it is appended
    assert _sample_slots(1214, 252) == [0, 252, 504, 756, 1008, 1213]
    assert len(_sample_slots(1214, 252)) == 6     # not the 5 that ceil(1214/252) gives


# ------------------------------------------------- ticker events: who traded under what, when


class EventsTransport:
    """The ticker events endpoint: a payload for a listed id, a 404 for any other, as the
    real HttpTransport raises it."""

    def __init__(self, known):
        self.known = known
        self.calls: list[str] = []

    def get_json(self, url: str):
        self.calls.append(url)
        identifier = url.split("/vX/reference/tickers/")[1].split("/events")[0]
        if identifier not in self.known:
            raise PolygonNotFoundError(f"HTTP 404 for {_redact(url)}: No events found for given ID")
        return {"status": "OK", "results": self.known[identifier]}


BNY_EVENTS = {
    "name": "Bank of New York Mellon Corporation",
    "composite_figi": "BBG000BD8PN9",
    "cik": "0001390777",
    "events": [
        {"ticker_change": {"ticker": "BNY"}, "type": "ticker_change", "date": "2026-05-21"},
        {"ticker_change": {"ticker": "BK"}, "type": "ticker_change", "date": "2007-07-02"},
    ],
}


def test_ticker_events_come_back_parsed_and_are_asked_for_once(tmp_path):
    transport = EventsTransport({"BNY": BNY_EVENTS})
    client = PolygonClient(api_key=KEY, transport=transport, cache_dir=tmp_path)
    found = client.ticker_events("BNY")
    assert found["composite_figi"] == "BBG000BD8PN9"
    assert [e["ticker_change"]["ticker"] for e in found["events"]] == ["BNY", "BK"]
    assert transport.calls[0].split("?")[0].endswith("/vX/reference/tickers/BNY/events")
    assert client.ticker_events("BNY") == found
    assert len(transport.calls) == 1


def test_no_events_for_a_delisted_security_is_an_answer_and_is_kept(tmp_path):
    """The endpoint covers listed securities only and answers 404 for any other. That is
    the answer for a delisted name, not a failure, so it comes back as None -- and is cached,
    so a rebuild asks once and every later run is offline -- without the key in the file."""
    transport = EventsTransport({})
    client = PolygonClient(api_key=KEY, transport=transport, cache_dir=tmp_path)
    assert client.ticker_events("CDAY") is None
    assert client.ticker_events("CDAY") is None
    assert len(transport.calls) == 1
    kept = (tmp_path / "events-CDAY.json").read_text(encoding="utf-8")
    assert json.loads(kept)["status"] == "NOT_FOUND"
    assert KEY not in kept


def test_a_404_is_not_found_with_the_servers_reason_and_is_not_retried():
    """A 404 is PolygonNotFoundError, still a PolygonError for every caller that catches those,
    and carries the server's message -- "No events found for given ID" says what "HTTP 404"
    does not -- with the key redacted. It will not improve, so it is asked once."""
    import io
    import urllib.request

    attempts = {"n": 0}
    body = b'{"status": "NOT_FOUND", "message": "No events found for given ID"}'

    def not_found(url, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(body))

    original = urllib.request.urlopen
    try:
        urllib.request.urlopen = not_found
        with pytest.raises(PolygonNotFoundError) as caught:
            HttpTransport(retries=3, backoff=0.0).get_json(f"https://x/y?apiKey={KEY}")
    finally:
        urllib.request.urlopen = original
    assert attempts["n"] == 1
    assert isinstance(caught.value, PolygonError)
    assert "No events found for given ID" in str(caught.value)
    assert KEY not in str(caught.value)
