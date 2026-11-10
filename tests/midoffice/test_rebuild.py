"""Tests for the rebuild's fetch and the gate before the build: each span's identity checked
by its first share sample, each split event classified by its ratio and the share counts
either side."""

from __future__ import annotations

import datetime as dt

from praxis.midoffice.refdata import rebuild
from praxis.midoffice.refdata.polygon import PolygonClient, PolygonNotFoundError, Split, _redact
from praxis.midoffice.refdata.rebuild import Span

KEY = "secretkey0123456789abcdef012345"
START, END = dt.date(2024, 1, 1), dt.date(2024, 1, 31)


def ms(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
               * 1000)


class FakePolygon:
    """Prices, reference records, split events and distributions, by ticker; a 404 for any
    record not given."""

    def __init__(self, records, splits=None, dividends=None):
        self.records = records    # (ticker, iso date) -> (figi, shares)
        self.splits = splits or {}
        self.dividends = dividends or {}
        self.calls: list[str] = []

    def get_json(self, url):
        self.calls.append(url)
        if "/v2/aggs/ticker/" in url:
            return {"status": "OK", "results": [
                {"t": ms(START + dt.timedelta(days=i)), "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                for i in range(3)
            ]}
        if "/v3/reference/splits" in url:
            ticker = url.split("ticker=")[1].split("&")[0]
            return {"status": "OK", "results": self.splits.get(ticker, [])}
        if "/v3/reference/dividends" in url:
            ticker = url.split("ticker=")[1].split("&")[0]
            return {"status": "OK", "results": self.dividends.get(ticker, [])}
        ticker = url.split("/v3/reference/tickers/")[1].split("?")[0]
        on = url.split("date=")[1].split("&")[0]
        if (ticker, on) not in self.records:
            raise PolygonNotFoundError(f"HTTP 404 for {_redact(url)}: Ticker not found.")
        figi, shares = self.records[(ticker, on)]
        return {"status": "OK", "results": {"composite_figi": figi, "cik": "0001", "name": figi,
                                            "type": "CS",
                                            "share_class_shares_outstanding": shares}}


def span(key, ticker, first, last, *samples):
    return Span(key, key, ticker, first, last, 1, tuple(samples))


D = [dt.date(2024, 1, d) for d in range(1, 32)]


def test_the_fetch_asks_once_for_each_answer_and_a_second_run_is_offline(tmp_path):
    """A resumed fetch asks only for what is not on disk: prices adjusted and not, split
    events, distributions and every share sample -- a "not found" kept as an answer -- each
    asked once; and the market series' prices, split events and distributions, as any
    ticker's."""
    paid = [{"ex_dividend_date": "2024-01-10", "cash_amount": 1.5, "currency": "USD",
             "dividend_type": "CD", "frequency": 4}]
    fake = FakePolygon({("AAA", "2024-01-01"): ("F1", 100.0)}, dividends={"SPY": paid})
    client = PolygonClient(api_key=KEY, transport=fake, cache_dir=tmp_path)
    spans = [span("F1", "AAA", D[0], D[9], D[0], D[9])]
    first = rebuild.fetch(client, spans, START, END, market="SPY", cached=client.has_cached)
    assert first.requests == 10 and len(fake.calls) == 10 and not first.failures
    assert first.records[("AAA", D[9])] is None and first.shares[("AAA", D[0])] == 100.0
    assert first.dividends["AAA"] == () and first.dividends["SPY"][0].cash_amount == 1.5
    again = rebuild.fetch(client, spans, START, END, market="SPY", cached=client.has_cached)
    assert again.requests == 0 and len(fake.calls) == 10
    assert again.records == first.records and again.dividends == first.dividends
    assert again.bars == {(t, basis): 3 for t in ("AAA", "SPY") for basis in ("adj", "raw")}


def test_a_span_whose_first_sample_names_another_security_or_none_stops_the_gate(tmp_path):
    """CRHC's span is CRH's only if CRHC's record on its first day names CRH. Another security
    there, or no record at all, fails the span -- and one failure stops the gate."""
    fake = FakePolygon({("AAA", "2024-01-01"): ("F1", 1.0), ("BBB", "2024-01-01"): ("OTHER", 1.0)})
    client = PolygonClient(api_key=KEY, transport=fake, cache_dir=tmp_path)
    spans = [span("F1", "AAA", D[0], D[5], D[0]), span("F2", "BBB", D[0], D[5], D[0]),
             span("F3", "CCC", D[0], D[5], D[0])]
    checks = rebuild.check_identities(spans, rebuild.fetch(client, spans, START, END))
    assert [c.passed for c in checks] == [True, False, False]
    assert "names OTHER" in checks[1].detail and "no reference record" in checks[2].detail
    verdict = rebuild.gate(checks, [])
    assert not verdict.passed and len(verdict.identity_failures) == 2


def test_a_split_is_a_whole_ratio_the_counts_follow_and_a_spin_off_a_fractional_one_they_leave(
    tmp_path,
):
    """A ten-for-one whose counts go from 100 to 1,003 is a split, applied to the carried count;
    a 1,000-to-1,281 -- GE's for GE HealthCare, whole numbers in form and 1.281 in fact -- whose
    counts stay at 100 is price-only, not applied. The two tests disagreeing stops the gate: a
    ten-for-one the counts do not follow, or a 1.281 they do -- DuPont's reported, not stopped
    on."""
    records = {
        ("AAA", "2024-01-01"): ("F1", 100.0), ("AAA", "2024-01-20"): ("F1", 1003.0),
        ("BBB", "2024-01-01"): ("F2", 100.0), ("BBB", "2024-01-20"): ("F2", 101.0),
        ("CCC", "2024-01-01"): ("F3", 100.0), ("CCC", "2024-01-20"): ("F3", 101.0),
        ("DDD", "2024-01-01"): ("F4", 100.0), ("DDD", "2024-01-20"): ("F4", 127.0),
        ("DD", "2024-01-01"): ("F5", 100.0), ("DD", "2024-01-20"): ("F5", 99.0),
    }
    ten = [{"execution_date": "2024-01-10", "split_from": 1, "split_to": 10}]
    spin = [{"execution_date": "2024-01-10", "split_from": 1000, "split_to": 1281}]
    fake = FakePolygon(records, {"AAA": ten, "BBB": spin, "CCC": ten, "DDD": spin, "DD": ten})
    client = PolygonClient(api_key=KEY, transport=fake, cache_dir=tmp_path)
    spans = [span(f"F{i}", t, D[0], D[19], D[0], D[19])
             for i, t in enumerate(("AAA", "BBB", "CCC", "DDD", "DD"), 1)]
    checks = rebuild.check_splits(spans, rebuild.fetch(client, spans, START, END))
    by = {c.split.ticker: c for c in checks}
    assert (by["AAA"].kind, by["AAA"].counts, by["AAA"].applies) == ("split", "follow", True)
    assert (by["BBB"].kind, by["BBB"].counts, by["BBB"].applies) == (
        "price-only", "unchanged", False)
    assert [by[t].kind for t in ("CCC", "DDD", "DD")] == ["contradiction"] * 3
    assert not any(by[t].applies for t in ("CCC", "DDD", "DD"))
    assert "whole-number ratio and the counts stay unchanged" in by["CCC"].detail
    assert "fractional ratio and the counts follow it" in by["DDD"].detail
    verdict = rebuild.gate([], checks)
    assert not verdict.passed and by["DD"].exempt
    assert [c.split.ticker for c in verdict.contradictions] == ["CCC", "DDD"]
    assert rebuild.gate([], [by["AAA"], by["BBB"], by["DD"]]).passed


def test_counts_that_move_by_neither_the_ratio_nor_nothing_confirm_neither_kind():
    """A ten-for-one the counts take from 100 to 300 is no split, and a 1.281 they take to 150
    no price-only adjustment: both are contradictions. How near is near enough is the
    tolerance -- a two-for-one taking 100 to 185 follows it within 10%, and not within 5%."""
    before = (D[0], 100.0)
    ten, spin, two = (Split(D[9], 1, 10, "A"), Split(D[9], 1000, 1281, "A"),
                      Split(D[9], 1, 2, "A"))
    assert rebuild.classify(ten, before, (D[19], 300.0))[1:3] == ("neither", "contradiction")
    assert rebuild.classify(spin, before, (D[19], 150.0))[1:3] == ("neither", "contradiction")
    assert rebuild.classify(two, before, (D[19], 185.0))[1:3] == ("follow", "split")
    assert rebuild.classify(two, before, (D[19], 185.0), tolerance=0.05)[1:3] == (
        "neither", "contradiction")


def test_where_the_counts_cannot_test_an_event_its_ratio_alone_classifies_it():
    """No share sample after a ten-for-one leaves it a split by its ratio, and applied. A
    1,000-to-1,033 with counts from 100 to 101 is within 10% of both the ratio and no change,
    so the counts cannot tell; it is price-only by its ratio. Both are reported as untested,
    and neither stops the gate."""
    ten, small = Split(D[9], 1, 10, "AAA"), Split(D[9], 1000, 1033, "LEN")
    fetched = rebuild.Fetched(
        splits={"AAA": (ten,), "LEN": (small,)},
        shares={("AAA", D[0]): 100.0, ("LEN", D[0]): 100.0, ("LEN", D[19]): 101.0},
    )
    spans = [span("F1", "AAA", D[0], D[19], D[0]), span("F2", "LEN", D[0], D[19], D[0], D[19])]
    aaa, len_ = rebuild.check_splits(spans, fetched)
    assert (aaa.counts, aaa.kind, aaa.applies) == (None, "split", True)
    assert "no share sample after" in aaa.detail
    assert (len_.counts, len_.kind, len_.applies) == ("either", "price-only", False)
    assert "untested" in len_.detail and not aaa.tested and not len_.tested
    verdict = rebuild.gate([], [aaa, len_])
    assert verdict.passed and verdict.untested == (aaa, len_)


def test_a_whole_number_ratio_is_small_in_lowest_terms_however_it_is_written():
    """Polygon writes GE's spin-off adjustment as 1,000 to 1,281 and DuPont's as 100 to 239:
    whole numbers, but in lowest terms no split's. 2,000 to 1,000 is a one-for-two."""
    def whole(f, t, **kw):
        return rebuild.is_whole(Split(D[0], f, t, "X"), **kw)

    assert whole(2, 3) and whole(1, 50) and whole(5, 1) and whole(2000, 1000)
    assert not whole(1000, 1281) and not whole(100, 239)
    assert not whole(1000, 1164)    # 250 to 291 in lowest terms
    assert not whole(1, 101) and whole(1, 101, largest_term=101)


def test_a_split_outside_the_securitys_span_under_the_ticker_is_another_securitys(tmp_path):
    """Split events are by ticker, and a ticker is not a security: a split dated where the
    ticker belongs to someone else is not checked against this security's counts."""
    records = {("AAA", "2024-01-01"): ("F1", 100.0), ("AAA", "2024-01-05"): ("F1", 100.0)}
    later = [{"execution_date": "2024-01-25", "split_from": 1, "split_to": 4}]
    fake = FakePolygon(records, {"AAA": later})
    client = PolygonClient(api_key=KEY, transport=fake, cache_dir=tmp_path)
    spans = [span("F1", "AAA", D[0], D[4], D[0], D[4])]
    assert rebuild.check_splits(spans, rebuild.fetch(client, spans, START, END)) == []


def test_the_plan_reads_back_as_written(tmp_path):
    path = tmp_path / "plan.csv"
    path.write_text(
        "key,name,ticker,first,last,held,member_days,samples,new_samples,sample_dates\n"
        "F1,Aaa,AAA,2024-01-01,2024-01-31,20,15,2,1,2024-01-01 2024-01-31\n",
        encoding="utf-8",
    )
    (plan,) = rebuild.read_plan(path)
    assert plan == Span("F1", "Aaa", "AAA", D[0], D[30], 15, (D[0], D[30]))
