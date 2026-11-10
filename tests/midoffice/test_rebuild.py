"""Tests for the rebuild's fetch and the gate before the build: each span's identity checked
by its first share sample with an identifier, each split event classified by its ratio, the
share counts either side -- stale ones read past -- and its raw price on the day."""

from __future__ import annotations

import datetime as dt

from praxis.midoffice.refdata import rebuild
from praxis.midoffice.refdata.identity import Record
from praxis.midoffice.refdata.polygon import PolygonClient, PolygonNotFoundError, Split, _redact
from praxis.midoffice.refdata.rebuild import Span

KEY = "secretkey0123456789abcdef012345"
START, END = dt.date(2024, 1, 1), dt.date(2024, 1, 31)


def ms(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
               * 1000)


class FakePolygon:
    """Prices, reference records, split events and distributions, by ticker; a 404 for any
    record not given. Three days of prices at 1, unless `bars` gives a ticker its own, as
    (date, open, close)."""

    def __init__(self, records, splits=None, dividends=None, bars=None):
        self.records = records    # (ticker, iso date) -> (figi, shares)
        self.splits = splits or {}
        self.dividends = dividends or {}
        self.bars = bars or {}
        self.calls: list[str] = []

    def get_json(self, url):
        self.calls.append(url)
        if "/v2/aggs/ticker/" in url:
            ticker = url.split("/v2/aggs/ticker/")[1].split("/")[0]
            given = self.bars.get(
                ticker, [(START + dt.timedelta(days=i), 1, 1) for i in range(3)])
            return {"status": "OK", "results": [
                {"t": ms(on), "o": o, "h": max(o, c), "l": min(o, c), "c": c, "v": 1}
                for on, o, c in given
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
    """A ten-for-one whose counts go from 100 to 1,003 and whose price opens at a tenth is a
    split, applied to the carried count; a 1,000-to-1,281 -- GE's for GE HealthCare, whole
    numbers in form and 1.281 in fact -- whose counts stay at 100 is price-only, not applied. A
    ten-for-one whose counts stay at 100 to the span's end rests on its price: it gaps, so it is
    a split. A 1.281 the counts follow, and a ten-for-one the counts follow and the price does
    not, stop the gate -- DuPont's reported, not stopped on."""
    records = {
        ("AAA", "2024-01-01"): ("F1", 100.0), ("AAA", "2024-01-20"): ("F1", 1003.0),
        ("BBB", "2024-01-01"): ("F2", 100.0), ("BBB", "2024-01-20"): ("F2", 101.0),
        ("CCC", "2024-01-01"): ("F3", 100.0), ("CCC", "2024-01-20"): ("F3", 101.0),
        ("DDD", "2024-01-01"): ("F4", 100.0), ("DDD", "2024-01-20"): ("F4", 127.0),
        ("EEE", "2024-01-01"): ("F6", 100.0), ("EEE", "2024-01-20"): ("F6", 1000.0),
        ("DD", "2024-01-01"): ("F5", 100.0), ("DD", "2024-01-20"): ("F5", 99.0),
    }
    ten = [{"execution_date": "2024-01-10", "split_from": 1, "split_to": 10}]
    spin = [{"execution_date": "2024-01-10", "split_from": 1000, "split_to": 1281}]
    gapped = [(D[8], 50.0, 50.0), (D[9], 5.0, 5.1)]
    flat = [(D[8], 50.0, 50.0), (D[9], 49.0, 49.5)]
    bars = {"AAA": gapped, "BBB": [(D[8], 50.0, 50.0), (D[9], 39.0, 39.0)], "CCC": gapped,
            "DDD": gapped, "EEE": flat, "DD": flat}
    fake = FakePolygon(records, {"AAA": ten, "BBB": spin, "CCC": ten, "DDD": spin, "EEE": ten,
                                 "DD": ten}, bars=bars)
    client = PolygonClient(api_key=KEY, transport=fake, cache_dir=tmp_path)
    tickers = ("AAA", "BBB", "CCC", "DDD", "DD", "EEE")
    spans = [span(f"F{i}", t, D[0], D[19], D[0], D[19]) for i, t in enumerate(tickers, 1)]
    fetched = rebuild.fetch(client, spans, START, END)

    def gaps(ticker, on):
        return rebuild.raw_gap(client.daily_bars(ticker, START, END, adjusted=False), on)

    checks = rebuild.check_splits(spans, fetched, gaps=gaps)
    by = {c.split.ticker: c for c in checks}
    assert (by["AAA"].kind, by["AAA"].counts, by["AAA"].prices, by["AAA"].applies) == (
        "split", "follow", "gap", True)
    assert (by["BBB"].kind, by["BBB"].counts, by["BBB"].applies) == (
        "price-only", "unchanged", False)
    assert (by["CCC"].kind, by["CCC"].counts, by["CCC"].prices) == ("split", "stale", "gap")
    assert by["CCC"].tested and by["CCC"].stale == ((D[19], 101.0),)
    assert "the price gaps by it; the counts stale to the span's end" in by["CCC"].detail
    assert [by[t].kind for t in ("DDD", "EEE", "DD")] == ["contradiction"] * 3
    assert "fractional ratio and the counts follow it" in by["DDD"].detail
    assert "raw price does not gap by it within 10%" in by["EEE"].detail
    verdict = rebuild.gate([], checks)
    assert not verdict.passed and by["DD"].exempt
    assert [c.split.ticker for c in verdict.contradictions] == ["DDD", "EEE"]
    assert rebuild.gate([], [by["AAA"], by["BBB"], by["CCC"], by["DD"]]).passed


def test_counts_that_move_by_neither_the_ratio_nor_nothing_confirm_neither_kind():
    """A ten-for-one the counts take from 100 to 300 is no split, and a 1.281 they take to 150
    no price-only adjustment: both are contradictions. How near is near enough is the
    tolerance -- a two-for-one taking 100 to 185 follows it within 10%, and not within 5%."""
    before = (D[0], 100.0)
    ten, spin, two = (Split(D[9], 1, 10, "A"), Split(D[9], 1000, 1281, "A"),
                      Split(D[9], 1, 2, "A"))

    def kind(split, *later, **kw):
        found = rebuild.classify(split, before, [(D[19], n) for n in later], **kw)
        return found["counts"], found["kind"]

    assert kind(ten, 300.0) == ("neither", "contradiction")
    assert kind(spin, 150.0) == ("neither", "contradiction")
    assert kind(two, 185.0) == ("follow", "split")
    assert kind(two, 185.0, tolerance=0.05) == ("neither", "contradiction")


def test_a_sample_still_at_the_pre_split_count_is_stale_and_the_first_to_follow_resets_it():
    """Polygon's exact counts lag a split by a filing: BKNG's sample after its twenty-five-for-
    one still reads the count before it. Such samples are stale, not a disagreement; the first
    that follows the ratio is where the counts speak. One that moves by neither, after stale
    ones, is still a contradiction; stale to the span's end, the counts cannot speak at all."""
    before, four = (D[0], 100.0), Split(D[9], 1, 4, "A")
    later = [(D[9], 100.0), (D[12], 101.0), (D[19], 402.0)]
    found = rebuild.classify(four, before, later)
    assert (found["counts"], found["kind"], found["followed"]) == ("follow", "split",
                                                                    (D[19], 402.0))
    assert found["stale"] == ((D[9], 100.0), (D[12], 101.0))
    assert "after 2 stale sample(s)" in found["detail"]
    neither = rebuild.classify(four, before, [(D[12], 101.0), (D[19], 250.0)])
    assert (neither["counts"], neither["kind"]) == ("neither", "contradiction")
    stale = rebuild.classify(four, before, [(D[12], 101.0), (D[19], 99.0)])
    assert (stale["counts"], stale["kind"]) == ("stale", "split")
    assert "untested, neither the counts nor the price can tell" in stale["detail"]


def test_a_split_rests_on_its_price_where_the_counts_cannot_speak_and_must_gap():
    """MNST's two-for-one, its counts stale to the window's end: the raw price opens at half
    the close before, within 10%, so it is a split, tested by its price. A two-for-one whose
    price opens at the close before is a contradiction, whatever the counts say; and one whose
    price moves by neither is too."""
    before, two = (D[0], 100.0), Split(D[9], 1, 2, "A")
    stale = [(D[12], 100.0)]
    halved = (D[8], 91.43, D[9], 46.37)
    found = rebuild.classify(two, before, stale, halved)
    assert (found["counts"], found["prices"], found["kind"]) == ("stale", "gap", "split")
    check = rebuild.SplitCheck("K", "K", two, before, stale[0], True, found["counts"],
                               found["kind"], False, found["detail"], prices=found["prices"])
    assert check.tested and check.applies
    unmoved = rebuild.classify(two, before, [(D[12], 199.0)], (D[8], 91.43, D[9], 90.0))
    assert (unmoved["counts"], unmoved["prices"], unmoved["kind"]) == (
        "follow", "no gap", "contradiction")
    odd = rebuild.classify(two, before, stale, (D[8], 91.43, D[9], 70.0))
    assert (odd["prices"], odd["kind"]) == ("neither", "contradiction")


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
    assert "no share sample on or after it" in aaa.detail
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


def test_the_raw_gap_is_the_open_on_the_day_over_the_close_on_the_trading_day_before():
    """A split executed on a Monday is read from Friday's close to Monday's open."""
    from praxis.midoffice.refdata.polygon import Bar

    def bar(day, o, c):
        return Bar(day, o, o, c, c, 1.0, False)

    friday, monday = dt.date(2026, 4, 2), dt.date(2026, 4, 6)
    bars = [bar(dt.date(2026, 4, 1), 4100.0, 4150.0), bar(friday, 4150.0, 4194.31),
            bar(monday, 166.25, 170.0)]
    assert rebuild.raw_gap(bars, monday) == (friday, 4194.31, monday, 166.25)
    assert rebuild.raw_gap(bars, dt.date(2026, 4, 3)) == (friday, 4194.31, monday, 166.25)
    assert rebuild.raw_gap(bars, dt.date(2026, 4, 1)) is None
    assert rebuild.raw_gap(bars, dt.date(2026, 4, 7)) is None


def test_a_record_with_no_identifier_and_the_change_days_stale_one_are_read_past():
    """Everest Group's first record under EG carries a name and no FIGI or CIK: no evidence,
    and the next sample decides. On a span's first day a record naming the ticker's previous
    holder is the change day's, read past too. A record naming a third security fails the
    span, and the failure says whether its share-class FIGI is the expected security's."""
    def record(ticker, on, figi, cik, share=None):
        return Record(ticker, on, figi, cik, figi or "Everest", "CS", None, share)

    fetched = rebuild.Fetched(records={
        ("EG", D[0]): record("EG", D[0], None, None),
        ("EG", D[9]): record("EG", D[9], None, "0001"),
        ("NEW", D[0]): record("NEW", D[0], "OLD", "0002"),
        ("NEW", D[9]): record("NEW", D[9], "F2", "0003", "S2"),
        ("ODD", D[0]): record("ODD", D[0], "F9", None, "S3"),
        ("ODD", D[9]): record("ODD", D[9], "F3", "0004", "S3"),
        ("NONE", D[0]): record("NONE", D[0], None, None),
    })
    spans = [span("CIK0001", "EG", D[0], D[19], D[0], D[9]),
             Span("F2", "F2", "NEW", D[0], D[19], 1, (D[0], D[9]), previous="OLD"),
             Span("F3", "F3", "ODD", D[0], D[19], 1, (D[0], D[9]), previous="OLD"),
             span("F4", "NONE", D[0], D[19], D[0])]
    eg, new, odd, none = rebuild.check_identities(spans, fetched)
    assert eg.passed and eg.record.on == D[9] and eg.skipped[0][0] == D[0]
    assert "neither a FIGI nor a CIK" in eg.detail
    assert new.passed and "the ticker's previous holder" in new.skipped[0][1]
    assert not odd.passed and "its share-class FIGI, S3, is the expected security's" in odd.detail
    assert not none.passed and "no sample carries an identifier" in none.detail


def test_the_plan_reads_back_as_written(tmp_path):
    path = tmp_path / "plan.csv"
    path.write_text(
        "key,name,ticker,first,last,held,member_days,samples,new_samples,sample_dates\n"
        "F1,Aaa,AAA,2024-01-01,2024-01-31,20,15,2,1,2024-01-01 2024-01-31\n",
        encoding="utf-8",
    )
    (plan,) = rebuild.read_plan(path)
    assert plan == Span("F1", "Aaa", "AAA", D[0], D[30], 15, (D[0], D[30]))
    corrected = tmp_path / "corrected.csv"
    corrected.write_text(
        "key,name,ticker,first,last,held,member_days,samples,new_samples,sample_dates,previous\n"
        "F1,Aaa,AAA,2024-01-01,2024-01-31,20,15,2,1,2024-01-01 2024-01-31,F0\n"
        "F2,Bbb,BBB,2024-01-01,2024-01-31,20,15,2,1,2024-01-01 2024-01-31,\n",
        encoding="utf-8",
    )
    first, second = rebuild.read_plan(corrected)
    assert first.previous == "F0" and second.previous is None
