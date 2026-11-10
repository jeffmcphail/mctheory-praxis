"""Fetch what the identity table's plan needs for the rebuild, and check it before anything is
built: every span's identity, and every split against the share counts.

    python scripts/fetch_sp500_rebuild.py                  # resumable: the cache keeps each answer
    python scripts/fetch_sp500_rebuild.py --fetch-only     # the fetch alone, the checks left
    python scripts/fetch_sp500_rebuild.py --offline        # the checks again, from the cache alone

The plan is `scripts/build_sp500_identity.py --resolve`'s `*_resolved_fetch.csv`: per member
security and span, the ticker it traded under and the dates to sample its share count -- inside
its own spans only, each span's first and last day among them. Asked for, through
`PolygonClient`'s cache, at the original panel fetch's rate: each ticker's daily prices over the
window, adjusted and not; the reference record at every sample; each ticker's split events and
cash distributions, the adjusted prices carrying none; and the market series' prices, split
events and distributions, as a ticker's.

Then the gate (`midoffice.refdata.rebuild`). It stops -- a non-zero status, and the report says
why -- if any span's first share sample names another security than the table expects, or has
no record at all, or if any split event but DuPont's is a contradiction: a whole-number ratio
the share counts either side do not follow, or a fractional one they do not leave unchanged.
Writes the report beside the plan, as `sp500_rebuild_gate_<first>_<last>.json`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from praxis.midoffice.refdata import rebuild  # noqa: E402
from praxis.midoffice.refdata.polygon import (  # noqa: E402
    PolygonClient,
    PolygonError,
    read_api_key,
)
from praxis.midoffice.refdata.prices import MARKET_PROXY  # noqa: E402

DEFAULT_PLAN = "data/external/polygon/sp500_identity_2022-01-03_2026-11-03_resolved_fetch.csv"


class Offline:
    """A transport that refuses: under --offline every answer must already be on disk."""

    def get_json(self, url: str):
        raise PolygonError("--offline: not in the cache")


def memory() -> str:
    """This process's memory and the machine's, logged with the progress: if the fetch is
    stopped for memory again, its log says whether the memory was its own."""
    if sys.platform != "win32":
        import resource

        return f"peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:,.0f} MB"
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak", "working_set", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
                "pagefile", "peak_pagefile", "private")]

    class Status(ctypes.Structure):  # MEMORYSTATUSEX
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total", "available", "total_pagefile", "available_pagefile", "total_virtual",
                "available_virtual", "available_extended")]

    mine, machine = Counters(), Status()
    mine.cb, machine.length = ctypes.sizeof(mine), ctypes.sizeof(machine)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process_info = ctypes.windll.psapi.GetProcessMemoryInfo
    process_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    if not (process_info(kernel32.GetCurrentProcess(), ctypes.byref(mine), mine.cb)
            and kernel32.GlobalMemoryStatusEx(ctypes.byref(machine))):
        return "memory unread"
    mb = 2**20
    return (f"this process {mine.working_set / mb:,.0f} MB (peak {mine.peak / mb:,.0f}, "
            f"{mine.private / mb:,.0f} committed); machine {machine.available / mb:,.0f} of "
            f"{machine.total / mb:,.0f} MB available")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--cache", default="data/external/polygon/cache")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--start", default="2022-01-03",
                        help="the first day of each ticker's prices asked for; the panel "
                             "fetch's, so its tickers are read back from the cache")
    parser.add_argument("--end", default="2026-11-04", help="the last day asked for, likewise")
    parser.add_argument("--market", default=MARKET_PROXY,
                        help="the market series, asked for as a ticker is: prices, split events "
                             f"and distributions ({MARKET_PROXY}, the panel's)")
    parser.add_argument("--min-interval", type=float, default=0.02,
                        help="seconds between requests (0.02, the panel fetch's own)")
    parser.add_argument("--offline", action="store_true",
                        help="ask Polygon for nothing; anything not cached is a failure")
    parser.add_argument("--fetch-only", action="store_true",
                        help="fetch, and stop before the checks: an --offline run makes them")
    parser.add_argument("--exempt", default="DD",
                        help="tickers whose split contradictions are reported but do not stop "
                             "the gate (comma-separated; DD by default, DuPont's)")
    parser.add_argument("--tolerance", type=float, default=rebuild.TOLERANCE,
                        help="how near the share counts must come to a split's ratio, or to "
                             "no change, to follow it or stay unchanged (0.10)")
    parser.add_argument("--largest-term", type=int, default=rebuild.LARGEST_TERM,
                        help="the largest term, in lowest terms, of a whole-number ratio (100)")
    args = parser.parse_args(argv)

    plan_path = Path(args.plan)
    spans = rebuild.read_plan(plan_path)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    cache = Path(args.cache)
    tickers = sorted({s.ticker for s in spans})
    samples = {(s.ticker, d) for s in spans for d in s.samples}
    print(f"Plan\n  {plan_path}: {len(spans)} spans, {len({s.key for s in spans})} securities, "
          f"{len(tickers)} tickers, {len(samples):,} share samples")
    if args.offline:
        client = PolygonClient(api_key="offline", transport=Offline(), cache_dir=cache)
    else:
        try:
            key = read_api_key(args.env)
        except PolygonError as error:
            raise SystemExit(f"  {error}; or run with --offline") from None
        client = PolygonClient(api_key=key, cache_dir=cache, min_interval=args.min_interval)

    began = time.monotonic()

    def progress(done: int, total: int, what: str) -> None:
        if done % 500 == 0 or done == total:
            print(f"  {done:>6,}/{total:,}  {what:<6}  {time.monotonic() - began:6.0f}s  "
                  f"{memory()}", flush=True)

    print(f"\nFetching\n  at the start: {memory()}", flush=True)
    fetched = rebuild.fetch(client, spans, start, end, market=args.market,
                            cached=client.has_cached, on_progress=progress)
    print(f"  {fetched.requests:,} requests over the network, in "
          f"{time.monotonic() - began:.0f}s; {len(fetched.failures)} failed")
    for what, why in fetched.failures[:20]:
        print(f"    {what}: {why[:140]}")
    if len(fetched.failures) > 20:
        print(f"    and {len(fetched.failures) - 20} more")

    # A security's distributions are those under its tickers inside its own spans: the events
    # under one ticker can be several securities'.
    paid: dict[str, list] = {}
    for span in spans:
        paid.setdefault(span.key, []).extend(
            d for d in fetched.dividends.get(span.ticker, ())
            if span.first <= d.ex_date <= span.last
        )
    inside = sorted((d for events in paid.values() for d in events), key=lambda d: d.ex_date)
    market = [d for d in fetched.dividends.get(args.market, ()) if start <= d.ex_date <= end]
    print("\nDistributions: the cash paid under each security's tickers inside its spans")
    print(f"  {len(inside):,} over {sum(1 for e in paid.values() if e)} of {len(paid)} "
          f"securities, {sum(1 for e in paid.values() if not e)} of them paying none; by type "
          f"{dict(Counter(d.kind or 'none' for d in inside))}; by currency "
          f"{dict(Counter(d.currency or 'none' for d in inside))}")
    print(f"  {args.market}, the market series: {len(market)} in the window, "
          f"{sum(d.cash_amount for d in market):.4f} a share")
    for d in inside:
        if d.kind != "CD" or (d.currency or "USD") != "USD":
            print(f"    {d.ex_date} {d.ticker:<6} {d.kind or '?':<3} {d.cash_amount:>10.4f} "
                  f"{d.currency or '?'}  frequency {d.frequency}")
    if args.fetch_only:
        print("\nChecks: not made (--fetch-only); run with --offline to make them")
        return 0 if not fetched.failures else 1

    identities = rebuild.check_identities(spans, fetched)
    splits = rebuild.check_splits(spans, fetched, exempt=args.exempt.split(","),
                                  tolerance=args.tolerance, largest_term=args.largest_term)
    verdict = rebuild.gate(identities, splits)
    print("\nIdentity: each span's first share sample against the security the table expects")
    print(f"  {sum(c.passed for c in identities)} of {len(identities)} spans pass")
    for check in verdict.identity_failures:
        span = check.span
        print(f"  FAILED  {span.ticker} {span.first}..{span.last} ({span.member_days} member-days)"
              f": {check.detail}")
    print(f"\nSplits: each event classified by its ratio -- whole numbers, no term over "
          f"{args.largest_term} in lowest terms, or not -- and by the security's share counts "
          f"either side, within {args.tolerance:.0%}")
    kinds = {kind: [c for c in splits if c.kind == kind]
             for kind in ("split", "price-only", "contradiction")}
    exempted = [c for c in kinds["contradiction"] if c.exempt]
    print(f"  {len(splits)} events inside the securities' spans: {len(kinds['split'])} splits, "
          f"applied to the carried share count; {len(kinds['price-only'])} price-only, not "
          f"applied; {len(kinds['contradiction'])} contradictions, {len(exempted)} of them "
          f"exempt ({args.exempt}); {len(verdict.untested)} untested, by the ratio alone")
    for kind, mark in (("contradiction", "CONTRADICTS"), ("price-only", "price-only"),
                       ("split", "split")):
        for check in kinds[kind]:
            label = ("exempt" if check.exempt else mark) if kind == "contradiction" else mark
            print(f"  {label:<11} {'' if check.tested else 'untested '}{check.name}: "
                  f"{check.detail}")
    print(f"\nGate: {'passes' if verdict.passed and not fetched.failures else 'STOPS'}")
    if fetched.failures:
        print(f"  {len(fetched.failures)} request(s) failed; run again to ask for them")
    report = plan_path.parent / (
        "sp500_rebuild_gate_" + "_".join(plan_path.stem.split("_")[2:4]) + ".json"
    )
    report.write_text(json.dumps({
        "plan": str(plan_path),
        "spans": len(spans),
        "requests": fetched.requests,
        "failures": fetched.failures,
        "identity": [
            {"key": c.span.key, "ticker": c.span.ticker, "first": str(c.span.first),
             "last": str(c.span.last), "member_days": c.span.member_days,
             "passed": c.passed, "detail": c.detail}
            for c in identities
        ],
        "market": args.market,
        "dividends": {
            "events": len(inside),
            "securities_paying": sum(1 for e in paid.values() if e),
            "securities": len(paid),
            "by_type": dict(Counter(d.kind or "none" for d in inside)),
            "by_currency": dict(Counter(d.currency or "none" for d in inside)),
            "market_events": len(market),
            "not_regular_or_not_usd": [
                {"ticker": d.ticker, "ex_date": str(d.ex_date), "kind": d.kind,
                 "cash_amount": d.cash_amount, "currency": d.currency, "frequency": d.frequency}
                for d in inside if d.kind != "CD" or (d.currency or "USD") != "USD"
            ],
        },
        "tolerance": args.tolerance,
        "largest_term": args.largest_term,
        "splits": [
            {"key": c.key, "name": c.name, "ticker": c.split.ticker,
             "execution_date": str(c.split.execution_date), "split_from": c.split.split_from,
             "split_to": c.split.split_to, "ratio": c.split.ratio,
             "before": None if c.before is None else [str(c.before[0]), c.before[1]],
             "after": None if c.after is None else [str(c.after[0]), c.after[1]],
             "whole": c.whole, "counts": c.counts, "kind": c.kind, "tested": c.tested,
             "applies": c.applies, "exempt": c.exempt, "detail": c.detail}
            for c in splits
        ],
        "passed": verdict.passed and not fetched.failures,
    }, indent=1), encoding="utf-8")
    print(f"  wrote {report}")
    return 0 if verdict.passed and not fetched.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
