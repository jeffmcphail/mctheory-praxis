"""Build the S&P 500 panel's identity table: who each ticker is, on every date it has a price,
and which security each index member is, on every date it is one.

    python scripts/build_sp500_identity.py data/external/polygon/sp500_daily_*.npz --offline
    python scripts/build_sp500_identity.py data/external/polygon/sp500_daily_*.npz

The panel names a security by a ticker string; see `midoffice.refdata.identity` for why that
is not an identity and how this one is built. The sources are the reference records the
panel fetch cached, Polygon's ticker events, and the panel's own prices. The events are the
only network this touches -- `/vX/reference/tickers/{id}/events`, asked once per member
ticker over the window and once per cached FIGI no answer carries -- and every answer, found
or not, is cached beside the panel's, so a second run is offline. `--offline` refuses to ask
at all, and stops if an answer is missing.

`--resolve` puts the member-days neither source names to the reference endpoint as well,
`/v3/reference/tickers/{t}?date=`, on the hypotheses in `HYPOTHESES`, and takes only what it
confirms (`identity.resolve_named`); its outputs carry `_resolved` in their names, so the
table without it is never overwritten.

`--correct` (with `--resolve`) then takes the rebuild's own evidence -- its prices and share
samples, cached by `scripts/fetch_sp500_rebuild.py` -- to the table, as midoffice07 ruled on
midoffice06's gate: each span starts at its first price, and one with none goes; a span every
record under its ticker refutes goes; `REORGANISATIONS` are dated by the identifier the table
keys on; `CONFIRMATIONS` add what the endpoint confirms at both ends. It reads membership
through the renames, maps it to securities for the plan, and writes `_corrected` files, with
two more: the links between securities, and each security's member-days. The records the
table is built from stay the panel fetch's own (`--panel-samples-every`), so the rebuild's
samples check it rather than make it.

Writes, beside the panel (gitignored, like it):

    sp500_identity_<first>_<last>.csv          the table: one row per security per ticker
                                               per span of the window, with its evidence
    sp500_identity_<first>_<last>_members.csv  each member ticker's runs of membership, the
                                               security each names, and how each date maps
    sp500_identity_<first>_<last>_fetch.csv    what a rebuild must fetch: per member security
                                               and span, the ticker, the share-count samples
    sp500_identity_<first>_<last>.json         the guards, the findings and the counts

and, with --correct, two more, carrying `_corrected` as the rest do:

    sp500_identity_<first>_<last>_links.csv       each reorganisation, one security under two
                                               identifiers, and each merger, two securities
    sp500_identity_<first>_<last>_membership.csv  each security's member-days, membership read
                                               through the renames

and stops with a non-zero status if a guard fails: every (ticker, date) the panel prices maps
to exactly one security; no security under two tickers on one date; no ticker naming two
securities on one date; every member membership() returns, on every date of the window, maps
to a security or is named as missing.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from praxis.midoffice.refdata import identity as idn  # noqa: E402
from praxis.midoffice.refdata import sp500  # noqa: E402
from praxis.midoffice.refdata.polygon import (  # noqa: E402
    PolygonClient,
    PolygonError,
    PolygonNotFoundError,
    read_api_key,
)

HYPOTHESES = (
    idn.Hypothesis("DAY", ("DAY",), "CDAY for DAY", same_as="CDAY"),
    idn.Hypothesis("WBD", ("WBD",), "DISCA for WBD", same_as="DISCA"),
    idn.Hypothesis("SW", ("WRK",), "WRK for SW"),
    idn.Hypothesis("CPAY", ("FLT",), "FLT for CPAY", same_as="CPAY"),
    idn.Hypothesis("SKYD", ("VIAC", "PARA"), "PARA for SKYD (and VIAC, PARA's ticker before)"),
    idn.Hypothesis("BG", ("BG",), "Bunge Limited for BG", called="Bunge Limited"),
    idn.Hypothesis("SMCI", ("SMCI",), "SMCI itself", same_as="SMCI"),
    idn.Hypothesis("KVUE", ("KVUE",), "KVUE itself (not among the brief's candidates)"),
)
"""The public-record candidates for the member-days neither the records nor the events name,
from the brief of midoffice06: hypotheses for the reference endpoint to confirm, never entries
in the table themselves. VIAC is asked beside PARA because PARA has no record before
2022-02-17; KVUE, which the brief does not list, under its own ticker."""

REORGANISATIONS = (
    idn.Reorganisation("BG", "CIK0001144519", "CIK0001996862",
                       "Bunge Limited to Bunge Global SA, the redomicile, dated by the CIK"),
    idn.Reorganisation("VYLR", "BBG024W5VJC3", "BBG024CQR6X9",
                       "Vylor's two FIGIs under one CIK, the prices unbroken"),
)
"""midoffice06's gate found two reorganisations the table did not date by its own key, and
midoffice07 ruled both so: Bunge's records carry Bunge Limited's CIK to 2023-11-07 under the new
name; Vylor's first record names another FIGI than its later ones, under one CIK."""

CONFIRMATIONS = (
    idn.Confirmation("IIVI", dt.date(2022, 1, 3), dt.date(2022, 9, 7),
                     "II-VI's history under IIVI, before it took COHR as Coherent Corp.",
                     expect="BBG000BLW102"),
    idn.Confirmation("COHR", dt.date(2022, 1, 3), dt.date(2022, 7, 1),
                     "COHR before II-VI took it: Coherent, Inc., whose FIGI the change-day "
                     "record of 2022-09-08 carries", expect="BBG000GS6QB4"),
)
"""Spans the table lacks, from midoffice06's gate: Coherent Corp. is II-VI Incorporated, which
bought Coherent, Inc. on 2022-07-01 and traded as COHR from 2022-09-08; its events record only
COHR. COHR's prices before then run unbroken to 2022-07-01, Coherent, Inc.'s last day."""


class Offline:
    """A transport that refuses: under --offline every answer must already be on disk."""

    def get_json(self, url: str):
        raise PolygonError("--offline: an events answer is not in the cache; run without "
                           "--offline once to ask for it")


def _members_by_date(snapshot, dates, renames=()):
    """membership() on every date, from one walk rather than one per date."""
    walked, _ = sp500.intervals(snapshot, renames)
    out = []
    for day in dates:
        interval = next((iv for iv in walked if day in iv), None)
        if interval is None:
            raise SystemExit(f"the snapshot cannot answer {day}: outside its coverage "
                             f"[{snapshot.coverage_start}, {snapshot.coverage_end}]")
        out.append(frozenset(interval.members))
    return out


def _runs_text(runs, limit=6):
    shown = "; ".join(f"{a}..{b} ({n})" for a, b, n in runs[:limit])
    return shown + (f"; and {len(runs) - limit} more" if len(runs) > limit else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("panel", help="the .npz written by fetch_sp500_panel.py")
    parser.add_argument("--snapshots", default="data/external/sp500",
                        help="the membership snapshots; the newest is used")
    parser.add_argument("--cache", default="data/external/polygon/cache")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--out", default=None,
                        help="where to write; defaults to the panel's own directory")
    parser.add_argument("--offline", action="store_true",
                        help="ask Polygon for nothing; stop if an answer is not cached")
    parser.add_argument("--shares-every", type=int, default=63,
                        help="trading days between the fetch plan's share-count samples inside "
                             "each security's span (default 63, a quarter)")
    parser.add_argument("--resolve", action="store_true",
                        help="put the member-days neither source names to the reference "
                             "endpoint, on HYPOTHESES; writes the *_resolved files")
    parser.add_argument("--correct", action="store_true",
                        help="with --resolve: take the rebuild's prices and share samples to "
                             "the table, as midoffice07 ruled; writes the *_corrected files")
    parser.add_argument("--panel-samples-every", type=int, default=252,
                        help="the panel fetch's share-sample grid, in trading days: the records "
                             "on it are the table's first source (252, the panel's; the "
                             "rebuild's samples, cached beside them, are its checks)")
    parser.add_argument("--bars-start", default="2022-01-03",
                        help="the first day of the rebuild's price requests "
                             "(fetch_sp500_rebuild.py's --start), so --correct reads them back")
    parser.add_argument("--bars-end", default="2026-11-04", help="their last, likewise")
    args = parser.parse_args(argv)
    if args.correct and not args.resolve:
        raise SystemExit("--correct corrects the resolved table: give --resolve too")

    path = Path(args.panel)
    if not path.exists():
        raise SystemExit(f"no such panel: {path}")
    data = np.load(path, allow_pickle=False)
    dates = [dt.date.fromisoformat(str(d)) for d in data["dates"]]
    securities = [str(s) for s in data["securities"]]
    prices = data["prices"]
    held = {t: np.isfinite(prices[:, j]) for j, t in enumerate(securities)}
    panel_tickers = set(securities)
    first, last = dates[0], dates[-1]
    stem = f"sp500_identity_{first}_{last}" + (
        "_corrected" if args.correct else "_resolved" if args.resolve else "")
    out_dir = Path(args.out) if args.out else path.parent
    grid = list(range(0, len(dates), args.panel_samples_every))
    if grid[-1] != len(dates) - 1:
        grid.append(len(dates) - 1)
    panel_samples = [dates[i] for i in grid]

    print(f"Panel\n  {path}: {len(securities)} tickers over {len(dates)} trading days, "
          f"{first} .. {last}; {sum(int(m.sum()) for m in held.values()):,} priced "
          f"(ticker, date) cells")

    snapshot = sp500.load_latest(args.snapshots)
    members = _members_by_date(snapshot, dates)
    union = sorted({t for day in members for t in day})
    named_in_window = sorted({
        t for change in snapshot.changes if first <= change.date <= last
        for t in change.added + change.removed
    })
    unreturned = sorted(set(named_in_window) - set(union))
    print(f"\nMembership\n  snapshot {snapshot.content_hash[:12]}, fetched "
          f"{snapshot.fetched_at:%Y-%m-%d %H:%M} UTC")
    stale = [i for i in sp500.validate(snapshot) if i.code == "current_stale"]
    for issue in stale:
        print(f"  {issue.detail}")
    print(f"  {len(union)} tickers are members on some date of the window; "
          f"{len(set(union) - panel_tickers)} of them are not among the panel's "
          f"{len(securities)}; the change table also names, inside the window, "
          f"{', '.join(unreturned) or 'nothing'} that membership() never returns")

    # ------------------------------------------------------------------- the sources
    records = idn.read_records(args.cache, securities, on=panel_samples)
    n_records = sum(len(r) for r in records.values())
    print(f"\nSources\n  reference records cached by the panel fetch: {n_records:,} over "
          f"{sum(bool(r) for r in records.values())} of the {len(securities)} tickers, at "
          f"{len({r.on for rs in records.values() for r in rs})} sample dates -- no requests; "
          f"read on its grid of every {args.panel_samples_every} trading days alone")
    if args.offline:
        client = PolygonClient(api_key="offline", transport=Offline(), cache_dir=Path(args.cache))
    else:
        try:
            key = read_api_key(args.env)
        except PolygonError as error:
            raise SystemExit(f"  {error}; or run with --offline") from None
        client = PolygonClient(api_key=key, cache_dir=Path(args.cache), min_interval=0.1)
    cache = Path(args.cache)

    def ask(ids):
        out, asked = {}, 0
        for identifier in ids:
            asked += not (cache / f"events-{identifier}.json").exists()
            try:
                results = client.ticker_events(identifier)
            except PolygonError as error:
                raise SystemExit(f"  events for {identifier}: {error}") from None
            out[identifier] = (None if results is None
                               else idn.history_from_events(identifier, results))
        return out, asked

    by_ticker, asked_tickers = ask(union + unreturned)
    carried = {h.key for h in by_ticker.values() if h is not None}
    figis = sorted({r.figi for rs in records.values() for r in rs if r.figi} - carried)
    by_figi, asked_figis = ask(figis)
    print(f"  ticker events, by ticker: {len(by_ticker)} asked -- the window's member tickers "
          f"and {len(unreturned)} more -- {sum(h is not None for h in by_ticker.values())} "
          f"found, {sum(h is None for h in by_ticker.values())} with none (delisted); "
          f"{asked_tickers} asked over the network this run, the rest read from the cache")
    print(f"  ticker events, by FIGI, for the cached records' securities no ticker found: "
          f"{len(by_figi)} asked, {sum(h is not None for h in by_figi.values())} found; "
          f"{asked_figis} over the network this run")
    histories = idn.histories_by_key({**by_ticker, **by_figi})

    # ------------------------------------------------------------------- the table
    identities = {t: idn.segment_ticker(t, dates, held[t], records[t], histories)
                  for t in securities}
    found_as: dict[str, str] = {}
    for ticker in union:
        history = by_ticker.get(ticker)
        if history is not None:
            found_as.setdefault(history.key, ticker)
    extra = [by_ticker[t].key for t in union
             if t not in panel_tickers and by_ticker.get(t) is not None]
    rows, conflicts, tenures_of = idn.build_rows(
        dates, identities, records, histories, extra, found_as
    )
    mapping = idn.map_members(dates, members, rows, by_ticker, tenures_of, panel_tickers)
    resolutions: list = []
    links: list = []
    if args.resolve:
        looked = {"asked": 0}

        def lookup(ticker, on):
            looked["asked"] += not (cache / f"ref-{ticker}-{on}.json").exists()
            try:
                results = client.ticker_details(ticker, on)
            except PolygonNotFoundError:
                return None
            except PolygonError as error:
                raise SystemExit(f"  reference record of {ticker} on {on}: {error}") from None
            if not results:
                return None
            return idn.Record(ticker, on, results.get("composite_figi") or None,
                              results.get("cik") or None, results.get("name") or None,
                              results.get("type") or None,
                              idn._date(results.get("list_date")),
                              results.get("share_class_figi") or None)

        named_before = len(mapping.named)
        resolutions = idn.resolve_named(dates, mapping, rows, HYPOTHESES, lookup)
        links = [r.link for r in resolutions if r.link is not None]
        rows = sorted([*rows, *(row for r in resolutions for row in r.rows)],
                      key=lambda r: (r.key, r.first, r.ticker))
        mapping = idn.map_members(dates, members, rows, by_ticker, tenures_of, panel_tickers,
                                  links)
        print(f"\nThe member-days neither source names, put to the reference endpoint: "
              f"{named_before:,} named before, {len(mapping.named):,} after; "
              f"{sum(r.asked for r in resolutions)} lookups, {looked['asked']} over the network "
              f"this run")
        for r in resolutions:
            span = (f"{r.named[0]}..{r.named[1]} ({r.named[2]})" if r.named else "none named")
            print(f"  {r.hypothesis.statement}: {span}; {r.covered} now accounted for, in "
                  f"{r.asked} lookups\n      {r.verdict}")
            for row in r.rows:
                print(f"      takes {row.ticker} {row.first}..{row.last}: {row.name} [{row.key}]")
            if r.link is not None:
                print(f"      link: {r.link.predecessor} before {r.link.successor}, to "
                      f"{r.link.until} -- {r.link.evidence}")
    corrections: list = []
    refused: list[str] = []
    uncorrected = list(rows)
    if args.correct:
        window = (dt.date.fromisoformat(args.bars_start), dt.date.fromisoformat(args.bars_end))
        trading_days: dict[str, tuple] = {}
        looked["bars"] = 0

        def priced(ticker, fetch=False):
            """The ticker's trading days, from the rebuild's prices read back from the cache;
            None where it has not fetched them, unless `fetch`."""
            if ticker not in trading_days:
                key = f"bars-{ticker}-{window[0]}-{window[1]}-adj"
                if not client.has_cached(key) and not fetch:
                    return None
                looked["bars"] += not client.has_cached(key)
                try:
                    bars = client.daily_bars(ticker, *window, adjusted=True)
                except PolygonError as error:
                    raise SystemExit(f"  prices of {ticker}: {error}") from None
                trading_days[ticker] = tuple(b.date for b in bars)
            return trading_days[ticker]

        asked_before = looked["asked"]
        # A correction the endpoint refuses is a finding, not a failure of the build: the
        # table goes on without it, and the gate reads the span as it stands.
        for spec in REORGANISATIONS:
            try:
                rows, correction = idn.reorganise(dates, rows, spec, lookup,
                                                  idn.read_answers(cache, spec.ticker))
            except ValueError as error:
                refused.append(str(error))
                continue
            corrections.append(correction)
            links = [k for k in links if (k.successor, k.predecessor) != (spec.after, spec.before)]
            links.append(correction.link)
        for spec in CONFIRMATIONS:
            try:
                rows, correction = idn.confirm_span(dates, rows, spec, lookup,
                                                    lambda t: priced(t, fetch=True))
            except ValueError as error:
                refused.append(str(error))
                continue
            corrections.append(correction)
        rows, moved = idn.start_at_first_price(rows, priced)
        answers: dict[str, list] = {}

        def answers_of(ticker):
            if ticker not in answers:
                answers[ticker] = idn.read_answers(cache, ticker)
            return answers[ticker]

        rows, refuted = idn.drop_refuted(rows, answers_of)
        corrections += moved + refuted
        # A tenure the records refute is no ticker of its security's: CRH's CRHC, from which
        # the renames would read a changes-table CRHC, had it one, as CRH.
        for correction in refuted:
            gone = correction.removed[0]
            tenures_of[correction.key] = [
                t for t in tenures_of.get(correction.key, ())
                if not (t.ticker == gone.ticker and t.start <= gone.last
                        and (t.end is None or t.end >= gone.first))
            ]
        mapping = idn.map_members(dates, members, rows, by_ticker, tenures_of, panel_tickers,
                                  links)
        print(f"\nThe rebuild's evidence, taken to the table: {len(corrections)} corrections; "
              f"{looked['asked'] - asked_before} reference lookups and {looked['bars']} price "
              f"request(s) over the network this run")
        for rule in ("reorganisation", "confirmed", "first price", "unpriced", "refuted"):
            for correction in (c for c in corrections if c.rule == rule):
                print(f"  {rule:<14} {correction.detail}"
                      + (f" [{correction.asked} lookups]" if correction.asked else ""))
                for row in correction.removed:
                    print(f"      was   {row.ticker} {row.first}..{row.last}: {row.name} "
                          f"[{row.key}]")
                for row in correction.added:
                    print(f"      now   {row.ticker} {row.first}..{row.last}: {row.name} "
                          f"[{row.key}]")
        for why in refused:
            print(f"  refused        {why}")
    guards = [
        idn.guard_cells(dates, held, rows),
        idn.guard_one_ticker(rows),
        idn.guard_ticker_overlap(rows),
        idn.guard_members(dates, members, mapping),
    ]
    keys = sorted({r.key for r in rows})
    with_records = {r.key for rs in records.values() for r in rs}
    with_events = set(histories)
    with_reference = {r.key for r in rows if r.source == "reference"}
    print(f"\nThe table\n  {len(rows)} rows: {len(keys)} securities, each under one ticker "
          f"over one span of the window; {sum(r.source == 'panel' for r in rows)} rows priced "
          f"by the panel, {sum(r.source == 'events' for r in rows)} from events alone, "
          f"{sum(r.source == 'presumed' for r in rows)} presumed")
    print(f"  identified by the records and the events {len(with_records & with_events)}, by "
          f"the records alone {len(with_records - with_events)}, by the events alone "
          f"{len(set(keys) - with_records - with_reference)}"
          + (f", by the reference endpoint alone "
             f"{len(with_reference - with_records - with_events)}" if args.resolve else ""))
    print("\nGuards")
    for guard in guards:
        print(f"  {'ok' if guard.passed else 'FAILED'}  {guard.name}: {guard.checked:,} "
              f"checked, {len(guard.failures)} failure(s)")
        for failure in guard.failures[:20]:
            print(f"      {failure}")

    # ------------------------------------------------------------------- findings
    boundaries = [b for ident in identities.values() for b in ident.boundaries]
    print("\nSplices: one ticker, two unrelated securities")
    for b in (b for b in boundaries if b.kind == "splice"):
        print(f"  {b.ticker:<6} {b.before_name} [{b.before}] to {b.last_before}; "
              f"{b.after_name} [{b.after}] from {b.first_after}\n         {b.evidence}")
    print("\nReorganisations: a new identifier under the same ticker, the prices unbroken")
    for b in (b for b in boundaries if b.kind == "reorganisation"):
        print(f"  {b.ticker:<6} {b.before} to {b.last_before}, {b.after} from {b.first_after}: "
              f"{b.before_name} -> {b.after_name}")
    unassigned = [(t, u) for t, ident in identities.items() for u in ident.unassigned]
    print(f"\nPriced dates no evidence assigns: {sum(u[2] for _, u in unassigned):,}")
    for ticker, (a, b, n, why) in unassigned:
        print(f"  {ticker:<6} {a}..{b} ({n}): {why}")
    print(f"\nWhere the events and the panel's records and prices disagree, the panel's taken: "
          f"{len(conflicts)}")
    for conflict in conflicts:
        print(f"  {conflict.name} [{conflict.key}]: {conflict.detail}")

    foreign = idn.foreign_prices(dates, identities, mapping)
    print("\nMember dates on which the panel's price under the member's ticker is another "
          "security's")
    for ticker, by_owner in sorted(foreign.items()):
        for owner, runs in by_owner.items():
            name = next((r.name for r in rows if r.key == owner), owner)
            print(f"  {ticker:<6} {sum(n for *_, n in runs):>5} day(s), {name} [{owner}]: "
                  f"{_runs_text(runs)}")
    found = idn.renames_from(tenures_of, rows, {v[0] for v in mapping.mapped.values()})
    today_list = set(sp500.membership(snapshot, snapshot.coverage_end))
    renames, skipped = idn.applicable_renames(found, today_list, snapshot.changes)
    renamed = _members_by_date(snapshot, dates, renames)
    phantoms, missed = idn.phantoms(dates, members, renamed, renames)
    at_date = {day: i for i, day in enumerate(dates)}
    phantom_set = {
        (t, dates[i]) for t, runs in phantoms.items() for a, b, _ in runs
        for i in range(at_date[a], at_date[b] + 1)
    }
    print("\nPhantoms: member dates membership() gives that the change table, read in "
          "today's tickers through the table's renames, does not")
    print(f"  the members' ticker changes in the table: {len(found)}; read through: "
          f"{len(renames)}; not: {len(skipped)}")
    for rename, why in skipped:
        print(f"    not {rename.old} -> {rename.new} ({rename.date}): {why}")
    for ticker, runs in sorted(phantoms.items()):
        hit = next((mapping.mapped.get((ticker, a)) for a, *_ in runs
                    if (ticker, a) in mapping.mapped), None)
        print(f"  {ticker:<6} {sum(n for *_, n in runs):>5} day(s): {_runs_text(runs)}"
              + (f" -- {hit[0]}, trading as {hit[1]}" if hit else ""))
    for ticker, runs in sorted(missed.items()):
        print(f"  {ticker:<6} the other way: {sum(n for *_, n in runs)} member day(s) it does "
              f"not give: {_runs_text(runs)}")
    if not phantoms and not missed:
        print("  none")
    walked = {(i.code, i.detail) for i in sp500.validate(snapshot)
              if i.code in ("added_absent", "removed_present")}
    read = {(i.code, i.detail) for i in sp500.validate(snapshot, renames)
            if i.code in ("added_absent", "removed_present")}
    print(f"  the change table's inconsistencies, read through the renames: {len(walked)} -> "
          f"{len(read)}")
    for label, group in (("resolved", walked - read), ("new", read - walked)):
        for code, detail in sorted(group):
            print(f"    {label}: {detail.split(' recorded')[0]} {code.replace('_', ' ')}, "
                  f"{detail.split(' on ')[1].split(' ')[0]}")

    member_lists, member_map = members, mapping
    if args.correct:
        # The panel's membership: read through the renames, so a phantom -- ECHO before
        # EchoStar joined as SATS -- is nobody's member-day, and mapped to securities.
        member_lists = renamed
        member_map = idn.map_members(dates, renamed, rows, by_ticker, tenures_of,
                                     panel_tickers, links)
        unasked = sorted({t for day in renamed for t in day} - set(by_ticker) - panel_tickers)
        read_guard = idn.guard_members(dates, renamed, member_map)
        read_guard = idn.GuardResult(read_guard.name + ", membership read through the renames",
                                     read_guard.checked, read_guard.failures)
        guards.append(read_guard)
        print(f"\nMembership read through the renames, for the plan and the panel: "
              f"{len(member_map.mapped):,} member-days mapped to securities, "
              f"{len(member_map.named):,} named; against membership() as given, "
              f"{len(mapping.mapped):,} and {len(mapping.named):,}")
        print(f"  {'ok' if read_guard.passed else 'FAILED'}  {read_guard.name}: "
              f"{read_guard.checked:,} checked, {len(read_guard.failures)} failure(s)")
        for failure in read_guard.failures[:20]:
            print(f"      {failure}")
        if unasked:
            print(f"  member tickers whose events were never asked: {', '.join(unasked)}")
        moved_days = collections.Counter(
            v[0] for v in mapping.mapped.values()) - collections.Counter(
            v[0] for v in member_map.mapped.values())
        for key, n in sorted(moved_days.items()):
            name = next((r.name for r in rows if r.key == key), key)
            print(f"  {name} [{key}]: {n:,} member-day(s) fewer, read through the renames")

    truncated = idn.truncations(dates, mapping, panel_tickers, exclude=phantom_set)
    print("\nTruncations: member dates on which the member traded under another ticker than "
          "the panel's, which has no price for it (phantom dates left out)")
    for ticker, by_pair in sorted(truncated.items()):
        for (key, traded), runs in by_pair.items():
            print(f"  {ticker:<6} as {traded:<6} {sum(n for *_, n in runs):>5} day(s): "
                  f"{_runs_text(runs)}")
    absent = sorted(t for t in union if t not in panel_tickers)
    absent_days = sum(1 for day in members for t in day if t not in panel_tickers)
    print(f"  and {len(absent)} member tickers the panel does not hold at all, "
          f"{absent_days:,} member-days")

    dups = idn.duplicates(dates, member_lists, member_map)
    print(f"\nDuplicates: one security named by two member tickers on one date: "
          f"{sum(len(v) for v in dups.values())}")
    for key, items in sorted(dups.items()):
        for tickers, (a, b, n) in items:
            print(f"  {key}: {', '.join(tickers)} {a}..{b} ({n})")

    # Named member-days, with the change table's same-day swaps as candidates.
    named_runs: dict[tuple[str, str], list[int]] = {}
    at = {day: i for i, day in enumerate(dates)}
    for (ticker, day), why in member_map.named.items():
        named_runs.setdefault((ticker, why), []).append(at[day])
    print(f"\nMember-days no security accounts for: {len(member_map.named):,} -- named, each with "
          f"why")
    candidates_of: dict[str, list] = {}
    for (ticker, why), cells in sorted(named_runs.items()):
        runs = idn.date_runs(dates, cells)
        print(f"  {ticker:<6} {len(cells):>5} day(s) {_runs_text(runs)}\n         {why}")
        added = runs[0][0]
        swap = next((c for c in snapshot.changes if c.date == added and ticker in c.added),
                    None)
        if swap is not None and why.startswith("missing"):
            found = idn.swap_candidates(added, swap.removed, dates, rows)
            candidates_of[ticker] = found
            for t, key, name, gone in found:
                print(f"         candidate: {t}, {name} [{key}], removed the same day and "
                      f"priced to {gone} -- a rename recorded as a swap?")
            if swap.removed and not found:
                print(f"         the change table removes {', '.join(swap.removed)} the same "
                      f"day, none of them gone from the prices then")

    # ------------------------------------------------------------------- the member listing
    how_by_run: dict[tuple, collections.Counter] = {}
    traded_by_run: dict[tuple, dict[str, list[int]]] = {}
    for (ticker, run_first, run_last), key in member_map.anchors.items():
        counts = collections.Counter()
        traded: dict[str, list[int]] = {}
        for i in range(at[run_first], at[run_last] + 1):
            hit = member_map.mapped.get((ticker, dates[i]))
            if hit is not None:
                counts[hit[2]] += 1
                traded.setdefault(hit[1], []).append(i)
            elif (ticker, dates[i]) in member_map.named:
                counts[member_map.named[(ticker, dates[i])].split(":")[0]] += 1
        how_by_run[(ticker, run_first, run_last)] = counts
        traded_by_run[(ticker, run_first, run_last)] = traded
    describe = {r.key: r for r in rows}
    member_rows = []
    for (ticker, run_first, run_last), key in sorted(member_map.anchors.items()):
        row = describe.get(key)
        traded = traded_by_run[(ticker, run_first, run_last)]
        sources = []
        if key in with_records:
            sources.append("records")
        if key in with_events:
            sources.append("events")
        if key in with_reference:
            sources.append("reference")
        member_rows.append({
            "ticker": ticker,
            "member_from": run_first,
            "member_to": run_last,
            "member_days": sum(how_by_run[(ticker, run_first, run_last)].values()),
            "key": key or "",
            "name": row.name if row else "",
            "figi": row.figi if row else "",
            "cik": row.cik if row else "",
            "traded_as": "; ".join(
                f"{t} {a}..{b}" for t, cells in traded.items()
                for a, b, _ in idn.date_runs(dates, cells)
            ),
            "how": "; ".join(f"{k} {v}" for k, v in how_by_run[(ticker, run_first, run_last)]
                             .most_common()),
            "sources": "+".join(sources) or "neither",
            "in_panel": ticker in panel_tickers,
        })
    by_source = collections.Counter(r["sources"] for r in member_rows)
    print(f"\nThe {len(union)} member tickers: {len(member_rows)} runs of membership; the "
          f"security each names identified by " + ", ".join(
              f"{k} {v}" for k, v in by_source.most_common()))

    # ------------------------------------------------------------------- the fetch plan
    member_keys = sorted({v[0] for v in member_map.mapped.values()})
    member_days = collections.Counter((v[0], v[1]) for v in member_map.mapped.values())
    cached_refs = {(ticker, r.on) for ticker, rs in records.items() for r in rs}
    # A corrected table keeps the grid its securities were sampled on: their first days in the
    # table before the corrections.
    origins = {}
    for row in uncorrected:
        if row.key not in origins or row.first < origins[row.key]:
            origins[row.key] = row.first
    plan = idn.fetch_plan(dates, rows, member_days, member_keys, cached_refs,
                          every=args.shares_every, origins=origins if args.correct else None)
    tickers = sorted({p.ticker for p in plan})
    new_tickers = [t for t in tickers if t not in panel_tickers]
    samples = sum(len(p.samples) for p in plan)
    new_samples = sum(p.new_samples for p in plan)
    print(f"\nThe fetch the rebuild needs, for the {len(member_keys)} securities that are "
          f"members on some date of the window")
    print(f"  {len(plan)} spans under {len(tickers)} tickers; {len(new_tickers)} of the tickers "
          f"are not the panel's: {' '.join(new_tickers)}")
    print(f"  prices: 2 requests a ticker, adjusted and not, over the window -- "
          f"{2 * len(new_tickers)} new, the panel's {len(tickers) - len(new_tickers)} cached "
          f"(cut to each security's spans)")
    print(f"  share counts: {samples:,} samples inside the securities' own spans (each span's "
          f"first and last day and every {args.shares_every} trading days), {new_samples:,} "
          f"not on disk")
    print(f"  split events, for the share-count rule recorded in midoffice03: one request a "
          f"ticker, {len(tickers)}")
    print(f"  in all {2 * len(new_tickers) + new_samples + len(tickers):,} requests")
    unresolved = sorted({t for (t, _) in member_map.named})
    print(f"  not in it: the member-days named above, under {', '.join(unresolved)}, which "
          f"need a source this one is not" if unresolved else
          "  every member-day is in it: none is named")

    # ------------------------------------------------------------------- write
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / f"{stem}.csv"
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "figi", "cik", "name", "type", "ticker", "first", "last",
                         "held", "source", "starts", "ends"])
        for r in rows:
            writer.writerow([r.key, r.figi or "", r.cik or "", r.name or "", r.kind or "",
                             r.ticker, r.first, r.last, r.held, r.source, r.starts, r.ends])
    listing = out_dir / f"{stem}_members.csv"
    with listing.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(member_rows[0]))
        writer.writeheader()
        writer.writerows(member_rows)
    fetch = out_dir / f"{stem}_fetch.csv"
    with fetch.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "name", "ticker", "first", "last", "held", "member_days",
                         "samples", "new_samples", "sample_dates"]
                        + (["previous"] if args.correct else []))
        for p in plan:
            writer.writerow([p.key, p.name or "", p.ticker, p.first, p.last, p.held,
                             p.member_days, len(p.samples), p.new_samples,
                             " ".join(str(d) for d in p.samples)]
                            + ([p.previous or ""] if args.correct else []))
    written = [table, listing, fetch]
    if args.correct:
        # The links the panel needs: a reorganisation is one security, its prices one series,
        # whether the panel's records found it or the corrections did; a merger is two.
        first_of, last_of = {}, {}
        for r in rows:
            first_of.setdefault(r.key, r)
            last_of[r.key] = r
        linked = [(b.after, b.before, "reorganisation", b.ticker, b.last_before, b.first_after,
                   b.evidence, "records") for b in boundaries if b.kind == "reorganisation"]
        linked += [(k.successor, k.predecessor, k.kind, last_of[k.predecessor].ticker, k.until,
                    first_of[k.successor].first, k.evidence, "reference") for k in links]
        joins = out_dir / f"{stem}_links.csv"
        with joins.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["successor", "predecessor", "kind", "ticker", "until",
                             "first_after", "evidence", "source"])
            writer.writerows(sorted(linked, key=lambda x: (x[4], x[0])))
        # Each security's member-days, read through the renames: who the panel holds when.
        runs_of: dict[tuple, list[int]] = {}
        for (ticker, day), (key, traded_as, how) in member_map.mapped.items():
            runs_of.setdefault((key, ticker, traded_as, how), []).append(at[day])
        member_file = out_dir / f"{stem}_membership.csv"
        with member_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["key", "member", "traded", "how", "first", "last", "days"])
            for (key, ticker, traded_as, how), cells in sorted(runs_of.items()):
                for a, b, n in idn.date_runs(dates, cells):
                    writer.writerow([key, ticker, traded_as, how, a, b, n])
        written += [joins, member_file]
    summary = out_dir / f"{stem}.json"
    summary.write_text(json.dumps({
        "panel": str(path),
        "snapshot": snapshot.content_hash,
        "window": [str(first), str(last)],
        "rows": len(rows),
        "securities": len(keys),
        "guards": [{"name": g.name, "checked": g.checked, "failures": list(g.failures)}
                   for g in guards],
        "boundaries": [{"ticker": b.ticker, "kind": b.kind, "before": b.before,
                        "after": b.after, "last_before": str(b.last_before),
                        "first_after": str(b.first_after), "evidence": b.evidence}
                       for b in boundaries],
        "conflicts": [{"key": c.key, "name": c.name, "detail": c.detail} for c in conflicts],
        "renames": [{"date": str(r.date), "old": r.old, "new": r.new,
                     "since": str(r.since) if r.since else None} for r in renames],
        "phantoms": {t: [[str(a), str(b), n] for a, b, n in runs]
                     for t, runs in phantoms.items()},
        "named": {f"{t}|{why}": len(cells) for (t, why), cells in named_runs.items()},
        "candidates": {t: [[c[0], c[1], c[2], str(c[3])] for c in found]
                       for t, found in candidates_of.items()},
        "mapped": dict(collections.Counter(v[2] for v in member_map.mapped.values())),
        "resolutions": [
            {"statement": r.hypothesis.statement, "member": r.hypothesis.member,
             "named": [str(r.named[0]), str(r.named[1]), r.named[2]] if r.named else None,
             "confirmed": r.confirmed, "covered": r.covered, "verdict": r.verdict,
             "asked": r.asked,
             "rows": [[row.key, row.ticker, str(row.first), str(row.last), row.name]
                      for row in r.rows],
             "link": None if r.link is None else [r.link.successor, r.link.predecessor,
                                                   str(r.link.until), r.link.evidence]}
            for r in resolutions
        ],
        **({"corrections": [
            {"rule": c.rule, "key": c.key, "ticker": c.ticker, "detail": c.detail,
             "asked": c.asked,
             "removed": [[r.key, r.ticker, str(r.first), str(r.last)] for r in c.removed],
             "added": [[r.key, r.ticker, str(r.first), str(r.last)] for r in c.added]}
            for c in corrections
        ], "links": [[k.successor, k.predecessor, str(k.until), k.kind, k.evidence]
                     for k in links],
            "refused": refused, "renamed_named": len(member_map.named)}
           if args.correct else {}),
    }, indent=1), encoding="utf-8")
    written.append(summary)
    print("\nWrote\n" + "\n".join(f"  {f}" for f in written))
    return 0 if all(g.passed for g in guards) else 1


if __name__ == "__main__":
    raise SystemExit(main())
