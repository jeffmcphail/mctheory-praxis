"""Build the S&P 500 panel on securities from the rebuild's fetch, after its gate.

    python scripts/build_sp500_securities.py             # the search asks what is not cached
    python scripts/build_sp500_securities.py --offline   # every answer from the cache

See `midoffice.refdata.securities` for what the panel holds and how. In order:

1. **The gate, offline** (`midoffice.refdata.rebuild`): every span's identity and every split
   event, from the cache alone. The build stops unless it passes.
2. **The first-count search.** A span that starts without a share count, no count carried into
   it, has its first quarter searched for its first record with one -- by bisection, between its
   first sample and its first with a count -- and its count carries from that record's own date,
   never back. Each lookup is cached, so a second run asks nothing; `--offline` asks nothing at
   all and stops if an answer is missing.
3. **The build**, from the plan's spans, the reorganisations in the table's links, its
   member-days read through the renames, and the cache: prices, total returns, traded and
   adjusted volumes, share counts, the market series.
4. **Its guards**, any failing a non-zero status: the identity table's, on the panel; every
   member-day the snapshot's walk through the table's renames gives, mapped to a column; and the
   split guard, exact. Then the weighted-count check and the count moves, standing and reported,
   never a source.

The counts follow midoffice08's rulings on what Polygon's records get wrong, each switchable:

- **A first count must be confirmed** (`--first-counts confirmed`): it stands only if the span's
  next sample confirms it within 10%; otherwise the search goes on to the first record the next
  sample confirms, and the days before it have no count. `found` takes the first record with
  any count, midoffice08's build.
- **Isolated errors are repaired** (`--repair isolated`): a sample whose two samples before and
  two after agree with each other within 10%, which differs from each beyond it with no split
  between, is not carried, and the count before it carries through -- unless its own record's
  weighted count confirms it within 10% (`--spare weighted`, midoffice09's ruling 1). Every one
  found is listed, spared or not.
- **A span ends where its security stops trading on an exchange** (`--listed exchange`,
  midoffice09's ruling 2): from its first sample whose record's market is not "stocks", OTC
  Link's after a failure, the span goes, but for any day the index still held it. Every span
  it ends is listed, with the days it removes.
- **The overrides file** (`--overrides`, `config/sp500_share_count_overrides.json`): rulings on
  single securities' counts, each with its evidence, applied and listed. `none` reads none.

Writes, beside the plan (gitignored, like it): `sp500_securities_<first>_<last><suffix>.npz`,
the panel, and `.json`, the report -- `--suffix` `_listed` by default. midoffice09's panel,
`_repaired`, is `--spare none --listed any`; midoffice08's, with no suffix, is `--first-counts
found --repair none --overrides none --spare none --listed any`. An existing file is never
overwritten without `--overwrite`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import functools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from praxis.midoffice.refdata import rebuild, securities, sp500  # noqa: E402
from praxis.midoffice.refdata.polygon import (  # noqa: E402
    PolygonClient,
    PolygonError,
    PolygonNotFoundError,
    read_api_key,
)
from praxis.midoffice.refdata.prices import MARKET_PROXY  # noqa: E402

DEFAULT_PLAN = "data/external/polygon/sp500_identity_2022-01-03_2026-11-03_amended_fetch.csv"


class Offline:
    """A transport that refuses: every answer must already be on disk."""

    def get_json(self, url: str):
        raise PolygonError("--offline: not in the cache")


def sibling(plan: Path, suffix: str) -> Path:
    """The identity build's file beside the plan: `..._fetch.csv` to `..._<suffix>`."""
    return plan.with_name(plan.name.replace("_fetch.csv", suffix))


def reference(cache: Path, ticker: str, day: dt.date) -> dict | None:
    """A cached reference answer's results, or None: not cached, not found, or empty."""
    path = cache / f"ref-{ticker}-{day}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("results") or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--cache", default="data/external/polygon/cache")
    parser.add_argument("--snapshots", default="data/external/sp500")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--start", default="2022-01-03",
                        help="the first day of the rebuild's price requests")
    parser.add_argument("--end", default="2026-11-04", help="their last, likewise")
    parser.add_argument("--market", default=MARKET_PROXY)
    parser.add_argument("--offline", action="store_true",
                        help="ask Polygon for nothing; stop if an answer is not cached")
    parser.add_argument("--min-interval", type=float, default=0.02)
    parser.add_argument("--exempt", default="DD")
    parser.add_argument("--tolerance", type=float, default=rebuild.TOLERANCE)
    parser.add_argument("--largest-term", type=int, default=rebuild.LARGEST_TERM)
    parser.add_argument("--out", default=None, help="where to write; the plan's directory")
    parser.add_argument("--first-counts", choices=("confirmed", "found"), default="confirmed",
                        help="a span's first count must be confirmed by its next sample, within "
                             "the tolerance (the default, midoffice08's ruling 1), or the first "
                             "record with any count stands (found, midoffice08's build)")
    parser.add_argument("--repair", choices=("isolated", "none"), default="isolated",
                        help="isolated: a sample whose two samples before and two after agree "
                             "and which differs from them, no split between, is not carried "
                             "(the default, midoffice08's ruling 2); none repairs nothing")
    parser.add_argument("--overrides", default="config/sp500_share_count_overrides.json",
                        help="the rulings on single securities' counts (midoffice08's ruling "
                             "3); none reads none")
    parser.add_argument("--spare", choices=("weighted", "none"), default="weighted",
                        help="weighted: an isolated error its own record's weighted count "
                             "confirms within the tolerance is spared (the default, midoffice09's "
                             "ruling 1); none spares nothing, midoffice09's build")
    parser.add_argument("--listed", choices=("exchange", "any"), default="exchange",
                        help="exchange: a span ends where its records place the security off an "
                             "exchange, but for any member-day (the default, midoffice09's "
                             "ruling 2); any keeps every span, midoffice09's build")
    parser.add_argument("--suffix", default="_listed",
                        help="appended to the output's stem (default _listed; midoffice09's panel "
                             "is _repaired, midoffice08's has none)")
    parser.add_argument("--overwrite", action="store_true",
                        help="write over an existing panel of the same name; without it the "
                             "build stops first")
    args = parser.parse_args(argv)

    plan_path = Path(args.plan)
    cache = Path(args.cache)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    spans = rebuild.read_plan(plan_path)
    offline = PolygonClient(api_key="offline", transport=Offline(), cache_dir=cache)
    if args.offline:
        client = offline
    else:
        try:
            key = read_api_key(args.env)
        except PolygonError as error:
            raise SystemExit(f"  {error}; or run with --offline") from None
        client = PolygonClient(api_key=key, cache_dir=cache, min_interval=args.min_interval)
    print(f"Plan\n  {plan_path}: {len(spans)} spans, {len({s.key for s in spans})} securities, "
          f"{len({s.ticker for s in spans})} tickers")

    # ------------------------------------------------------------------ 1. the gate, offline
    fetched = rebuild.fetch(offline, spans, start, end, market=args.market,
                            cached=offline.has_cached)
    if fetched.failures:
        raise SystemExit(f"  the gate needs every answer on disk; {len(fetched.failures)} are "
                         f"not: {fetched.failures[:3]} -- run fetch_sp500_rebuild.py")
    @functools.lru_cache(maxsize=32)
    def bars(ticker: str, basis: str) -> tuple:
        # Read back from the cache as each column needs them: holding every ticker's at once
        # is most of a gigabyte, on a machine that has run short of memory before.
        return offline.daily_bars(ticker, start, end, adjusted=basis == "adj")

    identities = rebuild.check_identities(spans, fetched)
    checks = rebuild.check_splits(
        spans, fetched, gaps=lambda t, on: rebuild.raw_gap(bars(t, "raw"), on),
        exempt=args.exempt.split(","), tolerance=args.tolerance, largest_term=args.largest_term)
    verdict = rebuild.gate(identities, checks)
    kinds = collections.Counter(c.kind for c in checks)
    reconciled = [c for c in identities if c.passed and c.reconciled]
    print(f"\nThe gate, offline: {'passes' if verdict.passed else 'STOPS'}\n"
          f"  identity: {sum(c.passed for c in identities)} of {len(identities)} spans pass, "
          f"{len(reconciled)} by the share-class FIGI"
          + "".join(f"\n    {c.span.ticker}: {c.detail}" for c in reconciled)
          + f"\n  splits: {len(checks)} events, {kinds['split']} splits, {kinds['price-only']} "
          f"price-only, {kinds['contradiction']} contradictions, {len(verdict.untested)} "
          f"untested")
    if not verdict.passed:
        for check in verdict.identity_failures:
            print(f"  FAILED  {check.span.ticker}: {check.detail}")
        for check in verdict.contradictions:
            print(f"  CONTRADICTS  {check.detail}")
        raise SystemExit(1)

    dates = tuple(bar.date for bar in bars(args.market, "adj"))
    at = {day: i for i, day in enumerate(dates)}
    links = list(csv.DictReader(sibling(plan_path, "_links.csv").open(encoding="utf-8")))
    reorganisations = {r["predecessor"]: r["successor"] for r in links
                       if r["kind"] == "reorganisation"}
    share_classes, ciks = rebuild.own_identifiers(spans, fetched)
    members = list(csv.DictReader(sibling(plan_path, "_membership.csv").open(encoding="utf-8")))
    member_days: dict[str, list[tuple[dt.date, dt.date]]] = {}
    for row in members:
        member_days.setdefault(row["key"], []).append(
            (dt.date.fromisoformat(row["first"]), dt.date.fromisoformat(row["last"])))

    def market_of(ticker, day):
        results = reference(cache, ticker, day)
        return None if not results else (results.get("market"), results.get("primary_exchange"))

    delistings: list[securities.Delisting] = []
    if args.listed == "exchange":
        spans, delistings = securities.listed_spans(spans, market_of, member_days, dates)
    print("\nOff an exchange: a span ends where its records place the security off one, but "
          "for any day the index still held it -- "
          + (f"{len(delistings)} span(s) ended" if args.listed == "exchange"
             else "none ended (--listed any)"))
    for d in delistings:
        span = d.span
        priced = [b.date for b in bars(span.ticker, "adj") if span.first <= b.date <= span.last
                  and (d.kept_to is None or b.date > d.kept_to)]
        print(f"  {span.ticker:<6} {span.key} {span.first}..{span.last}: off an exchange from "
              f"{d.off_exchange} ({d.market}, {d.exchange}); last on one "
              f"{d.last_listed or 'none in the span'}; "
              + (f"kept to {d.kept_to}" if d.kept_to else "the span goes")
              + (f", for the {len(d.held)} member-day(s) the index still held it, "
                 f"{', '.join(str(x) for x in d.held)}" if d.held else "")
              + f"; {len(priced)} priced day(s) removed"
              + (f", {priced[0]}..{priced[-1]}" if priced else ""))
    columns = securities.columns_of(spans, reorganisations)

    # ------------------------------------------------------------------ 2. the search
    print("\nThe first-count search: each span that starts without a share count, no count "
          "carried into it, searched between its first sample and its first with a count"
          + (f"; a first count the next sample does not confirm within {args.tolerance:.0%} is "
             f"passed over for the first it confirms, the days before that without a count"
             if args.first_counts == "confirmed" else "; the first record with any count "
             "stands (--first-counts found)"))
    searches: list[securities.FirstCount] = []
    asked = 0
    for column in columns:
        carried = False
        for span in column.spans:
            counts = [(day, fetched.shares.get((span.ticker, day))) for day in span.samples]
            first_count = next(((d, c) for d, c in counts if c), None)
            if not carried and not counts[0][1]:
                days = dates[at[span.samples[0]]:at[first_count[0]] + 1] if first_count else ()

                def lookup(ticker, on):
                    nonlocal asked
                    asked += not client.has_cached(f"ref-{ticker}-{on}")
                    try:
                        results = client.ticker_details(ticker, on)
                    except PolygonNotFoundError:
                        return None, None
                    record = rebuild.record_of(ticker, on, results)
                    return record, (results or {}).get("share_class_shares_outstanding")

                def names(record, key=span.key):
                    return record.key == key or bool(
                        rebuild.reconciles(record, key, share_classes, ciks))

                try:
                    found = securities.search_first_count(
                        span, list(days), first_count, lookup, names,
                        confirm=args.first_counts == "confirmed", tolerance=args.tolerance)
                except PolygonError as error:
                    raise SystemExit(f"  {span.ticker}: {error}") from None
                searches.append(found)
                print(f"  {span.ticker:<6} {span.first}..{span.last} ({span.member_days:>5} "
                      f"member-days): {found.detail}")
                if found.failure:
                    raise SystemExit("  the search stops on another security's record")
            if first_count:
                carried = True
    print(f"  {len(searches)} spans, {sum(len(s.probes) for s in searches)} lookups, {asked} "
          f"over the network this run")

    # ------------------------------------------------------------------ 3. the build
    samples: dict[str, list[tuple[dt.date, str, float]]] = {}
    for span in spans:
        for day in span.samples:
            count = fetched.shares.get((span.ticker, day))
            if count:
                samples.setdefault(span.key, []).append((day, span.ticker, count))
    for found in searches:
        if found.found is not None and found.found[0] not in found.span.samples:
            samples.setdefault(found.span.key, []).append(
                (found.found[0], found.span.ticker, found.found[1]))
    weighted_on: dict[tuple[str, dt.date], float] = {}
    if args.spare == "weighted":
        for key, rows in samples.items():
            for day, ticker, _count in rows:
                results = reference(cache, ticker, day)
                if results and results.get("weighted_shares_outstanding"):
                    weighted_on[(key, day)] = float(results["weighted_shares_outstanding"])
    repairs: list[securities.Repair] = []
    if args.repair == "isolated":
        repairs = securities.isolated_errors(columns, samples, checks, args.tolerance,
                                             weighted_on)
        removed = {(r.sample_key, r.day) for r in repairs if not r.spared}
        samples = {key: [row for row in rows if (key, row[0]) not in removed]
                   for key, rows in samples.items()}
    spared = [r for r in repairs if r.spared]
    print(f"\nIsolated errors: a sample whose two samples before and two after agree with each "
          f"other within {args.tolerance:.0%}, and which differs from each of them beyond it, no "
          f"split between, is not carried, unless its own record's weighted count confirms it -- "
          + (f"{len(repairs)} found, {len(repairs) - len(spared)} removed, {len(spared)} spared"
             if args.repair == "isolated" else "none removed (--repair none)"))
    for r in repairs:
        own = (f"its weighted count {r.weighted:,.0f}, {r.weighted / r.count:.3f} of it"
               if r.weighted is not None else "no weighted count")
        print(f"  {'SPARED ' if r.spared else 'removed'} {r.label:<6} {r.ticker:<6} {r.day} "
              f"{r.count:>16,.0f}; around it "
              + ", ".join(f"{d} {c:,.0f}" for d, c in r.neighbours)
              + f" -- {r.count / np.median([c for _, c in r.neighbours]):.3f} of their median; "
              + own)
    overrides = ([] if args.overrides == "none"
                 else securities.read_overrides(args.overrides))
    panel = securities.build_panel(
        dates, columns, bars, checks, samples, fetched.dividends, member_days,
        (bars(args.market, "adj"), bars(args.market, "raw"),
         fetched.dividends.get(args.market, ())), overrides)
    labels = [c.label for c in columns]
    repeated = [label for label, k in collections.Counter(labels).items() if k > 1]
    if repeated:
        labels = [f"{c.label}~{c.key}" if c.label in repeated else c.label for c in columns]
    label_of = {key: labels[j] for j, c in enumerate(columns) for key in c.keys}
    merged = [c for c in columns if len(c.keys) > 1]
    print(f"\nThe panel\n  {len(dates)} trading days, {dates[0]} .. {dates[-1]}; {len(columns)} "
          f"securities, {len(spans)} spans under {len({s.ticker for s in spans})} tickers; "
          f"{len(merged)} reorganisations merged, each one column under its successor's key")
    for column in merged:
        print(f"    {column.label:<6} {' -> '.join(column.keys)}")
    splices = sorted({s.ticker for s in spans if s.previous and s.previous not in
                      {k for c in columns for k in c.keys if s.key in c.keys}})
    print(f"  splices -- a ticker another security held before a column's span, which never "
          f"enters that column: {', '.join(splices) or 'none'}")
    if repeated:
        print(f"  labels repeated, so carried with the key: {', '.join(repeated)}")
    print(f"  a change of ticker inside a column, bridged by the raw closes: "
          f"{len(panel.bridges)}")
    for b in panel.bridges:
        print(f"    {label_of[b.key]:<6} {b.before} {b.last_before} "
              f"{b.raw_before:,.2f} -> {b.after} {b.first_after} {b.raw_after:,.2f}"
              + (f", split events {b.ratio:g}" if b.ratio != 1.0 else "")
              + f": return {b.relative - 1:+.4%}; the earlier run's scale {b.scale:.6g}")
    priced = int(np.isfinite(panel.prices).sum())
    print(f"  priced cells {priced:,}; member-days {int(panel.membership.sum()):,}, of them "
          f"priced {int((panel.membership & np.isfinite(panel.prices)).sum()):,}")

    print(f"\nTotal returns\n  {panel.distributions:,} distributions reinvested; exact "
          f"duplicates dropped {len(panel.duplicates)}"
          + "".join(f"\n    {d.ticker} {d.ex_date} {d.cash_amount} {d.kind} frequency "
                    f"{d.frequency}, paid {d.pay_date}" for d in panel.duplicates))
    print(f"  ex-dates without a price in their column, reinvested on the next priced day: "
          f"{len(panel.unpriced_ex_dates)}"
          + "".join(f"\n    {label_of[k]} {d.ex_date} {d.cash_amount} -> {on}"
                    for k, d, on in panel.unpriced_ex_dates))
    print(f"  on or before a column's first price, which no return carries: "
          f"{len(panel.before_first_price)}"
          + "".join(f"\n    {label_of[k]} {d.ex_date} {d.cash_amount}"
                    for k, d in panel.before_first_price))
    with np.errstate(divide="ignore", invalid="ignore"):
        price_r = panel.prices[1:] / panel.prices[:-1] - 1.0
        total_r = panel.total_prices[1:] / panel.total_prices[:-1] - 1.0
    gap = total_r - price_r
    print(f"  {int((np.abs(gap) > 0).sum()):,} security-days whose total return is not their "
          f"price return; mean annual difference over the members' days "
          f"{np.nanmean(np.where(panel.membership[1:], gap, np.nan)) * 252:.2%}")
    market_gap = panel.market_total_returns - panel.market_returns
    print(f"  {args.market}: {panel.market_distributions} distributions; total less price return "
          f"{np.nansum(market_gap):.4f} over the window, {np.nanmean(market_gap) * 252:.2%} a "
          f"year")

    print(f"\nShare counts\n  splits applied on their execution dates: {len(panel.applied)}; "
          f"stale samples read past: {len(panel.stale_read_past)}"
          + "".join(f"\n    {label_of[k]:<6} {day} at {count:,.0f}"
                    for k, day, count in panel.stale_read_past))
    for key, day, why in panel.unapplied:
        print(f"  ! a split on {day} not applied to {key}: {why}")
    print(f"  counts carried into a span from before it, until a sample of its own resets "
          f"them: {len(panel.carries)}")
    for carry in panel.carries:
        print(f"    {carry.ticker:<6} from {carry.first} at {carry.count:,.0f} "
              f"{'(a reorganisation)' if carry.reorganisation else '(from its own earlier span)'}"
              f", {'to ' + str(carry.until) if carry.until else 'to the span end, 1:1, FLAGGED'}")
    print(f"  overrides, {args.overrides}: {len(overrides)} in the file, "
          f"{len(panel.overridden)} used" if args.overrides != "none"
          else "  overrides: none read (--overrides none)")
    for o, days, member, unread in panel.overridden:
        what = f"carry {o.count:,.0f}" if o.action == "carry" else "no count"
        print(f"    {o.label:<6} {o.key} {o.first}..{o.last}: {what} -- {days} days in its spans, "
              f"{member} member-days, {unread} sample(s) inside not read; {o.ruled}\n"
              f"      {o.evidence}")
    used = {o for o, *_ in panel.overridden}
    for o in overrides:
        if o not in used:
            print(f"  ! an override no column uses: {o.label} {o.key} {o.first}..{o.last}")
    no_count = panel.membership & ~np.isfinite(panel.shares)
    unspanned = panel.membership & (panel.tickers == "")
    ruled_none = sum(member for o, _, member, _ in panel.overridden if o.action == "none")
    print(f"  member-days without a count: {int(no_count.sum()):,} (midoffice07, before the "
          f"search: 696) -- {int((no_count & ~unspanned).sum()) - ruled_none:,} inside the "
          f"spans, before a first count; {ruled_none:,} in an override's range of no count; "
          f"{int(unspanned.sum()):,} after a member's last trade, outside its spans, which no "
          f"price or count reaches")
    for j in np.flatnonzero((no_count & ~unspanned).any(axis=0)):
        days = np.flatnonzero(no_count[:, j] & ~unspanned[:, j])
        print(f"    {labels[j]:<6} {len(days):>4}: {dates[days[0]]}..{dates[days[-1]]}")
    moves = securities.count_moves(panel, checks, args.tolerance)
    print(f"  counts that move beyond {args.tolerance:.0%} from one day to the next, no split "
          f"the gate applies there: {len(moves)} -- a standing check; the gate tests counts "
          f"across split events alone")
    for move in moves:
        print(f"    {move.label:<6} {move.day}: {move.before:,.0f} to {move.after:,.0f}, "
              f"{move.ratio:.4f}")

    # ------------------------------------------------------------------ 4. guards
    snapshot = sp500.load_latest(args.snapshots)
    table = json.loads(sibling(plan_path, ".json").read_text(encoding="utf-8"))
    if table["snapshot"] != snapshot.content_hash:
        raise SystemExit(f"  the table was built on snapshot {table['snapshot'][:12]}, and the "
                         f"newest is {snapshot.content_hash[:12]}: rebuild the table first")
    renames = [sp500.Rename(dt.date.fromisoformat(r["date"]), r["old"], r["new"],
                            dt.date.fromisoformat(r["since"]) if r["since"] else None)
               for r in table["renames"]]
    walked, issues = sp500.intervals(snapshot, renames)
    stale = [i for i in issues if i.code == "current_stale"]
    print(f"\nMembership\n  snapshot {snapshot.content_hash[:12]}, fetched "
          f"{snapshot.fetched_at:%Y-%m-%d %H:%M} UTC, read through the table's "
          f"{len(renames)} renames")
    for issue in stale:
        print(f"  {issue.detail}")
    mapped: dict[tuple[str, dt.date], str] = {}
    for row in members:
        for i in range(at[dt.date.fromisoformat(row["first"])],
                       at[dt.date.fromisoformat(row["last"])] + 1):
            mapped[(row["member"], dates[i])] = row["key"]
    column_of = {k: j for j, c in enumerate(columns) for k in c.keys}
    walk_days, unmapped, off = 0, [], []
    for day in dates:
        interval = next(iv for iv in walked if day in iv)
        for ticker in interval.members:
            walk_days += 1
            key = mapped.get((ticker, day))
            if key is None:
                unmapped.append(f"{ticker} {day}")
            elif not panel.membership[at[day], column_of[key]]:
                off.append(f"{ticker} {day} -> {key}")
    outside = {(row["key"], dates[i]) for row in members
               if row["how"] in ("after its last price", "inside a gap in its prices")
               for i in range(at[dt.date.fromisoformat(row["first"])],
                              at[dt.date.fromisoformat(row["last"])] + 1)}
    guards = securities.identity_guards(panel, bars, outside)
    guards.append(securities.Guard(
        "every member-day of the walk through the renames is a column's member-day",
        walk_days, tuple(unmapped + off)))
    guards.append(securities.Guard(
        "the panel's member-days are the walk's, no more", int(panel.membership.sum()),
        () if int(panel.membership.sum()) == walk_days - len(unmapped) else
        (f"{int(panel.membership.sum())} against {walk_days - len(unmapped)}",)))
    guards.append(securities.split_guard(panel, checks, samples))
    print("\nGuards")
    for guard in guards:
        print(f"  {'ok' if guard.passed else 'FAILED'}  {guard.name}: {guard.checked:,} "
              f"checked, {len(guard.failures)} failure(s)")
        for failure in guard.failures[:10]:
            print(f"      {failure}")

    # ------------------------------------------------------------------ the weighted check
    weighted: dict[str, list] = {}
    for span in spans:
        for day in span.samples:
            results = reference(cache, span.ticker, day)
            if results and results.get("weighted_shares_outstanding"):
                weighted.setdefault(span.key, []).append(
                    (day, span.ticker, float(results["weighted_shares_outstanding"]),
                     results.get("share_class_shares_outstanding")))
    for found in searches:
        if found.found is not None:
            results = reference(cache, found.span.ticker, found.found[0])
            if results and results.get("weighted_shares_outstanding"):
                weighted.setdefault(found.span.key, []).append(
                    (found.found[0], found.span.ticker,
                     float(results["weighted_shares_outstanding"]),
                     results.get("share_class_shares_outstanding")))
    met = securities.weighted_check(panel, weighted)
    carried_met = [w for w in met if not w.own]
    print(f"\nThe weighted-count check: a record's weighted share count against the count the "
          f"panel carries that day, flagged beyond {args.tolerance:.0%}")
    for label, group in (("carried from an earlier record", carried_met),
                         ("the day's own record", [w for w in met if w.own])):
        flags = [w for w in group if w.flagged(args.tolerance)]
        ratios = np.array([w.ratio for w in group])
        print(f"  {label}: {len(group):,} met, {len(flags)} flagged"
              + (f"; ratio median {np.median(ratios):.4f}, p1 {np.percentile(ratios, 1):.4f}, "
                 f"p99 {np.percentile(ratios, 99):.4f}" if group else ""))
    if carried_met:
        print("  every carried one, the reorganisations' and the splits' among them:")
        for w in carried_met:
            print(f"    {w.label:<6} {w.ticker:<6} {w.day} weighted {w.weighted:,.0f} against "
                  f"{w.carried:,.0f}: {w.ratio:.4f}"
                  + ("  FLAGGED" if w.flagged(args.tolerance) else ""))
    own_flags = [w for w in met if w.own and w.flagged(args.tolerance)]
    by_security: dict[str, list] = {}
    for w in own_flags:
        by_security.setdefault(w.label, []).append(w)
    every = collections.Counter(w.label for w in met if w.own)
    print(f"  the day's own record flagged, by security -- {len(by_security)} securities; where "
          f"every record is, a class count beside a company-wide weighted one:")
    for label, group in sorted(by_security.items(), key=lambda kv: -len(kv[1])):
        ratios = [w.ratio for w in group]
        print(f"    {label:<6} {len(group):>3} of {every[label]:>3} records, ratio "
              f"{min(ratios):.4f}..{max(ratios):.4f}, from {group[0].day} to {group[-1].day}")

    # ------------------------------------------------------------------ industries
    sic, sic_on = [], []
    for column in columns:
        found = None
        for span in reversed(column.spans):
            for day in reversed(span.samples):
                results = reference(cache, span.ticker, day)
                if results and results.get("sic_code"):
                    found = (str(results["sic_code"]).strip().zfill(4), day)
                    break
            if found:
                break
        sic.append(found[0] if found else "")
        sic_on.append(str(found[1]) if found else "")
    print(f"\nIndustries\n  each security's SIC code from its own latest record with one: "
          f"{sum(bool(s) for s in sic)} of {len(columns)}"
          + (f"; none for {', '.join(lab for lab, s in zip(labels, sic) if not s)}"
             if not all(sic) else ""))

    # ------------------------------------------------------------------ write
    out_dir = Path(args.out) if args.out else plan_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sp500_securities_{dates[0]}_{dates[-1]}{args.suffix}"
    destination = out_dir / f"{stem}.npz"
    for path in (destination, out_dir / f"{stem}.json"):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"  {path} exists: give another --suffix or --out, or --overwrite")
    np.savez_compressed(
        destination,
        dates=np.array([d.isoformat() for d in dates]),
        securities=np.array([c.key for c in columns]),
        labels=np.array(labels),
        names=np.array([c.name for c in columns]),
        prices=panel.prices,
        total_return_prices=panel.total_prices,
        unadjusted_prices=panel.raw_prices,
        volumes=panel.volumes,
        adjusted_volumes=panel.adjusted_volumes,
        shares_outstanding=panel.shares,
        highs=panel.highs,
        lows=panel.lows,
        tickers=np.array(panel.tickers.tolist(), dtype=str),
        membership=panel.membership,
        market_returns=panel.market_returns,
        market_total_returns=panel.market_total_returns,
        sic_codes=np.array(sic),
        sic_dates=np.array(sic_on),
        snapshot=np.array(snapshot.content_hash),
        plan=np.array(str(plan_path)),
    )
    report = out_dir / f"{stem}.json"
    report.write_text(json.dumps({
        "plan": str(plan_path),
        "snapshot": snapshot.content_hash,
        "dates": [str(dates[0]), str(dates[-1]), len(dates)],
        "securities": len(columns),
        "gate": {"passed": verdict.passed, "identity": len(identities),
                 "reconciled": [[c.span.ticker, c.reconciled, c.detail] for c in reconciled],
                 "splits": dict(kinds)},
        "rules": {"first_counts": args.first_counts, "repair": args.repair,
                  "spare": args.spare, "listed": args.listed,
                  "overrides": args.overrides, "tolerance": args.tolerance},
        "delistings": [{"ticker": d.span.ticker, "key": d.span.key, "first": str(d.span.first),
                        "last": str(d.span.last), "off_exchange": str(d.off_exchange),
                        "market": d.market, "exchange": d.exchange,
                        "last_listed": str(d.last_listed) if d.last_listed else None,
                        "kept_to": str(d.kept_to) if d.kept_to else None,
                        "held": [str(x) for x in d.held]} for d in delistings],
        "repairs": [{"label": r.label, "key": r.key, "sample_key": r.sample_key,
                     "ticker": r.ticker, "day": str(r.day), "count": r.count,
                     "weighted": r.weighted, "spared": r.spared,
                     "neighbours": [[str(d), c] for d, c in r.neighbours]} for r in repairs],
        "overrides": [{"label": o.label, "key": o.key, "first": str(o.first), "last": str(o.last),
                       "action": o.action, "count": o.count, "days": days, "member_days": member,
                       "unread": unread, "ruled": o.ruled, "evidence": o.evidence}
                      for o, days, member, unread in panel.overridden],
        "searches": [{"ticker": s.span.ticker, "key": s.span.key,
                      "unconfirmed": [str(s.unconfirmed[0]), s.unconfirmed[1]]
                      if s.unconfirmed else None,
                      "bracket": [str(d) for d in s.bracket] if s.bracket else None,
                      "found": [str(s.found[0]), s.found[1]] if s.found else None,
                      "probes": [[str(d), c, k] for d, c, k in s.probes], "detail": s.detail}
                     for s in searches],
        "requests": asked,
        "reorganisations": [list(c.keys) for c in merged],
        "bridges": [{"key": b.key, "before": b.before, "after": b.after,
                     "last_before": str(b.last_before), "first_after": str(b.first_after),
                     "raw_before": b.raw_before, "raw_after": b.raw_after, "ratio": b.ratio,
                     "scale": b.scale} for b in panel.bridges],
        "distributions": panel.distributions,
        "duplicates": [[d.ticker, str(d.ex_date), d.cash_amount] for d in panel.duplicates],
        "unpriced_ex_dates": [[k, d.ticker, str(d.ex_date), str(on)]
                              for k, d, on in panel.unpriced_ex_dates],
        "before_first_price": [[k, d.ticker, str(d.ex_date)] for k, d in panel.before_first_price],
        "market_distributions": panel.market_distributions,
        "applied": [[k, str(day), r] for k, day, r in panel.applied],
        "stale_read_past": [[k, str(day), c] for k, day, c in panel.stale_read_past],
        "carries": [{"key": c.key, "ticker": c.ticker, "first": str(c.first), "count": c.count,
                     "until": str(c.until) if c.until else None,
                     "reorganisation": c.reorganisation} for c in panel.carries],
        "member_days_without_count": int(no_count.sum()),
        "member_days_after_last_trade": int(unspanned.sum()),
        "count_moves": [[m.label, str(m.day), m.before, m.after] for m in moves],
        "current_stale": [i.detail for i in stale],
        "guards": [{"name": g.name, "checked": g.checked, "failures": list(g.failures[:50])}
                   for g in guards],
        "weighted": [{"label": w.label, "ticker": w.ticker, "day": str(w.day),
                      "weighted": w.weighted, "carried": w.carried, "own": w.own,
                      "ratio": w.ratio, "flagged": w.flagged(args.tolerance)} for w in met],
        "sic": dict(zip(labels, sic)),
    }, indent=1), encoding="utf-8")
    print(f"\nWrote\n  {destination}\n  {report}")
    return 0 if all(g.passed for g in guards) else 1


if __name__ == "__main__":
    raise SystemExit(main())
