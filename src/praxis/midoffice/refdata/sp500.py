"""Point-in-time S&P 500 constituents, derived from the Wikipedia changes table.

Ingestion and query are separate operations, and that separation is the whole design:

    fetch()       touches the network, validates the page structure, and returns an
                  immutable Snapshot.  This is the only impure function in the module.
    membership()  is pure.  Given a Snapshot and a date it returns the constituent list,
                  or refuses.  No network, no clock, no hidden state.

Liveness is preserved -- fetch as often as you like and the newest snapshot carries the
newest changes.  Reproducibility is preserved -- a run records which snapshot it used, that
snapshot is content-addressed and immutable, and re-running against it gives the same answer
forever.

Method.  Wikipedia publishes the *current* constituent list and a table of historical
additions and removals.  Walking backwards from the current list and reversing each change
reconstructs membership at any earlier date, including for companies that were added and
later dropped -- which is what most free approaches get wrong.

Coverage is honest.  A snapshot can only answer between its earliest recorded change and the
moment it was fetched.  Outside that range membership() raises rather than extrapolating.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Wikipedia split this data across two articles: the current constituent list stayed on
# the original page, and the historical add/remove table moved to its own article.  Both are
# read and their tables pooled, then each is located by column signature - so a future
# re-merge or further split needs only this list changed, not the parser.
CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
DEFAULT_SOURCES = (CONSTITUENTS_URL, CHANGES_URL)

# Permissive on purpose: this validates shape, not existence.  Class shares (BRK.B, BF.B)
# and the occasional 5-letter ticker must pass.
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}(\.[A-Z]{1,2})?$")

# Sanity band for an index that is nominally 500 names.  Class shares and pending changes
# make the true count drift slightly; anything outside this is a parsing failure, not drift.
PLAUSIBLE_SIZE = (480, 520)


class StructureError(RuntimeError):
    """The source page no longer matches the structure this parser expects."""


class CoverageError(LookupError):
    """The requested date lies outside what this snapshot can answer."""


@dataclass(frozen=True, order=True)
class Change:
    """All index changes effective on one date, grouped.

    Grouping at parse time is deliberate.  The predecessor implementation re-queried the
    changes DataFrame by date inside the walk loop, where pandas returns a DataFrame for a
    date with several changes and a Series for a date with one -- so the single-change case
    iterated the ticker string character by character and appended 'W', 'H', 'R' instead of
    'WHR'.  Grouping once, here, removes the entire class of bug rather than patching it.
    """

    date: dt.date
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning"
    code: str
    detail: str


@dataclass(frozen=True)
class Interval:
    """Membership over the half-open date range [start, end)."""

    start: dt.date
    end: dt.date
    members: tuple[str, ...]

    def __contains__(self, d: dt.date) -> bool:
        return self.start <= d < self.end


@dataclass(frozen=True)
class Snapshot:
    source_url: str
    fetched_at: dt.datetime           # timezone-aware, UTC
    current: tuple[str, ...]          # constituents as at fetched_at
    changes: tuple[Change, ...]       # descending by date, one entry per date
    overrides: tuple[tuple[str, str], ...] = ()
    content_hash: str = ""            # sha256 over the normalised content, set by _finalise

    # -- coverage ---------------------------------------------------------------

    @property
    def coverage_start(self) -> dt.date:
        if not self.changes:
            return self.fetched_at.date()
        return min(c.date for c in self.changes)

    @property
    def coverage_end(self) -> dt.date:
        return self.fetched_at.date()

    def covers(self, d: dt.date) -> bool:
        return self.coverage_start <= d <= self.coverage_end

    # -- serialisation ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "source_url": self.source_url,
            "fetched_at": self.fetched_at.isoformat(),
            "content_hash": self.content_hash,
            "overrides": [list(p) for p in self.overrides],
            "current": list(self.current),
            "changes": [
                {"date": c.date.isoformat(), "added": list(c.added), "removed": list(c.removed)}
                for c in self.changes
            ],
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Snapshot":
        return cls(
            source_url=d["source_url"],
            fetched_at=dt.datetime.fromisoformat(d["fetched_at"]),
            current=tuple(d["current"]),
            changes=tuple(
                Change(
                    date=dt.date.fromisoformat(c["date"]),
                    added=tuple(c["added"]),
                    removed=tuple(c["removed"]),
                )
                for c in d["changes"]
            ),
            overrides=tuple(tuple(p) for p in d.get("overrides", ())),
            content_hash=d.get("content_hash", ""),
        )


def _normalised_payload(current: Sequence[str], changes: Sequence[Change]) -> str:
    """Canonical form the content hash is taken over.

    Hashes the parsed content rather than the raw HTML, so cosmetic page edits do not
    produce a new hash but any change to the data does.
    """
    return json.dumps(
        {
            "current": sorted(current),
            "changes": [
                {"date": c.date.isoformat(), "added": sorted(c.added), "removed": sorted(c.removed)}
                for c in sorted(changes, key=lambda c: c.date)
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _finalise(snapshot: Snapshot) -> Snapshot:
    payload = _normalised_payload(snapshot.current, snapshot.changes)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return dataclasses.replace(snapshot, content_hash=digest)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _flatten(col) -> str:
    """Collapse a possibly-MultiIndex column label to a single lowercase string."""
    if isinstance(col, tuple):
        parts = [str(p) for p in col if not str(p).startswith("Unnamed")]
        # De-duplicate the common ('Date', 'Date') case.
        seen, out = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return " ".join(out).strip().lower()
    return str(col).strip().lower()


def _locate_tables(tables: Sequence) -> tuple:
    """Find the constituents table and every changes table, by column signature.

    Positional indexing (`tables[0]`, `tables[1]`) is the single most fragile thing in the
    predecessor, and it has already been overtaken by events: Wikipedia moved the changes
    table to a separate article, so `tables[1]` on the original page is now a navbox.

    Every table matching the changes signature is returned, not just the first, because the
    historical article may split its history across several tables.
    """
    constituents = None
    changes: list = []
    for t in tables:
        cols = [_flatten(c) for c in t.columns]
        joined = " | ".join(cols)
        if constituents is None and any(c == "symbol" for c in cols) and "security" in joined:
            constituents = t
        elif "added" in joined and "removed" in joined and "date" in joined:
            changes.append(t)
    if constituents is None:
        raise StructureError(
            "no constituents table found in any source (expected columns including 'Symbol' "
            f"and 'Security'); saw {len(tables)} tables"
        )
    if not changes:
        raise StructureError(
            "no changes table found in any source (expected columns including 'Date', "
            f"'Added', 'Removed'); saw {len(tables)} tables. If Wikipedia has moved the "
            f"historical add/remove table again, update DEFAULT_SOURCES."
        )
    return constituents, changes


def _column(df, *must_contain: str) -> str:
    """Return the single column whose flattened label contains all the given fragments."""
    hits = [c for c in df.columns if all(f in _flatten(c) for f in must_contain)]
    if len(hits) != 1:
        raise StructureError(
            f"expected exactly one column matching {must_contain!r}, found {len(hits)}: "
            f"{[_flatten(c) for c in df.columns]}"
        )
    return hits[0]


def _clean_ticker(raw, overrides: Mapping[str, str]) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    # "\u2013"/"\u2014" are en- and em-dash, which Wikipedia uses as empty-cell markers.
    # Written as escapes so this file stays pure ASCII across platforms and encodings.
    if s in ("", "NAN", "NONE", "-", "\u2013", "\u2014", "N/A"):
        return ""
    # Wikipedia carries footnote markers, and template markup leaks a trailing pipe into
    # some cells -- "ALLE |", "JCP |", "ITT |" all appear in the live changes table.
    s = re.sub(r"\[.*?\]", "", s)
    s = s.split("|")[0].strip()
    return overrides.get(s, s)


def parse_tables(tables: Sequence, *, overrides: Mapping[str, str] | None = None) -> tuple:
    """Parse already-read HTML tables into (current_constituents, changes).

    Separated from fetch() so the parser is testable without a network call.
    """
    overrides = dict(overrides or {})
    const_df, change_dfs = _locate_tables(tables)

    sym_col = _column(const_df, "symbol")
    current = sorted(
        {
            t
            for t in (_clean_ticker(v, overrides) for v in const_df[sym_col].tolist())
            if t
        }
    )

    by_date: dict[dt.date, dict[str, set]] = {}
    unparsed_dates: list[str] = []
    total_rows = 0

    for change_df in change_dfs:
        date_col = _column(change_df, "date")
        added_col = _column(change_df, "added", "ticker")
        removed_col = _column(change_df, "removed", "ticker")
        total_rows += len(change_df)

        for raw_date, raw_added, raw_removed in zip(
            change_df[date_col].tolist(),
            change_df[added_col].tolist(),
            change_df[removed_col].tolist(),
        ):
            d = _parse_date(raw_date)
            if d is None:
                unparsed_dates.append(str(raw_date))
                continue
            bucket = by_date.setdefault(d, {"added": set(), "removed": set()})
            a = _clean_ticker(raw_added, overrides)
            r = _clean_ticker(raw_removed, overrides)
            if a:
                bucket["added"].add(a)
            if r:
                bucket["removed"].add(r)

    if unparsed_dates and len(unparsed_dates) > 0.05 * max(total_rows, 1):
        raise StructureError(
            f"{len(unparsed_dates)} of {total_rows} change dates failed to parse "
            f"(e.g. {unparsed_dates[:3]}) -- the date format has probably changed"
        )

    changes = tuple(
        sorted(
            (
                Change(date=d, added=tuple(sorted(v["added"])), removed=tuple(sorted(v["removed"])))
                for d, v in by_date.items()
            ),
            key=lambda c: c.date,
            reverse=True,
        )
    )
    return tuple(current), changes


def _parse_date(raw) -> dt.date | None:
    if raw is None:
        return None
    s = re.sub(r"\[.*?\]", "", str(raw)).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y", "%B %d %Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:  # last resort, and only for unambiguous ISO-ish forms
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def fetch(
    *,
    urls: Sequence[str] = DEFAULT_SOURCES,
    overrides: Mapping[str, str] | None = None,
    now: dt.datetime | None = None,
) -> Snapshot:
    """Read the live page and return a validated snapshot.

    Raises StructureError if the page no longer parses, so a broken snapshot is never
    written and the last good one keeps serving.

    `now` is injectable so the caller controls the clock; nothing in this module reads it
    implicitly.
    """
    import pandas as pd  # local import: querying a snapshot must not require pandas

    fetched_at = now or dt.datetime.now(dt.timezone.utc)
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")

    tables: list = []
    for source in urls:
        try:
            tables.extend(pd.read_html(source))
        except Exception as exc:  # noqa: BLE001 - re-raised with the source named
            raise StructureError(f"could not read tables from {source}: {exc!r}") from exc
    current, changes = parse_tables(tables, overrides=overrides)

    snap = _finalise(
        Snapshot(
            source_url=" ".join(urls),
            fetched_at=fetched_at,
            current=current,
            changes=changes,
            overrides=tuple(sorted((overrides or {}).items())),
        )
    )

    errors = [i for i in validate(snap) if i.severity == "error"]
    if errors:
        raise StructureError(
            "snapshot failed validation and was not written:\n"
            + "\n".join(f"  [{i.code}] {i.detail}" for i in errors)
        )
    return snap


def save(snapshot: Snapshot, directory: Path | str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = snapshot.fetched_at.strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"sp500_{stamp}_{snapshot.content_hash[:12]}.json"
    path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    return path


def load(path: Path | str) -> Snapshot:
    return Snapshot.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_latest(directory: Path | str) -> Snapshot:
    """Newest snapshot by fetch time. Explicit, so a run can also pin an older one."""
    paths = sorted(Path(directory).glob("sp500_*.json"))
    if not paths:
        raise FileNotFoundError(f"no snapshots in {directory}")
    return max((load(p) for p in paths), key=lambda s: s.fetched_at)


# ---------------------------------------------------------------------------
# Query -- pure
# ---------------------------------------------------------------------------


def intervals(snapshot: Snapshot) -> tuple[tuple[Interval, ...], tuple[Issue, ...]]:
    """Reconstruct the full membership history as contiguous half-open intervals.

    Returns the intervals ascending by date, plus any anomalies found while walking.  An
    anomaly means the source table disagrees with itself -- reversing a change produced a
    state the change could not have come from.  These are recorded, never swallowed.
    """
    members = set(snapshot.current)
    end = snapshot.fetched_at.date() + dt.timedelta(days=1)  # half-open, so today is included
    out: list[Interval] = []
    issues: list[Issue] = []

    for ch in snapshot.changes:  # descending
        if ch.date >= end:
            issues.append(
                Issue("warning", "future_change",
                      f"change dated {ch.date} is at or after the interval end {end}; skipped")
            )
            continue

        out.append(Interval(start=ch.date, end=end, members=tuple(sorted(members))))

        for t in ch.added:
            if t in members:
                members.discard(t)
            else:
                issues.append(
                    Issue("warning", "added_absent",
                          f"{t} recorded as added on {ch.date} but was not in the list being "
                          f"carried back; source table is internally inconsistent")
                )
        for t in ch.removed:
            if t in members:
                issues.append(
                    Issue("warning", "removed_present",
                          f"{t} recorded as removed on {ch.date} but was already present in the "
                          f"list being carried back; source table is internally inconsistent")
                )
            else:
                members.add(t)

        end = ch.date

    # `members` now holds membership immediately before the earliest recorded change.  How
    # far back that remains true is unknowable from this source, so no interval is emitted
    # for it and coverage_start stops here.
    out.reverse()
    return tuple(out), tuple(issues)


def membership(snapshot: Snapshot, as_of: dt.date) -> tuple[str, ...]:
    """Constituents as at `as_of`. Raises CoverageError outside the snapshot's range."""
    if isinstance(as_of, dt.datetime):
        as_of = as_of.date()
    if not snapshot.covers(as_of):
        raise CoverageError(
            f"{as_of} is outside this snapshot's coverage "
            f"[{snapshot.coverage_start}, {snapshot.coverage_end}]. "
            f"Earlier dates need a source with deeper history; later dates need a newer fetch."
        )
    for iv in intervals(snapshot)[0]:
        if as_of in iv:
            return iv.members
    raise CoverageError(f"{as_of} fell between intervals -- this is a bug, please report")


def all_constituents(snapshot: Snapshot) -> tuple[str, ...]:
    """Every ticker that was a member at any point within coverage."""
    seen: set[str] = set()
    for iv in intervals(snapshot)[0]:
        seen.update(iv.members)
    return tuple(sorted(seen))


# ---------------------------------------------------------------------------
# Validation -- controls that can fail
# ---------------------------------------------------------------------------


def validate(snapshot: Snapshot) -> tuple[Issue, ...]:
    """Structural and plausibility checks. Errors block ingestion; warnings are recorded."""
    issues: list[Issue] = []

    n = len(snapshot.current)
    if not PLAUSIBLE_SIZE[0] <= n <= PLAUSIBLE_SIZE[1]:
        issues.append(
            Issue("error", "size_implausible",
                  f"current constituent count is {n}, outside {PLAUSIBLE_SIZE}; "
                  f"the page structure has probably changed")
        )

    if len(snapshot.changes) < 50:
        issues.append(
            Issue("error", "changes_too_few",
                  f"only {len(snapshot.changes)} change dates parsed; expected hundreds")
        )

    # Shape-check every ticker the snapshot can ever return, not just today's list.  A
    # malformed name arriving through the changes table becomes a phantom member of some
    # historical interval, and checking only `current` would never see it.
    bad_current = sorted(t for t in snapshot.current if not TICKER_RE.match(t))
    bad_history = sorted(
        {t for c in snapshot.changes for t in (c.added + c.removed) if not TICKER_RE.match(t)}
    )
    if bad_current:
        issues.append(
            Issue("error", "malformed_tickers",
                  f"{len(bad_current)} current symbols do not look like tickers: "
                  f"{bad_current[:8]}")
        )
    if bad_history:
        issues.append(
            Issue("error", "malformed_tickers_history",
                  f"{len(bad_history)} symbols in the changes table do not look like tickers: "
                  f"{bad_history[:8]}; these would become phantom members of past intervals")
        )

    # The character-iteration bug splits one ticker into its letters, so it always produces
    # at least two one-character removals on a single date.  Fifty years of real history
    # contains no such date -- single-letter removals (K, M, Q, R, S, T, X) always appear
    # alone -- so this fires on the bug and not on healthy data.  An earlier draft counted
    # one-letter names across history and errored above twelve, which the live index trips
    # on its own (A, C, D, F, J, L, O, P, Q, T, V are all real).
    for ch in snapshot.changes:
        singles = sorted(t for t in ch.removed if len(t) == 1)
        if len(singles) >= 2:
            multi = [t for t in ch.removed if len(t) > 1]
            spelled = [m for m in multi if set(singles) <= set(m)]
            issues.append(
                Issue("error", "split_ticker_suspected",
                      f"{len(singles)} one-character removals on {ch.date}: {singles}"
                      + (f", which spell {spelled}" if spelled else "")
                      + ". This is the signature of a ticker string iterated character by "
                        "character rather than treated as one value.")
            )

    ivs, walk_issues = intervals(snapshot)
    issues.extend(walk_issues)

    # Interval sizes must stay near 500 walking back.  Drift upward is the contamination tell.
    oversized = [iv for iv in ivs if len(iv.members) > PLAUSIBLE_SIZE[1]]
    undersized = [iv for iv in ivs if len(iv.members) < PLAUSIBLE_SIZE[0]]
    if oversized:
        worst = max(oversized, key=lambda iv: len(iv.members))
        issues.append(
            Issue("error", "interval_oversized",
                  f"{len(oversized)} intervals exceed {PLAUSIBLE_SIZE[1]} members "
                  f"(worst: {len(worst.members)} at {worst.start}); membership is accumulating "
                  f"names it should not")
        )
    if undersized:
        worst = min(undersized, key=lambda iv: len(iv.members))
        issues.append(
            Issue("warning", "interval_undersized",
                  f"{len(undersized)} intervals fall below {PLAUSIBLE_SIZE[0]} members "
                  f"(worst: {len(worst.members)} at {worst.start})")
        )

    for a, b in zip(ivs, ivs[1:]):
        if a.end != b.start:
            issues.append(
                Issue("error", "interval_gap",
                      f"intervals are not contiguous: {a.start}..{a.end} then {b.start}..{b.end}")
            )

    return tuple(issues)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_probe(args) -> int:
    """Parse the live page and report, without writing anything."""
    import pandas as pd

    tables = []
    for source in args.source:
        print(f"reading {source}")
        found = pd.read_html(source)
        print(f"  pandas {pd.__version__}: {len(found)} tables")
        for i, t in enumerate(found[:8]):
            print(f"    [{i}] {t.shape[0]:>5} x {t.shape[1]:<3} {[_flatten(c) for c in t.columns][:6]}")
        tables.extend(found)

    print(f"\n{len(tables)} tables pooled across {len(args.source)} source(s)")
    current, changes = parse_tables(tables)
    snap = _finalise(
        Snapshot(
            source_url=" ".join(args.source),
            fetched_at=dt.datetime.now(dt.timezone.utc),
            current=current,
            changes=changes,
        )
    )
    print(f"\ncurrent constituents : {len(snap.current)}")
    print(f"change dates         : {len(snap.changes)}")
    print(f"coverage             : {snap.coverage_start} .. {snap.coverage_end}")
    print(f"content hash         : {snap.content_hash[:16]}")
    multi = [c for c in snap.changes if len(c.added) + len(c.removed) > 2]
    print(f"multi-change dates   : {len(multi)}  (single-change: {len(snap.changes) - len(multi)})")

    issues = validate(snap)
    errs = [i for i in issues if i.severity == "error"]
    warns = [i for i in issues if i.severity == "warning"]
    print(f"\nvalidation: {len(errs)} error(s), {len(warns)} warning(s)")
    for i in errs + warns[:10]:
        print(f"  [{i.severity}] {i.code}: {i.detail}")
    if len(warns) > 10:
        print(f"  ... and {len(warns) - 10} more warnings")
    return 1 if errs else 0


def _cmd_fetch(args) -> int:
    snap = fetch(urls=args.source)
    path = save(snap, args.out)
    print(f"wrote {path}")
    print(f"  {len(snap.current)} constituents, {len(snap.changes)} change dates")
    print(f"  coverage {snap.coverage_start} .. {snap.coverage_end}")
    return 0


def _cmd_members(args) -> int:
    snap = load(args.snapshot) if args.snapshot else load_latest(args.dir)
    try:
        members = membership(snap, dt.date.fromisoformat(args.date))
    except CoverageError as e:
        print(f"refused: {e}")
        return 1
    print(f"{len(members)} constituents as at {args.date} "
          f"(snapshot {snap.content_hash[:12]}, fetched {snap.fetched_at.date()})")
    print("\n".join(members))
    return 0


def _cmd_check(args) -> int:
    snap = load(args.snapshot) if args.snapshot else load_latest(args.dir)
    issues = validate(snap)
    errs = [i for i in issues if i.severity == "error"]
    print(f"snapshot {snap.content_hash[:12]} fetched {snap.fetched_at.isoformat()}")
    print(f"coverage {snap.coverage_start} .. {snap.coverage_end}")
    for i in issues:
        print(f"  [{i.severity}] {i.code}: {i.detail}")
    if not issues:
        print("  clean")
    return 1 if errs else 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sp500", description=__doc__.splitlines()[0])
    p.add_argument(
        "--source",
        action="append",
        default=None,
        help="source URL; repeatable. Defaults to the constituents and historical-components "
             "articles.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", help="parse the live page and report; writes nothing")
    sp.set_defaults(func=_cmd_probe)

    sf = sub.add_parser("fetch", help="fetch, validate and write a snapshot")
    sf.add_argument("--out", required=True)
    sf.set_defaults(func=_cmd_fetch)

    sm = sub.add_parser("members", help="constituents as at a date")
    sm.add_argument("date")
    sm.add_argument("--snapshot")
    sm.add_argument("--dir")
    sm.set_defaults(func=_cmd_members)

    sc = sub.add_parser("check", help="validate a stored snapshot")
    sc.add_argument("--snapshot")
    sc.add_argument("--dir")
    sc.set_defaults(func=_cmd_check)

    args = p.parse_args(argv)
    if not getattr(args, "source", None):
        args.source = list(DEFAULT_SOURCES)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
