"""Validate a Kibot GLD/GDX minute-bar delivery before it is used anywhere.

Establishes that the delivered files are what we think they are: schema,
timestamp integrity, quote sanity, session coverage, and -- the number the
Kibot purchase was actually for -- the realised bid/ask spread in bps.

Exits non-zero if any check FAILs.

    python -m engines.chan_cpo.validate_kibot_data --inspect --inspect-lines 5 \
        --gld data/external/kibot/GLD_1m_bidask_adj.txt \
        --gdx data/external/kibot/GDX_1m_bidask_adj.txt

    python -m engines.chan_cpo.validate_kibot_data -vv \
        --gld data/external/kibot/GLD_1m_bidask_adj.txt \
        --gdx data/external/kibot/GDX_1m_bidask_adj.txt \
        --expect-start 2009-01-01 --expect-end 2026-06-30
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from engines.chan_cpo.data_loader import (
    DEFAULT_BIDASK_SCHEMA,
    DEFAULT_OHLCV_SCHEMA,
    KibotSchema,
    KibotSchemaError,
    inspect_file,
    load_kibot,
)

RTH_OPEN = pd.Timestamp("09:30").time()
RTH_CLOSE = pd.Timestamp("15:59").time()
FULL_RTH_BARS = 390
OVERNIGHT_JUMP_PCT = 10.0


@dataclass
class Report:
    """Accumulates PASS/FAIL/WARN lines for one symbol or the pair."""

    label: str
    checks: list[tuple[str, str, str]] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    def check(self, status: str, name: str, detail: str = "") -> None:
        self.checks.append((status, name, detail))

    def ok(self, name: str, detail: str = "") -> None:
        self.check("PASS", name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.check("FAIL", name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.check("WARN", name, detail)

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for s, _, _ in self.checks)

    def render(self, verbose: int = 0) -> str:
        lines = [f"--- {self.label} ---"]
        for key, value in self.stats.items():
            lines.append(f"    {key:.<38} {value}")
        for status, name, detail in self.checks:
            if status == "PASS" and verbose < 1:
                continue
            lines.append(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
        return "\n".join(lines)


def _ohlc_violations(df: pd.DataFrame, o: str, h: str, lo_col: str, c: str) -> int:
    hi = df[h].to_numpy()
    lo = df[lo_col].to_numpy()
    op = df[o].to_numpy()
    cl = df[c].to_numpy()
    bad = (hi < lo) | (hi < np.maximum(op, cl)) | (lo > np.minimum(op, cl))
    return int(bad.sum())


def _session_stats(df: pd.DataFrame, rep: Report) -> pd.DataFrame:
    tod = df.index.time
    rth_mask = (tod >= RTH_OPEN) & (tod <= RTH_CLOSE)
    rth = df[rth_mask]

    rep.stats["total bars"] = f"{len(df):,}"
    rep.stats["first timestamp (ET)"] = str(df.index[0])
    rep.stats["last timestamp (ET)"] = str(df.index[-1])
    rep.stats["earliest time-of-day"] = str(min(tod))
    rep.stats["latest time-of-day"] = str(max(tod))
    rep.stats["RTH bars (09:30-15:59)"] = f"{len(rth):,}"
    rep.stats["extended-hours bars"] = f"{len(df) - len(rth):,}"

    if rth.empty:
        rep.fail("rth bars present", "no bars inside 09:30-15:59")
        return rth

    per_day = rth.groupby(rth.index.normalize()).size()
    rep.stats["session days (with RTH bars)"] = f"{len(per_day):,}"
    rep.stats["median RTH bars/day"] = f"{per_day.median():.0f} / {FULL_RTH_BARS}"
    rep.stats["mean RTH bars/day"] = f"{per_day.mean():.1f}"
    rep.stats["days at full 390 bars"] = (
        f"{int((per_day == FULL_RTH_BARS).sum()):,} "
        f"({100.0 * (per_day == FULL_RTH_BARS).mean():.1f}%)"
    )

    if min(tod) < RTH_OPEN or max(tod) > RTH_CLOSE:
        rep.warn(
            "extended hours present",
            f"bars outside RTH from {min(tod)} to {max(tod)} "
            f"({len(df) - len(rth):,} bars, "
            f"{100.0 * (len(df) - len(rth)) / len(df):.1f}%)",
        )
    else:
        rep.ok("session bounded to RTH")

    # Intraday gaps inside RTH, measured within each session day.
    deltas = rth.index.to_series().diff()
    same_day = rth.index.normalize().to_series().diff() == pd.Timedelta(0)
    gaps = deltas[(deltas > pd.Timedelta(minutes=1)) & same_day.to_numpy()]
    rep.stats["intraday RTH gaps (>1min)"] = f"{len(gaps):,}"
    if len(gaps):
        rep.stats["largest intraday gap"] = str(gaps.max())
        rep.warn(
            "intraday gaps",
            f"{len(gaps):,} gaps, largest {gaps.max()}, median {gaps.median()}",
        )
    else:
        rep.ok("no intraday RTH gaps")
    return rth


def _overnight_jumps(rth: pd.DataFrame, price_col: str, rep: Report) -> None:
    if rth.empty:
        return
    daily = rth.groupby(rth.index.normalize())[price_col].agg(["first", "last"])
    prev_close = daily["last"].shift(1)
    jump_pct = 100.0 * (daily["first"] - prev_close).abs() / prev_close
    flagged = jump_pct[jump_pct > OVERNIGHT_JUMP_PCT].dropna()
    rep.stats[f"overnight jumps >{OVERNIGHT_JUMP_PCT:.0f}%"] = f"{len(flagged):,}"
    if len(flagged):
        worst = flagged.sort_values(ascending=False).head(5)
        detail = ", ".join(f"{d.date()} {v:.1f}%" for d, v in worst.items())
        rep.warn("overnight jumps (possible unhandled corporate action)", detail)
    else:
        rep.ok("no overnight jumps beyond threshold")


QUOTE_PRICE_COLS = (
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
)


def _nonpositive_check(df: pd.DataFrame, cols: tuple[str, ...], rep: Report) -> None:
    """Zero/negative prices are unambiguous vendor corruption, not thin quotes."""
    bad_mask = (df[list(cols)] <= 0).any(axis=1)
    bad = df[bad_mask]
    rep.stats["bars with non-positive price"] = f"{len(bad):,}"
    if not len(bad):
        rep.ok("all prices positive")
        return
    tod = bad.index.time
    in_rth = int(((tod >= RTH_OPEN) & (tod <= RTH_CLOSE)).sum())
    detail = (
        f"{len(bad):,} bars ({in_rth:,} inside RTH); "
        f"times-of-day {sorted({str(t) for t in tod})[:4]}"
    )
    if in_rth:
        rep.fail("all prices positive", detail)
    else:
        rep.warn("non-positive prices confined to extended hours", detail)


def _quote_checks(
    df: pd.DataFrame, rth: pd.DataFrame, rep: Report, allow_crossed: int
) -> None:
    _nonpositive_check(df, QUOTE_PRICE_COLS, rep)

    crossed_close = int((df["bid_close"] > df["ask_close"]).sum())
    crossed_hard_mask = df["bid_low"] > df["ask_high"]
    crossed_hard = int(crossed_hard_mask.sum())
    hard_tod = df.index[crossed_hard_mask].time
    hard_rth = int(((hard_tod >= RTH_OPEN) & (hard_tod <= RTH_CLOSE)).sum())

    rep.stats["crossed at close (bid_c>ask_c)"] = (
        f"{crossed_close:,} ({1e4 * crossed_close / len(df):.1f} per 10k bars)"
    )
    rep.stats["crossed whole bar (bid_l>ask_h)"] = (
        f"{crossed_hard:,} ({hard_rth:,} inside RTH, "
        f"{100.0 * crossed_hard / len(df):.4f}% of bars)"
    )

    if crossed_hard > allow_crossed:
        rep.fail(
            "no unambiguously crossed bars",
            f"{crossed_hard:,} bars where bid_low > ask_high "
            f"(tolerance {allow_crossed:,}); {hard_rth:,} of them inside RTH",
        )
    elif crossed_hard:
        rep.warn(
            "crossed bars within tolerance",
            f"{crossed_hard:,} <= --allow-crossed {allow_crossed:,}",
        )
    else:
        rep.ok("no unambiguously crossed bars")

    if crossed_close:
        rth_crossed = int((rth["bid_close"] > rth["ask_close"]).sum()) if len(rth) else 0
        rep.warn(
            "crossed quotes at bar close",
            f"{crossed_close:,} all-hours ({rth_crossed:,} inside RTH); bar-close "
            "bid and ask are aggregated independently, so a crossed close is a "
            "sampling artefact rather than a locked market",
        )
    else:
        rep.ok("no crossed quotes at bar close")

    for tag, quartet in (
        ("bid", ("bid_open", "bid_high", "bid_low", "bid_close")),
        ("ask", ("ask_open", "ask_high", "ask_low", "ask_close")),
    ):
        bad = _ohlc_violations(df, *quartet)
        if bad:
            rep.fail(f"{tag} OHLC sanity", f"{bad:,} bars violate high/low bounds")
        else:
            rep.ok(f"{tag} OHLC sanity")


def _spread_stats(df: pd.DataFrame, rth: pd.DataFrame, rep: Report) -> None:
    for tag, frame in (("all-hours", df), ("RTH", rth)):
        if frame.empty:
            continue
        s = frame["spread_bps"]
        s = s[np.isfinite(s)]
        rep.stats[f"spread bps [{tag}] median"] = f"{s.median():.2f}"
        rep.stats[f"spread bps [{tag}] p95"] = f"{s.quantile(0.95):.2f}"
        if tag == "RTH":
            rep.stats["spread bps [RTH] p05"] = f"{s.quantile(0.05):.2f}"
            rep.stats["spread bps [RTH] p99"] = f"{s.quantile(0.99):.2f}"
            rep.stats["spread bps [RTH] mean"] = f"{s.mean():.2f}"

    if rth.empty:
        return
    # A locked bar (bid == ask) is never actually crossable, so leaving it in
    # biases the median cost down. Report the tradeable number alongside.
    locked = int((rth["spread"].abs() < 1e-9).sum())
    # Cost-facing, so read the floored column: a crossed bar must never
    # contribute a negative (i.e. paid-to-trade) spread to a cost figure.
    tradeable = rth["cost_spread_bps"]
    tradeable = tradeable[np.isfinite(tradeable) & (tradeable > 0)]
    rep.stats["locked RTH bars (bid == ask)"] = (
        f"{locked:,} ({100.0 * locked / len(rth):.2f}% of RTH)"
    )
    rep.stats["spread bps [RTH, unlocked] median"] = f"{tradeable.median():.2f}"
    rep.stats["spread bps [RTH, unlocked] p95"] = f"{tradeable.quantile(0.95):.2f}"
    rep.stats["one-way half-spread (median bps)"] = f"{tradeable.median() / 2:.2f}"


def _coverage(df: pd.DataFrame, rep: Report, start: str | None, end: str | None) -> None:
    if start:
        want = pd.Timestamp(start)
        got = df.index[0]
        if got.normalize() > want:
            rep.fail(
                "coverage start",
                f"expected data from {want.date()} but earliest bar is {got} "
                f"({(got.normalize() - want).days} days short)",
            )
        else:
            rep.ok("coverage start", f"{got}")
    if end:
        want = pd.Timestamp(end)
        got = df.index[-1]
        if got.normalize() < want:
            rep.fail(
                "coverage end",
                f"expected data through {want.date()} but latest bar is {got}",
            )
        else:
            rep.ok("coverage end", f"{got}")


def _timestamp_checks(df: pd.DataFrame, rep: Report) -> None:
    idx = df.index
    dupes = int(idx.duplicated().sum())
    if dupes:
        rep.fail("unique timestamps", f"{dupes:,} duplicate timestamps")
    else:
        rep.ok("unique timestamps")
    if not idx.is_monotonic_increasing:
        n_back = int((idx.to_series().diff() < pd.Timedelta(0)).sum())
        rep.fail("monotonic timestamps", f"{n_back:,} backward steps")
    else:
        rep.ok("monotonic timestamps")


def validate_symbol(
    label: str,
    path: Path,
    schema: KibotSchema,
    expect_start: str | None,
    expect_end: str | None,
    allow_crossed: int = 0,
) -> tuple[Report, pd.DataFrame]:
    rep = Report(f"{label}  ({path.name})")
    df = load_kibot(path, schema)
    rep.ok("schema", f"{schema.n_columns} columns as {schema.name}")

    _timestamp_checks(df, rep)
    rth = _session_stats(df, rep)
    _coverage(df, rep, expect_start, expect_end)

    if schema.name == "bidask":
        _quote_checks(df, rth, rep, allow_crossed)
        _spread_stats(df, rth, rep)
        _overnight_jumps(rth, "mid_close", rep)
    else:
        _nonpositive_check(df, ("open", "high", "low", "close"), rep)
        bad = _ohlc_violations(df, "open", "high", "low", "close")
        if bad:
            rep.fail("OHLC sanity", f"{bad:,} bars violate high/low bounds")
        else:
            rep.ok("OHLC sanity")
        neg = int((df["volume"] < 0).sum())
        if neg:
            rep.fail("volume non-negative", f"{neg:,} negative volumes")
        else:
            rep.ok("volume non-negative")
        rep.stats["total volume"] = f"{df['volume'].sum():,.0f}"
        _overnight_jumps(rth, "close", rep)
    return rep, df


def pair_report(gld: pd.DataFrame, gdx: pd.DataFrame) -> Report:
    rep = Report("PAIR ALIGNMENT  (GLD and GDX)")
    common = gld.index.intersection(gdx.index)
    rep.stats["GLD bars"] = f"{len(gld):,}"
    rep.stats["GDX bars"] = f"{len(gdx):,}"
    rep.stats["common bars"] = f"{len(common):,}"
    rep.stats["% of GLD retained"] = f"{100.0 * len(common) / len(gld):.2f}%"
    rep.stats["% of GDX retained"] = f"{100.0 * len(common) / len(gdx):.2f}%"
    if len(common):
        rep.stats["common first"] = str(common[0])
        rep.stats["common last"] = str(common[-1])
        rep.ok("pair overlap non-empty")
    else:
        rep.fail("pair overlap non-empty", "no shared timestamps")
        return rep

    tod = common.time
    rth = common[(tod >= RTH_OPEN) & (tod <= RTH_CLOSE)]
    rep.stats["common RTH bars"] = f"{len(rth):,}"
    if len(rth):
        per_day = pd.Series(1, index=rth).groupby(rth.normalize()).size()
        rep.stats["common session days"] = f"{len(per_day):,}"
        rep.stats["median common RTH bars/day"] = f"{per_day.median():.0f}"
    return rep


def run_inspect(paths: dict[str, Path], schema: KibotSchema, n_lines: int) -> int:
    print(
        f"=== INSPECT (expecting schema {schema.name}, "
        f"{schema.n_columns} columns) ==="
    )
    print(f"    {', '.join(schema.columns)}\n")
    rc = 0
    for label, path in paths.items():
        res = inspect_file(path, n_lines)
        print(f"--- {label}: {path} ---")
        for i, line in enumerate(res.raw_lines, 1):
            print(f"  {i}| {line}")
        counts = ", ".join(str(c) for c in res.column_counts)
        print(f"  observed column counts: [{counts}]")
        if res.n_columns is None:
            print("  [FAIL] ragged head -- column count is not constant")
            rc = 1
        elif res.matches(schema):
            print(f"  [PASS] {res.n_columns} columns matches schema {schema.name}")
        else:
            print(
                f"  [FAIL] {res.n_columns} columns but schema {schema.name} "
                f"expects {schema.n_columns}"
            )
            rc = 1
        print()
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gld", required=True, type=Path)
    ap.add_argument("--gdx", required=True, type=Path)
    ap.add_argument(
        "--ohlcv",
        action="store_true",
        help="file is the 7-column OHLCV layout, not the 10-column bid/ask one",
    )
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="print raw head lines and column counts, then exit",
    )
    ap.add_argument("--inspect-lines", type=int, default=5)
    ap.add_argument("--expect-start")
    ap.add_argument("--expect-end")
    ap.add_argument(
        "--allow-crossed",
        type=int,
        default=0,
        help="tolerated count of bid_low>ask_high bars before the check FAILs; "
             "use a documented non-zero value rather than ignoring the check",
    )
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args(argv)

    schema = DEFAULT_OHLCV_SCHEMA if args.ohlcv else DEFAULT_BIDASK_SCHEMA
    paths = {"GLD": args.gld, "GDX": args.gdx}

    if args.inspect:
        return run_inspect(paths, schema, args.inspect_lines)

    reports: list[Report] = []
    frames: dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        try:
            rep, df = validate_symbol(
                label,
                path,
                schema,
                args.expect_start,
                args.expect_end,
                allow_crossed=args.allow_crossed,
            )
        except KibotSchemaError as exc:
            rep = Report(f"{label}  ({path.name})")
            rep.fail("schema", str(exc))
            reports.append(rep)
            continue
        reports.append(rep)
        frames[label] = df

    if len(frames) == 2:
        reports.append(pair_report(frames["GLD"], frames["GDX"]))

    print(f"=== VALIDATION: schema {schema.name} ({schema.n_columns} columns) ===\n")
    for rep in reports:
        print(rep.render(args.verbose))
        print()

    failed = [r for r in reports if r.failed]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(reports)} sections)")
        return 1
    print(f"RESULT: PASS ({len(reports)} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
