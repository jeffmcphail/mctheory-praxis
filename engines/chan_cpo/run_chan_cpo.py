"""Chan CPO Layer A runner -- TRAIN grid, freeze, TEST gross, benchmarks.

    python -m engines.chan_cpo.run_chan_cpo                      # primary run
    python -m engines.chan_cpo.run_chan_cpo --validate           # checks only
    python -m engines.chan_cpo.run_chan_cpo --sweep              # A1 x A2 x A4

Everything the prereg leaves as a choice is a flag; nothing is hard-coded.
The reporting ORDER is fixed in code, not by convention: round trips per day
print before any Sharpe, because at 1.26 bps per round trip the trade count is
what decides the experiment (prereg section 6).
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .leakage_checks import run_all
from .metrics import cost_drag_bps
from .signal import (
    GRID_ENTRY_THRESHOLDS, GRID_GDX_WEIGHTS, GRID_LOOKBACKS, SESSION_END,
    SESSION_START, Ambiguities, Combination, Engineering, panel_path,
    prepare_bars,
)
from .unconditional import Window, build_grid, run_grid

# Prereg section 2.  UNCONDITIONAL row only; the conditional row is Phase 5.
TARGET = {
    "annual_return": 0.1729,
    "sharpe": 1.947,
    "calmar": 0.984,
    "cumulative_3y": 0.73,
}
TOLERANCE = {"sharpe": 0.2, "annual_return": 0.02}   # prereg section 9

TRAIN_START, TRAIN_END = "2009-09-28", "2017-12-31"
TEST_START, TEST_END = "2018-01-01", "2020-12-31"

OUT_DIR = Path("outputs/chan_cpo_layer_a")


def _pct(x: float) -> str:
    return "     n/a" if not np.isfinite(x) else f"{100.0 * x:7.2f}%"


def _num(x: float, w: int = 7, p: int = 3) -> str:
    return " " * (w - 3) + "n/a" if not np.isfinite(x) else f"{x:{w}.{p}f}"


def _rule(char: str = "=", n: int = 78) -> str:
    return char * n


def parse_time(text: str) -> dt.time:
    hh, mm = text.split(":")
    return dt.time(int(hh), int(mm))


def parse_floats(text: str | None, default: tuple) -> tuple:
    if not text:
        return default
    return tuple(float(v) for v in text.replace(" ", "").split(","))


def parse_ints(text: str | None, default: tuple) -> tuple:
    if not text:
        return default
    return tuple(int(v) for v in text.replace(" ", "").split(","))


# --------------------------------------------------------------------------
# reporting -- the order below is mandatory (brief, "Reporting order")
# --------------------------------------------------------------------------

def report_round_trips(result, verbose: int) -> None:
    """ITEM 1.  Before any Sharpe.  Always."""
    rt = result.test.round_trips
    drag = cost_drag_bps(rt["mean_per_day"])
    print(_rule())
    print("1. ROUND TRIPS PER DAY  (TEST window, reported BEFORE any Sharpe)")
    print(_rule())
    print(f"  mean per session      {rt['mean_per_day']:10.3f}")
    print(f"  median per session    {rt['median_per_day']:10.3f}")
    if verbose >= 2:
        print(f"  p95 per session       {rt['p95_per_day']:10.3f}")
        print(f"  max in one session    {rt['max_per_day']:10.0f}")
        print(f"  total round trips     {rt['total_round_trips']:10,d}")
        print(f"  sessions              {rt['sessions']:10,d}")
        print(f"  sessions with a trade {rt['sessions_with_a_trade']:10,d}"
              f"   ({100 * rt['share_of_sessions_traded']:.1f}%)")
        print(f"  mean per ACTIVE day   {rt['mean_per_active_day']:10.3f}")
    tr_econ = result.test.trips
    print()
    print(f"  mean holding period   {tr_econ['mean_hold_bars']:10.2f} bars "
          f"(median {tr_econ['median_hold_bars']:.0f})")
    print(f"  GROSS bps per round trip:  mean {tr_econ['mean_gross_bps']:.3f}  "
          f"median {tr_econ['median_gross_bps']:.3f}  "
          f"win rate {100 * tr_econ['win_rate']:.1f}%")
    print()
    print(f"  Implied cost drag at the pre-registered {drag['bps_per_round_trip']:.2f} "
          f"bps/round trip:")
    print(f"    {drag['bps_per_day']:.2f} bps/session  ->  "
          f"{100 * drag['annualised_drag']:.2f}% annualised")
    print(f"    breakeven cost is {tr_econ['breakeven_cost_bps_per_round_trip']:.3f} "
          f"bps/round trip; the prereg assumes "
          f"{drag['bps_per_round_trip']:.2f}.")
    print("    (Phase 4 owns the overlay; this is the arithmetic prereg section 6")
    print("     requires be shown next to the trade count, not a net result.)")
    print()


def report_metrics(result, verbose: int) -> None:
    """ITEM 2.  Gross TEST metrics, each against its pre-registered target."""
    p = result.test.performance
    print(_rule())
    print("2. GROSS METRICS ON TEST vs PRE-REGISTERED TARGET")
    print(_rule())
    print(f"  {'metric':22s} {'target':>10s} {'actual':>10s} {'verdict':>12s}")

    def line(name, target, actual, tol=None, fmt=_num):
        if tol is None:
            verdict = ""
        elif not np.isfinite(actual):
            verdict = "NO RESULT"
        else:
            verdict = "HIT" if abs(actual - target) <= tol else "MISS"
        print(f"  {name:22s} {fmt(target):>10s} {fmt(actual):>10s} {verdict:>12s}")

    line("Sharpe", TARGET["sharpe"], p.sharpe, TOLERANCE["sharpe"])
    line("annual return (arith)", TARGET["annual_return"], p.annual_arithmetic,
         TOLERANCE["annual_return"], _pct)
    line("annual return (CAGR)", TARGET["annual_return"], p.annual_cagr,
         TOLERANCE["annual_return"], _pct)
    line("3-year cumulative", TARGET["cumulative_3y"], p.cumulative, None, _pct)
    line("Calmar (CAGR/maxDD)", TARGET["calmar"], p.calmar_cagr, None)
    print()
    if verbose >= 2:
        print(f"  annualised vol        {_pct(p.vol_annual)}")
        print(f"  max drawdown          {_pct(p.max_drawdown)}")
        print(f"  hit rate (traded days){_pct(p.hit_rate)}")
        print(f"  best / worst session  {_pct(p.best_day)} / {_pct(p.worst_day)}")
        print(f"  sessions scored       {p.n_days:8,d}")
        print()
    print("  NOTE ON THE TARGET TABLE ITSELF: 17.29% annual and 73% three-year")
    print("  cumulative are mutually inconsistent -- compounding 17.29% for three")
    print("  years gives 61.4%, and 73% implies a 20.07% CAGR. Geometric can never")
    print("  exceed arithmetic on one series, so both published figures cannot be")
    print("  right. Both annualisations are shown rather than picking one.")
    print()


def report_winner(result, verbose: int) -> None:
    """ITEM 3.  The frozen combination and the settings that produced it."""
    w = result.winner
    tr = result.train
    print(_rule())
    print("3. FROZEN COMBINATION  (selected on TRAIN, applied unchanged to TEST)")
    print(_rule())
    print(f"  GDX_weight            {w.gdx_weight:10.2f}")
    print(f"  entry_threshold       {w.entry_threshold:10.2f}"
          f"   (exit {-0.6 * w.entry_threshold:+.2f})")
    print(f"  lookback              {w.lookback:10d} min")
    print()
    print(f"  ambiguities           {result.ambiguities.label()}")
    print(f"  engineering           {result.engineering.label()}")
    print()
    print(f"  TRAIN {tr.window.start.date()}..{tr.window.end.date()}  "
          f"cumulative {_pct(tr.performance.cumulative)}  "
          f"Sharpe {_num(tr.performance.sharpe)}  "
          f"rt/day {tr.round_trips['mean_per_day']:.2f}")
    if verbose >= 2:
        top = result.table.nlargest(5, "train_cumulative")
        print("\n  top 5 of the TRAIN grid by the selection objective:")
        print(f"    {'w':>5s} {'entry':>6s} {'lb':>5s} {'train cum':>12s} "
              f"{'train SR':>9s} {'test SR':>9s} {'test rt/d':>10s}")
        for _, r in top.iterrows():
            print(f"    {r['gdx_weight']:5.1f} {r['entry_threshold']:6.2f} "
                  f"{int(r['lookback']):5d} {_pct(r['train_cumulative']):>12s} "
                  f"{_num(r['train_sharpe'], 9):>9s} {_num(r['test_sharpe'], 9):>9s} "
                  f"{r['test_round_trips_per_day']:10.2f}")
    print()


def report_benchmarks(result, verbose: int) -> None:
    """ITEM 4.  Prereg section 7 -- the strategy is UNHEDGED."""
    b = result.test.benchmarks
    p = result.test.performance
    print(_rule())
    print("4. BENCHMARKS (prereg section 7 -- the strategy is UNHEDGED)")
    print(_rule())
    print(f"  {'series':28s} {'total':>10s} {'annual':>10s} {'Sharpe':>9s} {'maxDD':>9s}")
    for name, perf, total in (
        ("strategy (gross)", p, p.cumulative),
        ("a. buy-and-hold GLD", b.buy_and_hold, b.buy_and_hold_total),
        ("b. intraday-only long GLD", b.intraday_long, b.intraday_long_total),
    ):
        print(f"  {name:28s} {_pct(total):>10s} {_pct(perf.annual_arithmetic):>10s} "
              f"{_num(perf.sharpe, 9):>9s} {_pct(perf.max_drawdown):>9s}")
    print()
    print("  c. realised exposure of the strategy")
    print(f"     time in market                {_pct(b.time_in_market)}")
    print(f"     of in-market bar-time, long   {_pct(b.long_bar_share)}")
    print(f"     of in-market bar-time, short  {_pct(b.short_bar_share)}")
    print(f"     of round trips, long          {_pct(b.long_trip_share)}")
    print(f"     corr(daily strategy, daily b) {_num(b.correlation_to_intraday_long)}")
    print()
    verdict = "BEATS" if b.beats_intraday_long else "DOES NOT BEAT"
    print(f"  VERDICT vs the like-for-like control (b): strategy {verdict} it.")
    if not b.beats_intraday_long:
        print("  Per the prereg, a strategy that does not beat (b) is harvesting")
        print("  intraday gold drift, not conditional mean reversion -- whatever")
        print("  the Sharpe says. 2018-2020 is a major gold rally.")
    print()


def report(result, verbose: int) -> None:
    report_round_trips(result, verbose)
    report_metrics(result, verbose)
    report_winner(result, verbose)
    report_benchmarks(result, verbose)
    for w in result.warnings:
        print(f"  [WARN] {w}")


# --------------------------------------------------------------------------

def load_panel(args, verbose: int) -> pd.DataFrame:
    path = args.panel or panel_path(args.adjustment, args.data_dir)
    if not Path(path).exists():
        raise SystemExit(
            f"panel not found: {path}\n"
            f"build it first:  python -m engines.chan_cpo.build_panel "
            f"--adjustment {args.adjustment}"
        )
    if verbose >= 1:
        print(f"loading {path}")
    cols = ["gld_trade_close", "gdx_trade_close", "gld_mid_close", "gdx_mid_close"]
    return pd.read_parquet(path, columns=cols)


def run_one(panel: pd.DataFrame, args, ambiguities: Ambiguities,
            engineering: Engineering, verbose: int, checks: bool = True):
    train = Window("TRAIN", pd.Timestamp(args.train_start), pd.Timestamp(args.train_end))
    test = Window("TEST", pd.Timestamp(args.test_start), pd.Timestamp(args.test_end))
    session_start, session_end = parse_time(args.session_start), parse_time(args.session_end)

    bars = prepare_bars(panel, start=panel.index[0],
                        end=pd.Timestamp(args.test_end) + pd.Timedelta(days=1),
                        session_start=session_start, session_end=session_end,
                        engineering=engineering)
    if verbose >= 1:
        print(f"  bars {bars.index[0]} .. {bars.index[-1]}  "
              f"({bars.n_signal_bars:,} signal bars, "
              f"{bars.session_days.size:,} sessions, "
              f"{bars.dropped_price_bars} price-gap bars dropped)")

    combos = build_grid(
        parse_floats(args.gdx_weights, GRID_GDX_WEIGHTS),
        parse_floats(args.entry_thresholds, GRID_ENTRY_THRESHOLDS),
        parse_ints(args.lookbacks, GRID_LOOKBACKS),
    )

    t0 = time.time()
    progress = None
    if verbose >= 2:
        def progress(i, n, combo):
            if i % 50 == 0 or i == n:
                print(f"    grid {i:4d}/{n}  ({time.time() - t0:5.1f}s)")
    result = run_grid(bars, train, test, ambiguities, engineering,
                      combos=combos, progress=progress)
    if verbose >= 1:
        print(f"  grid of {len(combos)} combinations in {time.time() - t0:.1f}s")

    if checks:
        checks_out = run_all(bars, panel, train, test, ambiguities, engineering,
                             result.winner, session_start, session_end,
                             combos=combos, include_selection=True)
        print_checks(checks_out, verbose)
        if any(not ok for _, ok, _ in checks_out):
            raise SystemExit("LEAKAGE CHECKS FAILED -- no metric will be reported.")
    return result, bars, train, test


def print_checks(checks, verbose: int) -> None:
    print(_rule())
    print("0. LEAKAGE / CAUSALITY CHECKS  (all must pass before any metric)")
    print(_rule())
    for name, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name:24s} {detail}")
    print()


def write_artifacts(result, out_dir: Path, tag: str, verbose: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out_dir / f"grid_{tag}.csv", index=False)
    pd.DataFrame({
        "ret": result.test.daily,
        "round_trips": result.test.trades_per_day,
    }).to_csv(out_dir / f"test_daily_{tag}.csv")

    summary = {
        "phase": "cycle59_phase3_layer_a",
        "ambiguities": result.ambiguities.__dict__,
        "engineering": result.engineering.__dict__,
        "winner": result.winner.as_dict(),
        "targets": TARGET,
        "test_round_trips": result.test.round_trips,
        "test_trip_economics": result.test.trips,
        "test_performance": result.test.performance.as_dict(),
        "train_performance": result.train.performance.as_dict(),
        "benchmarks": result.test.benchmarks.as_dict(),
        "warnings": result.warnings,
    }
    (out_dir / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    if verbose >= 1:
        print(f"  artifacts -> {out_dir}/*_{tag}.*")


def sweep(panel: pd.DataFrame, args, verbose: int) -> list[dict]:
    """A1 x A2 x A4 -- the legitimate iteration surface (prereg section 8)."""
    rows = []
    grid = list(itertools.product(["var", "std"], ["simple", "log"], ["unadj", "adj"]))
    for i, (den, mode, adj) in enumerate(grid, 1):
        amb = Ambiguities(zscore_denominator=den, return_mode=mode, adjustment=adj)
        eng = Engineering(price_source=args.price_source,
                          bar_universe=args.bar_universe,
                          ewma_reset=args.ewma_reset,
                          flat_days=args.flat_days,
                          execution_lag=args.execution_lag)
        print(f"\n{_rule('-')}\nSWEEP {i}/{len(grid)}  {amb.label()}\n{_rule('-')}")
        sub = panel
        if adj != args.adjustment:
            sub = load_panel_for(args, adj, verbose)
        res, *_ = run_one(sub, args, amb, eng, verbose, checks=False)
        p = res.test.performance
        rows.append({
            "A1_zscore_denominator": den, "A2_return_mode": mode,
            "A4_adjustment": adj, **res.winner.as_dict(),
            "test_round_trips_per_day": res.test.round_trips["mean_per_day"],
            "test_sharpe": p.sharpe, "test_annual_arith": p.annual_arithmetic,
            "test_annual_cagr": p.annual_cagr, "test_cumulative": p.cumulative,
            "test_calmar": p.calmar_cagr,
            "test_gross_bps_per_round_trip": res.test.trips["mean_gross_bps"],
            "test_mean_hold_bars": res.test.trips["mean_hold_bars"],
            "beats_intraday_long": res.test.benchmarks.beats_intraday_long,
        })
        write_artifacts(res, Path(args.out_dir), f"{den}_{mode}_{adj}", verbose)
    return rows


_PANEL_CACHE: dict[str, pd.DataFrame] = {}


def load_panel_for(args, adjustment: str, verbose: int) -> pd.DataFrame:
    if adjustment not in _PANEL_CACHE:
        stashed, args.adjustment = args.adjustment, adjustment
        try:
            _PANEL_CACHE[adjustment] = load_panel(args, verbose)
        finally:
            args.adjustment = stashed
    return _PANEL_CACHE[adjustment]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_chan_cpo",
        description="Chan CPO Layer A (unconditional) replication -- GROSS.")

    a = p.add_argument_group("pre-registered ambiguities (prereg section 8)")
    a.add_argument("--zscore-denominator", choices=["var", "std"], default="var",
                   help="A1: the paper divides by VAR; conventional Bollinger by STD")
    a.add_argument("--return-mode", choices=["simple", "log"], default="simple",
                   help="A2: round-trip aggregation")
    a.add_argument("--adjustment", choices=["adj", "unadj"], default="unadj",
                   help="A4: unadjusted = actual traded prices")
    a.add_argument("--eighth-indicator", default="n/a",
                   help="A3: Layer B only; recorded for the run log")
    a.add_argument("--sweep", action="store_true",
                   help="run all 8 A1 x A2 x A4 configurations and tabulate")

    g = p.add_argument_group("grid (prereg section 3)")
    g.add_argument("--gdx-weights", default=None)
    g.add_argument("--entry-thresholds", default=None)
    g.add_argument("--lookbacks", default=None)

    w = p.add_argument_group("windows (prereg section 5 -- do not alter)")
    w.add_argument("--train-start", default=TRAIN_START)
    w.add_argument("--train-end", default=TRAIN_END)
    w.add_argument("--test-start", default=TEST_START)
    w.add_argument("--test-end", default=TEST_END)
    w.add_argument("--session-start", default=SESSION_START.strftime("%H:%M"))
    w.add_argument("--session-end", default=SESSION_END.strftime("%H:%M"))

    e = p.add_argument_group(
        "engineering decisions the prereg does not specify -- exposed, NOT tuned")
    e.add_argument("--price-source", choices=["trade", "mid"], default="trade")
    e.add_argument("--bar-universe", choices=["rth", "all"], default="rth")
    e.add_argument("--ewma-reset", choices=["none", "daily"], default="none")
    e.add_argument("--flat-days", choices=["include", "exclude"], default="include")
    e.add_argument("--execution-lag", type=int, default=0,
                   help="bars between the deciding close and the fill; 0 is what "
                        "the prereg specifies, 1 is the microstructure diagnostic")

    p.add_argument("--panel", default=None, help="explicit parquet path")
    p.add_argument("--data-dir", default="data/external/kibot")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--validate", action="store_true",
                   help="run the leakage/causality checks on a reduced grid and exit")
    p.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=3,
                   help="0=quiet .. 3=MAXIMUM (default)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    verbose = args.verbose

    print(_rule())
    print("CYCLE 59 PHASE 3 -- Chan CPO Layer A (unconditional), GROSS")
    print("governing document: claude/handoffs/CYCLE59_CHAN_CPO_PREREGISTRATION.md")
    print(_rule())

    panel = load_panel_for(args, args.adjustment, verbose)

    if args.validate:
        amb = Ambiguities(args.zscore_denominator, args.return_mode,
                          args.adjustment, args.eighth_indicator)
        eng = Engineering(args.price_source, args.bar_universe,
                          args.ewma_reset, args.flat_days, args.execution_lag)
        train = Window("TRAIN", pd.Timestamp(args.train_start), pd.Timestamp(args.train_end))
        test = Window("TEST", pd.Timestamp(args.test_start), pd.Timestamp(args.test_end))
        ss, se = parse_time(args.session_start), parse_time(args.session_end)
        bars = prepare_bars(panel, start=panel.index[0],
                            end=pd.Timestamp(args.test_end) + pd.Timedelta(days=1),
                            session_start=ss, session_end=se, engineering=eng)
        small = build_grid((2.0, 3.0), (0.5, 1.5), (60, 240))
        from .unconditional import select_winner_only
        winner = select_winner_only(bars, train, amb, eng, combos=small)
        checks = run_all(bars, panel, train, test, amb, eng, winner, ss, se,
                         combos=small, include_selection=True)
        print_checks(checks, verbose)
        failed = [n for n, ok, _ in checks if not ok]
        print(f"RESULT: {'FAIL ' + ', '.join(failed) if failed else 'PASS'} "
              f"({len(checks)} checks)")
        return 1 if failed else 0

    if args.sweep:
        rows = sweep(panel, args, verbose)
        table = pd.DataFrame(rows)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "ambiguity_sweep.csv", index=False)
        print(f"\n{_rule()}\nAMBIGUITY SWEEP A1 x A2 x A4 (prereg section 8)\n{_rule()}")
        print(f"  target: Sharpe {TARGET['sharpe']:.3f} +/- {TOLERANCE['sharpe']}, "
              f"annual {100 * TARGET['annual_return']:.2f}% +/- 2pp\n")
        print(f"  {'A1':>5s} {'A2':>7s} {'A4':>6s} {'w':>5s} {'entry':>6s} {'lb':>5s} "
              f"{'rt/day':>8s} {'Sharpe':>8s} {'annual':>9s} {'cum':>9s} {'hit?':>5s} "
              f"{'gross/rt':>7s}")
        for r in rows:
            hit = ("HIT" if abs(r["test_sharpe"] - TARGET["sharpe"]) <= TOLERANCE["sharpe"]
                   else "miss")
            print(f"  {r['A1_zscore_denominator']:>5s} {r['A2_return_mode']:>7s} "
                  f"{r['A4_adjustment']:>6s} {r['gdx_weight']:5.1f} "
                  f"{r['entry_threshold']:6.2f} {r['lookback']:5d} "
                  f"{r['test_round_trips_per_day']:8.2f} "
                  f"{_num(r['test_sharpe'], 8):>8s} "
                  f"{_pct(r['test_annual_arith']):>9s} "
                  f"{_pct(r['test_cumulative']):>9s} {hit:>5s} "
                  f"{r['test_gross_bps_per_round_trip']:7.3f}")
        print(f"\n  wrote {out / 'ambiguity_sweep.csv'}")
        return 0

    amb = Ambiguities(args.zscore_denominator, args.return_mode,
                      args.adjustment, args.eighth_indicator)
    eng = Engineering(args.price_source, args.bar_universe,
                      args.ewma_reset, args.flat_days, args.execution_lag)
    print(f"\nPRIMARY configuration: {amb.label()} | {eng.label()}\n")
    result, *_ = run_one(panel, args, amb, eng, verbose, checks=True)
    report(result, verbose)
    tag = f"{amb.zscore_denominator}_{amb.return_mode}_{amb.adjustment}"
    if (eng.price_source, eng.execution_lag) != ("trade", 0):
        tag += f"_{eng.price_source}_lag{eng.execution_lag}"
    write_artifacts(result, Path(args.out_dir), tag, verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
