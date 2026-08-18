"""Chan CPO Layer A -- the unconditional grid: search TRAIN, freeze, apply TEST.

Layer A is the diagnostic that matters (prereg section 9): it exercises the
data path, the backtest engine and the strategy logic with ZERO machine
learning.  If the published gross Sharpe cannot be reproduced here after
exhausting A1-A4, the framework-is-broken hypothesis is confirmed.

Selection discipline
--------------------
Signals are computed once over the continuous bar series so the EWMA reaches
the TEST window already warmed -- that is causal (an IIR filter at bar t sees
only bars <= t) and is what a live deployment would do.  Selection then slices
TRAIN SESSIONS ONLY.  Because the recursion is causal, a combination's TRAIN
daily series is bit-identical whether or not TEST bars were appended, and
`leakage_checks.check_selection_ignores_test` asserts exactly that rather than
leaving it as an argument.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import (
    Benchmarks, Performance, benchmarks, performance, round_trip_stats,
    trip_economics,
)
from .signal import (
    Ambiguities, Bars, Combination, Engineering, Simulation,
    GRID_ENTRY_THRESHOLDS, GRID_GDX_WEIGHTS, GRID_LOOKBACKS, simulate,
)

__all__ = [
    "Window", "WindowResult", "GridResult", "build_grid", "evaluate",
    "attach_benchmarks", "cumulative_train_return", "run_grid",
    "select_winner_only",
]


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def mask(self, sessions: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((sessions >= self.start) & (sessions <= self.end))

    @property
    def codes(self) -> tuple[int, int]:
        """Window bounds as int64 ns, to compare against `Bars.day_code`."""
        return (int(np.datetime64(self.start.to_datetime64(), "ns").astype("int64")),
                int(np.datetime64(self.end.to_datetime64(), "ns").astype("int64")))

    def __str__(self) -> str:
        return f"{self.name} {self.start.date()} .. {self.end.date()}"


@dataclass
class WindowResult:
    window: Window
    daily: pd.Series
    trades_per_day: pd.Series
    performance: Performance
    round_trips: dict
    trips: dict
    benchmarks: Benchmarks | None = None


@dataclass
class GridResult:
    ambiguities: Ambiguities
    engineering: Engineering
    table: pd.DataFrame
    winner: Combination
    train: WindowResult
    test: WindowResult
    warnings: list[str] = field(default_factory=list)


def build_grid(gdx_weights=GRID_GDX_WEIGHTS,
               entry_thresholds=GRID_ENTRY_THRESHOLDS,
               lookbacks=GRID_LOOKBACKS) -> list[Combination]:
    """Cartesian product, ordered lookback-outermost so the z-cache hits."""
    combos = [
        Combination(float(w), float(e), int(lb))
        for lb, w, e in itertools.product(lookbacks, gdx_weights, entry_thresholds)
    ]
    return combos


def _slice(sim: Simulation, window: Window, engineering: Engineering
           ) -> tuple[pd.Series, pd.Series]:
    keep = window.mask(sim.daily.index)
    daily = sim.daily[keep]
    counts = sim.trades_per_day[keep]
    if engineering.flat_days == "exclude":
        live = counts.to_numpy() > 0
        daily, counts = daily[live], counts[live]
    return daily, counts


def evaluate(sim: Simulation, window: Window, ambiguities: Ambiguities,
             engineering: Engineering) -> WindowResult:
    daily, counts = _slice(sim, window, engineering)
    lo, hi = window.codes
    keep = (sim.trip_day >= lo) & (sim.trip_day <= hi)
    return WindowResult(
        window=window,
        daily=daily,
        trades_per_day=counts,
        performance=performance(daily, ambiguities.return_mode),
        round_trips=round_trip_stats(counts),
        trips=trip_economics(sim.trip_return[keep], sim.entry_idx[keep],
                             sim.exit_idx[keep], ambiguities.return_mode),
    )


def attach_benchmarks(result: WindowResult, sim: Simulation, bars: Bars,
                      ambiguities: Ambiguities) -> WindowResult:
    """Prereg section 7 -- computed on the window's bars, not the whole series."""
    price = bars.gld[bars.signal_mask]
    lo, hi = result.window.codes
    in_window = (bars.day_code >= lo) & (bars.day_code <= hi)
    trip_in = (sim.trip_day >= lo) & (sim.trip_day <= hi)

    result.benchmarks = benchmarks(
        result.daily, ambiguities.return_mode,
        price=price[in_window],
        day_code=bars.day_code[in_window],
        day_start=bars.day_start[in_window],
        day_end=bars.day_end[in_window],
        position=sim.position[in_window],
        side=sim.side[trip_in],
    )
    return result


def cumulative_train_return(daily: pd.Series, return_mode: str) -> float:
    """The prereg's selection objective: cumulative IN-SAMPLE return."""
    if daily.empty:
        return float("-inf")
    if return_mode == "log":
        return float(np.expm1(daily.to_numpy().sum()))
    return float(np.prod(1.0 + daily.to_numpy()) - 1.0)


def run_grid(bars: Bars, train: Window, test: Window,
             ambiguities: Ambiguities, engineering: Engineering,
             combos: list[Combination] | None = None,
             progress=None) -> GridResult:
    """Exhaustive TRAIN search, freeze the argmax, apply unchanged to TEST."""
    combos = combos or build_grid()
    z_cache: dict = {}
    rows: list[dict] = []

    last_lookback = None
    for i, combo in enumerate(combos):
        if combo.lookback != last_lookback:
            z_cache.clear()          # bound memory: one lookback at a time
            last_lookback = combo.lookback
        sim = simulate(bars, combo, ambiguities, engineering, z_cache=z_cache)

        tr_daily, tr_counts = _slice(sim, train, engineering)
        te_daily, te_counts = _slice(sim, test, engineering)
        tr_perf = performance(tr_daily, ambiguities.return_mode)
        te_perf = performance(te_daily, ambiguities.return_mode)

        rows.append({
            **combo.as_dict(),
            "train_cumulative": cumulative_train_return(tr_daily, ambiguities.return_mode),
            "train_sharpe": tr_perf.sharpe,
            "train_annual": tr_perf.annual_arithmetic,
            "train_round_trips_per_day": float(tr_counts.mean()),
            "test_cumulative": te_perf.cumulative,
            "test_sharpe": te_perf.sharpe,
            "test_annual": te_perf.annual_arithmetic,
            "test_round_trips_per_day": float(te_counts.mean()),
        })
        if progress is not None:
            progress(i + 1, len(combos), combo)

    table = pd.DataFrame(rows)
    best_row = table.loc[table["train_cumulative"].idxmax()]
    winner = Combination(float(best_row["gdx_weight"]),
                         float(best_row["entry_threshold"]),
                         int(best_row["lookback"]))

    winner_sim = simulate(bars, winner, ambiguities, engineering)
    train_res = attach_benchmarks(
        evaluate(winner_sim, train, ambiguities, engineering),
        winner_sim, bars, ambiguities)
    test_res = attach_benchmarks(
        evaluate(winner_sim, test, ambiguities, engineering),
        winner_sim, bars, ambiguities)

    warnings: list[str] = []
    if not np.isfinite(best_row["train_cumulative"]):
        warnings.append("winning combination has a non-finite TRAIN cumulative")
    if test_res.round_trips["total_round_trips"] == 0:
        warnings.append("winning combination takes NO trades in TEST")

    return GridResult(ambiguities=ambiguities, engineering=engineering,
                      table=table, winner=winner, train=train_res,
                      test=test_res, warnings=warnings)


def select_winner_only(bars: Bars, train: Window, ambiguities: Ambiguities,
                       engineering: Engineering,
                       combos: list[Combination] | None = None) -> Combination:
    """TRAIN-only argmax, used by the leakage check to prove TEST is irrelevant."""
    combos = combos or build_grid()
    z_cache: dict = {}
    best, best_val = None, -np.inf
    last_lookback = None
    for combo in combos:
        if combo.lookback != last_lookback:
            z_cache.clear()
            last_lookback = combo.lookback
        sim = simulate(bars, combo, ambiguities, engineering, z_cache=z_cache)
        daily, _ = _slice(sim, train, engineering)
        val = cumulative_train_return(daily, ambiguities.return_mode)
        if val > best_val:
            best, best_val = combo, val
    return best
