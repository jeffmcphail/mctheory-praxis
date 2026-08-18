"""Automated causality / leakage checks for Chan CPO Layer A.

Same contract as `engines/infobar_lstm/leakage_checks.py`: each check returns
(passed: bool, detail: str), and the runner refuses to print a single metric
until every one of them passes.  Structural assertions are the primary defence
-- "the recursion is causal by construction" is an argument, and arguments are
what this experiment exists to distrust.

Two of these are the ones the brief names explicitly:
  * `check_future_perturbation`  -- wreck every bar after i, assert the signal
    and the position at bars <= i are bit-identical
  * `check_selection_ignores_test` -- rerun the FULL 400-combination TRAIN
    search on bars truncated at TRAIN end and assert the frozen combination is
    the same one the TRAIN+TEST run froze

The remaining three are engine self-tests: the vectorised EWMA, the vectorised
Bollinger state machine and the vectorised round-trip extractor are each
checked against a naive Python loop written straight from the prereg text.  A
vectorisation bug would otherwise look exactly like a market result.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .signal import (
    EXIT_RATIO, Ambiguities, Bars, Combination, Engineering,
    extract_trades, positions, simulate, spread_series, zscore,
)
from .unconditional import Window, build_grid, select_winner_only

Check = tuple[bool, str]


def _probe_indices(n: int, count: int = 6) -> list[int]:
    """Deterministic spread of probe rows (no RNG)."""
    fracs = np.linspace(0.25, 0.95, count)
    return sorted({int(f * n) for f in fracs if 1000 < int(f * n) < n - 2})


# --------------------------------------------------------------------------
# engine self-tests: vectorised implementation == the prereg recursion
# --------------------------------------------------------------------------

def check_ewma_recursion(bars: Bars, lookback: int = 90) -> Check:
    """lfilter must reproduce Spread_EMA / Spread_VAR exactly as written."""
    x = spread_series(bars.gld, bars.gdx, 3.0)[:5000]
    alpha = 2.0 / lookback

    ema = np.empty_like(x)
    var = np.empty_like(x)
    ema[0] = x[0]
    var[0] = 0.0
    for t in range(1, x.size):
        ema[t] = alpha * x[t] + (1.0 - alpha) * ema[t - 1]
        var[t] = alpha * (x[t] - ema[t]) ** 2 + (1.0 - alpha) * var[t - 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        z_loop = (x - ema) / var
    z_vec, _ = zscore(x, lookback, denominator="var")

    finite = np.isfinite(z_loop) & np.isfinite(z_vec)
    if not np.allclose(z_loop[finite], z_vec[finite], rtol=1e-11, atol=1e-11):
        worst = np.max(np.abs(z_loop[finite] - z_vec[finite]))
        return False, f"vectorised EWMA differs from the prereg recursion (max |dz|={worst:.3e})"
    return True, (f"EWMA/VAR recursion matches a naive loop over {x.size:,} bars "
                  f"(max |dz|={np.max(np.abs(z_loop[finite] - z_vec[finite])):.2e})")


def check_state_machine(bars: Bars, combo: Combination = Combination(3.0, 1.0, 90)) -> Check:
    """The forward-fill must reproduce an explicit exit-then-enter loop."""
    n = min(200_000, bars.n_signal_bars)
    spread = spread_series(bars.gld, bars.gdx, combo.gdx_weight)
    z_full, finite = zscore(spread, combo.lookback)
    warm = np.arange(z_full.size) >= combo.lookback
    z = z_full[bars.signal_mask][:n]
    valid = (finite & warm)[bars.signal_mask][:n]
    day_start = bars.day_start[:n].copy()
    day_end = bars.day_end[:n].copy()
    day_end[-1] = True

    vec = positions(z, combo.entry_threshold, valid=valid,
                    day_start=day_start, day_end=day_end)

    E = combo.entry_threshold
    exit_thr = EXIT_RATIO * E
    loop = np.zeros(n, dtype=np.int8)
    state = 0
    for t in range(n):
        if day_start[t]:
            state = 0
        if not valid[t]:
            state = 0
        else:
            if state == 1 and z[t] > exit_thr:        # exit long
                state = 0
            elif state == -1 and z[t] < -exit_thr:    # exit short
                state = 0
            if state == 0:                            # then (re)enter
                if z[t] < -E:
                    state = 1
                elif z[t] > E:
                    state = -1
        if day_end[t]:
            loop[t] = 0
            state = 0
        else:
            loop[t] = state

    bad = np.flatnonzero(vec != loop)
    if bad.size:
        return False, (f"vectorised state machine differs from the loop at "
                       f"{bad.size} of {n:,} bars (first at {bad[0]})")
    return True, f"Bollinger state machine matches an explicit loop over {n:,} bars"


def check_trade_extraction(bars: Bars, combo: Combination = Combination(3.0, 1.0, 90)) -> Check:
    """Round-trip blocks must match a naive scan, and must never span sessions."""
    sim = simulate(bars, combo, Ambiguities(), Engineering())
    pos = sim.position

    entries, exits, sides = [], [], []
    t = 0
    while t < pos.size:
        if pos[t] != 0:
            s = t
            while t + 1 < pos.size and pos[t + 1] == pos[s]:
                t += 1
            entries.append(s)
            exits.append(t + 1)
            sides.append(int(pos[s]))
        t += 1

    e_vec, x_vec, s_vec = extract_trades(pos)
    if (e_vec.size != len(entries)
            or not np.array_equal(e_vec, np.asarray(entries))
            or not np.array_equal(x_vec, np.asarray(exits))
            or not np.array_equal(s_vec, np.asarray(sides))):
        return False, (f"trade extraction differs from a naive scan "
                       f"(vectorised {e_vec.size} trips, loop {len(entries)})")
    spans = bars.day_code[e_vec] != bars.day_code[x_vec]
    if spans.any():
        return False, f"{int(spans.sum())} round trips span a session boundary"
    return True, (f"{e_vec.size:,} round trips match a naive scan; none spans a "
                  f"session boundary")


# --------------------------------------------------------------------------
# the two checks the brief names
# --------------------------------------------------------------------------

def check_signal_causality(bars: Bars, combo: Combination = Combination(3.0, 1.0, 90)) -> Check:
    """Prefix invariance: z at bar i must not change when later bars are added."""
    spread = spread_series(bars.gld, bars.gdx, combo.gdx_weight)
    full, _ = zscore(spread, combo.lookback)
    bad = []
    for i in _probe_indices(spread.size):
        prefix, _ = zscore(spread[: i + 1], combo.lookback)
        a, b = full[i], prefix[i]
        if not (np.isclose(a, b, rtol=1e-10, atol=1e-12) or (np.isnan(a) and np.isnan(b))):
            bad.append(i)
    if bad:
        return False, f"z prefix-invariance FAILED at bars {bad}"
    return True, "z(i) is unchanged by bars > i (prefix-invariant)"


def check_future_perturbation(bars: Bars,
                              combo: Combination = Combination(3.0, 1.0, 90)) -> Check:
    """Wreck every bar after i; the position at bars <= i must be bit-identical."""
    amb, eng = Ambiguities(), Engineering()
    base = simulate(bars, combo, amb, eng)

    sig_pos = np.flatnonzero(bars.signal_mask)
    bad = []
    for i in _probe_indices(bars.n_signal_bars, count=4):
        cut = sig_pos[i]
        gld = bars.gld.copy()
        gdx = bars.gdx.copy()
        gld[cut + 1:] *= 1.37          # arbitrary, large, and not a no-op
        gdx[cut + 1:] *= 0.61
        wrecked = dataclasses.replace(bars, gld=gld, gdx=gdx)
        pert = simulate(wrecked, combo, amb, eng)
        if not np.array_equal(base.position[: i + 1], pert.position[: i + 1]):
            bad.append(i)
    if bad:
        return False, f"positions changed when FUTURE bars were perturbed at {bad}"
    return True, ("positions at bars <= i survive a 37%/-39% shock to every bar "
                  "after i, at 4 probe points")


def check_no_overnight_carry(bars: Bars, sim=None,
                             combo: Combination = Combination(3.0, 1.0, 90)) -> Check:
    """Force-liquidation at 16:00 must be structural, not incidental."""
    sim = sim or simulate(bars, combo, Ambiguities(), Engineering())
    held_at_close = int((sim.position[bars.day_end] != 0).sum())
    if held_at_close:
        return False, f"{held_at_close} sessions end with a live position"
    carried = int((sim.position[bars.day_start] != 0).sum())
    entered_at_open = int(np.isin(np.flatnonzero(bars.day_start), sim.entry_idx).sum())
    if carried != entered_at_open:
        return False, (f"{carried - entered_at_open} session-open positions were "
                       f"carried in rather than entered")
    return True, (f"0 of {int(bars.day_start.sum()):,} sessions carry a position in "
                  f"or out; {entered_at_open} open-bar entries are genuine entries")


def check_selection_ignores_test(panel: pd.DataFrame, train: Window, test: Window,
                                 ambiguities: Ambiguities, engineering: Engineering,
                                 winner: Combination, session_start, session_end,
                                 combos=None) -> Check:
    """Rerun the whole TRAIN search on bars that STOP at TRAIN end.

    If any TEST bar could influence selection, the two argmaxes would be free
    to differ.  This reruns the full 400-combination grid rather than sampling
    it, because the claim being tested is about the argmax, not about a
    representative combination.
    """
    from .signal import prepare_bars

    truncated = prepare_bars(panel, start=panel.index[0], end=train.end,
                             session_start=session_start, session_end=session_end,
                             engineering=engineering)
    if truncated.index[-1] > train.end:
        return False, "truncated bars still contain post-TRAIN timestamps"

    alt = select_winner_only(truncated, train, ambiguities, engineering,
                             combos=combos or build_grid())
    if alt != winner:
        return False, (f"frozen combination CHANGED when TEST bars were removed: "
                       f"{winner} (with TEST present) vs {alt} (TRAIN only)")
    return True, (f"frozen combination is {winner} whether or not TEST bars are "
                  f"present (full {len(combos or build_grid())}-combination rerun)")


def check_windows_disjoint(train: Window, test: Window, bars: Bars) -> Check:
    """TRAIN and TEST must not overlap, and no bar may exceed TEST end."""
    if train.end >= test.start:
        return False, f"TRAIN ends {train.end} but TEST starts {test.start}"
    if bars.index[-1] > test.end + pd.Timedelta(days=1):
        return False, (f"bars run to {bars.index[-1]}, past TEST end {test.end} "
                       f"-- walk-forward data is visible to Phase 3")
    return True, (f"TRAIN {train.start.date()}..{train.end.date()} and TEST "
                  f"{test.start.date()}..{test.end.date()} are disjoint; no bar "
                  f"past {bars.index[-1]}")


# --------------------------------------------------------------------------

def run_all(bars: Bars, panel: pd.DataFrame, train: Window, test: Window,
            ambiguities: Ambiguities, engineering: Engineering,
            winner: Combination | None, session_start, session_end,
            combos=None, include_selection: bool = True) -> list[tuple[str, bool, str]]:
    """Every check, in the order they would fail most informatively."""
    results = [
        ("windows_disjoint", *check_windows_disjoint(train, test, bars)),
        ("ewma_recursion", *check_ewma_recursion(bars)),
        ("state_machine", *check_state_machine(bars)),
        ("trade_extraction", *check_trade_extraction(bars)),
        ("signal_causality", *check_signal_causality(bars)),
        ("future_perturbation", *check_future_perturbation(bars)),
        ("no_overnight_carry", *check_no_overnight_carry(bars)),
    ]
    if include_selection and winner is not None:
        results.append((
            "selection_ignores_test",
            *check_selection_ignores_test(panel, train, test, ambiguities,
                                          engineering, winner,
                                          session_start, session_end, combos),
        ))
    return results
