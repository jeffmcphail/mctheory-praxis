# Retro: Cycle 59 Phase 3 -- Chan CPO Layer A (unconditional) replication

**Brief:** `claude/handoffs/BRIEF_chan_cpo_phase3_layer_a.md` (`9280424`)
**Governing document:** `claude/handoffs/CYCLE59_CHAN_CPO_PREREGISTRATION.md` (`e21e0e9`)
**Date:** 2026-08-18
**Mode:** Code (implementation + execution)
**Status:** DONE -- Layer A did NOT reproduce the target. Phase 5 stays gated.
**Predecessor:** Cycle 59 Phase 2 (`2e3205c`) -- Kibot data layer

---

## Summary

Built `engines/chan_cpo/` Layer A (signal, grid, metrics, leakage checks, CLI)
and ran the pre-registered unconditional replication GROSS. All eight
leakage/causality checks pass, including the full 400-combination re-selection
on bars truncated at TRAIN end. The result does not reproduce Chan: it
**overshoots by a factor of three** (TEST Sharpe 5.974 against a target of
1.947) at **50.1 round trips per session**, and every one of the eight
A1 x A2 x A4 configurations overshoots the same way (Sharpe 5.41 to 5.99).

The explanation is not a defect in the engine and not a defect in the market.
It is the **selection rule**. Gross cumulative return on a zero-cost minute-bar
grid is monotone in turnover, so "maximise cumulative in-sample return" always
lands on the highest-turnover corner. Chan's 1.947 sits at the OPPOSITE corner:
it is essentially the **minimum of our own 400-cell TEST Sharpe distribution**
(min 1.931, at 720-minute lookback and 3.3 round trips/day). The engine
produces Chan's number; the pre-registered objective simply never picks it.

Economically the whole family is dead on cost regardless of which corner you
stand in: gross alpha at the frozen cell is **0.425 bps per round trip against
the pre-registered 1.26 bps round-trip cost**, and only 47 of 400 cells earn
more per trip than one round trip costs.

---

## REPORTING ORDER (as mandated by the brief)

### 1. ROUND TRIPS PER DAY -- TEST window, before any Sharpe

| statistic | value |
|---|---|
| **mean per session** | **50.104** |
| **median per session** | **49.000** |
| p95 / max per session | 70 / 91 |
| total round trips | 37,879 over 756 sessions |
| sessions with a trade | 756 (100.0%) |
| mean holding period | 7.40 bars (median 3) |
| gross bps per round trip | mean **0.425**, median 0.593, win rate 55.5% |

At the pre-registered 1.26 bps per round trip that is 63.13 bps per session,
**159.09% annualised**, against a 53.67% gross annual return. The breakeven
cost is **0.425 bps per round trip**; the prereg assumes 1.26. Phase 4 owns the
overlay -- this is only the arithmetic section 6 requires next to the count.

The prereg's own worked example assumed 5 round trips/day. The strategy as
specified does **ten times** that.

### 2. GROSS METRICS ON TEST vs PRE-REGISTERED TARGET

| metric | target | actual | verdict |
|---|---|---|---|
| Sharpe | 1.947 | **5.974** | MISS (+4.03, tol +/-0.2) |
| annual return (arithmetic) | 17.29% | **53.67%** | MISS (tol +/-2pp) |
| annual return (CAGR) | 17.29% | **70.26%** | MISS |
| 3-year cumulative | 73.00% | 393.55% | -- |
| Calmar (CAGR / maxDD) | 0.984 | 16.955 | -- |

Supporting: annualised vol 8.99%, max drawdown 4.14%, hit rate 67.6%,
756 sessions scored.

**Every miss is an OVERSHOOT.** The engine is not too pessimistic; it is
wildly too optimistic relative to the published figure.

**A defect in the target table itself.** 17.29% annual and 73% three-year
cumulative are mutually inconsistent: compounding 17.29% for three years gives
61.4%, and 73% over three years implies a 20.07% CAGR. Geometric can never
exceed arithmetic on one series, so both published numbers cannot be right.
The same ~2.5pp gap appears in the conditional row (19.77% / 83%). Both
annualisations are reported rather than picking the flattering one.

### 3. FROZEN COMBINATION

| | |
|---|---|
| GDX_weight | 2.00 |
| entry_threshold | 1.00 (exit -0.60) |
| lookback | 30 minutes |
| ambiguities | A1=var, A2=simple, A4=unadj |
| engineering | price=trade, bars=rth, ewma_reset=none, flat_days=include, lag=0 |

TRAIN 2009-09-28..2017-12-31: cumulative **10,878.53%**, Sharpe 5.570,
50.02 round trips/day. An in-sample cumulative of that size is itself the tell:
the selection objective is being maximised by turnover, not by edge.

The winner sits at the **shortest** lookback in the grid (30 min) and the
selection is stable -- the top five TRAIN cells are all lookback=30, w=2.0.

### 4. BENCHMARKS (prereg section 7 -- the strategy is UNHEDGED)

| series | total | annual | Sharpe | maxDD |
|---|---|---|---|---|
| strategy (gross) | 393.55% | 53.67% | 5.974 | 4.14% |
| a. buy-and-hold GLD | 43.11% | 12.97% | 0.910 | 14.01% |
| b. intraday-only long GLD | -6.16% | -1.67% | -0.176 | 15.87% |

Realised exposure: time in market 95.57%; of in-market bar-time 49.65% long /
50.35% short; 49.96% of round trips long. Correlation of daily strategy
returns to benchmark (b): **0.014**.

**The strategy BEATS control (b).** The 2018-2020 gold rally is carried
entirely by the overnight gap -- intraday-only long GLD is *negative* over the
window (-6.16%) while buy-and-hold is +43.11%. So the section 7 trap does not
fire: this result is not intraday gold drift. Exposure is near-perfectly
balanced long/short and correlation to (b) is ~0. Whatever the number is, it
is not directional gold.

---

## What the number actually is

Three findings, in order of decisiveness.

**(i) The pre-registered selection rule is degenerate at zero cost.** Across
the 400-cell grid, TEST Sharpe rises monotonically with turnover and gross
alpha per round trip falls monotonically:

| lookback | rt/day | TEST Sharpe | gross annual | gross bps/round trip | cost @1.26bps | net |
|---|---|---|---|---|---|---|
| 30 | 48.8 | 5.399 | 47.4% | 0.385 | 154.8% | -107.4% |
| 60 | 32.9 | 4.983 | 42.6% | 0.514 | 104.3% | -61.7% |
| 90 | 25.7 | 4.801 | 40.8% | 0.630 | 81.6% | -40.8% |
| 120 | 21.3 | 4.525 | 38.1% | 0.710 | 67.7% | -29.6% |
| 180 | 16.2 | 4.076 | 33.6% | 0.832 | 51.3% | -17.7% |
| 240 | 13.2 | 3.828 | 30.9% | 0.944 | 41.8% | -10.9% |
| 360 | 9.7 | 3.478 | 26.9% | 1.129 | 30.8% | -3.9% |
| 720 | 5.7 | 2.857 | 20.5% | 1.498 | 18.0% | +2.5% |

(mean over the 50 cells at each lookback; corr(rt/day, Sharpe) = +0.849,
corr(rt/day, bps per trip) = -0.868.)

Maximising gross cumulative return therefore *must* select the top row. Cost,
which is charged per round trip, is what would have selected the bottom row.

**(ii) Chan's 1.947 is the floor of our own distribution, not a value we
cannot reach.** TEST Sharpe over the 400 cells: min 1.931, median 4.327,
max 6.243. Exactly two cells land inside the pre-registered +/-0.2 tolerance,
both at lookback=720:

| w | entry | lookback | rt/day | Sharpe | gross annual | bps/trip | cost@1.26 | net |
|---|---|---|---|---|---|---|---|---|
| 2.0 | 2.5 | 720 | 3.34 | 1.931 | 11.20% | 1.330 | 10.61% | +0.59% |
| 4.0 | 2.5 | 720 | 1.88 | 1.972 | 7.16% | 1.512 | 5.96% | +1.19% |

Chan's Sharpe is reproducible inside the pre-registered grid. It is reachable
only at 2-3 round trips per day -- which is consistent with a paper that says
"multiple round trips per day" and reports 17.29% annual, and flatly
inconsistent with 50. The annual return at those cells (11.2% / 7.2%) still
undershoots 17.29%, so this is a partial match, not a reproduction.

**(iii) The gross edge is real but sub-spread, and the fill convention
inflates it.** Two out-of-protocol diagnostics, run to explain the overshoot,
never to select anything:

| run | rt/day | Sharpe | gross annual | bps/trip | win rate |
|---|---|---|---|---|---|
| primary (trade close, lag 0) | 50.1 | 5.974 | 53.67% | 0.425 | 55.5% |
| D1 quote mid, lag 0 | 47.9 | 5.072 | 44.79% | 0.371 | 53.1% |
| D2 trade close, **lag 1 bar** | 29.0 | 3.254 | 31.51% | 0.432 | 48.0% |
| D3 quote mid, lag 1 bar | 28.0 | 3.310 | 32.14% | 0.455 | 46.1% |

Filling one bar after the close that produced the signal -- instead of at that
very close -- halves the gross Sharpe (5.974 -> 3.254) and the trade count.
Switching signal and fill to the quote mid barely moves it, so this is a
timing artefact, not simple bid-ask bounce. Even after both, per-trip alpha
stays at ~0.43 bps: about **half of GLD's own 0.79 bps median quoted spread**
(prereg section 6). An edge smaller than one spread is not a tradeable edge at
any turnover.

---

## Ambiguity sweep A1 x A2 x A4 -- all eight MISS

| A1 | A2 | A4 | w | entry | lb | rt/day | Sharpe | annual | cumulative | gross bps/trip |
|---|---|---|---|---|---|---|---|---|---|---|
| var | simple | unadj | 2.0 | 1.00 | 30 | 50.10 | 5.974 | 53.67% | 393.55% | 0.425 |
| var | simple | adj | 2.0 | 2.50 | 30 | 47.96 | 5.570 | 46.79% | 302.21% | 0.387 |
| var | log | unadj | 2.0 | 1.00 | 30 | 50.10 | 5.994 | 54.13% | 400.36% | 0.428 |
| var | log | adj | 2.0 | 0.70 | 30 | 50.72 | 5.459 | 51.54% | 362.52% | 0.402 |
| std | simple | unadj | 2.0 | 0.20 | 30 | 46.79 | 5.630 | 50.46% | 348.37% | 0.428 |
| std | simple | adj | 2.0 | 0.20 | 30 | 46.87 | 5.412 | 49.44% | 334.63% | 0.419 |
| std | log | unadj | 2.0 | 0.20 | 30 | 46.79 | 5.653 | 50.91% | 354.29% | 0.431 |
| std | log | adj | 2.0 | 0.20 | 30 | 46.87 | 5.437 | 49.89% | 340.48% | 0.422 |

Gross alpha per round trip is 0.387-0.431 bps in **every** configuration --
under a third of the 1.26 bps the prereg charges per round trip, and about half
GLD's 0.79 bps median quoted spread.

The four interpretation gaps are **not** the explanation for anything: every
configuration selects lookback=30 and w=2.0, trades 47-51 times a session, and
overshoots by 3.5x to 4x. A1 (var vs std) shifts only *which* entry threshold
wins, because dividing by VAR rather than STD rescales z by roughly 1/spread
and makes the threshold nearly inoperative -- with A1=var the winning threshold
drifts anywhere in the grid, with A1=std it pins to 0.20 every time. Neither
changes the outcome.

Per prereg section 9, A1-A4 are now exhausted.

---

## Files created

| file | lines | what it does |
|---|---|---|
| `engines/chan_cpo/signal.py` | 479 | spread, causal EWMA/VAR recursions (A1), Bollinger state machine, execution lag, round-trip extraction, daily returns (A2) |
| `engines/chan_cpo/unconditional.py` | 228 | 400-combination TRAIN grid, freeze, apply to TEST, window slicing |
| `engines/chan_cpo/metrics.py` | 220 | performance metrics, section 7 benchmarks, round-trip and per-trip economics |
| `engines/chan_cpo/leakage_checks.py` | 273 | eight causality/leakage checks; must pass before any metric prints |
| `engines/chan_cpo/run_chan_cpo.py` | 498 | CLI (`--validate`, `--sweep`, all ambiguities and engineering knobs), fixed reporting order |
| `outputs/chan_cpo_layer_a/` | 1.1 MB | 12 grid CSVs, 12 summary JSONs, 12 daily-return CSVs, ambiguity_sweep.csv |

No existing file was modified. The Phase 2 data layer (`data_loader.py`,
`build_panel.py`) was consumed unchanged.

---

## Key decisions

**Reporting order is enforced in code, not by convention.** `report()` calls
`report_round_trips()` before `report_metrics()`; there is no path that prints
a Sharpe first.

**Selection is proved TRAIN-only rather than argued.** Signals are computed
once over the continuous bar series so the EWMA reaches TEST already warmed
(causal -- an IIR filter at bar t sees only bars <= t, and that is what a live
deployment does). The check then reruns the *entire* 400-combination TRAIN
search on bars physically truncated at TRAIN end and asserts the frozen
combination is identical. It is.

**Walk-forward data is cut at load, not ignored downstream.** `prepare_bars`
takes a hard `end`, so Phase 3 could not see a post-2020 bar even by accident.

**Two spread columns, honoured.** Layer A is gross and consumes no spread at
all; when Phase 4 does, `cost_spread_bps` (floored) is the only column
available to it. The signed column stays diagnostic-only.

**The vectorised state machine was wrong on first write, and the check caught
it.** A naive forward-fill carries a long position through the *short* hold
band `[0.6E, E]`, but a long's exit test (`z > -0.6E`) is satisfied there, so it
must exit. Rewritten as two prefix maxima (last decisive bar; last
opposite-band carry bar) and verified bar-for-bar against an explicit
exit-then-enter loop over 200,000 bars. Had this shipped, it would have looked
exactly like a market result.

**Decisions the prereg does not specify were exposed as flags, not resolved
silently** (see Open Items). Primary is fixed at the most faithful reading of
each; the two that were varied are labelled out-of-protocol diagnostics and
selected nothing.

---

## Test results

`python -m engines.chan_cpo.run_chan_cpo --validate` -- **8 of 8 PASS**, and
the same eight run inside every primary invocation before any metric prints.

| check | result |
|---|---|
| windows_disjoint | PASS -- no bar past 2020-12-31 15:59 |
| ewma_recursion | PASS -- lfilter matches the prereg recursion loop, max abs dz = 0.00e+00 over 5,000 bars |
| state_machine | PASS -- matches an explicit exit-then-enter loop over 200,000 bars |
| trade_extraction | PASS -- 69,476 round trips match a naive scan; none spans a session boundary |
| signal_causality | PASS -- z(i) prefix-invariant at 6 probe points |
| future_perturbation | PASS -- positions at bars <= i survive a +37%/-39% shock to every later bar, 4 probes |
| no_overnight_carry | PASS -- 0 of 2,836 sessions carry a position in or out |
| selection_ignores_test | PASS -- frozen combination identical with TEST bars removed, full 400-combination rerun |

Runtime: 400-combination grid in 22s over 1,099,917 signal bars; full primary
including all checks and the re-selection, 40s.

---

## Failures and debugging trail

**Vectorised state machine, first version -- WRONG (caught pre-run).** Wrote
the Bollinger machine as `np.where(...)` + forward-fill of a NaN "hold" band.
Realised while writing the naive-loop check that the hold band is
state-dependent: `[-E, -0.6E]` carries a long but must flatten a short, and
`[0.6E, E]` carries a short but must flatten a long. A single-bar jump across
the inner band -- entirely possible at E=0.2, where the inner band is
(-0.12, 0.12) -- would have kept a position the specification exits. Rewrote
using two prefix maxima; the check now pins the implementation to the loop.

**`Window.mask` returned a pandas array, not ndarray.** `(sessions >= start) &
(...)` on a `DatetimeIndex` in pandas 3.0 returns an ndarray already, so the
`.to_numpy()` call raised `AttributeError`. Replaced with `np.asarray`.

**A patch silently did nothing, and two diagnostic runs were wasted.** The
`--execution-lag` flag was wired into the `--sweep` and `--validate` paths but
the third replacement -- the main path -- did not match its anchor, so it was a
no-op. D2 and D3 ran with lag=0 and returned numbers byte-identical to the
primary and to D1, which is what exposed it. Re-patched by line number with an
assertion on the target line, then rerun. Lesson: a string replacement that
silently matches nothing is worse than a crash; assert the match count.

**Ambiguity in Chan's target table, not in our code.** See item 2 above; the
17.29%/73% pair cannot both be right. Recorded rather than resolved.

---

## Git commits

| commit | what |
|---|---|
| `e21e0e9` | Cycle 59 Phase 0 -- pre-registration, standalone, before any strategy code |
| `9280424` | Cycle 59 Phase 3 brief, standalone, before any Phase 3 code (delta anchor) |
| `db487d6` | Cycle 59 Phase 3 -- Layer A engine (5 modules) + 39 result artifacts |
| (this retro) | retro log |

Note: commits `c524a97`, `8ef7414`, `b0441bf`, `be86c9d`, `df301db`,
`6d8c737`, `28baeca` landed on master between the Phase 0 and Phase 3 commits
from outside this session (brief/handoff tracking cleanup and the `split_zip.py`
delta-mode upgrade). They do not touch `engines/chan_cpo/`, and
`BRIEF_chan_cpo_phase3_layer_a.md` is still the most recent committed
`BRIEF_*.md`, so the delta anchor resolves correctly.

---

## Open items for Chat

**1. The decision tree has no node for this outcome, and it needs one.**
On the letter of prereg section 10, "cannot reproduce gross 1.947 after A1-A4"
is branch **A -- FRAMEWORK BROKEN, all prior atlas negatives suspect**. That
inference does not follow here, for three reasons: every structural leakage
check passes including the full re-selection; the engine *does* produce
1.93-1.97 inside the same pre-registered grid; and the miss is an OVERSHOOT,
so an engine biased this way would make prior negatives *more* negative once
corrected, not less. The substantive finding is branch **B** -- published
figures are gross, transaction cost is the gap -- reached by a different route
than the tree anticipated. **Chat must record which branch this is.** Code will
not choose.

**2. The selection rule, not the strategy, is what failed.** "Maximise
cumulative in-sample return" is unusable on a zero-cost minute grid because
gross return is monotone in turnover. If Chan used it, they cannot have got
1.947 from this grid; if they did not, the paper does not say what they used.
This is a genuine gap in the source. Options: leave the prereg's rule as
written and record the result as-is (current state), or amend the prereg to
select on a cost-aware objective and re-run -- which is a **protocol amendment
and Chat's call**, not a Code judgement.

**3. Decisions the prereg does not specify, exposed as flags and NOT tuned.**
Each is fixed at its most faithful reading for the primary and needs
ratification:
   - `--price-source trade` -- "GLD_close" read as the 1-minute bar's trade
     close. `mid` runs; it moves the Sharpe from 5.974 to 5.072.
   - `--bar-universe rth` -- the EWMA consumes only 09:30-15:59 bars. The
     alternative reads "signals 09:30-15:59" as implying extended-hours bars
     also feed the recursion. Not run.
   - `--ewma-reset none` -- one continuous recursion, initialised once at
     t=0 as section 3 writes it, rather than restarting each session. Not run.
   - `--flat-days include` -- zero-trade sessions score as 0.0. Immaterial here
     (100% of TEST sessions trade) but material at the low-turnover corner.
   - `--execution-lag 0` -- fill at the very close whose z produced the signal,
     as the paper implies. lag=1 halves the Sharpe (see D2/D3), so this is the
     single most consequential unspecified decision in the whole experiment.

**4. Phase 5 (Layer B / conditional) stays gated.** Prereg section 9: "Layer B
does not run until Layer A passes." It did not pass. Recommend it stays shut
regardless of how branch A/B is recorded -- the uplift Chan reports
(1.947 -> 2.325) is small next to a 3x reproduction gap.

**5. Phase 4 is now the decisive phase, and its answer is already visible.**
Gross alpha per round trip at the frozen cell is 0.425 bps against a
pre-registered 1.26 bps round-trip cost; 353 of 400 cells earn less per trip
than a trip costs; and per-trip alpha is roughly half GLD's own 0.79 bps median
quoted spread. Phase 4 should confirm, not discover, this.

**6. The target table itself is internally inconsistent.** 17.29% / 73% cannot
both be right (nor 19.77% / 83%). Worth deciding which figure Phase 5's
comparison, if it ever runs, is held to.
