# Retro: Mid-office 02 -- empty windows, and the regression lagged a day

**Brief:** `claude/handoffs/BRIEF_midoffice02_empty_windows_and_lag.md` -- written in Chat and
pasted into the Code session, transcribed verbatim
**Date:** 2026-11-05
**Duration:** about 50 minutes of implementation and measurement (15:20-16:10 EST by the file
times), then this retro
**Status:** COMPLETE, uncommitted. Nothing in the brief was left undone; six decisions are left
for Jeff, below.
**Builds on:** 8cdd037 (exposures standardised over the estimation universe) and f0ae2a9
(Rule 43), both Jeff's commits. Second retro of the `midofficeNN` series.

---

## Summary

Two correctness fixes to the regression's inputs, made in the brief's order and decomposed the
way A to D was. **Change 1:** the window descriptors are missing, not zero, when their window
holds no valid observation, and `compute_descriptors(min_coverage=)` extends that to thinly
covered windows, default 0. On the S&P 500 panel the default changes exactly the 94,882
empty-window zeros -- RSTR and CMRA 18,645 each, STOM 22,512, STOQ 20,780, STOA 14,300 -- and
not one other value; a 50% rule would remove 29,506 more (7,086 on members), reported and not
applied. Its effect is small: R-squared -0.000094, Volatility vol -1.6%, and JPM off the floor
list. **Change 2:** each period's regression reads its exposures (and so their
standardisation), its weights and its membership from the day before, and only its returns from
day t; the forecast exposures are still the last date's. One new function,
`regression_design`, assembles every period's inputs for the estimator and for
`regression_universe`; the estimator's new `lag` defaults to 1; and a price-perturbation test
shows period t's inputs bit-identical when day t is repriced -- end to end on a synthetic panel,
with a lag-0 twin that fails, and once on the real panel through the whole run script. Its
effect is large: R-squared 0.312157 -> 0.294197, Volatility vol 9.40% -> 7.61%, and Momentum's
mean return +26.94% -> +2.44% a year (t 9.1 -> 0.8), because RSTR held the day's own return.
Combined, R-squared 0.312250 -> 0.294197 and Volatility 9.55% -> 7.61%, near the lag probe's
0.295 and 7.79% as Jeff expected. **Size's mean return rose, +0.05% -> +4.92%, where Jeff
expected it to fall.** His mechanism is real -- Size alone is inflated same-day, +8.3% against
+4.1% -- but in the joint regression Momentum, which holds 3.7 times as much of the day's return
and is +0.375 correlated with Size, takes the look-ahead and over-corrects Size downward. The
membership mask moves on 62 security-periods, 43 of which change a regression. Reported and
not changed: `fit_index_weights` is whole-sample and pairs same-day, and the run never calls it;
share counts are carried forward, point-in-time, but stale for 621 member-days after splits;
SERDP now enters no regression yet moves the forecast Volatility exposure by 0.27 rms, which the
run flags. Tests 2,221 passed, 14 new. Rule 43 corrected as its own change.

---

## What was asked (verbatim)

The brief, as pasted (also in `claude/handoffs/BRIEF_midoffice02_empty_windows_and_lag.md`):

```text
== Next part: start only after I reply "committed" ==

Two correctness fixes to the regression's inputs, both already found.
Make them in this order and decompose the effect of each, as you did
for A to D.

1. Window-based descriptors return missing, not zero, when the window
   has no valid observation: RSTR, CMRA, STOM, STOQ and STOA, 14,000 to
   23,000 values each, 4,669 of them on members before their first
   price. Add a minimum-coverage parameter -- the fraction of the window
   with valid data -- whose default changes only the empty case, and
   report how many values a 50% rule would remove so I can decide that
   separately.
   Guard: a synthetic security whose prices start after the window gets
   missing for every window descriptor; on the real panel, no window
   descriptor value is computed from an empty window.

2. Lag the estimation by one day. Everything that defines period t's
   regression is fixed at t-1 -- the exposures, their standardisation
   and the estimation universe -- and only the returns come from day t.
   That is Barra's convention and the descriptor screen's existing rule,
   applied to the model. It moves the membership mask by a day; report
   how many security-periods change. Forecast exposures stay at the
   last date, since they are as of today and forecast forward.
   - Guard: multiply every price dated t by a random factor, rerun with
     the descriptor weights held fixed, and assert that period t's
     exposures and estimation universe are bit-identical and only its
     return vector moved. That turns look-ahead into a test failure
     instead of something someone has to notice.
   - If fit_index_weights draws on returns from the whole sample, say so
     in the retro. That is in-sample weight selection: ordinary for a
     research model, but look-ahead in the strict sense, and I want it
     stated rather than discovered.
   - Check how sampled share counts fill the dates between samples.
     Carried forward from the last sample is point-in-time; back-filled
     or interpolated from a later sample is look-ahead in the cap, and
     so in LNCAP and the regression weights. Report which it is, but do
     not change it here; the rebuild resamples shares anyway.
   - SERDP exists only at the last date, so after this it enters no
     regression, only the forecast exposures. Report whether that leaves
     the forecast Volatility exposure built from a different descriptor
     set than the one its factor was estimated with.

Report per change and combined: R-squared, the five style factor vols
and their mean returns annualised, noise shares, the residual spread,
the specific risk percentiles and the floor list. You measured the lag
alone at 0.312 to 0.295 R-squared and Volatility at 9.55% to 7.79%, so
the combined result should land near that. I expect Size's mean return
to fall as well: today's cap containing today's return makes big stocks
look as though they outperformed, every day.

Do not touch the size pair. Retro and git add command at the end, as
above, then stop.
```

### The go-ahead, and the rulings that came with it (verbatim)

The first "committed" had not landed: `master` was still f72310b, nothing was staged, and
`.git/COMMIT_EDITMSG` held the standardisation message from 15:16:42 -- a commit attempted with
nothing added. Code said so and edited nothing, since this work changes the same files and the
whole-file `git add` would have swept it into midoffice01's commit. Jeff then committed and
wrote:

```text
committed -- two commits, the standardisation and then Rule 43 on its
own. On the retro's open decisions: keep the descriptor screen
members-only, since it should rank descriptors on the universe the
model is estimated on; and keep clipped_outside counting both bounds,
since for LNCAP the +/-4 bound is the one that binds. One correction
for the record: the midoffice briefs are written in Chat and pasted in,
so future BRIEF_ files should say that rather than "given in the Code
session".
```

Code checked `git log` (8cdd037, f0ae2a9, tree clean) before touching anything. What became of
each ruling:

- **The screen stays members-only.** Under the lag the model's universe is fixed with the
  exposures, so the screen now reads its membership on the exposure's day rather than the
  return's -- the universe the model is estimated on, as the ruling says. It moves the screen by
  at most 0.01 on any t-statistic or factor vol (A against A0 in the appendix).
- **`clipped_outside` counts both bounds:** unchanged.
- **The brief correction:** this brief's Source line says written in Chat and pasted in; Rule
  43 and `.claude/commands/retro.md` now say so (a separate change, its own `git add` at the
  end). midoffice01's committed brief and retro still carry the old line -- left for Jeff.

---

## Files modified

| File | Change | Lines (new numbering) |
|---|---|---|
| `src/praxis/midoffice/model/descriptors.py` | `WINDOW_DESCRIPTORS`; `_covered`, the trailing-window coverage mask; `_rolling(min_coverage=, valid=)` masks windows with no valid observation or too little; `_market_model(min_coverage)`; `compute_descriptors(min_coverage=0.0)`, validated, documented, passed to every window descriptor | 63-79, 209-242, 261-280, 308, 331-353, 446-448, 474-506, 556-563, 585-591 |
| `src/praxis/midoffice/model/crosssectional.py` | module docstring: the timing convention; `CrossSectionalNotes.lag` and `.factor_returns`; `_lagged`; `RegressionDesign` and `regression_design`; `regression_universe(lag=1)`; the estimator's `lag=1`, its periods from `regression_design`, the specific risk model fed each residual's own exposures and capitalisations | 55-70, 169-173, 473-555, 564-644, 694-736, 882-894, 1011-1012 |
| `scripts/run_sp500_model.py` | the screen's membership on the exposure date; `trailing_counts` and `window_coverage`, the independent empty-window check and 50% count; `--min-coverage`; the timing line; the membership-move line; the forecast-blend check; the style guard over the dates read; the per-period table on the exposure date; an annual-mean column | 38-42, 90-103, 172-238, 338-341, 408-433, 553-583, 595-681 |
| `tests/midoffice/test_crosssectional.py` | import; `_fit` reads its same-day worlds at `lag=0`; five `regression_universe` calls likewise; 7 new tests | 24, 74-76, 594, 1096-1119, 1300-1469 |
| `tests/midoffice/test_descriptors.py` | import; 2 helpers and 4 new tests | 25, 689-812 |
| `tests/midoffice/test_lookahead.py` (new) | the price-perturbation guard: 3 tests | 1-187 |
| `docs/FACTOR_MODEL.md` | new "Each regression is fixed the day before its return" and "A window with no data gives no value"; the estimation-universe section's guard wording and outsider count brought up to date | 376-379, 384-420, 616-628 |
| `claude/handoffs/BRIEF_midoffice02_empty_windows_and_lag.md` (new) | the brief, verbatim | 1-69 |
| `claude/retros/RETRO_midoffice02_empty_windows_and_lag.md` (new) | this retro | -- |

The separate change: `claude/CLAUDE_CODE_RULES.md` (v1.6: lines 5, 17, 414) and
`.claude/commands/retro.md` (lines 23-25).

---

## What changed and why

### Change 1: an empty window gives no value

`_rolling` already left a value missing before a security's first full window, but a full
window with nothing in it still went through `np.nansum`, and a sum over nothing is zero. So
RSTR and CMRA read zero on every day whose year held no return, and STOM, STOQ and STOA read
zero turnover wherever a carried-forward share count met a window with no volume. On the
committed code a synthetic late starter shows exactly that: 0.0 for CMRA, RSTR, STOM, STOQ, STOA
and STO5, and missing for DASTD, BTSG, HALPHA, SERDP and VLVR, which already had floors of two or
three observations (`teeth.txt`).

Now `_covered` counts each trailing window's valid observations with one cumulative sum, and a
window descriptor is missing unless its window holds at least one and at least `min_coverage`
of the window. What counts as valid is what the descriptor is computed from: a return for
DASTD, RSTR and CMRA; a return on a day the market has one for the market model (BTSG, HALPHA,
and so SERDP, whose residuals exist only where the last window was fitted); a volume for the
turnovers -- the share count is the denominator, carried forward, and does not make a window
covered; and for VLVR both of its windows, the year's dollar volume and the market model's.
`WINDOW_DESCRIPTORS` names the eleven; HILO, LPRI, LNCAP and LCAPCB are one day's values.

`_rolling` masks after computing, so a value whose window has data is computed exactly as
before. On the real panel, value by value against the committed code: the 94,882 empty-window
zeros became missing and nothing else moved (details.txt, part 1).

### Change 2: each regression fixed the day before

`regression_design` builds, for every date the two panels share, the return of that date and,
`lag` shared dates earlier, the exposures, the weights and the membership, plus the estimator's
own test of entry on them. The estimator and `regression_universe` both take their periods from
it, so they cannot disagree, and the look-ahead test reads the same object the fit is run on.
The first `lag` dates have nothing earlier to read and are not periods -- not counted as thin.
The lag counts shared dates, so a date one panel lacks is stepped over: staler, never later.

"Everything that defines period t's regression" was taken to include two things the brief did
not list. The **regression weights**, the square root of capitalisation, are read on t-1: today's
cap holds today's return. And the **specific risk model** is fed, with each residual, the
exposures and capitalisation its own regression read, so V is never explained by an exposure
built from the price that produced the residual. The standardisation needs no change: it is per
date, so the exposures read for period t were standardised over the members, caps and clip
bounds of the date they were read from. The forecast exposures still come from
`exposure_index`, the last date.

**The default is 1.** Barra's convention, and the only safe one for characteristics computed
from prices; a caller who forgets gets no look-ahead, and the look-ahead test runs the
defaults. The estimator tests' synthetic worlds build day t's return from day t's
characteristics, so they are now read at `lag=0`, said in one place (`_fit`) and in five
`regression_universe` calls; 17 of those tests failed at the new default without it. No
assertion changed.

`CrossSectionalNotes` gains `lag` and `factor_returns`, the series the covariance was estimated
from; the run's new mean-return column reads it.

### The run script

- **The screen** holds each univariate regression to the members on the exposure's day, as the
  model's own now are (see the ruling above).
- **New standing lines.** The window guard, worked out from the panel's prices, volumes and
  market series rather than from the descriptor code; the 50% count beside it; the timing
  convention as the notes record it; how many security-periods reading membership a day
  earlier moves; a `!` line for any descriptor in the forecast exposures and in no estimated
  period's (SERDP fires it); an annual-mean column in the factor table.
- **Two existing checks re-aimed.** The index-at-zero guard now covers the 939 dates a
  regression or the forecast reads (938 exposure dates and the forecast's), not the period
  dates; the per-period table counts the index on the exposure date, so "in the regression"
  cannot exceed "of them in the panel" on a day a name leaves.
- `--min-coverage` exposes the rule; default 0.

### Docs

`docs/FACTOR_MODEL.md` gains "Each regression is fixed the day before its return" (the
convention, the guard, the Momentum and Size findings, SERDP, `fit_index_weights`) and "A window
with no data gives no value"; the estimation-universe section's guard wording and outsider
count (100,018 -> 100,011) are brought up to date.

---

## Results

### The numbers asked for, per change and combined

A is the committed code. **A0 -- A's raw descriptors run through the new code at lag 0 --
reproduces A bit for bit**: factor returns, every specific risk, every forecast exposure and
every summary field (analyse.txt), so every difference below is the two changes and nothing
else. B is change 1 alone, C both (the new code; `final_with.txt` is its report), and L change
2 alone, A's descriptors at lag 1, to show the interaction.

| | A: HEAD | B: change 1 | C: both (new) | L: change 2 alone |
|---|---|---|---|---|
| mean R-squared | 0.312250 | 0.312157 | 0.294197 | 0.294170 |
| periods estimated | 939 | 939 | 938 | 938 |
| first estimated period | 2023-02-07 | 2023-02-07 | 2023-02-08 | 2023-02-08 |
| **annual vol** | | | | |
| Volatility | 9.55% | 9.40% | 7.61% | 7.72% |
| Momentum | 7.06% | 7.08% | 7.22% | 7.20% |
| Size | 7.51% | 7.50% | 7.58% | 7.58% |
| Size Nonlinearity | 6.55% | 6.55% | 6.69% | 6.69% |
| Trading Activity | 5.16% | 5.16% | 5.30% | 5.30% |
| **annual mean return (t-stat)** | | | | |
| Volatility | +2.93% (+0.67) | +2.82% (+0.67) | +4.51% (+1.36) | +4.63% (+1.36) |
| Momentum | +26.78% (+9.10) | +26.94% (+9.11) | +2.44% (+0.81) | +2.39% (+0.80) |
| Size | +0.05% (+0.02) | +0.02% (+0.00) | +4.92% (+1.54) | +4.93% (+1.54) |
| Size Nonlinearity | +1.31% (+0.50) | +1.33% (+0.50) | -1.44% (-0.54) | -1.45% (-0.54) |
| Trading Activity | +6.51% (+3.23) | +6.47% (+3.21) | +3.91% (+2.11) | +3.92% (+2.11) |
| **noise share** | | | | |
| Volatility | 3.6% | 3.4% | 5.4% | 5.6% |
| Momentum | 4.0% | 4.0% | 3.9% | 3.9% |
| Size | 9.8% | 9.8% | 9.8% | 9.8% |
| Size Nonlinearity | 17.3% | 17.4% | 17.0% | 17.0% |
| Trading Activity | 10.8% | 10.8% | 10.5% | 10.5% |
| residual spread (relative residual sd) | 0.3804 | 0.3801 | 0.3679 | 0.3682 |
| its largest shares | FRC 61.1%, ECHO 4.0%, LUMN 2.7% | FRC 61.0%, ECHO 4.0%, LUMN 2.7% | FRC 71.3%, SIVB 4.0%, LUMN 2.9% | FRC 71.3%, SIVB 3.9%, LUMN 2.9% |
| specific risk p1 / p10 / p50 | 9.17% / 14.87% / 23.10% | 9.18% / 14.86% / 23.11% | 9.51% / 15.11% / 23.52% | 9.51% / 15.11% / 23.51% |
| specific risk p90 / p99 | 36.12% / 56.14% | 36.12% / 55.92% | 36.36% / 53.35% | 36.36% / 53.35% |
| specific risk min / max | 3.54% / 104.36% | 3.65% / 104.36% | 9.02% / 106.68% | 9.02% / 106.68% |
| forecasts | 470 | 470 | 470 | 470 |
| floor list: characteristic forecast below the floor | JPM | none | none | none |
| floor list: delivered the floor | none | none | none | none |
| regression universe | 485 | 485 | 482 | 482 |
| fallback (in no regression) | 7 | 7 | 10 | 10 |

### Per change: the steps

| step | | dR-squared | Volatility | Momentum | Size | Size Nonlinearity | Trading Activity |
|---|---|---|---|---|---|---|---|
| A -> B | change 1 (empty windows), vol | -0.000094 | -1.6% | +0.3% | -0.0% | -0.1% | -0.0% |
| | mean, points a year | | -0.11 | +0.16 | -0.04 | +0.02 | -0.04 |
| B -> C | change 2 (the lag), after 1, vol | -0.017960 | -19.0% | +2.0% | +1.0% | +2.2% | +2.7% |
| | mean, points a year | | +1.69 | -24.51 | +4.90 | -2.77 | -2.56 |
| A -> C | combined, vol | -0.018053 | -20.3% | +2.3% | +0.9% | +2.1% | +2.6% |
| | mean, points a year | | +1.58 | -24.34 | +4.87 | -2.75 | -2.60 |
| A -> L | change 2 alone, vol | -0.018080 | -19.1% | +2.0% | +0.9% | +2.2% | +2.6% |
| | mean, points a year | | +1.70 | -24.39 | +4.88 | -2.76 | -2.59 |
| L -> C | change 1 after 2, vol | +0.000027 | -1.4% | +0.3% | -0.0% | -0.1% | +0.0% |
| | mean, points a year | | -0.12 | +0.05 | -0.01 | +0.01 | -0.01 |
| B -> Cx | lag: exposures, vol | -0.017222 | -18.3% | +2.5% | +1.0% | +1.8% | +1.4% |
| | mean, points a year | | +3.36 | -24.90 | +4.30 | -2.62 | -2.28 |
| Cx -> Cxw | lag: + weights, vol | -0.000716 | -0.8% | -0.7% | -0.0% | +0.2% | +1.2% |
| | mean, points a year | | -1.67 | +0.40 | +0.60 | -0.14 | -0.28 |
| Cxw -> C | lag: + membership, vol | -0.000021 | -0.0% | +0.1% | +0.0% | +0.1% | +0.1% |
| | mean, points a year | | +0.01 | -0.01 | -0.00 | -0.01 | -0.00 |
| C -> E | a 50% coverage rule, vol | +0.000014 | +0.3% | -0.8% | -0.1% | -0.2% | -5.9% |
| | mean, points a year | | +0.01 | +0.01 | -0.00 | +0.00 | -0.00 |

Change 1 does nearly the same either side of the lag (A -> B against L -> C), so the two fixes
barely interact. The lag is the whole move, and lagging the exposures is 96% of its R-squared
change.

### Where the lag's effect comes from

| | B | Cx | Cxw | C |
|---|---|---|---|---|
| exposures read on | t | t-1 | t-1 | t-1 |
| weights read on | t | t | t-1 | t-1 |
| membership read on | t | t | t | t-1 |
| mean R-squared | 0.312157 | 0.294934 | 0.294218 | 0.294197 |
| Volatility vol / mean | 9.40% / +2.82% | 7.68% / +6.18% | 7.61% / +4.50% | 7.61% / +4.51% |
| Momentum vol / mean | 7.08% / +26.94% | 7.26% / +2.04% | 7.21% / +2.45% | 7.22% / +2.44% |
| Size vol / mean | 7.50% / +0.02% | 7.58% / +4.32% | 7.57% / +4.92% | 7.58% / +4.92% |
| Size Nonlinearity vol / mean | 6.55% / +1.33% | 6.67% / -1.29% | 6.68% / -1.44% | 6.69% / -1.44% |
| Trading Activity vol / mean | 5.16% / +6.47% | 5.23% / +4.19% | 5.30% / +3.91% | 5.30% / +3.91% |
| residual spread | 0.3801 | 0.3716 | 0.3686 | 0.3679 |
| fallback | 7 | 10 | 10 | 10 |

Exposures carry the R-squared, the Volatility vol and all of Momentum's mean. The weights add
the rest of Size's rise (+0.60), take 1.67 points a year off Volatility's mean and 2.2 points off
the industries' average (+20.28% -> +18.10%, `industry_means.txt`) -- Jeff's "today's cap holds
today's return", in the place it acts on: the sqrt-cap weights tilt each day's regression
toward that day's winners. Membership adds almost nothing.

### Against Jeff's expectations

1. **"The combined result should land near" 0.312 -> 0.295 and 9.55% -> 7.79%.** It lands at
   0.294197 and 7.61%. The probe lagged the exposures alone on the old descriptors; the closest
   variant here is Cx, the exposures alone on the new ones -- 0.294934 and 7.68%. The rest is
   the weights (-0.8% on Volatility, Cx -> Cxw) and change 1 (-1.4% within the lagged model,
   L -> C).
2. **"I expect Size's mean return to fall."** It rose: +0.05% (t 0.02) -> +4.92% (t 1.54).
   mechanism.txt finds why, with every regression's intercept, weights and membership read on
   t-1 so that only the exposures' date differs:

   | daily regression, no industries | Size, exposures from t | from t-1 | Momentum, from t | from t-1 |
   |---|---|---|---|---|
   | Size alone | +8.30% | +4.08% | | |
   | Momentum alone | | | +28.60% | +4.11% |
   | Size + Momentum | -0.83% | +2.96% | +29.45% | +3.51% |
   | Size + Size Nonlinearity | +8.35% | +8.01% | | |
   | Size + Momentum + Volatility | +0.22% | +2.70% | +28.48% | +3.05% |
   | all five styles | +1.46% | +4.84% | +28.48% | +3.21% |
   | the model, with industries (B, C) | +0.02% | +4.92% | +26.94% | +2.44% |

   Jeff's mechanism is there where Size stands alone: same-day it earns +8.30% against +4.08%,
   because its exposure moves 0.86 standard deviations per unit of the day's return. But
   Momentum's moves 3.17 -- RSTR is a sum that includes today -- and the two exposures are
   +0.375 correlated across members, so the joint regression gives the day's return to
   Momentum and pays for it out of Size: to first order Size's bias goes as
   0.86 - 0.375 x 3.17, which is negative. Size Nonlinearity, 0.933 correlated with Size and
   holding the day's return in the same proportion, does not reverse it: beside it Size earns
   +8.35% same-day against +8.01%.
   Same-day, Size was biased down by about five points a year; lagged, it shows what this
   window did -- large caps outperformed.
3. **"Do not touch the size pair":** untouched. Size and Size Nonlinearity move only through the
   lag: vols +1.0% and +2.2%, means +4.90 and -2.77 points.

### Change 1 in detail

details.txt, part 1, the committed code's raw descriptors against B's:

| descriptor | was a value, now missing | every one a zero before | on members | members, before their first price |
|---|---|---|---|---|
| RSTR | 18,645 | yes | 4,679 | 4,334 |
| CMRA | 18,645 | yes | 4,679 | 4,334 |
| STOM | 22,512 | yes | 705 | 0 |
| STOQ | 20,780 | yes | 567 | 0 |
| STOA | 14,300 | yes | 335 | 0 |
| the other nine | 0 | -- | 0 | 0 |

94,882 values, all of them zeros, and no other value changed in any descriptor. The brief's
"4,669 of them on members before their first price" is close but not exact: the probe counted
windows with no *price* and members on any day, so it was 4,669 on members; with no *return* as
the test it is 4,679 on members, of which 4,334 fall before the security's first price and 345
in gaps or after a last price. Twelve securities are members before their first price, 7,232
member-days: BALL, COR, CPAY, ECHO, EG, ELV, GEN, MRSH, RVTY, SKYD, SW, VMRK. The turnovers have
no such values, because those securities have no share count before their first price either --
the share count is never carried backward (below). The run-script comment says 4,679 and 4,334.

**The guard on the real panel** is a standing line now: "window descriptors need any valid
observation; from a window with none, over 3,610,993 values: none".

**What a 50% rule would remove**, per descriptor, all values and on members -- the run report's
independent count, which matches exactly what `min_coverage=0.5` removes (coverage_check.txt):

| | DASTD | RSTR | CMRA | BTSG | HALPHA | SERDP | STOM | STOQ | STOA | VLVR | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1,612 | 4,847 | 4,648 | 5,009 | 5,009 | 12 | 467 | 1,363 | 3,931 | 2,608 | 29,506 |
| on members | 533 | 1,339 | 1,339 | 1,111 | 1,111 | 5 | 66 | 189 | 423 | 970 | 7,086 |

And what it would do to the model, variant E against C:

| | C: default, empty windows only | E: --min-coverage 0.5 |
|---|---|---|
| mean R-squared | 0.294197 | 0.294211 |
| Volatility vol | 7.61% | 7.63% |
| Momentum vol | 7.22% | 7.16% |
| Size vol | 7.58% | 7.57% |
| Size Nonlinearity vol | 6.69% | 6.67% |
| Trading Activity vol | 5.30% | 4.99% |
| Volatility noise share | 5.4% | 5.3% |
| Momentum noise share | 3.9% | 4.0% |
| Size noise share | 9.8% | 9.9% |
| Size Nonlinearity noise share | 17.0% | 17.1% |
| Trading Activity noise share | 10.5% | 11.5% |
| residual spread | 0.3679 | 0.3682 |
| forecasts | 470 | 468 |
| fallback | ECHO, INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VMRK, VNO | INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO |

Its one visible effect is on Trading Activity, whose turnovers are the most often thinly
covered: vol -5.9%, noise share +1.0 point. ECHO and VMRK lose their only complete row and with it
any forecast.

**The floor list.** The floor is 0.1 x kappa x S -- 3.54% in A and B, 3.48% in C -- and a
security is on the list when its characteristic forecast kappa (1 + V) S is below it, that is
when 1 + V < 0.1 (floor_serdp.txt). JPM sat just under: 1 + V = 0.0767 in A, a characteristic
forecast of 2.72%. Change 1 nudged gamma (Volatility +0.442 -> +0.420) and JPM to 0.1026, 3.63%,
off the list by a hair; the lag then moved gamma a long way -- Volatility +0.420 -> +0.188,
Trading Activity -0.064 -> +0.053, Size -0.133 -> -0.082 -- and JPM to 0.3190, 11.11%. Its
delivered forecast barely moves (12.83%, 12.84%, 13.17%): the blend rests on its own history.
INFO, a fallback and so never on the list, was delivered the floor itself in A (3.54%, the
panel's lowest forecast) and 17.37% in C.

### Change 2 in detail

**The membership mask, a day earlier** (details.txt, part 2). Over the 938 estimated periods
the mask differs between t-1 and t on **62 security-periods**: 61 on the day a name left the
index, one on the day one joined (ILMN, re-added 2026-09-21). **43 of them change a regression:**
42 names are now regressed on the day they left (LUMN 2023-03-20 through CTVA 2026-10-06), and
ILMN is not regressed on its rejoining day. The other 19 had no return that day -- acquired or
delisted on it, SBNY, SIVB and FRC among them. The run reports the counts each time.

**The guard on the real panel** (perturb_real.txt). Every price dated 2025-06-02 times its own
random factor, the whole run script repeated: that period's 24 exposure columns (industries
included), its weights, its membership and its 446-security estimation universe come back bit
for bit, all 479 of its returns move, every earlier period's design is bit-identical, and the
579 earlier factor returns are bit-identical while 2025-06-02's moves. At lag 0 the same
repricing moves that period's exposures and weights.

**The regression universe, 485 -> 482.** ECHO, MRSH and VMRK each have exactly one complete
style row, on 2026-11-03 -- their only share count is the last sample -- so their one same-day
regression was on the last day, on that day's own exposures. Read the day before, they have
none. They join the fallback list (7 -> 10), ECHO drops out of the residual spread's leaders
(it was 4.0% of it), and the last period's regression holds 422 rather than 425.

**The specific risk model** loses half its Volatility coefficient (+0.420 -> +0.188): same-day
HILO, DASTD and CMRA hold today's range, and today's range predicts today's absolute residual.
The residual spread falls 0.3801 -> 0.3679, and FRC's share of it rises 61.0% -> 71.3%.

### The four reports the brief asked for

1. **`fit_index_weights` draws on the whole sample: yes.** It is one pooled least squares over
   every date and security with a return and all of the index's descriptors, normalised to sum
   to one -- in-sample weight selection, look-ahead in the strict sense. It is also same-day:
   it pairs each descriptor with the return of its own date, the defect change 2 removes from
   the model. **The run script never calls it**: its five risk indices are equal-weighted blends
   (`build_risk_indices` without weights), so no in-sample weight selection reaches this model.
   `test_lookahead.py`'s third test pins the cost: refitted after one day is repriced, the
   weights move, and with them the exposures of every period back to the first.
2. **Share counts are carried forward from the last sample: point-in-time in the fill**
   (shares_fill.txt). 21 samples, every 63 trading days from 2022-01-03 and the last day. All
   603,658 between-sample security-days equal the sample at or before them -- nothing
   back-filled or interpolated -- and every security's first count falls on a sample date with
   nothing before it: the 1,401 priced days before a first count have no cap, not a borrowed
   one. SKYD has no count at all. Not checkable here: whether Polygon's
   `share_class_shares_outstanding` "as of" a date is what was known that day; that is the
   vendor's point-in-time claim. **One staleness the forward carry does cause:** across a split,
   the traded price moves on the split date and the count only at the next sample. 36 splits;
   896 security-days with a post-split price on a pre-split count, **621 of them on dates an
   estimated period reads, all on members** -- AVGO at a tenth of its cap for 59 days after
   2024-07-15, NVDA for 19 after 2024-06-10, CMG at a fiftieth for 8, AMCR at five times its cap
   for 59 after its 2026-01-15 reverse split. On those days LNCAP is off by the log of the split
   factor (2.3 for ten-for-one) and the regression weight by its square root. Not changed, as
   asked; the rebuild resamples. DD's two steps (2.39 and 0.33) are not round split ratios and
   may be adjustment factors for other corporate actions; whether its caps are stale is not
   verified.
3. **SERDP: yes, the forecast Volatility exposure is built from a different set.** SERDP has
   values on 2026-11-03 alone. Under the lag it is in none of the 938 estimated periods'
   exposures, but the forecast blend gives it a sixth (floor_serdp.txt). Without it the
   members' forecast exposures would differ by 0.211 on average, 0.273 rms, 1.007 at most
   (correlation 0.964); over all 492 securities with both, by up to 3.60 (correlation 0.912).
   The largest member moves: GL -0.82 -> -1.82, MPWR +2.76 -> +1.87, GEN +0.42 -> +1.31, CMI
   +0.12 -> -0.73, NVR +0.96 -> +0.16. SERDP itself correlates +0.184 with the five-descriptor
   blend, so its sixth is close to independent of the rest. The run now flags it each time. It
   is the extreme case of a wider one: BTSG, HALPHA and VLVR reach the estimated exposures only
   from 2025-04-23, 386 of 938 periods, so Volatility, Momentum and Trading Activity change
   composition inside the window; the forecast's blends match the last 386 periods' except for
   SERDP.
4. **Security-periods the membership move changes:** 62 mask flips, 43 regression changes,
   above.

### Every factor, A against C

| factor | vol A | vol C | mean A | mean C | noise A | noise C |
|---|---|---|---|---|---|---|
| Construction | 17.38% | 17.44% | +15.92% | +21.43% | 49.3% | 50.1% |
| Finance & Real Estate | 15.70% | 15.69% | +26.64% | +22.67% | 13.5% | 13.8% |
| Manufacturing | 11.46% | 11.25% | +27.73% | +21.80% | 18.5% | 19.7% |
| Mining | 20.95% | 20.64% | +24.90% | +12.37% | 14.5% | 15.2% |
| Retail | 13.69% | 13.34% | +22.79% | +13.88% | 9.6% | 10.3% |
| SIC20 | 15.38% | 13.18% | +26.59% | +9.14% | 20.9% | 29.0% |
| SIC28 | 14.77% | 13.54% | +30.69% | +15.99% | 7.9% | 9.6% |
| SIC35 | 17.10% | 17.16% | +29.05% | +26.46% | 6.4% | 6.5% |
| SIC36 | 20.34% | 20.26% | +21.01% | +28.05% | 5.4% | 5.5% |
| SIC37 | 14.29% | 13.96% | +23.77% | +13.78% | 17.4% | 18.7% |
| SIC38 | 14.46% | 14.49% | +27.89% | +15.62% | 10.8% | 11.0% |
| SIC49 | 14.56% | 14.21% | +28.86% | +23.22% | 14.2% | 15.2% |
| SIC60 | 20.87% | 20.59% | +26.88% | +21.73% | 8.0% | 8.4% |
| SIC63 | 15.85% | 14.76% | +29.52% | +19.13% | 11.9% | 14.0% |
| SIC67 | 15.66% | 15.11% | +31.12% | +17.41% | 15.9% | 17.5% |
| SIC73 | 16.55% | 17.28% | +23.28% | +13.48% | 3.8% | 3.5% |
| Services | 13.87% | 13.16% | +17.96% | +14.31% | 22.4% | 25.4% |
| Transport & Utilities | 13.73% | 13.40% | +21.32% | +13.28% | 15.8% | 16.9% |
| Wholesale | 15.73% | 14.48% | +20.49% | +19.98% | 34.7% | 41.9% |
| Volatility | 9.55% | 7.61% | +2.93% | +4.51% | 3.6% | 5.4% |
| Momentum | 7.06% | 7.22% | +26.78% | +2.44% | 4.0% | 3.9% |
| Size | 7.51% | 7.58% | +0.05% | +4.92% | 9.8% | 9.8% |
| Size Nonlinearity | 6.55% | 6.69% | +1.31% | -1.44% | 17.3% | 17.0% |
| Trading Activity | 5.16% | 5.30% | +6.51% | +3.91% | 10.8% | 10.5% |
| average of the 19 industries' means, A | | | +25.08% | | | |
| average of the 19 industries' means, C | | | +18.09% | | | |

The industries' average mean falls 7 points a year: 4.8 from the exposures and 2.2 from the
weights. With no market factor they carry the market's level, which the same-day sqrt-cap
weights had tilted toward each day's winners.

### Without --membership

Not asked for, run to check the flagless path works (without_compare.txt): R-squared 0.304 ->
0.286, Volatility vol 10.50% -> 8.13%, Size's mean +4.96%, the floor list BNY, JPM -> JPM, the
same SERDP flag, no window computed from an empty window.

---

## Key decisions taken (Code)

1. **The lag lives in the estimator, default 1.** The alternative -- default 0 and the run
   script passing 1 -- leaves look-ahead the library's default for the next caller. Cost: 17
   synthetic same-day tests say `lag=0`, assertions untouched.
2. **The weights and the specific-risk inputs are lagged too**, under the brief's "everything
   that defines period t's regression"; the brief listed exposures, standardisation and
   universe. The weights are where Jeff's cap mechanism acts (Cx -> Cxw).
3. **One public assembly point, `regression_design`**, so the guard checks what the estimator
   regresses on rather than a reconstruction.
4. **`min_coverage` covers all eleven window descriptors**, each with one definition of valid
   data; the turnovers count days with a volume, not share counts; VLVR needs both windows. The
   default leaves every non-empty window's value bit-identical.
5. **The screen reads membership on the exposure date**, from the ruling plus the lag.
6. **The standing report gained** the window guard (independent of the code it checks), the
   50% count, the timing line, the membership-move line, the forecast-blend check and the mean
   column; the index-at-zero guard and the per-period table were re-aimed at the dates the
   regressions read.
7. **The guard test holds *fitted* blend weights fixed** (fitted once by `fit_index_weights`),
   the stricter reading of "descriptor weights held fixed" than the run's equal weights, and a
   third test shows why they must be held.
8. **Not changed:** SERDP, `fit_index_weights`, the share fill, the size pair, and
   midoffice01's committed brief and retro.

## Decisions left for Jeff

1. **The coverage rule:** keep 0, empty windows only, or adopt 0.5 -- 29,506 values removed,
   7,086 on members; Trading Activity vol -5.9%, R-squared +0.000014; ECHO and VMRK lose their
   only complete row.
2. **SERDP:** compute it historically (each window's residual series, a storage question), drop
   it from the forecast blend so the forecast matches the estimation, or accept a 0.27 rms
   mismatch on members. The same question, smaller, for the blends before 2025-04-23.
3. **`fit_index_weights`**, if it is ever used: lag it -- descriptor on t, return on t+1 -- and
   fit it out of sample or on an expanding window.
4. **Split-stale caps** for the rebuild: 621 member-days in the window; resample at splits, or
   carry split-adjusted counts against adjusted prices.
5. **midoffice01's Source lines** -- its brief's line 5 and its retro's line 4 still say "given in
   the Code session": correct them, one line each as their own change, or leave the committed
   record as it stands, with this retro and Rule 43 v1.6 saying what is true.
6. **Size's mean under the lag**, +4.92% at t 1.54, as input to the size-pair decision.

---

## Tests

- **Full suite:** 2,221 passed, 1 skipped, 19 warnings (`--ignore=tests/test_market_data.py`,
  its `mctheory` import error as before); 2,207 before. **midoffice:** 770 passed, 1 skipped
  (756 + 14).
- **Ruff,** on the six Python files touched: 3 findings, all present at HEAD, which had 4 on the
  same files (checked on HEAD's copies inside the repo, where `praxis` is first-party) -- using
  `field` cleared one. The one I introduced (E501 in the new test file) is fixed. AST-checked;
  every edited and new file ASCII-only.
- **The tests have teeth.** Against the committed descriptors, the two empty-window tests fail
  (on CMRA, then RSTR) and the file's other 43 pass (teeth.txt). The look-ahead guard's lag-0
  twin fails by design, on the synthetic panel and on the real one.

The 14 new tests, and what each guards:

| test | guards |
|---|---|
| `test_descriptors.py::test_a_security_whose_prices_start_after_the_window_gets_missing_for_every_one` | The brief's synthetic guard: a late starter is missing on every window descriptor while its windows are empty, a security gone before the last window is missing on all eleven there (SERDP included), the share count present throughout; a window with one return still gives exactly that return |
| `test_descriptors.py::test_no_window_descriptor_is_computed_from_an_empty_window_anywhere` | Against each descriptor's inputs, counted in the test: missing wherever its window holds no valid observation -- long gaps and scattered ones, all eleven, and not vacuous |
| `test_descriptors.py::test_min_coverage_removes_thinly_covered_windows_and_alters_no_value` | The rule at coverages of exactly 0.5, 0.583 and 0.1: which descriptors survive 0.5, 0.55 and 1.0; the default keeps all; a rule only removes, and what it keeps is the default's value bit for bit |
| `test_descriptors.py::test_min_coverage_is_a_fraction_of_the_window` | Values outside [0, 1] refused |
| `test_crosssectional.py::test_each_period_reads_its_exposures_weights_and_membership_from_the_date_before` | `regression_design` row by row: returns from t, exposures, weights and membership from t-1, exposure dates recorded, the first date in no regression; lag 0 is same-day |
| `test_crosssectional.py::test_a_name_is_regressed_on_the_day_it_leaves_and_not_on_the_day_it_joins` | The membership move: in on the leaving day, out on the joining day, and a one-day member nowhere |
| `test_crosssectional.py::test_the_estimator_fits_the_lagged_design_and_forecasts_with_the_last_dates_exposures` | Every period's factor returns equal the weighted regression of t's returns on t-1's design, redone by hand; the first date is not a thin period; forecast exposures are the last date's exactly |
| `test_crosssectional.py::test_the_regression_universe_agrees_with_the_estimator_under_the_lag` | `regression_universe` and the estimator agree at lag 0 and 1, and the lag removes a last-day-only member |
| `test_crosssectional.py::test_the_specific_risk_model_learns_from_the_exposures_each_residual_was_regressed_on` | The specific risk model is handed the lagged exposures and lagged capitalisations, bit for bit |
| `test_crosssectional.py::test_a_lag_must_be_a_whole_number_of_periods_the_panel_can_supply` | Negative, fractional and boolean lags refused; a lag as long as the panel refused |
| `test_crosssectional.py::test_the_notes_carry_the_factor_returns_the_covariance_was_estimated_from` | `notes.factor_returns`: the estimated dates, the factor names, one row per period |
| `test_lookahead.py::test_pricing_day_t_differently_moves_its_returns_and_nothing_its_regression_is_fixed_by` | **The brief's guard**, end to end from prices with fitted blend weights held: period t's exposures, weights, membership and universe bit-identical, every return with a price moved, all earlier periods and fits bit-identical, t+1's exposures moved |
| `test_lookahead.py::test_regressing_same_day_fails_the_same_check_which_is_what_the_check_is_for` | At lag 0 the same repricing moves period t's exposures and weights: the check can fail |
| `test_lookahead.py::test_refitting_the_blend_after_repricing_moves_every_period_the_earliest_included` | `fit_index_weights` is whole-sample: refitted after one day is repriced, its weights and the first period's exposures move |

The 17 existing tests that failed at the new default, all in `test_crosssectional.py`, all on
worlds built same-day, and all passing unchanged once `_fit` (and the five `regression_universe`
calls beside it) read those worlds at `lag=0`:
`test_the_estimator_recovers_the_factor_returns_it_was_built_from`,
`test_specific_risk_is_unbiased_on_a_narrow_cross_section`,
`test_specific_risk_is_unbiased_on_a_wide_cross_section_too`,
`test_noise_share_measures_the_bias_it_claims_to`,
`test_noise_share_falls_as_the_cross_section_widens`,
`test_a_thin_period_is_dropped_and_named`,
`test_a_collinear_design_is_flagged_rather_than_hidden`,
`test_a_factor_nobody_carries_in_a_period_is_left_out_of_that_period`,
`test_a_factor_carried_by_one_security_is_counted_where_noise_share_cannot_see_it`,
`test_the_collinearity_note_names_a_constant_column_and_no_absent_market_factor`,
`test_a_collinearity_with_no_identifiable_cause_does_not_invent_one`,
`test_the_regression_universe_is_known_before_estimating_and_agrees_with_it`,
`test_a_zero_weight_removes_a_security_from_that_period`,
`test_a_blend_says_which_floor_was_hit_the_characteristic_forecasts_or_the_delivered_one`,
`test_membership_restricts_each_regression_exactly_as_blanking_the_non_members_would`,
`test_the_regression_universe_counts_only_periods_in_which_a_security_was_a_member`,
`test_the_notes_record_each_estimated_period_and_the_size_of_its_regression`.

## Failures and debugging trail

1. **The first "committed" had not landed** (above): `master` at f72310b, nothing staged,
   `COMMIT_EDITMSG` written at 15:16:42. Code edited nothing until the commits were in.
2. **`_covered`'s first draft** would have mis-sliced a window longer than the panel (a negative
   stop); given an explicit guard before anything ran. Its cumulative sum now names its dtype,
   int64, rather than inheriting a 32-bit platform int.
3. **The new default failed 17 tests**, all same-day synthetic worlds; resolved by saying
   `lag=0` where they are fitted, not by changing what they assert.
4. **A shell heredoc failed on quotes** ("unexpected EOF while looking for matching `''"), as in
   midoffice01; the test patch went through a script written with the Write tool. An earlier
   append through a heredoc had worked but left `test_descriptors.py` with 124 LF lines in a
   CRLF file; normalised. The index stores LF either way.
5. **The forecast-exposure test first looped over every security** and hit a KeyError on S0001:
   under historical specific risk a short-history security gets no exposure vector. It loops
   over the model's exposures now.
6. **The run-script comment first repeated the brief's 4,669**; the value-by-value check found
   4,679 on members and 4,334 before a first price, and the comment says that.
7. **The brief's text:** the first transcript match was the compaction summary's quoted copy;
   the source is the original message of 18:27:33, a paste whose closing tag was stripped.
8. Ruff E501 in the new test file; fixed.

## Problems found, not fixed

1. **Split-stale capitalisations:** 621 member-days in the estimation window across 26 splits,
   36 in the panel (above).
2. **SERDP exists at the last date only**, and is now in the forecast and no regression (flagged
   in the run).
3. **The blends change composition inside the window:** BTSG, HALPHA and VLVR only from
   2025-04-23.
4. **`fit_index_weights` is in-sample and same-day**; unused by the run.
5. **The phantoms ECHO, VMRK and MRSH** have one share count each, at the last sample, so one
   complete row, at the last date: no regression after the lag, fallback forecasts, and under a
   50% rule none at all. The identity rebuild's ground.
6. **CMRA is missing where RSTR is not** on 5,335 security-dates (cmra_gap.txt): where the
   window's cumulative log return falls below -1, USE3's log(1 + Z) is undefined. The
   definition, not a defect introduced here, but the two Momentum-adjacent measures cover
   different sets.
7. **The market model is fitted twice per run, on the same inputs**, once for BTSG, HALPHA and
   SERDP and again for VLVR's residual sd. Not timed here.
8. **Still open from midoffice01:** LCAPCB's MAD clip binding at |z| about 1.2; style exposures
   beyond +/-4 against `exposure_basis.DEFAULT_BOUND`'s docstring; the pre-existing lint;
   `tests/test_market_data.py`'s import error.

## Commits

None by Code (Rule 43). This work sits on Jeff's 8cdd037 and f0ae2a9.

## Open items for Chat

1. Size's lagged mean, +4.92% (t 1.54), for the size-pair decision.
2. The coverage rule: 0 or 0.5.
3. SERDP: historical, dropped from the forecast, or accepted.
4. The rebuild's share sampling has to handle splits.
5. Anything that read this model's factor returns before now read same-day ones -- Momentum
   +26.8% a year at t 9.1, the industries 7 points a year high. The descriptor screen was always
   lagged and is not affected.

---

## Appendix: scratch outputs, copied in full

The session scratch directory is under Temp and can vanish, so every output the retro quotes is here verbatim. The scripts that wrote them are kept in the gitignored `claude/scratch/midoffice02/`; regenerate in this order: run_head.py (needs the HEAD tree exported to headtree/), decompose.py, then the rest.

### final_with.txt

The new standing report: run_sp500_model.py --industries --membership on the working tree (identical to variant_C.txt).

```text
Panel
  1214 trading days, 506 securities
  2022-01-03 .. 2026-11-03
  583,081 of 614,284 return observations present (94.9%)

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, covering 1976-07-01 .. 2026-11-04
  one array for every standardisation and every regression below, each held to that day's members: 506 of the panel's 506 on 2022-01-03, 426 on 2026-11-03
  3 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
  - 2026-03-23 SATS recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 157 panel day(s) from then
  - 2026-10-30 OKTA recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then
  - 2026-10-30 KVUE recorded as removed, but membership() still has it after that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then

Descriptors
  computed 14: LNCAP, LCAPCB, LPRI, DASTD, HILO, CMRA, BTSG, HALPHA, SERDP, RSTR, STOM, STOQ, STOA, VLVR
  - CMRA: computed on raw rather than excess returns: no risk_free supplied
  - RSTR: computed on raw rather than excess returns: no risk_free supplied
  - SERDP: computed at the final period only; the market-model residual series is retained for one window rather than all of them, so SERDP is a current exposure and not a time series
  - STO5: needs 1380 period(s) of history and the panel has 1214, so every value would be missing
  window descriptors need any valid observation; from a window with none, over 3,610,993 values: none
  from under half their window, which a 50% rule would remove (all/on members):
    DASTD 1,612/533  RSTR 4,847/1,339  CMRA 4,648/1,339  BTSG 5,009/1,111  HALPHA 5,009/1,111
    SERDP 12/5  STOM 467/66  STOQ 1,363/189  STOA 3,931/423  VLVR 2,608/970
  347,903 raw values winsorised before standardising
  100,011 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/47  HILO 8/45  CMRA 5/51  BTSG 8/66
    HALPHA 7/66  SERDP 1/66  RSTR 3/54  STOM 13/46  STOQ 14/48  STOA 16/54  VLVR 18/54

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         97%      386    8.39%      9.76%     1.88
  DASTD        93%    1,149    6.13%      7.78%     1.35
  RSTR         95%      938    5.30%      7.11%     1.11
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         94%      938    3.38%      5.58%     1.39
  HALPHA       97%      386    3.32%      6.10%    -0.29
  STOQ         93%    1,145    2.99%      5.21%     1.26
  STOM         93%    1,191    2.95%      5.25%     1.28
  STOA         95%      938    2.56%      4.85%     1.51
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  LPRI         93%    1,213    1.77%      3.77%    -0.41
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  VLVR         95%      386    1.50%      3.92%     0.64
  SERDP           -        0   too few usable periods

Risk indices
  Volatility           BTSG, DASTD, HILO, LPRI, CMRA, SERDP
  Momentum             RSTR, HALPHA
  Size                 LNCAP
  Size Nonlinearity    LCAPCB
  Trading Activity     STOM, STOQ, STOA, VLVR

Industries
  33 classification(s) carried from an earlier date (delisted inside the window): ABMD, ANSS, ATVI, AVB, CDAY
  482 of 506 securities are in at least one estimated period's regression while an index member; every count below is over them
  19 industries over 482 securities in the universe (506 classified); largest SIC73 with 53 (11%), 45 thin group(s) merged into their division
    SIC73                      53
    SIC38                      39
    SIC28                      39
    SIC49                      33
    Manufacturing              33
    Retail                     30
    Transport & Utilities      28
    SIC35                      27
  - 1 industry/industries have no member in the universe and are not factors: SIC24
  - 2 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture, Unclassified
  - Other Industries pooled only 3, below the hard floor of 5, so it is not a factor. 3 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA, FRC, SBNY
  - 3 industry/industries still have fewer than 15 members after merging and their factor returns are largely their members' specific returns: Construction (6), Mining (14), Wholesale (11)
  - 2 security(ies) have no usable SIC code and are left with no industry factor: FRC, SBNY
  - 24 classified security(ies) are outside the universe and count toward no industry: ABMD, CERN, CTXS, DISCA, DISCK

Model
  regression weights: square root of capitalisation
  24-factor model, annualised, estimated on 938 daily period(s) for 470 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.294
  0 collinear period(s) of 938 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 43 security-period(s) of the regressions move -- 42 in, on the day a name left the index, 1 out, on the day one joined; the mask itself differs on 62
  ! Volatility: SERDP is in 492 securities' forecast exposures and in none of the 938 estimated periods', so the factor was estimated on a blend without it

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 3.3e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  -7.7e-18   1.000000   426
  Momentum                    +2.3e-17   1.000000   426
  Size                        -3.1e-17   1.000000   425
  Size Nonlinearity           -7.7e-18   1.000000   425
  Trading Activity            +3.9e-17   1.000000   426

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     504    506          504          503
  of them in the panel      426     458    486          486          426
  in the regression         422     451    477          477          422

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               17.44%     +21.43%       50.1%      938          0      0
  Finance & Real Estate      15.69%     +22.67%       13.8%      938          0      0
  Manufacturing              11.25%     +21.80%       19.7%      938          0      0
  Mining                     20.64%     +12.37%       15.2%      938          0      0
  Retail                     13.34%     +13.88%       10.3%      938          0      0
  SIC20                      13.18%      +9.14%       29.0%      938          0      0
  SIC28                      13.54%     +15.99%        9.6%      938          0      0
  SIC35                      17.16%     +26.46%        6.5%      938          0      0
  SIC36                      20.26%     +28.05%        5.5%      938          0      0
  SIC37                      13.96%     +13.78%       18.7%      938          0      0
  SIC38                      14.49%     +15.62%       11.0%      938          0      0
  SIC49                      14.21%     +23.22%       15.2%      938          0      0
  SIC60                      20.59%     +21.73%        8.4%      938          0      0
  SIC63                      14.76%     +19.13%       14.0%      938          0      0
  SIC67                      15.11%     +17.41%       17.5%      938          0      0
  SIC73                      17.28%     +13.48%        3.5%      938          0      0
  Services                   13.16%     +14.31%       25.4%      938          0      0
  Transport & Utilities      13.40%     +13.28%       16.9%      938          0      0
  Wholesale                  14.48%     +19.98%       41.9%      938          0      0
  Volatility                  7.61%      +4.51%        5.4%      938          0      0
  Momentum                    7.22%      +2.44%        3.9%      938          0      0
  Size                        7.58%      +4.92%        9.8%      938          0      0
  Size Nonlinearity           6.69%      -1.44%       17.0%      938          0      0
  Trading Activity            5.30%      +3.91%       10.5%      938          0      0

  specific risk   median 23.52%   p10 15.11%   p90 36.36%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
                    relative residual sd 0.368, largest shares of its square: FRC 71.3%, SIVB 4.0%, LUMN 2.9%, BNY 1.5%, DISH 1.4%

Qualifications
  - 275 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 272 more
  - 36 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CDAY
  - 50% of the estimated variance of Construction is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 10 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: ECHO, INFO, IPGP, MRSH, PENN
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### head_with.txt

The committed code's report, same flags (HEAD f0ae2a9; identical to variant_A.txt).

```text
Panel
  1214 trading days, 506 securities
  2022-01-03 .. 2026-11-03
  583,081 of 614,284 return observations present (94.9%)

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, covering 1976-07-01 .. 2026-11-04
  one array for every standardisation and every regression below, each held to that day's members: 506 of the panel's 506 on 2022-01-03, 426 on 2026-11-03
  3 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
  - 2026-03-23 SATS recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 157 panel day(s) from then
  - 2026-10-30 OKTA recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then
  - 2026-10-30 KVUE recorded as removed, but membership() still has it after that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then

Descriptors
  computed 14: LNCAP, LCAPCB, LPRI, DASTD, HILO, CMRA, BTSG, HALPHA, SERDP, RSTR, STOM, STOQ, STOA, VLVR
  - CMRA: computed on raw rather than excess returns: no risk_free supplied
  - RSTR: computed on raw rather than excess returns: no risk_free supplied
  - SERDP: computed at the final period only; the market-model residual series is retained for one window rather than all of them, so SERDP is a current exposure and not a time series
  - STO5: needs 1380 period(s) of history and the panel has 1214, so every value would be missing
  348,002 raw values winsorised before standardising
  100,018 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/47  HILO 8/45  CMRA 5/77  BTSG 8/66
    HALPHA 7/66  SERDP 1/66  RSTR 3/80  STOM 13/80  STOQ 14/80  STOA 16/80  VLVR 18/54

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         97%      386    8.39%      9.76%     1.88
  DASTD        93%    1,149    6.13%      7.78%     1.35
  RSTR        100%      938    5.29%      7.08%     1.11
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         99%      938    3.38%      5.69%     1.39
  HALPHA       97%      386    3.32%      6.09%    -0.28
  STOQ        100%    1,145    2.99%      5.22%     1.26
  STOM        100%    1,191    2.95%      5.26%     1.27
  STOA        100%      938    2.56%      4.86%     1.52
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  LPRI         93%    1,213    1.77%      3.77%    -0.40
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  VLVR         95%      386    1.50%      3.92%     0.64
  SERDP           -        0   too few usable periods

Risk indices
  Volatility           BTSG, DASTD, HILO, LPRI, CMRA, SERDP
  Momentum             RSTR, HALPHA
  Size                 LNCAP
  Size Nonlinearity    LCAPCB
  Trading Activity     STOM, STOQ, STOA, VLVR

Industries
  33 classification(s) carried from an earlier date (delisted inside the window): ABMD, ANSS, ATVI, AVB, CDAY
  485 of 506 securities are in at least one estimated period's regression while an index member; every count below is over them
  19 industries over 485 securities in the universe (506 classified); largest SIC73 with 53 (11%), 45 thin group(s) merged into their division
    SIC73                      53
    SIC38                      39
    SIC28                      39
    SIC49                      33
    Manufacturing              33
    Retail                     30
    Transport & Utilities      29
    SIC35                      27
  - 1 industry/industries have no member in the universe and are not factors: SIC24
  - 2 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture, Unclassified
  - Other Industries pooled only 3, below the hard floor of 5, so it is not a factor. 3 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA, FRC, SBNY
  - 3 industry/industries still have fewer than 15 members after merging and their factor returns are largely their members' specific returns: Construction (6), Mining (14), Wholesale (11)
  - 2 security(ies) have no usable SIC code and are left with no industry factor: FRC, SBNY
  - 21 classified security(ies) are outside the universe and count toward no industry: ABMD, CERN, CTXS, DISCA, DISCK

Model
  regression weights: square root of capitalisation
  24-factor model, annualised, estimated on 939 daily period(s) for 470 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 939
  mean cross-sectional R-squared 0.312
  0 collinear period(s) of 939 estimated; 0 factor-period(s) left out for having no member present

  Style exposures over each estimated period's index members with a value, all 939 periods: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 3.3e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  -7.7e-18   1.000000   426
  Momentum                    +2.3e-17   1.000000   426
  Size                        -3.1e-17   1.000000   425
  Size Nonlinearity           -7.7e-18   1.000000   425
  Trading Activity            +3.9e-17   1.000000   426

  Securities per estimated period, over 939
                            min  median    max   2023-02-07   2026-11-03
  index members             503     504    506          504          503
  of them in the panel      426     458    486          486          426
  in the regression         422     451    477          477          425

  Over the 939 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                    periods with a member present
  factor                  annual vol  noise share      >=1  exactly 1   none
  Construction               17.38%       49.3%      939          0      0
  Finance & Real Estate      15.70%       13.5%      939          0      0
  Manufacturing              11.46%       18.5%      939          0      0
  Mining                     20.95%       14.5%      939          0      0
  Retail                     13.69%        9.6%      939          0      0
  SIC20                      15.38%       20.9%      939          0      0
  SIC28                      14.77%        7.9%      939          0      0
  SIC35                      17.10%        6.4%      939          0      0
  SIC36                      20.34%        5.4%      939          0      0
  SIC37                      14.29%       17.4%      939          0      0
  SIC38                      14.46%       10.8%      939          0      0
  SIC49                      14.56%       14.2%      939          0      0
  SIC60                      20.87%        8.0%      939          0      0
  SIC63                      15.85%       11.9%      939          0      0
  SIC67                      15.66%       15.9%      939          0      0
  SIC73                      16.55%        3.8%      939          0      0
  Services                   13.87%       22.4%      939          0      0
  Transport & Utilities      13.73%       15.8%      939          0      0
  Wholesale                  15.73%       34.7%      939          0      0
  Volatility                  9.55%        3.6%      939          0      0
  Momentum                    7.06%        4.0%      939          0      0
  Size                        7.51%        9.8%      939          0      0
  Size Nonlinearity           6.55%       17.3%      939          0      0
  Trading Activity            5.16%       10.8%      939          0      0

  specific risk   median 23.10%   p10 14.87%   p90 36.12%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, FRC 2.1%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
                    relative residual sd 0.380, largest shares of its square: FRC 61.1%, ECHO 4.0%, LUMN 2.7%, SIVB 2.6%, BNY 2.2%

Qualifications
  - 275 period(s) dropped for having fewer than 30 securities: 2022-01-03, 2022-01-04, 2022-01-05 and 272 more
  - 36 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CDAY
  - 49% of the estimated variance of Construction is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 1 security(ies) have a characteristic forecast below the specific-risk floor, their linear V having run past the end of its validity; blended with their own residual history, no forecast delivered is the floor: JPM
  - 7 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: INFO, IPGP, PENN, PVH, UA
  - 275 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not

```

### decompose.out

decompose.py: one line per variant.

```text
A0   lag 0  R2 0.312250  periods 939  Volati 9.5468%/+2.93%  Moment 7.0575%/+26.78%  Size 7.5062%/+0.05%  Size N 6.5506%/+1.31%  Tradin 5.1644%/+6.51%
B    lag 0  R2 0.312157  periods 939  Volati 9.3970%/+2.82%  Moment 7.0810%/+26.94%  Size 7.5034%/+0.02%  Size N 6.5462%/+1.33%  Tradin 5.1624%/+6.47%
Cx   lag 1  R2 0.294934  periods 938  Volati 7.6783%/+6.18%  Moment 7.2609%/+2.04%  Size 7.5762%/+4.32%  Size N 6.6661%/-1.29%  Tradin 5.2330%/+4.19%
Cxw  lag 1  R2 0.294218  periods 938  Volati 7.6139%/+4.50%  Moment 7.2129%/+2.45%  Size 7.5750%/+4.92%  Size N 6.6797%/-1.44%  Tradin 5.2959%/+3.91%
C    lag 1  R2 0.294197  periods 938  Volati 7.6113%/+4.51%  Moment 7.2204%/+2.44%  Size 7.5764%/+4.92%  Size N 6.6872%/-1.44%  Tradin 5.3005%/+3.91%
L    lag 1  R2 0.294170  periods 938  Volati 7.7190%/+4.63%  Moment 7.2014%/+2.39%  Size 7.5775%/+4.93%  Size N 6.6917%/-1.45%  Tradin 5.2984%/+3.92%
E    lag 1  R2 0.294211  periods 938  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%
```

### analyse.txt

analyse.py: exactness, per-variant tables, steps, specific risk, every factor A against C.

```text
=== Exactness: A0 (new code, A's descriptors, lag 0) against A (the committed code)
  factor returns bit-identical: True; specific risks identical: True; forecast exposures identical: True
  every summary field equal: True

=== Per variant
              R2  periods  universe  Volatility    Momentum        Size  Size Nonli  Trading Ac
  A     0.312250      939       485      9.55%       7.06%       7.51%       6.55%       5.16%   HEAD (empty-window zeros, same day)
  A0    0.312250      939       485      9.55%       7.06%       7.51%       6.55%       5.16%   A's descriptors, new code at lag 0
  B     0.312157      939       485      9.40%       7.08%       7.50%       6.55%       5.16%   change 1: empty windows missing
  Cx    0.294934      938       482      7.68%       7.26%       7.58%       6.67%       5.23%   B + exposures from t-1
  Cxw   0.294218      938       482      7.61%       7.21%       7.57%       6.68%       5.30%   Cx + weights from t-1
  C     0.294197      938       482      7.61%       7.22%       7.58%       6.69%       5.30%   B + exposures, weights, membership from t-1
  L     0.294170      938       482      7.72%       7.20%       7.58%       6.69%       5.30%   change 2 alone (A's descriptors, lag 1)
  E     0.294211      938       482      7.63%       7.16%       7.57%       6.67%       4.99%   C with --min-coverage 0.5 (not adopted)

  style vol by variant
       Volatilit  Momentum      Size Size Nonl Trading A
  A        9.55%     7.06%     7.51%     6.55%     5.16%
  A0       9.55%     7.06%     7.51%     6.55%     5.16%
  B        9.40%     7.08%     7.50%     6.55%     5.16%
  Cx       7.68%     7.26%     7.58%     6.67%     5.23%
  Cxw      7.61%     7.21%     7.57%     6.68%     5.30%
  C        7.61%     7.22%     7.58%     6.69%     5.30%
  L        7.72%     7.20%     7.58%     6.69%     5.30%
  E        7.63%     7.16%     7.57%     6.67%     4.99%

  style mean by variant
       Volatilit  Momentum      Size Size Nonl Trading A
  A       +2.93%   +26.78%    +0.05%    +1.31%    +6.51%
  A0      +2.93%   +26.78%    +0.05%    +1.31%    +6.51%
  B       +2.82%   +26.94%    +0.02%    +1.33%    +6.47%
  Cx      +6.18%    +2.04%    +4.32%    -1.29%    +4.19%
  Cxw     +4.50%    +2.45%    +4.92%    -1.44%    +3.91%
  C       +4.51%    +2.44%    +4.92%    -1.44%    +3.91%
  L       +4.63%    +2.39%    +4.93%    -1.45%    +3.92%
  E       +4.52%    +2.44%    +4.92%    -1.44%    +3.90%

  style tstat by variant
       Volatilit  Momentum      Size Size Nonl Trading A
  A        +0.67     +9.10     +0.02     +0.50     +3.23
  A0       +0.67     +9.10     +0.02     +0.50     +3.23
  B        +0.67     +9.11     +0.00     +0.50     +3.21
  Cx       +1.85     +0.68     +1.35     -0.49     +2.26
  Cxw      +1.36     +0.82     +1.54     -0.54     +2.11
  C        +1.36     +0.81     +1.54     -0.54     +2.11
  L        +1.36     +0.80     +1.54     -0.54     +2.11
  E        +1.37     +0.81     +1.54     -0.54     +2.14

  style noise by variant
       Volatilit  Momentum      Size Size Nonl Trading A
  A         3.6%      4.0%      9.8%     17.3%     10.8%
  A0        3.6%      4.0%      9.8%     17.3%     10.8%
  B         3.4%      4.0%      9.8%     17.4%     10.8%
  Cx        5.3%      3.9%      9.9%     17.2%     10.8%
  Cxw       5.3%      3.9%      9.8%     17.1%     10.5%
  C         5.4%      3.9%      9.8%     17.0%     10.5%
  L         5.6%      3.9%      9.8%     17.0%     10.5%
  E         5.3%      4.0%      9.9%     17.1%     11.5%

=== Steps (each row: from -> to, change in R2 and in each style's vol, in percent)
    A -> B    change 1 (empty windows)     dR2 -0.000094  Volati -1.6%  Moment +0.3%  Size -0.0%  Size N -0.1%  Tradin -0.0%
                                         mean  Volati -0.11%  Moment +0.16%  Size -0.04%  Size N +0.02%  Tradin -0.04%
    B -> C    change 2 (the lag), after 1  dR2 -0.017960  Volati -19.0%  Moment +2.0%  Size +1.0%  Size N +2.2%  Tradin +2.7%
                                         mean  Volati +1.69%  Moment -24.51%  Size +4.90%  Size N -2.77%  Tradin -2.56%
    A -> C    combined                     dR2 -0.018053  Volati -20.3%  Moment +2.3%  Size +0.9%  Size N +2.1%  Tradin +2.6%
                                         mean  Volati +1.58%  Moment -24.34%  Size +4.87%  Size N -2.75%  Tradin -2.60%
    A -> L    change 2 alone               dR2 -0.018080  Volati -19.1%  Moment +2.0%  Size +0.9%  Size N +2.2%  Tradin +2.6%
                                         mean  Volati +1.70%  Moment -24.39%  Size +4.88%  Size N -2.76%  Tradin -2.59%
    L -> C    change 1 after 2             dR2 +0.000027  Volati -1.4%  Moment +0.3%  Size -0.0%  Size N -0.1%  Tradin +0.0%
                                         mean  Volati -0.12%  Moment +0.05%  Size -0.01%  Size N +0.01%  Tradin -0.01%
    B -> Cx   lag: exposures               dR2 -0.017222  Volati -18.3%  Moment +2.5%  Size +1.0%  Size N +1.8%  Tradin +1.4%
                                         mean  Volati +3.36%  Moment -24.90%  Size +4.30%  Size N -2.62%  Tradin -2.28%
   Cx -> Cxw  lag: + weights               dR2 -0.000716  Volati -0.8%  Moment -0.7%  Size -0.0%  Size N +0.2%  Tradin +1.2%
                                         mean  Volati -1.67%  Moment +0.40%  Size +0.60%  Size N -0.14%  Tradin -0.28%
  Cxw -> C    lag: + membership            dR2 -0.000021  Volati -0.0%  Moment +0.1%  Size +0.0%  Size N +0.1%  Tradin +0.1%
                                         mean  Volati +0.01%  Moment -0.01%  Size -0.00%  Size N -0.01%  Tradin -0.00%
    C -> E    a 50% coverage rule          dR2 +0.000014  Volati +0.3%  Moment -0.8%  Size -0.1%  Size N -0.2%  Tradin -5.9%
                                         mean  Volati +0.01%  Moment +0.01%  Size -0.00%  Size N +0.00%  Tradin -0.00%

=== Residual spread, specific risk percentiles, floor lists
  A    spread 0.3804  p1 9.17% p10 14.87% p50 23.10% p90 36.12% p99 56.14%  min 3.54% max 104.36%  n 470
       spread shares FRC 61.1%, ECHO 4.0%, LUMN 2.7%, SIVB 2.6%, BNY 2.2%
       V^2 shares    MRNA 2.2%, FRC 2.1%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
       floored (characteristic forecast below the floor): ['JPM']; delivered the floor: none
       fallback (7): INFO, IPGP, PENN, PVH, UA, UAA, VNO
  A0   spread 0.3804  p1 9.17% p10 14.87% p50 23.10% p90 36.12% p99 56.14%  min 3.54% max 104.36%  n 470
       spread shares FRC 61.1%, ECHO 4.0%, LUMN 2.7%, SIVB 2.6%, BNY 2.2%
       V^2 shares    MRNA 2.2%, FRC 2.1%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
       floored (characteristic forecast below the floor): ['JPM']; delivered the floor: none
       fallback (7): INFO, IPGP, PENN, PVH, UA, UAA, VNO
  B    spread 0.3801  p1 9.18% p10 14.86% p50 23.11% p90 36.12% p99 55.92%  min 3.65% max 104.36%  n 470
       spread shares FRC 61.0%, ECHO 4.0%, LUMN 2.7%, SIVB 2.6%, BNY 2.2%
       V^2 shares    MRNA 2.2%, FRC 2.1%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
       floored (characteristic forecast below the floor): none; delivered the floor: none
       fallback (7): INFO, IPGP, PENN, PVH, UA, UAA, VNO
  Cx   spread 0.3716  p1 9.51% p10 15.11% p50 23.52% p90 36.49% p99 53.48%  min 9.02% max 105.02%  n 470
       spread shares FRC 70.9%, SIVB 4.4%, LUMN 3.1%, BNY 1.4%, DISH 1.3%
       V^2 shares    MRNA 2.3%, FRC 2.0%, ENPH 1.3%, ALB 1.0%, GL 0.9%
       floored (characteristic forecast below the floor): none; delivered the floor: none
       fallback (10): ECHO, INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VMRK, VNO
  Cxw  spread 0.3686  p1 9.51% p10 15.12% p50 23.52% p90 36.35% p99 53.30%  min 9.02% max 106.68%  n 470
       spread shares FRC 71.1%, SIVB 3.9%, LUMN 3.3%, BNY 1.5%, DISH 1.3%
       V^2 shares    MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
       floored (characteristic forecast below the floor): none; delivered the floor: none
       fallback (10): ECHO, INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VMRK, VNO
  C    spread 0.3679  p1 9.51% p10 15.11% p50 23.52% p90 36.36% p99 53.35%  min 9.02% max 106.68%  n 470
       spread shares FRC 71.3%, SIVB 4.0%, LUMN 2.9%, BNY 1.5%, DISH 1.4%
       V^2 shares    MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
       floored (characteristic forecast below the floor): none; delivered the floor: none
       fallback (10): ECHO, INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VMRK, VNO
  L    spread 0.3682  p1 9.51% p10 15.11% p50 23.51% p90 36.36% p99 53.35%  min 9.02% max 106.68%  n 470
       spread shares FRC 71.3%, SIVB 3.9%, LUMN 2.9%, BNY 1.5%, DISH 1.4%
       V^2 shares    MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
       floored (characteristic forecast below the floor): none; delivered the floor: none
       fallback (10): ECHO, INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VMRK, VNO
  E    spread 0.3682  p1 9.50% p10 15.06% p50 23.52% p90 36.34% p99 53.35%  min 9.02% max 106.66%  n 468
       spread shares FRC 71.3%, SIVB 4.0%, LUMN 2.9%, BNY 1.5%, DISH 1.4%
       V^2 shares    MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
       floored (characteristic forecast below the floor): none; delivered the floor: none
       fallback (8): INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO

=== All factors, A against C: annual vol, annual mean, noise share
  factor                    vol A    vol C   mean A   mean C  noise A  noise C
  Construction             17.38%   17.44%  +15.92%  +21.43%    49.3%    50.1%
  Finance & Real Estate    15.70%   15.69%  +26.64%  +22.67%    13.5%    13.8%
  Manufacturing            11.46%   11.25%  +27.73%  +21.80%    18.5%    19.7%
  Mining                   20.95%   20.64%  +24.90%  +12.37%    14.5%    15.2%
  Retail                   13.69%   13.34%  +22.79%  +13.88%     9.6%    10.3%
  SIC20                    15.38%   13.18%  +26.59%   +9.14%    20.9%    29.0%
  SIC28                    14.77%   13.54%  +30.69%  +15.99%     7.9%     9.6%
  SIC35                    17.10%   17.16%  +29.05%  +26.46%     6.4%     6.5%
  SIC36                    20.34%   20.26%  +21.01%  +28.05%     5.4%     5.5%
  SIC37                    14.29%   13.96%  +23.77%  +13.78%    17.4%    18.7%
  SIC38                    14.46%   14.49%  +27.89%  +15.62%    10.8%    11.0%
  SIC49                    14.56%   14.21%  +28.86%  +23.22%    14.2%    15.2%
  SIC60                    20.87%   20.59%  +26.88%  +21.73%     8.0%     8.4%
  SIC63                    15.85%   14.76%  +29.52%  +19.13%    11.9%    14.0%
  SIC67                    15.66%   15.11%  +31.12%  +17.41%    15.9%    17.5%
  SIC73                    16.55%   17.28%  +23.28%  +13.48%     3.8%     3.5%
  Services                 13.87%   13.16%  +17.96%  +14.31%    22.4%    25.4%
  Transport & Utilities    13.73%   13.40%  +21.32%  +13.28%    15.8%    16.9%
  Wholesale                15.73%   14.48%  +20.49%  +19.98%    34.7%    41.9%
  Volatility                9.55%    7.61%   +2.93%   +4.51%     3.6%     5.4%
  Momentum                  7.06%    7.22%  +26.78%   +2.44%     4.0%     3.9%
  Size                      7.51%    7.58%   +0.05%   +4.92%     9.8%     9.8%
  Size Nonlinearity         6.55%    6.69%   +1.31%   -1.44%    17.3%    17.0%
  Trading Activity          5.16%    5.30%   +6.51%   +3.91%    10.8%    10.5%

=== Style factor returns, A against C, over the 938 periods both estimate
  correlation  Volati +0.902  Moment +0.973  Size +0.985  Size N +0.983  Tradin +0.839
```

### mechanism.txt

mechanism.py: why Size's mean rises under the lag.

```text
Annualised mean factor return: exposures from day t (same day) | from day t-1 (lagged)
Intercept, sqrt-cap weights and membership from t-1 in both; no industries.
  Size                       Size +8.30% | +4.08%
  Mome                       Momentum +28.60% | +4.11%
  Size + Mome                Size -0.83% | +2.96%   Momentum +29.45% | +3.51%
  Size + Size                Size +8.35% | +8.01%
  Size + Mome + Vola         Size +0.22% | +2.70%   Momentum +28.48% | +3.05%
  Vola + Mome + Size + Size + Trad Momentum +28.48% | +3.21%   Size +1.46% | +4.84%

Members' cross-sectional correlation of the style exposures, median over the periods
                      Volatili  Momentum      Size  Size Non  Trading 
  Volatility            +1.000    -0.031    +0.025    -0.018    +0.564
  Momentum              -0.031    +1.000    +0.375    +0.356    -0.125
  Size                  +0.025    +0.375    +1.000    +0.933    -0.326
  Size Nonlinearity     -0.018    +0.356    +0.933    +1.000    -0.360
  Trading Activity      +0.564    -0.125    -0.326    -0.360    +1.000

How much of today's return each exposure holds: the members' cross-sectional
regression slope of the day-t exposure's change since t-1 on day t's return, median
  Volatility            -0.45 standard deviations of exposure per unit return
  Momentum              +3.17 standard deviations of exposure per unit return
  Size                  +0.86 standard deviations of exposure per unit return
  Size Nonlinearity     +0.88 standard deviations of exposure per unit return
  Trading Activity      -0.03 standard deviations of exposure per unit return
```

### details.txt

details.py: change 1 value by value; the membership read a day earlier; ECHO, MRSH, VMRK; the specific-risk extremes.

```text
1. Change 1, raw descriptors A (committed) against B (empty windows missing)
  descriptor  A had, B missing  all zeros in A  on members  members, before 1st price  any other change
  LNCAP                    0            True           0                          0                 0
  LCAPCB                   0            True           0                          0                 0
  LPRI                     0            True           0                          0                 0
  DASTD                    0            True           0                          0                 0
  HILO                     0            True           0                          0                 0
  CMRA                18,645            True       4,679                      4,334                 0
  BTSG                     0            True           0                          0                 0
  HALPHA                   0            True           0                          0                 0
  SERDP                    0            True           0                          0                 0
  RSTR                18,645            True       4,679                      4,334                 0
  STOM                22,512            True         705                          0                 0
  STOQ                20,780            True         567                          0                 0
  STOA                14,300            True         335                          0                 0
  VLVR                     0            True           0                          0                 0
  total 94,882 values, every one a zero in A; nothing else changed
  members before their first price: 12 securities, 7,232 member-days: BALL, COR, CPAY, ECHO, EG, ELV, GEN, MRSH, RVTY, SKYD, SW, VMRK

2. Membership read on the exposures' date, over C's estimated periods
  62 security-periods where the mask differs between t-1 and t: 61 on the day a name left, 1 on the day one joined
  43 of them had a return on t and a complete row on t-1, so the regression moved: 42 in (left), 1 out (joined)
    2023-03-15 SBNY   left    return on t False complete on t-1 False in the regression now False
    2023-03-15 SIVB   left    return on t False complete on t-1 False in the regression now False
    2023-03-20 LUMN   left    return on t True  complete on t-1 True  in the regression now True
    2023-05-04 FRC    left    return on t False complete on t-1 False in the regression now False
    2023-06-20 DISH   left    return on t True  complete on t-1 True  in the regression now True
    2023-08-25 AAP    left    return on t True  complete on t-1 True  in the regression now True
    2023-09-18 LNC    left    return on t True  complete on t-1 True  in the regression now True
    2023-09-18 NWL    left    return on t True  complete on t-1 True  in the regression now True
    2023-10-03 DXC    left    return on t True  complete on t-1 True  in the regression now True
    2023-10-18 ATVI   left    return on t False complete on t-1 False in the regression now False
    2023-10-18 OGN    left    return on t True  complete on t-1 True  in the regression now True
    2023-12-18 ALK    left    return on t True  complete on t-1 True  in the regression now True
    2023-12-18 SEDG   left    return on t True  complete on t-1 True  in the regression now True
    2023-12-18 SEE    left    return on t True  complete on t-1 True  in the regression now True
    2024-02-01 CDAY   left    return on t False complete on t-1 True  in the regression now False
    2024-03-18 WHR    left    return on t True  complete on t-1 True  in the regression now True
    2024-03-18 ZION   left    return on t True  complete on t-1 True  in the regression now True
    2024-04-03 VFC    left    return on t True  complete on t-1 True  in the regression now True
    2024-04-03 XRAY   left    return on t True  complete on t-1 True  in the regression now True
    2024-05-08 PXD    left    return on t False complete on t-1 False in the regression now False
    2024-06-24 CMA    left    return on t True  complete on t-1 True  in the regression now True
    2024-06-24 ILMN   left    return on t True  complete on t-1 True  in the regression now True
    2024-06-24 RHI    left    return on t True  complete on t-1 True  in the regression now True
    2024-09-23 AAL    left    return on t True  complete on t-1 True  in the regression now True
    2024-09-23 BIO    left    return on t True  complete on t-1 True  in the regression now True
    2024-09-23 ETSY   left    return on t True  complete on t-1 True  in the regression now True
    2024-10-01 BBWI   left    return on t True  complete on t-1 True  in the regression now True
    2024-11-26 MRO    left    return on t False complete on t-1 False in the regression now False
    2024-12-23 CTLT   left    return on t False complete on t-1 False in the regression now False
    2024-12-23 QRVO   left    return on t True  complete on t-1 True  in the regression now True
    2025-03-24 BWA    left    return on t True  complete on t-1 True  in the regression now True
    2025-03-24 CE     left    return on t True  complete on t-1 True  in the regression now True
    2025-03-24 FMC    left    return on t True  complete on t-1 True  in the regression now True
    2025-03-24 TFX    left    return on t True  complete on t-1 True  in the regression now True
    2025-05-19 DFS    left    return on t False complete on t-1 True  in the regression now False
    2025-07-09 JNPR   left    return on t False complete on t-1 False in the regression now False
    2025-07-18 ANSS   left    return on t False complete on t-1 False in the regression now False
    2025-07-23 HES    left    return on t False complete on t-1 False in the regression now False
    2025-08-28 WBA    left    return on t False complete on t-1 True  in the regression now False
    2025-09-22 CZR    left    return on t True  complete on t-1 True  in the regression now True
    2025-09-22 ENPH   left    return on t True  complete on t-1 True  in the regression now True
    2025-09-22 MKTX   left    return on t True  complete on t-1 True  in the regression now True
    2025-10-31 KMX    left    return on t True  complete on t-1 True  in the regression now True
    2025-11-04 EMN    left    return on t True  complete on t-1 True  in the regression now True
    2025-11-28 IPG    left    return on t False complete on t-1 True  in the regression now False
    2025-12-11 K      left    return on t False complete on t-1 True  in the regression now False
    2025-12-22 LKQ    left    return on t True  complete on t-1 True  in the regression now True
    2025-12-22 MHK    left    return on t True  complete on t-1 True  in the regression now True
    2026-03-23 LW     left    return on t True  complete on t-1 True  in the regression now True
    2026-03-23 MTCH   left    return on t True  complete on t-1 True  in the regression now True
    2026-03-23 PAYC   left    return on t True  complete on t-1 True  in the regression now True
    2026-04-09 HOLX   left    return on t False complete on t-1 False in the regression now False
    2026-05-07 CTRA   left    return on t False complete on t-1 True  in the regression now False
    2026-06-02 EPAM   left    return on t True  complete on t-1 True  in the regression now True
    2026-06-22 CPB    left    return on t True  complete on t-1 True  in the regression now True
    2026-06-22 POOL   left    return on t True  complete on t-1 True  in the regression now True
    2026-06-30 CAG    left    return on t True  complete on t-1 True  in the regression now True
    2026-08-05 EA     left    return on t False complete on t-1 True  in the regression now False
    2026-08-18 AVB    left    return on t False complete on t-1 False in the regression now False
    2026-09-21 ILMN   joined  return on t True  complete on t-1 True  in the regression now False
    2026-09-21 TAP    left    return on t True  complete on t-1 True  in the regression now True
    2026-10-06 CTVA   left    return on t True  complete on t-1 True  in the regression now True

3. Out of every regression under the lag
  ECHO: 93 priced day(s) (2026-06-24 .. 2026-11-03), 1 complete style row(s), a member on 1214 day(s); same-day regressions 1: 2026-11-03
      2026-11-03: price on t True, on t-1 True; complete row on t True, on t-1 False
  MRSH: 203 priced day(s) (2026-01-14 .. 2026-11-03), 1 complete style row(s), a member on 1214 day(s); same-day regressions 1: 2026-11-03
      2026-11-03: price on t True, on t-1 True; complete row on t True, on t-1 False
  VMRK: 55 priced day(s) (2026-08-18 .. 2026-11-03), 1 complete style row(s), a member on 1214 day(s); same-day regressions 1: 2026-11-03
      2026-11-03: price on t True, on t-1 True; complete row on t True, on t-1 False

4. The extremes of the delivered specific risk
  A: lowest INFO 3.54%, WEC 8.88%, LNT 9.03%, DTE 9.12%; highest SBNY 71.17%, LUMN 72.75%, MRNA 104.36%
  C: lowest DUK 9.02%, WEC 9.04%, LNT 9.11%, CMS 9.21%; highest SBNY 70.04%, LUMN 70.20%, MRNA 106.68%
```

### floor_serdp.txt

floor_serdp.py: JPM and the floor; SERDP in the forecast Volatility exposure; the blends' composition over time.

```text
Specific-risk floor: floor_fraction 0.1 x kappa x S. A forecast is floored when its
characteristic forecast kappa (1 + V) S falls below it, that is when 1 + V < 0.1.

A0: forecast level S 25.2371%, kappa 1.4022, floor 3.54%; gamma Volatility +0.442, Momentum -0.019, Size -0.134, Size Nonlinearity +0.021, Trading Activity -0.065
  JPM   1+V +0.0767  characteristic 2.72%  delivered 12.83%  (Vola -1.39, Mome -0.02, Size +0.45, Size +0.57, Trad -0.66)
  INFO  1+V +0.0370  characteristic 1.31%  delivered 3.54%  (Vola -3.23, Mome -0.26, Size -4.00, Size -3.25, Trad -2.21)
  BRK.B 1+V +0.4300  characteristic 15.22%  delivered 13.38%  (Vola -1.03, Mome -0.07, Size +0.34, Size +0.57, Trad -0.54)
  AAPL  1+V +0.7454  characteristic 26.38%  delivered 15.51%  (Vola +0.23, Mome -0.03, Size +1.77, Size +0.57, Trad -0.22)
  MSFT  1+V +0.6691  characteristic 23.68%  delivered 16.57%  (Vola +0.03, Mome -0.71, Size +1.69, Size +0.57, Trad -0.12)

B: forecast level S 25.2371%, kappa 1.4028, floor 3.54%; gamma Volatility +0.420, Momentum -0.018, Size -0.133, Size Nonlinearity +0.022, Trading Activity -0.064
  JPM   1+V +0.1026  characteristic 3.63%  delivered 12.84%  (Vola -1.39, Mome -0.02, Size +0.45, Size +0.57, Trad -0.66)
  INFO  1+V +0.1030  characteristic 3.65%  delivered 3.65%  (Vola -3.23, Mome -0.26, Size -4.00, Size -3.25, Trad -2.21)
  BRK.B 1+V +0.4521  characteristic 16.01%  delivered 13.38%  (Vola -1.03, Mome -0.07, Size +0.34, Size +0.57, Trad -0.54)
  AAPL  1+V +0.7445  characteristic 26.36%  delivered 15.51%  (Vola +0.23, Mome -0.03, Size +1.77, Size +0.57, Trad -0.22)
  MSFT  1+V +0.6712  characteristic 23.76%  delivered 16.57%  (Vola +0.03, Mome -0.71, Size +1.69, Size +0.57, Trad -0.12)

C: forecast level S 25.1624%, kappa 1.3836, floor 3.48%; gamma Volatility +0.188, Momentum -0.031, Size -0.082, Size Nonlinearity +0.003, Trading Activity +0.053
  JPM   1+V +0.3190  characteristic 11.11%  delivered 13.17%  (Vola -1.39, Mome -0.02, Size +0.45, Size +0.57, Trad -0.66)
  INFO  1+V +0.4988  characteristic 17.37%  delivered 17.37%  (Vola -3.23, Mome -0.26, Size -4.00, Size -3.25, Trad -2.21)
  BRK.B 1+V +0.5971  characteristic 20.79%  delivered 13.39%  (Vola -1.03, Mome -0.07, Size +0.34, Size +0.57, Trad -0.54)
  AAPL  1+V +0.8001  characteristic 27.86%  delivered 15.53%  (Vola +0.23, Mome -0.03, Size +1.77, Size +0.57, Trad -0.22)
  MSFT  1+V +0.7804  characteristic 27.17%  delivered 17.28%  (Vola +0.03, Mome -0.71, Size +1.69, Size +0.57, Trad -0.12)

SERDP in the forecast Volatility exposure, on C's descriptors and the last date
  SERDP has values on 1 date(s): ['2026-11-03']
  the model's forecast Volatility exposures are the six-descriptor blend: True
  members        n 426  correlation 0.9643  mean |diff| 0.211  rms 0.273  max |diff| 1.007
  all with both  n 492  correlation 0.9122  mean |diff| 0.265  rms 0.449  max |diff| 3.600
  largest moves among members: GL -0.82 -> -1.82, MPWR +2.76 -> +1.87, GEN +0.42 -> +1.31, CMI +0.12 -> -0.73, NVR +0.96 -> +0.16
  SERDP itself among members on the last date: n 426, correlation with the five-descriptor Volatility +0.184
  BTSG   in 386 of the 938 estimated periods' exposures, from 2025-04-23
  HALPHA in 386 of the 938 estimated periods' exposures, from 2025-04-23
  VLVR   in 386 of the 938 estimated periods' exposures, from 2025-04-23
  DASTD  in 938 of the 938 estimated periods' exposures, from 2023-02-08
  CMRA   in 938 of the 938 estimated periods' exposures, from 2023-02-08
  RSTR   in 938 of the 938 estimated periods' exposures, from 2023-02-08
```

### coverage_check.txt

coverage_check.py: the report's 50% counts against what the rule removes.

```text
descriptor  removed  reported  on members  reported
DASTD       1,612     1,612         533       533
CMRA        4,648     4,648       1,339     1,339
BTSG        5,009     5,009       1,111     1,111
HALPHA      5,009     5,009       1,111     1,111
SERDP          12        12           5         5
RSTR        4,847     4,847       1,339     1,339
STOM          467       467          66        66
STOQ        1,363     1,363         189       189
STOA        3,931     3,931         423       423
VLVR        2,608     2,608         970       970

the run report's independent count matches what the rule removes: True
```

### perturb_real.txt

perturb_real.py: the look-ahead guard on the real panel.

```text
repriced 2025-06-02 (row 855); 24 exposure columns, industries included

lag 1: period 2025-06-02
  exposures      bit-identical: True
  weights        bit-identical: True
  members        bit-identical: True
  in_regression  bit-identical: True
  returns moved for 479 of 479 securities with a return
  every earlier period's design bit-identical: True
  in that period's regression: 446 securities

lag 0: period 2025-06-02
  exposures      bit-identical: False
  weights        bit-identical: False
  members        bit-identical: True
  in_regression  bit-identical: True
  returns moved for 479 of 479 securities with a return
  every earlier period's design bit-identical: True
  in that period's regression: 446 securities

model (lag 1): 579 periods before 2025-06-02; their factor returns bit-identical: True; 2025-06-02's moved: True; the industry scheme the same: True
```

### shares_fill.txt

shares_fill.py: how the sampled share counts fill.

```text
1214 days x 506 securities; 21 sample dates every 63 trading days: 2022-01-03, 2022-04-04, ... 2026-10-12, 2026-11-03

between samples, 603,658 security-days: 0 differ from the sample at or before them
every security's first share count falls on a sample date: True
no share count at any sample: 1 (SKYD)
first share count after the panel's first day: 11 securities, e.g. BALL 2023-01-04, ELV 2023-01-04, GEN 2023-01-04, COR 2024-01-05, EG 2024-01-05, RVTY 2024-01-05, CPAY 2025-01-07, SW 2025-01-07
priced days before the security's first share count (cap missing, not back-filled): 1,401

sample-to-sample changes in the count: 2,357
splits (traded/adjusted ratio steps by 1.5x or more): 36; security-days between a split and the next sample, when the cap pairs a post-split price with a pre-split count: 896
  2022-03-24 WRB    factor   1.50, 7 trading day(s) to the next sample
  2022-06-06 AMZN   factor  20.00, 20 trading day(s) to the next sample
  2022-06-13 DXCM   factor   4.00, 15 trading day(s) to the next sample
  2022-06-23 FTNT   factor   5.00, 8 trading day(s) to the next sample
  2022-07-18 GOOG   factor  20.00, 55 trading day(s) to the next sample
  2022-07-18 GOOGL  factor  20.00, 55 trading day(s) to the next sample
  2022-08-25 TSLA   factor   3.00, 27 trading day(s) to the next sample
  2022-08-29 NDAQ   factor   3.00, 25 trading day(s) to the next sample
  2022-11-04 CPRT   factor   2.00, 40 trading day(s) to the next sample
  2022-11-30 TECH   factor   4.00, 23 trading day(s) to the next sample
  2023-03-28 MNST   factor   2.00, 6 trading day(s) to the next sample
  2023-08-22 CPRT   factor   2.00, 31 trading day(s) to the next sample

of the 896 stale security-days: 621 on a date an estimated period reads its exposures and weights from, 621 of them on index members
every split, its date, and the trading days until the next sample:
  2022-03-24 WRB      1.50    7
  2022-06-06 AMZN    20.00   20
  2022-06-13 DXCM     4.00   15
  2022-06-23 FTNT     5.00    8
  2022-07-18 GOOG    20.00   55
  2022-07-18 GOOGL   20.00   55
  2022-08-25 TSLA     3.00   27
  2022-08-29 NDAQ     3.00   25
  2022-11-04 CPRT     2.00   40
  2022-11-30 TECH     4.00   23
  2023-03-28 MNST     2.00    6 (read by an estimated period)
  2023-08-22 CPRT     2.00   31 (read by an estimated period)
  2023-08-22 SRE      2.00   31 (read by an estimated period)
  2024-02-20 COO      4.00   33 (read by an estimated period)
  2024-02-26 WMT      3.00   29 (read by an estimated period)
  2024-03-28 ODFL     2.00    6 (read by an estimated period)
  2024-06-10 NVDA    10.00   19 (read by an estimated period)
  2024-06-12 APH      2.00   17 (read by an estimated period)
  2024-06-26 CMG     50.00    8 (read by an estimated period)
  2024-07-15 AVGO    10.00   59 (read by an estimated period)
  2024-09-12 CTAS     4.00   17 (read by an estimated period)
  2024-10-03 LRCX    10.00    2 (read by an estimated period)
  2024-12-04 ANET     4.00   22 (read by an estimated period)
  2024-12-13 ETR      2.00   15 (read by an estimated period)
  2024-12-20 TSCO     5.00   10 (read by an estimated period)
  2025-05-22 FAST     2.00   33 (read by an estimated period)
  2025-06-10 ORLY    15.00   21 (read by an estimated period)
  2025-11-03 DD       2.39   46 (read by an estimated period)
  2025-11-17 NFLX    10.00   36 (read by an estimated period)
  2025-12-18 NOW      5.00   14 (read by an estimated period)
  2026-01-15 AMCR     0.20   59 (read by an estimated period)
  2026-04-06 BKNG    25.00    5 (read by an estimated period)
  2026-06-12 KLAC    10.00   20 (read by an estimated period)
  2026-06-24 DD       0.33   13 (read by an estimated period)
  2026-08-11 MNST     2.00   43 (read by an estimated period)
  2026-09-03 APH      2.00   26 (read by an estimated period)
```

### teeth.txt

the new descriptor tests run against the committed code.

```text
>               assert np.isnan(values[59:100, 0]).all(), name
E               AssertionError: CMRA
E               assert np.False_
test_descriptors_on_head.py:730: AssertionError
>           assert np.isnan(values[empty]).all(), name
E           AssertionError: RSTR
E           assert np.False_
test_descriptors_on_head.py:776: AssertionError
FAILED test_descriptors_on_head.py::test_a_security_whose_prices_start_after_the_window_gets_missing_for_every_one
FAILED test_descriptors_on_head.py::test_no_window_descriptor_is_computed_from_an_empty_window_anywhere
2 failed, 43 passed in 11.09s
committed code, S00 at period 99 (no price yet) and S01 at the last period (gone):
  DASTD  S00@99 np.float64(nan)        S01@last np.float64(nan)
  CMRA   S00@99 np.float64(0.0)        S01@last np.float64(0.0)
  BTSG   S00@99 np.float64(nan)        S01@last np.float64(nan)
  HALPHA S00@99 np.float64(nan)        S01@last np.float64(nan)
  SERDP  S00@99 np.float64(nan)        S01@last np.float64(nan)
  RSTR   S00@99 np.float64(0.0)        S01@last np.float64(0.0)
  STOM   S00@99 np.float64(0.0)        S01@last np.float64(0.0)
  STOQ   S00@99 np.float64(0.0)        S01@last np.float64(0.0)
  STOA   S00@99 np.float64(0.0)        S01@last np.float64(0.0)
  STO5   S00@99 np.float64(0.0)        S01@last np.float64(0.0)
  VLVR   S00@99 np.float64(nan)        S01@last np.float64(nan)
```

### screen_compare.txt

the descriptor screen in A, A0 (the new screen rule on A's descriptors), B and C.

```text
== variant A
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         97%      386    8.39%      9.76%     1.88
  DASTD        93%    1,149    6.13%      7.78%     1.35
  RSTR        100%      938    5.29%      7.08%     1.11
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         99%      938    3.38%      5.69%     1.39
  HALPHA       97%      386    3.32%      6.09%    -0.28
  STOQ        100%    1,145    2.99%      5.22%     1.26
  STOM        100%    1,191    2.95%      5.26%     1.27
  STOA        100%      938    2.56%      4.86%     1.52
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  LPRI         93%    1,213    1.77%      3.77%    -0.40
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  VLVR         95%      386    1.50%      3.92%     0.64
  SERDP           -        0   too few usable periods
== variant A0
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         97%      386    8.39%      9.76%     1.88
  DASTD        93%    1,149    6.13%      7.78%     1.35
  RSTR        100%      938    5.29%      7.08%     1.10
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         99%      938    3.38%      5.69%     1.39
  HALPHA       97%      386    3.32%      6.10%    -0.29
  STOQ        100%    1,145    2.99%      5.22%     1.25
  STOM        100%    1,191    2.95%      5.25%     1.28
  STOA        100%      938    2.56%      4.86%     1.51
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  LPRI         93%    1,213    1.77%      3.77%    -0.41
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  VLVR         95%      386    1.50%      3.92%     0.64
  SERDP           -        0   too few usable periods
== variant B
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         97%      386    8.39%      9.76%     1.88
  DASTD        93%    1,149    6.13%      7.78%     1.35
  RSTR         95%      938    5.30%      7.11%     1.11
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         94%      938    3.38%      5.58%     1.39
  HALPHA       97%      386    3.32%      6.10%    -0.29
  STOQ         93%    1,145    2.99%      5.21%     1.26
  STOM         93%    1,191    2.95%      5.25%     1.28
  STOA         95%      938    2.56%      4.85%     1.51
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  LPRI         93%    1,213    1.77%      3.77%    -0.41
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  VLVR         95%      386    1.50%      3.92%     0.64
  SERDP           -        0   too few usable periods
== variant C
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         97%      386    8.39%      9.76%     1.88
  DASTD        93%    1,149    6.13%      7.78%     1.35
  RSTR         95%      938    5.30%      7.11%     1.11
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         94%      938    3.38%      5.58%     1.39
  HALPHA       97%      386    3.32%      6.10%    -0.29
  STOQ         93%    1,145    2.99%      5.21%     1.26
  STOM         93%    1,191    2.95%      5.25%     1.28
  STOA         95%      938    2.56%      4.85%     1.51
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  LPRI         93%    1,213    1.77%      3.77%    -0.41
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  VLVR         95%      386    1.50%      3.92%     0.64
  SERDP           -        0   too few usable periods
```

### industry_means.txt

the industries' average mean return per variant.

```text
A average industry mean +25.08%  (n=19)
B average industry mean +25.07%  (n=19)
Cx average industry mean +20.28%  (n=19)
Cxw average industry mean +18.10%  (n=19)
C average industry mean +18.09%  (n=19)
L average industry mean +18.09%  (n=19)
```

### cmra_gap.txt

RSTR against CMRA coverage on B's descriptors.

```text
RSTR finite 456489 CMRA finite 451154 RSTR without CMRA 5335 CMRA without RSTR 0
```

### without_compare.txt

key lines, HEAD against the working tree, without --membership.

```text
== head_without.txt
  493 of 506 securities are in at least one estimated period's regression; every count below is over them
  mean cross-sectional R-squared 0.304
  Volatility                 10.50%        3.5%      939          0      0
  Momentum                    7.36%        3.8%      939          0      0
  Size                        7.65%        9.6%      939          0      0
  Size Nonlinearity           6.36%       18.0%      939          0      0
  Trading Activity            5.30%       11.7%      939          0      0
  specific risk   median 23.32%   p10 14.91%   p90 36.38%
                    relative residual sd 0.493, largest shares of its square: SBNY 43.5%, FRC 30.3%, ECHO 1.7%, SIVB 1.2%, BNY 0.9%
  - 2 security(ies) have a characteristic forecast below the specific-risk floor, their linear V having run past the end of its validity; blended with their own residual history, no forecast delivered is the floor: BNY, JPM
== final_without.txt
  window descriptors need any valid observation; from a window with none, over 3,610,993 values: none
  490 of 506 securities are in at least one estimated period's regression; every count below is over them
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.286
  ! Volatility: SERDP is in 492 securities' forecast exposures and in none of the 938 estimated periods', so the factor was estimated on a blend without it
  Volatility                  8.13%      +5.32%        5.3%      938          0      0
  Momentum                    7.81%      +2.94%        3.8%      938          0      0
  Size                        7.64%      +4.96%        9.9%      938          0      0
  Size Nonlinearity           6.48%      -1.04%       17.8%      938          0      0
  Trading Activity            5.30%      +4.04%       11.5%      938          0      0
  specific risk   median 23.69%   p10 14.98%   p90 36.78%
                    relative residual sd 0.489, largest shares of its square: SBNY 48.5%, FRC 33.6%, SIVB 1.9%, LUMN 1.4%, SEDG 1.1%
  - 1 security(ies) have a characteristic forecast below the specific-risk floor, their linear V having run past the end of its validity; blended with their own residual history, no forecast delivered is the floor: JPM
  - 3 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: ECHO, MRSH, VMRK
```

### timings.txt

run times.

```text
head_with.txt (HEAD, --industries --membership):
elapsed 0m54.265s
new_with.txt (working tree, same flags):
elapsed 1m0.274s
decompose.py, seven variants: 15:47:30 -> 15:54:12 by file times, about 57 s a variant
```

---

## A separate change made in the same session

Jeff's correction, recorded where the standing instructions live: the mid-office briefs are
written in Chat and pasted in, and a `BRIEF_` file says so. `claude/CLAUDE_CODE_RULES.md` v1.6
(Rule 43's "Where" item and a changelog line) and `.claude/commands/retro.md`. Its own change,
with its own commit and file list, and not in the `git add` below:

```
git add claude/CLAUDE_CODE_RULES.md .claude/commands/retro.md
```

---

## git add -- this change

Every file of the two fixes, tracked and new, this retro included, and nothing else:

```
git add src/praxis/midoffice/model/descriptors.py src/praxis/midoffice/model/crosssectional.py scripts/run_sp500_model.py tests/midoffice/test_crosssectional.py tests/midoffice/test_descriptors.py tests/midoffice/test_lookahead.py docs/FACTOR_MODEL.md claude/handoffs/BRIEF_midoffice02_empty_windows_and_lag.md claude/retros/RETRO_midoffice02_empty_windows_and_lag.md
```
