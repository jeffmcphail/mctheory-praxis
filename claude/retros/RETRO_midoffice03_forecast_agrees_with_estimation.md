# Retro: Mid-office 03 -- the forecast agrees with the estimation, and two definitions measured

**Brief:** `claude/handoffs/BRIEF_midoffice03_forecast_agrees_with_estimation.md` -- written in
Chat and pasted into the Code session, transcribed verbatim
**Date:** 2026-11-06
**Duration:** about 36 minutes of implementation and measurement (12:06-12:43 EST: from
the brief's arrival, by the transcript, to the last measurement, by the scratch file times), then this retro
**Status:** COMPLETE, uncommitted. All six items done in the brief's order, each measured
against the one before; the split rule is recorded for the rebuild, not implemented. Neither
descriptor default was changed. Decisions left for Jeff are listed below.

---

## Summary

Five changes, measured in the brief's order, and a separate one-line correction. (1)
`min_coverage` now defaults to 0.5. That is the committed code run at 0.5, bit for bit: mean
R-squared +0.000014, Trading Activity's volatility -5.9%, and ECHO and VMRK lose their only
complete row. (2) SERDP is in no blend, so every forecast exposure is built from the descriptors
the last estimated period's were. The run's old flag is now a guard that stops the run, and
putting SERDP back stops it. A standing line shows each index's blend over the estimated periods,
with the 2025-04-23 entry of BTSG, HALPHA and VLVR. The estimation moves only by rounding (at
most 2.5e-16 in a factor return); the forecast Volatility exposures move by 0.276 rms. (3)
`fit_index_weights` pairs each descriptor with the next period's return. It stays whole-sample,
and its docstring says so. The run is untouched. On the real panel the lag turns Momentum's fit
from RSTR +2.41 / HALPHA -1.41 into -0.63 / +1.63. (4) RSTR takes a skip, default 0. At 21,
Momentum's mean rises from +2.44% to +3.23% a year (t 0.81 to 1.09) and its volatility moves from
7.16% to 7.11%. (5) The market model takes a half-life, default none. USE4's 252 days and
half-life 63 put BTSG, HALPHA and VLVR in all of the default's 938 periods, against 386 now. On
those 938 periods, mean R-squared rises from 0.2942 to 0.2994 and Volatility's volatility from
7.63% to 8.63%. The 252-day window does most of it. (6) midoffice01's brief and retro now say
the brief was written in Chat and pasted in. Tests: 2,225 passed, 1 skipped (4 new); midoffice
774.

## What was asked

The brief, verbatim (it is also the `BRIEF_` file):

```text
committed -- midoffice02, then the Rule 43 correction on its own.

Next piece, midoffice03: make the forecast agree with the estimation,
and measure two descriptor definitions before anyone chooses them. Same
discipline as before: in this order, each change decomposed against the
one before it, retro and git add line at the end, then stop.

1. Coverage: set min_coverage's default to 0.5. A twelve-month
   descriptor computed from under six months of its window is not that
   descriptor, and the measured cost is R-squared +0.000014 and two
   phantom identities, ECHO and VMRK, that the rebuild will fix. 0
   stays available as a value.

2. SERDP: drop it from the forecast blend, so every forecast exposure
   is built from the descriptors the last estimated period's exposures
   were built from. Turn the run's flag into a guard that stops the
   run: no descriptor may contribute to a forecast exposure if it
   contributed to no estimated period's. Computing SERDP historically
   waits for a reason to; it has never had a usable period in the
   screen. Add a standing line giving each risk index's blend
   composition over the estimated periods, with the date each
   descriptor enters, so the 2025-04-23 change is visible in every run.

3. fit_index_weights: lag it now -- descriptor on t, return on t+1 --
   though the run does not call it, so it cannot bring back what the lag
   removed. Leave it whole-sample, say so in its docstring, and keep
   test_lookahead's third test as the record of it.

4. RSTR: add a skip parameter, the trading days excluded from the end
   of its window, default 0, which is today's definition. Report
   Momentum's vol, mean and t at 0 and at 21, USE4's skip. Do not change
   the default.

5. The market model: BTSG, HALPHA and VLVR first reach the estimated
   exposures on 2025-04-23, so for 552 of 938 periods Volatility has no
   beta, the descriptor the screen ranks first. Report the market
   model's window and half-life, and rerun the screen and the model
   with the market model on 252 trading days and half-life 63, as USE4
   estimates beta: when each of the three first appears, the screen,
   R-squared and the style vols. Do not change the default.

6. As a separate one-line change with its own git add: correct the
   Source lines in midoffice01's brief and retro to say the brief was
   written in Chat and pasted in.

Not now, but record it for the rebuild: share counts must change on a
split's execution date, using the split events rather than the price
adjustment factor -- DD's 2.39 and 0.33 suggest that factor also
carries actions that are not splits. Check each applied split against
the next share sample, flag any that disagree, and guard on zero
member-days pairing a post-split price with a pre-split count.
```

Standing constraints carried from earlier rulings: do not touch the size pair; Code does not
commit (Rule 43); scratch helpers in gitignored `claude/scratch/`; ASCII only.

## Files modified

| file | change | lines (working tree) |
|---|---|---|
| `src/praxis/midoffice/model/descriptors.py` | `min_coverage` default 0.5 and its docstring; `rstr_skip`; `beta_half_life` and the weighted market model; `blend_composition` (new); `fit_index_weights` lagged, its whole-sample fit stated | 261-267, 272-276, 283-285, 295-301, 306-319, 336-338, 362-385, 399-416, 534-540, 544-548, 575, 655-657, 863-896, 903, 912-924, 929-930, 942-956 |
| `scripts/run_sp500_model.py` | `--min-coverage` default 0.5 and the window guard under the rule; `UNBLENDED` and `without` (SERDP out of every blend); the blend-composition line and the stop; `--rstr-skip`; `--beta-days`, `--beta-half-life` and the standing market-model line | 23, 45, 63-79, 209-215, 221-222, 231-242, 251-257, 261-262, 366-383, 447-451, 456-466, 470-499, 533, 538-541, 545-549, 638-642, 667-722 |
| `tests/midoffice/test_descriptors.py` | 4 new tests; 4 changed for the new default and the lag | 27, 134-170, 180-224, 449-479, 483-485, 493-496, 511-542, 813-814, 819-825, 873-874, 878-894, 941-946, 954, 964-968 |
| `tests/midoffice/test_lookahead.py` | the third test's docstring: lagged, still whole-sample | 175-179 |
| `docs/FACTOR_MODEL.md` | SERDP and the guard, `fit_index_weights`, the 0.5 default, the two definitions measured, the split rule for the rebuild | 411-426, 629-667, 690-699 |
| `claude/handoffs/BRIEF_midoffice03_forecast_agrees_with_estimation.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice03_forecast_agrees_with_estimation.md` | new: this retro | new |
| **separate change** `claude/handoffs/BRIEF_midoffice01_standardise_over_members.md` | Source line: written in Chat and pasted in | 5 |
| **separate change** `claude/retros/RETRO_midoffice01_standardise_over_members.md` | Brief line: written in Chat and pasted in | 3-4 |

## How it was measured

Every variant is `scripts/run_sp500_model.py <panel> --industries --membership`, run from a
snapshot of the source tree taken after each change (`tree_0` = HEAD 6e29b2c exported with
`git archive`, then `tree_1` .. `tree_5`). The runs used `main()` itself, with the estimator,
`compute_descriptors`, the screen and `build_risk_indices` wrapped to keep what they were handed.
Each variant's report, summary and arrays are saved. "Bit for bit" means the factor-return
array, the style exposures on every date, every forecast exposure and every specific risk
compare equal byte for byte (`exact.py`). The runner first reproduced midoffice02's final run
exactly: HEAD's report is identical, line for line, to `final_with.txt` from midoffice02, and
its summary to midoffice02's variants C and E.

| variant | what it is |
|---|---|
| H | HEAD 6e29b2c as committed: `min_coverage` 0 |
| H50 | HEAD run with `--min-coverage 0.5` |
| V1_0 | change 1's tree run with `--min-coverage 0` |
| V1 | change 1: `min_coverage` default 0.5 |
| V2 | change 2: SERDP in no blend; the guard; the composition line |
| V3 | change 3: `fit_index_weights` lagged |
| V4 | change 4: `rstr_skip`, at its default 0 |
| V4s21 | V4 with `--rstr-skip 21` |
| V4b0 | the other reading's matched baseline: RSTR at 0 on V4b21's cells (917 periods) |
| V4b21 | the other reading: twelve months ending 21 days back, matched (917 periods) |
| V5 | change 5: `beta_half_life`, at its default none (the defaults throughout) |
| V5w | V5 with `--beta-days 252` |
| V5u | V5 with `--beta-days 252 --beta-half-life 63` |
| V5w938 | V5w held to the default's 938 periods |
| V5u938 | V5u held to the default's 938 periods |
| F | the finished working tree, the defaults (must be V5) |
| F_nomem | the finished tree with `--industries` alone, no membership |

## Results

### Each change against the one before, at the defaults

V3, V4 and V5 are V2 bit for bit (the run never calls `fit_index_weights`, and changes 4 and 5
keep today's definitions by default), so one column stands for all four.

| | HEAD (min_coverage 0) | change 1 | change 2 = 3 = 4 = 5, the defaults |
|---|---|---|---|
| mean R-squared | 0.294197 | 0.294211 | 0.294211 |
| periods estimated | 938 | 938 | 938 |
| first estimated period | 2023-02-08 | 2023-02-08 | 2023-02-08 |
| **annual vol** | | | |
| Volatility | 7.61% | 7.63% | 7.63% |
| Momentum | 7.22% | 7.16% | 7.16% |
| Size | 7.58% | 7.57% | 7.57% |
| Size Nonlinearity | 6.69% | 6.67% | 6.67% |
| Trading Activity | 5.30% | 4.99% | 4.99% |
| **annual mean return (t-stat)** | | | |
| Volatility | +4.51% (+1.36) | +4.52% (+1.37) | +4.52% (+1.37) |
| Momentum | +2.44% (+0.81) | +2.44% (+0.81) | +2.44% (+0.81) |
| Size | +4.92% (+1.54) | +4.92% (+1.54) | +4.92% (+1.54) |
| Size Nonlinearity | -1.44% (-0.54) | -1.44% (-0.54) | -1.44% (-0.54) |
| Trading Activity | +3.91% (+2.11) | +3.90% (+2.14) | +3.90% (+2.14) |
| **noise share** | | | |
| Volatility | 5.4% | 5.3% | 5.3% |
| Momentum | 3.9% | 4.0% | 4.0% |
| Size | 9.8% | 9.9% | 9.9% |
| Size Nonlinearity | 17.0% | 17.1% | 17.1% |
| Trading Activity | 10.5% | 11.5% | 11.5% |
| residual spread (relative residual sd) | 0.3679 | 0.3682 | 0.3682 |
| its largest shares | FRC 71.3%, SIVB 4.0%, LUMN 2.9% | FRC 71.3%, SIVB 4.0%, LUMN 2.9% | FRC 71.3%, SIVB 4.0%, LUMN 2.9% |
| specific risk p1 / p10 / p50 | 9.51% / 15.11% / 23.52% | 9.50% / 15.06% / 23.52% | 9.50% / 15.06% / 23.53% |
| specific risk p90 / p99 | 36.36% / 53.35% | 36.34% / 53.35% | 36.35% / 54.52% |
| specific risk min / max | 9.02% / 106.68% | 9.02% / 106.66% | 9.02% / 106.68% |
| forecasts | 470 | 468 | 468 |
| floor list: characteristic forecast below the floor | none | none | none |
| floor list: delivered the floor | none | none | none |
| regression universe | 482 | 482 | 482 |
| fallback (in no regression) | 10: ECHO, INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VMRK, VNO | 8: INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO | 8: INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO |

### The steps

| from -> to | step | dR-squared | Volatility | Momentum | Size | Size Nonlinearity | Trading Activity |
|---|---|---|---|---|---|---|---|
| H -> V1 | change 1: min_coverage 0.5: vol | +0.000014 | +0.3% | -0.8% | -0.1% | -0.2% | -5.9% |
| | mean, points a year | | +0.01 | +0.01 | -0.00 | +0.00 | -0.00 |
| V1 -> V2 | change 2: SERDP out: vol | +0.000000 | -0.0% | -0.0% | +0.0% | +0.0% | -0.0% |
| | mean, points a year | | +0.00 | -0.00 | -0.00 | -0.00 | -0.00 |
| V2 -> V3 | change 3: fit_index_weights: vol | +0.000000 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| | mean, points a year | | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| V3 -> V4 | change 4: default: vol | +0.000000 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| | mean, points a year | | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| V4 -> V5 | change 5: default: vol | +0.000000 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| | mean, points a year | | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| H -> V5 | combined, at the defaults: vol | +0.000014 | +0.3% | -0.8% | -0.1% | -0.2% | -5.9% |
| | mean, points a year | | +0.01 | +0.01 | -0.00 | +0.00 | -0.00 |
| V4 -> V4s21 | RSTR skip 21 (the brief's reading): vol | -0.000575 | -0.6% | -0.8% | +1.3% | +0.9% | +0.4% |
| | mean, points a year | | -0.13 | +0.79 | -0.29 | +0.07 | +0.09 |
| V4b0 -> V4b21 | RSTR 21 back (reading b, matched): vol | -0.000596 | -0.5% | -0.2% | +1.2% | +1.1% | -0.2% |
| | mean, points a year | | -0.17 | -0.09 | -0.18 | +0.27 | +0.10 |
| V5 -> V5w | 252-day window, as run: vol | +0.004741 | +6.7% | -1.7% | -6.1% | -4.0% | -1.9% |
| | mean, points a year | | +0.42 | +0.71 | -0.25 | -1.11 | +0.62 |
| V5w -> V5u | + half-life 63, as run: vol | +0.000454 | +5.9% | -6.6% | -2.9% | -0.3% | +1.8% |
| | mean, points a year | | -0.22 | -0.41 | +0.23 | -0.03 | +0.07 |
| V5 -> V5u | USE4's market model, as run: vol | +0.005195 | +13.0% | -8.3% | -8.8% | -4.3% | -0.1% |
| | mean, points a year | | +0.20 | +0.30 | -0.02 | -1.14 | +0.69 |
| V5 -> V5w938 | 252-day window, matched: vol | +0.004551 | +6.7% | -1.7% | -6.1% | -4.0% | -1.9% |
| | mean, points a year | | -0.80 | +1.68 | -0.67 | -0.60 | +0.21 |
| V5w938 -> V5u938 | + half-life 63, matched: vol | +0.000667 | +5.9% | -6.6% | -2.9% | -0.3% | +1.8% |
| | mean, points a year | | -0.25 | -0.48 | +0.44 | -0.20 | -0.00 |
| V5 -> V5u938 | USE4's market model, matched: vol | +0.005218 | +13.0% | -8.3% | -8.8% | -4.3% | -0.1% |
| | mean, points a year | | -1.05 | +1.20 | -0.23 | -0.80 | +0.21 |

### Exactness

| | | factor returns | style exposures | forecast exposures | specific risks | summary fields differing | report lines only in the first / second |
|---|---|---|---|---|---|---|---|
| V1_0 vs H | change 1's code at 0 is the committed code | True | True | True | True | none | 0 / 0 |
| V1 vs H50 | change 1's default is the committed code run at 0.5 | True | True | True | True | none | 1 / 4 |
| V3 vs V2 | change 3 leaves the run alone | True | True | True | True | none | 0 / 0 |
| V4 vs V3 | change 4's default is today's RSTR | True | True | True | True | none | 0 / 0 |
| V5 vs V4 | change 5's default is today's market model | True | True | True | True | none | 1 / 0 |

V2 against V1 is the one step that is not bit for bit at the defaults, and by rounding alone in the
estimation: its factor returns are at most 2.5e-16 apart (697 of 22,512 cells identical).
Its forecast exposures move for real -- see change 2.

The report lines that differ, V1 against H50:

```
+ window descriptors need at least 50% of the window valid; from a window with none, over 3,581,487 values: none; from one under 50% valid: none
- window descriptors need at least 50% of the window valid; from a window with none, over 3,581,487 values: none
- from under half their window, which a 50% rule would remove (all/on members):
- DASTD 0/0  RSTR 0/0  CMRA 0/0  BTSG 0/0  HALPHA 0/0
- SERDP 0/0  STOM 0/0  STOQ 0/0  STOA 0/0  VLVR 0/0
```

The report lines that differ, V5 against V4:

```
+ the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 828 trading days (36 months), every day weighted equally
```

## Change 1: `min_coverage` defaults to 0.5

`compute_descriptors(min_coverage=0.5)` is now the default, and so is the run's
`--min-coverage 0.5`; 0 stays available. The run's window guard now checks two things against
each descriptor's own inputs, counted independently: no value from an empty window, and none
from under the rule's fraction of one. The "what a 50% rule would remove" table prints only when
the rule is below 0.5. At the default, on the real panel:

    window descriptors need at least 50% of the window valid; from a window with none, over
    3,581,487 values: none; from one under 50% valid: none

That is 29,506 fewer values than at 0 (3,610,993), exactly the count midoffice02's report said
a 50% rule would remove (7,086 of them on members). The model is the committed code run at
`--min-coverage 0.5`, bit for bit (H50, which reproduces midoffice02's variant E). At 0 the new
code is the committed code bit for bit, report included. Against HEAD: mean R-squared
0.294197 -> 0.294211, Trading Activity's volatility 5.30% -> 4.99% (-5.9%), Momentum's 7.22%
-> 7.16%, the others within 0.3%. ECHO and VMRK lose their only complete row, so there are 468
forecasts, not 470, and 8 fallbacks, not 10. The brief calls them "two phantom identities ...
that the rebuild will fix".

## Change 2: SERDP in no blend; the guard; the blend composition

**What.** The run script's `UNBLENDED` names SERDP with its reason, and the blends are built from
the standardised descriptors without it (`without`). SERDP is still computed, screened (its row
still reads "too few usable periods") and reported. Table 8-1's membership in `descriptors.py`
still lists it, because that is the published table. SERDP has a value on the last date alone,
so leaving it out of every blend is leaving it out of the forecast blend. Every forecast exposure
is then built from the descriptors the last estimated period's were:

```
each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386)
    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
```

**The guard.** For each index, the run asks which descriptors contributed to the forecast
exposures: those with a value for a security that has a forecast. It also asks which contributed
to each estimated period's: those with a value for a security in that period's regression, by
the estimator's own `regression_design(...).in_regression`. A descriptor in the forecast and in
no estimated period stops the run. Any other difference between the forecast's set and the last
estimated period's is flagged with "!". Put SERDP back (`UNBLENDED` emptied) and the run stops:

    the run stops: a descriptor contributes to a forecast exposure and to no estimated period's,
    so its factor would be read on a blend it was never estimated on -- Volatility's SERDP, in
    466 securities' forecast exposures and none of the 938 estimated periods'

The full stopped report is in the appendix (`final_guard.txt`, from the finished tree). The
composition comes from a new library helper, `descriptors.blend_composition(panel, index, rows,
counted)`, which is tested.

**What it moved.** The estimation only by rounding. Without SERDP, `blend_to_index`'s equal
weights are 1/5 where they were 1/6, so every Volatility exposure differs in its last bits (at
most 2.2e-15 before the last date). The factor returns differ by at most 2.5e-16, against a
typical size of 4.8e-3; mean R-squared, the residual spread and every volatility agree to every
printed digit. The forecast moved for real:

```
factor returns: (938, 24); identical cells 697 of 22512; largest |difference| 2.498e-16, against a typical |factor return| 4.809e-03; relative to each factor's sd at most 5.944e-14
style exposures before the last date: same missing pattern True; largest |difference| per style: Volatility 2.2e-15, Momentum 0.0e+00, Size 0.0e+00, Size Nonlinearity 0.0e+00, Trading Activity 0.0e+00
  Volatility         vol 7.6312022004% -> 7.6312022004%  mean +4.5208441450% -> +4.5208441450%
  Momentum           vol 7.1647237439% -> 7.1647237439%  mean +2.4444100057% -> +2.4444100057%
  Size               vol 7.5658677619% -> 7.5658677619%  mean +4.9168933660% -> +4.9168933660%
  Size Nonlinearity  vol 6.6705083256% -> 6.6705083256%  mean -1.4378072908% -> -1.4378072908%
  Trading Activity   vol 4.9891698587% -> 4.9891698587%  mean +3.9049349848% -> +3.9049349848%
R2 0.29421084071845693 -> 0.29421084071845693; spread 0.3681986132134846 -> 0.3681986132134846
forecast Volatility exposure, 468 securities with a forecast in both: correlation 0.9655, rms change 0.276, mean |change| 0.213, largest 0.987
  largest moves: GL -1.80 -> -0.81, MPWR +1.85 -> +2.74, GEN +1.30 -> +0.42, INFO -3.19 -> -2.33, SBNY +1.34 -> +2.18, CMI -0.72 -> +0.12
every other forecast exposure: largest |change| 0.0e+00
specific risk forecasts: rms change 0.371 points, largest 5.566 points (INFO); median 23.5194% -> 23.5268%
gamma Volatility 0.18773549745640492 -> 0.18773549745640464
floored [] -> []; fallback 8 -> 8; forecasts 468 -> 468
```

## Change 3: `fit_index_weights` pairs each descriptor with the next period's return

`fit_index_weights(..., lag=1)` pairs the descriptor on t with the return on t+1, and reads
membership on the descriptor's date, as the regressions do. `lag=0` is the old same-day pairing,
kept for a synthetic world whose descriptors are known before their returns, and validated like
`regression_design`'s lag. It is still one pooled, whole-sample regression. The docstring now
says so: "It is fitted on the whole sample, and that is look-ahead too ... Out of sample would
mean refitting on each date from the returns before it alone. Nothing here does that." The third
test in `test_lookahead.py` still passes unchanged and is its record; only its docstring changed,
to say the fit is lagged and still whole-sample. The run never calls it: V3 is V2 bit for bit,
report included.

What the lag does to a fit on the real panel (members, the blendable descriptors, whole-sample),
not used by anything:

```
fit_index_weights on the real panel: whole-sample, over index members, the blendable descriptors (SERDP out), at lag 0 (same day) and lag 1 (the new default)
Volatility        all descriptors          lag 0: BTSG -0.958, DASTD +6.098, HILO -6.450, LPRI -1.337, CMRA +3.647 (n 166,740)  lag 1: BTSG -0.204, DASTD +1.000, HILO +1.050, LPRI -1.258, CMRA +0.412 (n 166,308)
Volatility        present from the start   lag 0: DASTD -4.260, HILO +7.193, LPRI -0.542, CMRA -1.391 (n 420,521)  lag 1: DASTD +1.904, HILO +1.257, LPRI -1.977, CMRA -0.183 (n 420,081)
Momentum          all descriptors          lag 0: RSTR +2.411, HALPHA -1.411 (n 167,378)  lag 1: RSTR -0.628, HALPHA +1.628 (n 166,946)
Trading Activity  all descriptors          lag 0: STOM -1.985, STOQ +3.159, STOA +1.024, VLVR -1.197 (n 167,345)  lag 1: STOM -0.328, STOQ +1.029, STOA +1.218, VLVR -0.919 (n 166,912)
Trading Activity  present from the start   lag 0: STOM -1.978, STOQ +2.074, STOA +0.904 (n 422,559)  lag 1: STOM -0.270, STOQ +0.546, STOA +0.724 (n 422,115)
```

Same-day, Momentum's fit hands RSTR, which holds the day's return, a weight of +2.41 and HALPHA
-1.41. Lagged, the signs reverse. The normalised weights are unstable in both, because
coefficients of mixed sign are divided by their sum. That is one more reason the blends stay
equal-weighted.

## Change 4: RSTR's skip

`compute_descriptors(rstr_skip=k)` (`--rstr-skip`) excludes the last k trading days from RSTR's
twelve-month window, as the brief words it. The window keeps its length and its start, so RSTR
has a value from the same first date; the sum covers its first 276 - k periods, 255 at k = 21.
The coverage rule and the run's window guard count the periods summed. 0, the default, is
today's RSTR bit for bit (V4 is V3, report included). At 21 the guard still finds no value from
an empty or thin window.

| | RSTR | periods | Momentum vol | mean (t) | noise share | mean R-squared | screen: RSTR's R-squared, vol, t |
|---|---|---|---|---|---|---|---|
| V4 | skip 0 (the default) | 938 | 7.16% | +2.44% (+0.81) | 4.0% | 0.294211 | 5.31%, 7.12%, +1.11 |
| V4s21 | skip 21, the brief's definition | 938 | 7.11% | +3.23% (+1.09) | 4.0% | 0.293636 | 5.21%, 7.03%, +1.33 |
| V4b0 | skip 0, matched to V4b21 | 917 | 7.17% | +1.85% (+0.61) | 4.0% | 0.294984 | 5.32%, 7.14%, +0.87 |
| V4b21 | twelve months ending 21 back | 917 | 7.15% | +1.75% (+0.58) | 4.0% | 0.294388 | 5.33%, 7.17%, +0.67 |

At 21 the RSTR values themselves change little. Their correlation with skip 0's is
0.956 pooled; across securities, day by day, median 0.967 and lowest 0.657.
Skip 21 has a value on 256 cells that skip 0 lacks, and lacks 99 that skip 0 has, because the
coverage is counted over the shorter span. Mean R-squared falls 0.000575; Size's mean
moves from +4.92% to +4.63%.

**The other reading.** USE4 states its RSTR as "trailing T = 504 trading days with a lag of L =
21": a full window ending 21 days back, not a window with its end cut off. The brief defines the
parameter as days "excluded from the end of its window", and that is what was built. The other
reading was measured in scratch only: RSTR at skip 0, read 21 days later. It has no value for its
first 21 days, which would drop 21 periods, so it is compared on matched cells: both RSTRs
blanked wherever either lacks a value, 917 periods each. The two readings disagree in direction:
the brief's raises Momentum's mean by 0.79 points a year, the other lowers it by 0.09.

## Change 5: the market model's window and half-life

**Today's.** The run's market model, behind BTSG, HALPHA, SERDP and VLVR's residual sd, is OLS
over `--beta-window 36` months = 828 trading days, every day weighted equally, so no half-life.
`compute_descriptors`' own default, 60 months, is 1,380 days, more than the panel's 1,214: it
would fit once, on the last date. Every run now prints the model's window and half-life:

    the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 828 trading days (36 months), every day weighted equally

**What was added.** `compute_descriptors(beta_half_life=h)`, in the panel's periods, as
`ExponentialCovariance`'s half-life is: weighted least squares, each day's weight halving every
h days back from the window's end. The residual sd is then the weighted one, with its degrees of
freedom on Kish's effective count, which is n - 2 when the weights are equal. USE4 says only
"the volatility of residual returns". At 252/63 the correction is a factor of about 1.006, and
standardisation removes most of it. The run takes `--beta-half-life` and `--beta-days`, the
window in trading days in place of `--beta-window`'s months, because 252 days is no round
number of USE3 months. The default (none) is today's market model bit for bit (V5 is V4; its
report gains only the standing line).

**When each of the three first appears:**

```
default: 36 months, equal weights: window 828 trading days, effective observations 828.0
  BTSG   first value 2025-04-22 (row 827), on 30 members 2025-04-22 (row 827); the first period that can read it 2025-04-23
  HALPHA first value 2025-04-22 (row 827), on 30 members 2025-04-22 (row 827); the first period that can read it 2025-04-23
  VLVR   first value 2025-04-22 (row 827), on 30 members 2025-04-22 (row 827); the first period that can read it 2025-04-23
USE4: 252 days, half-life 63: window 252 trading days, effective observations 160.4; the oldest day's weight 0.0632, the window's last quarter 53.3% of the weight
  BTSG   first value 2023-01-03 (row 251), on 30 members 2023-01-03 (row 251); the first period that can read it 2023-01-04
  HALPHA first value 2023-01-03 (row 251), on 30 members 2023-01-03 (row 251); the first period that can read it 2023-01-04
  VLVR   first value 2023-02-07 (row 275), on 30 members 2023-02-07 (row 275); the first period that can read it 2023-02-08
compute_descriptors' own default, 60 months: 1380 trading days, capped at the panel's 1214, so one fit, on the last date (window 1214)
```

So at 252/63 all three are in every one of the default's 938 periods, and BTSG and HALPHA in
the 24 earlier ones the run then also estimates:

As run (V5u):

```
each risk index's blend over the 962 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI from 2023-01-04 (962); CMRA from 2023-02-08 (938)
    Momentum             HALPHA from 2023-01-04 (962); RSTR from 2023-02-08 (938)
    Size                 LNCAP from 2023-01-04 (962)
    Size Nonlinearity    LCAPCB from 2023-01-04 (962)
    Trading Activity     STOM, STOQ from 2023-01-04 (962); STOA, VLVR from 2023-02-08 (938)
  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
```

On the default's periods (V5u938):

```
each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2023-02-08 (938)
    Momentum             RSTR, HALPHA from 2023-02-08 (938)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA, VLVR from 2023-02-08 (938)
  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
```

**The 24 extra periods.** As run, the 252-day window also starts the estimation 24 trading days
earlier, on 2023-01-04 rather than 2023-02-08. HALPHA alone gives Momentum a value before RSTR's
twelve-month window fills, and a complete row is all a period needs. For those 24 periods
Momentum is HALPHA alone, Volatility lacks CMRA, and Trading Activity lacks STOA and VLVR; the
new blend line shows it (above). Both 252-day variants were therefore also run held to the
default's 938 periods (every descriptor blanked before 2023-02-07), so the market model's own
effect is measured on the same sample.

**R-squared and the style volatilities:**

| | 828 days, equal (default) | 252 days, equal | 252 days, half-life 63 | 252, equal, matched | 252 / 63, matched |
|---|---|---|---|---|---|
| mean R-squared | 0.294211 | 0.298952 | 0.299406 | 0.298762 | 0.299429 |
| periods estimated | 938 | 962 | 962 | 938 | 938 |
| first estimated period | 2023-02-08 | 2023-01-04 | 2023-01-04 | 2023-02-08 | 2023-02-08 |
| **annual vol** | | | | | |
| Volatility | 7.63% | 8.14% | 8.63% | 8.14% | 8.63% |
| Momentum | 7.16% | 7.04% | 6.57% | 7.04% | 6.57% |
| Size | 7.57% | 7.10% | 6.90% | 7.10% | 6.90% |
| Size Nonlinearity | 6.67% | 6.40% | 6.38% | 6.40% | 6.38% |
| Trading Activity | 4.99% | 4.90% | 4.98% | 4.90% | 4.98% |
| **annual mean return (t-stat)** | | | | | |
| Volatility | +4.52% (+1.37) | +4.94% (+1.37) | +4.72% (+1.30) | +3.72% (+1.02) | +3.47% (+0.94) |
| Momentum | +2.44% (+0.81) | +3.16% (+1.07) | +2.74% (+0.94) | +4.12% (+1.38) | +3.64% (+1.22) |
| Size | +4.92% (+1.54) | +4.66% (+1.56) | +4.90% (+1.65) | +4.25% (+1.40) | +4.69% (+1.56) |
| Size Nonlinearity | -1.44% (-0.54) | -2.55% (-0.99) | -2.57% (-1.01) | -2.03% (-0.78) | -2.24% (-0.86) |
| Trading Activity | +3.90% (+2.14) | +4.52% (+2.82) | +4.59% (+2.86) | +4.11% (+2.54) | +4.11% (+2.53) |
| **noise share** | | | | | |
| Volatility | 5.3% | 4.5% | 3.9% | 4.5% | 3.9% |
| Momentum | 4.0% | 3.8% | 4.4% | 3.9% | 4.4% |
| Size | 9.9% | 11.0% | 11.6% | 11.0% | 11.6% |
| Size Nonlinearity | 17.1% | 18.5% | 18.6% | 18.5% | 18.6% |
| Trading Activity | 11.5% | 10.6% | 10.0% | 10.6% | 10.0% |
| residual spread (relative residual sd) | 0.3682 | 0.2821 | 0.2793 | 0.3660 | 0.3632 |
| its largest shares | FRC 71.3%, SIVB 4.0%, LUMN 2.9% | FRC 57.7%, VNO 4.8%, LUMN 3.4% | FRC 57.9%, VNO 3.7%, LUMN 3.2% | FRC 71.9%, SIVB 3.8%, LUMN 2.9% | FRC 72.0%, SIVB 3.7%, LUMN 2.4% |
| specific risk p1 / p10 / p50 | 9.50% / 15.06% / 23.53% | 9.56% / 15.22% / 23.40% | 9.55% / 15.18% / 23.43% | 9.54% / 15.12% / 23.36% | 9.53% / 15.10% / 23.46% |
| specific risk p90 / p99 | 36.35% / 54.52% | 36.11% / 53.84% | 36.06% / 53.82% | 36.45% / 54.28% | 36.35% / 54.24% |
| specific risk min / max | 9.02% / 106.68% | 8.99% / 106.39% | 8.97% / 106.42% | 8.99% / 107.73% | 8.98% / 107.77% |
| forecasts | 468 | 468 | 468 | 468 | 468 |
| floor list: characteristic forecast below the floor | none | none | none | none | none |
| floor list: delivered the floor | none | none | none | none | none |
| regression universe | 482 | 483 | 483 | 482 | 482 |
| fallback (in no regression) | 8: INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO | 7: INFO, IPGP, MRSH, PENN, PVH, UA, UAA | 7: INFO, IPGP, MRSH, PENN, PVH, UA, UAA | 8: INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO | 8: INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO |

The 252-day window does most of it: mean R-squared +0.0046 on the matched sample, Volatility's
volatility +6.7%. The half-life adds +0.0007 and +5.9%. Momentum's and Size's volatilities fall
by 8.3% and 8.8%; why was not decomposed further. Because the covariance's half-life is 90
days, the extra 24 periods barely touch the volatilities, but they do move the means.

**The screen**, ranked by mean R-squared (the market model's rows):

| | V5 | V5w | V5u | V5w938 | V5u938 |
|---|---|---|---|---|---|
| BTSG: periods, R-squared, vol, t | 386, 8.43%, 9.77%, +1.87 | 962, 8.03%, 9.09%, +1.40 | 962, 8.36%, 9.32%, +1.34 | 938, 7.97%, 9.10%, +1.02 | 938, 8.30%, 9.32%, +0.96 |
| BTSG's rank | 1 | 1 | 1 | 1 | 1 |
| HALPHA: periods, R-squared, vol, t | 386, 3.38%, 6.16%, -0.29 | 962, 4.75%, 6.57%, +1.02 | 962, 4.43%, 6.30%, +0.56 | 938, 4.75%, 6.56%, +1.32 | 938, 4.48%, 6.35%, +0.78 |
| HALPHA's rank | 6 | 4 | 4 | 4 | 4 |
| VLVR: periods, R-squared, vol, t | 386, 1.52%, 3.94%, +0.58 | 938, 1.74%, 3.73%, +1.95 | 938, 1.78%, 3.76%, +1.95 | 938, 1.74%, 3.73%, +1.95 | 938, 1.78%, 3.76%, +1.95 |
| VLVR's rank | 13 | 12 | 11 | 12 | 11 |
| SERDP | 0 periods | 0 periods | 0 periods | 0 periods | 0 periods |
| the ranking | BTSG, DASTD, RSTR, HILO, CMRA, HALPHA, STOQ, STOM, STOA, LNCAP, LPRI, LCAPCB, VLVR | BTSG, DASTD, RSTR, HALPHA, HILO, CMRA, STOQ, STOM, STOA, LNCAP, LPRI, VLVR, LCAPCB | BTSG, DASTD, RSTR, HALPHA, HILO, CMRA, STOQ, STOM, STOA, LNCAP, VLVR, LPRI, LCAPCB | BTSG, RSTR, DASTD, HALPHA, CMRA, HILO, STOA, STOQ, STOM, LNCAP, LCAPCB, VLVR, LPRI | BTSG, RSTR, DASTD, HALPHA, CMRA, HILO, STOA, STOQ, STOM, LNCAP, VLVR, LCAPCB, LPRI |

BTSG leads in every variant. HALPHA rises from sixth to fourth.

**Specific risk.** As run, the residual spread falls from 0.368 to 0.279. That is the sample
start, not the market model: FRC sets the spread, and its share of the square falls from 71% to
58% when the estimation starts 24 days earlier, while VNO, a member in those days, joins the
universe. On the matched sample the spread is 0.363 (FRC 72%).

| | HEAD | the defaults | 252 / 63, as run | 252 / 63, matched |
|---|---|---|---|---|
| periods / first | 938 / 2023-02-08 | 938 / 2023-02-08 | 962 / 2023-01-04 | 938 / 2023-02-08 |
| residual spread | 0.3679 | 0.3682 | 0.2793 | 0.3632 |
| its largest shares | FRC 71.3%, SIVB 4.0%, LUMN 2.9% | FRC 71.3%, SIVB 4.0%, LUMN 2.9% | FRC 57.9%, VNO 3.7%, LUMN 3.2% | FRC 72.0%, SIVB 3.7%, LUMN 2.4% |
| specific p1 / p10 / p50 / p90 / p99 | 9.51 / 15.11 / 23.52 / 36.36 / 53.35 | 9.50 / 15.06 / 23.53 / 36.35 / 54.52 | 9.55 / 15.18 / 23.43 / 36.06 / 53.82 | 9.53 / 15.10 / 23.46 / 36.35 / 54.24 |
| universe / forecasts | 482 / 470 | 482 / 468 | 483 / 468 | 482 / 468 |
| fallback | ECHO, INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VMRK, VNO | INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO | INFO, IPGP, MRSH, PENN, PVH, UA, UAA | INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO |
| floor list | none | none | none | none |

## Change 6 (separate): midoffice01's Source lines

`BRIEF_midoffice01_standardise_over_members.md`'s Source line ("Jeff's request in the Claude Code
session ... No Chat-written brief preceded it ...") and the retro's Brief line ("... it was given
in the Code session, not as a Chat-written brief") now say written in Chat and pasted into the
Code session, in midoffice02's words. Nothing else in either file changed. The brief's heading
"The request, verbatim" was left as it is, because it makes no claim about where the brief was
written. Its own `git add` is below, before this change's.

## Recorded for the rebuild, not done

From the brief: share counts must change on a split's execution date, using the split events
rather than the price adjustment factor -- DD's 2.39 and 0.33 suggest that factor also carries
actions that are not splits. Check each applied split against the next share sample, flag any
that disagree, and guard on zero member-days pairing a post-split price with a pre-split count.
This is now in `docs/FACTOR_MODEL.md` ("Two price bases"), with midoffice02's count: 896
security-days on the S&P 500 panel, 621 of them member-days an estimated period reads, across 26
splits. It is also in memory. Nothing in the code changed for it.

## Key decisions

1. **The window guard became exact under the rule.** At 0.5 it checks that no value comes from
   under half its window (none) as well as from an empty one (none). The 50% table it used to
   print is redundant at the default, so it prints only below 0.5. At 0 the report is the
   committed one, line for line.
2. **SERDP out of every blend, by the run script, not by editing Table 8-1.** `_RISK_INDEX`
   is the published membership and still lists SERDP; the run's `UNBLENDED` says why it is left
   out. Since SERDP exists on the last date alone, "out of every blend" and "out of the forecast
   blend" are the same thing.
3. **The guard counts the regression, not the panel.** A descriptor contributes to an estimated
   period only through a security in that period's regression (`in_regression`), and to the
   forecast only through a security with a forecast. The old flag counted any security.
4. **Only the brief's direction stops the run.** A descriptor in an estimated period and not in
   the forecast, or any other difference from the last estimated period's set, prints "!". The
   brief asked for a stop only on the first.
5. **The stop comes after the estimation.** The estimated periods are the estimator's own
   (`notes.estimated_dates`), so the guard cannot disagree with them. The cost is one
   estimation before it fires.
6. **Change 2's rounding was reported, not engineered away.** Making the blend's arithmetic
   independent of the descriptor count would have moved HEAD's bits everywhere.
7. **`fit_index_weights` got a `lag` parameter, not a hard-wired shift.** Default 1, 0 for a
   synthetic same-day world, validated like `regression_design`'s. Two existing tests were
   rebuilt for the lag: the known-blend test now drives returns from the previous period's
   descriptors, and the members test blanks by the previous period's membership. Each asserts
   what it did before.
8. **RSTR's skip as the brief worded it.** The window keeps its start, the last k days are
   excluded, and coverage is counted over the days summed. USE4's "trailing T with a lag of L"
   reading was measured, not built.
9. **The half-life is in the panel's periods**, like `ExponentialCovariance`'s. **The weighted
   residual sd uses Kish's effective count for its degrees of freedom**, so the equal-weight
   case is the old n - 2.
10. **`--beta-days` was added** because 252 trading days is 10.96 USE3 months. **Every run now
    prints the market model's window and half-life.** The brief asked for them to be reported,
    and the 828-day window is what kept beta out of 552 periods.
11. **Change 5 was also measured on the default's 938 periods.** Otherwise the 24 extra periods
    the shorter window allows would have been counted as the market model's effect.
12. **No default changed except change 1's.** `rstr_skip` 0, `beta_half_life` none, and the
    run's 36-month window are as they were. The size pair was not touched.

## Decisions left for Jeff

1. **RSTR's skip:** 0 or 21. If 21, which reading: the window with its last 21 days cut off
   (built), or a full twelve months ending 21 days back (measured). They move Momentum's mean in
   opposite directions here (+0.79 and -0.09 points a year).
2. **The market model:** 828 equal-weighted days (today) or 252 days with half-life 63 (USE4).
   If 252, what the estimation should do with the 24 earlier periods in which Momentum is HALPHA
   alone: start where every index has its full blend, require RSTR, or accept.
3. **`compute_descriptors`' default `beta_window_months=60`** is 1,380 days, longer than a
   five-year daily panel, so it fits once. The run overrides it with 36. Align the library
   default, or leave it.
4. **The size pair**, as input: Size's mean is +4.92% at the defaults, +4.63% at skip 21, and
   +4.69% under USE4's market model on the matched sample.
5. **SERDP historically** still waits for a reason, as the brief says.

## Tests

- **Full suite**, with `--ignore=tests/test_market_data.py` (its `mctheory` import error
  predates this, as in midoffice02); 2,221 before:
  2,225 passed, 1 skipped, 19 warnings in 64.57s (0:01:04).
- **midoffice:** 774 passed, 1 skipped in 21.94s; 770 before.
- **ruff** on the four changed Python files: one finding, F401 for the unused
  `HorizonCovariance` import in `run_sp500_model.py`, which HEAD has too. The I001 that inserting
  `UNBLENDED` caused was fixed.

New tests (`tests/midoffice/test_descriptors.py`):

| test | what it guards |
|---|---|
| `test_a_blends_composition_is_read_row_by_row_over_the_securities_counted` | `blend_composition`, exactly: Momentum is RSTR from period 11 and HALPHA from 59; Volatility carries SERDP on the last date only; counting only a security without HALPHA sees no HALPHA; a `counted` of the wrong shape or dtype is refused |
| `test_fitted_weights_pair_each_descriptor_with_the_next_periods_return` | changing the first period's returns or the last period's descriptors leaves the weights bit for bit; at lags 0 and 1 the fit equals the pooled regression computed independently; lag 0 differs from 1; bad lags are refused |
| `test_rstr_can_skip_the_end_of_its_window_and_by_default_skips_nothing` | skip 0 is the old RSTR byte for byte; skip 3 is the window's sum less its last three, exactly; the first value falls on the same date; coverage is counted over the periods summed (seven of twelve kept, four of nine refused); bad skips are refused |
| `test_the_market_model_can_be_weighted_by_a_half_life_and_by_default_is_not` | the default equals an explicit None byte for byte; with a half-life, HALPHA and BTSG equal a weighted fit with Kish's dof computed independently, on a window with gaps, for every security; bad half-lives are refused |

Changed tests:

| test | change |
|---|---|
| `test_a_security_whose_prices_start_after_the_window_gets_missing_for_every_one` | the empty-window checks run at 0 and at the default; the single-observation value is checked at 0, and its absence at the default |
| `test_min_coverage_removes_thinly_covered_windows_and_alters_no_value` | "keeps them all" is now at an explicit 0; adds that the default is the 0.5 rule byte for byte |
| `test_fitted_weights_recover_a_known_blend` | returns driven by the previous period's descriptors, as the lagged fit reads them; same tolerances |
| `test_fitted_weights_learn_from_the_members_alone` | membership read on the descriptor's date (the previous period's blanking); adds that same-period blanking differs |
| `test_lookahead.py::test_refitting_the_blend_after_repricing_moves_every_period_the_earliest_included` | docstring only: lagged, still whole-sample, the record the brief asked to keep |

## Failures and debugging trail

1. **The first baseline run stopped at once**: "--membership needs a snapshot: no snapshots in
   data/external/sp500". The run script's paths are relative and it ran from the scratch
   directory. Rerun from the repo, as midoffice02's were.
2. **The report diff paired lines by position** and misaligned after one removed line. It now
   uses difflib. A heredoc patch to it did not apply (escaping), so the file was rewritten.
3. **The new default failed one test**, as expected: the single-observation RSTR at period 101.
   The test now checks both rules.
4. **Change 2 was not bit-identical**, though I had expected it to be. Diagnosed as the 1/5
   against 1/6 equal weights, and measured (above).
5. **The composition test expected no CMRA** without a risk-free rate; CMRA computes on raw
   returns. The expected list was fixed.
6. **`fit_real.py` asserted descriptor order** where the panel's order is not Table 8-1's; it now
   compares names as sets.
7. **ruff I001** after `UNBLENDED`: one blank line after the imports, not two.
8. **The full suite stopped on collection** at `tests/test_market_data.py` (`mctheory`), the
   same pre-existing error as in midoffice02; rerun with it ignored.
9. **A reused name.** The run's new mismatch list was called `differs`, which the industry check
   above already uses for a set, and its loop variable was `line`. They were renamed after the
   change 5 snapshot. The finished tree was then run again: its default report and model are
   change 5's, bit for bit, and the stop still fires (`final_check.py`).

## Problems found, not fixed

1. **The 24 extra periods under a 252-day window** (above): Momentum is HALPHA alone in them.
   Nothing in the estimator requires a full blend before a period is estimated.
2. **The residual spread is FRC's.** Moving the start of the sample by 24 days moved it from
   0.368 to 0.279.
3. **`compute_descriptors`' default beta window** exceeds a five-year daily panel (above).
4. **VLVR runs the market model a second time** for its residual sd. That doubles the cost of
   the slowest descriptor; it is not wrong.
5. **Split-stale share counts**: recorded for the rebuild (above).
6. **The pre-existing F401** in the run script, and `tests/test_market_data.py`'s import error.

## Scratch

Everything is under the session scratchpad's `m03/`, copied without the pickles to
`claude/scratch/midoffice03/` (gitignored). Its outputs are in the appendix.

| file | what |
|---|---|
| `common.py`, `runv.py` | load a tree's run script, run `main()` per variant, save report/summary/arrays |
| `extract_brief.py`, `write_brief.py` | the brief from the transcript's original message, and the `BRIEF_` file from it |
| `exact.py` | bit-for-bit comparison of two variants, report diffed |
| `guard_check.py`, `final_check.py` | change 2's guard both ways; the finished tree's rerun |
| `step2.py` | what change 2 moved |
| `fit_real.py` | change 3's fits on the real panel |
| `rstr_variants.py` | change 4's variants, both readings |
| `beta_matched.py`, `first_values.py` | change 5 on the default's periods; when the three first appear |
| `analyse.py`, `build_retro.py` | the tables, and this retro |

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. Both `git add` lists below were checked
with `git add --dry-run`, and every changed path is in exactly one of them.

## Open items for Chat

- The two definitions (Decisions 1 and 2). They interact: under USE4's market model, BTSG and
  HALPHA are in every period, and RSTR's skip was measured only under today's.
- Whether a period should need every index's full blend, not just a complete row (Problem 1).
- The split rule above, for the rebuild's brief.

## Appendix: scratch outputs, verbatim

### runs_tree0.out -- the baseline: HEAD as committed, and at 0.5

```
H      lag 1  R2 0.294197  periods 938  forecasts 470  Volati 7.6113%/+4.51%  Moment 7.2204%/+2.44%  Size 7.5764%/+4.92%  Size N 6.6872%/-1.44%  Tradin 5.3005%/+3.91%  [55s; --industries --membership]
H50    lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [37s; --industries --membership --min-coverage 0.5]
```

### runs_tree1.out -- change 1's tree at its default and at 0

```
V1     lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [48s; --industries --membership]
V1_0   lag 1  R2 0.294197  periods 938  forecasts 470  Volati 7.6113%/+4.51%  Moment 7.2204%/+2.44%  Size 7.5764%/+4.92%  Size N 6.6872%/-1.44%  Tradin 5.3005%/+3.91%  [34s; --industries --membership --min-coverage 0]
```

### exact_step1.txt -- change 1 against the baseline

```
V1_0 vs H: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 0
V1 vs H50: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 5
   -  window descriptors need at least 50% of the window valid; from a window with none, over 3,581,487 values: none; from one under 50% valid: none
   +  window descriptors need at least 50% of the window valid; from a window with none, over 3,581,487 values: none
   +  from under half their window, which a 50% rule would remove (all/on members):
   +    DASTD 0/0  RSTR 0/0  CMRA 0/0  BTSG 0/0  HALPHA 0/0
   +    SERDP 0/0  STOM 0/0  STOQ 0/0  STOA 0/0  VLVR 0/0
H vs V1: factor returns False, style exposures False, forecast exposures False, specific risks False; summary fields differing: ['r2', 'n_estimated', 'vol', 'mean', 'tstat', 'noise', 'spread', 'spread_shares', 'v2_shares', 'gamma', 'specific', 'n_specific', 'fallback', 'no_exposures', 'screen', 'compute_kw']; report lines differing: 93
   -  window descriptors need any valid observation; from a window with none, over 3,610,993 values: none
   -  from under half their window, which a 50% rule would remove (all/on members):
   -    DASTD 1,612/533  RSTR 4,847/1,339  CMRA 4,648/1,339  BTSG 5,009/1,111  HALPHA 5,009/1,111
   -    SERDP 12/5  STOM 467/66  STOQ 1,363/189  STOA 3,931/423  VLVR 2,608/970
   -  347,903 raw values winsorised before standardising
   -  100,011 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
   -    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/47  HILO 8/45  CMRA 5/51  BTSG 8/66
   -    HALPHA 7/66  SERDP 1/66  RSTR 3/54  STOM 13/46  STOQ 14/48  STOA 16/54  VLVR 18/54
   +  window descriptors need at least 50% of the window valid; from a window with none, over 3,581,487 values: none; from one under 50% valid: none
   +  344,062 raw values winsorised before standardising
   +  95,646 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
   +    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/46  HILO 8/45  CMRA 5/46  BTSG 6/59
   +    HALPHA 6/59  SERDP 1/59  RSTR 3/49  STOM 13/45  STOQ 14/46  STOA 16/49  VLVR 17/49
   -  BTSG         97%      386    8.39%      9.76%     1.88
   -  DASTD        93%    1,149    6.13%      7.78%     1.35
   -  RSTR         95%      938    5.30%      7.11%     1.11
   +  BTSG         95%      386    8.43%      9.77%     1.87
   +  DASTD        93%    1,149    6.13%      7.77%     1.36
   +  RSTR         93%      938    5.31%      7.12%     1.11
   -  CMRA         94%      938    3.38%      5.58%     1.39
   -  HALPHA       97%      386    3.32%      6.10%    -0.29
   +  CMRA         92%      938    3.39%      5.59%     1.37
   +  HALPHA       95%      386    3.38%      6.16%    -0.29
   -  STOA         95%      938    2.56%      4.85%     1.51
   +  STOA         93%      938    2.57%      4.85%     1.50
   -  VLVR         95%      386    1.50%      3.92%     0.64
   +  VLVR         93%      386    1.52%      3.94%     0.58
   -  24-factor model, annualised, estimated on 938 daily period(s) for 470 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
   +  24-factor model, annualised, estimated on 938 daily period(s) for 468 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
   -  ! Volatility: SERDP is in 492 securities' forecast exposures and in none of the 938 estimated periods', so the factor was estimated on a blend without it
   +  ! Volatility: SERDP is in 480 securities' forecast exposures and in none of the 938 estimated periods', so the factor was estimated on a blend without it
   -  Volatility                  -7.7e-18   1.000000   426
   -  Momentum                    +2.3e-17   1.000000   426
   +  Volatility                  +0.0e+00   1.000000   426
   +  Momentum                    -1.5e-17   1.000000   423
   -  Trading Activity            +3.9e-17   1.000000   426
   +  Trading Activity            -3.1e-17   1.000000   425
   -  in the regression         422     451    477          477          422
   +  in the regression         422     451    476          476          422
   -  Construction               17.44%     +21.43%       50.1%      938          0      0
```

### runs_tree2.out -- change 2's tree, and its guard's stop

```
V2     lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [49s]
stopped:   the run stops: a descriptor contributes to a forecast exposure and to no estimated period's, so its factor would be read on a blend it was never estimated on -- Volatility's SERDP, in 466 securities' forecast exposures and none of the 938 estimated periods'
risk-index line with SERDP blended: ['  Volatility           BTSG, DASTD, HILO, LPRI, CMRA, SERDP']
```

### guard_check.txt -- change 2's stopped run (SERDP put back), from tree_2

```
SERDP put back into the blend (UNBLENDED emptied); the run's Model section up to the stop, and its exit message:

Model
  regression weights: square root of capitalisation
  24-factor model, annualised, estimated on 938 daily period(s) for 468 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.294
  0 collinear period(s) of 938 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 43 security-period(s) of the regressions move -- 42 in, on the day a name left the index, 1 out, on the day one joined; the mask itself differs on 62
  each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386); SERDP in none
    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
  ! Volatility: the forecast exposures are blended from BTSG, DASTD, HILO, LPRI, CMRA, SERDP and the last estimated period's from BTSG, DASTD, HILO, LPRI, CMRA

SystemExit:   the run stops: a descriptor contributes to a forecast exposure and to no estimated period's, so its factor would be read on a blend it was never estimated on -- Volatility's SERDP, in 466 securities' forecast exposures and none of the 938 estimated periods'
```

### exact_step2.txt -- change 2 against change 1

```
V1 vs V2: factor returns False, style exposures False, forecast exposures False, specific risks False; summary fields differing: ['vol', 'mean', 'tstat', 'noise', 'spread_shares', 'v2_shares', 'gamma', 'specific', 'blended_names']; report lines differing: 15
   -  Volatility           BTSG, DASTD, HILO, LPRI, CMRA, SERDP
   +  Volatility           BTSG, DASTD, HILO, LPRI, CMRA
   +  - SERDP is in no blend: its residual series is kept for the last window alone, so it has a value on the last date and no other; no estimated period reads it, and blended it would enter the forecast exposures alone
   -  ! Volatility: SERDP is in 480 securities' forecast exposures and in none of the 938 estimated periods', so the factor was estimated on a blend without it
   +  each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
   +    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386)
   +    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
   +    Size                 LNCAP from 2023-02-08 (938)
   +    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
   +    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
   +  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
   -  Volatility                  +0.0e+00   1.000000   426
   +  Volatility                  -3.9e-18   1.000000   426
   -  specific risk   median 23.52%   p10 15.06%   p90 36.34%
   +  specific risk   median 23.53%   p10 15.06%   p90 36.35%
```

### step2.txt -- what change 2 moved

```
factor returns: (938, 24); identical cells 697 of 22512; largest |difference| 2.498e-16, against a typical |factor return| 4.809e-03; relative to each factor's sd at most 5.944e-14
style exposures before the last date: same missing pattern True; largest |difference| per style: Volatility 2.2e-15, Momentum 0.0e+00, Size 0.0e+00, Size Nonlinearity 0.0e+00, Trading Activity 0.0e+00
  Volatility         vol 7.6312022004% -> 7.6312022004%  mean +4.5208441450% -> +4.5208441450%
  Momentum           vol 7.1647237439% -> 7.1647237439%  mean +2.4444100057% -> +2.4444100057%
  Size               vol 7.5658677619% -> 7.5658677619%  mean +4.9168933660% -> +4.9168933660%
  Size Nonlinearity  vol 6.6705083256% -> 6.6705083256%  mean -1.4378072908% -> -1.4378072908%
  Trading Activity   vol 4.9891698587% -> 4.9891698587%  mean +3.9049349848% -> +3.9049349848%
R2 0.29421084071845693 -> 0.29421084071845693; spread 0.3681986132134846 -> 0.3681986132134846
forecast Volatility exposure, 468 securities with a forecast in both: correlation 0.9655, rms change 0.276, mean |change| 0.213, largest 0.987
  largest moves: GL -1.80 -> -0.81, MPWR +1.85 -> +2.74, GEN +1.30 -> +0.42, INFO -3.19 -> -2.33, SBNY +1.34 -> +2.18, CMI -0.72 -> +0.12
every other forecast exposure: largest |change| 0.0e+00
specific risk forecasts: rms change 0.371 points, largest 5.566 points (INFO); median 23.5194% -> 23.5268%
gamma Volatility 0.18773549745640492 -> 0.18773549745640464
floored [] -> []; fallback 8 -> 8; forecasts 468 -> 468
```

### runs_tree3.out -- change 3's tree, and the real-panel fits

```
V3     lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [41s]
fit_index_weights on the real panel: whole-sample, over index members, the blendable descriptors (SERDP out), at lag 0 (same day) and lag 1 (the new default)
Volatility        all descriptors          lag 0: BTSG -0.958, DASTD +6.098, HILO -6.450, LPRI -1.337, CMRA +3.647 (n 166,740)  lag 1: BTSG -0.204, DASTD +1.000, HILO +1.050, LPRI -1.258, CMRA +0.412 (n 166,308)
Volatility        present from the start   lag 0: DASTD -4.260, HILO +7.193, LPRI -0.542, CMRA -1.391 (n 420,521)  lag 1: DASTD +1.904, HILO +1.257, LPRI -1.977, CMRA -0.183 (n 420,081)
Momentum          all descriptors          lag 0: RSTR +2.411, HALPHA -1.411 (n 167,378)  lag 1: RSTR -0.628, HALPHA +1.628 (n 166,946)
Trading Activity  all descriptors          lag 0: STOM -1.985, STOQ +3.159, STOA +1.024, VLVR -1.197 (n 167,345)  lag 1: STOM -0.328, STOQ +1.029, STOA +1.218, VLVR -0.919 (n 166,912)
Trading Activity  present from the start   lag 0: STOM -1.978, STOQ +2.074, STOA +0.904 (n 422,559)  lag 1: STOM -0.270, STOQ +0.546, STOA +0.724 (n 422,115)
```

### exact_step3.txt -- change 3 against change 2

```
V2 vs V3: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 0
```

### runs_tree4.out -- change 4's variants

```
V4     lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [44s]
V4s21  lag 1  R2 0.293636  periods 938  forecasts 468  Volati 7.5863%/+4.39%  Moment 7.1074%/+3.23%  Size 7.6636%/+4.63%  Size N 6.7311%/-1.37%  Tradin 5.0077%/+4.00%  [33s]
V4b0   lag 1  R2 0.294984  periods 917  forecasts 468  Volati 7.6268%/+4.83%  Moment 7.1709%/+1.85%  Size 7.5926%/+4.77%  Size N 6.6920%/-1.36%  Tradin 4.9926%/+4.02%  [32s]
V4b21  lag 1  R2 0.294388  periods 917  forecasts 468  Volati 7.5886%/+4.66%  Moment 7.1538%/+1.75%  Size 7.6839%/+4.59%  Size N 6.7632%/-1.10%  Tradin 4.9851%/+4.12%  [32s]
```

### runs_tree5_a.out -- change 5's default

```
V5     lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [46s; --industries --membership]
```

### runs_tree5_b.out -- change 5: 252 days, equal weights

```
V5w    lag 1  R2 0.298952  periods 962  forecasts 468  Volati 8.1436%/+4.94%  Moment 7.0406%/+3.16%  Size 7.1017%/+4.66%  Size N 6.4016%/-2.55%  Tradin 4.8959%/+4.52%  [60s; --industries --membership --beta-days 252]
```

### runs_tree5_c.out -- change 5: 252 days, half-life 63

```
V5u    lag 1  R2 0.299406  periods 962  forecasts 468  Volati 8.6255%/+4.72%  Moment 6.5735%/+2.74%  Size 6.8966%/+4.90%  Size N 6.3820%/-2.57%  Tradin 4.9838%/+4.59%  [66s; --industries --membership --beta-days 252 --beta-half-life 63]
```

### runs_tree5_d.out -- change 5: 252 days, equal weights, on the default's periods

```
V5w938 lag 1  R2 0.298762  periods 938  forecasts 468  Volati 8.1434%/+3.72%  Moment 7.0406%/+4.12%  Size 7.1018%/+4.25%  Size N 6.4017%/-2.03%  Tradin 4.8961%/+4.11%  [55s]
```

### runs_tree5_e.out -- change 5: 252 / 63 on the default's periods

```
V5u938 lag 1  R2 0.299429  periods 938  forecasts 468  Volati 8.6253%/+3.47%  Moment 6.5735%/+3.64%  Size 6.8968%/+4.69%  Size N 6.3822%/-2.24%  Tradin 4.9840%/+4.11%  [62s]
```

### first_values.txt -- when the market model's descriptors first have values

```
default: 36 months, equal weights: window 828 trading days, effective observations 828.0
  BTSG   first value 2025-04-22 (row 827), on 30 members 2025-04-22 (row 827); the first period that can read it 2025-04-23
  HALPHA first value 2025-04-22 (row 827), on 30 members 2025-04-22 (row 827); the first period that can read it 2025-04-23
  VLVR   first value 2025-04-22 (row 827), on 30 members 2025-04-22 (row 827); the first period that can read it 2025-04-23
USE4: 252 days, half-life 63: window 252 trading days, effective observations 160.4; the oldest day's weight 0.0632, the window's last quarter 53.3% of the weight
  BTSG   first value 2023-01-03 (row 251), on 30 members 2023-01-03 (row 251); the first period that can read it 2023-01-04
  HALPHA first value 2023-01-03 (row 251), on 30 members 2023-01-03 (row 251); the first period that can read it 2023-01-04
  VLVR   first value 2023-02-07 (row 275), on 30 members 2023-02-07 (row 275); the first period that can read it 2023-02-08
compute_descriptors' own default, 60 months: 1380 trading days, capped at the panel's 1214, so one fit, on the last date (window 1214)
```

### runs_final.out -- the finished tree

```
F      lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [45s; --industries --membership]
F_nomem lag 1  R2 0.286071  periods 938  forecasts 468  Volati 7.8218%/+4.98%  Moment 7.5353%/+2.82%  Size 7.6452%/+4.94%  Size N 6.4599%/-1.01%  Tradin 4.8903%/+3.99%  [29s; --industries]
stopped:   the run stops: a descriptor contributes to a forecast exposure and to no estimated period's, so its factor would be read on a blend it was never estimated on -- Volatility's SERDP, in 466 securities' forecast exposures and none of the 938 estimated periods'
```

### exact_final.txt -- the finished tree against change 5's snapshot

```
V5 vs F: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 0
```

### final_guard.txt -- the finished tree's stopped run (SERDP put back)

```
The finished tree with SERDP put back into the blend (UNBLENDED emptied): the Model section up to the stop, and the exit message.

Model
  regression weights: square root of capitalisation
  24-factor model, annualised, estimated on 938 daily period(s) for 468 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.294
  0 collinear period(s) of 938 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 43 security-period(s) of the regressions move -- 42 in, on the day a name left the index, 1 out, on the day one joined; the mask itself differs on 62
  each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386); SERDP in none
    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
  ! Volatility: the forecast exposures are blended from BTSG, DASTD, HILO, LPRI, CMRA, SERDP and the last estimated period's from BTSG, DASTD, HILO, LPRI, CMRA

SystemExit:   the run stops: a descriptor contributes to a forecast exposure and to no estimated period's, so its factor would be read on a blend it was never estimated on -- Volatility's SERDP, in 466 securities' forecast exposures and none of the 938 estimated periods'
```

### analyse.txt -- every table, as analyse.py printed it

```
=== Exactness
  V1_0 vs H (change 1's code at 0 is the committed code): factor returns True, style exposures True, forecast exposures True, specific risks True, summary fields differing none; report lines only in V1_0: 0, only in H: 0
  V1 vs H50 (change 1's default is the committed code run at 0.5): factor returns True, style exposures True, forecast exposures True, specific risks True, summary fields differing none; report lines only in V1: 1, only in H50: 4
  V3 vs V2 (change 3 leaves the run alone): factor returns True, style exposures True, forecast exposures True, specific risks True, summary fields differing none; report lines only in V3: 0, only in V2: 0
  V4 vs V3 (change 4's default is today's RSTR): factor returns True, style exposures True, forecast exposures True, specific risks True, summary fields differing none; report lines only in V4: 0, only in V3: 0
  V5 vs V4 (change 5's default is today's market model): factor returns True, style exposures True, forecast exposures True, specific risks True, summary fields differing none; report lines only in V5: 1, only in V4: 0
  V2 vs V1: factor returns differ by at most 2.50e-16 (697 of 22512 cells identical) -- the blend's equal weights are 1/5 where they were 1/6

=== Per variant
                 R2  periods       first  univ  fcst Volatility   Momentum       Size Size Nonli Trading Ac
  H        0.294197      938  2023-02-08   482   470     7.61%      7.22%      7.58%      6.69%      5.30%  HEAD 6e29b2c, as committed (min_coverage 0)
  H50      0.294211      938  2023-02-08   482   468     7.63%      7.16%      7.57%      6.67%      4.99%  HEAD run with --min-coverage 0.5
  V1_0     0.294197      938  2023-02-08   482   470     7.61%      7.22%      7.58%      6.69%      5.30%  change 1's tree at --min-coverage 0
  V1       0.294211      938  2023-02-08   482   468     7.63%      7.16%      7.57%      6.67%      4.99%  change 1: min_coverage default 0.5
  V2       0.294211      938  2023-02-08   482   468     7.63%      7.16%      7.57%      6.67%      4.99%  change 2: SERDP in no blend; the guard
  V3       0.294211      938  2023-02-08   482   468     7.63%      7.16%      7.57%      6.67%      4.99%  change 3: fit_index_weights lagged
  V4       0.294211      938  2023-02-08   482   468     7.63%      7.16%      7.57%      6.67%      4.99%  change 4: rstr_skip, default 0
  V4s21    0.293636      938  2023-02-08   482   468     7.59%      7.11%      7.66%      6.73%      5.01%  V4 with --rstr-skip 21
  V4b0     0.294984      917  2023-03-10   481   468     7.63%      7.17%      7.59%      6.69%      4.99%  reading (b), its matched baseline
  V4b21    0.294388      917  2023-03-10   481   468     7.59%      7.15%      7.68%      6.76%      4.99%  reading (b): 276 days ending 21 back
  V5       0.294211      938  2023-02-08   482   468     7.63%      7.16%      7.57%      6.67%      4.99%  change 5: beta_half_life, default none
  V5w      0.298952      962  2023-01-04   483   468     8.14%      7.04%      7.10%      6.40%      4.90%  V5 with --beta-days 252
  V5u      0.299406      962  2023-01-04   483   468     8.63%      6.57%      6.90%      6.38%      4.98%  V5 with --beta-days 252 --beta-half-life 63
  V5w938   0.298762      938  2023-02-08   482   468     8.14%      7.04%      7.10%      6.40%      4.90%  V5w held to the default's 938 periods
  V5u938   0.299429      938  2023-02-08   482   468     8.63%      6.57%      6.90%      6.38%      4.98%  V5u held to the default's 938 periods

  style mean by variant
          Volatilit  Momentum      Size Size Nonl Trading A
  H          +4.51%    +2.44%    +4.92%    -1.44%    +3.91%
  H50        +4.52%    +2.44%    +4.92%    -1.44%    +3.90%
  V1_0       +4.51%    +2.44%    +4.92%    -1.44%    +3.91%
  V1         +4.52%    +2.44%    +4.92%    -1.44%    +3.90%
  V2         +4.52%    +2.44%    +4.92%    -1.44%    +3.90%
  V3         +4.52%    +2.44%    +4.92%    -1.44%    +3.90%
  V4         +4.52%    +2.44%    +4.92%    -1.44%    +3.90%
  V4s21      +4.39%    +3.23%    +4.63%    -1.37%    +4.00%
  V4b0       +4.83%    +1.85%    +4.77%    -1.36%    +4.02%
  V4b21      +4.66%    +1.75%    +4.59%    -1.10%    +4.12%
  V5         +4.52%    +2.44%    +4.92%    -1.44%    +3.90%
  V5w        +4.94%    +3.16%    +4.66%    -2.55%    +4.52%
  V5u        +4.72%    +2.74%    +4.90%    -2.57%    +4.59%
  V5w938     +3.72%    +4.12%    +4.25%    -2.03%    +4.11%
  V5u938     +3.47%    +3.64%    +4.69%    -2.24%    +4.11%

  style tstat by variant
          Volatilit  Momentum      Size Size Nonl Trading A
  H           +1.36     +0.81     +1.54     -0.54     +2.11
  H50         +1.37     +0.81     +1.54     -0.54     +2.14
  V1_0        +1.36     +0.81     +1.54     -0.54     +2.11
  V1          +1.37     +0.81     +1.54     -0.54     +2.14
  V2          +1.37     +0.81     +1.54     -0.54     +2.14
  V3          +1.37     +0.81     +1.54     -0.54     +2.14
  V4          +1.37     +0.81     +1.54     -0.54     +2.14
  V4s21       +1.33     +1.09     +1.44     -0.51     +2.18
  V4b0        +1.43     +0.61     +1.46     -0.50     +2.16
  V4b21       +1.39     +0.58     +1.40     -0.40     +2.21
  V5          +1.37     +0.81     +1.54     -0.54     +2.14
  V5w         +1.37     +1.07     +1.56     -0.99     +2.82
  V5u         +1.30     +0.94     +1.65     -1.01     +2.86
  V5w938      +1.02     +1.38     +1.40     -0.78     +2.54
  V5u938      +0.94     +1.22     +1.56     -0.86     +2.53

  style noise by variant
          Volatilit  Momentum      Size Size Nonl Trading A
  H            5.4%      3.9%      9.8%     17.0%     10.5%
  H50          5.3%      4.0%      9.9%     17.1%     11.5%
  V1_0         5.4%      3.9%      9.8%     17.0%     10.5%
  V1           5.3%      4.0%      9.9%     17.1%     11.5%
  V2           5.3%      4.0%      9.9%     17.1%     11.5%
  V3           5.3%      4.0%      9.9%     17.1%     11.5%
  V4           5.3%      4.0%      9.9%     17.1%     11.5%
  V4s21        5.4%      4.0%      9.6%     16.8%     11.4%
  V4b0         5.4%      4.0%      9.8%     17.1%     11.6%
  V4b21        5.4%      4.0%      9.6%     16.7%     11.6%
  V5           5.3%      4.0%      9.9%     17.1%     11.5%
  V5w          4.5%      3.8%     11.0%     18.5%     10.6%
  V5u          3.9%      4.4%     11.6%     18.6%     10.0%
  V5w938       4.5%      3.9%     11.0%     18.5%     10.6%
  V5u938       3.9%      4.4%     11.6%     18.6%     10.0%

=== Steps (from -> to: dR2, and each style's vol change in percent and mean change in points a year)
       H -> V1     +0.000014  Volati  +0.3%/+0.01  Moment  -0.8%/+0.01  Size  -0.1%/-0.00  Size N  -0.2%/+0.00  Tradin  -5.9%/-0.00  change 1: min_coverage 0.5
      V1 -> V2     +0.000000  Volati  -0.0%/+0.00  Moment  -0.0%/-0.00  Size  +0.0%/-0.00  Size N  +0.0%/-0.00  Tradin  -0.0%/-0.00  change 2: SERDP out
      V2 -> V3     +0.000000  Volati  +0.0%/+0.00  Moment  +0.0%/+0.00  Size  +0.0%/+0.00  Size N  +0.0%/+0.00  Tradin  +0.0%/+0.00  change 3: fit_index_weights
      V3 -> V4     +0.000000  Volati  +0.0%/+0.00  Moment  +0.0%/+0.00  Size  +0.0%/+0.00  Size N  +0.0%/+0.00  Tradin  +0.0%/+0.00  change 4: default
      V4 -> V5     +0.000000  Volati  +0.0%/+0.00  Moment  +0.0%/+0.00  Size  +0.0%/+0.00  Size N  +0.0%/+0.00  Tradin  +0.0%/+0.00  change 5: default
       H -> V5     +0.000014  Volati  +0.3%/+0.01  Moment  -0.8%/+0.01  Size  -0.1%/-0.00  Size N  -0.2%/+0.00  Tradin  -5.9%/-0.00  combined, at the defaults
      V4 -> V4s21  -0.000575  Volati  -0.6%/-0.13  Moment  -0.8%/+0.79  Size  +1.3%/-0.29  Size N  +0.9%/+0.07  Tradin  +0.4%/+0.09  RSTR skip 21 (the brief's reading)
    V4b0 -> V4b21  -0.000596  Volati  -0.5%/-0.17  Moment  -0.2%/-0.09  Size  +1.2%/-0.18  Size N  +1.1%/+0.27  Tradin  -0.2%/+0.10  RSTR 21 back (reading b, matched)
      V5 -> V5w    +0.004741  Volati  +6.7%/+0.42  Moment  -1.7%/+0.71  Size  -6.1%/-0.25  Size N  -4.0%/-1.11  Tradin  -1.9%/+0.62  252-day window, as run
     V5w -> V5u    +0.000454  Volati  +5.9%/-0.22  Moment  -6.6%/-0.41  Size  -2.9%/+0.23  Size N  -0.3%/-0.03  Tradin  +1.8%/+0.07  + half-life 63, as run
      V5 -> V5u    +0.005195  Volati +13.0%/+0.20  Moment  -8.3%/+0.30  Size  -8.8%/-0.02  Size N  -4.3%/-1.14  Tradin  -0.1%/+0.69  USE4's market model, as run
      V5 -> V5w938 +0.004551  Volati  +6.7%/-0.80  Moment  -1.7%/+1.68  Size  -6.1%/-0.67  Size N  -4.0%/-0.60  Tradin  -1.9%/+0.21  252-day window, matched
  V5w938 -> V5u938 +0.000667  Volati  +5.9%/-0.25  Moment  -6.6%/-0.48  Size  -2.9%/+0.44  Size N  -0.3%/-0.20  Tradin  +1.8%/-0.00  + half-life 63, matched
      V5 -> V5u938 +0.005218  Volati +13.0%/-1.05  Moment  -8.3%/+1.20  Size  -8.8%/-0.23  Size N  -4.3%/-0.80  Tradin  -0.1%/+0.21  USE4's market model, matched

=== Momentum at RSTR skip 0 and 21
  V4     Momentum vol 7.16%, mean +2.44% (t +0.81), noise 4.0%; R2 0.294211 over 938; screen RSTR: 938 periods, R2 5.31%, vol 7.12%, t +1.11, cover 93%
  V4s21  Momentum vol 7.11%, mean +3.23% (t +1.09), noise 4.0%; R2 0.293636 over 938; screen RSTR: 938 periods, R2 5.21%, vol 7.03%, t +1.33, cover 94%
  V4b0   Momentum vol 7.17%, mean +1.85% (t +0.61), noise 4.0%; R2 0.294984 over 917; screen RSTR: 917 periods, R2 5.32%, vol 7.14%, t +0.87, cover 93%
  V4b21  Momentum vol 7.15%, mean +1.75% (t +0.58), noise 4.0%; R2 0.294388 over 917; screen RSTR: 917 periods, R2 5.33%, vol 7.17%, t +0.67, cover 93%
  RSTR cells: with a value at skip 0 only 99, at 21 only 256, both 451,543; correlation of the raw values pooled 0.956, across securities by day median 0.967, lowest 0.657

=== The market model: screen rows for its descriptors, and the blends' composition
  V5      rank: BTSG, DASTD, RSTR, HILO, CMRA, HALPHA, STOQ, STOM, STOA, LNCAP, LPRI, LCAPCB, VLVR
          BTSG     386 periods  R2 8.43%  vol 9.77%  t +1.87  cover 95%
          HALPHA   386 periods  R2 3.38%  vol 6.16%  t -0.29  cover 95%
          VLVR     386 periods  R2 1.52%  vol 3.94%  t +0.58  cover 93%
          SERDP      0 periods  too few usable periods
  V5w     rank: BTSG, DASTD, RSTR, HALPHA, HILO, CMRA, STOQ, STOM, STOA, LNCAP, LPRI, VLVR, LCAPCB
          BTSG     962 periods  R2 8.03%  vol 9.09%  t +1.40  cover 93%
          HALPHA   962 periods  R2 4.75%  vol 6.57%  t +1.02  cover 93%
          VLVR     938 periods  R2 1.74%  vol 3.73%  t +1.95  cover 93%
          SERDP      0 periods  too few usable periods
  V5u     rank: BTSG, DASTD, RSTR, HALPHA, HILO, CMRA, STOQ, STOM, STOA, LNCAP, VLVR, LPRI, LCAPCB
          BTSG     962 periods  R2 8.36%  vol 9.32%  t +1.34  cover 93%
          HALPHA   962 periods  R2 4.43%  vol 6.30%  t +0.56  cover 93%
          VLVR     938 periods  R2 1.78%  vol 3.76%  t +1.95  cover 93%
          SERDP      0 periods  too few usable periods
  V5w938  rank: BTSG, RSTR, DASTD, HALPHA, CMRA, HILO, STOA, STOQ, STOM, LNCAP, LCAPCB, VLVR, LPRI
          BTSG     938 periods  R2 7.97%  vol 9.10%  t +1.02  cover 93%
          HALPHA   938 periods  R2 4.75%  vol 6.56%  t +1.32  cover 93%
          VLVR     938 periods  R2 1.74%  vol 3.73%  t +1.95  cover 93%
          SERDP      0 periods  too few usable periods
  V5u938  rank: BTSG, RSTR, DASTD, HALPHA, CMRA, HILO, STOA, STOQ, STOM, LNCAP, VLVR, LCAPCB, LPRI
          BTSG     938 periods  R2 8.30%  vol 9.32%  t +0.96  cover 93%
          HALPHA   938 periods  R2 4.48%  vol 6.35%  t +0.78  cover 93%
          VLVR     938 periods  R2 1.78%  vol 3.76%  t +1.95  cover 93%
          SERDP      0 periods  too few usable periods
  V2:
    each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386)
    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
    the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
  V5:
    each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386)
    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
    the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
  V5w:
    each risk index's blend over the 962 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI from 2023-01-04 (962); CMRA from 2023-02-08 (938)
    Momentum             HALPHA from 2023-01-04 (962); RSTR from 2023-02-08 (938)
    Size                 LNCAP from 2023-01-04 (962)
    Size Nonlinearity    LCAPCB from 2023-01-04 (962)
    Trading Activity     STOM, STOQ from 2023-01-04 (962); STOA, VLVR from 2023-02-08 (938)
    the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
  V5u:
    each risk index's blend over the 962 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI from 2023-01-04 (962); CMRA from 2023-02-08 (938)
    Momentum             HALPHA from 2023-01-04 (962); RSTR from 2023-02-08 (938)
    Size                 LNCAP from 2023-01-04 (962)
    Size Nonlinearity    LCAPCB from 2023-01-04 (962)
    Trading Activity     STOM, STOQ from 2023-01-04 (962); STOA, VLVR from 2023-02-08 (938)
    the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
  V5w938:
    each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2023-02-08 (938)
    Momentum             RSTR, HALPHA from 2023-02-08 (938)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA, VLVR from 2023-02-08 (938)
    the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
  V5u938:
    each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2023-02-08 (938)
    Momentum             RSTR, HALPHA from 2023-02-08 (938)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA, VLVR from 2023-02-08 (938)
    the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
  V4s21:
    each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386)
    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
    the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index

=== Forecasts and specific risk
  H       forecasts 470, universe 482, fallback 10 (ECHO, INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VMRK, VNO); floored none; spread 0.3679; specific p1/p10/p50/p90/p99 9.51/15.11/23.52/36.36/53.35
  V1      forecasts 468, universe 482, fallback 8 (INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO); floored none; spread 0.3682; specific p1/p10/p50/p90/p99 9.50/15.06/23.52/36.34/53.35
  V2      forecasts 468, universe 482, fallback 8 (INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO); floored none; spread 0.3682; specific p1/p10/p50/p90/p99 9.50/15.06/23.53/36.35/54.52
  V5      forecasts 468, universe 482, fallback 8 (INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO); floored none; spread 0.3682; specific p1/p10/p50/p90/p99 9.50/15.06/23.53/36.35/54.52
  V4s21   forecasts 468, universe 482, fallback 8 (INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO); floored none; spread 0.3733; specific p1/p10/p50/p90/p99 9.50/15.03/23.54/36.36/54.31
  V5u     forecasts 468, universe 483, fallback 7 (INFO, IPGP, MRSH, PENN, PVH, UA, UAA); floored none; spread 0.2793; specific p1/p10/p50/p90/p99 9.55/15.18/23.43/36.06/53.82
  V5u938  forecasts 468, universe 482, fallback 8 (INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO); floored none; spread 0.3632; specific p1/p10/p50/p90/p99 9.53/15.10/23.46/36.35/54.24
```

### pytest_mid.txt -- the midoffice tests, last lines

```
tests\midoffice\test_valuation.py ..............................         [100%]

======================= 774 passed, 1 skipped in 21.94s =======================
```

### ruff.txt -- ruff on the changed Python files

```
scripts\run_sp500_model.py:34:5: F401 [*] `praxis.midoffice.model.covariance.HorizonCovariance` imported but unused
Found 1 error.
[*] 1 fixable with the `--fix` option.
```

### variant_V5.txt -- the run's own report, change 5: `beta_half_life`, at its default none (the defaults throughout)

```
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
  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 828 trading days (36 months), every day weighted equally
  window descriptors need at least 50% of the window valid; from a window with none, over 3,581,487 values: none; from one under 50% valid: none
  344,062 raw values winsorised before standardising
  95,646 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/46  HILO 8/45  CMRA 5/46  BTSG 6/59
    HALPHA 6/59  SERDP 1/59  RSTR 3/49  STOM 13/45  STOQ 14/46  STOA 16/49  VLVR 17/49

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         95%      386    8.43%      9.77%     1.87
  DASTD        93%    1,149    6.13%      7.77%     1.36
  RSTR         93%      938    5.31%      7.12%     1.11
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         92%      938    3.39%      5.59%     1.37
  HALPHA       95%      386    3.38%      6.16%    -0.29
  STOQ         93%    1,145    2.99%      5.21%     1.26
  STOM         93%    1,191    2.95%      5.25%     1.28
  STOA         93%      938    2.57%      4.85%     1.50
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  LPRI         93%    1,213    1.77%      3.77%    -0.41
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  VLVR         93%      386    1.52%      3.94%     0.58
  SERDP           -        0   too few usable periods

Risk indices
  Volatility           BTSG, DASTD, HILO, LPRI, CMRA
  Momentum             RSTR, HALPHA
  Size                 LNCAP
  Size Nonlinearity    LCAPCB
  Trading Activity     STOM, STOQ, STOA, VLVR
  - SERDP is in no blend: its residual series is kept for the last window alone, so it has a value on the last date and no other; no estimated period reads it, and blended it would enter the forecast exposures alone

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
  24-factor model, annualised, estimated on 938 daily period(s) for 468 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.294
  0 collinear period(s) of 938 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 43 security-period(s) of the regressions move -- 42 in, on the day a name left the index, 1 out, on the day one joined; the mask itself differs on 62
  each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386)
    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 3.3e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  -3.9e-18   1.000000   426
  Momentum                    -1.5e-17   1.000000   423
  Size                        -3.1e-17   1.000000   425
  Size Nonlinearity           -7.7e-18   1.000000   425
  Trading Activity            -3.1e-17   1.000000   425

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     504    506          504          503
  of them in the panel      426     458    486          486          426
  in the regression         422     451    476          476          422

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               17.46%     +21.40%       50.0%      938          0      0
  Finance & Real Estate      15.70%     +22.66%       13.8%      938          0      0
  Manufacturing              11.24%     +21.79%       19.7%      938          0      0
  Mining                     20.63%     +12.34%       15.2%      938          0      0
  Retail                     13.35%     +13.88%       10.3%      938          0      0
  SIC20                      13.19%      +9.12%       29.0%      938          0      0
  SIC28                      13.52%     +15.95%        9.6%      938          0      0
  SIC35                      17.16%     +26.44%        6.5%      938          0      0
  SIC36                      20.26%     +28.08%        5.5%      938          0      0
  SIC37                      13.96%     +13.79%       18.7%      938          0      0
  SIC38                      14.49%     +15.58%       11.0%      938          0      0
  SIC49                      14.24%     +23.20%       15.2%      938          0      0
  SIC60                      20.61%     +21.72%        8.4%      938          0      0
  SIC63                      14.76%     +19.10%       14.0%      938          0      0
  SIC67                      15.13%     +17.37%       17.5%      938          0      0
  SIC73                      17.28%     +13.53%        3.6%      938          0      0
  Services                   13.16%     +14.33%       25.4%      938          0      0
  Transport & Utilities      13.41%     +13.28%       16.9%      938          0      0
  Wholesale                  14.50%     +19.90%       42.0%      938          0      0
  Volatility                  7.63%      +4.52%        5.3%      938          0      0
  Momentum                    7.16%      +2.44%        4.0%      938          0      0
  Size                        7.57%      +4.92%        9.9%      938          0      0
  Size Nonlinearity           6.67%      -1.44%       17.1%      938          0      0
  Trading Activity            4.99%      +3.90%       11.5%      938          0      0

  specific risk   median 23.53%   p10 15.06%   p90 36.35%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
                    relative residual sd 0.368, largest shares of its square: FRC 71.3%, SIVB 4.0%, LUMN 2.9%, BNY 1.5%, DISH 1.4%

Qualifications
  - 275 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 272 more
  - 38 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CDAY
  - 50% of the estimated variance of Construction is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 8 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: INFO, IPGP, MRSH, PENN, PVH
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### variant_V5u.txt -- the run's own report, V5 with `--beta-days 252 --beta-half-life 63`

```
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
  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days
  window descriptors need at least 50% of the window valid; from a window with none, over 4,402,359 values: none; from one under 50% valid: none
  369,070 raw values winsorised before standardising
  102,179 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/46  HILO 8/45  CMRA 5/46  BTSG 7/49
    HALPHA 4/49  SERDP 0/49  RSTR 3/49  STOM 13/45  STOQ 14/46  STOA 16/49  VLVR 16/49

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         93%      962    8.36%      9.32%     1.34
  DASTD        93%    1,149    6.13%      7.77%     1.36
  RSTR         93%      938    5.31%      7.12%     1.11
  HALPHA       93%      962    4.43%      6.30%     0.56
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         92%      938    3.39%      5.59%     1.37
  STOQ         93%    1,145    2.99%      5.21%     1.26
  STOM         93%    1,191    2.95%      5.25%     1.28
  STOA         93%      938    2.57%      4.85%     1.50
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  VLVR         93%      938    1.78%      3.76%     1.95
  LPRI         93%    1,213    1.77%      3.77%    -0.41
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  SERDP           -        0   too few usable periods

Risk indices
  Volatility           BTSG, DASTD, HILO, LPRI, CMRA
  Momentum             RSTR, HALPHA
  Size                 LNCAP
  Size Nonlinearity    LCAPCB
  Trading Activity     STOM, STOQ, STOA, VLVR
  - SERDP is in no blend: its residual series is kept for the last window alone, so it has a value on the last date and no other; no estimated period reads it, and blended it would enter the forecast exposures alone

Industries
  33 classification(s) carried from an earlier date (delisted inside the window): ABMD, ANSS, ATVI, AVB, CDAY
  483 of 506 securities are in at least one estimated period's regression while an index member; every count below is over them
  19 industries over 483 securities in the universe (506 classified); largest SIC73 with 53 (11%), 45 thin group(s) merged into their division
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
  - 23 classified security(ies) are outside the universe and count toward no industry: ABMD, CERN, CTXS, DISCA, DISCK

Model
  regression weights: square root of capitalisation
  24-factor model, annualised, estimated on 962 daily period(s) for 468 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.4 effective observation(s) of 962
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.299
  0 collinear period(s) of 962 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 44 security-period(s) of the regressions move -- 43 in, on the day a name left the index, 1 out, on the day one joined; the mask itself differs on 63
  each risk index's blend over the 962 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI from 2023-01-04 (962); CMRA from 2023-02-08 (938)
    Momentum             HALPHA from 2023-01-04 (962); RSTR from 2023-02-08 (938)
    Size                 LNCAP from 2023-01-04 (962)
    Size Nonlinearity    LCAPCB from 2023-01-04 (962)
    Trading Activity     STOM, STOQ from 2023-01-04 (962); STOA, VLVR from 2023-02-08 (938)
  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index

  Style exposures over the index members with a value, on all 963 dates an estimated period or the forecast reads: cap-weighted mean at most 2.2e-16 from zero (1.0 eps), equal-weighted sd at most 3.3e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  +1.2e-17   1.000000   426
  Momentum                    -5.4e-17   1.000000   423
  Size                        -3.1e-17   1.000000   425
  Size Nonlinearity           -7.7e-18   1.000000   425
  Trading Activity            -7.7e-18   1.000000   425

  Securities per estimated period, over 962
                            min  median    max   2023-01-04   2026-11-03
  index members             503     504    506          504          503
  of them in the panel      426     458    487          487          426
  in the regression         422     451    477          475          422

  Over the 962 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               17.13%     +26.09%       51.5%      962          0      0
  Finance & Real Estate      15.60%     +24.85%       13.8%      962          0      0
  Manufacturing              11.09%     +22.89%       20.0%      962          0      0
  Mining                     20.26%     +12.00%       15.6%      962          0      0
  Retail                     13.13%     +14.41%       10.6%      962          0      0
  SIC20                      12.87%      +8.84%       29.9%      962          0      0
  SIC28                      13.44%     +16.18%        9.5%      962          0      0
  SIC35                      17.35%     +28.23%        6.4%      962          0      0
  SIC36                      20.72%     +31.19%        5.2%      962          0      0
  SIC37                      13.99%     +16.30%       18.4%      962          0      0
  SIC38                      14.00%     +16.64%       11.6%      962          0      0
  SIC49                      15.54%     +21.75%       12.5%      962          0      0
  SIC60                      21.43%     +22.85%        7.7%      962          0      0
  SIC63                      14.74%     +19.22%       13.9%      962          0      0
  SIC67                      15.22%     +19.30%       17.0%      962          0      0
  SIC73                      17.29%     +17.28%        3.5%      962          0      0
  Services                   12.93%     +17.69%       26.0%      962          0      0
  Transport & Utilities      13.30%     +16.12%       16.9%      962          0      0
  Wholesale                  14.50%     +20.84%       41.3%      962          0      0
  Volatility                  8.63%      +4.72%        3.9%      962          0      0
  Momentum                    6.57%      +2.74%        4.4%      962          0      0
  Size                        6.90%      +4.90%       11.6%      962          0      0
  Size Nonlinearity           6.38%      -2.57%       18.6%      962          0      0
  Trading Activity            4.98%      +4.59%       10.0%      962          0      0

  specific risk   median 23.43%   p10 15.18%   p90 36.06%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.5%, FRC 2.0%, ENPH 1.3%, ALB 0.9%, CTVA 0.9%
                    relative residual sd 0.279, largest shares of its square: FRC 57.9%, VNO 3.7%, LUMN 3.2%, BNY 2.5%, ENPH 1.8%

Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 38 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CDAY
  - 51% of the estimated variance of Construction is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 7 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: INFO, IPGP, MRSH, PENN, PVH
  - 252 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

## git add

The separate change first -- midoffice01's Source lines, its own commit:

```
git add claude/handoffs/BRIEF_midoffice01_standardise_over_members.md claude/retros/RETRO_midoffice01_standardise_over_members.md
```

This change -- midoffice03, every file, the brief and this retro included:

```
git add src/praxis/midoffice/model/descriptors.py scripts/run_sp500_model.py tests/midoffice/test_descriptors.py tests/midoffice/test_lookahead.py docs/FACTOR_MODEL.md claude/handoffs/BRIEF_midoffice03_forecast_agrees_with_estimation.md claude/retros/RETRO_midoffice03_forecast_agrees_with_estimation.md
```
