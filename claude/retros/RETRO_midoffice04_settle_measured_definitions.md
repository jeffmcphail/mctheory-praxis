# Retro: Mid-office 04 -- settle the definitions midoffice03 measured

**Brief:** `claude/handoffs/BRIEF_midoffice04_settle_measured_definitions.md` -- written in
Chat and pasted into the Code session, transcribed verbatim
**Date:** 2026-11-09
**Duration:** about 49 minutes of implementation and measurement (10:26-11:15 EST: from the brief's arrival, by the transcript, to the last measurement, by the scratch file times), then this retro
**Status:** COMPLETE, uncommitted. All five items done in the brief's order, each measured
against the one before. Decisions left for Jeff are listed below.

---

## Summary

Five changes, in the brief's order, each measured against the one before.

1. **The market model is USE4's by default.** 252 trading days with half-life 63, set in
   `compute_descriptors` itself, and the run no longer overrides it. The new defaults are
   midoffice03's USE4 variant bit for bit. `--beta-window 36 --beta-half-life inf` reproduces
   the committed run bit for bit; its report differs in the market-model line alone.
2. **One blend per factor.** A period is estimated only if every risk index's blend in it has
   exactly the forecast's descriptors, and the run asserts this of every estimated period.
   - It drops 24 periods, 2023-01-04 to 2023-02-07. In them Momentum is HALPHA alone,
     Volatility has no CMRA, and Trading Activity has no STOA or VLVR.
   - The estimation starts on 2023-02-08: the same 938 periods, each one's factor returns item
     1's bit for bit. VNO leaves the universe.
3. **Size Nonlinearity is USE4's by default.** The Size exposure is cubed, standardised and
   clipped at +/-4, orthogonalised against Size by the regression weights, then standardised
   again.
   - Its regression-weighted covariance with Size is at most 1.0e-15 on all 939 dates read;
     the run stops otherwise.
   - Size's factor return is now what it is without Size Nonlinearity in the model:
     correlation 0.9998, against 0.7935 with the raw cube.
   - Size's volatility falls from 6.90% to 4.03% and Size Nonlinearity's from 6.38% to 2.54%.
     The two factor returns' correlation goes from -0.824 to -0.308, and mean R-squared from
     0.299429 to 0.298076.
   - "use3" is the raw cube, bit for bit.
4. **The residual spread is 1.4826 x the MAD by default:** 0.1333 on the 938 periods and 0.1334
   on the 962, where the standard deviation goes from 0.3612 to 0.2805. The characteristics'
   blend weight at the median history rises from 0.41% to 2.92%. The delivered forecasts move
   0.677 points rms, their median from 23.40% to 23.77%.
5. **RSTR's skip stays 0.** Only its docstring changes: it cuts the window's last k days, the
   academic twelve-minus-one reading, and USE4's lag, a full window ending k days back, was
   measured and not built. The run is item 4's bit for bit, report included.

Tests: 2,231 passed, 1 skipped, 6 of them new; the midoffice tests are 780.

## What was asked

The brief, verbatim (it is also the `BRIEF_` file):

```text
committed -- the midoffice01 provenance fix, then midoffice03.

Next piece, midoffice04: settle the definitions midoffice03 measured.
Same discipline: in this order, each change decomposed against the
one before it, retro and git add line at the end, then stop.

The policy behind items 1 to 3: USE3's definitions by default, and
USE4's wherever USE3's choice produces a measured problem on this
panel, each kept switchable.

1. The market model: default to 252 trading days with half-life 63,
   USE4's beta, in compute_descriptors itself and in the run, so the
   run no longer overrides the library. 828 equal-weighted days stays
   runnable. Today's choice keeps beta, the screen's strongest
   descriptor, out of 552 of 938 periods; USE4's puts it in all of
   them.

2. One definition per factor across the sample: a period is estimated
   only if every risk index's blend in it has exactly the descriptors
   the forecast's has. Extend midoffice03's guard from the last
   estimated period to every estimated period, as an assertion. Under
   item 1 this should start the estimation on 2023-02-08, the same 938
   periods, and drop the 24 earlier ones in which Momentum is HALPHA
   alone. Report what it drops.

3. The size pair: build Size Nonlinearity USE4's way, as a parameter --
   size_nonlinearity="use4" by default, "use3" keeping today's raw cube
   runnable. Cube the Size exposure; treat the cube like any descriptor
   except for the raw MAD clip, which on a cube binds at |z| about 1.2
   -- so standardise it and clip at +/-4; then orthogonalise it against
   Size, per period, with the regression weights, over the estimation
   universe; then the final exact standardisation.
   Guard: on every period, the regression-weighted covariance of the
   Size and Size Nonlinearity exposures over the estimation universe is
   zero to machine precision.
   Report against the raw cube: the two exposures' correlation (0.93
   now), their factor returns' correlation (-0.87), both vols and
   means, the pair's risk for the smallest 85 members, the largest 50
   and the index, R-squared, how many members the clip moves, and
   Size's factor returns with and without Size Nonlinearity in the
   model. Under USE4 those two series should be close; under USE3 they
   are not.

4. The residual spread in EQ 3-14's V fit: make its scale a parameter,
   defaulting to a MAD-based one (1.4826 x the MAD), with the standard
   deviation runnable. FRC is 71% of the squared spread today, and
   moving the sample start by 24 days moved the spread from 0.368 to
   0.279. Report the spread on the 938- and 962-period samples under
   both scales -- the MAD one should barely move -- and what it does
   to the blend weights and the delivered forecasts.

5. RSTR's skip stays 0: at 21 it changed nothing material and cost a
   little R-squared. Say in the parameter's docstring that it cuts the
   window's last k days, the academic twelve-minus-one reading, and
   that USE4's lag is a full window ending k days back, measured in
   midoffice03 and not built.
```

## Files modified

| File | Change | Lines (new file) |
|---|---|---|
| `src/praxis/midoffice/model/descriptors.py` | item 1: `USE4_BETA_WINDOW`, `USE4_BETA_HALF_LIFE`, `USE3_BETA_WINDOW_MONTHS`, `market_model_window`; `compute_descriptors`' beta defaults and docstring. Item 3: `SIZE_NONLINEARITY`, `index_descriptors`, `nonlinear_size`; `build_risk_indices` and `blend_composition` take `size_nonlinearity`. Item 5: `rstr_skip`'s docstring | 45-53, 269-302, 376, 417-436, 461, 619, 699, 743-827, 939-940, 954-959, 966, 974-984, 991-1009, 1020-1021, 1030-1037 |
| `src/praxis/midoffice/model/crosssectional.py` | item 2: `periods` on `estimate_cross_sectional_model` and `regression_universe`, `_periods_block`, `CrossSectionalNotes.excluded_periods` and its qualification. Item 4: `residual_spread` passed through and validated | 101, 133-135, 238-248, 466-481, 595, 611-612, 630-633, 645, 659, 684-690, 716-718, 726, 773, 792-794, 910-913, 955, 1046 |
| `src/praxis/midoffice/model/specificrisk.py` | item 4: `RESIDUAL_SPREADS`; `residual_spread` on `estimate_specific_risk_model`, the MAD spread by default; notes carry both measures | 82-85, 102-108, 402, 417-430, 512-519, 618-620 |
| `scripts/run_sp500_model.py` | item 1: no beta override, `market_model_window` for the standing line and the coverage check, `--beta-half-life inf`. Item 2: the rule before the industries, `periods` to the universe and the estimator, the assertion on every estimated period, what it drops, `--mixed-blends`. Item 3: weights before the risk indices, `--size-nonlinearity`, LCAPCB in no blend under use4, `size_pair_moments` and the orthogonality guard. Item 4: `--residual-spread`, the V-fit lines and the model-weight line. Item 5: `--rstr-skip`'s help | 43-54, 211, 225-226, 234, 307-350, 393-396, 420-444, 525-535, 543, 600-671, 699-700, 730-731, 784-789, 796-820, 827-835, 842-867, 902-936, 979-1002 |
| `tests/midoffice/test_descriptors.py` | 4 new tests (market-model defaults; USE4's Size Nonlinearity; its clip; use3), 2 changed (half-life test; the skip test's docstring) | 17, 33-35, 139-145, 184-190, 200, 225-266, 854-953 |
| `tests/midoffice/test_crosssectional.py` | 1 new test (`periods`); 3 changed (two blend tests pin `residual_spread="sd"`; the unknown-method test refuses an unknown spread) | 823-836, 881-882, 1002-1004, 1131-1166 |
| `tests/midoffice/test_specificrisk.py` | 1 new test (the MAD spread); 2 changed (the shares test pins sd; the monotone-weight test adds n = 10,000) | 191, 408-418, 428-471 |
| `docs/FACTOR_MODEL.md` | the SERDP paragraph's end; new "One definition per factor across the sample"; "Two definitions, measured" becomes "USE3 by default, USE4 where USE3's choice measures badly"; new "Size Nonlinearity, USE4's way" with its table; the MAD spread paragraph | 417-441, 621-633, 675-742 |
| `claude/handoffs/BRIEF_midoffice04_settle_measured_definitions.md` | new: the brief, verbatim | 1-73 |
| `claude/retros/RETRO_midoffice04_settle_measured_definitions.md` | new: this retro | all |

## How it was measured

The method is midoffice03's:

- **Snapshots.** The source tree is snapshotted after each change (`tree_0` to `tree_5`, from
  `git ls-files -co --exclude-standard src scripts`).
- **Runs.** The run script's `main()` is run from each snapshot, with wrappers that keep what
  each step was handed. That is `common.py` and `runv.py`, extended here to keep:
  - the model's covariance matrix;
  - the histories the specific-risk blend was given;
  - the periods the estimator excluded;
  - the regression universe.
- **Comparisons.** Variants are compared bit for bit by `exact.py`: factor returns, style
  exposures, forecast exposures, specific risks, every summary field, and the report's text,
  diffed.
- **Inputs.** Every run is `--industries --membership` on
  `data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz`, from the repo directory,
  because the run's paths are relative.
- **Baseline.** H, the committed tree 992f11a, reproduces midoffice03's finished run F bit for
  bit, report included. H with `--beta-days 252 --beta-half-life 63` reproduces midoffice03's
  V5u bit for bit, report included.

The variants:

| variant | what it is | mean R-squared | periods | first period | universe |
|---|---|---|---|---|---|
| H | HEAD 992f11a as committed: 828 days equal-weighted, mixed blends, the raw cube, the sd | 0.294211 | 938 | 2023-02-08 | 482 |
| H_u | H with `--beta-days 252 --beta-half-life 63` (= midoffice03's V5u) | 0.299406 | 962 | 2023-01-04 | 483 |
| V1 | item 1: USE4's market model by default | 0.299406 | 962 | 2023-01-04 | 483 |
| V1_828 | V1 with `--beta-window 36 --beta-half-life inf` | 0.294211 | 938 | 2023-02-08 | 482 |
| V2 | item 2: one blend per factor (938 periods) | 0.299429 | 938 | 2023-02-08 | 482 |
| V2_mixed | V2 with `--mixed-blends` (962 periods) | 0.299406 | 962 | 2023-01-04 | 483 |
| V2_828 | V2 with `--beta-window 36 --beta-half-life inf`: the rule under the old window | 0.301109 | 386 | 2025-04-23 | 449 |
| V3 | item 3: Size Nonlinearity USE4's | 0.298076 | 938 | 2023-02-08 | 482 |
| V3_use3 | V3 with `--size-nonlinearity use3` | 0.299429 | 938 | 2023-02-08 | 482 |
| V3_nosnl | V3 with Size Nonlinearity left out of the model (scratch wrapper) | 0.293802 | 938 | 2023-02-08 | 482 |
| V4 | item 4: the spread by the MAD (938 periods) | 0.298076 | 938 | 2023-02-08 | 482 |
| V4_sd | V4 with `--residual-spread sd` | 0.298076 | 938 | 2023-02-08 | 482 |
| V4_962 | V4 with `--mixed-blends` (962 periods) | 0.298059 | 962 | 2023-01-04 | 483 |
| V4_962_sd | V4 with `--mixed-blends --residual-spread sd` | 0.298059 | 962 | 2023-01-04 | 483 |
| V5 | item 5: RSTR's docstring (the defaults throughout) | 0.298076 | 938 | 2023-02-08 | 482 |
| F | the finished working tree, the defaults (must be V5) | 0.298076 | 938 | 2023-02-08 | 482 |
| F_nomem | the finished tree with `--industries` alone, no membership | 0.290175 | 938 | 2023-02-08 | 490 |
| F_eq | the finished tree with `--equal-weight` | 0.254680 | 938 | 2023-02-08 | 482 |
| F_noind | the finished tree with `--membership` alone, no industries | 0.165043 | 938 | 2023-02-08 | 482 |

## Results

### Each change against the one before, at the defaults

| | H | V1 | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|---|
| mean R-squared | 0.294211 | 0.299406 | 0.299429 | 0.298076 | 0.298076 | 0.298076 |
| estimated periods | 938 | 962 | 938 | 938 | 938 | 938 |
| first period | 2023-02-08 | 2023-01-04 | 2023-02-08 | 2023-02-08 | 2023-02-08 | 2023-02-08 |
| regression universe | 482 | 483 | 482 | 482 | 482 | 482 |
| forecasts | 468 | 468 | 468 | 468 | 468 | 468 |
| Volatility: vol | 7.63% | 8.63% | 8.63% | 8.77% | 8.77% | 8.77% |
| Volatility: mean (t) | +4.52% (+1.37) | +4.72% (+1.30) | +3.47% (+0.94) | +3.39% (+0.92) | +3.39% (+0.92) | +3.39% (+0.92) |
| Volatility: noise share | 5.3% | 3.9% | 3.9% | 3.8% | 3.8% | 3.8% |
| Momentum: vol | 7.16% | 6.57% | 6.57% | 6.62% | 6.62% | 6.62% |
| Momentum: mean (t) | +2.44% (+0.81) | +2.74% (+0.94) | +3.64% (+1.22) | +3.60% (+1.21) | +3.60% (+1.21) | +3.60% (+1.21) |
| Momentum: noise share | 4.0% | 4.4% | 4.4% | 4.4% | 4.4% | 4.4% |
| Size: vol | 7.57% | 6.90% | 6.90% | 4.03% | 4.03% | 4.03% |
| Size: mean (t) | +4.92% (+1.54) | +4.90% (+1.65) | +4.69% (+1.56) | +2.86% (+1.69) | +2.86% (+1.69) | +2.86% (+1.69) |
| Size: noise share | 9.9% | 11.6% | 11.6% | 7.1% | 7.1% | 7.1% |
| Size Nonlinearity: vol | 6.67% | 6.38% | 6.38% | 2.54% | 2.54% | 2.54% |
| Size Nonlinearity: mean (t) | -1.44% (-0.54) | -2.57% (-1.01) | -2.24% (-0.86) | +0.40% (+0.30) | +0.40% (+0.30) | +0.40% (+0.30) |
| Size Nonlinearity: noise share | 17.1% | 18.6% | 18.6% | 41.8% | 41.8% | 41.8% |
| Trading Activity: vol | 4.99% | 4.98% | 4.98% | 5.07% | 5.07% | 5.07% |
| Trading Activity: mean (t) | +3.90% (+2.14) | +4.59% (+2.86) | +4.11% (+2.53) | +4.40% (+2.67) | +4.40% (+2.67) | +4.40% (+2.67) |
| Trading Activity: noise share | 11.5% | 10.0% | 10.0% | 10.1% | 10.1% | 10.1% |
| residual spread used | 0.3682 | 0.2793 | 0.3632 | 0.3612 | 0.1333 | 0.1333 |
| its sd / 1.4826 x MAD | 0.3682 / -- | 0.2793 / -- | 0.3632 / -- | 0.3612 / -- | 0.3612 / 0.1333 | 0.3612 / 0.1333 |
| FRC's share of the sd's square | 71.3% | 57.9% | 72.0% | 72.4% | 72.4% | 72.4% |
| specific risk p10 / median / p90 | 15.06% / 23.53% / 36.35% | 15.18% / 23.43% / 36.06% | 15.10% / 23.46% / 36.35% | 15.16% / 23.40% / 36.11% | 15.40% / 23.77% / 36.50% | 15.40% / 23.77% / 36.50% |
| fallback forecasts | 8 | 7 | 8 | 8 | 8 | 8 |

### Exactness

| comparison | factor returns | style exposures | forecast exposures | specific risks | report |
|---|---|---|---|---|---|
| H (992f11a) vs midoffice03's F | same | same | same | same | identical |
| H with USE4's beta vs midoffice03's V5u | same | same | same | same | identical |
| V1 vs H with USE4's beta | same | same | same | same | the market-model line |
| V1 at `--beta-window 36 --beta-half-life inf` vs H | same | same | same | same | the market-model line |
| V2 with `--mixed-blends` vs V1 | same | same | same | same | the guard's lines |
| V2 vs midoffice03's V5u938 (blanked to 938 periods) | same | differ before 2023-02-07, by construction | same | same | as expected |
| V2's 938 periods vs V1's same dates | same | -- | -- | -- | -- |
| V3 with `--size-nonlinearity use3` vs V2 | same | same | same | same | one line added |
| V4 with `--residual-spread sd` vs V3 | same | same | same | same | the V-fit lines |
| V5 vs V4 | same | same | same | same | identical |
| F, the finished tree, vs V5 | same | same | same | same | the clip line's wording |

"Same" is bit for bit throughout. F differs from V5 in one report line: the no-membership run
showed that Size Nonlinearity's clip line said "member values" with no members, so it now
names the universe ("values of the index members").

midoffice03's V5u938 blanked every descriptor before 2023-02-07 to hold the estimation to the
default's periods. Its style exposures therefore differ where it blanked them, and its screen
with them. Its model is V2's.

## Item 1: the market model is USE4's by default

**What changed.** `compute_descriptors(beta_window_months=None, beta_half_life=None)` now
fits USE4's market model on a daily panel: 252 trading days, each day's weight halving every
63 back from the window's end. On a panel of any other frequency it keeps USE3's 60 months,
weighted equally.

- `market_model_window(panel, beta_window_months, beta_half_life)` resolves both and returns
  the window, capped at the panel's length, and the half-life. None means equal weights.
- `compute_descriptors` fits what it returns, and the run reads the window from it for its
  standing line and for the window-coverage check. So nothing in the run restates the
  library's default.
- `beta_half_life=None` now means the panel's default. `math.inf` weights every day equally,
  by the ordinary least squares path, bit for bit the unweighted fit.

The run's options:

- `--beta-window`, `--beta-days` and `--beta-half-life` default to None, which leaves the
  library's default.
- `--beta-half-life inf` asks for equal weights.
- The standing line names the setting: "(the library's default, USE4's)", or the flags given.

**Decomposed.** V1, the new default, is H run with `--beta-days 252 --beta-half-life 63` bit for
bit. V1 at `--beta-window 36 --beta-half-life inf` is H bit for bit. In each pair the two
reports differ by the market-model line alone. Against the committed default:

- **The sample.** BTSG and HALPHA have values from 2023-01-03, and VLVR from 2023-02-07,
  because its twelve-month dollar volume holds it back. HALPHA alone gives Momentum a value 24
  days before RSTR's window fills, so the estimation starts on 2023-01-04: 962 periods, not 938.
- **The model.** Mean R-squared rises from 0.294211 to 0.299406. The style volatilities move:

  | style | H | V1 |
  |---|---|---|
  | Volatility | 7.63% | 8.63% |
  | Momentum | 7.16% | 6.57% |
  | Size | 7.57% | 6.90% |
  | Size Nonlinearity | 6.67% | 6.38% |
  | Trading Activity | 4.99% | 4.98% |

- **The screen.** BTSG still ranks first, at 8.36% against 8.43%, now over 962 periods against
  386. HALPHA rises to fourth, 4.43% against 3.38%. VLVR's t-statistic goes from +0.58 to
  +1.95.
- **The specific risk.** The residual spread's standard deviation falls from 0.3682 to 0.2793:
  FRC is 57.9% of its square on 962 periods, against 71.3%. Seven securities are fallbacks,
  not eight, because VNO is regressed in the early periods.

## Item 2: one definition per factor across the sample

**The rule.** In the run, before the industries, `regression_design` gives each period's
regression. For each risk index, `blend_composition` reads which descriptors the blend drew on
over the securities each regression holds, and compares them with the forecast's: the
descriptors the forecast exposures are built from, over the securities with a complete row on
the last date. A period whose blend differs in any index is not estimated.

**The library.** `estimate_cross_sectional_model` and `regression_universe` take the rule as
`periods`, one boolean per return-panel date.

- A period marked False is not estimated, however many securities it holds.
- It is named in the notes' new `excluded_periods` and in a qualification line. A thin period
  stays thin: the thin test comes first.
- The industry universe is counted over the same periods.

**The assertion.** After estimation the run reads every estimated period's blend again, by the
estimator's own test of who each regression held. If any period's blend differs from the
forecast's in any index, the run stops.

- That replaces midoffice03's check of the last estimated period alone, which only flagged.
- midoffice03's stop for a descriptor in the forecast exposures and in no estimated period's
  is kept.
- `--mixed-blends` estimates every period as before. Then only that stop applies, and the
  periods blended otherwise are flagged with "!".

**What it drops**, as the run reports it:

```text
  every estimated period's blend is the forecast exposures', 2026-11-03's, in every index: one definition per factor across the sample
  24 period(s) held enough securities and are not estimated, a risk index's blend in each not being the forecast's:
    2023-01-04 to 2023-02-07 (24): Volatility from BTSG, DASTD, HILO, LPRI, without CMRA; Momentum from HALPHA, without RSTR; Trading Activity from STOM, STOQ, without STOA, VLVR
```

**Decomposed against V1:**

- **The periods.** Every one of V2's 938 periods has V1's factor returns for that date, bit for
  bit. No period's regression reads another's, and the industries are unchanged: still 19,
  over 482 securities. The model is midoffice03's V5u938, which held the estimation to those
  periods by blanking the descriptors.
- **The universe.** It loses VNO, regressed in the 24 dropped periods alone, so VNO's
  forecast becomes a fallback again: 8 fallbacks.
- **What moves:**
  - the covariance, slightly: the 24 periods carried little weight under the 90-day
    half-life;
  - the plain mean returns, which average over the periods: Volatility +4.72% to +3.47%,
    Momentum +2.74% to +3.64%;
  - the residual spread's standard deviation, 0.2793 to 0.3632. FRC's share of its square
    rises from 57.9% to 72.0% as the sample shortens. Item 4 is about this.
- Mean R-squared is 0.299429.

**Under the old window.** At `--beta-window 36 --beta-half-life inf` the rule drops 552
periods, 2023-02-08 to 2025-04-22, in which Volatility has no BTSG, Momentum no HALPHA and
Trading Activity no VLVR.

- It estimates 386 periods.
- The universe shrinks to 449, and the model to 23 factors, SIC37 no longer among them.
- 30 securities become fallbacks.

That is why item 1 comes first.

## Item 3: Size Nonlinearity, USE4's way

**What changed.** `build_risk_indices(..., size_nonlinearity="use4", regression_weights=...)`
builds Size Nonlinearity in `nonlinear_size`, date by date, over the estimation universe:

1. the Size exposure cubed;
2. standardised over the members (cap-weighted mean, equal-weighted sd), without the raw MAD
   clip;
3. clipped at +/-4;
4. orthogonalised against Size by a least-squares fit on Size and a constant, weighted by the
   regression weights, over the members with a Size exposure and a positive weight;
5. standardised again, exactly.

Everyone else with a Size exposure is put on the members' scale, as in every standardisation
here.

- **The other functions.** `index_descriptors` and `blend_composition` take the same switch, so
  under "use4" Size Nonlinearity's blend reads as built from LNCAP.
- **The run.**
  - It computes the regression weights before the risk indices.
  - LCAPCB is put in no blend under "use4"; it is still computed and screened.
  - `--size-nonlinearity use3` restores the raw cube.
- **The guard.** On every date an estimated period or the forecast reads, the run computes the
  regression-weighted covariance of the two exposures over the members, from the exposures
  themselves. It stops if any date's covariance is past its first-order rounding bound,
  n * eps * the weighted mean of |S - S_w| |N - N_w|.

On the defaults:

```text
  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,391 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.0e-15 (4.7 eps), inside rounding on 939 of the 939 dates
```

**Decomposed against V2.** `--size-nonlinearity use3` is V2 bit for bit. Its report adds one
line, the USE3 cube's correlation with Size: +0.933 at the median date, +0.931 on 2026-11-03.
The report the brief asked for, against the raw cube, uses midoffice01's definitions, so the
figures compare with its 0.93 and -0.87:

- the exposures' correlation, equal-weighted over the members with both;
- the factor returns' correlation, from the model's covariance;
- the pair's risk, sqrt(x' S x) over the Size block, for a portfolio's average exposures on the
  last date. The portfolios are the smallest 85 members by cap and the largest 50, equally
  weighted, and the members cap-weighted.

| | USE3's raw cube (V3_use3 = V2) | USE4's (V3) | no Size Nonlinearity (V3_nosnl) |
|---|---|---|---|
| exposures' correlation, equal-weighted, median of the 939 dates | +0.933 | +0.551 |  |
| the same, on 2026-11-03 | +0.931 | +0.585 |  |
| exposures' correlation by the regression weights, 2026-11-03 | +0.911 | +0.000 |  |
| factor returns' correlation, in the covariance | -0.824 | -0.308 |  |
| factor returns' correlation, the plain series | -0.851 | -0.322 |  |
| Size: vol | 6.90% | 4.03% | 4.03% |
| Size: mean a year (t) | +4.69% (+1.56) | +2.86% (+1.69) | +2.89% (+1.71) |
| Size: noise share | 11.6% | 7.1% | 7.1% |
| Size Nonlinearity: vol | 6.38% | 2.54% |  |
| Size Nonlinearity: mean a year (t) | -2.24% (-0.86) | +0.40% (+0.30) |  |
| Size Nonlinearity: noise share | 18.6% | 41.8% |  |
| Size against the model without Size Nonlinearity: correlation | 0.7935 | 0.9998 |  |
| ... the difference's vol a year | 3.79% | 0.06% |  |
| the pair's risk, smallest 85 members, EW (exposures) | 12.16% (-3.110, -2.814) | 11.94% (-3.110, -1.647) | 12.52% (-3.110) |
| the pair's risk, largest 50 members, EW (exposures) | 3.45% (+0.012, +0.551) | 0.38% (+0.012, +0.155) | 0.05% (+0.012) |
| the pair's risk, the index, cap-weighted (exposures) | 0.01% (+0.002, +0.001) | 0.01% (+0.002, +0.000) | 0.01% (+0.002) |
| other styles' vols: Volatility / Momentum / Trading Activity | 8.63% / 6.57% / 4.98% | 8.77% / 6.62% / 5.07% | 8.70% / 6.64% / 4.86% |
| mean R-squared | 0.299429 | 0.298076 | 0.293802 |
| member values the clip moves, over the 939 dates | 85,319 of 422,881 (20.2%) | 4,391 (1.04%) |  |
| ... a date, median; on 2026-11-03 | 91; 81 of 425 | 4; 4 of 425 |  |

What this shows:

- **Size no longer depends on Size Nonlinearity.** Under USE4, Size's factor return is
  0.9998 correlated with the model without Size Nonlinearity, and the difference has a
  volatility of 0.06% a year. Under USE3 the correlation is 0.7935 and the difference 3.79% a
  year. With the raw cube, Size's plain volatility is 1.785 times what it is without it.
- **The largest 50 members.** They hold no size bet, yet the raw cube gave them 3.45% of
  size-pair risk; under USE4 they carry 0.38%.
- **The clip.** LCAPCB's raw clip moved 85,319 of 422,881 member values, 20.2%: its bounds on
  2026-11-03 were [-1.783, +1.781], the cubes of about 1.21. USE4's clip moves 4,391, 1.04%:
  median 4 a date, at most 10.
- **Inside the regressions.** The orthogonality is set over the members with a Size exposure
  on the exposure date. A regression holds only those with a return the next day and a
  complete row. Over each regression, by its own weights, the correlation of the two exposures
  has median 0.0000 and ranges from -0.0106 to +0.0008. The regression leaves out at most 2
  such members, and in the median period none.
- **Equal-weighted.** The two are still +0.551 correlated at the median date. The
  orthogonality is in the regression weights, and those are what the regression uses.
- **What is left.** Size Nonlinearity's factor return has a volatility of 2.54%, and 41.8% of
  that variance is the sampling error of its own estimation, against 18.6% for the raw cube.
  Its factor returns are still -0.308 correlated with Size's.
- **Mean R-squared** falls from 0.299429 to 0.298076. Without Size Nonlinearity it is 0.293802.

## Item 4: the residual spread by the MAD

**What changed.** `estimate_specific_risk_model(..., residual_spread="mad")` measures
`relative_residual_sd`, the root of v_model in every blend weight, as 1.4826 times the median
absolute deviation of the securities' average relative residuals from their median.

- It falls back to the standard deviation where the MAD is zero.
- "sd" keeps the standard deviation.
- The notes carry both measures (`residual_spread_mad`, `residual_spread_sd`) and which was
  used. `residual_spread_shares` stays the standard deviation's shares, the names that would
  set it.
- `estimate_cross_sectional_model` passes the switch through and refuses an unknown value.
- The run has `--residual-spread`. Its V-fit lines give the spread used, the other measure, the
  sd's largest shares, and a new standing line: the characteristics' weight in a blended
  forecast at the median history.

**Decomposed against V3.** `--residual-spread sd` is V3 bit for bit, with the V-fit lines
reworded and the model-weight line added. The 962-period sample is `--mixed-blends`, item 2's
switch:

| | 938 periods, MAD (V4) | 938, sd (V4_sd) | 962, MAD (V4_962) | 962, sd (V4_962_sd) |
|---|---|---|---|---|
| spread used | 0.1333 | 0.3612 | 0.1334 | 0.2805 |
| 1.4826 x MAD / sd | 0.1333 / 0.3612 | 0.1333 / 0.3612 | 0.1334 / 0.2805 | 0.1334 / 0.2805 |
| FRC's share of the sd's square | 72.4% | 72.4% | 57.2% | 57.2% |
| model weight at the median history | 2.92% (n 938) | 0.41% (n 938) | 2.84% (n 962) | 0.66% (n 962) |
| model weight across securities: p90 / max | 3.85% / 58.46% | 0.54% / 16.08% | 3.81% / 96.56% | 0.89% / 86.40% |
| specific risk p10 / median / p90 | 15.40% / 23.77% / 36.50% | 15.16% / 23.40% / 36.11% | 15.44% / 23.66% / 36.27% | 15.20% / 23.35% / 36.00% |

| delivered forecasts | rms change | mean change | largest | past 1 point | largest moves |
|---|---|---|---|---|---|
| 938 periods, sd -> MAD | 0.677 pt | +0.299 pt | 10.020 pt | 15 | SBNY 69.50% -> 59.48%, LUMN 67.93% -> 63.93%, ALK 29.05% -> 32.20% |
| 962 periods, sd -> MAD | 0.436 pt | +0.271 pt | 3.428 pt | 12 | VNO 45.36% -> 41.93%, ALK 28.54% -> 31.13%, SBNY 57.50% -> 54.94% |
| MAD, 938 -> 962 periods | 0.455 pt | -0.068 pt | 5.798 pt | 11 | LUMN 63.93% -> 58.13%, SBNY 59.48% -> 54.94%, AAP 54.10% -> 51.29% |
| sd, 938 -> 962 periods | 0.808 pt | -0.040 pt | 12.003 pt | 12 | SBNY 69.50% -> 57.50%, LUMN 67.93% -> 57.90%, VNO 40.96% -> 45.36% |

**The spread barely moves under the MAD.** It is 0.1333 on 938 periods and 0.1334 on 962, as
the brief expected. The standard deviation moves from 0.3612 to 0.2805, 22% lower, because
FRC's share of its square falls from 72.4% to 57.2% when the sample starts 24 days earlier.

- **The blend weights.** At the median history, 938 periods, the characteristics' weight in a
  blended forecast rises from 0.41% to 2.92%. It is highest for the shortest history: 58.46%
  for SIVB, with 21 periods on the 938-period sample.
- **The delivered forecasts.** On 938 periods they move by 0.677 points rms. The mean move is
  +0.299 points, and the median goes from 23.40% to 23.77%. 15 forecasts move by more than a
  point; the largest is SBNY, 69.50% to 59.48%.
- **Between the two samples,** the delivered forecasts move 0.455 points rms under the MAD and
  0.808 under the standard deviation.
- No forecast is delivered the floor in any of the four, and no characteristic forecast is
  below it.

## Item 5: RSTR's skip stays 0

Only text changes.

- **`rstr_skip`'s docstring** now says the parameter cuts the window's last k days, the
  academic twelve-minus-one reading. At 21 nothing material moved and R-squared fell by
  0.0006. USE4's lag is a different reading, a full window ending k days back, measured in
  midoffice03 and not built.
- **Three places said the cut was USE4's:** midoffice03's `--rstr-skip` help ("USE4 skips
  21"), its test's docstring, and `docs/FACTOR_MODEL.md`. All three now say what the docstring
  says.

V5 is V4 bit for bit, report included. tree_5 differs from tree_4 in those two texts alone.

## Key decisions

1. **The market model's default depends on the frequency.**
   - On a daily panel it is USE4's 252/63; on any other it stays USE3's 60 months, equally
     weighted. USE4's beta is a daily estimator, and twelve monthly returns are too few for
     one.
   - This follows the volatility window's precedent: 65 trading days on a daily panel, twelve
     months otherwise.
   - It keeps every monthly-panel test's market model as it was.
2. **`beta_half_life=None` now means the panel's default, and `math.inf` equal weights.**
   - midoffice03 had None as equal weights. Keeping None both as "the default" and as "equal
     weights" would have needed a sentinel.
   - An infinite half-life is equal weights in the limit, and it takes the ordinary least
     squares path, so the 828 equally weighted days reproduce the old run bit for bit.
   - The run spells it `--beta-half-life inf`.
3. **One resolver for the market model, `market_model_window`.** The run reads the window from
   it instead of recomputing it from its own arguments, so no default lives in two places.
4. **Item 2 is a library parameter, with the rule in the run.**
   - `periods` on the estimator and `regression_universe` is generic. The composition rule is
     the run's, computed with `blend_composition` before the industries, because the industry
     universe must be the estimated periods'.
   - The assertion afterwards uses the estimator's own `in_regression` and the forecast
     exposures it delivered, so it checks the rule rather than sharing its code.
5. **Thin before excluded.** A period too thin to estimate is counted thin even where the rule
   would also rule it out, so `excluded_periods` holds only the periods the rule actually cost.
6. **`--mixed-blends` keeps item 2 switchable,** as the brief's policy asks of items 1 to 3.
   Under it the 962-period sample item 4 asks for is a plain run.
7. **The orthogonalisation includes a constant,** that is, it centres both exposures by their
   regression-weighted means. Without the constant the weighted covariance would not be zero:
   Size's weighted mean is not zero, only its cap-weighted one is.
8. **The orthogonalisation's universe.** It is the members with a Size exposure and a positive
   weight on the exposure date, the universe every standardisation here uses. It is not each
   regression's own subset, which is not known when the exposures are built: who has a return
   the next day.
9. **The size-pair guard stops the run,** as midoffice03's guard does, since a Size
   Nonlinearity that is not orthogonal is not the factor named. The equal-weighted
   correlation is reported, not guarded.
10. **The clip count is worked out in the run from the Size exposure,** like the other standing
    checks, not taken from the code that clipped.
11. **The MAD falls back to the standard deviation where it is zero,** as the winsorising
    scale does. The standard deviation's shares stay in the notes as the diagnostic.
12. **Tests calibrated on the standard deviation now say so.** Two blend tests pin
    `residual_spread="sd"`: the blend's ordering at four and sixty periods, and the
    which-floor test. On those fixtures the per-security averages are heteroskedastic, so the
    MAD is smaller and the model keeps more weight. One of the two also asserts the MAD's
    larger weight. The monotone-weight test gains n = 10,000, where the weight is below 0.1
    under either measure.

## Decisions left for Jeff

1. **USE4's Size Nonlinearity is small and noisy:** a volatility of 2.54%, 41.8% of its
   variance sampling error, mean +0.40% (t +0.30). USE4 keeps the factor. Keep it, or drop it
   from the model? R-squared without it is 0.293802 against 0.298076.
2. **The equal-weighted correlation of Size and Size Nonlinearity is still +0.551.** The brief's
   orthogonality is by the regression weights, and that is what is built and guarded. Is that
   the intended sense?
3. **The frequency-dependent market-model default:** USE4's on a daily panel, USE3's 60 months
   on any other. And `beta_half_life=None` now means the default, with `math.inf` as equal
   weights.
4. **The MAD raises the characteristics' weight at long histories** from 0.41% to 2.92%, and
   moves SBNY's forecast from 69.50% to 59.48%. Is that the intended trade-off, or should the
   weight stay closer to the history at 938 periods?
5. **Size's mean for the size pair.** midoffice03 left this open. It is now +2.86% a year
   (t +1.69) under USE4's Size Nonlinearity, against +4.69% with the raw cube on the same 938
   periods.

## Tests

The full suite, `pytest tests --ignore=tests/test_market_data.py`, gives 2,231 passed, 1 skipped,
19 warnings, all of them in files not touched here. The midoffice tests give 780 passed, 1 skipped.
Six tests are new:

- **`test_on_a_daily_panel_the_market_model_defaults_to_use4s_and_any_other_stays_runnable`.**
  `market_model_window` resolves:
  - a daily panel to (252, 63.0) and a monthly one to (60, None);
  - a window in months, capped at the panel's length;
  - an infinite half-life to equal weights.

  The daily default is the explicit 252/63 bit for bit, and has its first value on row 251.
  Ten months at an infinite half-life is the ordinary least squares fit, computed
  independently.
- **`test_a_period_the_caller_rules_out_is_not_estimated_and_is_named_apart_from_the_thin`.**
  - A ruled-out period is skipped and named in `excluded_periods`; a thin one stays thin.
  - Every other period's factor returns are the unruled estimation's, bit for bit.
  - A security regressed only in ruled-out periods leaves both `regression_universe` and the
    estimator's universe.
  - A mask of the wrong length or dtype is refused by both.
- **`test_size_nonlinearity_is_built_use4s_way_by_default_and_is_orthogonal_to_size`.** It
  matches an independent computation of the five steps, with leavers put on the members'
  scale. On every date the members' regression-weighted covariance is inside rounding, the
  cap-weighted mean is zero and the equal-weighted sd is one.
- **`test_nonlinear_sizes_clip_is_four_sds_of_the_cube_where_four_mads_would_bind_at_1_2`.**
  - On 20,000 normal draws the four-MAD bound on the cube is the cube of 1.22, and it would
    move over a fifth of the values.
  - `nonlinear_size`'s clip moves exactly the values whose standardised cube is past 4, and
    under 2% of them.
- **`test_size_nonlinearity_built_use3s_way_is_the_lcapcb_blend_as_before`.**
  - "use3" is the old LCAPCB blend, bit for bit, and every other index is identical under both.
  - "use4" builds without LCAPCB, and "use3" cannot.
  - `index_descriptors` and `blend_composition` read Size Nonlinearity by its construction.
  - An unknown value is refused.
- **`test_the_residual_spread_is_the_mad_by_default_which_no_one_security_sets`.**
  - One security, with its characteristic near zero, is made six times riskier.
  - The standard deviation more than triples; the MAD moves under 5%.
  - The MAD matches an independent computation, the notes carry both, and "sd" gives the
    standard deviation and a smaller model weight.
  - An unknown value is refused.

Changed:

- **The half-life test.** Renamed `..._and_on_a_monthly_panel_is_not`; equal weights are now
  `math.inf`, and the refused values swap inf for -inf.
- **The skip test's docstring** (item 5).
- **The spread-shares test** pins "sd" and checks the notes' sd.
- **The monotone-weight test** adds n = 10,000.
- **The blend's ordering test** pins "sd" and asserts the MAD keeps more weight.
- **The which-floor test** pins "sd".
- **The unknown-method test** also refuses an unknown `residual_spread`.

## Failures and debugging trail

- **The exactness check reported `spread_mad` as different between equal variants.**
  - Both sides were NaN: before item 4 the notes had no MAD. JSON reads that back as nan, and
    nan != nan, so `exact.py` listed the field.
  - Read for what it is, not fixed: the arrays and the report decide equality.
- **Under the MAD default, four tests failed.**
  - **My own new test.** It had made security 5 six times riskier. Its characteristic is
    +1.21, so it pulled gamma (0.301 to 0.330) and the MAD with it, +30%.
  - The fix was to make the riskier security the one whose characteristic is nearest zero.
    That is FRC's pattern on the real run, 2% of V^2 and most of the standard deviation's
    square. The MAD then moves -2% while the standard deviation multiplies by five.
  - **Three older tests,** calibrated on the standard deviation, failed on thresholds. On
    their heteroskedastic fixtures the MAD is lower, for example 0.059 against the sd's
    0.067 or more. The model weight at a given history is therefore higher: 0.124 at n = 1000
    against a bound of 0.1; 0.555 at sixty periods against 0.5; a long-history blend 1.6x the
    floor where the test wanted 3x. They are handled as decision 12 says.
- **Lint.**
  - **A `/tmp` file.** One HEAD-versus-tree lint comparison wrote to `/tmp`; it was deleted at
    once, and the comparison was redone from the repo by piping HEAD's text to ruff.
  - **Isort false positives.** Running ruff on copies outside the repo misreported isort
    findings, because first-party detection depends on the config's location.
  - **A long line.** A 102-character docstring line in `specificrisk.py` was the only new
    finding, and it was rewrapped.
- **A misworded line on another path.** The finished tree's first run (`final_check_first.txt`)
  passed, but on the no-membership path Size Nonlinearity's clip line said "member values" where
  there are no members. It now names the universe, as the covariance clause did, and the four
  finished-tree runs were repeated.
  - `final_ast.py` shows what changed after tree_5: docstrings in `descriptors.py` (a
    misphrased "four of 1.4826 times") and `specificrisk.py` (the rewrap and its figures), and
    that one printed line in the run script, whose code therefore no longer matches tree_5's.
  - F against V5: every array the same, the report differing in that line alone.
- **A variable name.** `periods` was already used inside the composition loop
  (`np.flatnonzero`) and would have shadowed the new mask. That variable is now `where`, and
  the rule's rows `period_rows`.
- **Two bugs caught before running.** In `nonlinear_size`, `inside = present` aliased the mask
  that `inside &= ...` then narrowed in place. It is now built fresh. A wrong-shape
  `regression_weights` is refused.

## Problems found, not fixed

- **Size Nonlinearity's noise share is 41.8%** under USE4: see decision 1.
- **FRC is still 72.4% of the residual standard deviation's square.** The MAD stops it setting
  the blend weights, but FRC itself is still in the sample, a collapsing bank that stayed in
  the index.
- **Still open from midoffice03:**
  - split-stale caps, 621 member-days, with the rule for the rebuild recorded in the docs;
  - `fit_index_weights` is whole-sample;
  - VLVR runs the market model a second time, now weighted.
- **Construction's noise share is about 50%,** the loudest factor; Wholesale's is about 42%.
- **Two stale figures in `docs/FACTOR_MODEL.md`:** 3,043 member values moved by the second
  clip, and 100,011 outsider values at a bound. Both predate midoffice02 and are not refreshed
  here; the run's report gives the current ones.

## Scratch

`claude/scratch/midoffice04/` (gitignored) holds every script and output below, without the
pickles:

- `common.py`, `runv.py` and `exact.py` (from midoffice03, extended);
- `runv3.py`, `steps.py`, `size_pair.py`, `spread_effect.py`, `in_regression.py`,
  `final_check.py` and `final_ast.py`;
- `extract_brief.py` and `write_brief.py`;
- `build_retro.py` with `retro_head.md`;
- `tree_1.patch` to `tree_5.patch`, each the cumulative diff from HEAD.

The session scratchpad under Temp also holds the snapshots and pickles, and can vanish.

## Git

No commits; Code does not commit (Rule 43). Two `git add` lists are not needed: everything here
is one change.

## Open items for Chat

- The five decisions above, the first two most of all.
- Whether the size pair's portfolios -- the smallest 85, the largest 50, the index -- should
  become a standing line in the run, since they are how a reader sees the pair's risk.
- With beta in every period and the pair orthogonal, the next loudest things in the model are
  Construction's and Wholesale's noise shares and the identity rebuild: splices, renames, the
  missing names, and the phantoms.

## Appendix: scratch outputs, verbatim

### runs_tree0.out -- the baseline: HEAD as committed, and with USE4's beta

```text
H      lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [54s; --industries --membership]
H_u    lag 1  R2 0.299406  periods 962  forecasts 468  Volati 8.6255%/+4.72%  Moment 6.5735%/+2.74%  Size 6.8966%/+4.90%  Size N 6.3820%/-2.57%  Tradin 4.9838%/+4.59%  [69s; --industries --membership --beta-days 252 --beta-half-life 63]
```

### exact_steps.txt -- every bit-for-bit comparison, in the order of the steps

```text
H vs F: factor returns False, style exposures False, forecast exposures False, specific risks False; summary fields differing: ['r2', 'vol', 'mean', 'tstat', 'noise', 'spread', 'spread_sd', 'spread_mad', 'spread_shares', 'v2_shares', 'gamma', 'specific', 'screen', 'compute_kw', 'blended_names']; report lines differing: 105
   -  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 828 trading days (36 months), every day weighted equally
   -  window descriptors need at least 50% of the window valid; from a window with none, over 3,581,487 values: none; from one under 50% valid: none
   -  344,062 raw values winsorised before standardising
   -  95,646 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
   -    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/46  HILO 8/45  CMRA 5/46  BTSG 6/59
   -    HALPHA 6/59  SERDP 1/59  RSTR 3/49  STOM 13/45  STOQ 14/46  STOA 16/49  VLVR 17/49
   +  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days (the library's default, USE4's)
   +  window descriptors need at least 50% of the window valid; from a window with none, over 4,402,359 values: none; from one under 50% valid: none
   +  369,070 raw values winsorised before standardising
   +  102,179 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
   +    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/46  HILO 8/45  CMRA 5/46  BTSG 7/49
   +    HALPHA 4/49  SERDP 0/49  RSTR 3/49  STOM 13/45  STOQ 14/46  STOA 16/49  VLVR 16/49
   -  BTSG         95%      386    8.43%      9.77%     1.87
   +  BTSG         93%      962    8.36%      9.32%     1.34
   +  HALPHA       93%      962    4.43%      6.30%     0.56
   -  HALPHA       95%      386    3.38%      6.16%    -0.29
   +  VLVR         93%      938    1.78%      3.76%     1.95
   -  VLVR         93%      386    1.52%      3.94%     0.58
   -  Size Nonlinearity    LCAPCB
   +  Size Nonlinearity    the Size exposure cubed, standardised and clipped at +/-4, orthogonalised against Size with the regression weights, standardised again (USE4)
   +  - LCAPCB is in no blend: USE3's raw cube; Size Nonlinearity is built from the Size exposure instead, USE4's way
   -  mean cross-sectional R-squared 0.294
   +  mean cross-sectional R-squared 0.298
   -    Volatility           DASTD, HILO, LPRI, CMRA from 2023-02-08 (938); BTSG from 2025-04-23 (386)
   -    Momentum             RSTR from 2023-02-08 (938); HALPHA from 2025-04-23 (386)
   +    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2023-02-08 (938)
   +    Momentum             RSTR, HALPHA from 2023-02-08 (938)
   -    Size Nonlinearity    LCAPCB from 2023-02-08 (938)
   -    Trading Activity     STOM, STOQ, STOA from 2023-02-08 (938); VLVR from 2025-04-23 (386)
   -  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
   +    Size Nonlinearity    LNCAP from 2023-02-08 (938)
   +    Trading Activity     STOM, STOQ, STOA, VLVR from 2023-02-08 (938)
   +  every estimated period's blend is the forecast exposures', 2026-11-03's, in every index: one definition per factor across the sample
   +  24 period(s) held enough securities and are not estimated, a risk index's blend in each not being the forecast's:
   +    2023-01-04 to 2023-02-07 (24): Volatility from BTSG, DASTD, HILO, LPRI, without CMRA; Momentum from HALPHA, without RSTR; Trading Activity from STOM, STOQ, without STOA, VLVR
   -  Volatility                  -3.9e-18   1.000000   426
   -  Momentum                    -1.5e-17   1.000000   423
   +  Volatility                  +1.2e-17   1.000000   426
   +  Momentum                    -5.4e-17   1.000000   423
   -  Size Nonlinearity           -7.7e-18   1.000000   425
H_u vs V5u: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 0
V1 vs H_u: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: ['spread_mad', 'compute_kw']; report lines differing: 2
   -  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days (the library's default, USE4's)
   +  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days
V1_828 vs H: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: ['spread_mad', 'compute_kw']; report lines differing: 2
   -  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 828 trading days, every day weighted equally (--beta-window 36, --beta-half-life inf)
   +  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 828 trading days (36 months), every day weighted equally
V2_mixed vs V1: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: ['spread_mad']; report lines differing: 4
   -  ! Volatility: 24 of the 962 estimated periods, from 2023-01-04 to 2023-02-07, are blended from other descriptors than the forecast exposures
   -  ! Momentum: 24 of the 962 estimated periods, from 2023-01-04 to 2023-02-07, are blended from other descriptors than the forecast exposures
   -  ! Trading Activity: 24 of the 962 estimated periods, from 2023-01-04 to 2023-02-07, are blended from other descriptors than the forecast exposures
   +  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
V2 vs V5u938: factor returns True, style exposures False, forecast exposures True, specific risks True; summary fields differing: ['screen', 'compute_kw']; report lines differing: 33
   -  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days (the library's default, USE4's)
   -  window descriptors need at least 50% of the window valid; from a window with none, over 4,402,359 values: none; from one under 50% valid: none
   -  369,070 raw values winsorised before standardising
   -  102,179 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
   +  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days
   +  window descriptors need at least 50% of the window valid; from a window with none, over 4,051,952 values: none; from one under 50% valid: none
   +  313,063 raw values winsorised before standardising
   +  99,518 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
   -  BTSG         93%      962    8.36%      9.32%     1.34
   -  DASTD        93%    1,149    6.13%      7.77%     1.36
   +  BTSG         93%      938    8.30%      9.32%     0.96
   -  HALPHA       93%      962    4.43%      6.30%     0.56
   -  HILO         93%    1,213    3.60%      6.25%     1.13
   +  DASTD        93%      938    5.31%      7.32%     1.28
   +  HALPHA       93%      938    4.48%      6.35%     0.78
   -  STOQ         93%    1,145    2.99%      5.21%     1.26
   -  STOM         93%    1,191    2.95%      5.25%     1.28
   +  HILO         93%      938    2.99%      5.85%     1.47
   -  LNCAP        93%    1,213    1.83%      3.95%     1.04
   +  STOQ         93%      938    2.56%      4.82%     1.11
   +  STOM         93%      938    2.46%      4.73%     1.12
   +  LNCAP        93%      938    1.96%      4.06%     1.53
   -  LPRI         93%    1,213    1.77%      3.77%    -0.41
   -  LCAPCB       93%    1,213    1.68%      3.78%     0.64
   +  LCAPCB       93%      938    1.76%      3.82%     1.21
   +  LPRI         93%      938    1.72%      3.66%     0.38
   -  every estimated period's blend is the forecast exposures', 2026-11-03's, in every index: one definition per factor across the sample
   -  24 period(s) held enough securities and are not estimated, a risk index's blend in each not being the forecast's:
   -    2023-01-04 to 2023-02-07 (24): Volatility from BTSG, DASTD, HILO, LPRI, without CMRA; Momentum from HALPHA, without RSTR; Trading Activity from STOM, STOQ, without STOA, VLVR
   +  the forecast exposures, 2026-11-03's, are blended from the descriptors the last estimated period's were, in every index
   -  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
   -  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
   +  - 275 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 272 more
V3_use3 vs V2: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: ['spread_mad']; report lines differing: 1
   -  Size Nonlinearity, USE3's raw cube, on the same dates: correlated with Size over the index members +0.933 at the median, +0.931 on 2026-11-03
V4_sd vs V3: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: ['spread_mad']; report lines differing: 4
   -                    relative residual spread 0.361, the sd of the securities' average relative residuals; 1.4826 x their MAD 0.133
   -                    largest shares of the sd's square: FRC 72.4%, SIVB 3.7%, LUMN 2.2%, BNY 1.5%, ENPH 1.0%
   -                    the characteristics' weight in a blended forecast, at the median history of 938 periods: 0.41%
   +                    relative residual sd 0.361, largest shares of its square: FRC 72.4%, SIVB 3.7%, LUMN 2.2%, BNY 1.5%, ENPH 1.0%
V5 vs V4: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 0
```

### runs_tree1.out -- item 1's tree: its default, and the old window

```text
V1     lag 1  R2 0.299406  periods 962  forecasts 468  Volati 8.6255%/+4.72%  Moment 6.5735%/+2.74%  Size 6.8966%/+4.90%  Size N 6.3820%/-2.57%  Tradin 4.9838%/+4.59%  [78s; --industries --membership]
V1_828 lag 1  R2 0.294211  periods 938  forecasts 468  Volati 7.6312%/+4.52%  Moment 7.1647%/+2.44%  Size 7.5659%/+4.92%  Size N 6.6705%/-1.44%  Tradin 4.9892%/+3.90%  [37s; --industries --membership --beta-window 36 --beta-half-life inf]
```

### runs_tree2.out -- item 2's tree: its default, --mixed-blends, and the old window

```text
V2     lag 1  R2 0.299429  periods 938  forecasts 468  Volati 8.6253%/+3.47%  Moment 6.5735%/+3.64%  Size 6.8968%/+4.69%  Size N 6.3822%/-2.24%  Tradin 4.9840%/+4.11%  [72s; --industries --membership]
V2_mixed lag 1  R2 0.299406  periods 962  forecasts 468  Volati 8.6255%/+4.72%  Moment 6.5735%/+2.74%  Size 6.8966%/+4.90%  Size N 6.3820%/-2.57%  Tradin 4.9838%/+4.59%  [58s; --industries --membership --mixed-blends]
V2_828 lag 1  R2 0.301109  periods 386  forecasts 468  Volati 7.7448%/+12.89%  Moment 7.2364%/+0.42%  Size 7.5541%/+0.58%  Size N 6.7614%/-0.68%  Tradin 4.9768%/+2.61%  [39s; --industries --membership --beta-window 36 --beta-half-life inf]
```

### runs_tree3.out -- item 3's tree: USE4, use3, and no Size Nonlinearity

```text
V3     lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [80s; defaults]
V3_use3 lag 1  R2 0.299429  periods 938  forecasts 468  Volati 8.6253%/+3.47%  Moment 6.5735%/+3.64%  Size 6.8968%/+4.69%  Size N 6.3822%/-2.24%  Tradin 4.9840%/+4.11%  [63s; --size-nonlinearity use3]
V3_nosnl  [62s; defaults]
```

### runs_tree4.out -- item 4's tree: the MAD and the sd, on 938 and 962 periods

```text
V4     lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [74s; --industries --membership]
V4_sd  lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [57s; --industries --membership --residual-spread sd]
V4_962 lag 1  R2 0.298059  periods 962  forecasts 468  Volati 8.7734%/+4.65%  Moment 6.6235%/+2.69%  Size 4.0325%/+2.80%  Size N 2.5424%/+0.29%  Tradin 5.0651%/+4.89%  [59s; --industries --membership --mixed-blends]
V4_962_sd lag 1  R2 0.298059  periods 962  forecasts 468  Volati 8.7734%/+4.65%  Moment 6.6235%/+2.69%  Size 4.0325%/+2.80%  Size N 2.5424%/+0.29%  Tradin 5.0651%/+4.89%  [57s; --industries --membership --mixed-blends --residual-spread sd]
```

### runs_tree5.out -- item 5's tree

```text
V5     lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [79s; --industries --membership]
```

### steps.txt -- steps.py: the decomposition, and item 2's check against V1

```text
              R2  periods       first  univ  fcst       Volatility         Momentum             Size       Size Nonli       Trading Ac  spread median sr
H       0.294211      938  2023-02-08   482   468  7.63% +4.52%(+1  7.16% +2.44%(+0  7.57% +4.92%(+1  6.67% -1.44%(-0  4.99% +3.90%(+2  0.3682    23.53%
V1      0.299406      962  2023-01-04   483   468  8.63% +4.72%(+1  6.57% +2.74%(+0  6.90% +4.90%(+1  6.38% -2.57%(-1  4.98% +4.59%(+2  0.2793    23.43%
V2      0.299429      938  2023-02-08   482   468  8.63% +3.47%(+0  6.57% +3.64%(+1  6.90% +4.69%(+1  6.38% -2.24%(-0  4.98% +4.11%(+2  0.3632    23.46%
V3      0.298076      938  2023-02-08   482   468  8.77% +3.39%(+0  6.62% +3.60%(+1  4.03% +2.86%(+1  2.54% +0.40%(+0  5.07% +4.40%(+2  0.3612    23.40%
V4      0.298076      938  2023-02-08   482   468  8.77% +3.39%(+0  6.62% +3.60%(+1  4.03% +2.86%(+1  2.54% +0.40%(+0  5.07% +4.40%(+2  0.1333    23.77%
V5      0.298076      938  2023-02-08   482   468  8.77% +3.39%(+0  6.62% +3.60%(+1  4.03% +2.86%(+1  2.54% +0.40%(+0  5.07% +4.40%(+2  0.1333    23.77%

H      noise Volatility 5.3%, Momentum 4.0%, Size 9.9%, Size Nonlinearity 17.1%, Trading Activity 11.5%; t +1.37, +0.81, +1.54, -0.54, +2.14; spread sd 0.3682 mad nan; specific p10/50/90 15.06%/23.53%/36.35%; fallback 8; floored none; delivered floor none
V1     noise Volatility 3.9%, Momentum 4.4%, Size 11.6%, Size Nonlinearity 18.6%, Trading Activity 10.0%; t +1.30, +0.94, +1.65, -1.01, +2.86; spread sd 0.2793 mad nan; specific p10/50/90 15.18%/23.43%/36.06%; fallback 7; floored none; delivered floor none
V2     noise Volatility 3.9%, Momentum 4.4%, Size 11.6%, Size Nonlinearity 18.6%, Trading Activity 10.0%; t +0.94, +1.22, +1.56, -0.86, +2.53; spread sd 0.3632 mad nan; specific p10/50/90 15.10%/23.46%/36.35%; fallback 8; floored none; delivered floor none
V3     noise Volatility 3.8%, Momentum 4.4%, Size 7.1%, Size Nonlinearity 41.8%, Trading Activity 10.1%; t +0.92, +1.21, +1.69, +0.30, +2.67; spread sd 0.3612 mad nan; specific p10/50/90 15.16%/23.40%/36.11%; fallback 8; floored none; delivered floor none
V4     noise Volatility 3.8%, Momentum 4.4%, Size 7.1%, Size Nonlinearity 41.8%, Trading Activity 10.1%; t +0.92, +1.21, +1.69, +0.30, +2.67; spread sd 0.3612 mad 0.1333; specific p10/50/90 15.40%/23.77%/36.50%; fallback 8; floored none; delivered floor none
V5     noise Volatility 3.8%, Momentum 4.4%, Size 7.1%, Size Nonlinearity 41.8%, Trading Activity 10.1%; t +0.92, +1.21, +1.69, +0.30, +2.67; spread sd 0.3612 mad 0.1333; specific p10/50/90 15.40%/23.77%/36.50%; fallback 8; floored none; delivered floor none

item 2: V2's 938 periods' factor returns are V1's for the same dates bit for bit: True; dropped 24, 2023-01-04 .. 2023-02-07; V2's excluded 24 = the dropped: True; universe 483 -> 482, leaving: ['VNO']
```

### size_pair.txt -- size_pair.py: item 3 against the raw cube and without the factor

```text
Exposures: Size and Size Nonlinearity over the members with both
  USE3 raw cube          equal-weighted correlation: last day +0.931, median over the 939 dates +0.933 (range +0.928 .. +0.943); sqrt-cap-weighted on the last day +0.911; SNL range among members on the last day [-3.25, +0.57]
  USE4                   equal-weighted correlation: last day +0.585, median over the 939 dates +0.551 (range +0.470 .. +0.598); sqrt-cap-weighted on the last day +0.000; SNL range among members on the last day [-3.61, +0.90]

Factor returns
  USE3 raw cube          R2 0.299429  Size vol 6.90% mean +4.69% (t +1.56, noise 11.6%); SNL vol 6.38% mean -2.24% (t -0.86, noise 18.6%); their correlation -0.824 in the covariance, -0.851 in the plain series
                         other styles' vols: Volatility 8.63%, Momentum 6.57%, Trading Activity 4.98%
  USE4                   R2 0.298076  Size vol 4.03% mean +2.86% (t +1.69, noise 7.1%); SNL vol 2.54% mean +0.40% (t +0.30, noise 41.8%); their correlation -0.308 in the covariance, -0.322 in the plain series
                         other styles' vols: Volatility 8.77%, Momentum 6.62%, Trading Activity 5.07%
  no Size Nonlinearity   R2 0.293802  Size vol 4.03% mean +2.89% (t +1.71, noise 7.1%)
                         other styles' vols: Volatility 8.70%, Momentum 6.64%, Trading Activity 4.86%

The pair's risk on the last date, sqrt(x' S x) over the Size block
  over the 423 members with a forecast in all three
  USE3 raw cube          smallest 85 members, EW    exposures -3.110 -2.814  risk 12.16%
  USE3 raw cube          largest 50 members, EW     exposures +0.012 +0.551  risk 3.45%
  USE3 raw cube          the index, cap-weighted    exposures +0.002 +0.001  risk 0.01%
  USE4                   smallest 85 members, EW    exposures -3.110 -1.647  risk 11.94%
  USE4                   largest 50 members, EW     exposures +0.012 +0.155  risk 0.38%
  USE4                   the index, cap-weighted    exposures +0.002 +0.000  risk 0.01%
  no Size Nonlinearity   smallest 85 members, EW    exposures -3.110  risk 12.52%
  no Size Nonlinearity   largest 50 members, EW     exposures +0.012  risk 0.05%
  no Size Nonlinearity   the index, cap-weighted    exposures +0.002  risk 0.01%

Size's factor return with and without Size Nonlinearity in the model
  USE3 raw cube          against none: correlation 0.7935, the difference's vol 3.79% a year and mean +1.79%, plain vol ratio 1.785, slope 1.416
  USE4                   against none: correlation 0.9998, the difference's vol 0.06% a year and mean -0.03%, plain vol ratio 1.005, slope 1.005
  no Size Nonlinearity   Size plain vol 3.25%, mean +2.89%

How many members the clip moves, over the same dates
  USE3, LCAPCB's raw clip at four 1.4826 MADs and the +/-4 guard: 85,319 of 422,881 member values (20.2%; the raw clip alone 85,319), median 91 a date, 81 of 425 on the last; its raw bounds there [-1.783, +1.781] -- |z| 1.21 and 1.21 cubed, roughly
  USE4, the cube standardised and clipped at +/-4: 4,391 (1.04%), median 4 a date, at most 10, 4 on the last
```

### in_regression.txt -- in_regression.py: the pair's orthogonality inside each regression

```text
use4: over each of the 938 regressions, by its weights -- covariance of Size and Size Nonlinearity median +3.35e-17, largest |.| 8.85e-03; correlation median +0.0000, range -0.0106 .. +0.0008; members with a Size exposure the regression leaves out (no next-day return or an incomplete row): median 0, max 2
use3: over each of the 938 regressions, by its weights -- covariance of Size and Size Nonlinearity median +1.19e+00, largest |.| 1.27e+00; correlation median +0.9077, range +0.8926 .. +0.9230; members with a Size exposure the regression leaves out (no next-day return or an incomplete row): median 0, max 2
```

### spread_effect.txt -- spread_effect.py: item 4 on both samples, both measures

```text
The spread, both measures, on both samples
  938 periods: 1.4826 x MAD 0.1333, sd 0.3612; largest shares of the sd's square: FRC 72.4%, SIVB 3.7%, LUMN 2.2%, BNY 1.5%, ENPH 1.0%
  962 periods: 1.4826 x MAD 0.1334, sd 0.2805; largest shares of the sd's square: FRC 57.2%, VNO 5.2%, LUMN 3.0%, BNY 2.5%, ENPH 1.7%
  mad: 938 -> 962 periods 0.1333 -> 0.1334 (+0.1%)
  sd: 938 -> 962 periods 0.3612 -> 0.2805 (-22.3%)
  938, MAD   used 0.1333
  938, sd    used 0.3612
  962, MAD   used 0.1334
  962, sd    used 0.2805

Blend weights: the characteristics' weight per security, from its residual count
  938, MAD   over 482 securities with a history (n 21 .. 938, median 938): weight at the median history 2.92%; across securities min 2.92%, p10 2.92%, median 2.92%, p90 3.85%, max 58.46%
  938, sd    over 482 securities with a history (n 21 .. 938, median 938): weight at the median history 0.41%; across securities min 0.41%, p10 0.41%, median 0.41%, p90 0.54%, max 16.08%
  962, MAD   over 483 securities with a history (n 2 .. 962, median 962): weight at the median history 2.84%; across securities min 2.84%, p10 2.84%, median 2.84%, p90 3.81%, max 96.56%
  962, sd    over 483 securities with a history (n 2 .. 962, median 962): weight at the median history 0.66%; across securities min 0.66%, p10 0.66%, median 0.66%, p90 0.89%, max 86.40%

The delivered forecasts
  938 periods, sd -> MAD: 468 forecasts; change rms 0.677 points (1.98% relative), mean +0.299, largest 10.020, 15 past 1 point; p10/median/p90 15.16%/23.40%/36.11% -> 15.40%/23.77%/36.50%
    largest: SBNY 69.50% -> 59.48%, LUMN 67.93% -> 63.93%, ALK 29.05% -> 32.20%, NWL 34.34% -> 37.33%, WHR 28.44% -> 30.77%, RHI 20.46% -> 22.42%
  962 periods, sd -> MAD: 468 forecasts; change rms 0.436 points (1.62% relative), mean +0.271, largest 3.428, 12 past 1 point; p10/median/p90 15.20%/23.35%/36.00% -> 15.44%/23.66%/36.27%
    largest: VNO 45.36% -> 41.93%, ALK 28.54% -> 31.13%, SBNY 57.50% -> 54.94%, NWL 33.54% -> 35.97%, WHR 28.68% -> 30.60%, RHI 20.73% -> 22.35%
  MAD, 938 -> 962 periods: 468 forecasts; change rms 0.455 points (1.14% relative), mean -0.068, largest 5.798, 11 past 1 point; p10/median/p90 15.40%/23.77%/36.50% -> 15.44%/23.66%/36.27%
    largest: LUMN 63.93% -> 58.13%, SBNY 59.48% -> 54.94%, AAP 54.10% -> 51.29%, DXC 44.07% -> 42.11%, ALGN 37.42% -> 39.07%, NWL 37.33% -> 35.97%
  sd, 938 -> 962 periods: 468 forecasts; change rms 0.808 points (1.57% relative), mean -0.040, largest 12.003, 12 past 1 point; p10/median/p90 15.16%/23.40%/36.11% -> 15.20%/23.35%/36.00%
    largest: SBNY 69.50% -> 57.50%, LUMN 67.93% -> 57.90%, VNO 40.96% -> 45.36%, AAP 52.64% -> 49.86%, DXC 42.79% -> 40.93%, ALGN 37.32% -> 39.02%
  938, MAD   median 23.77%, p10 15.40%, p90 36.50%; delivered floor none, characteristic forecast below it none
  938, sd    median 23.40%, p10 15.16%, p90 36.11%; delivered floor none, characteristic forecast below it none
  962, MAD   median 23.66%, p10 15.44%, p90 36.27%; delivered floor none, characteristic forecast below it none
  962, sd    median 23.35%, p10 15.20%, p90 36.00%; delivered floor none, characteristic forecast below it none
```

### final_check_first.txt -- the finished tree's first run, before the clip line's wording

```text
F      lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [73s; --industries --membership]
F_nomem lag 1  R2 0.290175  periods 938  forecasts 468  Volati 8.9372%/+3.66%  Moment 6.6476%/+3.76%  Size 4.3011%/+3.13%  Size N 3.5242%/+1.27%  Tradin 4.9177%/+4.63%  [70s; --industries]
F_eq   lag 1  R2 0.254680  periods 938  forecasts 468  Volati 7.1811%/+3.14%  Moment 6.3083%/+3.12%  Size 3.6587%/+3.31%  Size N 1.8254%/-0.13%  Tradin 3.4710%/+2.90%  [90s; --industries --membership --equal-weight]
F_noind lag 1  R2 0.165043  periods 938  forecasts 468  Volati 9.1946%/+4.94%  Moment 7.5405%/+4.15%  Size 4.3491%/+2.80%  Size N 2.8003%/+0.51%  Tradin 4.7549%/+3.94%  [65s; --membership]
F vs V5: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 0
```

### final_check.txt -- the finished tree, and its other paths

```text
F      lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [84s; --industries --membership]
F_nomem lag 1  R2 0.290175  periods 938  forecasts 468  Volati 8.9372%/+3.66%  Moment 6.6476%/+3.76%  Size 4.3011%/+3.13%  Size N 3.5242%/+1.27%  Tradin 4.9177%/+4.63%  [66s; --industries]
F_eq   lag 1  R2 0.254680  periods 938  forecasts 468  Volati 7.1811%/+3.14%  Moment 6.3083%/+3.12%  Size 3.6587%/+3.31%  Size N 1.8254%/-0.13%  Tradin 3.4710%/+2.90%  [66s; --industries --membership --equal-weight]
F_noind lag 1  R2 0.165043  periods 938  forecasts 468  Volati 9.1946%/+4.94%  Moment 7.5405%/+4.15%  Size 4.3491%/+2.80%  Size N 2.8003%/+0.51%  Tradin 4.7549%/+3.94%  [64s; --membership]
F vs V5: factor returns True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 2
   -  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,391 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.0e-15 (4.7 eps), inside rounding on 939 of the 939 dates
   +  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,391 member values, 4 on 2026-11-03; its regression-weighted covariance with Size over the index members is at most 1.0e-15 (4.7 eps), inside rounding on 939 of the 939 dates
```

### final_ast.txt -- the finished tree's code against tree_5's

```text
src/praxis/midoffice/model/descriptors.py: text identical to tree_5 False; code identical, docstrings aside, True
src/praxis/midoffice/model/crosssectional.py: text identical to tree_5 True; code identical, docstrings aside, True
src/praxis/midoffice/model/specificrisk.py: text identical to tree_5 False; code identical, docstrings aside, True
scripts/run_sp500_model.py: text identical to tree_5 False; code identical, docstrings aside, False
```

### pytest_tail.txt -- the full suite, last lines

```text
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,

tests/test_backtest_validation.py::TestEventDrivenFillModel::test_fill_on_close
  C:\Data\Development\Python\McTheoryApps\praxis\.venv\Lib\site-packages\numpy\_core\_methods.py:211: RuntimeWarning: invalid value encountered in scalar divide
    ret = ret.dtype.type(ret / rcount)

tests/test_microstructure_utils.py::TestModifiedZScore::test_degenerate_all_equal
  C:\Data\Development\Python\McTheoryApps\praxis\engines\microstructure_utils.py:55: RuntimeWarning: invalid value encountered in multiply
    result = np.where(x == med, 0.0, np.sign(x - med) * np.inf)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========== 2231 passed, 1 skipped, 19 warnings in 110.05s (0:01:50) ===========
```

### pytest_mid.txt -- the midoffice tests, last line

```text
======================= 780 passed, 1 skipped in 44.33s =======================
```

### ruff.txt -- ruff on the changed Python files, against HEAD

```text
ruff check, working tree (the three below were all in HEAD before this change; the E501 the first pass found was fixed):
scripts\run_sp500_model.py:34:5: F401 [*] `praxis.midoffice.model.covariance.HorizonCovariance` imported but unused
src\praxis\midoffice\model\crosssectional.py:92:20: F401 [*] `typing.Mapping` imported but unused
tests\midoffice\test_crosssectional.py:10:1: I001 [*] Import block is un-sorted or un-formatted
Found 3 errors.
[*] 3 fixable with the `--fix` option.

the same files at HEAD, checked from the repo (git stash is not used: the committed text is piped in):
All checks passed!
src\praxis\midoffice\model\crosssectional.py:92:20: F401 [*] `typing.Mapping` imported but unused
All checks passed!
scripts\run_sp500_model.py:34:5: F401 [*] `praxis.midoffice.model.covariance.HorizonCovariance` imported but unused
All checks passed!
tests\midoffice\test_crosssectional.py:10:1: I001 [*] Import block is un-sorted or un-formatted
All checks passed!
```

### variant_F.txt -- the run's own report, the finished tree at its defaults

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
  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days (the library's default, USE4's)
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
  Size Nonlinearity    the Size exposure cubed, standardised and clipped at +/-4, orthogonalised against Size with the regression weights, standardised again (USE4)
  Trading Activity     STOM, STOQ, STOA, VLVR
  - SERDP is in no blend: its residual series is kept for the last window alone, so it has a value on the last date and no other; no estimated period reads it, and blended it would enter the forecast exposures alone
  - LCAPCB is in no blend: USE3's raw cube; Size Nonlinearity is built from the Size exposure instead, USE4's way

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
  mean cross-sectional R-squared 0.298
  0 collinear period(s) of 938 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 43 security-period(s) of the regressions move -- 42 in, on the day a name left the index, 1 out, on the day one joined; the mask itself differs on 62
  each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2023-02-08 (938)
    Momentum             RSTR, HALPHA from 2023-02-08 (938)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LNCAP from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA, VLVR from 2023-02-08 (938)
  every estimated period's blend is the forecast exposures', 2026-11-03's, in every index: one definition per factor across the sample
  24 period(s) held enough securities and are not estimated, a risk index's blend in each not being the forecast's:
    2023-01-04 to 2023-02-07 (24): Volatility from BTSG, DASTD, HILO, LPRI, without CMRA; Momentum from HALPHA, without RSTR; Trading Activity from STOM, STOQ, without STOA, VLVR

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 3.3e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  +1.2e-17   1.000000   426
  Momentum                    -5.4e-17   1.000000   423
  Size                        -3.1e-17   1.000000   425
  Size Nonlinearity           +5.0e-17   1.000000   425
  Trading Activity            -7.7e-18   1.000000   425
  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,391 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.0e-15 (4.7 eps), inside rounding on 939 of the 939 dates

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     504    506          504          503
  of them in the panel      426     458    486          486          426
  in the regression         422     451    476          476          422

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               17.06%     +23.03%       52.0%      938          0      0
  Finance & Real Estate      15.31%     +21.56%       14.3%      938          0      0
  Manufacturing              10.95%     +20.59%       20.7%      938          0      0
  Mining                     20.16%     +10.85%       16.2%      938          0      0
  Retail                     13.09%     +13.07%       10.7%      938          0      0
  SIC20                      12.74%      +7.85%       30.6%      938          0      0
  SIC28                      13.11%     +15.02%        9.8%      938          0      0
  SIC35                      17.45%     +26.15%        6.3%      938          0      0
  SIC36                      20.75%     +28.22%        5.2%      938          0      0
  SIC37                      14.15%     +12.69%       18.2%      938          0      0
  SIC38                      14.19%     +15.35%       11.5%      938          0      0
  SIC49                      15.80%     +21.24%       12.3%      938          0      0
  SIC60                      21.92%     +19.03%        7.3%      938          0      0
  SIC63                      14.86%     +18.57%       13.8%      938          0      0
  SIC67                      15.19%     +17.12%       17.2%      938          0      0
  SIC73                      17.34%     +14.93%        3.5%      938          0      0
  Services                   12.86%     +14.04%       26.4%      938          0      0
  Transport & Utilities      13.34%     +12.35%       16.9%      938          0      0
  Wholesale                  14.77%     +18.69%       40.0%      938          0      0
  Volatility                  8.77%      +3.39%        3.8%      938          0      0
  Momentum                    6.62%      +3.60%        4.4%      938          0      0
  Size                        4.03%      +2.86%        7.1%      938          0      0
  Size Nonlinearity           2.54%      +0.40%       41.8%      938          0      0
  Trading Activity            5.07%      +4.40%       10.1%      938          0      0

  specific risk   median 23.77%   p10 15.40%   p90 36.50%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 0.9%, CTVA 0.9%
                    relative residual spread 0.133, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.361
                    largest shares of the sd's square: FRC 72.4%, SIVB 3.7%, LUMN 2.2%, BNY 1.5%, ENPH 1.0%
                    the characteristics' weight in a blended forecast, at the median history of 938 periods: 2.92%

Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
  - 38 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CDAY
  - 52% of the estimated variance of Construction is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 8 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: INFO, IPGP, MRSH, PENN, PVH
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### v2_828_excerpt.txt -- the rule under the old window: the report's blend and drop lines

```text
variant_V2_828.txt, the composition and drop lines (lines 73-86):
  23-factor model, annualised, estimated on 386 daily period(s) for 468 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 234.4 effective observation(s) of 386
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.301
  0 collinear period(s) of 386 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 17 security-period(s) of the regressions move -- 16 in, on the day a name left the index, 1 out, on the day one joined; the mask itself differs on 28
  each risk index's blend over the 386 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2025-04-23 (386)
    Momentum             RSTR, HALPHA from 2025-04-23 (386)
    Size                 LNCAP from 2025-04-23 (386)
    Size Nonlinearity    LCAPCB from 2025-04-23 (386)
    Trading Activity     STOM, STOQ, STOA, VLVR from 2025-04-23 (386)
  every estimated period's blend is the forecast exposures', 2026-11-03's, in every index: one definition per factor across the sample
  552 period(s) held enough securities and are not estimated, a risk index's blend in each not being the forecast's:
    2023-02-08 to 2025-04-22 (552): Volatility from DASTD, HILO, LPRI, CMRA, without BTSG; Momentum from RSTR, without HALPHA; Trading Activity from STOM, STOQ, STOA, without VLVR

its industries (lines 55-67):
  449 of 506 securities are in at least one estimated period's regression while an index member; every count below is over them
  18 industries over 449 securities in the universe (506 classified); largest SIC73 with 48 (11%), 45 thin group(s) merged into their division
    SIC73                      48
    Manufacturing              45
    SIC38                      36
    SIC28                      34
    SIC49                      33
    Retail                     28
    SIC35                      27
    SIC67                      25
  - 2 industry/industries have no member in the universe and are not factors: SIC24, Unclassified
  - 1 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture
  - Other Industries pooled only 1, below the hard floor of 5, so it is not a factor. 1 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA
```

## git add

Everything above is one change. Nothing is staged; Code does not commit (Rule 43).

```
git add src/praxis/midoffice/model/descriptors.py src/praxis/midoffice/model/crosssectional.py src/praxis/midoffice/model/specificrisk.py scripts/run_sp500_model.py tests/midoffice/test_descriptors.py tests/midoffice/test_crosssectional.py tests/midoffice/test_specificrisk.py docs/FACTOR_MODEL.md claude/handoffs/BRIEF_midoffice04_settle_measured_definitions.md claude/retros/RETRO_midoffice04_settle_measured_definitions.md
```
