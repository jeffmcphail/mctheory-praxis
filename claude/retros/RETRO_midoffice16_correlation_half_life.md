# Retro: Mid-office 16 -- midoffice15's rulings, then the calm half's shape: the correlation half-life on the regime-adjusted model

**Brief:** `claude/handoffs/BRIEF_midoffice16_correlation_half_life.md` -- written in Chat and
pasted into the Code session, transcribed verbatim
**Date:** 2026-11-13 (EST)
**Duration:** the brief arrived shortly before 15:56Z (10:56 EST), the first clock reading after
it.
- **The rulings:** read and built by 11:07, when the released panel was written into `data/`.
- **The correlation half-lives' one-pass test:** built, its tests written and five smoke runs
  made by 11:23.
- **The runs, one at a time through the runner:** the new defaults on midoffice15's panel (N16,
  11:24 to 11:33), on the new one (D16, 11:33 to 11:44), the correlation half-lives (C16,
  11:44 to 11:54), and two plain runs for the factor side's cost (P16x and P16, 11:54 to 12:02).
- **Then** the analyses, the docs, the state file and this retro.

**Status:** COMPLETE.
- **Rulings 1 to 4:** built and measured. The regime adjustment's factor side is the model's
  default at 21 days, its specific side and the eigenvalue adjustment off; the whole-test identity
  confirmed for every family and put in the docs; S&P Global's ruling released, the panel with
  every ruling applied the run's default.
- **Items 1 to 4:** built and measured -- the new defaults reproduced and run on the new panel;
  the correlation half-lives tested in one pass; judged by the rule written before the run.
- **No default changed** beyond rulings 1 and 4.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 4, S&P Global: released.** Every line of the ruling's arithmetic checks against the
  cache's bars (the WI close and the press as Chat quoted them). Kalkine's 436.69 is an intraday
  price of 07-02, and 436.69 / 1.0523 lands on Polygon's 414.97 within the rounding of 5.23%.
  - **The ruling:** the evidence carries the release and its four sources; `held` is gone; the
    mechanism stays.
  - **The panel with every ruling applied, `_released`,** is the default. Against `_revised`,
    SPGI's 2026-07-01 alone (+1.89% to +7.10%) and its index's rescaling; against `_spinoffs`,
    FDXF's counts alone, the total returns equal to the bit -- as the ruling said.
- **Ruling 1, the factor side the model's default at 21 days.** The identity holds:
  - **To rounding:** within 0.001 for the broad portfolios, the optimiser, the industries and the
    random long-only portfolios (ddof alone 0.0007).
  - **Where it fails:** a few thousandths where a portfolio's mean outcome turns between the
    halves; a member whose halves differ in size (45 of 504; JNPR 0.545 against 1.562, where the
    counts weigh); a family's summary (the industries' mean abs(bias - 1) 0.046 whole, 0.094 and
    0.110 by half).
  - **The docs:** beside the bias statistics. A whole-test band can pass a model wrong in both
    halves, so the rolling windows are the criterion.
- **Rulings 2 and 3:** the specific side and the eigenvalue adjustment stay off.
- **Item 1, the new defaults.**
  - **On `_revised` (N16):** V15's F 21, S off column to the bit in every array, the regime
    history included.
  - **On `_released` (D16):** nothing before 2026-07-01 moves. After it every forecast moves
    through that day's factor returns, the multiplier at most +3.5e-4; SPGI's outcome that day
    goes from +1.0 to +3.7 sd, and every other statistic is unchanged to the third decimal but
    one (0.887 to 0.886).
  - **The plain run's cost:** the factor side costs it 6.7 minutes, 7.5 against 0.8. It need not:
    each refit's factor returns are the full fit's through its date, to the bit, so the history
    from the full fit alone is the refits' to the bit in 0.3 s. Checked, not built.
- **Item 2, the correlation half-lives in one pass** (C16, 9.5 minutes; its model test D16's to
  the bit).
  - **The one pass:** each refit's own factor returns re-estimated by the model's estimator; a
    63-day setting equals a run at 63 alone to the bit.
  - **The regime multiplier:** the model's to rounding, 2.2e-16, not to the bit. The two-half-life
    diagonal, c / (sqrt(c) sqrt(c)) times the variance, rounds to one only half the time, so the
    volatilities' last bit moves with the correlations'. Found before the runs.
- **Item 3, the rule keeps 504.** 63 days is out (the unconstrained optimiser realising 10.23% a
  year against 10.19%); 126 and 252 are worse on the first half's criterion (0.104 and 0.100
  against 0.098). Each half prefers the other end, monotonically: at 63 the calm half's criterion
  goes from 0.123 to 0.083, the first half's from 0.098 to 0.116.
- **Item 4: a shorter half-life fixes part of the shape, and the optimiser pays in the small
  eigen-directions.**
  - **The part fixed:** at 63, on the model's own directions, the calm half's mean abs(bias - 1)
    goes from 0.222 to 0.168.
  - **The market direction:** only 0.814 to 0.858 -- 6% off its forecast volatility, where it
    would need 19% (1 - 0.814). The rest is on the volatility side.
  - **The price:** the optimiser's forecast variance in ranks 16-25 rises from 20.6% to 23.8%,
    those parts realising 1.25 times their forecast against 1.17, in both halves.
  - **The test that would tell,** proposed and not run: the one-sided eigenvalue adjustment on
    the 63-day correlations, by the same rule.
- **Which I would choose:** the correlations on 504 days; next, the largest members' specific
  risk. No default changed beyond rulings 1 and 4.
- **Tests:** 2,374 passed, 1 skipped; 2 new, 1 changed.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice16_correlation_half_life.md`, verbatim. In
outline:

- **"committed -- midoffice15."** Checked: 355b772 in `git log`. The context had been compacted
  while idle, so the state file and midoffice15's retro were read before anything else.
- **Rulings on midoffice15's decisions:**
  1. The regime adjustment's factor side on, at 21 days, the model's default. The whole test's
     level is not a cost of it: each portfolio's whole-test statistic is, to rounding, the root
     mean square of its two halves'. Confirm the identity for every family at one day, say where
     it fails and why, and put it in `docs/FACTOR_MODEL.md` beside the bias statistics -- a
     whole-test band can pass a model wrong in both halves, which is why the rolling windows are
     the criterion.
  2. Its specific side off: a level, set by the largest members, whose fix is by size.
  3. The eigenvalue adjustment, one-sided as two-sided: off.
  4. S&P Global: the ruling stands, the hold released. Polygon's 407.26 is the regular-way close
     of 2026-06-30 (SPGI WI 384.00, the 23.26 between about a Mobility share); 07-01's rally
     filled the gap; the press read the day the same way. Write it into the ruling's evidence
     with the sources, as quoted from Chat's search results of 2026-11-13 (the pages not opened),
     and check the arithmetic. Keep the held mechanism. The panel with every ruling applied,
     under a new name, the old ones kept, becomes the default: against `_revised` only SPGI's
     2026-07-01 return and its index's rescaling should change, against `_spinoffs` only FDXF's
     counts.
  5. Next: the correlation half-life on the regime-adjusted model; then the largest members'
     specific risk.
- **midoffice16, the calm half's shape,** the rulings first:
  1. The new defaults' bias run on `_revised` must reproduce V15's F 21, S off column, to the bit
     or to rounding with the reason; then on the new panel (D16), where only SPGI's 2026-07-01
     should move anything. Say what the factor side costs a plain run, in minutes.
  2. Correlation half-lives of 63, 126, 252 and 504 days (the default), the volatilities' 90
     kept, on the regime-adjusted model. The regime multiplier should be the model's under every
     setting: confirm it to the bit. One pass for all four if the factor returns can be shared.
  3. Judged as VE15 was, the rule in the code before the run: the minimum-variance portfolios'
     realised volatility first -- a setting under which either realises more over the whole test
     than at 504, to the hundredth of a point, is out -- then `REGIME_CRITERION`, chosen on the
     first half and checked on the second. Beside them the eigenportfolios by rank and half, rank
     1 first; the index and the optimiser by half, the members by capitalisation quintile, and
     how much each setting moves the index's forecast from day to day.
  4. If a shorter half-life fixes the shape but the optimiser pays for it in the small
     eigen-directions, say so and propose the test that would tell, the one-sided eigen
     adjustment on it; don't run it.
- Runs one at a time through the runner. **"Change no default beyond rulings 1 and 4. Report,
  and say which you would choose and why."** Retro with its `git add` and the state file in the
  same commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `scripts/run_sp500_model.py` | `DEFAULT_PANEL` `_released`; `REGIME_FACTOR_HALF_LIFE` 21, `--regime-factor-half-life`'s default (ruling 1); `--bias-correlation-half-lives` and its checks; the bias loop's correlation settings, their histories, their tests, the covariances kept and each setting's volatilities against the model's; `CORRELATION_RULE`, `_correlation_key`, `_member_quintiles`, `_report_correlation`; `_bias_arrays` keeps the optimiser's exposures; `_save_bias` the settings, their histories and the covariances; the plain run's regime section times its refits; the report names its panel | 55, 87-94, 126-133, 931-933, 1021-1038, 1115-1177, 1310-1312, 1322, 1753-1916, 1935-1937, 1949-1983, 2146-2161, 2174-2182, 2217-2229, 2270 |
| `src/praxis/midoffice/model/crosssectional.py` | `reestimated_covariance`: a fitted model's covariance by another estimator, from its own factor returns | 1529-1553 |
| `src/praxis/midoffice/model/bias.py` | `RiskForecast.from_model(covariance=)`; the bias test keeps the minimum-variance portfolios' factor exposures (`BiasRun.min_variance_exposure`) | 235-262, 437-439, 511, 588, 623, 689-690 |
| `scripts/build_sp500_securities.py` | `--suffix` `_released` by default; the docstring | 51-53, 59-63, 155-158, 165-166 |
| `tests/midoffice/test_bias.py` | 2 new tests | 398-466 |
| `tests/midoffice/test_securities.py` | the held-ruling test reads the repo's file: no ruling held, SPGI's released | 592-593, 622-626 |
| `config/sp500_spin_off_overrides.json` | SPGI's ruling: `held` removed, the release in its evidence and `ruled` | 245-247 |
| `docs/FACTOR_MODEL.md` | the whole-test identity beside the bias statistics; the regime adjustment's defaults; S&P Global's release; the released panel; the midoffice16 results | 260-264, 425-439, 1225-1245, 1334-1339, 1372-1397, 1605-1609, 1641-1694 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice16_correlation_half_life.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice16_correlation_half_life.md` | new: this retro | new |

Written outside the tracked files:

- `data/external/polygon/sp500_securities_2022-01-03_2026-11-03_released.npz` and `.json`, the
  panel with every ruling applied and its report -- new files; every earlier panel is untouched
  `_listed`, `_spinoffs`, `_revised` and `_released` all rebuild to the bit in the final checks).
- The cache: nothing asked, nothing written.
- `claude/scratch/midoffice16/` (gitignored): the scratch, below.

---

## Ruling 4: S&P Global's Mobility, released

Done first, since the panel it makes is the one the runs read.

**The arithmetic, every line recomputed against the cache's bars** (`release_spgi.txt`):

| The ruling's line | Recomputed | |
|---|---|---|
| 407.26 less 384.00 (SPGI WI's close, quoted) | 23.26 | ok |
| a regular-way close near 428, less 384.00: the Mobility it would need | 44.00 | ok |
| Mobility's first open | 20.30 (the cache) | ok |
| SPGI's close of 07-01 | 414.97 (the cache) | ok |
| that over the when-issued close | +8.0651%, "8.07%" | ok |
| its six peers' median that day (MCO, MSCI, ICE, CME, NDAQ, FDS) | +4.2659%, "+4.27%" | ok |
| 436.69 / 1.0523 (Kalkine) | 414.9862, "414.99" | ok |
| 414.99 / 1.0773, the base of "Wednesday's 7.73%" | 385.2095, "385.21" | ok |
| 407.26 less 385.21 | 22.05 | ok |
| Mobility's first close, 21.19, against 22.05 | -3.90% | ok |
| the ruling's 07-01 return, (414.97 + 21.19) / 407.26 - 1 | +7.0962% | ok |

Two details the arithmetic adds:
- **Kalkine's 436.69 is an intraday price of 07-02.** That day's range was 422.59 to 440.01 and
  it closed at 439.89. 436.69 is +5.234% over Polygon's 414.97. 5.23% rounded puts the close it
  implies between 414.97 and 415.01, so "414.99, Polygon's close" holds to that rounding.
- **2026-07-01 was a Wednesday**, as Kalkine's "Wednesday's 7.73% gain" has it.

**Written** (`release_spgi.py`, run once; it refuses a file already released):
- **The evidence:** the hold's reading condensed, then the release in the ruling's words, every
  number from the arithmetic above. The four sources are quoted as Chat's search results of
  2026-11-13, the pages not opened.
- **`ruled`:** gains "the hold released by midoffice16's ruling 4 (Jeff, 2026-11-13)".
- **`held`:** removed. The mechanism stays: the reader still reads it, the build still lists
  and refuses a held ruling, and the test still covers it on a ruling of its own. The repo's
  file now holds none.

**The panel, `_released`** (built into scratch, then into `data/`, the two equal in all 20
arrays and their reports byte for byte; `panel_changed.txt`):
- **Against `_revised`:** 19 of 20 arrays equal to the bit. In `total_return_prices` one
  member-day's return changes beyond 1e-12, SPGI's 2026-07-01 (+1.8931% to +7.0962%), and 36
  more differ in the last bits, SPGI's index after it rescaled.
- **Against `_spinoffs`:** 19 of 20 equal, `total_return_prices` among them. The one that
  differs is `shares_outstanding`: FDXF's 109 member-days, 119,303,378 to 148,943,044.

Exactly what the ruling says should change. The build reads 19 rulings, 0 held, 19 applied.

## Ruling 1: the factor side on, and the whole test's identity

**Built:** `REGIME_FACTOR_HALF_LIFE = 21.0`, the default of `--regime-factor-half-life`, with
`none` to turn it off. The specific side's default stays None (ruling 2), and the eigenvalue
adjustment stays off (ruling 3); nothing to build for either.

**The identity, every family at one day** (`rms_identity.py` on D15, the model, and on V15's F
21, S off, which N16 equals to the bit; `rms_identity.txt`). The statistic is the standard
deviation of z, ddof 1. Over halves of n1 and n2 outcomes, means m1 and m2, it is exactly:

    b^2 = [ (n1 - 1) b1^2 + (n2 - 1) b2^2 + (n1 n2 / n) (m1 - m2)^2 ] / (n - 1)

That is checked to 8e-16 on every portfolio. So b is `sqrt((b1^2 + b2^2) / 2)` where three
things are small: the halves' sizes differ little, ddof's (n1 - 1)/(n - 1) is near a half, and
the halves' mean outcomes are close.

| Family (D15) | portfolios | b less the RMS of its halves: median / largest | the largest, and why |
|---|---|---|---|
| broad portfolios | 4 | 0.0007 / 0.0007 | the index: ddof alone (b = RMS x sqrt(684/685)) |
| minimum variance | 2 | 0.0008 / 0.0008 | the same |
| industries | 20 | 0.0006 / 0.0009 | Finance & Real Estate: its mean z +0.096 then -0.016 |
| random long-only | 100 | 0.0007 / 0.0008 | ddof |
| random long-short | 100 | 0.0006 / 0.0047 | no. 52: its mean z -0.096 then +0.107 |
| eigenportfolios | 25 | 0.0006 / 0.0056 | rank 20: its mean z +0.110 then -0.140 |
| members, halves within 10% of each other in size | 459 | 0.0006 / 0.0154 | a member's trend |
| members, halves of unequal size | 45 | 0.0164 / 1.0170 | JNPR, 343 outcomes then 5: 0.545 against a root mean square of 1.562 |

- **Where it holds:** the brief's examples hold to rounding. The index's 1.015 is 1.224 and
  0.751 (root mean square 1.016); under F 21 its 0.934 is 1.063 and 0.785 (0.934); eigenportfolio
  1's 1.027 is 1.226 and 0.780 (1.028). V15's F 21 column reads the same family by family.
- **Where it fails, and why:**
  - **A turn in the mean:** the mean term grows where a portfolio's standardised outcomes
    trended one way in one half and the other in the next. It reaches a few thousandths.
  - **A member's halves of unequal size:** a member that joined or left the index during the
    test. The counts weigh, and the exact form holds where the plain root mean square does not.
  - **A family's summary:** a median or a mean of statistics is no identity at all. The
    industries' mean abs(bias - 1) is 0.046 over the test and 0.094 and 0.110 in its halves,
    the errors cancelling within each industry.
- **The docs:** a paragraph beside the bias statistics' definition: the identity, where it
  fails, and that a whole-test band can pass a model wrong in both halves, which is why the
  rolling windows are the criterion. midoffice15's bullet that read the index's 0.934 as the
  adjustment's cost now reads it as the calm half's shape.

## Rulings 2, 3 and 5

Nothing to build: the specific side and the eigenvalue adjustment stay off by default. Ruling 5
is this piece.

---

## Item 1: the new defaults

**On midoffice15's panel, `_revised` (N16):** the new defaults' bias run reproduces V15's F 21, S
off column to the bit in every array -- the forecasts and outcomes of every portfolio, every
member, every eigenportfolio, at 1, 5, 11 and 21 days, the optimiser's sizes, the PSD repair --
and the regime history it reads, D15's own (`equal_N16_V15.txt`).
- **The run:** `--bias --bias-horizons 1,5,11,21`, the guard off as V15 had it; 8.3 minutes,
  7.2 of them the refits.
- **Why to the bit:** V15 tested F 21 as the model's forecast scaled by its multiplier, and the
  default now does the same, through the same history.

**On the new panel, `_released` (D16, the guard on, 11.1 minutes):** where the panels differ --
SPGI's 2026-07-01 alone -- is the only thing that moves anything (`compare_bias.txt`).
- **The guard** passed on its five dates (2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27,
  2026-11-02).
- **Nothing before 2026-07-01 moves:** the forecasts, the members' forecasts, the eigenportfolios'
  and the factor covariance are N16's to the bit on every forecast date before it.
- **From 07-01 on** every forecast differs (87 of the 686 dates), through that day's factor
  returns alone: SPGI's +7.10% where N16 had +1.89% moves the day's regression.
  - **The regime history:** that day's B_F^2 goes from 1.632 to 1.659, the factor volatilities
    after it move with it, and lambda_F at 21 days moves at most +3.5e-4 -- 1.10511 to 1.10550, as
    of 07-01 itself.
  - **The statistics:** SPGI's own 07-01 outcome goes from +1.0 to +3.7 sd (D14's +3.8 on the
    unadjusted model), its member statistic from 0.991 to 1.002. Every other statistic is N16's
    to the third decimal; the one figure that moves is the unconstrained optimiser's second
    half, 0.887 to 0.886.
- **The run's own forecast** as of 2026-11-03 carries lambda_F 0.9640 (N16 0.9641), the factor
  volatilities' median 16.26% to 15.67%.

D16's headline figures, the new defaults' standing numbers:

| | whole | first half | second half |
|---|---|---|---|
| the index, cap-weighted | 0.934 (band 0.946-1.054; MRAD 0.098) | 1.063 | 0.785 |
| minimum variance, long-only / unconstrained | 0.990 / 1.033 | 1.109 / 1.163 | 0.855 / 0.886 |
| their realised volatility, % a year | 10.13 / 10.19 | 11.88 / 11.95 | 8.02 / 8.08 |
| members, median | 0.987 | 0.978 | 0.986 |

**What the factor side costs a plain run** (P16x, the factor side off, against P16, the default;
`plain_cost.txt`): **6.7 minutes.**
- **The times:** P16 7.5 minutes against P16x's 0.8, all of it the 686 as-of refits (6.8 minutes by
  P16's own line). The same refits took 8.3 and 8.6 minutes in the two later runs made beside other
  work (`check_fast_regime`, the final checks' capture): about seven minutes on a quiet machine.
- **The reports** differ only by the regime section's five lines; P16's own forecast as of
  2026-11-03 carries lambda_F 0.9640, D16's.
- **It need not cost that** (`check_fast_regime.txt`). Each refit's factor returns are the full
  fit's through its date, to the bit, on all 686 days. So the history read from the full fit alone
  -- the model's estimator on its factor returns' prefixes -- equals the refits' to the bit,
  lambda_F 0.964025715108432 both ways, in 0.3 s. A 30-day smoke of the same check agreed before the
  full run. Not built (decision 4).

---

## Item 2: the correlation half-lives, in one pass

**Built:**
- **`crosssectional.reestimated_covariance(model, estimator)`:** the factor covariance a fit with
  `estimator` would have given -- the estimate from the model's own factor returns, annualised
  as the model is. The covariance is estimated last, from the factor returns alone, so one fit's
  factor returns serve every estimator. Refused for a model whose covariance is not its
  estimator's alone (the sampling correction subtracted, the eigenfactor adjustment scaling it).
- **`RiskForecast.from_model(covariance=)`:** a forecast with that covariance, the model's
  exposures and specific risk; not with an eigen scale.
- **The run's `--bias-correlation-half-lives H,H`:**
  - each as-of refit's factor returns re-estimated by the model's own estimator with its
    correlations on each half-life -- `dataclasses.replace` of its `HorizonCovariance`, the
    volatilities' 90 days and the lags kept;
  - each setting tested beside the model, each on the volatility regime adjustment of its own
    forecasts: its own `RegimeHistory`, the specific side the model's;
  - one pass, 0.69 s a refit with four tests (C16: 9.5 minutes, 7.9 of them the loop);
  - refused without `--bias`, with `--correlation-half-life none`, the sampling correction or
    the eigen adjustment, and beside the eigen-scale and regime passes.
- **`--bias-out`:** each setting's arrays keyed `_c<h>` and its history; and, for every test,
  the factor covariance it forecast with by date, and the minimum-variance portfolios' factor
  exposures (for item 4).
- **The report** (`_report_correlation`, item 3's tables).

**The sharing checked.**
- **Smoke:** on the last 30 forecast dates, the one-pass test at 63 days equals a run with
  `--correlation-half-life 63` alone to the bit in every array: the forecasts and outcomes, the
  regime history its own forecasts give, the covariance, the optimiser's exposures
  (`smoke/equal_c63.txt`).
- **A test** pins the library half (below).
- **C16's model test** equals D16's to the bit in every array (`equal_C16_D16.txt`).

**The regime multiplier under every setting: the model's to rounding, not to the bit.** Each
setting's own history against the model's, in C16:

| | the factor volatilities, date-factors equal to the bit | B_F^2, days equal | lambda_F at 21 days, days equal | largest relative difference: volatilities / B_F^2 / lambda_F |
|---|---|---|---|---|
| 63 | 6,516 of 17,150 | 424 of 685 | 581 of 685 | 6.7e-16 / 4.4e-16 / 2.2e-16 |
| 126 | 6,693 of 17,150 | 441 of 685 | 586 of 685 | 6.7e-16 / 4.4e-16 / 2.2e-16 |
| 252 | 6,635 of 17,150 | 419 of 685 | 594 of 685 | 6.7e-16 / 4.4e-16 / 2.2e-16 |

**Why not to the bit.** The factor volatilities do not depend on the correlations in
arithmetic, but they do in the last bit:
- **The diagonal:** the two-half-life estimator's diagonal is each factor's correlation with
  itself -- c / (sqrt(c) sqrt(c)), from the long window -- times its variance from the short one.
- **The ratio:** that ratio rounds to exactly one only about half the time: 51% of a million
  random c, before any run (`diag_check.py`, its output in the trail below).
- **So:** its rounding moves with the correlations' half-life, and so do the volatilities' last
  bits and everything read from them.
- **In C16:** a setting's multiplier and the model's differ by at most 2.2e-16, one unit in the
  last place, on 13% to 15% of the days.
- **Kept:** each setting's own history. A run at that half-life alone would compute it, so the
  one-pass test equals that run to the bit.
- **The fix that would make it exact** -- the diagonal set to the short window's variances --
  moves every covariance's diagonal in the last bit, so every stored run would then reproduce
  only to rounding. Left for Jeff (decision 3).

---

## Item 3: judged by the rule

**The rule, in the code before the run** (`CORRELATION_RULE`): the minimum-variance portfolios'
realised volatility first -- a setting under which either realises more over the whole test than
at 504, each in % a year to the hundredth of a point, is out -- then `REGIME_CRITERION`, chosen on
the first half among the settings left, checked on the second.

**First, the optimiser's realised volatility**, % a year, whole / first half / second half:

| correlations | long-only | unconstrained | |
|---|---|---|---|
| 63 | 10.12 / 12.10 / 7.67 | **10.23** / 12.18 / 7.83 | out |
| 126 | 10.12 / 12.00 / 7.83 | 10.18 / 12.03 / 7.94 | in |
| 252 | 10.13 / 11.93 / 7.95 | 10.19 / 11.97 / 8.03 | in |
| 504, the model | 10.13 / 11.88 / 8.02 | 10.19 / 11.95 / 8.08 | in |

**Then the criterion**, whole / first half / second half:

| correlations | the regime criterion | the broad portfolios' MRAD | the optimiser's MRAD | the members' median MRAD |
|---|---|---|---|---|
| 63 | 0.104 / 0.116 / **0.083** | 0.077 / 0.056 / 0.113 | 0.150 / 0.212 / 0.056 | 0.084 / 0.080 / 0.081 |
| 126 | 0.105 / 0.104 / 0.110 | 0.087 / 0.059 / 0.143 | 0.144 / 0.174 / 0.106 | 0.084 / 0.078 / 0.082 |
| 252 | 0.105 / 0.100 / 0.119 | 0.092 / 0.065 / 0.155 | 0.139 / 0.157 / 0.121 | 0.084 / 0.078 / 0.082 |
| 504, the model | 0.105 / **0.098** / 0.123 | 0.094 / 0.068 / 0.159 | 0.135 / 0.150 / 0.125 | 0.086 / 0.076 / 0.084 |

**Chosen on the first half by the rule: 504, the model's own** (0.098). On the second half it
reads 0.123, the best there 63 days (0.083), which the rule puts out. Each half prefers the other
end: the shorter the half-life, the better in the calm second half and the worse in the first,
monotonically on the criterion and on the optimiser's realised volatility.

**The shape this piece is for: each setting's own eigenportfolios** -- the eigenvectors of its
covariance -- by rank and half, rank 1 first (the report's tables in full in the appendix):

| | 63 | 126 | 252 | 504 |
|---|---|---|---|---|
| rank 1, whole / first / second | 0.964 / 1.059 / 0.861 | 0.952 / 1.062 / 0.830 | 0.947 / 1.063 / 0.817 | 0.946 / 1.064 / 0.814 |
| ranks 2-25, mean | 1.144 / 1.071 / 1.211 | 1.130 / 1.037 / 1.215 | 1.130 / 1.031 / 1.220 | 1.132 / 1.031 / 1.223 |
| ranks 21-25, mean | 1.303 / 1.231 / 1.371 | 1.220 / 1.146 / 1.290 | 1.198 / 1.127 / 1.265 | 1.195 / 1.124 / 1.261 |
| mean abs(bias - 1) | 0.145 / 0.106 / 0.208 | 0.127 / 0.074 / 0.214 | 0.127 / 0.065 / 0.218 | 0.131 / 0.067 / 0.222 |

- **The calm half's market direction moves toward one, and not far:** rank 1 in the second half
  0.814 at 504, 0.861 at 63. The middle ranks there come down -- ranks 3 to 9 from 1.05-1.29 to
  1.00-1.09 at 63 -- so some of the shape is fixed.
- **The smallest eigen-directions get worse in both halves:** ranks 21-25 from 1.12 to 1.23 in
  the first half and from 1.26 to 1.37 in the second, at 63.
- **In the first half every shorter setting is worse:** the eigenportfolios' mean abs(bias - 1)
  0.067 to 0.106 at 63.

**The index and the optimiser by half**, the statistic, whole / first / second:

| correlations | the index | minimum variance, long-only | minimum variance, unconstrained |
|---|---|---|---|
| 63 | 0.953 / 1.061 / 0.833 | 1.044 / 1.163 / 0.911 | 1.105 / 1.237 / 0.956 |
| 126 | 0.940 / 1.062 / 0.801 | 1.011 / 1.135 / 0.870 | 1.057 / 1.191 / 0.905 |
| 252 | 0.935 / 1.063 / 0.788 | 0.996 / 1.119 / 0.858 | 1.039 / 1.171 / 0.890 |
| 504 | 0.934 / 1.063 / 0.785 | 0.990 / 1.109 / 0.855 | 1.033 / 1.163 / 0.886 |

- **The index:** in the calm half 0.785 to 0.833 at 63, its whole test 0.934 to 0.953, inside its
  band (0.946-1.054). Its first half does not move (1.063 to 1.061): there the correlations were
  not the problem.
- **The optimiser:** at 63 it under-forecasts more in the first half (1.163 and 1.237 against
  1.109 and 1.163) and realises more there (12.10% and 12.18% against 11.88% and 11.95%). In the
  second half it over-forecasts less (0.911 and 0.956 against 0.855 and 0.886) and realises less
  (7.67% and 7.83% against 8.02% and 8.08%).

**The members by capitalisation quintile, the largest first** (whole test): 63 1.039, 0.999,
0.991, 0.957, 0.972; 126 1.035, 0.996, 0.989, 0.953, 0.970; 252 1.029, 0.992, 0.988, 0.950, 0.965;
504 1.028, 0.992, 0.988, 0.949, 0.963. The correlations move the members little, the largest
quintile most (+0.011 at 63).

**Day to day**, the index's forecast, the median absolute change in log volatility (p95): 63
1.06% (3.86%), 126 1.01% (3.67%), 252 1.00% (3.61%), 504 1.00% (3.58%); every member's median,
1.43% to 1.48%. The correlations add little noise to what the regime adjustment brings.

---

## Item 4: the optimiser and the small eigen-directions

**Does a shorter half-life fix the shape? In part, and only in the calm half.** Two readings
from C16's stored arrays (`analyse_correlation.txt`):

- **The same directions.** The model's own eigenportfolios -- the eigenvectors of its 504-day
  covariance each date, as its test read them -- forecast by every setting's covariance against
  the one outcome they share. This separates the forecast's shape from each setting's choosing
  other directions.
  - **The second half:** their mean abs(bias - 1) goes from 0.222 to 0.168 at 63, the middle
    ranks coming down (rank 9 1.288 to 1.163).
  - **The first half:** they do not move (0.066 to 0.067 at every setting).
  - **The market direction** gains least. Rank 1 goes from 0.814 to 0.858 in the second half:
    the 63-day correlations take only 6% off its forecast volatility (the median ratio 0.942),
    where it would need about 19% to be right (1 - 0.814).
- **The correlations themselves.** The mean correlation between factors in the calm half is
  0.398 at 63 days against 0.432 at 504, and the market direction's share of the factor variance
  is 0.594 against 0.636. The correlations had fallen after the shock, and the long window lags
  that, but by little.
  - **What is left is on the volatility side:** the multiplier is one number over 25 factors,
    and in the calm half the market direction calmed further than the rest. Every other rank is
    under-forecast there (ranks 2-25 1.22).

**Does the optimiser pay for it in the small eigen-directions? Yes, in both halves.** Its
minimum-variance portfolios' forecast variance split along each setting's own eigen-directions,
the realised return along the same, each part's statistic (whole / first / second):

| | rank 1 | ranks 2-5 | ranks 6-15 | ranks 16-25 | specific |
|---|---|---|---|---|---|
| long-only at 504: share of the forecast variance | 41.6% | 13.2% | 8.4% | 20.6% | 16.2% |
| its statistic | 0.95 / 1.06 / 0.81 | 1.03 / 0.92 / 1.14 | 1.13 / 1.06 / 1.18 | 1.17 / 1.13 / 1.22 | 0.88 / 0.89 / 0.87 |
| long-only at 63: share | 37.9% | 13.7% | 9.1% | 23.8% | 15.6% |
| its statistic | 0.96 / 1.06 / 0.86 | 0.99 / 0.95 / 1.03 | 1.07 / 1.03 / 1.11 | 1.25 / 1.22 / 1.29 | 0.87 / 0.88 / 0.86 |
| unconstrained at 504: share | 35.4% | 14.1% | 10.2% | 23.4% | 16.9% |
| its statistic | 0.95 / 1.06 / 0.81 | 1.04 / 0.92 / 1.14 | 1.12 / 1.05 / 1.18 | 1.17 / 1.14 / 1.21 | 0.91 / 0.93 / 0.90 |
| unconstrained at 63: share | 31.1% | 13.3% | 11.1% | 28.2% | 16.3% |
| its statistic | 0.96 / 1.06 / 0.86 | 0.99 / 0.95 / 1.04 | 1.08 / 1.04 / 1.11 | 1.25 / 1.23 / 1.27 | 0.90 / 0.91 / 0.90 |

- **At 63 days the optimiser moves risk into the smallest eigen-directions,** 20.6% to 23.8% of
  its forecast (unconstrained 23.4% to 28.2%). Those parts realise more over their forecast,
  1.25 against 1.17, in both halves: the shorter window's smallest eigenvalues are too small, and
  the optimiser finds them. Its own ranks 21-25 read 1.303 against 1.195.
- **What it gains** is the market direction and the next ranks in the calm half: rank 1 0.86
  against 0.81, ranks 2-5 1.03 against 1.14, ranks 6-15 1.11 against 1.18.
- **Over the test the gain does not cover the price.** The unconstrained portfolio realises
  10.23% a year against 10.19%, so 63 days is out by the rule. The first half is worse throughout
  (12.10% and 12.18% against 11.88% and 11.95%).
- **The specific part reads 0.87-0.91 at every setting.** The realised return is split through
  the estimated factor returns -- their noise in the factor part, the regression's projection in
  the specific -- so this says where the risk sits more surely than how much; it does not move
  with the correlations.

**So, by the brief's item 4: a shorter half-life fixes part of the shape, and the optimiser pays
for it in the small eigen-directions.**

**The test that would tell -- proposed, not run:** the one-sided eigenvalue adjustment on the
63-day correlations.
- **The run:** one pass, as VE15 was:

  ```
  --correlation-half-life 63 --bias --bias-eigen-scales 1,2,3 --eigen-one-sided --bias-horizons 1,5,11,21 --bias-guard 0
  ```

  - the model's own test is then C16's 63-day setting, to the bit;
  - each scale's factor side is levelled again on its own forecasts' regime history;
  - the simulation re-estimates by the 63-day estimator, whose eigen bias is larger than the
    504-day one's;
  - about 15 minutes.
- **What would tell:** whether gamma >= 1 lifts the smallest eigenvalues enough that the
  optimiser stops loading on them. Judged by the same rule: both minimum-variance portfolios
  realising no more over the whole test than at 504 (10.13% and 10.19%), to the hundredth; then
  the criterion on the first half, against 504's 0.098. Read beside it: ranks 21-25, the
  optimiser's share and statistic in ranks 16-25, and whether the calm half keeps its 0.083.
- **What argues against it:** on the 504-day covariance the one-sided adjustment raised the
  optimiser's realised volatility at every a (VE15). It has more to correct on a noisier matrix,
  which is the case for trying it; that is all the evidence says.

---

## Which I would choose, and why

1. **The correlations stay on 504 days.** The rule written before the run keeps them: 63 days is
   out on the optimiser's realised volatility, and 126 and 252 are worse on the first half.
   Beyond the rule:
   - **The halves disagree:** each prefers the other end, monotonically -- the calm half the
     shortest window, the shock the longest -- so a fixed shorter half-life trades one regime
     for the other.
   - **What a shorter window buys is small:** 6% off the market direction's forecast volatility
     where it would need 19%. Its price is the optimiser's, in the small eigen-directions, in both
     halves.
2. **The calm half's shape is mostly not in the correlations.** It is in the volatilities: one
   multiplier over 25 factors, which the market direction and the rest pull apart in a calm
   market. The correlation half-life is not the lever. If it is pursued, the lever to test is on
   the volatility side -- the market direction read and scaled on its own -- but that is a new
   design, for Jeff to want or not, not something the measurements here can choose.
3. **Item 4's test: worth one run if Jeff wants the calm half pursued, not before.** On the
   evidence the eigenvalue adjustment has twice failed the optimiser's test on the 504-day
   covariance. If it fails on the 63-day one too, the correlation half-life is closed.
4. **Next, as ruling 5 orders it: the largest members' specific risk,** in the specific risk
   model, by size. The top quintile is still 1.028 under the new defaults. At the other end the
   optimiser's specific part reads 0.87-0.91, a hint that the same shrinkage over-forecasts the
   low-risk names it holds; that piece should read both ends.
5. **The plain run's cost:** about seven minutes for the factor side. It need not cost that. Each
   refit's factor returns are the full fit's through its date, to the bit, on all 686 days, so the
   plain run's history can be read from the full fit alone -- the model's own estimator on its
   factor returns' prefixes -- equal to the refits' to the bit in 0.3 s. I would build that for the
   plain run (decision 4); the bias test refits anyway.

---

## Key decisions

1. **S&P Global's release written by a script from the cache** (`release_spgi.py`): every number
   in the ruling's arithmetic recomputed from the cache's bars before anything was written, the
   quoted prices and press marked as Chat's search results. `held` removed and the mechanism
   kept -- the reader, the build's listing and refusal, the test -- with the hold's history in
   the evidence and `ruled`.
2. **The new panel named `_released`,** `_revised` kept and rebuildable from 355b772's rulings
   file (final checks).
3. **The factor side's default a named constant** (`REGIME_FACTOR_HALF_LIFE`), `none` to turn
   it off.
   - **Its cost reported:** a plain run now reports its refits' minutes.
   - **Its panel named:** the report now names its panel file, since this piece changes the
     default.
4. **The identity checked exactly,** the halves' means and counts in it, on three runs (D15,
   V15's F 21 and D16), so where it fails is measured, not supposed. It sits in the docs beside
   the statistic's definition.
5. **One pass by re-estimating each refit's covariance from its own factor returns,** a library
   function (`reestimated_covariance`) whose test says it is the fit with that estimator, to the
   bit; then checked against a run at 63 days alone, to the bit.
6. **Each correlation setting on its own forecasts' regime history,** not the model's: what a run
   at that half-life would compute, so the one pass equals it to the bit. The multiplier's
   agreement with the model's is measured and reported, not assumed -- it is to rounding, and
   the reason was found before the runs.
7. **The rule as the brief wrote it,** in the code before the run (`CORRELATION_RULE`): each
   realised volatility rounded to the hundredth, out if either exceeds 504's.
8. **The covariances and the optimiser's factor exposures kept in `--bias-out`,** so item 4 --
   the same directions under every setting, the optimiser's risk by eigen-direction -- was read
   from one run's arrays without another.
9. **The estimator's diagonal left as it is:** making it exact would move every stored run off
   the bit. Proposed instead (decision 3 below).
10. **The plain run's regime history from the full fit: checked, not built** (decision 4 below).
11. **No default changed beyond rulings 1 and 4:** the correlations stay on 504.

## Decisions left for Jeff

1. **The correlation half-life:** 504 days, as it is (my choice, the rule's), or a shorter one.
2. **Item 4's test:** the one-sided eigenvalue adjustment on the 63-day correlations, one pass of
   about 15 minutes, judged by the same rule -- run it (only if the calm half's shape is to be
   pursued; my choice), or not.
3. **The two-half-life estimator's diagonal:**
   - **set to the short window's variances,** so the volatilities and the regime multiplier no
     longer depend on the correlation half-life in the last bit -- every stored run then
     reproducing to rounding only;
   - **or left as it is** (my choice for now: the difference is 7e-16).
4. **The plain run's regime history from the full fit alone:** each refit's factor returns are the
   full fit's through its date, to the bit, on all 686 days, and the history read from the full fit
   alone -- the model's estimator on its factor returns' prefixes -- equals the refits' to the bit,
   the multiplier as of the last date 0.964025715108432 both ways, in 0.3 s against 8.3 minutes
   (`check_fast_regime.txt`). Build it for the plain run's factor side (my choice: a plain run back
   to under a minute, no number changed), or keep the refits. The specific side, off, would still
   need them, and the bias test refits anyway.
5. **The next piece:** the largest members' specific risk, by size (ruling 5's order, my
   choice), reading the optimiser's over-forecast specific part beside it; or the calm half's
   market direction, on the volatility side.

---

## Tests

**The suite:** 2,374 passed, 1 skipped (`final_checks.txt`), with `--ignore=tests/test_market_data.py`. Before this piece 2,372: 2
new, 1 changed, and every other passes unchanged -- the new parameters default to what was there.

**New, 2 (`test_bias.py`):**
- `test_a_covariance_reestimated_from_the_factor_returns_is_the_fit_with_that_estimator` -- the
  one-pass test's footing:
  - a model fitted with a two-half-life estimator, its covariance re-estimated by the same
    estimator with another correlation half-life, equals the fit with that estimator to the bit;
  - re-estimated by its own estimator, its own covariance;
  - a forecast with it is that fit's forecast -- covariance, exposures, specific risk, the repair;
  - the volatilities agree to the last bit (1e-15) and the correlations differ;
  - refused for a model with the sampling correction or the eigenfactor adjustment, and for a
    covariance of the wrong shape.
- `test_the_minimum_variance_portfolios_factor_exposures_are_kept` -- each date's long-only and
  unconstrained portfolios' factor exposures, its weights times the forecast's exposures.

**Changed, 1 (`test_securities.py`):** `test_a_held_ruling_is_read_with_its_reason_and_never_applied`
-- the held mechanism's own checks as before (a held ruling read with why, refused by the build and
the split check); the repo's file now reads no ruling held, SPGI's applied with the when-issued
close in its evidence.

**Measured, not unit-tested:**
- **The run's new paths:** five smoke runs; N16 V15's to the bit; C16's model test D16's; the
  one-pass test's 63-day setting a run's alone, to the bit.
- **The build:** `_released` against `_revised` and `_spinoffs`; the four panels rebuilt to the
  bit in the final checks.

**Hygiene:** every changed file parses, is ASCII and keeps its line endings
(`final_checks.txt`). **ruff:** clean over every changed Python file.

---

## Failures and debugging trail

- **The multiplier "to the bit" did not hold, and the reason was found before any run.**
  - **The check:** a random test of the estimator (`diag_check.py`). A million random c, and
    c / (sqrt(c) sqrt(c)) was exactly one for 51% of them. Fifty random 900-by-25 return
    histories, and the two-half-life matrix's diagonal at 63, 126 and 252 differed from 504's
    in 2,324 of 3,750 entries.
  - **What it means:** the multiplier read from the volatilities agrees across correlation
    half-lives to rounding, not to the bit.
  - **The design followed from it:** each setting reads its own history, as a run at that
    half-life alone would. The report measures the agreement rather than assuming it.
- **The harness blocks a foreground `sleep`.** The runs were waited on with the Monitor tool on
  the runner's log and background `until grep` loops; a state-file working note now says so.
- **The first smoke run's equality came before the optimiser's exposures were kept.** Both smoke
  runs were made again on the final code, and the exposures are among the arrays equal to the bit.
- **The report's first cut:**
  - the realised-volatility table misaligned a five-character value -- widened after the first
    smoke run;
  - the report never named its panel file, which this piece changes -- a line added (the path,
    and "DEFAULT_PANEL" when it is the default);
  - a `{rows and dates[rows[0]]}` slip in the plain run's timing line -- replaced before any run.
- **ruff:** one long line in the run script, fixed. The scratch patch scripts carry long string
  literals of the text they replace, left as they are.
- **The rulings' names.** The code, docs and test first called this brief's rulings
  "midoffice16's ruling k".
  - **The two usages:** the state file's convention -- and the code's docstrings, "midoffice11's
    ruling 2" for the correlation half-life -- names a ruling by the piece whose decisions it
    rules, so these are midoffice15's. midoffice15's own docs and rulings file had already used
    the other reading.
  - **The fix:** the code, docs and test now say "ruling k of midoffice16's brief"
    (`fix_rulings_naming.py`), which neither reading can mistake. The state file keeps "15 ruling
    k", and the rulings file keeps its own usage, with the date; a working note records it.
  - **Its timing:** the edit touched strings and comments only, and came during the final checks
    -- after the four rebuilds and the capture's start, before the suite, ruff and hygiene, which
    ran on it.
- **Two phrasings tightened** after a reread (`fix_wording.py`): what the 63-day correlations take
  off the market direction's forecast against what it would need, 1 - 0.814; and the same
  directions' first half, 0.066 to 0.067.
- **The runs:** none failed. N16 8.3 minutes, D16 11.1, C16 9.5, P16x 0.8, P16 7.5
  (`bias/runner_log.txt`), each started with over 4.3 GB available.

## Problems found, not fixed

1. **The calm half's market direction is still over-forecast:**
   - under the new defaults: rank 1 at 0.814 in the second half, the index at 0.785;
   - at a 63-day correlation half-life: 0.858.

   What is left is on the volatility side, which no correlation half-life reaches.
2. **A shorter correlation window's smallest eigenvalues are too small,** and the optimiser finds
   them: ranks 21-25 1.30 at 63 days, the optimiser's share in ranks 16-25 up by 3 to 5 points.
3. **The two-half-life covariance's diagonal depends on the correlation half-life in the last
   bit** (7e-16), and so does the regime multiplier (2.2e-16).
4. **A plain run takes about 7.5 minutes,** where it took under one, for the factor side's
   refits.
5. **The optimiser's specific part reads 0.87-0.91,** over-forecast at every setting -- read
   through the estimated factor returns, so its size is uncertain.
6. **The largest members' specific risk is still forecast too low** (the top quintile 1.028), as
   ruled for the next piece.
7. **As before:**
   - FedEx Freight and Vylor still lack Momentum;
   - six ruff findings in untouched files at HEAD;
   - `tests/test_market_data.py`'s `../core_repo/src`.

## Scratch

`claude/scratch/midoffice16/` (gitignored):

- **The rulings:**
  - `release_spgi.py` and `.txt` -- the writer of SPGI's release;
  - `rms_identity.py`, `rms_identity.txt` and `rms_identity_D16.txt` -- ruling 1's identity;
  - `patch_build.py`, `patch_test_securities.py`, `patch_bias_exposure.py`, `patch_state_*.py`,
    `fill_fast.py`, `fix_wording.py` and `fix_rulings_naming.py` -- the patches made by script.
- **The panels:** `panels/`, the released panel built into scratch, its build outputs (into
  scratch and into `data/`); `panel_changed.txt`.
- **The model:** `model/`, M16, its report and pickle; `model_compare.txt`.
- **The runs:**
  - `runner.py`;
  - `bias/`: N16, D16 and C16, each a report, progress and `.npz`; P16x and P16, each a report
    and progress; the runner's log;
  - `smoke/`: the five smoke runs -- the one-pass test at 63 and 126 days and a run at 63 alone,
    each twice, and a plain run -- `equal_c63.txt`, `check_fast_regime_smoke.txt`;
  - `check_fast_regime.py` and its full run's output.
- **The analyses:**
  - `equal_correlation.py`; `equal_N16_V15.txt`, `equal_C16_D16.txt`;
  - `compare_bias.py` and `.txt` -- D16 against N16;
  - `analyse_correlation.py` and `.txt`;
  - `diag_check.py` and `.txt`;
  - `plain_cost.txt`.
- **The checks:** `final_checks.sh` and `.txt`, `final/`; the retro's `append_appendix.py`.

The midoffice13 to 15 harnesses were used where they are: `capture.py`, `compare.py`,
`hygiene.py` (13), `compare_panels.py`, `common.py` (14), `smoke_last.py`, `equal_setting.py`
(15). The key outputs are copied into the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

## Open items for Chat

- **The five decisions above,** the correlation half-life first.
- **The next piece,** as I would order it: the largest members' specific risk, by size, with the
  optimiser's specific part beside it; the calm half's market direction after, if wanted.
- **The bias statistics are standing,** the correlation report among them under
  `--bias-correlation-half-lives`; every candidate is one run against D16, through the runner.

---

## Appendix: outputs, verbatim

Each output as its script wrote it, from `claude/scratch/midoffice16/`.

### release_spgi.txt -- ruling 4: S&P Global's release, the arithmetic recomputed against the cache's bars, and what was written into the ruling

```text
The cache's prices (raw bars):
  SPGI close 2026-06-30 407.26; 2026-07-01 open 399.58, close 414.97; 2026-07-02 low 422.59, high 440.01, close 439.89
  MBGL 2026-07-01 open 20.30, close 21.19
  the peers' 2026-07-01 close-to-close: MCO +3.41%, MSCI +3.93%, ICE +2.94%, CME +4.61%, NDAQ +4.96%, FDS +6.72%; median +4.27%

Quoted in the ruling from Chat's search results of 2026-11-13: SPGI WI's close of 2026-06-30 384.00; Kalkine 436.69, +5.23% on 07-02 and +7.73% on 07-01; Motley Fool almost +8%, Mobility almost -4%

The ruling's arithmetic, each line recomputed:
  407.26 less 384.00, the value the WI line takes out                             23.2600  stated   23.2600  ok
  a regular-way close near 428 less 384.00: the Mobility it would need            44.0000  stated   44.0000  ok
  Mobility's first open                                                           20.3000  stated   20.3000  ok
  SPGI's close of 07-01                                                          414.9700  stated  414.9700  ok
  SPGI's close over its when-issued close - 1, %                                   8.0651  stated    8.0700  ok
  its peers' median that day, %                                                    4.2659  stated    4.2700  ok
  436.69 / 1.0523                                                                414.9862  stated  414.9900  ok
    5.23% rounded puts the close it implies between 414.97 and 415.01: Polygon's 414.97 is inside; Kalkine's 436.69 is +5.234% over Polygon's close, inside 07-02's range 422.59 to 440.01 (its close 439.89), so an intraday price
  414.99 / 1.0773, the base of Wednesday's 7.73%                                 385.2095  stated  385.2100  ok
    on Polygon's 414.97 it is 385.19; 7.73% rounded puts it between 385.18 and 385.21
  407.26 less 385.21, Mobility on that base                                       22.0500  stated   22.0500  ok
  Mobility's first close, 21.19, against 22.05 - 1, %                             -3.9002  stated   -3.9000  ok
  the ruling's 07-01 total return, (414.97 + 21.19) / 407.26 - 1, %                7.0962  stated    7.0962  ok
    and Polygon's own that day, 414.97 / 407.26 - 1: +1.8931%
    Motley Fool's 'almost 8%' is +8.07% over the WI close; its Mobility 'almost 4%' is -3.90% against 22.05
  every line checks

The ruling's evidence, appended:
  The hold, midoffice15's: SPGI's raw prices show no fall of Mobility's value on the record date or on 07-01 -- that day SPGI opened -1.89% and closed +1.89% against its peers' +0.74% and +4.27%, and over 07-01 and 07-02 its raw return, +8.01%, is its peers' (+7.14% to +8.70%) without the distribution -- held until an outside price for SPGI's regular-way close of 2026-06-30 settled it. Released by midoffice16's ruling 4 (Jeff, 2026-11-13), on that price: Polygon's 407.26 is the regular-way close of 2026-06-30. The ex-distribution line, SPGI WI, closed at 384.00 that day, and 407.26 less 384.00 is 23.26, one Mobility share at about its when-issued price; a regular-way close near 428 would need Mobility near 44, and it opened at 20.30. The gap was there, and 07-01's rally filled it: SPGI closed at 414.97, 8.07% over its when-issued close, its peers' median +4.27%. The press read the day the same way. Motley Fool: "Why S&P Global Stock Zoomed Almost 8% Higher Today"; Mobility's shares "sank by almost 4% on their first day of trading". Kalkine: "SPGI shares rose 5.23% today to $436.69, building on Wednesday's 7.73% gain". 436.69 / 1.0523 is 414.99, Polygon's close of 07-01 (414.97) within the rounding of 5.23% (414.97 to 415.01) -- 436.69 an intraday price of 07-02, which closed at 439.89; 414.99 / 1.0773 is 385.21, which is 407.26 less 22.05 of Mobility, and Mobility's first close, 21.19, is 3.9% under that. So the ruling's 07-01 total return, (414.97 + 21.19) / 407.26 - 1 = +7.0962%, is right, and SPGI's +3.8 sd under it that day is the market's. SPGI WI's close and the press are as quoted from Chat's search results of 2026-11-13, the pages not opened: https://us.trendlyne.com/us/equity/Split/SPGI.WI/3563800/s-p-global-inc-when-issued-june-2026-split/; https://www.dividend.com/stocks/communications/media/publishing-broadcasting/spgi_wi-sandp-global-inc-when-issued-june-2026/; https://nasdaq.com/articles/why-sp-global-stock-zoomed-almost-8-higher-today; https://kalkine.com/news/financial/sp-global-nysespgi-surges-5-today-following-completion-of-mobility-global-spinoff. The arithmetic was checked against the cache's bars (claude/retros/RETRO_midoffice16_correlation_half_life.md).

The ruling's 'ruled':
  midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review; held by midoffice15's ruling 1 (Jeff, 2026-11-12); the hold released by midoffice16's ruling 4 (Jeff, 2026-11-13)

written: config\sp500_spin_off_overrides.json; no ruling is held
```

### panel_changed.txt -- the released panel, `_released`, against `_revised` and against `_spinoffs`: every changed member-day

```text
data/external/polygon/sp500_securities_2022-01-03_2026-11-03_revised.npz
  against claude/scratch/midoffice16/panels/sp500_securities_2022-01-03_2026-11-03_released.npz
  adjusted_volumes         equal to the bit (1214, 592)
  dates                    equal to the bit (1214,)
  highs                    equal to the bit (1214, 592)
  labels                   equal to the bit (592,)
  lows                     equal to the bit (1214, 592)
  market_returns           equal to the bit (1214,)
  market_total_returns     equal to the bit (1214,)
  membership               equal to the bit (1214, 592)
  names                    equal to the bit (592,)
  plan                     equal to the bit ()
  prices                   equal to the bit (1214, 592)
  securities               equal to the bit (592,)
  shares_outstanding       equal to the bit (1214, 592)
  sic_codes                equal to the bit (592,)
  sic_dates                equal to the bit (592,)
  snapshot                 equal to the bit ()
  tickers                  equal to the bit (1214, 592)
  total_return_prices      DIFFERS: 88 cells, 88 member-days
  unadjusted_prices        equal to the bit (1214, 592)
  volumes                  equal to the bit (1214, 592)

total_return_prices, by daily return: 1 security-days changed beyond 1e-12, 1 of them member-days; 36 more differ in the last bits
  2026-07-01 SPGI   member: +1.8931% -> +7.0962% (+5.20 points)

prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

shares_outstanding: 0 security-days changed, 0 member-days

==========

data/external/polygon/sp500_securities_2022-01-03_2026-11-03_spinoffs.npz
  against claude/scratch/midoffice16/panels/sp500_securities_2022-01-03_2026-11-03_released.npz
  adjusted_volumes         equal to the bit (1214, 592)
  dates                    equal to the bit (1214,)
  highs                    equal to the bit (1214, 592)
  labels                   equal to the bit (592,)
  lows                     equal to the bit (1214, 592)
  market_returns           equal to the bit (1214,)
  market_total_returns     equal to the bit (1214,)
  membership               equal to the bit (1214, 592)
  names                    equal to the bit (592,)
  plan                     equal to the bit ()
  prices                   equal to the bit (1214, 592)
  securities               equal to the bit (592,)
  shares_outstanding       DIFFERS: 109 cells, 109 member-days
  sic_codes                equal to the bit (592,)
  sic_dates                equal to the bit (592,)
  snapshot                 equal to the bit ()
  tickers                  equal to the bit (1214, 592)
  total_return_prices      equal to the bit (1214, 592)
  unadjusted_prices        equal to the bit (1214, 592)
  volumes                  equal to the bit (1214, 592)

total_return_prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

shares_outstanding: 109 security-days changed, 109 member-days
  FDXF   2026-06-01..2026-11-03 (109 days, 109 member-days): 119,303,378 -> 148,943,044, ... 119,303,378 -> 148,943,044
```

### panels/build_released.txt -- the build's report: its spin-off section

```text
Spin-offs: distributions of shares the cache records wrongly or not at all, config/sp500_spin_off_overrides.json: 19 rulings, 0 held, 19 applied -- on each ex-date the parent's total return (w (C_t + D_t) + k V) / C_s - 1, the child at its close; the stitched prices as Polygon has them
  APTV   2026-04-01 -> VGNT  0.333333 a share: 69.44 on 2026-03-31 to 62.09; VGNT 27.85 on 2026-04-01 (Versigent PLC): total return +2.7842%, the adjusted closes' -10.5847%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review; kept by midoffice15's ruling 1 (Jeff, 2026-11-12)
  BDX    2022-04-01 -> EMBC  0.2 a share: 266.00 on 2022-03-31 to 262.50; EMBC 30.50 on 2022-04-01 (Embecta Corp. Common Stock, BBG014L7D4G1): total return +0.9774%, the adjusted closes' -1.3158%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  BDX    2026-02-10 -> WAT   0.135 a share: 207.39 on 2026-02-09 to 171.68; WAT 327.02 on 2026-02-10 (no record that day): total return +4.0685%, the adjusted closes' -17.2188%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review; kept by midoffice15's ruling 1 (Jeff, 2026-11-12)
  BWA    2023-07-05 -> PHIN  0.2 a share: 49.92 on 2023-07-03 to 43.18; PHIN 36.75 on 2023-07-05 (PHINIA Inc., BBG01F6N6NM7): total return +1.2220%, the adjusted closes' -13.5016%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  CMCSA  2026-01-05 -> VSNT  0.04 a share: 29.54 on 2026-01-02 to 28.13; VSNT 40.57 on 2026-01-05 (Versant Media Group, Inc. Class A Common Stock, BBG01Y0M8K11): total return +0.7204%, the adjusted closes' -4.7732%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  CTVA   2026-10-01 -> VYLR  1 a share: 77.65 on 2026-09-30 to 12.57; VYLR 68.26 on 2026-10-01 (Vylor Inc., BBG024W5VJC3): total return +4.0953%, the adjusted closes' -83.8120%; midoffice14 brief (Jeff, 2026-11-12), item 1
  DHR    2023-10-02 -> VLTO  0.333333 a share: 248.10 on 2023-09-29 to 213.74; VLTO 85.12 on 2023-10-02 (Veralto Corporation, BBG019Q32XJ9): total return -2.4130%, the adjusted closes' -13.8493%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  EXC    2022-02-02 -> CEG   0.333333 a share: 57.83 on 2022-02-01 to 42.86; CEG 53.01 on 2022-02-02 (Constellation Energy Corporation Common Stock, BBG014KFRNP7): total return +4.6689%, the adjusted closes' -25.8862%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  FBIN   2022-12-15 -> MBC   1 a share: 61.90 on 2022-12-14 to 57.62; MBC 7.70 on 2022-12-15 (MasterBrand, Inc., BBG01B9G42Y3): total return +5.5250%, the adjusted closes' -6.9144%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  FDX    2026-06-01 -> FDXF  0.5 a share: 411.75 on 2026-05-29 to 338.49; FDXF 149.53 on 2026-06-01 (FedEx Freight Holding Company, Inc., BBG01ZPB91C4): total return +0.3655%, the adjusted closes' -17.7923%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; FDXF's count as Vylor's; grossed up by FedEx's retained share, midoffice15's ruling 3 (Jeff, 2026-11-12)
  HON    2026-06-29 -> HONA  0.5 a share, the parent's split 0.5 with it: 232.21 on 2026-06-26 to 227.80, w 0.5; HONA 220.19 on 2026-06-29 (Honeywell Aerospace Inc. Common Stock, BBG020QYV152): total return -3.5377%, the adjusted closes' -1.8991%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review; it supersedes midoffice08's decision 3 on HON's count; kept by midoffice15's rulings 1 and 2 (Jeff, 2026-11-12)
  J      2024-09-30 -> AMTM  1 a share: 155.53 on 2024-09-27 to 130.90; AMTM 32.25 on 2024-09-30 (Amentum Holdings, Inc., BBG01NY7B734): total return +4.8994%, the adjusted closes' -15.8362%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  J      2025-05-16 -> AMTM  0.060835 a share: 128.80 on 2025-05-15 to 129.17; AMTM 21.84 on 2025-05-16 (no record that day): total return +1.3188%, the adjusted closes' +0.2873%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review; kept by midoffice15's ruling 1 (Jeff, 2026-11-12)
  K      2023-10-02 -> KLG   0.25 a share: 59.51 on 2023-09-29 to 52.50; KLG 13.35 on 2023-10-02 (WK Kellogg Co, BBG018CYVVW5): total return -6.1712%, the adjusted closes' -11.7795%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  LEN    2025-01-21 -> MRP   0.5 a share: 141.54 on 2025-01-17 to 133.14; MRP 26.74 on 2025-02-07, its first close, none on the ex-date (Millrose Properties, Inc., BBG01RQYH2X7): total return +3.5114%, the adjusted closes' -2.8306%; midoffice14 brief (Jeff, 2026-11-12), item 2: LEN's adjustment of 2025-01-21, checked -- carried wrongly, the date right and the value a third of the distribution's; carried by item 1's rule at Millrose's first close; kept by midoffice15's ruling 4 (Jeff, 2026-11-12)
  SPGI   2026-07-01 -> MBGL  1 a share: 407.26 on 2026-06-30 to 414.97; MBGL 21.19 on 2026-07-01 (Mobility Global Inc., BBG022G54330): total return +7.0962%, the adjusted closes' +1.8931%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review; held by midoffice15's ruling 1 (Jeff, 2026-11-12); the hold released by midoffice16's ruling 4 (Jeff, 2026-11-13)
  T      2022-04-11 -> WBD   0.241917 a share: 24.14 on 2022-04-08 to 19.63; WBD 24.78 on 2022-04-11 (Warner Bros. Discovery, Inc. Series A Common Stock, BBG011386VF4): total return +6.1504%, the adjusted closes' -18.6827%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  WDC    2025-02-24 -> SNDK  0.333333 a share: 68.70 on 2025-02-21 to 49.02; SNDK 48.60 on 2025-02-24 (Sandisk Corporation Common Stock, BBG01R388JG1): total return -5.0724%, the adjusted closes' -28.6515%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  ZBH    2022-03-01 -> ZIMV  0.1 a share: 127.19 on 2022-02-28 to 124.14; ZIMV 25.53 on 2022-03-01 (ZimVie Inc. Common Stock, BBG014630587): total return -0.3908%, the adjusted closes' -2.3980%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  a split the cache does not record, applied to the count: HON 2-to-1 on 2026-06-29 (0.5, whole numbers): counts 635,675,701 on 2026-04-13, 316,940,010 on 2026-10-12, 0.499 times; raw price no raw price on a side; a split, the counts follow it; a spin-off ruling's split the cache does not record, its price not tested -- the distribution the same day offsets the gap
  FDXF   BBG01ZPB91C4: no count of its own -- FDX's 238,606,756 on the record date 2026-05-15 times 0.5, 119,303,378 distributed, over the 0.801 FDX did not keep, 148,943,044, from 2026-06-01 to the window's end, Polygon carrying none: 109 days, 109 member-days
  VYLR   BBG024CQR6X9: no count of its own -- CTVA's 668,820,000 on the record date 2026-09-24 times 1, 668,820,000, from 2026-10-01 to the window's end, Polygon carrying none: 24 days, 24 member-days
  a count override superseded by HON's ruling of 2026-06-29, not read: HON BBG000H556T9 2026-10-12..2026-11-03 (midoffice08, decision 3 (Jeff, 2026-11-10))
```

### diag_check.txt -- item 2: the two-half-life covariance's diagonal against the correlation half-life, on random data, before any run

```text
c/(sqrt(c)*sqrt(c)) == 1 exactly: 0.51161 values seen: [1. 1. 1. 1.]
diagonal entries differing from 504's: 2324 of 3750
```

### rms_identity.txt -- ruling 1: the whole test's statistic against the root mean square of its halves', every family at one day, on D15 and on V15's F 21, S off

```text
claude/scratch/midoffice15/bias/D15.npz, D15, the model (midoffice15's defaults): 686 forecast dates, 2024-02-08 to 2026-11-02; the halves 343 and 343 dates, as the run's report cuts them

The broad portfolios: 4 portfolios
  b against the root mean square of its halves: |gap| median 0.0007, largest 0.0007; within 0.0005 0%, within 0.005 100%
  the largest, the index, cap-weighted: b 1.0148 of 1.2244 and 0.7507 (n 343 and 343), root mean square 1.0155, gap -0.0007 -- the halves' weights -0.0007, the halves' means (+0.069 and +0.064) +0.0000
  over the family, |part| median: the weights 0.0007, the means 0.0000; the identity with both exact to 2.2e-16
    the index, cap-weighted          1.015 of 1.224 and 0.751: root mean square 1.016
    the index, equally weighted      0.966 of 1.143 and 0.749: root mean square 0.966
    smallest 85, equally weighted    0.956 of 1.065 and 0.834: root mean square 0.957
    largest 50, equally weighted     0.996 of 1.175 and 0.778: root mean square 0.997

The minimum-variance portfolios: 2 portfolios
  b against the root mean square of its halves: |gap| median 0.0008, largest 0.0008; within 0.0005 0%, within 0.005 100%
  the largest, minimum variance, unconstrained: b 1.1013 of 1.2984 and 0.8623 (n 343 and 343), root mean square 1.1021, gap -0.0008 -- the halves' weights -0.0008, the halves' means (+0.081 and +0.084) +0.0000
  over the family, |part| median: the weights 0.0008, the means 0.0000; the identity with both exact to 2.0e-16
    minimum variance, long-only      1.059 of 1.247 and 0.831: root mean square 1.060
    minimum variance, unconstrained  1.101 of 1.298 and 0.862: root mean square 1.102

The industries: 20 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0009; within 0.0005 40%, within 0.005 100%
  the largest, industry: Finance & Real Estate: b 0.9954 of 1.0784 and 0.9029 (n 343 and 343), root mean square 0.9945, gap +0.0009 -- the halves' weights -0.0007, the halves' means (+0.096 and -0.016) +0.0016
  over the family, |part| median: the weights 0.0007, the means 0.0003; the identity with both exact to 3.9e-16
  its summary, the mean abs(b - 1): 0.0460 whole, 0.0942 and 0.1100 by half, whose root mean square is 0.1024

The random portfolios, long-only: 100 portfolios
  b against the root mean square of its halves: |gap| median 0.0007, largest 0.0008; within 0.0005 26%, within 0.005 100%
  the largest, random long-only 92: b 1.0920 of 1.2653 and 0.8875 (n 343 and 343), root mean square 1.0928, gap -0.0008 -- the halves' weights -0.0008, the halves' means (+0.063 and +0.062) +0.0000
  over the family, |part| median: the weights 0.0007, the means 0.0000; the identity with both exact to 6.4e-16
  its summary, the median: 0.9709 whole, 1.1239 and 0.7978 by half, whose root mean square is 0.9746

The random portfolios, long-short: 100 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0047; within 0.0005 32%, within 0.005 100%
  the largest, random long-short 52: b 0.9543 of 0.9407 and 0.9584 (n 343 and 343), root mean square 0.9496, gap +0.0047 -- the halves' weights -0.0007, the halves' means (-0.096 and +0.107) +0.0054
  over the family, |part| median: the weights 0.0007, the means 0.0003; the identity with both exact to 5.9e-16
  its summary, the median: 1.0040 whole, 1.0005 and 1.0032 by half, whose root mean square is 1.0019

The eigenportfolios: 25 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0056; within 0.0005 36%, within 0.005 96%
  the largest, rank 20: b 1.2214 of 1.1667 and 1.2630 (n 343 and 343), root mean square 1.2158, gap +0.0056 -- the halves' weights -0.0009, the halves' means (+0.110 and -0.140) +0.0064
  over the family, |part| median: the weights 0.0008, the means 0.0003; the identity with both exact to 6.1e-16
  its summary, ranks 1-5, the mean: 1.0299 whole, 1.0651 and 0.9855 by half, whose root mean square is 1.0261
    rank 1                           1.027 of 1.226 and 0.780: root mean square 1.028
    rank 2                           0.974 of 0.958 and 0.991: root mean square 0.975
    rank 3                           1.046 of 1.020 and 1.073: root mean square 1.047

Every member singly (60 forecasts or more): 504 portfolios (39 with a half under two outcomes left out)
  b against the root mean square of its halves: |gap| median 0.0006, largest 1.0170; within 0.0005 38%, within 0.005 92%
  the largest, member JNPR: b 0.5446 of 0.4799 and 2.1557 (n 343 and 5), root mean square 1.5616, gap -1.0170 -- the halves' weights -1.0319, the halves' means (+0.005 and +1.066) +0.0149
  over the family, |part| median: the weights 0.0007, the means 0.0003; the identity with both exact to 7.7e-16
  its summary, the median: 1.0010 whole, 1.0383 and 0.9601 by half, whose root mean square is 1.0000
    halves within 10% of each other's count: 459, |gap| median 0.0006, largest 0.0154; the weights' part median 0.0007
    halves of unequal counts: 45, |gap| median 0.0164, largest 1.0170; the weights' part median 0.0194

==========

claude/scratch/midoffice15/bias/V15.npz, V15's F 21, S off -- the new defaults on _revised: 686 forecast dates, 2024-02-08 to 2026-11-02; the halves 343 and 343 dates, as the run's report cuts them

The broad portfolios: 4 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0007; within 0.0005 25%, within 0.005 100%
  the largest, the index, cap-weighted: b 0.9336 of 1.0630 and 0.7848 (n 343 and 343), root mean square 0.9343, gap -0.0007 -- the halves' weights -0.0007, the halves' means (+0.056 and +0.061) +0.0000
  over the family, |part| median: the weights 0.0007, the means 0.0001; the identity with both exact to 3.9e-16
    the index, cap-weighted          0.934 of 1.063 and 0.785: root mean square 0.934
    the index, equally weighted      0.893 of 0.983 and 0.793: root mean square 0.893
    smallest 85, equally weighted    0.902 of 0.921 and 0.882: root mean square 0.902
    largest 50, equally weighted     0.920 of 1.024 and 0.804: root mean square 0.921

The minimum-variance portfolios: 2 portfolios
  b against the root mean square of its halves: |gap| median 0.0007, largest 0.0008; within 0.0005 0%, within 0.005 100%
  the largest, minimum variance, unconstrained: b 1.0332 of 1.1628 and 0.8866 (n 343 and 343), root mean square 1.0340, gap -0.0008 -- the halves' weights -0.0008, the halves' means (+0.083 and +0.088) +0.0000
  over the family, |part| median: the weights 0.0007, the means 0.0000; the identity with both exact to 3.4e-16
    minimum variance, long-only      0.990 of 1.109 and 0.855: root mean square 0.990
    minimum variance, unconstrained  1.033 of 1.163 and 0.887: root mean square 1.034

The industries: 20 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0011; within 0.0005 40%, within 0.005 100%
  the largest, industry: SIC28: b 1.0129 of 0.9814 and 1.0413 (n 343 and 343), root mean square 1.0118, gap +0.0011 -- the halves' weights -0.0007, the halves' means (-0.000 and +0.121) +0.0018
  over the family, |part| median: the weights 0.0007, the means 0.0002; the identity with both exact to 2.7e-16
  its summary, the mean abs(b - 1): 0.0551 whole, 0.0647 and 0.0885 by half, whose root mean square is 0.0775

The random portfolios, long-only: 100 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0008; within 0.0005 33%, within 0.005 100%
  the largest, random long-only 82: b 0.9446 of 0.9822 and 0.9038 (n 343 and 343), root mean square 0.9438, gap +0.0008 -- the halves' weights -0.0007, the halves' means (-0.006 and +0.100) +0.0015
  over the family, |part| median: the weights 0.0007, the means 0.0001; the identity with both exact to 4.3e-16
  its summary, the median: 0.9137 whole, 0.9834 and 0.8376 by half, whose root mean square is 0.9134

The random portfolios, long-short: 100 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0047; within 0.0005 33%, within 0.005 100%
  the largest, random long-short 52: b 0.9465 of 0.9175 and 0.9655 (n 343 and 343), root mean square 0.9418, gap +0.0047 -- the halves' weights -0.0007, the halves' means (-0.094 and +0.108) +0.0054
  over the family, |part| median: the weights 0.0007, the means 0.0003; the identity with both exact to 4.4e-16
  its summary, the median: 0.9972 whole, 0.9745 and 1.0154 by half, whose root mean square is 0.9952

The eigenportfolios: 25 portfolios
  b against the root mean square of its halves: |gap| median 0.0007, largest 0.0056; within 0.0005 36%, within 0.005 96%
  the largest, rank 20: b 1.2384 of 1.0883 and 1.3620 (n 343 and 343), root mean square 1.2328, gap +0.0056 -- the halves' weights -0.0009, the halves' means (+0.098 and -0.156) +0.0065
  over the family, |part| median: the weights 0.0008, the means 0.0003; the identity with both exact to 5.7e-16
  its summary, ranks 1-5, the mean: 1.0108 whole, 0.9852 and 1.0307 by half, whose root mean square is 1.0082
    rank 1                           0.946 of 1.064 and 0.814: root mean square 0.947
    rank 2                           0.978 of 0.912 and 1.040: root mean square 0.978
    rank 3                           1.051 of 0.970 and 1.127: root mean square 1.052

Every member singly (60 forecasts or more): 504 portfolios (39 with a half under two outcomes left out)
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.9901; within 0.0005 39%, within 0.005 92%
  the largest, member JNPR: b 0.5168 of 0.4529 and 2.0824 (n 343 and 5), root mean square 1.5069, gap -0.9901 -- the halves' weights -1.0047, the halves' means (+0.004 and +1.029) +0.0147
  over the family, |part| median: the weights 0.0007, the means 0.0003; the identity with both exact to 6.3e-16
  its summary, the median: 0.9843 whole, 0.9785 and 0.9792 by half, whose root mean square is 0.9789
    halves within 10% of each other's count: 459, |gap| median 0.0006, largest 0.0115; the weights' part median 0.0007
    halves of unequal counts: 45, |gap| median 0.0209, largest 0.9901; the weights' part median 0.0194
```

### rms_identity_D16.txt -- the same on D16, the new defaults on the new panel

```text
claude/scratch/midoffice16/bias/D16.npz, D16, the new defaults on _released: 686 forecast dates, 2024-02-08 to 2026-11-02; the halves 343 and 343 dates, as the run's report cuts them

The broad portfolios: 4 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0007; within 0.0005 25%, within 0.005 100%
  the largest, the index, cap-weighted: b 0.9336 of 1.0630 and 0.7848 (n 343 and 343), root mean square 0.9343, gap -0.0007 -- the halves' weights -0.0007, the halves' means (+0.056 and +0.061) +0.0000
  over the family, |part| median: the weights 0.0007, the means 0.0001; the identity with both exact to 1.4e-16
    the index, cap-weighted          0.934 of 1.063 and 0.785: root mean square 0.934
    the index, equally weighted      0.893 of 0.983 and 0.793: root mean square 0.893
    smallest 85, equally weighted    0.902 of 0.921 and 0.882: root mean square 0.902
    largest 50, equally weighted     0.920 of 1.024 and 0.804: root mean square 0.921

The minimum-variance portfolios: 2 portfolios
  b against the root mean square of its halves: |gap| median 0.0007, largest 0.0008; within 0.0005 0%, within 0.005 100%
  the largest, minimum variance, unconstrained: b 1.0332 of 1.1628 and 0.8865 (n 343 and 343), root mean square 1.0339, gap -0.0008 -- the halves' weights -0.0008, the halves' means (+0.083 and +0.088) +0.0000
  over the family, |part| median: the weights 0.0007, the means 0.0000; the identity with both exact to 5.7e-16
    minimum variance, long-only      0.990 of 1.109 and 0.855: root mean square 0.990
    minimum variance, unconstrained  1.033 of 1.163 and 0.886: root mean square 1.034

The industries: 20 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0011; within 0.0005 40%, within 0.005 100%
  the largest, industry: SIC28: b 1.0129 of 0.9814 and 1.0413 (n 343 and 343), root mean square 1.0118, gap +0.0011 -- the halves' weights -0.0007, the halves' means (-0.000 and +0.121) +0.0018
  over the family, |part| median: the weights 0.0007, the means 0.0002; the identity with both exact to 4.2e-16
  its summary, the mean abs(b - 1): 0.0551 whole, 0.0647 and 0.0885 by half, whose root mean square is 0.0775

The random portfolios, long-only: 100 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0008; within 0.0005 33%, within 0.005 100%
  the largest, random long-only 82: b 0.9446 of 0.9822 and 0.9038 (n 343 and 343), root mean square 0.9438, gap +0.0008 -- the halves' weights -0.0007, the halves' means (-0.006 and +0.100) +0.0015
  over the family, |part| median: the weights 0.0007, the means 0.0001; the identity with both exact to 6.2e-16
  its summary, the median: 0.9137 whole, 0.9834 and 0.8376 by half, whose root mean square is 0.9134

The random portfolios, long-short: 100 portfolios
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.0047; within 0.0005 32%, within 0.005 100%
  the largest, random long-short 52: b 0.9465 of 0.9175 and 0.9654 (n 343 and 343), root mean square 0.9418, gap +0.0047 -- the halves' weights -0.0007, the halves' means (-0.094 and +0.108) +0.0054
  over the family, |part| median: the weights 0.0007, the means 0.0003; the identity with both exact to 6.8e-16
  its summary, the median: 0.9972 whole, 0.9745 and 1.0154 by half, whose root mean square is 0.9951

The eigenportfolios: 25 portfolios
  b against the root mean square of its halves: |gap| median 0.0007, largest 0.0056; within 0.0005 36%, within 0.005 96%
  the largest, rank 20: b 1.2382 of 1.0883 and 1.3617 (n 343 and 343), root mean square 1.2326, gap +0.0056 -- the halves' weights -0.0009, the halves' means (+0.098 and -0.156) +0.0065
  over the family, |part| median: the weights 0.0008, the means 0.0002; the identity with both exact to 4.4e-16
  its summary, ranks 1-5, the mean: 1.0108 whole, 0.9852 and 1.0308 by half, whose root mean square is 1.0082
    rank 1                           0.946 of 1.064 and 0.814: root mean square 0.947
    rank 2                           0.978 of 0.912 and 1.040: root mean square 0.978
    rank 3                           1.051 of 0.970 and 1.127: root mean square 1.051

Every member singly (60 forecasts or more): 504 portfolios (39 with a half under two outcomes left out)
  b against the root mean square of its halves: |gap| median 0.0006, largest 0.9901; within 0.0005 39%, within 0.005 92%
  the largest, member JNPR: b 0.5168 of 0.4529 and 2.0824 (n 343 and 5), root mean square 1.5069, gap -0.9901 -- the halves' weights -1.0047, the halves' means (+0.004 and +1.029) +0.0147
  over the family, |part| median: the weights 0.0007, the means 0.0003; the identity with both exact to 6.5e-16
  its summary, the median: 0.9843 whole, 0.9785 and 0.9791 by half, whose root mean square is 0.9788
    halves within 10% of each other's count: 459, |gap| median 0.0006, largest 0.0115; the weights' part median 0.0007
    halves of unequal counts: 45, |gap| median 0.0209, largest 0.9901; the weights' part median 0.0194
```

### equal_N16_V15.txt -- item 1: the new defaults on `_revised` (N16) against V15's F 21, S off, bit for bit

```text
claude/scratch/midoffice16/bias/N16.npz's model test against claude/scratch/midoffice15/bias/V15.npz's _rF21Sx: 686 forecast dates, 2024-02-08 to 2026-11-02
  forecast                 against forecast_rF21Sx                    equal to the bit
  realised                 against realised_rF21Sx                    equal to the bit
  member_forecast          against member_forecast_rF21Sx             equal to the bit
  eigen_forecast           against eigen_forecast_rF21Sx              equal to the bit
  eigen_realised           against eigen_realised_rF21Sx              equal to the bit
  repaired                 against repaired_rF21Sx                    equal to the bit
  held_long_only           against held_long_only_rF21Sx              equal to the bit
  held_unconstrained       against held_unconstrained_rF21Sx          equal to the bit
  realised_h5              against realised_h5_rF21Sx                 equal to the bit
  eigen_realised_h5        against eigen_realised_h5_rF21Sx           equal to the bit
  realised_h11             against realised_h11_rF21Sx                equal to the bit
  eigen_realised_h11       against eigen_realised_h11_rF21Sx          equal to the bit
  realised_h21             against realised_h21_rF21Sx                equal to the bit
  eigen_realised_h21       against eigen_realised_h21_rF21Sx          equal to the bit
  member_realised          (shared)                            equal to the bit
  member_realised_h5       (shared)                            equal to the bit
  member_realised_h11      (shared)                            equal to the bit
  member_realised_h21      (shared)                            equal to the bit
  caps                     (shared)                            equal to the bit
  regime_dates             (shared)                            equal to the bit
  regime_factor_squared    (shared)                            equal to the bit
  regime_specific_squared  (shared)                            equal to the bit
  regime_members           (shared)                            equal to the bit
  regime_effective         (shared)                            equal to the bit
every array equal to the bit
```

### compare_bias.txt -- item 1: the new defaults on `_released` (D16) against N16, every statistic recomputed, and where the two part

```text
N16: the new defaults on _revised; D16: the new defaults on _released. 686 forecast dates, 2024-02-08 to 2026-11-02; the first half to 2025-06-23, the second from 2025-06-24

The minimum-variance portfolios at one day: the bias statistic, the volatility a year (the forecast at its median, the realised from the returns), by part of the test
                                                                       N16                    D16
  minimum variance, long-only          whole           0.990  9.05% 10.13%    0.990  9.05% 10.13%
  minimum variance, long-only          first half      1.109  8.76% 11.88%    1.109  8.76% 11.88%
  minimum variance, long-only          second half     0.855  9.42%  8.02%    0.855  9.42%  8.02%
  minimum variance, unconstrained      whole           1.033  8.79% 10.19%    1.033  8.79% 10.19%
  minimum variance, unconstrained      first half      1.163  8.48% 11.95%    1.163  8.48% 11.95%
  minimum variance, unconstrained      second half     0.887  9.15%  8.08%    0.886  9.15%  8.08%
  minimum variance, long-only, N16: MRAD 0.1261
  minimum variance, long-only, D16: MRAD 0.1261
  minimum variance, unconstrained, N16: MRAD 0.1432
  minimum variance, unconstrained, D16: MRAD 0.1432

The minimum-variance portfolios at the other horizons, the windows overlapping: N16 / D16
  minimum variance, long-only           5 days 1.042/1.042  11 days 0.972/0.972  21 days 0.911/0.911
  minimum variance, unconstrained       5 days 1.078/1.078  11 days 0.988/0.988  21 days 0.932/0.932

The broad portfolios: the statistic at one day by part, and at 21 days -- N16 / D16
  the index, cap-weighted                0.934/0.934    1.063/1.063    0.785/0.785    0.807/0.807   (whole, first, second; 21 days)
  the index, equally weighted            0.893/0.893    0.983/0.983    0.793/0.793    0.752/0.752   (whole, first, second; 21 days)
  smallest 85, equally weighted          0.902/0.902    0.921/0.921    0.882/0.882    0.743/0.743   (whole, first, second; 21 days)
  largest 50, equally weighted           0.920/0.920    1.024/1.024    0.804/0.804    0.823/0.823   (whole, first, second; 21 days)
  the index, cap-weighted              volatility a year, forecast / realised: N16 14.73% / 15.27%, D16 14.73% / 15.27%; MRAD 0.0985 / 0.0985
  the index, equally weighted          volatility a year, forecast / realised: N16 13.81% / 13.96%, D16 13.81% / 13.96%; MRAD 0.0893 / 0.0893
  smallest 85, equally weighted        volatility a year, forecast / realised: N16 17.47% / 17.38%, D16 17.47% / 17.38%; MRAD 0.0844 / 0.0844
  largest 50, equally weighted         volatility a year, forecast / realised: N16 14.41% / 14.64%, D16 14.41% / 14.64%; MRAD 0.1037 / 0.1037

Every member singly (at least 63 forecasts):
  whole        543 / 543 members: median 0.987 / 0.987, mean abs(bias - 1) 0.088 / 0.088
  first half   518 / 518 members: median 0.978 / 0.978, mean abs(bias - 1) 0.099 / 0.099
  second half  519 / 519 members: median 0.986 / 0.986, mean abs(bias - 1) 0.098 / 0.098
  the members whose statistic moved most, N16 -> D16: SPGI 0.991 -> 1.002, MRNA 1.823 -> 1.823, PTC 1.178 -> 1.178, ROL 0.886 -> 0.886, MA 0.910 -> 0.910, ALLE 0.906 -> 0.906, BR 0.928 -> 0.928, ADSK 1.106 -> 1.106, MSFT 1.063 -> 1.062, FERG 0.751 -> 0.751, LDOS 0.921 -> 0.921, PYPL 1.037 -> 1.037

Each spin-off parent with an ex-date in the test, its member statistic N16 -> D16, and its outcome on the day in forecast sd:
  J      0.738 -> 0.738; 2024-09-30: +2.4 -> +2.4 sd; 2025-05-16: +0.5 -> +0.5 sd
  LEN    1.108 -> 1.108; 2025-01-21: +1.8 -> +1.8 sd
  WDC    1.136 -> 1.136; 2025-02-24: -2.0 -> -2.0 sd
  CMCSA  1.015 -> 1.015; 2026-01-05: +0.5 -> +0.5 sd
  BDX    0.947 -> 0.947; 2026-02-10: +2.3 -> +2.3 sd
  APTV   0.946 -> 0.946; 2026-04-01: +1.1 -> +1.1 sd
  FDX    1.029 -> 1.029; 2026-06-01: +0.2 -> +0.2 sd
  HON    0.843 -> 0.843; 2026-06-29: -1.8 -> -1.8 sd
  SPGI   0.991 -> 1.002; 2026-07-01: +1.0 -> +3.7 sd
  CTVA   0.867 -> 0.867; 2026-10-01: +2.1 -> +2.1 sd
  FDXF: held on 0 / 0 forecast dates
  VYLR: held on 0 / 0 forecast dates

The industries, mean abs(bias - 1) at one day: 0.055 / 0.055
Random portfolios, long-only, median at one day: 0.914 / 0.914
Random portfolios, long-short, median at one day: 0.997 / 0.997
Eigenportfolios 1-5, mean at one day: 1.011 / 1.011
Eigenportfolios 6-10, mean at one day: 1.084 / 1.084
Eigenportfolios 11-15, mean at one day: 1.111 / 1.111
Eigenportfolios 16-20, mean at one day: 1.223 / 1.223
Eigenportfolios 21-25, mean at one day: 1.195 / 1.195

The arrays' equality, N16 and D16: {'caps': True, 'held_long_only': False, 'held_unconstrained': True, 'repaired': True, 'forecast': False, 'realised': False, 'member_forecast': False, 'member_realised': False, 'eigen_forecast': False}
  the capitalisations that differ: 0 member-days, in 
  forecast: 87 of 686 forecast dates differ, the first 2026-07-01, none before
  member_forecast: 87 of 686 forecast dates differ, the first 2026-07-01, none before
  eigen_forecast: 87 of 686 forecast dates differ, the first 2026-07-01, none before
  factor_covariance: 87 of 686 dates differ, the first 2026-07-01
  the regime history's B_F^2: 87 of 685 days differ, the first 2026-07-01: 1.632412 -> 1.658606; the largest relative change after it 1.6e-02
  lambda_F at 21 days: the largest change 1.10511 -> 1.10550 as of 2026-07-01 (+3.48e-04); on the day itself 1.10511 -> 1.10550
```

### model_compare.txt -- the model on `_released` (M16) against midoffice15's on `_revised` (M15)

```text
claude/scratch/midoffice15/model/M15 against claude/scratch/midoffice16/model/M16: 21 fields compared; equal: 11; differing: covariance, observed_variance, sampling_variance, weighted_sampling_variance, noise_share, weighted_noise_share, mean_r2, factor_returns, specific, exposures
  covariance: largest difference 4.632e-05
  observed_variance: largest difference 4.632e-05
  sampling_variance: largest difference 1.030e-07
  weighted_sampling_variance: largest difference 5.077e-07
  noise_share: largest difference 1.467e-04
  weighted_noise_share: largest difference 1.786e-04
  mean_r2: largest difference 4.745e-06
  factor_returns: largest difference 9.885e-04
  specific: largest difference 4.388e-04
  exposures: largest difference 1.878e-01
  qualifications: 6 and 6; differing: 0
  report: 162 and 163 lines; 49 differing line(s)
    +  data/external/polygon/sp500_securities_2022-01-03_2026-11-03_released.npz (DEFAULT_PANEL)
    -  454,475 raw values winsorised before standardising
    -  183,725 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    +  454,483 raw values winsorised before standardising
    +  183,727 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    -  CMRA         93%      938    3.75%      6.27%     1.16
    +  CMRA         93%      938    3.75%      6.27%     1.15
    -  LCAPCB       93%    1,213    1.65%      3.92%     0.41
    +  LCAPCB       93%    1,213    1.65%      3.92%     0.42
    -  Volatility                  -1.6e-17   1.000000   503
    -  Momentum                    -2.9e-17   1.000000   500
    +  Volatility                  -7.2e-18   1.000000   503
    +  Momentum                    +7.2e-18   1.000000   500
    -  Trading Activity            +0.0e+00   1.000000   503
    +  Trading Activity            -3.6e-17   1.000000   503
    -  Finance & Real Estate      17.35%     +17.98%    24.1%      37.0%      938          0      0
    -  Manufacturing              12.07%     +20.72%    16.5%      24.3%      938          0      0
    -  Mining                     21.15%     +14.77%    14.2%      21.4%      938          0      0
    -  Retail                     14.75%     +14.56%     8.4%      12.9%      938          0      0
    -  SIC20                      13.35%     +10.89%    26.7%      45.3%      938          0      0
    -  SIC28                      16.14%     +16.55%     6.7%      10.6%      938          0      0
    +  Finance & Real Estate      17.34%     +17.97%    24.1%      37.1%      938          0      0
    +  Manufacturing              12.07%     +20.71%    16.5%      24.3%      938          0      0
    +  Mining                     21.16%     +14.75%    14.2%      21.4%      938          0      0
    +  Retail                     14.75%     +14.55%     8.4%      12.9%      938          0      0
    +  SIC20                      13.35%     +10.88%    26.7%      45.3%      938          0      0
    +  SIC28                      16.14%     +16.54%     6.7%      10.6%      938          0      0
    -  SIC38                      16.82%     +16.02%     8.0%      12.3%      938          0      0
    -  SIC49                      19.01%     +23.75%     7.8%      11.9%      938          0      0
    +  SIC38                      16.83%     +16.01%     8.0%      12.3%      938          0      0
    +  SIC49                      19.02%     +23.73%     7.7%      11.9%      938          0      0
    -  SIC62                      21.57%     +18.49%     8.6%      11.5%      938          0      0
    -  SIC63                      14.22%     +19.02%    15.1%      24.2%      938          0      0
    -  SIC67                      17.83%     +17.80%    11.8%      18.3%      938          0      0
    -  SIC73                      17.52%     +15.23%     3.1%       4.9%      938          0      0
    +  SIC62                      21.57%     +18.48%     8.6%      11.5%      938          0      0
    +  SIC63                      14.22%     +19.00%    15.1%      24.2%      938          0      0
    +  SIC67                      17.83%     +17.79%    11.8%      18.3%      938          0      0
    +  SIC73                      17.53%     +15.26%     3.1%       4.9%      938          0      0
    -  Transport & Utilities      16.26%     +15.55%    11.2%      17.5%      938          0      0
    -  Wholesale                  15.06%     +20.47%    39.5%      60.4%      938          0      0
    +  Transport & Utilities      16.26%     +15.54%    11.2%      17.5%      938          0      0
    +  Wholesale                  15.06%     +20.46%    39.5%      60.4%      938          0      0
    -  Size                        3.62%      +3.05%     8.0%      11.5%      938          0      0
    +  Size                        3.62%      +3.05%     8.0%      11.4%      938          0      0
    -  specific risk   median 28.98%   p10 16.56%   p90 48.89%
    +  specific risk   median 28.99%   p10 16.56%   p90 48.89%
    -                    each history weighted by a half-life of 63 trading days; the characteristics' weight in a blended forecast, at the median effective history of 181.8 observations of 938: 13.58%
    +                    each history weighted by a half-life of 63 trading days; the characteristics' weight in a blended forecast, at the median effective history of 181.8 observations of 938: 13.59%
```

### plain_cost.txt -- item 1: what the factor side costs a plain run

```text
What the regime adjustment's factor side costs a plain run: P16x, the factor side off, against P16, the default (the runner's log, one at a time, BLAS on one thread)
  2026-11-13 11:54:17  P16x: done in 0.8 min
  2026-11-13 12:01:50  P16: done in 7.5 min

P16's report, its regime section (the factor side's lines, the refits' time):
  The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
    the last day's B_F^2 0.424 over 25 factors, B_S^2 0.701 over 499 members (an effective number of 48)
    the model refitted as of each of 686 days for it, 2024-02-08 to 2026-11-02: 6.8 min

The two reports' difference, P16x against P16 (diff):
  163a164,168
  > 
  > The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
  >   as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
  >   the last day's B_F^2 0.424 over 25 factors, B_S^2 0.701 over 499 members (an effective number of 48)
  >   the model refitted as of each of 686 days for it, 2024-02-08 to 2026-11-02: 6.8 min
```

### check_fast_regime.txt -- item 1: the plain run's regime history from the refits and from the full fit alone, bit for bit

```text
The regime history two ways (midoffice16, check_fast_regime.py)
  the run's: 686 refits, 8.34 min; each refit's factor returns the full fit's through its date, to the bit, in 686 of 686
  the full fit's: 686 covariance estimates on its own factor returns' prefixes, 0.3 s; the factor volatilities the refits' to the bit on 686 of 686 days
  B_F^2: the same 686 days True; equal to the bit True
  lambda_F as of 2026-11-03: 0.964025715108432 by the refits, 0.964025715108432 by the full fit
```

### smoke/equal_c63.txt -- item 2: the one-pass test's 63-day setting against a run at 63 days alone, the last 30 forecast dates, bit for bit

```text
claude/scratch/midoffice16/smoke/c63.npz's model test against claude/scratch/midoffice16/smoke/c.npz's c63: 30 forecast dates, 2026-09-22 to 2026-11-02
  forecast                 against forecast_c63                       equal to the bit
  realised                 against realised_c63                       equal to the bit
  member_forecast          against member_forecast_c63                equal to the bit
  eigen_forecast           against eigen_forecast_c63                 equal to the bit
  eigen_realised           against eigen_realised_c63                 equal to the bit
  repaired                 against repaired_c63                       equal to the bit
  held_long_only           against held_long_only_c63                 equal to the bit
  held_unconstrained       against held_unconstrained_c63             equal to the bit
  factor_covariance        against factor_covariance_c63              equal to the bit
  exposure_long_only       against exposure_long_only_c63             equal to the bit
  exposure_unconstrained   against exposure_unconstrained_c63         equal to the bit
  realised_h5              against realised_h5_c63                    equal to the bit
  eigen_realised_h5        against eigen_realised_h5_c63              equal to the bit
  regime_dates             against regime_dates_c63                   equal to the bit
  regime_factor_squared    against regime_factor_squared_c63          equal to the bit
  regime_specific_squared  against regime_specific_squared_c63        equal to the bit
  regime_members           against regime_members_c63                 equal to the bit
  regime_effective         against regime_effective_c63               equal to the bit
  member_realised          (shared)                            equal to the bit
  caps                     (shared)                            equal to the bit
  member_realised_h5       (shared)                            equal to the bit
every array equal to the bit
```

### equal_C16_D16.txt -- item 2: C16's model test against D16, bit for bit

```text
claude/scratch/midoffice16/bias/D16.npz's model test against claude/scratch/midoffice16/bias/C16.npz's model test: 686 forecast dates, 2024-02-08 to 2026-11-02
  forecast                 against forecast                           equal to the bit
  realised                 against realised                           equal to the bit
  member_forecast          against member_forecast                    equal to the bit
  eigen_forecast           against eigen_forecast                     equal to the bit
  eigen_realised           against eigen_realised                     equal to the bit
  repaired                 against repaired                           equal to the bit
  held_long_only           against held_long_only                     equal to the bit
  held_unconstrained       against held_unconstrained                 equal to the bit
  factor_covariance        against factor_covariance                  equal to the bit
  exposure_long_only       against exposure_long_only                 equal to the bit
  exposure_unconstrained   against exposure_unconstrained             equal to the bit
  realised_h5              against realised_h5                        equal to the bit
  eigen_realised_h5        against eigen_realised_h5                  equal to the bit
  realised_h11             against realised_h11                       equal to the bit
  eigen_realised_h11       against eigen_realised_h11                 equal to the bit
  realised_h21             against realised_h21                       equal to the bit
  eigen_realised_h21       against eigen_realised_h21                 equal to the bit
  regime_dates             against regime_dates                       equal to the bit
  regime_factor_squared    against regime_factor_squared              equal to the bit
  regime_specific_squared  against regime_specific_squared            equal to the bit
  regime_members           against regime_members                     equal to the bit
  regime_effective         against regime_effective                   equal to the bit
  member_realised          (shared)                            equal to the bit
  caps                     (shared)                            equal to the bit
  member_realised_h5       (shared)                            equal to the bit
  member_realised_h11      (shared)                            equal to the bit
  member_realised_h21      (shared)                            equal to the bit
every array equal to the bit
```

### bias/report_C16.txt -- items 2 and 3: C16's report, the correlation half-lives

```text
  The factor correlations' half-life at one day: the model's 504 (*) and 63, 126, 252 trading days, the volatilities on their own half-life throughout, every setting on its own forecasts' volatility regime adjustment, F  21 S   -
  The factor volatilities do not depend on the correlations, so each setting's regime multiplier should be the model's: each setting's own against the model's -- its factor volatilities, its daily B_F^2 and lambda_F on 21 days
      63: factor volatilities equal to the bit in 6,516 of 17,150 date-factors, the largest relative difference 6.7e-16; B_F^2 on 424 of 685 days (4.4e-16); lambda_F on 581 of 685 (2.2e-16)
     126: factor volatilities equal to the bit in 6,693 of 17,150 date-factors, the largest relative difference 6.7e-16; B_F^2 on 441 of 685 days (4.4e-16); lambda_F on 586 of 685 (2.2e-16)
     252: factor volatilities equal to the bit in 6,635 of 17,150 date-factors, the largest relative difference 6.7e-16; B_F^2 on 419 of 685 days (4.4e-16); lambda_F on 594 of 685 (2.2e-16)
  where they differ it is the last bit: the two-half-life estimator's diagonal is each factor's correlation with itself, c / (sqrt(c) sqrt(c)) from the long window, times its variance from the short one, and that ratio rounds to exactly one only about half the time, its rounding moving with the correlations' half-life

  First, the minimum-variance portfolios' realised volatility, % a year, whole / first half / second half; a setting under which either realises more over the whole test than at 504, to the hundredth of a point, is out
                       long-only           unconstrained
  63       10.12 / 12.10 /  7.67   10.23 / 12.18 /  7.83   out
  126      10.12 / 12.00 /  7.83   10.18 / 12.03 /  7.94   in
  252      10.13 / 11.93 /  7.95   10.19 / 11.97 /  8.03   in
  504 *    10.13 / 11.88 /  8.02   10.19 / 11.95 /  8.08   in

  Then the regime criterion -- the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median) -- by part:
  63       0.104 / 0.116 / 0.083
  126      0.105 / 0.104 / 0.110
  252      0.105 / 0.100 / 0.119
  504 *    0.105 / 0.098 / 0.123
  chosen on the first half by the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own correlation half-life, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second: 504 * (0.098; the model's 0.098); on the second half it reads 0.123 against the model's 0.123, the best there 63

  The shape: each setting's own eigenportfolios -- the eigenvectors of its covariance -- by rank, 1 the largest eigenvalue, the market-like one; the statistic by part
  the whole test (2024-02-08 to 2026-11-02, 686)
  rank                      63     126     252   504 *
  1                      0.964   0.952   0.947   0.946
  2                      0.936   1.013   1.005   0.978
  3                      1.019   1.012   1.043   1.051
  4                      1.014   1.048   1.036   1.033
  5                      1.032   1.022   1.038   1.047
  6                      1.017   1.027   1.031   1.041
  7                      1.028   1.058   1.072   1.084
  8                      1.022   1.058   1.072   1.083
  9                      1.004   1.044   1.101   1.128
  10                     1.060   1.087   1.085   1.084
  11                     1.052   1.075   1.085   1.088
  12                     1.056   1.080   1.091   1.098
  13                     1.154   1.141   1.106   1.096
  14                     1.107   1.071   1.103   1.109
  15                     1.172   1.167   1.169   1.163
  16                     1.205   1.204   1.196   1.186
  17                     1.248   1.195   1.164   1.156
  18                     1.194   1.154   1.187   1.245
  19                     1.335   1.310   1.299   1.288
  20                     1.288   1.264   1.241   1.238
  21                     1.275   1.194   1.191   1.181
  22                     1.330   1.250   1.203   1.177
  23                     1.330   1.230   1.205   1.224
  24                     1.326   1.259   1.249   1.255
  25                     1.254   1.170   1.140   1.135
  ranks 2-25, mean       1.144   1.130   1.130   1.132
  ranks 21-25, mean      1.303   1.220   1.198   1.195
  mean abs(bias - 1)     0.145   0.127   0.127   0.131
  the first half (2024-02-08 to 2025-06-23, 343)
  rank                      63     126     252   504 *
  1                      1.059   1.062   1.063   1.064
  2                      0.825   0.922   0.919   0.912
  3                      1.042   0.973   0.981   0.970
  4                      0.981   1.003   1.000   1.017
  5                      1.033   0.984   0.973   0.963
  6                      0.967   0.939   0.920   0.911
  7                      0.957   0.968   0.973   0.979
  8                      0.972   0.989   0.995   0.996
  9                      0.951   0.926   0.940   0.942
  10                     0.989   0.965   0.951   0.959
  11                     0.910   0.933   0.948   0.942
  12                     1.042   1.017   0.989   0.998
  13                     1.088   1.055   1.064   1.062
  14                     1.005   0.978   0.999   1.006
  15                     1.110   1.091   1.068   1.057
  16                     1.010   1.033   1.092   1.122
  17                     1.250   1.149   1.076   1.058
  18                     1.077   0.983   1.011   1.040
  19                     1.231   1.165   1.127   1.112
  20                     1.110   1.090   1.077   1.088
  21                     1.274   1.173   1.152   1.149
  22                     1.228   1.151   1.119   1.115
  23                     1.284   1.183   1.164   1.160
  24                     1.234   1.175   1.172   1.174
  25                     1.135   1.050   1.025   1.021
  ranks 2-25, mean       1.071   1.037   1.031   1.031
  ranks 21-25, mean      1.231   1.146   1.127   1.124
  mean abs(bias - 1)     0.106   0.074   0.065   0.067
  the second half (2025-06-24 to 2026-11-02, 343)
  rank                      63     126     252   504 *
  1                      0.861   0.830   0.817   0.814
  2                      1.036   1.096   1.086   1.040
  3                      0.997   1.052   1.102   1.127
  4                      1.047   1.092   1.071   1.048
  5                      1.030   1.059   1.100   1.125
  6                      1.067   1.105   1.132   1.158
  7                      1.094   1.141   1.163   1.182
  8                      1.071   1.124   1.144   1.165
  9                      1.052   1.150   1.242   1.288
  10                     1.128   1.197   1.206   1.196
  11                     1.176   1.198   1.208   1.216
  12                     1.068   1.138   1.186   1.190
  13                     1.208   1.221   1.143   1.128
  14                     1.202   1.157   1.200   1.205
  15                     1.231   1.240   1.263   1.258
  16                     1.374   1.355   1.291   1.249
  17                     1.240   1.240   1.246   1.247
  18                     1.302   1.304   1.342   1.420
  19                     1.433   1.437   1.452   1.444
  20                     1.446   1.414   1.375   1.362
  21                     1.276   1.214   1.231   1.215
  22                     1.426   1.343   1.283   1.238
  23                     1.376   1.275   1.245   1.285
  24                     1.414   1.337   1.321   1.330
  25                     1.364   1.281   1.247   1.239
  ranks 2-25, mean       1.211   1.215   1.220   1.223
  ranks 21-25, mean      1.371   1.290   1.265   1.261
  mean abs(bias - 1)     0.208   0.214   0.218   0.222

  the whole test (2024-02-08 to 2026-11-02, 686), by correlation half-life (504 the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  63         0.953   0.940   0.952   0.936   1.044  10.124   1.105  10.232   0.992   0.145   0.077   0.150   0.084   0.104
  126        0.940   0.915   0.927   0.925   1.011  10.122   1.057  10.184   0.990   0.127   0.087   0.144   0.084   0.105
  252        0.935   0.900   0.911   0.921   0.996  10.131   1.039  10.187   0.989   0.127   0.092   0.139   0.084   0.105
  504 *      0.934   0.893   0.902   0.920   0.990  10.128   1.033  10.193   0.987   0.131   0.094   0.135   0.086   0.105

  the first half (2024-02-08 to 2025-06-23, 343), by correlation half-life (504 the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  63         1.061   1.020   0.979   1.018   1.163  12.104   1.237  12.182   0.989   0.106   0.056   0.212   0.080   0.116
  126        1.062   1.004   0.952   1.021   1.135  11.998   1.191  12.029   0.984   0.074   0.059   0.174   0.078   0.104
  252        1.063   0.991   0.932   1.023   1.119  11.930   1.171  11.975   0.980   0.065   0.065   0.157   0.078   0.100
  504 *      1.063   0.983   0.921   1.024   1.109  11.881   1.163  11.953   0.978   0.067   0.068   0.150   0.076   0.098

  the second half (2025-06-24 to 2026-11-02, 343), by correlation half-life (504 the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  63         0.833   0.853   0.925   0.848   0.911   7.666   0.956   7.828   0.988   0.208   0.113   0.056   0.081   0.083
  126        0.801   0.817   0.902   0.819   0.870   7.827   0.905   7.938   0.987   0.214   0.143   0.106   0.082   0.110
  252        0.788   0.800   0.889   0.807   0.858   7.953   0.890   8.029   0.987   0.218   0.155   0.121   0.082   0.119
  504 *      0.785   0.793   0.882   0.804   0.855   8.018   0.886   8.077   0.986   0.222   0.159   0.125   0.084   0.123

  The members by capitalisation quintile, the largest first, by part: the median statistic
  the whole test
  63       1.039, 0.999, 0.991, 0.957, 0.972
  126      1.035, 0.996, 0.989, 0.953, 0.970
  252      1.029, 0.992, 0.988, 0.950, 0.965
  504 *    1.028, 0.992, 0.988, 0.949, 0.963
  the first half
  63       1.033, 0.993, 0.991, 0.956, 0.970
  126      1.032, 0.988, 0.990, 0.948, 0.963
  252      1.029, 0.985, 0.987, 0.941, 0.958
  504 *    1.028, 0.985, 0.985, 0.940, 0.954
  the second half
  63       1.020, 0.993, 0.991, 0.946, 0.960
  126      1.023, 0.995, 0.983, 0.942, 0.962
  252      1.022, 0.993, 0.986, 0.944, 0.960
  504 *    1.024, 0.990, 0.984, 0.944, 0.960

  Day to day: the median absolute change in log forecast volatility from one forecast date to the next (its 95th percentile), the cap-weighted index's and every member's (the median of each one's median)
             index     p95  members     p95
  63         1.06%   3.86%    1.43%   6.44%
  126        1.01%   3.67%    1.46%   6.40%
  252        1.00%   3.61%    1.47%   6.39%
  504 *      1.00%   3.58%    1.48%   6.39%
```

### analyse_correlation.txt -- items 3 and 4: C16 read from its stored arrays -- the same directions under every setting, the optimiser by eigen-direction, the correlations

```text
C16: 686 forecast dates, 2024-02-08 to 2026-11-02; the first half to 2025-06-23, the second from 2025-06-24; correlation half-lives 63, 126, 252, 504 * (* the model's)

1. The minimum-variance portfolios' realised volatility, % a year, whole / first / second, and their statistic
  minimum variance, long-only          63: realised 10.12 / 12.10 /  7.67; forecast  8.46 at the median; statistic 1.044 / 1.163 / 0.911
  minimum variance, long-only         126: realised 10.12 / 12.00 /  7.83; forecast  8.82 at the median; statistic 1.011 / 1.135 / 0.870
  minimum variance, long-only         252: realised 10.13 / 11.93 /  7.95; forecast  8.99 at the median; statistic 0.996 / 1.119 / 0.858
  minimum variance, long-only       504 *: realised 10.13 / 11.88 /  8.02; forecast  9.05 at the median; statistic 0.990 / 1.109 / 0.855
  minimum variance, unconstrained      63: realised 10.23 / 12.18 /  7.83; forecast  8.17 at the median; statistic 1.105 / 1.237 / 0.956
  minimum variance, unconstrained     126: realised 10.18 / 12.03 /  7.94; forecast  8.52 at the median; statistic 1.057 / 1.191 / 0.905
  minimum variance, unconstrained     252: realised 10.19 / 11.97 /  8.03; forecast  8.71 at the median; statistic 1.039 / 1.171 / 0.890
  minimum variance, unconstrained   504 *: realised 10.19 / 11.95 /  8.08; forecast  8.79 at the median; statistic 1.033 / 1.163 / 0.886

2. The model's eigenportfolios -- the eigenvectors of its 504-day covariance each date -- forecast by every setting's covariance, against the outcome they share; by rank and part
  the model's eigenvalues recomputed from its stored covariance against its test's: largest relative difference 0.0e+00
  whole
  rank                    63     126     252   504 *
  1                    0.963   0.951   0.947   0.946
  2                    0.997   0.994   0.985   0.978
  3                    0.998   1.027   1.043   1.051
  4                    1.006   1.021   1.030   1.033
  5                    1.065   1.065   1.056   1.047
  6                    1.011   1.031   1.040   1.041
  7                    1.041   1.068   1.080   1.084
  8                    1.051   1.070   1.079   1.083
  9                    1.065   1.101   1.121   1.128
  10                   1.005   1.049   1.073   1.084
  11                   1.065   1.080   1.086   1.088
  12                   1.056   1.084   1.095   1.098
  13                   1.070   1.087   1.094   1.096
  14                   1.060   1.093   1.107   1.109
  15                   1.119   1.150   1.161   1.163
  16                   1.160   1.183   1.188   1.186
  17                   1.117   1.146   1.155   1.156
  18                   1.203   1.235   1.245   1.245
  19                   1.249   1.280   1.289   1.288
  20                   1.223   1.247   1.246   1.238
  21                   1.173   1.188   1.187   1.181
  22                   1.169   1.181   1.181   1.177
  23                   1.240   1.237   1.230   1.224
  24                   1.212   1.228   1.243   1.255
  25                   1.177   1.165   1.148   1.135
  ranks 2-25, mean     1.105   1.125   1.132   1.132
  ranks 21-25, mean    1.194   1.200   1.198   1.195
  mean abs(bias - 1)   0.103   0.123   0.130   0.131
  rank 1's forecast volatility against the model's, the median ratio: 63 0.990, 126 0.996, 252 0.999, 504 * 1.000
  first half
  rank                    63     126     252   504 *
  1                    1.060   1.062   1.063   1.064
  2                    0.963   0.939   0.922   0.912
  3                    0.946   0.955   0.964   0.970
  4                    1.010   1.011   1.015   1.017
  5                    0.995   0.974   0.966   0.963
  6                    0.937   0.922   0.914   0.911
  7                    0.985   0.979   0.978   0.979
  8                    1.007   0.998   0.995   0.996
  9                    0.958   0.949   0.944   0.942
  10                   0.930   0.942   0.952   0.959
  11                   0.953   0.942   0.941   0.942
  12                   0.972   0.982   0.992   0.998
  13                   1.053   1.054   1.059   1.062
  14                   1.010   1.008   1.006   1.006
  15                   1.047   1.047   1.052   1.057
  16                   1.134   1.124   1.121   1.122
  17                   1.030   1.038   1.049   1.058
  18                   1.068   1.049   1.042   1.040
  19                   1.144   1.119   1.112   1.112
  20                   1.115   1.100   1.091   1.088
  21                   1.119   1.138   1.146   1.149
  22                   1.142   1.127   1.119   1.115
  23                   1.169   1.165   1.162   1.160
  24                   1.129   1.148   1.164   1.174
  25                   1.072   1.048   1.031   1.021
  ranks 2-25, mean     1.037   1.032   1.031   1.031
  ranks 21-25, mean    1.126   1.125   1.124   1.124
  mean abs(bias - 1)   0.067   0.066   0.067   0.067
  rank 1's forecast volatility against the model's, the median ratio: 63 0.997, 126 0.997, 252 0.999, 504 * 1.000
  second half
  rank                    63     126     252   504 *
  1                    0.858   0.827   0.816   0.814
  2                    1.031   1.046   1.045   1.040
  3                    1.049   1.096   1.119   1.127
  4                    1.003   1.031   1.045   1.048
  5                    1.131   1.150   1.139   1.125
  6                    1.080   1.131   1.153   1.158
  7                    1.096   1.151   1.176   1.182
  8                    1.095   1.139   1.158   1.165
  9                    1.163   1.236   1.274   1.288
  10                   1.075   1.146   1.181   1.196
  11                   1.167   1.202   1.215   1.216
  12                   1.134   1.177   1.189   1.190
  13                   1.086   1.119   1.128   1.128
  14                   1.108   1.174   1.200   1.205
  15                   1.183   1.242   1.259   1.258
  16                   1.188   1.241   1.253   1.249
  17                   1.198   1.245   1.253   1.247
  18                   1.325   1.396   1.420   1.420
  19                   1.348   1.423   1.446   1.444
  20                   1.311   1.368   1.372   1.362
  21                   1.226   1.238   1.228   1.215
  22                   1.198   1.235   1.242   1.238
  23                   1.308   1.304   1.295   1.285
  24                   1.287   1.300   1.316   1.330
  25                   1.274   1.273   1.256   1.239
  ranks 2-25, mean     1.169   1.211   1.224   1.223
  ranks 21-25, mean    1.259   1.270   1.268   1.261
  mean abs(bias - 1)   0.168   0.210   0.222   0.222
  rank 1's forecast volatility against the model's, the median ratio: 63 0.942, 126 0.982, 252 0.997, 504 * 1.000

3. The minimum-variance portfolios' risk by eigen-direction of each setting's own covariance: the share of the forecast variance (the mean over dates), and the statistic of the realised part over its forecast; the factor part's eigen groups, then the specific part
  minimum variance, long-only
        63: rank 1 37.9% 0.96/1.06/0.86; ranks 2-5 13.7% 0.99/0.95/1.03; ranks 6-15  9.1% 1.07/1.03/1.11; ranks 16-25 23.8% 1.25/1.22/1.29; specific 15.6% 0.87/0.88/0.86
       126: rank 1 39.7% 0.95/1.06/0.83; ranks 2-5 13.8% 1.02/0.94/1.10; ranks 6-15  8.6% 1.08/1.03/1.13; ranks 16-25 22.1% 1.20/1.15/1.24; specific 15.7% 0.88/0.88/0.88
       252: rank 1 40.9% 0.95/1.06/0.82; ranks 2-5 13.5% 1.03/0.93/1.13; ranks 6-15  8.4% 1.11/1.04/1.16; ranks 16-25 21.1% 1.18/1.13/1.22; specific 16.0% 0.88/0.89/0.88
     504 *: rank 1 41.6% 0.95/1.06/0.81; ranks 2-5 13.2% 1.03/0.92/1.14; ranks 6-15  8.4% 1.13/1.06/1.18; ranks 16-25 20.6% 1.17/1.13/1.22; specific 16.2% 0.88/0.89/0.87
    (each cell: the share of the forecast daily variance, then the statistic whole/first/second)
  minimum variance, unconstrained
        63: rank 1 31.1% 0.96/1.06/0.86; ranks 2-5 13.3% 0.99/0.95/1.04; ranks 6-15 11.1% 1.08/1.04/1.11; ranks 16-25 28.2% 1.25/1.23/1.27; specific 16.3% 0.90/0.91/0.90
       126: rank 1 33.6% 0.95/1.06/0.83; ranks 2-5 14.3% 1.02/0.93/1.11; ranks 6-15 10.5% 1.08/1.03/1.13; ranks 16-25 25.3% 1.20/1.16/1.23; specific 16.3% 0.92/0.92/0.91
       252: rank 1 34.9% 0.95/1.06/0.82; ranks 2-5 14.3% 1.04/0.92/1.14; ranks 6-15 10.2% 1.11/1.04/1.17; ranks 16-25 24.0% 1.18/1.14/1.22; specific 16.6% 0.91/0.92/0.91
     504 *: rank 1 35.4% 0.95/1.06/0.81; ranks 2-5 14.1% 1.04/0.92/1.14; ranks 6-15 10.2% 1.12/1.05/1.18; ranks 16-25 23.4% 1.17/1.14/1.21; specific 16.9% 0.91/0.93/0.90
    (each cell: the share of the forecast daily variance, then the statistic whole/first/second)

4. The correlations: the market direction's share of the factor variance (the largest eigenvalue over the trace) and the mean correlation between factors, the mean over each part's dates
      63: whole 0.600 / +0.408;  first half 0.605 / +0.419;  second half 0.594 / +0.398
     126: whole 0.612 / +0.417;  first half 0.599 / +0.411;  second half 0.625 / +0.423
     252: whole 0.615 / +0.418;  first half 0.595 / +0.405;  second half 0.635 / +0.431
   504 *: whole 0.615 / +0.417;  first half 0.593 / +0.402;  second half 0.636 / +0.432
```

### bias/runner_log.txt -- the runs, one at a time

```text
2026-11-13 11:24:34  runner started, pid 27148, for N16, D16, C16, P16x, P16; 4.3 GB available
2026-11-13 11:24:34  N16: 4.3 GB available
2026-11-13 11:24:34  N16: attempt 1: data/external/polygon/sp500_securities_2022-01-03_2026-11-03_revised.npz --bias-guard 0 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-13 11:32:52  N16: done in 8.3 min
2026-11-13 11:32:52  D16: 6.1 GB available
2026-11-13 11:32:52  D16: attempt 1: --bias-guard 5 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-13 11:43:57  D16: done in 11.1 min
2026-11-13 11:43:57  C16: 5.0 GB available
2026-11-13 11:43:57  C16: attempt 1: --bias-guard 0 --bias-correlation-half-lives 63,126,252 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-13 11:53:30  C16: done in 9.5 min
2026-11-13 11:53:30  P16x: 5.4 GB available
2026-11-13 11:53:30  P16x: attempt 1: --regime-factor-half-life none --industries --membership
2026-11-13 11:54:17  P16x: done in 0.8 min
2026-11-13 11:54:17  P16: 5.3 GB available
2026-11-13 11:54:17  P16: attempt 1: --industries --membership
2026-11-13 12:01:50  P16: done in 7.5 min
2026-11-13 12:01:50  runner finished; every run done
```

### final_checks.txt -- the final code: the panels rebuilt, the defaults re-captured; the suite, ruff, hygiene

```text
build _listed: exit 0
20
build _spinoffs: exit 0
20
build _revised: exit 0
20
build _released: exit 0
20
  [regime] 50 of 686 refits, 0.3 min
  [regime] 100 of 686 refits, 0.6 min
  [regime] 150 of 686 refits, 0.9 min
  [regime] 200 of 686 refits, 1.3 min
  [regime] 250 of 686 refits, 1.8 min
  [regime] 300 of 686 refits, 2.4 min
  [regime] 350 of 686 refits, 3.0 min
  [regime] 400 of 686 refits, 3.7 min
  [regime] 450 of 686 refits, 4.5 min
  [regime] 500 of 686 refits, 5.2 min
  [regime] 550 of 686 refits, 6.1 min
  [regime] 600 of 686 refits, 6.9 min
  [regime] 650 of 686 refits, 7.9 min
  [regime] 686 of 686 refits, 8.6 min
FD16: status 0, mean R2 0.293825, 938 periods, 25 factors, 562s, the estimator 1.0s
claude/scratch/midoffice16/model/M16 against claude/scratch/midoffice16/final/FD16: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 163 and 168 lines; 5 differing line(s)
    +
    +The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    +  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
    +  the last day's B_F^2 0.424 over 25 factors, B_S^2 0.701 over 499 members (an effective number of 48)
    +  the model refitted as of each of 686 days for it, 2024-02-08 to 2026-11-02: 8.6 min
The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
  the model refitted as of each of 686 days for it, 2024-02-08 to 2026-11-02: 8.6 min
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========== 2374 passed, 1 skipped, 19 warnings in 103.14s (0:01:43) ===========
All checks passed!
ruff: clean over the changed Python files
scripts/run_sp500_model.py: CRLF (2957 lines), parses, ASCII
scripts/build_sp500_securities.py: CRLF (754 lines), parses, ASCII
src/praxis/midoffice/model/bias.py: LF (709 lines), parses, ASCII
src/praxis/midoffice/model/crosssectional.py: CRLF (1612 lines), parses, ASCII
tests/midoffice/test_bias.py: LF (466 lines), parses, ASCII
tests/midoffice/test_securities.py: CRLF (628 lines), parses, ASCII
config/sp500_spin_off_overrides.json: LF (263 lines), -, ASCII
docs/FACTOR_MODEL.md: CRLF (1715 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (834 lines), -, ASCII
claude/handoffs/BRIEF_midoffice16_correlation_half_life.md: LF (109 lines), -, ASCII
claude/retros/RETRO_midoffice16_correlation_half_life.md: LF (749 lines), -, ASCII
all ok
exit 0
11
0
```

## git add (always last -- Rule 43)

This piece -- midoffice15's rulings, the released panel's default, the correlation half-lives' one-pass test, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add scripts/run_sp500_model.py scripts/build_sp500_securities.py src/praxis/midoffice/model/bias.py src/praxis/midoffice/model/crosssectional.py tests/midoffice/test_bias.py tests/midoffice/test_securities.py config/sp500_spin_off_overrides.json docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice16_correlation_half_life.md claude/retros/RETRO_midoffice16_correlation_half_life.md
```
