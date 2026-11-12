# Retro: Mid-office 13 -- midoffice12's rulings, then the eigenvalue adjustment

**Brief:** `claude/handoffs/BRIEF_midoffice13_eigenvalue_adjustment.md` -- written in Chat and
pasted into the Code session, transcribed verbatim
**Date:** 2026-11-12 (EST)
**Duration:** the brief arrived shortly before 16:26Z (11:26 EST), the first clock reading after
it. The rulings, the adjustment and its tests, the bias section's variants and the reporting
view's candidates were built and measured by 12:02 EST; the runner ran the five bias runs one at
a time from 12:02; the comparison, the docs, the state file and this retro followed.
**Status:** COMPLETE.
- **Rulings 1 and 2:** built and measured -- the one-day default, its bias run equal to H1
  to the bit, the guard on; a longer horizon's report naming it a reporting view.
- **Rulings 3 to 6:** nothing else to build; the candidate is item 5.
- **Items 1 to 5:** built and measured. No default changed beyond ruling 1.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 1, one day:** the model's horizon is one day, and the new defaults' bias run equals
  midoffice12's H1 to the bit in every array, the guard on. **Ruling 2:** a longer horizon's
  report says it is a reporting view, with the caveat on its gain.
- **The eigenfactor risk adjustment** (item 1), an option, off (`--eigen-adjustment`): lambda by
  rank from 1,000 histories simulated from the day's covariance and re-estimated by the model's
  own estimator -- stable to 0.002 (item 4), 0.49 s a refit, 1.25 s a refit with it and five
  tests against 0.67 s without. At a = 0 it simulates and changes nothing, so one run tests every
  scale beside the model, and that run's model test equals the defaults' to the bit.
- **The simulated bias is small** -- lambda 1.00 at the largest eigenvalue, about 1.06 at the
  smallest five, below one at ranks 2 to 4, 6 and 10 -- a third of the small eigenvalues'
  realised 1.19, so a = 1, 2, 3, 4 were tested, not USE4's 1.2.
- **By the brief's test a = 2** (item 2): chosen on the first half (0.149 against the model's
  0.191), holding on the second (0.112 against 0.138); the small eigenfactors from about 1.2 to
  about 1.07.
- **But it is not ready:** the minimum-variance portfolios rebuilt on it realise more risk at
  every a and in both halves (unconstrained 9.97% to 10.13% a year at a = 2); it shrinks the
  eigenvalues the simulation calls overestimated, where the returns disagree (rank 3, 1.045 to
  1.096); it does not leave the broad portfolios alone (item 3: the smallest 85 0.956 to 0.928,
  out of the band; the members' median 1.003 to 0.990); and the optimiser's statistic barely
  moves (1.220 to 1.211).
- **A data error dominates the optimiser's statistics:** CTVA's 2026-10-01 total return is
  -83.8%, Corteva's separation, a spin-off the panel does not carry; both minimum-variance
  portfolios held 9.4% of it, a -13 sigma outcome. Without it they read 1.055 and 1.097, not
  1.167 and 1.220; by half they are the regime's.
- **The reporting view's candidates** (item 5): the lags on the volatilities alone, Newey-West at
  20 lags on the volatilities' 90 days with one-day correlations, read the broad portfolios at 21
  days as the 21-day model does (0.072 against 0.067) and halve what its correlations' lags cost
  the small eigen-directions (1.48 against 2.69); the variance-ratio volatilities go further
  toward the one-day model but over-forecast the broad portfolios more.
- **What I would choose:** the adjustment off; the CTVA spin-off fixed first; then the adjustment
  one-sided and a volatility regime adjustment; the lags on the volatilities alone as the 21-day
  reporting view.
- **The runs:** five, one at a time through the runner, 12:02 to 13:02 EST; none failed. The
  21-day model rerun equals midoffice12's to the bit.
- **Tests:** 2,356 passed, 1 skipped; 13 new.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice13_eigenvalue_adjustment.md`, verbatim. In
outline:

- **"committed -- midoffice12."** Checked: 16f2fb2 in `git log`.
- **Rulings on midoffice12's decisions:**
  1. **The default horizon: one day.** The new defaults' bias run must equal H1's to the bit,
     and the guard runs on it.
  2. **The h-day risk of broad portfolios:** a run at the holding period's horizon, as a
     reporting view, and the report says which it is; its gain rests on one 2.7-year window
     with the April 2025 shock in it, and is re-tested as data accumulates.
  3. **Lags on the volatilities alone, the correlations one-day:** measured, as the reporting
     view's candidate.
  4. **The eigenvalue adjustment:** next, on the one-day matrix.
  5. **The variance-ratio variant:** an option, off.
  6. **The specific history:** 63.
- **midoffice13, the eigenvalue adjustment,** the rulings first:
  1. **The eigenfactor risk adjustment** (Menchero, Wang and Orr; USE4), an option, off: on each
     forecast date diagonalise the factor covariance, simulate M histories from it over the
     periods and weights the estimator used, re-estimate each with the same estimator and
     diagonalise it; each rank's bias the root mean of the simulated eigenfactor's true variance
     over its estimate; each eigenvalue scaled by (a (bias - 1) + 1)^2; one simulation per date
     serving every a.
  2. **a: 1 and two or three values above it,** judged at one day by the eigenportfolios and the
     minimum-variance portfolios; chosen on the test's first half, the second half's statistics
     reported as well.
  3. **What it should leave alone:** the broad portfolios, the members, the industries and the
     random portfolios at one day.
  4. **M:** enough draws that each rank's bias is stable to 0.01; the time per refit.
  5. **The reporting view's candidate at h = 21:** lags on the volatilities alone, the
     correlations one-day, two ways -- Newey-West at 20 lags on the volatilities' 90 days, and
     the variance-ratio variant's volatilities -- each against H21 and the one-day model, read at
     21 days and at one: the broad portfolios, the optimiser's, the smallest eigenportfolios.
- Runs one at a time through the runner. **"Change no default beyond ruling 1. Report, and say
  which you would choose and why."** Retro with its `git add` and the state file in the same
  commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `src/praxis/midoffice/model/eigen.py` | new: the eigenfactor risk adjustment -- `EigenBias` (`scales`, `adjusted`, `basis`), `eigen_bias` (the simulation through the estimator itself), `EIGEN_DRAWS`, `EIGEN_SEED` | new, 136 lines |
| `src/praxis/midoffice/model/crosssectional.py` | the estimator's `eigen_adjustment`, `eigen_draws`, `eigen_seed`: the simulation on the estimate, the eigenvalues scaled unless a = 0, refused with the sampling-error correction; the notes' `eigen_adjustment` and `eigen`; the qualification; the docstring | 108, 185-188, 420-428, 895-897, 997-1017, 1290-1299, 1484-1485 |
| `src/praxis/midoffice/model/bias.py` | `RiskForecast.eigenbasis` and `from_model(eigen_scale=)`; `BiasTest` -- `evaluate`'s loop as `add` and `result`, the same arithmetic, the eigenbasis read where a forecast carries one -- and `evaluate` a loop over it | 222-253, 263, 460-461, 476-563, 571-666 |
| `src/praxis/midoffice/model/covariance.py` | `HorizonCovariance.correlation_lags` (`_correlation_estimator`), the ratio's matrix shared only where the lags agree, the method's label, the docstring | 251-255, 262, 284-293, 300-305, 323-327, 369, 407-408 |
| `scripts/run_sp500_model.py` | ruling 1: `HORIZON` 1; ruling 2: the horizon line names the model's horizon or a reporting view, with its caveat; item 1: `--eigen-adjustment`, `--eigen-draws`, their checks and the Model section's line; items 2-4: `--bias-eigen-scales` (`scale_list`), the bias section's one pass feeding a `BiasTest` per scale, the refit's time with the simulation, `_report_eigen` (`EIGEN_CRITERION`, `_eigen_reading`), `_bias_arrays` and `_save_bias` with the scales' arrays and the simulation; item 5: `--correlation-lags`, its check, the estimator and its horizon line; the imports | 42, 68, 110-115, 139-149, 847-858, 925-954, 1062-1066, 1204-1334, 1343-1344, 1453-1457, 1473-1487, 1503-1518, 1763-1806, 1818, 1838-1839, 1853-1865 |
| `tests/midoffice/test_eigen.py` | new: 6 tests | new, 117 lines |
| `tests/midoffice/test_crosssectional.py` | 3 new tests | 2061-2130 |
| `tests/midoffice/test_bias.py` | 3 new tests; the imports | 10, 18, 274-356 |
| `tests/midoffice/test_covariance.py` | 1 new test | 444-477 |
| `docs/FACTOR_MODEL.md` | the correlations' own lags; the horizon one day and a longer one a reporting view; a new section, "The eigenfactor risk adjustment"; the guard; the adjustment and the reporting view measured | 259-266, 319-328, 339-382, 1254-1256, 1343-1380 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice13_eigenvalue_adjustment.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice13_eigenvalue_adjustment.md` | new: this retro | new |

Written outside the tracked files: only `claude/scratch/midoffice13/` (gitignored). Nothing in
`data/` was written, overwritten or deleted; the cache gained nothing.

---

## Rulings 1 and 2: one day, and the reporting view named

- **Ruling 1:** `HORIZON` is 1, the factor covariance the one-day covariance (`--horizon 1`'s
  lags: none). Measured (`r1_compare.txt`): the new defaults (R1) equal midoffice12's
  `--horizon 1` capture in all 21 fields, 6 qualifications and 162 report lines. The bias run is
  below: equal to H1 to the bit, the guard on.
- **Ruling 2:** the report's horizon line says which it is -- "the horizon: 1 trading day, the
  model's horizon -- ..." at the default; at another horizon "a reporting view, not the model's
  horizon of one day: broad portfolios' risk over a holding period of 21 days. Its gain over the
  one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025
  shock in it, and is re-tested as data accumulates -- ...". Measured (`r2_compare.txt`): at one
  day and at 21, each equal to the run before in all 21 fields, one line changed.
- **Rulings 3 to 6:** the candidate is item 5; the eigenvalue adjustment is items 1 to 4; the
  variant stays an option, off; the history stays 63. Nothing else to build.

## Item 1: the eigenfactor risk adjustment, built

- **`model/eigen.py`** (new): `eigen_bias(covariance, estimator, n_periods, draws, seed)` takes
  the day's covariance as the truth, draws `draws` histories of `n_periods` independent normal
  periods from it, re-estimates each by calling the estimator itself -- so with its weights, the
  two half-lives and the lags -- diagonalises each, and returns `EigenBias`: the covariance's
  eigenvalues (largest first) and eigenvectors (signed as `bias.eigenportfolios` signs them),
  lambda by rank, its Monte Carlo standard error, the draws, the periods, a count of
  draw-ranks whose estimated eigenvalue was not positive (none here), the seconds.
  `EigenBias.adjusted(a)` is `U0 diag(gamma^2 D0) U0'`, gamma = a (lambda - 1) + 1, and at a = 0
  the covariance as given, to the bit; `basis(a)` the adjusted eigenvalues in the unadjusted
  order, so rank k stays one eigenfactor at every a. Seeded (`EIGEN_SEED`), `EIGEN_DRAWS` 1,000.
- **The estimator's option** (`estimate_cross_sectional_model(eigen_adjustment=, eigen_draws=,
  eigen_seed=)`, the run's `--eigen-adjustment A`, `--eigen-draws M`), off: the simulation on the
  estimate before annualising, then the eigenvalues scaled unless a = 0. The noise shares stay
  the estimate's; with the sampling-error correction it is refused, the two being alternative
  remedies with no order chosen between them. The notes carry `eigen_adjustment` and the
  `EigenBias`; a qualification names the scales applied; the run's Model section prints lambda
  at the largest and the smallest eigenvalue, its range, its standard error and the seconds.
- **The run's bias section tests several a in one pass.** `bias.evaluate` became `BiasTest`, the
  same arithmetic one forecast at a time (`add`, `result`), and `evaluate` a loop over it, so one
  pass of refits can feed several tests. `--bias-eigen-scales A,A` simulates the bias on every
  refit -- at a = 0 if the model scales nothing itself -- and each scale gets its own test,
  `RiskForecast.from_model(model, securities, eigen_scale=a)` reading the adjusted covariance
  and its eigenbasis from the model's own simulation. The report adds the simulation, every
  scale at one day over the whole test and each half, and a chosen on the first half by a
  criterion fixed in the code before the runs (`EIGEN_CRITERION`: half the eigenportfolios' mean
  abs(bias - 1) over every rank, half the two minimum-variance portfolios'). The `.npz` keeps
  each scale's arrays keyed `_a<scale>` but its members' outcomes -- the model's, checked equal
  before saving -- and the simulation by date.
- **Measured on the model** (`i1_compare.txt`): `--eigen-adjustment 0` equals the rulings' run
  (R2) in all 21 fields, the report one line longer: "lambda ... 1.002 at the largest eigenvalue,
  1.041 at the smallest, 0.983 to 1.071 over the ranks, its Monte Carlo standard error at most
  0.0012", 0.7 s. At a = 1 and 3 only the covariance moves (by at most 5.0e-4 and 1.5e-3 of
  annual variance), each eigenfactor's volatility multiplied by 0.983 to 1.071 and 0.950 to
  1.212; the noise shares, factor returns and specific risks are the estimate's.
- **Measured on the bias run, in miniature** (a smoke run of the first 70 forecasts): the default
  run equals midoffice12's H1 over them in every array to the bit; the run with
  `--bias-eigen-scales 1,2,3,4` equals the default in every array it shares with it, so the
  simulation at a = 0 leaves the model's test alone.

## Item 4: M, and the time per refit

The probe (`probe_eigen.txt`), on the run's own factor returns at 252, 600 and 938 periods, M up
to 4,000: the Monte Carlo standard error of every rank's lambda at M = 1,000 is at most 0.0019 (at
252 periods; 0.0012 at 938), and lambda at 1,000 is within 0.0030 of lambda at 4,000 at every
rank -- stable to 0.01 by a margin of three. At 500 draws the gap was 0.0044, at 250 0.0088, so
1,000 is the default. A draw costs 0.34 to 0.64 ms -- the estimator and an eigendecomposition --
so a refit's simulation about half a second.

## Ruling 1's bias run: the new defaults equal H1 (`equal_D.txt`)

The defaults' run (D: `--bias --bias-horizons 1,5,11,21`, the guard on) against midoffice12's
H1 over the 686 forecast dates: **every one of the 24 arrays equal to the bit** -- the
portfolios', the members' and the eigenportfolios' daily forecasts and outcomes, their 5, 11 and
21-day outcomes, the PSD repairs (none), the minimum-variance sizes, the caps. **The guard
passed** on the five dates (2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02). Its
model report equals H1's line for line but the horizon line, now "the horizon: 1 trading day,
the model's horizon -- ..." (`report_D_vs_H1.txt`). 686 refits in 7.7 minutes, 0.67 s each; the
run 11.6 minutes with the guard.

## Items 2 and 3: the adjustment at a = 1, 2, 3, 4, measured

E: the defaults with `--bias-eigen-scales 1,2,3,4`, the guard on (it passed, the simulation on
in both its fits). **Its model test equals D's to the bit in every array** (`equal_E.txt`): at
a = 0 the simulation leaves the model alone. 686 refits in 14.3 minutes, **1.25 s a refit**, of
which the simulation 0.49 s (0.31 to 0.72, growing with the history) and the rest the refit and
five tests; D's refits alone are 0.67 s.

**The simulation over the test** (`compare_eigen.txt`): lambda at rank 1 is 1.002; ranks 2, 3,
4, 6 and 10 sit below one (0.986, 0.977, 0.995, 0.991, 0.992 at the mean; rank 3 down to
0.939); from rank 11 it rises to 1.06 to 1.09 at ranks 18 to 25, the first half higher than the
second (rank 21 1.087 against 1.037, the history shorter). The Monte Carlo standard error is at
most 0.0019 on any rank and date, 0.0013 at the median date's worst rank. gamma at a = 2 runs
0.879 to 1.345, about 1.13 at the smallest five; at a = 4, 0.757 to 1.690.

**At one day, the whole test** (686 forecasts):

| | a = 0 (the model) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| eigenportfolios 1-5, mean | 1.030 | 1.033 | 1.037 | 1.041 | 1.047 |
| eigenportfolios 6-10 | 1.082 | 1.085 | 1.089 | 1.094 | 1.099 |
| eigenportfolios 11-15 | 1.114 | 1.092 | 1.072 | 1.052 | 1.034 |
| eigenportfolios 16-20 | 1.216 | 1.144 | 1.081 | 1.026 | 0.976 |
| eigenportfolios 21-25 | 1.194 | 1.122 | 1.060 | 1.005 | 0.956 |
| eigenportfolios, mean abs(bias - 1) | 0.129 | 0.096 | 0.068 | 0.050 | 0.054 |
| minimum variance, long-only / unconstrained | 1.167 / 1.220 | 1.158 / 1.214 | 1.152 / 1.211 | 1.147 / 1.210 | 1.144 / 1.211 |
| the criterion | 0.161 | 0.141 | 0.124 | 0.114 | 0.116 |
| the index, cap-weighted / equally weighted | 1.015 / 0.965 | 1.013 / 0.955 | 1.011 / 0.945 | 1.008 / 0.935 | 1.006 / 0.924 |
| smallest 85 / largest 50 | 0.956 / 0.996 | 0.942 / 0.993 | 0.928 / 0.991 | 0.913 / 0.988 | 0.899 / 0.985 |
| members: median / mean abs(bias - 1) | 1.003 / 0.090 | 0.998 / 0.090 | 0.990 / 0.091 | 0.984 / 0.092 | 0.977 / 0.094 |
| industries, mean abs(bias - 1) | 0.046 | 0.047 | 0.051 | 0.054 | 0.058 |
| random long-only / long-short, median | 0.971 / 1.004 | 0.960 / 0.999 | 0.951 / 0.993 | 0.943 / 0.989 | 0.933 / 0.985 |

**a chosen on the first half** (343 forecasts, 2024-02-08 to 2025-06-23) by the criterion fixed
before the runs -- half the eigenportfolios' mean abs(bias - 1) over every rank, half the two
minimum-variance portfolios': **a = 2**, 0.1488, against the model's 0.1911 and a near tie with
a = 3's 0.1494. **On the second half** (2025-06-24 to 2026-11-02) a = 2 reads 0.112 against the
model's 0.138, the best there a = 4 (0.097): the choice holds out of the half it was made on.

- **It does what it is for on the small eigenfactors, in both halves.** Ranks 16 to 25 from 1.19
  to 1.22 down to 1.06 to 1.08 at a = 2 and to about one at a = 3; the eigenportfolios' mean
  abs(bias - 1) over every rank from 0.110 to 0.046 on the first half and 0.164 to 0.116 on the
  second at a = 2.
- **It moves some large ones the wrong way.** Where the simulation puts lambda below one it
  shrinks the eigenvalue, and the returns disagree at rank 3 -- already under-forecast, 1.045,
  it goes to 1.096 at a = 2 and 1.155 at a = 4 -- and at ranks 6, 7 and 10.
- **It barely moves the minimum-variance portfolios' bias** (unconstrained 1.220 to 1.211), and
  **it makes them riskier** (`minvar_realised.txt`): re-optimised on the adjusted covariance,
  each day's minimum-variance portfolio realises more, at every a, in both halves, with or
  without the one day below -- unconstrained 9.97% a year at a = 0, 10.13% at 2, 10.32% at 4
  (without that day); long-only 9.89%, 9.99%, 10.13%. Its forecast rises a little more than its
  realised risk, so its statistic improves a little; but by the plainest test of a covariance for
  an optimiser -- the realised risk of its minimum-variance portfolio -- the adjusted covariance
  is worse. The optimiser leaves the directions whose variance went up for the ones whose
  variance went down, rank 3 among them, which were under-forecast already.
- **It does not leave the broad portfolios alone** (item 3): their forecasts rise with a -- the
  equally weighted index 0.965 to 0.945 at a = 2, just outside the band (0.946), the smallest 85
  0.956 to 0.928, further outside; the members' median 1.003 to 0.990; the industries 0.046 to
  0.051 (the thin three 0.020 to 0.039); the random long-only portfolios 0.971 to 0.951. The
  cap-weighted index and the largest 50, carried by rank 1, barely move.

**One day dominates the minimum-variance statistics, and it is a data error.** The forecast of
2026-09-30 realised -13.0 sigma for the long-only portfolio and -13.9 sigma for the
unconstrained one (`diagnose_day.txt`): both held 9.4% of CTVA, whose 2026-10-01 total return is
-83.8% (-41.9 of its own sd) -- its raw and adjusted prices 77.65 to 12.57, its share count
unchanged, no split and no event in the cache, and it leaves the index on 2026-10-06: Corteva's
separation, a spin-off whose distribution the panel's total return does not carry. Every 7th
forecast date's statistic for the long-only portfolio runs 0.87 to 1.80 by the starting offset,
the high one the sample that holds that day. Without that one forecast (a sensitivity, not a
test; `compare_eigen.txt`):

| at one day, without the forecast of 2026-09-30 | a = 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| the whole test: minimum variance, long-only / unconstrained | 1.055 / 1.097 | 1.044 / 1.087 | 1.035 / 1.081 | 1.029 / 1.077 | 1.024 / 1.075 |
| the whole test: the criterion | 0.103 | 0.081 | 0.063 | 0.051 | 0.052 |
| the second half: minimum variance | 0.822 / 0.853 | 0.812 / 0.844 | 0.804 / 0.838 | 0.796 / 0.834 | 0.790 / 0.832 |
| the second half: the criterion | 0.163 | 0.155 | 0.147 | 0.140 | 0.137 |

So the optimiser's portfolios are under-forecast by 5 to 10% over the test, not 17 to 22%, and
by half they are the regime's -- 1.25 and 1.30 in the first half with the April 2025 shock, 0.82
and 0.85 in the calm second -- which the adjustment moves by 0.01 to 0.02 either way.
**The scan** (`scan_extreme.txt`) found twelve member-days past +/-35% in the test window; two
were read against the panel and the cache. CTVA's has the spin-off's signature (a price that
stays down, an unchanged count, no record, an index exit days later); MRNA's +177% on
2026-08-19 (62.96 to 174.38, then 133.32, its count unchanged, no record) looks like news. The
other ten -- GL, DXCM, WST, CNC, ALGN, TTD, SKYD, ORCL, SNPS, FISV -- were not checked one by
one; most are earnings or guidance days of 2024-25 on the public record. The factor returns of
2026-10-01 show no outlier (the largest move Wholesale's, 2.5 of its prior year's sd).

**At 21 days** (the reporting horizon, for what the adjustment does there): the broad portfolios
read further below one at every a (the equally weighted index 0.805 to 0.788 at a = 2), the
minimum-variance portfolios a little nearer (long-only 1.005 to 0.987), the smallest
eigenportfolios 1.159 to 1.030.

## Item 5: the reporting view's candidates at 21 days

- **Built:** `HorizonCovariance.correlation_lags` (the run's `--correlation-lags L`): the
  correlations' own Newey-West lags, 0 making them the long window's one-period correlations
  while the volatilities keep the horizon's. With variance-ratio volatilities the ratio keeps the
  volatilities' lags on its own window (the correlations' matrix is no longer shared with it).
  Its diagonal is the volatility estimator's, so it keeps a weighted noise share. A test pins
  each part, and that `correlation_lags` equal to the lags is the plain split to the bit.
  Measured on the model (`i5_compare.txt`): `--horizon 21 --correlation-lags 20` equals
  `--horizon 21` in every field and line; `--correlation-lags 0` moves the correlations alone,
  the diagonal the same to 2e-17; with `--variance-ratio-half-life 504` the volatilities too.
- **The check:** the 21-day model rerun on this code (H21) equals midoffice12's H21 to the bit in
  every array (`equal_H21.txt`).

**Read at 21 days and at one** (`compare_eigen.txt`; D the one-day model, H21 the 21-day model,
V21N Newey-West at 20 lags on the volatilities' 90 days and one-day correlations, V21R the
variance-ratio variant's volatilities and one-day correlations):

| | D | H21 | V21N | V21R |
|---|---|---|---|---|
| at 21 days: the index, cap-weighted / equally weighted | 0.860 / 0.805 | 1.006 / 0.892 | 0.975 / 0.905 | 0.942 / 0.856 |
| at 21 days: smallest 85 / largest 50 | 0.775 / 0.873 | 0.866 / 1.019 | 0.859 / 0.974 | 0.817 / 0.943 |
| at 21 days: the four broad, mean abs(bias - 1) | 0.172 | 0.067 | 0.072 | 0.111 |
| at 21 days: minimum variance, long-only / unconstrained | 1.005 / 1.045 | 1.177 / 1.308 | 1.118 / 1.224 | 1.056 / 1.149 |
| at 21 days: eigenportfolios 21-25 / 1-5 | 1.159 / 0.913 | 2.686 / 0.930 | 1.476 / 1.043 | 1.291 / 0.976 |
| at 21 days: industries; members, median | 0.079; 0.938 | 0.054; 0.981 | 0.061; 0.990 | 0.062; 0.975 |
| at 1 day: the index, cap-weighted / equally weighted | 1.015 / 0.965 | 1.148 / 1.042 | 1.123 / 1.063 | 1.095 / 1.016 |
| at 1 day: minimum variance, long-only / unconstrained | 1.167 / 1.220 | 1.360 / 1.629 | 1.347 / 1.503 | 1.253 / 1.375 |
| at 1 day: eigenportfolios 21-25 | 1.194 | 2.670 | 1.454 | 1.318 |
| the index's forecast level against D (realised ratio at 21: 0.702) | 1 | 0.682 | 0.744 | 0.809 |
| the index's forecast, daily change in log volatility, sd | 0.0109 | 0.0248 | 0.0213 | 0.0114 |

- **The lags on the volatilities alone keep the 21-day model's broad portfolios and give back
  most of what the correlations' lags cost.** V21N reads the four broad portfolios as the 21-day
  model does (mean abs(bias - 1) 0.072 against 0.067; the one-day model scaled 0.172), the
  equally weighted index nearer one (0.905 against 0.892), while its smallest eigenportfolios
  fall from 2.69 to 1.48 and its unconstrained minimum-variance portfolio from 1.31 to 1.22. The
  correlations' 20 lags bought the broad portfolios nothing; they cost the small eigen-directions
  a factor of two.
- **The variance-ratio volatilities go further toward the one-day model:** the optimiser's
  portfolios 1.06 and 1.15, the smallest eigenportfolios 1.29, and the forecast as steady from day
  to day as the one-day model's (0.0114) -- but their two-year ratio is less mean-reverting than
  2024-26 was (the index at 0.809 of the one-day variance against a realised 0.702), so the broad
  portfolios read further below one (0.942, 0.856, 0.817, 0.943), the view's own job done worse.
- At one day every 21-day view is wrong as a daily model should not be (the index 1.10 to 1.15),
  as midoffice12 found: a reporting view, not the model.

---

## What I would choose, and why

1. **The eigenfactor risk adjustment stays off,** though by the brief's own test a = 2 is the
   choice. Chosen on the first half (0.149 against the model's 0.191), it holds on the second
   (0.112 against 0.138), and it does what it is for on the small eigenfactors: ranks 16 to 25 from
   about 1.2 to about 1.07. But it is not ready to be the model's covariance:
   - **the optimiser's portfolios, rebuilt on it, realise more risk** at every a and in both
     halves -- unconstrained 9.97% a year to 10.13% at a = 2 without the CTVA day, long-only 9.89%
     to 9.99% -- which is the plainest test of a covariance an optimiser uses, and it fails it;
   - **the formula shrinks the eigenvalues the simulation calls overestimated**, and at rank 3 the
     returns disagree (1.045 to 1.096): the optimiser moves into those directions;
   - **it does not leave the broad portfolios alone**: the smallest 85 out of the band (0.956 to
     0.928), the equally weighted index just out (0.945), the members' median 1.003 to 0.990;
   - **the minimum-variance statistic it was to fix barely moves** (1.220 to 1.211; 1.097 to
     1.081 without the one day), its bias the regime's -- 1.25 to 1.30 in the first half, 0.82 to
     0.85 in the second -- and, over the whole test, one data error's.
   The simulation explains about a third of the small eigenvalues' realised bias (1.19 to 1.12 at
   a = 1); the rest is not the sampling error of a stationary normal history.
2. **What to measure instead, in this order:**
   - **the CTVA spin-off first** -- the panel's 2026-10-01 total return should carry the
     distributed shares' value; until it does, any test window that holds the day has its
     optimiser's statistics dominated by one -13 sigma outcome (and a scan for raw price gaps with
     unchanged share counts and no record would find any other);
   - **the adjustment one-sided**, gamma = max(1, a (lambda - 1) + 1), so no eigenvalue is shrunk,
     judged by the same test and by the minimum-variance portfolio's realised risk;
   - **a volatility regime adjustment** (USE4's other step): the regime moves every portfolio far
     more than the eigenvalues do -- the index 1.22 in the first half, 0.75 in the second -- and a
     scale fitted to the recent cross-section of standardised returns is what addresses that.
3. **The reporting view at 21 days: the lags on the volatilities alone, V21N** (`--horizon 21
   --correlation-lags 0`). It keeps the 21-day model's accuracy on broad portfolios, the view's
   one job, and halves what the lags cost everything else; the correlations' lags add nothing for
   the broad portfolios. V21R is the better forecast for the optimiser but the worse view: if the
   optimiser's portfolios are what is reported, the one-day model already reads them best.
4. **The model stays one day,** the variance-ratio variant an option, off, the history 63, as
   ruled.

---

## Key decisions

1. **The adjustment lives in the estimator, simulated through the estimator itself.** Each draw
   calls `covariance_estimator.estimate`, so "the same estimator" holds by construction -- weights,
   half-lives, lags -- rather than by a reimplementation; the cost, half a second a refit at
   1,000 draws, was affordable.
2. **M = 1,000**, from the probe: the standard error at most 0.0019 and within 0.003 of 4,000
   draws -- stable to 0.01 by a factor of three.
3. **a = 1, 2, 3, 4, not USE4's 1.2:** the probe showed the simulation alone a third of the small
   eigenvalues' realised bias (lambda about 1.06 against a realised 1.19), so values near 1.2 could
   not be told from 1; 2 to 4 bracket the scale that would close it.
4. **The criterion was fixed before the runs** -- half the eigenportfolios' mean abs(bias - 1)
   over every rank, half the two minimum-variance portfolios' -- and a chosen by it on the first
   half alone, the second half reported beside it.
5. **The eigenportfolios are judged in the unadjusted covariance's eigenbasis,** so rank k is one
   eigenfactor at every a; re-sorting the adjusted eigenvalues would have swapped near-equal ranks.
6. **At a = 0 the simulation runs and the covariance is untouched to the bit,** so one bias run
   tests the model and every scale side by side, and its model test is the defaults' (checked).
7. **`evaluate` became `BiasTest`,** the same arithmetic one forecast at a time; the defaults' run
   equals H1 to the bit through it.
8. **The adjustment and the sampling-error correction are refused together:** two remedies for
   one error, with no order between them chosen.
9. **The CTVA day is a sensitivity, not an exclusion:** every statistic of record is the whole
   test's; the run without that forecast is reported beside it, labelled, and the data fix is
   left for a piece of its own (nothing in `data/` is changed without Jeff).
10. **The guard ran on D and on E** -- D as ruled, E because the eigen simulation is a new
    estimator option.
11. **BLAS on one thread** in every run, so the runs compare to the bit.

## Decisions left for Jeff

1. **The eigenfactor risk adjustment:** off, an option (my choice); or on at a = 2, the scale
   its test chose.
2. **The CTVA spin-off of 2026-10-01:** a data piece to carry the distribution into the total
   return, with a scan for others (my choice, first); or the test windows left as they are.
3. **What comes after:** the adjustment one-sided, and a volatility regime adjustment (my
   choice, in that order after the data fix); or neither.
4. **The reporting view at 21 days:** the lags on the volatilities alone (`--correlation-lags 0`)
   as its default (my choice); or the 21-day model as it is.

---

## Tests

**The suite:** 2,356 passed, 1 skipped, 19 warnings (101.4 s in the final checks), with
`--ignore=tests/test_market_data.py`. Before this piece 2,343: 13 new tests; the old tests of
`evaluate` pass through it unchanged.

**New, by file:**

`tests/midoffice/test_eigen.py` (new), 6:

- `test_the_bias_is_one_on_a_long_history_and_the_classic_pattern_on_a_short_one` -- lambda
  within 0.03 of one on 5,000 periods; on 30 periods of 10 factors below one at the largest
  eigenvalue (0.92) and above 1.3 at the smallest (1.76);
- `test_the_simulation_is_seeded_and_its_error_is_its_spread` -- the same draws on the same
  inputs; another seed within 4.5 standard errors at every rank;
- `test_the_adjusted_covariance_scales_each_eigenvalue_and_keeps_its_eigenvectors` -- each
  eigenvector of the day's covariance one of the adjusted one's, its eigenvalue times gamma
  squared; a = 0 the covariance itself;
- `test_the_adjustment_brings_the_estimates_eigenfactors_back_toward_their_true_risk` --
  Menchero, Wang and Orr's check in miniature: 60 estimates from 60 periods of a known
  covariance, the true variance of their eigenfactors over the estimate up to 1.61 at the
  smallest rank, every rank within 0.1 of one after the adjustment at a = 1 simulated from the
  estimate alone (it reads 0.99 to 1.06);
- `test_the_simulation_re_estimates_with_the_estimator_it_is_given` -- a split estimator's bias
  not a single half-life's;
- `test_bad_inputs_are_refused`.

`tests/midoffice/test_crosssectional.py`, 3:

- `test_the_eigen_adjustment_is_off_by_default_and_at_zero_simulates_and_changes_nothing` -- at
  a = 0 the covariance to the bit, the simulation on the estimate before annualising, no
  qualification;
- `test_the_eigen_adjustment_scales_the_eigenvalues_and_leaves_the_noise_shares_alone` -- the
  covariance the adjusted one, each eigenvalue gamma squared times its own; the noise shares and
  factor returns the estimate's; the qualification; the refusals;
- `test_nothing_after_a_date_moves_the_eigen_adjusted_model_as_of_it` -- everything after the
  date redrawn, the adjusted model as of it and its simulated bias the same to the bit.

`tests/midoffice/test_bias.py`, 3:

- `test_tests_built_alike_draw_the_same_portfolios_and_evaluate_is_one_test_run`;
- `test_a_test_reads_the_eigenportfolios_from_a_forecasts_own_eigenbasis` -- a basis with four
  times the eigenvalues doubles the eigen forecasts and leaves the portfolios' alone;
- `test_a_forecast_at_another_eigen_scale_reads_the_models_simulation` -- the adjusted covariance
  and its basis from the model's own simulation; the model's own scale its covariance; a model
  adjusted itself carries its basis; no simulation, refused.

`tests/midoffice/test_covariance.py`, 1:

- `test_correlation_lags_give_the_correlations_their_own_and_the_volatilities_keep_theirs` --
  the volatilities Newey-West, the correlations one-period, the ratio at the volatilities' lags,
  the plain split to the bit when the lags agree, the diagonal still a quadratic form.

**Hygiene:** every changed file parses, is ASCII and keeps its line endings (`final_checks.txt`).
**ruff:** clean over every changed Python file.

**The final code, re-run** (`final_checks.txt`): the defaults (FD) equal F1 in all 21 fields,
6 qualifications and 162 report lines, and so R2, R1 and midoffice12's `--horizon 1` capture;
`--horizon 11` (FD11) equals midoffice12's final defaults (its FD) in all 21 fields and 6
qualifications, the report differing only in the horizon line, which now names it a reporting
view.

---

## Failures and debugging trail

- **The report patch's own line endings were escaped twice.** `patch_eigen_report.py` wrote
  `"\\r\\n"` in its own code, outside the replacement text, so its CRLF check failed before it
  wrote anything; a heredoc meant to fix it mangled the backslashes again (the state file's
  working note), and the Edit tool fixed it.
- **`EigenBias` compared numpy arrays in a dataclass `==`** (an ambiguous truth value, found by
  its own test); it compares by identity now.
- **Two import-order findings** (ruff I001), from the new imports; both fixed.
- **A diagnostic's sample led to the data error.** Splitting the minimum-variance portfolios'
  risk into factor and specific parts on every 7th date gave a statistic of 0.89 where the whole
  test reads 1.17. The seven offsets read 0.87 to 1.80, and the largest outcomes showed one day at
  -13 sigma: CTVA's unadjusted spin-off. The decomposition on that sample is left unreported, the
  sample being unrepresentative; the day is reported instead.
- **The runs:** none failed. The runner ran the five one at a time from 12:02 to 13:02 EST, each
  started with 4.0 to 6.6 GB available -- D 11.6 minutes with the guard, E 18.6 with the guard and
  the simulation, V21N 9.7, V21R 9.2, H21 10.5; no retry, no wait, and the memory guard was not
  reached (`bias/runner_log.txt`).

## Problems found, not fixed

1. **CTVA's 2026-10-01 total return is -83.8%:** Corteva's separation, a spin-off whose
   distribution the panel does not carry. It dominates every minimum-variance statistic over a
   window holding that day; CTVA's own member statistic too.
2. **The eigen adjustment as briefed makes the optimiser's portfolios riskier** and moves the
   broad portfolios; one-sided, or with a regime adjustment, are untested.
3. **The regime dominates the halves:** the index 1.22 in the first, 0.75 in the second; no
   adjustment here addresses it.
4. **The optimiser's portfolios stay under-forecast at one day,** 1.055 and 1.097 without the
   CTVA day.
5. **Six ruff findings in untouched files,** at HEAD, and **`tests/test_market_data.py`'s
   `../core_repo/src`,** as before.

## Scratch

`claude/scratch/midoffice13/` (gitignored):

- **Harnesses:** `capture.py`, `compare.py`, `hygiene.py`, `smoke_bias.py` (midoffice12's);
  `runner.py`, its log and done marker in `bias/`; `patch_biastest.py`, `patch_eigen_report.py`
  (the bias module's and the run's code replaced between markers, line endings kept);
  `final_checks.sh`.
- **Measures:** `probe_eigen.py` (the simulation's cost and stability), `probe_lambda.py`
  (lambda across the test against H1's realised bias), `probe_menchero.py` (the test's numbers),
  `equal_npz.py` (two runs bit for bit), `compare_eigen.py`, `diagnose_minvar.py` (the sample
  that led to the CTVA day), `diagnose_day.py` (the day itself), `scan_extreme.py` (the test
  window's extreme member-days, CTVA's and MRNA's records).
- **The retro's own scripts:** `fill_retro.py`, `fix_head.py`, `fill_final.py`, `reflow.py`,
  `append_appendix.py`.
- **Captures:** `r1/` (R1), `r2/` (R2, R2h21), `i1/` (E0, E1, E3), `i5/` (F1, C20, V21N, V21R),
  `diag/` (DI and its inputs), `final/` (FD, FD11), each a report and a pickle; `smoke/` (the
  smoke runs and their checks).
- **Bias runs, `bias/`:** each run's report, progress and `.npz` -- D, E, V21N, V21R, H21.
- **Outputs:** `probe_eigen.txt`, `probe_lambda.txt`, `r1_compare.txt`, `r2_compare.txt`,
  `i1_compare.txt`, `i5_compare.txt`, `equal_D.txt`, `equal_E.txt`, `equal_H21.txt`,
  `report_D_vs_H1.txt`, `compare_eigen.txt`, `minvar_realised.txt`, `diagnose_day.txt`,
  `scan_extreme.txt`, `final_checks.txt`.

The key ones are copied into the appendix. Nothing in `data/` was written, overwritten or
deleted; it was read, CTVA's and MRNA's prices and records among it.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

## Open items for Chat

- **The four decisions above,** the CTVA data fix first among them.
- **The eigen adjustment's next form** -- one-sided, or after a regime adjustment -- judged by the
  same test and by the minimum-variance portfolio's realised risk, the test this one failed.
- **The bias statistics are standing,** with `--bias-eigen-scales` for any covariance-side scale;
  a later candidate is one run against the defaults, through the runner.

---

## Appendix: outputs, verbatim

Each output as its script wrote it, from `claude/scratch/midoffice13/`.

### probe_eigen.txt -- M: the simulation's stability and cost on the run's own factor returns

```text
T = 252: 0.34 ms per draw (estimate + eigh), M = 4000
  lambda by rank, largest first: 1.000 0.983 0.942 0.996 1.056 1.017 0.961 1.007 1.022 1.002 1.026 1.030 1.049 1.059 1.048 1.069 1.097 1.120 1.134 1.171 1.125 1.079 1.151 1.116 1.140
  its Monte Carlo standard error: 0.0008 0.0007 0.0005 0.0006 0.0006 0.0007 0.0006 0.0006 0.0007 0.0007 0.0007 0.0007 0.0007 0.0008 0.0008 0.0007 0.0007 0.0008 0.0008 0.0008 0.0008 0.0008 0.0008 0.0009 0.0009
  M =  250: largest |lambda_M - lambda_4000| 0.0088; standard error at M, the largest rank's 0.0037
  M =  500: largest |lambda_M - lambda_4000| 0.0044; standard error at M, the largest rank's 0.0026
  M = 1000: largest |lambda_M - lambda_4000| 0.0030; standard error at M, the largest rank's 0.0019
  M = 2000: largest |lambda_M - lambda_4000| 0.0017; standard error at M, the largest rank's 0.0013
T = 600: 0.48 ms per draw (estimate + eigh), M = 4000
  lambda by rank, largest first: 1.001 0.986 0.991 0.989 1.024 0.986 1.000 1.022 0.999 0.981 1.009 1.019 1.033 1.012 1.034 1.042 1.036 1.061 1.075 1.073 1.039 1.071 1.054 1.050 1.055
  its Monte Carlo standard error: 0.0006 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0006 0.0005 0.0005 0.0006 0.0006 0.0006 0.0006 0.0006 0.0006 0.0006 0.0006 0.0006 0.0006
  M =  250: largest |lambda_M - lambda_4000| 0.0042; standard error at M, the largest rank's 0.0026
  M =  500: largest |lambda_M - lambda_4000| 0.0035; standard error at M, the largest rank's 0.0018
  M = 1000: largest |lambda_M - lambda_4000| 0.0020; standard error at M, the largest rank's 0.0013
  M = 2000: largest |lambda_M - lambda_4000| 0.0011; standard error at M, the largest rank's 0.0009
T = 938: 0.64 ms per draw (estimate + eigh), M = 4000
  lambda by rank, largest first: 1.001 0.986 0.986 1.007 1.007 1.005 1.000 0.983 1.020 1.010 1.011 0.984 0.997 1.016 1.023 1.024 1.037 1.059 1.060 1.036 1.056 1.020 1.070 1.037 1.040
  its Monte Carlo standard error: 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0004 0.0005 0.0005 0.0005 0.0005 0.0004 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0006 0.0006
  M =  250: largest |lambda_M - lambda_4000| 0.0052; standard error at M, the largest rank's 0.0023
  M =  500: largest |lambda_M - lambda_4000| 0.0038; standard error at M, the largest rank's 0.0017
  M = 1000: largest |lambda_M - lambda_4000| 0.0025; standard error at M, the largest rank's 0.0012
  M = 2000: largest |lambda_M - lambda_4000| 0.0016; standard error at M, the largest rank's 0.0008
```

### probe_lambda.txt -- lambda across the test's dates beside the one-day model's realised eigenportfolio bias

```text
69 dates, T from 252 to 932, M = 1000
rank  lambda mean     min     max  realised H1  a to one
   1        1.000   0.998   1.002        1.027       nan
   2        0.985   0.975   0.993        0.975      1.68
   3        0.978   0.942   0.994        1.045     -2.02
   4        0.996   0.979   1.012        1.048    -11.51
   5        1.032   1.008   1.059        1.055      1.70
   6        0.993   0.976   1.020        1.036     -4.96
   7        1.000   0.958   1.023        1.090       nan
   8        1.005   0.985   1.026        1.079     14.86
   9        1.003   0.968   1.034        1.131     42.61
  10        0.993   0.975   1.009        1.075    -11.00
  11        1.011   0.999   1.028        1.104      9.33
  12        1.019   0.986   1.036        1.098      5.28
  13        1.024   0.992   1.054        1.110      4.58
  14        1.023   0.989   1.062        1.103      4.44
  15        1.031   1.007   1.059        1.156      5.02
  16        1.043   1.019   1.073        1.168      3.88
  17        1.050   1.025   1.098        1.160      3.18
  18        1.065   1.038   1.120        1.247      3.80
  19        1.088   1.057   1.153        1.285      3.24
  20        1.076   1.032   1.172        1.219      2.88
  21        1.062   1.024   1.138        1.194      3.12
  22        1.058   1.022   1.084        1.193      3.31
  23        1.074   1.042   1.152        1.226      3.06
  24        1.061   1.038   1.113        1.233      3.84
  25        1.067   1.041   1.141        1.125      1.86
ranks 21-25: lambda mean 1.064, realised 1.194, a to one 3.01
```

### r1_compare.txt and r2_compare.txt -- rulings 1 and 2 on the model

```text
claude/scratch/midoffice12/i1/H1 against claude/scratch/midoffice13/r1/R1: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
[...]
claude/scratch/midoffice13/r1/R1 against claude/scratch/midoffice13/r2/R2: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 2 differing line(s)
    -  the horizon: 1 trading day -- the factor covariance the one-day covariance, with no Newey-West lags, and the specific risk one day's variance
    +  the horizon: 1 trading day, the model's horizon -- the factor covariance the one-day covariance, with no Newey-West lags, and the specific risk one day's variance
claude/scratch/midoffice12/i1/H21 against claude/scratch/midoffice13/r2/R2h21: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 2 differing line(s)
    -  the horizon: 21 trading days -- the factor covariance Newey-West at 20 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 21-day returns; the specific risk one day's variance, with no lags, at every horizon
    +  the horizon: 21 trading days, a reporting view, not the model's horizon of one day: broad portfolios' risk over a holding period of 21 days. Its gain over the one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025 shock in it, and is re-tested as data accumulates -- the factor covariance Newey-West at 20 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 21-day returns; the specific risk one day's variance, with no lags, at every horizon
```

### i1_compare.txt -- the adjustment on the model at a = 0, 1, 3

```text
claude/scratch/midoffice13/r2/R2 against claude/scratch/midoffice13/i1/E0: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 163 lines; 1 differing line(s)
    +  the eigenfactor risk adjustment, a = 0 (simulated, nothing scaled): lambda by rank from 1000 histories simulated from the factor covariance over its 938 periods and re-estimated by its own estimator, in 0.7 s -- 1.002 at the largest eigenvalue, 1.041 at the smallest, 0.983 to 1.071 over the ranks, its Monte Carlo standard error at most 0.0012
claude/scratch/midoffice13/r2/R2 against claude/scratch/midoffice13/i1/E1: 21 fields compared; equal: 20; differing: covariance
  covariance: largest difference 4.961e-04
  qualifications: 6 and 7; differing: 3
  report: 162 and 164 lines; 56 differing line(s)
    +  the eigenfactor risk adjustment, a = 1: lambda by rank from 1000 histories simulated from the factor covariance over its 938 periods and re-estimated by its own estimator, in 0.7 s -- 1.002 at the largest eigenvalue, 1.041 at the smallest, 0.983 to 1.071 over the ranks, its Monte Carlo standard error at most 0.0012; each eigenfactor's volatility multiplied by 0.983 to 1.071
    -  smallest 85, equally weighted                -3.137             -1.808   10.99%
    -  largest 50, equally weighted                 +0.167             +0.052    0.57%
    +  smallest 85, equally weighted                -3.137             -1.808   11.29%
    +  largest 50, equally weighted                 +0.167             +0.052    0.59%
    -  Construction               22.22%     +24.76%    29.2%      38.7%      938          0      0
    -  Finance & Real Estate      17.38%     +18.11%    24.1%      37.8%      938          0      0
    -  Manufacturing              12.12%     +20.88%    16.4%      24.7%      938          0      0
    -  Mining                     21.20%     +14.94%    14.3%      21.8%      938          0      0
    -  Retail                     14.77%     +14.63%     8.4%      13.2%      938          0      0
    -  SIC20                      13.35%     +11.03%    27.0%      46.4%      938          0      0
    -  SIC28                      16.09%     +16.64%     6.8%      10.9%      938          0      0
    -  SIC35                      18.55%     +29.69%     5.4%       7.5%      938          0      0
    -  SIC36                      24.22%     +32.55%     3.4%       4.2%      938          0      0
    -  SIC37                      19.50%     +15.42%    10.2%      15.1%      938          0      0
    -  SIC38                      16.79%     +15.82%     8.1%      12.7%      938          0      0
    -  SIC49                      19.08%     +24.01%     7.8%      12.1%      938          0      0
    -  SIC60                      23.21%     +21.97%     6.6%       9.7%      938          0      0
    -  SIC62                      21.57%     +18.62%     8.7%      11.8%      938          0      0
    -  SIC63                      14.25%     +19.17%    15.2%      24.7%      938          0      0
    -  SIC67                      17.79%     +18.00%    11.9%      18.8%      938          0      0
    -  SIC73                      17.52%     +15.21%     3.2%       5.0%      938          0      0
    -  Services                   14.38%     +16.11%    21.9%      34.7%      938          0      0
    -  Transport & Utilities      16.21%     +15.46%    11.3%      18.0%      938          0      0
    -  Wholesale                  15.21%     +20.69%    39.0%      60.5%      938          0      0
    -  Volatility                 10.64%      +3.09%     2.7%       4.6%      938          0      0
    -  Momentum                    7.50%      +4.23%     3.4%       4.9%      938          0      0
    -  Size                        3.59%      +3.13%     8.2%      11.9%      938          0      0
    -  Size Nonlinearity           2.81%      -0.53%    31.6%      47.6%      938          0      0
    -  Trading Activity            5.31%      +2.48%    10.4%      15.1%      938          0      0
    +  Construction               22.20%     +24.76%    29.2%      38.7%      938          0      0
    +  Finance & Real Estate      17.41%     +18.11%    24.1%      37.8%      938          0      0
    +  Manufacturing              12.25%     +20.88%    16.4%      24.7%      938          0      0
    +  Mining                     21.14%     +14.94%    14.3%      21.8%      938          0      0
    +  Retail                     14.93%     +14.63%     8.4%      13.2%      938          0      0
    +  SIC20                      13.31%     +11.03%    27.0%      46.4%      938          0      0
    +  SIC28                      16.11%     +16.64%     6.8%      10.9%      938          0      0
    +  SIC35                      18.64%     +29.69%     5.4%       7.5%      938          0      0
    +  SIC36                      24.25%     +32.55%     3.4%       4.2%      938          0      0
    +  SIC37                      19.56%     +15.42%    10.2%      15.1%      938          0      0
    +  SIC38                      16.90%     +15.82%     8.1%      12.7%      938          0      0
    +  SIC49                      19.11%     +24.01%     7.8%      12.1%      938          0      0
    +  SIC60                      23.23%     +21.97%     6.6%       9.7%      938          0      0
    +  SIC62                      21.59%     +18.62%     8.7%      11.8%      938          0      0
    +  SIC63                      14.31%     +19.17%    15.2%      24.7%      938          0      0
    +  SIC67                      17.84%     +18.00%    11.9%      18.8%      938          0      0
    +  SIC73                      17.58%     +15.21%     3.2%       5.0%      938          0      0
    +  Services                   14.52%     +16.11%    21.9%      34.7%      938          0      0
    +  Transport & Utilities      16.28%     +15.46%    11.3%      18.0%      938          0      0
    +  Wholesale                  15.25%     +20.69%    39.0%      60.5%      938          0      0
    +  Volatility                 10.77%      +3.09%     2.7%       4.6%      938          0      0
    +  Momentum                    7.68%      +4.23%     3.4%       4.9%      938          0      0
    +  Size                        3.71%      +3.13%     8.2%      11.9%      938          0      0
    +  Size Nonlinearity           2.91%      -0.53%    31.6%      47.6%      938          0      0
    +  Trading Activity            5.45%      +2.48%    10.4%      15.1%      938          0      0
    +  - the factor covariance's eigenvalues are scaled by the eigenfactor risk adjustment, a = 1, from 1000 simulated histories: each eigenfactor's volatility multiplied by 0.983 to 1.071, the smallest eigenvalue's by 1.041
claude/scratch/midoffice13/r2/R2 against claude/scratch/midoffice13/i1/E3: 21 fields compared; equal: 20; differing: covariance
  covariance: largest difference 1.479e-03
  qualifications: 6 and 7; differing: 3
  report: 162 and 164 lines; 56 differing line(s)
    +  the eigenfactor risk adjustment, a = 3: lambda by rank from 1000 histories simulated from the factor covariance over its 938 periods and re-estimated by its own estimator, in 0.7 s -- 1.002 at the largest eigenvalue, 1.041 at the smallest, 0.983 to 1.071 over the ranks, its Monte Carlo standard error at most 0.0012; each eigenfactor's volatility multiplied by 0.950 to 1.212
    -  smallest 85, equally weighted                -3.137             -1.808   10.99%
    -  largest 50, equally weighted                 +0.167             +0.052    0.57%
    +  smallest 85, equally weighted                -3.137             -1.808   11.91%
    +  largest 50, equally weighted                 +0.167             +0.052    0.63%
    -  Construction               22.22%     +24.76%    29.2%      38.7%      938          0      0
    -  Finance & Real Estate      17.38%     +18.11%    24.1%      37.8%      938          0      0
    -  Manufacturing              12.12%     +20.88%    16.4%      24.7%      938          0      0
    -  Mining                     21.20%     +14.94%    14.3%      21.8%      938          0      0
    -  Retail                     14.77%     +14.63%     8.4%      13.2%      938          0      0
    -  SIC20                      13.35%     +11.03%    27.0%      46.4%      938          0      0
    -  SIC28                      16.09%     +16.64%     6.8%      10.9%      938          0      0
    -  SIC35                      18.55%     +29.69%     5.4%       7.5%      938          0      0
    -  SIC36                      24.22%     +32.55%     3.4%       4.2%      938          0      0
    -  SIC37                      19.50%     +15.42%    10.2%      15.1%      938          0      0
    -  SIC38                      16.79%     +15.82%     8.1%      12.7%      938          0      0
    -  SIC49                      19.08%     +24.01%     7.8%      12.1%      938          0      0
    -  SIC60                      23.21%     +21.97%     6.6%       9.7%      938          0      0
    -  SIC62                      21.57%     +18.62%     8.7%      11.8%      938          0      0
    -  SIC63                      14.25%     +19.17%    15.2%      24.7%      938          0      0
    -  SIC67                      17.79%     +18.00%    11.9%      18.8%      938          0      0
    -  SIC73                      17.52%     +15.21%     3.2%       5.0%      938          0      0
    -  Services                   14.38%     +16.11%    21.9%      34.7%      938          0      0
    -  Transport & Utilities      16.21%     +15.46%    11.3%      18.0%      938          0      0
    -  Wholesale                  15.21%     +20.69%    39.0%      60.5%      938          0      0
    -  Volatility                 10.64%      +3.09%     2.7%       4.6%      938          0      0
    -  Momentum                    7.50%      +4.23%     3.4%       4.9%      938          0      0
    -  Size                        3.59%      +3.13%     8.2%      11.9%      938          0      0
    -  Size Nonlinearity           2.81%      -0.53%    31.6%      47.6%      938          0      0
    -  Trading Activity            5.31%      +2.48%    10.4%      15.1%      938          0      0
    +  Construction               22.16%     +24.76%    29.2%      38.7%      938          0      0
    +  Finance & Real Estate      17.48%     +18.11%    24.1%      37.8%      938          0      0
    +  Manufacturing              12.52%     +20.88%    16.4%      24.7%      938          0      0
    +  Mining                     21.02%     +14.94%    14.3%      21.8%      938          0      0
    +  Retail                     15.26%     +14.63%     8.4%      13.2%      938          0      0
    +  SIC20                      13.24%     +11.03%    27.0%      46.4%      938          0      0
    +  SIC28                      16.15%     +16.64%     6.8%      10.9%      938          0      0
    +  SIC35                      18.83%     +29.69%     5.4%       7.5%      938          0      0
    +  SIC36                      24.31%     +32.55%     3.4%       4.2%      938          0      0
    +  SIC37                      19.67%     +15.42%    10.2%      15.1%      938          0      0
    +  SIC38                      17.11%     +15.82%     8.1%      12.7%      938          0      0
    +  SIC49                      19.18%     +24.01%     7.8%      12.1%      938          0      0
    +  SIC60                      23.28%     +21.97%     6.6%       9.7%      938          0      0
    +  SIC62                      21.65%     +18.62%     8.7%      11.8%      938          0      0
    +  SIC63                      14.44%     +19.17%    15.2%      24.7%      938          0      0
    +  SIC67                      17.95%     +18.00%    11.9%      18.8%      938          0      0
    +  SIC73                      17.71%     +15.21%     3.2%       5.0%      938          0      0
    +  Services                   14.83%     +16.11%    21.9%      34.7%      938          0      0
    +  Transport & Utilities      16.42%     +15.46%    11.3%      18.0%      938          0      0
    +  Wholesale                  15.34%     +20.69%    39.0%      60.5%      938          0      0
    +  Volatility                 11.04%      +3.09%     2.7%       4.6%      938          0      0
    +  Momentum                    8.06%      +4.23%     3.4%       4.9%      938          0      0
    +  Size                        3.93%      +3.13%     8.2%      11.9%      938          0      0
    +  Size Nonlinearity           3.11%      -0.53%    31.6%      47.6%      938          0      0
    +  Trading Activity            5.71%      +2.48%    10.4%      15.1%      938          0      0
    +  - the factor covariance's eigenvalues are scaled by the eigenfactor risk adjustment, a = 3, from 1000 simulated histories: each eigenfactor's volatility multiplied by 0.950 to 1.212, the smallest eigenvalue's by 1.123
```

### i5_compare.txt -- the defaults after every change, and the correlations' own lags on the model

```text
claude/scratch/midoffice13/r2/R2 against claude/scratch/midoffice13/i5/F1: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
claude/scratch/midoffice13/r2/R2h21 against claude/scratch/midoffice13/i5/C20: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
claude/scratch/midoffice13/r2/R2h21 against claude/scratch/midoffice13/i5/V21N: 21 fields compared; equal: 17; differing: covariance, observed_variance, noise_share, weighted_noise_share
  covariance: largest difference 1.027e-02
  observed_variance: largest difference 2.082e-17
  noise_share: largest difference 2.220e-16
  weighted_noise_share: largest difference 2.220e-16
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 8 differing line(s)
    -  the horizon: 21 trading days, a reporting view, not the model's horizon of one day: broad portfolios' risk over a holding period of 21 days. Its gain over the one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025 shock in it, and is re-tested as data accumulates -- the factor covariance Newey-West at 20 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 21-day returns; the specific risk one day's variance, with no lags, at every horizon
    -  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 20 lag(s), split horizon, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    +  the horizon: 21 trading day(s), a reporting view, not the model's horizon of one day: broad portfolios' risk over a holding period of 21 days. Its gain over the one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025 shock in it, and is re-tested as data accumulates -- the factor volatilities Newey-West at 20 lag(s) with Bartlett weights, each the per-day variance of 21-day returns; the factor correlations one-day, with no lags: the lags on the volatilities alone; the specific risk one day's variance, with no lags, at every horizon
    +  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 20 lag(s), split horizon, the correlations at 0 lag(s), half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    -  smallest 85, equally weighted                -3.137             -1.808   12.08%
    -  largest 50, equally weighted                 +0.167             +0.052    0.63%
    +  smallest 85, equally weighted                -3.137             -1.808   11.70%
    +  largest 50, equally weighted                 +0.167             +0.052    0.62%
claude/scratch/midoffice13/r2/R2h21 against claude/scratch/midoffice13/i5/V21R: 21 fields compared; equal: 16; differing: covariance, observed_variance, weighted_sampling_variance, noise_share, weighted_noise_share
  covariance: largest difference 1.035e-02
  observed_variance: largest difference 1.035e-02
  weighted_sampling_variance: largest difference nan
  noise_share: largest difference 1.573e-01
  weighted_noise_share: largest difference nan
  qualifications: 6 and 6; differing: 1
  report: 162 and 162 lines; 60 differing line(s)
    -  the horizon: 21 trading days, a reporting view, not the model's horizon of one day: broad portfolios' risk over a holding period of 21 days. Its gain over the one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025 shock in it, and is re-tested as data accumulates -- the factor covariance Newey-West at 20 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 21-day returns; the specific risk one day's variance, with no lags, at every horizon
    -  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 20 lag(s), split horizon, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    +  the horizon: 21 trading day(s), a reporting view, not the model's horizon of one day: broad portfolios' risk over a holding period of 21 days. Its gain over the one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025 shock in it, and is re-tested as data accumulates -- each factor volatility its one-day volatility on a half-life of 90 times the square root of its variance ratio at 20 lags, the ratio estimated on a half-life of 504; the factor correlations one-day, with no lags: the lags on the volatilities alone; the specific risk one day's variance, with no lags, at every horizon
    +  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 20 lag(s), split horizon, the correlations at 0 lag(s), each volatility one period's times its variance ratio on 504, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    -  smallest 85, equally weighted                -3.137             -1.808   12.08%
    -  largest 50, equally weighted                 +0.167             +0.052    0.63%
    +  smallest 85, equally weighted                -3.137             -1.808   11.28%
    +  largest 50, equally weighted                 +0.167             +0.052    0.59%
    -  Construction               18.77%     +24.76%    40.8%      50.0%      938          0      0
    -  Finance & Real Estate      14.65%     +18.11%    34.0%      49.0%      938          0      0
    -  Manufacturing               9.37%     +20.88%    27.5%      38.1%      938          0      0
    -  Mining                     18.99%     +14.94%    17.8%      25.1%      938          0      0
    -  Retail                     12.16%     +14.63%    12.4%      18.0%      938          0      0
    -  SIC20                      11.92%     +11.03%    33.8%      53.7%      938          0      0
    -  SIC28                      11.99%     +16.64%    12.3%      18.1%      938          0      0
    -  SIC35                      20.55%     +29.69%     4.4%       5.6%      938          0      0
    -  SIC36                      21.00%     +32.55%     4.6%       5.2%      938          0      0
    -  SIC37                      13.07%     +15.42%    22.6%      31.0%      938          0      0
    -  SIC38                      12.30%     +15.82%    15.2%      21.8%      938          0      0
    -  SIC49                      14.56%     +24.01%    13.3%      19.1%      938          0      0
    -  SIC60                      22.28%     +21.97%     7.1%       9.7%      938          0      0
    -  SIC62                      19.40%     +18.62%    10.7%      13.4%      938          0      0
    -  SIC63                      14.60%     +19.17%    14.5%      21.7%      938          0      0
    -  SIC67                      15.62%     +18.00%    15.5%      22.5%      938          0      0
    -  SIC73                      16.88%     +15.21%     3.4%       5.0%      938          0      0
    -  Services                   11.13%     +16.11%    36.6%      53.4%      938          0      0
    -  Transport & Utilities      12.30%     +15.46%    19.7%      28.8%      938          0      0
    -  Wholesale                  12.01%     +20.69%    62.5%      89.5%      938          0      0
    -  Volatility                  8.35%      +3.09%     4.4%       6.8%      938          0      0
    -  Momentum                    7.24%      +4.23%     3.6%       4.9%      938          0      0
    -  Size                        3.88%      +3.13%     7.1%       9.5%      938          0      0
    -  Size Nonlinearity           2.32%      -0.53%    46.4%      64.5%      938          0      0
    -  Trading Activity            6.65%      +2.48%     6.6%       8.9%      938          0      0
    +  Construction               20.81%     +24.76%    33.2%       nan%      938          0      0
    +  Finance & Real Estate      15.40%     +18.11%    30.8%       nan%      938          0      0
    +  Manufacturing               9.86%     +20.88%    24.8%       nan%      938          0      0
    +  Mining                     17.64%     +14.94%    20.6%       nan%      938          0      0
    +  Retail                     13.11%     +14.63%    10.7%       nan%      938          0      0
    +  SIC20                      11.21%     +11.03%    38.3%       nan%      938          0      0
    +  SIC28                      13.25%     +16.64%    10.0%       nan%      938          0      0
    +  SIC35                      17.86%     +29.69%     5.8%       nan%      938          0      0
    +  SIC36                      20.89%     +32.55%     4.6%       nan%      938          0      0
    +  SIC37                      15.12%     +15.42%    16.9%       nan%      938          0      0
    +  SIC38                      13.74%     +15.82%    12.2%       nan%      938          0      0
    +  SIC49                      15.04%     +24.01%    12.5%       nan%      938          0      0
    +  SIC60                      23.28%     +21.97%     6.5%       nan%      938          0      0
    +  SIC62                      20.31%     +18.62%     9.8%       nan%      938          0      0
    +  SIC63                      12.90%     +19.17%    18.5%       nan%      938          0      0
    +  SIC67                      15.39%     +18.00%    15.9%       nan%      938          0      0
    +  SIC73                      16.90%     +15.21%     3.4%       nan%      938          0      0
    +  Services                   13.48%     +16.11%    24.9%       nan%      938          0      0
    +  Transport & Utilities      13.77%     +15.46%    15.7%       nan%      938          0      0
    +  Wholesale                  12.79%     +20.69%    55.2%       nan%      938          0      0
    +  Volatility                  8.47%      +3.09%     4.3%       nan%      938          0      0
    +  Momentum                    6.63%      +4.23%     4.3%       nan%      938          0      0
    +  Size                        3.69%      +3.13%     7.8%       nan%      938          0      0
    +  Size Nonlinearity           2.85%      -0.53%    30.7%       nan%      938          0      0
    +  Trading Activity            5.94%      +2.48%     8.3%       nan%      938          0      0
    -  - 89% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is biased upward and is not corrected here
    +  - 55% of the estimated variance of Wholesale is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
```

### equal_D.txt -- the new defaults' bias run against midoffice12's H1, bit for bit

```text
claude/scratch/midoffice13/bias/D.npz against claude/scratch/midoffice12/bias/H1.npz: 686 forecast dates in common (686 and 686), 2024-02-08 to 2026-11-02
  caps                                 equal to the bit  (686, 592)
  dates                                equal to the bit  (686,)
  eigen_forecast                       equal to the bit  (25, 686)
  eigen_realised                       equal to the bit  (25, 686)
  eigen_realised_h11                   equal to the bit  (25, 686)
  eigen_realised_h21                   equal to the bit  (25, 686)
  eigen_realised_h5                    equal to the bit  (25, 686)
  forecast                             equal to the bit  (226, 686)
  held_long_only                       equal to the bit  (686,)
  held_unconstrained                   equal to the bit  (686,)
  horizons                             equal to the bit  (3,)
  member_forecast                      equal to the bit  (686, 592)
  member_realised                      equal to the bit  (686, 592)
  member_realised_h11                  equal to the bit  (686, 592)
  member_realised_h21                  equal to the bit  (686, 592)
  member_realised_h5                   equal to the bit  (686, 592)
  model_horizon                        equal to the bit  ()
  names                                equal to the bit  (226,)
  realised                             equal to the bit  (226, 686)
  realised_h11                         equal to the bit  (226, 686)
  realised_h21                         equal to the bit  (226, 686)
  realised_h5                          equal to the bit  (226, 686)
  repaired                             equal to the bit  (686,)
  securities                           equal to the bit  (592,)
every common array equal to the bit
```

### equal_E.txt -- the eigenfactor run's model test against the defaults', bit for bit

```text
claude/scratch/midoffice13/bias/E.npz against claude/scratch/midoffice13/bias/D.npz: 686 forecast dates in common (686 and 686), 2024-02-08 to 2026-11-02
  caps                                 equal to the bit  (686, 592)
  dates                                equal to the bit  (686,)
  eigen_forecast                       equal to the bit  (25, 686)
  eigen_realised                       equal to the bit  (25, 686)
  eigen_realised_h11                   equal to the bit  (25, 686)
  eigen_realised_h21                   equal to the bit  (25, 686)
  eigen_realised_h5                    equal to the bit  (25, 686)
  forecast                             equal to the bit  (226, 686)
  held_long_only                       equal to the bit  (686,)
  held_unconstrained                   equal to the bit  (686,)
  horizons                             equal to the bit  (3,)
  member_forecast                      equal to the bit  (686, 592)
  member_realised                      equal to the bit  (686, 592)
  member_realised_h11                  equal to the bit  (686, 592)
  member_realised_h21                  equal to the bit  (686, 592)
  member_realised_h5                   equal to the bit  (686, 592)
  model_horizon                        equal to the bit  ()
  names                                equal to the bit  (226,)
  realised                             equal to the bit  (226, 686)
  realised_h11                         equal to the bit  (226, 686)
  realised_h21                         equal to the bit  (226, 686)
  realised_h5                          equal to the bit  (226, 686)
  repaired                             equal to the bit  (686,)
  securities                           equal to the bit  (592,)
  only in the first: eigen_bias, eigen_forecast_a1, eigen_forecast_a2, eigen_forecast_a3, eigen_forecast_a4, eigen_realised_a1, eigen_realised_a2, eigen_realised_a3, eigen_realised_a4, eigen_realised_h11_a1, eigen_realised_h11_a2, eigen_realised_h11_a3, eigen_realised_h11_a4, eigen_realised_h21_a1, eigen_realised_h21_a2, eigen_realised_h21_a3, eigen_realised_h21_a4, eigen_realised_h5_a1, eigen_realised_h5_a2, eigen_realised_h5_a3, eigen_realised_h5_a4, eigen_scales, eigen_seconds, eigen_standard_error, forecast_a1, forecast_a2, forecast_a3, forecast_a4, held_long_only_a1, held_long_only_a2, held_long_only_a3, held_long_only_a4, held_unconstrained_a1, held_unconstrained_a2, held_unconstrained_a3, held_unconstrained_a4, member_forecast_a1, member_forecast_a2, member_forecast_a3, member_forecast_a4, realised_a1, realised_a2, realised_a3, realised_a4, realised_h11_a1, realised_h11_a2, realised_h11_a3, realised_h11_a4, realised_h21_a1, realised_h21_a2, realised_h21_a3, realised_h21_a4, realised_h5_a1, realised_h5_a2, realised_h5_a3, realised_h5_a4, repaired_a1, repaired_a2, repaired_a3, repaired_a4
every common array equal to the bit
```

### equal_H21.txt -- the 21-day model rerun against midoffice12's H21, bit for bit

```text
claude/scratch/midoffice13/bias/H21.npz against claude/scratch/midoffice12/bias/H21.npz: 686 forecast dates in common (686 and 686), 2024-02-08 to 2026-11-02
  caps                                 equal to the bit  (686, 592)
  dates                                equal to the bit  (686,)
  eigen_forecast                       equal to the bit  (25, 686)
  eigen_realised                       equal to the bit  (25, 686)
  eigen_realised_h11                   equal to the bit  (25, 686)
  eigen_realised_h21                   equal to the bit  (25, 686)
  eigen_realised_h5                    equal to the bit  (25, 686)
  forecast                             equal to the bit  (226, 686)
  held_long_only                       equal to the bit  (686,)
  held_unconstrained                   equal to the bit  (686,)
  horizons                             equal to the bit  (3,)
  member_forecast                      equal to the bit  (686, 592)
  member_realised                      equal to the bit  (686, 592)
  member_realised_h11                  equal to the bit  (686, 592)
  member_realised_h21                  equal to the bit  (686, 592)
  member_realised_h5                   equal to the bit  (686, 592)
  model_horizon                        equal to the bit  ()
  names                                equal to the bit  (226,)
  realised                             equal to the bit  (226, 686)
  realised_h11                         equal to the bit  (226, 686)
  realised_h21                         equal to the bit  (226, 686)
  realised_h5                          equal to the bit  (226, 686)
  repaired                             equal to the bit  (686,)
  securities                           equal to the bit  (592,)
every common array equal to the bit
```

### bias/report_E.txt -- the eigenfactor run's own report of the adjustment

```text
  The eigenfactor risk adjustment (Menchero, Wang and Orr; USE4) at one day: each eigenvalue of the day's factor covariance scaled by (a (lambda - 1) + 1)^2, the eigenportfolios tested in the unadjusted covariance's eigenbasis; the simulation 0.49 s a refit (0.31 to 0.72)
  lambda by rank over the 686 dates, the mean (min to max): 1: 1.002 (1.000-1.005), 2: 0.986 (0.972-0.994), 3: 0.977 (0.939-0.996), 5: 1.029 (1.005-1.056), 10: 0.992 (0.975-1.012), 15: 1.031 (1.005-1.063), 20: 1.076 (1.027-1.173), 21: 1.062, 22: 1.060, 23: 1.075, 24: 1.061, 25: 1.065; its Monte Carlo standard error at most 0.0019, 0.0013 at the median date's worst rank

  the whole test (2024-02-08 to 2026-11-02, 686), by a (0 the model):
                                                 a = 0     a = 1     a = 2     a = 3     a = 4
  eigenportfolios 1-5, mean                      1.030     1.033     1.037     1.041     1.047
  eigenportfolios 6-10, mean                     1.082     1.085     1.089     1.094     1.099
  eigenportfolios 11-15, mean                    1.114     1.092     1.072     1.052     1.034
  eigenportfolios 16-20, mean                    1.216     1.144     1.081     1.026     0.976
  eigenportfolios 21-25, mean                    1.194     1.122     1.060     1.005     0.956
  eigenportfolios, mean abs(bias - 1)            0.129     0.096     0.068     0.050     0.054
  minimum variance, long-only                    1.167     1.158     1.152     1.147     1.144
  minimum variance, unconstrained                1.220     1.214     1.211     1.210     1.211
  minimum variance, mean abs(bias - 1)           0.194     0.186     0.181     0.179     0.178
  the criterion                                  0.161     0.141     0.124     0.114     0.116
  the index, cap-weighted                        1.015     1.013     1.011     1.008     1.006
  the index, equally weighted                    0.965     0.955     0.945     0.935     0.924
  smallest 85, equally weighted                  0.956     0.942     0.928     0.913     0.899
  largest 50, equally weighted                   0.996     0.993     0.991     0.988     0.985
  members, median                                1.003     0.998     0.990     0.984     0.977
  members, mean abs(bias - 1)                    0.090     0.090     0.091     0.092     0.094
  industries, mean abs(bias - 1)                 0.046     0.047     0.051     0.054     0.058
  random long-only, median                       0.971     0.960     0.951     0.943     0.933
  random long-short, median                      1.004     0.999     0.993     0.989     0.985

  the first half (2024-02-08 to 2025-06-23, 343), by a (0 the model):
                                                 a = 0     a = 1     a = 2     a = 3     a = 4
  eigenportfolios 1-5, mean                      1.065     1.068     1.073     1.078     1.086
  eigenportfolios 6-10, mean                     1.019     1.024     1.030     1.036     1.044
  eigenportfolios 11-15, mean                    1.082     1.052     1.024     0.998     0.973
  eigenportfolios 16-20, mean                    1.156     1.073     1.003     0.942     0.889
  eigenportfolios 21-25, mean                    1.199     1.109     1.032     0.966     0.909
  eigenportfolios, mean abs(bias - 1)            0.110     0.070     0.046     0.052     0.079
  minimum variance, long-only                    1.246     1.233     1.224     1.218     1.215
  minimum variance, unconstrained                1.297     1.286     1.279     1.275     1.274
  minimum variance, mean abs(bias - 1)           0.272     0.260     0.252     0.247     0.244
  the criterion                                  0.191     0.165     0.149     0.149     0.161
  the index, cap-weighted                        1.224     1.222     1.220     1.217     1.215
  the index, equally weighted                    1.142     1.130     1.117     1.104     1.090
  smallest 85, equally weighted                  1.064     1.047     1.029     1.012     0.994
  largest 50, equally weighted                   1.175     1.172     1.169     1.165     1.162
  members, median                                1.037     1.028     1.022     1.013     1.005
  members, mean abs(bias - 1)                    0.105     0.104     0.103     0.103     0.103
  industries, mean abs(bias - 1)                 0.094     0.087     0.082     0.079     0.076
  random long-only, median                       1.124     1.112     1.100     1.088     1.076
  random long-short, median                      1.000     0.995     0.989     0.983     0.973

  the second half (2025-06-24 to 2026-11-02, 343), by a (0 the model):
                                                 a = 0     a = 1     a = 2     a = 3     a = 4
  eigenportfolios 1-5, mean                      0.986     0.988     0.992     0.995     1.000
  eigenportfolios 6-10, mean                     1.142     1.143     1.146     1.148     1.152
  eigenportfolios 11-15, mean                    1.145     1.130     1.117     1.104     1.091
  eigenportfolios 16-20, mean                    1.269     1.207     1.151     1.100     1.054
  eigenportfolios 21-25, mean                    1.189     1.136     1.087     1.042     1.002
  eigenportfolios, mean abs(bias - 1)            0.164     0.139     0.116     0.097     0.085
  minimum variance, long-only                    1.084     1.079     1.076     1.073     1.071
  minimum variance, unconstrained                1.139     1.139     1.140     1.142     1.146
  minimum variance, mean abs(bias - 1)           0.111     0.109     0.108     0.108     0.109
  the criterion                                  0.138     0.124     0.112     0.102     0.097
  the index, cap-weighted                        0.751     0.749     0.747     0.745     0.743
  the index, equally weighted                    0.749     0.742     0.736     0.729     0.722
  smallest 85, equally weighted                  0.835     0.825     0.815     0.804     0.794
  largest 50, equally weighted                   0.778     0.776     0.774     0.772     0.770
  members, median                                0.967     0.961     0.956     0.950     0.944
  members, mean abs(bias - 1)                    0.103     0.105     0.107     0.109     0.111
  industries, mean abs(bias - 1)                 0.110     0.114     0.120     0.126     0.133
  random long-only, median                       0.798     0.792     0.786     0.779     0.772
  random long-short, median                      1.006     1.003     0.999     0.995     0.991

  a chosen on the first half by the criterion -- half the eigenportfolios' mean abs(bias - 1) over every rank, half the minimum-variance portfolios', long-only and unconstrained: a = 2 (0.149; the model's 0.191); on the second half it reads 0.112 against the model's 0.138, the best there a = 4
  every forecast and outcome written to C:\Data\Development\Python\McTheoryApps\praxis\claude\scratch\midoffice13\bias\E.tmp.npz
```

### compare_eigen.txt -- every run, each statistic recomputed from its stored forecasts and outcomes

```text
The eigenfactor risk adjustment: 686 forecast dates, 2024-02-08 to 2026-11-02; the model (a = 0) and a = 1, 2, 3, 4
  the simulation: 0.49 s a refit (min 0.31, max 0.72), 5.6 min over the test; the Monte Carlo standard error of lambda at most 0.0019 (rank 25), its median over dates of each date's worst 0.0013

lambda by rank over the dates (rank 1 the largest eigenvalue)
                    mean       min       max  1st half  2nd half    max se
  rank 1           1.002     1.000     1.005     1.002     1.002     0.002
  rank 2           0.986     0.972     0.994     0.984     0.987     0.001
  rank 3           0.977     0.939     0.996     0.969     0.985     0.001
  rank 4           0.995     0.975     1.015     0.998     0.991     0.001
  rank 5           1.029     1.005     1.056     1.034     1.024     0.001
  rank 6           0.991     0.973     1.021     0.997     0.985     0.001
  rank 7           0.999     0.957     1.024     0.994     1.003     0.001
  rank 8           1.004     0.982     1.028     1.003     1.005     0.001
  rank 9           1.002     0.964     1.034     0.997     1.008     0.001
  rank 10          0.992     0.975     1.012     0.991     0.992     0.001
  rank 11          1.011     0.997     1.027     1.009     1.012     0.001
  rank 12          1.018     0.979     1.039     1.025     1.011     0.001
  rank 13          1.024     0.989     1.055     1.036     1.012     0.001
  rank 14          1.022     0.987     1.062     1.032     1.013     0.002
  rank 15          1.031     1.005     1.063     1.042     1.021     0.002
  rank 16          1.045     1.019     1.075     1.056     1.033     0.001
  rank 17          1.052     1.021     1.097     1.064     1.039     0.001
  rank 18          1.066     1.037     1.121     1.078     1.054     0.002
  rank 19          1.089     1.051     1.161     1.105     1.072     0.002
  rank 20          1.076     1.027     1.173     1.090     1.063     0.002
  rank 21          1.062     1.024     1.142     1.087     1.037     0.002
  rank 22          1.060     1.019     1.088     1.067     1.054     0.002
  rank 23          1.075     1.042     1.152     1.095     1.055     0.002
  rank 24          1.061     1.036     1.119     1.077     1.045     0.002
  rank 25          1.065     1.038     1.142     1.083     1.047     0.002

gamma = a (lambda - 1) + 1 over every rank and date
                     min    median       max     r 1-5   r 21-25     < 0.8
  a = 1            0.939     1.026     1.173     0.998     1.065     0.000
  a = 2            0.879     1.052     1.345     0.995     1.129     0.000
  a = 3            0.818     1.077     1.518     0.993     1.194     0.000
  a = 4            0.757     1.103     1.690     0.991     1.259     0.004

At one day, the whole test (2024-02-08 to 2026-11-02, 686), by a (0 the model)
                                                   a = 0     a = 1     a = 2     a = 3     a = 4
  eigenportfolios 1-5, mean                        1.030     1.033     1.037     1.041     1.047
  eigenportfolios 6-10, mean                       1.082     1.085     1.089     1.094     1.099
  eigenportfolios 11-15, mean                      1.114     1.092     1.072     1.052     1.034
  eigenportfolios 16-20, mean                      1.216     1.144     1.081     1.026     0.976
  eigenportfolios 21-25, mean                      1.194     1.122     1.060     1.005     0.956
  eigenportfolios, mean abs(bias - 1)              0.129     0.096     0.068     0.050     0.054
  minimum variance, long-only                      1.167     1.158     1.152     1.147     1.144
  minimum variance, unconstrained                  1.220     1.214     1.211     1.210     1.211
  minimum variance, mean abs(bias - 1)             0.194     0.186     0.181     0.179     0.178
  the criterion                                    0.161     0.141     0.124     0.114     0.116
  the index, cap-weighted                          1.015     1.013     1.011     1.008     1.006
  the index, equally weighted                      0.965     0.955     0.945     0.935     0.924
  smallest 85, equally weighted                    0.956     0.942     0.928     0.913     0.899
  largest 50, equally weighted                     0.996     0.993     0.991     0.988     0.985
  members, median                                  1.003     0.998     0.990     0.984     0.977
  members, mean abs(bias - 1)                      0.090     0.090     0.091     0.092     0.094
  industries, mean abs(bias - 1)                   0.046     0.047     0.051     0.054     0.058
  the thin three, mean abs(bias - 1)               0.020     0.029     0.039     0.048     0.058
  random long-only, median                         0.971     0.960     0.951     0.943     0.933
  random long-short, median                        1.004     0.999     0.993     0.989     0.985

At one day, the first half (2024-02-08 to 2025-06-23, 343), by a (0 the model)
                                                   a = 0     a = 1     a = 2     a = 3     a = 4
  eigenportfolios 1-5, mean                        1.065     1.068     1.073     1.078     1.086
  eigenportfolios 6-10, mean                       1.019     1.024     1.030     1.036     1.044
  eigenportfolios 11-15, mean                      1.082     1.052     1.024     0.998     0.973
  eigenportfolios 16-20, mean                      1.156     1.073     1.003     0.942     0.889
  eigenportfolios 21-25, mean                      1.199     1.109     1.032     0.966     0.909
  eigenportfolios, mean abs(bias - 1)              0.110     0.070     0.046     0.052     0.079
  minimum variance, long-only                      1.246     1.233     1.224     1.218     1.215
  minimum variance, unconstrained                  1.297     1.286     1.279     1.275     1.274
  minimum variance, mean abs(bias - 1)             0.272     0.260     0.252     0.247     0.244
  the criterion                                    0.191     0.165     0.149     0.149     0.161
  the index, cap-weighted                          1.224     1.222     1.220     1.217     1.215
  the index, equally weighted                      1.142     1.130     1.117     1.104     1.090
  smallest 85, equally weighted                    1.064     1.047     1.029     1.012     0.994
  largest 50, equally weighted                     1.175     1.172     1.169     1.165     1.162
  members, median                                  1.037     1.028     1.022     1.013     1.005
  members, mean abs(bias - 1)                      0.105     0.104     0.103     0.103     0.103
  industries, mean abs(bias - 1)                   0.094     0.087     0.082     0.079     0.076
  the thin three, mean abs(bias - 1)               0.035     0.038     0.040     0.043     0.046
  random long-only, median                         1.124     1.112     1.100     1.088     1.076
  random long-short, median                        1.000     0.995     0.989     0.983     0.973

At one day, the second half (2025-06-24 to 2026-11-02, 343), by a (0 the model)
                                                   a = 0     a = 1     a = 2     a = 3     a = 4
  eigenportfolios 1-5, mean                        0.986     0.988     0.992     0.995     1.000
  eigenportfolios 6-10, mean                       1.142     1.143     1.146     1.148     1.152
  eigenportfolios 11-15, mean                      1.145     1.130     1.117     1.104     1.091
  eigenportfolios 16-20, mean                      1.269     1.207     1.151     1.100     1.054
  eigenportfolios 21-25, mean                      1.189     1.136     1.087     1.042     1.002
  eigenportfolios, mean abs(bias - 1)              0.164     0.139     0.116     0.097     0.085
  minimum variance, long-only                      1.084     1.079     1.076     1.073     1.071
  minimum variance, unconstrained                  1.139     1.139     1.140     1.142     1.146
  minimum variance, mean abs(bias - 1)             0.111     0.109     0.108     0.108     0.109
  the criterion                                    0.138     0.124     0.112     0.102     0.097
  the index, cap-weighted                          0.751     0.749     0.747     0.745     0.743
  the index, equally weighted                      0.749     0.742     0.736     0.729     0.722
  smallest 85, equally weighted                    0.835     0.825     0.815     0.804     0.794
  largest 50, equally weighted                     0.778     0.776     0.774     0.772     0.770
  members, median                                  0.967     0.961     0.956     0.950     0.944
  members, mean abs(bias - 1)                      0.103     0.105     0.107     0.109     0.111
  industries, mean abs(bias - 1)                   0.110     0.114     0.120     0.126     0.133
  the thin three, mean abs(bias - 1)               0.060     0.065     0.071     0.077     0.083
  random long-only, median                         0.798     0.792     0.786     0.779     0.772
  random long-short, median                        1.006     1.003     0.999     0.995     0.991

At one day, the whole test without the forecast of 2026-09-30 (685), by a -- CTVA's unadjusted spin-off left out
                                                   a = 0     a = 1     a = 2     a = 3     a = 4
  eigenportfolios 1-5, mean                        1.030     1.033     1.037     1.042     1.048
  eigenportfolios 6-10, mean                       1.081     1.084     1.088     1.093     1.098
  eigenportfolios 11-15, mean                      1.114     1.092     1.071     1.052     1.034
  eigenportfolios 16-20, mean                      1.216     1.144     1.081     1.026     0.976
  eigenportfolios 21-25, mean                      1.193     1.121     1.059     1.004     0.955
  eigenportfolios, mean abs(bias - 1)              0.129     0.096     0.067     0.050     0.054
  minimum variance, long-only                      1.055     1.044     1.035     1.029     1.024
  minimum variance, unconstrained                  1.097     1.087     1.081     1.077     1.075
  minimum variance, mean abs(bias - 1)             0.076     0.066     0.058     0.053     0.050
  the criterion                                    0.103     0.081     0.063     0.051     0.052
  the index, cap-weighted                          1.015     1.013     1.011     1.009     1.007
  the index, equally weighted                      0.966     0.956     0.946     0.935     0.925
  smallest 85, equally weighted                    0.957     0.943     0.928     0.914     0.900
  largest 50, equally weighted                     0.997     0.994     0.991     0.989     0.986
  members, median                                  1.003     0.998     0.990     0.984     0.977
  members, mean abs(bias - 1)                      0.088     0.089     0.089     0.091     0.093
  industries, mean abs(bias - 1)                   0.046     0.047     0.051     0.055     0.059
  the thin three, mean abs(bias - 1)               0.021     0.030     0.040     0.049     0.059
  random long-only, median                         0.971     0.961     0.952     0.943     0.933
  random long-short, median                        1.002     0.998     0.993     0.990     0.986

At one day, the second half without it (342), by a -- CTVA's unadjusted spin-off left out
                                                   a = 0     a = 1     a = 2     a = 3     a = 4
  eigenportfolios 1-5, mean                        0.987     0.989     0.993     0.997     1.001
  eigenportfolios 6-10, mean                       1.140     1.142     1.144     1.147     1.150
  eigenportfolios 11-15, mean                      1.144     1.130     1.117     1.104     1.091
  eigenportfolios 16-20, mean                      1.270     1.207     1.151     1.100     1.054
  eigenportfolios 21-25, mean                      1.188     1.134     1.086     1.041     1.000
  eigenportfolios, mean abs(bias - 1)              0.164     0.138     0.116     0.096     0.085
  minimum variance, long-only                      0.822     0.812     0.804     0.796     0.790
  minimum variance, unconstrained                  0.853     0.844     0.838     0.834     0.832
  minimum variance, mean abs(bias - 1)             0.162     0.172     0.179     0.185     0.189
  the criterion                                    0.163     0.155     0.147     0.140     0.137
  the index, cap-weighted                          0.752     0.750     0.748     0.746     0.744
  the index, equally weighted                      0.749     0.743     0.736     0.730     0.723
  smallest 85, equally weighted                    0.836     0.826     0.816     0.805     0.795
  largest 50, equally weighted                     0.779     0.777     0.775     0.773     0.771
  members, median                                  0.966     0.961     0.956     0.950     0.945
  members, mean abs(bias - 1)                      0.100     0.102     0.104     0.106     0.108
  industries, mean abs(bias - 1)                   0.110     0.114     0.120     0.126     0.132
  the thin three, mean abs(bias - 1)               0.062     0.067     0.073     0.079     0.085
  random long-only, median                         0.798     0.792     0.787     0.780     0.773
  random long-short, median                        1.003     1.002     0.997     0.993     0.989

  a chosen on the first half by the criterion -- half the eigenportfolios' mean abs(bias - 1) over every rank, half the minimum-variance portfolios': a = 2, 0.149 against the model's 0.191; on the second half 0.112 against 0.138, the best there a = 4

Each eigenportfolio at one day, the whole test, by a
                   a = 0     a = 1     a = 2     a = 3     a = 4
  rank 1           1.027     1.025     1.023     1.021     1.019
  rank 2           0.975     0.991     1.007     1.024     1.041
  rank 3           1.045     1.070     1.096     1.124     1.155
  rank 4           1.048     1.053     1.058     1.064     1.070
  rank 5           1.055     1.026     0.999     0.974     0.950
  rank 6           1.036     1.047     1.059     1.072     1.085
  rank 7           1.090     1.092     1.095     1.100     1.105
  rank 8           1.079     1.075     1.071     1.067     1.063
  rank 9           1.131     1.129     1.127     1.126     1.127
  rank 10          1.075     1.084     1.094     1.104     1.115
  rank 11          1.104     1.094     1.085     1.076     1.067
  rank 12          1.098     1.079     1.061     1.044     1.028
  rank 13          1.110     1.084     1.061     1.039     1.019
  rank 14          1.103     1.081     1.061     1.041     1.023
  rank 15          1.156     1.122     1.091     1.061     1.034
  rank 16          1.168     1.118     1.073     1.031     0.994
  rank 17          1.160     1.105     1.056     1.012     0.971
  rank 18          1.247     1.175     1.112     1.056     1.005
  rank 19          1.285     1.183     1.098     1.026     0.963
  rank 20          1.219     1.137     1.066     1.004     0.949
  rank 21          1.194     1.124     1.065     1.013     0.968
  rank 22          1.193     1.126     1.067     1.014     0.967
  rank 23          1.226     1.142     1.069     1.006     0.951
  rank 24          1.233     1.161     1.098     1.043     0.993
  rank 25          1.125     1.058     1.000     0.948     0.903

Each eigenportfolio at one day, the first half (2024-02-08 to 2025-06-23, 343), by a
                   a = 0     a = 1     a = 2     a = 3     a = 4
  rank 1           1.226     1.224     1.221     1.219     1.217
  rank 2           0.960     0.978     0.997     1.017     1.037
  rank 3           1.017     1.048     1.082     1.119     1.160
  rank 4           1.077     1.078     1.080     1.082     1.085
  rank 5           1.046     1.014     0.984     0.956     0.929
  rank 6           0.964     0.968     0.973     0.979     0.985
  rank 7           1.035     1.043     1.051     1.062     1.074
  rank 8           1.072     1.069     1.067     1.064     1.062
  rank 9           1.009     1.016     1.025     1.034     1.045
  rank 10          1.016     1.025     1.034     1.043     1.053
  rank 11          1.024     1.016     1.009     1.002     0.995
  rank 12          1.070     1.045     1.021     0.999     0.977
  rank 13          1.143     1.103     1.066     1.032     1.000
  rank 14          1.051     1.017     0.986     0.958     0.931
  rank 15          1.123     1.078     1.037     0.998     0.963
  rank 16          1.177     1.113     1.057     1.006     0.959
  rank 17          1.111     1.045     0.986     0.934     0.887
  rank 18          1.133     1.056     0.989     0.931     0.880
  rank 19          1.190     1.075     0.982     0.906     0.841
  rank 20          1.168     1.077     1.000     0.933     0.876
  rank 21          1.226     1.128     1.046     0.977     0.918
  rank 22          1.220     1.145     1.079     1.021     0.968
  rank 23          1.231     1.124     1.036     0.961     0.897
  rank 24          1.232     1.141     1.064     0.997     0.939
  rank 25          1.089     1.005     0.935     0.874     0.821

Each eigenportfolio at one day, the second half (2025-06-24 to 2026-11-02, 343), by a
                   a = 0     a = 1     a = 2     a = 3     a = 4
  rank 1           0.780     0.778     0.777     0.775     0.773
  rank 2           0.992     1.005     1.019     1.033     1.047
  rank 3           1.074     1.092     1.112     1.132     1.152
  rank 4           1.018     1.027     1.036     1.046     1.056
  rank 5           1.064     1.039     1.015     0.992     0.971
  rank 6           1.104     1.122     1.140     1.158     1.178
  rank 7           1.143     1.141     1.139     1.138     1.137
  rank 8           1.088     1.082     1.076     1.071     1.066
  rank 9           1.243     1.232     1.222     1.213     1.205
  rank 10          1.131     1.141     1.151     1.162     1.173
  rank 11          1.178     1.167     1.155     1.144     1.133
  rank 12          1.127     1.113     1.101     1.089     1.078
  rank 13          1.076     1.066     1.056     1.047     1.038
  rank 14          1.154     1.143     1.131     1.120     1.110
  rank 15          1.187     1.163     1.141     1.119     1.099
  rank 16          1.161     1.125     1.090     1.058     1.028
  rank 17          1.208     1.163     1.122     1.084     1.048
  rank 18          1.349     1.281     1.221     1.165     1.115
  rank 19          1.372     1.282     1.203     1.133     1.071
  rank 20          1.257     1.183     1.118     1.060     1.008
  rank 21          1.163     1.122     1.084     1.049     1.016
  rank 22          1.167     1.108     1.056     1.009     0.966
  rank 23          1.223     1.159     1.102     1.050     1.003
  rank 24          1.233     1.180     1.131     1.086     1.045
  rank 25          1.161     1.109     1.062     1.019     0.979

The minimum-variance portfolios: forecast volatility at the median and realised, % a year, and names held at the median
                                forecast  realised      held
  a = 0, long-only                  9.42     10.93     88.00
  a = 0, unconstrained              9.13     11.07    501.50
  a = 1, long-only                  9.52     11.00     87.00
  a = 1, unconstrained              9.24     11.19    501.50
  a = 2, long-only                  9.63     11.09     87.00
  a = 2, unconstrained              9.35     11.32    501.50
  a = 3, long-only                  9.73     11.18     87.00
  a = 3, unconstrained              9.44     11.44    501.50
  a = 4, long-only                  9.82     11.29     86.00
  a = 4, unconstrained              9.52     11.57    501.50

At 21 days, the windows overlapping, by a
                                                   a = 0     a = 1     a = 2     a = 3     a = 4
  the index, cap-weighted                          0.860     0.858     0.856     0.855     0.853
  the index, equally weighted                      0.805     0.797     0.788     0.779     0.770
  smallest 85, equally weighted                    0.775     0.764     0.752     0.740     0.728
  largest 50, equally weighted                     0.873     0.871     0.868     0.866     0.864
  minimum variance, long-only                      1.005     0.995     0.987     0.982     0.979
  minimum variance, unconstrained                  1.045     1.041     1.039     1.039     1.041
  eigenportfolios 21-25, mean                      1.159     1.090     1.030     0.977     0.930


The reporting view's candidates at 21 days: D = the one-day model; H21 = the 21-day model; V21N = volatilities NW 20 lags, correlations one-day; V21R = variance-ratio volatilities, correlations one-day
  the band at 21 days, overlapping: 0.790-1.189; at one day 0.946-1.054

Read at 21 day(s), the windows overlapping
                                                       D       H21      V21N      V21R
  the index, cap-weighted                          0.860     1.006     0.975     0.942
  the index, equally weighted                      0.805     0.892     0.905     0.856
  smallest 85, equally weighted                    0.775     0.866     0.859     0.817
  largest 50, equally weighted                     0.873     1.019     0.974     0.943
  minimum variance, long-only                      1.005     1.177     1.118     1.056
  minimum variance, unconstrained                  1.045     1.308     1.224     1.149
  eigenportfolios 21-25, mean                      1.159     2.686     1.476     1.291
  eigenportfolios 1-5, mean                        0.913     0.930     1.043     0.976
  random long-only, median                         0.832     0.920     0.936     0.895
  random long-short, median                        0.975     0.997     1.009     1.000
  industries, mean abs(bias - 1)                   0.079     0.054     0.061     0.062
  members, median                                  0.938     0.981     0.990     0.975

Read at 1 day(s), the windows overlapping
                                                       D       H21      V21N      V21R
  the index, cap-weighted                          1.015     1.148     1.123     1.095
  the index, equally weighted                      0.965     1.042     1.063     1.016
  smallest 85, equally weighted                    0.956     1.041     1.043     0.999
  largest 50, equally weighted                     0.996     1.124     1.086     1.061
  minimum variance, long-only                      1.167     1.360     1.347     1.253
  minimum variance, unconstrained                  1.220     1.629     1.503     1.375
  eigenportfolios 21-25, mean                      1.194     2.670     1.454     1.318
  eigenportfolios 1-5, mean                        1.030     1.033     1.179     1.130
  random long-only, median                         0.971     1.050     1.067     1.029
  random long-short, median                        1.004     1.029     1.039     1.032
  industries, mean abs(bias - 1)                   0.046     0.097     0.119     0.085
  members, median                                  1.003     1.044     1.057     1.041

Their rolling windows at 21 days: MRAD
                                                       D       H21      V21N      V21R
  the index, cap-weighted, MRAD at 21              0.140     0.160     0.162     0.127
  the index, equally weighted, MRAD at 21          0.178     0.116     0.135     0.134
  smallest 85, equally weighted, MRAD at 21        0.220     0.121     0.127     0.173
  largest 50, equally weighted, MRAD at 21         0.142     0.187     0.173     0.140
  minimum variance, long-only, MRAD at 21          0.111     0.149     0.090     0.069
  minimum variance, unconstrained, MRAD at 21      0.099     0.188     0.199     0.134

The forecast levels: the median per-day variance over the one-day model's
                                                       D       H21      V21N      V21R
  the index, cap-weighted                          1.000     0.682     0.744     0.809
  the index, equally weighted                      1.000     0.755     0.776     0.842
  smallest 85, equally weighted                    1.000     0.732     0.809     0.872
  largest 50, equally weighted                     1.000     0.687     0.759     0.826
  minimum variance, long-only                      1.000     0.699     0.746     0.790
  minimum variance, unconstrained                  1.000     0.561     0.680     0.766
  eigenportfolio 1                                 1.000     0.646     0.723     0.813
  eigenportfolio 5                                 1.000     0.716     0.649     0.746
  eigenportfolio 15                                1.000     0.533     0.733     0.801
  eigenportfolio 20                                1.000     0.322     0.668     0.732
  eigenportfolio 25                                1.000     0.369     0.879     1.076

The realised variance ratio at 21 days over the test
                                                    21 d
  the index, cap-weighted                          0.702
  the index, equally weighted                      0.670
  smallest 85, equally weighted                    0.643
  largest 50, equally weighted                     0.749
  minimum variance, long-only                      0.700
  minimum variance, unconstrained                  0.696

Day to day: the sd of the daily change in the log forecast volatility
                                                       D       H21      V21N      V21R
  the index, cap-weighted                         0.0109    0.0248    0.0213    0.0114
  the index, equally weighted                     0.0177    0.0250    0.0237    0.0162
  smallest 85, equally weighted                   0.0196    0.0240    0.0221    0.0170
  largest 50, equally weighted                    0.0202    0.0295    0.0263    0.0194
  minimum variance, long-only                     0.0103    0.0163    0.0173    0.0109
  minimum variance, unconstrained                 0.0092    0.0146    0.0172    0.0105
```

### minvar_realised.txt -- the minimum-variance portfolios' realised risk by a

```text
The minimum-variance portfolios' realised volatility, % a year, by a (each day's portfolio held one day)
                                      a = 0    a = 1    a = 2    a = 3    a = 4
  long-only, whole test               10.93    11.00    11.09    11.18    11.29
  long-only, without 2026-09-30        9.89     9.93     9.99    10.05    10.13
  long-only, first half               11.47    11.54    11.62    11.73    11.85
  long-only, second half w/o it        8.02     8.02     8.03     8.05     8.06
  unconstrained, whole test           11.07    11.19    11.32    11.44    11.57
  unconstrained, without 2026-09-30     9.97    10.05    10.13    10.22    10.32
  unconstrained, first half           11.56    11.68    11.80    11.92    12.04
  unconstrained, second half w/o it     8.09     8.11     8.15     8.19     8.25
```

### diagnose_day.txt -- the -13 sigma day: the model as of 2026-09-30 and its minimum-variance portfolios' holdings

```text
as of 2026-09-30, realised on 2026-10-01: 501 members eligible
  the eligible members' worst next-day returns: CTVA -83.8% (-41.9 sd), SKYD -9.6% (-3.0 sd), MKC -4.9% (-2.5 sd), DHR -4.4% (-2.3 sd), INCY -3.5% (-1.7 sd), DIS -3.4% (-2.1 sd), AMGN -3.4% (-2.0 sd), TMO -3.3% (-1.8 sd)

  minimum variance, long-only: 98 names, forecast 0.588% a day, realised -7.646% (-13.0 sd); the largest contributions:
    CTVA   weight +0.0944  return -83.81%  contribution -7.9146%  its forecast 2.00% a day
    DIS    weight +0.0132  return -3.40%  contribution -0.0449%  its forecast 1.62% a day
    TMO    weight +0.0133  return -3.34%  contribution -0.0445%  its forecast 1.84% a day
    PEP    weight +0.0387  return -0.88%  contribution -0.0342%  its forecast 1.29% a day
    GOOGL  weight +0.0194  return -1.70%  contribution -0.0330%  its forecast 2.00% a day
    GOOG   weight +0.0171  return -1.71%  contribution -0.0291%  its forecast 1.94% a day
    JNJ    weight +0.0117  return -2.30%  contribution -0.0268%  its forecast 1.41% a day
    PM     weight +0.0187  return -1.38%  contribution -0.0258%  its forecast 1.68% a day
    the rest: +0.5071%

  minimum variance, unconstrained: 501 names, forecast 0.568% a day, realised -7.909% (-13.9 sd); the largest contributions:
    CTVA   weight +0.0938  return -83.81%  contribution -7.8604%  its forecast 2.00% a day
    BAC    weight +0.0383  return -1.29%  contribution -0.0493%  its forecast 1.46% a day
    DIS    weight +0.0130  return -3.40%  contribution -0.0442%  its forecast 1.62% a day
    JNJ    weight +0.0152  return -2.30%  contribution -0.0349%  its forecast 1.41% a day
    KMB    weight +0.0106  return -3.17%  contribution -0.0336%  its forecast 1.86% a day
    TMO    weight +0.0098  return -3.34%  contribution -0.0329%  its forecast 1.84% a day
    PEP    weight +0.0344  return -0.88%  contribution -0.0304%  its forecast 1.29% a day
    GOOGL  weight +0.0175  return -1.70%  contribution -0.0298%  its forecast 2.00% a day
    the rest: +0.2061%
```

### scan_extreme.txt -- the test window's extreme member-days, CTVA's and MRNA's prices and records, the factor returns of 2026-10-01

```text
member-days in the test window, 2024-02-09 to 2026-11-03, with a one-day total return beyond 35%: 12
  2024-04-11 GL     -53.1%
  2024-07-26 DXCM   -40.7%
  2025-02-13 WST    -38.2%
  2025-07-02 CNC    -40.4%
  2025-07-31 ALGN   -36.6%
  2025-08-08 TTD    -38.6%
  2025-08-13 SKYD   +36.7%
  2025-09-10 ORCL   +35.9%
  2025-09-10 SNPS   -35.8%
  2025-10-29 FISV   -44.0%
  2026-08-19 MRNA   +177.0%
  2026-10-01 CTVA   -83.8%

CTVA around 2026-10-01: raw close, share count, index member
  2026-09-29      77.87     668,820,000  True
  2026-09-30      77.65     668,820,000  True
  2026-10-01      12.57     668,820,000  True
  2026-10-02      11.92     668,820,000  True
  2026-10-05      12.39     668,820,000  True
  2026-10-06      13.91     668,820,000  False
  the cache's splits: []
  the cache's events: {"name": "Corteva, Inc. Common Stock", "composite_figi": "BBG00BN969C1", "cik": "0001755672", "events": [{"ticker_change": {"ticker": "CTVA"}, "type": "ticker_change", "date": "2019-05-24"}]}

MRNA around 2026-08-19: raw close, share count, index member
  2026-08-17      64.46     396,786,259  True
  2026-08-18      62.96     396,786,259  True
  2026-08-19     174.38     396,786,259  True
  2026-08-20     133.32     396,786,259  True
  2026-08-21     145.13     396,786,259  True
  2026-08-24     138.89     396,786,259  True
  the cache's splits: [{"execution_date": "2017-08-03", "id": "E243d231980ee6e97355fc51eb61ccc7d22beeaa35ec0825da8f212566140f960", "split_from": 10, "split_to": 1, "ticker": "MRNA"}, {"execution_date": "2011-12-23", "id": "E15ca63b1a9dde1f145d895983156e7d16e2244
  the cache's events: {"name": "Moderna, Inc. Common Stock", "composite_figi": "BBG003PHHZT1", "cik": "0001682852", "events": [{"ticker_change": {"ticker": "MRNA"}, "type": "ticker_change", "date": "2018-12-07"}]}

the factor returns of 2026-10-01 in units of their prior year's sd, the largest five: Wholesale +2.5, Services -1.5, Manufacturing +1.5, Volatility +1.3, SIC28 -1.2
```

### bias/runner_log.txt -- the five runs, one at a time

```text
2026-11-12 12:02:47  runner started, pid 19020; 4.0 GB available
2026-11-12 12:02:47  D: 4.0 GB available
2026-11-12 12:02:47  D: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 5
2026-11-12 12:14:22  D: done in 11.6 min
2026-11-12 12:14:22  E: 6.6 GB available
2026-11-12 12:14:22  E: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 5 --bias-eigen-scales 1,2,3,4
2026-11-12 12:32:59  E: done in 18.6 min
2026-11-12 12:32:59  V21N: 5.1 GB available
2026-11-12 12:32:59  V21N: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 0 --horizon 21 --correlation-lags 0
2026-11-12 12:42:43  V21N: done in 9.7 min
2026-11-12 12:42:43  V21R: 4.9 GB available
2026-11-12 12:42:43  V21R: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 0 --horizon 21 --correlation-lags 0 --variance-ratio-half-life 504
2026-11-12 12:51:54  V21R: done in 9.2 min
2026-11-12 12:51:54  H21: 4.9 GB available
2026-11-12 12:51:54  H21: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 0 --horizon 21
2026-11-12 13:02:25  H21: done in 10.5 min
2026-11-12 13:02:25  runner finished; every run done
```

### final_checks.txt -- the final code: the defaults and --horizon 11 against their references; the suite, ruff, hygiene

```text
FD: status 0, mean R2 0.293203, 938 periods, 25 factors, 63s, the estimator 1.4s
claude/scratch/midoffice13/i5/F1 against claude/scratch/midoffice13/final/FD: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
FD11: status 0, mean R2 0.293203, 938 periods, 25 factors, 65s, the estimator 1.3s
claude/scratch/midoffice12/final/FD against claude/scratch/midoffice13/final/FD11: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 2 differing line(s)
    -  the horizon: 11 trading days -- the factor covariance Newey-West at 10 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 11-day returns; the specific risk one day's variance, with no lags, at every horizon
    +  the horizon: 11 trading days, a reporting view, not the model's horizon of one day: broad portfolios' risk over a holding period of 11 days. Its gain over the one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025 shock in it, and is re-tested as data accumulates -- the factor covariance Newey-West at 10 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 11-day returns; the specific risk one day's variance, with no lags, at every horizon
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========== 2356 passed, 1 skipped, 19 warnings in 101.39s (0:01:41) ===========
All checks passed!
ruff: clean over the changed Python files
scripts/run_sp500_model.py: CRLF (2223 lines), parses, ASCII
src/praxis/midoffice/model/bias.py: LF (666 lines), parses, ASCII
src/praxis/midoffice/model/covariance.py: LF (650 lines), parses, ASCII
src/praxis/midoffice/model/crosssectional.py: CRLF (1577 lines), parses, ASCII
src/praxis/midoffice/model/eigen.py: LF (136 lines), parses, ASCII
tests/midoffice/test_bias.py: LF (356 lines), parses, ASCII
tests/midoffice/test_covariance.py: LF (871 lines), parses, ASCII
tests/midoffice/test_crosssectional.py: CRLF (2130 lines), parses, ASCII
tests/midoffice/test_eigen.py: LF (117 lines), parses, ASCII
docs/FACTOR_MODEL.md: CRLF (1399 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (604 lines), -, ASCII
claude/handoffs/BRIEF_midoffice13_eigenvalue_adjustment.md: LF (68 lines), -, ASCII
all ok
exit 0
```

## git add (always last -- Rule 43)

This piece -- midoffice12's rulings, the eigenfactor risk adjustment, the bias run's scales, the correlations' own lags, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add scripts/run_sp500_model.py src/praxis/midoffice/model/bias.py src/praxis/midoffice/model/covariance.py src/praxis/midoffice/model/crosssectional.py src/praxis/midoffice/model/eigen.py tests/midoffice/test_bias.py tests/midoffice/test_covariance.py tests/midoffice/test_crosssectional.py tests/midoffice/test_eigen.py docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice13_eigenvalue_adjustment.md claude/retros/RETRO_midoffice13_eigenvalue_adjustment.md
```
