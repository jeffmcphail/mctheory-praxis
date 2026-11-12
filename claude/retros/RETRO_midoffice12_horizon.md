# Retro: Mid-office 12 -- midoffice11's rulings, then the horizon, measured

**Brief:** `claude/handoffs/BRIEF_midoffice12_horizon.md` -- written in Chat and pasted into the
Code session, transcribed verbatim
**Date:** 2026-11-12 (EST)
**Duration:** the brief arrived before 14:10Z (09:10 EST), the first clock reading after it; the
session's window was closed once between the paste and that reading, and the work resumed from
the state file and midoffice11's retro, already read. The rulings, the horizon, the statistic at
the horizon and the variant were built and measured by 09:44 EST; the runner ran the six bias
runs one at a time from 09:44; the comparison, the docs, the state file and this retro followed.
**Status:** COMPLETE.
- **Rulings 1 and 2:** built and measured -- the 63-day specific history and the two-half-life
  covariance (90 / 504) are the run's defaults, and the new defaults' bias run equals H63T504 to
  the bit.
- **Rulings 3 to 6:** the horizon is item 1; nothing else to build.
- **Items 1 to 5:** built and measured. No default changed beyond rulings 1 and 2.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **The new defaults** (rulings 1 and 2): `--specific-half-life` 63 and `--correlation-half-life`
  504 by default, `none` for the old forms. Their run equals midoffice11's code run with the two
  options in every field and report line, and **their bias run equals H63T504 to the bit** in
  every array, midoffice11's 21-day check included; the guard passed on the new code.
- **`--horizon h`** (item 1, default 11): Newey-West at h - 1 lags on the volatilities and the
  correlations alike, `--newey-west-lags` overriding; the report states the horizon. A test pins
  the identity exactly -- Bartlett weights at h - 1 lags are the per-day variance of h-day
  returns -- and each `--horizon h` equals `--newey-west-lags h-1` in every field.
- **The statistic at the model's horizon** (item 2): every day's forecast, at h times its daily
  variance, against the next h days' return, bought and held; the windows overlap, so the band
  is simulated (2.7 times the plain band's width at 11 days); every h-th apart beside it; at one
  day exactly midoffice11's daily statistic.
- **Each model is right for the broad portfolios at its own horizon** (item 3): the index 1.015,
  0.991, 0.986, 1.006 at 1, 5, 11, 21 days, every broad portfolio inside its band. **The lags'
  cost grows with the horizon:** the smallest eigenportfolios 1.19, 1.46, 1.92, 2.69; and at
  every horizon read, the one-day model forecasts the optimiser's portfolios and the small
  eigen-directions best (unconstrained minimum variance 1.045 at 21 days, against the 21-day
  model's 1.308).
- **The variance-ratio variant at 21** (item 4) quiets the volatilities (the index forecast's
  daily noise 0.025 to 0.014) and eases the optimiser's bias (1.31 to 1.27), but the
  correlations' lags keep the smallest eigenportfolios at 2.4, and its two-year ratio
  over-forecasts the broad portfolios at 21 days more than the 21-day model does.
- **The specific history at 42** (item 5) is not clearly better (members 1.034 to 1.031): 63
  stays.
- **What I would choose:** one day for daily risk, and for portfolio construction at any holding
  period; the holding period's horizon as a reporting view for the h-day risk of broad,
  long-only portfolios; so **one day as the model's default horizon**, the eigenvalue adjustment
  then on the one-day matrix. Not 5, not the variant as built; if a long-horizon view is wanted,
  lags on the volatilities alone is the candidate the results point to.
- **The runs:** six, one at a time through the runner, 09:44 to 10:39 EST; none failed.
- **Tests:** 2,343 passed, 1 skipped; 5 new.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice12_horizon.md`, verbatim. In outline:

- **"committed -- midoffice11."** Checked: 7c7350e in `git log`.
- **Rulings on midoffice11's decisions:**
  1. **The specific-risk history:** a 63-day half-life, the default.
  2. **The factor covariance:** the two-half-life estimator, volatilities 90 and correlations 504,
     both Newey-West, the default. With ruling 1 that is the run H63T504; the new defaults' bias
     run must equal it.
  3. **The horizon:** named, and a parameter. Newey-West with Bartlett weights at L lags is
     exactly the per-day variance of (L+1)-day returns, so 10 lags is an 11-day horizon. The
     default stays 11 until this piece's measurements are in.
  4. **The eigenvalue adjustment:** after the horizon is settled, not with it.
  5. **The correction and its specific side:** options, off.
  6. **No fold.** Construction's split and the mega caps' weight in their industries are a scheme
     question for later.
- **midoffice12, the horizon, measured,** the rulings first:
  1. **`--horizon h`** in trading days: Newey-West at h - 1 lags on the volatilities and the
     correlations alike, `--newey-west-lags` still overriding it; the report states the horizon.
  2. **The bias statistics at the model's horizon:** every day's forecast against the next h
     days' return, the windows overlapping, the band for overlapping windows found by
     simulation, the non-overlapping statistic beside it; at h = 1 today's daily statistic.
  3. **Candidates on the new defaults, each judged at its own horizon:** h = 1, 5 and 21 against
     11; for each, what its lags cost -- the smallest eigenportfolios and the minimum-variance
     portfolios, at its horizon and at one day.
  4. **One variant against the lags' noise, at h = 21:** each volatility its 90-day daily
     volatility times the square root of its variance ratio at h, the ratio estimated on 504
     days.
  5. **The specific history at a 42-day half-life;** 63 stays unless 42 is clearly better.
- Runs one at a time through the runner. **"Change no default beyond rulings 1 and 2. Report,
  and say which horizon you would choose for which use, and why."** Retro with its `git add`
  and the state file in the same commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `scripts/run_sp500_model.py` | rulings 1 and 2: `SPECIFIC_HALF_LIFE` 63 and `CORRELATION_HALF_LIFE` 504 as the defaults, `half_life_or_none`; item 1: `HORIZON`, `--horizon`, `--newey-west-lags` defaulting to none and overriding, the horizon line in the report, the estimator at the horizon's lags; item 4: `--variance-ratio-half-life` and its line; item 2: `--bias-horizons` (`horizon_list`), the bias section at one day and at the model's horizon (`bias_section(horizon)`, its opening, the portfolios, industries, random portfolios, members and eigenportfolios at both, `_report_horizons`, `ordinal`, `_member_columns`), the `.npz`'s keys; the imports | 40-45, 97-135, 783-788, 813-833, 906-1006, 1017-1038, 1045, 1062-1175, 1183, 1207-1214, 1279-1332, 1557-1592, 1976-1977 |
| `src/praxis/midoffice/model/bias.py` | the statistic at a horizon: `OVERLAP_DRAWS`, `OVERLAP_SEED`, `_simulated_bands`, `overlapping_band`; `BiasStatistics.overlap` and its kurtosis band; `bias_statistics(overlap=)`; `BiasRun`'s horizons -- the outcomes over each, `portfolio_outcomes`, `member_outcomes`, `eigen_outcomes`, `statistics(horizon=, overlapping=)`; `evaluate(horizons=)`, every date's h-day outcomes; the docstrings | 27-33, 44, 55-61, 80-121, 146-202, 359-415, 450-481, 505-507, 562-594 |
| `src/praxis/midoffice/model/covariance.py` | item 4: `HorizonCovariance.ratio_half_life`, its validation, `_ratio_variances`, `volatility_estimator`'s refusal, the method's label; the docstring | 243-256, 268-311, 345-358, 385-387 |
| `tests/midoffice/test_bias.py` | 2 new tests, 2 updated to the new interface; the import | 25, 74-102, 215-267 |
| `tests/midoffice/test_covariance.py` | 3 new tests | 407-463, 657-679 |
| `tests/midoffice/test_crosssectional.py` | 1 test extended: the variance-ratio estimator has no weighted share | 1678-1686 |
| `docs/FACTOR_MODEL.md` | the two-half-life default and the variance-ratio variant; the lags as a horizon; the history half-life's default; the bias statistic at the model's horizon, the guard on the new defaults, and the horizon measured | 257-270, 309-318, 766-776, 1170-1200, 1254-1290 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43), with the BLAS working note | throughout |
| `claude/handoffs/BRIEF_midoffice12_horizon.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice12_horizon.md` | new: this retro | new |

Written outside the tracked files: only `claude/scratch/midoffice12/` (gitignored). Nothing in
`data/` was written, overwritten or deleted; the cache gained nothing.

---

## Rulings 1 and 2: the new defaults

- **Built in the run, not the library.** `--specific-half-life` defaults to 63
  (`SPECIFIC_HALF_LIFE`) and `--correlation-half-life` to 504 (`CORRELATION_HALF_LIFE`); each
  takes `none` for its other form -- equal weights, one half-life -- through `half_life_or_none`.
  The library's defaults are unchanged (`specific_half_life=None`; the covariance estimator the
  caller gives), as midoffice09 did with `specific_history`: the run holds the S&P 500 model's
  defaults, and the library stays generic.
- **Measured** (`r1_compare.txt`): the new defaults (R1) against midoffice11's code run with
  `--specific-half-life 63 --correlation-half-life 504` (A0): equal in all 21 fields, all 6
  qualifications and all 161 report lines. `none none` (R1old) against midoffice11's final
  defaults (its B4): equal in all 21 fields, 7 qualifications and 162 lines.
- **What the defaults now say** (R1 against B4): specific risk p10 / median / p90 16.58 / 28.98 /
  48.98%, where equal weights gave 15.58 / 24.76 / 40.49% -- the 63-day history follows the rise
  in residual dispersion through 2025 and 2026; the size pair's risk 11.40% for the smallest 85
  and 0.60% for the largest 50 (11.87 and 0.62); the covariance's effective observations the
  correlation window's, 826.4 of 938; mean R-squared 0.293203 unchanged. One qualification goes:
  the single-half-life Newey-West estimator always says it is not guaranteed positive
  semi-definite under exponential weights, while the two-half-life estimator says so only when
  its matrix actually has a negative eigenvalue, and the defaults' has none (nor on any bias
  date: no repair bound).

## Rulings 3 to 6

- **The horizon** is item 1, below. **The eigenvalue adjustment** waits, as ruled; nothing is
  built toward it. **The correction and its specific side** stay options, off; **no fold**:
  nothing to build.

## Item 1: `--horizon h`

- **The lags are the horizon.** `--horizon H` (default `HORIZON`, 11) puts H - 1 Newey-West lags
  on the volatilities and the correlations alike; `--newey-west-lags L`, now defaulting to none,
  overrides it, and the horizon is then L + 1. The report's Model section states it, e.g. "the
  horizon: 11 trading days -- the factor covariance Newey-West at 10 lag(s) with Bartlett
  weights, on the volatilities and the correlations alike, which is exactly the per-day variance
  of 11-day returns; the specific risk one day's variance, with no lags, at every horizon", with
  "; set by --newey-west-lags L" (", overriding --horizon H") when the lags were given. At one
  day it says the covariance is the one-day covariance.
- **The identity, pinned** (`test_bartlett_weights_at_h_minus_one_lags_are_the_per_day_variance_
  of_h_day_returns`): on a series padded with h - 1 zeros at each end and summing to zero, so
  every window that reaches a day holds it whole and the centring takes nothing off,
  Newey-West at h - 1 lags equals the h-day windows' sum of squares over h(n - 1) to 1e-12, for
  h = 1, 2, 5, 11, 21; at one day it is the sample covariance to the bit. Two days k apart fall
  together in h - k of the h-day windows: per day 1 - k/h, Bartlett's weight at L = h - 1.
- **Measured** (`i1_compare.txt`): the default (H11) against the rulings' run (R1): equal in all
  21 fields, the report one line longer -- the horizon's. `--horizon 11 --newey-west-lags 10`,
  `--horizon 1` against `--newey-west-lags 0`, and `--horizon 21` against `--newey-west-lags 20`:
  each pair equal in all 21 fields, the reports differing only in the horizon line's "set by".
  Against the default, `--horizon 1` and `--horizon 21` move only the covariance and what is read
  from it -- the factor volatilities, the noise shares, the size pair's risk (10.99% and 12.08%
  for the smallest 85, against 11.40%) -- and the weighted noise share the qualification quotes
  (Wholesale's 61% and 89%, against 69%).

## Item 2: the bias statistics at the model's horizon

- **The statistic** (`model.bias`): every forecast date's portfolios and members bought and held
  h days, a name that stops trading held as cash, against the forecast at h times its daily
  variance; the eigenportfolios against the sum of the h periods' estimated factor returns.
  The windows overlap, each starting a day after the last, so the outcomes are an MA(h - 1) and
  the statistic varies more than 2/T says: for large T its spread is sqrt((2h^2 + 1) / 3h)
  times the independent one's -- 1.84 at 5, 2.71 at 11, 3.74 at 21 -- and the sample mean takes
  a little more off.
- **The band, found by simulation** (`overlapping_band`): 10,000 tests of right forecasts of
  standard normal returns (`OVERLAP_DRAWS`, seeded, `OVERLAP_SEED`), each window the sum of h
  draws over sqrt(h), each test's statistic read off its prefix sums so one simulation serves
  every T up to its size (1,024, or the next power of two), cached per horizon. Its 2.5 and 97.5
  percentiles are the band; it sits a shade below one. A test checks it: at T = 682, 676, 666 and
  h = 5, 11, 21 its half-width is the asymptotic one to 6%, its midpoint below one, and over
  4,000 tests drawn apart from its own simulation it covers 95% to 1.2 points.
- **Beside it, every h-th forecast from the first, the windows apart,** its band the plain
  1 +/- sqrt(2/T); and at h = 1 both are the daily statistic itself, the band included, so the
  daily statistic is exactly midoffice11's. A statistic read with an overlap carries no
  kurtosis-widened band, which holds for independent outcomes only.
- **The run** (`--bias`): every forecast is evaluated at one day and at the model's horizon, and
  at any other horizon `--bias-horizons` names; the report gives the portfolios at one day, then
  at the model's horizon with the windows overlapping (T, statistic, simulated band, MRAD, the
  rolling windows' range) and apart; the industries, random portfolios, members and
  eigenportfolios at both; and, when more horizons are evaluated, every horizon side by side.
  The `.npz` keeps the daily forecasts and outcomes and each horizon's outcomes with the windows
  overlapping; the forecast at h is the daily one times sqrt(h), and the windows apart are every
  h-th column. At h = 1 the model's horizon is the daily statistic, and the report says so.
- **The new defaults' bias run must equal H63T504.** The first 70 forecasts (a smoke run,
  `smoke_bias.py 70`) equalled H63T504's to the bit in every array -- the daily forecasts and
  outcomes, the members', the eigenportfolios', the repairs, the minimum-variance sizes, the caps,
  and the 21-day check taken apart the same way -- once BLAS ran on one thread, as the runner
  runs it (Failures). The whole run is below.

### The new defaults' bias run equals H63T504 (`equal_D.txt`)

The new defaults' run (D: the defaults, `--bias --bias-horizons 1,5,11,21`, the guard on) against
midoffice11's H63T504 over their 686 forecast dates: **every array equal to the bit** -- the
portfolios' daily forecasts and outcomes (226 by 686), the members' (686 by 592), the
eigenportfolios' (25 by 686), the PSD repairs, the minimum-variance portfolios' sizes, the caps
-- and midoffice11's 21-day check, its 32 outcomes taken apart from the new 21-day outcomes the
same way, forecasts and realised returns for the portfolios, the members and the
eigenportfolios. Its model report equals H63T504's line for line but the one new horizon line.
**The guard passed** on the new code and defaults, on the same five dates (2024-02-08,
2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02); the PSD repair bound on none of the 686 dates;
18 member-days had no next-day return. The run took 11.8 minutes with the guard.

### The new defaults at their own horizon, 11 days

| Portfolio | 1 day | 11 days, overlapping (band 0.852-1.141) | 11 days, apart (band 0.820-1.180) |
|---|---|---|---|
| the index, cap-weighted | 1.078 under | 0.986 in | 0.920 |
| the index, equally weighted | 1.008 in | 0.920 in | 0.907 |
| smallest 85 | 1.028 in | 0.912 in | 0.937 |
| largest 50 | 1.063 under | 0.973 in | 0.924 |
| minimum variance, long-only | 1.275 under | 1.184 under | 1.139 |
| minimum variance, unconstrained | 1.418 under | 1.278 under | 1.313 under |
| random long-only / long-short, median | 1.017 / 1.016 | 0.937 / 0.998 | 0.929 / 0.989 |
| members, median | 1.034 | 0.979 | 0.983 |
| industries, mean abs(bias - 1) | 0.066 | 0.038 | 0.050 |
| eigenportfolios 21-25, mean | 1.861 | 1.920 | 1.897 |

At its own horizon the defaults read inside the band for every broad portfolio, the industries
best of all (the thin three 0.996, 0.976, 0.963), the random long-short portfolios at 0.998.
What stays wrong at every horizon is what the optimiser finds: the minimum-variance portfolios
and the smallest eigenvalues.

## Item 3: the horizon, each model judged at its own

The candidates, each one change against the new defaults: `--horizon 1, 5, 21`. Every run was
evaluated at 1, 5, 11 and 21 days (`--bias-horizons`), so each can also be read at the others'
(`compare_horizons.txt`). The bands, simulated: +/-0.054 at one day, 0.900-1.093 at 5 (T 682),
0.852-1.141 at 11 (T 676), 0.790-1.189 at 21 (T 666); apart, +/-0.121, 0.180 and 0.250.

**Each judged at its own horizon, the windows overlapping:**

| | H1 at 1 | H5 at 5 | D at 11 | H21 at 21 |
|---|---|---|---|---|
| the index, cap-weighted | 1.015 | 0.991 | 0.986 | 1.006 |
| the index, equally weighted | 0.965 | 0.935 | 0.920 | 0.892 |
| smallest 85 | 0.956 | 0.946 | 0.912 | 0.866 |
| largest 50 | 0.996 | 0.978 | 0.973 | 1.019 |
| minimum variance, long-only | 1.167 | 1.204 | 1.184 | 1.177 |
| minimum variance, unconstrained | 1.220 | 1.298 | 1.278 | 1.308 |
| random long-only / long-short, median | 0.971 / 1.004 | 0.953 / 1.008 | 0.937 / 0.998 | 0.920 / 0.997 |
| industries, mean abs(bias - 1) | 0.046 | 0.044 | 0.038 | 0.054 |
| members, median | 1.003 | 0.985 | 0.979 | 0.981 |
| eigenportfolios 1-5, mean | 1.030 | 0.995 | 0.954 | 0.930 |
| eigenportfolios 21-25, mean | 1.194 | 1.461 | 1.920 | 2.686 |

- **Every model is right at its own horizon for the broad portfolios:** all four inside their
  bands, the cap-weighted index 0.99 to 1.02 throughout. The equally weighted index and the
  small caps drift below one as h grows (0.965 to 0.892, 0.956 to 0.866): over three weeks the
  small caps mean-revert more (a realised variance ratio of 0.64 at 21 days) than the 21-day
  model's lags hold them to (0.73 of the one-day variance).
- **The lags' cost is in the small eigenvalues, and grows with them:** the smallest five
  eigenportfolios 1.19, 1.46, 1.92, 2.69. The minimum-variance portfolios cost about the same at
  each own horizon (unconstrained 1.22 to 1.31), but only because at the longer horizons the
  realised returns mean-revert as well; read at the same horizon the one-day model is the least
  biased (below).

**What the lags cost** -- the smallest eigenportfolios and the minimum-variance portfolios, at
each model's horizon and at one day, beside the one-day model read at the same horizon:

| | H1 | H5 | D (11) | H21 |
|---|---|---|---|---|
| eigenportfolios 21-25, at h / apart / at 1 day | 1.194 / - / 1.194 | 1.461 / 1.446 / 1.427 | 1.920 / 1.897 / 1.861 | 2.686 / 2.753 / 2.670 |
| the one-day model's, at h | 1.194 | 1.197 | 1.167 | 1.159 |
| minimum variance, long-only, at h / apart / at 1 day | 1.167 / - / 1.167 | 1.204 / 1.237 / 1.191 | 1.184 / 1.139 / 1.275 | 1.177 / 1.062 / 1.360 |
| the one-day model's, at h | 1.167 | 1.173 | 1.078 | 1.005 |
| minimum variance, unconstrained, at h / apart / at 1 day | 1.220 / - / 1.220 | 1.298 / 1.345 / 1.294 | 1.278 / 1.313 / 1.418 | 1.308 / 1.264 / 1.629 |
| the one-day model's, at h | 1.220 | 1.223 | 1.110 | 1.045 |

- **At every horizon the one-day model forecasts the optimiser's portfolios and the small
  eigen-directions best.** At 21 days its unconstrained minimum-variance portfolio reads 1.045
  where the 21-day model's reads 1.308, its smallest eigenportfolios 1.16 where the 21-day
  model's read 2.69. The lags put structure in the small eigen-directions -- at 20 lags their
  forecast variance falls to 0.32-0.37 of the one-day model's -- that no realised horizon
  follows: estimation noise, which the optimiser finds.
- **A caution on the one-day model's 1.045 at 21 days:** it is two errors cancelling. At one day
  its unconstrained portfolio realises 22% more than its forecast (1.220), the optimisation
  bias; over 21 days the portfolio's returns mean-revert (a realised variance ratio of 0.70),
  and the two offset. A one-day model whose optimisation bias was corrected -- the eigenvalue
  adjustment's job -- would over-forecast that portfolio's 21-day risk by the mean reversion.
  The lags' own estimate of it is too strong: the 21-day model holds the portfolio at 0.56 of
  the one-day variance.

**The forecast levels and what the returns did** -- each model's median per-day variance over
the one-day model's, and the realised variance ratio over the test (the h-day returns' variance,
overlapping, over h times the one-day variance):

| | H5 | D (11) | H21 | realised at 5 / 11 / 21 |
|---|---|---|---|---|
| the index, cap-weighted | 0.903 | 0.816 | 0.682 | 0.854 / 0.759 / 0.702 |
| the index, equally weighted | 0.873 | 0.802 | 0.755 | 0.840 / 0.754 / 0.670 |
| smallest 85 | 0.877 | 0.771 | 0.732 | 0.854 / 0.728 / 0.643 |
| minimum variance, unconstrained | 0.902 | 0.758 | 0.561 | 0.951 / 0.783 / 0.696 |
| eigenportfolio 1 / 20 / 25 | 0.843 / 0.795 / 0.800 | 0.762 / 0.571 / 0.526 | 0.646 / 0.322 / 0.369 | -- |

The lags track the realised mean reversion of the factor-dominated index well (0.682 at 20 lags,
the 21-day horizon, against 0.702), under-reach the small caps', and over-reach in the
optimiser's portfolio and the small eigen-directions. **The forecasts are noisier from day to
day as h grows:** the daily change in the index's log forecast volatility has an sd of 0.011
under the one-day model, 0.012, 0.018 and 0.025 under the 5, 11 and 21-day ones.

**Every model at every horizon, overlapping** (rows the model, columns the horizon it is read at):

| | 1 day | 5 | 11 | 21 |
|---|---|---|---|---|
| the index, cap-weighted: H1 / H5 / D / H21 | 1.015 / 1.040 / 1.078 / 1.148 | 0.958 / 0.991 / 1.032 / 1.103 | 0.914 / 0.946 / 0.986 / 1.054 | 0.860 / 0.894 / 0.937 / 1.006 |
| the index, equally weighted | 0.965 / 0.973 / 1.008 / 1.042 | 0.926 / 0.935 / 0.971 / 1.004 | 0.872 / 0.883 / 0.920 / 0.953 | 0.805 / 0.820 / 0.859 / 0.892 |
| minimum variance, unconstrained | 1.220 / 1.294 / 1.418 / 1.629 | 1.223 / 1.298 / 1.407 / 1.627 | 1.110 / 1.179 / 1.278 / 1.469 | 1.045 / 1.078 / 1.159 / 1.308 |
| members, median | 1.003 / 1.019 / 1.034 / 1.044 | 0.971 / 0.985 / 1.002 / 1.015 | 0.956 / 0.962 / 0.979 / 0.994 | 0.938 / 0.951 / 0.967 / 0.981 |
| industries, mean abs(bias - 1) | 0.046 / 0.055 / 0.066 / 0.097 | 0.047 / 0.044 / 0.056 / 0.086 | 0.062 / 0.052 / 0.038 / 0.063 | 0.079 / 0.064 / 0.045 / 0.054 |

Read down a column: at each horizon the model whose lags match it reads the broad portfolios
nearest one (at 21 days the one-day model scaled by sqrt(21) reads the index 0.860, its forecast
16% above what was realised), while for the optimiser's portfolio the one-day model is nearest
at every horizon. The industries are best read by the model at the reading's own horizon, or
the one nearest it.

**The regime swamps the horizon within a half-year:** the cap-weighted index's statistic by
half-year runs 0.63 to 1.50 under the one-day model at one day, 0.36 to 1.55 under the 21-day
model at 21 days (2025H1, the April tariff shock, the high; 2025H2 the low).

## Item 4: the variant, built

- **`HorizonCovariance.ratio_half_life`** (the run's `--variance-ratio-half-life H`, off): each
  factor's variance is its one-period variance on the volatility half-life times its variance
  ratio at the lags -- its Newey-West variance over its one-period variance -- both on
  `ratio_half_life`; the correlations are untouched, the long window's Newey-West. At 504, the
  correlation half-life, the ratio's Newey-West matrix is the correlations' own. A factor with no
  movement over the ratio's window keeps its one-period variance. It needs the lags (and the
  run, the two-half-life covariance), and refuses a non-positive half-life.
- **Its diagonal is no quadratic form** -- a product of two estimates over a third -- so
  `volatility_estimator()` refuses it and the weighted noise share falls back on the
  equal-weighted one, as the shrunk estimator's does; the run reports the factor table's weighted
  column empty and the qualification quotes the equal-weighted share (Wholesale's 55%).
- **Tested:** each variance is level x lagged / one-period to 1e-12, at the correlations' and at
  another half-life; the correlations are the plain split's; on the volatility half-life it is the
  plain split, and at no lags it is the plain split to the bit (the ratio exactly one, the
  exponential and zero-lag Newey-West estimators being the same arithmetic). And why it exists:
  over 40 AR(1) factor panels of 1,500 days at rho 0.2 and 20 lags, its variances scatter about
  the Bartlett target at under 0.7 of the plain split's spread (0.57 in the probe, over 60
  panels and three values of rho) and sit no further from it (0.98 of it on average, against the
  plain split's 0.93).
- **Measured** (`i4_compare.txt`): the defaults after it was built (F11) equal item 1's (H11) in
  every field and line; at `--horizon 21` it moves only the covariance and what is read from it
  against `--horizon 21` alone (the covariance's largest change 0.0104 of annual variance, the
  size pair's risk for the smallest 85 12.08% to 11.75%).

## Item 4: the variant at 21, measured

H21R -- the 21-day model with each volatility its 90-day one-day volatility times the square
root of its variance ratio at 20 lags, the ratio on 504 -- against the 21-day model (H21), with
the one-day model (H1) and the defaults (D) read at the same horizons:

| | H1 | D | H21 | H21R |
|---|---|---|---|---|
| at 21 days: the index, cap-weighted / equally weighted | 0.860 / 0.805 | 0.937 / 0.859 | 1.006 / 0.892 | 0.973 / 0.840 |
| at 21 days: smallest 85 / largest 50 | 0.775 / 0.873 | 0.854 / 0.952 | 0.866 / 1.019 | 0.817 / 0.987 |
| at 21 days: minimum variance, long-only / unconstrained | 1.005 / 1.045 | 1.112 / 1.159 | 1.177 / 1.308 | 1.106 / 1.266 |
| at 21 days: eigenportfolios 21-25 / 1-5 | 1.159 / 0.913 | 1.901 / 0.921 | 2.686 / 0.930 | 2.403 / 0.874 |
| at 21 days: random long-only / long-short | 0.832 / 0.975 | 0.895 / 0.992 | 0.920 / 0.997 | 0.874 / 0.991 |
| at 21 days: industries, mean abs(bias - 1) | 0.079 | 0.045 | 0.054 | 0.064 |
| at 1 day: the index, cap-weighted / equally weighted | 1.015 / 0.965 | 1.078 / 1.008 | 1.148 / 1.042 | 1.120 / 0.992 |
| at 1 day: minimum variance, long-only / unconstrained | 1.167 / 1.220 | 1.275 / 1.418 | 1.360 / 1.629 | 1.284 / 1.532 |
| at 1 day: eigenportfolios 21-25 | 1.194 | 1.861 | 2.670 | 2.491 |
| the index's forecast, daily change in log volatility, sd | 0.0109 | 0.0177 | 0.0248 | 0.0141 |
| the index's MRAD at 21 days | 0.140 | 0.107 | 0.160 | 0.123 |
| forecast level against H1: the index / minimum variance, unconstrained | 1 / 1 | 0.816 / 0.758 | 0.682 / 0.561 | 0.743 / 0.621 |

- **It does what it is for, on the volatilities.** The forecast's day-to-day noise falls to near
  the one-day model's (the index 0.025 to 0.014, against 0.011; the unconstrained portfolio
  0.0146 to 0.0106), the index's MRAD at 21 days 0.160 to 0.123, and the optimiser's bias eases
  (unconstrained 1.308 to 1.266 at 21 days, 1.629 to 1.532 at one day).
- **It does not reach the correlations' lags,** which carry most of the eigenvalue bias: the
  smallest eigenportfolios 2.69 to 2.40, against the one-day model's 1.16.
- **Its level is the two-year window's,** and over 2024-26 the returns mean-reverted more than
  that window says: its forecasts are higher than the 21-day model's (the index at 0.743 of the
  one-day variance, against 0.682 and a realised 0.702), so the broad portfolios read further
  below one at 21 days (equally weighted 0.892 to 0.840, small caps 0.866 to 0.817, random
  long-only 0.920 to 0.874).
- So it gives up some of the 21-day model's broad-portfolio accuracy for less noise and buys back
  a little of the optimiser's: the 21-day model's virtue diluted, not the one-day model's gained.
  Its idea is right for the volatilities; what it leaves is the correlations' lags.

## Item 5: the specific history at 42

S42 against D (63), both on the new defaults (`compare_horizons.txt`):

| | 63 (D) | 42 (S42) |
|---|---|---|
| members at one day: median / mean abs(bias - 1) | 1.034 / 0.097 | 1.031 / 0.096 |
| p10 / p90 | 0.892 / 1.169 | 0.890 / 1.159 |
| below / above the band | 20.3% / 40.9% | 21.0% / 40.1% |
| below / above the kurtosis band | 10.7% / 17.9% | 11.0% / 16.4% |
| by size quintile, largest first | 1.089, 1.046, 1.034, 0.992, 1.005 | 1.080, 1.043, 1.038, 0.989, 1.006 |
| by half-year, 2024H1 to 2026H2 | 0.912, 1.004, 1.133, 0.906, 1.084, 0.950 | 0.916, 1.003, 1.134, 0.917, 1.066, 0.951 |
| members at 11 days: median / mean abs(bias - 1) | 0.979 / 0.117 | 0.981 / 0.116 |
| random long-short at one day: median / above the band | 1.016 / 37% | 1.012 / 35% |
| industries / the thin three, mean abs(bias - 1) | 0.066 / 0.053 | 0.064 / 0.051 |
| the broad and minimum-variance portfolios | | within 0.003 |

**The trend continues, and flattens.** From equal weights to 252, 126 and 63 (midoffice11) the
members' median fell 1.075, 1.057, 1.045, 1.042; from 63 to 42 on the new defaults it moves 1.034
to 1.031, the mean absolute deviation 0.097 to 0.096, the share past the kurtosis band 17.9% to
16.4%, the largest quintile 1.089 to 1.080. Every statistic moves by 0.01 or less but one
half-year's (2026H1, 1.084 to 1.066), and the shares by under two points, some the other way
(below the band 20.3% to 21.0%). Not clearly better: **63 stays**, as ruled.

---

## Which horizon I would choose for which use, and why

1. **One day, for daily risk** -- a one-day VaR, limits, daily monitoring, the day's P&L
   explained. At one day the one-day model is right for every broad portfolio (1.015, 0.965,
   0.956, 0.996, all in the band), the members (1.003, the largest quintile 1.054 where the
   11-day model's is 1.089), the random portfolios (0.971, 1.004), the industries (0.046). The
   11-day model, read at one day, puts the index at 1.078, outside the band.
2. **One day, for portfolio construction** -- optimisation, hedging, anything that looks for the
   low-risk directions -- at every holding period measured. Its minimum-variance portfolios are
   the least under-forecast at every horizon (unconstrained 1.22, 1.22, 1.11, 1.045 at 1, 5, 11,
   21 days; the 11-day model's 1.42, 1.41, 1.28, 1.16; the 21-day model's 1.63 to 1.31), and so
   are its smallest eigenportfolios (1.16 to 1.20, against 1.86 to 1.92 and 2.67 to 2.70). The
   lags put structure in the small eigen-directions that no horizon realises, and the optimiser
   finds it. What the one-day model leaves there, about 1.2, is the eigenvalue adjustment's to
   correct (ruling 4), on a matrix without the lags' noise.
3. **The holding period's horizon, for the risk of broad, factor-dominated portfolios over h
   days that were not built against the model** -- a two-week or a monthly risk report on
   long-only holdings. The model at that horizon reads the cap-weighted index 0.986 at 11 days
   and 1.006 at 21, where the one-day model scaled by sqrt(h) reads 0.914 and 0.860, its
   forecasts 9 and 16% above what was realised: the lags hold the index's mean reversion about
   right (0.682 at the 21-day horizon, a realised 0.702). The gain is that and no more. The
   small caps and the equally weighted index are over-forecast even under the 21-day model
   (0.866, 0.892), its forecasts are twice as noisy from day to day, and anything an optimiser
   built is worst under it. A reporting view, run at the reporting horizon (`--horizon 21`),
   not the covariance for everything.
4. **So one day as the model's default horizon.** One covariance should serve the two uses a
   risk model behind an optimiser exists for, daily risk and construction; it costs the third 9
   to 16% of conservatism on broad portfolios over 11 to 21 days, which a run at that horizon
   recovers. The 11-day default kept now serves none of them fully: at one day the index
   realises 7.8% more than its forecast and the optimiser's portfolio 42% more; it is right only
   for broad portfolios over 11 days.
5. **Not 5, and not the variant as built.** The 5-day model sits between the others on every
   measure and is best at none (the industries at 5, by 0.002). The variant quiets the
   volatilities and eases the optimiser's bias a little, but the correlations' 20 lags keep the
   smallest eigenportfolios at 2.4, and its two-year ratio over-forecasts the broad portfolios
   over 21 days more than the 21-day model does. If a long-horizon view is wanted beside the
   one-day model, the candidate it points to is lags on the volatilities alone, the correlations
   one-day: the 21-day level for the broad portfolios without the lags in the small
   eigen-directions. Not built; a candidate for Jeff.
6. **The specific history stays at 63:** 42 is not clearly better.

---

## Key decisions

1. **The new defaults live in the run, not the library** (`SPECIFIC_HALF_LIFE`,
   `CORRELATION_HALF_LIFE`, `HORIZON`), each with a `none` or an explicit value to get back the
   old form, as midoffice09 did with `specific_history`.
2. **`--newey-west-lags` now defaults to none,** so the horizon sets the lags unless they are
   given; the horizon in effect is always lags + 1, and the report says which flag set it.
3. **Every model was evaluated at every horizon** (`--bias-horizons 1,5,11,21`), beyond the
   brief's reading of each at its own: which horizon serves which use is read by setting each
   model against the use's horizon. It costs a buy-and-hold per date and horizon; the refits are
   the run's cost.
4. **The band:** one seeded simulation per horizon, 10,000 tests, serving every T through prefix
   sums; at one day the analytic band, so the daily statistic is midoffice11's exactly. Normal
   returns, as briefed; a statistic read with an overlap carries no kurtosis band.
5. **The `.npz` keeps each horizon's overlapping outcomes alone;** the forecast at h is the daily
   one times sqrt(h) and the windows apart are every h-th column, so nothing is stored twice and
   midoffice11's 21-day check is rebuilt from them to the bit.
6. **The variant's ratio has its own half-life** (`ratio_half_life`), the brief's 504 being the
   correlations'; at 504 it reuses their Newey-West matrix. Its diagonal is no quadratic form, so
   it has no weighted noise share.
7. **The guard ran on the new defaults alone;** the candidates' inputs are the defaults'.
8. **BLAS on one thread** in every run, so the runs compare to the bit.

## Decisions left for Jeff

1. **The model's default horizon:** one day (my choice); 11, as now; or another.
2. **The h-day risk of broad portfolios:** a run at the holding period's horizon as a reporting
   view (my choice), or the one-day model scaled by sqrt(h).
3. **Lags on the volatilities alone, the correlations one-day:** measure it as the long-horizon
   view's candidate (my choice), or not.
4. **The eigenvalue adjustment:** next, on the matrix the horizon ruling leaves (ruling 4) --
   about 1.2 to correct at the smallest eigenvalues and the optimiser's portfolio if one day.
5. **The variance-ratio variant:** kept as an option, off (my choice), or removed.
6. **The specific history:** 63 stays (my choice, the brief's rule).

---

## Tests

**The suite:** 2,343 passed, 1 skipped, 19 warnings (80.8 s in the final checks), with
`--ignore=tests/test_market_data.py`. Before this piece 2,338: 5 new tests, one extended, two
updated to the new interface.

**New, by file:**

`tests/midoffice/test_covariance.py`, 3:

- `test_bartlett_weights_at_h_minus_one_lags_are_the_per_day_variance_of_h_day_returns` -- the
  horizon identity, to 1e-12 at h = 1, 2, 5, 11, 21 on a padded series; one day the sample
  covariance to the bit;
- `test_variance_ratio_volatilities_take_the_level_short_and_the_serial_correlation_long` -- the
  variant's arithmetic at two ratio half-lives, its correlations, its two special cases (to
  rounding and to the bit) and its refusals;
- `test_variance_ratio_volatilities_carry_less_of_the_lags_noise` -- 40 AR(1) panels: under 0.7
  of the plain split's spread, no further from the Bartlett target.

`tests/midoffice/test_bias.py`, 2, and 2 updated:

- `test_the_overlapping_band_is_simulated_and_wider_and_at_one_period_the_plain_one` -- the band
  at one period is `bias_band`'s; at 5, 11 and 21 its half-width is sqrt((2h^2 + 1) / 3h) times
  the independent one to 6%, its midpoint below one, the same on every reading; 95% coverage to
  1.2 points on 4,000 tests drawn apart from the simulation; a statistic read with an overlap
  takes that band and no kurtosis band;
- `test_at_a_horizon_every_forecast_is_held_h_periods_and_every_hth_is_apart` -- on the world
  whose risk is known: the h-day outcome is the buy-and-hold of the block, the forecast sqrt(h)
  times the daily one, the last h - 1 dates empty; the members' and eigenportfolios' h-day
  outcomes; the truth inside twice the simulated half-width, a 20% shortfall exactly its
  inverse; every h-th apart (139 at h = 5); one period the daily statistic itself; an
  unevaluated horizon refused by name;
- `test_forecasts_that_are_the_truth_score_one_...` and `test_a_member_with_no_next_return_...`
  read the 21-day check through the new interface (`horizons=`, `statistics(horizon=,
  overlapping=)`).

`tests/midoffice/test_crosssectional.py`, 1 extended:

- `test_the_two_half_life_estimators_diagonal_has_sampling_weights_and_the_shrunk_one_none` --
  the variance-ratio estimator has no sampling weights and no weighted share, and the model
  still fits.

**Hygiene:** every changed file parses, is ASCII and keeps its line endings (`final_checks.txt`).
**ruff:** clean over every changed Python file.

**The final code, re-run** (`final_checks.txt`): the defaults (FD) equal F11 in all 21 fields,
6 qualifications and 162 report lines, and so item 1's H11, the rulings' R1 and midoffice11's
code with the two options (A0) but for the horizon line; the old defaults (FDold, `none none`)
equal midoffice11's final defaults (its B4) in all 21 fields and 7 qualifications, the report
differing only by the horizon line.

---

## Failures and debugging trail

- **The session's window was closed once,** after the brief was pasted and before any file was
  touched; the work resumed with the state file and midoffice11's retro already read. The first
  clock reading, a compound shell command, had failed with exit code 107 just before; a plain
  `date` read 09:10 EST.
- **Multi-threaded BLAS moved the portfolio forecasts in the last bit.** The first smoke run's
  70 forecasts equalled H63T504's in every array but the portfolios' forecast volatilities, 520
  cells off by at most 3.5e-18. The portfolio variances are a matrix product (weights by
  exposures), which a multi-threaded BLAS sums in another order; midoffice11's runner ran BLAS
  on one thread. Rerun on one thread, the smoke run equalled H63T504 in every array to the bit,
  and this piece's runner pins one thread too. The state file now says so.
- **A test asserted too much.** The first version of the horizon-outcomes test required each of
  three portfolios inside its simulated 95% band; one industry of the known world read 1.100
  against 0.901-1.091, by chance -- the band is 95% per portfolio, and the known world's
  industries share factors. It now asks for twice the half-width, as the daily check beside it
  does; the band's own coverage is tested separately, on independent draws.
- **The harness's shell drifted** into the scratch directory once; every command uses the
  repo's absolute path.
- **A `diff <(...)` was tried once,** against the state file's working note, and replaced by a
  diff of two files.
- **The runs themselves:** none failed. The runner ran the six one at a time from 09:44 to
  10:39 EST, each started with 4.7 to 6.5 GB available, 8.3 to 9.1 minutes each and the
  defaults' 11.8 with the guard; no retry, no wait for memory, and the memory guard was not
  reached (`bias/runner_log.txt`).

## Problems found, not fixed

1. **The lags' noise in the small eigen-directions grows with the horizon** (1.19, 1.46, 1.92,
   2.69 at 1, 5, 11, 21) and the variant does not reach it: it is in the correlations' lags.
2. **The optimiser's portfolio is under-forecast at every setting at one day** (unconstrained
   1.22 at best); its near-one at 21 days under the one-day model is two errors cancelling.
3. **The small caps mean-revert more than any model's lags hold them to** (a realised 0.64 at 21
   days; the 21-day model 0.73), so their long-horizon risk is over-forecast under every model.
4. **The regime moves the statistics more than the horizon does** within a half-year: 0.36 to
   1.55 for the index under the 21-day model at 21 days.
5. **The largest names stay under-forecast at one day** (the top quintile 1.054 under the one-day
   model, 1.089 under the 11-day one): midoffice11's leverages.
6. **Six ruff findings in untouched files,** at HEAD, and **`tests/test_market_data.py`'s
   `../core_repo/src`,** as before.

## Scratch

`claude/scratch/midoffice12/` (gitignored):

- **Harnesses:** `capture.py`, `compare.py`, `hygiene.py` (midoffice11's, renamed);
  `smoke_bias.py` (the main loop truncated); `runner.py`, its log and done marker in `bias/`;
  `patch_bias_report.py` (the run's bias report replaced between two markers, CRLF kept);
  `final_checks.sh`.
- **Measures:** `equal_h63t504.py` (a run against midoffice11's H63T504, bit for bit);
  `compare_horizons.py`; `probe_ratio.py` (the variant's noise on AR(1) panels); `probe_test.py`.
- **Captures:** `base/` (A0), `r1/` (R1, R1old), `i1/` (H11, H11L, H1, H1L, H21, H21L), `i4/`
  (F11, H21R), `final/` (FD, FDold), each a report and a pickle; `smoke/` (the smoke runs, their
  checks, and the comparison script's trial on copies of the smoke run).
- **Bias runs, `bias/`:** each run's report, progress and `.npz` -- D, H1, H5, H21, H21R, S42.
- **Outputs:** `r1_compare.txt`, `i1_compare.txt`, `i4_compare.txt`, `equal_D.txt`,
  `compare_horizons.txt`, `final_checks.txt`.

The key ones are copied into the appendix. Nothing in `data/` was written, overwritten or
deleted.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

## Open items for Chat

- **The six decisions above.** The horizon ruling comes first: the eigenvalue adjustment waits
  on it (ruling 4).
- **Lags on the volatilities alone** is the one long-horizon candidate this piece's results point
  to, if a long-horizon view is wanted beside a one-day default.
- **The bias statistics are standing,** now at the model's horizon and any other: a later
  candidate is one `--bias --bias-horizons ...` run against the defaults, through the runner.

---

## Appendix: outputs, verbatim

Each output as its script wrote it, from `claude/scratch/midoffice12/`.

### r1_compare.txt -- rulings 1 and 2: the new defaults against midoffice11's code with the two options (A0), and `none none` against midoffice11's final defaults (B4)

```text
claude/scratch/midoffice12/base/A0 against claude/scratch/midoffice12/r1/R1: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 161 and 161 lines; 0 differing line(s)
claude/scratch/midoffice11/final/B4 against claude/scratch/midoffice12/r1/R1old: 21 fields compared; equal: 21; differing: none
  qualifications: 7 and 7; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
```

### i1_compare.txt -- item 1: the horizon's default against the rulings' run; each horizon against its lags given; 1 and 21 against 11

```text
claude/scratch/midoffice12/r1/R1 against claude/scratch/midoffice12/i1/H11: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 161 and 162 lines; 1 differing line(s)
    +  the horizon: 11 trading days -- the factor covariance Newey-West at 10 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 11-day returns; the specific risk one day's variance, with no lags, at every horizon
claude/scratch/midoffice12/i1/H11 against claude/scratch/midoffice12/i1/H11L: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 2 differing line(s)
    -  the horizon: 11 trading days -- the factor covariance Newey-West at 10 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 11-day returns; the specific risk one day's variance, with no lags, at every horizon
    +  the horizon: 11 trading days -- the factor covariance Newey-West at 10 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 11-day returns; the specific risk one day's variance, with no lags, at every horizon; set by --newey-west-lags 10
claude/scratch/midoffice12/i1/H1 against claude/scratch/midoffice12/i1/H1L: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 2 differing line(s)
    -  the horizon: 1 trading day -- the factor covariance the one-day covariance, with no Newey-West lags, and the specific risk one day's variance
    +  the horizon: 1 trading day -- the factor covariance the one-day covariance, with no Newey-West lags, and the specific risk one day's variance; set by --newey-west-lags 0
claude/scratch/midoffice12/i1/H21 against claude/scratch/midoffice12/i1/H21L: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 2 differing line(s)
    -  the horizon: 21 trading days -- the factor covariance Newey-West at 20 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 21-day returns; the specific risk one day's variance, with no lags, at every horizon
    +  the horizon: 21 trading days -- the factor covariance Newey-West at 20 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 21-day returns; the specific risk one day's variance, with no lags, at every horizon; set by --newey-west-lags 20
claude/scratch/midoffice12/i1/H11 against claude/scratch/midoffice12/i1/H1: 21 fields compared; equal: 16; differing: covariance, observed_variance, weighted_sampling_variance, noise_share, weighted_noise_share
  covariance: largest difference 1.780e-02
  observed_variance: largest difference 1.780e-02
  weighted_sampling_variance: largest difference 7.434e-04
  noise_share: largest difference 1.504e-01
  weighted_noise_share: largest difference 1.985e-01
  qualifications: 6 and 6; differing: 1
  report: 162 and 162 lines; 60 differing line(s)
    -  the horizon: 11 trading days -- the factor covariance Newey-West at 10 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 11-day returns; the specific risk one day's variance, with no lags, at every horizon
    -  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 10 lag(s), split horizon, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    +  the horizon: 1 trading day -- the factor covariance the one-day covariance, with no Newey-West lags, and the specific risk one day's variance
    +  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 0 lag(s), split horizon, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    -  smallest 85, equally weighted                -3.137             -1.808   11.40%
    -  largest 50, equally weighted                 +0.167             +0.052    0.60%
    +  smallest 85, equally weighted                -3.137             -1.808   10.99%
    +  largest 50, equally weighted                 +0.167             +0.052    0.57%
    -  Construction               19.42%     +24.76%    38.2%      48.6%      938          0      0
    -  Finance & Real Estate      15.34%     +18.11%    31.0%      46.5%      938          0      0
    -  Manufacturing              10.20%     +20.88%    23.2%      33.5%      938          0      0
    -  Mining                     18.60%     +14.94%    18.5%      27.2%      938          0      0
    -  Retail                     13.11%     +14.63%    10.7%      16.1%      938          0      0
    -  SIC20                      11.96%     +11.03%    33.6%      55.5%      938          0      0
    -  SIC28                      13.14%     +16.64%    10.2%      15.7%      938          0      0
    -  SIC35                      18.81%     +29.69%     5.2%       7.0%      938          0      0
    -  SIC36                      21.13%     +32.55%     4.5%       5.3%      938          0      0
    -  SIC37                      14.22%     +15.42%    19.1%      27.3%      938          0      0
    -  SIC38                      14.70%     +15.82%    10.6%      15.9%      938          0      0
    -  SIC49                      15.80%     +24.01%    11.3%      16.9%      938          0      0
    -  SIC60                      21.85%     +21.97%     7.4%      10.5%      938          0      0
    -  SIC62                      18.73%     +18.62%    11.5%      15.0%      938          0      0
    -  SIC63                      14.57%     +19.17%    14.5%      22.7%      938          0      0
    -  SIC67                      15.01%     +18.00%    16.8%      25.4%      938          0      0
    -  SIC73                      18.18%     +15.21%     2.9%       4.5%      938          0      0
    -  Services                   12.72%     +16.11%    28.0%      42.6%      938          0      0
    -  Transport & Utilities      13.43%     +15.46%    16.5%      25.1%      938          0      0
    -  Wholesale                  13.98%     +20.69%    46.2%      68.8%      938          0      0
    -  Volatility                  9.37%      +3.09%     3.5%       5.7%      938          0      0
    -  Momentum                    7.22%      +4.23%     3.6%       5.1%      938          0      0
    -  Size                        3.72%      +3.13%     7.7%      10.7%      938          0      0
    -  Size Nonlinearity           2.32%      -0.53%    46.6%      67.4%      938          0      0
    -  Trading Activity            6.25%      +2.48%     7.5%      10.5%      938          0      0
    +  Construction               22.22%     +24.76%    29.2%      38.7%      938          0      0
    +  Finance & Real Estate      17.38%     +18.11%    24.1%      37.8%      938          0      0
    +  Manufacturing              12.12%     +20.88%    16.4%      24.7%      938          0      0
    +  Mining                     21.20%     +14.94%    14.3%      21.8%      938          0      0
    +  Retail                     14.77%     +14.63%     8.4%      13.2%      938          0      0
    +  SIC20                      13.35%     +11.03%    27.0%      46.4%      938          0      0
    +  SIC28                      16.09%     +16.64%     6.8%      10.9%      938          0      0
    +  SIC35                      18.55%     +29.69%     5.4%       7.5%      938          0      0
    +  SIC36                      24.22%     +32.55%     3.4%       4.2%      938          0      0
    +  SIC37                      19.50%     +15.42%    10.2%      15.1%      938          0      0
    +  SIC38                      16.79%     +15.82%     8.1%      12.7%      938          0      0
    +  SIC49                      19.08%     +24.01%     7.8%      12.1%      938          0      0
    +  SIC60                      23.21%     +21.97%     6.6%       9.7%      938          0      0
    +  SIC62                      21.57%     +18.62%     8.7%      11.8%      938          0      0
    +  SIC63                      14.25%     +19.17%    15.2%      24.7%      938          0      0
    +  SIC67                      17.79%     +18.00%    11.9%      18.8%      938          0      0
    +  SIC73                      17.52%     +15.21%     3.2%       5.0%      938          0      0
    +  Services                   14.38%     +16.11%    21.9%      34.7%      938          0      0
    +  Transport & Utilities      16.21%     +15.46%    11.3%      18.0%      938          0      0
    +  Wholesale                  15.21%     +20.69%    39.0%      60.5%      938          0      0
    +  Volatility                 10.64%      +3.09%     2.7%       4.6%      938          0      0
    +  Momentum                    7.50%      +4.23%     3.4%       4.9%      938          0      0
    +  Size                        3.59%      +3.13%     8.2%      11.9%      938          0      0
    +  Size Nonlinearity           2.81%      -0.53%    31.6%      47.6%      938          0      0
    +  Trading Activity            5.31%      +2.48%    10.4%      15.1%      938          0      0
    -  - 69% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is biased upward and is not corrected here
    +  - 61% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is biased upward and is not corrected here
claude/scratch/midoffice12/i1/H11 against claude/scratch/midoffice12/i1/H21: 21 fields compared; equal: 16; differing: covariance, observed_variance, weighted_sampling_variance, noise_share, weighted_noise_share
  covariance: largest difference 6.873e-03
  observed_variance: largest difference 6.873e-03
  weighted_sampling_variance: largest difference 6.986e-04
  noise_share: largest difference 1.633e-01
  weighted_noise_share: largest difference 2.068e-01
  qualifications: 6 and 6; differing: 1
  report: 162 and 162 lines; 60 differing line(s)
    -  the horizon: 11 trading days -- the factor covariance Newey-West at 10 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 11-day returns; the specific risk one day's variance, with no lags, at every horizon
    -  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 10 lag(s), split horizon, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    +  the horizon: 21 trading days -- the factor covariance Newey-West at 20 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 21-day returns; the specific risk one day's variance, with no lags, at every horizon
    +  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 20 lag(s), split horizon, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    -  smallest 85, equally weighted                -3.137             -1.808   11.40%
    -  largest 50, equally weighted                 +0.167             +0.052    0.60%
    +  smallest 85, equally weighted                -3.137             -1.808   12.08%
    +  largest 50, equally weighted                 +0.167             +0.052    0.63%
    -  Construction               19.42%     +24.76%    38.2%      48.6%      938          0      0
    -  Finance & Real Estate      15.34%     +18.11%    31.0%      46.5%      938          0      0
    -  Manufacturing              10.20%     +20.88%    23.2%      33.5%      938          0      0
    -  Mining                     18.60%     +14.94%    18.5%      27.2%      938          0      0
    -  Retail                     13.11%     +14.63%    10.7%      16.1%      938          0      0
    -  SIC20                      11.96%     +11.03%    33.6%      55.5%      938          0      0
    -  SIC28                      13.14%     +16.64%    10.2%      15.7%      938          0      0
    -  SIC35                      18.81%     +29.69%     5.2%       7.0%      938          0      0
    -  SIC36                      21.13%     +32.55%     4.5%       5.3%      938          0      0
    -  SIC37                      14.22%     +15.42%    19.1%      27.3%      938          0      0
    -  SIC38                      14.70%     +15.82%    10.6%      15.9%      938          0      0
    -  SIC49                      15.80%     +24.01%    11.3%      16.9%      938          0      0
    -  SIC60                      21.85%     +21.97%     7.4%      10.5%      938          0      0
    -  SIC62                      18.73%     +18.62%    11.5%      15.0%      938          0      0
    -  SIC63                      14.57%     +19.17%    14.5%      22.7%      938          0      0
    -  SIC67                      15.01%     +18.00%    16.8%      25.4%      938          0      0
    -  SIC73                      18.18%     +15.21%     2.9%       4.5%      938          0      0
    -  Services                   12.72%     +16.11%    28.0%      42.6%      938          0      0
    -  Transport & Utilities      13.43%     +15.46%    16.5%      25.1%      938          0      0
    -  Wholesale                  13.98%     +20.69%    46.2%      68.8%      938          0      0
    -  Volatility                  9.37%      +3.09%     3.5%       5.7%      938          0      0
    -  Momentum                    7.22%      +4.23%     3.6%       5.1%      938          0      0
    -  Size                        3.72%      +3.13%     7.7%      10.7%      938          0      0
    -  Size Nonlinearity           2.32%      -0.53%    46.6%      67.4%      938          0      0
    -  Trading Activity            6.25%      +2.48%     7.5%      10.5%      938          0      0
    +  Construction               18.77%     +24.76%    40.8%      50.0%      938          0      0
    +  Finance & Real Estate      14.65%     +18.11%    34.0%      49.0%      938          0      0
    +  Manufacturing               9.37%     +20.88%    27.5%      38.1%      938          0      0
    +  Mining                     18.99%     +14.94%    17.8%      25.1%      938          0      0
    +  Retail                     12.16%     +14.63%    12.4%      18.0%      938          0      0
    +  SIC20                      11.92%     +11.03%    33.8%      53.7%      938          0      0
    +  SIC28                      11.99%     +16.64%    12.3%      18.1%      938          0      0
    +  SIC35                      20.55%     +29.69%     4.4%       5.6%      938          0      0
    +  SIC36                      21.00%     +32.55%     4.6%       5.2%      938          0      0
    +  SIC37                      13.07%     +15.42%    22.6%      31.0%      938          0      0
    +  SIC38                      12.30%     +15.82%    15.2%      21.8%      938          0      0
    +  SIC49                      14.56%     +24.01%    13.3%      19.1%      938          0      0
    +  SIC60                      22.28%     +21.97%     7.1%       9.7%      938          0      0
    +  SIC62                      19.40%     +18.62%    10.7%      13.4%      938          0      0
    +  SIC63                      14.60%     +19.17%    14.5%      21.7%      938          0      0
    +  SIC67                      15.62%     +18.00%    15.5%      22.5%      938          0      0
    +  SIC73                      16.88%     +15.21%     3.4%       5.0%      938          0      0
    +  Services                   11.13%     +16.11%    36.6%      53.4%      938          0      0
    +  Transport & Utilities      12.30%     +15.46%    19.7%      28.8%      938          0      0
    +  Wholesale                  12.01%     +20.69%    62.5%      89.5%      938          0      0
    +  Volatility                  8.35%      +3.09%     4.4%       6.8%      938          0      0
    +  Momentum                    7.24%      +4.23%     3.6%       4.9%      938          0      0
    +  Size                        3.88%      +3.13%     7.1%       9.5%      938          0      0
    +  Size Nonlinearity           2.32%      -0.53%    46.4%      64.5%      938          0      0
    +  Trading Activity            6.65%      +2.48%     6.6%       8.9%      938          0      0
    -  - 69% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is biased upward and is not corrected here
    +  - 89% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is biased upward and is not corrected here
```

### smoke/equal_smoke.txt and equal_smoke1.txt -- the smoke run's 70 forecasts against H63T504, BLAS multi-threaded, then on one thread

```text
claude/scratch/midoffice12/smoke/smoke.npz against claude/scratch/midoffice11/bias/H63T504.npz: 70 forecast dates in common (70 and 686), 2024-02-08 to 2024-05-17
  forecast                                     DIFFERS, largest difference 3.469e-18  (226, 70)
  realised                                     equal to the bit  (226, 70)
  eigen_forecast                               equal to the bit  (25, 70)
  eigen_realised                               equal to the bit  (25, 70)
  member_forecast                              equal to the bit  (70, 592)
  member_realised                              equal to the bit  (70, 592)
  caps                                         equal to the bit  (70, 592)
  repaired                                     equal to the bit  (70,)
  held_long_only                               equal to the bit  (70,)
  held_unconstrained                           equal to the bit  (70,)
  the 21-day check: 32 outcomes in midoffice11's, 4 in the new run's, 4 in common
  21-day dates                                 equal to the bit  (4,)
  21-day forecast (daily x sqrt(21))           DIFFERS, largest difference 2.776e-17  (226, 4)
  21-day realised                              equal to the bit  (226, 4)
  21-day members' forecast                     equal to the bit  (4, 592)
  21-day members' realised                     equal to the bit  (4, 592)
  21-day eigen forecast                        equal to the bit  (25, 4)
  21-day eigen realised                        equal to the bit  (25, 4)
DIFFERING: forecast, 21-day forecast (daily x sqrt(21))
[...]
claude/scratch/midoffice12/smoke/smoke1.npz against claude/scratch/midoffice11/bias/H63T504.npz: 70 forecast dates in common (70 and 686), 2024-02-08 to 2024-05-17
  forecast                                     equal to the bit  (226, 70)
  realised                                     equal to the bit  (226, 70)
  eigen_forecast                               equal to the bit  (25, 70)
  eigen_realised                               equal to the bit  (25, 70)
  member_forecast                              equal to the bit  (70, 592)
  member_realised                              equal to the bit  (70, 592)
  caps                                         equal to the bit  (70, 592)
  repaired                                     equal to the bit  (70,)
  held_long_only                               equal to the bit  (70,)
  held_unconstrained                           equal to the bit  (70,)
  the 21-day check: 32 outcomes in midoffice11's, 4 in the new run's, 4 in common
  21-day dates                                 equal to the bit  (4,)
  21-day forecast (daily x sqrt(21))           equal to the bit  (226, 4)
  21-day realised                              equal to the bit  (226, 4)
  21-day members' forecast                     equal to the bit  (4, 592)
  21-day members' realised                     equal to the bit  (4, 592)
  21-day eigen forecast                        equal to the bit  (25, 4)
  21-day eigen realised                        equal to the bit  (25, 4)
every array equal to the bit
```

### i4_compare.txt -- item 4: the defaults after the variant was built against item 1's; the variant at 21 against 21

```text
claude/scratch/midoffice12/i1/H11 against claude/scratch/midoffice12/i4/F11: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
claude/scratch/midoffice12/i1/H21 against claude/scratch/midoffice12/i4/H21R: 21 fields compared; equal: 16; differing: covariance, observed_variance, weighted_sampling_variance, noise_share, weighted_noise_share
  covariance: largest difference 1.035e-02
  observed_variance: largest difference 1.035e-02
  weighted_sampling_variance: largest difference nan
  noise_share: largest difference 1.573e-01
  weighted_noise_share: largest difference nan
  qualifications: 6 and 6; differing: 1
  report: 162 and 162 lines; 60 differing line(s)
    -  the horizon: 21 trading days -- the factor covariance Newey-West at 20 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 21-day returns; the specific risk one day's variance, with no lags, at every horizon
    -  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 20 lag(s), split horizon, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    +  the horizon: 21 trading days -- the factor correlations Newey-West at 20 lag(s) with Bartlett weights, which is exactly the per-day variance of 21-day returns, and each factor volatility its one-day volatility on a half-life of 90 times the square root of its variance ratio at 20 lags, the ratio estimated on a half-life of 504: the serial correlation from the long window, the level from the short; the specific risk one day's variance, with no lags, at every horizon
    +  25-factor model, annualised, estimated on 938 daily period(s) for 550 of 592 security(ies); covariance Newey-West, 20 lag(s), split horizon, each volatility one period's times its variance ratio on 504, half-life 90 vol / 504 corr, 826.4 effective observation(s) of 938
    -  smallest 85, equally weighted                -3.137             -1.808   12.08%
    -  largest 50, equally weighted                 +0.167             +0.052    0.63%
    +  smallest 85, equally weighted                -3.137             -1.808   11.75%
    +  largest 50, equally weighted                 +0.167             +0.052    0.60%
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

### equal_D.txt -- the new defaults' bias run against H63T504, bit for bit

```text
claude/scratch/midoffice12/bias/D.npz against claude/scratch/midoffice11/bias/H63T504.npz: 686 forecast dates in common (686 and 686), 2024-02-08 to 2026-11-02
  forecast                                     equal to the bit  (226, 686)
  realised                                     equal to the bit  (226, 686)
  eigen_forecast                               equal to the bit  (25, 686)
  eigen_realised                               equal to the bit  (25, 686)
  member_forecast                              equal to the bit  (686, 592)
  member_realised                              equal to the bit  (686, 592)
  caps                                         equal to the bit  (686, 592)
  repaired                                     equal to the bit  (686,)
  held_long_only                               equal to the bit  (686,)
  held_unconstrained                           equal to the bit  (686,)
  the 21-day check: 32 outcomes in midoffice11's, 32 in the new run's, 32 in common
  21-day dates                                 equal to the bit  (32,)
  21-day forecast (daily x sqrt(21))           equal to the bit  (226, 32)
  21-day realised                              equal to the bit  (226, 32)
  21-day members' forecast                     equal to the bit  (32, 592)
  21-day members' realised                     equal to the bit  (32, 592)
  21-day eigen forecast                        equal to the bit  (25, 32)
  21-day eigen realised                        equal to the bit  (25, 32)
every array equal to the bit
```

### bias/report_D.txt -- the new defaults' bias statistics, the run's own report

```text
Out-of-sample bias statistics
  686 forecasts, as of each trading day from 2024-02-08 -- the first whose covariance has a year (252) of estimated periods -- to 2026-11-02: each the whole estimator refitted on data through its date alone, daily -- the factor returns, the covariance, the specific risk model's fit and every history it blends, the exposures as of the date
  the industry scheme is held fixed over the sample, as Barra fixes a model's industry factors: built once over the full sample's regression universe, 20 industries
  each forecast taken to a daily scale (annual / 252) against the next trading day's total return; the bias statistic is the standard deviation of realised / forecast, one if the forecasts are right, its 95% band about 1 +/- sqrt(2/T); over the whole test, and over every 12-month (252-forecast) window with the windows' mean absolute deviation from one (MRAD)
  and at the model's horizon, 11 trading days: every forecast, at 11 times its daily variance, against the 11 trading days' total return after it, bought and held, a name that stops trading held as cash -- the windows overlapping, so their 95% band is found by simulation, right forecasts of normal returns (10,000 draws) -- and beside it every 11th forecast from the first, the windows apart, its band 1 +/- sqrt(2/T)
  also evaluated at 5, 21 trading days (--bias-horizons), the same way, summarised after the eigenportfolios
  guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model's inputs rebuilt from the panel cut at the date, the scheme held, equal the full run's through it to the bit -- returns, exposures, weights, capitalisations, periods -- and so do the forecasts made from them: covariance, exposures, specific risk, factor returns
  686 refits in 7.9 min; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every portfolio formed that day

  Portfolios at one day, each formed at its forecast date and held to the next
                                         T    bias           95% band    MRAD    min    max
  the index, cap-weighted              686   1.078 under  0.946-1.054   0.190  0.849  1.364
  the index, equally weighted          686   1.008    in  0.946-1.054   0.148  0.871  1.239
  smallest 85, equally weighted        686   1.028    in  0.946-1.054   0.110  0.849  1.212
  largest 50, equally weighted         686   1.063 under  0.946-1.054   0.153  0.835  1.309
  minimum variance, long-only          686   1.275 under  0.946-1.054   0.291  0.995  1.502
  minimum variance, unconstrained      686   1.418 under  0.946-1.054   0.436  1.041  1.729

  Portfolios at the model's horizon, 11 trading days, each bought at its forecast date and held 11 days: every forecast, the windows overlapping, its band simulated; and every 11th, the windows apart
                                         T    bias           95% band    MRAD    min    max   apart: T   bias           95% band
  the index, cap-weighted              676   0.986    in  0.852-1.141   0.135  0.803  1.263     62  0.920    in  0.820-1.180
  the index, equally weighted          676   0.920    in  0.852-1.141   0.137  0.709  1.156     62  0.907    in  0.820-1.180
  smallest 85, equally weighted        676   0.912    in  0.852-1.141   0.076  0.795  1.031     62  0.937    in  0.820-1.180
  largest 50, equally weighted         676   0.973    in  0.852-1.141   0.164  0.721  1.254     62  0.924    in  0.820-1.180
  minimum variance, long-only          676   1.184 under  0.852-1.141   0.160  0.908  1.318     62  1.139    in  0.820-1.180
  minimum variance, unconstrained      676   1.278 under  0.852-1.141   0.225  0.883  1.475     62  1.313 under  0.820-1.180
    minimum variance, long-only: 85 names held at the median (49 to 171); forecast 8.00% a year at the median, realised 10.59% from its daily returns; its KKT conditions met to 3.2e-13
    minimum variance, unconstrained: 501 names held at the median (499 to 503); forecast 7.33% a year at the median, realised 10.94% from its daily returns; its KKT conditions met to 3.2e-13

  Each industry's members, cap-weighted -- the scheme's 20 as built -- against its factor's noise share in the full run (equal / weighted)
  industry                 members     T   1 day          MRAD   11 days        apart      noise share
  Construction                   8   686   1.064 under   0.039     0.996    in  1.004    38.2% / 48.6%
  Finance & Real Estate         12   686   1.077 under   0.122     0.976    in  0.977    31.0% / 46.5%
  Manufacturing                 39   686   1.005    in   0.098     1.032    in  1.007    23.2% / 33.5%
  Mining                        16   686   0.936  over   0.060     0.941    in  0.862    18.5% / 27.2%
  Retail                        34   686   1.094 under   0.129     1.066    in  1.079    10.7% / 16.1%
  SIC20                         21   686   1.097 under   0.135     1.023    in  0.952    33.6% / 55.5%
  SIC28                         40   686   1.123 under   0.148     1.111    in  1.152    10.2% / 15.7%
  SIC35                         34   686   1.017    in   0.129     1.005    in  0.866      5.2% / 7.0%
  SIC36                         36   686   1.235 under   0.224     1.085    in  0.969      4.5% / 5.3%
  SIC37                         15   686   1.106 under   0.134     1.004    in  0.974    19.1% / 27.3%
  SIC38                         44   686   1.103 under   0.157     1.008    in  0.950    10.6% / 15.9%
  SIC49                         37   686   1.051    in   0.096     0.973    in  0.952    11.3% / 16.9%
  SIC60                         20   686   0.977    in   0.110     0.986    in  0.971     7.4% / 10.5%
  SIC62                         20   686   1.047    in   0.105     1.009    in  0.997    11.5% / 15.0%
  SIC63                         25   686   1.037    in   0.172     1.009    in  1.009    14.5% / 22.7%
  SIC67                         29   686   1.001    in   0.077     0.933    in  0.925    16.8% / 25.4%
  SIC73                         70   686   1.091 under   0.151     1.084    in  1.067      2.9% / 4.5%
  Services                      20   686   0.952    in   0.052     0.986    in  0.982    28.0% / 42.6%
  Transport & Utilities         32   686   0.974    in   0.096     0.920    in  0.952    16.5% / 25.1%
  Wholesale                     12   686   1.017    in   0.066     0.963    in  1.013    46.2% / 68.8%
  across the 20 with a factor of their own, the one-day bias statistic's correlation with the weighted noise share: -0.242

  Random portfolios, long-only, seed 11: 100 of 30 members, weights drawn once per member -- at one day, bias median 1.017, p5 0.942, p95 1.105; below the band 6%, above it 17%; MRAD median 0.132
    at 11 days, the windows overlapping: median 0.937, p5 0.845, p95 1.041; below the simulated band 8%, above it 0%; apart, median 0.929

  Random portfolios, long-short, seed 11: 100 of 30 members a side, weights drawn once per member -- at one day, bias median 1.016, p5 0.923, p95 1.181; below the band 13%, above it 37%; MRAD median 0.067
    at 11 days, the windows overlapping: median 0.998, p5 0.874, p95 1.150; below the simulated band 3%, above it 7%; apart, median 0.989

  Every member singly: 543 securities with 63 forecasts or more -- at one day, bias p5 0.841, p25 0.960, median 1.034, p75 1.107, p95 1.220
    below its band (over-forecast) 20.3%, above it (under-forecast) 40.9%, against 2.5% each for right forecasts of normal returns; with each member's band widened for its own kurtosis (median 9.2), 10.7% and 17.9%
    median by capitalisation quintile over the test, largest first: 1.089, 1.046, 1.034, 0.992, 1.005
    at 11 days, the windows overlapping: median 0.979, p5 0.754, p95 1.213; below its simulated band 14.0%, above it 12.7% (the band for normal returns)
    apart, every 11th forecast, members with 12 outcomes or more: median 0.983

  The factor covariance's eigenportfolios against the next period's estimated factor returns, by rank, 1 the largest eigenvalue; at 11 days against the sum of the 11 periods' after the forecast
  rank  vol, median     T   1 day          MRAD   11 days        apart
     1       54.77%   686   1.090 under   0.175     0.965    in  0.888
     2       21.27%   686   0.975    in   0.049     0.902    in  0.913
     3       18.11%   686   0.992    in   0.040     0.905    in  0.863
     4       16.06%   686   1.049    in   0.052     1.014    in  0.951
     5       14.16%   686   1.071 under   0.077     0.983    in  1.024
     6       12.33%   686   1.126 under   0.090     0.988    in  0.896
     7       11.37%   686   1.123 under   0.138     1.202 under  1.220
     8       10.17%   686   1.130 under   0.137     1.140    in  1.117
     9        9.50%   686   1.234 under   0.263     1.250 under  1.405
    10        8.12%   686   1.262 under   0.263     1.134    in  1.169
    11        7.50%   686   1.275 under   0.234     1.479 under  1.306
    12        7.01%   686   1.382 under   0.337     1.333 under  1.226
    13        6.71%   686   1.178 under   0.169     1.418 under  1.374
    14        6.24%   686   1.406 under   0.385     1.492 under  1.578
    15        5.94%   686   1.466 under   0.487     1.342 under  1.287
    16        5.68%   686   1.452 under   0.468     1.572 under  1.511
    17        5.29%   686   1.572 under   0.581     1.661 under  1.748
    18        4.95%   686   1.608 under   0.546     1.604 under  1.547
    19        4.55%   686   1.776 under   0.764     1.801 under  1.694
    20        4.26%   686   1.745 under   0.699     1.750 under  1.829
    21        3.82%   686   1.884 under   0.833     1.989 under  2.069
    22        3.24%   686   1.852 under   0.863     1.788 under  1.649
    23        2.65%   686   1.856 under   0.748     1.890 under  1.861
    24        1.99%   686   1.879 under   0.850     2.157 under  2.143
    25        1.35%   686   1.836 under   0.750     1.776 under  1.762

  At every horizon evaluated, each forecast at h times its daily variance: the statistic with the windows overlapping / apart
                                             1 day        5 days       11 days       21 days
  the index, cap-weighted                    1.078   1.032/1.051   0.986/0.920   0.937/0.933
  the index, equally weighted                1.008   0.971/0.979   0.920/0.907   0.859/0.831
  smallest 85, equally weighted              1.028   0.985/0.987   0.912/0.937   0.854/0.793
  largest 50, equally weighted               1.063   1.019/1.034   0.973/0.924   0.952/1.005
  minimum variance, long-only                1.275   1.291/1.328   1.184/1.139   1.112/1.030
  minimum variance, unconstrained            1.418   1.407/1.472   1.278/1.313   1.159/1.115
  random long-only, median                   1.017   0.985/0.995   0.937/0.929   0.895/0.866
  random long-short, median                  1.016   1.009/1.013   0.998/0.989   0.992/1.014
  every member singly, median                1.034   1.002/1.010   0.979/0.983   0.967/0.980
  industries, mean abs(bias - 1)             0.066   0.056/0.065   0.038/0.050   0.045/0.090
  eigenportfolios 1-5, mean                  1.035   0.981/0.980   0.954/0.928   0.921/0.899
  eigenportfolios 21-25, mean                1.861   1.892/1.859   1.920/1.897   1.901/2.015
  every forecast and outcome written to C:\Data\Development\Python\McTheoryApps\praxis\claude\scratch\midoffice12\bias\D.tmp.npz
```

### compare_horizons.txt -- every run at every horizon, each statistic recomputed from its stored forecasts and outcomes

```text
6 runs: D = the new defaults, horizon 11; H1 = horizon 1; H5 = horizon 5; H21 = horizon 21; H21R = horizon 21, variance-ratio volatilities; S42 = specific history half-life 42
686 forecasts, 2024-02-08 to 2026-11-02; each run evaluated at 1, 5, 11, 21 trading days

The bands at each horizon (simulated for overlapping windows, normal returns)
  h =  1: overlapping T = 686, band 0.946-1.054 (half-width 0.054); apart T = 686, band 0.946-1.054
  h =  5: overlapping T = 682, band 0.900-1.093 (half-width 0.096); apart T = 137, band 0.879-1.121
  h = 11: overlapping T = 676, band 0.852-1.141 (half-width 0.144); apart T = 62, band 0.820-1.180
  h = 21: overlapping T = 666, band 0.790-1.189 (half-width 0.199); apart T = 32, band 0.750-1.250

Each model judged at its own horizon, the windows overlapping
                                            H1@1      H5@5      D@11    H21@21   H21R@21
  the index, cap-weighted                  1.015     0.991     0.986     1.006     0.973
  the index, equally weighted              0.965     0.935     0.920     0.892     0.840
  smallest 85, equally weighted            0.956     0.946     0.912     0.866     0.817
  largest 50, equally weighted             0.996     0.978     0.973     1.019     0.987
  minimum variance, long-only              1.167     1.204     1.184     1.177     1.106
  minimum variance, unconstrained          1.220     1.298     1.278     1.308     1.266
  random long-only, median                 0.971     0.953     0.937     0.920     0.874
  random long-short, median                1.004     1.008     0.998     0.997     0.991
  industries, mean |bias - 1|              0.046     0.044     0.038     0.054     0.064
  members, median                          1.003     0.985     0.979     0.981     0.960
  eigenportfolios 1-5, mean                1.030     0.995     0.954     0.930     0.874
  eigenportfolios 21-25, mean              1.194     1.461     1.920     2.686     2.403

Each model judged at its own horizon, the windows apart
                                            H1@1      H5@5      D@11    H21@21   H21R@21
  the index, cap-weighted                  1.015     1.011     0.920     1.035     0.974
  the index, equally weighted              0.965     0.941     0.907     0.874     0.810
  smallest 85, equally weighted            0.956     0.945     0.937     0.812     0.761
  largest 50, equally weighted             0.996     0.994     0.924     1.107     1.049
  minimum variance, long-only              1.167     1.237     1.139     1.062     0.999
  minimum variance, unconstrained          1.220     1.345     1.313     1.264     1.232
  random long-only, median                 0.971     0.961     0.929     0.895     0.846
  random long-short, median                1.004     1.012     0.989     1.021     1.016
  industries, mean |bias - 1|              0.046     0.050     0.050     0.089     0.097
  eigenportfolios 21-25, mean              1.194     1.446     1.897     2.753     2.463

  Verdicts at the own horizon, overlapping (in / over / under):
  the index, cap-weighted                     in        in        in        in        in
  the index, equally weighted                 in        in        in        in        in
  smallest 85, equally weighted               in        in        in        in        in
  largest 50, equally weighted                in        in        in        in        in
  MRAD at the own horizon, overlapping:
  the index, cap-weighted                  0.198     0.162     0.135     0.160     0.123
  the index, equally weighted              0.176     0.161     0.137     0.116     0.135
  smallest 85, equally weighted            0.128     0.090     0.076     0.121     0.175
  largest 50, equally weighted             0.164     0.164     0.164     0.187     0.148
  minimum variance, long-only              0.231     0.249     0.160     0.149     0.099
  minimum variance, unconstrained          0.253     0.310     0.225     0.188     0.188

the index, cap-weighted: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    1.015    0.958    0.914    0.860    0.978    0.838    0.848
  H5 (horizon 5)                                    1.040    0.991    0.946    0.894    1.011    0.875    0.879
  D (the new defaults, horizon 11)                  1.078    1.032    0.986    0.937    1.051    0.920    0.933
  H21 (horizon 21)                                  1.148    1.103    1.054    1.006    1.123    0.989    1.035
  H21R (horizon 21, variance-ratio volatilities)    1.120    1.069    1.023    0.973    1.092    0.949    0.974

the index, equally weighted: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    0.965    0.926    0.872    0.805    0.928    0.843    0.776
  H5 (horizon 5)                                    0.973    0.935    0.883    0.820    0.941    0.864    0.786
  D (the new defaults, horizon 11)                  1.008    0.971    0.920    0.859    0.979    0.907    0.831
  H21 (horizon 21)                                  1.042    1.004    0.953    0.892    1.017    0.939    0.874
  H21R (horizon 21, variance-ratio volatilities)    0.992    0.946    0.898    0.840    0.958    0.877    0.810

smallest 85, equally weighted: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    0.956    0.908    0.836    0.775    0.903    0.849    0.721
  H5 (horizon 5)                                    0.989    0.946    0.873    0.812    0.945    0.894    0.755
  D (the new defaults, horizon 11)                  1.028    0.985    0.912    0.854    0.987    0.937    0.793
  H21 (horizon 21)                                  1.041    0.996    0.925    0.866    1.002    0.949    0.812
  H21R (horizon 21, variance-ratio volatilities)    0.989    0.940    0.872    0.817    0.942    0.889    0.761

largest 50, equally weighted: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    0.996    0.945    0.903    0.873    0.961    0.843    0.915
  H5 (horizon 5)                                    1.024    0.978    0.933    0.908    0.994    0.876    0.946
  D (the new defaults, horizon 11)                  1.063    1.019    0.973    0.952    1.034    0.924    1.005
  H21 (horizon 21)                                  1.124    1.082    1.037    1.019    1.098    0.991    1.107
  H21R (horizon 21, variance-ratio volatilities)    1.098    1.050    1.006    0.987    1.068    0.950    1.049

minimum variance, long-only: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    1.167    1.173    1.078    1.005    1.194    1.032    0.983
  H5 (horizon 5)                                    1.191    1.204    1.094    1.009    1.237    1.044    0.957
  D (the new defaults, horizon 11)                  1.275    1.291    1.184    1.112    1.328    1.139    1.030
  H21 (horizon 21)                                  1.360    1.377    1.257    1.177    1.430    1.226    1.062
  H21R (horizon 21, variance-ratio volatilities)    1.284    1.305    1.195    1.106    1.362    1.150    0.999

minimum variance, unconstrained: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    1.220    1.223    1.110    1.045    1.235    1.094    1.062
  H5 (horizon 5)                                    1.294    1.298    1.179    1.078    1.345    1.171    1.081
  D (the new defaults, horizon 11)                  1.418    1.407    1.278    1.159    1.472    1.313    1.115
  H21 (horizon 21)                                  1.629    1.627    1.469    1.308    1.706    1.556    1.264
  H21R (horizon 21, variance-ratio volatilities)    1.532    1.546    1.405    1.266    1.630    1.466    1.232

random long-only, median: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    0.971    0.934    0.891    0.832    0.942    0.870    0.813
  H5 (horizon 5)                                    0.982    0.953    0.907    0.859    0.961    0.890    0.827
  D (the new defaults, horizon 11)                  1.017    0.985    0.937    0.895    0.995    0.929    0.866
  H21 (horizon 21)                                  1.050    1.015    0.964    0.920    1.030    0.955    0.895
  H21R (horizon 21, variance-ratio volatilities)    1.003    0.969    0.916    0.874    0.979    0.902    0.846

random long-short, median: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    1.004    0.999    0.981    0.975    1.002    0.972    0.995
  H5 (horizon 5)                                    1.011    1.008    0.989    0.985    1.012    0.977    1.008
  D (the new defaults, horizon 11)                  1.016    1.009    0.998    0.992    1.013    0.989    1.014
  H21 (horizon 21)                                  1.029    1.015    1.007    0.997    1.024    0.990    1.021
  H21R (horizon 21, variance-ratio volatilities)    1.022    1.005    0.999    0.991    1.018    0.983    1.016

industries, mean |bias - 1|: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    0.046    0.047    0.062    0.079    0.048    0.090    0.129
  H5 (horizon 5)                                    0.055    0.044    0.052    0.064    0.050    0.075    0.110
  D (the new defaults, horizon 11)                  0.066    0.056    0.038    0.045    0.065    0.050    0.090
  H21 (horizon 21)                                  0.097    0.086    0.063    0.054    0.096    0.053    0.089
  H21R (horizon 21, variance-ratio volatilities)    0.071    0.065    0.060    0.064    0.076    0.069    0.097

members, median: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    1.003    0.971    0.956    0.938    0.980    0.954    0.947
  H5 (horizon 5)                                    1.019    0.985    0.962    0.951    0.990    0.968    0.960
  D (the new defaults, horizon 11)                  1.034    1.002    0.979    0.967    1.010    0.983    0.980
  H21 (horizon 21)                                  1.044    1.015    0.994    0.981    1.023    0.996    0.993
  H21R (horizon 21, variance-ratio volatilities)    1.025    0.993    0.974    0.960    1.006    0.975    0.971

eigenportfolios 1-5, mean: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    1.030    0.973    0.945    0.913    0.981    0.920    0.871
  H5 (horizon 5)                                    1.052    0.995    0.971    0.938    0.999    0.942    0.895
  D (the new defaults, horizon 11)                  1.035    0.981    0.954    0.921    0.980    0.928    0.899
  H21 (horizon 21)                                  1.033    0.995    0.966    0.930    1.004    0.908    0.912
  H21R (horizon 21, variance-ratio volatilities)    0.988    0.947    0.914    0.874    0.953    0.863    0.850

eigenportfolios 21-25, mean: each model (rows) at each horizon (columns), overlapping, then apart
                                                      1 d      5 d     11 d     21 d  5 apart 11 apart 21 apart
  H1 (horizon 1)                                    1.194    1.197    1.167    1.159    1.198    1.157    1.199
  H5 (horizon 5)                                    1.427    1.461    1.468    1.468    1.446    1.456    1.498
  D (the new defaults, horizon 11)                  1.861    1.892    1.920    1.901    1.859    1.897    2.015
  H21 (horizon 21)                                  2.670    2.678    2.696    2.686    2.658    2.623    2.753
  H21R (horizon 21, variance-ratio volatilities)    2.491    2.474    2.444    2.403    2.461    2.349    2.463

What each model's lags cost: the smallest eigenportfolios and the minimum-variance portfolios, at its horizon and at one day, beside the one-day model (H1) at the same horizon

  H1 (horizon 1), its horizon 1
                                            at h     apart     1 day      H1 at h  H1 1 day
  eigenportfolios 21-25, mean              1.194         -     1.194        1.194     1.194
  eigenportfolios 16-20, mean              1.216         -     1.216        1.216     1.216
  minimum variance, long-only              1.167         -     1.167        1.167     1.167
  minimum variance, unconstrained          1.220         -     1.220        1.220     1.220

  H5 (horizon 5), its horizon 5
                                            at h     apart     1 day      H1 at h  H1 1 day
  eigenportfolios 21-25, mean              1.461     1.446     1.427        1.197     1.194
  eigenportfolios 16-20, mean              1.392     1.403     1.391        1.256     1.216
  minimum variance, long-only              1.204     1.237     1.191        1.173     1.167
  minimum variance, unconstrained          1.298     1.345     1.294        1.223     1.220

  D (the new defaults, horizon 11), its horizon 11
                                            at h     apart     1 day      H1 at h  H1 1 day
  eigenportfolios 21-25, mean              1.920     1.897     1.861        1.167     1.194
  eigenportfolios 16-20, mean              1.678     1.666     1.631        1.256     1.216
  minimum variance, long-only              1.184     1.139     1.275        1.078     1.167
  minimum variance, unconstrained          1.278     1.313     1.418        1.110     1.220

  H21 (horizon 21), its horizon 21
                                            at h     apart     1 day      H1 at h  H1 1 day
  eigenportfolios 21-25, mean              2.686     2.753     2.670        1.159     1.194
  eigenportfolios 16-20, mean              2.338     2.303     2.156        1.272     1.216
  minimum variance, long-only              1.177     1.062     1.360        1.005     1.167
  minimum variance, unconstrained          1.308     1.264     1.629        1.045     1.220

  H21R (horizon 21, variance-ratio volatilities), its horizon 21
                                            at h     apart     1 day      H1 at h  H1 1 day
  eigenportfolios 21-25, mean              2.403     2.463     2.491        1.159     1.194
  eigenportfolios 16-20, mean              2.213     2.187     2.031        1.272     1.216
  minimum variance, long-only              1.106     0.999     1.284        1.005     1.167
  minimum variance, unconstrained          1.266     1.232     1.532        1.045     1.220

The forecast levels: each model's median per-day variance over the one-day model's (its forecast's square over H1's), by portfolio
                                              H1        H5         D       H21      H21R
  the index, cap-weighted                  1.000     0.903     0.816     0.682     0.743
  the index, equally weighted              1.000     0.873     0.802     0.755     0.771
  smallest 85, equally weighted            1.000     0.877     0.771     0.732     0.781
  largest 50, equally weighted             1.000     0.886     0.807     0.687     0.743
  minimum variance, long-only              1.000     0.928     0.792     0.699     0.693
  minimum variance, unconstrained          1.000     0.902     0.758     0.561     0.621
  industry: Construction                   1.000     0.840     0.842     0.855     0.914
  industry: SIC36                          1.000     0.889     0.794     0.721     0.746
  eigenportfolio 1                         1.000     0.843     0.762     0.646     0.724
  eigenportfolio 5                         1.000     0.856     0.761     0.716     0.774
  eigenportfolio 10                        1.000     0.947     0.855     0.717     0.833
  eigenportfolio 15                        1.000     0.890     0.697     0.533     0.631
  eigenportfolio 20                        1.000     0.795     0.571     0.322     0.374
  eigenportfolio 25                        1.000     0.800     0.526     0.369     0.440

How much each forecast moves from day to day, the lags' noise among it: the sd of the daily change in the log of the forecast volatility, by portfolio
                                              H1        H5         D       H21      H21R
  the index, cap-weighted                 0.0109    0.0124    0.0177    0.0248    0.0141
  the index, equally weighted             0.0177    0.0166    0.0213    0.0250    0.0170
  smallest 85, equally weighted           0.0196    0.0174    0.0216    0.0240    0.0182
  largest 50, equally weighted            0.0202    0.0207    0.0246    0.0295    0.0216
  minimum variance, long-only             0.0103    0.0110    0.0130    0.0163    0.0112
  minimum variance, unconstrained         0.0092    0.0102    0.0117    0.0146    0.0106

The realised variance ratio over the test: the variance of the h-day returns, overlapping, over h times the one-day variance (the returns' own serial correlation)
                                             5 d      11 d      21 d
  the index, cap-weighted                  0.854     0.759     0.702
  the index, equally weighted              0.840     0.754     0.670
  smallest 85, equally weighted            0.854     0.728     0.643
  largest 50, equally weighted             0.859     0.768     0.749
  minimum variance, long-only              0.952     0.800     0.700
  minimum variance, unconstrained          0.951     0.783     0.696

The variance-ratio variant at 21 (H21R) against H21, and D and H1 read at the same horizons, overlapping
                                                        H1         D       H21      H21R
  the index, cap-weighted, at 21                     0.860     0.937     1.006     0.973
  the index, equally weighted, at 21                 0.805     0.859     0.892     0.840
  smallest 85, equally weighted, at 21               0.775     0.854     0.866     0.817
  largest 50, equally weighted, at 21                0.873     0.952     1.019     0.987
  minimum variance, long-only, at 21                 1.005     1.112     1.177     1.106
  minimum variance, unconstrained, at 21             1.045     1.159     1.308     1.266
  the index, cap-weighted, at 1                      1.015     1.078     1.148     1.120
  the index, equally weighted, at 1                  0.965     1.008     1.042     0.992
  smallest 85, equally weighted, at 1                0.956     1.028     1.041     0.989
  largest 50, equally weighted, at 1                 0.996     1.063     1.124     1.098
  minimum variance, long-only, at 1                  1.167     1.275     1.360     1.284
  minimum variance, unconstrained, at 1              1.220     1.418     1.629     1.532
  eigenportfolios 21-25, at 21                       1.159     1.901     2.686     2.403
  eigenportfolios 1-5, at 21                         0.913     0.921     0.930     0.874
  random long-only, at 21                            0.832     0.895     0.920     0.874
  random long-short, at 21                           0.975     0.992     0.997     0.991
  industries mean |bias - 1|, at 21                  0.079     0.045     0.054     0.064
  eigenportfolios 21-25, at 1                        1.194     1.861     2.670     2.491
  eigenportfolios 1-5, at 1                          1.030     1.035     1.033     0.988
  random long-only, at 1                             0.971     1.017     1.050     1.003
  random long-short, at 1                            1.004     1.016     1.029     1.022
  industries mean |bias - 1|, at 1                   0.046     0.066     0.097     0.071
  the index, cap-weighted, MRAD at 21                0.140     0.107     0.160     0.123
  the index, equally weighted, MRAD at 21            0.178     0.124     0.116     0.135
  minimum variance, unconstrained, MRAD at 21        0.099     0.133     0.188     0.188

The specific history at 42 (S42) against 63 (D)
                                                         D       S42
  members at 1: median                               1.034     1.031
  members at 1: mean |bias - 1|                      0.097     0.096
  members at 1: p10                                  0.892     0.890
  members at 1: p90                                  1.169     1.159
  members at 1: below the band                       0.203     0.210
  members at 1: above the band                       0.409     0.401
  members at 1: above the kurtosis band              0.179     0.164
  members at 1: below the kurtosis band              0.107     0.110
  members at 1: median MRAD                          0.097     0.097
  members at 11: median                              0.979     0.981
  members at 11: mean |bias - 1|                     0.117     0.116
  members at 11: p10                                 0.810     0.805
  members at 11: p90                                 1.170     1.173
  members at 11: below the band                      0.140     0.151
  members at 11: above the band                      0.127     0.131
  random long-short at 1, median                     1.016     1.012
  random long-short at 1, above the band             0.370     0.350
  random long-only at 1, median                      1.017     1.016
  industries at 1, mean |bias - 1|                   0.066     0.064
  the thin three at 1, mean |bias - 1|               0.053     0.051
  random long-short at 11, median                    0.998     0.993
  random long-short at 11, above the band            0.070     0.070
  random long-only at 11, median                     0.937     0.935
  industries at 11, mean |bias - 1|                  0.038     0.037
  the thin three at 11, mean |bias - 1|              0.022     0.023
  the index, cap-weighted, at 1                      1.078     1.077
  the index, equally weighted, at 1                  1.008     1.008
  smallest 85, equally weighted, at 1                1.028     1.028
  largest 50, equally weighted, at 1                 1.063     1.063
  minimum variance, long-only, at 1                  1.275     1.278
  minimum variance, unconstrained, at 1              1.418     1.419

Members at one day by capitalisation quintile, largest first: median bias
                                                         D       S42
  quintile 1 (109)                                   1.089     1.080
  quintile 2 (109)                                   1.046     1.043
  quintile 3 (109)                                   1.034     1.038
  quintile 4 (108)                                   0.992     0.989
  quintile 5 (108)                                   1.005     1.006

Members' median one-day bias statistic by half-year
                                                         D       S42
  2024H1 (98)                                        0.912     0.916
  2024H2 (128)                                       1.004     1.003
  2025H1 (122)                                       1.133     1.134
  2025H2 (128)                                       0.906     0.917
  2026H1 (123)                                       1.084     1.066
  2026H2 (87)                                        0.950     0.951

the index, cap-weighted: by half-year, each model at one day and at its own horizon (overlapping windows starting in the half)
                           H1@1     H5@1     H5@5      D@1     D@11    H21@1   H21@21   H21R@1  H21R@21
  2024H1 (98)             0.882    0.891    0.817    0.892    0.840    0.933    0.832    0.911    0.813
  2024H2 (128)            1.135    1.145    1.168    1.175    1.058    1.305    0.891    1.231    0.843
  2025H1 (122)            1.503    1.503    1.396    1.542    1.399    1.655    1.549    1.617    1.457
  2025H2 (128)            0.627    0.693    0.629    0.759    0.504    0.788    0.364    0.752    0.349
  2026H1 (123)            0.903    0.970    0.907    1.025    1.044    1.050    1.163    1.071    1.178
  2026H2 (87)             0.703    0.745    0.691    0.787    0.638    0.812    0.461    0.816    0.469

the index, equally weighted: by half-year, each model at one day and at its own horizon (overlapping windows starting in the half)
                           H1@1     H5@1     H5@5      D@1     D@11    H21@1   H21@21   H21R@1  H21R@21
  2024H1 (98)             0.823    0.815    0.810    0.791    0.774    0.751    0.654    0.726    0.635
  2024H2 (128)            1.014    0.981    1.108    1.005    0.996    1.030    0.888    0.969    0.815
  2025H1 (122)            1.443    1.407    1.326    1.442    1.300    1.510    1.245    1.438    1.157
  2025H2 (128)            0.680    0.743    0.676    0.813    0.554    0.865    0.544    0.786    0.495
  2026H1 (123)            0.843    0.908    0.758    0.965    0.846    0.996    0.866    0.977    0.844
  2026H2 (87)             0.716    0.771    0.640    0.813    0.843    0.852    0.985    0.821    0.947
```

### bias/runner_log.txt -- the six runs, one at a time

```text
2026-11-12 09:44:01  runner started, pid 10488; 4.7 GB available
2026-11-12 09:44:01  D: 4.7 GB available
2026-11-12 09:44:01  D: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 5
2026-11-12 09:55:49  D: done in 11.8 min
2026-11-12 09:55:49  H1: 5.1 GB available
2026-11-12 09:55:49  H1: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 0 --horizon 1
2026-11-12 10:04:43  H1: done in 8.9 min
2026-11-12 10:04:43  H5: 6.5 GB available
2026-11-12 10:04:43  H5: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 0 --horizon 5
2026-11-12 10:13:41  H5: done in 9.0 min
2026-11-12 10:13:41  H21: 6.0 GB available
2026-11-12 10:13:41  H21: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 0 --horizon 21
2026-11-12 10:22:30  H21: done in 8.8 min
2026-11-12 10:22:30  H21R: 6.4 GB available
2026-11-12 10:22:30  H21R: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 0 --horizon 21 --variance-ratio-half-life 504
2026-11-12 10:30:49  H21R: done in 8.3 min
2026-11-12 10:30:49  S42: 6.5 GB available
2026-11-12 10:30:49  S42: attempt 1: --industries --membership --bias --bias-horizons 1,5,11,21 --bias-guard 0 --specific-half-life 42
2026-11-12 10:39:53  S42: done in 9.1 min
2026-11-12 10:39:53  runner finished; every run done
```

### final_checks.txt -- the final code: the defaults and the old defaults against their references; the suite, ruff, hygiene

```text
FD: status 0, mean R2 0.293203, 938 periods, 25 factors, 58s, the estimator 1.3s
claude/scratch/midoffice12/i4/F11 against claude/scratch/midoffice12/final/FD: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
FDold: status 0, mean R2 0.293203, 938 periods, 25 factors, 55s, the estimator 1.3s
claude/scratch/midoffice11/final/B4 against claude/scratch/midoffice12/final/FDold: 21 fields compared; equal: 21; differing: none
  qualifications: 7 and 7; differing: 0
  report: 162 and 163 lines; 1 differing line(s)
    +  the horizon: 11 trading days -- the factor covariance Newey-West at 10 lag(s) with Bartlett weights, on the volatilities and the correlations alike, which is exactly the per-day variance of 11-day returns; the specific risk one day's variance, with no lags, at every horizon
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2343 passed, 1 skipped, 19 warnings in 80.80s (0:01:20) ===========
All checks passed!
ruff: clean over the changed Python files
scripts/run_sp500_model.py: CRLF (1982 lines), parses, ASCII
src/praxis/midoffice/model/bias.py: LF (601 lines), parses, ASCII
src/praxis/midoffice/model/covariance.py: LF (626 lines), parses, ASCII
tests/midoffice/test_bias.py: LF (271 lines), parses, ASCII
tests/midoffice/test_covariance.py: LF (837 lines), parses, ASCII
tests/midoffice/test_crosssectional.py: CRLF (2060 lines), parses, ASCII
docs/FACTOR_MODEL.md: CRLF (1309 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (574 lines), -, ASCII
claude/handoffs/BRIEF_midoffice12_horizon.md: LF (66 lines), -, ASCII
all ok
```

## git add (always last -- Rule 43)

This piece -- midoffice11's rulings, the horizon as a parameter, the bias statistics at the model's horizon, the variance-ratio variant, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add scripts/run_sp500_model.py src/praxis/midoffice/model/bias.py src/praxis/midoffice/model/covariance.py tests/midoffice/test_bias.py tests/midoffice/test_covariance.py tests/midoffice/test_crosssectional.py docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice12_horizon.md claude/retros/RETRO_midoffice12_horizon.md
```
