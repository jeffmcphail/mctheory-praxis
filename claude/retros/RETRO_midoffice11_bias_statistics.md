# Retro: Mid-office 11 -- midoffice10's rulings, then the out-of-sample bias statistics

**Brief:** `claude/handoffs/BRIEF_midoffice11_bias_statistics.md` -- written in Chat and pasted
into the Code session, transcribed verbatim
**Date:** 2026-11-10 to 2026-11-12 (EST)
**Duration:** the brief arrived shortly before 2026-11-11T04:07Z (23:07 EST on 11-10). The
rulings, the machinery and its tests, and the first bias runs to about 00:00 EST; the runs were
stopped there by Claude Code's memory guard (Failures, below). Jeff's instructions for the
rerun came on 11-12; the runner ran the eleven runs and the diagnosis one at a time from 03:12
EST, and the final checks, the docs, the state file and this retro followed.
**Status:** COMPLETE.
- **Rulings 1 and 2:** built, tested, measured. The specific side of the correction is an
  option, off (`--leverage-correction`); the weighted noise share is always computed, reported
  beside the equal-weighted one, and quoted by the qualification.
- **Rulings 3 to 5:** nothing to build (no fold, Size Nonlinearity kept, FRCB as built). The two
  unused imports are removed.
- **Items 1 to 4:** built and measured. No default changed.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 1, the correction's specific side.** The algebra holds, with one qualification: the
  estimator already inflates every residual by n/(n - k), the leverage correction at the mean
  leverage, so under the correction as built a member is short by its leverage less the mean.
  On the real model the mean is 0.050; Construction's members average 0.156 (11% short); and
  the largest leverages are the mega caps -- TSLA 0.357, AAPL 0.319, NVDA 0.306 -- whose specific
  return their industry factor absorbs. Built as an option, off (`--leverage-correction`); it
  raises TSLA's specific risk from 32.3% to 39.0%.
- **Ruling 2:** the weighted noise share always computed, beside the equal-weighted one in the
  factor table, and quoted by the qualification (Wholesale's 69%). The two unused imports are
  removed, and a test that never ran (a duplicate name) runs again.
- **Item 1, the forecasts:** refitted daily -- each the whole estimator on its inputs cut at the
  date, about a second a refit -- 686 of them from 2024-02-08, the scheme held fixed. The guard
  rebuilt the inputs from the panel cut at five dates through the run's own pipeline and found
  them the full run's to the bit, for the defaults and for floor 15.
- **The defaults sit above one at the daily horizon, and that is Newey-West's horizon.** The
  broad portfolios read 1.07 to 1.12 (band +/-0.054), the 21-day check 0.88 to 1.02. Daily
  returns mean-revert, and ten lags hold the index's factor variance at 0.805 of its one-day
  variance, which alone predicts 1.114; without the lags the daily reads 1.00 to 1.02. Members
  read 1.075 at the median, the largest names worst (1.124).
- **The lag terms are most of the eigenvalue bias:** the smallest eigenportfolios read 2.2 with
  them and 1.25 without, the unconstrained minimum-variance portfolio 1.58 and 1.24.
- **The candidates, each against the defaults:**
  - **floor 15 hurts** (Construction's portfolio 1.066 to 1.319);
  - **the correction** hurts the optimiser's portfolio at both horizons (1.578 to 1.925) and
    centres the industries only at 21 days; its specific side moves little;
  - **a 63-day specific-risk half-life** improves every specific-side measure (members 1.075 to
    1.042, past the kurtosis band 36% to 20%), 126 and 252 less;
  - **the covariance at 45 is worse, at 180 a trade;** **two half-lives (90 / 504)** is the
    best covariance candidate (the optimiser's portfolio 1.430, random long-only 1.076 to 1.025);
  - the 63-day history and the two half-lives together add up.
- **What I would choose:** the 63-day history and the two half-lives as the defaults; Newey-West's
  long horizon kept, and said; the eigenvalue adjustment next, with the lag count measured in the
  same piece; the correction and its specific side off; no fold.
- **The runs** were stopped once by Claude Code's memory guard, when I launched eleven at once,
  and rerun one at a time by a runner out of its reach; none failed (Failures).
- **Tests:** 2,338 passed, 1 skipped; 26 new, and the shadowed one. No default changed; the
  defaults' run equals midoffice10's in every array but the weighted share it now carries.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice11_bias_statistics.md`, verbatim. In outline:

- **"committed -- midoffice10."** Checked: 4f27f23 in `git log`.
- **Rulings on midoffice10's decisions:**
  1. **The correction:** an option, off. And a reason to check: under the regression's own
     assumption a member's residual variance is its specific variance times (1 - its
     leverage), so the correction as built leaves each member short by its leverage. If the
     algebra holds, build the specific side's half as an option, off: each residual over
     sqrt(1 - h) for a member, over sqrt(1 + h) for a name outside the regression,
     h = w x' (X'WX)^-1 x. The eigenvalue adjustment waits for the bias statistics.
  2. **The diagnostic:** the weighted noise share always computed, reported beside the
     equal-weighted one, and the qualification quoting it.
  3. **The thin industries:** no fold yet; floor 15 by the Pearson rule is the candidate the bias
     statistics judge.
  4. **Size Nonlinearity:** kept.
  5. **FRCB's 2023-05-04 return:** dropped, as built.
  - And remove the two unused imports ruff has flagged since before midoffice10.
- **midoffice11, the out-of-sample bias statistics,** the rulings first:
  1. **Forecasts as of each date,** from data through the date alone: the factor returns, the
     specific-risk model's fit and histories, the exposures. Refit daily if affordable, else
     monthly; say which. The industry scheme held fixed; say so. The first forecast once the
     covariance has a year of factor returns. Guard: the forecast as of t on the panel cut at
     t, the scheme held, equals the full run's, for a handful of dates.
  2. **The statistic:** each forecast at a daily scale against the next day's total return; the
     standard deviation of their ratio, band about 1 +/- sqrt(2/T); over the whole test and
     rolling 12-month windows with the windows' mean absolute deviation from 1; a 21-day
     horizon, non-overlapping, as a check.
  3. **The portfolios,** each held from one forecast to the next: every member singly; the
     index cap- and equally weighted and the size pair; each industry's members cap-weighted,
     all 20, against their weighted noise shares; random portfolios, seeded; the
     minimum-variance portfolio, long-only if a solver is at hand; the factor covariance's
     eigenportfolios against the factor returns.
  4. **The defaults first;** if far from 1 across the board, find out why. Then each candidate,
     one change at a time: floor 15; the correction, and with its specific side; the
     specific-risk history at two or three half-lives; the factor covariance at a shorter and a
     longer half-life, and the two-half-life estimator.
- **"Change no default. Report, and say which you would choose and why."** Retro with its
  `git add` and the state file in the same commit, then stop.
- **After the stop (11-12):** run them one at a time out of the memory guard's reach, each only
  while 3 GB is available, resumable, the defaults first; the guards that passed not rerun if
  no code changed; retry a failure once; don't stop to ask; the stop and the rerun under
  failures; and the state file to say this machine has 25.8 GB usable, above 20 GB in use.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `src/praxis/midoffice/model/crosssectional.py` | ruling 2: every period's sampling covariance kept, the weighted share always computed (`sampling_weights(diagonal=)`, `_QUADRATIC_FORMS`); ruling 1: `leverage_correction`, `_leverages`, the notes' `leverage_*` and three qualifications; `specific_half_life`, `_history_spread`, the notes' `specific_half_life` and `history_effective`; `cross_sectional_models_as_of`; the unused `Mapping` removed; docstrings | 90-93, 104, 167-183, 212-215, 252-269, 415-431, 749-854, 879-880, 961-985, 1048-1068, 1122-1173, 1199-1200, 1234-1249, 1286-1287, 1330-1338, 1374-1377, 1434-1455, 1468-1528 |
| `src/praxis/midoffice/model/covariance.py` | `HorizonCovariance.newey_west_lags` and `volatility_estimator`; the split-horizon Newey-West's method label and PSD note | 237-269, 299-308, 336-349 |
| `src/praxis/midoffice/model/specificrisk.py` | the blend takes a fractional number of observations (an effective one) | 287-289, 305, 317-320 |
| `src/praxis/midoffice/model/bias.py` | new: the statistic, its band, rolling windows and MRAD; `RiskForecast`; the minimum-variance portfolios; eigenportfolios; buy-and-hold; `evaluate` over the standard portfolios | new, 501 lines |
| `scripts/run_sp500_model.py` | `ModelInputs`, `model_inputs`, `parse_merges`, `industry_codes` (the computation out of `main()`); `cut_panel` and the bias section with its guard and report; the factor table's two noise shares; `--leverage-correction`, `--specific-half-life`, `--correlation-half-life`, `--bias`, `--bias-out`, `--bias-guard`; the estimator's options gathered once; the unused import removed, `HorizonCovariance` imported again for its flag | 33, 40-47, 495-1003, 1072-1107, 1196-1216, 1292-1374, 1633-1654, 1708-1726 |
| `tests/midoffice/test_bias.py` | new: 10 functions | new, 200 lines |
| `tests/midoffice/test_crosssectional.py` | 13 new tests, one updated; the imports | 29-32, 1628-1692, 1800-2051 |
| `tests/midoffice/test_covariance.py` | 2 new tests; a duplicate name renamed; an unused import removed | 382-406, 522-524 |
| `docs/FACTOR_MODEL.md` | the leverage correction; the weighted share always; the two-half-life estimator's Newey-West; the history half-life; a new section, "Out-of-sample bias statistics" | 253-259, 494-516, 557-569, 744-751, 1138-1219 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43), with the machine's memory note | throughout |
| `claude/handoffs/BRIEF_midoffice11_bias_statistics.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice11_bias_statistics.md` | new: this retro | new |

Written outside the tracked files: only `claude/scratch/midoffice11/` (gitignored). Nothing in
`data/` was written, overwritten or deleted; the cache gained nothing.

---

## Ruling 1: the correction's specific side

**The algebra holds** (`leverage_algebra.txt`): a world under the regression's own assumption,
residual variance sigma^2 / w with square-root-cap weights, three industries of 8, 20 and 34
and two styles, simulated over 40,000 periods.

- **The noise in the factor returns is the members' specific returns:** f_hat - f equals
  (X'WX)^-1 X'W u to 2.8e-17, its covariance sigma^2 (X'WX)^-1.
- **A member's residual variance is its specific variance times (1 - h):** the thin industry's
  members 0.8543 simulated against 0.8537; every member within sampling error.
- **A name outside the regression carries (1 + h):** four outside names, 1.040 to 1.302,
  each within sampling error of 1 + h.
- **With both halves every portfolio is right,** in expectation exactly: 1.0000 for the thin
  industry cap- and equally weighted, its largest and smallest members, a large member, the
  cross-section.

**One qualification to the brief's arithmetic.** The estimator already inflates every residual
by sqrt(n / (n - k)), and since the leverages sum to the rank that is the leverage correction at
the mean leverage. So under the correction as built a member is short by its leverage less the
mean one, not by all of it: in the simulation the 8-member industry's members by 7.1%, where
their mean leverage is 14.6%. The option replaces the uniform inflation with each name's own;
applying both would correct twice.

**On the real model** (`ruling1.txt`, every estimated regression's leverages from the run's own
design):

- **The mean leverage is 0.0498:** 502 members and rank 25 at the median.
- **By industry:** Construction's members average 0.156 (0.122 to 0.198), so under the
  correction as built they are 11.2% short of their specific variance; Finance & Real Estate
  0.100 and Wholesale 0.098, 5.3% and 5.1% short. Members of the large industries, under the
  mean, are over-corrected by the uniform inflation, by up to 1.9% (SIC38, SIC73).
- **The largest leverages are the largest names,** which dominate their industries' regression
  weight under square-root-cap weighting: TSLA 0.357, AAPL 0.319, NVDA 0.306, AMZN 0.229, then
  PWR, JPM, DHI, AXP, NFLX, MSFT at 0.16 to 0.20. The industry factor absorbs much of a mega
  cap's specific return.

**Built as an option, off** (`leverage_correction`, the run's `--leverage-correction`): each
member's residual over sqrt(1 - h) in place of the uniform inflation; each covered specific
return outside the regression over sqrt(1 + h), its weight put on the members' scale. Leverages
by one SVD of the period's weighted design, with `lstsq`'s cut-off, so a rank-deficient period
reads as the fit reads it. A member at leverage one is left as it is (its residual is zero), an
outside name with no weight is left as measured, and both are counted; neither occurs here.
Factor returns, R-squared, the sampling covariance and the covariance are untouched. Used
without the factor side, the qualifications say the noise is then counted twice.

**Measured** (the reference portfolios on 2026-11-03, `ruling1.txt`):

| Portfolio | Defaults | Correction | Correction + specific side | Specific side alone |
|---|---|---|---|---|
| the index, cap-weighted | 11.82% | 11.69% | 11.79% | 11.92% |
| the index, equally weighted | 9.94% | 10.01% | 10.01% | 9.94% |
| smallest 85, equally weighted | 13.06% | 12.98% | 12.97% | 13.05% |
| largest 50, equally weighted | 12.75% | 12.62% | 12.65% | 12.78% |
| Construction (8) | 26.86% | 24.22% | 24.53% | 27.15% |
| Wholesale (10) | 16.18% | 13.35% | 13.53% | 16.33% |
| Finance & Real Estate (11) | 19.73% | 18.04% | 18.25% | 19.92% |

- **Specific risk:** the median forecast -0.84%; Construction's members a median +4.25%,
  Wholesale's +2.9%, Finance & Real Estate's +1.3%; the large industries' members about -1%.
- **The mega caps:** TSLA's specific risk 32.33% to 38.96%, AAPL's 17.58% to 20.64%, NVDA's
  24.00% to 27.94%, AMZN's 21.57% to 23.92%. The cap-weighted index's specific risk goes from
  3.22% to 3.59%.
- **Mean R-squared** 0.293203 throughout; the defaults with the option off equal the defaults
  before it in every field and every report line (B2 against B1).

## Ruling 2: the weighted noise share, always

- **Always computed.** Each estimated period's full sampling covariance is kept whether or not
  the correction is on, and `weighted_noise_share` is filled wherever the covariance's diagonal
  is a quadratic form in the returns. That now includes the two-half-life estimator, whose every
  variance is its volatility estimator's (`HorizonCovariance.volatility_estimator`); the
  correction still refuses it, its whole matrix being no quadratic form. The shrunk estimator
  has none, and the qualification falls back on the equal-weighted share.
- **Reported beside the equal-weighted one:** the run's factor table has both columns, "equal"
  and "weighted", under a line saying what each is.
- **Quoted by the qualification:** "69% of the estimated variance of Wholesale is the sampling
  error of its own estimation, weighted as the covariance weights the periods; the factor
  covariance diagonal is biased upward and is not corrected here" -- where midoffice10's quoted
  Size Nonlinearity's equal-weighted 47%.
- **Measured** (`r2_compare.txt`): the defaults' run against midoffice10's equals it in every
  field but the two weighted ones, now filled, and its report differs only in the factor table
  and that qualification. The correction's run equals midoffice10's in all 21 fields, the
  weighted share to the bit, its report differing only in the factor table.

## Rulings 3 to 5, and the imports

- **No fold, Size Nonlinearity kept, FRCB's return dropped:** nothing to build; floor 15 is run
  below as a candidate.
- **The two unused imports removed:** `typing.Mapping` from `crosssectional.py` and
  `HorizonCovariance` from `run_sp500_model.py`. The run script imports `HorizonCovariance` again
  later in this piece, used now by `--correlation-half-life`; ruff is clean over every changed
  file.
- **Two more, found in a file this piece edits:** `tests/midoffice/test_covariance.py` imported
  `CovarianceEstimate` unused, and defined `test_a_non_positive_half_life_is_refused` twice, so
  the first -- the exponential estimator's refusal of a non-positive half-life, parametrized --
  never ran. Both were at HEAD (checked). The second is renamed and now runs, and passes.

---

## Item 1: forecasts as of each date

**Refitted daily.** The whole estimator takes about a second on the S&P 500 run (1.1 s at the
end of the sample, 0.5 s in its middle), so every forecast date gets its own refit, by the
estimator itself on its inputs cut at the date
(`crosssectional.cross_sectional_models_as_of`): every return, characteristic, weight,
membership, capitalisation and period after it removed. Each forecast so rests on the factor
returns through its date, the covariance from them, the specific-risk model's fit and every
history it blends, and the date's own exposures. 686 refits take 7.7 minutes alone on the
machine. No monthly fallback was needed.

**The first forecast** is 2024-02-08, the date of the 252nd estimated period; the last
2026-11-02, the last with a next trading day. 686 forecasts, 32 of them every 21st.

**The industry scheme is held fixed over the sample,** as Barra fixes a model's industry
factors: built once over the full sample's regression universe, and every refit reads the same
industry exposures. The run says so in its report.

**The guard** (`--bias-guard`, five dates by default: the first, the last and three between).
On each date the run rebuilds the estimator's inputs from the panel cut at that date through its
own pipeline, the scheme held, and checks them against the full run's through the date, to the
bit -- returns, exposures, weights, capitalisations, periods -- and the model from them against
the forecast as of the date: covariance, exposures, specific risk, factor returns. It stops if
any differ. Both runs whose inputs differ (the defaults, floor 15) passed on all five dates:
2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02 (`guard_D.txt`, `guard_F15.txt`).

- For that, `run_sp500_model.py`'s computation from the panel to the estimator's inputs moved
  into one function, `model_inputs`, which the run and the guard both call. The report it feeds
  is unchanged line for line (B3 against B2).
- The blend rule (one definition per factor) reads the forecast row's blends, the cut panel's
  last; it marks the same periods on every guard date, so nothing in the inputs looks ahead but
  the scheme, held as ruled.
- A test pins the property on a synthetic world with every option on: redraw everything after a
  date and the model as of it is unchanged to the bit.

## Item 2: the statistic

`model.bias`: z = realised / forecast; the statistic is sd(z), ddof 1, its band
1 +/- sqrt(2/T), +/-0.054 at T = 686 and +/-0.25 at the 32 non-overlapping 21-day outcomes. The
rolling windows are every run of 252 consecutive forecasts, stepped daily (435 windows), and
MRAD their mean absolute deviation from 1. The 21-day check buys each portfolio at every 21st
forecast date and holds it 21 days, a name that stops trading held as cash; its forecast is the
daily one times sqrt(21). The kurtosis of z is reported beside the statistic, and the band it
implies, 1 +/- 1.96 sqrt((kurtosis - 1) / 4T), since a single stock can fall outside the normal
band from one earnings day. The factor covariance is used as the risk engine uses it, through
`nearest_psd` at the configured floor; it bound on none of the defaults' 686 dates.

## Item 3: the portfolios, at the defaults

Each is formed at its forecast date from that date's members with a forecast, a positive
capitalisation and a next-day return; 18 member-days had no next-day return -- the name's
prices stop there -- and were left out of that day's portfolios, as the index leaves them.

| Portfolio | T | Bias | | MRAD | Rolling min / max | 21-day |
|---|---|---|---|---|---|---|
| the index, cap-weighted | 686 | 1.096 | under | 0.169 | 0.893 / 1.356 | 0.952 |
| the index, equally weighted | 686 | 1.072 | under | 0.137 | 0.939 / 1.281 | 0.895 |
| smallest 85, equally weighted | 686 | 1.123 | under | 0.173 | 0.935 / 1.298 | 0.884 |
| largest 50, equally weighted | 686 | 1.080 | under | 0.134 | 0.881 / 1.293 | 1.019 |
| minimum variance, long-only | 686 | 1.357 | under | 0.395 | 1.109 / 1.596 | 1.063 |
| minimum variance, unconstrained | 686 | 1.578 | under | 0.637 | 1.246 / 1.897 | 1.269 under |

- **The minimum-variance portfolios.** Long-only, by non-negative least squares, holds a median
  81 names (48 to 218), forecast 7.27% a year at the median against 10.19% realised;
  unconstrained, all ~501, forecast 6.54% against 10.62%. KKT conditions met to 3e-13 on every
  date. The optimiser finds what the covariance underrates: the classic optimisation bias.
- **The industries** (cap-weighted members, the scheme's 20 as built): from 0.952 (Services) to
  1.232 (SIC36), 13 of 20 above the band; MRAD 0.041 (Construction) to 0.223 (SIC36). The thin
  three: Construction 1.066, Finance & Real Estate 1.096, Wholesale 1.055. Against the weighted
  noise share across the 20 the statistic correlates -0.207: the noisiest factors' portfolios
  are no more biased than the rest -- if anything a shade over-forecast, the double count.
- **Random portfolios** (seed 11): 100 long-only of 30 members, median 1.076 (p5 0.997, p95
  1.169), 70% above the band; 100 long-short of 30 a side, median 1.076 (0.958 to 1.250), 60%
  above, 2% below.
- **Every member singly** (543 with 63 forecasts or more): median 1.075 (p5 0.850, p95 1.315);
  16.0% below the band and 54.5% above, against 2.5% each for right forecasts of normal
  returns; with each member's band widened for its own kurtosis (median 8.8), 9.2% and 36.1%.
  By size quintile, largest first: 1.124, 1.083, 1.082, 1.045, 1.048 -- the largest names worst,
  as ruling 1's leverages predict. 21-day, members with 12 outcomes or more: median 1.019.
- **The eigenportfolios** against the next period's estimated factor returns: 1.10 at the
  largest eigenvalue, 0.87 to 1.17 over ranks 2 to 8, then rising steadily to 2.05 to 2.26 over
  ranks 19 to 25 (21-day 1.9 to 2.7). The smallest eigenvalues are forecast at about half their
  realised volatility: the pattern Barra's eigenfactor adjustment exists to correct. These
  realise against estimated factor returns, which carry the noise, so they cannot judge the
  correction.

## Item 4: the defaults first -- why they sit above one

The daily statistics sit above one across the board, 1.07 to 1.12 for the broad portfolios,
while the 21-day check sits at or below it, 0.88 to 1.02. The brief asks why before going on
(`diagnose_D.txt`):

1. **The horizon.** The covariance is Newey-West: a long-run variance per day, counting the
   factor returns' autocovariances. Over the test, Newey-West at 10 lags holds the index's
   factor variance at a median 0.805 of the same weights' one-day variance (0.546 to 1.112
   across the dates); as a volatility 0.897, so a daily statistic of **1.114 from the horizon
   alone**, where the index measures 1.096. Per factor the ratio runs from 0.68 (SIC37,
   Volatility) to 1.08 (Size Nonlinearity, Trading Activity).
2. **The returns agree.** The realised daily index return mean-reverts over the test:
   autocorrelations -0.056, +0.044, -0.109, -0.060 at lags 1 to 4; their Bartlett-weighted sum
   at 10 lags 0.740; the variance of its 21-day sums 0.650 of 21 times its daily variance.
3. **The rest is regime.** By half-year the index's statistic runs 0.892, 1.171, 1.528 (2025H1,
   the April tariff shock), 0.762, 1.109, 0.866; the members' median 0.904, 1.020, 1.187,
   0.929, 1.153, 1.052.

So the model's forecasts are long-horizon forecasts, by Newey-West's design, and a one-day test
reads them about 10% short for factor risk. That is common to every candidate but the
covariance's own, so they went on; the diagnosis was confirmed by running the defaults with no
Newey-West lags (D0, below). Specific risk carries no Newey-West, so for a single member the
daily horizon is the right one, and there the defaults are short by about 7.5% at the median.

## Item 4: each candidate against the defaults

Every candidate is one change against the defaults; all run with the guard off, their inputs
the defaults' (but floor 15's, whose guard passed). Full tables in `compare_bias.txt`.

### Floor 15 by the Pearson rule

Construction and Wholesale into Retail, Finance & Real Estate into SIC62.

| | Defaults | Floor 15 |
|---|---|---|
| Construction's portfolio, daily / 21-day | 1.066 / 0.850 | **1.319** / 1.068 |
| Finance & Real Estate's | 1.096 / 0.946 | 1.018 / 0.879 |
| Wholesale's | 1.055 / 0.953 | 0.979 / 0.877 |
| Retail's (the target) | 1.119 / 1.109 | 1.165 / 1.155 |
| SIC62's (the target) | 1.066 / 1.099 | 1.196 / 1.228 |
| industries, mean abs(bias - 1) | 0.084 | 0.099 |
| the thin three, mean abs(bias - 1) | 0.072 | 0.119 |
| minimum variance, unconstrained | 1.578 | 1.533 |
| broad portfolios, members, random | | unchanged to 0.01 |

**It hurts more than it helps.** Folded into Retail, the homebuilders' shared industry move
becomes specific risk, which the model takes to be independent across names, so the
eight-name portfolio realises about a third more than its forecast; and the two targets, now
carrying members
that do not move with theirs, are worse. Finance & Real Estate's and Wholesale's own portfolios
improve. The fold buys the unconstrained minimum-variance portfolio 0.045. Midoffice10 found
Construction to be two industries; this is what folding it whole costs.

### The correction, and with its specific side

| | Defaults | Correction | + specific side | Specific side alone |
|---|---|---|---|---|
| the index, cap-weighted, daily / 21-day | 1.096 / 0.952 | 1.102 / 0.957 | 1.096 / 0.951 | 1.090 / 0.946 |
| the index, equally weighted | 1.072 / 0.895 | 1.064 / 0.887 | 1.064 / 0.887 | 1.072 / 0.895 |
| minimum variance, long-only | 1.357 / 1.063 | 1.403 / 1.077 | 1.402 / 1.077 | 1.361 / 1.069 |
| minimum variance, unconstrained | 1.578 / 1.269 | **1.925 / 1.553** | 1.916 / 1.551 | 1.582 / 1.275 |
| its MRAD | 0.637 | 1.013 | 1.002 | 0.638 |
| industries, daily mean abs(bias - 1) | 0.084 | 0.125 | 0.111 | 0.073 |
| industries, 21-day mean abs(bias - 1) | 0.077 | 0.073 | 0.068 | 0.077 |
| industries, 21-day mean | 0.970 | 1.017 | 1.004 | 0.958 |
| the thin three, daily | 0.072 | 0.193 | 0.180 | 0.063 |
| members, median | 1.075 | 1.088 | 1.093 | 1.077 |
| members, largest quintile | 1.124 | 1.144 | 1.130 | 1.119 |
| the smallest five eigenportfolios | 2.22 | 1465 | 1465 | |

- **The near-riskless directions realise.** The correction's unconstrained minimum-variance
  portfolio was forecast at 5.89% a year at the median and realised 11.45%; outside the band at
  both horizons. Long-only it holds 67 names where the defaults' holds 81. The clipped
  eigenportfolios, forecast at the floor, realise the noise: a statistic of about 1,465.
- **At the 21-day horizon it centres the industry portfolios** -- 1.017, and 1.004 with the
  specific side, against the defaults' 0.970 -- the one place it helps. At the daily horizon the
  same shift compounds the Newey-West shortfall and every industry reads worse.
- **The specific side** recovers about a third of what the correction costs the industries at
  the daily horizon (0.125 to 0.111, the defaults 0.084) and almost nothing of the optimiser's;
  it raises the members' median, since it lowers most names' specific risk.
- **Alone,** the specific side raises the high-leverage names' specific risk -- the thin
  industries' members and the mega caps -- and the industries read better at the daily horizon
  (0.084 to 0.073) and the same at 21 days; the largest quintile 1.124 to 1.119. Small, and in
  theory the noise counted twice.

### The specific-risk history's half-life

The blend told each history's effective number of observations: at 63, a median 181.8 of 938,
the characteristics' weight 13.47%; at 126, 359.4 and 7.28%; at 252, 624.7 and 4.32%; at equal
weights 938 and 2.92%.

| | Equal | 252 | 126 | 63 |
|---|---|---|---|---|
| members, median bias | 1.075 | 1.057 | 1.045 | **1.042** |
| members, mean abs(bias - 1) | 0.133 | 0.116 | 0.106 | **0.098** |
| members, p90 | 1.256 | 1.219 | 1.197 | 1.173 |
| members above the band / the kurtosis band | 54.5% / 36.1% | 50.3% / 30.0% | 47.7% / 26.2% | 44.2% / 19.9% |
| members below the band | 16.0% | 16.9% | 16.8% | 18.2% |
| members, median MRAD | 0.123 | 0.114 | 0.106 | 0.099 |
| members by size, largest quintile | 1.124 | 1.114 | 1.100 | 1.093 |
| members by size, smallest quintile | 1.048 | 1.038 | 1.030 | 1.014 |
| random long-short, median / above the band | 1.076 / 60% | 1.051 / 49% | 1.035 / 42% | 1.021 / 36% |
| industries, mean abs(bias - 1) | 0.084 | 0.079 | 0.076 | 0.072 |
| the thin three | 0.072 | 0.066 | 0.061 | 0.056 |
| minimum variance, unconstrained | 1.578 | 1.565 | 1.557 | 1.548 |
| the broad portfolios | | within 0.005 | | |

Members' median by half-year, equal weights against 63: 0.904 / 0.914, 1.020 / 1.022, 1.187 /
1.141, 0.929 / 0.913, 1.153 / 1.091, 1.052 / 0.962. **Monotone, the shortest best on every
specific-side measure:** a faster history follows the rise in residual dispersion through 2025
and 2026 that an equal-weighted one averages away. At 63 the latest half is already a little
over-forecast.

### The factor covariance's half-life, and two half-lives

Newey-West at 10 lags throughout; the two-half-life estimator is volatilities on 90 and
correlations on 504 (BIM301S's two years), both Newey-West (`HorizonCovariance`'s new
`newey_west_lags`, so the one change is the correlations' half-life).

| | 45 | Defaults (90) | 180 | Two half-lives 90 / 504 |
|---|---|---|---|---|
| the index, cap-weighted, daily / 21-day | 1.132 / 0.990 | 1.096 / 0.952 | 1.101 / 0.944 | 1.080 / 0.935 |
| the index, equally weighted | 1.136 / 0.944 | 1.072 / 0.895 | 1.041 / 0.861 | 1.008 / 0.832 |
| smallest 85 | 1.177 / 0.931 | 1.123 / 0.884 | 1.085 / 0.848 | 1.032 / 0.797 |
| largest 50 | 1.117 / 1.043 | 1.080 / 1.019 | 1.084 / 1.015 | 1.068 / 1.010 |
| the index's MRAD | 0.168 | 0.169 | 0.213 | 0.190 |
| minimum variance, long-only | 1.524 | 1.357 | 1.306 | **1.275** |
| minimum variance, unconstrained, daily / 21-day | 1.863 / 1.512 | 1.578 / 1.269 | 1.462 / 1.169 | **1.430 / 1.147** |
| its MRAD | 0.959 | 0.637 | 0.490 | 0.451 |
| the smallest five eigenportfolios | 2.96 | 2.22 | 1.95 | 1.86 |
| random long-only, median / above the band | 1.121 / 89% | 1.076 / 70% | 1.051 / 49% | 1.025 / 28% |
| industries, mean abs(bias - 1) | 0.096 | 0.084 | 0.086 | 0.077 |
| members, median | 1.083 | 1.075 | 1.077 | 1.067 |

- **45 is worse nearly everywhere:** 130 effective observations of 25 factors and ten lags is
  more estimation error, which the optimiser finds.
- **180 trades responsiveness for estimation error:** the optimiser's portfolio and the small
  eigenvalues better, the index's MRAD 0.169 to 0.213, and slower in the shock (the equally
  weighted index 1.524 in 2025H1, the defaults' 1.455).
- **The two half-lives keep the volatilities' 90 days and give the correlations four times the
  history:** the best of the three on the optimiser's portfolios, the eigenportfolios, the random
  portfolios and the industries, as quick as the defaults in the shock (1.443 in 2025H1), at a
  cost of the index's MRAD 0.169 to 0.190. Its long correlations raise the diversified
  portfolios' forecasts, which helps at the daily horizon and reads lower at 21 days (the
  equally weighted 0.832, the smallest 85 0.797, inside the band).

### The defaults without Newey-West lags (a diagnosis)

Not a candidate: the defaults at `--newey-west-lags 0`, to confirm the horizon.

| | Defaults | No lags |
|---|---|---|
| the index, cap-weighted, daily / 21-day | 1.096 / 0.952 | 1.024 / 0.856 |
| the index, equally weighted | 1.072 / 0.895 | 0.996 / 0.813 |
| smallest 85 | 1.123 / 0.884 | 0.997 / 0.762 |
| largest 50 | 1.080 / 1.019 | 1.004 / 0.921 |
| minimum variance, long-only | 1.357 / 1.063 | 1.174 / 0.970 |
| minimum variance, unconstrained | 1.578 / 1.269 | 1.235 / 1.070 |
| its MRAD | 0.637 | 0.277 |
| eigenportfolios, ranks 1 / 10 / 15 / 20 / 25, daily | 1.10 / 1.24 / 1.63 / 2.12 / 2.21 | 1.03 / 1.08 / 1.16 / 1.26 / 1.19 |
| the smallest five, 21-day | 2.38 | 1.21 |
| industries, mean abs(bias - 1) | 0.084 | 0.042 |
| random long-only, median | 1.076 | 1.008 |
| members, median | 1.075 | 1.045 |

- **The horizon, confirmed:** with no lags the broad portfolios sit on one at the daily horizon
  and below it at 21 days, the mirror image of the defaults.
- **And more than the horizon: the lag terms are most of the eigenvalue bias.** With them the
  smallest eigenportfolios realise more than twice their forecast at both horizons, so it is no
  long-run effect the lags have found; it is the estimation error of ten lags of
  autocovariance on a 90-day half-life over 25 factors, pushing the small eigenvalues down, as
  `NeweyWestCovariance`'s docstring warns exponential weighting can. The optimiser's portfolio
  follows: 1.58 with them, 1.24 without, and better at 21 days too (1.27 against 1.07).

### The two chosen together (a check of the choice)

Not a candidate: the 63-day history and the two half-lives in one run, to see whether they add.

| | Defaults | 63-day history | Two half-lives | Both |
|---|---|---|---|---|
| members, median / mean abs(bias - 1) | 1.075 / 0.133 | 1.042 / 0.098 | 1.067 / 0.129 | **1.034 / 0.097** |
| members above the kurtosis band | 36.1% | 19.9% | 34.1% | 17.9% |
| random long-only, median / above the band | 1.076 / 70% | 1.067 / 58% | 1.025 / 28% | **1.017 / 17%** |
| random long-short | 1.076 / 60% | 1.021 / 36% | 1.070 / 60% | 1.016 / 37% |
| industries, mean abs(bias - 1) | 0.084 | 0.072 | 0.077 | **0.066** |
| minimum variance, unconstrained, daily / 21-day | 1.578 / 1.269 | 1.548 / 1.233 | 1.430 / 1.147 | 1.418 / 1.115 |
| the index, cap-weighted / equally weighted | 1.096 / 1.072 | 1.094 / 1.071 | 1.080 / 1.008 | 1.078 / 1.008 |

They add: each keeps its own gain, one on the specific side and one on the factor side.

---

## What I would choose, and why

1. **The specific-risk history at a 63-day half-life.** It is the one candidate that helps the
   specific side, at its right horizon -- the specific forecasts carry no Newey-West -- on every
   measure: the members' median 1.075 to 1.042, their mean absolute deviation from one 0.133 to
   0.098, the share past the kurtosis-widened band 36% to 20%, the long-short random portfolios
   1.076 to 1.021, each size quintile, the thin industries. It costs the factor-dominated
   portfolios nothing. The three half-lives are monotone, so 63 is the best measured, not
   necessarily the best; at 63 the latest half is already a little over-forecast (0.962), which
   argues against going much shorter without measuring it.
2. **The two-half-life covariance, volatilities 90 and correlations 504.** The best of the
   covariance candidates: the optimiser's portfolios, the small eigenvalues and the random
   portfolios all improve, the industries a little, and it reacts to the shock as the defaults
   do. Its cost, the index's MRAD 0.169 to 0.190, is the smaller one; 180 pays more for less.
   The two together add, and that is the pair I would make the defaults.
3. **Keep Newey-West's long horizon, and say so.** The daily statistics' 7-12% excess on the
   broad portfolios is the horizon, by the diagnosis and by the run without lags; the model
   quotes annualised risk, and its 21-day check sits at 0.88 to 1.02. A one-day reading of the
   model's risk is about 10% short for factor risk, and the run's report now shows both.
4. **Commission the eigenvalue adjustment next -- and measure the lag count with it.** The
   eigenportfolios call for it: 2.2 at the smallest eigenvalues, the optimiser's portfolio 1.36
   long-only and 1.58 unconstrained, and still 1.86 and 1.43 under the two half-lives. But the
   run without lags takes the smallest eigenportfolios to 1.25 and the optimiser's to 1.24, so
   the lag terms are most of the problem; an adjustment designed against the ten-lag matrix
   would be correcting their noise. Fewer lags, or lags on the volatilities alone, belong in
   the same piece, judged by the same statistics.
5. **The correction and its specific side stay options, off.** The correction makes the
   optimiser's portfolio worse at both horizons -- the near-riskless directions realise -- and
   helps only the industries, only at 21 days. Its specific side moves little either way. Both
   are worth another look once the eigenvalues are adjusted, not before.
6. **No fold.** Floor 15 makes Construction's own portfolio realise a third more than its
   forecast and both targets worse; what it buys the optimiser is 0.045. If the thin industries
   are to change, Construction needs splitting -- homebuilders and contractors -- not folding,
   and that is a scheme question.

---

## Key decisions

1. **Refitted daily, by the whole estimator on its inputs cut at each date,** not a monthly fit:
   a refit is about a second, and the cut makes each forecast the model fitted on data through
   its date by construction. No refactor of the estimator was needed.
2. **The guard runs the run's own pipeline on the cut panel.** `main()`'s computation from the
   panel to the estimator's inputs moved into `model_inputs`, so the run and its guard share one
   definition; the report it feeds is unchanged to the line (B3 against B2, B4 against B3).
3. **The guard is standing** (`--bias-guard`, five dates by default) and stops the run on a
   difference. Candidates whose inputs are the defaults' ran it off, their inputs being the
   defaults', whose guard passed; D's and F15's guards ran before the stop on this code and were
   not rerun (11-12 instruction 5).
4. **The covariance as the engine uses it:** each forecast's factor covariance through
   `nearest_psd` at the configured floor before any portfolio is priced -- it bound on no date of
   the defaults'; under the correction the matrix is already repaired.
5. **Portfolios formed at the forecast date over members with a forecast, a positive
   capitalisation and a next-day return;** a member with none that day (18 member-days) is left
   out, as the index leaves it. Over 21 days a name that stops is held as cash.
6. **The industry portfolios are the scheme's 20 as built,** in every run, so floor 15's folded
   industries are still judged as portfolios of their own members.
7. **Random portfolios:** 100 long-only of 30 members and 100 long-short of 30 a side, each
   member's key and weight drawn once from seed 11, so a portfolio keeps its names but for the
   index's changes. Long-short as well as long-only, since long-only portfolios of the index are
   mostly its factor risk.
8. **The long-only minimum-variance portfolio by non-negative least squares:** no QP library is
   installed, and scipy's NNLS solves the problem exactly after a change of variables -- KKT
   conditions checked every date, met to 3e-13 -- where its general solvers would be slow and
   approximate at 500 names; the
   unconstrained one by the push-through identity, which needs no inverse of a near-singular
   covariance.
9. **The members' band is the brief's, 1 +/- sqrt(2/T),** with the band their own kurtosis
   implies reported beside it, since single names' daily returns are fat-tailed (median kurtosis
   of z 8.8).
10. **The two-half-life candidate is Newey-West in both horizons** (`newey_west_lags`, new), so
    the one change against the defaults is the correlations' half-life; the library's plain
    exponential form would have dropped Newey-West too.
11. **The history half-life's ages count back from the panel's last row** -- the forecast date,
    in an as-of refit -- and its blend weight uses Kish's effective number, as briefed.
12. **Two runs beyond the brief's candidates, each labelled:** the defaults without lags, to
    find out why the defaults sit above one (the brief's instruction), and the two chosen changes
    together, to check the choice.
13. **The leverage correction replaces the uniform inflation rather than adding to it:** the
    inflation is the same correction at the mean leverage.

## Decisions left for Jeff

1. **The specific-risk history:** a 63-day half-life as the default (my choice); 126; equal
   weights, as now; or shorter half-lives measured first.
2. **The factor covariance:** the two-half-life estimator, volatilities 90 and correlations
   504, both Newey-West at 10 lags, as the default (my choice); or the single half-life 90.
3. **The horizon:** keep Newey-West's long horizon as the model's (my choice), its daily
   forecasts about 10% short for factor risk; or move to a shorter one.
4. **The eigenvalue adjustment:** commission it next, with the Newey-West lag count measured
   alongside (my choice); or the lag count alone first.
5. **The correction and its specific side:** left options, off (my choice).
6. **The thin industries:** no fold (my choice); or a split of Construction, a scheme change.

---

## Tests

**The suite:** 2,338 passed, 1 skipped, 19 warnings, with `--ignore=tests/test_market_data.py`
(91.9 s in the final checks). Before this piece 2,311: 26 new tests, and one that never ran now
does. midoffice alone: 887 passed, 1 skipped.

**New, by file:**

`tests/midoffice/test_bias.py`, 10 functions (11 tests, one parametrized twice):

- `test_the_statistic_is_one_for_standard_outcomes_and_scales_with_them`;
- `test_the_statistics_read_realised_over_forecast_and_roll_over_windows`;
- `test_a_forecast_prices_portfolios_as_the_dense_covariance_does` -- and refuses a held name
  with no forecast;
- `test_the_unconstrained_minimum_variance_portfolio_is_the_closed_form` -- a singular
  covariance too;
- `test_the_long_only_minimum_variance_portfolio_matches_a_general_solver` -- against SLSQP;
- `test_the_eigenportfolios_are_ordered_signed_and_rebuild_the_covariance`;
- `test_buy_and_hold_compounds_and_holds_a_missing_return_as_cash`;
- `test_forecasts_that_are_the_truth_score_one_and_a_known_shortfall_its_inverse` -- the whole
  test on a world whose risk is known: about one for the truth, exactly 1/0.8 times it for
  forecasts 20% low, members, eigenportfolios and the 21-day check included;
- `test_a_member_with_no_next_return_is_left_out_of_that_days_portfolios_and_counted`.

`tests/midoffice/test_crosssectional.py`, 13:

- ruling 2: `test_the_weighted_noise_share_is_the_same_whether_or_not_the_correction_is_made`;
  `test_the_two_half_life_estimators_diagonal_has_sampling_weights_and_the_shrunk_one_none`;
  `test_without_the_correction_the_loud_noise_qualification_quotes_the_weighted_share`; and
  `test_the_correction_is_off_by_default_...` updated to the ruling;
- ruling 1: `test_the_leverages_are_the_weighted_hat_matrix_diagonal_and_sum_to_the_rank`;
  `test_a_residual_lacks_its_leverage_of_its_specific_variance_and_the_correction_restores_it`
  -- 3,000 periods under the regression's assumption, the thin industry 6.3% short before and
  right after, outside names 7% over before and right after;
  `test_the_leverage_correction_changes_specific_risk_and_nothing_on_the_factor_side`;
  `test_the_leverage_correction_without_the_factor_side_is_named_as_counting_the_noise_twice`;
- the history half-life: `test_a_history_spread_is_the_sample_one_at_equal_weights_and_by_age_
  under_a_half_life`; `test_a_history_half_life_follows_a_rise_in_specific_risk_and_trusts_the_
  history_less`; `test_no_history_half_life_is_the_default_and_a_non_positive_one_is_refused`;
- the as-of model: `test_a_model_as_of_a_date_is_the_estimator_on_the_panels_cut_there`;
  `test_nothing_after_a_date_moves_the_model_as_of_it` -- everything after the date redrawn,
  every option on, the model as of it identical to the bit;
  `test_a_model_as_of_a_date_forecasts_with_that_dates_exposures_alone`.

`tests/midoffice/test_covariance.py`, 2, and a rename:

- `test_with_newey_west_lags_equal_half_lives_reduce_to_newey_west`;
- `test_with_newey_west_lags_each_horizon_is_its_newey_west_estimate`;
- `test_a_non_positive_half_life_is_refused_by_the_split_horizon`, renamed from a duplicate
  name, so the exponential estimator's parametrized refusal test it shadowed runs again.

**Hygiene:** every changed file parses, is ASCII and keeps its line endings (`final_checks.txt`).
**ruff:** clean over every changed Python file. Six findings remain in three files no piece has
touched since, each at HEAD (`ruff_head_untouched.txt`).

**The final code, re-run** (`final_checks.txt`): the defaults (B4) equal B3 in all 21 fields and
all 162 report lines, and so midoffice10's defaults but for the ruled weighted share; the
correction (C4) equals C2, and the correction with its specific side (CL4) CL2, in every field
and every line.

---

## Failures and debugging trail

- **The stop.** At 23:48 EST on 11-10 I launched all eleven bias runs at once as
  `run_in_background` shells: about 13 GB on a machine with 25.8 GB usable, already above 20 GB
  in use (PyCharm 2.4 GB, WSL 1.5 GB, Chrome). Seeing 3.7 GB available I stopped eight and kept
  three, then added a fourth; Claude Code's memory guard then stopped the launching shells as
  memory ran critically low. On Windows it stopped only the shells and left the Python running:
  four orphaned runs, 1.6 GB available. I stopped those too and reported, and did not restart.
  None had finished: the defaults and floor 15 were past their guards (both passed) at 50 of 686
  forecasts, the correction at 400, its specific side at 150.
- **The rerun** (11-12, Jeff's instructions): `runner.py`, started in its own minimized console
  out of the guard's reach, ran one run at a time, each only while at least 3 GB was available
  (5.9 to 6.9 GB each time), resumable from its results files (a run writes `<name>.tmp.npz`,
  renamed only on a clean exit), the defaults first, a failure retried once. A light background
  check of the files watched it. The guards that passed were not rerun -- no code had changed
  since (the run script's last change 23:48:20, the guards printed at 23:54) -- and their
  reports are kept (`guard_D.txt`, `guard_F15.txt`). All eleven ran first time, 8.7 to 12.5
  minutes each, 03:12 to 05:03 EST; then the diagnosis and the check of the choice, each started
  by restarting the runner, which skipped what existed. No run failed.
- **The first defaults run, before the stop, would have crashed:** its members' section broadcast
  a dates-length mask against a dates-by-securities array. A smoke test (`smoke_bias.py`, the main
  loop truncated to 70 dates) found it and one more, the members' capitalisations cut to all 686
  dates where a truncated run had 70; both fixed before any measured run, and that run stopped.
- **`forecast_row` and `time` were lost in the refactor,** caught by ruff (F821) before any run.
- **A capture was launched with `--leverage-correction` before the flag existed;** it stopped at
  argument parsing and was rerun.
- **A command with nested background subshells was refused by the safety check** as unparseable;
  rerun as plain commands.
- **stdout to a file is block-buffered,** so a run's report showed nothing while its guard ran;
  progress goes to stderr, flushed, and the runner runs Python with `-u`.

## Problems found, not fixed

1. **The lag terms are most of the eigenvalue bias** (D0): a question for the eigenvalue
   adjustment's piece, not built here.
2. **The minimum-variance portfolio is under-forecast at every setting measured,** at least 1.17
   daily: the optimiser finds the covariance's estimation error.
3. **Single members are under-forecast at the median even at the best setting,** 1.034, the
   largest names most (1.089): part leverage (ruling 1), part the regime.
4. **Construction is two industries** under one division; any fold of it whole costs its own
   portfolio.
5. **Six ruff findings in untouched files,** at HEAD.
6. **`tests/test_market_data.py` looks for `../core_repo/src`,** not on this machine.

## Scratch

`claude/scratch/midoffice11/` (gitignored):

- **Harnesses:** `capture.py` (a run in-process, its report and arrays); `compare.py` (two
  captures, field by field and line by line); `smoke_bias.py`; `runner.py` and its log and done
  markers in `bias/`; `launch_bias.sh`, the parallel launcher that caused the stop, kept as the
  record; `final_checks.sh`.
- **Measures:** `leverage_algebra.py`, `leverage_test_margins.py`, `ruling1.py`,
  `time_refit.py`, `diagnose_D.py`, `compare_bias.py`.
- **Captures:** `base/` (B0, C0, and B0's inputs), `r2/` (B1, C1), `r1/` (B2, C2, CL2, L2), `r3/`
  (B3), `final/` (B4, C4, CL4), each a report and a pickle.
- **Bias runs, `bias/`:** each run's report, progress and `.npz` -- D, F15, C, CL, L, H63, H126,
  H252, V45, V180, T504, D0, H63T504; `guard_D.txt`, `guard_F15.txt`; `stopped/`, the stopped
  runs' partial reports.
- **Outputs:** `leverage_algebra.txt`, `ruling1.txt`, `r2_compare.txt`, `diagnose_D.txt`,
  `compare_bias.txt`, `final_checks.txt`, `ruff_head_untouched.txt`.

The key ones are copied into the appendix. Nothing in `data/` was written, overwritten or
deleted.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

## Open items for Chat

- **The six decisions above.** The two default changes (the 63-day history, the two half-lives)
  are measured together; the eigenvalue adjustment is the next piece I would brief, with the
  lag count in it.
- **Shorter history half-lives** than 63 were not run; the trend says try, the latest half says
  carefully.
- **The bias statistics are now standing:** any later candidate is one `--bias` run against the
  defaults, one at a time on this machine.

---

## Appendix: outputs, verbatim

The key outputs the body rests on, from `claude/scratch/midoffice11/`, in full unless said
otherwise.

### leverage_algebra.txt -- ruling 1's algebra, simulated

```text
62 members, 5 factors; leverages sum to 5.000000 (the rank, 5)
  Thin (8): leverage mean 0.1463, min 0.0367, max 0.2391
  Mid (20): leverage mean 0.0821, min 0.0269, max 0.1932
  Large (34): leverage mean 0.0644, min 0.0060, max 0.2700
  the model's uniform inflation n / (n - k) = 1.0877, its mean leverage 0.0806

The noise in the factor returns against sigma^2 (X'WX)^-1, by its diagonal:
  simulated / exact: 1.004, 1.002, 0.997, 1.013, 1.016
  and it is the members' specific returns: max |(f_hat - f) - (X'WX)^-1 X'W u| = 2.8e-17

A member's residual variance over its specific variance, against 1 - h:
  every member: max |simulated - (1 - h)| = 0.0147 (sd of the simulated ratio ~0.0071)
  Thin: simulated 0.8543, exact 1 - h 0.8537
  Mid: simulated 0.9179, exact 1 - h 0.9179
  Large: simulated 0.9349, exact 1 - h 0.9356
An outside name's specific-return variance over its specific variance, against 1 + h:
  outside 0 (Thin): simulated 1.1250, exact 1 + h 1.1155
  outside 1 (Thin): simulated 1.1949, exact 1 + h 1.1896
  outside 2 (Mid): simulated 1.3023, exact 1 + h 1.2924
  outside 3 (Large): simulated 1.0395, exact 1 + h 1.0444

Each forecast over the truth -- variance, exact expectations (simulated in brackets):
  portfolio                                          no correction   the correction as built  the correction and the s   the specific half alone
  the Thin industry, cap-weighted                           1.2294                    0.9683                    1.0000                    1.2610   (realised 1.0032)
  the Thin industry, equally weighted                       1.2358                    0.9885                    1.0000                    1.2473   (realised 1.0031)
  the largest Thin member alone                             1.0410                    0.8941                    1.0000                    1.1470   (realised 1.0062)
  the smallest Thin member alone                            1.0765                    1.0432                    1.0000                    1.0333   (realised 0.9980)
  a Large member alone                                      1.0840                    1.0783                    1.0000                    1.0058   (realised 1.0075)
  the whole cross-section, cap-weighted                     1.1196                    0.9936                    1.0000                    1.1260   (realised 1.0079)

Where the specific half's member shortfall goes under the correction as built: a member's
specific variance forecast is (1 - h) n/(n - k) of the truth.
  Thin: (1 - h) n/(n-k) mean 0.9286, short by +0.0714
  Mid: (1 - h) n/(n-k) mean 0.9985, short by +0.0015
  Large: (1 - h) n/(n-k) mean 1.0177, short by -0.0177
```

### ruling1.txt -- ruling 1 on the real model: the leverages, the reference portfolios, the mega caps

```text
1. Leverages over the 938 estimated regressions: 502 members and rank 25 at the median, so the uniform inflation n/(n - k) carries a mean leverage of 0.0498
   Per industry, over its members' mean leverage across the periods that held them:
   industry                 members   mean h     min     max  (1-h)n/(n-k)  short by
   Construction                   8   0.1559  0.1220  0.1975        0.8884   +0.1116
   Finance & Real Estate         12   0.0997  0.0651  0.1900        0.9475   +0.0525
   Wholesale                     12   0.0984  0.0620  0.1536        0.9489   +0.0511
   SIC37                         15   0.0868  0.0385  0.3567        0.9611   +0.0389
   Mining                        16   0.0756  0.0495  0.1018        0.9728   +0.0272
   SIC62                         20   0.0661  0.0368  0.1150        0.9829   +0.0171
   SIC60                         20   0.0613  0.0332  0.1967        0.9879   +0.0121
   Services                      20   0.0594  0.0354  0.1800        0.9899   +0.0101
   SIC20                         21   0.0544  0.0333  0.1522        0.9952   +0.0048
   SIC36                         36   0.0543  0.0225  0.3061        0.9953   +0.0047
   SIC63                         25   0.0482  0.0262  0.1580        1.0017   -0.0017
   SIC35                         34   0.0464  0.0203  0.3193        1.0036   -0.0036
   Transport & Utilities         32   0.0453  0.0265  0.0906        1.0048   -0.0048
   Retail                        34   0.0427  0.0219  0.2289        1.0075   -0.0075
   SIC67                         29   0.0405  0.0273  0.0755        1.0099   -0.0099
   Manufacturing                 39   0.0385  0.0204  0.1234        1.0119   -0.0119
   SIC28                         40   0.0375  0.0188  0.1453        1.0130   -0.0130
   SIC49                         37   0.0323  0.0215  0.0654        1.0184   -0.0184
   SIC73                         70   0.0322  0.0137  0.1756        1.0186   -0.0186
   SIC38                         44   0.0318  0.0195  0.0639        1.0190   -0.0190
   Other Industries               3   0.0192  0.0102  0.0297        1.0323   -0.0323

2. The reference portfolios on 2026-11-03, over the 499 index members with a forecast; total risk (factor / specific), annual
   portfolio                                           defaults                  correction  correction + specific side         specific side alone
   the index, cap-weighted                11.82% (11.37%/3.22%)       11.69% (11.23%/3.22%)       11.79% (11.23%/3.59%)       11.92% (11.37%/3.59%)
   the index, equally weighted              9.94% (9.86%/1.26%)        10.01% (9.93%/1.26%)        10.01% (9.93%/1.26%)         9.94% (9.86%/1.26%)
   smallest 85, equally weighted          13.06% (12.65%/3.27%)       12.98% (12.56%/3.27%)       12.97% (12.56%/3.23%)       13.05% (12.65%/3.23%)
   largest 50, equally weighted           12.75% (12.22%/3.64%)       12.62% (12.08%/3.64%)       12.65% (12.08%/3.75%)       12.78% (12.22%/3.75%)
   Construction (8)                      26.86% (24.14%/11.78%)      24.22% (21.16%/11.78%)      24.53% (21.16%/12.41%)      27.15% (24.14%/12.41%)
   Wholesale (10)                         16.18% (14.10%/7.93%)       13.35% (10.74%/7.93%)       13.53% (10.74%/8.22%)       16.33% (14.10%/8.22%)
   Finance & Real Estate (11)             19.73% (17.90%/8.30%)       18.04% (16.02%/8.30%)       18.25% (16.02%/8.73%)       19.92% (17.90%/8.73%)

   Each run's total against the defaults':
   portfolio                                         correction  correction + specific side         specific side alone
   the index, cap-weighted                                -1.1%                       -0.2%                       +0.9%
   the index, equally weighted                            +0.8%                       +0.7%                       -0.0%
   smallest 85, equally weighted                          -0.6%                       -0.7%                       -0.1%
   largest 50, equally weighted                           -1.0%                       -0.8%                       +0.3%
   Construction (8)                                       -9.8%                       -8.7%                       +1.0%
   Wholesale (10)                                        -17.5%                      -16.4%                       +0.9%
   Finance & Real Estate (11)                             -8.6%                       -7.5%                       +1.0%

   Members' specific risk, median change against the defaults, by industry (thin first):
   industry                 members                  correction  correction + specific side         specific side alone
   Construction                   8                      +0.00%                      +4.25%                      +4.25%
   Finance & Real Estate         11                      +0.00%                      +1.31%                      +1.31%
   Wholesale                     10                      +0.00%                      +2.92%                      +2.92%
   SIC37                         14                      +0.00%                      +1.24%                      +1.24%
   Mining                        13                      +0.00%                      +1.09%                      +1.09%
   SIC60                         17                      +0.00%                      +0.27%                      +0.27%
   SIC62                         19                      +0.00%                      -0.42%                      -0.42%
   Services                      19                      +0.00%                      -0.19%                      -0.19%
   SIC20                         16                      +0.00%                      -0.09%                      -0.09%
   SIC63                         23                      +0.00%                      -0.55%                      -0.55%
   SIC67                         28                      +0.00%                      -0.82%                      -0.82%
   Transport & Utilities         25                      +0.00%                      -0.51%                      -0.51%
   Retail                        29                      +0.00%                      -1.03%                      -1.03%
   SIC35                         33                      +0.00%                      -0.97%                      -0.97%
   SIC36                         32                      +0.00%                      -0.88%                      -0.88%
   SIC49                         37                      +0.00%                      -1.13%                      -1.13%
   Manufacturing                 34                      +0.00%                      -0.96%                      -0.96%
   SIC28                         33                      +0.00%                      -0.94%                      -0.94%
   SIC38                         40                      +0.00%                      -1.21%                      -1.21%
   SIC73                         58                      +0.00%                      -1.43%                      -1.43%
   every forecast               550                      +0.00%                      -0.84%                      -0.84%
   defaults: specific risk p10 15.58%, median 24.76%, p90 40.49%; mean R2 0.293203
   correction: specific risk p10 15.58%, median 24.76%, p90 40.49%; mean R2 0.293203
   correction + specific side: specific risk p10 15.57%, median 24.74%, p90 40.14%; mean R2 0.293203
   specific side alone: specific risk p10 15.57%, median 24.74%, p90 40.14%; mean R2 0.293203

3. The twelve members with the highest mean leverage, and their specific risk on the forecast date
   security industry                mean h  cap rank                     defaults                  correction  correction + specific side         specific side alone
   TSLA     SIC37                   0.3567         7                       32.33%                      32.33%                      38.96%                      38.96%
   AAPL     SIC35                   0.3193         2                       17.58%                      17.58%                      20.64%                      20.64%
   NVDA     SIC36                   0.3061         1                       24.00%                      24.00%                      27.94%                      27.94%
   AMZN     Retail                  0.2289         4                       21.57%                      21.57%                      23.92%                      23.92%
   PWR      Construction            0.1975       118                       27.04%                      27.04%                      29.46%                      29.46%
   JPM      SIC60                   0.1967        13                       13.47%                      13.47%                      14.67%                      14.67%
   DHI      Construction            0.1940       265                       20.18%                      20.18%                      21.69%                      21.69%
   AXP      Finance & Real Estate   0.1900        51                       18.43%                      18.43%                      19.96%                      19.96%
   NFLX     Services                0.1800        43                       26.62%                      26.62%                      28.54%                      28.54%
   MSFT     SIC73                   0.1756         3                       18.14%                      18.14%                      19.41%                      19.41%
   FIX      Construction            0.1708       251                       35.33%                      35.33%                      34.07%                      34.07%
   LEN      Construction            0.1610       412                       20.57%                      20.57%                      21.57%                      21.57%
   The ten largest members on the forecast date:
   NVDA     SIC36                   0.3061         1                       24.00%                      24.00%                      27.94%                      27.94%
   AAPL     SIC35                   0.3193         2                       17.58%                      17.58%                      20.64%                      20.64%
   MSFT     SIC73                   0.1756         3                       18.14%                      18.14%                      19.41%                      19.41%
   AMZN     Retail                  0.2289         4                       21.57%                      21.57%                      23.92%                      23.92%
   GOOGL    SIC73                   0.0997         5                       22.20%                      22.20%                      22.89%                      22.89%
   AVGO     SIC36                   0.1588         6                       28.31%                      28.31%                      30.17%                      30.17%
   TSLA     SIC37                   0.3567         7                       32.33%                      32.33%                      38.96%                      38.96%
   GOOG     SIC73                   0.0987         8                       21.98%                      21.98%                      22.64%                      22.64%
   META     SIC73                   0.1171         9                       29.13%                      29.13%                      30.15%                      30.15%
   LLY      SIC28                   0.1453        10                       28.63%                      28.63%                      30.08%                      30.08%
```

### r2_compare.txt -- ruling 2: the defaults and the correction against midoffice10's runs

```text
claude/scratch/midoffice11/base/B0 against claude/scratch/midoffice11/r2/B1: 21 fields compared; equal: 19; differing: weighted_sampling_variance, weighted_noise_share
  weighted_sampling_variance: largest difference nan
  weighted_noise_share: largest difference nan
  qualifications: 7 and 7; differing: 1
  report: 161 and 162 lines; 57 differing line(s)
    -                                                                 periods with a member present
    -  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
    -  Construction               19.42%     +24.76%       38.2%      938          0      0
    -  Finance & Real Estate      15.34%     +18.11%       31.0%      938          0      0
    -  Manufacturing              10.20%     +20.88%       23.2%      938          0      0
    -  Mining                     18.60%     +14.94%       18.5%      938          0      0
    -  Retail                     13.11%     +14.63%       10.7%      938          0      0
    -  SIC20                      11.96%     +11.03%       33.6%      938          0      0
    -  SIC28                      13.14%     +16.64%       10.2%      938          0      0
    -  SIC35                      18.81%     +29.69%        5.2%      938          0      0
    -  SIC36                      21.13%     +32.55%        4.5%      938          0      0
    -  SIC37                      14.22%     +15.42%       19.1%      938          0      0
    -  SIC38                      14.70%     +15.82%       10.6%      938          0      0
    -  SIC49                      15.80%     +24.01%       11.3%      938          0      0
    -  SIC60                      21.85%     +21.97%        7.4%      938          0      0
    -  SIC62                      18.73%     +18.62%       11.5%      938          0      0
    -  SIC63                      14.57%     +19.17%       14.5%      938          0      0
    -  SIC67                      15.01%     +18.00%       16.8%      938          0      0
    -  SIC73                      18.18%     +15.21%        2.9%      938          0      0
    -  Services                   12.72%     +16.11%       28.0%      938          0      0
    -  Transport & Utilities      13.43%     +15.46%       16.5%      938          0      0
    -  Wholesale                  13.98%     +20.69%       46.2%      938          0      0
    -  Volatility                  9.37%      +3.09%        3.5%      938          0      0
    -  Momentum                    7.22%      +4.23%        3.6%      938          0      0
    -  Size                        3.72%      +3.13%        7.7%      938          0      0
    -  Size Nonlinearity           2.32%      -0.53%       46.6%      938          0      0
    -  Trading Activity            6.25%      +2.48%        7.5%      938          0      0
    +  noise share, equal: the mean sampling variance over the variance as estimated; weighted: each period's weighted as the covariance weights its return
    +                                                          noise share   periods with a member present
    +  factor                  annual vol  annual mean    equal   weighted      >=1  exactly 1   none
    +  Construction               19.42%     +24.76%    38.2%      48.6%      938          0      0
    +  Finance & Real Estate      15.34%     +18.11%    31.0%      46.5%      938          0      0
    +  Manufacturing              10.20%     +20.88%    23.2%      33.5%      938          0      0
    +  Mining                     18.60%     +14.94%    18.5%      27.2%      938          0      0
    +  Retail                     13.11%     +14.63%    10.7%      16.1%      938          0      0
    +  SIC20                      11.96%     +11.03%    33.6%      55.5%      938          0      0
    +  SIC28                      13.14%     +16.64%    10.2%      15.7%      938          0      0
    +  SIC35                      18.81%     +29.69%     5.2%       7.0%      938          0      0
    +  SIC36                      21.13%     +32.55%     4.5%       5.3%      938          0      0
    +  SIC37                      14.22%     +15.42%    19.1%      27.3%      938          0      0
    +  SIC38                      14.70%     +15.82%    10.6%      15.9%      938          0      0
    +  SIC49                      15.80%     +24.01%    11.3%      16.9%      938          0      0
    +  SIC60                      21.85%     +21.97%     7.4%      10.5%      938          0      0
    +  SIC62                      18.73%     +18.62%    11.5%      15.0%      938          0      0
    +  SIC63                      14.57%     +19.17%    14.5%      22.7%      938          0      0
    +  SIC67                      15.01%     +18.00%    16.8%      25.4%      938          0      0
    +  SIC73                      18.18%     +15.21%     2.9%       4.5%      938          0      0
    +  Services                   12.72%     +16.11%    28.0%      42.6%      938          0      0
    +  Transport & Utilities      13.43%     +15.46%    16.5%      25.1%      938          0      0
    +  Wholesale                  13.98%     +20.69%    46.2%      68.8%      938          0      0
    +  Volatility                  9.37%      +3.09%     3.5%       5.7%      938          0      0
    +  Momentum                    7.22%      +4.23%     3.6%       5.1%      938          0      0
    +  Size                        3.72%      +3.13%     7.7%      10.7%      938          0      0
    +  Size Nonlinearity           2.32%      -0.53%    46.6%      67.4%      938          0      0
    +  Trading Activity            6.25%      +2.48%     7.5%      10.5%      938          0      0
    -  - 47% of the estimated variance of Size Nonlinearity is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
    +  - 69% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is biased upward and is not corrected here

claude/scratch/midoffice11/base/C0 against claude/scratch/midoffice11/r2/C1: 21 fields compared; equal: 21; differing: none
  qualifications: 8 and 8; differing: 0
  report: 190 and 191 lines; 55 differing line(s)
    -                                                                 periods with a member present
    -  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
    -  Construction               15.05%     +24.76%       38.2%      938          0      0
    -  Finance & Real Estate      12.91%     +18.11%       31.0%      938          0      0
    -  Manufacturing               8.82%     +20.88%       23.2%      938          0      0
    -  Mining                     15.98%     +14.94%       18.5%      938          0      0
    -  Retail                     12.28%     +14.63%       10.7%      938          0      0
    -  SIC20                       9.35%     +11.03%       33.6%      938          0      0
    -  SIC28                      12.14%     +16.64%       10.2%      938          0      0
    -  SIC35                      18.25%     +29.69%        5.2%      938          0      0
    -  SIC36                      20.77%     +32.55%        4.5%      938          0      0
    -  SIC37                      12.18%     +15.42%       19.1%      938          0      0
    -  SIC38                      13.70%     +15.82%       10.6%      938          0      0
    -  SIC49                      14.51%     +24.01%       11.3%      938          0      0
    -  SIC60                      20.77%     +21.97%        7.4%      938          0      0
    -  SIC62                      17.49%     +18.62%       11.5%      938          0      0
    -  SIC63                      12.95%     +19.17%       14.5%      938          0      0
    -  SIC67                      13.29%     +18.00%       16.8%      938          0      0
    -  SIC73                      17.85%     +15.21%        2.9%      938          0      0
    -  Services                   10.34%     +16.11%       28.0%      938          0      0
    -  Transport & Utilities      11.80%     +15.46%       16.5%      938          0      0
    -  Wholesale                   9.51%     +20.69%       46.2%      938          0      0
    -  Volatility                  9.28%      +3.09%        3.5%      938          0      0
    -  Momentum                    7.11%      +4.23%        3.6%      938          0      0
    -  Size                        3.70%      +3.13%        7.7%      938          0      0
    -  Size Nonlinearity           1.71%      -0.53%       46.6%      938          0      0
    -  Trading Activity            6.07%      +2.48%        7.5%      938          0      0
    +  noise share, equal: the mean sampling variance over the variance as estimated; weighted: each period's weighted as the covariance weights its return
    +                                                          noise share   periods with a member present
    +  factor                  annual vol  annual mean    equal   weighted      >=1  exactly 1   none
    +  Construction               15.05%     +24.76%    38.2%      48.6%      938          0      0
    +  Finance & Real Estate      12.91%     +18.11%    31.0%      46.5%      938          0      0
    +  Manufacturing               8.82%     +20.88%    23.2%      33.5%      938          0      0
    +  Mining                     15.98%     +14.94%    18.5%      27.2%      938          0      0
    +  Retail                     12.28%     +14.63%    10.7%      16.1%      938          0      0
    +  SIC20                       9.35%     +11.03%    33.6%      55.5%      938          0      0
    +  SIC28                      12.14%     +16.64%    10.2%      15.7%      938          0      0
    +  SIC35                      18.25%     +29.69%     5.2%       7.0%      938          0      0
    +  SIC36                      20.77%     +32.55%     4.5%       5.3%      938          0      0
    +  SIC37                      12.18%     +15.42%    19.1%      27.3%      938          0      0
    +  SIC38                      13.70%     +15.82%    10.6%      15.9%      938          0      0
    +  SIC49                      14.51%     +24.01%    11.3%      16.9%      938          0      0
    +  SIC60                      20.77%     +21.97%     7.4%      10.5%      938          0      0
    +  SIC62                      17.49%     +18.62%    11.5%      15.0%      938          0      0
    +  SIC63                      12.95%     +19.17%    14.5%      22.7%      938          0      0
    +  SIC67                      13.29%     +18.00%    16.8%      25.4%      938          0      0
    +  SIC73                      17.85%     +15.21%     2.9%       4.5%      938          0      0
    +  Services                   10.34%     +16.11%    28.0%      42.6%      938          0      0
    +  Transport & Utilities      11.80%     +15.46%    16.5%      25.1%      938          0      0
    +  Wholesale                   9.51%     +20.69%    46.2%      68.8%      938          0      0
    +  Volatility                  9.28%      +3.09%     3.5%       5.7%      938          0      0
    +  Momentum                    7.11%      +4.23%     3.6%       5.1%      938          0      0
    +  Size                        3.70%      +3.13%     7.7%      10.7%      938          0      0
    +  Size Nonlinearity           1.71%      -0.53%    46.6%      67.4%      938          0      0
    +  Trading Activity            6.07%      +2.48%     7.5%      10.5%      938          0      0
```

### guard_D.txt and guard_F15.txt -- the guards, before the stop (each run's bias section to its guard line)

```text
Out-of-sample bias statistics
  686 forecasts, as of each trading day from 2024-02-08 -- the first whose covariance has a year (252) of estimated periods -- to 2026-11-02: each the whole estimator refitted on data through its date alone, daily -- the factor returns, the covariance, the specific risk model's fit and every history it blends, the exposures as of the date
  the industry scheme is held fixed over the sample, as Barra fixes a model's industry factors: built once over the full sample's regression universe, 20 industries
  each forecast taken to a daily scale (annual / 252) against the next trading day's total return; the bias statistic is the standard deviation of realised / forecast, one if the forecasts are right, its 95% band about 1 +/- sqrt(2/T); over the whole test, over every 12-month (252-forecast) window with the windows' mean absolute deviation from one (MRAD), and every 21st forecast over the 21 days after it, bought and held
  guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model's inputs rebuilt from the panel cut at the date, the scheme held, equal the full run's through it to the bit -- returns, exposures, weights, capitalisations, periods -- and so do the forecasts made from them: covariance, exposures, specific risk, factor returns
[...]
Out-of-sample bias statistics
  686 forecasts, as of each trading day from 2024-02-08 -- the first whose covariance has a year (252) of estimated periods -- to 2026-11-02: each the whole estimator refitted on data through its date alone, daily -- the factor returns, the covariance, the specific risk model's fit and every history it blends, the exposures as of the date
  the industry scheme is held fixed over the sample, as Barra fixes a model's industry factors: built once over the full sample's regression universe, 20 industries, folded as given (Construction into Retail, Finance & Real Estate into SIC62, Wholesale into Retail)
  each forecast taken to a daily scale (annual / 252) against the next trading day's total return; the bias statistic is the standard deviation of realised / forecast, one if the forecasts are right, its 95% band about 1 +/- sqrt(2/T); over the whole test, over every 12-month (252-forecast) window with the windows' mean absolute deviation from one (MRAD), and every 21st forecast over the 21 days after it, bought and held
  guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model's inputs rebuilt from the panel cut at the date, the scheme held, equal the full run's through it to the bit -- returns, exposures, weights, capitalisations, periods -- and so do the forecasts made from them: covariance, exposures, specific risk, factor returns
```

### bias/report_D.txt -- the defaults' bias statistics, the run's own report

```text
Out-of-sample bias statistics
  686 forecasts, as of each trading day from 2024-02-08 -- the first whose covariance has a year (252) of estimated periods -- to 2026-11-02: each the whole estimator refitted on data through its date alone, daily -- the factor returns, the covariance, the specific risk model's fit and every history it blends, the exposures as of the date
  the industry scheme is held fixed over the sample, as Barra fixes a model's industry factors: built once over the full sample's regression universe, 20 industries
  each forecast taken to a daily scale (annual / 252) against the next trading day's total return; the bias statistic is the standard deviation of realised / forecast, one if the forecasts are right, its 95% band about 1 +/- sqrt(2/T); over the whole test, over every 12-month (252-forecast) window with the windows' mean absolute deviation from one (MRAD), and every 21st forecast over the 21 days after it, bought and held
  686 refits in 7.7 min; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every portfolio formed that day

  Portfolios, each formed at its forecast date and held to the next
                                         T    bias           95% band    MRAD    min    max   21-day: T   bias
  the index, cap-weighted              686   1.096 under  0.946-1.054   0.169  0.893  1.356     32  0.952    in
  the index, equally weighted          686   1.072 under  0.946-1.054   0.137  0.939  1.281     32  0.895    in
  smallest 85, equally weighted        686   1.123 under  0.946-1.054   0.173  0.935  1.298     32  0.884    in
  largest 50, equally weighted         686   1.080 under  0.946-1.054   0.134  0.881  1.293     32  1.019    in
  minimum variance, long-only          686   1.357 under  0.946-1.054   0.395  1.109  1.596     32  1.063    in
  minimum variance, unconstrained      686   1.578 under  0.946-1.054   0.637  1.246  1.897     32  1.269 under
    minimum variance, long-only: 81 names held at the median (48 to 218); forecast 7.27% a year at the median, realised 10.19%; its KKT conditions met to 3.0e-13
    minimum variance, unconstrained: 501 names held at the median (499 to 503); forecast 6.54% a year at the median, realised 10.62%; its KKT conditions met to 3.7e-13

  Each industry's members, cap-weighted -- the scheme's 20 as built -- against its factor's noise share in the full run (equal / weighted)
  industry                 members     T    bias          MRAD   21-day      noise share
  Construction                   8   686   1.066 under   0.041    0.850    38.2% / 48.6%
  Finance & Real Estate         12   686   1.096 under   0.138    0.946    31.0% / 46.5%
  Manufacturing                 39   686   1.036    in   0.116    0.982    23.2% / 33.5%
  Mining                        16   686   0.959    in   0.054    0.888    18.5% / 27.2%
  Retail                        34   686   1.119 under   0.150    1.109    10.7% / 16.1%
  SIC20                         21   686   1.125 under   0.162    1.079    33.6% / 55.5%
  SIC28                         40   686   1.135 under   0.158    0.926    10.2% / 15.7%
  SIC35                         34   686   1.054    in   0.148    1.018      5.2% / 7.0%
  SIC36                         36   686   1.232 under   0.223    1.038      4.5% / 5.3%
  SIC37                         15   686   1.124 under   0.164    0.841    19.1% / 27.3%
  SIC38                         44   686   1.129 under   0.181    0.918    10.6% / 15.9%
  SIC49                         37   686   1.085 under   0.115    0.852    11.3% / 16.9%
  SIC60                         20   686   0.974    in   0.107    1.022     7.4% / 10.5%
  SIC62                         20   686   1.066 under   0.116    1.099    11.5% / 15.0%
  SIC63                         25   686   1.086 under   0.190    0.957    14.5% / 22.7%
  SIC67                         29   686   1.048    in   0.080    0.935    16.8% / 25.4%
  SIC73                         70   686   1.094 under   0.153    1.030      2.9% / 4.5%
  Services                      20   686   0.952    in   0.054    1.070    28.0% / 42.6%
  Transport & Utilities         32   686   0.993    in   0.098    0.883    16.5% / 25.1%
  Wholesale                     12   686   1.055 under   0.101    0.953    46.2% / 68.8%
  across the 20 with a factor of their own, the bias statistic's correlation with the weighted noise share: -0.207

  Random portfolios, long-only, seed 11: 100 of 30 members, weights drawn once per member -- bias median 1.076, p5 0.997, p95 1.169; below the band 0%, above it 70%; MRAD median 0.142

  Random portfolios, long-short, seed 11: 100 of 30 members a side, weights drawn once per member -- bias median 1.076, p5 0.958, p95 1.250; below the band 2%, above it 60%; MRAD median 0.083

  Every member singly: 543 securities with 63 forecasts or more -- bias p5 0.850, p25 0.985, median 1.075, p75 1.175, p95 1.315
    below its band (over-forecast) 16.0%, above it (under-forecast) 54.5%, against 2.5% each for right forecasts of normal returns; with each member's band widened for its own kurtosis (median 8.8), 9.2% and 36.1%
    median by capitalisation quintile over the test, largest first: 1.124, 1.083, 1.082, 1.045, 1.048
    21-day, non-overlapping, members with 12 outcomes or more: median 1.019

  The factor covariance's eigenportfolios against the next period's estimated factor returns, by rank, 1 the largest eigenvalue
  rank  vol, median     T    bias          MRAD   21-day
     1       53.09%   686   1.104 under   0.157    0.940
     2       21.51%   686   0.874  over   0.123    0.782
     3       19.08%   686   0.963    in   0.080    0.800
     4       16.33%   686   0.923  over   0.080    0.873
     5       14.31%   686   1.104 under   0.138    0.944
     6       12.54%   686   1.138 under   0.117    0.864
     7       11.34%   686   1.058 under   0.085    1.258
     8       10.44%   686   1.173 under   0.191    0.867
     9        9.47%   686   1.297 under   0.287    1.192
    10        8.02%   686   1.238 under   0.315    1.201
    11        7.23%   686   1.356 under   0.316    1.304
    12        6.71%   686   1.438 under   0.411    1.677
    13        6.14%   686   1.482 under   0.512    1.526
    14        5.77%   686   1.469 under   0.450    1.634
    15        5.37%   686   1.628 under   0.628    1.599
    16        4.88%   686   1.622 under   0.622    1.550
    17        4.48%   686   1.704 under   0.734    2.049
    18        4.24%   686   1.854 under   0.880    2.038
    19        3.96%   686   2.049 under   1.086    2.103
    20        3.60%   686   2.124 under   1.131    2.358
    21        3.10%   686   2.179 under   1.173    2.701
    22        2.59%   686   2.184 under   1.211    1.915
    23        2.15%   686   2.262 under   1.259    2.462
    24        1.62%   686   2.257 under   1.306    2.731
    25        1.05%   686   2.214 under   1.157    2.086
  every forecast and outcome written to C:\Data\Development\Python\McTheoryApps\praxis\claude\scratch\midoffice11\bias\D.tmp.npz
```

### diagnose_D.txt -- why the defaults' daily statistics sit above one

```text
1. Newey-West at 10 lags over the same weights with none, every 20th forecast date
   the cap-weighted index's factor variance: median 0.805 (range 0.546 to 1.112); as a volatility 0.897, so a daily bias statistic of 1.114 from the horizon alone
   per factor, the median variance ratio:
     SIC37                    0.682
     Volatility               0.699
     SIC63                    0.723
     Manufacturing            0.734
     SIC49                    0.739
     SIC36                    0.741
     SIC28                    0.742
     Finance & Real Estate    0.747
     SIC38                    0.749
     SIC67                    0.755
     SIC20                    0.768
     Transport & Utilities    0.771
     Momentum                 0.785
     Mining                   0.810
     Construction             0.827
     Wholesale                0.839
     Retail                   0.849
     SIC35                    0.882
     SIC62                    0.894
     Size                     0.926
     SIC73                    0.929
     Services                 0.938
     SIC60                    0.952
     Size Nonlinearity        1.077
     Trading Activity         1.080

2. The realised cap-weighted index return over the test's 686 days
   autocorrelation at lags 1-10: -0.056, +0.044, -0.109, -0.060, +0.008, +0.024, -0.044, +0.014, -0.027, +0.052
   the Bartlett-weighted sum at 10 lags, 1 + 2 sum (1 - k/11) rho_k: 0.740
   variance of 21-day sums over 21 x the daily variance: 0.650 (32 blocks)

3. The bias statistic by half-year: the cap-weighted index, the equally weighted, and the members' median
   half      days   index   equal  members
   2024H1      98   0.892   0.802    0.904
   2024H2     128   1.171   1.084    1.020
   2025H1     122   1.528   1.455    1.187
   2025H2     128   0.762   0.826    0.929
   2026H1     123   1.109   1.107    1.153
   2026H2      87   0.866   0.976    1.052
   whole test: index 1.096, kurtosis of its z 8.4
```

### compare_bias.txt -- every run against the defaults, each statistic recomputed from its stored forecasts and outcomes

```text
13 runs: D = defaults; F15 = floor 15 (Pearson); C = correction; CL = correction + specific side; L = specific side alone; H63 = history half-life 63; H126 = history half-life 126; H252 = history half-life 252; V45 = covariance half-life 45; V180 = covariance half-life 180; T504 = two half-lives, 90 / 504; D0 = D0; H63T504 = H63T504
686 daily forecasts, 2024-02-08 to 2026-11-02; 32 non-overlapping 21-day ones

The bias statistic, daily (band for T = 686: +/-0.054)
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  the index, cap-weighted               1.096    1.096    1.102    1.096    1.090    1.094    1.095    1.096    1.132    1.101    1.080    1.024    1.078
  the index, equally weighted           1.072    1.072    1.064    1.064    1.072    1.071    1.071    1.071    1.136    1.041    1.008    0.996    1.008
  smallest 85, equally weighted         1.123    1.123    1.113    1.114    1.123    1.118    1.120    1.121    1.177    1.085    1.032    0.997    1.028
  largest 50, equally weighted          1.080    1.080    1.087    1.085    1.078    1.075    1.077    1.079    1.117    1.084    1.068    1.004    1.063
  minimum variance, long-only           1.357    1.354    1.403    1.402    1.361    1.348    1.349    1.351    1.524    1.306    1.275    1.174    1.275
  minimum variance, unconstrained       1.578    1.533    1.925    1.916    1.582    1.548    1.557    1.565    1.863    1.462    1.430    1.235    1.418

Its rolling 12-month windows' mean absolute deviation from one (MRAD)
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  the index, cap-weighted               0.169    0.169    0.171    0.169    0.168    0.169    0.169    0.169    0.168    0.213    0.190    0.183    0.190
  the index, equally weighted           0.137    0.137    0.133    0.133    0.137    0.137    0.137    0.137    0.171    0.168    0.148    0.172    0.148
  smallest 85, equally weighted         0.173    0.173    0.161    0.161    0.173    0.169    0.170    0.171    0.215    0.153    0.110    0.136    0.110
  largest 50, equally weighted          0.134    0.133    0.138    0.137    0.133    0.131    0.132    0.133    0.134    0.171    0.154    0.148    0.153
  minimum variance, long-only           0.395    0.389    0.465    0.461    0.398    0.389    0.389    0.391    0.583    0.325    0.292    0.244    0.291
  minimum variance, unconstrained       0.637    0.576    1.013    1.002    0.638    0.606    0.616    0.624    0.959    0.490    0.451    0.277    0.436

The 21-day check, non-overlapping (band +/-0.25)
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  the index, cap-weighted               0.952    0.952    0.957    0.951    0.946    0.949    0.951    0.952    0.990    0.944    0.935    0.856    0.933
  the index, equally weighted           0.895    0.895    0.887    0.887    0.895    0.894    0.894    0.894    0.944    0.861    0.832    0.813    0.831
  smallest 85, equally weighted         0.884    0.884    0.875    0.875    0.885    0.880    0.881    0.883    0.931    0.848    0.797    0.762    0.793
  largest 50, equally weighted          1.019    1.019    1.026    1.023    1.017    1.014    1.016    1.017    1.043    1.015    1.010    0.921    1.005
  minimum variance, long-only           1.063    1.064    1.077    1.077    1.069    1.048    1.055    1.059    1.128    1.057    1.038    0.970    1.030
  minimum variance, unconstrained       1.269    1.280    1.553    1.551    1.275    1.233    1.249    1.258    1.512    1.169    1.147    1.070    1.115

Each industry's members, cap-weighted: the bias statistic
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  Construction                          1.066    1.319    1.180    1.168    1.057    1.057    1.057    1.060    1.097    1.061    1.073    0.996    1.064
  Finance & Real Estate                 1.096    1.018    1.177    1.164    1.086    1.085    1.090    1.093    1.116    1.091    1.087    1.003    1.077
  Manufacturing                         1.036    1.037    1.081    1.075    1.032    1.021    1.026    1.031    1.078    1.023    1.021    0.994    1.005
  Mining                                0.959    0.959    1.015    1.012    0.957    0.948    0.952    0.955    1.001    0.940    0.948    0.947    0.936
  Retail                                1.119    1.165    1.145    1.110    1.086    1.090    1.099    1.108    1.118    1.134    1.125    1.081    1.094
  SIC20                                 1.125    1.126    1.240    1.232    1.120    1.105    1.111    1.117    1.147    1.128    1.117    1.027    1.097
  SIC28                                 1.135    1.136    1.175    1.165    1.126    1.112    1.117    1.124    1.140    1.152    1.148    1.084    1.123
  SIC35                                 1.054    1.051    1.076    1.036    1.016    1.015    1.029    1.040    1.055    1.078    1.056    1.031    1.017
  SIC36                                 1.232    1.231    1.240    1.204    1.197    1.237    1.238    1.237    1.240    1.247    1.230    1.111    1.235
  SIC37                                 1.124    1.123    1.147    1.068    1.050    1.118    1.119    1.119    1.132    1.129    1.113    1.084    1.106
  SIC38                                 1.129    1.129    1.167    1.167    1.129    1.119    1.123    1.126    1.156    1.128    1.113    1.030    1.103
  SIC49                                 1.085    1.084    1.129    1.129    1.084    1.079    1.079    1.081    1.119    1.062    1.057    0.983    1.051
  SIC60                                 0.974    0.974    1.001    0.996    0.969    0.977    0.976    0.976    1.020    0.939    0.973    0.997    0.977
  SIC62                                 1.066    1.196    1.106    1.103    1.065    1.064    1.066    1.067    1.099    1.058    1.048    1.044    1.047
  SIC63                                 1.086    1.087    1.140    1.128    1.075    1.057    1.066    1.075    1.097    1.085    1.064    0.971    1.037
  SIC67                                 1.048    1.048    1.105    1.105    1.048    1.043    1.045    1.046    1.100    1.012    1.005    0.958    1.001
  SIC73                                 1.094    1.090    1.105    1.099    1.088    1.088    1.091    1.093    1.095    1.115    1.098    1.060    1.091
  Services                              0.952    0.952    1.007    0.987    0.936    0.956    0.956    0.956    0.981    0.944    0.948    0.948    0.952
  Transport & Utilities                 0.993    0.994    1.038    1.036    0.992    0.985    0.988    0.991    1.014    0.998    0.982    0.940    0.974
  Wholesale                             1.055    0.979    1.223    1.206    1.045    1.027    1.036    1.044    1.085    1.040    1.044    0.982    1.017

Each industry's members, cap-weighted: the 21-day check (band +/-0.25)
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  Construction                          0.850    1.068    0.940    0.931    0.843    0.842    0.843    0.846    0.872    0.849    0.864    0.801    0.855
  Finance & Real Estate                 0.946    0.879    1.014    1.003    0.937    0.938    0.941    0.944    0.970    0.932    0.939    0.853    0.932
  Manufacturing                         0.982    0.982    1.023    1.019    0.978    0.968    0.973    0.977    1.032    0.962    0.966    0.935    0.951
  Mining                                0.888    0.888    0.941    0.938    0.886    0.879    0.882    0.885    0.934    0.856    0.869    0.866    0.860
  Retail                                1.109    1.155    1.136    1.099    1.075    1.090    1.096    1.102    1.115    1.113    1.111    1.078    1.090
  SIC20                                 1.079    1.081    1.190    1.183    1.074    1.062    1.068    1.073    1.117    1.054    1.053    0.975    1.037
  SIC28                                 0.926    0.926    0.958    0.950    0.918    0.909    0.913    0.919    0.932    0.931    0.930    0.889    0.911
  SIC35                                 1.018    1.017    1.042    1.000    0.979    0.983    0.996    1.006    1.035    1.023    1.019    0.974    0.983
  SIC36                                 1.038    1.037    1.046    1.015    1.009    1.046    1.046    1.043    1.031    1.052    1.034    0.932    1.041
  SIC37                                 0.841    0.840    0.857    0.802    0.790    0.811    0.823    0.831    0.849    0.843    0.831    0.805    0.801
  SIC38                                 0.918    0.918    0.950    0.950    0.918    0.908    0.911    0.914    0.924    0.918    0.908    0.824    0.898
  SIC49                                 0.852    0.852    0.885    0.885    0.852    0.850    0.849    0.850    0.905    0.822    0.817    0.765    0.815
  SIC60                                 1.022    1.023    1.051    1.046    1.018    1.025    1.025    1.024    1.060    0.987    1.024    1.005    1.027
  SIC62                                 1.099    1.228    1.141    1.138    1.097    1.098    1.099    1.100    1.138    1.079    1.081    1.048    1.080
  SIC63                                 0.957    0.959    1.003    0.992    0.947    0.925    0.938    0.947    0.957    0.959    0.932    0.809    0.902
  SIC67                                 0.935    0.936    0.982    0.982    0.935    0.932    0.933    0.934    1.019    0.882    0.892    0.839    0.890
  SIC73                                 1.030    1.025    1.040    1.033    1.023    1.025    1.027    1.029    1.063    1.024    1.029    0.921    1.024
  Services                              1.070    1.069    1.121    1.099    1.050    1.079    1.079    1.076    1.086    1.068    1.062    1.040    1.072
  Transport & Utilities                 0.883    0.884    0.928    0.926    0.882    0.878    0.880    0.882    0.914    0.871    0.873    0.820    0.868
  Wholesale                             0.953    0.877    1.094    1.079    0.943    0.924    0.934    0.942    0.991    0.935    0.932    0.866    0.904

Across the industries
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  mean |bias - 1|                       0.084    0.099    0.125    0.111    0.073    0.072    0.076    0.079    0.096    0.086    0.077    0.042    0.066
  outside the band                      0.650    0.550    0.800    0.750    0.600    0.600    0.600    0.600    0.800    0.800    0.600    0.300    0.500
  mean MRAD                             0.127    0.143    0.166    0.155    0.120    0.119    0.122    0.125    0.131    0.137    0.123    0.104    0.115
  the three thin: mean |bias - 1|       0.072    0.119    0.193    0.180    0.063    0.056    0.061    0.066    0.100    0.064    0.068    0.009    0.053

Random portfolios, long-only
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  median bias                           1.076    1.073    1.073    1.073    1.076    1.067    1.070    1.073    1.121    1.051    1.025    1.008    1.017
  p5                                    0.997    0.997    0.997    0.997    0.997    0.989    0.991    0.994    1.037    0.969    0.949    0.935    0.942
  p95                                   1.169    1.169    1.162    1.163    1.170    1.157    1.162    1.165    1.215    1.147    1.115    1.083    1.105
  below the band                        0.000    0.000    0.010    0.010    0.000    0.020    0.010    0.010    0.000    0.030    0.050    0.080    0.060
  above the band                        0.700    0.700    0.660    0.660    0.700    0.580    0.610    0.660    0.890    0.490    0.280    0.130    0.170
  median MRAD                           0.142    0.137    0.138    0.138    0.141    0.135    0.137    0.139    0.153    0.153    0.132    0.150    0.132

Random portfolios, long-short
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  median bias                           1.076    1.077    1.091    1.093    1.078    1.021    1.035    1.051    1.076    1.076    1.070    1.056    1.016
  p5                                    0.958    0.975    0.978    0.980    0.961    0.922    0.931    0.941    0.959    0.961    0.962    0.943    0.923
  p95                                   1.250    1.255    1.262    1.264    1.251    1.179    1.201    1.222    1.241    1.261    1.257    1.193    1.181
  below the band                        0.020    0.020    0.010    0.010    0.020    0.150    0.070    0.060    0.020    0.020    0.030    0.060    0.130
  above the band                        0.600    0.610    0.650    0.660    0.610    0.360    0.420    0.490    0.590    0.630    0.600    0.510    0.370
  median MRAD                           0.083    0.085    0.092    0.094    0.084    0.064    0.070    0.070    0.082    0.085    0.086    0.077    0.067

Every member singly
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  tested                              543.000  543.000  543.000  543.000  543.000  543.000  543.000  543.000  543.000  543.000  543.000  543.000  543.000
  median bias                           1.075    1.073    1.088    1.093    1.077    1.042    1.045    1.057    1.083    1.077    1.067    1.045    1.034
  p10                                   0.909    0.915    0.922    0.926    0.910    0.903    0.909    0.910    0.917    0.902    0.897    0.870    0.892
  p90                                   1.256    1.258    1.276    1.278    1.259    1.173    1.197    1.219    1.266    1.259    1.247    1.203    1.169
  mean |bias - 1|                       0.133    0.133    0.141    0.141    0.133    0.098    0.106    0.116    0.135    0.135    0.129    0.114    0.097
  below the band                        0.160    0.151    0.133    0.131    0.158    0.182    0.168    0.169    0.144    0.169    0.175    0.228    0.203
  above the band                        0.545    0.547    0.587    0.604    0.562    0.442    0.477    0.503    0.573    0.532    0.517    0.466    0.409
  above the kurtosis band               0.361    0.365    0.416    0.416    0.366    0.199    0.262    0.300    0.378    0.366    0.341    0.262    0.179
  median MRAD                           0.123    0.124    0.130    0.131    0.125    0.099    0.106    0.114    0.123    0.126    0.121    0.114    0.097

Members by capitalisation quintile, largest first: median bias
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  quintile 1 (109)                      1.124    1.127    1.144    1.130    1.119    1.093    1.100    1.114    1.130    1.130    1.121    1.098    1.089
  quintile 2 (109)                      1.083    1.074    1.103    1.104    1.076    1.053    1.057    1.062    1.090    1.080    1.080    1.049    1.046
  quintile 3 (109)                      1.082    1.076    1.089    1.094    1.079    1.046    1.049    1.060    1.083    1.085    1.069    1.045    1.034
  quintile 4 (108)                      1.045    1.046    1.052    1.060    1.050    1.001    1.023    1.034    1.055    1.039    1.028    0.998    0.992
  quintile 5 (108)                      1.048    1.050    1.051    1.062    1.053    1.014    1.030    1.038    1.054    1.046    1.038    1.013    1.005

Eigenportfolios against the estimated factor returns (rank 1 the largest)
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  rank 1                                1.104    1.100    1.115    1.115    1.104    1.104    1.104    1.104    1.118    1.104    1.090    1.031    1.090
  rank 2                                0.874    0.914    0.934    0.934    0.874    0.874    0.874    0.874    0.811    0.938    0.975    0.974    0.975
  rank 3                                0.963    1.008    1.007    1.007    0.963    0.963    0.963    0.963    0.947    0.961    0.992    1.009    0.992
  rank 5                                1.104    1.056    1.165    1.165    1.104    1.104    1.104    1.104    1.105    1.121    1.071    1.053    1.071
  rank 10                               1.238    1.257    1.619    1.619    1.238    1.238    1.238    1.238    1.546    1.243    1.262    1.076    1.262
  rank 15                               1.628    1.671    2.935    2.935    1.628    1.628    1.628    1.628    2.020    1.477    1.466    1.161    1.466
  rank 20                               2.124    2.103 1607.522 1607.522    2.124    2.124    2.124    2.124    2.813    1.904    1.745    1.265    1.745
  rank 22                               2.184    2.100 1321.932 1321.932    2.184    2.184    2.184    2.184    2.977    1.968    1.852    1.311    1.852
  rank 23                               2.262        - 1398.529 1398.529    2.262    2.262    2.262    2.262    2.967    1.987    1.856    1.278    1.856
  rank 24                               2.257        - 1486.658 1486.658    2.257    2.257    2.257    2.257    3.073    1.968    1.879    1.261    1.879
  rank 25                               2.214        - 1689.641 1689.641    2.214    2.214    2.214    2.214    2.941    1.910    1.836    1.194    1.836

Eigenportfolios: the smallest five, mean bias
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  ranks k-4..k                          2.219    2.080 1465.303 1465.303    2.219    2.219    2.219    2.219    2.963    1.948    1.861    1.254    1.861

Members' median bias statistic by half-year
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  2024H1 (98)                           0.904    0.903    0.916    0.913    0.903    0.914    0.909    0.905    0.932    0.891    0.903    0.903    0.912
  2024H2 (128)                          1.020    1.027    1.040    1.038    1.019    1.022    1.019    1.021    1.038    1.003    1.004    1.002    1.004
  2025H1 (122)                          1.187    1.189    1.198    1.202    1.181    1.141    1.153    1.169    1.150    1.202    1.176    1.152    1.133
  2025H2 (128)                          0.929    0.933    0.940    0.941    0.933    0.913    0.912    0.920    0.959    0.921    0.924    0.864    0.906
  2026H1 (123)                          1.153    1.161    1.173    1.180    1.157    1.091    1.114    1.133    1.160    1.152    1.155    1.115    1.084
  2026H2 (87)                           1.052    1.055    1.070    1.072    1.052    0.962    0.984    1.009    1.051    1.055    1.040    1.005    0.950

The equally weighted index's bias statistic by half-year
                                            D      F15        C       CL        L      H63     H126     H252      V45     V180     T504       D0  H63T504
  2024H1 (98)                           0.802    0.802    0.800    0.800    0.802    0.802    0.802    0.802    0.886    0.764    0.791    0.846    0.791
  2024H2 (128)                          1.084    1.084    1.076    1.076    1.084    1.084    1.084    1.084    1.172    1.015    1.005    1.063    1.005
  2025H1 (122)                          1.455    1.455    1.446    1.446    1.456    1.455    1.455    1.455    1.356    1.524    1.443    1.448    1.442
  2025H2 (128)                          0.826    0.826    0.814    0.814    0.826    0.826    0.826    0.826    0.937    0.801    0.814    0.674    0.813
  2026H1 (123)                          1.107    1.109    1.104    1.104    1.107    1.106    1.106    1.106    1.275    1.011    0.966    0.889    0.965
  2026H2 (87)                           0.976    0.977    0.968    0.968    0.976    0.973    0.974    0.975    1.077    0.881    0.815    0.821    0.813
```

### bias/runner_log.txt -- the rerun, one at a time

```text
2026-11-12 03:12:07  runner started, pid 29292; 6.5 GB available
2026-11-12 03:12:07  D: 6.5 GB available
2026-11-12 03:12:07  D: attempt 1: --industries --membership --bias --bias-guard 0
2026-11-12 03:20:48  D: done in 8.7 min
2026-11-12 03:20:48  F15: 6.9 GB available
2026-11-12 03:20:48  F15: attempt 1: --industries --membership --bias --bias-guard 0 --merge-industry Construction=Retail --merge-industry Finance & Real Estate=SIC62 --merge-industry Wholesale=Retail
2026-11-12 03:30:07  F15: done in 9.3 min
2026-11-12 03:30:07  C: 6.2 GB available
2026-11-12 03:30:07  C: attempt 1: --industries --membership --bias --bias-guard 0 --sampling-correction
2026-11-12 03:41:01  C: done in 10.9 min
2026-11-12 03:41:01  CL: 5.9 GB available
2026-11-12 03:41:01  CL: attempt 1: --industries --membership --bias --bias-guard 0 --sampling-correction --leverage-correction
2026-11-12 03:53:30  CL: done in 12.5 min
2026-11-12 03:53:30  L: 6.6 GB available
2026-11-12 03:53:30  L: attempt 1: --industries --membership --bias --bias-guard 0 --leverage-correction
2026-11-12 04:05:39  L: done in 12.1 min
2026-11-12 04:05:39  H63: 6.8 GB available
2026-11-12 04:05:39  H63: attempt 1: --industries --membership --bias --bias-guard 0 --specific-half-life 63
2026-11-12 04:15:24  H63: done in 9.8 min
2026-11-12 04:15:24  H126: 6.5 GB available
2026-11-12 04:15:24  H126: attempt 1: --industries --membership --bias --bias-guard 0 --specific-half-life 126
2026-11-12 04:25:05  H126: done in 9.7 min
2026-11-12 04:25:05  H252: 6.3 GB available
2026-11-12 04:25:05  H252: attempt 1: --industries --membership --bias --bias-guard 0 --specific-half-life 252
2026-11-12 04:34:42  H252: done in 9.6 min
2026-11-12 04:34:42  V45: 6.3 GB available
2026-11-12 04:34:42  V45: attempt 1: --industries --membership --bias --bias-guard 0 --half-life 45
2026-11-12 04:44:24  V45: done in 9.7 min
2026-11-12 04:44:24  V180: 6.3 GB available
2026-11-12 04:44:24  V180: attempt 1: --industries --membership --bias --bias-guard 0 --half-life 180
2026-11-12 04:53:56  V180: done in 9.5 min
2026-11-12 04:53:56  T504: 6.2 GB available
2026-11-12 04:53:56  T504: attempt 1: --industries --membership --bias --bias-guard 0 --correlation-half-life 504
2026-11-12 05:03:27  T504: done in 9.5 min
2026-11-12 05:03:27  runner finished; every run done
2026-11-12 05:03:46  runner started, pid 26604; 5.9 GB available
2026-11-12 05:03:46  D: results exist, skipped
2026-11-12 05:03:46  F15: results exist, skipped
2026-11-12 05:03:46  C: results exist, skipped
2026-11-12 05:03:46  CL: results exist, skipped
2026-11-12 05:03:46  L: results exist, skipped
2026-11-12 05:03:46  H63: results exist, skipped
2026-11-12 05:03:46  H126: results exist, skipped
2026-11-12 05:03:46  H252: results exist, skipped
2026-11-12 05:03:46  V45: results exist, skipped
2026-11-12 05:03:46  V180: results exist, skipped
2026-11-12 05:03:46  T504: results exist, skipped
2026-11-12 05:03:46  D0: 5.9 GB available
2026-11-12 05:03:46  D0: attempt 1: --industries --membership --bias --bias-guard 0 --newey-west-lags 0
2026-11-12 05:11:49  D0: done in 8.1 min
2026-11-12 05:11:49  runner finished; every run done
2026-11-12 05:12:11  runner started, pid 33032; 6.0 GB available
2026-11-12 05:12:11  D: results exist, skipped
2026-11-12 05:12:11  F15: results exist, skipped
2026-11-12 05:12:11  C: results exist, skipped
2026-11-12 05:12:11  CL: results exist, skipped
2026-11-12 05:12:11  L: results exist, skipped
2026-11-12 05:12:11  H63: results exist, skipped
2026-11-12 05:12:11  H126: results exist, skipped
2026-11-12 05:12:11  H252: results exist, skipped
2026-11-12 05:12:11  V45: results exist, skipped
2026-11-12 05:12:11  V180: results exist, skipped
2026-11-12 05:12:11  T504: results exist, skipped
2026-11-12 05:12:11  D0: results exist, skipped
2026-11-12 05:12:11  H63T504: 6.0 GB available
2026-11-12 05:12:11  H63T504: attempt 1: --industries --membership --bias --bias-guard 0 --specific-half-life 63 --correlation-half-life 504
2026-11-12 05:20:39  H63T504: done in 8.5 min
2026-11-12 05:20:39  runner finished; every run done
```

### bias/stopped/ -- the stopped runs' last progress lines

```text
D: [bias] 50 of 686 forecasts, 0.5 min, about 5.8 min left
F15: [bias] 50 of 686 forecasts, 0.4 min, about 5.3 min left
C: [bias] 400 of 686 forecasts, 4.9 min, about 3.5 min left
CL: [bias] 150 of 686 forecasts, 1.9 min, about 6.6 min left
```

### final_checks.txt -- the final code: the defaults, the correction and its specific side against their captures; the suite, ruff, hygiene

```text
B4: status 0, mean R2 0.293203, 938 periods, 25 factors, 50s, the estimator 1.3s
claude/scratch/midoffice11/r3/B3 against claude/scratch/midoffice11/final/B4: 21 fields compared; equal: 21; differing: none
  qualifications: 7 and 7; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
C4: status 0, mean R2 0.293203, 938 periods, 25 factors, 48s, the estimator 1.3s
claude/scratch/midoffice11/r1/C2 against claude/scratch/midoffice11/final/C4: 21 fields compared; equal: 21; differing: none
  qualifications: 8 and 8; differing: 0
  report: 191 and 191 lines; 0 differing line(s)
CL4: status 0, mean R2 0.293203, 938 periods, 25 factors, 49s, the estimator 1.8s
claude/scratch/midoffice11/r1/CL2 against claude/scratch/midoffice11/final/CL4: 21 fields compared; equal: 21; differing: none
  qualifications: 8 and 8; differing: 0
  report: 192 and 192 lines; 0 differing line(s)
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2338 passed, 1 skipped, 19 warnings in 91.90s (0:01:31) ===========
All checks passed!
ruff: clean over the changed Python files
scripts/run_sp500_model.py: CRLF (1731 lines), parses, ASCII
src/praxis/midoffice/model/bias.py: LF (501 lines), parses, ASCII
src/praxis/midoffice/model/crosssectional.py: CRLF (1528 lines), parses, ASCII
src/praxis/midoffice/model/covariance.py: LF (577 lines), parses, ASCII
src/praxis/midoffice/model/specificrisk.py: CRLF (639 lines), parses, ASCII
tests/midoffice/test_bias.py: LF (200 lines), parses, ASCII
tests/midoffice/test_crosssectional.py: CRLF (2051 lines), parses, ASCII
tests/midoffice/test_covariance.py: LF (757 lines), parses, ASCII
docs/FACTOR_MODEL.md: CRLF (1201 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (500 lines), -, ASCII
claude/handoffs/BRIEF_midoffice11_bias_statistics.md: LF (98 lines), -, ASCII
all ok
```

### ruff_head_untouched.txt -- ruff over HEAD's copies of the files with findings no piece touched

```text
claude\scratch\midoffice11\head\characteristics.py:30:101: E501 Line too long (154 > 100)
claude\scratch\midoffice11\head\characteristics.py:46:39: F401 [*] `typing.Sequence` imported but unused
claude\scratch\midoffice11\head\exposure_basis.py:45:20: F401 [*] `typing.Mapping` imported but unused
claude\scratch\midoffice11\head\exposure_basis.py:113:5: N806 Variable `T` in function should be lowercase
claude\scratch\midoffice11\head\exposure_basis.py:141:5: N806 Variable `T` in function should be lowercase
claude\scratch\midoffice11\head\timeseries.py:31:39: F401 [*] `typing.Sequence` imported but unused
Found 6 errors.
[*] 3 fixable with the `--fix` option.
claude\scratch\midoffice11\head\test_covariance.py:22:5: F401 [*] `praxis.midoffice.model.covariance.CovarianceEstimate` imported but unused
claude\scratch\midoffice11\head\test_covariance.py:498:5: F811 Redefinition of unused `test_a_non_positive_half_life_is_refused` from line 234: `test_a_non_positive_half_life_is_refused` redefined here
Found 2 errors.
[*] 1 fixable with the `--fix` option.
```

## git add (always last -- Rule 43)

This piece -- midoffice10's rulings, the out-of-sample bias statistics and their options, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add src/praxis/midoffice/model/crosssectional.py src/praxis/midoffice/model/covariance.py src/praxis/midoffice/model/specificrisk.py src/praxis/midoffice/model/bias.py scripts/run_sp500_model.py tests/midoffice/test_bias.py tests/midoffice/test_crosssectional.py tests/midoffice/test_covariance.py docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice11_bias_statistics.md claude/retros/RETRO_midoffice11_bias_statistics.md
```
