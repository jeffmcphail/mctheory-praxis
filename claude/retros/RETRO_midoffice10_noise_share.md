# Retro: Mid-office 10 -- midoffice09's rulings, and the noise share measured

**Brief:** `claude/handoffs/BRIEF_midoffice10_noise_share.md` -- written in Chat and pasted
into the Code session, transcribed verbatim
**Date:** 2026-11-10
**Duration:** the brief arrived shortly before 18:51Z (13:51 EST). The two rulings, built and
measured, then the correction, its tests and its run, to about 14:20; the folds, the floors and
Size Nonlinearity's runs to about 14:38; the final checks, the docs, the state file and this
retro to about 14:50.
**Status:** COMPLETE.
- **Rulings 1 and 2:** built, measured each against the one before; the build's defaults and the
  run's default panel (`_listed`) now follow them, as ruled.
- **Items 1 to 3:** measured. No default changed: the correction and the folds are options, off.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 1, PCG spared.** Of the 7 samples the isolated-error rule finds, 6 are removed and
  PCG's of 2026-01-09 is spared: its own record's weighted count is 1.000 of it. Its count changes
  over 63 member-days. Mean R-squared stays 0.293204; PCG's forecast moves from 23.3340% to
  23.3506%. Its series stays flagged as alternating.
- **Ruling 2, spans end off an exchange.**
  - **Where it trades.** The reference records say, in `market` and `primary_exchange`. Of the
    11,950 samples, 21 are OTC Link's, all Signature Bank's and First Republic's: there is no
    other candidate, and no overrides entry was needed.
  - **What it ends.** SBNY's span from 2023-03-28 goes, 864 priced days. FRCB's is kept to
    2023-05-03, the one member-day the index held it on OTC Link -- the exception to the guard's
    premise -- and 138 priced days go after it. 1,002 priced cells change and no member-day;
    every build guard passes.
  - **One security-period leaves a regression.** The 2023-05-04 period reads FRCB's membership
    from 2023-05-03 and its return from 2023-05-04, an OTC price the rule removes: 502 names
    that day become 501.
  - **The model.** Mean R-squared 0.293204 to 0.293203. Signature Bank's 395.72% forecast goes,
    551 forecasts to 550, and 966 specific returns measured outside the regressions go with the
    two spans (SBNY 829, FRCB 137).
- **Item 1, the sampling-error correction** (`--sampling-correction`, off by default).
  - **What it subtracts.** Each period's sampling covariance, weighted exactly as the covariance
    estimator carries that period's noise. Newey-West at 10 lags carries 0.9625 of it, which a
    simulation confirms.
  - **Verified:** over 12 replications, the corrected diagonal is unbiased.
  - **The noise is larger than reported.** The noise share weighted as the covariance is runs
    about half again over the equal-weighted one the run reports: Wholesale 68.8% (46.2%), Size
    Nonlinearity 67.4% (46.6%), SIC20 55.5%, Construction 48.6%, Finance & Real Estate 46.5%.
    Every factor's sampling variance in 2026 is 1.3 to 2.0 times its 2023-2026 mean, and a 90-day
    half-life sits on 2026.
  - **Vols.** The correction takes 0.5% (Size) to 32% (Wholesale) off them.
  - **The repair binds hard:** 9 of 25 eigenvalues negative, the most negative -0.0070 a year,
    in contrasts of the thin industries. That is the estimator's own sampling error: drawn from
    a truth comfortably above the noise and estimated the same way, the subtraction leaves a
    median of 7. Clipped, those directions are close to riskless.
  - **Reference portfolios.** The index -1.1%, the equal-weighted index +0.8%, the smallest 85
    -0.6%, the largest 50 -1.0%; the thin industries' portfolios -9.8% (Construction), -17.5%
    (Wholesale), -8.6% (Finance & Real Estate).
- **Item 2, the thin industries.**
  - **The rule's targets.** By the correlation of the factor returns over the estimated periods:
    Construction into Retail (0.606), Wholesale into Retail (0.709), Finance & Real Estate into
    SIC62 (0.761). The model's recent-weighted correlation points elsewhere for two -- SIC36 for
    Construction, SIC38 for Wholesale -- and both fit worse when run.
  - **Folded each alone:** mean R-squared loses 0.0030, 0.0018 and 0.0024, about what a regressor
    of pure noise adds; the members' specific risk rises a median 12.4%, 1.6% and 9.4%.
  - **Floors 15, 20 and 25,** the same rule smallest first: 3, 5 and 8 folds; mean R-squared
    0.2861, 0.2769 and 0.2628. The correction's negative eigenvalues fall from 9 to 6, 5 and 2.
- **Item 3, Size Nonlinearity.**
  - **Corrected:** its vol goes from 2.32% to 1.71% (1.32% before the repair).
  - **Without it:** mean R-squared loses 0.0036, and no reference portfolio's risk moves by more
    than 0.07 of a point.
  - **The folds** leave its noise share at 62% to 67%.
- **What I would choose.**
  - **Not the correction as the default:** its subtract-and-clip makes riskless directions,
    whatever the truth.
  - **Floor 15:** the three thin industries folded by the rule.
  - **Size Nonlinearity kept, as ruled.**
  - **The weighted noise share** reported as the standing diagnostic.

  Reasons under "What I would choose, and why".
- **Docs:** `docs/FACTOR_MODEL.md` brought in line, though the brief listed none: the two
  rulings, the correction, the folds, Size Nonlinearity's measurement.
- **Tests:** 2,311 passed, 1 skipped. 16 are new.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice10_noise_share.md`, verbatim. In outline:

- **"committed -- midoffice09."** Checked: fccc480 in `git log`.
- **Rulings on midoffice09's decisions:**
  1. **PCG:** spare it, by the general addition: "a sample its own record's weighted count
     confirms within 10% is not an error". Its series stays flagged as alternating.
  2. **Leavers:** end a security's span when it stops trading on an exchange.
     - If the reference records say where a ticker trades, use them, and report every span the
       rule ends and the specific returns it removes.
     - If they do not, an overrides entry for SBNY with its evidence, and report the other
       candidates the prices suggest.
     - Guard: no member-day changes, since the index holds only listed shares.
     - A leaver that stays listed keeps forecasting from its own history, as ruled.
  3. **History weighting:** leave it equal for now.
- **midoffice10:** "Rulings 1 and 2 first, each measured. Then the noise-share question,
  measured before anything is chosen."
  1. **The sampling-error correction.**
     - What: the factor covariance less the mean sampling covariance of the factor returns,
       weighted as the covariance is, with the PSD repair where it is needed.
     - Report every factor's vol before and after, and whether and where the repair binds.
     - Report the risk of reference portfolios before and after: the index, an equal-weighted
       index, the size pair's two, and a single-industry portfolio for each thin industry.
  2. **The thin industries:** Construction (8), Wholesale (12), Finance & Real Estate (12),
     each already a whole division.
     - For each: the industry factor its own factor return correlates with most, and the model
       with it merged there -- R-squared, noise shares, its members' specific risk.
     - Then a floor of 15, 20 and 25 members applied the same way to every industry.
  3. **Size Nonlinearity at 46.6%:** the correction's effect on it, and the model without it
     (`--size-nonlinearity none`).
- **"Change no default. Report, and say which you would choose and why."** Retro with its
  `git add` and the state file in the same commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `src/praxis/midoffice/refdata/securities.py` | ruling 1: `Repair.weighted`, `Repair.spared`, `isolated_errors(weighted=...)`; ruling 2: `Delisting`, `listed_spans`, held member-days counted on the trading calendar | 238, 240-242, 254-257, 265, 267-271, 297, 299-300, 304-367 |
| `scripts/build_sp500_securities.py` | `--spare weighted` and `--listed exchange` (the new defaults), the suffix `_listed`; the weighted counts read before the repairs; the off-exchange cut after the gate, each span listed with the days it removes; the report's `rules`, `delistings`, `weighted`, `spared`; the docstring | 31-37, 42-45, 123-133, 194-223, 283-310, 570-580 |
| `scripts/run_sp500_model.py` | `DEFAULT_PANEL` the `_listed` panel; `--sampling-correction` and its table; `--merge-industry A=B` | 67, 75-78, 555-563, 828-829, 871-886, 920, 1202-1217 |
| `src/praxis/midoffice/model/crosssectional.py` | `sampling_correction` (off by default): each period's full sampling covariance kept, `sampling_weights` and `_weighted_sampling_bias`, the subtraction and `nearest_psd`; the notes' `sampling_correction`, `weighted_sampling_variance`, `corrected_variance`, `psd_repair`, `sampling_filled`, `weighted_noise_share`; the qualifications; the docstrings | 96, 99-105, 165-173, 234-258, 363-396, 714-764, 788, 860-868, 932-933, 1030-1048, 1073-1082, 1254-1258 |
| `src/praxis/midoffice/refdata/industries.py` | `merge_industries`; `IndustryScheme.folded`; `describe()` names the folds | 44, 145-148, 195-197, 379-413 |
| `tests/midoffice/test_securities.py` | 2 new tests | 332-387 |
| `tests/midoffice/test_crosssectional.py` | 11 new tests; the covariance imports | 19-33, 1569-1734 |
| `tests/midoffice/test_industries.py` | 3 new tests | 27, 368-400 |
| `docs/FACTOR_MODEL.md` | the correction and the folds beside the noise diagnostic; Size Nonlinearity's measurement; the two rulings in the build; the default panel | 512-567, 829-835, 1023-1041, 1085-1089 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice10_noise_share.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice10_noise_share.md` | new: this retro | new |

Written outside the tracked files, all gitignored and all new; nothing in `data/` was
overwritten or deleted:

- `data/external/polygon/sp500_securities_2022-01-03_2026-11-03_listed.npz` and `.json`: the
  panel with midoffice09's two rulings, and its report. midoffice09's `_repaired` is untouched.
- `claude/scratch/midoffice10/`: the scratch, including the panels of each step (`p0` to `p4`).

The cache gained nothing: every record the rulings read was already in it.

---

## Ruling 1: a sample its own weighted count confirms is not an error

`isolated_errors` takes each sample's own record's weighted count. A sample the rule finds is
spared where that count is within 10% of it -- two fields of one record agreeing, the same day's
record, so nothing looks ahead -- and is still listed, as spared. The build reads the weighted
counts from the cache before the repairs (`--spare weighted`, the default; `none` is
midoffice09's build).

| Sample | Count | Its weighted count, over it | |
|---|---|---|---|
| DGX 2025-04-09 | 11,610,000 | 9.559 | removed |
| EG 2025-04-09 | 47,790,000 | 0.898 | removed |
| NWS 2024-07-09 | 162,170,000 | 3.509 (multi-class) | removed |
| PCG 2026-01-09 | 2,197,910,425 | 1.000 | **spared** |
| RDDT 2025-12-22 | 51,670,930 | 3.667 (multi-class) | removed |
| SBUX 2026-07-14 | 139,700,000 | 8.158 | removed |
| XYZ 2026-01-09 | 59,998,000 | 10.128 | removed |

EG's weighted count sits 10.2% below the sample, just outside the tolerance, and within 1% of
its neighbours: it confirms the repair.

**Against midoffice09's panel:** 63 member-days change, PCG's count from 2026-01-09 to
2026-04-10, and nothing else in the panel; membership identical. **The model** (M9 to Q1):

- **R-squared:** 0.293204 both;
- **factors:** every vol within 0.01%, every mean within 0.01%;
- **specific risk:** PCG's 23.3340% to 23.3506%, the largest move.

## Ruling 2: a span ends where its security stops trading on an exchange

**The records say where a ticker trades.** Every share-count sample's cached reference record
carries `market` and `primary_exchange`. Over the panel's 11,950 samples:

- markets: "stocks" 11,929, "otc" 21;
- primary exchanges: XNYS 8,143, XNAS 3,765, OTC Link 21, BATS 21 (Cboe BZX, an exchange).

The 21 OTC samples are SBNY's 17 and FRCB's 4. So the rule reads the records, and there are no
other candidates for the prices to suggest. SBNY's override was not needed.

**The rule** (`securities.listed_spans`, the build's `--listed exchange`):

- **Where the span leaves an exchange:** its first sample whose record's market is not "stocks".
- **Where it ends:** on its last sample on an exchange before that, or it goes whole if it has
  none.
- **The guard:** it runs on through any member-day after that, so no member-day changes. Those
  days are counted on the panel's trading calendar.

The build lists each span it ends, with the priced days it removes:

| Span | Off an exchange | Kept to | Member-days held | Priced days removed |
|---|---|---|---|---|
| SBNY, BBG000M6TR37, 2023-03-28..2026-11-03 | 2023-03-28 (OTC Link) | the span goes | none | 864, 2023-03-28..2026-11-03 |
| FRCB, BBG0019LSQ49, 2023-05-03..2023-11-17 | 2023-05-03 (OTC Link) | 2023-05-03 | 2023-05-03 | 138, 2023-05-04..2023-11-17 |

**The guard's premise has one exception.** The index held First Republic on 2023-05-03, a day
it traded on OTC Link as FRCB. The member-day stays, as the guard requires.

**Against ruling 1's panel:** 1,002 priced cells change -- prices, total-return prices, raw
prices, volumes, highs and lows -- and 1,043 ticker and share-count cells, none on a member-day.
Membership is identical, and every build guard passes.

**One security-period leaves a regression, through the lag.** Period 2023-05-04 regresses that
day's returns on 2023-05-03's membership, when FRCB was still a member. Its 2023-05-04 return,
from an OTC price the rule removes, goes, so that period holds 501 names instead of 502. A span
run one trading day past its last member-day would keep it (a decision below).

**The specific returns it removes:** 966 measured outside the regressions (55,830 to 54,864):
SBNY's 829 and FRCB's 137. FRCB's one inside the regression goes too, its 138 days in all.

**The model** (Q1 to Q2):

- **R-squared:** 0.293204 to 0.293203.
- **Factors:** every vol and mean within 0.01%.
- **The residual spread:** its MAD unchanged at 0.1333, its sd 0.3403 to 0.3446. FRCB's share
  of the sd's square goes from 67.6% to 68.4%, its average now over one fewer day.
- **Specific risk:** p10 / median / p90 15.58 / 24.79 / 40.56% to 15.58 / 24.76 / 40.49%.
  Signature Bank's 395.72% forecast goes, 551 forecasts to 550; no other moves past 0.01 of a
  point.

**The two rulings together are the build's defaults** (`--spare weighted --listed exchange`,
suffix `_listed`), and the run's `DEFAULT_PANEL` is the `_listed` panel. midoffice09's panel
rebuilds from `--spare none --listed any` exactly.

---

## Item 1: the sampling-error correction

### What it subtracts

The estimated factor returns are the true ones plus each period's estimation error,
independent across periods. So their covariance estimates the factor covariance plus the
errors', and `estimate_cross_sectional_model(sampling_correction=True)` subtracts that.

- **The input.** Each period's full sampling covariance, `sigma^2 (X'WX)^-1` over the factors
  it estimated, kept as the regression runs. The same quantity `noise_share` averages the
  diagonal of.
- **The weighting.** The brief's "weighted as the covariance is", taken exactly. The sample,
  exponential and Newey-West estimators are each a quadratic form `R' K R`, so independent noise
  of covariance `S_t` adds `sum_t K_tt S_t` to their estimate, and `K_tt` is the estimate of a
  series one in period t and zero elsewhere (`sampling_weights`).
  - The weighting, its centring and Newey-West's lag terms are all included.
  - Newey-West at 10 lags and half-life 90 carries 0.9625 of a period's noise: centring on the
    weighted mean makes a period's noise correlate slightly negatively with its neighbours'.
  - A 4,000-draw simulation agrees: 1.2031 predicted, 1.1991 +/- 0.0045 simulated.
  - The lag-0 weights alone, `(w - w^2) / (1 - sum w^2)`, carry all of it, so they would
    subtract 1 / 0.9625 of what the estimate holds: 3.9% too much.
  - The two-half-life and shrunk estimators are not quadratic forms, and are refused.
- **The repair.** The subtracted matrix, annualised, goes through `risk.basis.nearest_psd` at
  the configured floor (1e-8 of the largest eigenvalue).
- **The notes** keep, beside the uncorrected diagonal (`observed_variance`):
  - the bias's diagonal (`weighted_sampling_variance`);
  - the diagonal before the repair (`corrected_variance`);
  - the repair itself (`psd_repair`);
  - any period without a sampling covariance, filled with the others' weighted mean -- none
    here.
- **The qualifications** name every factor the subtraction left at or below zero -- none here,
  every subtracted diagonal is positive -- and quote the weighted share.
- **Verified** on synthetic worlds whose factor returns are known: over 12 replications,
  (corrected - realized) / predicted averages zero, where the uncorrected diagonal's averages
  one. An all-noise factor's subtraction lands below zero in 3 of 10 seeds, and in each the
  repair binds and the qualification names it.

### Every factor's vol, before and after

At the defaults with `--sampling-correction` (run C). Its factor returns, R-squared, exposures
and specific risk equal the defaults' run (Q2b) exactly, and the uncorrected covariance
re-estimated from its factor returns equals Q2b's to the bit.

| Factor | Vol as estimated | Noise share, eq. | Noise share, weighted | Subtracted | Repaired | Change |
|---|---|---|---|---|---|---|
| Construction | 19.42% | 38.2% | 48.6% | 13.91% | 15.05% | -22.5% |
| Finance & Real Estate | 15.34% | 31.0% | 46.5% | 11.22% | 12.91% | -15.9% |
| Manufacturing | 10.20% | 23.2% | 33.5% | 8.32% | 8.82% | -13.5% |
| Mining | 18.60% | 18.5% | 27.2% | 15.88% | 15.98% | -14.1% |
| Retail | 13.11% | 10.7% | 16.1% | 12.01% | 12.28% | -6.3% |
| SIC20 | 11.96% | 33.6% | 55.5% | 7.98% | 9.35% | -21.8% |
| SIC28 | 13.14% | 10.2% | 15.7% | 12.06% | 12.14% | -7.6% |
| SIC35 | 18.81% | 5.2% | 7.0% | 18.14% | 18.25% | -3.0% |
| SIC36 | 21.13% | 4.5% | 5.3% | 20.56% | 20.77% | -1.7% |
| SIC37 | 14.22% | 19.1% | 27.3% | 12.13% | 12.18% | -14.4% |
| SIC38 | 14.70% | 10.6% | 15.9% | 13.48% | 13.70% | -6.8% |
| SIC49 | 15.80% | 11.3% | 16.9% | 14.40% | 14.51% | -8.1% |
| SIC60 | 21.85% | 7.4% | 10.5% | 20.67% | 20.77% | -4.9% |
| SIC62 | 18.73% | 11.5% | 15.0% | 17.27% | 17.49% | -6.6% |
| SIC63 | 14.57% | 14.5% | 22.7% | 12.81% | 12.95% | -11.1% |
| SIC67 | 15.01% | 16.8% | 25.4% | 12.97% | 13.29% | -11.5% |
| SIC73 | 18.18% | 2.9% | 4.5% | 17.77% | 17.85% | -1.8% |
| Services | 12.72% | 28.0% | 42.6% | 9.64% | 10.34% | -18.8% |
| Transport & Utilities | 13.43% | 16.5% | 25.1% | 11.62% | 11.80% | -12.1% |
| Wholesale | 13.98% | 46.2% | 68.8% | 7.81% | 9.51% | -32.0% |
| Volatility | 9.37% | 3.5% | 5.7% | 9.11% | 9.28% | -1.1% |
| Momentum | 7.22% | 3.6% | 5.1% | 7.03% | 7.11% | -1.5% |
| Size | 3.72% | 7.7% | 10.7% | 3.52% | 3.70% | -0.5% |
| Size Nonlinearity | 2.32% | 46.6% | 67.4% | 1.32% | 1.71% | -26.2% |
| Trading Activity | 6.25% | 7.5% | 10.5% | 5.92% | 6.07% | -3.0% |

"Subtracted" is the vol the subtraction leaves, "Repaired" the model's, "Change" the model's
against the estimate's.

**Why the weighted shares are larger.** The equal-weighted share divides the full sample's mean
sampling variance by a variance Newey-West weights toward the recent periods. The sampling
variance is not constant over the sample (`sampling_profile.txt`). As a multiple of its
2023-2026 mean, by half-year:

| | 2023H1 | 2023H2 | 2024H1 | 2024H2 | 2025H1 | 2025H2 | 2026H1 | 2026H2 |
|---|---|---|---|---|---|---|---|---|
| Wholesale | 0.70 | 0.60 | 0.69 | 0.87 | 0.97 | 0.96 | 1.62 | 1.76 |
| Size Nonlinearity | 0.83 | 0.66 | 0.71 | 0.84 | 0.96 | 0.93 | 1.55 | 1.71 |
| SIC73 | 0.79 | 0.65 | 0.71 | 0.83 | 0.89 | 0.85 | 1.67 | 1.82 |

- **What moved.** The cross-sections hold about 500 names a period throughout, so it is
  residual dispersion that rose in 2026.
- **The ratio.** Newey-West's weights carry 1.23 to 1.72 times the mean, times 0.9625: the
  weighted share is about 1.45 times the reported one across the factors.
- **What it means.** The run's noise-share column understates the noise in the current regime
  by about a third.

### Whether and where the repair binds

It binds. The subtraction leaves 9 of the 25 eigenvalues negative, the most negative -0.0070 a
year. Each direction's largest loadings, with its observed and noise variance:

| Eigenvalue | Direction (largest loadings) | Observed | Noise |
|---|---|---|---|
| -0.0070 | Finance & Real Estate +0.67, Construction +0.57, SIC62 -0.20, SIC20 -0.19 | 0.0048 | 0.0118 |
| -0.0046 | Wholesale +0.67, Finance & Real Estate -0.36, Construction +0.31, SIC62 +0.27 | 0.0056 | 0.0102 |
| -0.0035 | SIC20 +0.52, Wholesale +0.47, SIC36 +0.31, SIC49 -0.24 | 0.0041 | 0.0075 |
| -0.0027 | SIC20 +0.48, Construction +0.34, SIC36 -0.33, Services -0.33 | 0.0037 | 0.0064 |
| -0.0024 | Services -0.58, SIC67 +0.55, SIC20 -0.31, SIC38 +0.21 | 0.0033 | 0.0057 |
| -0.0011 | Manufacturing +0.68, Transport & Utilities -0.31, SIC60 +0.30, Mining -0.27 | 0.0028 | 0.0039 |
| -0.0010 | SIC60 +0.42, Manufacturing -0.40, SIC62 -0.35, Transport & Utilities -0.32 | 0.0029 | 0.0039 |
| -0.0001 | Size Nonlinearity +0.69, Size +0.50, SIC67 -0.25, Services -0.23 | 0.0012 | 0.0013 |
| -0.0000 | Size Nonlinearity -0.52, Services -0.39, Trading Activity +0.37, SIC67 -0.32 | 0.0026 | 0.0026 |

**Observed against noise, direction by direction.** The generalized eigenvalues run from 0.25
up, and 9 directions' observed variance is below the noise it should contain. In expectation
none would be.

**It is the estimator's own sampling error.** 200 draws each of 938 daily periods, from a truth
plus the measured noise, estimated as the run does, the noise subtracted:

| Truth | Negative eigenvalues, median (p10-p90) | Most negative, median |
|---|---|---|
| the repaired matrix (9 directions at zero) | 10 (9-10) | -0.0064 |
| the repaired matrix, its clipped eigenvalues raised to the noise's median | 7 (7-8) | -0.0055 |
| the matrix as estimated, no direction below its noise | 7 (6-8) | -0.0041 |
| **the run** | **9** | **-0.0070** |

- **Why.** Newey-West at 10 lags on a 90-day half-life is a noisy estimate of 25 factors: its
  smallest eigenvalues sit well below the truth, and subtracting the noise takes them through
  zero.
- **What it means.** The repair will bind here whatever the truth, and clipping leaves those
  directions close to riskless. The repair puts variance back on the diagonal: Wholesale's
  subtracted vol of 7.81% comes out at 9.51%.

### The reference portfolios, before and after

On the forecast date, 2026-11-03, over the 499 index members with a forecast. Each industry
portfolio is its members, cap-weighted (Wholesale's 10 and Finance & Real Estate's 11 are the
members on that date).

| Portfolio | Factor, as est. | Factor, subtracted | Factor, repaired | Specific | Total, as est. | Total, subtracted | Total, repaired | Change |
|---|---|---|---|---|---|---|---|---|
| the index, cap-weighted | 11.37% | 11.22% | 11.23% | 3.22% | 11.82% | 11.67% | 11.69% | -1.1% |
| the index, equally weighted | 9.86% | 9.72% | 9.93% | 1.26% | 9.94% | 9.80% | 10.01% | +0.8% |
| smallest 85, equally weighted | 12.65% | 11.84% | 12.56% | 3.27% | 13.06% | 12.28% | 12.98% | -0.6% |
| largest 50, equally weighted | 12.22% | 12.06% | 12.08% | 3.64% | 12.75% | 12.59% | 12.62% | -1.0% |
| Construction (8) | 24.14% | 20.21% | 21.16% | 11.78% | 26.86% | 23.40% | 24.22% | -9.8% |
| Wholesale (10) | 14.10% | 8.38% | 10.74% | 7.93% | 16.18% | 11.54% | 13.35% | -17.5% |
| Finance & Real Estate (11) | 17.90% | 14.72% | 16.02% | 8.30% | 19.73% | 16.90% | 18.04% | -8.6% |

- **Broad portfolios** move by about 1%, and the repair moves them back toward the estimate: the
  equal-weighted index ends above it.
- **The size pair's own line** goes from 11.87% to 11.51% for the smallest 85, and from 0.62% to
  0.61% for the largest 50.

---

## Item 2: the thin industries

### Where each correlates most

The industries are counted over the regression universe:

- Construction 8, Finance & Real Estate 12, Wholesale 12;
- SIC37 15, Mining 16;
- SIC60, SIC62 and Services 20 each, SIC20 21;
- SIC63 25, SIC67 29;
- Transport & Utilities 32, Retail 34, SIC35 34;
- SIC36 36, SIC49 37, Manufacturing 39, SIC28 40, SIC38 44, SIC73 70.

**The rule** is the brief's words: the industry factor its own factor return correlates with
most, as Pearson's correlation of the daily factor returns over the 938 estimated periods.
Three other measures are reported beside it:

- the model's covariance (Newey-West, half-life 90);
- the corrected covariance;
- Pearson with each industry factor's beta to the cap-weighted market of them taken out. Every
  industry factor here carries the market, there being no market factor.

| Thin industry | Pearson (the rule) | The model's covariance | Corrected | Market taken out |
|---|---|---|---|---|
| Construction | **Retail 0.606**, SIC67 0.577 | SIC36 0.748, SIC67 0.543 | SIC36 0.936 | SIC67 0.241, Retail 0.228 |
| Wholesale | **Retail 0.709**, SIC38 0.700 | SIC38 0.540, Retail 0.507 | SIC38 0.749 | Retail 0.224 |
| Finance & Real Estate | **SIC62 0.761**, SIC73 0.732 | SIC62 0.644 | Services 0.780 | SIC63 0.456, SIC62 0.339 |

Where the measures disagree, the alternatives were run too.

### Each folded alone

`--merge-industry A=B` folds A's members into B once the scheme is built. Each run carries
`--sampling-correction`, whose factor returns, R-squared and specific risk are the uncorrected
model's; the uncorrected covariance is re-estimated from them. Against C:

| Fold | Mean R-squared | Folded factor's noise, eq. / wt. | Merged factor's noise, eq. / wt. | Members' specific risk, median | Its portfolio, total |
|---|---|---|---|---|---|
| Construction into **Retail** | -0.003011 | 38.2 / 48.6% | Retail 10.7 / 16.1% to 10.1 / 14.9% | 23.80% to 27.91% (+12.4%) | 26.86% to 22.95% |
| Construction into SIC36 | -0.004490 | | SIC36 4.5 / 5.3% to 4.3 / 5.1% | 23.80% to 28.92% (+25.3%) | 26.86% to 28.18% |
| Construction into SIC67 | -0.003286 | | SIC67 16.8 / 25.4% to 14.8 / 21.8% | 23.80% to 28.07% (+13.5%) | 26.86% to 23.12% |
| Wholesale into **Retail** | -0.001759 | 46.2 / 68.8% | Retail to 10.4 / 15.6% | 23.56% to 24.60% (+1.6%) | 16.18% to 16.95% |
| Wholesale into SIC38 | -0.002031 | | SIC38 10.6 / 15.9% to 10.5 / 15.9% | 23.56% to 24.77% (+4.9%) | 16.18% to 15.68% |
| Finance & Real Estate into **SIC62** | -0.002358 | 31.0 / 46.5% | SIC62 11.5 / 15.0% to 10.2 / 14.1% | 19.39% to 22.55% (+9.4%) | 19.73% to 19.00% |
| Finance & Real Estate into Services | -0.002363 | | Services 28.0 / 42.6% to 19.5 / 29.7% | 19.39% to 20.75% (+7.8%) | 19.73% to 17.20% |
| Finance & Real Estate into SIC63 | -0.002468 | | SIC63 14.5 / 22.7% to 12.6 / 19.5% | 19.39% to 21.49% (+5.2%) | 19.73% to 16.87% |

- **The yardstick.** A regressor of pure noise adds about 0.0016 to 0.0026 of mean R-squared
  (`docs/FACTOR_MODEL.md`, Size Nonlinearity). The three thin industries each lose about that
  when folded; Construction a little more.
- **The rule's target fits best in every case,** Finance & Real Estate's three within 0.0001.
- **Into SIC63** raises its members' specific risk least. They are mostly insurance brokers
  (AJG, AON, BRO, MRSH, WTW), and SIC63 is the insurers.
- **The specific risk of the target's own members** barely moves: Retail's 34 a median +0.3%
  and +0.0%, SIC62's 20 +3.0%.
- **The correction's repair**, if it were on, binds a little less after each fold: 8 negative
  eigenvalues, the most negative -0.0053 to -0.0070.

**Construction is two industries.**

- **The homebuilders** (DHI, LEN, NVR, PHM) fit Retail. Folded there, their specific risk rises
  from about 19% to 21-28%.
- **The contractors** (EME, FIX, PWR, J) lately move with SIC36, whence the model's
  recent-weighted choice. Their specific risk barely moves under any fold.
- Into SIC36 the homebuilders' specific risk rises to 28-33%.

### The floors

**The procedure.** Every industry below the floor is folded by the same rule, the smallest
first, ties by name. The model is re-estimated after each fold, so the next is chosen on the
factors as they then are. Floor 15's folds are the first three of floor 20's, and floor 20's the
first five of floor 25's, so one sequence of 8 runs measures all three (F1 to F8; F1 equals the
single Construction fold to the bit).

| Fold | Source | Into | Correlation | R-squared lost | Market taken out would pick |
|---|---|---|---|---|---|
| 1 | Construction (8) | Retail (34) | 0.606 | 0.003011 | SIC67 |
| 2 | Finance & Real Estate (12) | SIC62 (20) | 0.760 | 0.002360 | SIC63 |
| 3 | Wholesale (12) | Retail (42) | 0.714 | 0.001780 | Retail |
| **floor 15 holds** | | | | | |
| 4 | SIC37 (15) | SIC62 (32) | 0.700 | 0.005306 | SIC60 |
| 5 | Mining (16) | Manufacturing (39) | 0.770 | 0.003834 | Manufacturing |
| **floor 20 holds** | | | | | |
| 6 | SIC60 (20) | SIC62 (47) | 0.815 | 0.006257 | SIC62 |
| 7 | Services (20) | SIC73 (70) | 0.760 | 0.003662 | SIC73 |
| 8 | SIC20 (21) | Retail (54) | 0.637 | 0.004222 | SIC49 |
| **floor 25 holds** | | | | | |

| | C | floor 15 | floor 20 | floor 25 |
|---|---|---|---|---|
| industries | 20 | 17 | 15 | 12 |
| mean R-squared | 0.293203 | 0.286053 | 0.276914 | 0.262773 |
| largest industry noise share, eq. / wt. | 46.2 / 68.8% | 33.4 / 55.1% | 33.9 / 56.0% | 17.2 / 25.8% |
| Size Nonlinearity's noise share, wt. | 67.4% | 66.8% | 62.3% | 62.7% |
| the correction's negative eigenvalues (most negative) | 9 (-0.0070) | 6 (-0.0031) | 5 (-0.0031) | 2 (-0.0011) |
| specific risk median | 24.76% | 24.78% | 24.98% | 24.99% |
| the folded members' specific risk, median change | | +3.3% (31) | +3.2% (59) | +7.1% (118) |
| the index, as est. / corrected | 11.82 / 11.69% | 11.82 / 11.68% | 11.83 / 11.69% | 11.84 / 11.70% |
| the equal-weighted index | 9.94 / 10.01% | 9.92 / 9.91% | 9.93 / 9.84% | 9.93 / 9.82% |
| smallest 85 | 13.06 / 12.98% | 13.07 / 12.83% | 13.05 / 12.69% | 13.14 / 12.71% |
| largest 50 | 12.75 / 12.62% | 12.71 / 12.56% | 12.70 / 12.55% | 12.69 / 12.54% |
| Construction (8) | 26.86 / 24.22% | 22.21 / 21.70% | 22.20 / 21.63% | 21.19 / 20.64% |
| Wholesale (10) | 16.18 / 13.35% | 16.53 / 16.05% | 16.48 / 15.90% | 15.64 / 15.08% |
| Finance & Real Estate (11) | 19.73 / 18.04% | 19.02 / 18.24% | 17.59 / 17.04% | 18.04 / 17.63% |

**Past 15 the folds take signal.**

- SIC37 costs 0.0053 and SIC60 0.0063, two to four noise regressors.
- The rule's homes get odd: transportation equipment into the brokers, on a correlation the
  market makes.
- **A member floor is not a noise floor.** At floor 15 the noisiest industry is SIC20, with 21
  members and 55.1% noise; at floor 20, still SIC20 (56.0%), then Services (43.2%).

---

## Item 3: Size Nonlinearity

**The correction's effect on it.**

- **Its vol:** 2.32% as estimated, 1.32% after the subtraction, 1.71% after the repair. Two of
  the clipped directions load on it and on Size.
- **Its noise share:** 46.6% equal-weighted, 67.4% weighted.
- **Its correlation with Size:** -0.132 as estimated, -0.243 subtracted, -0.169 repaired.
- **Folding industries** leaves its noise share at 62% to 67%.

**The model without it** (`--size-nonlinearity none`, run SN0, with the correction so both
covariances are measured):

- **Mean R-squared** 0.293203 to 0.289600: 0.0036, more than a noise regressor's 0.0016 to 0.0026.
- **Size:** its vol (3.72%) and noise share (10.7%) are unchanged; they are orthogonal by
  construction.
- **The correction's repair:** 8 negative eigenvalues, the most negative -0.0071.
- **Specific risk:** the smallest 85's median rises from 27.42% to 27.69%, the largest 50's from
  21.38% to 21.58%; over every name the median goes from 24.76% to 24.79%.

| Portfolio | With, as est. | With, corrected | Without, as est. | Without, corrected |
|---|---|---|---|---|
| the index, cap-weighted | 11.82% | 11.69% | 11.78% | 11.65% |
| the index, equally weighted | 9.94% | 10.01% | 9.94% | 9.97% |
| smallest 85, equally weighted | 13.06% | 12.98% | 12.99% | 13.03% |
| largest 50, equally weighted | 12.75% | 12.62% | 12.72% | 12.59% |
| Construction (8) | 26.86% | 24.22% | 26.92% | 24.21% |
| Wholesale (10) | 16.18% | 13.35% | 16.28% | 13.39% |
| Finance & Real Estate (11) | 19.73% | 18.04% | 19.76% | 18.02% |

**No portfolio moves by more than 0.07 of a point.** The smallest 85 carry a mean Size
Nonlinearity exposure of -1.808, and the size pair's own line is 11.51% with it (corrected),
11.48% for Size alone.

---

## What I would choose, and why

1. **The correction: not as the default, in this form.**
   - **Its diagonal is right.** It is unbiased factor by factor, and the noise it removes is
     larger than the run reports.
   - **The matrix it leaves is not safe.** The full subtraction leaves 7 to 10 negative
     eigenvalues by the estimator's own sampling error, whatever the truth, and clipping makes
     those directions close to riskless. A portfolio built in one -- which an optimiser will
     find -- would get no factor risk.
   - **It buys little where the model is used most:** the broad portfolios move by about 1%.
   - **If a correction is wanted, it should come with the eigenvalues' own correction.** The
     smallest eigenvalues of a sample covariance are biased low -- the reason the subtraction
     goes through zero -- and correcting that, Barra's eigenfactor adjustment, is the
     established answer: a piece of its own.
2. **The weighted noise share as the standing diagnostic.** It is the consistent ratio and the
   one any correction would subtract; the equal-weighted one understates the current noise by
   about a third. That changes the run's report, so it is a decision below.
3. **Floor 15: the three thin industries folded by the rule.** Construction and Wholesale go
   into Retail, Finance & Real Estate into SIC62.
   - **The cost is small.** Each costs about a noise regressor's worth of R-squared, 0.0072 for
     the three, and they are 47% to 69% noise.
   - **It removes the worst of the problem.** It takes out the three noisiest industry factors
     and halves the correction's worst negative eigenvalue, should a correction come.
   - **The rule's targets fit best** of every alternative measured.
   - **Not 20 or 25.** Those floors fold industries that carry signal and land them by
     correlations the market makes.
   - **The price.** The folded members' specific risk rises -- the homebuilders' most, DHI from
     20% to 28% -- since their industry effect is now specific risk; a factor half noise was
     carrying it badly anyway.
   - **Whether the folded forecasts are better** is for the out-of-sample bias statistics, next
     in Jeff's sequence; they should judge this before it stands.
4. **Size Nonlinearity kept, as ruled (04 rulings 1-2).** It adds more R-squared than noise
   would, and its noise does no harm at the portfolio level: dropping it moves no reference
   portfolio by more than 0.07 of a point. The measurement gives no reason to reopen the ruling.

---

## Key decisions

1. **"Weighted as the covariance is" taken exactly:** each period's weight is the estimator's
   own estimate of a one-period indicator, Newey-West's lag terms included. The lag-0 weights
   alone would subtract 3.9% too much.
2. **The full matrix subtracted, not its diagonal alone:** the brief's "mean sampling
   covariance". The classical `sigma^2 (X'WX)^-1`, as `noise_share` uses; a
   heteroskedasticity-robust sandwich was not measured (below).
3. **The repair is `nearest_psd` at the configured floor,** the model's covariance PSD and the
   repair recorded. A period without a sampling covariance would get the others' weighted mean
   (none did).
4. **The correction refuses the two-half-life and shrunk estimators,** which are not quadratic
   forms, rather than guess their weighting.
5. **Every measured run carries the correction,** its uncorrected covariance re-estimated from
   its factor returns. On C against the defaults' run that is equal to the bit, so each run gives
   both.
6. **The correlation rule is Pearson over the estimated periods' daily factor returns.** The
   three other measures are reported beside it, and their targets run where they differ.
7. **The floors are folded smallest first, re-estimating after each fold;** one sequence serves
   all three floors.
8. **The industry portfolios are cap-weighted over the forecast date's members.**
9. **`merge_industries` is in the library,** recording its folds in `folded` apart from the
   grouping's `merged`, with a chain resolved and a cycle refused.
10. **Under the correction, the noise-share qualification quotes the weighted share,** the one
    subtracted. My first version quoted the equal-weighted one, 47% for Size Nonlinearity where
    68.8% for Wholesale was subtracted; corrected, and re-run.
11. **`listed_spans` counts held member-days on the trading calendar.** My first version walked
    calendar days, harmless for SBNY's none and FRCB's one, wrong for a run over a weekend; the
    rebuilt panel is identical.
12. **The rulings are the build's defaults, as ruled,** and the run's default panel moved to
    `_listed`. "Change no default" is read as the brief's three items.
13. **The docs brought in line, though the brief listed none:** they said the bias is never
    subtracted, and listed seven isolated errors.

## Decisions left for Jeff

1. **The sampling-error correction.**
   - **Leave it an option, off** (my choice).
   - **Or make it the default as built,** clipping and all.
   - **Or commission the correction with an eigenvalue adjustment.**
2. **The noise-share diagnostic.** Report the weighted share in the run's factor table, beside
   or instead of the equal-weighted one. It needs each period's sampling matrix whether or not
   the correction is on.
3. **The thin industries.**
   - **Floor 15 by the Pearson rule** (my choice): Construction and Wholesale into Retail,
     Finance & Real Estate into SIC62.
   - **No folds.**
   - **A higher floor.**
   - **Another rule** -- the market-removed measure, which would send Finance & Real Estate to
     SIC63 at 0.0001 more R-squared lost and less specific risk added.
   - Construction's two halves are a question for any of them.
4. **Size Nonlinearity:** keep, as ruled (my choice), or leave out.
5. **FRCB's 2023-05-04 return.** The span now ends on its last member-day, so the lag's
   reading of the next day's return goes, one security-period.
   - **Keep that** (as built).
   - **Or run a span one trading day past its last member-day,** so the regression keeps it.

---

## Tests

**The suite:** 2,311 passed, 1 skipped, 19 warnings, in 72.70 s, with
`--ignore=tests/test_market_data.py`. midoffice alone: 860 passed, 1 skipped. Before this piece:
2,295 and 844.

**New: 16.**

`tests/midoffice/test_securities.py`, 2:

- `test_a_sample_its_own_weighted_count_confirms_is_spared_and_still_reported` -- PCG's case
  spared within the tolerance, one its weighted count contradicts removed, one with none
  removed; all three listed.
- `test_a_span_off_an_exchange_ends_but_for_the_days_the_index_still_held_it`:
  - a span ending at its last listed sample and running on through the trading days of its
    membership, weekends not counted;
  - a span wholly off an exchange gone;
  - a listed span, and one with no records, kept as they are.

`tests/midoffice/test_crosssectional.py`, 11:

- `test_the_sampling_weights_are_the_diagonal_of_the_estimators_own_kernel` -- four estimators,
  against the estimate of the identity, to 1e-15.
- `test_equal_weights_carry_each_periods_noise_as_its_mean`.
- `test_newey_west_lags_carry_a_little_less_of_the_noise_than_their_weights_alone`.
- `test_an_estimator_that_is_no_quadratic_form_is_refused_by_the_correction` -- the
  two-half-life and shrunk estimators.
- `test_a_period_without_a_sampling_covariance_is_given_the_others_weighted_mean`.
- `test_the_correction_is_off_by_default_and_the_covariance_is_as_estimated`.
- `test_the_correction_changes_the_factor_covariance_and_nothing_else` -- factor returns,
  variances, specific risk identical; under equal weights the subtraction is exactly what
  `noise_share` measures.
- `test_the_correction_removes_the_bias_noise_share_measures` -- 12 replications.
- `test_the_correction_weights_each_periods_noise_as_the_covariance_does` -- noise growing late
  in the sample, an exponential covariance subtracting more of it.
- `test_a_factor_that_is_all_noise_is_repaired_where_the_subtraction_leaves_it_negative` -- 3
  of its 10 seeds land below zero, and each is repaired, PSD and named.
- `test_under_the_correction_the_loud_noise_qualification_quotes_the_share_subtracted`.

`tests/midoffice/test_industries.py`, 3:

- `test_an_industry_folded_into_another_takes_its_members_with_it`;
- `test_a_chain_of_folds_resolves_to_its_end_and_a_cycle_is_refused` -- one call and two
  equal;
- `test_a_fold_names_only_industries_that_are_factors` -- a grouped-away major group, an absent
  industry, a fold into itself.

**Hygiene:** every changed Python file parses, is ASCII and keeps its line endings, counted in
bytes (`hygiene_final.txt`). `test_industries.py` is LF in the working copy, as it was. ruff
over the changed files finds two things, each at HEAD too (checked) and left:

- `run_sp500_model.py`'s unused `HorizonCovariance`;
- `crosssectional.py`'s unused `Mapping`.

`test_crosssectional.py`'s import-order finding is gone: the covariance import it flagged is
rewritten in order.

**The final code, re-run** (`final_checks.txt`):

- **the defaults' run** equals midoffice09's defaults' run (captured before any midoffice10 code):
  its report line for line, every array;
- **the correction's run** equals C in every field;
- **midoffice09's panel** rebuilt by `--spare none --listed any`, every array exact, its report
  equal but for the fields this piece adds;
- **the default build** rebuilt into scratch equals `data/`'s `_listed`, arrays and report.

---

## Failures and debugging trail

- **A heredoc patch with `\\n` escapes failed its assertion.** Written as a patch script in a
  file, or done with the Edit tool, instead.
- **ruff F841 and F541** in the build patch: an unused name, an f-string without a field. Fixed.
- **A fix-up script missed its match,** and the patch it meant to correct ran anyway: the
  `dataclasses.replace` import was missing and an import out of order. Caught by ruff (F821)
  before any run, and fixed directly.
- **`item2.py` failed to find its pickles:** run from inside the scratch directory, its relative
  paths missed. The harness's shell had drifted; every later call cd's to the repo by its
  absolute path.
- **A regex in a patch script carried a doubled escape,** and its assertion failed; done with
  Edit instead.
- **The floors' report parser read "run F3" as "run".** Fixed before any output was used.
- **The state file's own note on heredocs was mangled by a heredoc's escapes,** its backslashes
  doubled. Fixed with the Edit tool.
- **midoffice09's panel rebuilt by its flags matched in every array, but its report did not.**
  Each repair now carries `weighted` and `spared` (None and False without `--spare`), and the
  report has `delistings` and the two new rules. Equal without them, as `final_checks.txt` shows.

## Problems found, not fixed

1. **The repair binds whatever the truth** (item 1): a correction worth defaulting needs an
   eigenvalue adjustment.
2. **The sampling covariance is the classical one,** which assumes each name's residual variance
   is proportional to the inverse of its regression weight. A heteroskedasticity-robust sandwich
   would say whether it over- or under-states the noise of an 8-member industry. Not measured.
3. **The run's noise-share column understates the current noise by about a third** (a decision
   above).
4. **A member floor is not a noise floor:** SIC20, with 21 members, is 55% noise.
5. **FRCB's 2023-05-04 return** leaves a regression through the lag (a decision above).
6. **Two ruff findings predate this piece** (above).
7. **`tests/test_market_data.py` looks for `../core_repo/src`,** not on this machine.

## Scratch

`claude/scratch/midoffice10/` (gitignored). The scripts:

- `hygiene.py` (from midoffice09), and the patch scripts: `patch_rulings.py`,
  `patch_build.py`, `patch_correction.py`, `patch_correction2.py`, `patch_flags.py`,
  `patch_tests.py`, `patch_tests2.py`, `patch_tests_securities.py`, `patch_qualification.py`,
  `patch_docs.py`, `patch_docs2.py`;
- `capture.py`, the rulings' runs (M9, Q1, Q2 in `capture/`);
- `capture2.py`, which runs the model in-process and keeps the model, the final scheme and the
  matrix the repair is handed. Its `capture2/` holds each run's report and pickle:
  - Q2b, Q2c (the defaults), C, C2 (the correction);
  - M_CON, M_WHO, M_FRE (the rule's folds), A_CON, A_WHO, A_FRE (the model's covariance's),
    X_CON, X_FRE (the market-removed measure's);
  - F1 to F8 (the floors), SN0 (no Size Nonlinearity);
- `otc_scan.py`, `rulings_changed.py`, `compare_rulings.py`, `item1.py`, `item2.py`,
  `floors.py`, `sampling_profile.py`, `final_checks.py`.

Their outputs:

- the builds: `build_p0.txt` to `build_p4.txt`, `build_data.txt`;
- the measures: `otc_scan.txt`, `rulings_changed.txt`, `compare_rulings.txt`, `item1.txt`,
  `item2.txt`, `floors.txt`, `sampling_profile.txt`;
- the checks: `final_checks.txt`, `pytest_final.txt`, `ruff_final.txt`, `hygiene_final.txt`;
- the panels `p0` to `p4`.

The key ones are copied into the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

## Open items for Chat

- **The five decisions above:**
  - the correction;
  - the diagnostic;
  - the thin industries;
  - Size Nonlinearity;
  - FRCB's one return.

  The thin industries' choice is best confirmed by the out-of-sample bias statistics, next in the
  sequence.
- **For the bias statistics:** the history weighting (09's third decision, left equal) and the
  folds both bear on them.

---

## Appendix: outputs, verbatim

The key outputs the body rests on, from `claude/scratch/midoffice10/`, in full unless said
otherwise.

### build_data.txt -- the default build into data/ (the _listed panel): the off-exchange cut, the isolated errors, the guards (lines 9-12, 31-38, 228-235)

```text
Off an exchange: a span ends where its records place the security off one, but for any day the index still held it -- 2 span(s) ended
  SBNY   BBG000M6TR37 2023-03-28..2026-11-03: off an exchange from 2023-03-28 (otc, OTC Link); last on one none in the span; the span goes; 864 priced day(s) removed, 2023-03-28..2026-11-03
  FRCB   BBG0019LSQ49 2023-05-03..2023-11-17: off an exchange from 2023-05-03 (otc, OTC Link); last on one none in the span; kept to 2023-05-03, for the 1 member-day(s) the index still held it, 2023-05-03; 138 priced day(s) removed, 2023-05-04..2023-11-17

[...]
Isolated errors: a sample whose two samples before and two after agree with each other within 10%, and which differs from each of them beyond it, no split between, is not carried, unless its own record's weighted count confirms it -- 7 found, 6 removed, 1 spared
  removed DGX    DGX    2025-04-09       11,610,000; around it 2024-10-07 111,320,000, 2025-01-07 111,620,000, 2025-07-11 111,635,472, 2025-10-09 111,823,441 -- 0.104 of their median; its weighted count 110,978,069, 9.559 of it
  removed EG     EG     2025-04-09       47,790,000; around it 2024-10-07 43,270,000, 2025-01-07 42,980,000, 2025-07-11 42,534,728, 2025-10-09 41,943,291 -- 1.118 of their median; its weighted count 42,934,097, 0.898 of it
  removed NWS    NWS    2024-07-09      162,170,000; around it 2024-01-05 191,390,000, 2024-04-08 191,100,000, 2024-10-07 190,080,000, 2025-01-07 189,770,000 -- 0.851 of their median; its weighted count 569,013,132, 3.509 of it
  SPARED  PCG    PCG    2026-01-09    2,197,910,425; around it 2025-07-11 2,675,428,603, 2025-10-09 2,675,580,992, 2026-04-13 2,675,711,544, 2026-07-14 2,679,968,318 -- 0.821 of their median; its weighted count 2,197,910,425, 1.000 of it
  removed RDDT   RDDT   2025-12-22       51,670,930; around it 2025-06-24 129,546,488, 2025-09-23 134,585,853, 2026-03-25 139,649,508, 2026-06-25 141,965,642 -- 0.377 of their median; its weighted count 189,495,765, 3.667 of it
  removed SBUX   SBUX   2026-07-14      139,700,000; around it 2026-01-09 1,137,100,000, 2026-04-13 1,139,300,000, 2026-10-12 1,140,000,000, 2026-11-03 1,140,000,000 -- 0.123 of their median; its weighted count 1,139,700,000, 8.158 of it
  removed XYZ    XYZ    2026-01-09       59,998,000; around it 2025-07-11 555,001,000, 2025-10-09 549,574,000, 2026-04-13 539,103,000, 2026-07-14 535,195,000 -- 0.110 of their median; its weighted count 607,665,000, 10.128 of it
[...]
Guards
  ok  each (ticker, date) price feeds one column: 684,090 checked, 0 failure(s)
  ok  no column takes two tickers on one date: its spans do not overlap: 627 checked, 0 failure(s)
  ok  every member-day is a column's, on a day inside its spans or one the table maps outside them: 610,842 checked, 0 failure(s)
  ok  every bar under a span's ticker inside the span is its column's, and only those: 684,087 checked, 0 failure(s)
  ok  every member-day of the walk through the renames is a column's member-day: 610,842 checked, 0 failure(s)
  ok  the panel's member-days are the walk's, no more: 610,842 checked, 0 failure(s)
  ok  no day on or after a split carries a pre-split count, and the split's day carries the count before it times the ratio (25,260 of the days checked member-days): 25,846 checked, 0 failure(s)
```

### otc_scan.txt -- every share-count sample's record, its market and primary exchange

```text
SBNY   BBG000M6TR37 2023-03-28..2026-11-03 member-days 300
   2023-03-28 otc OTC Link
   2023-04-05 otc OTC Link
   2023-07-07 otc OTC Link
   2023-10-05 otc OTC Link
   2024-01-05 otc OTC Link
   2024-04-08 otc OTC Link
   2024-07-09 otc OTC Link
   2024-10-07 otc OTC Link
   2025-01-07 otc OTC Link
   2025-04-09 otc OTC Link
   2025-07-11 otc OTC Link
   2025-10-09 otc OTC Link
   2026-01-09 otc OTC Link
   2026-04-13 otc OTC Link
   2026-07-14 otc OTC Link
   2026-10-12 otc OTC Link
   2026-11-03 otc OTC Link
FRCB   BBG0019LSQ49 2023-05-03..2023-11-17 member-days 1
   2023-05-03 otc OTC Link
   2023-07-07 otc OTC Link
   2023-10-05 otc OTC Link
   2023-11-17 otc OTC Link
markets over every sample: {'stocks': 11929, 'otc': 21}
primary exchanges over every sample: {'XNYS': 8143, 'XNAS': 3765, 'OTC Link': 21, 'BATS': 21}
```

### rulings_changed.txt -- the panel cells each ruling changes, each panel against the one before

```text
ruling 1, against midoffice09:
  shares_outstanding       63 cells change, 63 of them member-days; in PCG 2026-01-09..2026-04-10
  membership identical: True
ruling 2, against ruling 1:
  prices                1,002 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  total_return_prices   1,002 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  unadjusted_prices     1,002 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  volumes               1,002 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  adjusted_volumes      1,002 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  shares_outstanding    1,043 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  highs                 1,002 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  lows                  1,002 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  tickers               1,043 cells change, 0 of them member-days; in FRCB 2023-05-04..2023-11-17, SBNY 2023-03-28..2026-11-03
  membership identical: True
```

### compare_rulings.txt -- the model on each ruling's panel against the one before

```text
====================================================================================================
Step ruling 1: PCG spared, its own weighted count confirming its sample: M9 -> Q1
  M9: data/external/polygon/sp500_securities_2022-01-03_2026-11-03_repaired.npz --industries --membership
  Q1: claude/scratch/midoffice10/p1/sp500_securities_2022-01-03_2026-11-03_r1.npz --industries --membership
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    M9: min 499 median 502 max 503 first 502 last 499
    Q1: min 499 median 502 max 503 first 502 last 499
    Q1 less M9, per period: min +0 median +0 max +0
  mean cross-sectional R-squared: M9 0.293204, Q1 0.293204, difference -0.000000
  factor                   vol M9   vol Q1   mean M9   mean Q1   d mean  noise M9  noise Q1
  Construction             19.42%   19.42%   +24.76%   +24.76%   +0.00%     38.2%     38.2%
  Finance & Real Estate    15.34%   15.34%   +18.11%   +18.11%   +0.00%     31.0%     31.0%
  Manufacturing            10.20%   10.20%   +20.88%   +20.88%   +0.00%     23.2%     23.2%
  Mining                   18.61%   18.60%   +14.94%   +14.94%   +0.00%     18.5%     18.5%
  Retail                   13.11%   13.11%   +14.63%   +14.63%   +0.00%     10.7%     10.7%
  SIC20                    11.96%   11.96%   +11.03%   +11.03%   +0.00%     33.6%     33.6%
  SIC28                    13.14%   13.14%   +16.64%   +16.64%   +0.00%     10.2%     10.2%
  SIC35                    18.81%   18.81%   +29.69%   +29.69%   +0.00%      5.2%      5.2%
  SIC36                    21.13%   21.13%   +32.55%   +32.55%   -0.00%      4.5%      4.5%
  SIC37                    14.22%   14.22%   +15.42%   +15.42%   +0.00%     19.1%     19.1%
  SIC38                    14.69%   14.70%   +15.82%   +15.82%   +0.00%     10.6%     10.6%
  SIC49                    15.80%   15.80%   +24.02%   +24.01%   -0.01%     11.3%     11.3%
  SIC60                    21.85%   21.85%   +21.97%   +21.97%   +0.00%      7.4%      7.4%
  SIC62                    18.73%   18.73%   +18.62%   +18.62%   +0.00%     11.5%     11.5%
  SIC63                    14.57%   14.57%   +19.16%   +19.17%   +0.00%     14.5%     14.5%
  SIC67                    15.01%   15.01%   +18.00%   +18.00%   +0.00%     16.8%     16.8%
  SIC73                    18.18%   18.18%   +15.21%   +15.21%   +0.00%      2.9%      2.9%
  Services                 12.72%   12.72%   +16.11%   +16.11%   +0.00%     28.0%     28.0%
  Transport & Utilities    13.43%   13.43%   +15.46%   +15.46%   +0.00%     16.5%     16.5%
  Wholesale                13.98%   13.98%   +20.69%   +20.69%   +0.00%     46.2%     46.2%
  Volatility                9.38%    9.37%    +3.09%    +3.09%   -0.00%      3.5%      3.5%
  Momentum                  7.22%    7.22%    +4.23%    +4.23%   +0.00%      3.6%      3.6%
  Size                      3.72%    3.72%    +3.13%    +3.13%   +0.00%      7.7%      7.7%
  Size Nonlinearity         2.32%    2.32%    -0.53%    -0.52%   +0.00%     46.6%     46.6%
  Trading Activity          6.25%    6.25%    +2.48%    +2.48%   +0.00%      7.5%      7.5%
  residual spread (1.4826 x MAD; sd): M9 0.1333 (0.3404); Q1 0.1333 (0.3403)
    largest shares of the sd's square, M9: FRCB 67.6%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of the sd's square, Q1: FRCB 67.6%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of V^2, M9: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    largest shares of V^2, Q1: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: M9 p10 15.58% median 24.79% p90 40.56% (551); Q1 p10 15.58% median 24.79% p90 40.56% (551)
  specific risk, the 551 in both: M9 p10 15.58% median 24.79% p90 40.56% (551); Q1 p10 15.58% median 24.79% p90 40.56% (551)
    of those in both, the largest rises: PCG +0.02%, CF +0.00%, EIX +0.00%, OGN +0.00%, NWS +0.00%, PENN +0.00%, HOOD +0.00%, DOW +0.00%
    the largest falls: HBAN -0.00%, FBIN -0.00%, WM -0.00%, AKAM -0.00%, ENPH -0.00%, MU -0.00%, SBNY -0.00%, SNDK -0.01%
  fallback forecasts: M9 0: 
                      Q1 0: 
    in Q1 not M9: none; in M9 not Q1: none
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    M9 -3.137 -1.808 risk 11.87%; Q1 -3.137 -1.808 risk 11.87%
    largest 50, equally weighted     M9 +0.167 +0.052 risk 0.62%; Q1 +0.167 +0.052 risk 0.62%
    the index, cap-weighted          M9 +0.001 -0.001 risk 0.01%; Q1 +0.001 -0.001 risk 0.01%
  regression universe: M9 567, Q1 567; in M9 not Q1: none
    in Q1 not M9 (0): 
====================================================================================================
Step ruling 2: spans end where they leave an exchange: Q1 -> Q2
  Q1: claude/scratch/midoffice10/p1/sp500_securities_2022-01-03_2026-11-03_r1.npz --industries --membership
  Q2: --industries --membership
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    Q1: min 499 median 502 max 503 first 502 last 499
    Q2: min 499 median 502 max 503 first 502 last 499
    Q2 less Q1, per period: min -1 median +0 max +0
  mean cross-sectional R-squared: Q1 0.293204, Q2 0.293203, difference -0.000000
  factor                   vol Q1   vol Q2   mean Q1   mean Q2   d mean  noise Q1  noise Q2
  Construction             19.42%   19.42%   +24.76%   +24.76%   +0.00%     38.2%     38.2%
  Finance & Real Estate    15.34%   15.34%   +18.11%   +18.11%   -0.00%     31.0%     31.0%
  Manufacturing            10.20%   10.20%   +20.88%   +20.88%   -0.00%     23.2%     23.2%
  Mining                   18.60%   18.60%   +14.94%   +14.94%   -0.00%     18.5%     18.5%
  Retail                   13.11%   13.11%   +14.63%   +14.63%   -0.00%     10.7%     10.7%
  SIC20                    11.96%   11.96%   +11.03%   +11.03%   +0.00%     33.6%     33.6%
  SIC28                    13.14%   13.14%   +16.64%   +16.64%   -0.00%     10.2%     10.2%
  SIC35                    18.81%   18.81%   +29.69%   +29.69%   -0.00%      5.2%      5.2%
  SIC36                    21.13%   21.13%   +32.55%   +32.55%   -0.00%      4.5%      4.5%
  SIC37                    14.22%   14.22%   +15.42%   +15.42%   -0.00%     19.1%     19.1%
  SIC38                    14.70%   14.70%   +15.82%   +15.82%   -0.00%     10.6%     10.6%
  SIC49                    15.80%   15.80%   +24.01%   +24.01%   -0.00%     11.3%     11.3%
  SIC60                    21.85%   21.85%   +21.97%   +21.97%   -0.00%      7.4%      7.4%
  SIC62                    18.73%   18.73%   +18.62%   +18.62%   -0.00%     11.5%     11.5%
  SIC63                    14.57%   14.57%   +19.17%   +19.17%   -0.00%     14.5%     14.5%
  SIC67                    15.01%   15.01%   +18.00%   +18.00%   -0.00%     16.8%     16.8%
  SIC73                    18.18%   18.18%   +15.21%   +15.21%   -0.00%      2.9%      2.9%
  Services                 12.72%   12.72%   +16.11%   +16.11%   -0.00%     28.0%     28.0%
  Transport & Utilities    13.43%   13.43%   +15.46%   +15.46%   -0.00%     16.5%     16.5%
  Wholesale                13.98%   13.98%   +20.69%   +20.69%   -0.00%     46.2%     46.2%
  Volatility                9.37%    9.37%    +3.09%    +3.09%   +0.00%      3.5%      3.5%
  Momentum                  7.22%    7.22%    +4.23%    +4.23%   -0.00%      3.6%      3.6%
  Size                      3.72%    3.72%    +3.13%    +3.13%   -0.00%      7.7%      7.7%
  Size Nonlinearity         2.32%    2.32%    -0.52%    -0.53%   -0.00%     46.6%     46.6%
  Trading Activity          6.25%    6.25%    +2.48%    +2.48%   +0.00%      7.5%      7.5%
  residual spread (1.4826 x MAD; sd): Q1 0.1333 (0.3403); Q2 0.1333 (0.3446)
    largest shares of the sd's square, Q1: FRCB 67.6%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of the sd's square, Q2: FRCB 68.4%, SIVB 3.5%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of V^2, Q1: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    largest shares of V^2, Q2: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: Q1 p10 15.58% median 24.79% p90 40.56% (551); Q2 p10 15.58% median 24.76% p90 40.49% (550)
  specific risk, the 550 in both: Q1 p10 15.58% median 24.76% p90 40.49% (550); Q2 p10 15.58% median 24.76% p90 40.49% (550)
    of those in both, the largest rises: ZION +0.00%, PENN +0.00%, MOS +0.00%, NRG +0.00%, JKHY +0.00%, Q +0.00%, OGN +0.00%, LNC +0.00%
    the largest falls: PNW -0.00%, HST -0.00%, TWLO -0.00%, MRNA -0.00%, APP -0.00%, VNO -0.00%, DDOG -0.00%, CVNA -0.00%
  fallback forecasts: Q1 0: 
                      Q2 0: 
    in Q2 not Q1: none; in Q1 not Q2: none
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    Q1 -3.137 -1.808 risk 11.87%; Q2 -3.137 -1.808 risk 11.87%
    largest 50, equally weighted     Q1 +0.167 +0.052 risk 0.62%; Q2 +0.167 +0.052 risk 0.62%
    the index, cap-weighted          Q1 +0.001 -0.001 risk 0.01%; Q2 +0.001 -0.001 risk 0.01%
  regression universe: Q1 567, Q2 567; in Q1 not Q2: none
    in Q2 not Q1 (0): 
====================================================================================================
The histories ruling 2 shortens, and the forecasts that go
  SBNY   specific returns 851 -> None; forecast 395.72% -> none
  forecasts: 551 -> 550; in Q1 not Q2: ['SBNY']
  specific returns outside the regressions: 55,830 -> 54,864 (966 removed)
  PCG: forecast 23.3340% -> 23.3506%
```

### item1.txt -- item 1: the sampling-error correction

```text
Item 1: the sampling-error correction on the defaults' model (Q2b, C)
  ['--industries', '--membership', '--sampling-correction']
  Newey-West, 10 lags, half-life 90, over 938 periods carries 0.9625 of a period's noise; 0 period(s) filled
  the repair: binds: 9 negative eigenvalue(s) of 25, most negative -0.0070 (annual variance), floored at 2.14e-09

  factor                   vol as est. noise, eq. noise, wt. subtracted  repaired  change repair adds
  Construction                  19.42%      38.2%      48.6%     13.91%    15.05%  -22.5%     0.00329
  Finance & Real Estate         15.34%      31.0%      46.5%     11.22%    12.91%  -15.9%     0.00407
  Manufacturing                 10.20%      23.2%      33.5%      8.32%     8.82%  -13.5%     0.00086
  Mining                        18.60%      18.5%      27.2%     15.88%    15.98%  -14.1%     0.00031
  Retail                        13.11%      10.7%      16.1%     12.01%    12.28%   -6.3%     0.00066
  SIC20                         11.96%      33.6%      55.5%      7.98%     9.35%  -21.8%     0.00238
  SIC28                         13.14%      10.2%      15.7%     12.06%    12.14%   -7.6%     0.00020
  SIC35                         18.81%       5.2%       7.0%     18.14%    18.25%   -3.0%     0.00038
  SIC36                         21.13%       4.5%       5.3%     20.56%    20.77%   -1.7%     0.00089
  SIC37                         14.22%      19.1%      27.3%     12.13%    12.18%  -14.4%     0.00012
  SIC38                         14.70%      10.6%      15.9%     13.48%    13.70%   -6.8%     0.00060
  SIC49                         15.80%      11.3%      16.9%     14.40%    14.51%   -8.1%     0.00033
  SIC60                         21.85%       7.4%      10.5%     20.67%    20.77%   -4.9%     0.00045
  SIC62                         18.73%      11.5%      15.0%     17.27%    17.49%   -6.6%     0.00077
  SIC63                         14.57%      14.5%      22.7%     12.81%    12.95%  -11.1%     0.00036
  SIC67                         15.01%      16.8%      25.4%     12.97%    13.29%  -11.5%     0.00085
  SIC73                         18.18%       2.9%       4.5%     17.77%    17.85%   -1.8%     0.00029
  Services                      12.72%      28.0%      42.6%      9.64%    10.34%  -18.8%     0.00139
  Transport & Utilities         13.43%      16.5%      25.1%     11.62%    11.80%  -12.1%     0.00042
  Wholesale                     13.98%      46.2%      68.8%      7.81%     9.51%  -32.0%     0.00295
  Volatility                     9.37%       3.5%       5.7%      9.11%     9.28%   -1.1%     0.00031
  Momentum                       7.22%       3.6%       5.1%      7.03%     7.11%   -1.5%     0.00011
  Size                           3.72%       7.7%      10.7%      3.52%     3.70%   -0.5%     0.00013
  Size Nonlinearity              2.32%      46.6%      67.4%      1.32%     1.71%  -26.2%     0.00012
  Trading Activity               6.25%       7.5%      10.5%      5.92%     6.07%   -3.0%     0.00018
  (vols annualised; 'repair adds' is the variance the repair puts back on the diagonal)

  Where the repair binds: the subtracted matrix's negative eigenvalues, each direction's largest loadings, and its observed and noise variance
    -0.0070  Finance & Real Estate +0.67, Construction +0.57, SIC62 -0.20, SIC20 -0.19   observed 0.0048, noise 0.0118
    -0.0046  Wholesale +0.67, Finance & Real Estate -0.36, Construction +0.31, SIC62 +0.27   observed 0.0056, noise 0.0102
    -0.0035  SIC20 +0.52, Wholesale +0.47, SIC36 +0.31, SIC49 -0.24   observed 0.0041, noise 0.0075
    -0.0027  SIC20 +0.48, Construction +0.34, SIC36 -0.33, Services -0.33   observed 0.0037, noise 0.0064
    -0.0024  Services -0.58, SIC67 +0.55, SIC20 -0.31, SIC38 +0.21   observed 0.0033, noise 0.0057
    -0.0011  Manufacturing +0.68, Transport & Utilities -0.31, SIC60 +0.30, Mining -0.27   observed 0.0028, noise 0.0039
    -0.0010  SIC60 +0.42, Manufacturing -0.40, SIC62 -0.35, Transport & Utilities -0.32   observed 0.0029, noise 0.0039
    -0.0001  Size Nonlinearity +0.69, Size +0.50, SIC67 -0.25, Services -0.23   observed 0.0012, noise 0.0013
    -0.0000  Size Nonlinearity -0.52, Services -0.39, Trading Activity +0.37, SIC67 -0.32   observed 0.0026, noise 0.0026
  observed over noise, direction by direction (generalized eigenvalues): 0.25 0.34 0.42 0.45 0.56 0.63 0.72 0.88 0.99 1.28 1.42 1.90 1.95 2.13 3.02 3.94 4.29 4.95 6.04 9.18 9.76 12.81 29.16 53.34 126.78
  9 direction(s) whose observed variance is below the noise it should contain

  Reference portfolios on the forecast date: risk as estimated, after the subtraction alone, and after the repair (the model's)
                                            factor                  specific   total                  change
  portfolio                                as est.  subtr.  repair           as est.  subtr.  repair        
  the index, cap-weighted                   11.37%  11.22%  11.23%     3.22%  11.82%  11.67%  11.69%   -1.1%
  the index, equally weighted                9.86%   9.72%   9.93%     1.26%   9.94%   9.80%  10.01%   +0.8%
  smallest 85, equally weighted             12.65%  11.84%  12.56%     3.27%  13.06%  12.28%  12.98%   -0.6%
  largest 50, equally weighted              12.22%  12.06%  12.08%     3.64%  12.75%  12.59%  12.62%   -1.0%
  Construction (8), cap-weighted            24.14%  20.21%  21.16%    11.78%  26.86%  23.40%  24.22%   -9.8%
  Wholesale (10), cap-weighted              14.10%   8.38%  10.74%     7.93%  16.18%  11.54%  13.35%  -17.5%
  Finance & Real Estate (11), cap-weighted  17.90%  14.72%  16.02%     8.30%  19.73%  16.90%  18.04%   -8.6%

  Simulated: 938 daily periods drawn from a truth plus the measured noise, estimated as the run does (Newey-West 10 lags, half-life 90), the noise subtracted; 200 draws each
    truth the repaired matrix itself (9 directions at zero): negative eigenvalues median 10 (p10 9, p90 10); most negative median -0.0064
    truth the repaired matrix, its 9 clipped eigenvalues raised to the noise's median: negative eigenvalues median 7 (p10 7, p90 8); most negative median -0.0055
    truth the matrix as estimated (no direction below its noise): negative eigenvalues median 7 (p10 6, p90 8); most negative median -0.0041
    the run: 9 negative, most negative -0.0070
```

### sampling_profile.txt -- each factor's sampling variance over time

```text
Each factor's sampling variance (annualised) over the 938 periods: the equal-weighted mean, Newey-West's weighting of it (normalised to sum one), their ratio, and by half-year the mean
  factor                    eq. mean    NW wt.  ratio  2023H1    2023H2    2024H1    2024H2    2025H1    2025H2    2026H1    2026H2    
  Construction               0.01440   0.01905   1.32      0.86    0.70    0.76    0.90    1.12    1.00    1.29    1.48
  Finance & Real Estate      0.00729   0.01138   1.56      0.76    0.65    0.75    0.85    0.90    0.89    1.61    1.80
  Manufacturing              0.00241   0.00362   1.50      0.72    0.62    0.71    0.88    0.98    1.01    1.54    1.70
  Mining                     0.00641   0.00977   1.53      0.67    0.56    0.69    0.91    1.05    1.01    1.54    1.75
  Retail                     0.00184   0.00287   1.56      0.71    0.62    0.69    0.86    0.95    0.98    1.56    1.82
  SIC20                      0.00481   0.00825   1.72      0.58    0.52    0.63    0.81    0.96    1.00    1.71    2.03
  SIC28                      0.00176   0.00282   1.60      0.62    0.56    0.66    0.84    1.00    1.04    1.64    1.83
  SIC35                      0.00185   0.00256   1.39      0.84    0.68    0.77    0.89    1.00    0.97    1.44    1.54
  SIC36                      0.00201   0.00247   1.23      0.93    0.76    0.82    0.97    1.07    0.90    1.33    1.31
  SIC37                      0.00386   0.00573   1.48      0.72    0.61    0.75    0.90    1.02    0.97    1.49    1.70
  SIC38                      0.00229   0.00357   1.55      0.68    0.60    0.69    0.86    0.98    0.99    1.60    1.77
  SIC49                      0.00282   0.00439   1.56      0.71    0.63    0.74    0.86    0.94    0.94    1.57    1.81
  SIC60                      0.00354   0.00521   1.47      0.79    0.67    0.73    0.90    0.98    0.94    1.49    1.68
  SIC62                      0.00404   0.00546   1.35      0.91    0.76    0.84    0.92    0.94    0.88    1.37    1.53
  SIC63                      0.00308   0.00500   1.62      0.66    0.57    0.67    0.82    0.93    1.01    1.68    1.87
  SIC67                      0.00378   0.00594   1.57      0.70    0.61    0.71    0.84    0.94    0.98    1.59    1.82
  SIC73                      0.00097   0.00153   1.58      0.79    0.65    0.71    0.83    0.89    0.85    1.67    1.82
  Services                   0.00453   0.00716   1.58      0.71    0.62    0.70    0.86    0.94    0.93    1.61    1.84
  Transport & Utilities      0.00298   0.00471   1.58      0.66    0.58    0.71    0.86    0.97    1.00    1.58    1.83
  Wholesale                  0.00902   0.01396   1.55      0.70    0.60    0.69    0.87    0.97    0.96    1.62    1.76
  Volatility                 0.00031   0.00052   1.68      0.76    0.59    0.63    0.81    0.93    0.90    1.64    2.01
  Momentum                   0.00019   0.00028   1.47      0.85    0.63    0.67    0.91    0.98    0.90    1.65    1.55
  Size                       0.00011   0.00015   1.45      0.78    0.69    0.77    0.85    0.98    0.97    1.49    1.63
  Size Nonlinearity          0.00025   0.00038   1.50      0.83    0.66    0.71    0.84    0.96    0.93    1.55    1.71
  Trading Activity           0.00029   0.00043   1.45      0.82    0.64    0.71    0.93    1.06    0.91    1.41    1.69
  (by half-year as a multiple of the equal-weighted mean)

  Newey-West's weights carry 0.9625 of the noise; normalised, the ratio is the time profile alone. The weighted noise share is k.S / observed = 0.9625 x ratio x the equal-weighted share's numerator.
  securities per regression by half-year: 2023H1 502, 2023H2 502, 2024H1 501, 2024H2 501, 2025H1 503, 2025H2 502, 2026H1 502, 2026H2 500
```

### capture2/report_C2.txt -- the run with --sampling-correction, final code: its correction table and qualifications (lines 147-173, from 'Qualifications')

```text
  The sampling-error correction: the factor covariance less the sampling covariance of the factor returns, weighted as the covariance carries each period's; covariance repaired: 9 negative eigenvalue(s), most negative -7.031e-03, floored at 2.137e-09
  factor                  as estimated  weighted noise  subtracted  corrected
  Construction                 19.42%          48.6%     13.91%     15.05%
  Finance & Real Estate        15.34%          46.5%     11.22%     12.91%
  Manufacturing                10.20%          33.5%      8.32%      8.82%
  Mining                       18.60%          27.2%     15.88%     15.98%
  Retail                       13.11%          16.1%     12.01%     12.28%
  SIC20                        11.96%          55.5%      7.98%      9.35%
  SIC28                        13.14%          15.7%     12.06%     12.14%
  SIC35                        18.81%           7.0%     18.14%     18.25%
  SIC36                        21.13%           5.3%     20.56%     20.77%
  SIC37                        14.22%          27.3%     12.13%     12.18%
  SIC38                        14.70%          15.9%     13.48%     13.70%
  SIC49                        15.80%          16.9%     14.40%     14.51%
  SIC60                        21.85%          10.5%     20.67%     20.77%
  SIC62                        18.73%          15.0%     17.27%     17.49%
  SIC63                        14.57%          22.7%     12.81%     12.95%
  SIC67                        15.01%          25.4%     12.97%     13.29%
  SIC73                        18.18%           4.5%     17.77%     17.85%
  Services                     12.72%          42.6%      9.64%     10.34%
  Transport & Utilities        13.43%          25.1%     11.62%     11.80%
  Wholesale                    13.98%          68.8%      7.81%      9.51%
  Volatility                    9.37%           5.7%      9.11%      9.28%
  Momentum                      7.22%           5.1%      7.03%      7.11%
  Size                          3.72%          10.7%      3.52%      3.70%
  Size Nonlinearity             2.32%          67.4%      1.32%      1.71%
  Trading Activity              6.25%          10.5%      5.92%      6.07%
[...]
Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
  - 42 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CERN
  - 69% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is corrected for it
  - the factor covariance less the sampling covariance of the factor returns, weighted as the covariance carries each period's: covariance repaired: 9 negative eigenvalue(s), most negative -7.031e-03, floored at 2.137e-09
  - 11 security(ies) are in no estimated period's regression and have specific returns of their own, measured against the members' factor returns: their forecast blends that history with the characteristic forecast extrapolated to them: AMTM, FBIN, GAP, IPGP, MBC
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### item2.txt -- items 2 and 3: the folds, the floors, Size Nonlinearity

```text
Item 2: the thin industries
  counts over the universe: Construction 8, Finance & Real Estate 12, Wholesale 12, SIC37 15, Mining 16, SIC60 20, SIC62 20, Services 20, SIC20 21, SIC63 25, SIC67 29, Transport & Utilities 32, Retail 34, SIC35 34, SIC36 36, SIC49 37, Manufacturing 39, SIC28 40, SIC38 44, SIC73 70
  The industry each thin industry's factor return correlates with most, by three measures (top four)
    Construction           Pearson, daily, the estimated periods (the rule)     Retail 0.606, SIC67 0.577, SIC38 0.555, SIC35 0.554
    Construction           the model's covariance (Newey-West, half-life 90)    SIC36 0.748, SIC67 0.543, SIC35 0.497, Retail 0.479
    Construction           the corrected covariance                             SIC36 0.936, SIC67 0.742, SIC35 0.655, Transport & Utilities 0.649
    Construction           Pearson with the cap-weighted market taken out       SIC67 0.241, Retail 0.228, SIC49 0.164, Wholesale 0.077
    Wholesale              Pearson, daily, the estimated periods (the rule)     Retail 0.709, SIC38 0.700, Services 0.652, SIC35 0.651
    Wholesale              the model's covariance (Newey-West, half-life 90)    SIC38 0.540, Retail 0.507, SIC60 0.463, Transport & Utilities 0.450
    Wholesale              the corrected covariance                             SIC38 0.749, Services 0.738, Retail 0.718, Finance & Real Estate 0.678
    Wholesale              Pearson with the cap-weighted market taken out       Retail 0.224, SIC63 0.191, SIC20 0.174, SIC28 0.151
    Finance & Real Estate  Pearson, daily, the estimated periods (the rule)     SIC62 0.761, SIC73 0.732, SIC63 0.718, SIC60 0.713
    Finance & Real Estate  the model's covariance (Newey-West, half-life 90)    SIC62 0.644, SIC63 0.598, Services 0.590, SIC73 0.580
    Finance & Real Estate  the corrected covariance                             Services 0.780, SIC62 0.734, SIC63 0.718, Wholesale 0.678
    Finance & Real Estate  Pearson with the cap-weighted market taken out       SIC63 0.456, SIC62 0.339, SIC60 0.331, SIC73 0.228

  Each folded by the rule (Pearson), against C

  Construction (8) into Retail (34): Retail now 42  [M_CON]
    mean R-squared 0.293203 -> 0.290192 (-0.003011); factors 25 -> 24
    Construction's factor              vol 19.42%, noise 38.2% eq. / 48.6% wt.; corrected vol 15.05%
    Retail's factor                    vol 13.11%, noise 10.7% eq. / 16.1% wt.; corrected vol 12.28%
    the merged Retail                  vol 12.79%, noise 10.1% eq. / 14.9% wt.; corrected vol 11.93%
    the correction's repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0055
    Construction's 8 members' specific risk: median 23.80% -> 27.91%; change median +12.4%, range -9.6% to +39.9%
      DHI 20.2%->28.2%, EME 28.6%->29.7%, FIX 35.3%->36.3%, J 27.6%->25.0%, LEN 20.6%->27.6%, NVR 18.8%->21.5%, PHM 18.1%->24.8%, PWR 27.0%->29.9%
    Retail's 34 members' specific risk: median 26.04% -> 26.10%; change median +0.3%, range -3.3% to +1.9%
    the Construction (8), cap-weighted portfolio: factor 24.14% -> 18.87%, specific 11.78% -> 13.06%, total 26.86% -> 22.95%; corrected total 24.22% -> 22.39%

  Wholesale (12) into Retail (34): Retail now 46  [M_WHO]
    mean R-squared 0.293203 -> 0.291445 (-0.001759); factors 25 -> 24
    Wholesale's factor                 vol 13.98%, noise 46.2% eq. / 68.8% wt.; corrected vol 9.51%
    Retail's factor                    vol 13.11%, noise 10.7% eq. / 16.1% wt.; corrected vol 12.28%
    the merged Retail                  vol 12.32%, noise 10.4% eq. / 15.6% wt.; corrected vol 11.53%
    the correction's repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0069
    Wholesale's 12 members' specific risk: median 23.56% -> 24.60%; change median +1.6%, range -6.8% to +28.2%
      CAH 19.6%->23.9%, COR 17.8%->22.8%, DPZ 28.4%->28.9%, FERG 29.2%->27.2%, GPC 24.9%->24.8%, GWW 19.2%->19.2%, HSIC 23.1%->24.0%, LKQ 28.9%->28.6%, MCK 20.6%->25.9%, POOL 27.2%->26.4%, SYY 20.1%->20.6%, TEL 24.1%->24.4%
    Retail's 34 members' specific risk: median 26.04% -> 26.07%; change median +0.0%, range -2.0% to +1.9%
    the Wholesale (10), cap-weighted portfolio: factor 14.10% -> 14.32%, specific 7.93% -> 9.08%, total 16.18% -> 16.95%; corrected total 13.35% -> 16.43%

  Finance & Real Estate (12) into SIC62 (20): SIC62 now 32  [M_FRE]
    mean R-squared 0.293203 -> 0.290846 (-0.002358); factors 25 -> 24
    Finance & Real Estate's factor     vol 15.34%, noise 31.0% eq. / 46.5% wt.; corrected vol 12.91%
    SIC62's factor                     vol 18.73%, noise 11.5% eq. / 15.0% wt.; corrected vol 17.49%
    the merged SIC62                   vol 16.36%, noise 10.2% eq. / 14.1% wt.; corrected vol 15.26%
    the correction's repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0054
    Finance & Real Estate's 11 members' specific risk: median 19.39% -> 22.55%; change median +9.4%, range -4.3% to +32.4%
      AJG 17.7%->22.5%, AON 17.3%->21.8%, AXP 18.4%->17.8%, BRO 18.4%->22.5%, CBRE 25.2%->24.1%, COIN 65.2%->64.8%, ERIE 29.7%->32.5%, INVH 21.9%->21.2%, MRSH 13.4%->17.8%, SYF 25.3%->24.2%, WTW 19.4%->22.6%
    SIC62's 20 members' specific risk: median 22.24% -> 22.67%; change median +3.0%, range -4.6% to +9.3%
    the Finance & Real Estate (11), cap-weighted portfolio: factor 17.90% -> 16.93%, specific 8.30% -> 8.61%, total 19.73% -> 19.00%; corrected total 18.04% -> 18.32%

  Each folded where the model's own covariance points instead, against C

  Construction (8) into SIC36 (36): SIC36 now 44  [A_CON]
    mean R-squared 0.293203 -> 0.288714 (-0.004490); factors 25 -> 24
    Construction's factor              vol 19.42%, noise 38.2% eq. / 48.6% wt.; corrected vol 15.05%
    SIC36's factor                     vol 21.13%, noise 4.5% eq. / 5.3% wt.; corrected vol 20.77%
    the merged SIC36                   vol 20.46%, noise 4.3% eq. / 5.1% wt.; corrected vol 20.05%
    the correction's repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0055
    Construction's 8 members' specific risk: median 23.80% -> 28.92%; change median +25.3%, range -6.2% to +63.1%
      DHI 20.2%->32.9%, EME 28.6%->28.4%, FIX 35.3%->35.8%, J 27.6%->25.9%, LEN 20.6%->31.9%, NVR 18.8%->27.8%, PHM 18.1%->29.4%, PWR 27.0%->27.7%
    SIC36's 36 members' specific risk: median 35.16% -> 35.46%; change median +0.3%, range -5.6% to +2.4%
    the Construction (8), cap-weighted portfolio: factor 24.14% -> 25.12%, specific 11.78% -> 12.78%, total 26.86% -> 28.18%; corrected total 24.22% -> 27.99%

  Wholesale (12) into SIC38 (44): SIC38 now 56  [A_WHO]
    mean R-squared 0.293203 -> 0.291173 (-0.002031); factors 25 -> 24
    Wholesale's factor                 vol 13.98%, noise 46.2% eq. / 68.8% wt.; corrected vol 9.51%
    SIC38's factor                     vol 14.70%, noise 10.6% eq. / 15.9% wt.; corrected vol 13.70%
    the merged SIC38                   vol 13.55%, noise 10.5% eq. / 15.9% wt.; corrected vol 12.58%
    the correction's repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0070
    Wholesale's 12 members' specific risk: median 23.56% -> 24.77%; change median +4.9%, range -7.3% to +27.8%
      CAH 19.6%->23.3%, COR 17.8%->22.2%, DPZ 28.4%->29.7%, FERG 29.2%->27.1%, GPC 24.9%->26.2%, GWW 19.2%->20.3%, HSIC 23.1%->23.0%, LKQ 28.9%->29.4%, MCK 20.6%->26.4%, POOL 27.2%->27.8%, SYY 20.1%->21.2%, TEL 24.1%->23.0%
    SIC38's 43 members' specific risk: median 25.53% -> 25.46%; change median +0.2%, range -2.3% to +2.3%
    the Wholesale (10), cap-weighted portfolio: factor 14.10% -> 12.77%, specific 7.93% -> 9.10%, total 16.18% -> 15.68%; corrected total 13.35% -> 15.17%

  Finance & Real Estate (12) into Services (20): Services now 32  [A_FRE]
    mean R-squared 0.293203 -> 0.290841 (-0.002363); factors 25 -> 24
    Finance & Real Estate's factor     vol 15.34%, noise 31.0% eq. / 46.5% wt.; corrected vol 12.91%
    Services's factor                  vol 12.72%, noise 28.0% eq. / 42.6% wt.; corrected vol 10.34%
    the merged Services                vol 12.26%, noise 19.5% eq. / 29.7% wt.; corrected vol 10.91%
    the correction's repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0054
    Finance & Real Estate's 11 members' specific risk: median 19.39% -> 20.75%; change median +7.8%, range -5.1% to +12.9%
      AJG 17.7%->19.9%, AON 17.3%->19.1%, AXP 18.4%->20.1%, BRO 18.4%->19.8%, CBRE 25.2%->24.9%, COIN 65.2%->66.3%, ERIE 29.7%->30.3%, INVH 21.9%->20.8%, MRSH 13.4%->14.9%, SYF 25.3%->27.0%, WTW 19.4%->21.3%
    Services's 21 members' specific risk: median 28.30% -> 29.05%; change median +0.5%, range -4.2% to +5.8%
    the Finance & Real Estate (11), cap-weighted portfolio: factor 17.90% -> 14.69%, specific 8.30% -> 8.95%, total 19.73% -> 17.20%; corrected total 18.04% -> 16.54%

  The floors: smallest industry below the floor first, folded into the industry its factor return correlates with most (Pearson, daily, the estimated periods), re-estimated after each fold
    fold 1: Construction (8) into Retail (34), correlation 0.606; next SIC67 0.577, SIC38 0.555
      F1: status 0, mean R2 0.290192, 938 periods, 24 factors, 76s
    fold 2: Finance & Real Estate (12) into SIC62 (20), correlation 0.760; next SIC73 0.732, SIC63 0.718
      F2: status 0, mean R2 0.287833, 938 periods, 23 factors, 66s
    fold 3: Wholesale (12) into Retail (42), correlation 0.714; next SIC38 0.700, Services 0.652
      F3: status 0, mean R2 0.286053, 938 periods, 22 factors, 59s
    floor 15 holds after 3 fold(s): Construction=Retail, Finance & Real Estate=SIC62, Wholesale=Retail
    fold 4: SIC37 (15) into SIC62 (32), correlation 0.700; next Retail 0.691, SIC38 0.680
      F4: status 0, mean R2 0.280747, 938 periods, 21 factors, 75s
    fold 5: Mining (16) into Manufacturing (39), correlation 0.770; next SIC35 0.555, SIC62 0.541
      F5: status 0, mean R2 0.276914, 938 periods, 20 factors, 72s
    floor 20 holds after 5 fold(s): Construction=Retail, Finance & Real Estate=SIC62, Wholesale=Retail, SIC37=SIC62, Mining=Manufacturing
    fold 6: SIC60 (20) into SIC62 (47), correlation 0.815; next Transport & Utilities 0.680, SIC35 0.678
      F6: status 0, mean R2 0.270657, 938 periods, 19 factors, 75s
    fold 7: Services (20) into SIC73 (70), correlation 0.760; next Retail 0.750, SIC38 0.748
      F7: status 0, mean R2 0.266994, 938 periods, 18 factors, 75s
    fold 8: SIC20 (21) into Retail (54), correlation 0.637; next SIC67 0.614, SIC28 0.613
      F8: status 0, mean R2 0.262773, 938 periods, 17 factors, 66s
    floor 25 holds after 8 fold(s): Construction=Retail, Finance & Real Estate=SIC62, Wholesale=Retail, SIC37=SIC62, Mining=Manufacturing, SIC60=SIC62, Services=SIC73, SIC20=Retail
  floor 15: run F3
  floor 20: run F5
  floor 25: run F8
  with the market taken out, each fold's source would correlate most with:
    fold 1: Construction -> SIC67 (0.241), not Retail
    fold 2: Finance & Real Estate -> SIC63 (0.457), not SIC62
    fold 3: Wholesale -> Retail (0.223)
    fold 4: SIC37 -> SIC60 (0.164), not SIC62
    fold 5: Mining -> Manufacturing (0.614)
    fold 6: SIC60 -> SIC62 (0.511)
    fold 7: Services -> SIC73 (0.223)
    fold 8: SIC20 -> SIC49 (0.304), not Retail

                                                         C    floor 15    floor 20    floor 25
  industries                                            20          17          15          12
  smallest industry                                      8          15          20          25
  mean R-squared                                  0.293203    0.286053    0.276914    0.262773
  largest industry noise share, eq.                  46.2%       33.4%       33.9%       17.2%
  largest industry noise share, wt.                  68.8%       55.1%       56.0%       25.8%
  Size Nonlinearity noise share, wt.                 67.4%       66.8%       62.3%       62.7%
  repair: negative eigenvalues                           9           6           5           2
  repair: most negative                            -0.0070     -0.0031     -0.0031     -0.0011
  specific risk median                              24.76%      24.78%      24.98%      24.99%
  the index, cap-weighted, as est.                  11.82%      11.82%      11.83%      11.84%
  the index, cap-weighted, corrected                11.69%      11.68%      11.69%      11.70%
  the index, equally weighted, as est.               9.94%       9.92%       9.93%       9.93%
  the index, equally weighted, corrected            10.01%       9.91%       9.84%       9.82%
  smallest 85, equally weighted, as est.            13.06%      13.07%      13.05%      13.14%
  smallest 85, equally weighted, corrected          12.98%      12.83%      12.69%      12.71%
  largest 50, equally weighted, as est.             12.75%      12.71%      12.70%      12.69%
  largest 50, equally weighted, corrected           12.62%      12.56%      12.55%      12.54%
  Construction (8), cap-weighted, as est.           26.86%      22.21%      22.20%      21.19%
  Construction (8), cap-weighted, corrected         24.22%      21.70%      21.63%      20.64%
  Wholesale (10), cap-weighted, as est.             16.18%      16.53%      16.48%      15.64%
  Wholesale (10), cap-weighted, corrected           13.35%      16.05%      15.90%      15.08%
  Finance & Real Estate (11), cap-weighted, as est.      19.73%      19.02%      17.59%      18.04%
  Finance & Real Estate (11), cap-weighted, corrected      18.04%      18.24%      17.04%      17.63%
  floor 15: the 31 folded members' specific risk median 21.87% -> 24.24%; change median +3.3%, range -10.7% to +41.2%; folds Construction into Retail, Finance & Real Estate into SIC62, Wholesale into Retail
    noisiest industries, wt.: SIC20 (21) 55.1%, Services (20) 42.4%, Manufacturing (39) 33.4%, SIC37 (15) 27.3%
  floor 20: the 59 folded members' specific risk median 23.06% -> 24.71%; change median +3.2%, range -12.9% to +41.1%; folds Construction into Retail, Finance & Real Estate into SIC62, Mining into Manufacturing, SIC37 into SIC62, Wholesale into Retail
    noisiest industries, wt.: SIC20 (21) 56.0%, Services (20) 43.2%, SIC67 (29) 25.6%, Transport & Utilities (32) 25.1%
  floor 25: the 118 folded members' specific risk median 21.69% -> 24.00%; change median +7.1%, range -13.3% to +42.4%; folds Construction into Retail, Finance & Real Estate into SIC62, Mining into Manufacturing, SIC20 into Retail, SIC37 into SIC62, SIC60 into SIC62, Services into SIC73, Wholesale into Retail
    noisiest industries, wt.: SIC67 (29) 25.8%, Transport & Utilities (32) 25.3%, SIC63 (25) 23.0%, Manufacturing (55) 22.0%

Item 3: Size Nonlinearity
  in C: vol as estimated 2.32%, noise 46.6% eq. / 67.4% wt.; the subtraction leaves 1.32%, the repair 1.71%
    Size and Size Nonlinearity, as estimated  vols 3.72% / 2.32%, correlation -0.132
    Size and Size Nonlinearity, subtracted    vols 3.52% / 1.32%, correlation -0.243
    Size and Size Nonlinearity, corrected     vols 3.70% / 1.71%, correlation -0.169
  without it [SN0, --industries --membership --sampling-correction --size-nonlinearity none]
    mean R-squared 0.293203 -> 0.289600 (-0.003603)
    Size: vol 3.72% -> 3.72%, noise wt. 10.7% -> 10.7%
    the repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0071
                                               with, as est. with, corr.   without  w/o, corr.
  the index, cap-weighted                             11.82%      11.69%    11.78%      11.65%
  the index, equally weighted                          9.94%      10.01%     9.94%       9.97%
  smallest 85, equally weighted                       13.06%      12.98%    12.99%      13.03%
  largest 50, equally weighted                        12.75%      12.62%    12.72%      12.59%
  Construction (8), cap-weighted                      26.86%      24.22%    26.92%      24.21%
  Wholesale (10), cap-weighted                        16.18%      13.35%    16.28%      13.39%
  Finance & Real Estate (11), cap-weighted            19.73%      18.04%    19.76%      18.02%
    the smallest 85's specific risk: median 27.42% -> 27.69%; change median +0.5%, range -2.2% to +2.8%
    the largest 50's specific risk: median 21.38% -> 21.58%; change median +0.0%, range -4.0% to +3.4%
    every name with a forecast's specific risk: median 24.76% -> 24.79%; change median +0.0%, range -4.0% to +5.2%
    the smallest 85's mean Size Nonlinearity exposure -1.808

  Each folded where the market-removed measure points (Wholesale's is the rule's Retail), against C

  Construction (8) into SIC67 (29): SIC67 now 37  [X_CON]
    mean R-squared 0.293203 -> 0.289918 (-0.003286); factors 25 -> 24
    Construction's factor              vol 19.42%, noise 38.2% eq. / 48.6% wt.; corrected vol 15.05%
    SIC67's factor                     vol 15.01%, noise 16.8% eq. / 25.4% wt.; corrected vol 13.29%
    the merged SIC67                   vol 14.49%, noise 14.8% eq. / 21.8% wt.; corrected vol 13.13%
    the correction's repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0055
    Construction's 8 members' specific risk: median 23.80% -> 28.07%; change median +13.5%, range -5.3% to +40.9%
      DHI 20.2%->28.4%, EME 28.6%->29.4%, FIX 35.3%->36.7%, J 27.6%->26.1%, LEN 20.6%->27.7%, NVR 18.8%->22.8%, PHM 18.1%->25.0%, PWR 27.0%->28.6%
    SIC67's 29 members' specific risk: median 17.83% -> 18.49%; change median +1.6%, range -3.6% to +6.0%
    the Construction (8), cap-weighted portfolio: factor 24.14% -> 19.29%, specific 11.78% -> 12.75%, total 26.86% -> 23.12%; corrected total 24.22% -> 22.46%

  Finance & Real Estate (12) into SIC63 (25): SIC63 now 37  [X_FRE]
    mean R-squared 0.293203 -> 0.290736 (-0.002468); factors 25 -> 24
    Finance & Real Estate's factor     vol 15.34%, noise 31.0% eq. / 46.5% wt.; corrected vol 12.91%
    SIC63's factor                     vol 14.57%, noise 14.5% eq. / 22.7% wt.; corrected vol 12.95%
    the merged SIC63                   vol 13.47%, noise 12.6% eq. / 19.5% wt.; corrected vol 12.20%
    the correction's repair: 9 negative eigenvalue(s), most negative -0.0070 -> 8 negative eigenvalue(s), most negative -0.0053
    Finance & Real Estate's 11 members' specific risk: median 19.39% -> 21.49%; change median +5.2%, range -2.1% to +16.6%
      AJG 17.7%->18.7%, AON 17.3%->18.7%, AXP 18.4%->21.5%, BRO 18.4%->19.1%, CBRE 25.2%->27.2%, COIN 65.2%->67.1%, ERIE 29.7%->29.1%, INVH 21.9%->21.9%, MRSH 13.4%->14.1%, SYF 25.3%->27.6%, WTW 19.4%->20.4%
    SIC63's 25 members' specific risk: median 20.27% -> 19.62%; change median -0.9%, range -5.8% to +5.3%
    the Finance & Real Estate (11), cap-weighted portfolio: factor 17.90% -> 14.08%, specific 8.30% -> 9.29%, total 19.73% -> 16.87%; corrected total 18.04% -> 16.23%

  Mean R-squared each fold costs, against the run before it
    fold 1: Construction (8) into Retail: -0.003011
    fold 2: Finance & Real Estate (12) into SIC62: -0.002360
    fold 3: Wholesale (12) into Retail: -0.001780
    fold 4: SIC37 (15) into SIC62: -0.005306
    fold 5: Mining (16) into Manufacturing: -0.003834
    fold 6: SIC60 (20) into SIC62: -0.006257
    fold 7: Services (20) into SIC73: -0.003662
    fold 8: SIC20 (21) into Retail: -0.004222
    Construction into Retail                 alone, against C: -0.003011
    Wholesale into Retail                    alone, against C: -0.001759
    Finance & Real Estate into SIC62         alone, against C: -0.002358
    Construction into SIC36                  alone, against C: -0.004490
    Wholesale into SIC38                     alone, against C: -0.002031
    Finance & Real Estate into Services      alone, against C: -0.002363
    Construction into SIC67                  alone, against C: -0.003286
    Finance & Real Estate into SIC63         alone, against C: -0.002468
    Size Nonlinearity left out               alone, against C: -0.003603
```

### floors.txt -- the floors' fold sequence, as it ran

```text
Floors: smallest industry below the floor first, folded into the industry its factor return correlates with most (Pearson, daily, the estimated periods), re-estimated after each fold
  fold 1: Construction (8) into Retail (34), correlation 0.606; next SIC67 0.577, SIC38 0.555
    F1: status 0, mean R2 0.290192, 938 periods, 24 factors, 76s
  fold 2: Finance & Real Estate (12) into SIC62 (20), correlation 0.760; next SIC73 0.732, SIC63 0.718
    F2: status 0, mean R2 0.287833, 938 periods, 23 factors, 66s
  fold 3: Wholesale (12) into Retail (42), correlation 0.714; next SIC38 0.700, Services 0.652
    F3: status 0, mean R2 0.286053, 938 periods, 22 factors, 59s
  floor 15 holds after 3 fold(s): Construction=Retail, Finance & Real Estate=SIC62, Wholesale=Retail
  fold 4: SIC37 (15) into SIC62 (32), correlation 0.700; next Retail 0.691, SIC38 0.680
    F4: status 0, mean R2 0.280747, 938 periods, 21 factors, 75s
  fold 5: Mining (16) into Manufacturing (39), correlation 0.770; next SIC35 0.555, SIC62 0.541
    F5: status 0, mean R2 0.276914, 938 periods, 20 factors, 72s
  floor 20 holds after 5 fold(s): Construction=Retail, Finance & Real Estate=SIC62, Wholesale=Retail, SIC37=SIC62, Mining=Manufacturing
  fold 6: SIC60 (20) into SIC62 (47), correlation 0.815; next Transport & Utilities 0.680, SIC35 0.678
    F6: status 0, mean R2 0.270657, 938 periods, 19 factors, 75s
  fold 7: Services (20) into SIC73 (70), correlation 0.760; next Retail 0.750, SIC38 0.748
    F7: status 0, mean R2 0.266994, 938 periods, 18 factors, 75s
  fold 8: SIC20 (21) into Retail (54), correlation 0.637; next SIC67 0.614, SIC28 0.613
    F8: status 0, mean R2 0.262773, 938 periods, 17 factors, 66s
  floor 25 holds after 8 fold(s): Construction=Retail, Finance & Real Estate=SIC62, Wholesale=Retail, SIC37=SIC62, Mining=Manufacturing, SIC60=SIC62, Services=SIC73, SIC20=Retail
floor 15: run F3
floor 20: run F5
floor 25: run F8
```

### final_checks.txt -- the final code: the defaults, the correction, midoffice09's panel

```text
The defaults, on the final code (Q2c), against midoffice09's defaults (capture/Q2):
  report: 161 and 161 lines, identical: True
  vol: equal
  factor returns: equal
  noise share: equal
  mean R-squared: equal
  specific risk: equal
  exposures: equal
  estimated dates: equal
  cross-section sizes: equal
  the correction off: True, weighted share 0 entries, repair None

The correction, on the final code (C2), against C:
  27 fields compared; differing: none; the matrix the repair was handed equal: True
  report: 190 and 190 lines; the lines that differ:
    -  - 47% of the estimated variance of Size Nonlinearity is the sampling error of its own estimation; the factor covariance diagonal is corrected for it
    +  - 69% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is corrected for it

midoffice09's panel by its flags (--spare none --listed any), on the final code, against data/'s _repaired:
  arrays: 20 and 20; every one equal: True
  report: keys of midoffice09's that differ, beside rules: ['repairs']
    the repairs equal without their two new fields: True; those fields: DGX None False, EG None False, NWS None False, PCG None False, RDDT None False, SBUX None False, XYZ None False
  rules {'first_counts': 'confirmed', 'repair': 'isolated', 'overrides': 'config/sp500_share_count_overrides.json', 'tolerance': 0.1} -> {'first_counts': 'confirmed', 'repair': 'isolated', 'spare': 'none', 'listed': 'any', 'overrides': 'config/sp500_share_count_overrides.json', 'tolerance': 0.1}; new keys: ['delistings']
```

### pytest_final.txt -- the full suite (last line); pytest_mid.txt -- the midoffice tests

```text
=========== 2311 passed, 1 skipped, 19 warnings in 72.70s (0:01:12) ===========
======================= 860 passed, 1 skipped in 29.32s =======================
```

### ruff_final.txt -- ruff over the changed Python files

```text
scripts\run_sp500_model.py:41:5: F401 [*] `praxis.midoffice.model.covariance.HorizonCovariance` imported but unused
src\praxis\midoffice\model\crosssectional.py:92:20: F401 [*] `typing.Mapping` imported but unused
Found 2 errors.
[*] 2 fixable with the `--fix` option.
```

### hygiene_final.txt -- parse, ASCII and line endings, by bytes (this retro checked after it was written: LF, ASCII)

```text
src/praxis/midoffice/refdata/securities.py: CRLF (933 lines), parses, ASCII
src/praxis/midoffice/model/crosssectional.py: CRLF (1287 lines), parses, ASCII
src/praxis/midoffice/refdata/industries.py: CRLF (473 lines), parses, ASCII
scripts/build_sp500_securities.py: CRLF (626 lines), parses, ASCII
scripts/run_sp500_model.py: CRLF (1266 lines), parses, ASCII
tests/midoffice/test_securities.py: CRLF (387 lines), parses, ASCII
tests/midoffice/test_crosssectional.py: CRLF (1734 lines), parses, ASCII
tests/midoffice/test_industries.py: LF (400 lines), parses, ASCII
docs/FACTOR_MODEL.md: CRLF (1113 lines), -, ASCII
claude/handoffs/BRIEF_midoffice10_noise_share.md: LF (69 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (492 lines), -, ASCII
all ok
```

## git add (always last -- Rule 43)

This piece -- midoffice09's two rulings in the build and the run's default panel, the sampling-error correction, the industry folds, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add src/praxis/midoffice/refdata/securities.py src/praxis/midoffice/model/crosssectional.py src/praxis/midoffice/refdata/industries.py scripts/build_sp500_securities.py scripts/run_sp500_model.py tests/midoffice/test_securities.py tests/midoffice/test_crosssectional.py tests/midoffice/test_industries.py docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice10_noise_share.md claude/retros/RETRO_midoffice10_noise_share.md
```
