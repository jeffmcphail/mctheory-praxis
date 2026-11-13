# Retro: Mid-office 17 -- midoffice16's rulings, then the calm half's shape: two levers

**Brief:** `claude/handoffs/BRIEF_midoffice17_calm_half_two_levers.md` -- written in Chat and
pasted into the Code session, transcribed verbatim
**Date:** 2026-11-13 (EST)
**Duration:** the brief arrived shortly before 17:45Z (12:45 EST), the first clock reading after
it. This session is the one midoffice16 ran in, so the state file and midoffice16's retro were in
hand.
- **Ruling 4, built:** the plain run's regime history from the full fit behind its guard, its
  test written, by 12:48.
- **Item 2's pass, built** by 12:54: the volatility half-lives beside the correlations'.
- **Four smoke runs by 12:58:** the volatility pass and a run at 42 days alone, the eigen pass on
  the 63-day correlations, a plain run.
- **The judging script for items 1 and 2:** written at 12:59, before the runs.
- **The runs, one at a time through the runner:** E17 (item 1, 13:03 to 13:19), V17 (item 2,
  14:01 to 14:11) and the plain runs P17x and P17 (14:16 to 14:18); the runner waited for 3 GB of
  memory before each but P17, 42 minutes before V17.
- **Item 3,** which needs no run, while the runner waited and ran; then the analyses, the docs, the
  state file and this retro.

**Status:** COMPLETE.
- **Rulings 1 to 5:** ruling 4 built -- the only default change, and it changes no number; the
  others need nothing built (ruling 3's exact diagonal recorded for the first default change that
  moves every covariance).
- **Items 1 to 4:** item 1's test run and judged against D16; item 2's built and run and judged
  against 90; item 3 measured from D16's arrays; item 4 does not arise -- neither lever passed.
- **No default changed** beyond ruling 4.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 4, built -- the only default change, and it changes no number.**
  - **What it does:** a plain run reads the regime adjustment's factor side from the full fit --
    the model's estimator on the full fit's factor returns through each day before -- behind a
    guard of five refits spread over the history. The run stops on any difference.
  - **Its time:** 0.9 minutes, against 7.5 by the refits.
  - **Its report** is the refits' line for line but the last two (lambda_F 0.9640 both ways).
  - **A test** pins the history to the refits' on a synthetic panel, to the bit.
- **Rulings 1, 2, 3 and 5:** the correlations stay on 504; item 1 is ruling 2's test; the exact
  diagonal waits for the first default change that moves every covariance; this piece is ruling
  5's order.
- **Item 1, the one-sided eigenvalue adjustment on the 63-day correlations (E17): every a out.**
  - **The setup:** its model test is C16's 63-day setting to the bit, and the simulation finds
    nearly three times the 504-day matrix's excess bias to correct (about 1.17 at the smallest
    five, against 1.06).
  - **The statistics it is for:** it fixes them -- ranks 21-25 1.30 to 1.04 at a = 2, the
    optimiser's part in ranks 16-25 1.25 to 1.01.
  - **The optimiser's test, first:** it fails, and more at every a: unconstrained 10.28%, 10.36%
    and 10.46% a year against D16's 10.19%. The raised small eigenvalues move the optimiser's risk
    into the market direction (38% to 44% of its forecast variance), which the volatile first
    half cost.
- **Item 2, the volatilities' half-life (V17, a new one-pass option): every setting out.**
  - **The setup:** 21, 42 and 63 days against 90; its model test is D16's to the bit.
  - **The optimiser:** it realises more the shorter the window (unconstrained 10.26%, 10.36%,
    10.61%).
  - **The calm half's market direction** barely moves (0.814 to 0.843 at 21); the eigenportfolios'
    shape worsens there. The volatilities do more of the regime adjustment's work themselves
    (B_F^2 1.18 and 1.01 by half at 21 days, against 1.33 and 0.90).
- **Item 3, no run: the calm half's market direction is the correlations, not the volatilities --
  midoffice16's reading was wrong.**
  - **The correlations:** realised 0.316 between factors over the second half against the
    forecast's 0.432, and the market direction's share 0.47 against 0.64. They fell from 0.55 to
    0.25 between 2025 Q2 and Q3 and stayed near 0.30; a 63-day window followed a quarter late.
  - **The volatilities:** the market direction's industries were over-forecast by about 5%.
  - **Rebuilt:** on the realised correlations rank 1 goes from 0.814 to 0.946; on realised
    volatilities only to 0.861; on both, 1.000.
- **Item 4:** neither lever passed, so there is nothing to combine.
- **Which I would choose:** the factor covariance as it stands. The calm half's shape is left,
  understood: every lever that follows the correlations faster trades the shock half for the calm
  half and costs the optimiser. Next, the largest members' specific risk; a regime multiplier of
  the market direction's own is the design the evidence points to, Jeff's to want or not.
- **Tests:** 2,375 passed, 1 skipped; 1 new, 1 extended.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice17_calm_half_two_levers.md`, verbatim. In
outline:

- **"committed -- midoffice16."** Checked: eb2b7f9 in `git log`, the tree clean.
- **Rulings on midoffice16's decisions:**
  1. The correlations stay on 504 days. The trade is narrower than the halves make it look: at 63
     days the broad portfolios improve in both halves, and the whole cost is the optimiser's, in
     the small eigen-directions -- the eigenvalue adjustment's job, on a matrix with more for it
     to correct.
  2. Item 4's test: run it (item 1).
  3. The estimator's diagonal: left as it is; fold the exact diagonal into the first default
     change that moves every covariance anyway, so the stored runs' baseline breaks once.
  4. The plain run's regime history from the full fit: build it, with a guard -- a few as-of
     refits spread over the history, each checked against the full fit's prefix to the bit, the
     run refusing on any difference. Report the plain run's time. The bias test keeps its refits.
  5. Next: the calm half's shape once more, then the largest members' specific risk -- the factor
     covariance settled first, so the specific side is measured on the factor side it will keep.
- **midoffice17, the calm half's shape, two levers,** the rulings first:
  1. The one-sided eigenvalue adjustment, a = 1, 2, 3, on the 63-day correlations, one pass, its
     model test C16's 63-day setting to the bit. Judged by the rule against D16: both
     minimum-variance portfolios realising no more over the whole test than 10.13% and 10.19% a
     year, to the hundredth; then the criterion on the first half against 0.098, checked on the
     second. Beside it: ranks 21-25, the optimiser's share and statistic in ranks 16-25, the broad
     portfolios' MRAD by half, and rank 1 by half.
  2. The volatilities' half-life: 21, 42 and 63 days against 90, the correlations on 504, on the
     regime-adjusted model, one pass as C16 was, each setting on its own forecasts' regime
     history. The same rule, against 90. Beside it, C16's tables, with each setting's regime
     multiplier by half.
  3. No run: from D16's stored arrays and the model's factor returns, what the calm half's market
     direction is made of -- the realised factor correlations over the half against the
     forecast's, and each factor's own statistic by half, the market direction's heaviest factors
     first. "Your reading is that what is left is on the volatility side; this says whether it
     is."
  4. If both levers pass the rule, don't combine them this piece: say what the combination's test
     would be.
- Runs one at a time through the runner. **"Change no default beyond ruling 4. Report, and say
  which you would choose and why."** Retro with its `git add` and the state file in the same
  commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `scripts/run_sp500_model.py` | ruling 4: `regime_section` reads the factor side from the full fit behind its guard (`REGIME_GUARD` 5), refits where it must; the report's regime lines for a factor-only history; item 2: `--bias-volatility-half-lives` and its checks, beside `--bias-correlation-half-lives`, generalised -- the settings keyed c<h> or v<h> (`_setting_key`, `PLURAL`), `_report_settings` (was `_report_correlation`) with each volatility setting's regime multiplier by part, `COVARIANCE_RULE` (`CORRELATION_RULE` kept as its other name); the eigen scales' covariances kept in `--bias-out` | 72-77, 132-137, 915-971, 1061-1085, 1162-1217, 1359-1371, 1802-1889, 1924-1964, 2029-2065, 2267-2275, 2310-2325 |
| `src/praxis/midoffice/model/regime.py` | ruling 4: `forecast_volatilities`, `RegimeHistory.add_factor` and `has_specific` (the specific side refused on a factor-only history), `factor_history`, `factor_history_guard`; the module docstring | 41-61, 111-119, 212-256, 294-364 |
| `tests/midoffice/test_regime.py` | 1 new test | 19-29, 216-259 |
| `tests/midoffice/test_bias.py` | the re-estimation test extended to the volatilities' half-life | 405-406, 443-451 |
| `docs/FACTOR_MODEL.md` | the plain run's regime history from the full fit; midoffice16's volatility-side reading corrected; item 3's market direction; items 1 and 2 | 438-455, 1696-1700, 1716-1767 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice17_calm_half_two_levers.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice17_calm_half_two_levers.md` | new: this retro | new |

Written outside the tracked files: nothing in `data/` (no panel is built in this piece), nothing in
the cache; `claude/scratch/midoffice17/` (gitignored), the scratch, below.

---

## Ruling 4: the plain run's regime history from the full fit, behind a guard

**Built, in the library** (`regime.py`):
- **`factor_history(model, estimator, as_of)`:** the factor side's history read from one fit.
  Each day's B_F^2 comes from the model's own factor return that day and the factor volatilities
  `estimator` gives on the model's factor returns through the day before, as the risk engine
  uses them (`forecast_volatilities`: through `nearest_psd`, the diagonal over the periods a
  year).
  - **Refused** for a model whose covariance is not its estimator's alone (the sampling
    correction, the eigenvalue adjustment).
  - **The specific side is not read:** `RegimeHistory.add_factor` leaves it nan,
    `has_specific` says so, and its multiplier and path are refused.
- **`factor_history_guard(model, estimator, refits)`:** each refit as of a chosen date against the
  full fit. Its estimated periods, its factor returns, and the factor volatilities its own forecast
  carries must be the full fit's through its date, to the bit; what differs is named, by date.

**Built, in the run** (`regime_section`):
- **The guard first:** `REGIME_GUARD` = 5 refits -- the first forecast date, the last and three
  between, as the bias guard picks its dates -- and the run stops on any difference.
- **Then the history from the full fit**, and the report's line says so, with both times.
- **The refits stay** wherever they are needed: the specific side on, the sampling correction,
  the eigenvalue adjustment. The bias test keeps its refits.

**Checked:**
- **The test** (`test_regime.py`, below): on the synthetic panel, the history read from one fit
  equals the refits' to the bit -- B_F^2, the factor counts, the multiplier's path. The guard finds
  nothing on true refits. It names a refit whose factor returns differ in one bit, and every
  refit read against another estimator.
- **The smoke** (the last 30 forecast dates): the plain run's own forecast carries the refits'
  lambda_F, 0.9604, the guard's five refits 6.2 s, the history under 0.1 s.
- **On the full run (P17):** 0.9 minutes, against P17x's 0.8 with the factor side off and
  midoffice16's P16, 7.5 by the refits (`plain_cost.txt`). The guard's five refits took 3.2 s and
  the history 0.2 s. P17's report is P16's line for line but the last two -- "the specific side not
  read" for B_S^2, and the timing and guard line -- lambda_F 0.9640 as of 2026-11-03 both ways;
  P17x's report is P16x's exactly.
- **midoffice16's check** stands behind it: all 686 refits had the full fit's factor returns
  through their dates, and lambda_F as of the last date was 0.964025715108432 both ways.

## Rulings 1, 2, 3 and 5

Nothing to build for 1, 2 and 5: the correlations stay on 504, item 1 is ruling 2's test, and this
piece is ruling 5's order. Ruling 3, the exact diagonal: recorded in the state file's next steps,
to be folded into the first default change that moves every covariance anyway.

---

## Item 1: the one-sided eigenvalue adjustment on the 63-day correlations (E17)

**The run:** `--correlation-half-life 63 --bias --bias-eigen-scales 1,2,3 --eigen-one-sided
--bias-horizons 1,5,11,21 --bias-guard 0`.
- **Time:** 16.0 minutes, 1.26 s a refit with the simulation (0.51 s of it) and four tests.
- **Its model test** is C16's 63-day setting to the bit in every array -- forecasts, outcomes,
  covariance, the optimiser's exposures, the regime history (`equal_E17_C16.txt`).
- **The simulation finds more to correct on the 63-day matrix,** as ruling 1 said: lambda at
  ranks 21 to 25 is about 1.17 on average, against about 1.06 on the 504-day matrix (VE15), rank
  20 at 1.178 against 1.076.

**Judged by the rule against D16** (`judge.py`, written before the run; `judge_E17.txt`).

The optimiser's realised volatility first, % a year, whole / first half / second half:

| | long-only | unconstrained | |
|---|---|---|---|
| D16, the reference | 10.13 / 11.88 / 8.02 | 10.19 / 11.95 / 8.08 | |
| 63 days, a = 0 | 10.12 / 12.10 / 7.67 | **10.23** / 12.18 / 7.83 | out |
| a = 1 | **10.18** / 12.27 / 7.54 | **10.28** / 12.32 / 7.73 | out |
| a = 2 | **10.28** / 12.47 / 7.51 | **10.36** / 12.47 / 7.72 | out |
| a = 3 | **10.40** / 12.66 / 7.52 | **10.46** / 12.61 / 7.75 | out |

**Every setting is out, and D16 stays.** The adjustment makes the optimiser realise more, and
more at every a: against D16, the first half up by 0.4 to 0.8 of a point (long-only, a = 1 to 3),
the calm half down by 0.3 to 0.5.

Then the criterion (whole / first / second) and the rest the brief asks for beside it:

| | regime criterion | broad portfolios' MRAD | rank 1 | ranks 21-25 | the optimiser's ranks 16-25, long-only: share / statistic |
|---|---|---|---|---|---|
| D16 | 0.105 / **0.098** / 0.123 | 0.094 / 0.068 / 0.159 | 0.946 / 1.064 / 0.814 | 1.195 / 1.124 / 1.261 | 20.6% / 1.17 |
| 63, a = 0 | 0.104 / 0.116 / 0.083 | 0.077 / 0.056 / 0.113 | 0.964 / 1.059 / 0.861 | 1.303 / 1.231 / 1.371 | 23.8% / 1.25 |
| a = 1 | 0.105 / 0.118 / 0.088 | 0.073 / 0.066 / 0.102 | 0.993 / 1.088 / 0.889 | 1.152 / 1.079 / 1.221 | 23.5% / 1.11 |
| a = 2 | 0.107 / 0.125 / 0.091 | 0.074 / 0.080 / 0.096 | 1.018 / 1.114 / 0.914 | 1.037 / 0.964 / 1.105 | 22.7% / 1.01 |
| a = 3 | 0.111 / 0.134 / 0.092 | 0.079 / 0.096 / 0.092 | 1.042 / 1.138 / 0.938 | 0.946 / 0.875 / 1.012 | 21.7% / 0.92 |

(The optimiser's statistic in ranks 16-25 is over the whole test. Unconstrained, a = 0 to 3: shares
28.2%, 24.8%, 21.9%, 19.4%, statistics 1.25, 1.12, 1.01, 0.93.)

- **It does what ruling 1 said it would do in the statistics.** The small eigen-directions come
  to one: ranks 21-25 1.30 to 1.04 at a = 2, the optimiser's part in ranks 16-25 1.25 to 1.01.
- **It does not take the price back in realised risk.**
  - **The shift:** with the small eigenvalues raised, the optimiser moves its forecast risk out
    of them and into the market direction -- 37.9% of its long-only forecast variance to 44.0%
    at a = 3.
  - **The first half's cost:** in that volatile half the portfolios realise more, 12.10% to
    12.66% a year.
  - **The calm half's gain:** they realise a little less, 7.67% to 7.52%.
- **The re-levelled multiplier moves the market direction.** Each scale reads the regime
  adjustment from its own forecasts, whose factor volatilities the adjustment raises, so its
  multiplier is lower.
  - **Calm half:** rank 1 gains, 0.861 to 0.938 at a = 3.
  - **First half:** it loses, 1.059 to 1.138.
  - **The broad portfolios:** their first-half MRAD worsens (0.056 to 0.096) as their second half
    improves (0.113 to 0.092).
- **The criterion:** every scale is worse than D16 on the first half (0.118 to 0.134 against
  0.098) and better on the second (0.088 to 0.092 against 0.123) -- the halves pulling apart again,
  as at 63 days alone.

---

## Item 2: the volatilities' half-life on the regime-adjusted model (V17)

**Built:** `--bias-volatility-half-lives H,H`, the correlation pass's machinery generalised.
- **Each setting:** each refit's own factor returns re-estimated by the model's estimator with
  `volatility_half_life` replaced (the correlations' 504 kept), keyed `_v<h>`.
- **Its regime history:** each setting reads the regime adjustment from its own forecasts -- now
  in earnest, since the volatilities set B_F.
- **The report** (`_report_settings`, midoffice16's correlation report generalised): each
  setting's regime multiplier by part, then C16's tables.
- **The rule:** `COVARIANCE_RULE` (midoffice16's `CORRELATION_RULE`, the same rule, renamed for
  either half-life).
- **One pass at a time:** either half-life pass alone, refused beside the other passes.
- **Checked:** on the last 30 forecast dates the one-pass test's 42-day setting equals a run with
  `--half-life 42` alone to the bit in every array (`smoke/equal_v42.txt`), and a test pins the
  library half.

**The run:** `--bias --bias-volatility-half-lives 21,42,63 --bias-horizons 1,5,11,21 --bias-guard
0`; 10.1 minutes, 0.72 s a refit with four tests. Its model test is D16's to the bit in every array
(`equal_V17_D16.txt`).

**Each setting's regime multiplier, from its own forecasts** -- the mean of B_F^2, and lambda_F on
21 days, its mean (min to max):

| volatilities | whole | first half | second half |
|---|---|---|---|
| 21 | 1.092, 1.037 (0.824-1.639) | 1.175, 1.080 (0.917-1.639) | 1.009, 0.993 (0.824-1.169) |
| 42 | 1.087, 1.031 (0.777-1.788) | 1.236, 1.097 (0.889-1.788) | 0.939, 0.965 (0.777-1.143) |
| 63 | 1.098, 1.033 (0.769-1.889) | 1.286, 1.110 (0.874-1.889) | 0.910, 0.956 (0.769-1.123) |
| 90, the model | 1.115, 1.039 (0.776-1.977) | 1.332, 1.121 (0.864-1.977) | 0.899, 0.956 (0.776-1.129) |

The shorter the window, the more of the regime the volatilities follow themselves, and the less is
left for the multiplier: at 21 days B_F^2 is 1.18 and 1.01 by half, against 1.33 and 0.90.

**Judged by the rule against 90** (`judge_V17.txt`; the report's own tables in the appendix). The
optimiser's realised volatility first, % a year, whole / first half / second half:

| volatilities | long-only | unconstrained | |
|---|---|---|---|
| 21 | **10.37** / 12.31 / 7.99 | **10.61** / 12.42 / 8.43 | out |
| 42 | **10.22** / 12.09 / 7.93 | **10.36** / 12.19 / 8.16 | out |
| 63 | **10.15** / 11.96 / 7.96 | **10.26** / 12.06 / 8.08 | out |
| 90, the model | 10.13 / 11.88 / 8.02 | 10.19 / 11.95 / 8.08 | in |

**Every shorter half-life is out, and 90 stays.** The optimiser realises more the shorter the
window, in the first half most; in the calm half the long-only portfolio realises a little less,
and the unconstrained one the same at 63 days and more at 42 and 21.

The criterion and the rest, whole / first / second:

| volatilities | regime criterion | broad MRAD | optimiser MRAD | rank 1 | ranks 21-25 | eigen mean abs(bias - 1) |
|---|---|---|---|---|---|---|
| 21 | 0.107 / 0.113 / 0.098 | 0.090 / 0.059 / 0.156 | 0.146 / 0.201 / 0.058 | 0.931 / 1.013 / 0.843 | 1.215 / 1.118 / 1.304 | 0.161 / 0.073 / 0.272 |
| 42 | 0.104 / 0.103 / 0.108 | 0.090 / 0.061 / 0.150 | 0.136 / 0.169 / 0.090 | 0.942 / 1.040 / 0.834 | 1.194 / 1.116 / 1.266 | 0.144 / 0.068 / 0.245 |
| 63 | 0.104 / 0.099 / 0.117 | 0.092 / 0.065 / 0.153 | 0.135 / 0.157 / 0.112 | 0.945 / 1.054 / 0.824 | 1.192 / 1.119 / 1.260 | 0.136 / 0.066 / 0.232 |
| 90 | 0.105 / **0.098** / 0.123 | 0.094 / 0.068 / 0.159 | 0.135 / 0.150 / 0.125 | 0.946 / 1.064 / 0.814 | 1.195 / 1.124 / 1.261 | 0.131 / 0.067 / 0.222 |

- **The calm half's market direction barely moves:** rank 1 0.814 to 0.843 at 21 days, the most
  any volatility setting gives -- what item 3 said the volatility side holds.
- **The eigenportfolios' shape gets worse,** not better: their mean abs(bias - 1) in the calm half
  0.222 to 0.272 at 21. A shorter volatility window follows each factor's own level, not the
  correlations.
- **The halves trade again:** the criterion worse on the first half at every setting (0.099 to
  0.113 against 0.098) and better on the second (0.098 to 0.117 against 0.123); the optimiser's
  MRAD the same way, more steeply.
- **The members** barely move: the largest quintile 1.028 at every setting, the smallest three
  quintiles 0.94 to 0.99.
- **The noise:** the index's forecast moves 1.09%, 1.25% and 1.66% a day at the median at 63, 42
  and 21 days, against 1.00%.
- **The optimiser's directions** shift a little from the market direction to the middle ranks
  (long-only rank 1 41.6% to 39.9% at 21), its specific share up (16.2% to 17.1%). Its part in
  ranks 16-25 reads 1.13-1.15 against 1.17.

---

## Item 3: what the calm half's market direction is made of

No run: D16's stored arrays (each forecast date's factor covariance as its test used it, the
regime adjustment in it) and the model's factor returns (M16's full fit, the bias test's
outcomes). `market_direction.py`, `market_direction.txt` in full in the appendix. The outcomes
check against the test's own: rank 1's return recomputed from the factor returns matches its test's
to 1.1e-13.

**The correlations, forecast against realised:**

| part | mean correlation between factors, forecast / realised | the market direction's share of the factor variance, forecast / realised |
|---|---|---|
| whole | 0.417 / 0.416 | 0.615 / 0.604 |
| first half | 0.402 / 0.488 | 0.593 / 0.709 |
| second half | 0.432 / **0.316** | 0.636 / **0.472** |

By calendar quarter the realised correlations rose to 0.55 in 2025 Q2, the shock, and fell to 0.25
in Q3, the calm, staying near 0.30 to 0.33 through 2026. The model's forecast held at 0.40 to 0.46
throughout; C16's 63-day setting followed about a quarter late (0.481 in 2025 Q3, 0.381, 0.366,
0.338 and 0.315 through 2026, against realised 0.330, 0.318, 0.296 and 0.288).

**Each factor's own statistic, the market direction's heaviest first** (their shares of its
variance, about 2% to 9% each, the industries; the style factors carry almost none of it):

| factor | share, 2nd half | whole / first / second |
|---|---|---|
| SIC60 (banks) | 0.088 | 0.994 / 1.057 / 0.928 |
| SIC62 (brokers) | 0.083 | 1.017 / 1.055 / 0.979 |
| SIC36 (electronics) | 0.076 | 1.017 / 1.052 / 0.982 |
| SIC35 (machinery) | 0.072 | 1.002 / 1.081 / 0.917 |
| SIC37 (transport equipment) | 0.064 | 1.015 / 1.096 / 0.929 |
| ... the 20 industries | | the second half from 0.80 (Manufacturing) to 1.04 (SIC49) |
| weighted by the market direction's shares | | 0.994 / 1.038 / **0.945** |

**Rank 1 rebuilt on each side** (its statistic, the forecast's leading eigenportfolio against its
next-day return):

| part | as forecast | (a) the realised correlations | (b) each volatility at its realised level | (c) both |
|---|---|---|---|---|
| whole | 0.946 | 0.947 | 0.952 | 0.953 |
| first half | 1.064 | 0.976 | 1.025 | 0.940 |
| second half | **0.814** | **0.946** | 0.861 | 1.000 |

**So, to the brief's question: it is not the volatility side.** What is left of the calm half's
market direction is mostly the correlations.
- **The correlations:** they fell after the shock -- realised 0.316 against the forecast's 0.432 --
  and the market direction's share of the variance with them, 0.47 against 0.64. Rebuilt on the
  realised correlations, rank 1 goes from 0.814 to 0.946.
- **The volatilities:** the market direction's industries were over-forecast by about 5% (0.945
  weighted). With each at its realised level, rank 1 goes only to 0.861.
- **midoffice16's reading was wrong.** It took the 63-day setting's small gain -- 6% off the market
  direction's forecast volatility -- as a sign that the correlations were not the lever. They are;
  a 63-day window follows their fall about a quarter late, and the gain was the part it caught.
  The docs' paragraph says so now.
- **The first half is the mirror:** the correlations rose (0.488 against 0.402), and on them rank
  1's 1.064 goes to 0.976.

---

## Item 4: the combination

Neither lever passed the rule, so item 4's case -- both passing -- does not arise, and there is
nothing to combine. For the record, the test a combination would have needed:
- **The pass:** one run on the combined model, the correlations at the passing half-life with the
  volatilities' setting beside it: `--correlation-half-life H --bias-volatility-half-lives H,H`.
- **Its model test:** the correlation setting's own, to the bit.
- **Judged:** by the same rule against D16, the optimiser's realised volatility first.

On this evidence I would not run it: each lever failed alone on the optimiser's realised risk, in
the same half.

---

## Which I would choose, and why

1. **The factor covariance as it stands:** correlations on 504 days, volatilities on 90, the
   regime adjustment's factor side on 21, no eigenvalue adjustment. Both levers fail the rule
   written before the runs: every setting makes the optimiser realise more over the whole test --
   the eigen scales on the 63-day correlations by 0.05 to 0.27 of a point a year, the shorter
   volatility windows by 0.02 to 0.42.
2. **The calm half's shape stays, understood.** Item 3 says what it is:
   - **The correlations:** they fell after the shock and stayed down, and the forecast's window
     does not follow -- not the volatilities, as I read it in midoffice16.
   - **The levers:** every one tried that follows the correlations faster -- the 63-day window,
     the eigenvalue adjustment on it, and indirectly the shorter volatility windows -- buys the
     calm half with the shock half. The optimiser pays in the shock, where risk matters most.
3. **The lever that would read the market direction on its own** -- a regime multiplier for the
   market direction alone, from its own out-of-sample record, the small directions left on the
   long window where the optimiser cannot find their noise -- is the design the evidence now
   points to. It is new, not a setting, so it is Jeff's to want or not; I would not build it
   before the specific side.
4. **Next, the largest members' specific risk,** as ruling 5 orders it: the factor side is settled
   at the defaults, so the specific side is measured on the factor side it will keep. The top
   quintile is 1.028 under the defaults, and the optimiser's specific part reads 0.87 to 0.91 at
   the other end.
5. **Ruling 4's plain run:** 0.9 minutes, its forecast the refits' to the bit, the guard checking
   five refits each run.

---

## Key decisions

1. **Ruling 4's history in the library** (`factor_history`, `factor_history_guard`), not in the
   run alone, so the test pins it on a synthetic panel: the one-fit history equals the refits' to
   the bit, and the guard names a one-bit difference.
2. **The guard's five dates as the bias guard picks them** -- the first forecast date, the last
   and three between -- checking the refit's estimated periods, factor returns and factor
   volatilities, in that order; a fixed count (`REGIME_GUARD`), not an option, since it costs
   three seconds.
3. **The fast path only where it is exact:** the specific side on, the sampling correction and the
   eigenvalue adjustment keep the refits, by an explicit condition and by the library's refusal.
4. **A factor-only history says so:** the specific side nan, `has_specific` false, its multiplier
   refused; the report prints "the specific side not read" where it printed B_S^2.
5. **Item 2 by generalising midoffice16's pass,** not by a second copy: one machinery for either
   half-life, keyed c or v, one report, one rule (`COVARIANCE_RULE`, `CORRELATION_RULE` kept as
   its other name); the two passes exclusive, one thing tested at a time.
6. **The eigen scales' covariances kept in `--bias-out`,** so item 1's optimiser could be read by
   eigen-direction from the run's arrays.
7. **Items 1 and 2 judged by a script written before the runs** (`judge.py`): item 1's reference is
   D16, which the run itself does not contain. The script reads every test through the run's own
   `_reading`, and reads the factor returns once, from the model test, for every test's
   directions.
8. **Item 3 from the stored arrays,** checked first against the test's own outcomes (rank 1's to
   1.1e-13). It attributes rank 1's over-forecast by rebuilding the forecasts on each side --
   correlations, volatilities, both -- rather than by reading the factors one at a time.
9. **The memory bar kept** while the machine's other applications held free memory under it for
   42 minutes; one notification, no lowering.
10. **No default changed beyond ruling 4,** and ruling 4 changes no number.

## Decisions left for Jeff

1. **The factor covariance:** as it stands (my choice) -- or any of the settings measured here,
   each of which fails the rule.
2. **The calm half's market direction:**
   - **left for now** (my choice);
   - **or a later piece** on a regime multiplier of its own for the market direction, the small
     directions left on the long window.
3. **The next piece:** the largest members' specific risk (my choice, ruling 5's order).

---

## Tests

**The suite:** 2,375 passed, 1 skipped, with `--ignore=tests/test_market_data.py`. Before this
piece 2,374: 1 new, 1 extended, and every other passes unchanged -- the plain run's new path changes no number,
and the new options default to off.

**New, 1 (`test_regime.py`):**
`test_the_factor_side_read_from_one_fit_is_the_refits_and_its_guard_names_a_difference` -- ruling
4's footing.
- **The history:** on the synthetic panel, the factor side read from one fit equals the history
  the model refitted as of every day gives, to the bit -- B_F^2, the factor counts, the
  multiplier's path.
- **The guard:** it finds nothing on true refits; it names a refit whose factor returns differ
  in one bit, and every refit read against an estimator not the model's.
- **The refusals:** a factor-only history refuses the specific side's multiplier, and the
  library refuses a model whose covariance is the eigenvalue adjustment's.

**Extended, 1 (`test_bias.py`):**
`test_a_covariance_reestimated_from_the_factor_returns_is_the_fit_with_that_estimator` -- a
covariance re-estimated with another volatility half-life is the fit with it, to the bit; its
volatilities move and its correlations do not (item 2's footing, as the correlations' was
midoffice16's).

**Measured, not unit-tested:**
- **The smoke runs:** the volatility pass's 42-day setting a run's at 42 alone, to the bit; the
  eigen pass's model test midoffice16's 63-day smoke setting; the plain run's guard and history.
- **The full runs:** E17's model test C16's 63-day setting, and V17's D16's, to the bit; P17's
  forecast P16's.

**Hygiene:** every changed file parses, is ASCII and keeps its line endings
(`final_checks.txt`). **ruff:** clean over every changed Python file.

---

## Failures and debugging trail

- **A plural:** the first volatility smoke run reported "the factor volatilitys' half-life", the
  kind's name with an "s". One map of the plurals (`PLURAL`) fixed both places before the full
  runs.
- **An import order:** the new test's imports unsorted (ruff I001), fixed.
- **A patch that did not apply:** the first extension of `market_direction.py` matched its
  docstring with one space too many and stopped on its assertion; the second, written to a file,
  applied.
- **The memory bar:** the runner waited for 3 GB three times -- 13:01 to 13:03 before E17, 13:19
  to 14:01 before V17 (42 minutes, free memory falling as low as 0.6 GB; one notification sent),
  and 14:11 to 14:16 before P17x -- the machine's other applications holding free memory under it.
  It waited as designed; a working note now says so. One of this session's Git Bash calls crashed
  in the squeeze and left `bash.exe.stackdump` in the repo root (13:48, a 1.2 KB msys stack trace),
  removed.
- **The runs:** none failed. E17 16.0 minutes, V17 10.1, P17x 0.8, P17 0.9 (`bias/runner_log.txt`),
  each started with at least 3.1 GB available.

## Problems found, not fixed

1. **The calm half's market direction is over-forecast for its correlations.** They fell abruptly
   after the April 2025 shock (realised 0.25 between factors in 2025 Q3) and stayed near 0.30. The
   504-day forecast held at 0.40-0.46, and a 63-day window followed a quarter late. No lever
   measured here reaches it without the optimiser paying in the volatile half.
2. **The halves pull apart for every lever tried:** the 63-day correlations, the eigenvalue
   adjustment on them, and the shorter volatility windows, are better in the calm half and worse in
   the shock.
3. **The eigenvalue adjustment fixes the small eigen-directions' statistics and not the
   optimiser's realised risk** -- for the third time (midoffice13 two-sided; VE15 and E17
   one-sided).
4. **midoffice16's "volatility side" reading was wrong;** corrected in the docs and here.
5. **As before:** the two-half-life diagonal's last bit (ruled to wait); the largest members'
   specific risk (next); FedEx Freight and Vylor without Momentum; six ruff findings in untouched
   files at HEAD; `tests/test_market_data.py`'s `../core_repo/src`.

## Scratch

`claude/scratch/midoffice17/` (gitignored):

- **The patches, by script:** `patch_regime_section.py` (ruling 4), `patch_settings.py`,
  `patch_settings_args.py`, `patch_plural.py` (item 2), `patch_state_*.py` (the state file),
  `extend_market_direction.py`.
- **The runs:**
  - `runner.py`;
  - `bias/`: E17 and V17, each a report, progress and `.npz`; P17x and P17, each a report and
    progress; the runner's log;
  - `smoke/`: the four smoke runs and their checks -- `equal_v42.txt`, `equal_e_c63.txt`.
- **The analyses:**
  - `judge.py`, written before the runs, and `judge_E17.txt`, `judge_V17.txt`;
  - `market_direction.py` and `.txt` -- item 3;
  - `equal_E17_C16.txt`, `equal_V17_D16.txt` (midoffice16's `equal_correlation.py`);
  - `plain_cost.txt`.
- **The checks:** `final_checks.sh` and `.txt`, `final/`; the retro's drafts (`retro_*.md`),
  `fill_retro.py` and `append_appendix.py`.

The midoffice13 to 16 harnesses were used where they are: `capture.py`, `compare.py`,
`hygiene.py` (13), `compare_panels.py` (14), `smoke_last.py` (15), `equal_correlation.py` (16).
The key outputs are copied into the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

---

## Open items for Chat

- **The three decisions above,** the factor covariance first.
- **The next piece,** as I would order it: the largest members' specific risk, by size, with the
  optimiser's specific part beside it; the calm half's market direction, on its own, after, if
  wanted.
- **The bias statistics are standing,** with `judge.py` to judge a run against D16 by the rule;
  every candidate is one run against D16, through the runner.

---

## Appendix: outputs, verbatim

Each output as its script wrote it, from `claude/scratch/midoffice17/`.

### smoke/report_plain.txt -- ruling 4: the plain run's regime lines on the last 30 forecast dates, the factor side read from the full fit behind its guard

```text
The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2026-09-23 to 2026-11-03, 30 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
  as of 2026-11-03: lambda_F 0.9604, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.61%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
  the last day's B_F^2 0.424 over 25 factors, the specific side not read
  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 30 days, 0.0 s; guard: as of 2026-09-22, 2026-10-01, 2026-10-12, 2026-10-22, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (6.2 s)
```

### smoke/equal_v42.txt -- item 2: the one-pass test's 42-day volatility setting against a run at 42 days alone, the last 30 forecast dates, bit for bit

```text
claude/scratch/midoffice17/smoke/v42.npz's model test against claude/scratch/midoffice17/smoke/v.npz's v42: 30 forecast dates, 2026-09-22 to 2026-11-02
  forecast                 against forecast_v42                       equal to the bit
  realised                 against realised_v42                       equal to the bit
  member_forecast          against member_forecast_v42                equal to the bit
  eigen_forecast           against eigen_forecast_v42                 equal to the bit
  eigen_realised           against eigen_realised_v42                 equal to the bit
  repaired                 against repaired_v42                       equal to the bit
  held_long_only           against held_long_only_v42                 equal to the bit
  held_unconstrained       against held_unconstrained_v42             equal to the bit
  factor_covariance        against factor_covariance_v42              equal to the bit
  exposure_long_only       against exposure_long_only_v42             equal to the bit
  exposure_unconstrained   against exposure_unconstrained_v42         equal to the bit
  realised_h5              against realised_h5_v42                    equal to the bit
  eigen_realised_h5        against eigen_realised_h5_v42              equal to the bit
  regime_dates             against regime_dates_v42                   equal to the bit
  regime_factor_squared    against regime_factor_squared_v42          equal to the bit
  regime_specific_squared  against regime_specific_squared_v42        equal to the bit
  regime_members           against regime_members_v42                 equal to the bit
  regime_effective         against regime_effective_v42               equal to the bit
  member_realised          (shared)                            equal to the bit
  caps                     (shared)                            equal to the bit
  member_realised_h5       (shared)                            equal to the bit
every array equal to the bit
```

### smoke/equal_e_c63.txt -- item 1: the eigen pass's model test on the 63-day correlations against midoffice16's 63-day smoke setting, bit for bit

```text
claude/scratch/midoffice17/smoke/e.npz's model test against claude/scratch/midoffice16/smoke/c.npz's c63: 30 forecast dates, 2026-09-22 to 2026-11-02
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

### market_direction.txt -- item 3: what the calm half's market direction is made of

```text
D16: 686 forecast dates, 2024-02-08 to 2026-11-02, the outcomes 2024-02-09 to 2026-11-03; the halves 343 and 343 (the second from 2025-06-24). The factor returns: M16's full fit, 938 periods, 25 factors; outcome days without one: 0
  check: rank 1's outcome from the factor returns against its test's, the largest relative difference 1.1e-13

1. The correlations, forecast (the mean over the part's dates) against realised (the sample over the part): the mean correlation between factors, and the market direction's share of the factor variance
                          mean correlation                market share
                    forecast      realised      forecast      realised
  whole                0.417         0.416         0.615         0.604
  first half           0.402         0.488         0.593         0.709
  second half          0.432         0.316         0.636         0.472
  by calendar quarter, the mean correlation forecast / realised: 2024 Q1 0.38/0.40; 2024 Q2 0.38/0.37; 2024 Q3 0.39/0.47; 2024 Q4 0.39/0.38; 2025 Q1 0.40/0.48; 2025 Q2 0.47/0.55; 2025 Q3 0.46/0.25; 2025 Q4 0.45/0.42; 2026 Q1 0.43/0.33; 2026 Q2 0.42/0.32; 2026 Q3 0.41/0.30; 2026 Q4 0.40/0.29

2. Each factor's own statistic -- its next-day return over its forecast volatility -- by part, the market direction's heaviest factors first: its share of the market direction's variance (the mean of v_1k^2) in each half, its forecast volatility at the median and its realised volatility, % a year, in the second half
  factor                   weight 1st    2nd   whole   first  second  fcst vol  real vol
  SIC60                         0.080  0.088   0.994   1.057   0.928    24.23%    21.94%
  SIC62                         0.067  0.083   1.017   1.055   0.979    22.07%    20.74%
  SIC36                         0.092  0.076   1.017   1.052   0.982    22.31%    21.68%
  SIC35                         0.068  0.072   1.002   1.081   0.917    19.67%    17.48%
  SIC37                         0.059  0.064   1.015   1.096   0.929    19.87%    18.72%
  Finance & Real Estate         0.051  0.055   0.994   1.050   0.933    18.29%    16.70%
  SIC38                         0.043  0.052   0.991   1.037   0.944    16.56%    15.73%
  SIC73                         0.057  0.050   0.990   1.040   0.938    17.52%    16.07%
  Mining                        0.048  0.048   0.989   1.037   0.939    21.08%    19.48%
  Transport & Utilities         0.044  0.048   0.978   1.049   0.903    17.12%    15.26%
  Construction                  0.066  0.045   1.001   0.975   1.028    21.40%    21.93%
  SIC67                         0.051  0.042   0.975   0.975   0.977    16.97%    16.39%
  Services                      0.047  0.041   0.949   1.022   0.871    15.35%    13.46%
  Manufacturing                 0.042  0.040   0.931   1.045   0.802    14.83%    11.69%
  Retail                        0.042  0.040   0.976   1.018   0.934    15.14%    14.01%
  SIC49                         0.033  0.036   1.010   0.983   1.036    17.13%    17.17%
  Wholesale                     0.031  0.033   1.014   1.041   0.987    15.50%    14.65%
  SIC28                         0.027  0.030   1.017   1.038   0.991    15.19%    14.72%
  SIC63                         0.026  0.028   0.971   1.033   0.907    15.65%    14.21%
  SIC20                         0.022  0.021   0.958   0.967   0.950    14.57%    13.44%
  Volatility                    0.005  0.008   1.120   1.113   1.128     7.98%     9.17%
  Size Nonlinearity             0.000  0.000   1.071   0.992   1.146     2.61%     2.84%
  Trading Activity              0.000  0.000   1.127   1.051   1.200     4.00%     4.94%
  Momentum                      0.000  0.000   1.071   0.985   1.153     6.55%     7.46%
  Size                          0.000  0.000   1.063   0.940   1.174     3.16%     3.51%
  whole: the statistic weighted by the market direction's shares 0.994; its heaviest five SIC36, SIC60, SIC62, SIC35, SIC37 (37% of it)
  first half: the statistic weighted by the market direction's shares 1.038; its heaviest five SIC36, SIC60, SIC35, SIC62, Construction (37% of it)
  second half: the statistic weighted by the market direction's shares 0.945; its heaviest five SIC60, SIC62, SIC36, SIC35, SIC37 (38% of it)

3. Rank 1's statistic -- the forecast's leading eigenportfolio, its next-day return over its forecast volatility -- as forecast, and with each part's forecasts rebuilt
  whole: as forecast 0.946; (a) the realised correlations 0.947; (b) each volatility at its realised level 0.952; (c) both 0.953
  first half: as forecast 1.064; (a) the realised correlations 0.976; (b) each volatility at its realised level 1.025; (c) both 0.940
  second half: as forecast 0.814; (a) the realised correlations 0.946; (b) each volatility at its realised level 0.861; (c) both 1.000

4. The forecasts' mean correlation between factors by calendar quarter -- the model's (504 days), the 63-day setting's (C16) -- against the realised over the quarter's outcomes
  quarter    days     504      63  realised
  2024 Q1      35   0.379   0.369     0.397
  2024 Q2      63   0.382   0.379     0.373
  2024 Q3      64   0.390   0.403     0.472
  2024 Q4      64   0.392   0.399     0.381
  2025 Q1      60   0.399   0.418     0.483
  2025 Q2      62   0.466   0.532     0.549
  2025 Q3      64   0.457   0.481     0.246
  2025 Q4      64   0.448   0.442     0.416
  2026 Q1      61   0.433   0.381     0.330
  2026 Q2      62   0.422   0.366     0.318
  2026 Q3      64   0.408   0.338     0.296
  2026 Q4      23   0.399   0.315     0.288
  whole: the 63-day forecasts' mean correlation 0.408, the market direction's share 0.600
  first half: the 63-day forecasts' mean correlation 0.419, the market direction's share 0.605
  second half: the 63-day forecasts' mean correlation 0.398, the market direction's share 0.594
```

### equal_E17_C16.txt -- item 1: E17's model test against C16's 63-day setting, bit for bit

```text
claude/scratch/midoffice17/bias/E17.npz's model test against claude/scratch/midoffice16/bias/C16.npz's c63: 686 forecast dates, 2024-02-08 to 2026-11-02
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
  realised_h11             against realised_h11_c63                   equal to the bit
  eigen_realised_h11       against eigen_realised_h11_c63             equal to the bit
  realised_h21             against realised_h21_c63                   equal to the bit
  eigen_realised_h21       against eigen_realised_h21_c63             equal to the bit
  regime_dates             against regime_dates_c63                   equal to the bit
  regime_factor_squared    against regime_factor_squared_c63          equal to the bit
  regime_specific_squared  against regime_specific_squared_c63        equal to the bit
  regime_members           against regime_members_c63                 equal to the bit
  regime_effective         against regime_effective_c63               equal to the bit
  member_realised          (shared)                            equal to the bit
  caps                     (shared)                            equal to the bit
  member_realised_h5       (shared)                            equal to the bit
  member_realised_h11      (shared)                            equal to the bit
  member_realised_h21      (shared)                            equal to the bit
every array equal to the bit
```

### bias/report_E17.txt -- item 1: E17's report, the eigenfactor adjustment one-sided on the 63-day correlations

```text
  The eigenfactor risk adjustment (Menchero, Wang and Orr; USE4) at one day: each eigenvalue of the day's factor covariance scaled by max(1, a (lambda - 1) + 1)^2, one-sided, the eigenportfolios tested in the unadjusted covariance's eigenbasis; the simulation 0.51 s a refit (0.33 to 0.73); on the model's volatility regime adjustment, F  21 S   -, each scale's factor multiplier from the history of its own forecasts' bias statistics
  lambda by rank over the 686 dates, the mean (min to max): 1: 1.002 (0.996-1.007), 2: 0.954 (0.921-0.994), 3: 0.968 (0.935-1.016), 5: 1.013 (0.941-1.068), 10: 1.021 (0.970-1.075), 15: 1.088 (1.059-1.122), 20: 1.178 (1.117-1.256), 21: 1.166, 22: 1.169, 23: 1.177, 24: 1.174, 25: 1.164; its Monte Carlo standard error at most 0.0023, 0.0020 at the median date's worst rank

  the whole test (2024-02-08 to 2026-11-02, 686), by a (0 the model):
                                                       a = 0     a = 1     a = 2     a = 3
  eigenportfolios 1-5, mean                            0.993     1.020     1.043     1.065
  eigenportfolios 6-10, mean                           1.026     1.049     1.070     1.088
  eigenportfolios 11-15, mean                          1.108     1.079     1.052     1.026
  eigenportfolios 16-20, mean                          1.254     1.129     1.032     0.953
  eigenportfolios 21-25, mean                          1.303     1.152     1.037     0.946
  eigenportfolios, mean abs(bias - 1)                  0.145     0.089     0.048     0.057
  minimum variance, long-only                          1.044     1.028     1.023     1.024
  minimum variance, unconstrained                      1.105     1.086     1.081     1.084
  minimum variance, mean abs(bias - 1)                 0.075     0.057     0.052     0.054
  minimum variance, long-only: realised vol, %        10.124    10.175    10.285    10.403
  minimum variance, unconstrained: realised vol, %    10.232    10.276    10.360    10.457
  the criterion                                        0.110     0.073     0.050     0.055
  the index, cap-weighted                              0.953     0.979     1.002     1.023
  the index, equally weighted                          0.940     0.940     0.935     0.927
  smallest 85, equally weighted                        0.952     0.938     0.920     0.899
  largest 50, equally weighted                         0.936     0.960     0.980     0.999
  members, median                                      0.992     0.987     0.980     0.972
  members, mean abs(bias - 1)                          0.086     0.089     0.092     0.096
  industries, mean abs(bias - 1)                       0.048     0.050     0.056     0.063
  random long-only, median                             0.951     0.949     0.947     0.937
  random long-short, median                            0.994     0.990     0.985     0.980
  broad portfolios, mean MRAD                          0.077     0.073     0.074     0.079
  minimum variance, mean MRAD                          0.150     0.153     0.157     0.161
  members, median MRAD                                 0.084     0.088     0.089     0.092
  the regime criterion                                 0.104     0.105     0.107     0.111

  the first half (2024-02-08 to 2025-06-23, 343), by a (0 the model):
                                                       a = 0     a = 1     a = 2     a = 3
  eigenportfolios 1-5, mean                            0.988     1.010     1.030     1.048
  eigenportfolios 6-10, mean                           0.967     0.990     1.010     1.028
  eigenportfolios 11-15, mean                          1.031     1.001     0.972     0.945
  eigenportfolios 16-20, mean                          1.136     1.021     0.931     0.859
  eigenportfolios 21-25, mean                          1.231     1.079     0.964     0.875
  eigenportfolios, mean abs(bias - 1)                  0.106     0.055     0.050     0.088
  minimum variance, long-only                          1.163     1.159     1.163     1.171
  minimum variance, unconstrained                      1.237     1.228     1.231     1.239
  minimum variance, mean abs(bias - 1)                 0.200     0.194     0.197     0.205
  minimum variance, long-only: realised vol, %        12.104    12.267    12.470    12.658
  minimum variance, unconstrained: realised vol, %    12.182    12.320    12.467    12.606
  the criterion                                        0.153     0.124     0.124     0.146
  the index, cap-weighted                              1.061     1.087     1.111     1.133
  the index, equally weighted                          1.020     1.021     1.015     1.006
  smallest 85, equally weighted                        0.979     0.963     0.943     0.920
  largest 50, equally weighted                         1.018     1.042     1.062     1.080
  members, median                                      0.989     0.978     0.972     0.961
  members, mean abs(bias - 1)                          0.098     0.100     0.103     0.107
  industries, mean abs(bias - 1)                       0.055     0.061     0.067     0.075
  random long-only, median                             1.016     1.014     1.009     0.998
  random long-short, median                            0.980     0.972     0.962     0.953
  broad portfolios, mean MRAD                          0.056     0.066     0.080     0.096
  minimum variance, mean MRAD                          0.212     0.205     0.209     0.217
  members, median MRAD                                 0.080     0.082     0.087     0.090
  the regime criterion                                 0.116     0.118     0.125     0.134

  the second half (2025-06-24 to 2026-11-02, 343), by a (0 the model):
                                                       a = 0     a = 1     a = 2     a = 3
  eigenportfolios 1-5, mean                            0.994     1.025     1.052     1.077
  eigenportfolios 6-10, mean                           1.082     1.106     1.126     1.145
  eigenportfolios 11-15, mean                          1.177     1.150     1.123     1.098
  eigenportfolios 16-20, mean                          1.359     1.226     1.121     1.036
  eigenportfolios 21-25, mean                          1.371     1.221     1.105     1.012
  eigenportfolios, mean abs(bias - 1)                  0.208     0.154     0.112     0.079
  minimum variance, long-only                          0.911     0.879     0.862     0.853
  minimum variance, unconstrained                      0.956     0.924     0.909     0.905
  minimum variance, mean abs(bias - 1)                 0.066     0.099     0.115     0.121
  minimum variance, long-only: realised vol, %         7.666     7.542     7.506     7.517
  minimum variance, unconstrained: realised vol, %     7.828     7.729     7.717     7.753
  the criterion                                        0.137     0.126     0.114     0.100
  the index, cap-weighted                              0.833     0.858     0.880     0.900
  the index, equally weighted                          0.853     0.854     0.849     0.841
  smallest 85, equally weighted                        0.925     0.913     0.897     0.878
  largest 50, equally weighted                         0.848     0.872     0.893     0.912
  members, median                                      0.988     0.983     0.976     0.968
  members, mean abs(bias - 1)                          0.094     0.097     0.100     0.104
  industries, mean abs(bias - 1)                       0.080     0.080     0.082     0.086
  random long-only, median                             0.884     0.882     0.877     0.869
  random long-short, median                            1.008     1.005     1.003     0.999
  broad portfolios, mean MRAD                          0.113     0.102     0.096     0.092
  minimum variance, mean MRAD                          0.056     0.082     0.093     0.096
  members, median MRAD                                 0.081     0.081     0.085     0.089
  the regime criterion                                 0.083     0.088     0.091     0.092

  a chosen on the first half by the criterion -- half the eigenportfolios' mean abs(bias - 1) over every rank, half the minimum-variance portfolios', long-only and unconstrained: a = 2 (0.124; the model's 0.153); on the second half it reads 0.114 against the model's 0.137, the best there a = 3

  a chosen on the first half by the regime criterion -- the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median): a = 0 (0.116; the model's 0.116); on the second half it reads 0.083 against the model's 0.083, the best there a = 0

  a chosen on the first half by the minimum-variance portfolios' realised volatility, the mean of the two: a = 0 (12.14% a year; the model's 12.14%); on the second half 7.75% against the model's 7.75%, the lowest there a = 2
```

### judge_E17.txt -- item 1: every E17 test judged by the rule against D16

```text
The reference: claude/scratch/midoffice16/bias/D16.npz's model test. The run: claude/scratch/midoffice17/bias/E17.npz, its tests the model, _a1, _a2, _a3. 686 forecast dates, 2024-02-08 to 2026-11-02, the halves 343 and 343.
The rule (COVARIANCE_RULE): the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second; the criterion: the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median)

1. The minimum-variance portfolios' realised volatility, % a year, whole / first / second, against the reference's; out if either rounds above it over the whole test
  reference  10.13 / 11.88 /  8.02   10.19 / 11.95 /  8.08
  model      10.12 / 12.10 /  7.67   10.23 / 12.18 /  7.83   out
  _a1        10.18 / 12.27 /  7.54   10.28 / 12.32 /  7.73   out
  _a2        10.28 / 12.47 /  7.51   10.36 / 12.47 /  7.72   out
  _a3        10.40 / 12.66 /  7.52   10.46 / 12.61 /  7.75   out

2. The regime criterion, whole / first / second, and its three MRADs (the broad portfolios', the optimiser's, the members' median)
  reference  criterion 0.105 / 0.098 / 0.123   broad 0.094 / 0.068 / 0.159   optimiser 0.135 / 0.150 / 0.125   members 0.086 / 0.076 / 0.084
  model      criterion 0.104 / 0.116 / 0.083   broad 0.077 / 0.056 / 0.113   optimiser 0.150 / 0.212 / 0.056   members 0.084 / 0.080 / 0.081
  _a1        criterion 0.105 / 0.118 / 0.088   broad 0.073 / 0.066 / 0.102   optimiser 0.153 / 0.205 / 0.082   members 0.088 / 0.082 / 0.081
  _a2        criterion 0.107 / 0.125 / 0.091   broad 0.074 / 0.080 / 0.096   optimiser 0.157 / 0.209 / 0.093   members 0.089 / 0.087 / 0.085
  _a3        criterion 0.111 / 0.134 / 0.092   broad 0.079 / 0.096 / 0.092   optimiser 0.161 / 0.217 / 0.096   members 0.092 / 0.090 / 0.089
  the rule: every test is out on the optimiser's realised volatility; the reference stays

3. Beside it: each test's own eigenportfolios, rank 1 and ranks 21-25 (mean), the statistic whole / first / second
  reference  rank 1 0.946 / 1.064 / 0.814   ranks 21-25 1.195 / 1.124 / 1.261   mean abs(bias - 1) 0.131 / 0.067 / 0.222
  model      rank 1 0.964 / 1.059 / 0.861   ranks 21-25 1.303 / 1.231 / 1.371   mean abs(bias - 1) 0.145 / 0.106 / 0.208
  _a1        rank 1 0.993 / 1.088 / 0.889   ranks 21-25 1.152 / 1.079 / 1.221   mean abs(bias - 1) 0.089 / 0.055 / 0.154
  _a2        rank 1 1.018 / 1.114 / 0.914   ranks 21-25 1.037 / 0.964 / 1.105   mean abs(bias - 1) 0.048 / 0.050 / 0.112
  _a3        rank 1 1.042 / 1.138 / 0.938   ranks 21-25 0.946 / 0.875 / 1.012   mean abs(bias - 1) 0.057 / 0.088 / 0.079

4. The optimiser along its own covariance's eigen-directions: each group's share of the forecast variance (the mean over dates), and the statistic of its realised part, whole / first / second; the factor returns recovered from the run's model test
  minimum variance, long-only
    model    rank 1 37.9% 0.96/1.06/0.86; ranks 2-5 13.7% 0.99/0.95/1.03; ranks 6-15  9.1% 1.07/1.03/1.11; ranks 16-25 23.8% 1.25/1.22/1.29; specific 15.6% 0.87/0.88/0.86
    _a1      rank 1 39.9% 0.99/1.09/0.89; ranks 2-5 12.8% 1.02/0.97/1.06; ranks 6-15  8.4% 1.09/1.04/1.13; ranks 16-25 23.5% 1.11/1.08/1.14; specific 15.4% 0.85/0.86/0.83
    _a2      rank 1 42.0% 1.02/1.11/0.91; ranks 2-5 12.0% 1.04/0.98/1.09; ranks 6-15  7.8% 1.10/1.06/1.14; ranks 16-25 22.7% 1.01/0.99/1.02; specific 15.4% 0.84/0.86/0.81
    _a3      rank 1 44.0% 1.04/1.14/0.94; ranks 2-5 11.3% 1.06/0.99/1.12; ranks 6-15  7.4% 1.11/1.05/1.15; ranks 16-25 21.7% 0.92/0.90/0.94; specific 15.5% 0.83/0.87/0.80
  minimum variance, unconstrained
    model    rank 1 31.1% 0.96/1.06/0.86; ranks 2-5 13.3% 0.99/0.95/1.04; ranks 6-15 11.1% 1.08/1.04/1.11; ranks 16-25 28.2% 1.25/1.23/1.27; specific 16.3% 0.90/0.91/0.90
    _a1      rank 1 33.9% 0.99/1.09/0.89; ranks 2-5 14.0% 1.02/0.96/1.07; ranks 6-15 11.2% 1.09/1.05/1.13; ranks 16-25 24.8% 1.12/1.10/1.13; specific 16.1% 0.90/0.91/0.90
    _a2      rank 1 36.3% 1.02/1.11/0.91; ranks 2-5 14.4% 1.04/0.98/1.10; ranks 6-15 11.2% 1.10/1.05/1.14; ranks 16-25 21.9% 1.01/1.00/1.03; specific 16.2% 0.91/0.91/0.90
    _a3      rank 1 38.2% 1.04/1.14/0.94; ranks 2-5 14.8% 1.07/1.00/1.13; ranks 6-15 11.1% 1.12/1.06/1.16; ranks 16-25 19.4% 0.93/0.92/0.93; specific 16.5% 0.91/0.92/0.91
```

### equal_V17_D16.txt -- item 2: V17's model test against D16, bit for bit

```text
claude/scratch/midoffice16/bias/D16.npz's model test against claude/scratch/midoffice17/bias/V17.npz's model test: 686 forecast dates, 2024-02-08 to 2026-11-02
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

### bias/report_V17.txt -- item 2: V17's report, the volatility half-lives

```text
  The factor volatilities' half-life at one day: the model's 90 (*) and 21, 42, 63 trading days, the correlations on their own half-life throughout, every setting on its own forecasts' volatility regime adjustment, F  21 S   -
  Each setting's regime multiplier, read from its own forecasts' history: the mean of B_F^2, and lambda_F on 21 days -- its mean (min to max) -- by part
                           the whole test                   the first half                  the second half
  21           1.092  1.037 (0.824-1.639)       1.175  1.080 (0.917-1.639)       1.009  0.993 (0.824-1.169)
  42           1.087  1.031 (0.777-1.788)       1.236  1.097 (0.889-1.788)       0.939  0.965 (0.777-1.143)
  63           1.098  1.033 (0.769-1.889)       1.286  1.110 (0.874-1.889)       0.910  0.956 (0.769-1.123)
  90 *         1.115  1.039 (0.776-1.977)       1.332  1.121 (0.864-1.977)       0.899  0.956 (0.776-1.129)

  First, the minimum-variance portfolios' realised volatility, % a year, whole / first half / second half; a setting under which either realises more over the whole test than at 90, to the hundredth of a point, is out
                       long-only           unconstrained
  21       10.37 / 12.31 /  7.99   10.61 / 12.42 /  8.43   out
  42       10.22 / 12.09 /  7.93   10.36 / 12.19 /  8.16   out
  63       10.15 / 11.96 /  7.96   10.26 / 12.06 /  8.08   out
  90 *     10.13 / 11.88 /  8.02   10.19 / 11.95 /  8.08   in

  Then the regime criterion -- the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median) -- by part:
  21       0.107 / 0.113 / 0.098
  42       0.104 / 0.103 / 0.108
  63       0.104 / 0.099 / 0.117
  90 *     0.105 / 0.098 / 0.123
  chosen on the first half by the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second: 90 * (0.098; the model's 0.098); on the second half it reads 0.123 against the model's 0.123, the best there 21

  The shape: each setting's own eigenportfolios -- the eigenvectors of its covariance -- by rank, 1 the largest eigenvalue, the market-like one; the statistic by part
  the whole test (2024-02-08 to 2026-11-02, 686)
  rank                      21      42      63    90 *
  1                      0.931   0.942   0.945   0.946
  2                      0.979   0.982   0.981   0.978
  3                      1.056   1.061   1.059   1.051
  4                      1.056   1.037   1.036   1.033
  5                      1.111   1.086   1.061   1.047
  6                      1.058   1.047   1.042   1.041
  7                      1.078   1.082   1.085   1.084
  8                      1.154   1.124   1.108   1.083
  9                      1.096   1.103   1.115   1.128
  10                     1.109   1.128   1.103   1.084
  11                     1.123   1.081   1.082   1.088
  12                     1.101   1.098   1.107   1.098
  13                     1.132   1.124   1.100   1.096
  14                     1.153   1.119   1.112   1.109
  15                     1.206   1.201   1.173   1.163
  16                     1.220   1.181   1.188   1.186
  17                     1.334   1.228   1.193   1.156
  18                     1.271   1.249   1.249   1.245
  19                     1.295   1.293   1.283   1.288
  20                     1.317   1.305   1.269   1.238
  21                     1.224   1.175   1.176   1.181
  22                     1.200   1.189   1.170   1.177
  23                     1.272   1.223   1.228   1.224
  24                     1.252   1.250   1.252   1.255
  25                     1.129   1.130   1.133   1.135
  ranks 2-25, mean       1.164   1.146   1.138   1.132
  ranks 21-25, mean      1.215   1.194   1.192   1.195
  mean abs(bias - 1)     0.161   0.144   0.136   0.131
  the first half (2024-02-08 to 2025-06-23, 343)
  rank                      21      42      63    90 *
  1                      1.013   1.040   1.054   1.064
  2                      0.890   0.901   0.907   0.912
  3                      0.999   0.994   0.984   0.970
  4                      1.005   0.999   1.010   1.017
  5                      0.973   0.969   0.963   0.963
  6                      0.906   0.908   0.910   0.911
  7                      0.923   0.938   0.961   0.979
  8                      1.038   1.019   1.003   0.996
  9                      0.936   0.942   0.944   0.942
  10                     0.963   0.972   0.962   0.959
  11                     0.936   0.921   0.937   0.942
  12                     1.041   1.008   0.999   0.998
  13                     1.003   1.037   1.056   1.062
  14                     1.029   1.026   1.011   1.006
  15                     1.078   1.075   1.050   1.057
  16                     1.085   1.092   1.126   1.122
  17                     1.153   1.093   1.072   1.058
  18                     1.084   1.074   1.047   1.040
  19                     1.125   1.101   1.113   1.112
  20                     1.098   1.088   1.089   1.088
  21                     1.105   1.128   1.147   1.149
  22                     1.089   1.095   1.097   1.115
  23                     1.159   1.144   1.149   1.160
  24                     1.190   1.187   1.182   1.174
  25                     1.048   1.027   1.023   1.021
  ranks 2-25, mean       1.036   1.031   1.031   1.031
  ranks 21-25, mean      1.118   1.116   1.119   1.124
  mean abs(bias - 1)     0.073   0.068   0.066   0.067
  the second half (2025-06-24 to 2026-11-02, 343)
  rank                      21      42      63    90 *
  1                      0.843   0.834   0.824   0.814
  2                      1.057   1.058   1.051   1.040
  3                      1.111   1.124   1.131   1.127
  4                      1.107   1.074   1.063   1.048
  5                      1.230   1.193   1.152   1.125
  6                      1.191   1.172   1.160   1.158
  7                      1.210   1.210   1.197   1.182
  8                      1.259   1.222   1.205   1.165
  9                      1.228   1.239   1.264   1.288
  10                     1.235   1.267   1.229   1.196
  11                     1.284   1.221   1.212   1.216
  12                     1.159   1.177   1.206   1.190
  13                     1.245   1.195   1.142   1.128
  14                     1.267   1.205   1.207   1.205
  15                     1.319   1.303   1.278   1.258
  16                     1.342   1.264   1.247   1.249
  17                     1.493   1.352   1.303   1.247
  18                     1.435   1.403   1.424   1.420
  19                     1.447   1.461   1.435   1.444
  20                     1.506   1.486   1.420   1.362
  21                     1.333   1.222   1.205   1.215
  22                     1.298   1.278   1.241   1.238
  23                     1.373   1.295   1.302   1.285
  24                     1.310   1.312   1.320   1.330
  25                     1.205   1.226   1.235   1.239
  ranks 2-25, mean       1.277   1.248   1.235   1.223
  ranks 21-25, mean      1.304   1.266   1.260   1.261
  mean abs(bias - 1)     0.272   0.245   0.232   0.222

  the whole test (2024-02-08 to 2026-11-02, 686), by volatility half-life (90 the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  21         0.923   0.885   0.889   0.909   1.032  10.368   1.129  10.607   0.979   0.161   0.090   0.146   0.086   0.107
  42         0.930   0.892   0.896   0.916   1.006  10.216   1.070  10.365   0.984   0.144   0.090   0.136   0.086   0.104
  63         0.932   0.893   0.899   0.919   0.995  10.154   1.047  10.260   0.986   0.136   0.092   0.135   0.085   0.104
  90 *       0.934   0.893   0.902   0.920   0.990  10.128   1.033  10.193   0.987   0.131   0.094   0.135   0.086   0.105

  the first half (2024-02-08 to 2025-06-23, 343), by volatility half-life (90 the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  21         1.015   0.949   0.894   0.983   1.143  12.307   1.225  12.418   0.958   0.073   0.059   0.201   0.078   0.113
  42         1.039   0.968   0.908   1.004   1.124  12.091   1.187  12.190   0.968   0.068   0.061   0.169   0.078   0.103
  63         1.053   0.977   0.916   1.016   1.114  11.964   1.172  12.061   0.972   0.066   0.065   0.157   0.075   0.099
  90 *       1.063   0.983   0.921   1.024   1.109  11.881   1.163  11.953   0.978   0.067   0.068   0.150   0.076   0.098

  the second half (2025-06-24 to 2026-11-02, 343), by volatility half-life (90 the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  21         0.823   0.818   0.884   0.829   0.909   7.989   1.026   8.432   0.987   0.272   0.156   0.058   0.082   0.098
  42         0.808   0.809   0.884   0.821   0.874   7.928   0.939   8.160   0.989   0.245   0.150   0.090   0.084   0.108
  63         0.796   0.801   0.883   0.812   0.862   7.962   0.905   8.084   0.987   0.232   0.153   0.112   0.086   0.117
  90 *       0.785   0.793   0.882   0.804   0.855   8.018   0.886   8.077   0.986   0.222   0.159   0.125   0.084   0.123

  The members by capitalisation quintile, the largest first, by part: the median statistic
  the whole test
  21       1.028, 0.988, 0.965, 0.940, 0.956
  42       1.027, 0.991, 0.977, 0.945, 0.963
  63       1.028, 0.991, 0.984, 0.947, 0.963
  90 *     1.028, 0.992, 0.988, 0.949, 0.963
  the first half
  21       1.008, 0.969, 0.969, 0.930, 0.944
  42       1.018, 0.978, 0.979, 0.934, 0.952
  63       1.023, 0.982, 0.983, 0.936, 0.954
  90 *     1.028, 0.985, 0.985, 0.940, 0.954
  the second half
  21       1.027, 0.999, 0.979, 0.942, 0.951
  42       1.027, 0.995, 0.981, 0.944, 0.956
  63       1.026, 0.994, 0.986, 0.945, 0.959
  90 *     1.024, 0.990, 0.984, 0.944, 0.960

  Day to day: the median absolute change in log forecast volatility from one forecast date to the next (its 95th percentile), the cap-weighted index's and every member's (the median of each one's median)
             index     p95  members     p95
  21         1.66%   5.59%    1.64%   7.09%
  42         1.25%   4.42%    1.54%   6.64%
  63         1.09%   3.88%    1.50%   6.48%
  90 *       1.00%   3.58%    1.48%   6.39%
```

### judge_V17.txt -- item 2: every V17 test judged by the rule against D16, its own model's

```text
The reference: claude/scratch/midoffice16/bias/D16.npz's model test. The run: claude/scratch/midoffice17/bias/V17.npz, its tests the model, _v21, _v42, _v63. 686 forecast dates, 2024-02-08 to 2026-11-02, the halves 343 and 343.
The rule (COVARIANCE_RULE): the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second; the criterion: the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median)

1. The minimum-variance portfolios' realised volatility, % a year, whole / first / second, against the reference's; out if either rounds above it over the whole test
  reference  10.13 / 11.88 /  8.02   10.19 / 11.95 /  8.08
  model      10.13 / 11.88 /  8.02   10.19 / 11.95 /  8.08   in
  _v21       10.37 / 12.31 /  7.99   10.61 / 12.42 /  8.43   out
  _v42       10.22 / 12.09 /  7.93   10.36 / 12.19 /  8.16   out
  _v63       10.15 / 11.96 /  7.96   10.26 / 12.06 /  8.08   out

2. The regime criterion, whole / first / second, and its three MRADs (the broad portfolios', the optimiser's, the members' median)
  reference  criterion 0.105 / 0.098 / 0.123   broad 0.094 / 0.068 / 0.159   optimiser 0.135 / 0.150 / 0.125   members 0.086 / 0.076 / 0.084
  model      criterion 0.105 / 0.098 / 0.123   broad 0.094 / 0.068 / 0.159   optimiser 0.135 / 0.150 / 0.125   members 0.086 / 0.076 / 0.084
  _v21       criterion 0.107 / 0.113 / 0.098   broad 0.090 / 0.059 / 0.156   optimiser 0.146 / 0.201 / 0.058   members 0.086 / 0.078 / 0.082
  _v42       criterion 0.104 / 0.103 / 0.108   broad 0.090 / 0.061 / 0.150   optimiser 0.136 / 0.169 / 0.090   members 0.086 / 0.078 / 0.084
  _v63       criterion 0.104 / 0.099 / 0.117   broad 0.092 / 0.065 / 0.153   optimiser 0.135 / 0.157 / 0.112   members 0.085 / 0.075 / 0.086
  the rule: in -- model; the lowest on the first half model (0.098) against the reference's 0.098: not below it, so the reference stays; on the second half it reads 0.123 against 0.123

3. Beside it: each test's own eigenportfolios, rank 1 and ranks 21-25 (mean), the statistic whole / first / second
  reference  rank 1 0.946 / 1.064 / 0.814   ranks 21-25 1.195 / 1.124 / 1.261   mean abs(bias - 1) 0.131 / 0.067 / 0.222
  model      rank 1 0.946 / 1.064 / 0.814   ranks 21-25 1.195 / 1.124 / 1.261   mean abs(bias - 1) 0.131 / 0.067 / 0.222
  _v21       rank 1 0.931 / 1.013 / 0.843   ranks 21-25 1.215 / 1.118 / 1.304   mean abs(bias - 1) 0.161 / 0.073 / 0.272
  _v42       rank 1 0.942 / 1.040 / 0.834   ranks 21-25 1.194 / 1.116 / 1.266   mean abs(bias - 1) 0.144 / 0.068 / 0.245
  _v63       rank 1 0.945 / 1.054 / 0.824   ranks 21-25 1.192 / 1.119 / 1.260   mean abs(bias - 1) 0.136 / 0.066 / 0.232

4. The optimiser along its own covariance's eigen-directions: each group's share of the forecast variance (the mean over dates), and the statistic of its realised part, whole / first / second; the factor returns recovered from the run's model test
  minimum variance, long-only
    model    rank 1 41.6% 0.95/1.06/0.81; ranks 2-5 13.2% 1.03/0.92/1.14; ranks 6-15  8.4% 1.13/1.06/1.18; ranks 16-25 20.6% 1.17/1.13/1.22; specific 16.2% 0.88/0.89/0.87
    _v21     rank 1 39.9% 0.93/1.01/0.84; ranks 2-5 11.6% 1.07/0.92/1.21; ranks 6-15 10.8% 1.15/1.05/1.23; ranks 16-25 20.5% 1.13/1.08/1.18; specific 17.1% 0.85/0.90/0.80
    _v42     rank 1 40.7% 0.94/1.04/0.83; ranks 2-5 12.5% 1.05/0.92/1.17; ranks 6-15  9.6% 1.14/1.06/1.21; ranks 16-25 20.5% 1.13/1.09/1.18; specific 16.6% 0.86/0.89/0.82
    _v63     rank 1 41.2% 0.94/1.05/0.82; ranks 2-5 13.0% 1.04/0.92/1.16; ranks 6-15  8.9% 1.13/1.06/1.20; ranks 16-25 20.5% 1.15/1.11/1.20; specific 16.4% 0.87/0.89/0.85
  minimum variance, unconstrained
    model    rank 1 35.4% 0.95/1.06/0.81; ranks 2-5 14.1% 1.04/0.92/1.14; ranks 6-15 10.2% 1.12/1.05/1.18; ranks 16-25 23.4% 1.17/1.14/1.21; specific 16.9% 0.91/0.93/0.90
    _v21     rank 1 31.0% 0.93/1.01/0.84; ranks 2-5 12.5% 1.07/0.91/1.21; ranks 6-15 14.2% 1.18/1.05/1.29; ranks 16-25 23.0% 1.18/1.10/1.26; specific 19.3% 0.90/0.92/0.89
    _v42     rank 1 33.5% 0.94/1.04/0.83; ranks 2-5 13.6% 1.06/0.91/1.18; ranks 6-15 11.9% 1.15/1.06/1.22; ranks 16-25 23.1% 1.15/1.10/1.20; specific 17.9% 0.90/0.92/0.88
    _v63     rank 1 34.7% 0.94/1.05/0.82; ranks 2-5 14.0% 1.05/0.92/1.16; ranks 6-15 10.9% 1.13/1.06/1.20; ranks 16-25 23.2% 1.16/1.12/1.20; specific 17.3% 0.90/0.92/0.89
```

### plain_cost.txt -- ruling 4: the plain run's time

```text
Ruling 4: the plain run's time, the factor side read from the full fit behind its guard -- P17x, the factor side off, against P17, the default (the runner's log, one at a time, BLAS on one thread); midoffice16's P16, the refits, beside them
  2026-11-13 14:17:37  P17x: done in 0.8 min
  2026-11-13 14:18:29  P17: done in 0.9 min
  midoffice16: 2026-11-13 11:54:17  P16x: done in 0.8 min
  midoffice16: 2026-11-13 12:01:50  P16: done in 7.5 min

P17's report, its regime section:
  The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
    the last day's B_F^2 0.424 over 25 factors, the specific side not read
    the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.2 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.2 s)

P16's (midoffice16, the refits), the same section:
  The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
    the last day's B_F^2 0.424 over 25 factors, B_S^2 0.701 over 499 members (an effective number of 48)
    the model refitted as of each of 686 days for it, 2024-02-08 to 2026-11-02: 6.8 min

P17's report against P16's, every line (diff):
  167,168c167,168
  <   the last day's B_F^2 0.424 over 25 factors, B_S^2 0.701 over 499 members (an effective number of 48)
  <   the model refitted as of each of 686 days for it, 2024-02-08 to 2026-11-02: 6.8 min
  ---
  >   the last day's B_F^2 0.424 over 25 factors, the specific side not read
  >   the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.2 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.2 s)

P17x's against P16x's (diff; both the factor side off):
  (end)
```

### bias/runner_log.txt -- the runs, one at a time

```text
2026-11-13 13:01:40  runner started, pid 5904, for E17, V17, P17x, P17; 2.3 GB available
2026-11-13 13:01:40  E17: waiting, 2.3 GB available, 3 needed
2026-11-13 13:03:40  E17: 3.1 GB available
2026-11-13 13:03:40  E17: attempt 1: --bias-guard 0 --correlation-half-life 63 --bias-eigen-scales 1,2,3 --eigen-one-sided --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-13 13:19:40  E17: done in 16.0 min
2026-11-13 13:19:40  V17: waiting, 2.9 GB available, 3 needed
2026-11-13 13:29:40  V17: waiting, 2.7 GB available, 3 needed
2026-11-13 13:39:40  V17: waiting, 2.5 GB available, 3 needed
2026-11-13 13:49:40  V17: waiting, 1.2 GB available, 3 needed
2026-11-13 13:59:40  V17: waiting, 2.7 GB available, 3 needed
2026-11-13 14:01:40  V17: 3.1 GB available
2026-11-13 14:01:40  V17: attempt 1: --bias-guard 0 --bias-volatility-half-lives 21,42,63 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-13 14:11:46  V17: done in 10.1 min
2026-11-13 14:11:46  P17x: waiting, 2.6 GB available, 3 needed
2026-11-13 14:16:46  P17x: 3.4 GB available
2026-11-13 14:16:46  P17x: attempt 1: --regime-factor-half-life none --industries --membership
2026-11-13 14:17:37  P17x: done in 0.8 min
2026-11-13 14:17:37  P17: 4.0 GB available
2026-11-13 14:17:37  P17: attempt 1: --industries --membership
2026-11-13 14:18:29  P17: done in 0.9 min
2026-11-13 14:18:29  runner finished; every run done
```

### final_checks.txt -- the final code: the default panel rebuilt, the defaults re-captured; the suite, ruff, hygiene

```text
build _released: exit 0
20
FD17: status 0, mean R2 0.293825, 938 periods, 25 factors, 56s, the estimator 1.1s
claude/scratch/midoffice16/model/M16 against claude/scratch/midoffice17/final/FD17: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 163 and 168 lines; 5 differing line(s)
    +
    +The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    +  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
    +  the last day's B_F^2 0.424 over 25 factors, the specific side not read
    +  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.3 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.3 s)
The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.3 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.3 s)
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2375 passed, 1 skipped, 19 warnings in 90.26s (0:01:30) ===========
All checks passed!
ruff: clean over the changed Python files
scripts/run_sp500_model.py: CRLF (3050 lines), parses, ASCII
src/praxis/midoffice/model/regime.py: LF (364 lines), parses, ASCII
tests/midoffice/test_regime.py: LF (259 lines), parses, ASCII
tests/midoffice/test_bias.py: LF (476 lines), parses, ASCII
docs/FACTOR_MODEL.md: CRLF (1785 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (853 lines), -, ASCII
claude/handoffs/BRIEF_midoffice17_calm_half_two_levers.md: LF (80 lines), -, ASCII
claude/retros/RETRO_midoffice17_calm_half_two_levers.md: LF (568 lines), -, ASCII
all ok
exit 0
8
0
```

## git add (always last -- Rule 43)

This piece -- midoffice16's rulings (the plain run's regime history from the full fit), the volatility half-lives' one-pass test, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add scripts/run_sp500_model.py src/praxis/midoffice/model/regime.py tests/midoffice/test_regime.py tests/midoffice/test_bias.py docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice17_calm_half_two_levers.md claude/retros/RETRO_midoffice17_calm_half_two_levers.md
```
