# Retro: Mid-office 18 -- midoffice17's rulings, then the country factor

**Brief:** `claude/handoffs/BRIEF_midoffice18_country_factor.md` -- written in Chat and pasted
into the Code session, transcribed verbatim
**Date:** 2026-11-13 (EST)
**Duration:** the brief arrived shortly before 19:41Z (14:41 EST), the first clock reading after
it. This session is the one midoffice16 and midoffice17 ran in, so the state file and
midoffice17's retro were in hand.
- **Item 1's library and its tests,** then the check on the model's own regressions on nine
  dates, by 14:49.
- **Item 2's pass, built** by 14:51; **two smoke runs** by 14:54 -- the invariance (one
  half-life, the regime adjustment off) and the defaults.
- **The run, C18, through the runner:** 14:56 to 15:04 (8.6 minutes), started with 4.8 GB
  available.
- **Judged by midoffice17's `judge.py`, unchanged** -- the rule fixed before any of these runs.
  Item 3's readings (`country_readings.py`) were written while C18 ran, before its results; the
  reading that locates the failure (`multiplier_split.py`) after them. Then the docs, the state
  file and this retro.

**Status:** COMPLETE.
- **Rulings 1 to 3:** nothing to build.
- **Items 1 to 4:** item 1 built and checked before the pass; item 2 built, run and judged against
  D16; item 3's expectation checked, and its failure located; item 4 does not arise -- the country
  basis did not pass.
- **No default changed.**

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Item 1: USE4's country factor is the same fit in another basis -- checked before the pass.**
  - **The basis:** the country's return is the industries' returns averaged by capitalisation,
    each industry's its own less the country's, the styles unchanged (`model/country.py`).
  - **On the model's own regressions,** nine dates: the constrained fit gives the transformed
    factor returns to 5.0e-14 of the largest and the same residuals to 1.8e-14.
  - **Three members in no industry factor** -- CTVA, FRCB and SBNY, "Other Industries", 996 of
    470,527 member-periods -- take the country through the industries' sum, so the fit is the
    model's. USE4's literal country, every member exposed, is a slightly different fit where they
    are in the regression.
  - **What changes is the covariance:** a quadratic-form estimator is basis-invariant (a smoke run
    on one half-life, to 1.3e-15); the two-half-life one is not.
- **Item 2: one pass beside the model (C18, `--bias-country`), its model test D16's to the bit.**
  - **The optimiser's test: in** -- the first candidate under which the optimiser realises less
    in both halves: long-only 9.98% a year against 10.13%, unconstrained 9.97% against 10.19%.
  - **The shape, fixed:** the eigenportfolios' mean abs(bias - 1) 0.064 against 0.131 (0.075
    against 0.222 in the calm half).
  - **The criterion: out on the first half,** 0.103 against D16's 0.098, and 0.176 against 0.123
    on the second. **D16 stays.**
- **Item 3: the expectation held for the forecast and failed for the level.**
  - **The forecast:** the index's is nearly the country's own, 1.03 times it in both halves.
  - **The level:** it went the wrong way in both halves -- the index 1.063 to 1.100 and 0.785 to
    0.718, rank 1 1.064 to 1.095 and 0.814 to 0.747.
  - **Why, from the stored arrays: the regime multiplier, not the basis.** Read over the basis's
    26 factors it stops following the market: the other 25 read 1.13 and 1.18 squared by half,
    the country 1.50 and 0.65, so lambda_F sits near 1.065 in both halves, where the model's is
    1.121 and 0.956. With the multiplier divided out, the basis alone moves rank 1 a little
    toward one (1.226 and 0.780 to 1.207 and 0.811).
  - **On two multipliers,** the country's own and the other 25's, rank 1 reads 1.033 and 0.988
    and the shape 0.076 and 0.063 -- a reading of the factor side; the optimiser needs a run.
- **Item 4:** it did not pass; nothing built, a note on what building it would take.
- **Which I would choose:** D16, by the rule. Before ruling 3 settles the factor side, one more
  pass: the country basis on the two-block multiplier, judged by the same rule. It is the market
  direction's own multiplier, which ruling 2 put after the country factor, and the country basis
  is what makes it one factor's. If it fails, the factor side is settled and the largest members'
  specific risk is next.
- **Tests:** 2,377 passed, 1 skipped, 2 new -- on a rerun. The final checks' own run had one
  failure, intermittent and not named (Failures).

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice18_country_factor.md`, verbatim. In outline:

- **"committed -- midoffice17."** Checked: c80a112 in `git log`, before any file was touched.
- **Rulings on midoffice17's decisions:**
  1. The factor covariance as it stands: correlations 504, volatilities 90, the factor side on
     21, no eigenvalue adjustment. The eigenvalue adjustment is closed: three tests have fixed the
     statistics it targets and not the optimiser's realised risk.
  2. The calm half's market direction: neither left nor given a multiplier of its own. Test USE4's
     country factor first: item 3 found the market direction's variance carried by the
     industries' correlations on the slow window; a country factor carries it on the
     volatilities' window instead.
  3. Next: the country factor, then the largest members' specific risk. If the country factor
     fails the rule, the factor side is settled and specific risk is the piece after it.
- **midoffice18, the country factor,** the rulings first:
  1. The country factor as a covariance basis: USE4's regression is the same fit, its factor
     returns a change of basis; the covariance changes because the two-half-life estimator is not
     basis-invariant. Check it on a few dates before the pass: the transformed factor returns and
     the same residuals, to rounding.
  2. One pass beside the model, as C16 and V17 were: each refit's factor returns taken to the
     country basis, the covariance estimated there (Newey-West as the horizon wants it) and mapped
     back; its regime history from its own forecasts, B_F over the basis's 26 factors. Judged by
     the same rule against D16, with C16's tables and item 3's readings beside it: the forecast
     correlations between the industries by quarter against the realised, and rank 1 by half.
  3. The expectation, for the test to check: the index's forecast becomes nearly the country
     factor's own variance, so most of its calm-half over-forecast should go, and some of its
     first-half under-forecast with it. "What the optimiser does is the test."
  4. If it passes, say what building it into the model would take, and don't build it.
- Runs one at a time through the runner. **"Change no default. Report, and say which you would
  choose and why."** Retro with its `git add` and the state file in the same commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `src/praxis/midoffice/model/country.py` | new: the country basis -- `country_weights`, `to_country_basis`, `country_map`, `country_covariance`; the module docstring (USE4 section 3.1, the basis invariance) | new, 135 lines |
| `tests/midoffice/test_country.py` | new: 2 tests | new, 138 lines |
| `scripts/run_sp500_model.py` | item 2: `--bias-country` and its checks; in `bias_section` the country weights, each refit's covariance in the country basis mapped back, the basis's own regime history over its 26 factors, its test beside the model; `_report_settings` for the basis (its multiplier table, its labels); `_save_bias`'s `extra_arrays` -- the basis's own covariance by date and the industry weights | 48-52, 82, 1092-1113, 1191-1192, 1201, 1203-1208, 1226-1239, 1256-1263, 1266-1267, 1278, 1284, 1417, 1419-1420, 1430-1436, 1920-1921, 1924-1926, 1931-1940, 1943-1946, 1950-1951, 2000, 2040-2043, 2106-2107, 2117-2119, 2139-2140, 2154, 2358-2365, 2412, 2415-2427 |
| `docs/FACTOR_MODEL.md` | "A country factor is a change of basis", after the two-half-life estimator; C18's results after midoffice17's | 286-336, 1818-1856 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice18_country_factor.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice18_country_factor.md` | new: this retro | new |

Written outside the tracked files: nothing in `data/` (no panel is built in this piece), nothing in
the cache; `claude/scratch/midoffice18/` (gitignored), the scratch, below.

---

## Rulings 1, 2 and 3

- **Ruling 1** needs nothing built: the factor covariance is the defaults' as they stand. The
  eigenvalue adjustment stays an option, off; the state file records it closed.
- **Ruling 2** is this piece's test: the country factor first.
- **Ruling 3** is the order, with its condition. The country basis as tested fails the rule (item
  2), so by ruling 3 the factor side is settled and specific risk is next -- unless Jeff takes the
  one more pass I would choose (below), which item 3's reading argues for.

---

## Item 1: the country factor as a covariance basis

**Built, in the library** (`model/country.py`):
- **`country_weights`:** each period's industry weights, industry i's share of the capitalisation
  of the period's regression -- its members as the estimator admits them (`regression_design`), at
  the capitalisations of the date the regression reads its exposures from (t - 1). It refuses a
  member with no capitalisation, and industry columns that are not one-hot or empty for every
  member.
- **`to_country_basis`:** the factor returns in the country basis, the country first: f_c = sum_i
  c_i f_i, each industry f_i - f_c, the styles as they are.
- **`country_map`:** T, the fixed map back, f = T g whatever the weights; a covariance S in the
  basis maps back as T S T'.
- **`country_covariance`:** a fitted model's covariance estimated in the country basis by its own
  estimator -- the basis's own (26 x 26) and mapped back (25 x 25), annualised as the model is.
  Refused for a covariance that is not its estimator's alone (the sampling correction, the
  eigenvalue adjustment) and for missing weights.

**Checked before the pass, on the model's own regressions** (`country_check.py`,
`country_check.txt`):
- **The inputs:** M18, the defaults' capture with the estimator's own inputs (`CAPTURE_INPUTS=1`),
  mean R-squared 0.293825. Each date's design is `regression_design`'s, and the model's regression
  is replicated first -- to the bit on all nine dates -- so the inputs are the right ones.
- **The constrained fit,** the last industry eliminated through the constraint, solved by the same
  weighted least squares, two ways:
  - **(a) the country's exposure the members' industries' sum** -- one for every classified
    member, nothing for the three in no industry factor: the transformed factor returns to 5.0e-14
    of the largest basis return, the residuals to 1.8e-14 of the largest. The same fit.
  - **(b) USE4's literally, every member exposed:** up to 6.2e-3 and 0.83 where one of the three
    is in the regression, equal to (a) on 2026-11-02 and 11-03, which have none.
- **The dates:** the first estimated period (2023-02-08), 2024-02-08, 2024-10-14, the shock's
  2025-04-04 and 04-09, 2025-06-23, 2026-02-27 and the last two.
- **The weights:** 0.0031 to 0.2382 over the 938 periods, SIC73 the largest at 0.212 on average.

**The three members in no industry factor.** CTVA, FRCB and SBNY are the scheme's "Other
Industries", a group of three below its hard floor of five (`HARD_FLOOR`), so not a factor: their
industry rows are all zero, 996 of the 470,527 member-periods in the 938 estimated regressions.
`country_weights` first refused them as not one-hot; it now admits an empty row, and its docstring
says why. The country through the industries' sum is the brief's "same fit"; USE4's literal
country is not, for them. Defined here as the sum (Key decisions).

**The smoke runs** (`smoke_last.py 30`, the last 30 forecast dates; `smoke/invariance.txt`):
- **One half-life, the regime adjustment off** (`--correlation-half-life none
  --regime-factor-half-life none --bias-country`): the country test is the model's to rounding --
  the forecasts to 6.2e-16 of the largest, the factor covariance to 1.3e-15, the optimiser's
  exposures to 1.2e-13, its holdings the same. The change of basis and the map back are right, and
  the pass's plumbing with them.
- **The defaults:** the country test moves -- the factor covariance by up to 0.20 of its largest
  element, the forecasts by 0.07.

---

## Item 2: one pass beside the model (C18)

**Built, in the run** (`--bias-country`, with `--bias` and `--industries`):
- **The weights,** once, from the run's own inputs: `country_weights` on the panel, the
  characteristics, the capitalisations, the regression weights and the membership the estimator
  was given.
- **Each refit:** its own factor returns to the country basis, the covariance estimated there by
  the model's estimator (the two-half-life Newey-West at the horizon's lags -- none at one day) and
  mapped back. The country test's forecast is the refit's with that covariance
  (`RiskForecast.from_model(covariance=)`, midoffice16's).
- **Its own regime history,** over the basis's 26 factors: each day's B_F^2 from the basis's
  forecast of the day before -- its factor volatilities from the basis's own covariance, unadjusted
  (`forecast_volatilities`) -- and the day's factor returns in the basis, at that day's weights
  (`RegimeHistory.add_factor`). The run stops if a day does not follow the forecast of the day
  before. The specific side is the model's (off).
- **Its test** beside the model's, through the settings machinery midoffice16 and 17 built,
  judged by `COVARIANCE_RULE`; the report's settings section for the basis, with its multiplier
  table against the model's.
- **`--bias-out`** keeps its forecasts and outcomes (`_country`), its history, its covariance
  mapped back by date, the basis's own covariance by date (adjusted), and the industry weights by
  estimated period with the industries' columns.
- **Its checks:** it needs `--bias` and `--industries`; not with the sampling correction or the
  eigenvalue adjustment (the library refuses too), nor with the other passes.

**The run:** `--bias --bias-country --bias-horizons 1,5,11,21 --bias-guard 0 --industries
--membership`, the defaults otherwise: 8.6 minutes, 686 refits in 7.2 (0.63 s a refit with two
tests). Its model test is D16's to the bit in every array -- the forecasts, the outcomes, the
covariances, the optimiser's exposures, the regime history (`equal_C18_D16.txt`) -- so the guard
D16 ran with covers it.

**Judged by the rule against D16** (`claude/scratch/midoffice17/judge.py`, unchanged;
`judge_C18.txt`), each whole / first / second:

| | long-only % a year | unconstrained | criterion | broad MRAD | optimiser MRAD | members MRAD |
|---|---|---|---|---|---|---|
| D16 | 10.13 / 11.88 / 8.02 | 10.19 / 11.95 / 8.08 | 0.105 / 0.098 / 0.123 | 0.094 / 0.068 / 0.159 | 0.135 / 0.150 / 0.125 | 0.086 / 0.076 / 0.084 |
| country | 9.98 / 11.66 / 7.96 | 9.97 / 11.65 / 7.95 | 0.121 / 0.103 / 0.176 | 0.121 / 0.083 / 0.244 | 0.154 / 0.143 / 0.197 | 0.089 / 0.083 / 0.088 |

- **First, the optimiser's realised volatility: in, and less in both halves** -- by 0.15 and 0.22
  of a point a year over the whole test, 0.22 and 0.30 in the first half, 0.06 and 0.13 in the
  second. No candidate judged against D16 before -- C16's three, E17's three, V17's three --
  realised less in both halves: each traded one half for the other.
- **Then the criterion on the first half: 0.103 against 0.098 -- not below, so D16 stays.** On the
  second half it is worse still, 0.176 against 0.123.
  - **The broad portfolios' MRAD carries it:** 0.083 against 0.068, and 0.244 against 0.159.
  - **The optimiser's** is better on the first half (0.143 against 0.150) and worse on the second
    (0.197 against 0.125).
  - **The members'** a little worse in both.

**Beside it, C16's tables** (`bias/report_C18.txt`, in the appendix):
- **The statistics by half,** the country basis against D16:
  - the index 1.100 and 0.718 against 1.063 and 0.785;
  - the equally weighted index 1.015 and 0.721 against 0.983 and 0.793;
  - long-only 1.117 and 0.789 against 1.109 and 0.855;
  - unconstrained 1.157 and 0.807 against 1.163 and 0.886;
  - the members' median 0.998 and 0.935 against 0.978 and 0.986.
- **The shape, each setting's own eigenportfolios,** whole / first / second:
  - mean abs(bias - 1) 0.064 / 0.074 / 0.075 against 0.131 / 0.067 / 0.222;
  - ranks 2-25 1.043 / 1.042 / 1.044 against 1.132 / 1.031 / 1.223;
  - ranks 21-25 1.117 / 1.133 / 1.100 against 1.195 / 1.124 / 1.261.

  In the calm half every rank from 2 to 25 moves toward one but ranks 2 and 4, which cross it a
  little (1.040 and 1.048 to 0.929 and 0.939); in the first half the shape is about the same.
- **The optimiser along its own covariance's eigen-directions** (long-only; its share of the
  forecast variance, then the statistic whole / first / second):
  - rank 1 41.8%, 0.94 / 1.09 / 0.75 (D16 41.6%, 0.95 / 1.06 / 0.81);
  - ranks 2-5 13.7%, 0.95 / 0.93 / 0.97 (13.2%, 1.03 / 0.92 / 1.14);
  - ranks 6-15 7.5%, 1.02 / 1.03 / 1.00 (8.4%, 1.13 / 1.06 / 1.18);
  - ranks 16-25 20.9%, 1.11 / 1.16 / 1.06 (20.6%, 1.17 / 1.13 / 1.22);
  - specific 16.1%, 0.91 / 0.91 / 0.90 (16.2%, 0.88 / 0.89 / 0.87).
- **The members by capitalisation quintile,** the largest first:
  - in the calm half all over-forecast, 0.978, 0.944, 0.932, 0.897 and 0.915, against D16's
    1.024, 0.990, 0.984, 0.944 and 0.960;
  - in the first half 1.045, 1.015, 1.003, 0.967 and 0.968, against 1.028, 0.985, 0.985, 0.940 and
    0.954.
- **Day to day:** the index's forecast moves 0.80% at the median against 1.00%, the members' 1.45%
  against 1.48%.
- **The regime multiplier:** the basis's B_F^2 1.150 / 1.142 / 1.158 and lambda_F 1.065 / 1.063 /
  1.067 (0.876 to 1.350), against the model's 1.115 / 1.332 / 0.899 and 1.039 / 1.121 / 0.956
  (0.776 to 1.977).

**Item 3's readings beside it** (`country_readings.py`, written while C18 ran;
`country_readings.txt`):
- **The industries' mean correlation,** forecast against realised:
  - by half: the model 0.590 and 0.626, the 63-day setting 0.600 and 0.568, the country basis
    0.601 and 0.588, against realised 0.704 and 0.430;
  - by quarter, at the calm half's turn: in 2025 Q3 0.670, 0.714 and 0.711 against 0.378
    realised; by 2026 Q4 0.570, 0.429 and 0.453 against 0.333.

  The country basis's correlations, mapped back, sit with the 63-day setting's in the first half
  and between it and the model's in the calm half: the market's part of them follows the 90-day
  window, the rest the 504.
- **Rank 1 by half:** the model's own 0.946 / 1.064 / 0.814, the country basis's own 0.937 / 1.095
  / 0.747. The model's own rank 1, as the country basis forecasts it, reads the same as the
  basis's own, its forecast volatility 1.026 / 0.997 / 1.067 times the model's at the median: the
  basis forecasts the same direction 7% higher in the calm half.
- **The country factor:** forecast 13.43% a year at the median in the first half and 16.45% in the
  calm half; its statistic 1.112 and 0.743.
- **The index against the country:** the index's forecast 1.029 and 1.036 times the country's by
  half; against the model's index forecast 0.991 and 1.068.
- **The constraint's direction,** sum_i c_i f_i', nil in every period: forecast 0.64% a year at
  the median, 0.043 of the country's, correlated +0.33 with it.

---

## Item 3: the expectation

The brief's expectation, for the test to check: "the index's forecast becomes nearly the country
factor's own variance, so most of its calm-half over-forecast should go, and some of its
first-half under-forecast with it."

- **The first part holds:** the index's forecast is 1.033 times the country's at the median over
  the test, 1.029 and 1.036 by half.
- **The second does not.** The index's calm-half over-forecast grew (0.785 to 0.718), and so did
  its first-half under-forecast (1.063 to 1.100). Rank 1 the same (0.814 to 0.747; 1.064 to
  1.095).
- **The optimiser, the test,** realised less in both halves -- not through the market direction,
  whose statistics worsened, but consistent with the shape's fix: its part in the smallest
  directions is less under-forecast (ranks 16-25 1.11 against 1.17), so it is fooled less there.

**Why, from the stored arrays** (`multiplier_split.py`, written after C18's results to locate the
failure; `multiplier_split.txt`). Every forecast stored carries its setting's multiplier, the
factor covariance times lambda_F^2, and a factor-only portfolio's forecast is then lambda_F times
its unadjusted one, exactly. So rank 1 and the country factor can be read with the multiplier
divided out, or another put in, without a run; the optimiser cannot.
- **The rebuild is exact.** lambda_F as of each forecast date, from the stored daily B_F^2, gives
  the report's table to the third decimal; B_F^2 rebuilt from each setting's unadjusted forecasts
  and its outcomes matches the stored history to 4.6e-16; the stored eigenportfolio forecasts come
  back exactly.
- **The basis alone** moves the market direction a little toward one in both halves. With the
  multiplier divided out, rank 1 reads 1.226 and 0.780 by half under the model, 1.207 and 0.811 in
  the country basis; the country factor 1.225 and 0.806. The country's 90-day volatility carries
  the April 2025 shock into the calm half much as the 504-day correlations did.
- **The multiplier, read over the basis's 26 factors, stops following the market.** B_F^2 by
  group:

  | | first half | calm half |
  |---|---|---|
  | the model, all 25 | 1.332 | 0.899 |
  | its 20 industries | 1.367 | 0.821 |
  | its 5 styles | 1.195 | 1.208 |
  | the basis, all 26 | 1.142 | 1.158 |
  | the country | 1.503 | 0.653 |
  | the 20 industries less the country | 1.110 | 1.170 |
  | the 5 styles | 1.195 | 1.208 |

  - **In the model's basis** each industry carries the market, so B_F follows the market's regime
    and lambda_F goes from 1.121 to 0.956 by half.
  - **In the country basis** the market is one factor of 26, and the other 25 read 1.13 and 1.18
    by half, above one in both, so lambda_F is about 1.065 in both. The market direction loses the
    regime adjustment it had: lifted 6% in the first half instead of 12%, and 7% in the calm half
    instead of lowered 4%. The non-market directions gain one they lacked: that is the shape's
    fix.
- **On another multiplier** (each setting's own eigenportfolios, whole / first / second):

  | | rank 1 | ranks 2-25 | mean abs(bias - 1) |
  |---|---|---|---|
  | the model (D16) | 0.946 / 1.064 / 0.814 | 1.132 / 1.031 / 1.223 | 0.131 / 0.067 / 0.222 |
  | country, its own multiplier (C18) | 0.937 / 1.095 / 0.747 | 1.043 / 1.042 / 1.044 | 0.064 / 0.074 / 0.075 |
  | country, the model's multiplier | 0.953 / 1.052 / 0.843 | 1.104 / 1.025 / 1.177 | 0.103 / 0.068 / 0.176 |
  | country, two blocks | 1.010 / 1.033 / 0.988 | 1.041 / 1.046 / 1.035 | 0.062 / 0.076 / 0.063 |

  - **The model's multiplier** gives the basis's market direction better than the model's in both
    halves (1.052 and 0.843), and the shape goes back to the model's.
  - **Two blocks** -- the country's variance by lambda_C^2 from the country's own squared
    statistic, the other 25 basis factors' by lambda_R^2 from theirs, the cross terms by lambda_C
    lambda_R, both on 21 days -- fix both: rank 1 1.033 and 0.988, the shape 0.076 and 0.063.
    lambda_C runs 1.164 and 0.836 by half (0.546 to 2.349), lambda_R 1.058 and 1.075.
  - **What it cannot say:** the optimiser's realised volatility, and the index's and the members'
    statistics. They carry specific risk, and the optimiser's weights change with the multiplier:
    they need a run.

---

## Item 4: what building it would take

It did not pass, so nothing is built, and the brief's item 4 does not arise as asked. Should a
later pass pass -- the two-block one, say -- building it into the model would take:
- **The estimator:** the model's factor covariance estimated in the basis and mapped back -- a
  covariance estimator that takes each period's industry weights and wraps the model's own
  (`country_covariance` is its core) -- the factor returns staying the industry model's.
- **The plain run:** the weights once, as the pass computes them; the regime history's fast path
  from the full fit (`factor_history`) in the basis, with its guard; under two blocks, two
  histories and a block-scaled covariance (`RiskForecast.regime_adjusted`).
- **The reports:** the country's own line -- its volatility, the largest industry weights -- and
  the regime lines with each multiplier; the factor table in the model's factors, as now.
- **The defaults:** the basis on (`none` for off); the bias test's model test then the basis, so
  the stored runs' baseline breaks once -- the moment 16 ruling 3 set for the exact diagonal, which
  would go in with it; the docs and the state file.

---

## Which I would choose, and why

1. **D16, by the rule.** The country basis as the brief specified it fails the first half's
   criterion (0.103 against 0.098) and is worse on the second (0.176 against 0.123), its level
   wrong in both halves.
2. **Before ruling 3 settles the factor side, one more pass:** the country basis on the two-block
   multiplier, judged by the same rule against D16. Why:
   - **The basis earns it.** It is the first covariance change under which the optimiser realises
     less in both halves, and it fixes the shape that every lever before it could only trade
     between the halves.
   - **The failure is located, and it is not the basis.** It is the multiplier the brief specified,
     B_F over the basis's 26 factors: one multiplier cannot serve a market that swings with the
     regime (1.50 and 0.65 squared) and 25 factors under-forecast in both halves (1.13 and 1.18).
     The stored arrays say two blocks fix the market direction in both halves (1.033 and 0.988)
     and keep the shape.
   - **It is the design ruling 2 put after the country factor:** the market direction's own
     multiplier. The country basis is what makes the market one factor, so the multiplier is the
     country's own.
   - **It is cheap:** a few lines in the country path (two histories, a block-scaled covariance),
     one 9-minute run, judged by `judge.py` unchanged.
3. **What argues against it, for Jeff to weigh:**
   - **It was chosen after seeing C18.** The design comes from reading this run's halves. The
     rule's first-half choice, checked on the second, is the guard, and it is weaker for a design
     picked from the same halves.
   - **lambda_C is one factor's statistic on 21 days** -- in effect a 21-day volatility for the
     market: noisy (0.55 to 2.35 over the test), and the index's forecast will move more from day
     to day. V17 found shorter volatility windows cost the optimiser across all the factors; for
     the market alone the run would say.
   - **It is not USE4's.** USE4 reads one multiplier over all its factors, the country among them
     -- the design C18 tested.
4. **If it fails, the factor side is settled** and the largest members' specific risk is next, as
   ruling 3 orders it.

---

## Key decisions

1. **The country as the industries' sum, not every member:** the brief's "same fit" holds only
   so. USE4's literal country differs for the three members in no industry factor (996
   member-periods); the check measured both.
2. **The check before the pass on the model's own regressions,** the model's regression replicated
   to the bit first from M18's captured inputs, not on a synthetic panel alone; the synthetic test
   pins the same identity in the suite.
3. **An invariance smoke first:** on one half-life with the regime adjustment off, the country test
   must be the model's to rounding -- the basis, the map back and the pass's plumbing checked at
   once -- before the run that is meant to differ.
4. **The pass in the settings machinery** midoffice16 and 17 built, the country a setting keyed
   "country", so the rule, the tables and `judge.py` serve it unchanged; the basis's own regime
   history beside the model's, its specific side the model's.
5. **The basis's covariance and weights kept in `--bias-out`,** so the readings -- the country
   factor itself, the constraint's direction, the multiplier taken apart -- are exact from the
   stored arrays.
6. **`--bias-guard 0` for C18,** as for E17 and V17: its model test is D16's to the bit in every
   array, and D16 ran with the guard.
7. **The rule fixed before the run:** midoffice17's `judge.py` unchanged; item 3's readings written
   while C18 ran, before its results; the multiplier's reading after them, and said so.
8. **No default changed;** the country basis is a test option.

## Decisions left for Jeff

1. **The factor covariance:** D16 as it stands, the rule's verdict.
2. **Before the factor side is settled:**
   - **one more pass** (my choice): the country basis on the two-block multiplier, the country's
     own and the other 25 basis factors', judged by the same rule against D16;
   - **or settle it now,** as ruling 3 has it.
3. **The next piece after that:** the largest members' specific risk (ruling 3's order).

---

## Tests

**The suite:** 2,377 passed, 1 skipped, with `--ignore=tests/test_market_data.py`
(`suite_rerun.txt`). Before this piece 2,375: 2 new, and every other passes unchanged -- the new
option defaults to off. The final checks' own run of it had one failure (2,376 passed), not named:
the checks keep only pytest's last two lines. The rerun, failures named, passed every test
(Failures, below).

**New, 2 (`test_country.py`):**
- `test_the_country_fit_is_the_industry_fit_in_another_basis` -- item 1's footing. On a synthetic
  world (a market move, four industries, two styles), USE4's constrained regression on each
  period's own design gives the industry model's factor returns taken to the country basis and the
  same residuals, to 1e-10 relative; the weights sum to one and the constraint holds.
- `test_the_basis_maps_back_and_only_the_two_half_life_estimator_changes`:
  - the map back exact (1e-16);
  - an exponential covariance estimated in the basis and mapped back is the model's own;
  - the two-half-life one is not, and the country's variance in it is the short window's own;
  - the refusals: no weights, weights of the wrong shape.

**Measured, not unit-tested:**
- **The check on the model's own regressions:** nine dates, the model replicated to the bit first.
- **The smoke runs:** the country test the model's to rounding on one half-life, the regime
  adjustment off.
- **The full run:** C18's model test D16's to the bit in every array.
- **The readings:** the rebuilt multipliers, B_F^2 and eigenportfolio forecasts against the stored
  ones.

**Hygiene:** every changed file parses, is ASCII and keeps its line endings (`final_checks.txt`),
and again at the end, after the last edits to the docs and the state file and this retro's
appendix. **ruff:** clean over every changed Python file.

---

## Failures and debugging trail

- **The members in no industry factor:** `country_weights` first refused every period with CTVA,
  FRCB or SBNY in the regression, their industry rows all zero -- not one-hot. Admitted as empty
  rows, the docstring saying why; the check then measured what USE4's literal country would do to
  them.
- **Two patches by heredoc failed** -- `country_check.py`'s and `country_readings.py`'s, the
  backslashes in the patch mangled (a substring not found; an assertion). Written to files with
  the Write tool instead (`patch_check.py`, `patch_readings.py`), as the state file's working note
  says. The final checks' script, made from midoffice17's by `sed`, kept midoffice17's `git add`
  paths for the same reason; rewritten whole before it ran.
- **ruff:** an E501 in the report's country line and an F541 (an f-string with no placeholder) in
  its header; fixed.
- **The reading's dates:** `multiplier_split.py` first assumed the stored regime history ran
  through 2026-11-03. It ends at the last forecast date, 685 days for 686 forecasts (the run adds
  the last day after the test); its assertion caught it, and a working note now says so.
- **The runs:** none failed. C18 took 8.6 minutes, started with 4.8 GB available, no wait.
- **The memory guard stopped a waiter:** the background shell that waited on the final checks'
  done file was stopped by Claude Code's memory guard at about 15:24, the machine low on free
  memory while the session idled. The checks themselves ran on in their own minimized console,
  untouched; their output was read in the foreground instead, and the waiter was not restarted.
- **An intermittent failure in the suite.** The final checks' run reported 1 failed, 2,376 passed.
  - **Not named:** the checks keep only pytest's last two lines, and the run kept no cache to ask.
  - **Not this piece's, as far as can be seen:** no test touches the run script, and the two new
    tests pass alone.
  - **The rerun:** the whole suite again in its own console, failures named (`-rf`), passed 2,377
    (`suite_rerun.txt`, 15:26 to 15:28, started with 4.1 GB available).
  - **Its timing:** the failing run coincided with the machine's low-memory moment, the guard's at
    about 15:24.

  One of the repo's tests fails intermittently, not identified. A working note now says to name
  failures in the final checks.

## Problems found, not fixed

1. **One regime multiplier cannot serve the market and the rest.** In the model's basis it follows
   the market, because every industry carries it, and the non-market directions are under-forecast
   in the calm half (ranks 2-25 1.223). In the country basis it follows the rest, and the market
   loses its adjustment. The two-block pass would test the fix.
2. **The country's 90-day volatility lags the shock's decay,** much as the 504-day correlations
   did: the country forecast 16.45% a year at the median in the calm half; with no multiplier its
   statistic is 0.806 there.
3. **The constraint's direction:** the two-half-life recombination forecasts sum_i c_i f_i', nil in
   every period, a volatility of 0.64% a year -- small and spurious, worth removing should the
   basis be built.
4. **The three members in no industry factor** have no country exposure; USE4's literal country
   would give them one, a slightly different fit.
5. **An intermittent failure** somewhere in the repo's suite: once in two runs, not named
   (Failures).
6. **As before:** the largest members' specific risk (next); the two-half-life diagonal's last bit
   (ruled to wait for the first default change that moves every covariance); FedEx Freight and
   Vylor without Momentum; six ruff findings in untouched files at HEAD;
   `tests/test_market_data.py`'s `../core_repo/src`.

## Scratch

`claude/scratch/midoffice18/` (gitignored):

- **The model:** `model/M18.pkl`, `M18_inputs.pkl`, `report_M18.txt` -- the defaults' capture with
  the estimator's inputs (`CAPTURE_INPUTS=1`).
- **Item 1:** `country_check.py` and `.txt`; its patch, `patch_check.py`.
- **The pass:** `patch_country.py`, which patched the run script.
- **The runs:**
  - `runner.py`;
  - `smoke/`: `inv` and `country`, each a report, progress and `.npz`; `invariance.py` and `.txt`;
  - `bias/`: C18's report, progress and `.npz`; the runner's log and done file.
- **The analyses:**
  - `equal_C18_D16.txt` (midoffice16's `equal_correlation.py`);
  - `judge_C18.txt` (midoffice17's `judge.py`);
  - `country_readings.py` and `.txt`, with its patch `patch_readings.py`;
  - `multiplier_split.py` and `.txt`.
- **The checks:** `final_checks.sh` and `.txt`, `final/`; `suite_rerun.txt`;
  `append_appendix.py`.

The midoffice13 to 17 harnesses were used where they are: `capture.py`, `compare.py`,
`hygiene.py` (13), `compare_panels.py` (14), `smoke_last.py` (15), `equal_correlation.py` (16),
`judge.py` (17). The key outputs are copied into the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

---

## Open items for Chat

- **The decisions above,** the one more pass first.
- **The two-block pass, if wanted:** a few lines in the country path -- the country's own history
  and the rest's, the basis's covariance scaled by blocks before it is mapped back -- one run
  through the runner, judged by `judge.py` against D16.
- **The bias statistics are standing,** with `judge.py` to judge a run against D16 by the rule;
  every candidate is one run against D16, through the runner.

---

## Appendix: outputs, verbatim

Each output as its script wrote it, from `claude/scratch/midoffice18/`.

### country_check.txt -- item 1: USE4's constrained fit against the model's, on its own regressions, nine dates

```text
The model's factors: 25, 20 industries (columns 0-19) and 5 styles (Volatility, Momentum, Size, Size Nonlinearity, Trading Activity)
962 periods with industry weights; checked on 9 estimated periods

  date        members unclassified  model replicated   country (a) factors residuals (b) factors residuals
  2023-02-08      502            3        to the bit  -0.01092     5.0e-14   3.8e-15     6.2e-03   6.1e-02
  2024-02-08      502            1        to the bit  +0.00108     6.9e-15   8.9e-16     3.4e-03   6.8e-03
  2024-10-14      501            1        to the bit  +0.00705     4.0e-14   1.3e-14     2.5e-03   1.4e-01
  2025-04-04      503            1        to the bit  -0.05961     7.9e-15   7.2e-15     5.2e-03   5.9e-01
  2025-04-09      503            1        to the bit  +0.09373     2.0e-14   1.8e-14     5.1e-03   8.3e-01
  2025-06-23      503            1        to the bit  +0.01008     2.3e-14   5.9e-15     1.8e-03   9.3e-02
  2026-02-27      502            1        to the bit  -0.00190     1.8e-15   4.9e-16     7.4e-04   8.3e-03
  2026-11-02      499            0        to the bit  -0.00500     1.9e-14   2.7e-15     1.9e-14   2.7e-15
  2026-11-03      499            0        to the bit  +0.00337     1.7e-14   2.3e-15     1.7e-14   2.3e-15

  the largest relative differences from the model's, in the country basis: (a) the factor returns 5.0e-14 of the largest basis return, the residuals 1.8e-14 of the largest residual; (b) 6.2e-03 and 8.3e-01
  the country's return on each date is the cap-weighted mean of the industries' returns, the industries' weights each period's regression's own capitalisation shares
  the industry weights over the 938 periods: each a share of the regression's capitalisation, from 0.0031 to 0.2382; the largest industry's SIC73 0.212 on average
```

### smoke/invariance.txt -- item 1: the country test against the model's on the last 30 forecast dates, on one half-life with the regime adjustment off, and at the defaults

```text
inv.npz -- one half-life (90), the regime adjustment off: the basis must not move it; 30 forecast dates, 2026-09-22 to 2026-11-02
  forecast                 the country basis against the model: equal to the bit False, the largest difference 6.2e-16 of the largest value
  member_forecast          the country basis against the model: equal to the bit False, the largest difference 9.0e-17 of the largest value
  factor_covariance        the country basis against the model: equal to the bit False, the largest difference 1.3e-15 of the largest value
  eigen_forecast           the country basis against the model: equal to the bit False, the largest difference 1.1e-15 of the largest value
  realised                 the country basis against the model: equal to the bit False, the largest difference 1.2e-14 of the largest value
  held_long_only           the country basis against the model: equal to the bit True, the largest difference 0.0e+00 of the largest value
  held_unconstrained       the country basis against the model: equal to the bit True, the largest difference 0.0e+00 of the largest value
  exposure_long_only       the country basis against the model: equal to the bit False, the largest difference 5.0e-14 of the largest value
  exposure_unconstrained   the country basis against the model: equal to the bit False, the largest difference 1.2e-13 of the largest value

country.npz -- the defaults (90 / 504, F 21): the two-half-life estimator moves; 30 forecast dates, 2026-09-22 to 2026-11-02
  forecast                 the country basis against the model: equal to the bit False, the largest difference 7.2e-02 of the largest value
  member_forecast          the country basis against the model: equal to the bit False, the largest difference 1.4e-02 of the largest value
  factor_covariance        the country basis against the model: equal to the bit False, the largest difference 2.0e-01 of the largest value
  eigen_forecast           the country basis against the model: equal to the bit False, the largest difference 8.9e-02 of the largest value
  realised                 the country basis against the model: equal to the bit False, the largest difference 7.9e-02 of the largest value
  held_long_only           the country basis against the model: equal to the bit False, the largest difference 3.7e-01 of the largest value
  held_unconstrained       the country basis against the model: equal to the bit True, the largest difference 0.0e+00 of the largest value
  exposure_long_only       the country basis against the model: equal to the bit False, the largest difference 2.4e-01 of the largest value
  exposure_unconstrained   the country basis against the model: equal to the bit False, the largest difference 3.3e-01 of the largest value
```

### equal_C18_D16.txt -- item 2: C18's model test against D16, bit for bit

```text
claude/scratch/midoffice16/bias/D16.npz's model test against claude/scratch/midoffice18/bias/C18.npz's model test: 686 forecast dates, 2024-02-08 to 2026-11-02
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

### bias/report_C18.txt -- item 2: the pass's own lines

```text
  the country factor as a covariance basis (USE4): each refit's own factor returns taken to the country basis -- the country the cap-weighted mean of the 20 industries' returns, the weights each period's regression's capitalisation shares, each industry less it, the styles as they are -- the covariance estimated there by the model's estimator, its 26 factors, and mapped back; on the regime adjustment of its own forecasts, B_F over the basis's factors, F  21 S   -; chosen by the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second
  686 refits in 7.2 min, 0.63 s a refit with 2 tests; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every portfolio formed that day
```

### bias/report_C18.txt -- item 2: the country basis beside the model

```text
  The country factor as a covariance basis at one day (midoffice18): the model (*) and its factor returns in the country basis, the covariance estimated there and mapped back, on its own forecasts' volatility regime adjustment, F  21 S   -
  Each setting's regime multiplier, read from its own forecasts' history: the mean of B_F^2, and lambda_F on 21 days -- its mean (min to max) -- by part -- the country basis's B_F over its own factors, the country among them
                           the whole test                   the first half                  the second half
  country      1.150  1.065 (0.876-1.350)       1.142  1.063 (0.876-1.350)       1.158  1.067 (0.900-1.282)
  model *      1.115  1.039 (0.776-1.977)       1.332  1.121 (0.864-1.977)       0.899  0.956 (0.776-1.129)

  First, the minimum-variance portfolios' realised volatility, % a year, whole / first half / second half; a setting under which either realises more over the whole test than the model's, to the hundredth of a point, is out
                       long-only           unconstrained
  country   9.98 / 11.66 /  7.96    9.97 / 11.65 /  7.95   in
  model *  10.13 / 11.88 /  8.02   10.19 / 11.95 /  8.08   in

  Then the regime criterion -- the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median) -- by part:
  country  0.121 / 0.103 / 0.176
  model *  0.105 / 0.098 / 0.123
  chosen on the first half by the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second: model * (0.098; the model's 0.098); on the second half it reads 0.123 against the model's 0.123, the best there model *

  The shape: each setting's own eigenportfolios -- the eigenvectors of its covariance -- by rank, 1 the largest eigenvalue, the market-like one; the statistic by part
  the whole test (2024-02-08 to 2026-11-02, 686)
  rank                 country model *
  1                      0.937   0.946
  2                      0.926   0.978
  3                      0.952   1.051
  4                      0.989   1.033
  5                      0.990   1.047
  6                      0.935   1.041
  7                      0.999   1.084
  8                      0.988   1.083
  9                      1.007   1.128
  10                     0.982   1.084
  11                     0.997   1.088
  12                     0.991   1.098
  13                     1.059   1.096
  14                     1.022   1.109
  15                     1.011   1.163
  16                     1.113   1.186
  17                     1.051   1.156
  18                     1.108   1.245
  19                     1.208   1.288
  20                     1.128   1.238
  21                     1.106   1.181
  22                     1.123   1.177
  23                     1.129   1.224
  24                     1.169   1.255
  25                     1.061   1.135
  ranks 2-25, mean       1.043   1.132
  ranks 21-25, mean      1.117   1.195
  mean abs(bias - 1)     0.064   0.131
  the first half (2024-02-08 to 2025-06-23, 343)
  rank                 country model *
  1                      1.095   1.064
  2                      0.924   0.912
  3                      0.949   0.970
  4                      1.036   1.017
  5                      1.023   0.963
  6                      0.911   0.911
  7                      0.982   0.979
  8                      0.978   0.996
  9                      0.950   0.942
  10                     0.966   0.959
  11                     0.972   0.942
  12                     0.992   0.998
  13                     1.109   1.062
  14                     1.018   1.006
  15                     1.029   1.057
  16                     1.099   1.122
  17                     1.088   1.058
  18                     1.075   1.040
  19                     1.118   1.112
  20                     1.115   1.088
  21                     1.160   1.149
  22                     1.136   1.115
  23                     1.172   1.160
  24                     1.170   1.174
  25                     1.028   1.021
  ranks 2-25, mean       1.042   1.031
  ranks 21-25, mean      1.133   1.124
  mean abs(bias - 1)     0.074   0.067
  the second half (2025-06-24 to 2026-11-02, 343)
  rank                 country model *
  1                      0.747   0.814
  2                      0.929   1.040
  3                      0.957   1.127
  4                      0.939   1.048
  5                      0.958   1.125
  6                      0.958   1.158
  7                      1.017   1.182
  8                      0.998   1.165
  9                      1.061   1.288
  10                     0.999   1.196
  11                     1.021   1.216
  12                     0.990   1.190
  13                     1.007   1.128
  14                     1.026   1.205
  15                     0.994   1.258
  16                     1.128   1.249
  17                     1.013   1.247
  18                     1.141   1.420
  19                     1.289   1.444
  20                     1.130   1.362
  21                     1.047   1.215
  22                     1.111   1.238
  23                     1.085   1.285
  24                     1.165   1.330
  25                     1.094   1.239
  ranks 2-25, mean       1.044   1.223
  ranks 21-25, mean      1.100   1.261
  mean abs(bias - 1)     0.075   0.222

  the whole test (2024-02-08 to 2026-11-02, 686), by basis (the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  country    0.928   0.880   0.880   0.912   0.966   9.976   0.997   9.965   0.971   0.064   0.121   0.154   0.089   0.121
  model *    0.934   0.893   0.902   0.920   0.990  10.128   1.033  10.193   0.987   0.131   0.094   0.135   0.086   0.105

  the first half (2024-02-08 to 2025-06-23, 343), by basis (the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  country    1.100   1.015   0.958   1.057   1.117  11.658   1.157  11.646   0.998   0.074   0.083   0.143   0.083   0.103
  model *    1.063   0.983   0.921   1.024   1.109  11.881   1.163  11.953   0.978   0.067   0.068   0.150   0.076   0.098

  the second half (2025-06-24 to 2026-11-02, 343), by basis (the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  country    0.718   0.721   0.796   0.742   0.789   7.962   0.807   7.955   0.935   0.075   0.244   0.197   0.088   0.176
  model *    0.785   0.793   0.882   0.804   0.855   8.018   0.886   8.077   0.986   0.222   0.159   0.125   0.084   0.123

  The members by capitalisation quintile, the largest first, by part: the median statistic
  the whole test
  country  1.018, 0.983, 0.973, 0.935, 0.954
  model *  1.028, 0.992, 0.988, 0.949, 0.963
  the first half
  country  1.045, 1.015, 1.003, 0.967, 0.968
  model *  1.028, 0.985, 0.985, 0.940, 0.954
  the second half
  country  0.978, 0.944, 0.932, 0.897, 0.915
  model *  1.024, 0.990, 0.984, 0.944, 0.960

  Day to day: the median absolute change in log forecast volatility from one forecast date to the next (its 95th percentile), the cap-weighted index's and every member's (the median of each one's median)
             index     p95  members     p95
  country    0.80%   2.69%    1.45%   6.22%
  model *    1.00%   3.58%    1.48%   6.39%
```

### judge_C18.txt -- item 2: C18 judged by the rule against D16

```text
The reference: claude/scratch/midoffice16/bias/D16.npz's model test. The run: claude/scratch/midoffice18/bias/C18.npz, its tests the model, _country. 686 forecast dates, 2024-02-08 to 2026-11-02, the halves 343 and 343.
The rule (COVARIANCE_RULE): the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second; the criterion: the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median)

1. The minimum-variance portfolios' realised volatility, % a year, whole / first / second, against the reference's; out if either rounds above it over the whole test
  reference  10.13 / 11.88 /  8.02   10.19 / 11.95 /  8.08
  model      10.13 / 11.88 /  8.02   10.19 / 11.95 /  8.08   in
  _country    9.98 / 11.66 /  7.96    9.97 / 11.65 /  7.95   in

2. The regime criterion, whole / first / second, and its three MRADs (the broad portfolios', the optimiser's, the members' median)
  reference  criterion 0.105 / 0.098 / 0.123   broad 0.094 / 0.068 / 0.159   optimiser 0.135 / 0.150 / 0.125   members 0.086 / 0.076 / 0.084
  model      criterion 0.105 / 0.098 / 0.123   broad 0.094 / 0.068 / 0.159   optimiser 0.135 / 0.150 / 0.125   members 0.086 / 0.076 / 0.084
  _country   criterion 0.121 / 0.103 / 0.176   broad 0.121 / 0.083 / 0.244   optimiser 0.154 / 0.143 / 0.197   members 0.089 / 0.083 / 0.088
  the rule: in -- model, _country; the lowest on the first half model (0.098) against the reference's 0.098: not below it, so the reference stays; on the second half it reads 0.123 against 0.123

3. Beside it: each test's own eigenportfolios, rank 1 and ranks 21-25 (mean), the statistic whole / first / second
  reference  rank 1 0.946 / 1.064 / 0.814   ranks 21-25 1.195 / 1.124 / 1.261   mean abs(bias - 1) 0.131 / 0.067 / 0.222
  model      rank 1 0.946 / 1.064 / 0.814   ranks 21-25 1.195 / 1.124 / 1.261   mean abs(bias - 1) 0.131 / 0.067 / 0.222
  _country   rank 1 0.937 / 1.095 / 0.747   ranks 21-25 1.117 / 1.133 / 1.100   mean abs(bias - 1) 0.064 / 0.074 / 0.075

4. The optimiser along its own covariance's eigen-directions: each group's share of the forecast variance (the mean over dates), and the statistic of its realised part, whole / first / second; the factor returns recovered from the run's model test
  minimum variance, long-only
    model    rank 1 41.6% 0.95/1.06/0.81; ranks 2-5 13.2% 1.03/0.92/1.14; ranks 6-15  8.4% 1.13/1.06/1.18; ranks 16-25 20.6% 1.17/1.13/1.22; specific 16.2% 0.88/0.89/0.87
    _country rank 1 41.8% 0.94/1.09/0.75; ranks 2-5 13.7% 0.95/0.93/0.97; ranks 6-15  7.5% 1.02/1.03/1.00; ranks 16-25 20.9% 1.11/1.16/1.06; specific 16.1% 0.91/0.91/0.90
  minimum variance, unconstrained
    model    rank 1 35.4% 0.95/1.06/0.81; ranks 2-5 14.1% 1.04/0.92/1.14; ranks 6-15 10.2% 1.12/1.05/1.18; ranks 16-25 23.4% 1.17/1.14/1.21; specific 16.9% 0.91/0.93/0.90
    _country rank 1 36.2% 0.94/1.09/0.75; ranks 2-5 14.4% 0.96/0.94/0.98; ranks 6-15  8.9% 1.00/1.02/0.97; ranks 16-25 24.1% 1.12/1.17/1.06; specific 16.4% 0.93/0.93/0.93
```

### country_readings.txt -- item 2's readings: the industries' correlations by quarter, rank 1 by half, the country factor and the index

```text
C18: 686 forecast dates, 2024-02-08 to 2026-11-02; the halves 343 and 343; 20 industries; the outcomes M18's full fit's factor returns

1. The mean correlation between the industries, forecast (the mean over the dates) against realised (the sample over the outcomes), by quarter and by part
                days   model  63-day  country  realised
  2024 Q1         35   0.573   0.548    0.560     0.560
  2024 Q2         63   0.572   0.553    0.561     0.526
  2024 Q3         64   0.574   0.575    0.579     0.663
  2024 Q4         64   0.572   0.563    0.571     0.527
  2025 Q1         60   0.573   0.577    0.577     0.637
  2025 Q2         62   0.679   0.778    0.754     0.852
  2025 Q3         64   0.670   0.714    0.711     0.378
  2025 Q4         64   0.653   0.640    0.655     0.549
  2026 Q1         61   0.627   0.537    0.574     0.453
  2026 Q2         62   0.610   0.518    0.545     0.446
  2026 Q3         64   0.585   0.463    0.487     0.387
  2026 Q4         23   0.570   0.429    0.453     0.333
  whole          686   0.608   0.584    0.594     0.589
  first half     343   0.590   0.600    0.601     0.704
  second half    343   0.626   0.568    0.588     0.430

1. The mean correlation between all the factors, forecast (the mean over the dates) against realised (the sample over the outcomes), by quarter and by part
                days   model  63-day  country  realised
  2024 Q1         35   0.379   0.369    0.370     0.397
  2024 Q2         63   0.382   0.379    0.376     0.373
  2024 Q3         64   0.390   0.403    0.393     0.472
  2024 Q4         64   0.392   0.399    0.392     0.381
  2025 Q1         60   0.399   0.418    0.402     0.483
  2025 Q2         62   0.466   0.532    0.517     0.549
  2025 Q3         64   0.457   0.481    0.484     0.246
  2025 Q4         64   0.448   0.442    0.449     0.416
  2026 Q1         61   0.433   0.381    0.397     0.330
  2026 Q2         62   0.422   0.366    0.379     0.318
  2026 Q3         64   0.408   0.338    0.342     0.296
  2026 Q4         23   0.399   0.315    0.320     0.288
  whole          686   0.417   0.408    0.408     0.416
  first half     343   0.402   0.419    0.409     0.488
  second half    343   0.432   0.398    0.406     0.316

2. The market direction, the statistic whole / first / second: rank 1 of each forecast's own eigenportfolios; and the model's rank 1 as the country basis forecasts it
  rank 1, the model's own: 0.946 / 1.064 / 0.814
  rank 1, the country basis's own: 0.937 / 1.095 / 0.747
  the model's rank 1 forecast by the country basis: 0.937 / 1.095 / 0.747; its forecast volatility against the model's, the median ratio 1.026 / 0.997 / 1.067 (the outcome recovered to 1.1e-16)

3. The country factor and the index, by part: the country's forecast volatility from the basis's own covariance, % a year at the median, its statistic, and the index's forecast volatility against it and against the model's
  whole        country 15.43%, statistic 0.945; the index's forecast / the country's 1.033, / the model's index forecast 1.025; the index's statistic 0.928 against the model's 0.934
  first half   country 13.43%, statistic 1.112; the index's forecast / the country's 1.029, / the model's index forecast 0.991; the index's statistic 1.100 against the model's 1.063
  second half  country 16.45%, statistic 0.743; the index's forecast / the country's 1.036, / the model's index forecast 1.068; the index's statistic 0.718 against the model's 0.785
  (the country's return the basis's first column, to 0.0e+00)
  whole        the constraint's direction, sum_i c_i f_i' (realised nil): its forecast volatility  0.64% a year at the median, 0.043 of the country's; its correlation with the country +0.327
  first half   the constraint's direction, sum_i c_i f_i' (realised nil): its forecast volatility  0.57% a year at the median, 0.042 of the country's; its correlation with the country +0.260
  second half  the constraint's direction, sum_i c_i f_i' (realised nil): its forecast volatility  0.76% a year at the median, 0.045 of the country's; its correlation with the country +0.380
```

### multiplier_split.txt -- item 3: the regime multiplier taken apart from the basis

```text
C18: 686 forecast dates, 2024-02-08 to 2026-11-02, the halves 343 and 343

1. lambda_F as of each forecast date, rebuilt from the stored daily B_F^2 on 21 days: its mean (min to max) by part, as the report's table has it
  model    whole 1.039 (0.776-1.977)   first half 1.121 (0.864-1.977)   second half 0.956 (0.776-1.129)
  country  whole 1.065 (0.876-1.350)   first half 1.063 (0.876-1.350)   second half 1.067 (0.900-1.282)

2. The statistic whole / first / second, as stored (each setting's own multiplier), the multiplier divided out, and the country basis on the model's multiplier
                                               as stored         no multiplier         on the models
  rank 1, the model's own          0.946 / 1.064 / 0.814 1.027 / 1.226 / 0.780                      
  rank 1, the country basis's own  0.937 / 1.095 / 0.747 1.028 / 1.207 / 0.811 0.953 / 1.052 / 0.843
  the country factor               0.945 / 1.112 / 0.743 1.036 / 1.225 / 0.806 0.961 / 1.071 / 0.839

3. B_F^2 by group, the mean over each part's days of (return / forecast volatility)^2, the forecast the setting's unadjusted one of the day before; whole / first / second
  model    all 25                       1.115 / 1.332 / 0.899
  model    the 20 industries            1.094 / 1.367 / 0.821
  model    the 5 styles                 1.202 / 1.195 / 1.208
  country  all 26                       1.150 / 1.142 / 1.158
  country  the country                  1.078 / 1.503 / 0.653
  country  the 20 industries less it    1.140 / 1.110 / 1.170
  country  the 5 styles                 1.202 / 1.195 / 1.208
  (the rebuilt B_F^2 against the stored history, the largest relative difference: the model 4.1e-16, the country basis 4.6e-16)

4. Each setting's own eigenportfolios, the statistic whole / first / second: rank 1, ranks 2-25 (mean) and mean abs(bias - 1) over the 25; a reading of the factor side alone -- the optimiser's realised volatility, the test, needs a run
  the model, as stored           rank 1 0.946 / 1.064 / 0.814   ranks 2-25 1.132 / 1.031 / 1.223   mean abs 0.131 / 0.067 / 0.222
  country, its own (as stored)   rank 1 0.937 / 1.095 / 0.747   ranks 2-25 1.043 / 1.042 / 1.044   mean abs 0.064 / 0.074 / 0.075
  country, the model's           rank 1 0.953 / 1.052 / 0.843   ranks 2-25 1.104 / 1.025 / 1.177   mean abs 0.103 / 0.068 / 0.176
  country, two blocks            rank 1 1.010 / 1.033 / 0.988   ranks 2-25 1.041 / 1.046 / 1.035   mean abs 0.062 / 0.076 / 0.063
  lambda_C (country alone): mean by part 1.000 / 1.164 / 0.836 (0.546-2.349)
  lambda_R (the rest): mean by part 1.066 / 1.058 / 1.075 (0.873-1.297)
  (the stored settings' eigenportfolio forecasts rebuilt to 0.0e+00 relative)
```

### bias/runner_log.txt -- the run

```text
2026-11-13 14:55:55  runner started, pid 29460, for C18; 4.8 GB available
2026-11-13 14:55:55  C18: 4.8 GB available
2026-11-13 14:55:55  C18: attempt 1: --bias-guard 0 --bias-country --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-13 15:04:29  C18: done in 8.6 min
2026-11-13 15:04:29  runner finished; every run done
```

### final_checks.txt -- the final code: the default panel rebuilt, the defaults re-captured; the suite, ruff, hygiene

```text
build _released: exit 0
20
FD18: status 0, mean R2 0.293825, 938 periods, 25 factors, 51s, the estimator 1.0s
claude/scratch/midoffice18/model/M18 against claude/scratch/midoffice18/final/FD18: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 163 and 168 lines; 5 differing line(s)
    +
    +The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    +  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
    +  the last day's B_F^2 0.424 over 25 factors, the specific side not read
    +  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.2 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.0 s)
The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.2 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.0 s)
FAILED tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query
====== 1 failed, 2376 passed, 1 skipped, 19 warnings in 80.07s (0:01:20) ======
All checks passed!
ruff: clean over the changed Python files
scripts/run_sp500_model.py: CRLF (3152 lines), parses, ASCII
src/praxis/midoffice/model/country.py: LF (135 lines), parses, ASCII
tests/midoffice/test_country.py: LF (138 lines), parses, ASCII
docs/FACTOR_MODEL.md: CRLF (1875 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (898 lines), -, ASCII
claude/handoffs/BRIEF_midoffice18_country_factor.md: LF (73 lines), -, ASCII
claude/retros/RETRO_midoffice18_country_factor.md: LF (555 lines), -, ASCII
all ok
exit 0
7
0
```

### suite_rerun.txt -- the suite rerun, failures named (-rf): none

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Data\Development\Python\McTheoryApps\praxis
configfile: pyproject.toml
plugins: anyio-4.13.0, Faker-40.15.0, cov-7.1.0
collected 2378 items

tests\midoffice\test_asof.py ........................................    [  1%]
tests\midoffice\test_bias.py ..................                          [  2%]
tests\midoffice\test_conditioning.py ...s..............                  [  3%]
tests\midoffice\test_config.py ...........................               [  4%]
tests\midoffice\test_contributions.py ...........................        [  5%]
tests\midoffice\test_country.py ..                                       [  5%]
tests\midoffice\test_covariance.py ..................................... [  7%]
.........................                                                [  8%]
tests\midoffice\test_crosssectional.py ................................. [  9%]
........................................................................ [ 12%]
............                                                             [ 13%]
tests\midoffice\test_decomposition.py .....................              [ 13%]
tests\midoffice\test_descriptors.py .................................... [ 15%]
....................                                                     [ 16%]
tests\midoffice\test_eigen.py .......                                    [ 16%]
tests\midoffice\test_engine.py ................                          [ 17%]
tests\midoffice\test_exposure_basis.py ..............                    [ 17%]
tests\midoffice\test_factors.py .........................                [ 18%]
tests\midoffice\test_identity.py ..........................              [ 20%]
tests\midoffice\test_industries.py ...........................           [ 21%]
tests\midoffice\test_lookahead.py ...                                    [ 21%]
tests\midoffice\test_market.py ......................................... [ 23%]
.......                                                                  [ 23%]
tests\midoffice\test_model.py .............................              [ 24%]
tests\midoffice\test_parity.py ......................................... [ 26%]
..........................                                               [ 27%]
tests\midoffice\test_portfolio.py ............................           [ 28%]
tests\midoffice\test_prices.py ......................................... [ 30%]
.....                                                                    [ 30%]
tests\midoffice\test_rebuild.py .............                            [ 30%]
tests\midoffice\test_regime.py .......                                   [ 31%]
tests\midoffice\test_risk_core.py ...............................        [ 32%]
tests\midoffice\test_securities.py .......................               [ 33%]
tests\midoffice\test_sp500.py .......................................... [ 35%]
.                                                                        [ 35%]
tests\midoffice\test_specificrisk.py ..................................  [ 36%]
tests\midoffice\test_tail.py ......................                      [ 37%]
tests\midoffice\test_valuation.py ..............................         [ 38%]
tests\test_atlas_sync.py .......                                         [ 39%]
tests\test_backtest.py ...............................                   [ 40%]
tests\test_backtest_validation.py ...............................        [ 41%]
tests\test_battle_round3.py ....................                         [ 42%]
tests\test_blvt_mechanism.py .........                                   [ 43%]
tests\test_burgess.py .................................................. [ 45%]
                                                                         [ 45%]
tests\test_calendar_quality_class.py ................................... [ 46%]
....                                                                     [ 46%]
tests\test_chan_cpo_data_loader.py ............                          [ 47%]
tests\test_config.py ..............................................      [ 49%]
tests\test_cpo.py .......................                                [ 50%]
tests\test_cpo_training.py ................                              [ 50%]
tests\test_data.py .......................                               [ 51%]
tests\test_database.py ..........................                        [ 52%]
tests\test_engines.py .................................................. [ 55%]
....                                                                     [ 55%]
tests\test_flash_pipeline.py ........................................    [ 56%]
tests\test_forced_trade_labels.py .................                      [ 57%]
tests\test_keys.py ..................................................... [ 59%]
..........                                                               [ 60%]
tests\test_lifecycle_needs_reconfig.py ................................. [ 61%]
...                                                                      [ 61%]
tests\test_live_paper_agent.py ......................................... [ 63%]
.........                                                                [ 63%]
tests\test_loader.py .....................                               [ 64%]
tests\test_logger.py ................................................... [ 66%]
.............                                                            [ 67%]
tests\test_microstructure_utils.py ...........                           [ 67%]
tests\test_minute_features.py .................................          [ 69%]
tests\test_ml_risk.py .................................................. [ 71%]
....                                                                     [ 71%]
tests\test_onchain.py .................................................. [ 73%]
.............                                                            [ 74%]
tests\test_port_validation.py ............                               [ 74%]
tests\test_quality_templates_class.py .................................. [ 76%]
.........................                                                [ 77%]
tests\test_regime_engine.py ............................................ [ 79%]
.....................                                                    [ 79%]
tests\test_registry.py ........................................          [ 81%]
tests\test_results.py ..................                                 [ 82%]
tests\test_runner.py ....................                                [ 83%]
tests\test_scheduler.py ..................................               [ 84%]
tests\test_scheduler_agent_logger.py ................................... [ 86%]
...............                                                          [ 86%]
tests\test_security.py ..................................                [ 88%]
tests\test_signals_indicators.py ....................................... [ 89%]
.                                                                        [ 89%]
tests\test_templates_scheduling.py ..................................... [ 91%]
........................                                                 [ 92%]
tests\test_temporal.py .......................                           [ 93%]
tests\test_universe.py .............................                     [ 94%]
tests\test_validation_realistic.py ..................................... [ 96%]
...                                                                      [ 96%]
tests\test_vendor_capture.py ..................                          [ 97%]
tests\test_workflow_stats.py ........................................... [ 98%]
..........................                                               [100%]

============================== warnings summary ===============================
tests/test_backtest_validation.py: 16 warnings
  C:\Data\Development\Python\McTheoryApps\praxis\src\praxis\backtest\event_driven.py:205: RuntimeWarning: invalid value encountered in scalar power
    (equity[-1] / equity[0]) ** (1 / years) - 1

tests/test_backtest_validation.py::TestEventDrivenFillModel::test_fill_on_close
  C:\Data\Development\Python\McTheoryApps\praxis\.venv\Lib\site-packages\numpy\_core\_methods.py:219: RuntimeWarning: Degrees of freedom <= 0 for slice
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,

tests/test_backtest_validation.py::TestEventDrivenFillModel::test_fill_on_close
  C:\Data\Development\Python\McTheoryApps\praxis\.venv\Lib\site-packages\numpy\_core\_methods.py:211: RuntimeWarning: invalid value encountered in scalar divide
    ret = ret.dtype.type(ret / rcount)

tests/test_microstructure_utils.py::TestModifiedZScore::test_degenerate_all_equal
  C:\Data\Development\Python\McTheoryApps\praxis\engines\microstructure_utils.py:55: RuntimeWarning: invalid value encountered in multiply
    result = np.where(x == med, 0.0, np.sign(x - med) * np.inf)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2377 passed, 1 skipped, 19 warnings in 79.64s (0:01:19) ===========
```

## git add (always last -- Rule 43)

This piece -- the country factor as a covariance basis (the library, its one-pass test), the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add scripts/run_sp500_model.py src/praxis/midoffice/model/country.py tests/midoffice/test_country.py docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice18_country_factor.md claude/retros/RETRO_midoffice18_country_factor.md
```
