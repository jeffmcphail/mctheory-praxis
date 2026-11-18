# Retro: Mid-office 19 -- midoffice18's rulings, then every member on the market and the market's own multiplier

**Brief:** `claude/handoffs/BRIEF_midoffice19_every_member_on_the_market.md` -- written in Chat and
pasted into the Code session, transcribed verbatim
**Date:** 2026-11-13 (EST)
**Duration:** the brief arrived shortly before 21:32Z (16:32 EST), the first clock reading after
it. This session is the one midoffice16 to 18 ran in, its context summarised once before this
piece; the state file, midoffice18's retro and its `final_checks.txt` were read again first.
- **Item 1's reading before the fix** (D16's holdings rebuilt from its stored arrays) by 16:47.
- **Item 3:** the fifty runs alone 16:49 to 16:53, the in-process probe and the fix by 16:55, the
  fifty runs after it by 16:58.
- **Ruling 3, built:** the rulings file by 16:53; the new defaults' model (M19) and the old
  scheme's check against M18 by 16:58.
- **Item 2's setting and the holdings** 17:01 to 17:05, the tests, three smoke runs 17:05 to
  17:13.
- **The runs, through the runner:** D19 17:13 to 17:25 (11.7 minutes, started with 5.1 GB
  available), C19 17:25 to 17:35 (9.7 minutes, 4.6 GB). C19's judge was written at 17:14 and the
  D19 comparison at 17:15, before either run's results.
- Then the docs, the state file, the final checks (from 17:37) and this retro.

**Status:** COMPLETE.
- **Rulings 1 to 5:** ruling 3 built, the default; ruling 4 recorded, and my midoffice18 error
  with it; ruling 2's bar in the run and the judge.
- **Items 1 to 3:** item 1 read before the fix (D16) and after it (D19); item 2 run and judged
  (C19); item 3's cause found -- an unseeded draw -- and fixed, as the brief allows.
- **One default changed, ruling 3's.**

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 3, every member on the market, is the default.** A rulings file
  (`config/sp500_industry_overrides.json`), read by the run (`--industry-overrides`), each entry
  with its evidence:
  - **FRCB and SBNY get SIC 6022,** state commercial banks, and join SIC60. Polygon's records
    carry no code for either; the California DFPI and New York's DFS closed them in 2023 (the
    FDIC's press releases cited).
  - **Agriculture, CTVA alone below the floor of five, joins SIC28,** its nearest industry by
    business: crop protection chemicals are SIC 2879, beside CF, MOS and FMC. The model's returns
    put the Manufacturing division a little nearer (27.9% against 28.2% residual volatility
    before the test) -- Jeff's to rule.
  - **VYLR gets 0100,** Corteva's seeds; in no regression yet, it changes no number.
  - **The standing check:** no member-day goes without the market (996 before). A group below the
    floor with no entry stops the run. The old scheme, `--industry-overrides none`, is midoffice18's
    model to the bit.
- **Item 1: CTVA was a tenth of the optimiser and a fifth of its risk; now it is none of either.**
  - **Before (D16, its holdings rebuilt from the stored arrays, exactly):** CTVA averaged 10.9% of
    the long-only minimum-variance portfolio (11.2% and 10.6% by half), its largest holding on 520
    of 686 dates, up to 32.8%, and carried 20.8% of its realised variance (21.0% and 20.2%). The
    unconstrained: 10.7% and 19.7%.
  - **After (D19):** 0.00% on average, held on 38 dates, at most 0.29%; 0.02% of the variance.
  - **The optimiser realises less:** long-only 9.62% a year against 10.13%, unconstrained 9.83%
    against 10.19%. Its statistics fall below one: 0.900 and 0.951, the calm half 0.801 and 0.846.
  - **Little else moves:** the index 0.934 to 0.931, four members by 0.01 or more, no factor
    volatility by 1%.
- **Item 2: the market's own multiplier is out by the rule, so the factor side is settled.**
  - **The two blocks fail the optimiser's test:** the unconstrained portfolio realises 9.88% a year
    against D19's 9.83% (11.74% against 11.25% in the first half), though the long-only realises
    less in both halves.
  - **They fix the level:** criterion 0.059 and 0.067 against 0.068 and 0.138, below on both halves
    as ruling 2 asked; rank 1 1.032 and 0.988; the index 1.036 and 0.938. Their price: the index's
    forecast moves 1.62% a day at the median against 1.00%.
  - **C18's single multiplier** is out again on the first half, 0.078 against 0.068. D19 stays.
- **Item 3: the intermittent test depended on an unseeded draw.** The surface seeded only its
  first batch of random walks: alone fifty times, 2 failed; on one seed, twenty repetitions gave
  twenty sets of values. Fixed, every batch seeded: fifty of fifty pass.
- **Ruling 4: I was wrong in midoffice18.** The failure was named in its `final_checks.txt`; the
  retro and the state file said it was not. Corrected.
- **Which I would choose:** D19, by the rule; the factor side settled; the largest members'
  specific risk next.
- **Tests:** 2,384 passed, 1 skipped, 7 new; the final checks' run, failures named (`-rf`):
  none.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice19_every_member_on_the_market.md`, verbatim. In
outline:

- **"committed -- midoffice18."** Checked: caca97e in `git log`, before any file was touched.
- **Rulings on midoffice18's decisions:**
  1. The factor covariance: D16's, by the rule, for now.
  2. The two-block pass: yes, after ruling 3's fix, with a higher bar -- the design came from
     reading C18's halves, so it must beat the reference's criterion on both halves, after the
     optimiser's test. Its half-life the regime adjustment's 21, untuned.
  3. No member goes without the market. CTVA, FRCB and SBNY each the industry its business makes
     it: by its SIC code where the data carry one, by an override with its evidence where they do
     not; a group below the floor joins its nearest industry instead of going without one. The
     default, the old scheme or panel kept and rebuildable.
  4. The intermittent failure is named in `final_checks.txt`:
     `tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query`.
     Record it in the state file.
  5. Next: the largest members' specific risk -- or, if the two-block pass passes, its build first
     (midoffice18's item 4, the constraint's spurious direction removed, the exact diagonal in
     it), then specific risk.
- **midoffice19, every member on the market, then the market's own multiplier,** the rulings
  first:
  1. Before the fix, from D16's stored arrays: CTVA's weight in each minimum-variance portfolio by
     half, and its share of their realised variance. After it, the new defaults' bias run (D19)
     against D16: the same two readings, the minimum-variance portfolios' realised volatility and
     statistics by half, and wherever else anything moves.
  2. The country basis in one pass beside D19's model, on two multipliers as two settings --
     C18's, and the two blocks -- judged by the rule against D19, with ruling 2's bar for the two
     blocks; beside them rank 1, the country factor and the index by half, lambda_C's path by half,
     the index forecast's day-to-day moves, the optimiser by eigen-direction, the members by
     quintile.
  3. The failing test: run it alone fifty times and say what it depends on; fix it only if the
     cause is a missing seed, and say so.
- Runs one at a time through the runner. **"Change no default beyond ruling 3. Report, and say
  which you would choose and why."** Retro with its `git add` and the state file in the same
  commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `config/sp500_industry_overrides.json` | new: the industries' rulings -- codes for FRCB, SBNY and VYLR, Agriculture's join, each with its evidence | new, 33 lines |
| `src/praxis/midoffice/refdata/industries.py` | ruling 3: `build_scheme(joins=)`, a group below the floor joining its nearest industry; `joined`, `joined_members`, their report lines; `IndustryOverrides`, `read_industry_overrides`, `apply_code_overrides`, `members_without_industry`; the module docstring | 43, 46, 100-109, 161-164, 211-215, 251-258, 310, 312-313, 321-327, 384, 387-422, 450-451, 490-583 |
| `scripts/run_sp500_model.py` | ruling 3: `INDUSTRY_OVERRIDES`, `--industry-overrides`, the codes and joins applied, the report's lines and the standing check with its stop; item 2: the two-block setting in the country pass (`blocks`, its two histories, the scaled basis covariance, `BLOCKS_BAR`, the report's multiplier rows, lambda_C's path, ruling 2's verdict line, the country factor by part); item 1: the holdings in `--bias-out` | 49, 51, 89, 93, 95, 157-165, 692, 697-701, 796-802, 1127, 1134-1141, 1221, 1237-1243, 1273-1276, 1278-1286, 1303-1307, 1319-1320, 1470-1495, 1498-1499, 1511-1512, 1962-1968, 2000, 2012, 2023-2026, 2044-2051, 2055, 2063-2083, 2137-2145, 2186-2203, 2238-2239, 2242-2243, 2406-2413, 2520-2525, 2708-2718, 2721-2729, 2833-2836, 2838-2853, 2871-2886 |
| `src/praxis/midoffice/model/country.py` | `block_scaled`; the module docstring | 23-29, 120-135 |
| `src/praxis/midoffice/model/bias.py` | the minimum-variance holdings kept (`BiasRun.min_variance_weights`) | 440-442, 592, 628, 696-697 |
| `src/praxis/stats/surface.py` | item 3: every batch seeded from the requirement's seed in `_compute_single_point` | 696, 699-709 |
| `tests/midoffice/test_industries.py` | 5 new tests | 12, 23, 29, 31, 405-536 |
| `tests/midoffice/test_country.py` | 1 new test | 16, 140-162 |
| `tests/midoffice/test_bias.py` | 1 new test | 477-498 |
| `docs/FACTOR_MODEL.md` | the country section on two multipliers; "Every member on the market" after the thin industries; D19's and C19's results after C18's | 306-310, 312-313, 331-341, 814-845, 1898-1957 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43); midoffice18's wrong working note corrected (ruling 4) | throughout |
| `claude/handoffs/BRIEF_midoffice19_every_member_on_the_market.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice19_every_member_on_the_market.md` | new: this retro | new |

Written outside the tracked files: nothing in `data/` (no panel is built: the rulings are read by
the run), nothing in the cache; `claude/scratch/midoffice19/` (gitignored), the scratch, below.

---

## Rulings 1 to 5

- **Ruling 1** needs nothing built: the defaults' factor covariance stands -- D16's, now D19's on
  the new scheme.
- **Ruling 2** is item 2's bar: `BLOCKS_BAR` in the run, beside `COVARIANCE_RULE`, and in the judge
  written before C19 (`claude/scratch/midoffice19/judge.py`, midoffice17's with the bar added).
  The blocks' half-life is the factor side's own, `--regime-factor-half-life`, 21.
- **Ruling 3,** built first (below).
- **Ruling 4: I was wrong in midoffice18.** The failure was named in `final_checks.txt`: pytest's
  short summary names a failure by default (its `-r` is `fE`), and the last two lines the checks
  kept were the `FAILED` line and the counts. The midoffice18 retro, the state file's working note
  and its open item all said it was not named. Recorded now: the working note rewritten, the open
  item replaced by item 3's finding; this piece's checks run the suite with `-rf` and keep its last
  four lines.
- **Ruling 5** is the order after this piece: in "What is next" in the state file.

---

## Ruling 3: every member on the market

**The rulings file,** `config/sp500_industry_overrides.json`, in the repo beside the count and
spin-off rulings, read by the run, not the build: the SIC codes are the scheme's input, so no
panel is rebuilt and nothing in `data/` changes.
- **Codes,** each a key, label, four-digit code and evidence, applied only where the panel's own
  records carry none -- the brief's "by its SIC code where the data carry one" -- and refused
  where the key is not the panel's for the label:
  - **FRCB, First Republic Bank, 6022.** Polygon's records carry no code, under FRC (XNYS, CIK
    0001132979) and FRCB (OTC Link from 2023-05-03), and describe "private banking and wealth
    management services"; the California DFPI closed it on 2023-05-01, the FDIC receiver (FDIC
    press release of that day) -- chartered by California. SVB Financial Group, closed by the
    same regulator, carries 6022 in Polygon's records.
  - **SBNY, Signature Bank, 6022.** No code; Polygon: "a New York-based commercial bank ...
    Commercial banking and Specialty finance"; New York's DFS closed it on 2023-03-12 (FDIC press
    release, Signature Bridge Bank, N.A.).
  - **VYLR, Vylor Inc., 0100.** No code; Polygon (from 2026-10-05): "a seed and crop genetics
    pure-play company ... spun off from Corteva" -- the business Corteva's own 0100 was given for.
    Not named in the ruling: VYLR is in no regression and has no forecast here (no momentum
    history), so its code changes no number; it keeps a member from going without the market
    once it has one. Mine, for Jeff to keep or drop.
  - **The FDIC's releases** were found by web search in this session, the closing regulators read
    from them: https://fdic.gov/index.php/news/press-releases/2023/pr23018.html (Signature) and
    https://www.fdic.gov/news/press-releases/2023/pr23034.html (First Republic).
- **Joins,** each a group, the industry it joins and the evidence: **Agriculture into SIC28.**
  - **By business.** Polygon's CTVA records: "Seeds generate a little over half of total company
    profits and crop protection chemicals make up the remainder" (2022), "the majority" from seeds
    (2024), "a crop protection pure play" once Vylor was spun off (2026-10). Crop protection
    chemicals are SIC 2879, in major group 28, where the panel's other agricultural-chemical
    makers are (CF, MOS 2870; FMC 2800).
  - **The model's returns, a check** (`nearest_probe.py`, on midoffice18's fit; CTVA's residual
    with the styles' part taken out, put in each industry):

    | | none | Manufacturing | SIC28 | SIC20 | best by returns |
    |---|---|---|---|---|---|
    | before the test (252 periods) | 30.6% | 27.9% | 28.2% | 28.9% | Manufacturing, then SIC28 |
    | the test (686) | 28.9% | 24.4% | 26.9% | 27.5% | Manufacturing, SIC35, Services |
    | the whole (938) | 29.3% | 25.4% | 27.2% | 27.9% | Manufacturing, SIC35, Retail |

    Every industry gives CTVA the market, the large part (30.6% to 27.9-29.0% before the test);
    which industry is the smaller part. SIC28's factor return is two-thirds pharmaceuticals by
    regression weight on the last date (SIC 283 0.67, agricultural chemicals 0.02); the Manufacturing division is a
    third refiners. By returns, the division is nearer; by business, SIC28. I built the business
    reading -- it is the ruling's word, the classification's own basis, and known at the time --
    and leave the choice to Jeff (Decisions).
- **FRCB and SBNY by returns** agree with their code: SIC60 is FRCB's nearest (237.8% against 255.2%
  in none over its 56 periods); SBNY's 22 periods put SIC62 first, SIC60 second.

**The scheme** (`build_scheme(joins=)`): with `joins`, a group still below the hard floor after its
division merge joins its entry's industry; nothing is pooled. Refused: a group below the floor
with no entry, a join to anything not a factor at or above the floor, and Unclassified below it --
a member with no SIC code needs a code, not a neighbour. Without `joins`, the pooling as before.
The scheme records `joined` and `joined_members` and reports them.

**The run:** `--industry-overrides` (the file by default; `none` for midoffice18's scheme). The
report names every code and join used and any entry not applied, and a standing line counts the
member-days from the first estimated period with every style exposure and no industry factor --
a member forecast with no market. Under the rulings it reads "none" and the run stops otherwise;
under `none` it reads 996: CTVA 917, FRCB 57, SBNY 22, midoffice18's 996 member-periods.

**The new scheme,** M19 (`report_M19.txt`): 20 industries as before; SIC28 41 (40), SIC60 22 (20);
Unclassified dropped, its two left (DISCA, DISCK) in no estimated period. **The old scheme
rebuilds exactly:** `--industry-overrides none` gives M18's model in all 21 fields, to the bit
(`old_scheme_vs_M18.txt`, and again at the end, `final_checks.txt`).

**The model, M19 against M18** (`model_M19_vs_M18.txt`): mean R-squared 0.293825 to 0.294466;
specific risk p10 / median / p90 16.56 / 28.99 / 48.89% to 16.67 / 28.97 / 48.88%; SIC60's mean
factor return 21.78% a year to 20.45%, its volatility 23.21% to 23.24%; every other factor
volatility within 0.07 of a point; lambda_F as of 2026-11-03 0.9640 to 0.9629; the size pair and
the regression counts unchanged.

---

## Item 1: CTVA in the optimiser, before and after

**Before: from D16's stored arrays** (`ctva_holdings.py`, `ctva_D16.txt`). D16 keeps the
optimiser's factor exposures, not its holdings, so they are rebuilt exactly as the test made them:
the stored factor covariance by date, each eligible member's specific variance from its stored
forecast less x'Fx, the forecast exposures the full run's on the date (M18's captured inputs, the
guard's equality), and the library's own `minimum_variance`. Against everything D16 stored of the
optimiser: its factor exposures to 9.8e-13 of the largest, its forecast to 1.3e-15, its realised
return to 1.3e-14, every held count, and its realised volatility to the hundredth (10.13 / 11.88
/ 8.02%). From midoffice19 the run stores the holdings (`min_variance_weights_*`); on D19 the
rebuild meets them to 4.1e-13.

CTVA's share of a portfolio's realised variance is cov(c, r_p) / var(r_p), c its piece w r of
the portfolio's return r_p: the holdings' shares sum to one, the variance the tables' realised
volatility squares.

| | long-only: weight mean (max) | largest holding on | share of realised variance | unconstrained: weight mean | share |
|---|---|---|---|---|---|
| D16, whole | 10.89% (32.83%, 2025-04-16) | 520 of 686 dates | 20.76% | 10.71% | 19.68% |
| D16, first half | 11.19% | 224 of 343 | 21.02% | 11.23% | 20.33% |
| D16, second half | 10.60% | 296 of 343 | 20.19% | 10.19% | 18.24% |
| D19, whole | 0.00% (0.29%, 2025-03-10) | 0 | 0.02% | -0.02% | 0.07% |
| D19, first half | 0.01% | 0 | 0.03% | -0.01% | 0.13% |
| D19, second half | 0.00% | 0 | -0.00% | -0.03% | -0.04% |

In D16 CTVA was held on every date it was eligible (666; it left the index on 2026-10-06), at
about 1.3 times the next-largest holding on average (8.26%). In D19 the long-only portfolio holds
it on 38 dates.

**After: the new defaults' bias run, D19, against D16** (`compare_d19.py`, written before D19's
results; `compare_D19_D16.txt`, `report_D19_vs_D16.txt`). The same members on the same days.

| whole / first / second | D16 | D19 |
|---|---|---|
| long-only realised, % a year | 10.13 / 11.88 / 8.02 | 9.62 / 11.05 / 7.97 |
| long-only forecast, median | 9.05 / 8.76 / 9.42 | 9.55 / 9.23 / 10.02 |
| long-only statistic | 0.990 / 1.109 / 0.855 | 0.900 / 0.990 / 0.801 |
| long-only names held, median | 86 / 82 / 93 | 93 / 89 / 96 |
| unconstrained realised | 10.19 / 11.95 / 8.08 | 9.83 / 11.25 / 8.18 |
| unconstrained forecast, median | 8.79 / 8.48 / 9.15 | 9.26 / 8.97 / 9.68 |
| unconstrained statistic | 1.033 / 1.163 / 0.886 | 0.951 / 1.047 / 0.846 |
| the regime criterion | 0.105 / 0.098 / 0.123 | 0.091 / 0.068 / 0.138 |
| its optimiser MRAD | 0.135 / 0.150 / 0.125 | 0.093 / 0.062 / 0.169 |

- **The optimiser realises less** -- 0.50 and 0.36 of a point a year over the test, 0.84 and 0.70
  in the first half -- and its forecast rises to meet it: without CTVA's free diversification the
  least forecast variance is higher. The unconstrained portfolio realises a tenth of a point more
  in the calm half.
- **The first half's under-forecast was largely CTVA's:** the long-only portfolio reads 0.990 there
  now, against 1.109. In the calm half both read over-forecast (0.801, 0.846), with the index
  (0.783): the calm half's level, unchanged.
- **The long-only portfolio is now over-forecast over the whole test,** 0.900, below its band
  (0.946-1.054); at 5, 11 and 21 days 0.960, 0.895 and 0.878 (overlapping), against 1.042, 0.972
  and 0.911.
- **Wherever else anything moves:**
  - **the named portfolios:** 5 of 226 move by 0.005 or more in any part -- the two optimiser
    portfolios, SIC60's (0.971 / 1.051 / 0.884 to 0.964 / 1.040 / 0.883) and two random long-short
    portfolios by under 0.011; the index 0.934 to 0.931;
  - **the members:** 4 of 543 by 0.01 or more -- CTVA 0.867 / 0.810 / 0.926 to 0.879 / 0.832 / 0.928
    (its forecast 30.14% and 30.23% a year by half to 29.48% and 29.77%), and JPM, BAC and WFC
    down about 0.01, their factor now carrying the two failed banks' 2023; the median move 0.001;
    the quintiles within 0.001;
  - **the eigenportfolios:** within 0.016 at every rank and part (rank 15's first half 1.057 to
    1.073 the most);
  - **the factor covariance:** no factor volatility by 1% of itself at the median in any part; the
    factors' mean correlation 0.417 to 0.419;
  - **the regime multiplier:** lambda_F the same to the third decimal, 1.121 and 0.956 by half.

---

## Item 2: the market's own multiplier (C19)

**Built, in the run** (`--bias-country` now runs two country settings in its pass):
- **`country`,** midoffice18's: the regime multiplier over the basis's 26 factors.
- **`blocks`:** lambda_C from the country's own squared statistic and lambda_R from the other 25
  factors', each its own `RegimeHistory` from the basis's forecasts -- the same unadjusted
  forecasts C18's history reads, split -- on 21 days, one until 21 days; the basis's covariance
  scaled as D S D (`country.block_scaled`) before it is mapped back. Its specific side is the
  model's (off). The run checks the split: over the 26 factors B_F^2 is (B_C^2 + 25 B_R^2) / 26
  on all 685 days, to 4.4e-16.
- **Reported** (`_report_settings`): the blocks' two multiplier rows, lambda_C's path by quarter
  and half beside lambda_R, the country setting's and the model's, ruling 2's verdict line, and
  the country factor by part; `--bias-out` keeps `_blocks`, the scaled basis covariance, the two
  multipliers by date and both histories.

**Smoke runs first** (`smoke_last.py 30`, the last 30 forecast dates):
- **The old scheme:** every one of the 48 arrays midoffice18's country smoke kept is equal to the
  bit (`old_country_vs_18.txt`): the new setting and the holdings perturb nothing.
- **The regime adjustment off:** both multipliers one, so `blocks` must be `country` -- every pair
  of arrays equal to the bit (`off_blocks_vs_country.txt`).
- **The new defaults,** the guard on two dates: it passes with the scheme held.

**The run:** `--bias --bias-country --bias-horizons 1,5,11,21 --bias-guard 0 --industries
--membership`, the defaults otherwise: 9.7 minutes, 686 refits in 8.1 (0.71 s a refit with three
tests). Its model test is D19's to the bit in every array (`equal_C19_D19.txt`; 34 of 34 shared
arrays, `arrays_C19_D19.txt`), so the guard D19 ran covers it.

**Judged against D19** by the rule and, for the blocks, ruling 2's bar (`judge.py`, written before
the run; `judge_C19.txt`), whole / first / second:

| | long-only % a year | unconstrained | criterion | broad MRAD | optimiser MRAD | members MRAD |
|---|---|---|---|---|---|---|
| D19 | 9.62 / 11.05 / 7.97 | 9.83 / 11.25 / 8.18 | 0.091 / 0.068 / 0.138 | 0.094 / 0.067 / 0.160 | 0.093 / 0.062 / 0.169 | 0.085 / 0.076 / 0.084 |
| country | 9.47 / 10.95 / 7.72 | 9.62 / 11.13 / 7.84 | 0.113 / 0.078 / 0.200 | 0.121 / 0.081 / 0.245 | 0.127 / 0.068 / 0.267 | 0.089 / 0.083 / 0.087 |
| blocks | 9.17 / 10.59 / 7.50 | 9.88 / 11.74 / 7.59 | 0.069 / 0.059 / 0.067 | 0.062 / 0.061 / 0.053 | 0.059 / 0.034 / 0.061 | 0.087 / 0.081 / 0.086 |

- **The blocks: out, first, on the optimiser's test.** The unconstrained portfolio realises 9.88%
  a year against 9.83% over the whole test -- 0.49 of a point more in the first half, 0.59 less in
  the calm one. The long-only realises less in both halves (0.46 and 0.47 of a point). The rule
  takes either portfolio over the whole test, to the hundredth.
- **Their criterion would have met ruling 2's bar:** 0.059 against 0.068 on the first half, 0.067
  against 0.138 on the second. The optimiser's test comes first.
- **The country setting:** in on the optimiser (less in both halves, both portfolios), out on the
  first half's criterion, 0.078 against 0.068 -- C18's verdict again.
- **So D19 stays.**

**Beside them** (the run's report, `report_C19.txt`, and the judge):
- **The statistics:**

  | | index | equally weighted | long-only | unconstrained | members' median |
  |---|---|---|---|---|---|
  | D19 | 0.931 / 1.060 / 0.783 | 0.893 / 0.984 / 0.794 | 0.900 / 0.990 / 0.801 | 0.951 / 1.047 / 0.846 | 0.987 / 0.978 / 0.986 |
  | country | 0.926 / 1.097 / 0.716 | 0.880 / 1.016 / 0.721 | 0.867 / 0.999 / 0.714 | 0.908 / 1.049 / 0.742 | 0.971 / 0.998 / 0.935 |
  | blocks | 0.987 / 1.036 / 0.938 | 0.923 / 0.945 / 0.903 | 0.918 / 0.940 / 0.896 | 0.963 / 1.007 / 0.919 | 0.974 / 0.970 / 0.964 |

- **Rank 1 by half,** each setting's own: D19 1.064 and 0.814, country 1.095 and 0.747, blocks
  1.032 and 0.988 -- midoffice18's stored-array reading predicted 1.033 and 0.988.
- **The shape,** the eigenportfolios' mean abs(bias - 1): 0.067 / 0.222 by half under D19, 0.076 /
  0.075 country, 0.079 / 0.064 blocks; ranks 21-25 1.126 / 1.262, 1.136 / 1.100, 1.143 / 1.092.
- **The country factor** -- the industries' factor returns weighted by the outcome period's
  capitalisation shares, each setting's forecast of that portfolio -- its forecast % a year at the
  median and its statistic, first / second half: D19 13.73% 1.067 / 15.32% 0.795; country 13.65%
  1.104 / 16.88% 0.725; blocks 13.68% 1.046 / 12.52% 0.964.
- **The multipliers:**

  | mean (min-max) | whole | first half | second half |
  |---|---|---|---|
  | lambda_C (blocks) | 0.999 (0.545-2.349) | 1.164 (0.753-2.349) | 0.835 (0.545-1.200) |
  | lambda_R (blocks) | 1.066 (0.873-1.297) | 1.057 (0.873-1.293) | 1.075 (0.907-1.297) |
  | lambda_F (country) | 1.065 (0.875-1.349) | 1.063 (0.875-1.349) | 1.067 (0.900-1.282) |
  | lambda_F (model) | 1.039 (0.774-1.979) | 1.121 (0.862-1.979) | 0.956 (0.774-1.128) |

  lambda_C's low 0.545 on 2025-10-09, its high 2.349 on 2025-04-10; by quarter's end it ran
  0.892, 0.776, 1.131, 1.087, 1.272, 1.133, 0.583, 0.700, 1.086, 1.010, 0.795 and 0.858 (the
  report). midoffice18's reading from C18's arrays had 1.164 and 0.836 by half, 0.546 to 2.349.
- **Day to day,** the median absolute change in log forecast volatility (95th percentile): the
  index D19 1.00% (3.60%), country 0.79% (2.68%), blocks 1.62% (5.20%); the members 1.47%, 1.45%,
  1.40%.
- **The optimiser by eigen-direction** (each group's share of the forecast variance, the mean over
  dates; the statistic whole / first / second):

  | long-only | rank 1 | ranks 2-5 | ranks 6-15 | ranks 16-25 | specific |
  |---|---|---|---|---|---|
  | D19 | 47.1%, 0.95/1.06/0.81 | 15.8%, 1.04/0.92/1.14 | 9.4%, 1.12/1.06/1.18 | 21.6%, 1.18/1.13/1.22 | 6.1%, 0.86/0.89/0.83 |
  | blocks | 46.0%, 1.01/1.03/0.99 | 15.9%, 0.96/0.95/0.96 | 8.6%, 1.00/1.02/0.96 | 22.6%, 1.11/1.19/1.03 | 6.9%, 0.84/0.93/0.74 |
  | country | 47.6%, 0.94/1.10/0.75 | 16.4%, 0.95/0.93/0.97 | 8.4%, 1.01/1.03/0.99 | 22.0%, 1.11/1.16/1.05 | 5.6%, 0.89/0.92/0.85 |

  | unconstrained | rank 1 | ranks 2-5 | ranks 6-15 | ranks 16-25 | specific |
  |---|---|---|---|---|---|
  | D19 | 39.9%, 0.95/1.06/0.81 | 15.8%, 1.04/0.92/1.15 | 11.5%, 1.12/1.05/1.18 | 26.3%, 1.18/1.14/1.21 | 6.6%, 0.86/0.84/0.88 |
  | blocks | 39.2%, 1.01/1.03/0.99 | 16.3%, 0.96/0.96/0.96 | 10.1%, 0.99/1.01/0.95 | 27.6%, 1.12/1.20/1.04 | 6.8%, 0.88/0.86/0.90 |
  | country | 41.0%, 0.94/1.10/0.75 | 16.3%, 0.96/0.94/0.98 | 10.0%, 1.00/1.02/0.97 | 27.2%, 1.12/1.18/1.06 | 5.5%, 0.88/0.85/0.90 |

  Under the blocks every group reads nearer one over the test; in the first half the smallest
  directions read 1.19 and 1.20 against 1.13 and 1.14. That is where the unconstrained portfolio's
  extra first-half risk shows; that the market's multiplier, at up to 2.35, drives it there is a
  reading, not a test.
- **The members by capitalisation quintile,** largest first, first / second half: D19 1.027,
  0.983, 0.986, 0.941, 0.956 / 1.025, 0.992, 0.985, 0.944, 0.961; blocks 1.014, 0.979, 0.974,
  0.931, 0.948 / 1.018, 0.968, 0.968, 0.930, 0.938; country 1.045, 1.015, 1.004, 0.968, 0.969 /
  0.978, 0.944, 0.934, 0.898, 0.915.

---

## Item 3: the intermittent test

**Run alone fifty times** (`flaky_runs.py`, each run its own pytest process, `-q -rf -p
no:cacheprovider`, as the final checks run it; `flaky_before.txt`): **2 failed, 48 passed** --
runs 48 and 50, at 21:52Z. Each run recorded the memory available at its start (3.2 to 3.7 GB, the
two failures at 3.7 and 3.6 GB, with passes at both on either side) and its duration (3.4 to 4.6
s, the failures 3.9 s): neither separates the failures. pytest cut the assertion's text from the
`FAILED` line -- the test's id alone fills the 80 columns a subprocess gets -- so the values were
read in-process.

**What it depends on: an unseeded draw** (`flaky_probe.py`, the test's own computation repeated in
one process; `flaky_probe_before.txt`):
- **Twenty repetitions, seed 42 every time, gave twenty different sets of values,** and three
  failed the test's assertion.
- **Why:** `_compute_single_point` in `src/praxis/stats/surface.py` seeded only the first batch of
  random walks (`current_seed = None  # Only seed the first batch`). The test asks 50 samples a
  grid point, and a batch gives one sample per asset: 5 batches at 10 assets, 3 at 20, every one
  after the first drawn from the operating system's entropy (`RandomState(None)`).
- **How it fails:** the test asks the 5% value interpolated at the midpoint to lie within three
  spreads of the two diagonal corners' average, the spread |lo - hi|. When the corners happen to
  fall close together the tolerance collapses: in the failing repetitions |mid - avg| was 0.2205,
  0.0596 and 0.1805 against 3 x spread of 0.1182, 0.0539 and 0.0436.
- **Not the clock:** the code reads it only to log the elapsed time. **Not memory:** the store is
  an in-memory DuckDB per test, the computation sequential (`n_workers=1`).

**Fixed, as the brief allows -- the cause a missing seed** (`surface.py`, `_compute_single_point`):
every batch seeded from the requirement's seed -- the first by the seed itself, as before, so its
draws are unchanged, each later one by its own seed derived from it (`np.random.SeedSequence([seed,
batch])`); an unseeded requirement stays unseeded. After it:
- **Twenty repetitions: one set of values,** passing with margin (|mid - avg| 0.1199 against
  1.0078; `flaky_probe_after.txt`).
- **Fifty runs alone: 50 passed** (`flaky_after.txt`).
- **The whole suite** passes with it (below).

**Not fixed, the same pattern in two other places,** neither on the failing test's path:
`surface._compute_multi_point` (the same two lines) and `monte_carlo.generate_adf_critical_values`
(`current_seed = None  # Only seed the first batch`). Seeding them changes what their callers
draw; left for a ruling.

---

## Which I would choose, and why

1. **D19, by the rule.** The two blocks fail the optimiser's test -- the unconstrained portfolio
   realises 9.88% a year against 9.83% over the whole test -- and C18's multiplier fails the first
   half's criterion (0.078 against 0.068). The rule was fixed before C19 ran, and ruling 2 raised
   the bar for this design; it does not meet the ordinary one.
2. **So the factor side is settled** (17 ruling 3, 18 ruling 5), and the largest members' specific
   risk is next.
3. **What the blocks showed, for the record:** the calm half's over-forecast is the market
   direction's level, and a multiplier of its own fixes it -- the criterion 0.067 against 0.138 in
   the calm half, the index 0.938 against 0.783, the country factor 0.964 against 0.795 -- at the
   price of a noisier forecast (1.62% a day against 1.00%) and the unconstrained portfolio's
   first half. I would not reopen it now. If it is reopened, the question is that first half, and
   C19 keeps every setting's holdings to read it from.
4. **Ruling 3 as built:** Agriculture into SIC28 by business. The Manufacturing division, nearer by
   the model's returns, is the alternative; the difference to CTVA is a point or two of residual
   volatility, against the market's three that any industry gives (Decisions).

---

## Key decisions

1. **The rulings in a file the run reads, not a new panel.** The SIC codes are the scheme's input;
   so nothing in `data/` is written, and the old scheme is a switch (`--industry-overrides none`),
   its model M18's in every field to the bit -- the brief's "old scheme or panel kept and
   rebuildable".
2. **"Nearest" by business, the returns a check.** Known at the time, the classification's own
   basis, as Barra assigns industries; SIC28 is second by the returns before the test, 0.3 of a
   point behind. Recorded with its evidence; the alternative left to Jeff.
3. **Refusals, not fallbacks:** a group below the floor with no entry, a join to a non-factor, and
   Unclassified below the floor each stop the run; a code where the records carry one does not
   apply, and is listed.
4. **VYLR's code included,** though the ruling named three: it changes no number now and keeps a
   member from going without the market once it has a forecast. Jeff's to keep or drop.
5. **A standing check with a stop** (the memory's rule: a defect-catching diagnostic in the run
   report): member-days without the market, "none" under the rulings or the run stops.
6. **D16's holdings rebuilt, not rerun:** exact against everything D16 stored of the optimiser,
   so item 1's "before" is D16's own; the holdings stored from 19 on, and the rebuild checked
   against D19's stored ones (4.1e-13).
7. **The blocks inside `--bias-country`,** both settings in one pass as the brief asked; their
   multipliers inside the covariance, the specific side the model's; their histories checked as
   the basis's split (4.4e-16) and, with the regime adjustment off, the setting checked equal to
   the country setting to the bit.
8. **The rule fixed before each run:** C19's judge (midoffice17's rule unchanged, ruling 2's bar
   added) at 17:14, D19's comparison at 17:15, both before the runs' results.
9. **The surface fix kept to the failing test's path,** the first batch's draws unchanged; the two
   other copies reported, not touched.
10. **The rulings' reader in the library** (`refdata/industries.py`), tested, beside the count
    rulings' reader in `securities.py`; the run converts its errors to stops.
11. **No default changed beyond ruling 3.** The holdings in `--bias-out` and the second country
    setting are test outputs.

## Decisions left for Jeff

1. **The factor covariance:** D19's, the factor side settled -- the rule's verdict on C19.
2. **Ruling 3's details:**
   - Agriculture's nearest industry: SIC28 by CTVA's business (built), or the Manufacturing
     division by the model's returns (27.9% against 28.2% before the test, 24.4% against 26.9% in
     it);
   - VYLR's code, beyond the three named: keep or drop;
   - the two other unseeded batches (`surface._compute_multi_point`,
     `monte_carlo.generate_adf_critical_values`): fix or leave.
3. **The next piece:** the largest members' specific risk (18 ruling 5).

---

## Tests

**The suite:** 2,384 passed, 1 skipped, with `--ignore=tests/test_market_data.py`, failures named
(`-rf`): none (`final_checks.txt`). Before this piece 2,377: 7 new, and every other passes -- the
intermittent one now deterministic.

**New, 7:**
- `test_industries.py`, 5:
  - `test_a_group_below_the_floor_joins_its_nearest_industry_and_no_member_goes_without_one` --
    ruling 3: with `joins` a group below the floor joins its entry's industry, nothing is pooled,
    no member is factorless, the join recorded and reported; without, the pooling; an unused entry
    leaves the scheme the old one's.
  - `test_a_group_below_the_floor_with_no_entry_or_no_code_is_refused` -- the refusals: no entry, a
    join to a non-factor (four kinds, and a target itself below the floor), Unclassified below the
    floor; outside the universe a no-code member is left as before.
  - `test_the_rulings_file_is_read_and_its_codes_apply_only_where_the_records_have_none` -- the
    reader and its refusals; a code applies only where the records have none, the records' code
    standing and listed; a key that is not the panel's is refused.
  - `test_the_repo_rulings_file_gives_the_three_their_industries` -- the committed file: its codes,
    its join, evidence on every entry, ASCII.
  - `test_the_standing_check_counts_member_days_without_the_market` -- the check counts a member-day
    with every style and no industry, not one missing a style or not a member's; none after the
    join.
- `test_country.py`, 1: `test_two_blocks_scale_the_country_and_the_rest_apart_and_map_back` -- D S D
  exactly, symmetric; on one multiplier it is S lambda^2, so mapped back it is midoffice18's
  setting; the refusals.
- `test_bias.py`, 1: `test_the_minimum_variance_portfolios_holdings_are_kept` -- the holdings fully
  invested, the source of the stored exposures and the realised returns; long-only non-negative,
  held where positive.

**Measured, not unit-tested:** the old scheme's model M18's to the bit (twice); the smoke runs (the
old scheme's arrays midoffice18's to the bit; the blocks the country setting's with the regime
off); D19's guard on five dates; C19's model test D19's to the bit; the holdings' rebuild, exact
on D16 and D19.

**Hygiene:** every changed file parses and keeps its line endings, and all but one are ASCII
(`final_checks.txt`, and again at the end over all 13 paths, after this retro's appendix:
`final_hygiene.txt`). The hygiene script's "PROBLEMS" there is `surface.py` alone: the
box-drawing characters it had at HEAD, 7,584 non-ASCII bytes then and now, from line 6; the 12
lines this piece added are ASCII. **ruff:** clean over every changed mid-office Python file;
`surface.py`'s five findings are HEAD's, none in the lines changed. **The dry-run `git add`** names
the 13 paths, nothing staged; `git status` shows those 13 and nothing else.

---

## Failures and debugging trail

- **psutil is not installed:** the first fifty-run start stopped at a check before the loop began;
  the loop reads memory as the runners do (`GlobalMemoryStatusEx`).
- **A heredoc patch dropped `country.py`'s module docstring's closing quotes:** the replacement's
  own `"""` ended the patch's string. ruff's parse caught it; restored. A working note says so.
- **Two more heredocs failed on apostrophes** (the two-block patch, the judge's) -- written as patch
  files instead (`patch_blocks.py`, `make_judge.py`); one patch's anchor had a space too many, and
  its assertion stopped it before it wrote anything.
- **ruff:** three F541 and an E501 in the new lines, then an F401 (`json`) once the reader moved to
  the library; fixed.
- **A `git stash` round trip,** to compare `surface.py`'s ruff findings at HEAD, rewrote the working
  copies through `core.autocrlf`: `bias.py` and `country.py`, LF in the working copy, came back
  CRLF. The index is LF either way and the diff is the same; a working note says so.
- **`surface.py` holds box-drawing characters,** so the parse check needs UTF-8.
- **The final checks' hygiene line stopped at this retro,** not yet written then, and the dry-run
  `git add` with it; both rerun at the end (`final_hygiene.txt`).
- **The runs:** none failed. D19 11.7 minutes (5.1 GB available at its start), C19 9.7 (4.6 GB).

## Problems found, not fixed

1. **The unseeded batches in two more places:** `surface._compute_multi_point` and
   `monte_carlo.generate_adf_critical_values` (item 3).
2. **The calm half's level:** with every member on the market the optimiser reads over-forecast
   there with the index (0.801, 0.846, 0.783 under D19). The blocks fixed it and failed the
   optimiser's test.
3. **The blocks' unconstrained first half,** 0.49 of a point more realised: read by eigen-direction,
   not explained.
4. **DISCA and DISCK have no SIC code:** members only before the first estimated period, in no
   regression and with no forecast.
5. **`surface.py`'s five ruff findings at HEAD.**
6. **As before:** the largest members' specific risk (next); the constraint's spurious direction,
   should the basis ever be built; the two-half-life diagonal's last bit; FedEx Freight and Vylor
   without Momentum; six ruff findings in untouched files; `tests/test_market_data.py`'s
   `../core_repo/src`.

## Scratch

`claude/scratch/midoffice19/` (gitignored):

- **Ruling 3:** `nearest_probe.py` (CTVA, FRCB and SBNY under each industry, on midoffice18's fit);
  `model/` -- M19 (the new defaults, with inputs) and M19old (`--industry-overrides none`), their
  reports; `old_scheme_vs_M18.txt`, `model_M19_vs_M18.txt`.
- **Item 1:** `ctva_holdings.py`, `ctva_D16.txt`, `ctva_D19.txt`; `compare_d19.py`,
  `compare_D19_D16.txt`, `report_D19_vs_D16.txt`.
- **Item 2:** `patch_blocks.py`, `patch_blocks_report.py`, `patch_country_factor.py`,
  `patch_library_overrides.py`, `patch_ruff.py`, `patch_tests.py`; `smoke/` -- `old_country`,
  `off_country` and `new_country` (reports, progress, `.npz`), `compare_npz.py`,
  `old_country_vs_18.txt`, `off_blocks_vs_country.txt`; `make_judge.py`, `judge.py`,
  `judge_C19.txt`, `equal_C19_D19.txt`, `arrays_C19_D19.txt`.
- **Item 3:** `flaky_runs.py`, `flaky_before.txt`, `flaky_after.txt`, `flaky_probe.py`,
  `flaky_probe_before.txt` (the terminal's output, saved verbatim: the probe was run to the
  terminal first), `flaky_probe_after.txt`.
- **The runs:** `runner.py`; `bias/` -- D19's and C19's reports, progress, `.npz`, the runner's log
  and done files.
- **The write-up:** `retro_static.md`, `retro_second.md`, `state_current.txt`,
  `patch_state_*.py`; `final_checks.sh` and `.txt`, `final/`, `final_hygiene.txt`;
  `append_appendix.py`.

The harnesses of earlier pieces were used where they are: `capture.py`, `compare.py`,
`hygiene.py` (13), `compare_panels.py` (14), `smoke_last.py` (15), `equal_correlation.py` (16),
`judge.py` (17, as the base of 19's).

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run` and names every changed and new path of this piece, the state file included.

---

## Open items for Chat

- **The decisions above,** ruling 3's details among them.
- **The next piece, the largest members' specific risk:** every later candidate is one bias run
  against D19, judged by a script written before it (`claude/scratch/midoffice19/judge.py`); the
  holdings are stored now, so a member's place in the optimiser is read directly.

---

## Appendix: outputs, verbatim

Each output as its script wrote it, from `claude/scratch/midoffice19/`.

### nearest_probe.txt -- ruling 3: CTVA, FRCB and SBNY put in each industry, their residual volatility a year, on midoffice18's fit

```text

CTVA: 938 periods with a return and its styles, 2023-02-08 to 2026-11-03
  pre-test (< 2024-02-09): n 252, no industry 30.6% a year; best: Manufacturing 27.9% (corr +0.42), SIC28 28.2% (corr +0.39), SIC37 28.7% (corr +0.37), Retail 28.7% (corr +0.35), SIC20 28.9% (corr +0.34), SIC35 29.0% (corr +0.36)
     SIC20 28.9%  SIC28 28.2%  SIC60 32.9%  Manufacturing 27.9%
  test: n 686, no industry 28.9% a year; best: Manufacturing 24.4% (corr +0.53), SIC35 26.1% (corr +0.48), Services 26.3% (corr +0.43), Retail 26.4% (corr +0.42), SIC38 26.7% (corr +0.42), Finance & Real Estate 26.7% (corr +0.43)
     SIC20 27.5%  SIC28 26.9%  SIC60 29.2%  Manufacturing 24.4%
  whole: n 938, no industry 29.3% a year; best: Manufacturing 25.4% (corr +0.50), SIC35 26.9% (corr +0.45), Retail 27.0% (corr +0.40), Services 27.2% (corr +0.40), SIC28 27.2% (corr +0.38), Finance & Real Estate 27.4% (corr +0.41)
     SIC20 27.9%  SIC28 27.2%  SIC60 30.2%  Manufacturing 25.4%

FRCB: 56 periods with a return and its styles, 2023-02-08 to 2023-04-28
  pre-test (< 2024-02-09): n 56, no industry 255.2% a year; best: SIC60 237.8% (corr +0.56), SIC62 246.9% (corr +0.44), Finance & Real Estate 248.1% (corr +0.43), SIC63 249.0% (corr +0.36), Mining 249.4% (corr +0.29), Manufacturing 249.7% (corr +0.36)
     SIC20 255.0%  SIC28 252.6%  SIC60 237.8%  Manufacturing 249.7%
  whole: n 56, no industry 255.2% a year; best: SIC60 237.8% (corr +0.56), SIC62 246.9% (corr +0.44), Finance & Real Estate 248.1% (corr +0.43), SIC63 249.0% (corr +0.36), Mining 249.4% (corr +0.29), Manufacturing 249.7% (corr +0.36)
     SIC20 255.0%  SIC28 252.6%  SIC60 237.8%  Manufacturing 249.7%

SBNY: 22 periods with a return and its styles, 2023-02-08 to 2023-03-10
  pre-test (< 2024-02-09): n 22, no industry 77.3% a year; best: SIC62 60.2% (corr +0.76), SIC60 65.1% (corr +0.47), SIC67 65.2% (corr +0.61), Finance & Real Estate 66.5% (corr +0.59), SIC35 67.0% (corr +0.63), Services 68.3% (corr +0.52)
     SIC20 71.9%  SIC28 71.4%  SIC60 65.1%  Manufacturing 70.3%
  whole: n 22, no industry 77.3% a year; best: SIC62 60.2% (corr +0.76), SIC60 65.1% (corr +0.47), SIC67 65.2% (corr +0.61), Finance & Real Estate 66.5% (corr +0.59), SIC35 67.0% (corr +0.63), Services 68.3% (corr +0.52)
     SIC20 71.9%  SIC28 71.4%  SIC60 65.1%  Manufacturing 70.3%
```

### composition.txt -- ruling 3: what the candidate industries are made of

```text
SIC28: 40 members in the universe; the last date's regression weight by three-digit SIC: 283 0.67, 284 0.15, 281 0.07, 282 0.05, 286 0.03, 287 0.02, 285 0.01, 280 0.00
Manufacturing: 39 members in the universe; the last date's regression weight by three-digit SIC: 291 0.34, 211 0.11, 335 0.07, 232 0.06, 331 0.05, 349 0.04, 302 0.04, 267 0.04, 265 0.03, 342 0.03, 324 0.03, 348 0.02, 271 0.02, 310 0.02, 399 0.02, 262 0.02, 341 0.01, 394 0.01, 343 0.01, 230 0.01, 227 0.01, 308 0.01
  MO 2111, PM 2111, MHK 2273, LULU 2300, CTAS 2320, RL 2320, VFC 2320, IP 2621, PKG 2650, SW 2650, WRK 2650, AVY 2670, KMB 2670, NWS 2711, NWSA 2711, COP 2911, CVX 2911, HES 2911, MPC 2911, PSX 2911, VLO 2911, XOM 2911, DECK 3021, NKE 3021, NWL 3089, TPR 3100, CRH 3241, NUE 3312, STLD 3312, HWM 3350, GLW 3357, BALL 3411, SNA 3420, SWK 3420, MAS 3430, AXON 3480, PH 3490, HAS 3944, AMCR 3990
SIC20: 21 members in the universe; the last date's regression weight by three-digit SIC: 208 0.52, 200 0.12, 203 0.10, 207 0.09, 201 0.06, 206 0.04, 204 0.04, 209 0.03
Mining: 16 members in the universe; the last date's regression weight by three-digit SIC: 131 0.50, 138 0.15, 140 0.13, 104 0.12, 100 0.10
```

### model/report_M19.txt -- ruling 3: the new defaults' industries section

```text
Industries
  each security's SIC code from its own latest record with one: 587 of 592; by override, with its evidence: FRCB 6022, SBNY 6022, VYLR 0100; none for DISCA, DISCK
  the industries' rulings (--industry-overrides config/sp500_industry_overrides.json): no member goes without the market (ruling 3 of midoffice19's brief) -- 3 code(s), 1 join(s)
  567 of 592 securities are in at least one estimated period's regression while an index member; every count below is over them
  20 industries over 567 securities in the universe (592 classified); largest SIC73 with 70 (12%), 43 thin group(s) merged into their division, 1 group(s) below the hard floor of 5 joined their nearest industry: Agriculture (1) into SIC28
    SIC73                      70
    SIC38                      44
    SIC28                      41
    Manufacturing              39
    SIC49                      37
    SIC36                      36
    Retail                     34
    SIC35                      34
  - 3 industry/industries have no member in the universe and are not factors: SIC24, SIC25, Unclassified
  - 1 group(s) had fewer than 5 members even after merging into their division and joined their nearest industry instead of going without one, their industry effect the nearest industry's: Agriculture into SIC28 (CTVA)
  - 3 industry/industries still have fewer than 15 members after merging and their factor returns are largely their members' specific returns: Construction (8), Finance & Real Estate (12), Wholesale (12)
  - 25 classified security(ies) are outside the universe and count toward no industry: ABMD, AMTM, CERN, CTXS, DISCA
  members without the market -- member-days from the first estimated period (2023-02-08) with every style exposure and no industry factor: none
```

### old_scheme_vs_M18.txt -- ruling 3: --industry-overrides none against midoffice18's model

```text
claude/scratch/midoffice18/model/M18 against claude/scratch/midoffice19/model/M19old: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 163 and 170 lines; 7 differing line(s)
    +  --industry-overrides none: the records' codes alone, and a group below the hard floor pooled into Other Industries (midoffice18's scheme)
    +  members without the market -- member-days from the first estimated period (2023-02-08) with every style exposure and no industry factor: 996, CTVA 917, FRCB 57, SBNY 22
    +
    +The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    +  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
    +  the last day's B_F^2 0.424 over 25 factors, the specific side not read
    +  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.2 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.1 s)
```

### model_M19_vs_M18.txt -- ruling 3: the new defaults' model against midoffice18's

```text
claude/scratch/midoffice18/model/M18 against claude/scratch/midoffice19/model/M19: 21 fields compared; equal: 9; differing: covariance, observed_variance, sampling_variance, weighted_sampling_variance, noise_share, weighted_noise_share, mean_r2, factor_returns, specific, exposures, assignment, counts
  covariance: largest difference 4.283e-04
  observed_variance: largest difference 2.359e-04
  sampling_variance: largest difference 2.795e-05
  weighted_sampling_variance: largest difference 3.622e-05
  noise_share: largest difference 2.698e-03
  weighted_noise_share: largest difference 3.416e-03
  mean_r2: largest difference 6.408e-04
  factor_returns: largest difference 1.557e-02
  specific: largest difference 5.054e-03
  exposures: largest difference 0.000e+00
  assignment: largest difference nan
  counts: largest difference 2.000e+00
  qualifications: 6 and 6; differing: 0
  report: 163 and 168 lines; 79 differing line(s)
    -  each security's SIC code from its own latest record with one: 587 of 592; none for DISCA, DISCK, FRCB, SBNY, VYLR
    +  each security's SIC code from its own latest record with one: 587 of 592; by override, with its evidence: FRCB 6022, SBNY 6022, VYLR 0100; none for DISCA, DISCK
    +  the industries' rulings (--industry-overrides config/sp500_industry_overrides.json): no member goes without the market (ruling 3 of midoffice19's brief) -- 3 code(s), 1 join(s)
    -  20 industries over 567 securities in the universe (592 classified); largest SIC73 with 70 (12%), 45 thin group(s) merged into their division
    +  20 industries over 567 securities in the universe (592 classified); largest SIC73 with 70 (12%), 43 thin group(s) merged into their division, 1 group(s) below the hard floor of 5 joined their nearest industry: Agriculture (1) into SIC28
    -    SIC28                      40
    +    SIC28                      41
    -  - 2 industry/industries have no member in the universe and are not factors: SIC24, SIC25
    -  - 2 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture, Unclassified
    -  - Other Industries pooled only 3, below the hard floor of 5, so it is not a factor. 3 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA, FRCB, SBNY
    +  - 3 industry/industries have no member in the universe and are not factors: SIC24, SIC25, Unclassified
    +  - 1 group(s) had fewer than 5 members even after merging into their division and joined their nearest industry instead of going without one, their industry effect the nearest industry's: Agriculture into SIC28 (CTVA)
    -  - 2 security(ies) have no usable SIC code and are left with no industry factor: FRCB, SBNY
    +  members without the market -- member-days from the first estimated period (2023-02-08) with every style exposure and no industry factor: none
    -  Construction               22.23%     +25.53%    28.9%      37.8%      938          0      0
    -  Finance & Real Estate      17.34%     +17.97%    24.1%      37.1%      938          0      0
    -  Manufacturing              12.07%     +20.71%    16.5%      24.3%      938          0      0
    -  Mining                     21.16%     +14.75%    14.2%      21.4%      938          0      0
    -  Retail                     14.75%     +14.55%     8.4%      12.9%      938          0      0
    -  SIC20                      13.35%     +10.88%    26.7%      45.3%      938          0      0
    -  SIC28                      16.14%     +16.54%     6.7%      10.6%      938          0      0
    -  SIC35                      18.56%     +29.73%     5.3%       7.3%      938          0      0
    -  SIC36                      24.22%     +32.61%     3.4%       4.1%      938          0      0
    -  SIC37                      19.59%     +15.49%    10.0%      14.8%      938          0      0
    -  SIC38                      16.83%     +16.01%     8.0%      12.3%      938          0      0
    -  SIC49                      19.02%     +23.73%     7.7%      11.9%      938          0      0
    -  SIC60                      23.21%     +21.78%     6.5%       9.5%      938          0      0
    -  SIC62                      21.57%     +18.48%     8.6%      11.5%      938          0      0
    -  SIC63                      14.22%     +19.00%    15.1%      24.2%      938          0      0
    -  SIC67                      17.83%     +17.79%    11.8%      18.3%      938          0      0
    -  SIC73                      17.53%     +15.26%     3.1%       4.9%      938          0      0
    -  Services                   14.47%     +15.95%    21.5%      33.5%      938          0      0
    -  Transport & Utilities      16.26%     +15.54%    11.2%      17.5%      938          0      0
    -  Wholesale                  15.06%     +20.46%    39.5%      60.4%      938          0      0
    -  Volatility                 10.62%      +2.98%     2.7%       4.5%      938          0      0
    -  Momentum                    7.49%      +4.29%     3.4%       4.8%      938          0      0
    -  Size                        3.62%      +3.05%     8.0%      11.4%      938          0      0
    -  Size Nonlinearity           2.83%      -0.48%    31.0%      45.9%      938          0      0
    -  Trading Activity            5.32%      +2.50%    10.3%      14.7%      938          0      0
    +  Construction               22.26%     +25.64%    28.8%      37.7%      938          0      0
    +  Finance & Real Estate      17.38%     +18.00%    23.9%      36.9%      938          0      0
    +  Manufacturing              12.11%     +20.80%    16.3%      24.2%      938          0      0
    +  Mining                     21.20%     +14.80%    14.1%      21.3%      938          0      0
    +  Retail                     14.77%     +14.56%     8.4%      12.9%      938          0      0
    +  SIC20                      13.40%     +10.99%    26.5%      45.0%      938          0      0
    +  SIC28                      16.07%     +16.78%     6.7%      10.5%      938          0      0
    +  SIC35                      18.57%     +29.76%     5.3%       7.3%      938          0      0
    +  SIC36                      24.20%     +32.57%     3.4%       4.1%      938          0      0
    +  SIC37                      19.60%     +15.49%    10.0%      14.8%      938          0      0
    +  SIC38                      16.86%     +16.03%     8.0%      12.3%      938          0      0
    +  SIC49                      19.08%     +23.81%     7.7%      11.8%      938          0      0
    +  SIC60                      23.24%     +20.45%     6.5%       9.4%      938          0      0
    +  SIC62                      21.60%     +18.47%     8.6%      11.5%      938          0      0
    +  SIC63                      14.25%     +19.06%    15.1%      24.2%      938          0      0
    +  SIC67                      17.89%     +17.77%    11.7%      18.2%      938          0      0
    +  SIC73                      17.53%     +15.23%     3.1%       4.9%      938          0      0
    +  Services                   14.51%     +15.94%    21.4%      33.4%      938          0      0
    +  Transport & Utilities      16.29%     +15.54%    11.1%      17.4%      938          0      0
    +  Wholesale                  15.10%     +20.54%    39.2%      60.0%      938          0      0
    +  Volatility                 10.63%      +3.03%     2.7%       4.5%      938          0      0
    +  Momentum                    7.48%      +4.18%     3.3%       4.8%      938          0      0
    +  Size                        3.62%      +3.04%     8.0%      11.5%      938          0      0
    +  Size Nonlinearity           2.84%      -0.48%    30.8%      45.6%      938          0      0
    +  Trading Activity            5.32%      +2.52%    10.3%      14.7%      938          0      0
    -  specific risk   median 28.99%   p10 16.56%   p90 48.89%
    -  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, SMCI 1.9%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    -                    relative residual spread 0.133, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.345
    -                    largest shares of the sd's square: FRCB 68.4%, SIVB 3.5%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    +  specific risk   median 28.97%   p10 16.67%   p90 48.88%
    +  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, SMCI 1.9%, FRCB 1.5%, FSLR 1.1%, ENPH 1.0%
    +                    relative residual spread 0.134, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.322
    +                    largest shares of the sd's square: FRCB 64.8%, SIVB 4.2%, SMCI 2.0%, LUMN 2.0%, SKYD 0.9%
    -                    each history weighted by a half-life of 63 trading days; the characteristics' weight in a blended forecast, at the median effective history of 181.8 observations of 938: 13.59%
    +                    each history weighted by a half-life of 63 trading days; the characteristics' weight in a blended forecast, at the median effective history of 181.8 observations of 938: 13.36%
    +
    +The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    +  as of 2026-11-03: lambda_F 0.9629, lambda_S 1.0000; the factor volatilities' median 16.29% to 15.68%, the specific risk's p10 / median / p90 16.67% / 28.97% / 48.88% to 16.67% / 28.97% / 48.88%
    +  the last day's B_F^2 0.422 over 25 factors, the specific side not read
    +  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.3 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.3 s)
```

### ctva_D16.txt -- item 1, before the fix: D16's holdings rebuilt, CTVA's weight and share

```text
D16.npz: 686 forecast dates, 2024-02-08 to 2026-11-02; the halves 343 and 343; the exposures from M18_inputs.pkl
the rebuild against what the run stored, the largest differences: the optimiser's factor exposures 9.8e-13 of the largest; its forecast 1.3e-15 relative; its realised return 1.3e-14; held counts wrong on 0 of 1372 portfolio-dates; no holdings stored
the holdings read: rebuilt

minimum variance, long-only: CTVA
  whole        weight mean 0.1089, median 0.1053, max 0.3283 (2025-04-16); held on 666 of 686 dates; the largest holding on 520, the largest other's weight 0.0826 at the mean; its share of the realised variance +0.2076; the portfolio 10.13% a year, its returns less CTVA's piece 8.39%
  first half   weight mean 0.1119, median 0.1009, max 0.3283 (2025-04-16); held on 343 of 343 dates; the largest holding on 224, the largest other's weight 0.0841 at the mean; its share of the realised variance +0.2102; the portfolio 11.88% a year, its returns less CTVA's piece 9.69%
  second half  weight mean 0.1060, median 0.1142, max 0.1659 (2025-06-24); held on 323 of 343 dates; the largest holding on 296, the largest other's weight 0.0810 at the mean; its share of the realised variance +0.2019; the portfolio 8.02% a year, its returns less CTVA's piece 6.86%

minimum variance, unconstrained: CTVA
  whole        weight mean 0.1071, median 0.1031, max 0.3273 (2025-04-16); held on 666 of 686 dates; the largest holding on 642, the largest other's weight 0.0462 at the mean; its share of the realised variance +0.1968; the portfolio 10.19% a year, its returns less CTVA's piece 8.55%
  first half   weight mean 0.1123, median 0.1017, max 0.3273 (2025-04-16); held on 343 of 343 dates; the largest holding on 319, the largest other's weight 0.0450 at the mean; its share of the realised variance +0.2033; the portfolio 11.95% a year, its returns less CTVA's piece 9.84%
  second half  weight mean 0.1019, median 0.1046, max 0.1569 (2025-06-24); held on 323 of 343 dates; the largest holding on 323, the largest other's weight 0.0475 at the mean; its share of the realised variance +0.1824; the portfolio 8.08% a year, its returns less CTVA's piece 7.05%
```

### ctva_D19.txt -- item 1, after it: D19's stored holdings (and the rebuild against them)

```text
D19.npz: 686 forecast dates, 2024-02-08 to 2026-11-02; the halves 343 and 343; the exposures from M19_inputs.pkl
the rebuild against what the run stored, the largest differences: the optimiser's factor exposures 8.7e-13 of the largest; its forecast 1.6e-15 relative; its realised return 1.8e-14; held counts wrong on 0 of 1372 portfolio-dates; the stored holdings 4.1e-13
the holdings read: as stored

minimum variance, long-only: CTVA
  whole        weight mean 0.0000, median 0.0000, max 0.0029 (2025-03-10); held on 38 of 686 dates; the largest holding on 0, the largest other's weight 0.0887 at the mean; its share of the realised variance +0.0002; the portfolio 9.62% a year, its returns less CTVA's piece 9.62%
  first half   weight mean 0.0001, median 0.0000, max 0.0029 (2025-03-10); held on 24 of 343 dates; the largest holding on 0, the largest other's weight 0.0885 at the mean; its share of the realised variance +0.0003; the portfolio 11.05% a year, its returns less CTVA's piece 11.04%
  second half  weight mean 0.0000, median 0.0000, max 0.0018 (2026-04-10); held on 14 of 343 dates; the largest holding on 0, the largest other's weight 0.0889 at the mean; its share of the realised variance -0.0000; the portfolio 7.97% a year, its returns less CTVA's piece 7.97%

minimum variance, unconstrained: CTVA
  whole        weight mean -0.0002, median -0.0002, max 0.0037 (2025-04-03); held on 294 of 686 dates; the largest holding on 0, the largest other's weight 0.0491 at the mean; its share of the realised variance +0.0007; the portfolio 9.83% a year, its returns less CTVA's piece 9.82%
  first half   weight mean -0.0001, median -0.0002, max 0.0037 (2025-04-03); held on 154 of 343 dates; the largest holding on 0, the largest other's weight 0.0476 at the mean; its share of the realised variance +0.0013; the portfolio 11.25% a year, its returns less CTVA's piece 11.24%
  second half  weight mean -0.0003, median -0.0001, max 0.0023 (2025-09-19); held on 140 of 343 dates; the largest holding on 0, the largest other's weight 0.0505 at the mean; its share of the realised variance -0.0004; the portfolio 8.18% a year, its returns less CTVA's piece 8.18%
```

### compare_D19_D16.txt -- item 1: D19 against D16, wherever anything moves

```text
D16.npz against D19.npz: 686 forecast dates, 2024-02-08 to 2026-11-02, the halves 343 and 343
  the same members on the same days: True; the same eligible member-days: True

1. The readings by part, whole / first / second: D16, D19, and D19 less D16
  eigenportfolios 1-5, mean                              1.011 /   0.985 /   1.031     1.010 /   0.983 /   1.031    -0.001 /  -0.002 /  +0.000
  eigenportfolios 6-10, mean                             1.084 /   0.957 /   1.198     1.084 /   0.956 /   1.198    -0.000 /  -0.001 /  +0.001
  eigenportfolios 11-15, mean                            1.111 /   1.013 /   1.199     1.112 /   1.016 /   1.200    +0.001 /  +0.003 /  +0.000
  eigenportfolios 16-20, mean                            1.223 /   1.084 /   1.344     1.222 /   1.079 /   1.346    -0.001 /  -0.005 /  +0.002
  eigenportfolios 21-25, mean                            1.195 /   1.124 /   1.261     1.195 /   1.126 /   1.262    +0.001 /  +0.002 /  +0.000
  eigenportfolios, mean abs(bias - 1)                    0.131 /   0.067 /   0.222     0.131 /   0.067 /   0.222    +0.000 /  +0.000 /  +0.001
  minimum variance, long-only                            0.990 /   1.109 /   0.855     0.900 /   0.990 /   0.801    -0.090 /  -0.119 /  -0.054
  minimum variance, unconstrained                        1.033 /   1.163 /   0.886     0.951 /   1.047 /   0.846    -0.082 /  -0.116 /  -0.040
  minimum variance, mean abs(bias - 1)                   0.022 /   0.136 /   0.129     0.074 /   0.029 /   0.176    +0.053 /  -0.108 /  +0.047
  minimum variance, long-only: realised vol, %          10.128 /  11.881 /   8.018     9.624 /  11.046 /   7.968    -0.504 /  -0.835 /  -0.050
  minimum variance, unconstrained: realised vol, %      10.193 /  11.953 /   8.077     9.829 /  11.252 /   8.179    -0.364 /  -0.701 /  +0.102
  the criterion                                          0.076 /   0.101 /   0.175     0.103 /   0.048 /   0.199    +0.026 /  -0.054 /  +0.024
  the index, cap-weighted                                0.934 /   1.063 /   0.785     0.931 /   1.060 /   0.783    -0.002 /  -0.003 /  -0.001
  the index, equally weighted                            0.893 /   0.983 /   0.793     0.893 /   0.984 /   0.794    +0.000 /  +0.000 /  +0.000
  smallest 85, equally weighted                          0.902 /   0.921 /   0.882     0.903 /   0.923 /   0.883    +0.001 /  +0.002 /  +0.001
  largest 50, equally weighted                           0.920 /   1.024 /   0.804     0.918 /   1.021 /   0.803    -0.002 /  -0.003 /  -0.001
  members, median                                        0.987 /   0.978 /   0.986     0.987 /   0.978 /   0.986    -0.000 /  +0.000 /  +0.001
  members, mean abs(bias - 1)                            0.088 /   0.099 /   0.098     0.088 /   0.099 /   0.097    -0.000 /  -0.000 /  -0.000
  industries, mean abs(bias - 1)                         0.055 /   0.065 /   0.088     0.055 /   0.064 /   0.089    +0.000 /  -0.001 /  +0.000
  random long-only, median                               0.914 /   0.983 /   0.838     0.915 /   0.985 /   0.839    +0.001 /  +0.002 /  +0.002
  random long-short, median                              0.997 /   0.974 /   1.015     0.997 /   0.974 /   1.016    +0.000 /  -0.000 /  +0.000
  broad portfolios, mean MRAD                            0.094 /   0.068 /   0.159     0.094 /   0.067 /   0.160    +0.000 /  -0.001 /  +0.000
  minimum variance, mean MRAD                            0.135 /   0.150 /   0.125     0.093 /   0.062 /   0.169    -0.041 /  -0.088 /  +0.045
  members, median MRAD                                   0.086 /   0.076 /   0.084     0.085 /   0.076 /   0.084    -0.001 /  +0.000 /  +0.000
  the regime criterion                                   0.105 /   0.098 /   0.123     0.091 /   0.068 /   0.138    -0.014 /  -0.030 /  +0.015

2. The named portfolios' one-day statistic by part: those that move by 0.005 or more in any part, largest first (D16 -> D19, whole / first / second)
  5 of 226 move by 0.005 or more; 226 move at all; the largest move 0.1191
  minimum variance, long-only              0.990 / 1.109 / 0.855  ->  0.900 / 0.990 / 0.801
  minimum variance, unconstrained          1.033 / 1.163 / 0.886  ->  0.951 / 1.047 / 0.846
  industry: SIC60                          0.971 / 1.051 / 0.884  ->  0.964 / 1.040 / 0.883
  random long-short 27                     1.060 / 0.955 / 1.156  ->  1.053 / 0.945 / 1.152
  random long-short 30                     1.073 / 1.002 / 1.140  ->  1.080 / 1.009 / 1.146
  minimum variance, long-only, D16: forecast median 9.05 / 8.76 / 9.42% a year; realised 10.13 / 11.88 / 8.02%
  minimum variance, long-only, D19: forecast median 9.55 / 9.23 / 10.02% a year; realised 9.62 / 11.05 / 7.97%
  minimum variance, unconstrained, D16: forecast median 8.79 / 8.48 / 9.15% a year; realised 10.19 / 11.95 / 8.08%
  minimum variance, unconstrained, D19: forecast median 9.26 / 8.97 / 9.68% a year; realised 9.83 / 11.25 / 8.18%

3. The members: each one's statistic, whole / first / second, those that move by 0.01 or more in any part (a quarter's forecasts or more in it), largest first
  543 members tested; 4 move by 0.01 or more, 543 at all; the median move 0.0010
  CTVA   0.867 / 0.810 / 0.926  ->  0.879 / 0.832 / 0.928   (Agriculture (factorless) to SIC28)
  JPM    0.946 / 1.029 / 0.854  ->  0.935 / 1.014 / 0.850
  BAC    1.004 / 1.069 / 0.935  ->  0.995 / 1.056 / 0.931
  WFC    1.071 / 1.139 / 0.999  ->  1.063 / 1.127 / 0.996
  CTVA D16: forecast 30.14 / 30.23% a year at the median by half, its statistic 0.867 / 0.810 / 0.926, on 666 dates
  CTVA D19: forecast 29.48 / 29.77% a year at the median by half, its statistic 0.879 / 0.832 / 0.928, on 666 dates
  FRCB: no forecast in the test (not a member with a forecast on any date)
  SBNY: no forecast in the test (not a member with a forecast on any date)

4. The eigenportfolios by rank, whole / first / second: D16 -> D19
   1  0.946 / 1.064 / 0.814  ->  0.947 / 1.064 / 0.814
   2  0.978 / 0.912 / 1.040  ->  0.976 / 0.910 / 1.039
   3  1.051 / 0.970 / 1.127  ->  1.050 / 0.973 / 1.124
   4  1.033 / 1.017 / 1.048  ->  1.031 / 1.012 / 1.049
   5  1.047 / 0.963 / 1.125  ->  1.046 / 0.956 / 1.129
   6  1.041 / 0.911 / 1.158  ->  1.043 / 0.912 / 1.160
   7  1.084 / 0.979 / 1.182  ->  1.084 / 0.978 / 1.182
   8  1.083 / 0.996 / 1.165  ->  1.085 / 0.994 / 1.169
   9  1.128 / 0.942 / 1.288  ->  1.126 / 0.943 / 1.285
  10  1.084 / 0.959 / 1.196  ->  1.082 / 0.955 / 1.196
  11  1.088 / 0.942 / 1.216  ->  1.087 / 0.942 / 1.216
  12  1.098 / 0.998 / 1.190  ->  1.098 / 0.995 / 1.192
  13  1.096 / 1.062 / 1.128  ->  1.098 / 1.065 / 1.131
  14  1.109 / 1.006 / 1.205  ->  1.106 / 1.002 / 1.202
  15  1.163 / 1.057 / 1.258  ->  1.169 / 1.073 / 1.257
  16  1.186 / 1.122 / 1.249  ->  1.184 / 1.109 / 1.256
  17  1.156 / 1.058 / 1.247  ->  1.157 / 1.053 / 1.254
  18  1.245 / 1.040 / 1.420  ->  1.244 / 1.046 / 1.412
  19  1.288 / 1.112 / 1.444  ->  1.284 / 1.098 / 1.447
  20  1.238 / 1.088 / 1.362  ->  1.239 / 1.087 / 1.365
  21  1.181 / 1.149 / 1.215  ->  1.183 / 1.150 / 1.217
  22  1.177 / 1.115 / 1.238  ->  1.176 / 1.114 / 1.237
  23  1.224 / 1.160 / 1.285  ->  1.225 / 1.162 / 1.286
  24  1.255 / 1.174 / 1.330  ->  1.257 / 1.183 / 1.327
  25  1.135 / 1.021 / 1.239  ->  1.136 / 1.021 / 1.241

5. Each factor's forecast volatility, % a year at the median by part (whole / first / second), those that move by 1% of it or more; and the factors' mean correlation by half
  D16: the factors' mean correlation 0.417 / 0.402 / 0.432
  D19: the factors' mean correlation 0.419 / 0.404 / 0.434

6. lambda_F as each forecast carried it, from each run's history on 21 days: the mean (min to max) by part
  D16: whole 1.039 (0.776-1.977)   first 1.121 (0.864-1.977)   second 0.956 (0.776-1.129); B_F^2 by half 1.332 / 0.899
  D19: whole 1.039 (0.774-1.979)   first 1.121 (0.862-1.979)   second 0.956 (0.774-1.128); B_F^2 by half 1.332 / 0.898
```

### report_D19_vs_D16.txt -- item 1: D19's report against D16's, the bias section's changed lines

```text
report_D16.txt against report_D19.txt, from 'Out-of-sample bias statistics': 100 and 100 lines
73 lines out, 73 in
-  686 refits in 7.4 min; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every po
+  686 refits in 7.8 min; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every po
-  the index, cap-weighted              686   0.934  over  0.946-1.054   0.098  0.764  1.157     14.73%   15.27%
-  the index, equally weighted          686   0.893  over  0.946-1.054   0.089  0.798  1.067     13.81%   13.96%
-  smallest 85, equally weighted        686   0.902  over  0.946-1.054   0.084  0.840  0.995     17.47%   17.38%
-  largest 50, equally weighted         686   0.920  over  0.946-1.054   0.104  0.737  1.109     14.41%   14.64%
-  minimum variance, long-only          686   0.990    in  0.946-1.054   0.126  0.792  1.211      9.05%   10.13%
-  minimum variance, unconstrained      686   1.033    in  0.946-1.054   0.143  0.833  1.274      8.79%   10.19%
-    minimum variance, long-only: 86 names held at the median (28 to 145); its KKT conditions met to 8.6e-13; realised 10.13% a year, the first half 11.88% (bias 1.109), the second 8.02% (0.855)
-    minimum variance, unconstrained: 501 names held at the median (499 to 503); its KKT conditions met to 1.0e-12; realised 10.19% a year, the first half 11.95% (bias 1.163), the second 8.08% (0.886)
+  the index, cap-weighted              686   0.931  over  0.946-1.054   0.099  0.762  1.154     14.76%   15.27%
+  the index, equally weighted          686   0.893  over  0.946-1.054   0.089  0.799  1.068     13.80%   13.96%
+  smallest 85, equally weighted        686   0.903  over  0.946-1.054   0.083  0.843  0.996     17.45%   17.38%
+  largest 50, equally weighted         686   0.918  over  0.946-1.054   0.105  0.735  1.106     14.44%   14.64%
+  minimum variance, long-only          686   0.900  over  0.946-1.054   0.104  0.739  1.071      9.55%    9.62%
+  minimum variance, unconstrained      686   0.951    in  0.946-1.054   0.082  0.801  1.137      9.26%    9.83%
+    minimum variance, long-only: 93 names held at the median (37 to 148); its KKT conditions met to 7.9e-13; realised 9.62% a year, the first half 11.05% (bias 0.990), the second 7.97% (0.801)
+    minimum variance, unconstrained: 501 names held at the median (499 to 503); its KKT conditions met to 1.1e-12; realised 9.83% a year, the first half 11.25% (bias 1.047), the second 8.18% (0.846)
-  Construction                   8   686   0.975    in   0.074    28.9% / 37.8%
-  Finance & Real Estate         12   686   0.955    in   0.048    24.1% / 37.1%
-  Manufacturing                 39   686   0.916  over   0.081    16.5% / 24.3%
-  Mining                        16   686   0.873  over   0.122    14.2% / 21.4%
-  Retail                        34   686   1.009    in   0.050     8.4% / 12.9%
-  SIC20                         21   686   0.999    in   0.026    26.7% / 45.3%
-  SIC28                         40   686   1.013    in   0.031     6.7% / 10.6%
-  SIC35                         34   686   0.947    in   0.091      5.3% / 7.3%
-  SIC36                         36   686   1.091 under   0.074      3.4% / 4.1%
-  SIC37                         15   686   1.027    in   0.081    10.0% / 14.8%
+  Construction                   8   686   0.976    in   0.074    28.8% / 37.7%
+  Finance & Real Estate         12   686   0.955    in   0.048    23.9% / 36.9%
+  Manufacturing                 39   686   0.916  over   0.081    16.3% / 24.2%
+  Mining                        16   686   0.874  over   0.121    14.1% / 21.3%
+  Retail                        34   686   1.007    in   0.049     8.4% / 12.9%
+  SIC20                         21   686   0.999    in   0.026    26.5% / 45.0%
+  SIC28                         41   686   1.012    in   0.031     6.7% / 10.5%
+  SIC35                         34   686   0.945  over   0.092      5.3% / 7.3%
+  SIC36                         36   686   1.089 under   0.074      3.4% / 4.1%
+  SIC37                         15   686   1.026    in   0.081    10.0% / 14.8%
-  SIC49                         37   686   0.917  over   0.090     7.7% / 11.9%
-  SIC60                         20   686   0.971    in   0.080      6.5% / 9.5%
-  SIC62                         20   686   0.996    in   0.043     8.6% / 11.5%
-  SIC63                         25   686   0.907  over   0.093    15.1% / 24.2%
-  SIC67                         29   686   0.889  over   0.118    11.8% / 18.3%
-  SIC73                         70   686   0.995    in   0.057      3.1% / 4.9%
-  Services                      20   686   0.910  over   0.087    21.5% / 33.5%
-  Transport & Utilities         32   686   0.895  over   0.113    11.2% / 17.5%
-  Wholesale                     12   686   0.938  over   0.059    39.5% / 60.4%
-  across the 20 with a factor of their own, the one-day bias statistic's correlation with the weighted noise share: -0.275
+  SIC49                         37   686   0.917  over   0.091     7.7% / 11.8%
+  SIC60                         22   686   0.964    in   0.077      6.5% / 9.4%
+  SIC62                         20   686   0.997    in   0.043     8.6% / 11.5%
+  SIC63                         25   686   0.906  over   0.093    15.1% / 24.2%
+  SIC67                         29   686   0.889  over   0.118    11.7% / 18.2%
+  SIC73                         70   686   0.993    in   0.057      3.1% / 4.9%
+  Services                      20   686   0.909  over   0.088    21.4% / 33.4%
+  Transport & Utilities         32   686   0.895  over   0.113    11.1% / 17.4%
+  Wholesale                     12   686   0.939  over   0.059    39.2% / 60.0%
+  across the 20 with a factor of their own, the one-day bias statistic's correlation with the weighted noise share: -0.264
-  Random portfolios, long-only, seed 11: 100 of 30 members, weights drawn once per member -- at one day, bias median 0.914, p5 0.851, p95 0.983; below the band 81%, above it 0%; MRAD median 0.081
+  Random portfolios, long-only, seed 11: 100 of 30 members, weights drawn once per member -- at one day, bias median 0.915, p5 0.853, p95 0.983; below the band 78%, above it 0%; MRAD median 0.080
-  Random portfolios, long-short, seed 11: 100 of 30 members a side, weights drawn once per member -- at one day, bias median 0.997, p5 0.901, p95 1.122; below the band 24%, above it 24%; MRAD median 0.069
+  Random portfolios, long-short, seed 11: 100 of 30 members a side, weights drawn once per member -- at one day, bias median 0.997, p5 0.901, p95 1.122; below the band 24%, above it 23%; MRAD median 0.069
-  Every member singly: 543 securities with 63 forecasts or more -- at one day, bias p5 0.806, p25 0.911, median 0.987, p75 1.056, p95 1.162
-    below its band (over-forecast) 34.3%, above it (under-forecast) 24.7%, against 2.5% each for right forecasts of normal returns; with each member's band widened for its own kurtosis (median 9.0), 20.3% and 7.6%
-    median by capitalisation quintile over the test, largest first: 1.028, 0.992, 0.988, 0.949, 0.963
+  Every member singly: 543 securities with 63 forecasts or more -- at one day, bias p5 0.807, p25 0.912, median 0.987, p75 1.057, p95 1.163
+    below its band (over-forecast) 33.1%, above it (under-forecast) 24.5%, against 2.5% each for right forecasts of normal returns; with each member's band widened for its own kurtosis (median 9.0), 20.1% and 7.6%
+    median by capitalisation quintile over the test, largest first: 1.028, 0.991, 0.989, 0.950, 0.963
-     1       61.70%   686   0.946    in   0.091
-     2       21.22%   686   0.978    in   0.074
-     3       18.36%   686   1.051    in   0.066
-     4       16.94%   686   1.033    in   0.070
-     5       16.20%   686   1.047    in   0.068
-     6       13.02%   686   1.041    in   0.075
-     7       12.02%   686   1.084 under   0.078
-     8       10.95%   686   1.083 under   0.104
-     9        9.69%   686   1.128 under   0.154
-    10        9.12%   686   1.084 under   0.111
-    11        8.74%   686   1.088 under   0.099
+     1       61.87%   686   0.947    in   0.091
+     2       21.28%   686   0.976    in   0.074
+     3       18.35%   686   1.050    in   0.065
+     4       16.95%   686   1.031    in   0.070
+     5       16.21%   686   1.046    in   0.070
+     6       13.01%   686   1.043    in   0.074
+     7       12.02%   686   1.084 under   0.079
+     8       10.94%   686   1.085 under   0.104
+     9        9.67%   686   1.126 under   0.155
+    10        9.11%   686   1.082 under   0.105
+    11        8.72%   686   1.087 under   0.102
-    13        7.90%   686   1.096 under   0.101
-    14        7.44%   686   1.109 under   0.100
-    15        7.15%   686   1.163 under   0.147
-    16        6.85%   686   1.186 under   0.170
-    17        6.47%   686   1.156 under   0.133
-    18        6.10%   686   1.245 under   0.216
-    19        5.88%   686   1.288 under   0.253
-    20        5.28%   686   1.238 under   0.259
-    21        4.68%   686   1.181 under   0.134
-    22        3.97%   686   1.177 under   0.181
-    23        3.51%   686   1.224 under   0.164
-    24        2.50%   686   1.255 under   0.259
-    25        1.80%   686   1.135 under   0.149
+    13        7.88%   686   1.098 under   0.102
+    14        7.43%   686   1.106 under   0.100
+    15        7.16%   686   1.169 under   0.153
+    16        6.84%   686   1.184 under   0.170
+    17        6.47%   686   1.157 under   0.136
+    18        6.09%   686   1.244 under   0.214
+    19        5.87%   686   1.284 under   0.250
+    20        5.26%   686   1.239 under   0.262
+    21        4.69%   686   1.183 under   0.134
+    22        3.96%   686   1.176 under   0.179
+    23        3.52%   686   1.225 under   0.165
+    24        2.49%   686   1.257 under   0.263
+    25        1.80%   686   1.136 under   0.150
-  the index, cap-weighted                    0.934   0.894/0.898   0.856/0.772   0.807/0.791
-  the index, equally weighted                0.893   0.877/0.869   0.819/0.802   0.752/0.696
-  smallest 85, equally weighted              0.902   0.880/0.869   0.807/0.826   0.743/0.651
-  largest 50, equally weighted               0.920   0.887/0.891   0.848/0.787   0.823/0.853
-  minimum variance, long-only                0.990   1.042/1.048   0.972/0.985   0.911/0.886
-  minimum variance, unconstrained            1.033   1.078/1.074   0.988/1.019   0.932/0.966
-  random long-only, median                   0.914   0.895/0.898   0.850/0.836   0.794/0.758
-  random long-short, median                  0.997   0.992/0.995   0.977/0.960   0.970/0.983
-  every member singly, median                0.987   0.963/0.966   0.949/0.944   0.926/0.925
-  industries, mean abs(bias - 1)             0.055   0.051/0.054   0.075/0.103   0.098/0.147
-  eigenportfolios 1-5, mean                  1.011   0.961/0.962   0.941/0.911   0.915/0.861
-  eigenportfolios 21-25, mean                1.195   1.203/1.204   1.187/1.176   1.187/1.224
+  the index, cap-weighted                    0.931   0.892/0.895   0.854/0.770   0.805/0.789
+  the index, equally weighted                0.893   0.877/0.869   0.820/0.802   0.752/0.696
+  smallest 85, equally weighted              0.903   0.882/0.871   0.808/0.827   0.744/0.652
+  largest 50, equally weighted               0.918   0.884/0.889   0.845/0.785   0.821/0.851
+  minimum variance, long-only                0.900   0.960/0.966   0.895/0.913   0.878/0.810
+  minimum variance, unconstrained            0.951   1.002/1.001   0.921/0.950   0.904/0.916
+  random long-only, median                   0.915   0.897/0.900   0.851/0.838   0.794/0.759
+  random long-short, median                  0.997   0.992/0.995   0.977/0.960   0.968/0.983
+  every member singly, median                0.987   0.964/0.966   0.949/0.945   0.927/0.924
+  industries, mean abs(bias - 1)             0.055   0.051/0.054   0.076/0.103   0.098/0.147
+  eigenportfolios 1-5, mean                  1.010   0.959/0.961   0.940/0.910   0.914/0.860
+  eigenportfolios 21-25, mean                1.195   1.205/1.207   1.189/1.175   1.189/1.224
-  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
-  the last day's B_F^2 0.424 over 25 factors, B_S^2 0.701 over 499 members (an effective number of 48)
-  every forecast and outcome written to C:\Data\Development\Python\McTheoryApps\praxis\claude\scratch\midoffice16\bias\D16.tmp.npz
+  as of 2026-11-03: lambda_F 0.9629, lambda_S 1.0000; the factor volatilities' median 16.29% to 15.68%, the specific risk's p10 / median / p90 16.67% / 28.97% / 48.88% to 16.67% / 28.97% / 48.88%
+  the last day's B_F^2 0.422 over 25 factors, B_S^2 0.700 over 499 members (an effective number of 48)
+  every forecast and outcome written to C:\Data\Development\Python\McTheoryApps\praxis\claude\scratch\midoffice19\bias\D19.tmp.npz
```

### smoke/old_country_vs_18.txt -- item 2's smoke: the old scheme against midoffice18's country smoke

```text
country.npz against old_country.npz
  48 arrays in both: 48 equal to the bit
  in the first alone: none
  in the second alone: basis_covariance_blocks, blocks_multipliers, eigen_forecast_blocks, eigen_realised_blocks, eigen_realised_h5_blocks, exposure_long_only_blocks, exposure_unconstrained_blocks, factor_covariance_blocks, forecast_blocks, held_long_only_blocks, held_unconstrained_blocks, member_forecast_blocks, min_variance_weights_long_only, min_variance_weights_long_only_blocks, min_variance_weights_long_only_country, min_variance_weights_unconstrained, min_variance_weights_unconstrained_blocks, min_variance_weights_unconstrained_country, realised_blocks, realised_h5_blocks, regime_dates_blocks_c, regime_dates_blocks_r, regime_effective_blocks_c, regime_effective_blocks_r, regime_factor_squared_blocks_c, regime_factor_squared_blocks_r, regime_members_blocks_c, regime_members_blocks_r, regime_specific_squared_blocks_c, regime_specific_squared_blocks_r, repaired_blocks
```

### smoke/off_blocks_vs_country.txt -- item 2's smoke: the regime adjustment off, the blocks against the country setting

```text
off_country.npz: each _country array against its _blocks twin
  basis_covariance                 equal to the bit
  eigen_forecast                   equal to the bit
  eigen_realised                   equal to the bit
  eigen_realised_h5                equal to the bit
  exposure_long_only               equal to the bit
  exposure_unconstrained           equal to the bit
  factor_covariance                equal to the bit
  forecast                         equal to the bit
  held_long_only                   equal to the bit
  held_unconstrained               equal to the bit
  member_forecast                  equal to the bit
  min_variance_weights_long_only   equal to the bit
  min_variance_weights_unconstrained equal to the bit
  realised                         equal to the bit
  realised_h5                      equal to the bit
  repaired                         equal to the bit
```

### smoke/report_new_country.txt -- item 2's smoke on the new defaults: the guard and the new lines

```text
  members without the market -- member-days from the first estimated period (2023-02-08) with every style exposure and no industry factor: none
  guard: as of 2026-09-22, 2026-11-02, the model's inputs rebuilt from the panel cut at the date, the scheme held, equal the full run's through it to the bit -- returns, exposures, weights, capitalisations, periods -- and so do the forecasts made from them: covariance, exposures, specific risk, factor returns
  the blocks' histories split the basis's: B_F^2 over its 26 factors is (B_C^2 + 25 B_R^2) / 26 on all 29 days, to 3.3e-16 relative
  the blocks by ruling 2 of midoffice19's brief, in only if the optimiser's test passes and its regime criterion is below the model's on both halves, not chosen on the first and checked on the second: the optimiser's test in; the criterion nan against nan on the first half, nan against nan on the second -- out
```

### equal_C19_D19.txt -- item 2: C19's model test against D19, bit for bit

```text
claude/scratch/midoffice19/bias/D19.npz's model test against claude/scratch/midoffice19/bias/C19.npz's model test: 686 forecast dates, 2024-02-08 to 2026-11-02
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

### arrays_C19_D19.txt -- item 2: every array D19 and C19 share

```text
D19.npz against C19.npz
  34 arrays in both: 34 equal to the bit
  in the first alone: none
  in the second alone: basis_covariance_blocks, basis_covariance_country, blocks_multipliers, country_industry_columns, country_weight_dates, country_weights, eigen_forecast_blocks, eigen_forecast_country, eigen_realised_blocks, eigen_realised_country, eigen_realised_h11_blocks, eigen_realised_h11_country, eigen_realised_h21_blocks, eigen_realised_h21_country, eigen_realised_h5_blocks, eigen_realised_h5_country, exposure_long_only_blocks, exposure_long_only_country, exposure_unconstrained_blocks, exposure_unconstrained_country, factor_covariance_blocks, factor_covariance_country, forecast_blocks, forecast_country, held_long_only_blocks, held_long_only_country, held_unconstrained_blocks, held_unconstrained_country, member_forecast_blocks, member_forecast_country, min_variance_weights_long_only_blocks, min_variance_weights_long_only_country, min_variance_weights_unconstrained_blocks, min_variance_weights_unconstrained_country, realised_blocks, realised_country, realised_h11_blocks, realised_h11_country, realised_h21_blocks, realised_h21_country, realised_h5_blocks, realised_h5_country, regime_dates_blocks_c, regime_dates_blocks_r, regime_dates_country, regime_effective_blocks_c, regime_effective_blocks_r, regime_effective_country, regime_factor_squared_blocks_c, regime_factor_squared_blocks_r, regime_factor_squared_country, regime_members_blocks_c, regime_members_blocks_r, regime_members_country, regime_specific_squared_blocks_c, regime_specific_squared_blocks_r, regime_specific_squared_country, repaired_blocks, repaired_country
```

### bias/report_C19.txt -- item 2: the pass's own lines

```text
  the country factor as a covariance basis (USE4): each refit's own factor returns taken to the country basis -- the country the cap-weighted mean of the 20 industries' returns, the weights each period's regression's capitalisation shares, each industry less it, the styles as they are -- the covariance estimated there by the model's estimator, its 26 factors, and mapped back; on two regime multipliers, two settings, each read from the basis's own forecasts, F  21 S   -: 'country' one over the basis's factors (midoffice18), 'blocks' two (midoffice19) -- lambda_C from the country's own squared statistic, lambda_R from the other 25 factors', the basis's covariance scaled by blocks before it is mapped back, the country's variance by lambda_C^2, the others' by lambda_R^2, their cross terms by lambda_C lambda_R; chosen by the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second, the blocks held to ruling 2 of midoffice19's brief: in only if the optimiser's test passes and its regime criterion is below the model's on both halves, not chosen on the first and checked on the second
  686 refits in 8.1 min, 0.71 s a refit with 3 tests; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every portfolio formed that day
  the blocks' histories split the basis's: B_F^2 over its 26 factors is (B_C^2 + 25 B_R^2) / 26 on all 685 days, to 4.4e-16 relative
```

### bias/report_C19.txt -- item 2: the country basis beside the model, both multipliers

```text
  The country factor as a covariance basis at one day (midoffice18): the model (*) and its factor returns in the country basis, the covariance estimated there and mapped back, on its own forecasts' volatility regime adjustment, F  21 S   -; 'country' on one multiplier over the basis's factors, 'blocks' on two (midoffice19), the country's own and the other factors'
  Each setting's regime multiplier, read from its own forecasts' history: the mean of B_F^2, and lambda_F on 21 days -- its mean (min to max) -- by part -- the country basis's B_F over its own factors, the country among them
                           the whole test                   the first half                  the second half
  country      1.149  1.065 (0.875-1.349)       1.140  1.063 (0.875-1.349)       1.158  1.067 (0.900-1.282)
  blocks C     1.077  0.999 (0.545-2.349)       1.503  1.164 (0.753-2.349)       0.653  0.835 (0.545-1.200)
  blocks R     1.152  1.066 (0.873-1.297)       1.125  1.057 (0.873-1.293)       1.179  1.075 (0.907-1.297)
  model *      1.115  1.039 (0.774-1.979)       1.332  1.121 (0.862-1.979)       0.898  0.956 (0.774-1.128)
  lambda_C's path (midoffice19): as of each calendar quarter's last forecast date, beside lambda_R, the country setting's lambda_F and the model's
                    blocks C blocks R  country  model *
  2024-03-28           0.892    0.892    0.892    0.871
  2024-06-28           0.776    0.995    0.988    0.900
  2024-09-30           1.131    0.993    0.999    1.042
  2024-12-31           1.087    1.011    1.014    1.066
  2025-03-31           1.272    1.073    1.082    1.178
  2025-06-30           1.133    0.966    0.973    1.057
  2025-09-30           0.583    0.966    0.954    0.787
  2025-12-31           0.700    0.907    0.900    0.786
  2026-03-31           1.086    1.128    1.126    1.091
  2026-06-30           1.010    1.210    1.203    1.100
  2026-09-30           0.795    1.007    0.999    0.928
  2026-11-02           0.858    1.058    1.051    0.972
  min                  0.545    0.873    0.875    0.774   on 2025-10-09, 2024-04-08, 2024-04-08, 2025-10-09
  max                  2.349    1.297    1.349    1.979   on 2025-04-10, 2026-02-12, 2025-04-10, 2025-04-10
  mean, first half     1.164    1.057    1.063    1.121
  mean, second half    0.835    1.075    1.067    0.956

  First, the minimum-variance portfolios' realised volatility, % a year, whole / first half / second half; a setting under which either realises more over the whole test than the model's, to the hundredth of a point, is out
                       long-only           unconstrained
  country   9.47 / 10.95 /  7.72    9.62 / 11.13 /  7.84   in
  blocks    9.17 / 10.59 /  7.50    9.88 / 11.74 /  7.59   out
  model *   9.62 / 11.05 /  7.97    9.83 / 11.25 /  8.18   in

  Then the regime criterion -- the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median) -- by part:
  country  0.113 / 0.078 / 0.200
  blocks   0.069 / 0.059 / 0.067
  model *  0.091 / 0.068 / 0.138
  chosen on the first half by the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second: model * (0.068; the model's 0.068); on the second half it reads 0.138 against the model's 0.138, the best there blocks
  the blocks by ruling 2 of midoffice19's brief, in only if the optimiser's test passes and its regime criterion is below the model's on both halves, not chosen on the first and checked on the second: the optimiser's test out; the criterion 0.059 against 0.068 on the first half, 0.067 against 0.138 on the second -- out

  The shape: each setting's own eigenportfolios -- the eigenvectors of its covariance -- by rank, 1 the largest eigenvalue, the market-like one; the statistic by part
  the whole test (2024-02-08 to 2026-11-02, 686)
  rank                 country  blocks model *
  1                      0.937   1.010   0.947
  2                      0.925   0.931   0.976
  3                      0.953   0.931   1.050
  4                      0.986   1.000   1.031
  5                      0.992   0.991   1.046
  6                      0.937   0.933   1.043
  7                      0.998   0.993   1.084
  8                      0.990   0.985   1.085
  9                      1.004   0.998   1.126
  10                     0.973   0.970   1.082
  11                     1.005   1.003   1.087
  12                     0.978   0.978   1.098
  13                     1.070   1.069   1.098
  14                     1.019   1.015   1.106
  15                     1.012   1.011   1.169
  16                     1.115   1.111   1.184
  17                     1.047   1.044   1.157
  18                     1.113   1.110   1.244
  19                     1.203   1.197   1.284
  20                     1.127   1.124   1.239
  21                     1.106   1.110   1.183
  22                     1.122   1.131   1.176
  23                     1.129   1.121   1.225
  24                     1.171   1.168   1.257
  25                     1.061   1.060   1.136
  ranks 2-25, mean       1.043   1.041   1.132
  ranks 21-25, mean      1.118   1.118   1.195
  mean abs(bias - 1)     0.065   0.063   0.131
  the first half (2024-02-08 to 2025-06-23, 343)
  rank                 country  blocks model *
  1                      1.095   1.032   1.064
  2                      0.923   0.926   0.910
  3                      0.951   0.919   0.973
  4                      1.029   1.072   1.012
  5                      1.025   1.037   0.956
  6                      0.912   0.915   0.912
  7                      0.980   0.981   0.978
  8                      0.978   0.985   0.994
  9                      0.950   0.944   0.943
  10                     0.965   0.966   0.955
  11                     0.973   0.976   0.942
  12                     0.965   0.970   0.995
  13                     1.133   1.137   1.065
  14                     1.014   1.019   1.002
  15                     1.032   1.030   1.073
  16                     1.098   1.100   1.109
  17                     1.087   1.089   1.053
  18                     1.077   1.080   1.046
  19                     1.115   1.120   1.098
  20                     1.111   1.113   1.087
  21                     1.161   1.175   1.150
  22                     1.135   1.148   1.114
  23                     1.174   1.181   1.162
  24                     1.180   1.182   1.183
  25                     1.028   1.031   1.021
  ranks 2-25, mean       1.041   1.046   1.031
  ranks 21-25, mean      1.136   1.143   1.126
  mean abs(bias - 1)     0.076   0.079   0.067
  the second half (2025-06-24 to 2026-11-02, 343)
  rank                 country  blocks model *
  1                      0.747   0.988   0.814
  2                      0.929   0.938   1.039
  3                      0.956   0.944   1.124
  4                      0.940   0.920   1.049
  5                      0.958   0.943   1.129
  6                      0.961   0.950   1.160
  7                      1.016   1.005   1.182
  8                      1.002   0.985   1.169
  9                      1.056   1.049   1.285
  10                     0.982   0.976   1.196
  11                     1.036   1.027   1.216
  12                     0.991   0.986   1.192
  13                     1.005   0.999   1.131
  14                     1.025   1.013   1.202
  15                     0.993   0.992   1.257
  16                     1.130   1.121   1.256
  17                     1.007   0.998   1.254
  18                     1.149   1.141   1.412
  19                     1.283   1.268   1.447
  20                     1.131   1.124   1.365
  21                     1.046   1.039   1.217
  22                     1.110   1.116   1.237
  23                     1.084   1.059   1.286
  24                     1.163   1.154   1.327
  25                     1.095   1.090   1.241
  ranks 2-25, mean       1.044   1.035   1.224
  ranks 21-25, mean      1.100   1.092   1.262
  mean abs(bias - 1)     0.075   0.064   0.222

  the whole test (2024-02-08 to 2026-11-02, 686), by basis (the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  country    0.926   0.880   0.882   0.910   0.867   9.469   0.908   9.618   0.971   0.065   0.121   0.127   0.089   0.113
  blocks     0.987   0.923   0.896   0.972   0.918   9.174   0.963   9.882   0.974   0.063   0.062   0.059   0.087   0.069
  model *    0.931   0.893   0.903   0.918   0.900   9.624   0.951   9.829   0.987   0.131   0.094   0.093   0.085   0.091

  the first half (2024-02-08 to 2025-06-23, 343), by basis (the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  country    1.097   1.016   0.960   1.054   0.999  10.954   1.049  11.130   0.998   0.076   0.081   0.068   0.083   0.078
  blocks     1.036   0.945   0.896   0.995   0.940  10.595   1.007  11.737   0.970   0.079   0.061   0.034   0.081   0.059
  model *    1.060   0.984   0.923   1.021   0.990  11.046   1.047  11.252   0.978   0.067   0.067   0.062   0.076   0.068

  the second half (2025-06-24 to 2026-11-02, 343), by basis (the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
             index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  country    0.716   0.721   0.796   0.740   0.714   7.719   0.742   7.836   0.935   0.075   0.245   0.267   0.087   0.200
  blocks     0.938   0.903   0.898   0.949   0.896   7.502   0.919   7.594   0.964   0.064   0.053   0.061   0.086   0.067
  model *    0.783   0.794   0.883   0.803   0.801   7.968   0.846   8.179   0.986   0.222   0.160   0.169   0.084   0.138

  The members by capitalisation quintile, the largest first, by part: the median statistic
  the whole test
  country  1.018, 0.983, 0.973, 0.937, 0.955
  blocks   1.026, 0.978, 0.967, 0.938, 0.954
  model *  1.028, 0.991, 0.989, 0.950, 0.963
  the first half
  country  1.045, 1.015, 1.004, 0.968, 0.969
  blocks   1.014, 0.979, 0.974, 0.931, 0.948
  model *  1.027, 0.983, 0.986, 0.941, 0.956
  the second half
  country  0.978, 0.944, 0.934, 0.898, 0.915
  blocks   1.018, 0.968, 0.968, 0.930, 0.938
  model *  1.025, 0.992, 0.985, 0.944, 0.961

  The country factor -- the industries' factor returns weighted by the outcome period's capitalisation shares, each setting's forecast of it from its covariance mapped back -- by part: its forecast volatility % a year at the median, and its statistic
  the whole test     country 15.68 0.933   blocks 12.96 1.005   model * 14.43 0.940
  the first half     country 13.65 1.104   blocks 13.68 1.046   model * 13.73 1.067
  the second half    country 16.88 0.725   blocks 12.52 0.964   model * 15.32 0.795

  Day to day: the median absolute change in log forecast volatility from one forecast date to the next (its 95th percentile), the cap-weighted index's and every member's (the median of each one's median)
             index     p95  members     p95
  country    0.79%   2.68%    1.45%   6.21%
  blocks     1.62%   5.20%    1.40%   6.10%
  model *    1.00%   3.60%    1.47%   6.39%
```

### judge_C19.txt -- item 2: C19 judged by the rule against D19, the blocks by ruling 2's bar

```text
The reference: claude/scratch/midoffice19/bias/D19.npz's model test. The run: claude/scratch/midoffice19/bias/C19.npz, its tests the model, _blocks, _country. 686 forecast dates, 2024-02-08 to 2026-11-02, the halves 343 and 343.
The rule (COVARIANCE_RULE): the minimum-variance portfolios' realised volatility first -- a setting under which either, long-only or unconstrained, realises more over the whole test than the model's own setting, each in % a year to the hundredth of a point, is out -- then the regime criterion, chosen on the first half among the settings left and checked on the second; the criterion: the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median)

1. The minimum-variance portfolios' realised volatility, % a year, whole / first / second, against the reference's; out if either rounds above it over the whole test
  reference   9.62 / 11.05 /  7.97    9.83 / 11.25 /  8.18
  model       9.62 / 11.05 /  7.97    9.83 / 11.25 /  8.18   in
  _blocks     9.17 / 10.59 /  7.50    9.88 / 11.74 /  7.59   out
  _country    9.47 / 10.95 /  7.72    9.62 / 11.13 /  7.84   in

2. The regime criterion, whole / first / second, and its three MRADs (the broad portfolios', the optimiser's, the members' median)
  reference  criterion 0.091 / 0.068 / 0.138   broad 0.094 / 0.067 / 0.160   optimiser 0.093 / 0.062 / 0.169   members 0.085 / 0.076 / 0.084
  model      criterion 0.091 / 0.068 / 0.138   broad 0.094 / 0.067 / 0.160   optimiser 0.093 / 0.062 / 0.169   members 0.085 / 0.076 / 0.084
  _blocks    criterion 0.069 / 0.059 / 0.067   broad 0.062 / 0.061 / 0.053   optimiser 0.059 / 0.034 / 0.061   members 0.087 / 0.081 / 0.086
  _country   criterion 0.113 / 0.078 / 0.200   broad 0.121 / 0.081 / 0.245   optimiser 0.127 / 0.068 / 0.267   members 0.089 / 0.083 / 0.087
  the rule: in -- model, _country; the lowest on the first half model (0.068) against the reference's 0.068: not below it, so the reference stays; on the second half it reads 0.138 against 0.138
  the blocks, by ruling 2 of midoffice19's brief (in only if the optimiser's test passes and its regime criterion is below the model's on both halves, not chosen on the first and checked on the second): the optimiser's test out; the criterion 0.059 against 0.068 on the first half (below), 0.067 against 0.138 on the second half (below) -- OUT

3. Beside it: each test's own eigenportfolios, rank 1 and ranks 21-25 (mean), the statistic whole / first / second
  reference  rank 1 0.947 / 1.064 / 0.814   ranks 21-25 1.195 / 1.126 / 1.262   mean abs(bias - 1) 0.131 / 0.067 / 0.222
  model      rank 1 0.947 / 1.064 / 0.814   ranks 21-25 1.195 / 1.126 / 1.262   mean abs(bias - 1) 0.131 / 0.067 / 0.222
  _blocks    rank 1 1.010 / 1.032 / 0.988   ranks 21-25 1.118 / 1.143 / 1.092   mean abs(bias - 1) 0.063 / 0.079 / 0.064
  _country   rank 1 0.937 / 1.095 / 0.747   ranks 21-25 1.118 / 1.136 / 1.100   mean abs(bias - 1) 0.065 / 0.076 / 0.075

3b. Beside it: the index, cap-weighted, and the members' median, the statistic whole / first / second
  reference  the index 0.931 / 1.060 / 0.783   the members' median 0.987 / 0.978 / 0.986
  model      the index 0.931 / 1.060 / 0.783   the members' median 0.987 / 0.978 / 0.986
  _blocks    the index 0.987 / 1.036 / 0.938   the members' median 0.974 / 0.970 / 0.964
  _country   the index 0.926 / 1.097 / 0.716   the members' median 0.971 / 0.998 / 0.935

4. The optimiser along its own covariance's eigen-directions: each group's share of the forecast variance (the mean over dates), and the statistic of its realised part, whole / first / second; the factor returns recovered from the run's model test
  minimum variance, long-only
    model    rank 1 47.1% 0.95/1.06/0.81; ranks 2-5 15.8% 1.04/0.92/1.14; ranks 6-15  9.4% 1.12/1.06/1.18; ranks 16-25 21.6% 1.18/1.13/1.22; specific  6.1% 0.86/0.89/0.83
    _blocks  rank 1 46.0% 1.01/1.03/0.99; ranks 2-5 15.9% 0.96/0.95/0.96; ranks 6-15  8.6% 1.00/1.02/0.96; ranks 16-25 22.6% 1.11/1.19/1.03; specific  6.9% 0.84/0.93/0.74
    _country rank 1 47.6% 0.94/1.10/0.75; ranks 2-5 16.4% 0.95/0.93/0.97; ranks 6-15  8.4% 1.01/1.03/0.99; ranks 16-25 22.0% 1.11/1.16/1.05; specific  5.6% 0.89/0.92/0.85
  minimum variance, unconstrained
    model    rank 1 39.9% 0.95/1.06/0.81; ranks 2-5 15.8% 1.04/0.92/1.15; ranks 6-15 11.5% 1.12/1.05/1.18; ranks 16-25 26.3% 1.18/1.14/1.21; specific  6.6% 0.86/0.84/0.88
    _blocks  rank 1 39.2% 1.01/1.03/0.99; ranks 2-5 16.3% 0.96/0.96/0.96; ranks 6-15 10.1% 0.99/1.01/0.95; ranks 16-25 27.6% 1.12/1.20/1.04; specific  6.8% 0.88/0.86/0.90
    _country rank 1 41.0% 0.94/1.10/0.75; ranks 2-5 16.3% 0.96/0.94/0.98; ranks 6-15 10.0% 1.00/1.02/0.97; ranks 16-25 27.2% 1.12/1.18/1.06; specific  5.5% 0.88/0.85/0.90
```

### flaky_before.txt -- item 3: the test alone, fifty runs, before the fix

```text
tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query, alone, 50 runs, each its own process: python -m pytest tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query -q -rf -p no:cacheprovider
   1 21:49:54Z    3.2 GB available    4.6 s  pass
   2 21:49:58Z    3.2 GB available    3.6 s  pass
   3 21:50:02Z    3.2 GB available    3.9 s  pass
   4 21:50:06Z    3.2 GB available    3.8 s  pass
   5 21:50:10Z    3.2 GB available    3.8 s  pass
   6 21:50:14Z    3.2 GB available    3.8 s  pass
   7 21:50:17Z    3.2 GB available    3.6 s  pass
   8 21:50:21Z    3.2 GB available    3.6 s  pass
   9 21:50:25Z    3.2 GB available    3.6 s  pass
  10 21:50:28Z    3.2 GB available    3.6 s  pass
  11 21:50:32Z    3.2 GB available    3.8 s  pass
  12 21:50:35Z    3.2 GB available    3.6 s  pass
  13 21:50:39Z    3.3 GB available    3.5 s  pass
  14 21:50:42Z    3.3 GB available    3.6 s  pass
  15 21:50:46Z    3.2 GB available    3.8 s  pass
  16 21:50:50Z    3.2 GB available    3.7 s  pass
  17 21:50:54Z    3.2 GB available    3.6 s  pass
  18 21:50:57Z    3.2 GB available    3.5 s  pass
  19 21:51:01Z    3.2 GB available    3.6 s  pass
  20 21:51:04Z    3.2 GB available    3.8 s  pass
  21 21:51:08Z    3.2 GB available    3.6 s  pass
  22 21:51:12Z    3.3 GB available    3.6 s  pass
  23 21:51:15Z    3.3 GB available    3.6 s  pass
  24 21:51:19Z    3.3 GB available    3.5 s  pass
  25 21:51:22Z    3.3 GB available    3.5 s  pass
  26 21:51:26Z    3.3 GB available    3.6 s  pass
  27 21:51:29Z    3.3 GB available    3.5 s  pass
  28 21:51:33Z    3.2 GB available    3.8 s  pass
  29 21:51:37Z    3.6 GB available    3.5 s  pass
  30 21:51:40Z    3.7 GB available    3.6 s  pass
  31 21:51:44Z    3.7 GB available    3.7 s  pass
  32 21:51:47Z    3.7 GB available    3.7 s  pass
  33 21:51:51Z    3.7 GB available    3.8 s  pass
  34 21:51:55Z    3.7 GB available    3.8 s  pass
  35 21:51:59Z    3.7 GB available    3.8 s  pass
  36 21:52:03Z    3.7 GB available    3.7 s  pass
  37 21:52:06Z    3.7 GB available    3.6 s  pass
  38 21:52:10Z    3.7 GB available    3.6 s  pass
  39 21:52:14Z    3.7 GB available    3.7 s  pass
  40 21:52:17Z    3.7 GB available    3.6 s  pass
  41 21:52:21Z    3.7 GB available    3.8 s  pass
  42 21:52:25Z    3.6 GB available    3.7 s  pass
  43 21:52:28Z    3.7 GB available    3.5 s  pass
  44 21:52:32Z    3.7 GB available    3.6 s  pass
  45 21:52:35Z    3.7 GB available    3.5 s  pass
  46 21:52:39Z    3.7 GB available    3.4 s  pass
  47 21:52:42Z    3.7 GB available    3.6 s  pass
  48 21:52:46Z    3.7 GB available    3.9 s  FAIL  FAILED tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query
  49 21:52:50Z    3.7 GB available    3.6 s  pass
  50 21:52:53Z    3.6 GB available    3.9 s  FAIL  FAILED tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query
2 failed, 48 passed
```

### flaky_probe_before.txt -- item 3: its computation repeated in-process on seed 42, before

```text
   1  5% at the midpoint -5.326769, at (10, 200) -5.527526, at (20, 400) -5.566922; |mid - avg| 0.2205 against 3 x spread 0.1182: FAIL
   2  5% at the midpoint -5.143145, at (10, 200) -5.076720, at (20, 400) -5.487552; |mid - avg| 0.1390 against 3 x spread 1.2325: pass
   3  5% at the midpoint -5.104455, at (10, 200) -4.973874, at (20, 400) -5.460611; |mid - avg| 0.1128 against 3 x spread 1.4602: pass
   4  5% at the midpoint -5.345784, at (10, 200) -5.172742, at (20, 400) -5.248994; |mid - avg| 0.1349 against 3 x spread 0.2288: pass
   5  5% at the midpoint -5.113310, at (10, 200) -5.121763, at (20, 400) -5.487552; |mid - avg| 0.1913 against 3 x spread 1.0974: pass
   6  5% at the midpoint -5.366852, at (10, 200) -5.417490, at (20, 400) -5.435457; |mid - avg| 0.0596 against 3 x spread 0.0539: FAIL
   7  5% at the midpoint -5.350661, at (10, 200) -5.178972, at (20, 400) -5.487552; |mid - avg| 0.0174 against 3 x spread 0.9257: pass
   8  5% at the midpoint -5.488406, at (10, 200) -5.165481, at (20, 400) -5.965311; |mid - avg| 0.0770 against 3 x spread 2.3995: pass
   9  5% at the midpoint -5.136609, at (10, 200) -5.164702, at (20, 400) -5.487552; |mid - avg| 0.1895 against 3 x spread 0.9685: pass
  10  5% at the midpoint -5.195006, at (10, 200) -4.819901, at (20, 400) -5.639363; |mid - avg| 0.0346 against 3 x spread 2.4584: pass
  11  5% at the midpoint -5.073315, at (10, 200) -5.261067, at (20, 400) -5.246518; |mid - avg| 0.1805 against 3 x spread 0.0436: FAIL
  12  5% at the midpoint -5.065946, at (10, 200) -4.991152, at (20, 400) -5.313251; |mid - avg| 0.0863 against 3 x spread 0.9663: pass
  13  5% at the midpoint -5.188940, at (10, 200) -5.182351, at (20, 400) -6.219001; |mid - avg| 0.5117 against 3 x spread 3.1099: pass
  14  5% at the midpoint -5.109326, at (10, 200) -5.058172, at (20, 400) -5.487552; |mid - avg| 0.1635 against 3 x spread 1.2881: pass
  15  5% at the midpoint -5.178651, at (10, 200) -5.186933, at (20, 400) -5.487552; |mid - avg| 0.1586 against 3 x spread 0.9019: pass
  16  5% at the midpoint -5.093420, at (10, 200) -4.974011, at (20, 400) -5.487552; |mid - avg| 0.1374 against 3 x spread 1.5406: pass
  17  5% at the midpoint -5.403741, at (10, 200) -5.571843, at (20, 400) -5.351644; |mid - avg| 0.0580 against 3 x spread 0.6606: pass
  18  5% at the midpoint -5.223081, at (10, 200) -5.391929, at (20, 400) -5.515963; |mid - avg| 0.2309 against 3 x spread 0.3721: pass
  19  5% at the midpoint -5.194747, at (10, 200) -4.997252, at (20, 400) -5.519640; |mid - avg| 0.0637 against 3 x spread 1.5672: pass
  20  5% at the midpoint -5.225869, at (10, 200) -5.104785, at (20, 400) -5.598535; |mid - avg| 0.1258 against 3 x spread 1.4813: pass
20 repetitions, seed 42 every time: 20 distinct sets of values; 3 failed the test's assertion
```

### flaky_probe_after.txt -- item 3: the same, every batch seeded

```text
   1  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
   2  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
   3  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
   4  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
   5  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
   6  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
   7  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
   8  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
   9  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  10  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  11  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  12  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  13  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  14  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  15  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  16  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  17  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  18  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  19  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
  20  5% at the midpoint -5.199685, at (10, 200) -5.151620, at (20, 400) -5.487552; |mid - avg| 0.1199 against 3 x spread 1.0078: pass
20 repetitions, seed 42 every time: 1 distinct sets of values; 0 failed the test's assertion
```

### flaky_after.txt -- item 3: the test alone, fifty runs, after the fix

```text
tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query, alone, 50 runs, each its own process: python -m pytest tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query -q -rf -p no:cacheprovider
   1 21:54:57Z    3.6 GB available    3.5 s  pass
   2 21:55:01Z    3.6 GB available    3.6 s  pass
   3 21:55:04Z    3.6 GB available    3.6 s  pass
   4 21:55:08Z    3.6 GB available    3.5 s  pass
   5 21:55:11Z    3.6 GB available    3.7 s  pass
   6 21:55:15Z    3.6 GB available    3.8 s  pass
   7 21:55:19Z    3.6 GB available    3.6 s  pass
   8 21:55:23Z    3.6 GB available    3.6 s  pass
   9 21:55:26Z    3.6 GB available    3.6 s  pass
  10 21:55:30Z    3.6 GB available    3.6 s  pass
  11 21:55:33Z    3.6 GB available    3.9 s  pass
  12 21:55:37Z    3.6 GB available    3.6 s  pass
  13 21:55:41Z    3.6 GB available    3.5 s  pass
  14 21:55:44Z    3.6 GB available    3.6 s  pass
  15 21:55:48Z    3.6 GB available    3.5 s  pass
  16 21:55:51Z    3.6 GB available    3.7 s  pass
  17 21:55:55Z    3.6 GB available    3.8 s  pass
  18 21:55:59Z    3.6 GB available    3.6 s  pass
  19 21:56:02Z    3.6 GB available    3.7 s  pass
  20 21:56:06Z    3.6 GB available    3.8 s  pass
  21 21:56:10Z    3.6 GB available    3.5 s  pass
  22 21:56:13Z    3.6 GB available    3.7 s  pass
  23 21:56:17Z    3.6 GB available    3.9 s  pass
  24 21:56:21Z    3.6 GB available    3.9 s  pass
  25 21:56:25Z    3.4 GB available    3.8 s  pass
  26 21:56:29Z    3.5 GB available    3.7 s  pass
  27 21:56:32Z    3.4 GB available    3.8 s  pass
  28 21:56:36Z    3.4 GB available    4.0 s  pass
  29 21:56:40Z    4.7 GB available    3.7 s  pass
  30 21:56:44Z    4.8 GB available    3.8 s  pass
  31 21:56:48Z    4.7 GB available    3.7 s  pass
  32 21:56:51Z    4.7 GB available    3.6 s  pass
  33 21:56:55Z    4.6 GB available    3.8 s  pass
  34 21:56:59Z    4.7 GB available    3.7 s  pass
  35 21:57:02Z    4.7 GB available    3.7 s  pass
  36 21:57:06Z    4.7 GB available    3.9 s  pass
  37 21:57:10Z    4.7 GB available    4.1 s  pass
  38 21:57:14Z    4.2 GB available    4.3 s  pass
  39 21:57:18Z    4.9 GB available    3.7 s  pass
  40 21:57:22Z    4.9 GB available    3.6 s  pass
  41 21:57:26Z    4.9 GB available    3.8 s  pass
  42 21:57:29Z    4.7 GB available    3.8 s  pass
  43 21:57:33Z    4.7 GB available    3.7 s  pass
  44 21:57:37Z    4.7 GB available    3.7 s  pass
  45 21:57:41Z    4.7 GB available    3.7 s  pass
  46 21:57:44Z    4.7 GB available    3.8 s  pass
  47 21:57:48Z    4.7 GB available    3.7 s  pass
  48 21:57:52Z    4.7 GB available    3.8 s  pass
  49 21:57:56Z    4.6 GB available    3.8 s  pass
  50 21:57:59Z    4.7 GB available    3.7 s  pass
0 failed, 50 passed
```

### bias/runner_log.txt -- the runs

```text
2026-11-13 17:13:27  runner started, pid 37092, for D19; 5.1 GB available
2026-11-13 17:13:27  D19: 5.1 GB available
2026-11-13 17:13:27  D19: attempt 1: --bias-guard 5 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-13 17:25:11  D19: done in 11.7 min
2026-11-13 17:25:11  runner finished; every run done
2026-11-13 17:25:27  runner started, pid 26308, for C19; 4.6 GB available
2026-11-13 17:25:27  C19: 4.6 GB available
2026-11-13 17:25:27  C19: attempt 1: --bias-guard 0 --bias-country --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-13 17:35:11  C19: done in 9.7 min
2026-11-13 17:35:11  runner finished; every run done
```

### final_checks.txt -- the final code: the build, the defaults and the old scheme re-captured; the suite (-rf), ruff, hygiene (stopped at this retro, then unwritten; rerun at the end)

```text
build _released: exit 0
20
FD19: status 0, mean R2 0.294466, 938 periods, 25 factors, 53s, the estimator 1.0s
claude/scratch/midoffice19/model/M19 against claude/scratch/midoffice19/final/FD19: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 168 and 168 lines; 2 differing line(s)
    -  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.3 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.3 s)
    +  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.2 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.1 s)
FD19old: status 0, mean R2 0.293825, 938 periods, 25 factors, 54s, the estimator 1.0s
claude/scratch/midoffice18/model/M18 against claude/scratch/midoffice19/final/FD19old: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 163 and 170 lines; 7 differing line(s)
    +  --industry-overrides none: the records' codes alone, and a group below the hard floor pooled into Other Industries (midoffice18's scheme)
    +  members without the market -- member-days from the first estimated period (2023-02-08) with every style exposure and no industry factor: 996, CTVA 917, FRCB 57, SBNY 22
    +
    +The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
    +  as of 2026-11-03: lambda_F 0.9640, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.99% / 48.89% to 16.56% / 28.99% / 48.89%
    +  the last day's B_F^2 0.424 over 25 factors, the specific side not read
    +  the factor side read from the full fit, without refitting: the model's estimator on its factor returns through each day before, 686 days, 0.2 s; guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model refitted as of the date has the full fit's factor returns through it and its factor volatilities to the bit (3.2 s)
    result = np.where(x == med, 0.0, np.sign(x - med) * np.inf)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2384 passed, 1 skipped, 19 warnings in 84.24s (0:01:24) ===========
All checks passed!
ruff: clean over the changed mid-office Python files
No fixes available (1 hidden fix can be enabled with the `--unsafe-fixes` option).
scripts/run_sp500_model.py: CRLF (3365 lines), parses, ASCII
src/praxis/midoffice/refdata/industries.py: CRLF (643 lines), parses, ASCII
src/praxis/midoffice/model/country.py: CRLF (158 lines), parses, ASCII
src/praxis/midoffice/model/bias.py: CRLF (716 lines), parses, ASCII
tests/midoffice/test_industries.py: LF (536 lines), parses, ASCII
tests/midoffice/test_country.py: LF (162 lines), parses, ASCII
tests/midoffice/test_bias.py: LF (498 lines), parses, ASCII
src/praxis/stats/surface.py: CRLF (2314 lines), parses, 7584 non-ASCII bytes (first at line 6)
config/sp500_industry_overrides.json: LF (33 lines), -, ASCII
docs/FACTOR_MODEL.md: CRLF (1975 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (973 lines), -, ASCII
claude/handoffs/BRIEF_midoffice19_every_member_on_the_market.md: LF (79 lines), -, ASCII
Traceback (most recent call last):
  File "C:\Data\Development\Python\McTheoryApps\praxis\claude\scratch\midoffice13\hygiene.py", line 10, in <module>
    data = Path(name).read_bytes()
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\jmcphail\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1019, in read_bytes
    with self.open(mode='rb') as f:
         ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\jmcphail\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'claude\\retros\\RETRO_midoffice19_every_member_on_the_market.md'
exit 1
fatal: pathspec 'claude/retros/RETRO_midoffice19_every_member_on_the_market.md' did not match any files
0
0
```

## git add (always last -- Rule 43)

This piece -- every member on the market (the rulings file, the scheme's joins, the run's switch and standing check), the two-block setting and the stored holdings, the seed fix, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add scripts/run_sp500_model.py src/praxis/midoffice/refdata/industries.py src/praxis/midoffice/model/country.py src/praxis/midoffice/model/bias.py src/praxis/stats/surface.py tests/midoffice/test_industries.py tests/midoffice/test_country.py tests/midoffice/test_bias.py config/sp500_industry_overrides.json docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice19_every_member_on_the_market.md claude/retros/RETRO_midoffice19_every_member_on_the_market.md
```
