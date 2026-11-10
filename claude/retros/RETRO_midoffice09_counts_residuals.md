# Retro: Mid-office 09 -- the count rulings, the defaults, and residuals for every covered name

**Brief:** `claude/handoffs/BRIEF_midoffice09_counts_residuals.md` -- written in Chat and pasted
into the Code session, transcribed verbatim
**Date:** 2026-11-10
**Duration:** the brief arrived shortly before 17:57Z (12:57 EST). The count rulings, their
panels and model runs to about 13:05; the defaults to about 13:10; residuals for every covered
name, its tests and runs to about 13:20; the docs, the state file and this retro to about 13:35.
**Status:** COMPLETE.
- **Item 1:** the three count rulings are built and measured, each against the one before, and
  the model is run against midoffice08's run D.
- **Item 2:** the run at its new defaults equals item 1's run exactly.
- **Item 3:** each security's own history is now every specific return it has; the estimation
  is unchanged.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Item 1, the count rulings.** The panel is rebuilt as
  `sp500_securities_2022-01-03_2026-11-03_repaired.npz`, a new file; midoffice08's is
  untouched. Member-days whose count changes, each ruling against the one before:

| Ruling | Member-days | What |
|---|---|---|
| 1, confirmed first counts | 117 | GE Vernova 59 (its first count GE's, 3.97x), Amentum 58 (0.63x): the first record the next sample confirms is taken, the 3 and 4 days before it without a count; WBD's 0.92 stands |
| 2, isolated errors repaired | 378 | 7 samples removed: DGX, EG, NWS, PCG, SBUX, XYZ (63 member-days each) and RDDT (none, not a member then) |
| 3, the overrides | 38 | HON carries 635,675,701 over 17 member-days, 2 contaminated samples not read; SKYD has no count over 21 |

- **Item 1, the model.** At run D's configuration (total returns, traded volumes):
  - **R-squared:** 0.293189 to 0.293204.
  - **Factors:** every mean moves by 0.10% or less, every vol by 0.09% or less.
  - **Members per period:** fall by at most one, SKYD's days without a count.
  - **Fallbacks:** the same 11.
  - **Specific risk:** the largest moves are the corrected names' -- SBUX -0.30 points, DGX
    -0.11, HON -0.09.
  - Each ruling was also run on its own, to attribute the change (R1, R2, R3).
- **One finding against the ruling's own words.** The isolated-error rule, as worded, removes
  **PCG's sample of 2026-01-09**: its two samples either side agree at 2.675 to 2.680 billion.
  The ruling expected PCG left whole, and both of Polygon's counts that day agree with the removed
  2,197,910,425. It is applied as worded, and is a decision below. CEG's alternation, and PCG's
  in 2022, are left, as ruled.
- **The weighted counts, on the removed samples.** They confirm the repairs of DGX, EG and SBUX:
  their records' weighted counts sit within 0.6% of the count carried through. They say nothing
  for NWS, RDDT and XYZ, which have several share classes.
- **Item 2, the defaults.** `run_sp500_model.py --industries --membership` now runs the
  repaired panel on total returns. **It equals item 1's run exactly:** its report line for line,
  its model array for array, 23 fields of 23. The old panel stops with a message unless given
  `--returns price`. With `--specific-history regression` as well, it reproduces midoffice08's
  run A exactly.
- **Item 3, residuals for every covered name** (`specific_history="covered"`, the run's default).
  - **What stays the regression's:** factor returns, R-squared, the V fit, kappa and the blend's
    model error are unchanged to the bit.
  - **The histories:** 55,830 specific returns are measured outside the regressions.
  - **The fallback list:** 11 to none, every one now blended with a history of its own.
  - **The 21 joiners in their first year as members:** the characteristics' weight falls from a
    median 16.4% to 2.9%.
  - **Leavers:** 52 still forecast, their histories running on after they left.
  - **Specific risk:** p10 and median unchanged (15.58%, 24.79%), p75 31.69% to 31.95%, p90
    40.69% to 40.56%. 108 of 551 forecasts move.
  - **The largest move:** Signature Bank's, 58.25% to 395.72%. After its failure its shares
    traded at fractions of a cent, a 412% annual volatility.
- **Docs:** `docs/FACTOR_MODEL.md` was brought in line, though the brief listed no docs: the share
  counts' three rulings, the standing checks after them, the run's defaults, and a subsection on
  covered histories.
- **Tests:** 2,295 passed, 1 skipped (midoffice: 844). 5 are new.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice09_counts_residuals.md`, verbatim. In outline:

- **"committed -- midoffice08."** Checked: cfe1245 in `git log`.
- **Rulings on midoffice08's decisions:**
  1. Transitional first counts, **(b)**: a first count stands only if the span's next sample
     confirms it within 10%; otherwise the search goes on to the first record the next sample
     confirms, the days before it without a count.
  2. **Single bad samples:** repair the isolated ones. A sample whose two samples before and two
     after agree with each other within 10%, and from which it differs beyond 10% with no split
     to explain it, is not carried, the count before it carrying through. Alternating samples,
     "as CEG's and PCG's do", are not repaired and stay flagged. Report every removal.
     Hindsight in cleaning the data, not in what the model knows.
  3. **HON and SKYD:** an overrides file in the repo, each entry a security, a range, what the
     build does there, and the evidence:
     - HON from 2026-10-12, carry 635,675,701;
     - SKYD from 2026-10-06, no count;

     the build listing every entry it used.
  4. **Total returns by default**, price still runnable.
  5. **The panel on securities by default**, the old one runnable for comparison.
- **midoffice09, in order, each change measured against the one before:**
  1. The count rulings: rebuild the panel, report the member-days each one changes, and the
     model against midoffice08's run D.
  2. The defaults: the run at its new defaults should equal item 1's run exactly; say whether it
     does.
  3. **Residuals for every covered name.**
     - Factor returns stay the estimation universe's, and the specific-risk model's estimation
       (V fit, kappa, the blend's model error) stays on its residuals.
     - Every security with a return and a complete characteristic row gets each period's
       specific return, and each security's own history is built from all of them.
     - Report:
       - the fallback list;
       - the blend weights of the joiners in their first year as members;
       - the specific-risk percentiles;
       - the names whose forecasts move most.
- **"After this: the model-wide noise-share question, then out-of-sample bias statistics."**

---

## Files modified

| File | Change | Lines |
|---|---|---|
| `src/praxis/midoffice/refdata/securities.py` | ruling 1: `search_first_count(confirm=...)`, `FirstCount.unconfirmed` and `next_count`; ruling 2: `Repair`, `isolated_errors`; ruling 3: `Override`, `read_overrides`, `build_panel(overrides=...)`, `Panel.overridden` | 63-65, 131-195, 199-339, 423-425, 454-462, 577-605, 628-630 (hunks) |
| `src/praxis/midoffice/model/crosssectional.py` | `specific_history` ("regression", the default, unchanged; "covered"): each estimated period's specific return for every covered security, the histories built from them, outside securities with a history blended; the notes' `specific_history`, `outside_residuals`, `blended_outside`, `history_lengths`; the fallback and blended-outside qualifications | 175-186, 380-401, 687, 747-763, 829-835, 899-904, 986-988, 1033-1047, 1069-1080, 1137-1143 (hunks) |
| `scripts/build_sp500_securities.py` | `--first-counts`, `--repair`, `--overrides`, `--suffix` (`_repaired`), `--overwrite`; the repairs and the overrides applied and listed; the member-days without a count by why; the report's `rules`, `repairs`, `overrides`, `unconfirmed`; the docstring | 20-37, 104-120, 182-186, 212-214, 238-262, 327-346, 478-482, 515-526 (hunks) |
| `scripts/run_sp500_model.py` | `DEFAULT_PANEL`, the panel argument optional; `--returns total` by default, the old panel refusing it; `--specific-history covered` by default; the covered line and the blend weight's typical history; the docstring | 3-10, 74-77, 125-127, 492-498, 553-559, 884, 1197-1207 |
| `config/sp500_share_count_overrides.json` | new: HON's and SKYD's rulings with their evidence | new, 26 lines |
| `tests/midoffice/test_securities.py` | 4 new tests; `small_panel` takes overrides | 8, 94, 118, 224-331 |
| `tests/midoffice/test_crosssectional.py` | 1 new test | 1109-1154 |
| `docs/FACTOR_MODEL.md` | the share counts' rulings, the standing checks, the run's defaults; a subsection on covered histories | 637-657, 952-974, 980-987, 1007-1019 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice09_counts_residuals.md` | new: the brief, verbatim | new, 81 lines |
| `claude/retros/RETRO_midoffice09_counts_residuals.md` | new: this retro | new |

Written outside the tracked files, all gitignored and all new; nothing in `data/` was
overwritten or deleted:

- `data/external/polygon/sp500_securities_2022-01-03_2026-11-03_repaired.npz` and `.json`: the
  panel with the three rulings, and its report. midoffice08's unsuffixed panel is untouched.
- In `data/external/polygon/cache/`: 9 answers, the records ruling 1's second searches asked for
  (15,354 to 15,363).
- `claude/scratch/midoffice09/`: the scratch, including the panels of each step (`p0` to `p3`).

---

## Item 1: the count rulings

### Midoffice08's rules first: its panel exactly

Built with all three rulings off -- `--first-counts found --repair none --overrides none
--suffix ""`, offline into the scratch -- the panel's 20 arrays are identical to midoffice08's.
The report differs only in its new keys (`build_p0.txt`). Each ruling is then added in turn: P1,
P2, P3. P3 is the default build, written to `data/` as `_repaired`; its arrays are identical to
the scratch P3's.

### Ruling 1: a first count must be confirmed

`search_first_count` runs the first bisection as before. If the first record with a count is not
within 10% of the span's next sample -- the bracket's end, the first plan sample with a count -- a
second bisection between the two finds the first record the next sample confirms. That assumes a
record once confirmed stays confirmed. The days before that record have no count.
`confirm=False` (`--first-counts found`) is midoffice08's rule.

| Span | First record with a count | Against the next sample | First confirmed | Lookups |
|---|---|---|---|---|
| GEV | 2024-04-03, 1,088,330,000 | 3.971 of 274,090,000 | 2024-04-08, 272,080,000 | 11 |
| AMTM | 2024-10-01, 153,280,000 | 0.630 of 243,300,000 | 2024-10-07, 243,300,000 | 11 |
| WBD | 2022-04-12, 2,233,730,000 | 0.920 of 2,426,840,000 | stands | 5 |
| the other nine | -- | 0.986 to 1.002 | stand | 5 or 6 |

78 lookups in all, 9 over the network: GEV's and AMTM's second searches (`build_p1.txt`).

**Member-days changed, against midoffice08's panel:** 117, and nothing else in the panel.

- **AMTM:** 58 member-days, 2024-10-01 to 2024-12-20. 4 lose their count; 54 move from
  153,280,000 to 243,300,000.
- **GEV:** 59 member-days, 2024-04-03 to 2024-06-26. 3 lose their count; 56 move from
  1,088,330,000 to 272,080,000.

### Ruling 2: isolated errors are not carried

`securities.isolated_errors` reads each security's samples in date order: the plan's samples and
the confirmed first counts, under all its keys, the stale ones excluded. For each sample with
two samples before it and two after:

- **the four neighbours must agree with each other within 10%** -- their largest over their
  smallest within 1.10;
- **it must differ from each of them beyond 10%;**
- **no split the gate applies may fall between the outer two.**

It is one pass over the samples as they are, so two wrong samples in a row, or samples
alternating, leave the neighbours disagreeing and nothing is removed. **Seven are removed**
(`build_p2.txt`):

| Security | Sample | Count | Its neighbours | Of their median | Member-days |
|---|---|---|---|---|---|
| DGX | 2025-04-09 | 11,610,000 | 111,320,000; 111,620,000; 111,635,472; 111,823,441 | 0.104 | 63 |
| EG | 2025-04-09 | 47,790,000 | 43,270,000; 42,980,000; 42,534,728; 41,943,291 | 1.118 | 63 |
| NWS | 2024-07-09 | 162,170,000 | 191,390,000; 191,100,000; 190,080,000; 189,770,000 | 0.851 | 63 |
| PCG | 2026-01-09 | 2,197,910,425 | 2,675,428,603; 2,675,580,992; 2,675,711,544; 2,679,968,318 | 0.821 | 63 |
| RDDT | 2025-12-22 | 51,670,930 | 129,546,488; 134,585,853; 139,649,508; 141,965,642 | 0.377 | 0 |
| SBUX | 2026-07-14 | 139,700,000 | 1,137,100,000; 1,139,300,000; 1,140,000,000; 1,140,000,000 | 0.123 | 63 |
| XYZ | 2026-01-09 | 59,998,000 | 555,001,000; 549,574,000; 539,103,000; 535,195,000 | 0.110 | 63 |

**Member-days changed, against ruling 1's panel:** 378, and nothing else. Each carries the count
before the removed sample: DGX 111,620,000, EG 42,980,000, NWS 191,100,000, PCG 2,675,580,992,
SBUX 1,139,300,000, XYZ 549,574,000. The count-move report falls from 103 moves to 87.

**PCG, against the ruling's words.** The ruling said nothing is repaired "where the samples
alternate, as CEG's and PCG's do". The rule as worded leaves CEG's 2026 flips and PCG's 2022 ones,
whose neighbours disagree. But PCG's 2026-01-09 sample has four neighbours agreeing at 2.675 to
2.680 billion, so the rule removes it.

Its own record's weighted count that day is the removed value exactly, 2,197,910,425, so both of
Polygon's counts agree with it. PCG's records alternate in both fields, near 2.2 and 2.68 billion,
over years. Applied as worded; a decision below.

**The weighted counts, set against the repairs.** A removed sample's own record still meets the
weighted check, now against the count carried through it:

- **They confirm three repairs:** DGX 0.9942, EG 0.9989, SBUX 1.0004.
- **They say nothing for three:** NWS 2.98, RDDT 1.41, XYZ 1.11, all companies with several
  share classes, whose weighted count spans them.
- **They question PCG's:** 0.8215 against the carried count.

### Ruling 3: the overrides file

`config/sp500_share_count_overrides.json` sits in the repo; `securities.read_overrides` reads and
validates it:

- an action, "carry" or "none";
- a count for a carry and only for a carry;
- a range that runs forward;
- evidence;
- no two ranges of one security overlapping.

In `build_panel`, a "carry" sets the carried count on the range's first day, and a "none" leaves
none and carries none out. The samples inside either are not read, and a split inside still
applies. The build lists every entry it uses (`build_data.txt`):

```text
HON    BBG000H556T9 2026-10-12..2026-11-03: carry 635,675,701 -- 17 days in its spans, 17 member-days, 2 sample(s) inside not read
SKYD   BBG01VS5NK99 2026-10-06..2026-11-03: no count -- 21 days in its spans, 21 member-days, 0 sample(s) inside not read
```

**Member-days changed, against ruling 2's panel:** 38, and nothing else.

- **HON:** 17, from 316,940,010 and 317,358,518 to 635,675,701.
- **SKYD:** 21, which lose their count.

Each entry's `last` is the window's last day, 2026-11-03, the last its evidence reaches. A panel
over a longer window needs the entry reviewed, rather than carrying a ruling past its evidence.

### The repaired panel

All seven guards pass (`build_data.txt`).

- **Member-days without a count:** 234.
  - **176 inside spans, before a first count:**
    - FDXF 109 and VYLR 24;
    - Q 13 and SOLS 13;
    - AMTM 5, GEV 4, and SKYD's 2 before PSKY's first count;
    - CEG, GEHC, SOLV, VLTO (2) and WBD, one or two days each.
  - **21 by SKYD's ruling.**
  - **37 past a member's last trade.**
- **The count-move report:** 86.
- **The weighted check:**
  - carried: 47 met, 8 flagged -- CCL, HON's three contaminated records, and NWS, PCG, RDDT
    and XYZ above;
  - own records: 11,846 met, 754 flagged.

### The model against midoffice08's run D

Run D's configuration on each panel (`compare_counts.txt`). Run D, re-run first on midoffice08's
panel, reproduces midoffice08's run D: its report identical, its factor returns, specific risk and
R-squared too.

| Step | Mean R-squared | Members a period | Largest specific-risk moves | Size pair: smallest 85 |
|---|---|---|---|---|
| run D | 0.293189 | 499-503, median 502 | -- | 11.82% |
| + ruling 1 (R1) | 0.293183 | the same | GEV +0.04 | 11.82% |
| + ruling 2 (R2) | 0.293165 | the same | SBUX -0.30, DGX -0.11; OKTA, SBNY and six fallbacks +0.09 to +0.12 | 11.88% |
| + ruling 3 (R3) | 0.293204 | 499-503, at most one fewer (SKYD) | HON -0.08, ECHO +0.08 | 11.87% |

- **Against run D:** every factor's mean is within 0.10% (Retail -0.10%, Transport & Utilities
  +0.08%, Services -0.07%), and every vol within 0.09%.
- **The noise shares:** Construction 38.1% to 38.2%, Finance & Real Estate 30.9% to 31.0%,
  Wholesale 46.2% unchanged, Size Nonlinearity 47.0% to 46.6%.
- **Unchanged:** the residual spread (0.1330 to 0.1333), the fallbacks (the same 11) and the
  regression universe (567).
- **Specific risk:** forecast for 551 securities, against 552. SKYD, with no count on the
  forecast date, has no Size exposure and so no forecast.

---

## Item 2: the defaults

`scripts/run_sp500_model.py` changes in three ways:

- **The panel argument is optional:** `DEFAULT_PANEL` is the repaired panel.
- **`--returns` defaults to total.**
- **The old panel, on tickers, needs `--returns price`.** Without it the run stops: "this panel,
  on tickers, carries price returns alone: run it with --returns price; total returns, the
  default, need build_sp500_securities.py's".

**The run at its new defaults, `run_sp500_model.py --industries --membership`, equals item 1's
run exactly** -- item 1's being the repaired panel given explicitly with `--returns total
--volumes traded`:

- the reports are identical line for line;
- every one of the captured model's 23 fields is identical (`defaults_equal.txt`).

The old panel with `--returns price` reproduces midoffice08's run A, its report identical. After
item 3 it needs `--specific-history regression` too, and then it reproduces run A again, exactly.

---

## Item 3: residuals for every covered name

### The change

`estimate_cross_sectional_model(..., specific_history=...)`:

- **"regression"** is the default and every caller's behaviour as it was, to the byte.
- **"covered":** in each estimated period, after the regression on its members, every security
  with a return and a complete characteristic row that the regression did not hold -- not a
  member, or no weight -- gets its specific return: its return less its exposures, the market
  column included, times the period's factor returns.
  - **No degrees-of-freedom inflation.** That corrects fitted residuals for what the fit
    absorbed, and no fit absorbed these.
  - **The regression's own residuals stay as they were,** inflation and all.
- **Each security's own history is built from every specific return it has:**
  - the standard deviation `historical` takes;
  - the history `blend` weighs;
  - the short-history count.
- **The specific risk model is still fitted on the regression residuals alone,** over the
  regression universe: S, V, gamma, kappa and the relative residual spread.
- **A security outside the regression universe with two or more specific returns is blended:**
  its own history, weighed against the characteristic forecast extrapolated to it. It is listed
  in `blended_outside`, no longer a fallback. A fallback is now one with no specific return of
  its own.
- **The floored list stays about the universe's estimates.**

`run_sp500_model.py --specific-history covered` is the default, and `regression` stays runnable.
The run's specific-risk lines add the count of out-of-regression returns and the outside-blended
securities. The characteristics' weight is now read at the median history the blend actually
uses.

### Checked: the estimation is the regression's

Covered (NC) against regression (NR), both at the new defaults (`covered_report.txt`):

- **Identical:**
  - the factor returns and vols;
  - mean R-squared;
  - the cross-section sizes;
  - the regression universe;
  - every exposure;
  - the residual spread (MAD and sd);
  - the shares of V-squared and of the spread's square.
- **Regression mode:** NR's report is identical, line for line, to item 2's run.

### The fallback list: 11 to none

| Security | Specific returns | Characteristics' weight | Forecast, NR | Forecast, NC |
|---|---|---|---|---|
| AMTM | 399 | 6.60% | 43.33% | 41.41% |
| FBIN | 938 | 2.91% | 55.12% | 32.66% |
| GAP | 938 | 2.91% | 45.52% | 47.15% |
| IPGP | 938 | 2.91% | 51.56% | 43.73% |
| MBC | 847 | 3.22% | 59.81% | 40.61% |
| PENN | 938 | 2.91% | 49.14% | 43.91% |
| PVH | 938 | 2.91% | 49.63% | 38.79% |
| SOLS | 127 | 18.25% | 57.0375% | 57.0378% |
| UA | 938 | 2.91% | 43.67% | 41.97% |
| UAA | 938 | 2.91% | 51.13% | 44.76% |
| VNO | 938 | 2.91% | 37.88% | 30.75% |

Every one now blends a history of its own. Nine left the index before the first estimated
period and traded on. AMTM and SOLS were members only while too new to have a Momentum exposure,
so they reached no regression. SOLS's own volatility, about 55%, sits close to its characteristic
forecast, so its forecast barely moves (`sols_sbny.txt`).

### The joiners in their first year as members

87 securities joined during the window. These are the 21 that joined in the 252 trading days
before the forecast date, 2026-11-03, and have a forecast:

| | Joined | Periods as a member | History, NR | History, NC | Weight, NR | Weight, NC | Forecast, NR | Forecast, NC |
|---|---|---|---|---|---|---|---|---|
| Q | 2025-11-03 | 252 | 125 | 125 | 18.49% | 18.49% | 38.16% | 38.16% |
| SNDK | 2025-11-28 | 234 | 233 | 300 | 10.81% | 8.60% | 72.61% | 77.03% |
| ARES | 2025-12-11 | 225 | 224 | 938 | 11.20% | 2.91% | 30.27% | 23.96% |
| CRH | 2025-12-22 | 218 | 217 | 654 | 11.52% | 4.13% | 31.68% | 27.47% |
| CVNA | 2025-12-22 | 218 | 217 | 938 | 11.52% | 2.91% | 52.73% | 83.35% |
| FIX | 2025-12-22 | 218 | 217 | 938 | 11.52% | 2.91% | 30.94% | 35.33% |
| CIEN | 2026-02-09 | 186 | 185 | 938 | 13.26% | 2.91% | 56.35% | 42.36% |
| COHR | 2026-03-23 | 157 | 156 | 938 | 15.36% | 2.91% | 66.50% | 54.52% |
| ECHO | 2026-03-23 | 157 | 156 | 938 | 15.36% | 2.91% | 57.44% | 75.74% |
| LITE | 2026-03-23 | 157 | 156 | 938 | 15.36% | 2.91% | 67.93% | 52.95% |
| VRT | 2026-03-23 | 157 | 156 | 938 | 15.36% | 2.91% | 47.76% | 44.07% |
| CASY | 2026-04-09 | 145 | 144 | 938 | 16.44% | 2.91% | 46.05% | 30.11% |
| VEEV | 2026-05-07 | 125 | 124 | 938 | 18.61% | 2.91% | 36.24% | 32.06% |
| FLEX | 2026-06-22 | 95 | 94 | 938 | 23.22% | 2.91% | 43.33% | 37.58% |
| MRVL | 2026-06-22 | 95 | 94 | 938 | 23.22% | 2.91% | 48.86% | 46.84% |
| FERG | 2026-08-05 | 64 | 63 | 440 | 31.21% | 6.02% | 26.74% | 29.20% |
| RDDT | 2026-08-18 | 55 | 54 | 531 | 34.67% | 5.04% | 72.16% | 70.38% |
| BE | 2026-09-21 | 32 | 31 | 938 | 48.39% | 2.91% | 65.76% | 77.14% |
| P | 2026-09-21 | 32 | 31 | 938 | 48.39% | 2.91% | 56.22% | 48.48% |
| TWLO | 2026-10-06 | 21 | 20 | 938 | 59.69% | 2.91% | 60.44% | 47.59% |
| OKTA | 2026-10-30 | 3 | 2 | 938 | 96.57% | 2.91% | 67.75% | 46.76% |

- **The characteristics' weight over them:** median 16.44% to 2.91%, mean 26.20% to 4.23%.
- **The weight:** w = v_hist / (v_model + v_hist), v_hist = 1/(2(n-1)), v_model = 0.1333^2,
  the same in both runs.
- **The exceptions:** Q joined on its first day and had no Momentum for half a year, so it brings
  nothing from before. SNDK, CRH, FERG and RDDT bring part of a history.
- **Across all 81 joiners with a forecast:** the covered history adds a median 345 specific
  returns, at most 936; 20 gain none.

### Leavers

52 securities that left the index are still forecast on 2026-11-03. Every one's history now runs
to the forecast date: 938 specific returns, or as many as it traded with a complete row.

- **Those that left early and were fallbacks:** GAP, IPGP, UA, UAA, PENN, PVH, FBIN, MBC and VNO
  (above).
- **The rest:** each history rises from what it had as a member -- 22 to 918 -- to 938, or
  SBNY's 851. Their forecasts move both ways:

| Leaver | Forecast, NR | Forecast, NC |
|---|---|---|
| LUMN | 63.19% | 88.53% |
| SEDG | 54.27% | 81.04% |
| NWL | 37.34% | 53.00% |
| OGN | 33.60% | 48.93% |
| FMC | 42.13% | 56.96% |
| RHI | 22.64% | 35.76% |
| ZION | 35.34% | 25.01% |
| TTD | 59.72% | 51.80% |

**SBNY**, Signature Bank, failed in March 2023. Its shares traded on, from 0.0022 to about 0.2,
under SBNY, the identity table's span running to the window's end. 829 daily returns after
2023-03-28 give a 412% annual volatility, the largest +448%. Its history goes from 22 regression
residuals to 851 specific returns, and its forecast from 58.25% to 395.72%. It is not a member
on the forecast date, and its count is the pre-failure 57,630,000 carried 1:1. The covered history
does what was ruled; whether a forecast should come from a history like that is a decision below.

### The specific-risk percentiles

| | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| all 551 with a forecast, NR | 15.58% | 19.55% | 24.79% | 31.69% | 40.69% |
| all 551 with a forecast, NC | 15.58% | 19.55% | 24.79% | 31.95% | 40.56% |

Over the 499 members on 2026-11-03: p10 15.31% to 15.31%, median 24.00% to 23.96%, p90 38.19% to
38.01%.

### The names whose forecasts move most

108 of 551 forecasts move.

- **The largest falls:**
  - FBIN, 55.12% to 32.66%;
  - OKTA, 67.75% to 46.76%;
  - MBC, 59.81% to 40.61%;
  - CASY, 46.05% to 30.11%;
  - LITE, 67.93% to 52.95%;
  - CIEN, 56.35% to 42.36%;
  - TWLO, 60.44% to 47.59%;
  - COHR, 66.50% to 54.52%;
  - DDOG, 58.73% to 47.68%;
  - PVH, 49.63% to 38.79%;
  - ZION, 35.34% to 25.01%;
  - TTD, 59.72% to 51.80%.
- **The largest rises:**
  - SBNY, 58.25% to 395.72%;
  - CVNA, 52.73% to 83.35%;
  - SEDG, 54.27% to 81.04%;
  - LUMN, 63.19% to 88.53%;
  - ECHO, 57.44% to 75.74%;
  - NWL, 37.34% to 53.00%;
  - OGN, 33.60% to 48.93%;
  - FMC, 42.13% to 56.96%;
  - RHI, 22.64% to 35.76%;
  - COIN, 52.84% to 65.23%;
  - BE, 65.76% to 77.14%;
  - WHR, 30.65% to 40.28%.
- **Two patterns:**
  - Joiners whose pre-joining years were volatile rise: Carvana's 2022 collapse and recovery,
    EchoStar's.
  - Joiners whose pre-joining years were calmer fall.

The history is still equally weighted over the window, so that past counts as much as the
present (a decision below).

---

## Key decisions

1. **Ruling 2 read exactly:**
   - "agree with each other within 10%" as the four neighbours' largest over their smallest
     within 1.10;
   - "differs beyond 10%" as from each of the four;
   - "no split to explain it" as no applied split between the outer two neighbours;
   - one pass over the samples as they are;
   - the sequence: the plan's samples and the confirmed first counts, under all the security's
     keys, stale ones excluded.
2. **PCG's removal applied as the rule is worded,** flagged with the evidence against it, not
   carved out by a special case.
3. **Ruling 1's confirmation is against the span's next sample with a count** (the bracket's
   end), found by a second bisection; the samples the first bisection asked for are reused from
   the cache.
4. **The overrides' semantics and validation:**
   - a carry sets the count on the range's first day, a none carries nothing out;
   - samples inside a range are not read, and splits inside still apply;
   - the file is validated;
   - each `last` is the evidence's last day, not open-ended.
5. **The repaired panel is a new file, `_repaired`.** The build refuses to overwrite, and its
   flags rebuild midoffice08's panel exactly and each intermediate.
6. **Each ruling was run in the model on its own (R1, R2, R3),** beyond the brief's single
   comparison, to attribute the change.
7. **The old panel stops without `--returns price`,** rather than silently falling back to price
   returns.
8. **Covered histories:**
   - the library's default stays "regression", so existing callers are unchanged, and the run's
     default is "covered";
   - no degrees-of-freedom inflation outside the regression;
   - an outside security with a history is blended, and listed;
   - the floored list stays about the universe's estimates;
   - history lengths are kept in the notes;
   - the run's blend-weight line uses the covered histories.
9. **The docs brought in line, though the brief listed none:** the doc stated the old count rules
   and nothing of covered histories.

## Decisions left for Jeff

1. **PCG's sample of 2026-01-09.** The rule as worded removes it, and both of Polygon's counts
   that day agree with it.
   - **Keep the removal.**
   - **Or spare it:**
     - an overrides entry restoring 2,197,910,425 for 2026-01-09 to 2026-04-10;
     - or a general addition: a sample its own record's weighted count confirms, within 10%, is
       no error. That would spare PCG's alone of the seven: the weighted counts of DGX, EG and
       SBUX confirm the repairs, and NWS, RDDT and XYZ are multi-class.
2. **Leavers' histories after leaving.** As ruled, a leaver forecasts from its own history after
   it leaves. For Signature Bank that is a failed bank's shares at fractions of a cent: 396%.
   - **Keep.**
   - **Or bound it:** stop a history -- or the forecast -- once a security no longer trades
     normally, below a price or after a seizure. Or, for the identity table: should Signature
     Bank's post-failure prices be a span of the panel at all?
3. **How a history is weighted.** It is equally weighted over the window. Covered histories bring
   joiners' oldest volatility in at full weight. A half-life on the history is the natural
   alternative, a question for the out-of-sample bias statistics.

---

## Tests

**The suite:** 2,295 passed, 1 skipped, 19 warnings, in 68.09 s, with
`--ignore=tests/test_market_data.py`. midoffice alone: 844 passed, 1 skipped. Before this piece:
2,290 and 839.

**New: 5.**

`tests/midoffice/test_securities.py`, 4:

- `test_a_first_count_the_next_sample_does_not_confirm_is_passed_over` -- a first count four
  times the next sample is passed over for the first record confirmed; `confirm=False` keeps it;
  one at 0.92 stands.
- `test_an_isolated_bad_sample_is_removed_and_alternation_steps_and_splits_are_left` -- DGX's
  case removed with its four neighbours. None of these is removed:
  - two wrong samples in a row;
  - alternation;
  - a real step;
  - a sample without two on each side;
  - one with a split between its neighbours, removed only without the split.
- `test_an_override_carries_its_count_or_none_and_reads_no_sample_inside` -- a carry over a
  range after a split; a none ending a reorganisation's carry; the samples inside not read;
  each override's days, member-days and unread samples.
- `test_the_overrides_file_is_read_with_its_rules` -- a good entry read; each of these refused:
  - a carry without a count;
  - a none with one;
  - an unknown action;
  - a backward range;
  - no evidence;
  - an overlap.

  The repo's file reads with HON and SKYD.

`tests/midoffice/test_crosssectional.py`, 1:

- `test_covered_histories_measure_every_specific_return_and_leave_the_fit_alone`:
  - the factor returns, covariance, R-squared, V fit, spread and exposures are identical;
  - the leavers' histories run to the end;
  - the never-member is blended, not a fallback;
  - the outside-return count is exact;
  - an outside specific return is the return less exposures times factor returns, to 1e-12,
    through the historical method;
  - an unknown mode is refused.

**Hygiene:** every changed Python file parses, is ASCII and keeps CRLF, counted in bytes. The JSON
and the brief are LF (`hygiene_final.txt`). ruff over the changed files finds three things, each
at HEAD too (checked) and left:

- `run_sp500_model.py`'s unused `HorizonCovariance`;
- `crosssectional.py`'s unused `Mapping`;
- `test_crosssectional.py`'s import order.

**The final code, re-run** (all in the appendix):

- **midoffice08's panel:** rebuilt by its rules, its arrays exact.
- **midoffice08's run D:** reproduced exactly.
- **midoffice08's run A:** reproduced exactly with `--returns price --specific-history
  regression`.
- **Item 2's run:** reproduced by `--specific-history regression` at the new defaults, line for
  line.

---

## Failures and debugging trail

- **A patch dropped the `--returns` help string's closing parenthesis.** The hygiene parse caught
  the syntax error before any run, and it was fixed.
- **The state file's rewrite was refused:** the file had changed on disk since my last write,
  converted to CRLF at the end of midoffice08. Read, and written again.
- **A test reading the repo's overrides file used a path relative to the working directory.** It
  now uses the test file's own location.
- **A test line over 100 characters** (ruff E501): wrapped.
- **`diff <(...) file` reported a difference that was not there:** process substitution fails in
  this Git Bash. A plain diff of two files showed run D's report identical. Noted in the state
  file.
- **SOLS's forecast looked unchanged under covered histories.** At four decimals it moves,
  57.0375% to 57.0378%: its own volatility sits close to its characteristic forecast.
- **The heredocs with apostrophes worked this time,** unlike in midoffice06 to 08.

## Problems found, not fixed

1. **PCG's 2026-01-09 sample is removed** against the ruling's expectation (a decision above).
   PCG's records alternate between two counts in both fields.
2. **Signature Bank's post-failure history** gives it a 396% forecast (a decision above).
3. **The specific-risk history is equally weighted over the window** (a decision above).
4. **FDXF and Vylor carry no share-class count in the window,** so they enter no regression.
5. **Three ruff findings predate this piece** (above).
6. **`tests/test_market_data.py` looks for `../core_repo/src`,** not on this machine.

## Scratch

`claude/scratch/midoffice09/` (gitignored). The scripts:

- `hygiene.py` (from midoffice08), and the patch scripts: `patch_overrides.py`,
  `patch_build_rules.py`, `patch_covered.py`, `patch_docs.py`;
- `capture.py`, which runs the model in-process and keeps it -- its `capture/` directory holds
  each run's report and pickle (D, R1, R2, R3, N, A, AR, NR, NC);
- `compare_counts.py` (midoffice08's `compare.py`, its steps changed), `member_days_changed.py`
  and `covered_report.py`.

Their outputs:

- the builds: `build_p0.txt` to `build_p3.txt`, `build_data.txt`, `overwrite_refusal.txt`;
- the measures: `member_days_changed.txt`, `compare_counts.txt`, `defaults_equal.txt`,
  `covered_report.txt`, `sols_sbny.txt`, `old_without_price.txt`;
- the checks: `pytest_all.txt`, `pytest_mid.txt`, `ruff_final.txt`, `hygiene_final.txt`;
- the panels `p0` to `p3`.

The key ones are copied into the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

## Open items for Chat

- **The three decisions above:**
  - PCG's sample;
  - leavers' histories after leaving;
  - how a history is weighted.

  The third bears on the out-of-sample bias statistics the brief puts after the noise-share
  question.
- **The noise-share question,** at the new defaults:
  - Construction 38.2% (8 members);
  - Wholesale 46.2% (12);
  - Finance & Real Estate 31.0% (12);
  - Size Nonlinearity 46.6%.

---

## Appendix: outputs, verbatim

The key outputs the body rests on, from `claude/scratch/midoffice09/`, in full unless said
otherwise.

### build_p0.txt -- midoffice08's rules (--first-counts found --repair none --overrides none --suffix ''), its first-count search and last lines

```text
The first-count search: each span that starts without a share count, no count carried into it, searched between its first sample and its first with a count; the first record with any count stands (--first-counts found)
  AMTM   2024-09-30..2026-11-03 (   59 member-days): AMTM: between 2024-09-30, without a count, and 2024-12-23, 243,300,000, the first record with a count is 2024-10-01's, 153,280,000, 0.630 of the next sample's, confirmed; in 5 lookup(s)
  CEG    2022-02-02..2026-11-03 ( 1193 member-days): CEG: between 2022-02-02, without a count, and 2022-04-20, 326,660,000, the first record with a count is 2022-02-03's, 326,500,000, 1.000 of the next sample's, confirmed; in 5 lookup(s)
  FDXF   2026-06-01..2026-11-03 (  109 member-days): FDXF: no sample in 2026-06-01..2026-11-03 carries a share-class count, so there is nothing to search between
  GEHC   2023-01-04..2026-11-03 (  962 member-days): GEHC: between 2023-01-04, without a count, and 2023-03-20, 453,930,000, the first record with a count is 2023-01-05's, 454,710,000, 1.002 of the next sample's, confirmed; in 5 lookup(s)
  GEV    2024-04-02..2026-11-03 (  651 member-days): GEV: between 2024-04-02, without a count, and 2024-06-27, 274,090,000, the first record with a count is 2024-04-03's, 1,088,330,000, 3.971 of the next sample's, confirmed; in 5 lookup(s)
  KVUE   2023-05-04..2026-10-30 (  798 member-days): KVUE: between 2023-05-04, without a count, and 2023-08-03, 1,914,890,000, the first record with a count is 2023-05-08's, 1,888,970,000, 0.986 of the next sample's, confirmed; in 6 lookup(s)
  Q      2025-11-03..2026-11-03 (  252 member-days): Q: between 2025-11-03, without a count, and 2026-01-28, 209,463,194, the first record with a count is 2025-11-20's, 209,463,194, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  PSKY   2025-08-07..2026-10-05 (  292 member-days): PSKY: between 2025-08-07, without a count, and 2025-11-05, 1,064,544,631, the first record with a count is 2025-08-11's, 1,064,544,631, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  SNDK   2025-02-24..2026-11-03 (  234 member-days): SNDK: between 2025-02-24, without a count, and 2025-05-15, 145,300,000, the first record with a count is 2025-02-26's, 144,000,000, 0.991 of the next sample's, confirmed; in 6 lookup(s)
  SOLS   2025-10-30..2026-11-03 (   36 member-days): SOLS: between 2025-10-30, without a count, and 2026-01-21, 158,727,456, the first record with a count is 2025-11-18's, 158,727,456, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  SOLV   2024-04-01..2026-11-03 (  652 member-days): SOLV: between 2024-04-01, without a count, and 2024-06-26, 172,710,000, the first record with a count is 2024-04-02's, 172,710,000, 1.000 of the next sample's, confirmed; in 5 lookup(s)
  VLTO   2023-10-02..2026-11-03 (  776 member-days): VLTO: between 2023-10-02, without a count, and 2023-12-27, 246,310,000, the first record with a count is 2023-10-04's, 246,060,000, 0.999 of the next sample's, confirmed; in 6 lookup(s)
  VYLR   2026-10-01..2026-10-02 (    2 member-days): VYLR: no sample in 2026-10-01..2026-10-02 carries a share-class count, so there is nothing to search between
  VYLR   2026-10-05..2026-11-03 (   22 member-days): VYLR: no sample in 2026-10-05..2026-11-03 carries a share-class count, so there is nothing to search between
  WBD    2022-04-11..2026-10-06 ( 1125 member-days): WBD: between 2022-04-11, without a count, and 2022-07-13, 2,426,840,000, the first record with a count is 2022-04-12's, 2,233,730,000, 0.920 of the next sample's, confirmed; in 5 lookup(s)
  15 spans, 66 lookups, 0 over the network this run

Isolated errors: a sample whose two samples before and two after agree with each other within 10%, and which differs from each of them beyond it, no split between, is not carried -- none removed (--repair none)

...
Wrote
  claude\scratch\midoffice09\p0\sp500_securities_2022-01-03_2026-11-03.npz
  claude\scratch\midoffice09\p0\sp500_securities_2022-01-03_2026-11-03.json
```

### build_p1.txt -- ruling 1, its first-count search

```text
The first-count search: each span that starts without a share count, no count carried into it, searched between its first sample and its first with a count; a first count the next sample does not confirm within 10% is passed over for the first it confirms, the days before that without a count
  AMTM   2024-09-30..2026-11-03 (   59 member-days): AMTM: between 2024-09-30, without a count, and 2024-12-23, 243,300,000, the first record with a count is 2024-10-01's, 153,280,000, 0.630 of the next sample's, unconfirmed; the first the next sample confirms is 2024-10-07's, 243,300,000; in 11 lookup(s)
  CEG    2022-02-02..2026-11-03 ( 1193 member-days): CEG: between 2022-02-02, without a count, and 2022-04-20, 326,660,000, the first record with a count is 2022-02-03's, 326,500,000, 1.000 of the next sample's, confirmed; in 5 lookup(s)
  FDXF   2026-06-01..2026-11-03 (  109 member-days): FDXF: no sample in 2026-06-01..2026-11-03 carries a share-class count, so there is nothing to search between
  GEHC   2023-01-04..2026-11-03 (  962 member-days): GEHC: between 2023-01-04, without a count, and 2023-03-20, 453,930,000, the first record with a count is 2023-01-05's, 454,710,000, 1.002 of the next sample's, confirmed; in 5 lookup(s)
  GEV    2024-04-02..2026-11-03 (  651 member-days): GEV: between 2024-04-02, without a count, and 2024-06-27, 274,090,000, the first record with a count is 2024-04-03's, 1,088,330,000, 3.971 of the next sample's, unconfirmed; the first the next sample confirms is 2024-04-08's, 272,080,000; in 11 lookup(s)
  KVUE   2023-05-04..2026-10-30 (  798 member-days): KVUE: between 2023-05-04, without a count, and 2023-08-03, 1,914,890,000, the first record with a count is 2023-05-08's, 1,888,970,000, 0.986 of the next sample's, confirmed; in 6 lookup(s)
  Q      2025-11-03..2026-11-03 (  252 member-days): Q: between 2025-11-03, without a count, and 2026-01-28, 209,463,194, the first record with a count is 2025-11-20's, 209,463,194, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  PSKY   2025-08-07..2026-10-05 (  292 member-days): PSKY: between 2025-08-07, without a count, and 2025-11-05, 1,064,544,631, the first record with a count is 2025-08-11's, 1,064,544,631, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  SNDK   2025-02-24..2026-11-03 (  234 member-days): SNDK: between 2025-02-24, without a count, and 2025-05-15, 145,300,000, the first record with a count is 2025-02-26's, 144,000,000, 0.991 of the next sample's, confirmed; in 6 lookup(s)
  SOLS   2025-10-30..2026-11-03 (   36 member-days): SOLS: between 2025-10-30, without a count, and 2026-01-21, 158,727,456, the first record with a count is 2025-11-18's, 158,727,456, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  SOLV   2024-04-01..2026-11-03 (  652 member-days): SOLV: between 2024-04-01, without a count, and 2024-06-26, 172,710,000, the first record with a count is 2024-04-02's, 172,710,000, 1.000 of the next sample's, confirmed; in 5 lookup(s)
  VLTO   2023-10-02..2026-11-03 (  776 member-days): VLTO: between 2023-10-02, without a count, and 2023-12-27, 246,310,000, the first record with a count is 2023-10-04's, 246,060,000, 0.999 of the next sample's, confirmed; in 6 lookup(s)
  VYLR   2026-10-01..2026-10-02 (    2 member-days): VYLR: no sample in 2026-10-01..2026-10-02 carries a share-class count, so there is nothing to search between
  VYLR   2026-10-05..2026-11-03 (   22 member-days): VYLR: no sample in 2026-10-05..2026-11-03 carries a share-class count, so there is nothing to search between
  WBD    2022-04-11..2026-10-06 ( 1125 member-days): WBD: between 2022-04-11, without a count, and 2022-07-13, 2,426,840,000, the first record with a count is 2022-04-12's, 2,233,730,000, 0.920 of the next sample's, confirmed; in 5 lookup(s)
  15 spans, 78 lookups, 9 over the network this run

```

### build_p2.txt -- ruling 2, its removals

```text
Isolated errors: a sample whose two samples before and two after agree with each other within 10%, and which differs from each of them beyond it, no split between, is not carried -- 7 removed
  DGX    DGX    2025-04-09       11,610,000; around it 2024-10-07 111,320,000, 2025-01-07 111,620,000, 2025-07-11 111,635,472, 2025-10-09 111,823,441 -- 0.104 of their median
  EG     EG     2025-04-09       47,790,000; around it 2024-10-07 43,270,000, 2025-01-07 42,980,000, 2025-07-11 42,534,728, 2025-10-09 41,943,291 -- 1.118 of their median
  NWS    NWS    2024-07-09      162,170,000; around it 2024-01-05 191,390,000, 2024-04-08 191,100,000, 2024-10-07 190,080,000, 2025-01-07 189,770,000 -- 0.851 of their median
  PCG    PCG    2026-01-09    2,197,910,425; around it 2025-07-11 2,675,428,603, 2025-10-09 2,675,580,992, 2026-04-13 2,675,711,544, 2026-07-14 2,679,968,318 -- 0.821 of their median
  RDDT   RDDT   2025-12-22       51,670,930; around it 2025-06-24 129,546,488, 2025-09-23 134,585,853, 2026-03-25 139,649,508, 2026-06-25 141,965,642 -- 0.377 of their median
  SBUX   SBUX   2026-07-14      139,700,000; around it 2026-01-09 1,137,100,000, 2026-04-13 1,139,300,000, 2026-10-12 1,140,000,000, 2026-11-03 1,140,000,000 -- 0.123 of their median
  XYZ    XYZ    2026-01-09       59,998,000; around it 2025-07-11 555,001,000, 2025-10-09 549,574,000, 2026-04-13 539,103,000, 2026-07-14 535,195,000 -- 0.110 of their median

```

### build_data.txt -- all three rulings, the default build into data/ (the repaired panel), in full

```text
Plan
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended_fetch.csv: 628 spans, 600 securities, 617 tickers

The gate, offline: passes
  identity: 628 of 628 spans pass, 1 by the share-class FIGI
    COHR: COHR on 2022-09-08 names Coherent Corp. Common Stock (BBG000GS6QB4), reconciled to Coherent Corp. (BBG000BLW102) by its share-class FIGI, BBG001S5S6N6, which the expected security's own records carry
  splits: 55 events, 47 splits, 8 price-only, 0 contradictions, 2 untested

The first-count search: each span that starts without a share count, no count carried into it, searched between its first sample and its first with a count; a first count the next sample does not confirm within 10% is passed over for the first it confirms, the days before that without a count
  AMTM   2024-09-30..2026-11-03 (   59 member-days): AMTM: between 2024-09-30, without a count, and 2024-12-23, 243,300,000, the first record with a count is 2024-10-01's, 153,280,000, 0.630 of the next sample's, unconfirmed; the first the next sample confirms is 2024-10-07's, 243,300,000; in 11 lookup(s)
  CEG    2022-02-02..2026-11-03 ( 1193 member-days): CEG: between 2022-02-02, without a count, and 2022-04-20, 326,660,000, the first record with a count is 2022-02-03's, 326,500,000, 1.000 of the next sample's, confirmed; in 5 lookup(s)
  FDXF   2026-06-01..2026-11-03 (  109 member-days): FDXF: no sample in 2026-06-01..2026-11-03 carries a share-class count, so there is nothing to search between
  GEHC   2023-01-04..2026-11-03 (  962 member-days): GEHC: between 2023-01-04, without a count, and 2023-03-20, 453,930,000, the first record with a count is 2023-01-05's, 454,710,000, 1.002 of the next sample's, confirmed; in 5 lookup(s)
  GEV    2024-04-02..2026-11-03 (  651 member-days): GEV: between 2024-04-02, without a count, and 2024-06-27, 274,090,000, the first record with a count is 2024-04-03's, 1,088,330,000, 3.971 of the next sample's, unconfirmed; the first the next sample confirms is 2024-04-08's, 272,080,000; in 11 lookup(s)
  KVUE   2023-05-04..2026-10-30 (  798 member-days): KVUE: between 2023-05-04, without a count, and 2023-08-03, 1,914,890,000, the first record with a count is 2023-05-08's, 1,888,970,000, 0.986 of the next sample's, confirmed; in 6 lookup(s)
  Q      2025-11-03..2026-11-03 (  252 member-days): Q: between 2025-11-03, without a count, and 2026-01-28, 209,463,194, the first record with a count is 2025-11-20's, 209,463,194, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  PSKY   2025-08-07..2026-10-05 (  292 member-days): PSKY: between 2025-08-07, without a count, and 2025-11-05, 1,064,544,631, the first record with a count is 2025-08-11's, 1,064,544,631, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  SNDK   2025-02-24..2026-11-03 (  234 member-days): SNDK: between 2025-02-24, without a count, and 2025-05-15, 145,300,000, the first record with a count is 2025-02-26's, 144,000,000, 0.991 of the next sample's, confirmed; in 6 lookup(s)
  SOLS   2025-10-30..2026-11-03 (   36 member-days): SOLS: between 2025-10-30, without a count, and 2026-01-21, 158,727,456, the first record with a count is 2025-11-18's, 158,727,456, 1.000 of the next sample's, confirmed; in 6 lookup(s)
  SOLV   2024-04-01..2026-11-03 (  652 member-days): SOLV: between 2024-04-01, without a count, and 2024-06-26, 172,710,000, the first record with a count is 2024-04-02's, 172,710,000, 1.000 of the next sample's, confirmed; in 5 lookup(s)
  VLTO   2023-10-02..2026-11-03 (  776 member-days): VLTO: between 2023-10-02, without a count, and 2023-12-27, 246,310,000, the first record with a count is 2023-10-04's, 246,060,000, 0.999 of the next sample's, confirmed; in 6 lookup(s)
  VYLR   2026-10-01..2026-10-02 (    2 member-days): VYLR: no sample in 2026-10-01..2026-10-02 carries a share-class count, so there is nothing to search between
  VYLR   2026-10-05..2026-11-03 (   22 member-days): VYLR: no sample in 2026-10-05..2026-11-03 carries a share-class count, so there is nothing to search between
  WBD    2022-04-11..2026-10-06 ( 1125 member-days): WBD: between 2022-04-11, without a count, and 2022-07-13, 2,426,840,000, the first record with a count is 2022-04-12's, 2,233,730,000, 0.920 of the next sample's, confirmed; in 5 lookup(s)
  15 spans, 78 lookups, 0 over the network this run

Isolated errors: a sample whose two samples before and two after agree with each other within 10%, and which differs from each of them beyond it, no split between, is not carried -- 7 removed
  DGX    DGX    2025-04-09       11,610,000; around it 2024-10-07 111,320,000, 2025-01-07 111,620,000, 2025-07-11 111,635,472, 2025-10-09 111,823,441 -- 0.104 of their median
  EG     EG     2025-04-09       47,790,000; around it 2024-10-07 43,270,000, 2025-01-07 42,980,000, 2025-07-11 42,534,728, 2025-10-09 41,943,291 -- 1.118 of their median
  NWS    NWS    2024-07-09      162,170,000; around it 2024-01-05 191,390,000, 2024-04-08 191,100,000, 2024-10-07 190,080,000, 2025-01-07 189,770,000 -- 0.851 of their median
  PCG    PCG    2026-01-09    2,197,910,425; around it 2025-07-11 2,675,428,603, 2025-10-09 2,675,580,992, 2026-04-13 2,675,711,544, 2026-07-14 2,679,968,318 -- 0.821 of their median
  RDDT   RDDT   2025-12-22       51,670,930; around it 2025-06-24 129,546,488, 2025-09-23 134,585,853, 2026-03-25 139,649,508, 2026-06-25 141,965,642 -- 0.377 of their median
  SBUX   SBUX   2026-07-14      139,700,000; around it 2026-01-09 1,137,100,000, 2026-04-13 1,139,300,000, 2026-10-12 1,140,000,000, 2026-11-03 1,140,000,000 -- 0.123 of their median
  XYZ    XYZ    2026-01-09       59,998,000; around it 2025-07-11 555,001,000, 2025-10-09 549,574,000, 2026-04-13 539,103,000, 2026-07-14 535,195,000 -- 0.110 of their median

The panel
  1214 trading days, 2022-01-03 .. 2026-11-03; 592 securities, 628 spans under 617 tickers; 8 reorganisations merged, each one column under its successor's key
    BG     CIK0001144519 -> CIK0001996862
    BLK    BBG000C2PW58 -> BBG01PSW2WN4
    CCL    BBG000BF6LY3 -> CIK0000815097
    J      BBG000BMFFQ0 -> BBG019C1BQR4
    LH     BBG000D9DMK0 -> BBG01MMT6PL7
    OKE    BBG000BQHGR6 -> BBG024TZWVN1
    VYLR   BBG024W5VJC3 -> BBG024CQR6X9
    XOM    BBG000GZQ728 -> BBG023CY9MM1
  splices -- a ticker another security held before a column's span, which never enters that column: BNY, COHR, DOC, META
  a change of ticker inside a column, bridged by the raw closes: 26
    BALL   BLL 2022-05-09 69.49 -> BALL 2022-05-10 69.09: return -0.5756%; the earlier run's scale 1
    BNY    BK 2026-05-20 137.16 -> BNY 2026-05-21 138.98: return +1.3269%; the earlier run's scale 1
    COHR   IIVI 2022-09-07 41.95 -> COHR 2022-09-08 43.26: return +3.1228%; the earlier run's scale 1
    COR    ABC 2023-08-29 179.98 -> COR 2023-08-30 178.90: return -0.6001%; the earlier run's scale 1
    CPAY   FLT 2024-03-22 303.26 -> CPAY 2024-03-25 307.66: return +1.4509%; the earlier run's scale 1
    DAY    CDAY 2024-01-31 69.52 -> DAY 2024-02-01 69.94: return +0.6041%; the earlier run's scale 1
    DOC    PEAK 2024-03-01 17.10 -> DOC 2024-03-04 16.85: return -1.4620%; the earlier run's scale 1
    ECHO   SATS 2026-06-23 103.92 -> ECHO 2026-06-24 99.86: return -3.9022%; the earlier run's scale 1
    EG     RE 2023-07-07 351.28 -> EG 2023-07-10 347.06: return -1.2013%; the earlier run's scale 1
    ELV    ANTM 2022-06-27 482.58 -> ELV 2022-06-28 473.09: return -1.9665%; the earlier run's scale 1
    EXE    CHK 2024-10-01 81.46 -> EXE 2024-10-02 83.24: return +2.1851%; the earlier run's scale 1
    FBIN   FBHS 2022-12-14 61.90 -> FBIN 2022-12-15 57.62: return -6.9144%; the earlier run's scale 1
    FISV   FI 2025-11-10 63.80 -> FISV 2025-11-11 64.26: return +0.7210%; the earlier run's scale 1
    FISV   FISV 2023-06-06 114.23 -> FI 2023-06-07 115.78: return +1.3569%; the earlier run's scale 1
    FRCB   FRC 2023-04-28 3.51 -> FRCB 2023-05-03 0.33: return -90.4957%; the earlier run's scale 1
    GAP    GPS 2024-08-21 24.55 -> GAP 2024-08-22 24.44: return -0.4481%; the earlier run's scale 1
    GEN    NLOK 2022-11-07 21.66 -> GEN 2022-11-08 21.62: return -0.1847%; the earlier run's scale 1
    META   FB 2022-06-08 196.64 -> META 2022-06-09 184.00: return -6.4280%; the earlier run's scale 1
    MRSH   MMC 2026-01-13 182.70 -> MRSH 2026-01-14 183.27: return +0.3120%; the earlier run's scale 1
    P      PSTG 2026-04-16 67.80 -> P 2026-04-17 66.97: return -1.2242%; the earlier run's scale 1
    PARA   VIAC 2022-02-16 29.58 -> PARA 2022-02-17 29.35: return -0.7776%; the earlier run's scale 200
    RVTY   PKI 2023-05-15 115.24 -> RVTY 2023-05-16 114.21: return -0.8938%; the earlier run's scale 1
    SKYD   PSKY 2026-10-05 9.78 -> SKYD 2026-10-06 9.53: return -2.5064%; the earlier run's scale 1
    VMRK   EQR 2026-08-17 63.66 -> VMRK 2026-08-18 64.03: return +0.5812%; the earlier run's scale 1
    WTW    WLTW 2022-01-07 231.56 -> WTW 2022-01-10 232.62: return +0.4578%; the earlier run's scale 1
    XYZ    SQ 2025-01-17 86.96 -> XYZ 2025-01-21 89.50: return +2.9209%; the earlier run's scale 1
  priced cells 685,089; member-days 610,842, of them priced 610,803

Total returns
  8,229 distributions reinvested; exact duplicates dropped 1
    TEL 2024-11-22 0.65 CD frequency 4, paid 2024-12-06
  ex-dates without a price in their column, reinvested on the next priced day: 0
  on or before a column's first price, which no return carries: 4
    A 2022-01-03 0.21
    DG 2022-01-03 0.42
    PWR 2022-01-03 0.07
    RJF 2022-01-03 0.34
  207,046 security-days whose total return is not their price return; mean annual difference over the members' days 1.90%
  SPY: 19 distributions; total less price return 0.0633 over the window, 1.31% a year

Share counts
  splits applied on their execution dates: 47; stale samples read past: 15
    APH    2026-10-12 at 1,232,983,457
    APH    2026-11-03 at 1,232,983,457
    BKNG   2026-04-13 at 31,673,346
    CRWD   2026-07-14 at 254,564,820
    CVNA   2026-07-14 at 143,257,677
    FAST   2025-07-11 at 573,620,638
    IBKR   2025-07-11 at 108,958,830
    KLAC   2026-07-14 at 130,627,521
    MNST   2026-10-12 at 979,525,882
    MNST   2026-11-03 at 979,525,882
    NOW    2026-01-09 at 207,678,139
    ORLY   2025-07-11 at 56,989,455
    SMCI   2024-10-01 at 58,560,000
    TPL    2026-01-09 at 22,979,410
    WSM    2024-07-09 at 64,310,000
  counts carried into a span from before it, until a sample of its own resets them: 7
    CCL    from 2026-05-07 at 1,239,000,212 (a reorganisation), to 2026-08-07
    COHR   from 2022-09-08 at 130,870,000 (from its own earlier span), to 2022-12-07
    FRCB   from 2023-05-03 at 176,740,000 (from its own earlier span), to the span end, 1:1, FLAGGED
    OKE    from 2026-09-11 at 630,369,667 (a reorganisation), to the span end, 1:1, FLAGGED
    SBNY   from 2023-03-28 at 57,630,000 (from its own earlier span), to the span end, 1:1, FLAGGED
    SMCI   from 2024-10-01 at 585,600,000 (from its own earlier span), to 2024-10-07
    XOM    from 2026-07-02 at 4,144,947,162 (a reorganisation), to the span end, 1:1, FLAGGED
  overrides, config/sp500_share_count_overrides.json: 2 in the file, 2 used
    HON    BBG000H556T9 2026-10-12..2026-11-03: carry 635,675,701 -- 17 days in its spans, 17 member-days, 2 sample(s) inside not read; midoffice08, decision 3 (Jeff, 2026-11-10)
      A contaminated record. Polygon's HON records halve from 2026-10-12 -- 316,940,010 that day and 317,358,518 on 2026-11-03, against 635,675,701 on 2026-04-13 -- to within 0.06% of Honeywell Aerospace's own count (HONA: 316,826,559 on 2026-06-29, weighted 316,952,725 and 317,031,836 after), and HON's weighted count of 2026-07-14 is HONA's first count exactly. No split event under HON but 2025-10-30's price-only one, no raw close moving more than 15% in a day, and on HONA's first day HON opened higher: nothing halved Honeywell's shares. The last sample before the contamination, 2026-04-13, is carried.
    SKYD   BBG01VS5NK99 2026-10-06..2026-11-03: no count -- 21 days in its spans, 21 member-days, 0 sample(s) inside not read; midoffice08, decision 3 (Jeff, 2026-11-10)
      After Skydance took Warner Bros. Discovery, trading as SKYD from 2026-10-06, its records carry no share-class count, and the pre-merger count that would carry, 1,090,445,692, is 4.6 times under the weighted count of 2026-11-03, 5,042,178,087. No count is better than a wrong one.
  member-days without a count: 234 (midoffice07, before the search: 696) -- 176 inside the spans, before a first count; 21 in an override's range of no count; 37 after a member's last trade, outside its spans, which no price or count reaches
    AMTM      5: 2024-09-30..2024-10-04
    CEG       1: 2022-02-02..2022-02-02
    FDXF    109: 2026-06-01..2026-11-03
    GEHC      1: 2023-01-04..2023-01-04
    GEV       4: 2024-04-02..2024-04-05
    Q        13: 2025-11-03..2025-11-19
    SKYD     23: 2025-08-07..2026-11-03
    SOLS     13: 2025-10-30..2025-11-17
    SOLV      1: 2024-04-01..2024-04-01
    VLTO      2: 2023-10-02..2023-10-03
    VYLR     24: 2026-10-01..2026-11-03
    WBD       1: 2022-04-11..2022-04-11
  counts that move beyond 10% from one day to the next, no split the gate applies there: 86 -- a standing check; the gate tests counts across split events alone
    AJG    2025-01-07: 219,100,000 to 249,800,000, 1.1401
    AMCR   2025-10-09: 1,445,343,212 to 2,305,359,941, 1.5950
    AMD    2022-04-04: 1,207,610,000 to 1,627,370,000, 1.3476
    APA    2024-04-08: 306,720,000 to 370,890,000, 1.2092
    APO    2022-04-04: 246,580,000 to 570,920,000, 2.3154
    APP    2022-04-04: 226,410,000 to 297,160,000, 1.3125
    APTV   2025-01-07: 265,760,000 to 235,040,000, 0.8844
    AVGO   2024-01-05: 412,740,000 to 468,140,000, 1.1342
    BA     2025-01-07: 616,170,000 to 748,180,000, 1.2142
    BE     2023-10-05: 192,870,000 to 223,860,000, 1.1607
    BE     2026-04-13: 236,510,755 to 282,410,088, 1.1941
    BG     2025-11-05: 134,404,972 to 200,062,018, 1.4885
    BRO    2025-10-09: 286,680,034 to 329,842,812, 1.1506
    CCL    2022-10-04: 989,700,000 to 1,096,760,000, 1.1082
    CCL    2026-08-07: 1,239,000,212 to 1,369,649,119, 1.1054
    CEG    2026-01-27: 312,405,579 to 362,355,476, 1.1599
    CEG    2026-04-28: 362,355,476 to 313,309,685, 0.8646
    CEG    2026-07-29: 313,309,685 to 359,102,017, 1.1462
    CFG    2022-07-06: 422,140,000 to 495,450,000, 1.1737
    CHTR   2022-07-06: 191,490,000 to 167,860,000, 0.8766
    CHTR   2026-07-14: 141,178,369 to 122,984,536, 0.8711
    COF    2025-10-09: 383,137,556 to 639,517,306, 1.6692
    COHR   2022-09-07: 106,460,000 to 130,870,000, 1.2293
    COHR   2026-03-17: 157,153,611 to 187,481,852, 1.1930
    COIN   2022-04-04: 155,240,000 to 171,170,000, 1.1026
    COP    2025-01-07: 1,161,250,000 to 1,293,560,000, 1.1139
    CRM    2026-07-14: 923,000,000 to 819,000,000, 0.8873
    CVNA   2022-07-06: 90,130,000 to 105,740,000, 1.1732
    CVX    2026-01-09: 1,727,990,376 to 2,013,521,597, 1.1652
    DASH   2022-07-06: 317,160,000 to 367,210,000, 1.1578
    DOC    2024-03-04: 547,170,000 to 708,200,000, 1.2943
    DVN    2026-07-14: 621,440,299 to 1,153,403,107, 1.8560
    ECHO   2024-01-05: 36,160,000 to 140,060,000, 3.8733
    ECHO   2025-04-09: 140,490,000 to 155,200,000, 1.1047
    EG     2023-10-05: 39,280,000 to 43,400,000, 1.1049
    EQT    2023-10-05: 361,640,000 to 411,260,000, 1.1372
    EQT    2024-10-07: 441,590,000 to 594,700,000, 1.3467
    EXE    2023-01-04: 120,850,000 to 133,970,000, 1.1086
    EXE    2024-10-07: 131,280,000 to 226,900,000, 1.7284
    EXR    2023-10-05: 135,050,000 to 211,280,000, 1.5645
    FANG   2024-10-07: 178,340,000 to 294,740,000, 1.6527
    FITB   2026-04-13: 661,044,771 to 905,563,498, 1.3699
    GEN    2022-10-04: 580,060,000 to 665,570,000, 1.1474
    GM     2024-04-08: 1,369,480,000 to 1,154,430,000, 0.8430
    GPN    2026-04-13: 236,743,716 to 275,243,013, 1.1626
    HBAN   2026-04-13: 1,459,390,757 to 2,037,119,660, 1.3959
    INCY   2024-10-07: 224,860,000 to 192,600,000, 0.8565
    IP     2025-04-09: 347,410,000 to 527,880,000, 1.5195
    KEY    2025-04-09: 991,280,000 to 1,095,720,000, 1.1054
    KKR    2022-10-04: 592,610,000 to 859,830,000, 1.4509
    LITE   2026-10-12: 77,800,000 to 90,582,856, 1.1643
    LNC    2025-10-09: 170,692,628 to 189,580,412, 1.1107
    MBC    2026-09-18: 127,984,590 to 203,490,490, 1.5900
    MHK    2026-10-12: 60,953,145 to 67,738,145, 1.1113
    MTB    2022-07-06: 129,060,000 to 179,420,000, 1.3902
    NDAQ   2024-01-05: 491,320,000 to 576,970,000, 1.1743
    NEM    2024-01-05: 794,800,000 to 1,152,490,000, 1.4500
    NRG    2026-04-13: 191,639,408 to 214,556,589, 1.1196
    O      2024-04-08: 723,920,000 to 861,120,000, 1.1895
    OKE    2023-10-05: 447,450,000 to 582,480,000, 1.3018
    OMC    2026-01-09: 193,722,889 to 314,603,664, 1.6240
    PANW   2026-04-13: 697,000,000 to 816,000,000, 1.1707
    PAYC   2026-04-13: 56,269,005 to 47,666,129, 0.8471
    PCG    2022-04-04: 1,985,370,000 to 2,465,130,000, 1.2416
    PCG    2022-07-06: 2,465,130,000 to 1,987,480,000, 0.8062
    PCG    2025-07-11: 2,193,580,000 to 2,675,428,603, 1.2197
    PLD    2022-10-04: 740,340,000 to 923,250,000, 1.2471
    RDDT   2024-06-20: 36,880,000 to 44,440,000, 1.2050
    RDDT   2024-09-19: 44,440,000 to 94,960,000, 2.1368
    RDDT   2024-12-18: 94,960,000 to 119,740,000, 1.2610
    SNPS   2025-10-09: 155,160,938 to 185,748,673, 1.1971
    SPGI   2022-04-04: 241,000,000 to 347,030,000, 1.4400
    STZ    2023-01-04: 161,220,000 to 184,470,000, 1.1442
    SW     2024-10-04: 261,100,000 to 519,960,000, 1.9914
    TKO    2023-12-11: 52,060,000 to 83,370,000, 1.6014
    TPR    2025-04-09: 233,040,000 to 207,020,000, 0.8883
    TSLA   2026-04-13: 3,325,819,167 to 3,752,431,984, 1.1283
    TTWO   2022-07-06: 115,420,000 to 161,980,000, 1.4034
    VEEV   2024-01-05: 146,160,000 to 161,020,000, 1.1017
    VICI   2022-04-04: 628,950,000 to 748,410,000, 1.1899
    VICI   2022-07-06: 748,410,000 to 963,000,000, 1.2867
    WAT    2026-04-13: 59,547,678 to 98,165,491, 1.6485
    WHR    2026-07-14: 56,540,150 to 64,823,121, 1.1465
    XOM    2024-07-09: 3,967,840,000 to 4,485,930,000, 1.1306
    XYL    2023-07-07: 180,280,000 to 239,400,000, 1.3279
    XYZ    2022-04-04: 399,180,000 to 518,360,000, 1.2986

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, read through the table's 51 renames
  the constituent list predates the change of 2026-10-30 -- OKTA added, KVUE removed -- and is read with it applied

Guards
  ok  each (ticker, date) price feeds one column: 685,133 checked, 0 failure(s)
  ok  no column takes two tickers on one date: its spans do not overlap: 628 checked, 0 failure(s)
  ok  every member-day is a column's, on a day inside its spans or one the table maps outside them: 610,842 checked, 0 failure(s)
  ok  every bar under a span's ticker inside the span is its column's, and only those: 685,089 checked, 0 failure(s)
  ok  every member-day of the walk through the renames is a column's member-day: 610,842 checked, 0 failure(s)
  ok  the panel's member-days are the walk's, no more: 610,842 checked, 0 failure(s)
  ok  no day on or after a split carries a pre-split count, and the split's day carries the count before it times the ratio (25,260 of the days checked member-days): 25,846 checked, 0 failure(s)

The weighted-count check: a record's weighted share count against the count the panel carries that day, flagged beyond 10%
  carried from an earlier record: 47 met, 8 flagged; ratio median 1.0001, p1 0.4985, p99 2.2556
  the day's own record: 11,846 met, 754 flagged; ratio median 1.0000, p1 0.9825, p99 2.0918
  every carried one, the reorganisations' and the splits' among them:
    ACGL   ACGL   2026-07-14 weighted 349,389,588 against 356,272,841: 0.9807
    ACGL   ACGL   2026-10-12 weighted 341,229,126 against 356,272,841: 0.9578
    ACGL   ACGL   2026-11-03 weighted 326,400,000 against 356,272,841: 0.9162
    APH    APH    2026-10-12 weighted 2,465,966,914 against 2,460,468,890: 1.0022
    APH    APH    2026-11-03 weighted 2,469,261,188 against 2,460,468,890: 1.0036
    APTV   APTV   2026-11-03 weighted 207,633,331 against 207,633,331: 1.0000
    BKNG   BKNG   2026-04-13 weighted 791,833,650 against 805,845,375: 0.9826
    CCL    CCL    2026-05-07 weighted 1,385,139,577 against 1,239,000,212: 1.1179  FLAGGED
    CRWD   CRWD   2026-07-14 weighted 1,018,259,280 against 1,014,456,360: 1.0037
    CVNA   CVNA   2026-07-14 weighted 716,288,385 against 714,968,845: 1.0018
    DGX    DGX    2025-04-09 weighted 110,978,069 against 111,620,000: 0.9942
    EG     EG     2025-04-09 weighted 42,934,097 against 42,980,000: 0.9989
    FAST   FAST   2025-07-11 weighted 1,147,241,276 against 1,146,900,000: 1.0003
    GRMN   GRMN   2025-07-11 weighted 192,541,456 against 192,400,000: 1.0007
    GRMN   GRMN   2025-10-09 weighted 192,493,945 against 192,400,000: 1.0005
    GRMN   GRMN   2026-01-09 weighted 192,334,806 against 192,400,000: 0.9997
    GRMN   GRMN   2026-04-13 weighted 192,480,830 against 192,400,000: 1.0004
    HON    HON    2026-07-14 weighted 316,826,559 against 635,675,701: 0.4984  FLAGGED
    HON    HON    2026-10-12 weighted 316,940,010 against 635,675,701: 0.4986  FLAGGED
    HON    HON    2026-11-03 weighted 316,940,010 against 635,675,701: 0.4986  FLAGGED
    HONA   HONA   2026-09-15 weighted 316,952,725 against 316,826,559: 1.0004
    HONA   HONA   2026-11-03 weighted 317,031,836 against 316,826,559: 1.0006
    IBKR   IBKR   2025-07-11 weighted 435,835,720 against 435,720,000: 1.0003
    KLAC   KLAC   2026-07-14 weighted 1,306,275,210 against 1,310,766,110: 0.9966
    MNST   MNST   2026-10-12 weighted 1,959,051,764 against 1,956,016,220: 1.0016
    MNST   MNST   2026-11-03 weighted 1,959,051,764 against 1,956,016,220: 1.0016
    NOW    NOW    2026-01-09 weighted 1,055,142,985 against 1,040,000,000: 1.0146
    NWS    NWS    2024-07-09 weighted 569,013,132 against 191,100,000: 2.9776  FLAGGED
    OKE    OKE    2026-09-11 weighted 630,369,667 against 630,369,667: 1.0000
    OKE    OKE    2026-11-03 weighted 630,450,437 against 630,369,667: 1.0001
    ORLY   ORLY   2025-07-11 weighted 854,841,825 against 858,600,000: 0.9956
    PCG    PCG    2026-01-09 weighted 2,197,910,425 against 2,675,580,992: 0.8215  FLAGGED
    Q      Q      2026-10-28 weighted 209,208,543 against 209,339,207: 0.9994
    Q      Q      2026-11-03 weighted 209,208,543 against 209,339,207: 0.9994
    RDDT   RDDT   2025-12-22 weighted 189,495,765 against 134,585,853: 1.4080  FLAGGED
    SBUX   SBUX   2026-07-14 weighted 1,139,700,000 against 1,139,300,000: 1.0004
    SKYD   PSKY   2026-10-05 weighted 1,124,520,841 against 1,090,445,692: 1.0312
    SMCI   SMCI   2024-10-01 weighted 585,565,270 against 585,600,000: 0.9999
    SNDK   SNDK   2026-11-03 weighted 142,289,365 against 148,089,758: 0.9608
    SOLS   SOLS   2026-10-21 weighted 158,847,369 against 158,804,273: 1.0003
    SOLS   SOLS   2026-11-03 weighted 158,847,369 against 158,804,273: 1.0003
    TPL    TPL    2026-01-09 weighted 68,938,230 against 68,961,978: 0.9997
    WSM    WSM    2024-07-09 weighted 128,616,166 against 128,220,000: 1.0031
    XOM    XOM    2026-07-02 weighted 4,144,947,162 against 4,144,947,162: 1.0000
    XOM    XOM    2026-10-01 weighted 4,111,911,960 against 4,144,947,162: 0.9920
    XOM    XOM    2026-11-03 weighted 4,079,000,000 against 4,144,947,162: 0.9841
    XYZ    XYZ    2026-01-09 weighted 607,665,000 against 549,574,000: 1.1057  FLAGGED
  the day's own record flagged, by security -- 71 securities; where every record is, a class count beside a company-wide weighted one:
    ECHO    22 of  22 records, ratio 1.8237..2.3492, from 2022-01-03 to 2026-11-03
    ABNB    21 of  21 records, ratio 1.4053..1.8320, from 2022-01-03 to 2026-11-03
    AOS     21 of  21 records, ratio 1.1895..1.2384, from 2022-01-03 to 2026-11-03
    BF.B    21 of  21 records, ratio 1.5457..1.5803, from 2022-01-03 to 2026-11-03
    BIO     21 of  21 records, ratio 1.1841..1.2335, from 2022-01-03 to 2026-11-03
    BRK.B   21 of  21 records, ratio 1.5204..1.7161, from 2022-01-03 to 2026-11-03
    COIN    21 of  21 records, ratio 1.1818..1.3863, from 2022-01-03 to 2026-11-03
    DELL    21 of  21 records, ratio 1.9454..3.0178, from 2022-01-03 to 2026-11-03
    EL      21 of  21 records, ratio 1.4621..1.5534, from 2022-01-03 to 2026-11-03
    ERIE    21 of  21 records, ratio 1.1321..1.1321, from 2022-01-03 to 2026-11-03
    FOX     21 of  21 records, ratio 1.8845..2.2853, from 2022-01-03 to 2026-11-03
    FOXA    21 of  21 records, ratio 1.7780..2.1306, from 2022-01-03 to 2026-11-03
    GOOG    21 of  21 records, ratio 2.0890..2.2365, from 2022-01-03 to 2026-11-03
    GOOGL   21 of  21 records, ratio 2.0741..2.2066, from 2022-01-03 to 2026-11-03
    HOOD    21 of  21 records, ratio 1.1347..1.1786, from 2022-01-03 to 2026-11-03
    HSY     21 of  21 records, ratio 1.3643..1.4165, from 2022-01-03 to 2026-11-03
    LEN     21 of  21 records, ratio 1.1035..1.1484, from 2022-01-03 to 2026-11-03
    META    21 of  22 records, ratio 1.1553..1.1800, from 2022-01-03 to 2026-11-03
    NKE     21 of  21 records, ratio 1.2309..1.2529, from 2022-01-03 to 2026-11-03
    NWSA    21 of  21 records, ratio 1.4841..1.5170, from 2022-01-03 to 2026-11-03
    RL      21 of  21 records, ratio 1.5104..1.6374, from 2022-01-03 to 2026-11-03
    TSN     21 of  21 records, ratio 1.2317..1.2484, from 2022-01-03 to 2026-11-03
    UA      21 of  21 records, ratio 1.8811..2.1021, from 2022-01-03 to 2026-11-03
    UAA     21 of  21 records, ratio 2.2448..2.5247, from 2022-01-03 to 2026-11-03
    UPS     21 of  21 records, ratio 1.1354..1.1946, from 2022-01-03 to 2026-11-03
    WDAY    21 of  21 records, ratio 1.2240..1.2953, from 2022-01-03 to 2026-11-03
    XYZ     21 of  21 records, ratio 1.1073..1.1556, from 2022-01-03 to 2026-11-03
    NWS     20 of  20 records, ratio 2.9362..2.9929, from 2022-01-03 to 2026-11-03
    UHS     20 of  21 records, ratio 1.1074..1.1402, from 2022-04-04 to 2026-11-03
    V       20 of  21 records, ratio 1.1017..1.3037, from 2022-01-03 to 2026-10-12
    CCL     19 of  21 records, ratio 1.1179..1.1956, from 2022-01-03 to 2026-05-06
    APP     13 of  21 records, ratio 1.1236..1.6528, from 2022-01-03 to 2025-01-07
    PCG     13 of  20 records, ratio 0.8214..1.2235, from 2022-10-04 to 2026-11-03
    RDDT    11 of  11 records, ratio 1.3141..4.3111, from 2024-03-21 to 2026-11-03
    DISH     9 of   9 records, ratio 1.8056..1.8212, from 2022-01-03 to 2023-12-29
    TTD      6 of  21 records, ratio 1.1001..1.1022, from 2022-01-03 to 2026-11-03
    VEEV     6 of  21 records, ratio 1.1001..1.1103, from 2022-01-03 to 2023-04-05
    STZ      4 of  21 records, ratio 1.1412..1.1456, from 2022-01-03 to 2022-10-04
    DDOG     3 of  21 records, ratio 1.1127..1.2131, from 2022-01-03 to 2022-07-06
    DISCA    3 of   3 records, ratio 2.9627..2.9900, from 2022-01-03 to 2022-04-08
    DISCK    3 of   3 records, ratio 1.5325..1.5393, from 2022-01-03 to 2022-04-08
    BE       2 of  21 records, ratio 1.1008..1.1877, from 2022-01-03 to 2022-10-04
    CPT      2 of  21 records, ratio 1.1511..1.1511, from 2026-10-12 to 2026-11-03
    EG       2 of  21 records, ratio 1.1053..1.1053, from 2023-07-07 to 2023-07-10
    FMC      2 of  21 records, ratio 1.2421..1.2422, from 2026-10-12 to 2026-11-03
    MLM      2 of  21 records, ratio 1.1824..1.1824, from 2026-10-12 to 2026-11-03
    VMRK     2 of  22 records, ratio 2.0637..2.0637, from 2026-10-12 to 2026-11-03
    AAL      1 of  21 records, ratio 2.0533..2.0533, from 2023-04-05 to 2023-04-05
    AMCR     1 of  21 records, ratio 1.5853..1.5853, from 2025-07-11 to 2025-07-11
    APTV     1 of  20 records, ratio 0.8841..0.8841, from 2024-10-07 to 2024-10-07
    AVB      1 of  20 records, ratio 2.6391..2.6391, from 2026-07-14 to 2026-07-14
    BG       1 of  22 records, ratio 1.4885..1.4885, from 2025-08-07 to 2025-08-07
    BR       1 of  21 records, ratio 1.3129..1.3129, from 2022-10-04 to 2022-10-04
    BRO      1 of  21 records, ratio 1.1505..1.1505, from 2025-07-11 to 2025-07-11
    BX       1 of  21 records, ratio 1.6443..1.6443, from 2026-07-14 to 2026-07-14
    CEG      1 of  21 records, ratio 1.1563..1.1563, from 2026-04-28 to 2026-04-28
    CHTR     1 of  21 records, ratio 0.8828..0.8828, from 2026-04-13 to 2026-04-13
    COF      1 of  21 records, ratio 1.6695..1.6695, from 2025-07-11 to 2025-07-11
    CRWD     1 of  20 records, ratio 1.1015..1.1015, from 2022-01-03 to 2022-01-03
    CVNA     1 of  20 records, ratio 1.5266..1.5266, from 2026-11-03 to 2026-11-03
    CVX      1 of  21 records, ratio 1.1660..1.1660, from 2025-10-09 to 2025-10-09
    DASH     1 of  21 records, ratio 1.1010..1.1010, from 2022-01-03 to 2022-01-03
    INCY     1 of  21 records, ratio 0.8761..0.8761, from 2024-07-09 to 2024-07-09
    JNJ      1 of  21 records, ratio 1.1980..1.1980, from 2023-04-05 to 2023-04-05
    KEY      1 of  21 records, ratio 1.1163..1.1163, from 2025-01-07 to 2025-01-07
    MHK      1 of  21 records, ratio 0.8582..0.8582, from 2026-11-03 to 2026-11-03
    SKYD     1 of   5 records, ratio 0.6333..0.6333, from 2025-08-11 to 2025-08-11
    SMCI     1 of  22 records, ratio 9.9994..9.9994, from 2024-09-30 to 2024-09-30
    SWKS     1 of  21 records, ratio 1.5628..1.5628, from 2026-11-03 to 2026-11-03
    WBD      1 of  19 records, ratio 0.2275..0.2275, from 2022-04-12 to 2022-04-12
    WHR      1 of  21 records, ratio 1.1397..1.1397, from 2026-04-13 to 2026-04-13

Industries
  each security's SIC code from its own latest record with one: 587 of 592; none for DISCA, DISCK, FRCB, SBNY, VYLR

Wrote
  data\external\polygon\sp500_securities_2022-01-03_2026-11-03_repaired.npz
  data\external\polygon\sp500_securities_2022-01-03_2026-11-03_repaired.json
```

### overwrite_refusal.txt -- the build run again over the repaired panel

```text
  data\external\polygon\sp500_securities_2022-01-03_2026-11-03_repaired.npz exists: give another --suffix or --out, or --overwrite
```

### member_days_changed.txt -- the member-days each ruling changes, each panel against the one before

```text
ruling 1, against midoffice08: 117 security-days change their count, 117 of them member-days; every other array the same: True
  AMTM   2024-10-01..2024-12-20: 58 days, 58 member-days (4 lose a count, 0 gain one); was 153,280,000; now 243,300,000
  GEV    2024-04-03..2024-06-26: 59 days, 59 member-days (3 lose a count, 0 gain one); was 1,088,330,000; now 272,080,000
ruling 2, against ruling 1: 441 security-days change their count, 378 of them member-days; every other array the same: True
  DGX    2025-04-09..2025-07-10: 63 days, 63 member-days (0 lose a count, 0 gain one); was 11,610,000; now 111,620,000
  EG     2025-04-09..2025-07-10: 63 days, 63 member-days (0 lose a count, 0 gain one); was 47,790,000; now 42,980,000
  NWS    2024-07-09..2024-10-04: 63 days, 63 member-days (0 lose a count, 0 gain one); was 162,170,000; now 191,100,000
  PCG    2026-01-09..2026-04-10: 63 days, 63 member-days (0 lose a count, 0 gain one); was 2,197,910,425; now 2,675,580,992
  RDDT   2025-12-22..2026-03-24: 63 days, 0 member-days (0 lose a count, 0 gain one); was 51,670,930; now 134,585,853
  SBUX   2026-07-14..2026-10-09: 63 days, 63 member-days (0 lose a count, 0 gain one); was 139,700,000; now 1,139,300,000
  XYZ    2026-01-09..2026-04-10: 63 days, 63 member-days (0 lose a count, 0 gain one); was 59,998,000; now 549,574,000
ruling 3, against ruling 2: 38 security-days change their count, 38 of them member-days; every other array the same: True
  HON    2026-10-12..2026-11-03: 17 days, 17 member-days (0 lose a count, 0 gain one); was 316,940,010, 317,358,518; now 635,675,701
  SKYD   2026-10-06..2026-11-03: 21 days, 21 member-days (21 lose a count, 0 gain one); was 1,090,445,692; now none
```

### capture logs -- the runs captured in-process

```text
D: status 0, mean R2 0.293189, 938 periods, 67s
R1: status 0, mean R2 0.293183, 938 periods, 54s
R2: status 0, mean R2 0.293165, 938 periods, 67s
R3: status 0, mean R2 0.293204, 938 periods, 54s
N: status 0, mean R2 0.293204, 938 periods, 55s
A: status 0, mean R2 0.298076, 938 periods, 76s
AR: status 0, mean R2 0.298076, 938 periods, 68s
NR: status 0, mean R2 0.293204, 938 periods, 66s
NC: status 0, mean R2 0.293204, 938 periods, 67s
```

### compare_counts.txt -- the model on each ruling's panel against the one before, and all three against run D

```text
====================================================================================================
Step ruling 1: confirmed first counts: D -> R1
  D: data/external/polygon/sp500_securities_2022-01-03_2026-11-03.npz --industries --membership --returns total --volumes traded
  R1: claude/scratch/midoffice09/p1/sp500_securities_2022-01-03_2026-11-03_r1.npz --industries --membership --returns total --volumes traded
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    D: min 499 median 502 max 503 first 502 last 500
    R1: min 499 median 502 max 503 first 502 last 500
    R1 less D, per period: min +0 median +0 max +0
  mean cross-sectional R-squared: D 0.293189, R1 0.293183, difference -0.000006
  factor                    vol D   vol R1    mean D   mean R1   d mean   noise D  noise R1
  Construction             19.43%   19.43%   +24.77%   +24.79%   +0.01%     38.1%     38.1%
  Finance & Real Estate    15.35%   15.35%   +18.11%   +18.12%   +0.02%     30.9%     30.9%
  Manufacturing            10.20%   10.20%   +20.90%   +20.89%   -0.00%     23.2%     23.2%
  Mining                   18.63%   18.63%   +14.96%   +14.94%   -0.02%     18.5%     18.5%
  Retail                   13.14%   13.14%   +14.73%   +14.72%   -0.01%     10.7%     10.7%
  SIC20                    11.98%   11.98%   +11.04%   +11.02%   -0.02%     33.5%     33.5%
  SIC28                    13.14%   13.14%   +16.64%   +16.64%   -0.00%     10.2%     10.2%
  SIC35                    18.81%   18.81%   +29.69%   +29.70%   +0.01%      5.2%      5.2%
  SIC36                    21.13%   21.13%   +32.58%   +32.56%   -0.02%      4.5%      4.5%
  SIC37                    14.24%   14.24%   +15.43%   +15.41%   -0.02%     19.1%     19.1%
  SIC38                    14.70%   14.70%   +15.82%   +15.83%   +0.01%     10.6%     10.6%
  SIC49                    15.82%   15.82%   +24.01%   +24.00%   -0.00%     11.3%     11.3%
  SIC60                    21.86%   21.86%   +21.96%   +21.96%   -0.00%      7.4%      7.4%
  SIC62                    18.75%   18.75%   +18.62%   +18.63%   +0.01%     11.5%     11.5%
  SIC63                    14.57%   14.57%   +19.16%   +19.17%   +0.01%     14.5%     14.5%
  SIC67                    15.02%   15.02%   +18.00%   +18.01%   +0.01%     16.7%     16.8%
  SIC73                    18.15%   18.15%   +15.18%   +15.18%   +0.00%      3.0%      3.0%
  Services                 12.74%   12.74%   +16.18%   +16.18%   -0.00%     27.9%     27.9%
  Transport & Utilities    13.52%   13.52%   +15.39%   +15.38%   -0.01%     16.3%     16.3%
  Wholesale                13.97%   13.97%   +20.70%   +20.69%   -0.01%     46.2%     46.2%
  Volatility                9.37%    9.37%    +3.08%    +3.06%   -0.02%      3.5%      3.5%
  Momentum                  7.21%    7.21%    +4.23%    +4.23%   +0.00%      3.6%      3.6%
  Size                      3.72%    3.72%    +3.14%    +3.14%   +0.00%      7.7%      7.7%
  Size Nonlinearity         2.31%    2.31%    -0.51%    -0.50%   +0.00%     47.0%     47.0%
  Trading Activity          6.21%    6.21%    +2.48%    +2.51%   +0.02%      7.6%      7.6%
  residual spread (1.4826 x MAD; sd): D 0.1330 (0.3405); R1 0.1332 (0.3405)
    largest shares of the sd's square, D: FRCB 67.5%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of the sd's square, R1: FRCB 67.5%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of V^2, D: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    largest shares of V^2, R1: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: D p10 15.59% median 24.80% p90 40.78% (552); R1 p10 15.59% median 24.80% p90 40.77% (552)
  specific risk, the 552 in both: D p10 15.59% median 24.80% p90 40.78% (552); R1 p10 15.59% median 24.80% p90 40.77% (552)
    of those in both, the largest rises: GEV +0.04%, SBNY +0.01%, PLTR +0.01%, LUMN +0.01%, UAA +0.01%, PENN +0.00%, SKYD +0.00%, RDDT +0.00%
    the largest falls: Q -0.01%, AVGO -0.01%, RHI -0.01%, NWL -0.01%, FLEX -0.01%, ALK -0.01%, VST -0.01%, MRVL -0.01%
  fallback forecasts: D 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
                      R1 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
    in R1 not D: none; in D not R1: none
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    D -3.139 -1.810 risk 11.82%; R1 -3.139 -1.810 risk 11.82%
    largest 50, equally weighted     D +0.170 +0.052 risk 0.62%; R1 +0.170 +0.052 risk 0.62%
    the index, cap-weighted          D +0.001 -0.001 risk 0.01%; R1 +0.001 -0.001 risk 0.01%
  regression universe: D 567, R1 567; in D not R1: none
    in R1 not D (0): 
====================================================================================================
Step ruling 2: isolated errors repaired: R1 -> R2
  R1: claude/scratch/midoffice09/p1/sp500_securities_2022-01-03_2026-11-03_r1.npz --industries --membership --returns total --volumes traded
  R2: claude/scratch/midoffice09/p2/sp500_securities_2022-01-03_2026-11-03_r2.npz --industries --membership --returns total --volumes traded
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    R1: min 499 median 502 max 503 first 502 last 500
    R2: min 499 median 502 max 503 first 502 last 500
    R2 less R1, per period: min +0 median +0 max +0
  mean cross-sectional R-squared: R1 0.293183, R2 0.293165, difference -0.000017
  factor                   vol R1   vol R2   mean R1   mean R2   d mean  noise R1  noise R2
  Construction             19.43%   19.41%   +24.79%   +24.76%   -0.02%     38.1%     38.2%
  Finance & Real Estate    15.35%   15.34%   +18.12%   +18.11%   -0.02%     30.9%     31.0%
  Manufacturing            10.20%   10.20%   +20.89%   +20.88%   -0.01%     23.2%     23.2%
  Mining                   18.63%   18.60%   +14.94%   +14.93%   -0.01%     18.5%     18.5%
  Retail                   13.14%   13.11%   +14.72%   +14.62%   -0.10%     10.7%     10.7%
  SIC20                    11.98%   11.95%   +11.02%   +11.04%   +0.01%     33.5%     33.6%
  SIC28                    13.14%   13.14%   +16.64%   +16.64%   -0.00%     10.2%     10.2%
  SIC35                    18.81%   18.81%   +29.70%   +29.69%   -0.01%      5.2%      5.2%
  SIC36                    21.13%   21.12%   +32.56%   +32.55%   -0.00%      4.5%      4.5%
  SIC37                    14.24%   14.25%   +15.41%   +15.42%   +0.01%     19.1%     19.1%
  SIC38                    14.70%   14.69%   +15.83%   +15.82%   -0.01%     10.6%     10.6%
  SIC49                    15.82%   15.80%   +24.00%   +24.02%   +0.01%     11.3%     11.3%
  SIC60                    21.86%   21.85%   +21.96%   +21.97%   +0.01%      7.4%      7.4%
  SIC62                    18.75%   18.73%   +18.63%   +18.62%   -0.01%     11.5%     11.5%
  SIC63                    14.57%   14.56%   +19.17%   +19.16%   -0.01%     14.5%     14.5%
  SIC67                    15.02%   15.00%   +18.01%   +18.00%   -0.01%     16.8%     16.8%
  SIC73                    18.15%   18.18%   +15.18%   +15.21%   +0.03%      3.0%      2.9%
  Services                 12.74%   12.72%   +16.18%   +16.11%   -0.06%     27.9%     28.0%
  Transport & Utilities    13.52%   13.52%   +15.38%   +15.38%   -0.00%     16.3%     16.3%
  Wholesale                13.97%   13.97%   +20.69%   +20.69%   +0.00%     46.2%     46.2%
  Volatility                9.37%    9.39%    +3.06%    +3.08%   +0.02%      3.5%      3.5%
  Momentum                  7.21%    7.22%    +4.23%    +4.22%   -0.01%      3.6%      3.6%
  Size                      3.72%    3.72%    +3.14%    +3.14%   -0.00%      7.7%      7.7%
  Size Nonlinearity         2.31%    2.33%    -0.50%    -0.51%   -0.01%     47.0%     46.2%
  Trading Activity          6.21%    6.26%    +2.51%    +2.48%   -0.02%      7.6%      7.5%
  residual spread (1.4826 x MAD; sd): R1 0.1332 (0.3405); R2 0.1338 (0.3404)
    largest shares of the sd's square, R1: FRCB 67.5%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of the sd's square, R2: FRCB 67.5%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of V^2, R1: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    largest shares of V^2, R2: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: R1 p10 15.59% median 24.80% p90 40.77% (552); R2 p10 15.58% median 24.80% p90 40.78% (552)
  specific risk, the 552 in both: R1 p10 15.59% median 24.80% p90 40.77% (552); R2 p10 15.58% median 24.80% p90 40.78% (552)
    of those in both, the largest rises: OKTA +0.12%, GAP +0.12%, FBIN +0.12%, PVH +0.12%, SOLS +0.10%, MBC +0.10%, SBNY +0.09%, PENN +0.09%
    the largest falls: FIX -0.02%, YUM -0.03%, FLEX -0.03%, ALK -0.03%, Q -0.04%, MRVL -0.05%, DGX -0.11%, SBUX -0.30%
  fallback forecasts: R1 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
                      R2 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
    in R2 not R1: none; in R1 not R2: none
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    R1 -3.139 -1.810 risk 11.82%; R2 -3.139 -1.810 risk 11.88%
    largest 50, equally weighted     R1 +0.170 +0.052 risk 0.62%; R2 +0.170 +0.052 risk 0.63%
    the index, cap-weighted          R1 +0.001 -0.001 risk 0.01%; R2 +0.001 -0.001 risk 0.01%
  regression universe: R1 567, R2 567; in R1 not R2: none
    in R2 not R1 (0): 
====================================================================================================
Step ruling 3: the overrides: R2 -> R3
  R2: claude/scratch/midoffice09/p2/sp500_securities_2022-01-03_2026-11-03_r2.npz --industries --membership --returns total --volumes traded
  R3: data/external/polygon/sp500_securities_2022-01-03_2026-11-03_repaired.npz --industries --membership --returns total --volumes traded
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    R2: min 499 median 502 max 503 first 502 last 500
    R3: min 499 median 502 max 503 first 502 last 499
    R3 less R2, per period: min -1 median +0 max +0
  mean cross-sectional R-squared: R2 0.293165, R3 0.293204, difference +0.000038
  factor                   vol R2   vol R3   mean R2   mean R3   d mean  noise R2  noise R3
  Construction             19.41%   19.42%   +24.76%   +24.76%   -0.00%     38.2%     38.2%
  Finance & Real Estate    15.34%   15.34%   +18.11%   +18.11%   +0.00%     31.0%     31.0%
  Manufacturing            10.20%   10.20%   +20.88%   +20.88%   +0.00%     23.2%     23.2%
  Mining                   18.60%   18.61%   +14.93%   +14.94%   +0.00%     18.5%     18.5%
  Retail                   13.11%   13.11%   +14.62%   +14.63%   +0.00%     10.7%     10.7%
  SIC20                    11.95%   11.96%   +11.04%   +11.03%   -0.00%     33.6%     33.6%
  SIC28                    13.14%   13.14%   +16.64%   +16.64%   +0.00%     10.2%     10.2%
  SIC35                    18.81%   18.81%   +29.69%   +29.69%   +0.00%      5.2%      5.2%
  SIC36                    21.12%   21.13%   +32.55%   +32.55%   -0.00%      4.5%      4.5%
  SIC37                    14.25%   14.22%   +15.42%   +15.42%   -0.00%     19.1%     19.1%
  SIC38                    14.69%   14.69%   +15.82%   +15.82%   -0.00%     10.6%     10.6%
  SIC49                    15.80%   15.80%   +24.02%   +24.02%   +0.00%     11.3%     11.3%
  SIC60                    21.85%   21.85%   +21.97%   +21.97%   +0.00%      7.4%      7.4%
  SIC62                    18.73%   18.73%   +18.62%   +18.62%   +0.01%     11.5%     11.5%
  SIC63                    14.56%   14.57%   +19.16%   +19.16%   +0.00%     14.5%     14.5%
  SIC67                    15.00%   15.01%   +18.00%   +18.00%   -0.00%     16.8%     16.8%
  SIC73                    18.18%   18.18%   +15.21%   +15.21%   -0.00%      2.9%      2.9%
  Services                 12.72%   12.72%   +16.11%   +16.11%   -0.00%     28.0%     28.0%
  Transport & Utilities    13.52%   13.43%   +15.38%   +15.46%   +0.08%     16.3%     16.5%
  Wholesale                13.97%   13.98%   +20.69%   +20.69%   +0.00%     46.2%     46.2%
  Volatility                9.39%    9.38%    +3.08%    +3.09%   +0.00%      3.5%      3.5%
  Momentum                  7.22%    7.22%    +4.22%    +4.23%   +0.00%      3.6%      3.6%
  Size                      3.72%    3.72%    +3.14%    +3.13%   -0.00%      7.7%      7.7%
  Size Nonlinearity         2.33%    2.32%    -0.51%    -0.53%   -0.01%     46.2%     46.6%
  Trading Activity          6.26%    6.25%    +2.48%    +2.48%   -0.00%      7.5%      7.5%
  residual spread (1.4826 x MAD; sd): R2 0.1338 (0.3404); R3 0.1333 (0.3404)
    largest shares of the sd's square, R2: FRCB 67.5%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of the sd's square, R3: FRCB 67.6%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of V^2, R2: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    largest shares of V^2, R3: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: R2 p10 15.58% median 24.80% p90 40.78% (552); R3 p10 15.58% median 24.79% p90 40.69% (551)
  specific risk, the 551 in both: R2 p10 15.58% median 24.78% p90 40.69% (551); R3 p10 15.58% median 24.79% p90 40.69% (551)
    of those in both, the largest rises: ECHO +0.08%, OKTA +0.05%, FLEX +0.05%, TWLO +0.04%, MRVL +0.04%, P +0.03%, BE +0.03%, VEEV +0.02%
    the largest falls: EXPE -0.01%, ZION -0.01%, BKNG -0.01%, LII -0.01%, TMUS -0.01%, LUMN -0.03%, SBNY -0.04%, HON -0.08%
  fallback forecasts: R2 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
                      R3 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
    in R3 not R2: none; in R2 not R3: none
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    R2 -3.139 -1.810 risk 11.88%; R3 -3.137 -1.808 risk 11.87%
    largest 50, equally weighted     R2 +0.170 +0.052 risk 0.63%; R3 +0.167 +0.052 risk 0.62%
    the index, cap-weighted          R2 +0.001 -0.001 risk 0.01%; R3 +0.001 -0.001 risk 0.01%
  regression universe: R2 567, R3 567; in R2 not R3: none
    in R3 not R2 (0): 
====================================================================================================
Step the three rulings together, against midoffice08's run D: D -> R3
  D: data/external/polygon/sp500_securities_2022-01-03_2026-11-03.npz --industries --membership --returns total --volumes traded
  R3: data/external/polygon/sp500_securities_2022-01-03_2026-11-03_repaired.npz --industries --membership --returns total --volumes traded
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    D: min 499 median 502 max 503 first 502 last 500
    R3: min 499 median 502 max 503 first 502 last 499
    R3 less D, per period: min -1 median +0 max +0
  mean cross-sectional R-squared: D 0.293189, R3 0.293204, difference +0.000015
  factor                    vol D   vol R3    mean D   mean R3   d mean   noise D  noise R3
  Construction             19.43%   19.42%   +24.77%   +24.76%   -0.01%     38.1%     38.2%
  Finance & Real Estate    15.35%   15.34%   +18.11%   +18.11%   +0.00%     30.9%     31.0%
  Manufacturing            10.20%   10.20%   +20.90%   +20.88%   -0.01%     23.2%     23.2%
  Mining                   18.63%   18.61%   +14.96%   +14.94%   -0.03%     18.5%     18.5%
  Retail                   13.14%   13.11%   +14.73%   +14.63%   -0.10%     10.7%     10.7%
  SIC20                    11.98%   11.96%   +11.04%   +11.03%   -0.01%     33.5%     33.6%
  SIC28                    13.14%   13.14%   +16.64%   +16.64%   -0.00%     10.2%     10.2%
  SIC35                    18.81%   18.81%   +29.69%   +29.69%   -0.00%      5.2%      5.2%
  SIC36                    21.13%   21.13%   +32.58%   +32.55%   -0.03%      4.5%      4.5%
  SIC37                    14.24%   14.22%   +15.43%   +15.42%   -0.02%     19.1%     19.1%
  SIC38                    14.70%   14.69%   +15.82%   +15.82%   +0.00%     10.6%     10.6%
  SIC49                    15.82%   15.80%   +24.01%   +24.02%   +0.01%     11.3%     11.3%
  SIC60                    21.86%   21.85%   +21.96%   +21.97%   +0.01%      7.4%      7.4%
  SIC62                    18.75%   18.73%   +18.62%   +18.62%   +0.01%     11.5%     11.5%
  SIC63                    14.57%   14.57%   +19.16%   +19.16%   -0.00%     14.5%     14.5%
  SIC67                    15.02%   15.01%   +18.00%   +18.00%   +0.00%     16.7%     16.8%
  SIC73                    18.15%   18.18%   +15.18%   +15.21%   +0.03%      3.0%      2.9%
  Services                 12.74%   12.72%   +16.18%   +16.11%   -0.07%     27.9%     28.0%
  Transport & Utilities    13.52%   13.43%   +15.39%   +15.46%   +0.08%     16.3%     16.5%
  Wholesale                13.97%   13.98%   +20.70%   +20.69%   -0.00%     46.2%     46.2%
  Volatility                9.37%    9.38%    +3.08%    +3.09%   +0.00%      3.5%      3.5%
  Momentum                  7.21%    7.22%    +4.23%    +4.23%   -0.01%      3.6%      3.6%
  Size                      3.72%    3.72%    +3.14%    +3.13%   -0.01%      7.7%      7.7%
  Size Nonlinearity         2.31%    2.32%    -0.51%    -0.53%   -0.02%     47.0%     46.6%
  Trading Activity          6.21%    6.25%    +2.48%    +2.48%   -0.00%      7.6%      7.5%
  residual spread (1.4826 x MAD; sd): D 0.1330 (0.3405); R3 0.1333 (0.3404)
    largest shares of the sd's square, D: FRCB 67.5%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of the sd's square, R3: FRCB 67.6%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of V^2, D: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    largest shares of V^2, R3: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: D p10 15.59% median 24.80% p90 40.78% (552); R3 p10 15.58% median 24.79% p90 40.69% (551)
  specific risk, the 551 in both: D p10 15.59% median 24.78% p90 40.68% (551); R3 p10 15.58% median 24.79% p90 40.69% (551)
    of those in both, the largest rises: OKTA +0.18%, FBIN +0.15%, PVH +0.13%, GAP +0.13%, SOLS +0.12%, PENN +0.10%, MBC +0.10%, ECHO +0.10%
    the largest falls: SWKS -0.02%, MRVL -0.02%, YUM -0.03%, LII -0.03%, Q -0.03%, HON -0.09%, DGX -0.11%, SBUX -0.30%
  fallback forecasts: D 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
                      R3 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
    in R3 not D: none; in D not R3: none
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    D -3.139 -1.810 risk 11.82%; R3 -3.137 -1.808 risk 11.87%
    largest 50, equally weighted     D +0.170 +0.052 risk 0.62%; R3 +0.167 +0.052 risk 0.62%
    the index, cap-weighted          D +0.001 -0.001 risk 0.01%; R3 +0.001 -0.001 risk 0.01%
  regression universe: D 567, R3 567; in D not R3: none
    in R3 not D (0): 
```

### old_without_price.txt -- the old panel at the new defaults, without --returns price (its last line)

```text
  this panel, on tickers, carries price returns alone: run it with --returns price; total returns, the default, need build_sp500_securities.py's
```

### defaults_equal.txt -- item 2: the run at its new defaults against item 1's run, field by field

```text
item 1's run (R3): data/external/polygon/sp500_securities_2022-01-03_2026-11-03_repaired.npz --industries --membership --returns total --volumes traded
the new defaults (N): --industries --membership
  status             identical
  dates              identical
  securities         identical
  factor_names       identical
  vol                identical
  factor_returns     identical
  estimated_dates    identical
  noise_share        identical
  mean_r2            identical
  sizes              identical
  fallback           identical
  universe           identical
  specific           identical
  exposures          identical
  spread_mad         identical
  spread_sd          identical
  spread             identical
  v_squared_shares   identical
  spread_shares      identical
  size_pair          identical
  pair_names         identical
  caps_last          identical
  member_last        identical
```

### covered_report.txt -- item 3: covered histories against regression residuals alone

```text
NR: --industries --membership --specific-history regression
NC: --industries --membership --specific-history covered
  spread            identical
  spread_sd         identical
  spread_mad        identical
  v_squared_shares  identical
  spread_shares     identical
  factor_returns    identical
  vol               identical
  mean_r2           identical
  sizes             identical
  universe          identical
  exposures         identical

The blend's model error, the relative residual spread: 0.1333 in both; the characteristics' weight at n periods is 1/(2(n-1)) / (0.017773 + 1/(2(n-1)))
specific returns measured outside the regressions: 55,830

The fallback list: NR 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
                   NC 0: none
  blended outside every regression in NC: 11
    AMTM   history  399 specific returns, the characteristics' weight 6.60%; forecast 43.33% -> 41.41%
    FBIN   history  938 specific returns, the characteristics' weight 2.91%; forecast 55.12% -> 32.66%
    GAP    history  938 specific returns, the characteristics' weight 2.91%; forecast 45.52% -> 47.15%
    IPGP   history  938 specific returns, the characteristics' weight 2.91%; forecast 51.56% -> 43.73%
    MBC    history  847 specific returns, the characteristics' weight 3.22%; forecast 59.81% -> 40.61%
    PENN   history  938 specific returns, the characteristics' weight 2.91%; forecast 49.14% -> 43.91%
    PVH    history  938 specific returns, the characteristics' weight 2.91%; forecast 49.63% -> 38.79%
    SOLS   history  127 specific returns, the characteristics' weight 18.25%; forecast 57.04% -> 57.04%
    UA     history  938 specific returns, the characteristics' weight 2.91%; forecast 43.67% -> 41.97%
    UAA    history  938 specific returns, the characteristics' weight 2.91%; forecast 51.13% -> 44.76%
    VNO    history  938 specific returns, the characteristics' weight 2.91%; forecast 37.88% -> 30.75%

Joiners: 87 securities whose first member-day is after 2022-01-03; 21 in their first year as members on 2026-11-03 (joined within its last 252 trading days) and with a forecast:
         joined     periods as  history  history   weight   weight  forecast  forecast
                      a member       NR       NC       NR       NC        NR        NC
  Q      2025-11-03        252      125      125   18.49%   18.49%    38.16%    38.16%
  SNDK   2025-11-28        234      233      300   10.81%    8.60%    72.61%    77.03%
  ARES   2025-12-11        225      224      938   11.20%    2.91%    30.27%    23.96%
  CRH    2025-12-22        218      217      654   11.52%    4.13%    31.68%    27.47%
  CVNA   2025-12-22        218      217      938   11.52%    2.91%    52.73%    83.35%
  FIX    2025-12-22        218      217      938   11.52%    2.91%    30.94%    35.33%
  CIEN   2026-02-09        186      185      938   13.26%    2.91%    56.35%    42.36%
  COHR   2026-03-23        157      156      938   15.36%    2.91%    66.50%    54.52%
  ECHO   2026-03-23        157      156      938   15.36%    2.91%    57.44%    75.74%
  LITE   2026-03-23        157      156      938   15.36%    2.91%    67.93%    52.95%
  VRT    2026-03-23        157      156      938   15.36%    2.91%    47.76%    44.07%
  CASY   2026-04-09        145      144      938   16.44%    2.91%    46.05%    30.11%
  VEEV   2026-05-07        125      124      938   18.61%    2.91%    36.24%    32.06%
  FLEX   2026-06-22         95       94      938   23.22%    2.91%    43.33%    37.58%
  MRVL   2026-06-22         95       94      938   23.22%    2.91%    48.86%    46.84%
  FERG   2026-08-05         64       63      440   31.21%    6.02%    26.74%    29.20%
  RDDT   2026-08-18         55       54      531   34.67%    5.04%    72.16%    70.38%
  BE     2026-09-21         32       31      938   48.39%    2.91%    65.76%    77.14%
  P      2026-09-21         32       31      938   48.39%    2.91%    56.22%    48.48%
  TWLO   2026-10-06         21       20      938   59.69%    2.91%    60.44%    47.59%
  OKTA   2026-10-30          3        2      938   96.57%    2.91%    67.75%    46.76%
  the characteristics' weight over them: median 16.44% -> 2.91%; mean 26.20% -> 4.23%

Every joiner with a forecast (81): specific returns added by the covered history -- median 345, max 936; none for 20

Leavers still forecast on 2026-11-03: 52
  GAP    left after 2022-02-02: history 0 -> 938; forecast 45.52% -> 47.15%
  IPGP   left after 2022-06-17: history 0 -> 938; forecast 51.56% -> 43.73%
  UA     left after 2022-06-17: history 0 -> 938; forecast 43.67% -> 41.97%
  UAA    left after 2022-06-17: history 0 -> 938; forecast 51.13% -> 44.76%
  PENN   left after 2022-09-16: history 0 -> 938; forecast 49.14% -> 43.91%
  PVH    left after 2022-09-16: history 0 -> 938; forecast 49.63% -> 38.79%
  FBIN   left after 2022-12-16: history 0 -> 938; forecast 55.12% -> 32.66%
  MBC    left after 2022-12-16: history 0 -> 847; forecast 59.81% -> 40.61%
  VNO    left after 2023-01-04: history 0 -> 938; forecast 37.88% -> 30.75%
  SBNY   left after 2023-03-14: history 22 -> 851; forecast 58.25% -> 395.72%
  LUMN   left after 2023-03-17: history 28 -> 938; forecast 63.19% -> 88.53%
  AAP    left after 2023-08-24: history 138 -> 938; forecast 53.98% -> 56.64%
  LNC    left after 2023-09-15: history 153 -> 938; forecast 27.66% -> 30.24%
  NWL    left after 2023-09-15: history 153 -> 938; forecast 37.34% -> 53.00%
  DXC    left after 2023-10-02: history 164 -> 938; forecast 44.45% -> 39.42%
  OGN    left after 2023-10-17: history 175 -> 938; forecast 33.60% -> 48.93%
  ALK    left after 2023-12-15: history 217 -> 938; forecast 32.79% -> 34.81%
  SEDG   left after 2023-12-15: history 217 -> 938; forecast 54.27% -> 81.04%
  WHR    left after 2024-03-15: history 278 -> 938; forecast 30.65% -> 40.28%
  ZION   left after 2024-03-15: history 278 -> 938; forecast 35.34% -> 25.01%
  VFC    left after 2024-04-02: history 289 -> 938; forecast 43.15% -> 46.12%
  XRAY   left after 2024-04-02: history 289 -> 938; forecast 24.94% -> 33.29%
  RHI    left after 2024-06-21: history 345 -> 938; forecast 22.64% -> 35.76%
  AAL    left after 2024-09-20: history 408 -> 938; forecast 31.09% -> 35.29%
  BIO    left after 2024-09-20: history 408 -> 938; forecast 29.17% -> 31.88%
  ETSY   left after 2024-09-20: history 408 -> 938; forecast 39.79% -> 43.74%
  BBWI   left after 2024-09-30: history 414 -> 938; forecast 35.41% -> 42.55%
  AMTM   left after 2024-12-20: history 0 -> 399; forecast 43.33% -> 41.41%
  BWA    left after 2025-03-21: history 532 -> 938; forecast 29.17% -> 31.20%
  CE     left after 2025-03-21: history 532 -> 938; forecast 35.82% -> 41.86%
  FMC    left after 2025-03-21: history 532 -> 938; forecast 42.13% -> 56.96%
  TFX    left after 2025-03-21: history 532 -> 938; forecast 29.01% -> 31.35%
  CZR    left after 2025-09-19: history 657 -> 938; forecast 31.56% -> 35.05%
  ENPH   left after 2025-09-19: history 657 -> 938; forecast 60.34% -> 64.04%
  MKTX   left after 2025-09-19: history 657 -> 938; forecast 35.73% -> 38.43%
  KMX    left after 2025-10-30: history 686 -> 938; forecast 32.21% -> 36.69%
  EMN    left after 2025-11-03: history 688 -> 938; forecast 25.37% -> 26.88%
  LKQ    left after 2025-12-19: history 721 -> 938; forecast 25.13% -> 28.94%
  MHK    left after 2025-12-19: history 721 -> 938; forecast 28.44% -> 30.40%
  SOLS   left after 2025-12-19: history 0 -> 127; forecast 57.04% -> 57.04%
  LW     left after 2026-03-20: history 782 -> 938; forecast 39.25% -> 38.19%
  MOH    left after 2026-03-20: history 782 -> 938; forecast 41.79% -> 42.30%
  MTCH   left after 2026-03-20: history 782 -> 938; forecast 33.35% -> 32.86%
  PAYC   left after 2026-03-20: history 782 -> 938; forecast 39.39% -> 40.69%
  EPAM   left after 2026-06-01: history 831 -> 938; forecast 38.53% -> 39.46%
  CPB    left after 2026-06-18: history 844 -> 938; forecast 20.00% -> 20.95%
  POOL   left after 2026-06-18: history 844 -> 938; forecast 25.88% -> 27.24%
  CAG    left after 2026-06-29: history 850 -> 938; forecast 18.61% -> 19.19%
  BLDR   left after 2026-09-18: history 690 -> 938; forecast 36.96% -> 37.18%
  TAP    left after 2026-09-18: history 907 -> 938; forecast 20.77% -> 20.87%
  TTD    left after 2026-09-18: history 295 -> 938; forecast 59.72% -> 51.80%
  CTVA   left after 2026-10-05: history 918 -> 938; forecast 53.50% -> 53.43%

Specific risk over the 551 securities with a forecast in both:
  p10 15.58% -> 15.58%
  p25 19.55% -> 19.55%
  p50 24.79% -> 24.79%
  p75 31.69% -> 31.95%
  p90 40.69% -> 40.56%
  over the 499 members on 2026-11-03: p10 15.31% -> 15.31%, median 24.00% -> 23.96%, p90 38.19% -> 38.01%
  forecasts that move: 108 of 551
  the largest falls:
    FBIN   55.12% -> 32.66% (-22.46%); history 0 -> 938; not a member then
    OKTA   67.75% -> 46.76% (-20.99%); history 2 -> 938; a member on the forecast date
    MBC    59.81% -> 40.61% (-19.20%); history 0 -> 847; not a member then
    CASY   46.05% -> 30.11% (-15.93%); history 144 -> 938; a member on the forecast date
    LITE   67.93% -> 52.95% (-14.98%); history 156 -> 938; a member on the forecast date
    CIEN   56.35% -> 42.36% (-13.98%); history 185 -> 938; a member on the forecast date
    TWLO   60.44% -> 47.59% (-12.85%); history 20 -> 938; a member on the forecast date
    COHR   66.50% -> 54.52% (-11.99%); history 156 -> 938; a member on the forecast date
    DDOG   58.73% -> 47.68% (-11.05%); history 333 -> 938; a member on the forecast date
    PVH    49.63% -> 38.79% (-10.84%); history 0 -> 938; not a member then
    ZION   35.34% -> 25.01% (-10.33%); history 278 -> 938; not a member then
    TTD    59.72% -> 51.80% (-7.92%); history 295 -> 938; not a member then
  the largest rises:
    SBNY   58.25% -> 395.72% (+337.47%); history 22 -> 851; not a member then
    CVNA   52.73% -> 83.35% (+30.62%); history 217 -> 938; a member on the forecast date
    SEDG   54.27% -> 81.04% (+26.78%); history 217 -> 938; not a member then
    LUMN   63.19% -> 88.53% (+25.34%); history 28 -> 938; not a member then
    ECHO   57.44% -> 75.74% (+18.30%); history 156 -> 938; a member on the forecast date
    NWL    37.34% -> 53.00% (+15.66%); history 153 -> 938; not a member then
    OGN    33.60% -> 48.93% (+15.33%); history 175 -> 938; not a member then
    FMC    42.13% -> 56.96% (+14.83%); history 532 -> 938; not a member then
    RHI    22.64% -> 35.76% (+13.12%); history 345 -> 938; not a member then
    COIN   52.84% -> 65.23% (+12.39%); history 367 -> 938; a member on the forecast date
    BE     65.76% -> 77.14% (+11.38%); history 31 -> 938; a member on the forecast date
    WHR    30.65% -> 40.28% (+9.62%); history 278 -> 938; not a member then
```

### sols_sbny.txt -- SOLS's forecast to more places, and SBNY's history

```text
SOLS NR 0.5703754806925532 NC 0.5703783527753036 history 0 127
SBNY NR 0.5824837146400126 NC 3.957223633104606 history 22 851
SBNY: priced days 1162, returns 1126, member-days 300; tickers [np.str_('SBNY')]
  daily return sd 0.2238 (annual 355.22%); raw close range 0.0022..366.00
  largest daily returns: 2023-11-27 +448.4% (0.0031->0.0170), 2023-11-02 +220.0% (0.0050->0.0160), 2023-11-10 +163.9% (0.0054->0.0143), 2023-11-30 +150.0% (0.0200->0.0500), 2023-12-15 +149.9% (0.0800->0.1999), 2023-09-13 +127.3% (0.0220->0.0500), 2023-09-26 +105.9% (0.0170->0.0350), 2023-12-13 +96.7% (0.0305->0.0600)
  from 2023-03-28: sd 411.81% annual over 829 returns
SOLS: priced days 254, returns 253, member-days 36; tickers [np.str_('SOLS')]
  daily return sd 0.0345 (annual 54.77%); raw close range 41.4300..88.60
  largest daily returns: 2026-02-11 +17.5% (63.8000->74.9500), 2026-07-06 -15.1% (80.1900->68.0500), 2026-08-28 +12.8% (56.3400->63.5300), 2025-11-05 +10.7% (43.6500->48.3300), 2026-05-11 +9.2% (78.2500->85.4200), 2026-07-07 -8.7% (68.0500->62.1000), 2026-03-20 -8.3% (73.0800->67.0500), 2026-10-22 -8.0% (64.9900->59.8000)
```

### capture/report_R3.txt -- item 1's run: the repaired panel, total returns, traded volumes, regression histories, in full

```text
Panel
  1214 trading days, 592 securities
  2022-01-03 .. 2026-11-03
  on securities, each a share class labelled by its last ticker; total returns, the market's total return; traded volumes
  684,460 of 718,688 return observations present (95.2%)

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, covering 1976-07-01 .. 2026-11-04
  the constituent list predates the change of 2026-10-30 -- OKTA added, KVUE removed -- and is read with it applied
  the panel's own: membership() read through the identity table's renames and mapped to securities, a reorganisation's under its successor
  one array for every standardisation and every regression below, each held to that day's members: 505 of the panel's 592 on 2022-01-03, 503 on 2026-11-03
  1 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
  - 2026-03-23 SATS recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 157 panel day(s) from then

Descriptors
  computed 14: LNCAP, LCAPCB, LPRI, DASTD, HILO, CMRA, BTSG, HALPHA, SERDP, RSTR, STOM, STOQ, STOA, VLVR
  - CMRA: computed on raw rather than excess returns: no risk_free supplied
  - RSTR: computed on raw rather than excess returns: no risk_free supplied
  - SERDP: computed at the final period only; the market-model residual series is retained for one window rather than all of them, so SERDP is a current exposure and not a time series
  - STO5: needs 1380 period(s) of history and the panel has 1214, so every value would be missing
  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days (the library's default, USE4's)
  window descriptors need at least 50% of the window valid; from a window with none, over 5,170,595 values: none; from one under 50% valid: none
  462,134 raw values winsorised before standardising
  191,758 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/52  LCAPCB 52/52  LPRI 4/52  DASTD 4/55  HILO 9/52  CMRA 5/54  BTSG 7/58
    HALPHA 4/58  SERDP 0/58  RSTR 4/58  STOM 15/53  STOQ 13/55  STOA 17/58  VLVR 19/58

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         94%      962    8.58%     10.03%     1.09
  DASTD        94%    1,149    6.51%      8.42%     1.04
  RSTR         94%      938    5.49%      7.78%     1.04
  HALPHA       94%      962    4.42%      6.67%     0.56
  HILO         94%    1,213    3.88%      6.74%     0.81
  CMRA         93%      938    3.74%      6.26%     1.16
  STOQ         94%    1,145    3.60%      5.92%     0.68
  STOM         94%    1,191    3.48%      5.88%     0.69
  STOA         94%      938    3.27%      5.79%     0.60
  LPRI         94%    1,213    1.86%      4.03%    -0.91
  LNCAP        93%    1,213    1.78%      4.03%     0.66
  LCAPCB       93%    1,213    1.65%      3.93%     0.41
  VLVR         94%      938    1.55%      3.71%     1.55
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
  each security's SIC code from its own latest record with one: 587 of 592; none for DISCA, DISCK, FRCB, SBNY, VYLR
  567 of 592 securities are in at least one estimated period's regression while an index member; every count below is over them
  20 industries over 567 securities in the universe (592 classified); largest SIC73 with 70 (12%), 45 thin group(s) merged into their division
    SIC73                      70
    SIC38                      44
    SIC28                      40
    Manufacturing              39
    SIC49                      37
    SIC36                      36
    Retail                     34
    SIC35                      34
  - 2 industry/industries have no member in the universe and are not factors: SIC24, SIC25
  - 2 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture, Unclassified
  - Other Industries pooled only 3, below the hard floor of 5, so it is not a factor. 3 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA, FRCB, SBNY
  - 3 industry/industries still have fewer than 15 members after merging and their factor returns are largely their members' specific returns: Construction (8), Finance & Real Estate (12), Wholesale (12)
  - 2 security(ies) have no usable SIC code and are left with no industry factor: FRCB, SBNY
  - 25 classified security(ies) are outside the universe and count toward no industry: ABMD, AMTM, CERN, CTXS, DISCA

Model
  regression weights: square root of capitalisation
  25-factor model, annualised, estimated on 938 daily period(s) for 551 of 592 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.293
  0 collinear period(s) of 938 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 105 security-period(s) of the regressions move -- 47 in, on the day a name left the index, 58 out, on the day one joined; the mask itself differs on 140
  each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2023-02-08 (938)
    Momentum             RSTR, HALPHA from 2023-02-08 (938)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LNCAP from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA, VLVR from 2023-02-08 (938)
  every estimated period's blend is the forecast exposures', 2026-11-03's, in every index: one definition per factor across the sample
  24 period(s) held enough securities and are not estimated, a risk index's blend in each not being the forecast's:
    2023-01-04 to 2023-02-07 (24): Volatility from BTSG, DASTD, HILO, LPRI, without CMRA; Momentum from HALPHA, without RSTR; Trading Activity from STOM, STOQ, without STOA, VLVR

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 1.7e-16 from zero (0.8 eps), equal-weighted sd at most 2.2e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  +2.2e-17   1.000000   503
  Momentum                    +7.2e-18   1.000000   500
  Size                        +0.0e+00   1.000000   500
  Size Nonlinearity           -3.6e-17   1.000000   500
  Trading Activity            +0.0e+00   1.000000   501
  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,106 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates

  The size pair on 2026-11-03, over the 499 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
                                                 Size  Size Nonlinearity     risk
  smallest 85, equally weighted                -3.137             -1.808   11.87%
  largest 50, equally weighted                 +0.167             +0.052    0.62%
  the index, cap-weighted                      +0.001             -0.001    0.01%

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     503    505          503          503
  of them in the panel      503     503    505          503          503
  in the regression         499     502    503          502          499
  member-periods over the 938 periods: 471,826; in the regression 470,528; out of it, by why:
    not in the panel              0
    no weight                   224   FDXF 108, VYLR 23, SKYD 22, Q 13, SOLS 13, AMTM 5
    no return                    23   WBA 1, WBD 1, WRK 1, SBNY 1, SIVB 1, PXD 1
    an exposure missing       1,051   SW 126, SOLV 125, VLTO 124, SKYD 124, GEV 122, Q 113
    the first exposure missing: Momentum 1,051

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               19.42%     +24.76%       38.2%      938          0      0
  Finance & Real Estate      15.34%     +18.11%       31.0%      938          0      0
  Manufacturing              10.20%     +20.88%       23.2%      938          0      0
  Mining                     18.61%     +14.94%       18.5%      938          0      0
  Retail                     13.11%     +14.63%       10.7%      938          0      0
  SIC20                      11.96%     +11.03%       33.6%      938          0      0
  SIC28                      13.14%     +16.64%       10.2%      938          0      0
  SIC35                      18.81%     +29.69%        5.2%      938          0      0
  SIC36                      21.13%     +32.55%        4.5%      938          0      0
  SIC37                      14.22%     +15.42%       19.1%      938          0      0
  SIC38                      14.69%     +15.82%       10.6%      938          0      0
  SIC49                      15.80%     +24.02%       11.3%      938          0      0
  SIC60                      21.85%     +21.97%        7.4%      938          0      0
  SIC62                      18.73%     +18.62%       11.5%      938          0      0
  SIC63                      14.57%     +19.16%       14.5%      938          0      0
  SIC67                      15.01%     +18.00%       16.8%      938          0      0
  SIC73                      18.18%     +15.21%        2.9%      938          0      0
  Services                   12.72%     +16.11%       28.0%      938          0      0
  Transport & Utilities      13.43%     +15.46%       16.5%      938          0      0
  Wholesale                  13.98%     +20.69%       46.2%      938          0      0
  Volatility                  9.38%      +3.09%        3.5%      938          0      0
  Momentum                    7.22%      +4.23%        3.6%      938          0      0
  Size                        3.72%      +3.13%        7.7%      938          0      0
  Size Nonlinearity           2.32%      -0.53%       46.6%      938          0      0
  Trading Activity            6.25%      +2.48%        7.5%      938          0      0

  specific risk   median 24.79%   p10 15.58%   p90 40.69%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
                    relative residual spread 0.133, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.340
                    largest shares of the sd's square: FRCB 67.6%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
                    the characteristics' weight in a blended forecast, at the median history of 938 periods: 2.91%

Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
  - 41 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CERN
  - 47% of the estimated variance of Size Nonlinearity is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 11 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: AMTM, FBIN, GAP, IPGP, MBC
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### capture/report_NC.txt -- item 3's run: the new defaults, covered histories, in full

```text
Panel
  1214 trading days, 592 securities
  2022-01-03 .. 2026-11-03
  on securities, each a share class labelled by its last ticker; total returns, the market's total return; traded volumes
  684,460 of 718,688 return observations present (95.2%)

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, covering 1976-07-01 .. 2026-11-04
  the constituent list predates the change of 2026-10-30 -- OKTA added, KVUE removed -- and is read with it applied
  the panel's own: membership() read through the identity table's renames and mapped to securities, a reorganisation's under its successor
  one array for every standardisation and every regression below, each held to that day's members: 505 of the panel's 592 on 2022-01-03, 503 on 2026-11-03
  1 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
  - 2026-03-23 SATS recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 157 panel day(s) from then

Descriptors
  computed 14: LNCAP, LCAPCB, LPRI, DASTD, HILO, CMRA, BTSG, HALPHA, SERDP, RSTR, STOM, STOQ, STOA, VLVR
  - CMRA: computed on raw rather than excess returns: no risk_free supplied
  - RSTR: computed on raw rather than excess returns: no risk_free supplied
  - SERDP: computed at the final period only; the market-model residual series is retained for one window rather than all of them, so SERDP is a current exposure and not a time series
  - STO5: needs 1380 period(s) of history and the panel has 1214, so every value would be missing
  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days (the library's default, USE4's)
  window descriptors need at least 50% of the window valid; from a window with none, over 5,170,595 values: none; from one under 50% valid: none
  462,134 raw values winsorised before standardising
  191,758 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/52  LCAPCB 52/52  LPRI 4/52  DASTD 4/55  HILO 9/52  CMRA 5/54  BTSG 7/58
    HALPHA 4/58  SERDP 0/58  RSTR 4/58  STOM 15/53  STOQ 13/55  STOA 17/58  VLVR 19/58

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         94%      962    8.58%     10.03%     1.09
  DASTD        94%    1,149    6.51%      8.42%     1.04
  RSTR         94%      938    5.49%      7.78%     1.04
  HALPHA       94%      962    4.42%      6.67%     0.56
  HILO         94%    1,213    3.88%      6.74%     0.81
  CMRA         93%      938    3.74%      6.26%     1.16
  STOQ         94%    1,145    3.60%      5.92%     0.68
  STOM         94%    1,191    3.48%      5.88%     0.69
  STOA         94%      938    3.27%      5.79%     0.60
  LPRI         94%    1,213    1.86%      4.03%    -0.91
  LNCAP        93%    1,213    1.78%      4.03%     0.66
  LCAPCB       93%    1,213    1.65%      3.93%     0.41
  VLVR         94%      938    1.55%      3.71%     1.55
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
  each security's SIC code from its own latest record with one: 587 of 592; none for DISCA, DISCK, FRCB, SBNY, VYLR
  567 of 592 securities are in at least one estimated period's regression while an index member; every count below is over them
  20 industries over 567 securities in the universe (592 classified); largest SIC73 with 70 (12%), 45 thin group(s) merged into their division
    SIC73                      70
    SIC38                      44
    SIC28                      40
    Manufacturing              39
    SIC49                      37
    SIC36                      36
    Retail                     34
    SIC35                      34
  - 2 industry/industries have no member in the universe and are not factors: SIC24, SIC25
  - 2 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture, Unclassified
  - Other Industries pooled only 3, below the hard floor of 5, so it is not a factor. 3 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA, FRCB, SBNY
  - 3 industry/industries still have fewer than 15 members after merging and their factor returns are largely their members' specific returns: Construction (8), Finance & Real Estate (12), Wholesale (12)
  - 2 security(ies) have no usable SIC code and are left with no industry factor: FRCB, SBNY
  - 25 classified security(ies) are outside the universe and count toward no industry: ABMD, AMTM, CERN, CTXS, DISCA

Model
  regression weights: square root of capitalisation
  25-factor model, annualised, estimated on 938 daily period(s) for 551 of 592 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.293
  0 collinear period(s) of 938 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 105 security-period(s) of the regressions move -- 47 in, on the day a name left the index, 58 out, on the day one joined; the mask itself differs on 140
  each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2023-02-08 (938)
    Momentum             RSTR, HALPHA from 2023-02-08 (938)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LNCAP from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA, VLVR from 2023-02-08 (938)
  every estimated period's blend is the forecast exposures', 2026-11-03's, in every index: one definition per factor across the sample
  24 period(s) held enough securities and are not estimated, a risk index's blend in each not being the forecast's:
    2023-01-04 to 2023-02-07 (24): Volatility from BTSG, DASTD, HILO, LPRI, without CMRA; Momentum from HALPHA, without RSTR; Trading Activity from STOM, STOQ, without STOA, VLVR

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 1.7e-16 from zero (0.8 eps), equal-weighted sd at most 2.2e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  +2.2e-17   1.000000   503
  Momentum                    +7.2e-18   1.000000   500
  Size                        +0.0e+00   1.000000   500
  Size Nonlinearity           -3.6e-17   1.000000   500
  Trading Activity            +0.0e+00   1.000000   501
  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,106 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates

  The size pair on 2026-11-03, over the 499 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
                                                 Size  Size Nonlinearity     risk
  smallest 85, equally weighted                -3.137             -1.808   11.87%
  largest 50, equally weighted                 +0.167             +0.052    0.62%
  the index, cap-weighted                      +0.001             -0.001    0.01%

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     503    505          503          503
  of them in the panel      503     503    505          503          503
  in the regression         499     502    503          502          499
  member-periods over the 938 periods: 471,826; in the regression 470,528; out of it, by why:
    not in the panel              0
    no weight                   224   FDXF 108, VYLR 23, SKYD 22, Q 13, SOLS 13, AMTM 5
    no return                    23   WBA 1, WBD 1, WRK 1, SBNY 1, SIVB 1, PXD 1
    an exposure missing       1,051   SW 126, SOLV 125, VLTO 124, SKYD 124, GEV 122, Q 113
    the first exposure missing: Momentum 1,051

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               19.42%     +24.76%       38.2%      938          0      0
  Finance & Real Estate      15.34%     +18.11%       31.0%      938          0      0
  Manufacturing              10.20%     +20.88%       23.2%      938          0      0
  Mining                     18.61%     +14.94%       18.5%      938          0      0
  Retail                     13.11%     +14.63%       10.7%      938          0      0
  SIC20                      11.96%     +11.03%       33.6%      938          0      0
  SIC28                      13.14%     +16.64%       10.2%      938          0      0
  SIC35                      18.81%     +29.69%        5.2%      938          0      0
  SIC36                      21.13%     +32.55%        4.5%      938          0      0
  SIC37                      14.22%     +15.42%       19.1%      938          0      0
  SIC38                      14.69%     +15.82%       10.6%      938          0      0
  SIC49                      15.80%     +24.02%       11.3%      938          0      0
  SIC60                      21.85%     +21.97%        7.4%      938          0      0
  SIC62                      18.73%     +18.62%       11.5%      938          0      0
  SIC63                      14.57%     +19.16%       14.5%      938          0      0
  SIC67                      15.01%     +18.00%       16.8%      938          0      0
  SIC73                      18.18%     +15.21%        2.9%      938          0      0
  Services                   12.72%     +16.11%       28.0%      938          0      0
  Transport & Utilities      13.43%     +15.46%       16.5%      938          0      0
  Wholesale                  13.98%     +20.69%       46.2%      938          0      0
  Volatility                  9.38%      +3.09%        3.5%      938          0      0
  Momentum                    7.22%      +4.23%        3.6%      938          0      0
  Size                        3.72%      +3.13%        7.7%      938          0      0
  Size Nonlinearity           2.32%      -0.53%       46.6%      938          0      0
  Trading Activity            6.25%      +2.48%        7.5%      938          0      0

  specific risk   median 24.79%   p10 15.58%   p90 40.56%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
                    relative residual spread 0.133, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.340
                    largest shares of the sd's square: FRCB 67.6%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
                    each security's own history is every specific return it has, member or not: 55,830 measured outside the regressions; 11 security(ies) in no regression blended with their own
                    the characteristics' weight in a blended forecast, at the median history of 938 periods: 2.91%

Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
  - 41 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CERN
  - 47% of the estimated variance of Size Nonlinearity is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 11 security(ies) are in no estimated period's regression and have specific returns of their own, measured against the members' factor returns: their forecast blends that history with the characteristic forecast extrapolated to them: AMTM, FBIN, GAP, IPGP, MBC
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### pytest_all.txt -- the full suite (last line)

```text
=========== 2295 passed, 1 skipped, 19 warnings in 68.09s (0:01:08) ===========
```

### pytest_mid.txt -- the midoffice tests (last line)

```text
======================= 844 passed, 1 skipped in 24.59s =======================
```

### ruff_final.txt -- ruff over every changed Python file (each finding at HEAD too)

```text
scripts\run_sp500_model.py:41:5: F401 [*] `praxis.midoffice.model.covariance.HorizonCovariance` imported but unused
src\praxis\midoffice\model\crosssectional.py:92:20: F401 [*] `typing.Mapping` imported but unused
tests\midoffice\test_crosssectional.py:10:1: I001 [*] Import block is un-sorted or un-formatted
Found 3 errors.
[*] 3 fixable with the `--fix` option.
```

### hygiene_final.txt -- parse, ASCII and line endings, by bytes

```text
src/praxis/midoffice/refdata/securities.py: CRLF (858 lines), parses, ASCII
src/praxis/midoffice/model/crosssectional.py: CRLF (1155 lines), parses, ASCII
scripts/build_sp500_securities.py: CRLF (564 lines), parses, ASCII
scripts/run_sp500_model.py: CRLF (1220 lines), parses, ASCII
tests/midoffice/test_securities.py: CRLF (331 lines), parses, ASCII
tests/midoffice/test_crosssectional.py: CRLF (1560 lines), parses, ASCII
config/sp500_share_count_overrides.json: LF (26 lines), -, ASCII
docs/FACTOR_MODEL.md: CRLF (1038 lines), -, ASCII
claude/handoffs/BRIEF_midoffice09_counts_residuals.md: LF (81 lines), -, ASCII
all ok
```

## git add (always last -- Rule 43)

This piece -- the three count rulings and the overrides file, the run's new defaults, residuals for every covered name, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add src/praxis/midoffice/refdata/securities.py src/praxis/midoffice/model/crosssectional.py scripts/build_sp500_securities.py scripts/run_sp500_model.py config/sp500_share_count_overrides.json tests/midoffice/test_securities.py tests/midoffice/test_crosssectional.py docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice09_counts_residuals.md claude/retros/RETRO_midoffice09_counts_residuals.md
```
