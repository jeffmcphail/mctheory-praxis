# Retro: Mid-office 14 -- midoffice13's rulings, then the spin-offs the total return does not carry

**Brief:** `claude/handoffs/BRIEF_midoffice14_spin_offs.md` -- written in Chat and pasted into the
Code session, transcribed verbatim
**Date:** 2026-11-12 (EST)
**Duration:** the brief arrived shortly before 21:10Z (16:10 EST), the first clock reading after
it. Rulings 4 and 5 were built by 16:39 EST, when the runner started their check run (R4, done
16:48); the public record and the data were read, the rulings written and the panel rebuilt by
16:56; the runner ran the defaults' bias run on it from 16:57 (D14, done 17:08); the comparison,
the docs, the state file and this retro followed.
**Status:** COMPLETE.
- **Rulings 4 and 5:** built and measured -- the 21-day view's bias run equals V21N to the bit;
  the minimum-variance portfolios' realised volatility a standing line, reproducing midoffice13's
  hand measurement to the hundredth.
- **Rulings 1 to 3:** nothing to build (the adjustment stays off; this piece is ruling 2; ruling
  3 orders the next pieces).
- **Items 1 to 3:** built and measured. No default changed beyond rulings 4 and 5 and the
  rebuilt panel.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 4, the correlations' lags:** at any horizon above one day the run gives the
  correlations no lags by default (`--correlation-lags horizon` restores midoffice12's view). The
  21-day view's bias run (R4, on midoffice13's panel) equals V21N to the bit in all 24 arrays;
  its report differs only in ruling 5's new lines, the refit time and the output path.
- **Ruling 5, the realised volatility:** every portfolio in the bias tables now carries its
  forecast and realised volatility a year beside its statistic, at one day and the model's
  horizon; the minimum-variance portfolios' by half; and each eigen scale's in its tables. On
  midoffice13's stored eigenfactor run the run script's own functions reproduce
  `minvar_realised.txt` exactly, every a.
- **Item 1, CTVA:** a new overrides file, `config/sp500_spin_off_overrides.json`, carries the
  ruling with its evidence; the build sets the parent's ex-date total return to
  (w (C_t + D_t) + k V) / C_s - 1, the stitched prices left as Polygon has them. CTVA's
  2026-10-01 return goes from -83.81% to +4.10%; Vylor takes Corteva's count on the record date,
  668,820,000, over its 24 member-days. Built alone, it changes exactly those cells.
- **Item 2, the others:** of the 20 pairs the brief names, 12 are not carried (EXC's -25.9% day,
  T's -18.7%, WDC's -28.7%, FDX's -17.8%, ...), Lennar's is carried wrongly (Polygon's 1.033 a
  third of the distribution's value), and 7 are carried by Polygon's fractional ratios, within
  10% of the brief's value. The public record adds five the list does not name, each a member's:
  Jacobs' second Amentum distribution, BD's Waters shares (-17.2%), Aptiv's Versigent (-10.6%),
  Honeywell's Aerospace and S&P Global's Mobility. 19 rulings in all. FedEx Freight's count is
  seeded as Vylor's.
- **Honeywell made a 1-for-2 reverse split one minute after its Aerospace distribution,** which
  the cache does not record: the ruling applies it to the count, checked by Honeywell's own
  samples, and so supersedes midoffice08's HON count ruling -- its "contaminated" records were
  Honeywell's own post-split count. HON's capitalisation from 2026-06-29 was double.
- **Item 3, the rebuilt panel,** `_spinoffs`, the run's default, `_listed` kept and rebuilt to the
  bit by `--spin-offs none`: 19 member-day returns and 223 count member-days changed, nothing
  else. The defaults' bias run on it (D14, the guard on): the minimum-variance portfolios'
  statistics fall from 1.167 and 1.220 to 1.059 and 1.101, their realised volatility from 10.93%
  and 11.07% a year to 9.92% and 10.01% -- midoffice13's sensitivity without the CTVA day, the
  -13 sigma outcome gone; the broad portfolios, the members, the industries and the
  eigenportfolios move in the third decimal or not at all; CTVA's member statistic 1.828 to 0.874.
- **What I would do next:** the volatility regime adjustment, as ruled. With the data error gone
  the optimiser's bias is the regime's -- 1.25 and 1.30 in the first half, 0.83 and 0.86 in the
  second, the index 1.22 then 0.75. Then the eigenvalue adjustment one-sided on it; and Jeff's
  rulings on the five spin-offs the record added and on Honeywell's count.
- **Tests:** 2,362 passed, 1 skipped; 6 new.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice14_spin_offs.md`, verbatim. In outline:

- **"committed -- midoffice13."** Checked: cee34ca in `git log`. Read the state file and the
  midoffice13 retro first, as asked.
- **Rulings on midoffice13's decisions:**
  1. The eigenfactor risk adjustment: an option, off.
  2. CTVA: next, as a data piece, with the search for others.
  3. After it, the volatility regime adjustment, then the eigenvalue adjustment one-sided,
     re-tested on the regime-adjusted model -- a shape adjustment judged once the level is right.
  4. The reporting view at 21 days: the lags on the volatilities alone; at any horizon above one
     day the correlations take no lags by default; the 21-day view's bias run must equal V21N's
     to the bit.
  5. The minimum-variance portfolios' realised volatility a standing line of the bias report,
     beside their statistic, kept for every covariance candidate.
- **midoffice14, spin-offs the total return does not carry,** the rulings first:
  1. **CTVA:** one Vylor share a Corteva share, record 2026-09-24, distributed 2026-10-01, VYLR
     regular-way that day; CTVA's total return that day (CTVA's close + VYLR's close) / CTVA's
     prior close - 1, plus any dividend, carried through an overrides file with its evidence;
     Vylor's missing count Corteva's on the record date at the ratio, until Polygon carries one.
  2. **The others:** check every security's first day in the panel against the members' raw
     moves, and the 20 named pairs from 2022 on against the cache and the panel; the data and
     the public record the authority, a pair whose parent was no member needing nothing; each
     carried, carried wrongly or not carried, with the fix for the last two; FDXF's missing count
     as Vylor's; LEN's adjustment of 2025-01-21 checked with the rest.
  3. **The rebuilt panel** under a new name, the old one kept, the run's default; measured: every
     changed member-day, the defaults' run and their bias run against midoffice13's, the
     minimum-variance portfolios' statistics and realised volatility first.
- Runs one at a time through the runner. **"Change no default beyond rulings 4 and 5 and the
  rebuilt panel. Report, and say what you would do next and why."** Retro with its `git add` and
  the state file in the same commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `scripts/run_sp500_model.py` | ruling 4: `--correlation-lags` takes a number or `horizon` (`lags_or_horizon`), none at any horizon above one day by default, its checks and help, `HORIZON`'s docstring; ruling 5: `_volatility`, `_halves`, the portfolio tables' forecast and realised volatility columns at one day and the model's horizon, the optimiser's lines by half, each eigen scale's realised volatility in `_eigen_reading`, `_report_eigen` on `_halves`; item 3: `DEFAULT_PANEL` the `_spinoffs` panel | 84-89, 116-119, 143-155, 984-1023, 1242-1262, 1280-1283, 1324-1325, 1506-1511, 1567-1569, 1581-1591, 1835, 1880 |
| `src/praxis/midoffice/refdata/securities.py` | `SpinOff`, `read_spin_offs`, `spin_off_splits` (a ruling's split as a split check, by `rebuild.classify` on the parent's samples), `SpinOffUse`, `Seeded`; `Panel.spin_offs`, `.seeded`, `.opens`; `build_panel(spin_offs=)`: the raw opens, the ex-date's total return by the ruling (the old expression kept on every other day), a child's count seeded from its parent's; `FirstDay`, `first_days` (the standing check); the imports | 65, 72-73, 418-568, 653-658, 688-717, 773, 793-825, 899-926, 1154-1219 |
| `scripts/build_sp500_securities.py` | `--spin-offs` (default the new file), `--suffix` default `_spinoffs`; the ruling's split added to the gate's checks; a count override a ruling supersedes not read, listed; the Spin-offs section of the report (each ruling's ex-date, the child's record checked against its key, the split, the seeded counts, the superseded override); the first-day check; the JSON's new keys; the docstring | 41-59, 146-157, 303-307, 338-348, 401-440, 582-595, 657-680 |
| `tests/midoffice/test_securities.py` | 6 new tests; the fixture's child ticker `KID`; `small_panel(spin_offs=, bars_of=)` | 74-75, 96-97, 120-122, 392-557 |
| `config/sp500_spin_off_overrides.json` | new: the 19 rulings with their evidence | new, 262 lines |
| `docs/FACTOR_MODEL.md` | the correlations' lags by default above one day; the spin-off rulings and the first-day check in the build; the count rulings' HON entry superseded; the counts without one; the weighted check; the default panel's changes; the bias section's standing realised-volatility line and the rebuilt panel's results | 264-270, 1157-1212, 1240-1247, 1419-1447 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice14_spin_offs.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice14_spin_offs.md` | new: this retro | new |

Written outside the tracked files:

- `data/external/polygon/sp500_securities_2022-01-03_2026-11-03_spinoffs.npz` and `.json`, the
  rebuilt panel and its report -- new files; every earlier panel is untouched (their timestamps
  are midoffice10's and before).
- `data/external/polygon/cache/`: 19 new answers -- the raw bars over the window of ZIMV, EMBC,
  FTRE, PHIN, KLG, RAL, MRP, VSNT, VGNT and MBGL, and nine reference records on their ex-dates
  (ZIMV's of 2022-03-01 and 03-02, EMBC's, PHIN's, KLG's, MRP's of 2025-02-07, VSNT's, VGNT's,
  MBGL's). Nothing overwritten; asked through the repo's client with `.env`'s key, read only.
- `claude/scratch/midoffice14/` (gitignored): the scratch, below.

---

## Ruling 4: the correlations' own lags, none above one day by default

- **Built:** `--correlation-lags` takes a whole number or `horizon`. Unset, the run passes the
  estimator `correlation_lags=0` at any horizon above one day with the two-half-life covariance,
  and nothing at one day, where there are no lags to give (so the defaults' estimator is the one
  midoffice13 ran, to the bit). `horizon` passes `None`, the horizon's lags for both,
  midoffice12's view; a number is passed as before. The estimator's own default is unchanged.
  The report's horizon line already said "the lags on the volatilities alone".
- **Measured** (`equal_R4.txt`): R4, `--horizon 21` at the new default on midoffice13's panel
  (`_listed`, given first, so the panel change cannot enter), equals midoffice13's V21N in all 24
  arrays to the bit, 686 forecast dates. Its report against V21N's (`report_R4_vs_V21N.txt`):
  the refit time, ruling 5's new columns and lines, and the output path; nothing else.

## Ruling 5: the minimum-variance portfolios' realised volatility, standing

- **Built:** `_volatility(run, name, part, horizon)` -- the forecast volatility at its median and
  the realised from the returns, a year, over a part of the test (at a horizon past one day from
  the h-day returns, the windows overlapping) -- and `_halves(run)`, the test's two halves by
  forecast date (the first n // 2), shared with the eigen report. The bias report now prints:
  - in the one-day table, every portfolio's forecast and realised volatility beside its
    statistic; in the model's-horizon table, the same from the h-day windows;
  - for each minimum-variance portfolio, its realised volatility over the whole test and each
    half, with the statistic by half, in place of midoffice13's one realised figure;
  - under `--bias-eigen-scales`, each scale's two minimum-variance portfolios' realised
    volatility in every table of the eigen report (whole test and halves), beside their
    statistics -- the test the adjustment failed, there for every scale and every later
    covariance candidate.
- **Measured** (`ruling5_check.txt`): the run script's own `_volatility` and `_halves` on
  midoffice13's stored E run give 10.93 / 11.00 / 11.09 / 11.18 / 11.29% (long-only, whole test,
  a = 0 to 4) and 11.07 / 11.19 / 11.32 / 11.44 / 11.57% (unconstrained), and the first half's,
  exactly midoffice13's `minvar_realised.txt`; the second half's are new (10.38 / 10.57% at
  a = 0, the CTVA day in it). On R4 the 21-day view's optimiser realised 10.76% and 11.18% a year
  at one day against forecasts of 7.75% and 7.19% (its statistics 1.347 and 1.503).
- No array changes: R4's 24 arrays are V21N's to the bit.

---

## Item 1: CTVA, carried through an overrides file

- **The rule.** On a ruling's ex-date the parent's total return is
  (w (C_t + D_t) + k V) / C_s - 1: C its raw closes on the ex-date t and the priced day s before
  it, D its cash distributions that day (raw, as paid), k the child's shares a parent share, V
  the child's raw close that day -- or its first close after, where it has none that day -- and
  w the ratios of the splits the build applies between s and t. With no split it is the brief's
  (C_t + V) / C_{t-1} - 1 plus the day's dividend over C_{t-1}, exactly; with a split the same
  day, the parent's new shares count as the adjusted closes count them (a test pins both). The
  stitched prices stay Polygon's adjusted closes: the ruling is the total return's, and a
  price-only factor Polygon put on the day does not enter it.
- **The file:** `config/sp500_spin_off_overrides.json`, in the repo, read by
  `securities.read_spin_offs`: each entry the parent's key, label and name, the child's ticker
  and composite FIGI, the ratio, the record, distribution and ex-dates, any split the parent made
  with it, `child_count` "parent" for a child with no count of its own, the count overrides it
  supersedes, the evidence (the public record's terms and the data's) and who ruled it. It is
  written by `claude/scratch/midoffice14/make_spin_offs.py` from the record and the data, so every
  number in its evidence is the cache's.
- **CTVA's ruling:** Corteva (BBG00BN969C1) to Vylor (VYLR), 1 a share, record 2026-09-24,
  distributed and ex 2026-10-01. The data: CTVA's raw close 77.65 on 2026-09-30, 12.57 on
  2026-10-01 (open 14.44), its count 668,820,000 either side, no split, distribution or event in
  the cache; VYLR's close 68.26 (Vylor Inc., BBG024W5VJC3). No CTVA dividend that day (its last
  ex-date 2026-09-01, the next 2026-12-01). Total return: **-83.8120% to +4.0953%**.
- **Vylor's count:** its records carry no share-class count in the window; Corteva's carried
  count on the record date, 668,820,000, times 1, from 2026-10-01 to the window's end, 24
  member-days (the build's `child_count`, a post-pass that fills a child's days with no count from
  the ex-date until its own first, never back). Its weighted count, 667,302,930 from 2026-10-08,
  is within 0.3%; 711,546,000 on 2026-10-02 and 10-05, within 6.4%.
- **Built alone and measured** (`panel_item1.txt`, `--spin-offs` a file holding CTVA's ruling
  only, into scratch): against `_listed`, one daily return moves beyond 1e-12 -- CTVA's of
  2026-10-01, +87.91 points -- and Vylor's 24 member-days gain the count; CTVA's index after the
  day is rescaled, its later daily returns equal to the last bits; every other array equal to
  the bit.

---

## Item 2: the others

### How they were found

- **The first-day check** (`firstdays.txt`, which skipped a member whose ticker changed that day;
  then standing in the build, which bridges it): 21 securities have their first priced day inside
  the window; on each, the members ranked by their raw opening gap less SPY's. A child that joined
  the index shows its parent there -- CEG's day EXC (-29.6 points), WBD's T (-21.0), MBC's FBIN
  (-16.3, in the build's check; its ticker changed that day), VLTO's DHR and K, AMTM's J,
  SNDK's WDC, FDXF's FDX, VYLR's CTVA -- and GEHC's, SOLV's, GEV's and Q's parents are explained
  by Polygon's price-only records. HONA's day shows no Honeywell gap at all: its reverse split
  offset it (below). KVUE, TKO, CRH, RDDT, SW, FERG and SKYD are an IPO, mergers and listings,
  their days' gaps news.
- **The pairs** (`pairs.txt`, `check_events.txt`): each against the cache's split events and
  distributions, the raw closes and opens, the panel's returns and counts, and the child's
  close, from the cache or, for the ten children that never joined the index, fetched into it
  (`fetch_children.txt`).
- **The public record:** the searches and pages below for each pair's terms, and the year lists
  of completed spin-offs (stockanalysis.com, 2022 to 2026) filtered to parents the panel held on
  the ex-date: of about 80 listed, the brief's pairs, and five more whose parent was a member:
  Jacobs' second Amentum distribution, BD's Waters shares (a Reverse Morris Trust), Aptiv's
  Versigent, Honeywell's Aerospace and S&P Global's Mobility. Illumina's GRAIL and Flex's Nextpower
  came on days their parents were no member (Illumina left the index on 2024-06-24; GRAIL's
  ex-date was 06-25): nothing needed. The list omits Constellation, so it is not exhaustive
  either; together with the first-day check, which sees every child that joined the index, it is
  the coverage this piece has.

### Each pair: carried, carried wrongly, or not carried

The panel's total return on the ex-date (`_listed`) against the brief's; "carried" where
Polygon's fractional ratio values the distribution within 10% of the brief's -- the build's own
tolerance -- with the value's error beside it (`check_events.txt`).

| Pair | Ex-date | Record | k | Polygon's record | The panel's | The brief's | Verdict |
|---|---|---|---|---|---|---|---|
| EXC CEG | 2022-02-02 | 2022-01-20 | 1/3 | none | -25.89% | +4.67% | not carried: ruled |
| ZBH ZIMV | 2022-03-01 | 2022-02-15 | 1/10 | none | -2.40% | -0.39% | not carried: ruled |
| BDX EMBC | 2022-04-01 | 2022-03-22 | 1/5 | none | -1.32% | +0.98% | not carried: ruled |
| T WBD | 2022-04-11 | 2022-04-05 | 0.241917 | none | -18.68% | +6.15% | not carried: ruled |
| FBHS MBC | 2022-12-15 | 2022-12-02 | 1 | none | -6.91% | +5.53% | not carried: ruled |
| GE GEHC | 2023-01-04 | 2022-12-16 | 1/3 | 1.281 | +5.82% | +6.33% | carried, value -2.2% |
| LH FTRE | 2023-07-03 | 2023-06-20 | 1 | 1.164 | +0.77% | +1.84% | carried, value -7.0% |
| BWA PHIN | 2023-07-05 | 2023-06-23 | 1/5 | none | -13.50% | +1.22% | not carried: ruled |
| DHR VLTO | 2023-10-02 | 2023-09-13 | 1/3 | none | -13.85% | -2.41% | not carried: ruled |
| K KLG | 2023-10-02 | 2023-09-21 | 1/4 | none | -11.78% | -6.17% | not carried: ruled |
| MMM SOLV | 2024-04-01 | 2024-03-18 | 1/4 | 1.196 | +6.01% | +4.93% | carried, value +6.7% |
| GE GEV | 2024-04-02 | 2024-03-19 | 1/4 | 1.253 | -2.49% | -2.22% | carried, value -1.4% |
| J AMTM | 2024-09-30 | 2024-09-23 | 1 | none | -15.84% | +4.90% | not carried: ruled |
| LEN MRP | 2025-01-21 | 2025-01-21 | 1/2 | 1.033 | -2.83% | +3.51% | carried wrongly, value -67.1%: ruled |
| WDC SNDK | 2025-02-24 | 2025-02-12 | 1/3 | none | -28.65% | -5.07% | not carried: ruled |
| FTV RAL | 2025-06-30 | 2025-06-16 | 1/3 | 1.327 | -3.38% | -4.62% | carried, value +5.5% |
| HON SOLS | 2025-10-30 | 2025-10-17 | 1/4 | 1.061 | -0.27% | -0.28% | carried, value +0.2% |
| DD Q | 2025-11-03 | 2025-10-22 | 1/2 | 2.39 | +1.54% | +1.89% | carried, value -0.6% |
| CMCSA VSNT | 2026-01-05 | 2025-12-16 | 1/25 | none | -4.77% | +0.72% | not carried: ruled |
| FDX FDXF | 2026-06-01 | 2026-05-15 | 1/2 | none | -17.79% | +0.37% | not carried: ruled, FDXF's count seeded |
| CTVA VYLR (item 1) | 2026-10-01 | 2026-09-24 | 1 | none | -83.81% | +4.10% | not carried: ruled, VYLR's count seeded |
| J AMTM, the rest | 2025-05-16 | 2025-05-16 | 0.060835 | none | +0.29% | +1.32% | not carried: ruled (the record) |
| BDX WAT | 2026-02-10 | 2026-02-05 | 0.135 | none | -17.22% | +4.07% | not carried: ruled (the record) |
| APTV VGNT | 2026-04-01 | 2026-03-17 | 1/3 | none | -10.58% | +2.78% | not carried: ruled (the record) |
| HON HONA | 2026-06-29 | 2026-06-15 | 1/2, split 1/2 | none | -1.90% | -3.54% | not carried: ruled, the split applied (the record) |
| SPGI MBGL | 2026-07-01 | 2026-06-15 | 1 | none | +1.89% | +7.10% | not carried: ruled (the record) |
| ILMN GRAL | 2024-06-25 | 2024-06-13 | 1/6 | -- | -- | -- | parent no member: nothing |
| FLEX NXT | 2024-01-03 | -- | -- | -- | -- | -- | parent no member: nothing |

Every parent above was a member on its ex-date but Illumina and Flex. The ex-dates are the
record's first regular-way day of the child, where it gives one, and the day the parent's raw
price gaps -- one correction from memory: ZimVie traded regular-way from 2022-03-01, not 03-02.

### Lennar, 2025-01-21, checked

- **The date is right.** Millrose's 424B1: one MRP Class A share for every two Lennar shares
  held at the close on the record date, 2025-01-21, distributed 2025-02-07. At under a quarter of
  Lennar's price the distribution took the regular ex-date, the record date itself under T+1, and
  Lennar's raw price gaps that day (-3.91% at the open while three homebuilders opened +1.99%,
  midoffice06's reading), not in February. Polygon's 1000-to-1033 is on that day.
- **The value is not.** 1.033 values the distribution at 4.39 a Lennar share (3.3% of 133.14);
  half a Millrose share at its first close, 26.74 on 2025-02-07, is 13.37. Millrose had no price
  on the ex-date -- its bars start on 02-07 -- so no ex-date close exists; the market's own
  reading, Lennar against the homebuilders, was about 6.55 a share at the close of 01-21.
- **The fix:** the brief's formula at Millrose's first close, the one case where it must take the
  child's first close after the ex-date: +3.51% on 2025-01-21 where the panel had -2.83% (and the
  raw close -5.93%). It gets Lennar's return across the distribution right in total; its one day
  stands 4.8 points above the homebuilders, since Millrose's value on 01-21 is unobserved. The
  stitched prices keep Polygon's 1.033. A decision for Jeff below.

### Honeywell, 2026-06-29: a spin-off and a reverse split the cache does not record

- **The record** (Honeywell's release, read in full): one Honeywell Aerospace share for every two
  Honeywell shares held at the close on 2026-06-15, distributed 12:01 a.m. ET on 2026-06-29; a
  1-for-2 reverse split of Honeywell's shares at 12:02 a.m. the same day, contingent on the
  spin-off; HONAV when-issued and HONIV ex-distribution from 06-15 to 06-26; HONA regular-way from
  06-29.
- **The data:** HON's raw close 232.21 on 06-26 and 227.80 on 06-29 (open 240.65): the reverse
  split doubles the price as the distribution halves it, so no gap. Its samples: 635,675,701 on
  2026-04-13; none on 07-14, whose weighted count, 316,826,559, is HONA's first count exactly --
  each old share became half a share of each, so the two counts start equal; 316,940,010 on
  10-12 and 317,358,518 on 11-03. The cache records no split under HON but 2025-10-30's
  price-only one.
- **midoffice08's ruling read the halving as contamination** -- HON's records carrying Honeywell
  Aerospace's count -- and carried 635,675,701 from 2026-10-12; with the samples before it at the
  same count, `_listed` carried the pre-split count from 06-29 to the end: Honeywell's
  capitalisation double for 90 member-days, its regression weight 1.41 times, its size exposure
  0.69 high in log cap.
- **The ruling** carries the distribution (1/2 a share a pre-split share) and the split (1/2),
  which `securities.spin_off_splits` makes a split check the gate would classify: whole, the
  counts following it (635,675,701 to 316,940,010, 0.499 times), the price not tested since the
  distribution in the same minute offsets it. The build applies it on 06-29 with the gate's
  splits -- 317,837,850 until Polygon's 316,940,010 resets it on 10-12 -- the split guard checks
  it, and the ex-date's total return takes w = 1/2: (0.5 x 227.80 + 0.5 x 220.19) / 232.21 - 1 =
  -3.54%, where the panel had -1.90%. It supersedes midoffice08's HON entry: the build does not
  read that entry while the ruling is read, and lists it; the count file keeps it, and
  `--spin-offs none` reads it as before. The weighted check's three HON flags (0.50) go; the
  carried count on 07-14 is 0.997 of the weighted.

### FedEx Freight's count

The brief's rule, as Vylor's: FedEx's carried count on the record date, 238,606,756, times 1/2,
119,303,378, over FDXF's 109 member-days, Polygon carrying none in the window. FedEx kept about
19.9% of FedEx Freight (the record), so those are the shares distributed: FDXF's weighted count is
149,518,133 on 2026-08-26, and the weighted check flags the seeded count at 1.2533. Grossed up by
the retained stake it would be about 148.9 million. Applied as ruled; a decision for Jeff.

### The public record, by pair

The terms as the search results quoted the filings and releases (each search's summary read; the
Honeywell release and the spin-off lists opened in full). For each, the source the evidence
names:

| Pair | Source |
|---|---|
| EXC CEG | Exelon 8-K, 2022-01-06: https://investors.exeloncorp.com/static-files/0d368af4-babb-4d6a-9085-104ad1cb901a ; completion: https://www.businesswire.com/news/home/20220202005289/en |
| ZBH ZIMV | Zimmer Biomet, 2022-02-07: https://investor.zimmerbiomet.com/news-and-events/news/2022/02-07-2022-113120666 ; Nasdaq notice: https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-22 |
| BDX EMBC | BD, 2022-02-01: https://news.bd.com/2022-02-01-BD-Board-of-Directors-Approves-embecta-Spinoff-and-Declares-Distribution-of-embecta-Stock ; embecta 10-Q: https://www.sec.gov/Archives/edgar/data/1872789/000187278922000009/embc-20220331.htm |
| T WBD | AT&T cost-basis guide: https://investors.att.com/stockholder-services/cost-basis-guide/worksheet/atandt-inc-wbd ; close: https://www.businesswire.com/news/home/20220408005578/en |
| FBHS MBC | Fortune Brands 8-K: https://www.sec.gov/Archives/edgar/data/1519751/000119312522306146/d420098dex991.htm |
| BWA PHIN | BorgWarner 8-K, 2023-07-03: https://www.sec.gov/Archives/edgar/data/908255/000090825523000036/bwa-20230703.htm |
| DHR VLTO | Danaher, 2023-08-25: https://investors.danaher.com/2023-08-25-Danaher-Declares-pro-Rata-Dividend-of-Veralto-Common-Stock-and-Announces-Expected-When-issued-Trading-of-Veralto-Common-Stock |
| K KLG | WK Kellogg 8-K: https://www.sec.gov/Archives/edgar/data/1959348/000119312523249064/d697691d8k.htm |
| J AMTM | Jacobs 8-K: https://www.sec.gov/Archives/edgar/data/52988/000119312524219294/d876828dex991.htm |
| LEN MRP | Millrose 424B1: https://www.sec.gov/Archives/edgar/data/2017206/000119312525008418/d811267d424b1.htm |
| WDC SNDK | Western Digital 8-K: https://www.sec.gov/Archives/edgar/data/106040/000119312525019294/d922460dex991.htm ; Nasdaq: https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2025-38 |
| J AMTM, the rest | Jacobs, 2025-04-30: https://s205.q4cdn.com/384284279/files/doc_news/Jacobs-Solutions-Announces-Special-Dividend-of-7299065-Shares-of-Amentum-Common-Stock-2025.pdf ; 8-K: https://www.sec.gov/Archives/edgar/data/52988/000005298825000032/exhibit991-supplementalinf.htm |
| CMCSA VSNT | Versant, 2026-01-04: https://www.businesswire.com/news/home/20260104224814/en/ |
| BDX WAT | Waters, 2026-02-09: https://www.biospace.com/press-releases/waters-completes-combination-with-bds-biosciences-diagnostic-solutions-businesses ; BD: https://www.stocktitan.net/news/BDX/bd-completes-combination-of-biosciences-diagnostic-solutions-ulzm0wanq014.html |
| APTV VGNT | Aptiv, 2026-03-05: https://www.businesswire.com/news/home/20260305507243/en/aptiv-board-of-directors-approves-spin-off-of-versigent/ |
| FDX FDXF | FedEx Freight: https://newsroom.fedexfreight.com/fedex-board-of-directors-approves-spin-off-of-fedex-freight |
| HON HONA | Honeywell, June 2026: https://www.honeywell.com/us/en/news/press-releases/2026/06/honeywell-board-of-directors-sets-record-date-and-announces-expected-timing-for-spin-off-of-honeywell-aerospace-and-honeywell-reverse-stock-split |
| SPGI MBGL | S&P Global, 2026-05-21: https://press.spglobal.com/2026-05-21-S-P-Global-Announces-Approval-of-Separation-of-Mobility-Global |
| CTVA VYLR | Corteva 8-K: https://www.sec.gov/Archives/edgar/data/0001755672/000119312526414490/ck0001755672-20260929.htm |
| the lists | https://stockanalysis.com/actions/spinoffs/2022/ (and /2023/, /2024/, /2025/), https://stockanalysis.com/actions/spinoffs/ (2026) |

---

## Item 3: the rebuilt panel, the run's default

### Every changed member-day (`panel_changed.txt`)

`data/external/polygon/sp500_securities_2022-01-03_2026-11-03_spinoffs.npz` against `_listed`:
18 of the 20 arrays equal to the bit -- prices, unadjusted prices, volumes, highs, lows, tickers,
membership, the market's returns, the SIC codes among them. The two that change:

- **total_return_prices:** 19 daily returns move beyond 1e-12, every one a ruling's ex-date and a
  member-day (the index after each is rescaled, its later daily returns equal to the last bits):

  | Day | Member | Before | After | Points |
  |---|---|---|---|---|
  | 2022-02-02 | EXC | -25.8862% | +4.6689% | +30.56 |
  | 2022-03-01 | ZBH | -2.3980% | -0.3908% | +2.01 |
  | 2022-04-01 | BDX | -1.3158% | +0.9774% | +2.29 |
  | 2022-04-11 | T | -18.6827% | +6.1504% | +24.83 |
  | 2022-12-15 | FBIN | -6.9144% | +5.5250% | +12.44 |
  | 2023-07-05 | BWA | -13.5016% | +1.2220% | +14.72 |
  | 2023-10-02 | DHR | -13.8493% | -2.4130% | +11.44 |
  | 2023-10-02 | K | -11.7795% | -6.1712% | +5.61 |
  | 2024-09-30 | J | -15.8362% | +4.8994% | +20.74 |
  | 2025-01-21 | LEN | -2.8306% | +3.5114% | +6.34 |
  | 2025-02-24 | WDC | -28.6515% | -5.0724% | +23.58 |
  | 2025-05-16 | J | +0.2873% | +1.3188% | +1.03 |
  | 2026-01-05 | CMCSA | -4.7732% | +0.7204% | +5.49 |
  | 2026-02-10 | BDX | -17.2188% | +4.0685% | +21.29 |
  | 2026-04-01 | APTV | -10.5847% | +2.7842% | +13.37 |
  | 2026-06-01 | FDX | -17.7923% | +0.3655% | +18.16 |
  | 2026-06-29 | HON | -1.8991% | -3.5377% | -1.64 |
  | 2026-07-01 | SPGI | +1.8931% | +7.0962% | +5.20 |
  | 2026-10-01 | CTVA | -83.8120% | +4.0953% | +87.91 |

- **shares_outstanding:** 223 member-days -- FDXF's 109 (none to 119,303,378), HON's 90
  (635,675,701 to 317,837,850 from 2026-06-29, to Polygon's own 316,940,010 and 317,358,518 from
  10-12), VYLR's 24 (none to 668,820,000).

The build's report besides: member-days without a count 101 (234); the weighted check's carried
records 49 met, 6 flagged (46, 7) -- HON's three gone, FDXF's two new; count moves 88, as on
`_listed`; every guard passes, the split guard over one more split. The first-day check marks
every parent's gap on its child's first day by its ruling or Polygon's record.

### The defaults' run (`model_compare.txt`)

At the defaults on `_spinoffs` (M14) against midoffice13's final defaults on `_listed` (its FD):
11 of 21 fields equal; the factor returns (largest difference 0.026), the covariance (4.6e-4),
the exposures, the specific risk and the noise shares move:

| | `_listed` (FD) | `_spinoffs` (M14) |
|---|---|---|
| mean R-squared | 0.293203 | 0.293826 |
| members in the regression a period, median (range) | 502 (499-503) | 502 (499-503) |
| specific risk, p10 / median / p90 | 16.58 / 28.98 / 48.98% | 16.56 / 28.99 / 48.89% |
| size pair, smallest 85 / largest 50 / index | 10.99 / 0.57 / 0.01% | 11.05 / 0.60 / 0.01% |
| member-periods kept out with no weight | 224 | 93 |
| kept out with an exposure missing (all Momentum) | 1,051 | 1,182 |
| Wholesale's noise share, equal / weighted | 39.0 / 60.5% | 39.5 / 60.4% |
| Size Nonlinearity's | 31.6 / 47.6% | 31.0 / 45.9% |

FDXF's and VYLR's 131 member-periods now have a weight, and still miss Momentum, a new name's.
Industry means move by up to 0.8% a year (Construction +24.76% to +25.53%), the style factors'
by under 0.2%.

### The defaults' bias run (D14 against midoffice13's D)

D14, `--bias --bias-horizons 1,5,11,21 --bias-guard 5` at the defaults, through the runner, 16:57
to 17:08 EST: 11.0 minutes, the guard passing on its five dates (the inputs rebuilt from the panel
cut at each date equal the full run's through it to the bit). Every statistic recomputed from
the stored forecasts and outcomes by the library's `bias_statistics` (`compare_bias.txt`).

**The minimum-variance portfolios first,** at one day; the volatility a year, the forecast at its
median and the realised from the returns:

| | | D13 bias | forecast | realised | D14 bias | forecast | realised |
|---|---|---|---|---|---|---|---|
| long-only | whole test | 1.167 | 9.42% | 10.93% | 1.059 | 9.42% | 9.92% |
| long-only | first half | 1.246 | 8.29% | 11.47% | 1.247 | 8.28% | 11.47% |
| long-only | second half | 1.084 | 9.55% | 10.38% | 0.831 | 9.54% | 8.09% |
| unconstrained | whole test | 1.220 | 9.13% | 11.07% | 1.101 | 9.12% | 10.01% |
| unconstrained | first half | 1.297 | 8.05% | 11.56% | 1.298 | 8.05% | 11.57% |
| unconstrained | second half | 1.139 | 9.30% | 10.57% | 0.862 | 9.29% | 8.16% |

- **The -13 sigma day is gone.** D13's three lowest long-only outcomes were 2026-09-30 (-13.0
  sd), 2025-04-03 (-9.4), 2024-08-02 (-4.8); D14's are 2025-04-03, 2024-08-02 and 2024-12-17
  (-4.4). The unconstrained portfolio's likewise (-13.9 gone).
- **The whole test now reads as midoffice13's sensitivity without that day:** without it D13
  read 1.055 and 1.097 (realised 9.89% and 9.97%); D14 reads 1.059 and 1.101 with it, 1.059 and
  1.102 without (9.92%, 10.01%). The other rulings move the optimiser by 0.005 at most.
- **The forecasts barely move** (9.42% and 9.12% at the median, both runs): the fix is in the
  outcomes, one day's -83.8% gone from the realised returns. Realised volatility falls 1.0 point
  a year, to 0.5 and 0.9 above the forecast.
- **At the other horizons,** the windows overlapping: long-only 1.173 to 1.074 at 5 days, 1.078 to
  0.991 at 11, 1.005 to 0.917 at 21; unconstrained 1.223 to 1.112, 1.110 to 1.011, 1.045 to 0.941.
- **By half the bias is the regime's:** the first half unchanged (1.247, 1.298), the second
  from 1.084 and 1.139 to 0.831 and 0.862 -- over-forecast in the calm half as the first was
  under-forecast.

**What it should leave alone, it does:**

| | D13 | D14 |
|---|---|---|
| the index, cap-weighted: whole / first half / second / 21 days | 1.015 / 1.224 / 0.751 / 0.860 | 1.015 / 1.224 / 0.751 / 0.860 |
| the index, equally weighted | 0.965 / 1.142 / 0.749 / 0.805 | 0.966 / 1.143 / 0.749 / 0.805 |
| smallest 85 | 0.956 / 1.064 / 0.835 / 0.775 | 0.956 / 1.065 / 0.834 / 0.776 |
| largest 50 | 0.996 / 1.175 / 0.778 / 0.873 | 0.996 / 1.175 / 0.778 / 0.873 |
| every member singly, median; mean abs(bias - 1) | 1.003; 0.090 | 1.003; 0.088 |
| the industries, mean abs(bias - 1) | 0.046 | 0.046 |
| random portfolios, long-only / long-short, median | 0.971 / 1.004 | 0.971 / 1.004 |
| eigenportfolios 1-5 / 16-20 / 21-25, mean | 1.030 / 1.216 / 1.194 | 1.030 / 1.217 / 1.194 |

**The parents:** each with an ex-date in the test, its own outcome that day in forecast sd, D13
to D14 -- J 2024-09-30 -7.8 to +2.4 (2025-05-16 +0.1 to +0.6); LEN 2025-01-21 -1.6 to +1.9; WDC
-11.5 to -2.0; CMCSA -2.7 to +0.4; BDX 2026-02-10 -10.0 to +2.4; APTV -4.4 to +1.2; FDX -9.5 to
+0.2; HON -1.0 to -1.9; SPGI +1.0 to +3.8; CTVA -41.9 to +2.0. Their member statistics move most
of any member's: CTVA 1.828 to 0.874; K 0.787 to 0.830, BWA 0.808 to 0.831 and DHR 1.080 to 1.103,
whose ex-dates fell before the test and whose histories changed; FDX 1.081 to 1.041, J 0.779 to
0.748, BDX 1.043 to 1.016, WDC 1.166 to 1.144.

The arrays: the capitalisations differ (HON's from 2026-06-29, FDXF's and VYLR's), and so the
long-only portfolio's size on some dates; the unconstrained one's sizes and the PSD repair (none)
are the same.

---

## What I would do next, and why

1. **The volatility regime adjustment, next, as ruled** (midoffice13's ruling 3). With the CTVA
   day gone the optimiser's whole-test bias is no longer one data error's, and what is left is
   the level: 1.25 and 1.30 in the first half, 0.83 and 0.86 in the second, the cap-weighted
   index 1.22 then 0.75, every portfolio with it. A shape adjustment cannot move a level; USE4's
   regime adjustment -- the factor covariance scaled by the recent cross-sectional bias of the
   factors' standardised returns, on a short half-life -- is the step built for it, and the
   realised-volatility line now stands beside the statistic to judge it by.
2. **Then the eigenvalue adjustment one-sided, on the regime-adjusted model** (ruling 3): its test
   is now clean of the CTVA day, so a first-half choice and a second-half check mean what they
   say.
3. **Jeff's six decisions,** above all the five rulings from the public record and Honeywell's
   count. Until ruled they stand as built, each reversible in its file (a ruling removed, or
   `--spin-offs none`).
4. **Keep the first-day check standing.** It is how the next CTVA shows itself: a child joins the
   index, and on its first day its parent's gap stands unexplained.

---

## Key decisions

1. **The ruling carries the total return, not the prices.** The stitched prices stay Polygon's
   adjusted closes, which carry a spin-off only where Polygon records one; the brief defines the
   total return, which the run uses. Price returns (`--returns price`) still show the unrecorded
   days as losses (a decision for Jeff).
2. **The formula in the build** is (w (C_t + D_t) + k V) / C_s, w the applied splits between: it
   is the brief's exactly with no split, and with a split the same day counts the parent's new
   shares as the adjusted closes do. On every other day the old expression is kept, so a build
   without rulings is `_listed` to the bit.
3. **"Carried" means within 10% of the brief's value** -- the build's own tolerance for counts
   and gaps: seven of Polygon's eight fractional ratios are (-7.0% to +6.7%), and their
   ex-date returns stand within 1.23 points of the brief's; Lennar's is not (-67%).
4. **Lennar's value at Millrose's first close,** the one ruling whose child had no ex-date price:
   the record date was its ex-date, Millrose trading from 02-07.
5. **Five rulings beyond the brief's list,** from the public record, each a member's spin-off on
   its ex-date: the brief makes the data and the record the authority and its list not
   exhaustive. Each says so in its `ruled` field, for Jeff's review.
6. **Honeywell's reverse split applied as a split, its count ruling superseded:** the record and
   the samples agree, and the build's own split machinery checks it (the split guard, the stale
   rule, the count moves). The superseded entry stays in its file and `--spin-offs none` reads
   it, so `_listed` rebuilds to the bit.
7. **FDXF's count as ruled** -- the shares distributed -- with the retained stake and the
   weighted check's flag reported rather than a gross-up chosen.
8. **The ex-dates from the data and the record:** the child's first regular-way day and the
   parent's gap, ZimVie's corrected to 2022-03-01; BD's Waters ratio the releases' rounded 0.135.
9. **The new panel is `_spinoffs`,** built into `data/` beside `_listed`, which is kept and
   reproducible; first built into scratch and compared, then into `data/`, the two equal in all
   20 arrays.
10. **The first-day check is a standing check** of the build's report, not a scratch script: a
    future CTVA shows there the day its child joins the index.
11. **The correlations' default lags live in the run**, as midoffice11's and 12's defaults did;
    the estimator keeps its own default.
12. **The cache gained 19 answers,** the children's bars and records the rulings' evidence needs;
    nothing in `data/` was overwritten or deleted.

## Decisions left for Jeff

1. **The five rulings from the public record** (Jacobs' second Amentum distribution, BD's Waters
   shares, Aptiv's Versigent, Honeywell's Aerospace, S&P Global's Mobility): kept (my choice), or
   dropped.
2. **Honeywell:** midoffice08's count ruling superseded by the 1-for-2 reverse split of
   2026-06-29 (my choice); or the ruling restored.
3. **FedEx Freight's count:** the shares distributed, as ruled, 0.80 of its weighted count; or
   grossed up by FedEx's 19.9% (about 148.9 million).
4. **Lennar:** at Millrose's first close (+3.51% on 2025-01-21, my choice, right across the
   distribution in total); or left as Polygon has it (-2.83%), its total return across the
   distribution 6.5% short.
5. **The seven Polygon carries:** left (my choice); or every spin-off by the brief's formula, the
   largest change 1.23 points (FTV's).
6. **Price returns:** left as Polygon's adjusted closes (my choice, the run's returns are total);
   or the rulings applied to the prices too, as price factors.

---

## Tests

**The suite:** 2,362 passed, 1 skipped, 19 warnings (94.8 s in the final checks), with
`--ignore=tests/test_market_data.py`. Before this piece 2,356: 6 new tests, and every old one
passes unchanged -- `small_panel` gained two parameters, both defaulting to what it did.

**New, in `tests/midoffice/test_securities.py`,** 6:

- `test_a_spin_off_ruling_carries_the_childs_shares_in_the_parents_total_return` -- CTVA's case:
  the ex-date's return the brief's, with the day's cash; a price-only event the cache records
  there does not enter it; the stitched prices and the other days unchanged; the use recorded;
- `test_a_ruling_on_a_split_day_takes_the_split_as_the_adjusted_closes_do` -- w on a two-for-one's
  day;
- `test_a_rulings_split_is_applied_to_the_count_as_the_gates_are_when_its_samples_follow` --
  HON's case: the split check from the parent's samples, applied on the ex-date, the split guard
  holding, its w in the return; counts that do not follow it refused;
- `test_a_child_without_a_count_takes_its_parents_on_the_record_date_until_its_own` -- Vylor's
  case: seeded from the ex-date until its own first count, never back, listed; a child no column
  holds refused;
- `test_the_spin_off_file_is_read_with_its_rules` -- the reader's refusals, and the repo's file:
  CTVA's ruling with Vylor's count, HON's with its split and the override it supersedes;
- `test_the_first_day_check_lists_the_members_gaps_and_what_explains_them` -- a gap listed, and
  marked by a ruling, a split event or a distribution.

**Measured, not unit-tested:** the run script's ruling 4 (R4 equal to V21N to the bit) and
ruling 5 (`ruling5_check.txt`, its own functions on a stored run); the build's (`_listed` rebuilt
to the bit by `--spin-offs none`; item 1's panel; the full panel into scratch and into `data/`).

**Hygiene:** every changed file parses, is ASCII and keeps its line endings (`final_checks.txt`).
**ruff:** clean over every changed Python file.

---

## Failures and debugging trail

- **The total-return index moved in its last bits.** The first rebuild of `_listed` with the new
  code differed from the stored one in 656,657 cells of `total_return_prices` (its daily returns
  under 1e-12 apart): the ruling's branch had computed `total[s] * (step)` for every day, where
  the old code computed `total[s] * (p_t + r) / p_s`. The old expression is kept on every day
  without a ruling; the rebuild then equals `_listed` in all 20 arrays.
- **A heredoc failed twice:** a test block with apostrophes (bash: unexpected EOF), and a patch
  with `\n` escapes (the assertion failed before anything was written) -- the state file's
  working note. The Write tool and the Edit tool did both.
- **The appended test block was LF in a CRLF file,** caught by the hygiene script (MIXED); the
  file was normalised to CRLF. An earlier `grep -q $'\r'` check had misread every file as LF.
- **The capture of the run script in a scratch check** needed the module registered in
  `sys.modules` before `exec_module` (a dataclass looks its module up).
- **ruff:** one f-string without a placeholder (F541), and an unused variable (F841); both fixed.
- **ZimVie's ex-date** was 2022-03-02 from memory; the record and the data say 03-01 (ZIMV's first
  regular-way bar). The events were corrected before the rulings were written; BD's record date
  likewise (2022-03-22, not 03-18), which no number depends on.
- **The runs:** none failed. R4 9.2 minutes, D14 11.0 minutes with the guard on
  (`bias/runner_log.txt`); each started with over 5 GB available.

## Problems found, not fixed

1. **Price returns carry no unrecorded spin-off:** `--returns price` on `_spinoffs` still has
   CTVA's -83.8% day and the rest.
2. **Children's first days without a count:** AMTM 5, CEG 1, GEHC 1, GEV 4, Q 13, SOLS 13, SOLV
   1, VLTO 2, WBD 1 member-days before their first records; the brief's rule seeds Vylor's and
   FedEx Freight's alone, and would fill these the same way.
3. **FDXF's seeded count** is the shares distributed, 0.80 of its outstanding count.
4. **Lennar's day** stands 4.8 points above its peers, Millrose's value on 2025-01-21 unobserved.
5. **S&P Global's day** reads +7.10%: SPGI opened only 1.9% down without Mobility's $20.30 and
   rose through the day; the formula's, as the data have it, and plausibly news.
6. **Coverage:** the first-day check sees only children that joined the index; the lists of
   completed spin-offs are not exhaustive (one omits Constellation). A child that never joined
   and that no list names would be missed.
7. **The state file's "86" count moves was stale:** 86 is `_repaired`'s, 88 `_listed`'s since
   midoffice10; corrected.
8. **As before:** six ruff findings in untouched files at HEAD, and `tests/test_market_data.py`'s
   `../core_repo/src`.

## Scratch

`claude/scratch/midoffice14/` (gitignored):

- **Readers and checks:** `common.py` (the panel and the cache, read only), `events.py` (the
  spin-offs' terms), `pairs.py`, `firstdays.py`, `check_events.py`, `fetch_children.py` (the
  network fetch into the cache), `ruling5_check.py`, `compare_panels.py`, `compare_bias.py`.
- **Writers:** `make_spin_offs.py` (the rulings file, and item 1's CTVA-only one in `panels/`),
  `tests_block.py` (the new tests, appended), `runner.py`, `final_checks.sh`, the retro's
  `append_appendix.py`.
- **Panels, `panels/`:** `_listed` rebuilt, `_ctva` (item 1), `_spinoffs` (item 2), each with its
  build output and report; `final/`, the final checks' builds and captures.
- **Model, `model/`:** M14 (the defaults on `_spinoffs`), its report and pickle.
- **Bias runs, `bias/`:** R4 and D14, each a report, progress and `.npz`; the runner's log.
- **Outputs:** `pairs.txt`, `firstdays.txt`, `fetch_children.txt`, `check_events.txt`,
  `make_spin_offs.txt`, `ruling5_check.txt`, `equal_R4.txt`, `report_R4_vs_V21N.txt`,
  `panel_listed_rebuilt.txt`, `panel_item1.txt`, `panel_item2.txt`, `panel_changed.txt`,
  `w_listed.txt`, `w_spinoffs.txt`, `model_compare.txt`, `compare_bias.txt`, `final_checks.txt`.

The midoffice13 harnesses were used where they are: `capture.py`, `compare.py`, `equal_npz.py`,
`hygiene.py`. The key outputs are copied into the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

## Open items for Chat

- **The six decisions above,** the five rulings from the record and Honeywell's first among
  them.
- **The next piece,** by midoffice13's ruling 3: the volatility regime adjustment on the rebuilt
  panel, then the eigenvalue adjustment one-sided on the regime-adjusted model.
- **The bias statistics are standing,** the realised volatility beside the optimiser's now;
  every candidate is one run against D14, through the runner.

---

## Appendix: outputs, verbatim

Each output as its script wrote it, from `claude/scratch/midoffice14/`.

### check_events.txt -- each spin-off by an index member against the cache and the panel (`_listed`): the panel's ex-date return against the brief's, Polygon's ratio against the brief's value

```text
Each spin-off by an index member, 2022-01-03 to 2026-11-03: the panel's ex-date total return against the brief's formula

EXC -> CEG, ex 2022-02-02 (record 2022-01-20, distributed 2022-02-01; brief; record); column EXC BBG000J6XT05, a member that day: True
  raw 57.83 on 2022-02-01 -> 42.86; the child's close 53.01 on 2022-02-02; k 0.333333
  the cache: no split record on the ex-date
  total return: the panel's -25.8862%; the brief's +4.6689%; the panel less the brief's -30.56 points
  the ratio the prices imply, SPY's day +0.97% taken out: 0.2930, against the record's 0.3333 (0.879 of it)

ZBH -> ZIMV, ex 2022-03-01 (record 2022-02-15, distributed 2022-03-01; brief; record); column ZBH BBG000BKPL53, a member that day: True
  raw 127.19 on 2022-02-28 -> 124.14; the child's close 25.53 on 2022-03-01; k 0.100000
  the cache: no split record on the ex-date
  total return: the panel's -2.3980%; the brief's -0.3908%; the panel less the brief's -2.01 points
  the ratio the prices imply, SPY's day -1.52% taken out: 0.0436, against the record's 0.1000 (0.436 of it)

BDX -> EMBC, ex 2022-04-01 (record 2022-03-22, distributed 2022-04-01; brief; record); column BDX BBG000BCZYD3, a member that day: True
  raw 266.00 on 2022-03-31 -> 262.50; the child's close 30.50 on 2022-04-01; k 0.200000
  the cache: no split record on the ex-date
  total return: the panel's -1.3158%; the brief's +0.9774%; the panel less the brief's -2.29 points
  the ratio the prices imply, SPY's day +0.28% taken out: 0.1395, against the record's 0.2000 (0.697 of it)

T -> WBD, ex 2022-04-11 (record 2022-04-05, distributed 2022-04-08; brief; record); column T BBG000BSJK37, a member that day: True
  raw 24.14 on 2022-04-08 -> 19.63; the child's close 24.78 on 2022-04-11; k 0.241917
  the cache: no split record on the ex-date
  total return: the panel's -18.6827%; the brief's +6.1504%; the panel less the brief's -24.83 points
  the ratio the prices imply, SPY's day -1.71% taken out: 0.1654, against the record's 0.2419 (0.684 of it)

FBHS -> MBC, ex 2022-12-15 (record 2022-12-02, distributed 2022-12-14; brief; record); column FBIN BBG001B4BV87, a member that day: True
  raw 61.90 on 2022-12-14 -> 57.62; the child's close 7.70 on 2022-12-15; k 1.000000
  the cache: no split record on the ex-date
  total return: the panel's -6.9144%; the brief's +5.5250%; the panel less the brief's -12.44 points
  the ratio the prices imply, SPY's day -2.45% taken out: 0.3592, against the record's 1.0000 (0.359 of it)

GE -> GEHC, ex 2023-01-04 (record 2022-12-16, distributed 2023-01-03; brief; record); column GE BBG000BK6MB5, a member that day: True
  raw 84.98 on 2023-01-03 -> 70.20; the child's close 60.49 on 2023-01-04; k 0.333333
  the cache: split record(s) [(1000, 1281)], ratio 1.2810 on the ex-date
  total return: the panel's +5.8203%; the brief's +6.3348%; the panel less the brief's -0.51 points
  the ratio the prices imply, SPY's day +0.77% taken out: 0.2552, against the record's 0.3333 (0.766 of it)
  Polygon's factor 1.2810 against the brief's (C_t + kV_t) / C_t = 1.2872: the distribution's value 19.73 a share against 20.16, -2.2%

LH -> FTRE, ex 2023-07-03 (record 2023-06-20, distributed 2023-06-30; brief; record); column LH BBG01MMT6PL7, a member that day: True
  raw 241.33 on 2023-06-30 -> 208.92; the child's close 36.84 on 2023-07-03; k 1.000000
  the cache: split record(s) [(1000, 1164)], ratio 1.1640 on the ex-date
  total return: the panel's +0.7678%; the brief's +1.8357%; the panel less the brief's -1.07 points
  the ratio the prices imply, SPY's day +0.12% taken out: 0.8873, against the record's 1.0000 (0.887 of it)
  Polygon's factor 1.1640 against the brief's (C_t + kV_t) / C_t = 1.1763: the distribution's value 34.26 a share against 36.84, -7.0%

BWA -> PHIN, ex 2023-07-05 (record 2023-06-23, distributed 2023-07-03; brief; record); column BWA BBG000BJ49H3, a member that day: True
  raw 49.92 on 2023-07-03 -> 43.18; the child's close 36.75 on 2023-07-05; k 0.200000
  the cache: no split record on the ex-date
  total return: the panel's -13.5016%; the brief's +1.2220%; the panel less the brief's -14.72 points
  the ratio the prices imply, SPY's day -0.15% taken out: 0.1814, against the record's 0.2000 (0.907 of it)

DHR -> VLTO, ex 2023-10-02 (record 2023-09-13, distributed 2023-09-30; brief; record); column DHR BBG000BH3JF8, a member that day: True
  raw 248.10 on 2023-09-29 -> 213.74; the child's close 85.12 on 2023-10-02; k 0.333333
  the cache: no split record on the ex-date
  total return: the panel's -13.8493%; the brief's -2.4130%; the panel less the brief's -11.44 points
  the ratio the prices imply, SPY's day -0.04% taken out: 0.4025, against the record's 0.3333 (1.208 of it)

K -> KLG, ex 2023-10-02 (record 2023-09-21, distributed 2023-10-02; brief; record); column K BBG000BMKDM3, a member that day: True
  raw 59.51 on 2023-09-29 -> 52.50; the child's close 13.35 on 2023-10-02; k 0.250000
  the cache: no split record on the ex-date
  total return: the panel's -11.7795%; the brief's -6.1712%; the panel less the brief's -5.61 points
  the ratio the prices imply, SPY's day -0.04% taken out: 0.5233, against the record's 0.2500 (2.093 of it)

MMM -> SOLV, ex 2024-04-01 (record 2024-03-18, distributed 2024-04-01; brief; record); column MMM BBG000BP52R2, a member that day: True
  raw 106.07 on 2024-03-28 -> 94.02; the child's close 69.10 on 2024-04-01; k 0.250000
  the cache: split record(s) [(1000, 1196)], ratio 1.1960 on the ex-date
  total return: the panel's +6.0129%; the brief's +4.9260%; the panel less the brief's +1.09 points
  the ratio the prices imply, SPY's day -0.17% taken out: 0.1717, against the record's 0.2500 (0.687 of it)
  Polygon's factor 1.1960 against the brief's (C_t + kV_t) / C_t = 1.1837: the distribution's value 18.43 a share against 17.27, +6.7%

GE -> GEV, ex 2024-04-02 (record 2024-03-19, distributed 2024-04-02; brief; record); column GE BBG000BK6MB5, a member that day: True
  raw 175.36 on 2024-04-01 -> 136.47; the child's close 140.00 on 2024-04-02; k 0.250000
  the cache: split record(s) [(1000, 1253)], ratio 1.2530 on the ex-date
  total return: the panel's -2.4881%; the brief's -2.2183%; the panel less the brief's -0.27 points
  the ratio the prices imply, SPY's day -0.64% taken out: 0.2698, against the record's 0.2500 (1.079 of it)
  Polygon's factor 1.2530 against the brief's (C_t + kV_t) / C_t = 1.2565: the distribution's value 34.53 a share against 35.00, -1.4%

J -> AMTM, ex 2024-09-30 (record 2024-09-23, distributed 2024-09-27; brief; record); column J BBG019C1BQR4, a member that day: True
  raw 155.53 on 2024-09-27 -> 130.90; the child's close 32.25 on 2024-09-30; k 1.000000
  the cache: no split record on the ex-date
  total return: the panel's -15.8362%; the brief's +4.8994%; the panel less the brief's -20.74 points
  the ratio the prices imply, SPY's day +0.40% taken out: 0.7830, against the record's 1.0000 (0.783 of it)

LEN -> MRP, ex 2025-01-21 (record 2025-01-21, distributed 2025-02-07; brief; record); column LEN BBG000BN5HF7, a member that day: True
  raw 141.54 on 2025-01-17 -> 133.14; the child's close 26.74 on 2025-02-07 (its first; none on the ex-date); k 0.500000
  the cache: split record(s) [(1000, 1033)], ratio 1.0330 on the ex-date
  total return: the panel's -2.8306%; the brief's +3.5114%; the panel less the brief's -6.34 points
  the ratio the prices imply, SPY's day +0.92% taken out: 0.3626, against the record's 0.5000 (0.725 of it)
  Polygon's factor 1.0330 against the brief's (C_t + kV_t) / C_t = 1.1004: the distribution's value 4.39 a share against 13.37, -67.1%

WDC -> SNDK, ex 2025-02-24 (record 2025-02-12, distributed 2025-02-21; brief; record); column WDC BBG000BWNFZ9, a member that day: True
  raw 68.70 on 2025-02-21 -> 49.02; the child's close 48.60 on 2025-02-24; k 0.333333
  the cache: no split record on the ex-date
  total return: the panel's -28.6515%; the brief's -5.0724%; the panel less the brief's -23.58 points
  the ratio the prices imply, SPY's day -0.46% taken out: 0.3986, against the record's 0.3333 (1.196 of it)

J -> AMTM, ex 2025-05-16 (record 2025-05-16, distributed 2025-05-30; record); column J BBG019C1BQR4, a member that day: True
  raw 128.80 on 2025-05-15 -> 129.17; the child's close 21.84 on 2025-05-16; k 0.060835
  the cache: no split record on the ex-date
  total return: the panel's +0.2873%; the brief's +1.3188%; the panel less the brief's -1.03 points
  the ratio the prices imply, SPY's day +0.63% taken out: 0.0204, against the record's 0.0608 (0.336 of it)

FTV -> RAL, ex 2025-06-30 (record 2025-06-16, distributed 2025-06-28; brief; record); column FTV BBG00BLVZ228, a member that day: True
  raw 71.60 on 2025-06-27 -> 52.13; the child's close 48.49 on 2025-06-30; k 0.333333
  the cache: split record(s) [(1000, 1327)], ratio 1.3270 on the ex-date
  total return: the panel's -3.3848%; the brief's -4.6182%; the panel less the brief's +1.23 points
  the ratio the prices imply, SPY's day +0.48% taken out: 0.4086, against the record's 0.3333 (1.226 of it)
  Polygon's factor 1.3270 against the brief's (C_t + kV_t) / C_t = 1.3101: the distribution's value 17.05 a share against 16.16, +5.5%

HON -> SOLS, ex 2025-10-30 (record 2025-10-17, distributed 2025-10-30; brief; record); column HON BBG000H556T9, a member that day: True
  raw 212.89 on 2025-10-29 -> 200.11; the child's close 48.74 on 2025-10-30; k 0.250000
  the cache: split record(s) [(1000, 1061)], ratio 1.0610 on the ex-date
  total return: the panel's -0.2693%; the brief's -0.2795%; the panel less the brief's +0.01 points
  the ratio the prices imply, SPY's day -1.10% taken out: 0.2142, against the record's 0.2500 (0.857 of it)
  Polygon's factor 1.0610 against the brief's (C_t + kV_t) / C_t = 1.0609: the distribution's value 12.21 a share against 12.19, +0.2%

DD -> Q, ex 2025-11-03 (record 2025-10-22, distributed 2025-11-01; brief; record); column DD BBG00BN961G4, a member that day: True
  raw 81.65 on 2025-10-31 -> 34.69; the child's close 97.00 on 2025-11-03; k 0.500000
  the cache: split record(s) [(100, 239)], ratio 2.3900 on the ex-date
  total return: the panel's +1.5421%; the brief's +1.8861%; the panel less the brief's -0.34 points
  the ratio the prices imply, SPY's day +0.19% taken out: 0.4857, against the record's 0.5000 (0.971 of it)
  Polygon's factor 2.3900 against the brief's (C_t + kV_t) / C_t = 2.3981: the distribution's value 48.22 a share against 48.50, -0.6%

CMCSA -> VSNT, ex 2026-01-05 (record 2025-12-16, distributed 2026-01-02; brief; record); column CMCSA BBG000BFT2L4, a member that day: True
  raw 29.54 on 2026-01-02 -> 28.13; the child's close 40.57 on 2026-01-05; k 0.040000
  the cache: no split record on the ex-date
  total return: the panel's -4.7732%; the brief's +0.7204%; the panel less the brief's -5.49 points
  the ratio the prices imply, SPY's day +0.67% taken out: 0.0396, against the record's 0.0400 (0.990 of it)

BDX -> WAT, ex 2026-02-10 (record 2026-02-05, distributed 2026-02-09; record); column BDX BBG000BCZYD3, a member that day: True
  raw 207.39 on 2026-02-09 -> 171.68; the child's close 327.02 on 2026-02-10; k 0.135000
  the cache: no split record on the ex-date
  total return: the panel's -17.2188%; the brief's +4.0685%; the panel less the brief's -21.29 points
  the ratio the prices imply, SPY's day -0.26% taken out: 0.1075, against the record's 0.1350 (0.796 of it)

APTV -> VGNT, ex 2026-04-01 (record 2026-03-17, distributed 2026-04-01; record); column APTV CIK0001521332, a member that day: True
  raw 69.44 on 2026-03-31 -> 62.09; the child's close 27.85 on 2026-04-01; k 0.333333
  the cache: no split record on the ex-date
  total return: the panel's -10.5847%; the brief's +2.7842%; the panel less the brief's -13.37 points
  the ratio the prices imply, SPY's day +0.75% taken out: 0.2827, against the record's 0.3333 (0.848 of it)

FDX -> FDXF, ex 2026-06-01 (record 2026-05-15, distributed 2026-06-01; brief; record); column FDX BBG000BJF1Z8, a member that day: True
  raw 411.75 on 2026-05-29 -> 338.49; the child's close 149.53 on 2026-06-01; k 0.500000
  the cache: no split record on the ex-date
  total return: the panel's -17.7923%; the brief's +0.3655%; the panel less the brief's -18.16 points
  the ratio the prices imply, SPY's day +0.27% taken out: 0.4974, against the record's 0.5000 (0.995 of it)

HON -> HONA, ex 2026-06-29 (record 2026-06-15, distributed 2026-06-29; record); column HON BBG000H556T9, a member that day: True
  raw 232.21 on 2026-06-26 -> 227.80; the child's close 220.19 on 2026-06-29; k 0.500000, the parent's split 0.5
  the cache: no split record on the ex-date
  total return: the panel's -1.8991%; the brief's -3.5377%; the panel less the brief's +1.64 points
  the ratio the prices imply, SPY's day +1.65% taken out: 0.5547, against the record's 0.5000 (1.109 of it)

SPGI -> MBGL, ex 2026-07-01 (record 2026-06-15, distributed 2026-07-01; record); column SPGI BBG000BP1Q11, a member that day: True
  raw 407.26 on 2026-06-30 -> 414.97; the child's close 21.19 on 2026-07-01; k 1.000000
  the cache: no split record on the ex-date
  total return: the panel's +1.8931%; the brief's +7.0962%; the panel less the brief's -5.20 points
  the ratio the prices imply, SPY's day -0.14% taken out: -0.3898, against the record's 1.0000 (-0.390 of it)

CTVA -> VYLR, ex 2026-10-01 (record 2026-09-24, distributed 2026-10-01; brief; record); column CTVA BBG00BN969C1, a member that day: True
  raw 77.65 on 2026-09-30 -> 12.57; the child's close 68.26 on 2026-10-01; k 1.000000
  the cache: no split record on the ex-date
  total return: the panel's -83.8120%; the brief's +4.0953%; the panel less the brief's -87.91 points
  the ratio the prices imply, SPY's day +0.18% taken out: 0.9554, against the record's 1.0000 (0.955 of it)

Parents no member on the ex-date -- nothing needed (the brief):
  ILMN -> GRAL, ex 2024-06-25: ILMN a member that day: False, the day before: False
  FLEX -> NXT, ex 2024-01-03: FLEX a member that day: False, the day before: False

Summary: event, member, the panel's return, the brief's, the difference, Polygon's ratio
  2022-02-02 EXC   CEG   member True  panel  -25.89% brief   +4.67% diff  -30.56  no record
  2022-03-01 ZBH   ZIMV  member True  panel   -2.40% brief   -0.39% diff   -2.01  no record
  2022-04-01 BDX   EMBC  member True  panel   -1.32% brief   +0.98% diff   -2.29  no record
  2022-04-11 T     WBD   member True  panel  -18.68% brief   +6.15% diff  -24.83  no record
  2022-12-15 FBHS  MBC   member True  panel   -6.91% brief   +5.53% diff  -12.44  no record
  2023-01-04 GE    GEHC  member True  panel   +5.82% brief   +6.33% diff   -0.51  recorded 1.2810
  2023-07-03 LH    FTRE  member True  panel   +0.77% brief   +1.84% diff   -1.07  recorded 1.1640
  2023-07-05 BWA   PHIN  member True  panel  -13.50% brief   +1.22% diff  -14.72  no record
  2023-10-02 DHR   VLTO  member True  panel  -13.85% brief   -2.41% diff  -11.44  no record
  2023-10-02 K     KLG   member True  panel  -11.78% brief   -6.17% diff   -5.61  no record
  2024-04-01 MMM   SOLV  member True  panel   +6.01% brief   +4.93% diff   +1.09  recorded 1.1960
  2024-04-02 GE    GEV   member True  panel   -2.49% brief   -2.22% diff   -0.27  recorded 1.2530
  2024-09-30 J     AMTM  member True  panel  -15.84% brief   +4.90% diff  -20.74  no record
  2025-01-21 LEN   MRP   member True  panel   -2.83% brief   +3.51% diff   -6.34  recorded 1.0330
  2025-02-24 WDC   SNDK  member True  panel  -28.65% brief   -5.07% diff  -23.58  no record
  2025-05-16 J     AMTM  member True  panel   +0.29% brief   +1.32% diff   -1.03  no record
  2025-06-30 FTV   RAL   member True  panel   -3.38% brief   -4.62% diff   +1.23  recorded 1.3270
  2025-10-30 HON   SOLS  member True  panel   -0.27% brief   -0.28% diff   +0.01  recorded 1.0610
  2025-11-03 DD    Q     member True  panel   +1.54% brief   +1.89% diff   -0.34  recorded 2.3900
  2026-01-05 CMCSA VSNT  member True  panel   -4.77% brief   +0.72% diff   -5.49  no record
  2026-02-10 BDX   WAT   member True  panel  -17.22% brief   +4.07% diff  -21.29  no record
  2026-04-01 APTV  VGNT  member True  panel  -10.58% brief   +2.78% diff  -13.37  no record
  2026-06-01 FDX   FDXF  member True  panel  -17.79% brief   +0.37% diff  -18.16  no record
  2026-06-29 HON   HONA  member True  panel   -1.90% brief   -3.54% diff   +1.64  no record
  2026-07-01 SPGI  MBGL  member True  panel   +1.89% brief   +7.10% diff   -5.20  no record
  2026-10-01 CTVA  VYLR  member True  panel  -83.81% brief   +4.10% diff  -87.91  no record
```

### firstdays.txt -- every security's first day in the panel against the members' raw opening gaps that day

```text
21 securities whose first priced day is inside the window, after 2022-01-03; on each, the members ranked by their raw opening gap less SPY's

2022-02-02  CEG (Constellation Energy Corporation Common Stock, CEG): first priced, close 53.01; a member from 2022-02-02
    EXC    open gap less SPY's -29.56%, day -26.86%; total return -25.89%; raw 57.83 -> 41.06 -> 42.86
    PYPL   open gap less SPY's -20.99%, day -25.56%; total return -24.59%; raw 175.80 -> 139.89 -> 132.57
    CHRW   open gap less SPY's -11.39%, day -12.48%; total return -11.50%; raw 104.57 -> 93.25 -> 92.54
    WM     open gap less SPY's -5.09%, day -2.07%; total return -1.10%; raw 149.76 -> 142.98 -> 148.12
    GILD   open gap less SPY's -4.90%, day -4.84%; total return -3.87%; raw 68.47 -> 65.50 -> 65.82

2022-04-11  WBD (Warner Bros. Discovery, Inc. Series A Common Stock, WBD): first priced, close 24.78; a member from 2022-04-11
    T      open gap less SPY's -20.98%, day -16.97%; total return -18.68%; raw 24.14 -> 18.89 -> 19.63
    TSLA   open gap less SPY's -3.62%, day -3.12%; total return -4.83%; raw 1025.49 -> 980.40 -> 975.93
    NVDA   open gap less SPY's -3.14%, day -3.49%; total return -5.20%; raw 231.19 -> 222.14 -> 219.17
    FBIN   open gap less SPY's -2.30%, day -0.40%; total return -2.11%; raw 72.89 -> 70.65 -> 71.35
    MRNA   open gap less SPY's -2.24%, day +1.19%; total return -0.52%; raw 160.84 -> 156.00 -> 160.00

2022-12-15  MBC (MasterBrand, Inc., MBC): first priced, close 7.70; a member from 2022-12-15
    WDC    open gap less SPY's -4.50%, day -7.66%; total return -10.10%; raw 35.83 -> 33.76 -> 32.21
    NFLX   open gap less SPY's -3.67%, day -6.18%; total return -8.63%; raw 317.83 -> 302.12 -> 290.41
    TROW   open gap less SPY's -2.31%, day -3.14%; total return -4.57%; raw 118.58 -> 114.33 -> 111.96; the cache: distribution 2022-12-15
    SYY    open gap less SPY's -2.25%, day -2.24%; total return -4.69%; raw 82.14 -> 79.24 -> 78.29
    EL     open gap less SPY's -2.16%, day -1.01%; total return -3.46%; raw 249.22 -> 240.65 -> 240.60

2023-01-04  GEHC (GE HealthCare Technologies Inc. Common Stock, GEHC): first priced, close 60.49; a member from 2023-01-04
    GE     open gap less SPY's -20.12%, day -18.16%; total return +5.82%; raw 84.98 -> 68.41 -> 70.20; the cache: split 2023-01-04
    MSFT   open gap less SPY's -3.67%, day -5.15%; total return -4.37%; raw 239.58 -> 232.28 -> 229.10
    CHRW   open gap less SPY's -3.60%, day -0.20%; total return +0.58%; raw 90.28 -> 87.59 -> 90.80
    APA    open gap less SPY's -3.39%, day -4.14%; total return -3.37%; raw 43.62 -> 42.41 -> 42.15
    HON    open gap less SPY's -3.33%, day -2.76%; total return -1.99%; raw 214.30 -> 208.50 -> 210.04

2023-05-04  KVUE (Kenvue Inc., KVUE): first priced, close 26.90; a member from 2023-08-25
    PARA   open gap less SPY's -17.12%, day -27.64%; total return -28.35%; raw 22.89 -> 18.91 -> 16.40
    KEY    open gap less SPY's -7.60%, day -5.59%; total return -6.30%; raw 9.53 -> 8.78 -> 8.93
    QCOM   open gap less SPY's -7.53%, day -4.83%; total return -5.54%; raw 112.83 -> 104.03 -> 106.58
    ANSS   open gap less SPY's -6.10%, day -0.92%; total return -1.63%; raw 307.99 -> 288.37 -> 302.96
    ZTS    open gap less SPY's -5.21%, day +1.12%; total return +0.42%; raw 177.73 -> 168.00 -> 178.47

2023-09-12  TKO (TKO Group Holdings, Inc., TKO): first priced, close 103.05; a member from 2025-03-24
    ORCL   open gap less SPY's -11.21%, day -12.95%; total return -13.50%; raw 126.71 -> 112.08 -> 109.61
    RTX    open gap less SPY's -2.53%, day -1.19%; total return -1.74%; raw 76.90 -> 74.70 -> 75.56
    UPS    open gap less SPY's -1.85%, day -2.14%; total return -2.69%; raw 160.89 -> 157.37 -> 156.57
    CMA    open gap less SPY's -1.48%, day -1.29%; total return -1.84%; raw 46.32 -> 45.48 -> 45.47
    ADSK   open gap less SPY's -1.32%, day -2.62%; total return -3.17%; raw 218.74 -> 215.13 -> 211.80

2023-09-25  CRH (CRH Public Limited Company, CRH): first priced, close 56.35; a member from 2025-12-22
    CCL    open gap less SPY's -3.27%, day -1.49%; total return -1.07%; raw 14.05 -> 13.55 -> 13.90
    HPQ    open gap less SPY's -1.91%, day -2.18%; total return -1.76%; raw 26.77 -> 26.18 -> 26.30
    LVS    open gap less SPY's -1.85%, day -1.40%; total return -0.98%; raw 45.79 -> 44.81 -> 45.34
    NCLH   open gap less SPY's -1.50%, day -3.49%; total return -3.07%; raw 15.66 -> 15.38 -> 15.18
    WYNN   open gap less SPY's -1.41%, day -1.41%; total return -0.99%; raw 91.56 -> 90.00 -> 90.65

2023-10-02  VLTO (Veralto Corporation, VLTO): first priced, close 85.12; a member from 2023-10-02
    DHR    open gap less SPY's -11.52%, day -13.81%; total return -13.85%; raw 248.10 -> 219.02 -> 213.74
    K      open gap less SPY's -6.20%, day -11.74%; total return -11.78%; raw 59.51 -> 55.70 -> 52.50
    NSC    open gap less SPY's -3.74%, day -2.72%; total return -2.76%; raw 196.93 -> 189.17 -> 191.50
    SEDG   open gap less SPY's -3.20%, day -5.33%; total return -5.37%; raw 129.51 -> 125.11 -> 122.56
    TSLA   open gap less SPY's -1.96%, day +0.59%; total return +0.55%; raw 250.22 -> 244.81 -> 251.60

2024-03-21  RDDT (Reddit, Inc., RDDT): first priced, close 50.44; a member from 2026-08-18
    DRI    open gap less SPY's -6.20%, day -6.83%; total return -6.50%; raw 174.58 -> 164.73 -> 163.24
    ACN    open gap less SPY's -4.88%, day -9.64%; total return -9.31%; raw 380.44 -> 363.99 -> 345.03
    FDS    open gap less SPY's -3.35%, day -7.96%; total return -7.63%; raw 487.90 -> 474.26 -> 450.66
    CTSH   open gap less SPY's -2.78%, day -3.75%; total return -3.42%; raw 75.15 -> 73.48 -> 72.58
    EPAM   open gap less SPY's -2.47%, day -3.35%; total return -3.02%; raw 292.55 -> 286.95 -> 283.71

2024-04-01  SOLV (Solventum Corporation, SOLV): first priced, close 69.10; a member from 2024-04-01
    MMM    open gap less SPY's -14.31%, day -11.19%; total return +6.01%; raw 106.07 -> 91.05 -> 94.02; the cache: split 2024-04-01
    LW     open gap less SPY's -2.92%, day -1.28%; total return -1.45%; raw 106.53 -> 103.57 -> 104.98
    T      open gap less SPY's -2.47%, day -0.39%; total return -0.57%; raw 17.60 -> 17.19 -> 17.50
    FDX    open gap less SPY's -2.04%, day -3.14%; total return -3.32%; raw 289.74 -> 284.24 -> 280.13
    CHRW   open gap less SPY's -2.00%, day -3.50%; total return -3.68%; raw 76.14 -> 74.73 -> 73.34

2024-04-02  GEV (GE Vernova Inc., GEV): first priced, close 140.00; a member from 2024-04-02
    GE     open gap less SPY's -19.11%, day -21.54%; total return -2.49%; raw 175.36 -> 140.53 -> 136.47; the cache: split 2024-04-02
    HUM    open gap less SPY's -9.48%, day -12.77%; total return -13.41%; raw 351.45 -> 315.49 -> 304.33
    UNH    open gap less SPY's -5.40%, day -5.81%; total return -6.44%; raw 489.70 -> 459.60 -> 458.14
    TSLA   open gap less SPY's -5.22%, day -4.27%; total return -4.90%; raw 175.22 -> 164.75 -> 166.63
    CVS    open gap less SPY's -5.13%, day -6.58%; total return -7.21%; raw 79.56 -> 74.88 -> 73.82

2024-07-08  SW (Smurfit Westrock plc, SW): first priced, close 46.22; a member from 2024-07-08
    NOW    open gap less SPY's -5.47%, day -5.11%; total return -4.99%; raw 806.47 -> 763.50 -> 766.20
    TSLA   open gap less SPY's -1.66%, day +0.45%; total return +0.56%; raw 251.52 -> 247.71 -> 252.94
    CSX    open gap less SPY's -1.53%, day +0.15%; total return +0.27%; raw 33.30 -> 32.84 -> 33.39
    NSC    open gap less SPY's -1.40%, day -0.86%; total return -0.75%; raw 215.86 -> 213.14 -> 214.25
    INTU   open gap less SPY's -1.37%, day -1.49%; total return -1.37%; raw 668.75 -> 660.58 -> 659.58

2024-08-01  FERG (Ferguson Enterprises Inc., FERG): first priced, close 216.33; a member from 2026-08-05
    MRNA   open gap less SPY's -16.79%, day -19.60%; total return -21.01%; raw 119.22 -> 99.58 -> 94.17
    WDC    open gap less SPY's -10.33%, day -8.31%; total return -9.72%; raw 67.05 -> 60.34 -> 60.53
    IR     open gap less SPY's -8.69%, day -7.72%; total return -9.13%; raw 100.40 -> 92.00 -> 91.23
    LRCX   open gap less SPY's -7.54%, day -8.45%; total return -9.87%; raw 921.24 -> 854.75 -> 830.33
    MGM    open gap less SPY's -7.11%, day -11.80%; total return -13.22%; raw 42.97 -> 40.05 -> 37.29

2024-09-30  AMTM (Amentum Holdings, Inc., AMTM): first priced, close 32.25; a member from 2024-09-30
    J      open gap less SPY's -15.47%, day -16.24%; total return -15.84%; raw 155.53 -> 131.18 -> 130.90
    APTV   open gap less SPY's -3.36%, day -3.40%; total return -3.00%; raw 74.24 -> 71.61 -> 72.01
    GM     open gap less SPY's -3.17%, day -3.93%; total return -3.53%; raw 46.48 -> 44.92 -> 44.84
    NXPI   open gap less SPY's -2.89%, day -2.41%; total return -2.01%; raw 244.93 -> 237.39 -> 240.01
    BAX    open gap less SPY's -2.80%, day -2.82%; total return -2.42%; raw 38.91 -> 37.75 -> 37.97

2025-02-24  SNDK (Sandisk Corporation Common Stock, SNDK): first priced, close 48.60; a member from 2025-11-28
    WDC    open gap less SPY's -23.77%, day -28.20%; total return -28.65%; raw 68.70 -> 52.61 -> 49.02
    PLTR   open gap less SPY's -5.74%, day -10.07%; total return -10.53%; raw 101.35 -> 95.88 -> 90.68
    MRNA   open gap less SPY's -4.99%, day -4.13%; total return -4.59%; raw 35.53 -> 33.88 -> 33.90
    DPZ    open gap less SPY's -4.97%, day -1.00%; total return -1.46%; raw 462.37 -> 441.00 -> 455.64
    ANET   open gap less SPY's -4.67%, day -5.25%; total return -5.71%; raw 98.30 -> 94.05 -> 92.69

2025-08-07  SKYD (Skydance Corporation, PSKY): first priced, close 11.74; a member from 2025-08-07
    FTNT   open gap less SPY's -23.62%, day -21.95%; total return -22.03%; raw 96.58 -> 74.30 -> 75.30
    LLY    open gap less SPY's -10.02%, day -14.05%; total return -14.14%; raw 746.37 -> 675.66 -> 640.86
    ABNB   open gap less SPY's -8.30%, day -7.94%; total return -8.02%; raw 130.50 -> 120.38 -> 120.03
    CF     open gap less SPY's -6.89%, day -7.71%; total return -7.80%; raw 89.90 -> 84.20 -> 82.89
    VST    open gap less SPY's -5.84%, day +2.44%; total return +2.36%; raw 200.85 -> 190.21 -> 205.59

2025-10-30  SOLS (Solstice Advanced Materials Inc. Common Stock, SOLS): first priced, close 48.74; a member from 2025-10-30
    CMG    open gap less SPY's -20.85%, day -17.08%; total return -18.18%; raw 39.76 -> 31.27 -> 32.53
    BAX    open gap less SPY's -15.01%, day -13.44%; total return -14.54%; raw 22.42 -> 18.94 -> 19.16
    EME    open gap less SPY's -13.91%, day -15.50%; total return -16.60%; raw 777.00 -> 665.00 -> 648.00
    EBAY   open gap less SPY's -13.50%, day -14.78%; total return -15.88%; raw 99.54 -> 85.60 -> 83.73
    CI     open gap less SPY's -10.90%, day -16.29%; total return -17.39%; raw 299.12 -> 265.00 -> 247.10

2025-11-03  Q (Qnity Electronics, Inc., Q): first priced, close 97.00; a member from 2025-11-03
    DD     open gap less SPY's -59.27%, day -57.70%; total return +1.54%; raw 81.65 -> 33.69 -> 34.69; the cache: split 2025-11-03
    KMB    open gap less SPY's -13.08%, day -14.76%; total return -14.57%; raw 119.71 -> 104.69 -> 102.27
    ARE    open gap less SPY's -2.90%, day -4.53%; total return -4.35%; raw 58.22 -> 56.84 -> 55.69
    FRT    open gap less SPY's -2.63%, day -0.72%; total return -0.53%; raw 96.19 -> 94.17 -> 95.68
    CMS    open gap less SPY's -2.58%, day -1.63%; total return -1.44%; raw 73.55 -> 72.04 -> 72.49

2026-06-01  FDXF (FedEx Freight Holding Company, Inc., FDXF): first priced, close 149.53; a member from 2026-06-01
    FDX    open gap less SPY's -20.92%, day -18.06%; total return -17.79%; raw 411.75 -> 325.01 -> 338.49
    QCOM   open gap less SPY's -6.90%, day -9.05%; total return -8.78%; raw 251.02 -> 233.33 -> 228.99
    COIN   open gap less SPY's -5.05%, day -3.67%; total return -3.40%; raw 189.03 -> 179.21 -> 182.61
    HOOD   open gap less SPY's -4.48%, day -4.06%; total return -3.79%; raw 94.30 -> 89.94 -> 90.73
    INTC   open gap less SPY's -4.43%, day -4.94%; total return -4.67%; raw 114.68 -> 109.43 -> 109.33

2026-06-29  HONA (Honeywell Aerospace Inc. Common Stock, HONA): first priced, close 220.19; a member from 2026-06-29
    MLM    open gap less SPY's -6.55%, day -7.30%; total return -5.65%; raw 616.06 -> 582.07 -> 581.23
    VMC    open gap less SPY's -3.78%, day -4.07%; total return -2.42%; raw 311.35 -> 302.81 -> 303.82
    CRH    open gap less SPY's -3.42%, day -4.72%; total return -3.07%; raw 112.32 -> 109.64 -> 108.87
    VZ     open gap less SPY's -3.12%, day -6.89%; total return -5.24%; raw 46.54 -> 45.57 -> 44.10
    TXN    open gap less SPY's -2.82%, day -1.63%; total return +0.02%; raw 285.43 -> 280.33 -> 285.48

2026-10-01  VYLR (Vylor Inc., VYLR): first priced, close 68.26; a member from 2026-10-01
    CTVA   open gap less SPY's -81.63%, day -83.99%; total return -83.81%; raw 77.65 -> 14.44 -> 12.57
    MRNA   open gap less SPY's -3.32%, day -2.06%; total return -1.89%; raw 192.57 -> 186.61 -> 188.94
    SKYD   open gap less SPY's -2.36%, day -9.76%; total return -9.58%; raw 10.33 -> 10.11 -> 9.34
    FCX    open gap less SPY's -2.24%, day -1.21%; total return -1.03%; raw 70.00 -> 68.59 -> 69.28
    APD    open gap less SPY's -2.19%, day -1.94%; total return -1.11%; raw 278.23 -> 272.78 -> 273.33; the cache: distribution 2026-10-01

Flagged: a member's opening gap below SPY's by more than 1.5 points on a security's first day, with no split or distribution on record -- (day, new security, member, gap, day's move, member's prior close, new security's close, and their ratio of value)
  2022-02-02 CEG    EXC    gap -29.56% day -26.86%; the member's drop at the open 17.10 a share against the new security's close 53.01: 0.3225
  2022-02-02 CEG    PYPL   gap -20.99% day -25.56%; the member's drop at the open 36.90 a share against the new security's close 53.01: 0.6961
  2022-02-02 CEG    CHRW   gap -11.39% day -12.48%; the member's drop at the open 11.91 a share against the new security's close 53.01: 0.2247
  2022-02-02 CEG    WM     gap -5.09% day -2.07%; the member's drop at the open 7.62 a share against the new security's close 53.01: 0.1438
  2022-02-02 CEG    GILD   gap -4.90% day -4.84%; the member's drop at the open 3.36 a share against the new security's close 53.01: 0.0633
  2022-04-11 WBD    T      gap -20.98% day -16.97%; the member's drop at the open 5.06 a share against the new security's close 24.78: 0.2043
  2022-04-11 WBD    TSLA   gap -3.62% day -3.12%; the member's drop at the open 37.16 a share against the new security's close 24.78: 1.4997
  2022-04-11 WBD    NVDA   gap -3.14% day -3.49%; the member's drop at the open 7.26 a share against the new security's close 24.78: 0.2931
  2022-04-11 WBD    FBIN   gap -2.30% day -0.40%; the member's drop at the open 1.68 a share against the new security's close 24.78: 0.0677
  2022-04-11 WBD    MRNA   gap -2.24% day +1.19%; the member's drop at the open 3.60 a share against the new security's close 24.78: 0.1451
  2022-12-15 MBC    WDC    gap -4.50% day -7.66%; the member's drop at the open 1.61 a share against the new security's close 7.70: 0.2094
  2022-12-15 MBC    NFLX   gap -3.67% day -6.18%; the member's drop at the open 11.65 a share against the new security's close 7.70: 1.5132
  2022-12-15 MBC    SYY    gap -2.25% day -2.24%; the member's drop at the open 1.85 a share against the new security's close 7.70: 0.2404
  2022-12-15 MBC    EL     gap -2.16% day -1.01%; the member's drop at the open 5.39 a share against the new security's close 7.70: 0.6997
  2023-01-04 GEHC   MSFT   gap -3.67% day -5.15%; the member's drop at the open 8.79 a share against the new security's close 60.49: 0.1453
  2023-01-04 GEHC   CHRW   gap -3.60% day -0.20%; the member's drop at the open 3.25 a share against the new security's close 60.49: 0.0537
  2023-01-04 GEHC   APA    gap -3.39% day -4.14%; the member's drop at the open 1.48 a share against the new security's close 60.49: 0.0245
  2023-01-04 GEHC   HON    gap -3.33% day -2.76%; the member's drop at the open 7.13 a share against the new security's close 60.49: 0.1178
  2023-05-04 KVUE   PARA   gap -17.12% day -27.64%; the member's drop at the open 3.92 a share against the new security's close 26.90: 0.1457
  2023-05-04 KVUE   KEY    gap -7.60% day -5.59%; the member's drop at the open 0.72 a share against the new security's close 26.90: 0.0269
  2023-05-04 KVUE   QCOM   gap -7.53% day -4.83%; the member's drop at the open 8.50 a share against the new security's close 26.90: 0.3159
  2023-05-04 KVUE   ANSS   gap -6.10% day -0.92%; the member's drop at the open 18.80 a share against the new security's close 26.90: 0.6988
  2023-05-04 KVUE   ZTS    gap -5.21% day +1.12%; the member's drop at the open 9.26 a share against the new security's close 26.90: 0.3441
  2023-09-12 TKO    ORCL   gap -11.21% day -12.95%; the member's drop at the open 14.21 a share against the new security's close 103.05: 0.1379
  2023-09-12 TKO    RTX    gap -2.53% day -1.19%; the member's drop at the open 1.94 a share against the new security's close 103.05: 0.0189
  2023-09-12 TKO    UPS    gap -1.85% day -2.14%; the member's drop at the open 2.98 a share against the new security's close 103.05: 0.0289
  2023-09-25 CRH    CCL    gap -3.27% day -1.49%; the member's drop at the open 0.46 a share against the new security's close 56.35: 0.0081
  2023-09-25 CRH    HPQ    gap -1.91% day -2.18%; the member's drop at the open 0.51 a share against the new security's close 56.35: 0.0091
  2023-09-25 CRH    LVS    gap -1.85% day -1.40%; the member's drop at the open 0.85 a share against the new security's close 56.35: 0.0150
  2023-10-02 VLTO   DHR    gap -11.52% day -13.81%; the member's drop at the open 28.58 a share against the new security's close 85.12: 0.3358
  2023-10-02 VLTO   K      gap -6.20% day -11.74%; the member's drop at the open 3.69 a share against the new security's close 85.12: 0.0434
  2023-10-02 VLTO   NSC    gap -3.74% day -2.72%; the member's drop at the open 7.36 a share against the new security's close 85.12: 0.0865
  2023-10-02 VLTO   SEDG   gap -3.20% day -5.33%; the member's drop at the open 4.14 a share against the new security's close 85.12: 0.0486
  2023-10-02 VLTO   TSLA   gap -1.96% day +0.59%; the member's drop at the open 4.91 a share against the new security's close 85.12: 0.0576
  2024-03-21 RDDT   DRI    gap -6.20% day -6.83%; the member's drop at the open 10.83 a share against the new security's close 50.44: 0.2146
  2024-03-21 RDDT   ACN    gap -4.88% day -9.64%; the member's drop at the open 18.58 a share against the new security's close 50.44: 0.3683
  2024-03-21 RDDT   FDS    gap -3.35% day -7.96%; the member's drop at the open 16.37 a share against the new security's close 50.44: 0.3245
  2024-03-21 RDDT   CTSH   gap -2.78% day -3.75%; the member's drop at the open 2.09 a share against the new security's close 50.44: 0.0414
  2024-03-21 RDDT   EPAM   gap -2.47% day -3.35%; the member's drop at the open 7.24 a share against the new security's close 50.44: 0.1435
  2024-04-01 SOLV   LW     gap -2.92% day -1.28%; the member's drop at the open 3.11 a share against the new security's close 69.10: 0.0451
  2024-04-01 SOLV   T      gap -2.47% day -0.39%; the member's drop at the open 0.44 a share against the new security's close 69.10: 0.0063
  2024-04-01 SOLV   FDX    gap -2.04% day -3.14%; the member's drop at the open 5.92 a share against the new security's close 69.10: 0.0857
  2024-04-01 SOLV   CHRW   gap -2.00% day -3.50%; the member's drop at the open 1.52 a share against the new security's close 69.10: 0.0220
  2024-04-02 GEV    HUM    gap -9.48% day -12.77%; the member's drop at the open 33.32 a share against the new security's close 140.00: 0.2380
  2024-04-02 GEV    UNH    gap -5.40% day -5.81%; the member's drop at the open 26.42 a share against the new security's close 140.00: 0.1887
  2024-04-02 GEV    TSLA   gap -5.22% day -4.27%; the member's drop at the open 9.15 a share against the new security's close 140.00: 0.0654
  2024-04-02 GEV    CVS    gap -5.13% day -6.58%; the member's drop at the open 4.08 a share against the new security's close 140.00: 0.0292
  2024-07-08 SW     NOW    gap -5.47% day -5.11%; the member's drop at the open 44.13 a share against the new security's close 46.22: 0.9549
  2024-07-08 SW     TSLA   gap -1.66% day +0.45%; the member's drop at the open 4.17 a share against the new security's close 46.22: 0.0903
  2024-07-08 SW     CSX    gap -1.53% day +0.15%; the member's drop at the open 0.51 a share against the new security's close 46.22: 0.0110
  2024-08-01 FERG   MRNA   gap -16.79% day -19.60%; the member's drop at the open 20.02 a share against the new security's close 216.33: 0.0925
  2024-08-01 FERG   WDC    gap -10.33% day -8.31%; the member's drop at the open 6.92 a share against the new security's close 216.33: 0.0320
  2024-08-01 FERG   IR     gap -8.69% day -7.72%; the member's drop at the open 8.72 a share against the new security's close 216.33: 0.0403
  2024-08-01 FERG   LRCX   gap -7.54% day -8.45%; the member's drop at the open 69.43 a share against the new security's close 216.33: 0.3210
  2024-08-01 FERG   MGM    gap -7.11% day -11.80%; the member's drop at the open 3.06 a share against the new security's close 216.33: 0.0141
  2024-09-30 AMTM   J      gap -15.47% day -16.24%; the member's drop at the open 24.06 a share against the new security's close 32.25: 0.7462
  2024-09-30 AMTM   APTV   gap -3.36% day -3.40%; the member's drop at the open 2.49 a share against the new security's close 32.25: 0.0773
  2024-09-30 AMTM   GM     gap -3.17% day -3.93%; the member's drop at the open 1.47 a share against the new security's close 32.25: 0.0457
  2024-09-30 AMTM   NXPI   gap -2.89% day -2.41%; the member's drop at the open 7.09 a share against the new security's close 32.25: 0.2198
  2024-09-30 AMTM   BAX    gap -2.80% day -2.82%; the member's drop at the open 1.09 a share against the new security's close 32.25: 0.0338
  2025-02-24 SNDK   WDC    gap -23.77% day -28.20%; the member's drop at the open 16.33 a share against the new security's close 48.60: 0.3361
  2025-02-24 SNDK   PLTR   gap -5.74% day -10.07%; the member's drop at the open 5.82 a share against the new security's close 48.60: 0.1198
  2025-02-24 SNDK   MRNA   gap -4.99% day -4.13%; the member's drop at the open 1.77 a share against the new security's close 48.60: 0.0365
  2025-02-24 SNDK   DPZ    gap -4.97% day -1.00%; the member's drop at the open 22.97 a share against the new security's close 48.60: 0.4727
  2025-02-24 SNDK   ANET   gap -4.67% day -5.25%; the member's drop at the open 4.59 a share against the new security's close 48.60: 0.0945
  2025-08-07 SKYD   FTNT   gap -23.62% day -21.95%; the member's drop at the open 22.81 a share against the new security's close 11.74: 1.9428
  2025-08-07 SKYD   LLY    gap -10.02% day -14.05%; the member's drop at the open 74.79 a share against the new security's close 11.74: 6.3706
  2025-08-07 SKYD   ABNB   gap -8.30% day -7.94%; the member's drop at the open 10.83 a share against the new security's close 11.74: 0.9228
  2025-08-07 SKYD   CF     gap -6.89% day -7.71%; the member's drop at the open 6.19 a share against the new security's close 11.74: 0.5274
  2025-08-07 SKYD   VST    gap -5.84% day +2.44%; the member's drop at the open 11.74 a share against the new security's close 11.74: 0.9998
  2025-10-30 SOLS   CMG    gap -20.85% day -17.08%; the member's drop at the open 8.29 a share against the new security's close 48.74: 0.1700
  2025-10-30 SOLS   BAX    gap -15.01% day -13.44%; the member's drop at the open 3.37 a share against the new security's close 48.74: 0.0691
  2025-10-30 SOLS   EME    gap -13.91% day -15.50%; the member's drop at the open 108.06 a share against the new security's close 48.74: 2.2170
  2025-10-30 SOLS   EBAY   gap -13.50% day -14.78%; the member's drop at the open 13.43 a share against the new security's close 48.74: 0.2756
  2025-10-30 SOLS   CI     gap -10.90% day -16.29%; the member's drop at the open 32.60 a share against the new security's close 48.74: 0.6689
  2025-11-03 Q      KMB    gap -13.08% day -14.76%; the member's drop at the open 15.65 a share against the new security's close 97.00: 0.1614
  2025-11-03 Q      ARE    gap -2.90% day -4.53%; the member's drop at the open 1.69 a share against the new security's close 97.00: 0.0174
  2025-11-03 Q      FRT    gap -2.63% day -0.72%; the member's drop at the open 2.53 a share against the new security's close 97.00: 0.0261
  2025-11-03 Q      CMS    gap -2.58% day -1.63%; the member's drop at the open 1.90 a share against the new security's close 97.00: 0.0196
  2026-06-01 FDXF   FDX    gap -20.92% day -18.06%; the member's drop at the open 86.13 a share against the new security's close 149.53: 0.5760
  2026-06-01 FDXF   QCOM   gap -6.90% day -9.05%; the member's drop at the open 17.32 a share against the new security's close 149.53: 0.1158
  2026-06-01 FDXF   COIN   gap -5.05% day -3.67%; the member's drop at the open 9.54 a share against the new security's close 149.53: 0.0638
  2026-06-01 FDXF   HOOD   gap -4.48% day -4.06%; the member's drop at the open 4.22 a share against the new security's close 149.53: 0.0282
  2026-06-01 FDXF   INTC   gap -4.43% day -4.94%; the member's drop at the open 5.08 a share against the new security's close 149.53: 0.0340
  2026-06-29 HONA   MLM    gap -6.55% day -7.30%; the member's drop at the open 40.36 a share against the new security's close 220.19: 0.1833
  2026-06-29 HONA   VMC    gap -3.78% day -4.07%; the member's drop at the open 11.76 a share against the new security's close 220.19: 0.0534
  2026-06-29 HONA   CRH    gap -3.42% day -4.72%; the member's drop at the open 3.84 a share against the new security's close 220.19: 0.0174
  2026-06-29 HONA   VZ     gap -3.12% day -6.89%; the member's drop at the open 1.45 a share against the new security's close 220.19: 0.0066
  2026-06-29 HONA   TXN    gap -2.82% day -1.63%; the member's drop at the open 8.05 a share against the new security's close 220.19: 0.0365
  2026-10-01 VYLR   CTVA   gap -81.63% day -83.99%; the member's drop at the open 63.39 a share against the new security's close 68.26: 0.9286
  2026-10-01 VYLR   MRNA   gap -3.32% day -2.06%; the member's drop at the open 6.40 a share against the new security's close 68.26: 0.0937
  2026-10-01 VYLR   SKYD   gap -2.36% day -9.76%; the member's drop at the open 0.24 a share against the new security's close 68.26: 0.0036
  2026-10-01 VYLR   FCX    gap -2.24% day -1.21%; the member's drop at the open 1.57 a share against the new security's close 68.26: 0.0230
```

### fetch_children.txt -- the children the cache did not hold, fetched into it

```text
ZIMV 2022-03-02: 913 raw bars, first 2022-03-01; on the day close 24.31; record 'ZimVie Inc. Common Stock' BBG014630587 list 2022-02-14; new cache files ['bars-ZIMV-2022-01-03-2026-11-04-raw.json', 'ref-ZIMV-2022-03-02.json']
EMBC 2022-04-01: 1153 raw bars, first 2022-04-01; on the day close 30.50; record 'Embecta Corp. Common Stock' BBG014L7D4G1 list None; new cache files ['bars-EMBC-2022-01-03-2026-11-04-raw.json', 'ref-EMBC-2022-04-01.json']
FTRE 2023-07-03: 840 raw bars, first 2023-07-03; on the day close 36.84; record 'Fortrea Holdings Inc. Common Stock' BBG01GZF9VY6 list None; new cache files ['bars-FTRE-2022-01-03-2026-11-04-raw.json']
PHIN 2023-07-05: 839 raw bars, first 2023-07-05; on the day close 36.75; record 'PHINIA Inc.' BBG01F6N6NM7 list None; new cache files ['bars-PHIN-2022-01-03-2026-11-04-raw.json', 'ref-PHIN-2023-07-05.json']
KLG 2023-10-02: 498 raw bars, first 2023-10-02; on the day close 13.35; record 'WK Kellogg Co' BBG018CYVVW5 list None; new cache files ['bars-KLG-2022-01-03-2026-11-04-raw.json', 'ref-KLG-2023-10-02.json']
RAL 2025-06-30: 341 raw bars, first 2025-06-30; on the day close 48.49; record 'Ralliant Corporation' BBG01V4YRP24 list None; new cache files ['bars-RAL-2022-01-03-2026-11-04-raw.json']
MRP 2025-02-07: 438 raw bars, first 2025-02-07; on the day close 26.74; record 'Millrose Properties, Inc.' BBG01RQYH2X7 list None; new cache files ['bars-MRP-2022-01-03-2026-11-04-raw.json', 'ref-MRP-2025-02-07.json']
VSNT 2026-01-05: 211 raw bars, first 2026-01-05; on the day close 40.57; record 'Versant Media Group, Inc. Class A Common Stock' BBG01Y0M8K11 list None; new cache files ['bars-VSNT-2022-01-03-2026-11-04-raw.json', 'ref-VSNT-2026-01-05.json']
VGNT 2026-04-01: 151 raw bars, first 2026-04-01; on the day close 27.85; record 'Versigent PLC' None list None; new cache files ['bars-VGNT-2022-01-03-2026-11-04-raw.json', 'ref-VGNT-2026-04-01.json']
MBGL 2026-07-01: 89 raw bars, first 2026-07-01; on the day close 21.19; record 'Mobility Global Inc.' BBG022G54330 list None; new cache files ['bars-MBGL-2022-01-03-2026-11-04-raw.json', 'ref-MBGL-2026-07-01.json']
```

### make_spin_offs.txt -- the rulings written to `config/sp500_spin_off_overrides.json`

```text
wrote C:\Data\Development\Python\McTheoryApps\praxis\config\sp500_spin_off_overrides.json: 19 rulings
  2022-02-02 EXC   BBG000J6XT05   -> CEG   BBG014KFRNP7 k 0.333333  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2022-03-01 ZBH   BBG000BKPL53   -> ZIMV  BBG014630587 k 0.1  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2022-04-01 BDX   BBG000BCZYD3   -> EMBC  BBG014L7D4G1 k 0.2  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2022-04-11 T     BBG000BSJK37   -> WBD   BBG011386VF4 k 0.241917  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2022-12-15 FBIN  BBG001B4BV87   -> MBC   BBG01B9G42Y3 k 1  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2023-07-05 BWA   BBG000BJ49H3   -> PHIN  BBG01F6N6NM7 k 0.2  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2023-10-02 DHR   BBG000BH3JF8   -> VLTO  BBG019Q32XJ9 k 0.333333  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2023-10-02 K     BBG000BMKDM3   -> KLG   BBG018CYVVW5 k 0.25  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2024-09-30 J     BBG019C1BQR4   -> AMTM  BBG01NY7B734 k 1  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2025-01-21 LEN   BBG000BN5HF7   -> MRP   BBG01RQYH2X7 k 0.5  [midoffice14 brief (Jeff, 2026-11-12), item 2: LEN's adjustme]
  2025-02-24 WDC   BBG000BWNFZ9   -> SNDK  BBG01R388JG1 k 0.333333  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2025-05-16 J     BBG019C1BQR4   -> AMTM  BBG01NY7B734 k 0.060835  [midoffice14, item 2: from the public record, a pair the brie]
  2026-01-05 CMCSA BBG000BFT2L4   -> VSNT  BBG01Y0M8K11 k 0.04  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2026-02-10 BDX   BBG000BCZYD3   -> WAT   BBG000FQRVM3 k 0.135  [midoffice14, item 2: from the public record, a pair the brie]
  2026-04-01 APTV  CIK0001521332  -> VGNT  None k 0.333333  [midoffice14, item 2: from the public record, a pair the brie]
  2026-06-01 FDX   BBG000BJF1Z8   -> FDXF  BBG01ZPB91C4 k 0.5 seeded  [midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it name]
  2026-06-29 HON   BBG000H556T9   -> HONA  BBG020QYV152 k 0.5 split 0.5  [midoffice14, item 2: from the public record, a pair the brie]
  2026-07-01 SPGI  BBG000BP1Q11   -> MBGL  BBG022G54330 k 1  [midoffice14, item 2: from the public record, a pair the brie]
  2026-10-01 CTVA  BBG00BN969C1   -> VYLR  BBG024CQR6X9 k 1 seeded  [midoffice14 brief (Jeff, 2026-11-12), item 1]
```

### ruling5_check.txt -- ruling 5's standing line on midoffice13's stored E run, against its `minvar_realised.txt`

```text
The standing line on midoffice13's E run: the minimum-variance portfolios' realised volatility, % a year, by a -- the run script's own functions
                                          a = 0    a = 1    a = 2    a = 3    a = 4
  long-only, whole test                   10.93    11.00    11.09    11.18    11.29
  long-only, the first half               11.47    11.54    11.62    11.73    11.85
  long-only, the second half              10.38    10.45    10.54    10.63    10.72
  unconstrained, whole test               11.07    11.19    11.32    11.44    11.57
  unconstrained, the first half           11.56    11.68    11.80    11.92    12.04
  unconstrained, the second half          10.57    10.70    10.83    10.96    11.10

midoffice13's minvar_realised.txt, for the same rows:
  a = 0    a = 1    a = 2    a = 3    a = 4
  long-only, whole test               10.93    11.00    11.09    11.18    11.29
  long-only, first half               11.47    11.54    11.62    11.73    11.85
  unconstrained, whole test           11.07    11.19    11.32    11.44    11.57
  unconstrained, first half           11.56    11.68    11.80    11.92    12.04

the halves: the first half (2024-02-08 to 2025-06-23, 343); the second half (2025-06-24 to 2026-11-02, 343)
```

### equal_R4.txt -- ruling 4: the 21-day view at the new default (R4) against midoffice13's V21N, bit for bit

```text
claude/scratch/midoffice14/bias/R4.npz against claude/scratch/midoffice13/bias/V21N.npz: 686 forecast dates in common (686 and 686), 2024-02-08 to 2026-11-02
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

### report_R4_vs_V21N.txt -- R4's report against V21N's, every differing line

```text
170c170
<   686 refits in 8.5 min; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every portfolio formed that day
---
>   686 refits in 7.9 min; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every portfolio formed that day
172,190c172,190
<   Portfolios at one day, each formed at its forecast date and held to the next
<                                          T    bias           95% band    MRAD    min    max
<   the index, cap-weighted              686   1.123 under  0.946-1.054   0.249  0.833  1.466
<   the index, equally weighted          686   1.063 under  0.946-1.054   0.197  0.886  1.343
<   smallest 85, equally weighted        686   1.043    in  0.946-1.054   0.136  0.900  1.225
<   largest 50, equally weighted         686   1.086 under  0.946-1.054   0.192  0.822  1.370
<   minimum variance, long-only          686   1.347 under  0.946-1.054   0.368  1.056  1.601
<   minimum variance, unconstrained      686   1.503 under  0.946-1.054   0.528  1.141  1.817
< 
<   Portfolios at the model's horizon, 21 trading days, each bought at its forecast date and held 21 days: every forecast, the windows overlapping, its band simulated; and every 21st, the windows apart
<                                          T    bias           95% band    MRAD    min    max   apart: T   bias           95% band
<   the index, cap-weighted              666   0.975    in  0.790-1.189   0.162  0.795  1.255     32  1.001    in  0.750-1.250
<   the index, equally weighted          666   0.905    in  0.790-1.189   0.135  0.683  1.121     32  0.894    in  0.750-1.250
<   smallest 85, equally weighted        666   0.859    in  0.790-1.189   0.127  0.727  0.984     32  0.810    in  0.750-1.250
<   largest 50, equally weighted         666   0.974    in  0.790-1.189   0.173  0.714  1.241     32  1.051    in  0.750-1.250
<   minimum variance, long-only          666   1.118    in  0.790-1.189   0.090  0.935  1.280     32  1.067    in  0.750-1.250
<   minimum variance, unconstrained      666   1.224 under  0.790-1.189   0.199  1.037  1.372     32  1.146    in  0.750-1.250
<     minimum variance, long-only: 90 names held at the median (55 to 145); forecast 7.75% a year at the median, realised 10.76% from its daily returns; its KKT conditions met to 2.8e-13
<     minimum variance, unconstrained: 501 names held at the median (499 to 503); forecast 7.19% a year at the median, realised 11.18% from its daily returns; its KKT conditions met to 2.8e-13
---
>   Portfolios at one day, each formed at its forecast date and held to the next; beside each, its volatility % a year -- the forecast at its median, the realised from its returns
>                                          T    bias           95% band    MRAD    min    max   forecast realised
>   the index, cap-weighted              686   1.123 under  0.946-1.054   0.249  0.833  1.466     13.38%   15.27%
>   the index, equally weighted          686   1.063 under  0.946-1.054   0.197  0.886  1.343     12.37%   13.95%
>   smallest 85, equally weighted        686   1.043    in  0.946-1.054   0.136  0.900  1.225     16.10%   17.38%
>   largest 50, equally weighted         686   1.086 under  0.946-1.054   0.192  0.822  1.370     13.13%   14.64%
>   minimum variance, long-only          686   1.347 under  0.946-1.054   0.368  1.056  1.601      7.75%   10.76%
>   minimum variance, unconstrained      686   1.503 under  0.946-1.054   0.528  1.141  1.817      7.19%   11.18%
> 
>   Portfolios at the model's horizon, 21 trading days, each bought at its forecast date and held 21 days: every forecast, the windows overlapping, its band simulated; and every 21st, the windows apart; the volatility % a year from the 21-day returns, the windows overlapping
>                                          T    bias           95% band    MRAD    min    max   apart: T   bias           95% band   forecast realised
>   the index, cap-weighted              666   0.975    in  0.790-1.189   0.162  0.795  1.255     32  1.001    in  0.750-1.250     13.38%   12.79%
>   the index, equally weighted          666   0.905    in  0.790-1.189   0.135  0.683  1.121     32  0.894    in  0.750-1.250     12.37%   11.42%
>   smallest 85, equally weighted        666   0.859    in  0.790-1.189   0.127  0.727  0.984     32  0.810    in  0.750-1.250     16.10%   13.94%
>   largest 50, equally weighted         666   0.974    in  0.790-1.189   0.173  0.714  1.241     32  1.051    in  0.750-1.250     13.13%   12.67%
>   minimum variance, long-only          666   1.118    in  0.790-1.189   0.090  0.935  1.280     32  1.067    in  0.750-1.250      7.75%    8.63%
>   minimum variance, unconstrained      666   1.224 under  0.790-1.189   0.199  1.037  1.372     32  1.146    in  0.750-1.250      7.19%    8.77%
>     minimum variance, long-only: 90 names held at the median (55 to 145); its KKT conditions met to 2.8e-13; realised 10.76% a year, the first half 11.57% (bias 1.461), the second 9.89% (1.225)
>     minimum variance, unconstrained: 501 names held at the median (499 to 503); its KKT conditions met to 2.8e-13; realised 11.18% a year, the first half 12.25% (bias 1.661), the second 10.01% (1.328)
270c270
<   every forecast and outcome written to C:\Data\Development\Python\McTheoryApps\praxis\claude\scratch\midoffice13\bias\V21N.tmp.npz
---
>   every forecast and outcome written to C:\Data\Development\Python\McTheoryApps\praxis\claude\scratch\midoffice14\bias\R4.tmp.npz
```

### panel_listed_rebuilt.txt -- `--spin-offs none --suffix _listed`, the final securities code, against midoffice10's `_listed`

```text
data/external/polygon/sp500_securities_2022-01-03_2026-11-03_listed.npz
  against claude/scratch/midoffice14/panels/sp500_securities_2022-01-03_2026-11-03_listed.npz
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
  total_return_prices      equal to the bit (1214, 592)
  unadjusted_prices        equal to the bit (1214, 592)
  volumes                  equal to the bit (1214, 592)

total_return_prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

shares_outstanding: 0 security-days changed, 0 member-days
```

### panel_item1.txt -- item 1, CTVA's ruling alone, against `_listed`

```text
data/external/polygon/sp500_securities_2022-01-03_2026-11-03_listed.npz
  against claude/scratch/midoffice14/panels/sp500_securities_2022-01-03_2026-11-03_ctva.npz
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
  shares_outstanding       DIFFERS: 24 cells, 24 member-days
  sic_codes                equal to the bit (592,)
  sic_dates                equal to the bit (592,)
  snapshot                 equal to the bit ()
  tickers                  equal to the bit (1214, 592)
  total_return_prices      DIFFERS: 24 cells, 3 member-days
  unadjusted_prices        equal to the bit (1214, 592)
  volumes                  equal to the bit (1214, 592)

total_return_prices, by daily return: 1 security-days changed beyond 1e-12, 1 of them member-days; 11 more differ in the last bits
  2026-10-01 CTVA   member: -83.8120% -> +4.0953% (+87.91 points)

prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

shares_outstanding: 24 security-days changed, 24 member-days
  VYLR   2026-10-01..2026-11-03 (24 days, 24 member-days): nan -> 668,820,000, ... nan -> 668,820,000
```

### panel_item2.txt -- item 2, every ruling, against item 1's panel

```text
claude/scratch/midoffice14/panels/sp500_securities_2022-01-03_2026-11-03_ctva.npz
  against claude/scratch/midoffice14/panels/sp500_securities_2022-01-03_2026-11-03_spinoffs.npz
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
  shares_outstanding       DIFFERS: 199 cells, 199 member-days
  sic_codes                equal to the bit (592,)
  sic_dates                equal to the bit (592,)
  snapshot                 equal to the bit ()
  tickers                  equal to the bit (1214, 592)
  total_return_prices      DIFFERS: 9,855 cells, 8,476 member-days
  unadjusted_prices        equal to the bit (1214, 592)
  volumes                  equal to the bit (1214, 592)

total_return_prices, by daily return: 18 security-days changed beyond 1e-12, 18 of them member-days; 4445 more differ in the last bits
  2022-02-02 EXC    member: -25.8862% -> +4.6689% (+30.56 points)
  2022-03-01 ZBH    member: -2.3980% -> -0.3908% (+2.01 points)
  2022-04-01 BDX    member: -1.3158% -> +0.9774% (+2.29 points)
  2022-04-11 T      member: -18.6827% -> +6.1504% (+24.83 points)
  2022-12-15 FBIN   member: -6.9144% -> +5.5250% (+12.44 points)
  2023-07-05 BWA    member: -13.5016% -> +1.2220% (+14.72 points)
  2023-10-02 DHR    member: -13.8493% -> -2.4130% (+11.44 points)
  2023-10-02 K      member: -11.7795% -> -6.1712% (+5.61 points)
  2024-09-30 J      member: -15.8362% -> +4.8994% (+20.74 points)
  2025-01-21 LEN    member: -2.8306% -> +3.5114% (+6.34 points)
  2025-02-24 WDC    member: -28.6515% -> -5.0724% (+23.58 points)
  2025-05-16 J      member: +0.2873% -> +1.3188% (+1.03 points)
  2026-01-05 CMCSA  member: -4.7732% -> +0.7204% (+5.49 points)
  2026-02-10 BDX    member: -17.2188% -> +4.0685% (+21.29 points)
  2026-04-01 APTV   member: -10.5847% -> +2.7842% (+13.37 points)
  2026-06-01 FDX    member: -17.7923% -> +0.3655% (+18.16 points)
  2026-06-29 HON    member: -1.8991% -> -3.5377% (-1.64 points)
  2026-07-01 SPGI   member: +1.8931% -> +7.0962% (+5.20 points)

prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

shares_outstanding: 199 security-days changed, 199 member-days
  FDXF   2026-06-01..2026-11-03 (109 days, 109 member-days): nan -> 119,303,378, ... nan -> 119,303,378
  HON    2026-06-29..2026-11-03 (90 days, 90 member-days): 635,675,701 -> 317,837,850, ... 635,675,701 -> 317,358,518
```

### panel_changed.txt -- the rebuilt panel, `_spinoffs`, against `_listed`: every changed member-day

```text
data/external/polygon/sp500_securities_2022-01-03_2026-11-03_listed.npz
  against claude/scratch/midoffice14/panels/sp500_securities_2022-01-03_2026-11-03_spinoffs.npz
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
  shares_outstanding       DIFFERS: 223 cells, 223 member-days
  sic_codes                equal to the bit (592,)
  sic_dates                equal to the bit (592,)
  snapshot                 equal to the bit ()
  tickers                  equal to the bit (1214, 592)
  total_return_prices      DIFFERS: 9,879 cells, 8,479 member-days
  unadjusted_prices        equal to the bit (1214, 592)
  volumes                  equal to the bit (1214, 592)

total_return_prices, by daily return: 19 security-days changed beyond 1e-12, 19 of them member-days; 4456 more differ in the last bits
  2022-02-02 EXC    member: -25.8862% -> +4.6689% (+30.56 points)
  2022-03-01 ZBH    member: -2.3980% -> -0.3908% (+2.01 points)
  2022-04-01 BDX    member: -1.3158% -> +0.9774% (+2.29 points)
  2022-04-11 T      member: -18.6827% -> +6.1504% (+24.83 points)
  2022-12-15 FBIN   member: -6.9144% -> +5.5250% (+12.44 points)
  2023-07-05 BWA    member: -13.5016% -> +1.2220% (+14.72 points)
  2023-10-02 DHR    member: -13.8493% -> -2.4130% (+11.44 points)
  2023-10-02 K      member: -11.7795% -> -6.1712% (+5.61 points)
  2024-09-30 J      member: -15.8362% -> +4.8994% (+20.74 points)
  2025-01-21 LEN    member: -2.8306% -> +3.5114% (+6.34 points)
  2025-02-24 WDC    member: -28.6515% -> -5.0724% (+23.58 points)
  2025-05-16 J      member: +0.2873% -> +1.3188% (+1.03 points)
  2026-01-05 CMCSA  member: -4.7732% -> +0.7204% (+5.49 points)
  2026-02-10 BDX    member: -17.2188% -> +4.0685% (+21.29 points)
  2026-04-01 APTV   member: -10.5847% -> +2.7842% (+13.37 points)
  2026-06-01 FDX    member: -17.7923% -> +0.3655% (+18.16 points)
  2026-06-29 HON    member: -1.8991% -> -3.5377% (-1.64 points)
  2026-07-01 SPGI   member: +1.8931% -> +7.0962% (+5.20 points)
  2026-10-01 CTVA   member: -83.8120% -> +4.0953% (+87.91 points)

prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

shares_outstanding: 223 security-days changed, 223 member-days
  FDXF   2026-06-01..2026-11-03 (109 days, 109 member-days): nan -> 119,303,378, ... nan -> 119,303,378
  HON    2026-06-29..2026-11-03 (90 days, 90 member-days): 635,675,701 -> 317,837,850, ... 635,675,701 -> 317,358,518
  VYLR   2026-10-01..2026-11-03 (24 days, 24 member-days): nan -> 668,820,000, ... nan -> 668,820,000
```

### build_spin_offs_section.txt -- the build's report: its spin-off rulings, its guards, the first-day check

```text
Spin-offs: distributions of shares the cache records wrongly or not at all, config/sp500_spin_off_overrides.json: 19 rulings, 19 applied -- on each ex-date the parent's total return (w (C_t + D_t) + k V) / C_s - 1, the child at its close; the stitched prices as Polygon has them
  APTV   2026-04-01 -> VGNT  0.333333 a share: 69.44 on 2026-03-31 to 62.09; VGNT 27.85 on 2026-04-01 (Versigent PLC): total return +2.7842%, the adjusted closes' -10.5847%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review
  BDX    2022-04-01 -> EMBC  0.2 a share: 266.00 on 2022-03-31 to 262.50; EMBC 30.50 on 2022-04-01 (Embecta Corp. Common Stock, BBG014L7D4G1): total return +0.9774%, the adjusted closes' -1.3158%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  BDX    2026-02-10 -> WAT   0.135 a share: 207.39 on 2026-02-09 to 171.68; WAT 327.02 on 2026-02-10 (no record that day): total return +4.0685%, the adjusted closes' -17.2188%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review
  BWA    2023-07-05 -> PHIN  0.2 a share: 49.92 on 2023-07-03 to 43.18; PHIN 36.75 on 2023-07-05 (PHINIA Inc., BBG01F6N6NM7): total return +1.2220%, the adjusted closes' -13.5016%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  CMCSA  2026-01-05 -> VSNT  0.04 a share: 29.54 on 2026-01-02 to 28.13; VSNT 40.57 on 2026-01-05 (Versant Media Group, Inc. Class A Common Stock, BBG01Y0M8K11): total return +0.7204%, the adjusted closes' -4.7732%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  CTVA   2026-10-01 -> VYLR  1 a share: 77.65 on 2026-09-30 to 12.57; VYLR 68.26 on 2026-10-01 (Vylor Inc., BBG024W5VJC3): total return +4.0953%, the adjusted closes' -83.8120%; midoffice14 brief (Jeff, 2026-11-12), item 1
  DHR    2023-10-02 -> VLTO  0.333333 a share: 248.10 on 2023-09-29 to 213.74; VLTO 85.12 on 2023-10-02 (Veralto Corporation, BBG019Q32XJ9): total return -2.4130%, the adjusted closes' -13.8493%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  EXC    2022-02-02 -> CEG   0.333333 a share: 57.83 on 2022-02-01 to 42.86; CEG 53.01 on 2022-02-02 (Constellation Energy Corporation Common Stock, BBG014KFRNP7): total return +4.6689%, the adjusted closes' -25.8862%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  FBIN   2022-12-15 -> MBC   1 a share: 61.90 on 2022-12-14 to 57.62; MBC 7.70 on 2022-12-15 (MasterBrand, Inc., BBG01B9G42Y3): total return +5.5250%, the adjusted closes' -6.9144%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  FDX    2026-06-01 -> FDXF  0.5 a share: 411.75 on 2026-05-29 to 338.49; FDXF 149.53 on 2026-06-01 (FedEx Freight Holding Company, Inc., BBG01ZPB91C4): total return +0.3655%, the adjusted closes' -17.7923%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; FDXF's count as Vylor's
  HON    2026-06-29 -> HONA  0.5 a share, the parent's split 0.5 with it: 232.21 on 2026-06-26 to 227.80, w 0.5; HONA 220.19 on 2026-06-29 (Honeywell Aerospace Inc. Common Stock, BBG020QYV152): total return -3.5377%, the adjusted closes' -1.8991%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review; it supersedes midoffice08's decision 3 on HON's count
  J      2024-09-30 -> AMTM  1 a share: 155.53 on 2024-09-27 to 130.90; AMTM 32.25 on 2024-09-30 (Amentum Holdings, Inc., BBG01NY7B734): total return +4.8994%, the adjusted closes' -15.8362%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  J      2025-05-16 -> AMTM  0.060835 a share: 128.80 on 2025-05-15 to 129.17; AMTM 21.84 on 2025-05-16 (no record that day): total return +1.3188%, the adjusted closes' +0.2873%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review
  K      2023-10-02 -> KLG   0.25 a share: 59.51 on 2023-09-29 to 52.50; KLG 13.35 on 2023-10-02 (WK Kellogg Co, BBG018CYVVW5): total return -6.1712%, the adjusted closes' -11.7795%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  LEN    2025-01-21 -> MRP   0.5 a share: 141.54 on 2025-01-17 to 133.14; MRP 26.74 on 2025-02-07, its first close, none on the ex-date (Millrose Properties, Inc., BBG01RQYH2X7): total return +3.5114%, the adjusted closes' -2.8306%; midoffice14 brief (Jeff, 2026-11-12), item 2: LEN's adjustment of 2025-01-21, checked -- carried wrongly, the date right and the value a third of the distribution's; carried by item 1's rule at Millrose's first close
  SPGI   2026-07-01 -> MBGL  1 a share: 407.26 on 2026-06-30 to 414.97; MBGL 21.19 on 2026-07-01 (Mobility Global Inc., BBG022G54330): total return +7.0962%, the adjusted closes' +1.8931%; midoffice14, item 2: from the public record, a pair the brief's list does not name; carried by item 1's rule -- Code's reading, for Jeff's review
  T      2022-04-11 -> WBD   0.241917 a share: 24.14 on 2022-04-08 to 19.63; WBD 24.78 on 2022-04-11 (Warner Bros. Discovery, Inc. Series A Common Stock, BBG011386VF4): total return +6.1504%, the adjusted closes' -18.6827%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  WDC    2025-02-24 -> SNDK  0.333333 a share: 68.70 on 2025-02-21 to 49.02; SNDK 48.60 on 2025-02-24 (Sandisk Corporation Common Stock, BBG01R388JG1): total return -5.0724%, the adjusted closes' -28.6515%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  ZBH    2022-03-01 -> ZIMV  0.1 a share: 127.19 on 2022-02-28 to 124.14; ZIMV 25.53 on 2022-03-01 (ZimVie Inc. Common Stock, BBG014630587): total return -0.3908%, the adjusted closes' -2.3980%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  a split the cache does not record, applied to the count: HON 2-to-1 on 2026-06-29 (0.5, whole numbers): counts 635,675,701 on 2026-04-13, 316,940,010 on 2026-10-12, 0.499 times; raw price no raw price on a side; a split, the counts follow it; a spin-off ruling's split the cache does not record, its price not tested -- the distribution the same day offsets the gap
  FDXF   BBG01ZPB91C4: no count of its own -- FDX's 238,606,756 on the record date 2026-05-15 times 0.5, 119,303,378, from 2026-06-01 to the window's end, Polygon carrying none: 109 days, 109 member-days
  VYLR   BBG024CQR6X9: no count of its own -- CTVA's 668,820,000 on the record date 2026-09-24 times 1, 668,820,000, from 2026-10-01 to the window's end, Polygon carrying none: 24 days, 24 member-days
  a count override superseded by HON's ruling of 2026-06-29, not read: HON BBG000H556T9 2026-10-12..2026-11-03 (midoffice08, decision 3 (Jeff, 2026-11-10))


Guards
  ok  each (ticker, date) price feeds one column: 684,090 checked, 0 failure(s)
  ok  no column takes two tickers on one date: its spans do not overlap: 627 checked, 0 failure(s)
  ok  every member-day is a column's, on a day inside its spans or one the table maps outside them: 610,842 checked, 0 failure(s)
  ok  every bar under a span's ticker inside the span is its column's, and only those: 684,087 checked, 0 failure(s)
  ok  every member-day of the walk through the renames is a column's member-day: 610,842 checked, 0 failure(s)
  ok  the panel's member-days are the walk's, no more: 610,842 checked, 0 failure(s)
  ok  no day on or after a split carries a pre-split count, and the split's day carries the count before it times the ratio (25,350 of the days checked member-days): 25,936 checked, 0 failure(s)


The first-day check: on each security's first day in the panel, the members whose raw price opened more than 1.5 points below SPY's, the three lowest, each with what explains it -- a spin-off's parent gaps by the child's value on its ex-date: 21 first days, 50 gap(s) nothing on record explains
  2022-02-02 CEG    EXC -29.56% (ruling), PYPL -20.99%, CHRW -11.39%
  2022-04-11 WBD    T -20.98% (ruling), TSLA -3.62%, NVDA -3.14%
  2022-12-15 MBC    FBIN -16.25% (ruling), WDC -4.50%, NFLX -3.67%
  2023-01-04 GEHC   GE -20.12% (split), MSFT -3.67%, CHRW -3.60%
  2023-05-04 KVUE   PARA -17.12%, KEY -7.60%, QCOM -7.53%
  2023-09-12 TKO    ORCL -11.21%, RTX -2.53%, UPS -1.85%
  2023-09-25 CRH    CCL -3.27%, HPQ -1.91%, LVS -1.85%
  2023-10-02 VLTO   DHR -11.52% (ruling), K -6.20% (ruling), NSC -3.74%
  2024-03-21 RDDT   DRI -6.20%, ACN -4.88%, FDS -3.35%
  2024-04-01 SOLV   MMM -14.31% (split), LW -2.92%, T -2.47%
  2024-04-02 GEV    GE -19.11% (split), HUM -9.48%, UNH -5.40%
  2024-07-08 SW     NOW -5.47%, TSLA -1.66%, CSX -1.53%
  2024-08-01 FERG   MRNA -16.79%, WDC -10.33%, IR -8.69%
  2024-09-30 AMTM   J -15.47% (ruling), APTV -3.36%, GM -3.17%
  2025-02-24 SNDK   WDC -23.77% (ruling), PLTR -5.74%, MRNA -4.99%
  2025-08-07 SKYD   FTNT -23.62%, LLY -10.02%, ABNB -8.30%
  2025-10-30 SOLS   CMG -20.85%, BAX -15.01%, EME -13.91%
  2025-11-03 Q      DD -59.27% (split), KMB -13.08%, ARE -2.90%
  2026-06-01 FDXF   FDX -20.92% (ruling), QCOM -6.90%, COIN -5.05%
  2026-06-29 HONA   MLM -6.55%, VMC -3.78%, CRH -3.42%
  2026-10-01 VYLR   CTVA -81.63% (ruling), MRNA -3.32%, SKYD -2.36%
```

### weighted_diff.txt -- the weighted-count check, `_listed`'s build against `_spinoffs`'

```text
2,3c2,3
<   carried from an earlier record: 46 met, 7 flagged; ratio median 1.0002, p1 0.4985, p99 2.2713
<   the day's own record: 11,847 met, 754 flagged; ratio median 1.0000, p1 0.9825, p99 2.0918
---
>   carried from an earlier record: 49 met, 6 flagged; ratio median 1.0003, p1 0.9361, p99 2.2242
>   the day's own record: 11,849 met, 754 flagged; ratio median 1.0000, p1 0.9825, p99 2.0917
17a18,19
>     FDXF   FDXF   2026-08-26 weighted 149,518,133 against 119,303,378: 1.2533  FLAGGED
>     FDXF   FDXF   2026-11-03 weighted 149,521,266 against 119,303,378: 1.2533  FLAGGED
22,24c24
<     HON    HON    2026-07-14 weighted 316,826,559 against 635,675,701: 0.4984  FLAGGED
<     HON    HON    2026-10-12 weighted 316,940,010 against 635,675,701: 0.4986  FLAGGED
<     HON    HON    2026-11-03 weighted 316,940,010 against 635,675,701: 0.4986  FLAGGED
---
>     HON    HON    2026-07-14 weighted 316,826,559 against 317,837,850: 0.9968
45a46,48
>     VYLR   VYLR   2026-10-02 weighted 711,546,000 against 668,820,000: 1.0639
>     VYLR   VYLR   2026-10-05 weighted 711,546,000 against 668,820,000: 1.0639
>     VYLR   VYLR   2026-11-03 weighted 667,302,930 against 668,820,000: 0.9977
124c127
< The first-day check: on each security's first day in the panel, the members whose raw price opened more than 1.5 points below SPY's, the three lowest, each with what explains it -- a spin-off's parent gaps by the child's value on its ex-date: 21 first days, 59 gap(s) nothing on record explains
---
> The first-day check: on each security's first day in the panel, the members whose raw price opened more than 1.5 points below SPY's, the three lowest, each with what explains it -- a spin-off's parent gaps by the child's value on its ex-date: 21 first days, 50 gap(s) nothing on record explains
```

### model_compare.txt -- the defaults' model on `_spinoffs` (M14) against midoffice13's final defaults on `_listed` (FD)

```text
claude/scratch/midoffice13/final/FD against claude/scratch/midoffice14/model/M14: 21 fields compared; equal: 11; differing: covariance, observed_variance, sampling_variance, weighted_sampling_variance, noise_share, weighted_noise_share, mean_r2, factor_returns, specific, exposures
  covariance: largest difference 4.551e-04
  observed_variance: largest difference 4.551e-04
  sampling_variance: largest difference 9.325e-05
  weighted_sampling_variance: largest difference 4.177e-04
  noise_share: largest difference 6.393e-03
  weighted_noise_share: largest difference 1.679e-02
  mean_r2: largest difference 6.229e-04
  factor_returns: largest difference 2.554e-02
  specific: largest difference 8.028e-01
  exposures: largest difference 3.383e+00
  qualifications: 6 and 6; differing: 1
  report: 162 and 162 lines; 112 differing line(s)
    -  window descriptors need at least 50% of the window valid; from a window with none, over 5,162,627 values: none; from one under 50% valid: none
    -  454,187 raw values winsorised before standardising
    -  183,582 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    -    LNCAP 18/51  LCAPCB 51/51  LPRI 3/51  DASTD 3/54  HILO 8/51  CMRA 4/53  BTSG 7/57
    -    HALPHA 3/57  SERDP 0/57  RSTR 4/57  STOM 15/52  STOQ 13/54  STOA 17/57  VLVR 18/57
    +  window descriptors need at least 50% of the window valid; from a window with none, over 5,162,837 values: none; from one under 50% valid: none
    +  454,423 raw values winsorised before standardising
    +  183,708 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    +    LNCAP 18/51  LCAPCB 51/51  LPRI 3/51  DASTD 2/54  HILO 8/51  CMRA 4/54  BTSG 7/57
    +    HALPHA 2/57  SERDP 0/57  RSTR 3/57  STOM 15/52  STOQ 13/54  STOA 17/57  VLVR 18/57
    -  BTSG         94%      962    8.58%     10.03%     1.09
    -  DASTD        94%    1,149    6.51%      8.42%     1.04
    -  RSTR         94%      938    5.49%      7.78%     1.04
    -  HALPHA       94%      962    4.42%      6.67%     0.56
    +  BTSG         94%      962    8.60%     10.03%     1.08
    +  DASTD        94%    1,149    6.54%      8.42%     1.03
    +  RSTR         94%      938    5.50%      7.78%     1.05
    +  HALPHA       94%      962    4.43%      6.67%     0.56
    -  CMRA         93%      938    3.74%      6.26%     1.16
    -  STOQ         94%    1,145    3.60%      5.92%     0.68
    -  STOM         93%    1,191    3.47%      5.88%     0.69
    +  CMRA         93%      938    3.75%      6.27%     1.15
    +  STOQ         94%    1,145    3.61%      5.92%     0.68
    +  STOM         94%    1,191    3.48%      5.87%     0.69
    -  LPRI         94%    1,213    1.86%      4.03%    -0.91
    +  LPRI         94%    1,213    1.86%      4.03%    -0.94
    -  LCAPCB       93%    1,213    1.65%      3.93%     0.41
    -  VLVR         94%      938    1.55%      3.71%     1.55
    +  LCAPCB       93%    1,213    1.65%      3.93%     0.42
    +  VLVR         94%      938    1.56%      3.71%     1.54
    -  mean cross-sectional R-squared 0.293
    +  mean cross-sectional R-squared 0.294
    -  Volatility                  +2.2e-17   1.000000   503
    +  Volatility                  -4.0e-17   1.000000   503
    -  Size                        +0.0e+00   1.000000   500
    -  Size Nonlinearity           -3.6e-17   1.000000   500
    -  Trading Activity            -1.4e-17   1.000000   501
    -  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,106 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates
    +  Size                        +7.2e-17   1.000000   502
    +  Size Nonlinearity           -2.2e-17   1.000000   502
    +  Trading Activity            -2.2e-17   1.000000   503
    +  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,107 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates
    -  smallest 85, equally weighted                -3.137             -1.808   10.99%
    -  largest 50, equally weighted                 +0.167             +0.052    0.57%
    -  the index, cap-weighted                      +0.001             -0.001    0.01%
    +  smallest 85, equally weighted                -3.137             -1.806   11.05%
    +  largest 50, equally weighted                 +0.172             +0.051    0.60%
    +  the index, cap-weighted                      +0.003             -0.001    0.01%
    -    no weight                   224   FDXF 108, VYLR 23, SKYD 22, Q 13, SOLS 13, AMTM 5
    +    no weight                    93   SKYD 22, SOLS 13, Q 13, AMTM 5, GEV 4, JNPR 4
    -    an exposure missing       1,051   SW 126, SOLV 125, VLTO 124, SKYD 124, GEV 122, Q 113
    -    the first exposure missing: Momentum 1,051
    +    an exposure missing       1,182   SW 126, SOLV 125, SKYD 124, VLTO 124, GEV 122, Q 113
    +    the first exposure missing: Momentum 1,182
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
    +  Construction               22.23%     +25.53%    28.9%      37.8%      938          0      0
    +  Finance & Real Estate      17.34%     +17.97%    24.1%      37.1%      938          0      0
    +  Manufacturing              12.07%     +20.71%    16.5%      24.3%      938          0      0
    +  Mining                     21.16%     +14.75%    14.2%      21.4%      938          0      0
    +  Retail                     14.75%     +14.55%     8.4%      12.9%      938          0      0
    +  SIC20                      13.35%     +10.88%    26.7%      45.3%      938          0      0
    +  SIC28                      16.14%     +16.53%     6.7%      10.6%      938          0      0
    +  SIC35                      18.56%     +29.73%     5.3%       7.3%      938          0      0
    +  SIC36                      24.22%     +32.61%     3.4%       4.1%      938          0      0
    +  SIC37                      19.59%     +15.49%    10.0%      14.8%      938          0      0
    +  SIC38                      16.82%     +16.01%     8.0%      12.3%      938          0      0
    +  SIC49                      19.02%     +23.73%     7.7%      11.9%      938          0      0
    +  SIC60                      23.21%     +21.77%     6.5%       9.5%      938          0      0
    +  SIC62                      21.57%     +18.48%     8.6%      11.5%      938          0      0
    +  SIC63                      14.22%     +19.00%    15.1%      24.2%      938          0      0
    +  SIC67                      17.83%     +17.79%    11.8%      18.3%      938          0      0
    +  SIC73                      17.53%     +15.26%     3.1%       4.9%      938          0      0
    +  Services                   14.47%     +15.94%    21.5%      33.5%      938          0      0
    +  Transport & Utilities      16.26%     +15.54%    11.2%      17.5%      938          0      0
    +  Wholesale                  15.06%     +20.46%    39.5%      60.4%      938          0      0
    +  Volatility                 10.62%      +2.97%     2.7%       4.5%      938          0      0
    +  Momentum                    7.49%      +4.29%     3.4%       4.8%      938          0      0
    +  Size                        3.62%      +3.05%     8.0%      11.4%      938          0      0
    +  Size Nonlinearity           2.83%      -0.48%    31.0%      45.9%      938          0      0
    +  Trading Activity            5.32%      +2.50%    10.3%      14.7%      938          0      0
    -  specific risk   median 28.98%   p10 16.58%   p90 48.98%
    -  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    +  specific risk   median 28.99%   p10 16.56%   p90 48.89%
    +  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, SMCI 1.9%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    -                    each history weighted by a half-life of 63 trading days; the characteristics' weight in a blended forecast, at the median effective history of 181.8 observations of 938: 13.47%
    +                    each history weighted by a half-life of 63 trading days; the characteristics' weight in a blended forecast, at the median effective history of 181.8 observations of 938: 13.59%
    -  - 61% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is biased upward and is not corrected here
    +  - 60% of the estimated variance of Wholesale is the sampling error of its own estimation, weighted as the covariance weights the periods; the factor covariance diagonal is biased upward and is not corrected here
```

### compare_bias.txt -- the defaults' bias run on `_spinoffs` (D14) against midoffice13's D, every statistic recomputed

```text
D13: midoffice13's defaults on _listed; D14: the defaults on _spinoffs. 686 forecast dates, 2024-02-08 to 2026-11-02; the first half to 2025-06-23, the second from 2025-06-24

The minimum-variance portfolios at one day: the bias statistic, the volatility a year (the forecast at its median, the realised from the returns), by part of the test
                                                                       D13                    D14
  minimum variance, long-only          whole           1.167  9.42% 10.93%    1.059  9.42%  9.92%
  minimum variance, long-only          first half      1.246  8.29% 11.47%    1.247  8.28% 11.47%
  minimum variance, long-only          second half     1.084  9.55% 10.38%    0.831  9.54%  8.09%
  minimum variance, unconstrained      whole           1.220  9.13% 11.07%    1.101  9.12% 10.01%
  minimum variance, unconstrained      first half      1.297  8.05% 11.56%    1.298  8.05% 11.57%
  minimum variance, unconstrained      second half     1.139  9.30% 10.57%    0.862  9.29%  8.16%
  minimum variance, long-only, D13: its three lowest outcomes in forecast sd -- 2026-09-30 -13.0, 2025-04-03 -9.4, 2024-08-02 -4.8
  minimum variance, long-only, D14: its three lowest outcomes in forecast sd -- 2025-04-03 -9.4, 2024-08-02 -4.8, 2024-12-17 -4.4
  minimum variance, unconstrained, D13: its three lowest outcomes in forecast sd -- 2026-09-30 -13.9, 2025-04-03 -9.8, 2024-08-02 -4.7
  minimum variance, unconstrained, D14: its three lowest outcomes in forecast sd -- 2025-04-03 -9.8, 2024-08-02 -4.7, 2024-12-17 -4.4

  without the forecast of 2026-09-30 (realised 2026-10-01, CTVA's ex-date), D13 against D14 -- the statistic, and the realised volatility a year:
    minimum variance, long-only              1.055, 9.89%     1.059, 9.92%
    minimum variance, unconstrained          1.097, 9.97%    1.102, 10.01%

The minimum-variance portfolios at the other horizons, the windows overlapping: D13 / D14
  minimum variance, long-only           5 days 1.173/1.074  11 days 1.078/0.991  21 days 1.005/0.917
  minimum variance, unconstrained       5 days 1.223/1.112  11 days 1.110/1.011  21 days 1.045/0.941

The broad portfolios: the statistic at one day by part, and at 21 days -- D13 / D14
  the index, cap-weighted                1.015/1.015    1.224/1.224    0.751/0.751    0.860/0.860   (whole, first, second; 21 days)
  the index, equally weighted            0.965/0.966    1.142/1.143    0.749/0.749    0.805/0.805   (whole, first, second; 21 days)
  smallest 85, equally weighted          0.956/0.956    1.064/1.065    0.835/0.834    0.775/0.776   (whole, first, second; 21 days)
  largest 50, equally weighted           0.996/0.996    1.175/1.175    0.778/0.778    0.873/0.873   (whole, first, second; 21 days)
  the index, cap-weighted              volatility a year, forecast / realised: D13 15.48% / 15.27%, D14 15.48% / 15.27%
  the index, equally weighted          volatility a year, forecast / realised: D13 13.98% / 13.95%, D14 13.98% / 13.96%
  smallest 85, equally weighted        volatility a year, forecast / realised: D13 17.70% / 17.38%, D14 17.70% / 17.38%
  largest 50, equally weighted         volatility a year, forecast / realised: D13 15.94% / 14.64%, D14 15.94% / 14.64%

Every member singly (at least 63 forecasts):
  whole        543 / 543 members: median 1.003 / 1.003, mean abs(bias - 1) 0.090 / 0.088
  first half   518 / 518 members: median 1.037 / 1.037, mean abs(bias - 1) 0.105 / 0.105
  second half  519 / 519 members: median 0.967 / 0.965, mean abs(bias - 1) 0.103 / 0.101
  the members whose statistic moved most, D13 -> D14: CTVA 1.828 -> 0.874, K 0.787 -> 0.830, FDX 1.081 -> 1.041, J 0.779 -> 0.748, BDX 1.043 -> 1.016, DHR 1.080 -> 1.103, BWA 0.808 -> 0.831, WDC 1.166 -> 1.144, SPGI 1.010 -> 1.021, LEN 1.100 -> 1.110, CMCSA 1.025 -> 1.029, APTV 0.959 -> 0.963

Each spin-off parent with an ex-date in the test, its member statistic D13 -> D14, and its outcome on the day in forecast sd:
  J      0.779 -> 0.748; 2024-09-30: -7.8 -> +2.4 sd; 2025-05-16: +0.1 -> +0.6 sd
  LEN    1.100 -> 1.110; 2025-01-21: -1.6 -> +1.9 sd
  WDC    1.166 -> 1.144; 2025-02-24: -11.5 -> -2.0 sd
  CMCSA  1.025 -> 1.029; 2026-01-05: -2.7 -> +0.4 sd
  BDX    1.043 -> 1.016; 2026-02-10: -10.0 -> +2.4 sd
  APTV   0.959 -> 0.963; 2026-04-01: -4.4 -> +1.2 sd
  FDX    1.081 -> 1.041; 2026-06-01: -9.5 -> +0.2 sd
  HON    0.868 -> 0.867; 2026-06-29: -1.0 -> -1.9 sd
  SPGI   1.010 -> 1.021; 2026-07-01: +1.0 -> +3.8 sd
  CTVA   1.828 -> 0.874; 2026-10-01: -41.9 -> +2.0 sd

The industries, mean abs(bias - 1) at one day: 0.046 / 0.046
Random portfolios, long-only, median at one day: 0.971 / 0.971
Random portfolios, long-short, median at one day: 1.004 / 1.004
Eigenportfolios 1-5, mean at one day: 1.030 / 1.030
Eigenportfolios 6-10, mean at one day: 1.082 / 1.083
Eigenportfolios 11-15, mean at one day: 1.114 / 1.114
Eigenportfolios 16-20, mean at one day: 1.216 / 1.217
Eigenportfolios 21-25, mean at one day: 1.194 / 1.194

The arrays' equality, D13 and D14: {'caps': False, 'held_long_only': False, 'held_unconstrained': True, 'repaired': True}
```

### bias/report_D14.txt -- D14's own report: its guard, its portfolios with ruling 5's columns, every horizon

```text
  guard: as of 2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02, the model's inputs rebuilt from the panel cut at the date, the scheme held, equal the full run's through it to the bit -- returns, exposures, weights, capitalisations, periods -- and so do the forecasts made from them: covariance, exposures, specific risk, factor returns
  686 refits in 7.3 min; the factor covariance as the engine uses it, repaired to positive semi-definite at the configured floor, on 0 of the 686 dates; 18 member-day(s) with a forecast and no next-day return left out of every portfolio formed that day

  Portfolios at one day, each formed at its forecast date and held to the next; beside each, its volatility % a year -- the forecast at its median, the realised from its returns
                                         T    bias           95% band    MRAD    min    max   forecast realised
  the index, cap-weighted              686   1.015    in  0.946-1.054   0.198  0.721  1.337     15.48%   15.27%
  the index, equally weighted          686   0.966    in  0.946-1.054   0.177  0.743  1.250     13.98%   13.96%
  smallest 85, equally weighted        686   0.956    in  0.946-1.054   0.129  0.797  1.152     17.70%   17.38%
  largest 50, equally weighted         686   0.996    in  0.946-1.054   0.164  0.704  1.281     15.94%   14.64%
  minimum variance, long-only          686   1.059 under  0.946-1.054   0.229  0.774  1.376      9.42%    9.92%
  minimum variance, unconstrained      686   1.101 under  0.946-1.054   0.247  0.814  1.434      9.12%   10.01%
    minimum variance, long-only: 87 names held at the median (52 to 164); its KKT conditions met to 3.0e-13; realised 9.92% a year, the first half 11.47% (bias 1.247), the second 8.09% (0.831)
    minimum variance, unconstrained: 501 names held at the median (499 to 503); its KKT conditions met to 3.7e-13; realised 10.01% a year, the first half 11.57% (bias 1.298), the second 8.16% (0.862)
[...]
  At every horizon evaluated, each forecast at h times its daily variance: the statistic with the windows overlapping / apart
                                             1 day        5 days       11 days       21 days
  the index, cap-weighted                    1.015   0.958/0.978   0.914/0.838   0.860/0.848
  the index, equally weighted                0.966   0.927/0.929   0.873/0.844   0.805/0.776
  smallest 85, equally weighted              0.956   0.909/0.905   0.837/0.849   0.776/0.722
  largest 50, equally weighted               0.996   0.945/0.961   0.903/0.843   0.873/0.915
  minimum variance, long-only                1.059   1.074/1.094   0.991/1.008   0.917/0.900
  minimum variance, unconstrained            1.101   1.112/1.121   1.011/1.046   0.941/0.977
  random long-only, median                   0.971   0.934/0.943   0.891/0.870   0.832/0.813
  random long-short, median                  1.004   1.000/1.003   0.979/0.967   0.971/0.996
  every member singly, median                1.003   0.970/0.979   0.955/0.954   0.938/0.947
  industries, mean abs(bias - 1)             0.046   0.046/0.048   0.062/0.090   0.079/0.129
  eigenportfolios 1-5, mean                  1.030   0.974/0.980   0.946/0.917   0.913/0.875
  eigenportfolios 21-25, mean                1.194   1.198/1.199   1.171/1.158   1.166/1.204
```

### bias/runner_log.txt -- the two runs, one at a time

```text
2026-11-12 16:39:41  runner started, pid 22128, for R4; 5.1 GB available
2026-11-12 16:39:41  R4: 5.1 GB available
2026-11-12 16:39:41  R4: attempt 1: data/external/polygon/sp500_securities_2022-01-03_2026-11-03_listed.npz --bias-guard 0 --horizon 21 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-12 16:48:50  R4: done in 9.2 min
2026-11-12 16:48:50  runner finished; every run done
2026-11-12 16:57:51  runner started, pid 5096, for D14; 5.4 GB available
2026-11-12 16:57:51  D14: 5.4 GB available
2026-11-12 16:57:51  D14: attempt 1: --bias-guard 5 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-12 17:08:52  D14: done in 11.0 min
2026-11-12 17:08:52  runner finished; every run done
```

### final_checks.txt -- the final code: both panels rebuilt, the defaults and the 21-day view re-captured; the suite, ruff, hygiene

```text
build _listed: exit 0
20
build _spinoffs: exit 0
20
FD14: status 0, mean R2 0.293826, 938 periods, 25 factors, 60s, the estimator 1.0s
claude/scratch/midoffice14/model/M14 against claude/scratch/midoffice14/final/FD14: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
FH21: status 0, mean R2 0.293203, 938 periods, 25 factors, 62s, the estimator 1.2s
claude/scratch/midoffice13/i5/V21N against claude/scratch/midoffice14/final/FH21: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 2 differing line(s)
    -  the horizon: 21 trading day(s), a reporting view, not the model's horizon of one day: broad portfolios' risk over a holding period of 21 days. Its gain over the one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025 shock in it, and is re-tested as data accumulates -- the factor volatilities Newey-West at 20 lag(s) with Bartlett weights, each the per-day variance of 21-day returns; the factor correlations one-day, with no lags: the lags on the volatilities alone; the specific risk one day's variance, with no lags, at every horizon
    +  the horizon: 21 trading days, a reporting view, not the model's horizon of one day: broad portfolios' risk over a holding period of 21 days. Its gain over the one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with the April 2025 shock in it, and is re-tested as data accumulates -- the factor volatilities Newey-West at 20 lag(s) with Bartlett weights, each the per-day variance of 21-day returns; the factor correlations one-day, with no lags: the lags on the volatilities alone; the specific risk one day's variance, with no lags, at every horizon
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2362 passed, 1 skipped, 19 warnings in 94.76s (0:01:34) ===========
All checks passed!
ruff: clean over the changed Python files
scripts/run_sp500_model.py: CRLF (2285 lines), parses, ASCII
scripts/build_sp500_securities.py: CRLF (735 lines), parses, ASCII
src/praxis/midoffice/refdata/securities.py: CRLF (1219 lines), parses, ASCII
tests/midoffice/test_securities.py: CRLF (557 lines), parses, ASCII
config/sp500_spin_off_overrides.json: LF (262 lines), -, ASCII
docs/FACTOR_MODEL.md: CRLF (1466 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (683 lines), -, ASCII
claude/handoffs/BRIEF_midoffice14_spin_offs.md: LF (75 lines), -, ASCII
claude/retros/RETRO_midoffice14_spin_offs.md: LF (657 lines), -, ASCII
all ok
exit 0
```

## git add (always last -- Rule 43)

This piece -- midoffice13's rulings 4 and 5, the spin-off rulings and the build that carries them, the default panel, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add scripts/run_sp500_model.py scripts/build_sp500_securities.py src/praxis/midoffice/refdata/securities.py tests/midoffice/test_securities.py config/sp500_spin_off_overrides.json docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice14_spin_offs.md claude/retros/RETRO_midoffice14_spin_offs.md
```
