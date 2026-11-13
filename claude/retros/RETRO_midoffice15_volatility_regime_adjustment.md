# Retro: Mid-office 15 -- midoffice14's rulings, then the volatility regime adjustment

**Brief:** `claude/handoffs/BRIEF_midoffice15_volatility_regime_adjustment.md` -- written in Chat
and pasted into the Code session, transcribed verbatim
**Date:** 2026-11-12 (EST)
**Duration:** the brief arrived shortly before 23:20Z (18:20 EST), the first clock reading after
it.
- **The rulings:** read and built by 18:43, when the revised panel was written into `data/`.
- **The regime adjustment:** built, its tests written and three smoke runs made by 19:06.
- **The runs, one at a time through the runner:** the defaults on the revised panel (D15, 19:07
  to 19:18), the half-life grid (V15, to 19:32), and the eigenvalue adjustment on the chosen
  setting (VE15, 19:38 to 19:54).
- **Then** the analyses, the docs, the state file and this retro.

**Status:** COMPLETE.
- **Rulings 1 to 6:** built and measured. S&P Global's ruling is held, FedEx Freight's count
  grossed up, Honeywell's count ruling marked superseded, the kept rulings marked so, and the
  price series left, none read across a ruled ex-date.
- **Items 1 to 3:** built and measured -- the volatility regime adjustment an option, off; its
  half-lives tested in one pass; the eigenvalue adjustment one-sided on the chosen setting.
- **No default changed** beyond the rulings: the run's panel is the revised one; nothing else.

Uncommitted: one `git add` at the end, with `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 1, S&P Global's Mobility: held.** The record makes 2026-07-01 the first day SPGI
  trades without Mobility (SPGI WI from about 06-26), so the record date is not the ex-date, as
  Lennar's was. And the data show no fall of Mobility's value on either day:
  - **on the record date** SPGI opened with its peers (0.07 points apart);
  - **on 07-01** it opened 2.63 points below them, about half of Mobility's 4.98%, and took that
    back the next day;
  - **over the two days** its raw return, +8.01%, is its six peers' (+7.14% to +8.70%) without
    the distribution, +13.53% with it.

  The ruling stays in the file, marked held with what the data show; the build lists it and does
  not apply it.
- **Ruling 2, Honeywell:** the reverse split stands; midoffice08's count override is marked
  superseded and wrong in its file, kept as the record so `_listed` still rebuilds.
- **Ruling 3, FedEx Freight's count:** the 119,303,378 shares distributed over the 0.801 FedEx did
  not keep, 148,943,044 (a `retained` share on the ruling), 0.9962 of its weighted count, which no
  longer flags it.
- **Rulings 4 and 5:** Lennar at Millrose's first close and the seven Polygon carries, as built;
  the kept rulings say so in their `ruled` fields.
- **Ruling 6, the price series:** at the defaults no descriptor reads the stitched prices, and
  the panel's highs and lows are the raw bars', read by HILO within each day. Blanking the
  stitched prices leaves every descriptor the same to the bit; scaling the highs and lows before
  every ruled ex-date moves HILO by 2e-16. So nothing reads them across a ruled ex-date, and they
  are left.
- **The revised panel, `_revised`, the run's default:** SPGI's 2026-07-01 back to Polygon's
  +1.89% and FDXF's 109 counts, nothing else. The defaults' bias run on it (D15) is D14's to the
  third decimal.
- **Item 1, the volatility regime adjustment,** an option, off, each side separately.
  - **The statistics:** each day's factor and specific cross-sectional bias statistics are read
    from the forecasts the bias test's own refits made the day before, unadjusted.
  - **The multiplier:** the root of their exponentially weighted mean square, one until 21 days
    of history. The test's first 21 forecast dates, 2024-02-08 to 2024-03-08, are unadjusted.
  - **Out of sample:** a test redraws everything after a date and finds every multiplier as of it
    unchanged, to the bit.
- **Item 2, the half-lives:** every pairing of 21, 42 and 84 days on the two sides, either side
  off among them, in one pass (V15, 16 tests).
  - **The criterion's choice:** the one fixed beforehand chose the factor side at 21 days and the
    specific at 42 on the first half -- 0.094 against the model's 0.172 -- and it holds on the
    second (0.128 against 0.141).
  - **The factor side at 21 days** is the best in both halves, alone or with any specific side.
    It brings each year's level toward one: the broad portfolios' MRAD 0.167 to 0.094; by half
    the index 1.224 and 0.751 to 1.063 and 0.785, the optimiser's first half 1.25 and 1.30 to
    1.11 and 1.16.
  - **It costs:**
    - the index's whole-test statistic, 1.015 to 0.934, below its band;
    - about 0.2 of a point a year more realised by the optimiser;
    - the index's forecast moving 1.00% a day at the median, against the model's 0.27%.
  - **The calm half is a shape, not a level:** the market-like eigenportfolio reads 0.78 while
    every other rank reads 0.99 to 1.37, and one multiplier cannot lower the one and raise the
    rest.
  - **The specific side is a level, not a regime:** its squared statistic is 1.09 in both halves,
    cap-weighted -- the largest members' specific risk forecast too low -- and raising every
    member's to fix it pushes the median member to 0.976.
- **Item 3, the eigenvalue adjustment one-sided** on the factor side at 21 days, each scale's
  factor multiplier levelled again by its own forecasts' history (VE15):
  - it shapes the eigenportfolios -- their mean abs(bias - 1) 0.131 to 0.076 at a = 3;
  - it fails the test the brief puts first: the optimiser realises more risk at every a,
    long-only 10.13% a year to 10.17%, 10.23% and 10.30%, its statistics unmoved;
  - off, as midoffice13's two-sided one is.
- **Which I would choose:** the regime adjustment's factor side on 21 days, its specific side off,
  the eigenvalue adjustment off -- and next, the calm half's shape, by a shorter correlation
  half-life on the regime-adjusted model.
- **Tests:** 2,372 passed, 1 skipped; 10 new.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice15_volatility_regime_adjustment.md`, verbatim. In
outline:

- **"committed -- midoffice14."** Checked: d2b2864 in `git log`.
- **Rulings on midoffice14's decisions:**
  1. Keep Jacobs' second Amentum distribution, BD's Waters shares, Aptiv's Versigent and
     Honeywell's Aerospace. Hold S&P Global's -- no gap on 2026-07-01 -- and find the gap in
     SPGI's raw prices around its record date, 2026-06-15; if it is there, move the ruling to that
     day at Mobility's first close, as Lennar's; if the data show no gap on either day, report
     what they show and leave the ruling out until it is understood.
  2. Honeywell: the reverse split stands; midoffice08's count ruling is superseded, and was wrong.
  3. FedEx Freight's count grossed up by FedEx's retained 19.9%, to about 148.9 million, checked
     against its weighted count; Vylor as built.
  4. Lennar at Millrose's first close, as built.
  5. The seven Polygon carries left as they are.
  6. Price returns: say which series each descriptor reads; if any reads the stitched prices,
     highs or lows across a ruled ex-date, apply the rulings to those series as price factors, as
     Polygon's own fractional records are applied; if none does, leave them.
- **midoffice15, the volatility regime adjustment,** the rulings first:
  1. USE4's adjustment, an option, off, for the factor covariance and the specific risk
     separately:
     - each day's factor cross-sectional bias, the root mean square over the factors of each
       factor's return over its forecast volatility from the day before;
     - its multiplier the root of the exponentially weighted mean of the squared statistic on a
       short half-life, scaling the factor covariance by its square;
     - the specific side the same over the members, cap-weighted;
     - out of sample throughout, a day's standardised returns on the forecasts made the day
       before, unadjusted;
     - say how long the history must be before the multiplier starts and what it is before then;
     - a test that nothing after a date moves the multiplier as of it.
  2. Half-lives 21, 42 and 84 days on each side, one pass serving every half-life; chosen on the
     test's first half and checked on the second.
     - **Judged by:** the rolling 12-month windows' mean absolute deviation from one; the
       statistics by half; the broad portfolios, the members, and the minimum-variance
       portfolios' statistics and realised volatility.
     - **Also report:** the multipliers' path over the test, and how much each setting moves the
       forecasts from day to day.
  3. Then the eigenvalue adjustment one-sided on the regime adjustment I would choose: gamma =
     max(1, a (lambda - 1) + 1), a = 1, 2, 3, judged by the same test, the minimum-variance
     portfolios' realised volatility first.
- Runs one at a time through the runner. **"Change no default beyond rulings 1 to 6. Report, and
  say which you would choose and why."** Retro with its `git add` and the state file in the same
  commit, then stop.

---

## Files modified

| File | Change | Lines (hunks) |
|---|---|---|
| `src/praxis/midoffice/model/regime.py` | new: the volatility regime adjustment -- `factor_bias`, `specific_bias`, `regime_multiplier`, `RegimeHistory` (`add`, `add_refit`, `multiplier`, `path`), `regime_history` (the as-of reading the bias test makes), `REGIME_MIN_DAYS`, `REGIME_HALF_LIVES` | new, 244 lines |
| `src/praxis/midoffice/model/bias.py` | `RiskForecast.regime_adjusted`; `from_model(eigen_one_sided=)` | 42, 229-287 |
| `src/praxis/midoffice/model/eigen.py` | one-sided: `scales`, `adjusted`, `basis` take `one_sided`; the docstring | 26-30, 74-90 |
| `src/praxis/midoffice/model/crosssectional.py` | `eigen_one_sided` option, its note and qualification, refused without the adjustment | 189-190, 423-428, 902, 1010-1015, 1308, 1495 |
| `scripts/run_sp500_model.py` | `DEFAULT_PANEL` `_revised`; `--regime-factor-half-life`, `--regime-specific-half-life`, `--bias-regime-half-lives`, `--eigen-one-sided` and their checks; `forecast_rows`, `regime_step`, `regime_multipliers`, `regime_label`, `report_regime_forecast`, `regime_section`; the bias loop's histories and regime tests; `_reading` (was `_eigen_reading`) with the MRADs and `REGIME_CRITERION`; `_report_regime`, `regime_path`, `_daily_moves`, `REGIME_COLUMNS`, `_regime_suffix`; `_report_eigen` one-sided, regime-levelled, a chosen three ways; `_save_bias` with the regime tests and histories | 30, 71, 86-92, 170-180, 827-922, 936-1122, 1251-1264, 1429-1753, 1907-1971, 2307-2326, 2689-2690 |
| `src/praxis/midoffice/refdata/securities.py` | `SpinOff.retained`, `.held`; `read_spin_offs` reads and checks them; `_refuse_held` in `spin_off_splits` and `build_panel`; the seeded count over 1 - retained | 435-441, 459-470, 484-493, 506-535, 585-586, 726-728, 931-952 |
| `scripts/build_sp500_securities.py` | held rulings listed and not applied; the seeded line's gross-up; the JSON's `retained` and `held_spin_offs`; `--suffix` `_revised`; the docstring | 49-61, 153-165, 410-443, 687-691 |
| `tests/midoffice/test_regime.py` | new: 6 tests | new, 209 lines |
| `tests/midoffice/test_securities.py` | 2 new tests; `spin_off(retained=, held=)` | 397-400, 558-626 |
| `tests/midoffice/test_eigen.py` | 1 new test | 118-138 |
| `tests/midoffice/test_bias.py` | 1 new test | 357-397 |
| `config/sp500_spin_off_overrides.json` | SPGI's ruling `held`, with what the data show; FDX's `retained` 0.199 and its evidence; the kept rulings' `ruled` marked; the description | 2, 132-247 |
| `config/sp500_share_count_overrides.json` | HON's entry `superseded`, with ruling 2 | 13-14 |
| `docs/FACTOR_MODEL.md` | the one-sided eigen adjustment; the volatility regime adjustment's section; the spin-off rulings revised; ruling 6's table; the weighted check; the revised panel; the bias section's midoffice15 results | 387-426, 1199-1255, 1274-1276, 1310-1316, 1525-1585 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice15_volatility_regime_adjustment.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice15_volatility_regime_adjustment.md` | new: this retro | new |

Written outside the tracked files:

- `data/external/polygon/sp500_securities_2022-01-03_2026-11-03_revised.npz` and `.json`, the
  revised panel and its report -- new files; every earlier panel is untouched (`_spinoffs` and
  `_listed` rebuild to the bit in the final checks).
- The cache: nothing asked, nothing written.
- `claude/scratch/midoffice15/` (gitignored): the scratch, below.

---

## Ruling 1: S&P Global's Mobility, held

**The record** (`spgi_record.txt`; one search, its summary quoting S&P Global's release of
2026-05-21 and the reports of the completion):
- **The terms:** one Mobility share for each SPGI share held at the close on 2026-06-15, the
  record date, distributed 12:01 a.m. ET on 2026-07-01.
- **Two ways to trade SPGI** from about 06-26 through 06-30: regular-way with the entitlement,
  and "SPGI WI" without it.
- **So 07-01 is the first day SPGI trades without Mobility**, the ex-date ruled in midoffice14.
- **Unlike Lennar's,** the record date is not the ex-date. No result gives a price after 07-01.

**The data** (`spgi_gap.txt`, SPGI's raw bars from 06-01 to 07-10 against SPY and its six peers
-- MCO, MSCI, ICE, CME, NDAQ, FDS):

| | SPGI open / close | peers' median open / close | SPGI less the peers |
|---|---|---|---|
| 2026-06-15, the record date | -0.16% / +1.23% | -0.09% / +0.54% | -0.07 / +0.69 points |
| 2026-06-16 | +1.39% / +2.16% | +0.82% / +0.11% | +0.57 / +2.05 |
| 2026-07-01, the ruled ex-date | -1.89% / +1.89% | +0.74% / +4.27% | -2.63 / -2.38 |
| 2026-07-01 and 07-02 together, close to close | +8.01% | +7.86% (+7.14% to +8.70%) | +0.15 |

- **Mobility's size:** 4.98% of SPGI's 06-30 close at its first open (20.30), 5.20% at its first
  close (21.19).
- **07-01's opening gap** below the peers, -2.63 points, is the largest from 06-02 to 07-10 (the
  next -0.80), and about half of Mobility.
- **Over 07-01 and 07-02 together** SPGI's raw move is its peers' without the distribution. With
  it, SPGI's would be +13.53%, 5.7 points above every peer.
- **Nowhere else** -- in the cumulative path from the record date, SPGI against its peers --
  does a 5% fall appear.
- **The cache records nothing:** no split, distribution or event for SPGI near either day, and
  the adjusted bars equal the raw ones throughout the window.

**What the data show, then:** no gap of Mobility's size on the record date or the ruled ex-date.
Jeff's ruling's second branch: the ruling is left out until it is understood.

- **The reading I can offer:** SPGI's raw series before 07-01 may not be the regular-way line's,
  which would carry Mobility. It is not a fact. An outside price for SPGI's regular-way close of
  2026-06-30 would settle it. Polygon has 407.26: if the regular-way close was that, the series
  is the regular-way line's and the gap is missing from the market itself. If it was about 428,
  407.26 plus Mobility's when-issued value, Polygon's series is not the regular-way line's that
  day, and the distribution's fall is not in Polygon's prices.
- **Built:** a ruling can be held (`held`, with why) -- read with the rest, listed in the build's
  report, never applied. `build_panel` and `spin_off_splits` refuse a held ruling outright.
- **SPGI's own entry** keeps its evidence, and its `held` text is what the data show
  (`amend_rulings.txt`).
- **The panel:** SPGI's 2026-07-01 total return is Polygon's +1.8931% again (`_spinoffs` had
  +7.0962%), and SPGI's column equals `_listed`'s to the bit.

## Ruling 2: Honeywell

Nothing to build: the reverse split ruling of midoffice14 stands and supersedes midoffice08's
count override. The override's entry is marked `superseded`, with the ruling: the halved records
were Honeywell's own post-split count and its reading was wrong. It is kept as the record, the
build reading it only under `--spin-offs none`, so midoffice10's `_listed` still rebuilds to the
bit (final checks). The count reader ignores the new field.

## Ruling 3: FedEx Freight's count, grossed up

- **Built:** a ruling's `retained`, the share of the child the parent kept. The seeded count is
  the parent's carried count on the record date times the ratio, over 1 - retained, to the
  nearest share. It is validated as at least 0 and under 1, and only on a seeded count.
- **FDXF:** 238,606,756 x 0.5 = 119,303,378 distributed; / 0.801 = 148,943,043.7, so
  **148,943,044**, over its 109 member-days.
- **Checked against its weighted counts:** 149,518,133 on 2026-08-26 and 149,521,266 on 11-03,
  each 1.0039 times it (the seeded count 0.9962 of them). The weighted check no longer flags
  FDXF: of the 49 carried records it reads, 4 flagged where `_spinoffs` had 6.
- **Vylor,** wholly distributed: 668,820,000, as built.

## Rulings 4 and 5

Nothing to build: Lennar's value at Millrose's first close, and the seven Polygon carries left.
The kept rulings' `ruled` fields say so -- J's of 2025-05-16, BD's of 2026-02-10 and APTV's
(ruling 1), HON's (rulings 1 and 2), LEN's (ruling 4) -- as SPGI's says it is held and FDX's that
it is grossed up (`amend_rulings.txt`).

## Ruling 6: which series each descriptor reads

From the code (`descriptors.compute_descriptors`, the run's `load_panel`) and checked on the data
(`ruling6_series.txt`). At the defaults, `--returns total`, the SecurityPanel the descriptors read
holds the total-return index as its prices, the raw closes, the highs and lows, the traded volumes
and the carried counts -- not the stitched prices:

| Descriptor | Series |
|---|---|
| DASTD, CMRA, RSTR; BTSG, HALPHA, SERDP, VLVR's residual sd (the market model) | the total-return index's daily returns, the rulings in them, against the market's total return |
| LNCAP, LCAPCB | the raw close times the carried count, the day's own |
| LPRI | the raw close, the day's own |
| HILO | the day's raw high over its raw low |
| STOM, STOQ, STOA | the traded volumes over the carried counts |
| VLVR's dollar volume | the traded volume times the raw close |

- **The highs and lows are the raw bars'**, not stitched: the build sets `highs[i, j] = r.high`
  from the raw bar. They equal the cache's raw bars on every applied ruling's ex-date, and the raw
  close lies between them on all 684,087 cells with the three.
- **With the stitched prices blanked everywhere,** every default descriptor is the same to the
  bit.
- **With the highs and lows scaled** before each applied ruling's ex-date by its price factor --
  the rulings applied as Polygon's fractional records are -- HILO moves by 2.2e-16 and nothing
  else.

So none reads the stitched prices, highs or lows across a ruled ex-date, and by the ruling they
are left. Under `--returns price` the stitched prices are the run's returns and every return
descriptor's, by that option's definition.

## The revised panel, and the defaults on it

`_revised` (`data/external/polygon/sp500_securities_2022-01-03_2026-11-03_revised.npz`), built
into scratch first and then into `data/`, the two equal in all 20 arrays:

- **Against `_spinoffs`** (`panel_changed.txt`), 18 of 20 arrays are equal to the bit:
  - **total_return_prices:** one member-day's return beyond 1e-12, SPGI's 2026-07-01 (+7.0962%
    to +1.8931%), and 36 in the last bits, SPGI's index after it rescaled;
  - **shares_outstanding:** FDXF's 109 member-days, 119,303,378 to 148,943,044.
- **Against `_listed`** (`panel_vs_listed.txt`): the 18 applied rulings' returns and 223 count
  member-days (FDXF 109, HON 90, VYLR 24).
- **The build's report:**
  - 19 rulings, 1 held, 18 applied;
  - FDXF's seeded line with its gross-up;
  - count moves 88, as on `_listed` and `_spinoffs`;
  - every guard passing.
- **The model on it (M15) against M14** (`model_compare.txt`):
  - mean R-squared 0.293826 to 0.293821;
  - specific risk 16.56 / 28.99 / 48.89% to 16.56 / 28.98 / 48.89%;
  - the size pair unchanged;
  - the largest exposure change 0.19, FDXF's size. FDXF has no forecast -- it still lacks
    Momentum -- so its count enters through the standardisation's cap-weighted means alone.
- **The defaults' bias run on it (D15, the guard on, 11.4 minutes) against D14**
  (`compare_bias.txt`):
  - SPGI's 2026-07-01 outcome +3.8 to +1.0 sd, its member statistic 1.021 to 1.010;
  - the unconstrained minimum-variance portfolio realises 10.00% a year (10.01%);
  - every other statistic is D14's to the third decimal: the optimiser 1.059 and 1.101, the
    index 1.015, the members' median 1.003.

---

## Item 1: the volatility regime adjustment, built

`src/praxis/midoffice/model/regime.py`, on Menchero, Orr and Wang (2011), the brief's definition:

- **The factor statistic:** each day t, `B_F(t)^2 = (1/K) sum_k (f_k(t) / sigma_k(t-1))^2`, over
  the 25 factors. f(t) is the day's factor returns, the last row of the refit as of t; sigma the
  factor volatilities of the forecast made at t - 1's close, unadjusted, at the day's scale.
- **The specific statistic:** `B_S(t)^2 = sum_n w_n (u_n(t) / s_n(t-1))^2`, sum w = 1, over the
  members at t - 1 with a forecast, a positive capitalisation and a return on day t -- the
  cap-weighted index's own members.
  - u_n(t) = r_n(t) - x_n(t-1)' f(t), the member's specific return on its forecast exposures;
    for a member in day t's regression, exactly its residual before the degrees-of-freedom
    inflation.
  - w is its capitalisation at t - 1: 502 members a day at the median, an effective number of 51.
- **The multiplier as of T:** `lambda(T) = sqrt(sum_t 0.5^((T - t)/h) B(t)^2 / sum_t 0.5^((T -
  t)/h))` over the days through T, the weights normalised over the days there are. The forecast's
  factor covariance is scaled by lambda_F^2 -- its eigenbasis too, where it carries one -- and
  each specific volatility by lambda_S (`RiskForecast.regime_adjusted`).
- **Out of sample throughout:** B(t) reads the forecast as of t - 1 and the day's returns; the
  multiplier as of T reads B through T alone. The forecasts are the bias test's own refits, each
  the whole estimator on data through its date.
- **How long the history must be:** it starts the day after the test's first forecast (2024-02-08,
  the first date whose covariance has a year of estimated periods), and a side's multiplier waits
  for **21 days of it** (`REGIME_MIN_DAYS`).
  - **Before then it is one,** the forecast the model's own: the test's first 21 forecast dates,
    2024-02-08 to 2024-03-08; the first adjusted is 2024-03-11.
  - **Why 21:** the shortest half-life tested, so every setting starts on the same day; a
    21-day history on a half-life of 84 is near an equal-weighted mean of it.
- **The option:** the run's `--regime-factor-half-life H` and `--regime-specific-half-life H`,
  off by default, each side separately.
  - **A plain run** with either refits the model as of every day from the test's first, the
    test's own reading (about eight minutes). It reports the multipliers its own forecast carries
    as of the last date: at the factor side's 21 days, lambda_F 0.9641 as of 2026-11-03, the
    factor volatilities' median 16.26% to 15.67%.
  - **Under `--bias`** the test's loop serves.
  - **`--bias-regime-half-lives H,H`** tests every pairing beside the model in one pass.
  - **Under `--bias-eigen-scales`** each scale's factor side is levelled by its own forecasts'
    history (item 3).
- **The tests** (`test_regime.py`, below) include the out-of-sample one the brief asks for. On a
  synthetic panel the history is built through `cross_sectional_models_as_of`, as the bias test
  reads it (`regime_history`, `add_refit` -- the run's own `regime_step` calls `add_refit`). Then
  everything after a date is redrawn -- returns, characteristics, capitalisations, membership.
  The history through the date, and every multiplier as of a date on or before it, is the same to
  the bit; after it they move.

**Three smoke runs** on the run's own code, before the long ones:
- 60 forecast dates of the grid;
- 40 of the eigen scales on a regime-adjusted model, the guard on;
- 30 of a plain run with the adjustment.

The last two run on the last N forecast rows (`smoke_last.py`), so the run's own forecast as of
the last date follows the last one tested. Every report path ran. The analyses' checks were run
on the smokes' stored arrays before the full runs.

---

## Item 2: the half-lives (V15)

**The run:**
- **The pass:** `--bias --bias-regime-half-lives 21,42,84 --bias-horizons 1,5,11,21`, the guard
  off (D15 ran it); 13.6 minutes; 0.97 s a refit with its 16 tests.
- **Its model test** equals D15's to the bit in every array (`equal_V15_D15.txt`).
- **Every factor-side setting's eigenportfolio forecasts** are the model's times lambda_F,
  recomputed from the stored history, to 4.8e-14 (`analyse_regime.txt`).

**The criterion, fixed in the code before the run** (`REGIME_CRITERION`): the mean of three MRADs
-- the mean absolute deviation from one of the rolling 12-month windows' statistics, over the
part's own windows:
- the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85, the largest
  50, their mean);
- the minimum-variance portfolios' (long-only and unconstrained, their mean);
- every member's singly (their median).

It is chosen on the first half and checked on the second.

### The daily statistics and the multipliers' path

| | whole test | first half | second half |
|---|---|---|---|
| B_F^2, mean | 1.115 | 1.332 | 0.899 |
| B_S^2, mean | 1.089 | 1.088 | 1.090 |

**By calendar quarter, B_F^2 / B_S^2:**

| 2024 Q1 | Q2 | Q3 | Q4 | 2025 Q1 | Q2 | Q3 | Q4 | 2026 Q1 | Q2 | Q3 | Q4 (to 11-02) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.81/0.99 | 0.91/1.13 | 1.36/1.02 | 1.14/1.09 | 1.37/1.19 | 2.10/1.04 | 0.56/1.01 | 0.71/0.98 | 1.20/1.19 | 1.15/1.26 | 0.92/1.09 | 0.95/0.98 |

| lambda | F 21 | F 42 | F 84 | S 21 | S 42 | S 84 |
|---|---|---|---|---|---|---|
| min | 0.776 (2025-10-09) | 0.858 (2025-12-31) | 0.887 (2024-04-01) | 0.908 (2025-07-14) | 0.958 (2024-04-10) | 0.965 (2024-04-10) |
| max | 1.977 (2025-04-10) | 1.628 (2025-04-10) | 1.412 (2025-04-10) | 1.252 (2026-07-31) | 1.168 (2026-07-31) | 1.113 (2026-07-31) |
| mean, first half | 1.121 | 1.098 | 1.073 | 1.045 | 1.044 | 1.041 |
| mean, second half | 0.956 | 0.995 | 1.035 | 1.038 | 1.040 | 1.041 |
| median daily change, abs(d ln lambda) | 0.83% | 0.45% | 0.25% | 0.59% | 0.30% | 0.16% |

The full path by month is in `analyse_regime.txt` (appendix). At 21 days the factor multiplier
runs from 0.87 in March 2024 to 1.69 at the end of April 2025, back to 1.06 by June, 0.79 to 0.81
through the calm second half of 2025, and 1.0 to 1.1 in 2026. The specific multipliers stay near
1.04 on every half-life throughout.

### How much each setting moves the forecasts from day to day

The median absolute change in log forecast volatility from one forecast date to the next (its
95th percentile), for the cap-weighted index and for every member singly (the median of each one's
median):

| setting | the index | p95 | members | p95 |
|---|---|---|---|---|
| the model | 0.27% | 0.97% | 1.34% | 5.67% |
| F 21, S off | 1.00% | 3.58% | 1.48% | 6.39% |
| F 42, S off | 0.67% | 2.45% | 1.43% | 6.21% |
| F 84, S off | 0.49% | 1.85% | 1.40% | 6.13% |
| F off, S 21 | 0.29% | 0.95% | 1.53% | 6.12% |
| F off, S 42 | 0.28% | 0.93% | 1.42% | 5.75% |
| F off, S 84 | 0.27% | 0.95% | 1.38% | 5.66% |
| F 21, S 42 | 0.99% | 3.55% | 1.55% | 6.42% |

### The settings, by part

The criterion, every setting (the report's tables, recomputed in `analyse_regime.txt`):

| setting | whole | first half | second half |
|---|---|---|---|
| the model | 0.166 | 0.172 | 0.141 |
| F off, S 21 / 42 / 84 | 0.162 / 0.162 / 0.162 | 0.165 / 0.165 / 0.165 | 0.146 / 0.145 / 0.145 |
| F 21, S off / 21 / 42 / 84 | 0.105 / 0.104 / 0.103 / 0.103 | 0.098 / 0.094 / 0.094 / 0.094 | 0.123 / 0.128 / 0.128 / 0.127 |
| F 42, S off / 21 / 42 / 84 | 0.114 / 0.113 / 0.112 / 0.113 | 0.111 / 0.105 / 0.105 / 0.105 | 0.131 / 0.135 / 0.135 / 0.134 |
| F 84, S off / 21 / 42 / 84 | 0.126 / 0.125 / 0.125 / 0.125 | 0.124 / 0.117 / 0.117 / 0.118 | 0.147 / 0.154 / 0.152 / 0.151 |

**Chosen on the first half:** the factor side at 21 days and the specific at 42 (0.094; the
model's 0.172; F 21 with S 21 or S 84 the same to the third decimal). On the second half it reads
0.128 against the model's 0.141, the best there the factor side at 21 days alone (0.123).
- **The factor side alone:** 21 days on the first half (0.098), and the best on the second
  (0.123).
- **The specific side alone:** 21 days on the first half (0.165), 0.146 on the second against
  the model's 0.141 -- the best there is the side off.

The model, the factor side alone at 21 days, and the criterion's pick, in full:

| | | the model | F 21, S off | F 21, S 42 |
|---|---|---|---|---|
| the index, cap-weighted | whole / first / second | 1.015 / 1.224 / 0.751 | 0.934 / 1.063 / 0.785 | 0.932 / 1.061 / 0.783 |
| the index, equally weighted | | 0.966 / 1.143 / 0.749 | 0.893 / 0.983 / 0.793 | 0.893 / 0.983 / 0.793 |
| smallest 85 | | 0.956 / 1.065 / 0.834 | 0.902 / 0.921 / 0.882 | 0.901 / 0.920 / 0.881 |
| largest 50 | | 0.996 / 1.175 / 0.778 | 0.920 / 1.024 / 0.804 | 0.918 / 1.022 / 0.802 |
| minimum variance, long-only | | 1.059 / 1.247 / 0.831 | 0.990 / 1.109 / 0.855 | 0.976 / 1.092 / 0.845 |
| minimum variance, unconstrained | | 1.101 / 1.298 / 0.862 | 1.033 / 1.163 / 0.887 | 1.018 / 1.145 / 0.875 |
| long-only realised, % a year | | 9.92 / 11.47 / 8.09 | 10.13 / 11.88 / 8.02 | 10.04 / 11.76 / 7.96 |
| unconstrained realised, % a year | | 10.00 / 11.57 / 8.16 | 10.19 / 11.95 / 8.08 | 10.10 / 11.84 / 8.02 |
| members, median | | 1.003 / 1.037 / 0.965 | 0.987 / 0.978 / 0.986 | 0.962 / 0.955 / 0.962 |
| MRAD, the broad portfolios | | 0.167 / 0.165 / 0.195 | 0.094 / 0.068 / 0.159 | 0.095 / 0.068 / 0.161 |
| MRAD, the optimiser | | 0.238 / 0.259 / 0.145 | 0.135 / 0.150 / 0.125 | 0.126 / 0.134 / 0.137 |
| MRAD, the members (median) | | 0.094 / 0.092 / 0.082 | 0.085 / 0.076 / 0.084 | 0.089 / 0.080 / 0.085 |
| eigenportfolios 1-5, mean | | 1.030 / 1.065 / 0.986 | 1.011 / 0.985 / 1.031 | 1.011 / 0.985 / 1.031 |
| industries, mean abs(bias - 1) | | 0.046 / 0.094 / 0.110 | 0.055 / 0.065 / 0.088 | 0.058 / 0.065 / 0.093 |
| random long-only, median | | 0.971 / 1.124 / 0.798 | 0.914 / 0.983 / 0.838 | 0.909 / 0.979 / 0.833 |
| random long-short, median | | 1.004 / 1.001 / 1.003 | 0.997 / 0.974 / 1.015 | 0.966 / 0.948 / 0.984 |

**The members by capitalisation quintile, the largest first, over the whole test:**

| setting | quintiles |
|---|---|
| the model | 1.054, 1.011, 1.001, 0.968, 0.981 |
| F 21, S off | 1.028, 0.992, 0.988, 0.949, 0.963 |
| F off, S 42 | 1.022, 0.984, 0.973, 0.946, 0.953 |
| F 21, S 42 | 1.005, 0.969, 0.965, 0.929, 0.940 |

### What it says

- **The factor side follows the regime and brings each year's level toward one.** Within each
  half nearly everything improves: the broad portfolios' first-half MRAD 0.165 to 0.068, the
  optimiser's 0.259 to 0.150, the random long-only portfolios 1.124 to 0.983 and 0.798 to 0.838,
  the industries 0.094 and 0.110 to 0.065 and 0.088. The shortest half-life is the best of the
  three in both halves.
- **Over the whole test its level is too high.**
  - **The statistics fall:** the index 1.015 to 0.934, below its band (0.946 to 1.054), the
    largest 50 0.996 to 0.920, the random long-only 0.971 to 0.914.
  - **Why:** the factors as a whole were forecast too low -- their squared statistic averages
    1.115 over the test -- while the market's direction, which the index and the long-only
    portfolios load on, was right on average (eigenportfolio 1 at 1.027).
  - **And the multiplier is a mean of squares:** it rises fast in a shock and keeps the
    forecasts high for weeks after it (1.33 at the end of May 2025 at 21 days, the calm already
    back).
- **The calm half is a shape, not a level** (`eigen_by_half.txt`).
  - **The market direction alone is over-forecast:** in the second half the market-like
    eigenportfolio, rank 1, reads 0.780, as the index does (0.751). Every other rank reads 0.99
    to 1.37 under the model; their mean squared outcome is 1.36 against rank 1's 0.61.
  - **One multiplier cannot reach it:** it lowers every direction alike, so it brings rank 1
    only to 0.814 and the index to 0.785, while the small ranks get worse (ranks 18 to 20 from
    1.26-1.37 to 1.36-1.44).
  - **The first half is a level:** rank 1 at 1.226 and the rest near 1.1, every rank too low,
    and there the multiplier works.
- **The optimiser realises a little more risk under the factor side:**
  - **Whole test:** 10.13% and 10.19% a year against 9.92% and 10.00%.
  - **By half:** more in the first half (11.88% against 11.47%), a little less in the second
    (8.02% against 8.09%).
  - **Why:** in a shock the multiplier makes factor risk dearer than specific risk, and the
    portfolios it then builds realised more.
- **The forecasts move more:** the index's forecast changes 1.00% a day at the median under the
  factor side at 21 days, 0.67% at 42, 0.49% at 84, against the model's 0.27%; every member's a
  little more (1.48% against 1.34%).
- **The specific side is a level, not a regime.**
  - **No time variation:** its squared statistic is 1.088 in the first half and 1.090 in the
    second, and its multiplier sits near 1.04 on every half-life.
  - **Cap-weighted, it reads the largest members:** an effective number of 51, their specific
    risk forecast too low -- the top quintile's 1.054, midoffice11's and 12's open item. It
    raises every member's to fix it: the top quintile 1.054 to 1.022 at 42 days, the other four
    to 0.95-0.98, the members' median 1.003 to 0.976, the long-short random portfolios
    (specific-dominated) 1.004 to 0.971.
  - **On the criterion:** it helps through the optimiser's MRAD in the first half and hurts in
    the second; with the factor side at 21 days it moves the criterion by 0.004 one way on the
    first half and 0.005 the other way on the second.

---

## Item 3: the eigenvalue adjustment one-sided on the regime adjustment (VE15)

**Built:**
- **The option:** `eigen_one_sided` -- `EigenBias.scales`, `.adjusted` and `.basis` take
  `one_sided`, gamma = max(1, a (lambda - 1) + 1). It is the estimator's option and the run's
  `--eigen-one-sided`, off, for `--eigen-adjustment` and `--bias-eigen-scales` alike.
- **Under the model's own regime adjustment** each scale's factor side is levelled again by the
  history of its own forecasts' statistics: USE4's order, the eigen adjustment and then the
  regime adjustment on it, so the eigen adjustment is judged as a shape.
- **The specific side** is the model's for every scale.

**The run:**
- **The pass:** the factor side at 21 days and the specific side off as the model's own (my
  choice, below), `--bias-eigen-scales 1,2,3 --eigen-one-sided`; 15.4 minutes; 1.21 s a refit
  with the simulation (0.49 s of it) and 4 tests.
- **Its model test** equals V15's at that setting to the bit in every array, its regime history
  too (`equal_VE15_V15.txt`).
- **The chain:** each scale's eigenportfolio forecasts are the model's times the day's one-sided
  gamma times its own factor multiplier over the model's, to 2.4e-14 (`analyse_eigen.txt`).
- **The ranks raised:** ranks 21 to 25 on every date, ranks 1 to 5 on 47% of them; gamma at the
  smallest rank 1.065, 1.129 and 1.194 on average at a = 1, 2, 3.
- **Each scale's own factor multiplier is lower** -- the eigen adjustment raises the factor
  volatilities: 1.024, 1.010 and 0.997 on average against the model's 1.039.

**The minimum-variance portfolios' realised volatility first** (% a year):

| a | long-only whole / first / second | unconstrained whole / first / second | their statistics, whole |
|---|---|---|---|
| 0, the model | 10.13 / 11.88 / 8.02 | 10.19 / 11.95 / 8.08 | 0.990 / 1.033 |
| 1 | 10.17 / 11.97 / 7.99 | 10.26 / 12.07 / 8.07 | 0.989 / 1.032 |
| 2 | 10.23 / 12.09 / 7.97 | 10.33 / 12.19 / 8.07 | 0.990 / 1.034 |
| 3 | 10.30 / 12.22 / 7.96 | 10.40 / 12.31 / 8.08 | 0.992 / 1.037 |

**Then the rest:**

| a | eigen mean abs(bias - 1) | ranks 16-20 / 21-25 | the index | members, median | the criterion | the regime criterion |
|---|---|---|---|---|---|---|
| 0 | 0.131 | 1.223 / 1.195 | 0.934 | 0.987 | 0.076 | 0.105 |
| 1 | 0.110 | 1.168 / 1.140 | 0.945 | 0.985 | 0.066 | 0.105 |
| 2 | 0.092 | 1.120 / 1.093 | 0.956 | 0.983 | 0.057 | 0.106 |
| 3 | 0.076 | 1.077 / 1.050 | 0.966 | 0.981 | 0.049 | 0.107 |

**By half:**
- **The first half:** the eigenportfolios' mean abs(bias - 1) 0.067 to 0.042 at a = 1, ranks
  21-25 to 0.94 at a = 3.
- **The second half:** 0.222 to 0.180 at a = 3, ranks 16-20 still 1.21.

**The choices on the first half:**
- **Midoffice13's criterion:** a = 1 (0.090 against 0.101), the best on the second half a = 3.
- **The regime criterion:** a = 0.
- **The optimiser's realised volatility:** a = 0 (11.92% against 11.92%); on the second half
  a = 3 is lowest, by 0.03.

**What it says:**
- **The shape:** the one-sided adjustment does what it is for on the eigenportfolios, and does
  not lower the largest eigenvalues as midoffice13's two-sided one did.
- **The optimiser:** it still realises more risk at every a, in the whole test and most in the
  first half (+0.34 of a point at a = 3), a few hundredths less in the second; and the statistic
  it was to fix does not move.
- **Its rise in the index's statistic** (0.934 to 0.966) is the levelling's: a lower factor
  multiplier on the market direction, which the eigen adjustment leaves alone.
- **Judged as the brief asks**, the optimiser's realised volatility first, it fails: off.

---

## Which I would choose, and why

1. **The regime adjustment's factor side on a 21-day half-life.** It is the criterion's choice on
   the first half, alone or with any specific side, and the best on the second; it brings each
   year's forecasts toward one for every family of portfolio -- the purpose of a regime
   adjustment, and what the rolling windows read. Its costs, for Jeff to weigh:
   - **the whole test's level:** the index 0.934, below its band, because the factors as a whole
     were forecast too low while the market's direction was right;
   - **the optimiser:** about 0.2 of a point a year more realised;
   - **the noise:** the index's forecast moving four times as much from day to day.

   42 days is the steadier alternative (0.67% a day; the criterion 0.111 and 0.131 against 0.098
   and 0.123).
2. **Its specific side off.** The criterion's joint pick, 42 days, differs from off by 0.004 on
   the first half and reverses on the second. And the multiplier is a level -- the cap-weighted
   statistic reading the largest members' under-forecast -- which it fixes by over-forecasting
   the rest. That level belongs in the specific risk model, by size.
3. **The eigenvalue adjustment off,** one-sided as two-sided: it shapes the eigenportfolios and
   the optimiser realises more risk under it.
4. **Next, the calm half's shape:** the second half's over-forecast is the market direction's
   while every other direction is under-forecast -- a change in the correlations, which the
   504-day correlation half-life follows slowly. The candidate is a shorter correlation half-life
   on the regime-adjusted model, judged by the same test. Then the largest members' specific
   risk, in the specific risk model, before the specific side is tested again.

---

## Key decisions

1. **S&P Global's ruling held, not deleted:** a `held` field the reader keeps and the build lists
   and refuses to apply, so the evidence and the ruling stay in the file until it is understood.
2. **The gross-up as `retained` on the ruling,** the seeded count rounded to the share; Vylor's
   count is its unrounded product as before, so the earlier panels rebuild to the bit.
3. **The rulings files written by a script from the data** (`amend_rulings.py`), every number in
   the evidence the cache's; HON's count override marked `superseded` and kept, so
   `--spin-offs none` still builds `_listed`; the kept rulings' `ruled` fields marked.
4. **A new panel, `_revised`,** beside `_spinoffs` and `_listed`; `_spinoffs` rebuilds from
   d2b2864's rulings file (final checks).
5. **Ruling 6 read from the code and checked on the data,** by blanking the stitched prices and by
   scaling the highs and lows: none read across a ruled ex-date, so nothing applied.
6. **The regime adjustment's statistics from the bias test's own as-of refits,** not from a
   cheaper approximation: every forecast it reads is one the model made at that date's close, so
   the out-of-sample property holds by construction and the test checks it; a plain run refits as
   the test does.
7. **One until 21 days of history,** the history from the day after the test's first forecast --
   the first with a year of estimated periods -- and the exponential weights normalised over the
   days there are.
8. **The specific side over the cap-weighted index's own members,** the specific return on the
   forecast exposures -- each member's regression residual where the regression held it.
9. **One pass for every pairing,** either side off among them: 16 tests in under a second a
   refit; `--bias-regime-half-lives` and `--bias-eigen-scales` exclusive, so each pass tests one
   thing beside its model.
10. **The criterion fixed in the code before the run** (`REGIME_CRITERION`), on the MRADs the
    brief names first, with the statistics by half, the broad portfolios, the members and the
    optimiser's realised volatility reported beside it.
11. **Item 3 on the factor side at 21 days with the specific side off** -- my choice, not the
    criterion's joint pick -- for the reasons above; and each eigen scale levelled again by its
    own forecasts' history, so the eigen adjustment is judged as a shape on a level that is
    right.
12. **The as-of reading in the library** (`regime_history`, `add_refit`), so the out-of-sample
    test runs the code the run uses.

## Decisions left for Jeff

1. **The regime adjustment's factor side as the model's default:** on 21 days (my choice), on 42
   (steadier), or off.
2. **Its specific side:** off (my choice), or on 42 days, the criterion's joint pick.
3. **The eigenvalue adjustment one-sided:** off (my choice), or at a = 1, midoffice13's
   criterion's first-half choice.
4. **S&P Global's held ruling:** held until an outside price for SPGI's regular-way close of
   2026-06-30 settles it (my choice), or dropped from the file.
5. **The next piece:** the correlation half-life on the regime-adjusted model (my choice); the
   largest members' specific risk; or another.

---

## Tests

**The suite:** 2,372 passed, 1 skipped (`final_checks.txt`), with
`--ignore=tests/test_market_data.py`. Before this piece 2,362: 10 new, and every old one passes
unchanged -- the new parameters all default to what was there.

**New, 10:**

- `test_securities.py`, 2:
  - `test_a_seeded_count_is_grossed_up_by_the_share_the_parent_kept` -- FDXF's case: the parent's
    count times the ratio over 1 - retained, to the share, until the child's own count; the total
    returns and every other column unchanged;
  - `test_a_held_ruling_is_read_with_its_reason_and_never_applied` -- a held ruling read with why,
    refused by the build and the split check; the reader's retained rules; the repo's file: SPGI
    held, FDX retained 0.199, CTVA's as before.
- `test_regime.py`, 6:
  - `test_the_statistics_are_root_mean_squares_of_the_standardised_returns` -- B_F^2 and B_S^2 on
    known values, the weights normalised, the missing left out;
  - `test_the_multiplier_is_one_until_its_history_is_long_enough_then_a_weighted_root` -- one off
    and under 21 days; the weighted root; a short half-life follows the latest days;
  - `test_a_day_is_read_from_the_forecast_made_the_day_before` -- the factor volatilities at the
    day's scale; the specific returns on the forecast exposures, cap-weighted over the eligible;
    the history runs forward; the path is each day's multiplier;
  - `test_forecasts_half_the_truth_give_multipliers_of_two_and_right_ones_of_one` -- returns drawn
    at twice the forecast volatility give two on both sides, at the forecast's own one;
  - `test_the_regime_adjusted_forecast_scales_the_covariance_and_the_specific_risk` -- the
    covariance and eigenbasis by lambda_F^2, the specific by lambda_S, a portfolio's variance so
    scaled, at one and one the forecast itself;
  - `test_nothing_after_a_date_moves_the_multiplier_as_of_it` -- the brief's test: the as-of
    history through the date, and every multiplier as of it, the same to the bit with everything
    after it redrawn.
- `test_eigen.py`, 1: `test_the_one_sided_adjustment_raises_the_small_eigenvalues_and_lowers_none`
  -- gamma at least one, the largest left, the small raised as two-sided; two-sided unchanged.
- `test_bias.py`, 1: `test_a_forecast_reads_another_scale_one_sided_and_a_one_sided_model_its_own`
  -- from_model one-sided; at a = 0 nothing; a one-sided model's covariance, basis and note; the
  option refused without the adjustment.

**Measured, not unit-tested:** the run's regime and eigen paths (three smoke runs; V15's model
test D15's to the bit; VE15's V15's; the multipliers and the chain recomputed from the stored
arrays to 5e-14); the build's (`_revised` against `_spinoffs` and `_listed`; the three panels
rebuilt to the bit in the final checks).

**Hygiene:** every changed file parses, is ASCII and keeps its line endings (`final_checks.txt`).
**ruff:** clean over every changed Python file.

---

## Failures and debugging trail

- **A heredoc mangled backslashes twice:**
  - a block of the amendment script, whose `\n` became a newline inside a string;
  - an adaptation of midoffice14's comparison script, whose replacement then failed.

  The state file's working note; the Write and Edit tools did both. The config files were
  restored from HEAD (`git checkout --` on files only this piece had changed) and the script
  rerun.
- **The amendment's evidence text needed three passes:**
  - a gap in points printed as a fraction (-0.03);
  - HON's note with doubled parentheses;
  - "the record date is not its ex-date, as Lennar's was", which read both ways.

  Each time the two files were restored from HEAD and the script rerun, so the files are one
  run's output.
- **midoffice13's smoke harness keeps the first N forecast dates.** The regime adjustment's last
  step needs the forecast as of the day before the last date, so it would stop there. Written:
  `smoke_last.py`, the last N rows.
- **The equality script** first read the eigen variants' horizon arrays as the model's, then
  failed on a string array under `equal_nan`; both fixed before its result was read.
- **ruff:** three long lines, four f-strings without a placeholder, an unsorted import in a test;
  fixed.
- **The report's labels:** the regime report's first column cut its part labels short, and the
  eigen report's longest rows overran theirs; widened after the smoke runs.
- **The runs:** none failed. D15 11.4 minutes, V15 13.6, VE15 15.4 (`bias/runner_log.txt`), each
  started with over 4.5 GB available.

## Problems found, not fixed

1. **S&P Global's Mobility** is held, not understood: the record makes 07-01 its ex-date and the
   raw prices show no fall of its value. An outside price for SPGI's regular-way close of
   2026-06-30 would settle whether Polygon's series is the regular-way line's.
2. **The calm half's over-forecast is the market direction's,** a shape: no level adjustment
   reaches it.
3. **Under the regime adjustment the whole test's level is too high** (the index 0.934, below its
   band), the factor statistic equal-weighted over the factors while the broad portfolios load on
   the market's direction.
4. **The regime adjustment's day-to-day noise:** the index's forecast moves four times as much at
   21 days.
5. **The optimiser realises a little more risk** under the regime adjustment's factor side (+0.2
   of a point a year) and under the eigen adjustment (+0.05 to +0.17 more at a = 1 to 3).
6. **The largest members' specific risk is forecast too low,** steadily (the top quintile 1.054;
   the cap-weighted specific statistic 1.09): a specific risk model question.
7. **FedEx Freight and Vylor still lack Momentum,** so they enter no regression and have no
   forecast; their counts reach the model only through the cap-weighted means.
8. **As before:** six ruff findings in untouched files at HEAD, and `tests/test_market_data.py`'s
   `../core_repo/src`.

## Scratch

`claude/scratch/midoffice15/` (gitignored):

- **The rulings:**
  - `spgi_gap.py` and `.txt`, `spgi_record.txt`;
  - `amend_rulings.py` and `.txt` -- the writer of the two rulings files;
  - `ruling6_series.py` and `.txt`;
  - `tests_rulings_block.py`, the new securities tests, appended.
- **The panels:** `panels/`, the revised panel built into scratch, its build outputs (into scratch
  and into `data/`); `panel_changed.txt`, `panel_vs_listed.txt`.
- **The model:** `model/`, M15, its report and pickle; `model_compare.txt`.
- **The runs:**
  - `runner.py`;
  - `bias/`: D15, V15 and VE15, each a report, progress and `.npz`; the runner's log;
  - `smoke/` and `smoke_last.py`, the three smoke runs.
- **The analyses:**
  - `compare_bias.py` and `.txt` -- D15 against D14;
  - `analyse_regime.py` and `.txt`; `eigen_by_half.py` and `.txt`;
  - `analyse_eigen.py` and `.txt`;
  - `equal_setting.py`, `equal_VE15_V15.txt`, `equal_V15_D15.txt`.
- **The checks:** `final_checks.sh` and `.txt`, `final/`; the retro's `append_appendix.py`.

The midoffice13 and 14 harnesses were used where they are: `capture.py`, `compare.py`,
`equal_npz.py`, `hygiene.py`, `common.py`, `compare_panels.py`. The key outputs are copied into
the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`, and names every changed and new path of this piece, the state file included.

## Open items for Chat

- **The five decisions above,** the regime adjustment's factor side first.
- **The next piece,** as I would order it: a shorter correlation half-life on the regime-adjusted
  model, for the calm half's shape; then the largest members' specific risk.
- **The bias statistics are standing,** the regime report among them under
  `--bias-regime-half-lives`; every candidate is one run against D15, through the runner.

---

## Appendix: outputs, verbatim

Each output as its script wrote it, from `claude/scratch/midoffice15/`.

### spgi_gap.txt -- ruling 1: SPGI's raw bars around its record date and ruled ex-date, against SPY and its peers; MBGL's first bars; the cache's records; the panels' total return

```text
SPGI's raw and adjusted bars, 2026-06-01 to 2026-07-10, against SPY and its peers
  gap: the open over the prior close - 1, raw; ret: close over prior close - 1
  day        SPGI open    close     gap     ret  adj ret  SPY gap  SPY ret peers gap peers ret  SPGI less SPY gap/ret
  2026-06-02    427.08   417.46  -0.35%  -2.59%   -2.59%   -0.20%   +0.14%    -0.83%    -2.54%  -0.15% -2.73%
  2026-06-03    414.30   412.29  -0.76%  -1.24%   -1.24%   -0.19%   -0.70%    -0.32%    -1.20%  -0.57% -0.54%
  2026-06-04    418.00   420.12  +1.38%  +1.90%   +1.90%   -0.28%   +0.38%    +1.94%    +1.11%  +1.67% +1.52%  <--
  2026-06-05    422.88   424.44  +0.66%  +1.03%   +1.03%   -0.63%   -2.58%    +0.52%    -0.08%  +1.29% +3.61%  <--
  2026-06-08    421.57   417.09  -0.68%  -1.73%   -1.73%   +0.79%   +0.23%    -0.41%    -1.88%  -1.46% -1.96%
  2026-06-09    415.13   424.82  -0.47%  +1.85%   +1.85%   +0.60%   -0.29%    -0.20%    +1.12%  -1.07% +2.15%
  2026-06-10    423.66   426.38  -0.27%  +0.37%   +0.37%   -0.50%   -1.58%    -0.28%    +0.16%  +0.22% +1.94%
  2026-06-11    424.72   413.34  -0.39%  -3.06%   -3.06%   +0.46%   +1.70%    +0.06%    -1.47%  -0.85% -4.76%  <--
  2026-06-12    415.23   418.91  +0.46%  +1.35%   +1.35%   +0.40%   +0.54%    +0.61%    +1.64%  +0.06% +0.81%
  2026-06-15    418.26   424.07  -0.16%  +1.23%   +1.23%   +1.36%   +1.76%    -0.09%    +0.54%  -1.52% -0.53%  <--
  2026-06-16    429.95   433.25  +1.39%  +2.16%   +2.16%   -0.04%   -0.60%    +0.82%    +0.11%  +1.42% +2.76%
  2026-06-17    428.47   418.02  -1.10%  -3.52%   -3.52%   +0.13%   -1.25%    -0.95%    -3.68%  -1.23% -2.27%
  2026-06-18    418.00   410.92  -0.00%  -1.70%   -1.70%   +0.92%   +0.78%    +0.44%    -1.85%  -0.92% -2.48%
  2026-06-22    406.85   407.39  -0.99%  -0.86%   -0.86%   +0.13%   -0.31%    -0.19%    -0.61%  -1.12% -0.54%
  2026-06-23    412.12   400.16  +1.16%  -1.77%   -1.77%   -1.42%   -1.45%    +1.29%    -0.18%  +2.58% -0.32%  <--
  2026-06-24    402.38   402.35  +0.55%  +0.55%   +0.55%   +0.22%   -0.05%    +0.54%    -0.92%  +0.34% +0.59%
  2026-06-25    402.53   395.14  +0.04%  -1.79%   -1.79%   +0.77%   +0.14%    -0.02%    -3.88%  -0.73% -1.94%
  2026-06-26    395.15   408.16  +0.00%  +3.30%   +3.30%   -0.73%   -0.72%    +0.29%    +1.53%  +0.73% +4.02%  <--
  2026-06-29    412.48   408.56  +1.06%  +0.10%   +0.10%   +1.03%   +1.65%    +1.14%    -0.10%  +0.02% -1.55%
  2026-06-30    406.58   407.26  -0.48%  -0.32%   -0.32%   +0.04%   +0.78%    -0.42%    +0.26%  -0.52% -1.10%
  2026-07-01    399.58   414.97  -1.89%  +1.89%   +1.89%   -0.24%   -0.14%    +0.74%    +4.27%  -1.65% +2.03%  <--
  2026-07-02    423.26   439.89  +2.00%  +6.01%   +6.01%   +0.22%   -0.13%    +1.51%    +3.02%  +1.78% +6.14%  <--
  2026-07-06    439.57   447.22  -0.07%  +1.67%   +1.67%   +0.53%   +0.87%    -0.08%    +1.14%  -0.60% +0.79%
  2026-07-07    447.30   443.46  +0.02%  -0.84%   -0.84%   -0.14%   -0.48%    +0.50%    +1.73%  +0.16% -0.37%
  2026-07-08    441.99   430.79  -0.33%  -2.86%   -2.86%   -0.61%   -0.31%    -0.70%    -1.50%  +0.28% -2.55%
  2026-07-09    426.65   432.96  -0.96%  +0.50%   +0.50%   +0.26%   +0.85%    -1.08%    -0.76%  -1.22% -0.34%
  2026-07-10    438.34   430.50  +1.24%  -0.57%   -0.57%   +0.05%   +0.43%    +1.02%    +0.23%  +1.20% -1.00%

SPGI's intraday range on each day (raw): low and high against the prior close
  2026-06-02 low   412.87 (-3.66%)  high   427.08 (-0.35%)  volume    1,936,381
  2026-06-03 low   407.31 (-2.43%)  high   416.31 (-0.28%)  volume    1,619,330
  2026-06-04 low   418.00 (+1.38%)  high   425.90 (+3.30%)  volume    2,133,423
  2026-06-05 low   420.55 (+0.10%)  high   427.24 (+1.69%)  volume    2,259,864
  2026-06-08 low   416.12 (-1.96%)  high   423.73 (-0.17%)  volume    2,120,229
  2026-06-09 low   412.80 (-1.03%)  high   426.67 (+2.30%)  volume    1,888,698
  2026-06-10 low   421.05 (-0.89%)  high   430.27 (+1.28%)  volume    1,829,248
  2026-06-11 low   411.51 (-3.49%)  high   426.00 (-0.09%)  volume    2,126,582
  2026-06-12 low   409.33 (-0.97%)  high   419.98 (+1.61%)  volume    1,762,088
  2026-06-15 low   418.00 (-0.22%)  high   429.39 (+2.50%)  volume    1,650,268
  2026-06-16 low   428.01 (+0.93%)  high   436.37 (+2.90%)  volume    1,967,075
  2026-06-17 low   417.09 (-3.73%)  high   434.08 (+0.19%)  volume    2,082,559
  2026-06-18 low   407.98 (-2.40%)  high   418.00 (-0.00%)  volume    2,860,384
  2026-06-22 low   403.16 (-1.89%)  high   411.70 (+0.19%)  volume    2,925,922
  2026-06-23 low   399.88 (-1.84%)  high   415.00 (+1.87%)  volume    1,845,042
  2026-06-24 low   395.67 (-1.12%)  high   404.97 (+1.20%)  volume    5,001,006
  2026-06-25 low   395.00 (-1.83%)  high   413.00 (+2.65%)  volume    2,962,211
  2026-06-26 low   393.03 (-0.53%)  high   416.04 (+5.29%)  volume    4,606,879
  2026-06-29 low   405.17 (-0.73%)  high   414.19 (+1.48%)  volume    2,577,395
  2026-06-30 low   402.59 (-1.46%)  high   412.51 (+0.97%)  volume    2,408,847
  2026-07-01 low   397.00 (-2.52%)  high   419.76 (+3.07%)  volume    3,956,773
  2026-07-02 low   422.59 (+1.84%)  high   440.01 (+6.03%)  volume    3,749,612
  2026-07-06 low   427.75 (-2.76%)  high   447.70 (+1.78%)  volume    2,573,392
  2026-07-07 low   440.57 (-1.49%)  high   450.31 (+0.69%)  volume    3,022,003
  2026-07-08 low   430.57 (-2.91%)  high   442.91 (-0.12%)  volume    2,553,880
  2026-07-09 low   422.70 (-1.88%)  high   433.34 (+0.59%)  volume    1,589,827
  2026-07-10 low   427.06 (-1.36%)  high   440.61 (+1.77%)  volume    1,320,967

MBGL's raw bars, its first ten
  2026-07-01 open   20.30 close   21.19 volume   97,879,134
  2026-07-02 open   20.95 close   19.80 volume   26,968,147
  2026-07-06 open   18.85 close   19.26 volume   35,850,605
  2026-07-07 open   19.25 close   21.20 volume   21,719,177
  2026-07-08 open   20.72 close   21.50 volume   17,409,516
  2026-07-09 open   21.19 close   22.26 volume    9,072,722
  2026-07-10 open   21.53 close   20.80 volume    5,922,016
  2026-07-13 open   20.05 close   20.26 volume    5,462,231
  2026-07-14 open   20.38 close   20.57 volume    6,641,887
  2026-07-15 open   20.36 close   21.00 volume    6,074,216

The cache's splits for SPGI:
   {"execution_date": "2005-05-18", "id": "Ebfdf158b2f95422d558ef9fd0e10a1ba515d678cdbca9c8ec7c4cb25527f89de", "split_from": 1, "split_to": 2, "ticker": "SPGI"}
The cache's dividends for SPGI from 2026-01-01:
   {"cash_amount": 0.97, "currency": "USD", "declaration_date": "2026-09-29", "dividend_type": "CD", "ex_dividend_date": "2026-11-25", "frequency": 4, "id": "E71a2ca3fe2f7022026a528916cd90dc505891ddf7033ad43f9c99c08174ff19c", "pay_date": "2026-12-10", "record_date": "2026-11-25", "ticker": "SPGI"}
   {"cash_amount": 0.97, "currency": "USD", "declaration_date": "2026-06-22", "dividend_type": "CD", "ex_dividend_date": "2026-08-26", "frequency": 4, "id": "Eafa571ab6ef6da5b025530e4e1b32cd3f85eee5ccf539bf992471e0091754fe6", "pay_date": "2026-09-10", "record_date": "2026-08-26", "ticker": "SPGI"}
   {"cash_amount": 0.97, "currency": "USD", "declaration_date": "2026-05-19", "dividend_type": "CD", "ex_dividend_date": "2026-05-29", "frequency": 4, "id": "E5e470854e5067512c12748f6ba33c4d3810a3cb3417d5fdcca524b4b6acec1f8", "pay_date": "2026-06-10", "record_date": "2026-05-29", "ticker": "SPGI"}
   {"cash_amount": 0.97, "currency": "USD", "declaration_date": "2026-01-14", "dividend_type": "CD", "ex_dividend_date": "2026-02-25", "frequency": 4, "id": "E7f4fa5c0e5e8c9382a9188ad2321472f69f2457b9b86a314bf3fdf5a7dba326e", "pay_date": "2026-03-11", "record_date": "2026-02-25", "ticker": "SPGI"}
The cache's ticker events for SPGI:
   {"name": "S&P Global Inc.", "composite_figi": "BBG000BP1Q11", "cik": "0000064040", "events": [{"ticker_change": {"ticker": "SPGI"}, "type": "ticker_change", "date": "2016-04-28"}]}

ref-MBGL-2026-07-01.json: name 'Mobility Global Inc.', list_date None, composite FIGI 'BBG022G54330', share-class shares None, weighted None, market 'stocks'

The panel's SPGI total return (_listed), column 487, key BBG000BP1Q11:
  2026-06-12  +1.3476%  member True  count 303,462,990
  2026-06-15  +1.2318%  member True  count 303,462,990
  2026-06-16  +2.1647%  member True  count 303,462,990
  2026-06-30  -0.3182%  member True  count 303,462,990
  2026-07-01  +1.8931%  member True  count 303,462,990
  2026-07-02  +6.0053%  member True  count 303,462,990

The panel's SPGI total return (_spinoffs), column 487, key BBG000BP1Q11:
  2026-06-12  +1.3476%  member True  count 303,462,990
  2026-06-15  +1.2318%  member True  count 303,462,990
  2026-06-16  +2.1647%  member True  count 303,462,990
  2026-06-30  -0.3182%  member True  count 303,462,990
  2026-07-01  +7.0962%  member True  count 303,462,990
  2026-07-02  +6.0053%  member True  count 303,462,990

Each peer's raw return on the days that matter (open gap / close-to-close)
  2026-06-15: SPGI -0.16%/+1.23%  MCO +0.65%/+1.39%  MSCI +0.18%/+2.01%  ICE -0.37%/-0.31%  CME -0.75%/-1.28%  NDAQ +0.72%/+2.32%  FDS -1.00%/-2.20%  SPY +1.36%/+1.76%
  2026-06-16: SPGI +1.39%/+2.16%  MCO +1.38%/+3.16%  MSCI +1.25%/-0.49%  ICE +0.68%/+0.84%  CME +0.83%/-1.68%  NDAQ +0.72%/-1.92%  FDS +0.81%/+0.72%  SPY -0.04%/-0.60%
  2026-06-26: SPGI +0.00%/+3.30%  MCO -0.23%/+2.54%  MSCI +0.78%/+1.89%  ICE +1.48%/-0.51%  CME +0.15%/-1.78%  NDAQ +0.44%/+1.17%  FDS -0.40%/+10.97%  SPY -0.73%/-0.72%
  2026-06-29: SPGI +1.06%/+0.10%  MCO +0.55%/+0.60%  MSCI +1.27%/+0.57%  ICE +1.73%/-0.77%  CME +1.39%/-1.10%  NDAQ +1.01%/-2.18%  FDS +0.82%/+0.81%  SPY +1.03%/+1.65%
  2026-06-30: SPGI -0.48%/-0.32%  MCO -0.54%/+0.04%  MSCI -0.88%/+0.37%  ICE -0.50%/+0.16%  CME +0.79%/+1.03%  NDAQ +0.19%/+2.57%  FDS -0.35%/-1.52%  SPY +0.04%/+0.78%
  2026-07-01: SPGI -1.89%/+1.89%  MCO +0.57%/+3.41%  MSCI +0.71%/+3.93%  ICE +0.77%/+2.94%  CME +1.31%/+4.61%  NDAQ +1.26%/+4.96%  FDS -3.51%/+6.72%  SPY -0.24%/-0.14%
  2026-07-02: SPGI +2.00%/+6.01%  MCO +1.26%/+4.72%  MSCI +1.73%/+3.62%  ICE +2.19%/+4.94%  CME -0.14%/+2.42%  NDAQ +1.28%/+2.33%  FDS +1.81%/+1.85%  SPY +0.22%/-0.13%

The cumulative raw return from the close of 2026-06-12 (the day before the record date): SPGI, the peers' median, SPGI less the peers
  2026-06-15  SPGI  +1.23%  peers  +0.54%  less +0.69%
  2026-06-16  SPGI  +3.42%  peers  +0.44%  less +2.98%
  2026-06-17  SPGI  -0.21%  peers  -4.78%  less +4.57%
  2026-06-18  SPGI  -1.91%  peers  -6.15%  less +4.25%
  2026-06-22  SPGI  -2.75%  peers  -6.85%  less +4.10%
  2026-06-23  SPGI  -4.48%  peers  -6.33%  less +1.85%
  2026-06-24  SPGI  -3.95%  peers  -7.88%  less +3.93%
  2026-06-25  SPGI  -5.67%  peers -12.07%  less +6.40%
  2026-06-26  SPGI  -2.57%  peers  -9.55%  less +6.98%
  2026-06-29  SPGI  -2.47%  peers  -9.70%  less +7.23%
  2026-06-30  SPGI  -2.78%  peers  -8.97%  less +6.19%
  2026-07-01  SPGI  -0.94%  peers  -4.94%  less +4.00%
  2026-07-02  SPGI  +5.01%  peers  -2.09%  less +7.10%
  2026-07-06  SPGI  +6.76%  peers  -0.75%  less +7.51%
  2026-07-07  SPGI  +5.86%  peers  -0.68%  less +6.54%
  2026-07-08  SPGI  +2.84%  peers  -0.77%  less +3.60%
  2026-07-09  SPGI  +3.35%  peers  -0.67%  less +4.02%
  2026-07-10  SPGI  +2.77%  peers  -0.04%  less +2.81%

Over 2026-07-01 and 07-02 together, raw: SPGI +8.01%, MCO +8.30%, MSCI +7.69%, ICE +8.03%, CME +7.14%, NDAQ +7.41%, FDS +8.70%, SPY -0.27%
  SPGI with Mobility at its first close: +13.53%; the peers' median +7.86%
  Mobility's value a SPGI share against SPGI's close of 06-30: 4.98% at its open, 5.20% at its close
```

### spgi_record.txt -- ruling 1: the public record's check, the search and what it quoted

```text
Ruling 1's public-record check, 2026-11-12 (one web search, standard mode):

  query: S&P Global SPGI ex-distribution date Mobility Global MBGL spin-off July 1 2026 shares

The results it listed:
  - S&P Global Inc. - Form 8-K - FY2026:
    https://www.sec.gov/Archives/edgar/data/0000064040/000110465926064848/tm2614786d3_ex99-1.htm
  - streetinsider.com: S&P Global completes spin-off of Mobility Global Inc.:
    https://www.streetinsider.com/Spinoffs/S%26P+Global+completes+spin-off+of+Mobility+Global+Inc./26716839.html
  - fool.com, 2026-06-01: S&P Global is spinning off its Mobility business
  - stockspinoffs.com: S&P Global Will Spin Off Mobility Global On July 1 -- With $2 Billion Of Debt
  - rallies.ai: S&P Global Approves One-for-One Mobility Global Spin-Off Effective July 1
  - S&P Global's release, 2026-05-21:
    https://press.spglobal.com/2026-05-21-S-P-Global-Announces-Approval-of-Separation-of-Mobility-Global

What the search's summary quoted from them:
  - "S&P Global shareholders will receive one share of Mobility Global common stock for every
    share of S&P Global common stock held at the close of business on the record date of June
    15, 2026."
  - "The distribution of Mobility Global shares is expected to be effective at 12:01 a.m. New
    York City time on July 1, 2026."
  - "Beginning on or about June 26, 2026, and through June 30, 2026, it is expected that there
    will be two ways to trade S&P Global common stock -- either with or without the distribution
    of Mobility Global common stock." Ex-distribution trades in that period under "SPGI WI";
    Mobility when-issued as "MBGL WI".
  - "S&P Global Inc. (NYSE: SPGI) completed the separation of its Mobility division into an
    independent public company, Mobility Global Inc., on July 1, 2026. Mobility Global common
    stock began trading on the New York Stock Exchange under the ticker symbol "MBGL.""
  - No result uses the words "ex-date"; 07-01 as the first ex-distribution day is the reading of
    the trading schedule (regular-way SPGI carries the entitlement through 06-30). No result
    gives SPGI's or MBGL's prices after 07-01.

No page was opened; the search's summary is the evidence, as quoted. With it, the record makes
07-01 the ruled ex-date and not the record date (06-15), which is no ex-date here: unlike
Lennar's, the distribution traded with due bills through the distribution date.
```

### amend_rulings.txt -- rulings 1 to 4 written into the two rulings files from the data

```text
Ruling 1, SPGI's ruling held:
  The record: S&P Global's release of 2026-05-21 -- two ways to trade SPGI from about 06-26 through 06-30, regular-way with the entitlement and SPGI WI without it -- so 07-01 is the first day SPGI trades without it, the ruled ex-date; unlike Lennar's, the record date is not the ex-date. The data show no gap of Mobility's size on either day. On the record date SPGI opened -0.16% and closed +1.23% against its peers' median -0.09% and +0.54% (MCO, MSCI, ICE, CME, NDAQ, FDS), and the day after +1.39% and +2.16% against +0.82% and +0.11%. On 07-01, Mobility worth 4.98% of SPGI's prior close at its first open and 5.20% at its first close, SPGI opened -1.89% against its peers' +0.74% -- -2.63 points, about half the distribution, the largest such gap from 06-02 to 07-10 (the next -0.80) -- and closed +1.89% against +4.27%; over 07-01 and 07-02 together its raw return, +8.01%, is its peers' (+7.14% to +8.70%, median +7.86%) without the distribution, and +13.53% with it. Neither the record date nor the ex-date shows the distribution in SPGI's raw prices, so the ruling is held until it is understood -- by an outside price for SPGI's regular-way close of 2026-06-30, say; the panel keeps Polygon's +1.89% on 07-01 meanwhile.

Ruling 3, FDXF's count: 238,606,756 x 0.5 = 119,303,378 distributed; / 0.801 = 148,943,043.695, to the share 148,943,044; against its weighted counts: 149,518,133 on 2026-08-26, 1.0039 times it; 149,521,266 on 2026-11-03, 1.0039 times it

Rulings 1, 2 and 4, the rulings kept and marked so: LEN 2025-01-21, J 2025-05-16, BDX 2026-02-10, APTV 2026-04-01, HON 2026-06-29

Ruling 2, HON's count override marked: By midoffice14's HON spin-off ruling, which applies Honeywell's 1-for-2 reverse split of 2026-06-29, made one minute after its Aerospace distribution: the halved records were Honeywell's own post-split count, not Honeywell Aerospace's, and this ruling, which read them as contamination, was wrong. The reverse split stands and this ruling is superseded: midoffice15's ruling 2 (Jeff, 2026-11-12). Kept as the record: the build reads it only under --spin-offs none, so midoffice10's panel rebuilds.

written: config\sp500_spin_off_overrides.json, config\sp500_share_count_overrides.json
```

### ruling6_series.txt -- ruling 6: which series the descriptors read, and whether any reads the stitched prices, highs or lows across a ruled ex-date

```text
1. The run's load at the defaults (sp500_securities_2022-01-03_2026-11-03_revised.npz, --returns total):
   SecurityPanel.prices              is the panel's total_return_prices  True
   SecurityPanel.unadjusted_prices   is the panel's unadjusted_prices    True
   SecurityPanel.highs               is the panel's highs                True
   SecurityPanel.lows                is the panel's lows                 True
   SecurityPanel.volumes             is the panel's volumes              True
   SecurityPanel.shares_outstanding  is the panel's shares_outstanding   True
   the market series is market_total_returns: True
   SecurityPanel.prices is the stitched prices: False

2. The panel's highs and lows on each applied ruling's ex-date, against the cache's raw and adjusted bars:
   APTV   2026-04-01  high     62.34 low     59.66  raw bar's     62.34     59.66  adjusted bar's     62.34     59.66  stitched close     62.09, raw     62.09
   BDX    2022-04-01  high    263.70 low    258.14  raw bar's    263.70    258.14  adjusted bar's    263.70    258.14  stitched close    262.50, raw    262.50
   BDX    2026-02-10  high    172.57 low    164.41  raw bar's    172.57    164.41  adjusted bar's    172.57    164.41  stitched close    171.68, raw    171.68
   BWA    2023-07-05  high     44.84 low     43.12  raw bar's     44.84     43.12  adjusted bar's     44.84     43.12  stitched close     43.18, raw     43.18
   CMCSA  2026-01-05  high     28.93 low     27.70  raw bar's     28.93     27.70  adjusted bar's     28.93     27.70  stitched close     28.13, raw     28.13
   CTVA   2026-10-01  high     14.44 low     11.84  raw bar's     14.44     11.84  adjusted bar's     14.44     11.84  stitched close     12.57, raw     12.57
   DHR    2023-10-02  high    221.66 low    208.96  raw bar's    221.66    208.96  adjusted bar's    221.66    208.96  stitched close    213.74, raw    213.74
   EXC    2022-02-02  high     43.09 low     40.80  raw bar's     43.09     40.80  adjusted bar's     43.09     40.80  stitched close     42.86, raw     42.86
   FBIN   2022-12-15  high     57.69 low     49.51  raw bar's     57.69     49.51  adjusted bar's     57.69     49.51  stitched close     57.62, raw     57.62
   FDX    2026-06-01  high    341.14 low    321.81  raw bar's    341.14    321.81  adjusted bar's    341.14    321.81  stitched close    338.49, raw    338.49
   HON    2026-06-29  high    252.00 low    227.16  raw bar's    252.00    227.16  adjusted bar's    252.00    227.16  stitched close    227.80, raw    227.80
   J      2024-09-30  high    132.23 low    128.71  raw bar's    132.23    128.71  adjusted bar's    132.23    128.71  stitched close    130.90, raw    130.90
   J      2025-05-16  high    129.70 low    127.78  raw bar's    129.70    127.78  adjusted bar's    129.70    127.78  stitched close    129.17, raw    129.17
   K      2023-10-02  high     55.82 low     51.83  raw bar's     55.82     51.83  adjusted bar's     55.82     51.83  stitched close     52.50, raw     52.50
   LEN    2025-01-21  high    139.19 low    132.49  raw bar's    139.19    132.49  adjusted bar's    139.19    132.49  stitched close    133.14, raw    133.14
   T      2022-04-11  high     19.77 low     18.85  raw bar's     19.77     18.85  adjusted bar's     19.77     18.85  stitched close     19.63, raw     19.63
   WDC    2025-02-24  high     52.67 low     48.23  raw bar's     52.67     48.23  adjusted bar's     52.67     48.23  stitched close     49.02, raw     49.02
   ZBH    2022-03-01  high    126.46 low    121.08  raw bar's    126.46    121.08  adjusted bar's    126.46    121.08  stitched close    124.14, raw    124.14
   the raw close between the day's low and high: 684,087 of 684,087 cells; outside, the worst: none

3. Every descriptor the defaults compute (LNCAP, LCAPCB, LPRI, DASTD, HILO, CMRA, BTSG, HALPHA, SERDP, RSTR, STOM, STOQ, STOA, VLVR), from the panel as built and with the stitched prices nan everywhere -- they are not in the SecurityPanel the run builds at --returns total, so nothing can read them:
   equal to the bit: True

4. The highs and lows scaled before every applied ruling's ex-date by the ruling's price factor, as Polygon's fractional records are applied: each descriptor's largest change
   LNCAP   equal to the bit
   LCAPCB  equal to the bit
   LPRI    equal to the bit
   DASTD   equal to the bit
   HILO    2.2e-16
   CMRA    equal to the bit
   BTSG    equal to the bit
   HALPHA  equal to the bit
   SERDP   equal to the bit
   RSTR    equal to the bit
   STOM    equal to the bit
   STOQ    equal to the bit
   STOA    equal to the bit
   VLVR    equal to the bit
```

### panel_changed.txt -- the revised panel, `_revised`, against midoffice14's `_spinoffs`: every changed member-day

```text
data/external/polygon/sp500_securities_2022-01-03_2026-11-03_spinoffs.npz
  against claude/scratch/midoffice15/panels/sp500_securities_2022-01-03_2026-11-03_revised.npz
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
  total_return_prices      DIFFERS: 88 cells, 88 member-days
  unadjusted_prices        equal to the bit (1214, 592)
  volumes                  equal to the bit (1214, 592)

total_return_prices, by daily return: 1 security-days changed beyond 1e-12, 1 of them member-days; 36 more differ in the last bits
  2026-07-01 SPGI   member: +7.0962% -> +1.8931% (-5.20 points)

prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

shares_outstanding: 109 security-days changed, 109 member-days
  FDXF   2026-06-01..2026-11-03 (109 days, 109 member-days): 119,303,378 -> 148,943,044, ... 119,303,378 -> 148,943,044
```

### panel_vs_listed.txt -- `_revised` against midoffice10's `_listed`: the 18 rulings applied and the counts

```text
total_return_prices, by daily return: 18 security-days changed beyond 1e-12, 18 of them member-days; 4420 more differ in the last bits
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
  2026-10-01 CTVA   member: -83.8120% -> +4.0953% (+87.91 points)

prices, by daily return: 0 security-days changed beyond 1e-12, 0 of them member-days

shares_outstanding: 223 security-days changed, 223 member-days
  FDXF   2026-06-01..2026-11-03 (109 days, 109 member-days): nan -> 148,943,044, ... nan -> 148,943,044
  HON    2026-06-29..2026-11-03 (90 days, 90 member-days): 635,675,701 -> 317,837,850, ... 635,675,701 -> 317,358,518
  VYLR   2026-10-01..2026-11-03 (24 days, 24 member-days): nan -> 668,820,000, ... nan -> 668,820,000
```

### panels/build_revised.txt -- the build's report: its spin-off section and the weighted check's carried records

```text
Spin-offs: distributions of shares the cache records wrongly or not at all, config/sp500_spin_off_overrides.json: 19 rulings, 1 held, 18 applied -- on each ex-date the parent's total return (w (C_t + D_t) + k V) / C_s - 1, the child at its close; the stitched prices as Polygon has them
  SPGI   2026-07-01 -> MBGL  held, not applied: The record: S&P Global's release of 2026-05-21 -- two ways to trade SPGI from about 06-26 through 06-30, regular-way with the entitlement and SPGI WI without it -- so 07-01 is the first day SPGI trades without it, the ruled ex-date; unlike Lennar's, the record date is not the ex-date. The data show no gap of Mobility's size on either day. On the record date SPGI opened -0.16% and closed +1.23% against its peers' median -0.09% and +0.54% (MCO, MSCI, ICE, CME, NDAQ, FDS), and the day after +1.39% and +2.16% against +0.82% and +0.11%. On 07-01, Mobility worth 4.98% of SPGI's prior close at its first open and 5.20% at its first close, SPGI opened -1.89% against its peers' +0.74% -- -2.63 points, about half the distribution, the largest such gap from 06-02 to 07-10 (the next -0.80) -- and closed +1.89% against +4.27%; over 07-01 and 07-02 together its raw return, +8.01%, is its peers' (+7.14% to +8.70%, median +7.86%) without the distribution, and +13.53% with it. Neither the record date nor the ex-date shows the distribution in SPGI's raw prices, so the ruling is held until it is understood -- by an outside price for SPGI's regular-way close of 2026-06-30, say; the panel keeps Polygon's +1.89% on 07-01 meanwhile.
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
  T      2022-04-11 -> WBD   0.241917 a share: 24.14 on 2022-04-08 to 19.63; WBD 24.78 on 2022-04-11 (Warner Bros. Discovery, Inc. Series A Common Stock, BBG011386VF4): total return +6.1504%, the adjusted closes' -18.6827%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  WDC    2025-02-24 -> SNDK  0.333333 a share: 68.70 on 2025-02-21 to 49.02; SNDK 48.60 on 2025-02-24 (Sandisk Corporation Common Stock, BBG01R388JG1): total return -5.0724%, the adjusted closes' -28.6515%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  ZBH    2022-03-01 -> ZIMV  0.1 a share: 127.19 on 2022-02-28 to 124.14; ZIMV 25.53 on 2022-03-01 (ZimVie Inc. Common Stock, BBG014630587): total return -0.3908%, the adjusted closes' -2.3980%; midoffice14 brief (Jeff, 2026-11-12), item 2: a pair it names, not carried; carried by item 1's rule
  a split the cache does not record, applied to the count: HON 2-to-1 on 2026-06-29 (0.5, whole numbers): counts 635,675,701 on 2026-04-13, 316,940,010 on 2026-10-12, 0.499 times; raw price no raw price on a side; a split, the counts follow it; a spin-off ruling's split the cache does not record, its price not tested -- the distribution the same day offsets the gap
  FDXF   BBG01ZPB91C4: no count of its own -- FDX's 238,606,756 on the record date 2026-05-15 times 0.5, 119,303,378 distributed, over the 0.801 FDX did not keep, 148,943,044, from 2026-06-01 to the window's end, Polygon carrying none: 109 days, 109 member-days
  VYLR   BBG024CQR6X9: no count of its own -- CTVA's 668,820,000 on the record date 2026-09-24 times 1, 668,820,000, from 2026-10-01 to the window's end, Polygon carrying none: 24 days, 24 member-days
  a count override superseded by HON's ruling of 2026-06-29, not read: HON BBG000H556T9 2026-10-12..2026-11-03 (midoffice08, decision 3 (Jeff, 2026-11-10))
[...]
The weighted-count check: a record's weighted share count against the count the panel carries that day, flagged beyond 10%
  carried from an earlier record: 49 met, 4 flagged; ratio median 1.0003, p1 0.9361, p99 2.2242
  the day's own record: 11,849 met, 754 flagged; ratio median 1.0000, p1 0.9825, p99 2.0917
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
    FDXF   FDXF   2026-08-26 weighted 149,518,133 against 148,943,044: 1.0039
    FDXF   FDXF   2026-11-03 weighted 149,521,266 against 148,943,044: 1.0039
    GRMN   GRMN   2025-07-11 weighted 192,541,456 against 192,400,000: 1.0007
    GRMN   GRMN   2025-10-09 weighted 192,493,945 against 192,400,000: 1.0005
    GRMN   GRMN   2026-01-09 weighted 192,334,806 against 192,400,000: 0.9997
    GRMN   GRMN   2026-04-13 weighted 192,480,830 against 192,400,000: 1.0004
    HON    HON    2026-07-14 weighted 316,826,559 against 317,837,850: 0.9968
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
    VYLR   VYLR   2026-10-02 weighted 711,546,000 against 668,820,000: 1.0639
    VYLR   VYLR   2026-10-05 weighted 711,546,000 against 668,820,000: 1.0639
    VYLR   VYLR   2026-11-03 weighted 667,302,930 against 668,820,000: 0.9977
    WSM    WSM    2024-07-09 weighted 128,616,166 against 128,220,000: 1.0031
    XOM    XOM    2026-07-02 weighted 4,144,947,162 against 4,144,947,162: 1.0000
    XOM    XOM    2026-10-01 weighted 4,111,911,960 against 4,144,947,162: 0.9920
    XOM    XOM    2026-11-03 weighted 4,079,000,000 against 4,144,947,162: 0.9841
    XYZ    XYZ    2026-01-09 weighted 607,665,000 against 549,574,000: 1.1057  FLAGGED
```

### model_compare.txt -- the defaults' model on `_revised` (M15) against midoffice14's on `_spinoffs` (M14)

```text
claude/scratch/midoffice14/model/M14 against claude/scratch/midoffice15/model/M15: 21 fields compared; equal: 11; differing: covariance, observed_variance, sampling_variance, weighted_sampling_variance, noise_share, weighted_noise_share, mean_r2, factor_returns, specific, exposures
  covariance: largest difference 4.895e-05
  observed_variance: largest difference 4.895e-05
  sampling_variance: largest difference 1.509e-07
  weighted_sampling_variance: largest difference 7.323e-07
  noise_share: largest difference 3.663e-04
  weighted_noise_share: largest difference 2.102e-04
  mean_r2: largest difference 5.560e-06
  factor_returns: largest difference 9.912e-04
  specific: largest difference 4.392e-04
  exposures: largest difference 1.878e-01
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 64 differing line(s)
    -  454,423 raw values winsorised before standardising
    -  183,708 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    +  454,475 raw values winsorised before standardising
    +  183,725 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    -  CMRA         93%      938    3.75%      6.27%     1.15
    +  CMRA         93%      938    3.75%      6.27%     1.16
    -  LCAPCB       93%    1,213    1.65%      3.93%     0.42
    +  LCAPCB       93%    1,213    1.65%      3.92%     0.41
    -  Volatility                  -4.0e-17   1.000000   503
    -  Momentum                    +7.2e-18   1.000000   500
    -  Size                        +7.2e-17   1.000000   502
    -  Size Nonlinearity           -2.2e-17   1.000000   502
    -  Trading Activity            -2.2e-17   1.000000   503
    -  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,107 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates
    +  Volatility                  -1.6e-17   1.000000   503
    +  Momentum                    -2.9e-17   1.000000   500
    +  Size                        +2.9e-17   1.000000   502
    +  Size Nonlinearity           +0.0e+00   1.000000   502
    +  Trading Activity            +0.0e+00   1.000000   503
    +  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,111 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates
    -  smallest 85, equally weighted                -3.137             -1.806   11.05%
    +  smallest 85, equally weighted                -3.138             -1.809   11.05%
    -  the index, cap-weighted                      +0.003             -0.001    0.01%
    +  the index, cap-weighted                      +0.004             -0.001    0.01%
    -  Finance & Real Estate      17.34%     +17.97%    24.1%      37.1%      938          0      0
    -  Manufacturing              12.07%     +20.71%    16.5%      24.3%      938          0      0
    -  Mining                     21.16%     +14.75%    14.2%      21.4%      938          0      0
    -  Retail                     14.75%     +14.55%     8.4%      12.9%      938          0      0
    -  SIC20                      13.35%     +10.88%    26.7%      45.3%      938          0      0
    -  SIC28                      16.14%     +16.53%     6.7%      10.6%      938          0      0
    +  Finance & Real Estate      17.35%     +17.98%    24.1%      37.0%      938          0      0
    +  Manufacturing              12.07%     +20.72%    16.5%      24.3%      938          0      0
    +  Mining                     21.15%     +14.77%    14.2%      21.4%      938          0      0
    +  Retail                     14.75%     +14.56%     8.4%      12.9%      938          0      0
    +  SIC20                      13.35%     +10.89%    26.7%      45.3%      938          0      0
    +  SIC28                      16.14%     +16.55%     6.7%      10.6%      938          0      0
    -  SIC38                      16.82%     +16.01%     8.0%      12.3%      938          0      0
    -  SIC49                      19.02%     +23.73%     7.7%      11.9%      938          0      0
    -  SIC60                      23.21%     +21.77%     6.5%       9.5%      938          0      0
    -  SIC62                      21.57%     +18.48%     8.6%      11.5%      938          0      0
    -  SIC63                      14.22%     +19.00%    15.1%      24.2%      938          0      0
    -  SIC67                      17.83%     +17.79%    11.8%      18.3%      938          0      0
    -  SIC73                      17.53%     +15.26%     3.1%       4.9%      938          0      0
    -  Services                   14.47%     +15.94%    21.5%      33.5%      938          0      0
    -  Transport & Utilities      16.26%     +15.54%    11.2%      17.5%      938          0      0
    -  Wholesale                  15.06%     +20.46%    39.5%      60.4%      938          0      0
    -  Volatility                 10.62%      +2.97%     2.7%       4.5%      938          0      0
    +  SIC38                      16.82%     +16.02%     8.0%      12.3%      938          0      0
    +  SIC49                      19.01%     +23.75%     7.8%      11.9%      938          0      0
    +  SIC60                      23.21%     +21.78%     6.5%       9.5%      938          0      0
    +  SIC62                      21.57%     +18.49%     8.6%      11.5%      938          0      0
    +  SIC63                      14.22%     +19.02%    15.1%      24.2%      938          0      0
    +  SIC67                      17.83%     +17.80%    11.8%      18.3%      938          0      0
    +  SIC73                      17.52%     +15.23%     3.1%       4.9%      938          0      0
    +  Services                   14.47%     +15.95%    21.5%      33.5%      938          0      0
    +  Transport & Utilities      16.26%     +15.55%    11.2%      17.5%      938          0      0
    +  Wholesale                  15.06%     +20.47%    39.5%      60.4%      938          0      0
    +  Volatility                 10.62%      +2.98%     2.7%       4.5%      938          0      0
    -  Size                        3.62%      +3.05%     8.0%      11.4%      938          0      0
    +  Size                        3.62%      +3.05%     8.0%      11.5%      938          0      0
    -  specific risk   median 28.99%   p10 16.56%   p90 48.89%
    +  specific risk   median 28.98%   p10 16.56%   p90 48.89%
    -                    each history weighted by a half-life of 63 trading days; the characteristics' weight in a blended forecast, at the median effective history of 181.8 observations of 938: 13.59%
    +                    each history weighted by a half-life of 63 trading days; the characteristics' weight in a blended forecast, at the median effective history of 181.8 observations of 938: 13.58%
```

### compare_bias.txt -- the defaults' bias run on `_revised` (D15) against midoffice14's D14, every statistic recomputed

```text
D14: midoffice14's defaults on _spinoffs; D15: the defaults on _revised. 686 forecast dates, 2024-02-08 to 2026-11-02; the first half to 2025-06-23, the second from 2025-06-24

The minimum-variance portfolios at one day: the bias statistic, the volatility a year (the forecast at its median, the realised from the returns), by part of the test
                                                                       D14                    D15
  minimum variance, long-only          whole           1.059  9.42%  9.92%    1.059  9.41%  9.92%
  minimum variance, long-only          first half      1.247  8.28% 11.47%    1.247  8.28% 11.47%
  minimum variance, long-only          second half     0.831  9.54%  8.09%    0.831  9.54%  8.09%
  minimum variance, unconstrained      whole           1.101  9.12% 10.01%    1.101  9.12% 10.00%
  minimum variance, unconstrained      first half      1.298  8.05% 11.57%    1.298  8.05% 11.57%
  minimum variance, unconstrained      second half     0.862  9.29%  8.16%    0.862  9.29%  8.16%
  minimum variance, long-only, D14: MRAD 0.2287
  minimum variance, long-only, D15: MRAD 0.2287
  minimum variance, unconstrained, D14: MRAD 0.2468
  minimum variance, unconstrained, D15: MRAD 0.2468

The minimum-variance portfolios at the other horizons, the windows overlapping: D14 / D15
  minimum variance, long-only           5 days 1.074/1.074  11 days 0.991/0.991  21 days 0.917/0.917
  minimum variance, unconstrained       5 days 1.112/1.112  11 days 1.011/1.011  21 days 0.941/0.941

The broad portfolios: the statistic at one day by part, and at 21 days -- D14 / D15
  the index, cap-weighted                1.015/1.015    1.224/1.224    0.751/0.751    0.860/0.860   (whole, first, second; 21 days)
  the index, equally weighted            0.966/0.966    1.143/1.143    0.749/0.749    0.805/0.805   (whole, first, second; 21 days)
  smallest 85, equally weighted          0.956/0.956    1.065/1.065    0.834/0.834    0.776/0.776   (whole, first, second; 21 days)
  largest 50, equally weighted           0.996/0.996    1.175/1.175    0.778/0.778    0.873/0.873   (whole, first, second; 21 days)
  the index, cap-weighted              volatility a year, forecast / realised: D14 15.48% / 15.27%, D15 15.48% / 15.27%; MRAD 0.1979 / 0.1979
  the index, equally weighted          volatility a year, forecast / realised: D14 13.98% / 13.96%, D15 13.98% / 13.96%; MRAD 0.1766 / 0.1766
  smallest 85, equally weighted        volatility a year, forecast / realised: D14 17.70% / 17.38%, D15 17.70% / 17.38%; MRAD 0.1288 / 0.1288
  largest 50, equally weighted         volatility a year, forecast / realised: D14 15.94% / 14.64%, D15 15.94% / 14.64%; MRAD 0.1644 / 0.1644

Every member singly (at least 63 forecasts):
  whole        543 / 543 members: median 1.003 / 1.003, mean abs(bias - 1) 0.088 / 0.088
  first half   518 / 518 members: median 1.037 / 1.037, mean abs(bias - 1) 0.105 / 0.105
  second half  519 / 519 members: median 0.965 / 0.965, mean abs(bias - 1) 0.101 / 0.101
  the members whose statistic moved most, D14 -> D15: SPGI 1.021 -> 1.010, MA 0.938 -> 0.938, PTC 1.199 -> 1.199, ROL 0.895 -> 0.895, MSFT 1.096 -> 1.096, ALLE 0.933 -> 0.933, ADSK 1.121 -> 1.121, BR 0.958 -> 0.958, LDOS 0.932 -> 0.932, UBER 1.025 -> 1.025, EFX 1.082 -> 1.082, FERG 0.748 -> 0.748

Each spin-off parent with an ex-date in the test, its member statistic D14 -> D15, and its outcome on the day in forecast sd:
  J      0.748 -> 0.748; 2024-09-30: +2.4 -> +2.4 sd; 2025-05-16: +0.6 -> +0.6 sd
  LEN    1.110 -> 1.110; 2025-01-21: +1.9 -> +1.9 sd
  WDC    1.144 -> 1.144; 2025-02-24: -2.0 -> -2.0 sd
  CMCSA  1.029 -> 1.030; 2026-01-05: +0.4 -> +0.4 sd
  BDX    1.016 -> 1.016; 2026-02-10: +2.4 -> +2.4 sd
  APTV   0.963 -> 0.963; 2026-04-01: +1.2 -> +1.2 sd
  FDX    1.041 -> 1.041; 2026-06-01: +0.2 -> +0.2 sd
  HON    0.867 -> 0.867; 2026-06-29: -1.9 -> -1.9 sd
  SPGI   1.021 -> 1.010; 2026-07-01: +3.8 -> +1.0 sd
  CTVA   0.874 -> 0.874; 2026-10-01: +2.0 -> +2.0 sd
  FDXF: held on 0 / 0 forecast dates
  VYLR: held on 0 / 0 forecast dates

The industries, mean abs(bias - 1) at one day: 0.046 / 0.046
Random portfolios, long-only, median at one day: 0.971 / 0.971
Random portfolios, long-short, median at one day: 1.004 / 1.004
Eigenportfolios 1-5, mean at one day: 1.030 / 1.030
Eigenportfolios 6-10, mean at one day: 1.083 / 1.083
Eigenportfolios 11-15, mean at one day: 1.114 / 1.114
Eigenportfolios 16-20, mean at one day: 1.217 / 1.217
Eigenportfolios 21-25, mean at one day: 1.194 / 1.194

The arrays' equality, D14 and D15: {'caps': False, 'held_long_only': False, 'held_unconstrained': True, 'repaired': True, 'forecast': False, 'realised': False, 'member_forecast': False, 'member_realised': False, 'eigen_forecast': False}
  the capitalisations that differ: 108 member-days, in FDXF
```

### equal_V15_D15.txt -- V15's model test against D15, bit for bit

```text
claude/scratch/midoffice15/bias/D15.npz against claude/scratch/midoffice15/bias/V15.npz: 686 forecast dates in common (686 and 686), 2024-02-08 to 2026-11-02
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
  only in the second: eigen_forecast_rF21S21, eigen_forecast_rF21S42, eigen_forecast_rF21S84, eigen_forecast_rF21Sx, eigen_forecast_rF42S21, eigen_forecast_rF42S42, eigen_forecast_rF42S84, eigen_forecast_rF42Sx, eigen_forecast_rF84S21, eigen_forecast_rF84S42, eigen_forecast_rF84S84, eigen_forecast_rF84Sx, eigen_forecast_rFxS21, eigen_forecast_rFxS42, eigen_forecast_rFxS84, eigen_realised_h11_rF21S21, eigen_realised_h11_rF21S42, eigen_realised_h11_rF21S84, eigen_realised_h11_rF21Sx, eigen_realised_h11_rF42S21, eigen_realised_h11_rF42S42, eigen_realised_h11_rF42S84, eigen_realised_h11_rF42Sx, eigen_realised_h11_rF84S21, eigen_realised_h11_rF84S42, eigen_realised_h11_rF84S84, eigen_realised_h11_rF84Sx, eigen_realised_h11_rFxS21, eigen_realised_h11_rFxS42, eigen_realised_h11_rFxS84, eigen_realised_h21_rF21S21, eigen_realised_h21_rF21S42, eigen_realised_h21_rF21S84, eigen_realised_h21_rF21Sx, eigen_realised_h21_rF42S21, eigen_realised_h21_rF42S42, eigen_realised_h21_rF42S84, eigen_realised_h21_rF42Sx, eigen_realised_h21_rF84S21, eigen_realised_h21_rF84S42, eigen_realised_h21_rF84S84, eigen_realised_h21_rF84Sx, eigen_realised_h21_rFxS21, eigen_realised_h21_rFxS42, eigen_realised_h21_rFxS84, eigen_realised_h5_rF21S21, eigen_realised_h5_rF21S42, eigen_realised_h5_rF21S84, eigen_realised_h5_rF21Sx, eigen_realised_h5_rF42S21, eigen_realised_h5_rF42S42, eigen_realised_h5_rF42S84, eigen_realised_h5_rF42Sx, eigen_realised_h5_rF84S21, eigen_realised_h5_rF84S42, eigen_realised_h5_rF84S84, eigen_realised_h5_rF84Sx, eigen_realised_h5_rFxS21, eigen_realised_h5_rFxS42, eigen_realised_h5_rFxS84, eigen_realised_rF21S21, eigen_realised_rF21S42, eigen_realised_rF21S84, eigen_realised_rF21Sx, eigen_realised_rF42S21, eigen_realised_rF42S42, eigen_realised_rF42S84, eigen_realised_rF42Sx, eigen_realised_rF84S21, eigen_realised_rF84S42, eigen_realised_rF84S84, eigen_realised_rF84Sx, eigen_realised_rFxS21, eigen_realised_rFxS42, eigen_realised_rFxS84, forecast_rF21S21, forecast_rF21S42, forecast_rF21S84, forecast_rF21Sx, forecast_rF42S21, forecast_rF42S42, forecast_rF42S84, forecast_rF42Sx, forecast_rF84S21, forecast_rF84S42, forecast_rF84S84, forecast_rF84Sx, forecast_rFxS21, forecast_rFxS42, forecast_rFxS84, held_long_only_rF21S21, held_long_only_rF21S42, held_long_only_rF21S84, held_long_only_rF21Sx, held_long_only_rF42S21, held_long_only_rF42S42, held_long_only_rF42S84, held_long_only_rF42Sx, held_long_only_rF84S21, held_long_only_rF84S42, held_long_only_rF84S84, held_long_only_rF84Sx, held_long_only_rFxS21, held_long_only_rFxS42, held_long_only_rFxS84, held_unconstrained_rF21S21, held_unconstrained_rF21S42, held_unconstrained_rF21S84, held_unconstrained_rF21Sx, held_unconstrained_rF42S21, held_unconstrained_rF42S42, held_unconstrained_rF42S84, held_unconstrained_rF42Sx, held_unconstrained_rF84S21, held_unconstrained_rF84S42, held_unconstrained_rF84S84, held_unconstrained_rF84Sx, held_unconstrained_rFxS21, held_unconstrained_rFxS42, held_unconstrained_rFxS84, member_forecast_rF21S21, member_forecast_rF21S42, member_forecast_rF21S84, member_forecast_rF21Sx, member_forecast_rF42S21, member_forecast_rF42S42, member_forecast_rF42S84, member_forecast_rF42Sx, member_forecast_rF84S21, member_forecast_rF84S42, member_forecast_rF84S84, member_forecast_rF84Sx, member_forecast_rFxS21, member_forecast_rFxS42, member_forecast_rFxS84, realised_h11_rF21S21, realised_h11_rF21S42, realised_h11_rF21S84, realised_h11_rF21Sx, realised_h11_rF42S21, realised_h11_rF42S42, realised_h11_rF42S84, realised_h11_rF42Sx, realised_h11_rF84S21, realised_h11_rF84S42, realised_h11_rF84S84, realised_h11_rF84Sx, realised_h11_rFxS21, realised_h11_rFxS42, realised_h11_rFxS84, realised_h21_rF21S21, realised_h21_rF21S42, realised_h21_rF21S84, realised_h21_rF21Sx, realised_h21_rF42S21, realised_h21_rF42S42, realised_h21_rF42S84, realised_h21_rF42Sx, realised_h21_rF84S21, realised_h21_rF84S42, realised_h21_rF84S84, realised_h21_rF84Sx, realised_h21_rFxS21, realised_h21_rFxS42, realised_h21_rFxS84, realised_h5_rF21S21, realised_h5_rF21S42, realised_h5_rF21S84, realised_h5_rF21Sx, realised_h5_rF42S21, realised_h5_rF42S42, realised_h5_rF42S84, realised_h5_rF42Sx, realised_h5_rF84S21, realised_h5_rF84S42, realised_h5_rF84S84, realised_h5_rF84Sx, realised_h5_rFxS21, realised_h5_rFxS42, realised_h5_rFxS84, realised_rF21S21, realised_rF21S42, realised_rF21S84, realised_rF21Sx, realised_rF42S21, realised_rF42S42, realised_rF42S84, realised_rF42Sx, realised_rF84S21, realised_rF84S42, realised_rF84S84, realised_rF84Sx, realised_rFxS21, realised_rFxS42, realised_rFxS84, regime_dates, regime_effective, regime_factor_squared, regime_members, regime_settings, regime_specific_squared, repaired_rF21S21, repaired_rF21S42, repaired_rF21S84, repaired_rF21Sx, repaired_rF42S21, repaired_rF42S42, repaired_rF42S84, repaired_rF42Sx, repaired_rF84S21, repaired_rF84S42, repaired_rF84S84, repaired_rF84Sx, repaired_rFxS21, repaired_rFxS42, repaired_rFxS84
every common array equal to the bit
```

### bias/report_V15.txt -- V15's report: the volatility regime adjustment at every setting

```text
  The volatility regime adjustment (Menchero, Orr and Wang; USE4) at one day: each day's factor cross-sectional bias statistic B_F, the root mean square over the 25 factors of each factor's return over its volatility forecast the day before, unadjusted, and the specific B_S over the members' specific returns, cap-weighted (502 members a day at the median, an effective number of 51); each side's multiplier the square root of B^2's exponentially weighted mean through the forecast date, one until a side has 21 days of history -- the test's first 21 forecast dates, 2024-02-08 to 2024-03-08, unadjusted
  the history: 685 days, 2024-02-09 to 2026-11-02 -- each forecast date's own, but the first's
  mean of B^2              F       S
  the whole test       1.115   1.089
  the first half       1.332   1.088
  the second half      0.899   1.090
  by calendar quarter, B_F^2 / B_S^2: 2024 Q1 0.81/0.99; 2024 Q2 0.91/1.13; 2024 Q3 1.36/1.02; 2024 Q4 1.14/1.09; 2025 Q1 1.37/1.19; 2025 Q2 2.10/1.04; 2025 Q3 0.56/1.01; 2025 Q4 0.71/0.98; 2026 Q1 1.20/1.19; 2026 Q2 1.15/1.26; 2026 Q3 0.92/1.09; 2026 Q4 0.95/0.98

  The multipliers' path: as of each calendar quarter's last forecast date, lambda on each side at each half-life
                       F 21    F 42    F 84    S 21    S 42    S 84
  2024-03-28          0.873   0.886   0.893   0.986   0.992   0.994
  2024-06-28          0.901   0.921   0.930   1.054   1.049   1.045
  2024-09-30          1.042   1.062   1.054   0.973   1.005   1.019
  2024-12-31          1.066   1.068   1.062   0.999   1.020   1.027
  2025-03-31          1.178   1.144   1.114   1.010   1.041   1.047
  2025-06-30          1.056   1.198   1.213   0.951   1.001   1.027
  2025-09-30          0.788   0.929   1.041   1.001   1.006   1.018
  2025-12-31          0.787   0.858   0.959   0.924   0.972   0.999
  2026-03-31          1.090   1.025   1.018   0.994   1.021   1.028
  2026-06-30          1.099   1.064   1.043   1.101   1.086   1.067
  2026-09-30          0.929   0.979   1.004   0.970   1.030   1.049
  2026-11-02          0.973   0.983   1.000   1.005   1.024   1.040
  min                 0.776   0.858   0.887   0.908   0.958   0.965   on 2025-10-09, 2025-12-31, 2024-04-01, 2025-07-14, 2024-04-10, 2024-04-10
  max                 1.977   1.628   1.412   1.252   1.168   1.113   on 2025-04-10, 2025-04-10, 2025-04-10, 2026-07-31, 2026-07-31, 2026-07-31
  mean, first half    1.121   1.098   1.073   1.045   1.044   1.041
  mean, second half   0.956   0.995   1.035   1.038   1.040   1.041

  Day to day: how much each setting moves the forecasts -- the median absolute change in log forecast volatility from one forecast date to the next (and its 95th percentile), for the cap-weighted index and for every member singly (the median of each one's median), and the multipliers' own
                   index     p95  members     p95  lambda_F  lambda_S
  F   - S   - *    0.27%   0.97%    1.34%   5.67%     0.00%     0.00%
  F   - S  21      0.29%   0.95%    1.53%   6.12%     0.00%     0.59%
  F   - S  42      0.28%   0.93%    1.42%   5.75%     0.00%     0.30%
  F   - S  84      0.27%   0.95%    1.38%   5.66%     0.00%     0.16%
  F  21 S   -      1.00%   3.58%    1.48%   6.39%     0.83%     0.00%
  F  21 S  21      1.01%   3.55%    1.66%   6.76%     0.83%     0.59%
  F  21 S  42      0.99%   3.55%    1.55%   6.42%     0.83%     0.30%
  F  21 S  84      0.99%   3.56%    1.50%   6.29%     0.83%     0.16%
  F  42 S   -      0.67%   2.45%    1.43%   6.21%     0.45%     0.00%
  F  42 S  21      0.67%   2.47%    1.60%   6.66%     0.45%     0.59%
  F  42 S  42      0.67%   2.46%    1.49%   6.31%     0.45%     0.30%
  F  42 S  84      0.67%   2.45%    1.45%   6.19%     0.45%     0.16%
  F  84 S   -      0.49%   1.85%    1.40%   6.13%     0.25%     0.00%
  F  84 S  21      0.51%   1.85%    1.58%   6.58%     0.25%     0.59%
  F  84 S  42      0.50%   1.84%    1.47%   6.19%     0.25%     0.30%
  F  84 S  84      0.49%   1.84%    1.43%   6.04%     0.25%     0.16%

  the whole test (2024-02-08 to 2026-11-02, 686), by setting (F   - S   - the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
                   index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  F   - S   - *    1.015   0.966   0.956   0.996   1.059   9.922   1.101  10.005   1.003   0.130   0.167   0.238   0.094   0.166
  F   - S  21      1.012   0.965   0.954   0.993   1.047   9.879   1.088   9.960   0.978   0.130   0.166   0.231   0.089   0.162
  F   - S  42      1.013   0.965   0.955   0.993   1.046   9.878   1.088   9.958   0.976   0.130   0.166   0.231   0.090   0.162
  F   - S  84      1.013   0.965   0.955   0.993   1.047   9.881   1.089   9.960   0.980   0.130   0.166   0.231   0.089   0.162
  F  21 S   -      0.934   0.893   0.902   0.920   0.990  10.128   1.033  10.194   0.987   0.131   0.094   0.135   0.085   0.105
  F  21 S  21      0.932   0.893   0.901   0.918   0.976  10.033   1.018  10.097   0.963   0.131   0.095   0.126   0.090   0.104
  F  21 S  42      0.932   0.893   0.901   0.918   0.976  10.037   1.018  10.101   0.962   0.131   0.095   0.126   0.089   0.103
  F  21 S  84      0.932   0.893   0.901   0.918   0.977  10.046   1.019  10.111   0.963   0.131   0.095   0.126   0.089   0.103
  F  42 S   -      0.945   0.898   0.898   0.931   0.997  10.057   1.040  10.132   0.985   0.113   0.102   0.152   0.087   0.114
  F  42 S  21      0.943   0.898   0.897   0.929   0.983   9.980   1.026  10.053   0.960   0.113   0.103   0.144   0.091   0.113
  F  42 S  42      0.943   0.898   0.897   0.929   0.983   9.981   1.026  10.053   0.961   0.113   0.103   0.143   0.091   0.112
  F  42 S  84      0.943   0.898   0.897   0.929   0.984   9.987   1.027  10.060   0.961   0.113   0.103   0.144   0.092   0.113
  F  84 S   -      0.952   0.902   0.898   0.938   1.003  10.025   1.047  10.106   0.982   0.097   0.116   0.173   0.089   0.126
  F  84 S  21      0.950   0.902   0.896   0.935   0.990   9.959   1.032  10.038   0.955   0.097   0.116   0.165   0.093   0.125
  F  84 S  42      0.950   0.902   0.896   0.936   0.990   9.958   1.032  10.037   0.957   0.097   0.116   0.165   0.093   0.125
  F  84 S  84      0.950   0.902   0.897   0.936   0.991   9.962   1.033  10.041   0.958   0.097   0.116   0.165   0.094   0.125

  the first half (2024-02-08 to 2025-06-23, 343), by setting (F   - S   - the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
                   index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  F   - S   - *    1.224   1.143   1.065   1.175   1.247  11.474   1.298  11.570   1.037   0.110   0.165   0.259   0.092   0.172
  F   - S  21      1.221   1.142   1.063   1.171   1.233  11.434   1.282  11.525   1.009   0.110   0.163   0.244   0.087   0.165
  F   - S  42      1.222   1.142   1.063   1.172   1.233  11.433   1.283  11.524   1.009   0.110   0.164   0.245   0.086   0.165
  F   - S  84      1.222   1.142   1.063   1.172   1.234  11.435   1.284  11.527   1.013   0.110   0.164   0.247   0.086   0.165
  F  21 S   -      1.063   0.983   0.921   1.024   1.109  11.881   1.163  11.953   0.978   0.067   0.068   0.150   0.076   0.098
  F  21 S  21      1.061   0.983   0.920   1.022   1.092  11.759   1.144  11.832   0.954   0.067   0.068   0.134   0.081   0.094
  F  21 S  42      1.061   0.983   0.920   1.022   1.092  11.762   1.145  11.837   0.955   0.067   0.068   0.134   0.080   0.094
  F  21 S  84      1.061   0.983   0.920   1.022   1.093  11.772   1.146  11.848   0.956   0.067   0.068   0.136   0.079   0.094
  F  42 S   -      1.097   1.015   0.949   1.056   1.136  11.736   1.190  11.812   0.990   0.071   0.079   0.171   0.081   0.111
  F  42 S  21      1.094   1.014   0.947   1.053   1.119  11.640   1.172  11.714   0.965   0.071   0.079   0.156   0.080   0.105
  F  42 S  42      1.095   1.014   0.948   1.053   1.120  11.641   1.172  11.717   0.966   0.071   0.079   0.157   0.079   0.105
  F  42 S  84      1.095   1.015   0.948   1.054   1.121  11.650   1.174  11.725   0.967   0.071   0.079   0.158   0.079   0.105
  F  84 S   -      1.127   1.044   0.976   1.084   1.161  11.643   1.215  11.724   1.002   0.077   0.096   0.191   0.084   0.124
  F  84 S  21      1.125   1.044   0.974   1.081   1.145  11.564   1.197  11.645   0.975   0.077   0.095   0.176   0.080   0.117
  F  84 S  42      1.125   1.044   0.974   1.081   1.145  11.564   1.198  11.645   0.976   0.077   0.095   0.177   0.080   0.117
  F  84 S  84      1.125   1.044   0.975   1.082   1.147  11.571   1.199  11.652   0.979   0.077   0.095   0.178   0.079   0.118

  the second half (2025-06-24 to 2026-11-02, 343), by setting (F   - S   - the model, *): the statistics, the optimiser's realised volatility % a year, the MRADs and the criterion
                   index   equal   small   large   mv LO     vol    mv U     vol members   eigen  MRAD b MRAD mv  MRAD m    crit
  F   - S   - *    0.751   0.749   0.834   0.778   0.831   8.094   0.862   8.163   0.965   0.165   0.195   0.145   0.082   0.141
  F   - S  21      0.749   0.749   0.833   0.776   0.821   8.046   0.852   8.116   0.942   0.165   0.197   0.158   0.083   0.146
  F   - S  42      0.749   0.749   0.833   0.776   0.820   8.044   0.851   8.113   0.942   0.165   0.197   0.157   0.082   0.145
  F   - S  84      0.749   0.749   0.833   0.776   0.821   8.047   0.851   8.114   0.944   0.165   0.197   0.156   0.082   0.145
  F  21 S   -      0.785   0.793   0.882   0.804   0.855   8.019   0.887   8.077   0.986   0.222   0.159   0.125   0.084   0.123
  F  21 S  21      0.783   0.793   0.881   0.802   0.845   7.959   0.876   8.011   0.962   0.222   0.161   0.138   0.084   0.128
  F  21 S  42      0.783   0.793   0.881   0.802   0.845   7.964   0.875   8.016   0.962   0.222   0.161   0.137   0.085   0.128
  F  21 S  84      0.783   0.793   0.881   0.802   0.845   7.973   0.875   8.024   0.962   0.222   0.161   0.135   0.085   0.127
  F  42 S   -      0.765   0.765   0.846   0.789   0.836   8.052   0.868   8.130   0.969   0.181   0.175   0.134   0.085   0.131
  F  42 S  21      0.763   0.764   0.844   0.786   0.826   8.001   0.857   8.073   0.946   0.181   0.177   0.148   0.081   0.135
  F  42 S  42      0.763   0.764   0.844   0.787   0.825   8.000   0.856   8.072   0.946   0.181   0.177   0.146   0.081   0.135
  F  42 S  84      0.763   0.764   0.844   0.787   0.826   8.004   0.856   8.076   0.946   0.181   0.176   0.145   0.082   0.134
  F  84 S   -      0.739   0.734   0.813   0.766   0.818   8.107   0.848   8.191   0.952   0.143   0.205   0.154   0.082   0.147
  F  84 S  21      0.737   0.734   0.812   0.763   0.808   8.057   0.838   8.137   0.930   0.143   0.207   0.168   0.087   0.154
  F  84 S  42      0.737   0.734   0.812   0.764   0.807   8.054   0.836   8.134   0.929   0.143   0.207   0.166   0.085   0.152
  F  84 S  84      0.737   0.734   0.812   0.764   0.807   8.055   0.836   8.135   0.930   0.143   0.206   0.165   0.083   0.151

  half-lives chosen on the first half by the criterion -- the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median): F  21 S  42 (0.094; the model's 0.172); on the second half it reads 0.128 against the model's 0.141, the best there F  21 S   -
  the factor side alone, the other off: F  21 S   - on the first half (0.098), 0.123 on the second, the best there F  21 S   -
  the specific side alone, the other off: F   - S  21 on the first half (0.165), 0.146 on the second, the best there F   - S   -
```

### analyse_regime.txt -- V15 recomputed from its stored forecasts: the multipliers checked, their path by month, every setting by part, the members by size

```text
V15.npz: 686 forecast dates, 2024-02-08 to 2026-11-02; 16 settings; the first half to 2025-06-23, the second from 2025-06-24

1. Each factor-side setting's eigenportfolio forecasts over the model's, against lambda_F recomputed from the stored history:
   F  21 S   -: the largest difference 4.7e-14
   F  21 S  21: the largest difference 4.7e-14
   F  21 S  42: the largest difference 4.7e-14
   F  21 S  84: the largest difference 4.7e-14
   F  42 S   -: the largest difference 3.7e-14
   F  42 S  21: the largest difference 3.7e-14
   F  42 S  42: the largest difference 3.7e-14
   F  42 S  84: the largest difference 3.7e-14
   F  84 S   -: the largest difference 4.8e-14
   F  84 S  21: the largest difference 4.8e-14
   F  84 S  42: the largest difference 4.8e-14
   F  84 S  84: the largest difference 4.8e-14
   every multiplier one on the first 21 forecast dates (2024-02-08 to 2024-03-08), the history then 21 days long -- REGIME_MIN_DAYS 21; the first adjusted 2024-03-11

2. The multipliers by month: lambda as of each month's last forecast date; and the mean of each day's squared statistic over the month
                  F 21    F 42    F 84    S 21    S 42    S 84   B_F^2   B_S^2
   2024-02-29    1.000   1.000   1.000   1.000   1.000   1.000    0.94    1.01
   2024-03-28    0.873   0.886   0.893   0.986   0.992   0.994    0.72    0.98
   2024-04-30    1.014   0.992   0.982   1.071   1.049   1.040    1.16    1.18
   2024-05-31    0.977   0.972   0.968   1.053   1.046   1.041    0.88    1.09
   2024-06-28    0.901   0.921   0.930   1.054   1.049   1.045    0.66    1.12
   2024-07-31    1.094   1.039   1.014   1.113   1.084   1.069    1.46    1.26
   2024-08-30    1.150   1.113   1.081   1.011   1.033   1.039    1.67    0.90
   2024-09-30    1.042   1.062   1.054   0.973   1.005   1.019    0.91    0.89
   2024-10-31    0.997   1.026   1.031   1.045   1.032   1.031    0.89    1.13
   2024-11-29    1.079   1.076   1.066   1.045   1.040   1.039    1.49    1.16
   2024-12-31    1.066   1.068   1.062   0.999   1.020   1.027    1.09    0.97
   2025-01-31    1.097   1.084   1.073   1.166   1.106   1.077    1.25    1.62
   2025-02-28    1.098   1.085   1.074   1.108   1.090   1.073    1.15    1.11
   2025-03-31    1.178   1.144   1.114   1.010   1.041   1.047    1.68    0.85
   2025-04-30    1.694   1.523   1.370   1.075   1.074   1.067    5.12    1.36
   2025-05-30    1.327   1.354   1.293   1.014   1.046   1.054    0.68    1.00
   2025-06-30    1.056   1.198   1.213   0.951   1.001   1.027    0.43    0.76
   2025-07-31    0.908   1.080   1.145   0.987   1.000   1.020    0.55    0.96
   2025-08-29    0.814   0.987   1.086   0.941   0.976   1.005    0.53    0.88
   2025-09-30    0.788   0.929   1.041   1.001   1.006   1.018    0.60    1.19
   2025-10-31    0.849   0.920   1.018   1.075   1.043   1.035    0.81    1.22
   2025-11-28    0.902   0.929   1.008   1.037   1.029   1.029    0.89    0.98
   2025-12-31    0.787   0.858   0.959   0.924   0.972   0.999    0.45    0.72
   2026-01-30    0.885   0.894   0.962   1.092   1.050   1.036    0.96    1.46
   2026-02-27    1.020   0.976   0.997   1.132   1.089   1.061    1.41    1.44
   2026-03-31    1.090   1.025   1.018   0.994   1.021   1.028    1.22    0.72
   2026-04-30    1.083   1.040   1.027   1.127   1.083   1.058    1.15    1.42
   2026-05-29    1.009   1.009   1.013   1.091   1.073   1.057    0.87    1.09
   2026-06-30    1.099   1.064   1.043   1.101   1.086   1.067    1.42    1.26
   2026-07-31    1.113   1.078   1.053   1.252   1.168   1.113    1.19    1.69
   2026-08-31    1.006   1.028   1.030   1.111   1.108   1.088    0.82    0.89
   2026-09-30    0.929   0.979   1.004   0.970   1.030   1.049    0.73    0.67
   2026-10-30    0.961   0.978   0.998   1.003   1.023   1.040    0.92    0.97
   2026-11-02    0.973   0.983   1.000   1.005   1.024   1.040    1.61    1.15
   mean, whole            1.039   1.047   1.054   1.041   1.042   1.041
   mean, first half       1.121   1.098   1.073   1.045   1.044   1.041
   mean, second half      0.956   0.995   1.035   1.038   1.040   1.041
   the day-to-day change, median |d ln lambda| after the start: F 21 0.83%, F 42 0.45%, F 84 0.25%, S 21 0.59%, S 42 0.30%, S 84 0.16%

3. Every setting by part: the statistic (and MRAD) of the broad portfolios and the optimiser; the optimiser's realised volatility a year; the members' median, mean abs(bias - 1) and median MRAD; the three MRADs' mean, the criterion

   whole
                        index         equal         small         large         mv LO          mv U  vol LO   vol U  members  |b-1|  MRAD m   crit
   F   - S   -   1.015/0.198   0.966/0.177   0.956/0.129   0.996/0.164   1.059/0.229   1.101/0.247    9.92%  10.00%    1.003  0.088   0.094  0.166
   F   - S  21   1.012/0.197   0.965/0.176   0.954/0.128   0.993/0.163   1.047/0.223   1.088/0.239    9.88%   9.96%    0.978  0.087   0.089  0.162
   F   - S  42   1.013/0.197   0.965/0.176   0.955/0.128   0.993/0.163   1.046/0.222   1.088/0.239    9.88%   9.96%    0.976  0.087   0.090  0.162
   F   - S  84   1.013/0.197   0.965/0.176   0.955/0.128   0.993/0.163   1.047/0.223   1.089/0.240    9.88%   9.96%    0.980  0.087   0.089  0.162
   F  21 S   -   0.934/0.098   0.893/0.089   0.902/0.084   0.920/0.104   0.990/0.126   1.033/0.143   10.13%  10.19%    0.987  0.088   0.085  0.105
   F  21 S  21   0.932/0.099   0.893/0.089   0.901/0.086   0.918/0.105   0.976/0.118   1.018/0.134   10.03%  10.10%    0.963  0.090   0.090  0.104
   F  21 S  42   0.932/0.099   0.893/0.089   0.901/0.086   0.918/0.105   0.976/0.118   1.018/0.134   10.04%  10.10%    0.962  0.089   0.089  0.103
   F  21 S  84   0.932/0.099   0.893/0.089   0.901/0.086   0.918/0.105   0.977/0.118   1.019/0.134   10.05%  10.11%    0.963  0.089   0.089  0.103
   F  42 S   -   0.945/0.110   0.898/0.101   0.898/0.090   0.931/0.109   0.997/0.143   1.040/0.160   10.06%  10.13%    0.985  0.088   0.087  0.114
   F  42 S  21   0.943/0.110   0.898/0.101   0.897/0.092   0.929/0.110   0.983/0.136   1.026/0.151    9.98%  10.05%    0.960  0.091   0.091  0.113
   F  42 S  42   0.943/0.110   0.898/0.101   0.897/0.092   0.929/0.109   0.983/0.136   1.026/0.151    9.98%  10.05%    0.961  0.090   0.091  0.112
   F  42 S  84   0.943/0.110   0.898/0.101   0.897/0.092   0.929/0.109   0.984/0.136   1.027/0.152    9.99%  10.06%    0.961  0.090   0.092  0.113
   F  84 S   -   0.952/0.132   0.902/0.114   0.898/0.098   0.938/0.120   1.003/0.164   1.047/0.182   10.02%  10.11%    0.982  0.089   0.089  0.126
   F  84 S  21   0.950/0.131   0.902/0.114   0.896/0.099   0.935/0.121   0.990/0.157   1.032/0.173    9.96%  10.04%    0.955  0.091   0.093  0.125
   F  84 S  42   0.950/0.131   0.902/0.114   0.896/0.099   0.936/0.121   0.990/0.157   1.032/0.173    9.96%  10.04%    0.957  0.091   0.093  0.125
   F  84 S  84   0.950/0.131   0.902/0.114   0.897/0.099   0.936/0.120   0.991/0.157   1.033/0.174    9.96%  10.04%    0.958  0.091   0.094  0.125

   first half
                        index         equal         small         large         mv LO          mv U  vol LO   vol U  members  |b-1|  MRAD m   crit
   F   - S   -   1.224/0.213   1.143/0.159   1.065/0.120   1.175/0.169   1.247/0.230   1.298/0.288   11.47%  11.57%    1.037  0.105   0.092  0.172
   F   - S  21   1.221/0.210   1.142/0.159   1.063/0.119   1.171/0.166   1.233/0.217   1.282/0.272   11.43%  11.52%    1.009  0.098   0.087  0.165
   F   - S  42   1.222/0.210   1.142/0.159   1.063/0.119   1.172/0.166   1.233/0.218   1.283/0.273   11.43%  11.52%    1.009  0.098   0.086  0.165
   F   - S  84   1.222/0.211   1.142/0.159   1.063/0.119   1.172/0.167   1.234/0.219   1.284/0.274   11.44%  11.53%    1.013  0.098   0.086  0.165
   F  21 S   -   1.063/0.079   0.983/0.055   0.921/0.084   1.024/0.055   1.109/0.122   1.163/0.177   11.88%  11.95%    0.978  0.099   0.076  0.098
   F  21 S  21   1.061/0.078   0.983/0.055   0.920/0.086   1.022/0.055   1.092/0.109   1.144/0.158   11.76%  11.83%    0.954  0.101   0.081  0.094
   F  21 S  42   1.061/0.078   0.983/0.055   0.920/0.086   1.022/0.055   1.092/0.109   1.145/0.159   11.76%  11.84%    0.955  0.101   0.080  0.094
   F  21 S  84   1.061/0.078   0.983/0.055   0.920/0.085   1.022/0.055   1.093/0.111   1.146/0.161   11.77%  11.85%    0.956  0.100   0.079  0.094
   F  42 S   -   1.097/0.106   1.015/0.075   0.949/0.060   1.056/0.076   1.136/0.144   1.190/0.199   11.74%  11.81%    0.990  0.098   0.081  0.111
   F  42 S  21   1.094/0.104   1.014/0.075   0.947/0.062   1.053/0.075   1.119/0.131   1.172/0.181   11.64%  11.71%    0.965  0.099   0.080  0.105
   F  42 S  42   1.095/0.105   1.014/0.075   0.948/0.062   1.053/0.075   1.120/0.131   1.172/0.182   11.64%  11.72%    0.966  0.099   0.079  0.105
   F  42 S  84   1.095/0.105   1.015/0.075   0.948/0.061   1.054/0.075   1.121/0.132   1.174/0.184   11.65%  11.73%    0.967  0.098   0.079  0.105
   F  84 S   -   1.127/0.132   1.044/0.093   0.976/0.063   1.084/0.097   1.161/0.163   1.215/0.220   11.64%  11.72%    1.002  0.098   0.084  0.124
   F  84 S  21   1.125/0.129   1.044/0.093   0.974/0.062   1.081/0.095   1.145/0.150   1.197/0.203   11.56%  11.64%    0.975  0.097   0.080  0.117
   F  84 S  42   1.125/0.129   1.044/0.093   0.974/0.062   1.081/0.095   1.145/0.150   1.198/0.204   11.56%  11.65%    0.976  0.097   0.080  0.117
   F  84 S  84   1.125/0.130   1.044/0.093   0.975/0.062   1.082/0.095   1.147/0.151   1.199/0.205   11.57%  11.65%    0.979  0.097   0.079  0.118

   second half
                        index         equal         small         large         mv LO          mv U  vol LO   vol U  members  |b-1|  MRAD m   crit
   F   - S   -   0.751/0.204   0.749/0.225   0.834/0.172   0.778/0.180   0.831/0.162   0.862/0.128    8.09%   8.16%    0.965  0.101   0.082  0.141
   F   - S  21   0.749/0.206   0.749/0.226   0.833/0.174   0.776/0.183   0.821/0.176   0.852/0.141    8.05%   8.12%    0.942  0.106   0.083  0.146
   F   - S  42   0.749/0.206   0.749/0.226   0.833/0.173   0.776/0.182   0.820/0.174   0.851/0.140    8.04%   8.11%    0.942  0.106   0.082  0.145
   F   - S  84   0.749/0.206   0.749/0.226   0.833/0.173   0.776/0.182   0.821/0.173   0.851/0.139    8.05%   8.11%    0.944  0.106   0.082  0.145
   F  21 S   -   0.785/0.171   0.793/0.182   0.882/0.128   0.804/0.156   0.855/0.142   0.887/0.107    8.02%   8.08%    0.986  0.098   0.084  0.123
   F  21 S  21   0.783/0.173   0.793/0.182   0.881/0.130   0.802/0.159   0.845/0.156   0.876/0.121    7.96%   8.01%    0.962  0.099   0.084  0.128
   F  21 S  42   0.783/0.173   0.793/0.182   0.881/0.129   0.802/0.159   0.845/0.154   0.875/0.120    7.96%   8.02%    0.962  0.099   0.085  0.128
   F  21 S  84   0.783/0.173   0.793/0.182   0.881/0.129   0.802/0.158   0.845/0.153   0.875/0.118    7.97%   8.02%    0.962  0.099   0.085  0.127
   F  42 S   -   0.765/0.185   0.765/0.201   0.846/0.150   0.789/0.165   0.836/0.152   0.868/0.117    8.05%   8.13%    0.969  0.099   0.085  0.131
   F  42 S  21   0.763/0.187   0.764/0.201   0.844/0.152   0.786/0.168   0.826/0.165   0.857/0.130    8.00%   8.07%    0.946  0.104   0.081  0.135
   F  42 S  42   0.763/0.186   0.764/0.201   0.844/0.152   0.787/0.167   0.825/0.164   0.856/0.129    8.00%   8.07%    0.946  0.104   0.081  0.135
   F  42 S  84   0.763/0.186   0.764/0.201   0.844/0.152   0.787/0.167   0.826/0.163   0.856/0.127    8.00%   8.08%    0.946  0.104   0.082  0.134
   F  84 S   -   0.739/0.213   0.734/0.234   0.813/0.184   0.766/0.189   0.818/0.171   0.848/0.137    8.11%   8.19%    0.952  0.104   0.082  0.147
   F  84 S  21   0.737/0.215   0.734/0.234   0.812/0.186   0.763/0.193   0.808/0.185   0.838/0.150    8.06%   8.14%    0.930  0.111   0.087  0.154
   F  84 S  42   0.737/0.215   0.734/0.234   0.812/0.186   0.764/0.192   0.807/0.183   0.836/0.149    8.05%   8.13%    0.929  0.110   0.085  0.152
   F  84 S  84   0.737/0.214   0.734/0.234   0.812/0.185   0.764/0.192   0.807/0.182   0.836/0.148    8.06%   8.13%    0.930  0.110   0.083  0.151

4. Every member singly by capitalisation quintile over the test, the largest first: the median statistic, by setting and part
   whole
   F   - S   -  1.054  1.011  1.001  0.968  0.981
   F   - S  21  1.027  0.983  0.976  0.944  0.953
   F   - S  42  1.022  0.984  0.973  0.946  0.953
   F   - S  84  1.025  0.986  0.972  0.948  0.954
   F  21 S   -  1.028  0.992  0.988  0.949  0.963
   F  21 S  21  1.005  0.970  0.969  0.928  0.938
   F  21 S  42  1.005  0.969  0.965  0.929  0.940
   F  21 S  84  1.006  0.970  0.963  0.930  0.940
   F  42 S   -  1.028  0.993  0.985  0.948  0.960
   F  42 S  21  1.003  0.963  0.964  0.925  0.932
   F  42 S  42  1.003  0.964  0.963  0.925  0.935
   F  42 S  84  1.005  0.966  0.963  0.926  0.937
   F  84 S   -  1.028  0.993  0.979  0.945  0.961
   F  84 S  21  1.002  0.960  0.963  0.923  0.932
   F  84 S  42  1.002  0.962  0.960  0.923  0.934
   F  84 S  84  1.004  0.963  0.959  0.924  0.937
   first half
   F   - S   -  1.088  1.048  1.038  0.998  1.004
   F   - S  21  1.054  1.019  1.010  0.974  0.977
   F   - S  42  1.061  1.020  1.011  0.975  0.982
   F   - S  84  1.065  1.022  1.012  0.977  0.984
   F  21 S   -  1.029  0.982  0.985  0.928  0.958
   F  21 S  21  1.005  0.960  0.961  0.913  0.928
   F  21 S  42  1.007  0.961  0.962  0.912  0.932
   F  21 S  84  1.008  0.963  0.963  0.915  0.934
   F  42 S   -  1.039  0.995  0.995  0.945  0.966
   F  42 S  21  1.014  0.973  0.971  0.922  0.938
   F  42 S  42  1.017  0.974  0.970  0.923  0.941
   F  42 S  84  1.020  0.976  0.972  0.924  0.945
   F  84 S   -  1.052  1.011  1.009  0.957  0.977
   F  84 S  21  1.025  0.986  0.982  0.933  0.948
   F  84 S  42  1.028  0.988  0.982  0.936  0.948
   F  84 S  84  1.031  0.989  0.984  0.937  0.953
   second half
   F   - S   -  1.013  0.972  0.961  0.942  0.939
   F   - S  21  0.984  0.947  0.929  0.917  0.914
   F   - S  42  0.983  0.945  0.931  0.918  0.913
   F   - S  84  0.985  0.947  0.936  0.919  0.913
   F  21 S   -  1.028  0.987  0.976  0.962  0.958
   F  21 S  21  1.003  0.965  0.947  0.937  0.929
   F  21 S  42  1.001  0.963  0.945  0.940  0.930
   F  21 S  84  1.001  0.963  0.947  0.941  0.933
   F  42 S   -  1.013  0.973  0.963  0.948  0.944
   F  42 S  21  0.986  0.951  0.934  0.920  0.919
   F  42 S  42  0.985  0.949  0.938  0.923  0.919
   F  42 S  84  0.988  0.950  0.942  0.925  0.921
   F  84 S   -  0.998  0.960  0.948  0.932  0.927
   F  84 S  21  0.970  0.938  0.921  0.906  0.904
   F  84 S  42  0.971  0.936  0.926  0.907  0.903
   F  84 S  84  0.973  0.936  0.926  0.909  0.905

5. The eigenportfolios' mean statistic by rank group, the industries' mean abs(bias - 1) and the random portfolios' median, at one day over the whole test and by half
   whole
   F   - S   -  eigen 1-5 1.030, 6-20 1.138, 21-25 1.194; industries 0.046; random long-only 0.971, long-short 1.004
   F   - S  21  eigen 1-5 1.030, 6-20 1.138, 21-25 1.194; industries 0.046; random long-only 0.964, long-short 0.971
   F   - S  42  eigen 1-5 1.030, 6-20 1.138, 21-25 1.194; industries 0.046; random long-only 0.964, long-short 0.971
   F   - S  84  eigen 1-5 1.030, 6-20 1.138, 21-25 1.194; industries 0.046; random long-only 0.965, long-short 0.973
   F  21 S   -  eigen 1-5 1.011, 6-20 1.139, 21-25 1.195; industries 0.055; random long-only 0.914, long-short 0.997
   F  21 S  21  eigen 1-5 1.011, 6-20 1.139, 21-25 1.195; industries 0.058; random long-only 0.909, long-short 0.966
   F  21 S  42  eigen 1-5 1.011, 6-20 1.139, 21-25 1.195; industries 0.058; random long-only 0.909, long-short 0.966
   F  21 S  84  eigen 1-5 1.011, 6-20 1.139, 21-25 1.195; industries 0.058; random long-only 0.909, long-short 0.967
   F  42 S   -  eigen 1-5 1.000, 6-20 1.121, 21-25 1.171; industries 0.057; random long-only 0.915, long-short 0.995
   F  42 S  21  eigen 1-5 1.000, 6-20 1.121, 21-25 1.171; industries 0.060; random long-only 0.910, long-short 0.964
   F  42 S  42  eigen 1-5 1.000, 6-20 1.121, 21-25 1.171; industries 0.059; random long-only 0.910, long-short 0.964
   F  42 S  84  eigen 1-5 1.000, 6-20 1.121, 21-25 1.171; industries 0.059; random long-only 0.910, long-short 0.965
   F  84 S   -  eigen 1-5 0.991, 6-20 1.102, 21-25 1.154; industries 0.059; random long-only 0.915, long-short 0.991
   F  84 S  21  eigen 1-5 0.991, 6-20 1.102, 21-25 1.154; industries 0.062; random long-only 0.910, long-short 0.962
   F  84 S  42  eigen 1-5 0.991, 6-20 1.102, 21-25 1.154; industries 0.061; random long-only 0.910, long-short 0.963
   F  84 S  84  eigen 1-5 0.991, 6-20 1.102, 21-25 1.154; industries 0.061; random long-only 0.911, long-short 0.964
   first half
   F   - S   -  eigen 1-5 1.065, 6-20 1.085, 21-25 1.197; industries 0.094; random long-only 1.124, long-short 1.001
   F   - S  21  eigen 1-5 1.065, 6-20 1.085, 21-25 1.197; industries 0.086; random long-only 1.116, long-short 0.969
   F   - S  42  eigen 1-5 1.065, 6-20 1.085, 21-25 1.197; industries 0.087; random long-only 1.116, long-short 0.968
   F   - S  84  eigen 1-5 1.065, 6-20 1.085, 21-25 1.197; industries 0.087; random long-only 1.117, long-short 0.970
   F  21 S   -  eigen 1-5 0.985, 6-20 1.018, 21-25 1.124; industries 0.065; random long-only 0.983, long-short 0.974
   F  21 S  21  eigen 1-5 0.985, 6-20 1.018, 21-25 1.124; industries 0.065; random long-only 0.978, long-short 0.947
   F  21 S  42  eigen 1-5 0.985, 6-20 1.018, 21-25 1.124; industries 0.065; random long-only 0.979, long-short 0.948
   F  21 S  84  eigen 1-5 0.985, 6-20 1.018, 21-25 1.124; industries 0.065; random long-only 0.979, long-short 0.950
   F  42 S   -  eigen 1-5 0.996, 6-20 1.026, 21-25 1.134; industries 0.061; random long-only 1.011, long-short 0.979
   F  42 S  21  eigen 1-5 0.996, 6-20 1.026, 21-25 1.134; industries 0.061; random long-only 1.005, long-short 0.950
   F  42 S  42  eigen 1-5 0.996, 6-20 1.026, 21-25 1.134; industries 0.061; random long-only 1.005, long-short 0.952
   F  42 S  84  eigen 1-5 0.996, 6-20 1.026, 21-25 1.134; industries 0.061; random long-only 1.006, long-short 0.954
   F  84 S   -  eigen 1-5 1.011, 6-20 1.039, 21-25 1.148; industries 0.064; random long-only 1.038, long-short 0.983
   F  84 S  21  eigen 1-5 1.011, 6-20 1.039, 21-25 1.148; industries 0.060; random long-only 1.031, long-short 0.954
   F  84 S  42  eigen 1-5 1.011, 6-20 1.039, 21-25 1.148; industries 0.061; random long-only 1.032, long-short 0.957
   F  84 S  84  eigen 1-5 1.011, 6-20 1.039, 21-25 1.148; industries 0.061; random long-only 1.032, long-short 0.959
   second half
   F   - S   -  eigen 1-5 0.986, 6-20 1.186, 21-25 1.190; industries 0.110; random long-only 0.798, long-short 1.003
   F   - S  21  eigen 1-5 0.986, 6-20 1.186, 21-25 1.190; industries 0.117; random long-only 0.791, long-short 0.972
   F   - S  42  eigen 1-5 0.986, 6-20 1.186, 21-25 1.190; industries 0.116; random long-only 0.792, long-short 0.974
   F   - S  84  eigen 1-5 0.986, 6-20 1.186, 21-25 1.190; industries 0.116; random long-only 0.792, long-short 0.975
   F  21 S   -  eigen 1-5 1.031, 6-20 1.247, 21-25 1.261; industries 0.088; random long-only 0.838, long-short 1.015
   F  21 S  21  eigen 1-5 1.031, 6-20 1.247, 21-25 1.261; industries 0.093; random long-only 0.833, long-short 0.986
   F  21 S  42  eigen 1-5 1.031, 6-20 1.247, 21-25 1.261; industries 0.093; random long-only 0.833, long-short 0.984
   F  21 S  84  eigen 1-5 1.031, 6-20 1.247, 21-25 1.261; industries 0.093; random long-only 0.833, long-short 0.984
   F  42 S   -  eigen 1-5 0.998, 6-20 1.206, 21-25 1.207; industries 0.103; random long-only 0.811, long-short 1.006
   F  42 S  21  eigen 1-5 0.998, 6-20 1.206, 21-25 1.207; industries 0.108; random long-only 0.805, long-short 0.977
   F  42 S  42  eigen 1-5 0.998, 6-20 1.206, 21-25 1.207; industries 0.108; random long-only 0.805, long-short 0.976
   F  42 S  84  eigen 1-5 0.998, 6-20 1.206, 21-25 1.207; industries 0.107; random long-only 0.806, long-short 0.976
   F  84 S   -  eigen 1-5 0.964, 6-20 1.161, 21-25 1.160; industries 0.124; random long-only 0.784, long-short 0.995
   F  84 S  21  eigen 1-5 0.964, 6-20 1.161, 21-25 1.160; industries 0.132; random long-only 0.777, long-short 0.965
   F  84 S  42  eigen 1-5 0.964, 6-20 1.161, 21-25 1.160; industries 0.132; random long-only 0.778, long-short 0.967
   F  84 S  84  eigen 1-5 0.964, 6-20 1.161, 21-25 1.160; industries 0.131; random long-only 0.778, long-short 0.968
```

### eigen_by_half.txt -- why the calm half is a shape: the eigenportfolios by rank and half, the model's and the factor side's at 21 days

```text
The eigenportfolios' statistic by rank (1 the largest eigenvalue, the market-like one), the model / the factor side at 21 days (F 21 S -), by part
  rank  1: whole 1.027/0.946   first half 1.226/1.064   second half 0.780/0.814
  rank  2: whole 0.974/0.978   first half 0.958/0.912   second half 0.991/1.040
  rank  3: whole 1.046/1.051   first half 1.020/0.970   second half 1.073/1.127
  rank  4: whole 1.046/1.033   first half 1.073/1.017   second half 1.019/1.049
  rank  5: whole 1.056/1.047   first half 1.048/0.963   second half 1.065/1.125
  rank  6: whole 1.038/1.041   first half 0.962/0.911   second half 1.109/1.158
  rank  7: whole 1.088/1.084   first half 1.036/0.979   second half 1.139/1.182
  rank  8: whole 1.083/1.083   first half 1.069/0.996   second half 1.099/1.164
  rank  9: whole 1.129/1.128   first half 1.009/0.942   second half 1.238/1.288
  rank 10: whole 1.076/1.084   first half 1.017/0.959   second half 1.131/1.196
  rank 11: whole 1.101/1.088   first half 1.022/0.942   second half 1.175/1.217
  rank 12: whole 1.099/1.098   first half 1.066/0.998   second half 1.130/1.190
  rank 13: whole 1.110/1.096   first half 1.144/1.062   second half 1.075/1.128
  rank 14: whole 1.103/1.109   first half 1.051/1.006   second half 1.155/1.205
  rank 15: whole 1.157/1.163   first half 1.118/1.057   second half 1.191/1.258
  rank 16: whole 1.173/1.186   first half 1.179/1.122   second half 1.169/1.248
  rank 17: whole 1.157/1.156   first half 1.113/1.058   second half 1.200/1.247
  rank 18: whole 1.251/1.245   first half 1.137/1.040   second half 1.355/1.421
  rank 19: whole 1.281/1.288   first half 1.190/1.112   second half 1.368/1.444
  rank 20: whole 1.221/1.238   first half 1.167/1.088   second half 1.263/1.362
  rank 21: whole 1.195/1.182   first half 1.226/1.149   second half 1.165/1.215
  rank 22: whole 1.190/1.177   first half 1.217/1.115   second half 1.163/1.238
  rank 23: whole 1.227/1.224   first half 1.229/1.160   second half 1.225/1.285
  rank 24: whole 1.234/1.255   first half 1.226/1.174   second half 1.239/1.330
  rank 25: whole 1.123/1.135   first half 1.087/1.021   second half 1.159/1.239
  the index, cap-weighted: whole 1.015/0.934   first half 1.224/1.063   second half 0.751/0.785
  the index, equally weighted: whole 0.966/0.893   first half 1.143/0.983   second half 0.749/0.793

The model's mean squared standardised outcome, by part: rank 1 against the mean over ranks 2 to 25
  whole        rank 1 1.059; ranks 2-25 1.286
  first half   rank 1 1.507; ranks 2-25 1.215
  second half  rank 1 0.611; ranks 2-25 1.358
```

### equal_VE15_V15.txt -- VE15's model test against V15's at the chosen setting, bit for bit

```text
claude/scratch/midoffice15/bias/VE15.npz's model test against claude/scratch/midoffice15/bias/V15.npz's _rF21Sx: 686 forecast dates, 2024-02-08 to 2026-11-02
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

### bias/report_VE15.txt -- VE15's report: the eigenfactor adjustment one-sided on the chosen regime adjustment, and the run's own forecast under it

```text
  The eigenfactor risk adjustment (Menchero, Wang and Orr; USE4) at one day: each eigenvalue of the day's factor covariance scaled by max(1, a (lambda - 1) + 1)^2, one-sided, the eigenportfolios tested in the unadjusted covariance's eigenbasis; the simulation 0.49 s a refit (0.31 to 0.68); on the model's volatility regime adjustment, F  21 S   -, each scale's factor multiplier from the history of its own forecasts' bias statistics
  lambda by rank over the 686 dates, the mean (min to max): 1: 1.002 (1.000-1.005), 2: 0.986 (0.972-0.994), 3: 0.977 (0.939-0.996), 5: 1.029 (1.005-1.056), 10: 0.992 (0.975-1.013), 15: 1.031 (1.004-1.062), 20: 1.076 (1.028-1.172), 21: 1.063, 22: 1.060, 23: 1.075, 24: 1.061, 25: 1.065; its Monte Carlo standard error at most 0.0019, 0.0013 at the median date's worst rank

  the whole test (2024-02-08 to 2026-11-02, 686), by a (0 the model):
                                                       a = 0     a = 1     a = 2     a = 3
  eigenportfolios 1-5, mean                            1.011     1.020     1.028     1.037
  eigenportfolios 6-10, mean                           1.084     1.095     1.106     1.116
  eigenportfolios 11-15, mean                          1.111     1.105     1.100     1.095
  eigenportfolios 16-20, mean                          1.223     1.168     1.120     1.077
  eigenportfolios 21-25, mean                          1.195     1.140     1.093     1.050
  eigenportfolios, mean abs(bias - 1)                  0.131     0.110     0.092     0.076
  minimum variance, long-only                          0.990     0.989     0.990     0.992
  minimum variance, unconstrained                      1.033     1.032     1.034     1.037
  minimum variance, mean abs(bias - 1)                 0.022     0.022     0.022     0.023
  minimum variance, long-only: realised vol, %        10.128    10.173    10.232    10.302
  minimum variance, unconstrained: realised vol, %    10.194    10.261    10.332    10.403
  the criterion                                        0.076     0.066     0.057     0.049
  the index, cap-weighted                              0.934     0.945     0.956     0.966
  the index, equally weighted                          0.893     0.896     0.899     0.900
  smallest 85, equally weighted                        0.902     0.900     0.897     0.893
  largest 50, equally weighted                         0.920     0.931     0.941     0.950
  members, median                                      0.987     0.985     0.983     0.981
  members, mean abs(bias - 1)                          0.088     0.089     0.090     0.092
  industries, mean abs(bias - 1)                       0.055     0.055     0.055     0.056
  random long-only, median                             0.914     0.915     0.915     0.916
  random long-short, median                            0.997     0.994     0.991     0.988
  broad portfolios, mean MRAD                          0.094     0.092     0.091     0.091
  minimum variance, mean MRAD                          0.135     0.136     0.139     0.142
  members, median MRAD                                 0.085     0.087     0.087     0.087
  the regime criterion                                 0.105     0.105     0.106     0.107

  the first half (2024-02-08 to 2025-06-23, 343), by a (0 the model):
                                                       a = 0     a = 1     a = 2     a = 3
  eigenportfolios 1-5, mean                            0.985     0.995     1.005     1.014
  eigenportfolios 6-10, mean                           0.957     0.970     0.983     0.995
  eigenportfolios 11-15, mean                          1.013     1.001     0.990     0.979
  eigenportfolios 16-20, mean                          1.084     1.022     0.968     0.922
  eigenportfolios 21-25, mean                          1.124     1.054     0.994     0.943
  eigenportfolios, mean abs(bias - 1)                  0.067     0.042     0.037     0.050
  minimum variance, long-only                          1.109     1.111     1.116     1.124
  minimum variance, unconstrained                      1.163     1.165     1.170     1.176
  minimum variance, mean abs(bias - 1)                 0.136     0.138     0.143     0.150
  minimum variance, long-only: realised vol, %        11.881    11.975    12.089    12.216
  minimum variance, unconstrained: realised vol, %    11.953    12.073    12.193    12.308
  the criterion                                        0.101     0.090     0.090     0.100
  the index, cap-weighted                              1.063     1.077     1.091     1.104
  the index, equally weighted                          0.983     0.987     0.989     0.990
  smallest 85, equally weighted                        0.921     0.918     0.914     0.908
  largest 50, equally weighted                         1.024     1.037     1.049     1.060
  members, median                                      0.978     0.976     0.971     0.968
  members, mean abs(bias - 1)                          0.099     0.100     0.102     0.103
  industries, mean abs(bias - 1)                       0.065     0.067     0.070     0.073
  random long-only, median                             0.983     0.986     0.987     0.987
  random long-short, median                            0.974     0.971     0.966     0.963
  broad portfolios, mean MRAD                          0.068     0.073     0.080     0.087
  minimum variance, mean MRAD                          0.150     0.151     0.155     0.161
  members, median MRAD                                 0.076     0.078     0.081     0.084
  the regime criterion                                 0.098     0.101     0.105     0.111

  the second half (2025-06-24 to 2026-11-02, 343), by a (0 the model):
                                                       a = 0     a = 1     a = 2     a = 3
  eigenportfolios 1-5, mean                            1.031     1.038     1.046     1.053
  eigenportfolios 6-10, mean                           1.198     1.207     1.216     1.225
  eigenportfolios 11-15, mean                          1.200     1.200     1.200     1.200
  eigenportfolios 16-20, mean                          1.344     1.295     1.250     1.210
  eigenportfolios 21-25, mean                          1.261     1.220     1.183     1.148
  eigenportfolios, mean abs(bias - 1)                  0.222     0.206     0.193     0.180
  minimum variance, long-only                          0.855     0.850     0.845     0.842
  minimum variance, unconstrained                      0.887     0.882     0.879     0.877
  minimum variance, mean abs(bias - 1)                 0.129     0.134     0.138     0.141
  minimum variance, long-only: realised vol, %         8.019     7.993     7.972     7.958
  minimum variance, unconstrained: realised vol, %     8.077     8.069     8.070     8.079
  the criterion                                        0.175     0.170     0.165     0.160
  the index, cap-weighted                              0.785     0.793     0.800     0.808
  the index, equally weighted                          0.793     0.797     0.800     0.802
  smallest 85, equally weighted                        0.882     0.882     0.881     0.879
  largest 50, equally weighted                         0.804     0.812     0.819     0.826
  members, median                                      0.986     0.984     0.981     0.980
  members, mean abs(bias - 1)                          0.098     0.098     0.099     0.100
  industries, mean abs(bias - 1)                       0.088     0.087     0.087     0.086
  random long-only, median                             0.838     0.840     0.841     0.842
  random long-short, median                            1.015     1.015     1.014     1.013
  broad portfolios, mean MRAD                          0.159     0.154     0.150     0.146
  minimum variance, mean MRAD                          0.125     0.128     0.131     0.132
  members, median MRAD                                 0.084     0.086     0.085     0.085
  the regime criterion                                 0.123     0.123     0.122     0.121

  a chosen on the first half by the criterion -- half the eigenportfolios' mean abs(bias - 1) over every rank, half the minimum-variance portfolios', long-only and unconstrained: a = 1 (0.090; the model's 0.101); on the second half it reads 0.170 against the model's 0.175, the best there a = 3

  a chosen on the first half by the regime criterion -- the mean of three MRADs -- each the mean absolute deviation from one of the rolling 12-month windows' statistics, over the part's own windows: the broad portfolios' (the index cap-weighted and equally weighted, the smallest 85 and the largest 50, their mean), the minimum-variance portfolios' (long-only and unconstrained, their mean) and every member's singly (their median): a = 0 (0.098; the model's 0.098); on the second half it reads 0.123 against the model's 0.123, the best there a = 3

  a chosen on the first half by the minimum-variance portfolios' realised volatility, the mean of the two: a = 0 (11.92% a year; the model's 11.92%); on the second half 8.05% against the model's 8.05%, the lowest there a = 3

The volatility regime adjustment (Menchero, Orr and Wang; USE4), F  21 S   -: the history of each day's bias statistics from 2024-02-09 to 2026-11-03, 686 days, each read from the forecast made the day before, unadjusted; a side's multiplier one until it has 21 days
  as of 2026-11-03: lambda_F 0.9641, lambda_S 1.0000; the factor volatilities' median 16.26% to 15.67%, the specific risk's p10 / median / p90 16.56% / 28.98% / 48.89% to 16.56% / 28.98% / 48.89%
  the last day's B_F^2 0.424 over 25 factors, B_S^2 0.701 over 499 members (an effective number of 48)
```

### analyse_eigen.txt -- VE15 recomputed from its stored forecasts: the chain checked, each scale's own multiplier, every scale by part

```text
VE15.npz: 686 forecast dates, 2024-02-08 to 2026-11-02; the model's regime adjustment F 21.0 S None, the eigen scales one-sided at a = 1, 2, 3 beside it (a = 0 the model)

1. The chain: each scale's eigenportfolio forecasts over the model's, against the day's one-sided gamma times its own factor multiplier over the model's
   a = 1: the largest relative difference 2.4e-14; the share of dates each rank is raised, ranks 1-5 0.47, 21-25 1.00; gamma at the smallest rank, the mean 1.065
   a = 2: the largest relative difference 2.4e-14; the share of dates each rank is raised, ranks 1-5 0.47, 21-25 1.00; gamma at the smallest rank, the mean 1.129
   a = 3: the largest relative difference 2.4e-14; the share of dates each rank is raised, ranks 1-5 0.47, 21-25 1.00; gamma at the smallest rank, the mean 1.194

2. Each scale's factor multiplier against the model's, the mean by part (the model's own and each scale's, levelled by its own history)
   whole        a = 0 1.0389  a = 1 1.0239  a = 2 1.0098  a = 3 0.9967  (lambda_S 1.0000)
   first half   a = 0 1.1214  a = 1 1.1039  a = 2 1.0878  a = 3 1.0726  (lambda_S 1.0000)
   second half  a = 0 0.9563  a = 1 0.9438  a = 2 0.9319  a = 3 0.9207  (lambda_S 1.0000)

3. Every scale by part, the minimum-variance portfolios' realised volatility a year first

   whole
             vol LO   vol U  fcst LO  fcst U  mv LO   mv U  index  equal  small  large  members  |b-1|  eig1-5  16-20  21-25  |e-1|    ind    rLO    rLS   crit  regime
   a = 0     10.13%  10.19%    9.05%   8.79%  0.990  1.033  0.934  0.893  0.902  0.920    0.987  0.088   1.011  1.223  1.195  0.131  0.055  0.914  0.997  0.076   0.105
   a = 1     10.17%  10.26%    9.08%   8.84%  0.989  1.032  0.945  0.896  0.900  0.931    0.985  0.089   1.020  1.168  1.140  0.110  0.055  0.915  0.994  0.066   0.105
   a = 2     10.23%  10.33%    9.14%   8.89%  0.990  1.034  0.956  0.899  0.897  0.941    0.983  0.090   1.028  1.120  1.093  0.092  0.055  0.915  0.991  0.057   0.106
   a = 3     10.30%  10.40%    9.18%   8.92%  0.992  1.037  0.966  0.900  0.893  0.950    0.981  0.092   1.037  1.077  1.050  0.076  0.056  0.916  0.988  0.049   0.107

   first half
             vol LO   vol U  fcst LO  fcst U  mv LO   mv U  index  equal  small  large  members  |b-1|  eig1-5  16-20  21-25  |e-1|    ind    rLO    rLS   crit  regime
   a = 0     11.88%  11.95%    8.76%   8.48%  1.109  1.163  1.063  0.983  0.921  1.024    0.978  0.099   0.985  1.084  1.124  0.067  0.065  0.983  0.974  0.101   0.098
   a = 1     11.97%  12.07%    8.83%   8.54%  1.111  1.165  1.077  0.987  0.918  1.037    0.976  0.100   0.995  1.022  1.054  0.042  0.067  0.986  0.971  0.090   0.101
   a = 2     12.09%  12.19%    8.92%   8.65%  1.116  1.170  1.091  0.989  0.914  1.049    0.971  0.102   1.005  0.968  0.994  0.037  0.070  0.987  0.966  0.090   0.105
   a = 3     12.22%  12.31%    9.01%   8.75%  1.124  1.176  1.104  0.990  0.908  1.060    0.968  0.103   1.014  0.922  0.943  0.050  0.073  0.987  0.963  0.100   0.111

   second half
             vol LO   vol U  fcst LO  fcst U  mv LO   mv U  index  equal  small  large  members  |b-1|  eig1-5  16-20  21-25  |e-1|    ind    rLO    rLS   crit  regime
   a = 0      8.02%   8.08%    9.42%   9.15%  0.855  0.887  0.785  0.793  0.882  0.804    0.986  0.098   1.031  1.344  1.261  0.222  0.088  0.838  1.015  0.175   0.123
   a = 1      7.99%   8.07%    9.44%   9.19%  0.850  0.882  0.793  0.797  0.882  0.812    0.984  0.098   1.038  1.295  1.220  0.206  0.087  0.840  1.015  0.170   0.123
   a = 2      7.97%   8.07%    9.47%   9.21%  0.845  0.879  0.800  0.800  0.881  0.819    0.981  0.099   1.046  1.250  1.183  0.193  0.087  0.841  1.014  0.165   0.122
   a = 3      7.96%   8.08%    9.49%   9.24%  0.842  0.877  0.808  0.802  0.879  0.826    0.980  0.100   1.053  1.210  1.148  0.180  0.086  0.842  1.013  0.160   0.121
```

### bias/runner_log.txt -- the runs, one at a time

```text
2026-11-12 19:06:53  runner started, pid 18988, for D15, V15; 5.6 GB available
2026-11-12 19:06:53  D15: 5.6 GB available
2026-11-12 19:06:53  D15: attempt 1: --bias-guard 5 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-12 19:18:16  D15: done in 11.4 min
2026-11-12 19:18:16  V15: 5.1 GB available
2026-11-12 19:18:16  V15: attempt 1: --bias-guard 0 --bias-regime-half-lives 21,42,84 --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-12 19:31:55  V15: done in 13.6 min
2026-11-12 19:31:55  runner finished; every run done
2026-11-12 19:38:20  runner started, pid 24588, for VE15; 4.6 GB available
2026-11-12 19:38:20  VE15: 4.6 GB available
2026-11-12 19:38:20  VE15: attempt 1: --bias-guard 0 --regime-factor-half-life 21 --bias-eigen-scales 1,2,3 --eigen-one-sided --industries --membership --bias --bias-horizons 1,5,11,21
2026-11-12 19:53:42  VE15: done in 15.4 min
2026-11-12 19:53:42  runner finished; every run done
```

### final_checks.txt -- the final code: the three panels rebuilt, the defaults re-captured; the suite, ruff, hygiene

```text
build _listed: exit 0
20
build _spinoffs: exit 0
20
build _revised: exit 0
20
FD15: status 0, mean R2 0.293821, 938 periods, 25 factors, 62s, the estimator 1.3s
claude/scratch/midoffice15/model/M15 against claude/scratch/midoffice15/final/FD15: 21 fields compared; equal: 21; differing: none
  qualifications: 6 and 6; differing: 0
  report: 162 and 162 lines; 0 differing line(s)
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2372 passed, 1 skipped, 19 warnings in 95.60s (0:01:35) ===========
All checks passed!
ruff: clean over the changed Python files
scripts/run_sp500_model.py: CRLF (2695 lines), parses, ASCII
scripts/build_sp500_securities.py: CRLF (751 lines), parses, ASCII
src/praxis/midoffice/refdata/securities.py: CRLF (1254 lines), parses, ASCII
src/praxis/midoffice/model/regime.py: LF (244 lines), parses, ASCII
src/praxis/midoffice/model/bias.py: LF (689 lines), parses, ASCII
src/praxis/midoffice/model/eigen.py: LF (142 lines), parses, ASCII
src/praxis/midoffice/model/crosssectional.py: CRLF (1587 lines), parses, ASCII
tests/midoffice/test_securities.py: CRLF (626 lines), parses, ASCII
tests/midoffice/test_regime.py: LF (209 lines), parses, ASCII
tests/midoffice/test_bias.py: LF (397 lines), parses, ASCII
tests/midoffice/test_eigen.py: LF (138 lines), parses, ASCII
config/sp500_spin_off_overrides.json: LF (264 lines), -, ASCII
config/sp500_share_count_overrides.json: LF (27 lines), -, ASCII
docs/FACTOR_MODEL.md: CRLF (1603 lines), -, ASCII
claude/MIDOFFICE_STATE.md: CRLF (761 lines), -, ASCII
claude/handoffs/BRIEF_midoffice15_volatility_regime_adjustment.md: LF (80 lines), -, ASCII
claude/retros/RETRO_midoffice15_volatility_regime_adjustment.md: LF (800 lines), -, ASCII
all ok
exit 0
```

## git add (always last -- Rule 43)

This piece -- midoffice14's rulings, the volatility regime adjustment and the one-sided eigenfactor adjustment, the tests, the docs, the brief, this retro, and the state file in the same commit:

```
git add scripts/run_sp500_model.py scripts/build_sp500_securities.py src/praxis/midoffice/refdata/securities.py src/praxis/midoffice/model/regime.py src/praxis/midoffice/model/bias.py src/praxis/midoffice/model/eigen.py src/praxis/midoffice/model/crosssectional.py tests/midoffice/test_securities.py tests/midoffice/test_regime.py tests/midoffice/test_bias.py tests/midoffice/test_eigen.py config/sp500_spin_off_overrides.json config/sp500_share_count_overrides.json docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice15_volatility_regime_adjustment.md claude/retros/RETRO_midoffice15_volatility_regime_adjustment.md
```
