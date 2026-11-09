# Retro: Mid-office 05 -- the identity gate

**Brief:** `claude/handoffs/BRIEF_midoffice05_identity_gate.md` -- written in Chat and pasted
into the Code session, transcribed verbatim
**Date:** 2026-11-09
**Duration:** about 76 minutes of implementation and measurement (12:42-13:58 EST: from the
brief's arrival, by the transcript, to the last measurement, by the scratch file times), then
this retro
**Status:** COMPLETE, uncommitted. Two changes, each with its own `git add` at the end: the
first change, then the identity gate. Every guard passes; decisions left for Jeff are below.

---

## Summary

**The first change, its own `git add`.** The two small items, and the "none" that decision 1
keeps runnable.

- **The size pair's portfolio line is standing in the run.** On the forecast date, over the 423
  index members with a forecast: the smallest 85 and the largest 50, equally weighted, and the
  index, cap-weighted -- each one's mean Size and Size Nonlinearity exposures and the pair's
  risk, sqrt(x' S x) over their block of the annualised covariance. At the defaults it prints
  11.94%, 0.38% and 0.01%, midoffice04's figures exactly; under `--size-nonlinearity use3`,
  12.16%, 3.45% and 0.01%; under `none`, Size alone, 12.52%, 0.05% and 0.01%.
- **`size_nonlinearity="none"`** (`--size-nonlinearity none`) is the model with Size alone. It
  is midoffice04's scratch variant without Size Nonlinearity bit for bit -- factor returns,
  covariance, style and forecast exposures; mean R-squared 0.293802 -- its specific risks
  differing only by the MAD spread that came after that variant.
- **The two stale figures in `docs/FACTOR_MODEL.md` are the current run's.** 3,043 member
  values moved by the +/-4 guard is now 4,021 -- three quarters of them LNCAP's and VLVR's
  lower tails, where the cap-weighted mean sits about 1.7 standard deviations above the median,
  so "nearly half in LNCAP's lower tail" became that -- and 100,011 outsider values at a bound
  is now 102,179. The doc also records decisions 1, 2 and 4.
- The model at the defaults is HEAD's bit for bit; the report gains the six lines of the
  portfolio block and nothing else.

**The identity gate, its own `git add`.** Offline except one endpoint, as asked: 634 requests
to `/vX/reference/tickers/{id}/events`, nothing else on the network, no prices, no model change.

- **The table:** 625 rows, one per security per ticker per span of the window, over 598
  securities -- 566 keyed by composite FIGI, 32 by CIK where Polygon has no FIGI -- with the
  evidence for every boundary. 518 rows are the panel's own prices, 106 come from events alone,
  and 1 is presumed. Written beside the panel, gitignored, by `scripts/build_sp500_identity.py`,
  which reads everything back from the cache and runs offline.
- **The guards, each exact, all pass:** all 583,628 priced (ticker, date) cells map to exactly
  one security; no security is under two tickers on one date (625 rows); no ticker names two
  securities on one date; and all 611,899 (member, date) pairs of the window are accounted for
  exactly once -- 607,086 mapped to a security, 4,813 named.
- **Splices, four:** BNY (the BlackRock New York municipal fund to 2026-02-06, BNY Mellon from
  2026-05-21), DOC (Physicians Realty to 2024-02-29, Healthpeak from 2024-03-04, no gap in the
  prices), INFO (IHS Markit to 2022-02-25, the Harbor ETF from 2024-10-10) and META (the
  Roundhill ETF to 2022-01-28, Meta from 2022-06-09). On 1,589 member-days the panel's price
  under a member's ticker is another security's: BNY 1,028, DOC 542, META 19.
- **Reorganisations, six, which the brief did not list:** a new FIGI under the same ticker, the
  prices unbroken -- BLK 2024-10-02, CCL 2026-05-07, J 2022-08-29, LH 2024-05-20, OKE
  2026-09-11 and XOM 2026-07-02. The index runs on through each; before it, the member is the
  old identifier (4,843 member-days, mapped as "predecessor").
- **Truncations, sixteen members:** BALL, BNY, COR, DOC, ECHO, EG, ELV, FBHS, FISV, FRC, GEN,
  META, MRSH, RVTY, SKYD and VMRK traded under another ticker for 6,451 of their member-days --
  each named, with its dates. The brief's ten are among them; CPAY and SW are not, because no
  source names what they traded as before (below).
- **Phantoms, one:** ECHO, a member from the window's start as membership() gives it, joined
  on 2026-03-23 as SATS -- 1,057 phantom member-days, 2022-01-03 to 2026-03-20.
- **Duplicates, none.** The brief's "AvalonBay twice, as AVB and VMRK" is not what the sources
  say: VMRK is Equity Residential -- CIK 0000906107, FIGI BBG000BG8M31, EQR until 2026-08-18 --
  not AvalonBay, CIK 0000915912. AvalonBay was removed on 2026-08-18 as EQR became Vivmark.
  EQR is missing from the panel for 1,159 member-days, a truncation, not a duplicate.
- **membership()'s three inconsistencies, item 5.** KVUE and OKTA are fixed at the source from
  the change table alone: the change of 2026-10-30 contradicts the constituent list in both of
  its tickers, so the list is read caught up with it, and validate() says so. ECHO cannot be
  fixed from the table alone -- it records the addition as SATS, and nothing in it says who SATS
  became -- so membership() now takes dated renames, and given the identity table's (SATS to
  ECHO on 2026-06-24) ECHO joins on 2026-03-23. Read through the table's renames, the change
  table's inconsistencies fall from 27 to 24.
- **What each source resolved.** The cached records name the security behind every priced cell
  of the 506 tickers; the events date every change of hands, name the old ticker of every
  truncated member, and identify 83 of the 86 member tickers the panel lacks. Neither source has
  DAY, KVUE or WBD at all (2,429 member-days, missing), nor who the member was before the first
  of Polygon's events for SW, CPAY, SKYD, SMCI and BG (2,384 member-days, unverified).
- **The fetch the rebuild needs:** 594 securities, 620 spans under 612 tickers -- 106 of them
  not the panel's -- costing 212 price requests, 9,001 share-count samples not on disk of the
  11,862 the plan takes (inside each security's own spans, every 63 trading days and on each
  span's first and last day) and 612 split-event requests: 9,825 in all.
- Tests: 2,253 passed, 1 skipped, 22 of them new. The model is unchanged by the gate, bit for
  bit; only the run's note on unreproduced membership changes shrinks, from 3 to 1.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice05_identity_gate.md`, verbatim from the
transcript's original message (17:42:11Z), the paste wrapper stripped. In outline:

- **Five decisions on midoffice04:** keep Size Nonlinearity, and keep "none" runnable beside
  "use3" and "use4"; regression-weighted orthogonality is the sense intended; USE4's market
  model on a daily panel and USE3's 60 months otherwise, None the default and inf equal
  weights, accepted; the MAD accepted, the weight of history against the model for
  out-of-sample bias statistics to settle; the linear size effect is about 2.9% a year, the raw
  cube having added 1.8 points by splitting it.
- **midoffice05, the identity gate**, the first part of the rebuild: "Offline except one
  endpoint; no prices fetched and no model change." First, as their own change with their own
  git add: the size pair's portfolio line becomes a standing line in the run, and the two stale
  figures in `docs/FACTOR_MODEL.md`, 3,043 and 100,011, are refreshed from the current run.
- Then the gate, six items: (1) an identity table, one row per security per span of dates,
  with the share-class identifier -- composite FIGI, CIK the cross-check -- the ticker, and the
  evidence for each boundary; (2) sources cheapest first, the cached reference records then
  `/vX/reference/tickers/{t}/events` for the 591-name union, reporting what each resolved and
  what neither did; (3) the checks across each ticker's cached snapshots -- name, CIK or FIGI,
  type, a list date after the first price; (4) three exact guards; (5) membership()'s three
  known inconsistencies, fixed at the source if the change table supports it, with tests, or
  reported; (6) the report -- the table's size, every splice, truncation, phantom and duplicate
  with dates, the identity of each of the 591 names, and the fetch the rebuild will need.
- "Write the table where the panel's other derived data lives, gitignored, with the script that
  regenerates it in the repo." Retro and git add lines at the end, then stop.

---

## Files modified

**The first change:**

| File | Change | Lines (new file) |
|---|---|---|
| `src/praxis/midoffice/model/descriptors.py` | `SIZE_NONLINEARITY` gains "none"; `index_descriptors` builds Size Nonlinearity from nothing under it; `build_risk_indices`' docstring says so | 743-764, 962-963 |
| `scripts/run_sp500_model.py` | `size_pair_portfolios`; the standing portfolio block after the size pair's moments; "none" in the help and LCAPCB's note | 351-381, 464, 646-648, 971-993 |
| `tests/midoffice/test_descriptors.py` | 1 new test | 954-989 |
| `docs/FACTOR_MODEL.md` | the two figures and the sentences around them; "none"; decisions 1, 2 and 4 recorded | 353-357, 380, 633-635, 724-725, 746-754 |

**The identity gate:**

| File | Change | Lines (new file) |
|---|---|---|
| `src/praxis/midoffice/refdata/polygon.py` | `PolygonNotFoundError`; a 404 carries the server's message; `ticker_events`, its no-events answer cached too | 60-68, 83-93, 116-119, 182-183, 193-197, 265-295 |
| `src/praxis/midoffice/refdata/sp500.py` | `Rename`; `todays_ticker`; `_read_through`; `_caught_up`; `intervals`, `membership`, `all_constituents` and `validate` take renames | 114-136, 522-705, 761 |
| `src/praxis/midoffice/refdata/identity.py` | new: the identity table -- evidence readers, segmentation, rows, guards, member mapping, renames, findings, fetch plan | new, 1,235 lines |
| `scripts/build_sp500_identity.py` | new: regenerates the table and its report, offline with `--offline` | new, 443 lines |
| `tests/midoffice/test_prices.py` | 3 new tests | 27, 572-653 |
| `tests/midoffice/test_sp500.py` | 4 new tests | 254-343 |
| `tests/midoffice/test_identity.py` | new: 14 tests | new, 304 lines |
| `claude/handoffs/BRIEF_midoffice05_identity_gate.md` | new: the brief, verbatim | new |
| `claude/retros/RETRO_midoffice05_identity_gate.md` | new: this retro | new |

Written outside the repo's tracked files, all gitignored: the table and its companions in
`data/external/polygon/` (below), 634 `events-*.json` answers in `data/external/polygon/cache/`,
and the scratch in `claude/scratch/midoffice05/`. Nothing in `data/` was overwritten or deleted;
every file written there is new.

---

## The first change: the size pair's standing line, the two figures, and "none"

### The standing line

`size_pair_portfolios` takes the model, the forecast date's caps and a mask of who is held --
index members with a forecast and a positive cap -- and returns, for the smallest 85 by cap and
the largest 50, equally weighted, and all of them cap-weighted, the mean forecast exposure to
each factor of the pair and the pair's risk, sqrt(x' S x) over that block of
`model.covariance.matrix`. These are midoffice04's `size_pair.py` definitions; there they were
taken over the 423 members with a forecast in all three variants, here over the run's own 423.
The block prints after the size pair's moments:

```
  The size pair on 2026-11-03, over the 423 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
                                                 Size  Size Nonlinearity     risk
  smallest 85, equally weighted                -3.110             -1.647   11.94%
  largest 50, equally weighted                 +0.012             +0.155    0.38%
  the index, cap-weighted                      +0.002             +0.000    0.01%
```

| | smallest 85 | largest 50 | the index |
|---|---|---|---|
| USE4 (the default) | -3.110 / -1.647, 11.94% | +0.012 / +0.155, 0.38% | +0.002 / +0.000, 0.01% |
| USE3, `--size-nonlinearity use3` | -3.110 / -2.814, 12.16% | +0.012 / +0.551, 3.45% | +0.002 / +0.001, 0.01% |
| none, `--size-nonlinearity none` | -3.110, 12.52% | +0.012, 0.05% | +0.002, 0.01% |

Each figure is midoffice04's `size_pair.txt`. The line handles "none" by printing Size alone.

### "none"

`SIZE_NONLINEARITY = ("use4", "use3", "none")`; under "none" `index_descriptors("Size
Nonlinearity")` is empty, so `build_risk_indices` leaves the index out as it does any index
with no descriptor in the panel, and `blend_composition` returns nothing for it. In the run,
LCAPCB's note says Size Nonlinearity is not in the model. midoffice04 measured "none" by
dropping the index from the call in scratch; this is that, as an option:

| | against | factor returns | covariance | style exposures | forecast exposures | specific risks |
|---|---|---|---|---|---|---|
| `none` | midoffice04's `V3_nosnl` | same | same | same | same | differ: the MAD spread came after it |
| `use3` | midoffice04's `V3_use3` | same | same | same | same | differ, as above |
| default | HEAD (H0) | same | same | same | same | same |

Mean R-squared 0.293802 under "none", 0.299429 under "use3", 0.298076 at the default.

### The two figures

`CharacteristicPanel.standardise` counts the raw clip's moves (`winsorised`) and the outsiders
at a bound (`clipped_outside`) but not the members the +/-4 guard moves, which is the 3,043.
`stale_figures.py` copies its loop with that one counter added, and checks the copy against the
library on the run's own descriptors first: every value bit for bit, winsorised 369,070 against
369,070, clipped_outside 102,179 against 102,179. Then:

| | the doc said | the current run |
|---|---|---|
| member values the guard moves, over the 1,214 days | 3,043 | 4,021 |
| of them in LNCAP's lower tail | "nearly half" | 1,408, 35.0% |
| in any lower tail | -- | 4,012: VLVR 1,562, LNCAP 1,408, RSTR 522, HALPHA 503, LPRI 17 |
| upper tail | -- | 9, HILO's |
| outsider values at a bound | 100,011 | 102,179 |
| LNCAP's outsiders on the last day at a bound | 19 of 45, one at the raw bound too | 19 of 45, one at the raw bound too (SBNY) |

The doc's explanation -- a cap-weighted mean far above the median puts the bound inside the raw
one -- was checked rather than kept on trust: in all 2,622 date-descriptor cells where the
guard moved a member's lower-tail value, the cap-weighted mean is above the median. For LNCAP
and VLVR the median gap is 1.783 and 1.703 standard deviations (at least 1.501 and 1.482); for
RSTR and HALPHA 0.554 and 0.448; in 20 cells the standard deviation is also below the MAD scale.
The sentence now reads: "4,021 of the S&P 500 panel's member values over its 1,214 days, all
but nine in a lower tail where a cap-weighted mean above the median puts the bound inside the
raw one: three quarters of them LNCAP's and VLVR's, whose means sit about 1.7 standard
deviations above their medians, and most of the rest RSTR's and HALPHA's."

The doc also records three of the decisions: "none" beside "use3", the portfolio rows being
standing, and Size Nonlinearity staying (decision 1, with its arithmetic); orthogonality in the
regression's weights being the sense intended (decision 2); and the MAD settling the scale with
the weight of history left to bias statistics (decision 4).

### Exactness

H0 -- HEAD's tree, run as committed -- against the finished first change at the defaults:
factor returns, covariance, style exposures, forecast exposures and specific risks bit for bit,
every summary field equal, and six report lines added, the portfolio block. The first change's
midoffice tests pass on their own: in a checkout holding only its source and HEAD's tests with
its `test_descriptors.py`, 779 passed and 3 skipped. The two extra skips are
`config/midoffice_risk.toml` tests, that file not being in a scratch checkout.

---

## The identity gate

### The sources, cheapest first (item 2)

| Source | Asked | Answered | Network |
|---|---|---|---|
| the reference records cached by the panel fetch | 506 tickers | 2,885 records at 6 sample dates, every ticker with at least one | none |
| ticker events, by ticker: the 591-name union, OKTA (a member once membership() reads the list caught up, item 5) and SATS (the window's change table names it; membership() never returns it) | 593 | 555 found, 38 none -- delisted | 593 requests, 132 s |
| ticker events, by FIGI: every composite FIGI in the cached records that no ticker's answer carries | 41 | 4 found -- Roundhill (META to METV on 2022-01-31), Fortune Brands (FBHS to FBIN on 2022-12-15), The Gap (GPS to GAP on 2024-08-22), First Republic (FRC to FRCB to S600) -- and 37 none | 41 requests, 6 s |

The FIGI requests are the same endpoint, which takes "a Ticker, CUSIP, or Composite FIGI"; they
are the only way to its answer for a security whose old ticker no longer finds it. Every answer
is cached, found or not (`PolygonClient.ticker_events`), so the build script, run since with
`--offline`, makes none.

**What the events are worth, and what they are not.** The first event of a security is where
Polygon's record of it starts, not necessarily when it began trading under that ticker: Super
Micro's is its split of 2024-10-01, Arista's its split of 2024-12-04, Amcor's its reverse split
of 2026-01-15, TE Connectivity's its redomicile of 2024-09-30. Only a change from one ticker to
another is a known start. The lists can also be short or wrong, and where they contradict the
panel's records and prices -- the panel asked for each ticker across the whole window, so its
prices say who traded under it -- the panel's are taken. The ten disagreements:

| Security | The events say | The panel's records and prices say | Taken |
|---|---|---|---|
| Alphabet Class A | GOOG since 2015-10-06 | GOOGL throughout | GOOGL |
| Brown-Forman Class B | BF since 2003 | BF.B throughout | BF.B |
| Berkshire Class B | BRK since 2010 | BRK.B throughout | BRK.B |
| Agilent | AWD since 2005-11-23 | A throughout | A |
| Monolithic Power | MPWRE since 2005-11-28 | MPWR throughout | MPWR |
| News Corp Class A | NWSAV since 2013-06-20 | NWSA throughout | NWSA |
| Fiserv | FI from 2023-06-07, no return | FISV again from 2025-11-11 | FI to 2025-11-10 |
| Honeywell | HONI 2026-06-15 to 06-28 | HON's prices on every one of those days | HONI dropped |
| Signature Bank | S217 from 2023-11-18 | SBNY on 2023-11-20 and after | S217 dropped: a weekend |
| Casey's | CASYV from 2010-08-27, no return | CASY finds it today | CASY, presumed |

### The checks across each ticker's cached snapshots (item 3)

Between consecutive records of one ticker there are 38 changes (`survey_refs.txt`):

- **Name only, 23 -- the same security:** BEN, BXP, CCI, CI, COO, CPB, CRM, EXPD, FISV, GE, HIG,
  K, KHC, LUMN, MAR, NEE, PENN, RHI, ROP, RTX, SLB, TEL, WMT.
- **CIK only, the FIGI unchanged, 5 -- the same security:** CMS three times (811156 and 201533
  alternate), SBNY (its CIK goes missing), INFO's Harbor ETF (one appears).
- **FIGI, 6 -- a reorganisation:** BLK (2024; the CIK too), CCL (2026; the FIGI goes missing), J
  (2022), LH (2024), OKE (2026), XOM (2026; the CIK too).
- **Unrelated securities, 4 -- a splice:** BNY (FUND to CS), DOC, INFO (CS to ETF; IHS Markit's
  record has no FIGI), META (ETF to CS).

A list date after the ticker's first price: 3 records, all INFO's Harbor ETF (list dates
2024-10-10, 2026-01-02 and 2024-10-09 against a first price of 2022-01-03). 138 records carry
no composite FIGI -- mostly companies domiciled abroad (ACN, LIN, MDT, ETN and others) -- which
is why 32 of the table's securities are keyed by CIK. No record lacks both.

### The table (item 1)

`data/external/polygon/sp500_identity_2022-01-03_2026-11-03.csv`: one row per security per
ticker per span, columns `key, figi, cik, name, type, ticker, first, last, held, source, starts,
ends` -- `held` the dates of the span the panel prices under the ticker, `source` "panel",
"events" or "presumed", and `starts` and `ends` the evidence for each boundary. 625 rows over
598 securities: 518 rows of the panel's own prices, 106 from events alone -- the old tickers of
renamed members, and the members the panel lacks -- and 1 presumed (Casey's, as CASY). Of the
598, 513 are common stock, 2 ETFs and 1 a fund; the other 82 have no record, only events, and
so no type.

How a ticker's prices are assigned (`identity.segment_ticker`): consecutive records naming one
security anchor it; between two anchors naming different securities, the change is dated by
the later one's events (the day it began under the ticker), the earlier one's (the day it left),
or a gap of 5 or more trading days in the prices, and failing all three the dates between are
left unassigned and reported. Before the first record and after the last, the prices go to that
record's security as far as they run unbroken and its events allow. On this panel every change
was dated -- by events, all ten -- and no priced cell is unassigned.

### The guards (item 4)

| Guard | Checked | Failures |
|---|---|---|
| every priced (ticker, date) maps to exactly one security | 583,628 cells | 0 |
| no security under two tickers on one date | 625 rows | 0 |
| no ticker names two securities on one date (the table's own consistency) | 625 rows | 0 |
| every member membership() returns, on every date of the window, maps to a security or is named | 611,899 member-days | 0 |

Each is exact -- a count, not a tolerance -- and the script exits non-zero if any fails. The
tests show each failing on a table built to fail it. The 611,899 member-days:

| | member-days |
|---|---|
| mapped: the panel's prices | 550,800 |
| mapped: events | 51,263 |
| mapped: a reorganisation's predecessor (the six) | 4,843 |
| mapped: presumed (CASY) | 145 |
| mapped: after the security's last price, before the index removed it | 31 |
| mapped: inside a gap in its own prices (FRC 2, SBNY 2) | 4 |
| named: unverified (SW 629, CPAY 558, SKYD 901, SMCI 136, BG 160) | 2,384 |
| named: missing (DAY 506, KVUE 798, WBD 1,125) | 2,429 |

"After its last price" is ATVI, SIVB, TWTR, XLNX and ten others, which stopped trading days
before the index removed them: the same security, and no price, which is right.

### Splices, reorganisations, truncations, phantoms, duplicates (item 6)

**Splices** -- one ticker, two unrelated securities:

| Ticker | Before | To | After | From | Dated by |
|---|---|---|---|---|---|
| BNY | BlackRock New York Municipal Income Trust (fund) | 2026-02-06 | Bank of New York Mellon | 2026-05-21 | BNY Mellon's change from BK |
| DOC | Physicians Realty Trust | 2024-02-29 | Healthpeak Properties | 2024-03-04 | Healthpeak's change from PEAK; no gap in the prices |
| INFO | IHS Markit | 2022-02-25 | Harbor PanAgora Dynamic Large Cap Core ETF | 2024-10-10 | the ETF's first event |
| META | Roundhill Ball Metaverse ETF | 2022-01-28 | Meta Platforms | 2022-06-09 | Meta's change from FB, and Roundhill's to METV on 2022-01-31 |

Member-days on which the panel's price under the member's ticker is the other security's: BNY
1,028 (the fund, 2022-01-03 to 2026-02-06), DOC 542 (Physicians Realty, 2022-01-03 to
2024-02-29), META 19 (Roundhill, 2022-01-03 to 2022-01-28). INFO's ETF prices start after IHS
Markit left the index, so none of INFO's member-days carries them.

**Reorganisations** -- a new identifier under the same ticker, the same type, the prices
unbroken: BLK (BBG000C2PW58 to 2024-10-01, BBG01PSW2WN4 from 2024-10-02), CCL (BBG000BF6LY3 to
2026-05-06, then Carnival Corporation Ltd., CIK 0000815097, no FIGI, from 2026-05-07), J (to
2022-08-26, from 2022-08-29), LH (to 2024-05-17, from 2024-05-20), OKE (to 2026-09-10, from
2026-09-11), XOM (BBG000GZQ728 to 2026-07-01, ExxonMobil Holdings BBG023CY9MM1 from 2026-07-02).

**Truncations** -- member-days on which the member traded under another ticker than the panel's,
which therefore has no price for it, ECHO's phantom days left out:

| Ticker | As | Member-days | Dates |
|---|---|---|---|
| BALL | BLL | 88 | 2022-01-03 to 2022-05-09 |
| BNY | BK | 1,099 | 2022-01-03 to 2026-05-20 |
| COR | ABC | 416 | 2022-01-03 to 2023-08-29 |
| DOC | PEAK | 543 | 2022-01-03 to 2024-03-01 |
| ECHO | SATS | 64 | 2026-03-23 to 2026-06-23 |
| EG | RE | 379 | 2022-01-03 to 2023-07-07 |
| ELV | ANTM | 121 | 2022-01-03 to 2022-06-27 |
| FBHS | FBIN | 2 | 2022-12-15 to 2022-12-16 |
| FISV | FI | 610 | 2023-06-07 to 2025-11-10 |
| FRC | FRCB | 1 | 2023-05-03 |
| GEN | NLOK | 214 | 2022-01-03 to 2022-11-07 |
| META | FB | 109 | 2022-01-03 to 2022-06-08 |
| MRSH | MMC | 1,011 | 2022-01-03 to 2026-01-13 |
| RVTY | PKI | 343 | 2022-01-03 to 2023-05-15 |
| SKYD | PSKY | 292 | 2025-08-07 to 2026-10-05 |
| VMRK | EQR | 1,159 | 2022-01-03 to 2026-08-17 |

6,451 member-days in all. Beyond them, the 86 member tickers the panel does not hold at all
carry 46,625 member-days. CPAY and SW are not in the table above: their panel prices begin on
2024-03-25 and 2024-07-08, the first of their securities' events, and neither source says what
they traded as before -- unverified, below.

**Phantoms** -- member-days membership() gives that the change table, read in today's tickers
through the identity table's renames, does not: ECHO alone, 1,057 member-days, 2022-01-03 to
2026-03-20. EchoStar traded as SATS then and joined on 2026-03-23.

**Duplicates** -- one security named by two member tickers on one date: none. GOOG and GOOGL,
FOX and FOXA, NWS and NWSA, DISCA and DISCK, UA and UAA are each two FIGIs of one CIK -- two
securities, as they should be.

### membership()'s three inconsistencies (item 5)

- **KVUE kept after its removal, OKTA never added -- fixed at the source.** The change table has
  OKTA replacing KVUE on 2026-10-30; the constituent list still carries KVUE and not OKTA.
  `_caught_up` brings the list up to the table where the table is ahead of it: a change the list
  contradicts in every ticker -- each addition absent, each removal present -- has not reached
  it, and is applied to it; the newest such changes only, stopping at the first the list
  reflects at all. A change the list contradicts in only some tickers -- SATS -- is a rename or a
  mis-recording, not lag, and is left; so is one that only adds or only removes. Each one applied
  is a `current_stale` warning in validate(). Now OKTA is a member from 2026-10-30 and KVUE to
  2026-10-29; the walk's 29 inconsistency warnings become 27, beside the one `current_stale`.
- **ECHO a member back to 1976 -- the change table alone cannot support it.** It records the
  addition of 2026-03-23 as SATS and has no row naming ECHO, and nothing in it links the two.
  So `Rename` carries a dated ticker change from outside the table -- here, from EchoStar's
  events: SATS from 2008-01-02, ECHO from 2026-06-24 -- and `membership(..., renames=...)` reads
  the table in today's tickers before walking it: the 2026-03-23 row's SATS is ECHO, and ECHO is
  a member from that date. `since` keeps a rename off rows naming the old ticker before the
  security held it. Of the table's 62 ticker changes for securities that are members in the
  window, 51 are read through; 11 are not, each with why -- WLTW to WTW, which the table already
  records as a same-day swap; FBHS to FBIN, GPS to GAP and FRC's two, where the security left
  the index under its old name; and six more whose chain ends off today's list -- event
  artifacts the panel overrules (A to AWD, MPWR to MPWRE, NWSA to NWSAV) and removed members'
  (Under Armour's two, LKQ's). Read through, the
  table's inconsistencies fall from 27 to 24: FB (2013), RE (2017) and SATS (2026) resolve, and
  none is added.
- The run does not pass renames: the brief said no model change, and ECHO has no price before
  2026-06-24, so no regression changes either way. The rebuild can pass them.

### The identity of each member ticker (item 6)

592 tickers are members on some date of the window, 593 runs of membership (ILMN has two: to
2024-06-21, and from 2026-09-21 again). 591 is the union as membership() gave it; OKTA is the
592nd once the list is read caught up. The security each run names, by source:

| Identified by | Runs |
|---|---|
| the records and the events | 476 |
| the events alone (members the panel does not hold) | 82 |
| the records alone (delisted, so no events): ABMD, ANSS, ATVI, AVB, CDAY, CERN, CMA, CTLT, CTRA, CTXS, DFS, DISCA, DISCK, DISH, DRE, EA, HES, HOLX, INFO, IPG, JNPR, K, MRO, NLSN, PBCT, PXD, QRVO, SEE, SIVB, TWTR, WBA, XLNX | 32 |
| neither: DAY, KVUE, WBD | 3 |

The full listing -- each run's dates, the security's key, name, FIGI and CIK, the tickers it
traded under and how each date maps -- is in the appendix, and in
`sp500_identity_2022-01-03_2026-11-03_members.csv`.

**Named member-days, each with why:**

- **Missing -- no security in either source.** DAY (506 member-days, 2024-02-01 to 2026-02-06):
  no events, not in the panel; the change table removes CDAY and adds DAY on 2024-02-01, and
  Ceridian's prices under CDAY end on 2024-01-31, so CDAY's security is the candidate -- a
  rename recorded as a swap -- but Dayforce's FIGI also has no events, and nothing confirms it.
  KVUE (798, 2023-08-25 to 2026-10-29): none, and no candidate. WBD (1,125, 2022-04-11 to
  2026-10-05): none; DISCA and DISCK, removed the same day with their prices ending on
  2022-04-08, are candidates.
- **Unverified -- the security at the run's end is known, but not on these dates.** SW (629,
  2022-01-03 to 2024-07-05): the panel's SW prices begin on 2024-07-08, Smurfit Westrock's first
  event; before then the member traded under another ticker or was another security, and
  neither source says which -- by public record WestRock, WRK, which the change table never
  names. CPAY (558, to 2024-03-22): Corpay's first event and first price are 2024-03-25; by
  public record FleetCor, FLT, which the table adds in 2018 and never removes or renames.
  SKYD (901, to 2025-08-06): before Skydance's first event, PSKY on 2025-08-07; by public
  record Paramount Global. SMCI (136, 2024-03-18 to 2024-09-30): before Super Micro's first
  event, its split; almost certainly SMCI itself. BG (160, 2023-03-15 to 2023-10-31): before
  Bunge Global SA's first event, its redomicile; by public record Bunge Limited.

The public-record notes are not evidence from either source and are not in the table.

### The fetch the rebuild needs (item 6)

`identity.fetch_plan`, written to `sp500_identity_2022-01-03_2026-11-03_fetch.csv`: for each of
the 594 securities that are members on some date of the window (the 588 that member tickers
name, and the six predecessors), each of its rows within the window, the ticker it traded under
there, and share-count samples inside the security's own spans only -- each span's first and
last day, so no count is carried across a change of identity or ticker, and every 63 trading
days across the security's first-to-last span -- each under the ticker of its day.

| | Spans | Tickers | Member-days | Share samples (not on disk) |
|---|---|---|---|---|
| the panel's tickers, cut to each security's spans | 514 | 506 | 556,582 | 10,113 (7,252) |
| tickers of the members the panel lacks | 83 | 83 | 43,900 | 1,507 (1,507) |
| old tickers of the members the panel truncates | 16 | 16 | 7,508 | 162 (162) |
| other tickers of member securities, outside their membership | 7 | 7 | 0 | 80 (80) |

The 7,508 includes ECHO's 1,057 phantom
member-days as SATS. The 106 new tickers: the 83 members' (ABNB through XYZ, in the breakdown in
the appendix), the 16 old ones -- ABC, ANTM, BK, BLL, EQR, FB, FBIN, FI, FRCB, MMC, NLOK, PEAK,
PKI, PSKY, RE, SATS -- and CEGV, CHK, CRHC, GAP, PSTG, S600 and SQ.

| Requests | |
|---|---|
| prices, adjusted and not, for the 106 new tickers over the window (the panel's 506 are cached) | 212 |
| share-count samples not on disk | 9,001 |
| split events, one a ticker, for the share-count rule midoffice03 recorded | 612 |
| in all | 9,825 |

At the original fetch's rate, about five requests a second, that is about 33 minutes. A
security has between 2 and 25 share samples, 21 at the median. The first sample of every span is
also an identity check: its reference record must name the security the table expects. That
matters for CRHC, which CRH's events give for 2021-09-23 to 2023-09-24 and which may well be
another company's ticker. Not in the plan: the 4,813 named member-days, under BG, CPAY, DAY,
KVUE, SKYD, SMCI, SW and WBD, which need a source this one is not.

### The model, unchanged

From the finished tree, at the defaults, against the first change's: factor returns,
covariance, style and forecast exposures and specific risks bit for bit, every summary field
equal. Four report lines differ, all in the run's note on changes membership() does not
reproduce: there were 3 (SATS, OKTA, KVUE) and there is 1 (SATS), OKTA and KVUE now being
reproduced. Neither is in the panel, so no cross-section changed.

---

## Key decisions

1. **A security is a composite FIGI**, as the brief has it, so the six reorganisations are new
   securities, each the old one's successor; the member-map carries the index across them
   ("predecessor"). Where no FIGI exists the key is the CIK; no CIK-keyed pair collided.
2. **The panel's records and prices outrank the events.** The panel asked for each ticker
   across the window, so its prices are evidence of who traded under it, and its records name
   them. Where the events disagree they are cut, and the cut reported -- ten disagreements.
3. **A security's first event is not a known start.** Members before it are named
   "unverified", not mapped by presumption: for SMCI that is too cautious, for SKYD and BG it is
   right, and the sources cannot tell them apart.
4. **"Missing" and "unverified" are both named, never guessed.** The candidates from same-day
   swaps (DAY from CDAY, WBD from DISCA or DISCK) are reported as candidates only.
5. **Item 5's ECHO fix is at the source but needs the identity table**, because the change
   table cannot support it alone; the brief's wording allowed reporting why it cannot, and I did
   both -- reported why, and built the dated renames that fix it given outside evidence.
6. **The KVUE/OKTA fix is on by default; renames are opt-in.** The roll-forward needs nothing
   but the snapshot and changes no member of the panel; the renames change membership for
   ECHO, which the run should take deliberately, in the rebuild.
7. **The renames read through are filtered** (`applicable_renames`): only those whose security
   trades under a ticker on today's list, and not those the table already records as a swap.
   Without the filter, three renames introduced new inconsistencies (FBIN, S600, WTW).
8. **Two git adds with disjoint files.** The identity gate's documentation is the module and
   script docstrings; `docs/FACTOR_MODEL.md` belongs to the first change, so describing the gate
   there waits for the rebuild.
9. **The FIGI requests** are the same endpoint the brief named, asked for the 41 cached
   securities no ticker answer covered; they resolved four.
10. **Halts:** a gap of fewer than 5 trading days neither dates a change nor breaks a run of
    prices.

## Decisions left for Jeff

1. **Reorganisations:** the six are new FIGIs and the index carries each on. Should the rebuild
   treat each as one security with a continuous history (one return series, the share count
   carried 1:1), or as two with a link?
2. **The 4,813 named member-days.** Resolving them needs another source. The cheapest is the
   reference endpoint at a few dates -- CDAY's successor DAY, WRK, FLT, PARA, SMCI and BG -- about
   ten requests. Is that endpoint authorised for the rebuild?
3. **Renames in the run:** reading membership through the table's renames removes ECHO's 1,057
   phantom member-days. It changes no regression on this panel, but it changes the index's own
   size on those days. Take it in the rebuild?
4. **The fetch plan's scope:** every member security across the whole window, as the panel
   does, ex-members included, with quarterly share samples (9,825 requests). A plan cut to each
   membership span plus a year's look-back would cost less. Which?
5. **Casey's:** CASY is presumed for its 145 member-days, its events ending with CASYV in 2010.
   The rebuild's first share sample checks it.

---

## Tests

2,253 passed, 1 skipped, 19 warnings, in 71.88 s -- the 19 warnings are in
`test_backtest_validation.py` and `test_microstructure_utils.py`, untouched. midoffice alone: 802
passed, 1 skipped. Before: 2,231 and 780. 22 new tests:

**The first change, 1:**

- `test_size_nonlinearity_none_leaves_it_out_and_every_other_index_as_it_was` -- "none" builds
  no Size Nonlinearity, asked for or not; every other index is bit for bit what "use4" and
  "use3" build; `index_descriptors` and `blend_composition` read it as built from nothing.

**The identity gate, 21:**

- `test_prices.py`, 3: `test_ticker_events_come_back_parsed_and_are_asked_for_once`;
  `test_no_events_for_a_delisted_security_is_an_answer_and_is_kept` (None, cached, no key in
  the file); `test_a_404_is_not_found_with_the_servers_reason_and_is_not_retried`.
- `test_sp500.py`, 4: `test_a_constituent_list_behind_its_changes_table_is_read_brought_up_to_it`
  (KVUE and OKTA); `test_a_change_the_list_contradicts_in_only_some_tickers_is_left_to_warn`
  (SATS, and one-sided changes); `test_renames_read_the_changes_table_in_todays_tickers` (ECHO);
  `test_a_rename_reaches_only_the_rows_inside_the_old_tickers_tenure_and_chains` (`since`,
  FISV to FI to FISV).
- `test_identity.py`, 14, one per rule: events as tenures; one security throughout; a take-over
  with no gap (DOC); dating by a gap, or leaving unassigned; a reorganisation's predecessor
  (XOM); a member truncated before its rename (BALL); events cut by the panel and a presumed
  ticker (GOOGL, CASY); a member named, never guessed (DAY, SW); after the last price and inside
  a gap (ATVI, SBNY); each guard failing on a table built to fail it; renames filtered (WLTW,
  GPS); phantoms in today's names (ECHO against GPS); the fetch plan's samples inside each span;
  the cache readers.

Five of the identity tests were checked by mutation, four mutations in all -- events never cut,
arrivals never dated (two tests), no gap ever long enough, the plan without span endpoints --
and each fails with its rule broken.
Lint (ruff): no new findings in any changed or new file; the 11 left are HEAD's, and two of
HEAD's are gone because the code now uses the imports. Every changed file is ASCII and parses.

---

## Failures and debugging trail

1. **The first draft of `identity.py`** worked out the rules but was tangled -- a redundant
   arrival computation before the first record, evidence strings built inline. Rewritten before
   any test.
2. **The first real run named delisting tails "missing".** ATVI and thirteen others stopped
   trading days before the index removed them; those are the same security after its last
   price, now mapped so (31 member-days).
3. **CASY came out "missing: took the ticker on 2003-09-10, after the membership ended".** Its
   events end with CASYV from 2010; the check read that as a later holder. A ticker that finds
   a security today is evidence it holds it now, so CASY is presumed instead.
4. **FRC's two days were "unverified: the panel's FRC prices begin on 2022-01-03"** -- true and
   beside the point. They lie between its last price as FRC and its first as FRCB: inside a gap
   in its own prices. **SKYD's reason** named the panel's prices rather than the first of
   Skydance's events; the branch order was changed.
5. **Phantoms compared by ticker string** found GPS and SBNY phantom, with GAP and S217 "the
   other way": the renamed walk calls a security renamed after it left the index by its new
   name. Compared in today's names (`identity.phantoms`).
6. **Reading every rename through added three inconsistencies** (FBIN, S600, WTW) while
   resolving three: FBHS was removed under its old name four days after its rename, FRC after it
   failed, and WLTW's rename is already a swap in the table. `applicable_renames` filters them.
7. **Signature Bank's S217** was a two-day weekend tenure, cut by the panel's SBNY prices to
   nothing; such a tenure is dropped and the one before runs on.
8. **Shell heredocs** broke three edits -- an unterminated f-string, a `sed` turning `\n` into
   newlines, and a continuation backslash eaten into one long line (an E501). Each edit was
   redone with a script file or the editor.
9. **`PolygonNotFound`** drew a new N818; it is `PolygonNotFoundError`, like `CoverageError`.
10. **`sed -i` rewrote two files' line endings to LF**; normalised back to CRLF, as the
    checkout has them, so `git add` raises no warning.
11. **Two of my sp500 test expectations were wrong**, not the code: a date before the toy
    snapshot's coverage, and a member left out of an expected tuple.
12. **`rm -rf` of a scratch tree was refused** by the sandbox; the snapshot went to a new
    directory instead.
13. **`/tmp` was written once**, a file list in the last dry-run command, against the rule that
    scratch goes to the scratchpad; deleted at once, and the dry runs' output read from the
    terminal instead.

## Problems found, not fixed

- **DAY, KVUE, WBD**: no identity in either source, 2,429 member-days. **SW, CPAY, SKYD, SMCI,
  BG** before their securities' first events, 2,384 member-days. Decision 2.
- **The run still reads membership as given**, so ECHO is a member from 2022 (no prices then).
  Decision 3.
- **The run does not print sp500's `current_stale` warning**, so the roll-forward is visible only
  through validate() (`python -m praxis.midoffice.refdata.sp500 check`) and the identity
  script's report. The rebuild touches the run anyway.
- **The events' quality**: first events that are splits or redomiciles, share classes under
  their root ticker, GOOGL under GOOG, lists that stop short -- Fiserv's return to FISV,
  Casey's to CASY, Coherent's II-VI years, Corpay's FleetCor years. Each that the panel could
  check was checked; Coherent and Corpay could not be.
- **CRHC** in CRH's events may be another company's ticker; the plan's first share sample checks
  it.
- **24 inconsistencies remain in the change table** read through the renames, all but SATS's
  before the window: FLT, UA.C, FOX (2015), KORS, DLPH, PCLN, HRS, COG, JEC and others added and
  never removed under those names, and AGN, T, AN, SUN and MCK removed while still held.
- **Share counts carried across identities**, the BNY fund's 24.1 million on BNY Mellon and the
  INFO ETF's on IHS Markit's dates, are still in the panel; the plan's per-span samples are the
  fix, in the rebuild. So are midoffice03's 621 split-stale member-days.
- **`docs/FACTOR_MODEL.md` does not describe the identity gate yet** (decision 8 above).

## Scratch

`claude/scratch/midoffice05/` (gitignored), 69 items: every script and output named here --
`common.py`, `runv.py` and `exact.py` (midoffice04's, adapted), `stale_figures.py`,
`survey_refs.py`, `survey_events.py`, `fetch_events.py`, `try_identity.py`,
`plan_breakdown.py`, `extract_brief.py`, `write_brief.py`, `build_retro.py` -- and their
outputs, the variants' reports and JSON, and `out_final/`, a copy of the four table files. Not
copied: the tree snapshots and the variant pickles, which stay in the session scratchpad.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. Both `git add` lists below were checked
with `git add --dry-run`, and every changed path is in exactly one of them.

## Open items for Chat

- The rebuild's shape, now that the identity is settled: which of decisions 1-4.
- Whether any second source -- the reference endpoint at a handful of dates -- is in scope for
  the rebuild, for the 4,813 named member-days.
- The noise-share question decision 1 deferred, model-wide, after the rebuild.

---

## Appendix: outputs, verbatim and generated

Every output named above, from `claude/scratch/midoffice05/` and the table's own files in `data/external/polygon/`. The listings are generated from those files.

### build_report.txt -- the build script's report, finished tree, --offline

```
Panel
  data\external\polygon\sp500_daily_2022-01-03_2026-11-04.npz: 506 tickers over 1214 trading days, 2022-01-03 .. 2026-11-03; 583,628 priced (ticker, date) cells

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC
  the constituent list predates the change of 2026-10-30 -- OKTA added, KVUE removed -- and is read with it applied
  592 tickers are members on some date of the window; 86 of them are not among the panel's 506; the change table also names, inside the window, SATS that membership() never returns

Sources
  reference records cached by the panel fetch: 2,885 over 506 of the 506 tickers, at 6 sample dates -- no requests
  ticker events, by ticker: 593 asked -- the window's member tickers and 1 more -- 555 found, 38 with none (delisted); 0 asked over the network this run, the rest read from the cache
  ticker events, by FIGI, for the cached records' securities no ticker found: 41 asked, 4 found; 0 over the network this run

The table
  625 rows: 598 securities, each under one ticker over one span of the window; 518 rows priced by the panel, 106 from events alone, 1 presumed
  identified by the records and the events 476, by the records alone 40, by the events alone 82

Guards
  ok  every priced (ticker, date) maps to exactly one security: 583,628 checked, 0 failure(s)
  ok  no security under two tickers on one date: 625 checked, 0 failure(s)
  ok  no ticker names two securities on one date: 625 checked, 0 failure(s)
  ok  every member on every date maps to a security or is named: 611,899 checked, 0 failure(s)

Splices: one ticker, two unrelated securities
  BNY    BlackRock New York Municipal Income Trust [BBG000BZF6G2] to 2026-02-06; Bank of New York Mellon Corporation [BBG000BD8PN9] from 2026-05-21
         Bank of New York Mellon Corporation began trading as BNY on 2026-05-21 (events: a change from another ticker)
  DOC    PHYSICIANS REALTY TRUST [BBG004MF59H0] to 2024-02-29; Healthpeak Properties, Inc. [BBG000BKYDP9] from 2024-03-04
         Healthpeak Properties, Inc. began trading as DOC on 2024-03-04 (events: a change from another ticker)
  INFO   IHS Markit Ltd. Common Shares [CIK0001316360] to 2022-02-25; Harbor PanAgora Dynamic Large Cap Core ETF [BBG01Q2P8F92] from 2024-10-10
         Harbor PanAgora Dynamic Large Cap Core ETF began trading as INFO on 2024-10-10 (events: the first of its record)
  META   Roundhill Ball Metaverse ETF [BBG011J1MN62] to 2022-01-28; Meta Platforms, Inc. Class A Common Stock [BBG000MM2P62] from 2022-06-09
         Meta Platforms, Inc. Class A Common Stock began trading as META on 2022-06-09 (events: a change from another ticker); Roundhill Ball Metaverse ETF traded as another ticker from 2022-01-31 (events)

Reorganisations: a new identifier under the same ticker, the prices unbroken
  BLK    BBG000C2PW58 to 2024-10-01, BBG01PSW2WN4 from 2024-10-02: Blackrock, Inc. -> Blackrock, Inc.
  CCL    BBG000BF6LY3 to 2026-05-06, CIK0000815097 from 2026-05-07: Carnival Corporation -> Carnival Corporation Ltd.
  J      BBG000BMFFQ0 to 2022-08-26, BBG019C1BQR4 from 2022-08-29: Jacobs Engineering Group -> Jacobs Solutions Inc.
  LH     BBG000D9DMK0 to 2024-05-17, BBG01MMT6PL7 from 2024-05-20: Laboratory Corporation of America Holdings -> Labcorp Holdings Inc.
  OKE    BBG000BQHGR6 to 2026-09-10, BBG024TZWVN1 from 2026-09-11: Oneok, Inc. -> Oneok, Inc.
  XOM    BBG000GZQ728 to 2026-07-01, BBG023CY9MM1 from 2026-07-02: Exxon Mobil Corporation -> ExxonMobil Holdings Corporation

Priced dates no evidence assigns: 0

Where the events and the panel's records and prices disagree, the panel's taken: 11
  Agilent Technologies Inc. [BBG000C2V3D6]: its events have it trading as AWD from 2005-11-23, but the panel holds its prices under A from 2022-01-03; the tenure is taken to end on 2022-01-02
  Brown-Forman Corporation Class B [BBG000BD2NY8]: its events have it trading as BF from 2003-09-10, but the panel holds its prices under BF.B from 2022-01-03; the tenure is taken to end on 2022-01-02
  BERKSHIRE HATHAWAY Class B [BBG000DWG505]: its events have it trading as BRK from 2010-01-22, but the panel holds its prices under BRK.B from 2022-01-03; the tenure is taken to end on 2022-01-02
  Fiserv, Inc. Common Stock [BBG000BJKPG0]: its events have it trading as FI from 2023-06-07, but the panel holds its prices under FISV from 2025-11-11; the tenure is taken to end on 2025-11-10
  Alphabet Inc. Class A Common Stock [BBG009S39JX6]: its events have it trading as GOOG from 2015-10-06, but the panel holds its prices under GOOGL from 2022-01-03; the tenure is taken to end on 2022-01-02
  Honeywell International, Inc. [BBG000H556T9]: its events have it trading as HONI from 2026-06-15 to 2026-06-28, but the panel holds its prices under HON from 2026-06-15; the tenure is dropped
  Monolithic Power Systems, Inc. [BBG000C30L48]: its events have it trading as MPWRE from 2005-11-28, but the panel holds its prices under MPWR from 2022-01-03; the tenure is taken to end on 2022-01-02
  News Corporation Class A Common Stock [BBG0035LY913]: its events have it trading as NWSAV from 2013-06-20, but the panel holds its prices under NWSA from 2022-01-03; the tenure is taken to end on 2022-01-02
  SIGNATURE BANK (NY) [BBG000M6TR37]: its events have it trading as S217 from 2023-11-18, but the panel holds its prices under SBNY from 2023-11-20; the tenure is taken to end on 2023-11-19
  SIGNATURE BANK (NY) [BBG000M6TR37]: its events have it trading as S217 from 2023-11-18 to 2023-11-19, which holds no trading day; the tenure is dropped
  Casey's General Stores Inc [BBG000BF0672]: its events end with CASYV from 2010-08-27, yet CASY finds it today; when it began trading as CASY again is in neither source, and it is presumed CASY from 2010-08-27

Member dates on which the panel's price under the member's ticker is another security's
  BNY     1028 day(s), BlackRock New York Municipal Income Trust [BBG000BZF6G2]: 2022-01-03..2026-02-06 (1028)
  DOC      542 day(s), PHYSICIANS REALTY TRUST [BBG004MF59H0]: 2022-01-03..2024-02-29 (542)
  META      19 day(s), Roundhill Ball Metaverse ETF [BBG011J1MN62]: 2022-01-03..2022-01-28 (19)

Phantoms: member dates membership() gives that the change table, read in today's tickers through the table's renames, does not
  the members' ticker changes in the table: 62; read through: 51; not: 11
    not A -> AWD (2005-11-23): AWD is not in today's list
    not MPWR -> MPWRE (2005-11-28): MPWRE is not in today's list
    not UARM -> UA (2006-12-18): UAA is not in today's list
    not LKQX -> LKQ (2012-05-23): LKQ is not in today's list
    not NWSA -> NWSAV (2013-06-20): NWSAV is not in today's list
    not UA -> UAA (2016-12-07): UAA is not in today's list
    not WLTW -> WTW (2022-01-10): the changes table records it as a swap on 2022-01-10
    not FBHS -> FBIN (2022-12-15): FBIN is not in today's list
    not FRC -> FRCB (2023-05-03): S600 is not in today's list
    not FRCB -> S600 (2023-11-18): S600 is not in today's list
    not GPS -> GAP (2024-08-22): GAP is not in today's list
  ECHO    1057 day(s): 2022-01-03..2026-03-20 (1057) -- BBG000TGLV00, trading as SATS
  the change table's inconsistencies, read through the renames: 27 -> 24
    resolved: FB added absent, 2013-12-23
    resolved: RE added absent, 2017-06-19
    resolved: SATS added absent, 2026-03-23

Truncations: member dates on which the member traded under another ticker than the panel's, which has no price for it (phantom dates left out)
  BALL   as BLL       88 day(s): 2022-01-03..2022-05-09 (88)
  BNY    as BK      1099 day(s): 2022-01-03..2026-05-20 (1099)
  COR    as ABC      416 day(s): 2022-01-03..2023-08-29 (416)
  DOC    as PEAK     543 day(s): 2022-01-03..2024-03-01 (543)
  ECHO   as SATS      64 day(s): 2026-03-23..2026-06-23 (64)
  EG     as RE       379 day(s): 2022-01-03..2023-07-07 (379)
  ELV    as ANTM     121 day(s): 2022-01-03..2022-06-27 (121)
  FBHS   as FBIN       2 day(s): 2022-12-15..2022-12-16 (2)
  FISV   as FI       610 day(s): 2023-06-07..2025-11-10 (610)
  FRC    as FRCB       1 day(s): 2023-05-03..2023-05-03 (1)
  GEN    as NLOK     214 day(s): 2022-01-03..2022-11-07 (214)
  META   as FB       109 day(s): 2022-01-03..2022-06-08 (109)
  MRSH   as MMC     1011 day(s): 2022-01-03..2026-01-13 (1011)
  RVTY   as PKI      343 day(s): 2022-01-03..2023-05-15 (343)
  SKYD   as PSKY     292 day(s): 2025-08-07..2026-10-05 (292)
  VMRK   as EQR     1159 day(s): 2022-01-03..2026-08-17 (1159)
  and 86 member tickers the panel does not hold at all, 46,625 member-days

Duplicates: one security named by two member tickers on one date: 0

Member-days no security accounts for: 4,813 -- named, each with why
  BG       160 day(s) 2023-03-15..2023-10-31 (160)
         unverified: before 2023-11-01, the first of Polygon's events for Bunge Global SA, the start of its record there; the member may be it, under BG or another ticker, or another security
  CPAY     558 day(s) 2022-01-03..2024-03-22 (558)
         unverified: the panel's CPAY prices begin on 2024-03-25; before then Corpay, Inc. traded under another ticker or the member was another security, and neither source says which
  DAY      506 day(s) 2024-02-01..2026-02-06 (506)
         missing: Polygon has no events for the ticker and the panel does not hold it -- delisted, and not among the panel's securities
         candidate: CDAY, Ceridian HCM Holding [BBG005D7PF34], removed the same day and priced to 2024-01-31 -- a rename recorded as a swap?
  KVUE     798 day(s) 2023-08-25..2026-10-29 (798)
         missing: Polygon has no events for the ticker and the panel does not hold it -- delisted, and not among the panel's securities
         the change table removes AAP the same day, none of them gone from the prices then
  SKYD     901 day(s) 2022-01-03..2025-08-06 (901)
         unverified: before 2025-08-07, the first of Polygon's events for Skydance Corporation, the start of its record there; the member may be it, under PSKY or another ticker, or another security
  SMCI     136 day(s) 2024-03-18..2024-09-30 (136)
         unverified: before 2024-10-01, the first of Polygon's events for Super Micro Computer, Inc. Common Stock, the start of its record there; the member may be it, under SMCI or another ticker, or another security
  SW       629 day(s) 2022-01-03..2024-07-05 (629)
         unverified: the panel's SW prices begin on 2024-07-08; before then Smurfit Westrock plc traded under another ticker or the member was another security, and neither source says which
  WBD     1125 day(s) 2022-04-11..2026-10-05 (1125)
         missing: Polygon has no events for the ticker and the panel does not hold it -- delisted, and not among the panel's securities
         candidate: DISCA, Discovery, Inc. Series A Common Stock [BBG00HW3TQG4], removed the same day and priced to 2022-04-08 -- a rename recorded as a swap?
         candidate: DISCK, Discovery, Inc. Series C Common Stock [CIK0001024333], removed the same day and priced to 2022-04-08 -- a rename recorded as a swap?

The 592 member tickers: 593 runs of membership; the security each names identified by records+events 476, events 82, records 32, neither 3

The fetch the rebuild needs, for the 594 securities that are members on some date of the window
  620 spans under 612 tickers; 106 of the tickers are not the panel's: ABC ABNB ACGL AMTM ANTM APO APP ARES AXON BE BG BK BLDR BLL BX CASY CEG CEGV CHK CIEN COHR COIN CPT CRH CRHC CRWD CSGP CVNA DASH DDOG DECK DELL EME EQR EQT ERIE EXE FB FBIN FDXF FERG FI FICO FIX FLEX FRCB FSLR GAP GDDY GEHC GEV HONA HOOD HUBB IBKR INVH JBL KDP KKR LII LITE LULU MBC MMC MOH MRVL NDSN NLOK OKTA ON P PANW PCG PEAK PKI PLTR PODD PSKY PSTG Q RDDT RE S600 SATS SMCI SNDK SOLS SOLV SQ STLD TKO TPL TRGP TTD TWLO UBER VEEV VICI VLTO VRT VST VYLR WDAY WSM WTW XYZ
  prices: 2 requests a ticker, adjusted and not, over the window -- 212 new, the panel's 506 cached (cut to each security's spans)
  share counts: 11,862 samples inside the securities' own spans (each span's first and last day and every 63 trading days), 9,001 not on disk
  split events, for the share-count rule recorded in midoffice03: one request a ticker, 612
  in all 9,825 requests
  not in it: the member-days named above, under BG, CPAY, DAY, KVUE, SKYD, SMCI, SW, WBD, which need a source this one is not

Wrote
  C:\Users\jmcphail\AppData\Local\Temp\claude\C--Data-Development-Python-McTheoryApps-praxis\a31f9dd1-7758-4ce2-83d1-9f3c02a3b2e5\scratchpad\m05\out_final\sp500_identity_2022-01-03_2026-11-03.csv
  C:\Users\jmcphail\AppData\Local\Temp\claude\C--Data-Development-Python-McTheoryApps-praxis\a31f9dd1-7758-4ce2-83d1-9f3c02a3b2e5\scratchpad\m05\out_final\sp500_identity_2022-01-03_2026-11-03_members.csv
  C:\Users\jmcphail\AppData\Local\Temp\claude\C--Data-Development-Python-McTheoryApps-praxis\a31f9dd1-7758-4ce2-83d1-9f3c02a3b2e5\scratchpad\m05\out_final\sp500_identity_2022-01-03_2026-11-03_fetch.csv
  C:\Users\jmcphail\AppData\Local\Temp\claude\C--Data-Development-Python-McTheoryApps-praxis\a31f9dd1-7758-4ce2-83d1-9f3c02a3b2e5\scratchpad\m05\out_final\sp500_identity_2022-01-03_2026-11-03.json
```


### runs_tree0.out, runs_treeA2.out, runs_treeB.out, runs_treeF.out -- HEAD; the first change at its defaults, use3 and none; the gate's tree; the finished tree

```
H0     lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [88s; --industries --membership]
A2     lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [68s; --industries --membership]
A2_none lag 1  R2 0.293802  periods 938  forecasts 468  Volati 8.6983%/+3.43%  Moment 6.6380%/+3.53%  Size 4.0255%/+2.89%  Tradin 4.8640%/+4.33%  [52s; --industries --membership --size-nonlinearity none]
A2_use3 lag 1  R2 0.299429  periods 938  forecasts 468  Volati 8.6253%/+3.47%  Moment 6.5735%/+3.64%  Size 6.8968%/+4.69%  Size N 6.3822%/-2.24%  Tradin 4.9840%/+4.11%  [52s; --industries --membership --size-nonlinearity use3]
B      lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [61s; --industries --membership]
F      lag 1  R2 0.298076  periods 938  forecasts 468  Volati 8.7732%/+3.39%  Moment 6.6235%/+3.60%  Size 4.0327%/+2.86%  Size N 2.5426%/+0.40%  Tradin 5.0653%/+4.40%  [66s; --industries --membership]
```


### exact_A.txt, exact_B.txt, exact_F.txt -- each tree against the one before, bit for bit

```
H0 vs A2: factor returns True, covariance True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 6
   +  The size pair on 2026-11-03, over the 423 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
   +                                                 Size  Size Nonlinearity     risk
   +  smallest 85, equally weighted                -3.110             -1.647   11.94%
   +  largest 50, equally weighted                 +0.012             +0.155    0.38%
   +  the index, cap-weighted                      +0.002             +0.000    0.01%
   +
A_none vs A2_none: factor returns True, covariance True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 2
   -  The size pair on 2026-11-03, over the 423 index members with a forecast (Size alone: --size-nonlinearity none): each portfolio's mean exposure, and the risk of Size alone, sqrt(x' S x) over their block of the annualised covariance
   +  The size pair on 2026-11-03, over the 423 index members with a forecast: Size alone, there being no Size Nonlinearity (--size-nonlinearity none) -- each portfolio's mean exposure, and its risk, sqrt(x' S x) on Size's annualised variance
A_use3 vs A2_use3: factor returns True, covariance True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 2
   -  The size pair on 2026-11-03, over the 423 index members with a forecast: each portfolio's mean exposure, and the risk of the two together, sqrt(x' S x) over their block of the annualised covariance
   +  The size pair on 2026-11-03, over the 423 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
A2 vs B: factor returns True, covariance True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 4
   -  3 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
   +  1 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
   -  - 2026-10-30 OKTA recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then
   -  - 2026-10-30 KVUE recorded as removed, but membership() still has it after that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then
B vs F: factor returns True, covariance True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 0
A2 vs F: factor returns True, covariance True, style exposures True, forecast exposures True, specific risks True; summary fields differing: none; report lines differing: 4
   -  3 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
   +  1 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
   -  - 2026-10-30 OKTA recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then
   -  - 2026-10-30 KVUE recorded as removed, but membership() still has it after that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then
```


### stale_figures.txt -- the two figures, from the current run

```
copy against the library: values bit for bit True; winsorised 369,070 against 369,070; clipped_outside 102,179 against 102,179

the guard after standardising moved 4,021 member values over the panel's 1,214 days; 1,408 of them LNCAP's lower tail (35.0%), 4,012 in any lower tail
  by descriptor (members moved by the guard / of them lower tail):
    LNCAP 1,408/1,408  LCAPCB 0/0  LPRI 17/17  DASTD 0/0  HILO 9/0  CMRA 0/0  BTSG 0/0  HALPHA 503/503  SERDP 0/0  RSTR 522/522  STOM 0/0  STOQ 0/0  STOA 0/0  VLVR 1,562/1,562

values of securities outside the index hitting one of the members' bounds: 102,179 (clipped_outside 102,179)
  LNCAP on 2026-11-03: 45 outsiders with a value, 19 at a bound; the raw bound 1 of them, the guard 19, both 1
  at the raw bound: SBNY

(date, descriptor) cells where the guard moved a member's lower-tail value: 2,622; cap-weighted mean above the median in 2,622, sd below the MAD scale in 20, neither in 0
  LNCAP    1142 cells, 1,408 values; (cap-weighted mean - median) / sd median +1.783, min +1.501; sd below the MAD scale in 1
  LPRI       17 cells,    17 values; (cap-weighted mean - median) / sd median +0.328, min +0.313; sd below the MAD scale in 0
  HALPHA    205 cells,   503 values; (cap-weighted mean - median) / sd median +0.448, min +0.079; sd below the MAD scale in 16
  RSTR      319 cells,   522 values; (cap-weighted mean - median) / sd median +0.554, min +0.174; sd below the MAD scale in 3
  VLVR      939 cells, 1,562 values; (cap-weighted mean - median) / sd median +1.703, min +1.482; sd below the MAD scale in 0
  upper-tail moves: 9 values in 3 cells, HILO
```


### fetch_events_tickers.txt, fetch_events_figis.txt -- the only network

```
union of membership() over 2022-01-03..2026-11-03: 591; the window's change-table tickers it never returns: ['OKTA', 'SATS']
  tickers 100/593: 95 found, 5 not found, 100 asked (23s)
  tickers 200/593: 183 found, 17 not found, 200 asked (44s)
  tickers 300/593: 277 found, 23 not found, 300 asked (66s)
  tickers 400/593: 372 found, 28 not found, 400 asked (90s)
  tickers 500/593: 467 found, 33 not found, 500 asked (112s)
  tickers: 593 ids, 555 found, 38 not found; 593 asked over the network in 132s
union of membership() over 2022-01-03..2026-11-03: 592; the window's change-table tickers it never returns: ['SATS']
composite FIGIs in the cached records: 488; carried by a ticker's answer: 447; to ask by FIGI: 41
  figis: 41 ids, 4 found, 37 not found; 41 asked over the network in 6s
```


### survey_refs.txt -- the checks across each ticker's cached records (item 3)

```
panel 506 tickers x 1214 dates; held cells 583,628; adjusted-only 0, unadjusted-only 0
sample dates: ['2022-01-03', '2023-01-04', '2024-01-05', '2025-01-07', '2026-01-09', '2026-11-03']

fields seen: {'ticker': 2885, 'name': 2885, 'market': 2885, 'locale': 2885, 'primary_exchange': 2885, 'type': 2885, 'active': 2885, 'currency_name': 2885, 'cik': 2879, 'composite_figi': 2747, 'share_class_figi': 2747, 'market_cap': 2877, 'phone_number': 2866, 'address': 2860, 'sic_code': 2866, 'sic_description': 2860, 'ticker_root': 2885, 'list_date': 2881, 'share_class_shares_outstanding': 2875, 'weighted_shares_outstanding': 2877, 'round_lot': 2881, 'description': 2388, 'homepage_url': 2384, 'total_employees': 2380, 'branding': 1881, 'ticker_suffix': 26}
types: {'CS': 2876, 'FUND': 5, 'ETF': 4}

records without a composite FIGI: 138
   ('ACN', '2022-01-03', 'Accenture PLC', '0001467373')
   ('ACN', '2023-01-04', 'Accenture PLC', '0001467373')
   ('ACN', '2024-01-05', 'Accenture PLC', '0001467373')
   ('ACN', '2025-01-07', 'Accenture PLC', '0001467373')
   ('ACN', '2026-01-09', 'Accenture PLC', '0001467373')
   ('ACN', '2026-11-03', 'Accenture PLC', '0001467373')
   ('ALLE', '2022-01-03', 'Allegion Public Limited Company', '0001579241')
   ('ALLE', '2023-01-04', 'Allegion Public Limited Company', '0001579241')
   ('ALLE', '2024-01-05', 'Allegion Public Limited Company', '0001579241')
   ('ALLE', '2025-01-07', 'Allegion Public Limited Company', '0001579241')
   ('ALLE', '2026-01-09', 'Allegion Public Limited Company', '0001579241')
   ('ALLE', '2026-11-03', 'Allegion Public Limited Company', '0001579241')
   ('AMCR', '2022-01-03', 'Amcor plc Ordinary Shares', '0001748790')
   ('AMCR', '2023-01-04', 'Amcor plc Ordinary Shares', '0001748790')
   ('AMCR', '2024-01-05', 'Amcor plc Ordinary Shares', '0001748790')
   ('AMCR', '2025-01-07', 'Amcor plc Ordinary Shares', '0001748790')
   ('AMCR', '2026-01-09', 'Amcor plc Ordinary Shares', '0001748790')
   ('AMCR', '2026-11-03', 'Amcor plc Ordinary Shares', '0001748790')
   ('AON', '2022-01-03', 'Aon plc Class A', '0000315293')
   ('AON', '2023-01-04', 'Aon plc Class A', '0000315293')
   ('AON', '2024-01-05', 'Aon plc Class A', '0000315293')
   ('AON', '2025-01-07', 'Aon plc Class A', '0000315293')
   ('AON', '2026-01-09', 'Aon plc Class A', '0000315293')
   ('AON', '2026-11-03', 'Aon plc Class A', '0000315293')
   ('APTV', '2022-01-03', 'Aptiv PLC', '0001521332')
   ('APTV', '2023-01-04', 'Aptiv PLC', '0001521332')
   ('APTV', '2024-01-05', 'Aptiv PLC', '0001521332')
   ('APTV', '2025-01-07', 'Aptiv PLC', '0001521332')
   ('APTV', '2026-01-09', 'Aptiv PLC', '0001521332')
   ('APTV', '2026-11-03', 'Aptiv PLC', '0001521332')
   ('CB', '2022-01-03', 'Chubb Limited', '0000896159')
   ('CB', '2023-01-04', 'Chubb Limited', '0000896159')
   ('CB', '2024-01-05', 'Chubb Limited', '0000896159')
   ('CB', '2025-01-07', 'Chubb Limited', '0000896159')
   ('CB', '2026-01-09', 'Chubb Limited', '0000896159')
   ('CB', '2026-11-03', 'Chubb Limited', '0000896159')
   ('CCL', '2026-11-03', 'Carnival Corporation Ltd.', '0000815097')
   ('DISCK', '2022-01-03', 'Discovery, Inc. Series C Common Stock', '0001024333')
   ('EG', '2024-01-05', 'Everest Group, Ltd.', '0001095073')
   ('EG', '2025-01-07', 'Everest Group, Ltd.', '0001095073')
   ('EG', '2026-01-09', 'Everest Group, Ltd.', '0001095073')
   ('EG', '2026-11-03', 'Everest Group, Ltd.', '0001095073')
   ('ETN', '2022-01-03', 'Eaton Corporation, plc Ordinary Shares', '0001551182')
   ('ETN', '2023-01-04', 'Eaton Corporation, plc Ordinary Shares', '0001551182')
   ('ETN', '2024-01-05', 'Eaton Corporation, plc Ordinary Shares', '0001551182')
   ('ETN', '2025-01-07', 'Eaton Corporation, plc Ordinary Shares', '0001551182')
   ('ETN', '2026-01-09', 'Eaton Corporation, plc Ordinary Shares', '0001551182')
   ('ETN', '2026-11-03', 'Eaton Corporation, plc Ordinary Shares', '0001551182')
   ('GRMN', '2022-01-03', 'Garmin Ltd', '0001121788')
   ('GRMN', '2023-01-04', 'Garmin Ltd', '0001121788')
   ('GRMN', '2024-01-05', 'Garmin Ltd', '0001121788')
   ('GRMN', '2025-01-07', 'Garmin Ltd', '0001121788')
   ('GRMN', '2026-01-09', 'Garmin Ltd', '0001121788')
   ('GRMN', '2026-11-03', 'Garmin Ltd', '0001121788')
   ('INFO', '2022-01-03', 'IHS Markit Ltd. Common Shares', '0001316360')
   ('IVZ', '2022-01-03', 'Invesco LTD', '0000914208')
   ('IVZ', '2023-01-04', 'Invesco LTD', '0000914208')
   ('IVZ', '2024-01-05', 'Invesco LTD', '0000914208')
   ('IVZ', '2025-01-07', 'Invesco LTD', '0000914208')
   ('IVZ', '2026-01-09', 'Invesco LTD', '0000914208')
   ('IVZ', '2026-11-03', 'Invesco LTD', '0000914208')
   ('JCI', '2022-01-03', 'Johnson Controls International plc', '0000833444')
   ('JCI', '2023-01-04', 'Johnson Controls International plc', '0000833444')
   ('JCI', '2024-01-05', 'Johnson Controls International plc', '0000833444')
   ('JCI', '2025-01-07', 'Johnson Controls International plc', '0000833444')
   ('JCI', '2026-01-09', 'Johnson Controls International plc', '0000833444')
   ('JCI', '2026-11-03', 'Johnson Controls International plc', '0000833444')
   ('LIN', '2022-01-03', 'Linde plc Ordinary Share', '0001707925')
   ('LIN', '2023-01-04', 'Linde plc Ordinary Share', '0001707925')
   ('LIN', '2024-01-05', 'Linde plc Ordinary Share', '0001707925')
   ('LIN', '2025-01-07', 'Linde plc Ordinary Share', '0001707925')
   ('LIN', '2026-01-09', 'Linde plc Ordinary Share', '0001707925')
   ('LIN', '2026-11-03', 'Linde plc Ordinary Share', '0001707925')
   ('LYB', '2022-01-03', 'LyondellBasell Industries N.V. Class A', '0001489393')
   ('LYB', '2023-01-04', 'LyondellBasell Industries N.V. Class A', '0001489393')
   ('LYB', '2024-01-05', 'LyondellBasell Industries N.V. Class A', '0001489393')
   ('LYB', '2025-01-07', 'LyondellBasell Industries N.V. Class A', '0001489393')
   ('LYB', '2026-01-09', 'LyondellBasell Industries N.V. Class A', '0001489393')
   ('LYB', '2026-11-03', 'LyondellBasell Industries N.V. Class A', '0001489393')
   ('MDT', '2022-01-03', 'Medtronic plc', '0001613103')
   ('MDT', '2023-01-04', 'Medtronic plc', '0001613103')
   ('MDT', '2024-01-05', 'Medtronic plc', '0001613103')
   ('MDT', '2025-01-07', 'Medtronic plc', '0001613103')
   ('MDT', '2026-01-09', 'Medtronic plc', '0001613103')
   ('MDT', '2026-11-03', 'Medtronic plc', '0001613103')
   ('NCLH', '2022-01-03', 'Norwegian Cruise Line Holdings Ltd. Ordinary Shares', '0001513761')
   ('NCLH', '2023-01-04', 'Norwegian Cruise Line Holdings Ltd. Ordinary Shares', '0001513761')
   ('NCLH', '2024-01-05', 'Norwegian Cruise Line Holdings Ltd. Ordinary Shares', '0001513761')
   ('NCLH', '2025-01-07', 'Norwegian Cruise Line Holdings Ltd. Ordinary Shares', '0001513761')
   ('NCLH', '2026-01-09', 'Norwegian Cruise Line Holdings Ltd. Ordinary Shares', '0001513761')
   ('NCLH', '2026-11-03', 'Norwegian Cruise Line Holdings Ltd. Ordinary Shares', '0001513761')
   ('NLSN', '2022-01-03', 'Nielsen Holdings plc', '0001492633')
   ('NXPI', '2022-01-03', 'NXP Semiconductors N.V.', '0001413447')
   ('NXPI', '2023-01-04', 'NXP Semiconductors N.V.', '0001413447')
   ('NXPI', '2024-01-05', 'NXP Semiconductors N.V.', '0001413447')
   ('NXPI', '2025-01-07', 'NXP Semiconductors N.V.', '0001413447')
   ('NXPI', '2026-01-09', 'NXP Semiconductors N.V.', '0001413447')
   ('NXPI', '2026-11-03', 'NXP Semiconductors N.V.', '0001413447')
   ('PNR', '2022-01-03', 'Pentair plc', '0000077360')
   ('PNR', '2023-01-04', 'Pentair plc', '0000077360')
   ('PNR', '2024-01-05', 'Pentair plc', '0000077360')
   ('PNR', '2025-01-07', 'Pentair plc', '0000077360')
   ('PNR', '2026-01-09', 'Pentair plc', '0000077360')
   ('PNR', '2026-11-03', 'Pentair plc', '0000077360')
   ('RCL', '2022-01-03', 'Royal Caribbean Group', '0000884887')
   ('RCL', '2023-01-04', 'Royal Caribbean Group', '0000884887')
   ('RCL', '2024-01-05', 'Royal Caribbean Group', '0000884887')
   ('RCL', '2025-01-07', 'Royal Caribbean Group', '0000884887')
   ('RCL', '2026-01-09', 'Royal Caribbean Group', '0000884887')
   ('RCL', '2026-11-03', 'Royal Caribbean Group', '0000884887')
   ('STE', '2022-01-03', 'STERIS plc', '0001757898')
   ('STE', '2023-01-04', 'STERIS plc', '0001757898')
   ('STE', '2024-01-05', 'STERIS plc', '0001757898')
   ('STE', '2025-01-07', 'STERIS plc', '0001757898')
   ('STE', '2026-01-09', 'STERIS plc', '0001757898')
   ('STE', '2026-11-03', 'STERIS plc', '0001757898')
   ('STX', '2022-01-03', 'Seagate Technology Holdings PLC Ordinary Shares (Ireland)', '0001137789')
   ('STX', '2023-01-04', 'Seagate Technology Holdings PLC Ordinary Shares (Ireland)', '0001137789')
   ('STX', '2024-01-05', 'Seagate Technology Holdings PLC Ordinary Shares (Ireland)', '0001137789')
   ('STX', '2025-01-07', 'Seagate Technology Holdings PLC Ordinary Shares (Ireland)', '0001137789')
   ('STX', '2026-01-09', 'Seagate Technology Holdings PLC Ordinary Shares (Ireland)', '0001137789')
   ('STX', '2026-11-03', 'Seagate Technology Holdings PLC Ordinary Shares (Ireland)', '0001137789')
   ('SW', '2025-01-07', 'Smurfit Westrock plc', '0002005951')
   ('SW', '2026-01-09', 'Smurfit Westrock plc', '0002005951')
   ('SW', '2026-11-03', 'Smurfit Westrock plc', '0002005951')
   ('TEL', '2022-01-03', 'TE CONNECTIVITY LTD', '0001385157')
   ('TEL', '2023-01-04', 'TE CONNECTIVITY LTD', '0001385157')
   ('TEL', '2024-01-05', 'TE CONNECTIVITY LTD', '0001385157')
   ('TEL', '2025-01-07', 'TE Connectivity plc', '0001385157')
   ('TEL', '2026-01-09', 'TE Connectivity plc', '0001385157')
   ('TEL', '2026-11-03', 'TE Connectivity plc', '0001385157')
   ('TT', '2022-01-03', 'Trane Technologies plc', '0001466258')
   ('TT', '2023-01-04', 'Trane Technologies plc', '0001466258')
   ('TT', '2024-01-05', 'Trane Technologies plc', '0001466258')
   ('TT', '2025-01-07', 'Trane Technologies plc', '0001466258')
   ('TT', '2026-01-09', 'Trane Technologies plc', '0001466258')
   ('TT', '2026-11-03', 'Trane Technologies plc', '0001466258')
   ('WLTW', '2022-01-03', 'Willis Towers Watson Public Limited Company Ordinary Shares', '0001140536')

changes between consecutive records of one ticker: 38
   ('BEN', '2026-01-09', '2026-11-03', ['name'], 'Franklin Resources, Inc.', 'Franklin Templeton, Inc.', 'BBG000BD0TF8', 'BBG000BD0TF8', '0000038777', '0000038777', 'CS', 'CS')
   ('BLK', '2024-01-05', '2025-01-07', ['cik', 'composite_figi', 'share_class_figi'], 'Blackrock, Inc.', 'Blackrock, Inc.', 'BBG000C2PW58', 'BBG01PSW2WN4', '0001364742', '0002012383', 'CS', 'CS')
   ('BNY', '2026-01-09', '2026-11-03', ['name', 'cik', 'composite_figi', 'share_class_figi', 'type'], 'BlackRock New York Municipal Income Trust', 'Bank of New York Mellon Corporation', 'BBG000BZF6G2', 'BBG000BD8PN9', '0001137390', '0001390777', 'FUND', 'CS')
   ('BXP', '2024-01-05', '2025-01-07', ['name'], 'Boston Properties, Inc.', 'BXP, Inc.', 'BBG000BS5CM9', 'BBG000BS5CM9', '0001037540', '0001037540', 'CS', 'CS')
   ('CCI', '2022-01-03', '2023-01-04', ['name'], 'Crown Castle Intl. Corp.', 'Crown Castle Inc.', 'BBG000FV1Z23', 'BBG000FV1Z23', '0001051470', '0001051470', 'CS', 'CS')
   ('CCL', '2026-01-09', '2026-11-03', ['name', 'composite_figi', 'share_class_figi'], 'Carnival Corporation', 'Carnival Corporation Ltd.', 'BBG000BF6LY3', None, '0000815097', '0000815097', 'CS', 'CS')
   ('CI', '2023-01-04', '2024-01-05', ['name'], 'CIGNA Corporation', 'The Cigna Group', 'BBG00KXRCDP0', 'BBG00KXRCDP0', '0001739940', '0001739940', 'CS', 'CS')
   ('CMS', '2024-01-05', '2025-01-07', ['cik'], 'CMS Energy Corporation', 'CMS Energy Corporation', 'BBG000BFVXX0', 'BBG000BFVXX0', '0000811156', '0000201533', 'CS', 'CS')
   ('CMS', '2025-01-07', '2026-01-09', ['cik'], 'CMS Energy Corporation', 'CMS Energy Corporation', 'BBG000BFVXX0', 'BBG000BFVXX0', '0000201533', '0000811156', 'CS', 'CS')
   ('CMS', '2026-01-09', '2026-11-03', ['cik'], 'CMS Energy Corporation', 'CMS Energy Corporation', 'BBG000BFVXX0', 'BBG000BFVXX0', '0000811156', '0000201533', 'CS', 'CS')
   ('COO', '2023-01-04', '2024-01-05', ['name'], 'The Cooper Companies, Inc.', 'The Cooper Companies, Inc. Common Stock', 'BBG000BG3445', 'BBG000BG3445', '0000711404', '0000711404', 'CS', 'CS')
   ('CPB', '2024-01-05', '2025-01-07', ['name'], 'Campbell Soup Company', "The Campbell's Company Common Stock", 'BBG000BG4202', 'BBG000BG4202', '0000016732', '0000016732', 'CS', 'CS')
   ('CRM', '2022-01-03', '2023-01-04', ['name'], 'salesforce.com, inc.', 'Salesforce, Inc.', 'BBG000BN2DC2', 'BBG000BN2DC2', '0001108524', '0001108524', 'CS', 'CS')
   ('DOC', '2024-01-05', '2025-01-07', ['name', 'cik', 'composite_figi', 'share_class_figi'], 'PHYSICIANS REALTY TRUST', 'Healthpeak Properties, Inc.', 'BBG004MF59H0', 'BBG000BKYDP9', '0001574540', '0000765880', 'CS', 'CS')
   ('EXPD', '2023-01-04', '2024-01-05', ['name'], 'Expeditors International of Washington Inc.', 'Expeditors International of Washington, Inc.', 'BBG000BJ5GK2', 'BBG000BJ5GK2', '0000746515', '0000746515', 'CS', 'CS')
   ('FISV', '2023-01-04', '2026-01-09', ['name'], 'Fiserv Inc', 'Fiserv, Inc. Common Stock', 'BBG000BJKPG0', 'BBG000BJKPG0', '0000798354', '0000798354', 'CS', 'CS')
   ('GE', '2024-01-05', '2025-01-07', ['name'], 'General Electric Company', 'GE Aerospace', 'BBG000BK6MB5', 'BBG000BK6MB5', '0000040545', '0000040545', 'CS', 'CS')
   ('HIG', '2025-01-07', '2026-01-09', ['name'], 'The Hartford Financial Services Group, Inc.', 'The Hartford Insurance Group, Inc.', 'BBG000G0Z878', 'BBG000G0Z878', '0000874766', '0000874766', 'CS', 'CS')
   ('INFO', '2022-01-03', '2025-01-07', ['name', 'cik', 'composite_figi', 'share_class_figi', 'type'], 'IHS Markit Ltd. Common Shares', 'Harbor PanAgora Dynamic Large Cap Core ETF', None, 'BBG01Q2P8F92', '0001316360', None, 'CS', 'ETF')
   ('INFO', '2025-01-07', '2026-01-09', ['cik'], 'Harbor PanAgora Dynamic Large Cap Core ETF', 'Harbor PanAgora Dynamic Large Cap Core ETF', 'BBG01Q2P8F92', 'BBG01Q2P8F92', None, '0001860434', 'ETF', 'ETF')
   ('J', '2022-01-03', '2023-01-04', ['name', 'composite_figi', 'share_class_figi'], 'Jacobs Engineering Group', 'Jacobs Solutions Inc.', 'BBG000BMFFQ0', 'BBG019C1BQR4', '0000052988', '0000052988', 'CS', 'CS')
   ('K', '2023-01-04', '2024-01-05', ['name'], 'Kellogg Company', 'Kellanova', 'BBG000BMKDM3', 'BBG000BMKDM3', '0000055067', '0000055067', 'CS', 'CS')
   ('KHC', '2026-01-09', '2026-11-03', ['name'], 'The Kraft Heinz Company Common Stock', 'The Kraft Heinz Company', 'BBG005CPNTQ2', 'BBG005CPNTQ2', '0001637459', '0001637459', 'CS', 'CS')
   ('LH', '2024-01-05', '2025-01-07', ['name', 'composite_figi', 'share_class_figi'], 'Laboratory Corporation of America Holdings', 'Labcorp Holdings Inc.', 'BBG000D9DMK0', 'BBG01MMT6PL7', '0000920148', '0000920148', 'CS', 'CS')
   ('LUMN', '2026-01-09', '2026-11-03', ['name'], 'Lumen Technologies, Inc.', 'Lumen Technologies, Inc. Common Stock', 'BBG000BGLRN3', 'BBG000BGLRN3', '0000018926', '0000018926', 'CS', 'CS')
   ('MAR', '2025-01-07', '2026-01-09', ['name'], 'Marriot International Class A Common Stock', 'Marriott International Class A Common Stock', 'BBG000BGD7W6', 'BBG000BGD7W6', '0001048286', '0001048286', 'CS', 'CS')
   ('META', '2022-01-03', '2023-01-04', ['name', 'cik', 'composite_figi', 'share_class_figi', 'type'], 'Roundhill Ball Metaverse ETF', 'Meta Platforms, Inc. Class A Common Stock', 'BBG011J1MN62', 'BBG000MM2P62', None, '0001326801', 'ETF', 'CS')
   ('NEE', '2025-01-07', '2026-01-09', ['name'], 'Nextra Energy, Inc.', 'NextEra Energy, Inc.', 'BBG000BJSBJ0', 'BBG000BJSBJ0', '0000753308', '0000753308', 'CS', 'CS')
   ('OKE', '2026-01-09', '2026-11-03', ['composite_figi', 'share_class_figi'], 'Oneok, Inc.', 'Oneok, Inc.', 'BBG000BQHGR6', 'BBG024TZWVN1', '0001039684', '0001039684', 'CS', 'CS')
   ('PENN', '2022-01-03', '2023-01-04', ['name'], 'Penn National Gaming Inc', 'PENN Entertainment, Inc. Common Stock', 'BBG000CDZLV8', 'BBG000CDZLV8', '0000921738', '0000921738', 'CS', 'CS')
   ('RHI', '2023-01-04', '2024-01-05', ['name'], 'Robert Half International Inc.', 'Robert Half Inc.', 'BBG000BS5DR2', 'BBG000BS5DR2', '0000315213', '0000315213', 'CS', 'CS')
   ('ROP', '2023-01-04', '2024-01-05', ['name'], 'Roper Technologies, Inc', 'Roper Technologies, Inc. Common Stock', 'BBG000F1ZSN5', 'BBG000F1ZSN5', '0000882835', '0000882835', 'CS', 'CS')
   ('RTX', '2023-01-04', '2024-01-05', ['name'], 'Raytheon Technologies Corporation', 'RTX Corporation', 'BBG000BW8S60', 'BBG000BW8S60', '0000101829', '0000101829', 'CS', 'CS')
   ('SBNY', '2023-01-04', '2024-01-05', ['name', 'cik'], 'Signature Bank', 'SIGNATURE BANK (NY)', 'BBG000M6TR37', 'BBG000M6TR37', '0001288784', None, 'CS', 'CS')
   ('SLB', '2025-01-07', '2026-01-09', ['name'], 'Schlumberger Limited', 'SLB Limited', 'BBG000BT41Q8', 'BBG000BT41Q8', '0000087347', '0000087347', 'CS', 'CS')
   ('TEL', '2024-01-05', '2025-01-07', ['name'], 'TE CONNECTIVITY LTD', 'TE Connectivity plc', None, None, '0001385157', '0001385157', 'CS', 'CS')
   ('WMT', '2025-01-07', '2026-01-09', ['name'], 'Walmart Inc.', 'Walmart Inc. Common Stock', 'BBG000BWXBC2', 'BBG000BWXBC2', '0000104169', '0000104169', 'CS', 'CS')
   ('XOM', '2026-01-09', '2026-11-03', ['name', 'cik', 'composite_figi', 'share_class_figi'], 'Exxon Mobil Corporation', 'ExxonMobil Holdings Corporation', 'BBG000GZQ728', 'BBG023CY9MM1', '0000034088', '0002115436', 'CS', 'CS')

list date after the ticker's first price: 3
   ('INFO', '2025-01-07', 'Harbor PanAgora Dynamic Large Cap Core ETF', '2024-10-10', '2022-01-03')
   ('INFO', '2026-01-09', 'Harbor PanAgora Dynamic Large Cap Core ETF', '2026-01-02', '2022-01-03')
   ('INFO', '2026-11-03', 'Harbor PanAgora Dynamic Large Cap Core ETF', '2024-10-09', '2022-01-03')

no record, by sample date: {'2023-01-04': 22, '2024-01-05': 24, '2025-01-07': 26, '2026-01-09': 32, '2026-11-03': 35, '2022-01-03': 12}
```


### survey_events.txt -- the events answers against the cached records

```
593 asked; 38 not found: ABMD ANSS ATVI AVB CDAY CERN CMA CTLT CTRA CTXS DAY DFS DISCA DISCK DISH DRE EA FBHS FRC GPS HES HOLX IPG JNPR K KVUE MRO NLSN PBCT PXD QRVO SEE SIVB TWTR WBA WBD WLTW XLNX
event types: {'ticker_change': 616}
not found and in the panel: 35; not found and missing from the panel: 3: DAY KVUE WBD

found, in the panel, and the answer's FIGI is not every cached record's:
  BLK    now 'Blackrock, Inc.' BBG01PSW2WN4 cik 0002012383; events [('2024-10-02', 'BLK')]; cached FIGIs ['BBG000C2PW58', 'BBG01PSW2WN4']; held 2022-01-03..2026-11-03
  BNY    now 'Bank of New York Mellon Corporation' BBG000BD8PN9 cik 0001390777; events [('2007-07-02', 'BK'), ('2026-05-21', 'BNY')]; cached FIGIs ['BBG000BD8PN9', 'BBG000BZF6G2']; held 2022-01-03..2026-11-03
  CCL    now 'Carnival Corporation Ltd.' None cik 0000815097; events [('2026-05-07', 'CCL')]; cached FIGIs ['-', 'BBG000BF6LY3']; held 2022-01-03..2026-11-03
  DOC    now 'Healthpeak Properties, Inc.' BBG000BKYDP9 cik 0000765880; events [('2019-11-05', 'PEAK'), ('2024-03-04', 'DOC')]; cached FIGIs ['BBG000BKYDP9', 'BBG004MF59H0']; held 2022-01-03..2026-11-03
  INFO   now 'Harbor PanAgora Dynamic Large Cap Core ETF' BBG01Q2P8F92 cik 0001860434; events [('2024-10-10', 'INFO')]; cached FIGIs ['-', 'BBG01Q2P8F92']; held 2022-01-03..2026-11-03
  J      now 'Jacobs Solutions Inc.' BBG019C1BQR4 cik 0000052988; events [('2022-08-29', 'J')]; cached FIGIs ['BBG000BMFFQ0', 'BBG019C1BQR4']; held 2022-01-03..2026-11-03
  LH     now 'Labcorp Holdings Inc.' BBG01MMT6PL7 cik 0000920148; events [('2024-05-20', 'LH')]; cached FIGIs ['BBG000D9DMK0', 'BBG01MMT6PL7']; held 2022-01-03..2026-11-03
  META   now 'Meta Platforms, Inc. Class A Common Stock' BBG000MM2P62 cik 0001326801; events [('2012-05-18', 'FB'), ('2022-06-09', 'META')]; cached FIGIs ['BBG000MM2P62', 'BBG011J1MN62']; held 2022-01-03..2026-11-03
  OKE    now 'Oneok, Inc.' BBG024TZWVN1 cik 0001039684; events [('2026-09-11', 'OKE')]; cached FIGIs ['BBG000BQHGR6', 'BBG024TZWVN1']; held 2022-01-03..2026-11-03
  XOM    now 'ExxonMobil Holdings Corporation' BBG023CY9MM1 cik 0002115436; events [('2026-07-02', 'XOM')]; cached FIGIs ['BBG000GZQ728', 'BBG023CY9MM1']; held 2022-01-03..2026-11-03

found, in the panel, the answer's ticker tenure not covering the panel's first price:
  AMCR   'Amcor plc Ordinary Shares' events [('2026-01-15', 'AMCR')]; first price 2022-01-03
  ANET   'Arista Networks' events [('2024-12-04', 'ANET')]; first price 2022-01-03
  APTV   'Aptiv PLC' events [('2024-12-18', 'APTV')]; first price 2022-01-03
  BF.B   'Brown-Forman Corporation Class B' events [('2003-09-10', 'BF')]; first price 2022-01-03
  BLK    'Blackrock, Inc.' events [('2024-10-02', 'BLK')]; first price 2022-01-03
  BNY    'Bank of New York Mellon Corporation' events [('2007-07-02', 'BK'), ('2026-05-21', 'BNY')]; first price 2022-01-03
  BRK.B  'BERKSHIRE HATHAWAY Class B' events [('2010-01-22', 'BRK')]; first price 2022-01-03
  CCL    'Carnival Corporation Ltd.' events [('2026-05-07', 'CCL')]; first price 2022-01-03
  COO    'The Cooper Companies, Inc. Common Stock' events [('2024-02-20', 'COO')]; first price 2022-01-03
  DD     'DuPont de Nemours, Inc. Common Stock' events [('2026-06-24', 'DD')]; first price 2022-01-03
  DOC    'Healthpeak Properties, Inc.' events [('2019-11-05', 'PEAK'), ('2024-03-04', 'DOC')]; first price 2022-01-03
  GOOGL  'Alphabet Inc. Class A Common Stock' events [('2015-10-06', 'GOOG')]; first price 2022-01-03
  HON    'Honeywell International, Inc.' events [('2026-06-15', 'HONI'), ('2026-06-29', 'HON')]; first price 2022-01-03
  INFO   'Harbor PanAgora Dynamic Large Cap Core ETF' events [('2024-10-10', 'INFO')]; first price 2022-01-03
  J      'Jacobs Solutions Inc.' events [('2022-08-29', 'J')]; first price 2022-01-03
  LH     'Labcorp Holdings Inc.' events [('2024-05-20', 'LH')]; first price 2022-01-03
  LIN    'Linde plc Ordinary Share' events [('2023-03-02', 'LIN')]; first price 2022-01-03
  LRCX   'Lam Research Corp' events [('2024-10-03', 'LRCX')]; first price 2022-01-03
  META   'Meta Platforms, Inc. Class A Common Stock' events [('2012-05-18', 'FB'), ('2022-06-09', 'META')]; first price 2022-01-03
  OKE    'Oneok, Inc.' events [('2026-09-11', 'OKE')]; first price 2022-01-03
  TEL    'TE Connectivity plc' events [('2024-09-30', 'TEL')]; first price 2022-01-03
  XOM    'ExxonMobil Holdings Corporation' events [('2026-07-02', 'XOM')]; first price 2022-01-03

found, missing from the panel (the 85 and OKTA, SATS):
  ABNB   'Airbnb, Inc. Class A Common Stock' BBG001Y2XS07 cik 0001559720; events [('2020-12-10', 'ABNB')]
  ACGL   'Arch Capital Group Ltd' None cik 0000947484; events [('2003-09-10', 'ACGL')]
  AMTM   'Amentum Holdings, Inc.' BBG01NY7B734 cik 0002011286; events [('2024-09-24', 'AMTM')]
  APO    'Apollo Global Management, Inc.' BBG00ZNLTFK3 cik 0001858681; events [('2022-01-03', 'APO')]
  APP    'Applovin Corporation Class A Common Stock' BBG006HFPX77 cik 0001751008; events [('2021-04-15', 'APP')]
  ARES   'Ares Management Corporation Class A Common Stock' BBG0068KVV91 cik 0001176948; events [('2018-11-26', 'ARES')]
  AXON   'Axon Enterprise, Inc. Common Stock' BBG000BHJWG1 cik 0001069183; events [('2017-04-06', 'AAXN'), ('2021-01-26', 'AXON')]
  BE     'Bloom Energy Corporation' BBG000N7KBZ3 cik 0001664703; events [('2018-07-24', 'BE')]
  BG     'Bunge Global SA' None cik 0001996862; events [('2023-11-01', 'BG')]
  BLDR   'Builders FirstSource, Inc.' BBG000BKD3K9 cik 0001316835; events [('2005-06-22', 'BLDR')]
  BX     'Blackstone Inc.' BBG000BH0106 cik 0001393818; events [('2019-07-01', 'BX')]
  CASY   "Casey's General Stores Inc" BBG000BF0672 cik 0000726958; events [('2003-09-10', 'CASY'), ('2010-08-27', 'CASYV')]
  CEG    'Constellation Energy Corporation Common Stock' BBG014KFRNP7 cik 0001868275; events [('2022-01-19', 'CEGV'), ('2022-02-02', 'CEG')]
  CIEN   'Ciena Corporation' BBG000BP1152 cik 0000936395; events [('2006-09-25', 'CIEND'), ('2006-10-23', 'CIEN')]
  COHR   'Coherent Corp.' BBG000BLW102 cik 0000820318; events [('2022-09-08', 'COHR')]
  COIN   'Coinbase Global, Inc. Class A Common Stock' BBG00ZGF7771 cik 0001679788; events [('2021-04-14', 'COIN')]
  CPT    'Camden Property Trust' BBG000BB0V03 cik 0000906345; events [('2003-09-10', 'CPT')]
  CRH    'CRH Public Limited Company' None cik 0000849395; events [('2021-09-23', 'CRHC'), ('2023-09-25', 'CRH')]
  CRWD   'CrowdStrike Holdings, Inc. Class A Common Stock' BBG00BLYKS03 cik 0001535527; events [('2019-06-12', 'CRWD')]
  CSGP   'CoStar Group Inc' BBG000D7JKW9 cik 0001057352; events [('2003-09-10', 'CSGP')]
  CVNA   'Carvana Co.' BBG00GCTWDJ3 cik 0001690820; events [('2017-04-27', 'CVNA')]
  DASH   'DoorDash, Inc. Class A Common Stock' BBG005D7QCJ3 cik 0001792789; events [('2020-12-08', 'DASH')]
  DDOG   'Datadog, Inc. Class A Common Stock' BBG003NJHZT9 cik 0001561550; events [('2019-09-19', 'DDOG')]
  DECK   'Deckers Outdoor Corp' BBG000BKXYX5 cik 0000910521; events [('2003-09-10', 'DECK')]
  DELL   'Dell Technologies Inc.' BBG00DW3SZS1 cik 0001571996; events [('2018-12-26', 'DELL')]
  EME    'EMCOR Group, Inc.' BBG000BBRPL4 cik 0000105634; events [('2003-09-10', 'EME')]
  EQT    'EQT CORP' BBG000BHZ5J9 cik 0000033213; events [('2009-02-10', 'EQT')]
  ERIE   'Erie Indemnity Co' BBG000BBTJZ1 cik 0000922621; events [('2003-09-10', 'ERIE')]
  EXE    'Expand Energy Corporation Common Stock' BBG00Z6DX554 cik 0000895126; events [('2021-02-10', 'CHK'), ('2024-10-02', 'EXE')]
  FDXF   'FedEx Freight Holding Company, Inc.' BBG01ZPB91C4 cik 0002082247; events [('2026-05-27', 'FDXF')]
  FERG   'Ferguson Enterprises Inc.' BBG01NMRD768 cik 0002011641; events [('2024-08-01', 'FERG')]
  FICO   'Fair Isaac Corporation' BBG000DW76Y6 cik 0000814547; events [('2003-09-10', 'FIC'), ('2009-08-19', 'FICO')]
  FIX    'Comfort Systems USA, Inc.' BBG000BWN2Y9 cik 0001035983; events [('2003-09-10', 'FIX')]
  FLEX   'Flex Ltd. Ordinary Shares' None cik 0000866374; events [('2003-09-10', 'FLEX')]
  FSLR   'First Solar, Inc.' BBG000BFL116 cik 0001274494; events [('2006-11-17', 'FSLR')]
  GDDY   'GoDaddy Inc' BBG006MDLY05 cik 0001609711; events [('2015-04-01', 'GDDY')]
  GEHC   'GE HealthCare Technologies Inc. Common Stock' BBG01BFR8YV1 cik 0001932393; events [('2022-12-15', 'GEHC')]
  GEV    'GE Vernova Inc.' BBG013G17W31 cik 0001996810; events [('2024-03-27', 'GEV')]
  HONA   'Honeywell Aerospace Inc. Common Stock' BBG020QYV152 cik 0002089271; events [('2026-06-15', 'HONA')]
  HOOD   'Robinhood Markets, Inc. Class A Common Stock' BBG008NMBXN8 cik 0001783879; events [('2021-07-29', 'HOOD')]
  HUBB   'Hubbell Incorporated' BBG000BLK267 cik 0000048898; events [('2015-12-28', 'HUBB')]
  IBKR   'Interactive Brokers Group, Inc. Class A Common Stock' BBG000LV0836 cik 0001381197; events [('2007-05-04', 'IBKR')]
  INVH   'Invitation Homes Inc. Common Stock' BBG00FQH6BS9 cik 0001687229; events [('2017-01-31', 'INVH')]
  JBL    'Jabil Inc.' BBG000BJNGN9 cik 0000898293; events [('2003-09-10', 'JBL')]
  KDP    'Keurig Dr Pepper Inc.' BBG000TJM7F0 cik 0001418135; events [('2018-07-10', 'KDP')]
  KKR    'KKR & Co. Inc.' BBG000BCQ6J8 cik 0001404912; events [('2018-07-02', 'KKR')]
  LII    'Lennox International Inc.' BBG000BB5B84 cik 0001069202; events [('2003-09-10', 'LII')]
  LITE   'Lumentum Holdings Inc. Common Stock' BBG0073F9RT7 cik 0001633978; events [('2015-07-23', 'LITE')]
  LULU   'lululemon athletica inc.' BBG000R8ZVD1 cik 0001397187; events [('2007-07-27', 'LULUV'), ('2007-07-30', 'LULU')]
  MBC    'MasterBrand, Inc.' BBG01B9G42Y3 cik 0001941365; events [('2022-12-09', 'MBC')]
  MOH    'Molina Healthcare, Inc.' BBG000MBHNC8 cik 0001179929; events [('2003-09-10', 'MOH')]
  MRVL   'Marvell Technology, Inc. Common Stock' BBG00ZXBJ153 cik 0001835632; events [('2021-04-21', 'MRVL')]
  NDSN   'Nordson Corp' BBG000BPWGR1 cik 0000072331; events [('2003-09-10', 'NDSN')]
  ON     'ON Semiconductor Corp' BBG000DV7MX4 cik 0001097864; events [('2003-09-10', 'ONNN'), ('2015-04-07', 'ON')]
  P      'Everpure, Inc.' BBG00212PVZ5 cik 0001474432; events [('2015-10-07', 'PSTG'), ('2026-04-17', 'P')]
  PANW   'Palo Alto Networks, Inc. Common Stock' BBG0014GJCT9 cik 0001327567; events [('2012-07-20', 'PANW')]
  PCG    'PG&E Corporation' BBG000BQWPC5 cik 0001004980; events [('2003-09-10', 'PCG')]
  PLTR   'Palantir Technologies Inc. Class A Common Stock' BBG000N7QR55 cik 0001321655; events [('2020-09-29', 'PLTR')]
  PODD   'Insulet Corporation' BBG000R7XX87 cik 0001145197; events [('2007-05-15', 'PODD')]
  Q      'Qnity Electronics, Inc.' BBG01TRL73S5 cik 0002058873; events [('2025-10-27', 'Q')]
  RDDT   'Reddit, Inc.' BBG005K1D4S0 cik 0001713445; events [('2024-03-20', 'RDDT')]
  SMCI   'Super Micro Computer, Inc. Common Stock' BBG000MYZDJ3 cik 0001375365; events [('2024-10-01', 'SMCI')]
  SNDK   'Sandisk Corporation Common Stock' BBG01R388JG1 cik 0002023554; events [('2025-02-13', 'SNDK')]
  SOLS   'Solstice Advanced Materials Inc. Common Stock' BBG01X5744P5 cik 0002064953; events [('2025-10-20', 'SOLS')]
  SOLV   'Solventum Corporation' BBG018YZH6T3 cik 0001964738; events [('2024-03-26', 'SOLV')]
  STLD   'Steel Dynamics Inc' BBG000HGYNZ9 cik 0001022671; events [('2003-09-10', 'STLD')]
  TKO    'TKO Group Holdings, Inc.' BBG01G9JKWV5 cik 0001973266; events [('2023-09-12', 'TKO')]
  TPL    'Texas Pacific Land Corporation' BBG00YRJXKB1 cik 0001811074; events [('2021-01-11', 'TPL')]
  TRGP   'Targa Resources Corp.' BBG0015XMW40 cik 0001389170; events [('2010-12-07', 'TRGP')]
  TTD    'The Trade Desk, Inc.' BBG00629NGT2 cik 0001671933; events [('2016-09-21', 'TTD')]
  TWLO   'Twilio Inc.' BBG0029ZX840 cik 0001447669; events [('2016-06-22', 'TWLO')]
  UBER   'Uber Technologies, Inc.' BBG002B04MT8 cik 0001543151; events [('2019-05-09', 'UBER')]
  VEEV   'Veeva Systems Inc.' BBG001CGB489 cik 0001393052; events [('2013-10-16', 'VEEV')]
  VICI   'VICI Properties Inc. Common Stock' BBG00HVVB499 cik 0001705696; events [('2018-02-01', 'VICI')]
  VLTO   'Veralto Corporation' BBG019Q32XJ9 cik 0001967680; events [('2023-09-27', 'VLTO')]
  VRT    'Vertiv Holdings Co Class A Common Stock' BBG00L2B8KW8 cik 0001674101; events [('2020-02-10', 'VRT')]
  VST    'Vistra Corp.' BBG00DXDL6Q1 cik 0001692819; events [('2017-05-10', 'VST')]
  VYLR   'Vylor Inc.' BBG024CQR6X9 cik 0002128626; events [('2026-10-01', 'VYLR')]
  WDAY   'Workday, Inc. Class A Common Stock' BBG000VC0T95 cik 0001327811; events [('2012-10-12', 'WDAY')]
  WSM    'Williams-Sonoma, Inc.' BBG000FSMWC3 cik 0000719955; events [('2003-09-10', 'WSM')]
  WTW    'Willis Towers Watson Public Limited Company Ordinary Shares' None cik 0001140536; events [('2016-01-05', 'WLTW'), ('2022-01-10', 'WTW')]
  XYZ    'Block, Inc.' BBG0018SLC07 cik 0001512673; events [('2015-11-18', 'SQ'), ('2025-01-21', 'XYZ')]
  OKTA   'Okta, Inc. Class A Common Stock' BBG001YV1SM4 cik 0001660134; events [('2017-04-07', 'OKTA')]
  SATS   'Strive, Inc. 15% Series A Perpetual SATS Preferred Stock' None cik 0001920406; events [('2026-10-17', 'STVE'), ('2026-10-20', 'SATS')]
```


### plan_breakdown.txt -- the table, the listing and the plan, broken down

```
The table
  rows 625: {'panel': 518, 'events': 106, 'presumed': 1}; securities 598
  securities by type: {'CS': 513, '-': 82, 'FUND': 1, 'ETF': 2}
  keyed by: {'FIGI': 566, 'CIK': 32}

The member listing
  runs 593 over 592 tickers; sources {'records+events': 476, 'records': 32, 'events': 82, 'neither': 3}
  records: ABMD ANSS ATVI AVB CDAY CERN CMA CTLT CTRA CTXS DFS DISCA DISCK DISH DRE EA HES HOLX INFO IPG JNPR K MRO NLSN PBCT PXD QRVO SEE SIVB TWTR WBA XLNX
  neither: DAY KVUE WBD
  member-days by how: {'panel': 550800, 'events': 51263, 'after its last price': 31, 'unverified': 2384, 'predecessor': 4843, 'presumed': 145, 'missing': 2429, 'inside a gap in its prices': 4}; total 611,899
  runs of tickers the panel lacks: 86; their sources {'events': 82, 'neither': 3, 'records+events': 1}

The fetch plan
  the panel's tickers: 514 spans, 506 tickers, 556,582 member-days, share samples 10,113 (7,252 new)
  tickers of members the panel lacks: 83 spans, 83 tickers, 43,900 member-days, share samples 1,507 (1,507 new)
    ABNB ACGL AMTM APO APP ARES AXON BE BG BLDR BX CASY CEG CIEN COHR COIN CPT CRH CRWD CSGP CVNA DASH DDOG DECK DELL EME EQT ERIE EXE FDXF FERG FICO FIX FLEX FSLR GDDY GEHC GEV HONA HOOD HUBB IBKR INVH JBL KDP KKR LII LITE LULU MBC MOH MRVL NDSN OKTA ON P PANW PCG PLTR PODD Q RDDT SMCI SNDK SOLS SOLV STLD TKO TPL TRGP TTD TWLO UBER VEEV VICI VLTO VRT VST VYLR WDAY WSM WTW XYZ
  old tickers of members the panel truncates: 16 spans, 16 tickers, 7,508 member-days, share samples 162 (162 new)
    ABC ANTM BK BLL EQR FB FBIN FI FRCB MMC NLOK PEAK PKI PSKY RE SATS
  other tickers of member securities, outside their membership: 7 spans, 7 tickers, 0 member-days, share samples 80 (80 new)
    CEGV CHK CRHC GAP PSTG S600 SQ
  new tickers 106: bars 212; share samples new 9,001; split events 612
  share samples per security: min 2, median 21, max 25
```


### The member tickers: each run of membership and the security it names (593 runs)

From `sp500_identity_2022-01-03_2026-11-03_members.csv`. `How` counts the run's dates by how each maps (see the guards above).

| Ticker | Member | Security | Name | FIGI / CIK | Traded as | How | Sources |
|---|---|---|---|---|---|---|---|
| A | 2022-01-03 to 2026-11-03 (1214) | BBG000C2V3D6 | Agilent Technologies Inc. | BBG000C2V3D6 / 0001090872 | A 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AAL | 2022-01-03 to 2024-09-20 (683) | BBG005P7Q881 | American Airlines Group Inc. | BBG005P7Q881 / 0000006201 | AAL 2022-01-03..2024-09-20 | panel 683 | records+events |
| AAP | 2022-01-03 to 2023-08-24 (413) | BBG000F7RCJ1 | ADVANCE AUTO PARTS INC | BBG000F7RCJ1 / 0001158449 | AAP 2022-01-03..2023-08-24 | panel 413 | records+events |
| AAPL | 2022-01-03 to 2026-11-03 (1214) | BBG000B9XRY4 | Apple Inc. | BBG000B9XRY4 / 0000320193 | AAPL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ABBV | 2022-01-03 to 2026-11-03 (1214) | BBG0025Y4RY4 | ABBVIE INC. | BBG0025Y4RY4 / 0001551152 | ABBV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ABMD | 2022-01-03 to 2022-12-21 (245) | BBG000C101X4 | Abiomed Inc | BBG000C101X4 / 0000815094 | ABMD 2022-01-03..2022-12-21 | panel 245 | records |
| ABNB | 2023-09-18 to 2026-11-03 (786) | BBG001Y2XS07 | Airbnb, Inc. Class A Common Stock | BBG001Y2XS07 / 0001559720 | ABNB 2023-09-18..2026-11-03 | events 786 | events |
| ABT | 2022-01-03 to 2026-11-03 (1214) | BBG000B9ZXB4 | Abbott Laboratories | BBG000B9ZXB4 / 0000001800 | ABT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ACGL | 2022-11-01 to 2026-11-03 (1005) | CIK0000947484 | Arch Capital Group Ltd | 0000947484 | ACGL 2022-11-01..2026-11-03 | events 1005 | events |
| ACN | 2022-01-03 to 2026-11-03 (1214) | CIK0001467373 | Accenture PLC | 0001467373 | ACN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ADBE | 2022-01-03 to 2026-11-03 (1214) | BBG000BB5006 | Adobe Inc. | BBG000BB5006 / 0000796343 | ADBE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ADI | 2022-01-03 to 2026-11-03 (1214) | BBG000BB6G37 | Analog Devices, Inc. | BBG000BB6G37 / 0000006281 | ADI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ADM | 2022-01-03 to 2026-11-03 (1214) | BBG000BB6WG8 | Archer Daniels Midland Company | BBG000BB6WG8 / 0000007084 | ADM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ADP | 2022-01-03 to 2026-11-03 (1214) | BBG000JG0547 | Automatic Data Processing | BBG000JG0547 / 0000008670 | ADP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ADSK | 2022-01-03 to 2026-11-03 (1214) | BBG000BM7HL0 | Autodesk Inc | BBG000BM7HL0 / 0000769397 | ADSK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AEE | 2022-01-03 to 2026-11-03 (1214) | BBG000B9X8C0 | Ameren Corporation | BBG000B9X8C0 / 0001002910 | AEE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AEP | 2022-01-03 to 2026-11-03 (1214) | BBG000BB9KF2 | American Electric Power Company, Inc. | BBG000BB9KF2 / 0000004904 | AEP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AES | 2022-01-03 to 2026-11-03 (1214) | BBG000C23KJ3 | AES Corporation | BBG000C23KJ3 / 0000874761 | AES 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AFL | 2022-01-03 to 2026-11-03 (1214) | BBG000BBBNC6 | Aflac Inc. | BBG000BBBNC6 / 0000004977 | AFL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AIG | 2022-01-03 to 2026-11-03 (1214) | BBG000BBDZG3 | American International Group, Inc. | BBG000BBDZG3 / 0000005272 | AIG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AIZ | 2022-01-03 to 2026-11-03 (1214) | BBG000BZX1N5 | Assurant, Inc. | BBG000BZX1N5 / 0001267238 | AIZ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AJG | 2022-01-03 to 2026-11-03 (1214) | BBG000BBHXQ3 | Arthur J. Gallagher & Co. | BBG000BBHXQ3 / 0000354190 | AJG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AKAM | 2022-01-03 to 2026-11-03 (1214) | BBG000BJQWD2 | Akamai Technologies Inc | BBG000BJQWD2 / 0001086222 | AKAM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ALB | 2022-01-03 to 2026-11-03 (1214) | BBG000BJ26K7 | Albemarle Corporation | BBG000BJ26K7 / 0000915913 | ALB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ALGN | 2022-01-03 to 2026-11-03 (1214) | BBG000BRNLL2 | Align Technology Inc | BBG000BRNLL2 / 0001097149 | ALGN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ALK | 2022-01-03 to 2023-12-15 (492) | BBG000BBL0Y1 | Alaska Air Group, Inc. | BBG000BBL0Y1 / 0000766421 | ALK 2022-01-03..2023-12-15 | panel 492 | records+events |
| ALL | 2022-01-03 to 2026-11-03 (1214) | BBG000BVMGF2 | The Allstate Corporation | BBG000BVMGF2 / 0000899051 | ALL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ALLE | 2022-01-03 to 2026-11-03 (1214) | CIK0001579241 | Allegion Public Limited Company | 0001579241 | ALLE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AMAT | 2022-01-03 to 2026-11-03 (1214) | BBG000BBPFB9 | Applied Materials Inc | BBG000BBPFB9 / 0000006951 | AMAT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AMCR | 2022-01-03 to 2026-11-03 (1214) | CIK0001748790 | Amcor plc Ordinary Shares | 0001748790 | AMCR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AMD | 2022-01-03 to 2026-11-03 (1214) | BBG000BBQCY0 | Advanced Micro Devices | BBG000BBQCY0 / 0000002488 | AMD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AME | 2022-01-03 to 2026-11-03 (1214) | BBG000B9XG87 | Ametek, Inc. | BBG000B9XG87 / 0001037868 | AME 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AMGN | 2022-01-03 to 2026-11-03 (1214) | BBG000BBS2Y0 | Amgen Inc | BBG000BBS2Y0 / 0000318154 | AMGN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AMP | 2022-01-03 to 2026-11-03 (1214) | BBG000G3QLY3 | Ameriprise Financial, Inc. | BBG000G3QLY3 / 0000820027 | AMP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AMT | 2022-01-03 to 2026-11-03 (1214) | BBG000B9XYV2 | American Tower Corporation | BBG000B9XYV2 / 0001053507 | AMT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AMTM | 2024-09-30 to 2024-12-20 (59) | BBG01NY7B734 | Amentum Holdings, Inc. | BBG01NY7B734 / 0002011286 | AMTM 2024-09-30..2024-12-20 | events 59 | events |
| AMZN | 2022-01-03 to 2026-11-03 (1214) | BBG000BVPV84 | Amazon.Com Inc | BBG000BVPV84 / 0001018724 | AMZN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ANET | 2022-01-03 to 2026-11-03 (1214) | BBG000N2HDY5 | Arista Networks | BBG000N2HDY5 / 0001596532 | ANET 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ANSS | 2022-01-03 to 2025-07-17 (887) | BBG000GXZ4W7 | Ansys Inc | BBG000GXZ4W7 / 0001013462 | ANSS 2022-01-03..2025-07-17 | panel 886; after its last price 1 | records |
| AON | 2022-01-03 to 2026-11-03 (1214) | CIK0000315293 | Aon plc Class A | 0000315293 | AON 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AOS | 2022-01-03 to 2026-11-03 (1214) | BBG000BC1L02 | A.O. Smith Corporation | BBG000BC1L02 / 0000091142 | AOS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| APA | 2022-01-03 to 2026-11-03 (1214) | BBG00YTS96G2 | APA Corporation Common Stock | BBG00YTS96G2 / 0001841666 | APA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| APD | 2022-01-03 to 2026-11-03 (1214) | BBG000BC4JJ4 | Air Products & Chemicals, Inc. | BBG000BC4JJ4 / 0000002969 | APD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| APH | 2022-01-03 to 2026-11-03 (1214) | BBG000B9YJ35 | Amphenol Corporation | BBG000B9YJ35 / 0000820313 | APH 2022-01-03..2026-11-03 | panel 1214 | records+events |
| APO | 2024-12-23 to 2026-11-03 (467) | BBG00ZNLTFK3 | Apollo Global Management, Inc. | BBG00ZNLTFK3 / 0001858681 | APO 2024-12-23..2026-11-03 | events 467 | events |
| APP | 2025-09-22 to 2026-11-03 (282) | BBG006HFPX77 | Applovin Corporation Class A Common Stock | BBG006HFPX77 / 0001751008 | APP 2025-09-22..2026-11-03 | events 282 | events |
| APTV | 2022-01-03 to 2026-11-03 (1214) | CIK0001521332 | Aptiv PLC | 0001521332 | APTV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ARE | 2022-01-03 to 2026-11-03 (1214) | BBG000BC33T9 | Alexandria Real Estate Equities, Inc. | BBG000BC33T9 / 0001035443 | ARE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ARES | 2025-12-11 to 2026-11-03 (225) | BBG0068KVV91 | Ares Management Corporation Class A Common Stock | BBG0068KVV91 / 0001176948 | ARES 2025-12-11..2026-11-03 | events 225 | events |
| ATO | 2022-01-03 to 2026-11-03 (1214) | BBG000BRNGM2 | Atmos Energy Corporation | BBG000BRNGM2 / 0000731802 | ATO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ATVI | 2022-01-03 to 2023-10-17 (450) | BBG000CVWGS6 | Activision Blizzard Inc. | BBG000CVWGS6 / 0000718877 | ATVI 2022-01-03..2023-10-17 | panel 447; after its last price 3 | records |
| AVB | 2022-01-03 to 2026-08-17 (1159) | BBG000BLPBL5 | AvalonBay Communities, Inc. | BBG000BLPBL5 / 0000915912 | AVB 2022-01-03..2026-08-17 | panel 1158; after its last price 1 | records |
| AVGO | 2022-01-03 to 2026-11-03 (1214) | BBG00KHY5S69 | Broadcom Inc. Common Stock | BBG00KHY5S69 / 0001730168 | AVGO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AVY | 2022-01-03 to 2026-11-03 (1214) | BBG000BCQ4P6 | Avery Dennison Corp. | BBG000BCQ4P6 / 0000008818 | AVY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AWK | 2022-01-03 to 2026-11-03 (1214) | BBG000TRJ294 | American Water Works Company, Inc | BBG000TRJ294 / 0001410636 | AWK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AXON | 2023-05-04 to 2026-11-03 (879) | BBG000BHJWG1 | Axon Enterprise, Inc. Common Stock | BBG000BHJWG1 / 0001069183 | AXON 2023-05-04..2026-11-03 | events 879 | events |
| AXP | 2022-01-03 to 2026-11-03 (1214) | BBG000BCQZS4 | American Express Company | BBG000BCQZS4 / 0000004962 | AXP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| AZO | 2022-01-03 to 2026-11-03 (1214) | BBG000C7LMS8 | AutoZone, Inc. | BBG000C7LMS8 / 0000866787 | AZO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BA | 2022-01-03 to 2026-11-03 (1214) | BBG000BCSST7 | Boeing Company | BBG000BCSST7 / 0000012927 | BA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BAC | 2022-01-03 to 2026-11-03 (1214) | BBG000BCTLF6 | Bank of America Corporation | BBG000BCTLF6 / 0000070858 | BAC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BALL | 2022-01-03 to 2026-11-03 (1214) | BBG000BDDNH5 | Ball Corporation | BBG000BDDNH5 / 0000009389 | BLL 2022-01-03..2022-05-09; BALL 2022-05-10..2026-11-03 | panel 1126; events 88 | records+events |
| BAX | 2022-01-03 to 2026-11-03 (1214) | BBG000BCVJ77 | Baxter International Inc. | BBG000BCVJ77 / 0000010456 | BAX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BBWI | 2022-01-03 to 2024-09-30 (689) | BBG000BNGTQ7 | Bath & Body Works, Inc. | BBG000BNGTQ7 / 0000701985 | BBWI 2022-01-03..2024-09-30 | panel 689 | records+events |
| BBY | 2022-01-03 to 2026-11-03 (1214) | BBG000BCWCG1 | Best Buy Company, Inc. | BBG000BCWCG1 / 0000764478 | BBY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BDX | 2022-01-03 to 2026-11-03 (1214) | BBG000BCZYD3 | Becton, Dickinson and Co. | BBG000BCZYD3 / 0000010795 | BDX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BE | 2026-09-21 to 2026-11-03 (32) | BBG000N7KBZ3 | Bloom Energy Corporation | BBG000N7KBZ3 / 0001664703 | BE 2026-09-21..2026-11-03 | events 32 | events |
| BEN | 2022-01-03 to 2026-11-03 (1214) | BBG000BD0TF8 | Franklin Templeton, Inc. | BBG000BD0TF8 / 0000038777 | BEN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BF.B | 2022-01-03 to 2026-11-03 (1214) | BBG000BD2NY8 | Brown-Forman Corporation Class B | BBG000BD2NY8 / 0000014693 | BF.B 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BG | 2023-03-15 to 2026-11-03 (914) | CIK0001996862 | Bunge Global SA | 0001996862 | BG 2023-11-01..2026-11-03 | events 754; unverified 160 | events |
| BIIB | 2022-01-03 to 2026-11-03 (1214) | BBG000C17X76 | Biogen Inc. Common Stock | BBG000C17X76 / 0000875045 | BIIB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BIO | 2022-01-03 to 2024-09-20 (683) | BBG000DY28W5 | Bio-Rad Laboratories, Inc.Class A | BBG000DY28W5 / 0000012208 | BIO 2022-01-03..2024-09-20 | panel 683 | records+events |
| BKNG | 2022-01-03 to 2026-11-03 (1214) | BBG000BLBVN4 | Booking Holdings Inc. Common Stock | BBG000BLBVN4 / 0001075531 | BKNG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BKR | 2022-01-03 to 2026-11-03 (1214) | BBG00GBVBK51 | Baker Hughes Company | BBG00GBVBK51 / 0001701605 | BKR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BLDR | 2023-12-18 to 2026-09-18 (690) | BBG000BKD3K9 | Builders FirstSource, Inc. | BBG000BKD3K9 / 0001316835 | BLDR 2023-12-18..2026-09-18 | events 690 | events |
| BLK | 2022-01-03 to 2026-11-03 (1214) | BBG01PSW2WN4 | Blackrock, Inc. | BBG01PSW2WN4 / 0002012383 | BLK 2022-01-03..2026-11-03 | predecessor 690; panel 524 | records+events |
| BMY | 2022-01-03 to 2026-11-03 (1214) | BBG000DQLV23 | Bristol-Myers Squibb Co. | BBG000DQLV23 / 0000014272 | BMY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BNY | 2022-01-03 to 2026-11-03 (1214) | BBG000BD8PN9 | Bank of New York Mellon Corporation | BBG000BD8PN9 / 0001390777 | BK 2022-01-03..2026-05-20; BNY 2026-05-21..2026-11-03 | events 1099; panel 115 | records+events |
| BR | 2022-01-03 to 2026-11-03 (1214) | BBG000PPFKQ7 | Broadridge Financial Solutions Inc | BBG000PPFKQ7 / 0001383312 | BR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BRK.B | 2022-01-03 to 2026-11-03 (1214) | BBG000DWG505 | BERKSHIRE HATHAWAY Class B | BBG000DWG505 / 0001067983 | BRK.B 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BRO | 2022-01-03 to 2026-11-03 (1214) | BBG000BWSGF4 | Brown & Brown, Inc. | BBG000BWSGF4 / 0000079282 | BRO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BSX | 2022-01-03 to 2026-11-03 (1214) | BBG000C0LW92 | Boston Scientific Corp. | BBG000C0LW92 / 0000885725 | BSX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| BWA | 2022-01-03 to 2025-03-21 (807) | BBG000BJ49H3 | BorgWarner Inc. | BBG000BJ49H3 / 0000908255 | BWA 2022-01-03..2025-03-21 | panel 807 | records+events |
| BX | 2023-09-18 to 2026-11-03 (786) | BBG000BH0106 | Blackstone Inc. | BBG000BH0106 / 0001393818 | BX 2023-09-18..2026-11-03 | events 786 | events |
| BXP | 2022-01-03 to 2026-11-03 (1214) | BBG000BS5CM9 | BXP, Inc. | BBG000BS5CM9 / 0001037540 | BXP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| C | 2022-01-03 to 2026-11-03 (1214) | BBG000FY4S11 | Citigroup Inc. | BBG000FY4S11 / 0000831001 | C 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CAG | 2022-01-03 to 2026-06-29 (1125) | BBG000BDXGP9 | Conagra Brands, Inc. | BBG000BDXGP9 / 0000023217 | CAG 2022-01-03..2026-06-29 | panel 1125 | records+events |
| CAH | 2022-01-03 to 2026-11-03 (1214) | BBG000D898T9 | Cardinal Health, Inc. | BBG000D898T9 / 0000721371 | CAH 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CARR | 2022-01-03 to 2026-11-03 (1214) | BBG00RP5HYS8 | Carrier Global Corporation | BBG00RP5HYS8 / 0001783180 | CARR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CASY | 2026-04-09 to 2026-11-03 (145) | BBG000BF0672 | Casey's General Stores Inc | BBG000BF0672 / 0000726958 | CASY 2026-04-09..2026-11-03 | presumed 145 | events |
| CAT | 2022-01-03 to 2026-11-03 (1214) | BBG000BF0K17 | Caterpillar Inc. | BBG000BF0K17 / 0000018230 | CAT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CB | 2022-01-03 to 2026-11-03 (1214) | CIK0000896159 | Chubb Limited | 0000896159 | CB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CBOE | 2022-01-03 to 2026-11-03 (1214) | BBG000QH56C1 | Cboe Global Markets, Inc. | BBG000QH56C1 / 0001374310 | CBOE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CBRE | 2022-01-03 to 2026-11-03 (1214) | BBG000C04224 | CBRE GROUP, INC. | BBG000C04224 / 0001138118 | CBRE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CCI | 2022-01-03 to 2026-11-03 (1214) | BBG000FV1Z23 | Crown Castle Inc. | BBG000FV1Z23 / 0001051470 | CCI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CCL | 2022-01-03 to 2026-11-03 (1214) | CIK0000815097 | Carnival Corporation Ltd. | 0000815097 | CCL 2022-01-03..2026-11-03 | predecessor 1089; panel 125 | records+events |
| CDAY | 2022-01-03 to 2024-01-31 (522) | BBG005D7PF34 | Ceridian HCM Holding | BBG005D7PF34 / 0001725057 | CDAY 2022-01-03..2024-01-31 | panel 522 | records |
| CDNS | 2022-01-03 to 2026-11-03 (1214) | BBG000C13CD9 | Cadence Design Systems | BBG000C13CD9 / 0000813672 | CDNS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CDW | 2022-01-03 to 2026-11-03 (1214) | BBG001P63B80 | CDW Corporation | BBG001P63B80 / 0001402057 | CDW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CE | 2022-01-03 to 2025-03-21 (807) | BBG000JYP7L8 | Celanese Corporation Common Stock | BBG000JYP7L8 / 0001306830 | CE 2022-01-03..2025-03-21 | panel 807 | records+events |
| CEG | 2022-02-02 to 2026-11-03 (1193) | BBG014KFRNP7 | Constellation Energy Corporation Common Stock | BBG014KFRNP7 / 0001868275 | CEG 2022-02-02..2026-11-03 | events 1193 | events |
| CERN | 2022-01-03 to 2022-06-07 (108) | BBG000BFDLV8 | Cerner Corp | BBG000BFDLV8 / 0000804753 | CERN 2022-01-03..2022-06-07 | panel 108 | records |
| CF | 2022-01-03 to 2026-11-03 (1214) | BBG000BWJFZ4 | CF Industries Holding, Inc. | BBG000BWJFZ4 / 0001324404 | CF 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CFG | 2022-01-03 to 2026-11-03 (1214) | BBG006Q0HY77 | Citizens Financial Group, Inc. | BBG006Q0HY77 / 0000759944 | CFG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CHD | 2022-01-03 to 2026-11-03 (1214) | BBG000BFJT36 | Church & Dwight Co., Inc. | BBG000BFJT36 / 0000313927 | CHD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CHRW | 2022-01-03 to 2026-11-03 (1214) | BBG000BTCH57 | C.H. Robinson Worldwide, Inc. | BBG000BTCH57 / 0001043277 | CHRW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CHTR | 2022-01-03 to 2026-11-03 (1214) | BBG000VPGNR2 | Charter Comm Inc Del CL A New | BBG000VPGNR2 / 0001091667 | CHTR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CI | 2022-01-03 to 2026-11-03 (1214) | BBG00KXRCDP0 | The Cigna Group | BBG00KXRCDP0 / 0001739940 | CI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CIEN | 2026-02-09 to 2026-11-03 (186) | BBG000BP1152 | Ciena Corporation | BBG000BP1152 / 0000936395 | CIEN 2026-02-09..2026-11-03 | events 186 | events |
| CINF | 2022-01-03 to 2026-11-03 (1214) | BBG000BFPK65 | Cincinnati Financial Corp | BBG000BFPK65 / 0000020286 | CINF 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CL | 2022-01-03 to 2026-11-03 (1214) | BBG000BFQYY3 | Colgate-Palmolive Company | BBG000BFQYY3 / 0000021665 | CL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CLX | 2022-01-03 to 2026-11-03 (1214) | BBG000BFS7D3 | Clorox Company | BBG000BFS7D3 / 0000021076 | CLX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CMA | 2022-01-03 to 2024-06-21 (620) | BBG000C75N77 | Comerica Incorporated | BBG000C75N77 / 0000028412 | CMA 2022-01-03..2024-06-21 | panel 620 | records |
| CMCSA | 2022-01-03 to 2026-11-03 (1214) | BBG000BFT2L4 | Comcast Corp | BBG000BFT2L4 / 0001166691 | CMCSA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CME | 2022-01-03 to 2026-11-03 (1214) | BBG000BHLYP4 | CME Group Inc. | BBG000BHLYP4 / 0001156375 | CME 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CMG | 2022-01-03 to 2026-11-03 (1214) | BBG000QX74T1 | Chipotle Mexican Grill, Inc. | BBG000QX74T1 / 0001058090 | CMG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CMI | 2022-01-03 to 2026-11-03 (1214) | BBG000BGPTV6 | Cummins Inc. | BBG000BGPTV6 / 0000026172 | CMI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CMS | 2022-01-03 to 2026-11-03 (1214) | BBG000BFVXX0 | CMS Energy Corporation | BBG000BFVXX0 / 0000201533 | CMS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CNC | 2022-01-03 to 2026-11-03 (1214) | BBG000BDXCJ5 | Centene Corporation | BBG000BDXCJ5 / 0001071739 | CNC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CNP | 2022-01-03 to 2026-11-03 (1214) | BBG000FDBX90 | CenterPoint Energy, Inc. | BBG000FDBX90 / 0001130310 | CNP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| COF | 2022-01-03 to 2026-11-03 (1214) | BBG000BGKTF9 | Capital One Financial | BBG000BGKTF9 / 0000927628 | COF 2022-01-03..2026-11-03 | panel 1214 | records+events |
| COHR | 2026-03-23 to 2026-11-03 (157) | BBG000BLW102 | Coherent Corp. | BBG000BLW102 / 0000820318 | COHR 2026-03-23..2026-11-03 | events 157 | events |
| COIN | 2025-05-19 to 2026-11-03 (368) | BBG00ZGF7771 | Coinbase Global, Inc. Class A Common Stock | BBG00ZGF7771 / 0001679788 | COIN 2025-05-19..2026-11-03 | events 368 | events |
| COO | 2022-01-03 to 2026-11-03 (1214) | BBG000BG3445 | The Cooper Companies, Inc. Common Stock | BBG000BG3445 / 0000711404 | COO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| COP | 2022-01-03 to 2026-11-03 (1214) | BBG000BQQH30 | ConocoPhillips | BBG000BQQH30 / 0001163165 | COP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| COR | 2022-01-03 to 2026-11-03 (1214) | BBG000MDCQC2 | Cencora, Inc. | BBG000MDCQC2 / 0001140859 | ABC 2022-01-03..2023-08-29; COR 2023-08-30..2026-11-03 | panel 798; events 416 | records+events |
| COST | 2022-01-03 to 2026-11-03 (1214) | BBG000F6H8W8 | Costco Wholesale Corp | BBG000F6H8W8 / 0000909832 | COST 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CPAY | 2022-01-03 to 2026-11-03 (1214) | BBG000GPXKX9 | Corpay, Inc. | BBG000GPXKX9 / 0001175454 | CPAY 2024-03-25..2026-11-03 | panel 656; unverified 558 | records+events |
| CPB | 2022-01-03 to 2026-06-18 (1119) | BBG000BG4202 | The Campbell's Company Common Stock | BBG000BG4202 / 0000016732 | CPB 2022-01-03..2026-06-18 | panel 1119 | records+events |
| CPRT | 2022-01-03 to 2026-11-03 (1214) | BBG000BM9RH1 | Copart Inc | BBG000BM9RH1 / 0000900075 | CPRT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CPT | 2022-04-04 to 2026-11-03 (1151) | BBG000BB0V03 | Camden Property Trust | BBG000BB0V03 / 0000906345 | CPT 2022-04-04..2026-11-03 | events 1151 | events |
| CRH | 2025-12-22 to 2026-11-03 (218) | CIK0000849395 | CRH Public Limited Company | 0000849395 | CRH 2025-12-22..2026-11-03 | events 218 | events |
| CRL | 2022-01-03 to 2026-11-03 (1214) | BBG000BG5BL8 | Charles River Laboratories International, Inc. | BBG000BG5BL8 / 0001100682 | CRL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CRM | 2022-01-03 to 2026-11-03 (1214) | BBG000BN2DC2 | Salesforce, Inc. | BBG000BN2DC2 / 0001108524 | CRM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CRWD | 2024-06-24 to 2026-11-03 (594) | BBG00BLYKS03 | CrowdStrike Holdings, Inc. Class A Common Stock | BBG00BLYKS03 / 0001535527 | CRWD 2024-06-24..2026-11-03 | events 594 | events |
| CSCO | 2022-01-03 to 2026-11-03 (1214) | BBG000C3J3C9 | Cisco Systems, Inc. Common Stock (DE) | BBG000C3J3C9 / 0000858877 | CSCO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CSGP | 2022-09-19 to 2026-11-03 (1036) | BBG000D7JKW9 | CoStar Group Inc | BBG000D7JKW9 / 0001057352 | CSGP 2022-09-19..2026-11-03 | events 1036 | events |
| CSX | 2022-01-03 to 2026-11-03 (1214) | BBG000BGJRC8 | CSX Corporation | BBG000BGJRC8 / 0000277948 | CSX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CTAS | 2022-01-03 to 2026-11-03 (1214) | BBG000H3YXF8 | Cintas Corp | BBG000H3YXF8 / 0000723254 | CTAS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CTLT | 2022-01-03 to 2024-12-20 (747) | BBG005XR47P5 | CATALENT, INC. | BBG005XR47P5 / 0001596783 | CTLT 2022-01-03..2024-12-20 | panel 744; after its last price 3 | records |
| CTRA | 2022-01-03 to 2026-05-06 (1089) | BBG000C3GN47 | Coterra Energy Inc. | BBG000C3GN47 / 0000858470 | CTRA 2022-01-03..2026-05-06 | panel 1089 | records |
| CTSH | 2022-01-03 to 2026-11-03 (1214) | BBG000BBDV81 | Cognizant Technology Solutions | BBG000BBDV81 / 0001058290 | CTSH 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CTVA | 2022-01-03 to 2026-10-05 (1193) | BBG00BN969C1 | Corteva, Inc. Common Stock | BBG00BN969C1 / 0001755672 | CTVA 2022-01-03..2026-10-05 | panel 1193 | records+events |
| CTXS | 2022-01-03 to 2022-09-30 (188) | BBG000FQ6PY6 | Citrix Systems Inc | BBG000FQ6PY6 / 0000877890 | CTXS 2022-01-03..2022-09-30 | panel 187; after its last price 1 | records |
| CVNA | 2025-12-22 to 2026-11-03 (218) | BBG00GCTWDJ3 | Carvana Co. | BBG00GCTWDJ3 / 0001690820 | CVNA 2025-12-22..2026-11-03 | events 218 | events |
| CVS | 2022-01-03 to 2026-11-03 (1214) | BBG000BGRY34 | CVS HEALTH CORPORATION | BBG000BGRY34 / 0000064803 | CVS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CVX | 2022-01-03 to 2026-11-03 (1214) | BBG000K4ND22 | Chevron Corporation | BBG000K4ND22 / 0000093410 | CVX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| CZR | 2022-01-03 to 2025-09-19 (932) | BBG0074Q3NK6 | Caesars Entertainment, Inc. Common Stock | BBG0074Q3NK6 / 0001590895 | CZR 2022-01-03..2025-09-19 | panel 932 | records+events |
| D | 2022-01-03 to 2026-11-03 (1214) | BBG000BGVW60 | Dominion Energy, Inc Common Stock | BBG000BGVW60 / 0000715957 | D 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DAL | 2022-01-03 to 2026-11-03 (1214) | BBG000R7Z112 | Delta Air Lines, Inc. | BBG000R7Z112 / 0000027904 | DAL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DASH | 2025-03-24 to 2026-11-03 (407) | BBG005D7QCJ3 | DoorDash, Inc. Class A Common Stock | BBG005D7QCJ3 / 0001792789 | DASH 2025-03-24..2026-11-03 | events 407 | events |
| DAY | 2024-02-01 to 2026-02-06 (506) | - | - | - | - | missing 506 | neither |
| DD | 2022-01-03 to 2026-11-03 (1214) | BBG00BN961G4 | DuPont de Nemours, Inc. Common Stock | BBG00BN961G4 / 0001666700 | DD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DDOG | 2025-07-09 to 2026-11-03 (334) | BBG003NJHZT9 | Datadog, Inc. Class A Common Stock | BBG003NJHZT9 / 0001561550 | DDOG 2025-07-09..2026-11-03 | events 334 | events |
| DE | 2022-01-03 to 2026-11-03 (1214) | BBG000BH1NH9 | Deere & Company | BBG000BH1NH9 / 0000315189 | DE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DECK | 2024-03-18 to 2026-11-03 (661) | BBG000BKXYX5 | Deckers Outdoor Corp | BBG000BKXYX5 / 0000910521 | DECK 2024-03-18..2026-11-03 | events 661 | events |
| DELL | 2024-09-23 to 2026-11-03 (531) | BBG00DW3SZS1 | Dell Technologies Inc. | BBG00DW3SZS1 / 0001571996 | DELL 2024-09-23..2026-11-03 | events 531 | events |
| DFS | 2022-01-03 to 2025-05-16 (846) | BBG000QBR5J5 | Discover Financial Services | BBG000QBR5J5 / 0001393612 | DFS 2022-01-03..2025-05-16 | panel 846 | records |
| DG | 2022-01-03 to 2026-11-03 (1214) | BBG000NV1KK7 | Dollar General Corp. | BBG000NV1KK7 / 0000029534 | DG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DGX | 2022-01-03 to 2026-11-03 (1214) | BBG000BN84F3 | Quest Diagnostics Inc. | BBG000BN84F3 / 0001022079 | DGX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DHI | 2022-01-03 to 2026-11-03 (1214) | BBG000DQTXY6 | D.R. Horton Inc. | BBG000DQTXY6 / 0000882184 | DHI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DHR | 2022-01-03 to 2026-11-03 (1214) | BBG000BH3JF8 | Danaher Corporation | BBG000BH3JF8 / 0000313616 | DHR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DIS | 2022-01-03 to 2026-11-03 (1214) | BBG000BH4R78 | The Walt Disney Company | BBG000BH4R78 / 0001744489 | DIS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DISCA | 2022-01-03 to 2022-04-08 (68) | BBG00HW3TQG4 | Discovery, Inc. Series A Common Stock | BBG00HW3TQG4 / 0001024333 | DISCA 2022-01-03..2022-04-08 | panel 68 | records |
| DISCK | 2022-01-03 to 2022-04-08 (68) | CIK0001024333 | Discovery, Inc. Series C Common Stock | 0001024333 | DISCK 2022-01-03..2022-04-08 | panel 68 | records |
| DISH | 2022-01-03 to 2023-06-16 (366) | BBG000C2YHG9 | DISH Network Corp. | BBG000C2YHG9 / 0001001082 | DISH 2022-01-03..2023-06-16 | panel 366 | records |
| DLR | 2022-01-03 to 2026-11-03 (1214) | BBG000Q5ZRM7 | Digital Realty Trust, Inc. | BBG000Q5ZRM7 / 0001297996 | DLR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DLTR | 2022-01-03 to 2026-11-03 (1214) | BBG000BSC0K9 | Dollar Tree Inc. | BBG000BSC0K9 / 0000935703 | DLTR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DOC | 2022-01-03 to 2026-11-03 (1214) | BBG000BKYDP9 | Healthpeak Properties, Inc. | BBG000BKYDP9 / 0000765880 | PEAK 2022-01-03..2024-03-01; DOC 2024-03-04..2026-11-03 | panel 671; events 543 | records+events |
| DOV | 2022-01-03 to 2026-11-03 (1214) | BBG000BHB3M6 | Dover Corporation | BBG000BHB3M6 / 0000029905 | DOV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DOW | 2022-01-03 to 2026-11-03 (1214) | BBG00BN96922 | Dow Inc. | BBG00BN96922 / 0001751788 | DOW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DPZ | 2022-01-03 to 2026-11-03 (1214) | BBG000P458P3 | Domino's Pizza Inc. | BBG000P458P3 / 0001286681 | DPZ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DRE | 2022-01-03 to 2022-09-30 (188) | BBG000BHD8S7 | Duke Realty Corp. | BBG000BHD8S7 / 0000783280 | DRE 2022-01-03..2022-09-30 | panel 188 | records |
| DRI | 2022-01-03 to 2026-11-03 (1214) | BBG000BBNYF6 | Darden Restaurants, Inc. | BBG000BBNYF6 / 0000940944 | DRI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DTE | 2022-01-03 to 2026-11-03 (1214) | BBG000BB29X4 | DTE Energy Company | BBG000BB29X4 / 0000936340 | DTE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DUK | 2022-01-03 to 2026-11-03 (1214) | BBG000BHGDH5 | Duke Energy Corporation | BBG000BHGDH5 / 0001326160 | DUK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DVA | 2022-01-03 to 2026-11-03 (1214) | BBG000MQ1SN9 | DaVita Inc. | BBG000MQ1SN9 / 0000927066 | DVA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DVN | 2022-01-03 to 2026-11-03 (1214) | BBG000BBVJZ8 | Devon Energy Corporation | BBG000BBVJZ8 / 0001090012 | DVN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| DXC | 2022-01-03 to 2023-10-02 (439) | BBG00FN64XT9 | DXC Technology Company | BBG00FN64XT9 / 0001688568 | DXC 2022-01-03..2023-10-02 | panel 439 | records+events |
| DXCM | 2022-01-03 to 2026-11-03 (1214) | BBG000QTF8K1 | DexCom, Inc. | BBG000QTF8K1 / 0001093557 | DXCM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EA | 2022-01-03 to 2026-08-04 (1150) | BBG000BP0KQ8 | Electronic Arts Inc | BBG000BP0KQ8 / 0000712515 | EA 2022-01-03..2026-08-04 | panel 1150 | records |
| EBAY | 2022-01-03 to 2026-11-03 (1214) | BBG000C43RR5 | eBay Inc | BBG000C43RR5 / 0001065088 | EBAY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ECHO | 2022-01-03 to 2026-11-03 (1214) | BBG000TGLV00 | EchoStar Corporation | BBG000TGLV00 / 0001415404 | SATS 2022-01-03..2026-06-23; ECHO 2026-06-24..2026-11-03 | events 1121; panel 93 | records+events |
| ECL | 2022-01-03 to 2026-11-03 (1214) | BBG000BHKYH4 | Ecolab, Inc. | BBG000BHKYH4 / 0000031462 | ECL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ED | 2022-01-03 to 2026-11-03 (1214) | BBG000BHLYS1 | Consolidated Edison, Inc. | BBG000BHLYS1 / 0001047862 | ED 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EFX | 2022-01-03 to 2026-11-03 (1214) | BBG000BHPL78 | Equifax, Incorporated | BBG000BHPL78 / 0000033185 | EFX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EG | 2022-01-03 to 2026-11-03 (1214) | CIK0001095073 | Everest Group, Ltd. | 0001095073 | RE 2022-01-03..2023-07-07; EG 2023-07-10..2026-11-03 | panel 835; events 379 | records+events |
| EIX | 2022-01-03 to 2026-11-03 (1214) | BBG000D7RKJ5 | Edison International | BBG000D7RKJ5 / 0000827052 | EIX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EL | 2022-01-03 to 2026-11-03 (1214) | BBG000FKJRC5 | The Estee Lauder Companies Inc. Class A | BBG000FKJRC5 / 0001001250 | EL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ELV | 2022-01-03 to 2026-11-03 (1214) | BBG000BCG930 | Elevance Health, Inc. | BBG000BCG930 / 0001156039 | ANTM 2022-01-03..2022-06-27; ELV 2022-06-28..2026-11-03 | panel 1093; events 121 | records+events |
| EME | 2025-09-22 to 2026-11-03 (282) | BBG000BBRPL4 | EMCOR Group, Inc. | BBG000BBRPL4 / 0000105634 | EME 2025-09-22..2026-11-03 | events 282 | events |
| EMN | 2022-01-03 to 2025-11-03 (963) | BBG000BLW530 | Eastman Chemical Company | BBG000BLW530 / 0000915389 | EMN 2022-01-03..2025-11-03 | panel 963 | records+events |
| EMR | 2022-01-03 to 2026-11-03 (1214) | BBG000BHX7N2 | Emerson Electric Co. | BBG000BHX7N2 / 0000032604 | EMR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ENPH | 2022-01-03 to 2025-09-19 (932) | BBG001R3MNY9 | Enphase Energy, Inc. | BBG001R3MNY9 / 0001463101 | ENPH 2022-01-03..2025-09-19 | panel 932 | records+events |
| EOG | 2022-01-03 to 2026-11-03 (1214) | BBG000BZ9223 | EOG Resources, Inc. | BBG000BZ9223 / 0000821189 | EOG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EPAM | 2022-01-03 to 2026-06-01 (1106) | BBG000MHTV89 | EPAM SYSTEMS, INC. | BBG000MHTV89 / 0001352010 | EPAM 2022-01-03..2026-06-01 | panel 1106 | records+events |
| EQIX | 2022-01-03 to 2026-11-03 (1214) | BBG000MBDGM6 | Equinix, Inc. Common Stock REIT | BBG000MBDGM6 / 0001101239 | EQIX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EQT | 2022-10-03 to 2026-11-03 (1026) | BBG000BHZ5J9 | EQT CORP | BBG000BHZ5J9 / 0000033213 | EQT 2022-10-03..2026-11-03 | events 1026 | events |
| ERIE | 2024-09-23 to 2026-11-03 (531) | BBG000BBTJZ1 | Erie Indemnity Co | BBG000BBTJZ1 / 0000922621 | ERIE 2024-09-23..2026-11-03 | events 531 | events |
| ES | 2022-01-03 to 2026-11-03 (1214) | BBG000BQ87N0 | Eversource Energy | BBG000BQ87N0 / 0000072741 | ES 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ESS | 2022-01-03 to 2026-11-03 (1214) | BBG000CGJMB9 | Essex Property Trust, Inc | BBG000CGJMB9 / 0000920522 | ESS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ETN | 2022-01-03 to 2026-11-03 (1214) | CIK0001551182 | Eaton Corporation, plc Ordinary Shares | 0001551182 | ETN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ETR | 2022-01-03 to 2026-11-03 (1214) | BBG000C1FQS9 | Entergy Corporation | BBG000C1FQS9 / 0000065984 | ETR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ETSY | 2022-01-03 to 2024-09-20 (683) | BBG000N7MXL8 | Etsy, Inc. | BBG000N7MXL8 / 0001370637 | ETSY 2022-01-03..2024-09-20 | panel 683 | records+events |
| EVRG | 2022-01-03 to 2026-11-03 (1214) | BBG00H433CR2 | Evergy, Inc. | BBG00H433CR2 / 0001711269 | EVRG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EW | 2022-01-03 to 2026-11-03 (1214) | BBG000BRXP69 | Edwards Lifesciences Corp | BBG000BRXP69 / 0001099800 | EW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EXC | 2022-01-03 to 2026-11-03 (1214) | BBG000J6XT05 | Exelon Corporation | BBG000J6XT05 / 0001109357 | EXC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EXE | 2025-03-24 to 2026-11-03 (407) | BBG00Z6DX554 | Expand Energy Corporation Common Stock | BBG00Z6DX554 / 0000895126 | EXE 2025-03-24..2026-11-03 | events 407 | events |
| EXPD | 2022-01-03 to 2026-11-03 (1214) | BBG000BJ5GK2 | Expeditors International of Washington, Inc. | BBG000BJ5GK2 / 0000746515 | EXPD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EXPE | 2022-01-03 to 2026-11-03 (1214) | BBG000QY3XZ2 | Expedia Group, Inc. Common Stock | BBG000QY3XZ2 / 0001324424 | EXPE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| EXR | 2022-01-03 to 2026-11-03 (1214) | BBG000PV27K3 | Extra Space Storage, Inc. | BBG000PV27K3 / 0001289490 | EXR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| F | 2022-01-03 to 2026-11-03 (1214) | BBG000BQPC32 | Ford Motor Company | BBG000BQPC32 / 0000037996 | F 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FANG | 2022-01-03 to 2026-11-03 (1214) | BBG002PHSYX9 | Diamondback Energy, Inc. | BBG002PHSYX9 / 0001539838 | FANG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FAST | 2022-01-03 to 2026-11-03 (1214) | BBG000BJ8YN7 | Fastenal Co | BBG000BJ8YN7 / 0000815556 | FAST 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FBHS | 2022-01-03 to 2022-12-16 (242) | BBG001B4BV87 | FORTUNE BRANDS HOME & SECURITY, INC. | BBG001B4BV87 / 0001519751 | FBHS 2022-01-03..2022-12-14; FBIN 2022-12-15..2022-12-16 | panel 240; events 2 | records+events |
| FCX | 2022-01-03 to 2026-11-03 (1214) | BBG000BJDB15 | Freeport-McMoran Inc. | BBG000BJDB15 / 0000831259 | FCX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FDS | 2022-01-03 to 2026-11-03 (1214) | BBG000BDJL83 | Factset Research Systems | BBG000BDJL83 / 0001013237 | FDS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FDX | 2022-01-03 to 2026-11-03 (1214) | BBG000BJF1Z8 | FedEx Corporation | BBG000BJF1Z8 / 0001048911 | FDX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FDXF | 2026-06-01 to 2026-11-03 (109) | BBG01ZPB91C4 | FedEx Freight Holding Company, Inc. | BBG01ZPB91C4 / 0002082247 | FDXF 2026-06-01..2026-11-03 | events 109 | events |
| FE | 2022-01-03 to 2026-11-03 (1214) | BBG000BB6M98 | FirstEnergy Corp. | BBG000BB6M98 / 0001031296 | FE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FERG | 2026-08-05 to 2026-11-03 (64) | BBG01NMRD768 | Ferguson Enterprises Inc. | BBG01NMRD768 / 0002011641 | FERG 2026-08-05..2026-11-03 | events 64 | events |
| FFIV | 2022-01-03 to 2026-11-03 (1214) | BBG000CXYSZ6 | F5, Inc. Common Stock | BBG000CXYSZ6 / 0001048695 | FFIV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FICO | 2023-03-20 to 2026-11-03 (911) | BBG000DW76Y6 | Fair Isaac Corporation | BBG000DW76Y6 / 0000814547 | FICO 2023-03-20..2026-11-03 | events 911 | events |
| FIS | 2022-01-03 to 2026-11-03 (1214) | BBG000BK2F42 | Fidelity National Information Services, Inc. | BBG000BK2F42 / 0001136893 | FIS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FISV | 2022-01-03 to 2026-11-03 (1214) | BBG000BJKPG0 | Fiserv, Inc. Common Stock | BBG000BJKPG0 / 0000798354 | FISV 2022-01-03..2023-06-06; FISV 2025-11-11..2026-11-03; FI 2023-06-07..2025-11-10 | events 610; panel 604 | records+events |
| FITB | 2022-01-03 to 2026-11-03 (1214) | BBG000BJL3N0 | Fifth Third Bancorp | BBG000BJL3N0 / 0000035527 | FITB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FIX | 2025-12-22 to 2026-11-03 (218) | BBG000BWN2Y9 | Comfort Systems USA, Inc. | BBG000BWN2Y9 / 0001035983 | FIX 2025-12-22..2026-11-03 | events 218 | events |
| FLEX | 2026-06-22 to 2026-11-03 (95) | CIK0000866374 | Flex Ltd. Ordinary Shares | 0000866374 | FLEX 2026-06-22..2026-11-03 | events 95 | events |
| FMC | 2022-01-03 to 2025-03-21 (807) | BBG000BJP882 | FMC Corporation | BBG000BJP882 / 0000037785 | FMC 2022-01-03..2025-03-21 | panel 807 | records+events |
| FOX | 2022-01-03 to 2026-11-03 (1214) | BBG00JHNKJY8 | Fox Corporation Class B Common Stock | BBG00JHNKJY8 / 0001754301 | FOX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FOXA | 2022-01-03 to 2026-11-03 (1214) | BBG00JHNJW99 | Fox Corporation Class A Common Stock | BBG00JHNJW99 / 0001754301 | FOXA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FRC | 2022-01-03 to 2023-05-03 (335) | BBG0019LSQ49 | First Republic Bank | BBG0019LSQ49 / 0001132979 | FRC 2022-01-03..2023-05-02; FRCB 2023-05-03..2023-05-03 | panel 332; inside a gap in its prices 2; events 1 | records+events |
| FRT | 2022-01-03 to 2026-11-03 (1214) | BBG000BJVN28 | Federal Realty Investment Trust | BBG000BJVN28 / 0000034903 | FRT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FSLR | 2022-12-19 to 2026-11-03 (972) | BBG000BFL116 | First Solar, Inc. | BBG000BFL116 / 0001274494 | FSLR 2022-12-19..2026-11-03 | events 972 | events |
| FTNT | 2022-01-03 to 2026-11-03 (1214) | BBG000BCMBG4 | Fortinet, Inc. | BBG000BCMBG4 / 0001262039 | FTNT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| FTV | 2022-01-03 to 2026-11-03 (1214) | BBG00BLVZ228 | Fortive Corporation | BBG00BLVZ228 / 0001659166 | FTV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GD | 2022-01-03 to 2026-11-03 (1214) | BBG000BK67C7 | General Dynamics Corporation | BBG000BK67C7 / 0000040533 | GD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GDDY | 2024-06-24 to 2026-11-03 (594) | BBG006MDLY05 | GoDaddy Inc | BBG006MDLY05 / 0001609711 | GDDY 2024-06-24..2026-11-03 | events 594 | events |
| GE | 2022-01-03 to 2026-11-03 (1214) | BBG000BK6MB5 | GE Aerospace | BBG000BK6MB5 / 0000040545 | GE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GEHC | 2023-01-04 to 2026-11-03 (962) | BBG01BFR8YV1 | GE HealthCare Technologies Inc. Common Stock | BBG01BFR8YV1 / 0001932393 | GEHC 2023-01-04..2026-11-03 | events 962 | events |
| GEN | 2022-01-03 to 2026-11-03 (1214) | BBG000BH2JM1 | Gen Digital Inc. Common Stock | BBG000BH2JM1 / 0000849399 | NLOK 2022-01-03..2022-11-07; GEN 2022-11-08..2026-11-03 | panel 1000; events 214 | records+events |
| GEV | 2024-04-02 to 2026-11-03 (651) | BBG013G17W31 | GE Vernova Inc. | BBG013G17W31 / 0001996810 | GEV 2024-04-02..2026-11-03 | events 651 | events |
| GILD | 2022-01-03 to 2026-11-03 (1214) | BBG000CKGBP2 | Gilead Sciences Inc | BBG000CKGBP2 / 0000882095 | GILD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GIS | 2022-01-03 to 2026-11-03 (1214) | BBG000BKCFC2 | General Mills, Inc. | BBG000BKCFC2 / 0000040704 | GIS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GL | 2022-01-03 to 2026-11-03 (1214) | BBG000BVD6X4 | Globe Life Inc. | BBG000BVD6X4 / 0000320335 | GL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GLW | 2022-01-03 to 2026-11-03 (1214) | BBG000BKFZM4 | Corning Incorporated | BBG000BKFZM4 / 0000024741 | GLW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GM | 2022-01-03 to 2026-11-03 (1214) | BBG000NDYB67 | General Motors Company | BBG000NDYB67 / 0001467858 | GM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GNRC | 2022-01-03 to 2026-11-03 (1214) | BBG000PQ4W72 | GENERAC HOLDINGS INC | BBG000PQ4W72 / 0001474735 | GNRC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GOOG | 2022-01-03 to 2026-11-03 (1214) | BBG009S3NB30 | Alphabet Inc. Class C Capital Stock | BBG009S3NB30 / 0001652044 | GOOG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GOOGL | 2022-01-03 to 2026-11-03 (1214) | BBG009S39JX6 | Alphabet Inc. Class A Common Stock | BBG009S39JX6 / 0001652044 | GOOGL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GPC | 2022-01-03 to 2026-11-03 (1214) | BBG000BKL348 | Genuine Parts Company | BBG000BKL348 / 0000040987 | GPC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GPN | 2022-01-03 to 2026-11-03 (1214) | BBG000CX0P89 | Global Payments, Inc. | BBG000CX0P89 / 0001123360 | GPN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GPS | 2022-01-03 to 2022-02-02 (22) | BBG000BKLH74 | The Gap, Inc. | BBG000BKLH74 / 0000039911 | GPS 2022-01-03..2022-02-02 | panel 22 | records+events |
| GRMN | 2022-01-03 to 2026-11-03 (1214) | CIK0001121788 | Garmin Ltd | 0001121788 | GRMN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GS | 2022-01-03 to 2026-11-03 (1214) | BBG000C6CFJ5 | Goldman Sachs Group Inc. | BBG000C6CFJ5 / 0000886982 | GS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| GWW | 2022-01-03 to 2026-11-03 (1214) | BBG000BKR1D6 | W.W. Grainger, Inc. | BBG000BKR1D6 / 0000277135 | GWW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HAL | 2022-01-03 to 2026-11-03 (1214) | BBG000BKTFN2 | Halliburton Company | BBG000BKTFN2 / 0000045012 | HAL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HAS | 2022-01-03 to 2026-11-03 (1214) | BBG000BKVJK4 | Hasbro, Inc. | BBG000BKVJK4 / 0000046080 | HAS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HBAN | 2022-01-03 to 2026-11-03 (1214) | BBG000BKWSR6 | Huntington Bancshares Inc | BBG000BKWSR6 / 0000049196 | HBAN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HCA | 2022-01-03 to 2026-11-03 (1214) | BBG000QW7VC1 | HCA Healthcare, Inc. | BBG000QW7VC1 / 0000860730 | HCA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HD | 2022-01-03 to 2026-11-03 (1214) | BBG000BKZB36 | Home Depot, Inc. | BBG000BKZB36 / 0000354950 | HD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HES | 2022-01-03 to 2025-07-22 (890) | BBG000BBD070 | Hess Corporation | BBG000BBD070 / 0000004447 | HES 2022-01-03..2025-07-22 | panel 887; after its last price 3 | records |
| HIG | 2022-01-03 to 2026-11-03 (1214) | BBG000G0Z878 | The Hartford Insurance Group, Inc. | BBG000G0Z878 / 0000874766 | HIG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HII | 2022-01-03 to 2026-11-03 (1214) | BBG001KJ2HM9 | Huntington Ingalls Industries, Inc. | BBG001KJ2HM9 / 0001501585 | HII 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HLT | 2022-01-03 to 2026-11-03 (1214) | BBG0058KMH30 | Hilton Worldwide Holdings Inc. | BBG0058KMH30 / 0001585689 | HLT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HOLX | 2022-01-03 to 2026-04-08 (1069) | BBG000C3NTN5 | Hologic Inc | BBG000C3NTN5 / 0000859737 | HOLX 2022-01-03..2026-04-08 | panel 1067; after its last price 2 | records |
| HON | 2022-01-03 to 2026-11-03 (1214) | BBG000H556T9 | Honeywell International, Inc. | BBG000H556T9 / 0000773840 | HON 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HONA | 2026-06-29 to 2026-11-03 (90) | BBG020QYV152 | Honeywell Aerospace Inc. Common Stock | BBG020QYV152 / 0002089271 | HONA 2026-06-29..2026-11-03 | events 90 | events |
| HOOD | 2025-09-22 to 2026-11-03 (282) | BBG008NMBXN8 | Robinhood Markets, Inc. Class A Common Stock | BBG008NMBXN8 / 0001783879 | HOOD 2025-09-22..2026-11-03 | events 282 | events |
| HPE | 2022-01-03 to 2026-11-03 (1214) | BBG0078W3NQ3 | Hewlett Packard Enterprise Company | BBG0078W3NQ3 / 0001645590 | HPE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HPQ | 2022-01-03 to 2026-11-03 (1214) | BBG000KHWT55 | HP Inc. | BBG000KHWT55 / 0000047217 | HPQ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HRL | 2022-01-03 to 2026-11-03 (1214) | BBG000BLF8D2 | Hormel Foods Corporation | BBG000BLF8D2 / 0000048465 | HRL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HSIC | 2022-01-03 to 2026-11-03 (1214) | BBG000BNMMJ3 | Henry Schein Inc | BBG000BNMMJ3 / 0001000228 | HSIC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HST | 2022-01-03 to 2026-11-03 (1214) | BBG000BL8804 | Host Hotels & Resorts, Inc. | BBG000BL8804 / 0001070750 | HST 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HSY | 2022-01-03 to 2026-11-03 (1214) | BBG000BLHRS2 | The Hershey Company | BBG000BLHRS2 / 0000047111 | HSY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HUBB | 2023-10-18 to 2026-11-03 (764) | BBG000BLK267 | Hubbell Incorporated | BBG000BLK267 / 0000048898 | HUBB 2023-10-18..2026-11-03 | events 764 | events |
| HUM | 2022-01-03 to 2026-11-03 (1214) | BBG000BLKK03 | Humana Inc. | BBG000BLKK03 / 0000049071 | HUM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| HWM | 2022-01-03 to 2026-11-03 (1214) | BBG00DYNJGH9 | Howmet Aerospace Inc. | BBG00DYNJGH9 / 0000004281 | HWM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IBKR | 2025-08-28 to 2026-11-03 (298) | BBG000LV0836 | Interactive Brokers Group, Inc. Class A Common Stock | BBG000LV0836 / 0001381197 | IBKR 2025-08-28..2026-11-03 | events 298 | events |
| IBM | 2022-01-03 to 2026-11-03 (1214) | BBG000BLNNH6 | International Business Machines Corporation | BBG000BLNNH6 / 0000051143 | IBM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ICE | 2022-01-03 to 2026-11-03 (1214) | BBG000C1FB75 | Intercontinental Exchange  Inc. | BBG000C1FB75 / 0001571949 | ICE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IDXX | 2022-01-03 to 2026-11-03 (1214) | BBG000BLRT07 | Idexx Laboratories Inc | BBG000BLRT07 / 0000874716 | IDXX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IEX | 2022-01-03 to 2026-11-03 (1214) | BBG000C1HN22 | IDEX Corporation | BBG000C1HN22 / 0000832101 | IEX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IFF | 2022-01-03 to 2026-11-03 (1214) | BBG000BLSL58 | International Flavors & Fragrances Inc. | BBG000BLSL58 / 0000051253 | IFF 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ILMN | 2022-01-03 to 2024-06-21 (620) | BBG000DSMS70 | Illumina Inc | BBG000DSMS70 / 0001110803 | ILMN 2022-01-03..2024-06-21 | panel 620 | records+events |
| ILMN | 2026-09-21 to 2026-11-03 (32) | BBG000DSMS70 | Illumina Inc | BBG000DSMS70 / 0001110803 | ILMN 2026-09-21..2026-11-03 | panel 32 | records+events |
| INCY | 2022-01-03 to 2026-11-03 (1214) | BBG000BNPSQ9 | Incyte Genomics Inc | BBG000BNPSQ9 / 0000879169 | INCY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| INFO | 2022-01-03 to 2022-03-01 (40) | CIK0001316360 | IHS Markit Ltd. Common Shares | 0001316360 | INFO 2022-01-03..2022-03-01 | panel 38; after its last price 2 | records |
| INTC | 2022-01-03 to 2026-11-03 (1214) | BBG000C0G1D1 | Intel Corp | BBG000C0G1D1 / 0000050863 | INTC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| INTU | 2022-01-03 to 2026-11-03 (1214) | BBG000BH5DV1 | Intuit Inc | BBG000BH5DV1 / 0000896878 | INTU 2022-01-03..2026-11-03 | panel 1214 | records+events |
| INVH | 2022-09-19 to 2026-11-03 (1036) | BBG00FQH6BS9 | Invitation Homes Inc. Common Stock | BBG00FQH6BS9 / 0001687229 | INVH 2022-09-19..2026-11-03 | events 1036 | events |
| IP | 2022-01-03 to 2026-11-03 (1214) | BBG000BM5SR2 | International Paper Co. | BBG000BM5SR2 / 0000051434 | IP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IPG | 2022-01-03 to 2025-11-26 (980) | BBG000C90DH9 | The Interpublic Group of Companies, Inc. | BBG000C90DH9 / 0000051644 | IPG 2022-01-03..2025-11-26 | panel 980 | records |
| IPGP | 2022-01-03 to 2022-06-17 (116) | BBG000DCGRL8 | IPG Photonics Corporation | BBG000DCGRL8 / 0001111928 | IPGP 2022-01-03..2022-06-17 | panel 116 | records+events |
| IQV | 2022-01-03 to 2026-11-03 (1214) | BBG00333FYS2 | IQVIA Holdings Inc. | BBG00333FYS2 / 0001478242 | IQV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IR | 2022-01-03 to 2026-11-03 (1214) | BBG002R1CW27 | Ingersoll Rand Inc. Common Stock | BBG002R1CW27 / 0001699150 | IR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IRM | 2022-01-03 to 2026-11-03 (1214) | BBG000KCZPC3 | Iron Mountain Inc. | BBG000KCZPC3 / 0001020569 | IRM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ISRG | 2022-01-03 to 2026-11-03 (1214) | BBG000BJPDZ1 | Intuitive Surgical Inc. | BBG000BJPDZ1 / 0001035267 | ISRG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IT | 2022-01-03 to 2026-11-03 (1214) | BBG000BB65D0 | Gartner, Inc. | BBG000BB65D0 / 0000749251 | IT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ITW | 2022-01-03 to 2026-11-03 (1214) | BBG000BMBL90 | Illinois Tool Works Inc. | BBG000BMBL90 / 0000049826 | ITW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| IVZ | 2022-01-03 to 2026-11-03 (1214) | CIK0000914208 | Invesco LTD | 0000914208 | IVZ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| J | 2022-01-03 to 2026-11-03 (1214) | BBG019C1BQR4 | Jacobs Solutions Inc. | BBG019C1BQR4 / 0000052988 | J 2022-01-03..2026-11-03 | panel 1050; predecessor 164 | records+events |
| JBHT | 2022-01-03 to 2026-11-03 (1214) | BBG000BMDBZ1 | JB Hunt Transport Services Inc | BBG000BMDBZ1 / 0000728535 | JBHT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| JBL | 2023-12-18 to 2026-11-03 (722) | BBG000BJNGN9 | Jabil Inc. | BBG000BJNGN9 / 0000898293 | JBL 2023-12-18..2026-11-03 | events 722 | events |
| JCI | 2022-01-03 to 2026-11-03 (1214) | CIK0000833444 | Johnson Controls International plc | 0000833444 | JCI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| JKHY | 2022-01-03 to 2026-11-03 (1214) | BBG000BMH2Q7 | Henry (Jack) & Associates | BBG000BMH2Q7 / 0000779152 | JKHY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| JNJ | 2022-01-03 to 2026-11-03 (1214) | BBG000BMHYD1 | Johnson & Johnson | BBG000BMHYD1 / 0000200406 | JNJ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| JNPR | 2022-01-03 to 2025-07-08 (880) | BBG000BY33P5 | Juniper Networks Inc | BBG000BY33P5 / 0001043604 | JNPR 2022-01-03..2025-07-08 | panel 876; after its last price 4 | records |
| JPM | 2022-01-03 to 2026-11-03 (1214) | BBG000DMBXR2 | JPMorgan Chase & Co. | BBG000DMBXR2 / 0000019617 | JPM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| K | 2022-01-03 to 2025-12-10 (989) | BBG000BMKDM3 | Kellanova | BBG000BMKDM3 / 0000055067 | K 2022-01-03..2025-12-10 | panel 989 | records |
| KDP | 2022-06-21 to 2026-11-03 (1098) | BBG000TJM7F0 | Keurig Dr Pepper Inc. | BBG000TJM7F0 / 0001418135 | KDP 2022-06-21..2026-11-03 | events 1098 | events |
| KEY | 2022-01-03 to 2026-11-03 (1214) | BBG000BMQPL1 | KeyCorp | BBG000BMQPL1 / 0000091576 | KEY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KEYS | 2022-01-03 to 2026-11-03 (1214) | BBG0059FN811 | Keysight Technologies, Inc. | BBG0059FN811 / 0001601046 | KEYS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KHC | 2022-01-03 to 2026-11-03 (1214) | BBG005CPNTQ2 | The Kraft Heinz Company | BBG005CPNTQ2 / 0001637459 | KHC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KIM | 2022-01-03 to 2026-11-03 (1214) | BBG000CN3S73 | Kimco Realty Corp. | BBG000CN3S73 / 0000879101 | KIM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KKR | 2024-06-24 to 2026-11-03 (594) | BBG000BCQ6J8 | KKR & Co. Inc. | BBG000BCQ6J8 / 0001404912 | KKR 2024-06-24..2026-11-03 | events 594 | events |
| KLAC | 2022-01-03 to 2026-11-03 (1214) | BBG000BMTFR4 | KLA Corporation Common Stock | BBG000BMTFR4 / 0000319201 | KLAC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KMB | 2022-01-03 to 2026-11-03 (1214) | BBG000BMW2Z0 | Kimberly-Clark Corp. | BBG000BMW2Z0 / 0000055785 | KMB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KMI | 2022-01-03 to 2026-11-03 (1214) | BBG0019JZ882 | Kinder Morgan, Inc. | BBG0019JZ882 / 0001506307 | KMI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KMX | 2022-01-03 to 2025-10-30 (961) | BBG000BLMZK6 | CarMax Inc. | BBG000BLMZK6 / 0001170010 | KMX 2022-01-03..2025-10-30 | panel 961 | records+events |
| KO | 2022-01-03 to 2026-11-03 (1214) | BBG000BMX289 | Coca-Cola Company | BBG000BMX289 / 0000021344 | KO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KR | 2022-01-03 to 2026-11-03 (1214) | BBG000BMY992 | The Kroger Co. | BBG000BMY992 / 0000056873 | KR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| KVUE | 2023-08-25 to 2026-10-29 (798) | - | - | - | - | missing 798 | neither |
| L | 2022-01-03 to 2026-11-03 (1214) | BBG000C45984 | Loews Corporation | BBG000C45984 / 0000060086 | L 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LDOS | 2022-01-03 to 2026-11-03 (1214) | BBG000C23PB0 | Leidos Holdings, Inc. | BBG000C23PB0 / 0001336920 | LDOS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LEN | 2022-01-03 to 2026-11-03 (1214) | BBG000BN5HF7 | Lennar Corporation Class A | BBG000BN5HF7 / 0000920760 | LEN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LH | 2022-01-03 to 2026-11-03 (1214) | BBG01MMT6PL7 | Labcorp Holdings Inc. | BBG01MMT6PL7 / 0000920148 | LH 2022-01-03..2026-11-03 | panel 617; predecessor 597 | records+events |
| LHX | 2022-01-03 to 2026-11-03 (1214) | BBG000BLGFJ9 | L3Harris Technologies, Inc. | BBG000BLGFJ9 / 0000202058 | LHX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LII | 2024-12-23 to 2026-11-03 (467) | BBG000BB5B84 | Lennox International Inc. | BBG000BB5B84 / 0001069202 | LII 2024-12-23..2026-11-03 | events 467 | events |
| LIN | 2022-01-03 to 2026-11-03 (1214) | CIK0001707925 | Linde plc Ordinary Share | 0001707925 | LIN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LITE | 2026-03-23 to 2026-11-03 (157) | BBG0073F9RT7 | Lumentum Holdings Inc. Common Stock | BBG0073F9RT7 / 0001633978 | LITE 2026-03-23..2026-11-03 | events 157 | events |
| LKQ | 2022-01-03 to 2025-12-19 (996) | BBG000PXDL44 | LKQ Corporation | BBG000PXDL44 / 0001065696 | LKQ 2022-01-03..2025-12-19 | panel 996 | records+events |
| LLY | 2022-01-03 to 2026-11-03 (1214) | BBG000BNBDC2 | Eli Lilly & Co. | BBG000BNBDC2 / 0000059478 | LLY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LMT | 2022-01-03 to 2026-11-03 (1214) | BBG000C1BW00 | Lockheed Martin Corp. | BBG000C1BW00 / 0000936468 | LMT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LNC | 2022-01-03 to 2023-09-15 (428) | BBG000BNC3Y9 | Lincoln National Corp. | BBG000BNC3Y9 / 0000059558 | LNC 2022-01-03..2023-09-15 | panel 428 | records+events |
| LNT | 2022-01-03 to 2026-11-03 (1214) | BBG000DW4Q75 | Alliant Energy Corporation Common Stock | BBG000DW4Q75 / 0000352541 | LNT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LOW | 2022-01-03 to 2026-11-03 (1214) | BBG000BNDN65 | Lowe's Companies Inc. | BBG000BNDN65 / 0000060667 | LOW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LRCX | 2022-01-03 to 2026-11-03 (1214) | BBG000BNFLM9 | Lam Research Corp | BBG000BNFLM9 / 0000707549 | LRCX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LULU | 2023-10-18 to 2026-11-03 (764) | BBG000R8ZVD1 | lululemon athletica inc. | BBG000R8ZVD1 / 0001397187 | LULU 2023-10-18..2026-11-03 | events 764 | events |
| LUMN | 2022-01-03 to 2023-03-17 (303) | BBG000BGLRN3 | Lumen Technologies, Inc. Common Stock | BBG000BGLRN3 / 0000018926 | LUMN 2022-01-03..2023-03-17 | panel 303 | records+events |
| LUV | 2022-01-03 to 2026-11-03 (1214) | BBG000BNJHS8 | Southwest Airlines Co. | BBG000BNJHS8 / 0000092380 | LUV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LVS | 2022-01-03 to 2026-11-03 (1214) | BBG000JWD753 | Las Vegas Sands Corp. | BBG000JWD753 / 0001300514 | LVS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LW | 2022-01-03 to 2026-03-20 (1057) | BBG003CVMLQ2 | Lamb Weston Holdings, Inc. | BBG003CVMLQ2 / 0001679273 | LW 2022-01-03..2026-03-20 | panel 1057 | records+events |
| LYB | 2022-01-03 to 2026-11-03 (1214) | CIK0001489393 | LyondellBasell Industries N.V. Class A | 0001489393 | LYB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| LYV | 2022-01-03 to 2026-11-03 (1214) | BBG000FQ7YR4 | Live Nation Entertainment Inc. | BBG000FQ7YR4 / 0001335258 | LYV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MA | 2022-01-03 to 2026-11-03 (1214) | BBG000F1ZSQ2 | Mastercard Incorporated | BBG000F1ZSQ2 / 0001141391 | MA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MAA | 2022-01-03 to 2026-11-03 (1214) | BBG000BLMY92 | Mid-America Apartment Communities, Inc. | BBG000BLMY92 / 0000912595 | MAA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MAR | 2022-01-03 to 2026-11-03 (1214) | BBG000BGD7W6 | Marriott International Class A Common Stock | BBG000BGD7W6 / 0001048286 | MAR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MAS | 2022-01-03 to 2026-11-03 (1214) | BBG000BNNKG9 | Masco Corporation | BBG000BNNKG9 / 0000062996 | MAS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MBC | 2022-12-15 to 2022-12-16 (2) | BBG01B9G42Y3 | MasterBrand, Inc. | BBG01B9G42Y3 / 0001941365 | MBC 2022-12-15..2022-12-16 | events 2 | events |
| MCD | 2022-01-03 to 2026-11-03 (1214) | BBG000BNSZP1 | McDonald's Corporation | BBG000BNSZP1 / 0000063908 | MCD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MCHP | 2022-01-03 to 2026-11-03 (1214) | BBG000BHCP19 | Microchip Technology Inc | BBG000BHCP19 / 0000827054 | MCHP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MCK | 2022-01-03 to 2026-11-03 (1214) | BBG000DYGNW7 | McKesson Corporation | BBG000DYGNW7 / 0000927653 | MCK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MCO | 2022-01-03 to 2026-11-03 (1214) | BBG000F86GP6 | Moody's Corporation | BBG000F86GP6 / 0001059556 | MCO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MDLZ | 2022-01-03 to 2026-11-03 (1214) | BBG000D4LWF6 | Mondelez International, Inc. Class A | BBG000D4LWF6 / 0001103982 | MDLZ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MDT | 2022-01-03 to 2026-11-03 (1214) | CIK0001613103 | Medtronic plc | 0001613103 | MDT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MET | 2022-01-03 to 2026-11-03 (1214) | BBG000BB6KF5 | MetLife, Inc. | BBG000BB6KF5 / 0001099219 | MET 2022-01-03..2026-11-03 | panel 1214 | records+events |
| META | 2022-01-03 to 2026-11-03 (1214) | BBG000MM2P62 | Meta Platforms, Inc. Class A Common Stock | BBG000MM2P62 / 0001326801 | FB 2022-01-03..2022-06-08; META 2022-06-09..2026-11-03 | panel 1105; events 109 | records+events |
| MGM | 2022-01-03 to 2026-11-03 (1214) | BBG000C2BXK4 | MGM RESORTS INTERNATIONAL | BBG000C2BXK4 / 0000789570 | MGM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MHK | 2022-01-03 to 2025-12-19 (996) | BBG000FLZN98 | Mohawk Industries, Inc. | BBG000FLZN98 / 0000851968 | MHK 2022-01-03..2025-12-19 | panel 996 | records+events |
| MKC | 2022-01-03 to 2026-11-03 (1214) | BBG000G6Y5W4 | McCormick & Company, Incorporated Non-VTG CS | BBG000G6Y5W4 / 0000063754 | MKC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MKTX | 2022-01-03 to 2025-09-19 (932) | BBG000BJBZ23 | MarketAxess Holdings Inc. | BBG000BJBZ23 / 0001278021 | MKTX 2022-01-03..2025-09-19 | panel 932 | records+events |
| MLM | 2022-01-03 to 2026-11-03 (1214) | BBG000BGYMH7 | Martin Marietta Materials | BBG000BGYMH7 / 0000916076 | MLM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MMM | 2022-01-03 to 2026-11-03 (1214) | BBG000BP52R2 | 3M Company | BBG000BP52R2 / 0000066740 | MMM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MNST | 2022-01-03 to 2026-11-03 (1214) | BBG008NVB1C0 | Monster Beverage Corporation | BBG008NVB1C0 / 0000865752 | MNST 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MO | 2022-01-03 to 2026-11-03 (1214) | BBG000BP6LJ8 | Altria Group, Inc. | BBG000BP6LJ8 / 0000764180 | MO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MOH | 2022-03-02 to 2026-03-20 (1017) | BBG000MBHNC8 | Molina Healthcare, Inc. | BBG000MBHNC8 / 0001179929 | MOH 2022-03-02..2026-03-20 | events 1017 | events |
| MOS | 2022-01-03 to 2026-11-03 (1214) | BBG000BFXHL6 | The Mosaic Company | BBG000BFXHL6 / 0001285785 | MOS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MPC | 2022-01-03 to 2026-11-03 (1214) | BBG001DCCGR8 | MARATHON PETROLEUM CORPORATION | BBG001DCCGR8 / 0001510295 | MPC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MPWR | 2022-01-03 to 2026-11-03 (1214) | BBG000C30L48 | Monolithic Power Systems, Inc. | BBG000C30L48 / 0001280452 | MPWR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MRK | 2022-01-03 to 2026-11-03 (1214) | BBG000BPD168 | Merck & Co., Inc. | BBG000BPD168 / 0000310158 | MRK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MRNA | 2022-01-03 to 2026-11-03 (1214) | BBG003PHHZT1 | Moderna, Inc. Common Stock | BBG003PHHZT1 / 0001682852 | MRNA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MRO | 2022-01-03 to 2024-11-25 (729) | BBG000C8H633 | Marathon Oil Corporation | BBG000C8H633 / 0000101778 | MRO 2022-01-03..2024-11-25 | panel 727; after its last price 2 | records |
| MRSH | 2022-01-03 to 2026-11-03 (1214) | BBG000BP4MH0 | Marsh | BBG000BP4MH0 / 0000062709 | MMC 2022-01-03..2026-01-13; MRSH 2026-01-14..2026-11-03 | events 1011; panel 203 | records+events |
| MRVL | 2026-06-22 to 2026-11-03 (95) | BBG00ZXBJ153 | Marvell Technology, Inc. Common Stock | BBG00ZXBJ153 / 0001835632 | MRVL 2026-06-22..2026-11-03 | events 95 | events |
| MS | 2022-01-03 to 2026-11-03 (1214) | BBG000BLZRJ2 | Morgan Stanley | BBG000BLZRJ2 / 0000895421 | MS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MSCI | 2022-01-03 to 2026-11-03 (1214) | BBG000RTDY25 | MSCI, Inc. | BBG000RTDY25 / 0001408198 | MSCI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MSFT | 2022-01-03 to 2026-11-03 (1214) | BBG000BPH459 | Microsoft Corp | BBG000BPH459 / 0000789019 | MSFT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MSI | 2022-01-03 to 2026-11-03 (1214) | BBG000BP8Z50 | Motorola Solutions, Inc. New | BBG000BP8Z50 / 0000068505 | MSI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MTB | 2022-01-03 to 2026-11-03 (1214) | BBG000D9KWL9 | M&T Bank Corp. | BBG000D9KWL9 / 0000036270 | MTB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MTCH | 2022-01-03 to 2026-03-20 (1057) | BBG00VT0KNC3 | Match Group, Inc | BBG00VT0KNC3 / 0000891103 | MTCH 2022-01-03..2026-03-20 | panel 1057 | records+events |
| MTD | 2022-01-03 to 2026-11-03 (1214) | BBG000BZCKH3 | Mettler-Toledo International | BBG000BZCKH3 / 0001037646 | MTD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| MU | 2022-01-03 to 2026-11-03 (1214) | BBG000C5Z1S3 | Micron Technology, Inc. | BBG000C5Z1S3 / 0000723125 | MU 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NCLH | 2022-01-03 to 2026-11-03 (1214) | CIK0001513761 | Norwegian Cruise Line Holdings Ltd. Ordinary Shares | 0001513761 | NCLH 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NDAQ | 2022-01-03 to 2026-11-03 (1214) | BBG000F5VVB6 | Nasdaq, Inc. Common Stock | BBG000F5VVB6 / 0001120193 | NDAQ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NDSN | 2022-02-15 to 2026-11-03 (1184) | BBG000BPWGR1 | Nordson Corp | BBG000BPWGR1 / 0000072331 | NDSN 2022-02-15..2026-11-03 | events 1184 | events |
| NEE | 2022-01-03 to 2026-11-03 (1214) | BBG000BJSBJ0 | NextEra Energy, Inc. | BBG000BJSBJ0 / 0000753308 | NEE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NEM | 2022-01-03 to 2026-11-03 (1214) | BBG000BPWXK1 | Newmont Corporation | BBG000BPWXK1 / 0001164727 | NEM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NFLX | 2022-01-03 to 2026-11-03 (1214) | BBG000CL9VN6 | NetFlix Inc | BBG000CL9VN6 / 0001065280 | NFLX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NI | 2022-01-03 to 2026-11-03 (1214) | BBG000BPZBB6 | NiSource Inc. | BBG000BPZBB6 / 0001111711 | NI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NKE | 2022-01-03 to 2026-11-03 (1214) | BBG000C5HS04 | Nike, Inc. | BBG000C5HS04 / 0000320187 | NKE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NLSN | 2022-01-03 to 2022-10-11 (195) | CIK0001492633 | Nielsen Holdings plc | 0001492633 | NLSN 2022-01-03..2022-10-11 | panel 195 | records |
| NOC | 2022-01-03 to 2026-11-03 (1214) | BBG000BQ2C28 | Northrop Grumman Corp. | BBG000BQ2C28 / 0001133421 | NOC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NOW | 2022-01-03 to 2026-11-03 (1214) | BBG000M1R011 | SERVICENOW, INC. | BBG000M1R011 / 0001373715 | NOW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NRG | 2022-01-03 to 2026-11-03 (1214) | BBG000D8RG11 | NRG Energy, Inc. | BBG000D8RG11 / 0001013871 | NRG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NSC | 2022-01-03 to 2026-11-03 (1214) | BBG000BQ5DS5 | Norfolk Southern Corp. | BBG000BQ5DS5 / 0000702165 | NSC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NTAP | 2022-01-03 to 2026-11-03 (1214) | BBG000FP1N32 | NetApp, Inc | BBG000FP1N32 / 0001002047 | NTAP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NTRS | 2022-01-03 to 2026-11-03 (1214) | BBG000BQ74K1 | Northern Trust Corp | BBG000BQ74K1 / 0000073124 | NTRS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NUE | 2022-01-03 to 2026-11-03 (1214) | BBG000BQ8KV2 | Nucor Corporation | BBG000BQ8KV2 / 0000073309 | NUE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NVDA | 2022-01-03 to 2026-11-03 (1214) | BBG000BBJQV0 | Nvidia Corp | BBG000BBJQV0 / 0001045810 | NVDA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NVR | 2022-01-03 to 2026-11-03 (1214) | BBG000BQBYR3 | NVR, Inc. | BBG000BQBYR3 / 0000906163 | NVR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NWL | 2022-01-03 to 2023-09-15 (428) | BBG000BQC9V2 | Newell Brands Inc. | BBG000BQC9V2 / 0000814453 | NWL 2022-01-03..2023-09-15 | panel 428 | records+events |
| NWS | 2022-01-03 to 2026-11-03 (1214) | BBG0035M2ZB7 | News Corporation Class B Common Stock | BBG0035M2ZB7 / 0001564708 | NWS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NWSA | 2022-01-03 to 2026-11-03 (1214) | BBG0035LY913 | News Corporation Class A Common Stock | BBG0035LY913 / 0001564708 | NWSA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| NXPI | 2022-01-03 to 2026-11-03 (1214) | CIK0001413447 | NXP Semiconductors N.V. | 0001413447 | NXPI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| O | 2022-01-03 to 2026-11-03 (1214) | BBG000DHPN63 | Realty Income Corporation | BBG000DHPN63 / 0000726728 | O 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ODFL | 2022-01-03 to 2026-11-03 (1214) | BBG000CHSS88 | Old Dominion Freight Line | BBG000CHSS88 / 0000878927 | ODFL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| OGN | 2022-01-03 to 2023-10-17 (450) | BBG00ZQRGW24 | Organon & Co. | BBG00ZQRGW24 / 0001821825 | OGN 2022-01-03..2023-10-17 | panel 450 | records+events |
| OKE | 2022-01-03 to 2026-11-03 (1214) | BBG024TZWVN1 | Oneok, Inc. | BBG024TZWVN1 / 0001039684 | OKE 2022-01-03..2026-11-03 | predecessor 1176; panel 38 | records+events |
| OKTA | 2026-10-30 to 2026-11-03 (3) | BBG001YV1SM4 | Okta, Inc. Class A Common Stock | BBG001YV1SM4 / 0001660134 | OKTA 2026-10-30..2026-11-03 | events 3 | events |
| OMC | 2022-01-03 to 2026-11-03 (1214) | BBG000BS9489 | Omnicom Group Inc. | BBG000BS9489 / 0000029989 | OMC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ON | 2022-06-21 to 2026-11-03 (1098) | BBG000DV7MX4 | ON Semiconductor Corp | BBG000DV7MX4 / 0001097864 | ON 2022-06-21..2026-11-03 | events 1098 | events |
| ORCL | 2022-01-03 to 2026-11-03 (1214) | BBG000BQLTW7 | Oracle Corp | BBG000BQLTW7 / 0001341439 | ORCL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ORLY | 2022-01-03 to 2026-11-03 (1214) | BBG000BGYWY6 | O'Reilly Automotive, Inc. | BBG000BGYWY6 / 0000898173 | ORLY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| OTIS | 2022-01-03 to 2026-11-03 (1214) | BBG00RP60KV0 | Otis Worldwide Corporation | BBG00RP60KV0 / 0001781335 | OTIS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| OXY | 2022-01-03 to 2026-11-03 (1214) | BBG000BQQ2S6 | Occidental Petroleum Corporation | BBG000BQQ2S6 / 0000797468 | OXY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| P | 2026-09-21 to 2026-11-03 (32) | BBG00212PVZ5 | Everpure, Inc. | BBG00212PVZ5 / 0001474432 | P 2026-09-21..2026-11-03 | events 32 | events |
| PANW | 2023-06-20 to 2026-11-03 (848) | BBG0014GJCT9 | Palo Alto Networks, Inc. Common Stock | BBG0014GJCT9 / 0001327567 | PANW 2023-06-20..2026-11-03 | events 848 | events |
| PAYC | 2022-01-03 to 2026-03-20 (1057) | BBG0064N0ZZ5 | PAYCOM SOFTWARE, INC. | BBG0064N0ZZ5 / 0001590955 | PAYC 2022-01-03..2026-03-20 | panel 1057 | records+events |
| PAYX | 2022-01-03 to 2026-11-03 (1214) | BBG000BQSQ38 | Paychex Inc | BBG000BQSQ38 / 0000723531 | PAYX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PBCT | 2022-01-03 to 2022-04-01 (63) | BBG008B8NDF2 | People's United Financial Inc | BBG008B8NDF2 / 0001378946 | PBCT 2022-01-03..2022-04-01 | panel 63 | records |
| PCAR | 2022-01-03 to 2026-11-03 (1214) | BBG000BQVTF5 | Paccar Inc | BBG000BQVTF5 / 0000075362 | PCAR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PCG | 2022-10-03 to 2026-11-03 (1026) | BBG000BQWPC5 | PG&E Corporation | BBG000BQWPC5 / 0001004980 | PCG 2022-10-03..2026-11-03 | events 1026 | events |
| PEG | 2022-01-03 to 2026-11-03 (1214) | BBG000BQZMH4 | Public Service Enterprise Group Incorporated | BBG000BQZMH4 / 0000788784 | PEG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PENN | 2022-01-03 to 2022-09-16 (178) | BBG000CDZLV8 | PENN Entertainment, Inc. Common Stock | BBG000CDZLV8 / 0000921738 | PENN 2022-01-03..2022-09-16 | panel 178 | records+events |
| PEP | 2022-01-03 to 2026-11-03 (1214) | BBG000DH7JK6 | PepsiCo, Inc. | BBG000DH7JK6 / 0000077476 | PEP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PFE | 2022-01-03 to 2026-11-03 (1214) | BBG000BR2B91 | Pfizer Inc. | BBG000BR2B91 / 0000078003 | PFE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PFG | 2022-01-03 to 2026-11-03 (1214) | BBG000NSCNT7 | Principal Financial Group, Inc. | BBG000NSCNT7 / 0001126328 | PFG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PG | 2022-01-03 to 2026-11-03 (1214) | BBG000BR2TH3 | Procter & Gamble Company | BBG000BR2TH3 / 0000080424 | PG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PGR | 2022-01-03 to 2026-11-03 (1214) | BBG000BR37X2 | Progressive Corporation | BBG000BR37X2 / 0000080661 | PGR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PH | 2022-01-03 to 2026-11-03 (1214) | BBG000BR3KL6 | Parker-Hannifin Corporation | BBG000BR3KL6 / 0000076334 | PH 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PHM | 2022-01-03 to 2026-11-03 (1214) | BBG000BR54L0 | Pultegroup, Inc. | BBG000BR54L0 / 0000822416 | PHM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PKG | 2022-01-03 to 2026-11-03 (1214) | BBG000BB8SW7 | Packaging Corp of America | BBG000BB8SW7 / 0000075677 | PKG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PLD | 2022-01-03 to 2026-11-03 (1214) | BBG000B9Z0J8 | PROLOGIS, INC. | BBG000B9Z0J8 / 0001045609 | PLD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PLTR | 2024-09-23 to 2026-11-03 (531) | BBG000N7QR55 | Palantir Technologies Inc. Class A Common Stock | BBG000N7QR55 / 0001321655 | PLTR 2024-09-23..2026-11-03 | events 531 | events |
| PM | 2022-01-03 to 2026-11-03 (1214) | BBG000J2XL74 | Philip Morris International Inc. | BBG000J2XL74 / 0001413329 | PM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PNC | 2022-01-03 to 2026-11-03 (1214) | BBG000BRD0D8 | PNC Financial Services Group | BBG000BRD0D8 / 0000713676 | PNC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PNR | 2022-01-03 to 2026-11-03 (1214) | CIK0000077360 | Pentair plc | 0000077360 | PNR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PNW | 2022-01-03 to 2026-11-03 (1214) | BBG000BRDSX5 | Pinnacle West Capital Corporation | BBG000BRDSX5 / 0000764622 | PNW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PODD | 2023-03-15 to 2026-11-03 (914) | BBG000R7XX87 | Insulet Corporation | BBG000R7XX87 / 0001145197 | PODD 2023-03-15..2026-11-03 | events 914 | events |
| POOL | 2022-01-03 to 2026-06-18 (1119) | BBG000BCVG28 | Pool Corporation | BBG000BCVG28 / 0000945841 | POOL 2022-01-03..2026-06-18 | panel 1119 | records+events |
| PPG | 2022-01-03 to 2026-11-03 (1214) | BBG000BRJ809 | PPG Industries, Inc. | BBG000BRJ809 / 0000079879 | PPG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PPL | 2022-01-03 to 2026-11-03 (1214) | BBG000BRJL00 | PPL Corporation | BBG000BRJL00 / 0000922224 | PPL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PRU | 2022-01-03 to 2026-11-03 (1214) | BBG000HCJMF9 | Prudential Financial, Inc. | BBG000HCJMF9 / 0001137774 | PRU 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PSA | 2022-01-03 to 2026-11-03 (1214) | BBG000BPPN67 | Public Storage | BBG000BPPN67 / 0001393311 | PSA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PSX | 2022-01-03 to 2026-11-03 (1214) | BBG00286S4N9 | PHILLIPS 66 | BBG00286S4N9 / 0001534701 | PSX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PTC | 2022-01-03 to 2026-11-03 (1214) | BBG000C2VBB0 | PTC, INC | BBG000C2VBB0 / 0000857005 | PTC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PVH | 2022-01-03 to 2022-09-16 (178) | BBG000BRRG02 | PVH Corp. | BBG000BRRG02 / 0000078239 | PVH 2022-01-03..2022-09-16 | panel 178 | records+events |
| PWR | 2022-01-03 to 2026-11-03 (1214) | BBG000BBL8V7 | Quanta Services, Inc. | BBG000BBL8V7 / 0001050915 | PWR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| PXD | 2022-01-03 to 2024-05-07 (589) | BBG000BXRPH1 | Pioneer Natural Resource Co. | BBG000BXRPH1 / 0001038357 | PXD 2022-01-03..2024-05-07 | panel 586; after its last price 3 | records |
| PYPL | 2022-01-03 to 2026-11-03 (1214) | BBG0077VNXV6 | PayPal Holdings, Inc. Common Stock | BBG0077VNXV6 / 0001633917 | PYPL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| Q | 2025-11-03 to 2026-11-03 (252) | BBG01TRL73S5 | Qnity Electronics, Inc. | BBG01TRL73S5 / 0002058873 | Q 2025-11-03..2026-11-03 | events 252 | events |
| QCOM | 2022-01-03 to 2026-11-03 (1214) | BBG000CGC1X8 | Qualcomm Inc | BBG000CGC1X8 / 0000804328 | QCOM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| QRVO | 2022-01-03 to 2024-12-20 (747) | BBG007TJF1N7 | Qorvo, Inc. | BBG007TJF1N7 / 0001604778 | QRVO 2022-01-03..2024-12-20 | panel 747 | records |
| RCL | 2022-01-03 to 2026-11-03 (1214) | CIK0000884887 | Royal Caribbean Group | 0000884887 | RCL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| RDDT | 2026-08-18 to 2026-11-03 (55) | BBG005K1D4S0 | Reddit, Inc. | BBG005K1D4S0 / 0001713445 | RDDT 2026-08-18..2026-11-03 | events 55 | events |
| REG | 2022-01-03 to 2026-11-03 (1214) | BBG000BL46Q4 | Regency Centers Corporation | BBG000BL46Q4 / 0000910606 | REG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| REGN | 2022-01-03 to 2026-11-03 (1214) | BBG000C734W3 | Regeneron Pharmaceuticals Inc | BBG000C734W3 / 0000872589 | REGN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| RF | 2022-01-03 to 2026-11-03 (1214) | BBG000Q3JN03 | Regions Financial Corp. | BBG000Q3JN03 / 0001281761 | RF 2022-01-03..2026-11-03 | panel 1214 | records+events |
| RHI | 2022-01-03 to 2024-06-21 (620) | BBG000BS5DR2 | Robert Half Inc. | BBG000BS5DR2 / 0000315213 | RHI 2022-01-03..2024-06-21 | panel 620 | records+events |
| RJF | 2022-01-03 to 2026-11-03 (1214) | BBG000BS73J1 | Raymond James Financial, Inc. | BBG000BS73J1 / 0000720005 | RJF 2022-01-03..2026-11-03 | panel 1214 | records+events |
| RL | 2022-01-03 to 2026-11-03 (1214) | BBG000BS0ZF1 | Ralph Lauren Corporation | BBG000BS0ZF1 / 0001037038 | RL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| RMD | 2022-01-03 to 2026-11-03 (1214) | BBG000L4M7F1 | ResMed Inc. | BBG000L4M7F1 / 0000943819 | RMD 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ROK | 2022-01-03 to 2026-11-03 (1214) | BBG000BBCDZ2 | Rockwell Automation, Inc. | BBG000BBCDZ2 / 0001024478 | ROK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ROL | 2022-01-03 to 2026-11-03 (1214) | BBG000BSBBP1 | Rollins, Inc. | BBG000BSBBP1 / 0000084839 | ROL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ROP | 2022-01-03 to 2026-11-03 (1214) | BBG000F1ZSN5 | Roper Technologies, Inc. Common Stock | BBG000F1ZSN5 / 0000882835 | ROP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ROST | 2022-01-03 to 2026-11-03 (1214) | BBG000BSBZH7 | Ross Stores Inc | BBG000BSBZH7 / 0000745732 | ROST 2022-01-03..2026-11-03 | panel 1214 | records+events |
| RSG | 2022-01-03 to 2026-11-03 (1214) | BBG000BPXVJ6 | Republic Services Inc. | BBG000BPXVJ6 / 0001060391 | RSG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| RTX | 2022-01-03 to 2026-11-03 (1214) | BBG000BW8S60 | RTX Corporation | BBG000BW8S60 / 0000101829 | RTX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| RVTY | 2022-01-03 to 2026-11-03 (1214) | BBG000FXW512 | Revvity, Inc. | BBG000FXW512 / 0000031791 | PKI 2022-01-03..2023-05-15; RVTY 2023-05-16..2026-11-03 | panel 871; events 343 | records+events |
| SBAC | 2022-01-03 to 2026-11-03 (1214) | BBG000D2M0Z7 | SBA Communications Corp | BBG000D2M0Z7 / 0001034054 | SBAC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SBNY | 2022-01-03 to 2023-03-14 (300) | BBG000M6TR37 | SIGNATURE BANK (NY) | BBG000M6TR37 | SBNY 2022-01-03..2023-03-14 | panel 298; inside a gap in its prices 2 | records+events |
| SBUX | 2022-01-03 to 2026-11-03 (1214) | BBG000CTQBF3 | Starbucks Corp | BBG000CTQBF3 / 0000829224 | SBUX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SCHW | 2022-01-03 to 2026-11-03 (1214) | BBG000BSLZY7 | The Charles Schwab Corporation | BBG000BSLZY7 / 0000316709 | SCHW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SEDG | 2022-01-03 to 2023-12-15 (492) | BBG0084BBZY6 | SolarEdge Technologies, Inc. | BBG0084BBZY6 / 0001419612 | SEDG 2022-01-03..2023-12-15 | panel 492 | records+events |
| SEE | 2022-01-03 to 2023-12-15 (492) | BBG000C22QV7 | Sealed Air Corp. | BBG000C22QV7 / 0001012100 | SEE 2022-01-03..2023-12-15 | panel 492 | records |
| SHW | 2022-01-03 to 2026-11-03 (1214) | BBG000BSXQV7 | The Sherwin-Williams Company | BBG000BSXQV7 / 0000089800 | SHW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SIVB | 2022-01-03 to 2023-03-14 (300) | BBG000BT0CM2 | SVB Financial Group | BBG000BT0CM2 / 0000719739 | SIVB 2022-01-03..2023-03-14 | panel 297; after its last price 3 | records |
| SJM | 2022-01-03 to 2026-11-03 (1214) | BBG000BT1715 | The J.M. Smucker Company | BBG000BT1715 / 0000091419 | SJM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SKYD | 2022-01-03 to 2026-11-03 (1214) | BBG01VS5NK99 | Skydance Corporation | BBG01VS5NK99 / 0002041610 | PSKY 2025-08-07..2026-10-05; SKYD 2026-10-06..2026-11-03 | unverified 901; events 292; panel 21 | records+events |
| SLB | 2022-01-03 to 2026-11-03 (1214) | BBG000BT41Q8 | SLB Limited | BBG000BT41Q8 / 0000087347 | SLB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SMCI | 2024-03-18 to 2026-11-03 (661) | BBG000MYZDJ3 | Super Micro Computer, Inc. Common Stock | BBG000MYZDJ3 / 0001375365 | SMCI 2024-10-01..2026-11-03 | events 525; unverified 136 | events |
| SNA | 2022-01-03 to 2026-11-03 (1214) | BBG000BT7JW9 | Snap-on Incorporated | BBG000BT7JW9 / 0000091440 | SNA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SNDK | 2025-11-28 to 2026-11-03 (234) | BBG01R388JG1 | Sandisk Corporation Common Stock | BBG01R388JG1 / 0002023554 | SNDK 2025-11-28..2026-11-03 | events 234 | events |
| SNPS | 2022-01-03 to 2026-11-03 (1214) | BBG000BSFRF3 | Synopsys Inc | BBG000BSFRF3 / 0000883241 | SNPS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SO | 2022-01-03 to 2026-11-03 (1214) | BBG000BT9DW0 | The Southern Company | BBG000BT9DW0 / 0000092122 | SO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SOLS | 2025-10-30 to 2025-12-19 (36) | BBG01X5744P5 | Solstice Advanced Materials Inc. Common Stock | BBG01X5744P5 / 0002064953 | SOLS 2025-10-30..2025-12-19 | events 36 | events |
| SOLV | 2024-04-01 to 2026-11-03 (652) | BBG018YZH6T3 | Solventum Corporation | BBG018YZH6T3 / 0001964738 | SOLV 2024-04-01..2026-11-03 | events 652 | events |
| SPG | 2022-01-03 to 2026-11-03 (1214) | BBG000BJ2D31 | Simon Property Group, Inc. | BBG000BJ2D31 / 0001063761 | SPG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SPGI | 2022-01-03 to 2026-11-03 (1214) | BBG000BP1Q11 | S&P Global Inc. | BBG000BP1Q11 / 0000064040 | SPGI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SRE | 2022-01-03 to 2026-11-03 (1214) | BBG000C2ZCH8 | Sempra | BBG000C2ZCH8 / 0001032208 | SRE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| STE | 2022-01-03 to 2026-11-03 (1214) | CIK0001757898 | STERIS plc | 0001757898 | STE 2022-01-03..2026-11-03 | panel 1214 | records+events |
| STLD | 2022-12-22 to 2026-11-03 (969) | BBG000HGYNZ9 | Steel Dynamics Inc | BBG000HGYNZ9 / 0001022671 | STLD 2022-12-22..2026-11-03 | events 969 | events |
| STT | 2022-01-03 to 2026-11-03 (1214) | BBG000BKFBD7 | State Street Corporation | BBG000BKFBD7 / 0000093751 | STT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| STX | 2022-01-03 to 2026-11-03 (1214) | CIK0001137789 | Seagate Technology Holdings PLC Ordinary Shares (Ireland) | 0001137789 | STX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| STZ | 2022-01-03 to 2026-11-03 (1214) | BBG000J1QLT0 | Constellation Brands, Inc. | BBG000J1QLT0 / 0000016918 | STZ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SW | 2022-01-03 to 2026-11-03 (1214) | CIK0002005951 | Smurfit Westrock plc | 0002005951 | SW 2024-07-08..2026-11-03 | unverified 629; panel 585 | records+events |
| SWK | 2022-01-03 to 2026-11-03 (1214) | BBG000BTQR96 | Stanley Black & Decker, Inc. | BBG000BTQR96 / 0000093556 | SWK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SWKS | 2022-01-03 to 2026-11-03 (1214) | BBG000KLB4Q1 | Skyworks Solutions Inc | BBG000KLB4Q1 / 0000004127 | SWKS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SYF | 2022-01-03 to 2026-11-03 (1214) | BBG00658F3P3 | SYNCHRONY FINANCIAL | BBG00658F3P3 / 0001601712 | SYF 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SYK | 2022-01-03 to 2026-11-03 (1214) | BBG000DN7P92 | Stryker Corporation | BBG000DN7P92 / 0000310764 | SYK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| SYY | 2022-01-03 to 2026-11-03 (1214) | BBG000BTVJ25 | Sysco Corporation | BBG000BTVJ25 / 0000096021 | SYY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| T | 2022-01-03 to 2026-11-03 (1214) | BBG000BSJK37 | AT&T Inc. | BBG000BSJK37 / 0000732717 | T 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TAP | 2022-01-03 to 2026-09-18 (1182) | BBG000BS7KS3 | Molson Coors Beverage Company Class B | BBG000BS7KS3 / 0000024545 | TAP 2022-01-03..2026-09-18 | panel 1182 | records+events |
| TDG | 2022-01-03 to 2026-11-03 (1214) | BBG000L8CBX4 | TransDigm Group Incorporated | BBG000L8CBX4 / 0001260221 | TDG 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TDY | 2022-01-03 to 2026-11-03 (1214) | BBG000BMT9T6 | Teledyne Technologies Incorporated | BBG000BMT9T6 / 0001094285 | TDY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TECH | 2022-01-03 to 2026-11-03 (1214) | BBG000C15114 | Bio-Techne Corp. | BBG000C15114 / 0000842023 | TECH 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TEL | 2022-01-03 to 2026-11-03 (1214) | CIK0001385157 | TE Connectivity plc | 0001385157 | TEL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TER | 2022-01-03 to 2026-11-03 (1214) | BBG000BV4DR6 | Teradyne, Inc. Common Stock | BBG000BV4DR6 / 0000097210 | TER 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TFC | 2022-01-03 to 2026-11-03 (1214) | BBG000BYYLS8 | Truist Financial Corporation | BBG000BYYLS8 / 0000092230 | TFC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TFX | 2022-01-03 to 2025-03-21 (807) | BBG000BV59Y6 | Teleflex Incorporated | BBG000BV59Y6 / 0000096943 | TFX 2022-01-03..2025-03-21 | panel 807 | records+events |
| TGT | 2022-01-03 to 2026-11-03 (1214) | BBG000H8TVT2 | Target Corporation | BBG000H8TVT2 / 0000027419 | TGT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TJX | 2022-01-03 to 2026-11-03 (1214) | BBG000BV8DN6 | TJX Companies, Inc. (The) | BBG000BV8DN6 / 0000109198 | TJX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TKO | 2025-03-24 to 2026-11-03 (407) | BBG01G9JKWV5 | TKO Group Holdings, Inc. | BBG01G9JKWV5 / 0001973266 | TKO 2025-03-24..2026-11-03 | events 407 | events |
| TMO | 2022-01-03 to 2026-11-03 (1214) | BBG000BVDLH9 | Thermo Fisher Scientific, Inc. | BBG000BVDLH9 / 0000097745 | TMO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TMUS | 2022-01-03 to 2026-11-03 (1214) | BBG000NDV1D4 | T-Mobile US, Inc. | BBG000NDV1D4 / 0001283699 | TMUS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TPL | 2024-11-26 to 2026-11-03 (485) | BBG00YRJXKB1 | Texas Pacific Land Corporation | BBG00YRJXKB1 / 0001811074 | TPL 2024-11-26..2026-11-03 | events 485 | events |
| TPR | 2022-01-03 to 2026-11-03 (1214) | BBG000BY29C7 | Tapestry, Inc. Common Stock | BBG000BY29C7 / 0001116132 | TPR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TRGP | 2022-10-12 to 2026-11-03 (1019) | BBG0015XMW40 | Targa Resources Corp. | BBG0015XMW40 / 0001389170 | TRGP 2022-10-12..2026-11-03 | events 1019 | events |
| TRMB | 2022-01-03 to 2026-11-03 (1214) | BBG000BNPS52 | Trimble Inc. Common Stock | BBG000BNPS52 / 0000864749 | TRMB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TROW | 2022-01-03 to 2026-11-03 (1214) | BBG000BVMPN3 | T Rowe Price Group Inc | BBG000BVMPN3 / 0001113169 | TROW 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TRV | 2022-01-03 to 2026-11-03 (1214) | BBG000BJ81C1 | The Travelers Companies, Inc. | BBG000BJ81C1 / 0000086312 | TRV 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TSCO | 2022-01-03 to 2026-11-03 (1214) | BBG000BLXZN1 | Tractor Supply Co | BBG000BLXZN1 / 0000916365 | TSCO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TSLA | 2022-01-03 to 2026-11-03 (1214) | BBG000N9MNX3 | Tesla, Inc. Common Stock | BBG000N9MNX3 / 0001318605 | TSLA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TSN | 2022-01-03 to 2026-11-03 (1214) | BBG000DKCC19 | Tyson Foods, Inc. | BBG000DKCC19 / 0000100493 | TSN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TT | 2022-01-03 to 2026-11-03 (1214) | CIK0001466258 | Trane Technologies plc | 0001466258 | TT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TTD | 2025-07-18 to 2026-09-18 (295) | BBG00629NGT2 | The Trade Desk, Inc. | BBG00629NGT2 / 0001671933 | TTD 2025-07-18..2026-09-18 | events 295 | events |
| TTWO | 2022-01-03 to 2026-11-03 (1214) | BBG000BS1YV5 | Take-Two Interactive Software Inc | BBG000BS1YV5 / 0000946581 | TTWO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TWLO | 2026-10-06 to 2026-11-03 (21) | BBG0029ZX840 | Twilio Inc. | BBG0029ZX840 / 0001447669 | TWLO 2026-10-06..2026-11-03 | events 21 | events |
| TWTR | 2022-01-03 to 2022-10-31 (209) | BBG000H6HNW3 | Twitter, Inc. | BBG000H6HNW3 / 0001418091 | TWTR 2022-01-03..2022-10-31 | panel 207; after its last price 2 | records |
| TXN | 2022-01-03 to 2026-11-03 (1214) | BBG000BVV7G1 | Texas Instruments Incorporated | BBG000BVV7G1 / 0000097476 | TXN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TXT | 2022-01-03 to 2026-11-03 (1214) | BBG000BVVQQ8 | Textron, Inc. | BBG000BVVQQ8 / 0000217346 | TXT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| TYL | 2022-01-03 to 2026-11-03 (1214) | BBG000BVWZF9 | Tyler Technologies, Inc. | BBG000BVWZF9 / 0000860731 | TYL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| UA | 2022-01-03 to 2022-06-17 (116) | BBG009DTD8H2 | Under Armour, Inc. Class C Common Stock, $0.0003 1/3 par value | BBG009DTD8H2 / 0001336917 | UA 2022-01-03..2022-06-17 | panel 116 | records+events |
| UAA | 2022-01-03 to 2022-06-17 (116) | BBG000BXM6V2 | Under Armour, Inc. | BBG000BXM6V2 / 0001336917 | UAA 2022-01-03..2022-06-17 | panel 116 | records+events |
| UAL | 2022-01-03 to 2026-11-03 (1214) | BBG000M65M61 | United Airlines Holdings, Inc. Common Stock | BBG000M65M61 / 0000100517 | UAL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| UBER | 2023-12-18 to 2026-11-03 (722) | BBG002B04MT8 | Uber Technologies, Inc. | BBG002B04MT8 / 0001543151 | UBER 2023-12-18..2026-11-03 | events 722 | events |
| UDR | 2022-01-03 to 2026-11-03 (1214) | BBG000C41023 | UDR, Inc. | BBG000C41023 / 0000074208 | UDR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| UHS | 2022-01-03 to 2026-11-03 (1214) | BBG000CB8Q50 | Universal Health Services, Inc. Class B | BBG000CB8Q50 / 0000352915 | UHS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ULTA | 2022-01-03 to 2026-11-03 (1214) | BBG00FWQ4VD6 | Ulta Beauty, Inc. Common Stock | BBG00FWQ4VD6 / 0001403568 | ULTA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| UNH | 2022-01-03 to 2026-11-03 (1214) | BBG000CH5208 | UNITEDHEALTH GROUP INCORPORATED (Delaware) | BBG000CH5208 / 0000731766 | UNH 2022-01-03..2026-11-03 | panel 1214 | records+events |
| UNP | 2022-01-03 to 2026-11-03 (1214) | BBG000BW3299 | Union Pacific Corp. | BBG000BW3299 / 0000100885 | UNP 2022-01-03..2026-11-03 | panel 1214 | records+events |
| UPS | 2022-01-03 to 2026-11-03 (1214) | BBG000L9CV04 | United Parcel Service, Inc. Class B | BBG000L9CV04 / 0001090727 | UPS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| URI | 2022-01-03 to 2026-11-03 (1214) | BBG000BXMFC3 | United Rentals, Inc. | BBG000BXMFC3 / 0001067701 | URI 2022-01-03..2026-11-03 | panel 1214 | records+events |
| USB | 2022-01-03 to 2026-11-03 (1214) | BBG000FFDM15 | U.S. Bancorp | BBG000FFDM15 / 0000036104 | USB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| V | 2022-01-03 to 2026-11-03 (1214) | BBG000PSKYX7 | VISA Inc. | BBG000PSKYX7 / 0001403161 | V 2022-01-03..2026-11-03 | panel 1214 | records+events |
| VEEV | 2026-05-07 to 2026-11-03 (125) | BBG001CGB489 | Veeva Systems Inc. | BBG001CGB489 / 0001393052 | VEEV 2026-05-07..2026-11-03 | events 125 | events |
| VFC | 2022-01-03 to 2024-04-02 (564) | BBG000BWCKB6 | V.F. Corporation | BBG000BWCKB6 / 0000103379 | VFC 2022-01-03..2024-04-02 | panel 564 | records+events |
| VICI | 2022-06-08 to 2026-11-03 (1106) | BBG00HVVB499 | VICI Properties Inc. Common Stock | BBG00HVVB499 / 0001705696 | VICI 2022-06-08..2026-11-03 | events 1106 | events |
| VLO | 2022-01-03 to 2026-11-03 (1214) | BBG000BBGGQ1 | Valero Energy Corporation | BBG000BBGGQ1 / 0001035002 | VLO 2022-01-03..2026-11-03 | panel 1214 | records+events |
| VLTO | 2023-10-02 to 2026-11-03 (776) | BBG019Q32XJ9 | Veralto Corporation | BBG019Q32XJ9 / 0001967680 | VLTO 2023-10-02..2026-11-03 | events 776 | events |
| VMC | 2022-01-03 to 2026-11-03 (1214) | BBG000BWGYF8 | Vulcan Materials Company(Holding Company) | BBG000BWGYF8 / 0001396009 | VMC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| VMRK | 2022-01-03 to 2026-11-03 (1214) | BBG000BG8M31 | Vivmark Residential | BBG000BG8M31 / 0000906107 | EQR 2022-01-03..2026-08-17; VMRK 2026-08-18..2026-11-03 | events 1159; panel 55 | records+events |
| VNO | 2022-01-03 to 2023-01-04 (253) | BBG000BWHD54 | Vornado Realty Trust | BBG000BWHD54 / 0000899689 | VNO 2022-01-03..2023-01-04 | panel 253 | records+events |
| VRSK | 2022-01-03 to 2026-11-03 (1214) | BBG000BCZL41 | Verisk Analytics, Inc. Common Stock | BBG000BCZL41 / 0001442145 | VRSK 2022-01-03..2026-11-03 | panel 1214 | records+events |
| VRSN | 2022-01-03 to 2026-11-03 (1214) | BBG000BGKHZ3 | VeriSign Inc | BBG000BGKHZ3 / 0001014473 | VRSN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| VRT | 2026-03-23 to 2026-11-03 (157) | BBG00L2B8KW8 | Vertiv Holdings Co Class A Common Stock | BBG00L2B8KW8 / 0001674101 | VRT 2026-03-23..2026-11-03 | events 157 | events |
| VRTX | 2022-01-03 to 2026-11-03 (1214) | BBG000C1S2X2 | Vertex Pharmaceuticals Inc | BBG000C1S2X2 / 0000875320 | VRTX 2022-01-03..2026-11-03 | panel 1214 | records+events |
| VST | 2024-05-08 to 2026-11-03 (625) | BBG00DXDL6Q1 | Vistra Corp. | BBG00DXDL6Q1 / 0001692819 | VST 2024-05-08..2026-11-03 | events 625 | events |
| VTR | 2022-01-03 to 2026-11-03 (1214) | BBG000FRVHB9 | Ventas, Inc. | BBG000FRVHB9 / 0000740260 | VTR 2022-01-03..2026-11-03 | panel 1214 | records+events |
| VTRS | 2022-01-03 to 2026-11-03 (1214) | BBG00Y4RQNH4 | Viatris Inc. Common Stock | BBG00Y4RQNH4 / 0001792044 | VTRS 2022-01-03..2026-11-03 | panel 1214 | records+events |
| VYLR | 2026-10-01 to 2026-11-03 (24) | BBG024CQR6X9 | Vylor Inc. | BBG024CQR6X9 / 0002128626 | VYLR 2026-10-01..2026-11-03 | events 24 | events |
| VZ | 2022-01-03 to 2026-11-03 (1214) | BBG000HS77T5 | Verizon Communications | BBG000HS77T5 / 0000732712 | VZ 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WAB | 2022-01-03 to 2026-11-03 (1214) | BBG000BDD940 | Wabtec Inc. | BBG000BDD940 / 0000943452 | WAB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WAT | 2022-01-03 to 2026-11-03 (1214) | BBG000FQRVM3 | Waters Corp | BBG000FQRVM3 / 0001000697 | WAT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WBA | 2022-01-03 to 2025-08-27 (916) | BBG000BWLMJ4 | Walgreens Boots Alliance, Inc | BBG000BWLMJ4 / 0001618921 | WBA 2022-01-03..2025-08-27 | panel 916 | records |
| WBD | 2022-04-11 to 2026-10-05 (1125) | - | - | - | - | missing 1125 | neither |
| WDAY | 2024-12-23 to 2026-11-03 (467) | BBG000VC0T95 | Workday, Inc. Class A Common Stock | BBG000VC0T95 / 0001327811 | WDAY 2024-12-23..2026-11-03 | events 467 | events |
| WDC | 2022-01-03 to 2026-11-03 (1214) | BBG000BWNFZ9 | Western Digital Corp. | BBG000BWNFZ9 / 0000106040 | WDC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WEC | 2022-01-03 to 2026-11-03 (1214) | BBG000BWP7D9 | WEC Energy Group, Inc. | BBG000BWP7D9 / 0000783325 | WEC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WELL | 2022-01-03 to 2026-11-03 (1214) | BBG000BKY1G5 | Welltower Inc. | BBG000BKY1G5 / 0000766704 | WELL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WFC | 2022-01-03 to 2026-11-03 (1214) | BBG000BWQFY7 | Wells Fargo & Co. | BBG000BWQFY7 / 0000072971 | WFC 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WHR | 2022-01-03 to 2024-03-15 (553) | BBG000BWSV34 | Whirlpool Corp. | BBG000BWSV34 / 0000106640 | WHR 2022-01-03..2024-03-15 | panel 553 | records+events |
| WLTW | 2022-01-03 to 2022-01-07 (5) | CIK0001140536 | Willis Towers Watson Public Limited Company Ordinary Shares | 0001140536 | WLTW 2022-01-03..2022-01-07 | panel 5 | records+events |
| WM | 2022-01-03 to 2026-11-03 (1214) | BBG000BWVSR1 | Waste Management, Inc. | BBG000BWVSR1 / 0000823768 | WM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WMB | 2022-01-03 to 2026-11-03 (1214) | BBG000BWVCP8 | Williams Companies Inc. | BBG000BWVCP8 / 0000107263 | WMB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WMT | 2022-01-03 to 2026-11-03 (1214) | BBG000BWXBC2 | Walmart Inc. Common Stock | BBG000BWXBC2 / 0000104169 | WMT 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WRB | 2022-01-03 to 2026-11-03 (1214) | BBG000BD1HP2 | W.R. Berkley Corporation | BBG000BD1HP2 / 0000011544 | WRB 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WSM | 2025-03-24 to 2026-11-03 (407) | BBG000FSMWC3 | Williams-Sonoma, Inc. | BBG000FSMWC3 / 0000719955 | WSM 2025-03-24..2026-11-03 | events 407 | events |
| WST | 2022-01-03 to 2026-11-03 (1214) | BBG000BX24N8 | West Pharmaceutical Services, Inc. | BBG000BX24N8 / 0000105770 | WST 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WTW | 2022-01-10 to 2026-11-03 (1209) | CIK0001140536 | Willis Towers Watson Public Limited Company Ordinary Shares | 0001140536 | WTW 2022-01-10..2026-11-03 | events 1209 | records+events |
| WY | 2022-01-03 to 2026-11-03 (1214) | BBG000BX3BL3 | Weyerhaeuser Company | BBG000BX3BL3 / 0000106535 | WY 2022-01-03..2026-11-03 | panel 1214 | records+events |
| WYNN | 2022-01-03 to 2026-11-03 (1214) | BBG000LD9JQ8 | Wynn Resorts Ltd | BBG000LD9JQ8 / 0001174922 | WYNN 2022-01-03..2026-11-03 | panel 1214 | records+events |
| XEL | 2022-01-03 to 2026-11-03 (1214) | BBG000BCTQ65 | Xcel Energy, Inc. | BBG000BCTQ65 / 0000072903 | XEL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| XLNX | 2022-01-03 to 2022-02-14 (30) | BBG004PYGYW2 | Xilinx Inc | BBG004PYGYW2 / 0000743988 | XLNX 2022-01-03..2022-02-14 | panel 29; after its last price 1 | records |
| XOM | 2022-01-03 to 2026-11-03 (1214) | BBG023CY9MM1 | ExxonMobil Holdings Corporation | BBG023CY9MM1 / 0002115436 | XOM 2022-01-03..2026-11-03 | predecessor 1127; panel 87 | records+events |
| XRAY | 2022-01-03 to 2024-04-02 (564) | BBG000BX57K1 | DENTSPLY SIRONA Inc. | BBG000BX57K1 / 0000818479 | XRAY 2022-01-03..2024-04-02 | panel 564 | records+events |
| XYL | 2022-01-03 to 2026-11-03 (1214) | BBG001D8R5D0 | Xylem Inc | BBG001D8R5D0 / 0001524472 | XYL 2022-01-03..2026-11-03 | panel 1214 | records+events |
| XYZ | 2025-07-23 to 2026-11-03 (324) | BBG0018SLC07 | Block, Inc. | BBG0018SLC07 / 0001512673 | XYZ 2025-07-23..2026-11-03 | events 324 | events |
| YUM | 2022-01-03 to 2026-11-03 (1214) | BBG000BH3GZ2 | Yum! Brands, Inc. | BBG000BH3GZ2 / 0001041061 | YUM 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ZBH | 2022-01-03 to 2026-11-03 (1214) | BBG000BKPL53 | Zimmer Biomet Holdings, Inc. | BBG000BKPL53 / 0001136869 | ZBH 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ZBRA | 2022-01-03 to 2026-11-03 (1214) | BBG000CC7LQ7 | Zebra Technologies Corporation | BBG000CC7LQ7 / 0000877212 | ZBRA 2022-01-03..2026-11-03 | panel 1214 | records+events |
| ZION | 2022-01-03 to 2024-03-15 (553) | BBG000BX9WL1 | Zions Bancorporation N.A. | BBG000BX9WL1 / 0000109380 | ZION 2022-01-03..2024-03-15 | panel 553 | records+events |
| ZTS | 2022-01-03 to 2026-11-03 (1214) | BBG0039320N9 | ZOETIS INC. | BBG0039320N9 / 0001555280 | ZTS 2022-01-03..2026-11-03 | panel 1214 | records+events |


### The table's securities with more than one row, or a row not the panel's

From `sp500_identity_2022-01-03_2026-11-03.csv`; the other 496 securities are one row each, one ticker priced by the panel throughout their span.

| Security | Name | Ticker | First | Last | Held | Source | Starts | Ends |
|---|---|---|---|---|---|---|---|---|
| BBG001Y2XS07 | Airbnb, Inc. Class A Common Stock | ABNB | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as ABNB from 2020-12-10) | the window's last day |
| BBG01NY7B734 | Amentum Holdings, Inc. | AMTM | 2024-09-24 | 2026-11-03 | 0 | events | the first of its events, 2024-09-24: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG00ZNLTFK3 | Apollo Global Management, Inc. | APO | 2022-01-03 | 2026-11-03 | 0 | events | the first of its events, 2022-01-03: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG006HFPX77 | Applovin Corporation Class A Common Stock | APP | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as APP from 2021-04-15) | the window's last day |
| CIK0000947484 | Arch Capital Group Ltd | ACGL | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as ACGL from 2003-09-10) | the window's last day |
| BBG0068KVV91 | Ares Management Corporation Class A Common Stock | ARES | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as ARES from 2018-11-26) | the window's last day |
| BBG000BHJWG1 | Axon Enterprise, Inc. Common Stock | AXON | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as AXON from 2021-01-26) | the window's last day |
| BBG000BDDNH5 | Ball Corporation | BLL | 2022-01-03 | 2022-05-09 | 0 | events | the window's first day (as BLL from 2003-09-10) | traded as another ticker from 2022-05-10 (events) |
| BBG000BDDNH5 | Ball Corporation | BALL | 2022-05-10 | 2026-11-03 | 1126 | panel | Ball Corporation began trading as BALL on 2022-05-10 (events: a change from another ticker) | the window's last day |
| BBG000BD8PN9 | Bank of New York Mellon Corporation | BK | 2022-01-03 | 2026-05-20 | 0 | events | the window's first day (as BK from 2007-07-02) | traded as another ticker from 2026-05-21 (events) |
| BBG000BD8PN9 | Bank of New York Mellon Corporation | BNY | 2026-05-21 | 2026-11-03 | 115 | panel | Bank of New York Mellon Corporation began trading as BNY on 2026-05-21 (events: a change from another ticker) | the window's last day |
| BBG000BH0106 | Blackstone Inc. | BX | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as BX from 2019-07-01) | the window's last day |
| BBG0018SLC07 | Block, Inc. | SQ | 2022-01-03 | 2025-01-17 | 0 | events | the window's first day (as SQ from 2015-11-18) | traded as another ticker from 2025-01-21 (events) |
| BBG0018SLC07 | Block, Inc. | XYZ | 2025-01-21 | 2026-11-03 | 0 | events | began trading as XYZ on 2025-01-21 (events) | the window's last day |
| BBG000N7KBZ3 | Bloom Energy Corporation | BE | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as BE from 2018-07-24) | the window's last day |
| BBG000BKD3K9 | Builders FirstSource, Inc. | BLDR | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as BLDR from 2005-06-22) | the window's last day |
| CIK0001996862 | Bunge Global SA | BG | 2023-11-01 | 2026-11-03 | 0 | events | the first of its events, 2023-11-01: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG000BB0V03 | Camden Property Trust | CPT | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as CPT from 2003-09-10) | the window's last day |
| BBG00GCTWDJ3 | Carvana Co. | CVNA | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as CVNA from 2017-04-27) | the window's last day |
| BBG000BF0672 | Casey's General Stores Inc | CASY | 2022-01-03 | 2026-11-03 | 0 | presumed | presumed: CASY finds it today, and its events' last change, on 2010-08-27, was away from it | the window's last day |
| BBG000MDCQC2 | Cencora, Inc. | ABC | 2022-01-03 | 2023-08-29 | 0 | events | the window's first day (as ABC from 2003-09-10) | traded as another ticker from 2023-08-30 (events) |
| BBG000MDCQC2 | Cencora, Inc. | COR | 2023-08-30 | 2026-11-03 | 798 | panel | Cencora, Inc. began trading as COR on 2023-08-30 (events: a change from another ticker) | the window's last day |
| BBG000BP1152 | Ciena Corporation | CIEN | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as CIEN from 2006-10-23) | the window's last day |
| BBG000BLW102 | Coherent Corp. | COHR | 2022-09-08 | 2026-11-03 | 0 | events | the first of its events, 2022-09-08: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG00ZGF7771 | Coinbase Global, Inc. Class A Common Stock | COIN | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as COIN from 2021-04-14) | the window's last day |
| BBG000BWN2Y9 | Comfort Systems USA, Inc. | FIX | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as FIX from 2003-09-10) | the window's last day |
| BBG014KFRNP7 | Constellation Energy Corporation Common Stock | CEGV | 2022-01-19 | 2022-02-01 | 0 | events | the first of its events, 2022-01-19: the start of Polygon's record of it, not necessarily of the ticker | traded as another ticker from 2022-02-02 (events) |
| BBG014KFRNP7 | Constellation Energy Corporation Common Stock | CEG | 2022-02-02 | 2026-11-03 | 0 | events | began trading as CEG on 2022-02-02 (events) | the window's last day |
| BBG000D7JKW9 | CoStar Group Inc | CSGP | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as CSGP from 2003-09-10) | the window's last day |
| CIK0000849395 | CRH Public Limited Company | CRHC | 2022-01-03 | 2023-09-22 | 0 | events | the window's first day (as CRHC from 2021-09-23) | traded as another ticker from 2023-09-25 (events) |
| CIK0000849395 | CRH Public Limited Company | CRH | 2023-09-25 | 2026-11-03 | 0 | events | began trading as CRH on 2023-09-25 (events) | the window's last day |
| BBG00BLYKS03 | CrowdStrike Holdings, Inc. Class A Common Stock | CRWD | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as CRWD from 2019-06-12) | the window's last day |
| BBG003NJHZT9 | Datadog, Inc. Class A Common Stock | DDOG | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as DDOG from 2019-09-19) | the window's last day |
| BBG000BKXYX5 | Deckers Outdoor Corp | DECK | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as DECK from 2003-09-10) | the window's last day |
| BBG00DW3SZS1 | Dell Technologies Inc. | DELL | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as DELL from 2018-12-26) | the window's last day |
| BBG005D7QCJ3 | DoorDash, Inc. Class A Common Stock | DASH | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as DASH from 2020-12-08) | the window's last day |
| BBG000TGLV00 | EchoStar Corporation | SATS | 2022-01-03 | 2026-06-23 | 0 | events | the window's first day (as SATS from 2008-01-02) | traded as another ticker from 2026-06-24 (events) |
| BBG000TGLV00 | EchoStar Corporation | ECHO | 2026-06-24 | 2026-11-03 | 93 | panel | EchoStar Corporation began trading as ECHO on 2026-06-24 (events: a change from another ticker) | the window's last day |
| BBG000BCG930 | Elevance Health, Inc. | ANTM | 2022-01-03 | 2022-06-27 | 0 | events | the window's first day (as ANTM from 2014-12-03) | traded as another ticker from 2022-06-28 (events) |
| BBG000BCG930 | Elevance Health, Inc. | ELV | 2022-06-28 | 2026-11-03 | 1093 | panel | Elevance Health, Inc. began trading as ELV on 2022-06-28 (events: a change from another ticker) | the window's last day |
| BBG000BBRPL4 | EMCOR Group, Inc. | EME | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as EME from 2003-09-10) | the window's last day |
| BBG000BHZ5J9 | EQT CORP | EQT | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as EQT from 2009-02-10) | the window's last day |
| BBG000BBTJZ1 | Erie Indemnity Co | ERIE | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as ERIE from 2003-09-10) | the window's last day |
| CIK0001095073 | Everest Group, Ltd. | RE | 2022-01-03 | 2023-07-07 | 0 | events | the window's first day (as RE from 2003-09-10) | traded as another ticker from 2023-07-10 (events) |
| CIK0001095073 | Everest Group, Ltd. | EG | 2023-07-10 | 2026-11-03 | 835 | panel | Everest Group, Ltd. began trading as EG on 2023-07-10 (events: a change from another ticker) | the window's last day |
| BBG00212PVZ5 | Everpure, Inc. | PSTG | 2022-01-03 | 2026-04-16 | 0 | events | the window's first day (as PSTG from 2015-10-07) | traded as another ticker from 2026-04-17 (events) |
| BBG00212PVZ5 | Everpure, Inc. | P | 2026-04-17 | 2026-11-03 | 0 | events | began trading as P on 2026-04-17 (events) | the window's last day |
| BBG00Z6DX554 | Expand Energy Corporation Common Stock | CHK | 2022-01-03 | 2024-10-01 | 0 | events | the window's first day (as CHK from 2021-02-10) | traded as another ticker from 2024-10-02 (events) |
| BBG00Z6DX554 | Expand Energy Corporation Common Stock | EXE | 2024-10-02 | 2026-11-03 | 0 | events | began trading as EXE on 2024-10-02 (events) | the window's last day |
| BBG000DW76Y6 | Fair Isaac Corporation | FICO | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as FICO from 2009-08-19) | the window's last day |
| BBG01ZPB91C4 | FedEx Freight Holding Company, Inc. | FDXF | 2026-05-27 | 2026-11-03 | 0 | events | the first of its events, 2026-05-27: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG01NMRD768 | Ferguson Enterprises Inc. | FERG | 2024-08-01 | 2026-11-03 | 0 | events | the first of its events, 2024-08-01: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG0019LSQ49 | First Republic Bank | FRC | 2022-01-03 | 2023-04-28 | 332 | panel | the window's first day | First Republic Bank traded as another ticker from 2023-05-03 (events) |
| BBG0019LSQ49 | First Republic Bank | FRCB | 2023-05-03 | 2023-11-17 | 0 | events | began trading as FRCB on 2023-05-03 (events) | traded as another ticker from 2023-11-18 (events) |
| BBG0019LSQ49 | First Republic Bank | S600 | 2023-11-20 | 2026-11-03 | 0 | events | began trading as S600 on 2023-11-18 (events) | the window's last day |
| BBG000BFL116 | First Solar, Inc. | FSLR | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as FSLR from 2006-11-17) | the window's last day |
| BBG000BJKPG0 | Fiserv, Inc. Common Stock | FISV | 2022-01-03 | 2023-06-06 | 358 | panel | the window's first day | its last price under the ticker |
| BBG000BJKPG0 | Fiserv, Inc. Common Stock | FI | 2023-06-07 | 2025-11-10 | 0 | events | began trading as FI on 2023-06-07 (events) | traded as another ticker from 2025-11-11 (events) |
| BBG000BJKPG0 | Fiserv, Inc. Common Stock | FISV | 2025-11-11 | 2026-11-03 | 246 | panel | its first price under the ticker | the window's last day |
| CIK0000866374 | Flex Ltd. Ordinary Shares | FLEX | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as FLEX from 2003-09-10) | the window's last day |
| BBG001B4BV87 | FORTUNE BRANDS HOME & SECURITY, INC. | FBHS | 2022-01-03 | 2022-12-14 | 240 | panel | the window's first day | FORTUNE BRANDS HOME & SECURITY, INC. traded as another ticker from 2022-12-15 (events) |
| BBG001B4BV87 | FORTUNE BRANDS HOME & SECURITY, INC. | FBIN | 2022-12-15 | 2026-11-03 | 0 | events | began trading as FBIN on 2022-12-15 (events) | the window's last day |
| BBG01BFR8YV1 | GE HealthCare Technologies Inc. Common Stock | GEHC | 2022-12-15 | 2026-11-03 | 0 | events | the first of its events, 2022-12-15: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG013G17W31 | GE Vernova Inc. | GEV | 2024-03-27 | 2026-11-03 | 0 | events | the first of its events, 2024-03-27: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG000BH2JM1 | Gen Digital Inc. Common Stock | NLOK | 2022-01-03 | 2022-11-07 | 0 | events | the window's first day (as NLOK from 2019-11-05) | traded as another ticker from 2022-11-08 (events) |
| BBG000BH2JM1 | Gen Digital Inc. Common Stock | GEN | 2022-11-08 | 2026-11-03 | 1000 | panel | Gen Digital Inc. Common Stock began trading as GEN on 2022-11-08 (events: a change from another ticker) | the window's last day |
| BBG006MDLY05 | GoDaddy Inc | GDDY | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as GDDY from 2015-04-01) | the window's last day |
| BBG000BKYDP9 | Healthpeak Properties, Inc. | PEAK | 2022-01-03 | 2024-03-01 | 0 | events | the window's first day (as PEAK from 2019-11-05) | traded as another ticker from 2024-03-04 (events) |
| BBG000BKYDP9 | Healthpeak Properties, Inc. | DOC | 2024-03-04 | 2026-11-03 | 671 | panel | Healthpeak Properties, Inc. began trading as DOC on 2024-03-04 (events: a change from another ticker) | the window's last day |
| BBG020QYV152 | Honeywell Aerospace Inc. Common Stock | HONA | 2026-06-15 | 2026-11-03 | 0 | events | the first of its events, 2026-06-15: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG000BLK267 | Hubbell Incorporated | HUBB | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as HUBB from 2015-12-28) | the window's last day |
| BBG000R7XX87 | Insulet Corporation | PODD | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as PODD from 2007-05-15) | the window's last day |
| BBG000LV0836 | Interactive Brokers Group, Inc. Class A Common Stock | IBKR | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as IBKR from 2007-05-04) | the window's last day |
| BBG00FQH6BS9 | Invitation Homes Inc. Common Stock | INVH | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as INVH from 2017-01-31) | the window's last day |
| BBG000BJNGN9 | Jabil Inc. | JBL | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as JBL from 2003-09-10) | the window's last day |
| BBG000TJM7F0 | Keurig Dr Pepper Inc. | KDP | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as KDP from 2018-07-10) | the window's last day |
| BBG000BCQ6J8 | KKR & Co. Inc. | KKR | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as KKR from 2018-07-02) | the window's last day |
| BBG000BB5B84 | Lennox International Inc. | LII | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as LII from 2003-09-10) | the window's last day |
| BBG000R8ZVD1 | lululemon athletica inc. | LULU | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as LULU from 2007-07-30) | the window's last day |
| BBG0073F9RT7 | Lumentum Holdings Inc. Common Stock | LITE | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as LITE from 2015-07-23) | the window's last day |
| BBG000BP4MH0 | Marsh | MMC | 2022-01-03 | 2026-01-13 | 0 | events | the window's first day (as MMC from 2003-09-10) | traded as another ticker from 2026-01-14 (events) |
| BBG000BP4MH0 | Marsh | MRSH | 2026-01-14 | 2026-11-03 | 203 | panel | Marsh began trading as MRSH on 2026-01-14 (events: a change from another ticker) | the window's last day |
| BBG00ZXBJ153 | Marvell Technology, Inc. Common Stock | MRVL | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as MRVL from 2021-04-21) | the window's last day |
| BBG01B9G42Y3 | MasterBrand, Inc. | MBC | 2022-12-09 | 2026-11-03 | 0 | events | the first of its events, 2022-12-09: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG000MM2P62 | Meta Platforms, Inc. Class A Common Stock | FB | 2022-01-03 | 2022-06-08 | 0 | events | the window's first day (as FB from 2012-05-18) | traded as another ticker from 2022-06-09 (events) |
| BBG000MM2P62 | Meta Platforms, Inc. Class A Common Stock | META | 2022-06-09 | 2026-11-03 | 1105 | panel | Meta Platforms, Inc. Class A Common Stock began trading as META on 2022-06-09 (events: a change from another ticker); Roundhill Ball Metaverse ETF traded as another ticker from 2022-01-31 (events) | the window's last day |
| BBG000MBHNC8 | Molina Healthcare, Inc. | MOH | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as MOH from 2003-09-10) | the window's last day |
| BBG000BPWGR1 | Nordson Corp | NDSN | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as NDSN from 2003-09-10) | the window's last day |
| BBG001YV1SM4 | Okta, Inc. Class A Common Stock | OKTA | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as OKTA from 2017-04-07) | the window's last day |
| BBG000DV7MX4 | ON Semiconductor Corp | ON | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as ON from 2015-04-07) | the window's last day |
| BBG000N7QR55 | Palantir Technologies Inc. Class A Common Stock | PLTR | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as PLTR from 2020-09-29) | the window's last day |
| BBG0014GJCT9 | Palo Alto Networks, Inc. Common Stock | PANW | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as PANW from 2012-07-20) | the window's last day |
| BBG000BQWPC5 | PG&E Corporation | PCG | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as PCG from 2003-09-10) | the window's last day |
| BBG01TRL73S5 | Qnity Electronics, Inc. | Q | 2025-10-27 | 2026-11-03 | 0 | events | the first of its events, 2025-10-27: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG005K1D4S0 | Reddit, Inc. | RDDT | 2024-03-20 | 2026-11-03 | 0 | events | the first of its events, 2024-03-20: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG000FXW512 | Revvity, Inc. | PKI | 2022-01-03 | 2023-05-15 | 0 | events | the window's first day (as PKI from 2003-09-10) | traded as another ticker from 2023-05-16 (events) |
| BBG000FXW512 | Revvity, Inc. | RVTY | 2023-05-16 | 2026-11-03 | 871 | panel | Revvity, Inc. began trading as RVTY on 2023-05-16 (events: a change from another ticker) | the window's last day |
| BBG008NMBXN8 | Robinhood Markets, Inc. Class A Common Stock | HOOD | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as HOOD from 2021-07-29) | the window's last day |
| BBG011J1MN62 | Roundhill Ball Metaverse ETF | META | 2022-01-03 | 2022-01-28 | 19 | panel | the window's first day | Meta Platforms, Inc. Class A Common Stock began trading as META on 2022-06-09 (events: a change from another ticker); Roundhill Ball Metaverse ETF traded as another ticker from 2022-01-31 (events) |
| BBG011J1MN62 | Roundhill Ball Metaverse ETF | METV | 2022-01-31 | 2026-11-03 | 0 | events | began trading as METV on 2022-01-31 (events) | the window's last day |
| BBG01R388JG1 | Sandisk Corporation Common Stock | SNDK | 2025-02-13 | 2026-11-03 | 0 | events | the first of its events, 2025-02-13: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG000M6TR37 | SIGNATURE BANK (NY) | SBNY | 2022-01-03 | 2023-03-10 | 298 | panel | the window's first day | its last price under the ticker |
| BBG000M6TR37 | SIGNATURE BANK (NY) | SBNY | 2023-03-28 | 2026-11-03 | 864 | panel | its first price under the ticker | the window's last day |
| BBG01VS5NK99 | Skydance Corporation | PSKY | 2025-08-07 | 2026-10-05 | 0 | events | the first of its events, 2025-08-07: the start of Polygon's record of it, not necessarily of the ticker | traded as another ticker from 2026-10-06 (events) |
| BBG01VS5NK99 | Skydance Corporation | SKYD | 2026-10-06 | 2026-11-03 | 21 | panel | Skydance Corporation began trading as SKYD on 2026-10-06 (events: a change from another ticker) | the window's last day |
| BBG01X5744P5 | Solstice Advanced Materials Inc. Common Stock | SOLS | 2025-10-20 | 2026-11-03 | 0 | events | the first of its events, 2025-10-20: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG018YZH6T3 | Solventum Corporation | SOLV | 2024-03-26 | 2026-11-03 | 0 | events | the first of its events, 2024-03-26: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG000HGYNZ9 | Steel Dynamics Inc | STLD | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as STLD from 2003-09-10) | the window's last day |
| BBG000MYZDJ3 | Super Micro Computer, Inc. Common Stock | SMCI | 2024-10-01 | 2026-11-03 | 0 | events | the first of its events, 2024-10-01: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG0015XMW40 | Targa Resources Corp. | TRGP | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as TRGP from 2010-12-07) | the window's last day |
| BBG00YRJXKB1 | Texas Pacific Land Corporation | TPL | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as TPL from 2021-01-11) | the window's last day |
| BBG000BKLH74 | The Gap, Inc. | GPS | 2022-01-03 | 2024-08-21 | 662 | panel | the window's first day | The Gap, Inc. traded as another ticker from 2024-08-22 (events) |
| BBG000BKLH74 | The Gap, Inc. | GAP | 2024-08-22 | 2026-11-03 | 0 | events | began trading as GAP on 2024-08-22 (events) | the window's last day |
| BBG00629NGT2 | The Trade Desk, Inc. | TTD | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as TTD from 2016-09-21) | the window's last day |
| BBG01G9JKWV5 | TKO Group Holdings, Inc. | TKO | 2023-09-12 | 2026-11-03 | 0 | events | the first of its events, 2023-09-12: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG0029ZX840 | Twilio Inc. | TWLO | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as TWLO from 2016-06-22) | the window's last day |
| BBG002B04MT8 | Uber Technologies, Inc. | UBER | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as UBER from 2019-05-09) | the window's last day |
| BBG001CGB489 | Veeva Systems Inc. | VEEV | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as VEEV from 2013-10-16) | the window's last day |
| BBG019Q32XJ9 | Veralto Corporation | VLTO | 2023-09-27 | 2026-11-03 | 0 | events | the first of its events, 2023-09-27: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG00L2B8KW8 | Vertiv Holdings Co Class A Common Stock | VRT | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as VRT from 2020-02-10) | the window's last day |
| BBG00HVVB499 | VICI Properties Inc. Common Stock | VICI | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as VICI from 2018-02-01) | the window's last day |
| BBG00DXDL6Q1 | Vistra Corp. | VST | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as VST from 2017-05-10) | the window's last day |
| BBG000BG8M31 | Vivmark Residential | EQR | 2022-01-03 | 2026-08-17 | 0 | events | the window's first day (as EQR from 2003-09-10) | traded as another ticker from 2026-08-18 (events) |
| BBG000BG8M31 | Vivmark Residential | VMRK | 2026-08-18 | 2026-11-03 | 55 | panel | Vivmark Residential began trading as VMRK on 2026-08-18 (events: a change from another ticker) | the window's last day |
| BBG024CQR6X9 | Vylor Inc. | VYLR | 2026-10-01 | 2026-11-03 | 0 | events | the first of its events, 2026-10-01: the start of Polygon's record of it, not necessarily of the ticker | the window's last day |
| BBG000FSMWC3 | Williams-Sonoma, Inc. | WSM | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as WSM from 2003-09-10) | the window's last day |
| CIK0001140536 | Willis Towers Watson Public Limited Company Ordinary Shares | WLTW | 2022-01-03 | 2022-01-07 | 5 | panel | the window's first day | Willis Towers Watson Public Limited Company Ordinary Shares traded as another ticker from 2022-01-10 (events) |
| CIK0001140536 | Willis Towers Watson Public Limited Company Ordinary Shares | WTW | 2022-01-10 | 2026-11-03 | 0 | events | began trading as WTW on 2022-01-10 (events) | the window's last day |
| BBG000VC0T95 | Workday, Inc. Class A Common Stock | WDAY | 2022-01-03 | 2026-11-03 | 0 | events | the window's first day (as WDAY from 2012-10-12) | the window's last day |


### The fetch plan's spans under tickers the panel does not have

From `sp500_identity_2022-01-03_2026-11-03_fetch.csv`; the panel's own 506 tickers' spans are there too, their prices cached.

| Ticker | Security | Name | First | Last | Member-days | Share samples |
|---|---|---|---|---|---|---|
| ABC | BBG000MDCQC2 | Cencora, Inc. | 2022-01-03 | 2023-08-29 | 416 | 8: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-08-29 |
| ABNB | BBG001Y2XS07 | Airbnb, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 786 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| ACGL | CIK0000947484 | Arch Capital Group Ltd | 2022-01-03 | 2026-11-03 | 1005 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| AMTM | BBG01NY7B734 | Amentum Holdings, Inc. | 2024-09-24 | 2026-11-03 | 59 | 10: 2024-09-24 2024-12-23 2025-03-27 2025-06-27 2025-09-26 2025-12-26 2026-03-30 2026-06-30 2026-09-29 2026-11-03 |
| ANTM | BBG000BCG930 | Elevance Health, Inc. | 2022-01-03 | 2022-06-27 | 121 | 3: 2022-01-03 2022-04-04 2022-06-27 |
| APO | BBG00ZNLTFK3 | Apollo Global Management, Inc. | 2022-01-03 | 2026-11-03 | 467 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| APP | BBG006HFPX77 | Applovin Corporation Class A Common Stock | 2022-01-03 | 2026-11-03 | 282 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| ARES | BBG0068KVV91 | Ares Management Corporation Class A Common Stock | 2022-01-03 | 2026-11-03 | 225 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| AXON | BBG000BHJWG1 | Axon Enterprise, Inc. Common Stock | 2022-01-03 | 2026-11-03 | 879 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| BE | BBG000N7KBZ3 | Bloom Energy Corporation | 2022-01-03 | 2026-11-03 | 32 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| BG | CIK0001996862 | Bunge Global SA | 2023-11-01 | 2026-11-03 | 754 | 13: 2023-11-01 2024-02-02 2024-05-03 2024-08-05 2024-11-01 2025-02-05 2025-05-07 2025-08-07 2025-11-05 2026-02-06 2026-05-08 2026-08-10 2026-11-03 |
| BK | BBG000BD8PN9 | Bank of New York Mellon Corporation | 2022-01-03 | 2026-05-20 | 1099 | 19: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-05-20 |
| BLDR | BBG000BKD3K9 | Builders FirstSource, Inc. | 2022-01-03 | 2026-11-03 | 690 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| BLL | BBG000BDDNH5 | Ball Corporation | 2022-01-03 | 2022-05-09 | 88 | 3: 2022-01-03 2022-04-04 2022-05-09 |
| BX | BBG000BH0106 | Blackstone Inc. | 2022-01-03 | 2026-11-03 | 786 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| CASY | BBG000BF0672 | Casey's General Stores Inc | 2022-01-03 | 2026-11-03 | 145 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| CEG | BBG014KFRNP7 | Constellation Energy Corporation Common Stock | 2022-02-02 | 2026-11-03 | 1193 | 21: 2022-02-02 2022-04-20 2022-07-21 2022-10-19 2023-01-20 2023-04-21 2023-07-24 2023-10-20 2024-01-23 2024-04-23 2024-07-24 2024-10-22 2025-01-24 2025-04-25 2025-07-28 2025-10-24 2026-01-27 2026-04-28 2026-07-29 2026-10-27 2026-11-03 |
| CEGV | BBG014KFRNP7 | Constellation Energy Corporation Common Stock | 2022-01-19 | 2022-02-01 | 0 | 2: 2022-01-19 2022-02-01 |
| CHK | BBG00Z6DX554 | Expand Energy Corporation Common Stock | 2022-01-03 | 2024-10-01 | 0 | 12: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-01 |
| CIEN | BBG000BP1152 | Ciena Corporation | 2022-01-03 | 2026-11-03 | 186 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| COHR | BBG000BLW102 | Coherent Corp. | 2022-09-08 | 2026-11-03 | 157 | 18: 2022-09-08 2022-12-07 2023-03-10 2023-06-09 2023-09-11 2023-12-08 2024-03-12 2024-06-11 2024-09-11 2024-12-10 2025-03-14 2025-06-13 2025-09-15 2025-12-12 2026-03-17 2026-06-16 2026-09-16 2026-11-03 |
| COIN | BBG00ZGF7771 | Coinbase Global, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 368 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| CPT | BBG000BB0V03 | Camden Property Trust | 2022-01-03 | 2026-11-03 | 1151 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| CRH | CIK0000849395 | CRH Public Limited Company | 2023-09-25 | 2026-11-03 | 218 | 15: 2023-09-25 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| CRHC | CIK0000849395 | CRH Public Limited Company | 2022-01-03 | 2023-09-22 | 0 | 8: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-09-22 |
| CRWD | BBG00BLYKS03 | CrowdStrike Holdings, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 594 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| CSGP | BBG000D7JKW9 | CoStar Group Inc | 2022-01-03 | 2026-11-03 | 1036 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| CVNA | BBG00GCTWDJ3 | Carvana Co. | 2022-01-03 | 2026-11-03 | 218 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| DASH | BBG005D7QCJ3 | DoorDash, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 407 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| DDOG | BBG003NJHZT9 | Datadog, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 334 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| DECK | BBG000BKXYX5 | Deckers Outdoor Corp | 2022-01-03 | 2026-11-03 | 661 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| DELL | BBG00DW3SZS1 | Dell Technologies Inc. | 2022-01-03 | 2026-11-03 | 531 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| EME | BBG000BBRPL4 | EMCOR Group, Inc. | 2022-01-03 | 2026-11-03 | 282 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| EQR | BBG000BG8M31 | Vivmark Residential | 2022-01-03 | 2026-08-17 | 1159 | 20: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-08-17 |
| EQT | BBG000BHZ5J9 | EQT CORP | 2022-01-03 | 2026-11-03 | 1026 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| ERIE | BBG000BBTJZ1 | Erie Indemnity Co | 2022-01-03 | 2026-11-03 | 531 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| EXE | BBG00Z6DX554 | Expand Energy Corporation Common Stock | 2024-10-02 | 2026-11-03 | 407 | 11: 2024-10-02 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| FB | BBG000MM2P62 | Meta Platforms, Inc. Class A Common Stock | 2022-01-03 | 2022-06-08 | 109 | 3: 2022-01-03 2022-04-04 2022-06-08 |
| FBIN | BBG001B4BV87 | FORTUNE BRANDS HOME & SECURITY, INC. | 2022-12-15 | 2026-11-03 | 2 | 18: 2022-12-15 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| FDXF | BBG01ZPB91C4 | FedEx Freight Holding Company, Inc. | 2026-05-27 | 2026-11-03 | 109 | 3: 2026-05-27 2026-08-26 2026-11-03 |
| FERG | BBG01NMRD768 | Ferguson Enterprises Inc. | 2024-08-01 | 2026-11-03 | 64 | 10: 2024-08-01 2024-10-30 2025-02-03 2025-05-05 2025-08-05 2025-11-03 2026-02-04 2026-05-06 2026-08-06 2026-11-03 |
| FI | BBG000BJKPG0 | Fiserv, Inc. Common Stock | 2023-06-07 | 2025-11-10 | 610 | 12: 2023-06-07 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2025-11-10 |
| FICO | BBG000DW76Y6 | Fair Isaac Corporation | 2022-01-03 | 2026-11-03 | 911 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| FIX | BBG000BWN2Y9 | Comfort Systems USA, Inc. | 2022-01-03 | 2026-11-03 | 218 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| FLEX | CIK0000866374 | Flex Ltd. Ordinary Shares | 2022-01-03 | 2026-11-03 | 95 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| FRCB | BBG0019LSQ49 | First Republic Bank | 2023-05-03 | 2023-11-17 | 1 | 4: 2023-05-03 2023-07-07 2023-10-05 2023-11-17 |
| FSLR | BBG000BFL116 | First Solar, Inc. | 2022-01-03 | 2026-11-03 | 972 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| GAP | BBG000BKLH74 | The Gap, Inc. | 2024-08-22 | 2026-11-03 | 0 | 11: 2024-08-22 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| GDDY | BBG006MDLY05 | GoDaddy Inc | 2022-01-03 | 2026-11-03 | 594 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| GEHC | BBG01BFR8YV1 | GE HealthCare Technologies Inc. Common Stock | 2022-12-15 | 2026-11-03 | 962 | 17: 2022-12-15 2023-03-20 2023-06-20 2023-09-19 2023-12-18 2024-03-20 2024-06-20 2024-09-19 2024-12-18 2025-03-24 2025-06-24 2025-09-23 2025-12-22 2026-03-25 2026-06-25 2026-09-24 2026-11-03 |
| GEV | BBG013G17W31 | GE Vernova Inc. | 2024-03-27 | 2026-11-03 | 651 | 12: 2024-03-27 2024-06-27 2024-09-26 2024-12-26 2025-03-31 2025-07-01 2025-09-30 2025-12-30 2026-04-01 2026-07-02 2026-10-01 2026-11-03 |
| HONA | BBG020QYV152 | Honeywell Aerospace Inc. Common Stock | 2026-06-15 | 2026-11-03 | 90 | 3: 2026-06-15 2026-09-15 2026-11-03 |
| HOOD | BBG008NMBXN8 | Robinhood Markets, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 282 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| HUBB | BBG000BLK267 | Hubbell Incorporated | 2022-01-03 | 2026-11-03 | 764 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| IBKR | BBG000LV0836 | Interactive Brokers Group, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 298 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| INVH | BBG00FQH6BS9 | Invitation Homes Inc. Common Stock | 2022-01-03 | 2026-11-03 | 1036 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| JBL | BBG000BJNGN9 | Jabil Inc. | 2022-01-03 | 2026-11-03 | 722 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| KDP | BBG000TJM7F0 | Keurig Dr Pepper Inc. | 2022-01-03 | 2026-11-03 | 1098 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| KKR | BBG000BCQ6J8 | KKR & Co. Inc. | 2022-01-03 | 2026-11-03 | 594 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| LII | BBG000BB5B84 | Lennox International Inc. | 2022-01-03 | 2026-11-03 | 467 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| LITE | BBG0073F9RT7 | Lumentum Holdings Inc. Common Stock | 2022-01-03 | 2026-11-03 | 157 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| LULU | BBG000R8ZVD1 | lululemon athletica inc. | 2022-01-03 | 2026-11-03 | 764 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| MBC | BBG01B9G42Y3 | MasterBrand, Inc. | 2022-12-09 | 2026-11-03 | 2 | 17: 2022-12-09 2023-03-14 2023-06-13 2023-09-13 2023-12-12 2024-03-14 2024-06-13 2024-09-13 2024-12-12 2025-03-18 2025-06-17 2025-09-17 2025-12-16 2026-03-19 2026-06-18 2026-09-18 2026-11-03 |
| MMC | BBG000BP4MH0 | Marsh | 2022-01-03 | 2026-01-13 | 1011 | 18: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-01-13 |
| MOH | BBG000MBHNC8 | Molina Healthcare, Inc. | 2022-01-03 | 2026-11-03 | 1017 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| MRVL | BBG00ZXBJ153 | Marvell Technology, Inc. Common Stock | 2022-01-03 | 2026-11-03 | 95 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| NDSN | BBG000BPWGR1 | Nordson Corp | 2022-01-03 | 2026-11-03 | 1184 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| NLOK | BBG000BH2JM1 | Gen Digital Inc. Common Stock | 2022-01-03 | 2022-11-07 | 214 | 5: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2022-11-07 |
| OKTA | BBG001YV1SM4 | Okta, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 3 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| ON | BBG000DV7MX4 | ON Semiconductor Corp | 2022-01-03 | 2026-11-03 | 1098 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| P | BBG00212PVZ5 | Everpure, Inc. | 2026-04-17 | 2026-11-03 | 32 | 4: 2026-04-17 2026-07-14 2026-10-12 2026-11-03 |
| PANW | BBG0014GJCT9 | Palo Alto Networks, Inc. Common Stock | 2022-01-03 | 2026-11-03 | 848 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| PCG | BBG000BQWPC5 | PG&E Corporation | 2022-01-03 | 2026-11-03 | 1026 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| PEAK | BBG000BKYDP9 | Healthpeak Properties, Inc. | 2022-01-03 | 2024-03-01 | 543 | 10: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-03-01 |
| PKI | BBG000FXW512 | Revvity, Inc. | 2022-01-03 | 2023-05-15 | 343 | 7: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-05-15 |
| PLTR | BBG000N7QR55 | Palantir Technologies Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 531 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| PODD | BBG000R7XX87 | Insulet Corporation | 2022-01-03 | 2026-11-03 | 914 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| PSKY | BBG01VS5NK99 | Skydance Corporation | 2025-08-07 | 2026-10-05 | 292 | 6: 2025-08-07 2025-11-05 2026-02-06 2026-05-08 2026-08-10 2026-10-05 |
| PSTG | BBG00212PVZ5 | Everpure, Inc. | 2022-01-03 | 2026-04-16 | 0 | 19: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-04-16 |
| Q | BBG01TRL73S5 | Qnity Electronics, Inc. | 2025-10-27 | 2026-11-03 | 252 | 6: 2025-10-27 2026-01-28 2026-04-29 2026-07-30 2026-10-28 2026-11-03 |
| RDDT | BBG005K1D4S0 | Reddit, Inc. | 2024-03-20 | 2026-11-03 | 55 | 12: 2024-03-20 2024-06-20 2024-09-19 2024-12-18 2025-03-24 2025-06-24 2025-09-23 2025-12-22 2026-03-25 2026-06-25 2026-09-24 2026-11-03 |
| RE | CIK0001095073 | Everest Group, Ltd. | 2022-01-03 | 2023-07-07 | 379 | 7: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 |
| S600 | BBG0019LSQ49 | First Republic Bank | 2023-11-20 | 2026-11-03 | 0 | 14: 2023-11-20 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| SATS | BBG000TGLV00 | EchoStar Corporation | 2022-01-03 | 2026-06-23 | 1121 | 19: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-06-23 |
| SMCI | BBG000MYZDJ3 | Super Micro Computer, Inc. Common Stock | 2024-10-01 | 2026-11-03 | 525 | 10: 2024-10-01 2024-12-31 2025-04-03 2025-07-07 2025-10-03 2026-01-05 2026-04-07 2026-07-08 2026-10-06 2026-11-03 |
| SNDK | BBG01R388JG1 | Sandisk Corporation Common Stock | 2025-02-13 | 2026-11-03 | 234 | 8: 2025-02-13 2025-05-15 2025-08-15 2025-11-13 2026-02-17 2026-05-18 2026-08-18 2026-11-03 |
| SOLS | BBG01X5744P5 | Solstice Advanced Materials Inc. Common Stock | 2025-10-20 | 2026-11-03 | 36 | 6: 2025-10-20 2026-01-21 2026-04-22 2026-07-23 2026-10-21 2026-11-03 |
| SOLV | BBG018YZH6T3 | Solventum Corporation | 2024-03-26 | 2026-11-03 | 652 | 12: 2024-03-26 2024-06-26 2024-09-25 2024-12-24 2025-03-28 2025-06-30 2025-09-29 2025-12-29 2026-03-31 2026-07-01 2026-09-30 2026-11-03 |
| SQ | BBG0018SLC07 | Block, Inc. | 2022-01-03 | 2025-01-17 | 0 | 14: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-01-17 |
| STLD | BBG000HGYNZ9 | Steel Dynamics Inc | 2022-01-03 | 2026-11-03 | 969 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| TKO | BBG01G9JKWV5 | TKO Group Holdings, Inc. | 2023-09-12 | 2026-11-03 | 407 | 14: 2023-09-12 2023-12-11 2024-03-13 2024-06-12 2024-09-12 2024-12-11 2025-03-17 2025-06-16 2025-09-16 2025-12-15 2026-03-18 2026-06-17 2026-09-17 2026-11-03 |
| TPL | BBG00YRJXKB1 | Texas Pacific Land Corporation | 2022-01-03 | 2026-11-03 | 485 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| TRGP | BBG0015XMW40 | Targa Resources Corp. | 2022-01-03 | 2026-11-03 | 1019 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| TTD | BBG00629NGT2 | The Trade Desk, Inc. | 2022-01-03 | 2026-11-03 | 295 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| TWLO | BBG0029ZX840 | Twilio Inc. | 2022-01-03 | 2026-11-03 | 21 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| UBER | BBG002B04MT8 | Uber Technologies, Inc. | 2022-01-03 | 2026-11-03 | 722 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| VEEV | BBG001CGB489 | Veeva Systems Inc. | 2022-01-03 | 2026-11-03 | 125 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| VICI | BBG00HVVB499 | VICI Properties Inc. Common Stock | 2022-01-03 | 2026-11-03 | 1106 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| VLTO | BBG019Q32XJ9 | Veralto Corporation | 2023-09-27 | 2026-11-03 | 776 | 14: 2023-09-27 2023-12-27 2024-03-28 2024-06-28 2024-09-27 2024-12-27 2025-04-01 2025-07-02 2025-10-01 2025-12-31 2026-04-02 2026-07-06 2026-10-02 2026-11-03 |
| VRT | BBG00L2B8KW8 | Vertiv Holdings Co Class A Common Stock | 2022-01-03 | 2026-11-03 | 157 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| VST | BBG00DXDL6Q1 | Vistra Corp. | 2022-01-03 | 2026-11-03 | 625 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| VYLR | BBG024CQR6X9 | Vylor Inc. | 2026-10-01 | 2026-11-03 | 24 | 2: 2026-10-01 2026-11-03 |
| WDAY | BBG000VC0T95 | Workday, Inc. Class A Common Stock | 2022-01-03 | 2026-11-03 | 467 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| WSM | BBG000FSMWC3 | Williams-Sonoma, Inc. | 2022-01-03 | 2026-11-03 | 407 | 21: 2022-01-03 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| WTW | CIK0001140536 | Willis Towers Watson Public Limited Company Ordinary Shares | 2022-01-10 | 2026-11-03 | 1209 | 21: 2022-01-10 2022-04-04 2022-07-06 2022-10-04 2023-01-04 2023-04-05 2023-07-07 2023-10-05 2024-01-05 2024-04-08 2024-07-09 2024-10-07 2025-01-07 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |
| XYZ | BBG0018SLC07 | Block, Inc. | 2025-01-21 | 2026-11-03 | 324 | 9: 2025-01-21 2025-04-09 2025-07-11 2025-10-09 2026-01-09 2026-04-13 2026-07-14 2026-10-12 2026-11-03 |


### pytest_tail.txt, pytest_mid.txt -- the full suite and the midoffice tests

```

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2253 passed, 1 skipped, 19 warnings in 71.88s (0:01:11) ===========
======================= 802 passed, 1 skipped in 29.27s =======================
```


### partA_check.txt -- the first change alone: its source, HEAD's tests and its test_descriptors.py

```
SKIPPED [1] tests\midoffice\test_conditioning.py:136: this platform's BLAS leaves a residue rather than cancelling exactly
SKIPPED [1] tests\midoffice\test_config.py:77: config/midoffice_risk.toml not present in this tree
SKIPPED [1] tests\midoffice\test_config.py:86: config/midoffice_risk.toml not present in this tree
======================= 779 passed, 3 skipped in 27.94s =======================
```


### ruff.txt -- every changed or new Python file, now against HEAD

```
== src/praxis/midoffice/model/descriptors.py
  now:
    
  HEAD:
    
== scripts/run_sp500_model.py
  now:
    F401 [*] `praxis.midoffice.model.covariance.HorizonCovariance` imported but unused
  HEAD:
    F401 [*] `praxis.midoffice.model.covariance.HorizonCovariance` imported but unused
== tests/midoffice/test_descriptors.py
  now:
    
  HEAD:
    
== src/praxis/midoffice/refdata/polygon.py
  now:
    F401 [*] `typing.Iterable` imported but unused
    F401 [*] `typing.Sequence` imported but unused
  HEAD:
    F401 [*] `typing.Iterable` imported but unused
    F401 [*] `typing.Sequence` imported but unused
== src/praxis/midoffice/refdata/sp500.py
  now:
    E501 Line too long (104 > 100)
    N818 Exception name `ParserUnavailable` should be named with an Error suffix
    N818 Exception name `SourceUnavailable` should be named with an Error suffix
  HEAD:
    E501 Line too long (104 > 100)
    F401 [*] `typing.Iterable` imported but unused
    N818 Exception name `ParserUnavailable` should be named with an Error suffix
    N818 Exception name `SourceUnavailable` should be named with an Error suffix
== tests/midoffice/test_prices.py
  now:
    F401 [*] `praxis.midoffice.refdata.polygon.Bar` imported but unused
    F401 [*] `praxis.midoffice.refdata.prices._sample_slots` imported but unused
    F401 [*] `urllib.error` imported but unused
    I001 [*] Import block is un-sorted or un-formatted
  HEAD:
    F401 [*] `json` imported but unused
    F401 [*] `praxis.midoffice.refdata.polygon.Bar` imported but unused
    F401 [*] `praxis.midoffice.refdata.prices._sample_slots` imported but unused
    F401 [*] `urllib.error` imported but unused
    I001 [*] Import block is un-sorted or un-formatted
== tests/midoffice/test_sp500.py
  now:
    F401 [*] `praxis.midoffice.refdata.sp500.Issue` imported but unused
  HEAD:
    F401 [*] `praxis.midoffice.refdata.sp500.Issue` imported but unused
== src/praxis/midoffice/refdata/identity.py
  now:
    
  HEAD:
    (new file)
== scripts/build_sp500_identity.py
  now:
    
  HEAD:
    (new file)
== tests/midoffice/test_identity.py
  now:
    
  HEAD:
    (new file)
```


### final_ast.txt -- ASCII and parse

```
src/praxis/midoffice/model/descriptors.py: non-ASCII lines 0; line ends CRLF 1130, LF 0, parses
scripts/run_sp500_model.py: non-ASCII lines 0; line ends CRLF 1070, LF 0, parses
tests/midoffice/test_descriptors.py: non-ASCII lines 0; line ends CRLF 1153, LF 0, parses
docs/FACTOR_MODEL.md: non-ASCII lines 0; line ends CRLF 862, LF 0
src/praxis/midoffice/refdata/polygon.py: non-ASCII lines 0; line ends CRLF 382, LF 0, parses
src/praxis/midoffice/refdata/sp500.py: non-ASCII lines 0; line ends CRLF 910, LF 0, parses
src/praxis/midoffice/refdata/identity.py: non-ASCII lines 0; line ends CRLF 1235, LF 0, parses
scripts/build_sp500_identity.py: non-ASCII lines 0; line ends CRLF 443, LF 0, parses
tests/midoffice/test_prices.py: non-ASCII lines 0; line ends CRLF 653, LF 0, parses
tests/midoffice/test_sp500.py: non-ASCII lines 0; line ends CRLF 736, LF 0, parses
tests/midoffice/test_identity.py: non-ASCII lines 0; line ends CRLF 304, LF 0, parses
claude/handoffs/BRIEF_midoffice05_identity_gate.md: non-ASCII lines 0; line ends CRLF 0, LF 89
```


### variant_F.txt -- the run's own report, the finished tree at its defaults

```
Panel
  1214 trading days, 506 securities
  2022-01-03 .. 2026-11-03
  583,081 of 614,284 return observations present (94.9%)

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, covering 1976-07-01 .. 2026-11-04
  one array for every standardisation and every regression below, each held to that day's members: 506 of the panel's 506 on 2022-01-03, 426 on 2026-11-03
  1 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
  - 2026-03-23 SATS recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 157 panel day(s) from then

Descriptors
  computed 14: LNCAP, LCAPCB, LPRI, DASTD, HILO, CMRA, BTSG, HALPHA, SERDP, RSTR, STOM, STOQ, STOA, VLVR
  - CMRA: computed on raw rather than excess returns: no risk_free supplied
  - RSTR: computed on raw rather than excess returns: no risk_free supplied
  - SERDP: computed at the final period only; the market-model residual series is retained for one window rather than all of them, so SERDP is a current exposure and not a time series
  - STO5: needs 1380 period(s) of history and the panel has 1214, so every value would be missing
  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression on the market over 252 trading days, half-life 63 trading days (the library's default, USE4's)
  window descriptors need at least 50% of the window valid; from a window with none, over 4,402,359 values: none; from one under 50% valid: none
  369,070 raw values winsorised before standardising
  102,179 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/46  HILO 8/45  CMRA 5/46  BTSG 7/49
    HALPHA 4/49  SERDP 0/49  RSTR 3/49  STOM 13/45  STOQ 14/46  STOA 16/49  VLVR 16/49

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         93%      962    8.36%      9.32%     1.34
  DASTD        93%    1,149    6.13%      7.77%     1.36
  RSTR         93%      938    5.31%      7.12%     1.11
  HALPHA       93%      962    4.43%      6.30%     0.56
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         92%      938    3.39%      5.59%     1.37
  STOQ         93%    1,145    2.99%      5.21%     1.26
  STOM         93%    1,191    2.95%      5.25%     1.28
  STOA         93%      938    2.57%      4.85%     1.50
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  VLVR         93%      938    1.78%      3.76%     1.95
  LPRI         93%    1,213    1.77%      3.77%    -0.41
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
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
  33 classification(s) carried from an earlier date (delisted inside the window): ABMD, ANSS, ATVI, AVB, CDAY
  482 of 506 securities are in at least one estimated period's regression while an index member; every count below is over them
  19 industries over 482 securities in the universe (506 classified); largest SIC73 with 53 (11%), 45 thin group(s) merged into their division
    SIC73                      53
    SIC38                      39
    SIC28                      39
    SIC49                      33
    Manufacturing              33
    Retail                     30
    Transport & Utilities      28
    SIC35                      27
  - 1 industry/industries have no member in the universe and are not factors: SIC24
  - 2 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture, Unclassified
  - Other Industries pooled only 3, below the hard floor of 5, so it is not a factor. 3 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA, FRC, SBNY
  - 3 industry/industries still have fewer than 15 members after merging and their factor returns are largely their members' specific returns: Construction (6), Mining (14), Wholesale (11)
  - 2 security(ies) have no usable SIC code and are left with no industry factor: FRC, SBNY
  - 24 classified security(ies) are outside the universe and count toward no industry: ABMD, CERN, CTXS, DISCA, DISCK

Model
  regression weights: square root of capitalisation
  24-factor model, annualised, estimated on 938 daily period(s) for 468 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.298
  0 collinear period(s) of 938 estimated; 0 factor-period(s) left out for having no member present
  membership read on the exposures' date: against same-day membership, 43 security-period(s) of the regressions move -- 42 in, on the day a name left the index, 1 out, on the day one joined; the mask itself differs on 62
  each risk index's blend over the 938 estimated periods: every descriptor from the first period whose regression reads it, and in how many
    Volatility           BTSG, DASTD, HILO, LPRI, CMRA from 2023-02-08 (938)
    Momentum             RSTR, HALPHA from 2023-02-08 (938)
    Size                 LNCAP from 2023-02-08 (938)
    Size Nonlinearity    LNCAP from 2023-02-08 (938)
    Trading Activity     STOM, STOQ, STOA, VLVR from 2023-02-08 (938)
  every estimated period's blend is the forecast exposures', 2026-11-03's, in every index: one definition per factor across the sample
  24 period(s) held enough securities and are not estimated, a risk index's blend in each not being the forecast's:
    2023-01-04 to 2023-02-07 (24): Volatility from BTSG, DASTD, HILO, LPRI, without CMRA; Momentum from HALPHA, without RSTR; Trading Activity from STOM, STOQ, without STOA, VLVR

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 3.3e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  +1.2e-17   1.000000   426
  Momentum                    -5.4e-17   1.000000   423
  Size                        -3.1e-17   1.000000   425
  Size Nonlinearity           +5.0e-17   1.000000   425
  Trading Activity            -7.7e-18   1.000000   425
  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,391 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.0e-15 (4.7 eps), inside rounding on 939 of the 939 dates

  The size pair on 2026-11-03, over the 423 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
                                                 Size  Size Nonlinearity     risk
  smallest 85, equally weighted                -3.110             -1.647   11.94%
  largest 50, equally weighted                 +0.012             +0.155    0.38%
  the index, cap-weighted                      +0.002             +0.000    0.01%

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     504    506          504          503
  of them in the panel      426     458    486          486          426
  in the regression         422     451    476          476          422

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               17.06%     +23.03%       52.0%      938          0      0
  Finance & Real Estate      15.31%     +21.56%       14.3%      938          0      0
  Manufacturing              10.95%     +20.59%       20.7%      938          0      0
  Mining                     20.16%     +10.85%       16.2%      938          0      0
  Retail                     13.09%     +13.07%       10.7%      938          0      0
  SIC20                      12.74%      +7.85%       30.6%      938          0      0
  SIC28                      13.11%     +15.02%        9.8%      938          0      0
  SIC35                      17.45%     +26.15%        6.3%      938          0      0
  SIC36                      20.75%     +28.22%        5.2%      938          0      0
  SIC37                      14.15%     +12.69%       18.2%      938          0      0
  SIC38                      14.19%     +15.35%       11.5%      938          0      0
  SIC49                      15.80%     +21.24%       12.3%      938          0      0
  SIC60                      21.92%     +19.03%        7.3%      938          0      0
  SIC63                      14.86%     +18.57%       13.8%      938          0      0
  SIC67                      15.19%     +17.12%       17.2%      938          0      0
  SIC73                      17.34%     +14.93%        3.5%      938          0      0
  Services                   12.86%     +14.04%       26.4%      938          0      0
  Transport & Utilities      13.34%     +12.35%       16.9%      938          0      0
  Wholesale                  14.77%     +18.69%       40.0%      938          0      0
  Volatility                  8.77%      +3.39%        3.8%      938          0      0
  Momentum                    6.62%      +3.60%        4.4%      938          0      0
  Size                        4.03%      +2.86%        7.1%      938          0      0
  Size Nonlinearity           2.54%      +0.40%       41.8%      938          0      0
  Trading Activity            5.07%      +4.40%       10.1%      938          0      0

  specific risk   median 23.77%   p10 15.40%   p90 36.50%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 0.9%, CTVA 0.9%
                    relative residual spread 0.133, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.361
                    largest shares of the sd's square: FRC 72.4%, SIVB 3.7%, LUMN 2.2%, BNY 1.5%, ENPH 1.0%
                    the characteristics' weight in a blended forecast, at the median history of 938 periods: 2.92%

Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
  - 38 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CDAY
  - 52% of the estimated variance of Construction is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 8 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: INFO, IPGP, MRSH, PENN, PVH
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

## git add

The separate change first -- the size pair's standing line, the two figures, and "none" -- its
own commit:

```
git add src/praxis/midoffice/model/descriptors.py scripts/run_sp500_model.py tests/midoffice/test_descriptors.py docs/FACTOR_MODEL.md
```

This change -- the identity gate, every file, the brief and this retro included:

```
git add src/praxis/midoffice/refdata/polygon.py src/praxis/midoffice/refdata/sp500.py src/praxis/midoffice/refdata/identity.py scripts/build_sp500_identity.py tests/midoffice/test_prices.py tests/midoffice/test_sp500.py tests/midoffice/test_identity.py claude/handoffs/BRIEF_midoffice05_identity_gate.md claude/retros/RETRO_midoffice05_identity_gate.md
```
