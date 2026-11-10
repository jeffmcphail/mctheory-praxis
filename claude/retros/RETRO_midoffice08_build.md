# Retro: Mid-office 08 -- the build

**Brief:** `claude/handoffs/BRIEF_midoffice08_build.md` -- written in Chat and pasted into the Code
session, transcribed verbatim
**Date:** 2026-11-10
**Duration:** the brief arrived shortly before 16:17Z (11:17 EST). Ruling 1 and the gate to
about 11:30; the first-count search and the build to about 12:05; the four model runs, the
comparison and the docs to about 12:20; the tests, the regressions and this retro to about 12:35.
**Status:** COMPLETE. The gate passes offline with ruling 1. The panel on securities is built --
prices stitched, total returns, traded volumes, counts per span with splits on their execution
dates, membership through the renames -- and all seven of its guards pass. The model is compared
in three steps, each measured on its own, and the docs are rewritten. Uncommitted: one `git add`
at the end, `claude/MIDOFFICE_STATE.md` in it.

---

## Summary

- **Ruling 1: the share-class rule, and Coherent, Inc.'s row.**
  - The rule is in the gate as a check, `rebuild.reconciles`: a record naming another composite
    FIGI names the expected security if its share-class FIGI is one that security's own records
    carry. The table stays keyed by composite FIGI.
  - Coherent, Inc.'s row went in under a new `--amend`, writing new `_amended` files. midoffice07's
    `_corrected` files regenerate byte for byte.
  - **One correction to midoffice07:** Coherent, Inc.'s last price under COHR is 2022-06-30, not
    2022-07-01. The endpoint names it on 2022-07-01, the day the purchase closed, and nobody traded
    then. The confirmation now states its last price, and checks the prices run to it exactly.
  - **The gate passes, offline, with no requests:** 628 of 628 spans, COHR's by its share class;
    47 splits, 8 price-only, no contradiction.
- **Ruling 2: the first-count search, and the weighted check.**
  - **The search:** 66 lookups over 12 spans, against the estimate of about 70.
  - **Member-days without a count:** 696 fall to 206.
    - 169 are inside spans: FedEx Freight's 109 and Vylor's 24, whose records carry none in the
      window, and the spin-offs' first days, 1 to 13 of them.
    - 37 are days the index carried a member past its last trade.
  - **Found:** some first records carry a transitional count, which Polygon corrects within days:
    - GE Vernova's carries GE's count, 4.0 times its own;
    - Amentum's is 0.63 of its own;
    - WBD's is 0.92 of its own.

    Carried as ruled and flagged; a decision below.
  - **The weighted check on carried counts:** 39 met, 3 flagged -- Carnival 1.118, Honeywell 0.498,
    Skydance 4.624. It confirms OKE's 1:1 carry (1.0000, 1.0001) and XOM's (1.0000, 0.9920,
    0.9841).
- **Ruling 3:** the panel stores traded volumes, the adjusted bars' beside them for the
  comparison.
- **The panel:**
  - **Shape:** 592 securities over 1,214 days. That is the 600 plan securities less the 8
    reorganisations merged into their successors.
  - **Prices:** 26 changes of ticker bridged by the raw closes; splices separated.
  - **Counts:** 47 splits applied, 15 stale samples read past.
  - **Total returns:** 8,229 distributions reinvested.
  - **Membership:** 610,842 member-days, exactly the walk through the renames.
  - **Guards:** all seven pass. The split guard checks 25,846 security-days, 25,260 of them
    member-days.
  - **Against the old panel:** on all 581,521 cells where the two take the same bar, the closes
    and volumes are identical and the returns agree to 2.2e-16. The old panel's 2,107 cells no new
    column takes are the splices' other securities.
- **The comparison, three steps** (938 periods from 2023-02-08 in each):

| | A: old panel | B: on securities | C: traded volumes | D: total returns |
|---|---|---|---|---|
| members in the regression a period, median (range) | 451 (422-476) | 502 (499-503) | 502 (499-503) | 502 (499-503) |
| mean R-squared | 0.298076 | 0.290903 | 0.292774 | 0.293189 |
| residual spread, 1.4826 x MAD (sd) | 0.1333 (0.3612) | 0.1362 (0.3418) | 0.1313 (0.3401) | 0.1330 (0.3405) |
| specific risk p10 / median / p90 | 15.40 / 23.77 / 36.50% | 15.60 / 24.83 / 40.75% | 15.68 / 24.79 / 40.72% | 15.59 / 24.80 / 40.78% |
| fallback forecasts | 8 | 11 | 11 | 11 |
| size pair: smallest 85 / largest 50 / index | 11.94 / 0.38 / 0.01% | 11.33 / 0.61 / 0.01% | 11.77 / 0.62 / 0.01% | 11.82 / 0.62 / 0.01% |

- **What each step does:**
  - **Identity (A to B):** 26 to 78 more members in each period's regression, and 87 more
    securities in the regression universe. SIC62 becomes its own industry. Nothing the old
    regressions held is lost: the two "missing", CDAY and FRC, are relabels (DAY, FRCB).
  - **Volumes (B to C):** Trading Activity's mean falls from +4.11% to +2.26%, and its volatility
    rises from 5.80% to 6.22%.
  - **Dividends (C to D):** every industry's mean rises, from +0.65% (Services) to +2.95%
    (SIC67), with correlation 0.945 against what its members paid. Volatility's falls 0.58%. The
    high yielders' specific risk falls most.
- **What the old panel had right that the new one loses:** on 252 member-days, the old panel's
  yearly samples carried the right count where the new panel's quarterly ones carry a count
  Polygon got wrong for a quarter: Quest Diagnostics at 0.104, Starbucks at 0.123, News Corp
  Class B at 0.849, Everest at 1.112. In all, 504 member-days sit on such single samples.
- **New standing checks:**
  - in the build: the weighted check, and every count move beyond 10% that no split explains --
    103, mostly mergers' issuance;
  - in the run: `current_stale`, and the members each regression keeps out, by why.
- **Docs:** `docs/FACTOR_MODEL.md`'s split paragraph is rewritten for the new rule, with the old
  panel's 4,196 member-days over 37 splits. It has a new section on the identity table, the gate,
  the build and what it changes.
- **Tests:** 2,290 passed, 1 skipped (midoffice: 839 passed, 1 skipped). 11 are new, 1 rewritten.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice08_build.md`, verbatim. In outline:

- **"Read claude/MIDOFFICE_STATE.md and do what its Start here section says."** Done: the
  rules (Rule 43 above all), the state file, midoffice07's retro, and `WORKFLOW_MODES_PRAXIS.md`
  (Rule 3). The brief has no "committed" line; `git log` shows midoffice07 committed (5ccbb70)
  and the state file after it (7228b01).
- **Rulings on midoffice07's four decisions:**
  1. COHR: (a), the share-class rule, kept a check, not a re-keying; add Coherent, Inc.'s row
     under COHR, BBG000BG1DH3, 2022-01-03 to 2022-07-01.
  2. Share counts the records do not carry: search each span's first quarter for its first
     record with a count, about 70 requests, and carry from that record's own date, forward
     only. Weighted counts are not data; they are a standing check, flagged beyond 10%.
  3. Volumes: the new panel stores traded volumes; the comparison measures the basis as its own
     step.
  4. Rule 43's ordering stays as worded; the state file travels in the retro's own commit.
- **midoffice08, the build:**
  - Run the gate first, offline.
  - Then items 2 to 5 of midoffice07's brief:
    - **2. the new panel:** securities as columns; prices stitched across tickers by the identity
      table, splices separated, reorganisations continuous; share counts per span, with splits
      applied on their execution dates; membership read through the renames and mapped to
      securities; sp500's `current_stale` warnings printed; the identity and split guards on the
      built panel.
    - **3. total returns:** each distribution on the prices' split basis by the ex-date's adjusted
      over raw close; one of TEL's pair kept; SPY's too.
    - **4. the model:** run and compared, reporting midoffice06's item 6 list -- per-period
      cross-section sizes and what still keeps a member out; R-squared; every factor's vol, mean
      and noise share, Construction's and Wholesale's especially; the residual spread; the
      specific-risk percentiles; the fallback list; the size pair's line; anything the old panel
      had right that the new one loses. For the dividends step, the factor and industry means
      especially.
    - **5. the docs:** the split paragraph and a section on the identity table and the rebuild.
  - The comparison is in three steps here, each measured on its own:
    - old panel against new, on price returns and adjusted volumes;
    - the new panel, adjusted volumes against traded;
    - the new panel, price returns against total returns.
- **"Retro and git add lines at the end, MIDOFFICE_STATE.md brought up to date in the same
  commit, then stop."**

---

## Files modified

| File | Change | Lines |
|---|---|---|
| `src/praxis/midoffice/refdata/rebuild.py` | the share-class rule: `own_identifiers`, `reconciles`, `IdentityCheck.reconciled`; `check_identities` reconciles a record by it, and a failure says why it could not; the module docstring | 19-21, 194-228, 240-242, 253-263, 291-305, 318 |
| `src/praxis/midoffice/refdata/identity.py` | `Confirmation.last_price`; `confirm_span` checks the prices run to it exactly, none after; its note | 1791-1794, 1806-1835 (hunks) |
| `src/praxis/midoffice/refdata/securities.py` | new: the panel on securities -- `columns_of`, `search_first_count`, `build_panel`, `identity_guards`, `split_guard`, `count_moves`, `weighted_check` | new, 685 lines |
| `scripts/build_sp500_identity.py` | `--amend` and `AMENDED`, Coherent, Inc.'s confirmation; the docstrings | 30-33, 44, 115-130, 179-181, 193-194, 207-208, 370-372 |
| `scripts/fetch_sp500_rebuild.py` | the amended plan by default; the spans passed by the share class printed, and `reconciled` in the report; the docstring | 8-12, 20-22, 26-27, 50, 197-204, 244-245 |
| `scripts/build_sp500_securities.py` | new: the gate offline, the first-count search, the build, its guards and standing checks, the industry codes, the panel and its report | new, 484 lines |
| `scripts/run_sp500_model.py` | the panel on securities: `load_panel`'s second kind, `--returns`, `--volumes`, the panel's own membership and SIC codes, its snapshot checked; `current_stale` printed; `kept_out` and its standing lines | 4-8, 90-161, 485-494, 555-566, 585-606, 805-830, 1113-1135 |
| `tests/midoffice/test_securities.py` | new: 9 tests | new, 222 lines |
| `tests/midoffice/test_rebuild.py` | the share-class test; ODD's case given another share class | 282, 294-295, 299-332 |
| `tests/midoffice/test_identity.py` | the last-price test | 556-580 |
| `docs/FACTOR_MODEL.md` | the split paragraph rewritten; a new section, the S&P 500 panel on securities | 777-804, 864-975 |
| `claude/MIDOFFICE_STATE.md` | brought up to date (Rule 43) | throughout |
| `claude/handoffs/BRIEF_midoffice08_build.md` | new: the brief, verbatim | new, 60 lines |
| `claude/retros/RETRO_midoffice08_build.md` | new: this retro | new |

Written outside the tracked files, all gitignored. Each is new; nothing that was in `data/`
before this piece was overwritten or deleted:

- `data/external/polygon/sp500_identity_2022-01-03_2026-11-03_amended.csv`, `.json`,
  `_members.csv`, `_fetch.csv`, `_links.csv`, `_membership.csv`: the amended table and plan.
- `data/external/polygon/sp500_rebuild_gate_2022-01-03_2026-11-03_amended.json`: the gate's
  report on the amended plan. midoffice07's `_corrected` report is untouched.
- `data/external/polygon/sp500_securities_2022-01-03_2026-11-03.npz` and `.json`: the panel and
  its report.
- In `data/external/polygon/cache/`, 101 answers:
  - 66 from the first-count search;
  - 35 not-found answers that the old panel's run asked for (below).
- `claude/scratch/midoffice08/`: the scratch, every script and output named here.

---

## Ruling 1: the share-class rule, and Coherent, Inc.'s row

### The rule, a check

`rebuild.reconciles(record, key, share_classes, ciks)` returns the share-class FIGI by which a
record names the security keyed `key`, or None. It reconciles only when all three hold:

- the record carries a composite FIGI;
- its share-class FIGI is one that security's own records carry;
- no CIK on it contradicts theirs.

`own_identifiers` reads "own" exactly as before: the records at the security's spans' samples
that name it by its key. `check_identities` passes such a record and says so -- "reconciled to
Coherent Corp. (BBG000BLW102) by its share-class FIGI, BBG001S5S6N6, which the expected
security's own records carry" -- and keeps it in `IdentityCheck.reconciled`. The table is not
re-keyed. A record that fails says which condition it failed:

- another share class;
- the expected security's share class, but a contradicting CIK;
- the expected security's share class, but no composite FIGI.

The CIK condition is stricter than the ruling, which says nothing of CIKs. I added it because
the CIK is the table's cross-check (05). It touches none of the seven records: COHR's carries no
CIK, and the six last-day records carry their securities' own.

### Coherent, Inc.'s row, and its last price

The ruling: "Add Coherent, Inc.'s row under COHR, BBG000BG1DH3, 2022-01-03 to 2022-07-01." It is
a confirmation, `AMENDED["COHR"]`, which replaces midoffice07's refused spec under `--amend`.

The first amended build refused it: "COHR's prices do not run unbroken from 2022-01-03 to
2022-07-01" (`amended_run1.txt`). The bars show why (`investigations.txt`):

- COHR's last price before II-VI took the ticker is 2022-06-30, at 266.22;
- nothing traded on 2022-07-01, the day II-VI's purchase closed;
- the next bar is Coherent Corp.'s, on 2022-09-08.

The endpoint names Coherent, Inc. on both 2022-01-03 and 2022-07-01. So midoffice07's account,
"its prices unbroken" to 2022-07-01, was a day too far. I kept the ruled span, both of whose ends
are records naming Coherent, Inc., and stated the last price in the spec:

- `Confirmation.last_price`, 2022-06-30;
- `confirm_span` checks the prices run unbroken from the first day to exactly that one, with
  none after it inside the span;
- every other confirmation is untouched, its last price its last day.

The table's row: `BBG000BG1DH3, COHR, 2022-01-03..2022-07-01, reference: COHR names Coherent
Inc (BBG000BG1DH3) on 2022-01-03 and 2022-07-01, its prices unbroken from the first to its last
price, on 2022-06-30`.

### The amended table

`build_sp500_identity.py --resolve --correct --amend --offline` wrote the `_amended` files.
Against `_corrected` (`amended_data.txt`, and the diff in the run log):

- **The table:** one row more -- 634 rows, 605 securities -- Coherent, Inc.'s.
- **The plan:** one cell -- COHR's span names `previous` BBG000BG1DH3.
- **The members, links and membership:** identical.
- **The four guards and the renamed-membership guard:** all pass.
- **No requests.**

`--correct` alone still writes midoffice07's table: re-run offline on the final code, its five
CSVs and its JSON are identical (`regressions.txt`).

### The gate again, offline: it passes

`fetch_sp500_rebuild.py --offline`, the amended plan now its default (`gate_amended.txt`):

- 628 spans, 600 securities, 617 tickers, 11,950 share samples;
- 0 requests, 0 failed;
- **identity:** 628 of 628 pass, 1 of them, COHR, by the share-class FIGI;
- **splits:** 55 events -- 47 splits, 8 price-only, 0 contradictions, 2 untested (LEN, HON);
- **Gate: passes.**

The report is a new file, `..._amended.json`. The gate re-run on a copy of midoffice07's
corrected plan changes one thing: COHR passes, reconciled. Its splits, distributions and failures
are unchanged, and its requests are 0 against the online run's 12 (`regress_gate_panel.txt`).

---

## Ruling 2: the first-count search, and the weighted check

### Which spans

A span is searched if it starts without a share count and no count is carried into it: none
from an earlier span of its security, none across a reorganisation. That is 15 spans
(`first_counts_plan.txt`):

- **Midoffice07's ten with member-days that had no count:** CEG, WBD, GEHC, VLTO, SOLV, GEV,
  AMTM, PSKY, SOLS, Q.
- **KVUE and SNDK:** the rule is general. Their first counts come before their first member-day,
  so no member-day turns on them.
- **FedEx Freight and Vylor's two spans:** no sample in them carries a count, so there is nothing
  to bracket.

### The search

`securities.search_first_count` bisects the trading days between the span's first sample (no
count) and its first sample with one. It assumes a record that once carries a count goes on
carrying one. Every record it asks for must name the span's security, by the gate's rule; one
naming another stops the search. All twelve brackets lie within a quarter.

| Span | Bracket | First record with a count | Its count | The bracket's end | Lookups |
|---|---|---|---|---|---|
| CEG | 2022-02-02..2022-04-20 | 2022-02-03 | 326,500,000 | 326,660,000 | 5 |
| WBD | 2022-04-11..2022-07-13 | 2022-04-12 | 2,233,730,000 | 2,426,840,000 | 5 |
| GEHC | 2023-01-04..2023-03-20 | 2023-01-05 | 454,710,000 | 453,930,000 | 5 |
| KVUE | 2023-05-04..2023-08-03 | 2023-05-08 | 1,888,970,000 | 1,914,890,000 | 6 |
| VLTO | 2023-10-02..2023-12-27 | 2023-10-04 | 246,060,000 | 246,310,000 | 6 |
| SOLV | 2024-04-01..2024-06-26 | 2024-04-02 | 172,710,000 | 172,710,000 | 5 |
| GEV | 2024-04-02..2024-06-27 | 2024-04-03 | 1,088,330,000 | 274,090,000 | 5 |
| AMTM | 2024-09-30..2024-12-23 | 2024-10-01 | 153,280,000 | 243,300,000 | 5 |
| SNDK | 2025-02-24..2025-05-15 | 2025-02-26 | 144,000,000 | 145,300,000 | 6 |
| PSKY | 2025-08-07..2025-11-05 | 2025-08-11 | 1,064,544,631 | 1,064,544,631 | 6 |
| SOLS | 2025-10-30..2026-01-21 | 2025-11-18 | 158,727,456 | 158,727,456 | 6 |
| Q | 2025-11-03..2026-01-28 | 2025-11-20 | 209,463,194 | 209,463,194 | 6 |

That makes 66 lookups, all over the network the first time and cached, so later runs ask
nothing. Only the found first record enters the panel, as a sample; the search's other records
with counts do not. That keeps the panel's grid the plan's, and it keeps out a record Polygon
got wrong: SNDK's 114,860,000 on 2025-03-14 and 2025-04-03, against 144 to 145 million either
side.

### What the search found: transitional first counts

Every record the search asked for is in `investigations.txt`. In three spans, the first record
with a count carries a figure Polygon corrects within days. Carried as ruled, each stands until
the plan's next sample, a quarter later:

| Span | First count | Polygon's later count | From | Panel's count, against it | Member-days |
|---|---|---|---|---|---|
| GEV | 1,088,330,000 (04-03, 04-05) -- GE's own count | 272,080,000 | 2024-04-11 | 4.0x | 59 |
| AMTM | 153,280,000 (10-01, 10-03) | 243,300,000 | 2024-10-09 | 0.63x | 58 of its 59 |
| WBD | 2,233,730,000 (04-12 to 04-21) | 2,426,840,000 | 2022-05-03 | 0.92x | 62 |

GE Vernova's weighted count is right throughout (272,083,576); Amentum's and WBD's are
transitional too. The weighted check flags GEV's and AMTM's first records, and the count-move
check flags their resets at the next sample. Left as ruled; a decision below.

### The member-days without a count: 696 to 206

Of the 206:

- **169 inside the spans, before a first count:**
  - FDXF, 109;
  - Vylor, 24;
  - Q, 13, and SOLS, 13 -- their first records with a count came on 2025-11-20 and 2025-11-18;
  - one or two first days each for AMTM, CEG, GEHC, GEV, SOLV, VLTO (2), WBD and Skydance (2).
- **37 past a member's last trade:** the index carried 15 members a day or few after their last
  trade (XLNX, INFO, CTXS, TWTR, SIVB and others), and two more across a gap in their prices
  (Signature Bank, First Republic). The identity table maps those days on purpose -- "after its
  last price", "inside a gap in its prices" -- and no price or count reaches them.

### The weighted check, standing

Wherever a record carries `weighted_shares_outstanding`, it is set against the count the panel
carries that day, and flagged beyond 10%: 11,894 met in all.

- **Carried from an earlier record: 39 met, 3 flagged** -- the ruling's check:

| Security | Day | Weighted | The panel's count | Ratio |
|---|---|---|---|---|
| CCL | 2026-05-07 | 1,385,139,577 | 1,239,000,212 | 1.1179 FLAGGED (Carnival's carry across its unification, as ruled) |
| HON | 2026-07-14 | 316,826,559 | 635,675,701 | 0.4984 FLAGGED |
| SKYD | 2026-11-03 | 5,042,178,087 | 1,090,445,692 | 4.6240 FLAGGED (Skydance after it took WBD) |
| OKE | 2026-09-11, 2026-11-03 | 630,369,667; 630,450,437 | 630,369,667 | 1.0000, 1.0001 |
| XOM | 2026-07-02, 2026-10-01, 2026-11-03 | 4,144,947,162; 4,111,911,960; 4,079,000,000 | 4,144,947,162 | 1.0000, 0.9920, 0.9841 |

  The other 31 carried ones lie within 0.92 to 1.04, all listed in the appendix.

- **The day's own record: 11,855 met, 762 flagged, over 75 securities.**
  - Nearly all are companies with several share classes, flagged on every record: GOOG, GOOGL,
    BRK.B, FOX, FOXA, NWS, NWSA, UA, UAA, ABNB, DELL, ECHO and others. A record's share-class
    count is its own class; its weighted count spans them all.
  - The rest are the single samples below, and AMTM's first record.

### Counts that move with no split: a second standing check

The gate tests counts across split events only. `securities.count_moves` reports every day a
column's carried count moves beyond 10% other than by a split the gate applies: 103 moves.

- **Most are mergers' issuance:**
  - 2022: AMD and Xilinx (1.348), APO and Athene (2.315), KKR (1.451);
  - 2024 and 2025: Smurfit Westrock (1.991), Exxon and Pioneer (1.131), Capital One and
    Discover (1.669);
  - 2026: Omnicom and IPG (1.624), Fifth Third (1.370).
- **Some are Polygon's errors.** A single sample wrong for a quarter, the next one reversing it:

| Security | Sample | Count | Against the count either side | Member-days | The old panel there |
|---|---|---|---|---|---|
| DGX | 2025-04-09 | 11,610,000 | 0.104 | 63 | 111,620,000, right |
| SBUX | 2026-07-14 | 139,700,000 | 0.123 | 63 | 1,137,100,000, right |
| XYZ | 2026-01-09 | 59,998,000 | 0.109 | 63 | not in the old panel |
| NWS | 2024-07-09 | 162,170,000 | 0.849 | 63 | 191,390,000, right |
| EG | 2025-04-09 | 47,790,000 | 1.112 | 63 | 42,980,000, right |
| CEG | 2026-01-27, 2026-04-28 | 362,355,476, 313,309,685 | 1.160, 0.865 | 126 | not in it |
| PCG | 2026-01-09 | 2,197,910,425 | 0.821 | 63 | not in it |
| RDDT | 2025-12-22 | 51,670,930 | 0.384 | 0 | not in it |

  That makes 504 member-days on such single samples. PCG also alternates between two counts over
  years, near 2.0 or 2.2 billion and 2.47 or 2.68 billion (`extras.txt`).

- **And two the data cannot settle:**
  - **HON** halves on 2026-10-12, from 635,675,701 to 316,940,010. There is no split event, and no
    day's raw close moves more than 15%; on HONA's first day, 2026-06-29, HON opened higher. HON's
    counts from then are within 0.06% of Honeywell Aerospace's (316,826,559 to 317,031,836), and
    its weighted count of 2026-07-14 is HONA's first count exactly. A contaminated record is the
    likelier reading. The panel carries the halved count from 2026-10-12, 17 trading days.
  - **SKYD**, after Skydance took WBD on 2026-10-06, has records with no share-class count, so the
    panel carries its pre-merger 1,090,445,692 to the end: 21 member-days, against a weighted
    count 4.6 times it.

---

## Ruling 3: traded volumes

The panel stores two volumes:

- **`volumes`**, the raw bars' share volume: the traded one, on the basis of the traded price and
  a point-in-time count.
- **`adjusted_volumes`**, the adjusted bars', put on the stitched prices' basis.

`run_sp500_model.py --volumes` picks between them, by default the stored traded ones. On the
cells shared with the old panel, the old panel's volume equals the adjusted volume put back on its
ticker's basis on all 581,521 cells. It equals the traded volume on 557,135. The other 24,386 are
exactly midoffice07's count of cells off by the adjustment (`crosscheck_old.txt`).

The turnovers' formula, a window's summed volume over its mean count, is consistent across a split
on traded volumes. Where turnover is constant either side, it is exact. On the adjusted bars'
volumes, every pre-split day inflated it.

---

## Item 2: the panel on securities

`scripts/build_sp500_securities.py`, offline in about 25 to 40 seconds; the module is
`refdata.securities`. In order:

1. **The gate, offline.** The build stops unless it passes. It passes: 628 of 628, 47 splits,
   8 price-only, 0 contradictions.
2. **The search**, as above.
3. **The build.**
4. **The guards**, any failing a non-zero status. Then the standing checks.

### What it holds

- **Columns:** one per security, labelled by its last ticker; no label repeats. 600 plan
  securities, less 8 reorganisations merged into their successors (`Column.keys`, oldest first),
  give 592 columns. The merged eight are BG, BLK, CCL, J, LH, OKE, VYLR, XOM.
- **Splices:** a ticker another security held before a column's span never enters that column --
  BNY, COHR, DOC and META, besides INFO, whose IHS Markit span comes first. On all four,
  checked: the old panel's 2,107 cells that no new column takes are BNY's muni trust (1,028),
  Physicians Realty under DOC (542), the ETF under INFO (518) and Roundhill under META (19).
- **Prices, stitched:**
  - within a run of one ticker's bars, the ticker's adjusted closes times one scale for the run;
  - across a change of ticker, the raw closes' return times any split event of the security's
    between the two days, each earlier run scaled to continue the later;
  - 26 changes of ticker bridged (`Bridge`). Every scale is 1 but VIAC's, 200: Paramount
    Global's adjusted history under PARA carries two reverse splits of the ticker's next holder,
    20-for-1 on 2026-05-08 and 10-for-1 on 2026-10-27, a constant over Paramount's span
    (`investigations.txt`). The scale absorbs it, and the returns are unaffected.
  - First Republic's bridge, FRC on 2023-04-28 to FRCB on 2023-05-03, is -90.5% across two
    unpriced days, so the return there is missing, as every return across a gap is.
- **The calendar:** SPY's adjusted bars, 1,214 days to 2026-11-03, the old panel's. The
  rebuild's bars fetched later reach 2026-11-04, outside it.
- **Counts:**
  - **Splits:** the 47 the gate classifies as splits are applied on their execution dates.
  - **Stale samples:** 15 read past, among them WSM's and SMCI's on their execution dates and
    MNST's and APH's to the window's end.
  - **Carries into a span:** 8 -- Carnival's reorganisation to its first sample, 2026-08-07;
    OKE's and XOM's 1:1 to the end, flagged; COHR from IIVI's last sample to its first; SMCI
    across its own two spans; FRCB, SBNY and SKYD to their spans' ends, flagged.
  - **Before a count:** nothing is back-filled; the days before a security's first count have
    none.
- **Membership:** the table's `_amended_membership.csv`, membership() read through the renames
  and mapped to securities, under each column's keys: 610,842 member-days. The walk through the
  table's 51 renames gives the same 610,842, every one mapped.
- **`current_stale`:** "the constituent list predates the change of 2026-10-30 -- OKTA added,
  KVUE removed -- and is read with it applied", printed by the build and by the run.
- **Industries:** each security's SIC code from its own latest sample record with one -- 587 of
  592. None for DISCA, DISCK, FRCB, SBNY and VYLR, none of whose cached records carries one: the
  old run had none for them either.

### The guards

| Guard | Checked | Failures |
|---|---|---|
| each (ticker, date) price feeds one column | 685,133 | 0 |
| no column takes two tickers on one date: its spans do not overlap | 628 | 0 |
| every member-day is a column's, inside its spans or on one the table maps outside them | 610,842 | 0 |
| every bar under a span's ticker inside the span is its column's, and only those | 685,089 | 0 |
| every member-day of the walk through the renames is a column's member-day | 610,842 | 0 |
| the panel's member-days are the walk's, no more | 610,842 | 0 |
| split guard, exact: no day on or after a split carries a pre-split count, and the split's day carries the count before it times the ratio | 25,846 (25,260 member-days) | 0 |

`split_guard` reads the built counts, not the code that built them. It catches both failure
kinds: a count from before the split on a later day, and an execution date not the count before
it times the ratio (tested).

### Against the old panel, cell by cell (`crosscheck_old.txt`)

- **Shared cells:** 581,521, where both panels take one ticker's bar on one day.
- **Raw closes:** equal exactly on all of them. **The adjusted bars' volume:** equal exactly.
- **Adjusted closes:** new over old, 1 on every cell -- each shared cell is in its column's last
  run, or in a run whose scale is 1.
- **Returns:** within a run, equal to 2.2e-16.
- **Counts:** on 11,529 shared member-days the old count is more than 10% from the new. Mostly
  the yearly grid's staleness across splits and issuance:
  - GM, 378 days at 1.105 to 1.218;
  - MPC, 378;
  - WRB's two splits, 320 at 0.67;
  - NVDA, 145 at 0.10;
  - CMG, 134 at 0.020.

  Some are identity: META's old count was Roundhill's, 0.027 of Meta's.

---

## Item 3: total returns

Every cash distribution under the security's tickers inside its spans, regular and special
alike, is reinvested on its ex-date. It is put on the prices' split basis by the ex-date's ratio
of the column's price to its raw close, so (P_t + D x P_t / raw_t) / P_s for the previous priced
day s.

- **Reinvested:** 8,229. That is the gate's 8,234, less TEL's duplicate of 2024-11-22 (its two
  records differ only in Polygon's `id`), less 4 on the window's first day, which no return can
  carry (A, DG, PWR, RJF on 2022-01-03).
- **Ex-dates without a price in their column:** none.
- **Over the members' days:** total less price return averages 1.90% a year.
- **SPY:** 19 distributions, 1.31% a year above its price return. The market series is SPY's
  price return or total return, to match `--returns`.

---

## Item 4: the model, in three steps

`run_sp500_model.py`, at today's defaults (`--industries --membership`), four runs:

- **A:** the old panel.
- **B:** the new panel, `--returns price --volumes adjusted`.
- **C:** the new panel, `--returns price --volumes traded`.
- **D:** the new panel, `--returns total --volumes traded`.

Each took 47 to 75 seconds. The reports are in the appendix in full. To read them at full
precision, `capture.py` (scratch) ran each again in-process, wrapping the estimator to keep its
model; its reports equal the standalone runs' but for the appended exit line. All four estimate
the same 938 periods, from 2023-02-08: 24 dropped for blends unlike the forecast's, 2023-01-04
to 2023-02-07, as before.

**Run A reproduces the old run** -- mean R-squared 0.298076 over 938 periods, the size pair at
11.94%, 0.38% and 0.01%, Construction 52.0%, Wholesale 40.0%, Size Nonlinearity 41.8% -- so
the run script's old path is unchanged.

### Step 1, identity: A to B

**Cross-section sizes.** Members in the regression a period:

| | min | median | max | first | last |
|---|---|---|---|---|---|
| A | 422 | 451 | 476 | 476 | 422 |
| B | 499 | 502 | 503 | 502 | 500 |

B less A, per period: +26 to +78, median +50. The index is 503 to 505 members.

**What keeps a member out**, over the 938 periods (the run's new standing line):

| | A | B |
|---|---|---|
| member-periods | 472,608 | 471,826 |
| in the regression | 422,203 | 470,548 |
| not in the panel | 44,038 | 0 |
| no weight (no cap on the day the weights are read) | 6,114 -- VMRK, SKYD, ECHO, MRSH 938 each, FISV 610, SW 481 | 197 -- FDXF 108, VYLR 23, Q 13, SOLS 13, JNPR 4, HES 3 |
| no return | 23 | 23 |
| an exposure missing, all Momentum | 230 -- FISV 126, GEN 65, COR 38, EG 1 | 1,058 -- SW 126, SOLV 125, GEV 125, VLTO 124, SKYD 124, Q 113 |

On securities, what still keeps a member out:

- **Mostly Momentum:** new listings and spin-offs have fewer than RSTR's half-year of returns.
- **Then the missing counts:** FDXF, Vylor, and Q's and SOLS's first days.

**The regression universe:** 482 securities in A, 567 in B.

- **In A, not in B:** CDAY and FRC only, now labelled DAY and FRCB.
- **In B, not in A: 87.** The index's joiners, among them ABNB, APO, ARES, BX, CEG, COIN,
  CRWD, DASH, GEHC, GEV, HOOD, KKR, KVUE, PLTR, SMCI, UBER and VST. Also the renamed members the
  old panel could not price: ECHO, MRSH, SKYD and VMRK.

**R-squared:** 0.298076 to 0.290903, -0.007173, with 87 more names, many of them volatile.

**Factors** (annual vol, annual mean, noise share; the 25th factor, SIC62, only in B):

| Factor | vol A | vol B | mean A | mean B | noise A | noise B |
|---|---|---|---|---|---|---|
| Construction | 17.06% | 19.30% | +23.03% | +24.86% | 52.0% | 38.8% |
| Finance & Real Estate | 15.31% | 15.36% | +21.56% | +17.96% | 14.3% | 31.0% |
| Manufacturing | 10.95% | 10.13% | +20.59% | +18.92% | 20.7% | 23.6% |
| Mining | 20.16% | 18.83% | +10.85% | +12.39% | 16.2% | 18.0% |
| Retail | 13.09% | 13.23% | +13.07% | +12.62% | 10.7% | 10.6% |
| SIC20 | 12.74% | 12.15% | +7.85% | +8.17% | 30.6% | 32.8% |
| SIC28 | 13.11% | 13.14% | +15.02% | +15.27% | 9.8% | 10.3% |
| SIC35 | 17.45% | 18.87% | +26.15% | +28.28% | 6.3% | 5.2% |
| SIC36 | 20.75% | 21.16% | +28.22% | +30.12% | 5.2% | 4.5% |
| SIC37 | 14.15% | 14.37% | +12.69% | +13.60% | 18.2% | 18.6% |
| SIC38 | 14.19% | 14.78% | +15.35% | +15.34% | 11.5% | 10.4% |
| SIC49 | 15.80% | 15.86% | +21.24% | +21.67% | 12.3% | 11.3% |
| SIC60 | 21.92% | 21.79% | +19.03% | +19.74% | 7.3% | 7.5% |
| SIC62 | -- | 18.77% | -- | +17.42% | -- | 11.5% |
| SIC63 | 14.86% | 14.51% | +18.57% | +18.43% | 13.8% | 14.7% |
| SIC67 | 15.19% | 15.00% | +17.12% | +15.69% | 17.2% | 16.8% |
| SIC73 | 17.34% | 18.23% | +14.93% | +14.84% | 3.5% | 2.9% |
| Services | 12.86% | 12.70% | +14.04% | +14.44% | 26.4% | 28.4% |
| Transport & Utilities | 13.34% | 13.71% | +12.35% | +12.84% | 16.9% | 15.9% |
| Wholesale | 14.77% | 13.96% | +18.69% | +19.57% | 40.0% | 46.5% |
| Volatility | 8.77% | 9.03% | +3.39% | +2.50% | 3.8% | 3.7% |
| Momentum | 6.62% | 7.25% | +3.60% | +4.26% | 4.4% | 3.6% |
| Size | 4.03% | 3.67% | +2.86% | +3.13% | 7.1% | 7.9% |
| Size Nonlinearity | 2.54% | 2.21% | +0.40% | -0.55% | 41.8% | 50.1% |
| Trading Activity | 5.07% | 5.80% | +4.40% | +4.11% | 10.1% | 7.5% |

- **Construction:** members 6 to 8; noise share 52.0% to 38.8%.
- **Wholesale:** members 11 to 12; noise share 40.0% to 46.5%.
- **Mining:** now over the 15-member floor (14 in A).
- **SIC62:** a factor of its own, 20 members with a forecast on 2026-11-03, with enough brokers
  and asset managers among the joiners to clear the floor. It takes 14 securities that were in Finance &
  Real Estate: AMP, BEN, BLK, CBOE, CME, GS, ICE, IVZ, MKTX, MS, NDAQ, RJF, SCHW and TROW. Those
  are the only industry changes among the 465 securities with one in both runs (`extras.txt`).
- **Finance & Real Estate:** left with 12 members, and a noise share of 31.0%.

**The residual spread:** 0.1333 to 0.1362, the sd 0.3612 to 0.3418. FRC's share of the sd's
square, 72.4%, is FRCB's 66.7%.

**Specific risk:**

- **Over every security with a forecast:**
  - A: p10 15.40%, median 23.77%, p90 36.50%, over 468;
  - B: p10 15.60%, median 24.83%, p90 40.75%, over 552.
- **Over the 467 in both:** p10 15.29%, median 23.78%, p90 36.46%, within about 0.1 of a point.
  The rise is the joiners'.
- **Of those in both, the largest falls:**
  - BNY by 14.0 points, MRSH by 11.0, CPAY by 3.2: their histories now their own;
  - AJG by 5.1, AON by 4.5, BRO by 4.4: the insurance brokers. Finance & Real Estate, without
    SIC62's fourteen, is twelve members, five of them brokers, Marsh McLennan and WTW now among
    those regressed.
- **The largest rises:**
  - FISV by 2.7 points: its whole history now, where the old FISV column missed its FI years;
  - DHI by 2.0, PHM by 1.8, LEN by 1.8, NVR by 1.7: Construction's two new members, the
    contractors EMCOR and Comfort Systems (EME, FIX), fit the homebuilders less closely;
  - CME and CBOE by 1.6 and 1.5: SIC62's brokers fit the exchanges less closely than Finance &
    Real Estate did.

**Fallback forecasts:**

- **A, 8:** INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO.
- **B, 11:** AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO.
- **New in B:** AMTM, FBIN, GAP, MBC, SOLS, in no estimated period's regression.
- **Gone from B:**
  - INFO, whose old forecast was built from an ETF's prices;
  - MRSH, now regressed.

**The size pair** on 2026-11-03 (mean Size, Size Nonlinearity exposure; the pair's risk):

| Portfolio | A | B |
|---|---|---|
| smallest 85, equally weighted | -3.110, -1.647: 11.94% | -3.139, -1.810: 11.33% |
| largest 50, equally weighted | +0.012, +0.155: 0.38% | +0.170, +0.052: 0.61% |
| the index, cap-weighted | +0.002, +0.000: 0.01% | +0.001, -0.001: 0.01% |

### Step 2, volumes: B to C

- **Sizes, the universe, the fallback list:** unchanged.
- **R-squared:** 0.290903 to 0.292774, +0.001871.
- **Trading Activity:** mean +4.11% to +2.26%, vol 5.80% to 6.22%, noise 7.5% to 7.6%.
- **The screen** (the turnover descriptors explain more, VLVR less):

| Descriptor | Mean R2, B | Mean R2, C | Factor vol, B | Factor vol, C |
|---|---|---|---|---|
| STOQ | 3.21% | 3.58% | 5.71% | 5.92% |
| STOM | 3.11% | 3.46% | 5.67% | 5.87% |
| STOA | 3.04% | 3.25% | 5.71% | 5.78% |
| VLVR | 1.77% | 1.55% | 3.95% | 3.71% |

- **The other factors move with Trading Activity.** Volatility's mean is +2.50% to +3.66%.
  Industries move by -1.06% (Construction) to +1.09% (Services). Vols move by 0.42% at most, and
  noise shares change little: Size Nonlinearity's is 50.1% to 47.1%.
- **The residual spread:** 0.1362 to 0.1313.
- **Specific risk percentiles:** 15.60 / 24.83 / 40.75% to 15.68 / 24.79 / 40.72%. The largest
  falls are IPGP 2.8 points, SBNY 2.4, AMTM 2.0, and the split securities TPL and ORLY.
- **The size pair's smallest 85:** 11.33% to 11.77%.

### Step 3, dividends: C to D

- **Sizes, the universe, the fallback list:** unchanged.
- **R-squared:** 0.292774 to 0.293189, +0.000415.
- **Vols:** move by 0.14% at most. **Noise shares:** by 0.6% at most.

**The factor and industry means.** Each factor's mean return shift, paired per period and
annualised, with its standard error. Beside it, the industry's members' own distributions: the
sqrt-cap-weighted average of total less price return over each period's members, annualised.

| Factor | mean C | mean D | shift (se) | members paid |
|---|---|---|---|---|
| SIC67 (REITs) | +15.05% | +18.00% | +2.948% (0.361%) | +3.891% |
| SIC60 (banks) | +19.57% | +21.96% | +2.391% (0.288%) | +3.162% |
| Manufacturing | +18.69% | +20.90% | +2.203% (0.253%) | +2.770% |
| SIC49 (utilities) | +21.87% | +24.01% | +2.134% (0.224%) | +3.168% |
| SIC20 (food) | +9.10% | +11.04% | +1.939% (0.272%) | +2.986% |
| SIC62 | +16.72% | +18.62% | +1.900% (0.208%) | +2.334% |
| SIC28 | +14.81% | +16.64% | +1.827% (0.181%) | +2.354% |
| Transport & Utilities | +13.72% | +15.39% | +1.663% (0.253%) | +2.243% |
| Mining | +13.34% | +14.96% | +1.623% (0.269%) | +2.173% |
| SIC36 | +31.06% | +32.58% | +1.513% (0.102%) | +0.942% |
| SIC37 | +13.94% | +15.43% | +1.494% (0.248%) | +1.682% |
| SIC35 | +28.43% | +29.69% | +1.260% (0.097%) | +1.255% |
| SIC63 | +17.93% | +19.16% | +1.231% (0.166%) | +1.695% |
| Retail | +13.62% | +14.73% | +1.113% (0.098%) | +1.367% |
| Finance & Real Estate | +17.05% | +18.11% | +1.061% (0.146%) | +1.266% |
| Wholesale | +19.72% | +20.70% | +0.977% (0.149%) | +1.431% |
| Construction | +23.80% | +24.77% | +0.972% (0.146%) | +0.673% |
| SIC73 | +14.26% | +15.18% | +0.915% (0.104%) | +0.645% |
| SIC38 | +15.08% | +15.82% | +0.733% (0.115%) | +0.829% |
| Services | +15.53% | +16.18% | +0.650% (0.135%) | +0.707% |
| Volatility | +3.66% | +3.08% | -0.575% (0.114%) | |
| Momentum | +4.45% | +4.23% | -0.219% (0.108%) | |
| Size | +3.23% | +3.14% | -0.089% (0.034%) | |
| Size Nonlinearity | -0.62% | -0.51% | +0.115% (0.041%) | |
| Trading Activity | +2.26% | +2.48% | +0.223% (0.059%) | |

- **Every industry's mean rises,** by +0.65% to +2.95% a year, the high yielders most: REITs,
  banks, utilities, food. The shifts track what the members paid, correlation 0.945 across the
  20 industries. They run 0.35% under it on average, at most 1.05%; plausibly the style factors
  take part of it -- Volatility's mean falls 0.58% (se 0.11%), its low-volatility end being the
  payers -- though that split was not measured.
- **Price returns understated every industry's mean,** by about each industry's yield, which is
  the bias the dividends were fetched to remove.
- **Specific risk:** the largest falls are the high yielders' -- MO 0.58 points, TDG 0.50 (its
  specials), VZ 0.44, PFE 0.37, PGR 0.34 (its annual variable), OKE 0.32, UPS 0.30. Every
  ex-date was a small regular move in their specific returns. Percentiles: 15.68 / 24.79 /
  40.72% to 15.59 / 24.80 / 40.78%.
- **The residual spread:** 0.1313 to 0.1330.
- **The size pair's smallest 85:** 11.77% to 11.82%.

### What the old panel had right that the new one loses

- **Share counts on Polygon's single bad samples.** On 252 member-days, the old panel's yearly
  samples carried the right count where the new panel's quarterly ones carry a sample Polygon got
  wrong: DGX (0.104), SBUX (0.123), NWS (0.849) and EG (1.112), 63 each. Sampling four times as
  often takes in more of Polygon's errors. 504 member-days in all are on single samples like
  these (above); a decision below.
- **Two industries' fit.**
  - The exchanges (CME, CBOE) and homebuilders (DHI, PHM, LEN, NVR) fit their new industries less
    closely: their specific risk rises 1.5 to 2.0 points.
  - Finance & Real Estate, without the brokers, is thin: 12 members, a noise share of 31%, where
    it was 14%.
  - Wholesale's noise share is 46.5% against 40.0%, and Size Nonlinearity's 50.1% against 41.8%.

  These are the industry scheme meeting a fuller universe rather than errors, and they belong to
  the model-wide noise-share question.
- **Nothing in prices or membership.** Every price the new panel drops was another security's,
  every regression member of the old run is in the new one (relabelled CDAY and FRC aside), and
  every member-day maps.

---

## Item 5: the docs

`docs/FACTOR_MODEL.md` (CRLF kept, ASCII):

- **The split paragraph** (lines 777-804), rewritten for the new rule:
  - the old panel's yearly grid and its measured staleness, 4,196 member-days over 37 splits;
  - its volume basis, 24,386 cells over 39 tickers;
  - the classification by ratio, counts and price, DuPont's 0.33 a whole ratio its counts follow;
  - the lag rule and the exact guard, with this build's figures.

  The old paragraph's "sampled every 63 trading days", "896 security-days, 621 member-days" and
  "DD's ... 0.33 are not split ratios" are gone.
- **A new section, "The S&P 500 panel on securities"** (lines 864-975), before "Status":
  - why a ticker is not a security;
  - the identity table: keys, sources, corrections, guards, the three kinds of boundary,
    membership through the renames;
  - the gate, with the share-class reconciliation;
  - the build: prices stitched, total returns, volumes, counts, the search;
  - the two standing checks;
  - what it changes, the four runs in one table.

  It says that the document's other figures are the old panel's.

---

## Key decisions

1. **The share-class rule coded once, `rebuild.reconciles`**, used by the gate and by the
   search's identity check. It has one condition stricter than ruled: a CIK on the record
   contradicting the expected security's blocks it. It touches none of the seven.
2. **Coherent, Inc.'s row under a new `--amend`, writing `_amended` files.** midoffice07's
   `_corrected` files stay as written and regenerate byte for byte. The gate's report is named
   after its plan, so midoffice07's is untouched.
3. **Coherent, Inc.'s span as ruled, to 2022-07-01, its last price 2022-06-30 stated**
   (`Confirmation.last_price`) and checked exactly. The alternative, ending the span on
   2022-06-30, would have contradicted both the ruling's dates and the endpoint's record.
4. **The search's scope, and how it searches:**
   - by bisection, its premise stated;
   - every probe's identity checked;
   - KVUE and SNDK searched, by the same general rule;
   - FDXF and Vylor not, there being nothing to bracket.
5. **Only the found first record enters the panel**, not the search's other records: the grid
   stays the plan's, and SNDK's bad record stays out.
6. **The transitional first counts carried as ruled** (GEV, AMTM, WBD), flagged, not corrected.
7. **The weighted check in two parts.** Carried counts, the ruling's check, are listed in full.
   Own records, where multi-class companies flag on every record, are listed by security.
8. **A second standing check, count moves**, in the build's report: it catches what the gate's
   split tests cannot.
9. **Member-days the table maps outside a security's spans** -- "after its last price", "inside
   a gap in its prices", 37 -- allowed by the guard, and reported.
10. **Columns keyed by FIGI or CIK, and labelled by their last ticker in the run**: none repeats,
    so the run's reports read as tickers and compare with the old run's.
11. **The run's defaults unchanged where a choice is Jeff's.** `--returns price` stays the
    default. `--volumes` defaults to what the panel stores, traded on securities. The run reads
    the new panel's own membership, checked against the newest snapshot, and its SIC codes, so
    `--industries` needs no key on it.
12. **Two new standing lines in the run:** `current_stale` (05 ruling 3), and the members each
    regression keeps out, by why (06 item 6).
13. **The build reads bars one column at a time,** through a small cache. Holding every ticker's
    is most of a gigabyte on a machine that has run short of memory before; 5.5 GB were free
    this time.
14. **Exact duplicate distributions kept once:** the rule catches TEL's pair only.
15. **The comparison read at full precision by capturing the models in-process** (scratch), the
    reports checked equal to the standalone runs'.
16. **New Python files in CRLF**, like the midoffice modules beside them.

## Decisions left for Jeff

1. **Transitional first counts.** GEV is carried at 4.0x its own count for 59 member-days, AMTM
   at 0.63x for 58 of its 59, WBD at 0.92x for 62.
   - **(a)** Leave as ruled, as built.
   - **(b)** A first count stands only if the span's next sample confirms it within 10%. Otherwise
     the search goes on to the first record the next sample confirms, and the days before it have
     no count. That costs a few requests, all cached. GEV's first confirmed record falls between
     2024-04-05 and 2024-04-11, AMTM's between 10-03 and 10-09; WBD's 0.92 is within 10% and
     stands.
   - **(c)** Take every record the search fetched as a sample. It costs no requests, but it brings
     in SNDK's bad record.
2. **Single samples Polygon got wrong for a quarter:** 504 member-days -- DGX 0.104, SBUX 0.123,
   XYZ 0.109, NWS 0.849, EG 1.112, CEG's and PCG's flips.
   - **Leave them.**
   - **Or rule that a sample moving beyond 10%, which the next sample reverses, is stale and not
     carried** -- the lag rule's cousin. The old panel carried the right count on DGX's, SBUX's,
     NWS's and EG's 252.
   - **PCG's alternation** between two counts over years wants its own reading.
3. **HON and SKYD.**
   - HON from 2026-10-12 at 316,940,010 -- within 0.06% of Honeywell Aerospace's count, with no
     split event and no price gap -- over 17 trading days.
   - SKYD from 2026-10-06 at its pre-merger count, 4.6 times under its weighted count, over 21
     member-days.

   Which counts to carry?
4. **The run's returns by default:** price, as now, or total, Barra's.
5. **The run's panel from now on:** the panel on securities, the old one kept for comparison.
   The state file now gives its command first.
6. **Then, per midoffice07's brief:** residuals for every covered name, and the model-wide
   noise-share question, on the new panel. As of run D:
   - Construction 38.1%;
   - Wholesale 46.2%;
   - Finance & Real Estate 30.9%;
   - Size Nonlinearity 47.0%.

---

## Tests

**The suite:** 2,290 passed, 1 skipped, 19 warnings, in 63.40 s, with
`--ignore=tests/test_market_data.py` (it looks for `../core_repo/src`, as before). The midoffice
tests alone: 839 passed, 1 skipped. Before this piece: 2,279 and 828.

**New: 11.**

`tests/midoffice/test_securities.py`, 9:

- `test_the_first_count_is_found_by_bisection_and_carries_from_its_own_date` -- the search finds
  day 7 between day 0 and day 20 in at most 5 lookups; with no bracket, nothing to search.
- `test_the_search_stops_on_a_record_naming_another_security`
- `test_the_columns_merge_a_reorganisation_under_its_successor_and_keep_a_splice_apart` -- one
  column under S for P and S; the splice's first holder's bars in no column.
- `test_prices_are_stitched_across_a_change_of_ticker_by_the_raw_closes` -- OLD's run scaled 0.5
  to continue NEW's, whose split halves its history; the return across the change the raw
  closes'; traded and adjusted volumes; the tickers per cell.
- `test_total_return_reinvests_each_distribution_on_the_prices_basis` -- a dollar on a
  half-adjusted day is half a dollar; a duplicate kept once; a distribution on a later run's
  first day reinvested; the market's two returns.
- `test_counts_carry_forward_with_the_split_applied_and_the_stale_sample_read_past` -- 100,
  then 200 from the split, a stale 100 read past, 201 resets; the reorganisation's carry until
  the successor's first count; no count before a first.
- `test_the_split_guard_is_exact` -- passes as built; fails if the stale sample resets the count,
  or if the split is not applied.
- `test_the_identity_guards_hold_on_the_built_panel` -- all pass; a member-day past the last trade
  fails unless the table maps it there.
- `test_the_weighted_check_and_the_count_moves_flag_what_the_samples_cannot_show` -- a carried
  count against a weighted one 1.4 times it flagged, an own record within 10% not; a move the
  split explains not flagged, the same move without the split flagged.

`tests/midoffice/test_rebuild.py`, 1:

- `test_a_record_naming_another_composite_figi_under_the_expected_share_class_names_it` --
  COHR's case reconciled, the gate passing; refused where the expected security's records carry
  no share class, where the CIK contradicts, where the record has no composite FIGI.

`tests/midoffice/test_identity.py`, 1:

- `test_a_confirmed_span_may_end_on_a_record_of_a_day_its_ticker_did_not_trade` -- the last price
  stated, the span ends on its record; a price after it, a last price not reached, or none
  stated refuse it.

**Rewritten: 1.** `test_a_record_with_no_identifier_and_the_change_days_stale_one_are_read_past`
gives ODD's first record another share class. Under the share-class rule its old one would now
pass, and the test is of a failure saying its share class is not the expected one.

**Hygiene:**

- Every changed Python file parses, is ASCII, and keeps one line ending: CRLF, counted in bytes
  (`hygiene_final.txt`).
- `docs/FACTOR_MODEL.md` keeps CRLF; the brief is LF, like midoffice07's.
- ruff over every changed Python file finds one thing, `HorizonCovariance` imported and unused in
  `run_sp500_model.py`, which is there at HEAD too (checked) and left.

**The final code, re-run offline into the scratch** (`regressions.txt`,
`regress_gate_panel.txt`):

- **The identity builds:**
  - midoffice05's, 06's, 07's and the amended one write all their CSVs byte for byte -- 16 files.
  - Their JSONs: midoffice07's and the amended one's are identical. midoffice05's lacks the
    `resolutions` key the code adds since midoffice06, and midoffice06's keeps the link-evidence
    wording midoffice07 already reported.
- **The gate:** on a copy of the amended plan, identical but for the plan path.
- **The panel:** its 20 arrays and its report identical.

---

## Failures and debugging trail

- **The first amended build came out identical to the corrected table** (`amended_run1.txt`):
  COHR's confirmation was refused on its prices, Coherent, Inc.'s last price being 2022-06-30.
  `Confirmation.last_price` added.
- **A note read "its prices unbroken between to its last"**; reworded before the build went to
  `data/`.
- **The first panel build exited 1** (`build_run1.txt`). The member-day guard failed on 37 days
  past a member's last trade, which the identity table maps on purpose; then on 4 more "inside a
  gap in its prices" (`build_run2.txt`). The guard now allows the table's own mappings and
  reports them.
- **The `Bars` alias sat above the import of `Bar`**: a NameError at collection. Moved.
- **A bash heredoc with apostrophes failed**, as in midoffice06 and midoffice07. The patch went
  into a file (`patch_build1.py`).
- **The cross-check against the old panel ran past its 10-minute timeout.** An npz member is
  decompressed afresh on each access, and the loop read members cell by cell. Stopped, and
  rewritten to read each array once: 1.2 s.
- **A test fixture's count moved 50 to 55**, exactly 10%, and rounding put it past the
  tolerance. The fixture's count is now 54.
- **`Path.write_text` without `newline=""` turned a new test file's line endings into CRLF.**
  All three new Python files were then put in CRLF, matching.
- **ruff:** two import orders, two f-strings without placeholders and a long docstring line,
  fixed. The `HorizonCovariance` finding predates this piece.
- **The build's carry lines called SMCI's and SBNY's carries "its own change of ticker"**, which
  they are not: same ticker, a later span. Now "from its own earlier span". The arrays and the
  JSON are unaffected (checked).
- **The run reports R-squared to three places.** The comparison captured the models in-process
  for full precision.
- **The old panel's run made 35 requests.** `--industries` asked the reference endpoint for each
  delisted ticker's record on the panel's last day, 2026-11-03 -- ABMD, ANSS, ATVI, XLNX, TWTR
  and 30 more. All 35 answered 404 and are now cached, so the run is offline from here. The state
  file had said it reads the cache only, and that is now true. Why the answers were not on disk
  before is not known.

## Problems found, not fixed

1. **Transitional first counts:** GEV 4.0x for 59 member-days, AMTM 0.63x for 58, WBD 0.92x
   for 62. A decision above.
2. **Single samples Polygon got wrong for a quarter:** 504 member-days -- DGX, SBUX, XYZ, NWS,
   EG, CEG, PCG -- and PCG's alternation over years.
3. **HON's halving from 2026-10-12, and SKYD's post-merger count**, which its records do not
   carry.
4. **FedEx Freight and Vylor carry no share-class count anywhere in the window**, so they have
   no capitalisation and enter no regression: 109 and 24 member-days.
5. **Momentum is missing for new listings in their first half-year** -- 1,058 member-periods --
   by RSTR's coverage rule, as designed.
6. **The fallback list grows to 11:** AMTM, FBIN, GAP, MBC and SOLS join it.
7. **Polygon adjusts a ticker's history for its later holders' events** (PARA's 200). The
   stitching's scale absorbs it, but anyone reading adjusted closes as levels would be misled.
8. **`scripts/run_sp500_model.py` imports `HorizonCovariance` unused** (ruff F401, before this
   piece).
9. **`tests/test_market_data.py` looks for `../core_repo/src`**, not on this machine.
10. **LEN's 2025-01-21 adjustment** is left as Polygon has it, as ruled (06 ruling 6).

## Scratch

`claude/scratch/midoffice08/` (gitignored). The scripts:

- `hygiene.py`, `first_counts_plan.py`, `patch_build1.py`, `crosscheck_old.py`;
- `capture.py` and its `capture/` directory -- the four reports and the pickled models;
- `compare.py`, `extras.py`, `investigations.py`, `regress_compare.py`, `regress_gate_panel.py`.

Their outputs:

- the identity builds: `amended_run1.txt`, `amended_run2.txt`, `amended_data.txt`,
  `regress_corrected.txt`;
- the gate and the panel builds: `gate_amended.txt`, `build_run1.txt` to `build_run3.txt`,
  `build_data.txt`;
- the four model runs: `run_A_old.txt`, `run_B_new_price_adjusted.txt`,
  `run_C_new_price_traded.txt`, `run_D_new_total_traded.txt`;
- the analysis: `crosscheck_old.txt`, `compare.txt`, `extras.txt`, `investigations.txt`;
- the checks: `regressions.txt`, `regress_gate_panel.txt`, `pytest_all.txt`, `pytest_mid.txt`,
  `ruff_final.txt`, `hygiene_final.txt`;
- the output directories `amended/`, `build1/` to `build3/`, `regress/`.

Their outputs are copied into the appendix. The scripts that changed repo files were
`patch_build1.py` and inline Python, each asserting its matches and writing nothing on a miss.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` at the end was checked with
`git add --dry-run`. It names every changed and new path of this piece, the state file
included, and nothing else.

## Open items for Chat

- **The four count questions:** transitional first counts; single bad samples; HON and SKYD;
  PCG's two counts. They decide how far Polygon's share-class counts can be taken as they come.
- **The returns default, and the new panel as the run's**, before the next piece.
- **The noise-share question:** it now has a fuller universe to be answered on --
  - Construction at 38%, with 8 members;
  - Wholesale at 46%, with 12;
  - Finance & Real Estate at 31%, with 12, after SIC62 took its brokers;
  - Size Nonlinearity at 47%.

---

## Appendix: outputs, verbatim

Every output the body rests on, from `claude/scratch/midoffice08/`, in full unless said
otherwise.

### amended_run1.txt -- the first amended build: Coherent, Inc.'s confirmation refused on its prices (its lines alone)

```text
  refused        COHR before II-VI took it: Coherent, Inc., which II-VI bought on 2022-07-01: COHR's prices do not run unbroken from 2022-01-03 to 2022-07-01
```

### amended_data.txt -- the amended table: build_sp500_identity.py --resolve --correct --amend --offline, into data/

```text
Panel
  data\external\polygon\sp500_daily_2022-01-03_2026-11-04.npz: 506 tickers over 1214 trading days, 2022-01-03 .. 2026-11-03; 583,628 priced (ticker, date) cells

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC
  the constituent list predates the change of 2026-10-30 -- OKTA added, KVUE removed -- and is read with it applied
  592 tickers are members on some date of the window; 86 of them are not among the panel's 506; the change table also names, inside the window, SATS that membership() never returns

Sources
  reference records cached by the panel fetch: 2,885 over 506 of the 506 tickers, at 6 sample dates -- no requests; read on its grid of every 252 trading days alone
  ticker events, by ticker: 593 asked -- the window's member tickers and 1 more -- 555 found, 38 with none (delisted); 0 asked over the network this run, the rest read from the cache
  ticker events, by FIGI, for the cached records' securities no ticker found: 41 asked, 4 found; 0 over the network this run

The member-days neither source names, put to the reference endpoint: 4,813 named before, 0 after; 128 lookups, 0 over the network this run
  CDAY for DAY: 2024-02-01..2026-02-06 (506); 506 now accounted for, in 23 lookups
      confirmed: the endpoint names BBG005D7PF34, CDAY's security in the table
      takes DAY 2024-02-01..2026-02-04: Dayforce, Inc. [BBG005D7PF34]
  DISCA for WBD: 2022-04-11..2026-10-05 (1125); 1125 now accounted for, in 15 lookups
      not confirmed: the endpoint names BBG011386VF4, and DISCA's security in the table is BBG00HW3TQG4
      takes WBD 2022-04-11..2026-10-06: Warner Bros. Discovery, Inc. Series A Common Stock [BBG011386VF4]
  WRK for SW: 2022-01-03..2024-07-05 (629); 629 now accounted for, in 12 lookups
      confirmed: WRK names WestRock Company from 2022-01-03 to 2024-07-05
      takes WRK 2022-01-03..2024-07-05: WestRock Company [BBG008NXC572]
      link: BBG008NXC572 before CIK0002005951, to 2024-07-05 -- WestRock Company traded as WRK to 2024-07-05, within a halt of Smurfit Westrock plc's first day under SW, 2024-07-08
  FLT for CPAY: 2022-01-03..2024-03-22 (558); 558 now accounted for, in 12 lookups
      confirmed: the endpoint names BBG000GPXKX9, CPAY's security in the table
      takes FLT 2022-01-03..2024-03-22: FleetCor Technologies, Inc. [BBG000GPXKX9]
  PARA for SKYD (and VIAC, PARA's ticker before): 2022-01-03..2025-08-06 (901); 901 now accounted for, in 34 lookups
      confirmed: VIAC names ViacomCBS Inc. Class B Common Stock from 2022-01-03 to 2022-02-16; PARA names Paramount Global Class B Common Stock from 2022-02-17 to 2025-08-07
      takes VIAC 2022-01-03..2022-02-16: ViacomCBS Inc. Class B Common Stock [BBG000C496P7]
      takes PARA 2022-02-17..2025-08-06: Paramount Global Class B Common Stock [BBG000C496P7]
      link: BBG000C496P7 before BBG01VS5NK99, to 2025-08-06 -- Paramount Global Class B Common Stock traded as PARA to 2025-08-06, within a halt of Skydance Corporation's first day under PSKY, 2025-08-07
  Bunge Limited for BG: 2023-03-15..2023-10-31 (160); 160 now accounted for, in 13 lookups
      confirmed: the endpoint names Bunge Limited
      takes BG 2022-01-03..2023-10-31: Bunge Limited [CIK0001144519]
      link: CIK0001144519 before CIK0001996862, to 2023-10-31 -- Bunge Limited traded as BG to 2023-10-31, within a halt of Bunge Global SA's first day under BG, 2023-11-01
  SMCI itself: 2024-03-18..2024-09-30 (136); 136 now accounted for, in 4 lookups
      confirmed: the endpoint names BBG000MYZDJ3, SMCI's security in the table
      takes SMCI 2022-01-03..2024-09-30: Super Micro Computer, Inc. Common Stock [BBG000MYZDJ3]
  KVUE itself (not among the brief's candidates): 2023-08-25..2026-10-29 (798); 798 now accounted for, in 15 lookups
      confirmed: KVUE names Kenvue Inc. from 2023-05-03 to 2026-10-30
      takes KVUE 2023-05-03..2026-10-30: Kenvue Inc. [BBG01C79X561]

The rebuild's evidence, taken to the table: 20 corrections; 0 reference lookups and 0 price request(s) over the network this run
  reorganisation Bunge Limited to Bunge Global SA, the redomicile, dated by the CIK: BG names CIK0001144519 to 2023-11-07 and CIK0001996862 from 2023-11-08 (the reference endpoint)
      was   BG 2023-11-01..2026-11-03: Bunge Global SA [CIK0001996862]
      was   BG 2022-01-03..2023-10-31: Bunge Limited [CIK0001144519]
      now   BG 2022-01-03..2023-11-07: Bunge Limited [CIK0001144519]
      now   BG 2023-11-08..2026-11-03: Bunge Global SA [CIK0001996862]
  reorganisation Vylor's two FIGIs under one CIK, the prices unbroken: VYLR names BBG024W5VJC3 to 2026-10-02 and BBG024CQR6X9 from 2026-10-05 (the reference endpoint)
      was   VYLR 2026-10-01..2026-11-03: Vylor Inc. [BBG024CQR6X9]
      now   VYLR 2026-10-01..2026-10-02: Vylor Inc. [BBG024W5VJC3]
      now   VYLR 2026-10-05..2026-11-03: Vylor Inc. [BBG024CQR6X9]
  confirmed      II-VI's history under IIVI, before it took COHR as Coherent Corp.: reference: IIVI names Ii-Vi Inc (BBG000BLW102) on 2022-01-03 and 2022-09-07, its prices unbroken between [2 lookups]
      now   IIVI 2022-01-03..2022-09-07: Ii-Vi Inc [BBG000BLW102]
  confirmed      COHR before II-VI took it: Coherent, Inc., which II-VI bought on 2022-07-01: reference: COHR names Coherent Inc (BBG000BG1DH3) on 2022-01-03 and 2022-07-01, its prices unbroken from the first to its last price, on 2022-06-30 [2 lookups]
      now   COHR 2022-01-03..2022-07-01: Coherent Inc [BBG000BG1DH3]
  first price    Reddit, Inc. under RDDT: from 2024-03-20 to 2024-03-21, its first price there
      was   RDDT 2024-03-20..2026-11-03: Reddit, Inc. [BBG005K1D4S0]
      now   RDDT 2024-03-21..2026-11-03: Reddit, Inc. [BBG005K1D4S0]
  first price    GE Vernova Inc. under GEV: from 2024-03-27 to 2024-04-02, its first price there
      was   GEV 2024-03-27..2026-11-03: GE Vernova Inc. [BBG013G17W31]
      now   GEV 2024-04-02..2026-11-03: GE Vernova Inc. [BBG013G17W31]
  first price    Solventum Corporation under SOLV: from 2024-03-26 to 2024-04-01, its first price there
      was   SOLV 2024-03-26..2026-11-03: Solventum Corporation [BBG018YZH6T3]
      now   SOLV 2024-04-01..2026-11-03: Solventum Corporation [BBG018YZH6T3]
  first price    Veralto Corporation under VLTO: from 2023-09-27 to 2023-10-02, its first price there
      was   VLTO 2023-09-27..2026-11-03: Veralto Corporation [BBG019Q32XJ9]
      now   VLTO 2023-10-02..2026-11-03: Veralto Corporation [BBG019Q32XJ9]
  first price    MasterBrand, Inc. under MBC: from 2022-12-09 to 2022-12-15, its first price there
      was   MBC 2022-12-09..2026-11-03: MasterBrand, Inc. [BBG01B9G42Y3]
      now   MBC 2022-12-15..2026-11-03: MasterBrand, Inc. [BBG01B9G42Y3]
  first price    GE HealthCare Technologies Inc. Common Stock under GEHC: from 2022-12-15 to 2023-01-04, its first price there
      was   GEHC 2022-12-15..2026-11-03: GE HealthCare Technologies Inc. Common Stock [BBG01BFR8YV1]
      now   GEHC 2023-01-04..2026-11-03: GE HealthCare Technologies Inc. Common Stock [BBG01BFR8YV1]
  first price    Kenvue Inc. under KVUE: from 2023-05-03 to 2023-05-04, its first price there
      was   KVUE 2023-05-03..2026-10-30: Kenvue Inc. [BBG01C79X561]
      now   KVUE 2023-05-04..2026-10-30: Kenvue Inc. [BBG01C79X561]
  first price    Amentum Holdings, Inc. under AMTM: from 2024-09-24 to 2024-09-30, its first price there
      was   AMTM 2024-09-24..2026-11-03: Amentum Holdings, Inc. [BBG01NY7B734]
      now   AMTM 2024-09-30..2026-11-03: Amentum Holdings, Inc. [BBG01NY7B734]
  first price    Sandisk Corporation Common Stock under SNDK: from 2025-02-13 to 2025-02-24, its first price there
      was   SNDK 2025-02-13..2026-11-03: Sandisk Corporation Common Stock [BBG01R388JG1]
      now   SNDK 2025-02-24..2026-11-03: Sandisk Corporation Common Stock [BBG01R388JG1]
  first price    Qnity Electronics, Inc. under Q: from 2025-10-27 to 2025-11-03, its first price there
      was   Q 2025-10-27..2026-11-03: Qnity Electronics, Inc. [BBG01TRL73S5]
      now   Q 2025-11-03..2026-11-03: Qnity Electronics, Inc. [BBG01TRL73S5]
  first price    Solstice Advanced Materials Inc. Common Stock under SOLS: from 2025-10-20 to 2025-10-30, its first price there
      was   SOLS 2025-10-20..2026-11-03: Solstice Advanced Materials Inc. Common Stock [BBG01X5744P5]
      now   SOLS 2025-10-30..2026-11-03: Solstice Advanced Materials Inc. Common Stock [BBG01X5744P5]
  first price    FedEx Freight Holding Company, Inc. under FDXF: from 2026-05-27 to 2026-06-01, its first price there
      was   FDXF 2026-05-27..2026-11-03: FedEx Freight Holding Company, Inc. [BBG01ZPB91C4]
      now   FDXF 2026-06-01..2026-11-03: FedEx Freight Holding Company, Inc. [BBG01ZPB91C4]
  first price    Honeywell Aerospace Inc. Common Stock under HONA: from 2026-06-15 to 2026-06-29, its first price there
      was   HONA 2026-06-15..2026-11-03: Honeywell Aerospace Inc. Common Stock [BBG020QYV152]
      now   HONA 2026-06-29..2026-11-03: Honeywell Aerospace Inc. Common Stock [BBG020QYV152]
  unpriced       First Republic Bank under S600 2023-11-20..2026-11-03: no price under the ticker inside the span, which began at began trading as S600 on 2023-11-18 (events)
      was   S600 2023-11-20..2026-11-03: First Republic Bank [BBG0019LSQ49]
  unpriced       Constellation Energy Corporation Common Stock under CEGV 2022-01-19..2022-02-01: no price under the ticker inside the span, which began at the first of its events, 2022-01-19: the start of Polygon's record of it, not necessarily of the ticker
      was   CEGV 2022-01-19..2022-02-01: Constellation Energy Corporation Common Stock [BBG014KFRNP7]
  refuted        CRH Public Limited Company under CRHC 2022-01-03..2023-09-22, from the window's first day (as CRHC from 2021-09-23): 4 record(s) name Cohn Robbins Holdings Corp. (CIK0001818212), 2022-01-03..2022-10-04; 4 answer(s) no record; none names it
      was   CRHC 2022-01-03..2023-09-22: CRH Public Limited Company [CIK0000849395]

The table
  634 rows: 605 securities, each under one ticker over one span of the window; 518 rows priced by the panel, 103 from events alone, 1 presumed
  identified by the records and the events 476, by the records alone 40, by the events alone 80, by the reference endpoint alone 7

Guards
  ok  every priced (ticker, date) maps to exactly one security: 583,628 checked, 0 failure(s)
  ok  no security under two tickers on one date: 634 checked, 0 failure(s)
  ok  no ticker names two securities on one date: 634 checked, 0 failure(s)
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
  the members' ticker changes in the table: 64; read through: 51; not: 13
    not A -> AWD (2005-11-23): AWD is not in today's list
    not MPWR -> MPWRE (2005-11-28): MPWRE is not in today's list
    not UARM -> UA (2006-12-18): UAA is not in today's list
    not LKQX -> LKQ (2012-05-23): LKQ is not in today's list
    not NWSA -> NWSAV (2013-06-20): NWSAV is not in today's list
    not UA -> UAA (2016-12-07): UAA is not in today's list
    not WLTW -> WTW (2022-01-10): the changes table records it as a swap on 2022-01-10
    not VIAC -> PARA (2022-02-17): PARA is not in today's list
    not FBHS -> FBIN (2022-12-15): FBIN is not in today's list
    not FRC -> FRCB (2023-05-03): S600 is not in today's list
    not FRCB -> S600 (2023-11-18): S600 is not in today's list
    not CDAY -> DAY (2024-02-01): DAY is not in today's list
    not GPS -> GAP (2024-08-22): GAP is not in today's list
  ECHO    1057 day(s): 2022-01-03..2026-03-20 (1057) -- BBG000TGLV00, trading as SATS
  the change table's inconsistencies, read through the renames: 27 -> 24
    resolved: FB added absent, 2013-12-23
    resolved: RE added absent, 2017-06-19
    resolved: SATS added absent, 2026-03-23

Membership read through the renames, for the plan and the panel: 610,842 member-days mapped to securities, 0 named; against membership() as given, 611,899 and 0
  ok  every member on every date maps to a security or is named, membership read through the renames: 610,842 checked, 0 failure(s)
  EchoStar Corporation [BBG000TGLV00]: 1,057 member-day(s) fewer, read through the renames

Truncations: member dates on which the member traded under another ticker than the panel's, which has no price for it (phantom dates left out)
  BALL   as BLL       88 day(s): 2022-01-03..2022-05-09 (88)
  BNY    as BK      1099 day(s): 2022-01-03..2026-05-20 (1099)
  COR    as ABC      416 day(s): 2022-01-03..2023-08-29 (416)
  CPAY   as FLT      558 day(s): 2022-01-03..2024-03-22 (558)
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

Member-days no security accounts for: 0 -- named, each with why

The 592 member tickers: 593 runs of membership; the security each names identified by records+events 475, events 80, records 31, records+reference 2, events+reference 2, reference 2, records+events+reference 1

The fetch the rebuild needs, for the 600 securities that are members on some date of the window
  628 spans under 617 tickers; 111 of the tickers are not the panel's: ABC ABNB ACGL AMTM ANTM APO APP ARES AXON BE BG BK BLDR BLL BX CASY CEG CHK CIEN COHR COIN CPT CRH CRWD CSGP CVNA DASH DAY DDOG DECK DELL EME EQR EQT ERIE EXE FB FBIN FDXF FERG FI FICO FIX FLEX FLT FRCB FSLR GAP GDDY GEHC GEV HONA HOOD HUBB IBKR IIVI INVH JBL KDP KKR KVUE LII LITE LULU MBC MMC MOH MRVL NDSN NLOK OKTA ON P PANW PARA PCG PEAK PKI PLTR PODD PSKY PSTG Q RDDT RE SATS SMCI SNDK SOLS SOLV SQ STLD TKO TPL TRGP TTD TWLO UBER VEEV VIAC VICI VLTO VRT VST VYLR WBD WDAY WRK WSM WTW XYZ
  prices: 2 requests a ticker, adjusted and not, over the window -- 222 new, the panel's 506 cached (cut to each security's spans)
  share counts: 11,950 samples inside the securities' own spans (each span's first and last day and every 63 trading days), 9,087 not on disk
  split events, for the share-count rule recorded in midoffice03: one request a ticker, 617
  in all 9,926 requests
  every member-day is in it: none is named

Wrote
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended_members.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended_fetch.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended_links.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended_membership.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended.json
```

### gate_amended.txt -- the gate, offline, on the amended plan (its 65 not-regular distributions and its 55 split lines left out: they are midoffice07's gate_run1.txt's, unchanged, as regress_gate_panel.txt checks)

```text
Plan
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended_fetch.csv: 628 spans, 600 securities, 617 tickers, 11,950 share samples

Fetching
  at the start: this process 37 MB (peak 37, 508 committed); machine 2,899 of 26,420 MB available
     500/14,422  AKAM         1s  this process 43 MB (peak 43, 514 committed); machine 2,984 of 26,420 MB available
   1,000/14,422  SNPS         2s  this process 46 MB (peak 46, 517 committed); machine 2,950 of 26,420 MB available
   1,500/14,422  BIO          4s  this process 49 MB (peak 49, 520 committed); machine 2,992 of 26,420 MB available
   2,000/14,422  LW           5s  this process 51 MB (peak 51, 522 committed); machine 3,092 of 26,420 MB available
   2,500/14,422  AME          6s  this process 52 MB (peak 52, 523 committed); machine 3,184 of 26,420 MB available
   3,000/14,422  NVDA         6s  this process 52 MB (peak 52, 523 committed); machine 3,184 of 26,420 MB available
   3,500/14,422  VRSK         6s  this process 52 MB (peak 52, 523 committed); machine 3,191 of 26,420 MB available
   4,000/14,422  CPB          6s  this process 54 MB (peak 54, 524 committed); machine 3,189 of 26,420 MB available
   4,500/14,422  ECL          7s  this process 54 MB (peak 54, 525 committed); machine 3,201 of 26,420 MB available
   5,000/14,422  FIS          7s  this process 54 MB (peak 54, 525 committed); machine 3,221 of 26,420 MB available
   5,500/14,422  HUBB         7s  this process 54 MB (peak 54, 525 committed); machine 3,207 of 26,420 MB available
   6,000/14,422  KO           7s  this process 55 MB (peak 55, 525 committed); machine 3,200 of 26,420 MB available
   6,500/14,422  MRK          7s  this process 55 MB (peak 55, 526 committed); machine 3,199 of 26,420 MB available
   7,000/14,422  PG           8s  this process 55 MB (peak 55, 526 committed); machine 3,197 of 26,420 MB available
   7,500/14,422  SCHW         8s  this process 56 MB (peak 56, 526 committed); machine 3,188 of 26,420 MB available
   8,000/14,422  VNO          8s  this process 56 MB (peak 56, 527 committed); machine 3,198 of 26,420 MB available
   8,500/14,422  AIZ          8s  this process 56 MB (peak 56, 528 committed); machine 3,193 of 26,420 MB available
   9,000/14,422  UDR          8s  this process 57 MB (peak 57, 528 committed); machine 3,182 of 26,420 MB available
   9,500/14,422  SBAC         9s  this process 57 MB (peak 57, 528 committed); machine 3,175 of 26,420 MB available
  10,000/14,422  MA           9s  this process 57 MB (peak 57, 529 committed); machine 3,165 of 26,420 MB available
  10,500/14,422  HON          9s  this process 58 MB (peak 58, 529 committed); machine 3,143 of 26,420 MB available
  11,000/14,422  EQIX         9s  this process 58 MB (peak 58, 529 committed); machine 3,112 of 26,420 MB available
  11,500/14,422  DFS          9s  this process 58 MB (peak 58, 530 committed); machine 3,138 of 26,420 MB available
  12,000/14,422  HII         10s  this process 59 MB (peak 59, 530 committed); machine 3,134 of 26,420 MB available
  12,500/14,422  RDDT        10s  this process 59 MB (peak 59, 530 committed); machine 3,123 of 26,420 MB available
  13,000/14,422  DOW         10s  this process 59 MB (peak 59, 531 committed); machine 3,119 of 26,420 MB available
  13,500/14,422  EXE         10s  this process 61 MB (peak 61, 532 committed); machine 3,115 of 26,420 MB available
  14,000/14,422  ACGL        10s  this process 61 MB (peak 61, 532 committed); machine 3,121 of 26,420 MB available
  14,422/14,422  SW          10s  this process 61 MB (peak 61, 532 committed); machine 3,146 of 26,420 MB available
  0 requests over the network, in 10s; 0 failed

Identity: each span's first share sample against the security the table expects
  628 of 628 spans pass, 1 of them by the share-class FIGI, the record naming another composite FIGI
  share class  COHR 2022-09-08..2026-11-03 (157 member-days): COHR on 2022-09-08 names Coherent Corp. Common Stock (BBG000GS6QB4), reconciled to Coherent Corp. (BBG000BLW102) by its share-class FIGI, BBG001S5S6N6, which the expected security's own records carry

Splits: each event classified by its ratio -- whole numbers, no term over 100 in lowest terms, or not -- by the security's share counts either side, within 10%, a sample still at the pre-split count stale until one follows, and by its raw price on the execution date, within the same of the ratio's reciprocal
  55 events inside the securities' spans: 47 splits, applied to the carried share count; 8 price-only, not applied; 0 contradictions, 0 of them exempt (DD); 2 untested, by the ratio alone
Gate: passes
  wrote data\external\polygon\sp500_rebuild_gate_2022-01-03_2026-11-03_amended.json
```

### first_counts_plan.txt -- the spans a first-count search would reach, from the plan and the cache, before any request

```text
CEG    BBG014KFRNP7   2022-02-02..2026-11-03 md  1193 first sample 2022-02-02         None; first count (datetime.date(2022, 4, 20), 326660000); samples [('2022-02-02', None), ('2022-04-20', 326660000), ('2022-07-21', 326700000), ('2022-10-19', 326850000)]
WBD    BBG011386VF4   2022-04-11..2026-10-06 md  1125 first sample 2022-04-11         None; first count (datetime.date(2022, 7, 13), 2426840000); samples [('2022-04-11', None), ('2022-07-13', 2426840000), ('2022-10-11', 2427590000), ('2023-01-11', 2428400000)]
GEHC   BBG01BFR8YV1   2023-01-04..2026-11-03 md   962 first sample 2023-01-04         None; first count (datetime.date(2023, 3, 20), 453930000); samples [('2023-01-04', None), ('2023-03-20', 453930000), ('2023-06-20', 454680000), ('2023-09-19', 454840000)]
KVUE   BBG01C79X561   2023-05-04..2026-10-30 md   798 first sample 2023-05-04         None; first count (datetime.date(2023, 8, 3), 1914890000); samples [('2023-05-04', None), ('2023-08-03', 1914890000), ('2023-11-01', 1914890000), ('2024-02-02', 1915000000)]
VLTO   BBG019Q32XJ9   2023-10-02..2026-11-03 md   776 first sample 2023-10-02         None; first count (datetime.date(2023, 12, 27), 246310000); samples [('2023-10-02', None), ('2023-12-27', 246310000), ('2024-03-28', 246540000), ('2024-06-28', 246850000)]
SOLV   BBG018YZH6T3   2024-04-01..2026-11-03 md   652 first sample 2024-04-01         None; first count (datetime.date(2024, 6, 26), 172710000); samples [('2024-04-01', None), ('2024-06-26', 172710000), ('2024-09-25', 172710000), ('2024-12-24', 172760000)]
GEV    BBG013G17W31   2024-04-02..2026-11-03 md   651 first sample 2024-04-02         None; first count (datetime.date(2024, 6, 27), 274090000); samples [('2024-04-02', None), ('2024-06-27', 274090000), ('2024-09-26', 274800000), ('2024-12-26', 275650000)]
AMTM   BBG01NY7B734   2024-09-30..2026-11-03 md    59 first sample 2024-09-30         None; first count (datetime.date(2024, 12, 23), 243300000); samples [('2024-09-30', None), ('2024-12-23', 243300000), ('2025-03-27', 243300000), ('2025-06-27', 243322468)]
SNDK   BBG01R388JG1   2025-02-24..2026-11-03 md   234 first sample 2025-02-24         None; first count (datetime.date(2025, 5, 15), 145300000); samples [('2025-02-24', None), ('2025-05-15', 145300000), ('2025-08-15', 145299942), ('2025-11-13', 145805548)]
PSKY   BBG01VS5NK99   2025-08-07..2026-10-05 md   292 first sample 2025-08-07         None; first count (datetime.date(2025, 11, 5), 1064544631); samples [('2025-08-07', None), ('2025-11-05', 1064544631), ('2026-02-06', 1071666977), ('2026-05-08', 1080241022)]
SOLS   BBG01X5744P5   2025-10-30..2026-11-03 md    36 first sample 2025-10-30         None; first count (datetime.date(2026, 1, 21), 158727456); samples [('2025-10-30', None), ('2026-01-21', 158727456), ('2026-04-22', 158795531), ('2026-07-23', 158804273)]
Q      BBG01TRL73S5   2025-11-03..2026-11-03 md   252 first sample 2025-11-03         None; first count (datetime.date(2026, 1, 28), 209463194); samples [('2025-11-03', None), ('2026-01-28', 209463194), ('2026-04-29', 209582425), ('2026-07-30', 209339207)]
FDXF   BBG01ZPB91C4   2026-06-01..2026-11-03 md   109 first sample 2026-06-01         None; first count None; samples [('2026-06-01', None), ('2026-08-26', None), ('2026-11-03', None)]
VYLR   BBG024W5VJC3   2026-10-01..2026-10-02 md     2 first sample 2026-10-01         None; first count None; samples [('2026-10-01', None), ('2026-10-02', None)]
VYLR   BBG024CQR6X9   2026-10-05..2026-11-03 md    22 first sample 2026-10-05         None; first count None; samples [('2026-10-05', None), ('2026-11-03', None)]
spans to search: 15
```

### investigations.txt -- every record the search asked for; COHR's bars and records; VIAC's and PARA's factors; HON's, HONA's and SKYD's records; HON's prices

```text
The first-count search: every record asked for, with the bracket's two ends
  AMTM bracket 2024-09-30..2024-12-23, found 2024-10-01 153,280,000
    2024-09-30 class           None weighted           None figi BBG01NY7B734 cik 0002011286
    2024-10-01 class      153280000 weighted      243293457 figi BBG01NY7B734 cik 0002011286
    2024-10-03 class      153280000 weighted      153280369 figi BBG01NY7B734 cik 0002011286
    2024-10-09 class      243300000 weighted      243302173 figi BBG01NY7B734 cik 0002011286
    2024-10-18 class      243300000 weighted      243302173 figi BBG01NY7B734 cik 0002011286
    2024-11-08 class      243300000 weighted      243302173 figi BBG01NY7B734 cik 0002011286
    2024-12-23 class      243300000 weighted      243302257 figi BBG01NY7B734 cik 0002011286
  CEG bracket 2022-02-02..2022-04-20, found 2022-02-03 326,500,000
    2022-02-02 class           None weighted           None figi BBG014KFRNP7 cik 0001868275
    2022-02-03 class      326500000 weighted      326000000 figi BBG014KFRNP7 cik 0001868275
    2022-02-07 class      326500000 weighted      326500000 figi BBG014KFRNP7 cik 0001868275
    2022-02-10 class      326500000 weighted      326500000 figi BBG014KFRNP7 cik 0001868275
    2022-02-22 class      326500000 weighted      326500000 figi BBG014KFRNP7 cik 0001868275
    2022-03-11 class      326660000 weighted      326663937 figi BBG014KFRNP7 cik 0001868275
    2022-04-20 class      326660000 weighted      326663937 figi BBG014KFRNP7 cik 0001868275
  FDXF  FDXF: no sample in 2026-06-01..2026-11-03 carries a share-class count, so there is nothing to search between
  GEHC bracket 2023-01-04..2023-03-20, found 2023-01-05 454,710,000
    2023-01-04 class           None weighted           None figi BBG01BFR8YV1 cik None
    2023-01-05 class      454710000 weighted           None figi BBG01BFR8YV1 cik None
    2023-01-09 class      454710000 weighted           None figi BBG01BFR8YV1 cik 0001932393
    2023-01-12 class      454710000 weighted      454710004 figi BBG01BFR8YV1 cik 0001932393
    2023-01-23 class      454710000 weighted      454710004 figi BBG01BFR8YV1 cik 0001932393
    2023-02-09 class      454710000 weighted      454710004 figi BBG01BFR8YV1 cik 0001932393
    2023-03-20 class      453930000 weighted      453926139 figi BBG01BFR8YV1 cik 0001932393
  GEV bracket 2024-04-02..2024-06-27, found 2024-04-03 1,088,330,000
    2024-04-02 class           None weighted           None figi BBG013G17W31 cik 0001996810
    2024-04-03 class     1088330000 weighted      272083576 figi BBG013G17W31 cik 0001996810
    2024-04-05 class     1088330000 weighted      272083576 figi BBG013G17W31 cik 0001996810
    2024-04-11 class      272080000 weighted      272083576 figi BBG013G17W31 cik 0001996810
    2024-04-23 class      272080000 weighted      272083576 figi BBG013G17W31 cik 0001996810
    2024-05-14 class      274090000 weighted      274085523 figi BBG013G17W31 cik 0001996810
    2024-06-27 class      274090000 weighted      274085523 figi BBG013G17W31 cik 0001996810
  KVUE bracket 2023-05-04..2023-08-03, found 2023-05-08 1,888,970,000
    2023-05-04 class           None weighted           None figi BBG01C79X561 cik 0001944048
    2023-05-05 class           None weighted     1867364000 figi BBG01C79X561 cik 0001944048
    2023-05-08 class     1888970000 weighted     1867364000 figi BBG01C79X561 cik 0001944048
    2023-05-09 class     1888970000 weighted     1867364000 figi BBG01C79X561 cik 0001944048
    2023-05-15 class     1888970000 weighted     1867364000 figi BBG01C79X561 cik 0001944048
    2023-05-25 class     1888970000 weighted     1888972560 figi BBG01C79X561 cik 0001944048
    2023-06-20 class     1914890000 weighted     1914894444 figi BBG01C79X561 cik 0001944048
    2023-08-03 class     1914890000 weighted     1914894444 figi BBG01C79X561 cik 0001944048
  Q bracket 2025-11-03..2026-01-28, found 2025-11-20 209,463,194
    2025-11-03 class           None weighted           None figi BBG01TRL73S5 cik None
    2025-11-12 class           None weighted      209433618 figi BBG01TRL73S5 cik None
    2025-11-17 class           None weighted      209433618 figi BBG01TRL73S5 cik 0002058873
    2025-11-19 class           None weighted      209463194 figi BBG01TRL73S5 cik 0002058873
    2025-11-20 class      209463194 weighted      209463194 figi BBG01TRL73S5 cik 0002058873
    2025-11-21 class      209463194 weighted      209463194 figi BBG01TRL73S5 cik 0002058873
    2025-12-15 class      209463194 weighted      209463194 figi BBG01TRL73S5 cik 0002058873
    2026-01-28 class      209463194 weighted      209463194 figi BBG01TRL73S5 cik 0002058873
  PSKY bracket 2025-08-07..2025-11-05, found 2025-08-11 1,064,544,631
    2025-08-07 class           None weighted           None figi BBG01VS5NK99 cik 0002041610
    2025-08-08 class           None weighted           None figi BBG01VS5NK99 cik 0002041610
    2025-08-11 class     1064544631 weighted      674224884 figi BBG01VS5NK99 cik 0002041610
    2025-08-12 class     1064544631 weighted      674224884 figi BBG01VS5NK99 cik 0002041610
    2025-08-18 class     1064544631 weighted      674224884 figi BBG01VS5NK99 cik 0002041610
    2025-08-28 class     1064544631 weighted     1096153498 figi BBG01VS5NK99 cik 0002041610
    2025-09-22 class     1064544631 weighted     1096153498 figi BBG01VS5NK99 cik 0002041610
    2025-11-05 class     1064544631 weighted     1096153498 figi BBG01VS5NK99 cik 0002041610
  SNDK bracket 2025-02-24..2025-05-15, found 2025-02-26 144,000,000
    2025-02-24 class           None weighted           None figi BBG01R388JG1 cik 0002023554
    2025-02-25 class           None weighted      144000000 figi BBG01R388JG1 cik 0002023554
    2025-02-26 class      144000000 weighted      144000000 figi BBG01R388JG1 cik 0002023554
    2025-02-27 class      144000000 weighted      144000000 figi BBG01R388JG1 cik 0002023554
    2025-03-05 class      144000000 weighted      144000000 figi BBG01R388JG1 cik 0002023554
    2025-03-14 class      114860000 weighted      114863251 figi BBG01R388JG1 cik 0002023554
    2025-04-03 class      114860000 weighted      144863251 figi BBG01R388JG1 cik 0002023554
    2025-05-15 class      145300000 weighted      145299218 figi BBG01R388JG1 cik 0002023554
  SOLS bracket 2025-10-30..2026-01-21, found 2025-11-18 158,727,456
    2025-10-30 class           None weighted           None figi BBG01X5744P5 cik 0002064953
    2025-11-07 class           None weighted      158721802 figi BBG01X5744P5 cik 0002064953
    2025-11-12 class           None weighted      158721802 figi BBG01X5744P5 cik 0002064953
    2025-11-14 class           None weighted      158721802 figi BBG01X5744P5 cik 0002064953
    2025-11-17 class           None weighted      158727456 figi BBG01X5744P5 cik 0002064953
    2025-11-18 class      158727456 weighted      158727456 figi BBG01X5744P5 cik 0002064953
    2025-12-09 class      158727456 weighted      158727456 figi BBG01X5744P5 cik 0002064953
    2026-01-21 class      158727456 weighted      158727456 figi BBG01X5744P5 cik 0002064953
  SOLV bracket 2024-04-01..2024-06-26, found 2024-04-02 172,710,000
    2024-04-01 class           None weighted           None figi BBG018YZH6T3 cik 0001964738
    2024-04-02 class      172710000 weighted      172709175 figi BBG018YZH6T3 cik 0001964738
    2024-04-04 class      172710000 weighted      172709175 figi BBG018YZH6T3 cik 0001964738
    2024-04-10 class      172710000 weighted      172709175 figi BBG018YZH6T3 cik 0001964738
    2024-04-22 class      172710000 weighted      172709175 figi BBG018YZH6T3 cik 0001964738
    2024-05-13 class      172710000 weighted      172709505 figi BBG018YZH6T3 cik 0001964738
    2024-06-26 class      172710000 weighted      172709505 figi BBG018YZH6T3 cik 0001964738
  VLTO bracket 2023-10-02..2023-12-27, found 2023-10-04 246,060,000
    2023-10-02 class           None weighted           None figi BBG019Q32XJ9 cik 0001967680
    2023-10-03 class           None weighted      246291242 figi BBG019Q32XJ9 cik 0001967680
    2023-10-04 class      246060000 weighted      246291242 figi BBG019Q32XJ9 cik 0001967680
    2023-10-05 class      246060000 weighted      246291242 figi BBG019Q32XJ9 cik 0001967680
    2023-10-11 class      246060000 weighted      246291242 figi BBG019Q32XJ9 cik 0001967680
    2023-10-23 class      246060000 weighted      246291242 figi BBG019Q32XJ9 cik 0001967680
    2023-11-13 class      246310000 weighted      246308057 figi BBG019Q32XJ9 cik 0001967680
    2023-12-27 class      246310000 weighted      246308057 figi BBG019Q32XJ9 cik 0001967680
  VYLR  VYLR: no sample in 2026-10-01..2026-10-02 carries a share-class count, so there is nothing to search between
  VYLR  VYLR: no sample in 2026-10-05..2026-11-03 carries a share-class count, so there is nothing to search between
  WBD bracket 2022-04-11..2022-07-13, found 2022-04-12 2,233,730,000
    2022-04-11 class           None weighted           None figi BBG011386VF4 cik 0001437107
    2022-04-12 class     2233730000 weighted      508221253 figi BBG011386VF4 cik 0001437107
    2022-04-14 class     2233730000 weighted      508221253 figi BBG011386VF4 cik 0001437107
    2022-04-21 class     2233730000 weighted     2426605846 figi BBG011386VF4 cik 0001437107
    2022-05-03 class     2426840000 weighted     2426844405 figi BBG011386VF4 cik 0001437107
    2022-05-25 class     2426840000 weighted     2426844405 figi BBG011386VF4 cik 0001437107
    2022-07-13 class     2426840000 weighted     2426844405 figi BBG011386VF4 cik 0001437107

COHR's bars around Coherent, Inc.'s last day (raw closes)
  2022-06-27 267.86
  2022-06-28 268.22
  2022-06-29 267.07
  2022-06-30 266.22
  2022-09-08 43.26
  2022-09-09 44.23
  ref COHR 2022-01-03: BBG000BG1DH3 share class BBG001S5Q0Q8 cik 0000021510 Coherent Inc
  ref COHR 2022-07-01: BBG000BG1DH3 share class BBG001S5Q0Q8 cik 0000021510 Coherent Inc
  ref COHR 2022-09-08: BBG000GS6QB4 share class BBG001S5S6N6 cik None Coherent Corp. Common Stock
  ref COHR 2022-12-07: BBG000BLW102 share class BBG001S5S6N6 cik 0000820318 Coherent Corp. Common Stock

VIAC and PARA: adjusted over raw close, and PARA's split events
  VIAC: [1.0]
  PARA: [1.0, 10.0, 200.0]
  splits-PARA: [{'execution_date': '2026-10-27', 'id': 'E3c40a11b5e455bdb63361031344bc04c0295468044eeee83e15d6dc3e3c6717c', 'split_from': 10, 'split_to': 1, 'ticker': 'PARA'}, {'execution_date': '2026-05-08', 'id': 'E16511f89639544c5fc21bf219a56773505b38108f9b672c748f90deda20dd186', 'split_from': 20, 'split_to': 1, 'ticker': 'PARA'}]

HON's and SKYD's records in 2026
  ref-HON-2026-01-09 BBG000H556T9 class 634887208 weighted 634887208 Honeywell International, Inc.
  ref-HON-2026-04-13 BBG000H556T9 class 635675701 weighted 635675701 Honeywell International, Inc.
  ref-HON-2026-07-14 BBG000H556T9 class None weighted 316826559 Honeywell International, Inc.
  ref-HON-2026-10-12 BBG000H556T9 class 316940010 weighted 316940010 Honeywell International, Inc.
  ref-HON-2026-11-03 BBG000H556T9 class 317358518 weighted 316940010 Honeywell International, Inc.
  ref-HONA-2026-06-15 None class None weighted None None
  ref-HONA-2026-06-29 BBG020QYV152 class 316826559 weighted None Honeywell Aerospace Inc. Common Stock
  ref-HONA-2026-09-15 BBG020QYV152 class None weighted 316952725 Honeywell Aerospace Inc. Common Stock
  ref-HONA-2026-11-03 BBG020QYV152 class None weighted 317031836 Honeywell Aerospace Inc. Common Stock
  ref-SKYD-2026-10-06 BBG01VS5NK99 class None weighted None Skydance Corporation
  ref-SKYD-2026-11-03 BBG01VS5NK99 class None weighted 5042178087 Skydance Corporation
  HON's split events: [{'execution_date': '2025-10-30', 'id': 'Padbaea00469097d9434039934affb5a3f7694a2774160b65d42aabb913d00804', 'split_from': 1000, 'split_to': 1061, 'ticker': 'HON'}]
  HON's raw closes moving more than 15% in a day: none
    2026-06-24 open 223.82 close 227.42
    2026-06-25 open 227.77 close 231.24
    2026-06-26 open 229.53 close 232.21
    2026-06-29 open 240.645 close 227.8
    2026-06-30 open 231.35 close 223.9
    2026-07-01 open 227.31 close 221.75
    2026-07-02 open 222.28 close 229.86
```

### build_run1.txt -- the first panel build, its guards (exit 1: 37 member-days past a member's last trade)

```text
Guards
  ok  each (ticker, date) price feeds one column: 685,133 checked, 0 failure(s)
  FAILED  every member-day is a column's, on a day inside its spans: 610,842 checked, 37 failure(s)
      XLNX 2022-02-14
      INFO 2022-02-28
      INFO 2022-03-01
      CTXS 2022-09-30
      TWTR 2022-10-28
      TWTR 2022-10-31
      SIVB 2023-03-10
      SBNY 2023-03-13
      SIVB 2023-03-13
      SBNY 2023-03-14
  ok  every bar under a span's ticker inside the span is its column's, and only those: 685,089 checked, 0 failure(s)
  ok  every member-day of the walk through the renames is a column's member-day: 610,842 checked, 0 failure(s)
  ok  the panel's member-days are the walk's, no more: 610,842 checked, 0 failure(s)
  ok  no day on or after a split carries a pre-split count, and the split's day carries the count before it times the ratio (25,260 of the days checked member-days): 25,846 checked, 0 failure(s)

```

### build_run2.txt -- the second, its guards (exit 1: 4 member-days inside a gap in prices)

```text
Guards
  ok  each (ticker, date) price feeds one column: 685,133 checked, 0 failure(s)
  FAILED  every member-day is a column's, on a day inside its spans or one the table maps outside them: 610,842 checked, 4 failure(s)
      SBNY 2023-03-13
      SBNY 2023-03-14
      FRCB 2023-05-01
      FRCB 2023-05-02
  ok  every bar under a span's ticker inside the span is its column's, and only those: 685,089 checked, 0 failure(s)
  ok  every member-day of the walk through the renames is a column's member-day: 610,842 checked, 0 failure(s)
  ok  the panel's member-days are the walk's, no more: 610,842 checked, 0 failure(s)
  ok  no day on or after a split carries a pre-split count, and the split's day carries the count before it times the ratio (25,260 of the days checked member-days): 25,846 checked, 0 failure(s)

```

### regress/panel_final.txt -- the panel build on the final code, offline (its arrays and report identical to data/'s, written by the run before its carry lines were reworded)

```text
Plan
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_amended_fetch.csv: 628 spans, 600 securities, 617 tickers

The gate, offline: passes
  identity: 628 of 628 spans pass, 1 by the share-class FIGI
    COHR: COHR on 2022-09-08 names Coherent Corp. Common Stock (BBG000GS6QB4), reconciled to Coherent Corp. (BBG000BLW102) by its share-class FIGI, BBG001S5S6N6, which the expected security's own records carry
  splits: 55 events, 47 splits, 8 price-only, 0 contradictions, 2 untested

The first-count search: each span that starts without a share count, no count carried into it, searched between its first sample and its first with a count
  AMTM   2024-09-30..2026-11-03 (   59 member-days): AMTM: between 2024-09-30, without a count, and 2024-12-23, with one, the first record with a count is 2024-10-01's, 153,280,000, in 5 lookup(s)
  CEG    2022-02-02..2026-11-03 ( 1193 member-days): CEG: between 2022-02-02, without a count, and 2022-04-20, with one, the first record with a count is 2022-02-03's, 326,500,000, in 5 lookup(s)
  FDXF   2026-06-01..2026-11-03 (  109 member-days): FDXF: no sample in 2026-06-01..2026-11-03 carries a share-class count, so there is nothing to search between
  GEHC   2023-01-04..2026-11-03 (  962 member-days): GEHC: between 2023-01-04, without a count, and 2023-03-20, with one, the first record with a count is 2023-01-05's, 454,710,000, in 5 lookup(s)
  GEV    2024-04-02..2026-11-03 (  651 member-days): GEV: between 2024-04-02, without a count, and 2024-06-27, with one, the first record with a count is 2024-04-03's, 1,088,330,000, in 5 lookup(s)
  KVUE   2023-05-04..2026-10-30 (  798 member-days): KVUE: between 2023-05-04, without a count, and 2023-08-03, with one, the first record with a count is 2023-05-08's, 1,888,970,000, in 6 lookup(s)
  Q      2025-11-03..2026-11-03 (  252 member-days): Q: between 2025-11-03, without a count, and 2026-01-28, with one, the first record with a count is 2025-11-20's, 209,463,194, in 6 lookup(s)
  PSKY   2025-08-07..2026-10-05 (  292 member-days): PSKY: between 2025-08-07, without a count, and 2025-11-05, with one, the first record with a count is 2025-08-11's, 1,064,544,631, in 6 lookup(s)
  SNDK   2025-02-24..2026-11-03 (  234 member-days): SNDK: between 2025-02-24, without a count, and 2025-05-15, with one, the first record with a count is 2025-02-26's, 144,000,000, in 6 lookup(s)
  SOLS   2025-10-30..2026-11-03 (   36 member-days): SOLS: between 2025-10-30, without a count, and 2026-01-21, with one, the first record with a count is 2025-11-18's, 158,727,456, in 6 lookup(s)
  SOLV   2024-04-01..2026-11-03 (  652 member-days): SOLV: between 2024-04-01, without a count, and 2024-06-26, with one, the first record with a count is 2024-04-02's, 172,710,000, in 5 lookup(s)
  VLTO   2023-10-02..2026-11-03 (  776 member-days): VLTO: between 2023-10-02, without a count, and 2023-12-27, with one, the first record with a count is 2023-10-04's, 246,060,000, in 6 lookup(s)
  VYLR   2026-10-01..2026-10-02 (    2 member-days): VYLR: no sample in 2026-10-01..2026-10-02 carries a share-class count, so there is nothing to search between
  VYLR   2026-10-05..2026-11-03 (   22 member-days): VYLR: no sample in 2026-10-05..2026-11-03 carries a share-class count, so there is nothing to search between
  WBD    2022-04-11..2026-10-06 ( 1125 member-days): WBD: between 2022-04-11, without a count, and 2022-07-13, with one, the first record with a count is 2022-04-12's, 2,233,730,000, in 5 lookup(s)
  15 spans, 66 lookups, 0 over the network this run

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
  counts carried into a span from before it, until a sample of its own resets them: 8
    CCL    from 2026-05-07 at 1,239,000,212 (a reorganisation), to 2026-08-07
    COHR   from 2022-09-08 at 130,870,000 (from its own earlier span), to 2022-12-07
    FRCB   from 2023-05-03 at 176,740,000 (from its own earlier span), to the span end, 1:1, FLAGGED
    OKE    from 2026-09-11 at 630,369,667 (a reorganisation), to the span end, 1:1, FLAGGED
    SBNY   from 2023-03-28 at 57,630,000 (from its own earlier span), to the span end, 1:1, FLAGGED
    SKYD   from 2026-10-06 at 1,090,445,692 (from its own earlier span), to the span end, 1:1, FLAGGED
    SMCI   from 2024-10-01 at 585,600,000 (from its own earlier span), to 2024-10-07
    XOM    from 2026-07-02 at 4,144,947,162 (a reorganisation), to the span end, 1:1, FLAGGED
  member-days without a count: 206 (midoffice07, before the search: 696) -- 169 inside the spans, before a first count; 37 after a member's last trade, outside its spans, which no price or count reaches
    AMTM      1: 2024-09-30..2024-09-30
    CEG       1: 2022-02-02..2022-02-02
    FDXF    109: 2026-06-01..2026-11-03
    GEHC      1: 2023-01-04..2023-01-04
    GEV       1: 2024-04-02..2024-04-02
    Q        13: 2025-11-03..2025-11-19
    SKYD      2: 2025-08-07..2025-08-08
    SOLS     13: 2025-10-30..2025-11-17
    SOLV      1: 2024-04-01..2024-04-01
    VLTO      2: 2023-10-02..2023-10-03
    VYLR     24: 2026-10-01..2026-11-03
    WBD       1: 2022-04-11..2022-04-11
  counts that move beyond 10% from one day to the next, no split the gate applies there: 103 -- a standing check; the gate tests counts across split events alone
    AJG    2025-01-07: 219,100,000 to 249,800,000, 1.1401
    AMCR   2025-10-09: 1,445,343,212 to 2,305,359,941, 1.5950
    AMD    2022-04-04: 1,207,610,000 to 1,627,370,000, 1.3476
    AMTM   2024-12-23: 153,280,000 to 243,300,000, 1.5873
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
    DGX    2025-04-09: 111,620,000 to 11,610,000, 0.1040
    DGX    2025-07-11: 11,610,000 to 111,635,472, 9.6155
    DOC    2024-03-04: 547,170,000 to 708,200,000, 1.2943
    DVN    2026-07-14: 621,440,299 to 1,153,403,107, 1.8560
    ECHO   2024-01-05: 36,160,000 to 140,060,000, 3.8733
    ECHO   2025-04-09: 140,490,000 to 155,200,000, 1.1047
    EG     2023-10-05: 39,280,000 to 43,400,000, 1.1049
    EG     2025-04-09: 42,980,000 to 47,790,000, 1.1119
    EG     2025-07-11: 47,790,000 to 42,534,728, 0.8900
    EQT    2023-10-05: 361,640,000 to 411,260,000, 1.1372
    EQT    2024-10-07: 441,590,000 to 594,700,000, 1.3467
    EXE    2023-01-04: 120,850,000 to 133,970,000, 1.1086
    EXE    2024-10-07: 131,280,000 to 226,900,000, 1.7284
    EXR    2023-10-05: 135,050,000 to 211,280,000, 1.5645
    FANG   2024-10-07: 178,340,000 to 294,740,000, 1.6527
    FITB   2026-04-13: 661,044,771 to 905,563,498, 1.3699
    GEN    2022-10-04: 580,060,000 to 665,570,000, 1.1474
    GEV    2024-06-27: 1,088,330,000 to 274,090,000, 0.2518
    GM     2024-04-08: 1,369,480,000 to 1,154,430,000, 0.8430
    GPN    2026-04-13: 236,743,716 to 275,243,013, 1.1626
    HBAN   2026-04-13: 1,459,390,757 to 2,037,119,660, 1.3959
    HON    2026-10-12: 635,675,701 to 316,940,010, 0.4986
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
    NWS    2024-07-09: 191,100,000 to 162,170,000, 0.8486
    NWS    2024-10-07: 162,170,000 to 190,080,000, 1.1721
    O      2024-04-08: 723,920,000 to 861,120,000, 1.1895
    OKE    2023-10-05: 447,450,000 to 582,480,000, 1.3018
    OMC    2026-01-09: 193,722,889 to 314,603,664, 1.6240
    PANW   2026-04-13: 697,000,000 to 816,000,000, 1.1707
    PAYC   2026-04-13: 56,269,005 to 47,666,129, 0.8471
    PCG    2022-04-04: 1,985,370,000 to 2,465,130,000, 1.2416
    PCG    2022-07-06: 2,465,130,000 to 1,987,480,000, 0.8062
    PCG    2025-07-11: 2,193,580,000 to 2,675,428,603, 1.2197
    PCG    2026-01-09: 2,675,580,992 to 2,197,910,425, 0.8215
    PCG    2026-04-13: 2,197,910,425 to 2,675,711,544, 1.2174
    PLD    2022-10-04: 740,340,000 to 923,250,000, 1.2471
    RDDT   2024-06-20: 36,880,000 to 44,440,000, 1.2050
    RDDT   2024-09-19: 44,440,000 to 94,960,000, 2.1368
    RDDT   2024-12-18: 94,960,000 to 119,740,000, 1.2610
    RDDT   2025-12-22: 134,585,853 to 51,670,930, 0.3839
    RDDT   2026-03-25: 51,670,930 to 139,649,508, 2.7027
    SBUX   2026-07-14: 1,139,300,000 to 139,700,000, 0.1226
    SBUX   2026-10-12: 139,700,000 to 1,140,000,000, 8.1603
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
    XYZ    2026-01-09: 549,574,000 to 59,998,000, 0.1092
    XYZ    2026-04-13: 59,998,000 to 539,103,000, 8.9853

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
  carried from an earlier record: 39 met, 3 flagged; ratio median 1.0003, p1 0.6572, p99 3.2917
  the day's own record: 11,855 met, 762 flagged; ratio median 1.0000, p1 0.9819, p99 2.0973
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
    FAST   FAST   2025-07-11 weighted 1,147,241,276 against 1,146,900,000: 1.0003
    GRMN   GRMN   2025-07-11 weighted 192,541,456 against 192,400,000: 1.0007
    GRMN   GRMN   2025-10-09 weighted 192,493,945 against 192,400,000: 1.0005
    GRMN   GRMN   2026-01-09 weighted 192,334,806 against 192,400,000: 0.9997
    GRMN   GRMN   2026-04-13 weighted 192,480,830 against 192,400,000: 1.0004
    HON    HON    2026-07-14 weighted 316,826,559 against 635,675,701: 0.4984  FLAGGED
    HONA   HONA   2026-09-15 weighted 316,952,725 against 316,826,559: 1.0004
    HONA   HONA   2026-11-03 weighted 317,031,836 against 316,826,559: 1.0006
    IBKR   IBKR   2025-07-11 weighted 435,835,720 against 435,720,000: 1.0003
    KLAC   KLAC   2026-07-14 weighted 1,306,275,210 against 1,310,766,110: 0.9966
    MNST   MNST   2026-10-12 weighted 1,959,051,764 against 1,956,016,220: 1.0016
    MNST   MNST   2026-11-03 weighted 1,959,051,764 against 1,956,016,220: 1.0016
    NOW    NOW    2026-01-09 weighted 1,055,142,985 against 1,040,000,000: 1.0146
    OKE    OKE    2026-09-11 weighted 630,369,667 against 630,369,667: 1.0000
    OKE    OKE    2026-11-03 weighted 630,450,437 against 630,369,667: 1.0001
    ORLY   ORLY   2025-07-11 weighted 854,841,825 against 858,600,000: 0.9956
    Q      Q      2026-10-28 weighted 209,208,543 against 209,339,207: 0.9994
    Q      Q      2026-11-03 weighted 209,208,543 against 209,339,207: 0.9994
    SKYD   PSKY   2026-10-05 weighted 1,124,520,841 against 1,090,445,692: 1.0312
    SKYD   SKYD   2026-11-03 weighted 5,042,178,087 against 1,090,445,692: 4.6240  FLAGGED
    SMCI   SMCI   2024-10-01 weighted 585,565,270 against 585,600,000: 0.9999
    SNDK   SNDK   2026-11-03 weighted 142,289,365 against 148,089,758: 0.9608
    SOLS   SOLS   2026-10-21 weighted 158,847,369 against 158,804,273: 1.0003
    SOLS   SOLS   2026-11-03 weighted 158,847,369 against 158,804,273: 1.0003
    TPL    TPL    2026-01-09 weighted 68,938,230 against 68,961,978: 0.9997
    WSM    WSM    2024-07-09 weighted 128,616,166 against 128,220,000: 1.0031
    XOM    XOM    2026-07-02 weighted 4,144,947,162 against 4,144,947,162: 1.0000
    XOM    XOM    2026-10-01 weighted 4,111,911,960 against 4,144,947,162: 0.9920
    XOM    XOM    2026-11-03 weighted 4,079,000,000 against 4,144,947,162: 0.9841
  the day's own record flagged, by security -- 75 securities; where every record is, a class count beside a company-wide weighted one:
    ECHO    22 of  22 records, ratio 1.8237..2.3492, from 2022-01-03 to 2026-11-03
    XYZ     22 of  22 records, ratio 1.1073..10.1281, from 2022-01-03 to 2026-11-03
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
    NWS     21 of  21 records, ratio 2.9362..3.5087, from 2022-01-03 to 2026-11-03
    NWSA    21 of  21 records, ratio 1.4841..1.5170, from 2022-01-03 to 2026-11-03
    RL      21 of  21 records, ratio 1.5104..1.6374, from 2022-01-03 to 2026-11-03
    TSN     21 of  21 records, ratio 1.2317..1.2484, from 2022-01-03 to 2026-11-03
    UA      21 of  21 records, ratio 1.8811..2.1021, from 2022-01-03 to 2026-11-03
    UAA     21 of  21 records, ratio 2.2448..2.5247, from 2022-01-03 to 2026-11-03
    UPS     21 of  21 records, ratio 1.1354..1.1946, from 2022-01-03 to 2026-11-03
    WDAY    21 of  21 records, ratio 1.2240..1.2953, from 2022-01-03 to 2026-11-03
    UHS     20 of  21 records, ratio 1.1074..1.1402, from 2022-04-04 to 2026-11-03
    V       20 of  21 records, ratio 1.1017..1.3037, from 2022-01-03 to 2026-10-12
    CCL     19 of  21 records, ratio 1.1179..1.1956, from 2022-01-03 to 2026-05-06
    APP     13 of  21 records, ratio 1.1236..1.6528, from 2022-01-03 to 2025-01-07
    PCG     13 of  21 records, ratio 0.8214..1.2235, from 2022-10-04 to 2026-11-03
    RDDT    12 of  12 records, ratio 1.3141..4.3111, from 2024-03-21 to 2026-11-03
    DISH     9 of   9 records, ratio 1.8056..1.8212, from 2022-01-03 to 2023-12-29
    TTD      6 of  21 records, ratio 1.1001..1.1022, from 2022-01-03 to 2026-11-03
    VEEV     6 of  21 records, ratio 1.1001..1.1103, from 2022-01-03 to 2023-04-05
    STZ      4 of  21 records, ratio 1.1412..1.1456, from 2022-01-03 to 2022-10-04
    DDOG     3 of  21 records, ratio 1.1127..1.2131, from 2022-01-03 to 2022-07-06
    DISCA    3 of   3 records, ratio 2.9627..2.9900, from 2022-01-03 to 2022-04-08
    DISCK    3 of   3 records, ratio 1.5325..1.5393, from 2022-01-03 to 2022-04-08
    EG       3 of  22 records, ratio 0.8984..1.1053, from 2023-07-07 to 2025-04-09
    BE       2 of  21 records, ratio 1.1008..1.1877, from 2022-01-03 to 2022-10-04
    CPT      2 of  21 records, ratio 1.1511..1.1511, from 2026-10-12 to 2026-11-03
    FMC      2 of  21 records, ratio 1.2421..1.2422, from 2026-10-12 to 2026-11-03
    MLM      2 of  21 records, ratio 1.1824..1.1824, from 2026-10-12 to 2026-11-03
    VMRK     2 of  22 records, ratio 2.0637..2.0637, from 2026-10-12 to 2026-11-03
    AAL      1 of  21 records, ratio 2.0533..2.0533, from 2023-04-05 to 2023-04-05
    AMCR     1 of  21 records, ratio 1.5853..1.5853, from 2025-07-11 to 2025-07-11
    AMTM     1 of  10 records, ratio 1.5872..1.5872, from 2024-10-01 to 2024-10-01
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
    DGX      1 of  21 records, ratio 9.5588..9.5588, from 2025-04-09 to 2025-04-09
    GEV      1 of  12 records, ratio 0.2500..0.2500, from 2024-04-03 to 2024-04-03
    INCY     1 of  21 records, ratio 0.8761..0.8761, from 2024-07-09 to 2024-07-09
    JNJ      1 of  21 records, ratio 1.1980..1.1980, from 2023-04-05 to 2023-04-05
    KEY      1 of  21 records, ratio 1.1163..1.1163, from 2025-01-07 to 2025-01-07
    MHK      1 of  21 records, ratio 0.8582..0.8582, from 2026-11-03 to 2026-11-03
    SBUX     1 of  21 records, ratio 8.1582..8.1582, from 2026-07-14 to 2026-07-14
    SKYD     1 of   5 records, ratio 0.6333..0.6333, from 2025-08-11 to 2025-08-11
    SMCI     1 of  22 records, ratio 9.9994..9.9994, from 2024-09-30 to 2024-09-30
    SWKS     1 of  21 records, ratio 1.5628..1.5628, from 2026-11-03 to 2026-11-03
    WBD      1 of  19 records, ratio 0.2275..0.2275, from 2022-04-12 to 2022-04-12
    WHR      1 of  21 records, ratio 1.1397..1.1397, from 2026-04-13 to 2026-04-13

Industries
  each security's SIC code from its own latest record with one: 587 of 592; none for DISCA, DISCK, FRCB, SBNY, VYLR

Wrote
  claude\scratch\midoffice08\regress\panel_final\sp500_securities_2022-01-03_2026-11-03.npz
  claude\scratch\midoffice08\regress\panel_final\sp500_securities_2022-01-03_2026-11-03.json
```

### crosscheck_old.txt -- the panel on securities against the old panel, cell by cell

```text
cells both panels take from one ticker's bar on one day: 581,521
  raw closes equal exactly: 581,521; differ 0
  the adjusted bars' volume, the old panel's against the new one's put back on its ticker's basis: equal 581,521, differ 0
  the old panel's volume equal to the new one's traded volume: 557,135; not 24,386
  new adjusted close over old, by run scale: 1 x581,521
  consecutive days within one run: 580,978; the new price return less the old, largest 2.22e-16
old-panel priced cells no new column takes (a ticker's days outside every member security's spans): 2,107 over 4 tickers
  BNY     1028
  DOC      542
  INFO     518
  META      19
member-days on cells both panels take, both with a count: 554,241; the old more than 10% from the new: 11,529
  GM      378 days 2024-04-08..2026-01-08, old over new 1.105..1.218
  MPC     378 days 2022-04-04..2025-01-06, old over new 1.102..1.235
  WRB     320 days 2022-03-24..2025-01-06, old over new 0.666..0.678
  TPR     315 days 2022-07-06..2026-01-08, old over new 1.110..1.159
  AMCR    264 days 2025-10-09..2026-11-02, old over new 0.627..5.000
  MNST    255 days 2023-03-28..2026-11-03, old over new 0.498..0.501
  APA     252 days 2022-10-04..2025-01-06, old over new 0.826..1.113
  O       252 days 2023-10-05..2025-01-06, old over new 0.831..0.885
  TSLA    232 days 2022-08-25..2026-11-02, old over new 0.320..0.886
  PCAR    228 days 2023-02-08..2024-01-04, old over new 0.665..0.667
  COO     222 days 2024-02-20..2025-01-06, old over new 0.249..0.250
  NOW     219 days 2025-12-18..2026-11-02, old over new 0.198..0.201
  WMT     218 days 2024-02-26..2025-01-06, old over new 0.333..0.335
  DOC     213 days 2024-03-04..2025-01-06, old over new 0.336..0.341
  ODFL    195 days 2024-03-28..2025-01-06, old over new 0.500..0.509
  AMD     189 days 2022-04-04..2023-01-03, old over new 0.742..0.748
  FOXA    189 days 2023-07-07..2025-01-06, old over new 1.124..1.209
  IP      189 days 2025-04-09..2026-01-08, old over new 0.658..0.658
  MGM     189 days 2022-07-06..2025-01-06, old over new 1.101..1.193
  SPGI    189 days 2022-04-04..2023-01-03, old over new 0.694..0.723
  APH     186 days 2024-06-12..2026-11-03, old over new 0.497..0.501
  FAST    159 days 2025-05-22..2026-01-08, old over new 0.499..0.500
  BKNG    147 days 2026-04-06..2026-11-02, old over new 0.040..0.043
  ORLY    147 days 2025-06-10..2026-01-08, old over new 0.067..0.068
  AMZN    146 days 2022-06-06..2023-01-03, old over new 0.050..0.050
  NVDA    145 days 2024-06-10..2025-01-06, old over new 0.099..0.101
  META    143 days 2022-06-09..2023-01-03, old over new 0.027..0.027
  DVA     142 days 2022-10-04..2026-11-02, old over new 1.107..1.147
  FITB    142 days 2026-04-13..2026-11-02, old over new 0.729..0.730
  GPN     142 days 2026-04-13..2026-11-02, old over new 0.860..0.895
  HBAN    142 days 2026-04-13..2026-11-02, old over new 0.716..0.722
  OMC     142 days 2026-04-13..2026-11-02, old over new 1.104..1.147
  WAT     142 days 2026-04-13..2026-11-02, old over new 0.606..0.607
  DXCM    141 days 2022-06-13..2023-01-03, old over new 0.247..0.249
  CMG     134 days 2024-06-26..2025-01-06, old over new 0.020..0.020
  CPRT    134 days 2022-11-04..2024-01-04, old over new 0.498..0.499
  FTNT    134 days 2022-06-23..2023-01-03, old over new 0.203..0.207
  BBWI    126 days 2022-07-06..2023-01-03, old over new 1.127..1.129
  BKR     126 days 2022-07-06..2023-01-03, old over new 0.860..0.884
  CFG     126 days 2022-07-06..2023-01-03, old over new 0.860..0.860
  COF     126 days 2022-10-04..2026-01-08, old over new 0.596..1.109
  EBAY    126 days 2022-07-06..2023-01-03, old over new 1.118..1.139
  MRO     126 days 2022-07-06..2023-01-03, old over new 1.100..1.149
  MTB     126 days 2022-07-06..2023-01-03, old over new 0.717..0.733
  TTWO    126 days 2022-07-06..2023-01-03, old over new 0.692..0.712
  WELL    126 days 2024-07-09..2025-01-06, old over new 0.880..0.896
  XYL     126 days 2023-07-07..2024-01-04, old over new 0.748..0.753
  CCL     124 days 2022-10-04..2026-11-02, old over new 0.852..0.894
  AVGO    122 days 2024-07-15..2025-01-06, old over new 0.100..0.101
  GOOG    118 days 2022-07-18..2023-01-03, old over new 0.051..0.052
```

### capture/report_A.txt -- run A: the old panel, --industries --membership

```text
Panel
  1214 trading days, 506 securities
  2022-01-03 .. 2026-11-03
  on tickers: price returns, the adjusted bars' volumes
  583,081 of 614,284 return observations present (94.9%)

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, covering 1976-07-01 .. 2026-11-04
  the constituent list predates the change of 2026-10-30 -- OKTA added, KVUE removed -- and is read with it applied
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
  member-periods over the 938 periods: 472,608; in the regression 422,203; out of it, by why:
    not in the panel         44,038
    no weight                 6,114   VMRK 938, SKYD 938, ECHO 938, MRSH 938, FISV 610, SW 481
    no return                    23   SBNY 1, SIVB 1, PXD 1, MRO 1, JNPR 1, K 1
    an exposure missing         230   FISV 126, GEN 65, COR 38, EG 1
    the first exposure missing: Momentum 230

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

### capture/report_B.txt -- run B: the panel on securities, --returns price --volumes adjusted

```text
Panel
  1214 trading days, 592 securities
  2022-01-03 .. 2026-11-03
  on securities, each a share class labelled by its last ticker; price returns, the market's price return; adjusted volumes
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
  window descriptors need at least 50% of the window valid; from a window with none, over 5,170,273 values: none; from one under 50% valid: none
  489,655 raw values winsorised before standardising
  192,974 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/52  LCAPCB 52/52  LPRI 4/52  DASTD 4/55  HILO 9/52  CMRA 6/54  BTSG 7/58
    HALPHA 3/58  SERDP 0/58  RSTR 4/58  STOM 15/53  STOQ 13/55  STOA 14/58  VLVR 17/58

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         94%      962    8.56%     10.03%     1.19
  DASTD        94%    1,149    6.49%      8.42%     1.14
  RSTR         94%      938    5.58%      7.83%     1.13
  HALPHA       94%      962    4.46%      6.69%     0.65
  HILO         94%    1,213    3.87%      6.74%     0.93
  CMRA         93%      938    3.64%      6.17%     1.18
  STOQ         94%    1,145    3.21%      5.71%     1.27
  STOM         94%    1,191    3.11%      5.67%     1.24
  STOA         94%      938    3.04%      5.71%     1.17
  LPRI         94%    1,213    1.85%      4.03%    -0.53
  LNCAP        93%    1,213    1.77%      4.03%     0.75
  VLVR         94%      938    1.77%      3.95%     1.85
  LCAPCB       93%    1,213    1.65%      3.93%     0.48
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
  25-factor model, annualised, estimated on 938 daily period(s) for 552 of 592 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
  each period regresses day t's returns on the exposures, weights and membership of 1 trading day(s) earlier; the forecast exposures are 2026-11-03's
  mean cross-sectional R-squared 0.291
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

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 2.2e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  +1.4e-17   1.000000   503
  Momentum                    -7.2e-18   1.000000   500
  Size                        +2.9e-17   1.000000   501
  Size Nonlinearity           -7.2e-18   1.000000   501
  Trading Activity            +7.2e-18   1.000000   501
  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,136 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates

  The size pair on 2026-11-03, over the 500 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
                                                 Size  Size Nonlinearity     risk
  smallest 85, equally weighted                -3.139             -1.810   11.33%
  largest 50, equally weighted                 +0.170             +0.052    0.61%
  the index, cap-weighted                      +0.001             -0.001    0.01%

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     503    505          503          503
  of them in the panel      503     503    505          503          503
  in the regression         499     502    503          502          500
  member-periods over the 938 periods: 471,826; in the regression 470,548; out of it, by why:
    not in the panel              0
    no weight                   197   FDXF 108, VYLR 23, Q 13, SOLS 13, JNPR 4, HES 3
    no return                    23   WBA 1, WBD 1, WRK 1, SBNY 1, SIVB 1, PXD 1
    an exposure missing       1,058   SW 126, SOLV 125, GEV 125, VLTO 124, SKYD 124, Q 113
    the first exposure missing: Momentum 1,058

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               19.30%     +24.86%       38.8%      938          0      0
  Finance & Real Estate      15.36%     +17.96%       31.0%      938          0      0
  Manufacturing              10.13%     +18.92%       23.6%      938          0      0
  Mining                     18.83%     +12.39%       18.0%      938          0      0
  Retail                     13.23%     +12.62%       10.6%      938          0      0
  SIC20                      12.15%      +8.17%       32.8%      938          0      0
  SIC28                      13.14%     +15.27%       10.3%      938          0      0
  SIC35                      18.87%     +28.28%        5.2%      938          0      0
  SIC36                      21.16%     +30.12%        4.5%      938          0      0
  SIC37                      14.37%     +13.60%       18.6%      938          0      0
  SIC38                      14.78%     +15.34%       10.4%      938          0      0
  SIC49                      15.86%     +21.67%       11.3%      938          0      0
  SIC60                      21.79%     +19.74%        7.5%      938          0      0
  SIC62                      18.77%     +17.42%       11.5%      938          0      0
  SIC63                      14.51%     +18.43%       14.7%      938          0      0
  SIC67                      15.00%     +15.69%       16.8%      938          0      0
  SIC73                      18.23%     +14.84%        2.9%      938          0      0
  Services                   12.70%     +14.44%       28.4%      938          0      0
  Transport & Utilities      13.71%     +12.84%       15.9%      938          0      0
  Wholesale                  13.96%     +19.57%       46.5%      938          0      0
  Volatility                  9.03%      +2.50%        3.7%      938          0      0
  Momentum                    7.25%      +4.26%        3.6%      938          0      0
  Size                        3.67%      +3.13%        7.9%      938          0      0
  Size Nonlinearity           2.21%      -0.55%       50.1%      938          0      0
  Trading Activity            5.80%      +4.11%        7.5%      938          0      0

  specific risk   median 24.83%   p10 15.60%   p90 40.75%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.1%, SMCI 1.9%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
                    relative residual spread 0.136, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.342
                    largest shares of the sd's square: FRCB 66.7%, SIVB 3.5%, SMCI 1.9%, LUMN 1.8%, SNDK 0.9%
                    the characteristics' weight in a blended forecast, at the median history of 938 periods: 2.79%

Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
  - 40 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CERN
  - 50% of the estimated variance of Size Nonlinearity is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 11 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: AMTM, FBIN, GAP, IPGP, MBC
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### capture/report_C.txt -- run C: the panel on securities, --returns price --volumes traded

```text
Panel
  1214 trading days, 592 securities
  2022-01-03 .. 2026-11-03
  on securities, each a share class labelled by its last ticker; price returns, the market's price return; traded volumes
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
  window descriptors need at least 50% of the window valid; from a window with none, over 5,170,273 values: none; from one under 50% valid: none
  462,586 raw values winsorised before standardising
  191,693 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/52  LCAPCB 52/52  LPRI 4/52  DASTD 4/55  HILO 9/52  CMRA 6/54  BTSG 7/58
    HALPHA 3/58  SERDP 0/58  RSTR 4/58  STOM 15/53  STOQ 13/55  STOA 16/58  VLVR 19/58

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         94%      962    8.56%     10.03%     1.19
  DASTD        94%    1,149    6.49%      8.42%     1.14
  RSTR         94%      938    5.58%      7.83%     1.13
  HALPHA       94%      962    4.46%      6.69%     0.65
  HILO         94%    1,213    3.87%      6.74%     0.93
  CMRA         93%      938    3.64%      6.17%     1.18
  STOQ         94%    1,145    3.58%      5.92%     0.73
  STOM         94%    1,191    3.46%      5.87%     0.71
  STOA         94%      938    3.25%      5.78%     0.65
  LPRI         94%    1,213    1.85%      4.03%    -0.53
  LNCAP        93%    1,213    1.77%      4.03%     0.75
  LCAPCB       93%    1,213    1.65%      3.93%     0.48
  VLVR         94%      938    1.55%      3.71%     1.61
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
  25-factor model, annualised, estimated on 938 daily period(s) for 552 of 592 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
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

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 2.2e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  +1.4e-17   1.000000   503
  Momentum                    -7.2e-18   1.000000   500
  Size                        +2.9e-17   1.000000   501
  Size Nonlinearity           -7.2e-18   1.000000   501
  Trading Activity            +0.0e+00   1.000000   501
  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,136 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates

  The size pair on 2026-11-03, over the 500 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
                                                 Size  Size Nonlinearity     risk
  smallest 85, equally weighted                -3.139             -1.810   11.77%
  largest 50, equally weighted                 +0.170             +0.052    0.62%
  the index, cap-weighted                      +0.001             -0.001    0.01%

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     503    505          503          503
  of them in the panel      503     503    505          503          503
  in the regression         499     502    503          502          500
  member-periods over the 938 periods: 471,826; in the regression 470,548; out of it, by why:
    not in the panel              0
    no weight                   197   FDXF 108, VYLR 23, Q 13, SOLS 13, JNPR 4, HES 3
    no return                    23   WBA 1, WBD 1, WRK 1, SBNY 1, SIVB 1, PXD 1
    an exposure missing       1,058   SW 126, SOLV 125, GEV 125, VLTO 124, SKYD 124, Q 113
    the first exposure missing: Momentum 1,058

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               19.40%     +23.80%       38.3%      938          0      0
  Finance & Real Estate      15.33%     +17.05%       31.1%      938          0      0
  Manufacturing              10.20%     +18.69%       23.2%      938          0      0
  Mining                     18.60%     +13.34%       18.5%      938          0      0
  Retail                     13.18%     +13.62%       10.6%      938          0      0
  SIC20                      12.07%      +9.10%       33.0%      938          0      0
  SIC28                      13.13%     +14.81%       10.3%      938          0      0
  SIC35                      18.77%     +28.43%        5.3%      938          0      0
  SIC36                      21.13%     +31.06%        4.5%      938          0      0
  SIC37                      14.22%     +13.94%       19.2%      938          0      0
  SIC38                      14.69%     +15.08%       10.7%      938          0      0
  SIC49                      15.88%     +21.87%       11.2%      938          0      0
  SIC60                      21.89%     +19.57%        7.4%      938          0      0
  SIC62                      18.80%     +16.72%       11.5%      938          0      0
  SIC63                      14.57%     +17.93%       14.5%      938          0      0
  SIC67                      15.16%     +15.05%       16.5%      938          0      0
  SIC73                      18.24%     +14.26%        2.9%      938          0      0
  Services                   12.76%     +15.53%       28.0%      938          0      0
  Transport & Utilities      13.56%     +13.72%       16.2%      938          0      0
  Wholesale                  13.89%     +19.72%       46.8%      938          0      0
  Volatility                  9.29%      +3.66%        3.6%      938          0      0
  Momentum                    7.27%      +4.45%        3.6%      938          0      0
  Size                        3.70%      +3.23%        7.8%      938          0      0
  Size Nonlinearity           2.31%      -0.62%       47.1%      938          0      0
  Trading Activity            6.22%      +2.26%        7.6%      938          0      0

  specific risk   median 24.79%   p10 15.68%   p90 40.72%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
                    relative residual spread 0.131, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.340
                    largest shares of the sd's square: FRCB 67.5%, SIVB 3.6%, SMCI 1.8%, LUMN 1.8%, SBNY 1.0%
                    the characteristics' weight in a blended forecast, at the median history of 938 periods: 3.00%

Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
  - 40 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CERN
  - 47% of the estimated variance of Size Nonlinearity is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 11 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: AMTM, FBIN, GAP, IPGP, MBC
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### capture/report_D.txt -- run D: the panel on securities, --returns total --volumes traded

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
  462,364 raw values winsorised before standardising
  191,705 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/52  LCAPCB 52/52  LPRI 4/52  DASTD 4/55  HILO 9/52  CMRA 5/54  BTSG 7/58
    HALPHA 4/58  SERDP 0/58  RSTR 4/58  STOM 15/53  STOQ 13/55  STOA 16/58  VLVR 19/58

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         94%      962    8.58%     10.03%     1.09
  DASTD        94%    1,149    6.51%      8.42%     1.04
  RSTR         94%      938    5.49%      7.78%     1.04
  HALPHA       94%      962    4.42%      6.67%     0.56
  HILO         94%    1,213    3.88%      6.74%     0.81
  CMRA         93%      938    3.74%      6.26%     1.16
  STOQ         94%    1,145    3.60%      5.92%     0.67
  STOM         94%    1,191    3.47%      5.87%     0.67
  STOA         94%      938    3.26%      5.78%     0.59
  LPRI         94%    1,213    1.86%      4.03%    -0.91
  LNCAP        93%    1,213    1.78%      4.03%     0.68
  LCAPCB       93%    1,213    1.65%      3.93%     0.44
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
  25-factor model, annualised, estimated on 938 daily period(s) for 552 of 592 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 938
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

  Style exposures over the index members with a value, on all 939 dates an estimated period or the forecast reads: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 2.2e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  -3.6e-18   1.000000   503
  Momentum                    +3.6e-17   1.000000   500
  Size                        +2.9e-17   1.000000   501
  Size Nonlinearity           -7.2e-18   1.000000   501
  Trading Activity            -7.2e-18   1.000000   501
  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved 4,136 values of the index members, 4 on 2026-11-03; its regression-weighted covariance with Size over them is at most 1.1e-15 (5.1 eps), inside rounding on 939 of the 939 dates

  The size pair on 2026-11-03, over the 500 index members with a forecast: each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over their block of the annualised covariance
                                                 Size  Size Nonlinearity     risk
  smallest 85, equally weighted                -3.139             -1.810   11.82%
  largest 50, equally weighted                 +0.170             +0.052    0.62%
  the index, cap-weighted                      +0.001             -0.001    0.01%

  Securities per estimated period, over 938
                            min  median    max   2023-02-08   2026-11-03
  index members             503     503    505          503          503
  of them in the panel      503     503    505          503          503
  in the regression         499     502    503          502          500
  member-periods over the 938 periods: 471,826; in the regression 470,548; out of it, by why:
    not in the panel              0
    no weight                   197   FDXF 108, VYLR 23, Q 13, SOLS 13, JNPR 4, HES 3
    no return                    23   WBA 1, WBD 1, WRK 1, SBNY 1, SIVB 1, PXD 1
    an exposure missing       1,058   SW 126, SOLV 125, GEV 125, VLTO 124, SKYD 124, Q 113
    the first exposure missing: Momentum 1,058

  Over the 938 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                                 periods with a member present
  factor                  annual vol  annual mean  noise share      >=1  exactly 1   none
  Construction               19.43%     +24.77%       38.1%      938          0      0
  Finance & Real Estate      15.35%     +18.11%       30.9%      938          0      0
  Manufacturing              10.20%     +20.90%       23.2%      938          0      0
  Mining                     18.63%     +14.96%       18.5%      938          0      0
  Retail                     13.14%     +14.73%       10.7%      938          0      0
  SIC20                      11.98%     +11.04%       33.5%      938          0      0
  SIC28                      13.14%     +16.64%       10.2%      938          0      0
  SIC35                      18.81%     +29.69%        5.2%      938          0      0
  SIC36                      21.13%     +32.58%        4.5%      938          0      0
  SIC37                      14.24%     +15.43%       19.1%      938          0      0
  SIC38                      14.70%     +15.82%       10.6%      938          0      0
  SIC49                      15.82%     +24.01%       11.3%      938          0      0
  SIC60                      21.86%     +21.96%        7.4%      938          0      0
  SIC62                      18.75%     +18.62%       11.5%      938          0      0
  SIC63                      14.57%     +19.16%       14.5%      938          0      0
  SIC67                      15.02%     +18.00%       16.7%      938          0      0
  SIC73                      18.15%     +15.18%        3.0%      938          0      0
  Services                   12.74%     +16.18%       27.9%      938          0      0
  Transport & Utilities      13.52%     +15.39%       16.3%      938          0      0
  Wholesale                  13.97%     +20.70%       46.2%      938          0      0
  Volatility                  9.37%      +3.08%        3.5%      938          0      0
  Momentum                    7.21%      +4.23%        3.6%      938          0      0
  Size                        3.72%      +3.14%        7.7%      938          0      0
  Size Nonlinearity           2.31%      -0.51%       47.0%      938          0      0
  Trading Activity            6.21%      +2.48%        7.6%      938          0      0

  specific risk   median 24.80%   p10 15.59%   p90 40.78%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
                    relative residual spread 0.133, 1.4826 x the MAD of the securities' average relative residuals; their sd 0.340
                    largest shares of the sd's square: FRCB 67.5%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
                    the characteristics' weight in a blended forecast, at the median history of 938 periods: 2.93%

Qualifications
  - 251 period(s) dropped for having fewer than 30 securities: 2022-01-04, 2022-01-05, 2022-01-06 and 248 more
  - 24 period(s) held enough securities and were not estimated, the caller's periods ruling them out: 2023-01-04, 2023-01-05, 2023-01-06 and 21 more
  - 40 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CERN
  - 47% of the estimated variance of Size Nonlinearity is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 11 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: AMTM, FBIN, GAP, IPGP, MBC
  - 276 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### capture_AC.txt and capture_BD.txt -- the runs captured in-process

```text
A: status 0, mean R2 0.298076, 938 periods, 66s
C: status 0, mean R2 0.292774, 938 periods, 47s
B: status 0, mean R2 0.290903, 938 periods, 47s
D: status 0, mean R2 0.293189, 938 periods, 47s
```

### compare.txt -- the comparison in three steps

```text
====================================================================================================
Step 1. identity: the old panel against the new one, price returns, adjusted volumes: A -> B
  A: data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz --industries --membership
  B: data/external/polygon/sp500_securities_2022-01-03_2026-11-03.npz --industries --membership --returns price --volumes adjusted
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    A: min 422 median 451 max 476 first 476 last 422
    B: min 499 median 502 max 503 first 502 last 500
    B less A, per period: min +26 median +50 max +78
  mean cross-sectional R-squared: A 0.298076, B 0.290903, difference -0.007173
  factor                    vol A    vol B    mean A    mean B   d mean   noise A   noise B
  Construction             17.06%   19.30%   +23.03%   +24.86%   +1.84%     52.0%     38.8%
  Finance & Real Estate    15.31%   15.36%   +21.56%   +17.96%   -3.60%     14.3%     31.0%
  Manufacturing            10.95%   10.13%   +20.59%   +18.92%   -1.67%     20.7%     23.6%
  Mining                   20.16%   18.83%   +10.85%   +12.39%   +1.55%     16.2%     18.0%
  Retail                   13.09%   13.23%   +13.07%   +12.62%   -0.44%     10.7%     10.6%
  SIC20                    12.74%   12.15%    +7.85%    +8.17%   +0.32%     30.6%     32.8%
  SIC28                    13.11%   13.14%   +15.02%   +15.27%   +0.24%      9.8%     10.3%
  SIC35                    17.45%   18.87%   +26.15%   +28.28%   +2.13%      6.3%      5.2%
  SIC36                    20.75%   21.16%   +28.22%   +30.12%   +1.90%      5.2%      4.5%
  SIC37                    14.15%   14.37%   +12.69%   +13.60%   +0.91%     18.2%     18.6%
  SIC38                    14.19%   14.78%   +15.35%   +15.34%   -0.01%     11.5%     10.4%
  SIC49                    15.80%   15.86%   +21.24%   +21.67%   +0.43%     12.3%     11.3%
  SIC60                    21.92%   21.79%   +19.03%   +19.74%   +0.71%      7.3%      7.5%
  SIC63                    14.86%   14.51%   +18.57%   +18.43%   -0.15%     13.8%     14.7%
  SIC67                    15.19%   15.00%   +17.12%   +15.69%   -1.43%     17.2%     16.8%
  SIC73                    17.34%   18.23%   +14.93%   +14.84%   -0.10%      3.5%      2.9%
  Services                 12.86%   12.70%   +14.04%   +14.44%   +0.40%     26.4%     28.4%
  Transport & Utilities    13.34%   13.71%   +12.35%   +12.84%   +0.48%     16.9%     15.9%
  Wholesale                14.77%   13.96%   +18.69%   +19.57%   +0.88%     40.0%     46.5%
  Volatility                8.77%    9.03%    +3.39%    +2.50%   -0.90%      3.8%      3.7%
  Momentum                  6.62%    7.25%    +3.60%    +4.26%   +0.66%      4.4%      3.6%
  Size                      4.03%    3.67%    +2.86%    +3.13%   +0.27%      7.1%      7.9%
  Size Nonlinearity         2.54%    2.21%    +0.40%    -0.55%   -0.95%     41.8%     50.1%
  Trading Activity          5.07%    5.80%    +4.40%    +4.11%   -0.29%     10.1%      7.5%
  SIC62                  only in B: vol 18.77%, mean +17.42%, noise 11.5%
  residual spread (1.4826 x MAD; sd): A 0.1333 (0.3612); B 0.1362 (0.3418)
    largest shares of the sd's square, A: FRC 72.4%, SIVB 3.7%, LUMN 2.2%, BNY 1.5%, ENPH 1.0%
    largest shares of the sd's square, B: FRCB 66.7%, SIVB 3.5%, SMCI 1.9%, LUMN 1.8%, SNDK 0.9%
    largest shares of V^2, A: MRNA 2.6%, FRC 2.0%, ENPH 1.3%, ALB 0.9%, CTVA 0.9%
    largest shares of V^2, B: MRNA 2.1%, SMCI 1.9%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: A p10 15.40% median 23.77% p90 36.50% (468); B p10 15.60% median 24.83% p90 40.75% (552)
  specific risk, the 467 in both: A p10 15.40% median 23.77% p90 36.50% (467); B p10 15.29% median 23.78% p90 36.46% (467)
    of those in both, the largest rises: FISV +2.74%, DHI +2.03%, PHM +1.84%, LEN +1.83%, NVR +1.65%, CME +1.64%, SYF +1.51%, CBOE +1.51%
    the largest falls: PWR -1.71%, UA -2.07%, CPAY -3.18%, BRO -4.44%, AON -4.51%, AJG -5.07%, MRSH -11.00%, BNY -13.99%
  fallback forecasts: A 8: INFO, IPGP, MRSH, PENN, PVH, UA, UAA, VNO
                      B 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
    in B not A: AMTM, FBIN, GAP, MBC, SOLS; in A not B: INFO, MRSH
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    A -3.110 -1.647 risk 11.94%; B -3.139 -1.810 risk 11.33%
    largest 50, equally weighted     A +0.012 +0.155 risk 0.38%; B +0.170 +0.052 risk 0.61%
    the index, cap-weighted          A +0.002 +0.000 risk 0.01%; B +0.001 -0.001 risk 0.01%
  regression universe: A 482, B 567; in A not B: CDAY, FRC
    in B not A (87): ABNB, ACGL, APO, APP, ARES, AXON, BE, BG, BLDR, BX, CASY, CEG, CIEN, COHR, COIN, CPT, CRH, CRWD, CSGP, CVNA, DASH, DAY, DDOG, DECK, DELL, ECHO, EME, EQT, ERIE, EXE, FERG, FICO, FIX, FLEX, FRCB, FSLR, GDDY, GEHC, GEV, HOOD, HUBB, IBKR, INVH, JBL, KDP, KKR, KVUE, LII, LITE, LULU, MOH, MRSH, MRVL, NDSN, OKTA, ON, P, PANW, PARA, PCG, PLTR, PODD, Q, RDDT, SKYD, SMCI, SNDK, SOLV, STLD, TKO, TPL, TRGP, TTD, TWLO, UBER, VEEV, VICI, VLTO, VMRK, VRT, VST, WBD, WDAY, WRK, WSM, WTW, XYZ
====================================================================================================
Step 2. volumes: the new panel, adjusted volumes against traded ones: B -> C
  B: data/external/polygon/sp500_securities_2022-01-03_2026-11-03.npz --industries --membership --returns price --volumes adjusted
  C: data/external/polygon/sp500_securities_2022-01-03_2026-11-03.npz --industries --membership --returns price --volumes traded
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    B: min 499 median 502 max 503 first 502 last 500
    C: min 499 median 502 max 503 first 502 last 500
    C less B, per period: min +0 median +0 max +0
  mean cross-sectional R-squared: B 0.290903, C 0.292774, difference +0.001871
  factor                    vol B    vol C    mean B    mean C   d mean   noise B   noise C
  Construction             19.30%   19.40%   +24.86%   +23.80%   -1.06%     38.8%     38.3%
  Finance & Real Estate    15.36%   15.33%   +17.96%   +17.05%   -0.91%     31.0%     31.1%
  Manufacturing            10.13%   10.20%   +18.92%   +18.69%   -0.23%     23.6%     23.2%
  Mining                   18.83%   18.60%   +12.39%   +13.34%   +0.95%     18.0%     18.5%
  Retail                   13.23%   13.18%   +12.62%   +13.62%   +0.99%     10.6%     10.6%
  SIC20                    12.15%   12.07%    +8.17%    +9.10%   +0.93%     32.8%     33.0%
  SIC28                    13.14%   13.13%   +15.27%   +14.81%   -0.46%     10.3%     10.3%
  SIC35                    18.87%   18.77%   +28.28%   +28.43%   +0.16%      5.2%      5.3%
  SIC36                    21.16%   21.13%   +30.12%   +31.06%   +0.95%      4.5%      4.5%
  SIC37                    14.37%   14.22%   +13.60%   +13.94%   +0.34%     18.6%     19.2%
  SIC38                    14.78%   14.69%   +15.34%   +15.08%   -0.26%     10.4%     10.7%
  SIC49                    15.86%   15.88%   +21.67%   +21.87%   +0.20%     11.3%     11.2%
  SIC60                    21.79%   21.89%   +19.74%   +19.57%   -0.18%      7.5%      7.4%
  SIC62                    18.77%   18.80%   +17.42%   +16.72%   -0.71%     11.5%     11.5%
  SIC63                    14.51%   14.57%   +18.43%   +17.93%   -0.49%     14.7%     14.5%
  SIC67                    15.00%   15.16%   +15.69%   +15.05%   -0.64%     16.8%     16.5%
  SIC73                    18.23%   18.24%   +14.84%   +14.26%   -0.57%      2.9%      2.9%
  Services                 12.70%   12.76%   +14.44%   +15.53%   +1.09%     28.4%     28.0%
  Transport & Utilities    13.71%   13.56%   +12.84%   +13.72%   +0.89%     15.9%     16.2%
  Wholesale                13.96%   13.89%   +19.57%   +19.72%   +0.16%     46.5%     46.8%
  Volatility                9.03%    9.29%    +2.50%    +3.66%   +1.16%      3.7%      3.6%
  Momentum                  7.25%    7.27%    +4.26%    +4.45%   +0.19%      3.6%      3.6%
  Size                      3.67%    3.70%    +3.13%    +3.23%   +0.10%      7.9%      7.8%
  Size Nonlinearity         2.21%    2.31%    -0.55%    -0.62%   -0.07%     50.1%     47.1%
  Trading Activity          5.80%    6.22%    +4.11%    +2.26%   -1.85%      7.5%      7.6%
  residual spread (1.4826 x MAD; sd): B 0.1362 (0.3418); C 0.1313 (0.3401)
    largest shares of the sd's square, B: FRCB 66.7%, SIVB 3.5%, SMCI 1.9%, LUMN 1.8%, SNDK 0.9%
    largest shares of the sd's square, C: FRCB 67.5%, SIVB 3.6%, SMCI 1.8%, LUMN 1.8%, SBNY 1.0%
    largest shares of V^2, B: MRNA 2.1%, SMCI 1.9%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    largest shares of V^2, C: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: B p10 15.60% median 24.83% p90 40.75% (552); C p10 15.68% median 24.79% p90 40.72% (552)
  specific risk, the 552 in both: B p10 15.60% median 24.83% p90 40.75% (552); C p10 15.68% median 24.79% p90 40.72% (552)
    of those in both, the largest rises: UAA +1.40%, OKTA +1.07%, PENN +0.95%, BE +0.84%, NFLX +0.62%, TWLO +0.57%, LKQ +0.49%, NWS +0.48%
    the largest falls: TPL -1.13%, ORLY -1.32%, LUMN -1.37%, VNO -1.68%, MBC -1.74%, AMTM -2.04%, SBNY -2.40%, IPGP -2.84%
  fallback forecasts: B 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
                      C 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
    in C not B: none; in B not C: none
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    B -3.139 -1.810 risk 11.33%; C -3.139 -1.810 risk 11.77%
    largest 50, equally weighted     B +0.170 +0.052 risk 0.61%; C +0.170 +0.052 risk 0.62%
    the index, cap-weighted          B +0.001 -0.001 risk 0.01%; C +0.001 -0.001 risk 0.01%
  regression universe: B 567, C 567; in B not C: none
    in C not B (0): 
====================================================================================================
Step 3. dividends: the new panel, price returns against total returns: C -> D
  C: data/external/polygon/sp500_securities_2022-01-03_2026-11-03.npz --industries --membership --returns price --volumes traded
  D: data/external/polygon/sp500_securities_2022-01-03_2026-11-03.npz --industries --membership --returns total --volumes traded
  estimated periods: 938 and 938, from 2023-02-08 and 2023-02-08; the same dates: True
  securities per period in the regression
    C: min 499 median 502 max 503 first 502 last 500
    D: min 499 median 502 max 503 first 502 last 500
    D less C, per period: min +0 median +0 max +0
  mean cross-sectional R-squared: C 0.292774, D 0.293189, difference +0.000415
  factor                    vol C    vol D    mean C    mean D   d mean   noise C   noise D
  Construction             19.40%   19.43%   +23.80%   +24.77%   +0.97%     38.3%     38.1%
  Finance & Real Estate    15.33%   15.35%   +17.05%   +18.11%   +1.06%     31.1%     30.9%
  Manufacturing            10.20%   10.20%   +18.69%   +20.90%   +2.20%     23.2%     23.2%
  Mining                   18.60%   18.63%   +13.34%   +14.96%   +1.62%     18.5%     18.5%
  Retail                   13.18%   13.14%   +13.62%   +14.73%   +1.11%     10.6%     10.7%
  SIC20                    12.07%   11.98%    +9.10%   +11.04%   +1.94%     33.0%     33.5%
  SIC28                    13.13%   13.14%   +14.81%   +16.64%   +1.83%     10.3%     10.2%
  SIC35                    18.77%   18.81%   +28.43%   +29.69%   +1.26%      5.3%      5.2%
  SIC36                    21.13%   21.13%   +31.06%   +32.58%   +1.51%      4.5%      4.5%
  SIC37                    14.22%   14.24%   +13.94%   +15.43%   +1.49%     19.2%     19.1%
  SIC38                    14.69%   14.70%   +15.08%   +15.82%   +0.73%     10.7%     10.6%
  SIC49                    15.88%   15.82%   +21.87%   +24.01%   +2.13%     11.2%     11.3%
  SIC60                    21.89%   21.86%   +19.57%   +21.96%   +2.39%      7.4%      7.4%
  SIC62                    18.80%   18.75%   +16.72%   +18.62%   +1.90%     11.5%     11.5%
  SIC63                    14.57%   14.57%   +17.93%   +19.16%   +1.23%     14.5%     14.5%
  SIC67                    15.16%   15.02%   +15.05%   +18.00%   +2.95%     16.5%     16.7%
  SIC73                    18.24%   18.15%   +14.26%   +15.18%   +0.91%      2.9%      3.0%
  Services                 12.76%   12.74%   +15.53%   +16.18%   +0.65%     28.0%     27.9%
  Transport & Utilities    13.56%   13.52%   +13.72%   +15.39%   +1.66%     16.2%     16.3%
  Wholesale                13.89%   13.97%   +19.72%   +20.70%   +0.98%     46.8%     46.2%
  Volatility                9.29%    9.37%    +3.66%    +3.08%   -0.58%      3.6%      3.5%
  Momentum                  7.27%    7.21%    +4.45%    +4.23%   -0.22%      3.6%      3.6%
  Size                      3.70%    3.72%    +3.23%    +3.14%   -0.09%      7.8%      7.7%
  Size Nonlinearity         2.31%    2.31%    -0.62%    -0.51%   +0.11%     47.1%     47.0%
  Trading Activity          6.22%    6.21%    +2.26%    +2.48%   +0.22%      7.6%      7.6%
  residual spread (1.4826 x MAD; sd): C 0.1313 (0.3401); D 0.1330 (0.3405)
    largest shares of the sd's square, C: FRCB 67.5%, SIVB 3.6%, SMCI 1.8%, LUMN 1.8%, SBNY 1.0%
    largest shares of the sd's square, D: FRCB 67.5%, SIVB 3.6%, LUMN 1.8%, SMCI 1.8%, SBNY 1.0%
    largest shares of V^2, C: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
    largest shares of V^2, D: MRNA 2.1%, SMCI 1.8%, FRCB 1.7%, FSLR 1.1%, ENPH 1.0%
  specific risk, every security with a forecast: C p10 15.68% median 24.79% p90 40.72% (552); D p10 15.59% median 24.80% p90 40.78% (552)
  specific risk, the 552 in both: C p10 15.68% median 24.79% p90 40.72% (552); D p10 15.59% median 24.80% p90 40.78% (552)
    of those in both, the largest rises: INTU +0.25%, ZION +0.25%, LUMN +0.22%, NWL +0.20%, IPGP +0.16%, ARES +0.08%, CHTR +0.08%, CPRT +0.08%
    the largest falls: UPS -0.30%, GAP -0.31%, OKE -0.32%, PGR -0.34%, PFE -0.37%, VZ -0.44%, TDG -0.50%, MO -0.58%
  fallback forecasts: C 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
                      D 11: AMTM, FBIN, GAP, IPGP, MBC, PENN, PVH, SOLS, UA, UAA, VNO
    in D not C: none; in C not D: none
  the size pair on the forecast date (Size, Size Nonlinearity):
    smallest 85, equally weighted    C -3.139 -1.810 risk 11.77%; D -3.139 -1.810 risk 11.82%
    largest 50, equally weighted     C +0.170 +0.052 risk 0.62%; D +0.170 +0.052 risk 0.62%
    the index, cap-weighted          C +0.001 -0.001 risk 0.01%; D +0.001 -0.001 risk 0.01%
  regression universe: C 567, D 567; in C not D: none
    in D not C (0): 
  the dividends step, paired per period: each factor's mean return shift, annualised, with its standard error
    Construction           +0.972%  (se 0.146%; the daily shift's sd 1.78e-04)
    Finance & Real Estate  +1.061%  (se 0.146%; the daily shift's sd 1.77e-04)
    Manufacturing          +2.203%  (se 0.253%; the daily shift's sd 3.08e-04)
    Mining                 +1.623%  (se 0.269%; the daily shift's sd 3.26e-04)
    Retail                 +1.113%  (se 0.098%; the daily shift's sd 1.20e-04)
    SIC20                  +1.939%  (se 0.272%; the daily shift's sd 3.31e-04)
    SIC28                  +1.827%  (se 0.181%; the daily shift's sd 2.20e-04)
    SIC35                  +1.260%  (se 0.097%; the daily shift's sd 1.18e-04)
    SIC36                  +1.513%  (se 0.102%; the daily shift's sd 1.24e-04)
    SIC37                  +1.494%  (se 0.248%; the daily shift's sd 3.02e-04)
    SIC38                  +0.733%  (se 0.115%; the daily shift's sd 1.40e-04)
    SIC49                  +2.134%  (se 0.224%; the daily shift's sd 2.73e-04)
    SIC60                  +2.391%  (se 0.288%; the daily shift's sd 3.50e-04)
    SIC62                  +1.900%  (se 0.208%; the daily shift's sd 2.53e-04)
    SIC63                  +1.231%  (se 0.166%; the daily shift's sd 2.02e-04)
    SIC67                  +2.948%  (se 0.361%; the daily shift's sd 4.39e-04)
    SIC73                  +0.915%  (se 0.104%; the daily shift's sd 1.26e-04)
    Services               +0.650%  (se 0.135%; the daily shift's sd 1.65e-04)
    Transport & Utilities  +1.663%  (se 0.253%; the daily shift's sd 3.08e-04)
    Wholesale              +0.977%  (se 0.149%; the daily shift's sd 1.81e-04)
    Volatility             -0.575%  (se 0.114%; the daily shift's sd 1.38e-04)
    Momentum               -0.219%  (se 0.108%; the daily shift's sd 1.31e-04)
    Size                   -0.089%  (se 0.034%; the daily shift's sd 4.16e-05)
    Size Nonlinearity      +0.115%  (se 0.041%; the daily shift's sd 4.95e-05)
    Trading Activity       +0.223%  (se 0.059%; the daily shift's sd 7.21e-05)
```

### extras.txt -- the dividends step against what the members paid; the industries old and new; the single bad samples

```text
(a) the dividends step: each industry's factor-mean shift against its members' distributions, sqrt-cap weighted over each period's members with a return, annualised
  industry               factor shift  members paid  members
  Construction                +0.972%       +0.673%        8
  Finance & Real Estate       +1.061%       +1.266%       11
  Manufacturing               +2.203%       +2.770%       40
  Mining                      +1.623%       +2.173%       13
  Retail                      +1.113%       +1.367%       34
  SIC20                       +1.939%       +2.986%       20
  SIC28                       +1.827%       +2.354%       38
  SIC35                       +1.260%       +1.255%       33
  SIC36                       +1.513%       +0.942%       36
  SIC37                       +1.494%       +1.682%       15
  SIC38                       +0.733%       +0.829%       43
  SIC49                       +2.134%       +3.168%       37
  SIC60                       +2.391%       +3.162%       18
  SIC62                       +1.900%       +2.334%       20
  SIC63                       +1.231%       +1.695%       25
  SIC67                       +2.948%       +3.891%       29
  SIC73                       +0.915%       +0.645%       66
  Services                    +0.650%       +0.707%       21
  Transport & Utilities       +1.663%       +2.243%       29
  Wholesale                   +0.977%       +1.431%       12
  correlation of the two over the 20 industries: 0.945; the shift less the paid, mean -0.351%, largest 1.046%

(b) the industry each security is in, old run against new, over the 465 with one in both: 14 differ
  AMP    Finance & Real Estate -> SIC62
  BEN    Finance & Real Estate -> SIC62
  BLK    Finance & Real Estate -> SIC62
  CBOE   Finance & Real Estate -> SIC62
  CME    Finance & Real Estate -> SIC62
  GS     Finance & Real Estate -> SIC62
  ICE    Finance & Real Estate -> SIC62
  IVZ    Finance & Real Estate -> SIC62
  MKTX   Finance & Real Estate -> SIC62
  MS     Finance & Real Estate -> SIC62
  NDAQ   Finance & Real Estate -> SIC62
  RJF    Finance & Real Estate -> SIC62
  SCHW   Finance & Real Estate -> SIC62
  TROW   Finance & Real Estate -> SIC62
  with an industry in the old run and none in the new: INFO

(c) moves beyond 10% that the next one reverses within 10%: a quarter's sample Polygon got wrong, as the new panel carries it
  [one sample]  CEG    2026-01-27..2026-04-28: 312,405,579 -> 362,355,476 (1.160) -> 313,309,685; 63 member-days
  [one sample]  CEG    2026-04-28..2026-07-29: 362,355,476 -> 313,309,685 (0.865) -> 359,102,017; 63 member-days
  [one sample]  DGX    2025-04-09..2025-07-11: 111,620,000 -> 11,610,000 (0.104) -> 111,635,472; 63 member-days; the old panel carried 111,620,000..111,620,000, 1.000..1.000 of the count before
  [one sample]  EG     2025-04-09..2025-07-11: 42,980,000 -> 47,790,000 (1.112) -> 42,534,728; 63 member-days; the old panel carried 42,980,000..42,980,000, 1.000..1.000 of the count before
  [one sample]  NWS    2024-07-09..2024-10-07: 191,100,000 -> 162,170,000 (0.849) -> 190,080,000; 63 member-days; the old panel carried 191,390,000..191,390,000, 1.002..1.002 of the count before
  [one sample]  PCG    2022-04-04..2022-07-06: 1,985,370,000 -> 2,465,130,000 (1.242) -> 1,987,480,000; 0 member-days
  [two counts alternating]  PCG    2022-07-06..2025-07-11: 2,465,130,000 -> 1,987,480,000 (0.806) -> 2,675,428,603; 694 member-days
  [two counts alternating]  PCG    2025-07-11..2026-01-09: 2,193,580,000 -> 2,675,428,603 (1.220) -> 2,197,910,425; 126 member-days
  [one sample]  PCG    2026-01-09..2026-04-13: 2,675,580,992 -> 2,197,910,425 (0.821) -> 2,675,711,544; 63 member-days
  [one sample]  RDDT   2025-12-22..2026-03-25: 134,585,853 -> 51,670,930 (0.384) -> 139,649,508; 0 member-days
  [one sample]  SBUX   2026-07-14..2026-10-12: 1,139,300,000 -> 139,700,000 (0.123) -> 1,140,000,000; 63 member-days; the old panel carried 1,137,100,000..1,137,100,000, 0.998..0.998 of the count before
  [one sample]  XYZ    2026-01-09..2026-04-13: 549,574,000 -> 59,998,000 (0.109) -> 539,103,000; 63 member-days
  member-days on one wrong sample: 504
```

### regressions.txt -- every identity build re-run offline on the final code, against data/'s

```text
plain:
  identical: sp500_identity_2022-01-03_2026-11-03.csv
  identical: sp500_identity_2022-01-03_2026-11-03_members.csv
  identical: sp500_identity_2022-01-03_2026-11-03_fetch.csv
  its JSON: the same keys False; differing ['resolutions']
    resolutions: data None
    resolutions: now  []
resolved:
  identical: sp500_identity_2022-01-03_2026-11-03_resolved.csv
  identical: sp500_identity_2022-01-03_2026-11-03_resolved_members.csv
  identical: sp500_identity_2022-01-03_2026-11-03_resolved_fetch.csv
  its JSON: the same keys True; differing ['resolutions']
    resolutions: data [{'statement': 'CDAY for DAY', 'member': 'DAY', 'named': ['2024-02-01', '2026-02-06', 506], 'confirmed': True, 'covered': 506, 'verdict': "confirmed: the endpoi
    resolutions: now  [{'statement': 'CDAY for DAY', 'member': 'DAY', 'named': ['2024-02-01', '2026-02-06', 506], 'confirmed': True, 'covered': 506, 'verdict': "confirmed: the endpoi
corrected:
  identical: sp500_identity_2022-01-03_2026-11-03_corrected.csv
  identical: sp500_identity_2022-01-03_2026-11-03_corrected_members.csv
  identical: sp500_identity_2022-01-03_2026-11-03_corrected_fetch.csv
  identical: sp500_identity_2022-01-03_2026-11-03_corrected_links.csv
  identical: sp500_identity_2022-01-03_2026-11-03_corrected_membership.csv
  its JSON: the same keys True; differing []
amended:
  identical: sp500_identity_2022-01-03_2026-11-03_amended.csv
  identical: sp500_identity_2022-01-03_2026-11-03_amended_members.csv
  identical: sp500_identity_2022-01-03_2026-11-03_amended_fetch.csv
  identical: sp500_identity_2022-01-03_2026-11-03_amended_links.csv
  identical: sp500_identity_2022-01-03_2026-11-03_amended_membership.csv
  its JSON: the same keys True; differing []
```

### regress_gate_panel.txt -- the gate on copies of both plans, and the panel, re-run offline

```text
the gate on a copy of the amended plan, against data/'s amended report: differing ['plan']; passed True then, True now
  requests 0, failures 0
the gate on a copy of the corrected plan, against data/'s corrected report: differing ['identity', 'passed', 'plan', 'requests']; passed False then, True now
  COHR 2022-09-08: passed False -> True; reconciled None -> BBG001S5S6N6
    now: COHR on 2022-09-08 names Coherent Corp. Common Stock (BBG000GS6QB4), reconciled to Coherent Corp. (BBG000BLW102) by its share-class FIGI, BBG001S5S6N6, which the expected security's own records carry
  splits, distributions, requests and failures unchanged: False
the panel re-built offline against data/'s: 20 of 20 arrays identical
  its report: differing keys []
```

### pytest_all.txt -- the full suite (last line)

```text
=========== 2290 passed, 1 skipped, 19 warnings in 63.40s (0:01:03) ===========
```

### pytest_mid.txt -- the midoffice tests (last line)

```text
======================= 839 passed, 1 skipped in 22.65s =======================
```

### ruff_final.txt -- ruff over every changed Python file

```text
scripts\run_sp500_model.py:39:5: F401 [*] `praxis.midoffice.model.covariance.HorizonCovariance` imported but unused
Found 1 error.
[*] 1 fixable with the `--fix` option.
```

### hygiene_final.txt -- parse, ASCII and line endings, by bytes

```text
src/praxis/midoffice/refdata/securities.py: CRLF (685 lines), parses, ASCII
src/praxis/midoffice/refdata/rebuild.py: CRLF (600 lines), parses, ASCII
src/praxis/midoffice/refdata/identity.py: CRLF (1840 lines), parses, ASCII
scripts/build_sp500_securities.py: CRLF (484 lines), parses, ASCII
scripts/build_sp500_identity.py: CRLF (768 lines), parses, ASCII
scripts/fetch_sp500_rebuild.py: CRLF (286 lines), parses, ASCII
scripts/run_sp500_model.py: CRLF (1196 lines), parses, ASCII
tests/midoffice/test_securities.py: CRLF (222 lines), parses, ASCII
tests/midoffice/test_rebuild.py: CRLF (350 lines), parses, ASCII
tests/midoffice/test_identity.py: CRLF (593 lines), parses, ASCII
docs/FACTOR_MODEL.md: CRLF (993 lines), -, ASCII
claude/handoffs/BRIEF_midoffice08_build.md: LF (60 lines), -, ASCII
all ok
```

## git add (always last -- Rule 43)

This piece -- ruling 1's rule and row, the first-count search, the panel on securities and its build, the run's new panel, the docs, the tests, the brief, this retro, and the state file in the same commit:

```
git add src/praxis/midoffice/refdata/rebuild.py src/praxis/midoffice/refdata/identity.py src/praxis/midoffice/refdata/securities.py scripts/build_sp500_identity.py scripts/fetch_sp500_rebuild.py scripts/build_sp500_securities.py scripts/run_sp500_model.py tests/midoffice/test_securities.py tests/midoffice/test_rebuild.py tests/midoffice/test_identity.py docs/FACTOR_MODEL.md claude/MIDOFFICE_STATE.md claude/handoffs/BRIEF_midoffice08_build.md claude/retros/RETRO_midoffice08_build.md
```
