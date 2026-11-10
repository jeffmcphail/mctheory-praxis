# Retro: Mid-office 07 -- the rest of the rebuild, stopped at its gate again

**Brief:** `claude/handoffs/BRIEF_midoffice07_rebuild.md` -- written in Chat and pasted into the
Code session, transcribed verbatim
**Date:** 2026-11-10
**Duration:** the brief arrived at 09:14 EST (14:14:57Z), just after Jeff's commit of midoffice06
(208ca80, 09:14:20). The corrections, the regenerated plan and the gate ran to about 09:50; the
tests, the diagnoses and this retro to about 10:10.
**Status:** STOPPED AT THE GATE, as the brief directs. Item 1 is done: every correction made, the
plan regenerated, the gate run again with the lag rule and the price-gap test. It stops on one
span: COHR's first share sample names a FIGI that the correction Jeff approved was to account
for, and the endpoint refutes the account. Every other identity check and every split passes.
Nothing is built, so items 2 to 5 wait, the docs among them ("with the build"). Uncommitted; one
`git add` at the end.

---

## Summary

- **The corrections, all made, in 21 requests** (the estimate was about 27). Bunge's boundary
  falls between 2023-11-07 and 11-08 by its CIK, already bracketed by item 1's bisection, so it
  cost nothing. Vylor's two FIGIs part between 2026-10-02 and 10-05, in 4 lookups. II-VI's IIVI
  history, 2022-01-03 to 2022-09-07, joins Coherent Corp., confirmed at both ends with its prices
  unbroken. 13 spans now start at their first price: the 11 spin-offs, and Reddit's and Kenvue's
  by a day each (the rule is general). S600, CEGV and CRHC are gone. No member-day moved but by
  the corrections' own boundaries.
- **The table is built from its own evidence still.** The rebuild's 9,105 share samples now sit
  in the cache beside the panel fetch's records, so the table reads records on the panel fetch's
  grid alone: midoffice05's and midoffice06's tables rebuild byte for byte, and the samples stay
  the gate's evidence, not the table's.
- **Membership read through the renames** for the plan: 610,842 member-days, EchoStar's 1,057
  phantom days gone. The plan: 600 securities, 628 spans, 617 tickers; each security's sampling
  grid kept, so the samples already fetched still count.
- **The gate stops on COHR alone. Identity: 627 of 628 pass.** EG passes by reading past its
  identifier-less record.
  - **COHR fails.** Its change-day record of 2022-09-08 names BBG000GS6QB4. The approved
    correction was to account for it as old Coherent's FIGI once a COHR record from before
    2022-07-01 showed that. The records refute it: Coherent, Inc. was BBG000BG1DH3.
  - **What the record does carry: Coherent Corp.'s share-class FIGI**, BBG001S5S6N6, on every
    II-VI and Coherent Corp. record. So do the six last-day records naming another composite
    FIGI (EA, OKE, JNPR, BLK, XOM, DISCA): each carries its expected security's share class.
    Vylor's first record is the one with another share class, and Jeff ruled it a
    reorganisation. Proposed: read the share-class FIGI. A decision below.
- **Splits: 55 events, no contradiction.** The lag rule and the price-gap test settle all 11 of
  midoffice06's.
  - **47 splits:**
    - 45 the counts follow and the price confirms;
    - MNST's and APH's rest on their prices, which opened at 0.507 and 0.505 of the close before,
      against 0.500.
  - **Every split's raw price gaps by its ratio:** the median deviation is 0.81%, the largest
    3.58% (DXCM).
  - **8 price-only:**
    - 6 tested, their prices gapping too;
    - LEN's and HON's untested, as before.
- **Found for the build:**
  - 696 member-days of 610,842 whose records carry no share count to carry. They are spin-offs'
    first quarters, and all of FedEx Freight's and Vylor's.
  - The old panel's volumes are split-adjusted beside point-in-time counts and traded prices.
    24,386 member-day cells over 39 tickers read their turnover and dollar volume by the
    adjustment factor, from 0.2 to 50. Polygon scales volume for spin-offs too.
  - Both are decisions below.
- **Tests:** 2,279 passed, 1 skipped (midoffice: 828 passed, 1 skipped); 10 new, and 3 rewritten
  for the new split rule.

---

## What was asked

The brief is in `claude/handoffs/BRIEF_midoffice07_rebuild.md`, verbatim, the paste wrapper
stripped. In outline:

- **"committed -- midoffice06, stopped at its gate."** Checked: 208ca80 in `git log`.
- **Rulings on midoffice06's seven decisions:**
  1. The identity corrections all as proposed: spin-off spans start at their first price; an
     identifier-less record is no evidence of another security; Bunge and Vylor are
     reorganisations, dated by the identifier the table keys on; II-VI's IIVI history joins
     Coherent Corp.; S600, CEGV and CRHC dropped; CRH's ADR history before 2023-09-25 stays out.
  2. The split lag rule as proposed, and a third test: "confirm every split by its raw price gap
     on the execution date ... within the same tolerance of the ratio's reciprocal ... A split
     whose raw price does not gap stops the gate."
  3. WBD accepted as its own security.
  4. Carnival: carry the predecessor's count until the successor's first sample. OKE and XOM:
     carry the predecessor's count 1:1 and flag it.
  5. TEL: keep one of the pair.
  6. LEN 2025-01-21: leave it, flagged.
  7. `docs/FACTOR_MODEL.md`'s split paragraph rewritten "with the build", with 4,196 member-days
     over 37 splits.
- **midoffice07:**
  1. Make the corrections ("the 27 or so requests"), regenerate the plan, run the gate again with
     the lag rule and the price-gap test. "The same stop conditions as before, plus the price-gap
     one."
  2. "If it passes, build the new panel": securities as columns, prices stitched by the table,
     splices separated, reorganisations continuous, counts per span with splits on their
     execution dates, membership through the renames mapped to securities, sp500's
     current_stale warnings printed; the identity and split guards on the built panel.
  3. Total returns: each distribution on the prices' split basis by the ex-date's ratio of
     adjusted to raw close; one of TEL's pair; SPY's too.
  4. The model in two steps: old panel against new, both on price returns; then the new panel's
     price returns against its total returns. Report midoffice06's item 6, and the factor and
     industry means for the dividends step.
  5. The docs: the split paragraph, and a section on the identity table and the rebuild.
- **"Same discipline -- retro and git add lines at the end, then stop."**

---

## Files modified

| File | Change | Lines |
|---|---|---|
| `src/praxis/midoffice/refdata/identity.py` | the module docstring's account of the rebuild's evidence; `Record.share_class` and `Record.identified`; `read_records(on=...)`; `segment_ticker` reads past a record with no identifier; `Link.kind`; `PlanRow.previous` and `fetch_plan(origins=...)`, the grid through each security's origin both ways; new: `read_answers`, `Correction`, `start_at_first_price`, `drop_refuted`, `Reorganisation` and `reorganise`, `Confirmation` and `confirm_span` | 46-54, 61, 105-119, 189-219, 382-387, 1198-1251, 1327-1330, 1567-1830 |
| `src/praxis/midoffice/refdata/rebuild.py` | the docstring's gate; `Span.previous`; `record_of` keeps the share class; `IdentityCheck.skipped` and `check_identities`' two rules, each failure's share class reported; `SplitCheck`'s new fields, `classify` with the lag rule and the price-gap test, `check_splits(gaps=...)`, `raw_gap` | 18-30, 43-90, 194-526 (hunks) |
| `scripts/build_sp500_identity.py` | `--correct`, `--panel-samples-every`, `--bars-start`, `--bars-end`; `REORGANISATIONS`, `CONFIRMATIONS`; the corrections and their report; refused corrections reported, not fatal; the records read on the panel's grid; membership through the renames for the plan, its guard; the plan's origins and `previous`; the `_links` and `_membership` files; the JSON's corrections, links and refusals | 20-46, 91-112, 157-188, 213-218, 286-388, 479-507, 520-568, 602-612, 630-736 (hunks) |
| `scripts/fetch_sp500_rebuild.py` | the docstring; the corrected plan by default; each split's raw gap from the cache; the gate's text; the report named after its plan; the JSON's skips, stale samples, followed sample, gap and price verdict | 8-23, 46, 181-189, 199-202, 219-223, 232-233, 259-264 |
| `tests/midoffice/test_rebuild.py` | 4 new tests, 3 rewritten for the new rule, the plan's reader extended; `FakePolygon` takes bars | 2-10, 25-43, 105-207, 223, 253-315 (hunks) |
| `tests/midoffice/test_identity.py` | 6 new tests and a helper | 425-568 |
| `claude/handoffs/BRIEF_midoffice07_rebuild.md` | new: the brief, verbatim | new, 84 lines |
| `claude/retros/RETRO_midoffice07_rebuild.md` | new: this retro | new |

Written outside the tracked files, all gitignored and all new in this piece -- nothing that was
in `data/` before it was overwritten or deleted. midoffice06's gate report is untouched: this
piece's carries its plan's suffix, `_corrected`.

- `data/external/polygon/sp500_identity_2022-01-03_2026-11-03_corrected.csv`, `_corrected.json`,
  `_corrected_members.csv`, `_corrected_fetch.csv`, `_corrected_links.csv`,
  `_corrected_membership.csv` -- the corrected table and plan, and the two new files.
- `data/external/polygon/sp500_rebuild_gate_2022-01-03_2026-11-03_corrected.json` -- the gate's
  report on the corrected plan.
- In `data/external/polygon/cache/`: 21 answers, one for each request -- IIVI's prices adjusted
  and raw, split events and distributions; 4 IIVI records, 2 COHR, 4 VYLR; and 7 first-day
  samples (VLTO, MBC, AMTM, SNDK, FDXF, HONA, RDDT).
- `claude/scratch/midoffice07/` -- the scratch, every script and output named here.

---

## Item 1: the corrections

### The table keeps its own evidence

The table is built from the records the panel fetch cached, at its share samples. The
midoffice06 fetch cached 9,105 more beside them, under the same names, so read as before the
records under the panel's tickers would have grown from 2,885 at 6 dates to 10,136 at 128, and
the table would have changed in ways nobody ruled on. Worse, the gate's evidence -- those samples
-- would have become the table's. So `read_records` now takes the dates to read
(`--panel-samples-every 252`, the panel's grid: its counts change only on 2023-01-04,
2024-01-05, 2025-01-07, 2026-01-09 and 2026-11-03). Checked: midoffice05's build, and
midoffice06's with `--resolve`, rebuilt offline into scratch, write their CSVs byte for byte
(`regressions.txt`); midoffice06's JSON differs only in the link evidence its own retro said the
code had reworded since.

The corrections then take the rebuild's evidence to the table, as `--correct`, writing
`_corrected` files beside the rest. Each is a rule or a stated hypothesis, and each correction is
recorded with why (`identity.Correction`):

| Rule | What it does | Where it applied |
|---|---|---|
| first price | a span the panel's prices did not draw starts at its first price under its ticker | the 11 spin-offs; RDDT 2024-03-20 to 03-21 and KVUE 2023-05-03 to 05-04 (each listed the day before it first traded) |
| unpriced | a span with no price under its ticker goes | S600, CEGV |
| refuted | a span every record under its ticker names another identified security in, and none its own, goes; its tenure too | CRHC: 4 records Cohn Robbins (CIK 0001818212), 4 no record. The rename CRHC to CRH goes with it: 64 renames found, 51 read through (65 and 52 before) |
| reorganisation | a new identifier under one ticker, its boundary bisected on the endpoint's answers, the two linked as one security | `REORGANISATIONS`: Bunge, Vylor |
| confirmed | a span the endpoint names at both ends, the prices unbroken between, joins its security | `CONFIRMATIONS`: IIVI; COHR before II-VI refused |
| (no identifier) | a record with neither a FIGI nor a CIK is no anchor in `segment_ticker` and no evidence in the gate | none among the panel's records (`rule_reach.txt`); EG's 2023-07-10 the only one in the whole cache |

How far each general rule reaches was measured before it ran (`span_prices.txt`,
`rule_reach.txt`). "First price" catches 13 spans, the 11 and RDDT and KVUE. "Unpriced" catches
2, "refuted" 1, and the identifier-less rule 0 of the table's records.

**The corrections' effect, span by span** (`corrected_run2.txt`, `plan_diff.txt`):

| Correction | Before | After | Member-days |
|---|---|---|---|
| Bunge, by its CIK | Bunge Limited BG to 2023-10-31; Bunge Global from 2023-11-01 | Bunge Limited to 2023-11-07; Bunge Global from 2023-11-08 | 160/754 to 165/749 |
| Vylor's two FIGIs | BBG024CQR6X9 VYLR from 2026-10-01 | BBG024W5VJC3 2026-10-01 to 10-02; BBG024CQR6X9 from 10-05 | 24 to 2 + 22 |
| II-VI's history | -- | IIVI 2022-01-03 to 2022-09-07, BBG000BLW102 (Ii-Vi Inc, CIK 0000820318, listed 1987-10-02) | 0 (not a member then) |
| first price | 13 spans from their first event (KVUE's from its first record) | from their first price, 1 to 13 trading days later | unchanged on every one |
| dropped | S600, CEGV, CRHC | -- | 0 on each |

**Bunge cost nothing.** Item 1's bisection had already asked BG's records around the change:
CIK 0001144519 on 2023-11-01 and 11-07, under the new name Bunge Global SA; CIK 0001996862 from
2023-11-08. The bracket was adjacent days, so `reorganise` asked nothing. The share count either
side is 145,290,000.

**Vylor: 4 lookups** (2026-10-16, 10-08, 10-05, 10-02), between its record of 2026-10-01
(BBG024W5VJC3) and 2026-11-03 (BBG024CQR6X9). The first FIGI held two trading days.

**The member map, read through the renames, for the plan.** 610,842 member-days mapped, 0
named; membership as given maps 611,899. The difference is exactly EchoStar's 1,057 phantom ECHO
days from 2022-01-03 to 2026-03-20, before it joined as SATS. Both guards pass.

**The table:** 633 rows, 604 securities (midoffice06: 634, 603); the four guards pass, and the
fifth, on the renamed membership.

**The regenerated plan:** 600 securities, 628 spans, 617 tickers, 11,950 samples (midoffice06:
599, 629, 619, 11,968). A span moved to its first price keeps its security's sampling grid:
`fetch_plan` runs the grid through the security's first day in the uncorrected table, both ways.
Coherent Corp.'s grid reaches back into its new IIVI span (2022-03-09, 2022-06-08). Each span now
also names the security the table has under its ticker before it (`previous`): 11 do, the
reorganisations and the splices (BNY, DOC, META).

### The requests: 21

The builder made 9: Vylor's 4 lookups; IIVI on 2022-01-03 and 2022-09-07; COHR on 2022-01-03 and
2022-07-01; and IIVI's adjusted prices, which `confirm_span` needs. The fetch made 12: IIVI's raw
prices, split events and distributions; IIVI's grid samples, 2022-03-09 and 2022-06-08; and the
first-day records of VLTO, MBC, AMTM, SNDK, FDXF, HONA and RDDT. Against the estimate of about 27:

- Bunge's estimated 6 were already on disk;
- Vylor's 5 were 4;
- II-VI and Coherent's 10 were 10, its events not needed;
- RDDT's 1 is the first-price rule reaching past the spin-offs;
- KVUE's first-day record was cached.

None failed.

---

## The gate again

### Identity: 627 of 628 spans pass

Every corrected span passes at its first sample with an identifier (`gate_run1.txt`):

- the 13 moved spans at their first priced day;
- Bunge's two and Vylor's two at their boundaries' days;
- IIVI on 2022-01-03;
- EG at 2023-10-05, reading past 2023-07-10's record, which has a name and no FIGI or CIK. It is
  the only sample read past in the gate.

The previous-holder rule, approved for COHR, fires nowhere.

### COHR: the hypothesis refuted, and what the record does carry

**The span.** COHR from 2022-09-08, 157 member-days. The table expects BBG000BLW102, Coherent
Corp. (II-VI Incorporated renamed), CIK 0000820318.

**The record.** Its first sample, the change day 2022-09-08, names composite FIGI BBG000GS6QB4,
"Coherent Corp. Common Stock", with no CIK, no list date and no share count.

**The approved account, refuted.** The correction Jeff approved was to read the span at its
second sample, "the change-day record accounted for once a COHR record from before 2022-07-01
shows BBG000GS6QB4 to be old Coherent's". The record was fetched -- COHR on 2022-01-03 and
2022-07-01 -- and it shows otherwise. Coherent, Inc. was **BBG000BG1DH3**, CIK 0000021510, listed
1990-03-26 (`CONFIRMATIONS`' second, refused, in `corrected_run2.txt`).

So BBG000GS6QB4 is neither old Coherent's composite FIGI nor Coherent Corp.'s. No other answer
in the cache names it.

**The share-class FIGI** (`share_class.txt`) does name the security:

| Record | Composite FIGI | Share-class FIGI |
|---|---|---|
| COHR 2022-09-08, the change day | BBG000GS6QB4 | BBG001S5S6N6 |
| IIVI 2022-01-03 and 2022-09-07; COHR from 2022-12-07, every one | BBG000BLW102 | BBG001S5S6N6 |
| COHR 2022-01-03 and 2022-07-01, Coherent, Inc. | BBG000BG1DH3 | BBG001S5Q0Q8 |

The change-day record names Coherent Corp.'s share class under a composite FIGI seen nowhere
else.

**The same pattern elsewhere.** It is not COHR's alone. Every sample in the plan whose composite
FIGI is not the one the table expects (midoffice06's samples, read from the cache):

| Span | Sample | Names | Its share class | The expected security's |
|---|---|---|---|---|
| COHR | 2022-09-08 | BBG000GS6QB4 | BBG001S5S6N6 | BBG001S5S6N6 |
| EA | 2026-08-04 | BBG000K1X329 | BBG001S5T5C9 | BBG001S5T5C9 |
| OKE | 2026-09-10 | BBG000JBSHW1 | BBG001S5TWK1 | BBG001S5TWK1 |
| JNPR | 2025-07-01 | BBG000BWS318 | BBG001SCTT05 | BBG001SCTT05 |
| BLK | 2024-10-01 | BBG000CN81W9 | BBG001SDBP55 | BBG001SDBP55 |
| XOM | 2026-07-01 | BBG000H5HTR8 | BBG001S69V32 | BBG001S69V32 |
| DISCA | 2022-04-04 | BBG000CHWP52 | BBG001SLZ9C9 | BBG001SLZ9C9 |
| VYLR | 2026-10-01 | BBG024W5VJC3 | BBG024W5VJF0 | BBG024CQR6Z7 |

Seven carry the expected security's share-class FIGI under another composite FIGI. Six of them
-- the last-day records midoffice06 reported and the CIK reconciled -- sit at or near a span's
end, at an acquisition or a reorganisation. COHR's sits at a span's start, with no CIK to reconcile it.
Vylor's is the one with another share class: a change of security, which Jeff ruled a
reorganisation. A FIGI's share-class level is the one every composite FIGI of a share class
shares, so the share-class FIGI is what names the security here.

**Proposed correction:** a record whose composite FIGI the table does not expect, but whose
share-class FIGI is the expected security's (as the security's own records carry it), names the
expected security. COHR's span then passes on its first sample. Nothing else in this gate turns
on it. No fetch. Alternatively, a one-off: read COHR's span at its second sample. Either way, I
would add Coherent, Inc.'s row under COHR, 2022-01-03 to 2022-07-01, BBG000BG1DH3: confirmed at
both ends, its prices unbroken, no member-days, no fetch. The check is coded and reported, not
applied: `check_identities` says, of a failure, whether its share class is the expected
security's.

### Splits: 55 events, no contradiction

Classified by the ratio, the counts with the lag rule, and the raw price on the execution date
(`gate_run1.txt`, `gate_gaps.txt`):

| Kind | Counts | Price | Events |
|---|---|---|---|
| split | follow | gaps by the ratio | 45 |
| split | stale to the span's end | gaps by the ratio | 2: MNST 2026-08-11, APH 2026-09-03 |
| price-only | unchanged | gaps by the ratio | 6 |
| price-only | within 10% of both | within 10% of both | 2: LEN 2025-01-21, HON 2025-10-30, untested |
| contradiction | -- | -- | 0 |

- **Every split's raw price gaps by its ratio.** The open on the execution date over the close
  on the trading day before deviates from 1/ratio by a median 0.81%, at most 3.58% (DXCM's
  four-for-one of 2022-06-13: 0.2410 against 0.2500). The eight price-only adjustments' prices
  gap by theirs too: median 1.39%, at most 3.53% (FTV, 0.7270 against 0.7536); LEN's and HON's,
  their ratios within 10% of one, are within 10% of no gap as well.
- **Thirteen splits' counts lag, and the lag rule reads past each stale sample:**
  - nine of midoffice06's eleven contradictions, by one sample each, as before;
  - MNST and APH, stale to the window's end (979,525,882 and 1,232,983,457 on 2026-11-03), resting
    on their prices: 0.5072 and 0.5049 against 0.5000;
  - two more, now read on their own execution day: WSM's sample of 2024-07-09 and SMCI's of
    2024-10-01 -- the first day of SMCI's second span -- still read the pre-split count.
    midoffice06's check read only the samples strictly after the day, which skipped them.
- **For the build:** a stale sample, the first day of a span among them, must not reset the
  carried count -- SMCI's carried count on 2024-10-01 is its count of 2024-09-30 times ten. The
  check records which samples are stale (`SplitCheck.stale`), and the split guard has them.
- **The first sample on or after the day.** A fractional ratio is read at it now, where before
  it was the first strictly after. One event reads its own day: GE's 2023-01-04, 1,092,670,000,
  0.996 times.

---

## Found on the way, for the build

### Share counts the records do not carry: 696 member-days

On the corrected plan, a panel carrying each security's count forward from its samples -- across
its own spans and a reorganisation, never backward -- has no count on 696 of the 610,842
member-days (`uncapped.txt`). Every one is a security whose first record with a count comes after
its membership starts.

| Security | Member-days without a count | First count | Still without, given weighted counts |
|---|---|---|---|
| FedEx Freight (FDXF) | all 109 | none in the window | 60, until its first weighted count, 2026-08-26 |
| WBD | 63 | 2022-07-13 | 63 |
| Skydance (PSKY) | 63 | 2025-11-05 | 63 |
| GE Vernova, Solventum, Veralto | 60 each | 2024-06-27, 2024-06-26, 2023-12-27 | 60 each |
| Amentum | 59 | 2024-12-23 | 59 |
| Qnity | 58 | 2026-01-28 | 58 |
| Constellation | 53 | 2022-04-20 | 53 |
| GE HealthCare | 51 | 2023-03-20 | 51 |
| Solstice | 36 | 2026-01-21 | 36 |
| Vylor | all 24 | none in the window | 1, until its first weighted count, 2026-10-02 |

**Why.** The first-day records of a spin-off or a new listing carry neither
`share_class_shares_outstanding` nor `weighted_shares_outstanding`. Where the share-class count is
missing later -- FDXF, Vylor, HONA after its first day, OKE's and XOM's successors -- the record
does carry `weighted_shares_outstanding`.

**Those weighted counts test the 1:1 carry Jeff ruled for OKE and XOM:**

- OKE's successor reads 630,450,437 on 2026-11-03, against its predecessor's last 630,369,667
  (1.0001);
- XOM's reads 4,111,911,960 on 2026-10-01, against 4,144,947,162 (0.9920).

**Options for the build** (a decision below):

- leave these days out until a count, as the fill never back-fills;
- search each span's first quarter for its first record with a count, about six requests a span
  and some 70 in all;
- or take the weighted count where the share-class count is missing. `prices.py` chose against
  it as an earnings-per-share denominator, and it would cover only 72 of the 696 days: 49 of
  FDXF's and 23 of Vylor's.

### The old panel's volumes are not the traded ones: 24,386 member-day cells

**The basis.** The old panel stores the adjusted bars' volume
(`build_security_panel`: `"volumes": _series(adjusted, ...)`) beside a point-in-time share count
and the traded price. Before a split, its turnover (volume over shares) and its dollar volume
(volume times traded price, VLVR's numerator) both read the split's ratio too high. NVDA on
2024-06-07 stored 412,263,200 shares traded, against 41,226,320 traded and 2.47 billion
outstanding.

**The measure** (`volume_basis.txt`): 24,386 cells, every one a member-day by the old run's
membership, over 39 tickers. The stored volume over the traded one runs from 0.2 (AMCR's
one-for-five) to 50 (CMG); BKNG's is 25, AMZN's and GOOGL's 20, NVDA's 10.

**Spin-offs too.** Polygon scales volume by its price adjustment for spin-offs as well, whose
share counts do not move: GE 1.25 and 1.61, FTV 1.33, MMM 1.20, LH 1.16, HON 1.06, LEN 1.03, and
DD's 0.80 before both its adjustments. So the adjusted volume is wrong for turnover across every
event, not only splits.

**Proposed for the build:** the new panel stores the traded volume, the one consistent with a
point-in-time count and a traded price. Its effect on the model would be measured as a step of
its own inside the old-against-new comparison: the new panel on adjusted volumes, then on traded
ones.

---

## Key decisions

1. **The table's records restricted to the panel fetch's grid** (`--panel-samples-every 252`),
   so the rebuild's samples check the table rather than build it. Verified by rebuilding
   midoffice05's and 06's tables byte for byte.
2. **"Spin-off spans start at their first price" applied as a rule to every span not drawn from
   the panel's prices.** It moved RDDT and KVUE by a day besides the 11; no member-day moved.
3. **The corrections written as rules and stated specs**, each recorded with its evidence, and a
   spec the endpoint refuses reported, not fatal. COHR's refusal is the gate's finding.
4. **Old Coherent's row not added.** The spec stated the approved hypothesis, `expect`
   BBG000GS6QB4, and the endpoint refused it. Restating the spec after seeing the answer would
   have fitted the hypothesis to the data, so the row waits for Jeff.
5. **The share-class reading coded as a report, not a rule.** Jeff did not approve it, so the
   gate stops on COHR, as the brief's stop conditions require.
6. **The grid keeps its phase** (`fetch_plan(origins=...)`). Re-anchored at the spin-offs' first
   prices, every one of their samples would have moved: about 130 more requests.
7. **The plan's membership read through the renames**, as the panel's will be (EchoStar's
   phantoms gone). The member listing and the plan use it; the findings before it (truncations,
   foreign prices) stay on membership as given, as before.
8. **The lag rule reads samples on the execution date too.** Two splits' samples on that day
   are stale (WSM, SMCI), and the build must see them.
9. **The price-gap test applied to every event**, deciding only splits, as ruled. A price-only
   adjustment's gap is reported; all six tested ones gap.
10. **The corrected plan the rebuild script's default**, and the gate's report named after its
    plan (`_corrected`), so midoffice06's report is untouched.
11. **Stopped at the gate.** No panel, no total returns, no model run, no docs. The docs'
    rewrite was ruled "with the build".

## Decisions left for Jeff

1. **COHR's change-day record.**
   - **(a) The share-class rule.** A record whose composite FIGI the table does not expect, but
     whose share-class FIGI is the expected security's, names the expected security. It accounts
     for all seven such records in the samples.
   - **(b) A one-off.** Read COHR's span at its second sample.
   - **Either way:** add Coherent, Inc.'s row under COHR (BBG000BG1DH3, 2022-01-03 to
     2022-07-01). No fetch for any of this; and since nothing else stops the gate, it would then
     pass.
2. **Share counts the records do not carry**, 696 member-days: leave them out (the default), or
   search for each first count (about 70 requests), or take weighted counts where the share-class
   count is missing.
3. **Volumes:** build the new panel on traded volumes, and measure the basis as its own step in
   the comparison?
4. **Then the build, as the brief has it.**

---

## Tests

**The suite.** 2,279 passed, 1 skipped, 19 warnings, in 87.06 s, with
`--ignore=tests/test_market_data.py` (it looks for `../core_repo/src`, as before). midoffice
alone: 828 passed, 1 skipped. Before this piece: 2,269 and 818.

**New: 10.**

`tests/midoffice/test_identity.py`, 6:

- `test_records_off_the_panels_grid_are_not_read_and_one_with_no_identifier_names_nobody` --
  `read_records(on=...)`; `Record.identified`; an identifier-less record between two of one
  security is no anchor, no boundary.
- `test_a_span_starts_at_its_first_price_and_one_with_none_is_dropped` -- the first-price and
  unpriced rules; the panel's rows and unfetched tickers untouched.
- `test_a_span_every_record_under_its_ticker_names_another_security_in_is_refuted` -- CRHC's
  case; a span with a record of its own, and one whose records name nobody, stay.
- `test_a_reorganisation_is_dated_where_the_identifier_changes` -- Vylor's case by bisection in
  six lookups, the earlier FIGI's span created and linked; Bunge's from answers on disk, no
  lookup, the predecessor extended; a day naming neither refused.
- `test_a_span_the_endpoint_names_at_both_ends_with_its_prices_unbroken_is_confirmed` -- IIVI's
  case; refused for another security than expected, a gap in the prices, an overlap.
- `test_a_corrected_plan_keeps_its_grid_and_names_each_spans_previous_holder` -- the grid
  through an origin both ways; the previous holder.

`tests/midoffice/test_rebuild.py`, 4:

- `test_a_sample_still_at_the_pre_split_count_is_stale_and_the_first_to_follow_resets_it` -- the
  lag rule; neither after stale is a contradiction; stale to the end, untested.
- `test_a_split_rests_on_its_price_where_the_counts_cannot_speak_and_must_gap` -- MNST's case; a
  price that does not gap, or moves by neither, is a contradiction.
- `test_the_raw_gap_is_the_open_on_the_day_over_the_close_on_the_trading_day_before`
- `test_a_record_with_no_identifier_and_the_change_days_stale_one_are_read_past` -- both rules;
  a third security fails, its share class reported; no identified sample fails.

**Rewritten for the new rule: 3.**

- `test_a_split_is_a_whole_ratio_..._they_leave` -- a ten-for-one whose counts stay put to the
  span's end is now a split resting on its price, and one whose price does not gap a
  contradiction.
- `test_counts_that_move_by_neither_the_ratio_nor_nothing_confirm_neither_kind` -- the new
  signature.
- `test_where_the_counts_cannot_test_an_event_its_ratio_alone_classifies_it` -- the new wording.

**Extended:** `test_the_plan_reads_back_as_written` reads a corrected plan's `previous`.

**Hygiene.** Every changed Python file parses, is ASCII and keeps its CRLF line endings, counted
in bytes. ruff over them: all checks passed (one import-order finding of this piece's fixed).

**The final code, re-run offline into scratch** (`regressions.txt`):

- the corrected build writes all five corrected CSVs byte for byte;
- the gate's report differs from the online run's only in its plan path and its request count
  (0, against 12);
- midoffice05's and 06's builds still write their CSVs byte for byte.

---

## Failures and debugging trail

- **The first corrected build stopped** (`corrected_run1.txt`, exit 1): the COHR confirmation was
  refused, which then raised. Refusals are now reported and the build goes on.
- **The truncation loop's variable `traded`** would have shadowed the corrections' cache of
  trading days, read by a closure. Renamed `trading_days` before any run.
- **A patch put a real newline inside an f-string** (the renamed-membership report): fixed
  before the first run, caught by the parse check.
- **A patch's search string didn't match** -- a backslash in the pattern. The patch scripts
  assert every match and write nothing on a miss, so nothing half-applied.
- **A bash heredoc with apostrophes failed again**, as in midoffice06. The gate patch went into
  files written with the Write tool.
- **One regression comparison failed only for its path.** A Git Bash path handed to Windows
  Python; rerun with a Windows path. The two regression builds first ran before the last code
  changes (the share class, the refusals, the docstrings), so both ran again on the final code;
  `regressions.txt` is the second run's.
- **The first plan diff keyed spans by (security, ticker)**, which collapses the three pairs of
  spans sharing both (Fiserv's FISV, Signature's SBNY, SMCI's). None of them changed, and the
  diff says so.
- **ruff found one import out of order** in the rewritten test file. Fixed.

## Problems found, not fixed

1. **COHR's change-day record** names a composite FIGI seen nowhere else; the gate stops on it.
2. **696 member-days have no share count** the records carry: spin-offs' first quarters, and
   all of FDXF's and Vylor's.
3. **The old panel's volumes are split-adjusted** beside point-in-time counts and traded prices:
   24,386 member-day cells over 39 tickers, the spin-offs' among them.
4. **Polygon's share counts lag splits**, now handled by the lag rule. Two samples taken on a
   split's own day read the pre-split count (WSM, SMCI).
5. **Composite FIGIs change at spans' ends under one share class** (EA, OKE, JNPR, BLK, XOM,
   DISCA): reconciled by the CIK, and by the share class if Jeff rules so.
6. **`docs/FACTOR_MODEL.md` lines 777-786** are still wrong (the 63-day grid, 896 and 621,
   DuPont's 0.33); the rewrite waits for the build, as ruled.
7. **`tests/test_market_data.py`** looks for `../core_repo/src`, not on this machine.

## Scratch

`claude/scratch/midoffice07/` (gitignored), 44 items: every script and output named here --
`extract_brief.py`, `write_brief.py`, `span_prices.py`, `rule_reach.py`, `share_class.py`,
`plan_diff.py`, `uncapped.py`, `volume_basis.py`, the patch scripts and their text files, the
runs (`corrected_run1.txt`, `corrected_run2.txt`, `gate_run1.txt`, `corrected_final.txt`,
`gate_final.txt`, `regress_plain.txt`, `regress_resolved.txt`) and their output directories, and
`build_retro.py`, which assembled this retro. Their outputs are copied into the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The two `git add` lines at the end were
checked with `git add --dry-run`; between them they name every changed path -- this piece's, then
the separate change Jeff asked for after it, the mid-office state file.

## Open items for Chat

- COHR's change-day record and the share-class reading; then the gate again, offline and
  instant, and the build.
- The build's choices: the share counts the records do not carry, and the traded volumes.
- The weighted counts, which here test the 1:1 carry for OKE and XOM, and which could be a
  standing check wherever a share-class count is missing.

---

## Appendix: outputs, verbatim

Every output the body rests on, from `claude/scratch/midoffice07/`, in full unless said otherwise. The session scratchpad's path is shortened to `<scratch>`.

### regressions.txt -- the builds and the gate re-run offline on the final code, compared with data/'s

```text
Each run with the final code, offline, into the scratch, and its files compared with data/'s.
1. midoffice05's build, no --resolve, the records read on the panel's own grid (regress_plain/):
   identical: sp500_identity_2022-01-03_2026-11-03.csv
   identical: sp500_identity_2022-01-03_2026-11-03_members.csv
   identical: sp500_identity_2022-01-03_2026-11-03_fetch.csv
2. midoffice06's build, --resolve (regress_resolved/):
   identical: sp500_identity_2022-01-03_2026-11-03_resolved.csv
   identical: sp500_identity_2022-01-03_2026-11-03_resolved_members.csv
   identical: sp500_identity_2022-01-03_2026-11-03_resolved_fetch.csv
   its JSON: the same keys True ; differing ['resolutions']
     link evidence now: WestRock Company traded as WRK to 2024-07-05, within a halt of Smurfit Westrock plc's first day under SW, 2024-07-08
       and then: WestRock Company traded to 2024-07-05, within a halt of Smurfit Westrock plc's first day under SW, 2024-07-08
     link evidence now: Paramount Global Class B Common Stock traded as PARA to 2025-08-06, within a halt of Skydance Corporation's first day under PSKY, 2025-08-07
       and then: ViacomCBS Inc. Class B Common Stock traded to 2025-08-06, within a halt of Skydance Corporation's first day under PSKY, 2025-08-07
     link evidence now: Bunge Limited traded as BG to 2023-10-31, within a halt of Bunge Global SA's first day under BG, 2023-11-01
       and then: Bunge Limited traded to 2023-10-31, within a halt of Bunge Global SA's first day under BG, 2023-11-01
3. this piece's build, --resolve --correct (final_corrected/):
   identical: sp500_identity_2022-01-03_2026-11-03_corrected.csv
   identical: sp500_identity_2022-01-03_2026-11-03_corrected_members.csv
   identical: sp500_identity_2022-01-03_2026-11-03_corrected_fetch.csv
   identical: sp500_identity_2022-01-03_2026-11-03_corrected_links.csv
   identical: sp500_identity_2022-01-03_2026-11-03_corrected_membership.csv
4. the gate on a copy of the corrected plan (final_gate/):
   against the online run: differing ['plan', 'requests'] ; requests 0 against 12 ; passed False False
```

### span_prices.txt -- midoffice06's spans against their prices: what the first-price and unpriced rules would reach

```text
<scratch>\span_prices.py:27: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  bars[ticker] = sorted(dt.datetime.utcfromtimestamp(b["t"] / 1000).date() for b in rows)
629 spans; 2 with no price inside; 13 whose first price is after their first day; 3 whose last price is before their last day

No price inside the span:
  S600   2023-11-20..2026-11-03 md    0 BBG0019LSQ49 'First Republic Bank' source events: began trading as S600 on 2023-11-18 (events)
  CEGV   2022-01-19..2022-02-01 md    0 BBG014KFRNP7 'Constellation Energy Corporati' source events: the first of its events, 2022-01-19: the start of Polygon's record of

First price after the span's first day:
  RDDT   2024-03-20..2026-11-03 first price 2024-03-21 md   55 BBG005K1D4S0 source events: the first of its events, 2024-03-20: the start of Polygon's record of it, not ne
  GEV    2024-03-27..2026-11-03 first price 2024-04-02 md  651 BBG013G17W31 source events: the first of its events, 2024-03-27: the start of Polygon's record of it, not ne
  SOLV   2024-03-26..2026-11-03 first price 2024-04-01 md  652 BBG018YZH6T3 source events: the first of its events, 2024-03-26: the start of Polygon's record of it, not ne
  VLTO   2023-09-27..2026-11-03 first price 2023-10-02 md  776 BBG019Q32XJ9 source events: the first of its events, 2023-09-27: the start of Polygon's record of it, not ne
  MBC    2022-12-09..2026-11-03 first price 2022-12-15 md    2 BBG01B9G42Y3 source events: the first of its events, 2022-12-09: the start of Polygon's record of it, not ne
  GEHC   2022-12-15..2026-11-03 first price 2023-01-04 md  962 BBG01BFR8YV1 source events: the first of its events, 2022-12-15: the start of Polygon's record of it, not ne
  KVUE   2023-05-03..2026-10-30 first price 2023-05-04 md  798 BBG01C79X561 source reference: reference: KVUE names Kenvue Inc. (BBG01C79X561) from 2023-05-03 to 2026-10-30;
  AMTM   2024-09-24..2026-11-03 first price 2024-09-30 md   59 BBG01NY7B734 source events: the first of its events, 2024-09-24: the start of Polygon's record of it, not ne
  SNDK   2025-02-13..2026-11-03 first price 2025-02-24 md  234 BBG01R388JG1 source events: the first of its events, 2025-02-13: the start of Polygon's record of it, not ne
  Q      2025-10-27..2026-11-03 first price 2025-11-03 md  252 BBG01TRL73S5 source events: the first of its events, 2025-10-27: the start of Polygon's record of it, not ne
  SOLS   2025-10-20..2026-11-03 first price 2025-10-30 md   36 BBG01X5744P5 source events: the first of its events, 2025-10-20: the start of Polygon's record of it, not ne
  FDXF   2026-05-27..2026-11-03 first price 2026-06-01 md  109 BBG01ZPB91C4 source events: the first of its events, 2026-05-27: the start of Polygon's record of it, not ne
  HONA   2026-06-15..2026-11-03 first price 2026-06-29 md   90 BBG020QYV152 source events: the first of its events, 2026-06-15: the start of Polygon's record of it, not ne

Last price before the span's last day:
  DAY    2024-02-01..2026-02-04 last price 2026-02-03 md  506 BBG005D7PF34 source reference: reference: DAY names Dayforce, Inc. (BBG005D7PF34) from 2024-02-01 to 2026-02-04
  WBD    2022-04-11..2026-10-06 last price 2026-10-05 md 1125 BBG011386VF4 source reference: reference: WBD names Warner Bros. Discovery, Inc. Series A Common Stock (BBG0113
  CRHC   2022-01-03..2023-09-22 last price 2022-11-09 md    0 CIK0000849395 source events: traded as another ticker from 2023-09-25 (events)
```

### rule_reach.txt -- how far the identifier-less and refuted rules would reach

```text
(a) identifier-less records
  among the panel's six-date records: 0 []
  among every cached record under the plan's 619 tickers: 1
    EG 2023-07-10 Everest Group, Ltd.

(b) rows whose cached records inside never name their own security
  S600   2023-11-20..2026-11-03 BBG0019LSQ49 'First Republic Bank' events: 14 records inside, own 0, other 0, none 14; []
  METV   2022-01-31..2026-11-03 BBG011J1MN62 'Roundhill Ball Metaverse ETF' events: 0 records inside, own 0, other 0, none 0; []
  CEGV   2022-01-19..2022-02-01 BBG014KFRNP7 'Constellation Energy Corporati' events: 2 records inside, own 0, other 0, none 2; []
  CRHC   2022-01-03..2023-09-22 CIK0000849395 'CRH Public Limited Company' events: 8 records inside, own 0, other 4, none 4; [('CIK0001818212', 4)]
```

### corrected_run1.txt -- the first corrected build, stopped by the refused confirmation (its last lines)

```text
      confirmed: KVUE names Kenvue Inc. from 2023-05-03 to 2026-10-30
      takes KVUE 2023-05-03..2026-10-30: Kenvue Inc. [BBG01C79X561]
  COHR before II-VI took it: Coherent, Inc., whose FIGI the change-day record of 2022-09-08 carries: not confirmed -- COHR names 2022-01-03: Coherent Inc (BBG000BG1DH3), 2022-07-01: Coherent Inc (BBG000BG1DH3)
```

### corrected_run2.txt -- the corrected build: scripts/build_sp500_identity.py --resolve --correct

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

The rebuild's evidence, taken to the table: 19 corrections; 0 reference lookups and 0 price request(s) over the network this run
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
  refused        COHR before II-VI took it: Coherent, Inc., whose FIGI the change-day record of 2022-09-08 carries: not confirmed -- COHR names 2022-01-03: Coherent Inc (BBG000BG1DH3), 2022-07-01: Coherent Inc (BBG000BG1DH3)

The table
  633 rows: 604 securities, each under one ticker over one span of the window; 518 rows priced by the panel, 103 from events alone, 1 presumed
  identified by the records and the events 476, by the records alone 40, by the events alone 80, by the reference endpoint alone 6

Guards
  ok  every priced (ticker, date) maps to exactly one security: 583,628 checked, 0 failure(s)
  ok  no security under two tickers on one date: 633 checked, 0 failure(s)
  ok  no ticker names two securities on one date: 633 checked, 0 failure(s)
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
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_corrected.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_corrected_members.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_corrected_fetch.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_corrected_links.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_corrected_membership.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_corrected.json
```

### plan_diff.txt -- the corrected plan against midoffice06's, span by span; and the share counts on the corrected spans' samples

```text
midoffice06's plan: 626 spans, 611,899 member-days, 11,943 samples
the corrected plan: 625 spans, 610,842 member-days, 11,925 samples

Gone:
  S600   2023-11-20..2026-11-03 md     0 BBG0019LSQ49 First Republic Bank
  CEGV   2022-01-19..2022-02-01 md     0 BBG014KFRNP7 Constellation Energy Corporation Common
  CRHC   2022-01-03..2023-09-22 md     0 CIK0000849395 CRH Public Limited Company

New:
  IIVI   2022-01-03..2022-09-07 md     0 BBG000BLW102 Ii-Vi Inc samples 2022-01-03 2022-03-09 2022-06-08 2022-09-07
  VYLR   2026-10-01..2026-10-02 md     2 BBG024W5VJC3 Vylor Inc. samples 2026-10-01 2026-10-02

Changed:
  SATS   2022-01-03..2026-06-23 -> 2022-01-03..2026-06-23; md 1121 -> 64; samples -[] +[]
  RDDT   2024-03-20..2026-11-03 -> 2024-03-21..2026-11-03; md 55 -> 55; samples -['2024-03-20'] +['2024-03-21']
  GEV    2024-03-27..2026-11-03 -> 2024-04-02..2026-11-03; md 651 -> 651; samples -['2024-03-27'] +['2024-04-02']
  SOLV   2024-03-26..2026-11-03 -> 2024-04-01..2026-11-03; md 652 -> 652; samples -['2024-03-26'] +['2024-04-01']
  VLTO   2023-09-27..2026-11-03 -> 2023-10-02..2026-11-03; md 776 -> 776; samples -['2023-09-27'] +['2023-10-02']
  MBC    2022-12-09..2026-11-03 -> 2022-12-15..2026-11-03; md 2 -> 2; samples -['2022-12-09'] +['2022-12-15']
  GEHC   2022-12-15..2026-11-03 -> 2023-01-04..2026-11-03; md 962 -> 962; samples -['2022-12-15'] +['2023-01-04']
  KVUE   2023-05-03..2026-10-30 -> 2023-05-04..2026-10-30; md 798 -> 798; samples -['2023-05-03'] +['2023-05-04']
  AMTM   2024-09-24..2026-11-03 -> 2024-09-30..2026-11-03; md 59 -> 59; samples -['2024-09-24'] +['2024-09-30']
  SNDK   2025-02-13..2026-11-03 -> 2025-02-24..2026-11-03; md 234 -> 234; samples -['2025-02-13'] +['2025-02-24']
  Q      2025-10-27..2026-11-03 -> 2025-11-03..2026-11-03; md 252 -> 252; samples -['2025-10-27'] +['2025-11-03']
  SOLS   2025-10-20..2026-11-03 -> 2025-10-30..2026-11-03; md 36 -> 36; samples -['2025-10-20'] +['2025-10-30']
  FDXF   2026-05-27..2026-11-03 -> 2026-06-01..2026-11-03; md 109 -> 109; samples -['2026-05-27'] +['2026-06-01']
  HONA   2026-06-15..2026-11-03 -> 2026-06-29..2026-11-03; md 90 -> 90; samples -['2026-06-15'] +['2026-06-29']
  VYLR   2026-10-01..2026-11-03 -> 2026-10-05..2026-11-03; md 24 -> 22; samples -['2026-10-01'] +['2026-10-05']
  BG     2022-01-03..2023-10-31 -> 2022-01-03..2023-11-07; md 160 -> 165; samples -['2023-10-31'] +['2023-11-07']
  BG     2023-11-01..2026-11-03 -> 2023-11-08..2026-11-03; md 754 -> 749; samples -['2023-11-01'] +['2023-11-08']

member-days changed on spans in both: 1,059
previous holders named: 11; on the spans: BNY<-BBG000BZF6G2, DOC<-BBG004MF59H0, META<-BBG011J1MN62, J<-BBG000BMFFQ0, LH<-BBG000D9DMK0, BLK<-BBG000C2PW58, XOM<-BBG000GZQ728, VYLR<-BBG024W5VJC3, OKE<-BBG000BQHGR6, CCL<-BBG000BF6LY3, BG<-CIK0001144519

Share counts on the corrected spans' samples:
  IIVI  2022-01-03..2022-09-07 BBG000BLW102: first 2022-01-03 106140000; 2022-03-09 106330000, 2022-06-08 106460000, 2022-09-07 130870000
  RDDT  2024-03-21..2026-11-03 BBG005K1D4S0: first 2024-03-21 36880000; 2024-06-20 44440000, 2024-09-19 94960000, 2024-12-18 119740000
  GEV   2024-04-02..2026-11-03 BBG013G17W31: first 2024-04-02 None; 2024-06-27 274090000, 2024-09-26 274800000, 2024-12-26 275650000
  SOLV  2024-04-01..2026-11-03 BBG018YZH6T3: first 2024-04-01 None; 2024-06-26 172710000, 2024-09-25 172710000, 2024-12-24 172760000
  VLTO  2023-10-02..2026-11-03 BBG019Q32XJ9: first 2023-10-02 None; 2023-12-27 246310000, 2024-03-28 246540000, 2024-06-28 246850000
  MBC   2022-12-15..2026-11-03 BBG01B9G42Y3: first 2022-12-15 128020000; 2023-03-14 128020000, 2023-06-13 128509999, 2023-09-13 127680000
  GEHC  2023-01-04..2026-11-03 BBG01BFR8YV1: first 2023-01-04 None; 2023-03-20 453930000, 2023-06-20 454680000, 2023-09-19 454840000
  KVUE  2023-05-04..2026-10-30 BBG01C79X561: first 2023-05-04 None; 2023-08-03 1914890000, 2023-11-01 1914890000, 2024-02-02 1915000000
  AMTM  2024-09-30..2026-11-03 BBG01NY7B734: first 2024-09-30 None; 2024-12-23 243300000, 2025-03-27 243300000, 2025-06-27 243322468
  SNDK  2025-02-24..2026-11-03 BBG01R388JG1: first 2025-02-24 None; 2025-05-15 145300000, 2025-08-15 145299942, 2025-11-13 145805548
  Q     2025-11-03..2026-11-03 BBG01TRL73S5: first 2025-11-03 None; 2026-01-28 209463194, 2026-04-29 209582425, 2026-07-30 209339207
  SOLS  2025-10-30..2026-11-03 BBG01X5744P5: first 2025-10-30 None; 2026-01-21 158727456, 2026-04-22 158795531, 2026-07-23 158804273
  FDXF  2026-06-01..2026-11-03 BBG01ZPB91C4: first 2026-06-01 None; 2026-08-26 None, 2026-11-03 None
  HONA  2026-06-29..2026-11-03 BBG020QYV152: first 2026-06-29 316826559; 2026-09-15 None, 2026-11-03 None
  VYLR  2026-10-05..2026-11-03 BBG024CQR6X9: first 2026-10-05 None; 2026-11-03 None
  VYLR  2026-10-01..2026-10-02 BBG024W5VJC3: first 2026-10-01 None; 2026-10-02 None
  BG    2022-01-03..2023-11-07 CIK0001144519: first 2022-01-03 141430000; 2022-04-04 142730000, 2022-07-06 142730000, 2022-10-04 151900000
  BG    2023-11-08..2026-11-03 CIK0001996862: first 2023-11-08 145290000; 2024-02-02 145290000, 2024-05-03 141600000, 2024-08-05 141650000
```

### gate_run1.txt -- the fetch and the gate on the corrected plan (progress lines but the last two left out)

```text
Plan
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_corrected_fetch.csv: 628 spans, 600 securities, 617 tickers, 11,950 share samples

Fetching
  at the start: this process 37 MB (peak 37, 508 committed); machine 736 of 26,420 MB available
  14,000/14,422  ACGL        16s  this process 66 MB (peak 66, 533 committed); machine 406 of 26,420 MB available
  14,422/14,422  SW          17s  this process 66 MB (peak 66, 534 committed); machine 396 of 26,420 MB available
  12 requests over the network, in 17s; 0 failed

Distributions: the cash paid under each security's tickers inside its spans
  8,234 over 467 of 600 securities, 133 of them paying none; by type {'CD': 8169, 'SC': 65}; by currency {'USD': 8234}
  SPY, the market series: 19 in the window, 32.8900 a share
    2022-02-17 WY     SC      1.4500 USD  frequency 0
    2022-04-13 FCX    SC      0.0750 USD  frequency 0
    2022-05-11 FANG   SC      2.3500 USD  frequency 0
    2022-06-03 LYB    SC      5.2000 USD  frequency 0
    2022-06-07 TPL    SC     20.0000 USD  frequency 0
    2022-06-14 EOG    SC      1.8000 USD  frequency 0
    2022-06-24 WRB    SC      0.5000 USD  frequency 0
    2022-07-14 FCX    SC      0.0750 USD  frequency 0
    2022-08-12 CTRA   SC      0.5000 USD  frequency 0
    2022-08-15 FANG   SC      2.3000 USD  frequency 0
    2022-08-18 TDG    SC     18.5000 USD  frequency 0
    2022-09-09 DVN    SC      1.3700 USD  frequency 0
    2022-09-14 EOG    SC      1.5000 USD  frequency 0
    2022-11-15 CTRA   SC      0.5300 USD  frequency 0
    2022-11-16 FANG   SC      1.5100 USD  frequency 0
    2022-12-09 DVN    SC      1.1700 USD  frequency 0
    2022-12-14 EOG    SC      1.5000 USD  frequency 0
    2022-12-15 PCAR   SC      2.8000 USD  frequency 0
    2022-12-27 CME    SC      4.5000 USD  frequency 1
    2022-12-29 HST    SC      0.2000 USD  frequency 0
    2023-01-12 WRB    SC      0.5000 USD  frequency 0
    2023-02-10 F      SC      0.6500 USD  frequency 0
    2023-02-14 WY     SC      0.9000 USD  frequency 0
    2023-03-02 FANG   SC      2.1500 USD  frequency 0
    2023-03-14 DVN    SC      0.6900 USD  frequency 0
    2023-03-14 MOS    SC      0.2500 USD  frequency 0
    2023-03-15 EOG    SC      1.0000 USD  frequency 0
    2023-03-15 CTRA   SC      0.3700 USD  frequency 0
    2023-09-22 WRB    SC      0.5000 USD  frequency 0
    2023-11-17 TDG    SC     35.0000 USD  frequency 0
    2023-12-05 FAST   SC      0.3800 USD  frequency 0
    2023-12-06 KIM    SC      0.0900 USD  frequency 0
    2023-12-08 DIS    SC      0.3000 USD  frequency 0
    2023-12-14 PCAR   SC      3.2000 USD  frequency 0
    2023-12-14 EOG    SC      1.5000 USD  frequency 0
    2023-12-15 WRB    SC      0.5000 USD  frequency 0
    2023-12-27 CME    SC      5.2500 USD  frequency 1
    2023-12-27 COST   SC     15.0000 USD  frequency 0
    2023-12-28 HST    SC      0.2500 USD  frequency 0
    2024-02-15 F      SC      0.1800 USD  frequency 0
    2024-02-15 COP    SC      0.2000 USD  frequency 0
    2024-02-15 WY     SC      0.1400 USD  frequency 0
    2024-03-04 FANG   SC      2.1800 USD  frequency 0
    2024-03-14 DVN    SC      0.2200 USD  frequency 0
    2024-03-14 DVN    SC      0.2200 USD  frequency 4
    2024-03-14 CRH    SC      0.3500 USD  frequency 0
    2024-05-23 CRH    SC      0.3500 USD  frequency 0
    2024-06-14 DVN    SC      0.1300 USD  frequency 0
    2024-06-24 WRB    SC      0.5000 USD  frequency 0
    2024-07-01 TPL    SC     10.0000 USD  frequency 0
    2024-08-23 CRH    SC      0.3500 USD  frequency 0
    2024-09-13 DVN    SC      0.2200 USD  frequency 4
    2024-09-23 WRB    SC      0.2500 USD  frequency 0
    2024-10-04 TDG    SC     75.0000 USD  frequency 0
    2024-11-18 ACGL   SC      5.0000 USD  frequency 0
    2024-11-22 CRH    SC      0.3500 USD  frequency 0
    2024-12-16 WRB    SC      0.5000 USD  frequency 0
    2024-12-20 PCAR   SC      3.0000 USD  frequency 0
    2024-12-27 CME    SC      5.8000 USD  frequency 1
    2025-02-18 F      SC      0.1500 USD  frequency 0
    2025-06-23 WRB    SC      0.5000 USD  frequency 0
    2025-09-02 TDG    SC     90.0000 USD  frequency 0
    2025-12-15 WRB    SC      1.0000 USD  frequency 0
    2026-03-10 CME    SC      6.1500 USD  frequency 1
    2026-06-30 HST    SC      0.7200 USD  frequency 0

Identity: each span's first share sample against the security the table expects
  627 of 628 spans pass
  FAILED  COHR 2022-09-08..2026-11-03 (157 member-days): COHR on 2022-09-08 names Coherent Corp. Common Stock (BBG000GS6QB4), not Coherent Corp. (BBG000BLW102); its share-class FIGI, BBG001S5S6N6, is the expected security's

Splits: each event classified by its ratio -- whole numbers, no term over 100 in lowest terms, or not -- by the security's share counts either side, within 10%, a sample still at the pre-split count stale until one follows, and by its raw price on the execution date, within the same of the ratio's reciprocal
  55 events inside the securities' spans: 47 splits, applied to the carried share count; 8 price-only, not applied; 0 contradictions, 0 of them exempt (DD); 2 untested, by the ratio alone
  price-only  GE Aerospace: GE 1000-to-1281 on 2023-01-04 (1.281, fractional): counts 1,096,550,000 on 2022-10-04, 1,092,670,000 on 2023-01-04, 0.996 times; raw price 84.98 at the close of 2023-01-03, 68.41 at the open of 2023-01-04, 0.805 times against 0.781; price-only, the counts within 10% of unchanged; the price gaps by it
  price-only  Laboratory Corporation of America Holdings: LH 1000-to-1164 on 2023-07-03 (1.164, fractional): counts 88,500,000 on 2023-04-05, 88,500,000 on 2023-07-07, 1.000 times; raw price 241.33 at the close of 2023-06-30, 206.15 at the open of 2023-07-03, 0.854 times against 0.859; price-only, the counts within 10% of unchanged; the price gaps by it
  price-only  3M Company: MMM 1000-to-1196 on 2024-04-01 (1.196, fractional): counts 552,320,000 on 2024-01-05, 553,360,000 on 2024-04-08, 1.002 times; raw price 106.07 at the close of 2024-03-28, 91.05 at the open of 2024-04-01, 0.858 times against 0.836; price-only, the counts within 10% of unchanged; the price gaps by it
  price-only  GE Aerospace: GE 1000-to-1253 on 2024-04-02 (1.253, fractional): counts 1,088,390,000 on 2024-01-05, 1,093,270,000 on 2024-04-08, 1.004 times; raw price 175.36 at the close of 2024-04-01, 140.53 at the open of 2024-04-02, 0.801 times against 0.798; price-only, the counts within 10% of unchanged; the price gaps by it
  price-only  untested Lennar Corporation Class A: LEN 1000-to-1033 on 2025-01-21 (1.033, fractional): counts 238,810,000 on 2025-01-07, 232,180,000 on 2025-04-09, 0.972 times; raw price 141.54 at the close of 2025-01-17, 136.01 at the open of 2025-01-21, 0.961 times against 0.968; untested, within 10% of both the ratio and no change; price-only by the ratio alone
  price-only  Fortive Corporation: FTV 1000-to-1327 on 2025-06-30 (1.327, fractional): counts 340,290,000 on 2025-04-09, 339,882,279 on 2025-07-11, 0.999 times; raw price 71.60 at the close of 2025-06-27, 52.05 at the open of 2025-06-30, 0.727 times against 0.754; price-only, the counts within 10% of unchanged; the price gaps by it
  price-only  untested Honeywell International, Inc.: HON 1000-to-1061 on 2025-10-30 (1.061, fractional): counts 634,896,562 on 2025-10-09, 634,887,208 on 2026-01-09, 1.000 times; raw price 212.89 at the close of 2025-10-29, 201.80 at the open of 2025-10-30, 0.948 times against 0.943; untested, within 10% of both the ratio and no change; price-only by the ratio alone
  price-only  DuPont de Nemours, Inc. Common Stock: DD 100-to-239 on 2025-11-03 (2.39, fractional): counts 418,716,685 on 2025-10-09, 418,975,324 on 2026-01-09, 1.001 times; raw price 81.65 at the close of 2025-10-31, 33.69 at the open of 2025-11-03, 0.413 times against 0.418; price-only, the counts within 10% of unchanged; the price gaps by it
  split       W.R. Berkley Corporation: WRB 2-to-3 on 2022-03-24 (1.5, whole numbers): counts 176,640,000 on 2022-01-03, 265,170,000 on 2022-04-04, 1.501 times; raw price 97.95 at the close of 2022-03-23, 64.64 at the open of 2022-03-24, 0.660 times against 0.667; a split, the counts follow it and the price gaps by it
  split       Amazon.Com Inc: AMZN 1-to-20 on 2022-06-06 (20, whole numbers): counts 508,840,000 on 2022-04-04, 10,174,410,000 on 2022-07-06, 19.995 times; raw price 2,447.00 at the close of 2022-06-03, 125.25 at the open of 2022-06-06, 0.051 times against 0.050; a split, the counts follow it and the price gaps by it
  split       DexCom, Inc.: DXCM 1-to-4 on 2022-06-13 (4, whole numbers): counts 97,390,000 on 2022-04-04, 392,500,000 on 2022-07-06, 4.030 times; raw price 293.85 at the close of 2022-06-10, 70.83 at the open of 2022-06-13, 0.241 times against 0.250; a split, the counts follow it and the price gaps by it
  split       Fortinet, Inc.: FTNT 1-to-5 on 2022-06-23 (5, whole numbers): counts 160,820,000 on 2022-04-04, 802,640,000 on 2022-07-06, 4.991 times; raw price 278.44 at the close of 2022-06-22, 56.20 at the open of 2022-06-23, 0.202 times against 0.200; a split, the counts follow it and the price gaps by it
  split       Alphabet Inc. Class C Capital Stock: GOOG 1-to-20 on 2022-07-18 (20, whole numbers): counts 313,380,000 on 2022-07-06, 6,163,000,000 on 2022-10-04, 19.666 times; raw price 2,255.34 at the close of 2022-07-15, 113.44 at the open of 2022-07-18, 0.050 times against 0.050; a split, the counts follow it and the price gaps by it
  split       Alphabet Inc. Class A Common Stock: GOOGL 1-to-20 on 2022-07-18 (20, whole numbers): counts 300,760,000 on 2022-07-06, 5,996,000,000 on 2022-10-04, 19.936 times; raw price 2,235.55 at the close of 2022-07-15, 112.64 at the open of 2022-07-18, 0.050 times against 0.050; a split, the counts follow it and the price gaps by it
  split       Tesla, Inc. Common Stock: TSLA 1-to-3 on 2022-08-25 (3, whole numbers): counts 1,036,010,000 on 2022-07-06, 3,133,470,000 on 2022-10-04, 3.025 times; raw price 891.29 at the close of 2022-08-24, 302.36 at the open of 2022-08-25, 0.339 times against 0.333; a split, the counts follow it and the price gaps by it
  split       Nasdaq, Inc. Common Stock: NDAQ 1-to-3 on 2022-08-29 (3, whole numbers): counts 164,680,000 on 2022-07-06, 491,230,000 on 2022-10-04, 2.983 times; raw price 182.07 at the close of 2022-08-26, 60.32 at the open of 2022-08-29, 0.331 times against 0.333; a split, the counts follow it and the price gaps by it
  split       Palo Alto Networks, Inc. Common Stock: PANW 1-to-3 on 2022-09-14 (3, whole numbers): counts 99,640,000 on 2022-07-06, 299,210,000 on 2022-10-04, 3.003 times; raw price 548.88 at the close of 2022-09-13, 183.75 at the open of 2022-09-14, 0.335 times against 0.333; a split, the counts follow it and the price gaps by it
  split       Copart Inc: CPRT 1-to-2 on 2022-11-04 (2, whole numbers): counts 238,060,000 on 2022-10-04, 476,300,000 on 2023-01-04, 2.001 times; raw price 113.39 at the close of 2022-11-03, 57.33 at the open of 2022-11-04, 0.506 times against 0.500; a split, the counts follow it and the price gaps by it
  split       Bio-Techne Corp.: TECH 1-to-4 on 2022-11-30 (4, whole numbers): counts 39,210,000 on 2022-10-04, 156,970,000 on 2023-01-04, 4.003 times; raw price 327.07 at the close of 2022-11-29, 81.74 at the open of 2022-11-30, 0.250 times against 0.250; a split, the counts follow it and the price gaps by it
  split       Paccar Inc: PCAR 2-to-3 on 2023-02-08 (1.5, whole numbers): counts 347,770,000 on 2023-01-04, 522,510,000 on 2023-04-05, 1.502 times; raw price 110.29 at the close of 2023-02-07, 73.50 at the open of 2023-02-08, 0.666 times against 0.667; a split, the counts follow it and the price gaps by it
  split       Monster Beverage Corporation: MNST 1-to-2 on 2023-03-28 (2, whole numbers): counts 521,740,000 on 2023-01-04, 1,044,819,999 on 2023-04-05, 2.003 times; raw price 104.67 at the close of 2023-03-27, 52.16 at the open of 2023-03-28, 0.498 times against 0.500; a split, the counts follow it and the price gaps by it
  split       Copart Inc: CPRT 1-to-2 on 2023-08-22 (2, whole numbers): counts 477,440,000 on 2023-07-07, 957,360,000 on 2023-10-05, 2.005 times; raw price 86.98 at the close of 2023-08-21, 43.33 at the open of 2023-08-22, 0.498 times against 0.500; a split, the counts follow it and the price gaps by it
  split       Sempra: SRE 1-to-2 on 2023-08-22 (2, whole numbers): counts 314,650,000 on 2023-07-07, 629,310,000 on 2023-10-05, 2.000 times; raw price 142.22 at the close of 2023-08-21, 71.24 at the open of 2023-08-22, 0.501 times against 0.500; a split, the counts follow it and the price gaps by it
  split       The Cooper Companies, Inc. Common Stock: COO 1-to-4 on 2024-02-20 (4, whole numbers): counts 49,530,000 on 2024-01-05, 198,760,000 on 2024-04-08, 4.013 times; raw price 372.01 at the close of 2024-02-16, 94.41 at the open of 2024-02-20, 0.254 times against 0.250; a split, the counts follow it and the price gaps by it
  split       Walmart Inc. Common Stock: WMT 1-to-3 on 2024-02-26 (3, whole numbers): counts 2,692,230,000 on 2024-01-05, 8,052,790,000 on 2024-04-08, 2.991 times; raw price 175.56 at the close of 2024-02-23, 59.12 at the open of 2024-02-26, 0.337 times against 0.333; a split, the counts follow it and the price gaps by it
  split       Texas Pacific Land Corporation: TPL 1-to-3 on 2024-03-27 (3, whole numbers): counts 7,680,000 on 2024-01-05, 23,010,000 on 2024-04-08, 2.996 times; raw price 1,679.18 at the close of 2024-03-26, 565.00 at the open of 2024-03-27, 0.336 times against 0.333; a split, the counts follow it and the price gaps by it
  split       Old Dominion Freight Line: ODFL 1-to-2 on 2024-03-28 (2, whole numbers): counts 109,110,000 on 2024-01-05, 217,670,000 on 2024-04-08, 1.995 times; raw price 427.95 at the close of 2024-03-27, 216.05 at the open of 2024-03-28, 0.505 times against 0.500; a split, the counts follow it and the price gaps by it
  split       Nvidia Corp: NVDA 1-to-10 on 2024-06-10 (10, whole numbers): counts 2,500,000,000 on 2024-04-08, 24,600,000,000 on 2024-07-09, 9.840 times; raw price 1,208.88 at the close of 2024-06-07, 120.37 at the open of 2024-06-10, 0.100 times against 0.100; a split, the counts follow it and the price gaps by it
  split       Amphenol Corporation: APH 1-to-2 on 2024-06-12 (2, whole numbers): counts 599,860,000 on 2024-04-08, 1,201,210,000 on 2024-07-09, 2.002 times; raw price 134.43 at the close of 2024-06-11, 68.01 at the open of 2024-06-12, 0.506 times against 0.500; a split, the counts follow it and the price gaps by it
  split       Chipotle Mexican Grill, Inc.: CMG 1-to-50 on 2024-06-26 (50, whole numbers): counts 27,420,000 on 2024-04-08, 1,373,350,000 on 2024-07-09, 50.086 times; raw price 3,283.04 at the close of 2024-06-25, 65.81 at the open of 2024-06-26, 0.020 times against 0.020; a split, the counts follow it and the price gaps by it
  split       Williams-Sonoma, Inc.: WSM 1-to-2 on 2024-07-09 (2, whole numbers): counts 64,110,000 on 2024-04-08, then 1 sample(s) still at it, to 2024-07-09 (64,310,000), 126,340,000 on 2024-10-07, 1.971 times; raw price 285.18 at the close of 2024-07-08, 141.46 at the open of 2024-07-09, 0.496 times against 0.500; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       W.R. Berkley Corporation: WRB 2-to-3 on 2024-07-11 (1.5, whole numbers): counts 255,660,000 on 2024-07-09, 380,550,000 on 2024-10-07, 1.489 times; raw price 79.63 at the close of 2024-07-10, 52.62 at the open of 2024-07-11, 0.661 times against 0.667; a split, the counts follow it and the price gaps by it
  split       Broadcom Inc. Common Stock: AVGO 1-to-10 on 2024-07-15 (10, whole numbers): counts 465,490,000 on 2024-07-09, 4,670,580,000 on 2024-10-07, 10.034 times; raw price 1,700.67 at the close of 2024-07-12, 170.00 at the open of 2024-07-15, 0.100 times against 0.100; a split, the counts follow it and the price gaps by it
  split       Cintas Corp: CTAS 1-to-4 on 2024-09-12 (4, whole numbers): counts 101,460,000 on 2024-07-09, 403,260,000 on 2024-10-07, 3.975 times; raw price 819.83 at the close of 2024-09-11, 205.50 at the open of 2024-09-12, 0.251 times against 0.250; a split, the counts follow it and the price gaps by it
  split       Deckers Outdoor Corp: DECK 1-to-6 on 2024-09-17 (6, whole numbers): counts 25,440,000 on 2024-07-09, 152,460,000 on 2024-10-07, 5.993 times; raw price 937.68 at the close of 2024-09-16, 156.33 at the open of 2024-09-17, 0.167 times against 0.167; a split, the counts follow it and the price gaps by it
  split       Super Micro Computer, Inc. Common Stock: SMCI 1-to-10 on 2024-10-01 (10, whole numbers): counts 58,560,000 on 2024-09-30, then 1 sample(s) still at it, to 2024-10-01 (58,560,000), 585,570,000 on 2024-10-07, 9.999 times; raw price 416.40 at the close of 2024-09-30, 41.75 at the open of 2024-10-01, 0.100 times against 0.100; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       Lam Research Corp: LRCX 1-to-10 on 2024-10-03 (10, whole numbers): counts 130,740,000 on 2024-07-09, 1,296,120,000 on 2024-10-07, 9.914 times; raw price 814.09 at the close of 2024-10-02, 79.93 at the open of 2024-10-03, 0.098 times against 0.100; a split, the counts follow it and the price gaps by it
  split       Arista Networks: ANET 1-to-4 on 2024-12-04 (4, whole numbers): counts 314,150,000 on 2024-10-07, 1,259,760,000 on 2025-01-07, 4.010 times; raw price 414.46 at the close of 2024-12-03, 106.01 at the open of 2024-12-04, 0.256 times against 0.250; a split, the counts follow it and the price gaps by it
  split       Entergy Corporation: ETR 1-to-2 on 2024-12-13 (2, whole numbers): counts 213,830,000 on 2024-10-07, 428,820,000 on 2025-01-07, 2.005 times; raw price 148.27 at the close of 2024-12-12, 74.13 at the open of 2024-12-13, 0.500 times against 0.500; a split, the counts follow it and the price gaps by it
  split       Palo Alto Networks, Inc. Common Stock: PANW 1-to-2 on 2024-12-16 (2, whole numbers): counts 325,600,000 on 2024-10-07, 656,200,000 on 2025-01-07, 2.015 times; raw price 393.12 at the close of 2024-12-13, 198.15 at the open of 2024-12-16, 0.504 times against 0.500; a split, the counts follow it and the price gaps by it
  split       Tractor Supply Co: TSCO 1-to-5 on 2024-12-20 (5, whole numbers): counts 107,870,000 on 2024-10-07, 534,200,000 on 2025-01-07, 4.952 times; raw price 264.03 at the close of 2024-12-19, 52.66 at the open of 2024-12-20, 0.199 times against 0.200; a split, the counts follow it and the price gaps by it
  split       Fastenal Co: FAST 1-to-2 on 2025-05-22 (2, whole numbers): counts 573,450,000 on 2025-04-09, then 1 sample(s) still at it, to 2025-07-11 (573,620,638), 1,147,636,617 on 2025-10-09, 2.001 times; raw price 81.46 at the close of 2025-05-21, 40.86 at the open of 2025-05-22, 0.502 times against 0.500; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       O'Reilly Automotive, Inc.: ORLY 1-to-15 on 2025-06-10 (15, whole numbers): counts 57,240,000 on 2025-04-09, then 1 sample(s) still at it, to 2025-07-11 (56,989,455), 848,496,680 on 2025-10-09, 14.823 times; raw price 1,348.10 at the close of 2025-06-09, 90.00 at the open of 2025-06-10, 0.067 times against 0.067; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       Interactive Brokers Group, Inc. Class A Common Stock: IBKR 1-to-4 on 2025-06-18 (4, whole numbers): counts 108,930,000 on 2025-04-09, then 1 sample(s) still at it, to 2025-07-11 (108,958,830), 445,246,976 on 2025-10-09, 4.087 times; raw price 208.17 at the close of 2025-06-17, 52.09 at the open of 2025-06-18, 0.250 times against 0.250; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       NetFlix Inc: NFLX 1-to-10 on 2025-11-17 (10, whole numbers): counts 424,926,346 on 2025-10-09, 4,237,323,340 on 2026-01-09, 9.972 times; raw price 1,112.17 at the close of 2025-11-14, 110.75 at the open of 2025-11-17, 0.100 times against 0.100; a split, the counts follow it and the price gaps by it
  split       SERVICENOW, INC.: NOW 1-to-5 on 2025-12-18 (5, whole numbers): counts 208,000,000 on 2025-10-09, then 1 sample(s) still at it, to 2026-01-09 (207,678,139), 1,036,155,572 on 2026-04-13, 4.982 times; raw price 782.39 at the close of 2025-12-17, 156.51 at the open of 2025-12-18, 0.200 times against 0.200; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       Texas Pacific Land Corporation: TPL 1-to-3 on 2025-12-23 (3, whole numbers): counts 22,987,326 on 2025-10-09, then 1 sample(s) still at it, to 2026-01-09 (22,979,410), 68,941,554 on 2026-04-13, 2.999 times; raw price 908.40 at the close of 2025-12-22, 305.48 at the open of 2025-12-23, 0.336 times against 0.333; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       Amcor plc Ordinary Shares: AMCR 5-to-1 on 2026-01-15 (0.2, whole numbers): counts 2,308,359,941 on 2026-01-09, 462,045,690 on 2026-04-13, 0.200 times; raw price 8.82 at the close of 2026-01-14, 44.56 at the open of 2026-01-15, 5.052 times against 5.000; a split, the counts follow it and the price gaps by it
  split       Booking Holdings Inc. Common Stock: BKNG 1-to-25 on 2026-04-06 (25, whole numbers): counts 32,233,815 on 2026-01-09, then 1 sample(s) still at it, to 2026-04-13 (31,673,346), 774,878,436 on 2026-07-14, 24.039 times; raw price 4,194.31 at the close of 2026-04-02, 166.25 at the open of 2026-04-06, 0.040 times against 0.040; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       Carvana Co.: CVNA 1-to-5 on 2026-05-08 (5, whole numbers): counts 142,993,769 on 2026-04-13, then 1 sample(s) still at it, to 2026-07-14 (143,257,677), 719,916,415 on 2026-10-12, 5.035 times; raw price 400.02 at the close of 2026-05-07, 81.03 at the open of 2026-05-08, 0.203 times against 0.200; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       KLA Corporation Common Stock: KLAC 1-to-10 on 2026-06-12 (10, whole numbers): counts 131,076,611 on 2026-04-13, then 1 sample(s) still at it, to 2026-07-14 (130,627,521), 1,305,006,492 on 2026-10-12, 9.956 times; raw price 2,411.64 at the close of 2026-06-11, 237.60 at the open of 2026-06-12, 0.099 times against 0.100; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       DuPont de Nemours, Inc. Common Stock: DD 3-to-1 on 2026-06-24 (0.3333, whole numbers): counts 408,923,857 on 2026-04-13, 135,019,401 on 2026-07-14, 0.330 times; raw price 46.67 at the close of 2026-06-23, 144.75 at the open of 2026-06-24, 3.102 times against 3.000; a split, the counts follow it and the price gaps by it
  split       CrowdStrike Holdings, Inc. Class A Common Stock: CRWD 1-to-4 on 2026-07-02 (4, whole numbers): counts 253,614,090 on 2026-04-13, then 1 sample(s) still at it, to 2026-07-14 (254,564,820), 1,026,129,839 on 2026-10-12, 4.046 times; raw price 772.74 at the close of 2026-07-01, 191.25 at the open of 2026-07-02, 0.248 times against 0.250; a split, the counts follow it after 1 stale sample(s) and the price gaps by it
  split       Monster Beverage Corporation: MNST 1-to-2 on 2026-08-11 (2, whole numbers): counts 978,008,110 on 2026-07-14, then 2 sample(s) still at it, to 2026-11-03 (979,525,882), to the span's end; raw price 91.43 at the close of 2026-08-10, 46.37 at the open of 2026-08-11, 0.507 times against 0.500; a split, the price gaps by it; the counts stale to the span's end
  split       Amphenol Corporation: APH 1-to-2 on 2026-09-03 (2, whole numbers): counts 1,230,234,445 on 2026-07-14, then 2 sample(s) still at it, to 2026-11-03 (1,232,983,457), to the span's end; raw price 160.08 at the close of 2026-09-02, 80.82 at the open of 2026-09-03, 0.505 times against 0.500; a split, the price gaps by it; the counts stale to the span's end

Gate: STOPS
  wrote data\external\polygon\sp500_rebuild_gate_2022-01-03_2026-11-03_corrected.json
```

### gate_gaps.txt -- each event's raw gap against its ratio, and the splits whose counts lag

```text
split: 47 events; the open over the close before, against the ratio's reciprocal: median deviation 0.81%, largest 3.58% (DXCM 2022-06-13, 0.2410 against 0.2500)
price-only: 8 events; the open over the close before, against the ratio's reciprocal: median deviation 1.39%, largest 3.53% (FTV 2025-06-30, 0.7270 against 0.7536)

The splits whose counts lag, and how:
  WSM   2024-07-09 1-to-2: stale 2024-07-09 64,310,000; followed 2024-10-07 126,340,000; open/close 0.4960 against 0.5000
  SMCI  2024-10-01 1-to-10: stale 2024-10-01 58,560,000; followed 2024-10-07 585,570,000; open/close 0.1003 against 0.1000
  FAST  2025-05-22 1-to-2: stale 2025-07-11 573,620,638; followed 2025-10-09 1,147,636,617; open/close 0.5016 against 0.5000
  ORLY  2025-06-10 1-to-15: stale 2025-07-11 56,989,455; followed 2025-10-09 848,496,680; open/close 0.0668 against 0.0667
  IBKR  2025-06-18 1-to-4: stale 2025-07-11 108,958,830; followed 2025-10-09 445,246,976; open/close 0.2502 against 0.2500
  NOW   2025-12-18 1-to-5: stale 2026-01-09 207,678,139; followed 2026-04-13 1,036,155,572; open/close 0.2000 against 0.2000
  TPL   2025-12-23 1-to-3: stale 2026-01-09 22,979,410; followed 2026-04-13 68,941,554; open/close 0.3363 against 0.3333
  BKNG  2026-04-06 1-to-25: stale 2026-04-13 31,673,346; followed 2026-07-14 774,878,436; open/close 0.0396 against 0.0400
  CVNA  2026-05-08 1-to-5: stale 2026-07-14 143,257,677; followed 2026-10-12 719,916,415; open/close 0.2026 against 0.2000
  KLAC  2026-06-12 1-to-10: stale 2026-07-14 130,627,521; followed 2026-10-12 1,305,006,492; open/close 0.0985 against 0.1000
  CRWD  2026-07-02 1-to-4: stale 2026-07-14 254,564,820; followed 2026-10-12 1,026,129,839; open/close 0.2475 against 0.2500
  MNST  2026-08-11 1-to-2: stale 2026-10-12 979,525,882, 2026-11-03 979,525,882; followed not in the window; open/close 0.5072 against 0.5000
  APH   2026-09-03 1-to-2: stale 2026-10-12 1,232,983,457, 2026-11-03 1,232,983,457; followed not in the window; open/close 0.5049 against 0.5000

Events whose first sample on or after is on the execution date itself:
  GE    2023-01-04 price-only: 1,092,670,000
  WSM   2024-07-09 split: 64,310,000
  SMCI  2024-10-01 split: 58,560,000
```

### share_class.txt -- every sample naming another composite FIGI than the table's, with both share classes

```text
COHR  2022-09-08 2022-09-08..2026-11-03 expects BBG000BLW102 (share class BBG001S5S6N6); names BBG000GS6QB4 'Coherent Corp. Common Stock' cik None share class BBG001S5S6N6: same share class
EA    2026-08-04 2022-01-03..2026-08-04 expects BBG000BP0KQ8 (share class BBG001S5T5C9); names BBG000K1X329 'Electronic Arts Inc' cik 0000712515 share class BBG001S5T5C9: same share class
OKE   2026-09-10 2022-01-03..2026-09-10 expects BBG000BQHGR6 (share class BBG001S5TWK1); names BBG000JBSHW1 'Oneok, Inc.' cik 0001039684 share class BBG001S5TWK1: same share class
JNPR  2025-07-01 2022-01-03..2025-07-01 expects BBG000BY33P5 (share class BBG001SCTT05); names BBG000BWS318 'Juniper Networks Inc' cik 0001043604 share class BBG001SCTT05: same share class
BLK   2024-10-01 2022-01-03..2024-10-01 expects BBG000C2PW58 (share class BBG001SDBP55); names BBG000CN81W9 'Blackrock, Inc.' cik 0001364742 share class BBG001SDBP55: same share class
XOM   2026-07-01 2022-01-03..2026-07-01 expects BBG000GZQ728 (share class BBG001S69V32); names BBG000H5HTR8 'Exxon Mobil Corporation' cik 0000034088 share class BBG001S69V32: same share class
DISCA 2022-04-04 2022-01-03..2022-04-08 expects BBG00HW3TQG4 (share class BBG001SLZ9C9); names BBG000CHWP52 'Discovery, Inc. Series A Common St' cik 0001024333 share class BBG001SLZ9C9: same share class
VYLR  2026-10-01 2026-10-01..2026-11-03 expects BBG024CQR6X9 (share class BBG024CQR6Z7); names BBG024W5VJC3 'Vylor Inc.' cik 0002128626 share class BBG024W5VJF0: another share class
```

### uncapped.txt -- the member-days a panel would have no share count for, by security

```text
592 member columns, 610,842 member-days; with no share-class count carried: 696; with neither it nor a weighted count: 624
  FDXF   FedEx Freight Holding Company, Inc.    BBG01ZPB91C4   member-days   109: none carried  109 (first class count None), with weighted   60 (first 2026-08-26)
  WBD    Warner Bros. Discovery, Inc. Series A  BBG011386VF4   member-days  1125: none carried   63 (first class count 2022-07-13), with weighted   63 (first 2022-07-13)
  PSKY   Skydance Corporation                   BBG01VS5NK99   member-days   313: none carried   63 (first class count 2025-11-05), with weighted   63 (first 2025-11-05)
  GEV    GE Vernova Inc.                        BBG013G17W31   member-days   651: none carried   60 (first class count 2024-06-27), with weighted   60 (first 2024-06-27)
  SOLV   Solventum Corporation                  BBG018YZH6T3   member-days   652: none carried   60 (first class count 2024-06-26), with weighted   60 (first 2024-06-26)
  VLTO   Veralto Corporation                    BBG019Q32XJ9   member-days   776: none carried   60 (first class count 2023-12-27), with weighted   60 (first 2023-12-27)
  AMTM   Amentum Holdings, Inc.                 BBG01NY7B734   member-days    59: none carried   59 (first class count 2024-12-23), with weighted   59 (first 2024-12-23)
  Q      Qnity Electronics, Inc.                BBG01TRL73S5   member-days   252: none carried   58 (first class count 2026-01-28), with weighted   58 (first 2026-01-28)
  CEG    Constellation Energy Corporation Commo BBG014KFRNP7   member-days  1193: none carried   53 (first class count 2022-04-20), with weighted   53 (first 2022-04-20)
  GEHC   GE HealthCare Technologies Inc. Common BBG01BFR8YV1   member-days   962: none carried   51 (first class count 2023-03-20), with weighted   51 (first 2023-03-20)
  SOLS   Solstice Advanced Materials Inc. Commo BBG01X5744P5   member-days    36: none carried   36 (first class count 2026-01-21), with weighted   36 (first 2026-01-21)
  VYLR   Vylor Inc.                             BBG024CQR6X9   member-days    24: none carried   24 (first class count None), with weighted    1 (first 2026-10-02)
```

### volume_basis.txt -- the old panel's stored volumes against the traded ones

```text
506 tickers: 24,386 priced cells whose stored volume is not the traded one (beyond 0.1%), 24,386 of them member-days, over 39 tickers
  APH     1171 cells,  1171 member-days; stored over traded 2, 4
  MNST    1154 cells,  1154 member-days; stored over traded 2, 4
  DD      1121 cells,  1121 member-days; stored over traded 0.33, 0.8
  KLAC    1114 cells,  1114 member-days; stored over traded 10
  BKNG    1066 cells,  1066 member-days; stored over traded 25
  AMCR    1012 cells,  1012 member-days; stored over traded 0.2
  NOW      994 cells,   994 member-days; stored over traded 5
  NFLX     972 cells,   972 member-days; stored over traded 10
  HON      960 cells,   960 member-days; stored over traded 1.06
  FTV      874 cells,   874 member-days; stored over traded 1.33
  ORLY     861 cells,   861 member-days; stored over traded 15
  FAST     849 cells,   849 member-days; stored over traded 2
  LEN      764 cells,   764 member-days; stored over traded 1.03
  TSCO     746 cells,   746 member-days; stored over traded 5
  ETR      741 cells,   741 member-days; stored over traded 2
  ANET     734 cells,   734 member-days; stored over traded 4
  LRCX     691 cells,   691 member-days; stored over traded 10
  CTAS     676 cells,   676 member-days; stored over traded 4
  AVGO     634 cells,   634 member-days; stored over traded 10
  WRB      632 cells,   632 member-days; stored over traded 1.5, 2.25
  CMG      622 cells,   622 member-days; stored over traded 50
  NVDA     611 cells,   611 member-days; stored over traded 10
  GE       563 cells,   563 member-days; stored over traded 1.25, 1.61
  MMM      562 cells,   562 member-days; stored over traded 1.2
  ODFL     561 cells,   561 member-days; stored over traded 2
  WMT      538 cells,   538 member-days; stored over traded 3
  COO      534 cells,   534 member-days; stored over traded 4
  CPRT     410 cells,   410 member-days; stored over traded 2, 4
  SRE      410 cells,   410 member-days; stored over traded 2
  LH       375 cells,   375 member-days; stored over traded 1.16
  PCAR     276 cells,   276 member-days; stored over traded 1.5
  TECH     229 cells,   229 member-days; stored over traded 4
  NDAQ     164 cells,   164 member-days; stored over traded 3
  TSLA     162 cells,   162 member-days; stored over traded 3
  GOOG     134 cells,   134 member-days; stored over traded 20
  GOOGL    134 cells,   134 member-days; stored over traded 20
  FTNT     118 cells,   118 member-days; stored over traded 5
  DXCM     111 cells,   111 member-days; stored over traded 4
  AMZN     106 cells,   106 member-days; stored over traded 20
```

### pytest_all.txt -- the full suite, --ignore=tests/test_market_data.py (last lines)

```text

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2279 passed, 1 skipped, 19 warnings in 87.06s (0:01:27) ===========
```

### pytest_mid.txt -- the midoffice tests (last lines)

```text
tests\midoffice\test_valuation.py ..............................         [100%]

======================= 828 passed, 1 skipped in 30.33s =======================
```

### ruff_final.txt -- ruff over every changed Python file

```text
All checks passed!
```

## git add

This change -- the corrections, the regenerated plan's code, the gate's new rules, every file, the brief and this retro included:

```
git add src/praxis/midoffice/refdata/identity.py src/praxis/midoffice/refdata/rebuild.py scripts/build_sp500_identity.py scripts/fetch_sp500_rebuild.py tests/midoffice/test_identity.py tests/midoffice/test_rebuild.py claude/handoffs/BRIEF_midoffice07_rebuild.md claude/retros/RETRO_midoffice07_rebuild.md
```

## git add -- the separate change: the mid-office state file

Made after this piece, at Jeff's request before he committed it: `claude/MIDOFFICE_STATE.md`, the state a fresh session resumes the mid-office work from -- how the work runs, the model's commits, every settled rule with the retro that holds it, the data and the commands that regenerate it, the run and its guards, the queue and the decisions waiting on Jeff. With it: its row in `claude/META_DOCS.md`'s read-at-session-start set (v1.1), and Rule 43's new step -- each piece brings the file up to date in the same commit as its retro -- in `claude/CLAUDE_CODE_RULES.md` (v1.7) and `.claude/commands/retro.md`. It goes after this piece's `git add`, as Jeff asked, though Rule 43 puts a separate change's before; the rule's wording is unchanged.

The old panel's regeneration command the file gives was run from the cache into the scratch with a dummy key, so that an uncached request would fail rather than fetch; every array came back identical, and the cache gained no file (`claude/scratch/midoffice07/state/`):

```text
python scripts/fetch_sp500_panel.py --start 2022-01-03 --end 2026-11-04 --shares-every 252 --out <scratch>/state/panel_check --env <scratch>/state/dummy.env  (a dummy key: an uncached request would fail, not fetch)
the same arrays: True
  dates                (1214,)      identical: True
  securities           (506,)       identical: True
  prices               (1214, 506)  identical: True
  unadjusted_prices    (1214, 506)  identical: True
  volumes              (1214, 506)  identical: True
  shares_outstanding   (1214, 506)  identical: True
  highs                (1214, 506)  identical: True
  lows                 (1214, 506)  identical: True
  market_returns       (1214,)      identical: True
```

Its own commit, after this piece's:

```
git add claude/MIDOFFICE_STATE.md claude/META_DOCS.md claude/CLAUDE_CODE_RULES.md .claude/commands/retro.md
```
