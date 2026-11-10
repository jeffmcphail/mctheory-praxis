# Retro: Mid-office 06 -- the rebuild, stopped at its gate

**Brief:** `claude/handoffs/BRIEF_midoffice06_rebuild.md` -- written in Chat and pasted into the
Code session, transcribed verbatim, with the two messages that amended it mid-piece
**Date:** 2026-11-09 to 2026-11-10
**Duration:** the brief arrived at 14:57 EST on 2026-11-09 (19:57:42Z); items 1 and 2's code and
the first fetch ran to 15:30, when Claude Code stopped the fetch for low memory. Jeff's message
of 01:07 on 2026-11-10 restarted it at 01:15 and changed the split rule; the fetch finished at
01:35, the gate's report at 01:36; his second message came at 01:38; the dividends, the
diagnoses and this retro ran to about 02:07.
**Status:** STOPPED AT THE GATE, as the brief directs. Items 1 and 2 are done; item 3, the gate,
stops -- 18 of 629 spans fail the identity check and 11 split events are contradictions -- so
nothing is built and items 4 to 6 wait. The dividend fetch, which does not depend on the gate,
is done. Uncommitted; one `git add` at the end.

---

## Summary

- **Item 1, the named member-days: all 4,813 resolved** through the reference endpoint on the
  brief's hypotheses, in 128 lookups; the table has 634 rows over 603 securities and every guard
  passes. DISCA for WBD is the one hypothesis the endpoint refutes: WBD's records name a
  security of its own, which accounts for WBD's member-days (a decision below).
- **Item 2, the fetch: complete.** 629 spans, 599 securities, 619 tickers, 11,968 share samples;
  6,640 requests over the network in the restarted run, none failed. The first run was stopped
  by Claude Code for low memory at step 5,000 of 13,825. It held every ticker's bars, about
  560 MB, and now counts them instead; its log records its own memory, 29 to 47 MB through the
  restarted run while the machine's available memory moved between 343 and 3,488 MB of 26,420.
  The pressure was the machine's: Chrome alone held 17.2 GB.
- **The split rule, changed before the gate ran** (Jeff, 01:07). Each event is classified by
  whether its ratio is one of whole numbers -- no term over 100 in lowest terms, since Polygon
  writes a spin-off as 1,000 to 1,281 -- and whether the share counts either side move by it,
  within 10%. Of the 55 events inside the spans: 36 splits; 8 price-only adjustments, each a
  spin-off's ex-date, seven named and confirmed at the endpoint and LEN's not; 11
  contradictions; and 2 of the 8 (LEN, HON) untested, their ratios too near one for the counts
  to tell from no change. DuPont's two classify cleanly -- the 2.39 is Qnity's spin-off, the
  0.33 a one-for-three the counts follow -- so its exemption was never used.
- **The gate stops, on both checks.**
  - **Identity, 18 spans.** 15 fail on their first day alone and pass at every later sample:
    11 spin-offs whose spans start at their list dates, before any price; EG and BG, whose
    change-day records carry no identifier, or the predecessor's; COHR, whose change-day record
    carries old Coherent's FIGI -- and COHR is II-VI, whose IIVI history the table lacks. 3
    never pass, and none carries a member-day: S600 and CEGV, with no prices or records at all,
    and CRHC, which was Cohn Robbins Holdings, a SPAC. Each with its correction and its fetch,
    below.
  - **Splits, 11 contradictions**, every one a whole-number split from 2025-05-22 on whose next
    sample still reads the pre-split count. Polygon's share counts change character in 2025 Q3
    -- rounded to 10,000 shares in 98% of samples before, exact in 94% after -- and the exact
    counts lag a split, as a filing's figure would wait for the next filing: by 99 to 157 days
    for nine of them, and past the window's end for MNST and APH.
- **Dividends, fetched** (Jeff, 01:38): 8,234 distributions inside the securities' spans, 8,169
  regular and 65 special, all in USD, over 467 of the 599 securities; and SPY's 19. Each cash
  amount is as paid at the time, so a total return puts it on the prices' split basis with the
  ex-date's ratio of adjusted to raw price. The median member's yield over its spans is 1.40% a
  year (p90 3.95%, the highest 7.78%), SPY's 1.27%. `prices.py`'s docstring is corrected, and
  so is `market_bars`', which made the same claim.
- **LEN, 2025-01-21: the gap is not at Millrose's distribution.** From 2025-02-05 to 02-11 LEN's
  opening gaps match three homebuilders' within 0.17 points; on 2025-01-21 LEN opened 5.90
  points below them, and the 1.033 step adjusts 3.2 of those points. Not misdated to Millrose;
  not corrected.
- **Tests:** 2,269 passed, 1 skipped (midoffice: 818 passed, 1 skipped); 16 new.

---

## What was asked

The brief and both amendments are in `claude/handoffs/BRIEF_midoffice06_rebuild.md`, verbatim
from the transcript, the paste wrappers stripped. In outline:

- **The brief** (2026-11-09, 19:57:42Z). Rulings on midoffice05's five decisions: a
  reorganisation is one security with one continuous return series, keyed by the successor's
  FIGI, the predecessor linked, its share count sampled either side of the boundary and the
  ratio reported; the named member-days to the reference endpoint, the public-record
  candidates hypotheses only, "only what the endpoint confirms enters the table"; renames in
  the run, with sp500's current_stale warnings printed; the whole window for every member
  security, a span-plus-look-back cut kept as a parameter; Casey's checked by its first share
  sample. Then midoffice06: (1) resolve the named member-days and update the table and the
  plan; (2) run the fetch -- resumable, cached, the original rate; prices adjusted and not,
  share counts inside each security's spans, split events; the first share sample of every span
  checks its identity; (3) the gate -- "if any span's identity check fails, or any split other
  than DD's disagrees with the next share sample, stop here and report before building
  anything"; (4) apply splits, with an exact guard; (5) build the panel on securities; (6) run
  the model on it and report against the old panel. "Same discipline -- retro and git add lines
  at the end, then stop."
- **The first amendment** (2026-11-10, 06:07:25Z): restart the fetch, logging its own memory
  every 500 requests; before the gate runs, change its split rule -- classify each event by two
  independent tests, the ratio's whole numbers and the counts moving by it within 10%, a
  parameter; whole and followed is a split, applied; fractional and unchanged is price-only, not
  applied, each reported with its distribution named where it can be; the tests disagreeing
  stops the gate; no sample on a side, classify by the ratio and report it untested; a field
  naming each event's kind, if there is one, a third test. The guard unchanged, counting splits
  only.
- **The second amendment** (queued 06:38:19Z, surfaced mid-turn): the open reading ruled -- a
  fractional ratio the counts move by neither it nor nothing stops the gate, and the cap of 100
  in lowest terms is the right reading, kept a parameter. Two additions: fetch the dividend
  events for every security in the plan, special ones included, and for SPY, build total
  returns from them on the prices' split basis, correct `prices.py`'s docstring, run the fetch
  now, and keep its effect separate in the comparison -- identity first, then dividends; and
  check LEN's raw prices around Millrose's distribution, reporting and not correcting. If the
  gate stops on identity: per failing span, the FIGI expected, the FIGIs its samples name with
  their dates, what the events and records said, the correction, any extra fetch -- "COHR may
  need IIVI's history, if II-VI is the company that took the ticker in 2022. Then stop, as you
  planned."

---

## Files modified

| File | Change | Lines |
|---|---|---|
| `src/praxis/midoffice/refdata/polygon.py` | a not-found answer cached and raised again; `has_cached`; `Split` and `splits()`; `Dividend` and `dividends()`; the `splits` and `market_bars` docstrings corrected | 149-179, 209-243, 257-269, 335-408, 469-471 |
| `src/praxis/midoffice/refdata/identity.py` | item 1: `map_members` follows `links`; `renames_from` reads panel and reference rows; `Hypothesis`, `Link`, `Resolution`, `_settled`, `_trimmed`, `resolve_named` | 55, 832-863, 935-939, 985-1001, 1248-1516 |
| `src/praxis/midoffice/refdata/rebuild.py` | new: the plan, the fetch (prices counted not held, split events, distributions, the market as a ticker), the identity check, the split classification (`is_whole`, `classify`, `check_splits`), the gate | new, 377 lines |
| `src/praxis/midoffice/refdata/prices.py` | `MARKET_PROXY`'s docstring: the adjusted prices are price returns, total returns need the distributions | 45-54 |
| `scripts/build_sp500_identity.py` | item 1: `--resolve`, the hypotheses, the resolution's report, the `_resolved` files | 15-19, 55-73, 115-130, 210-268, 408-409, 458-459, 507-517 |
| `scripts/fetch_sp500_rebuild.py` | new: the fetch and the gate; `memory()` on every progress line; `--fetch-only`, `--offline`, `--tolerance`, `--largest-term`, `--market`, `--exempt`; the distributions report; the gate's JSON | new, 253 lines |
| `tests/midoffice/test_prices.py` | 3 new tests; `Dividend` and `Split` imported | 24, 29, 656-744 |
| `tests/midoffice/test_identity.py` | 5 new tests | 305-424 |
| `tests/midoffice/test_rebuild.py` | new: 8 tests | new, 197 lines |
| `claude/handoffs/BRIEF_midoffice06_rebuild.md` | new: the brief and its two amendments, verbatim | new, 165 lines |
| `claude/retros/RETRO_midoffice06_rebuild.md` | new: this retro | new |

Written outside the tracked files, all gitignored and all new in this piece -- nothing that was
in `data/` before it was overwritten or deleted. The gate's report, this piece's own, was written
twice: offline at 01:36, and again at 01:47 with the distributions' summary added.

- `data/external/polygon/sp500_identity_2022-01-03_2026-11-03_resolved.csv`, `_resolved.json`,
  `_resolved_members.csv`, `_resolved_fetch.csv` -- the table and plan after item 1. The
  midoffice05 files beside them are unchanged; the build without `--resolve` still writes
  their three CSVs byte for byte, and their JSON with one key more, an empty "resolutions".
- `data/external/polygon/sp500_rebuild_gate_2022-01-03_2026-11-03.json` -- the gate's report:
  every span's identity check, every split event's classification, the distributions' summary.
- In `data/external/polygon/cache/`: item 1's 111 reference answers; the fetch's 226 price
  files, 619 split-event answers and 9,105 share samples; 620 distribution answers (619 tickers
  and SPY) and SPY's raw bars and split events; and the diagnoses' -- the 16 records naming the
  spin-offs, Millrose's 2 and its February-March bars, and LEN's distributions for a window.
- `claude/scratch/midoffice06/` -- the scratch, 74 items, this retro's builder among them.

---

## Item 1: the named member-days, through the reference endpoint

`build_sp500_identity.py --resolve` puts each named member-day group to
`/v3/reference/tickers/{t}?date=` on the brief's hypotheses (`HYPOTHESES` in the script) and
takes only what the endpoint confirms (`identity.resolve_named`). Each ticker asked is looked
up on the first and last of the member's named dates; for each security named there, the
trading days it held the ticker are found by binary search on the endpoint's own answers, out
to the window's ends -- the whole window, per the brief's ruling 4. What the security is to the
member decides what is taken: the member itself where its run named nothing (DAY, KVUE, WBD);
the run's own security under another ticker or on earlier dates (Corpay under FLT, Super Micro
before its split); or a predecessor, taken only if it traded to within a halt of the successor's
first day (WestRock, Paramount, Bunge Limited), with a link the member map follows. The
hypothesis is judged apart from what is taken: "CDAY for DAY" holds only if DAY's security is
CDAY's in the table.

128 lookups, cached (111 asked over the network the first time; every not-found answer kept).
4,813 named member-days before, 0 after. All four guards still pass, over 634 rows now.

| Hypothesis | Named member-days | Verdict | Taken |
|---|---|---|---|
| CDAY for DAY | 506 (2024-02-01 to 2026-02-06) | confirmed: DAY's records name BBG005D7PF34, CDAY's | DAY 2024-02-01 to 2026-02-04 (Dayforce's last record) |
| DISCA for WBD | 1,125 (2022-04-11 to 2026-10-05) | not confirmed: WBD's records name BBG011386VF4, CIK 0001437107; DISCA's is BBG00HW3TQG4, CIK 0001024333 | WBD 2022-04-11 to 2026-10-06, as a security of its own |
| WRK for SW | 629 (2022-01-03 to 2024-07-05) | confirmed: WestRock Company, BBG008NXC572 | WRK 2022-01-03 to 2024-07-05; linked as SW's predecessor |
| FLT for CPAY | 558 (2022-01-03 to 2024-03-22) | confirmed: FleetCor, BBG000GPXKX9, Corpay's FIGI | FLT 2022-01-03 to 2024-03-22 |
| PARA for SKYD (VIAC asked too) | 901 (2022-01-03 to 2025-08-06) | confirmed: Paramount Global, BBG000C496P7 | VIAC 2022-01-03 to 2022-02-16, PARA 2022-02-17 to 2025-08-06; linked as SKYD's predecessor |
| Bunge Limited for BG | 160 (2023-03-15 to 2023-10-31) | confirmed: Bunge Limited, CIK 0001144519 | BG 2022-01-03 to 2023-10-31; linked as Bunge Global's predecessor |
| SMCI itself | 136 (2024-03-18 to 2024-09-30) | confirmed: BBG000MYZDJ3, as before its split | SMCI 2022-01-03 to 2024-09-30 |
| KVUE itself (not a brief candidate) | 798 (2023-08-25 to 2026-10-29) | confirmed: Kenvue Inc., BBG01C79X561 | KVUE 2023-05-03 to 2026-10-30 |

VIAC was asked beside PARA because PARA has no record on 2022-01-03: the endpoint dates the
change to 2022-02-17 and answers under both that day, which goes to PARA. Paramount's and Bunge
Limited's records run on to their successors' first days; each predecessor ends the day before.
WBD is the one hypothesis the endpoint refutes, and its member-days are accounted for anyway --
by WBD's own records on its own dates -- which is a reading of "only what the endpoint confirms
enters the table" for Jeff to accept or overrule.

The updated table: 634 rows, 603 securities. The plan: 599 securities, 629 spans under 619
tickers -- 113 of them new -- needing 226 price requests, 9,105 share samples not on disk and
619 split-event requests, 9,950 in all; spans by source, 514 panel, 105 events, 1 presumed, 9
reference. 56,882 member-days rest on spans not from the panel; 7 spans carry none.

---

## Item 2: the fetch

`rebuild.fetch` asks, per ticker, for its prices over the window, adjusted and raw, its split
events and -- since 01:38 -- its distributions; then the reference record at every share sample,
inside each security's own spans. Everything goes through `PolygonClient`'s cache, a not-found
answer cached as one, so a stopped run resumes and a finished one is offline. The script runs it
at the panel fetch's rate, 0.02 s between requests.

### The first run, stopped

Started at about 15:19 on 2026-11-09; stopped by Claude Code at about 15:30, the machine
critically low on memory, at step 5,000 of 13,825 ("5,000/13,825 AVB 668s"). Not a failure of
the command. The cache then held 7,950 files: 5,458 reference answers, all 619 split-event
answers and 1,239 price files.

### The restart, and the fetch's memory

Restarted at 01:15 on 2026-11-10 with three changes:

- **It logs its own memory** on every progress line, every 500 steps, and at the start: this
  process's working set, its peak, its committed bytes, and the machine's available physical
  memory (`memory()` in the script: `GetProcessMemoryInfo` and `GlobalMemoryStatusEx` through
  ctypes on Windows, `resource` elsewhere).
- **It no longer holds the prices.** `Fetched` kept every ticker's bars, adjusted and raw: about
  1.5 million `Bar` objects at 414 bytes each, about 560 MB (measured on 100 tickers: +90 MB
  private, +90 MB working set, 229,240 bars). The gate reads none of them and the build reads
  them back from the cache, so the fetch now counts them (`Fetched.bars`).
- **`--fetch-only`**, so that the restarted fetch would not run the old split rule at its end;
  the checks ran after it, `--offline`, on the new rule.

The machine at the restart, no fetch running: 472 MB available of 26,420 MB; 69.8 GB committed
of a 94.3 GB limit. Committed by process: Chrome 17,176 MB over 53 processes, WSL 4,058, PyCharm
3,451, explorer 1,604, node 1,407, three Claude processes 2,690 between them, Radeon 973, Edge
935; 49,056 MB over every process. The restarted fetch ran at 29 to 47 MB (peak 47; 505 to 519
MB committed, about 470 of it numpy's OpenBLAS buffers, imported through `identity.py`), while
the machine's available memory moved between 343 and 3,488 MB. It finished in 1,229 s: 6,640
requests over the network, none failed. The pressure at 15:30 was the machine's; the fetch's
own 560 MB of bars, now gone, added to it.

The distributions run, finished at 01:47, added 622 requests -- 619 tickers' distributions, and
SPY's raw bars, split events and distributions -- in 105 s, none failed, at a peak of 66 MB.

---

## The split rule, changed before the gate ran

The brief's rule -- the count before, times the ratio, nearer the next sample than the unsplit
count -- would have stopped on every spin-off: Polygon records one as a fractional split, to
adjust the parent's price history, and the parent's share count does not move across it. The
new rule classifies every split event by two independent tests:

1. **Is the ratio one of whole numbers?** Polygon writes every event in whole numbers -- a
   spin-off as 1,000 to 1,281, DuPont's as 100 to 239 -- so the test is read in lowest terms:
   whole where no term is over 100 (`LARGEST_TERM`, a parameter, `--largest-term`). Over the
   window's 55 events the splits' largest terms run from 2 to 50 (Chipotle's 50-for-1) and the
   adjustments' from 239 to 1,327; any bound from 50 to 238 separates them.
2. **Do the share samples either side move by it, within 10%** (`TOLERANCE`, `--tolerance`)?
   The last sample before the execution date and the first after it, the security's own across
   its tickers. "follow": within 10% of the ratio; "unchanged": within 10% of no change;
   "either": within 10% of both -- which a ratio between about 0.82 and 1.22 allows, so the
   counts cannot tell; "neither".

| Ratio | Counts | Kind | Applied to the carried count | The gate |
|---|---|---|---|---|
| whole | follow | split | yes | passes |
| fractional | unchanged | price-only | no -- the adjusted prices carry it | passes |
| whole | unchanged or neither | contradiction | no -- the samples rule | stops (DuPont's exempt) |
| fractional | follow or neither | contradiction | no | stops (DuPont's exempt) |
| whole or fractional | no sample on a side, or "either" | the ratio's kind, untested | a split's, yes | passes, reported |

A fractional ratio the counts move by neither it nor nothing is a contradiction: Jeff's ruling
of 01:38, on the reading this piece put to him. No event in the window is one.

**The third test.** The events carry no field saying what kind each is: their fields are
`execution_date`, `id`, `split_from`, `split_to` and `ticker`. The id's first letter is E or P,
undocumented; it lines up with the ratio test on 54 of the 55 events -- every fractional one is
P -- but not on W. R. Berkley's three-for-two of 2024-07-11, a split the counts follow, which is
P. Not used as a test.

### What it found

55 events inside the securities' spans (5 more under their tickers in the window fall outside
the span of the security that uses the ticker):

- **36 splits**, every one followed by its next sample within 10%: from W. R. Berkley's
  three-for-two of 2022-03-24 (1.501 times) to DuPont's one-for-three of 2026-06-24 (0.330) and
  Amcor's one-for-five of 2026-01-15 (0.200). Among them NVDA 10 (9.840), CMG 50 (50.086), AMZN
  20 (19.995), WMT 3 (2.991), AVGO 10 (10.034), SMCI 10 (9.999, six days after).
- **8 price-only adjustments**, 6 tested and 2 untested, each named below.
- **11 contradictions**, every one a whole-number ratio the counts stay unchanged across, below.
- **The DuPont exemption went unused.** DD's 2.39 of 2025-11-03 is price-only (the counts
  1.001 times either side) and its 0.33 of 2026-06-24 is a split (408,923,857 to 135,019,401).

### The price-only adjustments, each a spin-off's ex-date

Each fractional event is a real ex-date: the raw price gaps down on its execution date by about
one less the ratio's reciprocal, and the adjusted series steps by exactly the ratio there. Each
spin-off is a public-record hypothesis, put to the reference endpoint: the child's record on
the parent's execution date and a week after.

| Event | Ratio | Counts either side | Raw opening gap; one less 1/ratio | Distribution | The endpoint |
|---|---|---|---|---|---|
| GE 2023-01-04 | 1.281 | 0.993 | -19.50%; 21.94% | GE HealthCare (GEHC) | record on the day, BBG01BFR8YV1; listed 2022-12-15 |
| LH 2023-07-03 | 1.164 | 1.000 | -14.58%; 14.09% | Fortrea (FTRE) | record on the day, BBG01GZF9VY6; listed 2023-06-16 |
| MMM 2024-04-01 | 1.196 | 1.002 | -14.16%; 16.39% | Solventum (SOLV) | record on the day, BBG018YZH6T3; listed 2024-03-26 |
| GE 2024-04-02 | 1.253 | 1.004 | -19.86%; 20.19% | GE Vernova (GEV) | record on the day, BBG013G17W31; listed 2024-03-27 |
| LEN 2025-01-21 | 1.033 | 0.972, untested | -3.91%; 3.19% | not named | Millrose (MRP): no record on 2025-01-21 or 01-28; listed 2025-02-05 |
| FTV 2025-06-30 | 1.327 | 0.999 | -27.30%; 24.64% | Ralliant (RAL) | record on the day, BBG01V4YRP24; listed 2025-06-25 |
| HON 2025-10-30 | 1.061 | 1.000, untested | -5.21%; 5.75% | Solstice Advanced Materials (SOLS) | record on the day, BBG01X5744P5; listed 2025-10-20 |
| DD 2025-11-03 | 2.39 | 1.001 | -58.74%; 58.16% | Qnity Electronics (Q) | record on the day, BBG01TRL73S5; listed 2025-10-27 |

---

## Item 3: the gate stops

### Identity: 18 of 629 spans fail

The check reads each span's first share sample: the reference record under the span's ticker on
its first day must name the security the table expects. 611 pass. Of the 18 that fail, 15 fail
on that first day alone, every later sample naming the expected security, and 3 never pass.
5,593 member-days sit on the 18 spans, every one of them on the 15.

Per failing span: the security expected; what its samples name, with their dates; what the
events and the records said; the correction proposed; the fetch it needs.

**The 11 spin-offs, their spans starting at the list date.** The events give one event, a
ticker change on the list date, which the table takes as the span's first day ("the first of
its events: the start of Polygon's record of it, not necessarily of the ticker"). The records,
from the second sample on, name the expected security with that list date. But no price under
the ticker comes before the regular-way start -- the distribution -- and the endpoint has no
record on the list date itself. Where a record on the first priced day is in the cache (fetched to
name the spin-offs), it names the expected FIGI.

| Span | Member-days | Expects | Samples name | List date; first price | Record on the first priced day |
|---|---|---|---|---|---|
| GEHC from 2022-12-15 | 962 | BBG01BFR8YV1 GE HealthCare | 2022-12-15 none; 2023-03-20 on (16) the expected | 2022-12-15; 2023-01-04 | BBG01BFR8YV1, cached |
| SOLV from 2024-03-26 | 652 | BBG018YZH6T3 Solventum | 2024-03-26 none; 2024-06-26 on (11) | 2024-03-26; 2024-04-01 | BBG018YZH6T3, cached |
| GEV from 2024-03-27 | 651 | BBG013G17W31 GE Vernova | 2024-03-27 none; 2024-06-27 on (11) | 2024-03-27; 2024-04-02 | BBG013G17W31, cached |
| VLTO from 2023-09-27 | 776 | BBG019Q32XJ9 Veralto | 2023-09-27 none; 2023-12-27 on (13) | 2023-09-27; 2023-10-02 | not fetched |
| MBC from 2022-12-09 | 2 | BBG01B9G42Y3 MasterBrand | 2022-12-09 none; 2023-03-14 on (16) | 2022-12-09; 2022-12-15 | not fetched |
| AMTM from 2024-09-24 | 59 | BBG01NY7B734 Amentum | 2024-09-24 none; 2024-12-23 on (9) | 2024-09-24; 2024-09-30 | not fetched |
| SNDK from 2025-02-13 | 234 | BBG01R388JG1 Sandisk | 2025-02-13 none; 2025-05-15 on (7) | 2025-02-13; 2025-02-24 | not fetched |
| Q from 2025-10-27 | 252 | BBG01TRL73S5 Qnity | 2025-10-27 none; 2026-01-28 on (5) | 2025-10-27; 2025-11-03 | BBG01TRL73S5, cached |
| SOLS from 2025-10-20 | 36 | BBG01X5744P5 Solstice | 2025-10-20 none; 2026-01-21 on (5) | 2025-10-20; 2025-10-30 | BBG01X5744P5, cached |
| FDXF from 2026-05-27 | 109 | BBG01ZPB91C4 FedEx Freight | 2026-05-27 none; 2026-08-26 on (2) | 2026-05-27; 2026-06-01 | not fetched |
| HONA from 2026-06-15 | 90 | BBG020QYV152 Honeywell Aerospace | 2026-06-15 none; 2026-09-15 on (2) | 2026-06-15; 2026-06-29 | not fetched |

Every span runs to 2026-11-03. SOLS's ticker also carries 344 days of another security's prices
before its span, which the span leaves out.

- **Correction:** start each spin-off's span at its first price under the ticker, not at its
  first event, and take the first share sample there. Every member-day of the 11 falls on or
  after the first price, so no member-day moves.
- **Fetch:** the record on the first priced day for the 6 not cached -- VLTO, MBC, AMTM, SNDK,
  FDXF, HONA: 6 requests. The plan's other samples stand.

**EG, from 2023-07-10, 835 member-days, expects CIK0001095073 (Everest Group).** Samples:
2023-07-10, a record with the name and the list date 1995-10-03 but neither FIGI nor CIK, so it
keys as "NAME:Everest Group, Ltd."; from 2023-10-05 on (14), CIK 0001095073. Events: Everest
Group, no FIGI, CIK 0001095073, ticker changes to RE on 2003-09-10 and to EG on 2023-07-10.
Records: list date 1995-10-03 throughout. The table: RE 2022-01-03 to 2023-07-07 (events), EG
from 2023-07-10 (panel).

- **Correction:** a record carrying no identifier is not evidence of another security; read the
  identity at the first sample that carries one -- 2023-10-05's, the expected CIK, whose name
  and list date the change-day record shares.
- **Fetch:** none needed; 1 request, 2023-07-11's record, would narrow the gap.

**BG, from 2023-11-01, 754 member-days, expects CIK0001996862 (Bunge Global SA).** Samples:
2023-11-01, "Bunge Global SA" under CIK 0001144519 -- Bunge Limited's; from 2024-02-02 on (12),
CIK 0001996862. Events: Bunge Global SA, no FIGI, CIK 0001996862, one event, the ticker change
to BG on 2023-11-01. Records: list date 2001-08-02 under both CIKs. The table: Bunge Limited
(CIK0001144519) under BG 2022-01-03 to 2023-10-31, from item 1's resolution, linked before
Bunge Global. The share count either side of the boundary: 145,290,000 and 145,290,000, 1:1.

- **Correction:** a reorganisation -- Bunge's redomicile, one security with one return series,
  keyed by the successor and the predecessor linked, as the brief's ruling 1 has it. Date the
  boundary by the identifier the table keys on, the CIK, not by the name, which changed first:
  between 2023-11-01 and 2024-02-02.
- **Fetch:** a binary search on the records between the two, about 6 requests.

**VYLR, 2026-10-01 to 2026-11-03, 24 member-days, expects BBG024CQR6X9 (Vylor Inc.).** Samples:
2026-10-01, BBG024W5VJC3 under CIK 0002128626; 2026-11-03, BBG024CQR6X9 under the same CIK.
Events: Vylor Inc., BBG024CQR6X9, CIK 0002128626, one event on 2026-10-01. Records: list date
2026-10-01. Prices from 2026-10-01, unbroken.

- **Correction:** a new FIGI under the same ticker and CIK, the prices unbroken, is the
  reorganisation's pattern: one security keyed by the later FIGI, the earlier linked, the
  boundary dated by binary search and its share counts sampled either side.
- **Fetch:** about 5 requests.

**COHR, from 2022-09-08, 157 member-days, expects BBG000BLW102 (Coherent Corp.).** Samples:
2022-09-08, BBG000GS6QB4 with no CIK, "Coherent Corp. Common Stock"; from 2022-12-07 on (17),
BBG000BLW102 under CIK 0000820318. Events: Coherent Corp., BBG000BLW102, CIK 0000820318, one
event -- the ticker change to COHR on 2022-09-08. Records: list date 1987-10-02 on every one
naming BBG000BLW102. The prices under COHR: 124 days before the span, to 2022-07-01, and 1,043
in it.

- **Yes, II-VI is the company that took the ticker.** The records' CIK and list date, 1987, are
  II-VI Incorporated's: it bought Coherent, Inc. on 2022-07-01, renamed itself Coherent Corp.
  and traded as COHR from 2022-09-08. The 124 days before the span are old Coherent's, and
  BBG000GS6QB4 is most likely old Coherent's FIGI, carried on the change-day record; the table
  has no row for it.
- **Correction:** add II-VI's span under IIVI, from the window's start to 2022-09-07, to
  BBG000BLW102 -- Coherent's own history before it took the ticker, now missing -- and read the
  span's identity at its second sample, the change-day record accounted for once a COHR record
  from before 2022-07-01 shows BBG000GS6QB4 to be old Coherent's.
- **Fetch:** IIVI's prices, adjusted and raw, its split events, distributions and events (5
  requests), its share samples (about 4), and one COHR record before 2022-07-01: about 10.

**The three that never pass, none carrying a member-day.**

- **S600, 2023-11-20 to 2026-11-03, expects BBG0019LSQ49 (First Republic Bank).** Samples: no
  record at any of 14. Events: none under S600; the span comes from First Republic's own
  ("began trading as S600 on 2023-11-18"), after FRC to 2023-04-28 (panel) and FRCB from
  2023-05-03 to 2023-11-17 (events). Prices: none under S600 in the window. **Correction:** drop
  the span. **Fetch:** none.
- **CEGV, 2022-01-19 to 2022-02-01, expects BBG014KFRNP7 (Constellation Energy).** Samples: no
  record at either. Events: none under CEGV; the span comes from Constellation's ("the first
  of its events, 2022-01-19"; "traded as another ticker from 2022-02-02"). Prices: none under
  CEGV. **Correction:** drop the span; Constellation's history starts at CEG's listing,
  2022-02-02. **Fetch:** none.
- **CRHC, 2022-01-03 to 2023-09-22, expects CIK0000849395 (CRH plc).** Samples: 2022-01-03 to
  2022-10-04 (4), Cohn Robbins Holdings Corp., CIK 0001818212, listed 2020-10-30; 2023-01-04 to
  2023-09-22 (4), no record. Events: CRH's, giving CRHC from 2021-09-23 and CRH from
  2023-09-25. Prices: 216 days under CRHC, 2022-01-03 to 2022-11-09 -- Cohn Robbins', a SPAC
  that dissolved. Every record refutes the events' CRHC tenure. **Correction:** drop the span;
  CRH's history starts at 2023-09-25. **Fetch:** none. (433 days of prices under CRH before
  2023-09-25, most likely CRH's American depositary receipt, a different security, are not
  proposed.)

**In all:** the corrections need about 27 requests -- 6 first-day records, about 6 for Bunge,
about 5 for Vylor, about 10 for II-VI and Coherent -- plus a regenerated plan, and the gate run
again.

### What else the samples say

The gate reads first samples only, as the brief has it. Every sample of every span, read from
the cache: 24 spans have one naming another identifier than the table's -- the 18 above, and 6
whose later samples do. Each of the 6 is one sample, on or near the span's last day, naming
another FIGI under the expected CIK, at an acquisition or a reorganisation's boundary: EA's of
2026-08-04 (BBG000K1X329), OKE's of 2026-09-10 (BBG000JBSHW1), JNPR's of 2025-07-01
(BBG000BWS318), BLK's of 2024-10-01 (BBG000CN81W9, under BlackRock's old CIK 0001364742), XOM's
of 2026-07-01 (BBG000H5HTR8) and DISCA's of 2022-04-04 (BBG000CHWP52). The CIK holds in each, and
nothing is proposed.

### Reorganisations: the share counts either side of each boundary

The brief's ruling 1 asks for the share count sampled either side of each boundary and the
ratio reported.

| Ticker | Predecessor, its last sample | Successor, its first sample | Ratio |
|---|---|---|---|
| BLK | BBG000C2PW58, 2024-10-01: 148,130,000 | BBG01PSW2WN4, 2024-10-02: 148,130,000 | 1.0000 |
| J | BBG000BMFFQ0, 2022-08-26: 127,610,000 | BBG019C1BQR4, 2022-08-29: 127,610,000 | 1.0000 |
| LH | BBG000D9DMK0, 2024-05-17: 84,290,000 | BBG01MMT6PL7, 2024-05-20: 84,290,000 | 1.0000 |
| BG | CIK0001144519, 2023-10-31: 145,290,000 | CIK0001996862, 2023-11-01: 145,290,000 | 1.0000 |
| CCL | BBG000BF6LY3, 2026-05-06: 1,239,000,212 | CIK0000815097: none on 2026-05-07; 1,369,649,119 on 2026-08-07 | 1.1054 |
| OKE | BBG000BQHGR6, 2026-09-10: 630,369,667 | BBG024TZWVN1: none on 2026-09-11 or 2026-11-03 | none |
| XOM | BBG000GZQ728, 2026-07-01: 4,144,947,162 | BBG023CY9MM1: none on 2026-07-02 or 2026-10-01 | none |

Four are 1:1. Carnival's is not -- 10.5% more shares, its dual listing folding in, as the ruling
supposed -- and its successor's first count comes three months after the boundary. OKE's and
XOM's successors carry no share count at any sample in the window. For the links that are not
reorganisations, the counts either side are reported for completeness: WestRock to Smurfit
Westrock, 258,149,999 to 261,100,000 (1.0114); Paramount to Paramount Skydance, 633,522,201 to
1,064,544,631 on the successor's second sample (1.6804).

### Splits: 11 contradictions, one pattern

Every contradiction is a whole-number split from 2025-05-22 on whose first sample after still
reads the pre-split count; in nine, the sample after that follows the ratio.

| Split | Before | First after | Follows at | Days after the split |
|---|---|---|---|---|
| FAST 2-for-1, 2025-05-22 | 573,450,000 (2025-04-09) | 573,620,638 (2025-07-11) | 1,147,636,617, 2.001 (2025-10-09) | 140 |
| ORLY 15-for-1, 2025-06-10 | 57,240,000 | 56,989,455 | 848,496,680, 14.823 (2025-10-09) | 121 |
| IBKR 4-for-1, 2025-06-18 | 108,930,000 | 108,958,830 | 445,246,976, 4.087 (2025-10-09) | 113 |
| NOW 5-for-1, 2025-12-18 | 208,000,000 (2025-10-09) | 207,678,139 (2026-01-09) | 1,036,155,572, 4.982 (2026-04-13) | 116 |
| TPL 3-for-1, 2025-12-23 | 22,987,326 | 22,979,410 | 68,941,554, 2.999 (2026-04-13) | 111 |
| BKNG 25-for-1, 2026-04-06 | 32,233,815 (2026-01-09) | 31,673,346 (2026-04-13) | 774,878,436, 24.039 (2026-07-14) | 99 |
| CVNA 5-for-1, 2026-05-08 | 142,993,769 (2026-04-13) | 143,257,677 (2026-07-14) | 719,916,415, 5.035 (2026-10-12) | 157 |
| KLAC 10-for-1, 2026-06-12 | 131,076,611 | 130,627,521 | 1,305,006,492, 9.956 (2026-10-12) | 122 |
| CRWD 4-for-1, 2026-07-02 | 253,614,090 | 254,564,820 | 1,026,129,839, 4.046 (2026-10-12) | 102 |
| MNST 2-for-1, 2026-08-11 | 978,008,110 (2026-07-14) | 979,525,882 (2026-10-12) | not in the window (2026-11-03: 979,525,882) | -- |
| APH 2-for-1, 2026-09-03 | 1,230,234,445 (2026-07-14) | 1,232,983,457 (2026-10-12) | not in the window (2026-11-03: 1,232,983,457) | -- |

**The likely cause: Polygon's share counts change in 2025 Q3.** They are rounded to 10,000
shares in 7,866 of the 7,982 samples through 2025 Q2 (98.5%), and exact in 536 of 569 in 2025 Q3
and in 93.8% to 95.9% of each quarter after. Every split whose next sample predates the change
was followed by it -- SMCI's six days later -- so the old counts moved with the split. The
exact counts behave as a filing's cover-page figure would, waiting for the next filing, so the
first sample after a split can predate the one that shows it; that is an inference from the
pattern, not something the records say. Three splits since the change were followed by their
next sample (NFLX 2025-11-17, AMCR 2026-01-15, DD 2026-06-24).

**What the brief's rule would do with them.** "A split the next share sample disagrees with is
not applied -- the samples rule there." Here the samples are stale, so the carried count would
pair post-split prices with pre-split counts for 99 to 157 days -- BKNG's capitalisation a 25th
of the truth for 99 days -- and for MNST and APH to the window's end. Exactly what the guard
exists to prevent, outside its reach, since a contradiction is not counted as a split.

**Correction proposed:**

- Apply every whole-number split on its execution date, as item 4 says.
- A sample after the split still within the tolerance of the pre-split count is a count from a
  filing made before the split -- stale, not a disagreement -- and does not reset the carried
  count; the first sample that follows the ratio resets it, as any sample does.
- The count test then reads: the counts follow a whole-number split if a later sample in the
  span follows it, every sample between staying at the pre-split count. Nine of the eleven do;
  MNST's and APH's, with no filing after them inside the window, would be untested and applied
  by the ratio alone.
- **Fetch:** none.

---

## Dividends

Fetched for every ticker in the plan and for SPY, each ticker's whole history in one request
(`PolygonClient.dividends`, `/v3/reference/dividends`, 1,000 to a page; a ticker needing a
second page is refused rather than cut), cached; a security's are those under its tickers
inside its spans, since the events under one ticker can be several securities'.

- **8,234 distributions over 467 of the 599 securities; 132 paid none.** By type: 8,169 regular
  (CD), 65 special (SC); no capital gains. All in USD. SPY: 19 in the window, $32.89 a share, all
  regular.
- **The specials**, all 65 listed in the appendix: TDG's $18.50, $35, $75 and $90; TPL's $20
  and $10; COST's $15; CME's annual variable dividends; the energy companies' variable ones.
  Polygon's types are not always right: CRH's quarterly $0.35 of 2024 is typed SC, and DVN's
  fixed $0.22 SC.
- **The split basis.** Each amount is per share as paid at the time: across the 19 splits with a
  regular distribution on both sides, the last before over the first after is the split ratio
  -- NDAQ 3.000, TECH 4.000, SRE 2.000, LRCX 10.000, AMCR 0.200, DD 0.333 -- or the ratio less a
  raise (WMT 2.747 on a three-for-one, TSCO 4.783 on a five-for-one, NVDA 4.0 on a ten-for-one
  with its 150% raise). So a total return puts each amount on the adjusted prices' basis by the
  ex-date's ratio of adjusted to raw close.
- **The same ticker and ex-date twice: 69 pairs**, all but one a two-part payout -- a fixed and
  a variable dividend (DVN, FANG, PXD, CTRA, CHK, COP, EXE) or a regular and a special (F, WRB,
  FCX, HST, KIM, LYB, TPL, CME, PGR's annual). One is an exact duplicate: TEL's $0.65 of
  2024-11-22, recorded twice; a total return keeps one.
- **How much the price returns miss.** A security's yield over its spans -- its distributions
  over its mean traded close, a year -- over the 587 with half a year of prices: median 1.40%,
  p75 2.80%, p90 3.95%, the highest 7.78% (Pioneer), 55 above 4% and 5 above 6% (Pioneer,
  Altria 7.41%, LyondellBasell 7.09%, Verizon 6.56%, Ford 6.13%); 127 paid none. SPY's is 1.27% a
  year on its mean close of $539.04. Every ex-date reads as a loss of the dividend now, so the
  high yielders' factor and industry means are biased down, and their specific variance carries
  a small regular addition.
- **The docstrings.** `prices.py`'s `MARKET_PROXY` said the adjusted prices are "split- *and
  dividend-* adjusted ... so they are total returns" and that "SPY's adjusted series is a total
  return"; `polygon.py`'s `market_bars` said SPY's adjusted bars "are a total return". Both now
  say the adjusted prices are split-adjusted only -- over 2022-2026 the adjusted close equals the
  traded one on every day for Altria, Verizon, AT&T and Exxon -- that the two sides match as
  price returns, and that total returns need the distributions.

Building the total returns is the build's, and waits for the gate; so does the comparison,
which takes the identity's effect first and the dividends' after, each on its own.

---

## LEN, 2025-01-21

Jeff's question: is the ex-date gap the 1.033 should have been at Millrose's distribution
instead? If so, Polygon misdated the adjustment and the adjusted series is wrong on both days.

- **It is not there.** Millrose lists on 2025-02-05 and trades from 2025-02-07. LEN's raw
  opening gaps on 2025-02-05, 02-06, 02-07, 02-10 and 02-11 are within 0.17 points of three
  homebuilders' (DHI, PHM, NVR) -- +0.17, +0.05, 0.00, +0.01 and +0.04 points. No gap of
  3.2%, or of any size, is LEN's own in February.
- **The gap is where Polygon put it, and larger.** On 2025-01-21 LEN opened 3.91% down while the
  three opened 1.99% up: 5.90 points below them at the open, 4.63 below by the close. The step
  adjusts 3.2 of those points, so LEN's adjusted return that day is -2.83% against a raw -5.93%.
- **Nothing on record explains it.** No distribution under LEN that day -- its only ones from
  2024-10-01 to 2025-04-30 are the regular $0.50 -- and no Millrose record until 2025-02-05.
  Millrose's own numbers: 154,180,000 class shares (166,000,265 weighted), closing $26.74 on
  2025-02-07 and $23.01 on 2025-02-10.

So Polygon did not move Millrose's adjustment two weeks early, and the adjusted series is not
wrong on two days. Whether the 2025-01-21 step is right at all the prices cannot settle; as
asked, nothing is corrected.

---

## The old panel's split staleness, corrected

Earlier in this piece the old panel's split staleness was put at 5,184 member-days across 45
split events, sampled yearly at six dates. That counted the eight spin-offs too, whose share
counts do not move. Under the classification: 30 splits, 3,529 member-days; the 7 of this
piece's contradictions inside the old panel's tickers (FAST, ORLY, NOW, BKNG, KLAC, MNST, APH),
real splits whose counts lag, 667; the 8 price-only events, 988, which were never stale. **The
old panel's split staleness is 4,196 member-days across 37 splits.** `docs/FACTOR_MODEL.md`
(lines 777-786) says the counts are "sampled every 63 trading days", puts the figure at 896
security-days and 621 member-days an estimated period reads, and says "DD's steps of 2.39 and
0.33 are not split ratios": the panel was sampled yearly, and DuPont's 0.33 is a split the
counts follow. The doc is not edited here (a decision below).

---

## Key decisions

1. **The fetch stopped holding prices**, which the gate does not read, and the restart ran
   `--fetch-only`, so that the old split rule never ran on the full data; the new one ran
   offline after it.
2. **"Whole numbers" read in lowest terms, capped at 100**, since Polygon writes every event in
   whole numbers; and **"either" counted as untested**, since within 10% of both the ratio and
   no change the counts cannot tell -- otherwise LEN's and HON's would be judged by noise. Jeff
   ruled both at 01:38, and the fractional-and-neither case with them.
3. **DuPont's exemption kept**, as the brief has it, and reported unused.
4. **Each spin-off named by putting the public-record candidate to the endpoint** -- the child's
   record on the parent's execution date -- in the brief's spirit for item 1: only what the
   endpoint confirms is named. LEN's is not.
5. **The dividend fetch takes each ticker's whole history** in one request, as the split events
   do, and filters to the security's spans at use; the market series is fetched as any ticker
   is -- prices, split events and distributions -- so the build has SPY's raw bars too.
6. **`market_bars`' docstring corrected with `prices.py`'s**, the same false claim; and
   `splits()`' own docstring, written earlier in this piece, which called DuPont's 0.33 not a
   split ratio.
7. **The gate re-ran with the distributions fetch**, on the same answers, so its JSON carries
   the distributions' summary; its verdict is unchanged.
8. **Stopped as the brief directs**: no extra fetch for the corrections, no panel, no build, no
   docs edit; every correction is proposed, with its fetch, for Jeff.

## Decisions left for Jeff

1. **The identity corrections**, as above: start each spin-off's span at its first price;
   read an identifier-less record (EG) as no evidence and take the next; Bunge and Vylor as
   reorganisations, their boundaries dated by the identifier the table keys on; II-VI's IIVI
   history added to Coherent; S600, CEGV and CRHC dropped. About 27 requests, a regenerated
   plan, and the gate again.
2. **The split lag rule**, as proposed -- a stale sample after a split does not reset the
   carried count, the first that follows does, and MNST's and APH's untested at the window's
   end -- or the brief's rule, under which the samples rule there.
3. **WBD:** its member-days are its own security's, the hypothesis refuted. Accept, or overrule?
4. **Reorganisations' counts:** Carnival's is 1.1054 and its successor's first count comes
   three months after the boundary -- carry the predecessor's until then, or take the
   successor's from the boundary? OKE's and XOM's successors have no count in the window --
   carry the predecessor's, 1:1, on the ruling for a holding-company reorganisation, unconfirmed
   by any sample?
5. **TEL's duplicate distribution:** keep one.
6. **LEN 2025-01-21:** left as Polygon has it, for now.
7. **`docs/FACTOR_MODEL.md`'s split-staleness paragraph:** correct it -- yearly samples, 4,196
   member-days over 37 splits, DuPont's 0.33 a split -- now, or with the build?

---

## Tests

2,269 passed, 1 skipped, 19 warnings, in 74.33 s, with `--ignore=tests/test_market_data.py` --
a file that cannot collect here, as before this piece (it looks for `../core_repo/src`, which
is not on this machine; the sibling is `mctheory-core`). midoffice alone: 818 passed, 1 skipped.
Before this piece: 2,253 and 802. 16 new tests:

`tests/midoffice/test_prices.py`, 3:

- `test_a_not_found_answer_is_cached_and_raised_again_without_asking` -- a 404 is an answer: a
  resumed fetch asks for it once.
- `test_split_events_come_back_as_dated_ratios_and_are_asked_for_once` -- ratios, dates and
  order; one request; a second page refused.
- `test_cash_distributions_come_back_dated_and_typed_and_are_asked_for_once` -- regular and
  special kept, in ex-date order, with amount, currency, type, frequency and pay date; one
  request; a second page refused.

`tests/midoffice/test_identity.py`, 5:

- `test_a_missing_member_is_named_by_its_own_ticker_and_its_span_found_by_bisection`
- `test_an_identity_the_endpoint_confirms_enters_though_the_hypothesis_does_not`
- `test_a_predecessor_counts_only_if_it_traded_to_within_a_halt_of_its_successor`
- `test_on_a_change_of_ticker_the_new_one_takes_the_day_and_a_predecessor_ends_before`
- `test_a_confirmed_span_is_cut_where_the_security_already_has_a_row`

`tests/midoffice/test_rebuild.py`, new, 8:

- `test_the_fetch_asks_once_for_each_answer_and_a_second_run_is_offline` -- prices, split
  events, distributions and samples each asked once, the market's too; a second run asks
  nothing; bars counted, not held.
- `test_a_span_whose_first_sample_names_another_security_or_none_stops_the_gate`
- `test_a_split_is_a_whole_ratio_the_counts_follow_and_a_spin_off_a_fractional_one_they_leave`
  -- split and price-only, and both contradictions stopping the gate, DuPont's exempt.
- `test_counts_that_move_by_neither_the_ratio_nor_nothing_confirm_neither_kind` -- and the
  tolerance as a parameter.
- `test_where_the_counts_cannot_test_an_event_its_ratio_alone_classifies_it` -- no sample after,
  and "either"; both untested, neither stopping the gate.
- `test_a_whole_number_ratio_is_small_in_lowest_terms_however_it_is_written` -- 2,000 to 1,000
  is whole, 1,000 to 1,281 and 100 to 239 are not, the cap a parameter.
- `test_a_split_outside_the_securitys_span_under_the_ticker_is_another_securitys`
- `test_the_plan_reads_back_as_written`

Every changed Python file parses and is ASCII. Line endings, counted in bytes: CRLF throughout
in `polygon.py`, `identity.py`, `rebuild.py`, `build_sp500_identity.py`,
`fetch_sp500_rebuild.py`, `test_prices.py`, `test_identity.py` and `test_rebuild.py`; LF
throughout in `prices.py`, as its working copy was before the edit (git normalises it to the LF
it holds). ruff over the changed files: 7 findings, every one at HEAD before this piece --
`polygon.py`'s unused `Iterable` and `Sequence`, `prices.py`'s import order, `test_prices.py`'s
import order and three unused imports.

---

## Failures and debugging trail

- **Heredocs with apostrophes** failed in Git Bash ("unexpected EOF while looking for matching
  `''"); patches went into scripts written with the Write tool instead.
- **A broken f-string** in the first resolution patch; rewritten.
- **PARA/SKYD and BG resolved to 0 member-days at first**: their predecessors' records ran on to
  the successors' first days, and VIAC and PARA overlapped on 2022-02-17. Fixed by `_settled`
  (the new ticker takes the change day) and cutting a predecessor at its successor's first day.
- **SMCI's reference row overlapped its events row**; fixed by `_trimmed`.
- **The link's evidence named VIAC's security for PARA**; it now names the latest row. The
  `_resolved.json` written before the fix keeps the old wording ("WestRock Company traded to
  2024-07-05"); the code's is "traded as WRK to".
- **E501 and two F541** in the new script; fixed. **New files written LF**; normalised to CRLF.
- **The first fetch was stopped for low memory** at step 5,000 (above). Not restarted until
  Jeff asked.
- **The memory reader's first try** passed the pseudo-handle without argument types and
  overflowed (`ctypes.ArgumentError: int too long to convert`); `argtypes` set.
- **A grep-based CRLF count was wrong**: it reported `prices.py` as CRLF; counted in bytes it is
  LF, as git's warning said. The checks above are byte counts.
- **The gate-summary helper broke on a patch** that put a newline inside an f-string; rewritten.
  Its "another company" label for EG and BG is wrong -- they are the company, under a missing or
  older identifier -- and the trace and detail outputs are the ones relied on.
- **The full suite would not collect** without `--ignore=tests/test_market_data.py`, as before.

## Problems found, not fixed

1. **The identity failures and split contradictions above**, proposed and not fixed.
2. **Polygon's share counts lag splits from 2025 Q3**, when they change from rounded to exact.
   If the exact counts are filings' figures, as they behave, every count since lags its
   company's last filing, at a split or not.
3. **OKE's and XOM's successors have no share count** at any sample in the window.
4. **Polygon's events gave CRH a CRHC tenure** that every record refutes (Cohn Robbins).
5. **Six last-day records name another FIGI under the right CIK** (EA, OKE, JNPR, BLK, XOM,
   DISCA).
6. **Dividends:** TEL's duplicate; types not always right (CRH's quarterly and DVN's fixed typed
   SC).
7. **LEN's 2025-01-21 step** is on a real gap of 5.90 points against its peers, of which it
   adjusts 3.2; nothing on record explains either.
8. **`docs/FACTOR_MODEL.md` lines 777-786** -- the 63-day grid, 896 and 621, DuPont's 0.33 --
   are wrong.
9. **`tests/test_market_data.py`** looks for `../core_repo/src`, not on this machine.
10. **Seven ruff findings at HEAD** in files this piece touched, left alone.

## Scratch

`claude/scratch/midoffice06/` (gitignored), 74 items, every script and output named here --
among them `bars_memory.py`, `split_survey.py`, `price_only_check.py`, `name_spinoffs.py`,
`len_event.py`, `len_millrose.py`, `identity_trace.py`, `identity_detail.py`,
`all_samples_scan.py`, `split_lag.py`, `boundary_ratios.py`, `dividends_check.py`,
`old_split_stale.py`, `gate_analysis.py`, the patch scripts, `extract_brief.py`,
`extract_amendments.py`, `amend_brief.py`, and the fetch logs `fetch_run1.txt`,
`fetch_run2.txt`, `fetch_run3_dividends.txt`, `gate_run.txt` -- plus `build_retro.py`, which
assembled this retro. Their outputs are copied into the appendix.

## Git

No commits; Jeff commits (Rule 43). Nothing is staged. The `git add` below was checked with
`git add --dry-run`, and it names every changed path.

## Open items for Chat

- The corrections and the lag rule, then the gate again; then items 4 to 6 -- the panel on
  securities, its total returns, and the model against the old panel, identity's effect first
  and the dividends' after.
- Whether a security that stops being sampled -- OKE's and XOM's successors -- needs a source
  for its share count beyond the reference endpoint.
- The span-plus-look-back cut, kept as a parameter for future panels, is not built yet: the
  plan's spans are the whole window, as the ruling asked for this one.

---

## Appendix: outputs, verbatim

Every output the body rests on, from `claude/scratch/midoffice06/`, in full unless said otherwise. The session scratchpad's path is shortened to `<scratch>`.

### fetch_run1.txt -- the first fetch, 2026-11-09, stopped by Claude Code at step 5,000

```text
Plan
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_resolved_fetch.csv: 629 spans, 599 securities, 619 tickers, 11,968 share samples

Fetching
     500/13,825  ADSK        38s
   1,000/13,825  REGN        67s
   1,500/13,825  LW         110s
   2,000/13,825  ABT        176s
   2,500/13,825  AMD        253s
   3,000/13,825  BK         337s
   3,500/13,825  COF        416s
   4,000/13,825  EQT        491s
   4,500/13,825  GLW        596s
   5,000/13,825  AVB        668s
```

### bars_memory.txt -- what holding the bars cost the fetch, measured on 100 tickers

```text
100 tickers of 619: 229,240 bars held; private 504 -> 595 MB (+90), working set 34 -> 124 MB
per bar 414 bytes; all 619 tickers ~560 MB
```

### fetch_run2.txt -- the restart, --fetch-only, with its memory on every progress line

```text
Plan
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_resolved_fetch.csv: 629 spans, 599 securities, 619 tickers, 11,968 share samples

Fetching
  at the start: this process 36 MB (peak 36, 507 committed); machine 3,345 of 26,420 MB available
     500/13,825  ADSK         2s  this process 39 MB (peak 39, 510 committed); machine 3,300 of 26,420 MB available
   1,000/13,825  REGN         4s  this process 39 MB (peak 39, 510 committed); machine 3,381 of 26,420 MB available
   1,500/13,825  LW           6s  this process 39 MB (peak 39, 510 committed); machine 3,245 of 26,420 MB available
   2,000/13,825  ABT          7s  this process 39 MB (peak 39, 510 committed); machine 3,488 of 26,420 MB available
   2,500/13,825  AMD          8s  this process 39 MB (peak 39, 510 committed); machine 3,474 of 26,420 MB available
   3,000/13,825  BK           8s  this process 39 MB (peak 39, 510 committed); machine 3,455 of 26,420 MB available
   3,500/13,825  COF          8s  this process 39 MB (peak 39, 511 committed); machine 3,433 of 26,420 MB available
   4,000/13,825  EQT          9s  this process 40 MB (peak 40, 511 committed); machine 3,393 of 26,420 MB available
   4,500/13,825  GLW          9s  this process 41 MB (peak 41, 512 committed); machine 3,381 of 26,420 MB available
   5,000/13,825  AVB          9s  this process 41 MB (peak 41, 512 committed); machine 3,424 of 26,420 MB available
   5,500/13,825  LLY         62s  this process 46 MB (peak 46, 513 committed); machine 2,944 of 26,420 MB available
   6,000/13,825  NI         132s  this process 46 MB (peak 46, 513 committed); machine 1,226 of 26,420 MB available
   6,500/13,825  PPG        200s  this process 46 MB (peak 46, 513 committed); machine 905 of 26,420 MB available
   7,000/13,825  SO         270s  this process 47 MB (peak 47, 514 committed); machine 1,615 of 26,420 MB available
   7,500/13,825  WEC        341s  this process 39 MB (peak 47, 515 committed); machine 1,782 of 26,420 MB available
   8,000/13,825  BIIB       407s  this process 39 MB (peak 47, 515 committed); machine 1,674 of 26,420 MB available
   8,500/13,825  GS         474s  this process 40 MB (peak 47, 515 committed); machine 1,053 of 26,420 MB available
   9,000/13,825  NRG        541s  this process 40 MB (peak 47, 516 committed); machine 1,185 of 26,420 MB available
   9,500/13,825  USB        610s  this process 31 MB (peak 47, 516 committed); machine 1,315 of 26,420 MB available
  10,000/13,825  STZ        684s  this process 31 MB (peak 47, 516 committed); machine 1,106 of 26,420 MB available
  10,500/13,825  SMCI       754s  this process 32 MB (peak 47, 516 committed); machine 343 of 26,420 MB available
  11,000/13,825  PODD       821s  this process 32 MB (peak 47, 517 committed); machine 406 of 26,420 MB available
  11,500/13,825  PSTG       897s  this process 29 MB (peak 47, 517 committed); machine 2,323 of 26,420 MB available
  12,000/13,825  SYF        969s  this process 29 MB (peak 47, 517 committed); machine 778 of 26,420 MB available
  12,500/13,825  DXC       1040s  this process 30 MB (peak 47, 517 committed); machine 1,312 of 26,420 MB available
  13,000/13,825  WBD       1112s  this process 31 MB (peak 47, 519 committed); machine 1,049 of 26,420 MB available
  13,500/13,825  WTW       1187s  this process 32 MB (peak 47, 519 committed); machine 736 of 26,420 MB available
  13,825/13,825  SW        1229s  this process 32 MB (peak 47, 519 committed); machine 1,975 of 26,420 MB available
  6,640 requests over the network, in 1229s; 0 failed

Checks: not made (--fetch-only); run with --offline to make them
exit 0
```

### fetch_run3_dividends.txt -- the distributions fetch, and the gate again on the same answers (progress lines but the last two left out; its checks are gate_run.txt's, line for line)

```text
Plan
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_resolved_fetch.csv: 629 spans, 599 securities, 619 tickers, 11,968 share samples

Fetching
  at the start: this process 37 MB (peak 37, 507 committed); machine 2,177 of 26,420 MB available
  14,000/14,448  CB         105s  this process 66 MB (peak 66, 533 committed); machine 1,753 of 26,420 MB available
  14,448/14,448  SW         105s  this process 66 MB (peak 66, 534 committed); machine 1,762 of 26,420 MB available
  622 requests over the network, in 105s; 0 failed

Distributions: the cash paid under each security's tickers inside its spans
  8,234 over 467 of 599 securities, 132 of them paying none; by type {'CD': 8169, 'SC': 65}; by currency {'USD': 8234}
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
  611 of 629 spans pass
  FAILED  COHR 2022-09-08..2026-11-03 (157 member-days): COHR on 2022-09-08 names Coherent Corp. Common Stock (BBG000GS6QB4), not Coherent Corp. (BBG000BLW102)
  FAILED  S600 2023-11-20..2026-11-03 (0 member-days): S600 on 2023-11-20: no reference record
  FAILED  GEV 2024-03-27..2026-11-03 (651 member-days): GEV on 2024-03-27: no reference record
  FAILED  CEGV 2022-01-19..2022-02-01 (0 member-days): CEGV on 2022-01-19: no reference record
  FAILED  SOLV 2024-03-26..2026-11-03 (652 member-days): SOLV on 2024-03-26: no reference record
  FAILED  VLTO 2023-09-27..2026-11-03 (776 member-days): VLTO on 2023-09-27: no reference record
  FAILED  MBC 2022-12-09..2026-11-03 (2 member-days): MBC on 2022-12-09: no reference record
  FAILED  GEHC 2022-12-15..2026-11-03 (962 member-days): GEHC on 2022-12-15: no reference record
  FAILED  AMTM 2024-09-24..2026-11-03 (59 member-days): AMTM on 2024-09-24: no reference record
  FAILED  SNDK 2025-02-13..2026-11-03 (234 member-days): SNDK on 2025-02-13: no reference record
  FAILED  Q 2025-10-27..2026-11-03 (252 member-days): Q on 2025-10-27: no reference record
  FAILED  SOLS 2025-10-20..2026-11-03 (36 member-days): SOLS on 2025-10-20: no reference record
  FAILED  FDXF 2026-05-27..2026-11-03 (109 member-days): FDXF on 2026-05-27: no reference record
  FAILED  HONA 2026-06-15..2026-11-03 (90 member-days): HONA on 2026-06-15: no reference record
  FAILED  VYLR 2026-10-01..2026-11-03 (24 member-days): VYLR on 2026-10-01 names Vylor Inc. (BBG024W5VJC3), not Vylor Inc. (BBG024CQR6X9)
  FAILED  CRHC 2022-01-03..2023-09-22 (0 member-days): CRHC on 2022-01-03 names Cohn Robbins Holdings Corp. (CIK0001818212), not CRH Public Limited Company (CIK0000849395)
  FAILED  EG 2023-07-10..2026-11-03 (835 member-days): EG on 2023-07-10 names Everest Group, Ltd. (NAME:Everest Group, Ltd.), not Everest Group, Ltd. (CIK0001095073)
  FAILED  BG 2023-11-01..2026-11-03 (754 member-days): BG on 2023-11-01 names Bunge Global SA (CIK0001144519), not Bunge Global SA (CIK0001996862)

Splits: each event classified by its ratio -- whole numbers, no term over 100 in lowest terms, or not -- and by the security's share counts either side, within 10%
  55 events inside the securities' spans: 36 splits, applied to the carried share count; 8 price-only, not applied; 11 contradictions, 0 of them exempt (DD); 2 untested, by the ratio alone
  CONTRADICTS Fastenal Co: FAST 1-to-2 on 2025-05-22 (2, whole numbers): 573,450,000 on 2025-04-09, 573,620,638 on 2025-07-11, 1.000 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS O'Reilly Automotive, Inc.: ORLY 1-to-15 on 2025-06-10 (15, whole numbers): 57,240,000 on 2025-04-09, 56,989,455 on 2025-07-11, 0.996 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Interactive Brokers Group, Inc. Class A Common Stock: IBKR 1-to-4 on 2025-06-18 (4, whole numbers): 108,930,000 on 2025-04-09, 108,958,830 on 2025-07-11, 1.000 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS SERVICENOW, INC.: NOW 1-to-5 on 2025-12-18 (5, whole numbers): 208,000,000 on 2025-10-09, 207,678,139 on 2026-01-09, 0.998 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Texas Pacific Land Corporation: TPL 1-to-3 on 2025-12-23 (3, whole numbers): 22,987,326 on 2025-10-09, 22,979,410 on 2026-01-09, 1.000 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Booking Holdings Inc. Common Stock: BKNG 1-to-25 on 2026-04-06 (25, whole numbers): 32,233,815 on 2026-01-09, 31,673,346 on 2026-04-13, 0.983 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Carvana Co.: CVNA 1-to-5 on 2026-05-08 (5, whole numbers): 142,993,769 on 2026-04-13, 143,257,677 on 2026-07-14, 1.002 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS KLA Corporation Common Stock: KLAC 1-to-10 on 2026-06-12 (10, whole numbers): 131,076,611 on 2026-04-13, 130,627,521 on 2026-07-14, 0.997 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS CrowdStrike Holdings, Inc. Class A Common Stock: CRWD 1-to-4 on 2026-07-02 (4, whole numbers): 253,614,090 on 2026-04-13, 254,564,820 on 2026-07-14, 1.004 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Monster Beverage Corporation: MNST 1-to-2 on 2026-08-11 (2, whole numbers): 978,008,110 on 2026-07-14, 979,525,882 on 2026-10-12, 1.002 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Amphenol Corporation: APH 1-to-2 on 2026-09-03 (2, whole numbers): 1,230,234,445 on 2026-07-14, 1,232,983,457 on 2026-10-12, 1.002 times; contradiction, a whole-number ratio and the counts stay unchanged
  price-only  GE Aerospace: GE 1000-to-1281 on 2023-01-04 (1.281, fractional): 1,096,550,000 on 2022-10-04, 1,089,290,000 on 2023-04-05, 0.993 times; price-only, the counts within 10% of unchanged
  price-only  Laboratory Corporation of America Holdings: LH 1000-to-1164 on 2023-07-03 (1.164, fractional): 88,500,000 on 2023-04-05, 88,500,000 on 2023-07-07, 1.000 times; price-only, the counts within 10% of unchanged
  price-only  3M Company: MMM 1000-to-1196 on 2024-04-01 (1.196, fractional): 552,320,000 on 2024-01-05, 553,360,000 on 2024-04-08, 1.002 times; price-only, the counts within 10% of unchanged
  price-only  GE Aerospace: GE 1000-to-1253 on 2024-04-02 (1.253, fractional): 1,088,390,000 on 2024-01-05, 1,093,270,000 on 2024-04-08, 1.004 times; price-only, the counts within 10% of unchanged
  price-only  untested Lennar Corporation Class A: LEN 1000-to-1033 on 2025-01-21 (1.033, fractional): 238,810,000 on 2025-01-07, 232,180,000 on 2025-04-09, 0.972 times; untested, within 10% of both the ratio and no change; price-only by the ratio alone
  price-only  Fortive Corporation: FTV 1000-to-1327 on 2025-06-30 (1.327, fractional): 340,290,000 on 2025-04-09, 339,882,279 on 2025-07-11, 0.999 times; price-only, the counts within 10% of unchanged
  price-only  untested Honeywell International, Inc.: HON 1000-to-1061 on 2025-10-30 (1.061, fractional): 634,896,562 on 2025-10-09, 634,887,208 on 2026-01-09, 1.000 times; untested, within 10% of both the ratio and no change; price-only by the ratio alone
  price-only  DuPont de Nemours, Inc. Common Stock: DD 100-to-239 on 2025-11-03 (2.39, fractional): 418,716,685 on 2025-10-09, 418,975,324 on 2026-01-09, 1.001 times; price-only, the counts within 10% of unchanged
  split       W.R. Berkley Corporation: WRB 2-to-3 on 2022-03-24 (1.5, whole numbers): 176,640,000 on 2022-01-03, 265,170,000 on 2022-04-04, 1.501 times; a split, the counts follow it
  split       Amazon.Com Inc: AMZN 1-to-20 on 2022-06-06 (20, whole numbers): 508,840,000 on 2022-04-04, 10,174,410,000 on 2022-07-06, 19.995 times; a split, the counts follow it
  split       DexCom, Inc.: DXCM 1-to-4 on 2022-06-13 (4, whole numbers): 97,390,000 on 2022-04-04, 392,500,000 on 2022-07-06, 4.030 times; a split, the counts follow it
  split       Fortinet, Inc.: FTNT 1-to-5 on 2022-06-23 (5, whole numbers): 160,820,000 on 2022-04-04, 802,640,000 on 2022-07-06, 4.991 times; a split, the counts follow it
  split       Alphabet Inc. Class C Capital Stock: GOOG 1-to-20 on 2022-07-18 (20, whole numbers): 313,380,000 on 2022-07-06, 6,163,000,000 on 2022-10-04, 19.666 times; a split, the counts follow it
  split       Alphabet Inc. Class A Common Stock: GOOGL 1-to-20 on 2022-07-18 (20, whole numbers): 300,760,000 on 2022-07-06, 5,996,000,000 on 2022-10-04, 19.936 times; a split, the counts follow it
  split       Tesla, Inc. Common Stock: TSLA 1-to-3 on 2022-08-25 (3, whole numbers): 1,036,010,000 on 2022-07-06, 3,133,470,000 on 2022-10-04, 3.025 times; a split, the counts follow it
  split       Nasdaq, Inc. Common Stock: NDAQ 1-to-3 on 2022-08-29 (3, whole numbers): 164,680,000 on 2022-07-06, 491,230,000 on 2022-10-04, 2.983 times; a split, the counts follow it
  split       Palo Alto Networks, Inc. Common Stock: PANW 1-to-3 on 2022-09-14 (3, whole numbers): 99,640,000 on 2022-07-06, 299,210,000 on 2022-10-04, 3.003 times; a split, the counts follow it
  split       Copart Inc: CPRT 1-to-2 on 2022-11-04 (2, whole numbers): 238,060,000 on 2022-10-04, 476,300,000 on 2023-01-04, 2.001 times; a split, the counts follow it
  split       Bio-Techne Corp.: TECH 1-to-4 on 2022-11-30 (4, whole numbers): 39,210,000 on 2022-10-04, 156,970,000 on 2023-01-04, 4.003 times; a split, the counts follow it
  split       Paccar Inc: PCAR 2-to-3 on 2023-02-08 (1.5, whole numbers): 347,770,000 on 2023-01-04, 522,510,000 on 2023-04-05, 1.502 times; a split, the counts follow it
  split       Monster Beverage Corporation: MNST 1-to-2 on 2023-03-28 (2, whole numbers): 521,740,000 on 2023-01-04, 1,044,819,999 on 2023-04-05, 2.003 times; a split, the counts follow it
  split       Copart Inc: CPRT 1-to-2 on 2023-08-22 (2, whole numbers): 477,440,000 on 2023-07-07, 957,360,000 on 2023-10-05, 2.005 times; a split, the counts follow it
  split       Sempra: SRE 1-to-2 on 2023-08-22 (2, whole numbers): 314,650,000 on 2023-07-07, 629,310,000 on 2023-10-05, 2.000 times; a split, the counts follow it
  split       The Cooper Companies, Inc. Common Stock: COO 1-to-4 on 2024-02-20 (4, whole numbers): 49,530,000 on 2024-01-05, 198,760,000 on 2024-04-08, 4.013 times; a split, the counts follow it
  split       Walmart Inc. Common Stock: WMT 1-to-3 on 2024-02-26 (3, whole numbers): 2,692,230,000 on 2024-01-05, 8,052,790,000 on 2024-04-08, 2.991 times; a split, the counts follow it
  split       Texas Pacific Land Corporation: TPL 1-to-3 on 2024-03-27 (3, whole numbers): 7,680,000 on 2024-01-05, 23,010,000 on 2024-04-08, 2.996 times; a split, the counts follow it
  split       Old Dominion Freight Line: ODFL 1-to-2 on 2024-03-28 (2, whole numbers): 109,110,000 on 2024-01-05, 217,670,000 on 2024-04-08, 1.995 times; a split, the counts follow it
  split       Nvidia Corp: NVDA 1-to-10 on 2024-06-10 (10, whole numbers): 2,500,000,000 on 2024-04-08, 24,600,000,000 on 2024-07-09, 9.840 times; a split, the counts follow it
  split       Amphenol Corporation: APH 1-to-2 on 2024-06-12 (2, whole numbers): 599,860,000 on 2024-04-08, 1,201,210,000 on 2024-07-09, 2.002 times; a split, the counts follow it
  split       Chipotle Mexican Grill, Inc.: CMG 1-to-50 on 2024-06-26 (50, whole numbers): 27,420,000 on 2024-04-08, 1,373,350,000 on 2024-07-09, 50.086 times; a split, the counts follow it
  split       Williams-Sonoma, Inc.: WSM 1-to-2 on 2024-07-09 (2, whole numbers): 64,110,000 on 2024-04-08, 126,340,000 on 2024-10-07, 1.971 times; a split, the counts follow it
  split       W.R. Berkley Corporation: WRB 2-to-3 on 2024-07-11 (1.5, whole numbers): 255,660,000 on 2024-07-09, 380,550,000 on 2024-10-07, 1.489 times; a split, the counts follow it
  split       Broadcom Inc. Common Stock: AVGO 1-to-10 on 2024-07-15 (10, whole numbers): 465,490,000 on 2024-07-09, 4,670,580,000 on 2024-10-07, 10.034 times; a split, the counts follow it
  split       Cintas Corp: CTAS 1-to-4 on 2024-09-12 (4, whole numbers): 101,460,000 on 2024-07-09, 403,260,000 on 2024-10-07, 3.975 times; a split, the counts follow it
  split       Deckers Outdoor Corp: DECK 1-to-6 on 2024-09-17 (6, whole numbers): 25,440,000 on 2024-07-09, 152,460,000 on 2024-10-07, 5.993 times; a split, the counts follow it
  split       Super Micro Computer, Inc. Common Stock: SMCI 1-to-10 on 2024-10-01 (10, whole numbers): 58,560,000 on 2024-09-30, 585,570,000 on 2024-10-07, 9.999 times; a split, the counts follow it
  split       Lam Research Corp: LRCX 1-to-10 on 2024-10-03 (10, whole numbers): 130,740,000 on 2024-07-09, 1,296,120,000 on 2024-10-07, 9.914 times; a split, the counts follow it
  split       Arista Networks: ANET 1-to-4 on 2024-12-04 (4, whole numbers): 314,150,000 on 2024-10-07, 1,259,760,000 on 2025-01-07, 4.010 times; a split, the counts follow it
  split       Entergy Corporation: ETR 1-to-2 on 2024-12-13 (2, whole numbers): 213,830,000 on 2024-10-07, 428,820,000 on 2025-01-07, 2.005 times; a split, the counts follow it
  split       Palo Alto Networks, Inc. Common Stock: PANW 1-to-2 on 2024-12-16 (2, whole numbers): 325,600,000 on 2024-10-07, 656,200,000 on 2025-01-07, 2.015 times; a split, the counts follow it
  split       Tractor Supply Co: TSCO 1-to-5 on 2024-12-20 (5, whole numbers): 107,870,000 on 2024-10-07, 534,200,000 on 2025-01-07, 4.952 times; a split, the counts follow it
  split       NetFlix Inc: NFLX 1-to-10 on 2025-11-17 (10, whole numbers): 424,926,346 on 2025-10-09, 4,237,323,340 on 2026-01-09, 9.972 times; a split, the counts follow it
  split       Amcor plc Ordinary Shares: AMCR 5-to-1 on 2026-01-15 (0.2, whole numbers): 2,308,359,941 on 2026-01-09, 462,045,690 on 2026-04-13, 0.200 times; a split, the counts follow it
  split       DuPont de Nemours, Inc. Common Stock: DD 3-to-1 on 2026-06-24 (0.3333, whole numbers): 408,923,857 on 2026-04-13, 135,019,401 on 2026-07-14, 0.330 times; a split, the counts follow it

Gate: STOPS
  wrote data\external\polygon\sp500_rebuild_gate_2022-01-03_2026-11-03.json
exit 1
```

### gate_run.txt -- the gate, offline, on the new split rule

```text
Plan
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_resolved_fetch.csv: 629 spans, 599 securities, 619 tickers, 11,968 share samples

Fetching
  at the start: this process 36 MB (peak 36, 507 committed); machine 1,731 of 26,420 MB available
  13,500/13,825  WTW         14s  this process 47 MB (peak 47, 518 committed); machine 1,300 of 26,420 MB available
  13,825/13,825  SW          14s  this process 47 MB (peak 47, 518 committed); machine 1,301 of 26,420 MB available
  0 requests over the network, in 14s; 0 failed

Identity: each span's first share sample against the security the table expects
  611 of 629 spans pass
  FAILED  COHR 2022-09-08..2026-11-03 (157 member-days): COHR on 2022-09-08 names Coherent Corp. Common Stock (BBG000GS6QB4), not Coherent Corp. (BBG000BLW102)
  FAILED  S600 2023-11-20..2026-11-03 (0 member-days): S600 on 2023-11-20: no reference record
  FAILED  GEV 2024-03-27..2026-11-03 (651 member-days): GEV on 2024-03-27: no reference record
  FAILED  CEGV 2022-01-19..2022-02-01 (0 member-days): CEGV on 2022-01-19: no reference record
  FAILED  SOLV 2024-03-26..2026-11-03 (652 member-days): SOLV on 2024-03-26: no reference record
  FAILED  VLTO 2023-09-27..2026-11-03 (776 member-days): VLTO on 2023-09-27: no reference record
  FAILED  MBC 2022-12-09..2026-11-03 (2 member-days): MBC on 2022-12-09: no reference record
  FAILED  GEHC 2022-12-15..2026-11-03 (962 member-days): GEHC on 2022-12-15: no reference record
  FAILED  AMTM 2024-09-24..2026-11-03 (59 member-days): AMTM on 2024-09-24: no reference record
  FAILED  SNDK 2025-02-13..2026-11-03 (234 member-days): SNDK on 2025-02-13: no reference record
  FAILED  Q 2025-10-27..2026-11-03 (252 member-days): Q on 2025-10-27: no reference record
  FAILED  SOLS 2025-10-20..2026-11-03 (36 member-days): SOLS on 2025-10-20: no reference record
  FAILED  FDXF 2026-05-27..2026-11-03 (109 member-days): FDXF on 2026-05-27: no reference record
  FAILED  HONA 2026-06-15..2026-11-03 (90 member-days): HONA on 2026-06-15: no reference record
  FAILED  VYLR 2026-10-01..2026-11-03 (24 member-days): VYLR on 2026-10-01 names Vylor Inc. (BBG024W5VJC3), not Vylor Inc. (BBG024CQR6X9)
  FAILED  CRHC 2022-01-03..2023-09-22 (0 member-days): CRHC on 2022-01-03 names Cohn Robbins Holdings Corp. (CIK0001818212), not CRH Public Limited Company (CIK0000849395)
  FAILED  EG 2023-07-10..2026-11-03 (835 member-days): EG on 2023-07-10 names Everest Group, Ltd. (NAME:Everest Group, Ltd.), not Everest Group, Ltd. (CIK0001095073)
  FAILED  BG 2023-11-01..2026-11-03 (754 member-days): BG on 2023-11-01 names Bunge Global SA (CIK0001144519), not Bunge Global SA (CIK0001996862)

Splits: each event classified by its ratio -- whole numbers, no term over 100 in lowest terms, or not -- and by the security's share counts either side, within 10%
  55 events inside the securities' spans: 36 splits, applied to the carried share count; 8 price-only, not applied; 11 contradictions, 0 of them exempt (DD); 2 untested, by the ratio alone
  CONTRADICTS Fastenal Co: FAST 1-to-2 on 2025-05-22 (2, whole numbers): 573,450,000 on 2025-04-09, 573,620,638 on 2025-07-11, 1.000 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS O'Reilly Automotive, Inc.: ORLY 1-to-15 on 2025-06-10 (15, whole numbers): 57,240,000 on 2025-04-09, 56,989,455 on 2025-07-11, 0.996 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Interactive Brokers Group, Inc. Class A Common Stock: IBKR 1-to-4 on 2025-06-18 (4, whole numbers): 108,930,000 on 2025-04-09, 108,958,830 on 2025-07-11, 1.000 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS SERVICENOW, INC.: NOW 1-to-5 on 2025-12-18 (5, whole numbers): 208,000,000 on 2025-10-09, 207,678,139 on 2026-01-09, 0.998 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Texas Pacific Land Corporation: TPL 1-to-3 on 2025-12-23 (3, whole numbers): 22,987,326 on 2025-10-09, 22,979,410 on 2026-01-09, 1.000 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Booking Holdings Inc. Common Stock: BKNG 1-to-25 on 2026-04-06 (25, whole numbers): 32,233,815 on 2026-01-09, 31,673,346 on 2026-04-13, 0.983 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Carvana Co.: CVNA 1-to-5 on 2026-05-08 (5, whole numbers): 142,993,769 on 2026-04-13, 143,257,677 on 2026-07-14, 1.002 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS KLA Corporation Common Stock: KLAC 1-to-10 on 2026-06-12 (10, whole numbers): 131,076,611 on 2026-04-13, 130,627,521 on 2026-07-14, 0.997 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS CrowdStrike Holdings, Inc. Class A Common Stock: CRWD 1-to-4 on 2026-07-02 (4, whole numbers): 253,614,090 on 2026-04-13, 254,564,820 on 2026-07-14, 1.004 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Monster Beverage Corporation: MNST 1-to-2 on 2026-08-11 (2, whole numbers): 978,008,110 on 2026-07-14, 979,525,882 on 2026-10-12, 1.002 times; contradiction, a whole-number ratio and the counts stay unchanged
  CONTRADICTS Amphenol Corporation: APH 1-to-2 on 2026-09-03 (2, whole numbers): 1,230,234,445 on 2026-07-14, 1,232,983,457 on 2026-10-12, 1.002 times; contradiction, a whole-number ratio and the counts stay unchanged
  price-only  GE Aerospace: GE 1000-to-1281 on 2023-01-04 (1.281, fractional): 1,096,550,000 on 2022-10-04, 1,089,290,000 on 2023-04-05, 0.993 times; price-only, the counts within 10% of unchanged
  price-only  Laboratory Corporation of America Holdings: LH 1000-to-1164 on 2023-07-03 (1.164, fractional): 88,500,000 on 2023-04-05, 88,500,000 on 2023-07-07, 1.000 times; price-only, the counts within 10% of unchanged
  price-only  3M Company: MMM 1000-to-1196 on 2024-04-01 (1.196, fractional): 552,320,000 on 2024-01-05, 553,360,000 on 2024-04-08, 1.002 times; price-only, the counts within 10% of unchanged
  price-only  GE Aerospace: GE 1000-to-1253 on 2024-04-02 (1.253, fractional): 1,088,390,000 on 2024-01-05, 1,093,270,000 on 2024-04-08, 1.004 times; price-only, the counts within 10% of unchanged
  price-only  untested Lennar Corporation Class A: LEN 1000-to-1033 on 2025-01-21 (1.033, fractional): 238,810,000 on 2025-01-07, 232,180,000 on 2025-04-09, 0.972 times; untested, within 10% of both the ratio and no change; price-only by the ratio alone
  price-only  Fortive Corporation: FTV 1000-to-1327 on 2025-06-30 (1.327, fractional): 340,290,000 on 2025-04-09, 339,882,279 on 2025-07-11, 0.999 times; price-only, the counts within 10% of unchanged
  price-only  untested Honeywell International, Inc.: HON 1000-to-1061 on 2025-10-30 (1.061, fractional): 634,896,562 on 2025-10-09, 634,887,208 on 2026-01-09, 1.000 times; untested, within 10% of both the ratio and no change; price-only by the ratio alone
  price-only  DuPont de Nemours, Inc. Common Stock: DD 100-to-239 on 2025-11-03 (2.39, fractional): 418,716,685 on 2025-10-09, 418,975,324 on 2026-01-09, 1.001 times; price-only, the counts within 10% of unchanged
  split       W.R. Berkley Corporation: WRB 2-to-3 on 2022-03-24 (1.5, whole numbers): 176,640,000 on 2022-01-03, 265,170,000 on 2022-04-04, 1.501 times; a split, the counts follow it
  split       Amazon.Com Inc: AMZN 1-to-20 on 2022-06-06 (20, whole numbers): 508,840,000 on 2022-04-04, 10,174,410,000 on 2022-07-06, 19.995 times; a split, the counts follow it
  split       DexCom, Inc.: DXCM 1-to-4 on 2022-06-13 (4, whole numbers): 97,390,000 on 2022-04-04, 392,500,000 on 2022-07-06, 4.030 times; a split, the counts follow it
  split       Fortinet, Inc.: FTNT 1-to-5 on 2022-06-23 (5, whole numbers): 160,820,000 on 2022-04-04, 802,640,000 on 2022-07-06, 4.991 times; a split, the counts follow it
  split       Alphabet Inc. Class C Capital Stock: GOOG 1-to-20 on 2022-07-18 (20, whole numbers): 313,380,000 on 2022-07-06, 6,163,000,000 on 2022-10-04, 19.666 times; a split, the counts follow it
  split       Alphabet Inc. Class A Common Stock: GOOGL 1-to-20 on 2022-07-18 (20, whole numbers): 300,760,000 on 2022-07-06, 5,996,000,000 on 2022-10-04, 19.936 times; a split, the counts follow it
  split       Tesla, Inc. Common Stock: TSLA 1-to-3 on 2022-08-25 (3, whole numbers): 1,036,010,000 on 2022-07-06, 3,133,470,000 on 2022-10-04, 3.025 times; a split, the counts follow it
  split       Nasdaq, Inc. Common Stock: NDAQ 1-to-3 on 2022-08-29 (3, whole numbers): 164,680,000 on 2022-07-06, 491,230,000 on 2022-10-04, 2.983 times; a split, the counts follow it
  split       Palo Alto Networks, Inc. Common Stock: PANW 1-to-3 on 2022-09-14 (3, whole numbers): 99,640,000 on 2022-07-06, 299,210,000 on 2022-10-04, 3.003 times; a split, the counts follow it
  split       Copart Inc: CPRT 1-to-2 on 2022-11-04 (2, whole numbers): 238,060,000 on 2022-10-04, 476,300,000 on 2023-01-04, 2.001 times; a split, the counts follow it
  split       Bio-Techne Corp.: TECH 1-to-4 on 2022-11-30 (4, whole numbers): 39,210,000 on 2022-10-04, 156,970,000 on 2023-01-04, 4.003 times; a split, the counts follow it
  split       Paccar Inc: PCAR 2-to-3 on 2023-02-08 (1.5, whole numbers): 347,770,000 on 2023-01-04, 522,510,000 on 2023-04-05, 1.502 times; a split, the counts follow it
  split       Monster Beverage Corporation: MNST 1-to-2 on 2023-03-28 (2, whole numbers): 521,740,000 on 2023-01-04, 1,044,819,999 on 2023-04-05, 2.003 times; a split, the counts follow it
  split       Copart Inc: CPRT 1-to-2 on 2023-08-22 (2, whole numbers): 477,440,000 on 2023-07-07, 957,360,000 on 2023-10-05, 2.005 times; a split, the counts follow it
  split       Sempra: SRE 1-to-2 on 2023-08-22 (2, whole numbers): 314,650,000 on 2023-07-07, 629,310,000 on 2023-10-05, 2.000 times; a split, the counts follow it
  split       The Cooper Companies, Inc. Common Stock: COO 1-to-4 on 2024-02-20 (4, whole numbers): 49,530,000 on 2024-01-05, 198,760,000 on 2024-04-08, 4.013 times; a split, the counts follow it
  split       Walmart Inc. Common Stock: WMT 1-to-3 on 2024-02-26 (3, whole numbers): 2,692,230,000 on 2024-01-05, 8,052,790,000 on 2024-04-08, 2.991 times; a split, the counts follow it
  split       Texas Pacific Land Corporation: TPL 1-to-3 on 2024-03-27 (3, whole numbers): 7,680,000 on 2024-01-05, 23,010,000 on 2024-04-08, 2.996 times; a split, the counts follow it
  split       Old Dominion Freight Line: ODFL 1-to-2 on 2024-03-28 (2, whole numbers): 109,110,000 on 2024-01-05, 217,670,000 on 2024-04-08, 1.995 times; a split, the counts follow it
  split       Nvidia Corp: NVDA 1-to-10 on 2024-06-10 (10, whole numbers): 2,500,000,000 on 2024-04-08, 24,600,000,000 on 2024-07-09, 9.840 times; a split, the counts follow it
  split       Amphenol Corporation: APH 1-to-2 on 2024-06-12 (2, whole numbers): 599,860,000 on 2024-04-08, 1,201,210,000 on 2024-07-09, 2.002 times; a split, the counts follow it
  split       Chipotle Mexican Grill, Inc.: CMG 1-to-50 on 2024-06-26 (50, whole numbers): 27,420,000 on 2024-04-08, 1,373,350,000 on 2024-07-09, 50.086 times; a split, the counts follow it
  split       Williams-Sonoma, Inc.: WSM 1-to-2 on 2024-07-09 (2, whole numbers): 64,110,000 on 2024-04-08, 126,340,000 on 2024-10-07, 1.971 times; a split, the counts follow it
  split       W.R. Berkley Corporation: WRB 2-to-3 on 2024-07-11 (1.5, whole numbers): 255,660,000 on 2024-07-09, 380,550,000 on 2024-10-07, 1.489 times; a split, the counts follow it
  split       Broadcom Inc. Common Stock: AVGO 1-to-10 on 2024-07-15 (10, whole numbers): 465,490,000 on 2024-07-09, 4,670,580,000 on 2024-10-07, 10.034 times; a split, the counts follow it
  split       Cintas Corp: CTAS 1-to-4 on 2024-09-12 (4, whole numbers): 101,460,000 on 2024-07-09, 403,260,000 on 2024-10-07, 3.975 times; a split, the counts follow it
  split       Deckers Outdoor Corp: DECK 1-to-6 on 2024-09-17 (6, whole numbers): 25,440,000 on 2024-07-09, 152,460,000 on 2024-10-07, 5.993 times; a split, the counts follow it
  split       Super Micro Computer, Inc. Common Stock: SMCI 1-to-10 on 2024-10-01 (10, whole numbers): 58,560,000 on 2024-09-30, 585,570,000 on 2024-10-07, 9.999 times; a split, the counts follow it
  split       Lam Research Corp: LRCX 1-to-10 on 2024-10-03 (10, whole numbers): 130,740,000 on 2024-07-09, 1,296,120,000 on 2024-10-07, 9.914 times; a split, the counts follow it
  split       Arista Networks: ANET 1-to-4 on 2024-12-04 (4, whole numbers): 314,150,000 on 2024-10-07, 1,259,760,000 on 2025-01-07, 4.010 times; a split, the counts follow it
  split       Entergy Corporation: ETR 1-to-2 on 2024-12-13 (2, whole numbers): 213,830,000 on 2024-10-07, 428,820,000 on 2025-01-07, 2.005 times; a split, the counts follow it
  split       Palo Alto Networks, Inc. Common Stock: PANW 1-to-2 on 2024-12-16 (2, whole numbers): 325,600,000 on 2024-10-07, 656,200,000 on 2025-01-07, 2.015 times; a split, the counts follow it
  split       Tractor Supply Co: TSCO 1-to-5 on 2024-12-20 (5, whole numbers): 107,870,000 on 2024-10-07, 534,200,000 on 2025-01-07, 4.952 times; a split, the counts follow it
  split       NetFlix Inc: NFLX 1-to-10 on 2025-11-17 (10, whole numbers): 424,926,346 on 2025-10-09, 4,237,323,340 on 2026-01-09, 9.972 times; a split, the counts follow it
  split       Amcor plc Ordinary Shares: AMCR 5-to-1 on 2026-01-15 (0.2, whole numbers): 2,308,359,941 on 2026-01-09, 462,045,690 on 2026-04-13, 0.200 times; a split, the counts follow it
  split       DuPont de Nemours, Inc. Common Stock: DD 3-to-1 on 2026-06-24 (0.3333, whole numbers): 408,923,857 on 2026-04-13, 135,019,401 on 2026-07-14, 0.330 times; a split, the counts follow it

Gate: STOPS
  wrote data\external\polygon\sp500_rebuild_gate_2022-01-03_2026-11-03.json
exit 1
```

### split_survey.txt -- every split event inside a member security's span, as Polygon writes it

```text
fields on the events: ['execution_date', 'id', 'split_from', 'split_to', 'ticker']
55 events inside spans; 5 in the window but outside the ticker's span for its security
  2022-03-24  WRB    W.R. Berkley Corporation               2:3      lowest 3:2     =   1.5000  id E
  2022-06-06  AMZN   Amazon.Com Inc                         1:20     lowest 20:1     =  20.0000  id E
  2022-06-13  DXCM   DexCom, Inc.                           1:4      lowest 4:1     =   4.0000  id E
  2022-06-23  FTNT   Fortinet, Inc.                         1:5      lowest 5:1     =   5.0000  id E
  2022-07-18  GOOG   Alphabet Inc. Class C Capital Stoc     1:20     lowest 20:1     =  20.0000  id E
  2022-07-18  GOOGL  Alphabet Inc. Class A Common Stock     1:20     lowest 20:1     =  20.0000  id E
  2022-08-25  TSLA   Tesla, Inc. Common Stock               1:3      lowest 3:1     =   3.0000  id E
  2022-08-29  NDAQ   Nasdaq, Inc. Common Stock              1:3      lowest 3:1     =   3.0000  id E
  2022-09-14  PANW   Palo Alto Networks, Inc. Common St     1:3      lowest 3:1     =   3.0000  id E
  2022-11-04  CPRT   Copart Inc                             1:2      lowest 2:1     =   2.0000  id E
  2022-11-30  TECH   Bio-Techne Corp.                       1:4      lowest 4:1     =   4.0000  id E
  2023-01-04  GE     GE Aerospace                        1000:1281   lowest 1281:1000  =   1.2810  id P
  2023-02-08  PCAR   Paccar Inc                             2:3      lowest 3:2     =   1.5000  id E
  2023-03-28  MNST   Monster Beverage Corporation           1:2      lowest 2:1     =   2.0000  id E
  2023-07-03  LH     Laboratory Corporation of America   1000:1164   lowest 291:250   =   1.1640  id P
  2023-08-22  CPRT   Copart Inc                             1:2      lowest 2:1     =   2.0000  id E
  2023-08-22  SRE    Sempra                                 1:2      lowest 2:1     =   2.0000  id E
  2024-02-20  COO    The Cooper Companies, Inc. Common      1:4      lowest 4:1     =   4.0000  id E
  2024-02-26  WMT    Walmart Inc. Common Stock              1:3      lowest 3:1     =   3.0000  id E
  2024-03-27  TPL    Texas Pacific Land Corporation         1:3      lowest 3:1     =   3.0000  id E
  2024-03-28  ODFL   Old Dominion Freight Line              1:2      lowest 2:1     =   2.0000  id E
  2024-04-01  MMM    3M Company                          1000:1196   lowest 299:250   =   1.1960  id P
  2024-04-02  GE     GE Aerospace                        1000:1253   lowest 1253:1000  =   1.2530  id P
  2024-06-10  NVDA   Nvidia Corp                            1:10     lowest 10:1     =  10.0000  id E
  2024-06-12  APH    Amphenol Corporation                   1:2      lowest 2:1     =   2.0000  id E
  2024-06-26  CMG    Chipotle Mexican Grill, Inc.           1:50     lowest 50:1     =  50.0000  id E
  2024-07-09  WSM    Williams-Sonoma, Inc.                  1:2      lowest 2:1     =   2.0000  id E
  2024-07-11  WRB    W.R. Berkley Corporation               2:3      lowest 3:2     =   1.5000  id P
  2024-07-15  AVGO   Broadcom Inc. Common Stock             1:10     lowest 10:1     =  10.0000  id E
  2024-09-12  CTAS   Cintas Corp                            1:4      lowest 4:1     =   4.0000  id E
  2024-09-17  DECK   Deckers Outdoor Corp                   1:6      lowest 6:1     =   6.0000  id E
  2024-10-01  SMCI   Super Micro Computer, Inc. Common      1:10     lowest 10:1     =  10.0000  id E
  2024-10-03  LRCX   Lam Research Corp                      1:10     lowest 10:1     =  10.0000  id E
  2024-12-04  ANET   Arista Networks                        1:4      lowest 4:1     =   4.0000  id E
  2024-12-13  ETR    Entergy Corporation                    1:2      lowest 2:1     =   2.0000  id E
  2024-12-16  PANW   Palo Alto Networks, Inc. Common St     1:2      lowest 2:1     =   2.0000  id E
  2024-12-20  TSCO   Tractor Supply Co                      1:5      lowest 5:1     =   5.0000  id E
  2025-01-21  LEN    Lennar Corporation Class A          1000:1033   lowest 1033:1000  =   1.0330  id P
  2025-05-22  FAST   Fastenal Co                            1:2      lowest 2:1     =   2.0000  id E
  2025-06-10  ORLY   O'Reilly Automotive, Inc.              1:15     lowest 15:1     =  15.0000  id E
  2025-06-18  IBKR   Interactive Brokers Group, Inc. Cl     1:4      lowest 4:1     =   4.0000  id E
  2025-06-30  FTV    Fortive Corporation                 1000:1327   lowest 1327:1000  =   1.3270  id P
  2025-10-30  HON    Honeywell International, Inc.       1000:1061   lowest 1061:1000  =   1.0610  id P
  2025-11-03  DD     DuPont de Nemours, Inc. Common Sto   100:239    lowest 239:100   =   2.3900  id P
  2025-11-17  NFLX   NetFlix Inc                            1:10     lowest 10:1     =  10.0000  id E
  2025-12-18  NOW    SERVICENOW, INC.                       1:5      lowest 5:1     =   5.0000  id E
  2025-12-23  TPL    Texas Pacific Land Corporation         1:3      lowest 3:1     =   3.0000  id E
  2026-01-15  AMCR   Amcor plc Ordinary Shares              5:1      lowest 1:5     =   0.2000  id E
  2026-04-06  BKNG   Booking Holdings Inc. Common Stock     1:25     lowest 25:1     =  25.0000  id E
  2026-05-08  CVNA   Carvana Co.                            1:5      lowest 5:1     =   5.0000  id E
  2026-06-12  KLAC   KLA Corporation Common Stock           1:10     lowest 10:1     =  10.0000  id E
  2026-06-24  DD     DuPont de Nemours, Inc. Common Sto     3:1      lowest 1:3     =   0.3333  id E
  2026-07-02  CRWD   CrowdStrike Holdings, Inc. Class A     1:4      lowest 4:1     =   4.0000  id E
  2026-08-11  MNST   Monster Beverage Corporation           1:2      lowest 2:1     =   2.0000  id E
  2026-09-03  APH    Amphenol Corporation                   1:2      lowest 2:1     =   2.0000  id E

largest term over 100: 8; at most 100: 47
largest terms seen: [2, 3, 4, 5, 6, 10, 15, 20, 25, 50, 239, 291, 299, 1033, 1061, 1253, 1281, 1327]
```

### price_only_check.txt -- each fractional event's raw and adjusted prices on its date

```text
  GE 2023-01-04 ratio 1.281: raw 2023-01-03 close 84.98 -> 2023-01-04 open 68.41 close 70.20 (raw -17.39%, open gap -19.50%); adjusted +5.82%; raw/adjusted 1.6051 -> 1.2530 (step 1.2810); 1 - 1/ratio = 21.94%
  LH 2023-07-03 ratio 1.164: raw 2023-06-30 close 241.33 -> 2023-07-03 open 206.15 close 208.92 (raw -13.43%, open gap -14.58%); adjusted +0.77%; raw/adjusted 1.1640 -> 1.0000 (step 1.1640); 1 - 1/ratio = 14.09%
  MMM 2024-04-01 ratio 1.196: raw 2024-03-28 close 106.07 -> 2024-04-01 open 91.05 close 94.02 (raw -11.36%, open gap -14.16%); adjusted +6.01%; raw/adjusted 1.1960 -> 1.0000 (step 1.1960); 1 - 1/ratio = 16.39%
  GE 2024-04-02 ratio 1.253: raw 2024-04-01 close 175.36 -> 2024-04-02 open 140.53 close 136.47 (raw -22.18%, open gap -19.86%); adjusted -2.49%; raw/adjusted 1.2530 -> 1.0000 (step 1.2530); 1 - 1/ratio = 20.19%
  LEN 2025-01-21 ratio 1.033: raw 2025-01-17 close 141.54 -> 2025-01-21 open 136.01 close 133.14 (raw -5.93%, open gap -3.91%); adjusted -2.83%; raw/adjusted 1.0330 -> 1.0000 (step 1.0330); 1 - 1/ratio = 3.19%
  LEN 2025-02-07 ratio 1.033: raw 2025-02-06 close 127.25 -> 2025-02-07 open 126.52 close 121.94 (raw -4.17%, open gap -0.57%); adjusted -4.17%; raw/adjusted 1.0000 -> 1.0000 (step 1.0000); 1 - 1/ratio = 3.19%
  LEN 2025-02-10 ratio 1.033: raw 2025-02-07 close 121.94 -> 2025-02-10 open 122.96 close 122.69 (raw +0.62%, open gap +0.84%); adjusted +0.62%; raw/adjusted 1.0000 -> 1.0000 (step 1.0000); 1 - 1/ratio = 3.19%
  LEN 2025-02-11 ratio 1.033: raw 2025-02-10 close 122.69 -> 2025-02-11 open 122.21 close 124.92 (raw +1.82%, open gap -0.39%); adjusted +1.82%; raw/adjusted 1.0000 -> 1.0000 (step 1.0000); 1 - 1/ratio = 3.19%
  FTV 2025-06-30 ratio 1.327: raw 2025-06-27 close 71.60 -> 2025-06-30 open 52.05 close 52.13 (raw -27.19%, open gap -27.30%); adjusted -3.38%; raw/adjusted 1.3270 -> 1.0000 (step 1.3270); 1 - 1/ratio = 24.64%
  HON 2025-10-30 ratio 1.061: raw 2025-10-29 close 212.89 -> 2025-10-30 open 201.80 close 200.11 (raw -6.00%, open gap -5.21%); adjusted -0.27%; raw/adjusted 1.0610 -> 1.0000 (step 1.0610); 1 - 1/ratio = 5.75%
  DD 2025-11-03 ratio 2.39: raw 2025-10-31 close 81.65 -> 2025-11-03 open 33.69 close 34.69 (raw -57.51%, open gap -58.74%); adjusted +1.54%; raw/adjusted 0.7967 -> 0.3333 (step 2.3900); 1 - 1/ratio = 58.16%
```

### name_spinoffs.txt -- each spin-off's child at the reference endpoint

```text
GE 2023-01-04 1.281: hypothesis GEHC, GE HealthCare Technologies
    GEHC on 2023-01-04: GE HealthCare Technologies Inc. Common Stock, listed None, CIK None, FIGI BBG01BFR8YV1, type CS, shares None
    GEHC on 2023-01-11: GE HealthCare Technologies Inc. Common Stock, listed 2022-12-15, CIK 0001932393, FIGI BBG01BFR8YV1, type CS, shares 454710000
LH 2023-07-03 1.164: hypothesis FTRE, Fortrea Holdings
    FTRE on 2023-07-03: Fortrea Holdings Inc. Common Stock, listed None, CIK 0001965040, FIGI BBG01GZF9VY6, type CS, shares 88600000
    FTRE on 2023-07-10: Fortrea Holdings Inc. Common Stock, listed 2023-06-16, CIK 0001965040, FIGI BBG01GZF9VY6, type CS, shares 88600000
MMM 2024-04-01 1.196: hypothesis SOLV, Solventum
    SOLV on 2024-04-01: Solventum Corporation, listed None, CIK 0001964738, FIGI BBG018YZH6T3, type CS, shares None
    SOLV on 2024-04-08: Solventum Corporation, listed 2024-03-26, CIK 0001964738, FIGI BBG018YZH6T3, type CS, shares 172710000
GE 2024-04-02 1.253: hypothesis GEV, GE Vernova
    GEV on 2024-04-02: GE Vernova Inc., listed None, CIK 0001996810, FIGI BBG013G17W31, type CS, shares None
    GEV on 2024-04-09: GE Vernova Inc., listed 2024-03-27, CIK 0001996810, FIGI BBG013G17W31, type CS, shares 272080000
LEN 2025-01-21 1.033: hypothesis MRP, Millrose Properties
    MRP on 2025-01-21: no record
    MRP on 2025-01-28: no record
FTV 2025-06-30 1.327: hypothesis RAL, Ralliant
    RAL on 2025-06-30: Ralliant Corporation, listed None, CIK 0002041385, FIGI BBG01V4YRP24, type CS, shares None
    RAL on 2025-07-07: Ralliant Corporation, listed 2025-06-25, CIK 0002041385, FIGI BBG01V4YRP24, type CS, shares None
HON 2025-10-30 1.061: hypothesis SOLS, Solstice Advanced Materials
    SOLS on 2025-10-30: Solstice Advanced Materials Inc. Common Stock, listed None, CIK 0002064953, FIGI BBG01X5744P5, type CS, shares None
    SOLS on 2025-11-06: Solstice Advanced Materials Inc. Common Stock, listed 2025-10-20, CIK 0002064953, FIGI BBG01X5744P5, type CS, shares None
DD 2025-11-03 2.39: hypothesis Q, Qnity Electronics
    Q on 2025-11-03: Qnity Electronics, Inc., listed None, CIK None, FIGI BBG01TRL73S5, type CS, shares None
    Q on 2025-11-10: Qnity Electronics, Inc., listed 2025-10-27, CIK None, FIGI BBG01TRL73S5, type CS, shares None
```

### len_event.txt -- LEN: Millrose's listing, LEN's distributions, LEN's largest gaps

```text
MRP on 2025-02-10: Millrose Properties, Inc., listed 2025-02-05, CIK 0002017206, FIGI BBG01RQYH2X7
MRP on 2025-02-14: Millrose Properties, Inc., listed 2025-02-05, CIK 0002017206, FIGI BBG01RQYH2X7
LEN dividend: ex 2025-04-23 pay 2025-05-07 0.5 USD type CD freq 4
LEN dividend: ex 2025-01-29 pay 2025-02-12 0.5 USD type CD freq 4
LEN dividend: ex 2024-10-09 pay 2024-10-24 0.5 USD type CD freq 4
LEN's five largest raw opening gaps down, 2025-01-02 to 2025-03-14:
  2025-01-21: open gap -3.91%, raw close-to-close -5.93%, adjusted -2.83%
  2025-02-19: open gap -2.96%, raw close-to-close -1.27%, adjusted -1.27%
  2025-02-12: open gap -2.31%, raw close-to-close -2.66%, adjusted -2.66%
  2025-02-03: open gap -2.01%, raw close-to-close -3.85%, adjusted -3.85%
  2025-01-31: open gap -1.25%, raw close-to-close -2.97%, adjusted -2.97%
```

### len_millrose.txt -- LEN against three homebuilders around both dates; Millrose's numbers

```text
LEN against DHI, PHM and NVR (raw): the opening gap and the day's return, and LEN's less the peers' mean
  day         LEN gap   peers    less    LEN day   peers    less   LEN adj day
  2025-01-14   +1.92%  +2.39%  -0.47%     +2.74%  +2.98%  -0.24%        +2.74%
  2025-01-15   +5.14%  +3.64%  +1.50%     +2.51%  +2.02%  +0.48%        +2.51%
  2025-01-16   -0.03%  -0.55%  +0.52%     +1.53%  +0.94%  +0.59%        +1.53%
  2025-01-17   +1.34%  +1.55%  -0.20%     +0.60%  +0.30%  +0.31%        +0.60%
  2025-01-21   -3.91%  +1.99%  -5.90%     -5.93%  -1.30%  -4.63%        -2.83%  <-
  2025-01-22   -0.21%  -0.40%  +0.19%     +2.25%  +1.20%  +1.05%        +2.25%
  2025-01-23   -1.20%  -0.52%  -0.67%     -1.19%  -0.63%  -0.56%        -1.19%
  2025-01-24   +0.05%  -0.07%  +0.12%     -1.60%  -1.35%  -0.25%        -1.60%
  2025-01-27   +0.45%  -0.08%  +0.53%     +4.13%  +2.43%  +1.69%        +4.13%
  2025-01-28   -0.07%  +0.23%  -0.29%     -1.30%  -2.86%  +1.56%        -1.30%
  2025-01-29   -0.25%  +0.01%  -0.26%     -2.79%  -1.47%  -1.32%        -2.79%
  2025-01-30   +0.01%  +1.41%  -1.41%     +2.29%  +3.30%  -1.01%        +2.29%
  2025-01-31   -1.25%  -0.85%  -0.40%     -2.97%  -2.76%  -0.21%        -2.97%
  2025-02-03   -2.01%  -1.56%  -0.45%     -3.85%  -3.56%  -0.29%        -3.85%
  2025-02-04   -0.15%  -0.23%  +0.08%     +1.66%  +1.15%  +0.51%        +1.66%
  2025-02-05   +1.47%  +1.30%  +0.17%     -0.62%  +0.09%  -0.72%        -0.62%
  2025-02-06   +0.09%  +0.03%  +0.05%     -0.19%  -1.27%  +1.08%        -0.19%
  2025-02-07   -0.57%  -0.58%  +0.00%     -4.17%  -3.79%  -0.38%        -4.17%
  2025-02-10   +0.84%  +0.82%  +0.01%     +0.62%  +0.36%  +0.26%        +0.62%
  2025-02-11   -0.39%  -0.43%  +0.04%     +1.82%  +0.23%  +1.58%        +1.82%
  2025-02-12   -2.31%  -2.34%  +0.04%     -2.66%  -2.00%  -0.65%        -2.66%
  2025-02-13   +1.90%  +0.67%  +1.23%     +1.77%  +0.82%  +0.95%        +1.77%
  2025-02-14   +0.55%  +0.68%  -0.13%     +0.07%  +0.66%  -0.58%        +0.07%
  2025-02-18   -1.01%  -0.46%  -0.54%     -1.08%  -1.22%  +0.14%        -1.08%
  2025-02-19   -2.96%  -2.85%  -0.11%     -1.27%  -1.40%  +0.12%        -1.27%
  2025-02-20   -0.01%  -0.30%  +0.30%     +1.45%  +1.17%  +0.27%        +1.45%
  2025-02-21   +0.44%  +0.39%  +0.05%     -1.73%  -2.31%  +0.58%        -1.73%

Millrose (MRP): its record and first prices
  ref-MRP-2025-02-10.json: shares (class) 154180000, weighted 166000265, market cap 3819666097.65, list 2025-02-05
  ref-MRP-2025-02-14.json: shares (class) 154180000, weighted 166000265, market cap 3804726073.8, list 2025-02-05
  MRP 2025-02-07: open 23.49 close 26.74 volume 41,730,735
  MRP 2025-02-10: open 25.70 close 23.01 volume 15,683,028
  MRP 2025-02-11: open 22.08 close 22.50 volume 13,758,085
  MRP 2025-02-12: open 22.00 close 21.79 volume 7,847,431
  MRP 2025-02-13: open 21.57 close 22.72 volume 5,841,722
  MRP 2025-02-14: open 23.00 close 22.92 volume 5,650,206
  LEN class A shares on 2025-04-09 (sample): 232,180,000; the 1.033 step implies 4.52 a LEN share at the 2025-01-17 close
```

### identity_trace.txt -- each failing span's samples, in date order

```text
18 spans fail

COHR 2022-09-08..2026-11-03, 157 member-days, expects BBG000BLW102 (Coherent Corp.); source ['events']
   ! 2022-09-08..2022-09-08 ( 1) BBG000GS6QB4     Coherent Corp. Common Stock                  CIK
   = 2022-12-07..2026-11-03 (17) BBG000BLW102     Coherent Corp. Common Stock                  CIK 0000820318
     table has no row for BBG000GS6QB4

S600 2023-11-20..2026-11-03, 0 member-days, expects BBG0019LSQ49 (First Republic Bank); source ['panel', 'events', 'events']
   ! 2023-11-20..2026-11-03 (14) no record                                                     CIK

GEV 2024-03-27..2026-11-03, 651 member-days, expects BBG013G17W31 (GE Vernova Inc.); source ['events']
   ! 2024-03-27..2024-03-27 ( 1) no record                                                     CIK
   = 2024-06-27..2026-11-03 (11) BBG013G17W31     GE Vernova Inc.                              CIK 0001996810

CEGV 2022-01-19..2022-02-01, 0 member-days, expects BBG014KFRNP7 (Constellation Energy Corporation Common Stock); source ['events', 'events']
   ! 2022-01-19..2022-02-01 ( 2) no record                                                     CIK

SOLV 2024-03-26..2026-11-03, 652 member-days, expects BBG018YZH6T3 (Solventum Corporation); source ['events']
   ! 2024-03-26..2024-03-26 ( 1) no record                                                     CIK
   = 2024-06-26..2026-11-03 (11) BBG018YZH6T3     Solventum Corporation                        CIK 0001964738

VLTO 2023-09-27..2026-11-03, 776 member-days, expects BBG019Q32XJ9 (Veralto Corporation); source ['events']
   ! 2023-09-27..2023-09-27 ( 1) no record                                                     CIK
   = 2023-12-27..2026-11-03 (13) BBG019Q32XJ9     Veralto Corporation                          CIK 0001967680

MBC 2022-12-09..2026-11-03, 2 member-days, expects BBG01B9G42Y3 (MasterBrand, Inc.); source ['events']
   ! 2022-12-09..2022-12-09 ( 1) no record                                                     CIK
   = 2023-03-14..2026-11-03 (16) BBG01B9G42Y3     MasterBrand, Inc.                            CIK 0001941365

GEHC 2022-12-15..2026-11-03, 962 member-days, expects BBG01BFR8YV1 (GE HealthCare Technologies Inc. Common Stock); source ['events']
   ! 2022-12-15..2022-12-15 ( 1) no record                                                     CIK
   = 2023-03-20..2026-11-03 (16) BBG01BFR8YV1     GE HealthCare Technologies Inc. Common Stock CIK 0001932393

AMTM 2024-09-24..2026-11-03, 59 member-days, expects BBG01NY7B734 (Amentum Holdings, Inc.); source ['events']
   ! 2024-09-24..2024-09-24 ( 1) no record                                                     CIK
   = 2024-12-23..2026-11-03 ( 9) BBG01NY7B734     Amentum Holdings, Inc.                       CIK 0002011286

SNDK 2025-02-13..2026-11-03, 234 member-days, expects BBG01R388JG1 (Sandisk Corporation Common Stock); source ['events']
   ! 2025-02-13..2025-02-13 ( 1) no record                                                     CIK
   = 2025-05-15..2026-11-03 ( 7) BBG01R388JG1     Sandisk Corporation Common Stock             CIK 0002023554

Q 2025-10-27..2026-11-03, 252 member-days, expects BBG01TRL73S5 (Qnity Electronics, Inc.); source ['events']
   ! 2025-10-27..2025-10-27 ( 1) no record                                                     CIK
   = 2026-01-28..2026-11-03 ( 5) BBG01TRL73S5     Qnity Electronics, Inc.                      CIK 0002058873

SOLS 2025-10-20..2026-11-03, 36 member-days, expects BBG01X5744P5 (Solstice Advanced Materials Inc. Common Stock); source ['events']
   ! 2025-10-20..2025-10-20 ( 1) no record                                                     CIK
   = 2026-01-21..2026-11-03 ( 5) BBG01X5744P5     Solstice Advanced Materials Inc. Common Stoc CIK 0002064953

FDXF 2026-05-27..2026-11-03, 109 member-days, expects BBG01ZPB91C4 (FedEx Freight Holding Company, Inc.); source ['events']
   ! 2026-05-27..2026-05-27 ( 1) no record                                                     CIK
   = 2026-08-26..2026-11-03 ( 2) BBG01ZPB91C4     FedEx Freight Holding Company, Inc.          CIK 0002082247

HONA 2026-06-15..2026-11-03, 90 member-days, expects BBG020QYV152 (Honeywell Aerospace Inc. Common Stock); source ['events']
   ! 2026-06-15..2026-06-15 ( 1) no record                                                     CIK
   = 2026-09-15..2026-11-03 ( 2) BBG020QYV152     Honeywell Aerospace Inc. Common Stock        CIK 0002089271

VYLR 2026-10-01..2026-11-03, 24 member-days, expects BBG024CQR6X9 (Vylor Inc.); source ['events']
   ! 2026-10-01..2026-10-01 ( 1) BBG024W5VJC3     Vylor Inc.                                   CIK 0002128626
   = 2026-11-03..2026-11-03 ( 1) BBG024CQR6X9     Vylor Inc.                                   CIK 0002128626
     table has no row for BBG024W5VJC3

CRHC 2022-01-03..2023-09-22, 0 member-days, expects CIK0000849395 (CRH Public Limited Company); source ['events', 'events']
   ! 2022-01-03..2022-10-04 ( 4) CIK0001818212    Cohn Robbins Holdings Corp.                  CIK 0001818212
   ! 2023-01-04..2023-09-22 ( 4) no record                                                     CIK
     table has no row for CIK0001818212

EG 2023-07-10..2026-11-03, 835 member-days, expects CIK0001095073 (Everest Group, Ltd.); source ['events', 'panel']
   ! 2023-07-10..2023-07-10 ( 1) no record        Everest Group, Ltd.                          CIK
   = 2023-10-05..2026-11-03 (14) CIK0001095073    Everest Group, Ltd.                          CIK 0001095073

BG 2023-11-01..2026-11-03, 754 member-days, expects CIK0001996862 (Bunge Global SA); source ['events']
   ! 2023-11-01..2023-11-01 ( 1) CIK0001144519    Bunge Global SA                              CIK 0001144519
   = 2024-02-02..2026-11-03 (12) CIK0001996862    Bunge Global SA                              CIK 0001996862
     table has CIK0001144519: Bunge Limited under BG 2022-01-03..2023-10-31 (reference)

```

### identity_detail.txt -- each failing span's table rows, events, list dates and prices

```text
=== COHR 2022-09-08..2026-11-03 (157 member-days), expects BBG000BLW102, Coherent Corp.
  table row: COHR 2022-09-08..2026-11-03 held 0 source events; starts: the first of its events, 2022-09-08: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under COHR: Coherent Corp., FIGI BBG000BLW102, CIK 0000820318: [('2022-09-08', 'ticker_change', 'COHR')]
  list dates on its records (FIGI, CIK, list date): [('BBG000BLW102', '0000820318', '1987-10-02')]
  prices under COHR: 1168 days in the window (2022-01-03..2026-11-04); in the span 1043 (2022-09-08..2026-11-03), before it 124, after it 1

=== S600 2023-11-20..2026-11-03 (0 member-days), expects BBG0019LSQ49, First Republic Bank
  table row: FRC 2022-01-03..2023-04-28 held 332 source panel; starts: the window's first day; ends: First Republic Bank traded as another ticker from 2023-05-03 (events)
  table row: FRCB 2023-05-03..2023-11-17 held 0 source events; starts: began trading as FRCB on 2023-05-03 (events); ends: traded as another ticker from 2023-11-18 (events)
  table row: S600 2023-11-20..2026-11-03 held 0 source events; starts: began trading as S600 on 2023-11-18 (events); ends: the window's last day
  events-S600: not cached
  list dates on its records (FIGI, CIK, list date): []
  prices under S600: 0 days in the window (-..-); in the span 0 (-..-), before it 0, after it 0

=== GEV 2024-03-27..2026-11-03 (651 member-days), expects BBG013G17W31, GE Vernova Inc.
  table row: GEV 2024-03-27..2026-11-03 held 0 source events; starts: the first of its events, 2024-03-27: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under GEV: GE Vernova Inc., FIGI BBG013G17W31, CIK 0001996810: [('2024-03-27', 'ticker_change', 'GEV')]
  list dates on its records (FIGI, CIK, list date): [('BBG013G17W31', '0001996810', '2024-03-27')]
  prices under GEV: 652 days in the window (2024-04-02..2026-11-04); in the span 651 (2024-04-02..2026-11-03), before it 0, after it 1

=== CEGV 2022-01-19..2022-02-01 (0 member-days), expects BBG014KFRNP7, Constellation Energy Corporation Common Stock
  table row: CEGV 2022-01-19..2022-02-01 held 0 source events; starts: the first of its events, 2022-01-19: the start of Polygon's record of it, not necessarily of the ticker; ends: traded as another ticker from 2022-02-02 (events)
  table row: CEG 2022-02-02..2026-11-03 held 0 source events; starts: began trading as CEG on 2022-02-02 (events); ends: the window's last day
  events-CEGV: not cached
  list dates on its records (FIGI, CIK, list date): []
  prices under CEGV: 0 days in the window (-..-); in the span 0 (-..-), before it 0, after it 0

=== SOLV 2024-03-26..2026-11-03 (652 member-days), expects BBG018YZH6T3, Solventum Corporation
  table row: SOLV 2024-03-26..2026-11-03 held 0 source events; starts: the first of its events, 2024-03-26: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under SOLV: Solventum Corporation, FIGI BBG018YZH6T3, CIK 0001964738: [('2024-03-26', 'ticker_change', 'SOLV')]
  list dates on its records (FIGI, CIK, list date): [('BBG018YZH6T3', '0001964738', '2024-03-26')]
  prices under SOLV: 653 days in the window (2024-04-01..2026-11-04); in the span 652 (2024-04-01..2026-11-03), before it 0, after it 1

=== VLTO 2023-09-27..2026-11-03 (776 member-days), expects BBG019Q32XJ9, Veralto Corporation
  table row: VLTO 2023-09-27..2026-11-03 held 0 source events; starts: the first of its events, 2023-09-27: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under VLTO: Veralto Corporation, FIGI BBG019Q32XJ9, CIK 0001967680: [('2023-09-27', 'ticker_change', 'VLTO')]
  list dates on its records (FIGI, CIK, list date): [('BBG019Q32XJ9', '0001967680', '2023-09-27')]
  prices under VLTO: 777 days in the window (2023-10-02..2026-11-04); in the span 776 (2023-10-02..2026-11-03), before it 0, after it 1

=== MBC 2022-12-09..2026-11-03 (2 member-days), expects BBG01B9G42Y3, MasterBrand, Inc.
  table row: MBC 2022-12-09..2026-11-03 held 0 source events; starts: the first of its events, 2022-12-09: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under MBC: MasterBrand, Inc., FIGI BBG01B9G42Y3, CIK 0001941365: [('2022-12-09', 'ticker_change', 'MBC')]
  list dates on its records (FIGI, CIK, list date): [('BBG01B9G42Y3', '0001941365', '2022-12-09')]
  prices under MBC: 975 days in the window (2022-12-15..2026-11-04); in the span 974 (2022-12-15..2026-11-03), before it 0, after it 1

=== GEHC 2022-12-15..2026-11-03 (962 member-days), expects BBG01BFR8YV1, GE HealthCare Technologies Inc. Common Stock
  table row: GEHC 2022-12-15..2026-11-03 held 0 source events; starts: the first of its events, 2022-12-15: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under GEHC: GE HealthCare Technologies Inc. Common Stock, FIGI BBG01BFR8YV1, CIK 0001932393: [('2022-12-15', 'ticker_change', 'GEHC')]
  list dates on its records (FIGI, CIK, list date): [('BBG01BFR8YV1', '0001932393', '2022-12-15')]
  prices under GEHC: 963 days in the window (2023-01-04..2026-11-04); in the span 962 (2023-01-04..2026-11-03), before it 0, after it 1

=== AMTM 2024-09-24..2026-11-03 (59 member-days), expects BBG01NY7B734, Amentum Holdings, Inc.
  table row: AMTM 2024-09-24..2026-11-03 held 0 source events; starts: the first of its events, 2024-09-24: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under AMTM: Amentum Holdings, Inc., FIGI BBG01NY7B734, CIK 0002011286: [('2024-09-24', 'ticker_change', 'AMTM')]
  list dates on its records (FIGI, CIK, list date): [('BBG01NY7B734', '0002011286', '2024-09-24')]
  prices under AMTM: 527 days in the window (2024-09-30..2026-11-04); in the span 526 (2024-09-30..2026-11-03), before it 0, after it 1

=== SNDK 2025-02-13..2026-11-03 (234 member-days), expects BBG01R388JG1, Sandisk Corporation Common Stock
  table row: SNDK 2025-02-13..2026-11-03 held 0 source events; starts: the first of its events, 2025-02-13: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under SNDK: Sandisk Corporation Common Stock, FIGI BBG01R388JG1, CIK 0002023554: [('2025-02-13', 'ticker_change', 'SNDK')]
  list dates on its records (FIGI, CIK, list date): [('BBG01R388JG1', '0002023554', '2025-02-13')]
  prices under SNDK: 428 days in the window (2025-02-24..2026-11-04); in the span 427 (2025-02-24..2026-11-03), before it 0, after it 1

=== Q 2025-10-27..2026-11-03 (252 member-days), expects BBG01TRL73S5, Qnity Electronics, Inc.
  table row: Q 2025-10-27..2026-11-03 held 0 source events; starts: the first of its events, 2025-10-27: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under Q: Qnity Electronics, Inc., FIGI BBG01TRL73S5, CIK 0002058873: [('2025-10-27', 'ticker_change', 'Q')]
  list dates on its records (FIGI, CIK, list date): [('BBG01TRL73S5', '0002058873', '2025-10-27')]
  prices under Q: 253 days in the window (2025-11-03..2026-11-04); in the span 252 (2025-11-03..2026-11-03), before it 0, after it 1

=== SOLS 2025-10-20..2026-11-03 (36 member-days), expects BBG01X5744P5, Solstice Advanced Materials Inc. Common Stock
  table row: SOLS 2025-10-20..2026-11-03 held 0 source events; starts: the first of its events, 2025-10-20: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under SOLS: Solstice Advanced Materials Inc. Common Stock, FIGI BBG01X5744P5, CIK 0002064953: [('2025-10-20', 'ticker_change', 'SOLS')]
  list dates on its records (FIGI, CIK, list date): [('BBG01X5744P5', '0002064953', '2025-10-20')]
  prices under SOLS: 599 days in the window (2022-01-03..2026-11-04); in the span 254 (2025-10-30..2026-11-03), before it 344, after it 1

=== FDXF 2026-05-27..2026-11-03 (109 member-days), expects BBG01ZPB91C4, FedEx Freight Holding Company, Inc.
  table row: FDXF 2026-05-27..2026-11-03 held 0 source events; starts: the first of its events, 2026-05-27: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under FDXF: FedEx Freight Holding Company, Inc., FIGI BBG01ZPB91C4, CIK 0002082247: [('2026-05-27', 'ticker_change', 'FDXF')]
  list dates on its records (FIGI, CIK, list date): [('BBG01ZPB91C4', '0002082247', '2026-05-27')]
  prices under FDXF: 110 days in the window (2026-06-01..2026-11-04); in the span 109 (2026-06-01..2026-11-03), before it 0, after it 1

=== HONA 2026-06-15..2026-11-03 (90 member-days), expects BBG020QYV152, Honeywell Aerospace Inc. Common Stock
  table row: HONA 2026-06-15..2026-11-03 held 0 source events; starts: the first of its events, 2026-06-15: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under HONA: Honeywell Aerospace Inc. Common Stock, FIGI BBG020QYV152, CIK 0002089271: [('2026-06-15', 'ticker_change', 'HONA')]
  list dates on its records (FIGI, CIK, list date): [('BBG020QYV152', '0002089271', '2026-06-15')]
  prices under HONA: 91 days in the window (2026-06-29..2026-11-04); in the span 90 (2026-06-29..2026-11-03), before it 0, after it 1

=== VYLR 2026-10-01..2026-11-03 (24 member-days), expects BBG024CQR6X9, Vylor Inc.
  table row: VYLR 2026-10-01..2026-11-03 held 0 source events; starts: the first of its events, 2026-10-01: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  events under VYLR: Vylor Inc., FIGI BBG024CQR6X9, CIK 0002128626: [('2026-10-01', 'ticker_change', 'VYLR')]
  list dates on its records (FIGI, CIK, list date): [('BBG024CQR6X9', '0002128626', '2026-10-01')]
  prices under VYLR: 25 days in the window (2026-10-01..2026-11-04); in the span 24 (2026-10-01..2026-11-03), before it 0, after it 1

=== CRHC 2022-01-03..2023-09-22 (0 member-days), expects CIK0000849395, CRH Public Limited Company
  table row: CRHC 2022-01-03..2023-09-22 held 0 source events; starts: the window's first day (as CRHC from 2021-09-23); ends: traded as another ticker from 2023-09-25 (events)
  table row: CRH 2023-09-25..2026-11-03 held 0 source events; starts: began trading as CRH on 2023-09-25 (events); ends: the window's last day
  events-CRHC: not cached
  list dates on its records (FIGI, CIK, list date): [(None, '0001818212', '2020-10-30')]
  prices under CRHC: 216 days in the window (2022-01-03..2022-11-09); in the span 216 (2022-01-03..2022-11-09), before it 0, after it 0

=== EG 2023-07-10..2026-11-03 (835 member-days), expects CIK0001095073, Everest Group, Ltd.
  table row: RE 2022-01-03..2023-07-07 held 0 source events; starts: the window's first day (as RE from 2003-09-10); ends: traded as another ticker from 2023-07-10 (events)
  table row: EG 2023-07-10..2026-11-03 held 835 source panel; starts: Everest Group, Ltd. began trading as EG on 2023-07-10 (events: a change from another ticker); ends: the window's last day
  events under EG: Everest Group, Ltd., FIGI None, CIK 0001095073: [('2003-09-10', 'ticker_change', 'RE'), ('2023-07-10', 'ticker_change', 'EG')]
  list dates on its records (FIGI, CIK, list date): [(None, '0001095073', '1995-10-03'), (None, None, '1995-10-03')]
  prices under EG: 835 days in the window (2023-07-10..2026-11-03); in the span 835 (2023-07-10..2026-11-03), before it 0, after it 0

=== BG 2023-11-01..2026-11-03 (754 member-days), expects CIK0001996862, Bunge Global SA
  table row: BG 2023-11-01..2026-11-03 held 0 source events; starts: the first of its events, 2023-11-01: the start of Polygon's record of it, not necessarily of the ticker; ends: the window's last day
  other row under BG: CIK0001144519 Bunge Limited 2022-01-03..2023-10-31 (reference)
  events under BG: Bunge Global SA, FIGI None, CIK 0001996862: [('2023-11-01', 'ticker_change', 'BG')]
  list dates on its records (FIGI, CIK, list date): [(None, '0001144519', '2001-08-02'), (None, '0001996862', '2001-08-02')]
  prices under BG: 1215 days in the window (2022-01-03..2026-11-04); in the span 754 (2023-11-01..2026-11-03), before it 460, after it 1

```

### all_samples_scan.txt -- every sample of every span against the table's identifier

```text
COHR   2022-09-08..2026-11-03 md  157 expects BBG000BLW102 (Coherent Corp.); 1 of 18 samples name otherwise  [first sample only]
    2022-09-08..2022-09-08 ( 1) BBG000GS6QB4     CIK None  Coherent Corp. Common Stock
EA     2022-01-03..2026-08-04 md 1150 expects BBG000BP0KQ8 (Electronic Arts Inc); 1 of 20 samples name otherwise
    2026-08-04..2026-08-04 ( 1) BBG000K1X329     CIK 0000712515  Electronic Arts Inc
OKE    2022-01-03..2026-09-10 md 1176 expects BBG000BQHGR6 (Oneok, Inc.); 1 of 20 samples name otherwise
    2026-09-10..2026-09-10 ( 1) BBG000JBSHW1     CIK 0001039684  Oneok, Inc.
JNPR   2022-01-03..2025-07-01 md  880 expects BBG000BY33P5 (Juniper Networks Inc); 1 of 15 samples name otherwise
    2025-07-01..2025-07-01 ( 1) BBG000BWS318     CIK 0001043604  Juniper Networks Inc
BLK    2022-01-03..2024-10-01 md  690 expects BBG000C2PW58 (Blackrock, Inc.); 1 of 12 samples name otherwise
    2024-10-01..2024-10-01 ( 1) BBG000CN81W9     CIK 0001364742  Blackrock, Inc.
XOM    2022-01-03..2026-07-01 md 1127 expects BBG000GZQ728 (Exxon Mobil Corporation); 1 of 19 samples name otherwise
    2026-07-01..2026-07-01 ( 1) BBG000H5HTR8     CIK 0000034088  Exxon Mobil Corporation
S600   2023-11-20..2026-11-03 md    0 expects BBG0019LSQ49 (First Republic Bank); 14 of 14 samples name otherwise
    2023-11-20..2026-11-03 (14) no record        CIK None
DISCA  2022-01-03..2022-04-08 md   68 expects BBG00HW3TQG4 (Discovery, Inc. Series A Common ); 1 of 3 samples name otherwise
    2022-04-04..2022-04-04 ( 1) BBG000CHWP52     CIK 0001024333  Discovery, Inc. Series A Common Stock
GEV    2024-03-27..2026-11-03 md  651 expects BBG013G17W31 (GE Vernova Inc.); 1 of 12 samples name otherwise  [first sample only]
    2024-03-27..2024-03-27 ( 1) no record        CIK None
CEGV   2022-01-19..2022-02-01 md    0 expects BBG014KFRNP7 (Constellation Energy Corporation); 2 of 2 samples name otherwise
    2022-01-19..2022-02-01 ( 2) no record        CIK None
SOLV   2024-03-26..2026-11-03 md  652 expects BBG018YZH6T3 (Solventum Corporation); 1 of 12 samples name otherwise  [first sample only]
    2024-03-26..2024-03-26 ( 1) no record        CIK None
VLTO   2023-09-27..2026-11-03 md  776 expects BBG019Q32XJ9 (Veralto Corporation); 1 of 14 samples name otherwise  [first sample only]
    2023-09-27..2023-09-27 ( 1) no record        CIK None
MBC    2022-12-09..2026-11-03 md    2 expects BBG01B9G42Y3 (MasterBrand, Inc.); 1 of 17 samples name otherwise  [first sample only]
    2022-12-09..2022-12-09 ( 1) no record        CIK None
GEHC   2022-12-15..2026-11-03 md  962 expects BBG01BFR8YV1 (GE HealthCare Technologies Inc. ); 1 of 17 samples name otherwise  [first sample only]
    2022-12-15..2022-12-15 ( 1) no record        CIK None
AMTM   2024-09-24..2026-11-03 md   59 expects BBG01NY7B734 (Amentum Holdings, Inc.); 1 of 10 samples name otherwise  [first sample only]
    2024-09-24..2024-09-24 ( 1) no record        CIK None
SNDK   2025-02-13..2026-11-03 md  234 expects BBG01R388JG1 (Sandisk Corporation Common Stock); 1 of 8 samples name otherwise  [first sample only]
    2025-02-13..2025-02-13 ( 1) no record        CIK None
Q      2025-10-27..2026-11-03 md  252 expects BBG01TRL73S5 (Qnity Electronics, Inc.); 1 of 6 samples name otherwise  [first sample only]
    2025-10-27..2025-10-27 ( 1) no record        CIK None
SOLS   2025-10-20..2026-11-03 md   36 expects BBG01X5744P5 (Solstice Advanced Materials Inc.); 1 of 6 samples name otherwise  [first sample only]
    2025-10-20..2025-10-20 ( 1) no record        CIK None
FDXF   2026-05-27..2026-11-03 md  109 expects BBG01ZPB91C4 (FedEx Freight Holding Company, I); 1 of 3 samples name otherwise  [first sample only]
    2026-05-27..2026-05-27 ( 1) no record        CIK None
HONA   2026-06-15..2026-11-03 md   90 expects BBG020QYV152 (Honeywell Aerospace Inc. Common ); 1 of 3 samples name otherwise  [first sample only]
    2026-06-15..2026-06-15 ( 1) no record        CIK None
VYLR   2026-10-01..2026-11-03 md   24 expects BBG024CQR6X9 (Vylor Inc.); 1 of 2 samples name otherwise  [first sample only]
    2026-10-01..2026-10-01 ( 1) BBG024W5VJC3     CIK 0002128626  Vylor Inc.
CRHC   2022-01-03..2023-09-22 md    0 expects CIK0000849395 (CRH Public Limited Company); 8 of 8 samples name otherwise
    2022-01-03..2022-10-04 ( 4) CIK0001818212    CIK 0001818212  Cohn Robbins Holdings Corp.
    2023-01-04..2023-09-22 ( 4) no record        CIK None
EG     2023-07-10..2026-11-03 md  835 expects CIK0001095073 (Everest Group, Ltd.); 1 of 15 samples name otherwise  [first sample only]
    2023-07-10..2023-07-10 ( 1) no identifier    CIK None  Everest Group, Ltd.
BG     2023-11-01..2026-11-03 md  754 expects CIK0001996862 (Bunge Global SA); 1 of 13 samples name otherwise  [first sample only]
    2023-11-01..2023-11-01 ( 1) CIK0001144519    CIK 0001144519  Bunge Global SA

11,968 samples over 629 spans; 24 spans have a sample naming another identifier than the table's: {'first sample only': 15, 'later samples': 9}
```

### split_lag.txt -- each contradiction's samples until one follows; rounding by quarter

```text
Contradictions: the samples from the last before the split until one follows it

FAST 1-to-2 on 2025-05-22 (Fastenal Co)
   2025-04-09 FAST:      573,450,000    1.000 x the count before
   2025-07-11 FAST:      573,620,638    1.000 x the count before
   2025-10-09 FAST:    1,147,636,617    2.001 x the count before  follows
   -> the count follows the split 140 days after it

ORLY 1-to-15 on 2025-06-10 (O'Reilly Automotive, Inc.)
   2025-04-09 ORLY:       57,240,000    1.000 x the count before
   2025-07-11 ORLY:       56,989,455    0.996 x the count before
   2025-10-09 ORLY:      848,496,680   14.823 x the count before  follows
   -> the count follows the split 121 days after it

IBKR 1-to-4 on 2025-06-18 (Interactive Brokers Group, Inc. Class A Common Stock)
   2025-04-09 IBKR:      108,930,000    1.000 x the count before
   2025-07-11 IBKR:      108,958,830    1.000 x the count before
   2025-10-09 IBKR:      445,246,976    4.087 x the count before  follows
   -> the count follows the split 113 days after it

NOW 1-to-5 on 2025-12-18 (SERVICENOW, INC.)
   2025-10-09 NOW:      208,000,000    1.000 x the count before
   2026-01-09 NOW:      207,678,139    0.998 x the count before
   2026-04-13 NOW:    1,036,155,572    4.982 x the count before  follows
   -> the count follows the split 116 days after it

TPL 1-to-3 on 2025-12-23 (Texas Pacific Land Corporation)
   2025-10-09 TPL:       22,987,326    1.000 x the count before
   2026-01-09 TPL:       22,979,410    1.000 x the count before
   2026-04-13 TPL:       68,941,554    2.999 x the count before  follows
   -> the count follows the split 111 days after it

BKNG 1-to-25 on 2026-04-06 (Booking Holdings Inc. Common Stock)
   2026-01-09 BKNG:       32,233,815    1.000 x the count before
   2026-04-13 BKNG:       31,673,346    0.983 x the count before
   2026-07-14 BKNG:      774,878,436   24.039 x the count before  follows
   -> the count follows the split 99 days after it

CVNA 1-to-5 on 2026-05-08 (Carvana Co.)
   2026-04-13 CVNA:      142,993,769    1.000 x the count before
   2026-07-14 CVNA:      143,257,677    1.002 x the count before
   2026-10-12 CVNA:      719,916,415    5.035 x the count before  follows
   -> the count follows the split 157 days after it

KLAC 1-to-10 on 2026-06-12 (KLA Corporation Common Stock)
   2026-04-13 KLAC:      131,076,611    1.000 x the count before
   2026-07-14 KLAC:      130,627,521    0.997 x the count before
   2026-10-12 KLAC:    1,305,006,492    9.956 x the count before  follows
   -> the count follows the split 122 days after it

CRWD 1-to-4 on 2026-07-02 (CrowdStrike Holdings, Inc. Class A Common Stock)
   2026-04-13 CRWD:      253,614,090    1.000 x the count before
   2026-07-14 CRWD:      254,564,820    1.004 x the count before
   2026-10-12 CRWD:    1,026,129,839    4.046 x the count before  follows
   -> the count follows the split 102 days after it

MNST 1-to-2 on 2026-08-11 (Monster Beverage Corporation)
   2026-07-14 MNST:      978,008,110    1.000 x the count before
   2026-10-12 MNST:      979,525,882    1.002 x the count before
   2026-11-03 MNST:      979,525,882    1.002 x the count before
   -> no sample in the span follows it

APH 1-to-2 on 2026-09-03 (Amphenol Corporation)
   2026-07-14 APH:    1,230,234,445    1.000 x the count before
   2026-10-12 APH:    1,232,983,457    1.002 x the count before
   2026-11-03 APH:    1,232,983,457    1.002 x the count before
   -> no sample in the span follows it

Every split event: are the counts either side rounded to 10,000 shares?
  2022-03-24 WRB    split         before rounded  after rounded
  2022-06-06 AMZN   split         before rounded  after rounded
  2022-06-13 DXCM   split         before rounded  after rounded
  2022-06-23 FTNT   split         before rounded  after rounded
  2022-07-18 GOOG   split         before rounded  after rounded
  2022-07-18 GOOGL  split         before rounded  after rounded
  2022-08-25 TSLA   split         before rounded  after rounded
  2022-08-29 NDAQ   split         before rounded  after rounded
  2022-09-14 PANW   split         before rounded  after rounded
  2022-11-04 CPRT   split         before rounded  after rounded
  2022-11-30 TECH   split         before rounded  after rounded
  2023-01-04 GE     price-only    before rounded  after rounded
  2023-02-08 PCAR   split         before rounded  after rounded
  2023-03-28 MNST   split         before rounded  after exact
  2023-07-03 LH     price-only    before rounded  after rounded
  2023-08-22 CPRT   split         before rounded  after rounded
  2023-08-22 SRE    split         before rounded  after rounded
  2024-02-20 COO    split         before rounded  after rounded
  2024-02-26 WMT    split         before rounded  after rounded
  2024-03-27 TPL    split         before rounded  after rounded
  2024-03-28 ODFL   split         before rounded  after rounded
  2024-04-01 MMM    price-only    before rounded  after rounded
  2024-04-02 GE     price-only    before rounded  after rounded
  2024-06-10 NVDA   split         before rounded  after rounded
  2024-06-12 APH    split         before rounded  after rounded
  2024-06-26 CMG    split         before rounded  after rounded
  2024-07-09 WSM    split         before rounded  after rounded
  2024-07-11 WRB    split         before rounded  after rounded
  2024-07-15 AVGO   split         before rounded  after rounded
  2024-09-12 CTAS   split         before rounded  after rounded
  2024-09-17 DECK   split         before rounded  after rounded
  2024-10-01 SMCI   split         before rounded  after rounded
  2024-10-03 LRCX   split         before rounded  after rounded
  2024-12-04 ANET   split         before rounded  after rounded
  2024-12-13 ETR    split         before rounded  after rounded
  2024-12-16 PANW   split         before rounded  after rounded
  2024-12-20 TSCO   split         before rounded  after rounded
  2025-01-21 LEN    price-only    before rounded  after rounded
  2025-05-22 FAST   contradiction before rounded  after exact
  2025-06-10 ORLY   contradiction before rounded  after exact
  2025-06-18 IBKR   contradiction before rounded  after exact
  2025-06-30 FTV    price-only    before rounded  after exact
  2025-10-30 HON    price-only    before exact    after exact
  2025-11-03 DD     price-only    before exact    after exact
  2025-11-17 NFLX   split         before exact    after exact
  2025-12-18 NOW    contradiction before rounded  after exact
  2025-12-23 TPL    contradiction before exact    after exact
  2026-01-15 AMCR   split         before exact    after exact
  2026-04-06 BKNG   contradiction before exact    after exact
  2026-05-08 CVNA   contradiction before exact    after exact
  2026-06-12 KLAC   contradiction before exact    after exact
  2026-06-24 DD     split         before exact    after exact
  2026-07-02 CRWD   contradiction before exact    after exact
  2026-08-11 MNST   contradiction before exact    after exact
  2026-09-03 APH    contradiction before exact    after exact

Share samples rounded to 10,000 shares, by quarter (rounded / exact):
  2022Q1 569/8  2022Q2 567/12  2022Q3 562/9  2022Q4 562/11  2023Q1 555/11  2023Q2 557/10  2023Q3 563/4  2023Q4 561/8  2024Q1 566/3  2024Q2 562/7  2024Q3 564/6  2024Q4 569/4  2025Q1 555/11  2025Q2 554/12  2025Q3 33/536  2025Q4 34/531  2026Q1 33/529  2026Q2 23/542  2026Q3 35/525  2026Q4 65/1013
```

### boundary_ratios.txt -- the share counts either side of each reorganisation and link

```text
ticker predecessor      last sample            count   successor        first sample           count   ratio
BLK    BBG000C2PW58     2024-10-01       148,130,000   BBG01PSW2WN4     2024-10-02       148,130,000   1.0000  [BBG000CN81W9 | BBG01PSW2WN4]
CCL    BBG000BF6LY3     2026-05-06     1,239,000,212   CIK0000815097    2026-05-07     1,369,649,119   1.1054 (second sample, 2026-08-07)  [BBG000BF6LY3 | 0000815097]
J      BBG000BMFFQ0     2022-08-26       127,610,000   BBG019C1BQR4     2022-08-29       127,610,000   1.0000  [BBG000BMFFQ0 | BBG019C1BQR4]
LH     BBG000D9DMK0     2024-05-17        84,290,000   BBG01MMT6PL7     2024-05-20        84,290,000   1.0000  [BBG000D9DMK0 | BBG01MMT6PL7]
OKE    BBG000BQHGR6     2026-09-10       630,369,667   BBG024TZWVN1     2026-09-11                 0   - (second sample, 2026-11-03)  [BBG000JBSHW1 | BBG024TZWVN1]
XOM    BBG000GZQ728     2026-07-01     4,144,947,162   BBG023CY9MM1     2026-07-02                 0   - (second sample, 2026-10-01)  [BBG000H5HTR8 | BBG023CY9MM1]
WRK    BBG008NXC572     2024-07-05       258,149,999   CIK0002005951    2024-07-08       261,100,000   1.0114  [BBG008NXC572 | 0002005951]  WRK -> SW
PARA   BBG000C496P7     2025-08-06       633,522,201   BBG01VS5NK99     2025-08-07     1,064,544,631   1.6804 (second sample, 2025-11-05)  [BBG000C496P7 | BBG01VS5NK99]  PARA -> PSKY
BG     CIK0001144519    2023-10-31       145,290,000   CIK0001996862    2023-11-01       145,290,000   1.0000  [0001144519 | 0001144519]
```

### dividends_check.txt -- the distributions' split basis, same-day pairs, and yields

```text
Split basis: the last distribution before each split and the first after, by security
  WRB       1.5-for-1 on 2022-03-24: 0.1300 on 2022-03-08 then 0.1000 on 2022-06-24 -- before over after 1.300; as paid then (split ratio)
  NDAQ        3-for-1 on 2022-08-29: 0.6000 on 2022-06-09 then 0.2000 on 2022-09-15 -- before over after 3.000; as paid then (split ratio)
  TECH        4-for-1 on 2022-11-30: 0.3200 on 2022-11-10 then 0.0800 on 2023-02-10 -- before over after 4.000; as paid then (split ratio)
  PCAR      1.5-for-1 on 2023-02-08: 0.3700 on 2022-11-14 then 0.2500 on 2023-02-13 -- before over after 1.480; as paid then (split ratio)
  SRE         2-for-1 on 2023-08-22: 1.1900 on 2023-07-03 then 0.5950 on 2023-09-26 -- before over after 2.000; as paid then (split ratio)
  WMT         3-for-1 on 2024-02-26: 0.5700 on 2023-12-07 then 0.2075 on 2024-03-14 -- before over after 2.747; as paid then (split ratio)
  TPL         3-for-1 on 2024-03-27: 3.5000 on 2024-02-29 then 1.1700 on 2024-06-03 -- before over after 2.991; as paid then (split ratio)
  ODFL        2-for-1 on 2024-03-28: 0.5200 on 2024-03-05 then 0.2600 on 2024-06-05 -- before over after 2.000; as paid then (split ratio)
  NVDA       10-for-1 on 2024-06-10: 0.0400 on 2024-03-05 then 0.0100 on 2024-06-11 -- before over after 4.000; unclear
  APH         2-for-1 on 2024-06-12: 0.2200 on 2024-03-18 then 0.1100 on 2024-06-18 -- before over after 2.000; as paid then (split ratio)
  WSM         2-for-1 on 2024-07-09: 1.1300 on 2024-04-18 then 0.5700 on 2024-07-19 -- before over after 1.982; as paid then (split ratio)
  WRB       1.5-for-1 on 2024-07-11: 0.1200 on 2024-06-24 then 0.0800 on 2024-09-23 -- before over after 1.500; as paid then (split ratio)
  AVGO       10-for-1 on 2024-07-15: 5.2500 on 2024-06-24 then 0.5300 on 2024-09-19 -- before over after 9.906; as paid then (split ratio)
  CTAS        4-for-1 on 2024-09-12: 1.5600 on 2024-08-15 then 0.3900 on 2024-11-15 -- before over after 4.000; as paid then (split ratio)
  LRCX       10-for-1 on 2024-10-03: 2.3000 on 2024-09-17 then 0.2300 on 2024-12-11 -- before over after 10.000; as paid then (split ratio)
  ETR         2-for-1 on 2024-12-13: 1.2000 on 2024-11-13 then 0.6000 on 2025-02-10 -- before over after 2.000; as paid then (split ratio)
  TSCO        5-for-1 on 2024-12-20: 1.1000 on 2024-11-25 then 0.2300 on 2025-02-26 -- before over after 4.783; as paid then (split ratio)
  AMCR      0.2-for-1 on 2026-01-15: 0.1300 on 2025-11-28 then 0.6500 on 2026-02-25 -- before over after 0.200; as paid then (split ratio)
  DD     0.333333-for-1 on 2026-06-24: 0.2000 on 2026-05-15 then 0.6000 on 2026-08-31 -- before over after 0.333; as paid then (split ratio)

The same ticker and ex-date twice:
  CHK 2022-03-04: [(1.33, 'CD', 4), (0.4375, 'CD', 4)]
  CHK 2022-05-18: [(0.5, 'CD', 4), (1.84, 'CD', 4)]
  CHK 2022-08-16: [(1.77, 'CD', 4), (0.55, 'CD', 4)]
  CHK 2022-11-14: [(0.55, 'CD', 4), (2.61, 'CD', 4)]
  CHK 2023-03-06: [(0.74, 'CD', 4), (0.55, 'CD', 4)]
  CME 2026-03-10: [(1.3, 'CD', 4), (6.15, 'SC', 1)]
  COP 2024-02-15: [(0.58, 'CD', 4), (0.2, 'SC', 0)]
  COP 2024-08-12: [(0.2, 'CD', 4), (0.58, 'CD', 4)]
  CTRA 2022-03-04: [(0.15, 'CD', 4), (0.41, 'CD', 0)]
  CTRA 2022-05-12: [(0.45, 'CD', 0), (0.15, 'CD', 4)]
  CTRA 2022-08-12: [(0.15, 'CD', 4), (0.5, 'SC', 0)]
  CTRA 2022-11-15: [(0.15, 'CD', 4), (0.53, 'SC', 0)]
  CTRA 2023-03-15: [(0.37, 'SC', 0), (0.2, 'CD', 4)]
  DVN 2022-03-11: [(0.16, 'CD', 4), (0.84, 'CD', 4)]
  DVN 2022-06-10: [(0.16, 'CD', 4), (1.11, 'CD', 4)]
  DVN 2022-09-09: [(1.37, 'SC', 0), (0.18, 'CD', 4)]
  DVN 2022-12-09: [(0.18, 'CD', 4), (1.17, 'SC', 0)]
  DVN 2023-03-14: [(0.2, 'CD', 4), (0.69, 'SC', 0)]
  DVN 2023-06-14: [(0.52, 'CD', 4), (0.2, 'CD', 4)]
  DVN 2023-09-14: [(0.2, 'CD', 4), (0.29, 'CD', 4)]
  DVN 2023-12-14: [(0.2, 'CD', 4), (0.57, 'CD', 4)]
  DVN 2024-03-14: [(0.22, 'SC', 0), (0.22, 'SC', 4)]
  DVN 2024-06-14: [(0.13, 'SC', 0), (0.22, 'CD', 4)]
  DVN 2024-09-13: [(0.22, 'SC', 4), (0.22, 'CD', 4)]
  EXE 2025-08-14: [(0.89, 'CD', 4), (0.575, 'CD', 4)]
  F 2023-02-10: [(0.15, 'CD', 4), (0.65, 'SC', 0)]
  F 2024-02-15: [(0.18, 'SC', 0), (0.15, 'CD', 4)]
  F 2025-02-18: [(0.15, 'SC', 0), (0.15, 'CD', 4)]
  FANG 2022-05-11: [(0.7, 'CD', 4), (2.35, 'SC', 0)]
  FANG 2022-08-15: [(0.75, 'CD', 4), (2.3, 'SC', 0)]
  FANG 2022-11-16: [(0.75, 'CD', 4), (1.51, 'SC', 0)]
  FANG 2023-03-02: [(2.15, 'SC', 0), (0.8, 'CD', 4)]
  FANG 2023-05-10: [(0.03, 'CD', 4), (0.8, 'CD', 4)]
  FANG 2023-11-15: [(2.53, 'CD', 4), (0.84, 'CD', 4)]
  FANG 2024-03-04: [(2.18, 'SC', 0), (0.9, 'CD', 4)]
  FANG 2024-05-14: [(0.9, 'CD', 4), (1.07, 'CD', 4)]
  FANG 2024-08-15: [(0.9, 'CD', 4), (1.44, 'CD', 4)]
  FCX 2022-04-13: [(0.075, 'CD', 4), (0.075, 'SC', 0)]
  FCX 2022-07-14: [(0.075, 'CD', 4), (0.075, 'SC', 0)]
  HST 2022-12-29: [(0.2, 'SC', 0), (0.12, 'CD', 4)]
  HST 2023-12-28: [(0.25, 'SC', 0), (0.2, 'CD', 4)]
  HST 2024-12-31: [(0.2, 'CD', 4), (0.1, 'CD', 1)]
  HST 2025-12-31: [(0.2, 'CD', 4), (0.15, 'CD', 1)]
  HST 2026-06-30: [(0.72, 'SC', 0), (0.2, 'CD', 4)]
  KIM 2023-12-06: [(0.09, 'SC', 0), (0.24, 'CD', 4)]
  LYB 2022-06-03: [(5.2, 'SC', 0), (1.19, 'CD', 4)]
  PGR 2024-01-18: [(0.75, 'CD', 1), (0.1, 'CD', 4)]
  PGR 2025-01-10: [(0.1, 'CD', 4), (4.5, 'CD', 1)]
  PGR 2026-01-02: [(0.1, 'CD', 4), (13.5, 'CD', 1)]
  PXD 2022-02-25: [(0.78, 'CD', 4), (3.0, 'CD', 4)]
  PXD 2022-05-27: [(6.6, 'CD', 4), (0.78, 'CD', 4)]
  PXD 2022-09-02: [(1.1, 'CD', 4), (7.47, 'CD', 4)]
  PXD 2022-11-29: [(4.61, 'CD', 4), (1.1, 'CD', 4)]
  PXD 2023-03-03: [(4.48, 'CD', 4), (1.1, 'CD', 4)]
  PXD 2023-05-31: [(2.09, 'CD', 4), (1.25, 'CD', 4)]
  PXD 2023-09-05: [(0.59, 'CD', 4), (1.25, 'CD', 4)]
  PXD 2023-11-29: [(1.95, 'CD', 4), (1.25, 'CD', 4)]
  PXD 2024-03-01: [(1.25, 'CD', 4), (1.31, 'CD', 4)]
  TEL 2024-11-22: [(0.65, 'CD', 4), (0.65, 'CD', 4)]  EXACT DUPLICATE
  TPL 2022-06-07: [(3.0, 'CD', 4), (20.0, 'SC', 0)]
  WRB 2022-06-24: [(0.5, 'SC', 0), (0.1, 'CD', 4)]
  WRB 2023-09-22: [(0.11, 'CD', 4), (0.5, 'SC', 0)]
  WRB 2023-12-15: [(0.11, 'CD', 4), (0.5, 'SC', 0)]
  WRB 2024-06-24: [(0.5, 'SC', 0), (0.12, 'CD', 4)]
  WRB 2024-09-23: [(0.08, 'CD', 4), (0.25, 'SC', 0)]
  WRB 2024-12-16: [(0.08, 'CD', 4), (0.5, 'SC', 0)]
  WRB 2025-06-23: [(0.5, 'SC', 0), (0.09, 'CD', 4)]
  WRB 2025-12-15: [(0.09, 'CD', 4), (1.0, 'SC', 0)]
  WRB 2026-06-23: [(0.5, 'CD', 2), (0.1, 'CD', 4)]

Yield over each security's spans: distributions over the mean traded close, a year
  587 securities with half a year of prices: median 1.40%, p75 2.80%, p90 3.95%, max 7.78%; over 4% a year 55, over 6% 5, none paid 127
    Pioneer Natural Resource Co.             7.78%
    Altria Group, Inc.                       7.41%
    LyondellBasell Industries N.V. Class A   7.09%
    Verizon Communications                   6.56%
    Ford Motor Company                       6.13%
    AT&T Inc.                                5.61%
    VICI Properties Inc. Common Stock        5.44%
    Oneok, Inc.                              5.32%
    Dow Inc.                                 5.32%
    Host Hotels & Resorts, Inc.              5.25%
    Walgreens Boots Alliance, Inc            5.24%
    Realty Income Corporation                5.21%
  SPY: 19 distributions, 32.8900 a share, 1.27% a year on its mean close 539.04; types ['CD']
```

### old_split_stale.txt -- the old panel's split staleness as first measured, before the classification (corrected in the body)

```text
splits under the old panel's tickers inside its window: 45; days on which its cap pairs a post-split price with a pre-split count: 5,184; of them member-days: 5,184
  2022-03-24 WRB    ratio   1.5000  stale days  196  member-days  196  count next changes 2023-01-04
  2022-06-06 AMZN   ratio  20.0000  stale days  146  member-days  146  count next changes 2023-01-04
  2022-06-13 DXCM   ratio   4.0000  stale days  141  member-days  141  count next changes 2023-01-04
  2022-06-23 FTNT   ratio   5.0000  stale days  134  member-days  134  count next changes 2023-01-04
  2022-07-18 GOOG   ratio  20.0000  stale days  118  member-days  118  count next changes 2023-01-04
  2022-07-18 GOOGL  ratio  20.0000  stale days  118  member-days  118  count next changes 2023-01-04
  2022-08-25 TSLA   ratio   3.0000  stale days   90  member-days   90  count next changes 2023-01-04
  2022-08-29 NDAQ   ratio   3.0000  stale days   88  member-days   88  count next changes 2023-01-04
  2022-11-04 CPRT   ratio   2.0000  stale days   40  member-days   40  count next changes 2023-01-04
  2022-11-30 TECH   ratio   4.0000  stale days   23  member-days   23  count next changes 2023-01-04
  2023-01-04 GE     ratio   1.2810  stale days    0  member-days    0  count next changes 2023-01-04
  2023-02-08 PCAR   ratio   1.5000  stale days  228  member-days  228  count next changes 2024-01-05
  2023-03-28 MNST   ratio   2.0000  stale days  195  member-days  195  count next changes 2024-01-05
  2023-07-03 LH     ratio   1.1640  stale days  129  member-days  129  count next changes 2024-01-05
  2023-08-22 CPRT   ratio   2.0000  stale days   94  member-days   94  count next changes 2024-01-05
  2023-08-22 SRE    ratio   2.0000  stale days   94  member-days   94  count next changes 2024-01-05
  2024-02-20 COO    ratio   4.0000  stale days  222  member-days  222  count next changes 2025-01-07
  2024-02-26 WMT    ratio   3.0000  stale days  218  member-days  218  count next changes 2025-01-07
  2024-03-28 ODFL   ratio   2.0000  stale days  195  member-days  195  count next changes 2025-01-07
  2024-04-01 MMM    ratio   1.1960  stale days  194  member-days  194  count next changes 2025-01-07
  2024-04-02 GE     ratio   1.2530  stale days  193  member-days  193  count next changes 2025-01-07
  2024-06-10 NVDA   ratio  10.0000  stale days  145  member-days  145  count next changes 2025-01-07
  2024-06-12 APH    ratio   2.0000  stale days  143  member-days  143  count next changes 2025-01-07
  2024-06-26 CMG    ratio  50.0000  stale days  134  member-days  134  count next changes 2025-01-07
  2024-07-11 WRB    ratio   1.5000  stale days  124  member-days  124  count next changes 2025-01-07
  2024-07-15 AVGO   ratio  10.0000  stale days  122  member-days  122  count next changes 2025-01-07
  2024-09-12 CTAS   ratio   4.0000  stale days   80  member-days   80  count next changes 2025-01-07
  2024-10-03 LRCX   ratio  10.0000  stale days   65  member-days   65  count next changes 2025-01-07
  2024-12-04 ANET   ratio   4.0000  stale days   22  member-days   22  count next changes 2025-01-07
  2024-12-13 ETR    ratio   2.0000  stale days   15  member-days   15  count next changes 2025-01-07
  2024-12-20 TSCO   ratio   5.0000  stale days   10  member-days   10  count next changes 2025-01-07
  2025-01-21 LEN    ratio   1.0330  stale days  244  member-days  244  count next changes 2026-01-09
  2025-05-22 FAST   ratio   2.0000  stale days  159  member-days  159  count next changes 2026-01-09
  2025-06-10 ORLY   ratio  15.0000  stale days  147  member-days  147  count next changes 2026-01-09
  2025-06-30 FTV    ratio   1.3270  stale days  134  member-days  134  count next changes 2026-01-09
  2025-10-30 HON    ratio   1.0610  stale days   48  member-days   48  count next changes 2026-01-09
  2025-11-03 DD     ratio   2.3900  stale days   46  member-days   46  count next changes 2026-01-09
  2025-11-17 NFLX   ratio  10.0000  stale days   36  member-days   36  count next changes 2026-01-09
  2025-12-18 NOW    ratio   5.0000  stale days   14  member-days   14  count next changes 2026-01-09
  2026-01-15 AMCR   ratio   0.2000  stale days  201  member-days  201  count next changes 2026-11-03
  2026-04-06 BKNG   ratio  25.0000  stale days  147  member-days  147  count next changes 2026-11-03
  2026-06-12 KLAC   ratio  10.0000  stale days   99  member-days   99  count next changes 2026-11-03
  2026-06-24 DD     ratio   0.3333  stale days   92  member-days   92  count next changes 2026-11-03
  2026-08-11 MNST   ratio   2.0000  stale days   59  member-days   59  count next changes 2026-11-03
  2026-09-03 APH    ratio   2.0000  stale days   42  member-days   42  count next changes 2026-11-03
```

### resolved_report.txt -- item 1: build_sp500_identity.py --resolve --offline

```text
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
      link: BBG008NXC572 before CIK0002005951, to 2024-07-05 -- WestRock Company traded to 2024-07-05, within a halt of Smurfit Westrock plc's first day under SW, 2024-07-08
  FLT for CPAY: 2022-01-03..2024-03-22 (558); 558 now accounted for, in 12 lookups
      confirmed: the endpoint names BBG000GPXKX9, CPAY's security in the table
      takes FLT 2022-01-03..2024-03-22: FleetCor Technologies, Inc. [BBG000GPXKX9]
  PARA for SKYD (and VIAC, PARA's ticker before): 2022-01-03..2025-08-06 (901); 901 now accounted for, in 34 lookups
      confirmed: VIAC names ViacomCBS Inc. Class B Common Stock from 2022-01-03 to 2022-02-16; PARA names Paramount Global Class B Common Stock from 2022-02-17 to 2025-08-07
      takes VIAC 2022-01-03..2022-02-16: ViacomCBS Inc. Class B Common Stock [BBG000C496P7]
      takes PARA 2022-02-17..2025-08-06: Paramount Global Class B Common Stock [BBG000C496P7]
      link: BBG000C496P7 before BBG01VS5NK99, to 2025-08-06 -- ViacomCBS Inc. Class B Common Stock traded to 2025-08-06, within a halt of Skydance Corporation's first day under PSKY, 2025-08-07
  Bunge Limited for BG: 2023-03-15..2023-10-31 (160); 160 now accounted for, in 13 lookups
      confirmed: the endpoint names Bunge Limited
      takes BG 2022-01-03..2023-10-31: Bunge Limited [CIK0001144519]
      link: CIK0001144519 before CIK0001996862, to 2023-10-31 -- Bunge Limited traded to 2023-10-31, within a halt of Bunge Global SA's first day under BG, 2023-11-01
  SMCI itself: 2024-03-18..2024-09-30 (136); 136 now accounted for, in 4 lookups
      confirmed: the endpoint names BBG000MYZDJ3, SMCI's security in the table
      takes SMCI 2022-01-03..2024-09-30: Super Micro Computer, Inc. Common Stock [BBG000MYZDJ3]
  KVUE itself (not among the brief's candidates): 2023-08-25..2026-10-29 (798); 798 now accounted for, in 15 lookups
      confirmed: KVUE names Kenvue Inc. from 2023-05-03 to 2026-10-30
      takes KVUE 2023-05-03..2026-10-30: Kenvue Inc. [BBG01C79X561]

The table
  634 rows: 603 securities, each under one ticker over one span of the window; 518 rows priced by the panel, 106 from events alone, 1 presumed
  identified by the records and the events 476, by the records alone 40, by the events alone 81, by the reference endpoint alone 5

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
  the members' ticker changes in the table: 65; read through: 52; not: 13
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

The 592 member tickers: 593 runs of membership; the security each names identified by records+events 475, events 81, records 31, records+reference 2, reference 2, records+events+reference 1, events+reference 1

The fetch the rebuild needs, for the 599 securities that are members on some date of the window
  629 spans under 619 tickers; 113 of the tickers are not the panel's: ABC ABNB ACGL AMTM ANTM APO APP ARES AXON BE BG BK BLDR BLL BX CASY CEG CEGV CHK CIEN COHR COIN CPT CRH CRHC CRWD CSGP CVNA DASH DAY DDOG DECK DELL EME EQR EQT ERIE EXE FB FBIN FDXF FERG FI FICO FIX FLEX FLT FRCB FSLR GAP GDDY GEHC GEV HONA HOOD HUBB IBKR INVH JBL KDP KKR KVUE LII LITE LULU MBC MMC MOH MRVL NDSN NLOK OKTA ON P PANW PARA PCG PEAK PKI PLTR PODD PSKY PSTG Q RDDT RE S600 SATS SMCI SNDK SOLS SOLV SQ STLD TKO TPL TRGP TTD TWLO UBER VEEV VIAC VICI VLTO VRT VST VYLR WBD WDAY WRK WSM WTW XYZ
  prices: 2 requests a ticker, adjusted and not, over the window -- 226 new, the panel's 506 cached (cut to each security's spans)
  share counts: 11,968 samples inside the securities' own spans (each span's first and last day and every 63 trading days), 9,105 not on disk
  split events, for the share-count rule recorded in midoffice03: one request a ticker, 619
  in all 9,950 requests
  every member-day is in it: none is named

Wrote
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_resolved.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_resolved_members.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_resolved_fetch.csv
  data\external\polygon\sp500_identity_2022-01-03_2026-11-03_resolved.json
```

### gate_analysis.txt -- the gate's report summarised (its identity labels are crude; see the body)

```text
spans 629; identity passed 611, failed 18; requests over the network 622; failed requests 0
  identity failures by kind: {'another company': 4, 'no record': 13, 'same CIK, another FIGI': 1}; member-days on failed spans: 5,593
    another company                        EG     2023-07-10..2026-11-03 member-days   835  CIK0001095073    EG on 2023-07-10 names Everest Group, Ltd. (NAME:Everest Group, Ltd.), not Everest Group, Ltd. (CIK0001095073)
    another company                        BG     2023-11-01..2026-11-03 member-days   754  CIK0001996862    BG on 2023-11-01 names Bunge Global SA (CIK0001144519), not Bunge Global SA (CIK0001996862)
    another company                        COHR   2022-09-08..2026-11-03 member-days   157  BBG000BLW102     COHR on 2022-09-08 names Coherent Corp. Common Stock (BBG000GS6QB4), not Coherent Corp. (BBG000BLW102)
    another company                        CRHC   2022-01-03..2023-09-22 member-days     0  CIK0000849395    CRHC on 2022-01-03 names Cohn Robbins Holdings Corp. (CIK0001818212), not CRH Public Limited Company (CIK0000849395)
    no record                              GEHC   2022-12-15..2026-11-03 member-days   962  BBG01BFR8YV1     GEHC on 2022-12-15: no reference record
    no record                              VLTO   2023-09-27..2026-11-03 member-days   776  BBG019Q32XJ9     VLTO on 2023-09-27: no reference record
    no record                              SOLV   2024-03-26..2026-11-03 member-days   652  BBG018YZH6T3     SOLV on 2024-03-26: no reference record
    no record                              GEV    2024-03-27..2026-11-03 member-days   651  BBG013G17W31     GEV on 2024-03-27: no reference record
    no record                              Q      2025-10-27..2026-11-03 member-days   252  BBG01TRL73S5     Q on 2025-10-27: no reference record
    no record                              SNDK   2025-02-13..2026-11-03 member-days   234  BBG01R388JG1     SNDK on 2025-02-13: no reference record
    no record                              FDXF   2026-05-27..2026-11-03 member-days   109  BBG01ZPB91C4     FDXF on 2026-05-27: no reference record
    no record                              HONA   2026-06-15..2026-11-03 member-days    90  BBG020QYV152     HONA on 2026-06-15: no reference record
    no record                              AMTM   2024-09-24..2026-11-03 member-days    59  BBG01NY7B734     AMTM on 2024-09-24: no reference record
    no record                              SOLS   2025-10-20..2026-11-03 member-days    36  BBG01X5744P5     SOLS on 2025-10-20: no reference record
    no record                              MBC    2022-12-09..2026-11-03 member-days     2  BBG01B9G42Y3     MBC on 2022-12-09: no reference record
    no record                              CEGV   2022-01-19..2022-02-01 member-days     0  BBG014KFRNP7     CEGV on 2022-01-19: no reference record
    no record                              S600   2023-11-20..2026-11-03 member-days     0  BBG0019LSQ49     S600 on 2023-11-20: no reference record
    same CIK, another FIGI                 VYLR   2026-10-01..2026-11-03 member-days    24  BBG024CQR6X9     VYLR on 2026-10-01 names Vylor Inc. (BBG024W5VJC3), not Vylor Inc. (BBG024CQR6X9)

split events: 55 (tolerance 0.1, largest term 100)
  contradiction unchanged         11
  price-only    either            2
  price-only    unchanged         5
  price-only    unchanged  exempt 1
  split         follow            35
  split         follow     exempt 1
```

### pytest_all_2.txt -- the full suite, --ignore=tests/test_market_data.py (last lines)

```text

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 2269 passed, 1 skipped, 19 warnings in 74.33s (0:01:14) ===========
```

### pytest_mid_2.txt -- the midoffice tests

```text
======================= 818 passed, 1 skipped in 25.94s =======================
```

### ruff_final.txt -- ruff over every changed Python file

```text
src\praxis\midoffice\refdata\polygon.py:45:20: F401 [*] `typing.Iterable` imported but unused
src\praxis\midoffice\refdata\polygon.py:45:49: F401 [*] `typing.Sequence` imported but unused
src\praxis\midoffice\refdata\prices.py:28:1: I001 [*] Import block is un-sorted or un-formatted
tests\midoffice\test_prices.py:13:1: I001 [*] Import block is un-sorted or un-formatted
tests\midoffice\test_prices.py:17:8: F401 [*] `urllib.error` imported but unused
tests\midoffice\test_prices.py:23:5: F401 [*] `praxis.midoffice.refdata.polygon.Bar` imported but unused
tests\midoffice\test_prices.py:558:49: F401 [*] `praxis.midoffice.refdata.prices._sample_slots` imported but unused
Found 7 errors.
[*] 7 fixable with the `--fix` option.
```

## git add

This change -- items 1 to 3, the split rule, the distributions fetch and the two docstrings, every file, the brief and this retro included:

```
git add src/praxis/midoffice/refdata/polygon.py src/praxis/midoffice/refdata/identity.py src/praxis/midoffice/refdata/rebuild.py src/praxis/midoffice/refdata/prices.py scripts/build_sp500_identity.py scripts/fetch_sp500_rebuild.py tests/midoffice/test_prices.py tests/midoffice/test_identity.py tests/midoffice/test_rebuild.py claude/handoffs/BRIEF_midoffice06_rebuild.md claude/retros/RETRO_midoffice06_rebuild.md
```
