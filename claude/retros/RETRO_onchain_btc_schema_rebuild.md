# Retro: Cycle 31 -- onchain_btc full Rule 35 conformance

**Brief:** `claude/handoffs/BRIEF_onchain_btc_schema_rebuild.md`
**Date:** 2026-05-07
**Mode:** Hybrid (Claude drafted brief + rebuild script; Code applied
init_db() + writer edits; user ran rebuild script)
**Status:** DONE
**Predecessor:** Cycle 30 (`63993be`) -- onchain scheduled task
registration. Surfaced the gap that this cycle closes.

---

## Summary

Brought `onchain_btc` into full Rule 35 compliance:
- Removed synthetic `id INTEGER PRIMARY KEY AUTOINCREMENT`
- Promoted `UNIQUE(date)` constraint to `PRIMARY KEY (date)`
- Added `timestamp INTEGER NOT NULL` (UTC midnight of `date` in ms;
  matches ohlcv_daily convention)
- Added `datetime TEXT NOT NULL` (ISO 8601 with `+00:00`)
- Preserved `total_btc` legacy column

**Migration program now 11/11 temporal-indexed tables conforming.**
Cycle 30's scoreboard flag of "10/10 with onchain_btc as deferred"
was wrong -- Rule 35 has no exception for daily-grain tables, and
this cycle closes the gap unconditionally.

Net change: 4 deletions / 4 insertions in `init_db()`,
4 deletions / 12 insertions in `collect_onchain_btc` writer;
rebuild script preserved 370 rows in 0.010s.

---

## Why this matters (post-Cycle-30 reframe)

When Cycle 30 closed, I framed onchain_btc's continued non-
conformance as "spirit of Rule 35 met because daily-grain
doesn't need ms timestamps." User pushed back correctly: Rule
35 is a contract, not a guideline. JOINs across tables on
`timestamp INTEGER` are exactly the use case the contract
serves. Even if joins on onchain_btc aren't currently common,
the contract being uniform across all temporal-indexed tables
is itself the value.

This is a generalizable lesson: when the program's own scoring
allows for "almost compliant" exceptions, those exceptions
become latent risk. The migration program should report 11/11
without asterisks, or report the asterisk count honestly. Cycle
31 closes the asterisk.

---

## Convention alignment with ohlcv_daily

Verified before designing the rebuild script:

| Date | ohlcv_daily.timestamp | onchain_btc.timestamp (post) |
|---|---|---|
| 2026-05-07 | 1778112000000 | 1778112000000 (matches) |
| 2026-05-06 | 1778025600000 | 1778025600000 (matches) |

Both tables use 00:00:00 UTC of the date as the canonical ms
instant. Cross-table JOINs `oc.timestamp = od.timestamp` align
perfectly for the same date.

The rebuild script's verification step picks the latest shared
date, computes the expected timestamp from `date`, and confirms
it matches both tables. PASSED. All 10 sampled dates from
2026-04-25 through 2026-05-06 had byte-identical timestamps
between onchain_btc and ohlcv_daily. Both tables compute date
-> ms via UTC midnight, so 2026-05-06 = 1778025600000 in both.
Cross-table JOIN on `timestamp` works exactly as Rule 35 intends.

---

## Execution log

### Step 1: init_db + writer update (Code)

Code edited `engines/crypto_data_collector.py`:

- `init_db()` `onchain_btc` CREATE TABLE: 1 deletion (`id`
  column), 2 additions (`timestamp`, `datetime`), promoted
  `UNIQUE(date)` to `PRIMARY KEY (date)`, moved `total_btc`
  to end of column list. 4 deletions / 4 insertions net.
- `collect_onchain_btc` writer (line ~631): added `date_dt`,
  `ts_ms`, `dt_iso` derivations; expanded the INSERT column
  list and values tuple. 4 deletions / 12 insertions net.

py_compile clean. Committed as `e595eb8`.

### Step 2: User-managed maintenance window

User actions:
1. `Disable-ScheduledTask -TaskName 'PraxisOnchainCollector'`
2. Verified no in-flight `collect-onchain` process (none expected
   since collector finishes in <10s and only fires daily).
3. Ran `python scripts/migrations/cycle31_onchain_btc_schema_
   rebuild.py`.

### Step 3: Rebuild script

Pre-flight checks PASSED:
- onchain_btc has expected pre-rebuild schema (`id` PK +
  UNIQUE(date) + 7 metric columns + `total_btc`).
- No onchain_btc_v2 leftover from prior attempts.
- BEGIN IMMEDIATE acquired the write lock cleanly (no other
  writer in flight).
- All 370 dates parsed cleanly as YYYY-MM-DD.

Rebuild transaction:
- [1/5] CREATE onchain_btc_v2 with new schema: instant
- [2/5] Copy 370 rows with derived timestamp + datetime:
  0.006s
- [3/5] DROP old onchain_btc: instant
- [4/5] RENAME onchain_btc_v2 -> onchain_btc: instant
- [5/5] No additional indexes (PK on `date` covers all access)
- TOTAL: 0.010s wall-clock (fastest single transaction in the
  migration program -- prior record was Cycle 25's 0.273s)

Post-state verification:
- PRAGMA table_info(onchain_btc): no `id` column; PK on
  `date`; presence of `timestamp INTEGER NOT NULL` and
  `datetime TEXT NOT NULL` confirmed.
- Row count: pre 370 -> post 370 (unchanged).
- Cross-table JOIN sanity check: latest onchain_btc row's
  timestamp matches ohlcv_daily's timestamp for the same date.
  Confirmed alignment.

### Step 4: Re-enable + verify writer against new schema

User actions:
1. `Enable-ScheduledTask -TaskName 'PraxisOnchainCollector'`
2. Manual `python -m engines.crypto_data_collector
   collect-onchain --days 7` to validate the writer immediately
   (without waiting for next 00:45 fire).

Result: Manual `collect-onchain --days 7` after re-enable
succeeded. 6 days of on-chain data stored against the new
schema. INSERT OR REPLACE semantics preserved. New rows have
populated `timestamp` + `datetime` columns (verified via
raw_query: 0 NULL ts, 0 NULL dt across all 370 rows). Live-MCP
`get_collector_health` reports all 11 monitored tables across
the 3 databases as `is_stale=false`; first time the migration
program is 11/11 conforming with no exceptions.

### Step 5: Doc trio + retro updates

- `docs/SCHEMA_NOTES.md`: onchain_btc row updated from
  "(id PK, date TEXT, no timestamp)" to "CONFORMING -- Cycle
  31"; per-table prose now describes compound conventions
  with ohlcv_daily.
- `docs/SCHEMA_MIGRATION_PLAN.md`: row #11 added marking
  onchain_btc as DONE | Cycle 31 | one-shot rebuild |
  `4cab1af`. Migration program scoreboard updated to
  **11/11 COMPLETE; no exceptions for daily-grain**, with an
  explicit reframing note acknowledging that Cycle 26's
  "10/10 with deferred onchain_btc" framing was incorrect (the
  program at end-of-Cycle-30 was actually 10/11).
- `claude/TODO.md`: Cycle 31 added to "Recently closed" with
  the post-state numbers and an explicit note correcting
  Cycle 30's "10/10 with deferred onchain_btc" framing.

---

## Notes

### Reframing Cycle 30's scoreboard

Cycle 30's retro celebrated "10/10 tables conforming" while
acknowledging onchain_btc as a "deferred TODO." That framing
was incorrect. Rule 35 doesn't have deferred TODOs -- a table
is either compliant or not. Cycle 31 retroactively reframes:
the migration program at end-of-Cycle-30 was 10/11, with
onchain_btc still non-conforming. Cycle 31 brings it to 11/11.

The scoreboard tables in `docs/SCHEMA_MIGRATION_PLAN.md` and
`claude/TODO.md` should be updated to reflect this correction
(Cycle 31's commit handles it).

### Hybrid workflow datapoint

Cycle 31 is the eighth or ninth hybrid cycle of the program
(after 23.5, 24.5, 25.5, 27, 27.5, 28, 29, 30 in some order).
The shape "Code edits init_db + writer; user runs rebuild
script during a maintenance window" is now a well-trodden
pattern -- Cycle 26 set the precedent and Cycle 31 reuses it
exactly. Active drafting time on Claude's side: short.

### Rule-35-as-contract lesson

Recorded for future programs: a schema standard's value
compounds the more uniformly it's applied. Daily-grain tables
joining hourly/minute-grain tables on `timestamp` is a real
use case. "Almost compliant" creates fragility -- the dev
later writing a JOIN doesn't know which tables don't have the
column. Better to enforce the standard universally even at
small per-table cost.

---

## Open items / next cycle inputs

- **Migration program: 11/11 COMPLETE** -- no further schema
  migrations queued.
- **MCP server restart at 22:19 UTC 2026-05-06 of unknown
  origin**: still unexplained, deferred. Probably benign.
- **`register_all_tasks.ps1`**: should be updated to include
  `register_onchain_task.ps1`. Separate small TODO.
