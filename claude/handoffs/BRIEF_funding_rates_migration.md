# BRIEF: Cycle 21 -- funding_rates Migration

**Series:** praxis
**Cycle:** 21
**Mode:** B (Brief -> Code)
**Author:** Chat (praxis_main_current)
**Date:** 2026-05-03
**Predecessor:** Cycle 20 (`ca316e3`, `6510650`) -- ohlcv_4h migration

---

## Process note (read first)

Cycle 20 was executed by Code without a Brief from Chat ("Mode B-lite,
proceeded directly per docs/SCHEMA_MIGRATION_PLAN.md"). The work itself
was correct, and Code caught two real plan-doc errors during execution
(naive datetime claim + reader audit). Even so: **Mode B requires a
Brief from Chat, every cycle, no exceptions.** "Mode B-lite" is not a
defined mode in `claude/WORKFLOW_MODES_PRAXIS.md`.

The dual-Claude split exists because Chat is the failsafe against drift.
Things Chat catches that Code under self-dispatch can miss:

- Plan-doc claims are verified empirically before the Brief encodes
  them (the "datetime already in +00:00 format" claim that turned out
  wrong)
- Reader audits happen against the actual codebase, not the plan-doc's
  notes (`lstm_predictor.py` is in fact NOT a reader of `ohlcv_4h`)
- Cross-cycle dependencies surface (the funding_rates retrain
  coordination flagged in this Brief)

If a future cycle looks "purely mechanical" -- write the Brief anyway
and Chat will deliver it quickly. The Brief is cheap (~10 min from
Chat); the audit is the value.

---

## Context

Cycle 21 migrates `funding_rates` to Rule 35: drop `id`, compound
`PRIMARY KEY (asset, timestamp)`, convert seconds -> ms, rewrite
`datetime` text from naive to ISO `+00:00`. ~2,212 rows expected
(some growth since this Brief was written; collector runs three times
daily). Pattern is simple stop-migrate-start; Binance API supports
full re-fetch if the gap window matters.

This is Cycle 18's recipe applied to a different table. Two
non-trivial differences from prior cycles worth flagging:

1. **`servers/praxis_mcp/tools/funding.py` is a reader that queries
   `timestamp` directly** (not via `datetime` text). Specifically, line
   55-63 builds a `WHERE timestamp >= cutoff` clause where `cutoff` is
   computed as either seconds or ms based on a runtime sample
   (`ms_mode = ts_sample > 1e12`). **This means the existing autodetect
   handles the migration transparently** -- the reader does not need
   code changes. But it does have a stale comment at the top of the
   file ("timestamp INTEGER (seconds)") that becomes wrong post-
   migration. Update the comment in this cycle.

2. **`phase3_models.joblib` retrain coordination.** Multiple engines
   consume `funding_rates` via DataFrame variables
   (`funding_rate_strategy.py`, `regime_engine.py`, `cpo_training.py`,
   `lstm_predictor.py`). All consume preprocessed data, not raw SQL,
   so they're reader-transparent for THIS cycle. **However**: the
   funding-rate model retrain (TODO in `claude/TODO.md`) reads
   `funding_rates` as the source of truth. After migration, the
   timestamp column unit changes. Confirm the retrain pipeline
   doesn't have a hardcoded "seconds" assumption that would break
   silently. **Spot-grep before migrating, document findings in retro.**

---

## Scope

### Task 1: Migrate `funding_rates` schema to Rule 35

**Current schema:**

```sql
CREATE TABLE funding_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,        -- UTC seconds
    datetime TEXT NOT NULL,            -- "YYYY-MM-DD HH:MM:SS" naive
    funding_rate REAL,
    UNIQUE(asset, timestamp)
)
```

**Target schema (Rule 35 conforming):**

```sql
CREATE TABLE funding_rates (
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,        -- UTC milliseconds
    datetime TEXT NOT NULL,            -- "YYYY-MM-DDTHH:MM:SS+00:00" ISO+offset
    funding_rate REAL,
    PRIMARY KEY (asset, timestamp)
)
```

Changes:
- `timestamp` units: seconds -> milliseconds (multiply existing by 1000)
- `datetime` format: naive `"YYYY-MM-DD HH:MM:SS"` -> ISO with offset
  `"YYYY-MM-DDTHH:MM:SS+00:00"`
- Drop `id` AUTOINCREMENT
- `(asset, timestamp)` compound PRIMARY KEY (subsumes UNIQUE)

**Migration script:** `scripts/migrations/cycle21_funding_rates_to_v2.py`

Idempotent recipe (same as Cycles 17/18/19/20):

1. Open `data/crypto_data.db` with explicit transaction control
   (Rule 34: fresh connection).
2. Confirm OLD schema (id PK present, naive datetime). Print
   "Already migrated" + exit cleanly if NEW schema detected.
3. Spot-grep for `funding_rates.id` foreign-key dependencies. Cycle 17
   established that `id` columns are unused; verify for this table
   too. Document in retro.
4. CREATE TABLE `funding_rates_new` with target schema.
5. INSERT INTO `funding_rates_new` SELECT
   - `asset`
   - `timestamp * 1000`
   - `strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch')` --
     re-derive from the timestamp (defense-in-depth, not a copy of old
     `datetime` text), matching Cycle 20's pattern
   - `funding_rate`
   FROM `funding_rates`.
6. Verify row counts match (expect ~2,212; capture exact pre-state
   for the retro).
7. Verify the latest row in `funding_rates_new` parses to the same
   UTC moment as the latest row in old (cross-check).
8. Spot-check 2-3 specific rows from before/after to confirm:
   `funding_rate` value preserved, `timestamp` is exactly old `* 1000`,
   `datetime` is ISO `+00:00`.
9. DROP TABLE `funding_rates`.
10. ALTER TABLE `funding_rates_new` RENAME TO `funding_rates`.
11. Print before/after row counts + latest timestamps for the retro.

Backup `data/crypto_data.db` to `data/crypto_data.db.cycle21_backup`
BEFORE running the migration. md5-verify the backup matches source per
Cycle 18 best practice. Cycle 19 backup can be deleted at the start of
this cycle (per the migration plan retention rule).

### Task 2: Update the `collect_funding_rates` writer

`engines/crypto_data_collector.py` `collect_funding_rates()` at
~line 422. Two changes:

1. **`init_db()` schema** (~line 102): match new shape (no id,
   compound PK).

2. **The INSERT path** (~line 461): update timestamp + datetime
   computation. Current code:
   ```python
   ts = int(rate["fundingTime"] / 1000)        # API returns ms; current code divides
   dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
   ```
   to:
   ```python
   ts = int(rate["fundingTime"])               # API returns ms; store ms directly
   dt = datetime.fromtimestamp(ts // 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
   ```

   Same pattern as Cycle 20. Read the context before editing -- the
   existing code may bind `ts` slightly differently from this snippet.

### Task 3: Update the comment header in `servers/praxis_mcp/tools/funding.py`

Current line 4:
```python
The funding_rates table schema (verified during v0.1 implementation):
    asset TEXT, timestamp INTEGER (seconds), datetime TEXT, funding_rate REAL
```

Change to:
```python
The funding_rates table schema (Rule 35 / Cycle 21 conforming):
    asset TEXT, timestamp INTEGER (UTC milliseconds), datetime TEXT (ISO +00:00),
    funding_rate REAL. Compound primary key on (asset, timestamp).
```

The body of the function has runtime ms/sec autodetection
(`ms_mode = ts_sample > 1e12`), so it works across both the old and
new states without code change. **Verify post-migration** that the
autodetect correctly identifies the migrated table as ms.

### Task 4: Cross-engine spot-grep for `funding_rates` raw-SQL readers

Check whether any engine has hardcoded `WHERE timestamp >
<seconds_value>` clauses against `funding_rates` that would silently
break post-migration. The four engines flagged in this Brief
(`cpo_training.py`, `funding_rate_strategy.py`, `regime_engine.py`,
`lstm_predictor.py`) use `funding_rates` as a Python variable
(DataFrame); but verify with a grep across `engines/` and `scripts/`
for any direct SQL.

**Pattern to grep**: `WHERE timestamp` AND `funding_rates` in the same
function/file. If any hit found that uses raw seconds-since-epoch
constants (not derived from `time.time()` or similar), surface in the
retro and decide before touching schema.

If clean, document in retro: "no raw-SQL WHERE-timestamp clauses
found against funding_rates outside the autodetect-aware MCP tool".

### Task 5: Verify Cycle 14 staleness threshold remains appropriate

Cycle 14 widened `funding_rates` MCP staleness threshold to 17h
(61,200 sec). Rationale: Binance funding events fire at 00:00, 08:00,
16:00 UTC; scheduled task runs at 00:05, 08:05, 16:05 local Toronto
(matching Cycle 13's reactivation). The lag between any given funding
event and the next collector run can legitimately approach 16h.

Post-migration, the threshold value doesn't change but verify
behavior: pull `get_collector_health` after migration and confirm
`funding_rates.is_stale = false`, `staleness_seconds < 61200`. If
post-migration timestamp interpretation goes wrong, the autodetect in
`meta.py._to_latest_ms` will still say it's fresh (because the column
is now ms and the heuristic handles it), but the staleness number
will look bizarre. Catch this at verification.

### Task 6: Update doc trio

**`docs/SCHEMA_NOTES.md`**: per-table prose for `funding_rates`:
NONCONFORMING -> CONFORMING (Cycle 21). Update column type/format
notes. Migration status table: change row to CONFORMING / 21.

**`docs/SCHEMA_MIGRATION_PLAN.md`**:
- Status summary row #5: `funding_rates | simple | DONE | <commit-hash>`
- Per-table spec section: mark DONE; note the `funding.py` runtime-
  autodetect-aware reader meant zero MCP code changes, only a comment
  update.

**`claude/TODO.md`**:
- Mark "Cycle 21" as DONE in Recently closed (with commit hash).
- Add: "Cycle 22: Migrate `ohlcv_1m` per
  `docs/SCHEMA_MIGRATION_PLAN.md`. ~530k rows; largest non-dual-write
  table; verify migration script performance. Same simple
  stop-migrate-start pattern, Binance-backfillable."
- Note in Recently closed: "phase3_models.joblib retrain
  coordination: spot-grep verified no engine has hardcoded
  seconds-since-epoch SQL filters against funding_rates. Retrain
  unblocked from this migration's perspective; the standalone retrain
  TODO remains tracked separately."

### Task 7: Retro

`claude/retros/RETRO_funding_rates_migration.md` with:

- Pre/post row counts and latest UTC delta
- Migration script output (both runs: first + idempotent re-run)
- New schema (PRAGMA table_info)
- Spot-check rows pre/post showing the seconds * 1000 = ms
  conversion + datetime format change
- `funding.py` autodetect verification post-migration
- MCP `get_collector_health` output for `funding_rates` post-migration
- Cross-engine SQL audit results (Task 4)
- Cycle 14 threshold verification (Task 5)
- Any deviations from this Brief

---

## Out of scope

- Migrating any table other than `funding_rates`
- Re-running `phase3_models.joblib` training (separate TODO; this
  cycle just confirms the migration is compatible)
- Touching `_to_latest_ms` autodetect heuristic (Cycle 27)
- Adding new MCP tools or new monitoring entries
- Touching readers beyond the comment update in `funding.py`

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | `data/crypto_data.db.cycle21_backup` created BEFORE migration runs (md5-verified) |
| 2 | `data/crypto_data.db.cycle19_backup` deleted at start of cycle (per retention rule) |
| 3 | `scripts/migrations/cycle21_funding_rates_to_v2.py` exists, idempotent (re-run exits 0 cleanly) |
| 4 | Pre/post migration row counts match exactly (no rows lost) |
| 5 | Pre/post latest UTC moments match (delta = 0s) |
| 6 | New schema: compound PK `(asset, timestamp)`, no `id`, ms timestamps, ISO+00:00 datetime |
| 7 | Spot-check 2-3 rows shows `timestamp = old_seconds * 1000` exactly |
| 8 | `engines/crypto_data_collector.py` `init_db` + `collect_funding_rates` updated |
| 9 | `servers/praxis_mcp/tools/funding.py` comment header updated |
| 10 | Cross-engine SQL spot-grep performed; results documented in retro |
| 11 | MCP `get_collector_health` reports `funding_rates` correctly post-migration (autodetect handles it; is_stale matches expectation) |
| 12 | `funding.py` `get_funding_rate_history()` returns rows correctly post-migration (spot-test: call with `lookback_days=7`, confirm rows returned) |
| 13 | `docs/SCHEMA_NOTES.md` updated (per-table prose + migration status table) |
| 14 | `docs/SCHEMA_MIGRATION_PLAN.md` updated (status row + per-table spec) |
| 15 | `claude/TODO.md` updated (close Cycle 21, add Cycle 22) |
| 16 | All committable files ASCII-only (Rule 20) |
| 17 | Retro at `claude/retros/RETRO_funding_rates_migration.md` |
| 18 | Two-commit pattern OK (main + hash patch) per Cycle 18/20 precedent, OR amend-before-push to keep single commit -- Code's choice, document in retro |

---

## Notes for Code

- Rule 34 throughout: fresh connection per logical pass in any
  diagnostic / migration script.
- Rule 21 N/A this cycle (no .bat/.ps1 changes).
- The `funding.py` runtime autodetect is the right pattern for any
  future MCP tool that reads timestamp columns -- it absorbed the
  Cycle 21 unit change without code changes. Worth keeping in mind
  for the dual-write tables in Cycles 23-26 (the autodetect approach
  also smooths the dual-write transition).
- Spot-grep across `engines/` and `scripts/` for any raw SQL
  `WHERE timestamp` clauses against `funding_rates`. The one MCP tool
  that does this (`funding.py`) is autodetect-safe. Anything else
  found needs to surface BEFORE migration.
- Verify post-migration via at least three checks: (a) PRAGMA
  table_info for new schema, (b) MCP `get_collector_health`
  reporting fresh and within threshold, (c) MCP
  `get_funding_rate_history(asset='BTC', lookback_days=7)` returning
  data (this hits `funding.py`'s autodetect path).
- Cycle 18/20 precedent: insert `<TBD>` for hash in plan-doc edit
  during the main commit, fix in a follow-up commit. Or amend before
  push to keep a single commit -- both work. Document the choice in
  the retro.
- The retro should explicitly answer the cross-engine SQL audit
  question (Task 4). If it finds nothing concerning, say so;
  if it finds something, surface it and stop before migrating data.
