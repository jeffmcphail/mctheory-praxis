# BRIEF: Cycle 17 -- Temporal Standard Rule + fear_greed Pilot Migration + onchain_btc Monitoring + SCHEMA_NOTES.md

**Series:** praxis
**Cycle:** 17
**Mode:** B (Brief -> Code; surgery on rules file + collector + MCP server + new docs)
**Author:** Chat (praxis_main_current)
**Date:** 2026-04-30

---

## Context

Five retros in a row (Cycles 9, 10, 11, 14, 15) have surfaced timestamp/datetime
heterogeneity across Praxis SQLite tables as a recurring source of bugs:

- Some tables store `timestamp` in seconds, others in milliseconds
- Some `datetime` text fields are naive (`"2026-04-30 04:00:00"`),
  others are ISO-with-Z (`"2026-04-30T19:06:08.087Z"`),
  others ISO-with-offset (`"2026-04-30T18:21:59+00:00"`)
- Some tables (`onchain_btc`, `market_data`,
  `smart_money.position_snapshots`) have NO `timestamp` column at all,
  only date/datetime text

Rule 34 (Cycle 15) addressed defensive *reads* against this. Rule 35
(this cycle) addresses the root cause by enforcing standardized
*writes*. After Rule 35 lands and all tables are migrated to conform,
the heterogeneity problem disappears.

Cycle 17 is the **pilot**: rule + first migration (`fear_greed`) +
documentation foundation. Cycle 18 will draft the full migration plan
and start the next table. Cycles 19-27 will work through the remaining
tables one per cycle, with dual-write protocol for high-frequency
actively-written tables.

This Brief is for Cycle 17 only. Do not exceed scope.

---

## Scope

### Task 1: Add Rule 35 to `claude/CLAUDE_CODE_RULES.md`, bump to v1.4

Insert as new Rule 35 in a new "Data Storage Rules" subsection between
"Testing and Diagnostics Rules" (current rules 27-34) and "Retro Rules"
(currently rules 35-41). After insertion, Retro Rules renumber from
35-41 to 36-42. Total rule count: 42.

**Rule 35 text** (verbatim, keep ASCII per Rule 20):

```
35. **Temporal data storage standard.** Every table that contains
    temporally indexed data MUST have:

    1. A `timestamp` column of INTEGER type, storing Unix epoch
       milliseconds in UTC. Always the same units (milliseconds, not
       seconds). Always UTC. The column name is always `timestamp`.
       Date-only data converts to midnight-UTC milliseconds before
       storage.

    2. `timestamp` is part of the table's primary key. Alone for
       single-asset tables, or compound (`(asset, timestamp)`,
       `(slug, timestamp)`, etc.) for multi-keyed tables. Whatever
       uniquely identifies a row, the temporal component IS the
       `timestamp` column.

    3. Optionally, a redundant `datetime` (or `date`) TEXT column if
       read-side speed matters. When present, it's strictly derived
       from `timestamp` and stored in ISO 8601 with explicit UTC offset
       (`+00:00`) for datetime, or `YYYY-MM-DD` (interpreted as UTC
       midnight) for date-only. The `timestamp` column remains
       canonical; `datetime`/`date` is a cache.

    4. For data feeds that return non-UTC timestamps, the collector MUST
       convert to UTC before storage. Common foot-guns: APIs that return
       local time without offset, APIs that return seconds-since-epoch
       but anchor to local-midnight, file-based feeds (CSV) where the
       timezone is documented in the source's API docs but invisible in
       the data itself. When in doubt, audit a known-time sample (e.g.
       fetch a Binance kline that should close at 04:00:00 UTC and
       confirm the stored timestamp matches).

    5. For new tables / new collectors, conformance is mandatory.
       For existing tables, migration is tracked in
       `docs/SCHEMA_MIGRATION_PLAN.md` and executes one table per cycle.

    6. For migrations of actively-written tables, use the dual-write
       pattern:
       - Phase 0: Build new schema as `<table>_v2` and a parallel
         collector path
       - Phase 1: Register a new scheduled task targeting `_v2`; old
         task keeps running
       - Phase 2: Backfill historical rows from old -> `_v2` (timestamp
         converted, UTC text rendered with `+00:00`)
       - Phase 3: Verify the overlap window (rows collected by both
         tasks) is identical at the source-API level, and all readers
         work against `_v2`
       - Phase 4: Stop old task, rename old table to `<table>_legacy`,
         rename `_v2` -> `<table>`, point readers at the renamed table
       - Phase 5: Burn-in observation (24-48h) before deleting `_legacy`

       For batch/daily collectors where the source API allows full
       re-fetch (Binance OHLCV, funding rates, etc.), the simpler
       stop-migrate-start pattern is acceptable as long as the gap
       window can be backfilled from the API after the new collector
       starts. Document which pattern is used for each table in
       `docs/SCHEMA_MIGRATION_PLAN.md`.

    Migration recipe (SQLite, simple pattern): create new table with
    target schema, INSERT-SELECT from old table converting units and
    rendering UTC text, DROP old, RENAME new. Verify all readers
    (engines, MCP tools, analysis scripts) still work before
    committing.
```

**Changelog entry** (append to existing v1.3 line):

```
- v1.4 (2026-04-30): One new rule from Cycle 17: new Rule 35 (temporal
  data storage standard). Establishes the canonical schema for any
  table holding temporally indexed data: INTEGER `timestamp` column in
  ms-since-epoch UTC, part of the primary key, optionally with a
  derived `datetime`/`date` TEXT cache in ISO 8601 with `+00:00`
  offset. Codifies the dual-write migration pattern for high-frequency
  tables. Lands in a new "Data Storage Rules" subsection between
  "Testing and Diagnostics Rules" and "Retro Rules". Renumbered Retro
  Rules from 35-41 to 36-42. Total rule count: 42 (was 41).
```

**Key Principles bullet** (add after Rule 34's bullet):

```
- Temporal data: store as INTEGER ms-since-epoch UTC `timestamp`
  column, primary key. Optional `datetime`/`date` TEXT cache renders
  ISO 8601 with `+00:00` (Rule 35).
```

### Task 1.5: UTC audit script

**Location:** `claude/scratch/audit_timestamp_timezone.py` (gitignored
per `.gitignore` `claude/scratch/` pattern).

**Purpose:** Before migrating any data, confirm the existing timestamps
in every table represent UTC moments (not local time). Empirical
checks done from chat already suggest yes, but a one-shot script
makes the verification durable.

**Behavior:** Open `data/crypto_data.db`, `data/live_collector.db`, and
`data/smart_money.db` (one connection per pass per Rule 34). For
each table that has a timestamp-style column:

- Sample the latest row's timestamp + datetime/date text
- Compute what UTC the numeric timestamp represents
- Compare against the text representation
- Print a one-line verdict per table: `OK` if they match, `MISMATCH`
  with details if not

Output goes to stdout. Execute once and capture output in the retro.
This script is exploratory; not committed long-term.

**Tables to audit:**

```
crypto_data.db:
  ohlcv_1m.timestamp + datetime
  ohlcv_4h.timestamp + datetime
  ohlcv_daily.timestamp + date
  funding_rates.timestamp + datetime
  order_book_snapshots.timestamp + datetime
  trades.timestamp + datetime
  fear_greed.timestamp + date
  onchain_btc.date (only date column; check it parses as a real date)

live_collector.db:
  price_snapshots.timestamp (no datetime column)

smart_money.db:
  position_snapshots.timestamp (TEXT ISO; verify parses as UTC if no offset)
```

If any table reports MISMATCH, STOP and surface to Chat before
proceeding with migrations. The empirical samples already done suggest
all are UTC; this is the safety net.

### Task 2: Migrate `fear_greed` table to Rule 35 standard

**Current schema:**

```sql
CREATE TABLE fear_greed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,        -- UTC seconds (current: e.g. 1777507200)
    date TEXT NOT NULL,                 -- 'YYYY-MM-DD' (already UTC by collector)
    value INTEGER,
    classification TEXT,
    UNIQUE(timestamp)
)
```

**Target schema (Rule 35 conforming):**

```sql
CREATE TABLE fear_greed (
    timestamp INTEGER PRIMARY KEY,     -- UTC milliseconds (e.g. 1777507200000)
    date TEXT NOT NULL,                 -- 'YYYY-MM-DD' (UTC midnight, derived from timestamp)
    value INTEGER,
    classification TEXT
)
```

Changes:
- `timestamp` units: seconds -> milliseconds (multiply existing values by 1000)
- `timestamp` becomes the primary key (drop `id` autoincrement)
- `date` semantics unchanged (still YYYY-MM-DD, still UTC midnight)
- `UNIQUE(timestamp)` constraint subsumed by PK

**Migration script:** `scripts/migrations/cycle17_fear_greed_to_v2.py`
(new directory `scripts/migrations/` -- this becomes the home for all
schema migration scripts going forward).

The script should:

1. Open `data/crypto_data.db` with explicit transaction control.
2. Confirm `fear_greed` table exists with the OLD schema (5 columns,
   id PK, timestamp INTEGER seconds). If not -- if it already looks
   like the new schema -- print "Already migrated" and exit 0.
3. Verify there are no other tables / engines depending on
   `fear_greed.id`. (Spot grep before starting; flag in retro.)
4. CREATE TABLE `fear_greed_new` with the target schema.
5. INSERT INTO `fear_greed_new` SELECT `timestamp * 1000`, `date`,
   `value`, `classification` FROM `fear_greed`.
6. Verify row counts match.
7. Verify the latest row in `fear_greed_new` parses to the same UTC
   moment as the latest row in `fear_greed` (cross-check arithmetic
   and the date string).
8. DROP TABLE `fear_greed`.
9. ALTER TABLE `fear_greed_new` RENAME TO `fear_greed`.
10. Print before/after row counts and latest timestamps for the retro.

The script must be idempotent: running it twice should detect "already
migrated" on the second run and exit cleanly.

**Update the writer:** `engines/crypto_data_collector.py`
`collect_fear_greed()` function (line ~387). Two changes:

- Schema in `init_db()` (line ~91): change to match the new schema.
  Tip: keep the CREATE TABLE IF NOT EXISTS form so that fresh
  databases create the right schema, but fresh installs still need to
  run the migration script if there's any old data -- defensive
  ordering.
- The INSERT statement (line ~403): change `(timestamp, ...)` to
  `(timestamp * 1000, ...)` -- since the source API returns seconds,
  the writer multiplies by 1000 before storing. Or equivalently, parse
  the API response and store ms.

The simplest correct change: in `collect_fear_greed()`, line ~399:
```python
ts = int(d.get("timestamp", 0)) * 1000  # Convert API seconds -> ms
```

And the date derivation (line ~400) keeps the integer-seconds API value
for `datetime.fromtimestamp()`, OR uses the new ms via
`datetime.fromtimestamp(ts // 1000, tz=timezone.utc)`. Either works;
prefer the latter for consistency.

**Verify readers still work:**

The query `SELECT date, value FROM fear_greed` (used by
`engines/lstm_predictor.py:81`) does NOT touch `timestamp` -- it
reads only `date` and `value`. Both are preserved with identical
semantics in the new schema. Reader continues to work without changes.

The MCP health check (`servers/praxis_mcp/tools/meta.py`) reads
`fear_greed.timestamp` to compute staleness. After migration, the
stored value is in MS (was seconds). The current parser handles BOTH
seconds and ms via heuristic detection (a number > 10^11 is ms;
otherwise sec). Confirm this still holds AFTER migration. If it does,
no MCP changes needed. If it doesn't, fix `meta.py` to detect ms
unambiguously for the migrated table.

**Acceptance criterion:** Pre- and post-migration row counts identical
(901 rows expected). Latest row's timestamp post-migration represents
the same UTC moment as latest row pre-migration (within the precision
of seconds). MCP health check still reports `fear_greed` as fresh
post-migration.

### Task 3: Add `onchain_btc` to MCP `get_collector_health` monitoring

**Current state:**
- `onchain_btc` is in the `unmonitored` list in
  `crypto_data` health output.
- It has 364 rows, latest date `2026-04-28`. The data is stale (no
  scheduled collector is running for it). Adding it to monitoring
  will surface this staleness as `is_stale=true` in the health output
  -- which is the correct alarm. A separate cycle will register a
  scheduled collector.

**Required change:** Extend
`servers/praxis_mcp/tools/meta.py` `_collect_db_health` and
`_to_latest_ms` (or whatever helper inspects timestamps; refer to
Cycle 14 retro for the current architecture). Add `onchain_btc` to
the monitored set with:

- Column: `date` (TEXT, format `YYYY-MM-DD`)
- Threshold: 172800 seconds (48 hours; rationale: source API publishes
  daily, so 24h would alarm on any pre-collection state, 48h gives
  one missed collection of slack, 72h would mean two consecutive
  failures which is too late)
- Latest extraction: `SELECT MAX(date) FROM onchain_btc`, parse as
  YYYY-MM-DD, treat as UTC midnight, convert to ms

**Note for SCHEMA_NOTES.md:** `onchain_btc` will likely show
`is_stale=true` immediately after this change because there's no
scheduled collector. That's intentional. SCHEMA_NOTES.md flags this.

### Task 4: SCHEMA_NOTES.md (NEW)

Path: `docs/SCHEMA_NOTES.md`. Praxis project content (not Claude
behavior), so lives in `docs/` per the meta-docs convention from
Cycle 16.

**Structure:**

```markdown
# Schema Notes

> About this file: documentation of every SQLite table across the
> Praxis databases, with column types, semantic meaning, conformance
> status against Rule 35 (temporal data storage standard), and
> read-pattern guidance.

## The standard (Rule 35 summary)

[restate Rule 35 in 4-5 bullets for in-context reference]

## Read patterns (Rule 34 summary)

[restate Rule 34's three acceptable patterns: fresh connections per
pass / isolation_level=None / explicit conn.commit() between SELECTs]

## Database inventory

### crypto_data.db (primary)

#### Table: ohlcv_1m
- Columns: id, asset, timestamp, datetime, open, high, low, close, volume
- timestamp: INTEGER, currently SECONDS (nonconforming)
- datetime: TEXT, naive `"YYYY-MM-DD HH:MM:SS"` (nonconforming -- no offset)
- Writer: engines/crypto_data_collector.py collect-1m
- Scheduled task: PraxisCrypto1mCollector
- Conformance: NONCONFORMING (units, datetime format)
- Migration cycle: TBD (Cycle 26 estimate)

[continue for all 10 crypto_data tables, 4 live_collector tables, 4 smart_money tables]

## Migration status

| Table | Conformance | Cycle |
|---|---|---|
| fear_greed | CONFORMING | 17 |
| (all others) | NONCONFORMING | TBD |

| Table | Conformance | Cycle |
|---|---|---|
| ohlcv_1m | NONCONFORMING | -- |
| ohlcv_4h | NONCONFORMING | -- |
| ohlcv_daily | NONCONFORMING | -- |
| funding_rates | NONCONFORMING | -- |
| order_book_snapshots | NONCONFORMING | -- |
| trades | NEAR-CONFORMING (datetime suffix only) | -- |
| fear_greed | **CONFORMING** | **17** |
| onchain_btc | NONCONFORMING | -- |
| market_data | EMPTY (no rows; defer until populated) | -- |
| live_collector.price_snapshots | NONCONFORMING | -- |
| smart_money.position_snapshots | NONCONFORMING (no timestamp column) | -- |

## Notes on currently-stale tables

- `onchain_btc`: monitored as of Cycle 17. No scheduled collector
  registered. Staleness alarms expected until a collector lands.
  Last collected during recovery; 364 rows over 1 year. Add a
  scheduled task in a future cycle.
- `market_data`: empty table (0 rows). Schema exists; collector logic
  may be in `crypto_data_collector.py` but not currently invoked.
  Investigate before populating; this is a pre-Praxis-recovery artifact.
```

Use the `praxis:list_tables` MCP tool from chat already-collected
schemas if helpful as a reference (Code can re-run `PRAGMA table_info`
on each table to confirm).

### Task 5: Update `claude/TODO.md`

Mark these as DONE in the "Recently closed" section, with cycle hash
to be added by Chat after commit:

- "docs/SCHEMA_NOTES.md documenting timestamp heterogeneity" --> Cycle 17
- "Add onchain_btc to MCP get_collector_health monitoring" --> Cycle 17

Add new entries under "Active TODOs":

- High priority: "Cycle 18: Write `docs/SCHEMA_MIGRATION_PLAN.md` and
  start migrating second table per Rule 35"
- Lower priority: "Register scheduled collector for `onchain_btc`
  table" (currently 2.7 days stale at 48h threshold; alarms will
  start firing after Cycle 17)

Remove from Active TODOs (now superseded by Rule 35 + migration plan):

- "Phase3 model retrain" stays (independent of Rule 35; just the
  funding_rates table migration will need to coordinate with retrain
  scheduling)

---

## Out of scope (do NOT do this cycle)

- Migrating any table other than `fear_greed`
- Writing `docs/SCHEMA_MIGRATION_PLAN.md` (deferred to Cycle 18)
- Registering a scheduled task for `onchain_btc` (deferred)
- Touching `market_data` (empty table; investigation deferred)
- Touching `crypto_predictor.py`'s `fear_greed_index` columns in
  another table (out of scope; they're embedded in a different table,
  not the standalone `fear_greed` table)
- Modifying any reader of `fear_greed` -- the migration preserves
  reader-visible semantics; no reader changes should be needed

---

## Acceptance Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | `claude/CLAUDE_CODE_RULES.md` is v1.4 with new Rule 35 in "Data Storage Rules" subsection | grep version + grep rule |
| 2 | Retro Rules renumbered 35-41 -> 36-42 | grep numbering |
| 3 | Total rule count is 42 in Code section | grep numbering count |
| 4 | UTC audit script run, all tables OK (or MISMATCHes surfaced before migration) | retro includes audit output |
| 5 | `fear_greed` table now has the new schema (timestamp PK, ms units) | PRAGMA table_info |
| 6 | Pre- and post-migration row counts of `fear_greed` are equal (901 expected) | retro |
| 7 | Latest pre/post UTC moments match | retro |
| 8 | `engines/crypto_data_collector.py` writer updated to write ms to fear_greed | grep collect_fear_greed |
| 9 | `lstm_predictor.py:81` query (`SELECT date, value FROM fear_greed`) still returns same data | spot test |
| 10 | MCP `get_collector_health` returns `fear_greed` as fresh post-migration | call get_collector_health |
| 11 | MCP `get_collector_health` returns `onchain_btc` as monitored (regardless of is_stale value) | call get_collector_health |
| 12 | `docs/SCHEMA_NOTES.md` exists, complete, ASCII-only | view file + grep |
| 13 | `claude/TODO.md` updated with Cycle 17 done items + Cycle 18 next item | grep TODO |
| 14 | All files ASCII-only (Rule 20) | grep -P "[^\x00-\x7F]" should be empty |
| 15 | NEW_CHAT_README.md not modified this cycle (out of scope) | git status |
| 16 | Retro at `claude/retros/RETRO_temporal_standard_pilot.md` includes audit output, migration before/after, and reader spot-check results | view retro |

---

## Notes for Code

- Rule 34 applies throughout: any analysis script you write should
  open fresh connections per pass or use `isolation_level=None`. The
  audit script in particular reads multiple DBs and should NOT keep
  one long-lived connection across them.
- The migration touches a populated DB. Back up `data/crypto_data.db`
  to `data/crypto_data.db.cycle17_backup` BEFORE running the migration
  script. Document the backup in the retro.
- The writer change to `engines/crypto_data_collector.py` only takes
  effect on the next scheduled run. The migration script handles the
  one-time conversion of existing data; the writer change handles all
  future runs. Both are needed.
- If the audit script flags any MISMATCH, STOP and surface to Chat
  before continuing. Do not proceed with migration on suspect data.
- Use `claude/scratch/` for the audit script. Use
  `scripts/migrations/` for the migration script (durable; this dir
  becomes the home for all future migrations).
- Use full ASCII per Rule 20 in all files committed to git. The audit
  script (in `claude/scratch/`) is gitignored so emoji/unicode are
  technically allowed there, but please don't.
- Retro must include: (a) the audit script output, (b) before/after
  row counts and timestamps, (c) the reader spot-check result,
  (d) any deviations from this Brief.
