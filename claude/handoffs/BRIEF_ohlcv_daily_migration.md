# BRIEF: Cycle 18 -- SCHEMA_MIGRATION_PLAN.md + ohlcv_daily Migration

**Series:** praxis
**Cycle:** 18
**Mode:** B (Brief -> Code)
**Author:** Chat (praxis_main_current)
**Date:** 2026-04-30
**Predecessor:** Cycle 17 (`a03fff6`) -- Rule 35 + fear_greed pilot

---

## Context

Cycle 17 landed Rule 35 (Temporal data storage standard) and migrated
`fear_greed` as the pilot. Cycle 18 is two-part:

1. **Plan**: Produce `docs/SCHEMA_MIGRATION_PLAN.md` -- the durable
   ordered roadmap for migrating all remaining nonconforming tables to
   Rule 35. This is the master list that subsequent cycles execute
   from.

2. **Execute**: Migrate `ohlcv_daily` as the second table. Chosen for:
   small row count (1,802 rows), single source API (Binance) that
   supports full re-fetch (so simple stop-migrate-start pattern is
   acceptable), already has both `timestamp` and `date` columns
   (semantics-preserving migration; just convert seconds -> ms and
   drop the autoincrement `id`).

The pattern from Cycle 17 stays: backup the DB, write idempotent
migration script, update the writer in `engines/crypto_data_collector.py`,
verify readers still work, audit MCP health afterwards.

---

## Scope

### Task 1: `docs/SCHEMA_MIGRATION_PLAN.md` (NEW)

Path: `docs/SCHEMA_MIGRATION_PLAN.md`. Pure Praxis project content
(not Claude behavior), so lives in `docs/` per the Cycle 16 meta-docs
convention.

Structure:

```markdown
# Schema Migration Plan

> About this file: ordered roadmap for migrating all Praxis SQLite
> tables to Rule 35 (canonical INTEGER ms-since-epoch UTC timestamp,
> primary key, optional ISO-8601 datetime/date cache). Each table
> migrates as its own cycle.
>
> Rule 35 lives in `claude/CLAUDE_CODE_RULES.md`. The fear_greed pilot
> (Cycle 17) established the recipe; subsequent cycles follow it.

## Status summary

| Cycle | Table | Pattern | Status | Commit |
|---|---|---|---|---|
| 17 | fear_greed | simple | DONE | a03fff6 |
| 18 | ohlcv_daily | simple | (this cycle) | -- |
| 19+ | (per below) | (per below) | (pending) | -- |

## Migration order and per-table specs

(Order chosen by complexity-and-risk: small / re-fetchable tables first;
high-frequency / live-written tables last. Dual-write pattern pilot
slotted at #7 to give 5+ simpler migrations of practice before tackling
the hardest tables.)

### #1 -- fear_greed (DONE in Cycle 17)
[brief recap of what landed: schema change, ms conversion, idempotent
migration script template, no reader changes needed.]

### #2 -- ohlcv_daily (THIS CYCLE)
- DB: crypto_data.db
- Rows: 1,802 (901 dates x 2 assets)
- Writer: engines/crypto_data_collector.py collect_ohlcv_daily()
- Reader: engines/lstm_predictor.py:68 (uses `date`, not `timestamp`)
- Pattern: simple (Binance API supports full re-fetch; daily cadence
  means a brief gap window is non-issue)
- Schema change: `id` AUTOINCREMENT PK + `UNIQUE(asset, timestamp)` ->
  `(asset, timestamp)` PRIMARY KEY (compound), no `id`
- Units: timestamp INTEGER seconds -> milliseconds (multiply x1000)
- Date column: stays `YYYY-MM-DD` UTC midnight (no semantic change)

### #3 -- market_data (next cycle)
- DB: crypto_data.db
- Rows: 0 (empty table; just a schema change, no data migration)
- Writer: not currently registered (dormant)
- Reader: none active
- Pattern: simple (no data to preserve)
- Note: investigate whether to keep the table at all or drop it.
  Current schema has `date TEXT` only (no timestamp). If kept, add
  `timestamp INTEGER PRIMARY KEY` (ms UTC) and convert `date` to a
  derived column. Defer the "should we keep this?" decision to cycle.

### #4 -- ohlcv_4h (next-next cycle)
- DB: crypto_data.db
- Rows: 10,806 (5,403 4-hour bars x 2 assets)
- Writer: engines/crypto_data_collector.py collect_ohlcv_4h()
- Reader: known reader is engines/lstm_predictor.py (queries via date)
- Pattern: simple (Binance API supports full re-fetch)
- Schema change: same shape as ohlcv_daily (compound PK on
  asset + timestamp, drop id)

### #5 -- funding_rates
- DB: crypto_data.db
- Rows: 2,194
- Writer: engines/crypto_data_collector.py collect_funding_rates()
- Reader: phase3 model retrain consumes this
- Pattern: simple (Binance API supports full re-fetch; 3-runs-per-day
  cadence)
- Schema change: same shape (compound PK on asset + timestamp)
- Note: Cycle 14 widened the staleness threshold to 17h. Verify the
  threshold remains appropriate after the migration (no change expected).

### #6 -- ohlcv_1m (high volume but not actively-written every second)
- DB: crypto_data.db
- Rows: 521,477 (~260k bars per asset over ~6 months at minute cadence)
- Writer: engines/crypto_data_collector.py collect_ohlcv_1m()
- Reader: multiple LSTM/quant strategies consume this
- Pattern: simple (Binance API supports full re-fetch; the 6-hour
  scheduled cadence means the gap window is naturally small;
  PraxisCrypto1mCollector runs every 6 hours)
- Schema change: compound PK on asset + timestamp; drop id; add datetime
  with `+00:00` offset (currently naive)
- Note: this is the LARGEST table by row count after `trades`. Migration
  script must handle ~520k rows; verify performance.

### #7 -- order_book_snapshots (DUAL-WRITE PILOT)
- DB: crypto_data.db
- Rows: 20,553 (growing ~5/min)
- Writer: engines/crypto_data_collector.py collect_order_book(),
  scheduled as PraxisOrderBookCollector with 60s cadence
- Reader: convergence detection / spike alerts
- Pattern: **DUAL-WRITE** (Phase 0-5 per Rule 35.6). High-frequency
  writes mean even a 30-second gap matters; the simpler pattern
  would create one.
- Why pilot here: this is the smallest dual-write table by far
  (vs. live_collector.price_snapshots at 50k+ and trades at 1.3M+).
  Best place to debug the dual-write recipe before applying to
  bigger volumes.
- Schema change: compound PK on asset + timestamp; drop id; convert
  timestamp seconds -> ms; datetime stays in current format (already
  uses `+00:00` offset)

### #8 -- live_collector.price_snapshots
- DB: live_collector.db (sidecar)
- Rows: 52,600 (growing ~50/min)
- Writer: engines/live_collector.py:319, scheduled as
  PraxisLiveCollector running continuously
- Reader: convergence detection
- Pattern: **DUAL-WRITE** (proven in #7 first)
- Schema change: add datetime column with `+00:00` (currently no
  datetime column at all; just `timestamp` in seconds)
- Note: per the Cycle 16 meta-docs convention, this is the live-
  collector sidecar; migration touches the sidecar DB independently.

### #9 -- smart_money.position_snapshots (no timestamp column today)
- DB: smart_money.db (sidecar)
- Rows: 4,140 (growing ~hourly)
- Writers: engines/smart_money.py:371 AND engines/smart_money.py:703
  (TWO insert sites; both need updating)
- Reader: smart_money_alerts.py
- Pattern: **DUAL-WRITE** (current `timestamp` is TEXT ISO without
  a numeric column, so this is a schema-shape change not just a unit
  change; needs both code paths to write the new INTEGER ms column)
- Schema change: ADD `timestamp INTEGER` column, populate from
  ISO TEXT parse, make it the PK (compound with whatever uniquely
  identifies a position snapshot)
- Note: this is the most invasive migration -- two writer sites,
  TEXT-only timestamp today, schema reshape rather than unit
  conversion.

### #10 -- trades (largest, last)
- DB: crypto_data.db
- Rows: 1,303,281+ (growing ~120/sec via WebSocket)
- Writer: engines/crypto_data_collector.py:802, scheduled as
  PraxisTradesCollector running continuously
- Reader: trade flow analytics
- Pattern: **DUAL-WRITE**, biggest stakes
- Schema change: timestamp ALREADY in ms (the only nearly-conforming
  table today). Just needs PK shape change (compound on
  asset + trade_id + timestamp probably) and datetime cosmetic
  (`Z` suffix -> `+00:00`).
- Note: trade_id is the natural unique key per asset. Verify whether
  the new PK should include trade_id or just asset+timestamp.

## Cross-cutting concerns

### Backups
Each migration cycle creates `data/<db_name>.cycle<N>_backup` before
touching anything. Backups are gitignored. Retain through the next
cycle's burn-in window; delete after the *following* cycle's commit
proves stable.

### Reader coordination
Most readers query by `date` or `datetime` text rather than by
`timestamp` directly. These migrations are typically reader-transparent.
The exception is anywhere code does `WHERE timestamp > ?` with a
specific epoch-seconds value -- those WHERE clauses break after
migration. Audit each table's readers BEFORE migrating; document any
that must change.

### MCP `meta.py` autodetect
The current `_to_latest_ms` autodetect heuristic
(`> 1e12 -> ms; else seconds`) handles mixed states gracefully during
the migration period. After all tables migrate, remove the heuristic
in favor of a strict ms-only path.

### Stale memory entry update
Cycle 16 noted `claude/TODO.md` carries the post-recovery state. As
each table migrates, update the table's row in the status table above
(cycle, status, commit hash). The plan doc is the authoritative log of
migration progress.
```

The plan is ~10 sections + cross-cutting notes; aim for ~250-350 lines.
Goal: any future chat (or you in 2 weeks) can read this and understand
where we are without re-deriving everything.

### Task 2: Migrate `ohlcv_daily` to Rule 35 standard

**Current schema:**

```sql
CREATE TABLE ohlcv_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,        -- UTC seconds
    date TEXT NOT NULL,                 -- 'YYYY-MM-DD'
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,
    UNIQUE(asset, timestamp)
)
```

**Target schema:**

```sql
CREATE TABLE ohlcv_daily (
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,        -- UTC milliseconds
    date TEXT NOT NULL,                 -- 'YYYY-MM-DD' (UTC midnight)
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,
    PRIMARY KEY (asset, timestamp)
)
```

Changes:
- `timestamp` units: seconds -> milliseconds (multiply existing by 1000)
- Drop `id` AUTOINCREMENT
- `(asset, timestamp)` compound PRIMARY KEY (subsumes the old UNIQUE
  constraint)
- `date` semantics unchanged

**Migration script:** `scripts/migrations/cycle18_ohlcv_daily_to_v2.py`

Same idempotent recipe as cycle17_fear_greed_to_v2.py:

1. Open `data/crypto_data.db` with explicit transaction control
   (Rule 34: fresh connection for the migration).
2. Confirm OLD schema (id PK present); print "Already migrated" and
   exit cleanly if NEW schema detected.
3. Verify no foreign-key dependencies on `ohlcv_daily.id`. (Spot grep.
   Cycle 17 confirmed no callers reference fear_greed.id; ohlcv_daily
   should be similar but verify.)
4. CREATE TABLE `ohlcv_daily_new` with target schema.
5. INSERT INTO `ohlcv_daily_new` SELECT
   `asset`, `timestamp * 1000`, `date`, `open`, `high`, `low`, `close`,
   `volume` FROM `ohlcv_daily`.
6. Verify row counts match (expected: 1,802).
7. Verify the latest row in `ohlcv_daily_new` parses to the same UTC
   moment as latest row in `ohlcv_daily` (cross-check).
8. DROP TABLE `ohlcv_daily`.
9. ALTER TABLE `ohlcv_daily_new` RENAME TO `ohlcv_daily`.
10. Print before/after row counts and latest timestamps for the retro.

**Update the writer:** `engines/crypto_data_collector.py`
`collect_ohlcv_daily()` function (line ~217). Two changes:

- `init_db` schema (line ~55): match new schema (no id, compound PK).
- The INSERT at line ~256: don't divide ms by 1000; store
  ms directly. Specifically, change:
  ```python
  ts = int(c[0] / 1000)        # current: c[0] is API ms, dividing to seconds
  date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
  ```
  to:
  ```python
  ts = int(c[0])               # API returns ms; store as ms
  date = datetime.fromtimestamp(ts // 1000, tz=timezone.utc).strftime("%Y-%m-%d")
  ```

**Reader spot-check:** `engines/lstm_predictor.py:68` reads
`SELECT date, open, high, low, close, volume FROM ohlcv_daily WHERE
asset=? ORDER BY date`. Does NOT touch `id` or `timestamp` directly.
Migration preserves both `date` and the OHLCV columns identically.
Reader continues to work without changes. Spot-test by re-running the
query post-migration and confirming row count + values.

**Backup:** Copy `data/crypto_data.db` to
`data/crypto_data.db.cycle18_backup` BEFORE running the migration.
Retain per cross-cutting policy in the plan doc.

### Task 3: Update `claude/TODO.md`

Mark `Cycle 18: Write docs/SCHEMA_MIGRATION_PLAN.md and start migrating
second table` as DONE in "Recently closed" with cycle hash (added by
Chat after commit).

Add new high-priority TODO:

- "Cycle 19: Migrate next table per `docs/SCHEMA_MIGRATION_PLAN.md`."
  (Specific table TBD by Chat at cycle start; suggested: market_data
  (empty, trivial schema-only change) OR ohlcv_4h (next non-trivial).
  Defer specific decision to next-cycle Brief.)

Remove the "Cycle 18" entry (now closed).

### Task 4: MCP autodetect drift watch

After this cycle, two tables have ms timestamps (`fear_greed`,
`ohlcv_daily`); the rest still have seconds. The autodetect heuristic
in `meta.py._to_latest_ms` (`> 1e12 -> ms`) handles both correctly.

**No code change required this cycle** -- just verify post-migration
that `get_collector_health` reports `ohlcv_daily` correctly via the
heuristic. Document in retro.

After ALL tables migrate (estimated end of Cycle 27 or so), a follow-up
cycle should simplify `_to_latest_ms` to drop the heuristic in favor of
strict ms-only handling. NOT this cycle.

---

## Out of scope

- Migrating any table other than `ohlcv_daily`
- Touching readers beyond the spot-check
- Modifying or removing the autodetect heuristic in `meta.py`
- Registering a scheduled collector for `onchain_btc` (separate TODO)
- Touching `market_data` (empty; deferred)
- Any work on the dual-write pattern (Cycle 23+ scope)

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | `docs/SCHEMA_MIGRATION_PLAN.md` exists, ASCII-only |
| 2 | Plan doc lists all 10 tables in priority order with patterns annotated |
| 3 | Plan doc has cross-cutting concerns section (backups, readers, autodetect, plan log) |
| 4 | `data/crypto_data.db.cycle18_backup` created BEFORE migration script runs |
| 5 | `scripts/migrations/cycle18_ohlcv_daily_to_v2.py` exists, idempotent |
| 6 | Pre/post migration row count of `ohlcv_daily` is 1,802 (no rows lost) |
| 7 | Pre/post latest UTC moments match (delta < 1 sec) |
| 8 | New schema is compound PK on (asset, timestamp); no id column |
| 9 | New `ohlcv_daily.timestamp` values are ms (e.g. latest = 1777507200000) |
| 10 | `engines/crypto_data_collector.py` writer + init_db schema updated |
| 11 | `engines/lstm_predictor.py:68` reader returns same data post-migration |
| 12 | MCP `get_collector_health` reports `ohlcv_daily` correctly post-migration (no special config; autodetect handles it) |
| 13 | `claude/TODO.md` updated (close Cycle 18 entry; add Cycle 19) |
| 14 | All committable files ASCII-only (Rule 20) |
| 15 | `data/crypto_data.db.cycle18_backup` is gitignored, NOT committed |
| 16 | Retro at `claude/retros/RETRO_ohlcv_daily_migration.md` includes migration before/after, reader spot-check, MCP verification |

---

## Notes for Code

- Apply Rule 34 throughout: each diagnostic / migration script opens a
  fresh sqlite3 connection per logical pass. The migration script is
  the canonical example.
- The migration script can reuse the structure of
  `scripts/migrations/cycle17_fear_greed_to_v2.py` -- copy it and
  adapt the schema diff. The control flow (detect old/new, backup
  check, idempotent guard) is identical.
- Run `audit_timestamp_timezone.py` (or a quick spot-check) after the
  migration to confirm UTC values still match. Capture output in the
  retro.
- Verification of the lstm_predictor reader: spot-test by running
  the exact `SELECT date, open, high, low, close, volume FROM
  ohlcv_daily WHERE asset='BTC' ORDER BY date` query and confirming
  ~901 rows. Compare first/last rows pre/post migration.
- ASCII per Rule 20 in committable files. Scratch helpers in
  `claude/scratch/` are gitignored but stay clean anyway.
- The retro must include: (a) row count before/after, (b) latest UTC
  moment delta, (c) reader spot-check result, (d) MCP health output
  showing ohlcv_daily monitored correctly, (e) any deviations from
  this Brief.
- Update Cycle 18's row in `docs/SCHEMA_MIGRATION_PLAN.md` status table
  AT END of cycle: change "(this cycle) | --" to "DONE | <commit-hash>"
  before final commit. Chat will provide the hash post-push; insert
  placeholder `<TBD>` in commit, fix in follow-up if needed -- or
  defer the hash insertion to the very last edit.
