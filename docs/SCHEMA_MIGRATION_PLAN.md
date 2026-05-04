# Schema Migration Plan

> About this file: ordered roadmap for migrating all Praxis SQLite tables
> to Rule 35 (canonical INTEGER ms-since-epoch UTC `timestamp`, primary
> key, optional ISO-8601 `+00:00` datetime/date cache). Each table
> migrates as its own cycle.
>
> Rule 35 lives in `claude/CLAUDE_CODE_RULES.md`. The `fear_greed` pilot
> (Cycle 17) established the recipe; subsequent cycles follow it.
> `docs/SCHEMA_NOTES.md` carries the per-table conformance audit;
> this file carries the ordering and per-cycle execution log.

---

## Status summary

| Cycle | Table | Pattern | Status | Commit |
|---|---|---|---|---|
| 17 | fear_greed | simple | DONE | a03fff6 |
| 18 | ohlcv_daily | simple | DONE | cc6a178 |
| 19 | market_data | schema-only + collector fix | DONE | 7e73128 |
| 20 | ohlcv_4h | simple | DONE | ca316e3 |
| 21 | funding_rates | simple | DONE | b977cd3 |
| 22 | ohlcv_1m | simple | DONE | 5c1f248 |
| 23 | order_book_snapshots | dual-write | DONE-PARTIAL | 10724bc |
| 23.5 | order_book_snapshots Phase 5 cleanup | code-only | pending | -- |
| 24 | live_collector.price_snapshots | dual-write | pending | -- |
| 25 | smart_money.position_snapshots | dual-write | pending | -- |
| 26 | trades | dual-write | pending | -- |
| 27 | _to_latest_ms cleanup | code-only | pending | -- |

Order rationale: small / re-fetchable / batch-cadence tables migrate
first using the simple stop-migrate-start pattern. The dual-write
pattern lands at #7 (`order_book_snapshots`) -- the smallest of the
high-frequency tables -- to debug the recipe before applying to the
larger live-collector and trades tables. Cycle 27 collapses the
autodetect heuristic in `_to_latest_ms` once every monitored timestamp
column is ms.

---

## Dual-write recipe (Rule 35.6 expanded, Cycle 23 pilot)

The dual-write pattern applies to actively-written tables where a
stop-migrate-start gap would lose data (high-cadence collectors,
WebSocket streams, anything sub-minute). Cycle 23 piloted this on
`order_book_snapshots`; Cycles 24-26 will use the same six-phase
recipe. The Cycle 23 retro is the canonical reference for actual
gotchas; this section is the durable summary.

### The six phases

**Phase 0 -- Build `<table>_v2` and the dual-write writer**

1. CREATE TABLE `<table>_v2` with the target Rule 35 schema alongside
   the existing table. Same database. Idempotent (`IF NOT EXISTS`).
2. Modify the writer to write to BOTH tables in the same transaction:
   - Old table: keep current behavior (preserve any pre-cycle
     reader's expectation; e.g., seconds-truncated timestamp)
   - New table: target Rule 35 representation (ms timestamp, etc.)
3. Both INSERTs in the same `cursor.execute` chain inside one
   `conn.commit()` for atomicity.
4. **Commit Phase 0 standalone** -- separate from Phases 2-4 -- so
   the live collector picks it up at its next process restart.
   For hourly-relaunched collectors, time the commit so the next
   `:00` boundary picks up the fresh code.

**Phase 1 -- Parallel collection burn-in (>=60 minutes)**

1. Wait until enough dual-write rows have accumulated to support
   meaningful verification (>=60 min was the Cycle 23 budget; the
   real constraint is "the collector has fired enough cycles to have
   a clean window of dual-write rows").
2. Verify counts grow at the same rate in both tables over a clean
   60-min window inside the dual-write era.
3. Spot-check 5 random rows: each (asset, datetime) appears in both
   tables with timestamp values consistent (e.g., for sec-vs-ms,
   `legacy_ts * 1000 + sub_second_ms == v2_ts`).

**Phase 2 -- Backfill legacy data into `<table>_v2`**

1. INSERT INTO `<table>_v2` SELECT FROM `<table>` WHERE NOT EXISTS in
   `<table>_v2` (key on `(asset, datetime)` since timestamp differs by
   unit between the two tables).
2. **Use pure-SQL INSERT-SELECT, not Python row-by-row**. Cycle 23
   first-cut Python implementation hung indefinitely on 87k rows due
   to lock contention with the live collector; rewriting as a single
   INSERT-SELECT statement reduced wall-clock to 7.2s.
3. **For seconds-to-ms conversion via SQLite, use ROUND not CAST**:
   ```sql
   CAST(ROUND((julianday(datetime) - 2440587.5) * 86400000) AS INTEGER)
   ```
   The product is a `double` and lands ~1 ULP below the integer for
   ~half of datetimes with .NNN-precision fractional seconds; CAST
   AS INTEGER truncates toward zero, producing off-by-1ms errors.
   Cycle 23 hit this on 43,596 of 87,668 backfilled rows; fix was
   a follow-up UPDATE with ROUND-derived ms.
4. Idempotent: re-running on a fully-backfilled state inserts zero
   rows.
5. **Add an index on `(asset, datetime)` on both tables** before
   running Phase 2 / Phase 3 verification. The NOT EXISTS subquery
   is O(n^2) without it -- Cycle 23 verification hung for 12+ minutes
   without the index, completed in 0.04s with it.

**Phase 3 -- Verification of overlap**

1. `count(<table>_v2) >= count(<table>)`.
2. Every legacy `(asset, datetime)` exists in `<table>_v2` exactly
   once (no missing, no duplicates).
3. Sample 100 random rows: every column except `id` (legacy only) and
   `timestamp` is byte-identical; ts relationship matches the
   migration formula (e.g., `legacy_ts == v2_ts // 1000`).
4. **ABORT if any check fails**. Surface to chat; do not proceed to
   Phase 4 cutover.

**Phase 4 -- Atomic cutover (the dangerous step)**

```sql
BEGIN;
ALTER TABLE <table> RENAME TO <table>_legacy;
ALTER TABLE <table>_v2 RENAME TO <table>;
COMMIT;
```

1. Single transaction, RENAME pair. SQLite executes both DDL
   statements atomically (or rolls back if either fails).
2. Idempotent: detect cut-over state via PRAGMA + sqlite_master.
3. **The writer must adapt at runtime to the new table names**.
   This is the load-bearing gotcha Cycle 23 surfaced: a writer that
   hardcodes `INSERT INTO <table>_v2` will break post-cutover because
   `<table>_v2` no longer exists. Two options:
   - **Runtime introspection** (Cycle 23's choice): the writer
     introspects the live table's PK shape on each iteration and
     adapts -- if `id` column present, it's pre-cutover (write sec
     to live + ms to `_v2`); else post-cutover (write ms to live +
     sec to `_legacy`).
   - **Bundled writer update + cutover commit**: rewrite the writer
     to use post-cutover names AND run cutover in the same commit;
     gives one moment of inconsistency but a simpler writer.
   Cycle 23 retrofitted runtime introspection mid-cycle; future
   cycles should consider the bundled-update approach as cleaner.
4. Verify post-cutover via PRAGMA table_info that the live table has
   the new schema; verify the writer's next iteration succeeds.

**Phase 5 -- Burn-in 24-48h, then drop legacy + collapse writer**

Always deferred to a follow-up cycle (Cycle 23.5 for this pilot).
Runs only after 24-48h of clean post-cutover operation:

1. Modify the writer to single-write (drop the `_legacy` INSERT).
2. DROP TABLE `<table>_legacy`.

Bundling Phase 5 into the main cycle defeats the burn-in safety net.

### Sequencing guidance for future dual-write cycles

| Step | What | When |
|------|------|------|
| 0 | Backup DB; delete previous-cycle backup | Before any change |
| 1 | Phase 0 commit (table + dual-write writer) | Standalone commit |
| 2 | Wait for collector to pick up new code | Next process restart |
| 3 | Phase 1 verification (count match) | After 60+ min dual-write |
| 4 | Add `(asset, datetime)` index on both tables | Before Phase 2 |
| 5 | Phase 2 backfill (pure-SQL, ROUND for ms) | After Phase 1 |
| 6 | Phase 2 idempotent re-run | Verify exit 0 |
| 7 | Phase 3 verification (ABORT on fail) | After Phase 2 |
| 8 | Phase 4 atomic cutover | After Phase 3 |
| 9 | Verify writer still works post-cutover | Within minutes |
| 10 | Update doc trio + retro | After cutover |
| 11 | Phases 2-4 commit (main + hash patch) | Final commit |
| 12 | (24-48h burn-in) | Real-world clock |
| 13 | Phase 5 cleanup cycle | Separate cycle |

---

## Migration order and per-table specs

### #1 -- fear_greed (DONE, Cycle 17, commit a03fff6)

- DB: crypto_data.db
- Rows: 901
- Writer: `engines/crypto_data_collector.py` `collect_fear_greed()`
- Reader: none active in committed code
- Pattern: simple (Alternative.me API supports full re-fetch)
- Schema change: `id` AUTOINCREMENT PK + `UNIQUE(timestamp)` ->
  `timestamp INTEGER PRIMARY KEY`, no `id`
- Units: timestamp INTEGER seconds -> milliseconds
- Recipe: `scripts/migrations/cycle17_fear_greed_to_v2.py` (idempotent,
  pre/post row-count + latest-UTC cross-check, transactional)

### #2 -- ohlcv_daily (DONE, Cycle 18, commit cc6a178)

- DB: crypto_data.db
- Rows: 1,802 (901 BTC + 901 ETH)
- Writer: `engines/crypto_data_collector.py` `collect_ohlcv_daily()`
- Reader: `engines/lstm_predictor.py:68` (uses `date`, not `timestamp`)
- Pattern: simple (Binance API supports full re-fetch; daily cadence)
- Schema change: `id` AUTOINCREMENT + `UNIQUE(asset, timestamp)` ->
  compound `PRIMARY KEY (asset, timestamp)`, no `id`
- Units: timestamp INTEGER seconds -> milliseconds (multiply x1000)
- `date` semantics unchanged
- Recipe: `scripts/migrations/cycle18_ohlcv_daily_to_v2.py`

### #3 -- market_data (DONE, Cycle 19, commit 7e73128)

- DB: crypto_data.db
- Rows: 3 (BTC + ETH + SOL for 2026-05-01; populates forward only)
- Writer: `engines/crypto_data_collector.py` `collect_market_data`
  (rewritten this cycle: now fetches `/global` for BTC dominance and
  populates the previously-unfilled `btc_dominance` column; ms
  timestamp computed from UTC midnight of the collection day)
- Reader: none active yet
- Scheduled task: `PraxisMarketDataCollector` (daily 00:35 local) --
  see "Outstanding admin step" below
- Pattern: schema-only migration (empty table) plus a four-part
  collector overhaul (CLI subcommand, /global call,
  btc_dominance population, scheduled task registration)
- Schema: dropped `id` AUTOINCREMENT; added `timestamp INTEGER NOT NULL`
  (UTC midnight ms); compound `PRIMARY KEY (asset, timestamp)`;
  `date` becomes a derived TEXT cache
- Limitation: CoinGecko's free-tier `/coins/{id}` endpoint returns
  current state only -- no historical backfill. Table populates
  from "today forward" only. Documented in `docs/SCHEMA_NOTES.md`.
- Outstanding admin step: `Register-ScheduledTask` requires
  Administrator privileges, which Code's Claude session does not have.
  Jeff must run `.\services\register_market_data_task.ps1` from an
  elevated PowerShell once. The .bat and .ps1 files are in place;
  the manual first-run backfill via `python -m
  engines.crypto_data_collector collect-market-data --asset all`
  has already seeded today's 3 rows.

### #4 -- ohlcv_4h (DONE, Cycle 20, commit ca316e3)

- DB: crypto_data.db
- Rows: 10,830 (5,415 4-hour bars x 2 assets)
- Writer: `engines/crypto_data_collector.py` `collect_ohlcv_4h()`
- Reader: none external; consumed inside the collector pipeline.
  `lstm_predictor.py` was hypothesized but not actually a reader
  (only `ohlcv_daily` is queried there).
- Pattern: simple (Binance API supports full re-fetch; daily cadence
  for the scheduled task)
- Schema change: same shape as ohlcv_daily -- compound PK on
  (asset, timestamp), drop id; timestamp seconds -> ms
- **Datetime correction**: prior plan note said the column was
  already `+00:00`; verified empirically it was naive
  (`'YYYY-MM-DD HH:MM:SS'`). Migration re-derives datetime from
  `timestamp` via SQLite `strftime('%Y-%m-%dT%H:%M:%S+00:00', ...,
  'unixepoch')` for defense in depth, matching `order_book_snapshots`'s
  format.

### #5 -- funding_rates (DONE, Cycle 21, commit b977cd3)

- DB: crypto_data.db
- Rows: 2,212 at migration time (1,106 BTC + 1,106 ETH; growing 3x/day)
- Writer: `engines/crypto_data_collector.py` `collect_funding_rates()`
- Readers: `servers/praxis_mcp/tools/funding.py`
  `get_funding_rate_history` (autodetect-aware, no logic change --
  comment header refresh only); `engines/lstm_predictor.py:86-90`
  `SELECT DATE(datetime) ... GROUP BY` (SQLite `DATE()` handles both
  naive and ISO formats, reader-transparent); `regime_engine.py`,
  `funding_rate_strategy.py`, `cpo_training.py` consume DataFrames not
  raw SQL, also reader-transparent. phase3 model retrain consumes via
  the same DataFrame path.
- Pattern: simple (Binance API supports full re-fetch; 3-runs-per-day
  cadence means even an hour-long gap is naturally backfilled)
- Schema change: same shape as ohlcv_4h -- compound PK on
  (asset, timestamp), drop `id`; timestamp seconds -> ms; datetime
  re-derived from timestamp via SQLite `strftime('%Y-%m-%dT%H:%M:%S+00:00',
  ..., 'unixepoch')` (was naive `'YYYY-MM-DD HH:MM:SS'`).
- Cycle 14 staleness threshold (17h / 61,200s) verified valid
  post-migration: `get_collector_health` reported `funding_rates`
  staleness ~11h with `is_stale=false` immediately after the migration
  ran. No threshold change required.
- Cross-engine SQL audit (per Brief Task 4): no raw `WHERE timestamp`
  clauses against `funding_rates` with hardcoded seconds-since-epoch
  constants found anywhere in `engines/` or `scripts/`. phase3 retrain
  is unblocked from this migration's perspective.

#### Hotfix (Cycle 21.5)

The post-Cycle-21 writer initially preserved Binance's sub-second
jitter on `fundingTime` (e.g., `1777795200003`), while the migration
produced seconds-aligned ms (`1777795200000`). The compound PK on
`(asset, timestamp)` did not collapse `.000` and `.NNN` for the same
event, so each new funding event accumulated a duplicate row. Over
~4 days, 26 duplicate rows accumulated (13 events x 2 assets BTC + ETH).

Cycle 21.5 fixed this with two surgical changes:

1. **Writer truncation**: `collect_funding_rates` now computes
   `ts = (int(r["timestamp"]) // 1000) * 1000` to drop the sub-second
   tail before storage. Future writes for the same event collapse
   correctly via `INSERT OR REPLACE`.
2. **One-shot dedup**: `scripts/migrations/cycle21_5_funding_rates_dedup.py`
   deleted the 26 jittered rows (`WHERE timestamp % 1000 != 0`),
   verified pre-DELETE that funding-rate values were byte-identical
   within each duplicate group (lossless), wrapped in a transaction.
   Idempotent: re-running on a clean table prints "Already deduped"
   and exits 0.

Cross-table sanity check (run before the dedup): `fear_greed`,
`ohlcv_daily`, `ohlcv_4h`, `market_data` all show 0 `(asset, datetime)`
duplicates. The bug pattern is isolated to `funding_rates` because
Binance's OHLCV `openTime` is bar-aligned by contract while
`fundingTime` carries reporting-clock jitter.

Lesson for future migration cycles: add a "writer alignment audit"
step that asks whether the new writer produces timestamp values
byte-identical (in their key-relevant bits) to the migrated legacy
data. If not, decide between (a) truncating in the writer,
(b) re-rendering the migration to match writer precision, or
(c) accepting duplicates with documentation. For Cycle 21's pattern,
(a) was correct.

### #6 -- ohlcv_1m (DONE, Cycle 22, commit 5c1f248)

- DB: crypto_data.db
- Rows: 530,836 at migration time (BTC: 265,419 + ETH: 265,417;
  asymmetry is a pre-existing 2-row data-quality footnote, ETH starts
  2 mins later than BTC on 2025-10-31)
- Writer: `engines/crypto_data_collector.py` `collect_ohlcv_1m()`
- Readers (exactly 2 raw-SQL readers found via cross-engine audit):
  - `engines/intrabar_predictor.py:96` (`load_intrabar_data`) --
    required a non-cosmetic fix at line 110 (see "Reader fix" below)
  - `servers/praxis_mcp/tools/ohlcv.py:30` (`get_recent_ohlcv`) --
    reader-transparent; docstring update only
- Pattern: simple (Binance API supports full re-fetch;
  PraxisCrypto1mCollector runs every 6h so a small gap window is
  naturally re-pulled)
- Schema change: compound PK on (asset, timestamp), drop id;
  timestamp seconds -> ms (multiply x1000); datetime re-derived
  from `timestamp` via SQLite `strftime('%Y-%m-%dT%H:%M:%S+00:00',
  ..., 'unixepoch')` (was naive `'YYYY-MM-DD HH:MM:SS'`).
- Performance result: 530,836-row INSERT-SELECT completed in
  0.567s wall-clock (Brief had budgeted 5-30s, with 2 minutes as
  the concerning threshold). Total transaction wall-clock 1.013s
  including DROP+RENAME. Single INSERT-SELECT inside a transaction
  is the right approach at this row count.
- **Reader fix (first non-cosmetic reader change in the migration
  program)**: `engines/intrabar_predictor.py` `load_intrabar_data`
  computed bar-bucket ids via integer floor-division of the
  timestamp by `bar_seconds = bar_minutes * 60`. Pre-Cycle-22 the
  timestamp column was UTC seconds and this worked correctly.
  Post-migration the timestamp is UTC ms but `bar_seconds` was
  still in seconds-magnitude, so every 1-min row got a unique
  bucket id and `bar_minutes >= 2` silently returned zero
  aggregated bars. Fixed at line 110:
  `bar_seconds = bar_minutes * 60 * 1000`. The variable name
  `bar_seconds` becomes a slight misnomer post-fix (it's now
  bar-ms) -- left as-is to keep the diff minimal; explanatory
  comment added in-place. Verified empirically post-fix:
  `bar_minutes=5` returns 100 aggregated bars at exact 5-min ms
  boundaries (300,000 ms apart). Cycle 21.5's writer-alignment-
  audit prescription predicted exactly this class of issue would
  surface in subsequent cycles; this is the first time it has,
  and the prescription caught it pre-merge.
- **Writer alignment audit** (per Cycle 21.5 lesson): Binance
  `fetch_ohlcv` returns kline `openTime` values bar-aligned by
  contract (no sub-second jitter). Confirmed empirically post-
  migration: all 530,836 rows have `timestamp % 1000 == 0`. The
  `funding_rates` jitter pattern does NOT apply to kline endpoints.
  Durable result for future cycles: Binance kline endpoints
  (1m / 4h / 1d / etc.) are jitter-free; only event-driven
  endpoints like `fetch_funding_rate_history` carry reporting
  jitter that requires writer-side truncation.

### #7 -- order_book_snapshots (DONE-PARTIAL, Cycle 23, commit 10724bc)

- DB: crypto_data.db
- Rows: 88,894 at cutover (BTC + ETH; growing ~12 rows/min via 10s
  cadence collector)
- Writer: `engines/crypto_data_collector.py` `collect_order_book_snapshot`,
  scheduled as PraxisOrderBookCollector (3550s windowed, hourly
  back-to-back, 10s cadence)
- Readers (exactly 2 raw-SQL readers found via cross-engine audit):
  - `servers/praxis_mcp/tools/order_book.py` `get_order_book_snapshot`
    + `get_order_book_range` (both pre-existing buggy due to unit
    mismatch; silently fixed by migration -- docstrings updated this
    cycle)
  - `servers/praxis_mcp/tools/meta.py` (monitoring config, autodetect-
    aware, reader-transparent)
- Pattern: **DUAL-WRITE** pilot (Rule 35.6 Phase 0-5).
- Why pilot here: smallest of the dual-write tables. Best place to
  debug the dual-write recipe before applying to bigger volumes.
- Schema change: compound PK on (asset, timestamp); drop `id`;
  timestamp seconds -> ms; **`datetime` ALREADY had microsecond
  precision** (verified empirically: `2026-05-04T19:35:51.647000+00:00`).
  The pre-Cycle-23 writer truncated Binance's ms timestamp to seconds
  via `ts_ms // 1000` while the matching `datetime` retained sub-second
  precision. The migration recovers ms by parsing `datetime` (the
  "real" precision) -- not by `legacy_ts * 1000`.
- **Phase 5 (cleanup) deferred to Cycle 23.5**: drop `_legacy`,
  collapse writer to single-write. Runs after 24-48h burn-in confirms
  no errors.
- Performance datapoints:
  - Phase 2 backfill (87,668 rows, pure-SQL INSERT-SELECT with
    julianday-derived ms): 7.219s wall-clock for the INSERT-SELECT;
    a follow-up ROUND-correction UPDATE of 43,596 off-by-1ms rows
    completed in 0.453s
  - Phase 4 atomic RENAME pair: 0.005s wall-clock
- Pre-existing MCP tool bugs silently fixed by migration:
  - `get_order_book_range`'s `WHERE timestamp BETWEEN start_ts_ms
    AND end_ts_ms` returned `total_in_range = 0` for any sane ms
    input pre-Cycle-23 (table stored sec, clients passed ms)
  - `get_order_book_snapshot`'s `ABS(timestamp - at_timestamp_ms)`
    math was unit-mismatched but happened to "return latest row" by
    accident pre-Cycle-23
- Lessons-learned (codified in retro and the new "Dual-write recipe"
  section above): the cutover RENAME pair invalidates the writer's
  hardcoded table names; the writer must adapt at runtime (introspect
  the live table's PK shape) or be updated in the same commit as the
  cutover. We chose the runtime-adaptive approach for this cycle.

### #8 -- live_collector.price_snapshots

- DB: live_collector.db (sidecar)
- Rows: ~52,600 (growing ~50/min)
- Writer: `engines/live_collector.py:319`, scheduled as
  PraxisLiveCollector running continuously
- Reader: convergence detection
- Pattern: **DUAL-WRITE** (proven in #7 first)
- Schema change: convert timestamp seconds -> ms; ADD `datetime` column
  with `+00:00` (currently no datetime column at all)
- Note: this is the live-collector sidecar; migration touches the
  sidecar DB independently per the Cycle 16 meta-docs convention.

### #9 -- smart_money.position_snapshots

- DB: smart_money.db (sidecar)
- Rows: ~4,140 (growing ~hourly)
- Writers: `engines/smart_money.py:371` AND `engines/smart_money.py:703`
  (TWO insert sites; both need updating)
- Reader: `smart_money_alerts.py`
- Pattern: **DUAL-WRITE** (current `timestamp` is TEXT ISO, no numeric
  column; this is a schema-shape change not just a unit change)
- Schema change: ADD `timestamp INTEGER` column, populate from ISO TEXT
  parse, make it part of the compound PK alongside whatever currently
  identifies a position snapshot (likely `(wallet, timestamp, market_slug)`
  per `docs/SCHEMA_NOTES.md`)
- Note: the most invasive migration -- two writer sites, TEXT-only
  timestamp today, schema reshape rather than unit conversion.

### #10 -- trades (largest, last)

- DB: crypto_data.db
- Rows: 1.3M+ (growing ~120/sec via WebSocket)
- Writer: `engines/crypto_data_collector.py:802`, scheduled as
  PraxisTradesCollector running continuously
- Reader: trade flow analytics
- Pattern: **DUAL-WRITE**, biggest stakes
- Schema change: timestamp ALREADY in ms (the only nearly-conforming
  table today). Just needs PK shape change (compound on
  (asset, trade_id) probably -- trade_id is the natural unique key per
  asset) and `datetime` cosmetic (`Z` suffix -> `+00:00`).
- Verify whether the new PK should include `trade_id` (almost certainly
  yes; trade_id is what dedups) or just (asset, timestamp).

---

## Deferred / out-of-scope tables

These tables exist in the Praxis SQLite databases but are not part of
the Rule 35 migration sequence:

- **State tables (Rule 35 N/A):** `live_collector.tracked_markets`,
  `smart_money.tracked_wallets`. Mutable state per row, not
  temporal-row data.
- **Empty tables (defer until populated):** `live_collector.spike_alerts`,
  `smart_money.convergence_signals`, `smart_money.position_changes`.
  Schema exists but no rows yet. Re-evaluate when the first writer
  lands.
- **Empty + TEXT-timestamp:** `live_collector.collection_log`. Empty
  today; if it grows, migrate as part of the live_collector cluster
  near Cycle 24.

See `docs/SCHEMA_NOTES.md` for the full per-table conformance audit.

---

## Cross-cutting concerns

### Backups

Each migration cycle creates `data/<db_name>.cycle<N>_backup` BEFORE
the migration script runs. Backups are gitignored under `data/`.
Retention rule: keep through the *next* cycle's burn-in window;
delete after the cycle-after-next commit proves stable. Concretely,
the Cycle 17 backup can be deleted once Cycle 18's commit has been
in place for 24-48h with no rollback signal.

### Reader coordination

Most readers query by `date` or `datetime` text rather than by
`timestamp` directly. These migrations are typically reader-transparent.
The exception is anywhere code does `WHERE timestamp > ?` with a
specific epoch-seconds value -- those WHERE clauses break after a
seconds -> ms migration. Audit each table's readers BEFORE migrating;
document any that must change in the per-cycle Brief.

For Cycle 18 specifically: the only reader is `lstm_predictor.py:68`
and it queries by `date` only, so no reader changes were needed.

### MCP `_to_latest_ms` autodetect

The current heuristic in `servers/praxis_mcp/tools/meta.py`
(`> 1e12 -> ms; else seconds`) handles mixed states gracefully during
the migration period. Verified post-Cycle-18 that both
`ohlcv_daily` (now ms) and `funding_rates` (still seconds) are reported
correctly by `get_collector_health`.

After all tables migrate (Cycle 26 finish line), Cycle 27 simplifies
`_to_latest_ms` to drop the `auto` heuristic in favor of strict ms-only
handling. Until then, leave the heuristic alone.

### Schema migration script template

Migration scripts live under `scripts/migrations/cycle<N>_<table>_to_v2.py`.
Required structure:

1. **Idempotency guard.** `detect_schema()` returns `'old'`, `'new'`,
   `'missing'`, or `'unknown'`. Re-running on a `'new'` table prints
   "Already migrated" and exits 0.
2. **Pre-migration snapshot.** Row count, ts_min, ts_max, per-asset
   row counts (where applicable), latest row.
3. **Transactional INSERT-SELECT.** `BEGIN` / `COMMIT` / `ROLLBACK`
   wrapping the entire DDL + DML. Apply Rule 34 (fresh connection per
   migration script).
4. **Verification block.** Row count match (overall and per-asset),
   latest UTC moment delta < 1 sec, OHLCV value preservation on the
   latest row, derived `date` text matches stored.
5. **DROP + RENAME.** Drop the old table, rename `<table>_new` -> `<table>`.
6. **Post-migration sanity.** Re-call `detect_schema` to confirm `'new'`,
   re-fetch latest row, log everything for the retro.

`scripts/migrations/cycle17_fear_greed_to_v2.py` and
`scripts/migrations/cycle18_ohlcv_daily_to_v2.py` are the canonical
examples. Subsequent cycles copy the structure and adapt the schema
diff.

### Plan-doc maintenance

Each migration cycle updates this file's Status summary table at the
end of the cycle: change `(this cycle) | --` to `DONE | <commit-hash>`.
If the commit hash is not yet known at edit time, write `<TBD>` and
fix in a follow-up commit (or defer the hash insertion to the very
last edit before commit). Always close the migration row before
opening the next cycle.

---

## Notes for future cycles

- The simple pattern (stop-migrate-start) is appropriate when the source
  API allows full re-fetch and the writer cadence is hourly-or-slower.
  Anything sub-minute or anything WebSocket-driven needs dual-write.
- The dual-write pattern (Rule 35.6 phases 0-5) has not yet been
  exercised. Cycle 23 will pilot it on `order_book_snapshots`. Expect
  to write the dual-write recipe up as a section in this doc once the
  pilot is complete.
- Reader audits are the silent risk: every cycle MUST grep for the
  table name across the entire repo, not just the listed reader, and
  flag any `WHERE timestamp` clauses that use raw epoch values.
- `git log -- engines/crypto_data_collector.py` after each cycle is the
  source of truth for which collectors have moved to the new schema;
  this plan doc reflects the same state but can lag.
