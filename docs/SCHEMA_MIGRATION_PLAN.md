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
| 18 | ohlcv_daily | simple | DONE | <TBD> |
| 19 | market_data | schema-only | pending | -- |
| 20 | ohlcv_4h | simple | pending | -- |
| 21 | funding_rates | simple | pending | -- |
| 22 | ohlcv_1m | simple | pending | -- |
| 23 | order_book_snapshots | dual-write | pending | -- |
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

### #2 -- ohlcv_daily (DONE, Cycle 18, commit <TBD>)

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

### #3 -- market_data (NEXT CYCLE, schema-only)

- DB: crypto_data.db
- Rows: 0 (empty table)
- Writer: not currently registered
- Reader: none active
- Pattern: schema-only (no data to preserve). Either drop or rebuild.
- Decision points to settle in next-cycle Brief:
  - Should this table even exist? Pre-Praxis-recovery artifact.
  - If kept: add `timestamp INTEGER` (ms UTC) and make
    `(asset, timestamp)` the compound PK. `date` remains as a derived
    column.
  - If dropped: confirm no consumers via grep before DROP.

### #4 -- ohlcv_4h

- DB: crypto_data.db
- Rows: 10,806 (5,403 4-hour bars x 2 assets)
- Writer: `engines/crypto_data_collector.py` `collect_ohlcv_4h()`
- Reader: `engines/lstm_predictor.py` (queries via date)
- Pattern: simple (Binance API supports full re-fetch; daily cadence
  for the scheduled task)
- Schema change: same shape as ohlcv_daily -- compound PK on
  (asset, timestamp), drop id
- `datetime` column already in `+00:00` ISO format; no semantic change

### #5 -- funding_rates

- DB: crypto_data.db
- Rows: ~2,200 (growing 3x/day)
- Writer: `engines/crypto_data_collector.py` `collect_funding_rates()`
- Reader: phase3 model retrain consumes this
- Pattern: simple (Binance API supports full re-fetch; 3-runs-per-day
  cadence means even an hour-long gap is naturally backfilled)
- Schema change: same shape (compound PK on (asset, timestamp), drop id)
- Note: Cycle 14 widened the staleness threshold to 17h. After
  migration, verify the threshold is still appropriate (no change
  expected since the cadence and API contract are unchanged).

### #6 -- ohlcv_1m

- DB: crypto_data.db
- Rows: ~521,000 (~260k bars per asset over ~6 months at minute cadence)
- Writer: `engines/crypto_data_collector.py` `collect_ohlcv_1m()`
- Reader: multiple LSTM/quant strategies
- Pattern: simple (Binance API supports full re-fetch;
  PraxisCrypto1mCollector runs every 6h so a small gap window is
  naturally re-pulled)
- Schema change: compound PK on (asset, timestamp), drop id
- `datetime` text currently naive; rewrite with `+00:00` offset during
  migration
- Performance note: largest table by row count after `trades`. Migration
  script must handle ~520k rows; verify performance (single
  INSERT-SELECT inside a transaction is expected to be sub-second on
  this volume but confirm before commit).

### #7 -- order_book_snapshots (DUAL-WRITE PILOT)

- DB: crypto_data.db
- Rows: ~20,000 (growing ~5/min)
- Writer: `engines/crypto_data_collector.py` `collect_order_book()`,
  scheduled as PraxisOrderBookCollector with 60s cadence
- Reader: convergence detection / spike alerts
- Pattern: **DUAL-WRITE** (Phase 0-5 per Rule 35.6). High-frequency
  writes mean even a 30-second gap matters.
- Why pilot here: smallest of the dual-write tables. Best place to
  debug the dual-write recipe before applying to bigger volumes.
- Schema change: compound PK on (asset, timestamp), drop id, convert
  timestamp seconds -> ms; `datetime` already uses `+00:00` offset

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
