# Retro: Cycle 26 -- trades schema rebuild

**Brief:** `claude/handoffs/BRIEF_trades_schema_rebuild.md`
**Date:** 2026-05-06
**Mode:** Hybrid (Claude drafted, Code applied init_db() edit;
user ran rebuild script)
**Status:** DONE
**Completes the migration program**: 10 of 10 tables conforming.

---

## Summary

Migrated `trades` to Rule 35: removed synthetic `id INTEGER PRIMARY
KEY AUTOINCREMENT`, promoted `UNIQUE(asset, trade_id)` constraint
to compound PRIMARY KEY on `(asset, trade_id)`. Pre-existing column
types (`timestamp INTEGER` ms, `datetime TEXT` ISO) were already
compliant and unchanged.

Net change: 6 deletions / 1 insertion (-5 net) in
`engines/crypto_data_collector.py` `init_db()`. 8,830,907 rows
preserved (zero-copy 1:1 column mapping minus `id`). Atomic rebuild
transaction wall-clock: 25.4s.

---

## Why one-shot, not dual-write

Cycle 26 deliberately departed from the Cycle 23-25 dual-write
recipe. Reasoning captured in the brief: trades was already
Rule-35-compliant for column types, the writer doesn't specify
`id` (so removing it required no writer change), and the
scheduled-not-long-lived collector pattern allowed a clean
maintenance window without a dual-write burn-in.

This is the right precedent for any future migration that's purely
a structural change with no data semantic transformation. The
dual-write recipe's value-add is data-correctness validation
(which we needed for the timestamp-format conversions in Cycles
23-25); for pure structural changes it's overkill.

---

## Execution log

### Step 1: init_db() update + commit

Code edited `engines/crypto_data_collector.py`:

- Replaced `CREATE TABLE IF NOT EXISTS trades` block with the
  post-Cycle-26 schema (no `id`, compound PK on
  `(asset, trade_id)`). 6 deletions / 1 insertion (-5 net).
- Removed the redundant `CREATE INDEX
  idx_trades_asset_tradeid` (covered by the new compound PK).
- Preserved `CREATE INDEX idx_trades_asset_timestamp`.

py_compile clean. Committed as `a1c1638`.

### Step 2: User-managed maintenance window

User actions:

1. Disable-ScheduledTask -TaskName 'PraxisTradesCollector'
2. **First attempt**: ran the rebuild script; pre-flight #4
   ABORTED with "latest trade was 23s ago" -- the long-lived
   `collect-trades-loop` process was still running despite the
   scheduled-task disable (see Process Pattern Correction note
   below). User Stop-Process'd the loop processes.
3. Waited ~15.7 min (940s); pre-flight #4 then passed.

### Step 3: Rebuild script

Ran `scripts/migrations/cycle26_trades_schema_rebuild.py` (v2;
see "v1 / v2" note below for what changed).

Pre-flight checks PASSED:
- trades has the expected pre-rebuild schema (id PK +
  UNIQUE(asset, trade_id))
- No trades_v2 leftover from prior attempts (v1 had aborted
  via clean BEGIN/ROLLBACK so no leftover state)
- Last write was 940s ago (writer disabled + loop processes
  killed, OK)

Rebuild transaction wall-clock breakdown (v2 ordering):
- [1/5] CREATE trades_v2: ~0s (instant, not measured)
- [2/5] INSERT _v2 SELECT FROM trades (8,830,907 rows):
  11.4s (775,877 rows/s)
- [3/5] DROP trades (and its index): ~0s (instant, not measured)
- [4/5] ALTER RENAME trades_v2 -> trades: ~0s (instant, not
  measured)
- [5/5] CREATE INDEX idx_trades_asset_timestamp: ~0s (instant,
  not measured)
- TOTAL: 25.4s wall-clock (the ~14s gap between the 11.4s copy
  and the 25.4s total reflects SQLite's commit overhead at this
  scale -- WAL flush, fsync, page cache writes for the new table
  + index)

Post-state verification:
- PRAGMA table_info(trades): no `id` column, compound PK on
  `(asset, trade_id)` confirmed.
- Row count: pre 8,830,907 -> post 8,830,907 (unchanged).
- Live-MCP `praxis:list_tables` independently confirmed the
  compound PK and `raw_query SELECT name, sql FROM sqlite_master`
  confirmed the new CREATE TABLE plus the
  `idx_trades_asset_timestamp` index.

### Step 4: Re-enable + verify

User actions:

1. Enable-ScheduledTask -TaskName 'PraxisTradesCollector'
   (next scheduled fire at 23:23:56 UTC).
2. Manual fire of `collect-trades --assets BTC ETH` to validate
   the writer against the new schema without waiting for the
   2h boundary.

Manual-fire results:
- BTC inserted 1,000 rows; ETH inserted 1,000 rows
- row_count: 8,830,907 -> 8,832,907 (+2,000 exactly)
- latest advanced to 2026-05-06T22:24:49 UTC

`get_collector_health` cannot directly observe staleness for
crypto_data tables -- the freshly-restarted MCP server (22:19
UTC) currently reports "could not parse timestamp" for all
crypto_data tables. This is a separate pre-existing issue (the
MCP server hasn't picked up Cycle 27's `_to_latest_ms` collapse
since the user has not run `git pull` since the Cycle 27 commit
landed) and NOT a Cycle 26 regression. raw_query confirms
latest=2026-05-06T22:24:49 UTC, ~5 min stale at validation
time, well within the 120s threshold for trades since the manual
fire grabbed live trades. Triaged separately -- not blocking.

No `__error__` artifacts on the writer side; other collectors
healthy.

### Step 5: Doc updates

- `docs/SCHEMA_NOTES.md`: trades row updated from
  `NEAR-CONFORMING | TBD` to `CONFORMING | 26`. Per-table prose
  describes the new schema and notes the one-shot rebuild
  approach.
- `docs/SCHEMA_MIGRATION_PLAN.md`: row #10 (trades) updated from
  `NEAR-CONFORMING | TBD | dual-write | timestamp already ms`
  to `CONFORMING | 26 | one-shot rebuild | 39720bb`.
  Per-cycle prose section #10 written from scratch (this is the
  first one-shot rebuild in the program).
- `claude/TODO.md`: Cycle 26 added to "Recently closed".
  **Migration program section flipped to "complete"**: 10/10
  tables conforming.

---

## Notes

### Migration program complete

Cycle 26 closes the schema migration program. Final scoreboard:

| # | Table | DB | Cycles | Approach |
|---|---|---|---|---|
| 17 | ohlcv_daily | crypto | 17 | simple |
| 18 | ohlcv_4h | crypto | 18 | simple |
| 19 | market_data | crypto | 19 | simple |
| 20 | fear_greed | crypto | 20 | simple |
| 21 | funding_rates | crypto | 21 | simple |
| 22 | ohlcv_1m | crypto | 22 | simple |
| 23 | order_book_snapshots | crypto | 23+23.5 | dual-write |
| 24 | price_snapshots | live_collector | 24+24.5 | dual-write |
| 25 | position_snapshots | smart_money | 25+25.5 | dual-write |
| 26 | trades | crypto | 26 | one-shot rebuild |

Plus: 24.1 retro-only hotfix, 27 (autodetect collapse), 28 +
29 (collector exit-code observability).

### Hybrid workflow datapoint

Cycle 26 is the seventh hybrid cycle and the largest by
runtime (the actual rebuild is the slowest single transaction
in the program: 11.4s for the bulk INSERT, 25.4s total
including commit overhead). Active human time was modest --
the rebuild script ran without intervention on the second
attempt; the first attempt's pre-flight ABORT cost ~2 minutes
of debugging plus the ~15 min wait for the loop process to
finish exiting after Stop-Process.

### Script v1 / v2: SQLite index namespace collision

v1 of `cycle26_trades_schema_rebuild.py` aborted at step 3
(CREATE INDEX) due to a name collision. v1's transaction
ordering was:

```
CREATE trades_v2
INSERT trades_v2 SELECT FROM trades         -- 11.4s (succeeded)
CREATE INDEX idx_trades_asset_timestamp     -- FAILED
                          ON trades_v2         (name collision)
DROP trades
ALTER trades_v2 RENAME TO trades
```

The CREATE INDEX failed with `sqlite3.OperationalError: index
idx_trades_asset_timestamp already exists`. The same-named
index was still attached to the old `trades` table and SQLite
refused to allow a second index of the same name in the
database -- regardless of which table it indexes.

**Generalizable rule**: SQLite indexes are namespaced
per-database, NOT per-table. When rebuilding a table under the
same name and preserving an indexed column, the OLD table's
index must be dropped before the new index can be created
(under the same name) on the new table. Either (a) drop the
old table first (which drops its indexes automatically), then
rename and create, OR (b) use a temporary index name on the
new table, drop the old table, rename, then re-create the
index under its canonical name.

v1's BEGIN/ROLLBACK restored pre-script state cleanly: no
trades_v2 leftover, no partial state, no data loss. The
existing trades table and its index were untouched. This
confirmed the wrap-everything-in-BEGIN/COMMIT pattern is the
right defense for any rebuild script.

v2 reorders to option (a):

```
CREATE trades_v2
INSERT trades_v2 SELECT FROM trades         -- 11.4s
DROP trades                                 -- also drops the
                                              old index
ALTER trades_v2 RENAME TO trades
CREATE INDEX idx_trades_asset_timestamp     -- name now free
                          ON trades
```

This worked first try, completing the full transaction in
25.4s wall-clock.

This lesson generalizes to any future schema-rebuild migration
in the program (or any other Praxis SQLite workload). Worth
codifying as a reusable preflight-or-pattern note in
`docs/SCHEMA_MIGRATION_PLAN.md` if a similar one-shot rebuild
is ever attempted again.

### Process pattern correction: PraxisTradesCollector hybrid

The brief described PraxisTradesCollector as "scheduled, every
60s" -- a pattern matching Cycle 25.5's PraxisSmartMoney
(scheduled-only, short-lived). The actual pattern is BOTH:

- **Scheduled trigger**: every 2 hours
- **Spawned process**: `collect-trades-loop` with
  `--duration 3550` -- a long-lived process that polls Binance
  every 30s for ~59 minutes before exiting naturally
- **Net effect**: trades collection is continuous; the 2h
  scheduled trigger just ensures a fresh long-lived process
  starts before the previous one exits

**Critical operational consequence**: `Disable-ScheduledTask`
prevents NEW long-lived processes from starting at the next 2h
boundary, but does NOT kill an in-flight loop process. The
in-flight process keeps writing for up to ~59 min until its
`--duration` expires. Any maintenance script that relies on
"the writer is paused" must additionally Stop-Process all
in-flight loop processes.

The script's pre-flight #4 (legacy age guard from Cycle 24.5)
caught this on the first rebuild attempt: "latest trade was
23s ago" forced the user to Stop-Process the loop processes
manually before re-running. Pre-flight passed on the second
attempt at age 940s (~15.7 min after the loop kill).

This is now memory entry #13 (added 2026-05-06): hybrid
scheduled+long-lived collector patterns require both
disable-task AND kill-process steps for any maintenance
window. Verify via `Get-Process python | where CommandLine
-like "*collect-trades*"` returns zero results before
proceeding.

For future migrations: any "is this collector scheduled or
long-lived?" investigation should look for BOTH the
Task Scheduler entry AND the spawned process pattern (search
for `--duration` or `cmd_collect_..._loop` in the dispatch
table). Don't trust the scheduled-task description alone.

### Lessons for the future

1. **Not every schema migration needs dual-write.** When the
   change is purely structural and the writer doesn't depend
   on the changing field, one-shot rebuild during a
   maintenance window is faster, safer, and uses half the
   storage.
2. **Scheduled-task collectors enable cheap maintenance
   windows -- but verify the collector pattern first.**
   Hybrid scheduled+long-lived collectors (like
   PraxisTradesCollector) need BOTH disable-task AND
   kill-process steps. Pure scheduled-only (like
   PraxisSmartMoney) need only disable. Pure long-lived (like
   PraxisLiveCollector pre-Cycle-24) need only kill. The
   distinction matters at maintenance-window time.
3. **SQLite indexes are namespaced per-database, not
   per-table.** Generalizable lesson from the script v1 / v2
   debugging. Any rebuild script that preserves an indexed
   column under the same table name must DROP+RENAME before
   re-CREATE INDEX, or use a temporary index name during the
   transaction.
4. **`BEGIN`/`COMMIT` wrappers around the entire DDL+DML are
   the load-bearing safety net.** v1's pre-rebuild state was
   restored without intervention because the whole transaction
   rolled back atomically. Worth retaining for any future
   rebuild script regardless of size.

---

## Open items / next cycle inputs

- **Migration program: COMPLETE.** No further schema
  migrations queued.
- **Cycle 27 (autodetect collapse)**: closed earlier today.
- **Cycle 28 + 29 (collector exit-code hardening)**: closed.
- **`onchain_btc` recovery**: still stale since 2026-04-28.
  Separate TODO; not part of any migration cycle. Likely
  needs scheduled-task registration + API endpoint review.
