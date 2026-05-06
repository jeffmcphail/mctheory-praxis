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

Net change: <LINES_INIT_DB> lines updated in
`engines/crypto_data_collector.py` `init_db()`. <ROW_COUNT> rows
preserved (zero-copy 1:1 column mapping minus `id`). Atomic rebuild
transaction wall-clock: <SECONDS>s.

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
  `(asset, trade_id)`). <LINES_INIT_DB> lines.
- Removed the redundant `CREATE INDEX
  idx_trades_asset_tradeid` (covered by the new compound PK).
- Preserved `CREATE INDEX idx_trades_asset_timestamp`.

py_compile clean. Committed as `<CYCLE_26_HASH_STEP_1>`.

### Step 2: User-managed maintenance window

User actions:

1. Disable-ScheduledTask -TaskName 'PraxisTradesCollector'
2. Verified no `cmd_collect_trades_loop` process running
3. Waited ~70s for in-flight writes to flush

### Step 3: Rebuild script

Ran `scripts/migrations/cycle26_trades_schema_rebuild.py`.

Pre-flight checks PASSED:
- trades has the expected pre-rebuild schema (id PK +
  UNIQUE(asset, trade_id))
- No trades_v2 leftover from prior attempts
- Last write was <AGE_SECONDS>s ago (writer disabled, OK)

Rebuild transaction wall-clock breakdown:
- CREATE trades_v2: <T_CREATE>s
- INSERT _v2 SELECT FROM trades (<ROW_COUNT> rows):
  <T_COPY>s (<ROWS_PER_S> rows/s)
- CREATE INDEX idx_trades_asset_timestamp: <T_INDEX>s
- DROP trades + ALTER RENAME _v2 -> trades: <T_RENAME>s
- TOTAL: <T_TOTAL>s

Post-state verification:
- PRAGMA table_info(trades): no `id` column, compound PK on
  `(asset, trade_id)` confirmed.
- Row count: pre <ROW_COUNT> -> post <ROW_COUNT> (unchanged).

### Step 4: Re-enable + verify

User actions:

1. Enable-ScheduledTask -TaskName 'PraxisTradesCollector'
2. Waited ~70s for next 60s fire.

`get_collector_health` post-fire reports trades:
- row_count: <POST_ROW_COUNT> (grew by ~<DELTA> from rebuild
  state, confirming new writes are landing)
- staleness_seconds: <STALENESS> (within 120s threshold)
- is_stale: false

No `__error__` artifacts; other collectors healthy.

### Step 5: Doc updates

- `docs/SCHEMA_NOTES.md`: trades row updated from
  `NEAR-CONFORMING | TBD` to `CONFORMING | 26`. Per-table prose
  describes the new schema and notes the one-shot rebuild
  approach.
- `docs/SCHEMA_MIGRATION_PLAN.md`: row #10 (trades) updated from
  `NEAR-CONFORMING | TBD | dual-write | timestamp already ms`
  to `CONFORMING | 26 | one-shot rebuild | <CYCLE_26_HASH>`.
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
in the program: <T_COPY>s for the bulk INSERT). But active
human time was modest -- the rebuild script ran without
intervention; the user just managed the disable/enable
window around it.

### Lessons for the future

1. **Not every schema migration needs dual-write.** When the
   change is purely structural and the writer doesn't depend
   on the changing field, one-shot rebuild during a
   maintenance window is faster, safer, and uses half the
   storage.
2. **Scheduled-task collectors enable cheap maintenance
   windows.** Long-lived collectors can't be paused this
   way; a future workload that adds a long-lived collector
   should think carefully about whether maintenance windows
   are required vs. dual-write being mandatory.

---

## Open items / next cycle inputs

- **Migration program: COMPLETE.** No further schema
  migrations queued.
- **Cycle 27 (autodetect collapse)**: closed earlier today.
- **Cycle 28 + 29 (collector exit-code hardening)**: closed.
- **`onchain_btc` recovery**: still stale since 2026-04-28.
  Separate TODO; not part of any migration cycle. Likely
  needs scheduled-task registration + API endpoint review.
