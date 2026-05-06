# Cycle 26 -- trades schema rebuild (NOT dual-write)

**Mode:** hybrid (Claude drafts, Code applies one small file edit;
user runs the migration script)
**Process pattern:** PraxisTradesCollector is a SCHEDULED task
(every 60s, NOT long-lived). The cmd_collect_trades_loop function
is a CLI helper for ad-hoc use, not the production path.

## What

Migrate `trades` to Rule 35 fully. The table is **already**
Rule-35-compliant for column types: `timestamp INTEGER NOT NULL`
(ms), `datetime TEXT NOT NULL`. The ONLY non-conforming aspect is
the synthetic `id INTEGER PRIMARY KEY AUTOINCREMENT` column with
the natural key as a `UNIQUE(asset, trade_id)` constraint.

Cycle 26 removes the synthetic `id` and promotes
`(asset, trade_id)` from UNIQUE constraint to compound PRIMARY KEY.

## Why NOT dual-write

This cycle deliberately departs from the dual-write recipe used
in Cycles 23, 24, and 25. Justification:

1. **No data transformation needed.** Rows copy 1:1 from old
   schema to new (just minus the `id` column). No timestamp
   conversion, no datetime regeneration, no null-handling.
2. **Writer requires no change for INSERTs.** The writer's
   `INSERT OR IGNORE INTO trades (asset, trade_id, timestamp,
   datetime, price, amount, quote_amount, is_buyer_maker, side)`
   does NOT specify `id` -- it relies on AUTOINCREMENT. Removing
   the `id` column changes nothing for the INSERT path.
3. **Scheduled-task collector.** PraxisTradesCollector fires
   every 60s and exits. Between fires we have a clean, no-writer
   window we can use for an atomic in-place rebuild.
4. **8.7M rows is large.** A dual-write Phase 2 backfill would
   double storage (~1+ GB on disk during the burn-in window).
   The one-shot rebuild only doubles temporarily during the
   transaction.
5. **The dual-write recipe's value-add was the data-transformation
   correctness validation.** That's not a concern here. The
   recipe's other safety properties (atomic cutover, idempotent
   migration scripts, pre-flight checks) are preserved in this
   cycle's design.

## Strategy

- **Step 1 (writer code update):** Update `init_db()` in
  `engines/crypto_data_collector.py`'s CREATE TABLE for `trades`
  to match the post-rebuild schema (no `id`, compound PK on
  `(asset, trade_id)`). This is purely for fresh-DB
  initializations -- the existing DB goes through the migration
  script. Commit + push.
- **Step 2 (user-managed maintenance window):**
  - Disable PraxisTradesCollector via Task Scheduler.
  - Confirm no `cmd_collect_trades_loop` process is running.
  - Wait ~70s to ensure any in-flight writes have flushed.
- **Step 3 (rebuild):** Run
  `scripts/migrations/cycle26_trades_schema_rebuild.py`. The
  script is fully transactional: BEGIN -> CREATE trades_v2 ->
  INSERT _v2 SELECT FROM trades -> DROP trades -> ALTER RENAME
  _v2 -> trades -> COMMIT. Either succeeds completely or rolls
  back to pre-script state. Pre-flight checks confirm: trades
  has the expected schema, no _v2 leftover from prior attempts,
  the writer is genuinely disabled (last write >60s ago).
- **Step 4 (re-enable):** Re-enable PraxisTradesCollector.
  Verify the next 60s fire writes successfully via
  `get_collector_health`.
- **Step 5 (close-out):** Doc trio + retro updates as standard.

## Specifics for Code

In `engines/crypto_data_collector.py`'s `init_db()`:

1. Find the `CREATE TABLE IF NOT EXISTS trades (...)` block.
2. Replace with the post-Cycle-26 schema:

   ```python
   conn.execute("""
       CREATE TABLE IF NOT EXISTS trades (
           asset TEXT NOT NULL,
           trade_id INTEGER NOT NULL,
           timestamp INTEGER NOT NULL,
           datetime TEXT NOT NULL,
           price REAL NOT NULL,
           amount REAL NOT NULL,
           quote_amount REAL NOT NULL,
           is_buyer_maker INTEGER NOT NULL,
           side TEXT NOT NULL,
           PRIMARY KEY (asset, trade_id)
       )
   """)
   ```

3. Keep the existing `idx_trades_asset_timestamp` index. The
   `idx_trades_asset_tradeid` index becomes redundant with the
   new compound PK (PK already orders by `(asset, trade_id)`),
   so drop that CREATE INDEX statement.
4. py_compile clean.
5. Commit + push using the commit message at the bottom of this
   brief.

The rebuild script (`scripts/migrations/cycle26_trades_schema_
rebuild.py`) is provided in this delta zip and is run by the
user, not by Code. It's fully self-contained.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | `init_db()` CREATE TABLE for `trades` matches the new schema |
| 2 | The `idx_trades_asset_tradeid` CREATE INDEX is removed (redundant with new PK) |
| 3 | `idx_trades_asset_timestamp` CREATE INDEX is preserved |
| 4 | py_compile clean |
| 5 | No changes to writer INSERT path needed (writer already doesn't specify `id`) -- verify by inspection |
| 6 | Rebuild script's pre-flight checks all pass (writer disabled, no _v2 leftover, expected schema present) |
| 7 | Rebuild script completes in <120s wall-clock for 8.7M rows |
| 8 | Post-rebuild PRAGMA table_info(trades) shows no `id`, compound PK on (asset, trade_id) |
| 9 | Post-rebuild row count == pre-rebuild row count |
| 10 | After re-enabling PraxisTradesCollector, the next 60s fire writes successfully (verify via get_collector_health) |
| 11 | Doc trio updated: SCHEMA_NOTES.md, SCHEMA_MIGRATION_PLAN.md, claude/TODO.md |
| 12 | Retro at claude/retros/RETRO_trades_schema_rebuild.md |

## Out of scope

- Adding new columns to trades.
- Changing data values.
- Migrating cmd_collect_trades_loop separately (it shares the
  same writer; the init_db() change covers it).
- A separate Cycle 26.5 -- since this is a one-shot rebuild not
  dual-write, there is no Phase 5 cleanup. The migration is
  complete after Step 4.

## Commit message for Step 1 (writer code update; use verbatim)

```
Cycle 26 step 1: trades init_db() schema update for one-shot rebuild

Updates engines/crypto_data_collector.py init_db()'s CREATE TABLE
for `trades` to the post-Cycle-26 Rule-35 schema: removes synthetic
`id INTEGER PRIMARY KEY AUTOINCREMENT`, promotes the existing
`UNIQUE(asset, trade_id)` constraint to compound PRIMARY KEY.

This is a one-shot rebuild cycle, NOT dual-write -- trades is
already Rule-35-compliant for column types (timestamp INTEGER ms,
datetime TEXT ISO), and no data transformation is needed. The
writer's INSERT statement doesn't specify `id` so removing it
requires no writer change. The actual table rebuild runs in
scripts/migrations/cycle26_trades_schema_rebuild.py during a brief
user-managed maintenance window with PraxisTradesCollector
disabled.

The redundant idx_trades_asset_tradeid index is dropped (the new
compound PK on (asset, trade_id) covers the same access pattern);
idx_trades_asset_timestamp is preserved.
```

## Commit message for Step 5 (close-out; use verbatim if no surprises)

```
Cycle 26 step 2: trades schema rebuild execution + doc trio updates

Rebuilt trades schema via scripts/migrations/cycle26_trades_schema_
rebuild.py during a maintenance window: removed synthetic `id` PK,
promoted (asset, trade_id) UNIQUE constraint to compound PRIMARY
KEY. Atomic transaction: <SECONDS>s for 8.7M rows. PraxisTradesCollector
re-enabled cleanly post-rebuild; next 60s fire wrote successfully.

Doc trio updated: row #10 of SCHEMA_MIGRATION_PLAN.md flipped
NEAR-CONFORMING -> CONFORMING; trades row in SCHEMA_NOTES.md updated
to reflect the new schema; claude/TODO.md Cycle 26 entry moved to
Recently closed. **Migration program complete: 10/10 tables done.**
```
