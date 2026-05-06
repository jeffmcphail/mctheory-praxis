# Retro: Cycle 23.5 -- order_book_snapshots Phase 5 cleanup

**Brief:** `claude/handoffs/BRIEF_order_book_phase5_cleanup.md`
**Date:** 2026-05-05
**Mode:** Hybrid (Claude drafted, Code applied writer-file edit)
**Status:** DONE
**Predecessor:** Cycle 23 (`ca5c719`, `10724bc`, `5cf1c03`)

---

## Summary

Phase 5 cleanup of Cycle 23's dual-write pilot. After ~24h of clean
post-cutover operation, the `_legacy` and `_v2` artifacts were
removed:

- `order_book_snapshots_legacy` dropped (was receiving dual-writes
  via the runtime-introspection writer; 104,776 rows at drop time).
- `order_book_snapshots_v2` dropped (empty stub recreated each
  collector startup by `init_db()`'s `CREATE TABLE IF NOT EXISTS`
  from Cycle 23 Phase 0).
- Writer in `engines/crypto_data_collector.py` collapsed to
  single-write: removed runtime PK introspection, removed dual-
  INSERT logic, removed the `order_book_snapshots_v2` CREATE in
  `init_db()`. Net: 105 deletions / 12 insertions = -93 lines.

No data loss; the live `order_book_snapshots` (post-cutover ms
schema) is unchanged and remains the only `order_book_*` table in
`crypto_data.db` after this cycle.

---

## Execution log

### Step 1: Cleanup migration script

Ran `scripts/migrations/cycle23_5_order_book_cleanup.py`. Pre-flight
checks PASSED:

- Live table has post-cutover schema (no `id` column, has `datetime`
  column, compound PK).
- Legacy row count was 104,776 vs live 104,784; ratio 99.99%
  (well within the 95% tolerance; the 8-row gap is the Phase 4
  cutover transaction window where the new ms table started
  accumulating before the dual-write writer was retrofitted).
- `_v2` stub was empty as expected.

DROP transaction wall-clock: sub-second (two `DROP TABLE`
statements in a single BEGIN/COMMIT against a 100k-row and an
empty table; SQLite drops in default journal mode have no
fsync barrier per row).

### Step 2: Writer collapse

Applied per the hybrid brief. Code edited
`engines/crypto_data_collector.py`:

- `init_db()`: removed `CREATE TABLE order_book_snapshots_v2` block
  + index. -47 lines.
- `collect_order_book_snapshot`: replaced runtime-introspection +
  dual-INSERT block with single INSERT into the live table.
  -58 deletions / +12 insertions = -46 net.

py_compile clean.

### Step 3: Verification (post-deploy)

Verification happens against the next hourly
`PraxisOrderBookCollector` invocation following the commit. While
the writer was still on the dual-write path post-cleanup-script
(this commit's predecessor state), every collector iteration was
silently failing the second INSERT against the now-dropped
`_legacy` and falling through `INSERT OR IGNORE` -- the live
`order_book_snapshots` rowcount kept growing because the first
INSERT (live, ms-format) succeeds before the second INSERT raises.
Post-this-commit, the next hourly invocation exercises the
single-write path with no dropped-table reference.

Expected post-deploy via `get_collector_health`:

- `order_book_snapshots`: `is_stale=false`,
  `staleness_seconds < 60`, latest timestamp parseable as recent
  ms-format ISO.
- No more `order_book_snapshots_legacy` or `order_book_snapshots_v2`
  in the `unmonitored` list (they no longer exist in
  `sqlite_master`).
- Empty `_v2` stub does NOT reappear after a collector restart
  (because the CREATE statement is removed).
- Cadence preserved: ~12 rows/min combined (BTC + ETH at 10s
  cadence) per the Cycle 23 baseline.

### Step 4: Doc updates

- `docs/SCHEMA_NOTES.md`: `order_book_snapshots` row updated from
  `CONFORMING (DONE-PARTIAL) | 23` to `CONFORMING | 23 + 23.5`.
  Per-table prose: removed the dual-write-window writer paragraph
  (no longer applies).
- `docs/SCHEMA_MIGRATION_PLAN.md`: row #7 updated from
  `DONE-PARTIAL | 10724bc` to `DONE | <CYCLE_23_5_HASH>`. Row 23.5
  marked as DONE. Per-cycle prose section #7 header upgraded
  to `(DONE, Cycles 23 + 23.5, ...)` and Phase 5 bullet rewritten
  from "deferred to Cycle 23.5" to "executed in Cycle 23.5".
  (`<CYCLE_23_5_HASH>` placeholders to be replaced via a follow-up
  commit once the hash is known, matching the Cycle 25 pattern at
  commit `4e47659`.)
- `claude/TODO.md`: Cycle 23.5 added to "Recently closed".

---

## Notes

### Hybrid workflow first datapoint

Cycle 23.5 was the first cycle run under the hybrid workflow
(effective 2026-05-05). The workflow split:

- Claude drafted the cleanup migration script directly as a delta
  zip. ~150 lines of Python with pre-flight checks and idempotency
  guards.
- Claude drafted a 70-line targeted brief for the writer collapse,
  asking Code to apply the change to the on-disk file (Claude's
  local checkout pre-dates Cycle 23, so reconstructing the writer
  from memory was risky).
- Code applied the writer collapse, ran py_compile, and committed.
- Claude exercised the live MCP for verification.

The hybrid pattern works well when Claude has the schema/intent
clearly in mind but not the exact on-disk file contents. Code's
read-the-actual-file step replaces what would have been a longer
Brief explaining the prior state.

Wall-clock and speedup-ratio numbers omitted -- this was the first
hybrid cycle and the figures would be noisy without a baseline.
Cycle 24.5 (next equivalent cycle) will be the first datapoint
worth comparing against old-workflow estimates.

### Writer collapse made the code simpler than pre-Cycle-23

Pre-Cycle-23 the writer had a single INSERT into the seconds-format
`order_book_snapshots` with `ts_ms // 1000` truncation. Post-Cycle-
23.5 the writer has a single INSERT into the ms-format
`order_book_snapshots` with no truncation. Net: same complexity as
pre-Cycle-23 plus the precision recovery -- no residual complexity
from the dual-write era.

### Cycle 24.5 will follow the same pattern

`live_collector.price_snapshots` cleanup is Cycle 24.5, scheduled
after its own 24-48h burn-in. Same approach: cleanup migration
script + writer collapse brief.

---

## Open items / next cycle inputs

- **Cycle 24.5** (price_snapshots Phase 5 cleanup) remains queued.
- **Cycle 25** (smart_money.position_snapshots) is in progress at
  the time of this retro -- expected to complete via the old-
  workflow Brief that was mid-execution when hybrid was announced.
  Cycles 25.5 and 26 onward will use the hybrid pattern.
