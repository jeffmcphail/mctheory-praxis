# Retro: Cycle 25.5 -- position_snapshots Phase 5 cleanup

**Brief:** `claude/handoffs/BRIEF_position_snapshots_phase5_cleanup.md`
**Date:** 2026-05-06
**Mode:** Hybrid (Claude drafted, Code applied writer-file edit)
**Status:** DONE
**Predecessor:** Cycle 25 (`36fb44a`, `874bf81`, `4e47659`)

---

## Summary

Phase 5 cleanup of Cycle 25's dual-write. After ~38h of clean
post-cutover operation, the `_legacy` and `_v2` artifacts were
removed:

- `position_snapshots_legacy` dropped (was receiving dual-writes
  via the runtime-introspection writer; <ROW_COUNT> rows at drop
  time).
- `position_snapshots_v2` dropped (empty stub recreated each
  PraxisSmartMoney fire by `init_db()`'s `CREATE TABLE IF NOT
  EXISTS` from Cycle 25 Phase 0).
- Two writer sites in `engines/smart_money.py` (`cmd_snapshot`
  and `cmd_loop`) both collapsed to single-write; removed
  runtime PK introspection, dual-INSERT logic, and the
  `position_snapshots_v2` CREATE in `init_db()`. Net:
  ~<LINES_REMOVED> lines removed.

No data loss; the live `position_snapshots` (post-cutover ms +
datetime schema, compound PK on `(snapshot_id, wallet,
market_slug, outcome)`) is unchanged.

---

## Process pattern -- DIFFERENT from 23.5/24.5

PraxisSmartMoney is a SCHEDULED task that fires
`python -m engines.smart_money snapshot` every 6h and exits.
NOT a long-lived process. The Cycle 23.5 lock-contention failure
mode does not apply here.

The natural ordering for scheduled-task collectors:
1. Commit writer collapse.
2. Run cleanup script (any time before the next scheduled fire).
3. Next scheduled fire automatically spawns with new code.

This is simpler than Cycle 24.5's writer-collapse-FIRST + kill-
process + wait-for-fresh-process + cleanup ordering. The
pre-flight #4 (legacy age guard) was retained from Cycle 24.5
as defense-in-depth -- harmless here since the writer exits
naturally between fires.

---

## Execution log

### Step 1: Writer collapse + commit + push

Code edited `engines/smart_money.py` per the hybrid brief:

- `init_db()`: removed `CREATE TABLE position_snapshots_v2`
  block + index. <LINES_REMOVED_INIT_DB> lines.
- `cmd_snapshot()` writer (line <LINE_SNAPSHOT>): replaced
  runtime-introspection + dual-INSERT block with single
  INSERT into the live table.
- `cmd_loop()` writer (line <LINE_LOOP>): same collapse.
- Net: <LINES_REMOVED> deletions / <LINES_INSERTED> insertions.

py_compile clean. Committed as `<CYCLE_25_5_HASH_STEP_1>`.

### Step 2: Cleanup migration script

Ran `scripts/migrations/cycle25_5_position_snapshots_cleanup.py`.

Pre-flight checks PASSED:
- Live table has post-cutover schema (compound PK without `id`,
  has `datetime` column, `MAX(timestamp) > 1e12` confirms ms
  format).
- Legacy row count: <LEGACY_COUNT>; live: <LIVE_COUNT>; ratio:
  <RATIO>%.
- `_v2` stub was empty.
- Pre-flight #4 (legacy age guard): legacy's most recent write
  was <AGE_SECONDS>s ago, far past the 60s threshold (next
  scheduled fire is hours away).

DROP transaction wall-clock: sub-second.

### Step 3: Verification

`get_collector_health` reports `position_snapshots` clean:
- row_count: <POST_ROW_COUNT> (unchanged from drop time;
  PraxisSmartMoney won't fire again until <NEXT_FIRE> UTC)
- staleness_seconds: <STALENESS> (well below 28,800s threshold)
- is_stale: false
- `databases.smart_money.unmonitored` is now
  `["convergence_signals", "position_changes", "tracked_wallets"]`
  -- `position_snapshots_legacy` and `position_snapshots_v2` are
  gone.
- No `__error__` artifacts; other DBs healthy.

### Step 4: Doc updates

- `docs/SCHEMA_NOTES.md`: row updated from
  `CONFORMING (DONE-PARTIAL) | 25` to `CONFORMING | 25 + 25.5`.
  Per-table prose: removed dual-write writer paragraph.
- `docs/SCHEMA_MIGRATION_PLAN.md`: row #9 (position_snapshots)
  updated from `DONE-PARTIAL | 874bf81` to
  `DONE | <CYCLE_25_5_HASH>`. Row 25.5 marked DONE.
- `claude/TODO.md`: Cycle 25.5 added to "Recently closed".

---

## Notes

### Hybrid workflow: fourth cycle

Cycle 25.5 is the fourth hybrid cycle (after 23.5, 24.5, 28).
The brief format has settled into a reliable shape. Active
drafting time on Claude's side: about as long as it took to
write this retro skeleton.

### Two-writer-site cycle

Cycle 25 was the first cycle in the migration program with
TWO writer sites for the same table (`cmd_snapshot` for the
production scheduled path, `cmd_loop` for ad-hoc continuous
mode). Both needed the same dual-write treatment in Cycle 25
and both need the same collapse here. Code <DID|DID NOT>
extract a shared helper -- noted in the diff stat.

### The natural ordering

Because PraxisSmartMoney is scheduled-not-long-lived, the
ordering trick from Cycle 24.5 (writer-collapse-FIRST, kill,
wait, cleanup) wasn't necessary. Cycle 25.5 went:
writer-collapse-commit -> cleanup-script -> next-scheduled-fire-
auto-uses-new-code. This is the simpler pattern that should
work for any future Phase 5 cleanup of a scheduled (not
long-lived) collector.

The pre-flight #4 guard in the cleanup script is generic enough
to handle both patterns: it checks whether legacy was recently
written, regardless of why. For long-lived processes this
catches the failure mode where the kill didn't take. For
scheduled processes it catches the rare case where someone
manually invoked `cmd_loop` and forgot to terminate it.

---

## Open items / next cycle inputs

- **Migration program: 9 of 10 tables done.** Only Cycle 26
  (trades) and Cycle 27 (`_to_latest_ms` autodetect heuristic
  collapse) remain.
- **Cycle 26** (trades): largest table (~8.4M rows). Already
  near-conforming (timestamp INTEGER ms). Need to investigate
  process pattern (long-lived loop vs scheduled) before
  drafting -- per Cycle 23.5/24.5 vs 25.5 contrast, the pattern
  determines the cleanup ordering.
- **Cycle 27**: `_to_latest_ms` autodetect heuristic collapse
  in `servers/praxis_mcp/tools/meta.py`. After all tables are
  ms-format, the autodetect helper's "auto" branch becomes
  redundant.
- **Cycle 29** (audit pass): apply Cycle 28's status-dict
  pattern to `collect_ohlcv_daily`, `collect_ohlcv_4h`,
  `collect_fear_greed`, `collect_funding_rates`,
  `collect_onchain_btc`. Half-hour cycle.
