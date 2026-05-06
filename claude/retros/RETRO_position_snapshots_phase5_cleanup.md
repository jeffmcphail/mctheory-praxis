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
  via the runtime-introspection writer; 79,076 rows at drop
  time).
- `position_snapshots_v2` dropped (empty stub recreated each
  PraxisSmartMoney fire by `init_db()`'s `CREATE TABLE IF NOT
  EXISTS` from Cycle 25 Phase 0).
- Two writer sites in `engines/smart_money.py` (`cmd_snapshot`
  and `cmd_monitor` -- the brief's "cmd_loop" was a memory-
  reconstruction error; the actual function name is `cmd_monitor`)
  both collapsed to single-write; removed runtime PK
  introspection, dual-INSERT logic, and the
  `position_snapshots_v2` CREATE in `init_db()`. Net:
  91 deletions / 20 insertions (-71 net).

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

- `init_db()` (line 58): removed `CREATE TABLE
  position_snapshots_v2` block + its preceding Cycle-25 comment.
  23 lines removed (no separate index existed for v2 in this
  cycle, unlike Cycles 23/24).
- `cmd_snapshot()` writer (line 342 post-collapse; was 411
  pre-collapse): replaced runtime-introspection + dual-INSERT
  block with single INSERT into the live table via the shared
  helper.
- `cmd_monitor()` writer (line 666 post-collapse; was 736
  pre-collapse -- and note the brief said "cmd_loop" but the
  actual function name is `cmd_monitor`): same collapse via the
  same helper.
- **DID extract a shared helper** `_insert_position_row` (line
  139 post-collapse) -- Code took the optional DRY refactor
  named in the brief. Replaces the prior `_insert_position_pair`
  helper (which now had no reason to exist as "pair" implied
  legacy + new). Also removed `_position_snapshots_pre_cutover`
  introspection helper entirely.
- Net: 91 deletions / 20 insertions (-71 net).

py_compile clean. Committed as `83ce624` (Cycle 25.5 step 1).

### Step 2: Cleanup migration script

Ran `scripts/migrations/cycle25_5_position_snapshots_cleanup.py`.

Pre-flight checks PASSED:
- Live table has post-cutover schema (compound PK without `id`,
  has `datetime` column, `MAX(timestamp) > 1e12` confirms ms
  format).
- Legacy row count: 79,076; live: 79,076; ratio: 100.00% exactly.
  **Cleanest cutover in the migration program** -- compare
  Cycle 23.5 (99.99%, 8-row gap from OrderBook in-flight) and
  Cycle 24.5 (99.25%, 3,396-row gap from LiveCollector
  kill-mid-write). The perfect ratio here is a direct consequence
  of the scheduled-not-long-lived process pattern: PraxisSmartMoney
  exits cleanly between fires, so there are zero in-flight writes
  to be lost during the Phase 4 RENAME pair. Future Phase 5 cycles
  on scheduled collectors should expect to see this pattern.
- `_v2` stub was empty.
- Pre-flight #4 (legacy age guard): legacy's most recent write
  was 7,777s ago, far past the 60s threshold (next scheduled
  fire is hours away).

DROP transaction wall-clock: sub-second.

### Step 3: Verification

`get_collector_health` reports `position_snapshots` clean:
- row_count: 79,076 (unchanged from drop time; PraxisSmartMoney
  won't fire again until 20:24 UTC)
- staleness_seconds: 7,869 (well below 28,800s threshold)
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
  `DONE | <CYCLE_25_5_HASH>`. Row 25.5 marked DONE. Per-cycle
  prose section #9 gained a Phase 5 paragraph noting the
  100.00% ratio + the scheduled-vs-long-lived process contrast.
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
production scheduled path, `cmd_monitor` for ad-hoc continuous
mode -- the brief's "cmd_loop" was a memory-reconstruction
mistake; the actual function is named `cmd_monitor`). Both
needed the same dual-write treatment in Cycle 25 and both got
the same collapse here. **Code DID extract a shared helper**
(`_insert_position_row`) per the optional DRY refactor named
in the brief -- a clean replacement for the prior
`_insert_position_pair` (the "pair" name no longer made sense
once there was only one row to write). Both call sites are
now ~5 lines vs the previous ~10-line introspection-plus-call
shape.

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
