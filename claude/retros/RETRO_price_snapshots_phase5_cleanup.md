# Retro: Cycle 24.5 -- price_snapshots Phase 5 cleanup

**Brief:** `claude/handoffs/BRIEF_price_snapshots_phase5_cleanup.md`
**Date:** 2026-05-06
**Mode:** Hybrid (Claude drafted, Code applied writer-file edit)
**Status:** DONE
**Predecessor:** Cycle 24 (`b8fa847`, `6ca1796`, `dbecb23`)

---

## Summary

Phase 5 cleanup of Cycle 24's dual-write. After ~30h of clean
post-cutover operation, the `_legacy` and `_v2` artifacts were
removed:

- `price_snapshots_legacy` dropped (was receiving dual-writes
  via the runtime-introspection writer; 448,941 rows at drop
  time).
- `price_snapshots_v2` dropped (empty stub recreated each
  collector startup by `init_db()`'s `CREATE TABLE IF NOT EXISTS`
  from Cycle 24 Phase 0).
- Writer in `engines/live_collector.py` collapsed to single-write:
  removed runtime PK introspection, removed dual-INSERT logic,
  removed the `price_snapshots_v2` CREATE in `init_db()`. Net:
  49 deletions / 5 insertions (44 lines net).

No data loss; the live `price_snapshots` (post-cutover ms schema)
is unchanged and remains the only `price_snapshots*` table in
`live_collector.db` after this cycle.

---

## Execution log -- ORDERING (per Cycle 23.5 lesson)

This was the FIRST cycle to apply the corrected ordering:

  1. Writer collapse committed FIRST.
  2. Long-lived PraxisLiveCollector process killed.
  3. Fresh process spawned with new code.
  4. Cleanup script run.

Cycle 23.5 went DB-cleanup-first and cascaded into a multi-collector
outage (trades/ohlcv_1m/funding/daily all died via SQLite write-lock
contention from the OrderBook process hitting dropped `_legacy`).
The reverse ordering here prevents that failure mode.

### Step 1: Writer collapse + commit + push

Code edited `engines/live_collector.py` per the hybrid brief:

- `init_db()`: removed `CREATE TABLE price_snapshots_v2` block
  + index. 16 lines.
- Price-write path: replaced runtime-introspection + dual-INSERT
  block with single INSERT into the live table.
  33 deletions / 5 insertions (28 net) for the writer block alone.

py_compile clean. Committed as `88e5d8d` (Cycle 24.5 step 1).

### Step 2: Kill long-lived process

PraxisLiveCollector is `python -u -m engines.live_collector start
--top 50 --interval 60`. Identified PID via:

```powershell
Get-Process python | Where-Object {
  (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine -like "*live_collector start*"
}
```

Killed via `Stop-Process -Id <PID> -Force`. PraxisLiveCollector is
configured with retry-on-exit, so a fresh process spawned within
~<SECONDS> seconds with the new code.

### Step 3: Cleanup migration script

Ran `scripts/migrations/cycle24_5_price_snapshots_cleanup.py`.

Pre-flight checks PASSED:
- Live table has post-cutover schema (ms timestamps confirmed via
  `MAX(timestamp) > 1e12`).
- Legacy row count: 448,941; live: 452,337; ratio: 99.25%.
- `_v2` stub was empty.
- **Pre-flight #4** (Cycle 24.5-specific guard): legacy's most
  recent write was 260s ago, well over the 60s threshold,
  confirming the writer collapse took effect and no rogue process
  is still hitting `_legacy`. **This pre-flight is the load-bearing
  prevention against the Cycle 23.5 cascade pattern**: it refuses to
  drop `_legacy` while a stale in-memory writer might still be
  hitting it, which is exactly what triggered the multi-collector
  outage in Cycle 23.5. Defense-in-depth on top of the corrected
  step ordering.

DROP transaction wall-clock: sub-second.

### Step 4: Verification

`get_collector_health` reports `price_snapshots` clean:
- row_count: 452,387 (growing normally; +50 over the cleanup window)
- staleness_seconds: 5.5 (well below 180s threshold)
- is_stale: false
- `databases.live_collector.unmonitored` is now
  `["collection_log", "spike_alerts", "tracked_markets"]` --
  `price_snapshots_legacy` and `price_snapshots_v2` are gone.
- No `__error__` artifacts; other DBs healthy.

No regressions: other collectors still healthy.

### Step 5: Doc updates

- `docs/SCHEMA_NOTES.md`: row updated from
  `CONFORMING (DONE-PARTIAL) | 24` to `CONFORMING | 24 + 24.5`.
  Per-table prose: removed dual-write writer paragraph.
- `docs/SCHEMA_MIGRATION_PLAN.md`: row updated from
  `DONE-PARTIAL | 6ca1796` to `DONE | 1016ea5`.
- `claude/TODO.md`: Cycle 24.5 added to "Recently closed".

---

## Notes

### First cycle with the corrected ordering

Cycle 23.5's DB-cleanup-first ordering caused the multi-collector
outage on 2026-05-05 night. This cycle reversed the order and ran
cleanly. Memory entry #11 captures the lesson.

The cleanup script's pre-flight #4 is the empirical guard that
prevents a recurrence: by checking how recently `_legacy` was
written, the script refuses to drop it if a stale in-memory
writer is still alive. This is defense-in-depth on top of the
ordering discipline.

### Hybrid workflow: second cycle

Cycle 23.5 was the first hybrid cycle. This is the second.
Same split: Claude drafts cleanup script + writer-collapse brief,
Code applies the on-disk file edit, Claude exercises the live
MCP for verification.

Wall-clock and active drafting time: <DURATION_TOTAL> /
<DURATION_ACTIVE>.

---

## Open items / next cycle inputs

- **Cycle 25.5** (position_snapshots Phase 5 cleanup) remains
  queued. PraxisSmartMoney is scheduled (not long-lived) -- so
  it follows Cycle 23.5's natural pattern, not this cycle's.
  Same ordering trick is fine since the next scheduled invocation
  picks up new code automatically.
- **Cycle 26** (trades): largest table; near-conforming already
  (timestamp already INTEGER ms). Determine process pattern
  (long-lived vs scheduled) before drafting Brief.
