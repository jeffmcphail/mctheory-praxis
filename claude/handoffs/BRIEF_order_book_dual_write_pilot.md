# BRIEF: Cycle 23 -- order_book_snapshots Dual-Write Pilot

**Series:** praxis
**Cycle:** 23
**Mode:** B (Brief -> Code)
**Author:** Chat (praxis_main_current)
**Date:** 2026-05-04
**Predecessor:** Cycle 22 (`5c1f248`, `43f4467`) -- ohlcv_1m migration

---

## Context

Cycle 23 migrates `order_book_snapshots` to Rule 35 using the dual-write
recipe (Rule 35.6, Phases 0-5). This is the **first dual-write cycle in
the migration program** -- an inflection point. The recipe was sketched
in Cycle 17's rules update but never executed end-to-end. This cycle
both runs the pilot AND writes up the recipe as a durable section in
`docs/SCHEMA_MIGRATION_PLAN.md` for use by Cycles 24-26.

**Why dual-write here, not stop-migrate-start.** `PraxisOrderBookCollector`
runs a **continuous 10-second-cadence loop for 3550 seconds per hourly
invocation**. Stopping the collector to migrate would create a
30-90-second gap of missed snapshots; the live collector loop has no
hot-reload mechanism. Dual-write avoids the gap by having the collector
write to BOTH the old and new tables during a burn-in window, then
flipping readers atomically.

**Pre-Brief audit findings.** Three things surfaced that the Brief needs
to address up front:

### Finding 1: The current writer truncates real precision (FIX REQUIRED)

`engines/crypto_data_collector.py:601` does:

```python
ts_ms = ob.get("timestamp") or int(time.time() * 1000)
dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
# ...
asset, ts_ms // 1000, dt, mid, ...   <-- LINE 648, EXPLICIT // 1000 TRUNCATION
```

The writer **already has** ms precision from Binance's API but
deliberately downgrades it to seconds before storing. The matching
`datetime` field preserves sub-second precision (e.g.,
`"2026-05-04T17:16:23.721000+00:00"`). So the schema already has
sub-second info via datetime, but timestamp throws it away.

**Two snapshots within the same second on the same asset have
distinct datetimes but identical timestamps.** Currently the table
has **no two such snapshots in production** (the 10s cadence makes
within-second collisions impossible) but if the cadence were ever
faster, they would collide on the compound `(asset, timestamp)` PK.

**Migration design (option B chosen)**: parse the existing `datetime`
field (which has the real sub-second precision) and convert to ms
during the migration. The new writer drops the `// 1000` truncation
and stores `ts_ms` directly. **Result: no precision loss; future
within-second snapshots get distinct timestamps.**

### Finding 2: The MCP `order_book.py` tool has pre-existing unit-mismatch bugs

`servers/praxis_mcp/tools/order_book.py`:

- `get_order_book_snapshot(asset, at_timestamp_ms)` (line 41-49): does
  `ABS(timestamp - at_timestamp_ms)` to find nearest. Currently the
  table stores seconds and clients pass ms; the math is mathematically
  off but produces "the latest row" in practice (because both inputs
  grow over time). **Buggy but functional.**
- `get_order_book_range(start_ts_ms, end_ts_ms)` (lines 80-114): does
  `WHERE timestamp BETWEEN ? AND ?`. Currently returns
  `total_in_range = 0` for any sane ms input. **Genuinely broken.**

Cycle 23's migration silently fixes both (timestamp becomes ms,
clients pass ms, math becomes meaningful). **The Brief surfaces this
as a known pre-existing bug being fixed incidentally**; the retro
should document it explicitly.

### Finding 3: Row count at Brief-write time

86,583 rows total (BTC: 43,291 + ETH: 43,292) at 2026-05-04 17:16 UTC.
Growing ~720 rows/hour during active collector loops. By cycle execution
time may be 87-95k.

---

## The dual-write recipe (Rule 35.6 expanded)

This section becomes the canonical recipe in
`docs/SCHEMA_MIGRATION_PLAN.md`. Six phases:

### Phase 0: Build `<table>_v2` and the dual-write writer

- CREATE TABLE `order_book_snapshots_v2` with the target Rule 35 schema
  alongside the existing table. **Same database**.
- Modify `collect_order_book_snapshot` to write to BOTH tables in the
  same transaction:
  - Old table: store `ts_ms // 1000` (preserve current behavior;
    legacy readers unaffected)
  - New table: store `ts_ms` directly (no truncation)
- Both INSERTs in the same `conn.execute(...)` chain inside one
  transaction. If either fails, both roll back.
- Deploy this writer change. Collector loop picks it up on its next
  hourly task invocation.

### Phase 1: Parallel collection burn-in

- Wait for at least 60 minutes of dual-write operation. Verify both
  tables grow at the same rate (`COUNT(*)` from each, modulo a small
  per-asset offset).
- Concretely: BTC + ETH ~12 rows/min, so 60 min = ~720 rows in each.
  If `_v2` lags significantly, surface the cause before proceeding.

### Phase 2: Backfill the legacy data into `_v2`

- One-shot script `scripts/migrations/cycle23_order_book_backfill_v2.py`
- For each row in legacy `order_book_snapshots` not already in
  `order_book_snapshots_v2` (i.e., row with `(asset, NEW_TS)` not
  present where NEW_TS is the ms-precision version of the legacy
  datetime), INSERT INTO `_v2` with timestamp parsed from
  `datetime` to ms.
- Idempotent: re-running on a fully-backfilled state produces zero
  inserts.
- This runs while the collector is still dual-writing -- `_v2` is
  growing from both sides.

### Phase 3: Verification of overlap

- Sanity check that `_v2` has at least every row that legacy has,
  modulo precision: for every row in legacy, there should be exactly
  one row in `_v2` with the same `(asset, datetime)` (regardless of
  exact timestamp).
- If any legacy row is missing from `_v2`, ABORT.
- Sample 100 random rows from the dual-write window (rows written
  by Phase 1's writer to both tables) and verify the data is
  byte-identical between legacy and `_v2`, modulo the timestamp
  precision difference.

### Phase 4: Atomic cutover (the dangerous step)

In a single transaction:
1. RENAME `order_book_snapshots` to `order_book_snapshots_legacy`.
2. RENAME `order_book_snapshots_v2` to `order_book_snapshots`.

Both reads and writes immediately switch to the new schema
representation. The collector's writer is still dual-writing, but
now: "old" = `_legacy`, "new" = the live table. Readers (MCP tools)
seamlessly start reading the new ms-precision data.

This RENAME pair is genuinely atomic in SQLite (single transaction,
both DDL statements complete or none do). Other DBs do this
differently; we benefit from SQLite's simplicity here.

### Phase 5: Burn-in 24-48h, then drop legacy + collapse writer

- Wait 24-48h. Confirm no MCP tool errors, no collector errors, no
  reader engine errors.
- After burn-in:
  1. Modify the writer to single-write to the new table only
     (drop the `_legacy` INSERT).
  2. DROP TABLE `order_book_snapshots_legacy`.
- Phase 5 is **deferred to Cycle 23.5 hotfix-style cleanup**, not
  bundled into this cycle's main commit. This keeps the cycle's main
  commit's blast-radius small and gives 24-48h of real runtime
  before we drop the rollback safety net.

---

## Scope (Cycle 23 = Phases 0-4 only; Phase 5 is Cycle 23.5)

### Task 1: Schema design + Phase 0 (dual-table + dual-write writer)

**Target schema** for `order_book_snapshots`:

```sql
CREATE TABLE order_book_snapshots (
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,         -- UTC milliseconds (Binance API native)
    datetime TEXT NOT NULL,             -- "YYYY-MM-DDTHH:MM:SS.NNNNNN+00:00" ISO+sub-sec
    mid_price REAL NOT NULL,
    best_bid REAL NOT NULL,
    best_ask REAL NOT NULL,
    spread REAL NOT NULL,
    spread_bps REAL NOT NULL,
    bid_price_1 REAL, bid_vol_1 REAL,
    -- ... all 10 bid + 10 ask levels as in current schema ...
    bid_volume_top10 REAL NOT NULL,
    ask_volume_top10 REAL NOT NULL,
    order_imbalance_top10 REAL NOT NULL,
    PRIMARY KEY (asset, timestamp)
)
```

Changes from current schema:
- Drop `id` AUTOINCREMENT
- `timestamp` INTEGER seconds -> milliseconds (preserves sub-second
  precision from `datetime`)
- `datetime` keeps sub-second precision (already conforming format
  per inspection)
- Compound PK on `(asset, timestamp)` (subsumes UNIQUE)

**Phase 0 implementation:**

1. Add a CREATE TABLE for `order_book_snapshots_v2` to `init_db()` in
   `engines/crypto_data_collector.py`. Create idempotently
   (`CREATE TABLE IF NOT EXISTS`).
2. Modify `collect_order_book_snapshot` (line 585+) so that the
   existing single INSERT becomes TWO INSERTs in the same transaction:

   ```python
   # OLD: single INSERT to order_book_snapshots
   # NEW: dual-write to both tables

   # ts_ms variable already exists from line 601
   # dt variable already exists from line 602

   try:
       cursor = conn.cursor()

       # Write to legacy table (preserve existing behavior; seconds-truncated)
       cursor.execute("""
           INSERT OR IGNORE INTO order_book_snapshots (
               asset, timestamp, datetime, ... [unchanged column list]
           ) VALUES (..., ?, ts_ms // 1000, dt, ...)
       """, [...])

       # Write to v2 table (Rule 35 compliant; ms preserved)
       cursor.execute("""
           INSERT OR IGNORE INTO order_book_snapshots_v2 (
               asset, timestamp, datetime, ... [same column list, no id]
           ) VALUES (..., ?, ts_ms, dt, ...)
       """, [...])

       conn.commit()
   except Exception:
       conn.rollback()
       # ... existing error handling
   ```

   Both INSERTs share the same column construction logic (just
   different `timestamp` value). Refactor minimally to avoid
   duplication: build the column list and values list once, then
   issue two INSERTs against them.

3. Verify the writer change loads cleanly via syntax check
   (`python -m py_compile engines/crypto_data_collector.py`).

4. **Do NOT manually trigger the collector during this phase.** The
   live collector loop will pick up the writer change on its next
   hourly invocation (`PraxisOrderBookCollector` task launches every
   hour at :00; runs for 3550 seconds). Code can verify this by
   waiting for the next hour boundary and confirming both tables grow.

5. **Phase 0 commit**: Stage the writer changes + the `_v2` table
   creation. Commit and push. Do NOT proceed to Phase 1 verification
   in the same commit window -- let the live collector pick it up
   first.

### Task 2: Phase 1 burn-in verification

After at least 60 minutes of dual-write operation (confirmed by checking
that both tables have grown):

1. Query both tables for the count over a 60-minute window starting
   30 minutes ago (a clean dual-write window where both tables should
   have been written to from the start):
   ```sql
   -- legacy
   SELECT COUNT(*) FROM order_book_snapshots
   WHERE timestamp BETWEEN ? AND ?  -- seconds boundaries

   -- v2
   SELECT COUNT(*) FROM order_book_snapshots_v2
   WHERE timestamp BETWEEN ? AND ?  -- ms boundaries
   ```
2. Counts should match within +/-2 (small tolerance for boundary timing).
3. Spot-check 5 random rows in this window: verify both tables have a
   matching `(asset, datetime)` pair, with timestamp values consistent
   (ms version = seconds version x 1000 + sub-second offset from datetime).

If counts diverge by more than 2 or any sample mismatches, **abort**
and surface to chat. Do not proceed to Phase 2 until this passes.

### Task 3: Phase 2 backfill script

Create `scripts/migrations/cycle23_order_book_backfill_v2.py`:

```python
"""
Cycle 23 Phase 2 -- backfill order_book_snapshots_v2 from legacy.

Idempotent: re-running after a successful backfill produces 0 inserts.
"""
```

Logic:

1. Open `data/crypto_data.db` (Rule 34: fresh connection).
2. SELECT every row from `order_book_snapshots` whose `(asset, datetime)`
   does NOT exist in `order_book_snapshots_v2`. (The `datetime` is the
   stable identifier across both tables; timestamp differs by 1000x.)
3. For each missing row:
   - Parse `datetime` to a `datetime.datetime` object (it's
     ISO+offset+microsecond format).
   - Compute `timestamp_ms = int(dt.timestamp() * 1000)`.
   - INSERT INTO `order_book_snapshots_v2` with the new ms timestamp +
     all other columns from the legacy row (omitting `id`).
4. Wrap in a single transaction. Print before/after counts.
5. Idempotent guard: if `n_missing == 0`, print "Already backfilled"
   and exit cleanly (exit 0).
6. Performance: ~87k rows; based on Cycle 22's empirical 0.567s for
   530k rows, this should be sub-second.

### Task 4: Phase 3 verification script

Either inline in the cycle's verification scratch script or as
`scripts/migrations/cycle23_order_book_verify.py`:

1. Total rows in `_v2` >= total rows in legacy (legacy is a strict
   subset of `_v2` after backfill, plus dual-write may have added
   newer rows to both). Strictly: `count_v2 == count_legacy +
   count_after_backfill_window`. The simpler check
   (`v2 >= legacy AND every (asset, datetime) in legacy is in v2`)
   is sufficient.
2. For every row in legacy: there is exactly one row in `_v2` with
   matching `(asset, datetime)`. (If multiple, something went wrong
   in dual-write; if zero, backfill missed it.)
3. Sample 100 random rows from BOTH a) the dual-write window
   (post-Phase-0 timestamps) and b) the backfilled range (pre-Phase-0
   timestamps). For each: compare every column except `timestamp`
   and `id` between legacy and `_v2`. They should be byte-identical.
4. Print summary: total rows in each table, missing count (must be 0),
   sample mismatch count (must be 0).

If any check fails, **abort the cycle**. Surface to chat. Do not
proceed to cutover.

### Task 5: Phase 4 atomic cutover

Single transaction in a cutover script
`scripts/migrations/cycle23_order_book_cutover.py`:

```sql
BEGIN;
ALTER TABLE order_book_snapshots RENAME TO order_book_snapshots_legacy;
ALTER TABLE order_book_snapshots_v2 RENAME TO order_book_snapshots;
COMMIT;
```

- Idempotent: if `order_book_snapshots_legacy` already exists AND
  `order_book_snapshots_v2` does NOT, the cutover already happened;
  print "Already cut over" and exit 0.
- Verify post-cutover via PRAGMA table_info that the live table has
  the new schema.
- Verify the writer (still dual-writing) doesn't break -- the next
  collector cycle should still INSERT successfully into both tables
  (one of which is now `_legacy`).

### Task 6: Update `meta.py` and `order_book.py`

**`servers/praxis_mcp/tools/meta.py`**:

The `monitored_tables` dict has `"order_book_snapshots": 3900`
(65 minutes). Threshold value unchanged post-migration. The autodetect
heuristic (`> 1e12 -> ms`) handles the timestamp unit change
transparently. **No code change required here**; just verify
post-cutover that `get_collector_health` reports `order_book_snapshots`
correctly with `is_stale=false`.

**`servers/praxis_mcp/tools/order_book.py`**:

Pre-existing unit-mismatch bug in `get_order_book_range` (returns 0
for any ms input against seconds storage) is silently fixed by the
migration. The `get_order_book_snapshot` function's `ABS(timestamp - ?)`
math becomes meaningful post-migration too.

**Required changes:**
1. Update the docstrings to confirm `timestamp` is ms (already says
   `at_timestamp_ms` and `start_ts_ms`/`end_ts_ms`, so the contract
   was always ms; the table just didn't honor it). Add a note that
   pre-Cycle-23 the tool returned wonky results due to the unit
   mismatch.
2. No body code changes. The math becomes correct as the data unit
   changes.

### Task 7: Update doc trio

**`docs/SCHEMA_NOTES.md`**:
- `order_book_snapshots` per-table prose: NONCONFORMING -> CONFORMING
  (Cycle 23). Update column type/format notes.
- Migration status table: row #7 -> CONFORMING / 23 / dual-write.
- Add a note that `get_order_book_range` MCP tool was silently fixed
  by the migration (pre-existing unit-mismatch bug).

**`docs/SCHEMA_MIGRATION_PLAN.md`**:
- Status summary row #7: change to
  `order_book_snapshots | dual-write | DONE-PARTIAL | <commit>`
  (DONE-PARTIAL because Phase 5 cleanup is deferred to Cycle 23.5).
- Per-table spec section: rewrite the full pilot history.
- **NEW SECTION at top of doc** (after status table, before per-table
  specs): "Dual-write recipe (Rule 35.6 expanded)" -- the six-phase
  recipe documented above. This becomes the durable template for
  Cycles 24-26.

**`claude/TODO.md`**:
- Mark Cycle 23 as DONE-PARTIAL in Recently closed (with commit hash).
- Add to Active high-priority: "Cycle 23.5: Phase 5 cleanup of
  order_book_snapshots dual-write -- drop the `_legacy` INSERT from
  the writer, DROP TABLE `order_book_snapshots_legacy`. Run after
  24-48h burn-in confirmed clean."
- Add: "Cycle 24: Migrate `live_collector.price_snapshots` per
  the dual-write recipe pattern established in Cycle 23." (Sidecar
  DB; new wrinkle: the writer is in `engines/live_collector.py`,
  not `crypto_data_collector.py`.)

### Task 8: Retro

`claude/retros/RETRO_order_book_dual_write_pilot.md` with:

- The full Phase 0/1/2/3/4 execution log
- Pre/post snapshots at each phase
- Performance datapoints (backfill wall-clock; cutover transaction time)
- Sample data showing the precision recovery (a row pre-migration:
  ts=1777914973, dt=...:13.435; same row post-migration: ts=1777914973435,
  dt=...:13.435 -- note the alignment)
- The two pre-existing MCP tool bugs that got silently fixed
- Verification that BTC/ETH writers still behave under dual-write
- Cross-table sanity check (carry forward Cycle 21.5's defensive habit)
- Lessons-learned for the dual-write recipe itself: anything that
  didn't go cleanly, gotchas, sequencing issues. **This retro is
  load-bearing for Cycles 24-26**; document the recipe's actual
  rough edges, not just its nominal steps.

---

## Out of scope

- Phase 5 cleanup (drop `_legacy`, single-write writer) -- deferred to
  Cycle 23.5 after 24-48h burn-in
- Migrating any other table
- Changing the staleness threshold for `order_book_snapshots`
- Refactoring the writer beyond what dual-write requires
- Touching `_to_latest_ms` autodetect heuristic (Cycle 27)
- Investigating the pre-existing MCP tool bugs deeper than noting
  they're fixed by the migration

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | `data/crypto_data.db.cycle23_backup` created BEFORE Phase 0 (md5-verified) |
| 2 | `data/crypto_data.db.cycle22_backup` deleted at start of cycle |
| 3 | Phase 0: `order_book_snapshots_v2` table created with target Rule 35 schema |
| 4 | Phase 0: writer modified to dual-write; py_compile clean |
| 5 | Phase 0: writer change committed BEFORE Phase 1 verification (committed and pushed; let collector pick it up) |
| 6 | Phase 1: at least 60 minutes of dual-write observed before backfill; both tables grew at the same rate (counts match within +-2 over a clean 60-min window) |
| 7 | Phase 1: spot-check 5 random rows in dual-write window confirms (asset, datetime) match across both tables |
| 8 | Phase 2: `scripts/migrations/cycle23_order_book_backfill_v2.py` exists, idempotent (re-run on backfilled state exits 0 with "Already backfilled") |
| 9 | Phase 2: backfill complete -- every legacy row has a matching `(asset, datetime)` in `_v2` |
| 10 | Phase 3: verification script confirms zero missing rows + zero sample mismatches |
| 11 | Phase 4: cutover script atomically renames `order_book_snapshots -> _legacy` AND `_v2 -> order_book_snapshots` in single transaction |
| 12 | Phase 4: post-cutover, PRAGMA table_info confirms live table has new Rule 35 schema |
| 13 | Phase 4: post-cutover, `get_collector_health` reports `order_book_snapshots` with is_stale=false; row count from new table |
| 14 | Phase 4: MCP `get_order_book_snapshot(asset='BTC', at_timestamp_ms=<recent>)` returns a row (not error); `get_order_book_range(asset='BTC', start_ts_ms=<5min ago>, end_ts_ms=<now>)` returns non-zero rows (this fixes the pre-existing bug) |
| 15 | Phase 4: writer (still dual-writing for next ~24h) continues to INSERT successfully into BOTH the live table (new schema) and `_legacy` (old schema). Verify by waiting for next collector cycle and checking row growth in both. |
| 16 | `servers/praxis_mcp/tools/order_book.py` docstrings updated (no body changes) |
| 17 | `docs/SCHEMA_NOTES.md` updated |
| 18 | `docs/SCHEMA_MIGRATION_PLAN.md` updated WITH new "Dual-write recipe" section near the top |
| 19 | `claude/TODO.md` updated (Cycle 23 DONE-PARTIAL, Cycle 23.5 + Cycle 24 added) |
| 20 | All committable files ASCII-only (Rule 20) |
| 21 | Retro at `claude/retros/RETRO_order_book_dual_write_pilot.md` documents the actual phase sequence, lessons, and any deviations |
| 22 | Cycle 23.5 (Phase 5 cleanup) explicitly NOT executed in this cycle; deferred to its own cycle after 24-48h burn-in |

---

## Notes for Code

- **The collector loop runs continuously**. When you change the writer
  and commit, the very next hourly invocation of
  `PraxisOrderBookCollector` (which fires at `:00` of every hour and
  runs for 3550s) picks it up. **Time the cycle so Phase 0 commits
  land at least 5 minutes before the next `:00` hour boundary** so
  the change is in place when the loop restarts. If the next
  invocation has already started before Phase 0 commits, that
  invocation continues writing the OLD (single-write) format until
  it exits at ~:59 and the next one starts.

- **Phase 0 commit is its own commit, deliberately separate from
  Phases 2-4.** This gives a clean rollback point if dual-write
  reveals issues. Push Phase 0 -> wait -> Phase 1 verify -> Phase 2-4
  in subsequent commits.

- **No backup created automatically by Phase 0 itself.** Code creates
  `data/crypto_data.db.cycle23_backup` BEFORE Phase 0 modifies
  anything (md5-verified). This is the rollback point if anything
  goes wrong before cutover.

- **Rollback path for Phase 4**: if the cutover transaction commits
  but post-cutover verification fails, the rollback is to RENAME the
  tables back. Both names exist; both ALTER TABLE RENAMEs in a single
  transaction. Document this in the retro.

- **Rule 34** throughout: fresh connection per logical pass.

- **Rule 32**: changes to `meta.py` (none needed this cycle but if
  any) require Claude Desktop full-restart to take effect. Cycle 23
  doesn't change meta.py.

- **Idempotency guards**: each script (backfill, verify, cutover)
  detects whether its phase has already been executed and exits
  cleanly if so. Re-running any individual script after a clean
  cycle should be a no-op.

- **The dual-write window itself is genuinely uncertain in duration**.
  The Brief asks for at least 60 minutes; in practice you might want
  to extend if the collector hasn't fired enough cycles. Use your
  judgment; document the actual elapsed dual-write time in the retro.

- **Don't bundle Phase 5 in this cycle.** Phase 5 is the "drop the
  safety net" step; it should run only after we've seen 24-48h of
  the new schema being read and written cleanly. Bundling Phase 5
  with Phases 0-4 defeats the whole point of having a burn-in
  window.

- **Sub-second precision** in the new ms timestamp is the meaningful
  precision recovery from this migration. Document it. Future
  cycles working with this table can rely on millisecond-level
  ordering instead of being seconds-truncated. **This is the kind
  of detail that's load-bearing for downstream HFT-style analytics.**

- **The writer-alignment audit** (per Cycle 21.5 lesson, codified in
  Cycle 22) has a different shape here. The writer doesn't truncate
  jitter -- it preserves the API's ms timestamp. Document in retro
  that the unit-mismatch in pre-Cycle-23 readers (the MCP tools)
  was the analog of the writer-alignment issue, just on the read
  side instead of the write side.

- **Post-cycle validation by Chat**: after push, Chat will verify
  via MCP that `get_order_book_range` works (where it didn't before),
  and that row counts continue to grow in both tables (proving
  dual-write is still live and supporting the 24-48h burn-in for
  Cycle 23.5).

- **Two-commit pattern OK.** This cycle will likely produce 2-3
  separate commits anyway: Phase 0 (writer + table creation), then
  Phases 2-4 (backfill, verify, cutover, doc updates). The plan-doc
  hash insertion follows Cycle 18/20/21/22 precedent -- `<TBD>` then
  hash-patch.
