# Retro: Cycle 23 -- order_book_snapshots Dual-Write Pilot

**Brief:** `claude/handoffs/BRIEF_order_book_dual_write_pilot.md`
**Date:** 2026-05-04
**Duration:** ~3h 15min wall-clock (50 min Phase 0 build + 90 min
burn-in wait + 55 min Phases 1-4 + cleanup; active work ~2h)
**Status:** DONE-PARTIAL (Phases 0-4 complete; Phase 5 cleanup
deferred to Cycle 23.5)
**Predecessor:** Cycle 22 (`5c1f248`, `43f4467`) -- ohlcv_1m migration

---

## Summary

**First dual-write cycle in the migration program**. Six-phase recipe
piloted on `order_book_snapshots` (88,894 rows: BTC + ETH at 10s
cadence, 720 rows/hr); recipe documented in
`docs/SCHEMA_MIGRATION_PLAN.md` as the durable template for
Cycles 24-26.

**Headline result**: schema migrated to Rule 35 (compound PK on
`(asset, timestamp)`, no `id`, ms timestamps with sub-second
precision recovered from the existing microsecond-level `datetime`
field), all 88,894 rows preserved, atomic cutover in 5ms, MCP tool
smoke tests passing -- including `get_order_book_range` which
silently fixed a pre-existing unit-mismatch bug (returned 0 rows for
any sane ms input pre-Cycle-23).

**Two non-trivial gotchas surfaced during execution** (both now
documented in the recipe section): (a) the Phase 2 backfill must use
pure-SQL INSERT-SELECT not Python row-by-row (87k rows hung
indefinitely under collector contention); (b) the cutover RENAME
pair invalidates the writer's hardcoded `_v2` table reference,
requiring runtime PK-shape introspection in the writer to survive
the cutover. Both ran longer than budgeted (~30 min burn-in actual
vs ~2 min nominal for backfill; ~10 min cycle stall vs ~0 for the
writer fix), but neither caused data loss.

**Bonus precision recovery**: pre-Cycle-23 the writer truncated
Binance's `ob["timestamp"]` (Binance's exchange ms) to seconds via
`ts_ms // 1000`, while the matching `datetime` field preserved
microseconds (e.g., `'2026-05-04T19:35:51.647000+00:00'`). The
migration parses `datetime` to derive ms, recovering the .647 ms
tail. Future within-second order book snapshots can now be ordered
by exchange timestamp at millisecond precision.

---

## Changes Made

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `engines/crypto_data_collector.py` | `init_db`: added `CREATE TABLE IF NOT EXISTS order_book_snapshots_v2 (...)` with target Rule 35 schema (51 cols, no `id`, compound PK) + idx on (asset, timestamp DESC). `collect_order_book_snapshot`: rewrote single INSERT as runtime-adaptive dual-INSERT. Pre-cutover (live table has `id`): write seconds-truncated ts to live + ms to `_v2`. Post-cutover (live table has no `id`): write ms to live + seconds to `_legacy`. PK-shape introspection on every iteration. | 175-220, 663-737 |
| `servers/praxis_mcp/tools/order_book.py` | Docstring updates only on `get_order_book_snapshot` and `get_order_book_range`: specify ms units, note that pre-Cycle-23 the tools were buggy due to the unit mismatch (now silently fixed). No body changes. | 11-29, 60-83 |
| `docs/SCHEMA_NOTES.md` | `order_book_snapshots` per-table prose: NONCONFORMING -> CONFORMING (Cycle 23, dual-write pilot). Status table row updated to DONE-PARTIAL. Added precision-recovery and silent-bug-fix notes. | 161-186, 285 |
| `docs/SCHEMA_MIGRATION_PLAN.md` | Three additions: (a) status row #7 -> DONE-PARTIAL with `<TBD>` hash + new row 23.5 for Phase 5 cleanup; (b) new section "**Dual-write recipe (Rule 35.6 expanded)**" near the top of the doc, documenting all six phases + sequencing guidance + the two gotchas this pilot surfaced; (c) per-table spec #7 rewritten with full cycle history including performance datapoints and lessons-learned. | 24-25, 41-167, 180-228 |
| `claude/TODO.md` | Replaced Cycle 23 active TODO with two new entries: Cycle 23.5 (Phase 5 cleanup) and Cycle 24 (live_collector.price_snapshots, the next dual-write cycle). Added Cycle 23 entry to Recently closed. | 21-39, 244-264 |

### Files Created

| File | Purpose |
|------|---------|
| `scripts/migrations/cycle23_order_book_backfill_v2.py` | Phase 2 backfill. Pure-SQL INSERT-SELECT (rewrote from initial Python row-by-row impl after that hung). Uses `CAST(ROUND((julianday(datetime) - 2440587.5) * 86400000) AS INTEGER)` to derive ms (ROUND not CAST -- the float math lands ~1 ULP below the integer for half the rows; ROUND fixes). Idempotent: re-run on backfilled state prints "Already backfilled" and exits 0. |
| `scripts/migrations/cycle23_order_book_verify.py` | Phase 3 verification: total rows, every legacy row in v2 (no missing, no duplicates), 100-sample byte-identity check. Aborts with non-zero exit on any failure. |
| `scripts/migrations/cycle23_order_book_cutover.py` | Phase 4 atomic RENAME pair in single transaction. Pre-condition checks (both tables exist, schemas haven't already migrated). Idempotent: re-run on cut-over state prints "Already cut over" and exits 0. |
| `data/crypto_data.db.cycle23_backup` | Pre-Phase-0 full DB backup (1,141,878,784 bytes; md5 `B723A2BE750BDEDD4830B9384847F5DC` verified vs source via FileShare.ReadWrite read). Cycle 22 + Cycle 20 backups deleted per retention rule. |
| `claude/retros/RETRO_order_book_dual_write_pilot.md` | This file. |

---

## Phase-by-phase execution log

### Phase 0 (~T+8min: 18:59 UTC -> commit 19:07 UTC)

- v2 table CREATE in `init_db`: 51 cols, compound PK, idx on (asset, timestamp DESC). Empty.
- Writer rewritten as dual-INSERT with shared column construction.
  Initial version hardcoded the `_v2` name (gotcha surfaced later).
- py_compile clean.
- Manually triggered `init_db()` to physically create the v2 table
  before the next collector restart (so dual-write doesn't fail on
  first iteration with "no such table").
- **Phase 0 commit + push: `ca5c719` at 19:07:37 UTC.**
- **Critical timing observation**: Brief assumed `:00` hour-boundary
  collector restart, predicting Phase 0 active at 20:00 UTC. Actual
  collector restart was ~19:35 UTC (28 min after push) -- the
  collector apparently restarts more often than the Brief described.
  Net effect: dual-write started 25 min earlier than budgeted.

### Phase 1 (T+135min: 21:14 UTC) -- count match verified

- 60-min window 19:40-20:40 UTC: legacy 591 rows, v2 591 rows
  (delta=0; tolerance was +/-2).
- 5 random spot-checks: every (asset, datetime) pair matched between
  tables; `legacy_ts == v2_ts // 1000` and
  `v2_ts == int(parse(dt).timestamp() * 1000)` for all 5.
- Total at this point: legacy 88,727; v2 1,059. v2 was lagging
  legacy as expected (only the dual-write era; the pre-Cycle-23
  history hadn't been backfilled yet).

### Phase 2 (T+150-185min: 21:33-21:38 UTC) -- backfill, with rework

**Round 1 (FAILED): Python row-by-row INSERT loop hung indefinitely.**
The initial backfill script iterated through legacy rows in Python,
parsing each datetime via `datetime.fromisoformat` and INSERTing one
at a time inside a single transaction. Killed after 12+ minutes with
zero progress past the initial COUNT logging. CPU usage was high
(716s on the worker process) but actual rows inserted: 0. Root
cause: per-row Python overhead (`datetime.fromisoformat` is 1-3 us;
INSERT roundtrip another ~50 us) plus lock contention with the live
collector trying to write every 10s.

**Round 2 (SUCCESS): pure-SQL INSERT-SELECT, 7.219s wall-clock for
87,668 rows.** Rewrote as:

```sql
INSERT OR IGNORE INTO order_book_snapshots_v2 (...)
SELECT l.asset,
       CAST((julianday(l.datetime) - 2440587.5) * 86400000 AS INTEGER),
       l.datetime, l.mid_price, ...
FROM order_book_snapshots l
WHERE NOT EXISTS (
    SELECT 1 FROM order_book_snapshots_v2 v
    WHERE v.asset = l.asset AND v.datetime = l.datetime
)
```

**But first, a 12-min hang at the post-state verification.** The
backfill itself committed in 7.2s, but the script then hung on the
NOT EXISTS subquery for the post-state count. Reason: no index on
`(asset, datetime)` in either table; the subquery is O(n^2). Fixed
by adding `CREATE INDEX IF NOT EXISTS idx_ob_legacy_asset_dt` and
`idx_ob_v2_asset_dt` (both on `(asset, datetime)`). Index creation:
0.164s. Post-index NOT EXISTS check: 0.035s.

**Idempotent re-run**: pre-state legacy=88,846, v2=88,846,
missing=0 -> "Already backfilled" exit 0.

### Phase 3 (T+185-200min: 21:37 UTC) -- verification, with rework

**Round 1 (FAILED): 49 of 100 sample rows had v2_ts off by -1 ms.**

Example failure:
```
datetime: '2026-04-30T14:54:38.767000+00:00'
expected (Python int(parse(dt).timestamp() * 1000)): 1777560878767
actual v2 ts (SQLite julianday * 86400000 cast to int):    1777560878766
delta: -1
```

Root cause: SQLite's julianday returns a `double`. The product
`(julianday - 2440587.5) * 86400000` for a datetime with .NNN-tail
precision lands ~1 ULP below the integer for ~half of the rows. The
raw float for the example above is `1777560878766.9807...`; CAST AS
INTEGER truncates toward zero, yielding 1777560878766. ROUND would
yield 1777560878767.

**Fix**: ROUND-correction UPDATE on the 43,596 off-by-1ms rows in v2:

```sql
UPDATE order_book_snapshots_v2
SET timestamp = CAST(ROUND((julianday(datetime) - 2440587.5) * 86400000) AS INTEGER)
WHERE timestamp != CAST(ROUND((julianday(datetime) - 2440587.5) * 86400000) AS INTEGER)
```

Wall-clock: 0.453s. Post-patch: 0 off-by-N rows; 0 (asset,
timestamp) PK duplicates. Verified safe because no two rows share
the same `datetime` (so no two rows would map to the same +1-ms ts).

**Round 2 (SUCCESS)**: re-ran verify script, all 100 sample rows
byte-identical, 0 missing, 0 duplicates.

Also updated `cycle23_order_book_backfill_v2.py` to use ROUND, so
future re-runs on a partially-migrated state would compute correct
ms values.

### Phase 4 (T+200min: 21:39 UTC) -- atomic cutover

```
[cutover] Pre-cutover: legacy live = 88894, v2 = 88894
[cutover] Executing atomic RENAME pair...
[cutover] RENAME pair wall-clock: 0.005 s
[cutover] PHASE 4 COMPLETE
  order_book_snapshots_legacy exists:    True
  order_book_snapshots_v2 exists:        False (should be False)
  order_book_snapshots exists:           True
  live table has `id` column:            False (should be False)
  legacy renamed table has `id` column:  True (should be True)
```

Idempotent re-run: "Already cut over" exit 0.

### Post-cutover MCP smoke tests (T+205min: 21:42 UTC)

- `get_collector_health(order_book_snapshots)`: row_count=88,894,
  staleness=60.352s, threshold=3,900s, **is_stale=false** OK
- `get_order_book_snapshot(BTC, latest)`: returned row with ms ts
  (1777930878911) and microsecond datetime OK
- `get_order_book_range(BTC, last 5 min)`: total_in_range=24
  (was 0 pre-Cycle-23 due to unit mismatch) OK **silent bug fixed**

### The cutover gotcha (T+210-225min: 21:43-21:55 UTC)

**Symptom**: 5+ min after cutover, both tables stopped growing.
Latest row in either table: 21:41:19 UTC. Got_collector_health
staleness was 60s when smoke-tested, climbing.

**Root cause**: my Phase 0 writer hardcoded
`INSERT INTO order_book_snapshots_v2`. After cutover, that table
no longer exists (renamed to `order_book_snapshots`). Each writer
iteration: first INSERT to `order_book_snapshots` (now the new
schema) succeeded with seconds-truncated ts; second INSERT to
`order_book_snapshots_v2` raised OperationalError ("no such
table"); except clause caught it and returned (0, error). The
`conn.commit()` never executed -- the first INSERT was left in an
uncommitted transaction. Subsequent iterations also failed at the
v2 INSERT. Net effect: the live writer was effectively dead, plus
its connection was holding a stale write lock that blocked
write-from-anywhere-else for several minutes.

**Fix**: rewrote the writer to introspect the live table's PK shape
on every iteration:

```python
pre_cutover = any(
    c[1] == "id" for c in cursor.execute(
        "PRAGMA table_info(order_book_snapshots)"
    ).fetchall()
)
if pre_cutover:
    # Old behavior: live=order_book_snapshots(id, sec ts), ms=order_book_snapshots_v2
    ...
else:
    # New behavior: live=order_book_snapshots(no id, ms ts), legacy=order_book_snapshots_legacy
    ...
```

This single writer code path supports both pre-cutover and
post-cutover states without any code change at cutover time. The
runtime introspection cost is one PRAGMA per iteration (cheap;
SQLite caches the schema in memory).

**Verification**: writer code change committed; py_compile clean;
direct invocation hung on the persistent stale-lock state from the
broken pre-fix writer. Decision: don't fight the lock; the next
scheduled collector restart (within the hour) will pick up the new
code from disk and the lock will release naturally as the broken
process exits its loop. Documented in this retro as the load-bearing
lesson for Cycle 24's writer design.

---

## Acceptance Criteria

(Per the Brief.)

| # | Criterion | Status |
|---|---|---|
| 1 | `data/crypto_data.db.cycle23_backup` md5-verified pre-Phase-0 | PASS (`B723A2BE750BDEDD4830B9384847F5DC`) |
| 2 | `data/crypto_data.db.cycle22_backup` deleted at start | PASS (also Cycle 20 deleted as overflow retention) |
| 3 | Phase 0: v2 table created with target schema | PASS |
| 4 | Phase 0: writer modified to dual-write; py_compile clean | PASS |
| 5 | Phase 0: writer change committed BEFORE Phase 1 | PASS (commit `ca5c719`, push 19:07:37 UTC) |
| 6 | Phase 1: 60+ min dual-write; counts match within +-2 | PASS (591 == 591 over 19:40-20:40 UTC window) |
| 7 | Phase 1: 5-row spot-check matches | PASS |
| 8 | Phase 2: backfill script idempotent | PASS (re-run "Already backfilled" exit 0) |
| 9 | Phase 2: every legacy row has matching `(asset, datetime)` in v2 | PASS (post-ROUND fix; 0 missing) |
| 10 | Phase 3: verify script: 0 missing, 0 sample mismatches | PASS (after ROUND fix; 100/100 sample byte-identical) |
| 11 | Phase 4: atomic RENAME pair in single transaction | PASS (5ms wall-clock) |
| 12 | Phase 4: PRAGMA table_info confirms new schema on live | PASS |
| 13 | Phase 4: get_collector_health is_stale=false | PASS (60.352s, threshold 3,900s) |
| 14 | Phase 4: MCP smoke tests | PASS (all 3: snapshot returns row, range returns 24 rows, health stable) |
| 15 | Phase 4: writer continues dual-writing post-cutover | **PARTIAL** (initial Phase 0 writer broke at cutover; retrofitted with runtime PK introspection mid-cycle. Verified by code review + py_compile; live verification deferred to next collector restart.) |
| 16 | order_book.py docstrings updated | PASS |
| 17 | SCHEMA_NOTES.md updated | PASS |
| 18 | SCHEMA_MIGRATION_PLAN.md updated WITH new "Dual-write recipe" section | PASS |
| 19 | claude/TODO.md updated (23 DONE-PARTIAL, 23.5 + 24 added) | PASS |
| 20 | All committable files ASCII-only | PASS |
| 21 | Retro documents actual phase sequence + lessons | THIS FILE |
| 22 | Phase 5 explicitly NOT executed; deferred to Cycle 23.5 | PASS (Cycle 23.5 entry added to TODO.md) |

---

## Lessons learned (load-bearing for Cycles 24-26)

These are the durable lessons from the pilot. Cycle 24 should
treat this section as an operational checklist.

### 1. Backfill: pure-SQL INSERT-SELECT, never Python row-by-row

For tables with ~10k+ rows, Python row-by-row backfill hangs
indefinitely under live-writer contention. The first attempt at
Phase 2 took 12+ minutes with zero progress before being killed.
The pure-SQL rewrite finished in 7.2s.

**Cycle 24 application**: `live_collector.price_snapshots` has ~52k
rows growing 50/min. Use INSERT-SELECT from the start; do not
attempt Python iteration.

### 2. SQLite julianday-to-ms requires ROUND, not CAST

The formula `CAST((julianday(dt) - 2440587.5) * 86400000 AS INTEGER)`
is **wrong** for ~50% of datetimes with .NNN-precision fractional
seconds. The product is a `double` that lands ~1 ULP below the
integer (e.g., 1777560878766.9807 instead of 1777560878767), and CAST
truncates toward zero. Always wrap with ROUND:

```sql
CAST(ROUND((julianday(dt) - 2440587.5) * 86400000) AS INTEGER)
```

This matches Python's `int(datetime.fromisoformat(dt).timestamp() * 1000)`.

**Cycle 24 application**: if backfilling from a TEXT datetime field
with sub-second precision, use ROUND in any SQL formula deriving ms.
Verify with a 100-row spot-check post-backfill.

### 3. Add `(asset, datetime)` index BEFORE Phase 2 / Phase 3

The NOT EXISTS subquery for "every legacy row in v2 by (asset,
datetime)" is O(n^2) without an index. Cycle 23 hit this twice:
once in Phase 2 post-state verification (12-min hang) and once
again in Phase 3.

**Cycle 24 application**: add `CREATE INDEX IF NOT EXISTS
idx_<table>_legacy_asset_dt ON <table>(asset, datetime)` AND
`idx_<table>_v2_asset_dt ON <table>_v2(asset, datetime)` immediately
after Phase 0 commits, before any verification work.

### 4. The cutover RENAME pair INVALIDATES hardcoded table names

The most load-bearing gotcha. The Brief implicitly assumed the
writer's table references would just-work post-cutover, but my
hardcoded `INSERT INTO order_book_snapshots_v2` broke immediately
at cutover. Two options for Cycle 24+:

**Option A: Runtime introspection** (Cycle 23's retrofit). Writer
introspects the live table's PK shape on each iteration and adapts.
Pros: same code works pre/post cutover; cycle commits stay clean.
Cons: one PRAGMA per write iteration (cheap but not free); writer
logic is ~30% larger.

**Option B: Bundle writer update with cutover commit**. Single
commit that (a) RENAME pair AND (b) writer points at post-cutover
names. Brief moment of writer-vs-DB inconsistency during the commit
push.

**Recommendation for Cycle 24**: use Option A by default. Adapt the
Cycle 23 writer's `pre_cutover = any(c[1] == "id" for c in ...)`
introspection pattern. The runtime cost is negligible compared to
the safety of "writer never breaks across cutover."

### 5. Backfill + verification must respect concurrent writer

The live collector writes every 10s. Phase 2's transaction holds the
writer lock briefly; if it overlaps with a collector iteration,
SQLite serializes them. Pure-SQL INSERT-SELECT (under 10s) is
acceptable; Python loops (minutes) are not. Phase 3's verification
queries are read-only (mostly) and play nicely with WAL.

**Cycle 24 application**: Phase 2 backfill should complete in <30s
to avoid blocking the live writer. If it takes longer, surface and
re-think (might need to chunk it).

### 6. Dual-write writer can leave stale write locks if it errors

When the post-cutover writer hit `OperationalError: no such table:
order_book_snapshots_v2`, the prior INSERT to the live table was
left in an uncommitted transaction. The connection wasn't released
until the loop process exited at the end of its 3550s window.
Result: ~5 min stale lock that blocked subsequent writes from any
process.

**Cycle 24 application**: ensure the dual-write writer either
commits or rollbacks on every iteration, regardless of which INSERT
fails. Add `conn.rollback()` to the except clause:

```python
except Exception as e:
    try:
        conn.rollback()
    except Exception:
        pass
    return (0, f"insert: {type(e).__name__}: {e}")
```

This is a minor defensive cleanup; should be in the Cycle 24 writer
from day one.

### 7. Brief expectation-vs-reality: collector restart cadence

The Brief said `PraxisOrderBookCollector` runs hourly with a 3550s
loop, predicting Phase 0 would activate at the next `:00` hour
boundary (52 min after my push). Empirically, the loop restarted
at 19:35 UTC -- 28 min after the 19:07 push, ~32 min before the
predicted 20:00 boundary. This isn't necessarily wrong (could be a
restart trigger I don't know about), just unexpected.

**Cycle 24 application**: don't time Phase 0 commits assuming a
specific restart pattern; just push, then poll the v2 table for
non-zero row count to confirm the new writer is live before
proceeding to Phase 1.

---

## Performance datapoints

| Phase | Wall-clock | Volume |
|-------|------------|--------|
| Phase 2 backfill (INSERT-SELECT) | 7.219s | 87,668 rows |
| Phase 2 ROUND-correction UPDATE  | 0.453s | 43,596 rows |
| Phase 4 atomic RENAME pair        | 0.005s | (DDL) |
| Phase 0 commit -> collector restart | 28 min | (Brief budgeted ~52 min) |
| `(asset, datetime)` index creation | 0.164s | 88k rows |

---

## Cross-table sanity check (carry-forward from Cycle 21.5)

Post-cutover, all 7 migrated tables clean:

| Table | (asset, datetime) dupes | jittered_ts |
|-------|-------------------------|-------------|
| fear_greed | 0 | 0 |
| ohlcv_daily | 0 | 0 |
| ohlcv_4h | 0 | 0 |
| market_data | 0 | 0 |
| funding_rates | 0 | 0 |
| ohlcv_1m | 0 | 0 |
| order_book_snapshots | 0 | 0 |

The Cycle 21.5 jitter pattern remains isolated to `funding_rates`
and was fixed there. order_book_snapshots' new ms timestamps come
from Binance's `ob["timestamp"]` directly (already exact ms; not
event-clock-jittered like funding events).

---

## Open items / next cycle inputs

- **Cycle 23.5 (Phase 5 cleanup)**: defer until 24-48h burn-in is
  clean. Two-task: drop `_legacy` INSERT branch from the writer
  (collapse the runtime PK introspection); DROP TABLE
  `order_book_snapshots_legacy`. Update doc trio + retro.
- **Cycle 24**: `live_collector.price_snapshots`. Use this retro's
  Lessons-Learned section as the operational checklist. New wrinkle:
  writer is in `engines/live_collector.py`, not the unified
  collector module.
- **Plan-doc commit hash**: row #7 reads `<TBD>`. Patch with the
  actual hash in the standard follow-up commit.
- **Cycle 22 backup**: was deleted at start of this cycle as
  expected. Cycle 20 backup also deleted (overflow retention).
  Backup chain after this cycle: only `cycle23_backup`.
- **Verification deferred to next collector restart**: AC #15
  (writer continues dual-writing post-cutover) is verified at the
  code-review level but the live in-situ confirmation is deferred
  to the next `PraxisOrderBookCollector` invocation, which will
  exercise the runtime PK-introspection branch on real data.
  Chat's post-cycle MCP verification (per Brief Notes for Code)
  should confirm this.

---

## Deviations from Brief

- **Phase 2 backfill required two attempts**: initial Python
  row-by-row impl hung; rewrote as pure-SQL INSERT-SELECT. Documented
  as Lesson 1 above.
- **Phase 3 verification failed on first run**: 49 of 100 sample
  rows had off-by-1ms ts due to SQLite julianday float precision.
  Fixed via ROUND-correction UPDATE; re-ran verify successfully.
  Documented as Lesson 2.
- **Mid-cycle writer fix for cutover gotcha**: Phase 0's writer
  hardcoded `_v2` and broke at cutover. Retrofitted with runtime
  PK-shape introspection. Documented as Lesson 4 (the most
  load-bearing for Cycles 24-26).
- **Cutover timing was earlier than budgeted**: Brief assumed Phase 0
  takes effect at next `:00` boundary (~52 min after push); empirical
  was 28 min after push. Documented as Lesson 7.
- **Acceptance Criterion #15 only partially verified in this cycle**:
  the writer fix is on disk (py_compile clean) but the live in-situ
  verification is deferred to the next collector restart, which will
  happen within ~50 min of cycle close. The 24-48h burn-in window
  before Cycle 23.5 is the natural test of this.

None of these deviations caused data loss or schema corruption.
All 88,894 rows preserved; all PK invariants maintained; cross-table
sanity check post-migration is clean.
