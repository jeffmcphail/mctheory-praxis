# BRIEF: Cycle 24 -- live_collector.price_snapshots Dual-Write Migration

**Series:** praxis
**Cycle:** 24
**Mode:** B (Brief -> Code)
**Author:** Chat (praxis_main_current)
**Date:** 2026-05-05
**Predecessor:** Cycle 23 (`ca5c719`, `10724bc`, `5cf1c03`) -- order_book_snapshots dual-write pilot

---

## Context

Cycle 24 migrates `live_collector.price_snapshots` to Rule 35 using the
dual-write recipe established in Cycle 23 (Phase 0-4; Phase 5 deferred
to Cycle 24.5). Recipe is now codified in
`docs/SCHEMA_MIGRATION_PLAN.md` "Dual-write recipe" section -- Cycle 24
is the second use, validating the recipe as durable for Cycles 25-26.

**Three differences from Cycle 23 worth surfacing up front:**

### Difference 1: No datetime column to recover precision from

`order_book_snapshots` had a `datetime TEXT` column with microsecond
precision even when the writer was downgrading the integer `timestamp`
to seconds. Cycle 23's migration was able to RECOVER ms precision by
parsing `datetime`.

`price_snapshots` has **no datetime column at all** -- only
`timestamp INTEGER` storing `int(time.time())` in seconds. There's no
sub-second precision in the legacy data to recover. **The migration
is a clean `legacy_ts * 1000` multiply** (no julianday/ROUND mess).
The new writer will write `int(time.time() * 1000)` going forward;
sub-second precision is gained for new rows only.

The target schema also **adds a `datetime TEXT` column** derived at
write time from the timestamp, matching every other Rule-35-conformant
table in the program. Migration backfills `datetime` via SQLite's
`strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch')` which
works deterministically off integer seconds.

### Difference 2: An in-process reader call requires atomic same-commit fix

`live_collector.py` `take_price_snapshot()` (line ~318) inserts a
row, then immediately calls `check_for_spikes(conn, slug, ...)` for
that slug. `check_for_spikes` (line 264) does:

```python
now = int(time.time())                       # seconds
window_start = now - (window_mins * 60)      # seconds
prices = conn.execute("""
    SELECT timestamp, yes_mid FROM price_snapshots
    WHERE slug=? AND timestamp >= ?            <-- compared against `window_start`
    ORDER BY timestamp ASC
""", (slug, window_start)).fetchall()
```

After the migration, `price_snapshots.timestamp` is ms, but
`window_start` is seconds. The `WHERE timestamp >= window_start`
clause becomes trivially true (column is always huge), and
`check_for_spikes` reads the entire history for that slug instead of
just the recent window.

**The reader fix and the writer change MUST land in the same Phase 0
commit.** Unlike Cycle 22's `intrabar_predictor` (offline analytics
reader, could be fixed any time), this is an in-process reader that
fires on every collector cycle. If we ship a writer change that produces
ms timestamps without simultaneously fixing this reader, **every
collector cycle returns wrong spike-detection results until the
reader is patched.**

Code MUST update `now`/`window_start` to ms units in the same Phase 0
commit:
```python
now = int(time.time() * 1000)
window_start = now - (window_mins * 60 * 1000)
```

### Difference 3: External reader fix in mev_executor.py

`engines/mev_executor.py:207-208` has the same pattern:
```python
now = int(time.time())                       # seconds
window_start = now - (window_mins * 60)      # seconds
```
Then queries `live_collector.price_snapshots` via
`WHERE slug=? AND timestamp >= ?`. Same break post-migration.

Plus stats-display reads in `live_collector.py:478-486`:
```python
first_dt = datetime.fromtimestamp(first, tz=timezone.utc)
```
This becomes wildly wrong post-migration if `first` is now ms (treats
ms as seconds, produces dates in the year ~58000+).

These are not in-process reader paths (mev_executor runs as its own
process; the stats display runs only via CLI), so they have somewhat
more flexibility on commit timing. But for safety, fix all three in
the same Phase 0 commit.

---

## Sequencing strategy: bundled writer + cutover OR runtime introspection?

Cycle 23's writer used **runtime PK-shape introspection** to adapt
across the cutover RENAME pair. The recipe section of
`SCHEMA_MIGRATION_PLAN.md` flags this as one of two options and
suggests Cycles 24-26 consider the **bundled writer-update + cutover
commit** approach as cleaner.

**Recommendation for Cycle 24: use the bundled approach.** Concretely:

- Phase 0 commit: ONLY the dual-write writer + reader fixes. The
  writer's INSERT statements hardcode `INTO price_snapshots` (legacy)
  and `INTO price_snapshots_v2` (new). No introspection needed because
  no cutover has happened yet.
- After backfill + verify, the cutover commit ALSO updates the writer
  in the same atomic landing: rename in DB + change writer's
  references from `price_snapshots_v2` to `price_snapshots` (new live)
  and from `price_snapshots` to `price_snapshots_legacy` (renamed
  legacy). Ship this as a SINGLE commit so the writer's source-of-truth
  always matches the DB's table-name reality.

Tradeoff: there's a ~1-iteration window where the python process has
old code in memory while the DB has the new layout. The SQLite
OperationalError that Cycle 23 hit IS still possible here. Mitigation:
**use `INSERT OR IGNORE` and wrap the dual-INSERT in try/except so a
single failed iteration doesn't crash the loop**. The very next
iteration will reload from disk if the process restarts, OR... hm,
actually the live_collector loop runs a long-lived process; it does
NOT auto-restart on file change. Need to think about this more
carefully.

**Alternative: Pre-Phase-4 writer prep commit.** Land a writer
change that uses **runtime sqlite_master-based name resolution** but
does so cleanly: query `sqlite_master` once at process start, cache
the resolution. Then after cutover, the cached resolution is wrong
for one iteration (until the user kills/restarts the live_collector
process). This is the same trade Cycle 23 made with PK-shape
introspection, just expressed differently.

**Final recommendation: stick with Cycle 23's runtime introspection
approach for parity.** Cycle 24 is not the cycle to invent a new
sequencing pattern -- the recipe section will get more useful when
multiple cycles use the SAME approach and we can compare fatigue
points. Code's writer should introspect on each iteration (cheap:
one PRAGMA call against the live table) and pick the right table
names. Same as Cycle 23.

If runtime introspection feels heavy for an in-loop writer (it does),
Code can cache the resolution at process start and accept the
"first iteration after cutover may insert into the wrong table"
window. Be explicit in the retro about what was chosen.

---

## Empirical state at Brief-write time

- `live_collector.db` `price_snapshots`: 351,615 rows
- Cadence: ~50 rows/min (continuous polling of all tracked Polymarket
  markets every 60 seconds, ~50 markets active)
- Writer: `engines/live_collector.py` `take_price_snapshot()` at line
  ~315
- Schema: `id`, `slug`, `timestamp` (sec), `yes_mid`, `yes_bid`,
  `yes_ask`, `spread`, `UNIQUE(slug, timestamp)`. NO datetime column.
- Scheduled task: `PraxisLiveCollector` (continuous long-lived
  process, no hourly restart -- different from
  PraxisOrderBookCollector's hourly invocation pattern)

**The continuous-process pattern is meaningful**: the
`PraxisLiveCollector` task launches the python process once and it
runs indefinitely. There is NO natural "next process restart" boundary
for the writer code change to take effect. The new writer code only
takes effect when:

1. The python process is manually killed and restarted, OR
2. The process crashes for some other reason and Task Scheduler
   relaunches it

Code's Phase 0 commit needs to be paired with an explicit "kill and
restart the live_collector process" step. This is different from
Cycle 23 where the hourly task scheduler did the restart automatically.

---

## Scope (Cycle 24 = Phases 0-4 only; Phase 5 is Cycle 24.5)

### Task 1: Phase 0 -- dual-write writer + reader fixes (single commit)

**Step 1.1: Add `price_snapshots_v2` schema to `init_db()` in
`engines/live_collector.py` (line ~70).**

```sql
CREATE TABLE IF NOT EXISTS price_snapshots_v2 (
    slug TEXT NOT NULL,
    timestamp INTEGER NOT NULL,         -- UTC milliseconds
    datetime TEXT NOT NULL,             -- "YYYY-MM-DDTHH:MM:SS+00:00" ISO+offset
    yes_mid REAL,
    yes_bid REAL,
    yes_ask REAL,
    spread REAL,
    PRIMARY KEY (slug, timestamp)
)
```

Plus an index on `(slug, timestamp DESC)` for the in-process reader.

**Step 1.2: Modify `take_price_snapshot()` to dual-write.**

Current (line ~318):
```python
now = int(time.time())
# ...
conn.execute("""
    INSERT OR IGNORE INTO price_snapshots
    (slug, timestamp, yes_mid)
    VALUES (?, ?, ?)
""", (slug, now, mid))
```

New:
```python
now_sec = int(time.time())
now_ms = int(time.time() * 1000)   # NB: not now_sec * 1000 -- captures fresh sub-second
dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

# Runtime introspection: which schema is the live name?
pre_cutover = any(
    c[1] == "id" for c in conn.execute(
        "PRAGMA table_info(price_snapshots)"
    ).fetchall()
)
if pre_cutover:
    # Live = OLD seconds schema; _v2 = NEW ms schema
    conn.execute("""
        INSERT OR IGNORE INTO price_snapshots
        (slug, timestamp, yes_mid)
        VALUES (?, ?, ?)
    """, (slug, now_sec, mid))
    conn.execute("""
        INSERT OR IGNORE INTO price_snapshots_v2
        (slug, timestamp, datetime, yes_mid)
        VALUES (?, ?, ?, ?)
    """, (slug, now_ms, dt, mid))
else:
    # Live = NEW ms schema; _legacy = OLD seconds schema
    conn.execute("""
        INSERT OR IGNORE INTO price_snapshots
        (slug, timestamp, datetime, yes_mid)
        VALUES (?, ?, ?, ?)
    """, (slug, now_ms, dt, mid))
    conn.execute("""
        INSERT OR IGNORE INTO price_snapshots_legacy
        (slug, timestamp, yes_mid)
        VALUES (?, ?, ?)
    """, (slug, now_sec, mid))
```

Note: only `yes_mid` is being written today; the schema reserves
`yes_bid`, `yes_ask`, `spread` columns but the writer never populates
them. **DO NOT change this in Cycle 24**; it's pre-existing behavior
to preserve. (Surface as a "known incomplete writer, follow-up
deferred" in the retro.)

The introspection is a single PRAGMA call per snapshot; cost is
negligible compared to the network round-trip to Polymarket CLOB.

**Step 1.3: Fix `check_for_spikes()` in same file (line ~264).**

```python
# OLD:
now = int(time.time())
window_start = now - (window_mins * 60)

# NEW (post-Cycle-24-Phase-0):
now = int(time.time() * 1000)
window_start = now - (window_mins * 60 * 1000)
```

Variable name "window_start" still semantically correct; just shifted
to ms units. Same pattern as Cycle 22's `intrabar_predictor` fix.

**Step 1.4: Fix `mev_executor.py` reader (line 207-208).**

Same shift to ms units. Same one-line change pattern.

**Step 1.5: Fix `live_collector.py` stats display (lines 478-486).**

```python
# OLD:
first_dt = datetime.fromtimestamp(first, tz=timezone.utc)
last_dt = datetime.fromtimestamp(last, tz=timezone.utc)

# NEW: divide by 1000 to convert ms back to seconds for fromtimestamp
# (or use ms-aware utcfromtimestamp variants if preferred)
first_dt = datetime.fromtimestamp(first / 1000, tz=timezone.utc)
last_dt = datetime.fromtimestamp(last / 1000, tz=timezone.utc)
```

Also at line 481 there's a similar pattern to inspect. Audit the
whole stats-display block (lines ~460-510) for any other timestamp
arithmetic; document findings in retro.

**Step 1.6: Fix `live_collector.py` export-to-spike-db (lines 539-555)
if it does timestamp arithmetic.**

The export copies `ps.timestamp` to a different DB (`spike_db_path`).
That different DB's `price_history.timestamp` was historically seconds.
Two options:
(a) Convert ms back to seconds at export: `INSERT INTO price_history
... VALUES (?, ts // 1000, price)`. Keeps the spike DB's contract.
(b) Migrate the spike DB too. Out of scope for this cycle.

Recommendation: **Option (a) -- convert at export.** Add a comment
documenting that the spike_db is intentionally seconds for compatibility
with whatever else reads it. Add to TODO: audit spike_db readers and
decide whether to migrate it in a future cycle.

**Step 1.7: Verify py_compile clean across all 3 files modified.**

**Step 1.8: Backup `data/live_collector.db` to
`data/live_collector.db.cycle24_backup` BEFORE committing. md5-verify.
Cycle 23 backup of `crypto_data.db` can be deleted at start of this
cycle (per retention rule -- it's now 2 cycles behind).**

**Step 1.9: Phase 0 commit standalone. Push.**

**Step 1.10: KILL THE LIVE COLLECTOR PROCESS** so Task Scheduler
relaunches it with the new code. Verify the relaunched process picks
up dual-write by checking that `price_snapshots_v2` starts growing
within 2 minutes.

This step is non-negotiable -- the live_collector is a long-lived
process that does NOT pick up file changes. Document the kill
command in the retro.

```powershell
# To find:
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
    if ($cmd -like "*live_collector*") { Write-Output "PID=$($_.Id): $cmd" }
}
# Then Stop-Process -Id <PID> -Force
# Task Scheduler will relaunch on the next "PraxisLiveCollector" trigger
```

### Task 2: Phase 1 -- burn-in verification (>=60 minutes)

After kill+relaunch confirmed picking up new code:

1. Wait at least 60 minutes of dual-write operation.
2. Verify counts grow at the same rate: legacy and v2 should each
   gain ~50 rows/min (continuous polling cadence).
3. Spot-check 5 random rows in the dual-write window: each
   `(slug, ts_seconds)` pair in legacy maps to a
   `(slug, ts_seconds * 1000 + sub_second_ms)` pair in v2 with
   identical `yes_mid`.

NB: because the new writer captures fresh `time.time()` for both ts
values rather than dividing, legacy_ts and v2_ts will NOT have an
exact `legacy_ts * 1000 == v2_ts` relationship. They will differ by
a few ms (one execution of `time.time()` to the next). Verify by
checking they're within ~10ms of each other rather than equal.

If counts diverge or sample rows mismatch, **abort**.

### Task 3: Phase 2 -- backfill script

`scripts/migrations/cycle24_price_snapshots_backfill_v2.py`:

```sql
INSERT OR IGNORE INTO price_snapshots_v2 (
    slug, timestamp, datetime, yes_mid, yes_bid, yes_ask, spread
)
SELECT
    l.slug,
    l.timestamp * 1000 AS ts_ms,
    strftime('%Y-%m-%dT%H:%M:%S+00:00', l.timestamp, 'unixepoch') AS dt,
    l.yes_mid, l.yes_bid, l.yes_ask, l.spread
FROM price_snapshots l
WHERE NOT EXISTS (
    SELECT 1 FROM price_snapshots_v2 v
    WHERE v.slug = l.slug AND v.timestamp = l.timestamp * 1000
)
```

**Note**: this is much simpler than Cycle 23's backfill because
- No julianday/ROUND nonsense (timestamp is integer seconds, multiply
  is exact)
- No `(asset, datetime)` cross-keying needed -- can use
  `(slug, timestamp * 1000)` directly because timestamps are exact
- No off-by-1ms problem possible

**Critical pre-step (per Cycle 23 lesson)**: add `(slug, timestamp)`
indexes on both tables BEFORE running backfill. The NOT EXISTS
subquery is O(n^2) without it. Confirm via `EXPLAIN QUERY PLAN`.

Idempotent: re-run on backfilled state inserts zero rows.

Performance expectation: 351k rows is ~4x larger than Cycle 23's 87k.
Expect ~30 seconds wall-clock for the INSERT-SELECT (pure-SQL).

### Task 4: Phase 3 -- verification script

`scripts/migrations/cycle24_price_snapshots_verify.py`:

1. `count_v2 >= count_legacy`
2. Every legacy `(slug, timestamp)` exists in `_v2` as
   `(slug, timestamp * 1000)` (no missing).
3. No duplicate `(slug, timestamp)` in `_v2` (no spurious).
4. Sample 100 random rows: every column except `id` (legacy only),
   `timestamp` (units differ by 1000), and `datetime` (only in v2)
   is byte-identical.

ABORT on any failure.

### Task 5: Phase 4 -- atomic cutover

`scripts/migrations/cycle24_price_snapshots_cutover.py`:

```sql
BEGIN;
ALTER TABLE price_snapshots RENAME TO price_snapshots_legacy;
ALTER TABLE price_snapshots_v2 RENAME TO price_snapshots;
COMMIT;
```

Idempotent (detect cut-over state via PRAGMA + sqlite_master).

After cutover, the runtime-introspection writer (from Phase 0)
automatically routes correctly.

**Verify post-cutover**:
- PRAGMA table_info(price_snapshots) shows new schema (no id,
  compound PK, datetime present)
- Next collector iteration successfully INSERTs into both
  `price_snapshots` (new live) and `price_snapshots_legacy` (renamed)
- `get_collector_health` reports `price_snapshots` is_stale=false
  (autodetect heuristic handles the unit change transparently)

### Task 6: Update doc trio

**`docs/SCHEMA_NOTES.md`**:
- `price_snapshots` per-table prose: NONCONFORMING -> CONFORMING
  (Cycle 24, dual-write).
- Migration status table: row -> CONFORMING (DONE-PARTIAL) / 24.
- Note about adding `datetime` column (didn't exist pre-cycle).

**`docs/SCHEMA_MIGRATION_PLAN.md`**:
- Status summary row #8: change to
  `live_collector.price_snapshots | dual-write | DONE-PARTIAL | <hash>`
- Per-table spec section: rewrite with actual numbers + lessons.
- Update "Dual-write recipe" section if Cycle 24 surfaces any new
  gotchas worth documenting (it shouldn't if everything goes per
  plan, but document any deviations).

**`claude/TODO.md`**:
- Mark Cycle 24 as DONE-PARTIAL in Recently closed (with hash).
- Add Cycle 24.5 (Phase 5 cleanup, after 24-48h burn-in).
- Add Cycle 25: `smart_money.position_snapshots`. Different sidecar
  DB; different writer file (`engines/smart_money.py`); known to
  have TWO writer sites at lines 371 and 703 (per Cycle 16 audit).
  No timestamp INTEGER column today (only TEXT ISO datetime), so
  this is a schema-shape change not just unit conversion.
- Note: `mev_executor.py` and `live_collector.py` both had reader
  fixes land in Cycle 24's Phase 0 commit. Document in TODO that
  these readers are now ms-aware.

### Task 7: Retro

`claude/retros/RETRO_price_snapshots_dual_write.md` with:

- The full Phase 0/1/2/3/4 execution log
- Performance datapoints (backfill wall-clock for 351k rows; cutover
  transaction time)
- The kill+relaunch step for the live_collector process (Code's
  observation of how long it took for new code to be live)
- Three reader fixes verified in-situ (check_for_spikes works
  post-cutover; mev_executor's window query returns the right rows;
  stats display shows correct dates)
- Cross-table sanity check (Cycle 21.5's defensive habit; verify
  `(slug, datetime)` duplicates are 0 in price_snapshots; spot-check
  other tables)
- **Compare Cycle 24 to Cycle 23**: what was easier (no
  julianday/ROUND, no MCP tool bugs), what was harder (no
  hourly-restart boundary; long-lived process)
- Note for Cycle 25-26: Cycle 24's experience with the long-lived
  process pattern. Smart_money also runs as a long-lived process?
  Trades runs as a... let me note that this is worth investigating
  before Cycle 25's Brief.

---

## Out of scope

- Phase 5 cleanup (drop `_legacy`, single-write collapse) -- deferred
  to Cycle 24.5 after 24-48h burn-in
- Migrating any other table
- Auditing/migrating `spike_db` (different DB, follow-up TODO)
- Auditing/migrating `data/market_prices.db` price_snapshots
  (sentiment_tracker.py uses it; different schema entirely)
- Auditing/migrating `data/mev_executor.db` price_snapshots
  (different schema)
- Adding `yes_bid`/`yes_ask`/`spread` to the writer (pre-existing
  incomplete writer, follow-up TODO)
- Touching `_to_latest_ms` autodetect heuristic (Cycle 27)

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | `data/live_collector.db.cycle24_backup` created BEFORE Phase 0 (md5-verified) |
| 2 | `data/crypto_data.db.cycle22_backup` deleted at start of cycle (Cycle 23's backup remains; this is the rolling 1-behind retention) |
| 3 | Phase 0: `price_snapshots_v2` table created in init_db() with target Rule 35 schema |
| 4 | Phase 0: writer modified to dual-write (with runtime PK introspection); py_compile clean |
| 5 | Phase 0: `check_for_spikes` reader fix applied (ms units); same commit as writer |
| 6 | Phase 0: `mev_executor.py` reader fix applied (ms units); same commit as writer |
| 7 | Phase 0: `live_collector.py` stats display fix applied; same commit as writer |
| 8 | Phase 0: `live_collector.py` export-to-spike-db converts ms back to sec for compatibility; comment documenting why |
| 9 | Phase 0 committed and pushed in a single commit BEFORE Phase 1 verification |
| 10 | live_collector process killed and relaunched after Phase 0 push; new code verified picking up via `price_snapshots_v2` row growth |
| 11 | Phase 1: at least 60 min of dual-write before Phase 2; counts grow at expected ~50/min cadence on both tables |
| 12 | Phase 1: 5 spot-check rows have matching slug, identical yes_mid, ts values within ~10ms of legacy*1000 (not exact equality) |
| 13 | Phase 2: `(slug, timestamp)` indexes added to both tables BEFORE backfill runs (Cycle 23 lesson) |
| 14 | Phase 2: backfill script exists at `scripts/migrations/cycle24_price_snapshots_backfill_v2.py`, idempotent |
| 15 | Phase 2: backfill completes in <2 min wall-clock; performance datapoint recorded in retro |
| 16 | Phase 2: every legacy row has matching `(slug, timestamp * 1000)` in v2 |
| 17 | Phase 3: verify script confirms zero missing rows + zero spurious dupes + zero sample mismatches |
| 18 | Phase 4: cutover script atomically renames in single transaction |
| 19 | Phase 4: post-cutover, PRAGMA confirms live table has new schema |
| 20 | Phase 4: post-cutover, `get_collector_health` reports `price_snapshots` is_stale=false |
| 21 | Phase 4: post-cutover, next collector iteration writes to BOTH new live (`price_snapshots`) and renamed `price_snapshots_legacy`; verify by waiting 2 min and checking row growth in both |
| 22 | `check_for_spikes` post-cutover: spike-detection works correctly (returns rows from the actual N-min window, not the entire history) |
| 23 | `mev_executor.py` post-cutover: spike-scan window query returns correct row counts |
| 24 | `docs/SCHEMA_NOTES.md` updated |
| 25 | `docs/SCHEMA_MIGRATION_PLAN.md` updated (status row + per-table spec; recipe section updated only if new gotchas surface) |
| 26 | `claude/TODO.md` updated (Cycle 24 DONE-PARTIAL, Cycle 24.5 + Cycle 25 added) |
| 27 | All committable files ASCII-only (Rule 20) |
| 28 | Cross-table sanity check (Cycle 21.5 defensive habit): `price_snapshots` shows 0 `(slug, timestamp)` duplicates post-migration; other migrated tables still clean |
| 29 | Retro at `claude/retros/RETRO_price_snapshots_dual_write.md` documents the actual phase sequence, the kill+relaunch step, lessons, and any deviations from this Brief |
| 30 | Cycle 24.5 (Phase 5 cleanup) explicitly NOT executed in this cycle |

---

## Notes for Code

- **The kill-and-relaunch step is mandatory** and asymmetric vs.
  Cycle 23. The PraxisLiveCollector task launches a long-lived python
  process; file changes don't auto-pick-up. Without an explicit
  process restart, Phase 0's writer change is dead code. Document
  the kill-and-relaunch in the retro with timestamps so future
  long-lived-process cycles can reference the actual elapsed time.

- **The reader fixes are atomic with the writer change**. Same commit.
  If you hit a snag mid-implementation and need to commit partial
  work, DON'T -- the partial state would have ms writes with
  seconds-aware reads, which silently breaks spike detection.

- **The pre-existing `yes_bid`/`yes_ask`/`spread` columns being
  unpopulated is NOT this cycle's problem to fix.** Note in retro;
  open a TODO for follow-up. Don't get distracted.

- **Cross-engine spot-grep for ANY other reader**: Cycle 24's audit
  surfaced 3 reader fix locations. Run a final grep before declaring
  done:
  ```
  grep -rnE "FROM price_snapshots|price_snapshots\." engines/ scripts/ servers/
  ```
  Confirm only `live_collector.py`, `mev_executor.py`, and the
  sentiment_tracker/mev_executor instances pointing at OTHER databases
  show up. If any new reader has appeared since the audit, surface
  before Phase 0 commit.

- **The sub-second precision is GAINED, not RECOVERED**, in this
  cycle. Pre-Cycle-24 data has no sub-second info. Don't try to
  invent precision that doesn't exist; the migration honestly stores
  legacy_ts * 1000 (`.000` aligned) and the new writer captures fresh
  ms via `int(time.time() * 1000)`. Mixing these is correct -- the
  former is a coarse historical representation, the latter is the
  new precision contract. Document explicitly in retro.

- **Three commits expected** (matching Cycle 23 pattern):
  1. Phase 0 (writer + reader fixes + table + Brief commit)
  2. Phases 2-4 (backfill + verify + cutover + docs + retro)
  3. Hash-patch follow-up

- **Two-PRAGMA-overhead writer**: introspecting on every iteration
  adds one PRAGMA call per snapshot. The collector polls ~50 markets
  per minute, so that's ~50 PRAGMA calls per minute -- negligible vs.
  the ~50 HTTP roundtrips to Polymarket CLOB. If perf matters at
  larger scales (more tracked markets), Cycle 24.5 cleanup could
  cache the resolution at process start. Don't pre-optimize.

- **Smart_money preview for Cycle 25**: Cycle 25's writer is in
  `engines/smart_money.py` with TWO writer sites (lines 371 and 703
  per Cycle 16 audit). It also has no INTEGER timestamp column today
  -- only TEXT ISO datetime. So Cycle 25 will be a SCHEMA-SHAPE
  change (add `timestamp INTEGER`, parse from existing `timestamp
  TEXT` ISO), not just a unit conversion. Worth Code thinking ahead
  about whether the Cycle 25 Brief will reuse the recipe cleanly or
  need adaptation. Out of scope for this cycle; just a heads-up.
