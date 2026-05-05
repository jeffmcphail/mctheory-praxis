# BRIEF: Cycle 25 -- smart_money.position_snapshots Dual-Write Migration

**Series:** praxis
**Cycle:** 25
**Mode:** B (Brief -> Code)
**Author:** Chat (praxis_main_current)
**Date:** 2026-05-05
**Predecessors:** Cycle 24 (`b8fa847`, `6ca1796`, `dbecb23`),
Cycle 24.1 retro (`5b742ba`)

---

## Context

Cycle 25 migrates `smart_money.position_snapshots` to Rule 35 using
the dual-write recipe established in Cycles 23 and 24 (Phase 0-4;
Phase 5 deferred to Cycle 25.5). This is the third use of the
recipe; it's also the first dual-write cycle with three
characteristic differences from the prior two:

### Difference 1: TEXT timestamp -> INTEGER ms (schema-shape change, not unit conversion)

Cycle 23 had `timestamp INTEGER` (seconds) and `datetime TEXT`
(microsecond ISO). Cycle 24 had `timestamp INTEGER` (seconds), no
datetime column. Cycle 25's source is fundamentally different:

```sql
CREATE TABLE position_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,        -- "20260505_082408" composite key
    timestamp TEXT NOT NULL,           -- "2026-05-05T08:24:08.314000+00:00" ISO
    wallet TEXT NOT NULL,
    market_slug TEXT,
    market_title TEXT,
    outcome TEXT,
    size REAL, avg_price REAL, current_price REAL,
    value_usd REAL, pnl_usd REAL,
    UNIQUE(snapshot_id, wallet, market_slug, outcome)
)
```

The `timestamp` column **has no INTEGER form today** -- it's a
microsecond-precise ISO string written by:

```python
# engines/smart_money.py:37  (cmd_snapshot)
# engines/smart_money.py:83  (cmd_loop)
snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
now = datetime.now(timezone.utc).isoformat()
```

The migration adds a NEW `timestamp INTEGER NOT NULL` column with
ms-since-epoch values, and **renames the existing TEXT column to
`datetime`** to match Rule 35's column-naming convention. The
target schema:

```sql
CREATE TABLE position_snapshots (                  -- target (Rule 35)
    snapshot_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,                    -- NEW: ms since epoch
    datetime TEXT NOT NULL,                        -- WAS named `timestamp`
    wallet TEXT NOT NULL,
    market_slug TEXT,
    market_title TEXT,
    outcome TEXT,
    size REAL, avg_price REAL, current_price REAL,
    value_usd REAL, pnl_usd REAL,
    PRIMARY KEY (snapshot_id, wallet, market_slug, outcome)
)
```

Note: the existing `UNIQUE(snapshot_id, wallet, market_slug,
outcome)` becomes the primary key; we drop the synthetic `id`
AUTOINCREMENT. **The PK is NOT `(snapshot_id, timestamp)`** because
within a single snapshot the timestamp is constant; the natural
key is `(snapshot_id, wallet, market_slug, outcome)`. This differs
from order_book_snapshots and price_snapshots and is correct for
this table's semantics.

### Difference 2: ZERO reader fixes required

Cross-engine grep across `engines/smart_money.py`,
`engines/smart_money_alerts.py`, `dashboards/`, `scripts/`, and
`servers/praxis_mcp/`: every reader of `position_snapshots` keys on
`snapshot_id` (a TEXT composite like `"20260505_082408"`), never
on `timestamp`. The DELETE retention clause uses `ORDER BY
snapshot_id DESC` -- works because snapshot_id is sortable as a
string.

The only `timestamp` reference in the smart_money codebase
(`smart_money_alerts.py:559` `ORDER BY st.timestamp DESC`) is
against a different table (`signal_trades.timestamp`), which is
out of scope.

**This is a huge simplification vs Cycle 24's 4 reader fixes**.
Cycle 25 has 0. The migration is purely a schema-shape change.

### Difference 3: Scheduled, NOT long-lived process

The `PraxisSmartMoney` task runs `smart_money_service.bat` which
invokes:
```
python -m engines.smart_money discover --category ALL
python -m engines.smart_money snapshot
```
Both processes exit when their work is done. Task Scheduler fires
the .bat every 6 hours. **No kill-and-relaunch step needed** --
same pattern as PraxisOrderBookCollector's hourly relaunch. The
next process invocation picks up the new code automatically.

This is asymmetric vs Cycle 24's PraxisLiveCollector (long-lived
process). Cycle 24.1's process note about hard-restart-vs-close-
and-reopen is **not relevant** for this cycle's writer; it IS
relevant for the MCP tool verification at Phase 4 (see Task 5).

### Bonus: two writer sites, but only one in production

- **Writer 1 (line 371)**: `cmd_snapshot()`. This is what the
  scheduled .bat invokes every 6h. Production path.
- **Writer 2 (line 703)**: `cmd_loop()`. A continuous-mode CLI
  for ad-hoc monitoring. Not in any scheduled task. Off-path but
  still needs the dual-write update for consistency.

Both writers compute `now` and `snapshot_id` identically:
```python
snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
now = datetime.now(timezone.utc).isoformat()
```

Both need the same dual-write treatment. Refactor candidate (DRY
the writer body into a helper) is **out of scope** for this cycle;
flag for Cycle 25.5 cleanup if Code wants.

---

## Empirical state at Brief-write time (13:56 UTC)

- `smart_money.db` `position_snapshots`: 61,935 rows
- Cadence: ~12 snapshots/day across all tracked wallets; each
  snapshot writes ~50-100 rows (per-wallet position list); recent
  cadence ~50 rows per 6h
- Latest row `datetime` (currently in TEXT `timestamp` column):
  `2026-05-05T08:24:08.314000+00:00`
- Next scheduled invocation: **14:24 UTC** (28 min from Brief-write
  time). Pattern is :24 of every 6th hour from process registration.

**Timing implication**: Code's Phase 0 commit must land BEFORE
14:24 UTC for the new dual-write writer to take effect on the next
cycle. If Phase 0 lands AFTER 14:24, dual-write doesn't start
until 20:24 UTC -- a 6-hour wait for the next burn-in window.

If Code can't land Phase 0 in time for 14:24, defer to a later
session (target the run before 20:24). Don't rush a cycle this
infrequent.

---

## Acceptance criterion changes per Cycle 24.1's process notes

Two new ACs in this Brief that weren't in Cycles 23/24 (per the
process notes captured in `RETRO_to_latest_ms_hotfix.md`):

- **AC for live-MCP exercise** (replaces Cycle 24's tautological
  AC #20): Chat exercises `get_collector_health` post-cutover and
  pastes the response. The relevant `position_snapshots` entry
  must report parseable ISO `latest`, `is_stale=false`,
  `staleness_seconds < 28800` (the 8h threshold), and
  `row_count > 0`. Primary-DB regression check: other monitored
  tables still report cleanly with no new `__error__` artifacts.

- **AC for hard-restart verification protocol**: After Phase 4
  commits the MCP `SIDECAR_DBS` config change (`"iso_text"` ->
  `"ms"`), Code instructs Jeff to perform a HARD restart of Claude
  Desktop (not just close-and-reopen). The verification protocol
  in Cycle 24.1's retro applies: end any python.exe MCP children
  via Task Manager OR verify their absence via `Get-Process python`
  before reopening Desktop. Chat's MCP exercise then runs against
  the freshly-spawned MCP children.

These ACs make the verification step a live observable, not a
self-claimed assertion.

---

## Scope (Cycle 25 = Phases 0-4 only; Phase 5 is Cycle 25.5)

### Task 1: Phase 0 -- dual-write writer + v2 schema (single commit)

**Step 1.1**: Add `position_snapshots_v2` to `init_db()` in
`engines/smart_money.py`:

```sql
CREATE TABLE IF NOT EXISTS position_snapshots_v2 (
    snapshot_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,            -- ms since epoch
    datetime TEXT NOT NULL,                -- ISO with +00:00, microsecond
    wallet TEXT NOT NULL,
    market_slug TEXT,
    market_title TEXT,
    outcome TEXT,
    size REAL,
    avg_price REAL,
    current_price REAL,
    value_usd REAL,
    pnl_usd REAL,
    PRIMARY KEY (snapshot_id, wallet, market_slug, outcome)
)
```

Plus an index on `(snapshot_id)` if not implicit from the PK
(verify with EXPLAIN QUERY PLAN; SQLite's compound PK should cover
it).

**Step 1.2**: Modify both writer sites (lines 371 and 703) to
dual-write with runtime PK introspection:

```python
# Both writer sites use this pattern:
now_iso = datetime.now(timezone.utc).isoformat()
now_ms = int(time.time() * 1000)

# Runtime introspection: which schema is the live name?
pre_cutover = any(
    c[1] == "id" for c in conn.execute(
        "PRAGMA table_info(position_snapshots)"
    ).fetchall()
)
if pre_cutover:
    # Live = OLD schema (id PK + timestamp TEXT); _v2 = NEW schema
    conn.execute("""
        INSERT OR REPLACE INTO position_snapshots
        (snapshot_id, timestamp, wallet, market_slug, market_title,
         outcome, size, avg_price, current_price, value_usd, pnl_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (snapshot_id, now_iso, address, slug, str(title)[:100],
          outcome, size, avg_price, cur_price, value, pnl))
    conn.execute("""
        INSERT OR REPLACE INTO position_snapshots_v2
        (snapshot_id, timestamp, datetime, wallet, market_slug,
         market_title, outcome, size, avg_price, current_price,
         value_usd, pnl_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (snapshot_id, now_ms, now_iso, address, slug,
          str(title)[:100], outcome, size, avg_price, cur_price,
          value, pnl))
else:
    # Live = NEW schema (compound PK + ms + datetime); _legacy = OLD
    conn.execute("""
        INSERT OR REPLACE INTO position_snapshots
        (snapshot_id, timestamp, datetime, wallet, market_slug,
         market_title, outcome, size, avg_price, current_price,
         value_usd, pnl_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (snapshot_id, now_ms, now_iso, address, slug,
          str(title)[:100], outcome, size, avg_price, cur_price,
          value, pnl))
    conn.execute("""
        INSERT OR REPLACE INTO position_snapshots_legacy
        (snapshot_id, timestamp, wallet, market_slug, market_title,
         outcome, size, avg_price, current_price, value_usd, pnl_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (snapshot_id, now_iso, address, slug, str(title)[:100],
          outcome, size, avg_price, cur_price, value, pnl))
```

Note: `INSERT OR REPLACE` (existing semantics), NOT `INSERT OR
IGNORE`. The original writer overwrites if the
`(snapshot_id, wallet, market_slug, outcome)` UNIQUE constraint is
hit; preserve that behavior.

DRY consideration: refactor candidate. Code may extract a helper
`_insert_position_pair(conn, pre_cutover, snapshot_id, now_iso,
now_ms, fields...)` to avoid duplication across two writer sites
each with two branches. Use judgment; small duplication is OK.

**Step 1.3**: py_compile clean.

**Step 1.4**: Backup `data/smart_money.db` to
`data/smart_money.db.cycle25_backup`. md5-verify.
Cycle 23 backup of `crypto_data.db` can be deleted now (it's
2 cycles behind; rolling 1-behind retention).

**Step 1.5**: Phase 0 commit + push standalone, BEFORE 14:24 UTC.

If Code lands Phase 0 after 14:24 UTC: **document the deferral in
the cycle's chat**, wait until after the next 14:24/20:24 invocation
to confirm dual-write started, then proceed. Don't run Phases 2-4
until at least one full 6h cycle has dual-written.

### Task 2: Phase 1 -- burn-in verification

After the next scheduled invocation (14:24 or 20:24 UTC):

1. Verify counts: legacy and v2 should each have gained the same
   ~50 rows from that single cycle.
2. Sample 3 rows from the new cycle: `(snapshot_id, wallet,
   market_slug, outcome)` matches between tables; `timestamp` in
   legacy is ISO TEXT, `timestamp` in v2 is INTEGER ms, `datetime`
   in v2 matches legacy `timestamp` exactly.
3. Verify `int(datetime.fromisoformat(legacy.timestamp).timestamp()
   * 1000)` is close to `v2.timestamp` (within a few ms; both
   should be the same wall-clock moment but writer captures
   `time.time() * 1000` as a separate call from `datetime.now()`).

If counts diverge or sample mismatches, **abort**.

**Note on burn-in window length**: Cycles 23 and 24 used 60-min
burn-in. Cycle 25's collector fires only every 6 hours; one full
cycle's worth of dual-write rows is the minimum we can verify
against. Wait for at least ONE post-Phase-0 scheduled invocation
to complete; two if you want to confirm idempotency (re-running
INSERT OR REPLACE within the same snapshot_id is benign because
of the UNIQUE constraint).

### Task 3: Phase 2 -- backfill script

`scripts/migrations/cycle25_position_snapshots_backfill_v2.py`:

```sql
-- Pre-step: add index on `timestamp` in legacy for the join
-- (Phase 2 lesson from Cycle 23: indexes matter)
CREATE INDEX IF NOT EXISTS idx_pos_legacy_ts
    ON position_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_pos_v2_pk
    ON position_snapshots_v2(snapshot_id, wallet, market_slug, outcome);

-- Backfill
INSERT OR IGNORE INTO position_snapshots_v2 (
    snapshot_id, timestamp, datetime, wallet, market_slug,
    market_title, outcome, size, avg_price, current_price,
    value_usd, pnl_usd
)
SELECT
    l.snapshot_id,
    CAST(ROUND((julianday(l.timestamp) - 2440587.5) * 86400000) AS INTEGER) AS ts_ms,
    l.timestamp AS dt,
    l.wallet, l.market_slug, l.market_title, l.outcome,
    l.size, l.avg_price, l.current_price, l.value_usd, l.pnl_usd
FROM position_snapshots l
WHERE NOT EXISTS (
    SELECT 1 FROM position_snapshots_v2 v
    WHERE v.snapshot_id = l.snapshot_id
      AND v.wallet = l.wallet
      AND COALESCE(v.market_slug, '') = COALESCE(l.market_slug, '')
      AND COALESCE(v.outcome, '') = COALESCE(l.outcome, '')
)
```

**Note: julianday + ROUND, per Cycle 23 lesson.** The ROUND is
required because `(julianday(dt) - 2440587.5) * 86400000` lands
~1 ULP below the integer for ~half of microsecond-precision
datetimes; `CAST AS INTEGER` truncates and produces off-by-1ms
errors. Same as Cycle 23's order_book_snapshots backfill.

**Note on COALESCE**: `market_slug` and `outcome` are nullable in
the schema. `NULL = NULL` is FALSE in SQL; we need `COALESCE(...,
'')` on both sides so NULL-NULL pairs match. Verify in Phase 3.

Idempotent: re-run on backfilled state inserts zero rows.

Performance expectation: 61,935 rows is smaller than Cycle 24's
358k. Should complete in well under 5 seconds. The
julianday/ROUND adds maybe 50% over a clean integer multiply.

### Task 4: Phase 3 -- verification script

`scripts/migrations/cycle25_position_snapshots_verify.py`:

1. `count_v2 >= count_legacy`.
2. Every legacy `(snapshot_id, wallet, market_slug, outcome)`
   exists in v2 (with COALESCE for nullables) -- zero missing.
3. Zero `(snapshot_id, wallet, market_slug, outcome)` duplicates
   in v2.
4. Sample 100 random rows: every column except `id` (legacy only),
   `timestamp` (units differ), and `datetime` (only in v2) is
   byte-identical. The `datetime` field in v2 should equal the
   legacy `timestamp` exactly (it's the same TEXT value, just
   renamed).
5. Sample 10 random rows: `int(datetime.fromisoformat(v2.datetime)
   .timestamp() * 1000) == v2.timestamp` (within 0 -- ROUND is
   exact for microsecond datetimes).

ABORT on any failure.

### Task 5: Phase 4 -- atomic cutover + MCP config flip

`scripts/migrations/cycle25_position_snapshots_cutover.py`:

```sql
BEGIN;
ALTER TABLE position_snapshots RENAME TO position_snapshots_legacy;
ALTER TABLE position_snapshots_v2 RENAME TO position_snapshots;
COMMIT;
```

Idempotent (detect cut-over state via PRAGMA + sqlite_master).

After cutover, in the same Phase 4 commit, update MCP
`SIDECAR_DBS` in `servers/praxis_mcp/server.py`:

```python
# Pre-cutover:
"position_snapshots": {
    "threshold_seconds": 28800,
    "timestamp_column": "timestamp",
    "timestamp_format": "iso_text",     # <-- WAS this
},
# Post-cutover:
"position_snapshots": {
    "threshold_seconds": 28800,
    "timestamp_column": "timestamp",
    "timestamp_format": "ms",           # <-- becomes this
},
```

Also update the schema comment block in the same file (the comment
block describing each sidecar table; per Cycle 24's analogous
update at L65-71).

**Verification (Code's responsibility before declaring Phase 4
done)**:

- Post-cutover, PRAGMA table_info(position_snapshots) shows new
  schema (no id, compound PK, INTEGER timestamp, TEXT datetime).
- Renamed `position_snapshots_legacy` retains old schema (id
  column present).
- `position_snapshots_v2` no longer exists.

**Verification (Chat's responsibility post-MCP-restart, per Cycle
24.1 process notes)**:

After Code commits + pushes Phases 2-4, **Code instructs Jeff to
HARD-restart Claude Desktop** (Task Manager / End Process on python.exe
MCP children, then reopen). Chat then exercises `get_collector_health`
and pastes the response.

PASS conditions:
- `databases.smart_money.tables.position_snapshots` reports:
  - `row_count > 0`
  - `latest` parseable ISO datetime in 2026-05-* (NOT year 58000)
  - `is_stale=false`
  - `staleness_seconds < 28800`
- `databases.smart_money.unmonitored` no longer contains
  `__error__` artifacts
- Primary-DB regression: `trades`, `order_book_snapshots`,
  `ohlcv_1m`, etc. still report correctly
- `live_collector.price_snapshots` (Cycle 24's table) still
  reports correctly

If any check fails: surface to chat with the new MCP response.
Do NOT iterate-and-commit defensively.

### Task 6: Update doc trio

**`docs/SCHEMA_NOTES.md`**:
- `position_snapshots` per-table prose: NONCONFORMING -> CONFORMING
  (Cycle 25, dual-write).
- Migration status table: row -> CONFORMING (DONE-PARTIAL) / 25.
- Note the schema-shape change (TEXT timestamp -> INTEGER + new
  datetime column).

**`docs/SCHEMA_MIGRATION_PLAN.md`**:
- Status summary row #9: change to
  `smart_money.position_snapshots | dual-write | DONE-PARTIAL | <hash>`
- Per-table spec section: rewrite with actual numbers + lessons.
- "Dual-write recipe" section: add a sub-bullet noting that
  Cycle 25 was the first cycle to migrate a TEXT-timestamp source
  (vs Cycle 23/24's INTEGER-seconds sources). The julianday-based
  Phase 2 backfill (already documented from Cycle 23) is the
  reference pattern.

**`claude/TODO.md`**:
- Mark Cycle 25 as DONE-PARTIAL in Recently closed (with hash).
- Add Cycle 25.5 (Phase 5 cleanup, after 24-48h burn-in).
- Add Cycle 26: `trades`. Largest table (~6.5M rows). Already
  near-conforming (`timestamp` is already INTEGER ms). Per
  `docs/SCHEMA_MIGRATION_PLAN.md` row #10. Will need investigation:
  is `trades` populated by a long-lived process or scheduled task?
  This affects the kill-and-relaunch decision per Cycle 24.1
  process notes.

### Task 7: Retro

`claude/retros/RETRO_position_snapshots_dual_write.md` with:

- Phase 0/1/2/3/4 execution log
- Performance datapoints (backfill wall-clock; cutover transaction
  time)
- Confirmation that ZERO reader fixes were needed (this is novel
  vs Cycles 23/24)
- Confirmation that the schedule pattern (6h scheduled task, no
  long-lived process) made Phase 0 timing simpler than Cycle 24
- Live-MCP exercise response paste (per AC #20-equivalent)
- Hard-restart protocol notes (reference Cycle 24.1)
- **Compare Cycle 25 to Cycles 23/24**: what was simpler (no
  reader fixes; scheduled task; smaller dataset), what was harder
  (TEXT-to-INTEGER schema shape; julianday/ROUND from Cycle 23
  applies; nullable column COALESCE in NOT EXISTS)
- Cross-table sanity check (Cycle 21.5 defensive habit)
- Notes for Cycle 26: trades is the last big migration

---

## Out of scope

- Phase 5 cleanup (drop `_legacy`, single-write collapse) --
  deferred to Cycle 25.5 after 24-48h burn-in
- Migrating any other table
- Refactoring the two writer sites into a shared helper
  (DRY candidate; flag for Cycle 25.5)
- Adding `position_snapshots_v2` indexes beyond the compound PK
  (PK should be sufficient for current readers; revisit if Phase 1
  burn-in shows query slowness)
- Touching `signal_trades` or `convergence_signals` tables in
  smart_money.db (other tables, separate scope)
- Touching `_to_latest_ms` autodetect (Cycle 27)

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | `data/smart_money.db.cycle25_backup` created BEFORE Phase 0 (md5-verified) |
| 2 | `data/crypto_data.db.cycle23_backup` deleted at start of cycle (rolling 1-behind retention) |
| 3 | Phase 0: `position_snapshots_v2` table created in `init_db()` with target Rule 35 schema (compound PK, INTEGER timestamp, TEXT datetime, no `id`) |
| 4 | Phase 0: BOTH writer sites (line 371 + 703) modified to dual-write with runtime PK introspection; py_compile clean |
| 5 | Phase 0: ZERO reader fixes shipped (cross-engine grep confirms no readers depend on `position_snapshots.timestamp` arithmetic; verified in retro) |
| 6 | Phase 0 commit + push standalone, IDEALLY before 14:24 UTC. If after, document the deferral and wait for the next scheduled invocation |
| 7 | Phase 1: at least ONE full scheduled invocation has completed dual-write before Phase 2; legacy and v2 row counts grew by the same delta |
| 8 | Phase 1: 3 spot-check rows have matching `(snapshot_id, wallet, market_slug, outcome)` keys, identical content, ts relationship verified |
| 9 | Phase 2: `(snapshot_id, wallet, market_slug, outcome)` indexes added to both tables BEFORE backfill (with COALESCE handling for nullables); pure-SQL INSERT-SELECT |
| 10 | Phase 2: backfill script idempotent (re-run inserts zero rows) |
| 11 | Phase 2: every legacy row has matching v2 row (COALESCE-aware match on the 4-tuple key); zero missing |
| 12 | Phase 3: verify script confirms zero missing + zero spurious dupes + zero sample mismatches; sample includes datetime-to-ms round-trip check |
| 13 | Phase 4: cutover script atomically renames in single transaction; idempotent |
| 14 | Phase 4: post-cutover, PRAGMA confirms live table has new schema (no `id`, compound PK, INTEGER timestamp + TEXT datetime) |
| 15 | Phase 4: MCP `SIDECAR_DBS` config updated `"iso_text"` -> `"ms"` in same commit; schema comment block updated |
| 16 | Phase 4: post-cutover, next scheduled collector iteration writes to BOTH new live and renamed `_legacy`; verify by waiting for the 6h scheduled run and checking row growth in both |
| 17 | **Live-MCP exercise (per Cycle 24.1 process note)**: Code instructs Jeff to HARD-restart Claude Desktop after Phases 2-4 commit. Chat exercises `get_collector_health` and pastes response. `databases.smart_money.tables.position_snapshots` reports `row_count > 0`, `latest` parseable as 2026-05-* ISO, `is_stale=false`, `staleness_seconds < 28800`. No `__error__` artifacts anywhere |
| 18 | Regression check (in same MCP exercise): primary-DB tables + `live_collector.price_snapshots` still report cleanly with no new errors |
| 19 | `docs/SCHEMA_NOTES.md` updated |
| 20 | `docs/SCHEMA_MIGRATION_PLAN.md` updated (status row + per-table spec; recipe section updated with TEXT-timestamp note) |
| 21 | `claude/TODO.md` updated (Cycle 25 DONE-PARTIAL, Cycle 25.5 + Cycle 26 added) |
| 22 | All committable files ASCII-only (Rule 20) |
| 23 | Cross-table sanity check: `position_snapshots` shows zero `(snapshot_id, wallet, market_slug, outcome)` duplicates post-migration; other migrated tables still clean |
| 24 | Retro at `claude/retros/RETRO_position_snapshots_dual_write.md` documents actual phase sequence, the ZERO-reader-fixes finding, schedule pattern compared to Cycles 23/24, MCP exercise response, and any deviations |
| 25 | Cycle 25.5 (Phase 5 cleanup) explicitly NOT executed in this cycle |

---

## Notes for Code

- **Phase 0 timing**: ideal commit-before-14:24-UTC window is
  tight (~28 min from Brief drop). If Code is still mid-Phase-0
  at 14:20, finish + commit + push without polishing; doc
  updates can come in the Phases 2-4 commit. If 14:24 has
  already passed, defer to before 20:24 UTC. **Don't rush a 6h
  cadence cycle.**

- **Two writer sites** are an unusual pattern. The reader logic
  isn't duplicated like this; it's specifically the writer's
  per-position INSERT loop that's repeated in cmd_snapshot vs
  cmd_loop. Refactor candidate noted; out of scope this cycle.

- **The TEXT->INTEGER schema-shape change is novel for the
  recipe.** All prior dual-writes had INTEGER source timestamps;
  this is the first TEXT source. The julianday/ROUND pattern
  Cycle 23 established for SQLite-side ms derivation IS the
  pattern here too -- the Phase 2 backfill SQL is essentially
  a copy of Cycle 23's, applied to a column that's a TEXT
  microsecond-ISO instead of being inferred from a separate
  datetime column.

- **The `datetime` column in v2 is identical to `timestamp` in
  legacy** -- it's the same TEXT value, just renamed. No
  conversion needed. Phase 2 is `l.timestamp AS dt` (rename) +
  julianday-derived integer ms.

- **The MCP `SIDECAR_DBS` flip from "iso_text" to "ms" is the
  Cycle-24-pattern equivalent, and the lesson from Cycle 24.1
  applies**: after the change, hard-restart Claude Desktop or
  the in-memory MCP subprocess will report stale config. Don't
  declare Task 5 done based on Code's reading of the diff;
  declare it done only after Chat pastes a clean MCP response.

- **No long-lived process to kill**. PraxisSmartMoney is a
  scheduled task that runs the .bat which exits when work is
  done. The next scheduled invocation picks up the new code
  automatically. This is asymmetric vs Cycle 24's
  PraxisLiveCollector and matches Cycle 23's
  PraxisOrderBookCollector pattern.

- **Phase 1 burn-in is 1-2 collector cycles, not 60 minutes**.
  The 6h cadence means one cycle is 6 hours of wall clock.
  Verifying after one cycle is sufficient; waiting for two would
  be belt-and-suspenders. Document elapsed time in retro.

- **COALESCE for nullable columns** in NOT EXISTS subqueries.
  `market_slug` and `outcome` are nullable. SQL `NULL = NULL` is
  FALSE; without COALESCE, every NULL-NULL pair would be
  considered "missing" and the backfill would insert spurious
  duplicates. Worth a Phase 3 sanity check that no row in the
  live data has NULL `market_slug` AND NULL `outcome`
  (probably rare but verify).

- **Three commits expected** matching Cycles 23/24:
  1. Phase 0 (writer + table + Brief)
  2. Phases 2-4 (backfill + verify + cutover + MCP config flip
     + docs + retro)
  3. Hash-patch follow-up

- **No reader fixes is the headline result.** Don't downplay it
  in the retro; it's the kind of cycle-by-cycle complexity
  reduction that means the recipe is working as a recipe (each
  cycle does less novel work than the prior). Document it
  prominently.

- **For Cycle 26 (trades)**: investigate whether the trades
  collector is long-lived or scheduled. If long-lived: the
  Cycle 24-style kill-and-relaunch step applies. If scheduled
  (via the PraxisTradesCollector task fired hourly per Cycle 9-10
  registration): Cycle 23/25-style auto-pickup applies. Worth
  determining before drafting Cycle 26's Brief.
