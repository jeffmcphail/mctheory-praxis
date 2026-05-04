# BRIEF: Cycle 22 -- ohlcv_1m Migration + intrabar_predictor Reader Fix

**Series:** praxis
**Cycle:** 22
**Mode:** B (Brief -> Code)
**Author:** Chat (praxis_main_current)
**Date:** 2026-05-03
**Predecessor:** Cycle 21.5 (`4862e77`) -- funding_rates writer alignment hotfix

---

## Context

Cycle 22 migrates `ohlcv_1m` to Rule 35: drop `id`, compound
`PRIMARY KEY (asset, timestamp)`, convert seconds -> ms, rewrite
`datetime` text from naive to ISO `+00:00`. ~530,117 rows expected
(BTC: 265,059 + ETH: 265,058; ~6.1 months of minute bars).

Pattern is simple stop-migrate-start. Binance API supports full
re-fetch (`collect_ohlcv_1m` already handles 180-day rolling
backfill, scheduled every 6 hours via `PraxisCrypto1mCollector`).
Migration script performance is the new variable -- 530k rows is
~250x larger than any prior cycle.

**Two non-trivial differences from prior cycles surfaced during
the pre-Brief audit, both flagged below:**

1. **Reader at `engines/intrabar_predictor.py:117` does timestamp
   arithmetic that is unit-sensitive.** Specifically:
   `bar_start = (ts // bar_seconds) * bar_seconds` where
   `bar_seconds = bar_minutes * 60`. This works only when `ts` is
   seconds. Post-migration, `ts` is ms, and the bar-grouping logic
   silently produces zero output for any `bar_minutes >= 1`. Detail
   in Task 4 below. **This is the writer-alignment-audit lesson from
   Cycle 21.5 made concrete.** Fix is a 1-line change to use
   `bar_seconds = bar_minutes * 60 * 1000`.

2. **Asymmetric row counts** -- ETH has 2 fewer rows than BTC at the
   start of coverage. BTC starts 2025-10-31 17:45:00 UTC; ETH starts
   17:47:00. Investigated and confirmed this is a pre-existing
   data-quality footnote (likely a 2-min lag in ETH's first
   collector backfill batch in October). NOT a migration concern;
   the migration preserves both assets' rows independently. Just
   call it out in the retro for completeness.

---

## Scope

### Task 1: Migrate `ohlcv_1m` schema to Rule 35

**Current schema** (from `engines/crypto_data_collector.py:79-87`):

```sql
CREATE TABLE ohlcv_1m (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,        -- UTC seconds
    datetime TEXT NOT NULL,            -- "YYYY-MM-DD HH:MM:SS" naive
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,
    UNIQUE(asset, timestamp)
)
```

**Target schema:**

```sql
CREATE TABLE ohlcv_1m (
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,        -- UTC milliseconds
    datetime TEXT NOT NULL,            -- "YYYY-MM-DDTHH:MM:SS+00:00" ISO+offset
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,
    PRIMARY KEY (asset, timestamp)
)
```

Changes: drop `id`, compound PK, seconds -> ms (multiply x 1000),
naive datetime -> ISO+offset.

**Migration script:** `scripts/migrations/cycle22_ohlcv_1m_to_v2.py`

Same idempotent recipe as Cycles 17/18/20/21:

1. Open `data/crypto_data.db` with explicit transaction control
   (Rule 34: fresh connection).
2. Confirm OLD schema; print "Already migrated" + exit cleanly if NEW.
3. Spot-grep `ohlcv_1m.id` foreign-key dependencies (expect zero;
   verify and document).
4. CREATE TABLE `ohlcv_1m_new` with target schema.
5. INSERT INTO `ohlcv_1m_new` SELECT
   - `asset`
   - `timestamp * 1000`
   - `strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch')`
     (defense-in-depth, not a copy of old `datetime` text; matches
     Cycle 20's pattern)
   - `open, high, low, close, volume`
   FROM `ohlcv_1m`.
6. **Performance check**: print elapsed wall-clock time for the
   INSERT-SELECT. Expect 5-30 seconds for 530k rows depending on
   I/O. If the migration takes >2 minutes, surface as concerning.
7. Verify pre/post row counts match (expect 530,117).
8. Spot-check 3 specific rows (latest BTC, latest ETH, oldest BTC):
   - `timestamp_new == timestamp_old * 1000` exactly
   - `datetime_new` is ISO `+00:00`
   - `datetime_new` parses to same UTC moment as `datetime_old`
     (modulo the timezone-naive-vs-aware difference)
   - OHLCV values byte-identical
9. DROP TABLE `ohlcv_1m`.
10. ALTER TABLE `ohlcv_1m_new` RENAME TO `ohlcv_1m`.
11. Print before/after row counts + latest timestamps for retro.

**Backup**: copy `data/crypto_data.db` to
`data/crypto_data.db.cycle22_backup` BEFORE migration. md5-verify
against source. Cycle 21 backup can be deleted at the start of this
cycle (per retention rule). Cycle 21.5 produced no backup, so the
chain is `cycle21_backup` -> `cycle22_backup`.

### Task 2: Update the `collect_ohlcv_1m` writer

`engines/crypto_data_collector.py` `collect_ohlcv_1m()` at line 319.
Two changes:

1. **`init_db()` schema** (lines 79-87): match new shape (no id,
   compound PK).

2. **The INSERT path** (line 371-375): update timestamp + datetime.
   Current:
   ```python
   ts = int(c[0] / 1000)
   dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
   ```
   to:
   ```python
   ts = int(c[0])               # API returns ms; store as ms
   dt = datetime.fromtimestamp(ts // 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
   ```

   Same pattern as Cycles 18/20/21.

**Writer alignment audit (per Cycle 21.5 lesson):** Binance's
`fetch_ohlcv` returns `c[0]` as ms with bar-aligned values
(`openTime` of each kline). Daily/4h/1m kline endpoints all
guarantee `.000` alignment. Verify by inspecting one fresh fetch
during the cycle (or by relying on the prior cycles' empirical
verification on `ohlcv_daily` and `ohlcv_4h`, which used the same
upstream contract). The funding_rates jitter pattern does NOT exist
on kline endpoints. Document this finding in the retro.

### Task 3: Update MCP tool `servers/praxis_mcp/tools/ohlcv.py`

**Reader audit result:** `ohlcv.py`'s `get_recent_ohlcv` reads
`SELECT ... FROM ohlcv_1m ORDER BY timestamp DESC LIMIT ?`. It does
NOT do arithmetic on `timestamp`; it just sorts and returns the
column to clients. **The query is reader-transparent across
seconds-vs-ms.** Returned `timestamp` field changes meaning (now ms
instead of sec) but the tool's docstring doesn't make a unit
guarantee.

**Required: update the tool's docstring** to specify ms units
post-migration. Add a brief note like:

```
Returns:
    Dict with asset, rows (list of bars with timestamp [UTC ms],
    open, high, low, close, volume, oldest-first), and count.
```

No code change to the tool's body. If clients of the MCP tool
(Atlas chat? other consumers?) were assuming seconds, they need to
adapt -- but this is genuinely a unit migration, not a bug.

### Task 4: Fix the `intrabar_predictor.py` reader

**This is the load-bearing reader fix that Cycle 21.5's
writer-alignment-audit lesson predicted would happen.**

`engines/intrabar_predictor.py:111-117`:

```python
bar_seconds = bar_minutes * 60
aggregated = []
current_group = []
current_bar_start = None

for r in filtered:
    ts = r["timestamp"]
    bar_start = (ts // bar_seconds) * bar_seconds
    ...
```

This computes a "bar bucket id" by integer-floor-dividing the
timestamp. Pre-migration, `ts` was UTC seconds and
`bar_seconds = bar_minutes * 60` (5-min bar = 300 seconds); the
bucketing aligned 1-min rows to 5-min boundaries.

Post-migration, `ts` is UTC ms but `bar_seconds` is still in
seconds-magnitude. **Result: every 1-min bar gets a unique bucket
id, the `current_group` never grows past length 1, the
`if len(current_group) == bar_minutes` check at line 125 never
passes, `aggregated` stays empty, and `intrabar_predictor` silently
returns zero N-min bars for any `bar_minutes >= 2`.**

(The `bar_minutes == 1` short-circuit path at line 108-109 is
unaffected because it returns `filtered` directly without
bucketing.)

**Fix**: change line 111 from:

```python
bar_seconds = bar_minutes * 60
```

to:

```python
# Operate in ms to match the post-Cycle-22 ohlcv_1m schema.
# Pre-Cycle-22 the timestamp column was UTC seconds; this code
# used `bar_minutes * 60`. Now ts is UTC ms.
bar_seconds = bar_minutes * 60 * 1000
```

Variable name `bar_seconds` becomes a slight misnomer post-fix
(it's now bar-ms), but renaming is cosmetic and risks producing a
larger diff. Leave the name; rely on the comment.

**Verification of fix**: after the migration AND the fix land, run
a spot test importing `intrabar_predictor.load_ohlcv` (or whatever
the public entry-point is named -- inspect the file) with
`bar_minutes=5`, `asset='BTC'`. Confirm the function returns a
non-zero number of 5-min bars. Pre-fix this would return [].

### Task 5: Cross-engine spot-grep for raw-SQL `ohlcv_1m` readers

Per the standard cycle template (cycles 21/21.5 retro lesson),
verify no other engines have raw `WHERE timestamp > <constant>`
clauses against `ohlcv_1m` with hardcoded seconds-since-epoch
values that would break post-migration.

Grep patterns to run:
- `WHERE timestamp` AND `ohlcv_1m` (in same file)
- `ts >` or `ts <` AND `ohlcv_1m` references
- Direct numeric comparisons where the constant could be
  interpreted as Unix seconds (e.g., `> 1700000000`)

Document findings in retro. If nothing concerning is found, say so.
If a hardcoded-seconds clause is found anywhere besides
`intrabar_predictor.py` (already addressed in Task 4), surface
BEFORE migrating data.

### Task 6: MCP `get_collector_health` verification

`servers/praxis_mcp/tools/meta.py` `primary_monitored` already
includes `ohlcv_1m` (Cycle 11). The autodetect heuristic
(`> 1e12 -> ms`) handles the migration transparently -- no code
change needed.

Verify post-migration that `get_collector_health` reports
`ohlcv_1m` correctly: `is_stale=false` (the table is actively
written; staleness should be small fraction of the 7h threshold),
`row_count` matches post-migration count.

### Task 7: Update doc trio

**`docs/SCHEMA_NOTES.md`**: per-table prose for `ohlcv_1m`:
NONCONFORMING -> CONFORMING (Cycle 22). Update column type/format
notes. Migration status table: change row to CONFORMING / 22.
Add a note about the `intrabar_predictor.py` reader fix
(Task 4) -- this is the second non-cosmetic reader change in the
migration program (after Cycle 19's `funding.py` comment-only
update, this is a real logic change, the first such case).

**`docs/SCHEMA_MIGRATION_PLAN.md`**:
- Status summary row #6: change to
  `ohlcv_1m | simple | DONE | <commit-hash>`
- Per-table spec section: add reader-fix note describing the
  `intrabar_predictor.py:111` change (`bar_seconds = bar_minutes
  * 60 -> bar_minutes * 60 * 1000`); document this as the first
  case where the migration required a non-cosmetic reader change.

**`claude/TODO.md`**:
- Mark Cycle 22 as DONE in Recently closed (with commit hash).
- Add: "Cycle 23: Migrate `order_book_snapshots` per
  `docs/SCHEMA_MIGRATION_PLAN.md` row #7. **Dual-write pilot
  cycle.** ~70k rows, growing 5/min via 60s-cadence collector.
  First migration that requires Phase 0-5 dual-write recipe per
  Rule 35.6."

### Task 8: Retro

`claude/retros/RETRO_ohlcv_1m_migration.md` with:

- Pre/post row counts (530,117 -> 530,117 expected)
- Per-asset breakdown (BTC: 265,059, ETH: 265,058)
- Migration script wall-clock time (performance datapoint)
- New schema (PRAGMA table_info)
- 3-row spot-check before/after
- `ohlcv.py` MCP docstring update verification
- **`intrabar_predictor.py` Task 4 fix verification** with empirical
  spot-test demonstrating that `bar_minutes=5` now returns non-zero
  bars (and showing what it returned pre-fix would have been [])
- Cross-engine SQL audit results (Task 5)
- MCP `get_collector_health` output for `ohlcv_1m` post-migration
- ETH vs BTC asymmetry footnote (2-row offset, pre-existing)
- Writer alignment audit result (kline `openTime` is `.000`-aligned
  by Binance contract, no jitter expected; verify against fresh
  fetch if convenient)
- Any deviations from this Brief

---

## Out of scope

- Migrating any table other than `ohlcv_1m`
- Touching `_to_latest_ms` autodetect heuristic (Cycle 27)
- Renaming `bar_seconds` variable in `intrabar_predictor.py`
- Investigating the BTC/ETH 2-row asymmetry (data-quality
  footnote; pre-existing; out of scope)
- Re-running `phase3_models.joblib` retrain
- Adding new MCP tools or new monitoring entries

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | `data/crypto_data.db.cycle22_backup` created BEFORE migration runs (md5-verified) |
| 2 | `data/crypto_data.db.cycle21_backup` deleted at start of cycle (per retention rule) |
| 3 | `scripts/migrations/cycle22_ohlcv_1m_to_v2.py` exists, idempotent (re-run exits 0 cleanly) |
| 4 | Pre/post migration row counts match exactly (530,117 +/- any new rows from collector firing during cycle) |
| 5 | Pre/post latest UTC moments match (delta = 0s) |
| 6 | New schema: compound PK `(asset, timestamp)`, no `id`, ms timestamps, ISO+00:00 datetime |
| 7 | Spot-check 3 rows shows `timestamp_new = timestamp_old * 1000` exactly |
| 8 | Migration script wall-clock time recorded in retro (performance datapoint) |
| 9 | `engines/crypto_data_collector.py` `init_db` + `collect_ohlcv_1m` writer updated |
| 10 | `servers/praxis_mcp/tools/ohlcv.py` docstring updated to specify ms units |
| 11 | `engines/intrabar_predictor.py:111` updated: `bar_seconds = bar_minutes * 60 * 1000` with explanatory comment |
| 12 | Empirical spot-test of intrabar_predictor at `bar_minutes=5` returns non-zero bars post-migration AND post-fix |
| 13 | Cross-engine SQL spot-grep performed; results documented in retro |
| 14 | MCP `get_collector_health` reports `ohlcv_1m` correctly post-migration (autodetect handles ms timestamps; is_stale false; row_count matches) |
| 15 | MCP `get_recent_ohlcv(asset='BTC', lookback_bars=10)` returns 10 rows post-migration with timestamp values in ms |
| 16 | `docs/SCHEMA_NOTES.md` updated (per-table prose + migration status table + intrabar reader-fix note) |
| 17 | `docs/SCHEMA_MIGRATION_PLAN.md` updated (status row + per-table spec + reader-fix note) |
| 18 | `claude/TODO.md` updated (close Cycle 22, add Cycle 23 dual-write pilot) |
| 19 | All committable files ASCII-only (Rule 20) |
| 20 | Two-commit pattern (main + hash patch) per Cycle 18/20/21 precedent OR amend-before-push -- Code's choice, document in retro |

---

## Notes for Code

- **Rule 34** throughout: fresh connection per logical pass.
- **Performance**: 530k rows is the largest migration so far. The
  INSERT-SELECT inside a single transaction is the right approach
  (atomic, fast). If wall-clock time exceeds 2 minutes, surface
  before declaring the cycle done -- something is wrong (e.g.,
  index-contention, WAL sync issue).
- **Writer alignment** (per Cycle 21.5 lesson): kline `openTime`
  values from Binance are bar-aligned by contract (verified during
  Cycle 18/20 via empirical inspection). The `funding_rates` jitter
  pattern does not apply here. Document in retro.
- **The intrabar_predictor.py fix is mandatory, not optional.**
  Without it, the migration ships a silent semantic break. Verify
  the fix empirically by importing the module and calling its
  loader function with `bar_minutes=5` post-fix. If the function
  signature has changed since the audit-time grep, surface in the
  retro before declaring the cycle done.
- **The MCP `get_recent_ohlcv` docstring update**: keep the change
  minimal -- just clarify the `timestamp` unit. Don't rewrite the
  whole docstring.
- **Cross-engine SQL audit**: if you find raw-SQL readers beyond
  the two already known (`intrabar_predictor.py` and `ohlcv.py`),
  STOP and surface to chat before touching schema. The pre-Brief
  audit found exactly 2 readers; if more exist now, the codebase
  has drifted and the migration needs re-scoping.
- **Plan-doc commit hash**: insert `<TBD>` and follow up with
  hash-patch commit per Cycle 18/20/21 precedent. OR amend before
  push to keep single commit. Document choice in retro.
- **The retro must explicitly note the writer-alignment-audit
  result for kline `openTime`** (reader-transparent because
  bar-aligned by Binance contract). This is the durable "audit
  result" for this class of upstream API; future cycles dealing
  with kline endpoints can reference it.
