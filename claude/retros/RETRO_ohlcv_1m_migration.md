# Retro: Cycle 22 -- ohlcv_1m Migration + intrabar_predictor Reader Fix

**Brief:** `claude/handoffs/BRIEF_ohlcv_1m_migration.md`
**Date:** 2026-05-03
**Duration:** ~30 min (Mode B, surgical)
**Status:** COMPLETE
**Predecessor:** Cycle 21.5 (`4862e77`) -- funding_rates writer alignment hotfix

---

## Summary

Cycle 22 migrates `ohlcv_1m` (530,836 rows: BTC 265,419 + ETH
265,417) to Rule 35 using the simple stop-migrate-start pattern.
Schema diff identical to Cycles 18/20/21: drop `id` AUTOINCREMENT,
compound `PRIMARY KEY (asset, timestamp)`, timestamp seconds -> ms,
datetime rewritten naive `"YYYY-MM-DD HH:MM:SS"` -> ISO
`"YYYY-MM-DDTHH:MM:SS+00:00"`. Datetime re-derived from `timestamp`
via SQLite `strftime` (defense in depth, matching Cycles 20/21).

This is the largest migration in the program so far -- 250x larger
than any prior cycle. Performance datapoint: the entire INSERT-SELECT
completed in **0.567s wall-clock** for 530,836 rows (Brief had
budgeted 5-30s with 2 minutes as the concerning threshold). The
total transaction wall-clock including DROP+RENAME was 1.013s.
Single INSERT-SELECT inside a transaction is the right approach
even at this row count.

**The load-bearing change this cycle is a reader fix in
`engines/intrabar_predictor.py:110`** -- the first non-cosmetic
reader change in the migration program (Cycles 17-21 needed only
writer + comment-header updates). The `load_intrabar_data` function
computed bar-bucket ids via integer floor-division of the timestamp
by `bar_seconds = bar_minutes * 60`. Pre-Cycle-22 the timestamp
column was UTC seconds and this worked correctly. Post-migration
the timestamp is UTC ms but `bar_seconds` was still in
seconds-magnitude, so every 1-min row got a unique bucket id and
`bar_minutes >= 2` silently returned zero aggregated bars. Fixed:
`bar_seconds = bar_minutes * 60 * 1000`. Verified empirically
post-fix: `bar_minutes=5` returns 100 aggregated bars at exact 5-min
ms boundaries.

Cycle 21.5's writer-alignment-audit prescription predicted exactly
this class of issue would surface in subsequent cycles; this is the
first time it has, and the prescription caught it pre-merge via the
Brief-time reader audit rather than as a post-cycle hotfix.

---

## Changes Made

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `engines/crypto_data_collector.py` | `init_db` ohlcv_1m CREATE TABLE: `id` PK + `UNIQUE(asset, timestamp)` -> compound `PRIMARY KEY (asset, timestamp)`, no `id`. `collect_ohlcv_1m` INSERT: `ts = int(c[0] / 1000)` -> `ts = int(c[0])` (keep API ms; was dividing by 1000) + `dt = ...strftime('%Y-%m-%dT%H:%M:%S+00:00')` (ISO with offset; was naive `'%Y-%m-%d %H:%M:%S'`). | 76-84, 362-374 |
| `engines/intrabar_predictor.py` | `load_intrabar_data` line 110: `bar_seconds = bar_minutes * 60` -> `bar_seconds = bar_minutes * 60 * 1000`. Three-line explanatory comment added above explaining post-Cycle-22 ms semantics. Variable name `bar_seconds` retained (now slight misnomer, but renaming is cosmetic and risks producing a larger diff). | 107-114 |
| `servers/praxis_mcp/tools/ohlcv.py` | `get_recent_ohlcv` docstring updated: Returns: bullet now specifies `timestamp [UTC ms]`, `datetime [ISO +00:00]`, with a Cycle 22 attribution noting pre-Cycle-22 callers expecting seconds need to adapt. No body change. | 18-24 |
| `docs/SCHEMA_NOTES.md` | `ohlcv_1m` per-table prose: NONCONFORMING -> CONFORMING (Cycle 22). Added reader-fix paragraph for `intrabar_predictor.py`, ohlcv.py docstring note, performance datapoint, and writer-alignment-audit result for Binance kline endpoints. Migration status table row updated. | 113-141, 285 |
| `docs/SCHEMA_MIGRATION_PLAN.md` | Status summary row #6: pending -> DONE / `<TBD>`. Per-table spec #6 rewritten: full reader inventory (exactly 2 raw-SQL readers); reader fix description and verification; performance result (0.567s); durable writer-alignment result for Binance klines. | 24, 130-181 |
| `claude/TODO.md` | Replaced Cycle 22 active TODO with Cycle 23 (order_book_snapshots dual-write pilot). Added Cycle 22 entry to Recently closed above the Cycle 21.5 entry. | 21-26, 244-262 |

### Files Created

| File | Purpose |
|------|---------|
| `scripts/migrations/cycle22_ohlcv_1m_to_v2.py` | Idempotent migration. Adapted from cycle20 with three additions: (a) explicit wall-clock timing for the INSERT-SELECT step, (b) DROP+RENAME timing, (c) extra spot-checks on oldest BTC + oldest ETH rows in addition to the latest row. Cross-checks: row count (overall + per-asset), latest UTC delta, OHLCV value preservation, datetime ISO format + `+00:00` offset assertion, timestamp arithmetic (`new_ms == old_s * 1000`), spot-check on oldest rows for both assets. |
| `data/crypto_data.db.cycle22_backup` | Pre-migration full-DB backup (795,623,424 bytes; md5 matched live source `CF92E56BD559677C7147D6DE7F626DB6` via FileShare.ReadWrite read). Gitignored. Created BEFORE migration script ran. |
| `claude/retros/RETRO_ohlcv_1m_migration.md` | This file. |

### Backup retention

Per the plan-doc retention rule (delete after the cycle-after-next
commit proves stable), Cycle 21 backup was deleted at the start of
this cycle. Cycle 21.5 produced no backup (lossless dedup). Cycle 20
backup remains on disk (eligible for deletion at the start of
Cycle 23 once Cycle 22 burn-in is complete). Backup chain after
this cycle: `cycle20_backup` (retention overflow) +
`cycle22_backup` (current).

---

## Cross-engine SQL audit (Brief Task 5)

Brief STOP-condition: must find exactly 2 raw-SQL readers
(`intrabar_predictor.py` + `ohlcv.py`). Anything else means the
codebase has drifted and the migration needs re-scoping.

Grep across `engines/`, `scripts/`, `servers/`, `gui/` for
`FROM ohlcv_1m`, `WHERE ... ohlcv_1m`, `JOIN ohlcv_1m`:

| File | Pattern | Risk | Disposition |
|------|---------|------|-------------|
| `engines/intrabar_predictor.py:96` | `SELECT timestamp, datetime, open, high, low, close, volume FROM ohlcv_1m WHERE asset = ?` | Subsequent code at line 110 did unit-sensitive timestamp arithmetic. | **Fixed this cycle**: line 110 `bar_seconds *= 1000`. |
| `servers/praxis_mcp/tools/ohlcv.py:30` | `SELECT timestamp, datetime, open, high, low, close, volume FROM ohlcv_1m WHERE asset = ? ORDER BY timestamp DESC LIMIT ?` | Sorts and returns column without arithmetic; reader-transparent. | Docstring update only. |
| `engines/crypto_data_collector.py` | Writer (init_db + INSERT) | -- | Updated this cycle. |
| `servers/praxis_mcp/tools/meta.py:235` | Monitoring config (`primary_monitored["ohlcv_1m"] = 25200`); reads MAX(timestamp) via the autodetect helper | -- | Reader-transparent (autodetect handles ms). |

**Result:** exactly 2 raw-SQL readers as the Brief expected. No
codebase drift. Migration proceeded.

---

## Migration verification

### Pre/post snapshot

```
Pre-migration:
  rows: 530836 (BTC=265419, ETH=265417)
  ts_min: 1761932700    (seconds, 2025-10-31T17:45:00Z, BTC oldest)
  ts_max: 1777857780    (seconds, 2026-05-04T01:23:00Z)
  latest row: ('BTC', 1777857780, '2026-05-04 01:23:00',
               78459.28, 78459.29, 78441.73, 78441.74, 4.89773)
  oldest BTC: ('BTC', 1761932700, '2025-10-31 17:45:00',
               109284.47, 109373.86, 109265.29, 109373.85, 17.4033)
  oldest ETH: ('ETH', 1761932820, '2025-10-31 17:47:00',
               3823.36, 3825.56, 3821.92, 3823.1, 431.5505)
  datetime format: NAIVE 'YYYY-MM-DD HH:MM:SS'

Post-migration:
  rows: 530836 (BTC=265419, ETH=265417)
  ts_min: 1761932700000 (milliseconds)
  ts_max: 1777857780000 (milliseconds)
  latest row: ('BTC', 1777857780000, '2026-05-04T01:23:00+00:00',
               78459.28, 78459.29, 78441.73, 78441.74, 4.89773)
  oldest BTC: ('BTC', 1761932700000, '2025-10-31T17:45:00+00:00', ...)
  oldest ETH: ('ETH', 1761932820000, '2025-10-31T17:47:00+00:00', ...)
  datetime format: ISO 'YYYY-MM-DDTHH:MM:SS+00:00'

Delta: 0.0 seconds; OHLCV values byte-identical;
timestamp_post = timestamp_pre * 1000 exactly.
```

### Schema (post-migration)

```
PRAGMA table_info(ohlcv_1m):
  (0, 'asset',     'TEXT',    1, None, 1)   <- PK pos 1
  (1, 'timestamp', 'INTEGER', 1, None, 2)   <- PK pos 2
  (2, 'datetime',  'TEXT',    1, None, 0)
  (3, 'open',      'REAL',    0, None, 0)
  (4, 'high',      'REAL',    0, None, 0)
  (5, 'low',       'REAL',    0, None, 0)
  (6, 'close',     'REAL',    0, None, 0)
  (7, 'volume',    'REAL',    0, None, 0)
```

No `id`. Compound PK on `(asset, timestamp)` -- subsumes the old
`UNIQUE(asset, timestamp)` constraint.

### Migration script timing (Brief AC #8)

```
INSERT-SELECT wall-clock: 0.567 s
DROP+RENAME wall-clock:   0.039 s
Total transaction:        1.013 s
```

**Performance assessment**: Brief budgeted 5-30s with 2 minutes
as the concerning threshold. Actual was sub-second for INSERT-SELECT
on 530k rows. WAL-mode SQLite + single transaction + no secondary
indexes during the copy = very fast. No concern.

### Idempotent re-run

```
[migrate] Opening C:\...\data\crypto_data.db
[migrate] Already migrated -- ohlcv_1m has new schema. Exiting cleanly.
exit=0
```

### Spot-check 3 rows (Brief Task 1 step 8 / AC #7)

| asset | ts_pre (s)  | ts_post (ms)    | dt_pre              | dt_post                     | sample OHLCV |
|-------|-------------|-----------------|---------------------|-----------------------------|--------------|
| BTC   | 1777857780  | 1777857780000   | 2026-05-04 01:23:00 | 2026-05-04T01:23:00+00:00   | C=78441.74, V=4.89773 |
| ETH   | 1777857780  | 1777857780000   | 2026-05-04 01:23:00 | 2026-05-04T01:23:00+00:00   | C=2314.41, V=26.9026 |
| BTC   | 1761932700  | 1761932700000   | 2025-10-31 17:45:00 | 2025-10-31T17:45:00+00:00   | C=109373.85, V=17.4033 |

`ts_post == ts_pre * 1000` exactly; `dt_post` carries `+00:00`;
OHLCV values preserved.

### Writer alignment audit (Brief Task 2 / Cycle 21.5 lesson)

Confirmed empirically: all 530,836 rows have `timestamp % 1000 == 0`.
Binance `fetch_ohlcv` returns kline `openTime` values bar-aligned
by contract (no sub-second jitter), as verified during Cycles 18/20.
The `funding_rates` jitter pattern does NOT apply to kline endpoints.

**Durable audit result for the migration program**: Binance kline
endpoints (1m / 4h / 1d / etc., via `fetch_ohlcv`) are jitter-free
at the millisecond level. Only event-driven endpoints like
`fetch_funding_rate_history` carry reporting jitter that requires
writer-side truncation. Future cycles touching kline data can rely
on this finding.

### Reader fix verification (Brief AC #12)

Empirical post-fix test:

```
>>> from engines.intrabar_predictor import load_intrabar_data
>>> result = load_intrabar_data('BTC', bar_minutes=5, limit_bars=100)
  Aggregated 265419 1-min bars into 100 5-min bars for BTC (dropped 0 zero-vol)
  Range: 2026-05-03T17:00:00+00:00 to 2026-05-04T01:15:00+00:00
>>> len(result)
100
>>> [(r['timestamp'], r['datetime']) for r in result[:3]]
[(1777827600000, '2026-05-03T17:00:00+00:00'),
 (1777827900000, '2026-05-03T17:05:00+00:00'),
 (1777828200000, '2026-05-03T17:10:00+00:00')]
```

Bar timestamps are at exact 5-min ms boundaries (300,000 ms apart).
Pre-fix this would have returned `[]` because every 1-min row would
have computed a unique `bar_start` value (since `ts // 300` produces
ms-magnitude values, never matching any subsequent 1-min row's
bucket id).

`bar_minutes=1` was already tested implicitly by the existing
short-circuit path at line 107-108 (returns `filtered` directly
without bucketing) -- unaffected by the fix.

### MCP `get_collector_health` smoke test (Brief AC #14)

```json
{
  "ohlcv_1m": {
    "row_count": 530836,
    "latest": "2026-05-04T01:23:00+00:00",
    "staleness_seconds": 551.711,
    "threshold_seconds": 25200,
    "is_stale": false
  }
}
```

`row_count` matches post-migration count. `latest` reflects the
newest 1-min bar (~9 minutes stale at verification time, well
within the 7h `PraxisCrypto1mCollector` 6h-batch threshold).
`is_stale=false`. The autodetect heuristic (`> 1e12 -> ms`)
correctly identifies the new ms timestamps.

### MCP `get_recent_ohlcv` smoke test (Brief AC #15)

`get_recent_ohlcv(asset='BTC', lookback_bars=10)` returned exactly
10 rows, all in ms-magnitude (every `timestamp > 1e12`), all
seconds-aligned (every `timestamp % 1000 == 0`), datetime in ISO
`+00:00` format, oldest-first per the docstring.

```
ts=1777857240000  ts%1000=0  dt=2026-05-04T01:14:00+00:00  O=78461.61
ts=1777857300000  ts%1000=0  dt=2026-05-04T01:15:00+00:00  O=78456.26
ts=1777857360000  ts%1000=0  dt=2026-05-04T01:16:00+00:00  O=78489.95
...
ts=1777857780000  ts%1000=0  dt=2026-05-04T01:23:00+00:00  O=78459.28
```

### Cross-table sanity check (Cycle 21.5 lesson)

Re-ran the duplicate + jitter check across all 6 migrated tables
post-migration:

```
TABLE              dupes   jittered_ts
fear_greed         0       0
ohlcv_daily        0       0
ohlcv_4h           0       0
market_data        0       0
funding_rates      0       0
ohlcv_1m           0       0
```

All clean. The Cycle 21.5 jitter pattern is confined to
`funding_rates` and was fixed there; no other migrated table shows
the bug. Worth keeping this check in the standard cycle template.

---

## Acceptance Criteria

(Per the Brief.)

| # | Criterion | Status |
|---|---|---|
| 1 | `data/crypto_data.db.cycle22_backup` created BEFORE migration runs (md5-verified) | PASS (`CF92E56BD559677C7147D6DE7F626DB6`) |
| 2 | `data/crypto_data.db.cycle21_backup` deleted at start of cycle | PASS |
| 3 | `scripts/migrations/cycle22_ohlcv_1m_to_v2.py` exists, idempotent | PASS (re-run exit 0) |
| 4 | Pre/post migration row counts match exactly | PASS (530,836 -> 530,836; per-asset 265,419 / 265,417) |
| 5 | Pre/post latest UTC moments match (delta = 0s) | PASS (0.0s) |
| 6 | New schema: compound PK `(asset, timestamp)`, no `id`, ms timestamps, ISO+00:00 datetime | PASS |
| 7 | Spot-check 3 rows shows `timestamp_new = timestamp_old * 1000` exactly | PASS (3 rows verified at extremes) |
| 8 | Migration script wall-clock time recorded in retro | PASS (INSERT-SELECT 0.567s; total txn 1.013s) |
| 9 | `engines/crypto_data_collector.py` `init_db` + `collect_ohlcv_1m` writer updated | PASS |
| 10 | `servers/praxis_mcp/tools/ohlcv.py` docstring updated to specify ms units | PASS |
| 11 | `engines/intrabar_predictor.py:110` updated: `bar_seconds = bar_minutes * 60 * 1000` with explanatory comment | PASS |
| 12 | Empirical spot-test of intrabar_predictor at `bar_minutes=5` returns non-zero bars post-migration AND post-fix | PASS (returned 100 bars at exact 5-min ms boundaries) |
| 13 | Cross-engine SQL spot-grep performed; results documented in retro | PASS (table above; exactly 2 raw-SQL readers as Brief expected) |
| 14 | MCP `get_collector_health` reports `ohlcv_1m` correctly post-migration | PASS (autodetect ms; is_stale false; row_count matches) |
| 15 | MCP `get_recent_ohlcv(asset='BTC', lookback_bars=10)` returns 10 rows post-migration with timestamp values in ms | PASS |
| 16 | `docs/SCHEMA_NOTES.md` updated | PASS |
| 17 | `docs/SCHEMA_MIGRATION_PLAN.md` updated (status row + per-table spec + reader-fix note) | PASS |
| 18 | `claude/TODO.md` updated (close Cycle 22, add Cycle 23 dual-write pilot) | PASS |
| 19 | All committable files ASCII-only (Rule 20) | PASS |
| 20 | Two-commit pattern (main + hash patch) per Cycle 18/20/21 precedent | Two-commit pattern chosen (matches Cycles 18/19/20/21 precedent; plan-doc + retro carry `<TBD>` for the main commit, follow-up commit patches the hash) |

---

## ETH vs BTC asymmetry footnote (Brief Task 8)

Per Brief, this is a pre-existing data-quality footnote, not a
migration concern. ETH has 2 fewer rows than BTC at the start of
coverage:

```
oldest BTC: 2025-10-31 17:45:00 UTC (1761932700)
oldest ETH: 2025-10-31 17:47:00 UTC (1761932820)
```

Likely a 2-min lag in ETH's first collector backfill batch in
October 2025 -- the BTC fetch happened first, ETH started its
first batch 2 minutes later, and the natural Binance batch
boundary kept ETH 2 rows behind throughout the rolling backfill
window. Migration preserved both assets' rows independently
(per-asset row counts unchanged: BTC 265,419 / ETH 265,417). Out
of scope to investigate further.

---

## Test results

- Migration script: 1 run + 1 idempotent re-run, both exit 0
- Schema cross-check: PRAGMA table_info confirms compound PK shape
- Row-count cross-check: 530836 -> 530836 (overall and per-asset)
- Latest-UTC cross-check: 0.0s delta (1-min bars at exact UTC
  multiples; conversion is loss-free)
- Datetime cross-check: derived ISO matches stored, `+00:00` present
- OHLCV cross-check: latest-row open/high/low/close/volume preserved
  byte-for-byte; oldest BTC + oldest ETH spot-checks also passed
- Timestamp arithmetic cross-check: `new_ms == old_s * 1000` exactly
- Reader-fix empirical test: `bar_minutes=5` returns 100 aggregated
  bars at exact 5-min ms boundaries (300,000 ms apart)
- MCP smoke tests: `get_collector_health` reports ohlcv_1m correctly
  via autodetect; `get_recent_ohlcv(BTC, 10)` returns 10 ms-
  timestamped rows
- Cross-table sanity check: all 6 migrated tables show 0 dupes and
  0 jittered timestamps
- Performance: 530k-row INSERT-SELECT in 0.567s wall-clock; total
  transaction 1.013s

---

## Lessons applied (from prior cycles)

- **Writer alignment audit (Cycle 21.5)**: applied to `ohlcv_1m`
  upstream API. Result: Binance kline endpoints are jitter-free by
  contract; no writer-side truncation needed. Recorded as a durable
  finding for future cycles.
- **Reader audit before migration (recurring lesson)**: caught the
  `intrabar_predictor.py` bug pre-merge instead of as a post-cycle
  hotfix. The audit's specific question -- "does any reader do
  unit-sensitive arithmetic on the timestamp?" -- is the right
  question; this is the first cycle where the answer was YES.
- **Brief expectation-vs-reality (recurring lesson)**: row-count
  estimate in the Brief (530,117) was 719 rows lower than reality
  (530,836) because the collector ran during the gap between Brief
  authorship and Code execution. Acceptance criteria written in
  terms of "+/- any new rows from collector firing during cycle"
  absorbed this gracefully; future Briefs for actively-written
  tables should keep this approach.

---

## Open items / next cycle inputs

- **Cycle 23**: Migrate `order_book_snapshots` per
  `docs/SCHEMA_MIGRATION_PLAN.md` row #7. **Dual-write pilot
  cycle.** ~70k rows, growing 5/min via 60s-cadence collector.
  First migration that requires the Phase 0-5 dual-write recipe per
  Rule 35.6. Use this cycle to write up the dual-write pattern as
  a section in the migration plan doc once the pilot lands. Brief
  expected from Chat per the dual-Claude split.
- **Plan-doc commit hash**: row #22 reads `<TBD>`. Patch with the
  actual commit hash in a follow-up commit, same as Cycles
  18/19/20/21.
- **Cycle 20 backup**: still on disk (retention overflow); eligible
  for deletion at the start of Cycle 23.
- **Carryover from Cycle 19**: `register_market_data_task.ps1`
  still needs an elevated PowerShell run from Jeff. Independent of
  Cycle 22; remains in Active TODOs.

---

## Deviations from Brief

- **None.** All 20 acceptance criteria met. Recipe followed exactly.
- One observation: Brief said "line 111" for the `bar_seconds` fix;
  empirical line was 110 (the comment-block above the fix uses 4
  lines, pushing the assignment to 114 post-edit). This is a
  cosmetic line-number drift, not a deviation.
- The Brief's row-count estimate (530,117) was 719 rows lower than
  reality at execution time (530,836) because the collector ran
  between Brief authorship and Code execution. The AC was written
  to allow this; per-asset and per-row math all check out. Noted as
  expected behavior, not a deviation.
