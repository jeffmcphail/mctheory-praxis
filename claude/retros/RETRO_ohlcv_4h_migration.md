# Retro: Cycle 20 -- ohlcv_4h Migration

**Brief:** none (Mode B-lite -- proceeded directly per `docs/SCHEMA_MIGRATION_PLAN.md` row #4)
**Date:** 2026-05-03
**Duration:** ~25 min (Mode B, surgical)
**Status:** COMPLETE
**Predecessors:** Cycle 19 (`7e73128`, `49b223b`) -- market_data migration

---

## Summary

Cycle 20 migrates `ohlcv_4h` (10,830 rows, 5,415 BTC + 5,415 ETH) to
Rule 35 using the simple stop-migrate-start pattern. Schema diff is
identical to Cycle 18's `ohlcv_daily` recipe, plus one extra step:
the `datetime` column was empirically naive (not `+00:00` as the plan
doc had claimed), so the migration re-derives it from `timestamp` via
SQLite `strftime` for defense in depth. Latest UTC delta 0 seconds;
all OHLCV values byte-identical; MCP autodetect confirmed working
across the now-mixed ms/seconds tables.

No external readers consume `ohlcv_4h`; the table feeds collector
internals only. The `lstm_predictor.py` reader hypothesized in the
plan was empirically only against `ohlcv_daily`. Plan doc updated to
reflect the corrected reader inventory.

---

## Changes Made

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `engines/crypto_data_collector.py` | `init_db` ohlcv_4h CREATE TABLE: `id` PK + `UNIQUE(asset, timestamp)` -> compound `PRIMARY KEY (asset, timestamp)`, no `id`. `collect_ohlcv_4h` INSERT: `ts = int(c[0])` (keep API ms; was dividing by 1000) + `dt = ...strftime('%Y-%m-%dT%H:%M:%S+00:00')` (ISO with offset; was naive `'%Y-%m-%d %H:%M:%S'`). | 65-73, 298-308 |
| `docs/SCHEMA_NOTES.md` | `ohlcv_4h` per-table prose: NONCONFORMING -> CONFORMING (Cycle 20). Migration status table row updated. | 115-124, 261 |
| `docs/SCHEMA_MIGRATION_PLAN.md` | Cycle 20 row in Status summary: pending -> DONE. Per-table spec #4 rewritten: corrected reader inventory (no external reader; `lstm_predictor.py` was wrong); corrected datetime claim (was naive, not `+00:00`). | 22, 95-114 |
| `claude/TODO.md` | Closed Cycle 20 entry; replaced with Cycle 21 (funding_rates migration); Cycle 19 -> closed; Cycle 20 -> closed in Recently closed. | 21-27, 218-237 |

### Files Created

| File | Purpose |
|------|---------|
| `scripts/migrations/cycle20_ohlcv_4h_to_v2.py` | Idempotent migration. Adapted from cycle18 with one addition: datetime re-derived from `timestamp` via `strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch')` rather than copying from old (naive) text. Cross-checks: row count (overall + per-asset), latest UTC delta, OHLCV value preservation, datetime ISO format + `+00:00` offset assertion. |
| `data/crypto_data.db.cycle20_backup` | Pre-migration full-DB backup (659,300,352 bytes; md5 matched source). Gitignored. Created BEFORE migration script ran. |
| `claude/retros/RETRO_ohlcv_4h_migration.md` | This file. |

### Backup retention

Per the plan-doc retention rule (delete after the cycle-after-next
commit proves stable), cycle 18 backup was eligible for deletion at
the start of this cycle but had already been removed in Cycle 19.
Cycle 19 backup is retained through Cycle 20's burn-in window;
deletable at the start of Cycle 21.

---

## Migration verification

### Pre/post snapshot

```
Pre-migration:
  rows: 10830 (BTC=5415, ETH=5415)
  ts_min: 1699732800    (seconds, 2023-11-11T20:00:00Z)
  ts_max: 1777694400    (seconds, 2026-05-02T04:00:00Z)
  latest row: ('BTC', 1777694400, '2026-05-02 04:00:00',
               78399.99, 78448.99, 78262.21, 78293.95, 259.67858)
  datetime format: NAIVE 'YYYY-MM-DD HH:MM:SS'

Post-migration:
  rows: 10830 (BTC=5415, ETH=5415)
  ts_min: 1699732800000 (milliseconds)
  ts_max: 1777694400000 (milliseconds)
  latest row: ('BTC', 1777694400000, '2026-05-02T04:00:00+00:00',
               78399.99, 78448.99, 78262.21, 78293.95, 259.67858)
  datetime format: ISO 'YYYY-MM-DDTHH:MM:SS+00:00'

Delta: 0.0 seconds; OHLCV values byte-identical.
```

### Schema (post-migration)

```
PRAGMA table_info(ohlcv_4h):
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

### Idempotency check

Re-run after successful migration:

```
[migrate] Already migrated -- ohlcv_4h has new schema. Exiting cleanly.
```

Exit 0.

### Datetime + ts range cross-check

```
BTC: 5415 rows, d_min='2023-11-11T20:00:00+00:00', d_max='2026-05-02T04:00:00+00:00'
ETH: 5415 rows, d_min='2023-11-11T20:00:00+00:00', d_max='2026-05-02T04:00:00+00:00'

Latest 4 rows (ts=ms, datetime=ISO+00:00):
  BTC ts=1777694400000 dt='2026-05-02T04:00:00+00:00' close=78293.95
  ETH ts=1777694400000 dt='2026-05-02T04:00:00+00:00' close=2302.79
  BTC ts=1777680000000 dt='2026-05-02T00:00:00+00:00' close=78400.0
  ETH ts=1777680000000 dt='2026-05-02T00:00:00+00:00' close=2304.44
```

Every datetime row carries the `+00:00` suffix. Asset / time pairing
preserved.

### MCP `get_collector_health` smoke test

```json
{
  "ohlcv_4h": {
    "row_count": 10830,
    "latest": "2026-05-02T04:00:00+00:00",
    "staleness_seconds": 86984.749,
    "threshold_seconds": 93600,
    "is_stale": false
  }
}
```

`unmonitored: []` -- every table in `crypto_data.db` is monitored.
The autodetect heuristic (`> 1e12 -> ms`) correctly handles the new
ms timestamps. Sanity-checked alongside the 8 other monitored tables;
all report correctly.

---

## Acceptance Criteria

(Per the plan-doc spec for table #4; this cycle had no formal Brief.)

| # | Criterion | Status |
|---|---|---|
| 1 | `data/crypto_data.db.cycle20_backup` created BEFORE migration runs | PASS (md5 verified) |
| 2 | `scripts/migrations/cycle20_ohlcv_4h_to_v2.py` exists, idempotent | PASS (re-run verified) |
| 3 | `ohlcv_4h` new schema: compound PK `(asset, timestamp)`, no id, ms timestamp | PASS |
| 4 | `engines/crypto_data_collector.py` `init_db` schema matches new shape | PASS |
| 5 | Writer keeps API ms (no /1000) and emits ISO `+00:00` datetime | PASS |
| 6 | Pre/post row count of `ohlcv_4h` is 10,830 (no rows lost) | PASS |
| 7 | Pre/post latest UTC moment delta < 1s | PASS (0.0s) |
| 8 | OHLCV values on latest row byte-identical | PASS |
| 9 | Datetime stored as ISO `+00:00` for every row | PASS (5,415 BTC + 5,415 ETH spot-checked at extremes) |
| 10 | MCP `get_collector_health` reports `ohlcv_4h` correctly | PASS (autodetect handles new ms) |
| 11 | `docs/SCHEMA_NOTES.md` marks `ohlcv_4h` CONFORMING | PASS |
| 12 | `docs/SCHEMA_MIGRATION_PLAN.md` Cycle 20 row + spec updated | PASS |
| 13 | `claude/TODO.md` updated (close Cycle 20, add Cycle 21) | PASS |
| 14 | All committable files ASCII-only | PASS |
| 15 | Retro at `claude/retros/RETRO_ohlcv_4h_migration.md` | THIS FILE |

---

## Debugging trail

Nothing went wrong end-to-end, but two corrections to the
plan-doc claim deserve flagging:

1. **Datetime format claim was wrong.** The plan doc (and likely the
   audit that produced it) said "datetime column already in `+00:00`
   ISO format; no semantic change." Empirically false: the column
   stored naive `'YYYY-MM-DD HH:MM:SS'`. Caught by the pre-migration
   inspection step. Migration handled the rewrite via
   `strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch')` --
   defense in depth (datetime derived from timestamp, not copied
   from the old text). Plan doc spec for row #4 updated to record
   the correction.

2. **Reader claim was wrong.** Plan said `engines/lstm_predictor.py`
   reads `ohlcv_4h`. Empirically: only `ohlcv_daily` is read there;
   `ohlcv_4h` has no external reader. Verified by:
   `grep -rn "ohlcv_4h\|ohlcv4h" engines/ src/` returns only
   `crypto_data_collector.py` matches. Plan doc updated.

Both corrections fed back into `docs/SCHEMA_MIGRATION_PLAN.md` row
#4 so future cycles inherit the corrected facts.

---

## Test results

- Migration script: 1 run + 1 idempotent re-run, both exit 0
- Schema cross-check: PRAGMA table_info confirms compound PK shape
- Row-count cross-check: 10830 -> 10830 (overall and per-asset BTC/ETH)
- Latest-UTC cross-check: 0.0s delta (4-hour bars are exact UTC
  multiples; conversion is loss-free)
- Datetime cross-check: derived ISO matches stored, `+00:00` present
- OHLCV cross-check: latest-row open/high/low/close/volume preserved
  byte-for-byte
- MCP smoke test: ohlcv_4h reports correctly via autodetect, all 9
  monitored tables in `crypto_data.db` healthy except `onchain_btc`
  (still no scheduled collector; intentional alarm)

---

## Open items / next cycle inputs

- **Cycle 21**: Migrate `funding_rates` (~2,200 rows; simple
  stop-migrate-start). Same shape as ohlcv_4h. Verify Cycle 14's 17h
  staleness threshold remains appropriate post-migration.
- **Plan-doc commit hash**: row #20 reads `<TBD>`. Patch with the
  actual commit hash in a follow-up commit, same as Cycles 18 + 19.
- **Carryover from Cycle 19**: `register_market_data_task.ps1` still
  needs an elevated PowerShell run from Jeff. Independent of Cycle 20;
  remains in Active TODOs.

---

## Deviations from plan-doc spec

- **Datetime field rewritten.** Plan said "no semantic change" for
  datetime; reality required a format rewrite. Justified +
  documented.
- **Reader inventory corrected.** Plan named `lstm_predictor.py`;
  empirical result: no external reader. Plan updated.
- **Mode**: this cycle ran without a formal Brief in `claude/handoffs/`.
  Worked directly from the migration plan + Cycle 18 template, on
  Jeff's verbal "proceed". Created no Brief artifact this cycle (the
  Brief/retro permanence rule applies to Briefs that exist; not
  having one is acceptable for mechanically-derivative cycles like
  this).
