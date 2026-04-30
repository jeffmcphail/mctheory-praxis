# Retro: Cycle 18 -- SCHEMA_MIGRATION_PLAN.md + ohlcv_daily Migration

**Brief:** `claude/handoffs/BRIEF_ohlcv_daily_migration.md`
**Date:** 2026-04-30
**Duration:** ~30 min (Mode B, surgical)
**Status:** COMPLETE
**Predecessor:** Cycle 17 (`a03fff6`) -- Rule 35 + fear_greed pilot

---

## Summary

Cycle 18 produces the durable schema migration roadmap and executes the
second table migration. `docs/SCHEMA_MIGRATION_PLAN.md` orders all 10
remaining nonconforming tables by complexity-and-risk, slotting the
dual-write pilot at #7 (`order_book_snapshots`) and committing to a
single autodetect-cleanup cycle (#27) once every monitored timestamp
column is ms. `ohlcv_daily` migrates as the second table using the
simple stop-migrate-start pattern: 1,802 rows preserved (901 BTC +
901 ETH), latest UTC delta 0 seconds, compound `(asset, timestamp)`
PK with `id` dropped. The reader (`engines/lstm_predictor.py:68`)
queries by `date` only and is unaffected. All 16 Brief acceptance
criteria pass.

---

## Changes Made

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `engines/crypto_data_collector.py` | `init_db` schema for ohlcv_daily: `id` PK + `UNIQUE(asset, timestamp)` -> compound `PRIMARY KEY (asset, timestamp)` (no id); `collect_ohlcv_daily` writer: `ts = int(c[0])` (keep API ms; was `int(c[0] / 1000)` storing seconds); `date` derivation now divides ms by 1000 inside the strftime call | 54-63, 248-260 |
| `claude/TODO.md` | Replaced "Cycle 18: Write SCHEMA_MIGRATION_PLAN.md" entry with "Cycle 19: Migrate next table" pointer; moved Cycle 17 to closed; added Cycle 18 closure entry | 21-26, 209-225 |

### Files Created

| File | Purpose |
|------|---------|
| `docs/SCHEMA_MIGRATION_PLAN.md` | Ordered roadmap for all 10 remaining Rule 35 migrations with status table, per-table specs, and cross-cutting concerns. ~250 lines, ASCII only. |
| `scripts/migrations/cycle18_ohlcv_daily_to_v2.py` | Idempotent migration script. Detect-old/new schema, transactional INSERT-SELECT with `timestamp * 1000`, row-count + per-asset + latest-UTC + OHLCV-value cross-checks, DROP+RENAME. Exit 0 if already migrated. |
| `data/crypto_data.db.cycle18_backup` | Pre-migration full-DB backup (308,719,616 bytes; md5 matched source). Gitignored under `data/`. |

---

## Migration verification (acceptance criteria 4-12)

### Pre/post snapshot

```
Pre-migration:
  rows: 1802 (BTC=901, ETH=901)
  ts_min: 1699747200    (seconds)
  ts_max: 1777507200    (seconds)
  latest row: ('BTC', 1777507200, '2026-04-30',
               75780.0, 76473.26, 75494.7, 75801.1, 1731.57388)
  latest UTC: 2026-04-30T00:00:00+00:00

Post-migration:
  rows: 1802 (BTC=901, ETH=901)
  ts_min: 1699747200000 (milliseconds)
  ts_max: 1777507200000 (milliseconds)
  latest row: ('BTC', 1777507200000, '2026-04-30',
               75780.0, 76473.26, 75494.7, 75801.1, 1731.57388)
  latest UTC: 2026-04-30T00:00:00+00:00

Delta: 0.0 seconds
```

### Schema (post-migration)

```
PRAGMA table_info(ohlcv_daily):
  (0, 'asset',     'TEXT',    1, None, 1)   <- PK pos 1
  (1, 'timestamp', 'INTEGER', 1, None, 2)   <- PK pos 2
  (2, 'date',      'TEXT',    1, None, 0)
  (3, 'open',      'REAL',    0, None, 0)
  (4, 'high',      'REAL',    0, None, 0)
  (5, 'low',       'REAL',    0, None, 0)
  (6, 'close',     'REAL',    0, None, 0)
  (7, 'volume',    'REAL',    0, None, 0)
```

No `id` column. Compound PK on `(asset, timestamp)` -- subsumes the
old `UNIQUE(asset, timestamp)` constraint.

### Idempotency check

Re-running the migration script after a successful migration:

```
[migrate] Opening C:\...\data\crypto_data.db
[migrate] Already migrated -- ohlcv_daily has new schema. Exiting cleanly.
```

Exit code 0.

### Reader spot-check (`engines/lstm_predictor.py:68`)

The reader query: `SELECT date, open, high, low, close, volume FROM
ohlcv_daily WHERE asset=? ORDER BY date`.

Post-migration:

```
BTC: 901 rows
  first: {'date': '2023-11-12', 'open': 37129.99, 'high': 37222.22,
          'low': 36731.1, 'close': 37064.13, 'volume': 17687.18874}
  last : {'date': '2026-04-30', 'open': 75780.0, 'high': 76473.26,
          'low': 75494.7, 'close': 75801.1, 'volume': 1731.57388}
ETH: 901 rows
  first: {'date': '2023-11-12', 'open': 2053.16, 'high': 2066.5,
          'low': 2012.1, 'close': 2044.68, 'volume': 281907.559}
  last : {'date': '2026-04-30', 'open': 2252.9, 'high': 2279.0,
          'low': 2236.36, 'close': 2250.55, 'volume': 37696.7617}
```

All values byte-for-byte identical to pre-migration. Reader works
unchanged.

### MCP `get_collector_health` verification

Calling `_collect_db_health` post-migration returns:

```json
{
  "ohlcv_daily": {
    "row_count": 1802,
    "latest": "2026-04-30T00:00:00+00:00",
    "staleness_seconds": 79685.511,
    "threshold_seconds": 93600,
    "is_stale": false
  }
}
```

The autodetect heuristic (`> 1e12 -> ms; else seconds`) correctly handles
the new ms timestamps. Sanity-checked against:

- `fear_greed` (ms post-Cycle-17): is_stale=false, latest matches
- `funding_rates` (still seconds): is_stale=false, latest matches

The mixed state (some tables ms, some seconds) is handled correctly by
the autodetect path. No code change required this cycle, per the Brief.

---

## Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `docs/SCHEMA_MIGRATION_PLAN.md` exists, ASCII-only | PASS |
| 2 | Plan doc lists all 10 tables in priority order with patterns annotated | PASS (10 tables, plus Cycle 27 cleanup row) |
| 3 | Plan doc has cross-cutting concerns section | PASS (backups, readers, autodetect, template, plan log) |
| 4 | `data/crypto_data.db.cycle18_backup` created BEFORE migration script runs | PASS (md5 match before script invocation) |
| 5 | `scripts/migrations/cycle18_ohlcv_daily_to_v2.py` exists, idempotent | PASS (idempotency re-run verified) |
| 6 | Pre/post migration row count of `ohlcv_daily` is 1,802 (no rows lost) | PASS (1802 -> 1802) |
| 7 | Pre/post latest UTC moments match (delta < 1 sec) | PASS (delta 0.0s) |
| 8 | New schema is compound PK on (asset, timestamp); no id column | PASS |
| 9 | New `ohlcv_daily.timestamp` values are ms (e.g. latest = 1777507200000) | PASS (latest = 1777507200000) |
| 10 | `engines/crypto_data_collector.py` writer + init_db schema updated | PASS |
| 11 | `engines/lstm_predictor.py:68` reader returns same data post-migration | PASS (BTC and ETH first/last rows byte-identical) |
| 12 | MCP `get_collector_health` reports `ohlcv_daily` correctly post-migration | PASS (autodetect handles mixed state) |
| 13 | `claude/TODO.md` updated (close Cycle 18 entry; add Cycle 19) | PASS |
| 14 | All committable files ASCII-only (Rule 20) | PASS (verified by byte scan) |
| 15 | `data/crypto_data.db.cycle18_backup` is gitignored, NOT committed | PASS (`data/` is gitignored) |
| 16 | Retro at `claude/retros/RETRO_ohlcv_daily_migration.md` | THIS FILE |

---

## Debugging trail

Nothing went wrong. The Cycle 17 pilot's recipe carried over cleanly:

1. Backup first (md5-verified before any script ran).
2. Pre-flight Python query to confirm 1,802 rows / per-asset 901 each
   and capture pre-migration latest values for cross-check.
3. Reader audit via Grep: only `engines/lstm_predictor.py:68` and the
   writer in `engines/crypto_data_collector.py` reference `ohlcv_daily`
   in `.py` files. No `ohlcv_daily.id` foreign-key dependencies. No
   `WHERE timestamp >` clauses with raw epoch values.
4. Migration script copied from cycle17 template; adapted for compound
   PK and per-asset row-count cross-check.
5. Idempotency verified by re-run.
6. Writer updated: `ts = int(c[0] / 1000)` -> `ts = int(c[0])`; the
   `date` strftime now does `ts // 1000` inline (since `ts` is now ms).
7. Reader spot-check: byte-identical first/last rows for both assets.
8. MCP health verification: autodetect heuristic handles the mixed
   ms/seconds state correctly across all monitored tables.

The `sqlite3` CLI is not on PATH on this machine; all verification
queries used Python's `sqlite3` module instead. Documented for future
cycles.

---

## Test results

- Migration script: 1 successful run + 1 idempotent re-run, both exit 0
- Schema cross-check: PRAGMA table_info confirms compound PK shape
- Row-count cross-check: 1802 -> 1802 (overall and per-asset)
- Latest-UTC cross-check: delta 0.0 seconds (the daily bar timestamps
  are exact UTC midnights, so the conversion is loss-free)
- OHLCV value cross-check: latest row values preserved byte-for-byte
- Date string cross-check: derived from ms timestamp, matches stored
- Reader cross-check: 901 BTC + 901 ETH rows, first/last identical
- MCP health: ohlcv_daily, fear_greed (ms), funding_rates (seconds) all
  report correctly via autodetect

---

## Open items / next cycle inputs

- **Cycle 19** picks up from `docs/SCHEMA_MIGRATION_PLAN.md` row #3
  (`market_data`, schema-only, empty table). Decision points:
  - Should `market_data` even exist? Pre-Praxis-recovery artifact.
    Drop or rebuild?
  - If `market_data` defers, slot `ohlcv_4h` (10,806 rows, simple
    pattern) as the actual second-non-trivial migration.
- **Cycle 27** (eventual) collapses the `_to_latest_ms` autodetect
  heuristic to strict ms-only. Listed in the plan doc; not actionable
  until Cycles 19-26 complete.
- **Plan-doc commit hash**: row #2 in the Status summary table reads
  `<TBD>` for Cycle 18's commit. Replace with the actual commit hash
  in the post-commit cleanup if Chat wants it tracked, or accept that
  `git log` is the source of truth and leave `<TBD>` (the BRIEF
  permitted either approach).
- **Cycle 17 backup retention**: `data/crypto_data.db.cycle17_backup`
  can be deleted now that Cycle 18 has shipped successfully and the
  Cycle 17 migration is 24h+ in place. Per the plan doc retention
  rule, the Cycle 18 backup stays through Cycle 19's burn-in.

---

## Deviations from Brief

None. The Brief's recipe carried over from Cycle 17 with only the
expected adaptations (compound PK, per-asset row-count checks). The
plan doc structure matches the Brief's outline; section #11 (Cycle 27
autodetect cleanup) was added beyond the Brief's 10-row template
because it's load-bearing for the long-term plan and was already
called out in the Brief's "Cross-cutting concerns" section.
