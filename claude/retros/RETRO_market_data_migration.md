# Retro: Cycle 19 -- market_data Migration + Collector Fix + Task Registration

**Brief:** `claude/handoffs/BRIEF_market_data_migration.md`
**Date:** 2026-05-01
**Duration:** ~50 min (Mode B, surgical)
**Status:** COMPLETE except for one admin-only step (see "Open items")
**Predecessors:** Cycle 18 (`cc6a178`, `cc7c591`) -- ohlcv_daily migration

---

## Summary

Cycle 19 turns `market_data` from a broken empty table into a healthy
daily collector. Four problems addressed in one cycle:

1. Schema migrated to Rule 35 (empty-table schema-only -- no data to
   preserve).
2. Writer fixed: now fetches CoinGecko `/global` once per cycle and
   populates the previously-unfilled `btc_dominance` column. Stores ms
   timestamp + matching `date` text.
3. CLI subcommand `collect-market-data` wired up with `--asset all`
   support that loops over `SUPPORTED_ASSETS`.
4. Service files (`.bat` + `.ps1`) created with CRLF line endings;
   MCP `get_collector_health` extended to monitor `market_data` with
   25h threshold; doc trio (SCHEMA_NOTES, SCHEMA_MIGRATION_PLAN, TODO)
   updated.

Manual first-run seeded 3 rows (BTC, ETH, SOL) for 2026-05-01 with
identical dominance 58.47%. The MCP smoke test confirmed every
monitored table reports correctly via the autodetect heuristic;
`unmonitored=[]`.

The one outstanding step is `Register-ScheduledTask`, which requires
Administrator privileges that the Code session does not have. Files
are in place; Jeff runs the registration once from an elevated shell.

---

## Changes Made

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `engines/crypto_data_collector.py` | `init_db` market_data CREATE TABLE: `id` PK + `UNIQUE(asset, date)` -> compound `PRIMARY KEY (asset, timestamp)`, no `id`, `timestamp INTEGER NOT NULL`. New `_fetch_btc_dominance()` helper. `collect_market_data` rewritten: accepts optional `btc_dominance` parameter, computes UTC-midnight ms timestamp, INSERT now writes 10 columns (was 8). New `cmd_collect_market_data` CLI handler. New parser entry + dispatch lambda. | 124-135, 535-602, ~1058-1062 (parser), ~1107 (dispatch), ~983-998 (handler) |
| `servers/praxis_mcp/tools/meta.py` | Added `"market_data": 90000` (25h: 24h + 1h slack) to `primary_monitored` | 245 |
| `docs/SCHEMA_NOTES.md` | `market_data` row + per-table prose: NONCONFORMING/EMPTY -> CONFORMING (Cycle 19) with notes about dominance + no-backfill. `ohlcv_daily` per-table prose: NONCONFORMING -> CONFORMING (Cycle 18; Cycle 18 missed this drift, fixed now). Migration status table updated. "Currently-stale tables" note for market_data updated. | 94-108, 126-135, 256, 261, 274-275, 285-289 |
| `docs/SCHEMA_MIGRATION_PLAN.md` | Cycle 19 row in Status summary: pending -> DONE. Per-table spec #3: rewritten to reflect what shipped. | 21, 87-110 |
| `claude/TODO.md` | Closed Cycle 19 entry; added Cycle 20 (ohlcv_4h migration) high-priority TODO; added admin-step TODO for the elevated registration. Cycle 18 -> closed; Cycle 19 -> closed-with-caveat in Recently closed. | 21-32, 209-228 |

### Files Created

| File | Purpose |
|------|---------|
| `scripts/migrations/cycle19_market_data_to_v2.py` | Idempotent empty-table schema-only migration. Detect-old/new, transactional CREATE-INSERT_SELECT-DROP-RENAME, post-migration sanity check. Defense-in-depth INSERT-SELECT preserves any rows even though pre-state was empty. |
| `services/market_data_collector_service.bat` | CRLF batch script invoking `python -m engines.crypto_data_collector collect-market-data --asset all`. Logs to `logs/market_data_collector.log`. |
| `services/register_market_data_task.ps1` | CRLF PowerShell script. Registers `PraxisMarketDataCollector` for daily 00:35 local. Mirrors `register_fear_greed_task.ps1` structure. |
| `data/crypto_data.db.cycle19_backup` | Pre-migration full-DB backup (483,004,416 bytes; md5 matched source). Gitignored. Created BEFORE migration script ran. |
| `claude/retros/RETRO_market_data_migration.md` | This file. |

### Files Deleted

| File | Reason |
|------|--------|
| `data/crypto_data.db.cycle17_backup` | Per migration plan retention policy: delete after the cycle-after-next ships. Cycle 17 -> drop at start of Cycle 19. Authorized by user. |
| `data/crypto_data.db.cycle18_backup` | Same retention policy: cycle 18 backup superseded by cycle 19's pre-state. Authorized by user. |

---

## Migration verification

### Pre/post snapshot

```
Pre-migration:
  rows: 0
  schema: id PK + UNIQUE(asset, date), no timestamp column

Post-migration:
  rows: 0
  schema: PRIMARY KEY (asset, timestamp), no id, timestamp INTEGER NOT NULL
```

### Schema (post-migration, post-first-run)

```
PRAGMA table_info(market_data):
  (0, 'asset',              'TEXT',    1, None, 1)   <- PK pos 1
  (1, 'timestamp',          'INTEGER', 1, None, 2)   <- PK pos 2
  (2, 'date',               'TEXT',    1, None, 0)
  (3, 'market_cap',         'REAL',    0, None, 0)
  (4, 'total_volume',       'REAL',    0, None, 0)
  (5, 'circulating_supply', 'REAL',    0, None, 0)
  (6, 'total_supply',       'REAL',    0, None, 0)
  (7, 'ath',                'REAL',    0, None, 0)
  (8, 'ath_change_pct',     'REAL',    0, None, 0)
  (9, 'btc_dominance',      'REAL',    0, None, 0)
```

### Idempotency check

```
[migrate] Already migrated -- market_data has new schema. Exiting cleanly.
```

Exit 0.

### Manual first-run output

```
Collecting market data for BTC...
  Price: $78,147.00
  Market cap: $1,565,146,027,333
  ATH distance: -38.0%
  BTC dominance: 58.47%

Collecting market data for ETH...
  Price: $2,299.95
  Market cap: $277,646,231,244
  ATH distance: -53.5%
  BTC dominance: 58.47%

Collecting market data for SOL...
  Price: $84.03
  Market cap: $48,380,276,429
  ATH distance: -71.4%
  BTC dominance: 58.47%
```

### Table state after first run

```
3 rows
asset=BTC ts=1777593600000 date=2026-05-01 dom=58.47% mcap=$1,565B
asset=ETH ts=1777593600000 date=2026-05-01 dom=58.47% mcap=$277B
asset=SOL ts=1777593600000 date=2026-05-01 dom=58.47% mcap=$48B

ts_as_utc: 2026-05-01T00:00:00+00:00 (all 3 -- UTC midnight, ms)
distinct btc_dominance: 1 value (58.474720208795446)
```

### MCP `get_collector_health` smoke test

```json
{
  "market_data": {
    "row_count": 3,
    "latest": "2026-05-01T00:00:00+00:00",
    "staleness_seconds": 61892.833,
    "threshold_seconds": 90000,
    "is_stale": false
  }
}
```

`unmonitored: []` -- every table in `crypto_data.db` is now under
monitoring. Sanity-checked alongside the other 8 monitored tables;
all report correctly via the autodetect heuristic.

---

## Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `data/crypto_data.db.cycle19_backup` created BEFORE migration runs | PASS (md5 verified) |
| 2 | `scripts/migrations/cycle19_market_data_to_v2.py` exists, idempotent | PASS (re-run verified) |
| 3 | `market_data` new schema: compound PK `(asset, timestamp)`, no id, ms timestamp | PASS |
| 4 | `engines/crypto_data_collector.py` `init_db()` schema matches new shape | PASS |
| 5 | Writer fetches `/global`, populates `btc_dominance`, writes ms timestamp | PASS |
| 6 | `collect-market-data` CLI subcommand wired up | PASS |
| 7 | `services/market_data_collector_service.bat` exists, CRLF, ASCII-only | PASS (20/20 CRLF, 0 non-ASCII) |
| 8 | `services/register_market_data_task.ps1` exists, CRLF | PASS (55/55 CRLF, 0 non-ASCII) |
| 9 | `PraxisMarketDataCollector` scheduled task registered | **DEFERRED** (admin required; see Open items) |
| 10 | After first run, `market_data` has 3 rows for today's date | PASS (BTC/ETH/SOL for 2026-05-01) |
| 11 | All 3 rows have non-null `btc_dominance` (same value) | PASS (58.474720208795446 across all 3) |
| 12 | All 3 rows have ms `timestamp` and matching `date` | PASS (1777593600000 = 2026-05-01T00:00:00+00:00) |
| 13 | MCP `primary_monitored` includes `market_data: 90000` | PASS |
| 14 | `docs/SCHEMA_NOTES.md` updated | PASS |
| 15 | `docs/SCHEMA_MIGRATION_PLAN.md` status table + per-table spec updated | PASS |
| 16 | `claude/TODO.md` updated (close Cycle 19, add Cycle 20) | PASS |
| 17 | All committable files ASCII-only | PASS |
| 18 | `.bat` and `.ps1` files preserve CRLF | PASS (verified by byte count) |
| 19 | Retro at `claude/retros/RETRO_market_data_migration.md` | THIS FILE |

18 of 19 PASS. #9 deferred -- the .ps1 file is in place and ready;
Jeff runs `.\services\register_market_data_task.ps1` once from an
elevated PowerShell. The manual backfill (which does not need admin)
already seeded today's rows so the table is healthy in the meantime.

---

## Debugging trail

### Initial backslash-escape disaster on .bat/.ps1 generation

First attempt to write the service files used Python triple-quoted
strings with backslashes in the path (`C:\Data\Development\...`).
Python interpreted `\a` as the alert escape and `\r` as carriage
return, mangling lines 8 and 3 of the .bat and .ps1 respectively
(`Scriptsctivate.bat` and `servicesegister_market_data_task.ps1`).
Caught by re-reading the files immediately after write. Fixed by
rewriting with raw-string literals (`r'...'`) for every line.
**Lesson recorded for the migration template:** always `Read` the
file back after writing CRLF content via Python; do not trust a
single CRLF-pair count alone.

### CoinGecko 429 on the third asset

First manual run got HTTP 429 on the SOL `/coins/{id}` call. Root
cause: the Brief recommended `_fetch_btc_dominance()` be called inside
`collect_market_data` for simplicity, which means N=3 assets x 2
calls/asset = 6 calls in a ~2-second burst. CoinGecko's free-tier
rate limit is documented as 30/min but in practice clamps tighter on
short bursts. **Deviation from Brief** (justified): refactored
`cmd_collect_market_data` to fetch `/global` once at the top and
thread the dominance value through to each per-asset call, with a
2-second sleep between assets. Total = 4 calls spread over 6 seconds.
Re-run succeeded for all 3 assets. The `btc_dominance` parameter
defaults to `None`, so direct calls to `collect_market_data(asset,
conn)` (e.g. from `cmd_collect_all`) still work and fetch their own.

### Cycle 18 SCHEMA_NOTES drift

While editing `SCHEMA_NOTES.md` for Cycle 19, noticed the
per-table-prose section for `ohlcv_daily` still said NONCONFORMING
even though the migration status table had been updated to CONFORMING
in Cycle 18. Fixed inline as part of this cycle's doc pass.
Surface for future cycles: when migrating, edit BOTH the prose
section AND the migration status table in `SCHEMA_NOTES.md`.

### Admin-required step

`Register-ScheduledTask` returned `Access is denied` (`HRESULT
0x80070005`). The Brief noted "Run as Administrator" in the .ps1
header comment but did not explicitly anticipate that Code's session
would run unprivileged. Documented as an Open item; the .ps1 file is
correct and registers cleanly when run with admin rights.

---

## Test results

- Migration script: 1 successful run + 1 idempotent re-run, exit 0
- CLI subcommand: `python -m engines.crypto_data_collector
  collect-market-data --asset all` succeeds end-to-end
- Manual first-run: 3 rows seeded; dominance non-null and identical
  across rows; ms timestamps + UTC-midnight `date` derived correctly
- MCP smoke test (in-process call to `_collect_db_health`): all 9
  monitored tables report correctly; `unmonitored=[]`
- ASCII verification (Rule 20): 0 non-ASCII bytes in every committed
  file
- CRLF verification (Rule 21): `.bat` 20 CRLF / 0 lone LF; `.ps1` 55
  CRLF / 0 lone LF

---

## Open items / next cycle inputs

- **One-shot admin step (HIGH PRIORITY)**: Jeff runs
  `.\services\register_market_data_task.ps1` from an elevated
  PowerShell to register `PraxisMarketDataCollector`. The .bat and
  .ps1 are committed and ready. Verify with `Get-ScheduledTask
  -TaskName PraxisMarketDataCollector`.
- **Cycle 20**: Migrate `ohlcv_4h` (10,818 rows; simple
  stop-migrate-start; same shape as Cycle 18's `ohlcv_daily` but with
  a `datetime` column that needs `+00:00` rewriting).
- **Plan-doc commit hash**: row #19 in the Status summary table reads
  `<TBD>`. Replace with the actual commit hash in the post-commit
  cleanup, same as Cycle 18.
- **CoinGecko free-tier re-evaluation**: if `market_data` collection
  ever fails repeatedly with 429 in production, consider:
  (a) adding exponential backoff to `_fetch_btc_dominance()` and
  `collect_market_data`, OR (b) upgrading to the paid CoinGecko tier
  for proper rate limits + historical backfill capability. Not
  pressing -- 4 calls/day is well under any sane limit.

---

## Deviations from Brief

1. **`/global` fetched once in CLI handler, not per asset.** The Brief
   recommended "fetch `/global` inside the function each time"
   (Section 2 / Recommendation block). Hit a CoinGecko 429 on the
   first attempt; refactored to fetch once and pass through. The
   `btc_dominance` parameter on `collect_market_data` defaults to
   `None` so single-asset / `cmd_collect_all` callers still work.
   This trade-off was the Brief's "Document the choice in the retro"
   contingency.
2. **Added 2-second sleep between assets** in `cmd_collect_market_data`
   to spread the 4 API calls over ~6 seconds rather than burst them
   in 2. Same anti-rate-limit motivation as #1.
3. **Cycle 18 SCHEMA_NOTES drift fix** added to this cycle's
   `docs/SCHEMA_NOTES.md` edits even though it's properly Cycle 18's
   work. Avoiding letting an internal-inconsistency rot.
4. **Acceptance criterion #9 deferred** rather than passed: scheduled
   task registration requires admin, which the Code session lacks.
   The Brief assumed admin context implicitly. Files are ready; only
   the registration command itself is pending.
