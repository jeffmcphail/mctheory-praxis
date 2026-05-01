# BRIEF: Cycle 19 -- market_data Migration + Collector Fix + Task Registration

**Series:** praxis
**Cycle:** 19
**Mode:** B (Brief -> Code)
**Author:** Chat (praxis_main_current)
**Date:** 2026-05-01
**Predecessor:** Cycle 18 (`cc6a178`, `cc7c591`) -- ohlcv_daily migration

---

## Context

`market_data` table exists in `crypto_data.db` but has 0 rows. Investigation
in chat surfaced four problems:

1. **No CLI subcommand** -- `collect_market_data()` function exists at
   `engines/crypto_data_collector.py:538` but no `collect-market-data`
   parser entry, so it's only reachable via `collect-all` (which is itself
   not scheduled).
2. **No scheduled task** -- nothing in Windows Task Scheduler invokes the
   collector. Table has been empty since creation.
3. **Schema doesn't conform to Rule 35** -- has `id INTEGER PRIMARY KEY
   AUTOINCREMENT`, `date TEXT` only (no `timestamp` column).
4. **Writer is buggy** -- `INSERT` statement uses 8 columns but schema
   defines 9 data columns; `btc_dominance` column is reserved but never
   populated. Even if scheduled, this column would always be NULL.
   The reason is that BTC dominance comes from `/global` endpoint, which
   the writer never calls -- it only hits `/coins/{id}` per asset.

Cycle 19 fixes all four problems together: migrate to Rule 35, add
`/global` API call to populate dominance, register a daily scheduled task,
add CLI subcommand, add MCP monitoring. After Cycle 19 lands, `market_data`
becomes a healthy ongoing collector that captures macro-state daily.

**One non-fixable limitation:** CoinGecko's free `/coins/{id}` endpoint
returns *current* state only -- there is no historical backfill capability
on the free tier. The table starts populating from "today" forward.
Document this honestly in SCHEMA_NOTES and accept it.

---

## Asset list

`SUPPORTED_ASSETS` in `engines/crypto_data_collector.py:42` includes BTC,
ETH, SOL. The migrated `market_data` should collect all three. BTC
dominance is a single global value, recorded identically across all three
asset rows on a given day (a normalization quirk worth a note in
SCHEMA_NOTES.md, but not worth normalizing into a separate table).

---

## Scope

### Task 1: Migrate `market_data` schema to Rule 35

**Current schema:**

```sql
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    date TEXT NOT NULL,                 -- 'YYYY-MM-DD'
    market_cap REAL,
    total_volume REAL,
    circulating_supply REAL,
    total_supply REAL,
    ath REAL,
    ath_change_pct REAL,
    btc_dominance REAL,
    UNIQUE(asset, date)
)
```

**Target schema (Rule 35 conforming):**

```sql
CREATE TABLE market_data (
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,         -- UTC ms (midnight of the date)
    date TEXT NOT NULL,                 -- 'YYYY-MM-DD' (UTC, derived from timestamp)
    market_cap REAL,
    total_volume REAL,
    circulating_supply REAL,
    total_supply REAL,
    ath REAL,
    ath_change_pct REAL,
    btc_dominance REAL,
    PRIMARY KEY (asset, timestamp)
)
```

Changes:
- Drop `id` AUTOINCREMENT
- Add `timestamp INTEGER NOT NULL` (UTC midnight ms of the collection date)
- Compound PK on `(asset, timestamp)`
- `date` becomes derived cache column (semantics unchanged)

**Migration script:** `scripts/migrations/cycle19_market_data_to_v2.py`

The table is empty (0 rows), so the migration is schema-only -- no data to
preserve. But still follow the Cycle 17/18 idempotent recipe for safety:

1. Open `data/crypto_data.db` with explicit transaction control (Rule 34).
2. Confirm OLD schema (id PK present, no timestamp column). Print
   "Already migrated" and exit cleanly if NEW schema detected.
3. CREATE TABLE `market_data_new` with target schema.
4. (No INSERT-SELECT needed -- empty table.)
5. Verify both old and new are empty.
6. DROP TABLE `market_data`.
7. ALTER TABLE `market_data_new` RENAME TO `market_data`.
8. Print pre/post schema for the retro.

Backup `data/crypto_data.db` to `data/crypto_data.db.cycle19_backup`
BEFORE running the migration script. Per the migration plan retention
policy, the cycle18 backup can also be deleted at the start of this cycle
(its data is durably preserved in cycle18 commit + this cycle's pre-state).

### Task 2: Fix and update the `collect_market_data` writer

**Current writer:** `engines/crypto_data_collector.py:538`. Single API call
to `/coins/{id}` per asset. Stores 8 of 9 columns; `btc_dominance` is
never populated.

**Fixes needed:**

1. **Add a `/global` API call** at the start of `collect_market_data`,
   before iterating over assets, to fetch dominance. Cache the value;
   write it to every row in this collection cycle. Endpoint:
   `https://api.coingecko.com/api/v3/global`. Response shape:
   ```json
   {
     "data": {
       "market_cap_percentage": {"btc": 58.4, "eth": 10.4, ...},
       ...
     }
   }
   ```
   Read `data.market_cap_percentage.btc` for dominance.

2. **Update INSERT to include `btc_dominance`** as the 9th column.

3. **Update INSERT to include `timestamp`** -- compute as
   `int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
   microsecond=0).timestamp() * 1000)`. The `date` column is derived
   from this same value via `strftime('%Y-%m-%d')`. Both reflect UTC
   midnight of the collection day.

4. **Update INSERT to use the new 11-column shape** (asset, timestamp,
   date, market_cap, total_volume, circulating_supply, total_supply, ath,
   ath_change_pct, btc_dominance) with `INSERT OR REPLACE` semantics
   matched to the new compound PK on `(asset, timestamp)`.

The `collect_market_data` signature should stay the same: `(asset, conn)`
per call, called once per asset. But the function should now fetch
dominance ONCE outside the per-asset loop and pass it in. Suggest a small
refactor: extract `_fetch_btc_dominance() -> float` as a helper, called
ONCE in `collect_all` before the per-asset loop, with the dominance
threaded through. OR call `/global` inside `collect_market_data` itself if
that's simpler -- it just means N calls per cycle instead of 1, but with
N=3 and the free tier at 30 calls/min, it's well under budget.

**Recommendation: keep `collect_market_data(asset, conn)` signature
intact, fetch `/global` inside the function each time.** Simpler and the
quota cost is negligible. Document the choice in the retro.

5. **Update `init_db()` schema definition** (~line 128) to match the new
   target schema -- so a fresh DB creates the right shape from scratch.

### Task 3: Add `collect-market-data` CLI subcommand

In the `argparse` block (~line 1030-1095 in
`engines/crypto_data_collector.py`):

Add parser:
```python
p_md = subs.add_parser("collect-market-data",
                       help="CoinGecko market data + BTC dominance")
p_md.add_argument("--asset", default="BTC",
                  help="Asset to collect (BTC, ETH, SOL, or 'all')")
```

Add dispatch lambda (~line 1099):
```python
"collect-market-data": lambda a: cmd_collect_market_data(a),
```

Add command handler `cmd_collect_market_data(args)`:
- If `args.asset == "all"`, loop over all `SUPPORTED_ASSETS` and call
  `collect_market_data(asset, conn)` for each
- Otherwise call `collect_market_data(args.asset, conn)` for the
  specified asset
- Open one connection via `init_db()` at the start; close at the end

The pattern mirrors `cmd_collect_fear_greed`.

### Task 4: Register Windows scheduled task for daily market_data collection

Create three files following the existing pattern:

**`services/market_data_collector_service.bat`** (CRLF-required per Rule 21):

```batch
@echo off
REM CoinGecko market data + BTC dominance collector.
REM Runs daily at 00:35 UTC.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\market_data_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting market_data collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-market-data --asset all >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
```

**`services/register_market_data_task.ps1`** (CRLF-required per Rule 21):

Copy the structure of `register_fear_greed_task.ps1`. Differences:
- `$TaskName = "PraxisMarketDataCollector"`
- `$BatFile = "$PraxisDir\services\market_data_collector_service.bat"`
- `$Trigger = New-ScheduledTaskTrigger -Daily -At "00:35"`
  (00:35 local Toronto, just after fear_greed at 00:30, before any
  end-of-day reporting cron)
- `-Description "Praxis market data collector -- daily at 00:35
  (CoinGecko per-asset stats + BTC dominance from /global)"`

The `register_all_tasks.ps1` orchestrator auto-picks-up new
`register_*_task.ps1` files via Get-ChildItem; no edits to that script
needed.

**Run the registration once at the end of the cycle:**

```powershell
.\services\register_market_data_task.ps1
```

Verify with:
```powershell
Get-ScheduledTask -TaskName "PraxisMarketDataCollector"
```

**Backfill the table immediately:**

After registration, manually trigger one collection to seed today's
row(s). Two options:

```powershell
Start-ScheduledTask -TaskName "PraxisMarketDataCollector"
```

OR direct invocation:

```bash
python -m engines.crypto_data_collector collect-market-data --asset all
```

Either way, verify post-run that the table has 3 rows (one per supported
asset for today's date).

### Task 5: Add `market_data` to MCP `get_collector_health` monitoring

Extend `servers/praxis_mcp/tools/meta.py` `primary_monitored` dict to add:

```python
"market_data": 90000,  # 25h: collector runs daily at 00:35; allow one missed cycle (24h + 1h slack)
```

This is the simple int-spec form (uses `timestamp` column with autodetect
ms/sec; new table is ms-conformant from day 1, so autodetect handles it).

After this lands, post-restart of Claude Desktop, `market_data` will move
from `unmonitored` into the monitored `tables` dict. After the first
collection runs, `is_stale=false`. Until then (tiny window), it'll show
`is_stale=true` because the table is empty -- handle this gracefully via
the existing "empty table" path in `_collect_db_health` (already returns
`{"row_count": 0, "error": "empty table"}` -- that's fine).

### Task 6: Update `docs/SCHEMA_NOTES.md`

In the `market_data` row of the inventory:
- Conformance: NONCONFORMING -> CONFORMING
- Add note: "BTC dominance is a single global value; populated identically
  across all rows for a given collection day (see writer in
  engines/crypto_data_collector.py)."
- Add note: "CoinGecko free tier `/coins/{id}` returns current state only;
  no historical backfill capability. Table populates from
  Cycle 19 forward."
- Update migration status table: `market_data | CONFORMING | 19`

### Task 7: Update `docs/SCHEMA_MIGRATION_PLAN.md`

In the status summary table at top:
- Cycle 19 row: change "schema-only | pending | --" to
  "schema-only-plus-collector-fix | DONE | <commit-hash>"

In the per-table specs section #3 (market_data), update the spec to
reflect what actually shipped:
- Note that the migration was empty-table schema-only
- Note that the collector was fixed (added /global, btc_dominance, CLI
  subcommand) and registered as a scheduled task
- Note that historical backfill is unavailable on free tier
- Mark as DONE

Plan-doc commit hash insertion follows Cycle 18 precedent: insert `<TBD>`
in this commit, fix in a follow-up commit OR leave as `<TBD>` (Code's
call -- both worked last cycle).

### Task 8: Update `claude/TODO.md`

Mark "Cycle 19: Migrate next table per docs/SCHEMA_MIGRATION_PLAN.md"
as DONE in "Recently closed" with cycle hash.

Add: "Cycle 20: Migrate `ohlcv_4h` (10,818 rows, simple pattern,
Binance-backfillable)."

The "Register scheduled collector for `onchain_btc` table" TODO can stay
as-is in Active TODOs -- it's a separate gap that this cycle does NOT
address. Different table, different API source.

---

## Out of scope

- Migrating any table other than `market_data`
- Touching `onchain_btc` collector registration (separate TODO; deferred)
- Paid CoinGecko tier upgrade for historical backfill (deferred)
- Adding new assets beyond what's already in `SUPPORTED_ASSETS`
- Touching the `_to_latest_ms` autodetect heuristic (Cycle 27)

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | `data/crypto_data.db.cycle19_backup` created BEFORE migration runs |
| 2 | `scripts/migrations/cycle19_market_data_to_v2.py` exists, idempotent |
| 3 | `market_data` new schema: compound PK `(asset, timestamp)`, no id, `timestamp INTEGER` ms |
| 4 | `engines/crypto_data_collector.py` `init_db()` schema matches new shape |
| 5 | `engines/crypto_data_collector.py` `collect_market_data` writer fetches `/global`, populates `btc_dominance`, writes ms timestamp |
| 6 | `collect-market-data` CLI subcommand wired up (parser entry + dispatch) |
| 7 | `services/market_data_collector_service.bat` exists, CRLF line endings, ASCII-only |
| 8 | `services/register_market_data_task.ps1` exists, CRLF line endings |
| 9 | `PraxisMarketDataCollector` scheduled task registered (verify via `Get-ScheduledTask`) |
| 10 | After manual first run, `market_data` has 3 rows (one per asset: BTC, ETH, SOL) for today's date |
| 11 | All 3 rows have non-null `btc_dominance` (same value across all three) |
| 12 | All 3 rows have `timestamp` in ms (e.g., 1777939200000 for 2026-05-05 if that were today) and matching `date` text |
| 13 | `servers/praxis_mcp/tools/meta.py` `primary_monitored` includes `market_data: 90000` |
| 14 | `docs/SCHEMA_NOTES.md` updated: market_data marked CONFORMING with notes |
| 15 | `docs/SCHEMA_MIGRATION_PLAN.md` status table + per-table spec updated |
| 16 | `claude/TODO.md` updated (close Cycle 19, add Cycle 20) |
| 17 | All committable files ASCII-only (Rule 20) |
| 18 | `.bat` and `.ps1` files preserve CRLF (Rule 21; verify with `tr -cd '\r' < file | wc -c` matches line count) |
| 19 | Retro at `claude/retros/RETRO_market_data_migration.md` includes migration log, first-run output, schedule task verification, MCP behavior |

---

## Notes for Code

- Rule 34: fresh connections per logical pass in any analysis script.
- Rule 21: CRLF line endings on `.bat` and `.ps1` files. Verify after
  edit/create.
- The `/global` API endpoint is unauthenticated; no API key needed at the
  free tier. Same as the existing `/coins/{id}` calls.
- CoinGecko free tier rate limit: 30 calls/min. This cycle adds 1 call
  per collection run (at most 4 calls per run total: 1x /global +
  3x /coins/{id}). Daily cadence = ~4 calls/day. Well under the 10k/month
  cap.
- `cycle18_ohlcv_daily_to_v2.py` and `cycle17_fear_greed_to_v2.py` are
  good templates; copy and adapt for empty-table schema-only migration.
- The first scheduled run after registration won't fire until 00:35
  tomorrow Toronto time. Don't wait -- run the manual backfill at the end
  of the cycle to seed today's row.
- Verify the manual backfill creates rows with `btc_dominance` populated
  (not NULL) -- this confirms the `/global` call worked. If it's NULL,
  there's a bug in the new code; surface to chat before declaring victory.
- Per Rule 32, MCP server changes (meta.py) require a Claude Desktop
  full-restart for the running server to pick them up. Code can verify
  the change loads cleanly via the smoke test
  (`python -m servers.praxis_mcp.test_smoke`); the actual `market_data`
  appearance in `get_collector_health` happens after Jeff's next restart.
- The retro should include: (a) migration script output, (b) first-run
  CLI output showing 3 rows inserted with dominance values, (c)
  PRAGMA table_info confirming new schema, (d) `Get-ScheduledTask`
  output showing the new task exists, (e) any deviations from this
  Brief.
