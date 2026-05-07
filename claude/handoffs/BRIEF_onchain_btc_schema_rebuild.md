# Cycle 31 -- onchain_btc full Rule 35 conformance

**Predecessor:** Cycle 30 (`63993be`) -- onchain scheduled task
registration. Surfaced the gap that onchain_btc still had a
synthetic `id` PK and no `timestamp INTEGER` / `datetime TEXT`
columns, even though the data is daily-grain temporally indexed.
**Mode:** Files-included delta zip (Code applies init_db update
+ writer update; user runs the rebuild script during a brief
maintenance window).
**Risk:** low. Pure structural change with deterministic column
derivation. 370 rows; transaction will be sub-second.

## What

Brings `onchain_btc` into full Rule 35 compliance:

1. Remove `id INTEGER PRIMARY KEY AUTOINCREMENT`
2. Add `timestamp INTEGER NOT NULL` (ms; UTC midnight of `date`,
   matching ohlcv_daily's convention so cross-table JOINs work)
3. Add `datetime TEXT NOT NULL` (ISO 8601 with `+00:00`)
4. Promote `UNIQUE(date)` to `PRIMARY KEY (date)`
5. Preserve `total_btc` (legacy column kept for compatibility;
   live writer doesn't populate it but nothing else does either)

The `date TEXT` column stays. It's the natural lookup key (just
like ohlcv_daily keeps both `timestamp` and `date`).

## Why this matters

Rule 35 is a contract, not a guideline. Every temporally-
indexed table must have `timestamp INTEGER` (ms UTC) for
cross-table joins. `onchain_btc` is daily-grain, but that
doesn't exempt it -- joins like `ohlcv_daily JOIN onchain_btc
ON same_timestamp` are legitimate use cases. Cycle 30's
scheduled-task registration surfaced this last gap; Cycle 31
closes it.

## Convention alignment

ohlcv_daily already uses 00:00:00 UTC of the date as its
ms timestamp. Verified directly:
- ohlcv_daily: `2026-05-07` -> timestamp `1778112000000`
- ohlcv_daily: `2026-05-06` -> timestamp `1778025600000`

onchain_btc will use the identical convention. JOINs like
`SELECT * FROM ohlcv_daily o JOIN onchain_btc oc ON
o.timestamp = oc.timestamp` will work directly post-Cycle-31.

The migration script's verification step picks the latest
onchain_btc date and confirms its computed timestamp matches
ohlcv_daily's timestamp for the same date, before declaring
success.

## Why one-shot rebuild (not dual-write)

Same justification as Cycle 26 (trades). Pure structural change
with deterministic column derivation (timestamp = parse(date) at
UTC midnight). The writer code currently doesn't reference `id`,
and the new `timestamp`/`datetime` columns are derivable from
`date`. No data semantic transformation. No burn-in needed.

## Two changes Code must make

### Change 1: `engines/crypto_data_collector.py` `init_db()` (around line 107)

Replace:

```python
conn.execute("""
    CREATE TABLE IF NOT EXISTS onchain_btc (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        active_addresses INTEGER,
        transaction_count INTEGER,
        hash_rate REAL,
        difficulty REAL,
        block_size REAL,
        total_btc REAL,
        market_cap REAL,
        UNIQUE(date)
    )
""")
```

with:

```python
conn.execute("""
    CREATE TABLE IF NOT EXISTS onchain_btc (
        date TEXT NOT NULL PRIMARY KEY,
        timestamp INTEGER NOT NULL,
        datetime TEXT NOT NULL,
        active_addresses INTEGER,
        transaction_count INTEGER,
        hash_rate REAL,
        difficulty REAL,
        block_size REAL,
        market_cap REAL,
        total_btc REAL
    )
""")
```

(Schema-shape change: drops `id`, adds `timestamp` + `datetime`,
moves `total_btc` to the end so the column ordering matches
the rebuild script's INSERT shape, promotes `UNIQUE(date)` to
`PRIMARY KEY (date)`.)

### Change 2: `engines/crypto_data_collector.py` `collect_onchain_btc` writer (around line 631)

Replace the existing INSERT block:

```python
conn.execute("""
    INSERT OR REPLACE INTO onchain_btc
    (date, active_addresses, transaction_count, hash_rate,
     difficulty, block_size, market_cap)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", (date,
      metrics_data.get("active_addresses", 0),
      metrics_data.get("transaction_count", 0),
      metrics_data.get("hash_rate", 0),
      metrics_data.get("difficulty", 0),
      metrics_data.get("block_size", 0),
      metrics_data.get("market_cap", 0)))
```

with:

```python
# Cycle 31: populate timestamp (ms, UTC midnight of `date`) and
# datetime (ISO 8601 +00:00) for Rule 35 compliance and
# cross-table JOIN compatibility (ohlcv_daily uses the same
# convention).
date_dt = datetime.strptime(date, "%Y-%m-%d").replace(
    tzinfo=timezone.utc)
ts_ms = int(date_dt.timestamp() * 1000)
dt_iso = date_dt.isoformat()
conn.execute("""
    INSERT OR REPLACE INTO onchain_btc
    (date, timestamp, datetime, active_addresses, transaction_count,
     hash_rate, difficulty, block_size, market_cap)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (date, ts_ms, dt_iso,
      metrics_data.get("active_addresses", 0),
      metrics_data.get("transaction_count", 0),
      metrics_data.get("hash_rate", 0),
      metrics_data.get("difficulty", 0),
      metrics_data.get("block_size", 0),
      metrics_data.get("market_cap", 0)))
```

(`datetime` and `timezone` are already imported at the top of
the file -- verify but they should already be there.)

### What Code does NOT touch

- The rebuild script (`scripts/migrations/cycle31_onchain_btc_
  schema_rebuild.py`) is provided in this delta zip and is run
  by the user, not Code.
- `meta.py`'s `primary_monitored` config: the `onchain_btc`
  entry already uses `timestamp_format="date"` with
  `timestamp_column="date"`. Post-Cycle-31 this still works
  correctly because `date` is preserved. Optionally Code can
  switch it to `timestamp_format="ms"` with `timestamp_column=
  "timestamp"` to use the new column, but it's NOT required --
  the existing config keeps working. **Recommend leaving
  meta.py alone in this cycle to minimize blast radius.**

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | `init_db()` CREATE TABLE for `onchain_btc` matches the new schema |
| 2 | `collect_onchain_btc` writer computes `ts_ms` + `dt_iso` from `date` and includes them in the INSERT |
| 3 | py_compile clean |
| 4 | Rebuild script's pre-flight checks all pass (no _v2 leftover, expected schema present) |
| 5 | Rebuild script completes in <1s wall-clock for 370 rows |
| 6 | Post-rebuild `PRAGMA table_info(onchain_btc)` shows no `id`, PK on `date`, presence of `timestamp INTEGER NOT NULL` and `datetime TEXT NOT NULL` |
| 7 | Post-rebuild row count == pre-rebuild row count (370) |
| 8 | Cross-table JOIN verification: `onchain_btc.timestamp` matches `ohlcv_daily.timestamp` for the latest shared date |
| 9 | After re-enabling PraxisOnchainCollector, the next manual fire writes successfully against the new schema |
| 10 | Doc trio updated: SCHEMA_NOTES.md (onchain_btc CONFORMING), SCHEMA_MIGRATION_PLAN.md (row #11 with hash), claude/TODO.md (migration program scoreboard updated to 11/11) |
| 11 | Retro at claude/retros/RETRO_onchain_btc_schema_rebuild.md |

## User maintenance window steps (reference)

```powershell
# Step 1: Disable the scheduled task
Disable-ScheduledTask -TaskName "PraxisOnchainCollector"

# Step 2: Verify (the collector finishes in <10s; unlike trades there's
# no long-lived loop process, so disable-task is sufficient)
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
  $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
  if ($cmd -like "*collect-onchain*") { Write-Output "STILL RUNNING: PID=$($_.Id)" }
}
# Should print nothing.

# Step 3: Run the rebuild
cd C:\Data\Development\Python\McTheoryApps\praxis
python scripts\migrations\cycle31_onchain_btc_schema_rebuild.py

# Step 4: Re-enable
Enable-ScheduledTask -TaskName "PraxisOnchainCollector"

# Step 5: Verify writer with manual fire
python -m engines.crypto_data_collector collect-onchain --days 7

# Step 6: Verify via MCP (after a moment)
# get_collector_health -> onchain_btc should still report cleanly
# raw_query -> SELECT date, timestamp, datetime FROM onchain_btc
#              ORDER BY date DESC LIMIT 3
# raw_query JOIN check -> SELECT oc.date, oc.timestamp AS oc_ts,
#                                od.timestamp AS od_ts
#                         FROM onchain_btc oc
#                         JOIN ohlcv_daily od ON od.date = oc.date
#                                              AND od.asset = 'BTC'
#                         ORDER BY oc.date DESC LIMIT 5;
# (oc_ts must equal od_ts for every row.)
```

## Out of scope

- Changing the `meta.py` monitoring config (still works with
  `timestamp_format="date"` post-rebuild; no functional reason
  to change it in this cycle).
- Adding `total_btc` population to the writer (that's a
  separate decision about data sourcing).
- Index optimization (370 rows; PK on `date` covers all access
  patterns).

## Commit message for Step 1 (init_db + writer update; use verbatim)

```
Cycle 31 step 1: onchain_btc init_db + writer update for Rule 35

Updates engines/crypto_data_collector.py for full Rule 35
compliance on onchain_btc:

- init_db() CREATE TABLE: removes synthetic
  `id INTEGER PRIMARY KEY AUTOINCREMENT`; promotes existing
  UNIQUE(date) constraint to PRIMARY KEY (date); adds
  `timestamp INTEGER NOT NULL` and `datetime TEXT NOT NULL`
  columns for cross-table JOIN compatibility (ohlcv_daily
  uses the same convention -- UTC midnight ms).

- collect_onchain_btc writer: computes ts_ms + dt_iso from each
  date (UTC midnight) before INSERT, populating both new
  columns alongside `date`. INSERT OR REPLACE semantics
  unchanged.

Cycle 30 surfaced this gap when end-of-session scoreboard
review noted that onchain_btc still had `id` PK and lacked
timestamp/datetime columns. Closing it brings the migration
program to 11/11 temporal-indexed tables conforming -- no
exceptions for daily-grain.

Schema rebuild for the existing data runs in
scripts/migrations/cycle31_onchain_btc_schema_rebuild.py
(separate user-managed maintenance window step, this commit
only updates the on-disk schema definition for fresh DBs and
the writer for new rows).
```

## Commit message for Step 5 (close-out; use verbatim if no surprises)

```
Cycle 31 step 2: onchain_btc schema rebuild execution + doc trio updates

Rebuilt onchain_btc schema via scripts/migrations/cycle31_onchain_btc_
schema_rebuild.py during a brief maintenance window: removed
synthetic `id` PK, promoted `date` to PRIMARY KEY, added timestamp
+ datetime columns (UTC midnight ms; matches ohlcv_daily convention).
370 rows preserved 1:1; sub-second transaction. Cross-table JOIN
verification confirmed onchain_btc.timestamp matches ohlcv_daily.
timestamp for shared dates.

Doc trio updated: row #11 of SCHEMA_MIGRATION_PLAN.md added; trades
row in SCHEMA_NOTES.md updated to reflect compound PK on date;
claude/TODO.md scoreboard now reads 11/11 conforming. Migration
program scoreboard sentence "10/10 tables done" updated to "11/11
tables done; no exceptions for daily-grain". 
```
