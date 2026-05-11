# Cycle 34 -- Info Bars v0.1: library + DB schema + backfill + scheduled collector

**Predecessors:** Cycle 26 (`trades` table, ~14.8M rows post-Cycle-26
schema, BTC + ETH); Cycle 32 LSTM v2 TODO entry.
**Mode:** Hybrid. Files-included delta zip with the new module,
migration script, backfill script, collector batch, and PowerShell
task registration. Code applies edits, runs migration, runs initial
backfill, registers scheduled task. User runs registration as
admin.
**Risk:** medium. New writer + new scheduled task. Initial backfill
is a one-shot scan over ~14.8M trades (estimated 30-90s for the 4
bar types x 2 assets x 2 thresholds = 16 slices on a modern box).
Live collector cadence 5-minute; per-slice incremental from last
closed bar's `end_timestamp`.

## Why now

The LSTM v2 upgrade plan (Cycle 32 TODO entry) is gated on
information-driven bars per Lopez de Prado AFML Ch. 2. The
post-Cycle-26 `trades` table (`crypto_data.trades`, 14.8M rows
BTC + ETH, ms timestamps, pre-computed `quote_amount` and `side`)
is the raw trade tape we need. Cycle 34 ships the substrate:

- The 4 information-driven bar types from AFML Ch. 2: dollar
  bars, volume bars, volume-imbalance bars (VIB), volume-run
  bars (VRB).
- A single generic `info_bars` table -- threshold is a column,
  not baked into the table name (per the "everything is a
  parameter" standing convention).
- A backfill script for cold starts and one-off threshold
  sweeps.
- A scheduled collector for incremental updates as new trades
  arrive in `crypto_data.trades`.

Cycle 35 then mass-backfills the remaining 13 atlas experiments
with revival_hypotheses that can reference dollar bars
concretely ("re-run with dollar bars at $X threshold"). The
LSTM v2 work (Cycle 36+ probably) consumes the bars + applies
triple-barrier labeling in bar-index space.

## Design decisions baked in (worth reading)

**One generic table, not one per (bar_type, threshold)**. The
table is `info_bars` with columns including `bar_type` (string)
and `threshold_value` (real). Pros: arbitrary threshold sweeps
without schema changes; matches "everything is a parameter".
Cons: queries always filter by (asset, bar_type, threshold);
indexes need to reflect that.

**Closed bars only, never UPDATEd**. A bar enters the table
only after its threshold is crossed. The collector finds the
"latest closed bar end_timestamp" for each slice and scans
trades from there. Partial / open bars are recomputed from
scratch on the next run. No UPDATEs, idempotent re-runs.

**Late-trade safety lag**. The collector waits a configurable
number of seconds (default 30) before considering a window
"safe to close" -- prevents late-arriving trades being silently
dropped after their bar would have already been persisted.
Concretely: when reading trades for the live collector run,
filter `timestamp_ms < (now_ms - 30_000)`.

**Per-slice watermark via MAX(end_timestamp)**. No separate
watermark table. Simpler state model; the data IS the
watermark.

**Tick rule for VIB / VRB**. The `trades` table has both
`is_buyer_maker` and a `side` field. For aggressor direction
(needed for imbalance / run bars), `side='buy'` means the
buyer was the aggressor (sold to a maker bid) -- wait, verify
this. Praxis collector sets `side` based on Binance
`isBuyerMaker`: if `isBuyerMaker=true`, the trade was buyer-as-
maker so the seller was aggressor (sell-side aggression). The
`side` field in the table reflects taker direction: `side='buy'`
means a taker bought (positive aggression), `side='sell'` means
a taker sold (negative aggression). Confirm by reading the
collector source before relying on this. Code: please verify
by reading the trades-writer in `engines/crypto_data_collector.py`
or equivalent and document the truth in the InfoBars module
docstring; if my reading is wrong, flip the sign in the
imbalance calculations.

**Cadence: 5 minutes scheduled, not long-lived**. Per memory
entries #11, #13: short-lived run-and-exit collector wired into
Task Scheduler. Each run scans recent trades for each
configured (asset, bar_type, threshold) slice, computes any
new closed bars, INSERTs them, exits.

**Exit code honesty**. Per memory entry #12: if the collector
attempted to process N>0 expected rows but wrote 0 due to
transient errors (DB lock contention, etc.), exit code is
non-zero. If 0 rows expected (no new trades since last run),
exit code 0 is correct.

## What

Five deliverables:

1. **`engines/info_bars/` module** (4 bar builders + driver):
   - `__init__.py`
   - `bars.py` -- the four bar-building classes (`DollarBars`,
     `VolumeBars`, `VolumeImbalanceBars`, `VolumeRunBars`)
   - `writer.py` -- the trade-scanning + bar-closing logic;
     calls into bars.py per slice; INSERTs closed bars
   - `tests/test_bars.py` -- unit tests on small synthetic
     fixtures (no live DB required)

2. **DB schema migration** (`scripts/migrations/cycle34_info_bars_schema.py`):
   New table `info_bars` in `crypto_data.db` with Rule 35
   conformance. Idempotent.

3. **Backfill script** (`scripts/cycle34_backfill_info_bars.py`):
   One-shot scan of `trades` for a configurable set of
   `(asset, bar_type, threshold_value)` slices. Initial
   backfill scope:
   - BTC + ETH x dollar bars at $1M and $5M = 4 slices
   - BTC + ETH x volume bars at 100 base and 500 base = 4
     slices (BTC) and at 1000 + 5000 base = 4 slices (ETH)

   Adjust ETH thresholds because ETH's per-trade base amount
   is roughly 10-30x BTC's. Wait, that's backwards -- ETH
   prices are lower so trade *base amounts* are larger.
   Verify with a quick `praxis:get_trade_flow_summary` call
   for both before finalizing thresholds.

   - BTC + ETH x volume-imbalance bars at $500k expected
     imbalance threshold = 2 slices (start lean; can add
     more thresholds in follow-up)
   - BTC + ETH x volume-run bars at $500k expected run
     threshold = 2 slices

   Total initial backfill: ~12-16 slices. Expected runtime
   30-90s depending on slice count and machine.

4. **Scheduled collector** (`services/info_bars_collector_service.bat`
   + `services/register_info_bars_task.ps1`):
   - Cadence: every 5 minutes (`PraxisInfoBarsCollector` task)
   - Iterates `DISTINCT (asset, bar_type, threshold_value)`
     from `info_bars` -- so adding a new threshold via the
     backfill script auto-picks-up in the next live run
   - For each slice: read `MAX(end_timestamp)` (or 0 if no
     rows yet); read trades after that with the safety lag
     filter; feed them to the appropriate bar builder; INSERT
     any closed bars
   - Exit code: 0 if all slices ran cleanly (regardless of
     whether new bars closed); non-zero if any slice
     attempted to write N>0 expected bars but wrote 0 due to
     transient error

5. **`get_collector_health` monitoring** (extension to
   `servers/praxis_mcp/tools/meta.py`):
   Add `info_bars` to the monitored tables list with an
   appropriate staleness threshold. Suggested threshold: 30
   minutes (i.e. `is_stale=true` if no new bar in 30 min,
   which is generously above the 5-min cadence but tolerant
   of low-trade periods like Sunday early-AM).

   Use the same MAX(timestamp_ms) latency-style check that
   the other tables use. Inspect the existing tool to follow
   the established pattern.

## Out of scope

- Live updating of open / partial bars. Bars are closed-only;
  open bars get rebuilt each cycle.
- Tick-imbalance bars (TIB) and tick-run bars (TRB). The
  "volume" variants subsume them with better statistical
  properties per AFML Ch. 2. Skip TIB/TRB unless they prove
  needed later.
- Dynamic threshold adaptation (EWMA-of-recent-volume to set
  the threshold). AFML Ch. 2 discusses this; v0.1 uses fixed
  thresholds.
- LSTM consumption of info bars. That's Cycle 36+.
- Atlas mass backfill of remaining 13 experiments. Cycle 35.
- Cross-asset bars (concurrent BTC+ETH bars). Single-asset
  only in v0.1.

## Specifics for Code

### Step 1: Schema migration

```powershell
python scripts\migrations\cycle34_info_bars_schema.py
```

The script creates `info_bars` in `data/crypto_data.db` with this
schema (Rule 35 conforming):

```sql
CREATE TABLE IF NOT EXISTS info_bars (
    asset            TEXT    NOT NULL,
    bar_type         TEXT    NOT NULL,           -- 'dollar', 'volume', 'vib', 'vrb'
    threshold_value  REAL    NOT NULL,           -- dollar amount or base-asset amount; semantics depend on bar_type
    bar_index        INTEGER NOT NULL,           -- 0-indexed monotonic within (asset, bar_type, threshold_value)
    start_timestamp  INTEGER NOT NULL,           -- ms UTC, first trade in bar
    end_timestamp    INTEGER NOT NULL,           -- ms UTC, last trade in bar
    start_datetime   TEXT    NOT NULL,           -- ISO 8601 with +00:00
    end_datetime     TEXT    NOT NULL,           -- ISO 8601 with +00:00
    open             REAL    NOT NULL,
    high             REAL    NOT NULL,
    low              REAL    NOT NULL,
    close            REAL    NOT NULL,
    base_volume      REAL    NOT NULL,           -- total base-asset units traded in bar
    quote_volume     REAL    NOT NULL,           -- total dollar amount traded in bar
    tick_count       INTEGER NOT NULL,           -- number of trades in bar
    buy_quote        REAL    NOT NULL,           -- aggressor-buy dollars in bar
    sell_quote       REAL    NOT NULL,           -- aggressor-sell dollars in bar
    imbalance_quote  REAL    NOT NULL,           -- buy_quote - sell_quote (positive = buy-aggressed)
    PRIMARY KEY (asset, bar_type, threshold_value, bar_index)
);

CREATE INDEX IF NOT EXISTS idx_info_bars_lookup
    ON info_bars (asset, bar_type, threshold_value, end_timestamp);

CREATE INDEX IF NOT EXISTS idx_info_bars_end_ts
    ON info_bars (end_timestamp);
```

The PK guarantees idempotency: re-inserting the same bar fails on
PK conflict, which the writer handles as "already there, skip".

### Step 2: `engines/info_bars/bars.py`

Four classes with a common interface. Pseudocode for the
contract:

```python
class BarBuilder:
    """Stateful bar builder. Push trades in order; receive closed
    bars when threshold is crossed."""

    def __init__(self, asset: str, threshold_value: float): ...

    def push(self, trade: Trade) -> Optional[ClosedBar]:
        """Push a single trade. Returns a ClosedBar if this trade
        crossed the threshold and closed a bar; otherwise None."""

    def flush_partial(self) -> Optional[OpenBar]:
        """Return the partial-state bar that has accumulated since
        the last close (for inspection only; collector does NOT
        persist this)."""


class DollarBars(BarBuilder):
    """Closes when cumulative quote_amount >= threshold_value."""

class VolumeBars(BarBuilder):
    """Closes when cumulative base amount >= threshold_value."""

class VolumeImbalanceBars(BarBuilder):
    """Closes when cumulative SIGNED quote (buy - sell) crosses
    +threshold_value OR -threshold_value. Bar carries direction
    sign in addition to imbalance magnitude."""

class VolumeRunBars(BarBuilder):
    """Closes when the running aggressor-side run (consecutive
    same-direction quote) exceeds threshold_value. The 'run' is
    the max of buy_run, sell_run computed against signed
    cumulative quote."""
```

The reference for VIB/VRB math is AFML Ch. 2 Sections 2.3.2.2
and 2.3.2.3. Lopez de Prado's notation uses theta_T for the
cumulative signed sum; the bar closes when |theta_T| crosses
threshold (VIB) or when one-sided run exceeds threshold (VRB).

For v0.1 keep VRB simple: the "run" is the larger of cumulative
buy_quote and cumulative sell_quote since the last bar close.
When max(buy_quote, sell_quote) >= threshold_value, close the
bar. AFML's more sophisticated version uses an adaptive
expected-run estimator; we'll add that in v0.2 if needed.

ClosedBar fields must include everything in the `info_bars`
schema (asset, threshold_value, start/end timestamps and
datetimes, OHLC, volumes, tick count, buy/sell quotes,
imbalance). `bar_index` is assigned by the writer, not the
builder (the writer knows what was previously persisted).

### Step 3: `engines/info_bars/writer.py`

Top-level driver. Public entrypoints:

```python
def backfill_slice(
    db_path: str,
    asset: str,
    bar_type: str,
    threshold_value: float,
    start_ts_ms: Optional[int] = None,
    end_ts_ms: Optional[int] = None,
) -> dict:
    """Backfill a single slice. Reads trades from `trades` table
    in the given time range (defaults to full table), feeds them
    into the appropriate BarBuilder, INSERTs closed bars.
    Returns {'slice': (asset, bar_type, threshold_value),
             'closed_bars': N, 'trades_processed': M,
             'first_bar_start': ts, 'last_bar_end': ts}."""


def live_update(
    db_path: str,
    safety_lag_seconds: int = 30,
) -> dict:
    """For each DISTINCT (asset, bar_type, threshold_value) in
    info_bars, find the last closed bar's end_timestamp, read
    trades after that (with safety lag filter), feed into the
    correct BarBuilder, INSERT any newly-closed bars.

    Returns {'slices_processed': N,
             'slices_with_new_bars': M,
             'total_new_bars': K,
             'errors': [list of (slice, error_msg)]}."""
```

The writer must:
- Read trades in chunks (don't load 14M rows at once); chunk
  size 100k rows is a reasonable starting point.
- Process trades in **(timestamp ASC, trade_id ASC)** order.
  Same timestamp_ms can have multiple trades; trade_id is
  the secondary ordering.
- Use a single transaction per slice for the INSERTs (commit
  at end of slice; rollback on error).
- Skip trades whose `timestamp_ms <= last_end_timestamp_ms`
  to avoid re-bucketing already-closed bars. (The PK would
  catch duplicates anyway, but skipping is faster.)
- Honor the safety lag in `live_update` only, not in
  `backfill_slice` (backfill operates on historical data
  where late arrivals aren't a concern).

Idempotency: re-running `backfill_slice` on the same range with
the same threshold reproduces the same bars (deterministic
given trade order); PK conflicts on re-INSERT silently skip.

### Step 4: `engines/info_bars/tests/test_bars.py`

Tiny test fixtures using synthetic trades (no DB). Cover:

- DollarBars: 3 trades of $400k each at threshold $1M -> first
  bar closes at the 3rd trade ($1.2M), second bar opens with
  remainder. (Actually: trade 3 triggers close; bar contains
  all 3 trades; the $200k overshoot stays with bar 1 -- AFML's
  convention; document this clearly).
- VolumeBars: equivalent test in base-asset units.
- VolumeImbalanceBars: a sequence of buy-aggressor trades that
  pushes signed cumulative quote above +threshold -> closes a
  +direction bar.
- VolumeRunBars: a sequence of buy-aggressor trades whose
  cumulative quote exceeds threshold -> closes a buy-run bar.
- Idempotency: feeding the same trade list twice should
  produce the same bars (modulo internal state reset).
- Partial-bar handling: `flush_partial` returns an OpenBar
  with the right OHLC/volume but the caller does NOT persist
  it.

Tests run under pytest; the `engines/info_bars/tests/` location
follows the established pattern (e.g., other engines/ tests).
If there's no shared `conftest.py` infrastructure for engines/
yet, just inline minimal fixtures.

### Step 5: `scripts/cycle34_backfill_info_bars.py`

CLI flags:

```
--asset {BTC|ETH|all}        default: all
--bar-type {dollar|volume|vib|vrb|all}   default: all
--threshold-set {default|<custom JSON>}  default: 'default'
--validate                    dry-run: counts only, no INSERTs
--verbose
```

The "default" threshold set is:

```python
DEFAULTS = {
    "dollar": {
        "BTC": [1_000_000, 5_000_000],
        "ETH": [1_000_000, 5_000_000],
    },
    "volume": {
        "BTC": [100, 500],         # base-asset BTC units
        "ETH": [1000, 5000],       # base-asset ETH units (verify scale!)
    },
    "vib": {
        "BTC": [500_000],
        "ETH": [500_000],
    },
    "vrb": {
        "BTC": [500_000],
        "ETH": [500_000],
    },
}
```

Code should `praxis:get_trade_flow_summary` for both BTC and ETH
to verify the volume thresholds are in the right ballpark before
running the full backfill. If a threshold would yield <50 closed
bars across the full 12-day trade window, log a warning -- it's
either too coarse for the data, or fine for low-cadence work but
suspicious for v0.1.

Output: per-slice summary table with closed_bars count,
trades_processed, first/last bar timestamps, slice runtime.

### Step 6: `services/info_bars_collector_service.bat`

Standard batch file pattern (matches existing
`services/fear_greed_collector_service.bat`,
`services/onchain_collector_service.bat`):

```batch
@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "C:\Data\Development\Python\McTheoryApps\praxis"
call .venv\Scripts\activate.bat
python -u -m engines.info_bars.writer --live
exit /b %errorlevel%
```

The `--live` flag dispatches to `live_update()`. The writer
module needs a `__main__` block that parses `--live` / `--backfill`
flags (or this can live in a wrapper script; Code decide what
fits the module structure best).

Strip all non-ASCII output per memory #12 (Windows pipes
through cp1252). Use `python -u` for unbuffered stdout. Set
`PYTHONUTF8=1`.

### Step 7: `services/register_info_bars_task.ps1`

Standard PowerShell registration pattern. Cadence: every 5 min,
starting at the next 5-min boundary after registration. Task
name `PraxisInfoBarsCollector`. Logon type: ServiceAccount or
the current user (match existing patterns).

User runs this as Administrator. Confirm with
`Get-ScheduledTask -TaskName PraxisInfoBarsCollector` after
registration that `LastTaskResult` is 267011 (task ready, not
yet run) initially, transitioning to 0 after first successful
run.

### Step 8: `servers/praxis_mcp/tools/meta.py` -- monitor info_bars

Add `info_bars` to the per-DB monitored tables config in
`get_collector_health`. Pattern from `ohlcv_1m` etc.:

- `staleness_threshold_seconds`: 1800 (30 min) -- generous vs.
  the 5-min cadence
- `timestamp_format`: 'ms' (Rule 35 default)
- `timestamp_column`: 'end_timestamp' (we monitor the most
  recent CLOSED bar's end timestamp -- not start, because
  newly-closed bars have a relevant end_timestamp)

Read the existing config dict structure carefully; the exact
key names matter and have to match what the runtime expects.

### Step 9: atlas_sync, regime_engine, etc. unaffected

No changes to `engines/atlas_sync.py`, `engines/regime_engine.py`,
the existing trades collector, or anything in the Atlas. Info
bars is a new branch of the engine framework that lives
parallel to the existing infrastructure. Future cycles will
add consumers (LSTM, atlas revival re-runs); v0.1 just builds
the substrate.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | `python scripts/migrations/cycle34_info_bars_schema.py` runs idempotently; PRAGMA table_info(info_bars) shows the schema above |
| 2 | `engines/info_bars/` module imports cleanly; py_compile clean across module + tests |
| 3 | `pytest engines/info_bars/tests/` passes |
| 4 | `python scripts/cycle34_backfill_info_bars.py --validate` reports projected closed-bar counts for all default slices without writing |
| 5 | `python scripts/cycle34_backfill_info_bars.py` runs to completion; info_bars table populated with bars for all default slices (10+ bars per slice expected over 12-day window) |
| 6 | Spot-check a closed bar: SUM(quote_amount) of its constituent trades equals bar.quote_volume (within float tolerance); bar.high = MAX(price), bar.low = MIN(price), bar.open = first trade price, bar.close = last trade price |
| 7 | Scheduled task `PraxisInfoBarsCollector` registered; runs at 5-min cadence; LastTaskResult=0 after first run |
| 8 | `praxis:get_collector_health` shows `info_bars` with `is_stale=false` after first scheduled run |
| 9 | Second backfill run reports 0 new bars inserted (idempotency via PK) |
| 10 | Live update writes new bars when new trades arrive; old bars unchanged |

## Step ordering (load-bearing)

1. Schema migration.
2. `engines/info_bars/` module (bars + writer + tests).
3. `pytest engines/info_bars/tests/` passes BEFORE any DB writes.
4. Backfill script `--validate` run first; check projected counts
   look sane (10s to 1000s of bars per slice over 12-day window).
5. Backfill script real run.
6. Spot-check criterion #6 manually (or as part of test suite).
7. Batch + PowerShell scripts.
8. Monitoring config in meta.py.
9. User registers scheduled task as admin.
10. Wait for first scheduled run; verify health.
11. Commit + push.

If any step 1-6 fails, no scheduled task gets registered.

## Process pattern (per memory entries #11, #13)

`PraxisInfoBarsCollector` is a **scheduled short-lived task**.
Run-and-exit pattern. NOT a long-lived loop process. Future
maintenance windows should verify the process pattern before
disabling the scheduled task (per memory #13).

## Commit messages

### Commit 1: schema + library + backfill

```
Cycle 34: Info Bars v0.1 -- library + DB schema + backfill

Adds the engines/info_bars/ module implementing the four
information-driven bar types from Lopez de Prado AFML Ch. 2:

- DollarBars: close when cumulative quote_amount >= threshold
- VolumeBars: close when cumulative base amount >= threshold
- VolumeImbalanceBars: close when |signed cumulative quote|
  crosses threshold (preserves direction sign)
- VolumeRunBars: close when the larger of cumulative
  buy_quote / sell_quote since the last close exceeds
  threshold

The bar builders are stateful, single-asset, push-trades-in
streaming objects. The writer module drives the builders
from the trades table and persists closed bars.

Adds a single generic info_bars table to crypto_data.db with
Rule 35 conformance (start_timestamp + end_timestamp in ms,
matching ISO 8601 datetimes with +00:00). bar_type and
threshold_value are columns -- threshold sweeps don't
require new tables.

Adds scripts/cycle34_backfill_info_bars.py for cold-start /
threshold-sweep backfills. Default initial slices: BTC + ETH
x dollar at $1M and $5M; volume bars at asset-appropriate
base amounts; VIB and VRB at $500k expected imbalance/run.

Backfill is idempotent via PK conflict skip; spot-check
criterion: SUM(constituent trade quote) == bar.quote_volume
within float tolerance.

Live collector (Cycle 34 commit 2) and atlas mass backfill
(Cycle 35) come next.
```

### Commit 2: scheduled collector + monitoring

```
Cycle 34 step 2: scheduled live collector + monitoring

Adds services/info_bars_collector_service.bat and
services/register_info_bars_task.ps1. Task name
PraxisInfoBarsCollector; cadence 5 minutes. The collector
runs engines.info_bars.writer in --live mode, which iterates
DISTINCT (asset, bar_type, threshold_value) from info_bars
and incrementally appends newly-closed bars since each
slice's last persisted end_timestamp.

Late-trade safety lag: 30 seconds. Trades with timestamp_ms
>= now_ms - 30_000 are excluded from the current run so
late-arriving trades aren't silently dropped after their bar
would have closed.

Exit code: 0 if all slices ran without error (regardless of
new bars closed); non-zero if any slice attempted writes but
got 0 due to transient errors. Per memory #12 (scheduled
collector exit-code honesty).

Extends servers/praxis_mcp/tools/meta.py to monitor info_bars
in get_collector_health. Staleness threshold 1800s (30 min);
timestamp_column end_timestamp.

Adding a new threshold = run a one-shot backfill for that
slice; live collector picks it up automatically next run via
the DISTINCT query.

After this commit + first scheduled run, praxis monitored
table count is 12 (was 11 post-Cycle 31). All 12 should
report is_stale=false.
```

## Post-cycle status

- [x] Schema migration program complete (Cycles 17-31)
- [x] Pre-Cycle-17 cleanup queue closed (Cycle 32)
- [x] Atlas schema extended + 7-engine taxonomy (Cycle 33)
- [x] MCP layer exposes new structured fields (Cycle 33.5)
- [x] **Info Bars v0.1: library + table + backfill + collector (Cycle 34)**
- [ ] Cycle 35: Atlas mass backfill of remaining 13 experiments
  (now genuinely useful with info bars available; Engine 2
  revival hypotheses can reference dollar bars concretely)
- [ ] Cycle 36+: First info-bar revival re-runs targeting
  highest-likelihood revival hypotheses
- [ ] LSTM v2: info bars + triple-barrier labeling + DL
  architecture refresh (Cycle 37+)
