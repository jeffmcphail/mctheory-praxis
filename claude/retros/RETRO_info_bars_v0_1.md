# Retro: Cycle 34 -- Info Bars v0.1

**Brief:** `claude/handoffs/BRIEF_info_bars_v0_1.md`
**Date:** 2026-05-11
**Mode:** Hybrid (Claude drafted brief + schema migration script
+ design decisions; Code wrote engines/info_bars/ module +
backfill + collector + monitoring extension; user registered
scheduled task as admin)
**Status:** DONE
**Predecessors:**
- Cycle 26 (`trades` table schema; ~14.8M rows BTC + ETH by
  Cycle 34 start)
- Cycle 32 (LSTM v2 upgrade plan added to TODO.md
  State/context)
**Commits:**
- `808f19e` -- library + schema + backfill
- `647d50c` -- scheduled collector + monitoring

---

## Summary

Ships information-driven bars per Lopez de Prado AFML Ch. 2 as
production infrastructure:

1. `engines/info_bars/` module with 4 bar builders (Dollar,
   Volume, VolumeImbalance, VolumeRun) + writer driver.
2. Single generic `info_bars` table in `crypto_data.db`
   (Rule 35 conforming; threshold parameterized as a column).
3. Backfill script for initial slice population.
4. Scheduled live collector (`PraxisInfoBarsCollector`, 5-min
   cadence) for incremental updates.
5. `get_collector_health` monitoring extended to cover
   `info_bars` (12 monitored tables now; was 11 post-Cycle-31).

Initial backfill populated 12 slices with 77,196 total closed
bars across the 12-day trade window
(2026-04-29T19:23:54+00:00 -> 2026-05-11T23:14:xx+00:00 at
backfill completion; live collector then advanced latest end
to 2026-05-11T23:37:51+00:00 by health-check time). Live
collector first run at 2026-05-11 19:38:33 local (23:38:33
UTC).

Net change:
- `engines/info_bars/` -- 874 lines (26 __init__ + 242 bars
  + 357 writer + 249 test_bars)
- `scripts/migrations/cycle34_info_bars_schema.py` -- 102 lines
- `scripts/cycle34_backfill_info_bars.py` -- 162 lines
- `services/info_bars_collector_service.bat` + `register_*.ps1`
  -- 95 lines (23 .bat + 72 .ps1)
- `servers/praxis_mcp/tools/meta.py` -- 5 lines

---

## Why this matters

Cycle 34 is the gating dependency for the LSTM v2 upgrade plan
captured in Cycle 32's TODO.md addition. The Financial
Innovation (Feb 2025) paper documents that "new" LSTM crypto
models work where v1-style models didn't, attributing the
difference largely to (a) information-driven bars instead of
time bars and (b) triple-barrier labeling. Cycle 34 ships
(a); Cycle 37+ work will combine it with triple-barrier
labeling and a refreshed deep-learning architecture.

Cycle 34 is also a near-term enabler for Cycle 35 atlas mass
backfill: Engine 2 revival hypotheses for the NEGATIVE TA
experiments can now reference dollar bars concretely as a
specific revival mechanism, instead of as an abstract
possibility.

---

## Design decisions (preserved for reference)

### Single generic table vs. one table per (bar_type, threshold)

Chose generic. Pros: arbitrary threshold sweeps without
schema changes; matches the "everything is a parameter"
standing convention. Cons: every query filters by (asset,
bar_type, threshold_value); the `(asset, bar_type,
threshold_value, end_timestamp)` index makes this efficient.

### Closed bars only, never UPDATEd

A bar enters the table only after threshold is crossed.
Re-running backfill on the same range produces identical bars
(deterministic given trade order); PK conflicts on re-INSERT
silently skip. The collector finds each slice's last
`end_timestamp` and starts from there. No UPDATE statements
on info_bars anywhere in the codebase.

### Late-trade safety lag (30s default)

Live collector excludes trades with `timestamp_ms >= now_ms -
30_000` to avoid silently dropping late-arriving trades that
would have belonged in an already-"closed" bar. Trade-off:
the most-recent ~30s of trade activity is always pending,
which is fine for downstream consumers (LSTM training,
revival re-runs operate on historical data; live predictions
will use the most-recent CLOSED bar, not partial).

### Per-slice watermark via MAX(end_timestamp)

No separate watermark table. The data IS the watermark.
Simpler state model; corruption-resistant (state can only be
"wrong" if the bars themselves are wrong, in which case the
watermark issue is moot).

### Scheduled task vs. long-lived process

Per memory entries #11, #13: scheduled short-lived run-and-exit
task. Each run reads recent trades for each configured slice,
appends any newly-closed bars, exits. No long-lived loop
process to manage. Future maintenance windows should verify
the process pattern before disabling the task (memory #13).

### Exit code honesty

Per memory entry #12: exit code 0 only if all slices ran
cleanly. If a slice attempted to write N>0 expected bars but
wrote 0 due to transient errors (DB lock contention, etc.),
exit code is non-zero. Distinguishes "ran cleanly with 0 new
bars (no new trades since last run)" from "ran cleanly with 0
new bars when N>0 expected (silent failure)".

---

## Execution log

### Step 1: Schema migration

```powershell
python scripts\migrations\cycle34_info_bars_schema.py
```

Output: "Post-state: 18 columns, 3 indexes. INFO_BARS TABLE
CREATED" (from initial local run). Idempotent confirmation:
re-running reports "info_bars already exists; indexes
ensured."

### Step 2: engines/info_bars/ module

874 lines across `bars.py`, `writer.py`, `tests/test_bars.py`
(plus `__init__.py`). py_compile clean. Module structure:
bars.py has 4 separate builder classes (DollarBars,
VolumeBars, VolumeImbalanceBars, VolumeRunBars) + ClosedBar
dataclass + build_for factory. writer.py has backfill_slice()
+ live_update() + __main__ for --live / --backfill flag
dispatch. tests/test_bars.py covers all 4 builders + boundary
cases (13 tests).

Notable implementation detail caught during development:
Aggressor-direction sign verified against
engines/crypto_data_collector.py:980-983 -- brief's reading
was correct (side='buy' = taker bought = +1 aggression). No
sign flips needed. Convention documented in module docstring
per brief.

### Step 3: pytest engines/info_bars/tests/

```powershell
pytest engines/info_bars/tests/ -v
```

Output: 13 passed. Initial test had a `flush_partial` ->
`push_all` confusion; fixed inline before commit 1.

### Step 4: backfill --validate

```powershell
python scripts/cycle34_backfill_info_bars.py --validate
```

Projected closed-bar counts: 12 slices, 77,184 bars projected,
range 338 (BTC volume 500) to 18,638 (BTC vrb $500k). No
sub-50 warnings -- all slices well above the 50-bar minimum
guidance.

### Step 5: backfill real run

```powershell
python scripts/cycle34_backfill_info_bars.py
```

Output: 12 slices, 77,196 total bars actually inserted (within
12 of projected -- difference is rounding of partial bars at
the trade-table boundary). Total runtime 281s real
wall-clock. 77,196 bars across 12 slices.

### Step 6: spot-check

For two closed bars, verified manually:

```sql
SELECT ... -- spot check the constituent trades sum to bar fields
```

Result for bar_index=0 (BTC dollar $1M): tick_count=721,
SUM(constituent quote_amount) = bar.quote_volume to 2e-16 ULP,
SUM(base) exact, OHLC + start/end_timestamp + buy/sell_quote
split all exact match. Mid-bar spot check (bar_index=N>0)
showed 361 trades in [start_ts, end_ts] window vs 357 in
tick_count; this is correct ms-boundary-tie behavior (same-ms
trades split across adjacent bars by trade_id order);
bar_index=0 has no left-edge tie ambiguity.

### Step 7-8: services + monitoring

`info_bars_collector_service.bat`, `register_info_bars_task.ps1`
written. `servers/praxis_mcp/tools/meta.py` extended with
`info_bars` monitoring entry (1800s staleness threshold,
`end_timestamp` column).

### Step 9: task registration

User ran `register_info_bars_task.ps1` as Administrator.
`Get-ScheduledTask -TaskName PraxisInfoBarsCollector`:
TaskName PraxisInfoBarsCollector, State Ready, registered
every 5 min starting 19:38:33 local. First run completed by
23:40 UTC with 42 new bars closed. LastTaskResult=0 after
first run at 2026-05-11 19:38:33 local (23:38:33 UTC).

### Step 10: health verification

```
praxis:get_collector_health
```

Output (info_bars portion): At 23:40:32 UTC,
info_bars row_count=77247, latest=23:37:51, staleness=161s,
threshold=1800s, is_stale=false. All 10 crypto_data.db
monitored tables is_stale=false. unmonitored=[]. Total
monitored tables in crypto_data.db: 10 (was 9; +info_bars per
the brief). Including the sidecar DBs (live_collector,
smart_money): 12 monitored tables total. is_stale=false.
Total monitored tables: 12 (was 11).

[Note: the post-Cycle-31 count of "11 monitored" referenced in
brief was incorrect by one; the correct pre-Cycle-34 count was
9 primary + 2 sidecar = 11 total. Cycle 34 brings it to 10 +
2 = 12. Math holds.]

### Step 11: idempotency check

Re-running BTC volume 500 slice: trades_processed=6864,
closed_bars=0, inserted=0. PK conflict skip confirmed; no
duplicate bars inserted.

---

## Notes

### Threshold calibration findings

Brief defaults landed without adjustment. Pre-flight
verification via direct SQL on trades (BTC 12.15d span,
$13.5B quote vol, 7.9M trades; ETH similar scale) confirmed
projected bar counts of 338 to 18,638 across all slices --
well above the 50-bar minimum guidance. No threshold changes
needed. Documented in the writer module docstring for future
reference.

### Aggressor-direction sign verification

Brief's assertion verified directly against
engines/crypto_data_collector.py:980-983. `side='buy'` when
`isBuyerMaker=False` (taker bought, positive aggression);
`side='sell'` when `isBuyerMaker=True` (taker sold, negative
aggression). VolumeImbalanceBars and VolumeRunBars use this
sign correctly without flipping.

### Module layout decision

Implemented as 4 separate builder classes (DollarBars,
VolumeBars, VolumeImbalanceBars, VolumeRunBars) sharing a
common stateful push-trades interface, rather than a single
parameterized BarBuilder. Cleaner inheritance for per-type
test fixtures; easier to extend with TIB/TRB or adaptive
variants in v0.2 without breaking existing call sites.

### Commit-message heredoc escaping gotcha

Commit 808f19e shipped with the literal string `M -> \$1M` in
the body where the intent was `$1M`. Root cause: bash
heredoc backslash-expansion when the commit message body
contains `$` characters -- the shell expanded `$1` as a
positional parameter (empty), and the escape sequence
artifacted into the persisted message. soft-reset to fix was
blocked by the auto-classifier; decision was to leave
history as-is since semantic meaning is preserved.

For future cycles: when the commit message contains `$`
characters, either use `git commit -m 'msg'` with single
quotes (and escape internally if needed), or use `git commit
-F <file>` to read the message from a file. Avoid the
double-quoted heredoc pattern for any message containing `$`.

### MCP server restart pattern revisited

The initial post-commit MCP-side verification showed
info_bars under `unmonitored=[...]` even after commit 647d50c
landed -- same stale-server pattern observed in Cycle 33.5's
`atlas_get` gap. Diagnosed correctly via direct in-process
call to the updated `get_collector_health` (which showed
info_bars present and is_stale=false) before landing any
no-op "fix" commits.

Memory entry #17 added to formalize: MCP server module code
requires Claude Desktop restart to be visible after any
`servers/` edit. The MCP runtime caches the imported modules
at server startup; edits to `servers/praxis_mcp/tools/*.py`
do not hot-reload. Symptom is always the same: tool returns
pre-edit behavior while the on-disk code shows the new
behavior. Always verify with an in-process Python call
before suspecting a code bug.

### Future v0.2 improvements (not in scope)

- Tick-imbalance bars (TIB) and tick-run bars (TRB) -- subsumed
  by volume variants per AFML Ch. 2 but available if needed.
- Adaptive threshold (EWMA-of-recent-volume) per AFML Ch. 2
  Section 2.3.2.1.
- Cross-asset bars (concurrent BTC+ETH bars closed by joint
  threshold).
- Open / partial bar persistence with `closed` flag (for
  real-time consumers).

---

## Open items / next cycle inputs

- **Cycle 35**: Atlas mass backfill of remaining 13
  experiments. Info bars now available; Engine 2 revival
  hypotheses for the NEGATIVE TA experiments can reference
  dollar bars concretely as a specific revival mechanism.
- **Cycle 36+**: First info-bar revival re-runs. Highest-
  likelihood candidates from the backfilled Atlas
  (post-Cycle-35) that name info bars as the revival lever.
- **Cycle 37+: LSTM v2 upgrade.** Info bars + triple-barrier
  labeling (from intrabar v8.1 XGBoost discipline, ported to
  LSTM) + deep-learning architecture refresh per Financial
  Innovation Feb 2025 paper.
- **Threshold sweep**: once revival re-runs identify which
  thresholds matter, additional thresholds can be backfilled
  via the same `scripts/cycle34_backfill_info_bars.py` script
  -- the live collector picks them up automatically via the
  DISTINCT (asset, bar_type, threshold_value) query.
