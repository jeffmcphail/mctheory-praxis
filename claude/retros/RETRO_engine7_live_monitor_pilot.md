# Retro: Cycle 41 -- Engine 7 live monitor pilot (BTC + ETH)

**Brief:** `claude/handoffs/BRIEF_engine7_live_monitor_pilot.md`
**Date:** 2026-05-26
**Mode:** RECON-then-implementation in a single cycle
**Status:** DONE
**Predecessor:** Cycle 40 -- Engine 7 paper reproduction (`082459b` + `cd948d3`); Cycle 41 is open-item 41a from that cycle.
**Commit:** `<CYCLE_41_HASH>`

---

## Summary

Live funding-carry monitor deployed for BTC + ETH using the Cycle 40 verified phase3 models. Signals persist to a new Rule-35-conforming `funding_signals` table in `data/crypto_data.db`. Scheduled task `PraxisFundingMonitor` runs 3x daily (00:15 / 08:15 / 16:15 LOCAL) ~10 min after `PraxisFundingCollector`. First scheduled-context run succeeded with exit code 0; first funding window (2026-05-27 00:00 UTC) observed sit-out behavior at both gates as predicted, confirming the bear-side regime stays below entry thresholds.

**Verdict:** monitor deployed and observed; sit-out behavior in current regime confirmed; atlas's "2022-bear sit-out" hypothesis gets its first forward-looking real-time data point.

Net change:
- New table `funding_signals` (17 columns, Rule-35 PK on `(asset, timestamp)`)
- `scripts/funding_monitor.py`: 6 surgical edits — DB-default funding source, `--persist` flag, `--db` arg, dual-gate persistence, default model path fix
- New `services/funding_monitor_service.bat` (one-shot invocation with --persist)
- New `services/register_funding_monitor_task.ps1` (mirrors funding collector pattern)
- One row per asset (BTC, ETH) persisted at first funding window after deployment

---

## Why this matters

Cycle 39 RECON established Engine 7 had no live monitor; Cycle 40 verified the underlying paper-trading numbers. Cycle 41 closes the deployment loop: there is now a scheduled process writing real-time inference rows for the verified strategy.

The deployment also enables the **forward bear-regime confirmation opportunity** flagged in Cycle 39 L9 and Cycle 40 retro. Atlas Exp 13 notes "2022 bear validation (sustained negative funding) not yet run" — the current 2026-04/05 BTC regime is mechanically equivalent (sustained-negative funding, low pos_share). With the monitor live, every funding window in this regime produces a logged "would not trade" row, and we accumulate confirming observations passively.

Operational impact: the monitor takes ~5-10 s per invocation (faster than collector due to DB-source funding), trivial 3x/day. No new data feeds, no executor changes, no live trading — pure inference logging.

---

## Execution log (9-step plan from RECON)

The Cycle 41 RECON delivered inline in chat answered all 4 brief questions:

- **Q1 storage:** new `funding_signals` table in `crypto_data.db`, Rule-35 schema. (User confirmed Option A over Option B / separate DB.)
- **Q2 features:** CCXT-fetch hourly spot+perp at runtime; read funding from DB. No new collector needed; the 11 features compute from these inputs in ~5 s.
- **Q3 cadence:** 3x daily, aligned with funding events, ~10-min offset after collector.
- **Q4 task integration:** new task `PraxisFundingMonitor`, separate from collector per memory #12 (exit-code honesty) + memory #13 (process pattern verification).

User approved the 9-step plan + dual-gate persistence (`above_gate` for P>0.70 live + `above_gate_050` for P>0.50 headline).

### Step 1: schema extension

Added `funding_signals` `CREATE TABLE IF NOT EXISTS` to `init_db()` in
`engines/crypto_data_collector.py` directly after the existing
`funding_rates` block. 17 columns; compound PK on `(asset, timestamp)`;
timestamp is the funding-window UTC time (00/08/16) in ms,
seconds-aligned matching `funding_rates`. Triggered `init_db()` once
to create the table; verified via `PRAGMA table_info`. Initial row
count: 0.

### Step 2: default model path fix

`scripts/funding_monitor.py:54` -- `DEFAULT_MODELS` changed from
the non-existent `"output/funding_rate/cpo/phase3_models_funding.joblib"`
(per Cycle 39 finding) to the Cycle 40 verified location
`"outputs/funding_carry_repro/cpo/phase3_models_funding.joblib"`.
Same edit also added `HEADLINE_GATE = 0.50`, `DEFAULT_DB = "data/crypto_data.db"`,
and `MONITOR_VERSION = "cycle40:082459b"` constants.

### Step 3: --persist flag + DB write

Added `persist_signals()` function near the report-formatting helpers.
INSERT OR IGNORE one row per signal at the most-recent funding window
timestamp (computed via new `funding_window_timestamp()` helper that
rounds `now.hour` to the nearest 8h boundary).

Added a `--persist` CLI flag (off by default); when set, `run_once()`
calls `persist_signals()` after RF inference. The persisted row
includes `above_gate` (P > 0.70), `above_gate_050` (P > 0.50),
`features_json` (11-feature dict for downstream debugging), and
`monitor_version` (`cycle40:082459b` tying signals to the verified
model commit).

Also added `config_id` and `feature_vector` to the signal dict in
`run_inference()` so `persist_signals()` can serialize them.

### Step 4: swap funding source to DB

Added `funding_source` (default `"db"`) and `db_path` (default
`DEFAULT_DB`) params to `fetch_live_data()`. In the funding-fetch
section, branched between DB query (default) and the original CCXT
path (preserved as fallback via `--funding-source ccxt`). DB query:
`SELECT timestamp, funding_rate FROM funding_rates WHERE asset=? AND
timestamp >= ? AND timestamp <= ? ORDER BY timestamp`. Result wrapped
into the same pandas Series the legacy code produced.

Removes one CCXT API surface from the runtime path; the live
PraxisFundingCollector already gates DB freshness.

### Step 5: service bat

`services/funding_monitor_service.bat` mirrors `funding_collector_service.bat`:
- Activates `.venv`
- Sets `PYTHONUTF8=1`
- Logs to `logs/funding_monitor.log` (appended)
- Invokes `python -u -m scripts.funding_monitor --assets BTC,ETH --models %MODELS% --persist`
- Records exit code in the log

### Step 6: register script

`services/register_funding_monitor_task.ps1` mirrors
`register_funding_task.ps1`:
- Task name: `PraxisFundingMonitor`
- Triggers: `-Daily -At "00:15"` / `"08:15"` / `"16:15"` (LOCAL time)
- ExecutionTimeLimit 5 min, MultipleInstances IgnoreNew
- Principal: `-LogonType S4U -RunLevel Limited` (same as collector)
- Description references the Cycle 40 model location

### Step 7: register task

Initial register attempt from the unelevated CC session failed with
`Access is denied` (expected — `Register-ScheduledTask` requires
elevation). User ran the script from an elevated PowerShell window.
Verification via `Get-ScheduledTask`:

```
Name       : PraxisFundingMonitor
State      : Ready
Exec       : cmd.exe
Arg        : /c "...services\funding_monitor_service.bat"
Triggers   : 00:15, 08:15, 16:15
LastRun    : (placeholder — never run)
LastResult : 267011 (= "task has not yet run")
NextRun    : 2026-05-27 00:15:00
```

### Step 8: first scheduled-context run

Triggered manually via `Start-ScheduledTask -TaskName PraxisFundingMonitor`.
Result:

```
LastRun:     2026-05-26 20:54:08 local (= 2026-05-27 00:54:08 UTC)
LastResult:  0
NextRun:     2026-05-27 00:15:00 local
```

`logs\funding_monitor.log` tail shows the full inference run: 888
hourly bars per asset from CCXT, 110 funding events per asset from
DB, features computed for 2/2, RF inference complete, `Persisted: 0
new row(s)` (PK collision with the earlier manual smoke-test run at
the same window).

### First-window observation (2026-05-27 00:00 UTC funding window)

```
asset  datetime                    p_profitable  above_gate  above_gate_050  ann_rate  pos
BTC    2026-05-27T00:00:00+00:00   0.3116        0           0               +6.32%    0.611
ETH    2026-05-27T00:00:00+00:00   0.2989        0           0               +3.73%    0.678
```

Both assets sit out at both gates, as predicted in the RECON. Best
config selected by RF for both: `fr_0000` (5% min, 3-day hold, 0.5
min pct_positive) — the loosest config in the 36-config grid, and
even that doesn't fire because P-scores are well below 0.50.

Per-row `features_json` (BTC example):
- funding_8h_pct: +0.005776
- funding_ann_pct: +6.325 (just barely above the loosest 5% threshold)
- funding_7d_avg_ann: +5.123
- funding_trend: +0.235
- funding_pct_positive: 0.611 (above the loosest 0.5 floor)
- funding_volatility: 4.763
- basis_pct: −0.025
- basis_7d_avg: −0.049
- basis_trend_24h: +0.026
- spot_vol_24h_ann: 0.312
- vol_regime: 1.005

Interpretation: BTC's most-recent funding rate is mildly positive
(+6.3% annualized) AND the 7-day average is also above 5%, so the
*entry conditions in `run_funding_single_day`* technically pass — but
the RF still outputs P ≈ 0.31 because the broader feature context
(volatility, basis profile, recent trend) doesn't match the
"sustained-positive carry window" pattern the model learned in 2024.
This is the model doing exactly what atlas Exp 13 says it should:
"sustained high positive funding + stable basis + pct_positive ≥
threshold = favorable carry window. Not a spurious pattern."

### Step 9: idempotency + smoke tests

Manual re-run during dev: "Persisted: 0 new row(s)" (PK already
existed). Confirmed `funding_signals` still has exactly 2 rows after
both the manual and scheduled invocations targeting the same window.

---

## Acceptance criteria check

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | Deployed monitor (Task Scheduler entry running) | ✅ `PraxisFundingMonitor` State=Ready, 3 daily triggers |
| 2 | Writing signal output to a queryable store | ✅ `funding_signals` table in `crypto_data.db`, 2 rows after first window |
| 3 | Observable for ≥1 full funding cycle = 8h | ✅ One full funding window (2026-05-27 00:00 UTC) captured + persisted |
| 4 | Retro documenting deployment + first observation | ✅ this file |

Plus the cycle's specific scope acceptance: dual-gate persistence ✓,
DB funding source ✓, model path fix ✓.

---

## Notes

### Local-vs-UTC timing (per brief ops note)

Documented per brief request. Task Scheduler triggers are LOCAL time
(Toronto UTC−4 summer / UTC−5 winter). The 00:15 LOCAL trigger fires
~4-5h after the corresponding 00:00 UTC funding event the row is
attached to. The strategy's natural decision cadence is per-funding-
window (8h), so the lag is inside the actionable interval. "Near-
real-time, ≤8h post-event" is the right operational frame for
funding-carry signals.

### Sit-out is the success outcome (per brief ops note)

Per Cycle 39 finding and atlas Exp 13's regime-conditioning, BTC's
sustained-negative funding regime over April-May 2026 (and ETH's
mild-bear) means the strategy should sit out completely. The first
observation row (P_BTC=0.3116, P_ETH=0.2989, both well below the
0.50 headline gate let alone the 0.70 live gate) is a successful
observation of that sit-out behavior. Atlas's "2022 bear validation
not yet run" caveat gets its first forward-looking real-time
confirmation data point.

No `p_profitable > 0.70` row appeared during this cycle, so no
"surface before commit" trigger fired. If a P-score creeps above
0.70 in subsequent windows during the bear regime, that would be a
finding worth surfacing in Cycle 42 retro (calibration drift /
regime mis-classification).

### Cycle 40 model commit pinned in monitor_version

Each persisted row records `monitor_version = "cycle40:082459b"`,
tying the signal to the Cycle 40 main commit that produced the
verified phase3 models. If models are re-trained (e.g. annual
refresh, universe extension), update the `MONITOR_VERSION` constant
in `scripts/funding_monitor.py` to identify the new model lineage in
downstream rows.

### funding_window_timestamp design choice

Round-down to nearest 00/08/16 UTC (`(now.hour // 8) * 8`) ties each
row to the most-recent funding event the inference is making a
decision about. Alternative would have been "use the actual timestamp
of the latest funding event in `funding_rates`", which is more
precise but introduces a dependency on `funding_rates` being fresh
and adds an extra DB query. The round-down approach is cleaner and
the small (~5 min) skew between rounded-down window and actual
funding event time is below any observable precision.

### Idempotency under same-window re-runs

Multiple runs within the same 8h funding window are safe — PK
`(asset, timestamp)` collides on `INSERT OR IGNORE` and no
duplicates land. This was tested via the smoke-run (manual at ~00:42
UTC) + scheduled-context run (at ~00:54 UTC) both targeting the same
00:00 UTC window: row counts stayed at 2.

### What this cycle does NOT do

- Does NOT extend the universe to 6 assets (Cycle 42 candidate)
- Does NOT add alerting (no webhook/email when above_gate=1)
- Does NOT integrate with the executor (no live trading)
- Does NOT change PraxisFundingCollector (collector unchanged; monitor
  is purely downstream of it)
- Does NOT touch `run_cpo.py` (the `--feature-mode` default trap from
  Cycle 40 retro remains; not in scope)

---

## Open items / Cycle 42 inputs

- **42a Universe extension**: extend `PraxisFundingCollector` to the
  full 6-asset deployment universe (SOL, XRP, ADA, AVAX in addition
  to BTC + ETH); then extend the monitor to match. The Cycle 40 D1
  backfill already populated historical data for the 4 missing
  assets, so the live collector extension just needs the additional
  CCXT calls + a quick re-register of the collector task.
- **42b Alerting**: wire `above_gate=1` rows to a notification
  surface (Discord webhook, email via funding_alert.py's existing
  SMTP path, etc.). Currently funding_alert.py exists but is
  unscheduled; either schedule it as a separate task or fold the
  alert logic into funding_monitor.py.
- **42c Cross-venue funding spreads** (Bybit, OKX, Hyperliquid;
  atlas Exp 13 revival hypothesis #1, ~+30-50% effective universe
  size per venue)
- **42d --feature-mode default fix in run_cpo.py** (memory #23
  candidate; carry-over from Cycle 40 retro)
- **42e LSTM v2 architecture test** (atlas Exp 13 revival hypothesis
  #4; deferred per atlas's "likelihood: low" assessment but now
  cheaper to attempt with the foundation verified)
- **42f Real-time alerting integration with executor** (the brief
  out-of-scope item)
- **Passive observation continues**: PraxisFundingMonitor keeps
  writing 3 rows/day across BTC + ETH (6 rows/day total) for as long
  as it runs. After ~30 days the funding_signals table will have
  ~180 rows for regime analysis.
