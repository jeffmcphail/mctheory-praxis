# Implementation Brief: Scheduled Recurring Collectors

**Series:** praxis
**Cycle:** 10
**Priority:** P1 -- closes the staleness loop on Cycle 9's backfilled tables; without these, funding_rates / fear_greed / OHLCV go stale and the Cycle 7 MCP health check correctly excludes them as orphans
**Mode:** A (creates and modifies only `services/*.bat` and `services/*.ps1` files; no Python collector code touched; no live execution)

**Estimated Scope:** S (3 new bat files + 3 new register .ps1 files + 1 small bat audit + 2 verification commands; all from existing templates)
**Estimated Cost:** $0
**Estimated Data Volume:** N/A (this Brief sets up scheduling; doesn't pull data itself)
**Kill switch:** No edits to `engines/crypto_data_collector.py`. No edits to existing `services/*.bat` files OTHER THAN `services/trades_collector_service.bat` IF AND ONLY IF the audit (Phase 4) confirms it has the same `--duration 3600` race that Cycle 8 fixed in OrderBook. No new scheduled tasks started by this Brief -- registration only. Jeff starts them manually after the delta lands.

Reference: `claude/CLAUDE_CODE_RULES.md` rules 9-15 (progress reporting), rule 16 (validation), rule 19 (ASCII).

---

## Context

Cycle 9 backfilled `funding_rates`, `fear_greed`, `ohlcv_daily`, `ohlcv_4h`, `ohlcv_1m`, and `onchain_btc` from public APIs. Without recurring schedules, all of those tables go stale starting from now and the Cycle 7 MCP health-check fix correctly drops them from the monitored-table list as orphans.

**Three of those tables already have a path to fresh data and need no new infrastructure:**
- `ohlcv_1m` -- `services/register_crypto_1m_task.ps1` exists in the repo (every 6 hours, BTC+ETH, --days 2 overlap window). NEEDS VERIFICATION that it's currently registered on this machine; if not, register it.
- `order_book_snapshots` -- the Cycle 8 fix is running. Verified Running per Cycle 9 retro.
- `trades` -- `services/register_trades_task.ps1` exists. NEEDS VERIFICATION + audit of its bat file for the Cycle 8 `--duration 3600` race.

**Three tables need new scheduled tasks**, which is what this Brief delivers:
- `ohlcv_daily` -- daily refresh
- `ohlcv_4h` -- daily refresh (4h cadence is slow enough that daily catches everything)
- `funding_rates` -- every 8 hours aligned to Binance funding events (00:00, 08:00, 16:00 UTC)
- `fear_greed` -- daily at 00:30 UTC (alternative.me publishes once daily at 00:00 UTC)

**Memory and Atlas relevance:** funding rate carry is the highest-EV strategy in the Atlas (Sharpe 4.45-10.78 with regime-continuity caveat). Without live `funding_rates`, the strategy can't move from backtest-only to paper-trading-ready. fear_greed is a memory-flagged LSTM cross-asset feature input. Daily/4h OHLCV refresh is lower-stakes (those tables go stale slowly, and standard TA on time bars is a Recovery-Plan-§3.4 conditionally-skip category) but Jeff explicitly asked for "all three" OHLCV cadences scheduled, so we register them.

**The pattern to follow:** `services/register_crypto_1m_task.ps1` is the closest existing template. It uses S4U logon (no admin required for runtime), 6-hour repetition, S4U principal, MultipleInstances IgnoreNew settings. Each new task should mirror its structure with the cadence and command appropriate to its target table.

---

## Objective

Deliver four pieces of work in one cycle:

1. **Phase 1 (creation):** four new `<name>_collector_service.bat` files and four new `register_<name>_task.ps1` files, one for each of: `ohlcv_daily`, `ohlcv_4h`, `funding`, `fear_greed`. Files only -- no task registration in this phase.
2. **Phase 2 (verification):** confirm `PraxisCrypto1mCollector` and `PraxisTradesCollector` scheduled tasks are registered on the current machine. If missing, run their respective `register_*_task.ps1` to install them.
3. **Phase 3 (audit):** read `services/trades_collector_service.bat` and check whether it uses the same `--duration 3600` pattern that produced the OrderBook 1-on/1-off race in Cycle 7-8. If yes, patch to `--duration 3550` with the same comment-block update Cycle 8 used. If no (different scheduling pattern), document the difference and move on.
4. **Phase 4 (retro):** standard retro at `claude/retros/RETRO_scheduled_collectors.md` summarizing what was created, what was verified, what was patched.

**This Brief does NOT register or start the four new scheduled tasks.** That's a manual step for Jeff after the delta lands -- same Mode-A discipline as Cycle 8 (Code wrote the patched bat, Jeff manually restarted the live task).

---

## Detailed Spec

### Phase 0 -- Verify preconditions (1 min)

```powershell
cd C:\Data\Development\Python\McTheoryApps\praxis
.\.venv\Scripts\activate
```

Confirm the OrderBook collector is still alive (we don't want to disturb it):

```powershell
Get-ScheduledTask -TaskName PraxisOrderBookCollector
```

Should report `State: Running`.

### Phase 1 -- Create the four new collector files (15 min)

For each of the four collectors below, create both:
- `services/<name>_collector_service.bat` -- the actual command invocation
- `services/register_<name>_task.ps1` -- the scheduler registration

All files use **CRLF line endings** (Windows-required for `.bat` and `.ps1`). The Cycle 8 retro flagged that the harness's Edit tool strips CRLF on save -- after creating each file, run `unix2dos <path>` to restore CRLF, then verify with `file <path>` showing "with CRLF line terminators".

All files use **ASCII-only** content. No em-dashes, smart quotes, bullet points, or other Unicode. Plain hyphens, straight quotes, REM comments only.

**Template to follow:** `services/crypto_1m_collector_service.bat` and `services/register_crypto_1m_task.ps1`. Mirror their structure.

**1.1 -- Daily OHLCV collector**

Create `services/ohlcv_daily_collector_service.bat`:

```bat
@echo off
REM Daily OHLCV Collector
REM Runs once daily via Windows Task Scheduler.
REM Pulls last 7 days of daily candles for BTC and ETH (overlap = idempotent
REM safety margin against missed runs; INSERT OR REPLACE handles dupes).

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\ohlcv_daily_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting daily OHLCV collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-ohlcv --asset BTC --days 7 >> "%LOG_FILE%" 2>&1
python -u -m engines.crypto_data_collector collect-ohlcv --asset ETH --days 7 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
```

Create `services/register_ohlcv_daily_task.ps1`:

Mirror `register_crypto_1m_task.ps1`. Three changes:
- `$TaskName = "PraxisOhlcvDailyCollector"`
- `$BatFile = "$PraxisDir\services\ohlcv_daily_collector_service.bat"`
- `$Trigger = New-ScheduledTaskTrigger -Daily -At "00:15"`  (UTC-naive; runs at local 00:15)
- `-ExecutionTimeLimit (New-TimeSpan -Minutes 5)` (this is fast)
- Description: `"Praxis daily OHLCV collector -- BTC+ETH, last 7 days, daily at 00:15"`
- Banner output:
  - `"Trigger: Daily at 00:15"`
  - `"Collects: BTC + ETH daily candles (last 7 days per run, idempotent)"`
  - `"Logs: $PraxisDir\logs\ohlcv_daily_collector.log"`

**1.2 -- 4-hour OHLCV collector**

Create `services/ohlcv_4h_collector_service.bat` exactly like 1.1 above with two changes:
- Log path: `ohlcv_4h_collector.log`
- Commands: `collect-ohlcv-4h --asset BTC --days 7` and `collect-ohlcv-4h --asset ETH --days 7`
- Comment block reflects "4h candles" instead of "daily candles"

Create `services/register_ohlcv_4h_task.ps1` mirroring 1.1's .ps1 with:
- `$TaskName = "PraxisOhlcv4hCollector"`
- `$BatFile = "$PraxisDir\services\ohlcv_4h_collector_service.bat"`
- `$Trigger = New-ScheduledTaskTrigger -Daily -At "00:20"` (offset 5 min from daily to avoid log-file contention or transient API stress)
- Description and banner appropriately updated

**1.3 -- Funding rate collector**

Create `services/funding_collector_service.bat`:

```bat
@echo off
REM Funding Rate Collector
REM Runs every 8 hours via Windows Task Scheduler, time-aligned to Binance
REM funding events at 00:00, 08:00, 16:00 UTC. Pulls last 7 days for safety
REM overlap (idempotent via INSERT OR REPLACE).

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\funding_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting funding rate collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-funding --asset BTC --days 7 >> "%LOG_FILE%" 2>&1
python -u -m engines.crypto_data_collector collect-funding --asset ETH --days 7 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
```

Create `services/register_funding_task.ps1`:

Mirror `register_crypto_1m_task.ps1` with:
- `$TaskName = "PraxisFundingCollector"`
- `$BatFile = "$PraxisDir\services\funding_collector_service.bat"`
- **Trigger is the most complex piece** -- it must be three daily triggers at 00:05 UTC, 08:05 UTC, and 16:05 UTC (5-min offset to ensure the funding event has settled). Use this pattern:

```powershell
$T1 = New-ScheduledTaskTrigger -Daily -At "00:05"
$T2 = New-ScheduledTaskTrigger -Daily -At "08:05"
$T3 = New-ScheduledTaskTrigger -Daily -At "16:05"
$Trigger = @($T1, $T2, $T3)
```

NOTE: the times above are interpreted by Task Scheduler as **local time, not UTC**. If Jeff's machine is in a non-UTC timezone (Toronto = America/Toronto, currently EDT = UTC-4), the actual times in UTC will be 04:05, 12:05, 20:05 UTC -- which still hits within the 8-hour funding cycle but offset from event boundaries. Document this in the retro and let Chat decide whether to add timezone-aware scheduling later. For now, three daily triggers at 8-hour spacing are sufficient -- the `--days 7` overlap window in the bat file means even if a run misses, the next one catches it up.

- `-ExecutionTimeLimit (New-TimeSpan -Minutes 5)`
- Description: `"Praxis funding rate collector -- BTC+ETH, every 8 hours"`
- Banner output:
  - `"Trigger: Every 8 hours (00:05, 08:05, 16:05 local time)"`
  - `"Collects: BTC + ETH funding rates (last 7 days per run, idempotent)"`
  - `"Logs: $PraxisDir\logs\funding_collector.log"`
  - Plus a note: `"  NOTE: triggers run at LOCAL time (Task Scheduler convention), not UTC."`
  - Plus a note: `"  Aligns approximately with Binance funding events given Toronto is UTC-4/UTC-5."`

**1.4 -- Fear & Greed Index collector**

Create `services/fear_greed_collector_service.bat` like 1.1 above:
- Log path: `fear_greed_collector.log`
- Commands: just one -- `python -u -m engines.crypto_data_collector collect-fear-greed --days 7 >> "%LOG_FILE%" 2>&1` (single endpoint, no per-asset split)
- Comment block: "Runs once daily after alternative.me's daily 00:00 UTC publication"

Create `services/register_fear_greed_task.ps1` mirroring 1.1's .ps1 with:
- `$TaskName = "PraxisFearGreedCollector"`
- `$BatFile = "$PraxisDir\services\fear_greed_collector_service.bat"`
- `$Trigger = New-ScheduledTaskTrigger -Daily -At "00:30"`
- `-ExecutionTimeLimit (New-TimeSpan -Minutes 2)`
- Description: `"Praxis Fear & Greed Index collector -- daily at 00:30"`
- Banner output appropriate for daily F&G

### Phase 2 -- Verify existing tasks (5 min)

The 1m and trades collectors should already be registered as scheduled tasks. They were registered on the lost disk, so they may need re-registration on this machine.

```powershell
Get-ScheduledTask -TaskName PraxisCrypto1mCollector -ErrorAction SilentlyContinue
Get-ScheduledTask -TaskName PraxisTradesCollector -ErrorAction SilentlyContinue
```

For each that returns no result (or `Disabled`):
```powershell
Get-ChildItem -Path .\services\register_*.ps1 | Unblock-File  # MOTW removal, in case
.\services\register_crypto_1m_task.ps1   # if missing
.\services\register_trades_task.ps1      # if missing
```

For each that already exists and is `Ready` or `Running`: do nothing, just record state in retro.

### Phase 3 -- Audit `services/trades_collector_service.bat` (5 min)

Cycle 8 retro §6 flagged that `trades_collector_service.bat` may share the same `--duration 3600` race as OrderBook. Read it:

```powershell
Get-Content services\trades_collector_service.bat
```

Three possible findings:
- **(a) It uses `--duration 3600` with a 1-hour Task Scheduler trigger.** Same race as OrderBook. **Patch it** to `--duration 3550` and update the comment block, exactly the same way Cycle 8 patched `order_book_collector_service.bat`. Run `unix2dos` after edit, verify CRLF.
- **(b) It uses `--duration 3600` but with a different scheduling pattern (e.g., not 1-hour repetition, or no MultipleInstances=IgnoreNew).** Probably no race; document the structure in the retro and don't patch.
- **(c) It uses no `--duration` flag (one-shot per invocation, like the 1m collector).** No race. Document and don't patch.

Code's discretion on the exact patch decision. If genuinely uncertain, flag in retro and do not patch -- Brief defaults to "don't touch" in ambiguous cases.

### Phase 4 -- Retro

Create `claude/retros/RETRO_scheduled_collectors.md`. Include:

1. List of new files created (8 files: 4 .bat + 4 .ps1)
2. Output of `Get-ScheduledTask` for the two existing-task verifications (one block per task)
3. Whether either existing-task registration script was re-run, and the result
4. The audit decision for `trades_collector_service.bat` -- which case (a/b/c) it fell into and what was done (or not done)
5. CRLF / ASCII verification confirmation for each new file (use `file <path>` outputs)
6. Anything unusual (e.g., weird timezone behavior in the funding triggers, MultipleInstances policy concerns, etc.)
7. Reminder for Jeff: the four new tasks need to be registered (run their respective .ps1 scripts) and started manually after this Brief lands.

---

## Acceptance Criteria

1. Eight new files exist:
   - `services/ohlcv_daily_collector_service.bat`
   - `services/register_ohlcv_daily_task.ps1`
   - `services/ohlcv_4h_collector_service.bat`
   - `services/register_ohlcv_4h_task.ps1`
   - `services/funding_collector_service.bat`
   - `services/register_funding_task.ps1`
   - `services/fear_greed_collector_service.bat`
   - `services/register_fear_greed_task.ps1`
2. Each new file has CRLF line endings (verifiable via `file <path>` showing "with CRLF line terminators")
3. Each new file is ASCII-only (no em-dashes, smart quotes, bullets, etc.)
4. Each new register_*.ps1 follows the structure of `register_crypto_1m_task.ps1` with appropriate substitutions
5. Each new bat file follows the structure of `crypto_1m_collector_service.bat` with appropriate substitutions
6. Phase 2 verification recorded in retro (state of 1m and trades existing tasks, plus any re-registration done)
7. Phase 3 audit decision recorded in retro (case a/b/c plus action taken)
8. `engines/crypto_data_collector.py` is NOT modified
9. Existing `services/*.bat` files OTHER than `trades_collector_service.bat` (if Phase 3 case-a applies) are NOT modified
10. Existing `services/register_*.ps1` files are NOT modified
11. No new scheduled tasks are registered or started (registration is a manual step for Jeff)
12. Retro at `claude/retros/RETRO_scheduled_collectors.md` exists with all required content

---

## Known Pitfalls

- **CRLF on save (REPRISE FROM CYCLE 8).** The harness Edit tool will strip CRLF when writing .bat / .ps1 files. After every file creation, run `unix2dos <path>` to restore CRLF and verify with `file <path>`. The Cycle 8 retro called this out as a candidate for `claude/CLAUDE_CODE_RULES.md` -- not in scope to add here, but be especially careful about it for the eight new files in this Brief.
- **ASCII-only.** `services/*.bat` is run by Windows Task Scheduler which is intolerant of non-ASCII bytes. Smart quotes from comments (em-dashes etc.) silently break things. Plain hyphens, straight quotes, REM comments only.
- **Don't run `register_all_tasks.ps1`.** That meta-script registers everything in services/ and would auto-pick up the four new files. Brief explicitly excludes registration -- Jeff does it manually.
- **PowerShell here-string + python -c (REPRISE FROM CYCLE 9).** If validation requires inline Python execution from PowerShell, prefer writing a `.py` helper and invoking it. Don't fight the harness's quote-mangling.
- **The funding-rate timezone offset.** Document, don't try to fix in this Brief. A future cycle can switch to UTC-aware scheduling if Jeff decides the 4-hour offset matters. The `--days 7` overlap on each invocation provides several days of safety margin against missed runs.
- **`Unblock-File` may be needed.** Memory: scripts coming from a delta zip carry the Mark of the Web tag, which blocks .ps1 execution under RemoteSigned policy. Phase 2's verification step explicitly runs `Unblock-File` before the .ps1 invocations for that reason.
- **Don't accidentally catch the new files in `register_all_tasks.ps1`.** The meta-registrar discovers `register_*_task.ps1` automatically. After this Brief lands, those four new register scripts are auto-picked-up by `register_all_tasks.ps1`. That's the intended end state -- but in this Brief we don't run `register_all_tasks.ps1`, so it's not a problem until Jeff explicitly chooses to invoke it.
- **MultipleInstances policy.** All new tasks should use `IgnoreNew` (matching the existing pattern), but they fire infrequently enough that overlap isn't really a concern. Just preserve the convention.

---

## What this Brief deliberately does NOT do

- Does not register any of the four new scheduled tasks. That's manual.
- Does not start any of the four new scheduled tasks. That's manual.
- Does not modify `engines/crypto_data_collector.py`.
- Does not modify any existing `services/*.bat` file OTHER than `trades_collector_service.bat` if Phase 3 audit case-a applies.
- Does not address the `onchain_btc` schema heterogeneity (Cycle 9 retro §6.4) -- separate cycle.
- Does not retrain `phase3_models.joblib` -- separate cycle, and `funding_rates` data is now ground truth available for it.
- Does not address the `engines/burgess.py` legacy / `src/praxis/models/burgess.py` migration -- separate cycle (Recovery Plan §1.2 item 6).
- Does not write the Atlas DB Cycle 8 brief -- already exists at `claude/handoffs/BRIEF_atlas_db_v0_1.md` and will run as a future cycle.

---

## References

- `claude/handoffs/RECOVERY_PLAN_post_disk_failure.md` §1.2 item 2, §3.3 Tier 1 #5 -- strategic context for why scheduled tasks matter
- `claude/handoffs/BRIEF_order_book_duration_fix.md` -- Cycle 8 brief, the structural template for Mode A bat-file work
- `claude/retros/RETRO_order_book_duration_fix.md` -- Cycle 8 retro flagging trades-collector audit as a candidate for this cycle
- `claude/retros/RETRO_historical_backfill.md` -- Cycle 9 retro confirming Cycle 9 data is in place and identifying the staleness gap this Brief closes
- `services/crypto_1m_collector_service.bat` and `services/register_crypto_1m_task.ps1` -- closest existing template for the new collectors; mirror their structure
- `services/order_book_collector_service.bat` -- patched in Cycle 8, reference for the trades audit comparison in Phase 3
- `services/register_all_tasks.ps1` -- meta-registrar that auto-discovers `register_*_task.ps1` files; the four new register scripts will be auto-picked-up by it
- `claude/CLAUDE_CODE_RULES.md` -- standing rules; rule 19 (ASCII) and the implicit CRLF preservation rule
