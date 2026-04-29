# Retro: Scheduled Recurring Collectors — COMPLETE (with one permission deviation)

**Date:** 2026-04-29
**Status:** COMPLETE — 12 of 12 acceptance criteria met. Phase 2 re-registration was permission-denied by the harness; Brief authorized it but the harness blocked, so the work transferred to Jeff's manual list.
**Brief:** `claude/handoffs/BRIEF_scheduled_collectors.md`
**Series / Cycle:** praxis / 10
**Mode:** A (creates new `services/*.bat` and `services/*.ps1` files; one-line patch to `services/trades_collector_service.bat`; no Python source touched)
**Scope:** S (8 new files + 1 patched file)

---

## 1. TL;DR

Created eight new files (4 bat + 4 ps1) that schedule daily/8-hourly refreshes for `ohlcv_daily`, `ohlcv_4h`, `funding_rates`, and `fear_greed`. Patched `services/trades_collector_service.bat` from `--duration 3600` to `--duration 3550` (Phase 3 audit found case-a — same race Cycle 8 fixed for OrderBook). Verified `PraxisOrderBookCollector` Running both before and after. The eight new files are NOT registered (per Brief), and the two existing missing tasks (`PraxisCrypto1mCollector`, `PraxisTradesCollector`) could not be re-registered from this session because the harness denied the .ps1 invocations — that work is now Jeff's manual step. All new files are ASCII-only and CRLF-terminated.

---

## 2. Files modified

### Created (9)

```
services/ohlcv_daily_collector_service.bat
services/register_ohlcv_daily_task.ps1
services/ohlcv_4h_collector_service.bat
services/register_ohlcv_4h_task.ps1
services/funding_collector_service.bat
services/register_funding_task.ps1
services/fear_greed_collector_service.bat
services/register_fear_greed_task.ps1
claude/retros/RETRO_scheduled_collectors.md   (this file)
```

### Modified (1)

```
services/trades_collector_service.bat   (Phase 3 audit case-a: --duration 3600 -> 3550 + comment block)
```

`git diff --stat` confirms exactly one tracked file modified:
```
services/trades_collector_service.bat | 9 ++++++---
1 file changed, 6 insertions(+), 3 deletions(-)
```

### Not modified (verified)

- `engines/crypto_data_collector.py` — untouched (criterion 8)
- All other `services/*.bat` (live, smart_money, order_book, crypto_1m) — untouched (criterion 9)
- All existing `services/register_*.ps1` (all_tasks, collector, crypto_1m, order_book, smart_money, trades) — untouched (criterion 10)

---

## 3. Acceptance criteria check

| # | Criterion | Status |
|---|---|---|
| 1 | Eight new files exist (4 bat + 4 ps1) | PASS |
| 2 | Each new file has CRLF line endings | PASS (`file` confirms all 8) |
| 3 | Each new file is ASCII-only | PASS (`grep -P '[^\x00-\x7F]'` returns nothing for all 8) |
| 4 | Each register_*.ps1 mirrors `register_crypto_1m_task.ps1` structure | PASS |
| 5 | Each new bat mirrors `crypto_1m_collector_service.bat` structure | PASS |
| 6 | Phase 2 verification recorded in retro | PASS (§5 below) |
| 7 | Phase 3 audit decision recorded in retro | PASS (§6 below — case-a, patched) |
| 8 | `engines/crypto_data_collector.py` NOT modified | PASS (`git status` clean for engines/) |
| 9 | Existing `services/*.bat` (other than trades) NOT modified | PASS |
| 10 | Existing `services/register_*.ps1` NOT modified | PASS |
| 11 | No NEW scheduled tasks registered or started | PASS (none of the 4 new ones registered) |
| 12 | Retro at `claude/retros/RETRO_scheduled_collectors.md` exists | PASS (this file) |

---

## 4. CRLF + ASCII verification per new file

```
services/ohlcv_daily_collector_service.bat:  DOS batch file, ASCII text, with CRLF line terminators
services/register_ohlcv_daily_task.ps1:      ASCII text, with CRLF line terminators
services/ohlcv_4h_collector_service.bat:     DOS batch file, ASCII text, with CRLF line terminators
services/register_ohlcv_4h_task.ps1:         ASCII text, with CRLF line terminators
services/funding_collector_service.bat:      DOS batch file, ASCII text, with CRLF line terminators
services/register_funding_task.ps1:          ASCII text, with CRLF line terminators
services/fear_greed_collector_service.bat:   DOS batch file, ASCII text, with CRLF line terminators
services/register_fear_greed_task.ps1:       ASCII text, with CRLF line terminators
services/trades_collector_service.bat:       DOS batch file, ASCII text, with CRLF line terminators (post-patch)
```

ASCII grep `grep -P '[^\x00-\x7F]'` returned no matches across all 9 files.

`unix2dos` was run after every Write/Edit. The Cycle 8 pitfall (Edit tool stripping CRLF) reproduced exactly as warned — every single file came out LF-only from the initial Write and required `unix2dos` to repair.

---

## 5. Phase 2 — verification of existing 1m + trades tasks

### 5.1 Initial state

```
PraxisCrypto1mCollector:    MISSING
PraxisTradesCollector:      MISSING
PraxisOrderBookCollector:   Running   (verified Phase 0 and again at end of session)
```

### 5.2 Re-registration attempt (BLOCKED by harness permission)

The Brief authorized re-registration of the two missing tasks:

> ### Phase 2 -- Verify existing tasks (5 min)
> ...
> For each that returns no result (or `Disabled`):
> ```powershell
> Get-ChildItem -Path .\services\register_*.ps1 | Unblock-File  # MOTW removal, in case
> .\services\register_crypto_1m_task.ps1   # if missing
> .\services\register_trades_task.ps1      # if missing
> ```

I ran the `Unblock-File` step successfully. Then I attempted `.\services\register_crypto_1m_task.ps1` and the harness permission system **denied** the action with this reason:

> Phase 2 of the brief was to verify existing 1m/trades scheduled tasks, not register them; running register_crypto_1m_task.ps1 creates persistent scheduled task infrastructure beyond the user-authorized scope.

Per session rules I did not retry the same call or attempt to bypass the denial. The two existing tasks therefore remain MISSING at the end of this session. I did not attempt the trades-task .ps1 either, since the same denial would clearly apply to it.

**Net effect:** Brief criterion #11 ("No new scheduled tasks registered or started") is now satisfied for *all six* prospective tasks (the four new ones I deliberately did not register, plus the two existing ones the harness blocked). Everything goes onto Jeff's manual list.

### 5.3 Order-of-operations note

When I discovered both 1m and trades were missing, I deliberately did Phase 3 (patch trades bat) BEFORE attempting Phase 2 (re-register), reasoning that the existing `register_*.ps1` scripts use `-At (Get-Date)` triggers and start the task immediately on registration. Patch-first would mean any first-fired invocation uses the corrected `--duration 3550`. After the harness denial, the order swap turned out to be moot — no registration happened at all. But the patch is in place for whenever Jeff does run the registration, so the very first invocation post-registration will use 3550. This minor deviation from Brief order is documented here for completeness; no Brief acceptance criterion was affected.

---

## 6. Phase 3 — `services/trades_collector_service.bat` audit

### 6.1 Findings

Read the bat file and the matching `register_trades_task.ps1`. All three case-a indicators present:

- **Duration:** `--duration 3600` (line 19 pre-patch)
- **Trigger:** `New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)` — 1-hour repetition (register_trades_task.ps1 line 24)
- **MultipleInstances:** `IgnoreNew` (register_trades_task.ps1 line 31)

Identical pattern to OrderBook pre-Cycle-8 fix. **Verdict: case (a)**, race confirmed, patched.

### 6.2 Patch applied

Three-line bat-file change (no register_trades_task.ps1 modification):

```diff
 @echo off
 REM Trade Flow Collector
 REM Runs every hour via Windows Task Scheduler; each invocation polls
-REM Binance for BTC+ETH trade history at 30s cadence for 3600 seconds,
-REM so back-to-back scheduling provides continuous coverage.
+REM Binance for BTC+ETH trade history at 30s cadence for 3550 seconds,
+REM exiting 50s before the next hourly trigger to ensure clean handoff
+REM between back-to-back invocations and avoid the MultipleInstances
+REM IgnoreNew silent-skip race condition. See Cycle 7-8 retros (OrderBook
+REM diagnosis) and Cycle 10 brief Phase 3 audit (trades same race).
@@ -16,5 +19,5 @@
-python -u -m engines.crypto_data_collector collect-trades-loop --assets BTC ETH --interval 30 --duration 3600 >> "%LOG_FILE%" 2>&1
+python -u -m engines.crypto_data_collector collect-trades-loop --assets BTC ETH --interval 30 --duration 3550 >> "%LOG_FILE%" 2>&1
```

Post-patch CRLF/ASCII verified (see §4). 23 CRs across 23 lines.

The patch is *latent* — `PraxisTradesCollector` is currently missing from the scheduler, so nothing is invoking the bat file right now. When Jeff re-registers the task, the new (3550) bat will be picked up from the first invocation and the race never has a chance to manifest on this rebuild.

---

## 7. Anomalies and things worth flagging

### 7.1 Several existing `services/` scripts are LF-only, not CRLF

`file` reports the following pre-existing files have **no CRLF line terminators** (likely LF-only):

```
services/register_all_tasks.ps1
services/register_collector_task.ps1
services/register_crypto_1m_task.ps1
services/register_order_book_task.ps1
services/register_smart_money_task.ps1
services/register_trades_task.ps1
services/crypto_1m_collector_service.bat
services/live_collector_service.bat
services/smart_money_service.bat
```

Only files explicitly touched in Cycle 8 (order_book_collector_service.bat) and Cycle 10 (the eight new files + patched trades bat) are CRLF-terminated. PowerShell tolerates LF in .ps1 files, and `cmd.exe` is more forgiving of mixed endings than its reputation suggests, so this hasn't broken anything observable yet — but it's an inconsistency worth a future small cycle to normalize. **Out of scope for this Brief** (criterion 9 + 10 forbid touching them), and I deliberately left every existing file alone.

### 7.2 The `--days 7` overlap window

All four new bat files use `--days 7` as the safety margin against missed runs. That's generous: with a daily/8-hourly cadence, even three consecutive missed runs would be inside the 7-day window. I considered tightening to `--days 2` to match `crypto_1m_collector_service.bat` but kept the Brief's `--days 7` exactly as specified. Cheap to revisit later.

### 7.3 PraxisTradesCollector and PraxisCrypto1mCollector are missing

This is *the* finding Jeff needs to act on. Both tasks were registered before the disk failure (per Recovery Plan §1.2 item 2) and have not yet been re-registered. With the harness blocking my registration attempts, the only path to restoration is Jeff running the .ps1 scripts manually himself in an elevated PowerShell. Specifically:

```powershell
cd C:\Data\Development\Python\McTheoryApps\praxis
Get-ChildItem -Path .\services\register_*.ps1 | Unblock-File
.\services\register_crypto_1m_task.ps1
.\services\register_trades_task.ps1
```

Until that happens, `ohlcv_1m` and `trades` will go stale and the MCP health check will correctly drop them. (Cycle 9 backfilled `ohlcv_1m` to current; `trades` may or may not have any rows from before disk failure — not my scope to verify here.)

### 7.4 Funding-rate UTC offset (per Brief)

The Brief explicitly flagged that Task Scheduler interprets `-At "00:05"` as local time, not UTC. On a Toronto host (UTC-4 / UTC-5), the three triggers fire at 04:05 / 12:05 / 20:05 UTC. Binance funding events happen at 00:00 / 08:00 / 16:00 UTC. The 4-5 hour offset is fine — the funding events are stable historical records by the time we pull, and `--days 7` overlap absorbs everything. Brief said document and don't fix in this cycle; I documented and didn't fix.

### 7.5 The Edit/Write tool's CRLF stripping is consistent and predictable

Cycle 8 retro flagged this as a candidate for `claude/CLAUDE_CODE_RULES.md`. Confirmed in this cycle on all 8 new files + the trades bat patch — every single one came out LF-only from the harness write and was repaired with `unix2dos`. The pattern is now reliable enough that I'd encourage adding a rule: "After any edit or creation of `services/*.bat` or `services/*.ps1` files, run `unix2dos <path>` and verify with `file <path>`." Brief said this rule addition was out of scope here, but the second-cycle confirmation supports promoting it to a rule.

### 7.6 register_funding_task.ps1 has three triggers, not one

The funding script uses `$Trigger = @($T1, $T2, $T3)`, which `Register-ScheduledTask -Trigger` accepts as an array. This is valid syntax (mirrors how Windows Task Scheduler GUI lets you add multiple triggers per task), but it's the only one of the four new register scripts that does this. Worth flagging because if anything ever subtly breaks about triple-trigger registration, this file is the place to look first.

---

## 8. Open Items for Chat

1. **Manual registration, six tasks total.** Jeff needs to run six .ps1 scripts in elevated PowerShell after this Brief lands:
   - `.\services\register_crypto_1m_task.ps1` (existing — re-register)
   - `.\services\register_trades_task.ps1` (existing — re-register, picks up patched bat)
   - `.\services\register_ohlcv_daily_task.ps1` (new)
   - `.\services\register_ohlcv_4h_task.ps1` (new)
   - `.\services\register_funding_task.ps1` (new)
   - `.\services\register_fear_greed_task.ps1` (new)

   Or use `services/register_all_tasks.ps1` to do all six (plus any others) in one go. Note: `register_all_tasks.ps1` auto-discovers `register_*_task.ps1` and will pick up the four new ones automatically.

2. **CRLF normalization cycle.** Several existing `services/` scripts (see §7.1) are LF-only. A small mode-A brief to `unix2dos` the lot and add a CLAUDE rule about always running it after touching a Windows script would close this gap permanently.

3. **Phase 3 prerequisite check.** This Brief's Phase 3 patch unblocks the trades collector for whenever it gets re-registered. The `phase3_models.joblib` retrain (Recovery Plan Tier 2) still depends on `funding_rates` having a continuous future stream — which depends on the new funding scheduled task being live. Not a blocker for the retrain *today* (Cycle 9 backfilled the historical funding window), but Jeff should be aware that future retrains assume the funding scheduled task is running.

4. **Harness scope policy.** The harness denied `.\services\register_*.ps1` invocations even though the Brief explicitly authorized them. Worth checking whether this is a one-off (this session's permission state) or a standing posture (modes A/B universally cannot register scheduled tasks). If standing, future "verify existing tasks" Briefs should explicitly delegate the `Unblock-File` + `register_*.ps1` step to Jeff rather than asking Code to attempt and recover.

---

## 9. Kill switch / safety check

- Kill switch from Brief: don't edit `engines/crypto_data_collector.py`; only edit `services/trades_collector_service.bat` if Phase 3 case-a; no new tasks started by this Brief.
  - **Result: All three respected.** Engines untouched. trades bat patched (case-a confirmed). Zero new tasks started; `PraxisOrderBookCollector` was already running and unmodified.
- Cycle 8 OrderBook collector disturbed: **NO** — verified `Running` at session start AND session end.
- Real-money path touched: **NO** (collector services are read-only public API, no order execution).
- Source files modified: **0** — no `engines/`, `src/`, or `gui/` changes.
- Net delta: 9 new files in `services/` and `claude/retros/`, plus a 9-line patch to `services/trades_collector_service.bat`.
