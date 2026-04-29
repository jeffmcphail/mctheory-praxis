# Implementation Brief: OrderBook Collector Duration Fix

**Series:** praxis
**Cycle:** 8
**Priority:** P1 — restoration of continuous microstructure coverage; gates the v8.3 microstructure thesis timeline
**Mode:** A (single-line edit to a batch file in `services/`; Mode A per `claude/WORKFLOW_MODES_PRAXIS.md` because it touches collector configuration but not collector code logic)

**Estimated Scope:** XS (5–10 minutes — one-line edit, one verification sequence)
**Estimated Cost:** $0
**Estimated Data Volume:** N/A (no new data, just fixes existing collection cadence)
**Kill switch:** No edits to `engines/crypto_data_collector.py`, no edits to the Python collection logic, no edits to other `services/*.bat` files. If the change requires touching anything other than `services/order_book_collector_service.bat`, stop and write a retro flagging the unexpected scope.

Reference: `claude/CLAUDE_CODE_RULES.md` rules 9-15 (progress reporting), rule 16 (validation), rule 19 (ASCII).

---

## Context

Memory item and Cycle 7 retro: the `PraxisOrderBookCollector` Windows Scheduled Task is configured to fire every 1 hour (`RepetitionInterval (Hours 1)`) and each invocation runs `--duration 3600` (i.e., the full 3600 seconds = 1 hour). Combined with `MultipleInstances IgnoreNew` in the task settings, this produces a 1-on/1-off coverage pattern:

- Hour N: invocation starts, runs to t=3600s, exits cleanly.
- Brief dead window after exit before the next hourly trigger fires (process exit + scheduler latency, typically 5-30 seconds).
- Hour N+1: next invocation fires at the scheduled trigger time. Since the previous instance has just exited, this one starts cleanly.
- However: in practice the Task Scheduler's hourly trigger plus the 3600s execution window can drift such that the next trigger sometimes fires *while* the previous instance is still running. With `MultipleInstances IgnoreNew`, the new trigger is silently skipped, leaving a full 1-hour gap.

**Net effect:** approximately 50% of wall-clock time has the collector active, producing ~30 minutes of actual order book data per hour rather than continuous coverage.

Cycle 7's MCP health-signal patch correctly identified this as the root cause of the apparent staleness on `order_book_snapshots` (the 65-minute threshold accommodates the 1-on/1-off pattern, but the underlying defect remains and the v8.3 microstructure thesis is accumulating data at half the intended rate).

**Strategic significance:** Memory item captures that v8.3 microstructure data accumulation is the open frontier (post-disk-failure data state). Every hour the collector runs at half-rate pushes the v8.3 readiness milestone back. With the disk crash having reset accumulation to zero, restoring continuous coverage is the single most time-sensitive collector fix — every day at half-rate is a day of v8.3 timeline lost.

---

## Objective

Change `services/order_book_collector_service.bat` to invoke the collector with `--duration 3550` instead of `--duration 3600`. The 50-second buffer ensures the previous invocation always exits cleanly *before* the next hourly trigger fires, eliminating the silent-skip race condition.

---

## Detailed Spec

### Phase 0 — Read the current state (1 min)

```bash
cat services/order_book_collector_service.bat
```

Confirm the current `--duration 3600` invocation is on the `python -u -m engines.crypto_data_collector collect-order-book-loop ...` line. If the file's current state differs from this (e.g., already shows 3550, or shows a different duration value, or lacks the duration argument entirely), stop and report — the assumption underlying this Brief is wrong.

### Phase 1 — Make the edit (1 min)

Single-character edit on the duration argument:
- Before: `python -u -m engines.crypto_data_collector collect-order-book-loop --assets BTC ETH --interval 10 --duration 3600 >> "%LOG_FILE%" 2>&1`
- After:  `python -u -m engines.crypto_data_collector collect-order-book-loop --assets BTC ETH --interval 10 --duration 3550 >> "%LOG_FILE%" 2>&1`

No other edits to this file. The comment block at the top of the file is now slightly stale ("each invocation polls Binance for BTC+ETH order books at 10s cadence for 3600 seconds, so back-to-back scheduling provides continuous coverage") — update it to match:

- Before: `REM Runs every hour via Windows Task Scheduler; each invocation polls`
         `REM Binance for BTC+ETH order books at 10s cadence for 3600 seconds,`
         `REM so back-to-back scheduling provides continuous coverage.`
- After:  `REM Runs every hour via Windows Task Scheduler; each invocation polls`
         `REM Binance for BTC+ETH order books at 10s cadence for 3550 seconds,`
         `REM exiting 50s before the next hourly trigger to ensure clean handoff`
         `REM between back-to-back invocations and avoid the MultipleInstances`
         `REM IgnoreNew silent-skip race condition. See Cycle 7 retro and`
         `REM Cycle 8 brief for the diagnosis.`

### Phase 2 — Verification (3 min)

The file is a Windows batch file, not Python, so there's no `python -c "import ast; ast.parse(...)"` to run against it. Verification is by inspection plus a smoke test:

1. **Visual inspection.** Re-`cat` the file and confirm only the duration value and comment block were edited.
2. **ASCII-only check.** Run `file services/order_book_collector_service.bat` (or equivalent) and confirm no non-ASCII bytes were introduced. Memory rule: Windows Task Scheduler is intolerant of non-ASCII characters in batch files. If the editor introduced any (em-dashes, smart quotes, etc.), strip them.
3. **Line-ending check.** Confirm the file uses CRLF line endings (Windows standard for `.bat` files). If it ends up LF-only after editing, batch interpretation can break in subtle ways. Use `file` or `dos2unix --info` to verify.
4. **No structural changes.** Other lines in the file (env vars, log path setup, venv activation, `cd /d`, `set PYTHONUTF8=1`) must be unchanged.

### Phase 3 — Trigger task restart (manual, by Jeff)

This Brief does NOT itself restart the scheduled task — that's Jeff's manual action after applying the delta. The retro should remind Jeff to do one of:

- Wait for the current running invocation (if any) to exit naturally, then the next hourly trigger picks up the patched `.bat`.
- Force-restart immediately:
  ```powershell
  Stop-ScheduledTask -TaskName PraxisOrderBookCollector
  Start-ScheduledTask -TaskName PraxisOrderBookCollector
  ```

The next time MCP `get_collector_health()` is called (after some minutes of running with the new duration), the response should show `order_book_snapshots` with stale-seconds well under the 3900-second threshold and approximately continuous coverage.

### Phase 4 — Retro

Standard retro at `claude/retros/RETRO_order_book_duration_fix.md`. Include:

- Confirmation that the only file edited was `services/order_book_collector_service.bat`.
- Diff of before/after for the affected lines.
- Reminder to Jeff about the manual scheduled-task restart step.
- Whether any incidental issues were observed (file encoding, line endings, comment block formatting).
- Note for future briefs: if `services/crypto_1m_collector_service.bat` and `services/trades_collector_service.bat` follow the same `--duration` pattern, they may have the same defect. NOT in scope for this cycle — but worth flagging for a future brief if so.

---

## Acceptance Criteria

1. `services/order_book_collector_service.bat` shows `--duration 3550` on the python invocation line.
2. The file's leading comment block reflects the new duration value and explains the rationale.
3. No other file in the repository is modified.
4. The file remains ASCII-only and CRLF-line-ended.
5. `git diff` shows changes confined to `services/order_book_collector_service.bat` only.
6. Retro file exists at `claude/retros/RETRO_order_book_duration_fix.md` with the required content.

---

## Known Pitfalls

- **Editor line-ending conversion.** Some editors silently convert CRLF to LF on save, which can break `.bat` execution in subtle ways (some commands work, others fail mysteriously). If the working environment auto-converts, configure the editor to preserve CRLF for `.bat` files, or use `unix2dos` after save.
- **ASCII-only enforcement.** The collector batch files are run by Windows Task Scheduler, which has historically had problems with non-ASCII characters. Specifically: don't let any em-dashes (—), smart quotes (" ", ' '), bullet points (•), or other Unicode characters slip into the comment block during the edit. Stick to plain ASCII hyphens, straight quotes, and `REM`.
- **Don't change `--interval 10`.** The collector samples at 10-second cadence. Changing this would alter the data semantics of the `order_book_snapshots` table, which is out of scope for this fix.
- **Don't change the `register_order_book_task.ps1` file.** The PowerShell registration script's `RepetitionInterval (Hours 1)` and `ExecutionTimeLimit (Minutes 65)` are correct as-is. Only the `.bat` file changes.
- **Don't restart the scheduled task as part of this cycle.** That's a manual step for Jeff after the delta is applied. Code attempting to restart it would touch a live system component, which is Mode B territory (live execution risk), not Mode A.

---

## What this Brief deliberately does NOT do

- No changes to `engines/crypto_data_collector.py` itself.
- No changes to other `services/*.bat` files (`trades_collector_service.bat`, `crypto_1m_collector_service.bat`, etc.).
- No changes to the PowerShell task registration scripts.
- No restart of the live scheduled task (Jeff does that manually).
- No verification of MCP health-signal output after the change (deferred to next session).
- No addressing of `funding_rates` and `fear_greed` orphan tables (separate Brief, queued).

---

## References

- `claude/WORKFLOW_MODES_PRAXIS.md` — Mode A definition.
- `claude/CLAUDE_CODE_RULES.md` — rules 9-15 (progress), rule 16 (validation), rule 19 (ASCII).
- `claude/retros/RETRO_praxis_mcp_server.md` — Cycle 7 retro that diagnosed the 1-on/1-off pattern as the root cause of `order_book_snapshots` apparent staleness.
- `services/order_book_collector_service.bat` — file to edit.
- `services/register_order_book_task.ps1` — reference for understanding the scheduling configuration (do NOT edit).
- `claude/handoffs/RECOVERY_PLAN_post_disk_failure.md` §1.2 item 1, §3.3 Tier 1 #1 — strategic context for why this is highest-priority among collector fixes after the disk failure.
