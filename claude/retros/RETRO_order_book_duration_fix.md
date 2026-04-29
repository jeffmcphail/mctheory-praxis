# Retro: OrderBook Collector Duration Fix (COMPLETE)

**Date:** 2026-04-29
**Status:** COMPLETE -- all 6 acceptance criteria met
**Brief:** `claude/handoffs/BRIEF_order_book_duration_fix.md`
**Series / Cycle:** praxis / 8
**Mode:** A (single-line edit to a `services/*.bat` file; no collector logic touched)
**Scope:** XS (one duration value + comment block update)

---

## 1. TL;DR

`services/order_book_collector_service.bat` now invokes the collector with `--duration 3550` instead of `--duration 3600`, with the leading `REM` comment block updated to explain the 50-second buffer and reference the Cycle 7 / Cycle 8 diagnosis. The 50s cushion ensures the previous hourly invocation always exits before the next Task Scheduler trigger fires, eliminating the `MultipleInstances IgnoreNew` silent-skip race that was producing the 1-on / 1-off coverage pattern (~50% wall-clock collection). The change is confined to that one file. No collector code, no PowerShell registration script, no other `services/*.bat` files, and no live scheduled task were touched.

---

## 2. Diff

```
services/order_book_collector_service.bat | 9 ++++++---
1 file changed, 6 insertions(+), 3 deletions(-)
```

```diff
@@ -1,8 +1,11 @@
 @echo off
 REM Order Book Snapshot Collector
 REM Runs every hour via Windows Task Scheduler; each invocation polls
-REM Binance for BTC+ETH order books at 10s cadence for 3600 seconds,
-REM so back-to-back scheduling provides continuous coverage.
+REM Binance for BTC+ETH order books at 10s cadence for 3550 seconds,
+REM exiting 50s before the next hourly trigger to ensure clean handoff
+REM between back-to-back invocations and avoid the MultipleInstances
+REM IgnoreNew silent-skip race condition. See Cycle 7 retro and
+REM Cycle 8 brief for the diagnosis.
@@ -16,5 +19,5 @@ call "%VENV%"
-python -u -m engines.crypto_data_collector collect-order-book-loop --assets BTC ETH --interval 10 --duration 3600 >> "%LOG_FILE%" 2>&1
+python -u -m engines.crypto_data_collector collect-order-book-loop --assets BTC ETH --interval 10 --duration 3550 >> "%LOG_FILE%" 2>&1
```

---

## 3. Acceptance criteria check

| # | Criterion | Status |
|---|---|---|
| 1 | `--duration 3550` on the python invocation line | PASS (line 22) |
| 2 | Leading comment block reflects new duration + rationale | PASS (lines 4-8) |
| 3 | No other file in repo modified | PASS (`git diff --stat` confined to one file) |
| 4 | File remains ASCII-only with CRLF line endings | PASS (verified post-fix; see `4. Incidental issues` below) |
| 5 | `git diff` confined to `services/order_book_collector_service.bat` | PASS |
| 6 | Retro at `claude/retros/RETRO_order_book_duration_fix.md` | PASS (this file) |

`file services/order_book_collector_service.bat` after fix-up: `DOS batch file, ASCII text, with CRLF line terminators`.

---

## 4. Incidental issues

**Editor stripped CRLF on save.** The `Edit` tool in this harness silently rewrote the file with LF-only line endings, which `git diff` flagged with the warning "LF will be replaced by CRLF the next time Git touches it." This is exactly the pitfall the brief called out (line-ending conversion can break `.bat` execution in subtle ways). Caught immediately by inspecting `git diff` output and confirmed via `tr -cd '\r' | wc -c` (0 CR, 23 LF). Restored with `unix2dos services/order_book_collector_service.bat`. Re-verified: `file` reports CRLF, `tr` count shows 23 CR + 23 LF, `git diff` no longer warns.

**Lesson for future Mode-A edits to `.bat` / `.ps1` / other CRLF-required files:** always run `file <path>` after the edit and run `unix2dos` to restore CRLF if the editor stripped it. Adding this to the standing list of post-edit verification steps for Windows-script edits would prevent the round-trip.

No issues with ASCII (no smart quotes, em-dashes, or other Unicode introduced — Edit tool preserves raw bytes for the strings it was given, and the new `REM` lines were written in plain ASCII).

---

## 5. Manual follow-up for Jeff

The scheduled task is **not** restarted by this Brief — the next invocation will pick up the patched `.bat` automatically when it fires. If you want the fix to take effect immediately rather than waiting up to ~1 hour for the natural turnover, force-restart it:

```powershell
Stop-ScheduledTask -TaskName PraxisOrderBookCollector
Start-ScheduledTask -TaskName PraxisOrderBookCollector
```

Verify the fix is working by calling MCP `get_collector_health()` after the new invocation has been running for a few minutes. The `order_book_snapshots` staleness should track the 10s sampling cadence (well under the 3900-second threshold) and stay there continuously rather than oscillating into the multi-thousand-second range during the dead window.

---

## 6. Out-of-scope items flagged for future briefs

Per the brief's Phase 4 instruction: it's worth confirming whether **other** `services/*.bat` collectors share the same `--duration 3600` defect. Current coverage:

- `services/order_book_collector_service.bat` -- FIXED in this cycle.
- `services/trades_collector_service.bat` -- not inspected this cycle. If it uses an identical hourly trigger + 3600s duration pattern, it has the same defect and should be patched the same way (3600 -> 3550).
- `services/crypto_1m_collector_service.bat` -- same caveat. The 1-minute OHLCV collector may use a different scheduling model (continuous vs. hourly), in which case the `--duration` race is irrelevant. Worth a quick read.
- Any other `services/*.bat` files -- worth a sweep next time someone is in the area.

Recommend a follow-up Brief: "Audit all `services/*.bat` collector duration / trigger pairs for the same race condition." XS scope, ~20 minutes total. Would close out the class of defect across all collectors.

---

## 7. References

- Brief: `claude/handoffs/BRIEF_order_book_duration_fix.md`
- Source: `services/order_book_collector_service.bat`
- Reference (do not edit): `services/register_order_book_task.ps1`
- Diagnosis context: `claude/retros/RETRO_praxis_mcp_server.md` (Cycle 7 -- 65-min staleness threshold work that surfaced the underlying defect)
- Strategic context: `claude/handoffs/RECOVERY_PLAN_post_disk_failure.md` (`order_book_snapshots` is on the v8.3 microstructure thesis critical path, post-disk-failure accumulation reset)
