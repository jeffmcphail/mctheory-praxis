# Implementation Brief: Collector Outage Triage

**Series:** praxis
**Priority:** P1 (multiple production collectors dark; blocks funding carry wiring and microstructure data accumulation)
**Mode:** B (read scheduled task state, read logs, potentially invoke `Start-ScheduledTask`, touches live services per `WORKFLOW_MODES_PRAXIS.md`)

**Estimated Scope:** M (60–90 min: per-collector log + task inspection + one-shot restart attempts where safe)
**Estimated Cost:** none
**Estimated Data Volume:** read-only — scheduled task metadata + log files (expect <10 MB total). No DB writes. No API calls on behalf of collectors other than possibly invoking `Start-ScheduledTask` for the one whitelisted task below.
**Kill switch:** **Do not edit any code files, bat files, or Python scripts in this cycle.** If diagnosis reveals a code fix is needed, stop, write the retro, flag the fix as a scope-follow-up for Chat. The only permitted write action is `Start-ScheduledTask` on the explicitly whitelisted task.

Reference: follow rules 9–15 (Progress Reporting) from `CLAUDE_CODE_RULES.md`. Diagnostic task per `WORKFLOW_MODES_PRAXIS.md` Mode B (touches Windows scheduled tasks). 5-minute progress cadence applies.

---

## Context

Chat verified the Praxis MCP server is live and healthy. `get_collector_health()` at 2026-04-23 19:27 UTC returned:

| Table | Latest row | Staleness | Row count |
|---|---|---|---|
| trades | 2026-04-23 19:27:38 UTC | 3 sec | 1,361,455 |
| order_book_snapshots | 2026-04-23 **18:46:46** UTC | **41 min** | 6,498 |
| ohlcv_1m | 2026-04-23 **16:34:00** UTC | **2 h 53 min** | 520,951 |
| funding_rates | 2026-04-13 00:00:00 UTC | **10.8 days** | 2,190 |
| fear_greed | 2026-04-13 00:00:00 UTC | **10.8 days** | 900 |

Interpretation per-collector:

- **PraxisTradesCollector — HEALTHY.** Don't touch. Reference only.
- **PraxisOrderBookCollector — BROKEN.** Bat file runs `collect-order-book-loop --duration 3600` on an hourly schedule (back-to-back 1-hour invocations). Latest snapshot is 18:46 — the 18:00-schedule invocation crashed ~14 min before completion, and the 19:00 invocation never wrote a row. Current time 19:27 UTC means a 19:00 invocation should be deep into producing data. It isn't.
- **PraxisCrypto1mCollector — MAYBE HEALTHY, verify.** `CLAUDE_CODE_RULES.md` lists this as a **6-hour** schedule. Latest bar 16:34 UTC is consistent with a completed 16:30-ish run awaiting the next 22:30-ish run. The `is_stale` flag in MCP (>1 hr threshold) will always fire for a 6-hour batch collector in steady state. **Default hypothesis: fine, not broken.** Confirm by looking at `Get-ScheduledTaskInfo` `LastRunTime` and `LastTaskResult`.
- **Praxis Funding Monitor — BROKEN.** 10.8 days stale. Funding rate data should arrive every 8 hours (per the bat file's 8h schedule note). Not a batch-cadence explanation.
- **Praxis Sentiment Collector — BROKEN.** Also 10.8 days stale. Same April 13 cutoff as funding.

**The April 13 simultaneous stop on funding + sentiment is the strongest clue.** Two independent collectors failing on the same date strongly suggests a shared cause: an OS-level event (Windows update, cert rotation, credential expiry), a shared dependency upgrade that broke both, or a manual action that disabled both tasks. Check Windows Event Log for April 13 alongside the collector logs.

---

## Objective

Produce a diagnosis report (in the retro) for each of the four non-healthy collectors. For each, identify:

1. **Current scheduled task state** (Registered / Running / Ready / Disabled / Missing)
2. **Last run time and result code** (`LastRunTime`, `LastTaskResult`)
3. **Log file location and last meaningful output** — what did the collector say right before it stopped producing data? Stack trace? Connection error? Clean exit?
4. **Root cause category** — one of:
   - `task_disabled_or_missing` — Task Scheduler isn't trying to run it
   - `task_runs_script_crashes` — task fires, script exits non-zero
   - `script_runs_api_fails` — script starts, fails on external API (network, credential, rate limit)
   - `script_runs_silently_idle` — running but not producing data (WebSocket stuck, etc.)
   - `data_layer_issue` — producing data but DB writes failing
   - `actually_healthy` — false positive from stale-threshold (the ohlcv_1m hypothesis)
5. **Recommended next step**, one of:
   - `restart_no_risk` — task is registered, last-result is benign, no credentials at stake, can be kicked with `Start-ScheduledTask`
   - `escalate_simple_fix` — trivial fix identified (e.g., .env rotation, task re-enable), writeup goes to Chat for a follow-up Brief
   - `escalate_complex_fix` — needs investigation beyond this Brief's scope (code change, external API issue, migration)
   - `no_action_false_positive` — not actually broken

---

## Detailed Spec

### Phase 0 — Orientation (5 min)

```powershell
# From the repo root, in a regular (non-admin) PowerShell:
Get-ScheduledTask | Where-Object { $_.TaskName -like "*Praxis*" -or $_.TaskName -like "*Funding*" -or $_.TaskName -like "*Sentiment*" } | Select-Object TaskName, State, LastRunTime
```

Report the full output. If any of the five tasks (Trades, OrderBook, Crypto1m, Funding, Sentiment) is missing entirely, note it and skip further phases for that one.

**Reference `Running Services` section of `CLAUDE_CODE_RULES.md`** — the list there is slightly out of date (doesn't mention `PraxisOrderBookCollector` or `PraxisTradesCollector`). Use the actual `Get-ScheduledTask` output as ground truth, not the doc.

### Phase 1 — PraxisCrypto1mCollector verification (5 min, do first — cheapest)

Goal: confirm or refute the "actually healthy, waiting for next 6h run" hypothesis.

```powershell
Get-ScheduledTaskInfo -TaskName "PraxisCrypto1mCollector" | Format-List *
```

Check:
- `LastRunTime` — is it close to 16:30 UTC today?
- `LastTaskResult` — is it 0 (success) or a Win32 error code?
- `NextRunTime` — is it ~22:30 UTC?

If `LastTaskResult == 0` and `NextRunTime` is in the expected window → classify as `actually_healthy`, `no_action_false_positive`. Move on. **Do not restart it.**

If non-zero result code, also grab the tail of its log:

```powershell
Get-Content "C:\Data\Development\Python\McTheoryApps\praxis\logs\crypto_1m_collector.log" -Tail 100
```

(Log path is a guess based on the pattern from `order_book_collector_service.bat`. Verify actual path by reading `services/crypto_1m_collector_service.bat`.)

### Phase 2 — PraxisOrderBookCollector diagnosis (15 min)

```powershell
Get-ScheduledTaskInfo -TaskName "PraxisOrderBookCollector" | Format-List *
Get-Content "C:\Data\Development\Python\McTheoryApps\praxis\logs\order_book_collector.log" -Tail 200
```

Specifically look for:
- The tail of the 18:00-schedule invocation — what message preceded its exit at ~18:46? Exception? Network error? Clean `Loop invocation exited.`?
- Any 19:00 invocation output — is there any? If zero bytes written for the 19:00 run, the task didn't fire or the Python process crashed pre-logging.
- Any repeated pattern (same error over N consecutive invocations)

**Whitelisted restart:** if task state is `Ready`, `LastTaskResult` is not a catastrophic error (e.g., not `0x80070005` access denied), and the script exit looked clean (not mid-exception), you may invoke:

```powershell
Start-ScheduledTask -TaskName "PraxisOrderBookCollector"
```

Then wait 30s and query `get_order_book_snapshot("BTC")` via MCP — or just:

```powershell
Get-Content "C:\Data\Development\Python\McTheoryApps\praxis\logs\order_book_collector.log" -Tail 30
```

Report whether it started writing rows again. **One restart attempt only.** If it crashes again, document and move on — don't loop.

### Phase 3 — Praxis Funding Monitor diagnosis (15 min)

```powershell
Get-ScheduledTaskInfo -TaskName "Praxis Funding Monitor" | Format-List *
# (exact task name may differ — use Phase 0 output to confirm)
```

The bat file is `scripts/funding_alert.bat` running `scripts/funding_alert.py --gate 0.70`. It does NOT redirect output to a log file (unlike sentiment which does). So stdout/stderr get swallowed unless Task Scheduler captured them.

Investigation path:
1. Check `LastRunTime` vs `LastTaskResult`. If LastRunTime is in April 13 era and stayed there, the task is failing consistently.
2. Read `scripts/funding_alert.py` to understand what it expects (API, env vars, DB writes)
3. Try a manual one-shot invocation of the script (**not the scheduled task**) directly in the venv to see the actual stdout/stderr:

```powershell
cd C:\Data\Development\Python\McTheoryApps\praxis
.\.venv\Scripts\python.exe scripts\funding_alert.py --gate 0.70
```

Capture all output. If it fails, the traceback tells us which part is broken.

**Noted during Brief prep:** `scripts/funding_alert.bat` has a non-ASCII character in its REM comment (shows as replacement char when viewed). Per rule 20, this is a known smell. It's a comment so unlikely to be the root cause, but flag it in the retro alongside any other ASCII violations you find.

**Do NOT restart the scheduled task.** If the script is failing on credentials or an API change, restarting compounds the problem (repeat failures, possible rate limits).

### Phase 4 — Praxis Sentiment Collector diagnosis (15 min)

```powershell
Get-ScheduledTaskInfo -TaskName "Praxis Sentiment Collector" | Format-List *
# (confirm exact name via Phase 0)
Get-Content "C:\Data\Development\Python\McTheoryApps\praxis\data\sentiment_collector.log" -Tail 200
```

Sentiment's bat (`sentiment_collect.bat`) DOES redirect to `data/sentiment_collector.log`. Look for the last successful run (around April 13) and the first failed run after it. What changed?

Same manual one-shot diagnostic if needed:

```powershell
cd C:\Data\Development\Python\McTheoryApps\praxis
.\.venv\Scripts\python.exe -m engines.sentiment_tracker collect
```

**Do NOT restart the scheduled task.**

### Phase 5 — Cross-collector correlation (10 min)

Both funding and sentiment stopped on 2026-04-13. Check:

```powershell
# Windows Event Log around April 13 for system-level events
Get-WinEvent -LogName System -MaxEvents 200 | Where-Object { $_.TimeCreated -ge "2026-04-12" -and $_.TimeCreated -le "2026-04-14" -and $_.LevelDisplayName -in @("Error","Warning") } | Format-Table TimeCreated, Id, ProviderName, Message -AutoSize
```

Also check:
- `.env` file modification time — `Get-Item C:\Data\Development\Python\McTheoryApps\praxis\.env | Select LastWriteTime`. If it was touched around April 13, secrets may have rotated.
- Python version in the venv: `.\.venv\Scripts\python.exe --version` (unlikely but cheap)
- Available pypi package versions of whatever libs the two scripts depend on — were there upgrades?

Looking for evidence of a single root cause (Windows update, cert expiry, .env change, dependency upgrade, manual disable).

### Phase 6 — Write the retro

Per rules 31-34 and retention rules in `WORKFLOW_MODES_PRAXIS.md`. Name: `claude/retros/RETRO_praxis_collector_outage_triage.md`.

Retro structure (follow the template in `CLAUDE_CODE_RULES.md` lines 165+):

- **Summary** — one paragraph: what's broken, what's fine, what you did/didn't touch.
- **Per-collector diagnosis table** — one row each for OrderBook, Crypto1m, Funding, Sentiment, with columns: task state, last result, log tail summary, root cause category, recommended next step. This is the primary deliverable.
- **Cross-collector correlation findings** — did the April 13 simultaneous outage have a shared cause? What's the evidence?
- **Actions taken** — which scheduled task (if any) was restarted, outcome, any logs pulled.
- **Files touched** — should be zero code/bat files. Only the retro itself.
- **Open items for Chat** — for each collector that needs a fix, a one-liner Chat can turn into a follow-up Brief (e.g., "Funding collector fails on BinanceClient auth — rotate key in .env and re-run").

---

## Acceptance Criteria

1. Retro exists at `claude/retros/RETRO_praxis_collector_outage_triage.md` with the per-collector diagnosis table filled out for all four collectors (OrderBook, Crypto1m, Funding, Sentiment).
2. For each broken collector, root cause category is one of the six enumerated categories (not free-text guesswork).
3. No code files modified. No bat files modified. No Python files modified. Only the retro is written.
4. At most one `Start-ScheduledTask` invocation, and only against `PraxisOrderBookCollector`, and only if the whitelist conditions in Phase 2 are met.
5. If the April 13 funding + sentiment outages have an identified common cause, it's stated explicitly. If not, the investigation trail is documented so Chat can scope follow-up work.

---

## Known Pitfalls

- **Task names may not match the strings in `CLAUDE_CODE_RULES.md`.** Use Phase 0 output as ground truth. The rules doc is stale — don't edit it in this cycle, just note the drift in the retro's open-items section.
- **Funding + sentiment bat files handle logging differently.** Sentiment redirects to `data/sentiment_collector.log`; funding doesn't redirect at all. Don't spend time looking for a funding log file that may not exist — go direct-invoke for that one.
- **Admin PowerShell is not required for this Brief.** `Get-ScheduledTask`, `Get-ScheduledTaskInfo`, and `Start-ScheduledTask` all work from a standard user shell as long as the task was registered for the current user. Re-registering or modifying task definitions WOULD need admin — don't go there in this cycle.
- **Don't confuse `scripts/funding_monitor.py` with `scripts/funding_alert.py`.** The scheduled task runs `funding_alert.py`. `funding_monitor.py` is a separate dashboard/CLI tool — not the collector.
- **Don't confuse `engines/funding_rate_strategy.py` with either of the above.** That file is the strategy engine (the thing we eventually want to wire for live execution). It's not involved in the data collection outage.
- **Rate-limit safety on direct-invoke.** If you run `funding_alert.py` manually during Phase 3 and it succeeds, that call counts against whatever API's rate limit. One-shot is fine. Do not loop it to "verify."
- **MCP verification.** After any restart attempt, you can verify the data layer via a Chat-side MCP call — but don't wait on me. Just pull the log and move on; Chat will re-check `get_collector_health()` when reviewing the retro.
- **The trades collector is healthy — do not touch, do not restart, do not investigate.** Reference only.

---

## References

- `claude/WORKFLOW_MODES_PRAXIS.md` — Mode B criteria (scheduled tasks)
- `claude/CLAUDE_CODE_RULES.md` rules 9–15 (progress reporting), rules 16-20 (file safety), running-services table (noted stale, don't edit here)
- `services/order_book_collector_service.bat` — reference for log path pattern
- `services/crypto_1m_collector_service.bat` — verify actual log path
- `scripts/funding_alert.bat`, `scripts/funding_alert.py` — funding collector
- `sentiment_collect.bat`, `engines/sentiment_tracker.py` — sentiment collector
- Prior cycle retros in `claude/retros/` — for tone/format reference
