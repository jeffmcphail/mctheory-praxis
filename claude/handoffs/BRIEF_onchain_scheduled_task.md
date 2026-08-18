# Cycle 30 -- onchain_btc scheduled task registration

**Predecessor:** None directly; standing TODO since Cycle 17.
**Mode:** Files-included delta zip (small enough that Code's role is
just to commit + push; the actual scheduled-task registration is
done by the user via PowerShell as Administrator).
**Risk:** very low. Adds two new files; no edits to existing
files; no DB state changes.

## What

Registers `PraxisOnchainCollector` as a Windows Scheduled Task,
running daily at 00:45 local Toronto time. Closes the standing
"no scheduled collector currently registered" gap that has been
flagged in `meta.py`'s `primary_monitored` config since Cycle 17.

Two new files in this delta zip (matching the existing
`fear_greed_collector_service.bat` /
`register_fear_greed_task.ps1` template):

- `services/onchain_collector_service.bat` -- the runtime wrapper.
  Activates the venv, sets `PYTHONUTF8=1`, runs
  `python -u -m engines.crypto_data_collector collect-onchain
  --days 7`, logs to `logs/onchain_collector.log`.

- `services/register_onchain_task.ps1` -- the registration helper
  the user runs once as Administrator to create the scheduled
  task.

## Background

The `collect-onchain` subcommand (NOT `collect-onchain-btc` -- I
had the name wrong in earlier briefs and I'm sorry; the actual
on-disk dispatch entry per `crypto_data_collector.py:1256` is
`collect-onchain`) was tested manually in this session and worked
cleanly: pulled 6 metrics across 6 days from blockchain.info and
stored 6 days of on-chain data. The only thing missing has always
been the scheduled-task registration.

`onchain_btc` data is keyed by `date` (YYYY-MM-DD UTC midnight).
INSERT OR IGNORE provides idempotency, so pulling 7 days every
day creates harmless overlap and protects against single-day
misses.

The 48h staleness threshold in `meta.py` accommodates daily-
publish cadence + one missed run of slack. After this cycle and
one successful scheduled fire, `get_collector_health` should
report `onchain_btc` as `is_stale=false`.

## Specifics for Code

1. Add the two new files in `services/` (already in the delta
   zip; just commit them).

2. py_compile is not relevant (no Python files added).

3. Commit + push using the commit message at the bottom.

4. Do NOT register the scheduled task. That's a user step
   (requires admin shell + the user's session credentials for
   `S4U` logon type per the registration script).

## User steps after Code's commit lands

```powershell
# Run as Administrator (Right-click PowerShell -> Run as Administrator)
cd C:\Data\Development\Python\McTheoryApps\praxis
.\services\register_onchain_task.ps1

# Optional: verify and trigger immediately
Get-ScheduledTask -TaskName "PraxisOnchainCollector" | Select-Object TaskName, State
Start-ScheduledTask -TaskName "PraxisOnchainCollector"

# After ~30 seconds, verify it ran
Get-ScheduledTaskInfo -TaskName "PraxisOnchainCollector" | Format-List
# LastTaskResult should be 0
```

Then via MCP:

```
get_collector_health
```

`onchain_btc` should show `is_stale=false` and a recent `latest`
date.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | `services/onchain_collector_service.bat` committed |
| 2 | `services/register_onchain_task.ps1` committed |
| 3 | Both files use ASCII-only (Windows Task Scheduler logs through cp1252 -- per memory entry on encoding) |
| 4 | Both files mirror the existing `fear_greed_*` template structurally |
| 5 | The bat file invokes `collect-onchain --days 7` (NOT `collect-onchain-btc`; that is not a valid subcommand) |
| 6 | The PS1 registers as `PraxisOnchainCollector` daily at 00:45 |
| 7 | Commit message mentions Cycle 30 and the Cycle 17 lineage |

## Out of scope

- Adding new `--days` defaults or other CLI changes.
- Touching the `collect_onchain_btc` Python function.
- Touching `meta.py` -- the existing `onchain_btc` entry already
  expects 48h threshold and `date` format.
- Adding the new task to `register_all_tasks.ps1` (separate
  follow-up if needed; that file's behavior with this new task
  needs verification).

## Commit message (use this verbatim)

```
Cycle 30: onchain_btc scheduled task registration

Adds services/onchain_collector_service.bat and
services/register_onchain_task.ps1, mirroring the existing
fear_greed_collector_service.bat / register_fear_greed_task.ps1
template.

Closes the standing "no scheduled collector currently registered"
gap flagged in meta.py's primary_monitored config since Cycle 17.
The collect-onchain subcommand was tested manually in this session
and worked cleanly (6 days of metrics from blockchain.info via
INSERT OR IGNORE on the date PK).

After this commit, USER runs services/register_onchain_task.ps1 as
Administrator to register PraxisOnchainCollector as a daily task at
00:45 local Toronto time. After one successful scheduled fire,
get_collector_health should report onchain_btc as is_stale=false.
```
