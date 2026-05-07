# Retro: Cycle 30 -- onchain_btc scheduled task registration

**Brief:** `claude/handoffs/BRIEF_onchain_scheduled_task.md`
**Date:** 2026-05-07
**Mode:** Hybrid (Claude drafted brief + service files; Code committed;
USER ran the registration script)
**Status:** DONE
**Lineage:** Cycle 17 -- the `onchain_btc` table was added to
`primary_monitored` with the comment "No scheduled collector currently
registered; will alarm is_stale=true until one lands." Cycle 30 lands
the task.

---

## Summary

Registered `PraxisOnchainCollector` as a Windows Scheduled Task
running daily at 00:45 local Toronto time. Closes the
9.6-day-stale `onchain_btc` alarm that had been correctly
flagged by `get_collector_health` since 2026-04-28.

Files added (both ASCII-only per Windows cp1252 logging
constraint):
- `services/onchain_collector_service.bat`
- `services/register_onchain_task.ps1`

No code changes; no DB state changes; no edits to existing
files.

---

## Diagnosis trail

1. End-of-Cycle-26 health check showed `onchain_btc` `is_stale=true`,
   row_count=364, latest=2026-04-28T00:00:00 UTC, staleness=824k+
   seconds (~9.5 days).
2. Investigation:
   - `Get-ScheduledTask -TaskName "*nchain*"` returned empty.
   - Manual `python -m engines.crypto_data_collector collect-onchain
     --days 7` succeeded, fetching 6 metrics across 6 days,
     storing 6 days of on-chain data. Row count grew 364 -> 370.
   - `meta.py`'s `primary_monitored` block had a comment from
     Cycle 17 explicitly noting that no scheduled collector had
     been registered.
3. Conclusion: the collector code works; the only thing missing
   was task registration.
4. Note on subcommand naming: earlier briefs in this session
   referenced `collect-onchain-btc` based on memory; the actual
   subcommand is `collect-onchain`. Argparse's helpful error
   message listed all valid choices, surfacing the correct name
   immediately.

---

## Execution log

### Step 1: Code commits the new service files

Code added the two new files to `services/` and committed.

Commit `63993be` on origin/master.

### Step 2: USER registers the scheduled task

```powershell
cd C:\Data\Development\Python\McTheoryApps\praxis
.\services\register_onchain_task.ps1   # as Administrator
```

Output: `Task registered: PraxisOnchainCollector`. Daily at 00:45.

### Step 3: USER triggers an immediate run for verification

```powershell
Start-ScheduledTask -TaskName "PraxisOnchainCollector"
# Wait ~30s
Get-ScheduledTaskInfo -TaskName "PraxisOnchainCollector"
# LastTaskResult should be 0
```

### Step 4: Verification via MCP

`get_collector_health` reports `onchain_btc`:
- row_count: 370 (unchanged from the in-session manual
  `collect-onchain --days 7` test; the 11:34 immediate-trigger
  run inserted 0 new rows because the 7-day window was already
  fully covered by the manual test, idempotent via `INSERT OR
  IGNORE` on the `date` PK)
- latest: 2026-05-06T00:00:00 UTC (will advance to 2026-05-07
  after tomorrow's 00:45 fire, once blockchain.info publishes it)
- staleness_seconds: 142548 (39.6h, well below the 172800s /
  48h threshold)
- is_stale: false

Post-Cycle-30 system state: **all 11 monitored tables across the
3 Praxis SQLite databases now report `is_stale=false`** -- first
time since `onchain_btc` went stale on 2026-04-28.

---

## Notes

### Closes Cycle 17's standing TODO

`meta.py:220-225` contained the comment:
> onchain_btc -- Cycle 17: monitored via `date` column
> (YYYY-MM-DD, UTC midnight). 48h threshold matches daily-publish
> cadence plus one missed run of slack. **No scheduled collector
> currently registered; will alarm is_stale=true until one lands.**

Cycle 30 closes this loop. The comment can stay -- it's
historically accurate -- but the alarm condition is no longer
active.

### Hybrid workflow with files-included delta zip

Cycle 30 is a hybrid cycle but with files included in the delta
zip rather than relying on Code to write them from a brief. The
files are short (a bat file and a PS1 script), share well-
established structure with their `fear_greed_*` siblings, and have
strict ASCII-only requirements that are easier to enforce
upstream than to validate via Code's edits.

### Subcommand name lesson

The brief originally said `collect-onchain-btc`. The actual name
is `collect-onchain`. Worth a brief note: subcommand names should
be verified against the on-disk argparse dispatch table, not from
memory. The Python function is `collect_onchain_btc()` (with
`_btc` suffix) -- the dispatch name is just `collect-onchain`
because the argparse parser was set up that way. Easy to confuse;
worth checking when in doubt.

---

## Open items / next cycle inputs

Cycle 30 closes the LAST open item from this two-day session.
After the close-out commit, every loose end is tied:
- Schema migration program: 10/10 tables (Cycle 26 completion)
- Observability hardening: 6/6 CCXT collectors (Cycles 28+29)
- Primary-DB monitoring readout: working (Cycle 27.5)
- All 11 monitored tables `is_stale=false` (Cycle 30)

Carry-forwards (not session-blocking):

- **`register_all_tasks.ps1` update**: not in this cycle's scope.
  The existing all-tasks registration helper should probably
  include `register_onchain_task.ps1` as well, but that touches
  a multi-task script and warrants its own change.
- **MCP server restart at 22:19 UTC 2026-05-06 of unknown origin**:
  still unexplained. Probably benign.
