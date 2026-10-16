# BRIEF - Cycle 67: Backup Integrity

**Estimated Scope:** M (1-2 hr)
**Estimated Cost:** none
**Mode:** Code (diagnosis + fix), with one step that needs Jeff's admin hands
**Retro:** `claude/retros/RETRO_cycle67_backup_integrity.md`
**Predecessor:** Cycle 66 (`43c529b`) - open items 5 and 6 of
`claude/retros/RETRO_cycle66_a2_labelled_validation.md`

---

## Why this cycle exists

Cycle 66 was a read-only research cycle. While checking why `McTheoryMirror` exited 0x1 it
found something worse and unrelated: **`crypto_data.db` has not been backed up since
2026-09-27.** Two independent failures, both silent, both for the same structural reason -
the component that would have reported the failure is the component that failed.

The immediate data exposure is already closed. On 2026-10-15 the Bybit liquidation labels
were copied to Dropbox as a parquet, covering rows through 2026-10-15 06:06 UTC. **This
cycle is about the ongoing failure, not an emergency.**

**There is a deadline.** The snapshot grows about 1.3 GB a week and the miss is a timing
race that widens every week, so **Sunday 2026-10-18 will miss too** unless T1 lands first.
If it does not land by Saturday night, repeat the manual copy on Sunday morning and the
deadline goes soft - do not rush a change to the backup path to beat a clock.

### The two failures, as Cycle 66 measured them

**The large-database race.** On full days (Sundays) `backup_to_dropbox.ps1` runs at 03:30
and takes `VACUUM INTO` snapshots of the large databases; `backup_large_dbs.ps1` runs at
03:45 and writes whatever is staged to restic. The snapshot now finishes after the restic
job starts:

| Sunday | snapshot finished | restic started | backed up |
|---|---|---|---|
| 09-27 | 03:44:39 | 03:45:03 | yes, with 24 seconds to spare |
| 10-04 | 03:50:52 | 03:45:05 | **no** |
| 10-11 | 03:54:23 | 03:45:06 | **no** |

Both misses logged `Nothing over 100MB staged. On a delta day the large databases are
deferred, so this is expected.` That sentence is true on a delta day and meaningless on a
full day, so a missed staging is indistinguishable from a correct deferral.

**The mirror.** `McTheoryMirror` exits 0x1 at 03:00:01 nightly; the newest file it ever
captured is dated 2026-09-04 09:05. There is no `mirror_*.log` within the 30-day retention,
because the logging is what broke. The hypothesis, with four confirmation steps, is in
Cycle 66 open item 5: the 09-04 edit replaced `%DATE%` slicing with `wmic os get
LocalDateTime` (`backup_mctheory.bat:39-44`), and WMIC is absent from this Windows build, so
`LDT` is never set, `TIMESTAMP` expands to garbage, `/LOG+:"%LOGFILE%"` is malformed and
robocopy rejects its parameters before copying anything.

**The mirror never carried `crypto_data.db`.** It excludes `*.db`, `*.db-wal` and `*.db-shm`
by design (`backup_mctheory.bat:19-25`, `:64`). The two failures are independent; only the
race touches the database.

---

## T1 - Fix the large-database race. Do this first; it has the deadline.

**The requirement, not the implementation:** the handoff from staging to shipping must not
depend on wall-clock timing. State which shape was chosen and why.

**Rejected, and say so in the retro if it is chosen anyway:** moving the restic job later.
At the current drift that buys roughly ten weeks and then fails the same way, silently. It
converts a present failure into a future one.

**Suggested shape.** Have the shipping job wait on a sentinel written by the staging job,
with a bounded timeout. If the sentinel does not appear in time, ship whatever is staged
**and log loudly that staging did not finish** - so a hang degrades to a reported partial
backup rather than a silent total one.

**The constraint that rules out the obvious alternative:** a straight chain, where staging
invokes shipping on completion, means one long `VACUUM INTO` blocks every other backup on
the machine. Whatever shape is chosen, **a hang in staging must not prevent the rest of the
backup from running.**

A cleaner long-term shape, worth noting but not required here: have the staging job push the
large databases to restic itself, so there is no handoff to get wrong.

## T2 - Make the silence loud

On a **full** day, nothing staged is an error, not a deferral. The message that hid two
missed Sundays must distinguish the two cases and fail visibly on the wrong one.

This is the smaller change and the one that would have caught the problem on 10-04 rather
than 10-15. Do it even if T1 slips.

## T3 - A staleness check that does not ask the jobs how they are doing

Every check in this path currently interrogates the **describing** layer: did the job report
success. Both failures passed that test. Add a check that reads the **acting** layer - query
the restic repository (and the Dropbox copy) for the newest `crypto_data.db` snapshot and
alert if it is older than a stated number of days.

**It must not share code or scheduling with the jobs it checks**, or it inherits their
blind spots. An independent check that can fail is the point.

Suggested threshold: 9 days, so one missed Sunday is tolerated and two are not.

## T4 - The mirror

Run the four confirmation steps already written in Cycle 66 open item 5, then fix. If the
WMIC hypothesis holds, the replacement is `Get-CimInstance` or a `%DATE%`/`%TIME%` parse that
does not depend on a deprecated Feature on Demand.

**Confirm before fixing.** Chat asserted a date and a cause for this failure on 2026-10-15
and was wrong on both; Cycle 66 corrected it by dating the script edit and reading the
exclusion list. Establish the cause from the machine, then fix it.

Lower priority than T1-T3: the mirror never carried the database, and restic, Dropbox and
the restore test all ran normally.

---

## Constraints

- **`crypto_data.db` is roughly 24 GiB and its backup posture is already degraded.** Treat
  anything destructive to it - `VACUUM`, schema migration, bulk delete - as needing a fresh
  verified snapshot first. This cycle should need none of those.
- Do not delete any existing backup, snapshot or archive.
- Keep test runs away from **03:00, 03:30 and 03:45**. If a dry run must exercise the real
  path, use a copy or a different target, not the live repository.
- **Task Scheduler registration is Jeff's hands (admin).** Provide the exact command; do not
  attempt registration.
- ASCII-only commit messages.

## Acceptance

**Read the destination, not the log.** The cycle is done when a Sunday run puts
`crypto_data.db` into restic and that is verified by querying the restic repository for the
snapshot, not by reading what the job said about itself. Until a real Sunday has passed,
state plainly that acceptance is pending rather than inferred.

## Hand back

1. **T1:** which shape was chosen, why, and what happens when staging hangs
2. **T2:** the full-day vs delta-day distinction, and what the message now says in each case
3. **T3:** the staleness check, its threshold, and confirmation that it shares no code or
   schedule with the jobs it checks
4. **T4:** the mirror cause **as established from the machine**, with the evidence, then the fix
5. Any Task Scheduler command needing Jeff's admin hands, ready to paste
6. Whether Sunday 2026-10-18 is expected to succeed, and what to check on Monday

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-10-16 (Chat: praxis_main_current)*
*Changes: initial brief. Covers Cycle 66 open items 5 and 6 - the large-database backup race
(deadline: Sunday 2026-10-18) and the silently failing local mirror.*
