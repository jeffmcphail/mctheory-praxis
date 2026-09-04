# BRIEF — Cycle 64: `smart_money` Crash-Safety

**Estimated Scope:** M (1–2 hr)
**Estimated Cost:** none
**Mode:** Code (diagnosis + hardening + measured verification)
**Retro:** `claude/retros/RETRO_cycle64_smart_money_crash_safety.md`
**Priority:** Ahead of the MCRMINI1 → MCRMINI2 handover.

---

## Why this jumps the handover queue

Cycle 63 T4 established: the 2026-09-03 04:24 run died mid-loop at wallet
**416/1571**, never reached `conn.close()`, and nothing ran until a manual restart at
2026-09-04 09:05. Committed `snapshot_id`s jump straight across the outage:

```
20260903_022409   <- last good
20260904_130554   <- manual restart
```

**Five runs missed, permanently unrecoverable** — the Polymarket data-api returns current
positions only, so `position_snapshots` can only ever be built by sampling forward. A
missed snapshot has no backfill path.

**This is losing data now. The handover is losing none. It goes first.**

---

## T1 — Establish the crash cause BEFORE changing anything

Do not skip to hardening. If the run died at wallet 416/1571 there is a reason, and a
crash-safe wrapper around an unfixed fault just converts a loud failure into a quiet
retry loop.

1. What is in the log at and around the death point — network timeout, API error, rate
   limit, unhandled exception, OOM, process kill?
2. Is wallet 416 in that ordering reproducible, or incidental?
3. Had it failed the same way before? Check for earlier `snapshot_id` gaps across the
   table's **full history** and report the frequency. This may not be the first
   occurrence, only the first noticed — and if it has been recurring for months, that
   changes what T2 has to survive.

Report the cause, **or state plainly that it cannot be determined from what was logged**
— which is itself a finding about the logging.

## T2 — Crash-safety on the connection

Currently one WAL connection is held across 1,586 wallets for ~21 minutes, 4x daily. That
single long transaction is why a mid-loop death orphans the sidecars.

- Context-manage the connection so it closes on **any** exit path, including unhandled
  exceptions and SIGTERM.
- **Commit incrementally** rather than once at the end. A run that dies at wallet 416
  must retain wallets 1–415, not lose all 1,586. Batch size a parameter.
- Per-wallet failures must not kill the run: catch, record, continue, and report the
  failed set at the end.
- The process **MUST non-zero-exit when the run is incomplete** — the standing collector
  rule. Distinguish "1,586 of 1,586" from "416 of 1,586" in the **exit code**, not only
  in the log.

## T3 — Make the gap detectable without a human noticing

Five missed runs went unremarked for 29 hours. The collector knew it had died; nothing
else did.

- Record incomplete/failed runs in `collector_gaps` (collector, venue, timestamp), same
  pattern as the liquidation collector — open on failure.
- `smart_money` is **not** backfillable, so those gaps are **terminal by construction**.
  Mark them unfillable rather than pending, so nothing retries forever.
- Add a staleness check: if the most recent `snapshot_id` is older than the expected
  cadence by more than a stated margin, exit non-zero and fire the existing
  `PRAXIS_ALERT_URL` push. Reuse the `funding_regime_alert` pattern.

> There is no T4. An earlier draft asked for the connection-hold window so the mirror
> could be scheduled around it. `ai_factory_main_current` has a better fix:
> `backup_mctheory.bat` currently mirrors live `.db` / `-wal` / `-shm` files that are
> already covered by `VACUUM INTO` plus restic, so **excluding** them removes the
> collision outright rather than scheduling around it. **Do not measure the window.**

---

## Constraints

- Read-only against `crypto_data.db`; `smart_money.db` is the target.
- No `VACUUM`, no migration on the 24 GB main DB.
- **Verify by measurement:** kill a run mid-loop deliberately and confirm partial results
  persisted, sidecars closed, exit code non-zero, gap recorded. **Do not verify by reading
  the new code.**
- **Scheduling:** keep the deliberate-kill test away from **03:00 and 03:30** — those are
  the mirror and backup windows on this estate.
- Everything a parameter; `--validate` / `--verbose` defaulting to maximum.

---

## Hand back

Retro containing:

1. The T1 crash cause, or an explicit "not determinable from logs"
2. The historical gap frequency across the full table
3. Proof-by-deliberate-kill that T2 and T3 work — partial results retained, non-zero
   exit, terminal gap recorded

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-09-04 (Chat: praxis_main_current)*
*Changes: Initial brief. Cycle 64 promoted ahead of the handover because smart-money is
actively losing unrecoverable data, per the Cycle 63 T4 finding.*
