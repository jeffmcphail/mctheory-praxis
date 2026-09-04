# RETRO — Cycle 64: `smart_money` Crash-Safety

**Date:** 2026-09-04
**Mode:** Code (diagnosis + hardening + measured verification)
**Brief:** `claude/handoffs/BRIEF_cycle64_smart_money_crash_safety.md` (`b917e64`)
**Constraints honoured:** `crypto_data.db` read-only throughout; no `VACUUM`, no migration
on the 24 GiB DB; kill tests run at ~16:15–16:25, far from the 03:00/03:30 mirror and
backup windows. T4 was cancelled before work began and was not measured.

---

## T1 — The crash cause

### It was not a crash in this collector. **The host died.**

Windows System event log, and it is unambiguous:

```
Event 6008: "The previous system shutdown at 4:25:57 AM on 9/3/2026 was unexpected."
Event 41  : "The system has rebooted without cleanly shutting down first. This error
             could be caused if the system stopped responding, crashed, or lost power
             unexpectedly."                                    (logged 2026-09-04 08:59:53)
```

Corroborated independently — **every collector on the box stops inside the same four
minutes and resumes together after the reboot**:

| Log | last line 09-03 | first line 09-04 |
|---|---|---|
| `crypto_1m_collector` | 03:24:05 | 09:05:23 |
| `order_book_collector` | 03:35:47 | 09:05:23 |
| `bybit_liquidation_collector` | 04:00:03 | 09:05:23 |
| `trades_collector` | 04:23:57 | 09:05:23 |
| `smart_money` | 04:24:03 (progress to 04:30:23) | 09:05:23 |
| `info_bars_collector` | **04:28:35** | 09:03:34 |

So the machine died at ≈04:28–04:30 and returned at ≈09:00 on 09-04 — **~28.5 hours of
whole-estate downtime.**

### This corrects Cycle 63

Cycle 63 T4 concluded the 09-03 run "died mid-loop … and never reached `conn.close()`,"
leaving orphaned sidecars, and read the 09-04 09:05 run as a manual restart. That was
wrong on both counts. smart_money was simply the process that happened to be mid-loop when
the host went down, and 09:05 was the first post-boot scheduled run, not a human. The
orphaned sidecars were a *consequence* of the power loss, not a cause of anything. Two
further statements in that retro do not survive either: no mirror ran at 09:05 on 09-03
(the machine was off), and the "~21 min held connection 4× daily" figure, while correctly
measured, is not what broke anything.

### 2. Is wallet 416/1571 reproducible?

**No — purely incidental.** It is where the loop happened to be when power was lost.
Nothing in the code or in wallet 416 is implicated. `fetch_positions` already wrapped every
request in `except Exception`, so a network fault at any wallet would have been swallowed
and the loop would have continued; a per-wallet fault could not have produced this.

### 3. Had it happened before? Yes — roughly monthly.

Full `position_snapshots` history, 504 snapshots over 127 days at a 6h cadence:

| Gap | Duration | Runs missed |
|---|---|---|
| 2026-05-08 08:24 → 20:24 | 12.0h | ~1 |
| 2026-06-10 02:24 → 14:24 | 12.0h | ~1 |
| 2026-07-14 14:24 → 07-15 08:24 | 18.0h | ~2 |
| 2026-08-29 14:24 → 08-30 02:24 | 12.0h | ~1 |
| **2026-09-03 02:24 → 09-04 13:05** | **34.7h** | **~5** |

**5 gap events, 10 runs missed, 98.8% coverage, mean 25.4 days between events.** So the
brief's suspicion was right: this was not the first occurrence, only the first noticed —
and the 09-03 event is 3–5× larger than any prior one.

**Every gap that can be checked is also a host event**, not a collector fault:

- **07-14** — planned Windows Update restart (`MoUsoCoreWorker.exe` → "Service pack
  (Planned)", then `TrustedInstaller.exe` → "Upgrade (Planned)")
- **08-29** — unexpected shutdown at 16:12:08
- **09-03** — unexpected shutdown at 04:25:57
- 05-08 and 06-10 — **cannot be determined**; the System log only retains back to
  2026-06-10 00:26

### The finding that outgrew the brief

Querying the whole retained window rather than just 09-03 turned up **five dirty reboots in
the last seven days**, including one **today at 11:55:02** (rebooted 12:34). Current uptime
at the time of writing was 3.6 h.

```
2026-08-29 16:12:08   unexpected shutdown
2026-09-01 07:14:17   unexpected shutdown  (+ cascade at 09:23, 09:25, 09:30)
2026-09-03 04:25:57   unexpected shutdown  <- the 34.7h outage
2026-09-04 11:55:02   unexpected shutdown  <- today
```

**MCRMINI1 is unstable, and that — not this collector — is the top-priority problem.** It
also reframes the handover: MCRMINI2 stops being a scheduling chore and becomes the fix.

### A finding about the logging itself

The **Task Scheduler Operational log is disabled** (`IsEnabled: False`), so there is no
scheduler-side record of task starts, terminations, or skipped triggers. Combined with
`PraxisSmartMoney`'s `MultipleInstances: IgnoreNew`, a missed trigger leaves **no trace
anywhere**. Enabling that log is cheap and would have made this diagnosis immediate.

---

## T2 — Crash-safety on the connection

`engines/smart_money.py`, `cmd_snapshot` rewritten. Note honestly what each change does
and does not buy, given T1:

| Change | Helps against a host power loss? |
|---|---|
| `contextlib.closing` on the connection | **No** — no `finally` runs when the machine dies. It fixes the unhandled-exception path only. |
| **Incremental commit every `--batch-size` wallets** | **Yes.** This is the one that matters. The 09-03 run lost all 1,586 wallets purely because nothing had been written yet. |
| Per-wallet failure isolation | Independent robustness; unrelated to the outage. |
| Non-zero exit on incomplete | Detection, not prevention. |

- Connection context-managed, closes on every exit path.
- **Incremental commit**, `--batch-size` (default 50). Worst-case loss is now bounded to
  one batch instead of the entire run.
- Per-wallet failures are caught, recorded, and reported at the end. This required a change
  to `fetch_positions`: it previously swallowed errors and returned `[]`, making a failed
  fetch indistinguishable from a wallet holding no positions. It now takes
  `raise_on_error` (default `False`, so every other caller is unaffected).
- Signal handlers for SIGINT/SIGTERM/SIGBREAK set a cooperative stop flag. Stated plainly:
  on Windows only SIGINT/SIGBREAK are reliably deliverable — `taskkill /F` and a power loss
  run no handler at all. The handlers make a *polite* stop tidy; the incremental commit is
  what actually protects data.
- **Honest exit codes**, and `main()` now calls `sys.exit()` — it previously returned
  nothing, so the process exited 0 no matter what happened:

```
0 = complete    1 = incomplete    2 = stale    3 = fatal
```

- `services/smart_money_service.bat` now passes `--alert` and **preserves the snapshot's
  exit code** instead of collapsing every failure to 1. The original
  `endlocal & exit /b !SNAP_RC!` idiom would have discarded the variable before the exit
  could read it; `exit /b` ends the batch and implicitly ends `setlocal`, so the explicit
  `endlocal` is both unnecessary and harmful here.

---

## T3 — Gap detection without a human

- `collector_gaps` created **in `smart_money.db`** (the brief holds `crypto_data.db`
  read-only), column-for-column identical to the Cycle 62A table so the two can be
  consolidated later without a migration.
- Gaps are written **CLOSED and `UNFILLABLE:`-prefixed**, never left open. smart_money has
  no backfill path — the Polymarket data-api serves current positions only — so an open
  gap would invite a retry that can never succeed. **Measured: 0 open gaps.**
- Staleness check runs at the start of every snapshot and as a standalone `staleness`
  subcommand: if the newest snapshot is older than `--cadence-hours` + 
  `--staleness-margin-hours` (6 + 2 by default), it records a gap, fires the
  `PRAXIS_ALERT_URL` push (same `resolve_alert_url` → `post_alert` shape as
  `scripts/funding_regime_alert.py`, with the legacy `TEAMS_WEBHOOK_URL` fallback), and
  exits 2.
- New `gaps` subcommand to inspect what has been recorded.
- Everything parameterised: `--db`, `--batch-size`, `--cadence-hours`,
  `--staleness-margin-hours`, `--alert`, `--limit`, `--verbose` (default **3**, maximum),
  `--validate` / `--no-validate` (default **on**).

---

## Verification — by deliberate kill, not by reading the code

Run against a scratch DB (`--db`) seeded with 400 real wallets, so no partial snapshots
were written into production.

### Test A — hard kill (`taskkill /F`), the power-loss analogue

No handler runs, which is exactly what happened on 09-03.

```
launched PID 10924, batch-size 10
after 30s: 33 distinct wallets committed
taskkill /F  ->  SUCCESS: The process with PID 10924 has been terminated.

snapshot 20260904_201641: 487 rows, 33 distinct wallets   <-- RETAINED
PRAGMA integrity_check: ok
sidecars: -wal / -shm gone after recovery open
collector_gaps rows: 0
```

**Under the old code this would have been 0 rows.** The database is intact and the
sidecars are cleaned. `collector_gaps: 0` is the honest limitation — a hard-killed process
cannot record its own gap, which is precisely why the staleness check in T3 exists: the
*next* run is what notices.

### Test B — graceful stop (CTRL_BREAK), the exit-code and gap path

```
new snapshot reached 25 wallets after 24s; sending CTRL_BREAK
  STOP REQUESTED (signal 21) -- committing current batch and closing cleanly
  Snapshot 20260904_201923: 32/400 wallets (INCOMPLETE)
  GAP RECORDED  smart_money/polymarket 2026-09-04T20:19:23Z
exit code: 1
```

**The first attempt at this test was invalid and is worth recording.** My poll matched
Test A's leftover snapshot, so the signal was sent at t=0 before handlers were installed;
the process died under the default Ctrl-C handler with `0xC000013A`. Re-running against
the new snapshot id fixed the test — but it then exposed a real defect: a
`UnicodeEncodeError` on the box-drawing separator under a cp1252 stdout crashed the
display *after* the gap was recorded. **An unhandled exception also exits 1** — the same
code as `EXIT_INCOMPLETE` — so a cosmetic rollup could forge or mask the run's real exit
status. The display is now ASCII-only and wrapped in its own `try/except`, and the test
re-run confirms the exit code comes from the clean path:

```
exit code: 1
traceback in output? NO - exit code is from the clean path
```

### Terminality — nothing retries forever

```
UNFILLABLE: incomplete run 20260904_201758 -- 33/400 wallets (stopped: signal 21)
UNFILLABLE: incomplete run 20260904_201923 -- 32/400 wallets (stopped: signal 21)
closed(gap_end set)=True   unfillable_prefix=True   (both rows)
OPEN/pending gaps: 0
```

### Test C — staleness

```
forced stale (cadence 0h, margin 0h): age=0.01h limit=0.00h -> STALE   exit 2  + alert
control  (cadence 6h, margin 2h)    : age=0.01h limit=8.00h -> ok      exit 0
```

### Control — a healthy run must not cry wolf

```
Snapshot 20260904_202029: 30/30 wallets (complete)
committed through wallet: 30/30
exit code: 0 (OK)
gaps before: 2   gaps after: 2   (unchanged)
```

### Production — the scheduled 16:24 run exercised the new code end to end

Rather than simulate, the naturally scheduled `PraxisSmartMoney` run at 16:24 was allowed
to execute the new code and was observed live:

```
STALENESS: latest=2026-09-04 14:24:08Z age=6.00h limit=8.00h -> ok
POSITION SNAPSHOT - 20260904_202408
Tracking 1594 wallets  (batch_size=50, commit every 50 wallets)
  COMMIT through wallet 50/1594
  COMMIT through wallet 100/1594
```

**The decisive observation.** Mid-run, a *separate read-only connection* could already see
committed rows:

```
('20260904_202408', 727 rows, 84 distinct wallets)   <-- visible while the run continues
```

Under the old single-commit-at-the-end code that query returns **zero rows** until the
final second. This is precisely the property whose absence cost all 1,586 wallets on
09-03, demonstrated working against production data.

The run then completed cleanly, which is the control that matters most in production:

```
Snapshot 20260904_202408: 48811 positions from 1594/1594 wallets (complete)
committed through wallet: 1594/1594
exit code: 0 (OK)
VALIDATE (read-back):
  latest snapshot_id : 20260904_202408
  rows committed     : 12227
  distinct wallets   : 1344
  collector_gaps rows: 0
[2026-09-04 16:52:36.83] Snapshot complete.
```

Exit 0 propagated through the bat, and **no gap was recorded on a healthy run** — the
detection does not cry wolf.

Also verified against the real `smart_money.db`: `staleness` → ok, exit 0; `gaps` → table
created and empty; `position_snapshots` schema unchanged, 504 prior snapshots intact.

### An unexpected timing result, and whether this cycle caused it

That run took **28m33s** (16:24:03 → 16:52:36) against a prior ~20–21 min. Loop throughput
fell from 77.3 to **55.8 wallets/min, a 28% slowdown**. Since this cycle added 31 commits
to that loop, the honest question is whether the hardening caused it. It did not:

- **The dips are not commit-aligned.** With `--batch-size 50`, a commit cost would show a
  sawtooth at wallets 950, 1000, 1050… The 17 slow intervals land at 945, 1062, 1073,
  1108, 1157, 1167, 1178, 1204, 1227, 1237, 1273… — scattered, and clustered in the second
  half of the run. The median fell uniformly rather than periodically.
- **Measured commit cost.** Benchmarking a representative ~384-row batch (12,227 rows ÷
  1,594 wallets × 50):

```
mean 3.3 ms per commit, max 8.7 ms
projected cost of 31 commits : 0.10 s
slowdown to be explained     : ~480 s
```

Commits account for **0.02%** of the gap. (The benchmark ran on the scratch DB, but SQLite
WAL commit cost scales with dirty pages in the transaction — which is identical either way
— not with database size; even a 100× penalty would be 10 s, not 480 s.)

Conclusion: the slowdown is **external Polymarket API latency at that hour**, not this
cycle. Worth confirming across the next few runs rather than treating one run as a trend.

### The operational risk that finding exposes — independent of its cause

`PraxisSmartMoney` carries `ExecutionTimeLimit: PT30M`. A 28m33s run leaves **87 seconds of
margin.** One more slow API hour and Task Scheduler terminates the run mid-loop.

The irony is that this is now a much softer failure than it was this morning: the partial
data survives, and the next run's staleness check reports it. But it should be raised
anyway. **Not changed here** — modifying a scheduled task is outside this brief.

---

## Bug found en route — and it is handover-critical

`init_db()`'s `CREATE TABLE IF NOT EXISTS position_snapshots` had **drifted from the live
table**. It still described the pre-Cycle-25 shape — an `AUTOINCREMENT id`, `timestamp` as
`TEXT`, and **no `datetime` column at all** — while `_insert_position_row` writes the
migrated Rule 35 shape (ms `timestamp` + ISO `datetime`, compound PK).

On the existing database this is invisible: `IF NOT EXISTS` is a no-op against the
already-migrated table. **On a fresh machine it is fatal** — `init_db()` would create the
stale schema and every insert would fail with `table position_snapshots has no column
named datetime`.

That fresh machine is **MCRMINI2**. Had this cycle not run, the smart-money collector would
have come up on the new host and failed on its first write. Fixed to match the live schema
exactly, and verified by building the scratch DB through the fixed `init_db()`:

```
['snapshot_id','timestamp','datetime','wallet','market_slug','market_title',
 'outcome','size','avg_price','current_price','value_usd','pnl_usd']
```

---

## A second encoding defect, found while verifying — and why it mattered

Verifying `discover` (which the service bat runs *before* `snapshot`, and whose failure
increments `FAIL_COUNT`) reproduced the same `UnicodeEncodeError` class under a cp1252
stdout. Production was masked from it only because the bat sets `PYTHONUTF8=1`; any ad-hoc
invocation crashed at the separator.

This is not cosmetic in a cycle about exit-code honesty: **an unhandled exception exits 1,
which is also `EXIT_INCOMPLETE`**, so a display failure could forge or mask a run's real
status. Fixed systemically rather than by hunting characters — `sys.stdout` / `sys.stderr`
are reconfigured to UTF-8 with `errors="replace"` at import, covering every subcommand
including `diff` / `signals` / `monitor` / `profile`. The ASCII-only snapshot display stays
as defence in depth.

Verified: `discover --category CRYPTO` under a cp1252 stdout now exits 0 with no traceback
where it previously raised; the complete-run control still exits 0 at 12/12 with no
traceback. Committed separately as `0757bd8`.

## Files changed

| File | Change |
|---|---|
| `engines/smart_money.py` | Crash-safety, incremental commit, per-wallet isolation, exit codes, `collector_gaps`, staleness + alert, `gaps` / `staleness` subcommands, `--db`, ASCII-safe output, **`init_db` schema-drift fix** |
| `services/smart_money_service.bat` | `--alert`; preserve the real exit code instead of collapsing to 1 |
| `claude/handoffs/BRIEF_cycle64_smart_money_crash_safety.md` | Brief (committed standalone, `b917e64`) |

---

## Carried forward

1. **MCRMINI1 host instability is now the top item** — 5 dirty reboots in 7 days, one
   today. Nothing in this cycle addresses it, and it is the actual cause of all measurable
   data loss. Worth diagnosing (PSU / thermal / driver / Windows Update reboots) or
   treating as further reason to accelerate the handover.
2. **Enable the Task Scheduler Operational log.** Currently disabled; with
   `MultipleInstances: IgnoreNew`, a skipped trigger leaves no trace anywhere.
3. **Raise `ExecutionTimeLimit` on `PraxisSmartMoney` — now urgent, not theoretical.** It is
   `PT30M`, and the 16:24 run took 28m33s: **87 seconds of margin.** Measured as external
   API latency rather than anything this cycle changed, but the next slow hour terminates a
   healthy run mid-loop. The wallet count is also growing (1,571 → 1,586 → 1,594 in two
   days), so the margin shrinks on its own.
4. The other collectors have the same single-commit-at-the-end shape and would lose the
   same way on the next host crash. Not audited this cycle.
5. Cycle 63's T4 section should be read alongside the correction at the top of this retro.
6. **A third live-order path exists in the repo.** Checking for callers turned up
   `engines/smart_money_alerts.py`: `cmd_trade` reads `POLYMARKET_PRIVATE_KEY` and builds a
   `py_clob_client.ClobClient` behind a `--live` flag. It is **not scheduled** — no service
   bat references it — so Cycle 63's T1 and T2 conclusions are unaffected and the scheduled
   set remains credential-free. But the handover quarantine list is now two items, not one:
   `carry_executor` (Binance/ccxt, via `scripts/run_carry.py --live`) and
   `smart_money_alerts` (Polymarket CLOB). Both travel with the repo to MCRMINI2 dormant
   rather than disabled, and `POLYMARKET_PRIVATE_KEY` travels in `.env` alongside them.

---

*Cycle 64 — `crypto_data.db` never written. Verification was by deliberate kill against a
scratch database; production `smart_money.db` received only the new empty `collector_gaps`
table and read-only checks.*
