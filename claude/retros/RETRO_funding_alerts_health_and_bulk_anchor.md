# Retro: Cycle 47 -- funding_alerts health monitoring + 44h-bulk anchoring sweep

**Brief:** `claude/handoffs/BRIEF_funding_alerts_health_and_bulk_anchor.md`
**Date:** 2026-05-30
**Mode:** RECON-then-implementation, one cycle, ~35 min including verification
**Status:** DONE
**Predecessor:** Cycle 46 (`d441c92` + `d38def0`)
**Commit:** `06153c6`

---

## Summary

Two follow-ons. **Sub-item 1:** added `funding_alerts` to the health-check
monitored set with 17h threshold matching `funding_signals`, plus
explanatory comment block noting the sparse-population semantic and
empty-table safety. **Sub-item 2:** anchored 17 `DB_PATH`-style
constants across 14 engine files via the Cycle 46
`Path(__file__).resolve().parent.parent` pattern. Refactored
`live_collector.py`'s 2 inline `spike_scanner.db` references to use
a new module-level `SPIKE_DB_PATH` constant rather than re-typing
the Path construction.

Net change:
- `servers/praxis_mcp/tools/meta.py`: comment block + dict entry
- 14 engine files, 17 constants total, all on the same anchoring
  pattern
- `mev_executor.py:80 EXECUTOR_DB` picked up as a same-file
  same-trap inline addition (1 extra constant beyond the brief's
  17-target list)

No schema changes. No behavior changes beyond CWD-independence and
the new monitored-table entry.

---

## RECON findings

The 14 files + 17 constants were already enumerated in Cycle 46's
retro § "Open items" (the "44h-bulk" list). No new surprises during
RECON; the audit map was correct.

**One additional in-scope constant surfaced during the per-file read
pass:** `engines/mev_executor.py:80` has `EXECUTOR_DB = Path("data/
mev_executor.db")` adjacent to `LIVE_DB` (line 79). Same trap, same
file, mechanical inclusion. Fixed inline; mentioned in commit.

**Three out-of-scope same-pattern constants surfaced and were
deliberately left alone:**
- `engines/lstm_predictor.py:42` -- `MODEL_DIR = Path("models/lstm")`
- `engines/crypto_predictor.py:49` -- `MODEL_DIR = Path("models/crypto")`
- `engines/spike_features.py:35` -- `OUTPUT_DIR = Path("data/training")`

All have the same CWD-vulnerability shape but are non-DB paths
(models/, training output dir). Logged in "Open items" below as a
future audit candidate.

---

## Execution log

### Sub-item 1: funding_alerts in primary_monitored

Two surgical edits to `servers/praxis_mcp/tools/meta.py`:
- Insert a `funding_alerts` comment block in the existing per-table
  documentation, between the funding_signals block and the fear_greed
  block. Explains: populates sparsely (only on above_gate=1
  firings); matches funding_signals' 17h threshold; empty-table
  surfaces row_count=0+error not is_stale=True.
- Insert dict entry `"funding_alerts": 61200` in `primary_monitored`
  immediately after `funding_signals`.

### Sub-item 2: 14 engines anchored

Pattern (mechanical, identical to Cycle 46):

    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "<dbname>"

Each edit added a brief inline comment "Cycle 47 (44h-bulk): anchor
to repo root via __file__ for CWD-independence (see Cycle 46)". For
files with multiple adjacent constants (live_collector,
smart_money_alerts, mev_executor) a single comment block above all
constants in that file.

**`live_collector.py` -- 3 changes:**
1. Line 40: `DB_PATH = Path("data/live_collector.db")` -> anchored
2. Added new line 41: `SPIKE_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "spike_scanner.db"` -- new module-level constant since this engine reads from spike_scanner's DB (cross-engine reference)
3. Line 210: `sqlite3.connect("data/spike_scanner.db")` refactored to `sqlite3.connect(str(SPIKE_DB_PATH))`
4. Line 539: `spike_db_path = Path("data/spike_scanner.db")` refactored to `spike_db_path = SPIKE_DB_PATH`

The refactor at 210/539 to use the module-level constant rather than
re-typing the Path construction follows the brief's "If they currently
re-type, refactor them to use the constant" guidance.

### Verification

**Verify A (syntax + path resolution from repo root):** Throwaway
probe `outputs/_cycle47_verify.py` (since cleaned up) imported all
14 modules and printed each anchored constant. All 17 constants
resolved to `C:\Data\Development\Python\McTheoryApps\praxis\data\
<dbname>.db`. Probe needed `sys.path.insert(0, REPO)` to make the
engines package importable from arbitrary CWD.

**Verify B (same paths from services/ cwd):** Re-ran the same probe
from `cwd = services/`. Identical absolute paths produced. 17/17 OK.

**Verify C-1 (PraxisSmartMoney smoke):**
```
Triggered: Start-ScheduledTask -TaskName PraxisSmartMoney
LastRun:   2026-05-30 11:08:28 local
LastResult: 0
Runtime:   ~10 min (typical based on Cycle 42 baseline)
```
`smart_money.db.position_snapshots` got a new snapshot row with
`snapshot_id=20260530_150835` and `datetime=2026-05-30T15:08:35`,
confirming the new anchored `smart_money.DB_PATH` reached the
correct DB end-to-end.

**Verify C-2 (PraxisLiveCollector status):**
```
State:      Ready (NOT Running)
LastRun:    2026-05-12 21:32:12 local
LastResult: 267014 (SCHED_S_TASK_TERMINATED)
```
**Subtle finding:** the scheduled task itself terminated on 2026-05-12,
but a long-lived `engines.live_collector` Python process is still
running -- `price_snapshots` continues to grow with 28s freshness
(latest write 2026-05-30T15:08:06 UTC). That process loaded the
OLD relative `Path("data/live_collector.db")` when it started on
05-12; my Cycle 47 anchor edit won't take effect until the process
is restarted. No breakage in the interim because the old relative
path resolves correctly given the bat sets
`cd /d "%PRAXIS_DIR%"` before launch, so CWD = repo root for the
duration of that process.

Net effect: same target DB either way; the anchor change is
forward-protective (next restart picks it up).

**Verify D (dormant engines):** 12 engines have no scheduled task
in the current Praxis* lineup and are not live-tested in this cycle:
- `lstm_predictor` -- ML inference; manually invoked
- `smart_money_alerts` -- manual CLI (`engines.smart_money_alerts diff/convergence/trade/...`)
- `spike_scanner` -- manual `collect`/`detect` subcommands
- `spike_features` -- manual ML feature extraction
- `event_classifier` -- manual classification
- `mev_executor`, `mev_scanner` -- MEV / Polymarket; manually invoked
- `actuarial` -- manual analysis
- `ai_ensemble` -- LLM cascade; manual invocation
- `crypto_predictor` -- manual prediction
- `flash_scanner` -- manual
- `negrisk_arb` -- manual arbitrage detection

All Path-anchored per Verify A + B. Live verification deferred to
"subsequent natural use will surface any regression."

**Verify 8 (funding_alerts monitored):**
```
funding_alerts in 'tables' bucket:  YES
funding_alerts in 'unmonitored':    NO
funding_alerts entry:               {'row_count': 0, 'error': 'empty table'}
primary unmonitored after Cycle 47: []
```
Empty-table handled correctly (`error="empty table"` not
`is_stale=True`). Primary DB's unmonitored list is now empty -- every
table has either an explicit threshold or is internal (sqlite_*).

---

## Acceptance criteria

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | funding_alerts in `meta.py:primary_monitored` with 17h threshold + comment block | ✅ |
| 2 | 14 engines + 2 live_collector inline refs anchored via `Path(__file__).resolve()` | ✅ + mev_executor EXECUTOR_DB picked up inline as same-file bonus |
| 3 | Verify A: 14 files import cleanly + paths absolute | ✅ 17/17 OK |
| 4 | Verify B: same paths from non-root cwd | ✅ identical |
| 5 | Verify C: smart_money + live_collector smoke clean | ✅ smart_money LastResult=0 with fresh row; live_collector still writing |
| 6 | Verify D: dormant engines documented | ✅ 12 engines listed above |
| 7 | Standard commit + push + SHA insertion follow-up | ✅ standard pattern |
| 8 | funding_alerts no longer in `unmonitored` | ✅ primary unmonitored list now empty |

---

## Notes

### live_collector long-lived process subtlety

Worth documenting because it explains why "scheduled task State=Ready
but LastRun is 2 weeks old" doesn't mean broken. The bat
`live_collector_service.bat` launches a long-running Python process
that loops indefinitely (or until killed). Once the bat exits (the
launcher exits after spawning the long-lived process), the scheduled
task is in `Ready` state -- not `Running`, because Task Scheduler
only counts the bat invocation itself, not its descendants.

Implication for verification: trigger-and-check doesn't apply to
long-lived collectors the same way it does to one-shot tasks (like
the funding collector). Instead, freshness-of-output is the right
signal. `price_snapshots` 28s fresh confirms the long-lived process
is healthy.

Implication for the Cycle 47 anchor edit: the in-process module
state on `engines.live_collector` is whatever was loaded when the
long-lived process started on 2026-05-12. My anchor edit lands in
the file but the running process is unaffected. Next restart of
the long-lived process (manual or via reboot) will pick up the new
anchor. No urgency to restart; the old relative path resolves
correctly while CWD remains at the project root.

If user wants to validate the anchor takes effect in the running
process specifically, they can manually:
1. Find the long-lived python.exe PID
2. Kill it
3. Re-trigger the scheduled task to restart
But that's intrusive for no operational benefit; logging here for
completeness.

### Out-of-scope same-pattern constants (deferred)

Three constants surfaced during RECON that have the same
CWD-vulnerability shape but aren't DB paths:

| File:Line | Constant | Surface |
|---|---|---|
| `lstm_predictor.py:42` | `MODEL_DIR` | models/lstm checkpoint dir |
| `crypto_predictor.py:49` | `MODEL_DIR` | models/crypto checkpoint dir |
| `spike_features.py:35` | `OUTPUT_DIR` | data/training output dir |

Fixing these is mechanically identical (same pattern) but they're
not the trap surface Cycle 43 caught (which was specifically the
init_db() phantom-DB). Logged as "44q non-DB path audit" candidate
below.

### What this cycle does NOT do

- Does NOT introduce the `engines/_paths.py` helper module (still
  queued for Cycle 48 "44h-refactor")
- Does NOT restart the long-lived `engines.live_collector` process
  (user discretion; not needed)
- Does NOT touch models/ paths or non-data/ Path constants
- Does NOT remove the `TEAMS_WEBHOOK_URL` fallback (still standing)
- Does NOT touch sidecar-table monitoring (smart_money's
  position_changes, convergence_signals; live_collector's
  spike_alerts, tracked_markets -- these tables aren't in
  `primary_monitored` or any sidecar-monitored spec yet)

---

## Open items / Cycle 48+ inputs

- **48 "44h-refactor"** -- factor into `engines/_paths.py` with
  `REPO_ROOT` + `DATA_DIR` (and possibly `MODELS_DIR`,
  `OUTPUT_DIR_ROOT`) constants. All consumers import from there.
  Done after Cycle 47-bulk's uniform anchoring landed, so this is
  a migration to centralization rather than a fix.
- **49 (or later) "44q non-DB path audit"** -- 3 known
  `Path("models/...")` or `Path("data/training")` constants in the
  codebase outside this cycle's scope. Same mechanical fix; tiny
  cycle.
- **50 (or later) "44q sidecar-table monitoring audit"** -- tables
  in sidecar DBs (live_collector.db, smart_money.db,
  smart_money_alerts.db) that aren't in any `monitored` spec yet:
  collection_log, spike_alerts, tracked_markets, convergence_signals,
  position_changes, tracked_wallets. Add health thresholds for the
  ones that should be monitored (e.g. spike_alerts ~24h; tracked_*
  pure-state, not freshness-monitored).
- **PraxisLiveCollector long-lived process restart** (optional;
  user-driven; would let Cycle 47 anchor take effect in-process,
  but no operational urgency)
- Plus the standing carry-forward queue: 46a TEAMS_WEBHOOK_URL
  fallback removal (after user's .env migrated), 46b cross-venue,
  46c LSTM v2, 46d executor, 46e regime accumulation, 46f PMA
  backfill, 46g atlas_search filter, 46h threshold tightening,
  46j FAIL_COUNT unhappy-path test, 46k post_teams_alert rename.
