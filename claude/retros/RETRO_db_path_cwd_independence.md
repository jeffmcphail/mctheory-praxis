# Retro: Cycle 46 -- DB_PATH CWD-independence fix (44h, Option A funding chain)

**Brief:** `claude/handoffs/BRIEF_db_path_cwd_independence.md`
**Date:** 2026-05-29
**Mode:** RECON-then-implementation, one cycle, ~20 min
**Status:** DONE
**Predecessor:** Cycle 45 (`b274693` + `8d51126`)
**Commit:** `d441c92`

---

## Summary

Three constants in the funding-monitoring chain re-anchored from
CWD-relative `Path("data/...")` to repo-root-anchored
`Path(__file__).resolve().parent.parent / "data" / "..."`. Closes
the Cycle 43 phantom-DB trap surface for the funding stack
specifically; logs the other 14 vulnerable engines as carry-forward
work.

Net change:
- `engines/crypto_data_collector.py`: `DB_PATH` anchored
- `scripts/funding_monitor.py`: `DEFAULT_DB` anchored (kept as `str`
  to preserve the argparse default type for `--db`)
- `scripts/backfill_funding_history.py`: `DB_PATH` anchored
- No other files touched. No schema changes. No behavior changes
  beyond CWD-independence.

---

## RECON findings

Grep audit surfaced 24 hits across 18 operational files. Categorized:

**Already safe** (`REPO_ROOT`-anchored): ~10 files including
`engines/atlas_sync.py`, `servers/praxis_mcp/server.py` (×4 sidecar
paths), `engines/info_bars/writer.py`, `engines/intrabar_predictor.py`,
`scripts/cycle34_backfill_info_bars.py`, and all `scripts/migrations/
cycle*_*.py` files. These already use the right pattern; good
precedent.

**CWD-vulnerable, funding chain** (this cycle's scope -- 4 hits, 3 files
after collapsing the meta.py docstring false positive):
- `engines/crypto_data_collector.py:40` -- `DB_PATH`
- `scripts/funding_monitor.py:63` -- `DEFAULT_DB`
- `scripts/backfill_funding_history.py:84` -- `DB_PATH`

**CWD-vulnerable, other engines** (deferred to Cycle 47+ "44h-bulk"):
14 files, listed in "Open items" below.

**False positive** (docstring, not code): `servers/praxis_mcp/tools/
meta.py:23` -- the `Path("data/live_collector.db")` is inside the
`register()` docstring as a schema example. The actual sidecar paths
are passed in from `servers/praxis_mcp/server.py:53-58` and are
already `REPO_ROOT`-anchored. Left as-is (illustrative).

**Out-of-scope one-offs**: `outputs/verify_cycle37.py`,
`outputs/snapshot_md_hashes.py`, `outputs/exp10_revival/verify_db.py`
-- ad-hoc scripts in `outputs/` meant to be run from repo root
explicitly. Per brief's scope, left alone.

---

## Execution log

### Edits

All three use the same pattern:
```python
Path(__file__).resolve().parent.parent / "data" / "<dbname>"
```

`scripts/funding_monitor.py` wraps in `str(...)` to preserve the
argparse `--db` default's type. The constant is consumed by
`os.getenv`-style lookups and `sqlite3.connect(args.db)` which both
accept either str or Path, but keeping str avoids any subtle
downstream surprises.

Comments added inline noting the cycle and the trap origin so a
future contributor sees the rationale at the constant definition.

### Verification chain

**Syntax check:** OK on all three modified files.

**Verify 1 (cwd = repo root):**
```
crypto_data_collector.DB_PATH    : C:\...\praxis\data\crypto_data.db
funding_monitor.DEFAULT_DB       : C:\...\praxis\data\crypto_data.db
backfill_funding_history.DB_PATH : C:\...\praxis\data\crypto_data.db
```

**Verify 2 (cwd = `services/` -- the original Cycle 43 trap CWD):**
```
crypto_data_collector.DB_PATH    : C:\...\praxis\data\crypto_data.db
funding_monitor.DEFAULT_DB       : C:\...\praxis\data\crypto_data.db
backfill_funding_history.DB_PATH : C:\...\praxis\data\crypto_data.db
```

Identical absolute path across both CWDs. **Load-bearing check.**

**Verify 2b (call `init_db()` from services/):** queried existing
`funding_signals` and `funding_alerts` tables (54 rows + 0 rows --
matches state pre-fix), confirming the call reached the REAL DB and
not a phantom. Pre-fix this same call would have created
`services/data/crypto_data.db` from scratch with the schema but no
historical rows.

**Verify 3 (no phantom DB):** `services/data/` does not exist
post-test. Trap closed.

**Verify 4 (PraxisFundingCollector smoke):**
```
LastRun:    2026-05-29 15:27:57
LastResult: 0
```
funding_rates fresh for all 6 assets through 2026-05-29T16:00 UTC
(3735 rows per asset, +1 vs the pre-trigger state of 3734 for the
16:00 UTC event that just landed).

**Verify 5 (PraxisFundingMonitor smoke):**
```
LastRun:    2026-05-29 15:27:57
LastResult: 0
```
funding_signals row count unchanged at 54 (PK collision skip on the
16:00 UTC window since Cycle 45's smoke had already populated those
rows). funding_alerts still 0 (current regime is sit-out; no signal
crosses the live P>0.70 gate).

---

## Acceptance criteria

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | DB_PATH (+ DEFAULT_DB + backfill DB_PATH) anchored | ✅ |
| 2 | Same absolute path from both CWDs | ✅ Verify 1 + 2 |
| 3 | No phantom DB | ✅ Verify 3 |
| 4 | Scheduled tasks smoke clean | ✅ Verify 4 + 5 (both LastResult=0) |
| 5 | Standard commit + push + SHA insertion follow-up | ✅ standard pattern |
| 6 | Retro captures inventory + deferrals | ✅ this file |

---

## Notes

### Why funding chain only

The Cycle 43 phantom DB was specifically from
`engines/crypto_data_collector.py:init_db()` getting called with a
relative `DB_PATH` while CWD was `services/`. Fixing the
funding-chain constants closes the proven-vulnerable surface. The
other 14 engines have the same shape of bug but no Cycle in recent
memory has caught a phantom from them, because their scheduled tasks
all `cd /d %PRAXIS_DIR%` in the bat before invoking python, and
interactive Code-side work with non-funding engines is rare.

Bulk-fixing 14 engines in one cycle without smoke-testing each one
risks introducing a regression that's hard to attribute. Deferring
to a dedicated cycle (or to a helper-module refactor that touches
all of them together) is the safer pattern.

### Why not the helper module refactor (option C) now

Option C -- `engines/_paths.py` with `REPO_ROOT` + `DATA_DIR`
constants and migrate all consumers to import from there -- is the
right long-term answer. It prevents any new engine from
re-introducing the trap. But: it's an architectural change that
touches every consumer module's import block, has import-cycle
risk for engines that already import from each other, and benefits
most from a uniform starting point. Doing it AFTER option B (when
every consumer is already on the same anchored pattern) is cleaner
than doing it instead of B (would need migration AND refactor in
the same pass).

### docstring false positive at meta.py:23

The grep flagged `Path("data/live_collector.db")` at line 23 of
`servers/praxis_mcp/tools/meta.py`. Read carefully: the line is
inside the `register()` function's docstring as a schema example.
Real sidecar paths are passed in from
`servers/praxis_mcp/server.py:53-58` which are already
`REPO_ROOT`-anchored.

The docstring example is technically misleading (a future
contributor following it literally would re-introduce the trap),
but rewriting docstring examples to use `Path(__file__).resolve()
.parent.parent / ...` makes them noisy. Left as-is; a future cycle
could replace with a clearly-illustrative absolute path like
`Path("/path/to/data/live_collector.db")` if this becomes a real
trap source.

### What this cycle does NOT do

- Does NOT touch the 14 other vulnerable engines (logged below)
- Does NOT introduce `engines/_paths.py` helper module
- Does NOT change argparse `--db` flag default behavior (the
  default's TYPE is preserved; only its VALUE now resolves
  correctly regardless of CWD)
- Does NOT touch `outputs/*.py` one-off verification scripts
- Does NOT remove the TEAMS_WEBHOOK_URL fallback (still standing)

---

## Open items / Cycle 47+ inputs

- **47 (or 47a) "44h-bulk"** -- apply the same anchoring pattern
  to the remaining 14 vulnerable engine constants. Mechanical edit;
  same Path(__file__).resolve().parent.parent / "data" / "..." in
  each. Per-subsystem smoke-test (or trust that scheduled tasks
  `cd /d %PRAXIS_DIR%` in their bats and treat as covered already).
  Full list:
  - `engines/lstm_predictor.py:41` (`DATA_DB`)
  - `engines/live_collector.py:40,210,539` (`DB_PATH` + 2 inline
    `spike_scanner.db` refs)
  - `engines/smart_money.py:43` (`DB_PATH`)
  - `engines/smart_money_alerts.py:38,39` (`DB_PATH` + `ALERTS_DB_PATH`)
  - `engines/spike_scanner.py:35` (`DB_PATH`)
  - `engines/spike_features.py:34` (`DB_PATH`)
  - `engines/event_classifier.py:40` (`DB_PATH`)
  - `engines/mev_executor.py:79` (`LIVE_DB`)
  - `engines/actuarial.py:45` (`DB_PATH`)
  - `engines/ai_ensemble.py:47` (`DB_PATH`)
  - `engines/crypto_predictor.py:48` (`DB_PATH`)
  - `engines/flash_scanner.py:42` (`DB_PATH`)
  - `engines/mev_scanner.py:41` (`DB_PATH`)
  - `engines/negrisk_arb.py:43` (`DB_PATH`)
- **48 (or 48a) "44h-refactor"** -- factor into `engines/_paths.py`
  helper module with `REPO_ROOT`, `DATA_DIR`, and standard
  per-subsystem DB path constants. All consumers import from there.
  Done after 47-bulk lands so the helper migration runs against a
  uniform codebase.
- **Plus the standing carry-forwards from Cycle 45's retro:** 46a
  TEAMS_WEBHOOK_URL fallback removal (after user's .env migrated),
  46b cross-venue, 46c LSTM v2, 46d executor, 46e regime accumulation,
  46f PMA backfill, 46g atlas_search filter, 46h threshold tightening,
  46j FAIL_COUNT unhappy-path test, 46k post_teams_alert rename.
  (Renumbering note: 46a-46k from Cycle 45's retro become 47a-47k or
  later in practice; the alphabet-cycling-on-cycle-number naming
  gets confusing once more than one cycle has the same number suffix.
  Mental model: these are "queue items" not "Cycle 46 items".)
