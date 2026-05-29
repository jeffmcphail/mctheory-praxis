# Cycle 46 -- DB_PATH CWD-independence fix (44h, Option A funding chain)

**Predecessor:** Cycle 45 (`b274693` + `8d51126`).
Small focused cycle picking up 44h from the standing carry-forward.

**Mode:** RECON-then-implementation, one cycle. ~20-30 min.

## Background

Cycle 43 RECON surfaced a "phantom DB" trap when a Bash `cd services`
left CWD at `services/` and a subsequent `init_db()` call created a
phantom `data/crypto_data.db` at `services/data/crypto_data.db`
instead of writing the real DB at the repo root.

Cleaned up in Cycle 43; the underlying mechanism (`DB_PATH = Path("data/
crypto_data.db")` with no anchoring) was logged as a candidate fix.

## Fix shape (Option A -- funding chain only)

`Path(__file__).resolve().parent.parent / "data" / "<db>"` anchors
the path to the repo root via the source file's location, regardless
of process CWD.

## Files

| File | Constant | Type |
|---|---|---|
| `engines/crypto_data_collector.py:40` | `DB_PATH` | `Path` |
| `scripts/funding_monitor.py:63` | `DEFAULT_DB` | `str` (wrap in `str(...)`) |
| `scripts/backfill_funding_history.py:84` | `DB_PATH` | `Path` |

Note: `servers/praxis_mcp/tools/meta.py:23` initially flagged by grep
is a docstring schema example, NOT actual code. Real sidecar paths
are passed at register-time from `servers/praxis_mcp/server.py:53-58`
which are already `REPO_ROOT`-anchored. Docstring left as illustrative.

## Verification

1. Syntax check
2. Import + print from repo root cwd
3. Import + print from `services/` cwd -- must yield same absolute path
4. Call `init_db()` from `services/` cwd -- query an existing table to
   confirm the path resolves to the real DB
5. No phantom `services/data/` created
6. Trigger PraxisFundingCollector + PraxisFundingMonitor; both
   LastResult=0; funding_rates fresh for all 6 assets; no phantom DB

## Out of scope

- Bulk fix of the remaining 14 engines (option B candidate; logged
  in retro as Cycle 47+ "44h-bulk")
- Helper module refactor `engines/_paths.py` (option C candidate;
  logged as Cycle 48+ "44h-refactor")
- One-offs in `outputs/` (intentional repo-root-invocation;
  confirmed skipped)
- All other 44+ carry-forwards (cross-venue, LSTM, executor, etc.)

## Acceptance

1. DB_PATH (and DEFAULT_DB, and the backfill DB_PATH) all anchored
2. Verification 1+2: same absolute path from both CWDs
3. Verification 3: no phantom DB exists post-fix
4. Verification 4+5: scheduled tasks smoke clean
5. Standard commit + push + SHA insertion follow-up
6. Retro captures audit inventory and option B/C deferrals
