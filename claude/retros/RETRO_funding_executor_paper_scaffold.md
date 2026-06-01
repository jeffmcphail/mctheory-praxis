# Retro: Cycle 51 -- 44c: Executor integration RECON + paper-trading scaffolding

**Brief:** `claude/handoffs/BRIEF_funding_executor_paper_scaffold.md`
**Date:** 2026-06-01
**Mode:** RECON-then-implementation, one cycle. Took ~2h Code time + smoke verification.
**Status:** DONE -- paper-trading executor scaffold committed; PraxisFundingExecutor task registration pending user-side admin step.
**Predecessor:** Cycle 50 (`d0d2aa8` + `fb9c9b6`) -- cross-venue spread DISCONFIRMED; redirect target was 44c.
**Commit:** `0a2476a`

---

## Summary

First step on the real-money path -- intentionally conservative. Engine 7
paper-trading executor scaffolded with a 9-control risk layer (no real
money, no exchange API). Reads funding_alerts (currently empty post-Cycle-45
flush); applies risk checks; logs decision (enter | skip + risk_checks_json
audit trail) to new paper_trades table. Idempotent per (asset, funding-window)
via PK + INSERT OR IGNORE + NOT EXISTS subquery.

D3 smoke test PASSED: synthetic BTC alert (P=0.71, fresh alerted_at) →
ENTER decision; size $500 long_spot_short_perp; all 9 risk checks ok;
2nd run found 0 pending (NOT EXISTS filter worked); cleanup left
funding_alerts + paper_trades both at 0 rows.

Safety-belt grep cleaned up post-docstring-rewrite (false-positive hits in
documentation describing absent patterns; rewrote to describe the invariant
without containing the forbidden tokens literally).

PraxisFundingExecutor task registration is a 1-line user action (admin
required, same as Cycle 41 + Cycle 47); script committed and waiting.

Net change:
- `engines/funding_executor.py` (new, ~330 lines): FundingExecutor class +
  RiskChecks dataclass + CLI; CWD-anchored DB path per Cycle 46
- `scripts/migrations/cycle51_paper_trades_schema.py` (new): CREATE TABLE
  paper_trades (idempotent)
- `engines/crypto_data_collector.py`: paper_trades CREATE TABLE IF NOT EXISTS
  added to init_db() for fresh-DB symmetry
- `services/funding_executor_service.bat` (new): single python invocation;
  exit code propagates naturally (no FAIL_COUNT needed)
- `services/register_funding_executor_task.ps1` (new): mirrors
  register_funding_monitor_task.ps1; 3x daily triggers 00:20/08:20/16:20
- `.env.example`: EXECUTOR_KILL_SWITCH stub + setup documentation
- `data/crypto_data.db`: paper_trades table created (empty)

---

## Execution log

### RECON pause-point batch (user approved with 3 refinements)

1. **9-control risk thresholds** -- approved as-proposed, plus a new 9th:
   `min_p_above_gate` default 0.0 (no-op in Cycle 51; tunable in Cycle 53+
   backtests via `--min-p-above-gate <float>` CLI override).

2. **paper_trades schema** -- approved with one added column:
   `funding_alert_alerted_at TEXT NOT NULL`. Captures the raw
   `funding_alerts.alerted_at` value at decision time, so future analysis
   can recompute different age windows without depending on funding_alerts
   row still existing.

3. **Task cadence 00:20/08:20/16:20 LOCAL** -- approved. Retro note: batch
   processing per executor run is normal (NOT EXISTS subquery returns ALL
   pending alerts; multiple assets firing in the same monitor cycle get
   processed together in one executor run).

4. **Safety-belt grep added** as explicit pre-commit step.

### D1 + D2 implementation

Wrote `engines/funding_executor.py` with the prescribed module shape:
RiskChecks dataclass capturing all 9 check outcomes; FundingExecutor class
with load_pending_alerts (NOT EXISTS subquery against paper_trades),
apply_risk_checks (with Cycle 51 stubs returning 0 for open_positions /
total_notional / daily_loss -- documented as Cycle 52+ refinement targets),
decide (constructs paper_trades row dict with skip_reason built from first
failing checks), persist (INSERT OR IGNORE), run_once (main loop with summary
return), main() CLI.

Migration `scripts/migrations/cycle51_paper_trades_schema.py`: CREATE TABLE
(not recreate-table since paper_trades is brand-new). 13 columns; PK
(asset, signal_timestamp); idempotent state detection. Ran once successfully
during D1; init_db() also updated for fresh-DB symmetry.

### D3 smoke test (synthetic alert + executor + cleanup)

Built throwaway smoke probe at `outputs/_cycle51_smoke.py` (deleted post-run).
Used future-dated synthetic alert (asset=BTC, window=2027-01-01 00:00 UTC) to
avoid collision with any real future monitor write. P=0.71, alerted_at=now,
gate=0.70.

Step 1: INSERT INTO funding_alerts → synthetic row created (monitor_version=`cycle51-smoke` for traceability).

Step 2: 1st executor invocation:
```
pending funding_alerts (not yet in paper_trades): 1
BTC    window=2027-01-01T00:00 P=0.7100 > gate 0.70  -> ENTER  (size $500 long_spot_short_perp)
Summary: processed=1  entered=1  skipped=0  duplicates=0
```

Step 3: Verified paper_trades row:
- decision=`enter`
- intended_size_usd=500.0
- intended_direction=`long_spot_short_perp`
- p_profitable=0.71, gate_threshold=0.70
- executor_version=`cycle51-paper-scaffold`
- 14 keys in risk_checks_json (13 from RiskChecks dataclass + `min_p_above_gate` value tracking)
- signal_age_seconds=0.186 (super fresh; well under 5400)
- All 9 risk checks → ok=True

Step 4: 2nd executor invocation:
```
pending funding_alerts (not yet in paper_trades): 0
Summary: processed=0
```
Idempotency verified: NOT EXISTS filter correctly identified the synthetic alert as already processed.

Step 5: Cleanup. DELETE both synthetic rows; both tables back to 0.

### Safety-belt grep -- one round-trip pause

First grep returned 3 false-positive hits in the executor module's
docstring (which described "NO `import ccxt`", "NO `requests.post`" etc as
the absence-of-pattern declarations). The grep can't distinguish docstring
from code.

Surfaced explicitly per brief's "pause and explain" instruction. Rewrote
the docstring to describe the same invariant without containing the
literal forbidden tokens ("No CCXT import" instead of "NO `import ccxt`",
etc). Re-ran grep: zero hits.

Trade-off recorded: the docstring is slightly more abstract but still
clear. The safety belt still documents what the executor doesn't do; just
not via keyword-grep-matchable form.

### D4 -- task registration

`services/funding_executor_service.bat` + `services/register_funding_executor_task.ps1` written. Single python invocation in the bat (no FAIL_COUNT loop needed; exit code propagates naturally).

Triggers: `-Daily -At 00:20`, `08:20`, `16:20` LOCAL. ExecutionTimeLimit 5min, MultipleInstances IgnoreNew, S4U / Limited principal -- same shape as register_funding_monitor_task.ps1 from Cycle 41.

Task registration itself requires admin elevation (Register-ScheduledTask access denied from unprivileged session, same as Cycle 41 + Cycle 47 patterns). After commit, user runs `.\services\register_funding_executor_task.ps1` in elevated PowerShell; verification via Start-ScheduledTask + log check follows.

---

## Architectural decisions captured

### Single-venue commit (Binance only)

Cycle 50's structural finding (ADA didn't trade in cross-venue despite
being Exp 13's top-Sharpe asset) means the cross-venue spread strategy is
disconfirmed but Exp 13's single-venue carry is unaffected. The executor
commits to Binance execution; Bybit data is for backup/audit, not alpha.

### Paper-mode-only for Cycle 51

Brief explicit: "Cycle 51 is the first step on that path -- scaffolding
only, no real money yet." Three Cycle 52+ work items prevent paper from
becoming real prematurely:
- D5+D6 (Cycle 52): position lifecycle (hold + exit; paper trades become
  two-sided entries)
- D7+D8 (Cycle 53): backtest the paper executor against Exp 13 OOS
  data (would the executor have replicated Sharpe +4.65?)
- Cycle 54+: real money rollout, gated on each prior stage performing
  as expected

### Cycle 51 risk-check stubs (open_positions / total_notional / daily_loss)

The 9-control framework includes 3 checks (concurrent_positions_ok,
total_notional_ok, daily_loss_ok) that require position-lifecycle
tracking. Cycle 51 = entry-only paper logging; no holds; no exits; no
P&L. So these 3 stub-methods return 0 / pass trivially in Cycle 51.
**This is documented at the module level + in each stub method's
docstring** so future auditors of paper_trades.risk_checks_json don't
assume the checks were meaningfully evaluating.

Cycle 52 will replace the stubs with real queries against paper_trades
(open position = entry without paired exit; daily P&L = sum of closed
positions' realized return that day).

### batch processing per executor run

User refinement #3 memorialized: if multiple assets fire in the same
funding window, the executor's NOT EXISTS subquery returns all unprocessed
alerts in one shot, and the run_once() loop processes them all in
sequence within a single invocation. Each gets its own paper_trades
decision row; commits happen at end of loop. Idempotent re-run safe via PK.

### Safety-belt grep false-positive on docstring

Documentation patterns containing the forbidden tokens (as descriptive
backticked names) tripped the grep. Three options surfaced; chose docstring
rewrite (keep grep simple, change docs). Lesson: future "describes
absence" docstrings should phrase the absent thing abstractly rather
than naming the literal pattern.

### EXECUTOR_KILL_SWITCH semantics

Env-var-driven; loaded via dotenv (memory #4 pattern). Set "1"/"true"/"yes"
to disable entries WITHOUT touching code or unregistering the task.
When on, the executor still RUNS each scheduled trigger (writes a row to
paper_trades) but every decision is `skip` with reason
`EXECUTOR_KILL_SWITCH=on`. This preserves the executor's heartbeat and
log trail even during emergency stop. Cycle 54+ when real money lands,
this will gate the exchange API call path identically.

---

## Acceptance criteria

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | funding_executor.py + paper_trades schema migration land | ✅ |
| 2 | PraxisFundingExecutor task registered + verified | ⏳ awaiting admin step from user (same workflow as Cycle 41) |
| 3 | Smoke test: synthetic alert -> paper_trades entry with correct decisions | ✅ ENTER decision, all 9 risk checks ok, idempotency confirmed, cleanup clean |
| 4 | NO exchange API calls -- safety-belt grep zero hits | ✅ post-docstring-rewrite |
| 5 | Standard commit + push + SHA insertion follow-up | ✅ standard pattern |
| 6 | Retro captures architectural decisions | ✅ this file |

---

## Open items / Cycle 52+ inputs

- **Cycle 52 (D5+D6): position lifecycle.** Convert paper_trades entries
  from entry-only logs into two-sided records. Add `exit_decided_at`,
  `exit_reason`, `realized_pnl_usd`, `hold_days_actual` columns (or a
  paired `paper_positions` table with FK to entries). Update the 3 stub
  risk checks (concurrent/total/daily-loss) to query real position state.
- **Cycle 53 (D7+D8): backtest replay.** Re-run the executor logic
  against Exp 13 OOS historical funding_signals + (synthetic)
  funding_alerts to verify the executor would have replicated atlas's
  +4.65 Sharpe. Required confidence-build step before real money.
- **Cycle 54+: real money rollout.** Small notional first (~$100/asset);
  scaling gated on each stage performing as expected. Will need:
  - Exchange API integration (CCXT or direct REST)
  - Order placement state machine
  - Position reconciliation (exchange-side vs our paper_trades)
  - Live P&L tracking
  - Real kill-switch enforcement (vs paper-mode "skip with reason")
- **D4 user-action item (now):** run `.\services\register_funding_executor_task.ps1`
  in elevated PowerShell, then I verify via Start-ScheduledTask + log
  check (same workflow as Cycle 41 + 47).
- Plus standing queue: 44d bear-regime accumulation (passive; ~14 more
  days), 44b LSTM v2 (low-likelihood), 44h-refactor (engines/_paths.py),
  44q sidecar audit + non-DB path audit, per-venue funding_rates health,
  TEAMS_WEBHOOK_URL fallback removal (46a).
