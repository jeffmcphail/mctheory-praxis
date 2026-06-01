# Cycle 51 -- 44c: Executor integration RECON + paper-trading scaffolding

**Predecessor:** Cycle 50 (commits `d0d2aa8` + `fb9c9b6`). Cross-venue
funding spread was DISCONFIRMED; redirect target 44c (executor
integration) is this cycle's scope.

**Mode:** RECON-then-implementation in a single cycle. ~2-3h Code time.
First step on the real-money path; intentionally conservative scope:
paper-trading executor only, no exchange API calls.

## Architectural decision: single-venue Binance commit

Per Cycle 50's structural finding (ADA didn't trade in the cross-venue
spread despite being top-Sharpe single-venue + largest spread tails),
the executor commits to single-venue execution at Binance for the
verified Exp 13 carry. Bybit remains backup data only.

## Scope (Cycle 51 only -- entry-side paper logging)

- D1: 9-control risk layer
  1. max_notional_per_asset_usd        = 500.0
  2. max_total_notional_usd           = 2500.0
  3. max_daily_loss_usd                = 50.0
  4. max_daily_loss_pct                = 0.02
  5. max_concurrent_positions_per_asset = 1
  6. max_signal_age_seconds          = 5400  (90 min)
  7. min_p_above_gate                 = 0.0   (Cycle 51 refinement;
                                                no-op default; tunable
                                                for Cycle 53+ backtest)
  8. asset_blacklist                  = []
  9. EXECUTOR_KILL_SWITCH env var     = off
- D2: engines/funding_executor.py
  - FundingExecutor class with load_pending_alerts / apply_risk_checks
    / decide / persist / run_once
  - NOT EXISTS subquery on paper_trades for idempotent re-runs
  - INSERT OR IGNORE for belt-and-suspenders
- D3: synthetic-alert smoke test (P=0.71, fresh alerted_at; expect
  ENTER decision; cleanup post-test)
- D4: PraxisFundingExecutor scheduled task @ 00:20/08:20/16:20 LOCAL,
  5 min after the monitor's :15 schedule

## Out of scope

- Real order placement (any exchange API)
- Position hold/exit lifecycle (Cycle 52)
- Backtest replay (Cycle 53)
- Real money rollout (Cycle 54+)
- Atlas update (no positive findings yet)

## Acceptance

1. engines/funding_executor.py + paper_trades schema migration land
2. PraxisFundingExecutor task registered + verified clean run
3. Smoke test: synthetic alert -> paper_trades entry with correct
   risk-control decisions
4. NO exchange API calls anywhere in the diff (safety-belt grep =
   zero hits)
5. Standard commit + push + SHA insertion follow-up
6. Retro captures architectural decisions

## Pause points (per brief)

- Mid-D1: surface 8 (later 9 per user refinement) risk-control
  values for approval before coding
- End-D2 before D3: confirm paper_trades schema (incl. user's
  `funding_alert_alerted_at` refinement)
- Pre-D4: confirm task cadence

All three pause-points batched into one Cycle 51 RECON pause; user
approved with 3 small refinements (min_p_above_gate knob,
funding_alert_alerted_at column, safety-belt grep before commit).

## Safety belt

- Pre-commit grep over the 3 implementation files for: `import ccxt`,
  `requests\.(post|put|delete)`, `urllib\.request`, `http\.client`,
  `aiohttp`. Expected: zero hits. If any hits, pause + explain.
- If accidental exchange API touch is detected at any point during
  implementation, cycle pauses immediately.
