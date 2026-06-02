# Cycle 52 -- D5 + D6: paper-trading position lifecycle

**Predecessor:** Cycle 51 (commits `0a2476a` + `0d902d7` + `4da4ff2`).
Paper-trading executor scaffold is live; PraxisFundingExecutor task
verified clean. Cycle 51 closing notes flagged 3 risk-check stubs
(open_positions / total_open_notional / daily_loss) returning 0
trivially because no lifecycle existed yet. Cycle 52 fills that gap.

**Mode:** RECON-then-implementation, one cycle. ~3-4h Code time.
Still NO real money, NO exchange API. Cycle 51 safety belt
carries forward.

## Pre-check / Sub-item 0

Add `paper_trades` to `primary_monitored` in
`servers/praxis_mcp/tools/meta.py`. Threshold 61200 (17h, matching
funding_alerts precedent from Cycle 47); timestamp column
`signal_timestamp` (INTEGER ms). Empty-table handling relies on the
existing `_collect_db_health` empty-table path (row_count=0,
error="empty table") rather than is_stale=true.

## RECON decisions (all surfaced + approved with refinements)

### A. Hold-period source of truth -- data-resolved
- Atlas Exp 13 grid: hold_days in {3, 7, 14}; live deploy: "RF selects
  per-day"
- `funding_signals.hold_days` carries the RF-selected value per
  (asset, funding-event); funding_alerts does NOT carry it
- Executor: LEFT JOIN funding_alerts with funding_signals on
  (asset, timestamp) at decision time. Writes JOIN-resolved
  hold_days into new paper_trades.hold_days column.
- JOIN miss + no override: skip with reason `hold_days_unknown`,
  hold_days column = NULL. **No fallback default** -- a JOIN miss is
  a data-integrity anomaly that surfaces, not silently defaulted.
- `--force-hold-days N` CLI override for Cycle 53 backtests
  (bypasses JOIN, sets hold_days to N).

### B. Schema -- Option B (append-only)
- `paper_position_exits` (new table) joined to paper_trades on
  (asset, signal_timestamp). One row per closure.
- `paper_trades` remains immutable post-write (audit trail).
- ADDITION: also add `hold_days INTEGER` (nullable) to paper_trades
  during D5a. Entry decision CAPTURES hold_days; decouples from
  funding_signals lifetime.

### C. Exit timing -- Option Y (window-aligned)
- target_exit_timestamp = signal_timestamp + hold_days * 86400000 ms
- Funding events SELECT'd with: timestamp > signal_timestamp AND
  timestamp <= target_exit_timestamp (exclusive entry, inclusive exit)
- Mirrors atlas Exp 13 `run_funding_hold` semantics
  (engines/funding_rate_strategy.py:173-177) exactly.
- Retro note: for integer-day holds with window-aligned entry
  timestamps (which funding_alerts.timestamp IS by construction),
  Options X and Y produce identical arithmetic. Window-alignment is
  implicit, not enforced. Matters only if a future cycle introduces
  fractional holds or non-aligned entries.

### D. Sign convention -- direct positive sum
- Long spot + short perp (delta-neutral).
- Positive funding rate -> longs pay shorts -> we receive.
- `funding_payments_usd = sum(rate * notional)` over collected events.
- `net_pnl_usd = funding_payments - tc_entry - tc_exit`
- TC baseline: 4 bps one-way (`TC_PCT_ONE_WAY = 0.0004`), applied
  once at entry + once at exit. Round-trip 8 bps.

## Scope

- D5a: schema
  - Migration `cycle52a_paper_trades_add_hold_days.py`: ALTER TABLE
    ADD COLUMN hold_days INTEGER NULL
  - Migration `cycle52b_paper_position_exits_schema.py`: CREATE TABLE
    (15 columns; PK (asset, signal_timestamp))
  - init_db() in engines/crypto_data_collector.py updated for both
    (fresh-DB symmetry)
- D5b: real position-state queries in funding_executor.py
  - `_open_positions_for_asset()` -> COUNT(paper_trades WHERE entered
    AND NOT EXISTS in paper_position_exits)
  - `_total_open_notional_usd()` -> SUM(intended_size_usd)
  - `_daily_loss_so_far_usd()` -> abs(SUM(net_pnl_usd) where exit
    is today UTC AND net < 0)
  - `get_open_positions(asset=None)` helper
  - RiskChecks gains `hold_days_known: bool` + `hold_days_value: int |
    None`; all_ok() includes hold_days_known
- D6: exit-reconciliation loop
  - `compute_exit(pos, conn)` -> dict | None
  - `persist_exit(exit_row, conn)` -> bool (INSERT OR IGNORE)
  - run_once() now runs entry loop THEN exit loop in a single
    invocation
  - CLI: `--force-hold-days N` override

## Smoke test plan

Two-tier (refinement: Tier 2 inserts paper_trades DIRECTLY to avoid
max_signal_age_seconds=5400 conflict with backdated alerts):

- **Tier 1 (D5b)**: synthetic funding_signals + funding_alerts
  (P=0.71, alerted_at=now, hold_days in {3,7,unspecified}); 3 windows
  covering entry path, concurrent-position skip with real queries,
  and JOIN-miss skip with reason `hold_days_unknown`.

- **Tier 2 (D6)**: synthetic paper_trades row inserted directly
  (signal_ts = real 8h funding boundary 4 days back; hold_days=3);
  exit-reconcile pulls 9 real Binance funding_rates events
  (2026-05-29 -> 2026-06-01); hand-compute funding_payments_usd =
  sum(rate * 500) and net_pnl_usd; verify match within float epsilon.

After Tier 2: safety-belt grep over the 4 implementation files
(engines/funding_executor.py + 2 migrations + service bat).

## Out of scope (Cycle 53+ surface)

- Real exchange API calls (still paper-only)
- Real money allocation
- Multi-hold-period concurrent positions per asset (max 1 per Cycle 51)
- Live-trading backtest replay (Cycle 53 = D7+D8)
- Position sizing dynamics beyond fixed $500 default
- Multi-venue (Bybit still data-only)
- Atlas update (Cycle 53 alongside backtest)

## Acceptance

1. paper_trades added to primary_monitored (sub-item 0)
2. paper_position_exits table created via migration + init_db
3. paper_trades.hold_days column added via migration + init_db
4. get_open_positions() returns correct results
5. 3 stub risk checks replaced with real queries; Tier 1 confirms
   concurrent-position skip path
6. Exit-reconciliation correctly computes P&L (Tier 2 hand-verify)
7. Idempotency: 2nd run produces 0 new entries / 0 new exits
8. Safety-belt grep clean
9. Standard commit + push + SHA insertion follow-up
10. Retro captures RECON decisions

## Pause points

- After RECON (A/B/C/D + refinements): approved with
  hold_days addition to paper_trades + smoke refinement
- After D5 Tier 1 smoke: brief confirm (done)
- After D6 Tier 2 smoke: full P&L numbers + funding_rates rows
  echoed for sign-convention review (done; arithmetic matched
  to 0.00e+00 delta on all 3 components)
- If safety-belt grep returns unexpected hits: pause + explain
