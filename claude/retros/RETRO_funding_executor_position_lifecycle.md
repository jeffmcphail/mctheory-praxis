# Retro: Cycle 52 -- D5 + D6: paper-trading position lifecycle

**Brief:** `claude/handoffs/BRIEF_funding_executor_position_lifecycle.md`
**Date:** 2026-06-02
**Mode:** RECON-then-implementation, one cycle. ~3h Code time.
**Status:** DONE -- paper-trading position lifecycle live; entry +
exit-reconcile end-to-end; safety-belt grep clean; arithmetic
hand-verified to 0.00e+00 delta on real Binance funding data.
**Predecessor:** Cycle 51 (`0a2476a` + `0d902d7` + `4da4ff2`).
**Commit:** `67157a5`

---

## Summary

Cycle 51's executor logged entry DECISIONS but did not track positions
through to closure. Cycle 52 closes the loop:

- `paper_trades.hold_days` column captures the RF-selected hold value
  at decision time (JOIN-resolved from funding_signals, or via
  `--force-hold-days N` CLI override).
- `paper_position_exits` (new table) carries one row per closure with
  the full funding-payment + TC + net-P&L breakdown.
- A position is OPEN if it's in paper_trades with decision='enter'
  AND not yet in paper_position_exits.
- 3 of the Cycle 51 risk-check stubs (open_positions / total_notional /
  daily_loss) replaced with real queries against the open-position
  set.
- 1 new check (`hold_days_known`, data-availability) added as 10th
  framework entry; JOIN miss with no override forces skip with reason
  `hold_days_unknown`. **No fallback default** -- the natural state
  is funding_alerts and funding_signals are co-populated by the same
  monitor invocation; a JOIN miss should surface, not be silently
  defaulted.
- Exit-reconcile loop runs at every executor invocation
  (00:20/08:20/16:20 LOCAL), processing entries first then sweeping
  open positions for hold-window expiry.

Sub-item 0 (pre-check): `paper_trades` added to `primary_monitored`
in `servers/praxis_mcp/tools/meta.py` so `get_collector_health`
covers it. Empty-table handling matches funding_alerts (returns
row_count=0, error="empty table" rather than is_stale=true).

Both smoke tests passed end-to-end:
- **Tier 1** (D5b entry path + JOIN-miss): 3 synthetic alerts
  exercised entry, concurrent-position skip with real query, and
  hold_days_unknown skip. All 4 sub-steps green.
- **Tier 2** (D6 exit reconcile with real funding data): synthetic
  paper_trades entry backdated 4 days; executor's exit-reconcile
  pulled 9 real Binance BTC funding events over 72h; hand-computed
  P&L matched executor's row to 0.00e+00 delta on all 3 components
  (funding_payments_usd, tc_entry+exit, net_pnl_usd).

Net change:
- `engines/funding_executor.py` (~580 lines; +250 lines vs Cycle 51):
  RiskChecks gains 2 fields; new compute_exit / persist_exit /
  get_open_positions methods; CLI `--force-hold-days N`; bumped
  EXECUTOR_VERSION to `cycle52-lifecycle`.
- `scripts/migrations/cycle52a_paper_trades_add_hold_days.py` (new):
  idempotent ALTER TABLE ADD COLUMN hold_days INTEGER NULL.
- `scripts/migrations/cycle52b_paper_position_exits_schema.py` (new):
  CREATE TABLE paper_position_exits (15 cols, compound PK).
- `engines/crypto_data_collector.py`: init_db() updated with new
  hold_days column + paper_position_exits CREATE block.
- `servers/praxis_mcp/tools/meta.py`: paper_trades added to
  primary_monitored.
- `data/crypto_data.db`: paper_trades.hold_days column added
  (0 rows pre/post); paper_position_exits table created (0 rows).

---

## Execution log

### RECON pause-point (4 design questions + 2 refinements)

A/B/C/D all approved with the two refinements documented in the brief:

1. Schema addition: `hold_days INTEGER` (nullable) on paper_trades.
   Rationale captured: entry decision CAPTURES hold_days; immutable-
   at-write matches other paper_trades columns; decouples paper_trades
   audit trail from funding_signals lifetime; paper_position_exits
   .hold_days copies from paper_trades.hold_days at exit time, no
   re-JOIN.

2. Smoke test refinement: Tier 1 = entry path with alerted_at NOW
   (within max_signal_age=5400); Tier 2 = synthetic paper_trades
   row inserted directly so backdating doesn't conflict with the
   alert-age check.

### Sub-item 0: paper_trades monitoring add

Single 7-line edit to `meta.py` `primary_monitored` dict. Verified
via direct call to `_collect_db_health`:
- paper_trades moved from `unmonitored` to monitored
- Reports `{"row_count": 0, "error": "empty table"}` (the desired
  empty-table path)
- `unmonitored` list is now empty across the primary DB

### D5a: schema migrations

Both migrations ran clean:
- `cycle52a`: paper_trades col count 13 -> 14; hold_days appears
  last (notnull=0 pk=0)
- `cycle52b`: paper_position_exits created with 15 columns + compound
  PK (asset, signal_timestamp)

init_db() updated for fresh-DB symmetry; both blocks have
explanatory comment headers citing Cycle 52a / Cycle 52b respectively.

### D5b: executor real-state queries + 10th check

`RiskChecks` dataclass gained `hold_days_known: bool` + `hold_days_value:
int | None`. `all_ok()` now includes hold_days_known. The 9-control
naming preserved by treating this 10th item as a "data-availability
check" rather than a 10th risk control. Documented at module level.

Three stubs replaced with real SQL:
- `_open_positions_for_asset`: COUNT(paper_trades.decision='enter'
  AND NOT EXISTS paper_position_exits) WHERE asset=?
- `_total_open_notional_usd`: SUM(intended_size_usd) over same predicate
- `_daily_loss_so_far_usd`: abs(SUM(net_pnl_usd)) over
  paper_position_exits where substr(exit_decided_at, 1, 10) = today
  AND net_pnl_usd < 0

`load_pending_alerts` query rewritten to LEFT JOIN funding_signals
on (asset, timestamp). `force_hold_days` constructor argument overrides
the JOIN result when set; otherwise NULL surfaces as
`hold_days_known=False`.

### D5b Tier 1 smoke -- PASSED

4 steps:
1. Synthetic BTC alert A (window 2027-01-01 00:00, hold_days=3) +
   matching funding_signals row -> ENTER, hold_days=3, size $500,
   long_spot_short_perp. `get_open_positions()` returned correct counts
   (all=1, BTC=1, ETH=0).
2. Alert B (window 08:00, hold_days=7) with A still open -> SKIP
   with `per_asset_notional_exhausted (open=1); concurrent_position_cap_per_asset
   (open=1)`. JOIN-resolved hold_days=7 preserved on the skip row.
3. Alert C (window 16:00) with NO funding_signals row -> SKIP with
   reasons including `hold_days_unknown`; hold_days column = NULL.
4. Idempotency re-run: 0 pending alerts, position A still holding
   (target_exit = 2027-01-04; today = 2026-06-02; ~215d remaining).

Observation captured: the double-mention of `per_asset_notional_exhausted`
+ `concurrent_position_cap_per_asset` in skip_reason is correct (both
checks read the same open count; both fail simultaneously at max=1).
Not a bug; the 10-check framework reports all failures rather than
short-circuiting on the first.

### D6 implementation

`compute_exit(pos, conn) -> dict | None`:
- Computes `target_exit_ts = signal_ts + hold_days * 86400000` (ms)
- Returns None if wall-clock < target (still holding)
- Else SELECT funding_rates WHERE asset=? AND venue='binance' AND
  timestamp > signal_ts AND timestamp <= target (exclusive entry,
  inclusive exit; matches atlas L173-177)
- `funding_payments_usd = sum(rate * notional)`
- `tc_entry = tc_exit = TC_PCT_ONE_WAY * notional` (0.0004 * 500 = $0.20)
- `net_pnl_usd = funding_payments - 2 * tc_each`

Constants pulled to module level for blame-clarity:
- `TC_PCT_ONE_WAY = 0.0004`
- `MS_PER_DAY = 86_400_000`
- `EXECUTION_VENUE = "binance"`

`persist_exit` uses INSERT OR IGNORE on (asset, signal_timestamp)
PK; returns True iff newly inserted.

`run_once` now runs entry-loop first (commits), then exit-loop
(commits). One sqlite3 connection across both. Print output shows
per-position status: HOLDING (with days remaining) or EXIT (with full
breakdown + the funding-rate events used). Summary dict expanded to
8 fields (entry: processed/entered/skipped/duplicates; exit: open/
exited/still_holding/duplicates).

### D6 Tier 2 smoke -- PASSED (hand-arithmetic match to 0.00e+00)

Window selection: scanned 4 candidate 8h boundaries 4-7 days back;
all had exactly 9 BTC binance funding events in their 72h window
(no missing data in current operational period). Chose
`signal_ts = 1780012800000 = 2026-05-29 00:00 UTC` ->
`target_exit_ts = 1780272000000 = 2026-06-01 00:00 UTC`.

The 9 funding events used (real Binance rates):

```
05-29 08:00  +0.0000539
05-29 16:00  +0.0000670
05-30 00:00  +0.0000346
05-30 08:00  +0.0000473
05-30 16:00  +0.0000621
05-31 00:00  +0.0000443
05-31 08:00  +0.0000566
05-31 16:00  +0.0000599
06-01 00:00  +0.0000570
```

sum(rate) = +0.0004828900.

Executor-computed paper_position_exits row:

| field | value |
|---|---|
| asset | BTC |
| signal_timestamp | 1780012800000 |
| entry_decided_at | 2026-05-29T00:00:00+00:00 (synthetic) |
| exit_decided_at  | 2026-06-02T14:51:23+00:00 |
| exit_timestamp   | 1780272000000 |
| exit_datetime    | 2026-06-01T00:00:00+00:00 |
| hold_days        | 3 |
| funding_events_count | 9 |
| funding_payments_usd | $+0.241445 |
| tc_entry_usd     | $0.200000 |
| tc_exit_usd      | $0.200000 |
| **net_pnl_usd**  | **$-0.158555** |
| notional_usd     | $500.00 |
| direction        | long_spot_short_perp |
| executor_version | cycle52-lifecycle |

Hand-computed:
- funding_payments = 0.0004828900 * 500 = $0.241445 -> delta 0.00e+00
- tc_each = 0.0004 * 500 = $0.200000 -> delta 0.00e+00 on each leg
- net_pnl = 0.241445 - 0.40 = -$0.158555 -> delta 0.00e+00

Idempotency re-run: 0 pending alerts; 0 open positions (the exit
closed it); 0 new exits.

**Sign-convention assertion**: all 9 events have positive rates ->
short perp receives -> funding_payments_usd > 0 (verified positive
sum). net_pnl_usd < 0 reflects 72h of current low-funding regime
not clearing the 8 bps round-trip TC. **Not a bug**: this is the
exact scenario where atlas's P>0.70 gate would have rejected the
signal at entry (low ann_rate + low expected_return -> low P
prediction). The exit arithmetic correctly produces the negative
P&L the trade would have realized in real money.

### Safety-belt grep

Ran post-rewrite over the 4 implementation files:
```
engines/funding_executor.py
scripts/migrations/cycle52a_paper_trades_add_hold_days.py
scripts/migrations/cycle52b_paper_position_exits_schema.py
services/funding_executor_service.bat
```
Pattern: `import ccxt|requests\.(post|put|delete)|urllib\.request|http\.client|aiohttp`.
Zero hits on first run -- the Cycle 51 docstring-rewrite lesson
carried forward, so this cycle's new docstrings phrase the safety
constraint abstractly. No round-trip needed.

---

## Architectural decisions captured

### "9 risk controls + 1 data-availability check" framing

Cycle 51 advertised "9-control risk layer." Cycle 52 adds
`hold_days_known` as a 10th check. Rather than renumber to "10
risk controls" (which would mis-frame data-availability as a risk
policy), the 10th is documented as a separate concept: a
data-availability gate that prevents the executor from making a
decision against incomplete inputs. The split is reflected in
the module docstring + risk_checks_json structure (RiskChecks now
carries both `hold_days_known: bool` AND `hold_days_value: int |
None` so the audit trail captures both the gate decision and the
value that would have been used).

### Cycle 52a vs Cycle 52b migrations (split rationale)

Two separate migration scripts rather than one. Reasoning:
- 52a is an ALTER TABLE (mutates existing schema)
- 52b is a CREATE TABLE (new schema)
- Each is idempotent independently
- A future rerun on a fresh DB hits 52a (paper_trades.hold_days
  already present from init_db()) -> "already exists" -> exit 0;
  then 52b ("already exists" -> exit 0). Order doesn't matter.
- A future rerun on a Cycle 51 DB (paper_trades exists, no
  hold_days, no paper_position_exits) hits 52a -> ADD COLUMN runs;
  then 52b -> CREATE TABLE runs. Both transitions clean.

Both migrations gate on row counts (52a logs current paper_trades
size pre-migration; would warn if non-zero rows existed). At cycle
end paper_trades had 0 real rows -- entry path correctly survived
the column addition because the executor's INSERT statement was
updated in the same commit to include the new column.

### Window-aligned exit timing -- X/Y collapse for integer-day holds

When hold_days is an integer AND signal_timestamp is funding-window-
aligned (which funding_alerts.timestamp IS by construction), Options
X (time-based exit) and Y (window-aligned exit) produce IDENTICAL
arithmetic:
- target_exit = signal_ts + 3*86400000 ms = signal_ts + 72h
- 72h from a funding boundary lands EXACTLY on the next funding
  boundary (every 8h; 72h = 9 * 8h)
- So target_exit is itself a funding boundary -> "next boundary at
  or after target" is target itself

Window-alignment is therefore implicit, not enforced. Becomes
material only if a future cycle introduces fractional holds (e.g.
"hold for the next 4.5 funding events") or non-aligned entries
(e.g. exit-driven entries from a different signal source). At that
point the executor would need explicit boundary-snap logic in
compute_exit. Cycle 52 documents this so a future author doesn't
need to re-derive it.

### Cycle 51 stub -> Cycle 52 real-query: daily_loss path uses
paper_position_exits

`_daily_loss_so_far_usd` now queries paper_position_exits filtered
by `substr(exit_decided_at, 1, 10) = today_utc` AND `net_pnl_usd < 0`.
Returns abs(sum) so the circuit-breaker compares positive USD
against the positive cap. UTC day boundary not local-day -- this is
the audit-trail timestamp's natural unit. The `max_daily_loss_pct`
config knob (0.02) is not yet wired into a separate check; will
fold in alongside multi-position sizing logic in Cycle 53+.

### Cycle 53 backtest path lit up

`--force-hold-days N` CLI override exists specifically so Cycle 53
backtest sweeps can iterate hold values without touching
funding_signals data. Use pattern (foreseen, not committed):
```
python -m engines.funding_executor --force-hold-days 3 ...
python -m engines.funding_executor --force-hold-days 7 ...
python -m engines.funding_executor --force-hold-days 14 ...
```
Cycle 53 will likely add a `--replay-mode` flag that loops over
historical funding_alerts ranges instead of just the live "pending"
set. The current architecture supports that cleanly via separate
input-source plumbing.

---

## Acceptance criteria

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | paper_trades added to primary_monitored | ✅ verified post-edit |
| 2 | paper_position_exits table created via migration + init_db | ✅ |
| 3 | paper_trades.hold_days added via migration + init_db | ✅ |
| 4 | get_open_positions() correct (Tier 1 step 1, 2) | ✅ all=1 BTC=1 ETH=0 |
| 5 | 3 stub risk checks replaced; concurrent-position skip works | ✅ Tier 1 step 2 |
| 6 | Exit-reconcile computes correct P&L | ✅ Tier 2 hand-match 0.00e+00 |
| 7 | Idempotency: 2nd run = 0 new entries / 0 new exits | ✅ both tiers |
| 8 | Safety-belt grep clean | ✅ first-run zero hits |
| 9 | Standard commit + push + SHA insertion follow-up | ✅ |
| 10 | Retro captures RECON decisions | ✅ this file |

---

## Open items / Cycle 53+ inputs

- **Cycle 53 (D7+D8): backtest replay.** Run the executor logic
  against the OOS funding_alerts + funding_signals historical range
  used by atlas Exp 13 (train 2024 -> test 2025-01-01..2026-03-26).
  Confirm executor's net_pnl_usd time series, summed and Sharpe'd,
  reproduces atlas's +4.65 Sharpe. This is the confidence-build
  step before any real-money move. Implementation outline:
  - Add `--replay-from <iso>` / `--replay-to <iso>` to executor
    (or build a sibling `backtest_executor.py` that calls
    `compute_exit` over historical alerts)
  - Compare per-asset Sharpe + cum return to atlas headline figures
  - Disconfirm if delta > 5%; confirm if delta < 1%
- **Cycle 54+: real-money rollout.** Small notional (~$100/asset).
  Gated on Cycle 53 reproduction. Will need exchange API integration,
  order placement state machine, position reconciliation,
  EXECUTOR_KILL_SWITCH wired into real-money path identically.
- **`max_daily_loss_pct` wiring**: currently config knob exists
  (0.02) but isn't compared against (notional-relative). Cycle 53
  or Cycle 54 should wire it to portfolio_total_notional so the
  pct cap can fire independently of the absolute $50 USD cap.
- **Multi-concurrent-position support**: current `max_concurrent_positions_per_asset=1`
  prevents reentry while a position is open. For higher-frequency
  signals (e.g. 14d hold + new entry alert mid-hold) this is
  conservative. Once Cycle 53 confirms behavior matches atlas,
  consider raising to 2 or 3 (atlas avg models/day was 1.6 at
  P>0.50 / 0.5 at P>0.70).
- **D4 follow-up from Cycle 51**: PraxisFundingExecutor task is
  registered and verified; no action needed. Cycle 52 changes are
  picked up automatically on the next scheduled trigger (next
  invocation will run cycle52-lifecycle code).
- Plus standing queue: 44d bear-regime accumulation, 44b LSTM v2
  (low-likelihood), 44h-refactor (engines/_paths.py), 44q sidecar +
  non-DB path audit, per-venue funding_rates health, TEAMS_WEBHOOK_URL
  fallback removal (46a).
