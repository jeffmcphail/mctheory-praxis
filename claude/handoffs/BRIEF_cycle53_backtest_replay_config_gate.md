# Cycle 53 (D7+D8): backtest replay + monitor/executor config-gate fix

**Predecessor:** Cycle 52 (`67157a5` + `f188a5b`) -- paper-trading
position lifecycle live.
**Mode:** Build-then-verify, one cycle. Option 1 (build as-is, decompose
the gap). Still NO real money, NO exchange API.

## Purpose

Reframe "verified" away from the literal Sharpe-match target (wrong
criterion for a deliberately funding-only executor) toward a P&L
DECOMPOSITION: prove the executor's funding-only P&L equals atlas Exp 13's
reported P&L minus a small, measured basis term, with config-gate zeroing
accounted for explicitly. If the residual is ~0 per asset, the executor's
omissions are understood and bounded -> "verified" in the sense that
matters before real money.

## D7 acceptance criteria (decompose the gap)

1. **Funding term exact match** -- executor cumulative funding per trade
   matches atlas's funding-only component to < 1e-6 USD/trade. Confirms the
   SQL aggregation equals the pandas aggregation over identical inputs.
2. **Basis term magnitude per asset** -- mean basis drag + basis volatility
   per asset over the OOS window; is the omitted basis small enough that
   funding-only is a usable proxy?
3. **Config-gate zeroing count** -- per asset, % of gate-passing days atlas
   zeroed (argmax config thresholds not met). The fraction the executor
   would book that atlas wouldn't.
4. **Concurrency adjustment** -- relax max_concurrent_positions to infinity
   during replay (production keeps 1). Documented.
5. **Decomposition output** -- per-asset table: exec_cum / atlas_cum /
   basis_drag / config_gate_zeroing / residual (~0), plus aggregate.
6. **Live-monitor config-gate check** -- does funding_monitor apply the
   per-config min_funding_ann + pct_positive thresholds when writing
   funding_alerts, or gate on P alone? Could unblock or block Cycle 54.

## Build approach (approved)

- D7a: regenerate per-(asset, day) predictions via `predict_model` over the
  OOS cache + joblib model (per-(asset,window) parquets don't exist; do NOT
  synthesize from phase4_portfolio aggregates).
- D7b: two alert streams (P>0.50 and P>0.70). Synthesize a harness DB
  (funding_rates from the identical cache series; funding_signals carrying
  argmax hold_days + config fields; funding_alerts per stream) and run the
  PRODUCTION executor with an injected post-OOS clock + relaxed caps.
  `--force-hold-days` cannot be used (atlas's hold varies per day).
- D7c: comparison table per the 6 criteria.
- now_func dependency injection added to the executor (cleanly factorable;
  optional, defaults to real now -> zero production change).

## D7 finding -> D8 (config-gate fix, approved Option 1)

Criterion 6 confirmed the BLOCKING branch: the monitor gates on argmax-P >
gate ALONE; the per-config hard thresholds live only in the backtest path.
So the live system has run a DIFFERENT (unverified) strategy than atlas's
Sharpe +4.65 measured since Cycle 41 -- it books trades atlas zeroed (3.3%
of P>0.70 signals, 17% of P>0.50, the latter netting -0.90%). It has not
bitten only because funding_alerts has stayed at 0 rows in the sit-out
regime. Cycle 54 real money must run the strategy atlas verified.

Two fixes enforce the SAME strategy definition at two layers:

- **D8a (monitor):** scripts/funding_monitor.py -- after argmax-P
  selection, before alert write, gate on
  `config_ok = (ann_rate >= cfg.min_funding_ann AND pct_positive >=
  cfg.min_pct_positive)`. Emit funding_alerts only if (P>gate AND
  config_ok). Write the four config fields to funding_signals so the
  executor can JOIN them.
- **D8b (executor):** engines/funding_executor.py -- 11th check
  `config_thresholds_met` (data-availability tier, like hold_days_known).
  Extend the funding_signals JOIN to pull ann_rate/pct_positive/
  min_funding_ann/min_pct_positive; skip with reason
  `config_thresholds_not_met` (or `_unknown` if any field NULL). Toggle via
  `enforce_config_gate` (default True; False only for backtest baseline).

Schema: funding_signals had 3 of 4 fields (Cycle 41); only min_pct_positive
missing -> small idempotent migration
(`cycle53_funding_signals_add_min_pct_positive.py`) + init_db update.

## Verification

- Re-run the harness with the monitor gate -> leaked alert count drops to 0
  (suppressed = the zeroed count exactly).
- Re-run with all alerts emitted + executor enforcement -> executor skips
  the config-fails; post-fix decomposition execOnly -> 0.0000.
- Synthetic executor smoke (Cycle 52 Tier-1 pattern): inject a
  funding_alerts + funding_signals pair where ann_rate < min_funding_ann ->
  decision='skip', reason 'config_thresholds_not_met'. Positive control
  enters.

## Out of scope for D8

- Re-running atlas's Exp 13 reproduction (Cycle 40 stands).
- Any executor logic beyond the 11th check.
- Atlas update (the atlas entry is already correct; this fixes the live
  system to match it).

## Acceptance (full Cycle 53)

1. D7 PASSED (corrected detector) -- commit as-is.
2. D8a monitor config-gate; re-decomposition shows leak booked = 0.
3. D8b executor 11th check; smoke confirms skip path.
4. funding_signals schema extended (migration).
5. Safety-belt grep clean.
6. Standard commit + push + SHA insertion follow-up.
7. Retro captures: self-correction story; why criterion-5's independent
   residual was the rigor that caught the gap; "unverified strategy since
   Cycle 41"; Cycle 54 real-money unblock status.
