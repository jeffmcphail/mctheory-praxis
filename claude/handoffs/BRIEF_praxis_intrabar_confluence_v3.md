# Implementation Brief: Intrabar Confluence v3 (Diagnostic Before Redesign)

**Series:** praxis
**Priority:** P1 (diagnostic gate -- decides whether we pivot or tune)
**Estimated Scope:** S (Small -- 30-60 min, no retraining required)
**Date:** 2026-04-22
**Follow-up to:** RETRO_praxis_intrabar_confluence.md (v2)

---

## Context

V2 pipeline works end-to-end. Training completes in 7 minutes, LSTM quantile calibration is near-perfect (80.0-81.9% coverage), XGBoost 5-bar hits 53.9% directional. But the dual-horizon confluence filter produced zero trades at default settings and -69% P&L at diagnostic settings on a 7-day held-out window.

The retro proposes 4 redesign options (A1-A4) but Chat's decision is: **don't iterate filter knobs yet.** First run two cheap diagnostics that tell us whether the strategy is fundamentally broken or trapped in a bad window.

The two diagnostics:

1. **Realistic fee model.** The 0.30% round-trip assumption is pessimistic. Re-run with 0.05%/side (Binance maker) and 0% (rebate venue) to see if underlying 47.9% hit rate is actually break-even with better execution. **Free** -- just rerun backtest with different cost param.

2. **Regime-conditional split.** Split the 603 z-score triggers by Hurst regime (mean-reverting H<0.45 vs trending H>=0.55 vs random). If the strategy has positive P&L in one regime and negative in another, that's the filter we should have been using all along. If P&L is negative across all regimes, strategy is dead. **Free** -- we already have the data.

Only AFTER these diagnostics do we decide between the redesign options.

---

## Objective

Add two diagnostic modes to `engines/intrabar_predictor.py` backtest command:
1. `--fees-mode` flag that lets us specify maker-only (0.10% RT), taker (0.20% RT), or custom (existing 0.30% RT default)
2. `--regime-split` flag that breaks down backtest results by Hurst regime

Run both on BTC. Interpret results and decide whether to pivot (A2/A3/A4) or tune the mean-reversion approach (A1).

---

## Detailed Spec

### Modifications to `engines/intrabar_predictor.py`

**1. Add fee preset constants** near the top of the backtest section:

```python
FEE_PRESETS = {
    "maker": 0.0005,      # 0.05% per side, Binance VIP maker
    "taker": 0.0010,      # 0.10% per side, Binance standard taker
    "default": 0.0015,    # 0.15% per side (v2 assumption)
    "zero": 0.0,          # rebate venue / paper trading
}
```

**2. Extend `cmd_backtest` to accept new flags:**

```python
p_bt = subs.add_parser("backtest", ...)
p_bt.add_argument("--asset", required=True)
p_bt.add_argument("--zscore", type=float, default=2.0)
p_bt.add_argument("--min-mag", type=float, default=0.05)
p_bt.add_argument("--fees-mode", choices=["maker", "taker", "default", "zero", "custom"],
                  default="default")
p_bt.add_argument("--fees-custom", type=float, default=None,
                  help="Custom per-side fee (e.g. 0.0008 for 0.08%)")
p_bt.add_argument("--regime-split", action="store_true",
                  help="Break down results by Hurst regime")
```

**3. Resolve the fee:**

```python
if args.fees_mode == "custom":
    fee_per_side = args.fees_custom or FEE_PRESETS["default"]
else:
    fee_per_side = FEE_PRESETS[args.fees_mode]
round_trip = fee_per_side * 2
print(f"  Fee mode: {args.fees_mode} | per-side: {fee_per_side*100:.3f}% | round-trip: {round_trip*100:.3f}%")
```

Thread `round_trip` through the backtest loop instead of the hardcoded 0.003.

**4. Regime tagging.** When iterating through signals in the backtest, tag each with its Hurst regime at entry:

```python
hurst_at_entry = features_row.get("hurst_60bar", 0.5)
if hurst_at_entry < 0.45:
    regime = "reverting"
elif hurst_at_entry >= 0.55:
    regime = "trending"
else:
    regime = "random"
signal_record["regime"] = regime
```

**5. Regime-split reporting.** After the main backtest summary, if `--regime-split` is set, emit a per-regime table:

```
Regime-split results:
  Regime       Trades   HitRate(noCost)  HitRate(wCost)   MeanPnL   CumPnL
  reverting    N        XX.X%            XX.X%            +X.XX%    +X.X%
  random       N        XX.X%            XX.X%            -X.XX%    -X.X%
  trending     N        XX.X%            XX.X%            -X.XX%    -X.X%
```

Include mean Hurst and signal direction breakdown (LONG vs SHORT count) per regime.

### What NOT to change

- Do NOT retrain. Current `models/intrabar/BTC_*.joblib` files are fine.
- Do NOT modify training loop, LSTM architecture, or feature engineering.
- Do NOT build ETH features -- still on hold.
- Do NOT commit code yet (v2 also uncommitted). Chat will consolidate commits after this diagnostic round.

---

## Commands to Run

After modifications, run these 6 backtests and report results in the retro:

```powershell
# Fee sensitivity (default z=1.5, min-mag=0.0 for max signal count)
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode zero
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode maker
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode taker
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode default

# Regime split at realistic settings
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode maker --regime-split
python -m engines.intrabar_predictor backtest --asset BTC --zscore 2.0 --min-mag 0.05 --fees-mode maker --regime-split
```

---

## Acceptance Criteria

- [ ] `backtest` accepts `--fees-mode`, `--fees-custom`, `--regime-split` flags
- [ ] Fee mode choice is printed in backtest output header
- [ ] All 6 commands run end-to-end without errors
- [ ] Regime-split output shows 3 rows (reverting/random/trending) with trade counts that sum to total signals
- [ ] File still passes AST parse and ASCII check
- [ ] Retro reports full output of all 6 runs

---

## Decision Framework (for Chat review of Retro)

After Code delivers the retro, Chat will use these decision rules:

**Case 1: maker-fee P&L is positive (hit rate > 50% at 0.05%/side)**
- The v2 finding was a fee artifact. Keep dual-horizon mean-reversion. Move to tuning filters.

**Case 2: one regime is significantly positive, others negative**
- Add regime as a gate. E.g., "only trade when Hurst < 0.45." Rebuild the v2 confluence with this filter.

**Case 3: all regimes are negative at all fee levels**
- Strategy is dead. Pivot to Option A2 (momentum continuation) or A4 (XGB-only 5-bar).

**Case 4: signals are concentrated in one direction (all SHORT or all LONG)**
- Confirms window bias. Need to extend dataset (Option C) before making any decision.

---

## Known Pitfalls

- **Regime tagging needs the feature at entry time, not current time.** Make sure you read Hurst from the feature row corresponding to the signal bar, not from a fresh calculation.
- **Hurst thresholds 0.45/0.55 are heuristic.** If the split gives weirdly uneven bucket sizes (e.g., 99% trending), that's interesting -- means 5-min BTC is structurally trending and the mean-reversion hypothesis was always fighting upstream.
- **Small bucket sizes.** If any regime has < 20 trades, note it in retro -- results from tiny buckets aren't actionable.
- **Fee presets.** Binance maker fees are 0.02% with BNB discount but we're using 0.05% as a conservative maker estimate. That's fine.

---

## Open Questions (OK to decide during implementation, document in retro)

1. Should the regime split also break down by signal direction (LONG vs SHORT within each regime)? (Recommendation: yes, it's almost free and reveals the window-bias issue)
2. Should we also report Sharpe-like metric per regime? (Recommendation: skip -- the sample is too small for meaningful Sharpe; cumulative P&L and hit rate are enough)

---

## References

- Brief v2: `BRIEF_praxis_intrabar_confluence_v2.md`
- Retro v2: `RETRO_praxis_intrabar_confluence.md`
- File to modify: `engines/intrabar_predictor.py` (existing, trained artifacts in `models/intrabar/`)
- Training artifacts: do not modify, just read
