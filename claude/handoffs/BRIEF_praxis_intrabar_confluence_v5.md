# Implementation Brief: Intrabar Confluence Retrain + Rediagnose on 180d (v5)

**Series:** praxis
**Priority:** P0 (resolves Case 3 vs Case 4 standoff from v3)
**Mode:** B (model training + backtest + live DB reads)
**Estimated Scope:** M (Medium -- 60-90 min wall-clock, mostly training)
**Date:** 2026-04-22
**Follow-up to:** RETRO_praxis_intrabar_data_extension.md (v4)

---

## Context

V4 extended `ohlcv_1m` from 34.8 days to 180 days (100K -> 518K rows). The dataset now spans October 2025 through April 2026, which includes multiple market regimes rather than V3's pure downtrend window. The V3 diagnostic concluded with:

- Case 3 (strategy dead): net-negative at all fees, R:R 0.57, all regimes negative
- Case 4 (window bias): 163 SHORT / 0 LONG suggests the strategy never got tested in balanced conditions

These two cases were ambiguous. Code's v3 recommendation weighted Case 4 higher (65/35) but that was before we had extended data. Now we can resolve the ambiguity directly.

This brief:
1. Rebuilds features on the extended 180-day dataset (BTC only; ETH still on hold)
2. Retrains the LSTM + XGBoost models
3. Reruns the v3 diagnostic suite (6 backtests: fee sensitivity + regime split)
4. Interprets results against the v3 decision framework

Expected outcome: either Case 4 resolves (balanced direction distribution, strategy viability confirmed or denied cleanly) OR direction bias persists (confirming structural filter defect, Case 3 terminal).

---

## Objective

Determine with high confidence whether the dual-horizon confluence strategy has edge, and what tuning direction (if any) to take next.

---

## Detailed Spec

### Step 1 -- Rebuild Features

```powershell
python -m engines.intrabar_predictor build-features --asset BTC
```

Expected: stale cleanup at top of `cmd_build_features` removes v3 artifacts (`models/intrabar/BTC_*.joblib`, `.pt`). New features written from 180-day data.

Expected rough counts:
- 1-min bars: ~259K -> aggregate to 5-min bars: ~51,800
- Feature rows: ~51,700 (after dropping first 60 for sequence warmup)
- LSTM sequences: ~51,600

Report actual counts in retro.

### Step 2 -- Retrain LSTM + XGBoost

```powershell
python -m engines.intrabar_predictor train --asset BTC
```

**Important training notes:**
- At 5.3x more data than v3 (51,600 vs 9,721 sequences), expect proportionally longer training. V3 ran 15s/epoch; v5 should run ~80s/epoch.
- V3 early-stopped at epoch 21. Extended data may converge faster per epoch OR slower (more diverse patterns to learn). Budget 25-90 minutes.
- **MUST use `run_in_background: true`** -- foreground timeout will kill it.
- **MUST verify single-training-process discipline BEFORE launch:**
  ```
  tasklist | findstr python
  ```
  Confirm no user-session training processes are active. Kill any that are. Do NOT launch a second train command.
- Monitor epoch progress via the log file. If no epoch prints for 10+ minutes after the first one, investigate before assuming it's hung.

**Expected reports from training:**
- Directional accuracy per horizon (v3: 48-55%)
- p10-p90 coverage (v3: 80.0-81.9%, near-perfect)
- LSTM median distribution width (v3: tightly clustered near zero)

If these metrics look radically different from v3, flag it in the retro -- could indicate either a fundamentally better signal OR a training instability.

### Step 3 -- Rerun V3 Diagnostic Suite

Same 6 commands as v3 Brief:

```powershell
# Fee sensitivity (z=1.5, min-mag=0.0 for max signal count)
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode zero
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode maker
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode taker
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode default

# Regime split
python -m engines.intrabar_predictor backtest --asset BTC --zscore 1.5 --min-mag 0.0 --fees-mode maker --regime-split
python -m engines.intrabar_predictor backtest --asset BTC --zscore 2.0 --min-mag 0.05 --fees-mode maker --regime-split
```

The backtest code from v3 doesn't need modification. Pipe each to a separate log file for archival.

### Step 4 -- Interpret Against V3 Decision Framework

Evaluate against the same 4 cases from v3:

**Case 1 (maker-fee P&L positive):** hit rate > 50% at 0.05%/side round-trip after confluence filter. Would mean dual-horizon is real, tune filters from here.

**Case 2 (one regime significantly positive):** one of reverting/random/trending shows net-positive P&L with sample > 30. Would mean add Hurst gate as a filter (A1).

**Case 3 (all regimes negative at all fees):** no regime profitable even at zero fees. Strategy structurally broken. Pivot to A2 (momentum inversion) or A4 (XGB-only 5-bar).

**Case 4 (direction concentration):** still > 90% one direction. Now that we have balanced market data, persistence of this would be structural filter defect, not window bias.

**Key question the retro must answer:**
- What is the LONG/SHORT split in the new diagnostic run? (Balanced = Case 4 was window bias, resolved. Still skewed = filter has structural defect.)
- Is the strategy net-positive at zero fees? (Yes = tune. No = structural.)
- Does any regime flip positive? (Yes = apply regime gate. No = pivot.)

---

## Acceptance Criteria

- [ ] `build-features` completes with ~51,600 sequences (+/- 5%) on 180-day data
- [ ] `train` completes (early stopping OR full 150 epochs), LSTM saved to `BTC_multi_horizon_lstm.pt`
- [ ] Training reports per-horizon directional accuracy and p10-p90 coverage
- [ ] All 6 backtests run end-to-end without errors, logged separately
- [ ] Retro reports a decision framework verdict (Case 1/2/3/4)
- [ ] Retro explicitly states: is the dual-horizon mean-reversion hypothesis viable, dead, or still ambiguous?
- [ ] AST parse and ASCII check pass on any files touched (expected: none; backtest code unchanged from v3)
- [ ] Single-training-process discipline verified pre-launch, mid-training, post-training

---

## Do NOT

- Do NOT modify the LSTM architecture, feature set, or loss function. This is a retrain with same code on extended data -- a controlled experiment.
- Do NOT touch ETH. Still on hold until BTC strategy decision lands.
- Do NOT modify the backtest code. It worked in v3; use it unchanged.
- Do NOT commit code yet. V2, v3, v4 edits are all uncommitted; Chat will consolidate after v5.
- Do NOT run multiple training processes in parallel (v2 collision). Verify with `tasklist | findstr python` before launch.
- Do NOT modify the scheduled task `PraxisCrypto1mCollector`.

---

## Known Pitfalls

- **Training time scaling.** 5.3x data, expect ~5x training time. At 80s/epoch and ~20-30 epochs to early stopping, budget 25-40 min realistic, up to 90 min if full 150 epochs.
- **Early stopping may not fire at epoch 21 this time.** More data = more patterns to learn. If loss curve is still descending at epoch 40, that's fine -- let it run.
- **Memory.** 51K sequences of (60, 5) float32 = ~60 MB raw, ~250 MB with PyTorch overhead for batching. Should fit fine, but monitor for OOM if other processes are running.
- **Direction bias interpretation.** If LONG/SHORT is still 0/N in v5 on balanced data, that is the answer to Case 4 -- it's a filter defect, not a window artifact. Do not try to explain it away.
- **Regime bucket sizes.** With 5.3x signals, the "reverting" bucket should now have enough trades (50+) to be actionable. If it's still <20, that's a structural finding (5-min BTC rarely mean-reverts) worth noting.
- **LSTM quantile calibration.** V3's near-perfect 80% coverage may degrade on more data (more outliers in training set). Report actual numbers -- if coverage drops to < 70%, that's a signal the model is overfitting the training distribution.

---

## Open Questions (OK to decide during implementation, document in retro)

1. Should training use a larger batch size given more data? (Recommendation: stick with v3's batch=64 for apples-to-apples comparison; tune later if retained)
2. If early stopping fires very fast (e.g., epoch 5), should we rerun with lower learning rate? (Recommendation: note it, don't rerun; collect diagnostic first)
3. If directional accuracy drops significantly below v3's 50-54% band, is that concerning or expected? (Recommendation: expected if data includes more volatile regimes; report and flag for Chat review)

---

## Time Budget

- Step 1 (features): < 2 min
- Step 2 (training): 25-90 min realistic
- Step 3 (6 backtests): < 30 seconds total
- Step 4 (retro): 10-15 min of Code's time

Total: 40-110 min wall-clock. Expect ~60 min typical.

---

## References

- Retro v3 (decision framework): `claude/retros/RETRO_praxis_intrabar_confluence.md`
- Retro v4 (data extension): `claude/retros/RETRO_praxis_intrabar_data_extension.md`
- Brief v3 (backtest flags): `claude/handoffs/BRIEF_praxis_intrabar_confluence_v3.md`
- File to modify: NONE (diagnostic replay, code unchanged since v3)
- Artifacts to replace: `models/intrabar/BTC_*` (stale cleanup handles this automatically)
- Workflow mode doc: `claude/WORKFLOW_MODES_PRAXIS.md`
