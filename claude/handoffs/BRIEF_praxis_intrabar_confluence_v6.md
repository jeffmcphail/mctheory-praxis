# Implementation Brief: Intrabar Confluence v6 (Chunked Forward Pass Fix + Retrain)

**Series:** praxis
**Priority:** P0 (unblocks v5's scoping, resolves v3 Case 3/4)
**Mode:** B (model training + backtest on trained artifacts + live DB reads)

**Estimated Scope:** M (45-75 min -- chunked-forward fix is ~5 min, print gate fix ~1 min, retrain ~25-40 min, backtest suite + retro ~10-15 min)
**Estimated Cost:** none (local CPU only, no LLM/API spend)
**Estimated Data Volume:** reuses existing `BTC_features.joblib` (51,765 sequences on 180-day ohlcv_1m); no new DB reads
**Kill switch:** if retrain exceeds 90 min wall-clock, pause for approval before continuing

---

## Context

V5 attempted to retrain the LSTM on the 180-day dataset. The retrain hung at 2h 15m wall-clock, consuming ~6h of user CPU, producing only 1 epoch of log output. Post-mortem in `RETRO_praxis_intrabar_confluence.md` (v5 PARTIAL) identified:

1. **Primary bug:** `test_pred = model(X_test)` at line 561 of `train_intrabar_lstm` does a single unchunked forward pass on the entire test set. At v3 scale (1,945 test sequences) this allocated ~90 MB transient activations and ran fast. At v5 scale (10,353 test sequences -- 5.3x) it allocates ~480 MB, hits MKL-BLAS memory-bandwidth saturation, and slows super-linearly to ~300-500s per epoch test eval vs v3's sub-second.

2. **Secondary bug (observability):** print gate at line 577 is `(epoch+1) % 25 == 0 or epoch == 0`. With 300-500s epochs, first post-epoch-1 print would have landed at 2h 5m to 3h 28m -- right in the kill-window. V3 logs looked dense only because early stopping fired at epoch 21 before the next scheduled print.

Both fixes are local to `train_intrabar_lstm`. No architecture, loss function, feature set, batch size, learning rate, or patience changes. This is strictly an apples-to-apples retrain with the v3 pipeline on the v4-extended dataset.

---

## Objective

1. Apply the two fixes to `engines/intrabar_predictor.py`.
2. Retrain LSTM on the existing `BTC_features.joblib`.
3. Rerun the v3 diagnostic suite (6 backtests: 4 fee sensitivity + 2 regime split).
4. Interpret results against v3's decision framework (Case 1/2/3/4) -- FINALLY resolve the v3 standoff.

---

## Detailed Spec

### Fix 1 -- Chunked Test Forward Pass (PRIMARY)

In `engines/intrabar_predictor.py`, modify `train_intrabar_lstm`. Find the per-epoch test eval block around line 561:

```python
# OLD (around line 561)
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = _pinball_loss(test_pred, y_test, quantiles_tensor).item()
```

Replace with:

```python
# NEW
TEST_CHUNK = 1024
model.eval()
with torch.no_grad():
    test_preds = []
    for start in range(0, len(X_test), TEST_CHUNK):
        chunk = X_test[start:start + TEST_CHUNK]
        test_preds.append(model(chunk))
    test_pred = torch.cat(test_preds, dim=0)
    test_loss = _pinball_loss(test_pred, y_test, quantiles_tensor).item()
```

**Also apply the same chunking pattern** to the final evaluation block (lines 593-612 per v5 retro). Same chunk size, same structure.

**Also grep for other unchunked forward passes** in the file:
- `cmd_predict` -- if it takes >1024 sequences in one call, chunk it
- `cmd_confluence` -- usually 1 sequence at a time, probably fine
- `cmd_backtest` -- likely iterates one prediction at a time, probably fine

Report findings in retro: which functions were touched, which were verified safe.

### Fix 2 -- Looser Print Gate (SECONDARY)

Around line 577:

```python
# OLD
if (epoch + 1) % 25 == 0 or epoch == 0:

# NEW
if epoch < 10 or (epoch + 1) % 5 == 0:
```

This prints every epoch for the first 10, then every 5 epochs. No model behavior change, just diagnostic cadence.

### What NOT to change

- LSTM architecture
- Feature set / feature engineering
- Loss function (pinball loss stays)
- Training batch size (64)
- Learning rate (unchanged)
- Patience counter (20)
- Early stopping logic
- Optimizer
- Any data loading code
- Any CLI argument
- `build-features` is NOT rerun -- use existing `BTC_features.joblib`

Run `ast.parse()` and ASCII check after edits. Grep for unchunked `model(X_*)` patterns to confirm Fix 1 has full coverage.

---

## Execution Steps

### Step 1 -- Verify pre-flight state

```powershell
tasklist | findstr python
```

Expected: only Session 0 services. If any user-session Python processes are running, kill them first.

### Step 2 -- Apply fixes

Edit `engines/intrabar_predictor.py` per Fix 1 and Fix 2 above.
AST parse + ASCII check.

### Step 3 -- Retrain (reuse existing features)

```powershell
python -m engines.intrabar_predictor train --asset BTC
```

**Use `run_in_background: true` with explicit logging to a file.** Monitor progress every 5 minutes per the new workflow protocol. Expected behavior:

- Epoch 1 print at ~80s (v5 showed 64s; v6 should be similar)
- Epochs 2-10 print individually (new gate fires `epoch < 10`)
- If per-epoch time is ~80s, epochs 2-10 complete in ~12 min
- Epochs 11+ print every 5 (15, 20, 25, ...)
- Early stopping likely fires between epoch 25-60 based on v3's epoch 21 + more data
- **Kill switch: if wall-clock exceeds 90 min OR no epoch prints within a 10-minute window after epoch 1, pause and report**

### Step 4 -- Validate training output

Report in retro:
- Final epoch reached
- Early stopping epoch (if fired)
- Per-epoch time average
- Directional accuracy per horizon at final epoch
- p10-p90 coverage per horizon
- XGBoost directional accuracy per horizon (5-bar especially)
- Any warnings or unusual metric behavior

If quantile coverage drops below 70% or directional accuracy is radically different from v3 (v3 was 48-55%), flag for Chat review before proceeding to backtests.

### Step 5 -- Run v3 diagnostic suite (6 backtests)

Same commands as v3 brief, run in a single bash script with unified log:

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

Pipe each to a separate log OR one combined log. Either is fine.

### Step 6 -- Interpret against v3 decision framework

The v3 framework (all four cases):

- **Case 1 (maker-fee P&L positive):** hit rate > 50% after dual-horizon confluence filter at 0.10% round-trip. Means dual-horizon is real, tune filters from here.
- **Case 2 (one regime significantly positive):** one of reverting/random/trending shows net-positive P&L with sample > 30. Add Hurst gate (A1).
- **Case 3 (all regimes negative at all fees):** no regime profitable even at zero fees. Structural. Pivot to A2 (momentum inversion) or A4 (XGB-only 5-bar).
- **Case 4 (direction concentration):** still > 90% one direction. Given balanced dataset now, persistence of this is structural filter defect, not window bias.

**Key questions the retro must answer:**
1. What is the LONG/SHORT split in the new diagnostic run? (Balanced = Case 4 was window bias, resolved.)
2. Is the strategy net-positive at zero fees on balanced data?
3. Does any regime flip net-positive?

### Step 7 -- Write retro

`claude/retros/RETRO_praxis_intrabar_confluence.md` (overwrite v5 PARTIAL). Structure:

- TL;DR with decision framework verdict
- What the fixes delivered (epoch time restored, print cadence fixed)
- Training results (final metrics, early stopping epoch)
- Backtest results table (6 runs)
- Regime split table with LONG/SHORT per regime
- Case 1/2/3/4 fire status
- Recommendation for v7 (tune / pivot / further diagnostic)
- State at session end

---

## Acceptance Criteria

- [ ] `engines/intrabar_predictor.py` modified with Fix 1 (chunked test forward + chunked final eval) and Fix 2 (looser print gate)
- [ ] AST parse and ASCII check pass
- [ ] Grep confirms no remaining unchunked `model(X_*)` calls on large tensors
- [ ] `train --asset BTC` completes within 90-min kill-switch budget
- [ ] Training log shows per-epoch prints for epochs 1-10, then every 5
- [ ] Final model artifacts saved: `BTC_multi_horizon_lstm.pt`, `BTC_quant_mh_models.joblib`, `BTC_multi_horizon.joblib`
- [ ] All 6 backtests run end-to-end without errors
- [ ] Retro reports v3 decision framework verdict (Case 1/2/3/4) with explicit citation of the evidence
- [ ] Retro explicitly states: is the dual-horizon mean-reversion hypothesis viable, dead, or still ambiguous?
- [ ] Cycle counter: this retro should advance v5 to "COMPLETE" or identify a new blocker

---

## Known Pitfalls

- **Feature reuse.** Do NOT rerun `build-features` -- v5's feature artifact is valid. Rerunning wastes 15s and the stale-cleanup step will delete the existing model cache (which is fine since there isn't one, but still pointless).
- **Chunk size selection.** 1024 is the recommended default. Smaller (512) is also fine if memory is a concern. Do NOT go above 2048 -- defeats the purpose.
- **Final eval block.** The one-shot final evaluation after training finishes also does a big forward pass. If you chunk the per-epoch test but not the final eval, training finishes in 25 min and then hangs at the very end for another hour. Apply Fix 1 to BOTH locations.
- **Device handling.** If model is on CPU (which it is on Windows laptop), chunks stay on CPU. No device migration needed. But verify with a pre-flight `model.device` check if unsure.
- **Single-process discipline.** Protocol holdover from v2 incident. Verify `tasklist | findstr python` before launch. DO NOT launch second training run.
- **Progress reporting per new workflow rules.** Every 5 minutes report current epoch, epoch time, elapsed. If no epoch prints within 10 min after epoch 1, investigate before the hang compounds. If projected runtime exceeds 90 min kill switch, pause for approval.

---

## Do NOT

- Do NOT modify LSTM architecture, feature set, loss function, batch size, learning rate, or patience. This is still a controlled retrain for apples-to-apples comparison.
- Do NOT touch ETH. Still on hold until BTC strategy decision lands.
- Do NOT modify backtest code.
- Do NOT commit code. v2/v3/v4/v6 edits accumulate uncommitted; Chat will consolidate after v6 lands cleanly.
- Do NOT modify the scheduled task.
- Do NOT rerun build-features.

---

## Open Questions (OK to decide during implementation, document in retro)

1. If the chunked forward pass runs much faster than v3's per-second-ish (e.g., each chunk is 20ms and there's no memory pressure), should we use a larger chunk size like 2048? (Recommendation: stick with 1024 for v6, tune later if performance warrants.)
2. If early stopping fires very fast (e.g., epoch 5-10), should we lower patience? (Recommendation: no, report and move on; early convergence on more data is acceptable.)
3. If LSTM quantile coverage is noticeably worse than v3's 80-82% band (e.g., drops to 65%), is that a structural problem or noise? (Recommendation: flag for Chat review; do NOT treat as proof of problem without discussion.)

---

## References

- Retro v5 PARTIAL: `claude/retros/RETRO_praxis_intrabar_confluence.md` (current state)
- Retro v4 data extension: `claude/retros/RETRO_praxis_intrabar_data_extension.md`
- Brief v5: `claude/handoffs/BRIEF_praxis_intrabar_confluence_v5.md`
- Brief v3 (decision framework + fee/regime flags): `claude/handoffs/BRIEF_praxis_intrabar_confluence_v3.md`
- File to modify: `engines/intrabar_predictor.py` (train_intrabar_lstm function; 2 small localized edits)
- Features to reuse: `models/intrabar/BTC_features.joblib` (51,765 sequences, 180-day dataset)
- Workflow modes doc: `claude/WORKFLOW_MODES_PRAXIS.md`
