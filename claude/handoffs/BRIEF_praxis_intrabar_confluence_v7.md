# Implementation Brief: Intrabar Confluence v7 (Diagnostic Gate Before Architecture)

**Series:** praxis
**Priority:** P0 (gates v8 direction -- retry current approach vs pivot)
**Mode:** B (CPU-intensive synthetic workload + model training on local features)

**Estimated Scope:** S (30-45 min total)
**Estimated Cost:** none (local CPU only, no LLM/API spend)
**Estimated Data Volume:** reuses existing `models/intrabar/BTC_features.joblib` (51,765 sequences, 180d) + synthetic matmul workload
**Kill switch:** 
- Diagnostic 1 (thermal test): if exceeds 10 min, pause
- Diagnostic 2 (XGB-only training): if exceeds 30 min, pause

---

## Context

V6 retro (`RETRO_praxis_intrabar_confluence.md`) surfaced two independent findings:

1. **Stage 2 slowdown:** LSTM per-epoch time grew ~1.7x per epoch from epoch 4 onward (76->93->94->153->288->478 s). Ratio structure too consistent for BLAS jitter. WorkingSet stable at ~650 MB. Code's H3 hypothesis (thermal throttling) ranked most likely based on monotonic-decay curve characteristic.

2. **Model not learning:** train loss moved 0.0013 over 6 epochs (pinball-loss baseline for coin-flip). Test loss flat at 0.0801 with exactly one 0.0800 reversion. Directional accuracy tight-clustered at 49-50% across all 5 horizons every epoch. Near-zero predictions producing coin-flip directional agreement with zero-mean targets.

Finding (1) is a pipeline bug. Finding (2) is a hypothesis problem: the 5-min BTC OHLCV + 25 features may not contain learnable signal at these horizons regardless of training pipeline health. These need to be disentangled before committing to any architectural change.

Before spending more cycles on the current approach (2-layer LSTM on 25 features predicting 1/3/5/10/15-bar quantiles), answer two questions cheaply:

- Does thermal throttling explain the Stage 2 timing behavior?
- Does the 180-day 5-min BTC dataset have directional signal that a non-LSTM learner can find?

This Brief executes two small, independent diagnostics and produces a decision for v8.

---

## Objective

Determine whether to:
- **v8a:** Fix thermal throttling + retry LSTM (if H3 confirmed AND XGB finds signal)
- **v8b:** Pivot to XGB-only ensemble (if H3 confirmed but LSTM can't find signal XGB can)
- **v8c:** Abandon 5-min directional prediction entirely (if neither LSTM nor XGB find signal)
- **v8d:** Dig deeper into memory/allocator behavior (if H3 NOT confirmed but LSTM still slowed down)

---

## Diagnostic 1: Thermal Throttling Test (~5 min)

### Purpose
Confirm or refute H3 from v6 retro. The hypothesis: Windows laptop under sustained MKL-BLAS load at 2.4 cores has its effective clock speed throttled after a few minutes, producing the observed ~1.7x per-epoch slowdown.

### Method
Create a new script `scripts/diag_thermal_cpu.py` that runs a fixed CPU-heavy workload 10 times in a tight loop, reporting per-iteration wall-clock. If thermal throttling is real, iteration N time will grow monotonically vs iteration 1.

### Sketch

```python
# scripts/diag_thermal_cpu.py
"""Diagnostic: does sustained MKL-BLAS load exhibit thermal throttling?

Runs a fixed matmul workload 10 times, reports per-iteration wall-clock.
If iteration-10 time is >=1.5x iteration-1 time, H3 is confirmed.
"""
import time
import numpy as np

# Size chosen for ~5-10 s per iteration (similar compute profile to LSTM epoch)
N = 8000   # matmul dimension; adjust based on first-iteration time

def workload():
    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    return (a @ b).sum()

print(f"CPU thermal throttling diagnostic: {N}x{N} matmul x 10 iterations")
print(f"{'Iter':>4} {'Time (s)':>10} {'Ratio vs iter 1':>18}")
print("-" * 36)

t0 = time.time()
times = []
for i in range(10):
    it_start = time.time()
    _ = workload()
    it_time = time.time() - it_start
    times.append(it_time)
    ratio = it_time / times[0] if times else 1.0
    print(f"{i+1:>4} {it_time:>10.2f} {ratio:>18.2f}x", flush=True)

total = time.time() - t0
max_ratio = max(times) / times[0]

print(f"\nTotal wall-clock: {total:.1f} s")
print(f"Max ratio: {max_ratio:.2f}x")
if max_ratio >= 1.5:
    print(f"VERDICT: H3 CONFIRMED -- thermal throttling or sustained-load slowdown detected.")
elif max_ratio >= 1.2:
    print(f"VERDICT: H3 PARTIAL -- some slowdown but not matching v6's 1.7x pattern.")
else:
    print(f"VERDICT: H3 RULED OUT -- per-iteration time stable. Slowdown cause is elsewhere.")
```

### Tuning Notes

- **N=8000**: gives ~5-10 s per iteration on a typical laptop. If iteration 1 is <2 s or >20 s, rerun with N tuned to land in the 5-10 s range. Don't skew the test by using a trivial workload.
- **10 iterations**: enough to see the ~1.7x growth pattern v6 showed across 6 epochs. More is fine but unnecessary.
- **Kill switch**: if total wall-clock > 10 min, pause and report. On a modern laptop this should finish in 1-3 min total.
- Before running, confirm no other heavy Python processes (no other training, no live backtest). `tasklist | findstr python` should show only Session 0 services + this Diagnostic's own process.

### Interpretation

- **Max ratio >= 1.5**: H3 confirmed. The Stage 2 slowdown in v6 was thermal/sustained-load. v8 should address this (see Section Next Steps).
- **Max ratio 1.2-1.5**: partial. Throttling exists but the 1.7x LSTM pattern may have had additional causes. Continue to Diagnostic 2 but flag.
- **Max ratio < 1.2**: H3 ruled out. Stage 2 slowdown must have a different cause (allocator fragmentation, optimizer state, something in PyTorch's autograd or checkpointing). v8 would need a different investigation vector.

Report verdict + full per-iteration table in the retro.

---

## Diagnostic 2: XGB-Only Signal Probe (~20-30 min)

### Purpose
Determine if the 180-day 5-min BTC dataset contains directional signal at 1/3/5/10/15-bar horizons that a non-LSTM model can find. V3's final training run had XGBoost 5-bar at 53.9% directional -- the only above-noise signal across all models and horizons. This test checks whether XGB finds a stronger signal with 5.3x more data, and answers whether the learnability failure in v6 is LSTM-specific or dataset-wide.

### Method
Load the existing `BTC_features.joblib`, train a fresh XGBoost quantile ensemble on the same 80/20 time split, and report directional accuracy per horizon. No LSTM. No retraining of the quantile models from v3 -- we want a clean run on the extended data.

### What to do

Add a new CLI subcommand to `engines/intrabar_predictor.py`:

```python
# New subcommand: xgb-only-probe
p_xgp = subs.add_parser("xgb-only-probe", help="Train XGBoost quantile ensemble on existing features, no LSTM.")
p_xgp.add_argument("--asset", required=True)
```

In the corresponding command function `cmd_xgb_only_probe`:
1. Load `models/intrabar/{asset}_features.joblib`
2. Extract the feature rows and return targets (same objects the main `train` command uses)
3. Skip all LSTM training
4. Call the existing `_train_xgb_quantile_ensemble()` function (or whatever name the function has in the v3 code path) with the same train/test split
5. Report per-horizon directional accuracy and coverage for the XGB ensemble
6. Save XGB artifacts to a separate file: `models/intrabar/{asset}_xgb_probe.joblib` so it doesn't overwrite any LSTM-paired artifacts
7. Do NOT save any LSTM artifacts or touch `BTC_multi_horizon_lstm.pt`

### What NOT to do

- Do NOT train the LSTM at all
- Do NOT modify feature engineering
- Do NOT modify the XGBoost hyperparameters from v3 (same learning rate, tree depth, etc.) -- we want apples-to-apples with v3
- Do NOT run backtests -- this is a signal-presence probe, not a P&L test

### Acceptance Criteria for Diagnostic 2

- `python -m engines.intrabar_predictor xgb-only-probe --asset BTC` runs end-to-end
- Completes in <30 min (likely 5-15 min given XGB is fast)
- Reports directional accuracy per horizon (5 numbers, one per horizon)
- Reports p10-p90 coverage per horizon
- Artifact saved to `models/intrabar/BTC_xgb_probe.joblib`
- No LSTM artifacts modified

### Interpretation

Record the per-horizon directional accuracy:

- **Any horizon >= 53%**: there IS directional signal in the dataset. The LSTM failed to find it but XGB did. v8 direction: either fix LSTM architecture/training OR pivot to XGB-only (which has the advantage of being the only model to find signal in v3 too).
- **All horizons 50-52%**: marginal signal at best. Dataset is near-noise at these horizons. v8 should seriously consider abandoning 5-min directional prediction for something else (different timescale, different target, different asset).
- **All horizons <= 50%**: no signal. Complete hypothesis failure. v8 should pivot entirely (longer horizons, different asset, different prediction target like volatility instead of direction).

Compare explicitly to v3's XGB-5-bar 53.9%. If v6's XGB does worse than v3's on 5.3x more data, that's a significant negative finding -- more data should help, not hurt, if the signal exists.

---

## Execution Steps

### Step 1: Pre-flight
```powershell
tasklist | findstr python
```
Confirm no user-session Python. Kill any that exist.

### Step 2: Diagnostic 1 -- Thermal test

Create `scripts/diag_thermal_cpu.py` per sketch above.
AST parse + ASCII check.
Run:
```powershell
python scripts/diag_thermal_cpu.py
```
Report full per-iteration table and verdict.

**If verdict is "H3 CONFIRMED"**: proceed to Diagnostic 2 but flag for Chat.
**If verdict is "H3 RULED OUT"**: proceed to Diagnostic 2 but flag as new mystery for Chat.

### Step 3: Diagnostic 2 -- XGB-only probe

Add the `xgb-only-probe` subcommand to `engines/intrabar_predictor.py` (minimal additions, do not touch existing code paths).
AST parse + ASCII check.

**Progress reporting per new rules 9-15:**
- Before launch: state estimate
- T+5 min: first check, report XGB training phase (which horizon it's on, fold progress if applicable)
- T+10 min: second check
- Every ~5 min thereafter
- If exceeds 30 min: pause per kill switch

Run:
```powershell
python -m engines.intrabar_predictor xgb-only-probe --asset BTC
```

Report per-horizon directional accuracy, coverage, total wall-clock.

### Step 4: Interpretation

In the retro, explicitly answer:
- Diagnostic 1 verdict: H3 confirmed / partial / ruled out
- Diagnostic 2 verdict: any horizon >=53% / all 50-52% / all <=50%
- Combined interpretation: which of v8a/v8b/v8c/v8d does this point to
- Recommendation for v8

### Step 5: Retro

Overwrite `claude/retros/RETRO_praxis_intrabar_confluence.md` with v7 PARTIAL or COMPLETE status (COMPLETE if both diagnostics ran cleanly, PARTIAL if either blocked).

Preserve v5 + v6 content by reference ("folded into Section X of v7 retro" or similar -- do NOT lose the 2-stage failure analysis from v6).

Advance cycle counter ONLY if both diagnostics complete and a clear v8 direction is determined. If diagnostics are ambiguous (both partials, or new mysteries uncovered), cycle counter holds.

---

## Acceptance Criteria (whole Brief)

- [ ] `scripts/diag_thermal_cpu.py` created, runs, produces verdict
- [ ] `xgb-only-probe` subcommand added to `engines/intrabar_predictor.py`, runs, produces per-horizon metrics
- [ ] AST parse + ASCII check pass on all modified/new files
- [ ] Retro includes: thermal verdict, XGB per-horizon accuracy, combined v8 recommendation
- [ ] Progress reporting per new CLAUDE_CODE_RULES.md rules 9-15 on Diagnostic 2
- [ ] Single-process discipline held throughout
- [ ] Retro includes explicit v8 direction recommendation (v8a/v8b/v8c/v8d)

---

## What NOT to do

- Do NOT attempt any fix to the LSTM training code
- Do NOT rerun `build-features`
- Do NOT rerun LSTM training
- Do NOT run backtests
- Do NOT commit code (everything stays in working tree per series convention)
- Do NOT modify ETH anything

---

## Known Pitfalls

- **Thermal test N=8000 too small**: iteration 1 runs in <2 s. Bump to N=10000 and rerun. We need each iteration to take ~5-10 s to be comparable to LSTM epoch load.
- **Thermal test N=8000 too large**: iteration 1 takes >20 s. Drop to N=6000. Don't let total wall-clock exceed 10 min.
- **XGB-only subcommand clobbering existing artifacts**: save to `{asset}_xgb_probe.joblib` not `{asset}_quant_mh_models.joblib`. Preserve v3's proven artifacts (which we'd reuse if v8a pivots back to LSTM+XGB ensemble).
- **Feature file loading**: the features joblib from v5 build-features may have serialized shapes that assume the full `train` pipeline. If `xgb-only-probe` can't load just the tabular features without the LSTM sequences, that's a bug in the new subcommand -- fix by loading only the fields needed.
- **Thermal confound**: if the user's laptop is under other CPU load (browser, IDE builds) during the thermal test, results are noisy. Ask Jeff to close heavy apps before Diagnostic 1.

---

## Progress Check Cadence (per CLAUDE_CODE_RULES.md rules 9-15)

Mechanical cadence for this Brief:

- T+0 (session start): restate scope, estimated runtime, kill-switch values
- After Diagnostic 1 completes: report verdict + table
- At T+5 min into Diagnostic 2: first progress check on XGB training
- At T+10 min into Diagnostic 2: second progress check
- Every ~5 min thereafter
- Any unexpected state change (per-iter time jumps, error messages, memory growth): immediate out-of-cadence report

---

## References

- Retro v6 PARTIAL (current): `claude/retros/RETRO_praxis_intrabar_confluence.md`
- Retro v4 data extension: `claude/retros/RETRO_praxis_intrabar_data_extension.md`
- Brief v6: `claude/handoffs/BRIEF_praxis_intrabar_confluence_v6.md`
- Workflow modes doc: `claude/WORKFLOW_MODES_PRAXIS.md`
- Rules doc (v1.1 with new progress rules): `claude/CLAUDE_CODE_RULES.md`
- Existing features (reuse, don't rebuild): `models/intrabar/BTC_features.joblib`
- Existing XGB function (reuse, don't rewrite): `engines/intrabar_predictor.py` has `_train_xgb_quantile_ensemble` or equivalent; find and call it
