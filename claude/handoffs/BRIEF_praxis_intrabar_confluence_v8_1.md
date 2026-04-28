# Implementation Brief: Intrabar Confluence v8.1 (Triple Barrier Labeling)

**Series:** praxis
**Priority:** P0 (tests whether target labeling was half the problem)
**Mode:** B (model training + backtest)

**Estimated Scope:** M (45-90 min)
**Estimated Cost:** none (local CPU only)
**Estimated Data Volume:** reuses existing `BTC_features.joblib` (51,765 sequences, 180-day dataset)
**Kill switch:** if retrain exceeds 60 min wall-clock, pause for approval

---

## Context

V7 diagnostic confirmed the issue with our intrabar LSTM approach is not thermal throttling (H3 ruled out). XGB-only probe on balanced 180-day data showed only 53.04% best-horizon directional accuracy -- below v3's 53.9% from a narrower window. V3's signal didn't generalize.

Literature review of recent crypto LSTM papers (2024-2025) surfaced a consistent finding: **target labeling method may matter as much as feature engineering**. The Springer Financial Innovation paper (Feb 2025) on triple-barrier labeling for crypto trading explicitly argues that next-bar return prediction -- which is what all our v1-v7 work used -- is mathematically well-defined but tradeably weak. Triple-barrier labels (TP/SL/timeout) directly measure whether a trade setup would have been profitable, and in their tests substantially outperformed next-bar targets.

This Brief tests whether switching from next-bar quantile regression to triple-barrier classification -- using our EXISTING 25-feature set -- produces meaningful directional signal. If yes, labeling was a significant part of the problem and we continue tuning this axis. If no, target labeling isn't the fix and we focus all energy on v8.2 (microstructure features).

This is the cheapest way to test "labeling vs features" as the bottleneck.

---

## Objective

Add a triple-barrier-classification training path to `engines/intrabar_predictor.py` as a separate command, train on existing features, report classification accuracy and tradability metrics. Do NOT remove the existing quantile-regression path -- add alongside.

---

## Detailed Spec

### What Triple Barrier Labeling Is

Given a bar at time t, a "triple barrier" looks ahead N bars and assigns one of three labels:
- **+1 (UP):** price hit upper barrier (e.g., +ATR_mult * ATR) first
- **-1 (DOWN):** price hit lower barrier (e.g., -ATR_mult * ATR) first
- **0 (TIMEOUT):** neither barrier hit within N bars

This directly measures "would a trade taken at time t with these TP/SL levels have been a winner, loser, or time-out." It's a classification target, not a regression target.

Key parameters:
- **Lookforward horizon N (bars):** how long the trade would stay open
- **Barrier multiplier on ATR:** how wide the TP/SL are
- **Symmetric barriers:** TP distance == SL distance (keeps things simple)

### Step 1: Add Triple-Barrier Label Generation

Add a new function in `engines/intrabar_predictor.py`:

```python
def _compute_triple_barrier_labels(
    close_prices,      # 1D array of close prices (5-min bars)
    atr_series,        # 1D array of ATR at each bar
    lookforward=15,    # how many bars ahead to look
    atr_multiplier=1.5 # barrier distance = 1.5 * ATR
):
    """Compute triple-barrier labels for each bar.
    
    Returns array of {-1, 0, +1} same length as close_prices.
    The last `lookforward` bars return NaN (no future to check).
    """
    n = len(close_prices)
    labels = np.full(n, np.nan)
    
    for i in range(n - lookforward):
        entry_price = close_prices[i]
        barrier = atr_multiplier * atr_series[i]
        upper = entry_price + barrier
        lower = entry_price - barrier
        
        future_slice = close_prices[i+1:i+1+lookforward]
        future_highs = future_slice.max() if len(future_slice) else entry_price
        future_lows = future_slice.min() if len(future_slice) else entry_price
        
        # Find which barrier was hit first (simplification: use close prices only,
        # not intrabar highs/lows -- adjust later if needed)
        upper_hit_idx = np.where(future_slice >= upper)[0]
        lower_hit_idx = np.where(future_slice <= lower)[0]
        
        if len(upper_hit_idx) == 0 and len(lower_hit_idx) == 0:
            labels[i] = 0  # timeout
        elif len(upper_hit_idx) == 0:
            labels[i] = -1  # only lower hit
        elif len(lower_hit_idx) == 0:
            labels[i] = 1   # only upper hit
        else:
            # Both hit -- take whichever came first
            labels[i] = 1 if upper_hit_idx[0] < lower_hit_idx[0] else -1
    
    return labels
```

**Note on simplification:** this implementation uses close-price-only barrier detection. A more accurate implementation would use bar high/low to detect barrier touches. For v8.1 stick with close-only -- if results look promising, v8.1.1 can refine this.

### Step 2: Add ATR Computation (if not already present)

If `engines/intrabar_predictor.py` doesn't already compute ATR, add:

```python
def _compute_atr(high, low, close, period=14):
    """Average True Range, simple implementation."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(period).mean().to_numpy()
    return atr
```

### Step 3: New CLI Subcommand `train-classifier`

Add alongside the existing `train` command. Does NOT replace the existing LSTM quantile regression path -- coexists with it.

```python
p_tc = subs.add_parser("train-classifier", 
    help="Train triple-barrier classifier on existing features.")
p_tc.add_argument("--asset", required=True)
p_tc.add_argument("--lookforward", type=int, default=15, 
    help="Barrier lookforward in 5-min bars (default 15 = 75 min)")
p_tc.add_argument("--atr-mult", type=float, default=1.5,
    help="Barrier distance as ATR multiple (default 1.5)")
p_tc.add_argument("--model", choices=["lstm", "xgb", "mlp"], default="xgb",
    help="Classifier architecture (default XGB, simplest and fastest)")
```

### Step 4: Implement `cmd_train_classifier`

Key steps:
1. Load existing features from `models/intrabar/{asset}_features.joblib`
2. Reconstruct the close-price series and compute ATR
3. Compute triple-barrier labels using the passed lookforward + atr_mult
4. Drop sequences where label is NaN (the last `lookforward` bars)
5. Train/test split 80/20 time-sorted (same as existing train command)
6. Train the chosen classifier:
   - **XGB (default):** XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, objective='multi:softprob') on tabular features (not sequences)
   - **MLP:** simple 3-layer MLP on flattened last-bar features (not sequences)
   - **LSTM:** use existing IntrabarQuantileLSTM but replace output head with 3-class classification (new architecture branch)
7. Report:
   - Class distribution (how many UP / DOWN / TIMEOUT labels)
   - Test accuracy
   - Per-class precision/recall (some classes dominate, want to see all three)
   - Confusion matrix
   - **Tradability metric:** of predictions labeled UP, what fraction were actually UP? Of predictions labeled DOWN, what fraction were actually DOWN? (This is the thing we care about for trading.)

### Step 5: Saving Artifacts

Save to `models/intrabar/{asset}_classifier_{model}.joblib` to avoid clobbering existing LSTM quantile artifacts.

---

## Progress Reporting (per CLAUDE_CODE_RULES.md rules 9-15)

**MANDATORY progress cadence** -- Code's v5/v6 missed reports. This is the new protocol.

- **T+0:** state estimate, label class distribution after computing triple-barrier labels
- **T+5 min:** first progress check -- which phase (label computation / train/test split / classifier training / evaluation)
- **T+10 min:** second check, per-epoch or per-fold timing if applicable
- **Every 5 min thereafter** until completion
- **Any state change** (class imbalance surprise, training error, unexpected timing): out-of-cadence report
- **Kill switch at 60 min wall-clock**: pause and report

Use background execution with explicit logging. If XGB (default), training should complete in under 5 min -- the whole Brief should finish well inside 30 min. If MLP, similar. If LSTM, budget more time per the v6 Stage 2 unresolved slowdown.

---

## Acceptance Criteria

- [ ] `train-classifier` subcommand added, accepts all three model choices
- [ ] AST parse and ASCII check pass
- [ ] Triple-barrier label computation produces labels in {-1, 0, +1}
- [ ] Class distribution printed before training (sanity check -- should not be 99% timeout or 99% UP)
- [ ] Training completes on XGB in under 10 min
- [ ] Test accuracy, per-class precision/recall, confusion matrix all reported
- [ ] Tradability metric reported: precision on UP and DOWN classes specifically
- [ ] Artifact saved to `models/intrabar/BTC_classifier_xgb.joblib`
- [ ] Existing LSTM quantile-regression path (`train` command) still works -- run it briefly to confirm no regression

---

## What NOT to change

- Feature set (still 25 features from existing `BTC_features.joblib`)
- LSTM architecture (untouched -- the existing `train` path is preserved)
- Build-features (not rerun)
- Backtest code (untouched in this Brief; interpretation of classifier results happens in retro, not as backtest)
- ETH (still on hold)

---

## Interpretation Guide for Retro

Report all three of these in the retro:

**Case A: Triple barrier works on existing features**
- XGB classifier achieves >55% accuracy OR UP-class precision >55% OR DOWN-class precision >55%
- Conclusion: target labeling was a meaningful part of the problem. Proceed with microstructure features (v8.2) AND triple-barrier labeling -- they stack.

**Case B: Triple barrier helps but modestly**
- XGB classifier accuracy 53-55%, similar to prior next-bar XGB probe
- Conclusion: labeling isn't the primary fix. v8.2 microstructure features are the bet.

**Case C: Triple barrier shows no improvement or worse**
- Accuracy at or below 52%, or class distribution is pathological (e.g., 95% timeout)
- Conclusion: labeling is not the issue -- features are the bottleneck. Drop labeling changes, pour energy into v8.2.

**Case D: Tradability clean even if overall accuracy is mediocre**
- Accuracy might be 52% overall BUT UP-class precision is 58% AND DOWN-class precision is 57%
- This is actually the most tradeable outcome! Even a classifier that mostly says "timeout" (the majority class) can be useful if when it does say UP/DOWN, it's right.
- Conclusion: highly actionable. Flag for Chat. Next step is filtering trades to only fire on high-confidence UP/DOWN predictions.

---

## Known Pitfalls

- **Class imbalance toward timeout.** With lookforward=15 and atr_mult=1.5, most bars may show timeout. If TIMEOUT class is >70%, consider lowering atr_mult to 1.0 or lookforward to 10. Report the distribution before judging results.
- **ATR scale mismatch with 5-min bars.** ATR(14) on 5-min bars is naturally smaller than ATR(14) on daily. The atr_multiplier needs calibration. 1.5 is a guess; if it produces 99% timeout, tune down.
- **Don't use intrabar high/low for barrier detection in v8.1.** Close-price-only is a known simplification. Matches "would I have gotten filled at TP/SL if I was only watching closes" which is conservative/realistic for non-HFT.
- **Leak risk:** do NOT use look-ahead info in features. The features are already computed correctly in the existing `BTC_features.joblib`, so just use them as-is.
- **Model choice:** default XGB because it's fastest. If results are promising, a v8.1.1 could try MLP or LSTM with the same labels.

---

## References

- Triple-barrier paper: Financial Innovation, Feb 2025 -- "Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning"
- Related: Lopez de Prado, "Advances in Financial Machine Learning" (this triple-barrier approach is from his book, widely adopted)
- v7 retro (XGB baseline at 53.04% with next-bar labels): `claude/retros/RETRO_praxis_intrabar_confluence.md`
- Existing features: `models/intrabar/BTC_features.joblib` (reuse, don't rebuild)
- Workflow modes: `claude/WORKFLOW_MODES_PRAXIS.md`
- Progress rules: `claude/CLAUDE_CODE_RULES.md` rules 9-15
