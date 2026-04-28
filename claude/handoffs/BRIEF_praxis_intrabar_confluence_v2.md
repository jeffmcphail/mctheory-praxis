# Implementation Brief: Intrabar Confluence v2 (5-min Bars)

**Series:** praxis
**Priority:** P0 (unblock testing of predict/confluence/backtest)
**Estimated Scope:** S (Small -- 30 min to 1 hour, mostly tuning the existing file)
**Date:** 2026-04-21
**Follow-up to:** RETRO_praxis_intrabar_confluence.md

---

## Context

The v1 intrabar predictor was built successfully (1408 lines, all 5 commands implemented, ASCII/AST/import clean). However, training on 48K 1-minute sequences proved infeasible on CPU within Claude Code's shell timeout budget. Epoch 1 alone took 76 seconds uncontested and ballooned to 306 seconds when two training runs collided on the same CPU.

More importantly: reviewing Jeff's original intent, the target is **"5-min vs 15-min mean-reversion where moves are large enough to trade."** On 1-minute bars, the model predicts 1-5 minute moves that are often smaller than transaction costs (0.06% 1-bar std vs 0.1% Binance taker fee). On 5-minute bars, the same horizon structure `[1, 3, 5, 10, 15]` becomes **5/15/25/50/75 minutes forward** -- which is actually the tradeable range Jeff described.

Decision: resample `ohlcv_1m` to 5-minute bars before training. This gives us a 5x reduction in sequence count (48K -> ~9.6K), per-epoch time drops from 76s to ~15s, full training completes in ~15-25 minutes. And it targets the correct timescale for the strategy.

---

## Objective

Modify `engines/intrabar_predictor.py` to aggregate 1-minute OHLCV data into 5-minute bars during data loading, then re-run the full pipeline end-to-end (build-features -> train -> confluence).

---

## Detailed Spec

### Minimal Changes to `engines/intrabar_predictor.py`

**1. Add a BAR_SIZE constant** at the top of the config section:
```python
BAR_SIZE_MINUTES = 5    # Aggregate 1-min data to this size before training
```

**2. Modify `load_1m_data()` to resample.** Rename to `load_intrabar_data()` and add aggregation:

```python
def load_intrabar_data(asset, bar_minutes=BAR_SIZE_MINUTES):
    """Load 1-min OHLCV and aggregate to `bar_minutes` bars.

    Aggregation rules:
      - open  = first open of the group
      - high  = max high of the group
      - low   = min low of the group
      - close = last close of the group
      - volume = sum of volumes
      - timestamp/datetime = first timestamp of the group (bar start)

    Drops any group with fewer than `bar_minutes` rows (incomplete bars).
    """
    conn = sqlite3.connect(CRYPTO_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, datetime, open, high, low, close, volume
        FROM ohlcv_1m
        WHERE asset = ? AND volume > 0
        ORDER BY timestamp ASC
    """, (asset,)).fetchall()
    conn.close()

    if bar_minutes == 1:
        return [dict(r) for r in rows]

    # Aggregate into bar_minutes groups
    bar_seconds = bar_minutes * 60
    aggregated = []
    current_group = []
    current_bar_start = None

    for r in rows:
        ts = r["timestamp"]
        bar_start = (ts // bar_seconds) * bar_seconds

        if current_bar_start is None:
            current_bar_start = bar_start
            current_group = [r]
        elif bar_start == current_bar_start:
            current_group.append(r)
        else:
            # Emit the completed bar (only if full)
            if len(current_group) == bar_minutes:
                aggregated.append(_aggregate_bars(current_group))
            current_bar_start = bar_start
            current_group = [r]

    # Last group
    if len(current_group) == bar_minutes:
        aggregated.append(_aggregate_bars(current_group))

    print(f"  Aggregated {len(rows)} 1-min bars into {len(aggregated)} {bar_minutes}-min bars")
    return aggregated


def _aggregate_bars(group):
    """Aggregate a list of 1-min bar rows into a single larger bar."""
    return {
        "timestamp": group[0]["timestamp"],
        "datetime": group[0]["datetime"],
        "open": group[0]["open"],
        "high": max(r["high"] for r in group),
        "low": min(r["low"] for r in group),
        "close": group[-1]["close"],
        "volume": sum(r["volume"] for r in group),
    }
```

**3. Update all call sites** of the old `load_1m_data()` function to `load_intrabar_data()`.

**4. Update the module docstring** and CLI help text to clarify the bar size:
- At top: `"Operates on N-minute OHLCV bars (default 5) aggregated from 1-min data."`
- In `cmd_build_features`: print the effective bar size prominently

**5. Update `MULTI_HORIZONS` comment** to clarify:
```python
# At 5-min bars, horizons translate to: 5, 15, 25, 50, 75 minutes forward
MULTI_HORIZONS = [1, 3, 5, 10, 15]
```

**6. Delete stale artifacts** so the rebuild is clean:
```python
# At the start of cmd_build_features:
for old_file in MODEL_DIR.glob(f"{asset}_*"):
    old_file.unlink()
    print(f"  Removed stale: {old_file.name}")
```

### Keep Everything Else Unchanged
- Model architecture: same `IntrabarQuantileLSTM`, 5 input channels
- Training loop, loss function, early stopping: unchanged
- Confluence logic: unchanged (z-score + dual-horizon + Hurst)
- Backtest logic: unchanged
- CLI command names: unchanged

---

## Acceptance Criteria

- [ ] `python -m engines.intrabar_predictor build-features --asset BTC` runs in under 2 minutes
- [ ] Output shows aggregation: "Aggregated 48265 1-min bars into ~9653 5-min bars"
- [ ] Feature file saved to `models/intrabar/BTC_features.joblib` with ~9600 sequences
- [ ] `python -m engines.intrabar_predictor train --asset BTC` completes in under 30 minutes wall-clock
- [ ] Reports directional accuracy per horizon (expected: 48-55% at 5-bar, declining at 15-bar)
- [ ] Reports p10-p90 coverage (expected: 70-85%)
- [ ] `python -m engines.intrabar_predictor confluence --asset BTC` runs end-to-end and reports current z-score + multi-horizon predictions
- [ ] `python -m engines.intrabar_predictor backtest --asset BTC` produces signal count, hit rate, equity curve
- [ ] File still passes AST parse and ASCII check
- [ ] Run ONLY ONE training process at a time -- verify with `tasklist | findstr python` before launching

---

## Training Safeguards (critical -- v1 had two concurrent runs)

Before calling `python -m engines.intrabar_predictor train`, Claude Code MUST:

1. Run `tasklist | findstr python` and confirm no user-session Python processes are active (Session 0 services are fine)
2. If any are found, kill them first
3. After launching training, do NOT launch a second train command under any circumstances
4. If the first train run appears stalled (no epoch progress for 10+ minutes), investigate the log file BEFORE starting a new run
5. Use `run_in_background: true` with a longer monitor window (target 30-45 minutes)

---

## Known Pitfalls

- **Incomplete bars at boundaries:** gaps in 1-min data (exchange downtime) will produce groups with fewer than 5 rows. The code drops these -- that's correct behavior, but means the aggregated bar count may be slightly less than `1m_count / 5`.
- **ORDER BY matters:** the 1-min rows MUST be in ascending timestamp order before grouping, otherwise the bar boundaries break.
- **Old features cache:** `models/intrabar/BTC_features.joblib` from the v1 run contains 1-min data. Delete before rebuild (step 6 above).
- **Old model artifacts:** any `BTC_multi_horizon_lstm.pt` or `BTC_multi_horizon.joblib` from v1 must also go.
- **Hurst at 5-min bars:** the 30-bar Hurst now looks at 30 * 5 = 150 minutes of data. This is actually better signal than 30 minutes of 1-min bars.
- **Z-score window:** 60-bar z-score now covers 5 hours instead of 1 hour -- more meaningful mean-reversion window.

---

## Open Questions (OK to decide during implementation, document in retro)

1. Should ETH features be rebuilt too? (Recommendation: only if BTC end-to-end test passes first -- don't waste compute otherwise)
2. If training still takes >30 minutes, should we cap max_epochs at 50? (Recommendation: try full 150 first with early stopping; if it doesn't converge, note in retro and Chat will decide)

---

## Expected Output

After successful run:
```
models/intrabar/
+-- BTC_features.joblib              (features, ~9600 sequences)
+-- BTC_multi_horizon.joblib         (training metadata)
+-- BTC_multi_horizon_lstm.pt        (PyTorch model state)
+-- BTC_quant_mh_models.joblib       (XGBoost quantile models)
```

Confluence output should look like:
```
CONFLUENCE SIGNALS -- BTC
Z-score threshold: +/-2.0
Short horizon: 1 bar (5 min)
Long horizon: 15 bar (75 min)
Current price: $71,XXX.XX
Z-score (60-bar): +0.XX
Hurst (60-bar): 0.XXX (trending/mean-reverting/random)

Multi-horizon predictions:
Horizon       p10      p25      p50      p75      p90  Direction
  1 bar    -0.2%    -0.1%    +0.0%    +0.1%    +0.3%   UP/DOWN
  3 bar    -0.3%    -0.1%    +0.1%    +0.2%    +0.5%   UP/DOWN
  5 bar    -0.5%    -0.2%    +0.1%    +0.4%    +0.8%   UP/DOWN
 10 bar    -0.8%    -0.3%    +0.2%    +0.6%    +1.2%   UP/DOWN
 15 bar    -1.1%    -0.4%    +0.2%    +0.8%    +1.6%   UP/DOWN

CONFLUENCE ANALYSIS
Z-score within threshold -> no signal
   OR
Z-score exceeds threshold -> [LONG/SHORT] candidate
  Short-term (5-min) ... Long-term (75-min) ...
  Confluence: [STRONG/MODERATE/NONE]
```

---

## References

- Brief v1: `BRIEF_praxis_intrabar_confluence.md`
- Retro v1: `RETRO_praxis_intrabar_confluence.md`
- File to modify: `engines/intrabar_predictor.py` (1408 lines, all 5 commands already implemented)
- Data: `data/crypto_data.db` -> `ohlcv_1m` table (96K rows across BTC+ETH)
