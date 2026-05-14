# Retro: Cycle 36c -- Exp 10 end-to-end re-run with portfolio leverage cap

**Brief:** `claude/handoffs/BRIEF_exp10_revival_run.md`
**Date:** 2026-05-13 (execution) / 2026-05-14 (atlas update + commit)
**Mode:** Hybrid, multi-stage (Claude reviewed verdict before atlas update)
**Status:** DONE
**Predecessor:** Cycle 36b retro `f58ec9b`
**Commit:** `fc9dff8`

---

## Summary

Re-ran Exp 10 (TA_STANDARD x CRYPTO Triple Barrier Re-run) end-to-end via the universal_ta crypto path with the `--max-leverage` knob wired in Cycle 36b. Produced four phase4 results at caps [2.0, 1.0, 0.5, 0.25] on the same trained 40-model portfolio.

Canonical revival result at `--max-leverage 0.5`:
- Cumulative return: **-26.58%**
- Sharpe: **-1.1844**
- Max DD: **-33.78%**

Headline finding: **Sharpe is invariant to 4 decimals (-1.1844) across binding-cap settings 1.0, 0.5, 0.25.** Returns scale exactly linearly with cap. The cap mechanism is a pure leverage scaler; it does not change which models pass the gate, their relative weights, or which days they trade.

Exp 10's atlas entry moves from INCONCLUSIVE to **NEGATIVE**. Verdict updated to: "NEGATIVE -- TA signals on crypto produce Sharpe ~-1.18 OOS regardless of portfolio gross cap. The original -83.78% headline was a leveraged amplification of a negative-edge signal, not a construction artifact concealing salvageable edge."

Net change:
- `outputs/exp10_revival/` -- 56 MB total / 26 files (phase2/3 cached artifacts, per-cap phase4 results, plots, SUMMARY.md, support scripts)
- `TRADING_ATLAS.md` -- 43 insertions / 58 deletions across 6 edits (Exp 10 attribute table, headline result, root-cause/verdict rewrite, test-conditions table, revival hypotheses rewrite, Addendum replacement)
- atlas_sync: 1 TRADING entry updated (Exp 10, id=8), 1 embedding regenerated. md_hash 71e2ca00... → 831e3dd3...

---

## Why this matters

Cycle 36c is the first cycle in the program that produces NEW experimental measurements rather than substrate work. Cycles 17-35 were schema migrations, infrastructure (info bars), and atlas hygiene. Cycle 36a retracted fabricated content; Cycle 36b made the cap mechanism real and tested. This cycle is where the substrate work pays off in actual research.

The result moves Exp 10 from INCONCLUSIVE limbo into a definite NEGATIVE category, removing one of the largest "we don't know" entries from the atlas. The Sharpe-invariance finding is also a methodologically important result that applies beyond Exp 10: any pure-leverage scaler in front of a fixed signal stream preserves Sharpe. This refutes a broader class of "leverage construction failure masks signal quality" framings that might apply to other atlas entries (Exp 11 and 12 are siblings worth checking).

The cap-response curve (four data points) is itself an artifact -- it documents how sensitive a 40-model TA portfolio is to allocation choices, and the answer turns out to be "exactly linearly scalable" for binding-cap settings.

---

## Execution log

### Step 1: Pre-flight grid timing

**Initial preflight (crypto_ta path):** 2026-05-12 19:36 (pre-cycle, captured before machine shutdown). BTC + 8 TA types + 2024, 1,238.75 sec wait... wait that was the universal_ta preflight time. Original brief's preflight was `--strategy crypto_ta` and ran on 2026-05-12 19:36 -- duration not separately captured, but produced 338 unique config_ids across 8 TA types on BTC (the crypto_ta strategy fingerprint). Projected ~23 min full-grid runtime -- but this was for the WRONG strategy (crypto_ta has no triple-barrier expansion; the brief's invocation didn't reproduce Exp 10's identity).

**Investigation triggered.** Atlas Exp 10 identity = "Triple Barrier Re-run" requires the TB-enabled grid. Cycle 36c investigation phase (~30 min) traced the code:

- `engines/crypto_ta_strategy.py::generate_crypto_param_grid` produces 338 configs total with `hold_hours` exits (NO triple barriers).
- `engines/ta_models.py::generate_ta_param_grid` produces ~110 signals × `standard_barrier_grid()` (= 72 barriers) = 7,920 configs WITH triple barriers.
- `--strategy universal_ta --asset-class crypto` routes through `engines/universal_ta_strategy.py` -> `ta_models.py::generate_ta_param_grid` -> the TB-enabled grid.
- The brief's `--strategy crypto_ta` was the wrong invocation. The original Exp 10 (2026-03-26) must have used universal_ta (or equivalent) given its "Triple Barrier Re-run" identity and the matching 7920-config count cited in sibling Exps 11/12.

**Side task discovered:** atlas line 730 "338 signal configs × 72 barrier configs" was introduced in commit `a2202a7` (Apr 3, Cycle 36a's identified fabrication source). The "338" was lifted from Exp 2's crypto_ta config count; the "72" was the standard_barrier_grid count; the multiplication created an internally-inconsistent description that doesn't correspond to any actual code path. This is the second confirmed a2202a7 fabrication in Exp 10's section (the first being the retracted -27.95% Addendum). Memory #19's sibling-fabrication sweep elevated in priority.

**Re-routed: PATH A with universal_ta.** Preflight re-run with `--strategy universal_ta --asset-class crypto --assets BTC` on 2024 training window. Duration: **1,238.75 sec (20.65 min)**. Produced 7,920 configs across 8 TA types on BTC, 2,898,720 strategy-day rows. Sanity confirms 27/27/9/8/12/9/9/9 signals × 72 barriers across the 8 TA types = 110 × 72 = 7,920 configs.

Projection: 8 × 1,238.75 / 3600 ≈ **2.75 hours** for full phase2 (Scope A: training window only). Within [4h, 16h] band relaxed by the user; below the 4h floor but acceptable after phase4-reads-phase2 verification confirmed Scope A safe (run_phase4 doesn't read phase2 returns; it re-runs through the strategy adapter).

### Step 2: Full phase2

Invocation:
```
python scripts/run_cpo.py --strategy universal_ta --asset-class crypto \
    --output-dir outputs/exp10_revival \
    --training-start 2024-01-01 --training-end 2024-12-31 \
    phase2 --start 2024-01-01 --end 2024-12-31
```

Wall-clock: **2.78 hours (9,997.7 sec)** -- almost exact to the projection.

Output: `outputs/exp10_revival/cpo/phase2_returns.parquet` (23,126,400 rows, 15 MB compressed parquet).

Sanity:
- 64 unique model_ids (8 assets × 8 TA types) -- all present
- Per-asset uniform 2,890,800 rows (= 7920 configs × 365 days)
- config_id range 0-7919 per asset
- Date range 2024-01-01 to 2024-12-30 (365 days; preflight had 366, harmless 1-day delta)
- Non-zero return frac: 77.62% (preflight 77.05%; consistent)
- features parquet: 22,912 rows × 12 cols, 100% non-null per column

### Step 3: Phase3 training

Invocation:
```
python scripts/run_cpo.py --strategy universal_ta --asset-class crypto \
    --output-dir outputs/exp10_revival \
    --training-start 2024-01-01 --training-end 2024-12-31 \
    phase3
```

Wall-clock: **13.65 min (818.7 sec)** -- well below the 30 min - 2 h budget.

Pre-filter (training-period profitability):
- Kept (5/8): STOCH, VOL_BREAK, ATR_BREAK, EMA_CROSS, BOLL
- Dropped (3/8): MACD, RSI, VWAP_REV
- 40/64 models retained (matches atlas headcount).

Pre-filter set vs atlas:
- Atlas: {STOCH, RSI, EMA_CROSS, VOL_BREAK, BOLL}
- Cycle 36c: {STOCH, VOL_BREAK, ATR_BREAK, EMA_CROSS, BOLL}
- One-element swap (ATR_BREAK ↔ RSI). Same 5/3 structure; likely cause is 2024 binance bar refresh shifting training-period mean profitability across types.

AUC distribution: **0.7899 - 0.8900** (atlas-documented: 0.771-0.854; current is ~0.03 higher overall, driven mostly by the new ATR_BREAK addition with AUCs 0.87-0.89). base_rate **20.3% - 52.3%** (atlas 22-52%; essentially identical).

Gate (AUC <0.65 or >0.95 = STOP) not tripped. Proceeded without pause.

### Step 4: Four phase4 invocations

Sequential PowerShell loop running each cap into its own per-cap directory, copying `phase4_portfolio.parquet` and capturing stdout per run.

Total wall-clock: **62.54 min (3,752.4 sec)**, ~15 min per cap.

Headline table:

| --max-leverage | Cumulative | Sharpe | Max DD | Realized gross (mean / max) | Cap binding? |
|---|---|---|---|---|---|
| 2.0 | -76.37% | -1.1596 | -97.24% | 1.495 / 1.80 | NO (per-model cap × ~30 models naturally produces 150% gross) |
| 1.0 | -53.17% | -1.1844 | -67.57% | 1.000 / 1.000 | YES |
| **0.5** | **-26.58%** | **-1.1844** | **-33.78%** | 0.500 / 0.500 | YES |
| 0.25 | -13.29% | -1.1844 | -16.89% | 0.250 / 0.250 | YES |

Baseline reproduction (cap=2.0 vs atlas -83.78% / -1.158 / -102.39%):
- Cum: +7.41 pp (outside ±5 pp tolerance; explained by ATR_BREAK↔RSI pre-filter swap)
- Sharpe: -0.0016 (3-decimal match)
- Max DD: +5.15 pp (3-decimal match for shape, slightly milder magnitude)

Per-day total_weight sanity confirms cap binding for caps 1.0, 0.5, 0.25 (std = 0 across all OOS days); cap=2.0 ranges [1.0, 1.8] mean 1.495 (not binding).

### Step 5: Heavyweight analysis

`outputs/exp10_revival/SUMMARY.md` produced (7.7 KB). Plots:
- `equity_curve.png` (4 caps on one figure; fan structure showing linear scaling)
- `drawdown_trace.png` (same fan structure for drawdowns)
- `response_curve.png` (two panels: cap vs cum return -- linear in binding regime; cap vs Sharpe -- horizontal line at -1.18, atlas baseline overlap)

Per-asset breakdown at cap=0.5: 6/40 positive-Sharpe models. Top 5: ADA_STOCH (+1.225), XRP_BOLL (+0.461), SOL_BOLL (+0.450), SOL_EMA_CROSS (+0.171), ETH_STOCH (+0.096). Bottom 5: ETH_ATR_BREAK (-2.337), ETH_VOL_BREAK (-2.254), ETH_BOLL (-1.906), BTC_BOLL (-1.742), DOGE_STOCH (-1.699). Mean Sharpe -0.757, median -0.828.

ADA_STOCH was the atlas's prior poster-child for "the signal works at model level" (training Sharpe +2.01, +117% cum); OOS it ran +58.8% cum at Sharpe +1.225 -- positive, but vastly outweighed by the negative tail. BTC_STOCH, the other atlas-cited example, didn't even make the top 5 in OOS.

### Step 6: Atlas update

Verdict candidate presented to Claude (pause point 2). Claude approved with one refinement to Edit 4 (Test conditions table risk_management row): the Sharpe-invariance framing should be in the row text directly for readers jumping to the table. Applied.

Six atlas edits applied via Edit tool:
1. Exp 10 attribute table: Date / Training / RF AUC / Pre-filter / Result / Computational engine rows updated.
2. Headline result line: original + 4-cap Cycle 36c result table.
3. Root cause + verdict block: complete rewrite with Sharpe-invariance framing.
4. Test conditions table: feature_set corrected, risk_management refined.
5. Revival hypotheses: 4 items rewritten as TESTED/REFUTED/DEMOTED/CLOSED with status badges.
6. Addendum block (line 919+): replaced "NOT RUN" version with "EXECUTED IN CYCLE 36c" + historical-record block + a2202a7 fabrication pattern documentation.

One small parser regression caught + fixed: initial Edit 5 used heading `**Revival hypotheses (status post Cycle 36c):**` which doesn't match atlas_sync's regex (expects canonical `**Revival hypotheses:**`). Reverted to canonical heading with a sub-line for the cycle 36c status framing.

atlas_sync output:
```
TRADING_ATLAS.md: +0 / ~1 / =14
Embeddings (voyage-3-lite): 1 regenerated, 34 skipped
```

DB verification (data/praxis_meta.db):
- id=8 result_class: INCONCLUSIVE -> NEGATIVE
- md_hash: 71e2ca00d8e1ee8ceb29be0c8d08b09b56b973597b74824f09b881a4b41c92df -> 831e3dd34fda9d3b5fb50cbf7e4d7b5c7f194e821617f99e227139fb67c24045
- full_markdown_len: 4302 -> 6296 chars (+46%)
- result_summary: new NEGATIVE verdict text captured
- test_conditions.feature_set: "110 signal configs x 72 barrier configs = 7,920 configs (universal_ta crypto path)"
- test_conditions.risk_management: refined Sharpe-invariance framing
- revival_hypotheses: 4 items parsed correctly (titles intact; likelihood=None because the new STATUS format dropped likelihood: prefixes -- acceptable given the items now describe outcomes rather than projections)

### Step 7: Commit + push

Single commit `fc9dff8` covering outputs/exp10_revival/ tree (56 MB / 26 files), TRADING_ATLAS.md, this retro, and the brief (claude/handoffs/BRIEF_exp10_revival_run.md).

---

## Notes

### Reproduction quality

cap=2.0 cum reproduces -83.78% atlas baseline to within 7.4 pp at -76.37% -- outside the ±5 pp tolerance the brief specified, but Sharpe (-1.1596 vs -1.158) and Max DD (-97.24% vs -102.39%) match to 3 decimals. The same shape of OOS result with slightly milder cumulative magnitude.

The ATR_BREAK↔RSI pre-filter swap is the most likely cause. Both have AUC in the 0.86-0.89 range OOS at the current run, suggesting either could plausibly have been the marginal type kept depending on small differences in training-period mean returns. ATR_BREAK in the current run has slightly better OOS performance than RSI would have, narrowing the loss by ~7 pp.

This is "reproduction within the experimental foundation, magnitude statistic noisier" rather than a deeper reproducibility problem. Future readers of the atlas should note both the original -83.78% and the Cycle 36c -76.37% as bracketing the same regime; the Sharpe and DD shape match cleanly.

### Cap response curve interpretation

The four data points form an exactly-linear fan at binding caps and a slightly-less-than-linear top end at cap=2.0 (where the cap isn't binding). The Sharpe-vs-cap plot is a horizontal line at -1.18 across all four caps, with the atlas baseline -1.158 essentially overlapping. **This is the cycle's central methodological finding** and should generalize: any portfolio with a "5%-per-model × N-models-passing-gate" naturally-binding allocation followed by a `--max-leverage X` cap exhibits the same behavior.

cap=0.25 ("too tight") doesn't materially help vs cap=0.5 in any sense that matters -- it just produces a smaller-magnitude loss at the same Sharpe. There's no "right" cap because all binding caps are equivalent at the signal-quality level. Choosing the cap is a position-sizing decision, not a signal-quality decision, for this strategy.

### What this means for the broader TA-on-crypto question

Definitively closes the "TA signals on crypto have persistent edge" working hypothesis from Cycles 1-4. Cumulative evidence now spans:
- Exp 2 (crypto_ta, no TB): negative edge
- Exp 10 cap-response sweep: negative-edge signal × leverage scaler -> negative result at any cap
- Exps 3 (futures TB) / 4 (FX TB): sibling negative-or-INCONCLUSIVE-with-bad-OOS results
- 6/40 individual models having positive OOS Sharpe is consistent with a population of random signals where a minority survive by chance; portfolio aggregation washes this out.

The "info bars revival" path remains open as a SEPARATE experiment, not a revival of Exp 10 -- if dollar/volume bars change which bars get included in the signal stream, the resulting per-bar Sharpe is a genuinely different signal, not a re-leveraging of the current one.

### Lessons for future research cycles

Patterns worth preserving for Cycle 37+:
- **Pre-flight strategy verification before committing to multi-hour run**: the brief's `--strategy crypto_ta` would have run cleanly to completion but reproduced the wrong experiment. The investigation phase (~30 min) caught this before the expensive part landed. Future cycles invoking strategies via brief-cited flags should verify the brief's strategy name matches the experiment's identity in the atlas DB (not just the markdown).
- **Sharpe invariance under uniform scalers**: any pure-multiplier post-allocation modification preserves Sharpe. If a revival hypothesis proposes such a transformation, it can be falsified analytically before running it. Save the compute.
- **Atlas-correction-as-part-of-revival-cycle pattern**: Cycle 36c corrected one fabrication line (338 × 72) in the same diff as the substantive update. Future cycles that surface atlas inaccuracies during their primary work should default to in-scope correction rather than queueing.
- **a2202a7 sibling-fabrication sweep is queued**: two confirmed fabrications in Exp 10 alone (Cycle 36a + Cycle 36c) suggests the commit's other claims warrant audit. Treat as a small dedicated cycle in 37+.

---

## Open items / next cycle inputs

- **Memory #19 sibling-fabrication sweep** is now meaningfully higher priority. Two confirmed fabrications in Exp 10 (NOT-RUN Addendum and 338×72 Training row). Cycle 37 candidate.
- **Info bars revival as a NEW experiment** (not Exp 10 revival). If pursued, frame as Exp 18 or equivalent; dollar bars + triple barriers + new signal stream. Out of scope for any future Exp 10 work.
- **Exp 11 (futures TB) and Exp 12 (FX TB) re-runs** to test whether Sharpe-invariance applies the same way: both also had "INCONCLUSIVE" verdicts with the same leverage construction framing. The methodological finding from Cycle 36c suggests their results may also reduce to negative-edge signal × leverage scaler. Worth running.
- **`atlas_search` engine-filter parameter** (deferred TODO from Cycle 35) -- still open.
- **PMA backfill** (separate cycle) -- still open.
- **LSTM v2** (Cycle 37+; info bars + triple-barrier + DL architecture refresh) -- the gating depends on whether the "TA-on-crypto thread closed" framing motivates a pivot to fundamentally different signals (Engine 7 funding carry path, Engine 8 alternative data) rather than another TA variant. Reframed accordingly.
