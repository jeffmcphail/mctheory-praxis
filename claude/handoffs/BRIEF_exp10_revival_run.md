# Cycle 36c -- Exp 10 end-to-end re-run with portfolio leverage cap

**Predecessor:** Cycle 36b (`a06b360` + retro `f58ec9b`). CPO_ALLOCATION.md
+ 7 allocation unit tests + dead-flag removal all in place.

**Mode:** Hybrid, multi-stage. Code runs the experiments; Claude reviews
the results before the atlas update lands. The cycle pauses between
"results produced" and "atlas updated" so the result_class change is
human-reviewed.

**Risk:** medium. End-to-end experimental run; phase2 is multi-hour and
non-resumable. The cycle includes early-stage sanity checks to surface
problems before the expensive part commits.

**Why this matters:** Exp 10 has lived in INCONCLUSIVE limbo since the
2026-03-26 original run. Cycle 36a established the prior "fixed" claim
was fabricated; Cycle 36b made the cap mechanism real and tested. Cycle
36c is the first cycle in the program that produces *new experimental
data*, not infrastructure. Result determines whether Exp 10 graduates
to PARTIAL or NEGATIVE.

## What this cycle delivers

Five deliverables in dependency order:

1. **Pre-flight grid timing** -- a tiny-subset phase2 run (single asset,
   single TA type) to project full-grid runtime before committing
   hours.
2. **Full phase2 + phase3** -- regenerates the cached training artifacts
   lost in the 2026-04-24 disk failure. One run; cached for all
   subsequent phase4 invocations.
3. **Four phase4 results** at `--max-leverage` settings 2.0, 1.0, 0.5,
   0.25. Per-cap subdirs in `outputs/exp10_revival/`.
4. **Heavyweight analysis**: equity curves, drawdown traces, per-asset
   breakdown, cap-vs-result response table.
5. **Atlas update** (gated on human review): Exp 10's result_class +
   result_summary + revival_hypotheses updated based on the actual
   numbers.

## Step ordering -- pause points are load-bearing

### Step 1: Pre-flight grid timing

Goal: project full-grid phase2 runtime before committing to the multi-hour
run. Code runs phase2 on a single (asset, TA_type) slice and times it.

Suggested invocation pattern (adapt to actual `run_cpo.py` flags):

```powershell
# Tiny subset: 1 asset, 1 TA type. Should complete in 1-3 minutes.
python scripts/run_cpo.py phase2 `
    --strategy crypto_ta `
    --start 2024-01-01 --end 2025-01-01 `
    --assets BTC `
    --ta-types STOCH `
    --output-dir outputs/exp10_revival/preflight `
    --verbose
```

If `run_cpo.py` doesn't accept `--assets` or `--ta-types` filters,
identify the smallest-scope equivalent (config grid restriction, time
window shortening, whatever lets us project the full-grid cost from a
known small fraction).

**Report:** wall-clock seconds + parquet row counts. Estimate full-grid
runtime as `(small_run_time / small_run_combos) * full_combos`. Full
grid is 8 assets × 8 TA types × 338 signal configs × 72 barrier configs
≈ 1.56M combos.

**Pause point 1:** if projected full-grid runtime is more than 6 hours
or less than 30 minutes, surface the projection to Claude before
proceeding. Both bounds are alarm bells -- too short means the grid
might not be what we think; too long means we need to discuss whether
to parallelize, shrink scope, or run overnight with a watchdog.

### Step 2: Full phase2

```powershell
python scripts/run_cpo.py phase2 `
    --strategy crypto_ta `
    --start 2024-01-01 --end 2026-03-25 `
    --output-dir outputs/exp10_revival/cpo `
    --verbose
```

(Adjust date range to match the original Exp 10's: training 2024,
OOS through 2026-03-25 per the atlas markdown.)

Important: this run produces `phase2_returns.parquet` and
`phase2_features.parquet` in the cpo output dir. These are big and
shared by all four phase4 invocations downstream.

**Sanity check after phase2:** read the parquet headers and report
- row count
- distinct (asset, ta_type, config_id) combinations -- should match
  expected total
- date range -- should span the training + OOS window
- non-null fraction of expected columns

If any are dramatically off, STOP and report. Don't proceed to phase3.

### Step 3: Phase3 training

```powershell
python scripts/run_cpo.py phase3 `
    --strategy crypto_ta `
    --output-dir outputs/exp10_revival/cpo `
    --verbose
```

Produces `phase3_models.joblib`.

**Sanity check after phase3:** load the joblib and report
- number of trained models (expected ~40 = 8 assets × 5 TA types kept)
- AUC range across models (expected 0.771-0.854 per the original atlas
  -- if dramatically lower, e.g. <0.6, the phase2 features are
  probably corrupted)
- base_rate range (expected 22-52%)

If AUC is in a different ballpark from the documented original, STOP and
report. The atlas baseline isn't the law of nature; data and TA configs
might have shifted since 2026-03-26. But we want to know about a big
divergence before we run four phase4 variants on a different
experimental foundation.

### Step 4: Four phase4 invocations

For each cap value in `[2.0, 1.0, 0.5, 0.25]`:

```powershell
python scripts/run_cpo.py phase4 `
    --strategy crypto_ta `
    --output-dir outputs/exp10_revival/cpo `
    --max-leverage <CAP> `
    --start 2025-01-01 --end 2026-03-25 `
    --output outputs/exp10_revival/cap_<CAP>/phase4_results.json `
    --verbose
```

(Adjust `--output` path to whatever phase4 actually accepts. If the
script auto-writes a fixed filename, plan around that; copy/rename
between runs.)

Each phase4 should produce, at minimum:
- Per-day portfolio P&L
- Per-day total gross exposure (to confirm cap is binding as expected)
- Cumulative return
- Sharpe
- Max drawdown
- Per-model contribution breakdown

**Sanity check after each phase4:**
- Total gross exposure max should equal min(max_leverage, N *
  max_weight_per_model) per day. For `--max-leverage 2.0`, expect
  consistent 175% (the runaway being reproduced). For
  `--max-leverage 0.5`, expect 50% capped.
- Cap=2.0 cumulative return should reproduce -83.78% +/- a tolerance
  (say, +/- 5%). If it doesn't, the experimental foundation is
  different from the original; flag it explicitly.

### Step 5: Heavyweight analysis

Build `outputs/exp10_revival/` with this structure:

```
outputs/exp10_revival/
├── cpo/
│   ├── phase2_returns.parquet
│   ├── phase2_features.parquet
│   └── phase3_models.joblib
├── cap_2.0/
│   ├── phase4_results.json
│   ├── equity_curve.png
│   ├── drawdown_trace.png
│   └── per_asset_breakdown.csv
├── cap_1.0/
│   └── (same shape)
├── cap_0.5/
│   └── (same shape)
├── cap_0.25/
│   └── (same shape)
├── SUMMARY.md
└── response_curve.png
```

`SUMMARY.md` content should include:
- Headline table: cap | cumulative_return | Sharpe | Max DD | gross_exposure_realized
- Reproduction check: cap=2.0 vs documented -83.78%
- Cap response analysis: how do returns / Sharpe change as cap tightens?
- Per-asset table at the canonical cap=0.5: which models contributed,
  which detracted
- Verdict candidate: based on the cap=0.5 result, propose the new
  result_class (PARTIAL if positive Sharpe, NEGATIVE if not, or
  INCONCLUSIVE only if results are bizarre)

Plot generation: use matplotlib; keep plots clean (no decorative
styling). Equity curve = cumulative return over time per cap setting;
drawdown trace = running max minus current; response curve = scatter of
cap vs cumulative return + cap vs Sharpe.

**Pause point 2:** After SUMMARY.md is ready and all four phase4 runs
complete, PAUSE. Report the full table + the SUMMARY.md content to
Claude. Do NOT update TRADING_ATLAS.md yet.

### Step 6: Atlas update (post-review)

Once Claude approves the verdict candidate, update Exp 10's TRADING_ATLAS.md
entry:

- **result_class**: update from INCONCLUSIVE to PARTIAL or NEGATIVE
  (whatever the cap=0.5 number supports)
- **Result row** in the attribute table: replace -83.78% / Sharpe -1.158
  headline with the cap=0.5 number, preserving the cap=2.0 figure as a
  reference baseline
- **Verdict line**: update "Verdict: INCONCLUSIVE — leverage construction
  failure masks signal quality. Re-run needed with portfolio leverage
  cap." to reflect what was actually found
- **Root cause section**: keep the original analysis (40-model gating,
  175% gross at defaults) but add a follow-up paragraph documenting
  Cycle 36c's findings
- **Revival hypotheses**: hypothesis #1 ("Hard portfolio leverage cap")
  should be marked as TESTED with the result; remaining hypotheses
  re-prioritized based on what cap=0.5 produced
- **Addendum** at line 915+ (the one we wrote in 36a as "NOT RUN"):
  update to "EXECUTED IN CYCLE 36c" with link to outputs/exp10_revival/

Run atlas_sync; expected 1 TRADING entry updated (Exp 10), 1 embedding
regenerated.

### Step 7: Commit + push

Single large commit covering:
- outputs/exp10_revival/ tree
- TRADING_ATLAS.md edits
- Brief + retro (after fill-in)

Use `git commit -F` for the message.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Pre-flight projection completed and reported before full phase2 launch |
| 2 | Full phase2 + phase3 ran cleanly; sanity-check outputs match expectations within tolerance (or divergences explicitly flagged) |
| 3 | Four phase4 results produced at caps 2.0, 1.0, 0.5, 0.25; per-day gross exposure confirms cap is binding as expected |
| 4 | cap=2.0 result reproduces -83.78% +/- 5% (or divergence is explicitly explained) |
| 5 | outputs/exp10_revival/ tree exists with all subdirs + plots + SUMMARY.md |
| 6 | SUMMARY.md includes the headline table, reproduction check, response analysis, and verdict candidate |
| 7 | PAUSE before atlas update; Claude reviews SUMMARY.md and approves verdict |
| 8 | TRADING_ATLAS.md Exp 10 updated with new result_class, headline numbers, verdict, addendum |
| 9 | atlas_sync reports 1 TRADING entry updated, 1 embedding regenerated; praxis:atlas_get(8) reflects the new result_class |
| 10 | Single commit + push; commit message via git commit -F |

## Out of scope

- Info bars revival (Cycle 37+; only relevant if 36c's cap=0.5 result
  is PARTIAL-positive and motivates further investigation).
- Other revival hypotheses (top-K filtering, etc.) -- the brief targets
  the leverage cap fix specifically.
- Refactoring run_cpo.py or compute_allocation -- 36b finalized those.
- LSTM v2 (Cycle 37+).
- Other atlas entries -- only Exp 10's content changes.

## Notes for Code

- The phase2 run is the long pole. Treat the pre-flight as a real
  decision point: if it projects badly, surface to Claude and we'll
  decide together whether to scope down or push through.
- The outputs/exp10_revival/ directory should NOT be in .gitignore --
  we want the artifacts committed as research evidence (Cycle 35's
  revival hypotheses + this cycle's measurements should be linkable
  from the atlas).
- Plot generation: matplotlib's default style is fine. Don't burn time
  on aesthetics; this is internal research deliverable.
- If any sanity check fails (steps 2-4), STOP and report. Don't push
  through "well it's close enough" -- the whole point of this cycle is
  reproducing a documented baseline and then varying it cleanly. If
  the baseline can't be reproduced, that itself is a finding worth
  understanding.
- The atlas update (Step 6) should be approved by Claude BEFORE
  application -- the result_class change is consequential and the
  verdict text should be calibrated against the actual numbers, not
  pre-written speculation.
- For the PAUSE point 2 reporting back to Claude, include the SUMMARY.md
  content verbatim, the headline table, and the verdict candidate. I'll
  approve, propose refinements to the verdict text, or in the unusual
  case the results are surprising, ask for more analysis before
  finalizing.

## Commit message template (fill before commit)

```
Cycle 36c: Exp 10 end-to-end re-run with portfolio leverage cap

Reproduces the original Exp 10 runaway baseline (-83.78% cum, Sharpe
-1.158 at --max-leverage 2.0) and produces four cap-setting comparisons
showing the response of cumulative return / Sharpe / drawdown to the
portfolio gross cap.

Headline result at --max-leverage 0.5 (the canonical revival cap):
<CAP_HALF_CUM_RETURN>% cum, Sharpe <SHARPE>, Max DD <DD>%.

Response curve summary:
- cap=2.0: <STATS> (reproduces atlas baseline within tolerance)
- cap=1.0: <STATS>
- cap=0.5: <STATS> (canonical revival)
- cap=0.25: <STATS> (too-tight reference)

Exp 10's atlas entry updated: result_class moves from INCONCLUSIVE to
<NEW_CLASS>. Verdict updated to <NEW_VERDICT_SHORT>. Per Cycle 35's
revival_hypotheses, hypothesis #1 (Hard portfolio leverage cap) is now
TESTED; <FOLLOWUP_HYPOTHESIS_NOTES>.

Outputs in outputs/exp10_revival/ include phase2/3 cached artifacts,
per-cap phase4 results + equity/drawdown plots + per-asset breakdown,
response curve plot, and SUMMARY.md with the full headline table.

atlas_sync: 1 TRADING updated (Exp 10), 1 embedding regenerated.
```
