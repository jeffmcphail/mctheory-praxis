# Cycle 36 RECON -- Exp 10 leverage cap revival, code-path mapping

**Predecessor:** Cycle 35 (atlas backfill); Cycle 35.5 (classifier fix).
**Mode:** Reconnaissance only. No code changes. No commits. Code reads
files, runs structured queries against the experiment artifacts, and
reports back with findings. Claude then drafts the implementation brief.

**Why split this into a recon phase:** The revival hypothesis is
"hard portfolio leverage cap (max gross 0.5) on Exp 10 TA crypto triple-
barrier rerun". To write a correct implementation brief I need answers
to specific scoped questions about where the existing code lives and
what re-running entails. Guessing leads to a wrong brief. Recon costs
~20 minutes of Code's time and saves a wrong-brief cycle.

## Background (Code can skip if already in context)

Exp 10 (TRADING_ATLAS.md entry id=8, atlas section "10. TA_STANDARD ×
CRYPTO (Triple Barrier Re-run)") ran 2026-03-26 with the following
shape:

- Universe: 8 crypto assets × 5 TA types kept (40 individual models)
- 40 models simultaneously pass `P > 0.50` gate every day
- Per-model cap: 5% weight
- Result: 35 × 5% = **175% gross daily leverage**
- 2025 OOS window was broadly corrective for crypto
- Portfolio result: -83.78% cumulative, Sharpe -1.158
- But individual models worked: ADA_STOCH +117%, BTC_STOCH +28.8%

The fix per the atlas: hard portfolio-level cap (e.g. max total weight
= 0.5) on top of the per-model cap. The constituent-model alpha is
real; the construction failed.

## Reconnaissance tasks

Code's job is to find and report on the following. **Do not change
anything.** This is read-only.

### Task R1: Locate the Exp 10 run script

Find the script that produced the 2026-03-26 Exp 10 results. It should
be somewhere under `scripts/` or `engines/` and probably has "triple
barrier" or "ta_crypto_tb" or similar in the filename. Report:

- Path to the script (relative to repo root)
- Approximate line count
- A 2-3 sentence summary of what it does end-to-end (training data
  load → RF train → portfolio simulation → output)

If multiple candidate scripts exist, list them and identify which
one produced the documented -83.78% result.

### Task R2: Locate the portfolio allocation code

The portfolio construction step is where the 175% gross leverage
emerged. Find the function(s) that takes per-model RF probabilities
and produces position weights. Report:

- File + line range for the allocation function(s)
- The variable names that hold per-model weight cap and gate threshold
- The exact arithmetic that produces gross daily exposure (e.g.
  `weights = (p > 0.5).astype(float) * 0.05` -- whatever the actual
  code is)
- Whether a portfolio-level cap variable or hook EXISTS but is
  unused, vs. needs to be added from scratch

### Task R3: Identify the test artifacts

Where do Exp 10's results currently live? Report:

- Output directory (e.g. `outputs/exp10/...` or wherever)
- File types produced (parquet? csv? sqlite?)
- The specific result file that backs the -83.78% / Sharpe -1.158
  claim
- A handful of representative rows (head of the daily portfolio
  P&L file, the per-model summary, whatever's clearly identifiable)

### Task R4: Training-data caching status

Critical for time budget: is the RF training output cached, or does
the script re-train each run? Report:

- Whether the trained RF models persist (pickle? joblib? sqlite?)
- If cached: re-running with a leverage cap change should be minutes,
  not hours (only portfolio construction recomputes)
- If not cached: re-run cost is dominated by full RF retraining, and
  any change probably means full retrain, which has implications for
  iteration speed

### Task R5: Pre-fix re-run estimate

Based on Tasks R1-R4, estimate:

- Time to re-run Exp 10 end-to-end with no changes (sanity check
  the documented -83.78% still reproduces): minutes/hours
- Time to re-run with leverage cap added: minutes/hours
- Disk space needed: small/medium/large
- Whether the re-run can be interrupted/resumed or is one-shot

### Task R6: Two leverage-cap implementation flavors

The brief acknowledges two implementation choices for the cap:

**Flavor A: Proportional scale-down.** All 35-40 models still
contribute, but weights renormalize so total ≤ 0.5.
```python
raw_weights = (probs > 0.50).astype(float) * 0.05  # per-model cap
total = raw_weights.sum()
if total > 0.5:
    raw_weights = raw_weights * (0.5 / total)
weights = raw_weights
```

**Flavor B: Top-K cutoff.** Only the top-K models by probability
get non-zero weight; rest sit at zero.
```python
# Keep top 10 models by P; each gets 5% cap; max gross = 50%
top_k_idx = probs.argsort()[-10:]
weights = np.zeros_like(probs)
weights[top_k_idx] = 0.05  # only those passing gate
```

Code's task: identify which flavor is the cleaner fit for the existing
code structure. If the existing code uses pandas with named columns,
one may be much easier than the other. Report:

- Which flavor is easier to drop in
- Whether both are feasible without major refactor
- Any unobvious gotchas (e.g. the existing cap might be applied at a
  different stage than expected)

### Task R7: Confirm no scope creep needed

Verify these are NOT required for the cycle:

- New data ingestion (info bars come later if anything)
- New RF training (we're re-using the cached training output if R4
  confirms caching)
- New regime ablation (Exp 10's regime status is "not measured" and
  stays that way)
- New TC model (4 bps round-trip stays)

If any of these IS required, surface it as a brief amendment.

## Report format

Reply with a structured report:

```
## R1: Run script
- Path: ...
- Lines: ...
- Summary: ...

## R2: Allocation code
- File:Line: ...
- Variables: ...
- Arithmetic: ...
- Hook status: ...

## R3: Result artifacts
- Directory: ...
- Files: ...
- Backing file: ...
- Sample rows: ...

## R4: Cache status
- Cached: yes/no
- If yes: where / how
- Re-run cost implication: ...

## R5: Time/space estimate
- No-change re-run: ...
- With leverage cap: ...
- Disk: ...
- Resumable: ...

## R6: Implementation flavor
- Recommendation: A or B
- Reason: ...
- Gotchas: ...

## R7: Scope sanity
- All four "out of scope" items confirmed out: yes/no
- If no: ...
```

## Step ordering

1. Read the brief.
2. Tasks R1 through R7 in order (later tasks build on earlier ones;
   R5 needs R4, R6 needs R2).
3. Reply with the structured report. No code changes. No commits.
4. Wait for Claude's implementation brief based on the report.

## Out of scope (recon-cycle specific)

- Running anything. The recon is metadata-only.
- Modifying anything. Read-only.
- Drafting the implementation brief. That's Claude's next step
  after Code's report.
