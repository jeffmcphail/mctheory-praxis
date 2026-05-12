# Retro: Cycle 36a -- Exp 10 addendum audit + atlas correction

**Brief:** `claude/handoffs/BRIEF_exp10_addendum_audit.md`
**Date:** 2026-05-12
**Mode:** Hybrid (Claude drafted brief + reviewed proposed correction;
Code performed git archeology + filesystem inspection + atlas
correction + sync)
**Status:** DONE
**Predecessor:** Cycle 35.5 retro `22e4989` + Cycle 36 RECON report
**Commit:** `ef889b0`

---

## Summary

Resolves the atlas-vs-code discrepancy surfaced by the Cycle 36 RECON
report. Audit verdict: **(iii) Aspirational**, conclusive.

Atlas Addendum for Exp 10 (TRADING_ATLAS.md section "10. TA_STANDARD ×
CRYPTO (Triple Barrier Re-run)") corrected to a retraction: heading
changed to "Addendum: Experiment 10 -- Crypto TA leverage cap re-run
(NOT RUN)"; body documents the smoking-gun finding (atomic commit
`a2202a7` contains both "pending action: fix leverage cap" AND
"Re-ran Phase 4..." with arithmetic-derived numbers); explicit
retraction of the prior "CONFIRMED ❌ NO EDGE" framing; forward
pointers to Cycle 36b (wire the cap mechanism) and Cycle 36c (execute
the actual re-run).

No code changes in this cycle. The wiring decision (Cycle 36b) and
re-run (Cycle 36c) are shaped by this verdict.

Net change:
- `TRADING_ATLAS.md`: +34 / -15 (Exp 10 Addendum section only, lines
  915-948 area)
- atlas_sync: 0 added / 0 updated / 0 removed; 0 embeddings
  regenerated. The Addendum at line 915+ sits OUTSIDE Exp 10's
  parsed-block boundary (source_line_start=725, source_line_end=779
  per atlas_get(8)), so the DB never contained the fabricated
  numbers and the correction touches only orphaned markdown.

---

## Why this matters

The Cycle 36 RECON correctly flagged that the implementation work
shouldn't proceed until the atlas-vs-code drift was resolved. Two
risks if we'd skipped this:

1. Cycle 36b could have implemented `--max-portfolio-weight` and
   produced a number that "should have matched" the atlas (-27.95%)
   but didn't, leading to head-scratching about whether the
   implementation is wrong or the atlas was.
2. Cycle 36c re-run could have used `--max-leverage 0.5` and gotten
   a result with no clean comparison baseline.

The audit cycle costs ~1 hour and removes both risks. Pattern for
future cycles: when atlas-as-spec and code-as-truth diverge, audit
before implementing.

---

## Execution log

### Task A1: Git history of TRADING_ATLAS.md around the Addendum

The Addendum was added in a single atomic commit `a2202a7` (Fri
2026-04-03 17:58:39, "feat(praxis): DEX arb scanner... experiments
16-17"). `git log -S "max-portfolio-weight 0.50"`, `-S "Sharpe
-1.197"`, `-S "Addendum: Experiment 10"`, and `-S "Re-ran Phase 4"`
all converge on this single SHA. No retouches since. The commit
body discusses Exps 16/17 entirely; Exp 10 work appears only in the
TRADING_ATLAS.md diff, not in the commit message itself. No
co-located retro for the supposed re-run.

### Task A2: Git history of cpo_core.py / run_cpo.py

`git log --all -S "max_portfolio_weight" -- engines/cpo_core.py
scripts/run_cpo.py` returns zero results. Only the hyphenated CLI
flag string "max-portfolio-weight" hits, in `a2202a7` -- the
introducing commit. `git log --all -S "args.max_portfolio_weight"`
across the entire repo history also returns zero results. No
revision in history shows `max_portfolio_weight` in the function
signatures or call chain. **Definitively rules out verdict (ii)
regression** -- the wiring never existed.

### Task A3: Filesystem search for prior run artifacts

`find . -name "*.parquet" -not -path "./.venv/*"` empty; same for
`*.joblib`; `output/` directory does not exist. No surviving log
files (`cycle*.log`, `exp10*.log`, `ps_history.txt`). Consistent
with `RECOVERY_PLAN_post_disk_failure.md` noting `data/*.db` and
`phase3_models.joblib` were lost in the 2026-04-24 disk failure --
but the fabricated Addendum predates that failure by 3 weeks
(`a2202a7` is from 2026-04-03), so its absence from artifacts can't
be blamed on the failure.

### Task A4: Adjacent retro files

`ls claude/retros/` and `ls claude/handoffs/` filtered for
`exp.?10|leverage|triple|ta.crypto|crypto.ta` returns only this
cycle's own files (`BRIEF_exp10_recon.md`,
`BRIEF_exp10_addendum_audit.md`, `RETRO_exp10_addendum_audit.md`).
No prior brief/retro pair documents an Exp 10 leverage-cap re-run.
No command-line record exists in the repo.

### Task A5: Synthesized verdict

**VERDICT: (iii) Aspirational** -- The Addendum was written as a
creative pass -- including the proposed fix narrative, the
recommended re-run, and predicted-by-arithmetic outcome -- all
framed as a completed result in a single commit that simultaneously
lists the underlying fix as a pending action.

Supporting evidence:
- Same atomic commit (`a2202a7`) contains both "pending action: fix
  portfolio leverage cap" AND "Re-ran Phase 4..." -- mutually
  exclusive in time.
- `--max-portfolio-weight` wiring has never existed in any revision
  (zero results for `max_portfolio_weight` in `cpo_core.py` /
  `run_cpo.py` function signatures across all history).
- The -27.95% number is derivable by proportional arithmetic from
  the -83.78% headline (200%/50% = ~4×); the Addendum body
  explicitly justifies it as "expected from 50% vs 200% exposure."
- Sharpe -1.197 is -1.158 rounded; justified in-line by "Sharpe is
  scale-invariant."
- No retro pair, no surviving artifacts, no log evidencing the
  command.

### Task A6: Proposed atlas correction

Proposed text (applied verbatim after Claude approval):

```markdown
### Addendum: Experiment 10 -- Crypto TA leverage cap re-run (NOT RUN)

The earlier version of this Addendum (commit `a2202a7`, 2026-04-03)
claimed a re-run with `--max-portfolio-weight 0.50` producing
-27.95% / Sharpe -1.197. Cycle 36a audit determined this was
aspirational, not measured:

- The same commit lists "fix portfolio leverage cap in cpo_core.py
  (max_portfolio_weight param)" as a pending action -- both can't
  be true in one atomic diff.
- The `--max-portfolio-weight` flag exists at
  `scripts/run_cpo.py:302` but has never, in any revision, been
  wired through `cmd_phase4` -> `run_phase4` -> `compute_allocation`.
- The -27.95% figure is derivable by proportional arithmetic from
  the -83.78% headline (200% / 50% gross = ~4x loss reduction), and
  was presented in-line as "expected from 50% vs 200% exposure."
- The Sharpe -1.197 is the -1.158 figure rounded, justified in-line
  by "scale-invariance."
- No retro, log, or surviving artifact evidences an actual re-run;
  the 2026-04-24 disk failure post-dates this commit but does not
  explain the missing wiring (which never existed).

The canonical Exp 10 result therefore remains -83.78% / Sharpe -1.158
with the INCONCLUSIVE verdict (see Exp 10 main block above and atlas
DB entry id=8). The "CONFIRMED NO EDGE" framing in the prior
Addendum was contingent on the projected -28% re-run number and is
hereby retracted; the working conclusion stays "INCONCLUSIVE,
construction failure suspected; leverage-cap revival genuinely
pending."

Cycle 36b wires the cap mechanism (decision deferred there: thread
`--max-portfolio-weight` properly, or deprecate it in favor of the
existing `--max-leverage` knob which achieves the same arithmetic).
Cycle 36c executes the actual re-run.
```

Claude approved as refined: tighter footprint synthesizing Code's
full coverage with a 0-changes-to-parsed-block guarantee (since the
Addendum sits in orphaned markdown outside Exp 10's parsed block at
source_line_start/end 725/779).

### Step 5-7: Apply correction + atlas_sync

```
python -m engines.atlas_sync
```

Output:
```
--- Sync summary ---
  TRADING_ATLAS.md
    + 0 added
    ~ 0 updated
    = 15 unchanged
  PREDICTION_MARKET_ATLAS.md
    + 0 added
    ~ 0 updated
    = 20 unchanged
  REGIME_MATRIX.md
    = 12 regime classes (full replace)
    = 60 strategy relevance rows (full replace)

  Embeddings (voyage-3-lite): 0 regenerated, 35 skipped
```

Round-trip confirms the correction touched only orphaned markdown --
atlas DB unchanged. 35/35 entries preserved.

### Step 8: Commit + push

Commit `ef889b0` to master via `git commit -F` (heredoc-safe).

---

## Notes

### Lessons for future audit-style cycles

Three:

1. **When atlas-as-spec and code-as-truth diverge, audit before
   implementing.** The wrong move was to write the implementation
   brief directly; the right move (which the RECON cycle surfaced)
   was a one-hour audit cycle. The audit cost is dwarfed by the
   cost of debugging a "the new code doesn't reproduce the atlas
   number" investigation that would have followed.
2. **Atlas content that lives outside parsed-block boundaries
   doesn't get indexed.** Always check `source_line_start` /
   `source_line_end` before assuming an edit will affect the DB.
   Orphaned markdown reads like atlas content to humans but isn't
   in `atlas_experiments` -- and round-trip sync confirms that
   distinction.
3. **`git log -S 'specific phrase'`** is faster than grepping commit
   messages or eyeballing diffs when looking for when a specific
   string first appeared. Adding to the future-cycle archeology
   toolkit. Particularly powerful when combined: running -S on the
   atlas number AND on the CLI flag string AND on the function
   parameter name in parallel converges on the truth in seconds.

### Implications for Cycle 36b/36c design

Based on the verdict, the next cycles should:

- **Cycle 36b** (wire the cap): Wire `--max-portfolio-weight`
  properly through `cmd_phase4` → `run_phase4` →
  `compute_allocation`, OR deprecate the flag in favor of the
  existing `--max-leverage` knob (which achieves the same
  proportional scale-down arithmetic). Claude's current lean: wire
  it as a DISTINCT mechanism from `--max-leverage` (the latter is a
  hard ceiling via per-model math `min(max_leverage/n,
  max_weight_per_model)`; the former is a post-allocation
  scale-down applied to the resulting total). But this is the
  Cycle 36b design discussion, not Cycle 36a.
- **Cycle 36c** (re-run): Three phase4 invocations at the same
  trained models: (1) no cap (reproduce the -83.78% baseline);
  (2) `--max-portfolio-weight 0.5`; (3) `--max-leverage 0.5`.
  Comparison of (2) vs (3) tells us whether they're genuinely
  distinct mechanisms or duplicates. First end-to-end run is the
  expensive one (~hours for phase2 + phase3 from scratch since
  no joblib survives); subsequent cap-tuning iterations on the
  cached `phase3_models.joblib` are minutes.

### MCP verification (user)

User confirmed via `praxis:atlas_get(8)` that the parsed-block
content is unchanged: `result_class=INCONCLUSIVE` preserved;
`result_summary="-83.78%, Sharpe -1.158, Max DD -102.39%"`
preserved; `md_hash=71e2ca00d8e1ee8ceb29be0c8d08b09b56b973597b74824f09b881a4b41c92df`
identical to Cycle 35's value; `synced_at=2026-05-12T00:37:08`
(Cycle 35's sync timestamp; no re-sync occurred since the parsed
block didn't change); `source_line_start/end=725/779` (Addendum at
915+ confirmed outside this range). DB integrity confirmed.

---

## Open items / next cycle inputs

- **Cycle 36b**: Wire `--max-portfolio-weight` through
  `cmd_phase4` → `run_phase4` → `compute_allocation` (or decide to
  deprecate the flag in favor of `--max-leverage` and update
  documentation accordingly). Shape determined by this cycle's
  verdict.
- **Cycle 36c**: Re-run Exp 10 end-to-end (phase2 + phase3 + phase4)
  with the corrected cap mechanism. First run is the expensive one
  (~hours for phase2); subsequent cap-tuning iterations are cheap.
  Plan to capture all three flavors (no-cap baseline, `--max-portfolio-
  weight 0.5`, `--max-leverage 0.5`) in one trained-models run so
  Cycle 36b's wiring decision is empirically validated.
- **`_classify_result_token` fix is in place** (Cycle 35.5; commit
  `106b515`). Result rows with WEAK POSITIVE / STRONG NEGATIVE
  now classify correctly. No regression risk from the corrected
  Addendum on classifier behavior.
- **Audit-cycle pattern is now proven.** When future atlas-vs-code
  discrepancies surface, consider a dedicated audit cycle (read-only,
  ~1 hour) before designing the implementation brief. Cycle 36a is
  the template.
