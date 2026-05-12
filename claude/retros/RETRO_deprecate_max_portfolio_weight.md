# Retro: Cycle 36b -- Deprecate --max-portfolio-weight, document --max-leverage as canonical portfolio cap

**Brief:** `claude/handoffs/BRIEF_deprecate_max_portfolio_weight.md`
**Date:** 2026-05-12
**Mode:** Hybrid (Claude drafted brief + CPO_ALLOCATION.md scaffold;
Code applied removal + added unit tests + refined docs + committed)
**Status:** DONE
**Predecessor:** Cycle 36a retro `4c9130e`
**Commit:** `a06b360`

---

## Summary

Removes the unwired `--max-portfolio-weight` CLI flag from
`scripts/run_cpo.py` and documents `--max-leverage` as the canonical
portfolio gross-cap mechanism via:

1. Help-text clarification in run_cpo.py argparse
2. Docstring update on `compute_allocation` in `engines/cpo_core.py`
3. New `docs/CPO_ALLOCATION.md` with the worked arithmetic
4. `engines/tests/test_compute_allocation.py` locking in the gross-cap
   behavior across 7 edge cases

The deprecation is purely subtractive on the code side. The removed
flag was a no-op since its introduction in commit `a2202a7`
(Cycle 36a established this); removing it changes the behavior of
zero existing invocations.

Net change (per `git show a06b360 --stat`):
- `scripts/run_cpo.py`: +11 / -5 (removal of argparse block + deprecation comment + expanded `--max-leverage` help text)
- `engines/cpo_core.py`: +15 / -0 (docstring "Portfolio gross cap" section + inline comment block above the equal_weight `min(...)` line)
- `docs/CPO_ALLOCATION.md`: 113 lines (new file)
- `engines/tests/test_compute_allocation.py`: 148 lines (new file, 7 test functions)

---

## Why this matters

After Cycle 36a, `--max-portfolio-weight` existed in the CLI but did
nothing. That state is worse than no flag: it implies semantics that
don't hold and would mislead anyone reading old shell history or
notes referencing the flag.

Documenting `--max-leverage` as the canonical portfolio gross cap
gives future readers (Cycles 36c, 37+, and beyond) a clear answer
to "which knob caps total gross exposure?" without having to read
the allocation arithmetic line-by-line.

The unit tests serve a second purpose beyond regression protection:
they document the arithmetic in executable form. A reader who wants
to know how the cap behaves at N=1 or N=50 can read the test cases
and immediately understand the binding-constraint logic.

---

## Execution log

### Step 1: Read current state

- `--max-portfolio-weight` argparse block was at
  `scripts/run_cpo.py:302-305` (4 lines: declaration + 3-line help
  string promising proportional scale-down).
- `grep` over `*.py` confirmed exactly one hit (the dead argparse
  line); **zero reads** of `args.max_portfolio_weight` anywhere.
  Recon premise stood; no escalation needed.
- Existing `--max-leverage` help text: none beyond
  `type=float, default=2.0` (no help string was previously set;
  just the argparse default).
- `compute_allocation` docstring was a short mode-summary at
  `engines/cpo_core.py:446-449` with a two-line "Modes:" block; no
  portfolio-cap discussion.

### Step 2-3: Removal + help/docstring update

- **Removed:** `--max-portfolio-weight` argparse block (4 lines).
- **Added:** 4-line deprecation comment in its place, pointing to
  `docs/CPO_ALLOCATION.md` and `claude/retros/RETRO_exp10_addendum_audit.md`.
- **Expanded:** `--max-leverage` now has a 7-line help string
  explaining the `min(max_leverage/N, max_weight_per_model)`
  arithmetic and identifying it as the canonical portfolio gross-cap
  knob.
- **Extended:** `compute_allocation` docstring gained a 10-line
  "Portfolio gross cap (both modes)" section describing the cap
  semantics for both `equal_weight` and `kelly` modes.
- **Added:** 5-line inline comment block above the equal_weight
  `min(...)` line explaining the cap derivation in-code (so a reader
  scrolling to the allocation arithmetic gets the explanation
  without having to scroll back up to the docstring).

### Step 4: docs/CPO_ALLOCATION.md

Brief's scaffold prose was accurate against real code:

- Default values match exactly (`max_leverage=2.0`,
  `max_weight_per_model=0.05`).
- Mode names match: `equal_weight`, `kelly`. (Brief's scaffold used
  "kelly_vector" in one spot — that's the internal function name
  inside kelly mode, not the mode name surfaced via CLI. Refined to
  use `kelly` consistently for the mode and reserve `kelly_vector`
  for the implementation reference.)
- Function line number verified (`compute_allocation` at
  `engines/cpo_core.py:428`).
- The "175% runaway" worked example matches actual Exp 10 conditions
  (N=35, defaults bind the per-model cap at 0.05, total = 1.75).
- The "50% capped" worked example matches the Cycle 36c target
  (N=35, `max_leverage=0.5` makes the leverage term bind at 0.0143
  per model, total = 0.5 exactly).

Per-cycle history section added: 36a (audit), 36b (this cycle),
36c (pending re-run). Provides a forward-pointing anchor so future
readers can trace why the doc exists and what's next.

### Step 5: Unit tests

```powershell
pytest engines/tests/test_compute_allocation.py -v
```

Output (relevant portion):
```
engines/tests/test_compute_allocation.py::test_per_model_cap_binds_small_n PASSED
engines/tests/test_compute_allocation.py::test_leverage_cap_binds_large_n PASSED
engines/tests/test_compute_allocation.py::test_gate_filters_models_below_threshold PASSED
engines/tests/test_compute_allocation.py::test_exp10_runaway_reproduction PASSED
engines/tests/test_compute_allocation.py::test_exp10_fix_with_max_leverage_half PASSED
engines/tests/test_compute_allocation.py::test_zero_gating_models_returns_empty PASSED
engines/tests/test_compute_allocation.py::test_single_model_per_model_cap_binds PASSED
```

7 tests passed.

Edge cases that surfaced during test authoring:

- **`compute_allocation` returns only above-gate models in its dict**
  (not all-models-with-zeros-for-below-gate). The brief's draft Test 3
  said "5 receive weight, 5 receive zero", but the actual contract is
  "5 receive a weight entry; the 5 below are absent from the returned
  dict." Tests reflect actual behavior; the executable docs are now
  authoritative on this point.
- **Zero-gating returns `{}`** (empty dict), not a dict of zeros.
  Test 6 locks this in.
- **The Exp 10 runaway is the per-model-cap-binds regime**, not the
  leverage-cap-binds regime — at N=35 with defaults,
  `max_leverage/N = 0.0571 > 0.05 = max_weight_per_model`, so the
  per-model 0.05 binds and total = 35 × 0.05 = 1.75. The leverage
  cap kicks in only when the user lowers `--max-leverage`, which
  flips which term is the smaller. The test reproduction (1.75
  exactly) confirms the framing.
- **Signature note** (helpful for future test authors): inputs are
  `list[dict]` with `model_id`, `p_profitable`, `expected_return`,
  `base_rate` keys; outputs are `dict[str, float]`. A small `_preds`
  helper at the top of the test file builds the synthetic input.

### Step 6: Full test suite sanity

```powershell
pytest engines/tests/ -v
```

Output (final line):
```
============================= 21 passed in 0.89s ==============================
```

21 passed (14 classifier from Cycle 35.5 + 7 allocation from this cycle).

atlas_sync round-trip:
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

0/0/0 confirmed across both atlases; 0 embeddings regenerated.
MCP-side verification (`praxis:atlas_get(8)`) confirmed
md_hash `71e2ca0...` and `synced_at 2026-05-12T00:37:08`
preserved verbatim from Cycle 35; source_line boundaries 725/779
unchanged. Atlas DB integrity contract held.

### Step 7: Commit + push

Commit `a06b360` to master via `git commit -F` (heredoc-safe).

---

## Notes

### Why we picked Option A over B

Recap of the Cycle 36b design discussion:

- Option A: deprecate `--max-portfolio-weight`, document `--max-leverage`
  as canonical gross cap.
- Option B: wire `--max-portfolio-weight` as a distinct post-allocation
  scale-down mechanism, keep both knobs.
- Option Hybrid: deprecate but add a clarifying alias.

We picked A because:

1. `max_leverage` already functions as a portfolio gross cap via the
   `min(max_leverage/N, max_weight_per_model)` arithmetic in both
   equal_weight and kelly modes. The two would only diverge in cases
   that don't arise under current allocation modes.
2. Dead code is worse than no code; the unwired flag implied semantics
   that didn't hold.
3. If a future allocation mode legitimately needs the distinction,
   the post-allocation scale-down can be added back as a distinct
   mechanism then. Reversible decision; cheap to revisit.

### CPO_ALLOCATION.md as a documentation anchor

This is the first dedicated doc on CPO mechanics in the repo (per
Cycle 32's cleanup, anyway). It establishes a pattern for future
docs: arithmetic + worked examples + CLI knob reference + per-cycle
history of changes. If other CPO mechanics warrant their own docs
(e.g., RF training discipline, gate threshold tuning), they can
follow this template.

### Unit tests as executable documentation

The 7 test cases in `test_compute_allocation.py` cover the
arithmetic edge cases more clearly than prose could. A future reader
who wants to know "what happens at N=1?" can read the test directly.
Adopting this pattern: when a CPO mechanic is documented in
CPO_ALLOCATION.md, the corresponding edge cases should be locked in
as tests.

---

## Open items / next cycle inputs

- **Cycle 36c**: end-to-end Exp 10 re-run. With `--max-leverage 0.5`
  the cap should bind at 50% gross. The first phase2 run is the
  expensive one (~hours; no joblib cache survives the 2026-04-24
  disk failure). Subsequent phase4-only iterations using the cached
  `phase3_models.joblib` are minutes.
- **Cycle 36c can now produce three comparable results in one
  training run:**
  1. `--max-leverage 2.0` (no effective cap, reproduces -83.78%
     baseline)
  2. `--max-leverage 0.5` (50% gross cap, the canonical fix)
  3. Optional: `--max-leverage 1.0` for an intermediate data point
- **Memory entry #19** (a2202a7 fabrication-risk note) is still
  open; consider a small audit cycle to sweep that commit for other
  similar issues. Deferred but not closed.
- **Memory entry #20** (don't infer Jeff's work duration from
  conversation elapsed time) is now in effect; Claude should not
  reference session length without asking.
- **`atlas_search` engine-filter parameter** (deferred TODO from
  Cycle 35); now genuinely useful but small priority.
- **PMA backfill** (separate cycle; PMA structure differs from
  TRADING).
- **LSTM v2** (Cycle 37+; info bars + triple-barrier + DL
  architecture refresh).
