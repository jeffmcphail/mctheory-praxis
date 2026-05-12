# Cycle 36b -- Deprecate --max-portfolio-weight, document --max-leverage as canonical portfolio cap

**Predecessor:** Cycle 36a (`ef889b0` + retro `4c9130e`); Cycle 36 RECON.

**Mode:** Hybrid. Claude drafts brief + docs; Code applies removals + adds
unit tests + writes commit message + commits.

**Risk:** low. Pure subtractive code change (remove unwired flag) +
documentation. Behavior of every existing run-cpo invocation is unchanged
because the removed flag was already a no-op. No DB changes. No
allocation arithmetic changes.

**Scope cap:** ~2 hours Code time. Most of the budget is the docs work;
the actual code removal is ~10 lines.

## What Cycle 36a established

Cycle 36a confirmed that `--max-portfolio-weight` was never wired through
`cmd_phase4` -> `run_phase4` -> `compute_allocation`. The CLI flag exists
at `scripts/run_cpo.py:302` with a docstring promising "Weights scaled
proportionally if total exceeds this cap", but `args.max_portfolio_weight`
is never read after parsing. Dead code.

Meanwhile, `--max-leverage` IS wired and already functions as a
proportional portfolio gross cap via the arithmetic at `cpo_core.py:468`:

```python
weight = min(max_leverage / n, max_weight_per_model)
```

For 35 models with `max_leverage=0.5`, `max_weight_per_model=0.05`:
`min(0.5/35, 0.05) = min(0.0143, 0.05) = 0.0143` per model, total =
`35 × 0.0143 = 0.5`. Total gross is capped at exactly `max_leverage`.

The recon also surfaced that `kelly_vector` mode at `cpo_core.py:423`
uses the same `max_leverage` cap mechanism via a proportional scale-down
idiom. So `max_leverage` is the canonical gross-cap mechanism across
BOTH allocation modes.

Conclusion: `--max-portfolio-weight` is redundant with `--max-leverage`
for the use case it was intended to address. Deprecating it removes a
misleading knob and simplifies the mental model. Per the design
discussion (Cycle 36b prep), Option A was selected.

## What this cycle does

Three deliverables, in order of dependency:

### Deliverable 1: Remove the unwired flag

**File:** `scripts/run_cpo.py`

Remove:
- The `--max-portfolio-weight` argparse line (around line 302 per recon
  report)
- Any reference to `args.max_portfolio_weight` in the file (recon
  confirmed there are zero; the flag is purely defined-and-discarded)

Add (in the same diff, replacing the deprecated flag's slot):
- A deprecation comment at the top of the argparse section noting that
  `--max-portfolio-weight` was removed in Cycle 36b after being unwired
  since introduction in commit a2202a7. This is for future readers who
  might find old shell history or notes referencing the flag.

Example comment:
```python
# Note: --max-portfolio-weight was removed in Cycle 36b (2026-05-12).
# The flag was unwired since its introduction (commit a2202a7); see
# docs/CPO_ALLOCATION.md and claude/retros/RETRO_exp10_addendum_audit.md.
# Use --max-leverage as the canonical portfolio gross cap.
```

### Deliverable 2: Clarify --max-leverage documentation

**File:** `scripts/run_cpo.py` (the argparse help text)

Current help text (unknown verbatim; Code will read it). Update to
something like:

```
--max-leverage FLOAT
    Maximum portfolio gross exposure across all surviving models
    (default: 2.0). With N models passing the gate, each gets weight
    = min(max_leverage/N, max_weight_per_model). When max_leverage/N
    is the binding constraint, total gross = max_leverage exactly,
    so this flag doubles as the canonical "portfolio gross cap"
    knob. See docs/CPO_ALLOCATION.md for the full arithmetic.
```

**File:** `engines/cpo_core.py`

Update the docstring on `compute_allocation` (and `run_phase4` if it
also docs leverage semantics) to be explicit about the cap behavior.
Specifically, the equal_weight branch should have a comment block
above the `weight = min(...)` line explaining the arithmetic and how
it relates to portfolio gross.

### Deliverable 3: Create docs/CPO_ALLOCATION.md

**File:** `docs/CPO_ALLOCATION.md` (new)

Sections (suggested; Code can refine prose):

```markdown
# CPO portfolio allocation arithmetic

## Overview

The CPO phase4 stage transforms per-model RF probabilities into
portfolio position weights. This document explains the arithmetic
and the role of each CLI knob.

## The two allocation modes

`compute_allocation` (engines/cpo_core.py) supports two modes,
selected via the strategy adapter or CLI:

- **equal_weight** (default): every model passing the probability
  gate receives an identical weight, capped at `max_weight_per_model`
  and additionally constrained so the total gross does not exceed
  `max_leverage`.
- **kelly_vector**: Kelly-style scaling using each model's expected
  edge and variance. Caps total gross at `max_leverage` via a
  proportional scale-down idiom.

Both modes use `max_leverage` as the binding portfolio gross cap.

## equal_weight arithmetic

For N surviving models (those whose RF probability exceeds
`prob_threshold`):

    weight_per_model = min(max_leverage / N, max_weight_per_model)
    total_gross     = N * weight_per_model

The `min(...)` ensures that whichever constraint is tighter binds:

- When N is small enough that `max_leverage/N > max_weight_per_model`,
  the per-model cap binds. Total gross = `N * max_weight_per_model`,
  which is below `max_leverage`.
- When N is large enough that `max_leverage/N < max_weight_per_model`,
  the leverage cap binds. Total gross = `max_leverage` exactly.

### Worked example (Exp 10 conditions)

35 models pass the gate, defaults `max_leverage=2.0`,
`max_weight_per_model=0.05`:

    weight_per_model = min(2.0/35, 0.05) = min(0.0571, 0.05) = 0.05
    total_gross     = 35 * 0.05 = 1.75 = 175% (RUNAWAY)

With `--max-leverage 0.5` (the Cycle 36c fix):

    weight_per_model = min(0.5/35, 0.05) = min(0.0143, 0.05) = 0.0143
    total_gross     = 35 * 0.0143 = 0.5 = 50% (CAPPED)

## CLI knobs

- `--max-leverage` (float, default 2.0): canonical portfolio gross
  cap. Documented above.
- `--max-weight-per-model` (float, default 0.05): per-model cap. Primary
  use is to prevent any single model from dominating the portfolio
  when N is small.
- `--prob-threshold` (float, default 0.50): RF probability gate.
  Models below the threshold receive zero weight.

## Why there is no separate `--max-portfolio-weight` knob

Earlier revisions of `run_cpo.py` defined a `--max-portfolio-weight`
flag with the intent of providing a post-allocation proportional
scale-down. The flag was never wired through `cmd_phase4` ->
`run_phase4` -> `compute_allocation` and was removed in Cycle 36b
after Cycle 36a's audit established it was dead code.

The use case the flag was intended to address (cap aggregate gross
exposure) is already served by `--max-leverage`. The two would only
diverge in a case where you wanted to permit unequal per-model
weights to float up to a high ceiling while still capping the total
- a case that does not arise under the current equal_weight or
kelly_vector implementations, both of which derive position size
from the same `max_leverage` knob.

If a future allocation mode legitimately needs that distinction
(e.g., a regime-conditional mode where different models get
different ceilings), the post-allocation scale-down can be added
back as a distinct mechanism at that time. Until then, one knob is
the simpler mental model.

## Per-Cycle history

- **Cycle 36a (2026-05-12)**: audited the false claim that Exp 10
  had been re-run with `--max-portfolio-weight 0.50`; established
  the flag was unwired.
- **Cycle 36b (2026-05-12)**: removed the unwired flag; documented
  `--max-leverage` as the canonical portfolio gross cap; added
  this doc.
- **Cycle 36c (pending)**: re-run Exp 10 with `--max-leverage 0.5`
  to produce the canonical post-cap-fix portfolio result.
```

### Deliverable 4: Unit test locking in the gross-cap behavior

**File:** `engines/tests/test_compute_allocation.py` (new)

Cover the arithmetic so a future refactor that subtly breaks the cap
behavior fails loudly. Suggested cases:

```python
# Test 1: per-model cap binds (small N)
# 5 models, max_leverage=2.0, max_weight_per_model=0.05
# Expected: each gets 0.05, total = 0.25 (below max_leverage)
assert weights.sum() == pytest.approx(0.25)
assert all(w == pytest.approx(0.05) for w in weights)

# Test 2: leverage cap binds (large N)
# 50 models, max_leverage=0.5, max_weight_per_model=0.05
# Expected: each gets 0.01, total = 0.5 exactly
assert weights.sum() == pytest.approx(0.5)
assert all(w == pytest.approx(0.01) for w in weights)

# Test 3: gate filters models below threshold
# 10 models, half above gate, half below
# Expected: 5 receive weight, 5 receive zero
# Total respects max_leverage
assert (weights > 0).sum() == 5
assert weights.sum() <= max_leverage + 1e-9

# Test 4: Exp 10 reproduction (175% runaway, no cap)
# 35 above gate, max_leverage=2.0, max_weight_per_model=0.05
# Expected: total = 1.75 (this IS the runaway being documented)
assert weights.sum() == pytest.approx(1.75)

# Test 5: Exp 10 fix (50% cap, --max-leverage 0.5)
# 35 above gate, max_leverage=0.5, max_weight_per_model=0.05
# Expected: total = 0.5 exactly
assert weights.sum() == pytest.approx(0.5)

# Test 6: zero gating models -> zero weights
# Expected: weights sum to 0
assert weights.sum() == 0

# Test 7: edge case n=1 (single model)
# 1 model, max_leverage=2.0, max_weight_per_model=0.05
# Expected: weight = 0.05 (per-model cap binds since 2.0/1 > 0.05)
assert weights[0] == pytest.approx(0.05)
```

Code should call `compute_allocation` directly with synthetic input
(probability arrays, threshold, max_leverage, max_weight_per_model).
No DB or RF model fixtures needed.

## Step ordering

1. Read current state:
   - `git show HEAD -- scripts/run_cpo.py | head -80` to see argparse
   - `git show HEAD -- engines/cpo_core.py | grep -n "max_leverage\|max_portfolio_weight\|max_weight_per_model"` to see all uses
   - Verify recon report's claim that `args.max_portfolio_weight` has
     zero reads outside of the argparse definition.

2. Apply Deliverable 1: remove the `--max-portfolio-weight` argparse
   block; add deprecation comment.

3. Apply Deliverable 2: update help text on `--max-leverage`; update
   docstring on `compute_allocation`.

4. Apply Deliverable 3: create `docs/CPO_ALLOCATION.md`. Code can use
   the suggested content as a starting point; refine prose for accuracy
   against the actual code.

5. Apply Deliverable 4: create `engines/tests/test_compute_allocation.py`
   with the 7 test cases above. Run pytest; all 7 must pass.

6. Round-trip sanity: run `python -m engines.atlas_sync` to confirm no
   atlas drift (expected: 0/0/0). Run the full
   `engines/tests/` pytest suite to confirm no regression in the
   existing 14 classifier tests plus the new 7 allocation tests = 21
   passes.

7. Commit + push. `git commit -F` for message safety.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | `--max-portfolio-weight` is removed from `scripts/run_cpo.py`; `grep -r "max-portfolio-weight\|args.max_portfolio_weight" .` returns zero hits |
| 2 | `--max-leverage` help text and `compute_allocation` docstring explicitly describe the cap-as-portfolio-gross-cap behavior |
| 3 | `docs/CPO_ALLOCATION.md` exists with at least the worked-example arithmetic + 175%-vs-50% comparison |
| 4 | `engines/tests/test_compute_allocation.py` contains 7+ test cases covering the arithmetic; pytest reports all passing |
| 5 | Running `engines/tests/` full suite reports 21 passed (14 classifier + 7 allocation) |
| 6 | `atlas_sync` round-trip: 0/0/0; no atlas drift |
| 7 | Single commit, signed off with `git commit -F` |

## Commit message

```
Cycle 36b: Deprecate --max-portfolio-weight, document --max-leverage as canonical portfolio cap

Removes the unwired --max-portfolio-weight CLI flag from
scripts/run_cpo.py (introduced in commit a2202a7 alongside the
fabricated Exp 10 Addendum that Cycle 36a retracted; never wired
through cmd_phase4 -> run_phase4 -> compute_allocation in any
revision). Documents --max-leverage as the canonical portfolio
gross-cap mechanism in both help text and a new
docs/CPO_ALLOCATION.md.

The deprecation is purely subtractive on the code side: removing
the dead flag changes the behavior of zero existing invocations
because the flag was a no-op. The intended use case (cap aggregate
gross exposure when N models pass the gate) is already served by
--max-leverage via the existing min(max_leverage/N,
max_weight_per_model) arithmetic at cpo_core.py:468. When the
max_leverage/N term binds (large N), total gross equals
max_leverage exactly.

Adds engines/tests/test_compute_allocation.py with 7 unit tests
locking in the gross-cap behavior across edge cases:
- per-model cap binds (small N)
- leverage cap binds (large N)
- gate filters models below threshold
- Exp 10 reproduction (175% runaway, no cap)
- Exp 10 fix (50% cap, --max-leverage 0.5)
- zero gating models
- n=1 edge case

Pytest reports 21 passed across engines/tests/ (14 classifier
from Cycle 35.5 + 7 new allocation).

atlas_sync round-trip: 0 added / 0 updated / 0 removed; 0
embeddings regenerated. No drift.

Cycle 36c can now run Exp 10 end-to-end with --max-leverage 0.5
producing the canonical post-cap-fix result. The 175%-runaway
baseline is reproducible via the no-cap defaults.

If a future allocation mode legitimately needs a post-allocation
scale-down distinct from max_leverage (e.g., regime-conditional
per-model ceilings), the flag can be reintroduced with proper
wiring. Until then, one knob is the simpler mental model.
```

## Out of scope

- Re-running Exp 10 (Cycle 36c).
- Other CPO refactoring (renaming `max_leverage` to
  `portfolio_gross_cap`, etc.) -- could be considered after 36c
  validates the current naming is the binding choice point.
- Wiring `--max-portfolio-weight` as a distinct mechanism -- explicitly
  rejected in the Cycle 36b design discussion (Option A vs B).
- Touching `engines/atlas_sync.py` or any atlas content -- Cycle 36a
  retracted the fabrication and Cycle 35 backfilled all 15 trading
  entries. Atlas is clean.

## Notes for Code

- The "deprecation comment" in run_cpo.py is for future readers, not
  for users. Users will see the absence of the flag and get the
  argparse error if they try to pass it; that's the right behavior.
- The unit test file goes in `engines/tests/` alongside Cycle 35.5's
  `test_atlas_classify.py`. The `__init__.py` was already created in
  that cycle, so no new package marker needed.
- If `compute_allocation`'s actual signature is different from what
  recon described, adapt the tests accordingly. The acceptance
  criteria are about behavior, not exact function calls.
- If you discover during reading that `--max-portfolio-weight` IS
  referenced anywhere other than its argparse definition (i.e.
  recon was wrong on this point), STOP and report. The cycle's whole
  premise is that the flag is dead code.
