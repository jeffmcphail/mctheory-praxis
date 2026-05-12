# CPO portfolio allocation arithmetic

## Overview

The CPO phase 4 stage transforms per-model RF probabilities into
portfolio position weights. This document explains the arithmetic
and the role of each CLI knob.

Source of truth: `engines/cpo_core.py::compute_allocation` (line 428).

## The two allocation modes

`compute_allocation` supports two modes, selected via the strategy
adapter or the `--alloc-mode` CLI flag:

- **equal_weight** (default): every model passing the probability
  gate receives an identical weight, capped at `max_weight_per_model`
  and additionally constrained so the total gross does not exceed
  `max_leverage`.
- **kelly**: Kelly-style scaling using each model's expected edge
  and (Ledoit-Wolf-shrunk) covariance, with correlation dedup as a
  pre-step. Caps total gross at `max_leverage` via a proportional
  scale-down idiom in `kelly_vector` (`cpo_core.py:423`).

Both modes use `max_leverage` as the binding portfolio gross cap.

## equal_weight arithmetic

For N surviving models (those whose RF probability exceeds
`prob_threshold`, or `base_rate + min_lift` if `min_lift > 0`):

    weight_per_model = min(max_leverage / N, max_weight_per_model)
    total_gross     = N * weight_per_model

The `min(...)` ensures that whichever constraint is tighter binds:

- When N is small enough that `max_leverage / N > max_weight_per_model`,
  the per-model cap binds. Total gross = `N * max_weight_per_model`,
  which is strictly below `max_leverage`.
- When N is large enough that `max_leverage / N < max_weight_per_model`,
  the leverage cap binds. Total gross = `max_leverage` exactly.

### Worked example: Exp 10 conditions

35 models pass the gate; defaults `max_leverage=2.0`,
`max_weight_per_model=0.05`:

    weight_per_model = min(2.0 / 35, 0.05) = min(0.0571, 0.05) = 0.05
    total_gross      = 35 * 0.05 = 1.75   (175% — RUNAWAY)

This is the per-model-cap-binds regime: `max_leverage / 35 = 0.0571`
exceeds the per-model ceiling of 0.05, so the leverage knob does
not actually constrain the portfolio.

With `--max-leverage 0.5` (the Cycle 36c fix):

    weight_per_model = min(0.5 / 35, 0.05) = min(0.0143, 0.05) = 0.0143
    total_gross      = 35 * 0.0143 = 0.5    (50% — CAPPED)

Now the leverage-cap term binds and total gross equals
`max_leverage` exactly.

## CLI knobs

- `--max-leverage` (float, default 2.0): canonical portfolio gross
  cap. Documented above.
- `--max-weight` (float, default 0.05): per-model cap
  (`max_weight_per_model` in the function signature). Primary use is
  to prevent any single model from dominating the portfolio when N
  is small.
- `--prob-threshold` (float, default 0.50): RF probability gate.
  Models below the threshold receive zero weight.
- `--min-lift` (float, default 0.0): lift-based gate alternative.
  When `> 0`, replaces `prob_threshold` with `base_rate + min_lift`
  on a per-model basis.

## Why there is no separate `--max-portfolio-weight` knob

Earlier revisions of `run_cpo.py` defined a `--max-portfolio-weight`
flag with the intent of providing a post-allocation proportional
scale-down. The flag was never wired through `cmd_phase4` →
`run_phase4` → `compute_allocation` in any revision (Cycle 36a
audit confirmed; see `claude/retros/RETRO_exp10_addendum_audit.md`)
and was removed in Cycle 36b.

The use case the flag was intended to address — capping aggregate
gross exposure when many models pass the gate — is already served
by `--max-leverage` via the `min(max_leverage / N,
max_weight_per_model)` arithmetic above. The two knobs would only
diverge in a case where you wanted to permit unequal per-model
weights to float up to a high ceiling while still capping the
total — a case that does not arise under either of the current
allocation modes, both of which derive position size from
`max_leverage`.

If a future allocation mode legitimately needs that distinction
(e.g., a regime-conditional mode where different models get
different ceilings), the post-allocation scale-down can be added
back as a distinct mechanism at that time. Until then, one knob is
the simpler mental model.

## Per-cycle history

- **Cycle 36a (2026-05-12)**: audited the false claim that Exp 10
  had been re-run with `--max-portfolio-weight 0.50`; established
  the flag was unwired. Atlas Addendum retracted in commit
  `ef889b0`.
- **Cycle 36b (2026-05-12)**: removed the unwired flag; documented
  `--max-leverage` as the canonical portfolio gross cap; added
  this doc; locked in the arithmetic via
  `engines/tests/test_compute_allocation.py`.
- **Cycle 36c (pending)**: re-run Exp 10 with `--max-leverage 0.5`
  to produce the canonical post-cap-fix portfolio result.
