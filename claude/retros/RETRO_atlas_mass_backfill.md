# Retro: Cycle 35 -- Atlas mass backfill: remaining 13 trading experiments

**Brief:** `claude/handoffs/BRIEF_atlas_mass_backfill.md`
**Date:** 2026-05-12
**Mode:** Hybrid (Claude drafted brief + per-experiment revival hypotheses
calibrated to each experiment's failure mode; Code applied markdown
edits + ran atlas_sync + verified)
**Status:** DONE
**Predecessor:** Cycle 34 (`647d50c` + retro `2f0d2ef`, Info Bars v0.1)
**Commit:** `4d3d328`

---

## Summary

Brings the 13 trading experiments not covered by Cycle 33's design
validation (Exps 1 and 13) to the same structured shape:
`computational_engine`, `test_conditions`, `revival_hypotheses`,
`regime_state_at_test`.

Plus three small ancillary cleanups bundled with the backfill:

1. **Exp 1 regime-key cleanup** -- split the combined-letter
   `B, C, F, H, J, K` bullet into 6 canonical single-letter bullets
   so `regime_state_at_test` keys collapse to clean
   `B_<class_name>` etc. instead of sanitized fallback
   `b,_c,_f,_h,_j,_k`.

2. **Exp 13 regime-key cleanup** -- rewrite regime bullets in
   canonical `F (Funding/positioning)` form; F-state details moved
   from bullet key to description text. Key becomes
   `F_fundingpositioning` instead of
   `f_=_+1,_+2_(positive_funding_sustained)`.

3. **Result-row fixes** for Exps 3, 4, 7, 8, 11 -- markdowns clearly
   stated verdicts but lacked explicit `**Result**` rows in the
   attribute table; added so atlas_sync's classifier captures
   result_class correctly.

After Cycle 35, all 15 TRADING_ATLAS experiments are fully backfilled
along the Cycle 33 structured schema, with strict-canonical regime
keys and complete result_class population.

Net change: +1498 / -8 lines in `TRADING_ATLAS.md` (plus the brief
itself; no other files touched).

---

## Why this matters

Cycles 33 + 33.5 shipped the schema and 2 example backfills. Cycle 34
shipped Info Bars v0.1 -- the substrate that lets revival hypotheses
reference dollar bars / volume bars / VIB at specific thresholds
concretely (instead of as abstract possibility).

Cycle 35 makes the Atlas DB queryable along all the dimensions that
matter for Cycle 36+ revival re-run selection: filter by engine, find
experiments whose revival_hypotheses reference info bars, surface the
highest-likelihood revival candidates by structural query rather than
semantic search alone.

---

## Per-experiment revival hypothesis design notes

(Captured here for future reference; the full content is in the
brief and in TRADING_ATLAS.md post-cycle.)

The revival_hypotheses field shape varies by result_class:

- **NEGATIVE** experiments: hypotheses focus on "what would change
  the math of why this didn't work?" Three shapes commonly recur:
  (a) substrate change (info bars), (b) overlay (carry / regime
  filter), (c) accept-the-verdict.
- **PARTIAL** experiments: hypotheses focus on "what would push
  this over the line?" -- usually a mix of (a) substrate change,
  (b) cross-period / cross-asset validation, (c) labeling-or-exit
  refinement.
- **INCONCLUSIVE** experiments: hypotheses focus on "what's the
  missing capability or data?" -- (a) data infrastructure
  (Exp 15 VRP), (b) construction fix (Exps 10 + 12 leverage
  cap), (c) larger sample (Exps 11 + 12 OOS extension).

Specific high-likelihood revivals identified by the backfill:

| Experiment | Highest-likelihood revival | Likelihood |
|---|---|---|
| Exp 10 (TA crypto triple-barrier) | Hard portfolio leverage cap (max gross 0.5) | very high |
| Exp 12 (TA FX triple-barrier) | Same leverage cap + OOS extension | very high |
| Exp 11 (TA futures triple-barrier) | Extend OOS to 200+ days | very high |
| Exp 15 (VRP) | Real IV data source (Amberdata / Kaiko / Tardis.dev) | very high |
| Exp 7 (MCb composite) | Cross-year validation 2022/2023 + daily regime classifier | high |
| Exp 9 (momentum triple-barrier) | Info bars + triple-barrier in bar-index space | high |
| Exp 8 (TSMOM) | Info bars (volume bars on ETH/SOL specifically) | medium-high |

These become the Cycle 36+ candidate shortlist.

---

## Execution log

### Step 1: regime_classes lookup

Code queried `data/praxis_meta.db`:
```
('A', 'Trend')
('B', 'Vol level')
('C', 'Vol trend')
('D', 'Serial correlation')
('E', 'Microstructure')
('F', 'Funding/positioning')
('G', 'Liquidity')
('H', 'Cross-asset corr')
('I', 'Volume participation')
('J', 'Term structure')
('K', 'Cross-sectional dispersion')
('L', 'RV / IV spread (VRP)')
```

12 canonical class letters and names captured for use in regime
bullets. One brief-to-DB drift surfaced: brief used "Vol-of-vol" for
C, but the DB canonical name is "Vol trend"; DB names won per the
brief's standing rule. Class L (RV / IV spread (VRP)) exists in the
table but is not referenced in any current trading experiment's
regime bullets -- it would be the natural addition if Exp 15 VRP
later gets real-IV-data revival.

### Step 2: Pre-existing cleanups (Exps 1 + 13)

- Exp 1: split combined-letter bullet into 6 individual canonical
  bullets. Used class names Vol level, Vol trend,
  Funding/positioning, Cross-asset corr, Term structure,
  Cross-sectional dispersion from the regime_classes lookup.
- Exp 13: regime bullets rewritten in canonical
  `F (Funding/positioning)` form. F-state details (+1,+2 / 0 /
  -1,-2) moved into description. Single canonical F-bullet replaces
  three compound-form bullets.

### Step 3: Result-row fixes

5 attribute-table rows added: Exps 3, 4, 7, 8, 11. Each gains a
`**Result**` row that classifier-parses to the correct result_class.

### Step 4: 13 fresh backfills

13 experiments backfilled mechanically (Exps 2, 3, 4, 7, 8, 9, 10,
11, 12, 14, 15, 16, 17). One mid-stream content adjustment needed:
Exp 8's Result row originally said `PARTIAL (WEAK POSITIVE; ...)`
which triggered the `_classify_result_token` POSITIVE-first
substring check; rephrased to `PARTIAL (weak; ...)` to thread the
classifier without changing semantic meaning. See Notes section
below for the underlying classifier ordering bug.

### Step 5: Pre-flight parser test

```powershell
python -m engines.atlas_sync --validate --no-embed --verbose
```

Output: pre-flight inline test showed all 15 trading experiments
with populated 4 new fields; all `regime_state_at_test` keys
canonical (no sanitized fallback shapes); 0 errors during parsing.
`atlas_sync --validate --no-embed` clean. Result_class populated
for all 15 (no NULLs).

### Step 6: atlas_sync real run

```powershell
python -m engines.atlas_sync
```

Output: TRADING_ATLAS.md 0 added / 15 updated / 0 removed; PMA 0
added / 0 updated / 20 unchanged; REGIME_MATRIX full replace (12
classes, 60 relevance rows); embeddings 15 regenerated, 20 skipped.
Matches brief expectation exactly.

### Step 7: MCP verification (user)

- `praxis:atlas_get(2)`: result_class=NEGATIVE,
  computational_engine=2, all 4 structured fields populated.
  test_conditions 8-key dict (matches brief's template),
  revival_hypotheses 4-item list of {title, likelihood,
  description}, regime_state_at_test 1-key dict
  `{general: "not_measured -- recommend re-running ..."}`.
- `praxis:atlas_get(12)` (Exp 14 grid bot): result_class=NEGATIVE,
  computational_engine=1 (Cointegration -- range-bound MR per
  brief).
- `praxis:atlas_get(13)` (Exp 15 VRP): result_class=INCONCLUSIVE,
  computational_engine=4 (Volatility/Options).
- `praxis:atlas_get(1)`: regime keys canonical -- 12 canonical
  keys (A_trend, D_serial_correlation, G_liquidity, I_volume,
  E_microstructure, B_vol_level, C_vol_trend, F_fundingpositioning,
  H_crossasset_corr, J_term_structure, K_crosssectional_dispersion)
  plus full_additive summary. No combined-letter fallback
  (b,_c,_f,_h,_j,_k is gone).
- `praxis:atlas_get(11)` (Exp 13 in DB): 3 canonical keys
  (F_fundingpositioning, A_trend, B_vol_level). F-state details
  (+1,+2 / 0 / -1,-2) moved into description text. No compound-form
  fallback shapes.
- `praxis:atlas_search("dollar bars revival")`: top 5 results all
  have dollar bars in revival hypotheses; top-2 are Exp 11
  (futures TB, sim 0.40) and Exp 1 (pairs MR, sim 0.36).
- `praxis:atlas_search("portfolio leverage cap")`: top-2 are
  Exp 10 (sim 0.40) and Exp 12 (sim 0.39) -- exactly the two
  experiments whose top revival is "hard portfolio leverage cap."
  Result_class properly populated for all 5 returned.

---

## Notes

### Light editorial findings (Q2 rich-scope)

One classifier-collision worked around during the edit pass: Exp 8's
brief content `PARTIAL (WEAK POSITIVE; ...)` triggered
`_classify_result_token`'s POSITIVE-first substring match.
Rephrased to `PARTIAL (weak; ...)` to thread the classifier without
changing meaning. Memory entry #18 added to formalize: the
underlying ordering bug in `_classify_result_token` (POSITIVE
pattern runs first; matches inside parens) is worth a small
follow-up cycle to fix at the source.

### Classifier substring-collision gotcha

`engines/atlas_sync.py::_classify_result_token` checks tokens in
this order: POSITIVE (if not NEGATIVE), NEGATIVE, INCONCLUSIVE /
BLOCKED, WEAK / PROMISING / PARTIAL. A Result-row value like
`**PARTIAL** (WEAK POSITIVE; ...)` evaluates the uppercased string
and finds the substring `POSITIVE` first (inside the parens), so
the function returns POSITIVE -- silently misclassifying an
otherwise-clearly-PARTIAL experiment.

Lesson: natural-language Result rows interact with token-based
classifiers in ways that can silently misclassify. Future cycles
using `WEAK POSITIVE`, `STRONG NEGATIVE`, etc. modifiers should
either (a) avoid the bare token inside parens, (b) reorder the
verdict patterns so `WEAK POSITIVE` matches before `POSITIVE`, or
(c) require explicit `result_class` metadata rather than relying
on classifier inference. For Cycle 35, option (a) was the
minimal-change path (rephrase `WEAK POSITIVE` -> `weak`); a
source-level fix is queued for a small follow-up cycle.

### Key-sanitization stripping of `/` and `-`

The brief predicted regime-key sanitization would collapse
`Funding/positioning` to `F_fundingpositioning` (no underscore in
the middle), and that prediction held exactly. The parser strips
`/` and `-` characters when building dict keys from class names,
producing valid-but-collapsed keys. Concrete sanitizations
observed:

- `Funding/positioning` -> `F_fundingpositioning`
- `Cross-asset corr` -> `H_crossasset_corr`
- `Cross-sectional dispersion` -> `K_crosssectional_dispersion`
- `RV / IV spread (VRP)` would sanitize to `L_rv_iv_spread_vrp`
  (not yet referenced in any experiment)

These are queryable canonical form even if visually a bit unusual;
spaces become underscores, `/` and `-` are dropped. Future cycles
adding class names with these characters should expect the same
collapse and not be surprised when querying.

### Strict-canonical regime keys: design holds

After the Exp 1 + Exp 13 cleanups, ALL 15 trading experiments have
canonical `<Letter>_<class_name>` regime keys. The strict-canonical
shape works: future experiments using compound-state descriptions
should follow the same pattern (canonical letter + class name in the
bullet key; state details in description text).

### Revival likelihood as design discipline

Designating revival hypotheses with explicit likelihood
(low / medium / high) forces priority discrimination. The Cycle 36+
selection process should heuristically: (a) prefer "very high"
likelihood revivals as first targets, (b) prefer revivals whose
infrastructure is already in place (info bars, leverage cap) over
those requiring new data acquisition (Exp 15 paid API), (c) prefer
revivals where the unblocking action is small (Exp 10 leverage cap)
over those requiring multi-cycle re-build (Exp 7 regime classifier).

### "Accept the verdict" as a legitimate revival hypothesis

Multiple NEGATIVE experiments include "accept the verdict" as a
high-likelihood option. This isn't surrender; it's signal that some
experiments produced conclusive enough evidence that further work
would be uncalibrated optimism. Recording these honestly prevents
future cycles from re-litigating closed questions.

---

## Open items / next cycle inputs

- **Cycle 36+**: First info-bar revival re-run(s). Candidates from
  the "very high likelihood" set above. The shortlist starts with
  Exp 10 (leverage cap fix -- cheapest test) and Exp 11 / 12 (OOS
  window extension -- pure data work). Higher-leverage tests (Exp 9
  info-bar + triple-barrier) come after the cheap fixes.
- **Engine 4 (VRP) data acquisition**: separate strand;
  evaluate Amberdata / Kaiko / Tardis.dev / CBOE BVOL pricing and
  pick a path forward.
- **`atlas_search` engine-filter parameter**: now genuinely useful
  with computational_engine fully populated across all 15
  experiments. ~10-line MCP tool change.
- **`_classify_result_token` ordering fix**: small source-level
  fix (reorder patterns so WEAK/PARTIAL match before bare
  POSITIVE/NEGATIVE). Memory entry #18 documents the workaround
  rationale; a follow-up cycle should land the proper fix.
- **PMA backfill**: parallel cycle eventually; the PMA structure is
  different (strategy categories vs. specific experiment results)
  and the schema fit needs design before backfilling.
- **LSTM v2 (Cycle 37+ probably)**: info bars + triple-barrier
  labeling + DL architecture refresh per Financial Innovation
  Feb 2025 paper. Now has both data substrate (Cycle 34) and atlas
  navigation to support hypothesis testing.
