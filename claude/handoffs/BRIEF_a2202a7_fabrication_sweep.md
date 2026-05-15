# Cycle 37 -- a2202a7 sibling-fabrication sweep

**Predecessor:** Cycle 36c (commits `fc9dff8` + `58add68`). Memory #19,
#21, project memory `project_a2202a7_fabrication_sweep_queued.md`.

**Mode:** Hybrid. Investigation is read-only (~30 min); corrections (if
any) staged for Claude review before atlas edits. Multi-stage with one
pause point between investigation report and corrections.

**Risk:** low. Investigation touches no code or data. Corrections (if
needed) are markdown-only documentation fixes, same shape as Cycle 36a
+ 36c parts of the Exp 10 cleanup.

**Scope cap:** ~1.5 h Code time total (~30 min investigation, ~30 min
discussion if findings warrant correction, ~30 min apply + sync +
commit).

## What this cycle does

Memory #19 records that commit `a2202a7` (2026-04-03, nominally "DEX
arb scanner, quoter, momentum detector -- experiments 16-17") contained
the two now-confirmed fabrications in Exp 10's section:

1. The "NOT RUN" Addendum claiming `--max-portfolio-weight 0.50` produced
   -27.95% / Sharpe -1.197 (retracted Cycle 36a)
2. The "338 signal configs × 72 barrier configs" line in Exp 10's
   Training row (corrected Cycle 36c)

This cycle audits the rest of a2202a7's diff for similar patterns. The
commit touched many files; we want to know whether the fabrication
pattern is isolated to Exp 10 or spread across other atlas entries.

## Patterns to look for

Based on the two confirmed examples:

**Pattern P1 -- Result-with-pending-fix-in-same-commit:** A claim of
"re-ran with X, result Y" lives in the same atomic diff as "fix
needed: X". The two are mutually exclusive in time -- you can't both
run with a fix and propose the fix in one commit.

**Pattern P2 -- Combine-unrelated-constants:** Two numbers from
genuinely different code paths multiplied together to describe a
single experiment, producing a config count or row count that no
actual code path produces.

**Pattern P3 -- Arithmetic-derived numbers presented as measured:**
Numbers that turn out to be cleanly derivable from other documented
numbers by simple arithmetic, with no surviving artifacts evidencing
measurement.

## Investigation tasks

### Task S1: Enumerate a2202a7's diff

```powershell
git show a2202a7 --stat
git show a2202a7 --name-only | grep -v "^$" | sort
```

Report:
- Total files touched
- Files in `docs/` and `*.md` (most likely fabrication targets)
- Files in `engines/` and `scripts/` (mostly real code per Cycle 36a)
- Files added (entirely new content; need full read-through) vs
  modified (only diff areas need scrutiny)

### Task S2: Atlas entries touched by a2202a7

For TRADING_ATLAS.md and PREDICTION_MARKET_ATLAS.md (if touched), find
which experiment sections were modified:

```powershell
git show a2202a7 -- TRADING_ATLAS.md | grep -E "^[+-].*###?" | head -30
git show a2202a7 -- PREDICTION_MARKET_ATLAS.md 2>nul | grep -E "^[+-].*###?" | head -30
```

For each touched section, capture the section title + experiment number
+ date_run if visible.

### Task S3: For each atlas entry touched in S2, scan for P1/P2/P3 patterns

Per touched entry, capture:

**P1 check:** Does the entry's body contain any "Re-ran X" / "Result:
Y" / "Final result" framings alongside "pending action" / "fix
needed" / "to-do" markers? If yes, surface the contradiction.

**P2 check:** Does the entry cite any product-of-two-numbers config
count (like "338 × 72" or "1728 (signal config × barrier config)")?
For each, verify the two factors against current code:
- Signal counts: check `engines/crypto_ta_strategy.py::generate_crypto_param_grid`,
  `engines/ta_models.py::generate_ta_param_grid`,
  `engines/universal_ta_strategy.py` (and any sibling).
- Barrier counts: `engines/triple_barrier.py::standard_barrier_grid`
- Asset counts: literal asset lists in the entry vs strategy adapter
  asset lists.

If the product doesn't match any actual code path, flag P2 hit.

**P3 check:** For each headline result number (cum return, Sharpe,
Max DD), check whether the entry has surviving artifact provenance:
- A linked retro file
- An outputs/ directory
- A commit referencing the run
- A SUMMARY.md or equivalent

If a headline number has NO surviving artifact AND can be cleanly
derived by arithmetic from another documented number in the same
section (or in a sibling section), flag P3 suspect.

### Task S4: Non-atlas files touched by a2202a7

`docs/praxis_main_series.md` was modified per Cycle 36a recon.
Other docs/* might have been too. Quick scan:

```powershell
git show a2202a7 -- docs/ | head -200
git show a2202a7 -- README.md 2>nul | head -100
```

For each non-atlas doc, scan for the same P1/P2/P3 patterns. The
priority is lower than atlas (atlas is the structured source of
truth; docs/ are more narrative), but worth a quick read.

### Task S5: New files added in a2202a7

`git show a2202a7 --diff-filter=A --name-only` lists files added in
the commit. Per Cycle 36a recon, these include "many new files for
Exps 16/17". For each new doc/md file:
- Skim for headline result claims
- If a result is cited, check whether artifacts exist alongside (e.g.
  if `docs/exp16_results.md` claims a Sharpe number, does
  `outputs/exp16/` exist?)

Lower-priority than S3 (atlas takes precedence) but ensures we don't
miss fabrications in places we don't normally look.

## Output: structured findings report

Reply with a single structured report following this template:

```
# a2202a7 fabrication sweep -- findings

## S1: Diff inventory
- Total files: N
- .md files: [list]
- engines/ files: [list]
- scripts/ files: [list]
- Files added: [list]
- Files modified: [list]

## S2: Atlas entries touched
- Exp X (section title, date_run): [confirmed/suspect/clean]
- Exp Y (section title, date_run): [confirmed/suspect/clean]
...

## S3: Per-entry findings (only entries with hits)

### Exp X (touched section title)
- P1 (pending-fix-in-same-commit): [hit / clear]
  - Evidence: [git-quote of the contradiction, or "none found"]
- P2 (unrelated-constants combo): [hit / clear]
  - Evidence: [the product, the actual code paths checked, mismatch]
- P3 (arithmetic-derived without artifacts): [suspect / clear]
  - Evidence: [the headline number, the derivation, artifact search results]

(repeat for each entry with at least one hit)

## S4: Non-atlas docs findings (only notable hits)

## S5: New files findings (only notable hits)

## Verdict summary
- Confirmed fabrications: N (count + entry list)
- Suspect (no smoking gun): N
- Clean: N
- Pattern observation: [is it just Exp 10, or did the pattern reach
  other entries?]

## Proposed corrections (if confirmed fabrications found)
[For each confirmed fabrication, draft the corrected text in the same
shape as Cycles 36a + 36c -- short retraction note + historical record
+ forward pointer to this cycle. Do NOT apply yet.]
```

## Pause point: review findings before corrections

After Code's report lands, PAUSE. I review the findings and approve
specific corrections (or decide to defer some to a later cycle). Code
should NOT apply any TRADING_ATLAS.md edits without explicit Claude
approval, even if a finding is unambiguous.

If the sweep returns CLEAN across all entries -- i.e. Exp 10 was the
sole carrier of fabrication patterns from a2202a7 -- the cycle ends
with a commit of the brief + retro alone (closing memory #19 with a
"swept, no further findings" note). This is a valid outcome and the
report should explicitly state it if it surfaces.

## After approval

For each approved correction:
1. Apply markdown edit to TRADING_ATLAS.md (or wherever)
2. Run atlas_sync; verify the expected update count
3. Update retro with execution log + per-correction diff stats

Single commit + push:

```
Cycle 37: a2202a7 fabrication sibling-sweep

Per memory #19 + project_a2202a7_fabrication_sweep_queued.md,
audited commit a2202a7 (2026-04-03) for fabrication patterns
beyond the two already caught in Exp 10 (Cycles 36a + 36c).

Findings:
- Confirmed: N (entries listed)
- Suspect: N
- Clean: N

Atlas updates: N TRADING entries corrected; M embeddings
regenerated.

[If clean across all entries:]
Sweep complete with no further findings. Exp 10 was the sole
carrier of fabrication patterns in a2202a7's diff. Memory #19
closed.

[If hits found:]
Pattern observation: the fabrication pattern reached [N] other
entries beyond Exp 10. [Brief characterization.]
```

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Investigation complete; structured report covers S1-S5 |
| 2 | Every atlas entry touched by a2202a7 has been scanned for P1/P2/P3 |
| 3 | Each finding is classified confirmed / suspect / clean with cited evidence |
| 4 | Proposed corrections drafted (if any) before any markdown edits |
| 5 | Claude reviews + approves before atlas edits land |
| 6 | atlas_sync round-trip clean; expected update count matches reality |
| 7 | Single commit + push via `git commit -F` |
| 8 | If clean across all entries: memory #19 can be closed/updated |

## Notes for Code

- The investigation pattern P1 / P2 / P3 mirrors what surfaced in
  Cycles 36a and 36c. Treat those cycles' retros as worked examples
  of what each pattern looks like in practice.
- For P2, the canonical config counts to keep in mind:
  - `crypto_ta_strategy`: 338 configs total (no triple barriers)
  - `universal_ta`: 110 signals × 72 barriers = 7920 configs
  - `standard_barrier_grid`: 72 barrier configs
  - `momentum_strategy` / `grid_bot_strategy`: import barriers; check
    each adapter's actual grid before flagging
- If a P3 finding is highly suspect but unprovable (no smoking gun),
  surface it as "suspect-deferred" -- we may want to investigate it
  in a future cycle when we have more context, but not block this
  cycle on it.
- Do not touch `docs/praxis_main_series.md` even if findings surface
  there -- the series doc is narrative; corrections there require
  separate consideration. Just report the finding.

## Out of scope

- Running any experiments (no phase2/phase3/phase4 reproductions)
- Re-running any analysis
- Other commits beyond a2202a7 (this is a focused single-commit audit)
- Other atlas hygiene (e.g. atlas_search engine-filter parameter from
  Cycle 35; deferred)
- PMA backfill (separate future cycle)
