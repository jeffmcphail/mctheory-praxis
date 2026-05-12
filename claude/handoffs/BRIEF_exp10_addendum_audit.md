# Cycle 36a -- Exp 10 addendum audit + atlas correction

**Predecessor:** Cycle 35.5 retro (`22e4989`); Cycle 36 RECON report.

**Mode:** Hybrid. Code performs the audit (git archeology + log inspection),
proposes a correction to the atlas markdown, executes the correction
after Claude confirms, runs atlas_sync, commits.

**Risk:** low. The atlas correction is a documentation fix; the underlying
experimental claim doesn't change, only the recorded mechanism. The
audit itself is read-only.

**Scope cap:** ~1 hour Code time. If the truth doesn't surface from
git log + commit messages + filesystem in that budget, stop and report
the inconclusive state -- a "don't know" finding is itself a finding,
and Cycle 36b/36c can be designed around the ambiguity if needed.

## Background

The Cycle 36 RECON report surfaced an atlas-vs-code discrepancy: Exp 10
(TRADING_ATLAS.md section "10. TA_STANDARD × CRYPTO (Triple Barrier
Re-run)") contains an Addendum claiming:

> "Re-ran Phase 4 with --max-portfolio-weight 0.50.
> Result: -27.95%, Sharpe -1.197."

But `--max-portfolio-weight` is currently unwired in the codebase:
`scripts/run_cpo.py:302` defines the flag (with docstring "Weights scaled
proportionally if total exceeds this cap"), but `cmd_phase4` never passes
it into `run_phase4`, and `run_phase4`'s signature doesn't accept it.

Three possible truths:

- **(i) Atlas misnaming.** The re-run actually used `--max-leverage 0.5`
  (which IS wired, at `cpo_core.py:468`). Same proportional-scale-down
  arithmetic, different CLI flag. The atlas Addendum used the wrong flag
  name.
- **(ii) Regression.** The `--max-portfolio-weight` flag was wired at a
  prior revision and got lost when the codebase rolled back or recovered
  from the 2026-04-24 disk failure.
- **(iii) Aspirational.** The Addendum's numbers were typed without ever
  having been run; documents a hypothesis, not an actual result.

Each truth points to a different Cycle 36b/36c design. Resolving this
ambiguity NOW (before any code change) is the cycle's purpose.

## Audit tasks

### Task A1: Git history of TRADING_ATLAS.md around the Addendum

Find when the Addendum was added to TRADING_ATLAS.md. Useful commands:

```powershell
git log --all --diff-filter=A -p TRADING_ATLAS.md | grep -n "max-portfolio-weight" | head -5
git log --all -p --grep="Exp 10" -- TRADING_ATLAS.md | head -100
git log --all -p TRADING_ATLAS.md -S "max-portfolio-weight 0.50" --oneline
git log --all -p TRADING_ATLAS.md -S "-27.95%" --oneline
git log --all -p TRADING_ATLAS.md -S "Sharpe -1.197" --oneline
```

(The `-S` flag finds commits where the added/removed lines contain the
string -- ideal for tracing when a specific phrase first appeared.)

Report:
- Commit SHA where the Addendum first landed
- Commit message
- Author + date
- Any sibling files touched in that commit (often a retro file or
  a results file landed alongside)

### Task A2: Git history of cpo_core.py / run_cpo.py

Check whether `max_portfolio_weight` was ever wired through `run_phase4`
or `compute_allocation`. Try:

```powershell
git log --all -p engines/cpo_core.py -S "max_portfolio_weight" --oneline
git log --all -p scripts/run_cpo.py -S "max_portfolio_weight" --oneline
git log --all -p -S "args.max_portfolio_weight" --oneline
```

Report:
- Commit SHAs where `max_portfolio_weight` was added, modified, or
  removed (likely just the introduction commit if it never went past
  the CLI flag stage)
- If any revision shows `run_phase4(max_portfolio_weight=...)` or
  `compute_allocation(... max_portfolio_weight=...)`, surface that
  -- it'd be evidence for truth (ii)

### Task A3: Filesystem search for prior run artifacts

The Cycle 36 RECON noted that `output/crypto_ta/cpo/` doesn't exist.
But the documented Exp 10 result must have come from somewhere. Search:

```powershell
git log --all --oneline --diff-filter=A | grep -i -E "exp[ _]?10|phase4|leverage cap"
dir /s /b *.parquet 2>nul
dir /s /b *_results.json 2>nul
dir /s /b *.joblib 2>nul
dir /s /b output 2>nul
```

Also check whether any logs / shell-history files survive:

```powershell
dir /s /b ps_history.txt 2>nul
dir /s /b cycle*.log 2>nul
dir /s /b exp10*.log 2>nul
```

Report any artifact whose timestamp is plausibly from the original
Exp 10 run (early-to-mid 2026 timeframe).

### Task A4: Inspect adjacent retro files

There may be a Cycle-N retro that documented the Exp 10 re-run. Check:

```powershell
dir claude\retros\ /b
dir claude\handoffs\ /b
```

For each retro file with a name suggesting "exp 10", "leverage",
"triple barrier", "ta crypto", or anything in that semantic
neighborhood, open and search for the actual command that was run.
Report the path + the relevant command line(s).

If a brief / retro pair exists for the original Exp 10 re-run, this is
the most reliable source of truth. Atlas Addendums are typically
written from retros; the retro has the verbatim command.

### Task A5: Synthesize the verdict

Combine findings from A1-A4. State the verdict in one of these forms:

- **VERDICT (i):** "Atlas mislabeled. The re-run used `--max-leverage 0.5`
  on date X (commit Y). Atlas Addendum needs the flag name corrected."
- **VERDICT (ii):** "Regression. The `--max-portfolio-weight` flag was
  wired at commit X and removed/lost at commit Y. The wiring should be
  restored as part of Cycle 36b."
- **VERDICT (iii):** "Aspirational. No commit, retro, log, or artifact
  evidences an actual re-run. Atlas Addendum needs to be reframed as
  'projected, not run' or removed."
- **VERDICT (inconclusive):** "After 1 hour of archeology, ambiguous.
  Best-guess interpretation: [...]. Cycle 36b should proceed as if
  truth (i) is correct, since (i) is the most parsimonious explanation."

Don't speculate beyond what evidence supports. "Inconclusive" is a
valid verdict; it just shapes the next cycle differently.

### Task A6: Propose the atlas correction (don't apply yet)

Based on the verdict, draft the corrected Addendum text. Show me
(Claude) the proposed change before applying. Likely shapes:

**If (i):**
> Re-ran Phase 4 with `--max-leverage 0.5` (proportional weight scale-
> down via the existing per-model arithmetic; the
> `--max-portfolio-weight` flag mentioned in earlier drafts is
> currently unwired and was not used).
> Result: -27.95%, Sharpe -1.197.

**If (ii):**
> Re-ran Phase 4 with `--max-portfolio-weight 0.50` (proportional weight
> scale-down). Result: -27.95%, Sharpe -1.197. NOTE: this wiring was
> lost in the 2026-04-24 disk failure; Cycle 36b restores it.

**If (iii):**
> Cap fix proposed: portfolio-level cap (e.g., `--max-portfolio-weight
> 0.5` or equivalent `--max-leverage 0.5`) is the recommended revival
> mechanism. Not yet executed; pending Cycle 36c.

**If (inconclusive):**
> Cap fix attempted (per earlier addendum; provenance unclear):
> reported -27.95% / Sharpe -1.197 with portfolio-level cap of 0.5.
> Replication pending Cycle 36c.

## Step ordering

1. Tasks A1-A4 in parallel where possible. Time-box each to ~10-15 min.
2. Task A5: write the verdict.
3. Task A6: propose the corrected Addendum text. PAUSE here; show
   Claude the proposal before applying.
4. Claude approves or refines the proposal.
5. Apply the markdown correction to TRADING_ATLAS.md.
6. Run `python -m engines.atlas_sync` to refresh the atlas DB.
7. Expected sync: TRADING_ATLAS.md reports 1 updated / 0 added /
   0 removed (just Exp 10). 1 embedding regenerated. If MORE changes
   surface, something else drifted -- stop and report.
8. Commit + push. Single commit; `git commit -F` for message safety.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Tasks A1-A4 completed with structured findings reported |
| 2 | A5 verdict is one of (i), (ii), (iii), or (inconclusive); supported by evidence from A1-A4 |
| 3 | Corrected Addendum text approved by Claude before applying |
| 4 | `atlas_sync` re-run reports exactly 1 TRADING entry updated (Exp 10); 0 added / 0 removed; 1 embedding regenerated |
| 5 | `praxis:atlas_get(8)` after the commit shows the corrected Addendum content; result_summary should still be "-83.78%, Sharpe -1.158" (the headline result, unchanged) |
| 6 | If the verdict surfaces information that would refine the Cycle 36b/36c briefs (e.g. specific commit hashes worth referencing, specific commands worth reproducing), capture it in the retro's "Open items / next cycle inputs" section |

## Commit message

After Claude approves the verdict + correction, use `git commit -F`:

```
Cycle 36a: Exp 10 addendum audit + atlas correction

Resolves the atlas-vs-code discrepancy surfaced by the Cycle 36
RECON report: TRADING_ATLAS.md's Exp 10 Addendum claimed the
re-run used --max-portfolio-weight 0.50, but that CLI flag is
currently unwired in scripts/run_cpo.py and engines/cpo_core.py.

Audit findings:
<VERDICT_SUMMARY>

Atlas Addendum corrected to <CORRECTION_SHAPE>.

No code changes. The wiring decision (restore --max-portfolio-
weight as a distinct mechanism, or document --max-leverage as
the canonical portfolio cap) is deferred to Cycle 36b.

atlas_sync re-run: 1 TRADING entry updated (Exp 10); 0 added /
0 removed; 1 embedding regenerated.
```

Fill <VERDICT_SUMMARY> and <CORRECTION_SHAPE> from the actual audit
findings.

## Out of scope

- Wiring `--max-portfolio-weight` into `run_phase4`/`compute_allocation`
  (that's Cycle 36b).
- Re-running Exp 10 to verify the documented -27.95% number (that's
  Cycle 36c).
- Adding tests, changing other atlas entries, or touching anything
  outside TRADING_ATLAS.md.
- Editing the existing Exp 10 result_summary / verdict / revival_
  hypotheses fields (those were verified correct in Cycle 35; only
  the Addendum is in question).
