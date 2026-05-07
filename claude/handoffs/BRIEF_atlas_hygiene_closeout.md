# Cycle 32 -- Atlas hygiene closeout + LSTM v2 TODO refresh

**Predecessors:** Cycle 12 (Atlas DB v0.1, commit `??`); Cycle 31
(`4cab1af`, schema migration program close).
**Mode:** Hybrid. Brief is content-only -- Code edits markdown, no
Python changes; runs `atlas_sync` at the end so the DB reflects
the new content. **Risk: low.** No DB schema changes, no live
collector touched.

## Why now

The pre-Cycle-17 work queue (snapshot: commit `6e8e00d`,
`claude/TODO.md` from 2026-04-30 morning) carried two Atlas
hygiene items that were displaced by the migration program:

1. **TRADING_ATLAS.md count reconciliation**: prose claims "17
   complete" experiments; the file has 15 unique numbered
   experiments (1-4, 7-17 -- gaps at 5, 6).
2. **Experiment 10 addendum parser fix** (Cycle 12 retro section
   7.1): claimed the parser was losing the "leverage runaway,
   not strategy failure" verdict.

Investigation in the predecessor turn shows item 2 is **already
resolved**: the Atlas DB record for Exp 10 (id 8) at
`atlas_get(8)` shows `full_markdown` includes the leverage
runaway content, the verdict, and the fix. Either the parser was
silently fixed in a later cycle or subsequent markdown edits
moved the addendum into the experiment body. Either way -- no
code work needed; just a close-out validation.

Item 1 is real. The "Pending experiments" placeholder block in
the markdown (lines 156-187) is stale: experiments 5 and 6 were
the placeholders for what *eventually* became Exp 8 (MOMENTUM x
CRYPTO TSMOM_4H+TSMOM_DAILY) and Exp 13 (MICROSTRUCTURE x CRYPTO
Funding Rate Carry) respectively, but the placeholders never got
removed when those experiments landed at their renumbered slots.

This cycle also picks up a related lightweight doc refresh that
came up during the predecessor turn: the **LSTM v2 upgrade plan**
isn't captured in `claude/TODO.md`. The current LSTM section
mentions "training run + walk-forward backtest" pending, but
nothing about info bars or triple-barrier labeling. The Financial
Innovation (Feb 2025) paper -- "Algorithmic crypto trading using
information-driven bars, triple barrier labeling and deep
learning" -- is the conceptual basis of the planned upgrade.
Brief intrabar v8.1 already used triple-barrier labeling for an
XGBoost classifier; the LSTM v2 upgrade should pull both
techniques (info bars + triple barrier labeling) into the
deep-learning model. Adding this as an explicit TODO closes the
"don't lose this" loop.

## What

Three files touched in `TRADING_ATLAS.md`, `claude/TODO.md`, and
the Atlas DB (via `atlas_sync` at the end). No code changes.

### Task 1: TRADING_ATLAS.md count reconciliation

Two count locations need correcting:

- **Line 235** (mid-file, "Total experiments | 17 complete |" in
  an earlier landscape-matrix block).
- **Line 768** ("Total experiments: 17 complete" + a breakdown of
  9 + 2 + 1 + 1 in the Final Updated Landscape Matrix v6).

Both should change to **15 complete**. The breakdown after the
total at line 768 needs a re-tally from the actual Atlas DB
records. Use `praxis:atlas_search` or read each entry's
`result_class` to get the count by category. Aim for the
following structure (re-verify each tally before publishing):

```
**Total experiments: 15 complete**
- NEGATIVE (no edge): N -- list (use atlas_search to verify)
- INCONCLUSIVE / BLOCKED: N -- list
- PARTIAL (weak positive): N -- list
- POSITIVE (confirmed edge): N -- list
```

The numbering gap (5, 6) is intentional -- preserve historical
entry IDs. Do NOT renumber 7-17 to fill the gap; that would break
historical citation references and force a full atlas re-embed
without any structural benefit.

### Task 2: TRADING_ATLAS.md "Pending experiments" section refresh

The section at lines 156-187 currently lists 6 placeholder
experiments. Status of each:

| Old # | Old title | Current state |
|---|---|---|
| 3 | TA_STANDARD x FUTURES_INDEX | DONE -- full entry at line 239 |
| 4 | TA_STANDARD x FX_G10 | DONE -- full entry at line 261 |
| 5 | MOMENTUM x CRYPTO | DONE -- became Exp 8 (line 377) |
| 6 | MICROSTRUCTURE x CRYPTO | DONE -- became Exp 13 (line 606) |
| 7 | FUNDAMENTAL x FX_G10 (carry) | NOT RUN -- still legitimately pending |
| 8 | ALTERNATIVE x CRYPTO (on-chain) | NOT RUN -- still legitimately pending |

Replace the section content with:

```markdown
## Pending experiments (legitimate; not yet run)

### FUNDAMENTAL x FX_G10 (carry)
- **Signal type**: Interest rate differential carry
- **Hypothesis**: Carry has documented risk premium; CPO may
  optimize timing. Originally listed as Exp 7 in earlier
  planning; not yet run as a standalone CPO experiment.

### ALTERNATIVE x CRYPTO (on-chain)
- **Signal type**: Active addresses, exchange flows, whale
  movements, hash rate trends.
- **Hypothesis**: Alpha from information edge rather than
  pattern edge. Originally listed as Exp 8 in earlier
  planning; data pipeline now in place via `onchain_btc`
  collector (Cycle 30 + Cycle 31), but no CPO experiment yet
  built.

> Historical note: this section previously listed pending
> experiments numbered 3-8. Items 3, 4 (TA_STANDARD x FUTURES,
> TA_STANDARD x FX_G10) became experiments 3 and 4. Items 5, 6
> (MOMENTUM, MICROSTRUCTURE) were renumbered to experiments 8
> and 13 when implemented. Items 7, 8 (FUNDAMENTAL FX, ALT
> CRYPTO) above remain genuinely pending.
```

### Task 3: claude/TODO.md updates

Two sections need touching:

**3a. Move two items to "Recently closed" with Cycle 32 reference:**

The active TODO at line 43 ("TRADING_ATLAS.md count
reconciliation") and line 48 ("Experiment 10 addendum parser
fix") move to the "Recently closed (last 30 days)" section. The
parser-fix entry should note "validated already-resolved; the
post-Cycle-12 markdown edits moved the addendum into the
experiment body, and atlas_sync re-ran picked up the change."

**3b. Add LSTM v2 upgrade plan**

In the "State / context" section (around line 170), the existing
"LSTM + Quantamental crypto prediction system: BUILT" entry
captures the v1 system. Below it, add a new subsection capturing
the v2 upgrade plan:

```markdown
### LSTM v2 upgrade plan: info bars + triple-barrier labeling

The v1 LSTM (`engines/lstm_predictor.py`, 1069 lines, 7
features x 60 timesteps with close/high/low/volume/FearGreed/
funding/Hurst) is built but not yet trained or validated. The
**v2 upgrade** plans to integrate two techniques from the
Financial Innovation (Feb 2025) paper, "Algorithmic crypto
trading using information-driven bars, triple barrier labeling
and deep learning":

1. **Information-driven bars** (replacing time bars): dollar
   bars, volume bars, or volume-imbalance bars per Lopez de
   Prado. Each bar represents equal economic activity (or
   equal directional conviction) rather than equal wall-clock
   time, which makes the LSTM's input distribution more
   stationary across regimes. Implementation comes in Cycle 34
   ("Info Bars v0.1").

2. **Triple-barrier labeling**: each training sample's label
   is determined by which of three barriers (TP, SL, time) is
   hit first within a lookforward window, rather than fixed
   N-step-ahead returns. This is the same labeling discipline
   used by intrabar confluence v8.1 (XGBoost) but applied to
   the LSTM. Aligns features and labels in bar-index space (not
   wall-clock space), which matters for event bars.

3. **Deep learning architecture refresh**: the paper documents
   architecture changes -- bidirectional layers, attention,
   regime-aware embedding -- that explain why "new" LSTMs
   work where v1-style architectures didn't.

**Pre-requisite**: Cycle 34 ships info bars (dollar +
volume-imbalance + volume run, per Lopez AFML Ch. 2). The LSTM
v2 cycle (Cycle 35+) consumes those bars and applies
triple-barrier labels in bar-index space.

**Data already collected** (from v1 prep): 1800 daily OHLCV,
2160 4h candles, 900 Fear & Greed, 2190 funding rates, 365
on-chain. For info bars, the 8.83M trades in
`crypto_data.trades` (post-Cycle-26 schema) provide the raw
trade tape needed to construct dollar/volume/imbalance bars at
arbitrary thresholds.

**Source paper**: search for "Algorithmic crypto trading using
information-driven bars, triple barrier labeling and deep
learning" -- Financial Innovation, February 2025.
```

The placement matters: this is a "State / context" addition
because the LSTM v2 isn't a single TODO item, it's a multi-cycle
plan that needs to live alongside the LSTM v1 BUILT entry. The
"Active TODOs" list will get a single entry pointing to this
plan when Cycle 33 completes (i.e., when info bars feel close).

### Task 4: atlas_sync re-run

After the TRADING_ATLAS.md edits, run:

```powershell
python -m engines.atlas_sync
```

This will re-parse the markdown, detect that 1 file changed,
re-embed any changed entries, update `data/praxis_meta.db`. The
"Pending experiments" section is parsed as non-experiment
content (it doesn't have `### N.` numbered headings under our
new structure), so atlas_sync should report 0 entries
added/changed/removed for the actual experiments table. Confirm
this in the output.

**Verify after sync** by re-running `atlas_search("MOMENTUM
crypto")` -- the existing results should be unchanged (same 15
experiments, same scores).

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | TRADING_ATLAS.md line 235 changed: 17 -> 15 |
| 2 | TRADING_ATLAS.md line 768 changed: 17 -> 15 + breakdown re-tallied from atlas DB |
| 3 | TRADING_ATLAS.md "Pending experiments" section reduced to FUNDAMENTAL x FX_G10 + ALTERNATIVE x CRYPTO + historical note |
| 4 | claude/TODO.md: lines 43, 48 moved to Recently closed with Cycle 32 + this commit hash |
| 5 | claude/TODO.md: new "LSTM v2 upgrade plan" subsection in State/context |
| 6 | atlas_sync runs cleanly; 0 experiments added/changed/removed (because the structural changes don't touch the 15 numbered experiments) |
| 7 | atlas_search("MOMENTUM crypto") returns same top-3 results as before |

## Out of scope

- Renumbering experiments to eliminate the 5, 6 gap.
- Filling in the now-pending FUNDAMENTAL x FX_G10 and
  ALTERNATIVE x CRYPTO experiments. Those are content work for
  future cycles, not hygiene.
- Refreshing the per-experiment `revival_hypotheses` /
  `test_conditions` data. That is Cycle 33 work and requires a
  schema change.
- Building info bars. That is Cycle 34.
- Implementing the LSTM v2 upgrade. That is Cycle 35+.

## Commit message (use verbatim)

```
Cycle 32: Atlas hygiene closeout + LSTM v2 TODO refresh

Closes two pre-Cycle-17 hygiene items displaced by the
schema migration program (Cycles 17-31):

1. TRADING_ATLAS.md count reconciliation. Prose claimed
   "17 complete" experiments; actual is 15 (numbered 1-4,
   7-17, with gaps at 5/6 from renumbering). Two count
   locations corrected (lines 235 + 768) plus a breakdown
   re-tally from the atlas DB.

2. TRADING_ATLAS.md "Pending experiments" placeholder
   refresh. The section listed 6 placeholders (3-8); items
   3-6 had since become Exps 3, 4, 8, 13. Section reduced
   to FUNDAMENTAL x FX_G10 (carry) and ALTERNATIVE x CRYPTO
   (on-chain) -- the two genuinely-pending experiments --
   plus a historical note explaining the renumbering.

Closes one already-resolved hygiene item: Experiment 10
addendum parser fix. atlas_get(8) confirms the parser
captures the leverage-runaway verdict correctly; the issue
was resolved by post-Cycle-12 markdown edits + atlas_sync
re-run.

Adds LSTM v2 upgrade plan to claude/TODO.md State/context
section. Captures the planned integration of (a)
information-driven bars (dollar/volume/imbalance per Lopez
de Prado, Cycle 34 deliverable), (b) triple-barrier
labeling (per intrabar v8.1's XGBoost, applied to the LSTM
in v2), (c) deep learning architecture refresh per the
Financial Innovation Feb 2025 paper.

atlas_sync re-run after edits: 0 experiments added/changed
(structural changes only, content of the 15 experiments
untouched).

Migration program scoreboard: 11/11 (Cycle 31).
Pre-Cycle-17 cleanup queue: 2 of 4 items closed
(this cycle); 2 items remain (Atlas count + Exp 10 are
both done; phase3_models retrain + burgess legacy cleanup
still open in lower-priority TODO bucket).
```
