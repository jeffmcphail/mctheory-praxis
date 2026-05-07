# Retro: Cycle 32 -- Atlas hygiene closeout + LSTM v2 TODO refresh

**Brief:** `claude/handoffs/BRIEF_atlas_hygiene_closeout.md`
**Date:** 2026-05-07
**Mode:** Hybrid (Claude drafted brief; Code edited markdown +
ran atlas_sync; user reviewed + approved push)
**Status:** DONE
**Predecessors:** Cycle 12 (Atlas DB v0.1); Cycle 31
(`4cab1af`, schema migration program close)

---

## Summary

Closes two pre-Cycle-17 hygiene items that were displaced by the
schema migration program (Cycles 17-31):

1. TRADING_ATLAS.md count reconciliation: prose corrected from
   "17 complete" -> "15 complete" with breakdown re-tallied
   from the Atlas DB.
2. "Pending experiments" placeholder section refreshed to reflect
   only the genuinely-pending FUNDAMENTAL x FX_G10 and
   ALTERNATIVE x CRYPTO experiments, plus a historical note
   explaining the renumbering of items 3-6.

Plus one already-resolved item validated:
3. Experiment 10 addendum parser fix: `atlas_get(8)` confirms
   the parser captures the leverage-runaway verdict correctly.
   No code work needed; issue self-resolved through later
   markdown edits.

Plus one new strategic doc addition:
4. LSTM v2 upgrade plan: added to `claude/TODO.md` State/context
   section, capturing the planned integration of info bars +
   triple-barrier labeling + deep-learning architecture refresh
   per the Financial Innovation (Feb 2025) paper.

Net change: 37 deletions / 31 insertions in TRADING_ATLAS.md
(net -6); 13 deletions / 93 insertions in claude/TODO.md (net
+80, mostly the LSTM v2 plan + Cycle 32 closed entry). atlas_sync
re-run after edits: 0 added / 0 updated / 15 unchanged on
TRADING_ATLAS.md (matches expectation -- structural changes only,
the 15 numbered experiment bodies untouched, so md_hashes and
embeddings are stable).

---

## Why this came up now

User flagged at end of Cycle 31 that the migration program
(Cycles 17-31) was an unplanned detour from "necessary cleanup
before info bars." The pre-Cycle-17 TODO snapshot (commit
`6e8e00d`, dated 2026-04-30 morning) carried 4 high-priority
hygiene items, of which 2 (SCHEMA_NOTES.md + onchain_btc
monitoring) got absorbed by the migration program itself, and 2
(Atlas count reconciliation + Exp 10 parser fix) were left
hanging. Cycle 32 closes those last two.

User also flagged a separate concern -- the LSTM v2 upgrade plan
was being lost across sessions. The current TODO.md mentioned
the v1 system as "BUILT" with "training run + walk-forward
backtest" pending, but no mention of info bars or
triple-barrier labeling. This cycle adds that plan as a
State/context entry so it persists across all future sessions.

---

## Execution log

### Step 1: Verify Experiment 10 already-resolved status

`praxis:atlas_get(8)` returned `full_markdown` containing the
leverage-runaway verdict, the fix, and the inconclusive
classification. The Cycle 12 retro section 7.1 issue is
historically real but currently moot -- post-Cycle-12 markdown
edits and the most-recent atlas_sync (synced_at
`2026-04-29T21:44:51`) capture the full content. Cycle 32 does
NOT do parser code work; it just notes the validation.

### Step 2: TRADING_ATLAS.md count corrections

Two corrections at lines 235 and 768 (prose count: 17 -> 15).
Breakdown at line 768 re-tallied from `praxis:atlas_search`
queries against each result_class:

- NEGATIVE: 7 (atlas DB IDs 1, 2, 3, 4, 12, 14, 15 -- mapping to
  Exps 1 mean-rev equity, 2 TA crypto, 3 TA futures regime-dep,
  4 TA FX, 14 grid bot, 16 DEX spatial arb, 17 1-min momentum)
- INCONCLUSIVE / BLOCKED: 3 (atlas DB IDs 8, 10, 13 -- Exps 10
  TA crypto triple-barrier re-run, 12 TA FX triple-barrier re-run,
  15 VRP × BTC/ETH blocked on real-IV data)
- PARTIAL: 4 (atlas DB IDs 5, 6, 7, 9 -- Exps 7 MCb CPO, 8 TSMOM
  4h+daily, 9 momentum ETH/SOL triple-barrier re-run, 11 TA
  futures triple-barrier re-run)
- POSITIVE: 1 (atlas DB ID 11 -- Exp 13 Funding Rate Carry × Crypto)

Note on Exp 3 / atlas DB ID 3: the parser stored `result_class`
as NULL for this row (no explicit result_class header in the
markdown for this entry). Folded into NEGATIVE for the breakdown
because the in-document conclusion reads "regime-dependent, not
structural" -- a NEGATIVE finding by content. A future cleanup
could re-tag this in the DB by editing the markdown to add an
explicit result_class line, but it's out of scope for this
content-only cycle.

### Step 3: TRADING_ATLAS.md Pending Experiments refresh

Section at lines 156-187 reduced from 6 placeholder items to 2
genuinely-pending items:

- FUNDAMENTAL x FX_G10 (carry) -- still pending
- ALTERNATIVE x CRYPTO (on-chain) -- still pending; data pipeline
  ready post-Cycles 30+31

Plus a historical note explaining the renumbering: original
items 3, 4 became Exps 3, 4 directly; items 5, 6 became Exps 8,
13 with renumbering; items 7, 8 (FUNDAMENTAL FX, ALT CRYPTO) are
the remaining unfilled slots.

### Step 4: claude/TODO.md updates

**Items moved to Recently closed:**
- Line 43 entry: TRADING_ATLAS.md count reconciliation -> closed
  by Cycle 32 commit `<CYCLE_32_HASH>`
- Line 48 entry: Experiment 10 addendum parser fix -> closed by
  validation (already-resolved); see this retro for context

**New State/context subsection added (around line 170, below
existing LSTM v1 entry):**
"### LSTM v2 upgrade plan: info bars + triple-barrier labeling"
captures (a) info bars from Lopez AFML Ch. 2 [Cycle 34
deliverable], (b) triple-barrier labeling per intrabar v8.1's
XGBoost discipline, (c) deep-learning architecture refresh per
the Financial Innovation (Feb 2025) paper. Placement: alongside
the LSTM v1 BUILT entry, since v2 is a multi-cycle plan not a
single TODO item.

### Step 5: atlas_sync re-run

```powershell
python -m engines.atlas_sync
```

Result: TRADING_ATLAS.md: 0 added / 0 updated / 15 unchanged.
Embeddings: 0 regenerated / 35 skipped. Total entries in
atlas_experiments: 35 (15 TRADING_ATLAS + 20 PREDICTION_MARKET).
PREDICTION_MARKET_ATLAS.md and REGIME_MATRIX.md also unchanged.
Confirms that the count reconciliation + Pending Experiments
rewrite are content-only, not schema-affecting.

### Step 6: Verification

`atlas_search("MOMENTUM crypto", top_k=3)` re-run post-sync
(via direct sqlite3 + voyage-3-lite query embedding, since the
praxis MCP server isn't connected to this Code session; the
underlying embeddings table is identical to what the MCP tool
reads). Top-3 results:

| Rank | id | sim | Section |
|---|---|---|---|
| 1 | 15 | 0.5349 | 17. SHORT-TERM MOMENTUM × CRYPTO (1-min signals) |
| 2 |  7 | 0.4925 | 9. MOMENTUM × CRYPTO (ETH+SOL Triple Barrier Re-run) |
| 3 |  6 | 0.4547 | 8. MOMENTUM × CRYPTO (TSMOM_4H + TSMOM_DAILY) |

All three are MOMENTUM × CRYPTO experiments as expected;
embeddings unchanged so this matches the pre-edit state and AC
#7 is satisfied.

---

## Notes

### Why preserving the numbering gap (5, 6) is correct

Renumbering 7-17 to 5-15 would force a full atlas re-embed (every
md_hash changes), break historical citations to the experiment
numbers in retros and chat history, and provide zero structural
benefit. The gap is a record of the project's organic evolution
-- experiments 5 and 6 were planned, then renumbered when
implemented because additional context emerged during the
work. That's worth preserving.

### Why the LSTM v2 plan goes in State/context, not Active TODOs

LSTM v2 is a multi-cycle plan with hard prerequisites (info bars
must ship first, in Cycle 34). Adding it as a single Active TODO
entry would either oversimplify the dependency or create
multiple sub-bullets that get out of sync. The State/context
section is for "background information that future sessions need
to know" -- which is exactly what this is. When Cycle 34 lands,
a clean Active TODO entry pointing to this State/context entry
gets added.

### Triple-barrier reuse from intrabar v8.1

Intrabar confluence v8.1 (XGBoost, lookforward=15, atr_mult=1.5)
already implements triple-barrier labeling in
`engines/intrabar/labels.py` (or similar; verify path during
Cycle 35). The LSTM v2 work should pull that labeling code
into a shared utility rather than reimplementing -- keeps the
labeling discipline consistent across XGBoost (intrabar) and
LSTM (v2) models.

### Pre-Cycle-17 cleanup queue: status update

The 4-item morning queue (commit `6e8e00d`):

- SCHEMA_NOTES.md timestamp heterogeneity doc -- absorbed and
  shipped by Cycle 17.
- onchain_btc MCP monitoring -- absorbed and shipped by Cycles
  17 + 30 + 31.
- TRADING_ATLAS.md count reconciliation -- closed this cycle.
- Experiment 10 addendum parser fix -- validated already-resolved
  this cycle.

**All 4 pre-Cycle-17 high-priority items are now closed.** The
detour ends here.

---

## Open items / next cycle inputs

- **Cycle 33: Atlas schema extension.** Add `test_conditions`,
  `revival_hypotheses`, `regime_state_at_test`,
  `computational_engine` columns + `docs/COMPUTATIONAL_ENGINES.md`.
  Backfill all 15 experiments.
- **Cycle 34: Info Bars v0.1.** Dollar bars, volume bars,
  volume-imbalance bars, volume-run bars per Lopez AFML Ch. 2.
  Built against the post-Cycle-26 `trades` table.
- **Cycle 35+: LSTM v2 upgrade.** Architectural refresh + info
  bars consumption + triple-barrier labeling. Per the Financial
  Innovation (Feb 2025) paper.
- **Atlas revival pass** (after Cycle 33 schema extension):
  for each experiment marked NEGATIVE or INCONCLUSIVE, fill in
  `revival_hypotheses` field. Top candidates whose hypotheses
  reference info bars become Brief candidates for the actual
  revival re-runs.
