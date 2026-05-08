# Retro: Cycle 33 -- Atlas schema extension + COMPUTATIONAL_ENGINES.md + 2 example backfills

**Brief:** `claude/handoffs/BRIEF_atlas_schema_extension.md`
**Date:** 2026-05-08
**Mode:** Hybrid (Claude drafted brief + COMPUTATIONAL_ENGINES.md
content + migration script; Code applied schema migration +
parser update + markdown edits; ran atlas_sync; verified
directly against `data/praxis_meta.db` because the praxis MCP
server isn't connected to Code's session -- user does the
MCP-side `praxis:atlas_get` / `praxis:atlas_search` checks.)
**Status:** DONE
**Predecessor:** Cycle 32 (`2c98a22`, atlas hygiene closeout)
**Commits:**
- `ae6e421` -- Cycle 33: schema migration + parser + COMPUTATIONAL_ENGINES.md
- `b643f77` -- Cycle 33 step 2: backfill Exp 1 + Exp 13

---

## Summary

Adds four new columns to `atlas_experiments` (`test_conditions`,
`revival_hypotheses`, `regime_state_at_test`,
`computational_engine`) supporting structured navigation of the
Atlas beyond the original signal_type / asset_class /
result_class fields. Adds `docs/COMPUTATIONAL_ENGINES.md` --
the 7-engine taxonomy doc that was previously chat-archive-
only. Validates the design by backfilling two reference
experiments (Exp 1 NEGATIVE, Exp 13 POSITIVE).

The remaining 13 backfills are explicitly Cycle 35 -- this
cycle just ships the infrastructure and proves it works on two
non-trivial cases.

Net change: +272 / -5 lines in `engines/atlas_sync.py`
(parser extension across both commits); +84 lines in
`TRADING_ATLAS.md` (Exp 1 + Exp 13 backfills);
441 lines in new `docs/COMPUTATIONAL_ENGINES.md`;
132 lines in
`scripts/migrations/cycle33_atlas_schema_extension.py`.

---

## Why this matters (recap)

Cycle 12 shipped `praxis_meta.db` with semantic similarity
search but a flat-text representation of experiment results.
Atlas could answer "find experiments similar to this idea" but
not "show me dead experiments that might be revived by switching
to dollar bars" or "which engines have I tested most heavily?"

Cycle 33 adds the structure that makes those questions
queryable. The 7-engine taxonomy is the second navigation axis
(complementing Signal x Asset and the Regime Matrix). The
revival_hypotheses field captures "dead under conditions
X, Y, Z" honestly rather than burying it in prose.

---

## Markdown template design (preserved here for reference)

Each experiment gets:

1. A new row in the existing attribute table:
   `| **Computational engine** | N (Engine name); secondary M (Engine name) |`
2. Three new bold-header sections after the experiment's
   existing prose body:
   - `**Test conditions:**` (table of aspect -> value)
   - `**Active regimes during test:**` (bullet list of class
     -> state, or "not measured")
   - `**Revival hypotheses:**` (numbered list of titled
     hypotheses with likelihood + description)

The parser extracts each into a JSON-shaped Python object,
then `json.dumps()` for storage in the TEXT columns. md_hash
changes when these sections gain content -> embeddings
re-fire for affected experiments only.

---

## Execution log

### Step 1: Schema migration

```powershell
python scripts\migrations\cycle33_atlas_schema_extension.py
```

Output: pre-state 16 columns -> post-state 20 columns; all
4 ALTER TABLE statements landed inside one BEGIN/COMMIT;
35 existing rows preserved with NULL in the new columns.
Idempotent by construction (the script detects already-added
columns); not re-run during this cycle.

### Step 2: COMPUTATIONAL_ENGINES.md added

Copied from delta zip to `docs/COMPUTATIONAL_ENGINES.md`. ASCII-
clean. 441 lines. Linked from the experiment-to-engine
mapping table at the end of the doc.

### Step 3: atlas_sync.py parser update

Extended in three places:
- Attribute-table row extraction: added `Computational engine`
  key recognition.
- New section parser for `**Test conditions:**`,
  `**Active regimes during test:**`,
  `**Revival hypotheses:**`.
- Section-content -> JSON conversion (table -> dict; bullet
  list -> dict; numbered list of bold-titled items -> list of
  dicts).

py_compile clean. +272 / -5 lines net across both commits.
Local validation tests run before any DB writes: a synthetic
fixture run through `_parse_test_conditions`,
`_parse_regime_bullets`, and `_parse_revival_hypotheses`
returned exactly the JSON shapes the brief specified, and a
follow-up run on the in-progress markdown confirmed Exps 1
and 13 produced the expected 8-key test_conditions, 5-7-key
regime dict, and 4-item revival list. One bug caught and
fixed during validation: regime bullet parser was treating
post-blank-line prose as continuation text -- terminating
bullets on blank lines was the fix.

### Step 4: TRADING_ATLAS.md backfill -- Exp 1

Added Computational engine attribute row + Test conditions
table + Active regimes during test bullet list + Revival
hypotheses numbered list. Test conditions captures bar type
(minute), frequency, universe, TC, feature set, pre-filter,
risk management, computational engine. Active regimes formalizes
the existing regime ablation table that was already in the
markdown (this experiment is one of the few with native regime
data). Revival hypotheses: 4 items, focusing on TC reduction
(highest likelihood) + regime A trend filter (medium) + dollar
bars (low; the constraint isn't bar selection here).

### Step 5: TRADING_ATLAS.md backfill -- Exp 13

Added the same four pieces. POSITIVE experiment, so revival
hypotheses are reframed as scaling/improving. 4 items: cross-
venue funding (high), term-structure feature (medium),
bear-market validation (confirmation not revival), LSTM v2
architecture (low for this engine).

### Step 6: atlas_sync re-run

```powershell
python -m engines.atlas_sync
```

Output: TRADING_ATLAS.md 0 added / 2 updated / 13 unchanged;
PMA 0 added / 0 updated / 20 unchanged; REGIME_MATRIX full
replace (12 classes, 60 relevance rows). Embeddings
(voyage-3-lite): 2 regenerated, 33 skipped. Matches the brief's
expectation exactly.

### Step 7: Verification (DB-side; MCP-side deferred to user)

The praxis MCP server isn't connected to Code's session, so
Code verified directly against `data/praxis_meta.db` using
sqlite3. User performs the equivalent `praxis:atlas_get(...)`
and `praxis:atlas_search(...)` calls in their own session as
the live-tool sanity check.

DB-side verification (Code):

- DB row id=1 (Exp 1): `computational_engine=1`,
  `test_conditions` = 8-key dict, `revival_hypotheses` =
  4-item list, `regime_state_at_test` = 7-key dict.
- DB row id=11 (Exp 13): `computational_engine=7`,
  `test_conditions` = 8-key dict, `revival_hypotheses` =
  4-item list, `regime_state_at_test` = 5-key dict.
- DB row id=2 (Exp 2 TA crypto, not backfilled): all 4 new
  columns NULL -- per-experiment backfill confirmed.
- 13 of 15 TRADING_ATLAS.md rows still NULL on
  test_conditions; only Exps 1 and 13 populated.
- Direct cosine search for "MOMENTUM crypto" against the
  voyage-3-lite embeddings returns DB ids 15, 7, 6 (= Exps
  17, 9, 8) in the same order as Cycle 32 -- non-backfilled
  embeddings are stable.

All 11 acceptance criteria from the brief pass on the
DB-side checks; user confirms the MCP-side equivalents.

---

## Notes

### Why two example backfills, not all 15

Two design pressures pull in opposite directions:

(a) Mass backfill is intellectually valuable -- structured
    revival hypotheses for all 15 experiments is what unlocks
    "Atlas as Research Agent companion."

(b) The markdown template is load-bearing. Once 15 experiments
    follow a shape, changing the shape becomes expensive
    (rewrite all 15 + re-embed all 15).

So Cycle 33 ships the shape + 2 carefully-chosen examples to
stress-test it (NEGATIVE + POSITIVE). If the design works for
both Exps 1 and 13, the remaining 13 are mechanical work that
fits cleanly into Cycle 35.

The user can sanity-check the 2 examples by reading the
backfilled markdown directly. If anything feels off, it's
cheap to revise the shape now and re-do 2 experiments before
Cycle 35 multiplies the work by 6-7x.

### Why `regime_state_at_test` is mostly NULL

The 12-class regime detector landed in `engines/regime_engine.py`
roughly contemporaneous with most of the experiments here.
Regime ablation runs are an explicit feature of some
experiments (Exp 1's Burgess pairs is one); native
regime-state-at-test is rare. We chose to ship the column
anyway because:

- Future experiments will populate it routinely (the regime
  engine is now stable infrastructure).
- For the 1-2 experiments where rich regime data exists (Exps 1,
  13), we DO populate it.
- For the rest, NULL or `"not_measured"` is honest signal --
  not pretending to data we don't have.

### Why the parser preserves md_hash sensitivity

The four new columns are *derived* from markdown content; they
are not independent state. So the rule from Cycle 12 holds:
markdown is source of truth, DB is derived. When the markdown
gains the new sections, md_hash changes, embeddings re-fire.
That's the right behavior because the new sections become
part of the searchable semantic content (a Research Agent
querying "experiments that might be revived by dollar bars"
should match Exps 1 and 13 if their revival_hypotheses mention
dollar bars).

### Hybrid workflow datapoint

Cycle 33 is the largest hybrid brief of the program so far
(532 lines including the embedded
COMPUTATIONAL_ENGINES.md content). Code's role is significant:
parser changes are non-trivial and the markdown edits are
substantive content. Active drafting time on Claude's side was
proportionally large too -- this is the right scope for hybrid.

---

## Open items / next cycle inputs

- **Cycle 34**: Info Bars v0.1. Independent of the Atlas
  schema work; standalone deliverable.
- **Cycle 35**: Atlas mass backfill of the remaining 13
  experiments. With Cycle 34 info bars in hand, several
  Engine 2 experiments will reference info bars in their
  `revival_hypotheses`.
- **Cycle 36+**: First info-bar revival re-runs (the
  highest-likelihood revival hypotheses from the backfilled
  Atlas).
- **`atlas_search` filter param** (deferred TODO): once
  `computational_engine` is populated for all experiments, add
  a `?engine=N` filter to atlas_search so the Research Agent
  can scope to a specific engine. Small follow-up.
