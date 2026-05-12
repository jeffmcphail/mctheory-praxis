# Retro: Cycle 35.5 -- _classify_result_token ordering fix

**Brief:** `claude/handoffs/BRIEF_classifier_ordering_fix.md`
**Date:** 2026-05-12
**Mode:** Hybrid (Claude drafted brief + 14-row truth table acceptance;
Code applied source fix + added pytest file + ran round-trip
verification)
**Status:** DONE
**Predecessor:** Cycle 35 (`4d3d328` + retro `b651b22`); Memory entry #18
**Commit:** `106b515`

---

## Summary

Reorders pattern matching in `engines/atlas_sync.py::_classify_result_token`
so compound verdict phrases (WEAK POSITIVE, STRONG NEGATIVE, PROMISING,
MARGINAL POSITIVE, etc.) match before bare POSITIVE / NEGATIVE patterns.

Bug surfaced during Cycle 35: Exp 8's Result row `**PARTIAL** (WEAK
POSITIVE; ...)` was misclassified as POSITIVE because the bare
POSITIVE pattern matched the substring inside the parens before the
compound WEAK POSITIVE pattern got a chance. Cycle 35 worked around
by rephrasing the markdown ("WEAK POSITIVE" -> "weak"); Cycle 35.5
fixes the underlying classifier.

Net change (per `git show 106b515 --stat`):
- `engines/atlas_sync.py`: +22 / -5 (function body grew from ~10
  lines to ~22 lines including a 3-line rationale comment)
- `engines/tests/test_atlas_classify.py`: 41 lines (new file)
- `engines/tests/__init__.py`: 0 lines (new package marker)
- `claude/handoffs/BRIEF_classifier_ordering_fix.md`: 188 lines
  (brief committed alongside)

Round-trip verification: re-running `atlas_sync` after the fix
reports 0 added / 0 updated / 0 removed across both TRADING and PMA
atlases, confirming all 35 existing entries classify identically
before and after the fix.

---

## Why this matters

Cycle 35 caught the bug because Code spot-checked the result_class
of 15 backfilled entries during pre-flight. Without that check the
bug would have shipped silently: Exp 8 would have appeared in atlas
DB with `result_class=POSITIVE`, and `atlas_search` filtering for
POSITIVE experiments would return a misleading result.

The substring-collision pattern (compound phrase containing the bare
word) is general enough that future entries using "STRONG NEGATIVE"
or "MARGINAL POSITIVE" would hit the same issue. Fixing it now (~30
min cycle) prevents N future cycles from independently rediscovering
the same gotcha.

---

## Execution log

### Step 1: Read current `_classify_result_token`

Implemented as an if-elif cascade in `_classify_result_token` at
`engines/atlas_sync.py:328`. Original order: bare POSITIVE (with
NEGATIVE exclusion) -> bare NEGATIVE -> INCONCLUSIVE/BLOCKED ->
WEAK/PROMISING/PARTIAL. The bare patterns came first, which is
what allowed the substring collision.

Note: the file also contains a separate `_VERDICT_PATTERNS` regex
list used by `_extract_result_class` for prose blocks; that list
was already correctly ordered (WEAK POSITIVE listed before bare
"no edge" / "confirmed positive" patterns) and was not touched by
this cycle. Only `_classify_result_token` (the attribute-table
value classifier) had the ordering bug.

### Step 2: Run the 12-row truth table against current code

```
OK    expect=POSITIVE       got=POSITIVE       '**POSITIVE** (Sharpe +4.65, ...)'
OK    expect=NEGATIVE       got=NEGATIVE       '**NEGATIVE** (Sharpe -0.94)'
OK    expect=PARTIAL        got=PARTIAL        '**PARTIAL** (weak; primary +1.91% Sharpe +0.545)'
FAIL  expect=PARTIAL        got=POSITIVE       '**PARTIAL** (WEAK POSITIVE; ...)'
FAIL  expect=PARTIAL        got=NEGATIVE       '**PARTIAL** (STRONG NEGATIVE; ...)'
OK    expect=INCONCLUSIVE   got=INCONCLUSIVE   '**INCONCLUSIVE** (-83.78% portfolio leverage runaway)'
OK    expect=INCONCLUSIVE   got=INCONCLUSIVE   '**INCONCLUSIVE** (BLOCKED on data infrastructure)'
FAIL  expect=PARTIAL        got=POSITIVE       'WEAK POSITIVE -- confirmed improvement'
OK    expect=PARTIAL        got=PARTIAL        'PROMISING -- best result in atlas'
OK    expect=NEGATIVE       got=NEGATIVE       '**NEGATIVE after TC**'
OK    expect=POSITIVE       got=POSITIVE       'Confirmed POSITIVE.'
OK    expect=NEGATIVE       got=NEGATIVE       'Confirmed NEGATIVE.'

3/12 failed
```

Confirmed: `**PARTIAL** (WEAK POSITIVE; ...)` -> POSITIVE (BUG).
Two additional failures surfaced:
- `**PARTIAL** (STRONG NEGATIVE; ...)` -> NEGATIVE -- symmetric
  counterpart to the WEAK POSITIVE failure; same root cause
  (compound phrase containing the bare word).
- `WEAK POSITIVE -- confirmed improvement` -> POSITIVE -- the same
  pattern in a different syntactic context (string start, no
  parens, no surrounding PARTIAL marker).

### Step 3: Apply the fix

Reordered the if-elif cascade in `_classify_result_token`. New
order: INCONCLUSIVE/BLOCKED -> compound (8 phrase variants) ->
bare POSITIVE -> bare NEGATIVE. The compound branch enumerates
8 phrases explicitly (WEAK / STRONG / MARGINAL crossed with
POSITIVE / NEGATIVE = 6, plus PROMISING and PARTIAL = 8) for
legibility rather than collapsing them into a regex. Added a
3-line comment block immediately above the compound branch
explaining the ordering rationale so future readers don't
re-shuffle the cascade and reintroduce the bug.

### Step 4: Re-run truth table

```
OK   expect=POSITIVE       got=POSITIVE       '**POSITIVE** (Sharpe +4.65, ...)'
OK   expect=NEGATIVE       got=NEGATIVE       '**NEGATIVE** (Sharpe -0.94)'
OK   expect=PARTIAL        got=PARTIAL        '**PARTIAL** (weak; primary +1.91% Sharpe +0.545)'
OK   expect=PARTIAL        got=PARTIAL        '**PARTIAL** (WEAK POSITIVE; ...)'
OK   expect=PARTIAL        got=PARTIAL        '**PARTIAL** (STRONG NEGATIVE; ...)'
OK   expect=INCONCLUSIVE   got=INCONCLUSIVE   '**INCONCLUSIVE** (-83.78% portfolio leverage runaway)'
OK   expect=INCONCLUSIVE   got=INCONCLUSIVE   '**INCONCLUSIVE** (BLOCKED on data infrastructure)'
OK   expect=PARTIAL        got=PARTIAL        'WEAK POSITIVE -- confirmed improvement'
OK   expect=PARTIAL        got=PARTIAL        'PROMISING -- best result in atlas'
OK   expect=NEGATIVE       got=NEGATIVE       '**NEGATIVE after TC**'
OK   expect=POSITIVE       got=POSITIVE       'Confirmed POSITIVE.'
OK   expect=NEGATIVE       got=NEGATIVE       'Confirmed NEGATIVE.'

0/12 failed
```

All 12 cases match expected classification. All 3 previous
failures now classify as PARTIAL.

### Step 5: Add test file

`engines/tests/test_atlas_classify.py` created with 14 cases (12
truth-table + 2 edge). `engines/tests/__init__.py` added as a
package marker since `engines/tests/` didn't previously exist
(only `engines/info_bars/tests/` did). Pytest output:

```
============================= 14 passed in 0.20s ==============================
```

14 passed (12 parametrized truth-table cases + 2 edge cases:
empty string -> None; "no verdict words at all" -> None).

### Step 6: Round-trip verification

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

TRADING_ATLAS.md 0 added / 0 updated / 0 removed; PMA 0 added /
0 updated / 20 unchanged; embeddings 0 regenerated. Identical
classification for all 35 entries before and after. Round-trip
safety contract satisfied. **Result: pass.**

### Step 7: Commit + push

Commit `106b515`. Pushed to master via `git commit -F` to avoid
the heredoc `$` gotcha noted in memory #17 / Cycle 34 commit
808f19e.

---

## Notes

### Other classification cases discovered

No other misclassification cases beyond the 3 surfaced in Step 2
(WEAK POSITIVE inside parens, STRONG NEGATIVE inside parens, WEAK
POSITIVE at string start). The classifier's behavior on
INCONCLUSIVE markers (INCONCLUSIVE NEGATIVE, BLOCKED ON DATA, etc.)
was already correct because INCONCLUSIVE/BLOCKED branch ran first
in the original cascade -- the bare POSITIVE/NEGATIVE branches
never got a chance to fire on those inputs. Likewise, `**PARTIAL**
(weak; ...)` (the Cycle 35 rephrase) classified correctly even
pre-fix because lowercase "weak" doesn't contain POSITIVE or
NEGATIVE as a substring.

### Why this fix doesn't need a memory entry update

Memory #18 documents the original Cycle 35 workaround rationale.
After this cycle the workaround is no longer needed (the classifier
handles "WEAK POSITIVE" correctly), but the memory entry remains
useful as historical record of (a) why Exp 8's markdown says "weak"
instead of "WEAK POSITIVE" and (b) the general lesson that
substring collisions are a pattern to watch for in token-based
classifiers.

### Round-trip as cheapest possible regression test

For pattern-ordering changes, "all existing entries must classify
identically" is the strongest test we can run cheaply. It costs one
atlas_sync invocation and proves the fix doesn't break anything in
the corpus. Adopting this as standard practice for any future
parser changes.

---

## Open items / next cycle inputs

- **Cycle 36 design**: Exp 10 leverage cap revival -- the cheapest
  "very high likelihood" candidate from Cycle 35's revival shortlist.
- **`atlas_search` engine-filter parameter** (deferred TODO; small
  MCP tool change).
- **PMA backfill** (separate cycle; PMA structure differs from
  TRADING).
- **LSTM v2** (Cycle 37+; info bars + triple-barrier).
