# Cycle 35.5 -- _classify_result_token ordering fix

**Predecessor:** Cycle 35 (`4d3d328` + retro `b651b22`). Memory entry #18.

**Mode:** Hybrid (Claude drafts brief + acceptance tests; Code makes the
source fix + adds tests + verifies via atlas_sync round-trip).

**Risk:** very low. Single function in `engines/atlas_sync.py`. Pure
pattern-ordering change. Round-trip verification (re-run atlas_sync
on TRADING_ATLAS.md, confirm zero md_hash changes) is the safety net:
if the fix accidentally changes any existing classification, we'll
see it immediately.

**Why now:** Cycle 35 worked around the substring-collision bug by
rephrasing Exp 8's Result row ("WEAK POSITIVE" -> "weak"). The
underlying bug remains; any future atlas entry using compound verdict
phrases ("WEAK POSITIVE", "STRONG NEGATIVE", "MARGINAL POSITIVE", etc.)
will silently misclassify. Fixing it now -- while the context is
still fresh -- prevents the next cycle from rediscovering the same
gotcha.

## What

Reorder pattern matching in `_classify_result_token` so MORE-SPECIFIC
patterns run before LESS-SPECIFIC patterns. The correct order from
most-specific to least-specific:

1. `INCONCLUSIVE` / `BLOCKED` (already first; keep)
2. `WEAK POSITIVE` / `WEAK NEGATIVE` / `STRONG POSITIVE` /
   `STRONG NEGATIVE` / `MARGINAL POSITIVE` / `PROMISING` -> PARTIAL
   (these contain POSITIVE or NEGATIVE as substring; they MUST
   match before bare POSITIVE / NEGATIVE)
3. `PARTIAL` -> PARTIAL
4. `POSITIVE` (bare) -> POSITIVE
5. `NEGATIVE` (bare) -> NEGATIVE

The principle: any compound phrase containing POSITIVE or NEGATIVE
must produce its compound-classification before the bare-word
classification can fire.

Code should read `engines/atlas_sync.py` to find the actual implementation
of `_classify_result_token` -- the structure may be a list of regex
patterns or a series of `if` statements; the brief doesn't dictate
which. The acceptance criteria below are the load-bearing contract.

## Acceptance criteria

**Behavioral acceptance (the truth table):**

| Input string (case-insensitive) | Expected classification |
|---|---|
| `**POSITIVE** (Sharpe +4.65, ...)` | POSITIVE |
| `**NEGATIVE** (Sharpe -0.94)` | NEGATIVE |
| `**PARTIAL** (weak; primary +1.91% Sharpe +0.545)` | PARTIAL |
| `**PARTIAL** (WEAK POSITIVE; ...)` | **PARTIAL** (was POSITIVE pre-fix) |
| `**PARTIAL** (STRONG NEGATIVE; ...)` | PARTIAL |
| `**INCONCLUSIVE** (-83.78% portfolio leverage runaway)` | INCONCLUSIVE |
| `**INCONCLUSIVE** (BLOCKED on data infrastructure)` | INCONCLUSIVE |
| `WEAK POSITIVE -- confirmed improvement` | PARTIAL |
| `PROMISING -- best result in atlas` | PARTIAL |
| `**NEGATIVE after TC**` | NEGATIVE |
| `Confirmed POSITIVE.` | POSITIVE |
| `Confirmed NEGATIVE.` | NEGATIVE |

**Round-trip acceptance:** Running `python -m engines.atlas_sync` after
the fix must produce:
- TRADING_ATLAS.md: 0 added / 0 updated / 0 removed
- PMA: 0 added / 0 updated / 0 removed
- Embeddings: 0 regenerated (since md_hash unchanged for all entries)

This proves the fix doesn't unintentionally change any existing
classification. If ANY entry changes, that's a flag: either the entry
was previously misclassified (which the fix corrected -- discuss
before committing) or the fix is too aggressive (which would mean
the brief's expected behavior is wrong somewhere).

**Test acceptance:** A new test file `engines/tests/test_atlas_classify.py`
(or wherever fits the existing test layout) covering the 12 truth-table
rows above, plus 2 edge cases:
- Empty string -> None
- String with no verdict words at all -> None

## Why round-trip is the safety net

The existing 35 atlas entries' result_class values are correct as of
Cycle 35 (with Exp 8 having been rephrased). If the fix is correct,
all 35 must re-classify to the same values. If any value changes
after the fix, one of two things is true:
- That entry was wrong before (the fix corrected it), OR
- The fix introduced new incorrect classification

Either way, it's worth investigating before commit. Round-trip is
the cheapest possible whole-system regression test.

## Step ordering

1. **Read current `_classify_result_token`** in `engines/atlas_sync.py`
   and adjacent helpers / pattern lists. Understand whether the
   implementation is a regex list, an if-else cascade, or something
   else. Document the current structure briefly in the retro.

2. **Run the brief's 12-row truth table against the current code**
   (small inline script: `for case in cases: print(case, _classify_result_token(case))`).
   Confirm at least the WEAK POSITIVE case currently misclassifies
   as POSITIVE. This proves the bug exists. Should also surface any
   OTHER cases that currently misclassify that the brief didn't
   anticipate.

3. **Apply the fix.** Reorder patterns so compound phrases run before
   bare words. Either:
   - If implementation uses a list of regex patterns: reorder list so
     WEAK/STRONG/MARGINAL/PROMISING patterns are listed BEFORE bare
     POSITIVE/NEGATIVE patterns.
   - If implementation uses if-else cascade: same -- check compound
     phrases first, fall through to bare words.

4. **Re-run the 12-row truth table.** All 12 must produce the
   expected classification.

5. **Add `engines/tests/test_atlas_classify.py`** (or fit existing
   test layout) with the 12 truth-table rows + 2 edge cases.
   Run via pytest; all 14 should pass.

6. **Round-trip verification.** Run `python -m engines.atlas_sync`
   on the actual TRADING_ATLAS.md + PREDICTION_MARKET_ATLAS.md.
   Expected: 0 changes across the board (all md_hashes stable;
   all result_class values stable; 0 embeddings regenerated).

7. **If any entry changed:** STOP and report which entry, what the
   old result_class was, what the new one is. Don't commit until
   we've discussed whether the change is a correct fix or a
   regression.

8. **Commit with `git commit -F` to avoid heredoc gotchas.**
   Commit message in step "Commit message" below.

## Commit message

Save to a temp file and `git commit -F`:

```
Cycle 35.5: _classify_result_token ordering fix

Fixes the substring-collision bug in engines/atlas_sync.py's
_classify_result_token surfaced during Cycle 35 (memory entry
#18): bare POSITIVE and NEGATIVE patterns matched before
compound phrases like WEAK POSITIVE or STRONG NEGATIVE,
causing Result rows of the form "**PARTIAL** (WEAK POSITIVE;
...)" to silently misclassify as POSITIVE.

Pattern matching is now ordered most-specific to least-specific:
INCONCLUSIVE/BLOCKED first (terminal markers); then compound
phrases (WEAK/STRONG/MARGINAL/PROMISING with POSITIVE/NEGATIVE
suffix) which all map to PARTIAL; then PARTIAL bare; then
POSITIVE bare; then NEGATIVE bare.

Adds engines/tests/test_atlas_classify.py covering 12 truth-
table cases plus 2 edge cases (empty input, no-verdict input).

Round-trip verification: re-running atlas_sync on the current
TRADING_ATLAS.md + PREDICTION_MARKET_ATLAS.md produces 0 added
/ 0 updated / 0 removed and 0 embeddings regenerated -- all
existing 35 entries classify identically before and after the
fix, confirming the change does not introduce regressions.

Cycle 35's Exp 8 markdown workaround (rephrasing "WEAK
POSITIVE" -> "weak") is left in place as a stylistic choice;
the classifier no longer requires it but the rephrased form
reads marginally cleaner in the atlas.
```

## Out of scope

- Re-writing Exp 8's Result row to put "WEAK POSITIVE" back. The
  Cycle 35 rephrase ("weak") is fine and reads cleaner; leaving it.
- Adding other result_class values (KILLED, WIP, etc.) -- whatever
  set exists today is what we're keeping.
- Changing how `result_class` is stored or queried -- this is a
  pure parser fix, no DB schema changes.

## Post-cycle status

After Cycle 35.5 lands:
- `_classify_result_token` handles compound verdict phrases correctly
- New test file documents the expected behavior for future edits
- Memory #18 stays in place as historical record of the workaround
  rationale
- Next: Cycle 36 design (Exp 10 leverage cap revival)
