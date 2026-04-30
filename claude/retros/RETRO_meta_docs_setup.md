# Retro: Meta-Docs Convention Setup (Praxis side)

**Series:** praxis
**Cycle:** 16
**Mode:** A (Chat-edited delta zip; pure documentation work plus memory-pruning that follows after zip applies)
**Outcome:** PASS pending zip application + memory cleanup
**Files created:** 5 (`claude/META_DOCS.md`, `claude/TODO.md`, `docs/TRADING_CONVENTIONS.md`, `docs/TRADING_FINDINGS.md`, `docs/MEV_BUILD_PLAN.md`)
**Files modified:** 1 (`claude/chat/NEW_CHAT_README.md`)

---

## 1. Trigger

`ai_factory_main_current` authored a `META_DOCS.md` convention earlier in
the day, establishing that per-repo state should live in versioned
`claude/*.md` files instead of shared cross-session memory. The doc
included Praxis-side instructions for `praxis_main_current` to do the
same on its side.

The Praxis-side execution required a memory-classification step (which
of the 25 entries are cross-project vs Praxis-specific) and a
file-creation step (where exactly does each Praxis-specific entry land).

---

## 2. Discovery: memory partitioning is per-Anthropic-Project

This chat (`praxis_main_current`) initially saw only 4 memory entries,
none of which matched AI Factory chat's 25. Investigation revealed:

- Claude.ai memory is partitioned per Anthropic Claude Project (the
  thing in the sidebar), NOT per chat / per repo / user-level
- This chat had been created OUTSIDE the McTheory Project by mistake
  during chat transition
- After Jeff added the chat to McTheory, the memory view jumped from
  4 entries to the 25-entry McTheory pool that AI Factory chat shares

This is a meaningful update to AI Factory chat's META_DOCS.md, which
asserted memory is "shared across all projects." The Praxis-side
META_DOCS.md (this cycle's deliverable) corrects this to "per-Anthropic-
Project" and adds a recovery note for future chats that find themselves
in the wrong pool.

---

## 3. Classification of the 25 memory entries

| Entry | Action | Destination |
|---|---|---|
| 1 (delta zips only) | KEEP in memory | Cross-project principle |
| 2 (no .env / .db in zips) | KEEP in memory | Cross-project safety rule |
| 3 (novel mechanics check) | KEEP in memory | Cross-project behavioral rule |
| 4 (VR profile experimental framework) | MOVE | `claude/TODO.md` -- low priority TODO |
| 5 (load_dotenv) | REWRITE + KEEP in memory | Drop "Praxis project:" prefix; cross-project Python hygiene |
| 6 (funding rates flip-positive alert) | MOVE | `claude/TODO.md` -- low priority TODO |
| 7 (funding rate model retrain trigger) | MOVE | `claude/TODO.md` -- mid priority TODO |
| 8 (actuarial engine for prediction markets) | MOVE | `claude/TODO.md` -- low priority TODO |
| 9 (Kraken Breakout signup) | MOVE | `claude/TODO.md` -- goals/long-term |
| 10 (LSTM + Quantamental BUILT status) | MOVE | `claude/TODO.md` -- state/context section |
| 11 (5-min BTC momentum strategy) | MOVE | `claude/TODO.md` -- low priority TODO |
| 12 (AI ensemble probability engine) | MOVE | `claude/TODO.md` -- low priority TODO |
| 13 (Polymarket market making bot) | MOVE | `claude/TODO.md` -- low priority TODO |
| 14 (validation-first principle) | KEEP in memory | Cross-project standard |
| 15 ("everything is a parameter") | KEEP in memory | Cross-project design principle |
| 16 (MEV build plan) | MOVE | `docs/MEV_BUILD_PLAN.md` -- standalone doc, multi-phase plan |
| 17 (Polymarket oracle check) | MOVE | `docs/TRADING_CONVENTIONS.md` -- pure domain rule |
| 18 (crypto prediction Corvino enhancements) | MOVE | `claude/TODO.md` -- low priority TODO |
| 19 (NegRisk arb non-executable) | MOVE | `docs/TRADING_FINDINGS.md` -- empirical finding |
| 20 (convergence speed detector) | MOVE | `claude/TODO.md` -- low priority TODO |
| 21 (Praxis MCP extensions) | MOVE + UPDATE | `claude/TODO.md` -- updated to reflect Cycle 12 (Atlas tools) and Cycle 14 (smart_money) shipping; only Alpaca + QuantConnect remain |
| 22 (weather bets gotchas) | MOVE | `docs/TRADING_CONVENTIONS.md` -- pure domain rule |
| 23 (cross-asset features for crypto LSTM) | MOVE | `claude/TODO.md` -- low priority TODO |
| 24 (Praxis reading queue) | MOVE | `claude/TODO.md` -- reading queue section |
| 25 (META_DOCS pointer) | KEEP in memory | Universal infrastructure rule |

**Net result**: memory drops from 25 entries to 7 (entries 1, 2, 3, 5
rewritten, 14, 15, 25). 18 entries moved to docs.

The "MOVE" entries are not deleted from memory yet -- the deletion
happens after Jeff applies this delta zip and confirms the destination
files contain the content durably. Two-step process is intentional: if
the zip apply fails or the files don't survive a commit, we'd lose
project knowledge.

---

## 4. Why the Praxis side differs from the AI Factory side

Two structural differences from AI Factory's `META_DOCS.md`:

**1. Pure-Praxis-domain content lives in `docs/`, not `claude/`.** Jeff's
discriminator: "is this about what Claude has to do, or is it about
Praxis content itself?" Claude-behavior content stays in `claude/`;
domain content (trading rules, market findings, build plans) goes in
`docs/` next to the existing project docs (TRADING_ATLAS, REGIME_MATRIX,
etc.). This is more granular than AI Factory's "everything in claude/"
approach, but appropriate for Praxis given that docs/ already has rich
project content.

**2. Memory model corrected to per-Anthropic-Project.** AI Factory's
META_DOCS.md asserted user-level shared memory; empirically false.
Praxis META_DOCS.md states the corrected model and adds a recovery
note for chats created outside McTheory.

---

## 5. Files created

### `claude/META_DOCS.md` (new)
Praxis-flavored convention doc. ~270 lines. Defines the meta-doc set
(7 docs read at session start, 6 read on demand), the memory-vs-meta-
docs split, reading discipline, cross-repo sync, the per-Project
memory partitioning gotcha, and the post-Cycle-16 memory state.

### `claude/TODO.md` (new)
Praxis work queue. Captures 18 Praxis-specific entries from memory
plus 10 open queue items from this morning's session (SCHEMA_NOTES,
onchain_btc monitoring, Atlas count reconciliation, Exp 10 addendum,
phase3 retrain, mojibake fix, Atlas surface in Claude.ai, Becker
dataset ingestion, UTC scheduler refactor, burgess.py legacy cleanup).

Organized by priority bucket then domain; includes a "State / context"
section for the LSTM BUILT status, a "Reading queue" section, and a
"Recently closed" section pointing to retros for the recovery sequence
(Cycles 8-15) so future chats have context on what just shipped.

### `docs/TRADING_CONVENTIONS.md` (new)
Pure Praxis domain operating rules. Two initial entries: Polymarket
oracle verification protocol (memory entry 17) and weather bet airport
station + multi-model ensemble guidance (memory entry 22). Set up to
grow.

### `docs/TRADING_FINDINGS.md` (new)
Empirical findings about market structure. One initial entry: NegRisk
arb non-executable (memory entry 19) with full setup / finding /
implication structure.

### `docs/MEV_BUILD_PLAN.md` (new)
The 5-phase MEV roadmap (memory entry 16). Standalone because of size
and ongoing relevance; future phases will likely add detail in place.

### `claude/chat/NEW_CHAT_README.md` (modified)
Inserted META_DOCS.md as 1st priority in the reading list (was 1st:
CLAUDE_CHAT_CONVENTIONS.md). All other priorities bumped down. Also
added TODO.md as 3rd priority since it carries a lot of post-Cycle-16
state.

### `claude/retros/RETRO_meta_docs_setup.md` (this file)

---

## 6. Required follow-up by Jeff

**Step 1: Apply this delta zip.** Files at root of zip (no nested
`praxis/` wrapper). Extract from inside praxis repo root.

**Step 2: Verify with git status / git diff.**
- Should show 1 modified (`claude/chat/NEW_CHAT_README.md`) + 5 new
  files. Should NOT show a `praxis/` untracked directory (that's the
  trap from Cycles 13-15; if it appears, manual cleanup needed).

**Step 3: Stage + commit + push.**

**Step 4: Memory cleanup.** Once Jeff confirms the commit pushed to
origin successfully, the chat (this chat or a future one) deletes the
18 Praxis-specific memory entries via `memory_user_edits` and rewrites
entry 5 to drop the "Praxis project:" prefix. Memory ends at 7 entries.

The two-step pattern (commit first, delete memory entries after) is
deliberate -- protects against zip apply / git push failures losing
project knowledge.

---

## 7. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `claude/META_DOCS.md` exists with Praxis-side convention | PASS (file built) |
| 2 | `claude/TODO.md` exists capturing all 18 Praxis-tagged entries | PASS |
| 3 | `docs/TRADING_CONVENTIONS.md` captures memory entries 17 + 22 | PASS |
| 4 | `docs/TRADING_FINDINGS.md` captures memory entry 19 | PASS |
| 5 | `docs/MEV_BUILD_PLAN.md` captures memory entry 16 | PASS |
| 6 | `claude/chat/NEW_CHAT_README.md` reading list updated | PASS |
| 7 | All files ASCII-only (Rule 20) | PASS |
| 8 | NEW_CHAT_README.md preserves CRLF line endings (Rule 21) | PASS (verified `tr -cd '\r' < file | wc -c` returns 164 = total lines) |
| 9 | Memory cleanup deferred to Jeff sign-off | PASS (documented in section 6) |

---

## 8. Open items / lessons

- **`docs/REGIME_MATRIX.md` should mention 13 classes, not 12.** Current
  Praxis state per recovery plan: Forced Flow Pressure was added as a
  13th regime class. Memory cleanup didn't touch `docs/REGIME_MATRIX.md`,
  but next time someone touches that file, update the count. Not
  blocking; flagging for awareness.

- **The "Recently closed" section of `claude/TODO.md`** should get
  appended whenever a cycle commits. Worth adding to `CLAUDE_CODE_RULES`
  as a maintenance step in Retro Rules. Or maybe just a one-line
  reminder at the bottom of TODO.md itself (which I included). Not
  enforcing it yet; let it emerge if it sticks.

- **The MCP extensions entry (memory 21)** had stale claims that smart_
  money and atlas tools were "future." Both shipped. Flagged the same
  drift in the doc-set itself: stale memory entries suggest a periodic
  audit step ("walk the memory entries, update for what's shipped").
  Not enforcing yet; might emerge as a need.

- **Schema heterogeneity continues to be a thread.** Open queue still
  includes `docs/SCHEMA_NOTES.md` as a high-priority TODO. The fact that
  it keeps coming up across retros (5 cycles now) is itself a signal it
  should be done sooner rather than later.

---

## 9. References

- AI Factory's `META_DOCS.md` (the trigger for this work)
- `claude/CLAUDE_CODE_RULES.md` v1.3 (the rules this cycle followed:
  Rule 19 ASCII / Rule 20 ASCII Unicode constants / Rule 21 CRLF
  preservation)
- `docs/praxis_main_series.md` and `docs/praxis_main_series_transcript.md`
  (the existing project narrative docs that META_DOCS.md references)
- All 8 retros from cycles 8-15 (the source material for the "Recently
  closed" section of TODO.md)
