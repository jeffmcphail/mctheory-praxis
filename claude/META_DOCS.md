# Meta-Docs Convention (Praxis side)

**Version:** 1.0
**Created:** 2026-04-30 (Chat: praxis_main_current -- Cycle 16)
**Applies to:** Praxis repo. Companion convention exists in `ai-agent-factory` and (when set up) `core`.

---

## Why this exists

Claude Chat has a hard cap on cross-session memory. Memory is also **partitioned per Anthropic Claude Project** (verified empirically 2026-04-30: a chat created outside the McTheory Project saw a different 4-entry memory pool, while chats inside McTheory share a 25-entry pool). Two consequences:

1. The cap is real and shared across all Praxis + AI Factory + future Core work in the McTheory Project. Every Praxis-specific TODO that lives in memory steals a slot from a cross-project operating principle.
2. If a chat ever gets created outside the McTheory Project by mistake, it will see a different memory pool entirely and won't learn anything we put there until moved.

So the discipline is: **memory holds only cross-project principles.** Repo-specific state lives in versioned `claude/*.md` files inside each repo (the "meta-docs"), and Chat reads them at session start. The files are the source of truth; memory is a runtime cache that may or may not have what we think it has.

This doc defines:

1. The Praxis meta-doc set -- which files are meta-docs, what each contains, what does NOT belong in them
2. The reading discipline -- when Chat reads each doc
3. The memory-vs-meta-docs split -- what stays in memory vs what gets written to a file
4. Cross-repo synchronization -- how Praxis stays aligned with AI Factory's parallel set

---

## What counts as a "meta-doc"

A meta-doc is a `claude/*.md` file that **Chat reads and obeys** -- at session start, on chat-series handoff, before specific actions, or as authoritative reference for project state.

Meta-docs are NOT:
- Briefs (`claude/handoffs/BRIEF_*.md`) -- work specs Chat writes for Code
- Retros (`claude/retros/RETRO_*.md`) -- post-execution reports Code writes for Chat
- Pure-Praxis-domain content like trading conventions, findings, multi-phase plans -- those live under `docs/`, not `claude/`, because they are project content not Claude-behavior content

Meta-docs are the durable rules and state Chat consults to decide *what* to do and *how* to do it.

---

## The Praxis meta-doc set

Order is roughly "read at session start" then "read on demand."

### Read at session start (always)

| Doc | Purpose | What goes in it |
|---|---|---|
| `claude/META_DOCS.md` | This file. Defines the meta-doc convention itself. | Convention rules, doc inventory, memory model, sync notes. |
| `claude/chat/CLAUDE_CHAT_CONVENTIONS.md` | Conventions hierarchy: defaults -> series section -> chat section -> standing overrides. The single source of truth for "how Chat operates on this repo." | Naming rules, zip checklist, override list, per-chat sections, version footer. |
| `claude/chat/NEW_CHAT_README.md` | Onboarding procedure -- the literal reading list for a new chat in the series, plus quick-start checklist. | Reading list, archived-chats table, mandatory critical overrides. |
| `claude/chat/CHAT_WORKAROUNDS.md` | Known platform limitations and the workarounds Chat applies before claiming a limitation. | Workarounds for Claude.ai chat platform quirks. |
| `claude/CLAUDE_CODE_RULES.md` | The dual-Claude protocol -- Chat=Strategist, Code=Implementer, Briefs/Retros conventions, 41 numbered rules. | Roles, brief format, retro format, do-not-do list for each role. |
| `claude/WORKFLOW_MODES_PRAXIS.md` | Mode arbitration: when Chat edits directly (delta zip) vs. when Code implements. | Mode A vs. Mode B selection criteria, instructions per mode. |
| `claude/TODO.md` | Praxis-specific outstanding work, deferred items, multi-cycle context. | Anything that would otherwise have been a Praxis-tagged memory entry but isn't a cross-project principle. |
| `docs/praxis_main_series.md` | Project genesis, architecture vision, implementation history, chat series index. | Long-form project narrative + chat-series-index table. |
| `docs/praxis_main_series_transcript.md` | Chronological conversation history with key decisions. | Exchange-by-exchange log of strategic conversations. |

### Read on demand (when relevant)

| Doc | Purpose | When Chat reads it |
|---|---|---|
| `claude/SERVICE_CHEATSHEET.md` | Quick command reference for running services. | When asked how to start/stop something or what port a service is on. |
| `docs/TRADING_CONVENTIONS.md` | Pure Praxis domain rules: Polymarket oracle check, weather bet airport stations, etc. | Before any trading work, or when designing/reviewing a strategy. |
| `docs/TRADING_FINDINGS.md` | Empirical findings about market structure (e.g., NegRisk arb is non-executable). | When a strategy idea overlaps with documented prior findings. |
| `docs/MEV_BUILD_PLAN.md` | The 5-phase MEV roadmap. | When planning MEV work. |
| `docs/REGIME_MATRIX.md` | The 13-class CPO regime matrix. | When working on regime detection or regime-aware strategies. |
| `docs/ATLAS_DB.md` | Atlas DB workflow + schema. | When working on Atlas, atlas_sync, or the atlas MCP tools. |

### NOT meta-docs (despite living under `claude/`)

Reference material under `claude/` that is NOT behavioral rules:
- `claude/handoffs/BRIEF_*.md` -- work specs Chat writes for Code
- `claude/retros/RETRO_*.md` -- post-execution reports Code writes for Chat
- `claude/scratch/*.py` -- diagnostic helpers (gitignored)

---

## Memory vs. meta-docs -- what goes where

### In Chat's cross-session memory (McTheory Project pool, currently ~7 entries used of 30 cap)

**Cross-project principles only.** A memory entry should apply identically whether Chat is working on `praxis`, `ai-agent-factory`, or `core`. The repo name should never appear in a memory entry except as a parenthetical example.

Current keepers (post-Cycle-16 cleanup):
1. Delta zips only, never full project zips
2. Never include `.env`, `*.db`, or DB files in zips
3. Novel-mechanics check -- don't accept tacit confirmation on complex/novel work
4. Always load secrets via `python-dotenv` (use `load_dotenv()` before any API key access)
5. Validation-first -- `--validate` and `--verbose` defaults max
6. "Everything is a parameter" design principle
7. **Pointer entry** -- "Per-repo state lives in `claude/META_DOCS.md` and the docs it indexes; read at session start. Memory holds only cross-project principles."

Anything Praxis-specific that was previously in memory has been moved to `claude/TODO.md` or to one of the `docs/` files listed above.

### In `claude/TODO.md`

Anything Praxis-specific that previously would have been a memory TODO:
- Outstanding work items (build X, integrate Y, alert when Z)
- Future enhancements with rough ordering
- Deferred-from-brief items
- Things to remember when next touching specific subsystems
- Cross-cycle context that Code/Chat needs but doesn't fit a Brief

`claude/TODO.md` is **about Claude's work** on Praxis, not about Praxis content itself. The distinction matters: "build a market-maker bot" is a TODO (work to do); "Polymarket oracle source determines payout terms" is project content (lives in `docs/TRADING_CONVENTIONS.md`).

### In `docs/TRADING_CONVENTIONS.md`

Pure Praxis-domain operating rules -- things that are true about the markets / domain regardless of who or what is acting:
- Polymarket resolution oracle verification rules
- Weather bet airport station gotchas (KLGA not JFK, etc.)
- Future: any other market-mechanics rules

### In `docs/TRADING_FINDINGS.md`

Empirical findings about market structure that should be respected by future work:
- NegRisk arb non-executable (illiquid placeholder outcomes)
- Future: any other "we tried this and it doesn't work because X" findings

### In `docs/MEV_BUILD_PLAN.md`

The 5-phase ordered roadmap for MEV work. Standalone because of size and ongoing relevance; future phases will likely add detail in place.

---

## Reading discipline

**At chat start in Praxis:**
1. Read every doc in the "Read at session start" table above, in order.
2. Confirm to Jeff: "I've read the conventions and am ready to work on praxis_main_current."

**On chat-series handoff (creating a new chat in the series):**
1. Apply the conventions from `CLAUDE_CHAT_CONVENTIONS.md` -- archive previous chat in the chat-section table, add new active section.
2. Update `docs/praxis_main_series.md` chat-series index.
3. Update `docs/praxis_main_series_transcript.md` with handoff entry.
4. Update `claude/chat/NEW_CHAT_README.md` archived-chats table.
5. Bump version footers on every file edited.

**Before claiming a platform limitation:**
1. Read `claude/chat/CHAT_WORKAROUNDS.md` first.

**Before assuming "I don't know about that":**
1. Check `claude/TODO.md`, `docs/praxis_main_series_transcript.md`, and the relevant docs/*.md files.

---

## Synchronization across repos

**Convention rule:** memory entries are universal -- they apply to every repo in McTheory. Repo-specific implementations of those principles live in each repo's meta-docs.

When a meta-doc convention is introduced in one repo (e.g., "we now use `claude/TODO.md` for repo-specific TODOs"), it must be propagated to all other McTheory repos -- in their structure, even if filenames differ.

The naming convention `<project>_main`, `<project>_main_current`, `<project>_main_series.md`, etc. is a wildcard -- `ai_factory_main_current`, `praxis_main_current`, `core_main_current` (when created) all follow identical patterns.

### Memory partitioning gotcha

Memory is partitioned **per Anthropic Claude Project**. If a chat is created outside the McTheory Project, it sees a different memory pool. Symptoms: memory view doesn't match what other McTheory chats see, and `memory_user_edits` writes don't propagate.

If this happens, move the chat into McTheory via the chat's three-dot menu -> "Add to project." Do not try to bulk-fix memory by re-typing entries -- they'll just create a parallel pool that diverges further.

---

## Memory state at end of Cycle 16

```
Before Cycle 16 (2026-04-30 mid-day):  25 entries
After Cycle 16:                          7 entries
Free slots:                             23

Remaining entries (cross-project only):
  1. Delta zips only
  2. Never include .env / .db / data files in zips
  3. Novel-mechanics check
  4. Use load_dotenv() before any API key access (was Praxis-tagged; rewritten to cross-project)
  5. Validation-first, --validate / --verbose default max
  6. "Everything is a parameter" design principle
  7. Pointer entry to META_DOCS.md
```

The 18 entries that were Praxis-specific have been moved to `claude/TODO.md`, `docs/TRADING_CONVENTIONS.md`, `docs/TRADING_FINDINGS.md`, or `docs/MEV_BUILD_PLAN.md` per the classification in this cycle's retro.

---

## Revision history

| Date | Cycle | Change |
|---|---|---|
| 2026-04-30 | Cycle 16 | Initial Praxis-side META_DOCS.md. Established meta-doc convention, classified 18 Praxis-specific memory entries into appropriate destinations (`claude/TODO.md` for Claude's work queue; `docs/TRADING_CONVENTIONS.md` for pure-Praxis operating rules; `docs/TRADING_FINDINGS.md` for empirical findings; `docs/MEV_BUILD_PLAN.md` for the multi-phase MEV roadmap). Memory pruned from 25 entries to 7 cross-project keepers. AI Factory side authored a parallel `META_DOCS.md` in its repo earlier same day. |
