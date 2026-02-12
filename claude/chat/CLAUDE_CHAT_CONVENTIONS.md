# Claude Chat Conventions

Instructions for Claude across all chats in this project. Each chat should read this file and adhere to these conventions.

> **Note:** This file is shared across all McTheory repos. If any chat updates a non-series-specific section, it must push the change to the other repos (or ask the user to if it can't).

---

## Chat Naming Framework

**Pattern:** `praxis_<series>_<i>_<tag>`

| Component | Purpose | Examples |
|-----------|---------|----------|
| `<series>` | The "branch" type | `main`, `spec`, `research`, `spike` |
| `<i>` | Iteration index | `1`, `2`, `3`... |
| `<tag>` | Short descriptor of focus/epoch | `phase0_spikes`, `phase1_keys`, `current` |

**Examples:**
- `praxis_main_1_phase0_spikes` — Main implementation, iteration 1, Phase 0 spike work
- `praxis_main_current` — Main implementation, currently active
- `praxis_spec_9_v9.3.1` — Specification series, iteration 9, version 9.3.1

When a chat becomes too large, create a new chat with the same series but incremented index. The new chat should gather context from the previous iteration(s) as needed. Archived chats keep their sections for reference.

---

## Conflict Resolution Hierarchy

1. **Overrides** — Highest priority, always wins
2. **Chat Section** — Individual chat-specific rules (e.g., `praxis_main_current`)
3. **Series Section** — Shared rules for chat series (e.g., `praxis_main`)
4. **Default** — Baseline fallback when not specified elsewhere

---

## Overrides

_Rules that override everything else. Use sparingly._

### Auto-Remove .env from Uploaded Zips

**Whenever the user uploads a project zip, Claude must sync .env.example then remove .env before doing anything else.**

This protects secrets from being stored in Claude's context or project files while keeping the template up to date.

**Procedure:**
1. After extracting an uploaded zip, compare `.env` structure against `.env.example`
2. If `.env` has new keys not in `.env.example`, add them to `.env.example` with placeholder values
3. If `.env` has removed keys that are in `.env.example`, remove them from `.env.example`
4. Do NOT copy actual secret values — only sync the key names and structure
5. Run: `rm -f /path/to/project/.env`
6. Continue with the task

*Added 2026-01-31: User preference to avoid secrets exposure and simplify zip workflow, while keeping .env.example in sync.*

### MANDATORY: Output Zip Creation Checklist

**Before creating ANY project zip for the user, Claude MUST execute this checklist and state completion explicitly.**

This is non-negotiable. Skipping this checklist overwrites user's API keys and credentials.

**Checklist:**
1. Run: `rm -f /path/to/project/.env`
2. Verify: `ls -la /path/to/project/.env` (should show "No such file")
3. Include `-x ".env"` flag in zip command
4. After creating zip, verify with: `unzip -l /path/to/zip | grep ".env"` (should only show `.env.example`)
5. State explicitly in response: "✅ Verified: .env excluded from zip"

**If any step fails, STOP and fix before delivering the zip.**

### Solution Quality Over Expediency

When implementing solutions, prefer well-thought-out approaches over expedient ones. Verify that the approach actually serves the core purpose before implementing. Don't just make tests pass — make the architecture right.

### Workaround-First

Check `claude/chat/CHAT_WORKAROUNDS.md` before claiming "I can't do X." Platform limitations often have documented solutions.

### Defensive Development

Make errors self-diagnosing with clear error messages. Document everything — future chats depend on your documentation.

### Markdown Versioning

Include version marker at bottom of all `.md` files.

This allows quick verification of which version you're looking at across chats.

---

## Default

_Baseline conventions. Apply unless overridden by series section, chat section, or overrides._

### Credentials & Configuration

- **Username/password credentials** → `.env` only (never JSON files, never `set_credentials()` methods)
- **OAuth services** (Google, etc.) → Follow their required protocol (JSON files for OAuth flow are acceptable)
- **One pattern per concern** - Avoid multiple ways to do the same thing; pick one and stick with it

### Code Style

- Always regenerate and present the full project zip after ANY file changes, even single files
- User prefers complete snapshots to avoid sync issues
- Any module that requires environment variables must auto-load `.env` using dotenv:
  ```python
  try:
      from dotenv import load_dotenv
      load_dotenv()
  except ImportError:
      pass
  ```

### Testing

- All tests must pass before delivering updates
- Include tests for new functionality

### Documentation

- Update relevant docs (README, series docs, etc.) when adding features
- Keep setup instructions simple and consistent

---

## Series Sections

_Shared rules for all chats within a series. Apply unless overridden by specific chat section or overrides._

### praxis_main

_Main implementation series — the primary development branch for McTheory Praxis._

**Scope:** All `praxis_main_*` chats share these conventions.

**Specific conventions:**
- This series owns the canonical project state
- Spec v9.3.1 and execution plan v1.2 are the governing documents (in Claude Project files)
- Reference files (dataUtilities.py, statsUtilities.py, main.py) are for domain logic understanding only — NOT code to port
- All implementations are fresh builds in Polars/modern patterns, validated against reference outputs
- DuckDB STRUCTs over JSON for model definitions (validated by Spike 2)
- Temporal views (vew_, vt2_, rpt_) — never direct table access
- Universal key pattern (_bpk/_base_id/_hist_id) on every dimension table
- When presenting file changes, always regenerate the full zip

---

## Chat Sections

_Individual chat-specific rules. Archived chats retain their sections for reference._

### praxis_main_current

_Active: Main implementation, current iteration. Phase 0 spikes + Phase 1 implementation._

**Status:** Active — SOURCE OF TRUTH for project files.

**Specific conventions:**
- Inherited from `praxis_main` series section
- (Add chat-specific conventions here as they emerge)

---

## How to Use This File

### For Claude (in any chat)

1. Read this file at the start of significant work
2. Follow the hierarchy: Overrides > Chat Section > Series Section > Default
3. When user establishes a new convention, add it to the appropriate section
4. When conventions conflict, ask user which should take precedence

### For Adding New Chat Sections

When a chat becomes too large and increments:
1. Mark the old chat section as "Archived"
2. Create a new section for the new chat
3. The new chat inherits from its series section

```markdown
### praxis_<series>_<i>_<tag>

_Brief description of chat focus_

**Status:** Active / Archived

**Specific conventions:**
- Convention 1
- Convention 2
```

### For Adding New Series Sections

When creating a new branch of work (e.g., `praxis_research_*`):

```markdown
### praxis_<series>

_Description of this series' purpose_

**Scope:** All `praxis_<series>_*` chats share these conventions.

**Specific conventions:**
- Convention 1
- Convention 2
```

### For Adding Overrides

Only add to Overrides when:
- A convention should apply to ALL chats unconditionally
- You've encountered conflicts that need a definitive resolution
- User explicitly says "this should always be true"

---
*Last updated: 2026-02-12 (Chat: praxis_main_current)*
