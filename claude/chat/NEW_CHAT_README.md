# NEW_CHAT_README.md — Onboarding Guide for praxis_main Chat Iterations

> **Purpose:** This document provides everything a new Claude chat instance needs to get up to speed on the McTheory Praxis project within the `praxis_main` series. Follow the steps below in order.

---

## 1. Inception Instructions (Copy-Paste Prompt for New Chats)

When starting a new chat in the `praxis_main` series, use this onboarding prompt (adjust the predecessor chat name):

```
Ok first I want you to get up to speed on this project. Read all of the md files in 
the repo, but especially focus on claude/chat/CLAUDE_CHAT_CONVENTIONS.md, docs/praxis_main_series.md 
and praxis_main_series_transcript.md. You are in the chat series praxis_main 
so need to get up to speed.

The last chat that was called praxis_main_current (your current name) is the chat 
now called praxis_main_<N>_<tag>. But at the time of the creation of the docs you 
are reading it had your name (as it was the current chat I was working on, but now that 
is you).

So, once you are up to speed, you will see that you have to try to update the 
praxis_main_series.md and praxis_main_series_transcript.md as described in 
CLAUDE_CHAT_CONVENTIONS.md.
```

Replace `<N>_<tag>` with the predecessor chat's final archived name.

---

## 2. Required Reading (In Order)

Upload the project zip, then read these files in this exact sequence:

| Priority | File | What It Contains |
|----------|------|------------------|
| **1st** | `claude/chat/CLAUDE_CHAT_CONVENTIONS.md` | Conventions hierarchy, overrides, naming rules, zip checklist |
| **2nd** | `docs/praxis_main_series.md` | Project overview, architecture, implementation phases, spike results |
| **3rd** | `docs/praxis_main_series_transcript.md` | Chronological decision log with key findings |
| **4th** | `claude/chat/NEW_CHAT_README.md` | This file — onboarding procedures |
| **5th** | `claude/chat/CHAT_WORKAROUNDS.md` | Known platform limitations and solutions |
| **6th** | `/mnt/transcripts/journal.txt` | Transcript catalog from all compacted chat sessions |

After these core documents, scan for architectural context:

- `spikes/spike_01_results.md` — vt2_ view performance validation
- `spikes/spike_02_results.md` — DuckDB STRUCT handling validation
- Spec v9.3.1 and execution plan v1.2 in Claude Project files

---

## 3. Chat Naming Convention

**Active chat pattern:** `praxis_<series>_current` (never has an index)

**Archived chat pattern:** `praxis_<series>_<i>_<tag>` (index assigned at retirement)

| Component | Description | Examples |
|-----------|-------------|----------|
| `<series>` | Branch type | `main`, `spec`, `research` |
| `<i>` | Iteration index (assigned at retirement, sequential) | `1`, `2`, `3` |
| `<tag>` | Short descriptor of work done | `phase0_spikes`, `phase1_keys`, `phase2_secmaster` |

**Lifecycle:**
1. The active chat is **always** called `praxis_<series>_current` — it NEVER has an index
2. When archived/retired, it gets renamed to `praxis_<series>_<N>_<tag>`
3. The **active** chat is always the SOURCE OF TRUTH for the project

**Archived chats in this series:**

| Chat Name | Status | Description |
|-----------|--------|-------------|
| `praxis_main_current` | **Active** (SOURCE OF TRUTH) | Phase 0 spikes + Phase 1 implementation |

> **Note:** Update this table when creating new chats.

---

## 4. Conventions Hierarchy

Always resolve conflicts using this priority order:

```
Overrides (highest priority)
    ↓
Chat Section conventions
    ↓
Series Section conventions
    ↓
Default conventions (lowest priority)
```

---

## 5. Critical Overrides (ALWAYS Apply)

### Override 1: Auto-Remove .env
- Sync `.env.example` structure with any new env vars
- Then immediately: `rm -f .env`
- **Never** include `.env` in zips — it overwrites Jeff's API keys

### Override 2: MANDATORY Zip Checklist
Before every zip delivery, execute ALL steps:
1. `rm -f .env` (delete .env)
2. `ls -la .env` (verify deletion)
3. Use `-x ".env"` flag in zip command
4. Verify `.env` is NOT in the zip file listing
5. State: "✅ Verified: .env excluded from zip"

### Override 3: Solution Quality
- Prefer well-thought-out solutions over expedient ones
- Verify approach serves core purpose before implementing
- Don't just make tests pass — make the architecture right

### Override 4: Workaround-First
- Check `claude/chat/CHAT_WORKAROUNDS.md` before claiming "I can't do X"

### Override 5: Defensive Development
- Make errors self-diagnosing with clear error messages
- Document everything — future chats depend on your documentation

### Override 6: Markdown Versioning
- Include version marker at bottom of all `.md` files

---

## 6. Context Transfer Protocol

When a chat is about to end or be replaced:

### Outgoing Chat Responsibilities
1. Update `docs/praxis_main_series.md` with work completed
2. Update `docs/praxis_main_series_transcript.md` with key decisions
3. Generate final zip with all changes
4. Document any in-progress work

### Incoming Chat Responsibilities
1. Follow the Required Reading list (Section 2 above)
2. Check `/mnt/transcripts/journal.txt` for recent transcripts
3. Read the most recent 2-3 transcripts for current context
4. Verify the test suite passes: `pytest`
5. Update series documentation to reflect new chat creation
6. Acknowledge conventions hierarchy and overrides

---

## 7. Quick Start Checklist for New Chats

- [ ] Receive project zip upload from Jeff
- [ ] Read `claude/chat/CLAUDE_CHAT_CONVENTIONS.md`
- [ ] Read `docs/praxis_main_series.md`
- [ ] Read `docs/praxis_main_series_transcript.md`
- [ ] Read `claude/chat/CHAT_WORKAROUNDS.md`
- [ ] Check `/mnt/transcripts/journal.txt`
- [ ] Sync `.env.example` and `rm -f .env`
- [ ] Run `pytest` to verify baseline
- [ ] Update series docs with new chat creation
- [ ] Acknowledge: "I've read the conventions and am ready to work on praxis_main"

---

*v1.0 — Created 2026-02-12*
