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

### Delta-Only Zip Delivery

**Routine deliveries are a delta zip and nothing else** — only the changed files, preserving repo-relative paths so the archive extracts straight to the project root.

**Praxis-specific stakes.** This repo carries more unrecoverable local state than most:

| Path | Size / nature | What a full zip would do |
|---|---|---|
| `data/crypto_data.db` | ~25 GB, WAL mode, written continuously by live collectors | Overwrite or corrupt the entire research dataset |
| `data/external/kibot/` | Licensed vendor data, paid for once | Redistribute licensed files; unrecoverable if clobbered |
| `data/external/kibot/*.parquet` | ~121 MB each, derived panels | Waste, and stale panels silently replacing rebuilt ones |
| `live_collector.db`, `smart_money.db` | Sidecar runtime state | Revert collector progress |

`.gitignore` covers all of these (`data/` at line 137, `*.db` at line 160), and `split_zip.py zip` respects `.gitignore`, so the tool is safe by construction. The rule exists for the manual case and for the second, subtler hazard: **sandbox-lag clobber.** A file rebuilt from a stale extraction silently reverts committed work — nothing errors, nothing fails a test, the code simply does less than it says.

**Procedure:**
1. Never include `.env`, `*.db`, or anything under `data/`. Verify by inspecting the archive (`unzip -l | grep`), never by inspecting the working tree.
2. Deltas must use correct repo-path directory structure.
3. Flag explicitly when a delta UPDATES an existing file, so the extract-or-skip decision is conscious.
4. Jeff extracts delta zips himself via Windows Explorer (double-click → extract → copy). **Do NOT give zip-extraction commands** — no `tar`, no `Expand-Archive`. Deliver the zip, then git/run commands after extraction.

**The two cases where a full repo zip IS correct:**
1. **Chat-series handoff** — the successor chat needs a complete current tree.
2. **Suspected gap** — there is reason to believe the sandbox copy is behind, and a clean snapshot is cheaper than guessing which files are stale.

Outside these, no full zip unless Jeff explicitly asks — and then reconfirm before building it.

### `split_zip.py` — the delta tool

Lives at the repo root. Stdlib only, no install. Copied from the AI Factory repo 2026-08-18; `BRIEF_DIR` verified to be `claude/handoffs`, which already matches Praxis, so **the script needs no modification**.

```
.venv\Scripts\python split_zip.py zip                     # full repo, respects .gitignore
.venv\Scripts\python split_zip.py zip --repo-delta        # changed since the latest BRIEF_*.md landed
.venv\Scripts\python split_zip.py zip --repo-delta <stem> # delta from a specific brief
.venv\Scripts\python split_zip.py split <zip> --chunk-mb 40
.venv\Scripts\python split_zip.py bundle                  # zip + split
```

Structural exclusions are built in: `.env` (unconditionally, even if not gitignored), `.git/`, and root `*.zip`. Output is `praxis_delta_<brief_stem>.zip` in the repo root — a fresh timestamp is how Jeff confirms it was actually produced rather than a stale one being picked up.

**The `--repo-delta` anchor is the brief, and that is deliberate.** The tool resolves the anchor by walking `git log --diff-filter=A` over `claude/handoffs/` and accepting **only** files matching `BRIEF_*.md`. A handoff named anything else can never anchor a delta. This is not a limitation to work around — it is the enforcement mechanism for brief discipline (see below).

### Brief and Retro naming is mandatory, and it is what makes deltas work

Naming is specified in `claude/CLAUDE_CODE_RULES.md`: briefs are `claude/handoffs/BRIEF_<slug>.md`, retros are `claude/retros/RETRO_<slug>.md` with a **matching slug**. The pair is the unit of record; reading `handoffs/` alongside `retros/` should let a future chat reconstruct the project's evolution.

**Both are permanent.** Do not delete briefs to tidy the directory — it is a historical record, not a todo list. Retros are never deleted.

**The failure this rule caught, recorded because it was Chat's:** as of 2026-08-18, 64 of 73 handoffs were correctly named `BRIEF_*` and 59 of 60 retros `RETRO_*` — the convention held for dozens of cycles. Then Cycles 55–60 produced handoffs named `CYCLE55_REAL_MONEY_RECON.md`, `CYCLE58_VALIDATION_PROTOCOL.md`, `CYCLE59_CHAN_CPO_PREREGISTRATION.md`, `CYCLE60_XSEC_REVERSAL_PREREG.md` and similar, and Cycles 55–59 produced **no retro at all**. The convention was written down in a repo file Chat had been handed and had not read.

The consequence was not cosmetic. The most recent anchorable brief was Cycle 53's, so `--repo-delta` produced a 102-file archive spanning eight cycles instead of one cycle's work — and that stale anchor was the **only** signal that the brief/retro pairing had lapsed. Nothing else complained. Treat a suspiciously large delta as evidence of missing briefs, not as a tooling defect.

**A pre-registration is a brief.** Cycle 59 legitimately needed two (`BRIEF_chan_cpo_phase2_kibot_load`, `BRIEF_chan_cpo_prereg`); multi-part cycles use the sub-letter form already in use (`RETRO_cycle54b_*`, `RETRO_cycle54c_*`).

Existing `CYCLE*`-named handoffs are **left in place as historical artifacts** and not retroactively renamed — renaming would fabricate a brief that never existed. The convention applies going forward.

### Commit After Every Delta Delivery

**Every delta delivery ends with a Commit Instructions block as the LAST item of the message.** The next action after extraction is a commit, before other work proceeds. Without it, deltas accumulate dirty in the working tree and any later brief that assumes "prior fixes are merged" fails the moment Code inspects `git status`.

**TWO code blocks, never one.**

**Block 1 — validation only.** Tests, guard scripts, `git diff --stat`. **No `git add`, no `git commit`, no `git push` may appear in it.** State the pass condition in plain words immediately after — "expect 0 failed", the predicted diffstat — so there is something to check the output *against*.

**Block 2 — commits through the push.** Meaningful only once block 1 passed. Safe to paste whole. Contains the `git add` / `git commit` pairs, `git status --short`, and `git push` as its final line.

**Why they are separate:** a single block invites one paste, and then the validation step has no power — its whole purpose is to be a decision point, and a decision point in the middle of a paste-able block is not one.

**Commit message rules:**
- One commit per logical change; bundle only when genuinely inseparable.
- Conventional prefixes: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`.
- **ASCII only.** PowerShell uses cp1252 — em dashes, BOM, and Unicode corrupt the message.
- Bodies explain WHY, not WHAT (the diff shows what).

**`git push` is mandatory and last.** Unpushed is the same defect as uncommitted, one step later and quieter: `git status` is clean, `git log` shows the work, `--repo-delta` picks it up, and a successor chat sees nothing wrong. Only `origin` is missing — and `origin` is the off-machine copy. This repo has already been recovered from disk failure once (`RECOVERY_PLAN_post_disk_failure.md`); anything unpushed that day was gone. One push after all commits, not one per commit, on its own final line with nothing after it.

If a delivery should NOT be pushed, say so explicitly in place of the push line, with the reason. Silence is not an instruction to hold — a block that omits the push is indistinguishable from one that forgot it.

**Does not apply to:** work Code executes from a brief (Code handles its own commits per `CLAUDE_CODE_RULES.md`); pure discussion messages with no delta. Doc-only deltas still commit, prefixed `docs:`.

### Stale-File Protocol

**Before editing any file not received from Jeff in the CURRENT exchange, name the files needed, request a `--repo-delta`, and WAIT.**

Not "probably fine." Not "I'll caveat it." Name them, ask, wait.

**The criterion is chain of custody, not literal recency.** Chat may edit a file whose every change since it last saw it is accountable — one Jeff sent last exchange that only Chat has edited since. Chat may NOT edit a file where Code, a commit, or another cycle could have changed it unseen.

**Every brief's final step closes this loop:** brief instructs Code to run `.venv\Scripts\python split_zip.py zip --repo-delta`, so the delta exists before Chat needs it. That step is standing, not optional, and Chat includes it without being asked.

**Chat audits its own brief sequence.** Delta filenames carry the brief stem, so the deltas received should line up with the briefs written. A brief with no corresponding delta means that cycle's work is in the repo and invisible to Chat — and every file it touched is a clobber waiting to happen. On spotting a gap, ask for that brief by name rather than requesting a full snapshot.

**The tell:** `git diff --stat` showing deletions Chat cannot individually name. State expected insertions AND deletions before Jeff runs the diff; any unexplained deletion halts the cycle.

**Surgical edits do not exempt.** A careful one-line change to a full file copied from a stale tree is exactly how this fails.

**Praxis instance, 2026-08-18:** Chat asserted in a brief that `engines/chan_cpo/` "was built and unit-tested in Phase 1" and told Code the schema fix was "a one-line change" — pointing at a package that had never been committed. Chat had produced the zip weeks earlier, the commit commands were never run, and Chat never verified. Code discovered the absence and rebuilt from scratch, correctly.

### Verify the edit landed — a string replace that matches nothing is silent

**Any programmatic edit must assert that it changed something.** A `str.replace` whose pattern does not match returns the input unchanged, writes a valid file, and reports success.

**Praxis instance, 2026-08-18:** a fix for a Deflated-Sharpe call used `stats['n']` while the file contained `stats["n"]`. The replace was a no-op, the patched file was packaged, committed, and shipped, and the DSR step silently skipped for a second consecutive run — the multiple-testing correction pre-registered as a success criterion never executed. The same class of error had been flagged in the same session.

Assert the match, or grep the result and show it.

### A control that cannot fail is not a control

**Every pre-registered control must be paired with a check that it CAN change the output.**

**Praxis instance, Cycle 60:** the pre-registered defence against "this edge is just beta dispersion" was `residualize="demean"` versus `"none"`. Cross-sectional demeaning subtracts the same scalar from every symbol at a timestamp, and positions are built by **ranking** the signal — ranking is invariant to a constant shift. The two arms were arithmetically identical: best and median net Sharpe matched to three decimals in all three tiers. The control was inert for the entire experiment, and the alternative hypothesis was never tested. It was visible only because the report printed both arms; a single summary statistic would have hidden it.

**Corollary:** identical numbers across arms are a red flag, never a result.

### Message shape: analysis and action are separated, and action comes last

1. **The actionable block is last.** Nothing follows it.
2. **Analysis contains no actions.** Jeff must be able to jump to the bottom and act without reading a word above it.

**Body** — findings, evidence, reasoning. Any length; verbose is wanted. It contains no instructions.

**Actionable block** — last, headed, and exactly one of **What I need from you** (imperative steps), **Decision needed** (options restated in full as bullets), or **What I'll do next** (a statement, nothing for him to do).

Rules for that block: imperative mood; self-contained (never "which option do you want" referring upward); literal values, not placeholders; say which machine and which file; a done-condition; **preconditions go INSIDE the block as its first command, never as prose beneath it** — text below a block is read after the block has been run. One owner per block: Jeff's tasks, Code's tasks and Chat's plan are never interleaved. Commit Instructions are the actionable block when a delta ships.

### Every command states its machine, and its filter matches that shell

Praxis work spans **PowerShell on Jeff's Windows box** and **bash on Linux/WSL/VM**. `grep` does not exist in the first; `Select-String` does not exist in the second. A command is wrong if its filter does not match the shell it will execute in.

- Say which machine before the command.
- PowerShell: `Select-String`, backtick continuation, `.venv\Scripts\Activate.ps1`.
- **Jeff's keyboard has no Windows key** — use Ctrl+Esc or right-click Start, never Win+X / Win+R / Win+I / Win+D.
- Give complete, copy-paste-ready commands. Never tell Jeff to "verify", "commit" or "diff" without the exact literal command; never make him construct a git command himself.
- Long log output gets a filter: pipe through `Select-String` with `-Context` rather than asking him to read 2,000 lines.

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
*Last updated: 2026-08-18 (Chat: praxis_main_current)*
*Changes: Recorded the arrival of split_zip.py at the repo root (BRIEF_DIR verified
as claude/handoffs; no modification needed) and adapted the AI Factory delivery policy
to Praxis. New overrides: Delta-Only Zip Delivery with this repo-specific data hazards
(25 GB crypto_data.db, licensed Kibot files, derived parquet panels); split_zip.py usage;
Brief/Retro naming as mandatory and as the delta anchor, recording the Cycles 55-60 lapse
that produced a 102-file eight-cycle delta; Commit After Every Delta Delivery with the
two-block validate-then-commit rule and mandatory trailing git push; Stale-File Protocol;
verify-the-edit-landed after a silent no-op string replace shipped twice; a control that
cannot fail is not a control, from Cycle 60s inert demean arm; message shape with the
actionable block last; per-machine command and shell-filter rules. Each rule is grounded
in a Praxis failure rather than an imported example.*
