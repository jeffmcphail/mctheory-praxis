# Retro: push rule -- Code pushes when Jeff says so

**Date:** 2026-11-20 (EST)
**Brief:** `claude/handoffs/BRIEF_push_rule.md`, written in Chat and pasted in, transcribed verbatim.
**Machine:** mcrmini1. The brief says mcrmini2, after the move; Code found the session still on
mcrmini1 and held. Jeff's reply, "I'm telling you to run it now on mcrmini1", is in the brief as its
amendment.
**Builds on:** aacd3b2, with midoffice20's 15 files uncommitted (on origin as `wip/midoffice20`,
e4f7e310).

## Summary

- **No file ever stated the rule as a rule.** There was no `CLAUDE.md` and no numbered rule in
  `CLAUDE_CODE_RULES.md`. "Jeff runs every push" lived in Code's auto memory, and in one clause of
  `claude/MIDOFFICE_STATE.md`.
- **The new rule is `CLAUDE.md`** at the repo root, created with the brief's section verbatim. The
  state-file clause and the auto-memory entry now point to it.
- **The repo's settings were never in the way.** `.claude/settings.local.json` allows `git push` and
  denies only force pushes. That matches the new rule, so it is quoted below and left alone.
- **Origins:**
  - praxis: `https://github.com/jeffmcphail/mctheory-praxis.git`;
  - mctheory-core: `https://github.com/jeffmcphail/mctheory-core.git`;
  - each the same for fetch and push.
- **No push, no commit, no test push.**

## 1. Where the old rule was

**Searched:** push, origin, "Jeff runs" and script, case-insensitive. Hit counts:

| File | push | origin | Jeff runs | script |
|---|---|---|---|---|
| `claude/CLAUDE_CODE_RULES.md` | 0 | 0 | 0 | 17 |
| `claude/WORKFLOW_MODES_PRAXIS.md` | 1 | 0 | 0 | 7 |
| `claude/chat/CLAUDE_CHAT_CONVENTIONS.md` | 6 | 1 | 1 | 12 |
| `claude/chat/NEW_CHAT_README.md` | 0 | 0 | 0 | 12 |
| `claude/chat/CHAT_WORKAROUNDS.md` | 1 | 0 | 0 | 1 |
| `claude/META_DOCS.md` | 0 | 0 | 0 | 3 |
| `claude/MIDOFFICE_STATE.md` | 1 | 0 | 0 | 30 |
| `claude/TODO.md` | 0 | 1 | 0 | 15 |
| `claude/SERVICE_CHEATSHEET.md` | 0 | 0 | 0 | 11 |
| `.claude/commands/` (brief, commit, retro, syntax-check) | 0 | 0 | 0 | 1 |
| `.claude/settings.local.json` | 4 | 1 | 0 | 0 |

**None of the "script" hits is about pushing, or about handing Jeff a script.** They are paths
(`scripts/...`), guard scripts and `split_zip.py`.

**Absent:**
- in the repo: `CLAUDE.md`, `.claude/CLAUDE.md`, `CLAUDE.local.md`, `.claude/rules/` and
  `.claude/settings.json`;
- at user level: `~/.claude/CLAUDE.md` and `~/.claude/rules/`.

No file under `claude/scratch/` contains "git push".

**The hits that state or bear on the rule:**

- `claude/MIDOFFICE_STATE.md:22`, the only repo text stating it:
  `   - **The commit.** Jeff commits with Chat's message; Code never commits or pushes. The next`
- **Auto memory, `MEMORY.md:2`:**
  `- [Don't push commits — leave local](feedback_git_push_local.md) — after commit + SHA follow-up, stop; Jeff pushes to origin himself at the next session start. Don't re-ask each cycle.`
- **Auto memory, `feedback_git_push_local.md`:**
  - line 3: `description: "Don't push commits to origin — leave them local; Jeff pushes himself at the start of the next session"`
  - lines 11-13: `After committing cycle work (including the SHA-insertion follow-up), STOP. Do` /
    ``not `git push origin master`, and don't re-ask about pushing each cycle —`` /
    ``leaving commits local on `master` is the default.``
  - line 15: `**Why:** Jeff makes the push to origin himself at the start of the next`
  - lines 22-24: `**How to apply:** Commit + SHA-insertion follow-up, then stop and report the` /
    `local commit SHAs + ahead-count. Only push if Jeff explicitly asks in the` / `moment. ...`
  - lines 26-28: `**Since 2026-11-05 (Rule 43):** by default Code does not commit at all. ...` /
    `"committed". Commit only when asked, and never push.`
- **`.claude/settings.local.json`** (gitignored, so local to this machine):
  - allow, lines 24 and 27: `"Bash(git push origin master)",` and `"Bash(git push:*)",`
  - deny, lines 38 and 39: `"Bash(git push --force*)",` and `"Bash(git push -f*)",`
- **`~/.claude/settings.json`** (user level, applies to every repo here). It has no git
  permissions. Its `autoMode.environment` describes only the factory as trusted:
  - line 31: `"**Source control**: The trusted repo (jeffmcphail/ai-agent-factory) and its origin remote (https://github.com/jeffmcphail/ai-agent-factory.git) only",`
  - line 43: `"**Trusted repo**: C:\\Data\\Development\\Python\\McTheoryApps\\ai-agent-factory and its origin remote (https://github.com/jeffmcphail/ai-agent-factory.git)",`
- **The other push and origin hits don't concern pushing:**
  - `WORKFLOW_MODES_PRAXIS.md:128`, "Jeff can defer (push to 5)";
  - `CHAT_WORKAROUNDS.md:5`, "pushes back";
  - `TODO.md:142`, "original".
  - `CLAUDE_CHAT_CONVENTIONS.md` is in section 4.

**The commit rule is separate, and stands.** Rule 43 (`CLAUDE_CODE_RULES.md:417`) has Jeff commit
and never mentions pushes: "Commit messages come from Chat; the file list comes from Code".

**Auto memory is this machine's own.** The session never left mcrmini1, so the folder neither came
over nor started empty: it is mcrmini1's, at
`~\.claude\projects\C--Data-Development-Python-McTheoryApps-praxis\memory\`. It reaches mcrmini2 only
if the move carries `~/.claude` (the move brief's target 7).

## 2. What changed

- **`CLAUDE.md`** (new, repo root, ASCII): the brief's section "Pushing to GitHub", verbatim,
  without the blockquote markers. Nothing else.
- **`claude/MIDOFFICE_STATE.md:22`:** "Code never commits or pushes." became "Code never commits.
  Pushing: `CLAUDE.md`, "Pushing to GitHub"." The three lines after it are rewrapped and otherwise
  unchanged. The file is still CRLF and ASCII.
- **Auto memory** (outside the repo):
  - `feedback_git_push_local.md` is now one line: "Pushing follows praxis's `CLAUDE.md`, "Pushing to
    GitHub" (BRIEF_push_rule, 2026-11-20)." Its name is kept, so the `[[feedback-git-push-local]]`
    links in other memories still resolve.
  - `MEMORY.md:2` reads "Pushing: see CLAUDE.md".
  - The commit half of the old entry is already in `feedback_retro_every_piece_of_work.md:42`.
- **`claude/handoffs/BRIEF_push_rule.md`:** the brief verbatim, with Jeff's amendment.
- **`CLAUDE_CODE_RULES.md`: not changed.** It has no numbered push rule, so there is no pointer to
  add and no version to bump.

## 3. What I left, and why

- **`.claude/settings.local.json`'s push entries:** quoted above, and left for Jeff. They allow
  pushes and deny force pushes, which is the new rule.
- **`~/.claude/settings.json`'s `autoMode.environment`:** user level, so left. Lines 31 and 43 name
  only the factory's origin as trusted. Auto mode may therefore question a push to mctheory-praxis
  or mctheory-core even when Jeff has said to push. Adding the two repos there is Jeff's call.
- **`claude/MIDOFFICE_STATE.md` is not in this piece's `git add`.** It is one of midoffice20's 15
  uncommitted files, and its diff also carries midoffice20's unfinished state (128+/97-, with `@@`
  placeholders still to fill). The pointer line goes in with midoffice20's commit. Until then
  `CLAUDE.md` carries the rule alone; it loads in every session.
- **Other memories:** `feedback_retro_every_piece_of_work.md:42` is about commits and links to the
  pointer. The "NOT pushed" notes in the project memories are history, not rules:
  - `project_forced_trade_collectors_cycle62a.md:12`;
  - `project_chan_cpo_replication_cycle59.md:46`;
  - `project_engine7_recon_cycle39.md:471, 519, 528`;
  - `project_infobar_directional_cycle58.md:13`.
- **Old briefs and retros:** records, untouched.
- **`claude/META_DOCS.md` doesn't list `CLAUDE.md`.** That wasn't asked; it's a one-line follow-up
  if wanted.

## 4. What depends on the old rule (unchanged)

- **`claude/chat/CLAUDE_CHAT_CONVENTIONS.md`, "Commit After Every Delta Delivery" (lines 147-169),
  is confirmed against the current file.** Block 2 is pasted by Jeff and ends with his `git push`:
  - line 155: "**Block 2 — commits through the push.** Meaningful only once block 1 passed. Safe to
    paste whole. Contains the `git add` / `git commit` pairs, `git status --short`, and `git push`
    as its final line."
  - line 165: "**`git push` is mandatory and last.** Unpushed is the same defect as uncommitted, one
    step later and quieter: ..."
  - line 167: "If a delivery should NOT be pushed, say so explicitly in place of the push line, with
    the reason. ..."
  - line 169: "**Does not apply to:** work Code executes from a brief (Code handles its own commits
    per `CLAUDE_CODE_RULES.md`); ..." That file has Jeff commit (Rule 43) and says nothing on
    pushes, which now live in `CLAUDE.md`.
- **Line 5 of the same file says it is shared:** "> **Note:** This file is shared across all
  McTheory repos. If any chat updates a non-series-specific section, it must push the change to
  the other repos (or ask the user to if it can't)."
- **The factory, read and not edited:**
  - **It has no `CLAUDE.md`.** Its push rule is the same section of its own
    `claude/chat/CLAUDE_CHAT_CONVENTIONS.md` (553-627). Under "Procedure (User/Code side)" (589):
    - line 595: "5. Either the user or Code may execute the commits; this override doesn't specify
      who"
    - line 596: "6. Push. If Chat's block has no `git push` line and no stated reason for
      withholding it, treat that as an omission and ask rather than assuming the work is meant to
      stay local"
    - line 582: "6. **Block 2 ends with `git push` on its own line, always.** ..."
  - **The two versions differ.** Praxis's has no User/Code side, so it never says Code may run the
    commits or the push. The factory's adds:
    - the `Chat: ai_factory_main_<NN>` trailer;
    - the no-`--basetemp` rule;
    - a longer rationale and a changelog.

    Praxis's cites `RECOVERY_PLAN_post_disk_failure.md` instead. The factory's file has no "shared
    across all McTheory repos" note.
  - **Both repos exclude briefed work in the same words:** "Code handles its own commits per
    `CLAUDE_CODE_RULES.md`" (factory line 600).
  - **The factory's tracked `.claude/settings.json`** holds the same four push entries as praxis's
    `settings.local.json` (lines 24-39 identical).

## Checks

- `CLAUDE.md` and the brief are ASCII. This retro is ASCII except for 5 em dashes, all inside
  quotes of `MEMORY.md`, `feedback_git_push_local.md` and the conventions file; they are kept so
  the quotes stay exact. None of the three files is ignored by git.
- The brief's transcription is identical to the paste, checked by `diff` against the copy saved at
  the time.
- `git remote -v`, as quoted in the Summary.
- No push, commit or test push was made.

## git add (always last -- Rule 43)

git add CLAUDE.md claude/handoffs/BRIEF_push_rule.md claude/retros/RETRO_push_rule.md
