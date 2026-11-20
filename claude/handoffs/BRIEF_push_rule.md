# Brief: push rule -- Code pushes when Jeff says so

**Source:** written in Chat and pasted into the Claude Code session (the midoffice20 session,
73727206) shortly before 2026-11-20T15:16Z (10:16 EST), the first clock reading after it;
transcribed verbatim below, the paste wrapper stripped, with the message that amended it after it.
**Retro:** `claude/retros/RETRO_push_rule.md`
**Builds on:** aacd3b2, with midoffice20's 15 files uncommitted (saved on origin as
`wip/midoffice20`, e4f7e310).

---

```text
# BRIEF: push rule -- Code pushes when Jeff says so

Save this brief as claude/handoffs/BRIEF_push_rule.md, then carry it out. It runs on mcrmini2, after the move.

## Why

Jeff wants Praxis to match the factory and Rivet: Code pushes to origin when he says so, in a message or in a brief, and never otherwise. Before the move, Praxis's Code said its standing rule was that Jeff runs every push, and handed him a script instead of pushing.

## 1. Find the old rule

Change nothing in this step. Look everywhere a rule for this repo's Code sessions can live, and quote each hit in the retro (file, line, exact text):

- the repo: CLAUDE.md, .claude/CLAUDE.md, CLAUDE.local.md, .claude/rules/, .claude/commands/, .claude/settings.json and .claude/settings.local.json (a permission that denies git push counts);
- the meta-docs: claude/CLAUDE_CODE_RULES.md, claude/WORKFLOW_MODES_PRAXIS.md, claude/chat/CLAUDE_CHAT_CONVENTIONS.md, claude/chat/NEW_CHAT_README.md, claude/META_DOCS.md, claude/MIDOFFICE_STATE.md, claude/TODO.md, and any other file under claude/ or .claude/ that isn't a brief or a retro;
- this machine's user level: ~/.claude/CLAUDE.md, ~/.claude/rules/, ~/.claude/settings.json. These apply to every repo on the machine, the factory and Rivet included, so report them and don't edit them;
- your auto memory for this repo (/memory, then the auto memory folder). Auto memory is per machine, so say whether it came over from the old machine or started empty here.

Search for: push, origin, "Jeff runs", script.

## 2. Write the new rule

It goes in praxis's CLAUDE.md (at the repo root, or .claude/CLAUDE.md if that's where it is), the file Claude Code loads in every session on any machine. If praxis has neither, create CLAUDE.md at the repo root with only this section. Copy the quoted section verbatim:

> ## Pushing to GitHub
>
> - Code pushes to origin when Jeff says so, in his message or in a brief, and never on its own initiative. Origin means praxis's and mctheory-core's own GitHub repos: jeffmcphail/mctheory-praxis and jeffmcphail/mctheory-core.
> - Never a force push, nothing that rewrites history, and never a push to anyone else's remote. So: no --force or --force-with-lease; no amend, rebase or reset of commits that are on origin; no deleting branches or tags there. Check git remote -v before the first push of a session.
> - After each push, confirm the branch on origin matches the local commit: git ls-remote origin refs/heads/<branch> must print the SHA that git rev-parse <branch> prints. Report both.
> - Anything done on GitHub's website stays Jeff's: pull requests, merges, releases, repository settings. Code doesn't do them, whether through the website, the gh CLI or the API.

## 3. Retire the old rule

- In each repo file where you found it, replace the old rule's text with one line pointing at the new section in CLAUDE.md. If it is a numbered rule in CLAUDE_CODE_RULES.md, keep the number on the pointer and renumber nothing; bump the file's version and add a changelog line, as the file does for every change.
- In your auto memory, replace any entry saying Jeff runs pushes or Code must not push with one line pointing at CLAUDE.md.
- If a settings file denies git push, quote the entry and leave it: Jeff decides.
- Leave old briefs and retros alone. They are records.

## 4. Report what depends on the old rule

List any other text in those files that assumes Jeff runs every push, and don't change it. The Praxis Chat knows of two from its 14 Nov copy of the repo; confirm them against the current files:

- claude/chat/CLAUDE_CHAT_CONVENTIONS.md, "Commit After Every Delta Delivery": the commit block Jeff runs ends with git push.
- That file says it is shared across all McTheory repos. If ai-agent-factory is on this machine, quote its push rule and its version of that section, and say whether they differ from praxis's. Don't edit the factory repo.

## Don't

- No push, no commit and no test push in this brief. Do run git remote -v in praxis and in mctheory-core, and report both origins.
- Don't edit anything outside praxis.

## Retro

claude/retros/RETRO_push_rule.md: where the old rule was (quoted), what changed, what you left and why, the two origins, and the git add line last (Rule 43).
```

---

## Amended, 2026-11-20

The brief says it runs on mcrmini2. Code found the session still on mcrmini1 (`hostname`), held
everything and said so. Jeff's reply, pasted shortly before 2026-11-20T15:22Z (10:22 EST),
transcribed verbatim:

```text
I'm telling you to run it now on mcrmini1
```
