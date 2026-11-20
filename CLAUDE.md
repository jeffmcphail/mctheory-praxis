## Pushing to GitHub

- Code pushes to origin when Jeff says so, in his message or in a brief, and never on its own initiative. Origin means praxis's and mctheory-core's own GitHub repos: jeffmcphail/mctheory-praxis and jeffmcphail/mctheory-core.
- Never a force push, nothing that rewrites history, and never a push to anyone else's remote. So: no --force or --force-with-lease; no amend, rebase or reset of commits that are on origin; no deleting branches or tags there. Check git remote -v before the first push of a session.
- After each push, confirm the branch on origin matches the local commit: git ls-remote origin refs/heads/<branch> must print the SHA that git rev-parse <branch> prints. Report both.
- Anything done on GitHub's website stays Jeff's: pull requests, merges, releases, repository settings. Code doesn't do them, whether through the website, the gh CLI or the API.
