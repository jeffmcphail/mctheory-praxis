Generate a retro log for the current session and save it to claude/retros/RETRO_<slug>.md

Every piece of work ends with one, briefed or not (Rule 43): the terminal output Jeff passes on
gets truncated, so the retro is the record. Write it in full.

The retro must include:
1. Summary (one paragraph)
2. What was asked -- the brief, verbatim when available
3. Files Modified table (file, change description, line ranges)
4. What changed and why, and every table and number that would otherwise be printed
5. Key Decisions made during implementation, and the decisions left for Jeff
6. Test Results (passed and failed): counts, and what each new test guards
7. Failures & Debugging Trail (what was tried, what failed, why)
8. Problems found but not fixed
9. The key content of every scratch output, copied in -- the session scratch directory is
   under Temp and can vanish
10. Git commits made, if any
11. Open Items for Chat (anything needing strategy discussion)
12. LAST: the exact `git add` command for the change -- every file, tracked and new, the retro
    included, and nothing else. Commit messages come from Chat; the file list comes from Code.

Name and index it like the existing retros, with its brief at claude/handoffs/BRIEF_<slug>.md.
Mid-office work is the `midofficeNN` series. In the terminal, finish with a short summary and
the retro's path.

Follow the format defined in claude/CLAUDE_CODE_RULES.md
