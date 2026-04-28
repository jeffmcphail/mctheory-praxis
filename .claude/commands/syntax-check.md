Run syntax validation on all recently modified Python files:

1. Find all .py files modified in the last commit or currently staged/unstaged:
   `git diff --name-only HEAD` and `git diff --name-only --cached`

2. For each .py file, run:
   `python -c "import ast; ast.parse(open('FILE').read()); print('OK: FILE')"`

3. Also check for non-ASCII characters (critical for Windows scheduled tasks):
   `grep -Pn "[^\x00-\x7F]" FILE`

4. Report results: which files pass, which fail, and the exact error.

If any file fails syntax check, do NOT commit until fixed.
