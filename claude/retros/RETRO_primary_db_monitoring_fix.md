# Retro: Cycle 27.5 -- primary-DB monitoring fix-forward

**Brief:** `claude/handoffs/BRIEF_primary_db_monitoring_fix.md`
**Date:** 2026-05-07
**Mode:** Hybrid (Claude drafted, Code applied)
**Status:** DONE
**Predecessor:** Cycle 27 (`5d1162f`) -- _to_latest_ms autodetect collapse

---

## Summary

Closes the regression introduced by Cycle 27: the primary-DB
monitoring path in `get_collector_health` passed
`timestamp_format="auto"` as default to `_to_latest_ms`, but
Cycle 27 removed the `"auto"` branch. Cycle 27's pre-condition
check looked at `SIDECAR_DBS` (sidecar config) and missed the
separate primary-DB config block (`primary_monitored`).

Fix: change the primary-DB default from `"auto"` to `"ms"`. All
int-valued entries in `primary_monitored` map to tables migrated
to INTEGER ms format in Cycles 17-26.

Net change: 4 deletions / 6 insertions (+2 net) in `meta.py`.
py_compile clean.

---

## Why "fix-forward" not "revert Cycle 27"

Reverting Cycle 27 would have re-added the magnitude-
classification heuristic (`if latest > 1e12: ms; if latest > 1e9:
s`). Cycle 27's intent was to remove that heuristic in favor of
explicit format declarations -- a real correctness improvement
because mis-declared formats now fail loudly instead of being
silently classified by magnitude. Reverting throws that away.

The actual fix is to finish what Cycle 27 started: declare an
explicit format for primary-DB tables, just as `SIDECAR_DBS`
already declares explicit formats for its entries. The change is
two characters in the default value plus comment hygiene.

---

## Diagnosis trail

1. Cycle 27 committed `5d1162f` removed the `"auto"` branch from
   `_to_latest_ms`.
2. Pre-condition check at the time confirmed zero `"auto"`
   entries in `SIDECAR_DBS` (in `servers/praxis_mcp/server.py`).
3. The MCP server was restarted at 22:19 UTC on 2026-05-06 (cause
   unknown -- not by user instruction; possibly OOM or auto-
   restart). The restart loaded the post-Cycle-27 code into the
   running process.
4. Subsequent `get_collector_health` calls returned
   `"error": "could not parse timestamp"` for all 8 int-valued
   primary-DB monitored tables. Sidecar tables continued to work
   because they pass explicit formats from `SIDECAR_DBS`.
5. Investigation on 2026-05-07 traced the path:
   `get_collector_health` -> `_collect_db_health(...,
   timestamp_format="auto")` (line 259) -> `_to_latest_ms(latest,
   "auto")` -> returns None (no matching branch post-Cycle-27)
   -> caller hits the "could not parse timestamp" error path
   (line 360-364).

---

## Execution log

### Step 1: Apply brief

Code edited `servers/praxis_mcp/tools/meta.py`:

- Line 259: `timestamp_format="auto"` -> `timestamp_format="ms"`
  with a refreshed comment noting all int-valued entries are
  post-migration ms.
- Line 227: refreshed the comment block introducing
  `primary_monitored` to remove the stale "auto ms/s detection"
  reference.
- Line 230: removed `"auto"` from the documented enum of valid
  `timestamp_format` values.

### Step 2: py_compile

Clean.

### Step 3: Commit + push

Commit `4988f26` on origin/master.

### Step 4: USER step -- MCP server restart

Code did NOT do this. The user manually killed the MCP server
processes (PIDs 24968 + 33408 from the previous restart) and let
the framework respawn them, OR the user explicitly invoked the
restart depending on how the MCP server is managed in the
deployment.

After restart, `praxis:get_collector_health` was invoked and
returned ISO `latest`, numeric `staleness_seconds`,
`threshold_seconds`, and `is_stale` boolean for every primary-DB
monitored table (except `onchain_btc`, which has a separate
issue tracked under "Open items").

---

## Notes

### Why Cycle 27's pre-condition check missed this

Cycle 27's brief specified checking `SIDECAR_DBS` for `"auto"`
entries because that was the most visible config block. The
primary-DB monitoring config is a different shape -- a Python
dict literal embedded in the `get_collector_health` closure, with
the `"auto"` declaration on the call site (line 259) rather than
inside the dict itself. A more thorough audit would have grep'd
the entire `meta.py` file for `"auto"` (or the keyword
`timestamp_format`) and found both call sites. Lesson recorded:

> When removing a value from an enum-like API, grep the entire
> codebase for the value being removed AND for the parameter
> name -- not just the highest-visibility config block.

This is a generalizable lesson for any future "remove deprecated
option" cycle.

### Fix-forward beats revert when the original change was sound

Cycle 27 was a real improvement. The pre-condition check just
wasn't thorough enough. The repair pattern is "complete the
audit Cycle 27 started" rather than "undo Cycle 27." Worth
preserving as the default response to similar regressions:
investigate, identify the missed call site, fix it forward.

### Hybrid-workflow datapoint

Cycle 27.5 is the eighth hybrid cycle. The brief is small (~150
lines including the table) and Code's edit is two characters
plus comment hygiene. Active drafting time on Claude's side:
roughly the time it took to find and read the buggy call site.

---

## Open items / next cycle inputs

- **`onchain_btc` staleness** (since 2026-04-28): not Cycle-27
  related. Separately diagnosed in this session. Resolution
  depends on USER's manual collector run -- if it succeeds, only
  a scheduled task needs registering; if it fails, the API
  endpoint or credentials need investigation.
- **MCP server restart at 22:19 UTC 2026-05-06 of unknown
  origin**: probably benign (OOM or Windows update). Worth
  monitoring for recurrence; not a blocker.
