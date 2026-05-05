# Retro: Cycle 24.1 -- `_to_latest_ms` ms-format "hotfix" (no-op)

**Brief:** `claude/handoffs/BRIEF_to_latest_ms_hotfix.md`
**Date:** 2026-05-05
**Mode:** B (Brief -> Code), retro-only
**Status:** CLOSED -- no code change required
**Predecessor:** Cycle 24 (`b8fa847`, `6ca1796`, `dbecb23`)

---

## Summary

The Brief diagnosed a year-58000 OSError in
`get_collector_health.databases.live_collector` and hypothesized a
missing `/1000` divide (or missing `"ms"` branch) inside whichever
helper handles sidecar staleness reporting. Code is expected to
locate the offending lines in `servers/praxis_mcp/server.py` and
apply a 1-line fix.

**No bug existed in the on-disk code.** The `_to_latest_ms` helper at
`servers/praxis_mcp/tools/meta.py:481-544` already had a correct
`"ms"` branch from before this session. The `__error__` Chat saw was
emitted by a Claude-Desktop-resident MCP subprocess running the
*pre-Cycle-24* configuration in memory; close-and-reopen of Claude
Desktop did NOT terminate that subprocess. Only a hard restart via
Task Manager (End Process on the python.exe MCP children) cleared
the stale state. Post-hard-restart verification at 12:20 UTC
confirms the live tool reports correctly.

The valuable artifact from this cycle is the two process notes
below, not a code change.

---

## Brief's hypothesis vs on-disk reality

The Brief proposed two bug shapes inside the sidecar staleness
helper:

```python
# HYPOTHESIS 1: missing divide
if timestamp_format == "ms":
    dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
    # BUG: should be datetime.fromtimestamp(latest_ts / 1000, ...)
```

```python
# HYPOTHESIS 2: branch never written; falls through to "s" default
if timestamp_format == "s":
    dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
elif timestamp_format == "iso_text":
    dt = datetime.fromisoformat(latest_ts)
# missing: elif timestamp_format == "ms": ...
```

Both hypotheses assume the helper goes through `datetime.fromtimestamp`,
which is where a year-58000 OSError(22) would originate on Windows.

Actual on-disk shape, `servers/praxis_mcp/tools/meta.py:481-544`:

```python
def _to_latest_ms(latest, fmt: str):
    """Convert a `latest` timestamp value to Unix milliseconds.

    fmt:
      "auto"     -- numeric: detect ms vs s by magnitude (>1e12 -> ms,
                    >1e9 -> s); else None.
      "ms"       -- numeric milliseconds since epoch.
      "s"        -- numeric seconds since epoch.
      ...
    Returns int (ms) or None if unparseable.
    """
    ...
    if not isinstance(latest, (int, float)):
        return None

    if fmt == "ms":
        return int(latest)
    if fmt == "s":
        return int(latest * 1000)
    # "auto"
    if latest > 1e12:
        return int(latest)
    if latest > 1e9:
        return int(latest * 1000)
    return None
```

The function never calls `datetime.fromtimestamp` on numeric input.
The `"ms"` branch is a one-line `return int(latest)` -- correct,
unit-preserving, no platform-specific failure mode. Staleness is
then computed at the call site as `now_ms - latest_ms`, also pure
integer arithmetic.

So neither hypothesis 1 (missing divide) nor hypothesis 2 (missing
branch) describes any bug present in the file.

Local repro confirmed the OSError is reachable in principle:

```python
>>> from datetime import datetime, timezone
>>> datetime.fromtimestamp(1777961148620, tz=timezone.utc)
OSError: [Errno 22] Invalid argument
>>> datetime.fromtimestamp(1777961148620 / 1000, tz=timezone.utc)
datetime.datetime(2026, 5, 5, 6, 5, 48, 620000, tzinfo=timezone.utc)
```

But that error path simply does not exist in `_to_latest_ms` as
shipped. Whoever wrote the helper avoided `fromtimestamp` on
numeric paths entirely -- exactly to dodge this Windows behavior.

---

## Real root cause: stale MCP subprocess

The `__error__` Chat was reading came from a **different** copy of
`meta.py` than the one on disk -- a copy held in the memory of the
FastMCP stdio child process that Claude Desktop had spawned before
Cycle 24's Phase 4 commit (`6ca1796`) flipped the
`SIDECAR_DBS["live_collector"]["tables"]["price_snapshots"]["timestamp_format"]`
string from `"s"` to `"ms"`.

That subprocess kept executing pre-cutover code, where the
`timestamp_format` was still `"s"` and the `_to_latest_ms` `"s"`
branch was multiplying ms-magnitude values by 1000 (yielding ~1.78e15)
before downstream display logic tried to render it -- producing the
year-58000 OSError. The on-disk code, which Code reads directly,
already had `"ms"` in the config and the corresponding correct
return path. The two views disagreed because the subprocess never
re-imported the module.

Crucially, **closing and reopening Claude Desktop did not fix it.**
Jeff confirmed via `Get-Process python` that python.exe MCP
children were still alive after a normal close/reopen cycle. Only
ending those processes via Task Manager (a "hard" restart) caused
Desktop to spawn fresh MCP children that re-imported the post-Cycle-24
`meta.py` and `SIDECAR_DBS`.

Post-hard-restart `get_collector_health` reports for `price_snapshots`:

```
row_count    = 374,755
latest       = 2026-05-05T12:19:48.522Z
is_stale     = false
staleness    = 37s
```

Clean. Year-58000 artifact is gone. Primary-DB tables and other
sidecar tables continue to report normally.

---

## Process notes (the actual deliverable)

### Note 1: Cycle 24's AC #20 was claimed-but-not-actually-verified

Cycle 24's retro at `claude/retros/RETRO_price_snapshots_dual_write.md`
line 225-227 asserted `get_collector_health` "interprets the new ms
timestamp correctly." That assertion was a tautology: the writer
sets `"ms"`, the helper has an `"ms"` branch, therefore it works.
Nobody actually called the live MCP tool against the post-cutover
state and read the response.

Had the verification step actually been performed at the end of
Cycle 24, the year-58000 artifact from the stale subprocess would
have surfaced inside Cycle 24, not as a separate hotfix cycle the
next day. The bug was never in the code, but the *symptom* would
have been visible in the live tool response, and the diagnostic
path (hard-restart MCP, retest) would have been exercised then.

**Prescription for future dual-write Briefs (Cycles 25-26 and beyond):**

Add an explicit acceptance criterion of the shape:

> Post-restart, Chat exercises `get_collector_health` against the
> live MCP tool and pastes the full response into the cycle's chat.
> The relevant sidecar entry must report a parseable ISO `latest`,
> `is_stale=false`, `staleness_seconds < threshold`, and
> `row_count > 0`. Primary-DB regression check: other monitored
> tables still report cleanly with no new `__error__` artifacts.

The bar is the live tool response, not the writer's belief that
the config change should work. "Verification" that consists only
of reading the diff and reasoning about what the diff implies is
not verification of the live system.

### Note 2: "Restart Claude Desktop" is ambiguous

The Cycle 24 Brief and retro both used the phrase "restart Claude
Desktop" without qualification. In practice on Windows, a normal
close-and-reopen of the Desktop window does **not** reliably
terminate the FastMCP stdio child processes it spawned. Those
children can survive into the next Desktop session and continue
serving the old in-memory state of `meta.py` / `SIDECAR_DBS` /
any other module that was imported before the change.

This is the failure mode that produced Cycle 24.1: on-disk code
was correct, but Chat was talking to a child process that predated
the fix.

**Prescription for future MCP-touching cycles:**

Replace "restart Claude Desktop" in Briefs and retros with one of:

- **(preferred) Hard restart via Task Manager**: close Desktop, open
  Task Manager, end any `python.exe` (or `pythonw.exe`) processes
  whose command line contains the praxis_mcp server path, then
  reopen Desktop.
- **(verification floor)** After closing Desktop, run
  `Get-Process python | Where-Object { $_.MainModule.FileName -like
  "*<praxis venv>*" }` (or whatever distinguishes the MCP children
  from unrelated Python processes) and confirm the count is zero
  before reopening. If non-zero, end them explicitly.

Either form makes the "fresh process re-imports the post-change
code" precondition explicit, instead of leaving it as a side effect
that may or may not happen depending on how the user closes the
window.

This is now a load-bearing convention for any cycle that changes
MCP tool behavior, schema-monitor configuration, or anything else
imported by a long-lived MCP child.

---

## Acceptance Criteria (Brief vs reality)

| # | Criterion | Status |
|---|---|---|
| 1 | Identify exact line(s) where ms-format sidecar timestamps are mishandled; quote before-state | N/A -- no such lines exist; on-disk `_to_latest_ms` already correct |
| 2 | Apply minimal fix (<=5 lines), match primary-DB ms pattern | N/A -- nothing to fix |
| 3 | py_compile clean | N/A -- no code change |
| 4 | Single commit + push | Retro + TODO commit only |
| 5 | Post-restart, Chat exercises `get_collector_health` and pastes response: row_count>0, latest parseable 2026-05-*, is_stale=false, staleness<180 | PASS (374,755 / 2026-05-05T12:19:48.522Z / false / 37s) |
| 6 | Regression check: primary-DB tables still report correctly | PASS |
| 7 | `claude/TODO.md` updated (Cycle 24.1 in Recently closed) | PASS (this commit) |
| 8 | Retro documents bug, fix, before/after MCP responses, process note re AC #20 | PASS (this file; expanded with hard-restart process note) |
| 9 | All committable files ASCII-only (Rule 20) | PASS |
| 10 | If diagnosis reveals bug is structurally larger than missing /1000 / missing branch, STOP and surface to chat | EXERCISED -- diagnosis revealed there is no bug; cycle scoped down to retro-only |

The Brief's `Notes for Code` correctly anticipated the third path
("If the bug is structural ... surface to chat; do NOT expand scope")
and AC #10 made the surface-and-stop behavior explicit. Both
applied.

---

## Lessons learned

1. **The next Brief that diagnoses an MCP-tool symptom should
   include a hard-restart step BEFORE concluding the on-disk code
   is wrong.** A symptom that reproduces against the in-process
   MCP server but disappears after a hard restart is, by
   definition, a stale-subprocess artifact, not an on-disk bug.
   Cycle 24.1's Brief jumped to the on-disk hypothesis without
   ruling out the stale-process explanation.

2. **"Cross-engine grep for similar bugs" was a worthwhile exercise
   even though no fix was needed.** The grep confirmed
   `datetime.fromtimestamp` is not called on raw `price_snapshots`
   ms values anywhere reachable from `get_collector_health`. That
   negative result is now durable -- if a year-58000 OSError
   resurfaces in a future cycle, the diagnostic tree should branch
   to "stale process" first, not "missing /1000."

3. **Verification ACs that are tautologies should be flagged when
   writing Briefs.** Cycle 24's "is_stale reports correctly" was
   really "the writer wrote `"ms"` and the helper has a `"ms"`
   branch, ergo." A useful AC has to involve an external observable
   that could fail independently of the code change -- in this
   case, "Chat pastes the full live tool response back."

---

## Open items / next cycle inputs

- **No follow-up cycle required.** Cycle 24.1 closes as retro-only.
- **Cycle 24.5 (Phase 5 cleanup)** is unchanged and remains queued
  for after the 24-48h burn-in window expires.
- **Cycle 25 (smart_money)** Brief, when written, should adopt
  both process notes above as ACs.

---

## Deviations from Brief

- **No code commit.** The Brief assumed a 1-line fix in
  `servers/praxis_mcp/server.py`. The on-disk code was already
  correct, so the cycle ships only this retro and a TODO update.
  AC #10's STOP-and-surface clause covers the deviation.
- **AC #4 reinterpreted** as "single commit (retro + TODO only)"
  rather than "single commit (code + retro + TODO)." No hash-patch
  needed (no doc references this cycle's commit hash).
