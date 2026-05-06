# Cycle 27 -- _to_latest_ms autodetect collapse (hybrid mini-brief)

**Predecessor:** Cycle 24.1 (`5b742ba`) -- original investigation
**Mode:** hybrid (Claude drafts, Code applies)
**Risk:** very low. Pure refactor of one helper function in
`servers/praxis_mcp/tools/meta.py`. No DB state changes, no
process changes, no schema changes.

## What

Remove the `"auto"` branch from `_to_latest_ms` in
`servers/praxis_mcp/tools/meta.py`. After Cycle 25's cutover, every
monitored sidecar table is configured with an explicit
`timestamp_format` (`"ms"`, `"s"`, or `"iso_text"`); the `"auto"`
branch is dead code.

This is the cleanup of Cycle 24.1's investigation: that retro
documented the helper's exact shape for the diagnosis, and noted
that "auto" was a heuristic kept for safety. With every numeric-
timestamp table now ms-format, the heuristic has nothing to
detect.

## Why now

After Cycle 25.5 (today, `9339221`), every monitored numeric-
timestamp sidecar table is post-cutover ms-format. The
`SIDECAR_DBS` config in `servers/praxis_mcp/server.py` should have
zero `"timestamp_format": "auto"` entries today. The autodetect
heuristic in `_to_latest_ms` (`if latest > 1e12: ms; if latest >
1e9: s; else None`) is unreachable from any monitored path.

If a future cycle introduces a new monitored table with a numeric
timestamp of unknown format, the right move is to declare its
format explicitly in `SIDECAR_DBS`, not lean on autodetect. The
`"auto"` branch hides bugs by silently classifying values; an
explicit declaration makes any classification mistake visible at
config-edit time.

## Specifics for Code

In `servers/praxis_mcp/tools/meta.py`'s `_to_latest_ms`:

1. **Verify pre-condition**: read `SIDECAR_DBS` in
   `servers/praxis_mcp/server.py` and confirm zero entries have
   `"timestamp_format": "auto"`. If any do, surface them and STOP
   -- those tables either need an explicit format declared first,
   or this cycle needs to be expanded to handle them. Do NOT
   proceed silently if "auto" is still in use.

2. **Remove the `"auto"` branch and the implicit fallback** at
   the bottom of `_to_latest_ms`. Per Cycle 24.1's retro
   (claude/retros/RETRO_to_latest_ms_hotfix.md), the function
   currently looks roughly like:

   ```python
   def _to_latest_ms(latest, fmt: str):
       """Convert a `latest` timestamp value to Unix milliseconds.
       fmt: "auto" / "ms" / "s" / "iso_text" / ...
       """
       ...
       if not isinstance(latest, (int, float)):
           # Handles iso_text and other non-numeric formats elsewhere
           return None

       if fmt == "ms":
           return int(latest)
       if fmt == "s":
           return int(latest * 1000)
       # "auto"  <-- REMOVE THIS BLOCK AND BELOW
       if latest > 1e12:
           return int(latest)
       if latest > 1e9:
           return int(latest * 1000)
       return None
   ```

   After Cycle 27:

   ```python
   def _to_latest_ms(latest, fmt: str):
       """Convert a `latest` timestamp value to Unix milliseconds.
       fmt: "ms" or "s" (numeric). Other formats handled at call
       site before invocation.
       """
       ...
       if not isinstance(latest, (int, float)):
           return None
       if fmt == "ms":
           return int(latest)
       if fmt == "s":
           return int(latest * 1000)
       # Cycle 27: autodetect "auto" branch removed. Every monitored
       # sidecar declares an explicit timestamp_format; unknown fmt
       # values now return None instead of being heuristically
       # classified.
       return None
   ```

3. **Update the docstring** to reflect that `fmt` no longer
   accepts `"auto"`. Per the brief's example above.

4. **Keep the `"s"` branch**: although no monitored table is
   currently `"s"`-format, removing it is more aggressive than
   needed. A future migration adding an `s`-format collector
   wouldn't break anything. Net change: maybe 6 lines removed.

5. **py_compile clean check**.

## Out of scope

- Removing `_to_latest_ms` entirely (it's still needed for `ms`
  and `s`).
- Removing `"s"` branch (defensive; cheap to keep).
- Touching the `iso_text` handling (different code path; works
  fine).
- Changing any `SIDECAR_DBS` entries (they should already all be
  explicit post-Cycle-25.5).

## Verification

py_compile only. End-to-end behavior should be identical because
no monitored table uses `"auto"`. Spot-check post-deploy via
`get_collector_health` -- every `latest` should still be reported
correctly. If something breaks, the failure mode would be a
sidecar table reporting `latest = null` instead of a number,
which would surface as `is_stale = true` in MCP output. None
should occur.

## Why a brief instead of a delta zip

Claude's local checkout pre-dates Cycle 14 entirely (`SIDECAR_DBS`
isn't there, and `_to_latest_ms` only exists in post-Cycle-14
versions). Code reads the actual on-disk file.

## Commit message (use this verbatim)

```
Cycle 27: _to_latest_ms autodetect collapse

After Cycle 25.5 every monitored numeric-timestamp sidecar table
declares an explicit timestamp_format ("ms" or "s"). The
autodetect "auto" branch in _to_latest_ms (heuristic: >1e12 ->
ms, >1e9 -> s) is now unreachable; removing it makes mis-
declared formats fail loudly (return None) instead of being
silently classified by magnitude.

Per Cycle 24.1's retro, the helper was originally written
defensively to dodge the OSError(22) that datetime.fromtimestamp
throws on Windows for ms-magnitude values passed through the
seconds path. With explicit declarations everywhere that defense
is no longer load-bearing.

Backwards-compatible for any caller that explicitly passes
fmt="ms" or fmt="s". Callers passing fmt="auto" now get None
instead of a magnitude-classified guess; no such callers exist
in the current codebase.
```
