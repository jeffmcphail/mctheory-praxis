# BRIEF: Cycle 24.1 -- `_to_latest_ms` ms-format hotfix

**Series:** praxis
**Cycle:** 24.1
**Mode:** B (Brief -> Code) hotfix
**Author:** Chat (praxis_main_current)
**Date:** 2026-05-05
**Predecessor:** Cycle 24 (`b8fa847`, `6ca1796`, `dbecb23`) -- price_snapshots dual-write

---

## Context

Cycle 24's Phase 4 commit changed
`SIDECAR_DBS["live_collector"]["tables"]["price_snapshots"]["timestamp_format"]`
from `"s"` to `"ms"` to track the table's new ms-precision timestamps.
The Cycle 24 retro at line 225-227 claimed `get_collector_health`
"interprets the new ms timestamp correctly" but did NOT actually exercise
the MCP tool against the post-cutover state.

In-situ verification post-Claude-Desktop-restart shows
`get_collector_health` returns:

```json
"live_collector": {
  "tables": {"__error__": "[Errno 22] Invalid argument"},
  "unmonitored": ["collection_log", "price_snapshots_legacy",
                  "spike_alerts", "tracked_markets"]
}
```

The `__error__` key replaces the entire per-table dict, so we don't
even see which table failed -- but `price_snapshots` is the only
monitored sidecar table for `live_collector.db`, and it's the one
that flipped to `"ms"` format.

**Disk-side verification confirms the data is healthy.** Latest row
from `python -c "..."`:

```
[('will-kim-kardashian-...', 1777961148620,
  '2026-05-05T06:05:48+00:00', 0.0075),
 ('will-tom-brady-...',      1777961148136,
  '2026-05-05T06:05:48+00:00', 0.0075),
 ('will-glenn-youngkin-...', 1777961147720,
  '2026-05-05T06:05:47+00:00', 0.0075)]
```

ms timestamps in the proper `~1.78e12` range, sub-second precision
preserved, datetime ISO-formatted with `+00:00` offset. **The
migration data is correct; the bug is purely in the monitoring tool.**

The bug pattern is the year-58000 explosion: passing a raw ms value
(`~1.78e12`) to `datetime.fromtimestamp()` interprets it as seconds,
producing a date in year ~58000 that Windows refuses with
`OSError(22) [Errno 22] Invalid argument`. Same pattern that motivated
the magnitude-detect fix in `dashboards/data_collector.py` during
Cycle 24 Phase 0.

For comparison: `order_book_snapshots` (also ms-format post-Cycle-23)
reports correctly because it's in the **primary** DB and uses a
different code path (the `monitored_tables` dict at the top of
`get_collector_health`), not the sidecar `SIDECAR_DBS` config path
that was flipped in Cycle 24's Phase 4.

---

## Root cause hypothesis

The `_to_latest_ms` function (or whatever the sidecar staleness
helper is named) almost certainly has an `if timestamp_format == "ms"`
branch that's missing the `/1000` divide before
`datetime.fromtimestamp()`. Two plausible shapes:

```python
# HYPOTHESIS 1: missing divide
if timestamp_format == "ms":
    dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
    # BUG: should be datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc)
```

```python
# HYPOTHESIS 2: branch never written; falls through to "s" default
if timestamp_format == "s":
    dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
elif timestamp_format == "iso_text":
    dt = datetime.fromisoformat(latest_ts)
# missing: elif timestamp_format == "ms": ...
# falls through, dt is undefined OR uses the "s" path with raw ms value
```

Code should read the actual function before assuming which.

---

## Scope (single-commit hotfix)

### Task 1: Diagnose

Read `servers/praxis_mcp/server.py` and locate the function that
handles sidecar staleness reporting. Likely names:
`_to_latest_ms`, `_compute_staleness`, `_sidecar_table_status`, or
inline inside `get_collector_health`. Quote the offending lines in
the retro.

Confirm the bug shape via a local reproduction:

```python
from datetime import datetime, timezone
# Bug repro: this throws OSError(22) on Windows
datetime.fromtimestamp(1777961148620, tz=timezone.utc)

# Fix: dividing by 1000 works
datetime.fromtimestamp(1777961148620 / 1000, tz=timezone.utc)
# -> datetime.datetime(2026, 5, 5, 6, 5, 48, 620000, tzinfo=timezone.utc)
```

### Task 2: Fix

Apply the minimal fix. Preferred shapes (in order of preference):

1. **If hypothesis 1 (missing divide)**: add `/ 1000` to the existing
   `"ms"` branch. Single-line change.

2. **If hypothesis 2 (missing branch entirely)**: add the `"ms"`
   branch with proper handling. Match the style of the existing `"s"`
   and `"iso_text"` (or whatever the primary-DB-path uses) branches.

3. **If the bug is structural (e.g., the function takes a raw value
   and the format declaration is informational only)**: surface to
   chat; do NOT expand scope without confirmation.

The primary-DB path is the reference implementation -- it correctly
handles ms timestamps from `order_book_snapshots`, `trades`, etc.
Whatever pattern that path uses, the sidecar path should match.

### Task 3: Verify (live MCP, not just unit)

py_compile clean.

Restart Claude Desktop. From the chat side I (Chat) will exercise
`get_collector_health` and paste the response. Verification PASSES iff:

- `databases.live_collector.tables.price_snapshots.row_count > 0`
- `databases.live_collector.tables.price_snapshots.latest` is a
  parseable ISO datetime in 2026-05 (not year 58000)
- `databases.live_collector.tables.price_snapshots.is_stale` is `false`
- `databases.live_collector.tables.price_snapshots.staleness_seconds`
  is `< 180`
- `databases.live_collector.unmonitored` no longer contains the
  `__error__` artifact
- Primary DB tables (`order_book_snapshots`, `trades`, etc.) still
  report correctly (regression check)

If verification fails, surface to chat with the new MCP response
content; do NOT iterate-and-commit.

### Task 4: Commit + push

Single commit. No hash-patch needed (no `<TBD>` references in any
doc).

Suggested message:
```
Cycle 24.1 hotfix: _to_latest_ms ms-format sidecar handling

Cycle 24's Phase 4 flipped SIDECAR_DBS["live_collector"]
["tables"]["price_snapshots"]["timestamp_format"] from "s" to
"ms" but the staleness helper's "ms" branch was missing the
/1000 divide before datetime.fromtimestamp(), producing
OSError(22) on Windows from year-58000 timestamps. Fix
matches the primary-DB ms-handling pattern. Verified via
get_collector_health post-restart.

Cycle 24's retro (line 225-227) claimed AC #20 was verified
but never actually exercised the MCP tool. Process note in
the hotfix retro flags this for future dual-write briefs.
```

### Task 5: Doc trio updates

**`claude/TODO.md`**:
- Add Cycle 24.1 to "Recently closed" with one-line summary
  + commit hash. Plain entry, not "(this cycle)".

**`claude/retros/RETRO_to_latest_ms_hotfix.md`** (new file):
- One page max.
- Quote the bug as found (before-state code + the year-58000 stack
  trace if reproducible).
- Quote the fix (after-state code).
- Pre-fix `get_collector_health` response excerpt (the `__error__`
  block).
- Post-fix `get_collector_health` response excerpt (clean
  price_snapshots entry).
- **Process note**: AC #20 of Cycle 24 was claimed verified but
  never actually exercised by calling the MCP tool. Future
  dual-write Briefs should include an AC like "Chat exercises
  `get_collector_health` post-restart and pastes the full response
  back into the cycle's chat" rather than just asserting "is_stale
  reports correctly". The bar for verification is the live tool
  response, not the writer's belief that the config change should
  work.

**`docs/SCHEMA_NOTES.md`** and **`docs/SCHEMA_MIGRATION_PLAN.md`**:
- No changes needed unless Code finds the bug had broader implications.
  These docs describe the schema, not the monitoring tool.

---

## Out of scope

- Adding new staleness-format support beyond `"ms"`/`"s"`/`"iso_text"`
- Refactoring `_to_latest_ms` (or equivalent) for clarity
- Investigating other latent bugs in `get_collector_health`
- Touching the autodetect heuristic that's queued for Cycle 27
- Re-verifying any other Cycle 24 acceptance criterion (Cycle 24 is
  closed; this hotfix only addresses the AC #20 miss)

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | Identify the exact line(s) where ms-format sidecar timestamps are mishandled; quote the before-state code in the retro |
| 2 | Apply minimal fix (preferably <=5 lines of code change). Preferred shape: match the primary-DB ms-handling pattern verbatim |
| 3 | py_compile clean |
| 4 | Single commit + push (no hash-patch) |
| 5 | Post-Claude-Desktop-restart, Chat exercises `get_collector_health` and pastes the response. `databases.live_collector.tables.price_snapshots` reports `row_count > 0`, `latest` parseable as 2026-05-* ISO datetime, `is_stale=false`, `staleness_seconds < 180` |
| 6 | Regression check: primary-DB tables (`order_book_snapshots`, `trades`, `ohlcv_1m`, etc.) still report correctly with no new errors |
| 7 | `claude/TODO.md` updated (Cycle 24.1 in Recently closed) |
| 8 | Retro at `claude/retros/RETRO_to_latest_ms_hotfix.md` documents bug, fix, before/after MCP responses, AND the process note that AC #20 of Cycle 24 was claimed-but-not-actually-verified |
| 9 | All committable files ASCII-only (Rule 20) |
| 10 | If diagnosis reveals the bug is structurally larger than "missing /1000 or missing branch", Code STOPS and surfaces to chat instead of expanding scope |

---

## Notes for Code

- **The MCP server is currently running with the broken code.** Code's
  fix lands on disk; Claude Desktop's MCP server keeps the broken
  version in memory until restarted. The verify step (AC #5) is
  Chat's responsibility post-restart, not Code's. Do NOT skip the
  commit just because Code can't verify in-process.

- **Don't add a magnitude-detect fallback** (`if ts > 1e12: ts /= 1000`)
  to the sidecar path "for safety". The format declaration in
  `SIDECAR_DBS` is the source of truth; magnitude-detect was a
  coping mechanism for `dashboards/data_collector.py` where the
  reader has no config to consult. Here the config IS the contract;
  honor it strictly. If the format says `"ms"`, divide by 1000.
  Period.

- **The retro is the artifact, not the bug fix.** The fix is probably
  one line. The valuable output is the process note: AC #20 was
  claimed verified but the verification was a tautology ("the change
  should work, therefore it works"). Cycle 24.1's retro should make
  this lesson durable for Cycles 25-26 (smart_money + trades) which
  will hit the same MCP-config touch points.

- **Cross-engine grep for any other sites** that might pass raw ms
  to `datetime.fromtimestamp()` and weren't caught in Cycle 24's
  audit. If the sidecar-staleness bug exists, similar bugs may exist
  in adjacent code. If found, surface in retro; out of scope to fix
  unless trivial.

- **Don't touch the running MCP process.** The fix is deploy-then-
  restart. No live patching.

---

## Apply sequence (~10-15 min wall-clock)

1. Drop zip into repo
2. Paste into Code chat: "process the brief at
   `claude/handoffs/BRIEF_to_latest_ms_hotfix.md`"
3. Code reads, diagnoses, applies fix, commits, pushes
4. Code reports the diagnosis + commit hash
5. Jeff restarts Claude Desktop
6. Chat exercises `get_collector_health` and pastes the response
7. If clean: Cycle 24.1 closes. If not: surface and iterate.
