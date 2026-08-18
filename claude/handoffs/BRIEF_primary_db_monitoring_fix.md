# Cycle 27.5 -- primary-DB monitoring fix-forward (hybrid brief)

**Predecessor:** Cycle 27 (`5d1162f`) -- _to_latest_ms autodetect collapse
**Mode:** hybrid (Claude drafts, Code applies)
**Risk:** very low. Two-line change in
`servers/praxis_mcp/tools/meta.py`, plus cosmetic comment update.

## What

Cycle 27 removed the `"auto"` branch from `_to_latest_ms`. Code's
pre-condition check confirmed zero `"auto"` entries in
`SIDECAR_DBS` (the sidecar monitoring config) and concluded the
`"auto"` branch was dead code. **It missed the primary-DB
monitoring path**, which uses a separate config block
(`primary_monitored` in `get_collector_health`) and explicitly
passes `timestamp_format="auto"` as the default for int-valued
entries (line 259).

Result: starting at the moment Cycle 27's commit was loaded by
the running MCP server, `get_collector_health` returns
`"error": "could not parse timestamp"` for all 8 int-valued
primary-DB monitored tables (every monitored crypto_data table
except `onchain_btc`, which has its own dict spec with explicit
`timestamp_format="date"`).

The trades, order_book_snapshots, ohlcv_*, funding_rates,
fear_greed, and market_data collectors are all writing fine. Only
the observability readout is broken.

## Why fix-forward, not revert

Cycle 27's intent was to remove a defensive heuristic in favor of
explicit format declarations -- a good change. The fix is to
finish what Cycle 27 started: declare explicit `"ms"` for the
primary-DB monitoring default, just as `SIDECAR_DBS` already
declares explicit formats for its entries.

Reverting Cycle 27 would re-add the "auto" magnitude-classification
branch and we'd lose Cycle 27's correctness improvement. The
fix-forward is two characters (`auto` -> `ms`) plus comment
hygiene.

## Pre-condition (verify before applying)

Confirm every int-valued entry in `primary_monitored` (lines
232-246 in the current `meta.py`) maps to a table where the
`timestamp` column is INTEGER ms format. Per the migration program
(Cycles 17, 18, 19, 20, 21, 22, 23, 26) every primary-DB monitored
table that has an `int`-valued spec uses `timestamp INTEGER` ms.
The `onchain_btc` entry has a dict spec with
`timestamp_format="date"` (explicit; not affected).

Status as of Cycle 26 close (commit `d6f8820`):

| Table | Migration cycle | timestamp column type |
|---|---|---|
| trades | 26 (post-rebuild) | INTEGER ms |
| order_book_snapshots | 23 + 23.5 | INTEGER ms |
| ohlcv_1m | 22 | INTEGER ms |
| funding_rates | 21 | INTEGER ms |
| fear_greed | 17 | INTEGER ms |
| ohlcv_daily | 18 | INTEGER ms |
| ohlcv_4h | 20 | INTEGER ms |
| market_data | 19 | INTEGER ms |
| onchain_btc | (date format, separate) | -- |

All int-valued entries are post-migration ms. Safe to declare the
default as `"ms"`.

## Specifics for Code

In `servers/praxis_mcp/tools/meta.py`:

1. **Line 259**: change

   ```python
   timestamp_format="auto",   # autodetect ms vs s for primary
   ```

   to

   ```python
   timestamp_format="ms",   # all primary-DB int-valued entries are ms post-migration program (Cycles 17-26)
   ```

2. **Line 227**: change

   ```python
   # column with auto ms/s detection) OR a dict spec:
   ```

   to

   ```python
   # column at ms precision -- all int-valued entries are ms
   # post-migration program; see SCHEMA_MIGRATION_PLAN.md) OR a
   # dict spec:
   ```

3. **Line 230**: change

   ```python
   #    "timestamp_format": "ms"|"s"|"iso_text"|"auto"|"date"}
   ```

   to

   ```python
   #    "timestamp_format": "ms"|"s"|"iso_text"|"date"}
   ```

   (Removes the now-invalid `"auto"` from the documented enum.)

4. py_compile clean check.

5. Commit + push using the commit message at the bottom.

## Verification (USER step, NOT Code)

After commit + push, USER restarts the MCP server so the new
code is loaded. Then verify via `praxis:get_collector_health`:
all primary-DB monitored tables (except `onchain_btc`) should
report ISO `latest`, numeric `staleness_seconds`,
`threshold_seconds`, and `is_stale` boolean -- no
`"could not parse timestamp"` errors.

The MCP server restart is the critical verification step. Code
should NOT attempt to do this -- it requires manual ops on the
host machine. Code's role ends with the commit + push.

## Out of scope

- Adding a defensive `"auto"` fallback back into `_to_latest_ms`.
  That would undo Cycle 27.
- Touching `_collect_db_health` or `_collect_db_health_sidecar`.
  Those functions are unchanged; only the per-DB config default
  needs updating.
- Cycle 14+'s sidecar handling. Sidecars already use explicit
  formats (per `SIDECAR_DBS` config). No change needed.
- The `_to_latest_ms` function itself. Cycle 27's code is correct
  -- the bug was in the call site's choice of default `fmt`.

## Why a brief instead of a delta zip

The change is small enough that a delta zip would have more
overhead than the edit itself. Code can apply directly from the
brief.

## Commit message (use this verbatim)

```
Cycle 27.5: fix primary-DB monitoring fmt default

Cycle 27 (5d1162f) removed the "auto" magnitude-classification
branch from _to_latest_ms. Code's pre-condition check confirmed
zero "auto" entries in SIDECAR_DBS but missed the primary-DB
monitoring path in meta.py's get_collector_health, which passes
timestamp_format="auto" as the default for int-valued entries
in primary_monitored (line 259).

Result: post-Cycle-27, all 8 int-valued primary-DB monitored
tables (trades + order_book_snapshots + ohlcv_1m/4h/daily +
funding_rates + fear_greed + market_data) returned
"error: could not parse timestamp" from get_collector_health.
The collectors themselves were healthy and writing; only the
observability readout was broken.

Fix-forward: change the primary_monitored default from "auto"
to "ms". All int-valued entries map to tables that were migrated
to INTEGER ms format in Cycles 17-26 (the schema migration
program). Also updated the now-stale "auto ms/s detection"
comment and removed "auto" from the documented timestamp_format
enum.

Verification: USER must restart the MCP server after this
commit to load the change. praxis:get_collector_health should
then report ISO timestamps + staleness + is_stale for all
primary-DB monitored tables, no "could not parse" errors.
```
