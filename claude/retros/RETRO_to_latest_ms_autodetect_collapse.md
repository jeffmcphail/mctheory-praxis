# Retro: Cycle 27 -- _to_latest_ms autodetect collapse

**Brief:** `claude/handoffs/BRIEF_to_latest_ms_autodetect_collapse.md`
**Date:** 2026-05-06
**Mode:** Hybrid (Claude drafted, Code applied)
**Status:** DONE
**Predecessor:** Cycle 24.1 (`5b742ba`) -- original investigation

---

## Summary

Removed the autodetect ("auto") branch from `_to_latest_ms` in
`servers/praxis_mcp/tools/meta.py`. After Cycle 25.5 every
monitored numeric-timestamp sidecar table declares an explicit
`timestamp_format` ("ms" or "s"); the autodetect heuristic (>1e12
-> ms, >1e9 -> s) is unreachable from any monitored call path
and removing it converts any future config error from a silent
magnitude-classification into an explicit None return.

Net change: 8 deletions / 5 insertions (-3 net) in `meta.py`.
The actual code removal is 5 lines (the `# "auto"` comment +
4-line magnitude-classification block); the rest is docstring
churn (removed the `"auto"` entry, added a Cycle-27 explanatory
sentence). py_compile clean.

---

## Pre-condition check (per Brief AC #1)

Before applying the change, verified `SIDECAR_DBS` in
`servers/praxis_mcp/server.py` contains zero entries with
`"timestamp_format": "auto"`. Found:

All entries explicit -- OK to proceed. `SIDECAR_DBS` at
`servers/praxis_mcp/server.py:81-102` contains exactly two
monitored numeric-timestamp tables and both are
`"timestamp_format": "ms"`:

- `live_collector.price_snapshots` (post-Cycle-24.5 cutover)
- `smart_money.position_snapshots` (post-Cycle-25.5 cutover)

Zero `"auto"` entries in the live config; the autodetect branch
in `_to_latest_ms` was provably dead code from any monitored
call path before this change.

---

## Execution log

### Step 1: Apply brief

Code edited `servers/praxis_mcp/tools/meta.py`:

- `_to_latest_ms` (line 481): removed the `# "auto"` comment
  and the 4-line magnitude-classification fallback at the bottom
  of the numeric branch. Function now returns `None` for any
  `fmt` not in `{"ms", "s", "iso_text", "date"}`.
- Docstring updated: removed the `"auto"` entry from the `fmt`
  enumeration and added an explanatory sentence about the Cycle-27
  removal so a future reader understands that an unknown `fmt`
  now fails loudly rather than being heuristically classified.

### Step 2: py_compile

Clean.

### Step 3: Commit + push

Commit `5d1162f` on origin/master.

### Step 4: Verification

`get_collector_health` still reports all monitored tables with
parseable `latest` ISO datetimes; no regressions. The change is
behaviorally identical for every current call path because no
caller currently passes `fmt="auto"`.

---

## Notes

### Cycle 24.1 closure

Cycle 24.1's retro flagged `_to_latest_ms` as having defensive
code that was load-bearing during the Windows-OSError(22)
investigation. That defense is no longer load-bearing because
every monitored table declares its format explicitly. Cycle 27
closes the loop: the helper is now leaner and any future
mis-declared format fails loudly.

### Hybrid workflow: fifth cycle

Cycle 27 is the fifth hybrid cycle (after 23.5, 24.5, 28, 25.5).
This was the smallest by far -- maybe the smallest cycle in the
whole migration program. ~6 lines removed, one docstring
updated. The brief format works for changes this small as well
as for larger ones; the discipline is the same regardless of
size.

---

## Open items / next cycle inputs

- **Cycle 26** (trades): now the only remaining schema migration.
- **Cycle 29** (CCXT collector exit-code audit): apply Cycle 28's
  status-dict pattern to `collect_ohlcv_daily`,
  `collect_ohlcv_4h`, `collect_fear_greed`,
  `collect_funding_rates`, `collect_onchain_btc`. ~30 min.
