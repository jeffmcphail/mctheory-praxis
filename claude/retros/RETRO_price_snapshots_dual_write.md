# Retro: Cycle 24 -- live_collector.price_snapshots Dual-Write Migration

**Brief:** `claude/handoffs/BRIEF_price_snapshots_dual_write.md`
**Date:** 2026-05-05
**Duration:** ~70 min wall-clock (~10 min Phase 0 + 60-min burn-in
+ ~5 min Phases 2-4 + commits)
**Status:** DONE-PARTIAL (Phases 0-4 complete; Phase 5 cleanup
deferred to Cycle 24.5)
**Predecessor:** Cycle 23 (`ca5c719`, `10724bc`, `5cf1c03`) --
order_book_snapshots dual-write pilot

---

## Summary

**Second use of the dual-write recipe**. Six-phase pattern
established in Cycle 23 reapplied to `live_collector.price_snapshots`
(358,715 rows at cutover, ~50 rows/min via continuous 60s polling of
~50 active Polymarket markets). Recipe held up cleanly across a new
DB (sidecar `live_collector.db`), a new writer file
(`engines/live_collector.py`), and a different process pattern
(continuous long-lived process vs Cycle 23's hourly restart).

**Headline result**: schema migrated to Rule 35 (compound PK on
`(slug, timestamp)`, no `id`, ms timestamps with sub-second precision
for new rows, ISO `+00:00` `datetime` column derived from
`timestamp`), all 358,715 legacy rows preserved (now in
`price_snapshots_legacy`) plus 361,961 in the new live
`price_snapshots`, atomic cutover in 4ms,
in-process spike detection + ms-aware reader fixes verified working
post-cutover.

**Three differences from Cycle 23 worth surfacing**:

1. **No precision to recover**. Cycle 23 had a microsecond-precision
   `datetime` column whose information was being thrown away by the
   writer's `ts_ms // 1000` truncation; the migration recovered ms
   from the existing `datetime`. Cycle 24's source has only an
   integer-seconds `timestamp` column and NO `datetime` column at
   all. The migration is a clean `legacy_ts * 1000` multiply (no
   julianday/ROUND), and `datetime` is a new column derived in pure
   SQL via `strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp,
   'unixepoch')` for backfilled rows. Sub-second precision is GAINED
   for new rows (writer captures fresh `time.time()` per insert), not
   recovered from old data.

2. **Long-lived process pattern**. PraxisOrderBookCollector restarts
   hourly; PraxisLiveCollector launches python once and runs
   indefinitely. File changes do NOT auto-pick-up. The Phase 0
   commit was paired with an explicit kill-and-relaunch step
   (Task Scheduler then relaunches with the new code). Elapsed time
   from `git push` to confirmed dual-write: ~immediate (Task
   Scheduler had already picked up the on-disk edits before the
   remote push landed; first v2 row at 03:04:28 UTC, ~4 min before
   commit b8fa847 pushed at ~03:08 UTC). This is a
   load-bearing operational detail for Cycles 25-26 -- TODO entry
   added to investigate whether smart_money and trades collectors
   share this pattern before their Briefs are written.

3. **Atomic-with-writer reader fixes**. Cycle 22's `intrabar_predictor`
   reader fix could ship after the migration because the reader is
   offline analytics. Cycle 24 has an in-process reader
   (`check_for_spikes`) that fires on EVERY collector cycle; shipping
   ms writes with seconds-aware reads silently breaks spike detection
   from the moment the writer change goes live. The Brief was
   explicit that the reader fix and writer change must land in the
   same Phase 0 commit. Three Brief-named reader sites + one
   audit-discovered fourth (in `dashboards/data_collector.py`) all
   shipped atomically.

---

## Changes Made

### Files Modified

| File | Change |
|------|--------|
| `engines/live_collector.py` | `init_db`: added `CREATE TABLE IF NOT EXISTS price_snapshots_v2 (...)` with target Rule 35 schema (slug, ms ts, ISO datetime, yes_*, spread, compound PK) + idx on (slug, timestamp DESC). `sample_all_markets`: rewrote single INSERT as runtime-adaptive dual-INSERT. Pre-cutover (live has `id`): write seconds to live + ms+datetime to `_v2`. Post-cutover (no `id`): write ms+datetime to live + seconds to `_legacy`. PK-shape introspection per cycle (single PRAGMA call). Captures fresh `time.time()` per insert so v2 retains true sub-second precision. `check_for_spikes`: shifted now/window_start to ms units (in-process reader; runs every cycle). Stats display: magnitude-detect (`>1e12 -> ms`) so `cmd stats` works during dual-write and post-cutover. `cmd_export`: divides ms by 1000 at export to preserve `spike_scanner.db.price_history` seconds contract; comment documenting why. |
| `engines/mev_executor.py` | `get_recent_spikes` window calc shifted to ms units (line 207-209). `lookback` updated for consistency though unused. Comment header explains the Cycle 24 unit change. |
| `dashboards/data_collector.py` | `get_live_collector_stats` MIN/MAX ts -> datetime conversion now uses magnitude-detect (`>1e12 -> ms`). Same pattern as the live_collector stats display. Surfaced during cross-engine audit; not in original Brief but same break pattern as the three Brief-named sites. |
| `servers/praxis_mcp/server.py` | SIDECAR_DBS `live_collector.price_snapshots.timestamp_format`: `"s"` -> `"ms"`. The Brief implied the autodetect heuristic in `_to_latest_ms` would handle the unit change transparently, but autodetect only runs when format=`"auto"`; explicit `"s"` was hardcoded. Without this change, post-cutover `get_collector_health` would interpret ms as seconds, place `last_sample` in year ~58000, and report `is_stale=true` continuously. Schema comment block (~L65-71) updated to describe the new ms-precision schema. |
| `docs/SCHEMA_NOTES.md` | `price_snapshots` per-table prose: NONCONFORMING -> CONFORMING (Cycle 24, dual-write). Status table row updated to DONE-PARTIAL. Added the "no precision to recover" note and the four reader fixes. |
| `docs/SCHEMA_MIGRATION_PLAN.md` | Status row #8 -> DONE-PARTIAL with hash `6ca1796` + new row 24.5 for Phase 5 cleanup. Per-table spec rewritten with full cycle history including performance datapoints, lessons-learned, and the long-lived-process gotcha. Recipe section unchanged -- Cycle 24 surfaced no new gotchas worth durably documenting (the dual-write pattern carried cleanly to a new DB + new writer file). |
| `claude/TODO.md` | Cycle 24 entry added to "Recently closed" with full execution summary (replacing the prior Cycle 23 "(this cycle)" label which is now just plain "Cycle 23"). Active TODOs: replaced "Migrate `live_collector.price_snapshots`" entry with three new entries -- Cycle 24.5 (Phase 5 cleanup), Cycle 25 (smart_money.position_snapshots, with explicit "investigate long-lived-process pattern before Brief" note), and a spike-DB reader audit follow-up. Plus a `yes_bid`/`yes_ask`/`spread` writer-completion follow-up TODO. |

### Files Created

| File | Purpose |
|------|---------|
| `scripts/migrations/cycle24_price_snapshots_backfill_v2.py` | Phase 2 backfill. Pure-SQL INSERT-SELECT. `legacy_ts * 1000` is exact (integer multiply); `datetime` derived in pure SQL via `strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch')`. Idempotent: re-run on backfilled state prints "Already backfilled" and exits 0. Defense-in-depth: ensures `(slug, timestamp)` indexes exist on both tables before the NOT EXISTS subquery runs (per Cycle 23 lesson). |
| `scripts/migrations/cycle24_price_snapshots_verify.py` | Phase 3 verification: total rows (v2 >= legacy), every legacy `(slug, timestamp)` -> v2 `(slug, timestamp*1000)` (no missing, no duplicates), 100-sample byte-identity check on `yes_mid`/`yes_bid`/`yes_ask`/`spread`. Aborts with non-zero exit on any failure. |
| `scripts/migrations/cycle24_price_snapshots_cutover.py` | Phase 4 atomic RENAME pair in single transaction. Pre-condition checks (both tables exist, schemas haven't already migrated). Idempotent: re-run on cut-over state prints "Already cut over" and exits 0. |
| `data/live_collector.db.cycle24_backup` | Pre-Phase-0 full DB backup (97,529,856 bytes; md5 `bf5dc24477124e3782cf76f0a9303b8e` verified vs source). WAL + SHM also captured (`live_collector.db.cycle24_backup-wal`, `-shm`) since the live process had pending committed writes in the WAL. `crypto_data.db.cycle22_backup` was already absent at cycle start; `crypto_data.db.cycle23_backup` retained per rolling 1-behind retention. |
| `claude/retros/RETRO_price_snapshots_dual_write.md` | This file. |

---

## Phase-by-phase execution log

### Phase 0 (commit `b8fa847` at 03:08 UTC)

- v2 table CREATE in `init_db`: 7 columns, compound PK on
  `(slug, timestamp)`, idx on `(slug, timestamp DESC)`. Empty.
- Writer rewritten as runtime-adaptive dual-INSERT with PK-shape
  introspection (single PRAGMA call per cycle). Captures fresh
  `time.time()` per insert for both legacy and v2 timestamps.
- Three Brief-named reader fixes applied (in_process
  `check_for_spikes`, separate-process `mev_executor.py` window,
  stats display) + one audit-discovered fourth
  (`dashboards/data_collector.py`).
- `cmd_export` (spike-DB export) updated to divide ms by 1000 at
  export time.
- py_compile clean across all 3 files.
- Backup created (md5 verified): `data/live_collector.db.cycle24_backup`
  + WAL + SHM.
- **Phase 0 commit + push: `b8fa847` at 03:08 UTC.**
- **Critical operational step**: PraxisLiveCollector is a
  long-lived process; file changes do NOT auto-pick-up. Killed PID
  1864 manually (per Brief Step 1.10) so Task Scheduler relaunched
  with the new code. v2 row growth confirmed almost immediately:
  first v2 row at 03:04:28 UTC (slightly BEFORE the commit-push
  time -- Task Scheduler had already relaunched the process from
  the local edits before the commit landed remotely). One full
  cycle = 50 rows verified within ~2 min.
- PID 27308 was also visible in the initial `Get-Process` output
  but did not need killing -- only one process was actually polling
  Polymarket (proven by v2 growing exactly 50 rows/cycle, not 100).
  Likely a transient or zombie reporting artifact.

### Phase 1 (04:08 UTC -- 63 min after first dual-write row at 03:04:28 UTC)

- 60-min window 03:08 -> 04:08 UTC: legacy +3000 rows (50.0/min),
  v2 +3000 rows (50.0/min). Delta = 0; perfect cadence match.
- 5 random spot-checks (within the 60-min window): every
  (slug, near-second) pair matched between tables; `yes_mid`
  byte-identical (5/5); v2 ts within legacy_ts*1000 ..
  legacy_ts*1000 + 999 ms (5/5; drifts observed: 119, 172, 213,
  314, 666 ms).
- **Important precision-model correction vs Brief**: the Brief
  predicted "ts values within ~10ms of legacy*1000". Empirically
  the drift is the SUB-SECOND PORTION of `time.time()` at the
  moment of capture, NOT the difference between two consecutive
  `time.time()` calls. So drifts of e.g. 170, 732, 346, 963, 595 ms
  are correct (each is the sub-second portion of the wallclock
  second). The brief's mental model was "two consecutive
  time.time() calls are <10ms apart"; correct mental model is
  "sub-second portion of `int(time.time())` is whatever fraction of
  the second the snapshot landed on". The verify script's tolerance
  is `0 <= drift < 1000` (handled correctly).
- Total at this point: legacy 358,615; v2 3,200. v2 lagging legacy
  as expected (only the dual-write era; pre-Cycle-24 history hadn't
  been backfilled yet).

### Phase 2 (04:09 UTC) -- backfill

- Dry-run check: 358,661 legacy rows missing from v2 (legacy=358,665;
  the 3,250 v2 rows from the dual-write era covered only a fraction
  of the legacy history).
- Pure-SQL INSERT-SELECT executed. `legacy_ts * 1000` is exact
  integer multiply; `datetime` derived in pure SQL.
- Wall-clock: 2.243s for 358,661 rows. Brief budgeted ~30s;
  beat that by 13x. About one-third Cycle 23's 7.219s for ~4x rows
  -- the simpler integer-multiply (vs Cycle 23's
  julianday-with-ROUND) plus the smaller column ladder (7 vs 51
  columns) accounts for the speedup.
- Post-state: legacy 358,665 -> 358,665; v2 3,250 -> 361,911;
  missing post-backfill: 0.
- Re-ran for idempotency check: "Already backfilled" path triggered,
  exit 0, 0 rows inserted.

### Phase 3 (04:09 UTC) -- verification

- Check 1 (total rows): legacy=358,665, v2=361,911; v2 >= legacy. PASS.
- Check 2 (coverage): 0 legacy rows missing from v2; 0
  `(slug, timestamp)` duplicates in v2. PASS.
- Check 3 (byte-identity sample): 100/100 random rows passed.
  All `yes_mid`/`yes_bid`/`yes_ask`/`spread` byte-identical between
  legacy and v2.
- All checks PASS; OK to proceed to Phase 4.

### Phase 4 (04:10 UTC) -- atomic cutover

- Pre-cutover state: live=358,715, v2=361,961; live has `id`
  column, v2 doesn't. Sanity checks PASS.
- BEGIN; ALTER TABLE price_snapshots RENAME TO price_snapshots_legacy;
  ALTER TABLE price_snapshots_v2 RENAME TO price_snapshots; COMMIT.
- Wall-clock: 0.004s (vs Cycle 23's 0.005s; both essentially
  instantaneous).
- Post-cutover PRAGMA verification: live `price_snapshots` has new
  schema (no `id`, compound PK, datetime present). Renamed
  `price_snapshots_legacy` retains old schema (`id` column present).
  `_v2` no longer exists.
- **MCP SIDECAR_DBS update applied in same commit**: changed
  `timestamp_format` from `"s"` to `"ms"` in
  `servers/praxis_mcp/server.py`. Without this, post-cutover
  `get_collector_health` would have read `MAX(timestamp)` as
  seconds (`~1.78e12 seconds since epoch` -> year ~58000), reported
  staleness in the trillions of seconds, and `is_stale=true`. The
  Brief implied autodetect would handle this; in practice the
  config hardcoded `"s"`. Surfaced during the cross-engine audit
  before Phase 0; tracked separately.
- 2-min post-cutover wait: confirmed next collector iterations
  wrote to BOTH new live (`price_snapshots`, +150 rows in 3
  cycles) and renamed `price_snapshots_legacy` (+150 rows in same
  window). Runtime introspection automatically routed correctly
  (no `id` column on new live -> "post-cutover" branch fired;
  ms+datetime written to live, seconds written to `_legacy`).
  Latest live ts: `1777954367621` (2026-05-05T04:12:47.621 UTC,
  sub-second precision preserved). Latest `_legacy` ts:
  `1777954367` (same wall-clock second, sec-aligned).
- **In-situ reader verifications**:
  - `check_for_spikes` window query (replicated): for high-activity
    slug `will-gavin-newsom-win-the-2028-...`, 60-min window
    returns 115 rows out of 7,242 total for that slug (1.59%); the
    boundary is correctly applied. Pre-fix this would have returned
    100% (entire history matched).
  - `mev_executor.get_recent_spikes` window query (replicated):
    5-min window returns 5 rows for the same slug -- exactly the
    expected ~1/min cadence.
  - Stats display equivalent (replicated): first/last_dt magnitude-
    detect produces year-2026 dates (`2026-04-30T02:24:03Z` ->
    `2026-05-05T04:13:47Z`; 5-day duration), not year-58000.
  - Dashboard panel: same magnitude-detect logic, same result.
  - `get_collector_health` MCP tool: SIDECAR_DBS config updated
    `"s"` -> `"ms"` in the same Phase 4 commit so `_to_latest_ms`
    interprets the new ms timestamp correctly.
- **Data-shape note for Cycle 24.5**: the dual-write era (~63 min,
  ~3,250 sub-second-ms rows) coexists in the live table with the
  backfilled rows for the same era (~3,250 sec-aligned ms rows).
  Each (slug, wallclock-second) in that window has TWO rows in
  the live table -- one at e.g. `1777950348170` (sub-second from
  dual-write capture) and one at `1777950348000` (sec-aligned from
  backfill). Both have unique compound-PK so they coexist
  correctly. `check_for_spikes` and `mev_executor` are unaffected
  (both compute first/last in window; duplicate at same second
  with same yes_mid is benign). Counts in the dual-write era
  appear ~2x cadence vs surrounding eras -- documented for
  Cycle 24.5 reference.

---

## Lessons learned

### What was easier than Cycle 23

- **No julianday/ROUND mess**. Cycle 23's seconds-to-ms conversion
  via SQLite's `julianday * 86400000` produced a `double` that
  landed ~1 ULP below the integer for ~half of `.NNN`-precision
  datetimes; CAST AS INTEGER truncated, requiring a follow-up
  ROUND-correction UPDATE on 43,596 rows. Cycle 24's
  `legacy_ts * 1000` is integer math: exact, no ROUND needed.
- **No MCP tool bugs silently fixed by the migration**. Cycle 23
  surfaced two pre-existing buggy MCP tools that were unit-mismatched
  in non-obvious ways (`get_order_book_range` returning 0 for sane
  inputs). No analogous tool in this migration; the
  `get_collector_health` config is the only MCP-side touch point.
- **Recipe held up across a new DB + writer**. The dual-write
  pattern documented in `docs/SCHEMA_MIGRATION_PLAN.md` reapplied
  cleanly: rename of "asset" -> "slug" and removal of the 51-column
  ladder were trivial; runtime PK introspection worked verbatim;
  pure-SQL INSERT-SELECT pattern carried over.

### What was harder than Cycle 23

- **No hourly-restart boundary**. PraxisOrderBookCollector restarts
  every hour, so Cycle 23's writer change auto-activated within an
  hour of the Phase 0 push. PraxisLiveCollector is a continuous
  process; without an explicit kill, the new writer code was dead.
  Brief flagged this clearly; the Step 1.10 kill-and-relaunch was
  non-negotiable.
- **In-process reader at risk during Phase 0**. Cycle 22's
  `intrabar_predictor` reader fix was offline analytics -- could
  ship at any time. Cycle 24's `check_for_spikes` is in-process and
  fires every collector cycle. Phase 0 had to ship writer + reader
  fixes in a single atomic commit, with no margin for mid-flight
  fixes.
- **Cross-engine audit found a 4th reader**. The Brief named three
  reader sites; the audit (per Brief: "Run a final grep before
  declaring done") surfaced a fourth in
  `dashboards/data_collector.py` with the same break pattern.
  Caught and fixed in the same Phase 0 commit. Lesson reinforced:
  always run the cross-engine grep before committing Phase 0,
  even for a recipe-following cycle.

### Brief-vs-reality calibration notes

- Brief's "~10ms drift" prediction for spot-checks was wrong; actual
  drift is the sub-second portion of `time.time()` (0-999ms). The
  verify script tolerates the full sub-second range correctly.
- Brief's "the autodetect heuristic handles the unit change
  transparently" (criterion #20) was wrong; SIDECAR_DBS hardcoded
  `"s"`. Caught in the cross-engine audit and surfaced before
  Phase 0; the format change shipped with Phase 4.
- Brief's "Cycle 23 backup of crypto_data.db can be deleted at
  start of this cycle" (text) contradicts criterion #2 ("Cycle 23's
  backup remains; rolling 1-behind retention"). Went with the
  criterion: kept `crypto_data.db.cycle23_backup`. Cycle 22's
  backup was already absent.

### Cross-table sanity check (Cycle 21.5 defensive habit)

- `live_collector.price_snapshots` post-migration:
  0 `(slug, timestamp)` duplicates.
- `crypto_data.order_book_snapshots`: 0 `(asset, timestamp)` dups.
- `crypto_data.funding_rates`: 0 `(asset, timestamp)` dups.
- `crypto_data.ohlcv_1m`: 0 `(asset, timestamp)` dups.

### Notes for Cycle 25-26

- **Investigate long-lived-process pattern for smart_money and
  trades collectors before their Briefs are written**. Cycle 24's
  PraxisLiveCollector pattern was substantially different from
  PraxisOrderBookCollector's hourly restart; if smart_money and/or
  trades share the long-lived pattern, their Briefs need explicit
  kill-and-relaunch steps. TODO entry added.
- **Two-PRAGMA-overhead writer cost**: introspecting on every
  iteration is ~50 PRAGMA calls/min vs ~50 HTTP roundtrips/min;
  negligible. Cycle 24.5 cleanup could cache the resolution at
  process start; don't pre-optimize for Cycles 25-26 -- prefer the
  recipe verbatim.
- **Spike-DB seconds contract** is now an explicit out-of-scope
  audit item. If smart_money has a similar sidecar-of-sidecar
  pattern (e.g., a separate analytics DB consuming smart_money
  positions), audit it before the Cycle 25 Brief.
- **Schema-shape change for Cycle 25**: smart_money.position_snapshots
  has TEXT-only timestamp today (no INTEGER column). The migration
  will be a SCHEMA-SHAPE change, not just unit conversion -- the
  recipe's "convert legacy_ts seconds -> ms" step becomes "parse
  legacy ISO TEXT to ms via SQLite `strftime('%s', ...)*1000` or
  Python equivalent". The dual-write pattern still applies but the
  backfill SQL will look different.
