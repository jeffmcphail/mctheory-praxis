# Retro: Cycle 17 -- Temporal Standard Pilot (Rule 35 + fear_greed migration + onchain_btc monitoring + SCHEMA_NOTES.md)

**Brief:** `claude/handoffs/BRIEF_temporal_standard_pilot.md`
**Date:** 2026-04-30
**Duration:** ~1.5 hours (Mode B, surgical)
**Status:** COMPLETE

---

## Summary

Cycle 17 lands the Rule 35 temporal data storage standard, completes the
pilot migration of `fear_greed` to ms-since-epoch UTC PK schema, adds
`onchain_btc` to MCP `get_collector_health` monitoring (intentionally
alarming until a collector is registered), and creates `docs/SCHEMA_NOTES.md`
documenting all 17 tables across the 3 Praxis SQLite DBs with conformance
status. All 16 acceptance criteria pass. No deviations from the Brief.

The big-picture significance: Rule 34 (Cycle 15) addressed defensive READS
against schema heterogeneity; Rule 35 addresses the root cause by enforcing
canonical WRITES. After Rule 35 lands and all tables conform, the read
heterogeneity problem disappears. Cycle 17 is the pilot. Cycle 18 will
produce `docs/SCHEMA_MIGRATION_PLAN.md` ordering the remaining ~8 migrations.

---

## Changes Made

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `claude/CLAUDE_CODE_RULES.md` | v1.3 -> v1.4: new Rule 35 (Data Storage Rules subsection); changelog entry; Key Principles bullet; Retro Rules renumbered 35-41 -> 36-42; total rule count 41 -> 42 | header + 337-399 + 401-407 + 449-450 |
| `engines/crypto_data_collector.py` | `init_db` schema for fear_greed: `id` PK + `UNIQUE(timestamp)` -> `timestamp INTEGER PRIMARY KEY` (no id); `collect_fear_greed` writer: `ts = int(...) * 1000` (write ms instead of seconds) | 90-96, 397-398 |
| `servers/praxis_mcp/tools/meta.py` | `primary_monitored` accepts dict-spec entries (column + format + threshold), not only int thresholds; added `onchain_btc` entry with `(date, "date" format, 48h)`; `_collect_db_health` rewritten to use per-table `ts_col`/`ts_fmt`; `_to_latest_ms` accepts new `"date"` format (YYYY-MM-DD as UTC midnight) | 220-243, 270-374, 463-525 |
| `claude/TODO.md` | Closed 2 items in Recently closed (Cycle 17); replaced their entries in Active TODOs with Cycle 18 plan + lower-priority `onchain_btc` collector registration | 19-26, 70-78, 209-217 |

### Files Created

| File | Purpose |
|------|---------|
| `claude/scratch/audit_timestamp_timezone.py` | UTC audit (Task 1.5). Confirms every timestamp/datetime/date column across the 3 DBs represents a UTC moment. Gitignored per scratch convention. |
| `scripts/migrations/cycle17_fear_greed_to_v2.py` | Idempotent fear_greed migration. New durable home for all schema migration scripts. |
| `claude/scratch/dump_all_schemas_cycle17.py` | One-shot helper used to populate SCHEMA_NOTES.md. Gitignored. |
| `claude/scratch/verify_health_cycle17.py` | One-shot verification of `get_collector_health` for fear_greed and onchain_btc post-migration. Gitignored. |
| `docs/SCHEMA_NOTES.md` | Documentation of all 17 tables with Rule 35 conformance status and read-pattern guidance. ASCII-only. |
| `data/crypto_data.db.cycle17_backup` | Pre-migration backup (294 MB). |

### Key Decisions

- **Migration script lives in `scripts/migrations/`** (new directory). This
  becomes the home for all future schema migrations going forward, per the
  Brief's recommendation.
- **Extended `monitored_tables` dict in `meta.py` to support either an
  int (back-compat: timestamp column with auto ms/s format) or a dict
  spec (column + format + threshold).** This is additive -- existing
  entries still work as ints. Clean way to slot in `onchain_btc.date`
  without forking the helper.
- **Added a new `"date"` format to `_to_latest_ms`** that treats the value
  as `YYYY-MM-DD` UTC midnight. Reuses the existing helper rather than
  duplicating logic.
- **For the writer (`collect_fear_greed`)**: chose `ts = int(...) * 1000`
  followed by `datetime.fromtimestamp(ts // 1000, tz=timezone.utc)` for
  date derivation, per the Brief's "prefer the latter for consistency"
  note. Reads cleanly: stored value is ms; date is derived from ms.

---

## Test Results

### Passed

**AC1-3 (Rules file):**
- Version line shows `**Version:** 1.4`
- Rule 35 present at line 337 in Data Storage Rules subsection
- Retro Rules renumbered to 36-42
- Total rule count in Code section: **42** (verified via `awk` between section markers)

**AC4 (UTC audit):**
```
=== crypto_data.db ===
  [OK] ohlcv_1m: int=1777577040 -> UTC 2026-04-30T19:24:00+00:00; text='2026-04-30 19:24:00'; delta=0ms
  [OK] ohlcv_4h: int=1777521600 -> UTC 2026-04-30T04:00:00+00:00; text='2026-04-30 04:00:00'; delta=0ms
  [OK] ohlcv_daily: int=1777507200 -> UTC 2026-04-30T00:00:00+00:00; text='2026-04-30'; delta=0ms
  [OK] funding_rates: int=1777536000 -> UTC 2026-04-30T08:00:00+00:00; text='2026-04-30 08:00:00'; delta=0ms
  [OK] order_book_snapshots: int=1777577908 -> UTC 2026-04-30T19:38:28+00:00; text='2026-04-30T19:38:28.496000+00:00'; delta=496ms
  [OK] trades: int=1777577891436 -> UTC 2026-04-30T19:38:11.436000+00:00; text='2026-04-30T19:38:11.436Z'; delta=0ms
  [OK] fear_greed: int=1777507200 -> UTC 2026-04-30T00:00:00+00:00; text='2026-04-30'; delta=0ms
  [OK] onchain_btc: text='2026-04-28' parses as UTC 2026-04-28T00:00:00+00:00

=== live_collector.db ===
  [OK] price_snapshots: int=1777577885 -> UTC 2026-04-30T19:38:05+00:00 (no text col)

=== smart_money.db ===
  [OK] position_snapshots: text='2026-04-30T14:24:08.676736+00:00' parses as UTC 2026-04-30T14:24:08.676000+00:00

=== SUMMARY ===
All 10 tables OK (or empty).
```

(The `delta=496ms` on `order_book_snapshots` is the natural sub-second drift
between integer-seconds storage and the higher-precision text. Within the
2-second tolerance threshold; not a true mismatch. Logged for the record.)

**AC5-7 (Migration):**
- Pre-migration: 901 rows, ts_min=1699660800 (s), ts_max=1777507200 (s),
  latest row `(1777507200, '2026-04-30', 29, 'Fear')`
- Post-migration: 901 rows, latest `(1777507200000, '2026-04-30', 29, 'Fear')`
- Latest UTC delta: 0.0 seconds
- New schema: `[(0, 'timestamp', 'INTEGER', 0, None, 1), (1, 'date', 'TEXT', 1, None, 0), (2, 'value', 'INTEGER', 0, None, 0), (3, 'classification', 'TEXT', 0, None, 0)]`
  -- timestamp is the sole PK (col 5 = 1); no `id` column.
- **Idempotency verified:** second run prints `Already migrated -- fear_greed has new schema. Exiting cleanly.` and exits 0.

**AC8 (Writer change):**
```
389:def collect_fear_greed(days, conn):
397:        for d in data:
398:            ts = int(d.get("timestamp", 0)) * 1000
399:            date = datetime.fromtimestamp(ts // 1000, tz=timezone.utc).strftime("%Y-%m-%d")
```
And `init_db` schema now reads `timestamp INTEGER PRIMARY KEY` (no id, no UNIQUE).

**AC9 (Reader spot-test):**
```
$ python -c "...SELECT date, value FROM fear_greed..."
rows=901
first 3: [('2023-11-11', 70), ('2023-11-12', 73), ('2023-11-13', 72)]
last 3: [('2026-04-28', 33), ('2026-04-29', 26), ('2026-04-30', 29)]
```
This is the exact query from `engines/lstm_predictor.py:81`. Both `date` and
`value` semantics preserved across the migration; reader needs no changes.

**AC10/11 (MCP health):**
```
=== fear_greed ===
{
  "row_count": 901,
  "latest": "2026-04-30T00:00:00+00:00",
  "staleness_seconds": 71274.168,
  "threshold_seconds": 93600,
  "is_stale": false        <-- AC10 passes
}

=== onchain_btc ===
{
  "row_count": 364,
  "latest": "2026-04-28T00:00:00+00:00",
  "staleness_seconds": 244074.168,
  "threshold_seconds": 172800,
  "is_stale": true         <-- AC11 passes; intentional alarm per Brief
}

=== unmonitored ===
['market_data']

=== monitored table keys ===
['fear_greed', 'funding_rates', 'ohlcv_1m', 'ohlcv_4h', 'ohlcv_daily',
 'onchain_btc', 'order_book_snapshots', 'trades']
```
Note: the autodetect heuristic in `_to_latest_ms` (`> 1e12 -> ms`) handles
the now-ms `fear_greed.timestamp` correctly, so no explicit format spec
was needed for that table -- the existing autodetect path covers it.

**AC12 (SCHEMA_NOTES.md):** 12,636 bytes, ASCII-only, contains Rule 35
summary, Rule 34 summary, full inventory of 17 tables across 3 DBs,
migration status table, and stale-table notes.

**AC13 (TODO.md):**
- Active high-priority entry replaced with `Cycle 18: Write
  docs/SCHEMA_MIGRATION_PLAN.md and start migrating second table`.
- Active lower-priority entry added: `Register scheduled collector for
  onchain_btc table` with rationale (~2.7 days stale at 48h threshold).
- Recently closed gained `Cycle 17` entry summarizing the cycle and the
  two TODOs it closes.

**AC14 (ASCII-only):** All 6 modified/created committable files clean (0
non-ASCII lines via Grep tool's locale-clean check). The two scratch
helpers are gitignored so don't count, but they're also clean.

**AC15 (NEW_CHAT_README.md):** Unmodified; not in `git status`.

**AC16 (Retro):** This file. Includes audit output, migration before/after,
reader spot-check.

### Failed / Known Issues

None for this cycle. The intentional `is_stale=true` for `onchain_btc` is
expected behavior, not a failure -- it's the alarm the brief wanted lit.

---

## Failures & Debugging Trail

The cycle ran cleanly. Two minor friction points worth noting:

### 1. fastmcp import path
First attempt at `claude/scratch/verify_health_cycle17.py` used
`from fastmcp import FastMCP`, which is what the chat-side examples
sometimes show. The actual import in this repo is
`from mcp.server.fastmcp import FastMCP` (per `servers/praxis_mcp/server.py`).
Caught immediately on first run, fixed.

### 2. sys.path in scratch helper
Same script needed `sys.path.insert(0, str(REPO))` because running from
`claude/scratch/` doesn't see the `servers/` package. Added before the
relative imports. Standard scratch-script ergonomics; logging here so
future scratch scripts in this directory copy the pattern.

---

## Spot-grep for fear_greed.id dependencies

Per Brief Task 2 step 3: confirmed no other tables/engines depend on
`fear_greed.id`. Only one read-side query exists in `engines/lstm_predictor.py:81`:
`SELECT date, value FROM fear_greed`. Does not touch `id` or `timestamp`,
so the PK shape change is invisible to it.

The `crypto_predictor.py` references to `fear_greed_index` are columns
in a different table (the predictions table), not the standalone
`fear_greed` table -- out of scope per Brief.

---

## Commits

(Pending; Chat to commit after review.)

Suggested commit message:

```
Cycle 17: Rule 35 (temporal data storage standard) + fear_greed pilot migration

- CLAUDE_CODE_RULES.md v1.4: new Rule 35 in Data Storage Rules subsection
- fear_greed table migrated to ms-since-epoch UTC PK schema (901 rows)
- engines/crypto_data_collector.py writer updated to write ms (not seconds)
- onchain_btc added to MCP get_collector_health (48h threshold, alarming)
- docs/SCHEMA_NOTES.md created documenting all 17 tables x 3 DBs
- claude/TODO.md updated: closed 2 items, added Cycle 18 plan
- New scripts/migrations/ directory for future schema migrations
```

Files to commit:
- `claude/CLAUDE_CODE_RULES.md`
- `engines/crypto_data_collector.py`
- `servers/praxis_mcp/tools/meta.py`
- `claude/TODO.md`
- `docs/SCHEMA_NOTES.md` (new)
- `scripts/migrations/cycle17_fear_greed_to_v2.py` (new, includes
  `scripts/migrations/` directory)
- `claude/handoffs/BRIEF_temporal_standard_pilot.md` (new, currently
  untracked)
- `claude/retros/RETRO_temporal_standard_pilot.md` (this file)

Files NOT to commit (gitignored):
- `claude/scratch/audit_timestamp_timezone.py`
- `claude/scratch/dump_all_schemas_cycle17.py`
- `claude/scratch/verify_health_cycle17.py`
- `data/crypto_data.db.cycle17_backup`

---

## Open Items for Chat

- **Cycle 18 scope:** as queued. Produce `docs/SCHEMA_MIGRATION_PLAN.md`
  ordering the remaining ~8 nonconforming tables, then begin the next
  migration. Decision points worth pre-thinking before the Brief: which
  table second (suggest `ohlcv_daily` -- low row count, re-fetchable,
  date column already conforms, would be a quick second pilot to prove
  the stop-migrate-start pattern), and how to handle the highest-frequency
  tables (`trades`, `order_book_snapshots`, `price_snapshots`) that need
  dual-write -- specifically how to design the parallel collector path
  (env-var toggle? sibling script? config flag?).

- **Backup retention policy:** `data/crypto_data.db.cycle17_backup` is
  294 MB. Should it be deleted after a burn-in window (24-48h)? Or kept
  until Cycle 18 starts? Currently it lives alongside the live DB.

- **MCP server restart:** Per Rule 32, the running MCP subprocess in
  Claude Desktop won't pick up the meta.py change until a full
  Desktop relaunch. Smoke test (`python -m servers.praxis_mcp.test_smoke`)
  confirms the changes load cleanly out-of-process. The next time
  Desktop is restarted, `praxis:get_collector_health` will reflect
  the new monitoring shape.

- **Pre-Cycle-17 fear_greed.id semantics:** The `id` AUTOINCREMENT
  column held no semantic meaning anywhere in the codebase, so dropping
  it is safe. Logged here in case Chat wants to confirm before commit.

---

## Artifacts

- `data/crypto_data.db` -- fear_greed migrated; all other tables
  unchanged.
- `data/crypto_data.db.cycle17_backup` -- pre-migration snapshot for
  rollback if needed.
- `claude/scratch/audit_timestamp_timezone.py` -- durable in scratch;
  rerun anytime to re-verify UTC conformance.
- `scripts/migrations/cycle17_fear_greed_to_v2.py` -- idempotent;
  documented as the template for future migration scripts.
