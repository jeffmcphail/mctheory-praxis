# Retro: Cycle 21 -- funding_rates Migration

**Brief:** `claude/handoffs/BRIEF_funding_rates_migration.md`
**Date:** 2026-05-03
**Duration:** ~30 min (Mode B, surgical)
**Status:** COMPLETE
**Predecessors:** Cycle 20 (`ca316e3`, `6510650`) -- ohlcv_4h migration

---

## Summary

Cycle 21 migrates `funding_rates` (2,212 rows: 1,106 BTC + 1,106 ETH)
to Rule 35 using the simple stop-migrate-start pattern. Schema diff
identical to Cycles 18/20: drop `id` AUTOINCREMENT, compound
`PRIMARY KEY (asset, timestamp)`, timestamp seconds -> ms,
datetime rewritten naive `"YYYY-MM-DD HH:MM:SS"` -> ISO
`"YYYY-MM-DDTHH:MM:SS+00:00"`. Datetime re-derived from `timestamp`
via SQLite `strftime` (defense in depth, matching Cycle 20).

Latest UTC delta 0.0s; funding_rate values byte-identical;
timestamp post = pre * 1000 exactly. MCP `get_funding_rate_history`
returns rows correctly via the existing autodetect-aware reader,
no logic change required (only a comment-header refresh).
`get_collector_health` reports `funding_rates` healthy at ~11h
staleness against the 17h Cycle 14 threshold. Cross-engine SQL
audit found no engine has hardcoded seconds-since-epoch SQL
filters against `funding_rates` -- phase3 model retrain is
unblocked from this migration's perspective.

This was the first cycle in the migration series to run from a
proper Brief since Cycle 19. Cycle 20 self-dispatched in "Mode
B-lite" -- the Brief's process note flagged this as a deviation
("Mode B requires a Brief from Chat, every cycle, no exceptions")
and Cycle 21 returned to that pattern.

---

## Changes Made

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `engines/crypto_data_collector.py` | `init_db` funding_rates CREATE TABLE: `id` PK + `UNIQUE(asset, timestamp)` -> compound `PRIMARY KEY (asset, timestamp)`, no `id`. `collect_funding_rates` INSERT: `ts = int(r["timestamp"])` (keep API ms; was dividing by 1000) + `dt = ...strftime('%Y-%m-%dT%H:%M:%S+00:00')` (ISO with offset; was naive `'%Y-%m-%d %H:%M:%S'`). | 97-105, 451-465 |
| `servers/praxis_mcp/tools/funding.py` | Module-docstring schema header refreshed: "timestamp INTEGER (seconds), datetime TEXT" -> "timestamp INTEGER (UTC milliseconds), datetime TEXT (ISO +00:00). Compound primary key on (asset, timestamp)." Added an explanatory note that the runtime ms/sec autodetect (`ms_mode = ts_sample > 1e12`) is retained as belt-and-braces robustness for any auxiliary or test DB still on pre-migration seconds. No body logic change. | 1-12 |
| `docs/SCHEMA_NOTES.md` | `funding_rates` per-table prose: NONCONFORMING -> CONFORMING (Cycle 21). Added reader-transparency notes for `funding.py` (autodetect) and `lstm_predictor.py` (`DATE(datetime)` GROUP BY). Migration status table row updated. | 81-93, 266 |
| `docs/SCHEMA_MIGRATION_PLAN.md` | Cycle 21 row in Status summary: pending -> DONE. Per-table spec #5 rewritten: full reader inventory across all four engines, datetime rewrite recipe documented, Cycle 14 threshold verification recorded, cross-engine SQL audit result noted. | 23, 116-141 |
| `claude/TODO.md` | Closed Cycle 21 entry; replaced with Cycle 22 (ohlcv_1m migration); appended Cycle 21 to Recently closed with retrain coordination + threshold verification + Brief-discipline notes. | 21-25, 244-258 |

### Files Created

| File | Purpose |
|------|---------|
| `scripts/migrations/cycle21_funding_rates_to_v2.py` | Idempotent migration. Adapted directly from cycle20 with table/column shape adjusted for funding_rates. Cross-checks: row count (overall + per-asset), latest UTC delta, funding_rate value preservation, datetime ISO format + `+00:00` offset assertion, timestamp arithmetic check (post = pre * 1000 exactly). |
| `data/crypto_data.db.cycle21_backup` | Pre-migration full-DB backup (725,671,936 bytes; md5 matched live source via FileShare.ReadWrite read). Gitignored. Created BEFORE migration script ran. |
| `claude/retros/RETRO_funding_rates_migration.md` | This file. |

### Backup retention

Per the plan-doc retention rule (delete after the cycle-after-next
commit proves stable), Cycle 19 backup was deleted at the start of
this cycle. Cycle 20 backup is retained through Cycle 21's burn-in
window; deletable at the start of Cycle 22.

---

## Cross-engine SQL audit (Brief Task 4)

Grep across `engines/`, `scripts/`, `servers/`, `tests/`, `gui/` for
any reader of the `funding_rates` table:

| File | Pattern | Risk | Disposition |
|------|---------|------|-------------|
| `servers/praxis_mcp/tools/funding.py:47-63` | Raw SQL `SELECT * FROM funding_rates WHERE asset = ? AND timestamp >= ?` | Cutoff is computed at runtime via `ms_mode = ts_sample > 1e12`. No hardcoded epoch constant. | Reader-transparent. Comment refresh only. |
| `engines/lstm_predictor.py:86-90` | `SELECT DATE(datetime) as d, AVG(funding_rate), SUM(funding_rate) FROM funding_rates WHERE asset=? GROUP BY d` | Reads `datetime` text via SQLite `DATE()`. SQLite `DATE()` accepts both naive `'YYYY-MM-DD HH:MM:SS'` and ISO `'YYYY-MM-DDTHH:MM:SS+00:00'`. No `WHERE timestamp` clause; no hardcoded epoch constant. | Reader-transparent. Verified post-migration: `DATE(datetime)` query returned correct daily aggregates after the format change. |
| `engines/crypto_data_collector.py:97-105, 451-465` | Writer (init_db + INSERT) | -- | Updated this cycle. |
| `engines/regime_engine.py` | `funding_rates` is a Python parameter name (DataFrame/Series/np.ndarray). No SQL. | -- | Reader-transparent. |
| `engines/funding_rate_strategy.py` | `funding_rates` is a Python parameter name (pd.Series). No SQL. | -- | Reader-transparent. |
| `engines/cpo_training.py` | `funding_rates` is a Python parameter name passed to `regime_engine.compute(...)`. No SQL. | -- | Reader-transparent. |
| `tests/test_regime_engine.py` | `_make_funding_rates` synthesizes test data; not a DB reader. | -- | Reader-transparent. |
| `servers/praxis_mcp/tools/meta.py` | Mentions `funding_rates` in monitoring config (61,200s threshold). Reads MAX(timestamp) via the autodetect helper. | -- | Reader-transparent (autodetect handles ms). |
| `gui/funding_monitor/dashboard.py` | References `phase3_models.joblib` path; consumes models, not the SQLite table directly. | -- | Reader-transparent. |
| `scripts/run_cpo.py`, `scripts/run_carry.py` | Reference `phase3_models*.joblib` paths; do not query `funding_rates` directly. | -- | Reader-transparent. |

**Result:** no raw `WHERE timestamp` clauses against `funding_rates`
with hardcoded seconds-since-epoch constants found anywhere in
`engines/` or `scripts/`. The phase3 model retrain (TODO under Mid
priority in `claude/TODO.md`) consumes funding rates via the
DataFrame path through `regime_engine.compute(...)` /
`_compute_funding_features` -- those paths are reader-transparent
across the unit change. Retrain is unblocked from this migration's
perspective; the standalone retrain TODO remains tracked separately.

---

## Migration verification

### Pre/post snapshot

```
Pre-migration:
  rows: 2212 (BTC=1106, ETH=1106)
  ts_min: 1745971200    (seconds, 2025-04-30T00:00:00Z)
  ts_max: 1777795200    (seconds, 2026-05-03T08:00:00Z)
  latest row: ('BTC', 1777795200, '2026-05-03 08:00:00', -1.398e-05)
  datetime format: NAIVE 'YYYY-MM-DD HH:MM:SS'

Post-migration:
  rows: 2212 (BTC=1106, ETH=1106)
  ts_min: 1745971200000 (milliseconds)
  ts_max: 1777795200000 (milliseconds)
  latest row: ('BTC', 1777795200000, '2026-05-03T08:00:00+00:00', -1.398e-05)
  datetime format: ISO 'YYYY-MM-DDTHH:MM:SS+00:00'

Delta: 0.0 seconds; funding_rate value byte-identical;
timestamp_post = timestamp_pre * 1000 exactly.
```

### Schema (post-migration)

```
PRAGMA table_info(funding_rates):
  (0, 'asset',        'TEXT',    1, None, 1)   <- PK pos 1
  (1, 'timestamp',    'INTEGER', 1, None, 2)   <- PK pos 2
  (2, 'datetime',     'TEXT',    1, None, 0)
  (3, 'funding_rate', 'REAL',    0, None, 0)
```

No `id`. Compound PK on `(asset, timestamp)` -- subsumes the old
`UNIQUE(asset, timestamp)` constraint.

### Migration script: first run + idempotent re-run

First run:

```
[migrate] Opening C:\...\data\crypto_data.db
[migrate] Detected OLD schema. Proceeding.
[migrate] Pre-migration: rows=2212, ts_min=1745971200, ts_max=1777795200
[migrate] Pre-migration per-asset: [('BTC', 1106), ('ETH', 1106)]
[migrate] Pre-migration latest row: ('BTC', 1777795200, '2026-05-03 08:00:00', -1.398e-05)
[migrate] Pre-migration latest UTC: 2026-05-03T08:00:00+00:00
[migrate] Pre-migration latest datetime text: '2026-05-03 08:00:00'
[migrate] Creating funding_rates_new and copying rows (timestamp *= 1000, datetime ISO+00:00)...
[migrate] Post-copy rows in funding_rates_new: 2212
[migrate] Post-copy per-asset: [('BTC', 1106), ('ETH', 1106)]
[migrate] Post-copy latest row: ('BTC', 1777795200000, '2026-05-03T08:00:00+00:00', -1.398e-05)
[migrate] Post-copy latest UTC: 2026-05-03T08:00:00+00:00
[migrate] Post-copy latest datetime text: '2026-05-03T08:00:00+00:00'
[migrate] Dropping old funding_rates and renaming funding_rates_new...
[migrate] MIGRATION COMPLETE
  schema after migration: new
  rows: 2212 -> 2212
  latest pre:  asset=BTC ts=1777795200 (s)  UTC=2026-05-03T08:00:00+00:00  dt='2026-05-03 08:00:00'
  latest post: asset=BTC ts=1777795200000 (ms) UTC=2026-05-03T08:00:00+00:00  dt='2026-05-03T08:00:00+00:00'
  delta from pre: 0.0 seconds
```

Idempotent re-run:

```
[migrate] Opening C:\...\data\crypto_data.db
[migrate] Already migrated -- funding_rates has new schema. Exiting cleanly.
exit=0
```

### Spot-check rows pre/post

| asset | ts_pre (s)  | ts_post (ms)    | dt_pre               | dt_post                     | funding_rate |
|-------|-------------|-----------------|----------------------|-----------------------------|--------------|
| BTC   | 1745971200  | 1745971200000   | 2025-04-30 00:00:00  | 2025-04-30T00:00:00+00:00   | -5.181e-05   |
| ETH   | 1745971200  | 1745971200000   | 2025-04-30 00:00:00  | 2025-04-30T00:00:00+00:00   | -7.371e-05   |
| BTC   | 1777795200  | 1777795200000   | 2026-05-03 08:00:00  | 2026-05-03T08:00:00+00:00   | -1.398e-05   |

`ts_post == ts_pre * 1000` exactly; `dt_post` carries `+00:00`;
funding_rate values preserved.

### `funding.py` autodetect verification (Brief Task 3)

Direct invocation of `get_funding_rate_history(asset='BTC',
lookback_days=7)` post-migration:

```
sample ts=1745971200000, ms_mode=True, cutoff=1777230010000
rows returned: 20
first 3:
  {'asset': 'BTC', 'timestamp': 1777795200000, 'datetime': '2026-05-03T08:00:00+00:00', 'funding_rate': -1.398e-05}
  {'asset': 'BTC', 'timestamp': 1777766400000, 'datetime': '2026-05-03T00:00:00+00:00', 'funding_rate': 2.29e-05}
  {'asset': 'BTC', 'timestamp': 1777737600000, 'datetime': '2026-05-02T16:00:00+00:00', 'funding_rate': -2.54e-05}
```

`ms_mode=True` correctly identified post-migration. 20 rows returned
for BTC over the last 7 days (matches Binance's 3 funding events/day
* 7 days = 21, less the in-progress current event).

### `lstm_predictor.py` reader cross-check (datetime format change)

The only other raw-SQL reader of `funding_rates` is the `DATE(datetime)
GROUP BY` query at `engines/lstm_predictor.py:86-90`. Cross-checked
post-migration:

```
SELECT DATE(datetime) as d, AVG(funding_rate), SUM(funding_rate)
FROM funding_rates WHERE asset='BTC' GROUP BY d ORDER BY d DESC LIMIT 3:
  ('2026-05-03', 4.46e-06,    8.92e-06)
  ('2026-05-02', -2.8217e-05, -8.465e-05)
  ('2026-05-01', -2.8997e-05, -8.699e-05)
```

SQLite `DATE()` correctly parses the ISO `'YYYY-MM-DDTHH:MM:SS+00:00'`
format and groups by date. Reader-transparent across the format
change (no code update needed).

### MCP `get_collector_health` smoke test (Brief Task 5)

```json
{
  "tables": {
    "funding_rates": {
      "row_count": 2212,
      "latest": "2026-05-03T08:00:00+00:00",
      "staleness_seconds": 39645.341,
      "threshold_seconds": 61200,
      "is_stale": false
    }
  }
}
```

- `row_count`: 2212 (matches migration output)
- `latest`: `2026-05-03T08:00:00+00:00` (autodetect picked the
  ms branch; conversion back to ISO matches stored datetime)
- `staleness_seconds`: ~39,645s (~11h) -- the run window between
  the most recent funding event (08:00 UTC) and the verification
  call (~19:00 UTC)
- `threshold_seconds`: 61,200 (Cycle 14's 17h threshold, unchanged)
- `is_stale`: `false`

Cycle 14 staleness threshold remains appropriate. No threshold
change required this cycle.

---

## Acceptance Criteria

(Per the Brief.)

| # | Criterion | Status |
|---|---|---|
| 1 | `data/crypto_data.db.cycle21_backup` created BEFORE migration runs (md5-verified) | PASS |
| 2 | `data/crypto_data.db.cycle19_backup` deleted at start of cycle | PASS |
| 3 | `scripts/migrations/cycle21_funding_rates_to_v2.py` exists, idempotent (re-run exits 0 cleanly) | PASS |
| 4 | Pre/post migration row counts match exactly | PASS (2212 -> 2212; per-asset 1106/1106) |
| 5 | Pre/post latest UTC moments match (delta = 0s) | PASS (0.0s) |
| 6 | New schema: compound PK `(asset, timestamp)`, no `id`, ms timestamps, ISO+00:00 datetime | PASS |
| 7 | Spot-check 2-3 rows shows `timestamp = old_seconds * 1000` exactly | PASS (3 rows verified at extremes) |
| 8 | `engines/crypto_data_collector.py` `init_db` + `collect_funding_rates` updated | PASS |
| 9 | `servers/praxis_mcp/tools/funding.py` comment header updated | PASS |
| 10 | Cross-engine SQL spot-grep performed; results documented in retro | PASS (table above) |
| 11 | MCP `get_collector_health` reports `funding_rates` correctly post-migration | PASS (autodetect ms, is_stale false) |
| 12 | `funding.py` `get_funding_rate_history()` returns rows correctly post-migration | PASS (20 rows for BTC/7d) |
| 13 | `docs/SCHEMA_NOTES.md` updated | PASS |
| 14 | `docs/SCHEMA_MIGRATION_PLAN.md` updated | PASS |
| 15 | `claude/TODO.md` updated (close Cycle 21, add Cycle 22) | PASS |
| 16 | All committable files ASCII-only (Rule 20) | PASS |
| 17 | Retro at `claude/retros/RETRO_funding_rates_migration.md` | THIS FILE |
| 18 | Two-commit pattern OK (main + hash patch) per Cycle 18/20 precedent, OR amend-before-push | Two-commit pattern chosen (matches Cycle 18/19/20 precedent; plan-doc + retro carry `<TBD>` for the main commit, follow-up commit patches the hash) |

---

## Test results

- Migration script: 1 run + 1 idempotent re-run, both exit 0
- Schema cross-check: PRAGMA table_info confirms compound PK shape
- Row-count cross-check: 2212 -> 2212 (overall and per-asset BTC/ETH)
- Latest-UTC cross-check: 0.0s delta (8-hour funding events are
  exact UTC multiples; conversion is loss-free)
- Datetime cross-check: derived ISO matches stored, `+00:00` present;
  SQLite `DATE()` parses both formats so the `lstm_predictor.py`
  GROUP BY reader is unaffected
- funding_rate cross-check: latest-row value preserved byte-for-byte
- Timestamp arithmetic cross-check: `new_ms == old_s * 1000` exactly
- MCP smoke test: `funding_rates` reports correctly via autodetect;
  `get_funding_rate_history` returns rows in ms-mode for BTC/7d

---

## Open items / next cycle inputs

- **Cycle 22**: Migrate `ohlcv_1m` (~530k rows; simple
  stop-migrate-start, but the largest non-dual-write table). Same
  shape as the other ohlcv_* tables; reader audit is the load-bearing
  step (multiple LSTM/quant strategies consume this; grep before
  migrating). Brief expected from Chat per the dual-Claude split
  re-affirmed in this cycle's Brief.
- **Plan-doc commit hash**: row #21 reads `<TBD>`. Patch with the
  actual commit hash in a follow-up commit (or amend before push),
  same as Cycles 18/19/20.
- **Carryover from Cycle 19**: `register_market_data_task.ps1` still
  needs an elevated PowerShell run from Jeff. Independent of Cycle 21;
  remains in Active TODOs.
- **phase3_models.joblib retrain**: cross-engine SQL audit confirmed
  no engine has hardcoded seconds-since-epoch SQL filters against
  `funding_rates`. The standalone retrain TODO (Mid priority in
  `claude/TODO.md`) is unblocked from this migration's perspective.
  Brief writing for the retrain is independent.

---

## Deviations from Brief

- **None.** Recipe followed exactly. All 18 acceptance criteria met.
- One observation worth recording for future cycles: the live
  `crypto_data.db` is held by an active process (the running
  scheduled collectors / MCP server) such that PowerShell's default
  `Get-FileHash` errors with "process cannot access the file." The
  workaround used was an explicit
  `[System.IO.File]::Open(path, 'Open', 'Read', 'ReadWrite')`
  stream into `MD5.ComputeHash`. The backup file itself opened
  cleanly via the standard `Get-FileHash`. Both hashes matched
  (`7F7FE97177BBA9C60FC963EE4756F4EA`). Worth noting for Cycles 22+
  on the same DB; future migration scripts could optionally add a
  built-in MD5 check rather than relying on PowerShell-side tooling.
