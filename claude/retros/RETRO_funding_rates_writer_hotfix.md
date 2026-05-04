# Retro: Cycle 21.5 -- funding_rates Writer Alignment Hotfix

**Brief:** `claude/handoffs/BRIEF_funding_rates_writer_hotfix.md`
**Date:** 2026-05-03
**Duration:** ~25 min (Mode B, surgical hotfix)
**Status:** COMPLETE
**Predecessor:** Cycle 21 (`b977cd3`, `765b38e`) -- funding_rates migration

---

## Summary

Cycle 21.5 fixes a duplicate-row bug introduced by Cycle 21's writer.
The post-migration `collect_funding_rates` stored Binance's
`fundingTime` raw, which carries sub-second reporting-clock jitter
(e.g., `1777795200003`). The migration script had converted legacy
seconds-since-epoch to ms via `* 1000`, which produced seconds-aligned
ms (`1777795200000`). The compound PK on `(asset, timestamp)` does
NOT collapse `.000` and `.NNN` into one row, so each new funding
event accumulated a duplicate row.

Two surgical changes resolved it:

1. **Writer truncation**: `engines/crypto_data_collector.py`
   `collect_funding_rates` now truncates `fundingTime` to
   seconds-aligned ms before storage
   (`ts = (int(r["timestamp"]) // 1000) * 1000`).
2. **One-shot dedup**: `scripts/migrations/cycle21_5_funding_rates_dedup.py`
   removed the 26 jittered rows that had accumulated between Cycle 21
   and the hotfix (`WHERE timestamp % 1000 != 0`), wrapped in a
   transaction with pre-DELETE byte-identity verification on
   `funding_rate` values within each duplicate group (lossless).

Cross-table sanity check (run BEFORE the dedup) confirmed the bug
pattern is isolated to `funding_rates`. `fear_greed`, `ohlcv_daily`,
`ohlcv_4h`, `market_data` all show 0 `(asset, datetime)` duplicates
and 0 jittered timestamps -- their writers feed off Binance's
bar-aligned `openTime` which has no jitter by contract.

No backup was created (lossless dedup, 26 rows, transactional script
with pre-DELETE verification).

---

## Changes Made

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `engines/crypto_data_collector.py` | `collect_funding_rates` writer: `ts = int(r["timestamp"])` -> `ts = (int(r["timestamp"]) // 1000) * 1000`. Added 5-line comment above explaining UTC-hour-boundary contract, why jitter has no info value, and why truncation matches the migration's representation (Cycle 21.5 hotfix attribution). | 451-462 |
| `docs/SCHEMA_NOTES.md` | `funding_rates` per-table prose: appended "Writer alignment (Cycle 21.5 hotfix)" bullet describing the truncation rule, why, and the one-shot 26-row dedup. Migration status table row unchanged (still CONFORMING / 21). | 92-100 |
| `docs/SCHEMA_MIGRATION_PLAN.md` | Added "Hotfix (Cycle 21.5)" subsection to Cycle 21 spec (#5): bug description, two-task fix summary, cross-table sanity check result, and the lessons-learned writer-alignment-audit prescription for future cycles. Status summary row #21 unchanged. | 142-176 |
| `claude/TODO.md` | Recently closed: added Cycle 21.5 entry above the existing Cycle 21 line. | 244-258 |

### Files Created

| File | Purpose |
|------|---------|
| `scripts/migrations/cycle21_5_funding_rates_dedup.py` | Idempotent dedup. Detects need via `COUNT(*) - COUNT(DISTINCT asset || '|' || datetime)`; if 0, prints "Already deduped" and exits 0. Otherwise: enumerates duplicate `(asset, datetime)` groups, verifies `funding_rate` is byte-identical within each group BEFORE deleting (lossless guarantee, aborts with exit 4 if any group disagrees), then `DELETE FROM funding_rates WHERE timestamp % 1000 != 0` inside a transaction with post-DELETE invariant checks (total == distinct, jittered_ts == 0, deleted_count == pre_dupes). Rolls back if any invariant fails. |
| `claude/retros/RETRO_funding_rates_writer_hotfix.md` | This file. |

### Backup decision

Per Brief: backup not required (lossless dedup, 26-row scope,
transactional script with pre-DELETE byte-identity verification + post-
DELETE invariant checks + ROLLBACK on failure). Code chose not to
create one. The `b977cd3` commit's `data/crypto_data.db.cycle21_backup`
is still on disk and remains the rollback point of last resort if
ever needed; it captures pre-Cycle-21 state which subsumes pre-Cycle-21.5.

---

## Migration verification

### Pre-state snapshot (just before any changes)

```
=== funding_rates state ===
total rows: 2240
distinct (asset|datetime): 2214
duplicates: 26
per-asset:        BTC: 1120  ETH: 1120
distinct dts:     BTC: 1107  ETH: 1107
jittered_ts (timestamp % 1000 != 0): 26

sample duplicate (asset, datetime) groups (latest first):
  ('BTC', '2026-05-03T08:00:00+00:00', 2)
  ('ETH', '2026-05-03T08:00:00+00:00', 2)
  ('BTC', '2026-05-03T00:00:00+00:00', 2)
  ('ETH', '2026-05-03T00:00:00+00:00', 2)
  ('BTC', '2026-05-02T16:00:00+00:00', 2)
  ('ETH', '2026-05-02T16:00:00+00:00', 2)

full row dump for one duplicate group (BTC 2026-05-03T08:00:00+00:00):
  ('BTC', 1777795200000, '2026-05-03T08:00:00+00:00', -1.398e-05)
  ('BTC', 1777795200003, '2026-05-03T08:00:00+00:00', -1.398e-05)
```

The dup-pair structure is exactly what the Brief described: one
`.000`-aligned legacy row from the migration + one `.NNN`-jittered row
from the post-Cycle-21 writer, both with byte-identical
`funding_rate` (because both came from the same Binance event).

### Cross-table sanity check (Brief AC #7) -- run BEFORE dedup

```
=== Cross-table duplicate sanity check ===
  fear_greed         dupes=0
  ohlcv_daily        dupes=0
  ohlcv_4h           dupes=0
  market_data        dupes=0
  funding_rates      dupes=26

=== Per-table jitter check (timestamp % 1000 != 0) ===
  fear_greed         jittered_ts=0
  ohlcv_daily        jittered_ts=0
  ohlcv_4h           jittered_ts=0
  market_data        jittered_ts=0
  funding_rates      jittered_ts=26
```

The bug is isolated to `funding_rates` exactly as the Brief
predicted. Binance's OHLCV `openTime` is bar-aligned by contract
(daily, 4h, 1m all `.000`). `market_data` writes use UTC midnight ms
computed in Python (no Binance timestamp at all). `fear_greed` uses
the Alternative.me API's `timestamp` (already aligned to days).

No follow-up cycle required.

### Dedup script: first run

```
[dedup] Opening C:\...\data\crypto_data.db
[dedup] Pre-state: total=2240, distinct=2214, dupes=26, jittered_ts=26
[dedup] Pre-state per-asset: [('BTC', 1120), ('ETH', 1120)]
[dedup] Found 26 duplicate (asset, datetime) groups
[dedup] First few: [('BTC', '2026-04-27T00:00:00+00:00', 2),
                    ('ETH', '2026-04-27T00:00:00+00:00', 2),
                    ('BTC', '2026-04-27T16:00:00+00:00', 2),
                    ('ETH', '2026-04-27T16:00:00+00:00', 2)]
[dedup] All duplicate groups have byte-identical funding_rate values.
        Dedup is lossless. Proceeding.
[dedup] DELETE FROM funding_rates WHERE timestamp % 1000 != 0
        -> 26 rows removed
[dedup] HOTFIX COMPLETE
  rows: 2240 -> 2214 (deleted 26)
  distinct (asset|datetime): 2214 -> 2214
  jittered_ts: 26 -> 0
  per-asset pre:  [('BTC', 1120), ('ETH', 1120)]
  per-asset post: [('BTC', 1107), ('ETH', 1107)]
```

The dup-events range from `2026-04-27T00:00:00+00:00` (earliest)
through `2026-05-03T08:00:00+00:00` (latest) -- ~6.3 days, matching
the Brief's "started accumulating immediately after Cycle 21 ran"
note. (Cycle 21 main commit `b977cd3` was earlier today, but the
post-Cycle-21 writer pulled `--days 365` from Binance on each
scheduled run, so the duplicates span the full re-fetch lookback
window from the first post-migration collector run forward. The
13 events per asset = 13 funding events written by the post-Cycle-21
writer between the migration finishing and the hotfix landing.)

### Dedup script: idempotent re-run

```
[dedup] Opening C:\...\data\crypto_data.db
[dedup] Pre-state: total=2214, distinct=2214, dupes=0, jittered_ts=0
[dedup] Pre-state per-asset: [('BTC', 1107), ('ETH', 1107)]
[dedup] Already deduped -- no duplicate (asset, datetime) groups found.
        Exiting cleanly.
exit=0
```

### Post-dedup verification

```
AC #4: total=2214, distinct=2214, equal? True
AC #6: timestamp % 1000 != 0 count: 0

Per-asset:
  BTC: 1107
  ETH: 1107

Latest 4 rows:
  ('BTC', 1777824000000, '2026-05-03T16:00:00+00:00', 7.03e-06)
  ('ETH', 1777824000000, '2026-05-03T16:00:00+00:00', 8.747e-05)
  ('BTC', 1777795200000, '2026-05-03T08:00:00+00:00', -1.398e-05)
  ('ETH', 1777795200000, '2026-05-03T08:00:00+00:00', -1.628e-05)

=== AC #7: cross-table check post-dedup ===
  fear_greed         dupes=0
  ohlcv_daily        dupes=0
  ohlcv_4h           dupes=0
  market_data        dupes=0
  funding_rates      dupes=0
```

Sample post-dedup rows showing `ms = seconds * 1000` representation
preserved across the dedup boundary:

```
('BTC', 1777248000000, '2026-04-27T00:00:00+00:00',  2.49e-06)
('ETH', 1777248000000, '2026-04-27T00:00:00+00:00', -6.636e-05)
('BTC', 1777593600000, '2026-05-01T00:00:00+00:00', -3.746e-05)
('ETH', 1777593600000, '2026-05-01T00:00:00+00:00', -4e-05)
('BTC', 1777795200000, '2026-05-03T08:00:00+00:00', -1.398e-05)
('ETH', 1777795200000, '2026-05-03T08:00:00+00:00', -1.628e-05)
```

All `.000`-aligned. `funding_rate` values match the pre-dedup
samples exactly.

### In-situ verification of the writer fix

A nice empirical bonus surfaced during verification: the
`2026-05-03T16:00:00+00:00` event appeared in the post-dedup state
with a `.000`-aligned timestamp (`1777824000000`), even though the
event happens at 16:00 UTC and the scheduled
`PraxisFundingCollector` run for that slot is `16:05` local Toronto.
Whichever collector run inserted that row did so AFTER the writer
fix was applied and produced a seconds-aligned ms -- in-situ
verification that the writer truncation works as intended.
Per-asset stable at 1,107 (no orphan jittered row from the new
event), total stable at 2,214 across the verification window.

---

## Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `engines/crypto_data_collector.py` `collect_funding_rates` writer truncates ts to seconds-aligned ms | PASS |
| 2 | Comment added explaining the truncation and why | PASS (5-line comment, references Cycle 21.5 + Binance contract) |
| 3 | `scripts/migrations/cycle21_5_funding_rates_dedup.py` exists, idempotent (re-run prints "Already deduped" + exit 0) | PASS |
| 4 | Post-dedup row count matches `COUNT(DISTINCT asset \|\| '\|' \|\| datetime)` exactly | PASS (2214 == 2214) |
| 5 | Per-asset count drops by 13 (BTC: 1,120 -> 1,107; ETH: 1,120 -> 1,107) | PASS (exactly 13 per asset = 26 total = pre_dupes) |
| 6 | All post-dedup `funding_rates.timestamp` values are divisible by 1000 | PASS (`SELECT COUNT(*) FROM funding_rates WHERE timestamp % 1000 != 0` returns 0) |
| 7 | Cross-table sanity check: ohlcv_daily, ohlcv_4h, market_data, fear_greed each show 0 `(asset, datetime)` duplicates | PASS (verified pre AND post dedup) |
| 8 | `docs/SCHEMA_NOTES.md` updated with the truncation note | PASS |
| 9 | `docs/SCHEMA_MIGRATION_PLAN.md` Cycle 21 spec updated with Hotfix subsection | PASS |
| 10 | `claude/TODO.md` "Recently closed" updated | PASS |
| 11 | Retro at `claude/retros/RETRO_funding_rates_writer_hotfix.md` includes pre/post counts, sample dupes, cross-table check | THIS FILE |
| 12 | All committable files ASCII-only (Rule 20) | PASS |
| 13 | Single-commit OR two-commit pattern (Code's choice) | Single-commit chosen (no doc references this hotfix's commit hash) |

---

## Test results

- Dedup script: 1 run + 1 idempotent re-run, both exit 0
- Pre-DELETE byte-identity check on `funding_rate` within each
  duplicate group: passed (all 26 groups had matching values, dedup
  proven lossless)
- Post-DELETE invariant checks (inside transaction): all passed
  (`total == distinct`, `jittered_ts == 0`,
  `deleted_count == pre_dupes`)
- Cross-table sanity check: ran twice (pre-dedup and post-dedup);
  both runs show the bug pattern isolated to funding_rates only
- Writer fix in-situ: the 16:00 UTC funding event of 2026-05-03 was
  written by the fixed writer with `.000`-aligned ts during the
  verification window

---

## Lessons learned (per Brief Notes for Code)

For future migration cycles, add a "writer alignment audit" step to
the cross-engine SQL audit (Task 4 in the standard cycle template):

- After the migration script and the writer change are both in
  place, fetch one fresh sample from the upstream API (or trace
  through the writer code path) and compare the resulting timestamp's
  representation to the migration's representation, byte-for-byte
  in the key-relevant bits.
- If they disagree, decide between:
  - **(a) Truncate in the writer** (Cycle 21.5's approach -- correct
    when the precision difference has no information value, like
    Binance's funding-event sub-second jitter)
  - **(b) Re-render the migration to match writer precision**
    (correct when the writer's precision IS the data, like a
    nanosecond-precision tick stream)
  - **(c) Accept duplicates with documentation** (rarely correct;
    PK invariants are usually load-bearing for downstream readers)
- For Cycle 21's pattern, (a) was correct.
- For the dual-write tables in Cycles 23-26, this question becomes
  load-bearing because the dual-write window means the new writer
  runs concurrently with the old writer for hours/days; any
  precision drift between them produces silent duplicates that the
  cutover step would then have to handle.

A weaker version of this audit would be: after each migration cycle,
sanity-check post-cycle row growth against the expected cadence
(e.g., funding_rates should grow by ~6 rows/day = 3 events x 2
assets, not by ~12 rows/day). The Brief caught this within hours;
a passive monitor on per-day row delta could catch it in minutes.
Consider adding to the next iteration of `get_collector_health` as
a "growth sanity" alarm. Out of scope for this cycle.

---

## Open items / next cycle inputs

- **Cycle 22**: Migrate `ohlcv_1m` per `docs/SCHEMA_MIGRATION_PLAN.md`
  row #6. ~530k rows; largest non-dual-write table; performance
  verification on the migration script is the main new variable.
  Apply the writer alignment audit prescription from this retro.
  Brief expected from Chat per the dual-Claude split.
- **Carryover**: `register_market_data_task.ps1` still needs an
  elevated PowerShell run from Jeff (Cycle 19 admin step).
- **phase3_models.joblib retrain**: still tracked as a Mid-priority
  TODO. The Cycle 21.5 dedup did not change `funding_rate` values
  for any event (lossless), so the retrain's view of the data is
  unchanged from Cycle 21's post-state.

---

## Deviations from Brief

- **None.** All 13 acceptance criteria met. Backup decision per
  Brief's "Code's call" -- chose not to create one given the
  pre-DELETE byte-identity check + transactional ROLLBACK already
  guarantee losslessness. Single-commit per AC #13 since no doc
  edit references this cycle's commit hash.
