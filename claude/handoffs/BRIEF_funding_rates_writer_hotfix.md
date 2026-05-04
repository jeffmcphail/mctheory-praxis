# BRIEF: Cycle 21.5 -- funding_rates Writer Alignment Hotfix

**Series:** praxis
**Cycle:** 21.5 (hotfix between Cycle 21 and Cycle 22)
**Mode:** B (Brief -> Code)
**Author:** Chat (praxis_main_current)
**Date:** 2026-05-03
**Predecessor:** Cycle 21 (`b977cd3`, `765b38e`) -- funding_rates migration

---

## Context

Cycle 21's migration introduced a subtle bug in `funding_rates` that
Chat caught during post-cycle independent verification.

**The bug**: post-migration, the writer in `collect_funding_rates`
stores Binance's funding event timestamp in raw milliseconds with
sub-second jitter intact (e.g., `1777795200003`). The migration
script converted legacy seconds-since-epoch to ms via `* 1000`, which
preserved seconds-alignment (e.g., `1777795200000`). When the new
writer fires for the same funding event, the timestamps differ by
~3-10 ms and the compound PK on `(asset, timestamp)` does NOT collapse
them via `INSERT OR REPLACE` -- so we accumulate duplicate rows.

**Empirical state at Brief-write time:**
- Total rows: 2,240
- Distinct `(asset, datetime)` events: 2,214
- Duplicates: 26 (one `.000`-ms legacy row + one `.NNN`-ms post-migration
  row per duplicate event, across 13 events x 2 assets BTC + ETH)
- Started accumulating immediately after Cycle 21 ran (2026-05-01 onward)

**Why this didn't bite the other migrated tables**: Binance's OHLCV
endpoints return kline `openTime` aligned exactly to bar boundaries
(daily/4h/1m all `.000`-aligned). Funding events are aligned to
hour boundaries by contract but Binance's reporting clock has
sub-second jitter that surfaces in the `fundingTime` field.
`fear_greed`, `ohlcv_daily`, `ohlcv_4h`, `market_data` are clean.

**Information value of sub-second precision**: zero. Binance's funding
contract specifies events at exact UTC hour boundaries (00:00, 08:00,
16:00). The `.NNN` ms tail is reporting jitter, not signal.

---

## Scope

Two tasks:

### Task 1: Fix the writer to truncate sub-second jitter

In `engines/crypto_data_collector.py` `collect_funding_rates()`
(around line 451-453, the post-Cycle-21 line numbers):

Current:
```python
ts = int(r["timestamp"])
dt = datetime.fromtimestamp(ts // 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
```

Change to:
```python
ts = (int(r["timestamp"]) // 1000) * 1000   # Truncate sub-second jitter; align to seconds
dt = datetime.fromtimestamp(ts // 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
```

The `(int(...) // 1000) * 1000` pattern explicitly drops the milliseconds
component, leaving an integer ending in `000`. This matches the legacy
migration data's representation, so future writes naturally collapse
into existing rows via the `INSERT OR REPLACE` PK semantics.

**Add a comment** above the new line explaining why:

```python
# Funding events are aligned to UTC hour boundaries by Binance contract.
# Binance's reporting clock includes sub-second jitter (e.g., .003 ms)
# which has no information value -- truncate to seconds-aligned ms so
# repeated INSERT OR REPLACE collapses rows for the same event,
# matching the migration's representation. (Cycle 21.5 hotfix.)
```

### Task 2: Deduplicate existing rows

`scripts/migrations/cycle21_5_funding_rates_dedup.py` (new file in
`scripts/migrations/`).

Idempotent script that:

1. Open `data/crypto_data.db` with explicit transaction control
   (Rule 34: fresh connection).
2. Detect whether dedup is needed by checking
   `SELECT COUNT(*) - COUNT(DISTINCT asset || '|' || datetime) FROM
   funding_rates`. If 0, print "Already deduped" and exit cleanly.
3. **For each duplicate `(asset, datetime)` group**: keep the row whose
   `timestamp` is divisible by 1000 (seconds-aligned), delete the
   others. This preserves the legacy migrated row and removes the
   post-Cycle-21 jitter rows. Both rows have identical `funding_rate`
   values (verified empirically; same Binance event), so this is a
   lossless collapse.
4. Verify post-dedup: `COUNT(*) == COUNT(DISTINCT asset || '|' ||
   datetime)` and the per-asset count drops by exactly the expected
   number.
5. Print before/after counts for the retro.

Backup is NOT required for this cycle -- the dedup is provably
lossless (duplicate pairs have identical funding_rate values), the
operation is small (26 rows currently), and the script is idempotent.
But still wrap the DELETE statements in a single transaction for
atomicity. If the cycle's verification fails post-dedup, ROLLBACK
restores pre-state.

If you prefer a backup anyway, copy to
`data/crypto_data.db.cycle21_5_backup` and delete after the cycle's
verification passes. Code's call.

### Task 3: Update doc trio

**`docs/SCHEMA_NOTES.md`** funding_rates section: add a one-paragraph
note about the writer alignment to `.000` ms ("Truncates Binance
funding event timestamps to seconds-aligned ms before storage; sub-
second jitter has no information value and would otherwise produce
duplicate rows for the same hourly event"). Migration status table:
no row change (still CONFORMING / 21).

**`docs/SCHEMA_MIGRATION_PLAN.md`** Cycle 21 per-table spec section:
add a "Hotfix (Cycle 21.5)" subsection noting:
- The writer initially preserved sub-second jitter, producing 26
  duplicate rows over 4 days
- Cycle 21.5 truncated the writer + deduped existing data
- Post-hotfix `(asset, timestamp)` PK semantics correctly collapse
  repeated writes for the same event

**`claude/TODO.md`**: add to "Recently closed" (above the existing
Cycle 21 entry):

```
- Cycle 21.5: funding_rates writer alignment hotfix. Caught during
  post-Cycle-21 independent verification: writer was preserving
  Binance's sub-second jitter (e.g., 1777795200003 vs migration's
  1777795200000), accumulating duplicate rows for each hourly event.
  Fixed via 2-task hotfix: writer now truncates to seconds-aligned ms
  before storage; deduplication script collapsed 26 existing dupes
  (lossless -- duplicate pairs had identical funding_rate values).
  Future migrations should sanity-check post-cycle row growth against
  expected cadence to catch this class of bug earlier.
```

### Task 4: Retro

`claude/retros/RETRO_funding_rates_writer_hotfix.md` with:

- Pre/post row counts (2,240 -> 2,214 expected)
- Per-asset breakdown (1,120 -> 1,107 each, expected)
- Sample of the duplicate rows BEFORE dedup, showing the
  `.000`-ms / `.NNN`-ms pair structure
- Sample query post-dedup confirming all timestamps are now
  divisible by 1000
- Writer change diff
- Cross-table check: confirm `ohlcv_daily`, `ohlcv_4h`, `market_data`,
  `fear_greed` are still clean (no `(asset, datetime)` duplicates).
  This is a sanity check, not a fix -- if any other table also has
  this issue, surface it as a follow-up cycle.
- Why no backup was created (or: confirmation that backup was created
  if Code chose to)
- Lessons learned (see Open Items below)

---

## Out of scope

- Migrating any new table
- Touching readers (the lstm_predictor.py DATE() group-by and
  funding.py autodetect both work transparently across `.000`-only
  and mixed `.000`/`.NNN` timestamp data)
- Modifying the migration script (`cycle21_funding_rates_to_v2.py`) --
  it ran correctly given its inputs; the bug was in the writer's
  post-migration behavior
- Running phase3 retrain (still tracked separately)
- Touching any other table's writer (`fear_greed`, `ohlcv_daily`,
  `ohlcv_4h`, `market_data` writers stay as-is; their source data is
  bar-aligned)

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | `engines/crypto_data_collector.py` `collect_funding_rates` writer truncates ts to seconds-aligned ms |
| 2 | Comment added explaining the truncation and why |
| 3 | `scripts/migrations/cycle21_5_funding_rates_dedup.py` exists, idempotent (re-run prints "Already deduped" + exit 0) |
| 4 | Post-dedup row count matches `COUNT(DISTINCT asset \|\| '\|' \|\| datetime)` exactly |
| 5 | Per-asset count drops by 13 (BTC: 1,120 -> 1,107; ETH: 1,120 -> 1,107). Note: actual numbers may differ if more dupes accumulate before Code runs the script -- verify the math (`pre - post == count_of_duplicate_groups_per_asset`) |
| 6 | All post-dedup `funding_rates.timestamp` values are divisible by 1000 (`SELECT COUNT(*) FROM funding_rates WHERE timestamp % 1000 != 0` returns 0) |
| 7 | Cross-table sanity check: ohlcv_daily, ohlcv_4h, market_data, fear_greed each show 0 `(asset, datetime)` duplicates |
| 8 | `docs/SCHEMA_NOTES.md` updated with the truncation note |
| 9 | `docs/SCHEMA_MIGRATION_PLAN.md` Cycle 21 spec updated with Hotfix subsection |
| 10 | `claude/TODO.md` "Recently closed" updated |
| 11 | Retro at `claude/retros/RETRO_funding_rates_writer_hotfix.md` includes pre/post counts, sample dupes, cross-table check |
| 12 | All committable files ASCII-only (Rule 20) |
| 13 | Single-commit OR two-commit pattern (Code's choice). No mid-cycle hash placeholder needed since the doc updates don't reference this cycle's commit hash explicitly. |

---

## Notes for Code

- This is a hotfix between Cycles 21 and 22. The cycle number in
  `docs/SCHEMA_MIGRATION_PLAN.md` status summary table stays at 21
  for funding_rates; the hotfix is a sub-bullet on Cycle 21's row.
- The dedup pattern (`keep the timestamp divisible by 1000, delete
  the rest`) is explicit and unambiguous because the legacy migration
  produced `.000`-aligned timestamps and the post-Cycle-21 writer
  produced `.NNN`-jittered ones. Don't second-guess this with a
  "keep the latest by inserted-order" or similar -- the explicit rule
  is correct.
- The cross-table sanity check in Task 4 retro is the critical
  defensive verification. If any OTHER table has duplicate
  `(asset, datetime)` groups (or for tables that don't have a `datetime`
  column, duplicate `(asset, MAX_OF_TIMESTAMP_TRUNCATED_TO_SECONDS)`
  groups), surface IMMEDIATELY -- that means Cycle 21's bug pattern
  is broader than thought.
- Specific defensive query for the cross-table check:
  ```sql
  -- For each migrated table, check for duplicate logical events
  SELECT 'fear_greed' AS t, COUNT(*) - COUNT(DISTINCT date) AS dupes FROM fear_greed
  UNION ALL SELECT 'ohlcv_daily', COUNT(*) - COUNT(DISTINCT asset || '|' || date) FROM ohlcv_daily
  UNION ALL SELECT 'ohlcv_4h', COUNT(*) - COUNT(DISTINCT asset || '|' || datetime) FROM ohlcv_4h
  UNION ALL SELECT 'market_data', COUNT(*) - COUNT(DISTINCT asset || '|' || date) FROM market_data
  ```
  All four should return `dupes = 0`.

- Lessons-learned entry for the retro: future migration cycles should
  include a "writer alignment audit" step in Task 4 (cross-engine SQL
  audit) that specifically asks: "Does the new writer produce
  timestamp values that are byte-identical in their key-relevant bits
  to the migrated legacy data?" If not, identify the source of drift
  and decide between (a) truncating in the writer, (b) re-rendering
  the migration to match the writer's precision, or (c) accepting
  duplicates with documentation. **For this cycle's pattern, (a) was
  correct.** For future tables (especially the dual-write tables in
  Cycles 23-26), this question becomes load-bearing.
