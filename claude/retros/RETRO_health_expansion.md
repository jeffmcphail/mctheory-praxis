# Retro: Expand MCP get_collector_health for Cycle 10 Scheduled Tables

**Series:** praxis
**Cycle:** 11
**Brief:** `claude/handoffs/BRIEF_health_expansion.md`
**Outcome:** PASS -- all acceptance criteria met
**Files modified:** 1 (`servers/praxis_mcp/tools/meta.py`)
**Files added (gitignored):** 1 (`claude/scratch/verify_health_expansion.py`)

---

## 1. What was done

Expanded `get_collector_health()` in `servers/praxis_mcp/tools/meta.py` to:

1. Added 4 entries to `monitored_tables` covering the Cycle 10 scheduled
   collectors (`funding_rates`, `fear_greed`, `ohlcv_daily`, `ohlcv_4h`)
   with cadence-derived staleness thresholds in seconds.
2. Replaced the hardcoded `unmonitored_tables = ["funding_rates", "fear_greed"]`
   list with a dynamic computation: `all_db_tables - monitored - sqlite_internal`.
3. Updated the inline comment block to describe all seven monitored
   collectors and the dynamic `unmonitored` derivation. The pre-Cycle-10
   claim that `funding_rates` and `fear_greed` are populated only by
   manual CLI runs is no longer true and was removed.

No other file was modified.

---

## 2. Diff

```diff
diff --git a/servers/praxis_mcp/tools/meta.py b/servers/praxis_mcp/tools/meta.py
index 214223b..b05d371 100644
--- a/servers/praxis_mcp/tools/meta.py
+++ b/servers/praxis_mcp/tools/meta.py
@@ -155,24 +155,54 @@ def register(mcp, db_path: Path):
             # Tables with active scheduled collectors. Threshold = natural
             # cadence + slack for scheduler jitter / windowed-gap behavior.
             #
-            # trades                -- PraxisTradesCollector, continuous 10s.
-            # order_book_snapshots  -- PraxisOrderBookCollector, 10s on-hour
-            #                          but currently 1h-on/1h-off pattern
-            #                          (see retro). 65 min tolerance covers
-            #                          the worst-case sampling moment.
+            # trades                -- PraxisTradesCollector, continuous 30s
+            #                          (now 3550s windowed per Cycle 10
+            #                          patch matching OrderBook).
+            # order_book_snapshots  -- PraxisOrderBookCollector, 10s on-hour,
+            #                          3550s windowed (Cycle 8 fix). 65 min
+            #                          tolerance covers the worst-case
+            #                          sampling moment plus inter-window gap.
             # ohlcv_1m              -- PraxisCrypto1mCollector, 6h batch.
             #                          7h tolerance covers batch + slack.
+            # funding_rates         -- PraxisFundingCollector, 8h cadence
+            #                          aligned approximately to Binance
+            #                          funding events (Cycle 10). 9h
+            #                          tolerance covers cadence + slack.
+            # fear_greed            -- PraxisFearGreedCollector, daily at
+            #                          00:30 local (Cycle 10). 26h tolerance.
+            # ohlcv_daily           -- PraxisOhlcvDailyCollector, daily at
+            #                          00:15 local (Cycle 10). 26h tolerance.
+            # ohlcv_4h              -- PraxisOhlcv4hCollector, daily at
+            #                          00:20 local (Cycle 10). 26h tolerance.
+            #                          (Daily refresh of 4h bars; the cadence
+            #                          is the refresh frequency, not the bar
+            #                          frequency.)
             #
-            # funding_rates and fear_greed are NOT monitored: no scheduled
-            # task writes to them. Populated only by manual one-shot CLI
-            # runs (python -m engines.crypto_data_collector collect-funding /
-            # collect-fear-greed). Listed under `unmonitored` below.
+            # Tables NOT in this dict are computed dynamically below as
+            # `unmonitored` -- the set of tables in the DB minus monitored
+            # minus SQLite internals. Currently this captures onchain_btc
+            # (no scheduled collector) and market_data (legacy/empty).
             monitored_tables = {
                 "trades": 120,
                 "order_book_snapshots": 3900,
                 "ohlcv_1m": 25200,
+                "funding_rates": 32400,    # 8h + 1h slack
+                "fear_greed": 93600,       # 24h + 2h slack
+                "ohlcv_daily": 93600,      # 24h + 2h slack
+                "ohlcv_4h": 93600,         # 24h + 2h slack
             }
-            unmonitored_tables = ["funding_rates", "fear_greed"]
+
+            # Dynamically compute unmonitored: every table in the DB that
+            # isn't monitored and isn't a SQLite internal table.
+            cursor.execute(
+                "SELECT name FROM sqlite_master WHERE type='table' "
+                "ORDER BY name"
+            )
+            all_tables = {row["name"] for row in cursor.fetchall()}
+            sqlite_internal = {"sqlite_sequence"}
+            unmonitored_tables = sorted(
+                all_tables - set(monitored_tables.keys()) - sqlite_internal
+            )

             now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
             result = {
```

---

## 3. Phase 0 -- Pre-edit smoke test (baseline)

Command: `python -m servers.praxis_mcp.test_smoke`

Result: **Smoke test PASSED.**

Tables observed in the DB at the time of the baseline:
fear_greed (900), funding_rates (2190), market_data (0), ohlcv_1m (518604),
ohlcv_4h (10800), ohlcv_daily (1800), onchain_btc (364),
order_book_snapshots (4577), sqlite_sequence (8), trades (104708).

---

## 4. Phase 2 -- Post-edit smoke test

Command: `python -m servers.praxis_mcp.test_smoke`

Result: **Smoke test PASSED.**

Tables observed at the post-edit run (some counts moved due to the live
trades and order_book_snapshots collectors filling rows in between the
two smoke runs):
fear_greed (900), funding_rates (2190), market_data (0), ohlcv_1m (518604),
ohlcv_4h (10800), ohlcv_daily (1800), onchain_btc (364),
order_book_snapshots (4583), sqlite_sequence (8), trades (105005).

The 6-row delta on order_book_snapshots and 297-row delta on trades
between Phase 0 and Phase 2 confirms the live collectors are running
through this Brief, which is expected -- this Brief did not touch any
collector and the smoke test is read-only.

---

## 5. Phase 2 -- verify_health_expansion.py

Command: `python claude/scratch/verify_health_expansion.py`

Output:

```
=== Verification: get_collector_health expansion ===

PASS: monitored set is exactly ['fear_greed', 'funding_rates', 'ohlcv_1m', 'ohlcv_4h', 'ohlcv_daily', 'order_book_snapshots', 'trades']
PASS: unmonitored set is exactly ['market_data', 'onchain_btc']

--- Per-table status for the four new monitored tables ---
  funding_rates: rows=2190 latest=2026-04-29T16:00:00+00:00 staleness=17548.225s threshold=32400s stale=False
  fear_greed: rows=900 latest=2026-04-29T00:00:00+00:00 staleness=75148.225s threshold=93600s stale=False
  ohlcv_daily: rows=1800 latest=2026-04-29T00:00:00+00:00 staleness=75148.225s threshold=93600s stale=False
  ohlcv_4h: rows=10800 latest=2026-04-29T16:00:00+00:00 staleness=17548.225s threshold=93600s stale=False

=== Verification complete ===
```

All four new monitored tables report healthy state (`is_stale=False`)
with reasonable staleness values relative to their thresholds. The
monitored and unmonitored sets are exactly what Cycle 10's landscape
predicts.

---

## 6. Acceptance criteria checklist

| # | Criterion | Status |
|---|---|---|
| 1 | `meta.py` modified with 7-entry `monitored_tables` and dynamic `unmonitored_tables` | PASS |
| 2 | Comment block updated for Cycle 10 collectors | PASS |
| 3 | Phase 0 smoke test passed (baseline) | PASS |
| 4 | Phase 2 smoke test passed (post-edit) | PASS |
| 5 | `verify_health_expansion.py` PASSes monitored + unmonitored sets | PASS |
| 6 | No edits outside `meta.py` (scratch helper is gitignored) | PASS |
| 7 | Retro at `claude/retros/RETRO_health_expansion.md` | PASS (this file) |

---

## 7. Observations

- The DB has 10 tables total at the time of writing. After the change,
  7 are monitored and 2 are unmonitored (`market_data` legacy/empty,
  `onchain_btc` no scheduled collector); `sqlite_sequence` is filtered
  out by the SQLite-internal exclusion set. 7 + 2 + 1 = 10, matches.
- The dynamic computation is self-maintaining: any new table added to
  the DB will appear in `unmonitored` automatically the next time
  `get_collector_health()` is called, surfacing the new table without
  flagging false-positive staleness on it.
- `onchain_btc` uses a `date` column rather than `timestamp`, so the
  existing fallback path in `get_collector_health()` for monitored
  date-only tables would handle it if it were ever monitored. Today
  it's unmonitored, so no per-table staleness call happens for it.
- All four new threshold constants are in seconds, matching the existing
  convention.

---

## 8. Notes for Jeff

The MCP server runs as a subprocess of Claude Desktop. **You will need
to restart Claude Desktop** for the running MCP server instance to pick
up the change in `meta.py`. The on-disk file is updated; the live
process is still on the pre-edit code.

After the restart, calling `get_collector_health()` from a Claude
Desktop conversation will show:
- 7 entries in `tables` (was 3)
- `unmonitored: ["market_data", "onchain_btc"]` (was
  `["funding_rates", "fear_greed"]`)

If any of the four newly-monitored tables shows `is_stale=True` after
Claude Desktop is restarted, the most likely cause is a Task Scheduler
issue with the corresponding `services/register_*.ps1` registration --
not the MCP code. Check the Task Scheduler GUI for "Praxis Funding
Collector", "Praxis Fear Greed Collector", "Praxis OHLCV Daily
Collector", and "Praxis OHLCV 4h Collector".

---

## 9. What this Brief did NOT change

- No edits to `funding.py`, `ohlcv.py`, `order_book.py`, `raw.py`,
  `trades.py`, `server.py`, `db.py`, `test_smoke.py`, or `__init__.py`.
- Did not change the thresholds for the three pre-Cycle-10 tables
  (`trades`, `order_book_snapshots`, `ohlcv_1m`); their original
  Cycle 7 values are preserved.
- Did not address the `onchain_btc.date` schema heterogeneity or the
  `trades.timestamp` ms-vs-s heterogeneity (separate cycles).
- Did not touch the `claude_desktop_config.json` (lives in MSIX
  virtualized path; out of repo).
