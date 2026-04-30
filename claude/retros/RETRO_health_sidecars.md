# Retro: get_collector_health Cycle 14 -- Sidecar DB Monitoring + Funding Threshold Fix

**Series:** praxis
**Cycle:** 14
**Mode:** A (Chat-edited delta zip; verifiable in sandbox)
**Outcome:** PASS -- delivered as delta zip; runtime verification deferred to live MCP after Claude Desktop relaunch
**Files modified:** 2 (`servers/praxis_mcp/server.py`, `servers/praxis_mcp/tools/meta.py`)
**Files added:** 0

---

## 1. What was done

Three coupled changes in `get_collector_health()`:

1. **Widened `funding_rates` threshold** from 32400s (9h) to 61200s (17h).
   Cycle 11 picked 9h thinking 8h cadence + 1h slack, but the actual
   gap between (now) and (latest data point in DB) can legitimately
   approach 16h depending on which scheduled run last touched the
   table. Three runs/day at 00:05/08:05/16:05 local Toronto, with
   Binance funding events at 00:00/08:00/16:00 UTC, makes 17h the
   correct conservative ceiling.

2. **Added sidecar DB monitoring** for the two PraxisLiveCollector and
   PraxisSmartMoney DBs reactivated in Cycle 13. Each sidecar has its
   own `threshold_seconds`, `timestamp_column`, and `timestamp_format`
   spec, since their schemas don't follow the primary `crypto_data.db`
   conventions. The response now includes a top-level `databases`
   dict scoped per DB, with the legacy top-level `tables` and
   `unmonitored` keys preserved for backward compat with the primary
   DB.

3. **Added ISO TEXT timestamp parsing** in the new helper
   `_to_latest_ms()`. `smart_money.db.position_snapshots.timestamp`
   stores values like `"2026-04-29 22:25:24.71"` (no tz suffix; treat
   as UTC). The pre-Cycle-14 `if latest > 1e12 elif latest > 1e9`
   logic only handled numeric. New helper handles ISO text via
   `datetime.fromisoformat()` with space-to-T separator handling and
   fractional-second fallback.

---

## 2. Diff summary

`servers/praxis_mcp/server.py`:
- Added `LIVE_DB_PATH` and `SMART_MONEY_DB_PATH` near `DB_PATH`
- Added `SIDECAR_DBS` config dict with monitoring specs for
  `live_collector.price_snapshots` (180s, "s" format) and
  `smart_money.position_snapshots` (28800s, "iso_text" format)
- Updated the `meta.register()` call to pass `sidecar_dbs=SIDECAR_DBS`
- Module docstring updated to mention Cycle 14

`servers/praxis_mcp/tools/meta.py`:
- `register()` signature extended with optional `sidecar_dbs: dict = None`
  for backward compat
- `funding_rates` threshold in `primary_monitored` dict: 32400 -> 61200
  with new comment block explaining the cadence vs data-arrival gap
- `get_collector_health()` body refactored to call two new helpers:
  - `_collect_db_health()` for the primary DB (preserves shape)
  - `_collect_db_health_sidecar()` for sidecars (explicit ts spec
    rather than autodetect)
- New `_to_latest_ms()` helper unifies ms/s/iso_text parsing across
  primary + sidecar code paths
- Response shape adds new top-level `databases` dict alongside the
  existing `tables` + `unmonitored` (which still describe the
  primary DB only, for compat)

Total: 121 lines server.py (was 70), 517 lines meta.py (was 314).

---

## 3. ASCII compliance + syntax

```
$ python -c "import ast; ast.parse(open('server.py').read()); print('OK')"
OK
$ python -c "import ast; ast.parse(open('tools/meta.py').read()); print('OK')"
OK
$ grep -P "[^\x00-\x7F]" server.py tools/meta.py
(empty -- no non-ASCII)
```

Both files compile and are ASCII-only per Rule 20.

---

## 4. Verification approach

This delta lands as Mode A (Chat-edited zip). Live verification happens
post-Claude-Desktop-relaunch via `get_collector_health()`. Expected
results:

**Primary DB (`crypto_data` in databases dict + top-level tables):**
- 7 monitored tables (unchanged from Cycle 11)
- `funding_rates` no longer reports `is_stale: true` immediately
  (threshold widened to 17h; current ~10.5h staleness now passes)
- `fear_greed`, `ohlcv_daily` will RESOLVE from is_stale: true once
  Jeff force-runs them (sequence below) or once their natural
  triggers fire at 00:15 / 00:30 local tomorrow

**Sidecar `live_collector`:**
- `price_snapshots`: should report ~30-90s staleness against 180s
  threshold (60s sample cadence; 1-2 samples from current). NOT stale.
- `unmonitored`: tracked_markets, collection_log, spike_alerts (these
  are auxiliary tables, not the primary write target)

**Sidecar `smart_money`:**
- `position_snapshots`: staleness depends on when last 6h trigger fired.
  ISO TEXT parsing should produce a valid datetime and threshold
  comparison.
- `unmonitored`: tracked_wallets, position_changes, convergence_signals

---

## 5. Outstanding from end of Cycle 13 / now in Cycle 14 scope

- **Three tables flagged stale in Cycle 13's diagnostic call**:
  - `funding_rates` (10.5h vs 9h threshold) -> resolved by widening
    threshold to 17h. No data action needed.
  - `fear_greed` (26.5h vs 26h threshold) -> NOT resolved by code
    change; needs `Start-ScheduledTask -TaskName PraxisFearGreedCollector`
    or wait for 00:30 local natural trigger.
  - `ohlcv_daily` (26.5h vs 26h threshold) -> NOT resolved by code
    change; needs `Start-ScheduledTask -TaskName PraxisOhlcvDailyCollector`
    or wait for 00:15 local natural trigger.
  - `ohlcv_4h` (similar shape; not currently flagged but force-run
    while we're at it for parity).

Recommendation: force-run all three immediately after applying the
delta zip + Claude Desktop relaunch, to clear the alarms before
midnight.

---

## 6. Open items for future cycles

- **Cycle 15 candidate**: docs/SCHEMA_NOTES.md documenting timestamp
  heterogeneity across all 12 known tables (primary + 2 sidecars).
  Schema heterogeneity has now appeared in 4 retros (Cycles 9, 10, 11,
  14).
- **Sidecar test coverage**: add `tests/test_sidecar_health.py` to
  catch regressions in `_to_latest_ms()` ISO TEXT parsing edge cases
  (timezone-suffixed, non-fractional, malformed).
- **`atlas_search` on Claude.ai chat surface**: still doesn't surface
  in this chat though it works in Claude Desktop. Out of scope for
  Cycle 14 but documented as known asymmetry.

---

## 7. Notes for Jeff

1. **Apply the delta zip**:
   ```powershell
   cd C:\Data\Development\Python\McTheoryApps\praxis
   Expand-Archive -Path "$env:USERPROFILE\Downloads\praxis_delta_cycle14_health_sidecars.zip" -DestinationPath . -Force
   git diff servers/praxis_mcp/server.py servers/praxis_mcp/tools/meta.py | Select-Object -First 60
   ```

2. **Force-run the stale daily tasks**:
   ```powershell
   Start-ScheduledTask -TaskName PraxisFearGreedCollector
   Start-ScheduledTask -TaskName PraxisOhlcvDailyCollector
   Start-ScheduledTask -TaskName PraxisOhlcv4hCollector
   ```

3. **Smoke test the new code**:
   ```powershell
   .\.venv\Scripts\activate
   python -m servers.praxis_mcp.test_smoke
   ```
   Expected: "Smoke test PASSED" with "Registered tools: 12".

4. **Restart Claude Desktop** (Rule 32):
   - Run `kill_claude.bat` or equivalent (kills Claude.exe + any
     orphan Python from praxis venv)
   - Relaunch Claude Desktop from Start Menu
   - Open a NEW chat (existing chats keep stale tool snapshots)

5. **Verify in the new chat**: ask Claude to call `get_collector_health()`.
   Expected response:
   - Top-level `tables` dict shows 7 primary monitored tables, none
     `is_stale: true` (funding now under threshold; daily tasks
     freshly run)
   - Top-level `databases` dict has 3 keys: `crypto_data`,
     `live_collector`, `smart_money`
   - Each sidecar reports its monitored table with healthy staleness

6. **Commit**:
   ```powershell
   git add servers/praxis_mcp/server.py servers/praxis_mcp/tools/meta.py
   git add claude/retros/RETRO_health_sidecars.md
   git commit -m "Cycle 14: get_collector_health sidecar DB monitoring + funding_rates threshold fix"
   git push origin master
   ```
