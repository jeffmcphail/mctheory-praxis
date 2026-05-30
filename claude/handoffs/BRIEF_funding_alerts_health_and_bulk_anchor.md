# Cycle 47 -- funding_alerts health monitoring + 44h-bulk anchoring sweep

**Predecessor:** Cycle 46 (`d441c92` + `d38def0`). Two follow-on
items from prior cycle retros:

- **Sub-item 1:** add `funding_alerts` to
  `servers/praxis_mcp/tools/meta.py` `primary_monitored` dict
  (the Cycle 43a alert ledger was overlooked when funding_signals
  was added in Cycle 42a).
- **Sub-item 2:** apply the Cycle 46 `Path(__file__).resolve().parent.parent`
  anchoring pattern to the remaining 14 vulnerable engine constants
  identified in Cycle 46's retro ("44h-bulk").

**Mode:** RECON-then-implementation, one cycle. Mechanical follow-on.

## 44h-bulk targets (14 engines, 17 constants)

| File | Constants |
|---|---|
| `engines/live_collector.py` | `DB_PATH` (line 40) + 2 inline `spike_scanner.db` refs at 210, 539 -- refactored to a new module-level `SPIKE_DB_PATH` |
| `engines/smart_money.py` | `DB_PATH` |
| `engines/smart_money_alerts.py` | `DB_PATH` + `ALERTS_DB_PATH` |
| `engines/spike_scanner.py` | `DB_PATH` |
| `engines/spike_features.py` | `DB_PATH` |
| `engines/event_classifier.py` | `DB_PATH` |
| `engines/mev_executor.py` | `LIVE_DB` + `EXECUTOR_DB` (adjacent same-trap; picked up inline) |
| `engines/actuarial.py` | `DB_PATH` |
| `engines/ai_ensemble.py` | `DB_PATH` |
| `engines/crypto_predictor.py` | `DB_PATH` |
| `engines/flash_scanner.py` | `DB_PATH` |
| `engines/mev_scanner.py` | `DB_PATH` |
| `engines/negrisk_arb.py` | `DB_PATH` |
| `engines/lstm_predictor.py` | `DATA_DB` |

Pattern (identical to Cycle 46 funding-chain):

    Path(__file__).resolve().parent.parent / "data" / "<dbname>"

## Sub-item 1: funding_alerts threshold

17h (61200 s), matching funding_signals' precedent. Same dynamics:
PK on `(asset, timestamp)` where timestamp is the UTC funding-window
time. Empty-table handling in `_collect_db_health` already surfaces
`row_count=0, error="empty table"` rather than `is_stale=True`, so
sparse population (current state: 0 rows) doesn't trigger false
alarms.

## Verification

- A. Syntax + import check on all 14 engines (17 constants total)
- B. Path resolution from non-root cwd (services/) -- identical absolute paths
- C. Scheduled task smoke for engines with active tasks:
  - PraxisSmartMoney -> trigger; verify LastResult=0 + fresh snapshot row
  - PraxisLiveCollector -> long-lived process; verify it's still running and price_snapshots is still being written
- D. Dormant engines documented (12 engines without active scheduled tasks): Path-anchored but not live-verified; subsequent natural use will surface any regression
- 8. funding_alerts no longer in `unmonitored` list

## Out of scope

- `engines/_paths.py` helper module refactor (44h-refactor, queued
  for Cycle 48)
- `models/` paths (`lstm_predictor.MODEL_DIR`, `crypto_predictor.MODEL_DIR`)
  and `OUTPUT_DIR` in `spike_features.py` -- same CWD-vulnerability but
  different surface (not DB paths); log as candidate for a future
  audit cycle
- Sidecar-table monitoring audit (collection_log, spike_alerts,
  tracked_markets, convergence_signals, position_changes,
  tracked_wallets) -- separate concern; log as candidate "44q
  sidecar-table monitoring audit"
- Any logic changes in the 14 engines beyond DB path anchoring
- Cross-venue, LSTM, executor, regime analysis (Cycle 49+)
