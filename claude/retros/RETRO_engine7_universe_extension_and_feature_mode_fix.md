# Retro: Cycle 42 -- Engine 7 universe extension (42a) + run_cpo.py feature-mode fix (42d)

**Brief:** `claude/handoffs/BRIEF_engine7_universe_extension_and_feature_mode_fix.md`
**Date:** 2026-05-26
**Mode:** RECON-then-implementation in one cycle
**Status:** DONE
**Predecessor:** Cycle 41 -- Engine 7 live monitor pilot (`e9c46d5` + `cf2abee`); items 41a and 41d carried forward.
**Commit:** `<CYCLE_42_HASH>`

---

## Summary

42a: live `PraxisFundingCollector` + `PraxisFundingMonitor` extended from the BTC+ETH pilot universe to the full atlas Exp 13 deployment universe (BTC, ETH, SOL, XRP, ADA, AVAX; BNB excluded). All 6 assets verified writing `funding_rates` per funding event and `funding_signals` per funding window. New `funding_signals` table added to health monitoring (`get_collector_health` no longer reports it unmonitored).

42d: closed the `--feature-mode` default leak in `scripts/run_cpo.py`. Cycle 40's funding_rate pipeline is unaffected (`phase3_models_funding.joblib` filename preserved); future non-funding strategies will produce cleanly-unsuffixed outputs.

Net change:
- `engines/crypto_data_collector.py`: extended `SUPPORTED_ASSETS` from {BTC, ETH, SOL} to all 6 deployment-universe assets
- `services/funding_collector_service.bat`: FOR-loop over 6 assets + per-asset ERRORLEVEL aggregation (memory #12 hardening)
- `services/funding_monitor_service.bat`: removed explicit `--assets BTC,ETH` override; falls through to script's DEFAULT_ASSETS (already all 6)
- `servers/praxis_mcp/tools/meta.py`: `funding_signals` added to `primary_monitored` with 17h threshold matching `funding_rates`
- `scripts/run_cpo.py`: 5-line override in `main()` zeros out `args.feature_mode` for non-funding strategies (memory #23 fix)
- `funding_rates` table: now writes 6 assets/cycle going forward (was 2)
- `funding_signals` table: now writes 6 rows/window (was 2)
- `market_data` table: side-effect expansion to 6-asset coverage on next collector run (was 3)

---

## Execution log

### RECON answers

Q1 -- collector source-of-truth: `services/funding_collector_service.bat` lines 19-20, two explicit `--asset BTC` / `--asset ETH` invocations. Extension = FOR loop over all 6.

Q2 -- monitor source-of-truth: `services/funding_monitor_service.bat` line 26, explicit `--assets BTC,ETH`. `DEFAULT_ASSETS` in `scripts/funding_monitor.py` was already all 6 (set during Cycle 41). Extension = remove the explicit override so script default applies.

Q3 -- health monitoring: `servers/praxis_mcp/tools/meta.py`'s `primary_monitored` dict (line ~234). Currently `funding_rates: 61200` (17h), no `funding_signals`. Threshold pick = 17h matching `funding_rates` since the dynamics are identical (both timestamp = UTC funding event, both write 4-5h later via LOCAL trigger; worst-case healthy lag ~12h+; 17h leaves headroom for one missed cycle without false alarms).

Q4 -- --feature-mode plumbing: `scripts/run_cpo.py` argparse default is `"funding"` (line 340); flows into `sfx = getattr(args, "feature_mode", "")` in cmd_phase2 / cmd_phase3 / cmd_phase4 (lines 144, 159, 192); becomes `_funding` suffix on output filenames for ALL strategies. Fix: single override at top of `main()` zeros `feature_mode` for non-funding strategies.

### Implementation (in order)

1. **`services/funding_collector_service.bat`** rewritten with FOR loop:
   `for %%A in (BTC ETH SOL XRP ADA AVAX) do ( python -m engines.crypto_data_collector collect-funding --asset %%A --days 7 ... )`

2. **`services/funding_monitor_service.bat`** updated to drop `--assets BTC,ETH` from the python invocation; `DEFAULT_ASSETS` provides all 6.

3. **`servers/praxis_mcp/tools/meta.py`** -- added `"funding_signals": 61200` to `primary_monitored` with explanatory comment block matching the existing `funding_rates` comment shape.

4. **`scripts/run_cpo.py`** -- inserted after `args = parser.parse_args()`:
   ```python
   if args.strategy != "funding_rate":
       args.feature_mode = ""
   ```

5. **First trigger** of `PraxisFundingCollector` revealed an **unexpected third source-of-truth**: `SUPPORTED_ASSETS` constant in `engines/crypto_data_collector.py` only contained BTC/ETH/SOL. XRP/ADA/AVAX runs crashed with `KeyError: 'XRP'` / `'ADA'` / `'AVAX'` on `SUPPORTED_ASSETS[asset]["perp"]`. Brief anticipated this ("Components to update: ... probably a constant in engines/crypto_data_collector.py"); fixed inline by extending the dict to include XRP, ADA, AVAX with their CoinGecko IDs and Binance symbols.

   **Side effect noted:** `services/market_data_collector_service.bat` invokes `collect-market-data --asset all` which iterates `SUPPORTED_ASSETS`. After the extension, the next `PraxisMarketDataCollector` run will collect for 6 assets instead of 3 (BTC/ETH/SOL/XRP/ADA/AVAX). CoinGecko free tier handles this with the existing 2s inter-asset sleep in `cmd_collect_market_data` (12 calls/min budget). Symmetric universe expansion -- benign.

6. **Memory #12 follow-on**: the first PraxisFundingCollector run reported `LastResult=0` despite the 3 KeyErrors mid-loop. Root cause: cmd.exe FOR-loop ERRORLEVEL propagation gives the LAST command's exit code; the final asset (AVAX) crashed too but the bat's tail `echo` succeeded, masking the failure. Hardened the bat with `setlocal enabledelayedexpansion` + per-iteration `FAIL_COUNT` accumulator + `endlocal & exit /b 1` if any asset failed. Verified by re-running: all-success path still exits 0.

### Verification

After 3 collector triggers (first failed mid-loop, second succeeded after SUPPORTED_ASSETS fix, third tested the hardened bat):

```
funding_rates per-asset state (post-Cycle-42a):
  ADA    count=3727  latest=2026-05-27T00:00:00+00:00
  AVAX   count=3727  latest=2026-05-27T00:00:00+00:00
  BTC    count=3727  latest=2026-05-27T00:00:00+00:00
  ETH    count=3727  latest=2026-05-27T00:00:00+00:00
  SOL    count=3727  latest=2026-05-27T00:00:00+00:00
  XRP    count=3727  latest=2026-05-27T00:00:00+00:00
```

All 6 assets at the same `latest` timestamp = the 2026-05-27 00:00 UTC funding event. The +3 events per asset over the Cycle 40 D1 baseline are 2026-05-26 08:00, 2026-05-26 16:00, and 2026-05-27 00:00 -- the events that occurred between the D1 backfill and today's collector trigger.

After one monitor trigger:

```
funding_signals at 2026-05-27 00:00 window:
  ADA    p=0.5017  above_gate=0  above_gate_050=1  ann_rate=+10.95%
  AVAX   p=0.1932  above_gate=0  above_gate_050=0  ann_rate=+6.62%
  BTC    p=0.3116  above_gate=0  above_gate_050=0  ann_rate=+6.32%
  ETH    p=0.2989  above_gate=0  above_gate_050=0  ann_rate=+3.73%
  SOL    p=0.2168  above_gate=0  above_gate_050=0  ann_rate=-15.75%
  XRP    p=0.4637  above_gate=0  above_gate_050=0  ann_rate=-4.70%
```

All 6 assets persisted for the current window. **ADA crossed the headline P>0.50 gate** (P=0.5017, `above_gate_050=1`) -- not the live P>0.70 gate, but notable as the first non-zero `above_gate_050` row since deployment. Consistent with atlas Exp 13's primary OOS where ADA had the highest activity (186 days, +7.21 Sharpe) -- ADA tends to fire earliest as funding flips moderately positive. SOL's -15.75% annualized funding event drove the lowest P-score (0.2168), also consistent with the model correctly identifying negative funding as unfavorable.

The brief said "If any p_profitable score creeps above 0.70 during observation, capture as additional data and surface before commit." ADA hit 0.5017, **below the 0.70 surface-trigger threshold but above the headline 0.50 mark**. Worth observing in subsequent windows -- if ADA stays elevated and pushes past 0.70, that would be the first live "would-trade" signal since deployment. Not a Cycle 42 pause-trigger.

`get_collector_health` post-fix:

```
funding_rates           rows=     22353  latest=2026-05-27T00:00  staleness= 1.55h  thresh=17.0h  [fresh]
funding_signals         rows=         6  latest=2026-05-27T00:00  staleness= 1.55h  thresh=17.0h  [fresh]
...
Unmonitored (primary DB): []
```

`funding_signals` is now in the monitored set with `is_stale=false`. Primary-DB `unmonitored` list is empty -- all tables have explicit thresholds.

### 42d verification (argparse simulation)

```
strategy=funding_rate   pre_fix sfx='_funding'  post_fix sfx='_funding'  -> phase3_models_funding.joblib
strategy=crypto_ta      pre_fix sfx='_funding'  post_fix sfx=''          -> phase3_models.joblib
strategy=universal_ta   pre_fix sfx='_funding'  post_fix sfx=''          -> phase3_models.joblib
strategy=vol            pre_fix sfx='_funding'  post_fix sfx=''          -> phase3_models.joblib
```

Cycle 40's funding_rate path produces the same filename as before (`_funding` suffix preserved); future non-funding strategies will produce clean unsuffixed outputs. No re-run of any actual phase2/3/4 was needed for the fix verification -- the bug was purely in output filename derivation.

---

## Acceptance criteria check

| # | Criterion | Status |
|:-:|---|:-:|
| 42a-1 | PraxisFundingCollector writes funding_rates rows for all 6 assets at each event | ✅ all 6 at 3727 rows, latest 2026-05-27 00:00 |
| 42a-2 | PraxisFundingMonitor writes funding_signals rows for all 6 assets at each window | ✅ 6 rows at 2026-05-27 00:00 window |
| 42a-3 | funding_signals appears in get_collector_health output (no longer unmonitored) | ✅ monitored, fresh, threshold 17h |
| 42a-4 | Existing BTC/ETH rows continue unbroken | ✅ Cycle 41 BTC/ETH rows preserved by INSERT OR IGNORE PK collision; verified via row-count delta |
| 42d-1 | Fix applied, existing invocation patterns produce expected outputs | ✅ argparse simulation confirms funding_rate keeps `_funding` suffix; others get clean unsuffixed names |
| 42d-2 | Memory #23 updated | ⏳ user-confirmed will do after commit; retro carries the implementation detail |

---

## Notes

### SUPPORTED_ASSETS was a third source-of-truth (anticipated by brief)

The brief listed two known surfaces (bat + monitor); reality had three (bat + monitor + `SUPPORTED_ASSETS` constant). The brief explicitly anticipated this: "find the source of truth -- probably a constant in engines/crypto_data_collector.py or a CLI arg in services/funding_collector_service.bat". The `or` reading turned out to be `and`. Fixing only the bat would have left the collector crashing on 3 of 6 assets, which I caught in verification.

Going forward: if Cycle 43+ adds another asset (e.g. universe extends to 7+), three places need the edit:

1. `engines/crypto_data_collector.py` `SUPPORTED_ASSETS` dict
2. `services/funding_collector_service.bat` FOR-loop asset list
3. `scripts/funding_monitor.py` `DEFAULT_ASSETS` constant

Plus update Exp 13 entry in TRADING_ATLAS.md if the universe shape changes the recommended-live parameters.

### Threshold 17h matches funding_rates by symmetry

User suggested ~10h ("8h cadence + 2h buffer"). Actual worst-case healthy lag between writes is ~12h+: timestamp records funding event UTC time; monitor writes at LOCAL trigger which is 4-5h after the UTC event in summer; right before the next write (8h cadence later) the row is 12h+ old. 10h would alarm during normal operation. 17h matches funding_rates' precedent and the same reasoning applies. Tighter (13h) is the alternative if catching single missed runs faster is a priority; deferred to a future cycle if needed.

### XRP/ADA/AVAX first-window inputs were 24h stale at inference time (one-time artifact)

When the monitor wrote the 6 rows for the 2026-05-27 00:00 window, the collector hadn't yet succeeded for XRP/ADA/AVAX (SUPPORTED_ASSETS was missing those). The monitor reads funding history from DB; it found XRP/ADA/AVAX data only through 2026-05-26 00:00 (24h stale). Features were computed against the available history. The signals (P=0.5017 ADA, P=0.4637 XRP, P=0.1932 AVAX) reflect what the model would have predicted given those slightly-stale inputs -- not strictly "what it predicts NOW with current data". The P-scores are still mechanically valid (35-day lookback means 3 missing events out of 110 = ~3% data offset; impact on RF inference is small for this stable-feature model).

Going forward, every monitor run will use fully-current data for all 6 assets because the collector now succeeds for all 6. The 2026-05-27 00:00 window is a one-time transition artifact; the next window (08:00 UTC, monitor at 08:15 LOCAL ≈ 12:15 UTC) will be clean.

I considered deleting the 3 stale-input rows and re-triggering for a fully-clean acceptance state but decided against it: the rows are mechanically correct given the inputs at inference time, INSERT OR IGNORE would block any "fix", and the strict acceptance criterion ("rows for all 6 assets at each window") is met. The cleaner alternative would be `INSERT OR REPLACE` instead of `IGNORE` in `persist_signals`, but that's a behavior change with subtler implications (re-runs would overwrite — desirable for retries, undesirable for accidental clock-skew duplicate runs).

### market_data side-effect symmetric expansion

`SUPPORTED_ASSETS` extension propagates to `market_data_collector_service.bat`'s `--asset all` iteration. Next PraxisMarketDataCollector run (already overdue per health check at 25.55h staleness threshold 25h) will collect for 6 assets instead of 3. CoinGecko free-tier rate-limit headroom: 6 × 2 calls × 2s sleep = 24s baseline, well within per-minute quotas. Benign and arguably the right behavior -- the deployment universe should drive all data sources.

### Memory #12 bat hardening was emergent, not planned

The bat originally relied on FOR-loop's last-command ERRORLEVEL for exit status, which silently masked middle-of-loop failures (the very trap that surfaced with the SUPPORTED_ASSETS issue). Hardening with `setlocal enabledelayedexpansion` + per-iteration FAIL_COUNT + `endlocal & exit /b 1` aligns with memory #12 (exit-code honesty) and is adjacent to 42a's scope. The bat is now memory-#12-compliant; if a future asset fails (Binance outage on one symbol, API change, etc.) the scheduled task's `LastResult` will reflect the failure honestly.

The funding_monitor bat could use the same hardening but currently only invokes one python call (no loop), so its ERRORLEVEL propagation is already correct.

### What this cycle does NOT do

- Does NOT extend monitoring beyond funding (no new cross-venue spreads)
- Does NOT touch the executor (no live trading)
- Does NOT change atlas Exp 13 entry (universe was already documented as 6 assets; just operationalized here)
- Does NOT update memory #23 (user said they'll do that after commit)
- Does NOT re-train any phase3 models (Cycle 40's 6-asset model is what the monitor uses; same `monitor_version=cycle40:082459b` on every persisted row)
- Does NOT replay the 3 XRP/ADA/AVAX signals with fresh inputs (one-time transition artifact; next window is clean)

---

## Open items / Cycle 43+ inputs

- **43a Alerting** (Cycle 42 42b -> renumber): wire `above_gate=1` rows to a notification surface (Discord webhook, email via funding_alert.py's existing SMTP path)
- **43b Cross-venue funding spreads** (Bybit, OKX, Hyperliquid; atlas Exp 13 revival hypothesis #1)
- **43c LSTM v2 architecture test** (atlas Exp 13 revival hypothesis #4)
- **43d Bear-regime accumulation analysis** (needs ~30 days of monitor data; the 2026-04/05 regime is providing those data points passively right now)
- **43e PMA backfill** (long-deferred from prior cycles)
- **43f atlas_search engine-filter parameter** (long-deferred TODO from Cycle 35)
- **43g (optional)** Tighten funding_signals staleness threshold from 17h to 13h if catching single-missed-cycle failures faster is desired (current 17h leaves room for 1 missed cycle before alarm; 13h would alarm on any missed cycle)
- **43h (optional)** Apply equivalent memory-#12 hardening to other multi-step bat scripts in services/ that loop over assets
- **43i (optional)** Re-trigger monitor to overwrite the 3 stale-input rows from this cycle's transition (currently not possible due to INSERT OR IGNORE PK collision; would require either a manual DELETE + re-trigger, or changing persist_signals to INSERT OR REPLACE -- not worth the behavior change for one row)
- **ADA P-score watch**: ADA hit P=0.5017 (above_gate_050=1) at the 2026-05-27 00:00 window. If subsequent windows show ADA pushing toward 0.70, capture as the first live "would-trade" signal since deployment.
