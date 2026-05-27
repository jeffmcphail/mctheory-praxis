# Retro: Cycle 43 -- Engine 7 alerting (43a) + bat hardening sweep (43h)

**Brief:** `claude/handoffs/BRIEF_engine7_alerting_and_bat_hardening.md`
**Date:** 2026-05-27
**Mode:** RECON-then-implementation in one cycle (no Cycle 43b follow-up needed)
**Status:** DONE (with 1 deferred acceptance item -- see below)
**Predecessor:** Cycle 42 -- universe extension + feature-mode fix (`dee17f6` + `84a1dd0`); items 42b (alerting) and 43h (bat hardening) carried forward.
**Commit:** `04b24b6`

---

## Summary

43a: Live alerting wired into `PraxisFundingMonitor`. New `funding_alerts` table in `data/crypto_data.db` (Rule-35-conforming compound PK on `(asset, timestamp)`). `scripts/funding_monitor.py` extended with `--alert` flag + `post_teams_alert()` helper + `process_alerts()` orchestrator. `services/funding_monitor_service.bat` updated to pass `--alert`. `.env.example` gains a `TEAMS_WEBHOOK_URL` placeholder with full Power Automate setup instructions.

43h: All 4 multi-step service bats (`crypto_1m_collector`, `ohlcv_4h_collector`, `ohlcv_daily_collector`, `smart_money`) hardened with the Cycle 42 funding_collector pattern. Each one verified by Start-ScheduledTask trigger returning LastResult=0.

**Deferred:** the synthetic above_gate=1 live test + idempotency re-run from brief acceptance #2 and #3. Both require an actual Power Automate webhook URL in `.env`, which is not currently populated. Deferred to "first natural firing or whenever the user populates the URL"; rationale below.

Net change:
- `engines/crypto_data_collector.py`: new `funding_alerts` table (7 cols) added to `init_db()` block
- `scripts/funding_monitor.py`: ~95 lines added (`post_teams_alert`, `process_alerts`, `--alert` flag, run_once hook); 0 lines removed
- `services/funding_monitor_service.bat`: 1-line change (`--alert` added to python invocation)
- `services/crypto_1m_collector_service.bat`, `ohlcv_4h_collector_service.bat`, `ohlcv_daily_collector_service.bat`, `smart_money_service.bat`: all rewritten with `setlocal enabledelayedexpansion` + FAIL_COUNT pattern
- `.env.example`: new TEAMS_WEBHOOK_URL section with Power Automate flow walkthrough

---

## Execution log

### RECON answers (3 of 3 questions resolved without pause)

Q1 -- webhook plumbing: **No existing Teams webhook code found** in
`engines/`, `scripts/`, or `.env.example`. `engines/smart_money_alerts.py`
is named "alerts" but only writes DB rows + stdout. The brief's "Per
project memory, that's the established channel" reflects an external
intent (Power Automate flow exists Teams-side) rather than wired code.
Brief pre-approved the fallback: "otherwise write a small helper inline."
Done. Surfaced explicitly to user before implementation.

Q2 -- idempotency mechanism: separate `funding_alerts` table per user's
lean. Cleanest Rule 35 fit; matches smart_money's "signals + alerts as
separate tables" pattern; funding_signals stays read-only from the
monitoring perspective. Row's existence = "alert was successfully POSTed";
failed POSTs do not insert (next monitor cycle retries naturally).

Q3 -- multi-asset bat inventory: 13 bats in `services/`, 4 with multi-step
sequential python invocations (the trap surface):

| Bat | Steps | Notes |
|---|---:|---|
| crypto_1m_collector_service.bat | 2 (BTC, ETH) | original pattern: trap |
| ohlcv_4h_collector_service.bat | 2 (BTC, ETH) | trap |
| ohlcv_daily_collector_service.bat | 2 (BTC, ETH) | trap |
| smart_money_service.bat | 2 (discover, snapshot) | trap |

The other 9 are single-invocation (python's exit code propagates
naturally) and don't need hardening:
fear_greed, info_bars, live_collector, market_data, onchain,
order_book, trades, funding_monitor. funding_collector was already
hardened in Cycle 42.

### Implementation

1. **Schema extension** -- added `funding_alerts` table to
   `engines/crypto_data_collector.py:init_db()` block. 7 columns:
   `asset, timestamp, datetime, alerted_at, p_profitable,
   gate_threshold, monitor_version`. Compound PK on `(asset, timestamp)`
   matching funding_signals. Triggered `init_db()` to create. 0 rows
   on initial state.

2. **`.env.example` placeholder** -- new `TEAMS WEBHOOK ALERTING (Cycle
   43 -- funding-carry live alerts)` section between EMAIL ALERTING and
   PYTHON RUNTIME TUNING. Includes step-by-step Power Automate flow
   setup walkthrough (request trigger -> Teams post action; body schema
   `{"text": "string"}`). Empty value documented as "--alert becomes
   a no-op" for graceful degradation.

3. **`scripts/funding_monitor.py` extensions:**

   - `post_teams_alert(alert_signal, monitor_version, webhook_url) ->
     (success_bool, response_excerpt)`: urllib.request POST with
     `{"text": "..."}` body, 10s timeout, catches network errors.
   - `process_alerts(signals, db_path, gate, monitor_version) -> int`:
     loops over above_gate=1 signals, dedups against funding_alerts
     PK, POSTs, inserts on success. Returns count of new alerts.
     Graceful no-op when `TEAMS_WEBHOOK_URL` env var is unset.
   - `--alert` CLI flag added to argparse (off by default).
   - Hook in `run_once()` after `persist_signals`: calls
     `process_alerts` when `args.alert` is set; prints alert count.

4. **Service bat update** -- `services/funding_monitor_service.bat`
   python invocation now passes `--persist --alert`.

5. **Bat hardening sweep** -- 4 bats rewritten with the standard
   pattern (mirrored from Cycle 42 funding_collector):
   ```bat
   setlocal enabledelayedexpansion
   ...
   set FAIL_COUNT=0
   python ... ; if errorlevel 1 ( set /a FAIL_COUNT+=1 ; echo FAILED... )
   python ... ; if errorlevel 1 ( ... )
   if !FAIL_COUNT! gtr 0 (
       echo Completed with !FAIL_COUNT! failure(s)...
       endlocal & exit /b 1
   )
   echo Complete.
   endlocal
   ```

### Verification

**No-op alert smoke test** (URL unset, gate=0.70):
```
WARN: TEAMS_WEBHOOK_URL not set in .env; --alert is a no-op (signals
still persisted to funding_signals)
Alerts:    0 new row(s) to funding_alerts
```
funding_alerts row count after: 0. ✓

**Bat hardening verification** -- triggered all 4 tasks; final state:

| Task | State | LastRun | LastResult |
|---|---|---|---:|
| PraxisCrypto1mCollector | Ready | 2026-05-27 01:00:25 | 0 |
| PraxisOhlcv4hCollector | Ready | 2026-05-27 01:00:25 | 0 |
| PraxisOhlcvDailyCollector | Ready | 2026-05-27 01:00:25 | 0 |
| PraxisSmartMoney | Ready | 2026-05-27 01:00:25 | 0 (~8 min runtime) |

All 4 succeed-path runs exit 0. ✓

**Deferred (brief acceptance #2, #3):** synthetic above_gate=1 test
(`--gate 0.50 --alert` to fire ADA P=0.5017) + idempotency re-run.
Both require `TEAMS_WEBHOOK_URL` populated in `.env`; user opted to
defer to first natural firing rather than gate the commit.

### CWD-corruption side-trip (worth noting for future cycles)

During RECON, a Bash `cd "C:/.../services" && for bat in *.bat...`
inventory call left the persisted CWD pointing at `services/`. The
subsequent `python -c "from engines.crypto_data_collector import
init_db; conn = init_db()"` call ran with that CWD, and since
`init_db()` uses a relative `Path("data/crypto_data.db")`, the
funding_alerts table got created in a phantom DB at
`services/data/crypto_data.db` rather than the real
`data/crypto_data.db`. Caught when the verification query failed
with `no such table: funding_alerts`.

Cleanup: phantom `services/data/` and `services/logs/` directories
removed; CWD reset to project root via `Set-Location`; init_db re-run
in correct CWD. funding_signals' Cycle 42 rows were preserved
(they're in the real DB; the phantom was a separate file).

**Latent improvement for the future:** consider making
`crypto_data_collector.DB_PATH` resolve relative to the module file's
location rather than process CWD. One-line change:
`DB_PATH = Path(__file__).resolve().parent.parent / "data" /
"crypto_data.db"`. Would prevent this whole class of trap. Not in
this cycle's scope; logging here.

---

## Acceptance criteria check

| # | 43a | Status |
|:-:|---|:-:|
| 1 | Webhook URL loaded from .env (not hardcoded) | ✅ verified via `os.getenv("TEAMS_WEBHOOK_URL", "").strip()` in process_alerts |
| 2 | Live test: synthetic above_gate=1 -> exactly one alert | ⏳ **DEFERRED** -- requires TEAMS_WEBHOOK_URL populated in user .env; user opted to defer to first natural firing |
| 3 | Idempotency verified: re-run doesn't re-fire | ⏳ **DEFERRED** with #2 -- DB PK enforces it unconditionally; live verification deferred |
| 4 | No alerts during sit-out (P<0.70 currently) | ✅ no-op smoke test at gate=0.70 verified 0 alerts |
| 5 | funding_alerts state queryable post-run | ✅ table exists, 7 cols, queryable, 0 rows |

| # | 43h | Status |
|:-:|---|:-:|
| 1 | Inventory of multi-asset/multi-step bats | ✅ 4 identified + documented above |
| 2 | Each safe (single-call) or hardened | ✅ 9 single-call skipped; 4 hardened |
| 3 | 1+ verification trigger per modified bat, all-success exit 0 | ✅ 4 of 4 LastResult=0 |

---

## Why the live-test deferral is operationally low-risk

1. **No-op path verified.** When `TEAMS_WEBHOOK_URL` is unset (current
   state), `process_alerts()` logs the warning and returns 0. Signals
   still persist to funding_signals; the monitor's primary job is
   unaffected. Until the URL is set, the system behaves exactly like
   pre-Cycle-43.

2. **DB-level PK idempotency is unconditional.** The PK constraint on
   `funding_alerts (asset, timestamp)` enforces "at most one alert per
   (asset, funding-window)" at the schema layer. Even if a future bug
   in `process_alerts` tried to insert duplicates, sqlite would
   reject. The live test would just confirm the application-layer
   dedup is in the right shape; the data-integrity guarantee is
   structural.

3. **First natural firing IS the live test.** At some future
   funding window when an asset's P-score crosses 0.70, the monitor
   will attempt to POST. Three outcomes:
   - Success: alert lands in Teams + funding_alerts gets a row.
     Cycle complete-in-effect.
   - 4xx/5xx from webhook: log captures the response; funding_alerts
     doesn't insert; next cycle retries. Real failure signal for
     debugging.
   - Network blip / timeout: same as above; retries naturally.

   In each case, the failure mode is observable and recoverable.
   The risk of shipping un-live-tested is bounded by these three
   cases, none of which are catastrophic.

4. **Current regime makes a natural firing unlikely soon.** The bear-
   side BTC + flat-to-mildly-positive ETH + below-threshold ann_rates
   on most assets mean P-scores hover well below 0.70 (Cycle 42's
   highest observation was ADA P=0.5017). The "wait for natural firing"
   horizon could be weeks. That's acceptable for a deferred test.

5. **The user can synthesize at any time** by running
   `python scripts\funding_monitor.py --assets ADA --gate 0.50 --persist
   --alert` once `TEAMS_WEBHOOK_URL` is populated. The synthetic test
   path is intact in the code.

---

## Notes

### CWD-related trap caught + recovery

See "CWD-corruption side-trip" in Execution log above. Recommend
considering the `Path(__file__).resolve().parent.parent` fix for
`DB_PATH` in a future cycle to make all `init_db()` callers
CWD-independent.

### Bat hardening pattern is now standardized

After Cycle 42 (funding_collector) + Cycle 43h (4 more), the 5
multi-step bats in the codebase all share the same hardening shape:
`setlocal enabledelayedexpansion` + per-step `FAIL_COUNT` accumulator
+ `endlocal & exit /b 1` on aggregate failure. If a future cycle adds
a new multi-step bat, this is the template.

The single-invocation bats (9 of 13 total) don't need this pattern
because python's exit code propagates to %ERRORLEVEL% directly; the
bat exits with python's status without any cmd.exe FOR-loop
ambiguity.

### Power Automate setup left to user (not committed)

`.env.example` documents the setup steps verbosely (request trigger
+ JSON schema + Teams post action) but the actual flow creation is a
user-side UI action in Power Automate. Once the URL is in `.env`,
the next monitor cycle that has an above_gate=1 firing will produce
a live Teams alert.

If the user opts not to use Power Automate (e.g., wants Discord or
Slack instead), the `{"text": "..."}` JSON shape is generic enough
that any webhook accepting that schema works. No code change needed
for a different webhook backend.

### What this cycle does NOT do

- Does NOT live-test the webhook (deferred -- see above)
- Does NOT touch the executor (no live trading)
- Does NOT touch the funding-rate strategy code or model
- Does NOT change funding_signals semantics; alerts are a downstream
  read-only consumer
- Does NOT add alerting for above_gate_050=1 events (intentional --
  the 0.50 column is for retrospective analysis, not deployment-
  relevant alerting)

---

## Open items / Cycle 44+ inputs

- **43a-live-test (NEW)**: populate TEAMS_WEBHOOK_URL in .env, then
  wait for or synthesize first above_gate=1 event, verify alert
  delivery + idempotency. Defers indefinitely; first natural firing
  serves as the live test. User-driven.
- **44a Cross-venue funding spreads** (Bybit, OKX, Hyperliquid;
  atlas Exp 13 revival hypothesis #1)
- **44b LSTM v2** test on the validated feature set
- **44c Real-money executor integration** (was 43e; depends on
  43a-live-test landing first so we have confidence in the alerting
  surface before money moves)
- **44d Bear-regime accumulation analysis** (~30 days of funding_signals
  data accumulating passively now; check back after sufficient sample)
- **44e PMA backfill** (long-deferred)
- **44f atlas_search engine-filter parameter** (long-deferred from
  Cycle 35)
- **44g (optional)** Tighten funding_signals threshold 17h -> 13h
- **44h (optional)** CWD-independence fix for `crypto_data_collector.DB_PATH`
  (one-line change to use `Path(__file__).resolve().parent.parent`;
  would prevent the CWD-corruption trap that surfaced this cycle)
- **44i (optional)** Per-bat exit-code tests with INTENTIONAL failures
  to verify the FAIL_COUNT logic actually catches them (this cycle
  verified the all-success path; a future cycle could synthesize a
  failure -- e.g., invalid asset arg -- to confirm the hardening fires
  correctly on the unhappy path too)
