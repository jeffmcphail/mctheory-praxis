# Cycle 43 -- Engine 7 alerting (43a) + bat hardening sweep (43h)

**Predecessor:** Cycle 42 -- universe extension + run_cpo feature-mode
fix (commits `dee17f6` + `84a1dd0`). Cycle 43 picks up items 42b
(alerting) and 43h (bat hardening) from that cycle's Open items list.

**Mode:** RECON-then-implementation in one cycle, same shape as Cycles
41 and 42.

## 43a -- Alerting on above_gate=1

Design decisions (pre-approved by user in brief):

1. **Trigger:** `above_gate=1` only (P>0.70 live gate). NOT above_gate_050.
   The live gate is the deployment-relevant threshold; the 0.50 column
   is for retrospective analysis only.

2. **Channel:** Teams webhook via Power Automate. Per project memory,
   that's the established channel; greenfield code-wise (no existing
   helper found; writing inline).

3. **Idempotency:** separate `funding_alerts` table with PK
   `(asset, timestamp)` matching funding_signals. Row existence =
   "alert was successfully POSTed". Failed POSTs (network blip,
   webhook URL invalid) do not insert -> next monitor run retries.

4. **Alert payload:** asset, datetime, p_profitable, ann_rate,
   basis_pct, best_config_id, hold_days, min_funding_ann,
   expected_return. Simple `{"text": "..."}` JSON shape; Power
   Automate flow side renders into Teams card.

5. **Webhook URL via .env** (memory #4 -- never hardcoded).
   `TEAMS_WEBHOOK_URL` placeholder added to `.env.example` with
   Power Automate setup instructions.

Implementation:
- `engines/crypto_data_collector.py`: new `funding_alerts` table
  added to `init_db()`, 7 cols, Rule-35-conforming.
- `scripts/funding_monitor.py`: `post_teams_alert()` helper and
  `process_alerts()` orchestrator; new `--alert` CLI flag (off by
  default; on in scheduled task).
- `services/funding_monitor_service.bat`: invocation gets `--alert`.
- `.env.example`: TEAMS_WEBHOOK_URL stub + Power Automate setup
  walkthrough.

Acceptance for 43a:
- Webhook URL loaded from .env (not hardcoded) -- code verified
- Live test: synthetic above_gate=1 -> exactly one alert to webhook
  -- DEFERRED pending URL population in user .env
- Idempotency re-run -- DEFERRED with the live test
- No alerts during normal sit-out behavior -- verified at gate=0.70
  (all P<0.70, no alerts; no-op skip when URL unset)
- funding_alerts state queryable post-run -- verified (table exists,
  schema validated)

## 43h -- Bat hardening sweep

Inventory of multi-step service bats (single-invocation bats skipped
since python's exit code propagates naturally):

| Bat | Steps | Trap? |
|---|---:|:---:|
| crypto_1m_collector_service.bat | 2 (BTC, ETH) | yes |
| ohlcv_4h_collector_service.bat | 2 (BTC, ETH) | yes |
| ohlcv_daily_collector_service.bat | 2 (BTC, ETH) | yes |
| smart_money_service.bat | 2 (discover, snapshot) | yes |

All 4 hardened with the Cycle 42 funding_collector pattern:
`setlocal enabledelayedexpansion` + per-step `if errorlevel 1
set /a FAIL_COUNT+=1` + `if !FAIL_COUNT! gtr 0 ... endlocal &
exit /b 1`.

Acceptance for 43h:
- Inventory done
- 4 bats hardened
- 1+ verification trigger per modified bat: all 4 returned
  LastResult=0 on the all-success path

## Out of scope (Cycle 44+ candidates)

- Cross-venue funding spreads (43b deferred from earlier cycles)
- LSTM v2 architecture test
- Real-money executor integration
- Bear-regime accumulation analysis (passively accumulating)
- atlas_search engine-filter parameter
- PMA backfill
- Tighter funding_signals threshold 17h -> 13h
