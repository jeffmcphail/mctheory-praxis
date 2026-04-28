# Retro: Praxis Intrabar Data Extension (v4)

**Date:** 2026-04-22
**Status:** COMPLETE — Option C UNBLOCKED
**Brief:** `claude/handoffs/BRIEF_praxis_intrabar_data_extension.md`
**Companion to:** `RETRO_praxis_intrabar_confluence.md` (v3) — does not overwrite it.

---

## TL;DR

The probe refuted the collector's "Binance retains ~30 days of 1-min data" assumption. **Binance's public `fetch_ohlcv` endpoint serves at least 730 days of 1-min and 5-min klines for both BTC/USDT and ETH/USDT** — every probe at 60/90/180/365/730 days returned the requested `since` timestamp exactly, with a full 1000-candle batch. Collector cap raised from 30 → 180 days, BTC + ETH backfilled. `ohlcv_1m` grew from 100,129 rows (34.8 days) to **518,405 rows (180.0 days)** — a 5.2× expansion per asset in 5 minutes of wall-clock. Option C is no longer a heavy-lift project; the extended dataset is live and ready for feature rebuild + retrain.

---

## 1. Step 1 — probe results

`scripts/test_binance_1m_history.py` with `enableRateLimit=True`. Ten probes per symbol (2 timeframes × 5 offsets). Full log: `models/intrabar/probe_binance_history.log`.

| Symbol | Timeframe | days_back | Status | Oldest returned | n |
|---|---|---:|---|---|---:|
| BTC/USDT | 1m | 60 | OK | 2026-02-21 19:17 | 1000 |
| BTC/USDT | 1m | 90 | OK | 2026-01-22 19:17 | 1000 |
| BTC/USDT | 1m | 180 | OK | 2025-10-24 19:17 | 1000 |
| BTC/USDT | 1m | 365 | OK | 2025-04-22 19:17 | 1000 |
| BTC/USDT | 1m | **730** | **OK** | **2024-04-22 19:17** | 1000 |
| BTC/USDT | 5m | 60–730 | OK (all) | matches requested `since` | 1000 each |
| ETH/USDT | 1m | 60–730 | OK (all) | matches requested `since` | 1000 each |
| ETH/USDT | 5m | 60–730 | OK (all) | matches requested `since` | 1000 each |

**Zero `CLAMPED` responses, zero `EMPTY` responses, zero errors.** For every probe the oldest returned timestamp was within 1 minute of the requested `since` (the 1-minute drift is just bar-boundary alignment — requested `19:16:31`, returned bar starting `19:17:00`). The probe's `CLAMPED` detector (drift > 86,400s) never fired.

**Bonus findings (open questions from brief):**
- Q1 (>365 days?): **Yes — 730 days served cleanly.** Didn't probe further since 180 was sufficient for the decision; 730+ is available as a ceiling reference.
- Q2 (5-min availability?): **Yes — identical coverage to 1-min.** This means we could optionally bypass the aggregation step in `load_intrabar_data()` entirely and pull 5m klines directly. Not done in this pass (aggregation still works and `ohlcv_5m` isn't a table yet), but worth noting as an architectural option.

---

## 2. Step 2 — collector patched + backfilled

Since Step 1's verdict was unambiguous YES, executed the Step 2-YES branch.

**Code change** — `engines/crypto_data_collector.py`:

- Line 271: `min(days, 30)` → `min(days, 180)`.
- Lines 254–261: replaced stale docstring ("Binance retains ~30 days of 1-min data. Run daily to accumulate.") with empirically-grounded note citing the probe script + this retro. Removed the two misleading runtime prints about 30-day retention.
- Passed AST + ASCII checks (22,356 bytes, all ≤ 127).

**Backfill** — run sequentially to respect Binance rate limits:

```
python -m engines.crypto_data_collector collect-ohlcv-1m --asset BTC --days 180
python -m engines.crypto_data_collector collect-ohlcv-1m --asset ETH --days 180
python -m engines.crypto_data_collector status
```

Full log: `models/intrabar/backfill_1m_180d.log`. Wall-clock: **5 min total** (2.5 min per asset).

**Pre-backfill row counts:**
| Asset | Rows | Span |
|---|---:|---|
| BTC | 50,065 | 2026-03-18 22:09 → 2026-04-22 16:33 (34.8 d) |
| ETH | 50,064 | 2026-03-18 22:10 → 2026-04-22 16:33 (34.8 d) |

**Post-backfill row counts:**
| Asset | Rows | Span | Growth |
|---|---:|---|---:|
| BTC | **259,202** | 2025-10-24 19:18 → 2026-04-22 19:19 (180.0 d) | **5.18×** |
| ETH | **259,203** | 2025-10-24 19:20 → 2026-04-22 19:22 (180.0 d) | **5.18×** |

Collector fetched 259,202 and 259,203 candles respectively (vs expected 180 × 1440 = 259,200 — within 3 candles of perfect; the tiny excess reflects fractional-day window drift at the endpoints). `INSERT OR REPLACE` dedup behaved as expected: no duplicate-key errors, existing rows in the 34.8-day overlap window got overwritten idempotently.

`status` confirms the extended coverage: `ohlcv_1m  518,405 rows | 2025-10-24 19:18:00 to 2026-04-22 19:22:00 | BTC, ETH`.

**Scheduled task `PraxisCrypto1mCollector` untouched** per brief — still runs with whatever `--days` value its definition uses (likely <180), so its routine behavior is unchanged.

---

## 3. Acceptance criteria — all green

- [x] Probe script runs without errors and prints probe results for both assets at 60/90/180/365 days *(plus 730 bonus)*
- [x] AST + ASCII check pass on new/modified files (probe script 3,923 bytes; collector 22,356 bytes)
- [x] Retro states whether Binance serves ≥90 days of 1-min data (**YES — ≥730 days actually**)
- [x] YES branch: collector modified, BTC + ETH backfilled, new row counts reported (**259K per asset**)
- [x] Open questions 1 & 2 both answered in the bonus probes

---

## 4. Recommendation for Chat

**Option C (dataset extension) is no longer blocked and no longer expensive.** The extended dataset is live. Next steps, ordered cheapest to most expensive:

1. **Rebuild features** on `ohlcv_1m` → 5-min bars. Expected: ~52,000 5-min bars vs v3's 9,796 (~5.3× more). Features tensor regenerated via `python -m engines.intrabar_predictor build-features --asset BTC`. Stale cleanup at the top of `cmd_build_features` will remove the v3 artifacts automatically. Est: <30 seconds given v3's 5-second run at 34.8 days.
2. **Retrain LSTM + XGBoost** on the extended dataset. At the same 15s-per-epoch rate from v3, a 5.3× dataset will take ~80s per epoch. Early stopping fired at epoch 21 in v3 — budget accordingly (~25 min total if similar convergence, up to 90 min for a full 150-epoch budget). Fits within tool timeout with `run_in_background`.
3. **Rerun the v3 diagnostic suite** (the 6 backtests — fee sensitivity + regime split) against the extended test window. This is what actually answers the Case 3 vs Case 4 question from v3.

**What to look for after retrain + rediagnostic:**
- **If direction bias disappears** (LONG/SHORT split becomes reasonably balanced, e.g., > 20% in the minority direction): Case 4 was the correct read. Then evaluate P&L — if still net-negative at maker fees across regimes, Case 3 becomes terminal and we pivot to A2 (momentum flip) or A4 (XGB-only 5-bar). If one regime flips positive, Case 2 resolves and we tune with a Hurst gate (A1).
- **If direction bias persists** (e.g., still >90% one direction): that's a genuine structural artifact of the dual-horizon confluence filter, not window bias — Case 3 terminal without needing further data. Pivot to A2/A4.

Either way, the 6-month dataset will span multiple market regimes (Oct 2025 → Apr 2026 includes both trend phases and at least one consolidation), so the ambiguity from v3 should resolve cleanly.

---

## 5. Files changed / created

- **NEW:** `scripts/test_binance_1m_history.py` — probe script (3,923 bytes). Reusable; can be rerun to verify retention in future.
- **NEW:** `models/intrabar/probe_binance_history.log` — full probe output.
- **NEW:** `models/intrabar/backfill_1m_180d.log` — full backfill output (BTC + ETH + status).
- **NEW:** `claude/retros/RETRO_praxis_intrabar_data_extension.md` (this file).
- **MODIFIED:** `engines/crypto_data_collector.py` — docstring block at lines 253–261, `min(days, 30)` → `min(days, 180)` at line 271. AST/ASCII clean.
- **DB mutation:** `data/crypto_data.db` `ohlcv_1m` grew from 100,129 rows → 518,405 rows. `INSERT OR REPLACE` dedup'd the overlap window. No other tables touched.

**NOT touched** per brief: `engines/intrabar_predictor.py`, model artifacts in `models/intrabar/`, the `PraxisCrypto1mCollector` scheduled task, no git commits. v3's uncommitted code is still in the working tree alongside this round's changes.

---

## 6. State at session end

- DB is live with 180-day `ohlcv_1m` coverage for BTC + ETH.
- Collector code supports `--days` up to 180 without further modification; can be bumped higher if Chat wants to probe 365+ days (we have evidence the API supports 730).
- No Python processes running. No training in progress.
- Blocker cleared. Chat call on whether to proceed with feature rebuild + retrain + rediagnostic (recommended path) or something else.
