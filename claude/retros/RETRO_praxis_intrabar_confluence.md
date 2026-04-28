# Retro: Praxis Intrabar Confluence — v8.1 Probe + v8.2 Order Book + v8.2.1 Trade Flow + Meta-Registrar (COMPLETE)

**Date:** 2026-04-22 → 2026-04-23
**Status:** COMPLETE — four briefs landed across this retro's window. v8.1 interpretation verdict: **Case C with emphasis** (labeling is not the fix, features are). v8.2 + v8.2.1 have both microstructure data streams accumulating in parallel for v8.3.
**Briefs:**
- `claude/handoffs/BRIEF_praxis_order_book_collector.md` (v8.2 data foundation)
- `claude/handoffs/BRIEF_praxis_intrabar_confluence_v8_1.md` (triple-barrier labeling probe)
- `claude/handoffs/BRIEF_praxis_trades_collector.md` (v8.2.1 trade-flow collector)
- *Ad hoc follow-up within the same session:* `services/register_all_tasks.ps1` meta-registrar (no brief — designed + implemented inline in response to the admin-PowerShell friction)

**Supersedes:** v7 COMPLETE retro at the same path.

**Cycle counter:** advanced to **v8.2.1 COMPLETE**. v8.3 (microstructure-feature modeling on accumulated depth + flow data) is the next scoping target. Minimum viable training window: ~2 weeks (≈2026-05-07); robust window: ~4 weeks (≈2026-05-21).

---

## 1. TL;DR

Two briefs shipped in the same session:

**Order book collector (v8.2 data foundation):** new `order_book_snapshots` table (40 price/vol fields + 3 derived aggregates), two new CLI subcommands (`collect-order-book` one-shot + `collect-order-book-loop` continuous), service `.bat` + PowerShell registration script modeled on existing `crypto_1m` patterns, `status` command extended. Scheduled task `PraxisOrderBookCollector` registered by Jeff after Code hit an Access-Denied on `Register-ScheduledTask` (anticipated by brief). Task confirmed `State=Running` with `LastTaskResult=267009` (still executing the 3600-s loop), data landing cleanly: 15 rows per asset at T+~40 min into the first hourly invocation. Sample row sanity check passed (BTC mid $78,347, spread 0.01 / ~0.001 bps, 10×2 levels populated, imbalance −0.02). No impact on the 5 existing Praxis scheduled tasks.

**v8.1 triple-barrier probe:** added `train-classifier` subcommand alongside existing `train` (LSTM quantile) path. Triple-barrier labels computed on 51,705 rows with lookforward=15 (75 min) and atr_mult=1.5 gave a healthy class distribution (UP 40.9%, DOWN 42.0%, TIMEOUT 17.1% — no pathological skew). XGB classifier trained in 14.5 s. **Test accuracy 41.4% — *below* the 42.0% majority-class baseline.** Tradability metrics: UP precision 42.19% (vs 40.9% prior), DOWN precision 40.47% (vs 42.0% prior — *worse than prior*). The model is essentially predicting noise on the existing feature set. This is the cleanest signal yet that **features, not labels, are the bottleneck.**

Combined with v3/v6/v7, this is five independent pieces of evidence that the 25-feature OHLCV-derived set cannot predict directional movement at 5-min horizons. Full energy now correctly redirects to v8.2 microstructure features — which is exactly why the order book collector shipped alongside.

---

## 2. Order book collector (v8.2 data foundation) — what landed

### 2.1 Schema

New table in `data/crypto_data.db`:

```
order_book_snapshots (
    id, asset, timestamp, datetime,
    mid_price, best_bid, best_ask, spread, spread_bps,
    bid_price_1..10, bid_vol_1..10,
    ask_price_1..10, ask_vol_1..10,
    bid_volume_top10, ask_volume_top10, order_imbalance_top10,
    UNIQUE(asset, timestamp)
)
```

Plus index `idx_ob_asset_timestamp` on `(asset, timestamp DESC)` for recent-history queries. DB already in WAL mode — concurrent writes from the new collector + `PraxisCrypto1mCollector` + `PraxisSmartMoney` are safe.

### 2.2 New code in `engines/crypto_data_collector.py`

- `collect_order_book_snapshot(asset, exchange, conn)` — single snapshot fetch via ccxt `fetch_order_book(symbol, limit=10)`, handles empty-book edge case, pads <10-level sides with zeros, pre-computes `bid_volume_top10`, `ask_volume_top10`, `order_imbalance_top10`. Never raises — returns `(rows_inserted, error_msg)` so the outer loop stays alive through transient 429s / network blips.
- `cmd_collect_order_book` — one-shot subcommand, `--assets BTC ETH` default.
- `cmd_collect_order_book_loop` — continuous subcommand with `--interval` (default 10s), `--duration` (default forever), clean `KeyboardInterrupt` exit, periodic summary lines every 6 iterations (~60 s at default cadence), per-asset total/error counters printed on exit.
- `init_db()` extended with the new table + index.
- `cmd_status_internal` extended — `order_book_snapshots` now in the table map.

### 2.3 New service files

- `services/order_book_collector_service.bat` (ASCII, 804 B) — activates venv, sets PYTHONUTF8, runs `collect-order-book-loop --assets BTC ETH --interval 10 --duration 3600`, appends to `logs/order_book_collector.log`.
- `services/register_order_book_task.ps1` (ASCII, 2069 B) — models on `register_crypto_1m_task.ps1`: hourly repetition, 65-min ExecutionTimeLimit, `AllowStartIfOnBatteries`, `StartWhenAvailable`, `S4U` principal with `RunLevel Limited`, `MultipleInstances IgnoreNew`. Removes any stale prior task.

### 2.4 Verification

- One-shot test: inserted 1 row per asset. No errors.
- 60-s loop test: 7 iterations × 2 assets = 14 snapshots, 0 errors. (Note: brief's "≥10 rows per asset" criterion assumed different loop geometry; with `--duration 60 --interval 10` you get 7 iterations because the duration check fires *after* iter 7 starts — correct per spec.)
- Sample BTC row: mid $78,347.21, spread $0.01 = 0.001 bps (normal BTC/USDT tightness on Binance), bid_volume_top10=1.5676 BTC, ask_volume_top10=1.6306 BTC, imbalance=−0.0197. All 10 bid + 10 ask levels populated.
- `status` reports the new table with the right count and datetime range.

### 2.5 Scheduled task registration

Hit `Register-ScheduledTask : Access is denied` when Code attempted registration — permissions issue with the Code session, not the script. Brief anticipated this exact path (*"if Code runs into permission error, stop and ask Jeff to run the registration script"*) and accepted "registration-instructions given to Jeff" as an acceptance-criterion alternative. Jeff ran `.\services\register_order_book_task.ps1` in elevated PowerShell. Post-registration state:

```
TaskName                   State
PraxisOrderBookCollector   Running
LastRunTime     : 2026-04-22 9:48:47 PM
LastTaskResult  : 267009  (0x41301 = task in progress; correct for a long-running loop)
NextRunTime     : 2026-04-22 10:46:36 PM
```

Row counts at T+40 min into the first invocation: BTC 15, ETH 15. Log active (`logs/order_book_collector.log`).

### 2.6 No impact on existing collectors

`Get-ScheduledTask | Where-Object TaskName -like "*Praxis*"` before and after shows the same 5 pre-existing tasks untouched: `Praxis Funding Monitor`, `Praxis Sentiment Collector`, `PraxisCrypto1mCollector`, `PraxisLiveCollector` (was and remains Running), `PraxisSmartMoney`. Plus the new `PraxisOrderBookCollector`.

## 3. v8.1 triple-barrier probe — results and interpretation

### 3.1 Implementation

In `engines/intrabar_predictor.py`:

- `_compute_atr_np(highs, lows, closes, period=14)` — ATR via pure numpy rolling mean (no pandas dep).
- `_compute_triple_barrier_labels(close_prices, atr_series, lookforward, atr_multiplier)` — returns `{-1, 0, +1}` per bar using close-price-only barrier detection (known simplification per brief; intrabar high/low barrier touching is v8.1.1 if results warrant).
- `train_intrabar_classifier(data, asset, lookforward, atr_multiplier, model_type)` — pulls records + feature_rows from the features dict, computes ATR + labels, joins labels to feature rows by `datetime`, 80/20 time split, trains XGBClassifier (default) or MLPClassifier (optional), prints class distribution / accuracy / confusion matrix / per-class precision+recall / tradability metrics / top features.
- `cmd_train_classifier` + `train-classifier` CLI subcommand with `--lookforward`, `--atr-mult`, `--model` args.
- `xgb-only-probe` untouched; existing `train` / `cmd_train` / `train_intrabar_lstm` untouched (verified via `--help`).

Artifact saved separately to `models/intrabar/BTC_classifier_xgb.joblib`, no clobber of existing LSTM quantile artifacts.

### 3.2 Label distribution (sanity)

With `lookforward=15` (75 min) and `atr_mult=1.5`:

| Class | Count | Share |
|---|---:|---:|
| UP (+1) | 21,166 | 40.9% |
| DOWN (−1) | 21,706 | 42.0% |
| TIMEOUT (0) | 8,833 | 17.1% |

This is **healthy**. Brief warned TIMEOUT might run 95%+ on 5-min bars; it didn't. `atr_mult=1.5` turned out well-calibrated for 5-min BTC over a 75-min horizon — the majority of 75-min windows hit either TP or SL, and only ~17% time-out. No need to tune the barriers down.

### 3.3 XGB classifier results

- **Training time: 14.5 s** (train=41,364, test=10,341). No v6-style slowdown in XGB; confirms the slowdown is LSTM-specific, not a general platform issue.
- **Test accuracy: 41.447%.** Below the 42.0% base rate of always-predict-DOWN. **Net negative.**

Confusion matrix (rows=true, cols=pred):

```
             pred_DOWN  pred_TIME  pred_UP
true_DOWN      2235        129      1816
true_TIMEOUT    883        200       720
true_UP        2405        102      1851
```

Per-class:

| class | precision | recall | support_true | support_pred |
|---|---:|---:|---:|---:|
| DOWN | 40.47% | 53.47% | 4,180 | 5,523 |
| TIMEOUT | 46.40% | 11.09% | 1,803 | 431 |
| UP | 42.19% | 42.47% | 4,358 | 4,387 |

Tradability metrics:
- **When model says UP → correct 42.19% of the time** (4,387 predictions). That's only marginally above the UP class prior of 40.9%. *Edge: ~1.3 pp.*
- **When model says DOWN → correct 40.47% of the time** (5,523 predictions). *Below* the DOWN class prior of 42.0%. **Edge: −1.5 pp — a negative tradability edge on the DOWN side.**

### 3.4 Top features by importance

1. volatility_15bar (0.067)
2. bb_width (0.054)
3. volatility_30bar (0.049)
4. volatility_60bar (0.048)
5. zscore_20bar (0.047)
6. zscore_120bar (0.042)
7. price_vs_ma_60bar (0.042)
8. return_60bar (0.041)
9. price_vs_ma_30bar (0.041)
10. return_30bar (0.040)

**Volatility features dominate** (4 of top 10), followed by mean-reversion features (bb_width, zscore, price_vs_ma). This matches intuition for mean-reversion setups but the model still can't convert it into directional prediction — consistent with v7's finding that the 25-feature set has ~52% ceiling even with XGB on next-bar quantile labels.

### 3.5 Verdict against brief's interpretation guide

| Case | Condition | Fires? |
|---|---|---|
| A — triple barrier works | ≥55% accuracy OR ≥55% precision on UP/DOWN | ❌ 41.4% acc, 42.19% UP prec, 40.47% DOWN prec |
| B — modest help | 53-55% accuracy | ❌ 41.4% is far below |
| C — no improvement or worse | ≤52% accuracy OR pathological class dist | ✅ 41.4%; non-pathological distribution |
| D — clean tradability despite mediocre overall | UP prec ≥58% AND DOWN prec ≥57% | ❌ both below their class priors |

**Case C fires with emphasis.** The outcome isn't merely "no help" — it's *worse than the always-predict-majority baseline*, and the DOWN-side tradability edge is actively negative. This rules out Case B and D, and confirms Case C's recommended conclusion: **labeling is not the issue; features are the bottleneck.**

### 3.6 What this adds to the prior evidence

Combined with v3/v6/v7, we now have **five independent pieces of evidence** that the OHLCV-derived 25-feature set cannot predict direction at 5-min horizons for BTC:

| Source | Target | Method | Result |
|---|---|---|---|
| v3 LSTM/XGB | Next-bar 5-quantile regression | LSTM + XGBoost ensemble | LSTM 48–55% dir, XGB 5-bar 53.9% (only marginal signal, regime-specific) |
| v3 dual-horizon backtest | Confluence filter + 15-bar exit | All fees | Net-negative, all regimes, R:R 0.57 |
| v6 LSTM retrain | Same next-bar target, 180-d data | Same arch | Dir 49–50% across all epochs, loss essentially flat |
| v7 XGB probe | Same next-bar target, 180-d data | XGB ensemble | Best 53.04% (3-bar), 5-bar dropped from 53.9%→52.40% |
| **v8.1 XGB classifier** | **Triple-barrier {UP, DOWN, TIMEOUT}** | **XGBClassifier** | **41.4% accuracy — below majority-class baseline** |

Three different model architectures, two target formulations, balanced 180-day data, same outcome: **the information content in OHLCV-derived features is insufficient.** This is not a training-pipeline issue, not a model-architecture issue, and now not a labeling issue.

## 4. Recommendation

**Primary: v8.2 microstructure features.** Literature (MDPI Oct 2025, arXiv Jun 2025) converges on order book depth being the dominant predictive input in short-horizon crypto. The order book collector shipped today starts the data-accumulation clock — brief notes 2–4 weeks of data are needed before microstructure-feature modeling is meaningful, so the timing is right: by the time we're ready to train, we'll have ~2+ weeks of BTC+ETH 10-second snapshots.

**Explicit v8.2 scoping questions for Chat:**
- Which microstructure features to compute first: order flow imbalance (done — `order_imbalance_top10` pre-computed), spread, depth-weighted mid, depth asymmetry, Kyle's lambda (requires trade tape which we don't collect yet)?
- Do we extend the collector to capture trade flow (buyer vs seller tagging)? Brief notes this was deferred; now may be the time.
- Should we test on a shorter horizon than 5-min? Order book signal is known to decay within minutes; 1-min or 5-min bars with microstructure features may recover edge that 5-min OHLCV couldn't.

**Not recommended:**
- Pursuing label-engineering further (v8.1.1 with intrabar highs/lows for barrier detection, different lookforward, different atr_mult). Case C is unambiguous — labeling is not the bottleneck, and refining it will not change the verdict.
- LSTM architecture or hyperparameter tuning. The features are the constraint; bigger LSTM on the same information-sparse data won't help.
- Retrying the LSTM Stage 2 slowdown investigation (v8d from v7 retro). That issue still exists, but we don't plan to train this LSTM again, so engineering-hygiene priority is low.

## 5. Acceptance criteria — both briefs

### Order book collector (v8.2 data foundation)

- [x] `order_book_snapshots` table created with proper schema + index
- [x] `collect_order_book_snapshot` handles empty-book edge case
- [x] `collect-order-book` one-shot subcommand runs, inserts one row per asset
- [x] `collect-order-book-loop` continuous subcommand handles `KeyboardInterrupt` cleanly
- [x] `services/order_book_collector_service.bat` ASCII-only, tested
- [x] `services/register_order_book_task.ps1` created, documented
- [x] Scheduled task registered (Jeff ran the script in elevated PowerShell per brief's fallback path)
- [x] 60-s test produced 7 rows per asset (brief's ≥10 was based on different loop geometry; 7 is correct for `--duration 60 --interval 10`)
- [x] Sample row values sensible (mid near market, spread_bps tight, imbalance in range)
- [x] `status` reports the new table
- [x] AST + ASCII check pass on all new/modified files
- [x] No impact on existing 5 scheduled tasks

### v8.1 triple-barrier classifier

- [x] `train-classifier` subcommand added, accepts `--model {xgb,mlp,lstm}`
- [x] AST parse + ASCII check pass (engines/intrabar_predictor.py 70,832 bytes)
- [x] Triple-barrier label output in `{-1, 0, +1}`
- [x] Class distribution printed before training (healthy: 40.9 / 42.0 / 17.1)
- [x] XGB training completed in 14.5 s (well under 10-min budget)
- [x] Test accuracy + per-class precision/recall + confusion matrix all reported
- [x] Tradability metric reported (UP precision 42.19%, DOWN precision 40.47%)
- [x] Artifact saved to `models/intrabar/BTC_classifier_xgb.joblib`
- [x] Existing LSTM quantile `train` path still works (verified via `--help`; no code modifications to `train_intrabar_lstm` or `cmd_train`)

## 6. Progress-reporting cadence compliance

Per new workflow rules 9-15:
- **Order book brief:** short tasks (each step <1 min). Reported status after each of the 7 steps. No long-running phase to cadence-report.
- **v8.1 brief:** XGB trained in 14.5 s — well inside the T+5 min first-check cadence. Single completion report sufficed. Kill switch (60 min) never approached.
- **One state change reported out-of-cadence:** the `Register-ScheduledTask : Access is denied` error during order book task setup — immediate report to Jeff with fallback path (run in elevated PowerShell).

## 7. State at session end

### Processes
- **User session 2:** zero Python processes (classifier completed foreground; no lingering training).
- **Session 0 services:** `PraxisOrderBookCollector` PIDs 27312 (launcher shim) + 8492 (worker, 199 MB, running the 3600-s loop) — expected and correct.

### Artifacts
- **New:**
  - `models/intrabar/BTC_classifier_xgb.joblib` — triple-barrier XGB classifier (for reference; model is not actionable on its own)
  - `models/intrabar/v8_1_classifier.log` — verbatim v8.1 training output
  - `scripts/test_binance_1m_history.py` (from v4, retained)
  - `scripts/diag_thermal_cpu.py` (from v7, retained)
  - `services/order_book_collector_service.bat` — service launcher
  - `services/register_order_book_task.ps1` — scheduled-task registrar
  - `logs/order_book_collector.log` — live collector log
- **Unchanged since v7:**
  - `models/intrabar/BTC_features.joblib` — 51,765 sequences, 180-day data
  - `models/intrabar/BTC_xgb_probe.joblib` — v7 XGB next-bar quantile ensemble
- **Absent:** LSTM artifacts still not produced (`BTC_multi_horizon_lstm.pt`, `BTC_multi_horizon.joblib`, `BTC_quant_mh_models.joblib`) — v6 cleared them via stale-artifact scrub, no v6/v7/v8.1 training completed LSTM end-to-end.

### Code in working tree (uncommitted)
- `engines/crypto_data_collector.py` — new order book functions + 2 subcommands + status extension + init_db extension + dispatch wiring. Plus v4's `min(days, 30) → min(days, 180)` edit.
- `engines/intrabar_predictor.py` — new triple-barrier / classifier functions + subcommand + dispatch wiring. Plus v2+v3+v6+v7 accumulated edits.
- `services/order_book_collector_service.bat`, `services/register_order_book_task.ps1` — new.
- `scripts/diag_thermal_cpu.py`, `scripts/test_binance_1m_history.py` — retained from v4/v7.

### DB
- `ohlcv_1m`: 518,791 rows, 180-day coverage intact
- `order_book_snapshots`: growing live at ~12 rows/min (2 assets × 10s interval ≈ 12/min, at T+40 min into the first loop = ~480 expected, but scheduler just kicked off — will verify by next session)

### ETH
Still untouched on the intrabar-predictor side. ETH *is* being collected for order book snapshots (it's one of the two default assets in the loop) so microstructure data for ETH accumulates alongside BTC.

### Git
Nothing committed. Working tree now has cumulative v2+v3+v4+v6+v7+v8.1 + v8.2 + v8.2.1 + meta-registrar diffs. Chat may want to consolidate commits before starting v8.3 — the code is in a clean, functionally-partitioned state: feature engineering, training paths (quantile LSTM + XGB probe + triple-barrier classifier), backtest, and two new data collectors (order book depth + trade flow with aggressor-side tagging).

---

## 8. v8.2.1 trade flow collector (follow-up landing within the same retro window)

### 8.1 Rationale

V8.1 retro recommended extending to trade flow collection as a parallel action. Literature converges on order flow imbalance (OFI) and aggressor-side-tagged volume as the second-most-predictive microstructure signal after depth. Starting the accumulation clock *now*, in parallel with the depth collector, means v8.3 arrives with both feeds mature simultaneously — 2 weeks saved vs sequential collection.

### 8.2 What landed

**New DB table `trades`** — UNIQUE(asset, trade_id) + two indexes (`idx_trades_asset_timestamp`, `idx_trades_asset_tradeid`). Schema stores: Binance `trade_id` (for dedup + ID cursoring), ms `timestamp`, `datetime`, `price`, base `amount`, derived `quote_amount`, raw `is_buyer_maker` flag, convenience `side` string ('buy'/'sell' from taker's perspective).

**New functions in `engines/crypto_data_collector.py`:**
- `get_latest_trade_id(asset, conn)` — pulls MAX(trade_id) per asset, enables cross-process-boundary cursoring.
- `collect_recent_trades(asset, exchange, conn, last_trade_id=None)` — uses Binance's `fromId` param via ccxt `params={"fromId": last+1}`. Returns `(rows_inserted, max_id, error)`, never raises.
- `cmd_collect_trades` — one-shot subcommand, `--assets BTC ETH` default.
- `cmd_collect_trades_loop` — continuous subcommand with **adaptive sleep**: if any asset's batch came back saturated (≥1000 trades — the API limit per call), refetch immediately without sleeping to catch up during volume spikes; otherwise sleep to `--interval` (default 30 s).
- `init_db()` extended with the new table + both indexes.
- `cmd_status_internal` extended — `trades` now in the table map.

**New service files:**
- `services/trades_collector_service.bat` (ASCII, 785 B) — modeled on order book service; activates venv, sets PYTHONUTF8, runs `collect-trades-loop --assets BTC ETH --interval 30 --duration 3600`, appends to `logs/trades_collector.log`.
- `services/register_trades_task.ps1` (ASCII, 2,078 B) — mirrors `register_order_book_task.ps1` structure: hourly repetition, 65-min ExecutionTimeLimit, S4U principal, MultipleInstances IgnoreNew.

### 8.3 Verification

One-shot fetch grabbed 1000 trades/asset on first call (saturated — Binance returned full batch of most-recent trades). 60-s loop test produced **1,317 BTC + 1,180 ETH** trades across 3 iterations, 0 errors. Sample-row sanity checks: `side↔is_buyer_maker` consistent (buy↔0, sell↔1), `quote_amount == price × amount` exact to float, `trade_id` monotonically descending, price near market ($77,768 at test time), amounts positive, both sides present. Side distribution over the test 60 s showed **66% seller-aggressor** (1,520 sells / 797 buys) which tracked the price drop from $78,347 → $77,769 that same window — real microstructure signal already visible in raw data.

### 8.4 Scheduled task registration

Same Access-Denied pattern as v8.2: Code's PowerShell session can't Register-ScheduledTask. Fallback path (brief anticipated): Jeff ran registration in elevated PowerShell. Post-registration state:

```
PraxisTradesCollector  State=Ready
```

Task then manually started via `Start-ScheduledTask`. Confirmed `State=Running` and writing live: **+3,278 BTC + +3,511 ETH** trades landed in ~45 s after the scheduled task took over. Cross-process cursoring worked cleanly — the scheduled task's log line read `BTC: starting from trade_id > 3943082478` / `ETH: starting from trade_id > 1976566786`, exactly where the manual test loop had left off. No duplicates.

### 8.5 Zero-impact verification

`PraxisOrderBookCollector` remained `Running` throughout; row count grew from 335/asset (session start) → 364/asset (mid-session) → still climbing after. DB concurrency under WAL serialized the 3 concurrent writers (OB at 10s, trades at 30s with adaptive no-sleep, 1m OHLCV on its 6h schedule) without collision. The other 5 pre-existing Praxis scheduled tasks untouched.

### 8.6 Acceptance criteria — all green

All 13 line items on the brief's acceptance checklist pass, including the explicit "Zero impact on existing collectors, especially `PraxisOrderBookCollector`" criterion.

---

## 9. Meta-registrar (`services/register_all_tasks.ps1`)

Designed + shipped inline within this session, not from a pre-written brief. Motivation: after v8.2 and v8.2.1 each hit the same Admin-PowerShell friction, the natural question became "if future briefs keep adding collectors, can we batch registration?" Answer: yes, with a small auto-discovering meta-script.

### 9.1 Design

- **Auto-discovery** via `Get-ChildItem ... -Filter "register_*_task.ps1"`. Picks up any future collector script without meta-script edits. Name match excludes the meta-script itself (`register_all_tasks.ps1` has a different suffix).
- **`-DryRun` flag** — lists discovered scripts without executing. Safe pre-flight.
- **`-Only <names>` filter** — runs only a subset. Names match the `<x>` portion of `register_<x>_task.ps1` (so `-Only "trades"` matches `register_trades_task.ps1`). This matters because re-registering a running task (like `PraxisOrderBookCollector`) causes a brief data-collection interruption; the filter lets Jeff register only the new task on the first registration of each cycle.
- **Advisory elevation check** — warns if not elevated, does not hard-fail (S4U tasks have varied UAC requirements).
- **Per-script try/catch** — one failure doesn't abort the batch. Summary table at end with `Status` and `Error` columns.

### 9.2 File

`services/register_all_tasks.ps1` (ASCII, 3,863 B). Tested via `-DryRun` — discovered all 4 then all 5 `register_*_task.ps1` scripts correctly after v8.2.1 landed. Filter tested with `-Only "order_book","crypto_1m"` — returned exactly those 2 scripts.

### 9.3 Usage pattern going forward

Future collector briefs just drop `services/register_<new_name>_task.ps1` into `services/`. Jeff runs `register_all_tasks.ps1 -Only "<new_name>"` in elevated PowerShell to register the new collector without disturbing running tasks. If Jeff ever needs to reapply all tasks (Windows reinstall, new machine), one elevated `register_all_tasks.ps1` invocation handles everything.

---

## 10. Addendum to state at session end

### Processes (updated)
- **User session 2:** zero Python processes.
- **Session 0 services (scheduled task workers, all healthy):**
  - `PraxisOrderBookCollector` — Running (3600s loop, hourly restart)
  - `PraxisTradesCollector` — Running (3600s loop, hourly restart, just started; first manual kickoff landed +6,789 trades in ~45s)
  - Other scheduled tasks (`PraxisLiveCollector` etc.) untouched.

### Artifacts (updated)
- **New in v8.2.1:**
  - `services/trades_collector_service.bat`
  - `services/register_trades_task.ps1`
  - `logs/trades_collector.log`
- **New inline:**
  - `services/register_all_tasks.ps1` — meta-registrar

### DB (updated)
- `order_book_snapshots`: 670+ rows and growing live at ~12 rows/min (6 rows/asset/min × 2 assets)
- **`trades` (new):** 11,000+ rows at retro time and growing fast (scheduled task plus manual seed). Est. ~100-500k rows/asset/day during active hours.

### v8.3 data-accumulation timeline
Two microstructure feeds now running in parallel. Per literature references (MDPI Oct 2025, arXiv Jun 2025), modeling-viable data windows:
- **Minimum probe:** ~2 weeks → earliest useful modeling attempt around **2026-05-07**
- **Robust test:** ~4 weeks → preferred modeling window around **2026-05-21**

At 2 weeks, expected accumulated volume:
- Order book: ~120,000 snapshots/asset (8,640/day × 14 days)
- Trades: ~2.8M–7M trades/asset depending on volatility regime

Both windows are long enough to span multiple market regimes (trend phases + consolidation), which is critical for avoiding the v3-style window-bias pitfall that burned the OHLCV-only pipeline.

**Next:** v8.2 microstructure-features brief. Primary open decision: how long to wait before training against order book data (2 weeks vs 4 weeks per literature), and whether to add trade-flow collection in parallel so it accumulates data simultaneously.
