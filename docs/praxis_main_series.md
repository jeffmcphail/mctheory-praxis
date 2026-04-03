# Praxis — Main Series Chat History

> Comprehensive documentation of the `praxis_main` chat series.
> Captures key decisions, experiment results, architecture evolution, and implementation milestones.
>
> **Purpose:** New chats in this series should read this file plus TRADING_ATLAS.md to be fully up to speed.
> **Last Updated:** 2026-03-28 (Chat: praxis_main_current → retiring as praxis_main_4_momentum-funding-vol)

---

## Project Overview

**Praxis** is a systematic trading research platform for discovering and validating structural edges across asset classes. It uses a CPO (Conditional Parameter Optimization) framework built on Random Forest + Kelly sizing, following the methodology of Ernest Chan and Andrew Burgess.

**Repo:** `C:\Data\Development\Python\McTheoryApps\praxis`

**Stack:** Python, CCXT (crypto/Binance), yfinance (futures/FX), scikit-learn (RF), Streamlit, FastAPI

**File layout:**
```
engines/          # Strategy implementations + CPO core
scripts/          # run_cpo.py, funding_monitor.py, download_dvol.py
gui/              # MCb Backtest Studio + funding monitor dashboard
data/             # Cached market data (funding_cache/, grid_cache/, vol_cache/)
output/           # CPO experiment results
docs/             # REGIME_MATRIX.md, praxis_main_series.md
TRADING_ATLAS.md  # Living research knowledge base (15 experiments)
```

---

## Chat Series Index

| Chat Name | Status | Focus |
|-----------|--------|-------|
| praxis_main_1_* | Archived | Initial platform build, MCb GUI, first CPO experiments |
| praxis_main_2_* | Archived | TA experiments, triple barrier, momentum |
| praxis_main_3_* | Archived | Funding rate carry discovery + validation |
| **praxis_main_4_momentum-funding-vol** | **Retiring** | Grid bot, vol/VRP, regime matrix design, feature space analysis |
| praxis_main_current | **Next** | Regime engine build, Chan/Burgess minute-frequency rebuild |

---

## CPO Framework

The core methodology is Chan (2021) CPO:

1. **Phase 2** — Parameter grid search: simulate strategy across all config combinations × all training days → returns DataFrame
2. **Phase 3** — RF training: each row = (daily_features, config_params) → profitable? The RF learns which market conditions make which configs work
3. **Phase 4** — OOS portfolio trading: RF predicts best config each day, gate on P(profitable) > threshold

**Key insight:** CPO is extreme regime detection at granular level. The RF implicitly learns "this config is profitable when the market is in state X." The regime matrix (docs/REGIME_MATRIX.md) makes this explicit.

---

## Experiment Results Summary

Full details in TRADING_ATLAS.md. Summary:

| Strategy | Verdict | Sharpe | Notes |
|----------|---------|--------|-------|
| Crypto TA (8 types × 8 assets) | ❌ No edge | -1.20 | Confirmed after leverage fix — signal failure not construction |
| Futures TA | ⚠️ Inconclusive | +1.53 | Only 47-day OOS — insufficient |
| FX TA | ❌ No edge | -0.47 | 52-day OOS |
| MCb CPO | ❌ No edge | — | Degenerate base rate |
| Momentum (ETH/SOL) | ⚠️ Weak positive | +0.78 | Triple barrier helped; needs feature redesign |
| **Funding Rate Carry** | **✅ CONFIRMED** | **+4.65 / +10.78** | Two validation periods, Max DD -0.03% |
| Grid Bot | ❌ No edge | -10.79 | Asymmetric P&L — structural failure |
| Vol/VRP | 🔒 Blocked | — | No historical IV data (Deribit API only 16 days) |
| Chan/Burgess pairs | ❌ No edge (TC) | -3.18 | Correct features (AUC 0.87), 57.6% gross WR, but TC > alpha |

---

## Confirmed Edge: Funding Rate Carry

**Strategy:** Long spot + short perp (delta-neutral), collect 8h funding payments, hold 3–14 days.

**Key files:**
- `engines/funding_rate_strategy.py`
- `scripts/funding_monitor.py`
- `gui/funding_monitor/dashboard.py`
- `output/funding_rate/cpo/phase3_models.joblib`

**Primary OOS (Train 2024 → Test 2025, Gate P>0.70):**
- Sharpe +4.45, +0.97% cum, Max DD -0.03%, 6/6 models positive

**Validation (Train 2023 → Test 2024, Gate P>0.70):**
- Sharpe +10.78, +16.73% cum, Max DD -0.03%, 7/7 models positive

**Live monitor:** `python scripts/funding_monitor.py --loop --gate 0.70`
**Dashboard:** `streamlit run gui/funding_monitor/dashboard.py`
**Current status (2026-03-28):** All funding rates negative — monitor sitting flat correctly.

**Why it works:** Clear economic mechanism (perp traders pay for leverage). Features are mechanistically coupled to P&L driver. This is the template for feature engineering.

---

## Critical Discovery: Chan CPO Implementation Error — RESOLVED

**Discovered:** 2026-03-28. **Fixed:** 2026-04-02.

Chan's paper specifies 8 indicators × 2 assets × 7 lookback windows = 112 features from minute bars. Our original implementation used ~17 features from daily close prices. The fix (`engines/minute_features.py`) improved AUC from 0.82 to 0.873.

**Result after fix:** The RF genuinely discriminates (57.6% gross win rate from 28% base rate), but the gross alpha (+5.3 bps/model-day) is insufficient to overcome TC (11 bps/day at 2.7 trades/day). The strategy failure is economic (TC > alpha), not methodological.

**Regime ablation finding:** Trend (A, +7.25% AUC lift) and Serial Correlation (D, +5.65%) are the most informative regime dimensions for equity pairs mean-reversion. Full regime (all classes) provides +10.39% lift, confirming classes are roughly additive.

---

## Regime Matrix Design

Designed in `docs/REGIME_MATRIX.md`. 12 regime classes:

| Class | States | Key use |
|-------|--------|---------|
| A. Trend | 5 ordinal (-2 to +2) | ADX + multi-period return alignment |
| B. Vol level | 4 ordinal | Garman-Klass RV vs 252d percentile rank |
| C. Vol trend | 3 | rv_1d / rv_7d ratio |
| D. Serial correlation | 5 ordinal | **Hurst exponent** (most important single feature) |
| E. Microstructure | 3 | OFI from OHLCV (Lee-Ready approximation) |
| F. Funding/positioning | 5 ordinal | Funding ann % + OI change |
| G. Liquidity | 4 ordinal | Corwin-Schultz spread + Amihud |
| H. Cross-asset corr | 3 | Rolling pairwise corr mean |
| I. Volume participation | 4 ordinal | Volume z-score + price-volume sign |
| J. Term structure | 3 | Funding term slope |
| K. Dispersion | 3 | Cross-sectional return std |
| L. RV/IV spread | 3 | GARCH vs DVOL (needs real options data) |

**Experimental protocol:**
1. Build `engines/regime_engine.py` — computes all 12 states daily
2. Run CPO ablation — which regime classes predict each strategy?
3. Drill into winning classes → extract raw sub-features
4. Rebuild strategy feature sets from raw sub-features
5. Switch winning strategies to dollar bars (Lopez)

---

## Infrastructure Built

### Engines
- `engines/cpo_core.py` — Phase 2/3/4 runner
- `engines/chan_cpo.py` — Burgess→Chan pairs CPO (v1, daily features — superseded by v2)
- `engines/chan_cpo_v2.py` — **NEW** CPO v2 integration: minute features + regime ablation
- `engines/minute_features.py` — **NEW** 112 Chan paper features from minute bars (8 indicators × 2 assets × 7 lookbacks)
- `engines/regime_engine.py` — **NEW** 12-class regime matrix detector (A–L)
- `engines/ta_models.py` — TA strategy (8 types)
- `engines/universal_ta_strategy.py` — Futures/FX TA
- `engines/momentum_strategy.py` — TSMOM/XSMOM/DUAL
- `engines/funding_rate_strategy.py` — Carry strategy ✅
- `engines/grid_bot_strategy.py` — Grid bot
- `engines/garch_model.py` — GARCH(1,1)/EGARCH/GJR ensemble
- `engines/vol_surface.py` — Deribit DVOL + options surface features
- `engines/vol_strategy.py` — VRP strategy (blocked on data)
- `engines/triple_barrier.py` — Triple barrier exits

### Scripts
- `scripts/run_cpo.py` — Unified CPO runner for all strategies
- `scripts/funding_monitor.py` — Live funding rate monitor (loop + webhook)
- `scripts/download_dvol.py` — Deribit DVOL downloader

### GUI
- `gui/mcb_studio/` — MCb Backtest Studio (React+Vite+FastAPI)
- `gui/funding_monitor/dashboard.py` — Streamlit live dashboard

---

## Pending TODOs

**Immediate priority:**
1. ~~Build `engines/regime_engine.py`~~ ✅ DONE (2026-04-01)
2. ~~Rebuild Chan/Burgess CPO with minute-frequency features~~ ✅ DONE (2026-04-01)
3. ~~Build v2 integration (chan_cpo_v2.py)~~ ✅ DONE (2026-04-01)
4. **Run regime ablation experiments** — requires Polygon minute data locally
5. **Run Chan CPO v2 with correct features** — expected to unlock results that previously appeared as failures

**Also outstanding:**
- Alert when crypto funding rates flip positive/sustained (bull phase)
- Funding rate model retrain trigger (6-month auto-retrain of phase3_models.joblib)
- Vol/VRP experiment with real Deribit data (Tardis.dev subscription or proxy)
- Flash loan + stat arb deep dive (from earlier praxis_main chat)
- Dollar bar data pipeline for validated strategies

---

## Key Principles Established

1. **Standard TA has no persistent edge** — confirmed across 6 experiments on 4 asset classes
2. **Funding rate carry is a structural (not statistical) edge** — driven by perp mechanics
3. **Implementation fidelity to source papers is critical** — Chan CPO failure was entirely due to daily vs. minute-frequency features; feature lookback windows (50-3200 min) were confused with trading parameter grid (30-720 min)
4. **Features must be mechanistically coupled to P&L** — funding carry works because features directly characterize the carry profitability condition
5. **CPO is granular regime detection** — the regime matrix makes this explicit and principled
6. **Hurst exponent is the most underused feature** — it directly answers "does this market favor momentum or mean-reversion?"

---

*Last updated: 2026-04-02 (Chat: praxis_main_current)*
*Changes: Chan CPO resolved — AUC 0.873 but TC>alpha. Regime ablation complete. Code cleanup (pairs_trading.py + cpo_training.py). OOS bugfixes.*
