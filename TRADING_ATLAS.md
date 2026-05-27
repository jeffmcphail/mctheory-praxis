# Trading Strategy Atlas

> **Sync state:** This file is the source of truth. After editing, run
> `python -m engines.atlas_sync` to update the queryable DB at
> `data/praxis_meta.db`. See `docs/ATLAS_DB.md`.

## Purpose

A systematic knowledge base mapping the landscape of trading strategies across asset classes, signal types, time horizons, and market regimes. Every experiment adds a data point. The atlas compounds — when a new trading idea appears, we can immediately contextualize it against accumulated evidence and optimize faster.

## How to use this atlas

1. **New paper/idea arrives** → Check atlas for the asset class + signal type combination
2. **Atlas has prior data** → Apply known risk management regime, optimal time horizon, TC sensitivity
3. **Atlas has no data** → Run the CPO framework, add results, expand the map
4. **Breakthrough in one cell** → Systematically test equivalent approach in adjacent cells

---

## Landscape Matrix

### Signal types (rows)
- **TA_STANDARD**: RSI, MACD, Bollinger, EMA Cross, Stochastic, ATR Breakout, Volume Breakout, VWAP Reversion
- **TA_ADVANCED**: Ichimoku, Heikin-Ashi, Keltner Channel, Donchian, Parabolic SAR, Elder Ray
- **MEAN_REVERSION**: Cointegration pairs, stat arb, Ornstein-Uhlenbeck, half-life based
- **MOMENTUM**: Time-series momentum, cross-sectional momentum, dual momentum
- **MICROSTRUCTURE**: Order flow imbalance, book pressure, trade arrival rate, Kyle's lambda
- **FUNDAMENTAL**: Carry, value, quality factors, earnings momentum
- **ALTERNATIVE**: Funding rate arb, cross-exchange arb, on-chain metrics, sentiment, news NLP
- **VOLATILITY**: Vol surface trading, variance risk premium, VIX term structure, gamma scalping

### Asset classes (columns)
- **CRYPTO**: BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE (24/7, low TC, high vol)
- **EQUITY_US**: SP500 constituents (market hours, moderate TC, regulated)
- **FUTURES_INDEX**: ES, NQ, YM, RTY (nearly 24h, very liquid, low TC)
- **FUTURES_COMMODITY**: CL, GC, SI, NG, ZC, ZS (session hours, liquid, clear regimes)
- **FX_G10**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, etc. (24/5, very liquid, lowest TC)
- **FX_EM**: USD/MXN, USD/ZAR, etc. (wider spreads, carry-dominated)
- **OPTIONS**: SPX/VIX vol surface, single-stock options (complex payoffs, Greeks)
- **FIXED_INCOME**: Treasuries, rate futures, credit spreads

---

## Completed experiments

### 1. MEAN_REVERSION × EQUITY_US (SP500 Pairs)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-22 (updated 2026-04-02) |
| **Framework** | Burgess pair discovery → Chan CPO intraday execution |
| **Signal** | Cointegration-based spread z-score, single-asset execution |
| **Data** | Polygon.io minute bars, 38 deduplicated pairs |
| **Training** | 2025 (240 configs × 247 days × 38 pairs = 2.3M strategy-days) |
| **OOS** | 2026-01-01 → 2026-03-20 (53 trading days, 45 with active models) |
| **TC** | 2 bps/leg × 2 legs = 4 bps round-trip |
| **RF AUC** | 0.855-0.896 (mean 0.873) — minute-frequency features per Chan paper |
| **Result** | **NEGATIVE after TC** |
| **Computational engine** | 1 (Cointegration/Mean-Reversion); secondary 3 (Allocation) |

**Corrected results (2026-04-02 — with correct minute-frequency features + OOS bugfixes):**

Previous run used daily-bar features (~17 features, AUC 0.77-0.87) and had two OOS bugs:
notional_capital defaulted to 1.0 (raw dollar P&L treated as %), and no spread_history for
z-score warmup (60% of configs produced zero trades). Both are now fixed.

| Metric | Old (daily features, broken OOS) | New (112 minute features, fixed OOS) |
|--------|----------------------------------|--------------------------------------|
| Feature count | 17 (daily bars) | 112 (minute bars, 8 indicators × 2 assets × 7 lookbacks) |
| RF AUC | 0.77-0.87 (mean 0.82) | 0.855-0.896 (mean 0.873) |
| Predictions with trades | ~6% (z-score warmup bug) | 99.6% |
| Gross win rate | unknown (broken normalization) | **57.6%** |
| Avg gross per model-day | unknown | **+0.053%** |
| Avg TC per model-day | unknown | **0.110%** (2.7 trades × 4 bps) |
| Avg net per model-day | unknown | **−0.056%** |
| OOS Sharpe (portfolio) | -1.44 (broken) | -3.18 (P>0.65 gate, 0.2 models/day) |
| OOS cumulative | unknown | -0.01% (essentially flat) |

**Key findings (updated):**
- The RF genuinely discriminates: 57.6% gross win rate from ~28% base rate is real conditional lift
- Minute-frequency features (Chan paper spec) improved AUC by +0.05 over daily features
- But the underlying gross alpha is only 5.3 bps per model-day
- TC at 2.7 trades/day × 4 bps = 11.0 bps, which is 2× the gross alpha
- No amount of feature engineering can fix gross_alpha < TC

**Regime ablation results (which market dimensions matter for pairs MR):**

| Rank | Class | AUC Lift | Interpretation |
|------|-------|----------|----------------|
| 1 | **A. Trend** | +7.25% | ADX tells you when NOT to mean-revert |
| 2 | **D. Serial corr** | +5.65% | Hurst confirms MR vs momentum regime |
| 3 | **G. Liquidity** | +4.75% | Can you execute the spread? |
| 4 | **I. Volume** | +3.98% | Is there participation? |
| 5 | E. Microstructure | +2.65% | OFI adds moderate value |
| 6-12 | B,C,F,H,J,K | ≤0.4% | Noise for equities |
| — | Full (all classes) | +10.39% | Classes are roughly additive |

**Atlas principle established:**
> Intraday mean-reversion on liquid US equities has a genuine gross signal (57.6% win rate, +5.3 bps/trade) but insufficient alpha to overcome retail TC (11 bps/day at 2.7 trades/day). The strategy requires TC < 2 bps round-trip to be viable. The RF learns real conditional patterns — trend regime (ADX) and serial correlation (Hurst) are the most informative features — but better features cannot rescue insufficient gross alpha.

**Risk management lessons:**
- OOS must match training exactly: notional_capital, spread_history warmup, same lookback_days
- Return normalization: always use notional = target_open + |HR| × hedge_open
- z-score warmup: pass previous 10 days of minute spread to avoid dead configs in OOS
- UTC→Eastern timezone conversion critical for US equity minute data
- Pair deduplication (canonical key = sorted tickers) prevents double-counting

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | minute time bars |
| Frequency | per-minute, 9:30-16:00 ET equity hours |
| Universe | 38 deduplicated SP500 cointegrated pairs (Burgess discovery) |
| TC | 4 bps round-trip per pair |
| Feature set | 112 minute-frequency features (Chan paper spec; 8 indicators × 2 assets × 7 lookbacks). See `engines/minute_features.py:51,54-63` for canonical constants. |
| Pre-filter | none -- all pairs from Burgess discovery feed CPO directly |
| Risk management | equal weight, P>0.65 gate, notional_capital normalization, spread_history warmup |
| Computational engine | Engine 1 (Cointegration); composes with Engine 3 (Allocation) for portfolio construction |

**Active regimes during test:**

Regime ablation was run during this experiment (rare
feature -- most experiments don't have this data). Top
predictive classes for pairs MR:

- A (Trend): +7.25% AUC lift -- ADX tells you when NOT to mean-revert
- D (Serial correlation): +5.65% lift -- Hurst confirms MR vs momentum regime
- G (Liquidity): +4.75% lift -- can you execute the spread?
- I (Volume): +3.98% lift -- is there participation?
- E (Microstructure): +2.65% lift -- OFI adds moderate value
- B (Vol level): <=0.4% AUC lift -- noise for equities
- C (Vol trend): <=0.4% AUC lift -- noise for equities
- F (Funding/positioning): <=0.4% AUC lift -- noise for equities
- H (Cross-asset corr): <=0.4% AUC lift -- noise for equities
- J (Term structure): <=0.4% AUC lift -- noise for equities
- K (Cross-sectional dispersion): <=0.4% AUC lift -- noise for equities
- Full additive: +10.39% lift across all classes

The OOS test window (2026-01-01 to 2026-03-20) was a
medium-trend, low-vol period for SPX with abundant liquidity.

**Revival hypotheses:**

1. **Lower-TC venue or different asset class** -- likelihood: high. The signal is real (57.6% gross win rate, +5.3 bps/trade) but TC of 4 bps RT doubles the gross alpha. A market with TC < 2 bps RT (e.g., direct broker access at institutional rates, or moving to FX major pairs) could flip this NEGATIVE -> POSITIVE.
2. **Dollar-bar resampling** -- likelihood: low-to-medium. Mechanism A (bar count drops 10-30× at a $100M/pair threshold) alone gets trade count well below the 1.325/day break-even at unchanged gross alpha; Mechanism B (per-bar gross alpha rises via event concentration) is magnitude-uncertain and could go either way. Worst case: positive but small (~+0.5 bps/day). Best case: comfortably positive. Cycle 38 RECON noted dollar bars could fix the TC arithmetic mechanically but cannot predict whether net gross alpha holds up under event-time sampling.
3. **Add ADX trend filter (Class A)** -- likelihood: medium. Class A produced the largest single-class lift (+7.25%). Only trade when ADX < 25 (no trend). Reduces trades, may improve net by avoiding adverse trend regimes.
4. **Re-run with intraday minute features but daily holding period** -- likelihood: low. The current setup already does this; the 2.7 trades/day count isn't the cost driver.

**Cycle 38 RECON addendum (2026-05-15):**

Read-only investigation cycle. Verified the 2026-04-02 implementation against Chan's spec and reviewed the TC arithmetic. No code, no atlas DB updates, no experimental output produced.

- **Implementation faithful to Chan paper spec.** `engines/minute_features.py:1-33` cites Chan, Belov, Ciobanu (2021) "Conditional Parameter Optimization" + Chan (2021) "Quantitative Trading" 2e, Ch.7 pp.139-147 as sources for the 8-indicator × 7-lookback feature design. Constants live at `engines/minute_features.py:51` (`FEATURE_LOOKBACKS = [50, 100, 200, 400, 800, 1600, 3200]`) and `54-63` (`INDICATOR_NAMES`). The atlas's prior "7 indicators × 16 lookback combinations" description was structurally wrong (right count 112 by coincidence, wrong factorization); corrected this cycle.
- **Gross/TC numbers (5.3 / 11 / -5.6 bps) internally consistent.** The chain 5.3 gross − (2.7 trades × 4 bps) = -5.5 net matches the atlas's -5.6 within rounding. The 5.3 gross figure itself is only reproducible by re-running phase4 -- `output/burgess/chan_cpo/phase4_portfolio.parquet` was deleted in commit a2202a7. Cycle 38 RECON chose not to re-run because the conclusion is mechanical: break-even trade count at 5.3 gross / 4 bps RT is ~1.325/day; current 2.7/day cannot be halved within the existing bar-type-and-TC regime.
- **Revival hypotheses ranked (re-prioritized post-RECON):** (a) lower-TC venue or asset class (highest likelihood; mechanical), (b) dollar-bar resampling (uncertain; Mechanism A dominates but B magnitude unknowable without running), (c) ADX trend filter (medium; smallest expected lift but cheapest to test).
- **Project-memory clarification.** A separate `lookback ∈ {30, 60, 90, 120, 180, 240, 360, 720}` min set referenced in project notes is the z-score TRADING-PARAM grid (`engines/minute_features.py:25-28` annotation), NOT a feature-lookback alternative to Chan's `{50, 100, 200, 400, 800, 1600, 3200}` set. Any future "8-lookback rebuild" framing should clarify which knob is being touched.

---

### 2. TA_STANDARD × CRYPTO

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-23 |
| **Framework** | CPO generic engine, CCXT (Binance) hourly bars |
| **Signal** | 8 TA types: RSI, MACD, BOLL, EMA_CROSS, STOCH, ATR_BREAK, VOL_BREAK, VWAP_REV |
| **Assets** | BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE |
| **Training** | 2024 (338 configs × 365 days × 64 models = ~1M strategy-days) |
| **OOS** | 2025-01-01 → 2025-12-31 (357 trading days) |
| **TC** | 2 bps/leg × 2 legs = 4 bps round-trip |
| **RF AUC** | 0.77-0.87 (consistent across all periods) |
| **Result** | **NEGATIVE** (-53.9% cum, Sharpe -0.94) |
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Validation (Train 2025, OOS 2026):** -11.2%, Sharpe -1.63. Consistent negative result.

**Pre-filter stability test:**

| TA Type | Profitable in 2024 training? | Profitable in 2025 training? |
|---------|------------------------------|------------------------------|
| MACD | NO (worst) | YES (best) |
| STOCH | YES (best) | NO |
| RSI | YES | NO |
| ATR_BREAK | NO | YES |
| BOLL | YES (marginal) | NO |
| EMA_CROSS | YES | YES (only stable one) |
| VWAP_REV | NO | NO |
| VOL_BREAK | YES | YES |

**Key findings:**
- RF config selection works (lifts win rate from ~35% base to ~42-44% actual)
- But config selection alone doesn't overcome the negative average alpha of the model universe
- **Model type rankings completely invert between years** — no stable structural edge
- Only EMA_CROSS and VOL_BREAK showed marginal consistency, but not enough to be portfolio-positive
- Overfit risk management (circuit breakers, rolling gates) can create appearance of profitability on any single OOS window but fails on validation

**Atlas principle established:**
> Standard TA signals on crypto have no persistent edge. Model type profitability cycles on ~annual timescale, making pre-filtering unreliable. Config optimization (the "Chan" procedure) provides genuine lift but cannot rescue a universe of models with zero average alpha.

**Overfitting lessons (CRITICAL for all future work):**
- Any risk management tuned while observing OOS results is overfit by definition
- Validation requires: train period A → test period B, then train period C → test period D, with NO parameter changes between runs
- Circuit breakers, rolling performance gates, and threshold tuning all failed validation
- The only honest portfolio construction: equal weight with per-model caps, P > 0.50 gate

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE (8 assets) |
| TC | 4 bps round-trip |
| Feature set | 8 TA types (RSI, MACD, BOLL, EMA_CROSS, STOCH, ATR_BREAK, VOL_BREAK, VWAP_REV) x 338 signal configs |
| Pre-filter | training-period top-performing TA types kept; unstable across years |
| Risk management | equal weight, P>0.50 gate, per-model cap |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) for portfolio construction |

**Active regimes during test:**

Native regime ablation was not run for this experiment (predates the
12-class regime engine instrumentation). The OOS window (2025 full
year + 2026 Q1) spans both bull and corrective crypto regimes, which
is consistent with the experiment's primary finding: model-type
rankings invert across years, suggesting the strategy is heavily
regime-dependent even though no specific class is identified.

- general: not_measured -- recommend re-running with regime engine
  features (Class A Trend, Class B Vol level, Class D Serial
  correlation) as ablation candidates

**Revival hypotheses:**

1. **Information-driven bars (dollar bars)** -- likelihood: medium. The annual model-type inversion (MACD <-> STOCH <-> RSI swapping which is profitable) is consistent with TA indicators sampling at wrong-frequency for the underlying market activity rhythm. Dollar bars at $1M threshold give consistent economic-activity granularity that may stabilize TA indicator behavior across regimes. Substrate is now available (Cycle 34 info_bars table, BTC + ETH at $1M and $5M thresholds; volume bars at 100/500 BTC and 1000/5000 ETH). Predicted lift: stabilization of model-type rankings, not necessarily positive absolute alpha.
2. **MCb-style composite gating** -- likelihood: low. Already explored in Exp 7 -- multi-indicator confluence helps signal quality but doesn't reach portfolio-positive even at GUI level on BTC 2024 bull, and CPO integration architecturally failed. Layering MCb over standard TA doesn't fix the underlying signal weakness.
3. **Triple-barrier labeling on TA signals** -- likelihood: low. Exp 10 already attempted this and produced INCONCLUSIVE (leverage runaway). Even if the leverage fix lands, the underlying TA-on-crypto signal is the binding constraint, not exit timing.
4. **Accept the verdict** -- likelihood: high. The cumulative evidence (this experiment + Exp 3 futures + Exp 4 FX + Exp 17 1-min momentum) is conclusive: standard TA has no persistent edge across asset classes. Future Engine 2 work should focus on either (a) structurally-grounded momentum like TSMOM (Exp 8/9) or (b) Engine 7 features (Engine 7 / Exp 13 funding carry is the confirmed-edge precedent).

---

## Pending experiments (legitimate; not yet run)

### FUNDAMENTAL × FX_G10 (carry)
- **Signal type**: Interest rate differential carry
- **Hypothesis**: Carry has documented risk premium; CPO may
  optimize timing. Originally listed as Exp 7 in earlier
  planning; not yet run as a standalone CPO experiment.

### ALTERNATIVE × CRYPTO (on-chain)
- **Signal type**: Active addresses, exchange flows, whale
  movements, hash rate trends.
- **Hypothesis**: Alpha from information edge rather than
  pattern edge. Originally listed as Exp 8 in earlier
  planning; data pipeline now in place via `onchain_btc`
  collector (Cycle 30 + Cycle 31), but no CPO experiment yet
  built.

> Historical note: this section previously listed pending
> experiments numbered 3-8. Items 3, 4 (TA_STANDARD × FUTURES,
> TA_STANDARD × FX_G10) became experiments 3 and 4. Items 5, 6
> (MOMENTUM, MICROSTRUCTURE) were renumbered to experiments 8
> and 13 when implemented. Items 7, 8 (FUNDAMENTAL FX, ALT
> CRYPTO) above remain genuinely pending.

---

## Cross-cutting principles (accumulated wisdom)

### Data pipeline
- US equities: Polygon.io (minute bars, UTC→Eastern conversion required)
- Crypto: CCXT/Binance (hourly bars, UTC, no API key needed for historical)
- Always cache to parquet with skip-if-exists
- Canonical date normalization at every daily/minute boundary

### RF architecture (stable across all experiments)
- Training: (daily_features, config_params) → P(gross_profitable)
- Train on GROSS (pre-TC) profitability for signal; size with NET returns
- 200 trees, max_depth=5, min_samples_leaf=20, class_weight="balanced"
- AUC consistently 0.77-0.87 across all asset classes tested
- Config selection lift: +5-10% win rate improvement (real, not overfit)

### Portfolio construction (first principles)
- Equal weight with per-model cap (0.05 default) — proven robust
- P > 0.50 gate (RF thinks selected config is better than random)
- Kelly mode available for persistent models with sufficient history
- Correlation dedup at 0.85 for equities, skip for crypto (too correlated)
- **No circuit breakers, rolling gates, or other reactive risk management**

### Overfitting defense protocol
- Never tune parameters while observing OOS results
- Validation: {train A, test B} then {train C, test D} with frozen parameters
- If result flips between validation periods, it's overfit
- Pre-filter must use ONLY training data (model type profitability)
- Document every parameter change and which data motivated it

### TC sensitivity
- Strategy is only viable if: gross_alpha > TC × trades_per_day
- SP500 intraday: ~8-12 bps/day TC, insufficient gross alpha
- Crypto hourly: ~4-8 bps/day TC, insufficient alpha from standard TA
- Any future strategy must pass TC viability check BEFORE RF optimization

---

## Atlas metadata

| Field | Value |
|-------|-------|
| Created | 2026-03-23 |
| Framework version | CPO v6 |
| Code location | engines/cpo_core.py, engines/*_strategy.py |
| Data sources | Polygon.io (equities), CCXT/Binance (crypto) |
| Total experiments | 15 complete |

---

### 3. TA_STANDARD × FUTURES (ES, NQ, YM, RTY, CL, GC, SI, NG)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-23 |
| **Framework** | CPO universal TA strategy, yfinance hourly bars |
| **Signal** | 8 TA types (same as crypto) |
| **Assets** | ES, NQ, YM, RTY (index), CL, GC, SI, NG (commodity) |
| **TC** | 1.0 bps/leg × 2 legs = 2 bps round-trip |
| **Result** | **NEGATIVE** (regime-dependent, failed cross-period validation) |
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Test A — Train 2025, OOS 2026 Q1:**
- Pre-filter KEEP: VWAP_REV, STOCH, ATR_BREAK, BOLL, RSI
- OOS: +7.16%, Sharpe +1.70, 24/40 positive — **appeared positive**

**Test B — Validation: Train H1 2025, OOS H2 2025:**
- Pre-filter KEEP: ATR_BREAK, MACD, EMA_CROSS (completely different!)
- OOS: -2.45%, Sharpe -1.38, 8/24 positive — **failed validation**

**Conclusion:** Futures TA result was regime-dependent, not structural.

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly (futures hours: equity index 24h via electronic; commodities ~22h) |
| Universe | ES, NQ, YM, RTY (equity index futures) + CL, GC, SI, NG (commodity futures) |
| TC | 2 bps round-trip (yfinance approximation) |
| Feature set | 8 TA types x 8 assets x config grid |
| Pre-filter | training-period top-performing TA types kept; selection inverted between Test A and Test B |
| Risk management | equal weight, per-model cap |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Native regime ablation not run. Test A (2025 train, Q1 2026 OOS)
selected one set of TA types and appeared positive; Test B (H1 2025
train, H2 2025 OOS) selected a completely different set and failed.
The cross-period TA-type inversion is the diagnostic finding.

- general: not_measured -- the strategy's regime sensitivity is implicit in the cross-period instability of pre-filter selection, but specific class contributions weren't measured

**Revival hypotheses:**

1. **Information-driven bars on futures** -- likelihood: medium. Same logic as Exp 2: if regime-dependence is a sampling-frequency issue, info bars stabilize the input distribution. Futures don't have the same retail microstructure as crypto, so the magnitude of improvement may be smaller, but the mechanism is the same. Caveat: info bars not yet built for futures; Cycle 34 only covered BTC + ETH crypto. Pre-requisite work: extend info_bars infrastructure to futures asset class.
2. **Regime D (Serial correlation) hard filter** -- likelihood: medium. Cross-period TA-type inversion is exactly what serial-correlation regime detection is designed to flag. Hurst exponent at ~0.5 means TA signals (which assume either trend or mean-reversion) will inconsistently work. Hard-gate trading to only Hurst >0.55 (trend regime) or <0.45 (MR regime) for the appropriate signal class.
3. **VWAP_REV-only on equity index futures** -- likelihood: low-medium. Exp 11 (triple-barrier re-run on futures) showed VWAP_REV is the standout signal; futures' natural daily-VWAP reference may give that signal more grip than other TA types. Worth isolating before declaring all-TA-on-futures dead.
4. **Accept the verdict** -- likelihood: medium. Less conclusive than crypto/FX TA because the small sample sizes (one test A, one test B) means we can't say with the same confidence as Exp 2. But the underlying logic (TA has no persistent edge) is consistent.

---

### 4. TA_STANDARD × FX_G10 (EUR/USD, GBP/USD, USD/JPY, AUD/USD, etc.)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-23 |
| **Framework** | CPO universal TA strategy, yfinance hourly bars |
| **Signal** | 8 TA types (same as crypto/futures) |
| **Assets** | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP |
| **TC** | 0.5 bps/leg × 2 legs = 1 bps round-trip (tightest of all classes) |
| **Result** | **NEGATIVE** (-0.31% cum, Sharpe -0.59 over Q1 2026 OOS) |
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Test — Train 2025, OOS 2026 Q1:**
- Pre-filter KEEP: EMA_CROSS, ATR_BREAK, STOCH, MACD
- OOS: -0.31%, Sharpe -0.59, 14/32 positive — flat/negative

**Conclusion:** Even with the lowest TC of any asset class, standard TA shows no edge on FX.

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, FX market hours (~24h Mon-Fri) |
| Universe | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP |
| TC | 1 bps round-trip (tightest of all asset classes tested) |
| Feature set | 8 TA types x 8 pairs |
| Pre-filter | EMA_CROSS, ATR_BREAK, STOCH, MACD kept |
| Risk management | equal weight, per-model cap |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Native regime ablation not run. FX markets are notably efficient with
narrow spreads; the experiment isolates TC as a non-issue (1 bps RT,
lowest of any tested class) which makes the negative result a stronger
TA-has-no-edge signal than crypto or futures.

- general: not_measured -- particularly relevant FX dimensions (Class
  F Funding/positioning differentials; Class A Trend per pair; central
  bank policy regime) weren't measured

**Revival hypotheses:**

1. **Carry + TA composite (Engine 7 + Engine 2)** -- likelihood: medium. FX has a structural carry signal (interest rate differentials) that's been documented for decades. Use carry as a structural conditioner (Engine 7-style) and TA as timing within the carry-positive direction. This is analogous to how funding rate carry (Exp 13) works on crypto perps but with central-bank-driven base rates instead of perpetual funding.
2. **News-event filter (high-frequency around scheduled releases)** -- likelihood: low. FX intraday alpha is concentrated around NFP, FOMC, ECB events; this is a different strategy entirely (event-driven, not systematic TA), would be its own experiment, not a TA revival.
3. **Information-driven bars on FX** -- likelihood: low. FX is dominated by institutional flow; sampling-frequency issues are smaller than crypto. Info bars unlikely to surface signal where 1 bps TC already can't.
4. **Accept the verdict** -- likelihood: high. Combined with Exps 2/3, the three-asset-class consensus is "standard TA has no persistent edge anywhere." FX is the strongest "even with negligible TC" data point.

---

## UNIVERSAL PRINCIPLE (confirmed across 4 asset classes, 6 experiments)

> **Standard TA signals have no persistent edge on any asset class.**
>
> Model type rankings invert on 6-12 month timescales. Config optimization
> provides genuine +5-10% win rate lift, but cannot rescue a model universe
> with zero average alpha. Any positive result from a single OOS window is
> regime-dependent, not structural.
>
> **Implication for future research:** Skip any paper proposing "TA indicator
> with optimized parameters" on any asset class. Focus instead on:
> - Structural/mechanical signals (funding rate, order flow, liquidation cascades)
> - Documented risk premia (momentum factor, carry, value)
> - Information edges (on-chain metrics, alternative data, NLP sentiment)

## Updated landscape matrix

| Signal \ Asset | Crypto | Equities | Futures | FX |
|---------------|--------|----------|---------|-----|
| TA_STANDARD | ❌ No edge | ❌ No edge (TC) | ❌ No edge (validated) | ❌ No edge |
| MEAN_REVERSION | — | ❌ No edge (TC) | — | — |
| MOMENTUM | **TODO** | — | **TODO** | — |
| MICROSTRUCTURE | **TODO** | — | — | — |
| FUNDAMENTAL | — | — | — | **TODO** (carry) |
| ALTERNATIVE | **TODO** | — | — | — |

Total experiments: 6 complete (4 negative, 1 false positive caught, 1 TC-killed)
Total experiments remaining: 4+ in promising regions

---

### 7. TA_COMPOSITE × CRYPTO — Market Cipher B (MCb) Strategies

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-24 |
| **Framework** | MCb Backtest Studio (GUI) + CPO integration attempt |
| **Signal** | Multi-oscillator composite: WaveTrend (WT1/WT2), RSI+MFI, Stochastic RSI |
| **Strategies** | Anchor & Trigger, Zero Line Rejection, Bullish Divergence, MFI Momentum |
| **Assets** | BTC, ETH, SOL (tested), BNB/XRP/ADA/AVAX/DOGE (pending) |
| **Data** | CCXT/Binance, 15m and 4h |
| **TC** | 10 bps round-trip (0.1%/leg, Binance spot maker) |
| **Result** | **PARTIAL** (GUI level promising for Anchor & Trigger 15m BTC; CPO integration architecturally failed) |
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

#### GUI Backtest Results (static params, 2024 OOS)

| Strategy | Interval | Return | Sharpe | Win Rate | Max DD | Trades |
|----------|----------|--------|--------|----------|--------|--------|
| Anchor & Trigger | 15m | **+11.78%** | **2.12** | 62.8% | -4.64% | 94 |
| Anchor & Trigger | 4h | +5.03% | ~1.5 | ~65% | ~-2% | ~10 |
| Bullish Divergence | 15m | +4.60% | 2.20 | 61.5% | -5.98% | 26 |
| Zero Line Rejection | 4h | 0.00% | — | — | — | 0 (signals too rare) |
| MFI Momentum | 4h | -12.74% | -2.06 | 28.6% | -16.44% | 98 |

**Note on Anchor & Trigger (15m):** Near-perfect wedge equity curve. Best curve shape observed in all experiments. 2024 BTC was a bull run (40K→100K) — long-only strategies benefit from this tailwind. Cross-year validation (2022 bear, 2023 choppy) is the critical next test.

#### CPO Integration Attempt — FAILED (architectural mismatch, not strategy failure)

| Attribute | Value |
|-----------|-------|
| **Training** | 2024, BTC+ETH+SOL, anchor_trigger, 3-day eval windows |
| **Phase 2** | 26,208 rows, base_rate 1–1.5% |
| **Phase 3** | AUC 0.997–0.999 (degenerate — RF learned "always predict 0") |
| **Phase 4** | -0.32% cum, Sharpe -0.47, inverted calibration |

**Root cause:** CPO requires daily signal frequency (base_rate 15–40%). Anchor & Trigger fires every 2–3 weeks per asset — only 1–2% of 3-day windows contain a trade. With 98.5% class-0 samples, the RF achieves perfect AUC by predicting 0 every day. The AUC of 0.999 is the diagnostic, not a good sign.

**Key findings:**
- MCb composite oscillator captures momentum regimes that simple TA misses — signal quality is higher because it waits for multi-indicator confluence
- Anchor & Trigger on 15m shows promising results after realistic TC — worth validating across years and assets
- MFI Momentum (high frequency, trades every MFI cross) performed worst — confirms MCb edge comes from *waiting*, not from trading every signal
- CPO "Chan procedure" is incompatible with monthly-frequency signals — needs base_rate > 10% to train a meaningful RF

**Atlas principle established:**
> MCb-style composite oscillators are architecturally incompatible with CPO parameter optimization due to low signal frequency (base_rate < 2%). The correct optimization approach is regime-level conditioning (binary on/off switch: "is today a good day to run this strategy?") rather than parameter selection. Build a daily regime classifier using yesterday's MCb features (WT2 position, MFI direction, vol regime) as a separate module — not as a CPO strategy.

**Next steps:**
1. Cross-year validation: Anchor & Trigger (15m BTC) on 2022 (bear) and 2023 (choppy)
2. Multi-asset: does the edge transfer to ETH/SOL or is it BTC-specific?
3. Regime classifier: daily binary model — "favorable regime for Anchor & Trigger?" — using MCb features as on/off gate

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | 15-minute and 4-hour time bars |
| Frequency | per-candle close, 24/7 |
| Universe | BTC, ETH, SOL tested; BNB/XRP/ADA/AVAX/DOGE pending |
| TC | 10 bps round-trip (Binance spot maker) |
| Feature set | WaveTrend (WT1/WT2), RSI+MFI, Stochastic RSI composite |
| Pre-filter | multi-indicator confluence (only fires on agreement; base_rate ~1-2%) |
| Risk management | none beyond signal-based entry/exit |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) for portfolio construction. CPO integration architecturally incompatible due to low base_rate. |

**Active regimes during test:**

Native regime ablation not run. The 2024 BTC bull run (40K -> 100K)
is the dominant regime in the GUI backtest period; cross-year
validation (2022 bear, 2023 choppy) is the explicit "is the edge
real or regime-conditioned" test, not yet run.

- general: not_measured -- bull-regime tailwind almost certainly inflates the 2024 results

**Revival hypotheses:**

1. **Cross-year validation (2022 bear, 2023 choppy) of GUI Anchor & Trigger 15m BTC** -- likelihood: high. The explicit "next step" in existing analysis. 2024 BTC was bull; if the edge survives 2022 bear and 2023 choppy at GUI level (no CPO), the structural claim is supported. Pure data run; no new infrastructure required.
2. **Daily regime classifier as binary gate (atlas-principle direction)** -- likelihood: high. Build a daily binary classifier ("favorable regime for Anchor & Trigger today?") using yesterday's MCb features as features, not CPO. This is the architectural fix for the base_rate issue: instead of trying to predict per-3-day-window whether the signal will be profitable (base_rate 1-2%), predict per-day whether *today is a good day to leave the strategy on*. Base rate becomes 30-50% (most days the strategy is at least neutral), making the RF trainable.
3. **Multi-asset transfer (ETH, SOL, etc.)** -- likelihood: medium. Easy parallel test of "is the edge BTC-specific or transferrable?". Cheaper than cross-year validation; uses existing GUI infrastructure.
4. **Triple-barrier exits on Anchor & Trigger signals** -- likelihood: medium. The 15m Anchor signal fires every 2-3 weeks per asset; current exits are time-based (hold N hours). Triple-barrier (TP/SL/timeout) could improve exit P&L without changing the entry-signal frequency. Doesn't fix the base_rate issue for CPO but doesn't depend on CPO either; pairs naturally with #1 above.

---

## Updated landscape matrix (v2)

| Signal \\ Asset | Crypto | Equities | Futures | FX |
|---------------|--------|----------|---------|-----|
| TA_STANDARD | ❌ No edge | ❌ No edge (TC) | ❌ No edge (validated) | ❌ No edge |
| TA_COMPOSITE (MCb) | ⚠️ GUI edge (2024 bull), cross-year TBD | — | — | — |
| MEAN_REVERSION | — | ❌ No edge (TC) | — | — |
| MOMENTUM | **TODO** | — | **TODO** | — |
| MICROSTRUCTURE | **TODO** | — | — | — |
| FUNDAMENTAL | — | — | — | **TODO** (carry) |
| ALTERNATIVE | **TODO** | — | — | — |

Total experiments: 7 complete (5 negative/incompatible, 1 false positive caught, 1 TC-killed, 1 promising pending validation)

---

### 8. MOMENTUM × CRYPTO (ETH + SOL, TSMOM_4H + TSMOM_DAILY)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-25 |
| **Framework** | CPO generic engine (MomentumCPOStrategy), CCXT/Binance hourly |
| **Signal** | Time-series momentum: past N-hour return predicts next N hours |
| **Types** | TSMOM_4H (4-48h lookback, 4-24h hold), TSMOM_DAILY (24-168h lookback, 12-48h hold) |
| **Assets** | ETH, SOL (BTC excluded — see findings) |
| **TC** | 2 bps/leg × 2 legs = 4 bps round-trip |
| **Gate** | P > 0.60 (RF probability threshold) |
| **Result** | **PARTIAL** (weak; primary +1.91% Sharpe +0.545; validation +0.46% Sharpe +0.245) |
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

#### Phase 3 RF Quality (Train 2024)
| Model | AUC | Base Rate |
|-------|-----|-----------|
| ETH_TSMOM_4H | 0.871 | 35.6% |
| ETH_TSMOM_DAILY | 0.884 | 28.3% |
| SOL_TSMOM_4H | 0.879 | 40.3% |
| SOL_TSMOM_DAILY | 0.895 | 31.6% |

AUC 0.87-0.90, base_rate 28-40% — architecturally correct CPO fit (not degenerate like MCb).

#### Primary OOS (Train 2024 → Test 2025)
| Model | Cum Return | Sharpe | Days |
|-------|-----------|--------|------|
| ETH_TSMOM_4H | +12.4% | +4.63 | 9 ⚠️ |
| SOL_TSMOM_DAILY | +18.3% | +1.37 | 71 |
| ETH_TSMOM_DAILY | +12.3% | +1.20 | 49 |
| SOL_TSMOM_4H | -4.8% | -0.60 | 35 |
| **Portfolio** | **+1.91%** | **+0.545** | 446 |

Calibration at 0.60-0.65: **+5.9% lift** (real conditional signal).

#### Validation (Train 2023 → Test 2024, frozen parameters)
| **Portfolio** | **+0.46%** | **+0.245** | 363 |

Calibration at 0.60-0.65: **+0.5% lift** (essentially flat).

**Full universe result (BTC+ETH+SOL, threshold 0.50):**
- BTC_TSMOM_4H: -53.2% (Sharpe -1.54) — BTC momentum mean-reverts fast
- ETH_TSMOM_4H: +62.4% (Sharpe +1.20) — ETH/SOL momentum persists
- Portfolio: -0.32% (models cancel)

**Key findings:**

**BTC-ETH divergence:** BTC momentum mean-reverts quickly (smart money takes profit fast). ETH/SOL momentum persists longer (retail chases). This is a documented phenomenon — BTC leads, altcoins follow with lag. Including BTC in a momentum portfolio cancels the altcoin signal.

**CPO fit confirmed:** Base rate 28-40% gives the RF genuine variance to learn from. AUC 0.87-0.90 reflects real conditional information, not degenerate prediction. This is the first experiment where the RF architecture worked as designed.

**ETH_TSMOM_4H sample size:** 9 trades in the primary run (Sharpe +4.63) — statistically meaningless. SOL_TSMOM_DAILY at 71 trades with Sharpe +1.37 is the most reliable result.

**Verdict: WEAK POSITIVE — not confirmed structural edge.**
- ✓ Both validation periods positive
- ✓ Correct direction in both periods
- ✓ RF architecture works (AUC/base_rate healthy)
- ⚠️ Calibration doesn't hold across regimes (5.9% lift 2024→2025 vs 0.5% lift 2023→2024)
- ⚠️ Asset selection (ETH+SOL, not BTC) was motivated by OOS observation
- ⚠️ Very sparse trading (0.3-0.4 models/day active) — small sample sizes

**Atlas principle established:**
> Crypto momentum is asset-class-specific. BTC (large-cap, institutionally traded) mean-reverts quickly after moves. ETH/SOL (mid-cap, retail-dominated) show persistent momentum. Including BTC in a momentum portfolio neutralizes the signal. Threshold 0.60 provides weak but consistent conditional lift on ETH/SOL momentum. The RF learns real conditional patterns but the signal is regime-sensitive and not robustly calibrated across years.

**Next steps to confirm or reject:**
1. Extend to full universe (BNB, ADA, AVAX, DOGE) to test if the altcoin pattern generalizes
2. Add VOLSCALE and DUAL momentum types — vol-scaled momentum is more theoretically grounded
3. Add 2022 (bear market) validation — momentum strategies are known to fail in bear markets
4. Longer training window (2022+2023 combined → test 2024) for more stable RF calibration

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | ETH, SOL (BTC excluded -- mean-reverts too fast for momentum) |
| TC | 4 bps round-trip |
| Feature set | TSMOM_4H (4-48h lookback, 4-24h hold) + TSMOM_DAILY (24-168h lookback, 12-48h hold) |
| Pre-filter | asset selection (ETH+SOL only) is itself a pre-filter, motivated by Exp 8 OOS observation -- a soft form of selection bias |
| Risk management | equal weight, P>0.60 gate, per-model cap |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Native regime ablation not run. Cross-year calibration drift (5.9%
lift 2024->2025 vs. 0.5% lift 2023->2024) is the primary regime-
sensitivity diagnostic. BTC vs ETH/SOL divergence (BTC momentum mean-
reverts fast; ETH/SOL momentum persists) is itself a microstructure-
regime difference rather than a market-state regime.

- A (Trend): not measured directly; momentum strategies are by construction trend-conditioned
- B (Vol level): not measured; momentum is documented to fail in low-vol regimes (no signal-to-noise)
- D (Serial correlation): not measured; the most relevant class for momentum-vs-mean-reversion regime detection

**Revival hypotheses:**

1. **Information-driven bars (volume bars or dollar bars)** -- likelihood: medium-high. Cross-year calibration drift is consistent with time-bar sampling-frequency issues -- a momentum lookback of "past 24 hours" captures different amounts of economic activity in low-vol vs. high-vol regimes. Volume bars normalize this. Substrate available (Cycle 34): BTC + ETH at multiple thresholds. ETH-specific volume bars at 1000 / 5000 ETH thresholds are the obvious starting point.
2. **Vol-scaled momentum (TSMOM-VS)** -- likelihood: medium. Standard TSMOM design: score = past_return / past_vol. Theoretically grounded (Moskowitz-Ooi-Pedersen "Time Series Momentum"); more robust across vol regimes than raw N-period returns. Stated as a "next step" in the existing analysis.
3. **Bear market (2022) validation** -- likelihood: needs-data. Momentum strategies are known to fail in bear markets; without bear validation we don't actually know the strategy's worst-case behavior. Required before any production deployment.
4. **Re-include BTC with separate model (not portfolio cancellation)** -- likelihood: low. BTC's mean-reversion is fast, ETH/SOL is persistent; including BTC in the same portfolio cancels both signals. Running BTC as a separate (mean-reversion) strategy and ETH/SOL as the momentum strategy might recover both edges, but this is two experiments not one.

---

## Updated landscape matrix (v3)

| Signal \\ Asset | Crypto | Equities | Futures | FX |
|---------------|--------|----------|---------|-----|
| TA_STANDARD | ❌ No edge | ❌ No edge (TC) | ❌ No edge (validated) | ❌ No edge |
| TA_COMPOSITE (MCb) | ⚠️ GUI edge (2024 bull), cross-year TBD | — | — | — |
| MEAN_REVERSION | — | ❌ No edge (TC) | — | — |
| MOMENTUM | ⚠️ Weak positive (ETH/SOL only, regime-sensitive) | — | **TODO** | — |
| MICROSTRUCTURE | **TODO** | — | — | — |
| FUNDAMENTAL | — | — | — | **TODO** (carry) |
| ALTERNATIVE | **TODO** | — | — | — |

Total experiments: 8 complete (5 negative, 1 false positive, 1 TC-killed, 2 weak positives pending validation)

---

## TRIPLE BARRIER RE-RUN SERIES (v2 — with proper TP/SL)

All experiments below use the unified triple barrier exit framework:
- `sl_pct` ∈ {1%, 2%, 3%}, `tp_pct` = 2× or 3× sl (2:1 to 3:1 ratios)
- `trail_pct` ∈ {0, 0.3%, 0.5%, 1.0%} — 0 = mean-reversion exit, >0 = momentum trailing
- `t_bars` ∈ {24, 48, 96} — vertical barrier
- 72 barrier combinations crossed with signal params for each strategy type

---

### 9. MOMENTUM × CRYPTO (ETH+SOL, Triple Barrier Re-run)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-26 |
| **vs Previous** | Experiment 8 (time-based exit, hold_hours only) |
| **Training** | 2024, ETH+SOL, TSMOM_4H + TSMOM_DAILY, 1728 configs |
| **OOS** | 2025-01-01 → 2026-03-25 (446 days) |
| **RF AUC** | 0.879-0.902, base_rate 27-39% |
| **Result** | **PARTIAL** (+2.43% cum, Sharpe +0.778, Max DD -2.49%; improved over Exp 8 by triple-barrier exits) |
| **Computational engine** | 2 (Momentum/Trend) primary + 7 (Event/Signal for triple-barrier labels); composes with Engine 3 (Allocation) |

**Result: +2.43%, Sharpe +0.778, Max DD -2.49%**

Improvement over time-exit version (Sharpe +0.545): triple barrier raised Sharpe by +0.23, reduced max drawdown from -3.08% to -2.49%. The benefit is modest but consistent — the barrier framework is correctly cutting losers and letting winners run, even if the calibration remains non-monotonic.

Calibration: non-monotonic (0.60-0.65: -1.1%, 0.65-0.70: +5.2%, 0.70-0.80: -8.3%). RF still regime-sensitive.

**Verdict: WEAK POSITIVE — confirmed improvement from triple barrier, same structural caveats as Experiment 8.**

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | ETH, SOL |
| TC | 4 bps round-trip |
| Feature set | TSMOM_4H + TSMOM_DAILY x 1728 (signal config x barrier config) combinations |
| Pre-filter | as Exp 8: asset selection ETH+SOL only |
| Risk management | equal weight, P>0.60 gate, per-model cap, triple-barrier TP/SL/timeout exits |
| Computational engine | Engine 2 (Momentum/Trend) primary; Engine 7 (Event/Signal) for triple-barrier labels; composes with Engine 3 (Allocation) |

**Active regimes during test:**

Same diagnostic as Exp 8 (no native ablation). Improvement from
Sharpe +0.55 to +0.78 vs Exp 8 isolates the triple-barrier exit
mechanism as the lift source. Non-monotonic calibration (0.60-0.65
-1.1%, 0.65-0.70 +5.2%, 0.70-0.80 -8.3%) confirms the underlying RF
remains regime-sensitive even with improved exits.

- A (Trend): not measured
- B (Vol level): not measured
- D (Serial correlation): not measured

**Revival hypotheses:**

1. **Info bars + triple-barrier (the Financial Innovation 2025 paper recipe)** -- likelihood: high. This is the canonical LSTM v2 setup. Move from hourly time bars to dollar bars (BTC + ETH info_bars available Cycle 34); keep the triple-barrier label discipline but apply it in bar-index space (not wall-clock space). Predicted lift: stabilize the non-monotonic calibration across regimes -- the same dollar-bar-event experiences should produce similar feature distributions regardless of when in calendar time they occur.
2. **Volume bars specifically** -- likelihood: medium-high. Dollar bars normalize by dollar volume; volume bars normalize by base-asset volume. For momentum strategies where the underlying physics is "more activity = more conviction = stronger persistence", volume bars may be more direction-stable than dollar bars. Both available in Cycle 34 substrate.
3. **LSTM v2 architecture (Cycle 37+ scope)** -- likelihood: medium for ultimate use, but the RF here works architecturally. Replacing RF with LSTM is a bigger architectural commitment that should happen alongside info-bar + triple-barrier integration, not separately.
4. **Bear market (2022) validation** -- likelihood: needs-data. Same as Exp 8.

---

### 10. TA_STANDARD × CRYPTO (Triple Barrier Re-run)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-26 (original); revival 2026-05-13 (Cycle 36c) |
| **Training** | 2024, 8 crypto assets × 8 TA types × 110 signal configs × 72 barrier configs = 7,920 configs (universal_ta crypto path) |
| **OOS** | 2025-01-01 → 2026-03-25 (441 days) |
| **RF AUC** | 0.771-0.854 (original); 0.790-0.890 (Cycle 36c re-train, slightly elevated). base_rate 22-52% both runs. |
| **Pre-filter kept** | STOCH, RSI, EMA_CROSS, VOL_BREAK, BOLL (5/8 types, original); STOCH, VOL_BREAK, ATR_BREAK, EMA_CROSS, BOLL (5/8 types, Cycle 36c). One-element swap (ATR_BREAK ↔ RSI). |
| **Result** | **NEGATIVE** (-26.58% / Sharpe -1.18 at cap=0.5; Sharpe invariant across binding-cap settings refutes leverage-cap revival) |
| **Computational engine** | 2 (Momentum/Trend) primary + 7 (Event/Signal for triple-barrier labels); composes with Engine 3 (Allocation -- cap mechanism is a pure leverage scaler, does not change signal Sharpe) |

**Original result (2026-03-26): -83.78%, Sharpe -1.158, Max DD -102.39%** (at cap=2.0 default; realized gross 1.5x avg, 1.8x peak)

**Cycle 36c result (2026-05-13, cap response sweep):**
- cap=2.0: -76.37% / Sharpe -1.1596 / Max DD -97.24% (reproduces original within experimental foundation; ~7pp milder cum, Sharpe & DD match to 3 decimals)
- cap=1.0: -53.17% / Sharpe -1.1844 / Max DD -67.57%
- **cap=0.5 (canonical revival): -26.58% / Sharpe -1.1844 / Max DD -33.78%**
- cap=0.25: -13.29% / Sharpe -1.1844 / Max DD -16.89%

**Root cause -- negative-edge signal amplified by default-construction gross leverage:**

The original -83.78% headline combined two effects: (1) a negative-Sharpe (-1.18) signal-level result from the 40-model TA portfolio, and (2) the equal-weight × per-model-cap construction producing ~150% gross exposure naturally (5% × ~30 models passing gate). The leverage amplified a -56%-magnitude raw signal-level loss to the -84% portfolio result.

Cycle 36c established this by running the cycle at four cap settings (2.0, 1.0, 0.5, 0.25). At binding-cap settings, **Sharpe is constant at -1.1844 to 4 decimals**, with cumulative return scaling exactly linearly with the cap. This is the canonical signature of a pure-leverage-scaler in front of a fixed signal stream. The cap mechanism does not change which models pass the gate, their relative weights, or which days they trade.

**The "leverage construction failure masks signal quality" framing in the prior atlas entry is refuted.** There is no cap setting -- including arbitrarily small caps -- that produces a positive-Sharpe portfolio from this universe. ADA_STOCH/BTC_STOCH "individual models confirm signal works" was based on training-period numbers (Sharpe +2.01, +1.59) that did not generalize OOS. At cap=0.5 OOS, ADA_STOCH led at Sharpe +1.225 / cum +58.8% (one of 6/40 positive-Sharpe models), but the negative tail (ETH_ATR_BREAK -2.34, ETH_VOL_BREAK -2.25, ETH_BOLL -1.91, BTC_BOLL -1.74, DOGE_STOCH -1.70) dominated the portfolio aggregate.

**Verdict: NEGATIVE -- TA signals on crypto produce Sharpe ~-1.18 OOS regardless of portfolio gross cap. The original -83.78% headline was a leveraged amplification of a negative-edge signal, not a construction artifact concealing salvageable edge. Cycle 36c ran four cap settings [0.25, 0.5, 1.0, 2.0]; Sharpe was invariant to 4 decimals across binding-cap settings, decisively refuting the leverage-cap revival hypothesis.**

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | 8 crypto assets x 5 TA types kept (40 individual models) |
| TC | 4 bps round-trip (2 bps/leg) |
| Feature set | 8 TA types x 110 signal configs x 72 barrier configs = 7,920 configs (universal_ta crypto path) |
| Pre-filter | training-period top 5/8 TA types kept; original {STOCH, RSI, EMA_CROSS, VOL_BREAK, BOLL}, Cycle 36c {STOCH, VOL_BREAK, ATR_BREAK, EMA_CROSS, BOLL}. One-element swap (data drift) |
| Risk management | equal weight (5% per-model cap). Default `--max-leverage 2.0` not binding (realized gross 1.5x mean, 1.8x peak). Cycle 36c found Sharpe invariant to 4 decimals (-1.1844) across binding caps 1.0/0.5/0.25 -- the cap is a pure leverage scaler, not a signal-quality filter. |
| Computational engine | Engine 2 + Engine 7 labels; composes with Engine 3 (Allocation -- cap is pure leverage scaler, does not change signal Sharpe) |

**Active regimes during test:**

Not measured. The catastrophic portfolio result masks any
regime-sensitivity analysis. 2025 was broadly corrective for crypto,
which is the worst possible environment for a 175%-leveraged long
basket, but even in a flat regime the construction would have been
unstable.

- general: not_measured -- the leverage construction failure is independent of regime and dominates the result

**Revival hypotheses:**

(Status post Cycle 36c re-run; see Addendum below for full execution log.)

1. **Hard portfolio leverage cap** -- **TESTED, REFUTED** (Cycle 36c, 2026-05-13). Sharpe-invariance across binding-cap settings refutes the mechanism. The cap is a pure leverage scaler; it does not change which models trade or their relative weights, so it cannot alter the underlying signal's Sharpe.
2. **Reduce model count to top-K by training Sharpe** -- **DEMOTED**. Same Sharpe-invariance argument applies: any uniform model-set transformation that preserves relative weights preserves Sharpe of the underlying signal. Top-K filtering on training Sharpe also faces the in-sample-vs-OOS generalization issue documented above (ADA_STOCH/BTC_STOCH training winners didn't carry to OOS).
3. **Info bars / dollar bars** -- **DEMOTED**. Bar construction does not naturally change the Sharpe of the per-bar return stream unless it changes which bars are included -- a distinct research question (sampling-frequency / input distribution stability) rather than a "revival" of this experiment. If pursued, frame as a new experiment, not a hypothesis for Exp 10.
4. **The "TA signals on crypto have persistent edge" working hypothesis from Cycles 1-4** -- **CLOSED, NEGATIVE**. Cumulative evidence (Exp 2 + Exp 10 cap-response sweep + Exps 3/4 sibling negative results) is now decisive across portfolio constructions.

---

### 11. TA_STANDARD × FUTURES (Triple Barrier Re-run)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-26 |
| **Training** | 2025, 8 futures × 8 TA types, 7920 configs |
| **OOS** | 2026-01-01 → 2026-03-25 (47 days only) |
| **RF AUC** | 0.821-0.920, base_rate 17-50% |
| **Pre-filter kept** | VWAP_REV, ATR_BREAK, STOCH (3/8 types) |
| **Result** | **PARTIAL** (+2.93% cum, Sharpe +1.528, Max DD -2.53% -- but only 47-day OOS) |
| **Computational engine** | 2 (Momentum/Trend) primary + 7 (Event/Signal for triple-barrier labels); composes with Engine 3 (Allocation) |

**Result: +2.93%, Sharpe +1.528, Max DD -2.53%**

Strongest calibration seen in any experiment: 0.65-0.70 bin shows **+14.1% lift**, 0.70-0.80 shows **+9.0% lift**. The RF is genuinely discriminating at high confidence levels. RTY and YM lead (VWAP_REV and STOCH). Gold (GC) is the main drag.

**Critical caveat: 47 trading days is not a meaningful OOS window.** Cannot distinguish luck from edge on this sample size. Needs 200+ days to be interpretable.

Interesting finding: pre-filter selected VWAP_REV as the top futures signal (not RSI or MACD). VWAP reversion on 1-2% moves with a 2:1 TP/SL barrier has a natural mean-reversion logic on futures that doesn't exist on crypto.

**Verdict: PROMISING but insufficient OOS data. Extend OOS window to 2025 when more data available.**

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly (futures hours) |
| Universe | ES, NQ, YM, RTY (equity index) + CL, GC, SI, NG (commodity) |
| TC | 2 bps round-trip |
| Feature set | 8 TA types x 8 futures x 7920 configs |
| Pre-filter | training top 3/8 TA types kept (VWAP_REV, ATR_BREAK, STOCH) |
| Risk management | equal weight, P>0.50 gate, triple-barrier exits |
| Computational engine | Engine 2 + Engine 7 labels; composes with Engine 3 (Allocation) |

**Active regimes during test:**

Native regime ablation not run. 47-day OOS window (Q1 2026) is far
too small to assess regime sensitivity meaningfully. Calibration
lift at 0.65-0.70 was +14.1% (strongest of any experiment), but
sample size doesn't separate signal from noise.

- general: not_measured -- 47 days is below the threshold for meaningful regime analysis

**Revival hypotheses:**

1. **Extend OOS window to 200+ days** -- likelihood: very high. The single largest defect of this experiment is sample size. Run the same configuration on 2025 H2 data (or 2025 full year if accessible) -- pure data accumulation, no infrastructure changes needed. Result determines whether the +14.1% calibration lift is real or sampling noise.
2. **Single-strategy focus on VWAP_REV** -- likelihood: medium. Pre-filter selected VWAP_REV as the strongest signal; isolating it (drop the other 7 TA types entirely) tests whether the futures TA edge is VWAP-specific or distributed. Cheap test.
3. **Volume-bar-equivalent on futures (contract-count bars)** -- likelihood: medium. Futures have well-defined contract sizes; the natural "info bar" unit is contracts (or notional dollars). Info bars infrastructure (Cycle 34) was built for crypto trades; extending to futures is its own infrastructure work but uses the same writer/builder framework.
4. **Per-asset triple-barrier tuning** -- likelihood: medium. Gold (GC) is documented as the main drag; the TP/SL barriers were fit on aggregate. Per-asset volatility-aware barrier sizing (e.g., barriers scaled by ATR) could improve the per-asset signal-to-noise.

---

### 12. TA_STANDARD × FX G10 (Triple Barrier Re-run)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-26 |
| **Training** | 2025, 8 FX pairs × 8 TA types, 7920 configs |
| **OOS** | 2026-01-01 → 2026-03-26 (52 days only) |
| **RF AUC** | 0.798-0.925, base_rate 20-54% |
| **Pre-filter kept** | EMA_CROSS, RSI, MACD, STOCH, BOLL (5/8 types) |
| **Result** | **INCONCLUSIVE** (-0.39% cum, Sharpe -0.471; same leverage issue as Exp 10 at smaller magnitude; 52-day OOS too short) |
| **Computational engine** | 2 (Momentum/Trend) primary + 7 (Event/Signal for triple-barrier labels); composes with Engine 3 (Allocation) |

**Result: -0.39%, Sharpe -0.471**

Same leverage issue as crypto TA but much smaller magnitude (40 models at 1.4× leverage, smaller per-trade returns on FX). Individual models are split — 22/40 positive by Sharpe, suggesting the signal has some content but the portfolio cancels out. EMA_CROSS on AUD/USD (Sharpe +4.07) and USD/JPY (+3.06) stand out.

**Verdict: INCONCLUSIVE — 52 days insufficient. Individual model quality mixed, portfolio construction same leverage issue as crypto.**

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, FX market hours |
| Universe | 8 G10 pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP) |
| TC | 1 bps round-trip |
| Feature set | 8 TA types x 8 pairs x 7920 configs |
| Pre-filter | training top 5/8 TA types kept |
| Risk management | equal weight, per-model cap -- same lack-of-portfolio-leverage-cap as Exp 10, smaller magnitude on FX |
| Computational engine | Engine 2 + Engine 7 labels; composes with Engine 3 (Allocation) |

**Active regimes during test:**

Not measured. 52-day OOS too short for regime analysis. Mixed
individual results (22/40 models positive by Sharpe; EMA_CROSS on
AUDUSD Sharpe +4.07, USDJPY +3.06) suggest pair-specific edge exists
but is diluted by 40-model portfolio.

- F (Funding/positioning): for FX, funding is interest rate differentials; carry-bearing pairs likely have a different regime profile than non-carry pairs (AUDUSD vs. EURUSD have very different carry profiles)
- general: not_measured

**Revival hypotheses:**

1. **Hard portfolio leverage cap + extended OOS** -- likelihood: very high. Both fixes are obvious and cheap. Combine the Exp 10 portfolio fix with the Exp 11 OOS-extension; the resulting 200+ day clean-leverage run will give a real verdict.
2. **Restrict to top-performing pairs (AUDUSD, USDJPY)** -- likelihood: medium. Top per-pair results (EMA_CROSS Sharpe +4.07 and +3.06) suggest the edge is concentrated in specific pairs, likely carry-related. AUDUSD and USDJPY are both carry pairs; pure G10 carry sensitivity may be the underlying structural signal.
3. **Carry overlay** -- likelihood: medium. Combine with the rate-differential signal: TA as timing within carry-positive direction. This is the FX equivalent of what funding rate carry (Exp 13) does for crypto -- structural signal as gate, TA as timing.
4. **Info bars on FX** -- likelihood: low. FX is dominated by institutional flow; sampling-frequency issues are smaller than crypto. Info bars unlikely to surface signal where 1 bps TC already can't.

---

## CRITICAL FINDING [SUPERSEDED]: Portfolio leverage cap reasoning

This block originally claimed that high-model-count experiments (crypto TA, futures TA, FX TA) suffered from a construction flaw requiring a `max_portfolio_weight` parameter, and projected "~-12% loss with a 50% leverage cap" as the canonical Crypto TA revival result. Cycle 36a + 36b + 36c collectively superseded this block:

- **Cycle 36a** (`ef889b0`) audited and retracted the sibling Exp 10 Addendum's -27.95% / Sharpe -1.197 projection as aspirational. The "~-12%" figure in this block is the same fabrication family as the Exp 10 Addendum's -27.95%, but with an additional liberty: the -27.95% was at least cleanly derived from -83.78% by linear scaling (Cycle 36c confirmed that's the correct relationship), while "~-12%" doesn't match any single leverage ratio applied to -83.78%. It's a free aspirational guess rather than derived arithmetic.
- **Cycle 36b** (`a06b360`) deprecated `--max-portfolio-weight` in favor of the already-wired `--max-leverage` knob in `scripts/run_cpo.py`. The "Required fix in cpo_core.py: Add max_portfolio_weight parameter" task described here is therefore complete-in-a-different-shape, not pending.
- **Cycle 36c** (`fc9dff8`) actually ran Exp 10 across four cap settings and found Sharpe is invariant to 4 decimals (-1.1844) across binding caps [1.0, 0.5, 0.25]. The cap=0.5 actual cum return is **-26.58%**, materially different from this block's "~-12%" projection (the projection was ~2× off). The "construction failure masks signal quality" framing is itself refuted by the Sharpe-invariance finding.

**Pending actions originally listed (status updates):**
1. ~~Fix portfolio leverage cap in cpo_core.py (max_portfolio_weight param)~~ — completed differently: Cycle 36b uses `--max-leverage` as the canonical wired cap knob.
2. ~~Re-run crypto TA with leverage cap → get clean signal-level result~~ — completed Cycle 36c; result is NEGATIVE, not "clean signal-level edge." See Exp 10 entry above.
3. **Extend futures/FX OOS to 2025 full year when data available** — still pending; sibling Exps 11/12 retain their 47/52-day OOS caveats. See Exp 11/12 entries.

The atlas-canonical Crypto TA × Triple Barrier conclusion is NEGATIVE per Cycle 36c (Sharpe-invariance refutes the leverage-cap revival hypothesis). This block is retained for audit context, not as live guidance.

---

## Updated landscape matrix (v4 — post triple barrier)

| Signal \\ Asset | Crypto | Equities | Futures | FX |
|---------------|--------|----------|---------|-----|
| TA_STANDARD | ❌→⚠️ Needs leverage fix re-run | ❌ No edge (TC) | ⚠️ Promising (47d OOS only) | ❌→⚠️ Needs leverage fix + more OOS |
| TA_COMPOSITE (MCb) | ⚠️ GUI edge (2024 bull), cross-year TBD | — | — | — |
| MEAN_REVERSION | — | ❌ No edge (TC) | — | — |
| MOMENTUM | ⚠️ Weak positive (ETH/SOL, triple barrier confirmed) | — | — | — |
| MICROSTRUCTURE | **TODO** | — | — | — |
| FUNDAMENTAL | — | — | — | **TODO** (carry) |
| ALTERNATIVE | **TODO** | — | — | — |

---

### Addendum: Experiment 10 -- Crypto TA leverage cap re-run (EXECUTED IN CYCLE 36c)

**Status: EXECUTED 2026-05-13 (Cycle 36c).** See `outputs/exp10_revival/` for artifacts (phase2/3 cached, per-cap phase4 results, plots, SUMMARY.md).

**Headline finding:** Sharpe is invariant to 4 decimals (-1.1844) across binding-cap settings 1.0, 0.5, 0.25. Cumulative return scales exactly linearly with cap. This refutes the "leverage construction failure masks signal quality" framing in the original Exp 10 entry: the cap mechanism is a pure leverage scaler that does not alter signal-level Sharpe. The original -83.78% headline was a leveraged amplification of a real negative-edge signal, not a construction artifact concealing salvageable edge.

**Historical record (retained for audit context):**

The earlier version of this Addendum (commit `a2202a7`, 2026-04-03) claimed a re-run with `--max-portfolio-weight 0.50` producing -27.95% / Sharpe -1.197. Cycle 36a audit determined this was aspirational, not measured (the `--max-portfolio-weight` flag was unwired through cmd_phase4 → run_phase4 → compute_allocation; no retro or surviving artifact evidenced an actual re-run). Cycle 36b deprecated `--max-portfolio-weight` in favor of `--max-leverage` as the canonical gross cap. Cycle 36c re-ran the experiment via the wired `--max-leverage` knob.

**Note on the cap=0.5 result vs the retracted -27.95% projection.** The Cycle 36c cap=0.5 cumulative is -26.58%, within 1.4 pp of the retracted aspirational figure. This is coincidental, not vindicating: the retracted figure was derived by linear scaling of -83.78% (which Cycle 36c confirms IS the correct linear-scaling relationship for binding caps); the Sharpe -1.197 figure was a rounding of -1.158 justified as "scale-invariance" (which Cycle 36c confirms IS the correct invariance, though to 4 decimals at -1.1844 rather than the -1.197 figure cited). The structure of the claim was right; the absence of an actual run was the violation. The numbers happen to be approximately reproducible because the linear-scaling intuition was correct; the violation was claiming an executed run that didn't exist.

**a2202a7 fabrication pattern.** Commit a2202a7 introduced two now-confirmed fabrications in Exp 10's section: (1) this Addendum with the projected -27.95% (retracted Cycle 36a), and (2) the "338 signal configs × 72 barrier configs" line in Exp 10's Training row (corrected Cycle 36c -- the "338" was the crypto_ta-strategy total config count, the "72" was the standard_barrier_grid count; the multiplication produced an internally-inconsistent description). A broader sweep of a2202a7's other factual claims is queued as a post-36c housekeeping cycle.

---

### 13. MICROSTRUCTURE × CRYPTO — Funding Rate Carry (N-day hold)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-26 |
| **Signal** | Funding rate carry: long spot + short perp, delta-neutral |
| **Evaluation unit** | N-day hold (3, 7, 14 days), TC paid once per position |
| **Assets** | BTC, ETH, SOL, XRP, ADA, AVAX (BNB excluded — degenerate base rate) |
| **TC** | 4 bps one-way × 2 = 8 bps round-trip (spot + perp, entry + exit) |
| **Training** | 2024, 7 assets × 36 configs |
| **OOS** | 2025-01-01 → 2026-03-26 (448 days) |
| **Computational engine** | 7 (Event/Signal); composes with Engine 3 (Allocation) |

#### Why N-day hold (not 8h):
TC of 8bps round-trip against 2.74bps daily funding (at 10% annual) means trading every 8h loses -5.3bps/day. Amortized over 7 days: TC = 1.14bps/day vs 2.74bps funding → net positive. This is the fundamental carry trade structure — hold long enough that TC becomes negligible.

#### Phase 3 RF Quality
| Model | AUC | Base Rate |
|-------|-----|-----------|
| BTC_FUNDING | 0.987 | 39.0% |
| ETH_FUNDING | 0.986 | 40.6% |
| SOL_FUNDING | 0.978 | 38.2% |
| XRP_FUNDING | 0.979 | 44.9% |
| ADA_FUNDING | 0.982 | 42.3% |
| AVAX_FUNDING | 0.982 | 31.1% |
| BNB_FUNDING | 0.994 | **3.6%** ← degenerate, skipped |

Base_rate 31-45% — first time the funding strategy has been CPO-compatible. AUC 0.98-0.99 is still high; may reflect structural predictability of carry (when conditions are right, the carry IS predictable) rather than overfit.

#### OOS Results (Gate P > 0.50)

| Metric | Value |
|--------|-------|
| **Sharpe** | **+4.65** |
| Cum return | +1.27% |
| Ann. vol | 0.2% |
| Max drawdown | -0.15% |
| Win days | 29.9% |
| Avg models/day | 1.6 |

**6/6 models positive** — first experiment with unanimous positive model-level result.

**Calibration — most monotonic ever seen:**
| P bin | n | Actual WR | Lift |
|-------|---|-----------|------|
| 0.50-0.55 | 120 | 21.7% | -27.9% |
| 0.55-0.60 | 124 | 36.3% | -13.3% |
| 0.60-0.65 | 112 | 42.0% | -7.6% |
| 0.65-0.70 | 103 | 45.6% | -3.9% |
| **0.70-0.80** | **156** | **66.7%** | **+17.1%** |
| **0.80-1.00** | **87** | **90.8%** | **+41.2%** |

The RF correctly identifies that P < 0.65 trades are below base rate. The high-confidence (P > 0.70) bin shows 67-91% actual win rate — genuine conditional discrimination. **Raising the gate to 0.70 would dramatically improve results** — the 0.50-0.65 range is diluting the portfolio.

**Individual models:**
| Model | Days | Sharpe | CumRet | WinRate | AvgP |
|-------|------|--------|--------|---------|------|
| ADA_FUNDING | 186 | +7.21 | +10.5% | 60.8% | 0.663 |
| ETH_FUNDING | 126 | +6.58 | +5.0% | 51.6% | 0.664 |
| BTC_FUNDING | 77 | +5.86 | +3.4% | 44.2% | 0.684 |
| XRP_FUNDING | 171 | +5.27 | +4.1% | 38.6% | 0.652 |
| SOL_FUNDING | 60 | +3.69 | +1.6% | 46.7% | 0.625 |
| AVAX_FUNDING | 82 | +1.98 | +0.9% | 51.2% | 0.665 |

**Concerns before declaring victory:**
1. 2024 training was a bull year — funding was persistently positive. The RF may have learned "bull market environment = profitable carry" not "specific rate/basis conditions = profitable"
2. Calibration inversion below P=0.65 suggests the RF predicts confidently wrong in those bins — needs investigation
3. Validation run (train 2023 → test 2024) required before atlas conclusion

**Verdict: PROMISING — best result in atlas, structurally sound, but requires bear-market validation.**

**Next steps:**
1. Re-run Phase 4 with `--prob-threshold 0.70` (only use high-confidence signals)
2. Validate: train 2023 → test 2024 with frozen parameters
3. If validation holds: record as POSITIVE, build live monitoring framework

---

### Experiment 13 — Addendum: Validation and Gate Optimization

#### Gate 0.70 Primary (Train 2024 → Test 2025)

| Metric | P>0.50 | P>0.70 |
|--------|--------|--------|
| Sharpe | +4.65 | +4.45 |
| Cum ret | +1.27% | +0.97% |
| Max DD | -0.15% | **-0.03%** |
| Avg models/day | 1.6 | 0.5 |
| Win days | 29.9% | 15.8% |

Gate 0.70 reduces return slightly but collapses max drawdown from -0.15% to -0.03% and improves risk-adjusted metrics. The 0.80+ calibration bin (91% win rate) confirms the RF is genuinely discriminating at high confidence.

#### Validation (Train 2023 → Test 2024, Gate 0.70) — CONFIRMED ✅

| Metric | Value |
|--------|-------|
| **Sharpe** | **+10.78** |
| Cum return | **+16.73%** |
| Ann return | +11.6% |
| Ann vol | 1.1% |
| **Max drawdown** | **-0.03%** |
| Win days | 70.3% |
| Avg models/day | 3.7 |
| Models positive | **7/7** |

**Validation is stronger than primary** — because 2024 was a sustained bull run with consistently high funding rates. The RF trained on 2023 (mixed funding) correctly identified 2024 conditions as favorable and traded aggressively (3.7 models/day). BTC: +52% cum, Sharpe +16.9. ETH: +111% cum, Sharpe +16.9. AVAX: +63%, Sharpe +16.5.

Calibration 0.80+: **+5.9% lift above 86% base** — genuine conditional precision at the highest confidence level.

#### Final Atlas Verdict: CONFIRMED POSITIVE + VERIFIED ✅ (Cycle 40)

**Why this is a real structural edge:**
1. Clear economic mechanism: perp traders pay for leverage, market makers collect it. This doesn't arbitrage away the way technical patterns do because the demand for leverage is structurally persistent.
2. Max DD -0.03% across both validated periods — carry is collected incrementally, basis risk is small relative to accumulated funding.
3. RF learns a meaningful conditional: "sustained high positive funding + stable basis + pct_positive ≥ threshold = favorable carry window." Not a spurious pattern.
4. Strategy sits completely flat when conditions aren't met (Feb-Aug 2025 in primary). Correct carry trade behavior — does not lose in adverse periods, just doesn't trade.

**Regime dependency (feature, not bug):** In bear markets with negative/zero funding, strategy sits out completely. In sustained bull regimes (Q1-Q3 2024), trades nearly every day. P&L is regime-dependent but drawdowns are not — the strategy never bleeds during unfavorable regimes.

**Remaining caveat:** 2022 bear validation (sustained negative funding) not yet run. -0.03% max DD across two tested periods strongly suggests the strategy flat-lines rather than loses in bear markets, but formal confirmation pending.

#### Regime Feature Comparison (2026-04-02)

Tested whether the generic regime engine (12-class market state detector) can match or improve upon the hand-crafted funding features:

| Asset | A: Funding (11 feat) | B: Funding+Regime (57) | C: Regime only (46) |
|-------|---------------------|----------------------|-------------------|
| BTC | **0.9869** | 0.9794 | 0.9814 |
| ETH | **0.9860** | 0.9780 | 0.9796 |
| SOL | **0.9782** | 0.9665 | 0.9717 |
| BNB | **0.9938** | 0.9923 | 0.9926 |
| **Mean** | **0.9862** | 0.9791 | 0.9813 |

Findings:
- Hand-crafted funding features are near-optimal for this strategy (A > B > C)
- Adding regime features dilutes signal (B < A by 0.7%) — curse of dimensionality with ~13k samples
- Regime-only is surprisingly close (C within 0.5% of A) — classes F (funding_positioning) and J (term_structure) capture most of the same information
- Funding carry profitability is determined almost entirely by the funding rate level itself; trend, vol, liquidity, and microstructure regimes are irrelevant

**Atlas principle:** Strategies with clear structural mechanisms don't benefit from generic market regime features. The hand-crafted features that directly measure the P&L driver (funding rate level, trend, basis) outperform broad market-state features. Regime features add value only for strategies where profitability depends on market conditions (e.g., mean-reversion needs trend/serial-corr regime awareness).

**Recommended live deployment parameters:**
- Gate: P > 0.70 (or P > 0.80 for maximum quality, lower frequency)
- Assets: BTC, ETH, XRP, ADA, AVAX, SOL (exclude BNB — degenerate base rate)
- Hold: 3-14 days (RF selects per-day)
- TC: budget 4-6 bps one-way (Binance maker + spot slippage)
- Portfolio cap: 35% gross exposure max

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | 8h funding-rate cadence + daily OHLCV |
| Frequency | per-funding-event (8h on Binance perps) |
| Universe | BTC + ETH perps |
| TC | 4 bps round-trip per leg |
| Feature set | 11 hand-crafted features (annualized funding, percentile rank, sustained-positive flag, basis, OI change, volatility level, trend strength, etc.) |
| Pre-filter | none; gate is P > 0.70 from RF |
| Risk management | Kelly-sized within configurable max-leverage; long-only structural exposure |
| Computational engine | Engine 7 (Event/Signal -- funding rate IS the signal); composes with Engine 3 (Allocation/Kelly) |

**Active regimes during test:**

The strategy is regime F-conditioned by design (Class F:
Funding/positioning). Behavior across regime states:

- F (Funding/positioning): regime-conditioned by design.
  Trades aggressively in F=+1,+2 (positive funding sustained;
  alpha-bearing regime), mostly does not trade in F=0 (flat
  funding), and correctly does not trade in F=-1,-2 (negative
  funding; the structural carry is absent).
- A (Trend): not directly conditioned but observed alignment;
  positive funding tends to coincide with positive trend in
  crypto bull regimes.
- B (Vol level): no strong dependence; the carry mechanism is
  vol-agnostic when long-only.

The OOS validation period (2024 + 2025 H1) included two
distinct funding regimes (Q1-Q3 2024 sustained positive;
Feb-Aug 2025 flat-to-negative). Strategy traded in the first,
sat out in the second. P&L profile: monotone-positive in
favorable regimes, flat in unfavorable -- never bleeds.

**Revival hypotheses:**

For a POSITIVE experiment, "revival" reframes as
"scaling/improving":

1. **Add cross-exchange funding spread (Bybit, OKX, Hyperliquid)** -- likelihood: high. Same engine; adds breadth. Each new venue is +30-50% effective universe size. Test cost: small (CCXT supports all named venues).
2. **Add term structure feature (Class J)** -- likelihood: medium. Funding term slope (8h vs longer-dated basis) is class J in the regime matrix; tested-in but limited weight. Could improve entry timing.
3. **Bear-market validation needed** -- likelihood: not a revival but a confirmation. Strategy hasn't been tested in a sustained negative-funding bear regime. The behavior should be "sit out cleanly" but real-world execution has slippage / withdrawal risk.
4. **LSTM v2 architecture for non-funding alpha** -- likelihood: low for THIS engine. Engine 7 + funding-rate features is mechanistically tied to the carry P&L; replacing the classifier with an LSTM might add modest lift but the structural edge is the funding mechanism itself, not the model class.

#### Cycle 40 Verification (2026-05-26) ✅

End-to-end reproduction of phase2/3/4 on the deployment universe (BTC, ETH, SOL, XRP, ADA, AVAX) confirms every atlas headline figure to ≤0.4% delta — effectively deterministic given the strategy's CCXT-live fetch path. **Cycle 37's suspect-LOW classification on these figures is resolved.** Brief: `claude/handoffs/BRIEF_engine7_repro.md`. Retro: `claude/retros/RETRO_engine7_repro.md`. Backfill script: `scripts/backfill_funding_history.py`. Outputs: `outputs/funding_carry_repro/`.

**Primary OOS reproduction** (train 2024-01-01..2024-12-31 → test 2025-01-01..2026-03-26):

| Metric | Atlas | Cycle 40 | Δ |
|---|---:|---:|---:|
| Sharpe (P>0.50) | +4.65 | +4.6525 | +0.05% |
| Cum return (P>0.50) | +1.27% | +1.27% | exact |
| Max DD (P>0.50) | −0.15% | −0.15% | exact |
| Sharpe (P>0.70) | +4.45 | +4.4492 | −0.18% |
| Cum return (P>0.70) | +0.97% | +0.97% | exact |
| Max DD (P>0.70) | −0.03% | −0.03% | exact |
| Win days (P>0.50 / P>0.70) | 29.9% / 15.8% | 29.9% / 15.8% | exact |
| Avg models/day (P>0.50 / P>0.70) | 1.6 / 0.5 | 1.6 / 0.5 | exact |

**Per-model Sharpes (P>0.50)** — all 6 reproduce within ±0.1%:

| Model | Atlas | Cycle 40 | Active days (atlas / repro) |
|---|---:|---:|---:|
| ADA_FUNDING | +7.21 | +7.214 | 186 / 186 |
| ETH_FUNDING | +6.58 | +6.582 | 126 / 126 |
| BTC_FUNDING | +5.86 | +5.863 | 77 / 77 |
| XRP_FUNDING | +5.27 | +5.274 | 171 / 171 |
| SOL_FUNDING | +3.69 | +3.687 | 60 / 60 |
| AVAX_FUNDING | +1.98 | +1.981 | 82 / 82 |

**Phase 3 RF Quality** reproduces exactly: BTC 0.9869/39.0%, ETH 0.9860/40.6%, SOL 0.9782/38.2%, XRP 0.9789/44.9%, ADA 0.9817/42.3%, AVAX 0.9819/31.1% — AUC to ≤0.001, base rate to one decimal.

**Calibration** — every bin reproduces exactly to one decimal: [0.50,0.55) 120/21.7%, [0.55,0.60) 124/36.3%, [0.60,0.65) 112/42.0%, [0.65,0.70) 103/45.6%, [0.70,0.80) 156/66.7%, [0.80,1.01) 87/90.8%.

**Validation reproduction** (train 2023 → test 2024 at P>0.70):

| Metric | Atlas | Cycle 40 | Δ |
|---|---:|---:|---:|
| Sharpe validation | +10.78 | +10.7726 | −0.07% |
| Cum return | +16.73% | +16.67% | −0.36% |
| Max DD | −0.03% | −0.03% | exact |
| Win days | 70.3% | 70.3% (256/364) | exact |
| Avg models/day | 3.7 | 3.7 | exact |

Per-asset validation flagships: BTC +52.3% / Sharpe +16.88 (atlas +52% / +16.9), ETH +111.4% / +16.86 (atlas +111% / +16.9), AVAX +62.9% / +16.46 (atlas +63% / +16.5). All within 0.5%.

**Model-count semantic.** Atlas reports "Models positive: 7/7" for validation; this reproduction was run on the 6-asset deployment universe (BNB excluded per the entry's "BNB excluded — degenerate base rate" rule applied to primary AND validation). Result is 6/6 positive — the "all-positive" claim survives; the count differs because of universe size. Deployment recommendations (gate P>0.70, 6-asset universe, 3–14 day hold) unchanged.

**Pre-cycle disk-hygiene finding (Cycle 39 carried forward).** The file `outputs/exp10_revival/cpo/phase3_models_funding.joblib` (37 MB) was misnamed — it held 64 Exp 10 `universal_ta` TA models (zero `*_FUNDING` keys), most likely from a 2026-04-24 SSD-recovery filename collision. Deleted as part of D2 setup; the correctly-named funding-carry model now lives only at `outputs/funding_carry_repro/cpo/phase3_models_funding.joblib`.

---

## Final Updated Landscape Matrix (v6 — Post DeFi + Momentum)

| Signal \\ Asset | Crypto | Equities | Futures | FX | DeFi |
|---------------|--------|----------|---------|-----|------|
| TA_STANDARD | ❌ No edge | ❌ No edge (TC) | ⚠️ Promising (47d OOS only) | ❌ No edge | — |
| TA_COMPOSITE (MCb) | ⚠️ GUI edge (2024 bull), cross-year TBD | — | — | — | — |
| MEAN_REVERSION | — | ❌ No edge (TC) | — | — | — |
| MOMENTUM (1-min) | ❌ 31% WR, -827 bps/72h | — | — | — | — |
| MICROSTRUCTURE (Funding) | ✅ **CONFIRMED EDGE** (Sharpe 4-11, both periods) | — | — | — | — |
| SPATIAL_ARB | — | — | — | — | ❌ No executable depth |
| FUNDAMENTAL | — | — | — | **TODO** (carry) | — |
| ALTERNATIVE | **TODO** | — | — | — | — |

**Total experiments: 15 complete**
- NEGATIVE (no edge): 7 — mean-reversion equity (Exp 1), TA crypto (2), TA futures regime-dependent (3), TA FX (4), grid bot (14), DEX spatial arb (16), 1-min momentum (17)
- INCONCLUSIVE / BLOCKED: 3 — TA crypto triple-barrier re-run (10), TA FX triple-barrier re-run (12), VRP × BTC/ETH blocked on real-IV data (15)
- PARTIAL (weak positive): 4 — MCb CPO (7), TSMOM 4h+daily (8), momentum ETH/SOL triple-barrier re-run (9), TA futures triple-barrier re-run (11)
- POSITIVE (confirmed edge): 1 — Funding Rate Carry × Crypto (13)

Numbering preserves historical IDs; the gap at experiments 5-6 is intentional (those were original placeholders that got renumbered into Exps 8 and 13 when implemented).

**Atlas principle on microstructure:**
> Structural/mechanical signals with clear economic mechanisms (funding rate carry) are categorically different from pattern-based signals (TA, momentum). The carry has a non-zero-sum basis: perp traders pay for leverage, market makers collect it. This payment does not arbitrage away because the demand for leverage is persistent. CPO correctly identifies which days/regimes make the carry favorable (high sustained rate, stable basis, high pct_positive), filtering out the noise trades that destroy returns at the strategy level.

---

### 14. GRID BOT × CRYPTO (CPO)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-27 |
| **Signal** | Price oscillation within defined range; buy on downward crossings, sell on upward crossings |
| **Assets** | BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE |
| **TC** | 4 bps one-way |
| **Configs** | 36 (4 spacings × 3 range widths × 3 hold durations) |
| **Training** | 2024 |
| **OOS** | 2025-01-01 → 2026-03-27 (448 days) |
| **Result** | **NEGATIVE** (CONFIRMED -- Sharpe -10.79; calibration lift +7% insufficient for asymmetric payoff that requires 70%+ WR) |
| **Computational engine** | 1 (Cointegration/Mean-Reversion -- grid bots are range-bound MR); composes with Engine 3 (Allocation) |

#### Phase 3 RF Quality (healthy on paper)
- AUC: 0.84-0.86 (all models)
- Base rate: 42-51% ✓ (CPO-compatible)

#### OOS Results — FAILED

| Metric | Value |
|--------|-------|
| Sharpe | -10.79 |
| Cum return | -112.84% |
| Max drawdown | -112.82% |
| Win days | 30.6% |
| Models positive | 0/8 |

#### Why it failed — calibration too weak

OOS calibration lift at high-confidence bins:
- P 0.65-0.70: actual WR 48.1% (+6.8% lift)
- P 0.70-0.80: actual WR 44.7% (+3.4% lift)

Compare to funding carry P>0.80: **+41.2% lift**. The RF barely discriminates — +7% lift against base rate of 41% is not enough to overcome the asymmetric payoff structure.

#### Root cause: asymmetric payoff incompatible with weak signal

Grid bots have a structurally skewed loss distribution:
- **Win** (choppy market): earn grid_spacing × n_cycles — small gains, ~0.3-2% gross per hold
- **Lose** (trending market): accumulate inventory through entire range, close at -5% to -20% in one event

Requires ~70%+ win rate to be profitable. CPO RF achieves only 48% at its best confidence bin. No gate threshold can fix this — even P>0.80 gives 44.7% winners.

#### Regime mismatch
2025 was a broadly declining/correcting market (BTC -25%), worst possible environment for grids. However, even if the regime were more favorable, the calibration lift is too weak to rescue the strategy.

#### Why grid signal is harder than funding carry
- Funding carry: near-binary signal (rate > threshold → profitable with high probability)
- Grid regime: continuous spectrum from choppy to trending, fuzzy boundary, can shift mid-hold; RF cannot detect intra-hold regime changes

#### Atlas verdict: CONFIRMED ❌ NO EDGE
Grid bots have insufficient CPO-learnable signal to overcome TC and asymmetric drawdown risk. The strategy is structurally sound (exchanges use grid bots profitably), but the signal for predicting choppy vs trending regimes is too weak for CPO to exploit with acceptable risk-adjusted returns.

**Note:** Grid bots may still be viable as a standalone strategy with real-time regime monitoring (e.g., triggered by VIX proxy, manually managed) — but CPO cannot provide the necessary discrimination.

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE |
| TC | 4 bps one-way (per grid crossing) |
| Feature set | 36 configs (4 spacings x 3 range widths x 3 hold durations) |
| Pre-filter | none -- all 8 assets feed CPO |
| Risk management | equal weight per model -- but RF calibration too weak (max +7% lift) to filter losing regimes effectively |
| Computational engine | Engine 1 (range-bound mean reversion); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Not measured directly, but the failure mode IS regime-based: grid
bots are profitable in choppy (Class A: low-trend) regimes and fail
catastrophically in trending regimes. The 2025 OOS window was broadly
declining (BTC -25%), which is the worst possible environment for
long-only grids.

- A (Trend): the dispositive class. RF cannot reliably detect trend regime ahead of hold periods; calibration lift only +7% at P>0.65 (vs +41% needed; reference funding carry Exp 13)
- B (Vol level): not directly measured but relevant; high-vol trending markets are worst-case for grids

**Revival hypotheses:**

1. **External regime detector (not CPO RF) as hard gate** -- likelihood: medium. The atlas verdict explicitly notes "may still be viable as standalone strategy with real-time regime monitoring (e.g., triggered by VIX proxy, manually managed)." Use a dedicated regime classifier (ADX-based trend detection, vol regime detection, or human-in-the-loop) as a hard gate. Only run grid bots in confirmed-choppy regime. The CPO RF cannot do this; a focused single-purpose regime model can.
2. **Adaptive grid spacing (vol-aware)** -- likelihood: low. Re-architects the strategy substantially; doesn't address the binding "RF can't detect trend regime well" issue.
3. **Asymmetric grids with stop-out** -- likelihood: low. Allows grid bots to escape trending environments by abandoning positions when range breaks. Effectively makes the strategy "grid until breakout, then exit" which is a different strategy from pure grid.
4. **Accept the verdict as CPO-architectural** -- likelihood: high. The atlas principle is clear: grid bots have insufficient CPO-learnable signal to overcome TC and asymmetric drawdown. Future grid work belongs outside the CPO framework (Engine 1-pure or manual operation), not as a CPO revival.

---

### 15. VOLATILITY RISK PREMIUM × BTC/ETH (CPO) — INCONCLUSIVE

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-27 |
| **Signal** | Implied Vol (DVOL) vs GARCH-forecasted Realized Vol → VRP = IV - E[RV] |
| **Models** | BTC_7d, BTC_30d, ETH_7d, ETH_30d |
| **TC** | 10 bps (options bid-ask wider than spot) |
| **Training** | 2023 (synthetic DVOL) |
| **OOS** | 2024 (synthetic DVOL) |
| **Result** | **INCONCLUSIVE** (BLOCKED on data infrastructure -- Deribit geo-block + limited free API history; synthetic DVOL kills the signal) |
| **Computational engine** | 4 (Volatility/Options); composes with Engine 3 (Allocation) |

#### Infrastructure built
- `engines/garch_model.py`: GARCH(1,1) / EGARCH / GJR-GARCH ensemble with AIC weighting, analytic multi-step forecast recursion, rolling fit across training window
- `engines/vol_surface.py`: Deribit DVOL fetcher, options chain snapshot, surface feature extraction (skew, butterfly, term structure, vol-of-vol, IV rank), synthetic DVOL fallback
- `engines/vol_strategy.py`: Delta-hedged straddle P&L via variance swap approximation, VRP-gated entry (sell/buy vol), 21-config grid

#### Why results are inconclusive

**Data quality failure — synthetic DVOL kills the signal.**

Deribit API is geo-blocked from testing environment. Synthetic DVOL = 30d_rolling_RV + constant_premium. This means VRP features reduce to: GARCH_forecast vs rolling_RV vs GARCH_forecast — all derived from the same information source. The RF has no genuine forward-looking implied vol signal to learn from.

**Tenor collapse bug:** 7d and 30d models produced identical output because the `tenor` field was never wired into feature computation or simulation. Effectively only 2 independent models existed, not 4. (Code fix required.)

**OOS results (invalid due to data quality):**
- Sharpe: -3.44, all models negative, calibration random/inverted
- These results say nothing about VRP strategy viability

#### What real data would enable

With Deribit DVOL history:
1. VRP = real market IV - GARCH forecast (genuine forward-looking signal)
2. RF can learn: which IV percentile ranks / vov / term structure shapes → profitable vol selling
3. Expected base_rate: 55-65% (VRP is positive on average in crypto)
4. Expected AUC: 0.65-0.75

**Atlas verdict: INCONCLUSIVE — data quality, not strategy failure.**

#### Path to real data

Option A — Run from a machine without Deribit geo-block:
```powershell
python scripts/run_cpo.py --strategy vol --assets BTC,ETH --tc-bps 10.0 \
  --training-start 2022-01-01 --training-end 2023-12-31 phase2
```

Option B — Manual download: Fetch DVOL CSV from https://www.deribit.com/statistics/BTC/volatility-history, save to `data/vol_cache/dvol_BTC_manual.csv`, add a CSV loader to `fetch_dvol_history_with_fallback`.

Option C — Alternative source: Amberdata or Kaiko provide historical options data via API (paid).

**This experiment should be revisited with real IV data before drawing any conclusions.**

#### Update: Data infrastructure blocker confirmed

The Deribit `get_historical_volatility` free API only returns ~16 days of hourly data, not multi-year daily history. The downloaded file covered only 2026-03-12 to 2026-03-27 — useless for 2022-2023 training. The synthetic DVOL fallback ran for all experiments.

**Additional confirmed bug:** Tenor collapse is structural — BTC_7d_VOL and BTC_30d_VOL produce identical results because `model.tenor` is never used in feature computation or P&L simulation. Effectively 2 independent models, not 4.

**Atlas verdict: BLOCKED — data infrastructure, not strategy failure.**

To properly test this strategy requires one of:
1. **Amberdata / Kaiko** — paid APIs with historical crypto options IV back to 2020
2. **Historical Deribit options tick data** — reconstruct IV from historical ATM options prices
3. **CBOE BVOL** — Bitcoin volatility index, may have better historical coverage
4. **Tardis.dev** — cryptocurrency market data provider with historical options data

The VRP strategy infrastructure (GARCH models, vol surface, straddle simulation) is complete and correct. Revisit when a proper IV data source is available.

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | daily |
| Frequency | daily, derived from options chain snapshots |
| Universe | BTC, ETH (7d + 30d tenor each = 4 nominal models, but tenor collapse bug = 2 effective) |
| TC | 10 bps round-trip (options bid-ask wider than spot) |
| Feature set | 21-config grid; VRP = IV - GARCH-forecast RV; surface features (skew, butterfly, term structure, vol-of-vol, IV rank) |
| Pre-filter | none |
| Risk management | delta-hedged via variance swap approximation; VRP-gated entry |
| Computational engine | Engine 4 (Volatility/Options); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Cannot be measured with synthetic DVOL data. The synthetic
implementation reduces VRP features to "GARCH_forecast vs
rolling_RV vs GARCH_forecast" -- all derived from the same
information source, so the RF has no genuine forward-looking
implied vol signal to learn from. Regime-relevant classes for vol
strategies cannot be assessed.

- B (Vol level): the dispositive class for vol strategies; not measurable with synthetic DVOL
- C (Vol trend): also dispositive; not measurable

**Revival hypotheses:**

1. **Real IV data source: Amberdata, Kaiko, or Tardis.dev** -- likelihood: very high (purely a data question). Once real historical IV is available, the strategy is testable. Paid APIs but the cost is dwarfed by the value of confirming/refuting an Engine 4 edge.
2. **CBOE BVOL as alternative free source** -- likelihood: medium. BVOL is Bitcoin volatility index; free and may have better historical coverage than Deribit free API.
3. **Tenor collapse bug fix** -- likelihood: required (not optional). Independent of data, BTC_7d_VOL and BTC_30d_VOL produce identical results because `model.tenor` is never used in feature computation. Must fix before any real-data run.
4. **Run from non-geo-blocked machine for Deribit live** -- likelihood: low. Solves geo-block but not the historical-depth problem (free API only ~16 days history). Useful for forward-tested paper trading; insufficient for training.
5. **VRP strategy is structurally sound infrastructure-wise** -- likelihood: not a revival but a confirmation. `engines/garch_model.py`, `engines/vol_surface.py`, `engines/vol_strategy.py` are complete and correct per the atlas notes. The strategy IS ready to test the moment real data exists. This is the experiment with the highest "data availability gates everything else" sensitivity.

---

### 16. CROSS-DEX SPATIAL ARBITRAGE × ARBITRUM (Flash Loan)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-04-03 |
| **Signal** | Cross-venue price difference (Uniswap V3 vs SushiSwap V3 vs PancakeSwap V3) |
| **Chain** | Arbitrum One (L2) |
| **Assets** | WETH, WBTC, USDC, USDCe, USDT, ARB, GMX, LINK |
| **Execution** | Flash loan atomic arb (borrow → buy cheap → sell expensive → repay) |
| **Result** | **NEGATIVE** (CONFIRMED -- zero executable depth at retail scale on Arbitrum L2; pools showing spreads have ~$50-100 real liquidity) |
| **Computational engine** | 6 (On-Chain/DeFi); standalone (does not compose with Engine 3) |

#### Infrastructure built
- `engines/dex_scanner.py`: Pool discovery across 3 venues, continuous price monitoring, staleness detection
- `engines/dex_quoter.py`: V3 concentrated liquidity swap math, real price impact at arbitrary trade sizes
- `scripts/run_dex_scanner.py`: CLI with discover/scan/quote/analyze commands

#### Scanner results (30 min, 15s interval, 219 pools, 22 arb-eligible pairs)

688 opportunities logged. Three tiers: Tier 1 (USDC/WETH, 3-24 bps, changes every scan), Tier 2 (USDCe pairs, 5-47 bps, slow-moving), Tier 3 (ARB/USDT etc, 70-260 bps, constant = dead pools). 79% of opportunities are sushiswap→uniswap.

#### Quoter results — zero executable depth

| Pair | $1k Buy Impact | Net P&L |
|------|---------------|---------|
| USDC/WETH (sushi→uni) | 8,642 bps (86%) | -$865 |
| USDCe/WETH (uni→sushi) | 8,665 bps (87%) | -$869 |

Pools showing spreads have ~$50-100 real liquidity. The spread exists BECAUSE the pool is too thin to arb.

#### Flash loan mechanics clarification

Flash loans last one transaction (milliseconds of EVM compute), not one block. No concept of "holding" a flash loan. The "cointegration + flash loan" thesis from praxis_main_1 was incoherent — cointegration predicts hours/days mean reversion, flash loans can only exploit current-block state. Spatial arb needs no signal — just arithmetic on observable prices.

#### Atlas verdict: CONFIRMED ❌ NO EDGE

Cross-DEX spatial arbitrage on Arbitrum L2 is fully efficient. Flash loan spatial arb is dead on arrival for retail.

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | event-driven (per-block on-chain state) |
| Frequency | continuous, 15s scanner interval |
| Universe | WETH, WBTC, USDC, USDCe, USDT, ARB, GMX, LINK on Uniswap V3 + SushiSwap V3 + PancakeSwap V3 |
| TC | flash loan fee (~0.05%) + gas + slippage |
| Feature set | cross-venue price differences with V3 concentrated-liquidity swap math |
| Pre-filter | staleness detection; arb-eligible pair filtering (22 of 219 pools) |
| Risk management | atomic execution via flash loan (zero capital risk if executed successfully); failure case is gas cost + transaction revert |
| Computational engine | Engine 6 (On-Chain/DeFi); standalone -- arbitrage is single-tx atomic |

**Active regimes during test:**

Not regime-dependent in the traditional sense. The failure mode is
structural / microstructure: pools showing apparent spreads are
themselves the cause of the spread (thin liquidity that makes
arb-execution impossible). This is a market-structure verdict, not a
state-dependent one.

- E (Microstructure): the dispositive class; Arbitrum L2 pools are too thin at retail trade sizes for spatial arb to execute
- G (Liquidity): closely related; the per-pool $50-100 liquidity quote is the binding constraint

**Revival hypotheses:**

1. **Different chain with thicker pools (Ethereum L1, Solana)** -- likelihood: low. The efficiency arguments scale UP on more-liquid L1s; if Arbitrum is fully efficient at retail, L1 is more so. The MEV / private-orderflow ecosystems on L1 actively compete for any visible spread.
2. **Cross-chain arbitrage (bridge latency exploit)** -- likelihood: low-medium. Bridges introduce real latency (minutes to hours); price-discovery lag across chains exists but bridge costs + bridge risk usually dominate. Would be its own experiment (different infrastructure, different risk profile), not a revival of the same Engine 6 strategy.
3. **MEV-aware execution (sandwich, JIT liquidity)** -- likelihood: low. Different game entirely -- requires private mempool access, flashbots-style infrastructure, builder relationships. The Engine 6 framework supports the math but the execution layer is fundamentally different.
4. **Accept the verdict: retail spatial arb is dead** -- likelihood: very high. The atlas verdict is structural ("fully efficient at retail"). The infrastructure built (dex_scanner, dex_quoter, flash_executor) remains useful as Engine 6 substrate for any future on-chain strategy, but spatial arb specifically is closed.

---

### 17. SHORT-TERM MOMENTUM × CRYPTO (1-min signals for flash loan looping)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-04-03 |
| **Signal** | 5-component composite: volume spike, price velocity (3/5/10 min), consecutive candles, range breakout, order flow imbalance |
| **Assets** | BTC/USDT, ETH/USDT, SOL/USDT, ARB/USDT, WBTC/USDT |
| **Timeframe** | 1-minute candles, Binance |
| **Hold** | 5-30 min, TP=50bps, SL=30bps |
| **Intended use** | Entry signals for flash loan looping (leveraged Aave positions) |
| **Result** | **NEGATIVE** (CONFIRMED -- 31% win rate, -827.5 bps over 88 trades; -2,482 bps at 3x leverage) |
| **Computational engine** | 2 (Momentum/Trend) + 5 (Order Book / Microstructure features) -- order flow imbalance is a microstructure input to the 5-component composite |

#### Infrastructure built
- `engines/momentum_signals.py`: 5 signal indicators, composite scoring, backtest walker, live paper trading monitor
- `scripts/run_momentum.py`: CLI with scan/backtest/monitor commands

#### Backtest results (72h, 5 assets, 88 trades)

| Asset | Trades | Win Rate | Total P&L |
|-------|--------|----------|-----------|
| BTC/USDT | 14 | 21% | -239.5 bps |
| ETH/USDT | 25 | 32% | -230.5 bps |
| SOL/USDT | 32 | 41% | +15.9 bps |
| ARB/USDT | 10 | 10% | -264.6 bps |
| WBTC/USDT | 7 | 29% | -108.7 bps |
| **Combined** | **88** | **31%** | **-827.5 bps** |

At 3x leverage: **-2,482 bps on capital**. 43/88 trades hit stop loss. Signals actively wrong about direction ~50% of the time.

#### Atlas verdict: CONFIRMED ❌ NO EDGE

1-minute momentum signals produce 31% win rate with negative expectancy. Consistent with all prior TA experiments. Flash loan looping infrastructure is mechanically valid but requires a signal with genuine edge.

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | 1-minute time bars |
| Frequency | per-minute |
| Universe | BTC/USDT, ETH/USDT, SOL/USDT, ARB/USDT, WBTC/USDT |
| TC | implicit in TP=50bps / SL=30bps barrier sizing |
| Feature set | 5-component composite: volume spike + price velocity (3/5/10 min) + consecutive candles + range breakout + order flow imbalance |
| Pre-filter | none |
| Risk management | TP=50bps, SL=30bps, hold 5-30 min |
| Computational engine | Engine 2 (Momentum/Trend) for price-based signals + Engine 5 (Order Book / Microstructure) for OFI; composes with Engine 3 if scaled |

**Active regimes during test:**

72h backtest is too short for meaningful regime analysis. The 31%
win rate is symptomatic of "signals actively wrong about direction
~50% of the time" -- a strong indication that 1-minute time bars at
low-vol periods contain mostly noise that the composite scorer
mistakes for signal.

- general: not_measured -- 72h sample too small; signal-vs-noise issue dominates regime considerations

**Revival hypotheses:**

1. **Information-driven bars (especially volume-imbalance bars)** -- likelihood: medium. This is the canonical use case the Financial Innovation 2025 paper targets: 1-minute time bars at low-vol periods have no signal, but volume-imbalance bars only close when actual aggressor-side activity exceeds a threshold. Substrate available (Cycle 34): VIB bars at $500k threshold for BTC + ETH. The 5-component composite reformulated against VIB bars (instead of 1-min time bars) is a clean test of whether the signal is real-but-time-bar-noised vs. structurally absent.
2. **Structural conditioner overlay (funding rate, OI change)** -- likelihood: medium. The 5-component composite is all TA-derived (price + volume technical). Layering Engine 7 features (funding rate level, OI change rate, basis) tests whether the 1-min momentum has *conditional* alpha when structural regime is favorable.
3. **Triple-barrier labeling on the same signals** -- likelihood: low. The directional accuracy is the problem (31% WR); better exit logic doesn't help when the entries are wrong half the time. Improving exits would help a 55% WR signal; it can't rescue 31%.
4. **Accept verdict: 1-min retail crypto momentum has no edge** -- likelihood: high. Consistent with Exp 2 (all TA on crypto), Exp 4 (TA on FX), Exp 3 (TA on futures). Adds the "high-frequency intraday" data point to the broader TA-has-no-edge conclusion.

---

## Handoff Note — praxis_main_current (2026-04-03)

### Completed this session

1. ✅ **Regime feature comparison on funding carry** — Hand-crafted (11) > combined (57) > regime-only (46). Structural strategies don't need regime features.
2. ✅ **DEX scanner + quoter** — Built, tested on Arbitrum mainnet, killed spatial arb thesis with real depth data
3. ✅ **Flash loan mechanics clarified** — Single-tx execution, can't hold positions, looping is valid use case
4. ✅ **Momentum signal detector** — 5-component composite, 72h backtest across 5 assets, 88 trades
5. ✅ **Two experiments added to atlas** — #16 (DEX arb ❌) and #17 (momentum ❌)
6. ✅ **Universal principle updated** — Now covers 17 experiments across 6 asset classes + DeFi

### What to work on next

1. **Deploy funding rate carry to live trading** — The one confirmed edge (Sharpe 4-11, max DD -0.03%). Live signal generation, position sizing, execution workflow.

2. **Dollar bar data pipeline** — For validated strategies, switch from time bars to dollar bars (Lopez AFML Ch. 2)

3. **Vol/VRP revisit** — When proper IV data source becomes available

### What works right now

- **Funding rate carry monitor** is live and operational
  - `python scripts/funding_monitor.py --loop --gate 0.70`
  - `streamlit run gui/funding_monitor/dashboard.py`
  - Alert Jeff when funding rates flip positive and sustain

### Reminder: model retrain trigger

When `output/funding_rate/cpo/phase3_models.joblib` is 6+ months old, re-run phase2+3 with updated training window.
