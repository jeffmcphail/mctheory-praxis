# Trading Strategy Atlas

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
| **Date** | 2026-03-22 |
| **Framework** | Burgess pair discovery → Chan CPO intraday execution |
| **Signal** | Cointegration-based spread z-score, single-asset execution |
| **Data** | Polygon.io minute bars, 38 deduplicated pairs |
| **Training** | 2025 (240 configs × 250 days × 38 pairs = 2.3M strategy-days) |
| **OOS** | 2026-01-01 → 2026-03-20 (53 trading days) |
| **TC** | 2 bps/leg × 2 legs = 4 bps round-trip |
| **RF AUC** | 0.77-0.87 (config selection works) |
| **Result** | **NEGATIVE after TC** |
| **Sharpe** | -1.44 (with TC), +3.99 (without TC — edge exists but costs eat it) |

**Key findings:**
- Daily EOD cointegration pairs trading on SP500 is saturated — profits arbitraged away
- Intraday mean-reversion on minute bars has signal pre-TC but costs 4bps/round-trip
- At 2-3 trades/day, daily TC drag is 8-12 bps which exceeds mean daily alpha
- RF config selection provides genuine lift (~10% improvement in win rate) but insufficient to overcome TC
- Hedge ratios from Burgess cointegration are stable but the spread dynamics are too weak intraday

**Atlas principle established:**
> Intraday mean-reversion on liquid US equities requires TC < 2 bps round-trip to be viable at retail. Institutional-grade execution (sub-1bps) may work but is not accessible.

**Risk management lessons:**
- UTC→Eastern timezone conversion is critical for US equity minute data
- Return normalization must use notional capital, not spread std dev
- NaN propagation defense needed at every feature→model boundary
- Pair deduplication (canonical key = sorted tickers) prevents double-counting

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

---

## Pending experiments

### 3. TA_STANDARD × FUTURES_INDEX
- **Status**: Framework ready, needs data pipeline (could use Polygon futures or CCXT)
- **Hypothesis**: Futures have clearer trend/mean-reversion regimes than crypto, lower noise
- **Expected finding**: Similar to crypto (TA is TA regardless of asset class)

### 4. TA_STANDARD × FX_G10
- **Status**: Framework ready, needs data pipeline
- **Hypothesis**: FX has longest history and most stable microstructure
- **Expected finding**: If TA works anywhere, it works here — but probably doesn't

### 5. MOMENTUM × CRYPTO
- **Signal type**: Time-series momentum (past N-hour returns predict next N hours)
- **Why different**: Momentum is a documented risk premium, not a technical pattern
- **Hypothesis**: Crypto momentum may persist due to retail-dominated order flow

### 6. MICROSTRUCTURE × CRYPTO
- **Signal type**: Funding rate, order book imbalance, liquidation cascades
- **Why different**: Structural/mechanical signals rather than price-pattern signals
- **Hypothesis**: Best candidate for genuine persistent edge in crypto

### 7. FUNDAMENTAL × FX_G10
- **Signal type**: Carry (interest rate differential), PPP, current account
- **Why different**: Macro factors, not technical patterns
- **Hypothesis**: Carry has documented risk premium; CPO may optimize timing

### 8. ALTERNATIVE × CRYPTO
- **Signal type**: On-chain metrics (active addresses, exchange flows, whale movements)
- **Why different**: Information not derivable from price alone
- **Hypothesis**: Alpha from information edge rather than pattern edge

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
| Total experiments | 2 complete, 6 pending |
