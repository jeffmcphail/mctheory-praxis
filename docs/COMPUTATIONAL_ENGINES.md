# Praxis Computational Engines

> **Sync state:** This file is the source of truth alongside
> `docs/REGIME_MATRIX.md` and the `TRADING_ATLAS.md` series.
> After editing, run `python -m engines.atlas_sync` to update
> `data/praxis_meta.db`.

This document defines the **seven computational engine
taxonomy** that organizes every systematic trading strategy in
Praxis. The taxonomy answers a different question than the
Landscape Matrix (Signal x Asset), the Regime Matrix
(market-state classes), or the Atlas (specific experiment
results): *what mathematical machinery does each strategy
require?*

---

## The Framework

A strategy has three orthogonal layers:

1. **Computational engine**: pure math operating on
   well-defined mathematical objects. No I/O, no business
   context, no decisions about which assets or when. Numpy /
   scipy work.
2. **Model context**: business layer specifying universe,
   temporal scope, transaction costs, risk constraints,
   execution rules. Different strategies that share an engine
   differ only in context.
3. **I/O adapters**: data providers and result stores.
   Pluggable, doesn't affect the math.

The discriminator that puts two strategies in different
**engines** isn't what they trade or when -- it's what
mathematical objects the engine operates on and what
transformations it applies. Pairs trading on equities and pairs
trading on FX use the *exact same engine*; momentum on crypto
and momentum on commodities use the *exact same engine*; but
pairs trading and momentum use *different engines* because the
math (residual extraction + stationarity tests vs.
return-series scoring) is genuinely different.

This is the discriminator. After analyzing the systematic
trading universe, **seven distinct engines** cover essentially
all systematic strategies. Anything else is a parametric
variation or a business context difference.

---

## The Seven Engines

### Engine 1 -- Cointegration / Mean-Reversion

**Code namespace:** `StatArbEngine` (in `engines/`,
implementations in `engines/burgess.py`,
`engines/cointegration.py`, `engines/pairs_trading.py`).

**Mathematical object:** residual series from regression of
multiple price series.

**Core math:**
- Stepwise / OLS / ridge regression -> residual extraction
- Stationarity testing (ADF, KPSS, Johansen)
- Hurst exponent estimation (R/S, DFA)
- Half-life of mean reversion (Ornstein-Uhlenbeck parameter
  fitting)
- Variance ratio profiles (multi-timescale)
- Monte Carlo null distribution generation
- Z-score signal generation on spread
- Markowitz-family weight optimization for basket construction

**Data shape:** `(n_obs, n_assets)` price matrix -> residual
vectors -> scalar test statistics.

**Strategies that use this engine** (same math, different
business context):

| Named strategy | Differentiator (business only) |
|---|---|
| Pairs trading | Universe = 2 assets |
| Statistical arbitrage | Universe = N assets, basket construction |
| ETF / index arbitrage | Universe = ETF + constituents |
| Convergence trades | Universe = related instruments (on-the-run / off-the-run bonds) |
| Cross-asset relative value | Universe spans asset classes |
| Basis trades (futures vs spot) | Universe = futures + underlying |
| Capital structure arb | Universe = equity + debt of same issuer |
| Grid bot (range-bound MR) | Universe = single asset oscillating in a range |

**Status in Praxis:** core implementation present
(`engines/burgess.py`, `engines/cointegration.py`,
`engines/pairs_trading.py`); Burgess pair discovery + Chan CPO
intraday execution validated as the canonical pipeline.

---

### Engine 2 -- Momentum / Trend-Following

**Code namespace:** `MomentumEngine` (in
`engines/btc_momentum.py`,
`engines/momentum_signals.py`,
`engines/crypto_ta_strategy.py`,
`engines/universal_ta_strategy.py`).

**Mathematical object:** return series (single asset or
cross-section).

**Core math:**
- Multi-period return computation (1d, 5d, 21d, 63d, 252d)
- Volatility estimation (realized, EWMA, GARCH)
- Volatility-targeted position sizing
- Cross-sectional ranking and percentile scoring
- Trend-strength filters (ADX, regression-slope significance)
- TA indicator computation (RSI, MACD, Bollinger, etc.) where
  the underlying signal is "is this trend persistent?"

**Data shape:** `(n_obs,)` price series per asset -> scalar
score per (asset, time).

**Strategies that use this engine:**

| Named strategy | Differentiator (business only) |
|---|---|
| Time-series momentum (TSMOM) | Score = return over lookback, scaled by vol |
| Cross-sectional momentum (XSMOM) | Score = relative rank within universe |
| Dual momentum (Antonacci) | TSMOM filter + XSMOM ranking |
| TA standard (RSI, MACD, etc.) | Score = thresholded indicator state |
| TA composite (Market Cipher B et al.) | Score = multi-indicator vote |
| Contrarian (mean-reversion of returns) | Same engine, signal_sign flipped |
| Short-term momentum (1-min, intrabar) | Same engine, finer frequency |

The contrarian-vs-momentum distinction is famously a sign flip
in the same engine, not a different engine. The architectural
implication: a single `MomentumEngine` with a `signal_sign`
parameter covers both.

**Status in Praxis:** rich implementation (multiple TA
strategies + TSMOM + intrabar variants); validated extensively
as the most-tested engine in the Atlas.

---

### Engine 3 -- Portfolio Allocation

**Code namespace:** `AllocationEngine` (in
`engines/allocation.py`, `engines/cpo_core.py`).

**Mathematical object:** covariance / correlation structure ->
weight vector.

**Core math:**
- Covariance estimation (sample, shrinkage, factor model)
- Mean-variance optimization (Markowitz)
- Risk parity (equal risk contribution)
- Hierarchical risk parity (HRP, Lopez de Prado)
- Minimum-variance optimization
- Black-Litterman blending of priors with views
- Kelly criterion (single-asset and joint)
- CPO ensemble allocation (RF-driven config selection)

**Data shape:** `(n_assets, n_assets)` covariance + `(n_assets,)`
expected returns -> `(n_assets,)` weights summing to a target
gross or net leverage.

**Strategies that use this engine** (almost everything in
production uses Engine 3 as a sub-component):

| Named strategy | Differentiator (business only) |
|---|---|
| Risk parity | Objective = equal risk contribution |
| Mean-variance | Objective = max Sharpe |
| Min-variance | Objective = min portfolio vol |
| Factor investing | Inputs = factor exposures rather than raw assets |
| Tactical allocation | Inputs are time-varying via macro signals |
| Black-Litterman | Inputs blend prior + views |

The architectural insight: allocation is rarely a standalone
strategy in Praxis -- it's the *composition layer* that turns
many Engine 1, 2, 4, or 7 sub-models into a portfolio. CPO is
the Praxis-specific version of this with RF-driven config
selection.

**Status in Praxis:** central to the CPO framework
(`engines/cpo_core.py`); all multi-model experiments use this
as the portfolio construction layer.

---

### Engine 4 -- Volatility / Options

**Code namespace:** `OptionsEngine` (in
`engines/garch_model.py`, `engines/vol_strategy.py`,
`engines/vol_surface.py`).

**Mathematical object:** volatility surface + Greeks.

**Core math:**
- Implied volatility extraction from option prices
- Vol surface fitting (SVI, SABR, polynomial)
- GARCH-family realized vol forecasting
- Greeks computation (delta, gamma, vega, theta)
- Continuous delta-hedging with arbitrary rebalance schedule
- Variance / volatility risk premium calculation
- Term structure modeling (VIX futures curve, BTC DVOL term)

**Data shape:** option chain (or proxy) -> 2D surface
indexed by (strike, expiry) -> Greeks per (strike, expiry, t).

**Strategies that use this engine:**

| Named strategy | Differentiator (business only) |
|---|---|
| Variance risk premium | Long realized vol, short implied vol |
| Vol surface trading | Trade specific surface dislocations |
| Gamma scalping | Long gamma, dynamic delta hedge |
| VIX term structure | Trade contango / backwardation |
| Tail risk hedging | Buy OTM puts as portfolio insurance |

This is the only engine that operates on a *2D surface* rather
than a 1D series; it's genuinely computationally distinct.

**Status in Praxis:** infrastructure present
(`engines/garch_model.py` for realized vol forecasting,
`engines/vol_strategy.py` for VRP simulation,
`engines/vol_surface.py` for Deribit DVOL ingestion); blocked
on real historical IV data per Atlas Exp 15.

---

### Engine 5 -- Order Book / Microstructure

**Code namespace:** (planned -- partially in
`engines/intrabar_predictor.py` and
`engines/momentum_signals.py`).

**Mathematical object:** limit order book state (level-2 or
level-3) + trade flow + inventory state.

**Core math:**
- Order flow imbalance (OFI) computation
- Kyle's lambda estimation (price impact)
- Spread / depth metrics (best bid/ask, top-N levels)
- Stochastic optimal control for market making (Avellaneda-
  Stoikov family)
- Inventory management dynamics
- Trade arrival rate modeling (Hawkes processes)
- Tick-rule classification (Lee-Ready trade direction)

**Data shape:** event-driven (every order book update or trade)
-> per-event feature vector.

**Strategies that use this engine:**

| Named strategy | Differentiator (business only) |
|---|---|
| Market making | Quote both sides, manage inventory |
| Momentum ignition | Detect order flow signals in real time |
| OFI alpha | Use OFI as a directional predictor |
| TWAP / VWAP execution | Schedule trades against intra-bar liquidity |

**Status in Praxis:** partial -- order book collector landed in
Cycle 22 (10s snapshots, top-10 levels), but no
full-machinery market-making implementation yet. Used today as
*input features* (OFI, spread_bps) to Engine 2 (momentum)
rather than as a standalone engine.

---

### Engine 6 -- On-Chain / DeFi

**Code namespace:** `engines/dex_quoter.py`,
`engines/dex_scanner.py`, `engines/flash_scanner.py`,
`engines/flash_executor.py`.

**Mathematical object:** blockchain state (pool reserves,
mempool transactions, account balances) + atomic-transaction
graph.

**Core math:**
- AMM curve math (constant-product, concentrated liquidity,
  stable curve)
- Cross-pool price quoting with slippage
- Multi-hop arbitrage path finding (graph optimization over
  pool-pair edges)
- Flash loan composition (single-tx atomic execution)
- MEV-aware transaction ordering
- On-chain metric aggregation (active addresses, hash rate,
  exchange flows)

**Data shape:** event-driven (block-by-block) + state-driven
(pool reserves at any block height).

**Strategies that use this engine:**

| Named strategy | Differentiator (business only) |
|---|---|
| Cross-DEX spatial arbitrage | Universe = DEX pool pairs on one chain |
| Cross-chain arbitrage | Adds bridge latency / cost modeling |
| Sandwich attacks | Exploits mempool ordering |
| JIT liquidity provision | Provides single-block LP for fee capture |
| Flash loan + stat arb | Engine 6 + Engine 1 composed |

**Status in Praxis:** v0.1 implementation in
`engines/dex_*` and `engines/flash_*`. Atlas Exp 16 confirmed
spatial arb has no executable depth on Arbitrum at retail
scale, so this engine isn't a current production target -- but
infrastructure remains.

---

### Engine 7 -- Event / Signal

**Code namespace:** `EventSignalEngine` (in
`engines/event_classifier.py`, `engines/event_signal.py`,
`engines/ai_ensemble.py`,
`engines/funding_rate_strategy.py`,
`engines/intrabar_predictor.py`).

**Mathematical object:** raw event / feature -> scalar alpha
score.

**Core math:**
- Feature engineering (technical, fundamental, alternative)
- Classification (XGBoost, RF, LSTM, neural)
- Triple-barrier labeling (Lopez de Prado)
- Probabilistic calibration (Platt scaling, isotonic)
- Time-series ML (LSTM, transformer, attention)
- Embedding-based similarity (NLP / news)
- Funding rate / OI / on-chain metric -> alpha

**Data shape:** heterogeneous features per (asset, time) ->
scalar score per (asset, time).

**Strategies that use this engine:**

| Named strategy | Differentiator (business only) |
|---|---|
| Funding rate carry | Score = funding rate trend + magnitude |
| LSTM crypto predictor | Score = neural network output |
| Triple-barrier-labeled classifier | Score = P(TP hit before SL hit) |
| News / sentiment trading | Score = NLP embedding similarity |
| AI ensemble probability engine | Score = multi-LLM consensus |
| Quantamental | Score = traditional fundamental + ML |

**Status in Praxis:** rich implementation. Engine 7 is the
*feeder* into Engines 2 (momentum sizing) and 3 (portfolio
construction); most experiments are Engine 7-flavored signal
generation followed by Engine 3 portfolio composition. The
single confirmed POSITIVE experiment in the Atlas (Funding
Rate Carry, Exp 13) is Engine 7.

---

## Architecture: how engines compose

The implementation pattern (validated in `engines/base.py`,
`engines/context/model_context.py`,
`engines/adapters/providers.py`, `engines/model.py`):

```
TimeSeriesEngine    SurfaceEngine    StateEngine
       |                  |               |
   (Engines 1, 2, 7)  (Engine 4)    (Engines 5, 6)
       |
       v
   Engine output (signals / scores) ---+
                                       |
                                       v
                              AllocationEngine (3)
                                       |
                                       v
                               Portfolio weights
                                       |
                                       v
                              ExecutionContext
                                       |
                                       v
                                I/O adapter -> trades
```

A strategy is `Engine + ModelContext + I/O adapters`. Same
engine + different ModelContext = different named strategy.
This is what makes the taxonomy compact and orthogonal.

---

## Mapping known atlas experiments to engines

(Updated as part of Cycle 33 backfill.)

| Exp | Section | Primary engine | Secondary engine |
|---|---|---|---|
| 1 | MEAN_REVERSION x EQUITY_US (SP500 Pairs) | 1 (Cointegration) | 3 (Allocation) |
| 2 | TA_STANDARD x CRYPTO | 2 (Momentum) | 3 (Allocation) |
| 3 | TA_STANDARD x FUTURES | 2 | 3 |
| 4 | TA_STANDARD x FX_G10 | 2 | 3 |
| 7 | TA_COMPOSITE x CRYPTO MCb | 2 | 3 |
| 8 | MOMENTUM x CRYPTO TSMOM_4H + TSMOM_DAILY | 2 | 3 |
| 9 | MOMENTUM x CRYPTO Triple Barrier | 2 + 7 (TB labels) | 3 |
| 10 | TA_STANDARD x CRYPTO Triple Barrier | 2 + 7 | 3 |
| 11 | TA_STANDARD x FUTURES Triple Barrier | 2 + 7 | 3 |
| 12 | TA_STANDARD x FX Triple Barrier | 2 + 7 | 3 |
| 13 | MICROSTRUCTURE x CRYPTO Funding Carry | 7 (Event/Signal) | 3 |
| 14 | GRID BOT x CRYPTO | 1 (range-bound MR) | 3 |
| 15 | VRP x BTC/ETH | 4 (Vol/Options) | -- |
| 16 | CROSS-DEX SPATIAL ARBITRAGE | 6 (DeFi) | -- |
| 17 | SHORT-TERM MOMENTUM x CRYPTO | 2 + 5 (microstructure) | -- |

The most-tested engine is 2 (Momentum), with 9 of 15
experiments. The single POSITIVE result is Engine 7 (Funding
Carry, Exp 13). Engine 4 (Vol/Options) is data-blocked. Engine
5 (Order Book / Microstructure) shows up only as a feature
input to other engines, not as a standalone experiment yet.

---

## Why this matters

The Atlas is currently structured by Signal Type x Asset Class
(the Landscape Matrix). That's a useful navigation grid. But
the taxonomy here adds a second navigation axis: **the
mathematical kinship between strategies**. When the LSTM v2
upgrade looks for revival hypotheses for Engine 2 + 7
experiments, it's looking specifically for what info bars +
triple-barrier labeling can do for the *math* of these
engines, not for what they can do for "TA on crypto" specific.

The Research Agent can use the engine column to filter:
"show me every Engine 1 experiment we've run" returns 2
experiments (Exps 1, 14). "Show me every Engine 7 experiment"
returns 6+ depending on how Triple Barrier overlaps are
counted. This is the structured complement to semantic search.

---

## Versioning

- v0.1 (Cycle 33): Initial taxonomy + atlas mapping +
  computational_engine column on `atlas_experiments`.
- Future: per-engine deep-dive subdocs (e.g.
  `docs/engines/engine_1_cointegration.md`) if the taxonomy
  proves load-bearing for navigation.
