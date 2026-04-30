# CPO Regime Matrix Design

> **Sync state:** This file is the source of truth. After editing, run
> `python -m engines.atlas_sync` to update the queryable DB at
> `data/praxis_meta.db`. See `docs/ATLAS_DB.md`.

## The Framework

The regime matrix is a principled hierarchical feature selection protocol:
1. Define 12 regime classes, each with discrete states
2. Encode current state of each class as a compact integer vector
3. Run CPO ablation to find which regime classes predict strategy success
4. Drill down to the raw sub-features that define winning regime classes
5. Rebuild strategy feature sets from those raw sub-features

---

## Regime Classes

| # | Class | States | Detection | Key data |
|---|-------|--------|-----------|---------|
| A | Trend | -2/-1/0/+1/+2 | ADX + multi-period return alignment + MA stack | OHLCV daily |
| B | Vol level | 0/1/2/3 (compressed→spike) | Garman-Klass RV vs 252d percentile rank | OHLCV hourly |
| C | Vol trend | -1/0/+1 (compressing/stable/expanding) | rv_1d / rv_7d ratio | OHLCV hourly |
| D | Serial correlation | -2/-1/0/+1/+2 (MR→momentum) | Hurst exponent (R/S) + variance ratio + autocorr | OHLCV hourly |
| E | Microstructure | -1/0/+1 (sell/balanced/buy flow) | Lee-Ready tick rule approximation from OHLCV; OFI | OHLCV hourly |
| F | Funding/positioning | -2/-1/0/+1/+2 | Annualized funding + OI change 7d | Funding rates + OI |
| G | Liquidity | 0/1/2/3 (abundant→crisis) | Corwin-Schultz spread + Amihud + volume z-score | OHLCV hourly |
| H | Cross-asset corr | 0/1/2 (idiosyncratic/normal/crisis) | Rolling pairwise corr mean across universe | Multi-asset OHLCV |
| I | Volume participation | 0/1/2/3 (dormant/normal/active/capitulation) | Vol 24h ratio + price-volume sign confirmation | OHLCV hourly |
| J | Term structure | -1/0/+1 (backwardation/flat/contango) | FR term slope (7d vs 30d ann) + perp basis | Funding + perp OHLCV |
| K | Cross-sectional dispersion | 0/1/2 (low/normal/high) | Std of cross-asset 24h returns, percentile rank | Multi-asset OHLCV |
| L | RV / IV spread (VRP) | -1/0/+1 (cheap vol/balanced/expensive vol) | GARCH RV vs DVOL (requires real options data) | DVOL + OHLCV |

---

## Relevance Matrix

| Strategy | A | B | C | D | E | F | G | H | I | J | K | L |
|----------|---|---|---|---|---|---|---|---|---|---|---|---|
| Crypto TA | ●●● | ●●● | ●● | ●●● | ●● | ● | ●● | ●● | ● | ● | ●● | ● |
| Momentum | ●●● | ●●● | ●● | ●●● | ●● | ●●● | ● | ●● | ●● | ● | ●● | ● |
| Funding carry | ● | ●● | ● | ● | ● | ●●● | ●● | ● | ● | ●●● | ● | ●● |
| Grid bot | ●●● | ●●● | ●●● | ●●● | ● | ● | ●● | ● | ●● | ● | ● | ● |
| Vol / VRP | ● | ●●● | ●●● | ● | ●● | ●● | ●● | ●●● | ●● | ●●● | ●● | ●●● |

● = minor / ●● = moderate / ●●● = high relevance

---

## Algorithmic Definitions

### A. Trend Regime

```python
# Component 1: ADX (Wilder's Average Directional Index)
tr = max(H-L, |H-prev_C|, |L-prev_C|)    # true range
+DM = H - prev_H if positive else 0
-DM = prev_L - L if positive else 0
+DI_14 = 100 * EMA(+DM, 14) / ATR_14
-DI_14 = 100 * EMA(-DM, 14) / ATR_14
DX = 100 * |+DI - -DI| / (+DI + -DI)
adx_14 = EMA(DX, 14)

# Component 2: Multi-period return sign alignment
ret_1d  = log(close[-1] / close[-2])
ret_7d  = log(close[-1] / close[-8])
ret_30d = log(close[-1] / close[-31])
direction = +1 if ret_7d > 0 else -1

# State: [+2, +1, 0, -1, -2]
if adx_14 > 40 and sign(ret_1d) == sign(ret_7d) == sign(ret_30d):
    state = direction * 2
elif adx_14 > 25:
    state = direction * 1
else:
    state = 0
```

Transition signal: ADX crossing 25 from below = trend inception. ADX peaked + price divergence = exhaustion.

---

### B & C. Volatility Level + Trend

```python
# Garman-Klass estimator (uses OHLC, ~4x lower variance than close-to-close)
gk_bar = 0.5 * (log(H/L))**2 - (2*log(2)-1) * (log(C/O))**2

rv_1d  = sqrt(mean(gk_bars[-24:])  * 24 * 365)   # annualized
rv_7d  = sqrt(mean(gk_bars[-168:]) * 24 * 365)
rv_30d = sqrt(mean(gk_bars[-720:]) * 24 * 365)

# Vol level (B) via percentile rank
vol_pct_rank = percentilerank(rv_1d, history_252d)
vol_level = {<0.20: 0, <0.60: 1, <0.85: 2, else: 3}  # compressed/normal/elevated/spike

# Vol trend (C)
vol_ratio = rv_1d / rv_7d
vol_trend = {>1.25: +1, <0.80: -1, else: 0}  # expanding/stable/compressing
```

---

### D. Serial Correlation Regime (most important single feature)

```python
# Hurst exponent via R/S analysis
# H > 0.5 = persistent (momentum favored)
# H < 0.5 = antipersistent (mean-reversion favored)
def hurst_rs(log_prices, min_lag=8, max_lag=200):
    lags = logspace(log(min_lag), log(max_lag), 20, base=e).astype(int)
    RS_values = [rescaled_range(log_prices, lag) for lag in lags]
    H, _ = polyfit(log(lags), log(RS_values), 1)
    return H

# Variance ratio test (faster, complementary)
# VR(q) = Var(q-period ret) / (q * Var(1-period ret))
# VR > 1 = momentum, VR < 1 = mean-reversion
vr_4  = variance_ratio(hourly_rets, q=4)    # 4h
vr_24 = variance_ratio(hourly_rets, q=24)   # 1-day

# State
hurst = hurst_rs(log_prices[-504:])   # 21 days of hourly
score = (hurst - 0.5) * 4             # normalized [-2, 2]
state = {>1.5: +2, >0.4: +1, <-1.5: -2, <-0.4: -1, else: 0}
```

The Hurst exponent should be in every strategy's feature set. It is the most direct measure of which strategy type (trend vs MR vs grid) the current market structure favors.

---

### E. Microstructure / Order Flow Imbalance

```python
# Lee-Ready tick rule approximation from OHLCV (no tick data needed)
buy_vol  = volume * clip((close - open) / (high - low + 1e-10), 0, 1)
sell_vol = volume - buy_vol
ofi_bar  = (buy_vol - sell_vol) / (volume + 1e-10)   # [-1, +1] per bar

# 24-hour rolling OFI
ofi_24h = rolling_mean(ofi_bar, 24)

# Amihud illiquidity: |return| / dollar_volume (price impact per unit notional)
amihud = |daily_ret| / (close * volume)

state = {ofi_24h > 0.15: +1, ofi_24h < -0.15: -1, else: 0}
```

---

### F. Funding / Positioning Regime

```python
fr_ann = funding_8h_rate * 3 * 365 * 100   # annualized %
oi_change_7d = (oi_now - oi_7d_ago) / oi_7d_ago

# Combined positioning state
if fr_ann > 30 and oi_change_7d > 0.10:   state = +2   # heavily long
elif fr_ann > 10:                           state = +1   # mildly long
elif fr_ann < -15 and oi_change_7d < -0.10: state = -2  # heavily short
elif fr_ann < -5:                           state = -1   # mildly short
else:                                       state = 0    # neutral
```

---

### G. Liquidity Regime

```python
# Corwin-Schultz (2012) spread estimator from H/L
# Best available OHLCV-based bid-ask spread proxy
beta  = (log(H_t/L_t))**2 + (log(H_2day/L_2day))**2
gamma = (log(max(H_t, H_t1) / min(L_t, L_t1)))**2
alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
cs_spread = max(2*(exp(alpha) - 1) / (1 + exp(alpha)), 0)

amihud_ratio = abs(daily_ret) / (close * volume)

# Combined liquidity score (higher = more liquid)
liq_score = -(z_score(cs_spread, 30d)) - (z_score(amihud, 30d)) + (z_score(volume, 30d))
state = {>1.0: 0, >-0.5: 1, >-2.0: 2, else: 3}  # abundant/normal/thin/crisis
```

---

### H / I / J / K — Cross-asset and structural regimes

```python
# H: Cross-asset correlation
corr_matrix = rolling_corr(all_asset_hourly_rets, window=168)   # 7d
mean_pairwise_corr = mean(upper_triangle(corr_matrix))
state = {>0.75: 2, >0.45: 1, else: 0}   # crisis/normal/idiosyncratic

# I: Volume participation
vol_ratio = volume_24h / volume_30d_mean
conviction = (sign(ret_24h) == (vol_ratio > 1.0))   # price-volume alignment
state = {<0.5: 0, <1.5 and conviction: 1, <3.0 and conviction: 2, else: 3}

# J: Term structure
fr_slope = (fr_30d_ann - fr_7d_ann) / (|fr_30d_ann| + 1e-6)
state = {>0.15: +1, <-0.15: -1, else: 0}   # contango/flat/backwardation

# K: Cross-sectional dispersion (alpha environment)
cross_rets_24h = [asset_ret_24h for asset in universe]
dispersion = std(cross_rets_24h)
state = {>70th_pct: 2, >30th_pct: 1, else: 0}   # high/normal/low dispersion
```

---

## Lopez Bar Types

### Dollar Bars (recommended primary bar type for CPO)
- Close bar when cumulative dollar volume = threshold
- Every bar represents equal economic activity
- Better statistical properties: more Gaussian returns, lower autocorrelation, better IID
- Threshold for BTC hourly equivalent: ~$100M (adjust to maintain ~6 bars/day)

### Volume Imbalance Bars (best for momentum strategies)
- Close bar when |Σ(buy_vol − sell_vol)| exceeds expected imbalance
- Bars form faster when directional conviction is high
- Embeds microstructure regime (E) directly into bar formation
- Requires trade-level data with buy/sell classification

### Volume Run Bars (best for regime detection)
- Close when a "run" of same-direction ticks exceeds expected length
- Captures serial correlation regime (D) in bar formation itself
- Best statistical properties of all alternative bar types (Lopez, AFML Ch. 2)

---

## Experimental Protocol

### Step 1: Build `engines/regime_engine.py`
Single module computing all 12 regime states from OHLCV + funding data.
Output: 12-element integer vector + 60-element raw sub-feature matrix.

### Step 2: CPO ablation — which regime classes predict each strategy?
For each strategy, run CPO three times:
- Full regime vector (12 features)
- Subset ablations (each class independently)
- Config params only (baseline)
Compare AUC and calibration lift across subsets.

### Step 3: Drill into winning classes → extract raw sub-features
Replace regime state integers with the raw sub-features that compute them.
Re-run CPO with raw features. Compare vs regime states — did specificity improve?

### Step 4: Rebuild strategy feature sets
Each strategy gets a purpose-built feature set:
- Momentum: {D raw, F raw, I raw} + config params
- TA: {A raw, B raw, D raw, K raw, multi-frequency signals} + config params
- Grid: {B raw, C raw, D raw, G raw} + config params
- Vol: {B raw, C raw, E raw, J raw, L raw} + config params

### Step 5: Switch to dollar bars for validated strategies
Rebuild data pipeline with dollar bars. Expected improvement: AUC +2-5%, stronger calibration monotonicity.
