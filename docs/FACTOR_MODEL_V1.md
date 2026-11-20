# The S&P 500 equity factor model, v1

Frozen 2026-11-17 by midoffice20 (ruling 1 of its brief): v1 is D19's model, the run's defaults
(`scripts/run_sp500_model.py --industries --membership`), stored by date in `data/model_store/v1`.
`docs/FACTOR_MODEL.md` (FM) is the full record. Each number cites its section there, in
`claude/MIDOFFICE_STATE.md` (ST) or in a retro (R*nn*: `claude/retros/RETRO_midoffice`*nn*`_*.md`).

## What v1 is

A daily cross-sectional model in Barra's mould: USE3's definitions, USE4's where USE3's measured
badly.

- **Data.** The S&P 500 point in time: 592 share classes keyed by composite FIGI, 1,214 trading
  days from 2022-01-03 to 2026-11-03, total returns (ST, The data).
- **25 factors, no market factor.** 20 industries: SIC major groups, a group under 15 members
  merged into its division. Every member is in one, by its code or by a ruling with evidence:
  FRCB and SBNY 6022, VYLR 0100, Agriculture into SIC28 (ST, No member goes without the market).
  5 styles, from 14 price and volume descriptors: Volatility, Momentum, Size, Size Nonlinearity
  and Trading Activity. The market model and Size Nonlinearity are USE4's (ST, Definitions).
- **Estimation.** WLS on sqrt(cap). Each day's return is regressed on that day's exposures,
  weights and members, all read the day before. Exposures are standardised over the members, with
  a cap-weighted mean and descriptors winsorised at 4 sd. 938 periods from 2023-02-08; mean
  R-squared 0.294; 502 members a regression at the median (ST, The run: At the defaults).
- **Factor covariance.** Volatilities on a 90-day half-life, correlations on 504, no lags at the
  one-day horizon, times lambda_F^2: the volatility regime adjustment's factor side on 21 days,
  0.963 as of 2026-11-03. No eigenfactor adjustment (ST, The factor covariance as it stands).
- **Specific risk.** USE3's characteristic model (EQ 3-12 to 3-14), blended by inverse variance
  with each security's own history: every specific return it has, on a 63-day half-life.
  p10 16.7%, median 29.0%, p90 48.9% a year (ST, At the defaults; FM, Specific risk from
  characteristics).

## How it was validated

- **Out of sample, refitted daily.** 686 forecasts, 2024-02-08 to 2026-11-02, each from the whole
  estimator refitted on data through its date. Each is set against the next day's return:
  B = sd(realised / forecast), band 1 +/- 0.054. Over the index, size, industry, optimiser,
  random and eigen portfolios and every member (FM, Out-of-sample bias statistics).
- **The halves.** A whole test's statistic is the root mean square of its halves'. So a pass can
  hide a model wrong in both halves: the unadjusted index's in-band 1.015 was 1.224, then 0.751.
  Candidates are read by half and by their rolling 12-month windows' mean absolute deviation from
  one (MRAD) (FM, same section).
- **The rule, fixed before each run** (`COVARIANCE_RULE`). The optimiser's realised volatility
  comes first: a setting under which either minimum-variance portfolio realises more over the
  test is out. Then the regime criterion, the mean of the broad, optimiser and member MRADs:
  chosen on the first half, checked on the second (FM, The correlation half-life on the
  regime-adjusted model). The two blocks had to beat the criterion on both halves (R19).
- **The guard.** On five dates (2024-02-08, 2024-10-14, 2025-06-23, 2026-02-27, 2026-11-02), the
  inputs rebuilt from the panel cut at the date equal the full run's to the bit, and so does the
  forecast made from them (FM, Out-of-sample bias statistics). The store passes the same guard (R20).

| v1, whole test / first half / second half | statistic | realised % a year |
|---|---|---|
| the index, cap-weighted | 0.931 / 1.060 / 0.783 | |
| minimum variance, long-only | 0.900 / 0.990 / 0.801 | 9.62 / 11.05 / 7.97 |
| minimum variance, unconstrained | 0.951 / 1.047 / 0.846 | 9.83 / 11.25 / 8.18 |
| the regime criterion | 0.091 / 0.068 / 0.138 | |

(FM, Every member on the market; The market's own multiplier.)

## Tried and rejected

Each one was measured against the model of its day (FM's bias sections; the retro named).

- **Industry floor of 15** (R11): Construction 1.066 to 1.319.
- **Sampling-error correction** (R11): the unconstrained optimiser's statistic 1.578 to 1.925.
- **One covariance half-life** (R11): at 45, worse nearly everywhere; at 180, the index's MRAD
  0.169 to 0.213.
- **Eigenfactor adjustment** (R13): the unconstrained optimiser realises 10.13% a year at a = 2,
  against 9.97%. One-sided, on the regime adjustment (R15): long-only 10.17% against 10.13%.
- **The regime adjustment's specific side** (R15): its squared statistic is 1.09 in both halves,
  a level set by the largest members rather than a regime.
- **Correlations on 63 days** (R16): the unconstrained optimiser realises 10.23% against 10.19%.
  At 126 and 252, the first half's criterion is 0.104 and 0.100, against 0.098.
- **One-sided eigenvalue adjustment on the 63-day correlations** (R17): unconstrained 10.23% to
  10.46%, against 10.19%.
- **Volatilities on 63, 42 or 21 days** (R17): unconstrained 10.26%, 10.36% and 10.61%, against
  10.19%.
- **Country factor as a covariance basis, one multiplier** (R18): the first half's criterion is
  0.103 against 0.098, and on v1 0.078 against 0.068 (R19).
- **Two blocks, lambda_C on the country and lambda_R on the rest** (R19): the unconstrained
  optimiser realises 9.88% against 9.83%, and 11.74% against 11.25% in the first half.
  **The leading candidate for a v2.** It fixes the calm half's level: the index reads 1.036 and
  0.938 by half, and its criterion is 0.059 and 0.067 against 0.068 and 0.138. Its price is noise
  and the optimiser's small directions. The index's forecast moves 1.62% a day at the median,
  against 1.00%. The unconstrained portfolio puts 27.6% of its forecast variance in the smallest
  ten eigen-directions, against 26.3%. Stored beside v1 as `v1-blocks`.

## Known limitations

- **The largest members' specific risk is forecast too low.** The top capitalisation quintile
  reads 1.028 (ST, What is next), and the cap-weighted specific statistic is 1.09 in both halves
  (FM, The volatility regime adjustment, measured). The fix is by size, in the specific risk
  model, and is not in v1 (ruling 5).
- **The calm half is over-forecast.** In the second half the index reads 0.783 and the optimiser
  0.801 and 0.846. What is left is the correlations: realised 0.25 to 0.33 between factors from
  2025 Q3, forecast 0.40 to 0.46 (ST, Open: the regime). The two blocks fix it, at their price.
- **The regime adjustment is noisy.** It moves the index's forecast 1.00% a day at the median,
  against 0.27% without it (FM, The volatility regime adjustment, measured).
- **Thin factors are part noise.** Weighted noise shares are Wholesale 60.0%, Size Nonlinearity
  45.6%, SIC20 45.0%, Construction 37.7% and Finance & Real Estate 36.9% (ST, At the defaults).
- **The optimiser's specific part reads over-forecast:** 0.83 to 0.89 by eigen-split (R19).
- **One test window.** 686 days, 2.7 years with the April 2025 shock in them; the longer horizons
  rest on its mean reversion (ST, A longer horizon is a reporting view).
- **Price and volume styles only.** There is no Value, Growth, Leverage or Yield. A member without
  a full exposure row has no forecast: 1,182 member-periods, all for want of Momentum (ST, At the
  defaults).
