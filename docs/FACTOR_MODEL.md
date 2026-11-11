# The factor model builder

*Mid-office market risk engine. Written 2026-11-01.*

## What it produces

Exactly the pair `compute_risk` consumes: per-security `ExposureVector`s carrying factor
loadings and specific risk, and a `FactorCovariance`. So a model built here frees the engine
from needing a vendor model at all.

```python
model = estimate_timeseries_model(panel, factor_returns)
result = compute_risk(as_of, book, model.exposures, model.covariance, ...)
```

`test_a_built_model_feeds_compute_risk` runs that end to end.

## Two directions, and why the difference matters

Fama-French and Barra build a factor model in opposite directions, and the engine this feeds
was ported from a Barra consumer.

| | exposures | factor returns |
|---|---|---|
| **time series** (Fama-French, built here) | estimated betas, by regressing each security on the factor returns | given -- constructed from sorted portfolios |
| **cross-sectional** (Barra) | observable characteristics, standardised | estimated per period, by regressing returns cross-sectionally on the exposures |

Both produce the same two objects, so both plug in. The practical difference is that a
time-series beta is backward-looking over its estimation window and measured with error, while
a characteristic is observable today. That is why Barra chose characteristics for risk
*forecasting*, and it is why RAT's exposure table holds loadings that are not betas. A model
built here inherits the limitation and `EstimationNotes` says so.

The cross-sectional path is the natural second estimator and shares everything except the
regression direction.

## Annualisation is explicit and recorded

A covariance estimated from monthly returns is a monthly covariance. The engine's tail measures
annualise by 252 trading days, so feeding it a monthly covariance understates risk by about
3.5x with nothing in the number to say so.

`Frequency` carries both scalings and keeps them distinct:

* **variance** scales linearly: multiply by periods per year
* **volatility** scales by the square root

Scaling a volatility linearly is the commonest annualisation error and turns a monthly 4% into
48% rather than 13.9%. A model built with `annualise=False` carries a qualification saying
exactly what the understatement factor would be.

Exposures are unaffected: a beta is a ratio of returns and carries no time units.

## The intercept, and where its absence lands

Fitted by default and **not** an exposure -- a factor risk model has nothing to say about alpha,
but omitting it forces every regression through the origin, so the slopes absorb whatever drift
the window contained.

The size of that bias was worth measuring rather than asserting. It lands in proportion to
`mean_f / E[f^2]`, so it concentrates on the **lowest-variance factor**, not the largest. On a
three-factor fixture with a 20% drift:

| factor | with intercept | without | shift |
|---|---|---|---|
| MKT (largest variance) | +1.257 | +1.083 | 14% |
| SMB | -0.224 | -0.276 | 23% |
| HML (smallest variance) | +0.094 | **-0.125** | sign flip |

So it is not a cosmetic choice: the factor whose exposure flips sign is the one a value
manager would be judged on. The first draft of the test asserted on the market beta at a
smaller drift, where it moves 3.3% -- below the threshold -- and failed against correct code.

At zero drift the two fits still disagree slightly, because the *in-sample* factor means are not
exactly zero. The test asserts proportionality -- the drift case shifts more than twenty times
the zero case -- rather than exact agreement, which is the property rather than the artifact.

## Missing is not zero

A security that did not trade, or did not yet exist, has no return for that period. Treating
that as a zero return pulls its estimated volatility down and its beta toward zero, which is a
bias toward looking safe. `nan` marks absence and the estimator counts what it actually had.

A gap in a *factor* series is different and is refused outright: one security's gap can be
skipped, but a factor's gap breaks every security's estimate for that period, so the period
should be dropped instead.

## Two exclusion rules, for different reasons

* **`min_observations`** (default 24, two years of monthly returns) is a judgement: a
  three-factor beta from six months is not a number anyone should act on. Excluded securities
  are named, and they arrive at the engine as uncovered.
* **Degrees of freedom** is arithmetic: with observations equal to parameters the fit is exact
  and the specific risk comes out as zero for mechanical reasons rather than because the
  factors explain the security. One degree of freedom is terrible but defined, so it is
  estimated -- the guard is about undefined, not unwise.

## This is where rank-deficient covariances come from

A sample covariance over fewer periods than factors is singular, and over barely more is
nearly so. That is precisely the construction `RiskConfig.psd_relative_floor` was calibrated
against: the fixtures in docs/CONDITIONING.md estimate a covariance from 30 observations of 40
factors on purpose.

So the estimator reports it at the point of creation rather than leaving it to bite downstream.
Below one observation per factor it says the matrix is singular by construction, names how many
zero eigenvalues that implies, and points at `psd_relative_floor`. Between one and three it says
the matrix is poorly conditioned and its smallest eigenvalues are mostly estimation noise.

## Verified statistically, not against a stored draw

A single realisation cannot distinguish a real bias from sampling noise. On the first seed tried
here, two of three recovered betas sat about two standard errors from truth and all three factor
volatilities came out 7 to 12 percent low -- which looks exactly like a broken estimator and was
not.

So the tests run a Monte Carlo and check the mean estimate against the theoretical standard
error of the mean: betas unbiased, factor variances unbiased, specific risk recovering the
residual volatility, and the spread of the betas matching
`sigma_resid / (sigma_factor * sqrt(n))`. That last one is the real check on the regression: if
the spread is wrong, the regression is wrong whatever the mean does.

## Covariance estimators

For a forecast these matter more than the regression does, and the estimator is a parameter:

```python
estimate_timeseries_model(panel, factors, covariance_estimator=ShrunkCovariance(
    ExponentialCovariance(half_life=12)
))
```

The default is the plain sample covariance, so adding them changed no existing caller's
numbers.

**Effective observations is the thread that ties them together.** Equal weights over `n`
periods give `n` effective observations; exponential weights give fewer, by Kish's measure
`(sum w)^2 / sum w^2`. That number, not the row count, determines how much of the matrix is
signal -- so it is what the conditioning judgements use. A 200-period history with a 2-period
half-life has plenty of rows and fewer than ten effective observations, and the row count
would call that matrix healthy.

| estimator | what it decides |
|---|---|
| `SampleCovariance` | nothing: equal weight on every period. Honest and the worst of the three |
| `ExponentialCovariance(half_life)` | which periods to believe |
| `ShrunkCovariance(inner, target)` | how much of the result to believe |

Shrinkage wraps rather than replaces, because shrinking an exponentially weighted covariance
is a thing practitioners do and the two concerns are independent.

### Half-lives, not lambdas

`ExponentialCovariance` takes a half-life in the panel's own periods. RiskMetrics publishes a
daily decay of 0.94 and that convention is hard to read: 0.94 and 0.97 look alike and differ by
a factor of two in memory, 11 days against 23. `.decay` exposes the lambda for comparison with
published figures.

The effective count approaches `(1 + lambda) / (1 - lambda)` and therefore barely depends on
history length past a few half-lives -- a 6-period half-life gives about 17 effective
observations whether the history is 60 periods or 480.

### Shrinkage, and why it earns its place

Ledoit and Wolf's insight is that the optimal intensity can be *estimated* rather than chosen:
the ratio of the sample covariance's own estimation variance to its distance from the target.
A hand-picked intensity is a free parameter nobody can defend.

Measured against a known truth over 200 draws of 10 factors, Frobenius distance:

| n / factors | sample | shrunk | gain | estimated intensity |
|---|---|---|---|---|
| 1.2 | 2.410 | 1.906 | **20.9%** | 0.41 |
| 2.0 | 1.887 | 1.610 | 14.7% | 0.34 |
| 6.0 | 1.108 | 1.032 | 6.9% | 0.16 |
| 48.0 | 0.386 | 0.384 | 0.7% | 0.02 |

Converging on the sample covariance at large samples is correct: with enough data it *is* the
right answer, and the estimated intensity falls to match.

**And it repairs the singularity at source.** With six observations of ten factors the sample
covariance is singular -- smallest eigenvalue -3.5e-16 -- and the shrunk estimate is positive
definite at +0.36. That is the problem `RiskConfig.psd_relative_floor` patches *downstream*,
so a shrunk estimator means the floor never has to fire for this reason. The conditioning
warning is suppressed when shrinkage has been applied, because it has repaired the thing the
warning is about.

Two targets. `DIAGONAL` scales the identity by the mean variance, saying "assume the factors
are uncorrelated and equally volatile". `CONSTANT_CORRELATION` keeps each variance and replaces
every correlation with their average. The diagonal is usually the more honest one for *factor*
returns, since a factor set is typically built close to orthogonal already.

### Two half-lives, not one

`HorizonCovariance` estimates correlations on a long half-life and volatilities on a short
one, then recombines. This is BIM301's design, and Barra's stated rationale is that
correlation structure is more stationary than volatility, and that the correlation matrix has
n(n-1)/2 elements to estimate against the volatilities' n -- so robustness binds far harder
there, and a short half-life on correlations is how you arrive at the spurious structure that
makes a minimum-risk portfolio forecast zero risk and then realise plenty of it.

That is a claim, so it is tested rather than quoted. On a fixture with a stationary
correlation structure and a volatility regime change in the last 60 of 600 periods, 200
replications:

| half-life | effective obs | correlation RMSE | volatility RMSE |
|---|---|---|---|
| 6 | 17.3 | 0.2095 | 0.4561 |
| 12 | 34.6 | 0.1532 | **0.3302** |
| 24 | 69.3 | 0.1251 | 0.3474 |
| 36 | 103.9 | 0.1163 | 0.5033 |
| 72 | 206.5 | **0.1040** | 0.8935 |

Correlation accuracy improves monotonically with half-life; volatility accuracy is U-shaped,
best at 12. The optima are at different half-lives, so a single parameter has to compromise --
and the compromise is expensive in both directions: half-life 72 is nearly three times the
best volatility error, half-life 6 double the best correlation error.

The split does not have to compromise:

| estimator | correlation RMSE | volatility RMSE |
|---|---|---|
| single half-life 12 | 0.1532 | 0.3302 |
| single half-life 72 | 0.1040 | 0.8935 |
| **split, vol 12 / corr 72** | **0.1040** | **0.3302** |

It attains both optima simultaneously, which is the entire reason to estimate one matrix with
two memories.

`bim301_long()` and `bim301_short()` carry Barra's published configurations -- volatility
half-life of one year and correlation of three for the long-horizon model, 90 trading days and
two years for the short. Both take `periods_per_year`, so a configuration survives a change of
panel frequency: BIM301S's 90 trading days is 90 periods on daily data, 18 on weekly and about
4.3 on monthly. Barra calls these typical rather than universal and their component models
vary, so they are starting configurations, not constants.

Two properties worth knowing. Equal half-lives reduce the estimator **exactly** to
`ExponentialCovariance`, pinned by a test, because an estimator whose general case does not
contain its special case is usually wrong in both. And the recombination is safe: a PSD
correlation matrix rescaled by a positive diagonal stays PSD, so the split cannot introduce the
negative eigenvalues `psd_relative_floor` exists to repair.

`effective_observations` reports the **correlation** window's figure, because that is the one
that decides whether the matrix is singular -- the same reasoning that motivates the long
correlation half-life in the first place.

One incidental confirmation. BIM301 notes in passing that "the effective observation window is
three times the half life." The closed form for an unbounded exponentially weighted window is
(1+lambda)/(1-lambda), which tends to 2/ln(2) = **2.885** times the half-life. The Kish measure
here reproduces that to three figures at every half-life tested, which is an independent check
that `effective_observations` means what Barra means by it.

### Newey-West, and what Bartlett weighting costs

The other two covariance corrections assume each period is an independent draw. Returns are
not, and BIM301 applies Newey-West throughout because of it -- serial correlation is
"significant enough to affect model performance, even out to two weeks for highly liquid
developed market stocks."

What goes wrong without it is specific. A risk number is almost always quoted for a horizon
longer than one period, and that conversion multiplies the one-period variance by the number
of periods -- a step only valid under independence. With positive autocorrelation the variance
of a sum grows faster than the number of terms, so annualising understates the risk by exactly
`(1 + rho) / (1 - rho)` for an AR(1). At rho = 0.3 that is a factor of 1.86.

`NeweyWestCovariance` estimates the long-run covariance instead, which *can* be scaled by the
number of periods:

    S = Gamma_0 + sum_j (1 - j/(L+1)) (Gamma_j + Gamma_j')

**Bartlett weighting leaves some of the adjustment on the table, and raising the lag count
does not recover it.** The weights discount every lag, so the expected recovery is
`1 + 2 sum_j (1 - j/(L+1)) rho^j`, not the full long-run ratio. Measured against that
expectation on AR(1) panels of 4000 periods:

| rho | true LRV ratio | L=5 | L=9 | L=20 | L=50 |
|---|---|---|---|---|---|
| 0.2 | 1.500 | 1.391 | 1.422 | 1.402 | 1.360 |
| 0.3 | 1.857 | 1.647 | 1.715 | 1.715 | 1.675 |
| 0.5 | 3.000 | 2.332 | 2.565 | 2.674 | 2.666 |

The measured value tracks the Bartlett expectation closely at small L (within 0.01 at L=5) and
falls increasingly short of it as L grows -- the gap at rho=0.5 widens from 0.01 at L=5 to 0.26
at L=50. That is the bias-variance tradeoff of a kernel estimator: more lags cut the truncation
bias but add estimation noise from high-lag autocovariances that should be zero. Past about
twenty lags the recovery stops improving and starts falling back. Newey and West's own rule,
`floor(4 (T/100)^(2/9))`, is conservative and close to the best available, which is why it is
the default.

**The positive semi-definiteness guarantee is real, and it is conditional.** Bartlett weights
make the equal-weighted estimator PSD by construction, and over 400 random panels its smallest
eigenvalue never fell below +0.08. Combined with exponential weighting the guarantee is gone,
and not academically: on short panels with many factors and a heavy decay -- 25 to 60 periods,
5 to 12 factors, a six-period half-life -- **310 of 400 panels came out indefinite**, the worst
eigenvalue at -0.487. That is a real negative variance forecast, not a rounding artifact. The
combination is still worth having, but `psd_relative_floor` is doing load-bearing work beneath
it, and the estimate says so in its notes.

## The cross-sectional estimator

`crosssectional.py` runs the other direction. Every period, security returns are regressed
*across securities* on their characteristics; the slopes are that period's factor returns.

    time-series        one regression per security, over time
                       factor returns given, exposures estimated

    cross-sectional    one regression per period, over securities
                       exposures given, factor returns estimated

The exposures are not estimated at all. A log market cap is a measurement, not a statistic
with a standard error, and it describes the company as it is today rather than averaging over
whatever it was during a five-year window. A firm that has doubled in size has a new size
exposure this month and a time-series beta that still half-remembers the old one.

But the estimation error does not vanish -- it moves to the factor returns, and that is
worse. A factor covariance built from estimated quantities has an inflated diagonal, and
every forecast built on it inherits the inflation, worst for the portfolios most exposed to
the noisiest factor.

### Standardisation, and the order that makes winsorising work

Raw characteristics are on incomparable scales: log market cap runs 10 to 30, book-to-price 0
to 3. Factor returns estimated against them would be on incomparable scales too, and their
covariance would mean nothing. So each characteristic is standardised cross-sectionally every
period.

The mean is **cap-weighted**, which is what puts the market portfolio at exactly zero
exposure to every style factor. Equal-weighting it leaves the market at +1.36 standard
deviations of size in a six-stock example, because small companies outnumber large ones --
and every portfolio's size tilt would then be measured against a benchmark nobody holds. The
standard deviation is deliberately **equal-weighted** even so: a cap-weighted dispersion is
set by a handful of names and jumps when one of them moves.

Winsorising has to happen on the **raw** values, before the mean and standard deviation are
computed. This was got wrong first time and the error is instructive. Clipping *standardised*
values cannot catch an outlier, because by then the outlier has set the scale it is being
measured against:

| book-to-price | 0.3 | 0.6 | 20.0 | 1.2 | 1.5 | 1.8 |
|---|---|---|---|---|---|---|
| standardised, clip at 4 after | -0.51 | -0.47 | **+2.04** | -0.39 | -0.35 | -0.31 |
| standardised, clip at 4 before | -0.85 | -0.67 | **+1.92** | -0.31 | -0.13 | +0.05 |

The outlier never looks extreme -- it makes everyone else look identical. The raw standard
deviation is 7.74 against a MAD-based scale of 0.89, 8.7x inflated, so the 20 standardises
to about two sigma and sails through any sane clip while compressing the five ordinary
companies into a range of 0.20. Clipping first restores their range to 0.90.

The scale used for the clip is the median absolute deviation, rescaled by 1.4826 so it
estimates the same quantity as the standard deviation on normal data. MAD is used precisely
because one extreme value cannot inflate it, which is what lets the clip bite. A second clip
after standardising is a guard that makes the parameter mean what it says. For the securities
that set the scale it is nearly always a no-op -- 4,021 of the S&P 500 panel's member values
over its 1,214 days, all but nine in a lower tail where a cap-weighted mean above the median
puts the bound inside the raw one: three quarters of them LNCAP's and VLVR's, whose means sit
about 1.7 standard deviations above their medians, and most of the rest RSTR's and HALPHA's. For the
securities put on that scale it is not; see below.

`CharacteristicPanel.winsorised` counts how many raw values moved. A handful means the clip
did its job; a large fraction means the characteristic itself is wrong -- wrong units, a sign
error, a vendor placeholder like -999 -- and no clipping will fix it.

### Whose values set the scale: the estimation universe

Given a membership, every standardisation is taken over that day's index members with a value:
the descriptors' clip bounds, cap-weighted mean and equal-weighted standard deviation, the
second pass over each blended risk index, and LCAPCB's own normalisation. Every other security
with a value is put on the members' scale -- clipped to their bounds, centred on their mean,
divided by their dispersion. That is Barra's estimation universe against its coverage universe.
A name that left the index still gets an exposure, as a holding of it needs, but no say in what
an exposure of one means. The run script builds one membership array and hands it to every
standardisation and every regression, so no two can disagree about who is a member.

What it buys is the index at zero. Standardised over every panel security, the S&P 500 run's
members carried a cap-weighted Size exposure of +0.013 on its last day, at a dispersion of
0.88: the ex-members trading on in the lower tail had widened the scale for everyone. Over the
members, the cap-weighted mean of every style exposure is zero to rounding on every date an
estimated period or the forecast reads -- at most 2e-16 -- and the run checks it each time.

The outsiders sit in the tails, so many hit a bound: 102,179 of their values over the panel.
`CharacteristicPanel.clipped_outside` counts them, by either bound, since for LNCAP the second
one binds: 19 of the 45 ex-members with a price on the last day hit it, one of them the raw
bound too.

### Each regression is fixed the day before its return

A characteristic dated t is computed from day t's close, and day t's return ends at that close,
so regressing one on the other puts the return inside its own exposure. Under `lag=1` -- the
estimator's default and Barra's convention -- period t's regression reads its exposures, their
standardisation, its weights and its membership from the shared date before, and only its
returns from day t. The forecast exposures are still the last date's: they are as of today and
forecast forward. `regression_design` is the one place each period's inputs are assembled, for
the estimator, for `regression_universe`, and for the test that guards it: price every security
on one day by its own random factor, rerun from prices with the blend's weights held fixed, and
that period's exposures, weights, membership and regression universe come back bit for bit
while every one of its returns moves. On the real panel too -- and at a lag of zero the same
check fails, which is what makes it a guard.

Same-day was not a small error on the S&P 500 run. Momentum earned +26.8% a year (t = 9.1),
because RSTR holds today's return: the exposure moves 3.2 standard deviations per unit of the
day's return. Read the day before, it earns +2.4% (t = 0.8). Mean R-squared falls from 0.312 to
0.294 and the Volatility factor's volatility from 9.4% to 7.6%, since HILO, DASTD and CMRA hold
today's move as well; the specific risk model's Volatility coefficient halves, +0.42 to +0.19,
for the same reason -- today's range predicting today's residual.

Size went the other way, and the decomposition says why. On its own, same-day Size is inflated
exactly as expected -- today's cap holds today's return -- at +8.3% a year against +4.1%
lagged. But Momentum's exposure moves nearly four times as far with the day's return and is
+0.38 correlated with Size, so in the joint regression it takes the look-ahead and over-corrects
Size: +0.05% a year same-day, +4.9% lagged.

SERDP exists at the last date alone, so under the lag it could enter no regression, only the
forecast Volatility exposure -- where it moved the members' exposures by 0.27 rms, the factor
read on a blend it was never estimated on. It is in no blend now, so every forecast exposure is
built from the descriptors the last estimated period's were, and the run stops if a descriptor
contributes to a forecast exposure and to no estimated period's. Computing SERDP historically
waits for a reason to; it has never had a usable period in the screen. The run also prints each
index's blend over the estimated periods, with the date each descriptor enters. Under the
828-day market model the run once used, BTSG, HALPHA and VLVR entered only on 2025-04-23, 386
of 938 periods, so for the first 552 Volatility had no beta; under USE4's 252 days, the default
now, every descriptor is in every estimated period, and the run holds each factor to one blend
across the sample -- see below.

### One definition per factor across the sample

`blend_to_index` averages whichever of an index's descriptors a security has, so an index whose
blend gains a descriptor partway through the sample is two factors under one name, estimated on
one and forecast on the other. The run estimates a period only if every risk index's blend in it
-- over the securities its regression holds -- has exactly the descriptors the forecast's has,
and afterwards asserts it of every estimated period, by the estimator's own test of who each
regression held; one that fails stops the run. `estimate_cross_sectional_model` and
`regression_universe` take the rule as `periods`, one boolean per date: a period ruled out is not
estimated whatever it holds, and the notes name it apart from the thin ones.

On the S&P 500 run the rule drops 24 periods, 2023-01-04 to 2023-02-07. HALPHA gives Momentum a
value 24 days before RSTR's twelve-month window fills, so in them Momentum is HALPHA alone,
Volatility has no CMRA and Trading Activity no STOA or VLVR. The estimation starts on 2023-02-08,
the same 938 periods as before; each one's factor returns are what they were, bit for bit, and
the regression universe loses VNO, regressed in the 24 alone. Under the 828-day window the rule
would leave 386 periods. `--mixed-blends` estimates every period as before; then only a
descriptor in the forecast exposures and in no estimated period's stops the run, and the periods
blended otherwise are flagged.

`fit_index_weights` now pairs each descriptor with the next period's return, as the regressions
read their exposures. Same-day, its Momentum fit on the S&P 500 panel gave RSTR +2.41 and HALPHA
-1.41; lagged, -0.63 and +1.63. It still pools every period's returns into one regression:
whole-sample weight selection, look-ahead for every period but the last, which its docstring
says and the look-ahead test's third case records. The run script does not call it -- its blends
are equal-weighted -- and the look-ahead test holds fitted weights fixed for that reason.

### Weights: two different arrays, easily confused

`regression_weights` is for the regression, and Barra uses the square root of market cap, on
the argument that residual variance scales roughly as the inverse of size. `standardise_weights`
is for the mean, and should be market cap itself -- the point of a cap-weighted mean is to put
the *market portfolio* at zero, and the market portfolio is weighted by cap, not by its root.
Passing the same array to both is a quiet error that nothing downstream would flag.

### The unit column changes meaning

In the time-series model the intercept is alpha, and alpha is emphatically not an exposure.
Here the unit column is the **market factor** and it *is* an exposure, because every security
really does have unit exposure to the market. Same column of ones, opposite meaning, opposite
treatment.

### Specific risk is degrees-of-freedom corrected

A Monte Carlo found a real bias here. Residuals from a fitted regression are systematically
too small: the fit absorbs `rank` of the period's `n` observations, so the residual variance
understates specific variance by `(n - k) / n`. Measured against a known truth of 0.0600:

| securities | factors | sqrt((n-k)/n) | uncorrected | corrected |
|---|---|---|---|---|
| 200 | 4 | 0.990 | 0.05936 | 0.05996 |
| 80 | 4 | 0.975 | 0.05848 | 0.06000 |
| 40 | 10 | 0.866 | 0.05207 | 0.06013 |
| 25 | 10 | 0.775 | 0.04653 | 0.06007 |

Both columns are measured, the uncorrected one by an independent plain-OLS computation rather
than by dividing the corrected figure back out.

At 200 securities on 4 factors the bias is one percent and ignorable. At 25 on 10 it is 22
percent, and specific risk is the dominant term for a concentrated portfolio -- so the
forecast was understated exactly where it mattered most. The correction is applied period by
period, because `n` varies.

### The estimation-noise diagnostic

The sample covariance of estimated factor returns estimates the true factor covariance *plus*
the average sampling variance of the estimates. For WLS that sampling variance is
`sigma^2 (X'WX)^-1`, computed per period and averaged, and `CrossSectionalNotes.noise_share`
reports it as a fraction of the observed variance.

Validated over 24 replications, as `(observed - realized) / predicted`, which should be 1.00
for a diagnostic that measures what it claims:

| securities | mean ratio | std err | mean noise share |
|---|---|---|---|
| 40 | 0.963 | 0.031 | 21.3% |
| 80 | 1.051 | 0.058 | 12.5% |
| 200 | 1.029 | 0.068 | 5.8% |
| 800 | 0.949 | 0.121 | 1.6% |

Every ratio is within two standard errors of one, and the share roughly halves as the
cross-section doubles, as `sigma^2/n` predicts.

One structural point worth noting: because sampling variance is roughly constant across
standardised factors, it is a larger *fraction* of a small factor's variance. The noise
concentrates on the lowest-variance factor -- the same shape as the time-series intercept bias
landing hardest on HML rather than on the market.

By default the bias **is not subtracted off**. The naive correction can drive a variance
negative, and flooring that silently would turn a diagnostic into a fabrication; Barra's own
adjustment is more careful than a subtraction. It is reported, and reported loudly past 25
percent.

`sampling_correction=True` (the run's `--sampling-correction`, off by default) makes the
subtraction, and says what the repair after it did (midoffice10):

- **Weighted as the covariance is.** Each period's full `sigma^2 (X'WX)^-1`, weighted by what
  the covariance estimator carries of that period's noise. The sample, exponential and
  Newey-West estimators are each a quadratic form `R' K R`, so independent noise of covariance
  `S_t` adds `sum_t K_tt S_t` to the estimate, and `K_tt` is the estimate of a series one in
  period t and zero elsewhere (`crosssectional.sampling_weights`). Newey-West at 10 lags and a
  90-day half-life carries 0.9625 of it: centring on the weighted mean makes a period's noise
  correlate slightly negatively with its neighbours'. The two-half-life and shrunk estimators are
  not quadratic forms, and are refused.
- **The repair.** The subtracted matrix goes through `nearest_psd`. The notes keep its diagonal
  before the repair, and the qualifications name every factor it left at or below zero.
- **Verified** over 12 replications: (corrected - realized) / predicted averages zero, where the
  uncorrected diagonal's averages one.

`noise_share` divides the equal-weighted mean of the sampling variance by a variance weighted
toward the recent periods; `weighted_noise_share` is the consistent ratio, what the correction
subtracts. On the S&P 500 model they differ by about half again -- Wholesale 46.2% and 68.8%,
Size Nonlinearity 46.6% and 67.4% -- because every factor's sampling variance in 2026 is 1.3 to
2.0 times its 2023-2026 mean, the cross-sections steady at about 500: residual dispersion rose,
and a 90-day half-life sits on 2026.

There, the subtraction leaves 9 of the 25 eigenvalues negative, the most negative -0.0070 a
year, in directions that are contrasts of the thin industries: Finance & Real Estate with
Construction, Wholesale against both, SIC20, Services against SIC67. That is the estimator's own
sampling error. Drawn from a truth comfortably above the noise in every direction and estimated
the same way, 938 periods leave a median of 7 negative. Clipped to the floor, those directions
are close to riskless: a portfolio in one has almost no factor risk. Broad portfolios move
little -- the index -1.1%, the equal-weighted index +0.8%, the smallest 85 -0.6%, the largest 50
-1.0% -- and the thin industries' own portfolios fall 9 to 18%.

**Thin industries, folded.** `refdata.industries.merge_industries` (the run's `--merge-industry
A=B`) folds an industry into another once the scheme is built, its members labelled B. A chain
resolves to its end, and the folds are recorded in `folded`, apart from the grouping's own
`merged`. Measured in midoffice10, each industry under 15 members folded into the industry its
factor return correlates with most over the estimated periods:

| Fold | mean R-squared lost | members' specific risk, median change |
|---|---|---|
| Construction (8) into Retail | 0.0030 | +12.4% |
| Wholesale (12) into Retail | 0.0018 | +1.6% |
| Finance & Real Estate (12) into SIC62 | 0.0024 | +9.4% |

Each loses about what a regressor of pure noise adds, 0.0016 to 0.0026 (Size Nonlinearity,
below). Folding all three, a floor of 15, leaves the correction 6 negative eigenvalues where it
had 9, the most negative -0.0031. Floors of 20 and 25, applied the same way, fold industries
that carry more -- SIC37 0.0053, SIC60 0.0063 -- and take mean R-squared to 0.2769 and 0.2628.
Construction is two industries under one division: its homebuilders move with Retail, its
contractors (EME, FIX, PWR) lately with SIC36.

### What it flags rather than fixes

* **Collinear designs.** Industry dummies sum to the unit column, so a design with both has
  no unique solution. The rank deficiency is detected, the period named, and `lstsq` returns
  the minimum-norm solution -- *a* solution, but not one with an economic reading. A real
  Barra model imposes a cap-weighted zero-sum constraint on the industry returns. Supply
  industries without a market factor, or accept the flag.
* **Modelled specific risk.** The sample standard deviation of a residual series is noisy for
  a short history and says nothing at all about a security with none. Barra regresses specific
  risk on characteristics so a new issue inherits a sensible forecast. The notes name the
  securities that got nothing.
* **Thin cross-sections.** `MIN_SECURITIES_PER_PERIOD` is 30, and unlike the time-series
  minimum it bites hard: with ten factors and twelve securities the factor returns are fitted
  through nearly every point and specific risk collapses for every security at once.

## Specific risk from characteristics

`specificrisk.py` implements Barra's EQ 3-12 to 3-14. The sample standard deviation of a
residual series -- what `crosssectional.py` produces on its own -- is named in the handbook as
the naive approach, and it has two failures: it is noisy on a short history, and it says
*nothing at all* about a security with none.

    sigma_it = kappa_it * (1 + V_it) * S_t          S_t = alpha + sum_j beta_j S_{t-j}
                                                    V_it = sum_k Z_ikt gamma_k

Three pieces, three different questions. **S** is the market-wide average absolute specific
return, forecast from its own recent history -- the part that moves together, which nothing
built from one security's residuals can see. **V** is how much riskier this security is than
that average, regressed cross-sectionally on characteristics -- the part that gives a new
issue a forecast, because it has characteristics even with no returns. **kappa** converts an
absolute-return forecast into a standard deviation.

Everything is built on absolute returns rather than squared ones, and the handbook gives the
reason in two words: "for robustness reasons."

### kappa has to be calibrated on normalised residuals

For a normal distribution E|X| = sigma*sqrt(2/pi), so kappa = sqrt(pi/2) = **1.2533**, and
recovering that from simulated normal residuals is the check that the three multiplied factors
are identified rather than absorbing each other's scale.

The first implementation calibrated kappa by comparing a security's realised standard
deviation against its mean absolute forecast directly, and a failing test caught the problem:
a quadratic measure and a linear one respond differently to a change in the level, so a window
spanning a regime shift biased kappa upward by 15%. Dividing both sides by `S_t` first removes
the level path entirely, which is right in principle -- kappa describes the *shape* of the
distribution, not its size:

| level shift in last quarter | kappa | vs sqrt(pi/2) | forecast | true |
|---|---|---|---|---|
| none | 1.2445 | -0.7% | 6.12% | 6.00% |
| 2x | 1.2445 | -0.7% | 12.24% | 12.00% |
| 3x | 1.2445 | -0.7% | 18.36% | 18.00% |
| 5x | 1.2445 | -0.7% | 30.60% | 30.00% |

USE3 goes further and fits kappa as a piecewise-linear function of relative capitalisation,
calibrated to zero the average bias within each capitalisation decile -- doing two jobs at
once, converting units and absorbing the size-related bias that remains because small
companies' specific returns are further from normal. `capitalisations` enables that; without
it kappa falls back to the normal constant and the notes say so.

### The linear form runs past its own validity

EQ 3-14 makes V linear in the characteristics, so `1 + V` turns negative for a security far
enough into the tail -- with a coefficient of 0.3, anything past -3.3 standard deviations.
Unfloored, that security is forecast to have **negative** specific risk, and flooring at zero
merely renames the problem: a portfolio concentrated in it would be reported as carrying only
factor risk. `forecast` floors at a tenth of the market-average specific risk and `floored()`
names what it caught, because a security at the floor has a forecast the model did not make.

### The model does not beat a sample standard deviation, and that is the point

This was worth measuring rather than assuming, and the measurement reversed the expected
conclusion. The characteristic model's error is **irreducible** -- it is the cross-sectional
variation the characteristics do not contain, and more history does not shrink it. A sample
standard deviation's error is **statistical**, falling as 1/sqrt(2(n-1)). So the sample
estimate overtakes the model as soon as there is enough history: on a panel where the
characteristics explain 20% of the cross-sectional variation, a plain 15-period trailing
standard deviation wins comfortably.

What the model does have is robustness to a shift in the level, and coverage of securities
with no history at all. Forecasting the next period after a 3x level shift, 60 periods of
history:

| estimator | stable level | 3x shift |
|---|---|---|
| characteristic model | 0.269 | 0.298 |
| sample sd, full 60 periods | **0.096** | 0.456 |
| sample sd, trailing 15 | 0.197 | 0.197 |
| **precision-weighted blend** | 0.163 | **0.164** |

(RMSE relative to the mean true specific volatility, 12 replications.)

So the answer is to use both, which is what BIM301's footnote says its specific risk models
do -- "combining assets' specific return history with other information." `blend` weights them
by inverse variance in the relative space they share:

    v_model = relative_residual_sd^2        (irreducible)
    v_hist  = 1 / (2 (n - 1))               (statistical, per security)
    w_model = v_hist / (v_model + v_hist)

which has the right limits without being told them: a security with no history gets the pure
model, one with a long history gets its own sample estimate, and better characteristics earn
more weight because `relative_residual_sd` is smaller. The blend beats both of its inputs, and
is the only one of the three that is good in every regime.

`relative_residual_sd` is measured across the securities' average relative residuals, by
default as 1.4826 times their median absolute deviation from the median -- the scale the
winsorising clip uses, for the same reason. Measured by their standard deviation, one security
far enough out sets it for every blend weight: on the S&P 500 run FRC, collapsing while still
in the index, is 72% of its square, and starting the sample 24 days earlier moved it from 0.361
to 0.281. The MAD-based spread is 0.1333 on the 938 periods and 0.1334 on the 962. At the
median history of 938 periods the characteristics' weight in a blended forecast rises from
0.41% to 2.92%, and the forecasts delivered move by 0.68 points rms, their median from 23.40% to
23.77%; the largest move is SBNY's, 69.5% to 59.5%. Between the two samples the delivered
forecasts move by 0.46 points rms under the MAD and 0.81 under the standard deviation.
`residual_spread="sd"` -- `--residual-spread sd` -- keeps the standard deviation, and the notes
carry both. The scale is settled (the ruling of 2026-11-09: the standard deviation was FRC's);
how much weight a security's history should get against the model is a different question, for
out-of-sample bias statistics to settle.

### Whose history: every covered name's specific returns

A security's own history was its residuals in the regressions that held it, so a joiner's
history began the day it joined and a leaver's stopped the day it left, and a security in no
regression had none: a fallback. `specific_history="covered"` -- the S&P 500 run's default since
midoffice09 -- adds, in each estimated period, the specific return of every security with a
return and a complete characteristic row that the regression did not hold: its return less its
exposures times the members' factor returns, with no degrees-of-freedom inflation, since no fit
absorbed it. Factor returns and the specific risk model's fit -- V, kappa, the relative residual
spread -- stay the regression's alone, unchanged to the bit; only the histories lengthen. A
joiner arrives with its history from before it joined, a leaver keeps forecasting from its own
history after, and a security in no regression with a history of its own is blended, not a
fallback.

On the S&P 500 run: 55,830 specific returns measured outside the regressions; all 11 fallbacks
blended with histories of their own; for the 21 securities in their first year as members, the
characteristics' weight falls from a median 16.4% to 2.9%; 108 of 551 forecasts move. The
largest is Signature Bank's, from 58% to 396%: its history after its failure is a stock trading
at fractions of a cent. The history is still equally weighted over the window, so a joiner's
oldest volatility counts as much as its newest.

## Descriptors: building the exposures instead of assuming them

`descriptors.py` implements Appendix A of the US-E3 handbook: the raw measurements underneath
the risk indices, from price, volume and shares outstanding alone. Fifteen descriptors across
five of the thirteen risk indices -- Volatility, Momentum, Size, Size Nonlinearity and Trading
Activity -- with no fundamental data required. The other eight indices need balance sheet and
earnings data and are not built.

Each is checked against its published definition computed independently, because the only
value in implementing a specification is that it matches: LNCAP, LPRI, HILO, CMRA and RSTR
come out exact, the turnover descriptors land on 0.96 for a stock turning over 8% a month, and
BTSG recovers beta times residual volatility with a mean ratio of 0.99.

Three details that are easy to get wrong and are pinned by tests:

* **LCAPCB is the cube of the *standardised* log capitalisation**, not of the raw one.
  Appendix A says "normalized", so the standardisation is part of the definition -- and given
  a membership it is taken over the members, like every other. The cube is not affine, so
  where it is centred changes its shape and not just its scale.
* **BTSG is set to zero when beta is negative**, which the definition states explicitly.
* **SERDP comes out a few percent below its AR(1) theoretical value**, consistently across
  rho. That is the definition rather than the implementation: it is built on *fitted*
  market-model residuals, and OLS forces those orthogonal to the regressors.

### A window with no data gives no value

RSTR, CMRA and the turnovers sum over their windows, and a sum over an empty window is zero.
Every day before a security's first price -- a current ticker carried back to before it traded
-- read as no momentum and no trading: 94,882 values on the S&P 500 panel, 10,965 of them on
index members. The window descriptors are now missing wherever their window holds no valid
observation, which on the S&P 500 panel changed those 94,882 zeros, checked value by value, and
nothing else; the model barely moved, mean R-squared by -0.0001.

`min_coverage` extends that to windows holding less than a given fraction of valid data, and
its default is 0.5: a twelve-month descriptor computed from under six months of its window is
not that descriptor. On the S&P 500 panel the rule removes 29,506 values, 7,086 of them on
members, for mean R-squared +0.000014 and Trading Activity's volatility -5.9%; ECHO and VMRK,
two phantom identities, lose their only complete row. 0 is still available. The run checks,
against each descriptor's own inputs counted independently, that no value was computed from an
empty window or from under the rule's fraction of one.

### USE3 by default, USE4 where USE3's choice measures badly

The policy for the definitions: USE3's by default, and USE4's wherever USE3's choice produces a
measured problem on this panel, each kept switchable.

**The market model is USE4's.** BTSG, HALPHA, SERDP and VLVR's residual sd come from one
regression on the market, and on a daily panel `compute_descriptors` now fits USE4's by
default -- 252 trading days, each day's weight halving every 63 back from the window's end --
and the run no longer overrides it. USE3's 60 months are 1,380 days, more than this panel's
1,214, so they would fit once, on the last date; the 828 days (36 months) the run used instead
kept the three blended descriptors out of 552 of its 938 estimated periods, beta among them,
the descriptor the screen ranks first. Under USE4's they have values from 2023-01-03 and VLVR,
held back by its twelve-month dollar volume, from 2023-02-07, so all three are in every
estimated period. On those 938 periods mean R-squared rises from 0.2942 to 0.2994 and the
Volatility factor's volatility from 7.63% to 8.63%, while Momentum's falls from 7.16% to 6.57%
and Size's from 7.57% to 6.90%; the 252-day window does most of it, the half-life adds +0.0007
of R-squared. `--beta-window 36 --beta-half-life inf` runs the 828 equally weighted days and
reproduces the old run bit for bit. On a panel of any other frequency the default stays USE3's
60 months, equally weighted: USE4's beta is a daily estimator.

**RSTR's skip stays 0.** `rstr_skip` cuts that many trading days from the end of RSTR's
twelve-month window -- the academic twelve-minus-one reading; the window keeps its start, so
RSTR has a value from the same date. On the S&P 500 run, cutting 21 raised Momentum's mean
return from +2.44% to +3.23% a year (t 0.81 to 1.09) and moved its volatility from 7.16% to
7.11%, nothing else material, and mean R-squared fell by 0.0006. USE4's lag is the other
reading, a full twelve months ending 21 days back; compared on the same cells it moved
Momentum's mean from +1.85% to +1.75%. It is not built.

### Size Nonlinearity, USE4's way

USE3's Size Nonlinearity is LCAPCB, the cube of the standardised log cap, and on the S&P 500 run
it is 0.933 correlated with Size across the members: the regression splits one size effect
between two collinear exposures, and their factor returns come out -0.82 correlated. The raw
clip every descriptor gets makes it worse. On a cube it binds at |z| of about 1.2 -- the median
|z| of a normal, 0.6745, cubed is 0.31, four times 1.4826 times that is 1.82, the cube of 1.22 --
and it moved 20% of the members' LCAPCB values, flattening the tails the cube is there to
measure.

`build_risk_indices` now builds it USE4's way, `size_nonlinearity="use4"` by default: the Size
exposure cubed; standardised over the estimation universe -- cap-weighted mean,
equal-weighted standard deviation -- and clipped at +/-4, without the raw clip, which moves 1%
of member values; orthogonalised against Size, date by date, by a fit on Size and a constant
weighted by the regression weights over the members; and standardised again, exactly. The run
checks that on every date an estimated period or the forecast reads, the regression-weighted
covariance of the two over the members is zero to rounding -- at most 1.0e-15 -- and stops if
it is not. LCAPCB is still computed and screened, and is in no blend; "use3" builds the raw cube
as before, bit for bit, and "none" leaves Size Nonlinearity out, the model with Size alone --
the table's third column, bit for bit. On the 938 periods:

| | USE3's raw cube | USE4's | no Size Nonlinearity |
|---|---|---|---|
| exposures' correlation, equal-weighted (median date) | +0.933 | +0.551 | |
| exposures' correlation, by the regression weights (last date) | +0.911 | 0.000 | |
| factor returns' correlation, in the covariance | -0.824 | -0.308 | |
| Size: vol, mean a year | 6.90%, +4.69% | 4.03%, +2.86% | 4.03%, +2.89% |
| Size Nonlinearity: vol, mean, noise share | 6.38%, -2.24%, 18.6% | 2.54%, +0.40%, 41.8% | |
| Size against no Size Nonlinearity: correlation | 0.7935 | 0.9998 | |
| the pair's risk, smallest 85 members | 12.16% | 11.94% | 12.52% |
| the pair's risk, largest 50 members | 3.45% | 0.38% | 0.05% |
| the pair's risk, the index | 0.01% | 0.01% | 0.01% |
| mean R-squared | 0.2994 | 0.2981 | 0.2938 |

With the cube orthogonal, Size's factor return is what it is without Size Nonlinearity in the
model -- their difference has a volatility of 0.06% a year, against 3.8% with the raw cube --
and the largest 50 members, which hold no size bet to speak of, carry 0.38% of size-pair risk
where the raw cube gave them 3.45%. What is left of Size Nonlinearity is small and noisy: a
volatility of 2.54%, 42% of its variance the sampling error of its own estimation. Equally
weighted the two exposures are still +0.55 correlated; the orthogonality is in the regression's
weights, which are the ones its fit uses, and that is the sense intended: it is the inner
product the regression uses. The run prints the three portfolios' exposures and the pair's risk
each time.

It stays in the model (the ruling of 2026-11-09). Its 0.0043 of mean R-squared is more than a
regressor of pure noise adds -- (1 - R-squared) / (n - k - 1), roughly 0.0016 to 0.0026 on about
450 names, as the weights cut the effective count -- so it carries some signal, if not much.
Its noise share is Construction's and Wholesale's problem too, and gets one model-wide answer
after the rebuild rather than a deletion per factor.

Measured on the panel on securities (midoffice10): weighted as the covariance weights the
periods, 67.4% of its variance is noise, and the sampling-error correction takes its volatility
from 2.32% to 1.71% (1.32% before the repair). Without it, mean R-squared falls by 0.0036, and no
reference portfolio's risk moves by more than 0.07 of a point -- the smallest 85's from 13.06% to
12.99%. Folding the thin industries leaves its noise share where it was: 62% to 67% at every
floor.

### Two price bases, and the trap between them

Returns need split-adjusted prices, so a four-for-one split is not a -75% day. Three
descriptors need the opposite -- what the stock actually traded at:

* **LNCAP**, because market capitalisation pairs price with a *point-in-time* share count.
  An adjusted price against a pre-split share count undercuts by the split factor.
* **LPRI**, because its economic content is that low-priced stocks are riskier. A stock that
  traded at $400 before a four-for-one split reads as $100 adjusted, and LPRI would record it
  as cheap in a year when it was not -- an error concentrated in the companies that split,
  which are the ones that went up.
* **VLVR**, whose numerator is the dollars that actually changed hands.

`SecurityPanel` therefore carries both, and omitting `unadjusted_prices` is allowed but
reported rather than silent.

This is not hypothetical. Polygon's own `market_cap` field makes exactly this mistake at a
split boundary: for AAPL at 2020-06-30 it reports $5.99tn, which is the unadjusted pre-split
price of $364.80 multiplied by the *post-split* share count of 16.4bn. Four times the true
figure, from mixing two bases in one product.

A panel's own share counts make the same mistake after every split unless it is built not to.
The first S&P 500 panel, on tickers, sampled each count every 252 trading days -- six dates over
the window -- and carried it forward, so until the next sample its capitalisation paired a
post-split price with a pre-split count: 4,196 member-days across 37 splits, measured in
midoffice06 (the eight spin-offs Polygon records as fractional splits left out: their counts do
not move). Its volumes made the mistake the other way. They were the adjusted bars', so before
every split, and every spin-off, its turnover and dollar volume read the adjustment too high:
24,386 member-day cells over 39 tickers, by a factor of 0.2 to 50.

The panel on securities (below) applies each split to the carried count on its execution date,
from Polygon's split events -- and not every event is a split. Polygon records a spin-off as a
fractional ratio that adjusts the parent's price history -- GE's 1.281 for GE HealthCare,
DuPont's 2.39 for Qnity -- and the parent's count does not move across it. So each event is
classified first (`rebuild.check_splits`), by its ratio and two tests. The ratio is whole numbers
or not: no term over 100 in lowest terms, so DuPont's later three-to-one consolidation, 0.33, is
whole, and its counts follow it. The counts either side move by the ratio, within 10%, or stay
within 10% of unchanged. And the raw price gaps by it on the execution date: the open over the
previous close, within the same of the ratio's reciprocal. A whole ratio the counts follow or
the price gaps by is a split; a fractional one the counts leave unchanged is price-only, which
the adjusted prices carry and the count does not; a test disagreeing stops the build. Polygon's
counts lag a split by up to five months, so a sample on or after one still at the pre-split
count is stale and does not reset the carried count; the first that follows the ratio does. The
guard is exact: no day on or after a split carries a pre-split count -- one carried before it,
or a stale sample's -- and the execution date carries the count before it times the ratio. Over
the window's 55 events: 47 splits and 8 price-only, none contradicting; 15 stale samples read
past; the guard checks 25,846 security-days, 25,260 of them member-days, and finds none. And the
panel's volumes are the raw bars', so turnover pairs the shares traded with the shares then in
existence.

### A frequency mismatch worth naming

USE3 mixes frequencies -- DASTD is 65 daily returns, RSTR is 12 monthly ones. A monthly panel
has no equivalent of 65 daily returns, and mapping the span naively gives a three-observation
standard deviation, which is noise: its correlation with true volatility is 0.69 against 0.94
for a twelve-month window. So the default window adapts by frequency, and anything short
enough to be a poor substitute is named in the notes. A longer monthly window measures
something real, but it is no longer DASTD -- it is a slower descriptor wearing the same name,
and the notes say so rather than letting the name imply otherwise.

### The one thing not reproduced

Appendix A states it plainly: *"The method of combining these descriptors into risk indices is
proprietary to BARRA."* The descriptor definitions are published; the weights are not.

So this stops at descriptors. `blend_to_index` equal-weights by default -- a stated convention
rather than a guess at theirs -- and renormalises over whichever descriptors are available,
since HALPHA needs sixty periods where RSTR needs twelve and an exposure from one descriptor
beats a hole. `fit_index_weights` estimates the weights the way the handbook describes the
real procedure, by explanatory power, and recovers a known blend to within 0.08 on synthetic
data.

That recovers Barra's *method* without claiming to recover their numbers. A model built this
way is a reimplementation of a published methodology, not a clone of USE3, and it should be
described as one.

### Why there are two standardisations, not one

Descriptors are not factors. RSTR and HALPHA never enter the regression; Momentum does. The
collapse from roughly sixty descriptors to thirteen risk indices is what keeps the factor
covariance estimable -- 115 factors would carry 6,555 free covariance parameters against 68
factors' 2,346 -- and what stops two measurements of the same thing splitting their shared
signal between two unstable coefficients. USE3's handbook records that its predecessor had
risk indices sharing descriptors, which "resulted in a certain amount of unnecessary
multicollinearity", and that separating them gave "more precise factor return estimates
(smaller standard errors, higher t-stats)".

The consequence is a scaling trap. A weighted average of standardised descriptors has a
standard deviation *below* one unless they are perfectly correlated -- 0.707 for two
uncorrelated, 0.894 at a correlation of 0.6 -- and the shrinkage grows with how many are
blended. Measured on a real descriptor panel, before the second standardisation:

| risk index | descriptors | scale |
|---|---|---|
| Size | 1 | 1.000 |
| Size Nonlinearity | 1 | 1.000 |
| Momentum | 2 | 0.834 |
| Volatility | 6 | 0.471 |
| Trading Activity | 5 | 0.382 |

A one-sigma move in the underlying characteristic would register as 1.00 in Size and 0.38 in
Trading Activity, the factor returns would be scaled inversely, and the covariance between
them would mean nothing -- the exact failure standardising exists to prevent. So the blended
index is standardised again, which is what Barra's own normalisation formula describes: it is
written for the *risk index*, not for the descriptor. `build_risk_indices` does both passes
and returns exposures that drop straight into
`estimate_cross_sectional_model(..., standardise=False)`.

## The S&P 500 panel on securities

The first S&P 500 panel named each security by a ticker: the index on its first day, each
ticker asked for its prices over the whole window. A ticker is not a security. That panel's META
was the Roundhill Ball Metaverse ETF until 2022-01-28 and Meta Platforms from 2022-06-09; its BNY
a municipal bond trust until 2026-02-06; its INFO an ETF from 2024-10-10, two years after IHS
Markit left. A member renamed since -- Ball as BLL, Marsh McLennan as MMC, Fiserv as FI -- had no
price before the rename, and the index's later joiners, 86 tickers with 46,625 member-days, were
not in it at all. The rebuild (midoffice05 to midoffice08) replaces it with a panel whose columns
are securities. Figures elsewhere in this document are the old panel's, as measured when they
were written.

### The identity table

A security is a share class, keyed by its composite FIGI, else its CIK; the CIK is the
cross-check, since GOOGL and GOOG are one CIK and two FIGIs, and both members. `refdata.identity`
works out who traded under each ticker on each date from three sources, none of them complete:
the reference records the panel fetch cached, Polygon's ticker events, and the prices.
Member-days neither source names went to the reference endpoint on stated public-record
hypotheses, and only what it confirmed entered. The rebuild's own evidence then corrected the
table: a span starts at its first price; a span with no price, or one every record under its
ticker refutes, goes; a reorganisation is dated by the identifier the table keys on; and a span
the endpoint names at both ends, its prices unbroken, joins its security. Four guards stop the
build: every priced (ticker, date) maps to one security; no security trades under two tickers on
one date; no ticker names two securities on one date; every member-day is mapped or named.

Three kinds of boundary are kept apart. A splice is two unrelated securities under one ticker
(BNY, DOC, INFO, META, COHR): two columns, each cut to its own spans. A reorganisation is one
security under a new identifier, the prices unbroken -- a holding-company exchange (BLK, XOM,
OKE, J, LH), a redomicile (Bunge), Carnival's unification, Vylor's two FIGIs -- and is one column
with one return series, keyed by its successor. A merger's predecessor (WestRock, Paramount) is
another security, linked only for membership.

Membership is membership() read through the table's renames, in today's tickers, and mapped to
securities, so EchoStar's 1,057 phantom days as ECHO, before it joined as SATS, are nobody's.

### The gate

`scripts/fetch_sp500_rebuild.py` fetches what the table plans, resumably and through the cache:
each ticker's bars, adjusted and raw, its split events and its cash distributions; and a share
count at each span's first and last day and every 63 trading days, inside the security's own
spans only. Then it checks two things before anything is built. Every span's first sample with
an identifier must name the security the table expects -- and a record naming another composite
FIGI under the expected security's own share-class FIGI names it: a check that reconciles the
record, not a re-keying (COHR's change-day record, Coherent Corp.'s share class under a FIGI seen
nowhere else). And every split event is classified, as above. On the 628 spans of the 600 member
securities, all pass, one by its share class; 47 splits, 8 price-only, no contradiction.

### The build

`scripts/build_sp500_securities.py` builds the panel (`refdata.securities`), offline once the
cache holds the fetch:

- **Prices, stitched.** Within a run of one ticker's bars, the ticker's adjusted closes times
  one scale for the run; across a change of ticker, the raw closes' return times any split
  between, each earlier run scaled to continue the later. Polygon adjusts each ticker's history
  for that ticker's events, a later holder's included -- Paramount Global's under PARA carries
  two reverse splits of whoever took the ticker after it, a constant 200 -- so two tickers'
  adjusted closes need not share a basis. 26 changes of ticker are bridged so. On the 581,521
  cells the old panel and the new take from the same bar, the closes are identical and the
  returns agree to 2.2e-16.
- **Total returns.** Every cash distribution under the security's tickers inside its spans,
  regular and special, reinvested on its ex-date on the prices' split basis -- the ex-date's
  adjusted over raw close. 8,229 of them, TEL's duplicate of 2024-11-22 kept once. The market
  series, SPY, likewise: 1.31% a year above its price return.
- **Volumes**: traded, the raw bars'. The adjusted bars' are kept beside them, for comparison.
- **Share counts**, as the split paragraph above has them, carried forward from each sample and
  never back, across a security's own spans and across a reorganisation. Three rulings
  (midoffice08's) correct what Polygon's records get wrong, each with a later sample: hindsight
  in cleaning the data, not in what the model knows, since the true counts were public at the
  time and a wrong count carries no information about the day's return.
  - **A first count must be confirmed.** A span that starts without a count, with none carried
    into it, has its first quarter searched by bisection for its first record with one. That
    count stands only if the span's next sample confirms it within 10%; otherwise the search
    goes on to the first record the next sample confirms, and the days before it have no count.
    GE Vernova's first records carry GE's own count, four times its own, and Amentum's 0.63 of
    its own: 78 lookups over 12 spans.
  - **An isolated error is not carried.** A sample whose two samples before and two after agree
    with each other within 10%, and which differs from each beyond it with no split between, is
    an error, and the count before it carries through: six, among them Quest Diagnostics' at
    0.104 of its neighbours, Starbucks' at 0.123 and Block's at 0.110. A sample its own record's
    weighted count confirms within 10% is no error (midoffice09's ruling 1): PCG's of
    2026-01-09, at 1.000 of it, is spared, and its series stays flagged as alternating. Samples
    alternating between two counts, as Constellation's do, are left: the rule cannot tell which
    is true.
  - **Rulings for what no rule can tell,** in `config/sp500_share_count_overrides.json`, each
    with its evidence: Honeywell's records carrying Honeywell Aerospace's count from
    2026-10-12, Skydance's carrying none after it took Warner Bros. Discovery. The build lists
    every entry it uses.

- **A span ends where its security stops trading on an exchange** (midoffice09's ruling 2). A
  span's records say where its ticker trades: from its first sample whose market is not
  "stocks", OTC Link's after a failure, the span goes, but for any day the index still held it.
  Signature Bank's span from 2023-03-28 goes, 864 priced days, and First Republic's is kept to
  2023-05-03, its one member-day on OTC Link, 138 priced days going after it. No member-day
  changes: the index holds only listed shares. A leaver that stays listed keeps forecasting
  from its own history.

  Member-days without a count: 234 -- FedEx Freight's 109 and Vylor's 24, whose records carry
  none in the window; the spin-offs' first days; Skydance's 21, by its ruling; and 37 days the
  index carried a member past its last trade.

Two standing checks report on the counts; neither is a source. Polygon's weighted share count
is an earnings-per-share denominator, an average over a period, not the shares in existence; so
wherever a record carries one, it is set against the count the panel carries that day and
flagged beyond 10%. It confirms the 1:1 carries of OKE's and XOM's reorganisations (1.0001 and
0.9920), and the repairs of Quest Diagnostics', Everest's and Starbucks' samples, whose own
records' weighted counts sit within 0.6% of the count carried through them; it flags Carnival's
unification (1.118) and Honeywell's contaminated records (0.50), and PG&E's, where both of
Polygon's counts agree with the sample the error rule removed. Where a record's own share-class
count is a single class of a company with several -- GOOGL, BRK.B, FOXA -- the weighted count
spans them all, so those flag on every record. The second check reports every move of a carried
count beyond 10% that no split explains: 86 after the rulings, mostly mergers' issuance, and
the counts that alternate.

### What it changes

At today's defaults, the 938 periods from 2023-02-08, in three steps measured each on its own:
the old panel against the new, both on price returns and the adjusted bars' volumes; then the
new on traded volumes; then on total returns.

| | old panel | on securities | traded volumes | total returns |
|---|---|---|---|---|
| members in the regression a period, median (range) | 451 (422-476) | 502 (499-503) | 502 | 502 |
| mean R-squared | 0.298076 | 0.290903 | 0.292774 | 0.293189 |
| residual spread, 1.4826 x MAD (sd) | 0.1333 (0.361) | 0.1362 (0.342) | 0.1313 (0.340) | 0.1330 (0.340) |
| specific risk, p10 / median / p90 | 15.40 / 23.77 / 36.50% | 15.60 / 24.83 / 40.75% | 15.68 / 24.79 / 40.72% | 15.59 / 24.80 / 40.78% |
| size-pair risk, smallest 85 / largest 50 | 11.94 / 0.38% | 11.33 / 0.61% | 11.77 / 0.62% | 11.82 / 0.62% |

On securities the regression holds 26 to 78 more members each period, 470,548 of the 471,826
member-periods; of the rest, 1,058 lack Momentum, new listings short of RSTR's half-year, 197 a
weight, 23 a return. The 87 securities it adds include enough brokers and asset managers for
SIC62 to be its own industry, leaving Finance & Real Estate with 12 members and a noise share
of 31%. The 467 securities in both runs keep their specific-risk percentiles within about 0.1
of a point; BNY's falls by 14 points and Marsh McLennan's by 11, their histories now their own.
Traded volumes take Trading Activity's mean from +4.11% to +2.26% a year and its volatility
from 5.80% to 6.22%. Total returns raise every industry's mean, from 0.65% (Services) to 2.95%
(SIC67, the REITs), in line with what their members paid (correlation 0.945 across the 20), and
lower Volatility's by 0.58%; the high yielders' specific risk falls most, Altria's by 0.58 of a
point.

Since midoffice10 the default panel is `_listed`: PCG's sample spared, which changes its count
over 63 member-days, and the two spans ended off an exchange, which change 1,002 priced cells
and no member-day. Mean R-squared moves from 0.293204 to 0.293203; Signature Bank's 395.72%
forecast goes, and 966 specific returns measured outside the regressions with it.

Since midoffice09 the run's defaults are the panel on securities with its counts repaired, total
returns, and each security's history built from every specific return it has. The three count
rulings move the model little -- 533 member-days change their count, and mean R-squared moves
from 0.293189 to 0.293204 -- and covered histories move no factor return at all: specific risk
is 15.58 / 24.79 / 40.56% at p10 / median / p90.

## Status

Built, 29 tests in `tests/midoffice/test_model.py` and 28 in `test_covariance.py`:
`Frequency`, `ReturnPanel`,
`FactorReturns`, date alignment with loss reporting, and the time-series estimator with
annualisation, the intercept, both exclusion rules and conditioning reporting; the sample,
exponentially weighted and shrunk covariance estimators with Ledoit-Wolf intensity.
Both estimation directions are built: time-series (Fama-French) and cross-sectional
(Barra), sharing one `FactorModel` output and one covariance-estimator interface.

Not built:

* **Factor construction from characteristics.** Factor returns are taken as given, so a
  published series can be used directly. Building SMB and HML by sorting on market cap and
  book-to-price is a separate set of choices -- breakpoints, rebalancing frequency,
  value-weighting -- and deserves its own step.
* **A factor tree.** Three flat factors need no hierarchy; an industry-plus-style model does,
  and `compute_risk` already takes one.
