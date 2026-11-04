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
after standardising is a guard that makes the parameter mean what it says; it is nearly always
a no-op.

`CharacteristicPanel.winsorised` counts how many raw values moved. A handful means the clip
did its job; a large fraction means the characteristic itself is wrong -- wrong units, a sign
error, a vendor placeholder like -999 -- and no clipping will fix it.

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

The bias **is not subtracted off**. The naive correction can drive a variance negative, and
flooring that at zero would turn a diagnostic into a fabrication; Barra's own adjustment is
more careful than a subtraction. It is reported, and reported loudly past 25 percent.

One structural point worth noting: because sampling variance is roughly constant across
standardised factors, it is a larger *fraction* of a small factor's variance. The noise
concentrates on the lowest-variance factor -- the same shape as the time-series intercept bias
landing hardest on HML rather than on the market.

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
  Appendix A says "normalized", so the standardisation is part of the definition.
* **BTSG is set to zero when beta is negative**, which the definition states explicitly.
* **SERDP comes out a few percent below its AR(1) theoretical value**, consistently across
  rho. That is the definition rather than the implementation: it is built on *fitted*
  market-model residuals, and OLS forces those orthogonal to the regressors.

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
