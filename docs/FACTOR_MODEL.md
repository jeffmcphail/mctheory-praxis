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

## Status

Built, 29 tests in `tests/midoffice/test_model.py` and 28 in `test_covariance.py`:
`Frequency`, `ReturnPanel`,
`FactorReturns`, date alignment with loss reporting, and the time-series estimator with
annualisation, the intercept, both exclusion rules and conditioning reporting; the sample,
exponentially weighted and shrunk covariance estimators with Ledoit-Wolf intensity.

Not built:

* **Factor construction from characteristics.** Factor returns are taken as given, so a
  published series can be used directly. Building SMB and HML by sorting on market cap and
  book-to-price is a separate set of choices -- breakpoints, rebalancing frequency,
  value-weighting -- and deserves its own step.
* **Newey-West.** The third covariance correction, for autocorrelation. It matters most at
  daily frequency, where returns autocorrelate and the other two do nothing about it. With
  Bartlett weights it is positive semi-definite by construction, so it composes with the
  others safely.
* **The cross-sectional estimator**, which is what Barra does.
* **A factor tree.** Three flat factors need no hierarchy; an industry-plus-style model does,
  and `compute_risk` already takes one.
