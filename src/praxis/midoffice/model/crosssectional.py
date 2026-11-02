"""Estimating a factor model by cross-sectional regression: the Barra construction.

Every period, security returns are regressed *across securities* on their observable
characteristics.  The slopes are that period's factor returns; collected over the window they
form a factor return series whose covariance is the factor covariance.  The residuals,
collected per security over time, give specific risk.  The exposures are not estimated at all
-- they are the characteristics, standardised, read off at a point in time.

This runs in the opposite direction from `timeseries.py`, and the inversion is the whole
point:

    time-series    factor returns given, exposures estimated  (one regression per security,
                                                                over time)
    cross-sectional  exposures given, factor returns estimated  (one regression per period,
                                                                over securities)

Three consequences follow, and they are why Barra chose this direction for risk forecasting.

**Exposures carry no estimation error.**  A time-series beta is a statistic with a standard
error, estimated over a window during which the company may have changed completely.  A log
market cap is a measurement.  A firm that has doubled in size has a new size exposure today
and a five-year beta that still half-remembers what it used to be.

**Exposures are current.**  The model's view of what a portfolio is exposed to comes from this
month's characteristics, not from an average over the estimation window.  For a forecast that
is the question actually being asked.

**But the factor returns now carry the estimation error instead.**  It does not vanish; it
moves.  And it moves somewhere more damaging, because the factor covariance is built from
estimated quantities, so its diagonal is inflated by the average sampling variance of the
estimates.  `CrossSectionalNotes.noise_share` measures that inflation per factor, which is the
diagnostic this construction needs and the time-series one does not.

One other thing inverts.  In a time-series regression the intercept is alpha, and alpha is
emphatically *not* an exposure -- a factor risk model has nothing to say about it.  Here the
unit column is the **market factor**, and it *is* an exposure, because every security really
does have unit exposure to the market.  Same column of ones, opposite meaning.

What this module does not do, and a real Barra model does:

- **Constrained industry factors.**  Industry dummies sum to the unit column, so a design with
  both is collinear and has no unique solution.  Barra imposes a cap-weighted zero-sum
  constraint on the industry returns.  Here the rank deficiency is detected, the period is
  named in the notes, and `numpy.linalg.lstsq` returns the minimum-norm solution -- which is
  *a* solution but not the one with an economic interpretation.  Supply industries without a
  market factor, or accept the flag.
- **Modelled specific risk.**  Specific volatility here is the sample standard deviation of a
  security's residual series, which is noisy for a short history and says nothing at all about
  a security with none.  Barra regresses specific risk on characteristics so that a new issue
  inherits a sensible forecast.  Deferred; the notes name the securities that got nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import numpy as np

from ..risk.basis import FactorCovariance
from ..risk.exposures import ExposureVector, SpecificRisk
from .characteristics import CharacteristicPanel
from .covariance import CovarianceEstimate, CovarianceEstimator, SampleCovariance
from .panel import FactorReturns, Frequency, ReturnPanel
from .timeseries import MIN_OBSERVATIONS, FactorModel

MIN_SECURITIES_PER_PERIOD = 30
"""A floor on the cross-section a single period's factor returns may be estimated from.

Unlike `MIN_OBSERVATIONS` this one bites hard, because the regression is *across* securities:
with ten factors and twelve securities a period's factor returns are fitted through almost
every point and the residuals are nearly zero, which understates specific risk for every
security at once rather than for one. Thirty is a judgement, not a theorem.
"""

MARKET_FACTOR = "Market"


@dataclass(frozen=True)
class CrossSectionalNotes:
    """What the estimation could not do properly, and under what conventions it ran."""

    n_periods: int = 0
    n_factors: int = 0
    n_securities: int = 0
    n_estimated: int = 0
    frequency: Frequency = Frequency.MONTHLY
    annualised: bool = True
    min_observations: int = MIN_OBSERVATIONS
    min_securities: int = MIN_SECURITIES_PER_PERIOD
    weighting: str = "equal"
    standardised: bool = True
    winsorised: int = 0
    market_factor: bool = True
    thin_periods: tuple[date, ...] = ()
    rank_deficient_periods: tuple[date, ...] = ()
    short_history: tuple[str, ...] = ()
    no_exposures: tuple[str, ...] = ()
    zero_specific_risk: tuple[str, ...] = ()
    mean_r_squared: float = float("nan")
    observed_variance: tuple[float, ...] = ()
    sampling_variance: tuple[float, ...] = ()
    factor_names: tuple[str, ...] = ()
    covariance: CovarianceEstimate | None = None

    @property
    def effective_observations(self) -> float:
        """How many periods the covariance estimator actually used, by Kish's measure."""
        if self.covariance is None:
            return float(self.n_periods)
        return self.covariance.effective_observations

    @property
    def covariance_is_rank_deficient(self) -> bool:
        if self.covariance is None:
            return self.n_periods <= self.n_factors
        return self.covariance.is_rank_deficient

    @property
    def noise_share(self) -> tuple[float, ...]:
        """Per factor, the fraction of its estimated variance that is estimation noise.

        The sample covariance of estimated factor returns estimates the true factor covariance
        *plus* the average sampling variance of the estimates, so its diagonal is biased
        upward.  This is that bias as a fraction: `mean sampling variance / observed
        variance`.

        A small number means the cross-section is wide enough and the fit tight enough that
        the factor return is well determined each period, and the estimated factor volatility
        is mostly signal.  A number approaching one means the factor volatility being fed to
        the risk engine is substantially the noise in its own estimation -- which inflates
        every forecast built on it, and inflates it most for the portfolios most exposed to
        that factor.

        Not subtracted off. The naive correction -- observed minus sampling -- can drive a
        variance negative, and silently flooring that at zero would turn a diagnostic into a
        fabrication. Barra's own adjustment is more careful than a subtraction.
        """
        if not self.observed_variance:
            return ()
        return tuple(
            float(s / o) if o > 0.0 else float("inf")
            for o, s in zip(self.observed_variance, self.sampling_variance)
        )

    def qualifications(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.n_periods and self.n_factors:
            if self.n_periods < self.min_observations:
                out.append(
                    f"factor returns estimated over {self.n_periods} "
                    f"{self.frequency.label} period(s), fewer than the "
                    f"{self.min_observations} this estimator asks for"
                )
        if self.thin_periods:
            shown = ", ".join(str(d) for d in self.thin_periods[:3])
            more = (
                f" and {len(self.thin_periods) - 3} more"
                if len(self.thin_periods) > 3
                else ""
            )
            out.append(
                f"{len(self.thin_periods)} period(s) dropped for having fewer than "
                f"{self.min_securities} securities: {shown}{more}"
            )
        if self.rank_deficient_periods:
            shown = ", ".join(str(d) for d in self.rank_deficient_periods[:3])
            more = (
                f" and {len(self.rank_deficient_periods) - 3} more"
                if len(self.rank_deficient_periods) > 3
                else ""
            )
            out.append(
                f"{len(self.rank_deficient_periods)} period(s) had a collinear design and "
                f"took a minimum-norm solution, which has no economic reading: "
                f"{shown}{more} -- industry dummies alongside a market factor are the "
                f"usual cause"
            )
        if self.short_history:
            out.append(
                f"{len(self.short_history)} security(ies) excluded for fewer than "
                f"{self.min_observations} residual observation(s): "
                f"{', '.join(self.short_history[:5])}"
            )
        if self.no_exposures:
            out.append(
                f"{len(self.no_exposures)} security(ies) excluded for an incomplete "
                f"characteristic row at the exposure date: "
                f"{', '.join(self.no_exposures[:5])}"
            )
        if self.zero_specific_risk:
            out.append(
                f"{len(self.zero_specific_risk)} security(ies) have zero specific risk, "
                f"which a cross-section this narrow can produce by fitting through every "
                f"point: {', '.join(self.zero_specific_risk[:5])}"
            )
        if self.winsorised:
            out.append(
                f"{self.winsorised} raw characteristic value(s) were winsorised before "
                f"standardising"
            )
        noise = self.noise_share
        if noise:
            worst = max(range(len(noise)), key=lambda i: noise[i])
            if noise[worst] > 0.25:
                name = (
                    self.factor_names[worst]
                    if worst < len(self.factor_names)
                    else f"factor {worst}"
                )
                out.append(
                    f"{noise[worst]:.0%} of the estimated variance of {name} is the "
                    f"sampling error of its own estimation; the factor covariance diagonal "
                    f"is biased upward and is not corrected here"
                )
        if not self.standardised:
            out.append(
                "characteristics were not standardised, so the factor returns are on "
                "incomparable scales and their covariance should not be read as one"
            )
        if self.covariance is not None:
            out.extend(self.covariance.notes)
        shrunk = (
            self.covariance is not None
            and (self.covariance.shrinkage_intensity or 0.0) > 0.0
        )
        if self.n_factors and self.covariance_is_rank_deficient and not shrunk:
            out.append(
                f"the factor covariance has {self.effective_observations:.1f} effective "
                f"observation(s) of {self.n_factors} factor(s), so it is singular by "
                f"construction -- see docs/CONDITIONING.md. A cross-sectional model reaches "
                f"this state easily, because adding a characteristic costs a factor but no "
                f"history at all, and the temptation is to add many"
            )
        return tuple(out)


def estimate_cross_sectional_model(
    returns: ReturnPanel,
    characteristics: CharacteristicPanel,
    *,
    regression_weights: np.ndarray | None = None,
    standardise: bool = True,
    standardise_weights: np.ndarray | None = None,
    winsorise: float | None = 4.0,
    fit_market: bool = True,
    covariance_estimator: CovarianceEstimator | None = None,
    min_observations: int = MIN_OBSERVATIONS,
    min_securities: int = MIN_SECURITIES_PER_PERIOD,
    exposure_index: int = -1,
    annualise: bool = True,
    first_factor_id: int = 1,
) -> FactorModel:
    """Regress returns cross-sectionally on characteristics, period by period.

    `regression_weights` is a dates-by-securities array of regression weights. Barra uses the
    square root of market capitalisation, on the argument that residual variance scales
    roughly as the inverse of size, which makes square-root-cap weights the efficient choice.
    Equal weighting lets the smallest, noisiest names set the factor returns -- there are far
    more of them, and in a cap-weighted world almost none of the money.

    `standardise_weights` is passed to `CharacteristicPanel.standardise` for the *mean*, and
    should be market capitalisation itself rather than its square root: the point of a
    cap-weighted mean is to put the market portfolio at zero exposure, and the market
    portfolio is weighted by cap.  Using the same array for both is a common and quiet error.

    `exposure_index` picks which period's characteristics become the model's exposures, and
    defaults to the last. This is the forecast-relevant choice and is the thing a time-series
    model cannot offer.
    """
    if returns.frequency != characteristics.frequency:
        raise ValueError(
            f"returns are {returns.frequency.label} and characteristics are "
            f"{characteristics.frequency.label}; align them before estimating"
        )

    # Dates and securities common to both, in panel order.
    char_date_at = {d: i for i, d in enumerate(characteristics.dates)}
    date_pairs = [
        (i, char_date_at[d]) for i, d in enumerate(returns.dates) if d in char_date_at
    ]
    char_sec_at = {s: j for j, s in enumerate(characteristics.securities)}
    sec_pairs = [
        (j, char_sec_at[s])
        for j, s in enumerate(returns.securities)
        if s in char_sec_at
    ]
    if not date_pairs:
        raise ValueError(
            "the return panel and the characteristic panel share no dates, so there is "
            "nothing to estimate"
        )
    if not sec_pairs:
        raise ValueError(
            "the return panel and the characteristic panel share no securities, so there "
            "is nothing to estimate"
        )

    shared_dates = tuple(returns.dates[i] for i, _ in date_pairs)
    shared_securities = tuple(returns.securities[j] for j, _ in sec_pairs)
    return_rows = [i for i, _ in date_pairs]
    char_rows = [k for _, k in date_pairs]
    return_cols = [j for j, _ in sec_pairs]
    char_cols = [k for _, k in sec_pairs]

    panel = characteristics
    if standardise:
        panel = characteristics.standardise(standardise_weights, winsorise=winsorise)

    r_block = returns.returns[np.ix_(return_rows, return_cols)]
    x_block = panel.values[np.ix_(char_rows, char_cols)]

    names = list(panel.names)
    if fit_market:
        names = [MARKET_FACTOR] + names
        ones = np.ones((len(shared_dates), len(shared_securities), 1))
        x_block = np.concatenate([ones, x_block], axis=2)

    n_periods_in, n_secs, n_factors = x_block.shape

    if regression_weights is not None:
        regression_weights = np.asarray(regression_weights, dtype=float)
        if regression_weights.shape != (len(returns.dates), len(returns.securities)):
            raise ValueError(
                f"regression weights have shape {regression_weights.shape}, expected "
                f"{(len(returns.dates), len(returns.securities))} to match the return panel"
            )
        w_block = regression_weights[np.ix_(return_rows, return_cols)]
    else:
        w_block = None

    kept_dates: list[date] = []
    factor_rows: list[np.ndarray] = []
    residuals = np.full((n_periods_in, n_secs), np.nan)
    thin: list[date] = []
    rank_deficient: list[date] = []
    r_squareds: list[float] = []
    sampling = np.zeros(n_factors)
    sampling_periods = 0

    for t in range(n_periods_in):
        r = r_block[t, :]
        x = x_block[t, :, :]
        usable = ~np.isnan(r) & ~np.isnan(x).any(axis=1)
        if w_block is not None:
            usable &= ~np.isnan(w_block[t, :]) & (w_block[t, :] > 0.0)
        n_t = int(np.count_nonzero(usable))
        if n_t < max(min_securities, n_factors + 1):
            thin.append(shared_dates[t])
            continue

        r_t = r[usable]
        x_t = x[usable, :]
        if w_block is None:
            w_t = np.ones(n_t)
        else:
            # Normalised to average one, so the residual variance below is comparable to an
            # unweighted one. The WLS solution itself is invariant to this scaling.
            w_t = w_block[t, usable]
            w_t = w_t * (n_t / w_t.sum())

        root_w = np.sqrt(w_t)
        design = x_t * root_w[:, None]
        target = r_t * root_w
        f_t, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < n_factors:
            rank_deficient.append(shared_dates[t])

        fitted = x_t @ f_t
        u_t = r_t - fitted
        # Inflate by the degrees of freedom this period's fit consumed. Residuals from a
        # fitted regression are systematically too small -- the fit absorbed `rank` of the
        # n_t observations -- so their variance understates specific variance by
        # (n_t - rank)/n_t. With 200 securities on 4 factors that is 1 percent of volatility
        # and ignorable; with 40 securities on 10 factors it is 13 percent, and specific risk
        # is the dominant term for a concentrated portfolio, so the forecast is understated
        # exactly where it matters most. n_t varies by period, so the correction has to be
        # applied here rather than once at the end.
        dof_inflation = np.sqrt(n_t / (n_t - rank)) if n_t > rank else 1.0
        residuals[t, usable] = u_t * dof_inflation

        # Weighted R-squared about the weighted mean, which is the right benchmark when
        # the regression itself is weighted. On the raw residuals: the degrees-of-freedom
        # inflation above belongs to the specific-risk estimate, not to a goodness-of-fit
        # measure nor to the sampling variance below.
        ss_res = float(np.sum(w_t * u_t**2))
        mean_r = float(np.average(r_t, weights=w_t))
        ss_tot = float(np.sum(w_t * (r_t - mean_r) ** 2))
        if ss_tot > 0.0:
            r_squareds.append(1.0 - ss_res / ss_tot)

        # Sampling variance of this period's factor returns: sigma^2 (X'WX)^-1.
        dof = n_t - rank
        if dof > 0 and rank == n_factors:
            sigma2 = ss_res / dof
            try:
                xtwx_inv = np.linalg.inv(design.T @ design)
            except np.linalg.LinAlgError:
                pass
            else:
                sampling += sigma2 * np.diag(xtwx_inv)
                sampling_periods += 1

        kept_dates.append(shared_dates[t])
        factor_rows.append(f_t)

    if not factor_rows:
        raise ValueError(
            f"every period had fewer than {max(min_securities, n_factors + 1)} usable "
            f"securities, so no factor return could be estimated; lower min_securities or "
            f"widen the cross-section"
        )

    scale = returns.frequency.variance_scale if annualise else 1.0
    vol_scale = returns.frequency.volatility_scale if annualise else 1.0

    factors = FactorReturns(
        dates=tuple(kept_dates),
        names=tuple(names),
        returns=np.vstack(factor_rows),
        frequency=returns.frequency,
    )

    estimator = covariance_estimator or SampleCovariance()
    estimate = estimator.estimate(factors.returns)
    covariance_matrix = estimate.matrix * scale
    factor_ids = tuple(range(first_factor_id, first_factor_id + len(names)))

    observed = tuple(float(v) for v in np.diag(covariance_matrix))
    mean_sampling = (
        tuple(float(v) for v in (sampling / sampling_periods) * scale)
        if sampling_periods
        else ()
    )

    # Specific risk: the standard deviation of each security's residual series.
    exposure_slice = panel.values[char_rows[exposure_index], :, :][char_cols, :]
    short_history: list[str] = []
    no_exposures: list[str] = []
    zero_specific: list[str] = []
    exposures: dict[str, ExposureVector] = {}

    for j, code in enumerate(shared_securities):
        row = exposure_slice[j, :]
        if np.isnan(row).any():
            no_exposures.append(code)
            continue
        u = residuals[:, j]
        present = ~np.isnan(u)
        if int(present.sum()) < min_observations:
            short_history.append(code)
            continue
        # Already degrees-of-freedom corrected period by period, above.
        specific = float(np.std(u[present], ddof=1)) * vol_scale
        if specific <= 0.0:
            zero_specific.append(code)
        loadings = dict(zip(factor_ids[1:] if fit_market else factor_ids, row))
        if fit_market:
            loadings = {factor_ids[0]: 1.0, **loadings}
        exposures[code] = ExposureVector(
            exposures={fid: float(v) for fid, v in loadings.items() if v != 0.0},
            specific=SpecificRisk.for_security(code, specific),
        )

    notes = CrossSectionalNotes(
        n_periods=len(kept_dates),
        n_factors=len(names),
        n_securities=len(shared_securities),
        n_estimated=len(exposures),
        frequency=returns.frequency,
        annualised=annualise,
        min_observations=min_observations,
        min_securities=min_securities,
        weighting="square-root-cap or supplied" if w_block is not None else "equal",
        standardised=standardise,
        winsorised=panel.winsorised,
        market_factor=fit_market,
        thin_periods=tuple(thin),
        rank_deficient_periods=tuple(rank_deficient),
        short_history=tuple(short_history),
        no_exposures=tuple(no_exposures),
        zero_specific_risk=tuple(zero_specific),
        mean_r_squared=float(np.mean(r_squareds)) if r_squareds else float("nan"),
        observed_variance=observed,
        sampling_variance=mean_sampling,
        factor_names=tuple(names),
        covariance=estimate,
    )

    return FactorModel(
        factor_ids=factor_ids,
        factor_names={fid: name for fid, name in zip(factor_ids, names)},
        covariance=FactorCovariance(factor_ids=factor_ids, matrix=covariance_matrix),
        exposures=exposures,
        alphas={},
        r_squared={},
        notes=notes,
    )
