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

**A factor nobody carries in a period is left out of that period.**  An industry with every
member missing on one day is an all-zero column that day, and it makes the design rank
deficient for a reason that has nothing to do with the other columns.  It is dropped from that
period's fit, its factor return recorded as zero -- what the minimum-norm solution gives it
anyway -- and counted per factor in the notes, next to the periods in which exactly one
security carried it.  That second count sees what `noise_share` cannot: a factor with one
member fits that member exactly, its residual is zero, and its sampling variance looks small.

**Specific risk is estimated over the regression and forecast beyond it.**  The specific risk
model learns only from the securities in at least one estimated period's regression -- the
rest have no residual to learn from -- but any security with a characteristic row at the
exposure date gets a forecast, because a holding outside the estimation universe still needs
a number.  Outside it the number is a fallback, the characteristic forecast extrapolated to a
security nothing was estimated on, and the notes name it as one.  It never appears on the
floor list: hitting the floor is a statement about an estimate.

**Each period's regression is fixed the day before its return.**  A characteristic dated t is
computed from day t's close -- LNCAP from that day's price -- and day t's return ends at the
same close, so regressing one on the other puts the return inside its own exposure: a stock
that rose today is bigger today, and Size looks as though it paid, every day.  With `lag=1`,
the default and Barra's convention, period t's regression takes its exposures, their
standardisation, its weights and its membership from the shared date before t, and only its
returns from t.  The model's exposures are still the last date's: they are as of today and
forecast forward.  `lag=0` regresses same-day, which is right only for characteristics known
before the return they are paired with -- a synthetic world built that way, say.

**Index membership is one more condition of entry, and only that.**  Given a membership, a
security enters period t's regression only if it was a member on the date its exposures are
read -- the day before t, under the lag -- as well as having a return on t, a complete row and
a weight.  A non-member cell is then exactly a missing return: no part in that period's factor
returns, no residual, nothing for the specific risk model to learn from.  Where this module
standardises, the same array decides whose values set the scale -- see
`CharacteristicPanel.standardise`.  It is not a filter on the output -- a security outside the
index on the exposure date still gets exposures and a forecast, as a holding outside it would.

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
from .characteristics import CharacteristicPanel, _checked_membership
from .covariance import CovarianceEstimate, CovarianceEstimator, SampleCovariance
from .panel import FactorReturns, Frequency, ReturnPanel
from .specificrisk import RESIDUAL_SPREADS, SpecificRiskModel, estimate_specific_risk_model
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
    # Periods that held enough securities and were not estimated, the caller's `periods`
    # ruling them out.
    excluded_periods: tuple[date, ...] = ()
    # Per estimated period, its date and how many securities its regression held. A
    # restriction to index members only removes, so these are where it shows.
    estimated_dates: tuple[date, ...] = ()
    cross_section_sizes: tuple[int, ...] = ()
    rank_deficient_periods: tuple[date, ...] = ()
    short_history: tuple[str, ...] = ()
    no_exposures: tuple[str, ...] = ()
    zero_specific_risk: tuple[str, ...] = ()
    specific_risk_method: str = "historical"
    specific_risk_model: SpecificRiskModel | None = None
    # In the regression universe, with a characteristic forecast below the floor: their
    # linear V ran past the end of its validity.
    floored_specific_risk: tuple[str, ...] = ()
    # In the regression universe, with the floor itself as the forecast delivered. Under the
    # model alone these are the floored ones; under a blend a security's own history can carry
    # a floored characteristic forecast clear of it, or bring an unfloored one down onto it.
    delivered_floor: tuple[str, ...] = ()
    # In no estimated period's regression, so nothing was estimated on them: their specific
    # risk is a fallback, the characteristic forecast extrapolated to them.
    fallback_specific_risk: tuple[str, ...] = ()
    mean_r_squared: float = float("nan")
    observed_variance: tuple[float, ...] = ()
    sampling_variance: tuple[float, ...] = ()
    factor_names: tuple[str, ...] = ()
    covariance: CovarianceEstimate | None = None
    # Per factor, over the estimated periods: how many had at least one security in the
    # regression with a nonzero exposure, how many exactly one, and how many none. A period
    # with none left the factor out of that period's fit.
    present_periods: tuple[int, ...] = ()
    single_member_periods: tuple[int, ...] = ()
    empty_periods: tuple[int, ...] = ()
    # Per factor, the collinear periods in which it was constant across the cross-section and
    # in the span of the other columns -- a cause that was checked, not presumed.
    collinear_constant: tuple[int, ...] = ()
    # The securities in at least one estimated period's regression.
    regression_universe: tuple[str, ...] = ()
    # How many shared dates before each return its exposures, weights and membership were
    # read; see `regression_design`.
    lag: int = 1
    # Whose residuals each security's own specific-risk history is built from: "regression",
    # its residuals in the regressions that held it; or "covered", its specific return in every
    # estimated period in which it had a return and a complete characteristic row, member or
    # not -- see `estimate_cross_sectional_model`.
    specific_history: str = "regression"
    # Under "covered": the specific returns measured outside the regressions, against the
    # members' factor returns; and the securities outside every regression whose forecast
    # blends a history of their own, so are no fallback.
    outside_residuals: int = 0
    blended_outside: tuple[str, ...] = ()
    # For each security with a forecast, how many specific returns its own history holds.
    history_lengths: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    # The estimated factor returns, period by period: what the covariance was estimated from.
    factor_returns: FactorReturns | None = field(default=None, repr=False, compare=False)

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
        if self.excluded_periods:
            shown = ", ".join(str(d) for d in self.excluded_periods[:3])
            more = (
                f" and {len(self.excluded_periods) - 3} more"
                if len(self.excluded_periods) > 3
                else ""
            )
            out.append(
                f"{len(self.excluded_periods)} period(s) held enough securities and were not "
                f"estimated, the caller's periods ruling them out: {shown}{more}"
            )
        empty = sum(self.empty_periods)
        if empty:
            named = [
                f"{name} ({k})" for name, k in zip(self.factor_names, self.empty_periods) if k
            ]
            out.append(
                f"{empty} factor-period(s) had no security in the regression exposed to the "
                f"factor and were left out of that period's fit, the factor return recorded "
                f"as zero: {', '.join(named[:5])}"
            )
        if self.rank_deficient_periods:
            shown = ", ".join(str(d) for d in self.rank_deficient_periods[:3])
            more = (
                f" and {len(self.rank_deficient_periods) - 3} more"
                if len(self.rank_deficient_periods) > 3
                else ""
            )
            # Name a cause only where one was checked. Empty columns never get here -- they
            # are left out of the period's fit -- so what can be identified is a constant one.
            constant = [
                f"{name} ({k})"
                for name, k in zip(self.factor_names, self.collinear_constant)
                if k
            ]
            if constant:
                cause = (
                    f" -- constant across the cross-section, and so in the span of the other "
                    f"columns: {', '.join(constant[:3])}"
                )
            elif self.market_factor:
                cause = " -- industry dummies alongside a market factor are the usual cause"
            else:
                cause = " -- no empty or constant column accounts for it"
            out.append(
                f"{len(self.rank_deficient_periods)} period(s) had a collinear design and "
                f"took a minimum-norm solution, which has no economic reading: "
                f"{shown}{more}{cause}"
            )
        singles = [
            f"{name} ({k} of {self.n_periods})"
            for name, k in zip(self.factor_names, self.single_member_periods)
            if k
        ]
        if singles:
            out.append(
                f"{len(singles)} factor(s) had exactly one security exposed in some estimated "
                f"period(s); there the factor return is that security's return, fitted "
                f"exactly, and noise_share reads it as well estimated: "
                f"{', '.join(singles[:5])}"
            )
        if self.short_history and self.specific_risk_method == "historical":
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
        blend = self.specific_risk_method == "blend"
        if self.floored_specific_risk and blend:
            # What `floored` tests is the characteristic forecast, and a blend does not
            # deliver that: JPM's was below the floor while the blend gave it 12.8 percent.
            carried = (
                "; blended with their own residual history, no forecast delivered is the "
                "floor"
                if not self.delivered_floor
                else ""
            )
            out.append(
                f"{len(self.floored_specific_risk)} security(ies) have a characteristic "
                f"forecast below the specific-risk floor, their linear V having run past the "
                f"end of its validity{carried}: {', '.join(self.floored_specific_risk[:5])}"
            )
        elif self.floored_specific_risk:
            out.append(
                f"{len(self.floored_specific_risk)} security(ies) hit the specific-risk "
                f"floor, so their forecast is the floor rather than the model: "
                f"{', '.join(self.floored_specific_risk[:5])}"
            )
        if self.delivered_floor and blend:
            out.append(
                f"{len(self.delivered_floor)} security(ies) are delivered the floor itself, "
                f"the blend of model and history having fallen below it: "
                f"{', '.join(self.delivered_floor[:5])}"
            )
        if self.fallback_specific_risk and self.specific_history == "covered":
            out.append(
                f"{len(self.fallback_specific_risk)} security(ies) have no specific return of "
                f"their own, in a regression or out of one, so their forecast is a fallback, "
                f"not a model one -- the characteristic forecast extrapolated to them: "
                f"{', '.join(self.fallback_specific_risk[:5])}"
            )
        elif self.fallback_specific_risk:
            out.append(
                f"{len(self.fallback_specific_risk)} security(ies) are in no estimated "
                f"period's regression, so the specific risk model was not estimated on them "
                f"and their forecast is a fallback, not a model one -- the characteristic "
                f"forecast extrapolated to them: {', '.join(self.fallback_specific_risk[:5])}"
            )
        if self.blended_outside:
            out.append(
                f"{len(self.blended_outside)} security(ies) are in no estimated period's "
                f"regression and have specific returns of their own, measured against the "
                f"members' factor returns: their forecast blends that history with the "
                f"characteristic forecast extrapolated to them: "
                f"{', '.join(self.blended_outside[:5])}"
            )
        if self.specific_risk_model is not None:
            out.extend(self.specific_risk_model.notes.qualifications())
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


def _align(
    returns: ReturnPanel, characteristics: CharacteristicPanel
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return rows, characteristic rows, return columns and characteristic columns for the
    dates and securities both panels have, in return-panel order."""
    if returns.frequency != characteristics.frequency:
        raise ValueError(
            f"returns are {returns.frequency.label} and characteristics are "
            f"{characteristics.frequency.label}; align them before estimating"
        )

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
    return (
        [i for i, _ in date_pairs],
        [k for _, k in date_pairs],
        [j for j, _ in sec_pairs],
        [k for _, k in sec_pairs],
    )


def _weight_block(
    regression_weights: np.ndarray | None,
    returns: ReturnPanel,
    return_rows: list[int],
    return_cols: list[int],
) -> np.ndarray | None:
    if regression_weights is None:
        return None
    weights = np.asarray(regression_weights, dtype=float)
    if weights.shape != (len(returns.dates), len(returns.securities)):
        raise ValueError(
            f"regression weights have shape {weights.shape}, expected "
            f"{(len(returns.dates), len(returns.securities))} to match the return panel"
        )
    return weights[np.ix_(return_rows, return_cols)]


def _membership_block(
    membership: np.ndarray | None,
    returns: ReturnPanel,
    return_rows: list[int],
    return_cols: list[int],
) -> np.ndarray | None:
    members = _checked_membership(
        membership,
        (len(returns.dates), len(returns.securities)),
        "to match the return panel",
    )
    if members is None:
        return None
    return members[np.ix_(return_rows, return_cols)]


def _periods_block(
    periods: np.ndarray | None, returns: ReturnPanel, return_rows: list[int]
) -> np.ndarray | None:
    """`periods`, refused unless it is one boolean per return-panel date, read on the shared
    dates."""
    if periods is None:
        return None
    allowed = np.asarray(periods)
    if allowed.dtype != bool or allowed.shape != (len(returns.dates),):
        raise ValueError(
            f"periods must be a boolean array with one entry per return-panel date, shape "
            f"({len(returns.dates)},); got {allowed.dtype} {allowed.shape}"
        )
    return allowed[return_rows]


def _in_regression(
    r_block: np.ndarray,
    x_block: np.ndarray,
    w_block: np.ndarray | None,
    m_block: np.ndarray | None = None,
) -> np.ndarray:
    """Dates by securities: who is in each period's regression.

    A return, a complete characteristic row, when the regression is weighted a positive
    weight, and when a membership is given, membership that day. One definition for the
    estimator and for `regression_universe`, because a count taken over a slightly different
    set is how a factor ends up with no members.
    """
    usable = ~np.isnan(r_block) & ~np.isnan(x_block).any(axis=2)
    if w_block is not None:
        usable &= ~np.isnan(w_block) & (w_block > 0.0)
    if m_block is not None:
        usable &= m_block
    return usable


def _lagged(block: np.ndarray | None, lag: int) -> np.ndarray | None:
    """`block` moved `lag` rows later: row t holds what row t - lag did, and the first `lag`
    rows hold nothing -- nan, or False for a membership."""
    if block is None or lag == 0:
        return block
    out = np.zeros_like(block) if block.dtype == bool else np.full_like(block, np.nan)
    out[lag:] = block[:-lag]
    return out


@dataclass(frozen=True)
class RegressionDesign:
    """What each period's regression is handed, before any market column is added.

    One row per date the return panel and the characteristic panel share, in return-panel
    order. Row t holds that date's returns and, read `lag` shared dates earlier, everything
    else that defines its regression: the exposures, the weights and the membership. The first
    `lag` rows have no earlier date to read, so they hold no exposures and are in no regression.
    `in_regression` is the estimator's own test of entry, applied to these.
    """

    dates: tuple[date, ...]
    exposure_dates: tuple[date | None, ...]
    securities: tuple[str, ...]
    names: tuple[str, ...]
    lag: int
    returns: np.ndarray
    exposures: np.ndarray
    weights: np.ndarray | None
    members: np.ndarray | None
    in_regression: np.ndarray


def regression_design(
    returns: ReturnPanel,
    characteristics: CharacteristicPanel,
    *,
    regression_weights: np.ndarray | None = None,
    membership: np.ndarray | None = None,
    lag: int = 1,
) -> RegressionDesign:
    """Each period's return, and the exposures, weights and membership its regression reads.

    One definition for the estimator and `regression_universe`, and the thing a test of the
    timing reads: under a lag of one, nothing in row t but its returns may depend on a price
    dated t. Pricing day t differently must move row t's returns and leave its exposures,
    weights and `in_regression` bit for bit where they were.

    The lag counts shared dates, so a date the characteristic panel lacks is stepped over
    rather than read: staler, never later.
    """
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError(f"lag must be a whole number of periods, zero or more; got {lag!r}")
    lag = int(lag)
    return_rows, char_rows, return_cols, char_cols = _align(returns, characteristics)
    if lag >= len(return_rows):
        raise ValueError(
            f"a lag of {lag} leaves none of the {len(return_rows)} shared date(s) with an "
            f"earlier one to read its exposures from"
        )
    dates = tuple(returns.dates[i] for i in return_rows)
    r_block = returns.returns[np.ix_(return_rows, return_cols)]
    x_block = _lagged(characteristics.values[np.ix_(char_rows, char_cols)], lag)
    w_block = _lagged(
        _weight_block(regression_weights, returns, return_rows, return_cols), lag
    )
    m_block = _lagged(
        _membership_block(membership, returns, return_rows, return_cols), lag
    )
    return RegressionDesign(
        dates=dates,
        exposure_dates=tuple(None if t < lag else dates[t - lag] for t in range(len(dates))),
        securities=tuple(returns.securities[j] for j in return_cols),
        names=characteristics.names,
        lag=lag,
        returns=r_block,
        exposures=x_block,
        weights=w_block,
        members=m_block,
        in_regression=_in_regression(r_block, x_block, w_block, m_block),
    )


def regression_universe(
    returns: ReturnPanel,
    characteristics: CharacteristicPanel,
    *,
    regression_weights: np.ndarray | None = None,
    membership: np.ndarray | None = None,
    min_securities: int = MIN_SECURITIES_PER_PERIOD,
    n_factors: int | None = None,
    lag: int = 1,
    periods: np.ndarray | None = None,
) -> tuple[str, ...]:
    """The securities in at least one estimated period's regression, known before estimating.

    The estimator's own test: a security is in a period's regression when it has a return, a
    complete characteristic row and a positive weight, and a period is estimated when at
    least `max(min_securities, n_factors + 1)` securities are. Anything that counts members
    to make a decision -- an industry scheme's floors -- should count over this, because a
    member that is never in a regression contributes nothing to its factor. The real S&P 500
    run counted DISCA and DISCK, whose last returns were ten months before the first estimated
    period, toward an industry's floor.

    `membership`, a dates-by-securities boolean array on the return panel's shape, adds one
    condition: membership on the date the exposures are read. The universe is then the
    securities that reach a regression on at least one estimated period in which they were
    members -- one that traded on after leaving the index does not count for having done so.
    Pass the estimator the same array and the same `lag`, or the two sets disagree -- and the
    same `periods`, the estimator's own: a period it rules out is estimated by neither.

    `n_factors` defaults to the panel's own columns; pass the final model's count when the
    panel is only part of it. Industry indicators need not be in `characteristics` at all:
    for a classified security they are never missing, so they cannot change who is in a
    regression. The estimator records the same set as
    `CrossSectionalNotes.regression_universe`, which is how a caller checks the two agreed.
    """
    design = regression_design(
        returns,
        characteristics,
        regression_weights=regression_weights,
        membership=membership,
        lag=lag,
    )
    usable = design.in_regression
    k = characteristics.values.shape[2] if n_factors is None else n_factors
    estimated = usable.sum(axis=1) >= max(min_securities, k + 1)
    at = {d: i for i, d in enumerate(returns.dates)}
    allowed = _periods_block(periods, returns, [at[d] for d in design.dates])
    if allowed is not None:
        estimated &= allowed
    ever = usable[estimated].any(axis=0)
    return tuple(code for code, inside in zip(design.securities, ever) if inside)


def estimate_cross_sectional_model(
    returns: ReturnPanel,
    characteristics: CharacteristicPanel,
    *,
    regression_weights: np.ndarray | None = None,
    membership: np.ndarray | None = None,
    lag: int = 1,
    periods: np.ndarray | None = None,
    standardise: bool = True,
    standardise_weights: np.ndarray | None = None,
    winsorise: float | None = 4.0,
    fit_market: bool = True,
    covariance_estimator: CovarianceEstimator | None = None,
    min_observations: int = MIN_OBSERVATIONS,
    min_securities: int = MIN_SECURITIES_PER_PERIOD,
    specific_risk: str = "historical",
    capitalisations: np.ndarray | None = None,
    specific_risk_floor: float = 0.1,
    exposure_index: int = -1,
    annualise: bool = True,
    first_factor_id: int = 1,
    residual_spread: str = "mad",
    specific_history: str = "regression",
) -> FactorModel:
    """Regress returns cross-sectionally on characteristics, period by period.

    `regression_weights` is a dates-by-securities array of regression weights. Barra uses the
    square root of market capitalisation, on the argument that residual variance scales
    roughly as the inverse of size, which makes square-root-cap weights the efficient choice.
    Equal weighting lets the smallest, noisiest names set the factor returns -- there are far
    more of them, and in a cap-weighted world almost none of the money.

    `membership` is a dates-by-securities boolean array on the same shape, True where the
    security was an index member that day. Given one, a period's regression holds only the
    members on the date its exposures are read; a non-member cell is treated exactly as a
    missing return. When this call standardises, the same array is the estimation universe
    the standardisation is taken over, so the two cannot disagree about who is a member. None,
    the default, regresses and standardises over every security with data.

    `lag` is how many shared dates before each return the rest of its regression is read:
    the exposures, the weights and the membership, all from the same earlier date -- see
    `regression_design`. The default, 1, is Barra's convention, and the one any
    characteristic computed from a day's prices needs: at 0 that day's return is regressed on
    exposures built from the price that produced it. The first `lag` dates are not periods at
    all. The exposures the model forecasts with are unaffected -- `exposure_index` reads them
    as of its date, as before.

    `periods`, one boolean per date of the return panel, says which dates' periods may be
    estimated. One marked False is not, however many securities it holds, and the notes name
    it among `excluded_periods` rather than the thin ones. A caller whose estimated periods
    must be one model -- each factor built from the same descriptors as the exposure it is
    forecast on, say -- marks the rest. None, the default, estimates every period with enough
    securities. Give `regression_universe` the same array.

    `standardise_weights` is passed to `CharacteristicPanel.standardise` for the *mean*, and
    should be market capitalisation itself rather than its square root: the point of a
    cap-weighted mean is to put the market portfolio at zero exposure, and the market
    portfolio is weighted by cap.  Using the same array for both is a common and quiet error.

    `exposure_index` picks which period's characteristics become the model's exposures, and
    defaults to the last. This is the forecast-relevant choice and is the thing a time-series
    model cannot offer.

    `specific_risk` chooses how idiosyncratic risk is forecast:

    * ``"historical"`` (the default, and what every existing caller gets) is the sample
      standard deviation of the security's own residual series. Securities with fewer than
      `min_observations` residuals are excluded, because there is nothing to compute.
    * ``"model"`` fits `specificrisk.estimate_specific_risk_model` to the residual panel and
      forecasts from characteristics instead. **This changes who is in the model**: a security
      needs only a characteristic row, not a return history, so short-history names are no
      longer dropped.
    * ``"blend"`` combines the two by inverse variance, which is what BIM301 describes its own
      specific risk models as doing. This is the better forecast in every regime measured --
      see docs/FACTOR_MODEL.md -- but it is not the default, because changing every existing
      caller's specific risk silently is not a thing to do by default.

    `capitalisations`, a dates-by-securities array on the return panel's shape, calibrates
    kappa against size the way USE3 does. Without it kappa falls back to the normal constant.

    `residual_spread` is how the specific risk model measures the spread its blend weights
    answer to -- see `specificrisk.estimate_specific_risk_model`: "mad", the default, or "sd".

    `specific_history` is whose residuals a security's own specific-risk history -- the
    standard deviation `historical` takes, and the history `blend` weighs -- is built from.
    "regression", the default, is its residuals in the regressions that held it. "covered"
    adds, in every estimated period, its specific return wherever it had a return and a
    complete characteristic row and the regression did not hold it -- not a member, or no
    weight: its return less its exposures times the period's factor returns, with no
    degrees-of-freedom inflation, since no fit absorbed it. A joiner then arrives with its
    history from before it joined, and a leaver keeps forecasting from its own history after.
    Factor returns still come from the regression alone, and the specific risk model's
    estimation -- the V fit, kappa, the blend's model error -- from its residuals alone; a
    security outside every regression with a history of its own is blended, not a fallback.
    """
    if specific_history not in ("regression", "covered"):
        raise ValueError(
            f"specific_history must be 'regression' or 'covered', got {specific_history!r}"
        )
    # Dates and securities common to both, in panel order.
    return_rows, char_rows, return_cols, char_cols = _align(returns, characteristics)
    shared_dates = tuple(returns.dates[i] for i in return_rows)
    shared_securities = tuple(returns.securities[j] for j in return_cols)

    m_block = _membership_block(membership, returns, return_rows, return_cols)
    allowed = _periods_block(periods, returns, return_rows)

    panel = characteristics
    if standardise:
        # Over the same members the regressions are held to. The characteristic panel's own
        # dates and securities outside the return panel's have no membership to read, so
        # they set no scale; nothing here uses them.
        standardise_members = None
        if m_block is not None:
            standardise_members = np.zeros(
                (len(characteristics.dates), len(characteristics.securities)), dtype=bool
            )
            standardise_members[np.ix_(char_rows, char_cols)] = m_block
        panel = characteristics.standardise(
            standardise_weights, winsorise=winsorise, membership=standardise_members
        )

    # Every period's inputs from one place: day t's returns, and everything else read `lag`
    # shared dates earlier. The standardisation above is per date, so the exposures read for
    # period t were standardised over the members of the date they were read from.
    design = regression_design(
        returns,
        panel,
        regression_weights=regression_weights,
        membership=membership,
        lag=lag,
    )
    lag = design.lag
    r_block = design.returns
    x_chars = design.exposures
    x_block = x_chars

    names = list(panel.names)
    if fit_market:
        names = [MARKET_FACTOR] + names
        ones = np.ones((len(shared_dates), len(shared_securities), 1))
        x_block = np.concatenate([ones, x_block], axis=2)

    n_periods_in, n_secs, n_factors = x_block.shape

    w_block = design.weights

    kept_dates: list[date] = []
    sizes: list[int] = []
    factor_rows: list[np.ndarray] = []
    residuals = np.full((n_periods_in, n_secs), np.nan)
    thin: list[date] = []
    excluded: list[date] = []
    rank_deficient: list[date] = []
    r_squareds: list[float] = []
    sampling = np.zeros(n_factors)
    sampling_periods = 0
    periods_present = np.zeros(n_factors, dtype=int)
    periods_single = np.zeros(n_factors, dtype=int)
    periods_empty = np.zeros(n_factors, dtype=int)
    periods_constant = np.zeros(n_factors, dtype=int)
    ever_in = np.zeros(n_secs, dtype=bool)

    in_regression = design.in_regression
    # Every security's specific return, member or not, wherever it has a return and a complete
    # characteristic row: the regression's own residuals, and the rest against its factors.
    covered = None
    if specific_history == "covered":
        covered = np.full((n_periods_in, n_secs), np.nan)
        reachable = np.isfinite(r_block) & np.isfinite(x_block).all(axis=2)
    outside_residuals = 0
    # The first `lag` dates have nothing earlier to read: they are not periods, thin or not.
    for t in range(lag, n_periods_in):
        usable = in_regression[t]
        n_t = int(np.count_nonzero(usable))
        if n_t < max(min_securities, n_factors + 1):
            thin.append(shared_dates[t])
            continue
        if allowed is not None and not allowed[t]:
            excluded.append(shared_dates[t])
            continue
        ever_in |= usable

        r_t = r_block[t, usable]
        x_t = x_block[t][usable, :]
        if w_block is None:
            w_t = np.ones(n_t)
        else:
            # Normalised to average one, so the residual variance below is comparable to an
            # unweighted one. The WLS solution itself is invariant to this scaling.
            w_t = w_block[t, usable]
            w_t = w_t * (n_t / w_t.sum())

        # Who carries each factor this period. A column nobody carries is all zero: the
        # period says nothing about that factor, and leaving it in would make the design rank
        # deficient for a reason unrelated to every other column. So it is left out, and its
        # return recorded as zero -- what the minimum-norm solution gives it anyway.
        exposed = np.count_nonzero(x_t, axis=0)
        periods_present += exposed > 0
        periods_single += exposed == 1
        periods_empty += exposed == 0
        active = exposed > 0
        n_active = int(np.count_nonzero(active))

        root_w = np.sqrt(w_t)
        design = x_t[:, active] * root_w[:, None]
        target = r_t * root_w
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        f_t = np.zeros(n_factors)
        f_t[active] = coefficients
        if rank < n_active:
            rank_deficient.append(shared_dates[t])
            for position, k in enumerate(np.flatnonzero(active)):
                if fit_market and k == 0:
                    continue    # constant by construction: it is what a constant duplicates
                column = x_t[:, k]
                if np.ptp(column) > 1e-12 * max(1.0, float(np.abs(column).max())):
                    continue
                # Constant -- but the cause only if the other columns span it.
                if np.linalg.matrix_rank(np.delete(design, position, axis=1)) == rank:
                    periods_constant[k] += 1

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
        if covered is not None:
            covered[t, usable] = residuals[t, usable]
            outside = reachable[t] & ~usable
            if outside.any():
                covered[t, outside] = r_block[t, outside] - x_block[t][outside, :] @ f_t
                outside_residuals += int(np.count_nonzero(outside))

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
        if dof > 0 and rank == n_active:
            sigma2 = ss_res / dof
            try:
                xtwx_inv = np.linalg.inv(design.T @ design)
            except np.linalg.LinAlgError:
                pass
            else:
                # A factor left out this period adds nothing: its zero was set, not estimated.
                sampling[active] += sigma2 * np.diag(xtwx_inv)
                sampling_periods += 1

        kept_dates.append(shared_dates[t])
        sizes.append(n_t)
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

    if specific_risk not in ("historical", "model", "blend"):
        raise ValueError(
            f"specific_risk must be 'historical', 'model' or 'blend', got "
            f"{specific_risk!r}"
        )
    if residual_spread not in RESIDUAL_SPREADS:
        raise ValueError(
            f"residual_spread must be one of {RESIDUAL_SPREADS}; got {residual_spread!r}"
        )

    exposure_slice = panel.values[char_rows[exposure_index], :, :][char_cols, :]
    short_history: list[str] = []
    no_exposures: list[str] = []
    zero_specific: list[str] = []
    exposures: dict[str, ExposureVector] = {}

    # The residuals carry the degrees-of-freedom inflation applied period by period above,
    # which is what the specific risk model wants: V is invariant to a per-period scale
    # factor, while S and the realised standard deviation both need it.
    risk_model: SpecificRiskModel | None = None
    modelled: dict[str, float] = {}
    floored: tuple[str, ...] = ()
    delivered_floor: tuple[str, ...] = ()
    fallback: tuple[str, ...] = ()
    blended_outside: tuple[str, ...] = ()
    # A security's own history: its residuals in the regressions, or every specific return.
    own = residuals if covered is None else covered
    if specific_risk != "historical":
        cap_block = None
        if capitalisations is not None:
            caps = np.asarray(capitalisations, dtype=float)
            if caps.shape != (len(returns.dates), len(returns.securities)):
                raise ValueError(
                    f"capitalisations have shape {caps.shape}, expected "
                    f"{(len(returns.dates), len(returns.securities))} to match the "
                    f"return panel"
                )
            # Read with the exposures, as the regression weights are: a security's size while
            # it was being measured is its size on the date its regression was fixed.
            cap_block = _lagged(caps[np.ix_(return_rows, return_cols)], lag)
        # Estimated over the regression universe and nothing else. A security in no estimated
        # period's regression has no residual; it must not shape S, V, gamma, kappa or the
        # blend weights, nor be counted in the notes as something the model was fitted on.
        # Each residual is paired with the exposures its own regression read.
        fitted_on = np.flatnonzero(ever_in)
        universe = {shared_securities[j] for j in fitted_on}
        risk_model = estimate_specific_risk_model(
            residuals[:, fitted_on],
            tuple(shared_securities[j] for j in fitted_on),
            x_chars[:, fitted_on, :],
            panel.names,
            capitalisations=None if cap_block is None else cap_block[:, fitted_on],
            annualisation=vol_scale,
            residual_spread=residual_spread,
        )
        exposure_rows = {
            code: {
                name: float(exposure_slice[j, k])
                for k, name in enumerate(panel.names)
            }
            for j, code in enumerate(shared_securities)
            if not np.isnan(exposure_slice[j, :]).any()
        }
        # Forecast beyond it, but say so. Outside the universe the characteristic forecast is
        # extrapolated to a security nothing was estimated on: a fallback, kept off the floor
        # list, which is about estimates.
        fallback_rows = {c: row for c, row in exposure_rows.items() if c not in universe}
        exposure_rows = {c: row for c, row in exposure_rows.items() if c in universe}
        if specific_risk == "blend":
            history: dict[str, tuple[float, int]] = {}
            for j, code in enumerate(shared_securities):
                u = own[:, j]
                present = ~np.isnan(u)
                if int(present.sum()) >= 2:
                    history[code] = (
                        float(np.std(u[present], ddof=1)) * vol_scale,
                        int(present.sum()),
                    )
            # Outside the universe with a history of its own: blended, its history measured,
            # only the characteristic forecast extrapolated to it.
            outside_rows = {c: row for c, row in fallback_rows.items()
                            if covered is not None and c in history}
            fallback_rows = {c: row for c, row in fallback_rows.items() if c not in outside_rows}
            blended_outside = tuple(outside_rows)
            modelled = risk_model.blend(
                {**exposure_rows, **outside_rows}, history, floor_fraction=specific_risk_floor
            )
        else:
            modelled = risk_model.forecast(
                exposure_rows, floor_fraction=specific_risk_floor
            )
        floored = risk_model.floored(
            exposure_rows, floor_fraction=specific_risk_floor
        )
        floor = risk_model.floor_level(specific_risk_floor)
        delivered_floor = tuple(c for c, value in modelled.items() if value <= floor)
        modelled.update(
            risk_model.forecast(fallback_rows, floor_fraction=specific_risk_floor)
        )
        fallback = tuple(fallback_rows)

    for j, code in enumerate(shared_securities):
        row = exposure_slice[j, :]
        if np.isnan(row).any():
            no_exposures.append(code)
            continue
        if specific_risk == "historical":
            u = own[:, j]
            present = ~np.isnan(u)
            if int(present.sum()) < min_observations:
                short_history.append(code)
                continue
            # Already degrees-of-freedom corrected period by period, above.
            specific = float(np.std(u[present], ddof=1)) * vol_scale
        else:
            # A security with no residual history at all still gets a forecast here, which
            # is the point of modelling specific risk rather than measuring it.
            specific = float(modelled[code])
            u = own[:, j]
            if int(np.count_nonzero(~np.isnan(u))) < min_observations:
                short_history.append(code)
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
        # What matters is whether the exposures ARE standardised, not whether this call did
        # it. `build_risk_indices` returns a standardised panel and callers then pass
        # standardise=False, which made the model warn that its own correctly-scaled
        # exposures were on incomparable scales. A warning that fires when nothing is wrong
        # costs more than it saves: it teaches the reader to skim the others.
        standardised=standardise or panel.standardised,
        winsorised=panel.winsorised,
        market_factor=fit_market,
        thin_periods=tuple(thin),
        excluded_periods=tuple(excluded),
        estimated_dates=tuple(kept_dates),
        cross_section_sizes=tuple(sizes),
        rank_deficient_periods=tuple(rank_deficient),
        short_history=tuple(short_history),
        no_exposures=tuple(no_exposures),
        zero_specific_risk=tuple(zero_specific),
        specific_risk_method=specific_risk,
        specific_risk_model=risk_model,
        floored_specific_risk=floored,
        delivered_floor=delivered_floor,
        fallback_specific_risk=fallback,
        mean_r_squared=float(np.mean(r_squareds)) if r_squareds else float("nan"),
        observed_variance=observed,
        sampling_variance=mean_sampling,
        factor_names=tuple(names),
        covariance=estimate,
        present_periods=tuple(int(v) for v in periods_present),
        single_member_periods=tuple(int(v) for v in periods_single),
        empty_periods=tuple(int(v) for v in periods_empty),
        collinear_constant=tuple(int(v) for v in periods_constant),
        regression_universe=tuple(
            code for code, inside in zip(shared_securities, ever_in) if inside
        ),
        lag=lag,
        specific_history=specific_history,
        outside_residuals=outside_residuals,
        blended_outside=blended_outside,
        history_lengths={
            code: int(np.count_nonzero(~np.isnan(own[:, shared_securities.index(code)])))
            for code in exposures
        },
        factor_returns=factors,
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
