"""Estimating a factor model by time-series regression: the Fama-French construction.

Each security's returns are regressed on the factor returns over a common window.  The slopes
are its exposures, the residual standard deviation is its specific risk, and the sample
covariance of the factor returns is the factor covariance.  That is the published
Fama-French approach, and it is the simpler of the two ways to build a factor model.

The other way is Barra's, and the difference is worth stating because the engine this feeds
was ported from a Barra consumer.  Barra's exposures are *observable characteristics* --
log market cap, book-to-price -- standardised, and the factor returns are then estimated
period by period by regressing security returns cross-sectionally on those exposures.  So the
two approaches run in opposite directions: here the factor returns are given and the exposures
are estimated; there the exposures are given and the factor returns are estimated.

Both produce the pair `compute_risk` needs, so both plug in. The practical difference is that
a time-series beta is backward-looking over its estimation window and measured with error,
while a characteristic is observable today -- which is why Barra chose characteristics for
risk forecasting.  A model built here inherits that limitation and the notes say so.

**This module is where rank-deficient covariances come from.**  A sample covariance over fewer
periods than factors is singular, and a sample covariance over barely more is nearly so. That
is precisely the construction `RiskConfig.psd_relative_floor` was calibrated against -- see
docs/CONDITIONING.md, where the test fixtures estimate a covariance from 30 observations of 40
factors on purpose. The estimator reports the ratio so the conditioning is visible at the point
it is created rather than only where it bites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..risk.basis import FactorCovariance
from ..risk.exposures import ExposureVector, SpecificRisk
from .covariance import CovarianceEstimate, CovarianceEstimator, SampleCovariance
from .panel import Alignment, FactorReturns, Frequency, ReturnPanel, align

MIN_OBSERVATIONS = 24
"""Two years of monthly returns. A floor on how little history a beta may be estimated from.

Not a statistical threshold so much as a statement that a three-factor beta from six months is
not a number anyone should act on. Barra requires a minimum and flags short-history names; the
estimator excludes them and says which.
"""


@dataclass(frozen=True)
class EstimationNotes:
    """What the estimation could not do properly, and under what conventions it ran."""

    n_periods: int = 0
    n_factors: int = 0
    n_securities: int = 0
    n_estimated: int = 0
    frequency: Frequency = Frequency.MONTHLY
    annualised: bool = True
    min_observations: int = MIN_OBSERVATIONS
    short_history: tuple[str, ...] = ()
    degenerate: tuple[str, ...] = ()
    zero_specific_risk: tuple[str, ...] = ()
    alignment: Alignment | None = None
    covariance: CovarianceEstimate | None = None

    @property
    def effective_observations(self) -> float:
        """How many periods the covariance estimator actually used, by Kish's measure.

        Equal to `n_periods` for an unweighted estimate and less for an exponentially
        weighted one. This rather than the row count is what the conditioning judgements
        below use: a long history with a short half-life has plenty of rows and very little
        information, and the row count would call that matrix healthy.
        """
        if self.covariance is None:
            return float(self.n_periods)
        return self.covariance.effective_observations

    @property
    def periods_per_factor(self) -> float:
        """Effective observations per factor. Below 1 the covariance is singular."""
        return (
            self.effective_observations / self.n_factors
            if self.n_factors
            else float("inf")
        )

    @property
    def covariance_is_rank_deficient(self) -> bool:
        return self.effective_observations <= self.n_factors

    @property
    def is_clean(self) -> bool:
        return not self.qualifications()

    def qualifications(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.alignment is not None:
            notes.extend(self.alignment.qualifications())
        if self.covariance is not None:
            notes.extend(self.covariance.notes)
        shrunk = (
            self.covariance is not None
            and (self.covariance.shrinkage_intensity or 0.0) > 0.0
        )
        if self.covariance_is_rank_deficient and not shrunk:
            missing = max(int(self.n_factors - self.effective_observations) + 1, 1)
            notes.append(
                f"the factor covariance has {self.effective_observations:.1f} effective "
                f"observation(s) of {self.n_factors} factor(s), so it is singular by "
                f"construction and has at least {missing} zero eigenvalue(s). A book lying "
                f"in one of those directions gets a risk number with no correct digits "
                f"unless psd_relative_floor is positive -- see docs/CONDITIONING.md. "
                f"Shrinking the estimate repairs this at source rather than patching it "
                f"downstream"
            )
        elif self.periods_per_factor < 3.0 and not shrunk:
            notes.append(
                f"the factor covariance has only {self.periods_per_factor:.1f} effective "
                f"observation(s) per factor, so it is poorly conditioned even though it is "
                f"not singular; its smallest eigenvalues are mostly estimation noise. "
                f"Shrinkage buys about {min(21.0, 25.0 / self.periods_per_factor):.0f}% in "
                f"accuracy at this sample size"
            )
        if self.short_history:
            shown = ", ".join(self.short_history[:5])
            more = (
                f" and {len(self.short_history) - 5} more"
                if len(self.short_history) > 5
                else ""
            )
            notes.append(
                f"{len(self.short_history)} security(ies) had fewer than "
                f"{self.min_observations} observation(s) and were left out of the model "
                f"entirely: {shown}{more}. They will be uncovered to the engine"
            )
        if self.degenerate:
            notes.append(
                f"{len(self.degenerate)} security(ies) had too few observations for the "
                f"degrees of freedom the regression needs and were left out: "
                f"{', '.join(self.degenerate[:5])}"
            )
        if self.zero_specific_risk:
            notes.append(
                f"{len(self.zero_specific_risk)} security(ies) came out with zero specific "
                f"risk, meaning the factors explain their returns exactly: "
                f"{', '.join(self.zero_specific_risk[:5])}. That is usually a security that "
                f"is itself one of the factor portfolios, or a duplicated return series"
            )
        if not self.annualised:
            notes.append(
                f"this model is on a {self.frequency.label} basis and has NOT been "
                f"annualised; the engine's tail measures assume annual inputs, so risk "
                f"numbers built on it will be understated by a factor of "
                f"{self.frequency.volatility_scale:.2f}"
            )
        return tuple(notes)


@dataclass(frozen=True)
class FactorModel:
    """A built model: exactly the pair `compute_risk` consumes, plus how it was made."""

    factor_ids: tuple[int, ...]
    factor_names: Mapping[int, str]
    covariance: FactorCovariance
    exposures: Mapping[str, ExposureVector]
    alphas: Mapping[str, float] = field(default_factory=dict)
    r_squared: Mapping[str, float] = field(default_factory=dict)
    notes: EstimationNotes = field(default_factory=EstimationNotes)

    @property
    def securities(self) -> tuple[str, ...]:
        return tuple(self.exposures)

    def qualifications(self) -> tuple[str, ...]:
        return self.notes.qualifications()

    def describe(self) -> str:
        basis = "annualised" if self.notes.annualised else self.notes.frequency.label
        head = (
            f"{len(self.factor_ids)}-factor model, {basis}, estimated on "
            f"{self.notes.n_periods} {self.notes.frequency.label} period(s) for "
            f"{self.notes.n_estimated} of {self.notes.n_securities} security(ies)"
        )
        if self.notes.covariance is None:
            return head
        return f"{head}; covariance {self.notes.covariance.describe()}"


def estimate_timeseries_model(
    panel: ReturnPanel,
    factors: FactorReturns,
    *,
    min_observations: int = MIN_OBSERVATIONS,
    annualise: bool = True,
    first_factor_id: int = 1,
    fit_intercept: bool = True,
    covariance_estimator: CovarianceEstimator | None = None,
) -> FactorModel:
    """Regress each security on the factor returns; return a model the engine can consume.

    `fit_intercept` defaults to true and the intercept is **not** an exposure. Omitting it
    forces every regression through the origin, which biases the slopes toward whatever drift
    the window happened to contain. The estimated alpha is returned separately, for inspection
    rather than for risk: a factor risk model has nothing to say about it.
    """
    alignment = align(panel, factors)
    n_periods = len(alignment.dates)
    n_factors = factors.n_factors
    if n_periods == 0:
        raise ValueError(
            "the panel and the factor returns share no dates, so there is nothing to "
            "estimate; check that both are on the same frequency and period convention"
        )

    factor_block = factors.returns[list(alignment.factor_rows), :]
    panel_block = panel.returns[list(alignment.panel_rows), :]

    # The design matrix: an intercept column, then the factors.
    design = (
        np.column_stack([np.ones(n_periods), factor_block])
        if fit_intercept
        else factor_block
    )
    n_parameters = design.shape[1]

    factor_ids = tuple(first_factor_id + i for i in range(n_factors))
    names = {fid: factors.names[i] for i, fid in enumerate(factor_ids)}

    exposures: dict[str, ExposureVector] = {}
    alphas: dict[str, float] = {}
    r_squared: dict[str, float] = {}
    short_history: list[str] = []
    degenerate: list[str] = []
    zero_specific: list[str] = []

    for j, code in enumerate(panel.securities):
        column = panel_block[:, j]
        usable = ~np.isnan(column)
        n_usable = int(np.count_nonzero(usable))

        if n_usable < min_observations:
            short_history.append(code)
            continue
        # One more observation than parameters, or the residual has no degrees of freedom and
        # the specific risk comes out as exactly zero for arithmetic reasons.
        if n_usable <= n_parameters:
            degenerate.append(code)
            continue

        x = design[usable, :]
        y = column[usable]
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ coefficients
        dof = n_usable - n_parameters
        specific_variance = float(residuals @ residuals) / dof
        specific_volatility = float(np.sqrt(max(specific_variance, 0.0)))
        if annualise:
            specific_volatility *= panel.frequency.volatility_scale

        betas = coefficients[1:] if fit_intercept else coefficients
        if fit_intercept:
            alphas[code] = float(coefficients[0])

        total_variance = float(np.var(y, ddof=0))
        r_squared[code] = (
            1.0 - float(residuals @ residuals) / (total_variance * n_usable)
            if total_variance > 0.0
            else 0.0
        )

        if specific_volatility == 0.0:
            zero_specific.append(code)

        exposures[code] = ExposureVector(
            {fid: float(b) for fid, b in zip(factor_ids, betas)},
            SpecificRisk.for_security(code, specific_volatility),
        )

    # The covariance of the factor returns over the same window. The estimator defaults to
    # the plain sample covariance, which keeps this function's behaviour unchanged for any
    # caller that does not ask for something better.
    estimator = covariance_estimator or SampleCovariance()
    if n_periods > 1:
        estimate = estimator.estimate(factor_block)
    else:
        estimate = CovarianceEstimate(
            matrix=np.zeros((n_factors, n_factors)),
            effective_observations=float(n_periods),
            method="degenerate",
            n_periods=n_periods,
        )
    matrix = estimate.matrix
    if annualise:
        matrix = matrix * panel.frequency.variance_scale
    matrix = (matrix + matrix.T) / 2.0  # kill float asymmetry before FactorCovariance checks

    return FactorModel(
        factor_ids=factor_ids,
        factor_names=names,
        covariance=FactorCovariance(factor_ids, matrix),
        exposures=exposures,
        alphas=alphas,
        r_squared=r_squared,
        notes=EstimationNotes(
            n_periods=n_periods,
            n_factors=n_factors,
            n_securities=panel.n_securities,
            n_estimated=len(exposures),
            frequency=panel.frequency,
            annualised=annualise,
            min_observations=min_observations,
            short_history=tuple(short_history),
            degenerate=tuple(degenerate),
            zero_specific_risk=tuple(zero_specific),
            alignment=alignment,
            covariance=estimate,
        ),
    )
