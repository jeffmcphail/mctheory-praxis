"""The country factor as a covariance basis (midoffice18; Barra USE4, section 3.1).

USE4 regresses each period's returns on a country factor -- every member exposed to it, one each --
the industries and the styles, the industries' factor returns constrained so that their
capitalisation-weighted sum is zero. Each member's industry exposures already sum to one, so the
country column lies in the industries' span: the fit is the industry model's own, the same fitted
values and the same specific returns, and its factor returns are a change of basis of the industry
model's. The country's return is the industries' returns averaged by capitalisation, each
industry's is its own less the country's, and the styles are as they were:

    f_c = sum_i c_i f_i,    f_i' = f_i - f_c,    so sum_i c_i f_i' = 0,

c_i industry i's share of the capitalisation of the period's regression. The industry model's
returns come back from the country basis by a fixed map whatever the weights, f_i = f_c + f_i'
(`country_map`).

A covariance estimator that is a quadratic form in the returns -- sample, exponential, Newey-West
-- is basis-invariant: estimated in the country basis and mapped back it is the industry model's
own. The two-half-life estimator is not. In the country basis it puts the market's variance on the
volatilities' window and the industries' correlations, net of the market, on the long one, where
the industry basis carries the market's variance on the industries' correlations, the long window
(`country_covariance`).
"""

from __future__ import annotations

from datetime import date
from typing import Mapping, Sequence

import numpy as np

from .crosssectional import regression_design
from .panel import ReturnPanel


def country_weights(
    returns: ReturnPanel,
    characteristics,
    capitalisations: np.ndarray,
    industry_columns: Sequence[int],
    *,
    regression_weights: np.ndarray | None = None,
    membership: np.ndarray | None = None,
    lag: int = 1,
) -> dict[date, np.ndarray]:
    """Each period's industry weights for the country's constraint, by date: industry i's share of
    the capitalisation of the period's regression -- its members as the estimator admits them
    (`regression_design`'s `in_regression`), at their capitalisations on the date the regression
    reads its exposures and weights from. `capitalisations` is dates by securities on the return
    panel's; `industry_columns` the industries' columns of `characteristics`, one-hot for every
    classified member. A member in no industry factor -- a group too small to be one, as the S&P
    500 scheme's "Other Industries" (CTVA, FRCB, SBNY) -- carries no industry weight: the country
    is the mean of the industries' returns, and the industries span it for the members they
    cover. A period with no member has no weights."""
    design = regression_design(returns, characteristics, regression_weights=regression_weights,
                               membership=membership, lag=lag)
    caps = np.asarray(capitalisations, dtype=float)
    at = {d: i for i, d in enumerate(returns.dates)}
    position = {code: j for j, code in enumerate(returns.securities)}
    columns = [position[code] for code in design.securities]
    industries = list(industry_columns)
    out: dict[date, np.ndarray] = {}
    for t, day in enumerate(design.dates):
        when = design.exposure_dates[t]
        usable = design.in_regression[t]
        if when is None or not usable.any():
            continue
        member_caps = caps[at[when]][columns][usable]
        if not (np.isfinite(member_caps).all() and (member_caps > 0.0).all()):
            raise ValueError(f"{day}: a member of the regression has no capitalisation to weight "
                             f"its industry by")
        indicators = design.exposures[t][usable][:, industries]
        rows = indicators.sum(axis=1)
        if not (np.isin(indicators, (0.0, 1.0)).all() and np.isin(rows, (0.0, 1.0)).all()):
            raise ValueError(f"{day}: the industry columns are not one-hot, or empty, for every "
                             f"member")
        mass = member_caps @ indicators
        out[day] = mass / mass.sum()
    return out


def to_country_basis(
    factor_returns: np.ndarray, industry_columns: Sequence[int], weights: np.ndarray
) -> np.ndarray:
    """The industry model's factor returns, periods by factors, in the country basis: the
    country's return first, f_c = sum_i c_i f_i, then the model's factors in their order, each
    industry less the country, the styles as they are. `weights` is periods by industries, each
    row summing to one."""
    f = np.atleast_2d(np.asarray(factor_returns, dtype=float))
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    industries = np.asarray(industry_columns, dtype=int)
    if w.shape != (f.shape[0], industries.size):
        raise ValueError(f"weights of shape {w.shape} for {f.shape[0]} periods and "
                         f"{industries.size} industries")
    country = np.sum(w * f[:, industries], axis=1)
    out = np.empty((f.shape[0], f.shape[1] + 1))
    out[:, 0] = country
    out[:, 1:] = f
    out[:, 1 + industries] -= country[:, None]
    return out


def country_map(n_factors: int, industry_columns: Sequence[int]) -> np.ndarray:
    """T, factors by the basis's factors: the industry model's factor returns from the country
    basis's, f = T g -- each industry the country plus its own, each style itself. A covariance S
    in the basis maps back as T S T'."""
    t = np.zeros((n_factors, n_factors + 1))
    t[:, 1:] = np.eye(n_factors)
    t[np.asarray(industry_columns, dtype=int), 0] = 1.0
    return t


def country_covariance(
    model, estimator, weights: Mapping[date, np.ndarray], industry_columns: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """A fitted model's factor covariance estimated in the country basis by `estimator`: the
    basis's own, the country first, and mapped back to the model's factors (T S T'), both
    annualised as the model is. `weights` by date holds each estimated period's industry weights
    (`country_weights`). Refused for a model whose covariance is not its estimator's alone: one
    with the sampling correction, or the eigenfactor adjustment scaling it."""
    notes = model.notes
    if getattr(notes, "sampling_correction", False) or getattr(notes, "eigen_adjustment", None):
        raise ValueError("this model's covariance is not its estimator's alone (the sampling "
                         "correction, or the eigenfactor adjustment); estimate it without")
    factors = notes.factor_returns
    missing = [day for day in factors.dates if day not in weights]
    if missing:
        raise ValueError(f"no industry weights for {len(missing)} estimated period(s), the first "
                         f"{missing[0]}")
    basis = to_country_basis(factors.returns, industry_columns,
                             np.array([weights[day] for day in factors.dates]))
    scale = notes.frequency.variance_scale if notes.annualised else 1.0
    own = estimator.estimate(basis).matrix * scale
    t = country_map(factors.returns.shape[1], industry_columns)
    return own, t @ own @ t.T
