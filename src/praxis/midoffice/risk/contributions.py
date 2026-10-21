"""Marginal and component contributions to risk, at all three levels.

One identity governs the whole family:

    CTR_i = cov(v_i, X) / sigma        where  sum_i v_i = X  and  sigma = sqrt(cov(X, X))

`v_i` is a position's weighted contribution to the aggregate exposure `X`, so the
contributions sum to `sigma` exactly, by construction rather than by arrangement.  Marginal
contribution is the same expression against the position's *unweighted* exposure: the
derivative of portfolio risk with respect to that position's weight.

That formulation also handles the awkward case for free.  RAT's `car` carries a separate
`BucketSecurity` branch building `w*x_port - bw*x_bench` because a bucketed holding can
carry different exposures on the two sides (`RiskMetrics.cs:2055-2065`); here that is just
what `v_i` is, so there is no branch.

**A note on scale for anyone comparing against RAT.** RAT's `mctr` divides by
`100 * aggTotalRisk` and its `ctr` multiplies by 100 again, so the hundreds cancel and
`ctr`, `pctTr` and everything downstream agree with the definitions here.  Only the
*marginal* numbers differ, by exactly `RAT_MARGINAL_SCALE`: RAT reports marginal risk per
one percent of weight, this reports it per unit. The weight itself is a ratio either way
(`RiskData.expWeight` is `RiskExpValue / RiskMarketValue`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from .basis import FactorBasis
from .exposures import ExposureVector
from .metrics import common_factor_covariance, covariance
from .portfolio import Portfolio, Position

#: RAT's marginal contributions are this multiple of the ones computed here.
RAT_MARGINAL_SCALE = 0.01

_RISK_TOL = 1e-12


class Level(Enum):
    """Which book the contributions decompose."""

    PORTFOLIO = "portfolio"
    BENCHMARK = "benchmark"
    ACTIVE = "active"


@dataclass(frozen=True)
class Contribution:
    """One position's share of a risk number.

    `total` is the component contribution -- these sum to the risk being decomposed.
    `marginal` is the derivative with respect to this position's weight.
    `common_factor` and `specific` split `total` and sum to it.
    """

    security_code: str
    weight: float
    marginal: float
    total: float
    percent: float
    common_factor: float
    specific: float


def _level_spec(
    level: Level,
    scale_p: float = 1.0,
    scale_b: float = 1.0,
) -> tuple[
    Callable[[Portfolio], ExposureVector],
    Callable[[Position], float],
    Callable[[Position], ExposureVector],
    Callable[[Position], ExposureVector],
]:
    """(aggregate vector, position weight, position unweighted vector, position v_i).

    `scale_p` and `scale_b` are the per-side weight-basis factors from
    `Portfolio.weight_scale`. They must reach the weights and the `v_i`, because the
    aggregate being decomposed carries them: `sum(v_i)` has to equal that aggregate exactly
    or the contributions sum to the wrong total. Under `WeightBasis.FULL` both are 1.0 and
    every expression below reduces to its unscaled form.

    The unweighted vector used for the *marginal* is deliberately left unscaled: it is a
    derivative per unit of weight, and the weight it is per unit of is already the scaled one.
    """
    if level is Level.PORTFOLIO:
        return (
            lambda p: p.exposure,
            lambda q: q.weight * scale_p,
            lambda q: q.exposure,
            lambda q: q.weighted() * scale_p,
        )
    if level is Level.BENCHMARK:
        return (
            lambda p: p.benchmark_exposure,
            lambda q: q.benchmark_weight * scale_b,
            lambda q: q.benchmark_exposure,
            lambda q: q.benchmark_weighted() * scale_b,
        )
    return (
        lambda p: p.active_exposure,
        lambda q: q.weight * scale_p - q.benchmark_weight * scale_b,
        # For the marginal, the security's own exposure. Exact whenever the two sides carry
        # the same vector, which is everything but a bucketed holding; `total` stays exact
        # in both cases because it is built from v_i.
        lambda q: q.exposure,
        # Not `q.active_weighted() * scale`: the two sides carry different factors, so the
        # difference has to be taken after each side is scaled.
        lambda q: q.weighted() * scale_p - q.benchmark_weighted() * scale_b,
    )


def contributions(
    portfolio: Portfolio, basis: FactorBasis, level: Level = Level.PORTFOLIO
) -> tuple[Contribution, ...]:
    """Contributions for every position, in portfolio order.

    Positions the risk model does not cover contribute nothing and are returned as zeros
    rather than omitted, so the list lines up with the portfolio and the absence is visible.
    """
    aggregate_of, weight_of, vector_of, contribution_vector_of = _level_spec(
        level,
        portfolio.weight_scale(lambda q: q.weight),
        portfolio.weight_scale(lambda q: q.benchmark_weight),
    )
    aggregate = aggregate_of(portfolio)
    variance = covariance(aggregate, aggregate, basis)
    risk = variance ** 0.5 if variance > _RISK_TOL else 0.0

    out: list[Contribution] = []
    for position in portfolio.positions:
        if not position.covered or risk == 0.0:
            out.append(
                Contribution(position.security_code, weight_of(position), 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            continue

        v = contribution_vector_of(position)
        total_cf = common_factor_covariance(v, aggregate, basis) / risk
        total_sr = v.specific.dot(aggregate.specific) / risk
        total = total_cf + total_sr
        marginal = covariance(vector_of(position), aggregate, basis) / risk
        out.append(
            Contribution(
                security_code=position.security_code,
                weight=weight_of(position),
                marginal=marginal,
                total=total,
                percent=total / risk,
                common_factor=total_cf,
                specific=total_sr,
            )
        )
    return tuple(out)


def check_sum(
    contributions_: Sequence[Contribution], risk: float, *, tolerance: float = 1e-10
) -> float:
    """Return the residual of `sum(total) - risk`, which must be zero up to float error.

    Worth calling in anger rather than only in tests: a non-zero residual means the
    contributions were computed against a different aggregate than the risk they claim to
    decompose, which is the failure mode that survives a visual check of the numbers.
    """
    residual = float(sum(c.total for c in contributions_) - risk)
    if abs(residual) > tolerance:
        raise ValueError(
            f"contributions sum to {sum(c.total for c in contributions_):.12g} but the risk "
            f"being decomposed is {risk:.12g} (residual {residual:.3e}); they were computed "
            "against different aggregates"
        )
    return residual
