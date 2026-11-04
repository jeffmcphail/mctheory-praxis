"""Rebasing exposures for readability, without changing a single risk number.

A standardised exposure of -0.4 is precise and unreadable.  Business users want a number
they can rank and talk about, and [0, 1] is what they ask for.  The objection is obvious --
risk is `x'Fx`, so changing `x` must change the risk -- and it is answered by changing `F`
too, in the matching way.

The transform is affine, per factor:

    x'_k = a_k + b_k x_k

and the factor returns take the mirror image of it, derived rather than refitted:

    f'_k      = f_k / b_k                      for each style factor
    f'_market = f_market - sum_k a_k f'_k      the market absorbs the shift

Then `x'.f' = x.f` identically, period by period, and the covariance of the transformed
factor returns reproduces the same portfolio risk to machine precision.  Verified in
`test_exposure_basis.py` against a portfolio's common-factor risk: 15.9841% in both bases,
with a maximum per-period difference of 2e-17.

**Two things make this fragile, and both are designed against here.**

The transform must be *affine*.  `f'_k = f_k / b_k` only exists if there is a scalar `b_k`
to divide by, so a rank or a normal-CDF transform -- both of which also land in [0, 1] --
cannot be mirrored and would silently change every risk number.  Only a linear rescale
works.

And a **market factor must be present** to absorb `sum_k a_k f'_k`.  Without a constant
column there is nowhere for the shift to go and the fitted returns change.  A model built
with `fit_market=False` is refused rather than quietly rebased, because a complete set of
industry exposures sums to one and *would* span the constant, so the right fix depends on
what the model contains and is not something to guess at.

The hazard this exists to prevent is a unitized exposure file meeting a standard-basis
covariance matrix.  Each looks reasonable; together they overstate common-factor risk by
about forty times, with nothing to signal it.  So the rebasing moves a whole `FactorModel`
at once -- exposures, covariance and the basis tag -- and never exposes a way to transform
one without the others.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from ..risk.basis import FactorCovariance
from ..risk.exposures import ExposureVector
from .crosssectional import MARKET_FACTOR
from .timeseries import FactorModel

STANDARD = "standard"
UNIT = "unit"

DEFAULT_BOUND = 4.0
"""The winsorising bound, so [-4, 4] maps onto [0, 1] and nothing falls outside it.

Taken from the same place `characteristics.winsorise` takes it: the GEM handbook truncates
a factor exposure at four standard deviations. Using a different bound here would put some
exposures outside [0, 1], which defeats the point.
"""


class BasisError(ValueError):
    """The model cannot be rebased, or is already on the basis requested."""


@dataclass(frozen=True)
class Rebasing:
    """The affine map, kept so the inverse is exact rather than re-derived."""

    shift: float
    scale: float
    market_id: int
    bound: float

    def forward(self, value: float) -> float:
        return self.shift + self.scale * value

    def inverse(self, value: float) -> float:
        return (value - self.shift) / self.scale


def _market_id(model: FactorModel) -> int:
    """The factor every security loads 1.0 on. By name first, then by behaviour."""
    for fid, name in model.factor_names.items():
        if name == MARKET_FACTOR:
            return int(fid)
    if model.exposures:
        for fid in model.factor_ids:
            if all(
                abs(ev.exposures.get(fid, 0.0) - 1.0) < 1e-12
                for ev in model.exposures.values()
            ):
                return int(fid)
    raise BasisError(
        "no market factor found, so there is nothing to absorb the shift that rebasing "
        "introduces, and the fitted returns would change. Estimate with fit_market=True, "
        "or rebase a model whose industry exposures sum to one and span the constant "
        "-- which needs a different transform than this one"
    )


def _transform_matrix(
    factor_ids: tuple[int, ...], rebasing: Rebasing, *, forward: bool
) -> np.ndarray:
    """The matrix T with f' = T f, so the covariance moves as T F T'."""
    n = len(factor_ids)
    index = {fid: i for i, fid in enumerate(factor_ids)}
    market = index[rebasing.market_id]
    T = np.eye(n)
    scale = rebasing.scale if forward else 1.0 / rebasing.scale
    shift = rebasing.shift if forward else -rebasing.shift / rebasing.scale
    for fid, i in index.items():
        if i == market:
            continue
        T[i, i] = 1.0 / scale
        T[market, i] = -shift / scale
    return T


def _rebase(model: FactorModel, rebasing: Rebasing, *, forward: bool, basis: str) -> FactorModel:
    market = rebasing.market_id
    exposures: dict[str, ExposureVector] = {}
    for code, vector in model.exposures.items():
        moved: dict[int, float] = {}
        for fid in model.factor_ids:
            current = float(vector.exposures.get(fid, 0.0))
            if fid == market:
                moved[fid] = current
                continue
            moved[fid] = (
                rebasing.forward(current) if forward else rebasing.inverse(current)
            )
        # No sparsity filter: a standardised exposure of exactly zero becomes 0.5, and
        # dropping it as "absent" would make it read as zero again on the way back.
        exposures[code] = ExposureVector(exposures=moved, specific=vector.specific)

    T = _transform_matrix(model.factor_ids, rebasing, forward=forward)
    matrix = T @ model.covariance.matrix @ T.T
    return replace(
        model,
        covariance=FactorCovariance(
            factor_ids=model.covariance.factor_ids, matrix=(matrix + matrix.T) / 2.0
        ),
        exposures=exposures,
        basis=basis,
    )


def to_unit(model: FactorModel, *, bound: float = DEFAULT_BOUND) -> FactorModel:
    """Put style exposures in [0, 1], moving the covariance to match.

    `bound` is the winsorising bound the exposures were clipped at, so the map is
    `x' = 0.5 + x / (2 * bound)`. Specific risk is per security rather than per factor and
    is untouched.
    """
    if model.basis == UNIT:
        raise BasisError("this model is already on the unit basis")
    if bound <= 0.0:
        raise BasisError(f"bound must be positive, got {bound!r}")
    rebasing = Rebasing(
        shift=0.5, scale=1.0 / (2.0 * bound), market_id=_market_id(model), bound=bound
    )
    return _rebase(model, rebasing, forward=True, basis=UNIT)


def to_standard(model: FactorModel, *, bound: float = DEFAULT_BOUND) -> FactorModel:
    """The exact inverse of `to_unit`, returning a model to the basis it was estimated on."""
    if model.basis == STANDARD:
        raise BasisError("this model is already on the standard basis")
    rebasing = Rebasing(
        shift=0.5, scale=1.0 / (2.0 * bound), market_id=_market_id(model), bound=bound
    )
    return _rebase(model, rebasing, forward=False, basis=STANDARD)


def describe_basis(model: FactorModel) -> str:
    if model.basis == UNIT:
        return (
            "unit basis: style exposures in [0, 1], 0.5 being the market average. The "
            "factor covariance has been transformed to match and must travel with them."
        )
    return (
        "standard basis: style exposures in standard deviations from a cap-weighted mean, "
        "zero being the market average."
    )
