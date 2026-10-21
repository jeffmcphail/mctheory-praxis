"""Factor exposures and specific risk, and how they compose.

Two quantities with different algebra travel together because a security's risk profile is
both: its exposures to common factors, and the idiosyncratic part that no factor explains.

Common factor exposures compose linearly -- a portfolio's exposure is the weighted sum of its
positions'.  Specific risk composes by issuer: contributions from the same issuer are
perfectly correlated and add linearly before being squared, contributions from different
issuers are independent and add in quadrature.  That is one sparse dot product over issuer
keys, which is what `SpecificRisk` is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .basis import FactorBasis


@dataclass(frozen=True)
class SpecificRisk:
    """Idiosyncratic risk, aggregated by issuer key.

    `by_issuer[key]` is the signed sum of weight x specific volatility for every position
    belonging to that issuer.  Same key means perfectly correlated, so the sum is taken before
    squaring; different keys are independent, so their squares add.

    **A security with no issuer key is its own issuer, keyed by its security code.** RAT mints
    a synthetic `"DEFAULT<n>"` from a `static int` counter for this case
    (`SpecificRiskVector.cs:16-23`).  The design is right -- unknown issuer correlates with
    nothing -- but the counter is a non-atomic read-modify-write in a library with no `lock`
    anywhere and a background reload thread, so a race mints duplicate keys and silently
    correlates two unrelated securities at 1.  Keying by security code gives identical
    semantics, deterministically, with no shared state.
    """

    by_issuer: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_issuer", dict(self.by_issuer))

    @classmethod
    def for_security(
        cls,
        security_code: str,
        specific_volatility: float,
        *,
        issuer_key: str | None = None,
    ) -> "SpecificRisk":
        key = issuer_key if issuer_key else security_code
        if not key:
            raise ValueError(
                "specific risk needs an issuer key or a security code to key itself by"
            )
        return cls({key: float(specific_volatility)})

    @classmethod
    def zero(cls) -> "SpecificRisk":
        return cls({})

    def dot(self, other: "SpecificRisk") -> float:
        """Covariance of two specific-risk vectors: the sum over shared issuers.

        Issuers present in only one side contribute nothing, which is the independence
        assumption doing its job.
        """
        # Iterate the smaller side; the result is symmetric either way.
        left, right = (
            (self.by_issuer, other.by_issuer)
            if len(self.by_issuer) <= len(other.by_issuer)
            else (other.by_issuer, self.by_issuer)
        )
        return float(sum(v * right[k] for k, v in left.items() if k in right))

    def variance(self) -> float:
        return float(sum(v * v for v in self.by_issuer.values()))

    def risk(self) -> float:
        return float(np.sqrt(self.variance()))

    def __add__(self, other: "SpecificRisk") -> "SpecificRisk":
        merged = dict(self.by_issuer)
        for key, value in other.by_issuer.items():
            merged[key] = merged.get(key, 0.0) + value
        return SpecificRisk(merged)

    def __sub__(self, other: "SpecificRisk") -> "SpecificRisk":
        return self + (-1.0 * other)

    def __mul__(self, scalar: float) -> "SpecificRisk":
        return SpecificRisk({k: v * scalar for k, v in self.by_issuer.items()})

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpecificRisk):
            return NotImplemented
        return self.by_issuer == other.by_issuer


@dataclass(frozen=True)
class ExposureVector:
    """Common-factor exposures plus specific risk, for a position, portfolio or benchmark.

    `exposures` is sparse by design -- only non-zero entries -- and is rendered dense against
    a `FactorBasis` at the point of calculation.  Nothing here carries a padded copy, and
    nothing needs to: see `basis.py` for why that matters.
    """

    exposures: Mapping[int, float]
    specific: SpecificRisk = SpecificRisk({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "exposures", {int(k): float(v) for k, v in self.exposures.items()}
        )

    @classmethod
    def zero(cls) -> "ExposureVector":
        return cls({}, SpecificRisk.zero())

    @property
    def factor_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.exposures))

    def dense(self, basis: FactorBasis) -> np.ndarray:
        return basis.dense(self.exposures)

    def __add__(self, other: "ExposureVector") -> "ExposureVector":
        merged = dict(self.exposures)
        for factor_id, value in other.exposures.items():
            merged[factor_id] = merged.get(factor_id, 0.0) + value
        return ExposureVector(merged, self.specific + other.specific)

    def __sub__(self, other: "ExposureVector") -> "ExposureVector":
        return self + (-1.0 * other)

    def __mul__(self, scalar: float) -> "ExposureVector":
        return ExposureVector(
            {k: v * scalar for k, v in self.exposures.items()},
            self.specific * scalar,
        )

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExposureVector):
            return NotImplemented
        return self.exposures == other.exposures and self.specific == other.specific


def aggregate(vectors: Iterable[ExposureVector]) -> ExposureVector:
    """Sum exposure vectors, typically weighted position vectors into a portfolio.

    Weighting is the caller's job -- scale each vector before summing -- because the weight
    basis (market value, clean market value, exposure value) is a modelling choice this layer
    has no business making.
    """
    total = ExposureVector.zero()
    for vector in vectors:
        total = total + vector
    return total
