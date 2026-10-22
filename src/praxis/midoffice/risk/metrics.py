"""The bilinear form every factor risk number derives from, and the first metrics on it.

One expression underlies the whole metric set:

    cov(p, q) = sigma_spec(p) . sigma_spec(q)  +  x_p' F x_q

Total risk is `sqrt(cov(p, p))`.  Benchmark risk is the same with the benchmark's vector.
Active risk -- tracking error -- is the same with `x_p - x_b`.  Beta is
`cov(p, b) / cov(b, b)`.  Correlation divides by both risks.  Marginal and component
contributions differentiate it.  Everything else is this expression with different arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .basis import FactorBasis, PSDRepair
from .config import DEFAULTS, RiskConfig
from .exposures import ExposureVector

# Variance below this is treated as zero rather than as an error.  After PSD repair the only
# negatives left should be float noise around a genuine zero -- an empty exposure vector, a
# fully hedged position.
_VARIANCE_TOL = 1e-12

# Decimal digits a float64 mantissa carries: -log10(2.22e-16).
MACHINE_DIGITS = -math.log10(np.finfo(float).eps)


def common_factor_abs_term(vector: ExposureVector, basis: FactorBasis) -> float:
    """|x|' |F| |x| -- the sum of the magnitudes the quadratic form adds and subtracts.

    The ratio of this to the variance itself is how much cancellation `x' F x` performs, and
    therefore how much precision it costs: a book whose factor bets offset each other
    computes a small number as the difference of large ones, and loses digits in proportion.
    It costs one extra matrix-vector product, the same order as the variance it accompanies.
    """
    if not len(basis):
        return 0.0
    x = np.abs(vector.dense(basis))
    return float(x @ np.abs(basis.covariance) @ x)


def common_factor_covariance(
    left: ExposureVector, right: ExposureVector, basis: FactorBasis
) -> float:
    """x_left' F x_right, on the given basis."""
    if len(basis) == 0:
        return 0.0
    x = left.dense(basis)
    y = right.dense(basis)
    return float(x @ basis.covariance @ y)


def covariance(
    left: ExposureVector, right: ExposureVector, basis: FactorBasis
) -> float:
    """The full covariance of two exposure vectors: common factor plus specific."""
    return common_factor_covariance(left, right, basis) + left.specific.dot(
        right.specific
    )


def _sqrt_variance(variance: float, what: str) -> float:
    if variance < -abs(_VARIANCE_TOL):
        raise ValueError(
            f"{what} variance is {variance:.6e}, which is negative beyond tolerance. "
            "The covariance was already PSD-repaired when the basis was built, so this is "
            "not a non-PSD vendor matrix -- look at the exposures or the basis construction."
        )
    return float(np.sqrt(max(variance, 0.0)))


def common_factor_variance(vector: ExposureVector, basis: FactorBasis) -> float:
    return common_factor_covariance(vector, vector, basis)


def common_factor_risk(vector: ExposureVector, basis: FactorBasis) -> float:
    return _sqrt_variance(common_factor_variance(vector, basis), "common factor")


def specific_variance(vector: ExposureVector) -> float:
    return vector.specific.variance()


def specific_risk(vector: ExposureVector) -> float:
    return _sqrt_variance(specific_variance(vector), "specific")


def total_variance(vector: ExposureVector, basis: FactorBasis) -> float:
    return covariance(vector, vector, basis)


def total_risk(vector: ExposureVector, basis: FactorBasis) -> float:
    return _sqrt_variance(total_variance(vector, basis), "total")


def beta(
    vector: ExposureVector, benchmark: ExposureVector, basis: FactorBasis
) -> float:
    """cov(p, b) / var(b).  Zero when the benchmark has no risk, rather than dividing by it."""
    denominator = total_variance(benchmark, basis)
    if denominator <= _VARIANCE_TOL:
        return 0.0
    return covariance(vector, benchmark, basis) / denominator


def correlation(
    left: ExposureVector, right: ExposureVector, basis: FactorBasis
) -> float:
    denominator = total_risk(left, basis) * total_risk(right, basis)
    if denominator <= _VARIANCE_TOL:
        return 0.0
    return covariance(left, right, basis) / denominator


@dataclass(frozen=True)
class RiskDecomposition:
    """A risk number together with what produced it.

    The provenance fields are the point.  RAT computes all of this and writes the
    qualifications -- a substituted model date, a repaired matrix, a proxied factor -- to a log
    stream running alongside the results.  Attaching them to the number instead is the single
    structural change this port makes, and it is what lets a run be reconstructed later rather
    than merely re-run.
    """

    total_risk: float
    common_factor_risk: float
    specific_risk: float
    total_variance: float
    common_factor_variance: float
    specific_variance: float

    n_factors: int
    covariance_repair: PSDRepair
    proxied_factors: Mapping[int, int] = field(default_factory=dict)
    abs_term: float = 0.0
    warn_digits: float = DEFAULTS.conditioning_warn_digits

    @property
    def amplification(self) -> float:
        """How much the arithmetic magnifies rounding error in this risk number.

        `(|x||F||x| + specific_variance) / total_variance`, which is at least 1.  The
        specific term appears on top because it is a sum of squares and cancels nothing; all
        the cancellation lives in the quadratic form.
        """
        if self.total_variance <= 0.0:
            return float("inf") if self.abs_term > 0.0 else 1.0
        return max((self.abs_term + self.specific_variance) / self.total_variance, 1.0)

    @property
    def digits_intact(self) -> float:
        """Decimal digits of this risk number that survive the arithmetic.

        A conservative bound: it assumes every rounding error aligns, which measurement shows
        understates reality by roughly 2.6 digits.  Conservative is the right direction for a
        warning.  The second term is the pairwise-summation cost, which grows with log2 of
        the factor count rather than with the count itself.

        **Negative values are meaningful**: they say the rounding error exceeds the number
        itself, so not one digit is correct.  A total cancellation returns `-inf` rather than
        zero, because zero would report a *better* figure for the worst possible case than an
        astronomical-but-finite amplification does, and a diagnostic that orders its own
        severity backwards is worse than none.
        """
        amplification = self.amplification
        if not math.isfinite(amplification):
            return float("-inf")
        summation = math.log10(math.log2(max(self.n_factors, 2)))
        return MACHINE_DIGITS - math.log10(amplification) - summation

    @property
    def is_well_conditioned(self) -> bool:
        return self.digits_intact >= self.warn_digits

    @property
    def is_exact(self) -> bool:
        """True when nothing was substituted or repaired to produce this number."""
        return not self.covariance_repair.repaired and not self.proxied_factors

    def qualifications(self) -> tuple[str, ...]:
        """Everything a reader needs to know before trusting this number."""
        notes: list[str] = []
        if self.covariance_repair.repaired:
            notes.append(self.covariance_repair.describe())
        if self.proxied_factors:
            notes.append(
                f"{len(self.proxied_factors)} factor(s) used a proxy row in the "
                f"covariance: {dict(list(self.proxied_factors.items())[:5])}"
            )
        if not self.is_well_conditioned:
            digits = self.digits_intact
            if not math.isfinite(digits):
                notes.append(
                    "this book's factor bets cancel exactly, so the variance is the "
                    "difference of equal magnitudes and the risk number is entirely rounding "
                    "debris -- not one digit of it is correct. The covariance has a null "
                    "direction the book lies in: raise psd_relative_floor above 0."
                )
            else:
                surviving = (
                    f"only about {digits:.1f} decimal digits of the risk number survive, "
                    f"against a {self.warn_digits:.0f}-digit threshold"
                    if digits > 0.0
                    else "the rounding error exceeds the risk number itself, so no digit of "
                    "it is correct"
                )
                notes.append(
                    f"this book's factor bets offset each other, so the variance is computed "
                    f"as a near-total cancellation (amplification {self.amplification:.2e}); "
                    f"{surviving}. Report fewer significant figures, or raise "
                    f"psd_relative_floor."
                )
        return tuple(notes)


def decompose(
    vector: ExposureVector, basis: FactorBasis, *, config: RiskConfig | None = None
) -> RiskDecomposition:
    """Total risk split into its common-factor and specific parts, with provenance.

    The identity `total_variance == common_factor_variance + specific_variance` holds exactly:
    the two parts are computed independently and are orthogonal by construction, since no
    factor exposure appears in the specific term and no issuer appears in the factor term.
    `test_variance_decomposition_is_exact` pins it.
    """
    config = config or DEFAULTS
    cfv = common_factor_variance(vector, basis)
    sv = specific_variance(vector)
    tv = cfv + sv
    return RiskDecomposition(
        total_risk=_sqrt_variance(tv, "total"),
        common_factor_risk=_sqrt_variance(cfv, "common factor"),
        specific_risk=_sqrt_variance(sv, "specific"),
        total_variance=tv,
        common_factor_variance=cfv,
        specific_variance=sv,
        n_factors=len(basis),
        covariance_repair=basis.repair,
        proxied_factors=dict(basis.proxied),
        abs_term=common_factor_abs_term(vector, basis),
        warn_digits=config.conditioning_warn_digits,
    )
