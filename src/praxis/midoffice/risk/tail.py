"""Value at risk and expected shortfall, under the normal assumption RAT makes.

Both are a **linear rescaling of risk**:

    VaR = sigma * z * sqrt(h / 252)                 z = inverse_normal_cdf(confidence)
    ES  = sigma * phi(z) / (1 - c) * sqrt(h / 252)

which means every decomposition already built carries over for free.  Contributions sum to
risk, so contributions scaled by the same multiplier sum to VaR.  There is no separate VaR
decomposition to write -- `scale` does it.

**A naming trap in the original, worth stating plainly.** RAT registers `VAR`, `CONTRIB_VAR`,
`CONDITIONAL_VAR` and `CONTRIB_CONDITIONAL_VAR` at group level, so its position-level `cvar`
means *contribution to* VaR, not conditional VaR; `conditionalVar` is the expected shortfall.
Reading `cvar` as conditional VaR gives a number that is wrong by a factor of the weight.
Nothing here is called `cvar`.

These are normal-distribution numbers.  Returns are not normal and the tails are the part
that is least normal, so these understate real tail loss.  RAT makes the same assumption and
the port reproduces it; a reader of the output should know which model produced it, which is
what `TailSpec` on the result is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

#: Acklam's rational approximation, as RAT uses it, without a refinement step.
ACKLAM_RELATIVE_ERROR = 1.15e-9

_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def inverse_normal_cdf(p: float, *, refine: bool = False) -> float:
    """Acklam's inverse normal CDF, matching RAT's `InvCumulativeNormalDistribution`.

    `refine` adds one Halley step, taking the result to roughly machine precision. RAT does
    not refine, so leave it off for parity and turn it on if the extra digits are wanted;
    the difference is around `ACKLAM_RELATIVE_ERROR` and is far below any tolerance a risk
    number is read at.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p > 1.0 - _P_LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )

    if refine:
        e = normal_cdf(x) - p
        u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
        x = x - u / (1.0 + x * u / 2.0)
    return x


@dataclass(frozen=True)
class TailSpec:
    """The assumptions a tail number was computed under. Travels with the result.

    `trading_days` is the annualisation basis; RAT hardcodes 252
    (`PortfolioRiskData.getVarMultiplier`).
    """

    confidence: float = 0.95
    horizon_days: float = 1.0
    trading_days: float = 252.0
    refine_inverse_normal: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(
                f"confidence must be strictly between 0 and 1, got {self.confidence}"
            )
        if self.horizon_days <= 0.0:
            raise ValueError(f"horizon must be positive, got {self.horizon_days}")
        if self.trading_days <= 0.0:
            raise ValueError(f"trading days must be positive, got {self.trading_days}")

    @property
    def z(self) -> float:
        return inverse_normal_cdf(self.confidence, refine=self.refine_inverse_normal)

    @property
    def horizon_scale(self) -> float:
        return math.sqrt(self.horizon_days / self.trading_days)

    @property
    def var_multiplier(self) -> float:
        """RAT's `getVarMultiplier`: sqrt(h / 252) * inverse_normal(confidence)."""
        return self.horizon_scale * self.z

    @property
    def expected_shortfall_multiplier(self) -> float:
        """phi(z) / (1 - c) * sqrt(h / 252)."""
        return self.horizon_scale * normal_pdf(self.z) / (1.0 - self.confidence)

    def describe(self) -> str:
        return (
            f"normal VaR at {self.confidence:.1%} over {self.horizon_days:g} day(s) "
            f"({self.trading_days:g}-day year)"
        )


def value_at_risk(risk: float, spec: TailSpec) -> float:
    """Normal VaR as a positive loss magnitude."""
    return risk * spec.var_multiplier


def expected_shortfall(risk: float, spec: TailSpec) -> float:
    """Expected loss given the VaR threshold is breached. RAT calls this `conditionalVar`."""
    return risk * spec.expected_shortfall_multiplier


@dataclass(frozen=True)
class TailRisk:
    """VaR and expected shortfall for one risk number, with the assumptions behind them."""

    risk: float
    value_at_risk: float
    expected_shortfall: float
    spec: TailSpec

    @property
    def shortfall_ratio(self) -> float:
        """ES / VaR. Depends only on confidence, never on the portfolio."""
        return (
            self.expected_shortfall / self.value_at_risk
            if self.value_at_risk
            else 0.0
        )


def tail_risk(risk: float, spec: TailSpec | None = None) -> TailRisk:
    spec = spec or TailSpec()
    return TailRisk(
        risk=risk,
        value_at_risk=value_at_risk(risk, spec),
        expected_shortfall=expected_shortfall(risk, spec),
        spec=spec,
    )


def scale(values: Iterable[float], multiplier: float) -> tuple[float, ...]:
    """Rescale a set of risk contributions into VaR or expected-shortfall terms.

    Because both tail measures are `risk * constant`, and contributions sum to risk, scaled
    contributions sum to the scaled measure. That is the whole of VaR decomposition: there
    is no separate calculation, and writing one would be an opportunity to disagree with the
    risk decomposition it is supposed to match.
    """
    return tuple(v * multiplier for v in values)
