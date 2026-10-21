"""Compare two record sets, and say precisely where they disagree.

Two decisions here carry most of the weight.

**Tolerances are derived from the case's recorded conventions, not chosen by hand.**  The
engine ports RAT's unrefined Acklam inverse normal, whose relative error is 1.15e-9; the
reference uses Wichura's AS241, which is accurate to float precision.  So the tail numbers
*must* differ by about 1.15e-9 when a case says `refine=False`, and must agree to float
noise when it says `refine=True`.  A single hand-picked tolerance can only be loose enough
to accept the first -- at which point it would also accept a real tail bug a hundred times
larger than float noise.  `policy_for` reads the conventions and tightens or loosens
accordingly, which is the reason those conventions live in the fixture at all.

**Absence is a result.**  A comparator that loops over one side can only report on numbers
that side produced: if the engine stops emitting a metric, every number it still emits
matches and the run passes.  So missing and extra keys are first-class findings, and a
known, documented gap must be declared -- a gap that is not declared fails, and a declared
gap that has quietly been filled fails too, so the list cannot rot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Iterable, Mapping

from .case import ParityCase
from .record import (
    EXPECTED_SHORTFALL,
    Record,
    VALUE_AT_RISK,
    index,
)
from ..risk.tail import ACKLAM_RELATIVE_ERROR

Key = tuple[str, str, str, str]

# Float noise for an O(1) number accumulated over a few hundred thousand terms is on the
# order of 1e-13 relative; 1e-10 leaves three orders of headroom while staying far below any
# error a wrong formula would produce.
DEFAULT_RTOL = 1e-10

# Absolute floor, for numbers that are legitimately near zero -- a contribution from a
# position with negligible weight, say -- where a relative test is meaningless. Fixtures are
# expressed in decimal returns, so O(1) numbers; a fixture in basis points scales everything
# and the relative term carries it.
DEFAULT_ATOL = 1e-13

# Margin on the derived tail bounds. The bounds below are first-order in the z error, and
# Acklam's actual error varies across the domain rather than sitting at its worst case
# everywhere, so a modest factor covers both without turning the test into a formality.
_TAIL_MARGIN = 1.5


@dataclass(frozen=True)
class Tolerance:
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL

    def holds(self, expected: float, actual: float) -> bool:
        return abs(actual - expected) <= self.atol + self.rtol * abs(expected)

    def describe(self) -> str:
        return f"rtol={self.rtol:g}, atol={self.atol:g}"


@dataclass(frozen=True)
class TolerancePolicy:
    default: Tolerance = field(default_factory=Tolerance)
    by_metric: Mapping[str, Tolerance] = field(default_factory=dict)
    rationale: str = ""

    def for_metric(self, metric: str) -> Tolerance:
        return self.by_metric.get(metric, self.default)


def policy_for(case: ParityCase) -> TolerancePolicy:
    """Derive tolerances from the conventions the case records.

    The only metrics that get special treatment are the tail measures, and only because the
    two sides deliberately use different inverse-normal algorithms.
    """
    if not case.tails:
        return TolerancePolicy(rationale="no tail measures in this case")

    unrefined = [t for t in case.tails if not t.refine]
    if not unrefined:
        return TolerancePolicy(
            rationale=(
                "every tail refines z, so both sides are at float precision and the "
                "default tolerance applies throughout"
            )
        )

    # The two tail measures do NOT inherit the z error equally, and assuming they do is an
    # easy and quiet mistake -- it was made in the first draft of this file and the harness
    # caught it.
    #
    #   VaR = sigma * z * sqrt(h/252)          d ln VaR / d ln z = 1
    #   ES  = sigma * phi(z)/(1-c) * sqrt(...) d ln ES  / d z     = -z
    #
    # Acklam's error is relative, so dz = rtol_z * |z|. VaR therefore carries rtol_z
    # unchanged, while ES carries |z| * dz = z^2 * rtol_z. At 95% that is a factor of 2.7,
    # and at 99% a factor of 5.4 -- far enough above float noise that a single shared
    # tolerance is either too loose for VaR or too tight for ES.
    z_max = max(abs(NormalDist().inv_cdf(t.confidence)) for t in unrefined)
    var_rtol = _TAIL_MARGIN * ACKLAM_RELATIVE_ERROR
    es_rtol = _TAIL_MARGIN * ACKLAM_RELATIVE_ERROR * z_max * z_max
    return TolerancePolicy(
        by_metric={
            VALUE_AT_RISK: Tolerance(rtol=var_rtol, atol=DEFAULT_ATOL),
            EXPECTED_SHORTFALL: Tolerance(rtol=es_rtol, atol=DEFAULT_ATOL),
        },
        rationale=(
            f"{len(unrefined)} of {len(case.tails)} tail measure(s) use RAT's unrefined "
            f"Acklam z (relative error {ACKLAM_RELATIVE_ERROR:g}); VaR is linear in z so it "
            f"gets rtol={var_rtol:g}, expected shortfall scales as z^2 (z_max={z_max:.4f}) "
            f"so it gets rtol={es_rtol:g}; everything else keeps rtol={DEFAULT_RTOL:g}"
        ),
    )


@dataclass(frozen=True)
class Mismatch:
    key: Key
    expected: float
    actual: float
    tolerance: Tolerance

    @property
    def abs_diff(self) -> float:
        return abs(self.actual - self.expected)

    @property
    def rel_diff(self) -> float:
        return self.abs_diff / abs(self.expected) if self.expected else float("inf")

    def describe(self) -> str:
        metric, level, entity, detail = self.key
        where = ".".join(x for x in (metric, level, entity, detail) if x)
        return (
            f"{where}: expected {self.expected!r}, got {self.actual!r} "
            f"(abs {self.abs_diff:.3e}, rel {self.rel_diff:.3e}, "
            f"tolerance {self.tolerance.describe()})"
        )


@dataclass(frozen=True)
class Report:
    compared: int
    matched: int
    mismatches: tuple[Mismatch, ...]
    missing: tuple[Key, ...]
    extra: tuple[Key, ...]
    expected_gaps: tuple[Key, ...]
    stale_gaps: tuple[Key, ...]
    policy: TolerancePolicy
    case_name: str = ""

    @property
    def ok(self) -> bool:
        return not (self.mismatches or self.missing or self.extra or self.stale_gaps)

    def describe(self) -> str:
        lines = [
            f"parity report for {self.case_name or '(unnamed case)'}: "
            f"{self.matched}/{self.compared} matched"
        ]
        if self.policy.rationale:
            lines.append(f"  tolerance: {self.policy.rationale}")
        if self.expected_gaps:
            lines.append(f"  {len(self.expected_gaps)} declared gap(s), as expected")
        if self.stale_gaps:
            lines.append(
                f"  {len(self.stale_gaps)} declared gap(s) are no longer gaps -- "
                f"remove them from the list:"
            )
            lines.extend(f"    {'.'.join(x for x in k if x)}" for k in self.stale_gaps)
        if self.missing:
            lines.append(f"  {len(self.missing)} undeclared missing from actual:")
            lines.extend(f"    {'.'.join(x for x in k if x)}" for k in self.missing)
        if self.extra:
            lines.append(f"  {len(self.extra)} present in actual but not expected:")
            lines.extend(f"    {'.'.join(x for x in k if x)}" for k in self.extra)
        if self.mismatches:
            lines.append(f"  {len(self.mismatches)} numeric mismatch(es), worst first:")
            lines.extend(f"    {m.describe()}" for m in self.mismatches)
        if self.ok:
            lines.append("  no disagreement")
        return "\n".join(lines)


def compare(
    expected: Iterable[Record],
    actual: Iterable[Record],
    *,
    policy: TolerancePolicy | None = None,
    known_gaps: Mapping[Key, str] | None = None,
    case_name: str = "",
) -> Report:
    """Compare two record sets key by key.

    `expected` is the reference side and `actual` the engine side, but nothing here depends
    on that: the comparison is symmetric apart from which side `known_gaps` excuses.
    """
    policy = policy or TolerancePolicy()
    gaps = dict(known_gaps or {})

    left = index(expected)
    right = index(actual)

    missing_keys = sorted(set(left) - set(right))
    extra_keys = sorted(set(right) - set(left))

    declared = [k for k in missing_keys if k in gaps]
    undeclared = [k for k in missing_keys if k not in gaps]
    # A declared gap that is present on both sides is stale: the engine has grown the metric
    # and the list now excuses something that no longer needs excusing.
    stale = sorted(k for k in gaps if k in left and k in right)

    mismatches: list[Mismatch] = []
    matched = 0
    for key in sorted(set(left) & set(right)):
        tolerance = policy.for_metric(key[0])
        if tolerance.holds(left[key], right[key]):
            matched += 1
        else:
            mismatches.append(Mismatch(key, left[key], right[key], tolerance))

    mismatches.sort(key=lambda m: (-m.rel_diff, m.key))

    return Report(
        compared=len(set(left) & set(right)),
        matched=matched,
        mismatches=tuple(mismatches),
        missing=tuple(undeclared),
        extra=tuple(extra_keys),
        expected_gaps=tuple(declared),
        stale_gaps=tuple(stale),
        policy=policy,
        case_name=case_name,
    )
