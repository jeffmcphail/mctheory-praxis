"""An independent scalar implementation of the risk algebra.

This module exists to answer one question the engine cannot answer about itself: are the
numbers right, or merely self-consistent?  The engine's unit tests check the engine against
hand-computed cases and against its own identities.  That is real verification, but a test
suite written alongside an implementation shares its author's misreadings, and an identity
like "contributions sum to risk" holds just as well for a consistently wrong formulation.

So the arithmetic here is deliberately *not* the engine's:

* No numpy in the computation.  The quadratic form is a sorted double loop over sparse
  dicts, which accumulates in a different order from BLAS.  Agreement to 1e-12 across
  randomised cases is therefore not an artifact of a shared summation order.
* Active exposure is built by differencing the two aggregates.  The engine aggregates
  per-position differences.  Linearity says these agree; doing it the other way round means
  a bug in either one shows up rather than cancelling.
* `z` comes from `statistics.NormalDist().inv_cdf`, which is Wichura's AS241 -- a different
  algorithm from the Acklam approximation the engine ports from RAT.  The two therefore
  disagree by about Acklam's documented error, and the test asserts that bound rather than
  asserting equality.  An assertion of equality here would only be satisfied by copying
  RAT's coefficients into the reference, which would test nothing.
* Every formula is written from the documented definition, not from the engine source.
  Where the two disagree, the disagreement is the finding: either the code or the docstring
  is wrong, and a harness that reproduced the code could not tell.

Scope is narrow on purpose.  Cases using factor proxies or a non-zero PSD floor are refused
rather than approximated: both have semantics that are surprising by design, and a reference
that guessed at them would manufacture disagreements that look like engine bugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Mapping

from .case import ParityCase, PositionSpec
from .record import (
    BETA,
    CORRELATION,
    COMMON_FACTOR_RISK,
    COVERAGE_COUNT,
    COVERAGE_WEIGHT,
    CTR_COMMON_FACTOR,
    CTR_MARGINAL,
    CTR_PERCENT,
    CTR_SPECIFIC,
    CTR_TOTAL,
    EXPECTED_SHORTFALL,
    ROW_CONTRIBUTION,
    ROW_EXPOSURE,
    ROW_LITTERMAN_PROPORTION,
    ROW_MARGINAL,
    ROW_RISK_PROPORTION,
    ROW_STANDALONE,
    SPECIFIC_RISK,
    TOTAL_RISK,
    TRACKING_ERROR,
    VALUE_AT_RISK,
    WEIGHT,
    Record,
)

TRADING_DAYS = 252
SPECIFIC_ROW_NAME = "Specific Risk"
_RISK_TOL = 1e-12

LEVELS = ("portfolio", "benchmark", "active")


class UnsupportedCase(ValueError):
    """A case whose semantics this reference deliberately does not reproduce."""


# --- sparse linear algebra, by hand ------------------------------------------------------


def _dot_specific(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Specific covariance: issuer keys that match multiply, the rest contribute nothing.

    This is the whole of RAT's `SpecificRiskVector` inner product
    (`SpecificRiskVector.cs:198`): same key multiplies, different key gives zero.  Two
    securities of one issuer are specifically correlated; two unrelated names are not.
    """
    total = 0.0
    for key in sorted(a):
        other = b.get(key)
        if other is not None:
            total += a[key] * other
    return total


def _quad(
    x: Mapping[int, float], cov: Mapping[int, Mapping[int, float]], y: Mapping[int, float]
) -> float:
    """x' F y, as a sorted double loop over the sparse entries."""
    total = 0.0
    for a in sorted(x):
        row = cov[a]
        xa = x[a]
        if xa == 0.0:
            continue
        for b in sorted(y):
            yb = y[b]
            if yb != 0.0:
                total += xa * row[b] * yb
    return total


@dataclass(frozen=True)
class _Vector:
    """A factor exposure map and a specific-risk map, keyed by issuer."""

    exposures: dict[int, float]
    specific: dict[str, float]

    def scaled(self, k: float) -> "_Vector":
        return _Vector(
            {f: v * k for f, v in self.exposures.items()},
            {s: v * k for s, v in self.specific.items()},
        )

    def minus(self, other: "_Vector") -> "_Vector":
        exposures = dict(self.exposures)
        for f, v in other.exposures.items():
            exposures[f] = exposures.get(f, 0.0) - v
        specific = dict(self.specific)
        for s, v in other.specific.items():
            specific[s] = specific.get(s, 0.0) - v
        return _Vector(exposures, specific)

    @staticmethod
    def zero() -> "_Vector":
        return _Vector({}, {})


def _cov(a: _Vector, b: _Vector, cov: Mapping[int, Mapping[int, float]]) -> float:
    return _dot_specific(a.specific, b.specific) + _quad(a.exposures, cov, b.exposures)


def _risk(v: _Vector, cov: Mapping[int, Mapping[int, float]]) -> float:
    return math.sqrt(max(_cov(v, v, cov), 0.0))


# --- case preparation --------------------------------------------------------------------


def _issuer(p: PositionSpec) -> str:
    return p.issuer_key or p.security_code


def _own_vector(p: PositionSpec, *, benchmark_side: bool) -> _Vector:
    """A security's own unweighted exposure on one side of the book."""
    exposures = (
        p.benchmark_exposures
        if benchmark_side and p.benchmark_exposures is not None
        else p.exposures
    )
    specific = {_issuer(p): p.specific_volatility} if p.specific_volatility else {}
    return _Vector(dict(exposures), specific)


def _covariance_map(case: ParityCase) -> dict[int, dict[int, float]]:
    ids = case.factor_ids
    return {
        a: {b: case.covariance[i][j] for j, b in enumerate(ids)} for i, a in enumerate(ids)
    }


def _check_supported(case: ParityCase) -> None:
    if case.proxies:
        raise UnsupportedCase(
            f"case {case.name!r} uses factor proxies, whose semantics (factors proxied to "
            f"one target become perfectly correlated and share its variance) this reference "
            f"deliberately does not reproduce; proxy behaviour is covered by the engine's "
            f"own unit tests"
        )
    if case.relative_floor != 0.0:
        raise UnsupportedCase(
            f"case {case.name!r} sets relative_floor={case.relative_floor!r}; this reference "
            f"requires a covariance that needs no PSD repair, so that the slice the engine "
            f"builds and the full matrix used here are the same matrix"
        )


def _assert_psd(case: ParityCase) -> None:
    """Guard, not computation: refuse a case whose covariance is not PSD.

    The engine slices the covariance down to the factors in play and repairs the slice if
    needed; this reference uses the full matrix.  Those agree only when no repair fires, and
    a principal submatrix of a PSD matrix is PSD -- so requiring PSD here is exactly the
    condition under which the two are comparable.

    numpy is used for the eigenvalue check and nowhere else.  The check is a precondition on
    the input, not part of the arithmetic under test.
    """
    if not case.factors:
        return
    import numpy as np

    eigenvalues = np.linalg.eigvalsh(np.array(case.covariance, dtype=float))
    smallest = float(eigenvalues.min())
    scale = max(1.0, float(abs(eigenvalues).max()))
    if smallest < -1e-12 * scale:
        raise UnsupportedCase(
            f"covariance in case {case.name!r} is not PSD (smallest eigenvalue "
            f"{smallest:.3e}); the engine would repair the slice it builds and this "
            f"reference would not, so the two are not comparable"
        )


# --- the computation ---------------------------------------------------------------------


def reference_records(case: ParityCase) -> tuple[Record, ...]:
    """Every number the engine claims to produce, computed independently."""
    _check_supported(case)
    _assert_psd(case)
    cov = _covariance_map(case)

    covered = [p for p in case.positions if p.covered]

    # Weight basis. `full` leaves uncovered names in the denominator, so covered weights sum
    # to less than one; `covered` renormalises each side over its own covered names. The
    # renormalisation has to reach the position weights as well as the aggregate, or the
    # contributions would no longer sum to the risk they decompose.
    # The factor is full/covered -- the uncovered weight redistributed across the covered
    # names, preserving the book's invested total -- and NOT 1/covered. The first draft of
    # this reference used 1/covered, which is what "renormalise over covered names" naturally
    # suggests. The two agree for any book whose weights sum to one, so the disagreement
    # surfaced only once the generator started producing books that do not.
    def _scale(weight_of) -> float:
        covered_sum = sum(weight_of(q) for q in covered)
        full_sum = sum(weight_of(q) for q in case.positions)
        if abs(covered_sum) <= 1e-15 or abs(full_sum - covered_sum) <= 1e-15:
            return 1.0
        return full_sum / covered_sum

    scale_p = scale_b = 1.0
    if case.weight_basis == "covered":
        scale_p = _scale(lambda q: q.weight)
        scale_b = _scale(lambda q: q.benchmark_weight)

    def weight_of(p: PositionSpec, level: str) -> float:
        if level == "portfolio":
            return p.weight * scale_p
        if level == "benchmark":
            return p.benchmark_weight * scale_b
        return p.weight * scale_p - p.benchmark_weight * scale_b

    def own_of(p: PositionSpec, level: str) -> _Vector:
        # The marginal is taken against the security's own exposure. On the active level that
        # is the portfolio-side vector, exact whenever both sides carry the same one.
        return _own_vector(p, benchmark_side=(level == "benchmark"))

    def contribution_vector(p: PositionSpec, level: str) -> _Vector:
        port = _own_vector(p, benchmark_side=False).scaled(p.weight * scale_p)
        bench = _own_vector(p, benchmark_side=True).scaled(p.benchmark_weight * scale_b)
        if level == "portfolio":
            return port
        if level == "benchmark":
            return bench
        return port.minus(bench)

    # Aggregates. Built per side and then differenced -- the opposite order from the engine.
    def aggregate(level: str) -> _Vector:
        if level == "active":
            return aggregate("portfolio").minus(aggregate("benchmark"))
        exposures: dict[int, float] = {}
        specific: dict[str, float] = {}
        for p in covered:
            v = contribution_vector(p, level)
            for f, value in v.exposures.items():
                exposures[f] = exposures.get(f, 0.0) + value
            for s, value in v.specific.items():
                specific[s] = specific.get(s, 0.0) + value
        return _Vector(exposures, specific)

    aggregates = {level: aggregate(level) for level in LEVELS}

    # The basis: every factor in play on either side, which is what the engine spans.
    basis_ids = sorted(
        set(aggregates["portfolio"].exposures) | set(aggregates["benchmark"].exposures)
    )

    out: list[Record] = []

    def emit(metric: str, level: str, entity: str, detail: str, value: float) -> None:
        out.append(Record(metric, level, entity, detail, value))

    # -- book level ----------------------------------------------------------------
    for level in LEVELS:
        v = aggregates[level]
        emit(TOTAL_RISK, level, "", "", _risk(v, cov))
        emit(COMMON_FACTOR_RISK, level, "", "", math.sqrt(max(_quad(v.exposures, cov, v.exposures), 0.0)))
        emit(SPECIFIC_RISK, level, "", "", math.sqrt(max(_dot_specific(v.specific, v.specific), 0.0)))

    port, bench = aggregates["portfolio"], aggregates["benchmark"]
    bench_variance = _cov(bench, bench, cov)
    emit(BETA, "portfolio", "", "", _cov(port, bench, cov) / bench_variance if bench_variance > _RISK_TOL else 0.0)
    port_risk, bench_risk = _risk(port, cov), _risk(bench, cov)
    emit(
        CORRELATION,
        "portfolio",
        "",
        "",
        _cov(port, bench, cov) / (port_risk * bench_risk)
        if port_risk > _RISK_TOL and bench_risk > _RISK_TOL
        else 0.0,
    )
    emit(TRACKING_ERROR, "active", "", "", _risk(aggregates["active"], cov))

    # -- coverage ------------------------------------------------------------------
    emit(COVERAGE_WEIGHT, "portfolio", "", "covered", sum(p.weight for p in covered))
    emit(COVERAGE_WEIGHT, "portfolio", "", "uncovered", sum(p.weight for p in case.positions if not p.covered))
    emit(COVERAGE_WEIGHT, "benchmark", "", "covered", sum(p.benchmark_weight for p in covered))
    emit(
        COVERAGE_WEIGHT,
        "benchmark",
        "",
        "uncovered",
        sum(p.benchmark_weight for p in case.positions if not p.covered),
    )
    emit(COVERAGE_COUNT, "portfolio", "", "positions", float(len(case.positions)))
    emit(COVERAGE_COUNT, "portfolio", "", "covered", float(len(covered)))
    emit(COVERAGE_COUNT, "portfolio", "", "held_not_in_benchmark", float(sum(1 for p in case.positions if p.weight != 0.0 and p.benchmark_weight == 0.0)))
    emit(COVERAGE_COUNT, "portfolio", "", "in_benchmark_not_held", float(sum(1 for p in case.positions if p.weight == 0.0 and p.benchmark_weight != 0.0)))

    # -- position contributions ----------------------------------------------------
    for level in LEVELS:
        aggregate_v = aggregates[level]
        risk = _risk(aggregate_v, cov)
        for p in case.positions:
            code = p.security_code
            emit(WEIGHT, level, code, "", weight_of(p, level))
            if not p.covered or risk <= _RISK_TOL:
                for metric in (CTR_TOTAL, CTR_MARGINAL, CTR_PERCENT, CTR_COMMON_FACTOR, CTR_SPECIFIC):
                    emit(metric, level, code, "", 0.0)
                continue
            v = contribution_vector(p, level)
            cf = _quad(v.exposures, cov, aggregate_v.exposures) / risk
            sp = _dot_specific(v.specific, aggregate_v.specific) / risk
            emit(CTR_COMMON_FACTOR, level, code, "", cf)
            emit(CTR_SPECIFIC, level, code, "", sp)
            emit(CTR_TOTAL, level, code, "", cf + sp)
            emit(CTR_PERCENT, level, code, "", (cf + sp) / risk)
            emit(CTR_MARGINAL, level, code, "", _cov(own_of(p, level), aggregate_v, cov) / risk)

    # -- factor decomposition ------------------------------------------------------
    names = case.factor_names
    for level in LEVELS:
        v = aggregates[level]
        total = _risk(v, cov)
        specific_variance = _dot_specific(v.specific, v.specific)
        # F x, needed for the single-factor marginal.
        fx = {
            a: sum(cov[a][b] * v.exposures.get(b, 0.0) for b in sorted(v.exposures))
            for a in basis_ids
        }
        for f in basis_ids:
            name = names.get(f, str(f))
            x_f = v.exposures.get(f, 0.0)
            emit(ROW_EXPOSURE, level, name, "factor", x_f)
            if total <= _RISK_TOL:
                for metric in (ROW_STANDALONE, ROW_MARGINAL, ROW_CONTRIBUTION, ROW_RISK_PROPORTION, ROW_LITTERMAN_PROPORTION):
                    emit(metric, level, name, "factor", 0.0)
                continue
            standalone = math.sqrt(max(x_f * cov[f][f] * x_f, 0.0))
            contribution = x_f * fx[f] / total
            emit(ROW_STANDALONE, level, name, "factor", standalone)
            emit(ROW_MARGINAL, level, name, "factor", fx[f] / total)
            emit(ROW_CONTRIBUTION, level, name, "factor", contribution)
            emit(ROW_RISK_PROPORTION, level, name, "factor", (standalone / total) ** 2)
            emit(ROW_LITTERMAN_PROPORTION, level, name, "factor", contribution / total)
        # The specific row completes the partition, which is what makes the rows sum.
        if total <= _RISK_TOL:
            for metric in (ROW_STANDALONE, ROW_CONTRIBUTION, ROW_RISK_PROPORTION, ROW_LITTERMAN_PROPORTION):
                emit(metric, level, SPECIFIC_ROW_NAME, "factor", 0.0)
        else:
            specific_risk = math.sqrt(max(specific_variance, 0.0))
            specific_contribution = specific_variance / total
            emit(ROW_STANDALONE, level, SPECIFIC_ROW_NAME, "factor", specific_risk)
            emit(ROW_CONTRIBUTION, level, SPECIFIC_ROW_NAME, "factor", specific_contribution)
            emit(ROW_RISK_PROPORTION, level, SPECIFIC_ROW_NAME, "factor", (specific_risk / total) ** 2)
            emit(ROW_LITTERMAN_PROPORTION, level, SPECIFIC_ROW_NAME, "factor", specific_contribution / total)

    # -- tail ----------------------------------------------------------------------
    normal = NormalDist()
    for tail in case.tails:
        z = normal.inv_cdf(tail.confidence)
        horizon = math.sqrt(tail.horizon_days / TRADING_DAYS)
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        detail = f"c={tail.confidence!r},h={tail.horizon_days}"
        for level in LEVELS:
            risk = _risk(aggregates[level], cov)
            emit(VALUE_AT_RISK, level, "", detail, risk * z * horizon)
            emit(EXPECTED_SHORTFALL, level, "", detail, risk * density / (1.0 - tail.confidence) * horizon)

    return tuple(out)
