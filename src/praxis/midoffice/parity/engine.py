"""Drive the real engine over a parity case and flatten everything it produces.

This is the other half of the comparison.  It emits the same `Record` shape as the
reference, so the comparator holds two interchangeable sets and cannot be accidentally
written to favour one side.

Two structural zeros are deliberately not emitted: the specific-risk row's `exposure` and
`marginal`.  The engine sets both to 0.0 because neither is defined for specific risk -- it
sits under no factor and has no exposure to be marginal with respect to -- so emitting them
would compare two hardcoded constants and report a pass that means nothing.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..risk.basis import FactorCovariance, active_basis
from ..risk.contributions import Level, contributions
from ..risk.decomposition import factor_decomposition
from ..risk.exposures import ExposureVector, SpecificRisk
from ..risk.portfolio import Portfolio, Position, WeightBasis, portfolio_risk
from ..risk.tail import TailSpec, tail_risk
from .case import ParityCase, PositionSpec
from .record import (
    BETA,
    COMMON_FACTOR_RISK,
    CORRELATION,
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

_LEVELS = (
    ("portfolio", Level.PORTFOLIO),
    ("benchmark", Level.BENCHMARK),
    ("active", Level.ACTIVE),
)


def _exposure_vector(
    exposures: Mapping[int, float], p: PositionSpec
) -> ExposureVector:
    specific = (
        SpecificRisk({p.issuer_key or p.security_code: p.specific_volatility})
        if p.specific_volatility
        else SpecificRisk({})
    )
    return ExposureVector(dict(exposures), specific)


def build_portfolio(case: ParityCase) -> Portfolio:
    """Turn a case into the engine's own objects."""
    positions = []
    for p in case.positions:
        override = (
            _exposure_vector(p.benchmark_exposures, p)
            if p.benchmark_exposures is not None
            else None
        )
        positions.append(
            Position(
                security_code=p.security_code,
                exposure=_exposure_vector(p.exposures, p),
                weight=p.weight,
                benchmark_weight=p.benchmark_weight,
                covered=p.covered,
                benchmark_exposure_override=override,
            )
        )
    return Portfolio.of(
        name=case.portfolio_name,
        positions=positions,
        benchmark_name=case.benchmark_name,
        weight_basis=WeightBasis(case.weight_basis),
    )


def build_covariance(case: ParityCase) -> FactorCovariance:
    return FactorCovariance(
        factor_ids=case.factor_ids,
        matrix=np.array(case.covariance, dtype=float) if case.factors else np.zeros((0, 0)),
    )


def engine_records(case: ParityCase) -> tuple[Record, ...]:
    """Every number the engine produces for this case, flattened."""
    portfolio = build_portfolio(case)
    covariance = build_covariance(case)
    basis = active_basis(
        covariance,
        portfolio.exposure.exposures,
        portfolio.benchmark_exposure.exposures,
        proxies=case.proxies or None,
        relative_floor=case.relative_floor,
    )

    result = portfolio_risk(portfolio, basis)
    out: list[Record] = []

    def emit(metric: str, level: str, entity: str, detail: str, value: float) -> None:
        out.append(Record(metric, level, entity, detail, float(value)))

    # -- book level ----------------------------------------------------------------
    for label, decomposition in (
        ("portfolio", result.portfolio),
        ("benchmark", result.benchmark),
        ("active", result.active),
    ):
        emit(TOTAL_RISK, label, "", "", decomposition.total_risk)
        emit(COMMON_FACTOR_RISK, label, "", "", decomposition.common_factor_risk)
        emit(SPECIFIC_RISK, label, "", "", decomposition.specific_risk)

    emit(BETA, "portfolio", "", "", result.beta)
    emit(CORRELATION, "portfolio", "", "", result.correlation)
    emit(TRACKING_ERROR, "active", "", "", result.tracking_error)

    # -- coverage ------------------------------------------------------------------
    coverage = result.coverage
    emit(COVERAGE_WEIGHT, "portfolio", "", "covered", coverage.covered_weight_sum)
    emit(COVERAGE_WEIGHT, "portfolio", "", "uncovered", coverage.uncovered_weight)
    emit(COVERAGE_WEIGHT, "benchmark", "", "covered", coverage.covered_benchmark_weight_sum)
    emit(COVERAGE_COUNT, "portfolio", "", "positions", float(coverage.n_positions))
    emit(COVERAGE_COUNT, "portfolio", "", "covered", float(coverage.n_positions - coverage.n_uncovered))
    emit(COVERAGE_COUNT, "portfolio", "", "held_not_in_benchmark", float(coverage.n_out_of_benchmark))
    emit(COVERAGE_COUNT, "portfolio", "", "in_benchmark_not_held", float(coverage.n_benchmark_not_held))

    # -- position contributions ----------------------------------------------------
    for label, level in _LEVELS:
        for c in contributions(portfolio, basis, level):
            emit(WEIGHT, label, c.security_code, "", c.weight)
            emit(CTR_TOTAL, label, c.security_code, "", c.total)
            emit(CTR_MARGINAL, label, c.security_code, "", c.marginal)
            emit(CTR_PERCENT, label, c.security_code, "", c.percent)
            emit(CTR_COMMON_FACTOR, label, c.security_code, "", c.common_factor)
            emit(CTR_SPECIFIC, label, c.security_code, "", c.specific)

    # -- factor decomposition ------------------------------------------------------
    names = case.factor_names
    for label, vector in (
        ("portfolio", portfolio.exposure),
        ("benchmark", portfolio.benchmark_exposure),
        ("active", portfolio.active_exposure),
    ):
        for row in factor_decomposition(vector, basis, names=names):
            if not row.is_specific:
                emit(ROW_EXPOSURE, label, row.name, "factor", row.exposure)
                emit(ROW_MARGINAL, label, row.name, "factor", row.marginal)
            emit(ROW_STANDALONE, label, row.name, "factor", row.standalone_risk)
            emit(ROW_CONTRIBUTION, label, row.name, "factor", row.contribution)
            emit(ROW_RISK_PROPORTION, label, row.name, "factor", row.risk_proportion)
            emit(ROW_LITTERMAN_PROPORTION, label, row.name, "factor", row.litterman_proportion)

    # -- tail ----------------------------------------------------------------------
    for tail in case.tails:
        spec = TailSpec(
            confidence=tail.confidence,
            horizon_days=float(tail.horizon_days),
            refine_inverse_normal=tail.refine,
        )
        detail = f"c={tail.confidence!r},h={tail.horizon_days}"
        for label, decomposition in (
            ("portfolio", result.portfolio),
            ("benchmark", result.benchmark),
            ("active", result.active),
        ):
            t = tail_risk(decomposition.total_risk, spec)
            emit(VALUE_AT_RISK, label, "", detail, t.value_at_risk)
            emit(EXPECTED_SHORTFALL, label, "", detail, t.expected_shortfall)

    return tuple(out)
