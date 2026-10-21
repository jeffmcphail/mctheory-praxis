"""Generate parity cases with known structure.

Agreement on one hand-built case is weak evidence: a shared misreading shows up on every
case, but a formula that is wrong only for, say, a benchmark-only holding or a negative
active weight will pass a case that happens not to contain one.  So the generator produces
the awkward shapes on purpose -- names held but not in the benchmark, names in the benchmark
but not held, uncovered names carrying real exposure, negative active weights, and issuers
spanning several securities.

The covariance is built as `B'B` plus a diagonal, which is PSD by construction.  That is a
requirement rather than a convenience: the reference compares against the full matrix while
the engine slices and would repair a non-PSD slice, so a non-PSD fixture would manufacture
disagreements that are an artifact of the fixture rather than a finding about the engine.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np

from .case import FactorSpec, ParityCase, PositionSpec, TailCase


def synthetic_case(
    *,
    seed: int,
    n_factors: int = 6,
    n_positions: int = 25,
    n_uncovered: int = 2,
    n_benchmark_only: int = 3,
    n_portfolio_only: int = 3,
    n_groups: int = 2,
    n_shared_issuers: int = 2,
    confidences: Sequence[float] = (0.95, 0.99),
    refine: bool = False,
    weight_basis: str = "full",
    exposure_density: float = 0.6,
) -> ParityCase:
    """A case with the awkward shapes present by construction."""
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    # PSD covariance. Scaled so volatilities land in a plausible range for annual equity
    # factor returns rather than being numerically exotic.
    b = nprng.normal(size=(n_factors, n_factors)) * 0.08
    matrix = b.T @ b + np.diag(nprng.uniform(0.001, 0.01, size=n_factors))
    matrix = (matrix + matrix.T) / 2.0  # exact symmetry, so validate() is satisfied

    groups = [f"G{i + 1}" for i in range(max(1, n_groups))]
    factors = tuple(
        FactorSpec(
            factor_id=100 + i,
            name=f"Factor {i + 1}",
            model="EQ_MODEL",
            group=groups[i % len(groups)],
        )
        for i in range(n_factors)
    )
    factor_ids = [f.factor_id for f in factors]

    issuers = [f"ISS{i + 1}" for i in range(max(1, n_shared_issuers))]

    positions: list[PositionSpec] = []
    for i in range(n_positions):
        loaded = [f for f in factor_ids if rng.random() < exposure_density]
        if not loaded:  # every covered name must touch at least one factor
            loaded = [rng.choice(factor_ids)]
        exposures = {f: round(rng.uniform(-1.5, 1.5), 6) for f in loaded}

        # The first few names share issuers, so specific risk is not diagonal and the
        # issuer-keyed inner product is actually exercised.
        issuer = issuers[i % len(issuers)] if i < 2 * len(issuers) else ""

        positions.append(
            PositionSpec(
                security_code=f"SEC{i + 1:03d}",
                weight=round(rng.uniform(0.001, 0.05), 8),
                benchmark_weight=round(rng.uniform(0.001, 0.05), 8),
                exposures=exposures,
                specific_volatility=round(rng.uniform(0.05, 0.45), 6),
                issuer_key=issuer,
                covered=True,
            )
        )

    def retag(start: int, count: int, **changes: object) -> None:
        for j in range(start, min(start + count, len(positions))):
            positions[j] = PositionSpec(
                security_code=positions[j].security_code,
                weight=float(changes.get("weight", positions[j].weight)),
                benchmark_weight=float(
                    changes.get("benchmark_weight", positions[j].benchmark_weight)
                ),
                exposures=positions[j].exposures,
                specific_volatility=positions[j].specific_volatility,
                issuer_key=positions[j].issuer_key,
                covered=bool(changes.get("covered", positions[j].covered)),
            )

    cursor = 0
    retag(cursor, n_uncovered, covered=False)
    cursor += n_uncovered
    retag(cursor, n_benchmark_only, weight=0.0)
    cursor += n_benchmark_only
    retag(cursor, n_portfolio_only, benchmark_weight=0.0)

    return ParityCase(
        name=f"synthetic-{seed}",
        factors=factors,
        covariance=tuple(tuple(float(x) for x in row) for row in matrix),
        positions=tuple(positions),
        weight_basis=weight_basis,
        tails=tuple(TailCase(confidence=c, refine=refine) for c in confidences),
        groups={g: "Total Risk" for g in groups},
        source=f"synthetic:synthetic_case:seed={seed}",
    )


def shapes_present(case: ParityCase) -> dict[str, int]:
    """Count the awkward shapes, so a test can assert the generator really made them."""
    return {
        "uncovered": sum(1 for p in case.positions if not p.covered),
        "benchmark_only": sum(1 for p in case.positions if p.weight == 0.0 and p.benchmark_weight != 0.0),
        "portfolio_only": sum(1 for p in case.positions if p.weight != 0.0 and p.benchmark_weight == 0.0),
        "negative_active": sum(1 for p in case.positions if p.weight - p.benchmark_weight < 0.0),
        "shared_issuer": len([p for p in case.positions if p.issuer_key]),
        "uncovered_with_exposure": sum(1 for p in case.positions if not p.covered and p.exposures),
    }
