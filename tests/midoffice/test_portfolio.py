"""Tests for positions, portfolios and the risk triple at both levels."""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.risk import metrics
from praxis.midoffice.risk.basis import FactorCovariance
from praxis.midoffice.risk.exposures import ExposureVector, SpecificRisk
from praxis.midoffice.risk.portfolio import (
    Portfolio,
    Position,
    portfolio_risk,
    position_risk,
)

# Two factors, two securities, numbers that close by hand.
#
#   F = [[0.04, 0.01],        A: factor 1, specific 0.1, w 0.6, bench 0.4
#        [0.01, 0.09]]        B: factor 2, specific 0.2, w 0.4, bench 0.6
#
#   portfolio x = [0.6, 0.4]   x'Fx = 0.0144 + 0.0048 + 0.0144 = 0.0336
#                 specific {A: .06, B: .08} -> 0.0036 + 0.0064 = 0.0100
#                 total variance 0.0436
#   benchmark x = [0.4, 0.6]   x'Fx = 0.0064 + 0.0048 + 0.0324 = 0.0436
#                 specific {A: .04, B: .12} -> 0.0016 + 0.0144 = 0.0160
#                 total variance 0.0596
#   active    x = [0.2, -0.2]  x'Fx = 0.0016 - 0.0008 + 0.0036 = 0.0044
#                 specific {A: .02, B: -.04} -> 0.0004 + 0.0016 = 0.0020
#                 total variance 0.0064 -> tracking error exactly 0.08
COV = FactorCovariance((1, 2), np.array([[0.04, 0.01], [0.01, 0.09]]))


def book() -> Portfolio:
    return Portfolio.of(
        "Fund",
        [
            Position(
                "A",
                ExposureVector({1: 1.0}, SpecificRisk.for_security("A", 0.1)),
                weight=0.6,
                benchmark_weight=0.4,
            ),
            Position(
                "B",
                ExposureVector({2: 1.0}, SpecificRisk.for_security("B", 0.2)),
                weight=0.4,
                benchmark_weight=0.6,
            ),
        ],
        benchmark_name="Index",
    )


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------


def test_position_weights_and_flags():
    p = Position("A", ExposureVector({1: 2.0}), weight=0.3, benchmark_weight=0.1)
    assert p.active_weight == pytest.approx(0.2)
    assert p.is_held and p.is_in_benchmark
    assert p.weighted().exposures == {1: pytest.approx(0.6)}
    assert p.benchmark_weighted().exposures == {1: pytest.approx(0.2)}
    assert p.active_weighted().exposures == {1: pytest.approx(0.4)}


def test_benchmark_name_not_held_is_a_position_not_an_absence():
    p = Position("X", ExposureVector({1: 1.0}), weight=0.0, benchmark_weight=0.05)
    assert not p.is_held and p.is_in_benchmark
    assert p.active_weight == pytest.approx(-0.05)


def test_position_risk_is_the_securitys_standalone_risk_unweighted():
    """RAT's position-level totalRisk uses rd.expVect -- the security, not the holding."""
    basis = COV.basis_for([1, 2])
    heavy = Position("A", ExposureVector({1: 1.0}, SpecificRisk.for_security("A", 0.1)), weight=0.9)
    light = Position("A2", ExposureVector({1: 1.0}, SpecificRisk.for_security("A2", 0.1)), weight=0.01)
    assert position_risk(heavy, basis).total_risk == pytest.approx(
        position_risk(light, basis).total_risk
    )
    # sqrt(0.04 + 0.01)
    assert position_risk(heavy, basis).total_risk == pytest.approx(np.sqrt(0.05))


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_portfolio_exposure_is_the_weighted_sum():
    p = book()
    assert p.exposure.exposures == {1: pytest.approx(0.6), 2: pytest.approx(0.4)}
    assert p.benchmark_exposure.exposures == {1: pytest.approx(0.4), 2: pytest.approx(0.6)}
    assert p.exposure.specific.by_issuer["A"] == pytest.approx(0.06)
    assert p.exposure.specific.by_issuer["B"] == pytest.approx(0.08)


def test_active_exposure_from_active_weights_equals_the_difference_of_sums():
    """Algebraically identical; built at position level where both sides are one security."""
    p = book()
    direct = p.active_exposure
    differenced = p.exposure - p.benchmark_exposure
    assert direct.exposures.keys() == differenced.exposures.keys()
    for factor_id, value in direct.exposures.items():
        assert value == pytest.approx(differenced.exposures[factor_id])
    for issuer, value in direct.specific.by_issuer.items():
        assert value == pytest.approx(differenced.specific.by_issuer[issuer])


def test_duplicate_security_raises():
    with pytest.raises(ValueError, match="duplicate security"):
        Portfolio.of("F", [Position("A"), Position("A")])


def test_uncovered_positions_contribute_no_exposure_but_are_counted():
    p = Portfolio.of(
        "F",
        [
            Position("A", ExposureVector({1: 1.0}), weight=0.7),
            Position("CASHLIKE", ExposureVector({1: 5.0}), weight=0.3, covered=False),
        ],
    )
    assert p.exposure.exposures == {1: pytest.approx(0.7)}
    cov = p.coverage()
    assert cov.n_uncovered == 1
    assert cov.uncovered_weight == pytest.approx(0.3)
    assert any("not covered" in q for q in cov.qualifications())


# --------------------------------------------------------------------------
# Coverage accounting
# --------------------------------------------------------------------------


def test_coverage_counts_both_asymmetries():
    p = Portfolio.of(
        "F",
        [
            Position("HELD_AND_BENCH", weight=0.5, benchmark_weight=0.5),
            Position("HELD_ONLY", weight=0.5, benchmark_weight=0.0),
            Position("BENCH_ONLY", weight=0.0, benchmark_weight=0.5),
        ],
    )
    c = p.coverage()
    assert c.n_positions == 3
    assert c.n_held == 2
    assert c.n_in_benchmark == 2
    assert c.n_out_of_benchmark == 1
    assert c.n_benchmark_not_held == 1
    assert c.weight_sum == pytest.approx(1.0)
    assert c.benchmark_weight_sum == pytest.approx(1.0)


def test_weight_sum_is_reported_not_asserted():
    """A long/short book legitimately will not sum to one. A sum near two is double counting."""
    long_short = Portfolio.of(
        "LS", [Position("A", weight=1.3), Position("B", weight=-0.3)]
    )
    assert long_short.coverage().weight_sum == pytest.approx(1.0)

    doubled = Portfolio.of("D", [Position("A", weight=1.0), Position("B", weight=1.0)])
    assert doubled.coverage().weight_sum == pytest.approx(2.0)


# --------------------------------------------------------------------------
# The triple
# --------------------------------------------------------------------------


def test_hand_computed_triple():
    p = book()
    basis = p.basis(COV)
    r = portfolio_risk(p, basis)

    assert r.portfolio.common_factor_variance == pytest.approx(0.0336)
    assert r.portfolio.specific_variance == pytest.approx(0.0100)
    assert r.portfolio.total_variance == pytest.approx(0.0436)

    assert r.benchmark.common_factor_variance == pytest.approx(0.0436)
    assert r.benchmark.specific_variance == pytest.approx(0.0160)

    assert r.active.common_factor_variance == pytest.approx(0.0044)
    assert r.active.specific_variance == pytest.approx(0.0020)
    assert r.active.total_variance == pytest.approx(0.0064)
    assert r.tracking_error == pytest.approx(0.08)


def test_tracking_error_matches_the_three_term_identity():
    p = book()
    basis = p.basis(COV)
    r = portfolio_risk(p, basis)
    identity = np.sqrt(
        r.portfolio.total_variance
        + r.benchmark.total_variance
        - 2 * metrics.covariance(p.exposure, p.benchmark_exposure, basis)
    )
    assert r.tracking_error == pytest.approx(identity, rel=0, abs=1e-15)


def test_a_portfolio_matching_its_benchmark_has_no_tracking_error():
    p = Portfolio.of(
        "Index Fund",
        [
            Position("A", ExposureVector({1: 1.0}, SpecificRisk.for_security("A", 0.1)),
                     weight=0.5, benchmark_weight=0.5),
            Position("B", ExposureVector({2: 1.0}, SpecificRisk.for_security("B", 0.2)),
                     weight=0.5, benchmark_weight=0.5),
        ],
    )
    basis = p.basis(COV)
    r = portfolio_risk(p, basis)
    assert r.tracking_error == pytest.approx(0.0, abs=1e-15)
    assert r.beta == pytest.approx(1.0)
    assert r.correlation == pytest.approx(1.0)
    assert r.portfolio.total_risk == pytest.approx(r.benchmark.total_risk)


def test_basis_spans_both_sides_even_when_holdings_do_not_overlap():
    p = Portfolio.of(
        "F",
        [
            Position("A", ExposureVector({1: 1.0}), weight=1.0),
            Position("B", ExposureVector({2: 1.0}), benchmark_weight=1.0),
        ],
    )
    basis = p.basis(COV)
    assert basis.factor_ids == (1, 2)
    r = portfolio_risk(p, basis)
    assert r.active.common_factor_variance == pytest.approx(0.04 - 2 * 0.01 + 0.09)


# --------------------------------------------------------------------------
# Provenance reaches the top
# --------------------------------------------------------------------------


def test_coverage_and_repair_qualifications_reach_the_result():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(2, 2)) * 0.1
    m = a @ a.T
    w, V = np.linalg.eigh((m + m.T) / 2)
    w[0] = -1e-8
    cov = FactorCovariance((1, 2), V @ np.diag(w) @ V.T)

    p = Portfolio.of(
        "F",
        [
            # spans both factors itself, so the basis is 2x2 and the repair bites
            Position("A", ExposureVector({1: 1.0, 2: 0.5}), weight=0.8),
            Position("U", ExposureVector({2: 1.0}), weight=0.2, covered=False),
        ],
    )
    basis = p.basis(cov)
    assert len(basis) == 2
    r = portfolio_risk(p, basis)
    assert not r.is_exact
    notes = r.qualifications()
    assert any("repaired" in n for n in notes)
    assert any("not covered" in n for n in notes)


def test_exact_when_nothing_was_substituted():
    p = book()
    r = portfolio_risk(p, p.basis(COV))
    assert r.is_exact
    assert r.qualifications() == ()


def test_empty_portfolio_is_zero_risk_not_an_error():
    p = Portfolio.of("Empty", [])
    basis = p.basis(COV)
    r = portfolio_risk(p, basis)
    assert r.portfolio.total_risk == 0.0
    assert r.tracking_error == 0.0
    assert r.beta == 0.0
    assert r.coverage.n_positions == 0


# --------------------------------------------------------------------------
# A larger book, to check nothing degrades with scale
# --------------------------------------------------------------------------


def test_realistic_book_decomposes_exactly():
    rng = np.random.default_rng(13)
    n_factors = 40
    a = rng.normal(size=(n_factors, n_factors)) * 0.03
    m = a @ a.T / n_factors
    cov = FactorCovariance(tuple(range(n_factors)), (m + m.T) / 2)

    positions = []
    for i in range(150):
        r = np.random.default_rng(1000 + i)
        ids = r.choice(n_factors, size=6, replace=False)
        positions.append(
            Position(
                f"SEC{i:04d}",
                ExposureVector(
                    {int(f): float(v) for f, v in zip(ids, r.normal(size=6))},
                    SpecificRisk.for_security(f"SEC{i:04d}", abs(r.normal()) * 0.2),
                ),
                weight=1.0 / 150,
                benchmark_weight=1.0 / 150 if i % 3 else 0.0,
            )
        )
    p = Portfolio.of("Big", positions, benchmark_name="Bench")
    r = portfolio_risk(p, p.basis(cov))

    for level in (r.portfolio, r.benchmark, r.active):
        assert level.total_variance == pytest.approx(
            level.common_factor_variance + level.specific_variance, rel=0, abs=1e-15
        )
        assert level.total_risk >= 0.0
    assert r.tracking_error > 0.0
    assert r.coverage.n_positions == 150
    assert r.coverage.weight_sum == pytest.approx(1.0)


def test_an_uncovered_positions_factors_do_not_enter_the_basis():
    """Coverage is upstream of the basis: an unmodelled name brings no factors with it."""
    p = Portfolio.of(
        "F",
        [
            Position("A", ExposureVector({1: 1.0}), weight=0.8),
            Position("U", ExposureVector({2: 9.0}), weight=0.2, covered=False),
        ],
    )
    basis = p.basis(COV)
    assert basis.factor_ids == (1,)
    assert p.exposure.exposures == {1: pytest.approx(0.8)}
