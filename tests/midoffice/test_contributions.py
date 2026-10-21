"""Tests for marginal and component contributions.

The governing property is that component contributions sum to the risk they decompose. It
holds by construction here -- sum(v_i) == X, so sum(cov(v_i, X))/sigma == cov(X,X)/sigma ==
sigma -- so these tests mostly check that the construction has not been broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.risk.basis import FactorCovariance
from praxis.midoffice.risk.contributions import (
    RAT_MARGINAL_SCALE,
    Contribution,
    Level,
    check_sum,
    contributions,
)
from praxis.midoffice.risk.exposures import ExposureVector, SpecificRisk
from praxis.midoffice.risk.portfolio import Portfolio, Position, portfolio_risk

COV = FactorCovariance((1, 2), np.array([[0.04, 0.01], [0.01, 0.09]]))


def book() -> Portfolio:
    return Portfolio.of(
        "Fund",
        [
            Position("A", ExposureVector({1: 1.0}, SpecificRisk.for_security("A", 0.1)),
                     weight=0.6, benchmark_weight=0.4),
            Position("B", ExposureVector({2: 1.0}, SpecificRisk.for_security("B", 0.2)),
                     weight=0.4, benchmark_weight=0.6),
        ],
        benchmark_name="Index",
    )


def big_book(n=120, n_factors=30, seed=4):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_factors, n_factors)) * 0.04
    m = a @ a.T / n_factors
    cov = FactorCovariance(tuple(range(n_factors)), (m + m.T) / 2)
    positions = []
    for i in range(n):
        r = np.random.default_rng(seed * 1000 + i)
        ids = r.choice(n_factors, size=5, replace=False)
        positions.append(
            Position(
                f"S{i:04d}",
                ExposureVector(
                    {int(f): float(v) for f, v in zip(ids, r.normal(size=5))},
                    SpecificRisk.for_security(f"S{i:04d}", abs(r.normal()) * 0.2),
                ),
                weight=1.0 / n,
                benchmark_weight=(1.5 / n) if i % 2 else (0.5 / n),
            )
        )
    p = Portfolio.of("Big", positions, benchmark_name="B")
    return p, cov, p.basis(cov)


# --------------------------------------------------------------------------
# The governing identity, at all three levels
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level,attr",
    [
        (Level.PORTFOLIO, "portfolio"),
        (Level.BENCHMARK, "benchmark"),
        (Level.ACTIVE, "active"),
    ],
)
def test_contributions_sum_to_the_risk_they_decompose(level, attr):
    p = book()
    basis = p.basis(COV)
    risk = getattr(portfolio_risk(p, basis), attr).total_risk
    cs = contributions(p, basis, level)
    assert check_sum(cs, risk) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("level", list(Level))
def test_identity_holds_on_a_large_book(level):
    p, _, basis = big_book()
    risk = {
        Level.PORTFOLIO: "portfolio",
        Level.BENCHMARK: "benchmark",
        Level.ACTIVE: "active",
    }[level]
    target = getattr(portfolio_risk(p, basis), risk).total_risk
    cs = contributions(p, basis, level)
    assert len(cs) == len(p)
    assert check_sum(cs, target) == pytest.approx(0.0, abs=1e-10)


def test_check_sum_raises_when_the_aggregate_does_not_match():
    p = book()
    basis = p.basis(COV)
    cs = contributions(p, basis, Level.PORTFOLIO)
    with pytest.raises(ValueError, match="different aggregates"):
        check_sum(cs, 999.0)


# --------------------------------------------------------------------------
# Structure of a single contribution
# --------------------------------------------------------------------------


def test_component_is_weight_times_marginal_when_both_sides_share_an_exposure():
    p = book()
    basis = p.basis(COV)
    for c in contributions(p, basis, Level.PORTFOLIO):
        assert c.total == pytest.approx(c.weight * c.marginal)


def test_factor_and_specific_parts_sum_to_the_component():
    p = book()
    basis = p.basis(COV)
    for c in contributions(p, basis, Level.PORTFOLIO):
        assert c.total == pytest.approx(c.common_factor + c.specific)


def test_percentages_sum_to_one():
    p, _, basis = big_book()
    cs = contributions(p, basis, Level.ACTIVE)
    assert sum(c.percent for c in cs) == pytest.approx(1.0)


def test_hand_computed_contribution():
    """Portfolio x = [0.6, 0.4], sigma^2 = 0.0436.

    A's contribution vector is 0.6*{1:1} with specific 0.06.
        cov(v_A, X) = 0.6*0.04*0.6 + 0.6*0.01*0.4 + 0.06*0.06
                    = 0.0144 + 0.0024 + 0.0036 = 0.0204
    B's is 0.4*{2:1} with specific 0.08.
        cov(v_B, X) = 0.4*0.01*0.6 + 0.4*0.09*0.4 + 0.08*0.08
                    = 0.0024 + 0.0144 + 0.0064 = 0.0232
    and 0.0204 + 0.0232 = 0.0436, the variance, as it must.
    """
    p = book()
    basis = p.basis(COV)
    sigma = np.sqrt(0.0436)
    cs = {c.security_code: c for c in contributions(p, basis, Level.PORTFOLIO)}
    assert cs["A"].total == pytest.approx(0.0204 / sigma)
    assert cs["B"].total == pytest.approx(0.0232 / sigma)
    assert cs["A"].total + cs["B"].total == pytest.approx(sigma)


# --------------------------------------------------------------------------
# Weights per level
# --------------------------------------------------------------------------


def test_each_level_uses_the_weight_of_the_side_it_decomposes():
    p = book()
    basis = p.basis(COV)
    by_level = {
        level: {c.security_code: c for c in contributions(p, basis, level)}
        for level in Level
    }
    assert by_level[Level.PORTFOLIO]["A"].weight == pytest.approx(0.6)
    assert by_level[Level.BENCHMARK]["A"].weight == pytest.approx(0.4)
    assert by_level[Level.ACTIVE]["A"].weight == pytest.approx(0.2)


def test_a_deviation_contributes_positively_whichever_way_it_runs():
    """An underweight adds tracking error just as an overweight does.

    B is underweight (active weight -0.2) and its marginal is also negative (-0.30), because
    increasing B toward neutral would reduce active risk. The product is positive. Two
    positions whose contributions sum to a positive tracking error cannot have one of them
    negative, which is the arithmetic saying the same thing.
    """
    p = book()
    basis = p.basis(COV)
    cs = {c.security_code: c for c in contributions(p, basis, Level.ACTIVE)}
    assert cs["A"].weight > 0 and cs["B"].weight < 0
    assert cs["B"].marginal < 0, "reducing the underweight would reduce active risk"
    assert cs["A"].total > 0 and cs["B"].total > 0
    assert cs["A"].total + cs["B"].total == pytest.approx(
        portfolio_risk(p, basis).active.total_risk
    )


def test_a_contribution_turns_negative_only_when_a_position_hedges_the_rest():
    """Possible, and worth pinning so nobody 'fixes' it: a position can reduce active risk.

    Here A is overweight a factor the rest of the book is short, so moving A further
    overweight brings the active exposure closer to neutral.
    """
    p = Portfolio.of(
        "Hedged",
        [
            Position("A", ExposureVector({1: 1.0}), weight=0.30, benchmark_weight=0.25),
            Position("C", ExposureVector({1: -1.0}), weight=0.70, benchmark_weight=0.25),
        ],
    )
    basis = p.basis(COV)
    cs = {c.security_code: c for c in contributions(p, basis, Level.ACTIVE)}
    assert min(c.total for c in cs.values()) < 0.0
    assert check_sum(
        tuple(cs.values()), portfolio_risk(p, basis).active.total_risk
    ) == pytest.approx(0.0, abs=1e-14)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_uncovered_positions_contribute_zero_and_stay_in_the_list():
    p = Portfolio.of(
        "F",
        [
            Position("A", ExposureVector({1: 1.0}, SpecificRisk.for_security("A", 0.1)), weight=0.8),
            Position("U", ExposureVector({2: 5.0}), weight=0.2, covered=False),
        ],
    )
    basis = p.basis(COV)
    cs = contributions(p, basis, Level.PORTFOLIO)
    assert [c.security_code for c in cs] == ["A", "U"]
    assert cs[1].total == 0.0 and cs[1].marginal == 0.0


def test_zero_risk_portfolio_gives_zero_contributions_not_a_division_error():
    p = Portfolio.of("Flat", [Position("A", ExposureVector({}), weight=1.0)])
    basis = p.basis(COV)
    cs = contributions(p, basis, Level.PORTFOLIO)
    assert all(c.total == 0.0 for c in cs)


def test_an_index_fund_has_zero_active_contributions():
    p = Portfolio.of(
        "IF",
        [
            Position("A", ExposureVector({1: 1.0}, SpecificRisk.for_security("A", 0.1)),
                     weight=0.5, benchmark_weight=0.5),
            Position("B", ExposureVector({2: 1.0}, SpecificRisk.for_security("B", 0.2)),
                     weight=0.5, benchmark_weight=0.5),
        ],
    )
    basis = p.basis(COV)
    assert all(c.total == 0.0 for c in contributions(p, basis, Level.ACTIVE))


# --------------------------------------------------------------------------
# The bucketed case, where the two sides carry different vectors
# --------------------------------------------------------------------------


def test_component_stays_exact_when_the_two_sides_carry_different_exposures():
    """RAT needs a BucketSecurity branch for this; here v_i is just what it is.

    `total` must still be exact -- the contributions must still sum to active risk -- even
    though `weight * marginal` no longer describes it.
    """
    p = Portfolio.of(
        "F",
        [
            Position(
                "BUCKET",
                ExposureVector({1: 1.0}, SpecificRisk.for_security("BUCKET", 0.1)),
                weight=0.5,
                benchmark_weight=0.5,
                benchmark_exposure_override=ExposureVector(
                    {2: 1.0}, SpecificRisk.for_security("BUCKET", 0.1)
                ),
            ),
            Position("B", ExposureVector({2: 1.0}, SpecificRisk.for_security("B", 0.2)),
                     weight=0.5, benchmark_weight=0.5),
        ],
    )
    basis = p.basis(COV)
    active_risk = portfolio_risk(p, basis).active.total_risk
    assert active_risk > 0.0, "differing exposures must produce active risk at equal weights"

    cs = contributions(p, basis, Level.ACTIVE)
    assert check_sum(cs, active_risk) == pytest.approx(0.0, abs=1e-14)

    bucket = next(c for c in cs if c.security_code == "BUCKET")
    assert bucket.weight == pytest.approx(0.0)
    assert bucket.total != pytest.approx(bucket.weight * bucket.marginal)


# --------------------------------------------------------------------------
# The RAT scale factor
# --------------------------------------------------------------------------


def test_rat_marginal_scale_is_documented_and_applied_consistently():
    """RAT divides marginal by 100 and multiplies it back in the component."""
    p = book()
    basis = p.basis(COV)
    for c in contributions(p, basis, Level.PORTFOLIO):
        rat_marginal = c.marginal * RAT_MARGINAL_SCALE
        assert 100 * rat_marginal * c.weight == pytest.approx(c.total)
