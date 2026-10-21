"""Tests for VaR and expected shortfall."""

from __future__ import annotations

import math

import numpy as np
import pytest

from praxis.midoffice.risk.basis import FactorCovariance
from praxis.midoffice.risk.contributions import Level, contributions
from praxis.midoffice.risk.exposures import ExposureVector, SpecificRisk
from praxis.midoffice.risk.portfolio import Portfolio, Position, portfolio_risk
from praxis.midoffice.risk.tail import (
    ACKLAM_RELATIVE_ERROR,
    TailSpec,
    expected_shortfall,
    inverse_normal_cdf,
    normal_cdf,
    normal_pdf,
    scale,
    tail_risk,
    value_at_risk,
)

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
    )


# --------------------------------------------------------------------------
# The inverse normal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,expected",
    [(0.95, 1.6448536269514722), (0.99, 2.3263478740408408), (0.5, 0.0),
     (0.975, 1.959963984540054), (0.01, -2.3263478740408408)],
)
def test_inverse_normal_matches_known_quantiles(p, expected):
    assert inverse_normal_cdf(p) == pytest.approx(expected, rel=ACKLAM_RELATIVE_ERROR * 10)


def test_unrefined_accuracy_is_within_the_documented_bound():
    """Acklam without refinement, as RAT uses it."""
    worst = 0.0
    for p in np.linspace(0.001, 0.999, 2001):
        x = inverse_normal_cdf(float(p))
        worst = max(worst, abs(normal_cdf(x) - p) / p)
    assert worst < ACKLAM_RELATIVE_ERROR * 10


def test_refinement_is_materially_more_accurate():
    for p in (0.01, 0.5, 0.95, 0.99, 0.999):
        plain = abs(normal_cdf(inverse_normal_cdf(p)) - p)
        refined = abs(normal_cdf(inverse_normal_cdf(p, refine=True)) - p)
        assert refined <= plain + 1e-18


def test_inverse_normal_is_antisymmetric():
    for p in (0.01, 0.1, 0.3, 0.45):
        assert inverse_normal_cdf(p) == pytest.approx(-inverse_normal_cdf(1 - p), rel=1e-9)


def test_degenerate_probabilities():
    assert inverse_normal_cdf(0.0) == -math.inf
    assert inverse_normal_cdf(1.0) == math.inf
    with pytest.raises(ValueError, match="must be in"):
        inverse_normal_cdf(1.5)


# --------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------


def test_var_multiplier_matches_rats_formula():
    """RAT: sqrt(horizon / 252) * inverse_normal(confidence)."""
    spec = TailSpec(confidence=0.95, horizon_days=10.0)
    assert spec.var_multiplier == pytest.approx(
        math.sqrt(10.0 / 252.0) * inverse_normal_cdf(0.95)
    )


def test_expected_shortfall_multiplier_matches_rats_formula():
    """RAT: exp(-q^2/2) * sqrt(h/252) / ((1 - c) * sqrt(2 pi))."""
    spec = TailSpec(confidence=0.99, horizon_days=1.0)
    q = inverse_normal_cdf(0.99)
    expected = math.exp(-(q ** 2) / 2) * math.sqrt(1.0 / 252.0) / (
        (1 - 0.99) * math.sqrt(2 * math.pi)
    )
    assert spec.expected_shortfall_multiplier == pytest.approx(expected)
    assert spec.expected_shortfall_multiplier == pytest.approx(
        spec.horizon_scale * normal_pdf(q) / (1 - 0.99)
    )


def test_horizon_scales_as_square_root_of_time():
    one = TailSpec(horizon_days=1.0).var_multiplier
    four = TailSpec(horizon_days=4.0).var_multiplier
    assert four == pytest.approx(2.0 * one)


def test_invalid_specs_are_refused():
    with pytest.raises(ValueError, match="strictly between"):
        TailSpec(confidence=1.0)
    with pytest.raises(ValueError, match="horizon must be positive"):
        TailSpec(horizon_days=0.0)
    with pytest.raises(ValueError, match="trading days"):
        TailSpec(trading_days=-1.0)


# --------------------------------------------------------------------------
# The measures
# --------------------------------------------------------------------------


def test_expected_shortfall_always_exceeds_var():
    for confidence in (0.90, 0.95, 0.99, 0.999):
        spec = TailSpec(confidence=confidence)
        t = tail_risk(0.2, spec)
        assert t.expected_shortfall > t.value_at_risk > 0.0


def test_shortfall_ratio_depends_only_on_confidence():
    spec = TailSpec(confidence=0.95)
    assert tail_risk(0.05, spec).shortfall_ratio == pytest.approx(
        tail_risk(5.0, spec).shortfall_ratio
    )


def test_both_measures_are_linear_in_risk():
    spec = TailSpec()
    assert value_at_risk(0.3, spec) == pytest.approx(3 * value_at_risk(0.1, spec))
    assert expected_shortfall(0.3, spec) == pytest.approx(
        3 * expected_shortfall(0.1, spec)
    )


def test_zero_risk_gives_zero_tail():
    t = tail_risk(0.0)
    assert t.value_at_risk == 0.0 and t.expected_shortfall == 0.0
    assert t.shortfall_ratio == 0.0


def test_the_spec_travels_with_the_result():
    t = tail_risk(0.1, TailSpec(confidence=0.99, horizon_days=10))
    assert t.spec.confidence == 0.99
    assert "99.0%" in t.spec.describe() and "10 day" in t.spec.describe()


# --------------------------------------------------------------------------
# Decomposition comes free
# --------------------------------------------------------------------------


def test_scaled_contributions_sum_to_var():
    """Contributions sum to risk, so scaled contributions sum to scaled risk.

    This is the whole of VaR decomposition. Writing a separate calculation would only create
    an opportunity for it to disagree with the risk decomposition it must match.
    """
    p = book()
    basis = p.basis(COV)
    spec = TailSpec(confidence=0.95, horizon_days=1.0)

    risk = portfolio_risk(p, basis).portfolio.total_risk
    cs = contributions(p, basis, Level.PORTFOLIO)
    scaled = scale((c.total for c in cs), spec.var_multiplier)

    assert sum(scaled) == pytest.approx(value_at_risk(risk, spec), abs=1e-15)


def test_scaled_contributions_sum_to_expected_shortfall():
    p = book()
    basis = p.basis(COV)
    spec = TailSpec(confidence=0.99)
    risk = portfolio_risk(p, basis).active.total_risk
    cs = contributions(p, basis, Level.ACTIVE)
    scaled = scale((c.total for c in cs), spec.expected_shortfall_multiplier)
    assert sum(scaled) == pytest.approx(expected_shortfall(risk, spec), abs=1e-15)


def test_tail_numbers_at_all_three_levels():
    p = book()
    basis = p.basis(COV)
    spec = TailSpec(confidence=0.95, horizon_days=1.0)
    r = portfolio_risk(p, basis)
    for level in (r.portfolio, r.benchmark, r.active):
        t = tail_risk(level.total_risk, spec)
        assert t.value_at_risk == pytest.approx(level.total_risk * spec.var_multiplier)
        assert t.expected_shortfall > t.value_at_risk


def test_hand_computed_var():
    """Portfolio total risk is sqrt(0.0436); one-day 95% VaR is that times z/sqrt(252)."""
    p = book()
    basis = p.basis(COV)
    spec = TailSpec(confidence=0.95, horizon_days=1.0)
    risk = math.sqrt(0.0436)
    expected = risk * inverse_normal_cdf(0.95) / math.sqrt(252.0)
    assert value_at_risk(
        portfolio_risk(p, basis).portfolio.total_risk, spec
    ) == pytest.approx(expected)
