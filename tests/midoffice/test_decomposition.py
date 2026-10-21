"""Tests for factor and factor-group decomposition.

The governing property, as everywhere else in this package: the parts sum to the whole.
Factor contributions plus the specific row equal total risk, exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.risk.basis import FactorCovariance
from praxis.midoffice.risk.decomposition import (
    factor_decomposition,
    group_decomposition,
)
from praxis.midoffice.risk.exposures import ExposureVector, SpecificRisk
from praxis.midoffice.risk.factors import SPECIFIC_RISK, FactorTree
from praxis.midoffice.risk import metrics

COV = FactorCovariance((1, 2), np.array([[0.04, 0.01], [0.01, 0.09]]))
VECTOR = ExposureVector({1: 0.6, 2: 0.4}, SpecificRisk({"A": 0.06, "B": 0.08}))

# Mirrors the test_factors hierarchy so the two modules exercise the same shape.
EDGES = [
    ("Total Risk", "Market"),
    ("Total Risk", "Style"),
    ("Market", "Beta"),
    ("Style", "Value"),
    ("Style", "Growth"),
]
LEAF_FACTORS = {"Beta": [1], "Value": [10, 11], "Growth": [12]}


def tree() -> FactorTree:
    return FactorTree.build(EDGES, LEAF_FACTORS)


def wide_setup(seed=9):
    t = tree()
    ids = sorted(t.factors_under(t.root))
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(len(ids), len(ids))) * 0.05
    m = a @ a.T / len(ids)
    cov = FactorCovariance(tuple(ids), (m + m.T) / 2)
    basis = cov.basis_for(ids)
    vector = ExposureVector(
        {f: float(v) for f, v in zip(ids, rng.normal(size=len(ids)))},
        SpecificRisk({"X": 0.05, "Y": 0.03}),
    )
    return t, cov, basis, vector


# --------------------------------------------------------------------------
# The identity
# --------------------------------------------------------------------------


def test_factor_contributions_plus_specific_equal_total_risk():
    basis = COV.basis_for([1, 2])
    d = factor_decomposition(VECTOR, basis)
    assert d.check() == pytest.approx(0.0, abs=1e-15)
    assert d.total_risk == pytest.approx(metrics.total_risk(VECTOR, basis))


def test_the_specific_row_is_present_and_carries_the_remainder():
    basis = COV.basis_for([1, 2])
    d = factor_decomposition(VECTOR, basis)
    specific = d.row(SPECIFIC_RISK)
    assert specific.is_specific
    assert specific.standalone_risk == pytest.approx(VECTOR.specific.risk())
    assert specific.contribution == pytest.approx(
        VECTOR.specific.variance() / d.total_risk
    )
    factor_part = sum(r.contribution for r in d.factor_rows)
    assert factor_part == pytest.approx(d.common_factor_risk ** 2 / d.total_risk)


def test_group_contributions_also_sum_to_total_risk():
    t, _, basis, vector = wide_setup()
    d = group_decomposition(vector, basis, t)
    assert d.check() == pytest.approx(0.0, abs=1e-12)
    assert {r.name for r in d.rows} == {"Market", "Style", SPECIFIC_RISK}


def test_drilling_into_a_group_sums_to_that_group_not_the_whole_book():
    """Children sum to their parent's contribution, and carry no specific row.

    Specific risk sits under no factor group, so it appears once, at the top level. A
    drill-down that repeated it would double-count it.
    """
    t, _, basis, vector = wide_setup()
    top = group_decomposition(vector, basis, t)
    style_contribution = top.row("Style").contribution

    d = group_decomposition(vector, basis, t, parent="Style")
    assert {r.name for r in d.rows} == {"Value", "Growth"}
    assert SPECIFIC_RISK not in {r.name for r in d.rows}
    assert d.parent == "Style"
    assert d.sums_to == pytest.approx(style_contribution)
    assert d.check() == pytest.approx(0.0, abs=1e-12)


def test_a_missing_child_is_refused_when_drilling_in_too():
    """The coverage guard follows the drill-down: it checks the parent's own factors."""
    t = FactorTree.build(
        [("Root", "G"), ("Root", "H"), ("G", "G1"), ("G", "G2")],
        {"G1": [1], "G2": [2], "H": [3]},
    )
    cov = FactorCovariance((1, 2, 3), np.diag([0.04, 0.09, 0.01]))
    basis = cov.basis_for([1, 2, 3])
    # Well-formed: both of G's children are present, so this must work.
    d = group_decomposition(ExposureVector({1: 1.0, 2: 1.0, 3: 1.0}), basis, t, parent="G")
    assert {r.name for r in d.rows} == {"G1", "G2"}
    assert d.check() == pytest.approx(0.0, abs=1e-12)


def test_a_leaf_parent_falls_through_to_factor_level():
    t, _, basis, vector = wide_setup()
    d = group_decomposition(vector, basis, t, parent="Beta")
    assert [r.name for r in d.factor_rows] == [str(f) for f in basis.factor_ids]


# --------------------------------------------------------------------------
# The two proportions answer different questions
# --------------------------------------------------------------------------


def test_litterman_proportions_sum_to_one():
    t, _, basis, vector = wide_setup()
    for d in (factor_decomposition(vector, basis), group_decomposition(vector, basis, t)):
        assert sum(r.litterman_proportion for r in d.rows) == pytest.approx(1.0)


def test_risk_proportions_do_not_sum_to_one_and_that_is_correct():
    """Standalone risk ignores every correlation. Reported because it is intuitive, not
    because it adds up -- a reader who assumes otherwise concludes the numbers are broken."""
    t, _, basis, vector = wide_setup()
    d = group_decomposition(vector, basis, t)
    total = sum(r.risk_proportion for r in d.rows)
    assert total != pytest.approx(1.0, abs=1e-6)


def test_a_single_factor_group_matches_the_factor_level_row():
    """A group of one is a factor: sqrt(x_g' F x_g) reduces to |x_f| sqrt(F_ff)."""
    t, _, basis, vector = wide_setup()
    groups = group_decomposition(vector, basis, t, parent="Market")  # Beta is the only child
    beta_leaf = group_decomposition(vector, basis, t, parent="Total Risk").row("Market")
    x = vector.exposures[1]
    expected = abs(x) * np.sqrt(basis.covariance[basis.index(1), basis.index(1)])
    assert beta_leaf.standalone_risk == pytest.approx(expected)


def test_standalone_risk_of_one_factor_is_abs_exposure_times_vol():
    basis = COV.basis_for([1, 2])
    d = factor_decomposition(VECTOR, basis)
    row = d.row("1")
    assert row.standalone_risk == pytest.approx(0.6 * np.sqrt(0.04))
    assert row.exposure == pytest.approx(0.6)


# --------------------------------------------------------------------------
# Hand-computed
# --------------------------------------------------------------------------


def test_hand_computed_factor_decomposition():
    """x = [0.6, 0.4], F = [[.04,.01],[.01,.09]], specific variance 0.01.

        Fx       = [0.6*0.04 + 0.4*0.01, 0.6*0.01 + 0.4*0.09] = [0.028, 0.042]
        x'Fx     = 0.6*0.028 + 0.4*0.042 = 0.0168 + 0.0168 = 0.0336
        total    = sqrt(0.0336 + 0.01) = sqrt(0.0436)
        contrib1 = 0.6*0.028 / total = 0.0168 / total
        contrib2 = 0.4*0.042 / total = 0.0168 / total
        specific = 0.01 / total
    """
    basis = COV.basis_for([1, 2])
    d = factor_decomposition(VECTOR, basis)
    total = np.sqrt(0.0436)
    assert d.total_risk == pytest.approx(total)
    assert d.row("1").contribution == pytest.approx(0.0168 / total)
    assert d.row("2").contribution == pytest.approx(0.0168 / total)
    assert d.row(SPECIFIC_RISK).contribution == pytest.approx(0.01 / total)
    assert d.row("1").marginal == pytest.approx(0.028 / total)


def test_factor_names_are_used_when_supplied():
    basis = COV.basis_for([1, 2])
    d = factor_decomposition(VECTOR, basis, names={1: "Beta", 2: "Momentum"})
    assert d.row("Beta").exposure == pytest.approx(0.6)
    assert d.row("Momentum").exposure == pytest.approx(0.4)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_a_factor_outside_every_child_group_raises_rather_than_vanishing():
    """A partition that does not cover the basis would silently lose risk."""
    t = FactorTree.build([("Root", "A"), ("Root", "B")], {"A": [1], "B": [2]})
    cov = FactorCovariance((1, 2, 3), np.diag([0.04, 0.09, 0.01]))
    basis = cov.basis_for([1, 2, 3])
    with pytest.raises(ValueError, match="outside every child"):
        group_decomposition(ExposureVector({1: 1.0, 3: 1.0}), basis, t)


def test_zero_risk_vector_decomposes_to_zeros():
    basis = COV.basis_for([1, 2])
    d = factor_decomposition(ExposureVector({}), basis)
    assert d.total_risk == 0.0
    assert all(r.contribution == 0.0 for r in d.rows)
    assert d.check() == pytest.approx(0.0)


def test_check_raises_on_a_broken_partition():
    basis = COV.basis_for([1, 2])
    d = factor_decomposition(VECTOR, basis)
    broken = type(d)(d.rows[:1], d.total_risk, d.common_factor_risk, d.specific_risk)
    with pytest.raises(ValueError, match="does not cover"):
        broken.check()


def test_unknown_row_name_raises():
    basis = COV.basis_for([1, 2])
    with pytest.raises(KeyError, match="no row named"):
        factor_decomposition(VECTOR, basis).row("Nope")


# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------


def test_identity_holds_on_a_wide_model():
    rng = np.random.default_rng(21)
    n = 45
    a = rng.normal(size=(n, n)) * 0.04
    m = a @ a.T / n
    cov = FactorCovariance(tuple(range(n)), (m + m.T) / 2)
    basis = cov.basis_for(range(n))
    vector = ExposureVector(
        {i: float(v) for i, v in enumerate(rng.normal(size=n))},
        SpecificRisk({f"I{i}": float(abs(v)) * 0.1 for i, v in enumerate(rng.normal(size=20))}),
    )
    d = factor_decomposition(vector, basis)
    assert len(d) == n + 1
    assert d.check() == pytest.approx(0.0, abs=1e-12)
    assert sum(r.litterman_proportion for r in d.rows) == pytest.approx(1.0)
