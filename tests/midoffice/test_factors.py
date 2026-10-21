"""Tests for factor identity and the factor-group hierarchy.

The integration test at the end is the reason this module exists: `factors_under(group)` fed
to `FactorCovariance.basis_for` is how factor-group decomposition slices the covariance.
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.risk.basis import FactorCovariance
from praxis.midoffice.risk.exposures import ExposureVector
from praxis.midoffice.risk.factors import (
    AmbiguousFactor,
    Factor,
    FactorList,
    FactorTree,
)
from praxis.midoffice.risk import metrics


# A small but realistically shaped equity model hierarchy.
#
#   Total Risk
#     Industry
#       Financials [22, 99]
#       Tech       [20, 21, 99]      <- 99 deliberately under two leaves
#     Market
#       Beta       [1]
#     Style
#       Growth     [12]
#       Momentum   [13]
#       Value      [10, 11]
EDGES = [
    ("Total Risk", "Market"),
    ("Total Risk", "Style"),
    ("Total Risk", "Industry"),
    ("Market", "Beta"),
    ("Style", "Value"),
    ("Style", "Growth"),
    ("Style", "Momentum"),
    ("Industry", "Tech"),
    ("Industry", "Financials"),
]
LEAF_FACTORS = {
    "Beta": [1],
    "Value": [10, 11],
    "Growth": [12],
    "Momentum": [13],
    "Tech": [20, 21, 99],
    "Financials": [22, 99],
}


def tree() -> FactorTree:
    return FactorTree.build(EDGES, LEAF_FACTORS)


# --------------------------------------------------------------------------
# FactorList
# --------------------------------------------------------------------------


def test_factor_lookup_both_ways():
    fl = FactorList.of([(1, "Beta", "EQ"), (10, "Value", "EQ")])
    assert fl.by_id(10).name == "Value"
    assert fl.id_for("Beta") == 1
    assert "Value" in fl and "Nope" not in fl
    assert fl.ids == (1, 10)


def test_duplicate_factor_id_raises():
    with pytest.raises(ValueError, match="duplicate factor id"):
        FactorList(( Factor(1, "A"), Factor(1, "B") ))


def test_unknown_factor_raises_on_both_lookups():
    fl = FactorList.of([(1, "Beta", "EQ")])
    with pytest.raises(KeyError, match="no factor with id 7"):
        fl.by_id(7)
    with pytest.raises(KeyError, match="no factor named"):
        fl.id_for("Missing")


def test_name_shared_across_models_is_ambiguous_rather_than_arbitrary():
    """RAT's single-argument getFactorId returns whichever it finds first."""
    fl = FactorList.of([(1, "Duration", "EQ"), (2, "Duration", "FI")])
    with pytest.raises(AmbiguousFactor, match="2 models"):
        fl.id_for("Duration")
    assert fl.id_for("Duration", model="FI") == 2
    assert fl.id_for("Duration", model="EQ") == 1


def test_model_scoped_lookup_reports_where_the_name_does_exist():
    fl = FactorList.of([(1, "Duration", "FI")])
    with pytest.raises(KeyError, match="present in"):
        fl.id_for("Duration", model="EQ")


def test_models_and_ids_per_model():
    fl = FactorList.of([(1, "A", "EQ"), (2, "B", "EQ"), (3, "C", "FI")])
    assert fl.models == ("EQ", "FI")
    assert fl.ids_for_model("EQ") == frozenset({1, 2})
    assert fl.ids_for_model("FI") == frozenset({3})


# --------------------------------------------------------------------------
# Tree construction and rollup
# --------------------------------------------------------------------------


def test_rollup_reaches_the_root():
    t = tree()
    assert t.root == "Total Risk"
    assert t.factors_under("Total Risk") == frozenset(
        {1, 10, 11, 12, 13, 20, 21, 22, 99}
    )
    assert t.factors_under("Style") == frozenset({10, 11, 12, 13})
    assert t.factors_under("Beta") == frozenset({1})


def test_a_factor_under_two_leaves_appears_once_in_the_rollup():
    """RAT counts multiplicity here; nothing reads the count, so this is a set."""
    t = tree()
    assert t.factors_under("Industry") == frozenset({20, 21, 22, 99})
    assert len(t.factors_under("Industry")) == 4


def test_own_factors_are_distinct_from_rolled_up_factors():
    t = tree()
    assert t.node("Style").own_factors == frozenset()
    assert t.node("Style").factors == frozenset({10, 11, 12, 13})
    assert t.node("Value").own_factors == t.node("Value").factors


def test_structure_flags_and_depth():
    t = tree()
    assert t.node("Total Risk").is_root and not t.node("Total Risk").is_leaf
    assert t.node("Value").is_leaf and not t.node("Value").is_root
    assert t.node("Total Risk").depth == 0
    assert t.node("Style").depth == 1
    assert t.node("Value").depth == 2
    assert t.leaves == ("Beta", "Financials", "Growth", "Momentum", "Tech", "Value")


def test_navigation():
    t = tree()
    assert t.children_of("Style") == ("Growth", "Momentum", "Value")
    assert t.descendants("Market") == ("Beta",)
    assert t.ancestors("Value") == ("Style", "Total Risk")
    assert t.ancestors("Total Risk") == ()
    assert len(t) == 10


def test_single_node_tree_is_legal():
    t = FactorTree.build([], {"Everything": [1, 2]})
    assert t.root == "Everything"
    assert t.node("Everything").is_root and t.node("Everything").is_leaf
    assert t.factors_under("Everything") == frozenset({1, 2})


def test_render_is_indented_by_depth():
    out = tree().render()
    assert out.splitlines()[0].startswith("Total Risk")
    assert any(line.startswith("    Value") for line in out.splitlines())


def test_render_can_name_factors():
    fl = FactorList.of([(1, "Beta"), (10, "BookToPrice"), (11, "EarnYield"),
                        (12, "Growth"), (13, "Momentum"), (20, "Semis"),
                        (21, "Software"), (22, "Banks"), (99, "Shared")])
    out = tree().render(start="Style", factor_list=fl)
    assert "BookToPrice" in out


# --------------------------------------------------------------------------
# Malformed trees are refused, not logged
# --------------------------------------------------------------------------


def test_leaf_with_no_factors_raises():
    """RAT logs this and continues, leaving every group above it incomplete."""
    edges = [("Root", "A"), ("Root", "B")]
    with pytest.raises(ValueError, match="no factors"):
        FactorTree.build(edges, {"A": [1]})


def test_factors_on_a_branch_node_raise():
    edges = [("Root", "A")]
    with pytest.raises(ValueError, match="only be attached to leaf"):
        FactorTree.build(edges, {"A": [1], "Root": [2]})


def test_two_parents_raise():
    edges = [("Root", "A"), ("Root", "B"), ("A", "Shared"), ("B", "Shared")]
    with pytest.raises(ValueError, match="two parents"):
        FactorTree.build(edges, {"Shared": [1]})


def test_cycle_is_rejected_because_it_has_no_root():
    """A pure cycle leaves every node with a parent, so there is no root to start from.

    A cycle reachable FROM a root would give some node two parents and be caught earlier.
    The walk keeps its own cycle guard as belt and braces, but these two checks are what
    actually reject malformed graphs.
    """
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    with pytest.raises(ValueError, match="exactly one root, found 0"):
        FactorTree.build(edges, {"C": [1]})


def test_cycle_hanging_off_a_root_is_caught_as_a_second_parent():
    edges = [("Root", "A"), ("A", "B"), ("B", "A")]
    with pytest.raises(ValueError, match="two parents"):
        FactorTree.build(edges, {"B": [1]})


def test_two_disconnected_roots_raise():
    edges = [("R1", "A"), ("R2", "B")]
    with pytest.raises(ValueError, match="exactly one root"):
        FactorTree.build(edges, {"A": [1], "B": [2]})


def test_named_root_with_orphans_raises():
    edges = [("R1", "A"), ("R2", "B")]
    with pytest.raises(ValueError, match="unreachable"):
        FactorTree.build(edges, {"A": [1], "B": [2]}, root="R1")


def test_unknown_node_lookup_raises():
    with pytest.raises(KeyError, match="no node named"):
        tree().node("Nope")


# --------------------------------------------------------------------------
# The integration that justifies the module
# --------------------------------------------------------------------------


def _covariance_over(factor_ids):
    rng = np.random.default_rng(5)
    n = len(factor_ids)
    a = rng.normal(size=(n, n)) * 0.05
    m = a @ a.T / n
    return FactorCovariance(tuple(factor_ids), (m + m.T) / 2)


def test_group_slice_reproduces_full_basis_risk_for_an_in_group_portfolio():
    """The invariant that makes group slicing valid.

    A portfolio exposed only to Style factors must get the same common-factor risk whether
    it is computed on the whole model's basis or on the Style sub-block. If that fails, the
    sub-block is not a sub-block.
    """
    t = tree()
    all_ids = sorted(t.factors_under(t.root))
    cov = _covariance_over(all_ids)

    style_only = ExposureVector({10: 1.5, 11: -0.5, 12: 0.75, 13: 0.25})

    full_basis = cov.basis_for(all_ids)
    style_basis = cov.basis_for(t.factors_under("Style"))

    assert style_basis.factor_ids == (10, 11, 12, 13)
    assert metrics.common_factor_risk(style_only, style_basis) == pytest.approx(
        metrics.common_factor_risk(style_only, full_basis), rel=0, abs=1e-15
    )


def test_every_group_in_the_tree_yields_a_usable_basis():
    t = tree()
    all_ids = sorted(t.factors_under(t.root))
    cov = _covariance_over(all_ids)
    for node in t.walk():
        basis = cov.basis_for(t.factors_under(node.name))
        assert len(basis) == len(node.factors)
        assert set(basis.factor_ids) == set(node.factors)
        # a unit exposure to every factor in the group must give non-negative variance
        vector = ExposureVector({f: 1.0 for f in node.factors})
        assert metrics.common_factor_variance(vector, basis) >= 0.0


def test_sibling_group_risks_do_not_sum_to_the_parent():
    """Not a defect -- cross-group covariance is real, and the test records the expectation."""
    t = tree()
    all_ids = sorted(t.factors_under(t.root))
    cov = _covariance_over(all_ids)
    exposures = {f: 1.0 for f in all_ids}

    parent_basis = cov.basis_for(t.factors_under("Total Risk"))
    parent = metrics.common_factor_variance(ExposureVector(exposures), parent_basis)

    children = 0.0
    for child in t.children_of("Total Risk"):
        ids = t.factors_under(child)
        basis = cov.basis_for(ids)
        children += metrics.common_factor_variance(
            ExposureVector({f: exposures[f] for f in ids}), basis
        )
    assert children != pytest.approx(parent)
