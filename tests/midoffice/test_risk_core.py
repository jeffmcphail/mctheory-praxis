"""Tests for the factor risk core.

Two of these are regression pins against specific defects found while surveying the C#
originals, and are named so nobody quietly relaxes them:

  test_repair_does_not_amplify_small_negative_eigenvalues
      MCRLib computed `-lambda / 1e-5` in its PSD repair, flipping the sign correctly and
      then multiplying magnitude by 100,000.

  test_same_proxy_makes_factors_perfectly_correlated
      not a bug, but a surprising consequence of RAT's getCovarBlockExtraFactors that is
      much better met in a test than in production output.
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.risk.basis import (
    FactorBasis,
    FactorCovariance,
    active_basis,
    nearest_psd,
)
from praxis.midoffice.risk.exposures import ExposureVector, SpecificRisk, aggregate
from praxis.midoffice.risk import metrics


# --------------------------------------------------------------------------
# A case small enough to check by hand
# --------------------------------------------------------------------------

#   F = [[0.04, 0.01],      x = [1.0, 0.5]      specific volatility = 0.1
#        [0.01, 0.09]]
#
#   x'Fx = 1*0.04*1 + 1*0.01*0.5 + 0.5*0.01*1 + 0.5*0.09*0.5
#        = 0.04 + 0.005 + 0.005 + 0.0225 = 0.0725
#   specific variance = 0.01
#   total variance    = 0.0825
HAND_COV = FactorCovariance((10, 20), np.array([[0.04, 0.01], [0.01, 0.09]]))
HAND_VECTOR = ExposureVector(
    {10: 1.0, 20: 0.5}, SpecificRisk.for_security("AAA", 0.1)
)


def test_hand_computed_case():
    basis = HAND_COV.basis_for([10, 20])
    assert metrics.common_factor_variance(HAND_VECTOR, basis) == pytest.approx(0.0725)
    assert metrics.specific_variance(HAND_VECTOR) == pytest.approx(0.01)
    assert metrics.total_variance(HAND_VECTOR, basis) == pytest.approx(0.0825)
    assert metrics.total_risk(HAND_VECTOR, basis) == pytest.approx(np.sqrt(0.0825))


def test_variance_decomposition_is_exact():
    basis = HAND_COV.basis_for([10, 20])
    d = metrics.decompose(HAND_VECTOR, basis)
    assert d.total_variance == pytest.approx(
        d.common_factor_variance + d.specific_variance, rel=0, abs=1e-15
    )
    assert d.total_risk == pytest.approx(np.sqrt(d.total_variance))


# --------------------------------------------------------------------------
# The basis
# --------------------------------------------------------------------------


def test_active_basis_is_the_union_sorted_and_deduped():
    cov = FactorCovariance((1, 2, 3, 4), np.eye(4) * 0.01)
    portfolio = {3: 1.0, 1: 2.0}
    benchmark = {1: 0.5, 4: 1.0}
    basis = active_basis(cov, portfolio, benchmark)
    assert basis.factor_ids == (1, 3, 4)
    assert len(basis) == 3
    assert 2 not in basis


def test_dense_places_exposures_by_basis_order_and_zeroes_the_rest():
    cov = FactorCovariance((1, 2, 3), np.eye(3) * 0.01)
    basis = cov.basis_for([1, 2, 3])
    dense = basis.dense({3: 7.0, 1: 5.0})
    np.testing.assert_allclose(dense, [5.0, 0.0, 7.0])


def test_factor_outside_the_basis_raises_rather_than_being_dropped():
    """The silent-drop failure is the one this whole design exists to prevent."""
    cov = FactorCovariance((1, 2), np.eye(2) * 0.01)
    basis = cov.basis_for([1, 2])
    with pytest.raises(KeyError, match="not in this basis"):
        basis.dense({1: 1.0, 99: 1.0})


def test_slicing_a_subset_matches_the_full_matrix_entries():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(6, 6))
    full = a @ a.T
    cov = FactorCovariance(tuple(range(6)), full)
    basis = cov.basis_for([1, 4, 5])
    np.testing.assert_allclose(basis.covariance, full[np.ix_([1, 4, 5], [1, 4, 5])])


def test_basis_covariance_is_immutable():
    cov = FactorCovariance((1, 2), np.eye(2) * 0.01)
    basis = cov.basis_for([1, 2])
    with pytest.raises(ValueError):
        basis.covariance[0, 0] = 99.0


def test_duplicate_factor_ids_in_the_covariance_raise():
    with pytest.raises(ValueError, match="duplicate factor ids"):
        FactorCovariance((1, 1, 2), np.eye(3))


def test_asymmetric_covariance_raises_rather_than_being_silently_symmetrised():
    bad = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(ValueError, match="not symmetric"):
        nearest_psd(bad)


def test_empty_basis_gives_zero_risk_not_an_error():
    cov = FactorCovariance((), np.zeros((0, 0)))
    basis = cov.basis_for([])
    vector = ExposureVector({}, SpecificRisk.zero())
    assert len(basis) == 0
    assert metrics.total_risk(vector, basis) == 0.0


# --------------------------------------------------------------------------
# PSD repair
# --------------------------------------------------------------------------


def _matrix_with_eigenvalues(eigenvalues, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(len(eigenvalues), len(eigenvalues))))
    m = q @ np.diag(eigenvalues) @ q.T
    return (m + m.T) / 2.0


def test_psd_matrix_is_left_alone():
    m = _matrix_with_eigenvalues([1.0, 0.5, 0.25])
    repaired, record = nearest_psd(m)
    assert not record.repaired
    assert record.n_negative == 0
    np.testing.assert_allclose(repaired, m, atol=1e-12)


def test_negative_eigenvalue_is_repaired():
    m = _matrix_with_eigenvalues([1.0, 0.5, -1e-6])
    repaired, record = nearest_psd(m)
    assert record.repaired
    assert record.n_negative == 1
    assert record.most_negative == pytest.approx(-1e-6, rel=1e-6)
    assert np.linalg.eigvalsh(repaired).min() >= -1e-14


def test_repair_does_not_amplify_small_negative_eigenvalues():
    """Regression pin. MCRLib's repair did `-lambda / 1e-5`, inflating -1e-6 into +0.1.

    The repaired matrix must stay close to the original: a marginally non-PSD matrix is
    mostly right, and repair should nudge it rather than restructure it.
    """
    m = _matrix_with_eigenvalues([1.0, 0.5, -1e-6])
    repaired, _ = nearest_psd(m)

    distance = float(np.linalg.norm(repaired - m, ord="fro"))
    assert distance < 1e-5, f"repair moved the matrix by {distance:.3e}"

    eigenvalues = np.linalg.eigvalsh(repaired)
    assert eigenvalues.max() == pytest.approx(1.0, rel=1e-9), (
        "repair must not create a new largest eigenvalue"
    )


def test_relative_floor_makes_the_matrix_positive_definite():
    m = _matrix_with_eigenvalues([1.0, 0.5, -1e-6])
    repaired, record = nearest_psd(m, relative_floor=1e-8)
    assert np.linalg.eigvalsh(repaired).min() > 0.0
    assert record.floor_applied == pytest.approx(1e-8)
    np.linalg.inv(repaired)  # must not raise


def test_repair_record_reaches_the_decomposition():
    """Provenance has to arrive on the result, which is the point of the whole port."""
    m = _matrix_with_eigenvalues([0.04, 0.01, -1e-7])
    cov = FactorCovariance((1, 2, 3), m)
    basis = cov.basis_for([1, 2, 3])
    d = metrics.decompose(ExposureVector({1: 1.0}), basis)
    assert d.covariance_repair.repaired
    assert not d.is_exact
    assert any("repaired" in q for q in d.qualifications())


# --------------------------------------------------------------------------
# Proxied factors
# --------------------------------------------------------------------------


def test_absent_factor_uses_its_proxy_row():
    cov = FactorCovariance((1, 2), np.array([[0.04, 0.01], [0.01, 0.09]]))
    basis = cov.basis_for([1, 2, 77], proxies={77: 2})
    assert basis.factor_ids == (1, 2, 77)
    assert basis.proxied == {77: 2}
    # factor 77 inherits factor 2's variance and its covariance with 1
    assert basis.covariance[basis.index(77), basis.index(77)] == pytest.approx(0.09)
    assert basis.covariance[basis.index(1), basis.index(77)] == pytest.approx(0.01)


def test_same_proxy_makes_factors_perfectly_correlated():
    """Surprising but intended: two factors proxied to one target ARE that factor."""
    cov = FactorCovariance((1,), np.array([[0.04]]))
    basis = cov.basis_for([50, 51], proxies={50: 1, 51: 1})
    i, j = basis.index(50), basis.index(51)
    corr = basis.covariance[i, j] / np.sqrt(
        basis.covariance[i, i] * basis.covariance[j, j]
    )
    assert corr == pytest.approx(1.0)


def test_factor_with_no_proxy_raises():
    cov = FactorCovariance((1, 2), np.eye(2) * 0.01)
    with pytest.raises(KeyError, match="no\\s+proxy"):
        cov.basis_for([1, 2, 99])


# --------------------------------------------------------------------------
# Specific risk
# --------------------------------------------------------------------------


def test_same_issuer_correlates_and_different_issuers_do_not():
    a1 = SpecificRisk.for_security("AAA1", 0.1, issuer_key="ACME")
    a2 = SpecificRisk.for_security("AAA2", 0.1, issuer_key="ACME")
    b = SpecificRisk.for_security("BBB", 0.1, issuer_key="OTHER")
    assert a1.dot(a2) == pytest.approx(0.01)
    assert a1.dot(b) == pytest.approx(0.0)


def test_same_issuer_adds_linearly_before_squaring():
    """Perfect correlation within an issuer, independence across: 0.04 against 0.02."""
    same = SpecificRisk.for_security("A", 0.1, issuer_key="ACME") + (
        SpecificRisk.for_security("B", 0.1, issuer_key="ACME")
    )
    different = SpecificRisk.for_security("A", 0.1, issuer_key="ACME") + (
        SpecificRisk.for_security("B", 0.1, issuer_key="OTHER")
    )
    assert same.variance() == pytest.approx(0.04)
    assert different.variance() == pytest.approx(0.02)


def test_a_security_without_an_issuer_key_is_its_own_issuer():
    """No synthetic DEFAULT<n>, no shared counter, no race -- just the security code."""
    x = SpecificRisk.for_security("XYZ", 0.2)
    y = SpecificRisk.for_security("PQR", 0.2)
    assert x.by_issuer == {"XYZ": 0.2}
    assert x.dot(y) == pytest.approx(0.0)
    assert x.dot(SpecificRisk.for_security("XYZ", 0.2)) == pytest.approx(0.04)


def test_specific_risk_needs_something_to_key_by():
    with pytest.raises(ValueError, match="issuer key"):
        SpecificRisk.for_security("", 0.1)


def test_specific_risk_dot_is_symmetric():
    a = SpecificRisk({"X": 0.1, "Y": 0.2, "Z": 0.3})
    b = SpecificRisk({"Y": 0.5, "Z": 0.1})
    assert a.dot(b) == pytest.approx(b.dot(a))
    assert a.dot(b) == pytest.approx(0.2 * 0.5 + 0.3 * 0.1)


# --------------------------------------------------------------------------
# Vector algebra
# --------------------------------------------------------------------------


def test_exposure_vectors_add_and_scale():
    a = ExposureVector({1: 1.0, 2: 2.0}, SpecificRisk({"X": 0.1}))
    b = ExposureVector({2: 3.0, 3: 1.0}, SpecificRisk({"X": 0.2, "Y": 0.3}))
    total = a + b
    assert total.exposures == {1: 1.0, 2: 5.0, 3: 1.0}
    assert total.specific.by_issuer == {"X": pytest.approx(0.3), "Y": pytest.approx(0.3)}

    scaled = 2.0 * a
    assert scaled.exposures == {1: 2.0, 2: 4.0}
    assert scaled.specific.by_issuer["X"] == pytest.approx(0.2)


def test_subtraction_is_addition_of_the_negative():
    a = ExposureVector({1: 3.0}, SpecificRisk({"X": 0.3}))
    b = ExposureVector({1: 1.0}, SpecificRisk({"X": 0.1}))
    assert (a - b).exposures == {1: 2.0}
    assert (a - b).specific.by_issuer["X"] == pytest.approx(0.2)


def test_aggregate_sums_weighted_positions():
    positions = [
        0.5 * ExposureVector({1: 2.0}, SpecificRisk.for_security("A", 0.2)),
        0.5 * ExposureVector({1: 4.0}, SpecificRisk.for_security("B", 0.2)),
    ]
    portfolio = aggregate(positions)
    assert portfolio.exposures == {1: 3.0}
    assert portfolio.specific.variance() == pytest.approx(0.1 ** 2 + 0.1 ** 2)


# --------------------------------------------------------------------------
# Active risk on a shared basis
# --------------------------------------------------------------------------


def test_active_vector_lives_on_the_same_basis_as_its_parents():
    cov = FactorCovariance((1, 2, 3), np.diag([0.04, 0.09, 0.01]))
    portfolio = ExposureVector({1: 1.0, 2: 0.5})
    benchmark = ExposureVector({2: 1.0, 3: 2.0})
    basis = active_basis(cov, portfolio.exposures, benchmark.exposures)

    active = portfolio - benchmark
    assert basis.factor_ids == (1, 2, 3)
    assert active.exposures == {1: 1.0, 2: -0.5, 3: -2.0}

    # 1*0.04*1 + (-0.5)*0.09*(-0.5) + (-2)*0.01*(-2) = 0.04 + 0.0225 + 0.04
    assert metrics.common_factor_variance(active, basis) == pytest.approx(0.1025)


def test_beta_of_the_benchmark_against_itself_is_one():
    cov = FactorCovariance((1, 2), np.array([[0.04, 0.01], [0.01, 0.09]]))
    basis = cov.basis_for([1, 2])
    benchmark = ExposureVector({1: 1.0, 2: 1.0}, SpecificRisk.for_security("B", 0.05))
    assert metrics.beta(benchmark, benchmark, basis) == pytest.approx(1.0)
    assert metrics.correlation(benchmark, benchmark, basis) == pytest.approx(1.0)


def test_beta_against_a_riskless_benchmark_is_zero_not_a_division_error():
    cov = FactorCovariance((1,), np.array([[0.04]]))
    basis = cov.basis_for([1])
    assert metrics.beta(ExposureVector({1: 1.0}), ExposureVector({}), basis) == 0.0


def test_covariance_is_symmetric_in_its_arguments():
    cov = FactorCovariance((1, 2), np.array([[0.04, 0.01], [0.01, 0.09]]))
    basis = cov.basis_for([1, 2])
    a = ExposureVector({1: 1.0, 2: 0.5}, SpecificRisk({"X": 0.1}))
    b = ExposureVector({1: -0.3, 2: 2.0}, SpecificRisk({"X": 0.2}))
    assert metrics.covariance(a, b, basis) == pytest.approx(
        metrics.covariance(b, a, basis)
    )


def test_active_risk_matches_the_three_term_identity():
    """sigma_a^2 == sigma_p^2 + sigma_b^2 - 2 cov(p, b).

    Computing active risk from the differenced vector and from the three separate
    quantities must agree exactly.  A basis mismatch between portfolio and benchmark is
    precisely what would break this, which is why it is worth pinning on a non-trivial case
    rather than a two-factor fixture.
    """
    rng = np.random.default_rng(3)
    n = 25
    a = rng.normal(size=(n, n)) * 0.02
    full = a @ a.T / n
    cov = FactorCovariance(tuple(range(n)), (full + full.T) / 2)

    def book(seed, count):
        r = np.random.default_rng(seed)
        return aggregate(
            [
                (1.0 / count)
                * ExposureVector(
                    {
                        int(f): float(v)
                        for f, v in zip(
                            r.choice(n, 6, replace=False), r.normal(size=6)
                        )
                    },
                    SpecificRisk.for_security(f"S{i:03d}", abs(r.normal()) * 0.2),
                )
                for i in range(count)
            ]
        )

    portfolio, benchmark = book(11, 40), book(22, 90)
    basis = active_basis(cov, portfolio.exposures, benchmark.exposures)

    direct = metrics.total_risk(portfolio - benchmark, basis)
    identity = np.sqrt(
        metrics.total_variance(portfolio, basis)
        + metrics.total_variance(benchmark, basis)
        - 2 * metrics.covariance(portfolio, benchmark, basis)
    )
    assert direct == pytest.approx(identity, rel=0, abs=1e-15)
