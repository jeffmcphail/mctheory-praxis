"""The eigenfactor risk adjustment (Menchero, Wang and Orr; USE4): the bias by simulation, and
the adjustment that undoes it.

Checked against worlds whose covariance is known: where the history is long the bias is one;
where it is short the classic pattern shows -- the largest eigenvalues estimated too large, the
smallest too small -- and the adjustment, its bias simulated from the estimate alone, brings the
estimate's eigenfactors back toward their true risk.
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.model.bias import eigenportfolios
from praxis.midoffice.model.covariance import (
    ExponentialCovariance,
    HorizonCovariance,
    SampleCovariance,
)
from praxis.midoffice.model.eigen import EigenBias, eigen_bias


def _truth(k: int = 10, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    values = np.geomspace(4.0, 0.05, k)
    q, _ = np.linalg.qr(a)
    return (q * values) @ q.T


def test_the_bias_is_one_on_a_long_history_and_the_classic_pattern_on_a_short_one():
    truth = _truth()
    long = eigen_bias(truth, SampleCovariance(), 5000, draws=200)
    assert np.all(np.abs(long.bias - 1.0) < 0.03)
    short = eigen_bias(truth, SampleCovariance(), 30, draws=400)
    # The largest eigenvalue estimated too large, the smallest far too small.
    assert short.bias[0] < 1.0
    assert short.bias[-1] > 1.3
    assert np.all(np.diff(short.bias[-4:]) > -0.05)
    assert short.n_periods == 30 and short.draws == 400 and short.degenerate == 0
    assert np.all(short.standard_error < 0.02)


def test_the_simulation_is_seeded_and_its_error_is_its_spread():
    truth = _truth()
    one = eigen_bias(truth, ExponentialCovariance(20.0), 60, draws=300)
    again = eigen_bias(truth, ExponentialCovariance(20.0), 60, draws=300)
    assert np.array_equal(one.bias, again.bias)
    assert np.array_equal(one.standard_error, again.standard_error)
    other = eigen_bias(truth, ExponentialCovariance(20.0), 60, draws=300, seed=99)
    assert not np.array_equal(one.bias, other.bias)
    # Two seeds differ by about their standard errors, not by more.
    gap = np.abs(one.bias - other.bias) / np.hypot(one.standard_error, other.standard_error)
    assert np.all(gap < 4.5)


def test_the_adjusted_covariance_scales_each_eigenvalue_and_keeps_its_eigenvectors():
    truth = _truth(k=6)
    bias = eigen_bias(truth, SampleCovariance(), 25, draws=200)
    values, vectors = eigenportfolios(truth)
    assert np.allclose(bias.values, values) and np.allclose(np.abs(bias.vectors), np.abs(vectors))
    assert bias.adjusted(0.0) is bias.covariance
    for a in (1.0, 2.5):
        gamma = a * (bias.bias - 1.0) + 1.0
        assert np.allclose(bias.scales(a), gamma)
        adjusted = bias.adjusted(a)
        assert np.allclose(adjusted, adjusted.T)
        # Each eigenvector of the day's covariance is one of the adjusted covariance's, its
        # eigenvalue scaled by gamma squared.
        for k in range(6):
            u = bias.vectors[:, k]
            assert np.allclose(adjusted @ u, gamma[k] ** 2 * bias.values[k] * u, atol=1e-12)
        scaled, kept = bias.basis(a)
        assert np.allclose(scaled, gamma**2 * bias.values) and kept is bias.vectors


def test_the_adjustment_brings_the_estimates_eigenfactors_back_toward_their_true_risk():
    """Menchero, Wang and Orr's own check, in miniature: estimate a covariance from a short
    history of a known one; its eigenfactors' true variance over their estimated one averages
    well above one at the small eigenvalues. Simulated from the estimate alone, the adjustment
    at a = 1 takes most of that away."""
    truth = _truth(k=8, seed=11)
    estimator = ExponentialCovariance(30.0)
    rng = np.random.default_rng(3)
    root = np.linalg.cholesky(truth)
    raw, adjusted = [], []
    for _ in range(60):
        estimate = estimator.estimate(rng.standard_normal((60, 8)) @ root.T).matrix
        bias = eigen_bias(estimate, estimator, 60, draws=150)
        true = np.einsum("ik,ij,jk->k", bias.vectors, truth, bias.vectors)
        raw.append(true / bias.values)
        adjusted.append(true / bias.basis(1.0)[0])
    raw, adjusted = np.mean(raw, axis=0), np.mean(adjusted, axis=0)
    assert raw[-3:].mean() > 1.4
    assert abs(adjusted[-3:].mean() - 1.0) < 0.1
    assert np.all(np.abs(adjusted - 1.0) < 0.1)


def test_the_simulation_re_estimates_with_the_estimator_it_is_given():
    """The two half-lives and the lags are the estimator's, so the simulation's: a split
    estimator's bias differs from a single half-life's on the same covariance."""
    truth = _truth(k=8)
    split = eigen_bias(truth, HorizonCovariance(20.0, 120.0, newey_west_lags=0), 150, draws=200)
    single = eigen_bias(truth, ExponentialCovariance(20.0), 150, draws=200)
    assert not np.allclose(split.bias, single.bias, atol=0.01)
    assert isinstance(split, EigenBias)


def test_bad_inputs_are_refused():
    truth = _truth(k=4)
    with pytest.raises(ValueError, match="square"):
        eigen_bias(truth[:3], SampleCovariance(), 50)
    with pytest.raises(ValueError, match="two draws"):
        eigen_bias(truth, SampleCovariance(), 50, draws=1)
    with pytest.raises(ValueError, match="two periods"):
        eigen_bias(truth, SampleCovariance(), 1)
