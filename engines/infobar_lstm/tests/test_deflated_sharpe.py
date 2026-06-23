"""
Tests for the Deflated Sharpe Ratio utility (Cycle 58b pre-registration).

The DSR is the Run-C gating metric; it must be correct BEFORE any model fit.
Run: pytest engines/infobar_lstm/tests/test_deflated_sharpe.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from engines.infobar_lstm.deflated_sharpe import (
    _norm_cdf, _norm_ppf, probabilistic_sharpe_ratio, expected_max_sharpe,
    deflated_sharpe_ratio, trial_sharpe_stats, effective_independent_trials,
)


def test_norm_cdf_known_values():
    assert abs(_norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(_norm_cdf(1.959963985) - 0.975) < 1e-6
    assert abs(_norm_cdf(-1.959963985) - 0.025) < 1e-6


def test_norm_ppf_known_values_and_roundtrip():
    assert abs(_norm_ppf(0.5)) < 1e-9
    assert abs(_norm_ppf(0.975) - 1.959963985) < 1e-6
    for p in (0.001, 0.02, 0.2, 0.5, 0.8, 0.98, 0.999):
        assert abs(_norm_cdf(_norm_ppf(p)) - p) < 1e-8


def test_norm_ppf_domain_guard():
    with pytest.raises(ValueError):
        _norm_ppf(0.0)
    with pytest.raises(ValueError):
        _norm_ppf(1.0)


def test_psr_benchmark_equal_is_half():
    # observed == benchmark -> exactly 50% probability of exceeding it.
    assert abs(probabilistic_sharpe_ratio(0.1, 0.1, 250) - 0.5) < 1e-9


def test_psr_monotonic_in_sr_and_n():
    base = probabilistic_sharpe_ratio(0.10, 0.0, 250)
    assert probabilistic_sharpe_ratio(0.20, 0.0, 250) > base       # higher SR
    assert probabilistic_sharpe_ratio(0.10, 0.0, 1000) > base      # more data
    assert 0.0 <= base <= 1.0


def test_psr_normal_matches_closed_form():
    # skew=0, kurt=3 -> denom = sqrt(1 + 0.5 sr^2); PSR = Phi(sr*sqrt(n-1)/denom)
    sr, n = 0.12, 300
    denom = math.sqrt(1 + 0.5 * sr * sr)
    expect = _norm_cdf(sr * math.sqrt(n - 1) / denom)
    assert abs(probabilistic_sharpe_ratio(sr, 0.0, n) - expect) < 1e-12


def test_expected_max_sharpe_monotonic():
    e10 = expected_max_sharpe(0.01, 10)
    e100 = expected_max_sharpe(0.01, 100)
    e672 = expected_max_sharpe(0.01, 672)
    assert 0 < e10 < e100 < e672              # more trials -> higher luck bar
    assert expected_max_sharpe(0.04, 672) > e672   # more SR variance -> higher
    assert expected_max_sharpe(0.01, 1) == 0.0     # <2 trials -> no deflation
    assert expected_max_sharpe(0.0, 672) == 0.0    # no SR variance -> no bar


def test_dsr_deflates_below_psr_for_many_trials():
    sr, var, n_obs = 0.15, 0.02, 252
    psr_vs_zero = probabilistic_sharpe_ratio(sr, 0.0, n_obs)
    res = deflated_sharpe_ratio(sr, var, n_trials=672, n_obs=n_obs)
    assert 0.0 <= res.dsr <= 1.0
    assert res.sr0_expected_max > 0.0
    assert res.dsr < psr_vs_zero               # selection-bias haircut bites


def test_dsr_equals_psr_when_single_trial():
    sr, n_obs = 0.15, 252
    res = deflated_sharpe_ratio(sr, 0.02, n_trials=1, n_obs=n_obs)
    assert abs(res.dsr - probabilistic_sharpe_ratio(sr, 0.0, n_obs)) < 1e-12


def test_dsr_luck_only_is_modest():
    # An "edge" exactly at the expected-max-under-null should give DSR ~ 0.5,
    # NOT a confident pass -- the whole point of the haircut.
    var, n_trials, n_obs = 0.02, 672, 252
    sr0 = expected_max_sharpe(var, n_trials)
    res = deflated_sharpe_ratio(sr0, var, n_trials, n_obs)
    assert abs(res.dsr - 0.5) < 0.05


def test_effective_independent_trials_bounds():
    n = 20
    ident = np.eye(n)
    assert abs(effective_independent_trials(ident) - n) < 1e-6      # all indep
    allcorr = np.ones((n, n))                                       # rank 1
    assert abs(effective_independent_trials(allcorr) - 1.0) < 1e-6  # all same
    # Two perfectly-correlated blocks -> ~2 effective trials.
    block = np.block([[np.ones((n, n)), np.zeros((n, n))],
                      [np.zeros((n, n)), np.ones((n, n))]])
    assert 1.5 < effective_independent_trials(block) < 2.5


def test_trial_sharpe_stats():
    st = trial_sharpe_stats([0.1, 0.2, 0.3, None, float("nan")])
    assert st["n"] == 3
    assert abs(st["mean"] - 0.2) < 1e-12
    assert st["var"] > 0.0
