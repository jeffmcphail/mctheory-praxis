"""Seeded Monte Carlo draws repeat: every batch is seeded from the seed, not only the first.

Three functions draw their random-walk universes in batches of `n_assets` paths until they have
`n_samples` samples, and each seeded only its first batch, so a seeded call drew the rest from
the operating system's entropy and differed from run to run. The surface's
`_compute_single_point` was fixed in midoffice19 (item 3) after an interpolation test failed
about one run in twenty; `_compute_multi_point` and `generate_adf_critical_values` the same way
by ruling 4 of midoffice20's brief: the first batch by the seed itself, so its draws are
unchanged, each later batch by a seed derived from it. Each test spans several batches.
"""

import numpy as np

from praxis.stats.monte_carlo import generate_adf_critical_values
from praxis.stats.surface import _compute_multi_point, _register_multi_builtins


def _adf(seed):
    # 20 paths a batch, 50 samples: three batches.
    return generate_adf_critical_values(n_assets=20, n_obs=100, n_vars=2, n_samples=50,
                                        seed=seed)


def _multi(seed):
    _register_multi_builtins()
    # 6 paths a batch, 30 samples: five batches.
    return _compute_multi_point(
        "stepwise_regression", "random_walk", ["adf_t", "hurst", "half_life"], ["vr_profile"],
        {"n_assets": 6, "n_obs": 100, "n_vars": 2}, 30, [1, 5, 10, 90, 95, 99], seed,
        {"vr_max_lag": 10},
    )


def test_adf_critical_values_repeat_on_the_same_seed():
    first, second = _adf(42), _adf(42)
    assert np.array_equal(first.t_values, second.t_values)
    assert first.critical_values.values == second.critical_values.values
    # Not vacuous: another seed draws other values.
    assert not np.array_equal(first.t_values, _adf(43).t_values)


def test_adf_first_batch_is_the_seed_itself():
    # One batch alone is what the seed always drew; with more batches after it, its values are
    # among theirs unchanged (the t-values come back sorted, so compared as a multiset).
    one = generate_adf_critical_values(n_assets=20, n_obs=100, n_vars=2, n_samples=20,
                                       seed=42).t_values
    three = _adf(42).t_values
    remaining = list(three)
    for value in one:
        remaining.remove(value)
    assert len(remaining) == 30


def test_multi_point_repeats_on_the_same_seed():
    first, second = _multi(42), _multi(42)
    assert first["scalars"] == second["scalars"]
    assert first["derived_scalars"] == second["derived_scalars"]
    assert first["profiles"] == second["profiles"]
    assert first["scalars"] != _multi(43)["scalars"]
