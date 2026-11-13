"""Tests for the volatility regime adjustment (midoffice15): the two cross-sectional bias
statistics, read out of sample from the forecasts made the day before; the multiplier, the root
of their exponentially weighted mean square, one until the history is long enough; the adjusted
forecast; and that nothing after a date moves the multiplier as of it.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date, timedelta

import numpy as np
import pytest

from praxis.midoffice.model.bias import RiskForecast, eigenportfolios
from praxis.midoffice.model.characteristics import CharacteristicPanel
from praxis.midoffice.model.covariance import ExponentialCovariance
from praxis.midoffice.model.crosssectional import (
    cross_sectional_models_as_of,
    estimate_cross_sectional_model,
)
from praxis.midoffice.model.panel import Frequency, ReturnPanel
from praxis.midoffice.model.regime import (
    REGIME_MIN_DAYS,
    RegimeHistory,
    factor_bias,
    factor_history,
    factor_history_guard,
    regime_history,
    regime_multiplier,
    specific_bias,
)


def test_the_statistics_are_root_mean_squares_of_the_standardised_returns():
    """B_F^2 the mean over the factors of (f / sigma)^2; B_S^2 the weighted mean over the
    securities of (u / s)^2, the weights normalised over those with all three -- a missing
    return, a zero volatility or a zero weight leaves a security out."""
    value, k = factor_bias(np.array([0.02, -0.01, 0.03, np.nan]), np.array([0.01, 0.01, 0.0, 0.1]))
    assert (value, k) == (pytest.approx((4.0 + 1.0) / 2.0), 2)
    u = np.array([0.02, -0.03, 0.01, np.nan, 0.05])
    s = np.array([0.01, 0.01, 0.02, 0.01, 0.01])
    w = np.array([3.0, 1.0, 0.0, 5.0, 1.0])
    value, n, effective = specific_bias(u, s, w)
    assert n == 3
    assert value == pytest.approx((3.0 * 4.0 + 1.0 * 9.0 + 1.0 * 25.0) / 5.0)
    assert effective == pytest.approx(1.0 / ((3 / 5) ** 2 + (1 / 5) ** 2 + (1 / 5) ** 2))
    assert math.isnan(factor_bias(np.array([np.nan]), np.array([0.1]))[0])
    with pytest.raises(ValueError):
        factor_bias(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        specific_bias(np.zeros(2), np.zeros(2), np.zeros(3))


def test_the_multiplier_is_one_until_its_history_is_long_enough_then_a_weighted_root():
    """lambda is one for a side off and while the history is shorter than min_days; from then on
    the root of the squared statistics' mean weighted 0.5 ** (age / h), the latest at age 0."""
    squared = [4.0] * (REGIME_MIN_DAYS - 1)
    assert regime_multiplier(squared, 21.0) == 1.0
    assert regime_multiplier(squared + [4.0], 21.0) == pytest.approx(2.0)
    assert regime_multiplier(squared + [4.0], None) == 1.0
    values = [1.0, 9.0, 4.0]
    weights = 0.5 ** (np.array([2.0, 1.0, 0.0]) / 5.0)
    assert regime_multiplier(values, 5.0, min_days=3) == pytest.approx(
        math.sqrt(weights @ values / weights.sum()), rel=1e-15)
    # A short half-life follows the latest days; a long one is near the plain mean.
    calm_then_storm = [1.0] * 200 + [9.0] * 10
    assert regime_multiplier(calm_then_storm, 5.0) > 2.5 > regime_multiplier(calm_then_storm, 500.0)
    assert regime_multiplier(calm_then_storm, 1e9) == pytest.approx(math.sqrt(290.0 / 210.0))
    with pytest.raises(ValueError, match="positive"):
        regime_multiplier(values, 0.0)


def _forecast(rng, n_secs=200, k=4, periods=252.0):
    a = rng.normal(size=(k, k))
    covariance = (a @ a.T / k + np.eye(k)) * 0.04
    exposures = np.column_stack([np.ones(n_secs), rng.normal(size=(n_secs, k - 1))])
    specific = rng.uniform(0.15, 0.45, size=n_secs)
    return RiskForecast(covariance, exposures, specific, periods)


def test_a_day_is_read_from_the_forecast_made_the_day_before():
    """Each factor's return over its forecast volatility at the period's scale; each member's
    specific return -- its return less its forecast exposures times the day's factor returns --
    over its specific volatility, weighted by its capitalisation the day before; a security with
    no forecast, no return, no capitalisation or no membership is left out."""
    rng = np.random.default_rng(3)
    forecast = _forecast(rng, n_secs=6, k=3)
    forecast = dataclasses.replace(forecast, specific=forecast.specific.copy())
    forecast.specific[4] = np.nan
    f = np.array([0.01, -0.02, 0.005])
    returns = forecast.exposures @ f + np.array([0.01, -0.02, 0.03, 0.0, 0.05, np.nan])
    caps = np.array([1.0, 2.0, 3.0, 0.0, 5.0, 6.0])
    members = np.array([True, True, False, True, True, True])
    history = RegimeHistory()
    history.add(date(2024, 1, 2), forecast, f, returns, caps, members)
    sigma = np.sqrt(np.diag(forecast.covariance) / 252.0)
    assert history.factor_squared[0] == pytest.approx(np.mean((f / sigma) ** 2), rel=1e-12)
    s = forecast.specific / math.sqrt(252.0)
    held = [0, 1]  # 2 no member, 3 no cap, 4 no forecast, 5 no return
    z = np.array([0.01, -0.02]) / s[held]
    assert history.specific_squared[0] == pytest.approx((1.0 * z[0] ** 2 + 2.0 * z[1] ** 2) / 3.0,
                                                        rel=1e-12)
    assert (history.factors, history.members) == ([3], [2])
    with pytest.raises(ValueError, match="runs forward"):
        history.add(date(2024, 1, 2), forecast, f, returns, caps, members)
    with pytest.raises(ValueError, match="factor returns"):
        history.add(date(2024, 1, 3), forecast, f[:2], returns, caps, members)
    history.add(date(2024, 1, 3), forecast, f * 2.0, returns, caps, members)
    assert np.array_equal(history.path("factor", 10.0), [
        regime_multiplier(history.factor_squared[:1], 10.0),
        regime_multiplier(history.factor_squared, 10.0)])


def test_forecasts_half_the_truth_give_multipliers_of_two_and_right_ones_of_one():
    """Returns drawn at twice the forecast volatility, factor and specific alike: both statistics
    read about four, and both multipliers about two; drawn at the forecast's own, about one."""
    rng = np.random.default_rng(11)
    forecast = _forecast(rng)
    root = np.linalg.cholesky(forecast.covariance / 252.0)
    caps = rng.uniform(1.0, 10.0, size=forecast.specific.size)
    for scale, expected in ((2.0, 2.0), (1.0, 1.0)):
        history = RegimeHistory()
        start = date(2024, 1, 1)
        for i in range(600):
            f = scale * root @ rng.standard_normal(root.shape[0])
            u = scale * forecast.specific / math.sqrt(252.0) * rng.standard_normal(caps.size)
            history.add(start + timedelta(days=i), forecast, f, forecast.exposures @ f + u, caps)
        assert history.multiplier("factor", 600.0) == pytest.approx(expected, rel=0.06)
        assert history.multiplier("specific", 600.0) == pytest.approx(expected, rel=0.06)
        assert history.multiplier("factor", None) == 1.0


def test_the_regime_adjusted_forecast_scales_the_covariance_and_the_specific_risk():
    """The factor covariance times lambda_F^2 and its eigenbasis with it, each specific volatility
    times lambda_S; a portfolio's variance is its two parts so scaled; at one and one the forecast
    itself."""
    rng = np.random.default_rng(5)
    forecast = _forecast(rng, n_secs=30)
    values, vectors = eigenportfolios(forecast.covariance)
    based = dataclasses.replace(forecast, eigenbasis=(values, vectors))
    adjusted = based.regime_adjusted(1.5, 0.8)
    assert np.array_equal(adjusted.covariance, forecast.covariance * 2.25)
    assert np.array_equal(adjusted.specific, forecast.specific * 0.8)
    assert np.array_equal(adjusted.eigenbasis[0], values * 2.25)
    assert adjusted.eigenbasis[1] is vectors
    w = rng.dirichlet(np.ones(30))
    x = w @ forecast.exposures
    expected = 2.25 * x @ forecast.covariance @ x + 0.64 * (w**2) @ forecast.specific**2
    assert adjusted.variances(w)[0] == pytest.approx(expected, rel=1e-12)
    assert forecast.regime_adjusted(1.0, 1.0) is forecast
    assert np.array_equal(forecast.regime_adjusted(2.0).specific, forecast.specific)
    with pytest.raises(ValueError, match="positive"):
        forecast.regime_adjusted(0.0, 1.0)


def _daily_world(n_periods=56, n_secs=40, seed=29):
    rng = np.random.default_rng(seed)
    dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(n_periods))
    secs = tuple(f"S{j:02d}" for j in range(n_secs))
    chars = CharacteristicPanel(dates, secs, ("C0", "C1"),
                                rng.normal(size=(n_periods, n_secs, 2)), Frequency.DAILY)
    returns = ReturnPanel(dates, secs, rng.normal(scale=0.012, size=(n_periods, n_secs)),
                          Frequency.DAILY)
    caps = rng.uniform(1.0, 9.0, size=(n_periods, n_secs))
    members = rng.random((n_periods, n_secs)) < 0.9
    return returns, chars, caps, members


def _options(caps):
    return dict(regression_weights=np.sqrt(caps), lag=1, standardise=True, annualise=True,
                min_securities=10, specific_risk="blend", specific_history="covered",
                specific_half_life=12.0, covariance_estimator=ExponentialCovariance(half_life=20.0))


def test_nothing_after_a_date_moves_the_multiplier_as_of_it():
    """The point-in-time property, through the bias test's own reading: the model refitted as of
    each date, each day's statistics from the forecast the day before and the refit as of the
    day. Everything after a date redrawn -- returns, characteristics, capitalisations,
    membership -- the history through it is the same to the bit, and so is every multiplier as
    of a date on or before it; after it they move."""
    returns, chars, caps, members = _daily_world()
    as_of = returns.dates[24:52]
    t = 40
    before = regime_history(returns, chars, as_of, capitalisations=caps, membership=members,
                            min_days=3, **_options(caps))
    rng = np.random.default_rng(71)
    later = returns.returns.copy()
    later[t + 1:] = rng.normal(scale=0.05, size=later[t + 1:].shape)
    later_chars = chars.values.copy()
    later_chars[t + 1:] = rng.normal(size=later_chars[t + 1:].shape)
    later_caps = caps.copy()
    later_caps[t + 1:] = rng.uniform(1.0, 9.0, size=later_caps[t + 1:].shape)
    later_members = members.copy()
    later_members[t + 1:] = rng.random(later_members[t + 1:].shape) < 0.5
    after = regime_history(dataclasses.replace(returns, returns=later),
                           dataclasses.replace(chars, values=later_chars), as_of,
                           capitalisations=later_caps, membership=later_members, min_days=3,
                           **_options(later_caps))
    assert before.dates == after.dates == list(as_of[1:])
    k = before.dates.index(returns.dates[t]) + 1
    assert before.factor_squared[:k] == after.factor_squared[:k]
    assert before.specific_squared[:k] == after.specific_squared[:k]
    assert before.members[:k] == after.members[:k]
    for side in ("factor", "specific"):
        for half_life in (2.0, 8.0):
            assert np.array_equal(before.path(side, half_life)[:k],
                                  after.path(side, half_life)[:k])
            assert not np.array_equal(before.path(side, half_life)[k:],
                                      after.path(side, half_life)[k:])
    assert before.multiplier("factor", 8.0) != 1.0
    with pytest.raises(ValueError, match="consecutive"):
        regime_history(returns, chars, [as_of[0], as_of[2]], capitalisations=caps,
                       membership=members, **_options(caps))


def test_the_factor_side_read_from_one_fit_is_the_refits_and_its_guard_names_a_difference():
    """midoffice17: the factor side's history read from one fit -- the estimator on the fit's own
    factor returns through each day before -- equals the history the model refitted as of every
    day gives, to the bit. The guard finds nothing on true refits, and names a refit whose factor
    returns differ in one bit and an estimator not the model's. A factor-only history refuses
    the specific side; a model whose covariance is not its estimator's alone is refused."""
    returns, chars, caps, members = _daily_world()
    as_of = returns.dates[24:52]
    options = _options(caps)
    estimator = options["covariance_estimator"]
    refitted = regime_history(returns, chars, as_of, capitalisations=caps, membership=members,
                              min_days=3, **options)
    full = estimate_cross_sectional_model(returns, chars, capitalisations=caps,
                                          membership=members, **options)
    fast = factor_history(full, estimator, as_of, min_days=3)
    assert fast.dates == refitted.dates == list(as_of[1:])
    assert fast.factor_squared == refitted.factor_squared
    assert fast.factors == refitted.factors
    assert np.array_equal(fast.path("factor", 8.0), refitted.path("factor", 8.0))
    assert refitted.has_specific and not fast.has_specific
    with pytest.raises(ValueError, match="specific"):
        fast.multiplier("specific", 8.0)
    assert fast.multiplier("specific", None) == 1.0
    picks = [as_of[0], as_of[13], as_of[-1]]
    refits = list(cross_sectional_models_as_of(returns, chars, picks, capitalisations=caps,
                                               membership=members, **options))
    assert factor_history_guard(full, estimator, refits) == []
    day, refit = refits[1]
    bent = refit.notes.factor_returns.returns.copy()
    bent[0, 0] = np.nextafter(bent[0, 0], np.inf)
    factors = dataclasses.replace(refit.notes.factor_returns, returns=bent)
    changed = dataclasses.replace(refit, notes=dataclasses.replace(refit.notes,
                                                                   factor_returns=factors))
    assert factor_history_guard(full, estimator, [(day, changed)]) == [
        f"{day}: its factor returns are not the full fit's through it"]
    problems = factor_history_guard(full, ExponentialCovariance(half_life=10.0), refits)
    assert len(problems) == 3 and all("factor volatilities" in p for p in problems)
    adjusted = estimate_cross_sectional_model(returns, chars, capitalisations=caps,
                                              membership=members, eigen_adjustment=1.0,
                                              eigen_draws=20, **options)
    with pytest.raises(ValueError, match="estimator's alone"):
        factor_history(adjusted, estimator, as_of)
