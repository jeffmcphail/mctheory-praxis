"""Out-of-sample bias statistics: the statistic, its band, the portfolios it is read on.

Checked against worlds whose risk is known, because a statistic of one is the claim: forecasts
that are the truth must score about one, and forecasts a known factor too low must score that
factor's inverse.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from scipy.optimize import minimize

from praxis.midoffice.model.bias import (
    RiskForecast,
    bias_band,
    bias_statistic,
    bias_statistics,
    buy_and_hold,
    eigenportfolios,
    evaluate,
    minimum_variance,
    rolling_bias,
)


def _forecast(n=40, k=4, seed=3, singular=False):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k)) * 0.1
    covariance = a @ a.T
    if singular:
        values, vectors = np.linalg.eigh(covariance)
        values[0] = 0.0
        covariance = vectors @ np.diag(values) @ vectors.T
    exposures = rng.normal(size=(n, k))
    specific = rng.uniform(0.15, 0.45, n)
    return RiskForecast(covariance, exposures, specific, 252.0)


def _dense(forecast):
    x = forecast.exposures
    return x @ forecast.covariance @ x.T + np.diag(forecast.specific**2)


# ------------------------------------------------------------------------ the statistic


def test_the_statistic_is_one_for_standard_outcomes_and_scales_with_them():
    z = np.random.default_rng(1).normal(size=20_000)
    low, high = bias_band(z.size)
    assert low < bias_statistic(z) < high
    assert bias_statistic(1.25 * z) == pytest.approx(1.25 * bias_statistic(z))
    assert bias_band(200) == pytest.approx((1 - 0.1, 1 + 0.1))
    assert np.isnan(bias_statistic(np.array([1.0, np.nan])))


def test_the_statistics_read_realised_over_forecast_and_roll_over_windows():
    rng = np.random.default_rng(2)
    forecast = rng.uniform(0.005, 0.02, 600)
    realised = rng.normal(size=600) * forecast * 1.1
    forecast[10] = np.nan
    stats = bias_statistics(forecast, realised, window=252)
    assert stats.n == 599
    assert stats.statistic == pytest.approx(bias_statistic(np.delete(realised / forecast, 10)))
    assert stats.rolling.size == 599 - 252 + 1
    assert stats.mrad == pytest.approx(np.mean(np.abs(stats.rolling - 1.0)))
    assert stats.verdict == "under" and abs(stats.kurtosis - 3.0) < 0.6
    assert rolling_bias(np.ones(10), 20).size == 0


# ------------------------------------------------------------------ forecasts and portfolios


def test_a_forecast_prices_portfolios_as_the_dense_covariance_does():
    forecast = _forecast()
    weights = np.random.default_rng(4).normal(size=(3, 40))
    dense = _dense(forecast)
    assert np.allclose(forecast.variances(weights), np.einsum("pi,ij,pj->p", weights, dense,
                                                              weights), rtol=1e-12)
    assert np.allclose(forecast.security_variances(), np.diag(dense), rtol=1e-12)
    gap = RiskForecast(forecast.covariance, forecast.exposures,
                       np.where(np.arange(40) == 5, np.nan, forecast.specific), 252.0)
    with pytest.raises(ValueError, match="no forecast"):
        gap.variances(np.eye(40)[5])


@pytest.mark.parametrize("singular", [False, True])
def test_the_unconstrained_minimum_variance_portfolio_is_the_closed_form(singular):
    forecast = _forecast(singular=singular)
    eligible = np.ones(40, dtype=bool)
    eligible[[0, 7]] = False
    portfolio = minimum_variance(forecast, eligible, long_only=False)
    dense = _dense(forecast)[np.ix_(eligible, eligible)]
    expected = np.linalg.solve(dense, np.ones(38))
    assert np.allclose(portfolio.weights[eligible], expected / expected.sum(), rtol=1e-9)
    assert portfolio.weights[[0, 7]].tolist() == [0.0, 0.0]
    assert portfolio.violation < 1e-9


def test_the_long_only_minimum_variance_portfolio_matches_a_general_solver():
    forecast = _forecast(n=30, seed=8)
    eligible = np.ones(30, dtype=bool)
    portfolio = minimum_variance(forecast, eligible, long_only=True)
    w = portfolio.weights
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0)
    dense = _dense(forecast)
    reference = minimize(
        lambda v: v @ dense @ v, np.full(30, 1 / 30), jac=lambda v: 2 * dense @ v,
        bounds=[(0, None)] * 30, constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}],
        method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000},
    ).x
    assert w @ dense @ w <= reference @ dense @ reference * (1 + 1e-9)
    assert np.abs(w - reference).max() < 1e-4
    assert portfolio.violation < 1e-8
    assert 0 < portfolio.held < 30


def test_the_eigenportfolios_are_ordered_signed_and_rebuild_the_covariance():
    covariance = _forecast().covariance
    values, vectors = eigenportfolios(covariance)
    assert np.all(np.diff(values) <= 0.0)
    assert np.allclose(vectors @ np.diag(values) @ vectors.T, covariance, atol=1e-14)
    lead = vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])]
    assert np.all(lead > 0.0)


def test_buy_and_hold_compounds_and_holds_a_missing_return_as_cash():
    block = np.array([[0.10, 0.0], [0.10, np.nan], [-0.5, np.nan]])
    out = buy_and_hold(block, np.array([[0.5, 0.5]]))
    assert out[0] == pytest.approx(0.5 * (1.1 * 1.1 * 0.5 - 1.0))


# ------------------------------------------------------------------- the whole test, known


def _known_world(n_dates=700, n_secs=150, seed=17, scale=1.0):
    """Returns drawn from a factor model whose covariance, exposures and specific risks are
    the forecasts, held fixed; `scale` multiplies every forecast volatility."""
    rng = np.random.default_rng(seed)
    k_ind, k_style = 3, 2
    industry = rng.integers(0, k_ind, n_secs)
    x = np.column_stack([np.eye(k_ind)[industry], rng.normal(size=(n_secs, k_style))])
    a = rng.normal(size=(5, 5)) * 0.08
    covariance = a @ a.T + np.diag([0.02, 0.02, 0.02, 0.003, 0.002])
    specific = rng.uniform(0.15, 0.45, n_secs)
    f = rng.multivariate_normal(np.zeros(5), covariance / 252.0, n_dates)
    u = rng.normal(size=(n_dates, n_secs)) * specific / np.sqrt(252.0)
    returns = f @ x.T + u
    forecast = RiskForecast(covariance * scale**2, x, specific * scale, 252.0)
    dates = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(n_dates))
    securities = tuple(f"S{j:03d}" for j in range(n_secs))
    caps = np.broadcast_to(np.exp(rng.normal(size=n_secs)), (n_dates, n_secs)).copy()
    membership = np.ones((n_dates, n_secs), dtype=bool)
    groups = {f"I{g}": industry == g for g in range(k_ind)}
    return forecast, dates, securities, returns, membership, caps, groups, f


def _run(world, rows, **kw):
    forecast, dates, securities, returns, membership, caps, groups, f = world
    return evaluate(((t, forecast) for t in rows), dates=dates, securities=securities,
                    returns=returns, membership=membership, capitalisations=caps,
                    groups=groups, factor_returns=f, smallest=20, largest=20,
                    random_portfolios=20, random_names=15, **kw)


def test_forecasts_that_are_the_truth_score_one_and_a_known_shortfall_its_inverse():
    rows = range(0, 699)
    truth = _run(_known_world(), rows)
    short = _run(_known_world(scale=0.8), rows)
    band = np.sqrt(2.0 / 699)
    for name in ("the index, cap-weighted", "the index, equally weighted",
                 "minimum variance, unconstrained", "industry: I0"):
        assert abs(truth.statistics(name).statistic - 1.0) < 2.0 * band
        assert short.statistics(name).statistic == pytest.approx(
            truth.statistics(name).statistic / 0.8, rel=1e-9)
    members = [bias_statistics(truth.member_forecast[:, j], truth.member_realised[:, j])
               for j in range(150)]
    assert abs(np.median([m.statistic for m in members]) - 1.0) < 0.02
    inside = np.mean([m.verdict == "in" for m in members])
    assert 0.85 < inside <= 1.0
    eigen = [bias_statistics(truth.eigen_forecast[k], truth.eigen_realised[k]) for k in range(5)]
    assert all(abs(e.statistic - 1.0) < 2.5 * band for e in eigen)
    # The horizon check: every 21st date, 21 periods ahead, non-overlapping.
    assert len(truth.horizon_dates) == 699 // 21 + (1 if 699 % 21 else 0) - 1
    assert abs(truth.statistics("the index, cap-weighted", horizon=True).statistic - 1.0) < 0.35
    assert truth.min_variance_violation["long-only"].max() < 1e-7


def test_a_member_with_no_next_return_is_left_out_of_that_days_portfolios_and_counted():
    world = list(_known_world(n_dates=60, n_secs=40, seed=19))
    returns = world[3].copy()
    returns[11, 5] = np.nan
    world[3] = returns
    run = _run(tuple(world), range(0, 59), horizon=21)
    assert run.dropped == 1
    assert np.isnan(run.member_forecast[10, 5]) and np.isfinite(run.member_forecast[11, 5])
    with pytest.raises(ValueError, match="no next period"):
        _run(tuple(world), [59])
