"""Tests for the country factor as a covariance basis (midoffice18; Barra USE4, section 3.1): the
industry model's fit with a country factor and the cap-weighted industry returns constrained to
sum to zero is the same fit, its factor returns a change of basis; a covariance estimator that is a
quadratic form is basis-invariant, and the two-half-life estimator is not.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from praxis.midoffice.model.characteristics import CharacteristicPanel
from praxis.midoffice.model.country import (
    country_covariance,
    country_map,
    country_weights,
    to_country_basis,
)
from praxis.midoffice.model.covariance import ExponentialCovariance, HorizonCovariance
from praxis.midoffice.model.crosssectional import (
    estimate_cross_sectional_model,
    regression_design,
)
from praxis.midoffice.model.panel import Frequency, ReturnPanel


def _industry_world(n_periods=150, n_secs=60, n_industries=4, seed=41):
    """Returns from a market move, industry moves and two styles; each security in one industry
    (one-hot columns 0-3), the styles columns 4-5; capitalisations drifting."""
    rng = np.random.default_rng(seed)
    dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(n_periods))
    secs = tuple(f"S{j:02d}" for j in range(n_secs))
    industry = rng.integers(0, n_industries, n_secs)
    industry[:n_industries] = np.arange(n_industries)
    values = np.zeros((n_periods, n_secs, n_industries + 2))
    values[:, np.arange(n_secs), industry] = 1.0
    values[:, :, n_industries:] = rng.normal(size=(n_periods, n_secs, 2))
    names = tuple(f"I{i}" for i in range(n_industries)) + ("Style A", "Style B")
    chars = CharacteristicPanel(dates, secs, names, values, Frequency.DAILY)
    market = rng.normal(scale=0.01, size=n_periods)
    sector = rng.normal(scale=0.006, size=(n_periods, n_industries))
    style = rng.normal(scale=0.004, size=(n_periods, 2))
    noise = rng.normal(scale=0.015, size=(n_periods, n_secs))
    r = (market[:, None] + sector[:, industry]
         + np.einsum("tk,tnk->tn", style, values[:, :, n_industries:]) + noise)
    returns = ReturnPanel(dates, secs, r, Frequency.DAILY)
    caps = np.exp(rng.normal(size=n_secs))[None, :] * np.exp(
        np.cumsum(rng.normal(scale=0.02, size=(n_periods, n_secs)), axis=0))
    members = rng.random((n_periods, n_secs)) < 0.95
    return returns, chars, caps, members, list(range(n_industries))


def _fit(world, estimator):
    returns, chars, caps, members, _ = world
    return estimate_cross_sectional_model(
        returns, chars, capitalisations=caps, regression_weights=np.sqrt(caps),
        membership=members, fit_market=False, standardise=False, min_securities=10,
        covariance_estimator=estimator)


def _constrained_country_fit(r, x, w, industries, c):
    """WLS of r on a country column, the industries and the styles, the industries' returns
    constrained to sum_i c_i f_i = 0 -- the last industry eliminated. The country's, the
    industries' and the styles' returns, and the residuals."""
    last, free = industries[-1], industries[:-1]
    styles = [k for k in range(x.shape[1]) if k not in industries]
    columns = np.column_stack([np.ones(len(r)),
                               x[:, free] - np.outer(x[:, last], c[:-1] / c[-1]),
                               x[:, styles]])
    root = np.sqrt(w)
    coefficients, *_ = np.linalg.lstsq(columns * root[:, None], r * root, rcond=None)
    country = coefficients[0]
    f_free = coefficients[1:1 + len(free)]
    f_industries = np.append(f_free, -(c[:-1] @ f_free) / c[-1])
    f_styles = coefficients[1 + len(free):]
    fitted = country + x[:, industries] @ f_industries + x[:, styles] @ f_styles
    return country, f_industries, f_styles, r - fitted


def test_the_country_fit_is_the_industry_fit_in_another_basis():
    """USE4's regression -- a country factor, the cap-weighted industry returns summing to zero --
    on each period's own design gives the industry model's factor returns taken to the country
    basis (the country the cap-weighted mean of the industries, each industry less it, the styles
    the same) and the same residuals, to rounding."""
    world = _industry_world()
    returns, chars, caps, members, industries = world
    model = _fit(world, ExponentialCovariance(half_life=30.0))
    weights = country_weights(returns, chars, caps, industries,
                              regression_weights=np.sqrt(caps), membership=members)
    design = regression_design(returns, chars, regression_weights=np.sqrt(caps),
                               membership=members)
    factors = model.notes.factor_returns
    at = {d: i for i, d in enumerate(design.dates)}
    for day in (factors.dates[0], factors.dates[50], factors.dates[-1]):
        t = at[day]
        usable = design.in_regression[t]
        r, x, w = design.returns[t, usable], design.exposures[t][usable], design.weights[t, usable]
        c = weights[day]
        assert c.sum() == pytest.approx(1.0, abs=1e-15)
        country, f_ind, f_sty, u = _constrained_country_fit(r, x, w, industries, c)
        f = factors.returns[factors.dates.index(day)]
        basis = to_country_basis(f[None], industries, c[None])[0]
        assert np.allclose([country, *f_ind, *f_sty], basis, rtol=1e-10, atol=1e-15)
        assert np.allclose(u, r - x @ f, rtol=1e-10, atol=1e-15)
        assert abs(c @ basis[1:1 + len(industries)]) < 1e-15


def test_the_basis_maps_back_and_only_the_two_half_life_estimator_changes():
    """The map back is exact; a quadratic-form estimator's covariance estimated in the country
    basis and mapped back is the industry model's own, to rounding, whatever the weights; the
    two-half-life estimator's is not -- in the basis the country's variance is the volatility
    window's, the market's on the industries' correlations no longer."""
    world = _industry_world()
    returns, chars, caps, members, industries = world
    weights = country_weights(returns, chars, caps, industries,
                              regression_weights=np.sqrt(caps), membership=members)
    single = ExponentialCovariance(half_life=30.0)
    model = _fit(world, single)
    factors = model.notes.factor_returns
    w = np.array([weights[d] for d in factors.dates])
    basis = to_country_basis(factors.returns, industries, w)
    t = country_map(factors.returns.shape[1], industries)
    assert np.allclose(basis @ t.T, factors.returns, rtol=0.0, atol=1e-16)
    own, mapped = country_covariance(model, single, weights, industries)
    assert own.shape == (7, 7)
    assert np.allclose(mapped, model.covariance.matrix, rtol=1e-10, atol=1e-16)
    split = HorizonCovariance(volatility_half_life=10.0, correlation_half_life=100.0)
    two = _fit(world, split)
    own, mapped = country_covariance(two, split, weights, industries)
    assert not np.allclose(mapped, two.covariance.matrix, rtol=1e-3)
    short = ExponentialCovariance(half_life=10.0).estimate(basis[:, :1]).matrix * 252.0
    assert own[0, 0] == pytest.approx(short[0, 0], rel=1e-12)
    with pytest.raises(ValueError, match="no industry weights"):
        country_covariance(model, single, {}, industries)
    with pytest.raises(ValueError, match="weights of shape"):
        to_country_basis(factors.returns, industries, w[:, :2])
