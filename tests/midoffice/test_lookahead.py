"""Look-ahead as a test failure, rather than something someone has to notice.

Every period's regression is fixed the day before its return: the exposures, their
standardisation, the weights and the membership. So pricing one day differently -- every price
dated t multiplied by its own random factor -- may move day t's returns and nothing else that
period's regression is defined by. Run end to end: prices, descriptors, both standardisations
and the risk-index blend, then the estimator. The blend's weights are held fixed, because
fitting them reads every period's returns -- the last test says what that costs.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from praxis.midoffice.model.crosssectional import (
    estimate_cross_sectional_model,
    regression_design,
)
from praxis.midoffice.model.descriptors import (
    SecurityPanel,
    build_risk_indices,
    compute_descriptors,
    fit_index_weights,
)
from praxis.midoffice.model.panel import Frequency, ReturnPanel

REPRICED = 70
"""The period whose prices are changed: inside the estimation window, after every descriptor
has its full history."""


def _index(n_periods: int = 96, n_secs: int = 50, seed: int = 31):
    """An index on a monthly panel: eight members leave at period 60 and trade on, one
    security has no price before period 30, and every price field is present otherwise."""
    rng = np.random.default_rng(seed)
    beta = np.linspace(0.4, 1.8, n_secs)
    market = rng.normal(scale=0.045, size=n_periods)
    returns = market[:, None] * beta + rng.normal(size=(n_periods, n_secs)) * np.linspace(
        0.02, 0.10, n_secs
    )
    prices = 50.0 * np.cumprod(1.0 + returns, axis=0)
    shares = np.tile(np.linspace(1e6, 5e8, n_secs), (n_periods, 1))
    volumes = shares * rng.uniform(0.02, 0.2, size=(n_periods, n_secs))
    highs = prices * rng.uniform(1.01, 1.10, size=(n_periods, n_secs))
    lows = prices * rng.uniform(0.90, 0.99, size=(n_periods, n_secs))
    for array in (prices, volumes, highs, lows):
        array[:30, 0] = np.nan
    panel = SecurityPanel(
        tuple(date(2000 + t // 12, t % 12 + 1, 28) for t in range(n_periods)),
        tuple(f"S{i:02d}" for i in range(n_secs)),
        prices,
        unadjusted_prices=prices.copy(),
        volumes=volumes,
        shares_outstanding=shares,
        highs=highs,
        lows=lows,
        frequency=Frequency.MONTHLY,
    )
    members = np.ones((n_periods, n_secs), dtype=bool)
    members[60:, 1:9] = False
    return panel, market, members


def _repriced(panel: SecurityPanel, row: int, seed: int = 41) -> SecurityPanel:
    """Every price dated `row` -- close, traded close, high and low -- times its own factor."""
    factor = np.random.default_rng(seed).uniform(0.7, 1.4, panel.n_securities)
    arrays = {}
    for name in ("prices", "unadjusted_prices", "highs", "lows"):
        values = getattr(panel, name).copy()
        values[row] *= factor
        arrays[name] = values
    return SecurityPanel(
        panel.dates,
        panel.securities,
        arrays["prices"],
        unadjusted_prices=arrays["unadjusted_prices"],
        volumes=panel.volumes,
        shares_outstanding=panel.shares_outstanding,
        highs=arrays["highs"],
        lows=arrays["lows"],
        frequency=panel.frequency,
    )


def _standardised(panel, market, members):
    caps = panel.market_capitalisation()
    descriptors, _ = compute_descriptors(
        panel, market_returns=market, beta_window_months=24.0, membership=members
    )
    return descriptors.standardise(caps, membership=members), caps


def _blend(panel, market, members):
    """Weights fitted by explanatory power, once, on the panel as given."""
    standardised, _ = _standardised(panel, market, members)
    return {
        index: fit_index_weights(standardised, index, panel.returns(), membership=members)
        for index in ("Momentum", "Trading Activity")
    }


def _model(panel, market, members, blend, lag=1):
    """The run script's pipeline: the design every period is handed, and the model."""
    standardised, caps = _standardised(panel, market, members)
    indices = build_risk_indices(
        standardised, weights=blend, cap_weights=caps, membership=members
    )
    returns = ReturnPanel(panel.dates, panel.securities, panel.returns(), panel.frequency)
    weights = np.sqrt(caps)
    design = regression_design(
        returns, indices, regression_weights=weights, membership=members, lag=lag
    )
    model = estimate_cross_sectional_model(
        returns,
        indices,
        regression_weights=weights,
        membership=members,
        lag=lag,
        standardise=False,
        min_securities=20,
        specific_risk="blend",
        capitalisations=caps,
    )
    return design, model


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def test_pricing_day_t_differently_moves_its_returns_and_nothing_its_regression_is_fixed_by():
    panel, market, members = _index()
    blend = _blend(panel, market, members)
    base, base_model = _model(panel, market, members, blend)
    moved, moved_model = _model(_repriced(panel, REPRICED), market, members, blend)
    t = REPRICED
    assert base.in_regression[t].sum() >= 20
    # Period t's exposures, weights, membership and estimation universe, bit for bit.
    assert _same_bits(base.exposures[t], moved.exposures[t])
    assert _same_bits(base.weights[t], moved.weights[t])
    assert _same_bits(base.members[t], moved.members[t])
    assert _same_bits(base.in_regression[t], moved.in_regression[t])
    # Only its returns moved -- every one with a price on t and the day before.
    priced = np.isfinite(base.returns[t])
    assert priced.sum() >= 40
    assert (base.returns[t][priced] != moved.returns[t][priced]).all()
    # Every earlier period is untouched, and so is every earlier fit.
    for block in ("returns", "exposures", "weights", "members", "in_regression"):
        assert _same_bits(getattr(base, block)[:t], getattr(moved, block)[:t]), block
    earlier = sum(d < panel.dates[t] for d in base_model.notes.estimated_dates)
    assert earlier >= 20
    before = base_model.notes.factor_returns.returns
    after = moved_model.notes.factor_returns.returns
    assert _same_bits(before[:earlier], after[:earlier])
    assert not np.array_equal(before[earlier], after[earlier])
    # Day t's prices belong in day t's exposures, which period t+1 reads.
    assert not _same_bits(base.exposures[t + 1], moved.exposures[t + 1])


def test_regressing_same_day_fails_the_same_check_which_is_what_the_check_is_for():
    """At a lag of zero, period t's regression reads day t's exposures and weights, built from
    the prices just changed -- LNCAP and the square-root-cap weight among them."""
    panel, market, members = _index()
    blend = _blend(panel, market, members)
    base, _ = _model(panel, market, members, blend, lag=0)
    moved, _ = _model(_repriced(panel, REPRICED), market, members, blend, lag=0)
    t = REPRICED
    assert not _same_bits(base.exposures[t], moved.exposures[t])
    assert not _same_bits(base.weights[t], moved.weights[t])


def test_refitting_the_blend_after_repricing_moves_every_period_the_earliest_included():
    """Why the blend is held fixed above, and the record of what fit_index_weights still does.
    It pairs each descriptor with the next period's return, as the regressions do, but pools
    every period's into one regression -- whole-sample weight selection, look-ahead in the
    strict sense -- so a price on day t moves the weights, and with them the exposures of
    every period before t."""
    panel, market, members = _index()
    repriced = _repriced(panel, REPRICED)
    blend = _blend(panel, market, members)
    refit = _blend(repriced, market, members)
    assert refit != blend
    base, _ = _model(panel, market, members, blend)
    moved, _ = _model(repriced, market, members, refit)
    first = int(np.flatnonzero(base.in_regression.any(axis=1))[0])
    assert first < REPRICED // 2
    assert not _same_bits(base.exposures[first], moved.exposures[first])
