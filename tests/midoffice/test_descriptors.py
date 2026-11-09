"""USE3's price-and-volume descriptors, from Appendix A of the US-E3 handbook.

Each descriptor is checked against its published definition computed independently, because
a descriptor that merely runs is worthless -- the whole value of implementing a published
specification is that it matches the specification.

The one thing deliberately not reproduced is the blend: Appendix A states that "the method of
combining these descriptors into risk indices is proprietary to BARRA", so `blend_to_index`
equal-weights by default and `fit_index_weights` estimates weights by explanatory power the
way the handbook describes. Both are tested as what they are -- a reimplementation of the
method -- rather than as a claimed reproduction of Barra's numbers.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from datetime import date

import numpy as np
import pytest

from praxis.midoffice.model.descriptors import (
    PRICE_DESCRIPTORS,
    WINDOW_DESCRIPTORS,
    SecurityPanel,
    blend_composition,
    blend_to_index,
    build_risk_indices,
    compute_descriptors,
    fit_index_weights,
    index_descriptors,
    market_model_window,
    nonlinear_size,
    risk_index_members,
)
from praxis.midoffice.model.panel import Frequency


def _dates(n: int) -> tuple[date, ...]:
    return tuple(date(2000 + t // 12, t % 12 + 1, 28) for t in range(n))


def _panel(
    n_periods: int = 120,
    n_secs: int = 40,
    seed: int = 3,
    frequency: Frequency = Frequency.MONTHLY,
    turnover: float = 0.08,
    split: float | None = None,
):
    rng = np.random.default_rng(seed)
    beta = np.linspace(0.4, 1.8, n_secs)
    residual = np.linspace(0.02, 0.10, n_secs)
    market = rng.normal(scale=0.045, size=n_periods)
    returns = market[:, None] * beta[None, :] + rng.normal(
        size=(n_periods, n_secs)
    ) * residual
    prices = 50.0 * np.cumprod(1.0 + returns, axis=0)
    shares = np.tile(np.linspace(1e6, 5e8, n_secs), (n_periods, 1))
    # A four-for-one split halfway through, for the odd-numbered securities only: before it
    # they traded at four times the adjusted price on a quarter of the shares.
    unadjusted = prices.copy()
    split_row = n_periods // 2
    if split is not None:
        unadjusted[:split_row, 1::2] *= split
        shares = shares.copy()
        shares[:split_row, 1::2] /= split
    panel = SecurityPanel(
        _dates(n_periods),
        tuple(f"S{i:02d}" for i in range(n_secs)),
        prices,
        unadjusted_prices=unadjusted,
        volumes=shares * turnover,
        shares_outstanding=shares,
        highs=prices * 1.05,
        lows=prices * 0.95,
        frequency=frequency,
    )
    return panel, market, beta, residual


def _column(panel_out, name: str) -> np.ndarray:
    return panel_out.values[-1, :, panel_out.names.index(name)]


# ----------------------------------------------------------- definitions, one at a time


def test_lncap_is_the_log_of_price_times_shares():
    panel, market, _, _ = _panel()
    out, _ = compute_descriptors(panel, market_returns=market)
    assert np.allclose(
        _column(out, "LNCAP"),
        np.log(panel.prices[-1] * panel.shares_outstanding[-1]),
    )


def test_lpri_is_the_log_of_the_price():
    panel, market, _, _ = _panel()
    out, _ = compute_descriptors(panel, market_returns=market)
    assert np.allclose(_column(out, "LPRI"), np.log(panel.prices[-1]))


def test_hilo_is_the_log_ratio_of_high_to_low():
    panel, market, _, _ = _panel()
    out, _ = compute_descriptors(panel, market_returns=market)
    assert np.allclose(_column(out, "HILO"), np.log(1.05 / 0.95))


def test_turnover_descriptors_are_exact_and_annualised_where_defined():
    """STOM is a plain monthly rate; the rest annualise, as Appendix A defines them.

    A stock turning over 8% of its shares every month has a 96% annual turnover, so STOQ,
    STOA and STO5 all land on 0.96 on a steady panel -- STOQ because it is `4 Vq / Nout` with
    Vq a quarter's worth, STO5 because it is `12 (1/T) sum Vs / Nout`. Agreement between the
    three on a steady panel is the check that the annualisers are right; they differ only
    when turnover is not steady.
    """
    panel, market, _, _ = _panel(turnover=0.08)
    out, _ = compute_descriptors(panel, market_returns=market)
    assert np.allclose(_column(out, "STOM"), 0.08)
    for name in ("STOQ", "STOA", "STO5"):
        assert np.allclose(_column(out, name), 0.96), name


def test_rstr_is_the_cumulative_twelve_period_excess_log_return():
    panel, market, _, _ = _panel()
    risk_free = np.full(panel.n_periods, 0.003)
    out, _ = compute_descriptors(
        panel, market_returns=market, risk_free=risk_free, names=("RSTR",)
    )
    expected = np.sum(np.log1p(panel.returns()[-12:] - 0.003), axis=0)
    assert np.allclose(_column(out, "RSTR"), expected)


def test_rstr_can_skip_the_end_of_its_window_and_by_default_skips_nothing():
    """The academic twelve-minus-one momentum cuts the last month from RSTR's window, so that a
    month's short-term reversal does not cancel the momentum being measured. A skip of k sums
    the window less its last k periods; 0, the default, sums them all -- the definition before
    the parameter, bit for bit. The window keeps its start, so the first value falls on the
    same date, and the coverage rule is applied to the periods summed: S00, with returns in
    the window's last seven periods alone, covers seven of twelve unskipped and four of the
    nine summed at a skip of three."""
    panel, market, _, _ = _panel()
    risk_free = np.full(panel.n_periods, 0.003)

    def rstr(source, **kw):
        out, _ = compute_descriptors(
            source, market_returns=market, risk_free=risk_free, names=("RSTR",), **kw
        )
        return out.values[:, :, 0]

    base = rstr(panel)
    assert base.tobytes() == rstr(panel, rstr_skip=0).tobytes()
    skipped = rstr(panel, rstr_skip=3)
    log_excess = np.log1p(panel.returns() - 0.003)
    t = panel.n_periods - 1
    assert np.array_equal(skipped[t], np.nansum(log_excess[t - 11 : t - 2], axis=0))
    assert not np.allclose(skipped[t], base[t])
    first = [int(np.flatnonzero(np.isfinite(v[:, 1]))[0]) for v in (base, skipped)]
    assert first == [11, 11]
    late = _blanked(panel, slice(0, t - 7), 0)
    assert np.isfinite(rstr(late)[t, 0])
    assert np.isnan(rstr(late, rstr_skip=3)[t, 0])
    anything = rstr(late, rstr_skip=3, min_coverage=0.0)[t, 0]
    excess = np.log1p(late.returns()[:, 0] - 0.003)
    assert anything == np.nansum(excess[t - 11 : t - 2])
    for bad in (-1, 12, 1.5, True):
        with pytest.raises(ValueError, match="rstr_skip"):
            rstr(panel, rstr_skip=bad)


def test_btsg_recovers_beta_times_residual_volatility():
    panel, market, beta, residual = _panel()
    out, _ = compute_descriptors(panel, market_returns=market, names=("BTSG",))
    measured = _column(out, "BTSG")
    truth = beta * residual
    assert np.corrcoef(measured, truth)[0, 1] > 0.9
    assert float(np.mean(measured / truth)) == pytest.approx(1.0, abs=0.1)


def test_the_market_model_can_be_weighted_by_a_half_life_and_on_a_monthly_panel_is_not():
    """USE4 estimates beta over 252 trading days with a half-life of 63. Given a half-life, the
    market model is weighted least squares, each period's weight halving every half-life back
    from the window's end, and the residual sd the weighted one, its degrees of freedom taken
    on Kish's effective count -- here against that fit computed independently, on a window
    with gaps. On a monthly panel the default is USE3's, every period weighted equally, as an
    infinite half-life asks."""
    panel, market, _, _ = _panel()
    rng = np.random.default_rng(4)
    gaps = rng.random(panel.prices.shape) < 0.1
    gapped = panel
    for j in range(panel.n_securities):
        gapped = _blanked(gapped, np.flatnonzero(gaps[:, j]), j)
    names = ("BTSG", "HALPHA", "SERDP", "VLVR")
    default, _ = compute_descriptors(gapped, market_returns=market, names=names)
    unweighted, _ = compute_descriptors(
        gapped, market_returns=market, names=names, beta_half_life=math.inf
    )
    assert default.values.tobytes() == unweighted.values.tobytes()
    weighted, _ = compute_descriptors(
        gapped, market_returns=market, names=names, beta_half_life=6.0
    )
    returns = gapped.returns()
    t, window = panel.n_periods - 1, 60
    x_all = market[t - window + 1 : t + 1]
    decay = 0.5 ** (np.arange(window - 1, -1, -1) / 6.0)
    for j in range(panel.n_securities):
        y_all = returns[t - window + 1 : t + 1, j]
        ok = np.isfinite(y_all) & np.isfinite(x_all)
        x, y, w = x_all[ok], y_all[ok], decay[ok]
        design = np.column_stack([np.ones(x.size), x])
        alpha, beta = np.linalg.solve(design.T @ (design * w[:, None]), design.T @ (w * y))
        error = y - alpha - beta * x
        effective = w.sum() ** 2 / (w @ w)
        sd = np.sqrt((w @ error**2) / w.sum() * effective / (effective - 2.0))
        got = weighted.values[t, j]
        assert got[names.index("HALPHA")] == pytest.approx(alpha, rel=1e-9, abs=1e-12), j
        assert got[names.index("BTSG")] == pytest.approx(max(beta, 0.0) * sd, rel=1e-9), j
    assert not np.allclose(
        weighted.values[t, :, 0], default.values[t, :, 0], equal_nan=True
    )
    for bad in (0.0, -6.0, float("nan"), -math.inf):
        with pytest.raises(ValueError, match="beta_half_life"):
            compute_descriptors(panel, market_returns=market, beta_half_life=bad)


def test_on_a_daily_panel_the_market_model_defaults_to_use4s_and_any_other_stays_runnable():
    """USE3's 60 months need more days than a five-year daily panel has, so on a daily panel
    the default is USE4's: 252 trading days, half-life 63 -- the same fit as asking for them,
    bit for bit. A window in months and an infinite half-life still give USE3's equal weights
    over any window: 36 months on a daily panel is 828 days, capped at the panel's length."""
    daily, market, _, _ = _panel(n_periods=300, n_secs=12, frequency=Frequency.DAILY)
    monthly, _, _, _ = _panel()
    assert market_model_window(daily) == (252, 63.0)
    assert market_model_window(monthly) == (60, None)
    assert market_model_window(daily, 36.0, math.inf) == (300, None)
    assert market_model_window(daily, 10.0, 21.0) == (230, 21.0)
    assert market_model_window(monthly, None, 6.0) == (60, 6.0)
    names = ("BTSG", "HALPHA", "SERDP", "VLVR")

    def fitted(**kw):
        out, _ = compute_descriptors(daily, market_returns=market, names=names, **kw)
        return out.values

    default = fitted()
    asked = fitted(beta_window_months=252 / 23.0, beta_half_life=63.0)
    assert default.tobytes() == asked.tobytes()
    assert np.isfinite(default[251, :, 0]).all() and np.isnan(default[250, :, 0]).all()
    equal = fitted(beta_window_months=10.0, beta_half_life=math.inf)
    returns = daily.returns()
    t = daily.n_periods - 1
    x = market[t - 229 : t + 1]
    for j in range(daily.n_securities):
        y = returns[t - 229 : t + 1, j]
        design = np.column_stack([np.ones(x.size), x])
        (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
        error = y - design @ np.array([alpha, beta])
        sd = np.sqrt(error @ error / (x.size - 2))
        assert equal[t, j, names.index("HALPHA")] == pytest.approx(alpha, rel=1e-12, abs=1e-15)
        assert equal[t, j, names.index("BTSG")] == pytest.approx(max(beta, 0.0) * sd, rel=1e-12)
    assert not np.allclose(equal[t], default[t], equal_nan=True)


def test_a_negative_beta_sets_btsg_to_zero_as_the_definition_requires():
    rng = np.random.default_rng(1)
    n_periods = 80
    market = rng.normal(scale=0.05, size=n_periods)
    # One security deliberately moves against the market.
    returns = np.column_stack([market * 1.2, -market * 0.9]) + rng.normal(
        scale=0.01, size=(n_periods, 2)
    )
    prices = 50.0 * np.cumprod(1.0 + returns, axis=0)
    panel = SecurityPanel(_dates(n_periods), ("UP", "DOWN"), prices)
    out, _ = compute_descriptors(panel, market_returns=market, names=("BTSG",))
    values = out.values[-1, :, 0]
    assert values[0] > 0.0
    assert values[1] == 0.0


def test_serdp_measures_serial_dependence_and_grows_with_it():
    """Theory for an AR(1) residual: 1 + (4 rho + 2 rho^2) / 3.

    Measured comes in a few percent below, consistently across rho, because Appendix A
    defines SERDP on *fitted* market-model residuals and OLS forces those orthogonal to the
    regressors. That is the definition doing it, not the implementation.
    """
    n_periods, n_secs = 120, 40
    market = np.random.default_rng(3).normal(scale=0.045, size=n_periods)
    measured = []
    for rho in (0.0, 0.2, 0.4):
        values = []
        for seed in range(4):
            rng = np.random.default_rng(50 + seed)
            errors = np.zeros((n_periods, n_secs))
            innovations = rng.normal(scale=0.05, size=(n_periods, n_secs))
            for t in range(1, n_periods):
                errors[t] = rho * errors[t - 1] + innovations[t]
            returns = market[:, None] * 1.0 + errors
            prices = 50.0 * np.cumprod(1.0 + returns, axis=0)
            panel = SecurityPanel(
                _dates(n_periods), tuple(f"S{i}" for i in range(n_secs)), prices
            )
            out, _ = compute_descriptors(
                panel, market_returns=market, names=("SERDP",)
            )
            values.append(float(np.nanmean(out.values[-1, :, 0])))
        measured.append(float(np.mean(values)))
    assert measured[0] == pytest.approx(1.0, abs=0.08)
    assert measured[0] < measured[1] < measured[2]
    for value, rho in zip(measured, (0.0, 0.2, 0.4)):
        theory = 1.0 + (4.0 * rho + 2.0 * rho**2) / 3.0
        assert value == pytest.approx(theory, rel=0.08)


def test_lcapcb_is_the_cube_of_the_standardised_log_cap_not_the_raw_one():
    """Appendix A says "the cube of the *normalized* log of market capitalization", so the
    standardisation is part of the definition rather than a later step."""
    panel, market, _, _ = _panel()
    out, _ = compute_descriptors(panel, market_returns=market)
    log_cap = _column(out, "LNCAP")
    standardised = (log_cap - log_cap.mean()) / log_cap.std(ddof=1)
    assert np.allclose(_column(out, "LCAPCB"), standardised**3)
    assert not np.allclose(_column(out, "LCAPCB"), log_cap**3)


def test_cmra_is_the_log_range_of_the_cumulative_excess_return():
    panel, market, _, _ = _panel()
    risk_free = np.full(panel.n_periods, 0.002)
    out, _ = compute_descriptors(
        panel, market_returns=market, risk_free=risk_free, names=("CMRA",)
    )
    block = np.log1p(panel.returns()[-12:] - 0.002)
    cumulative = np.cumsum(block, axis=0)
    expected = np.log((1.0 + cumulative.max(axis=0)) / (1.0 + cumulative.min(axis=0)))
    assert np.allclose(_column(out, "CMRA"), expected)


# -------------------------------------------------------------- windows and frequency


def test_the_volatility_window_defaults_by_frequency_and_flags_a_thin_one():
    panel, market, beta, residual = _panel()
    truth = np.sqrt(beta**2 * 0.045**2 + residual**2)

    default_out, default_notes = compute_descriptors(
        panel, market_returns=market, names=("DASTD",)
    )
    thin_out, thin_notes = compute_descriptors(
        panel, market_returns=market, names=("DASTD",), volatility_window_months=3.0
    )
    assert "DASTD" not in default_notes
    assert "DASTD" in thin_notes and "65 daily returns" in thin_notes["DASTD"]
    assert np.corrcoef(default_out.values[-1, :, 0], truth)[0, 1] > np.corrcoef(
        thin_out.values[-1, :, 0], truth
    )[0, 1]


def test_periods_for_translates_months_into_the_panels_own_periods():
    monthly, _, _, _ = _panel(n_periods=30)
    assert monthly.periods_for(12.0) == 12
    daily, _, _, _ = _panel(n_periods=30, frequency=Frequency.DAILY)
    assert daily.periods_for(1.0) == 23
    weekly, _, _, _ = _panel(n_periods=30, frequency=Frequency.WEEKLY)
    assert weekly.periods_for(12.0) == 52


# -------------------------------------------------------- what is missing, and why


def test_a_descriptor_that_cannot_be_computed_is_named_with_its_reason():
    panel, market, _, _ = _panel()
    bare = SecurityPanel(panel.dates, panel.securities, panel.prices)
    out, skipped = compute_descriptors(bare, market_returns=market)
    assert "LNCAP" not in out.names
    assert "shares_outstanding" in skipped["LNCAP"]
    assert "HILO" in skipped and "highs/lows" in skipped["HILO"]
    assert "STOM" in skipped


def test_omitting_the_market_names_the_three_descriptors_that_need_it():
    panel, _, _, _ = _panel()
    out, skipped = compute_descriptors(panel)
    for name in ("BTSG", "HALPHA", "SERDP"):
        assert name not in out.names
        assert "market_returns" in skipped[name]


def test_omitting_the_risk_free_rate_is_reported_rather_than_silently_absorbed():
    panel, market, _, _ = _panel()
    out, skipped = compute_descriptors(
        panel, market_returns=market, names=("RSTR", "CMRA")
    )
    assert "RSTR" in out.names
    assert "raw rather than excess returns" in skipped["RSTR"]


def test_an_unknown_descriptor_name_is_refused():
    panel, market, _, _ = _panel()
    with pytest.raises(ValueError, match="unknown descriptor"):
        compute_descriptors(panel, market_returns=market, names=("EBITDA",))


def test_a_market_series_of_the_wrong_length_is_refused():
    panel, _, _, _ = _panel()
    with pytest.raises(ValueError, match="market_returns has"):
        compute_descriptors(panel, market_returns=np.zeros(5))


def test_panel_shape_validation():
    with pytest.raises(ValueError, match="dates x securities"):
        SecurityPanel(_dates(5), ("A", "B"), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="ascending"):
        SecurityPanel(tuple(reversed(_dates(3))), ("A",), np.ones((3, 1)))


def test_market_capitalisation_without_shares_explains_itself():
    panel, _, _, _ = _panel()
    bare = SecurityPanel(panel.dates, panel.securities, panel.prices)
    with pytest.raises(ValueError, match="needs shares_outstanding"):
        bare.market_capitalisation()


# ------------------------------------------------------------- blending into indices


def test_the_risk_index_membership_is_the_published_table():
    assert risk_index_members("Momentum") == ("RSTR", "HALPHA")
    assert "LNCAP" in risk_index_members("Size")
    with pytest.raises(ValueError, match="unknown risk index"):
        risk_index_members("Quality")


def test_blending_refuses_an_unstandardised_panel():
    """Averaging a log market cap of 17 with a turnover of 0.4 is meaningless."""
    panel, market, _, _ = _panel()
    out, _ = compute_descriptors(panel, market_returns=market)
    with pytest.raises(ValueError, match="needs a standardised panel"):
        blend_to_index(out, "Volatility")


def test_blending_equal_weights_by_default():
    panel, market, _, _ = _panel()
    out, _ = compute_descriptors(panel, market_returns=market)
    standardised = out.standardise()
    # standardise=False because this tests the combining arithmetic; the second
    # standardisation is a separate concern with its own tests.
    blended = blend_to_index(standardised, "Momentum", standardise=False)
    columns = [standardised.names.index(n) for n in ("RSTR", "HALPHA")]
    block = standardised.values[:, :, columns]
    present = np.isfinite(block).any(axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanmean(np.where(np.isfinite(block), block, np.nan), axis=2)
    assert np.allclose(blended[present], expected[present])


def test_supplied_weights_are_used_and_normalised():
    panel, market, _, _ = _panel()
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    skewed = blend_to_index(
        standardised, "Momentum", weights={"RSTR": 3.0, "HALPHA": 1.0}, standardise=False
    )
    rstr = standardised.values[:, :, standardised.names.index("RSTR")]
    halpha = standardised.values[:, :, standardised.names.index("HALPHA")]
    both = np.isfinite(rstr) & np.isfinite(halpha)
    assert both.any()
    assert np.allclose(skewed[both], (0.75 * rstr + 0.25 * halpha)[both])


def test_a_partly_available_index_renormalises_over_what_is_there():
    """HALPHA needs sixty periods and RSTR needs twelve, so there is a stretch where only
    one of Momentum's two descriptors exists. An exposure from one descriptor is weaker than
    one from two, but it beats a hole -- and it must not be silently half-scaled."""
    panel, market, _, _ = _panel()
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    blended = blend_to_index(
        standardised, "Momentum", weights={"RSTR": 3.0, "HALPHA": 1.0}, standardise=False
    )
    rstr = standardised.values[:, :, standardised.names.index("RSTR")]
    halpha = standardised.values[:, :, standardised.names.index("HALPHA")]
    only_rstr = np.isfinite(rstr) & ~np.isfinite(halpha)
    assert only_rstr.any()
    assert np.allclose(blended[only_rstr], rstr[only_rstr])
    neither = ~np.isfinite(rstr) & ~np.isfinite(halpha)
    assert np.isnan(blended[neither]).all()


def test_a_blends_composition_is_read_row_by_row_over_the_securities_counted():
    """Which descriptors an index was built from, date by date. Momentum is RSTR alone until
    HALPHA's sixty-period window fills, and Volatility carries SERDP on the last date only --
    the case that must not reach a forecast exposure unseen. Counting only a security whose
    HALPHA is missing sees no HALPHA at all."""
    panel, market, _, _ = _panel()
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    rows = np.arange(panel.n_periods)
    momentum = blend_composition(standardised, "Momentum", rows)
    assert list(momentum) == ["RSTR", "HALPHA"]
    assert np.array_equal(momentum["RSTR"], rows >= 11)
    assert np.array_equal(momentum["HALPHA"], rows >= 59)
    volatility = blend_composition(standardised, "Volatility", rows)
    assert list(volatility) == ["BTSG", "DASTD", "HILO", "LPRI", "CMRA", "SERDP"]
    assert np.array_equal(volatility["SERDP"], rows == rows[-1])
    k = standardised.names.index("HALPHA")
    values = standardised.values.copy()
    values[:, 0, k] = np.nan
    blanked = dataclasses.replace(standardised, values=values)
    alone = np.zeros((rows.size, panel.n_securities), dtype=bool)
    alone[:, 0] = True
    counted = blend_composition(blanked, "Momentum", rows, alone)
    assert np.array_equal(counted["RSTR"], rows >= 11)
    assert not counted["HALPHA"].any()
    assert blend_composition(blanked, "Momentum", rows)["HALPHA"].any()
    with pytest.raises(ValueError, match="rows x securities"):
        blend_composition(standardised, "Momentum", rows, alone[:5])
    with pytest.raises(ValueError, match="rows x securities"):
        blend_composition(standardised, "Momentum", rows, alone.astype(int))


def test_fitted_weights_recover_a_known_blend():
    """The honest version of Barra's proprietary step: estimate the weights from returns.

    Returns are built to depend on a known mix of two descriptors as of the period before,
    and the fit should find that mix -- which is what makes this a reimplementation of the
    method rather than a guess at their table.
    """
    panel, market, _, _ = _panel(n_periods=120, n_secs=60, seed=9)
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    columns = [standardised.names.index(n) for n in ("RSTR", "HALPHA")]
    exposures = standardised.values[:, :, columns]
    rng = np.random.default_rng(9)
    truth = np.array([0.7, 0.3])
    driven = np.full(exposures.shape[:2], np.nan)
    driven[1:] = (
        np.nansum(exposures[:-1] * truth[None, None, :], axis=2) * 0.02
        + rng.normal(scale=0.004, size=(exposures.shape[0] - 1, exposures.shape[1]))
    )
    fitted = fit_index_weights(standardised, "Momentum", driven)
    assert fitted["RSTR"] == pytest.approx(0.7, abs=0.08)
    assert fitted["HALPHA"] == pytest.approx(0.3, abs=0.08)
    assert sum(fitted.values()) == pytest.approx(1.0)


def test_fitting_refuses_returns_of_the_wrong_shape():
    panel, market, _, _ = _panel()
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    with pytest.raises(ValueError, match="returns have shape"):
        fit_index_weights(standardised, "Momentum", np.zeros((5, 5)))


def test_fitted_weights_pair_each_descriptor_with_the_next_periods_return():
    """Descriptor on t, return on t+1, as the model's regressions read their exposures. So the
    first period's return and the last period's descriptors are never read: change either and
    the weights come back bit for bit. The fit is the pooled regression of the one on the
    other, computed here independently; a lag of 0 is the same-day pairing."""
    panel, market, _, _ = _panel(n_periods=120, n_secs=60, seed=9)
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    returns = panel.returns()
    returns[0] = np.random.default_rng(2).normal(scale=0.05, size=panel.n_securities)
    fitted = fit_index_weights(standardised, "Momentum", returns)
    moved = returns.copy()
    moved[0] *= 3.0
    values = standardised.values.copy()
    values[-1] *= -2.0
    later = dataclasses.replace(standardised, values=values)
    assert fit_index_weights(standardised, "Momentum", moved) == fitted
    assert fit_index_weights(later, "Momentum", returns) == fitted
    columns = [standardised.names.index(n) for n in ("RSTR", "HALPHA")]
    for lag in (0, 1):
        x = standardised.values[: panel.n_periods - lag][:, :, columns].reshape(-1, 2)
        y = returns[lag:].reshape(-1)
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        coefficients = np.linalg.lstsq(x[ok], y[ok], rcond=None)[0]
        expected = coefficients / coefficients.sum()
        got = fit_index_weights(standardised, "Momentum", returns, lag=lag)
        assert [got["RSTR"], got["HALPHA"]] == pytest.approx(list(expected), rel=1e-12), lag
    assert fit_index_weights(standardised, "Momentum", returns, lag=0) != fitted
    for bad in (-1, 1.0, True, panel.n_periods):
        with pytest.raises(ValueError, match="lag"):
            fit_index_weights(standardised, "Momentum", returns, lag=bad)


def test_every_published_descriptor_is_computable_from_a_complete_panel():
    """Present is not the same as unqualified.

    SERDP computes, but only at the final period, because keeping the market-model residual
    series for every window would mean storing periods x window x securities values. That is
    a note on a descriptor that exists, not a descriptor that is missing -- and the
    distinction is why `skipped` carries reasons rather than being a list of absentees.
    """
    panel, market, _, _ = _panel()
    out, notes = compute_descriptors(
        panel, market_returns=market, risk_free=np.full(panel.n_periods, 0.002)
    )
    assert set(out.names) == set(PRICE_DESCRIPTORS)
    assert set(notes) <= {"SERDP"}, notes
    assert "final period only" in notes.get("SERDP", "")


def test_a_descriptor_whose_window_exceeds_the_panel_says_so():
    """An all-nan column that quietly disappears leaves the reader unable to tell whether
    the descriptor failed or was never requested. STO5 needs five years."""
    panel, market, _, _ = _panel(n_periods=30)
    out, notes = compute_descriptors(panel, market_returns=market, names=("STO5", "STOM"))
    assert "STO5" not in out.names
    assert "needs 60 period(s) of history and the panel has 30" in notes["STO5"]
    assert "STOM" in out.names


# ------------------------------------------ the second standardisation, and the pipeline


def test_a_blend_of_standardised_descriptors_shrinks_below_unit_scale():
    """Why risk indices are standardised twice rather than once.

    A weighted average of standardised variables has a standard deviation below one unless
    they are perfectly correlated, and the shrinkage grows with how many are blended.
    """
    panel, market, _, _ = _panel()
    standardised = compute_descriptors(
        panel, market_returns=market, risk_free=np.full(panel.n_periods, 0.002)
    )[0].standardise()
    scales = {}
    for index in ("Size", "Momentum", "Volatility"):
        raw = blend_to_index(standardised, index, standardise=False)
        last = raw[-1][np.isfinite(raw[-1])]
        scales[index] = float(last.std(ddof=1))
    # One descriptor cannot shrink; more descriptors shrink more.
    assert scales["Size"] == pytest.approx(1.0, abs=1e-9)
    assert scales["Momentum"] < 0.95
    assert scales["Volatility"] < scales["Momentum"]


def test_the_second_standardisation_puts_every_index_back_on_one_scale():
    """Without it, a one-sigma move would register as 1.00 in Size and 0.47 in Volatility,
    the factor returns would be scaled inversely, and their covariance would mean nothing."""
    panel, market, _, _ = _panel()
    standardised = compute_descriptors(
        panel, market_returns=market, risk_free=np.full(panel.n_periods, 0.002)
    )[0].standardise()
    for index in ("Size", "Momentum", "Volatility", "Trading Activity"):
        blended = blend_to_index(standardised, index)
        last = blended[-1][np.isfinite(blended[-1])]
        assert float(last.std(ddof=1)) == pytest.approx(1.0, abs=1e-9), index


def test_cap_weighting_the_second_pass_puts_the_market_at_zero_exposure():
    panel, market, _, _ = _panel()
    caps = panel.market_capitalisation()
    standardised = compute_descriptors(
        panel, market_returns=market, risk_free=np.full(panel.n_periods, 0.002)
    )[0].standardise(caps)
    blended = blend_to_index(standardised, "Volatility", cap_weights=caps)
    present = np.isfinite(blended[-1])
    weights = caps[-1][present] / caps[-1][present].sum()
    assert abs(float(blended[-1][present] @ weights)) < 1e-10


def test_build_risk_indices_turns_descriptors_into_factors():
    panel, market, _, _ = _panel()
    caps = panel.market_capitalisation()
    descriptors = compute_descriptors(
        panel, market_returns=market, risk_free=np.full(panel.n_periods, 0.002)
    )[0].standardise(caps)
    indices = build_risk_indices(descriptors, cap_weights=caps)
    assert set(indices.names) == {
        "Volatility",
        "Momentum",
        "Size",
        "Size Nonlinearity",
        "Trading Activity",
    }
    assert indices.standardised
    assert indices.securities == panel.securities
    # The descriptors themselves are gone: they are ingredients, not factors.
    assert "RSTR" not in indices.names


def test_build_risk_indices_can_be_restricted_to_a_subset():
    panel, market, _, _ = _panel()
    descriptors = compute_descriptors(panel, market_returns=market)[0].standardise()
    indices = build_risk_indices(descriptors, indices=("Size", "Momentum"))
    assert indices.names == ("Size", "Momentum")


def test_build_risk_indices_refuses_an_unstandardised_panel_or_an_unknown_index():
    panel, market, _, _ = _panel()
    raw = compute_descriptors(panel, market_returns=market)[0]
    with pytest.raises(ValueError, match="needs a standardised descriptor panel"):
        build_risk_indices(raw)
    with pytest.raises(ValueError, match="unknown risk index"):
        build_risk_indices(raw.standardise(), indices=("Quality",))


def test_the_whole_pipeline_reaches_a_factor_model():
    """SecurityPanel -> descriptors -> risk indices -> cross-sectional model, end to end."""
    from praxis.midoffice.model.crosssectional import estimate_cross_sectional_model
    from praxis.midoffice.model.panel import ReturnPanel

    panel, market, beta, residual = _panel(n_periods=120, n_secs=60)
    caps = panel.market_capitalisation()
    descriptors = compute_descriptors(
        panel, market_returns=market, risk_free=np.full(panel.n_periods, 0.002)
    )[0].standardise(caps)
    exposures = build_risk_indices(descriptors, cap_weights=caps)
    model = estimate_cross_sectional_model(
        ReturnPanel(panel.dates, panel.securities, panel.returns(), panel.frequency),
        exposures,
        standardise=False,
        min_securities=20,
        specific_risk="blend",
    )
    assert "Market" in model.factor_names.values()
    assert "Momentum" in model.factor_names.values()
    assert "RSTR" not in model.factor_names.values()
    assert model.notes.n_estimated == 60
    assert 0.0 < model.notes.mean_r_squared < 1.0


def test_lpri_uses_what_the_stock_actually_traded_at():
    """A stock at $400 before a four-for-one split shows as $100 adjusted.

    LPRI's economic content is that low-priced stocks are riskier, so reading the adjusted
    price would record it as cheap in a year when it was not -- and the error is largest
    exactly for the companies that split, which are the ones that went up.
    """
    panel, market, _, _ = _panel(split=4.0)
    out, _ = compute_descriptors(panel, market_returns=market, names=("LPRI",))
    half = len(panel.dates) // 2
    lpri = out.values[:, :, 0]
    # Odd-numbered securities split; before the split their traded price was 4x adjusted.
    assert np.allclose(lpri[half - 1, 1::2], np.log(panel.prices[half - 1, 1::2] * 4.0))
    assert np.allclose(lpri[half - 1, 0::2], np.log(panel.prices[half - 1, 0::2]))


def test_market_capitalisation_pairs_traded_price_with_point_in_time_shares():
    """The pairing Polygon's own market_cap field gets wrong at a split boundary."""
    panel, market, _, _ = _panel(split=4.0)
    half = len(panel.dates) // 2
    cap = panel.market_capitalisation()
    # A split changes neither the capitalisation nor, here, anything else: four times the
    # price on a quarter of the shares is the same company.
    before = cap[half - 1, 1::2]
    naive = panel.prices[half - 1, 1::2] * panel.shares_outstanding[half - 1, 1::2]
    assert np.allclose(before, naive * 4.0)
    assert np.all(before > 0)


def test_a_panel_without_unadjusted_prices_says_which_descriptors_it_affected():
    panel, market, _, _ = _panel()
    adjusted_only = SecurityPanel(
        panel.dates,
        panel.securities,
        panel.prices,
        volumes=panel.volumes,
        shares_outstanding=panel.shares_outstanding,
        frequency=panel.frequency,
    )
    assert not adjusted_only.has_unadjusted
    out, skipped = compute_descriptors(adjusted_only, market_returns=market)
    for name in ("LNCAP", "LPRI", "VLVR"):
        assert name in out.names
        assert "split-adjusted prices" in skipped[name]


# ------------------------------- the estimation universe: whose values set the scale


def _leavers(n_periods: int = 120, n_secs: int = 60, seed: int = 3):
    """The twelve smallest securities leave the index halfway through and trade on."""
    panel, market, _, _ = _panel(n_periods=n_periods, n_secs=n_secs, seed=seed)
    members = np.ones((n_periods, n_secs), dtype=bool)
    members[n_periods // 2 :, :12] = False
    return panel, market, members


def _indices(panel, market, members):
    caps = panel.market_capitalisation()
    descriptors = compute_descriptors(
        panel,
        market_returns=market,
        risk_free=np.full(panel.n_periods, 0.002),
        membership=members,
    )[0].standardise(caps, membership=members)
    return build_risk_indices(descriptors, cap_weights=caps, membership=members), caps


def test_risk_indices_built_over_members_hold_the_index_at_zero_style_exposure():
    """Exact, on every period and every index: the members' cap-weighted mean is zero to
    machine precision and their equal-weighted standard deviation one. That is what makes an
    exposure read as relative to the index."""
    panel, market, members = _leavers()
    indices, caps = _indices(panel, market, members)
    checked = 0
    for t in range(panel.n_periods):
        for k, name in enumerate(indices.names):
            x = indices.values[t, :, k]
            inside = members[t] & np.isfinite(x)
            if inside.sum() < 2:
                continue
            mean = float(np.average(x[inside], weights=caps[t][inside]))
            assert abs(mean) < 1e-14, (t, name, mean)
            assert float(np.std(x[inside], ddof=1)) == pytest.approx(1.0, rel=1e-13), (t, name)
            checked += 1
    assert checked > 2 * panel.n_periods
    # Over everyone, the leavers in the lower tail pull the members off zero.
    everyone, _ = _indices(panel, market, None)
    size = everyone.values[-1, :, everyone.names.index("Size")]
    inside = members[-1] & np.isfinite(size)
    assert abs(float(np.average(size[inside], weights=caps[-1][inside]))) > 0.01


def test_a_security_outside_the_index_sets_no_part_of_any_members_exposure():
    """End to end: shrink the leavers' share counts once they have left -- their caps, LNCAP,
    LCAPCB and turnover all move -- and not one member's risk index moves by a bit. Without a
    membership, every member's does."""
    panel, market, members = _leavers()
    shares = panel.shares_outstanding.copy()
    shares[~members] *= 0.05
    shrunk = dataclasses.replace(panel, shares_outstanding=shares)
    before, _ = _indices(panel, market, members)
    after, _ = _indices(shrunk, market, members)
    assert np.array_equal(before.values[members], after.values[members], equal_nan=True)
    assert not np.array_equal(before.values[~members], after.values[~members], equal_nan=True)
    loose_before, _ = _indices(panel, market, None)
    loose_after, _ = _indices(shrunk, market, None)
    assert not np.array_equal(
        loose_before.values[members], loose_after.values[members], equal_nan=True
    )


def test_lcapcb_is_the_cube_of_log_cap_normalised_over_the_members():
    """The normalisation is part of the descriptor's definition, so it is taken over the
    estimation universe like every other: an ex-member does not set the centre the cube is
    taken about."""
    panel, market, members = _leavers()
    out, _ = compute_descriptors(
        panel, market_returns=market, names=("LNCAP", "LCAPCB"), membership=members
    )
    log_cap = out.values[:, :, 0]
    for t in (0, panel.n_periods - 1):
        inside = members[t]
        centre = float(log_cap[t][inside].mean())
        spread = float(log_cap[t][inside].std(ddof=1))
        expected = ((log_cap[t] - centre) / spread) ** 3
        assert out.values[t, :, 1] == pytest.approx(expected, rel=1e-12)
    loose, _ = compute_descriptors(panel, market_returns=market, names=("LNCAP", "LCAPCB"))
    assert np.array_equal(loose.values[0], out.values[0])
    assert not np.allclose(loose.values[-1, :, 1], out.values[-1, :, 1])


def test_size_nonlinearity_is_built_use4s_way_by_default_and_is_orthogonal_to_size():
    """USE4: the Size exposure cubed; standardised over the members -- cap-weighted mean,
    equal-weighted sd -- and clipped at +/-4; orthogonalised against Size by a fit on Size and
    a constant, weighted by the regression weights over the members; standardised again,
    exactly. Here against that computed independently, an outsider put on the members' scale
    like everyone else. On every date the members' regression-weighted covariance of the two
    is zero to rounding, their cap-weighted mean zero and their equal-weighted sd one."""
    panel, market, members = _leavers()
    caps = panel.market_capitalisation()
    weights = np.sqrt(caps)
    descriptors = compute_descriptors(panel, market_returns=market, membership=members)[
        0
    ].standardise(caps, membership=members)
    indices = build_risk_indices(
        descriptors, cap_weights=caps, membership=members, regression_weights=weights
    )
    size = indices.values[:, :, indices.names.index("Size")]
    snl = indices.values[:, :, indices.names.index("Size Nonlinearity")]
    eps = np.finfo(float).eps
    for t in range(panel.n_periods):
        inside = members[t] & np.isfinite(size[t])
        cube = size[t] ** 3
        z = np.clip(
            (cube - np.average(cube[inside], weights=caps[t][inside]))
            / np.std(cube[inside], ddof=1),
            -4.0,
            4.0,
        )
        w, s = weights[t][inside], size[t][inside]
        s_mean, z_mean = np.average(s, weights=w), np.average(z[inside], weights=w)
        slope = (w @ ((s - s_mean) * (z[inside] - z_mean))) / (w @ (s - s_mean) ** 2)
        e = z - z_mean - slope * (size[t] - s_mean)
        expected = (e - np.average(e[inside], weights=caps[t][inside])) / np.std(
            e[inside], ddof=1
        )
        assert snl[t] == pytest.approx(expected, rel=1e-9, abs=1e-12), t
        x = snl[t][inside]
        product = (s - np.average(s, weights=w)) * (x - np.average(x, weights=w))
        assert abs(w @ product) <= inside.sum() * eps * (w @ np.abs(product)), t
        assert abs(np.average(x, weights=caps[t][inside])) < 1e-14, t
        assert np.std(x, ddof=1) == pytest.approx(1.0, rel=1e-13), t


def test_nonlinear_sizes_clip_is_four_sds_of_the_cube_where_four_mads_would_bind_at_1_2():
    """The raw MAD clip every descriptor gets would bind on a cube at |z| of about 1.2 -- on
    20,000 normal draws the four-MAD bound on z cubed is the cube of 1.22, and it would move
    over a fifth of them -- flattening the tails the cube is there to measure. USE4's clip is
    at +/-4 of the cube standardised: it moves the values past that, and no others."""
    z = np.random.default_rng(5).normal(size=(1, 20_000))
    cube = z**3
    mad = 1.4826 * np.median(np.abs(cube - np.median(cube)))
    assert (4.0 * mad) ** (1.0 / 3.0) == pytest.approx(1.22, abs=0.02)
    assert np.mean(np.abs(cube) > 4.0 * mad) > 0.2
    _, moved = nonlinear_size(z)
    standardised = (cube - np.average(cube[0], weights=np.ones(z.size))) / np.std(
        cube[0], ddof=1
    )
    assert np.array_equal(moved, np.abs(standardised) > 4.0)
    assert 0 < moved.sum() < 0.02 * z.size


def test_size_nonlinearity_built_use3s_way_is_the_lcapcb_blend_as_before():
    """size_nonlinearity="use3" is today's raw cube, bit for bit, and needs LCAPCB; "use4"
    needs Size's descriptors alone, and every other index is the same under both. A blend's
    composition reads Size Nonlinearity as built from whichever it is built from."""
    panel, market, _, _ = _panel()
    caps = panel.market_capitalisation()
    descriptors = compute_descriptors(panel, market_returns=market)[0].standardise(caps)
    use3 = build_risk_indices(descriptors, cap_weights=caps, size_nonlinearity="use3")
    use4 = build_risk_indices(descriptors, cap_weights=caps)
    k = use3.names.index("Size Nonlinearity")
    assert use3.names == use4.names
    old = blend_to_index(descriptors, "Size Nonlinearity", cap_weights=caps)
    assert use3.values[:, :, k].tobytes() == old.tobytes()
    assert not np.allclose(use4.values[:, :, k], old, equal_nan=True)
    rest = [use3.names.index(name) for name in use3.names if name != "Size Nonlinearity"]
    assert use3.values[:, :, rest].tobytes() == use4.values[:, :, rest].tobytes()
    keep = [j for j, name in enumerate(descriptors.names) if name != "LCAPCB"]
    no_cube = dataclasses.replace(
        descriptors,
        names=tuple(descriptors.names[j] for j in keep),
        values=descriptors.values[:, :, keep],
    )
    assert "Size Nonlinearity" in build_risk_indices(no_cube, cap_weights=caps).names
    assert "Size Nonlinearity" not in build_risk_indices(
        no_cube, cap_weights=caps, size_nonlinearity="use3"
    ).names
    assert index_descriptors("Size Nonlinearity") == ("LNCAP",)
    assert index_descriptors("Size Nonlinearity", "use3") == ("LCAPCB",)
    assert index_descriptors("Momentum") == risk_index_members("Momentum")
    rows = np.arange(panel.n_periods)
    assert list(blend_composition(descriptors, "Size Nonlinearity", rows)) == ["LNCAP"]
    assert list(
        blend_composition(descriptors, "Size Nonlinearity", rows, size_nonlinearity="use3")
    ) == ["LCAPCB"]
    for bad in ("use5", None):
        with pytest.raises(ValueError, match="size_nonlinearity"):
            build_risk_indices(descriptors, size_nonlinearity=bad)


def test_fitted_weights_learn_from_the_members_alone():
    """Membership read on the descriptor's date: holding the fit to members is blanking the
    return of every security that was not a member the period before."""
    panel, market, members = _leavers(seed=9)
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    returns = panel.returns()
    fitted = fit_index_weights(standardised, "Momentum", returns, membership=members)
    before = np.zeros_like(members)
    before[1:] = members[:-1]
    blanked = fit_index_weights(standardised, "Momentum", np.where(before, returns, np.nan))
    assert fitted == blanked
    assert fitted != fit_index_weights(
        standardised, "Momentum", np.where(members, returns, np.nan)
    )
    assert fitted != fit_index_weights(standardised, "Momentum", returns)


def test_a_membership_of_the_wrong_shape_or_kind_is_refused_wherever_it_is_taken():
    panel, market, members = _leavers()
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    wrong = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="dates x securities"):
        compute_descriptors(panel, market_returns=market, membership=wrong)
    with pytest.raises(ValueError, match="dates x securities"):
        blend_to_index(standardised, "Size", membership=wrong)
    with pytest.raises(ValueError, match="dates x securities"):
        build_risk_indices(standardised, membership=wrong)
    with pytest.raises(ValueError, match="dates x securities"):
        fit_index_weights(standardised, "Momentum", panel.returns(), membership=wrong)
    with pytest.raises(TypeError, match="boolean"):
        blend_to_index(standardised, "Size", membership=members.astype(float))


# ------------------------------------------------------------- windows with no data in them


def _blanked(panel, rows, column):
    """`panel` with one security's prices, volumes, highs and lows missing on `rows`. The share
    count stays, as the real panel carries it forward from its last sample."""
    out = {}
    for name in ("prices", "unadjusted_prices", "volumes", "highs", "lows"):
        values = getattr(panel, name).copy()
        values[rows, column] = np.nan
        out[name] = values
    return dataclasses.replace(panel, **out)


def _trailing(valid: np.ndarray, window: int) -> np.ndarray:
    """How many of each trailing window's periods are valid, counted here and not by the code
    under test; -1 before the first full window."""
    out = np.full(valid.shape, -1)
    for t in range(window - 1, valid.shape[0]):
        out[t] = valid[t - window + 1 : t + 1].sum(axis=0)
    return out


def test_a_security_whose_prices_start_after_the_window_gets_missing_for_every_one():
    """A sum over an empty window is zero, and RSTR, CMRA and the turnovers used to record it:
    every day before a security's first price read as no momentum and no trading. S00's
    prices start at period 100, so every window ending before then is empty; S01's stop after
    period 50, so at the end of the panel every window is, SERDP's included. The share count
    is there throughout, which is what let the turnovers read zero rather than missing. At any
    coverage rule: 0, and the default."""
    panel, market, _, _ = _panel(n_periods=120)
    late = _blanked(panel, slice(0, 100), 0)
    gone = _blanked(late, slice(51, 120), 1)
    out, _ = compute_descriptors(gone, market_returns=market, min_coverage=0.0)
    default, _ = compute_descriptors(gone, market_returns=market)
    for result in (out, default):
        assert set(WINDOW_DESCRIPTORS) <= set(result.names)
        for name in WINDOW_DESCRIPTORS:
            values = result.values[:, :, result.names.index(name)]
            if name != "SERDP":
                # Every window is complete from period 59, the longest being 60 periods.
                assert np.isnan(values[59:100, 0]).all(), name
                assert np.isfinite(values[99, 2:]).all(), name
            assert np.isnan(values[-1, 1]), name
            assert np.isfinite(values[-1, 2:]).all(), name
    # At 0 a window holding a single observation still gives its value: only the empty ones
    # went. The default asks for half the window.
    rstr = out.values[101, 0, out.names.index("RSTR")]
    assert rstr == np.log1p(gone.returns()[101, 0])
    assert np.isnan(default.values[101, 0, default.names.index("RSTR")])


def test_no_window_descriptor_is_computed_from_an_empty_window_anywhere():
    """Against each descriptor's own inputs, counted independently: a return for DASTD, RSTR
    and CMRA, a return with the market's for the market model, a volume for the turnovers, a
    dollar volume and the market model's window for VLVR. Long gaps, so windows empty and
    refill, and scattered ones besides."""
    panel, market, _, _ = _panel(n_periods=120, n_secs=30, seed=11)
    rng = np.random.default_rng(5)
    gaps = rng.random(panel.prices.shape) < 0.2
    gaps[20:100, :6] = True
    gapped = panel
    for j in range(panel.n_securities):
        gapped = _blanked(gapped, np.flatnonzero(gaps[:, j]), j)
    out, _ = compute_descriptors(gapped, market_returns=market)
    returns = np.isfinite(gapped.returns())
    paired = returns & np.isfinite(market)[:, None]
    traded = np.isfinite(gapped.volumes)
    dollars = np.isfinite(gapped.volumes * gapped.unadjusted_prices)
    inputs = {
        "DASTD": [(returns, 12)],
        "RSTR": [(returns, 12)],
        "CMRA": [(returns, 12)],
        "BTSG": [(paired, 60)],
        "HALPHA": [(paired, 60)],
        "SERDP": [(paired, 60)],
        "STOM": [(traded, 1)],
        "STOQ": [(traded, 3)],
        "STOA": [(traded, 12)],
        "STO5": [(traded, 60)],
        "VLVR": [(dollars, 12), (paired, 60)],
    }
    assert set(inputs) == set(WINDOW_DESCRIPTORS)
    for name, sources in inputs.items():
        values = out.values[:, :, out.names.index(name)]
        empty = np.zeros(values.shape, dtype=bool)
        for valid, window in sources:
            empty |= _trailing(valid, window) == 0
        assert empty.any(), name
        assert np.isnan(values[empty]).all(), name
        assert np.isfinite(values[~empty]).any(), name


def test_min_coverage_removes_thinly_covered_windows_and_alters_no_value():
    """S00 trades in the last seven periods alone: six returns of a twelve-period window, so
    RSTR, CMRA and DASTD sit at a coverage of exactly one half; seven volumes of twelve for
    STOA; full windows for STOQ and STOM; six of sixty for the market model. A rule of 0 keeps
    them all. A rule only ever removes, and what it keeps is 0's value bit for bit. The
    default is the half-window rule."""
    panel, market, _, _ = _panel(n_periods=120)
    thin = _blanked(panel, slice(0, 113), 0)
    anything, _ = compute_descriptors(thin, market_returns=market, min_coverage=0.0)

    def kept(out) -> set[str]:
        return {
            name for name in WINDOW_DESCRIPTORS
            if np.isfinite(out.values[-1, 0, out.names.index(name)])
        }

    assert kept(anything) == set(WINDOW_DESCRIPTORS)
    expected = {
        0.5: {"RSTR", "CMRA", "DASTD", "STOA", "STOQ", "STOM"},
        0.55: {"STOA", "STOQ", "STOM"},
        1.0: {"STOQ", "STOM"},
    }
    for rule, survivors in expected.items():
        out, _ = compute_descriptors(thin, market_returns=market, min_coverage=rule)
        assert kept(out) == survivors, rule
        present = np.isfinite(out.values)
        assert (np.isfinite(anything.values) | ~present).all(), rule
        assert np.array_equal(out.values[present], anything.values[present]), rule
    default, _ = compute_descriptors(thin, market_returns=market)
    half, _ = compute_descriptors(thin, market_returns=market, min_coverage=0.5)
    assert default.values.tobytes() == half.values.tobytes()


def test_min_coverage_is_a_fraction_of_the_window():
    panel, market, _, _ = _panel(n_periods=24, n_secs=5)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="fraction of a window"):
            compute_descriptors(panel, market_returns=market, min_coverage=bad)
