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

import warnings
from datetime import date

import numpy as np
import pytest

from praxis.midoffice.model.descriptors import (
    PRICE_DESCRIPTORS,
    SecurityPanel,
    blend_to_index,
    build_risk_indices,
    compute_descriptors,
    fit_index_weights,
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


def test_btsg_recovers_beta_times_residual_volatility():
    panel, market, beta, residual = _panel()
    out, _ = compute_descriptors(panel, market_returns=market, names=("BTSG",))
    measured = _column(out, "BTSG")
    truth = beta * residual
    assert np.corrcoef(measured, truth)[0, 1] > 0.9
    assert float(np.mean(measured / truth)) == pytest.approx(1.0, abs=0.1)


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


def test_fitted_weights_recover_a_known_blend():
    """The honest version of Barra's proprietary step: estimate the weights from returns.

    Returns are built to depend on a known mix of two descriptors, and the fit should find
    that mix -- which is what makes this a reimplementation of the method rather than a guess
    at their table.
    """
    panel, market, _, _ = _panel(n_periods=120, n_secs=60, seed=9)
    standardised = compute_descriptors(panel, market_returns=market)[0].standardise()
    columns = [standardised.names.index(n) for n in ("RSTR", "HALPHA")]
    exposures = standardised.values[:, :, columns]
    rng = np.random.default_rng(9)
    truth = np.array([0.7, 0.3])
    driven = (
        np.nansum(exposures * truth[None, None, :], axis=2) * 0.02
        + rng.normal(scale=0.004, size=exposures.shape[:2])
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
