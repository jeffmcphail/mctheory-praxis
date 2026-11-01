"""Tests for the factor model builder.

The estimator is verified statistically rather than against a stored draw. A single
realisation tells you nothing: on one seed the recovered betas sat about two standard errors
from truth and the factor volatilities came out 7 to 12 percent low, which is indistinguishable
by eye from a real bias. `test_betas_are_unbiased` and `test_factor_variances_are_unbiased`
run a Monte Carlo and check the mean estimate against the theoretical standard error of the
mean, which is a property rather than an artifact.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from praxis.midoffice.model.panel import (
    FactorReturns,
    Frequency,
    ReturnPanel,
    align,
)
from praxis.midoffice.model.timeseries import (
    MIN_OBSERVATIONS,
    FactorModel,
    estimate_timeseries_model,
)

N = 120
DATES = tuple(date(2015 + i // 12, i % 12 + 1, 28) for i in range(N))
TRUE_VAR = np.array([0.0018, 0.0009, 0.0006])
TRUE_BETA = np.array([1.20, -0.30, 0.10])
RESID_SD = 0.03


def draw(seed: int, n=N, dates=DATES, resid_sd=RESID_SD, beta=TRUE_BETA):
    rng = np.random.default_rng(seed)
    f = rng.multivariate_normal(np.zeros(3), np.diag(TRUE_VAR), size=n)
    factors = FactorReturns(dates[:n], ("MKT", "SMB", "HML"), f, Frequency.MONTHLY)
    y = f @ np.asarray(beta) + rng.normal(0, resid_sd, n)
    panel = ReturnPanel.of(dates[:n], {"S": y}, Frequency.MONTHLY)
    return panel, factors


# --- the panel ------------------------------------------------------------------------


def test_a_panel_keeps_missing_as_missing():
    """Treating absence as a zero return biases volatility down and beta toward zero."""
    panel = ReturnPanel.of(DATES[:4], {"A": [0.01, np.nan, 0.02, np.nan]})
    assert panel.observations("A") == 2
    assert panel.coverage() == {"A": 2}


def test_unsorted_dates_are_refused():
    with pytest.raises(ValueError, match="ascending"):
        ReturnPanel(dates=(DATES[1], DATES[0]), securities=("A",), returns=np.zeros((2, 1)))


def test_duplicate_dates_and_securities_are_refused():
    with pytest.raises(ValueError, match="duplicate dates"):
        ReturnPanel(dates=(DATES[0], DATES[0]), securities=("A",), returns=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="duplicate securities"):
        ReturnPanel(dates=DATES[:2], securities=("A", "A"), returns=np.zeros((2, 2)))


def test_a_short_series_is_refused_rather_than_padded_silently():
    with pytest.raises(ValueError, match="pad with nan"):
        ReturnPanel.of(DATES[:4], {"A": [0.01, 0.02]})


def test_a_gap_in_a_factor_series_is_refused():
    """One security's gap can be skipped; a factor's gap breaks every security's estimate."""
    with pytest.raises(ValueError, match="missing factor return"):
        FactorReturns.of(DATES[:3], {"MKT": [0.01, np.nan, 0.02]})


# --- annualisation --------------------------------------------------------------------


@pytest.mark.parametrize(
    "frequency,periods", [(Frequency.DAILY, 252.0), (Frequency.MONTHLY, 12.0)]
)
def test_variance_scales_linearly_and_volatility_by_its_square_root(frequency, periods):
    """The commonest annualisation error inflates a monthly 4% into 48% rather than 13.9%."""
    assert frequency.variance_scale == periods
    assert frequency.volatility_scale == pytest.approx(np.sqrt(periods))
    assert 0.04 * Frequency.MONTHLY.volatility_scale == pytest.approx(0.1386, abs=1e-4)


def test_annualising_scales_the_covariance_by_periods_and_specific_risk_by_its_root():
    panel, factors = draw(3)
    monthly = estimate_timeseries_model(panel, factors, annualise=False)
    annual = estimate_timeseries_model(panel, factors, annualise=True)
    scale = Frequency.MONTHLY.variance_scale
    assert annual.covariance.matrix == pytest.approx(monthly.covariance.matrix * scale)
    assert annual.exposures["S"].specific.risk() == pytest.approx(
        monthly.exposures["S"].specific.risk() * np.sqrt(scale)
    )


def test_an_un_annualised_model_says_so_loudly():
    """The engine's tail measures assume annual inputs; a monthly model understates by 3.5x."""
    panel, factors = draw(3)
    monthly = estimate_timeseries_model(panel, factors, annualise=False)
    assert any("NOT been" in q and "annualised" in q for q in monthly.qualifications())
    assert estimate_timeseries_model(panel, factors).notes.is_clean


def test_exposures_are_unaffected_by_annualising():
    """A beta is a ratio of returns, so it carries no time units."""
    panel, factors = draw(5)
    monthly = estimate_timeseries_model(panel, factors, annualise=False)
    annual = estimate_timeseries_model(panel, factors, annualise=True)
    for fid in annual.factor_ids:
        assert annual.exposures["S"].exposures[fid] == pytest.approx(
            monthly.exposures["S"].exposures[fid]
        )


# --- the estimator, verified statistically --------------------------------------------


def test_betas_are_unbiased():
    """Mean estimate against the theoretical standard error of the mean, over 200 draws.

    A single draw cannot distinguish a real bias from sampling noise: on seed 7 two of three
    betas sat about two standard errors out, which looks exactly like a broken estimator.
    """
    estimates = []
    for seed in range(200):
        panel, factors = draw(seed)
        model = estimate_timeseries_model(panel, factors, annualise=False)
        estimates.append([model.exposures["S"].exposures[f] for f in model.factor_ids])
    estimates = np.asarray(estimates)
    for i in range(3):
        mean = estimates[:, i].mean()
        se_of_mean = estimates[:, i].std(ddof=1) / np.sqrt(len(estimates))
        z = (mean - TRUE_BETA[i]) / se_of_mean
        assert abs(z) < 3.0, f"factor {i}: mean {mean:.4f} vs true {TRUE_BETA[i]}, z={z:.2f}"


def test_the_spread_of_the_betas_matches_theory():
    """sigma_resid / (sigma_factor * sqrt(n)). If this is wrong the regression is wrong."""
    estimates = []
    for seed in range(200):
        panel, factors = draw(seed)
        model = estimate_timeseries_model(panel, factors, annualise=False)
        estimates.append([model.exposures["S"].exposures[f] for f in model.factor_ids])
    estimates = np.asarray(estimates)
    theoretical = RESID_SD / (np.sqrt(TRUE_VAR) * np.sqrt(N))
    for i in range(3):
        observed = estimates[:, i].std(ddof=1)
        assert observed == pytest.approx(theoretical[i], rel=0.25), (
            f"factor {i}: spread {observed:.4f} vs theory {theoretical[i]:.4f}"
        )


def test_factor_variances_are_unbiased():
    variances = []
    for seed in range(200):
        panel, factors = draw(seed)
        model = estimate_timeseries_model(panel, factors, annualise=False)
        variances.append(np.diag(model.covariance.matrix))
    variances = np.asarray(variances)
    for i in range(3):
        mean = variances[:, i].mean()
        se_of_mean = variances[:, i].std(ddof=1) / np.sqrt(len(variances))
        assert abs((mean - TRUE_VAR[i]) / se_of_mean) < 3.0


def test_specific_risk_recovers_the_residual_volatility():
    specifics = [
        estimate_timeseries_model(*draw(seed), annualise=False)
        .exposures["S"]
        .specific.risk()
        for seed in range(200)
    ]
    assert np.mean(specifics) == pytest.approx(RESID_SD, rel=0.02)


def test_r_squared_is_between_zero_and_one():
    panel, factors = draw(11)
    model = estimate_timeseries_model(panel, factors)
    assert 0.0 <= model.r_squared["S"] <= 1.0


# --- the intercept --------------------------------------------------------------------


def test_the_intercept_is_estimated_but_is_not_an_exposure():
    """A factor risk model has nothing to say about alpha, but omitting it biases the slopes."""
    panel, factors = draw(13)
    model = estimate_timeseries_model(panel, factors)
    assert set(model.exposures["S"].exposures) == set(model.factor_ids)
    assert "S" in model.alphas


def slopes(model):
    return np.array([model.exposures["S"].exposures[f] for f in model.factor_ids])


def fit_both(drift, seed=2):
    rng = np.random.default_rng(seed)
    f = rng.multivariate_normal(np.zeros(3), np.diag(TRUE_VAR), size=N)
    factors = FactorReturns(DATES, ("MKT", "SMB", "HML"), f, Frequency.MONTHLY)
    y = f @ TRUE_BETA + drift + rng.normal(0, RESID_SD, N)
    panel = ReturnPanel.of(DATES, {"S": y}, Frequency.MONTHLY)
    return (
        estimate_timeseries_model(panel, factors),
        estimate_timeseries_model(panel, factors, fit_intercept=False),
    )


def test_the_intercept_only_matters_when_there_is_drift_to_absorb():
    """With no drift the two fits agree; with drift the slopes have to absorb it.

    That is the whole argument for fitting it by default. The first draft of this test
    asserted on the market beta at a drift of 0.05, where it moves only 3.3% -- because the
    bias lands in proportion to `mean_f / E[f^2]`, so it concentrates on the
    SMALLEST-variance factor, not the largest. At the same drift HML moves 55%.
    """
    with_zero, without_zero = fit_both(0.0)
    with_drift, without_drift = fit_both(0.20)
    zero_shift = np.max(np.abs(slopes(without_zero) - slopes(with_zero)))
    drift_shift = np.max(np.abs(slopes(without_drift) - slopes(with_drift)))

    # Proportionality rather than two separate thresholds. At zero drift the slopes do not
    # agree *exactly*: the in-sample factor means are not exactly zero, so there is still a
    # little drift to absorb. Asserting rel=0.01 there failed on HML for that reason, which
    # is the in-sample mean rather than a defect.
    assert drift_shift > 20 * zero_shift, f"{drift_shift:.4f} vs {zero_shift:.4f}"
    assert zero_shift < 0.01
    assert drift_shift > 0.1
    assert with_drift.alphas["S"] == pytest.approx(0.20, abs=0.02)


def test_the_bias_lands_hardest_on_the_lowest_variance_factor():
    """And can flip its sign, which is why it is not a cosmetic choice.

    HML has the smallest factor variance here, so the drift it has to absorb per unit of
    slope is largest. At a drift of 0.20 its estimated exposure goes from +0.09 to -0.13.
    """
    with_drift, without_drift = fit_both(0.20)
    hml = len(TRUE_VAR) - 1  # the smallest-variance factor
    assert np.sign(slopes(with_drift)[hml]) != np.sign(slopes(without_drift)[hml])
    # And the market beta, with the largest variance, moves proportionally far less.
    shifts = np.abs(slopes(without_drift) - slopes(with_drift))
    assert shifts[hml] > shifts[0]


def test_without_an_intercept_no_alpha_is_reported():
    _, without = fit_both(0.05)
    assert without.alphas == {}


# --- what the estimation could not do -------------------------------------------------


def test_a_short_history_security_is_excluded_and_flagged():
    rng = np.random.default_rng(1)
    f = rng.multivariate_normal(np.zeros(3), np.diag(TRUE_VAR), size=N)
    factors = FactorReturns(DATES, ("MKT", "SMB", "HML"), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(
        DATES,
        {
            "LONG": f @ TRUE_BETA + rng.normal(0, RESID_SD, N),
            "SHORT": [np.nan] * (N - 10) + list(rng.normal(0, 0.04, 10)),
        },
        Frequency.MONTHLY,
    )
    model = estimate_timeseries_model(panel, factors)
    assert "SHORT" not in model.exposures
    assert model.notes.short_history == ("SHORT",)
    assert any("uncovered to the engine" in q for q in model.qualifications())


def test_observations_equal_to_parameters_is_excluded():
    """Four observations against three factors and an intercept: the fit is exact.

    The residual then has zero degrees of freedom, so the specific risk would come out as
    exactly zero for arithmetic reasons rather than because the factors explain the security.
    Five observations against the same four parameters leaves one degree of freedom, which is
    terrible but defined, so it is estimated -- the guard is about undefined, not about
    unwise, and `min_observations` is the knob for unwise.
    """
    rng = np.random.default_rng(1)
    short_dates = DATES[:4]
    f = rng.multivariate_normal(np.zeros(3), np.diag(TRUE_VAR), size=4)
    factors = FactorReturns(short_dates, ("MKT", "SMB", "HML"), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(short_dates, {"S": rng.normal(0, 0.03, 4)}, Frequency.MONTHLY)
    model = estimate_timeseries_model(panel, factors, min_observations=4)
    assert model.notes.degenerate == ("S",)
    assert model.exposures == {}
    assert any("degrees of freedom" in q for q in model.qualifications())


def test_one_degree_of_freedom_is_estimated_not_excluded():
    """Defined but worthless. min_observations is what keeps it out in practice."""
    rng = np.random.default_rng(1)
    dates = DATES[:5]
    f = rng.multivariate_normal(np.zeros(3), np.diag(TRUE_VAR), size=5)
    factors = FactorReturns(dates, ("MKT", "SMB", "HML"), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(dates, {"S": rng.normal(0, 0.03, 5)}, Frequency.MONTHLY)
    assert "S" in estimate_timeseries_model(panel, factors, min_observations=4).exposures
    assert estimate_timeseries_model(panel, factors).notes.short_history == ("S",)


def test_a_rank_deficient_covariance_is_flagged_where_it_is_created():
    """This module is where singular covariances come from, so it says so.

    docs/CONDITIONING.md calibrated psd_relative_floor against exactly this construction.
    """
    dates = DATES[:3]
    rng = np.random.default_rng(1)
    f = rng.multivariate_normal(np.zeros(3), np.diag(TRUE_VAR), size=3)
    factors = FactorReturns(dates, ("MKT", "SMB", "HML"), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(dates, {"S": rng.normal(0, 0.03, 3)}, Frequency.MONTHLY)
    model = estimate_timeseries_model(panel, factors, min_observations=2)
    assert model.notes.covariance_is_rank_deficient
    notes = " ".join(model.qualifications())
    assert "singular by construction" in notes
    assert "psd_relative_floor" in notes


def test_a_thin_but_not_singular_covariance_is_flagged_differently():
    dates = DATES[:8]
    rng = np.random.default_rng(1)
    f = rng.multivariate_normal(np.zeros(3), np.diag(TRUE_VAR), size=8)
    factors = FactorReturns(dates, ("MKT", "SMB", "HML"), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(dates, {"S": rng.normal(0, 0.03, 8)}, Frequency.MONTHLY)
    model = estimate_timeseries_model(panel, factors, min_observations=5)
    assert not model.notes.covariance_is_rank_deficient
    assert model.notes.periods_per_factor < 3.0
    assert any("per factor" in q for q in model.qualifications())


def test_a_healthy_model_is_clean():
    panel, factors = draw(17)
    assert estimate_timeseries_model(panel, factors).notes.is_clean


# --- alignment ------------------------------------------------------------------------


def test_alignment_intersects_and_reports_what_each_side_lost():
    rng = np.random.default_rng(1)
    factors = FactorReturns(
        DATES[:60], ("MKT",), rng.normal(0, 0.04, (60, 1)), Frequency.MONTHLY
    )
    panel = ReturnPanel.of(DATES[30:], {"S": rng.normal(0, 0.05, N - 30)}, Frequency.MONTHLY)
    alignment = align(panel, factors)
    assert len(alignment.dates) == 30
    assert not alignment.is_exact
    assert len(alignment.dropped_from_panel) == N - 60
    assert len(alignment.dropped_from_factors) == 30
    assert any("dropped from the estimation" in q for q in alignment.qualifications())


def test_mixing_frequencies_is_refused():
    rng = np.random.default_rng(1)
    factors = FactorReturns(DATES, ("MKT",), rng.normal(0, 0.04, (N, 1)), Frequency.MONTHLY)
    panel = ReturnPanel.of(DATES, {"S": rng.normal(0, 0.05, N)}, Frequency.DAILY)
    with pytest.raises(ValueError, match="annualise one of them wrong"):
        align(panel, factors)


def test_no_shared_dates_is_an_error_not_an_empty_model():
    rng = np.random.default_rng(1)
    factors = FactorReturns(
        DATES[:40], ("MKT",), rng.normal(0, 0.04, (40, 1)), Frequency.MONTHLY
    )
    panel = ReturnPanel.of(DATES[60:], {"S": rng.normal(0, 0.05, 60)}, Frequency.MONTHLY)
    with pytest.raises(ValueError, match="share no dates"):
        estimate_timeseries_model(panel, factors)


# --- the contract with the engine -----------------------------------------------------


def test_a_built_model_feeds_compute_risk():
    """The whole point: the builder's output is exactly the pair the engine consumes."""
    from praxis.midoffice.asof import CALENDARS, AsOf, DateCalendar
    from praxis.midoffice.engine import compute_risk
    from praxis.midoffice.value.market import (
        AssetClass,
        Holding,
        MarketData,
        Quote,
        value_book,
    )

    rng = np.random.default_rng(23)
    f = rng.multivariate_normal(np.zeros(3), np.diag(TRUE_VAR), size=N)
    factors = FactorReturns(DATES, ("MKT", "SMB", "HML"), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(
        DATES,
        {
            "AAPL": f @ np.array([1.1, -0.2, 0.1]) + rng.normal(0, 0.03, N),
            "MSFT": f @ np.array([0.9, 0.3, -0.1]) + rng.normal(0, 0.035, N),
        },
        Frequency.MONTHLY,
    )
    model = estimate_timeseries_model(panel, factors)

    md = MarketData.of(
        [
            Quote("AAPL", 190.0, currency="USD", price_multiplier=1.0),
            Quote("MSFT", 410.0, currency="USD", price_multiplier=1.0),
        ]
    )
    book, load = value_book(
        "FUND",
        [
            Holding("AAPL", AssetClass.EQUITY, quantity=1000.0, benchmark_quantity=900.0),
            Holding("MSFT", AssetClass.EQUITY, quantity=500.0, benchmark_quantity=600.0),
        ],
        md,
        benchmark_name="INDEX",
    )
    calendars = {n: DateCalendar.of(n, ["20150630"]) for n in CALENDARS}
    result = compute_risk(
        AsOf.resolve(calendars),
        book,
        model.exposures,
        model.covariance,
        factor_names=dict(model.factor_names),
        load=load,
    )
    # Annualised equity risk on plausible betas: a few percent to a few tens of percent.
    assert 0.01 < result.total_risk < 1.0
    assert result.tracking_error > 0.0
    assert result.is_exact, result.qualifications()
