"""Forecasting specific risk from characteristics (Barra EQ 3-12 to 3-14).

Three findings drove the shape of this module and each has a test here:

* the fitted coefficients and kappa recover a known truth, with kappa landing on sqrt(pi/2)
  when specific returns really are normal;
* the linear form of EQ 3-14 runs past its own validity, forecasting *negative* specific risk
  for a security far enough into the tail -- which is why there is a floor and why it names
  what it caught;
* the characteristic model does **not** dominate a plain sample standard deviation. Its error
  is irreducible where the sample estimate's is merely statistical, so the sample estimate
  wins outright given enough history. The precision-weighted blend beats both, and is the only
  one of the three that is good across regimes.
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.model.specificrisk import (
    NORMAL_KAPPA,
    SpecificRiskModel,
    estimate_specific_risk_model,
)


def _world(
    n_secs: int = 200,
    n_periods: int = 120,
    g: float = 0.3,
    seed: int = 0,
    level: float = 0.06,
    regime: float = 1.0,
    explained: float = 1.0,
):
    """A world whose specific volatility is known exactly.

    `g` is kept small enough that `1 + g*z` stays positive: the multiplicative form has no
    meaning for a negative volatility, and a fixture that produced one would be testing the
    estimator against folded absolute values rather than against its own model.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n_secs)
    z = (z - z.mean()) / z.std(ddof=1)
    unobserved = rng.normal(size=n_secs)
    unobserved = (unobserved - unobserved.mean()) / unobserved.std(ddof=1)
    visible = g * np.sqrt(explained)
    hidden = g * np.sqrt(1.0 - explained)
    cross = np.clip(1.0 + visible * z + hidden * unobserved, 0.1, None)
    levels = np.full(n_periods, level)
    if regime != 1.0:
        levels[-(n_periods // 4) :] *= regime
    true = levels[:, None] * cross[None, :]
    assert (true > 0).all()
    returns = rng.normal(size=(n_periods, n_secs)) * true
    characteristics = np.tile(z.reshape(1, -1, 1), (n_periods, 1, 1))
    securities = tuple(f"S{i:03d}" for i in range(n_secs))
    return returns, securities, characteristics, z, true


def _fit(*args, **kw) -> SpecificRiskModel:
    returns, securities, characteristics, _, _ = args[0]
    return estimate_specific_risk_model(
        returns, securities, characteristics, ("Z",), **kw
    )


# --------------------------------------------------------------------------- recovery


def test_gamma_recovers_the_true_sensitivity():
    for g in (0.1, 0.2, 0.3):
        model = _fit(_world(g=g, seed=7))
        assert model.notes.gamma["Z"] == pytest.approx(g, rel=0.06)


def test_a_characteristic_that_carries_nothing_gets_a_coefficient_of_zero():
    model = _fit(_world(g=0.0, seed=7))
    assert abs(model.notes.gamma["Z"]) < 0.02


def test_calibration_recovers_the_normal_constant():
    """kappa converts a mean absolute deviation into a standard deviation.

    For a normal, E|X| = sigma*sqrt(2/pi), so the factor is sqrt(pi/2) = 1.2533. Recovering
    it from simulated normal residuals is the check that the three factors of EQ 3-12 are
    identified the way the module claims rather than absorbing each other's scale.
    """
    for seed in (3, 7, 11):
        model = _fit(_world(g=0.2, seed=seed))
        assert model.kappa_constant == pytest.approx(NORMAL_KAPPA, rel=0.02)


def test_the_forecast_level_tracks_a_shift_in_market_wide_specific_risk():
    """The part no single security's residual history can see."""
    returns, securities, characteristics, _, true = _world(seed=13, regime=3.0)
    model = estimate_specific_risk_model(
        returns, securities, characteristics, ("Z",), average_periods=3
    )
    implied = model.kappa_constant * model.forecast_level
    assert implied == pytest.approx(true[-1].mean(), rel=0.15)
    assert implied > 2.0 * true[0].mean()


def test_the_autoregressive_average_model_also_tracks_the_shift():
    returns, securities, characteristics, _, true = _world(seed=13, regime=3.0)
    model = estimate_specific_risk_model(
        returns, securities, characteristics, ("Z",), average_model="autoregressive"
    )
    implied = model.kappa_constant * model.forecast_level
    assert implied == pytest.approx(true[-1].mean(), rel=0.2)


def test_an_unknown_average_model_is_refused():
    with pytest.raises(ValueError, match="moving' or 'autoregressive"):
        _fit(_world(n_secs=40, n_periods=40), average_model="kalman")


# ----------------------------------------------------- a security with no history at all


def test_a_security_with_no_history_still_gets_a_forecast():
    """The entire point. (1 + V) * S needs characteristics, not a residual series."""
    model = _fit(_world(g=0.3, seed=11))
    forecast = model.forecast({"NEWCO": {"Z": 1.5}})
    base = model.kappa_constant * model.forecast_level
    assert forecast["NEWCO"] == pytest.approx(base * (1.0 + 0.3 * 1.5), rel=0.1)


def test_an_unknown_characteristic_name_contributes_nothing():
    model = _fit(_world(seed=5))
    base = model.forecast({"A": {}})["A"]
    assert model.forecast({"B": {"Unmodelled": 9.0}})["B"] == pytest.approx(base)


# --------------------------------------------------------------------------- the floor


def test_the_linear_form_would_forecast_negative_specific_risk():
    """Not hypothetical: with a coefficient of 0.3, anything past -3.3 sigma goes negative.

    Pinning the defect as well as the fix, because a floor whose necessity is not demonstrated
    tends to get tuned away later.
    """
    model = _fit(_world(g=0.3, seed=3))
    raw = (
        model.kappa_constant
        * (1.0 + model.relative_level({"Z": -5.0}))
        * model.forecast_level
    )
    assert raw < 0.0


def test_the_floor_catches_it_and_names_what_it_caught():
    model = _fit(_world(g=0.3, seed=3))
    exposures = {"DEEP": {"Z": -5.0}, "MILD": {"Z": -1.0}, "AVERAGE": {"Z": 0.0}}
    forecast = model.forecast(exposures)
    base = model.kappa_constant * model.forecast_level
    assert forecast["DEEP"] == pytest.approx(0.1 * base)
    assert forecast["MILD"] > forecast["DEEP"]
    assert model.floored(exposures) == ("DEEP",)


def test_a_floor_of_zero_is_still_available_but_is_not_the_default():
    model = _fit(_world(g=0.3, seed=3))
    assert model.forecast({"DEEP": {"Z": -5.0}}, floor_fraction=0.0)["DEEP"] == 0.0


# ------------------------------------------------------------------------- the blend


def test_the_blend_is_the_pure_model_for_a_security_with_no_history():
    model = _fit(_world(seed=1))
    exposures = {"NEWCO": {"Z": 1.0}}
    assert model.blend(exposures)["NEWCO"] == pytest.approx(
        model.forecast(exposures)["NEWCO"]
    )


def test_the_blend_approaches_the_sample_estimate_given_enough_history():
    model = _fit(_world(seed=1))
    exposures = {"OLDCO": {"Z": 1.0}}
    sample = 0.09
    blended = model.blend(exposures, {"OLDCO": (sample, 100_000)})
    assert blended["OLDCO"] == pytest.approx(sample, rel=0.01)


def test_the_model_weight_falls_monotonically_with_history():
    model = _fit(_world(seed=1))
    weights = [model.model_weight(n) for n in (0, 2, 15, 60, 240, 1000)]
    assert weights[0] == 1.0
    assert all(a >= b for a, b in zip(weights, weights[1:]))
    assert weights[-1] < 0.1


def test_better_characteristics_earn_more_weight():
    """The weight is inverse-variance, so it has to respond to how much the
    characteristics actually explain -- not to a constant someone chose."""
    weak = _fit(_world(seed=4, explained=0.2))
    strong = _fit(_world(seed=4, explained=0.9))
    assert strong.model_weight(20) > weak.model_weight(20)


def test_the_blend_beats_both_of_its_inputs():
    """Measured, not asserted. The characteristic model's error is irreducible and the
    sample estimate's is statistical, so a precision weighting of the two beats either."""
    model_errors, sample_errors, blend_errors = [], [], []
    for seed in range(8):
        returns, securities, characteristics, z, true = _world(
            n_periods=60, seed=300 + seed, explained=0.5, regime=2.0
        )
        model = estimate_specific_risk_model(
            returns, securities, characteristics, ("Z",)
        )
        exposures = {c: {"Z": float(z[j])} for j, c in enumerate(securities)}
        window = 15
        history = {
            c: (float(np.std(returns[-window:, j], ddof=1)), window)
            for j, c in enumerate(securities)
        }
        target = true[-1]

        def rmse(estimates):
            values = np.array([estimates[c] for c in securities])
            return float(np.sqrt(np.mean((values - target) ** 2)) / target.mean())

        model_errors.append(rmse(model.forecast(exposures)))
        sample_errors.append(rmse({c: history[c][0] for c in securities}))
        blend_errors.append(rmse(model.blend(exposures, history)))

    assert np.mean(blend_errors) < np.mean(model_errors)
    assert np.mean(blend_errors) < np.mean(sample_errors)


def test_the_blend_survives_a_regime_shift_that_a_long_sample_does_not():
    """A long trailing standard deviation is the best of the three when the level is stable
    and the worst when it moves, because it is anchored to a level that no longer holds."""
    blend_errors, long_sample_errors = [], []
    for seed in range(8):
        returns, securities, characteristics, z, true = _world(
            n_periods=60, seed=400 + seed, explained=0.5, regime=3.0
        )
        model = estimate_specific_risk_model(
            returns, securities, characteristics, ("Z",)
        )
        exposures = {c: {"Z": float(z[j])} for j, c in enumerate(securities)}
        history = {
            c: (float(np.std(returns[-15:, j], ddof=1)), 15)
            for j, c in enumerate(securities)
        }
        target = true[-1]

        def rmse(values):
            return float(np.sqrt(np.mean((values - target) ** 2)) / target.mean())

        blend_errors.append(
            rmse(np.array([model.blend(exposures, history)[c] for c in securities]))
        )
        long_sample_errors.append(
            rmse(
                np.array(
                    [float(np.std(returns[:, j], ddof=1)) for j in range(len(securities))]
                )
            )
        )
    assert np.mean(blend_errors) < 0.5 * np.mean(long_sample_errors)


def test_a_zero_or_single_observation_history_is_ignored_rather_than_trusted():
    model = _fit(_world(seed=1))
    exposures = {"A": {"Z": 0.5}}
    pure = model.forecast(exposures)["A"]
    assert model.blend(exposures, {"A": (0.5, 1)})["A"] == pytest.approx(pure)
    assert model.blend(exposures, {"A": (0.0, 500)})["A"] == pytest.approx(pure)


# ----------------------------------------------------------------- kappa by capitalisation


def test_capitalisation_calibrates_kappa_into_piecewise_knots():
    returns, securities, characteristics, _, _ = _world(n_secs=200, seed=17)
    rng = np.random.default_rng(17)
    caps = np.tile(rng.lognormal(mean=10.0, sigma=2.0, size=200), (returns.shape[0], 1))
    model = estimate_specific_risk_model(
        returns, securities, characteristics, ("Z",), capitalisations=caps
    )
    assert model.notes.kappa_calibrated
    assert len(model.kappa_knots) == 10
    assert all(np.isfinite(v) and v > 0 for _, v in model.kappa_knots)


def test_kappa_interpolates_between_knots_and_is_flat_outside_them():
    model = SpecificRiskModel(
        names=("Z",),
        gamma=np.array([0.0]),
        forecast_level=0.05,
        kappa_knots=((0.25, 1.0), (0.75, 2.0)),
    )
    assert model.kappa(0.5) == pytest.approx(1.5)
    assert model.kappa(0.0) == pytest.approx(1.0)
    assert model.kappa(1.0) == pytest.approx(2.0)
    assert model.kappa(None) == pytest.approx(NORMAL_KAPPA)


def test_without_capitalisations_kappa_falls_back_and_says_so():
    model = _fit(_world(seed=17))
    assert not model.notes.kappa_calibrated
    assert model.kappa_knots == ()
    assert any("falls back to the normal constant" in q for q in model.notes.qualifications())


def test_a_cell_without_a_residual_takes_no_part_in_the_kappa_calibration():
    """kappa compares realised volatility with the predicted level, and the two have to be
    taken over the same cells. A characteristic row or a capitalisation in a period with no
    residual -- a period the security was not in the regression -- says nothing about how
    wrong the model was there. Before this was fixed both moved the calibration."""
    returns, securities, characteristics, _, _ = _world(n_secs=200, seed=17)
    rng = np.random.default_rng(17)
    caps = np.tile(rng.lognormal(mean=10.0, sigma=2.0, size=200), (returns.shape[0], 1))
    residuals = returns.copy()
    residuals[:60, :100] = np.nan
    wild_z, blank_z = characteristics.copy(), characteristics.copy()
    wild_z[:60, :100, :] = 3.0
    blank_z[:60, :100, :] = np.nan
    wild_caps, blank_caps = caps.copy(), caps.copy()
    wild_caps[:60, :100] = 1e15
    blank_caps[:60, :100] = np.nan
    wild = estimate_specific_risk_model(
        residuals, securities, wild_z, ("Z",), capitalisations=wild_caps
    )
    blank = estimate_specific_risk_model(
        residuals, securities, blank_z, ("Z",), capitalisations=blank_caps
    )
    assert wild.kappa_constant == blank.kappa_constant
    assert wild.kappa_knots == blank.kappa_knots


def test_capitalisations_of_the_wrong_shape_are_refused():
    world = _world(n_secs=40, n_periods=40)
    with pytest.raises(ValueError, match="capitalisations have shape"):
        _fit(world, capitalisations=np.ones((40, 9)))


# ------------------------------------------------------------------- shape and reporting


def test_mismatched_characteristics_are_refused():
    returns, securities, _, _, _ = _world(n_secs=40, n_periods=40)
    with pytest.raises(ValueError, match="share periods and securities"):
        estimate_specific_risk_model(
            returns, securities, np.zeros((40, 9, 1)), ("Z",)
        )


def test_a_name_count_mismatch_is_refused():
    returns, securities, characteristics, _, _ = _world(n_secs=40, n_periods=40)
    with pytest.raises(ValueError, match="characteristic column"):
        estimate_specific_risk_model(returns, securities, characteristics, ("Z", "Y"))


def test_an_entirely_missing_residual_panel_is_refused():
    returns, securities, characteristics, _, _ = _world(n_secs=40, n_periods=40)
    with pytest.raises(ValueError, match="no level to forecast"):
        estimate_specific_risk_model(
            np.full_like(returns, np.nan), securities, characteristics, ("Z",)
        )


def test_a_short_panel_is_reported_as_a_qualification():
    model = _fit(_world(n_secs=60, n_periods=8))
    assert any("fewer than the" in q for q in model.notes.qualifications())


def test_characteristics_that_explain_nothing_are_reported():
    model = _fit(_world(g=0.0, seed=21))
    assert any("barely better than one average" in q for q in model.notes.qualifications())


def test_the_residual_dispersion_measures_what_the_characteristics_miss():
    strong = _fit(_world(seed=23, explained=1.0, g=0.3))
    weak = _fit(_world(seed=23, explained=0.2, g=0.3))
    assert weak.relative_residual_sd > strong.relative_residual_sd


def test_annualisation_scales_the_level_and_therefore_the_forecast():
    world = _world(seed=29)
    monthly = _fit(world)
    annual = _fit(world, annualisation=float(np.sqrt(12.0)))
    assert annual.forecast_level == pytest.approx(monthly.forecast_level * np.sqrt(12.0))
    assert annual.forecast({"A": {"Z": 0.0}})["A"] == pytest.approx(
        monthly.forecast({"A": {"Z": 0.0}})["A"] * np.sqrt(12.0)
    )


def test_describe_names_the_level_and_the_kappa_convention():
    model = _fit(_world(seed=31))
    text = model.describe()
    assert "specific risk model on 1 characteristic" in text
    assert "normal constant" in text
