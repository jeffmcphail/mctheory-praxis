"""Rebasing exposures to [0, 1] without moving a risk number.

The whole claim is an invariance, so that is what gets tested: the same portfolio, the same
common-factor risk, both bases. Everything else here guards the two ways the invariance
breaks -- a non-affine transform, or a model with no market factor to absorb the shift --
and the failure mode it exists to prevent, which is unitized exposures meeting a
standard-basis covariance and overstating risk roughly forty-fold with nothing to signal it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from praxis.midoffice.model.characteristics import CharacteristicPanel
from praxis.midoffice.model.crosssectional import estimate_cross_sectional_model
from praxis.midoffice.model.exposure_basis import (
    DEFAULT_BOUND,
    STANDARD,
    UNIT,
    BasisError,
    describe_basis,
    to_standard,
    to_unit,
)
from praxis.midoffice.model.panel import Frequency, ReturnPanel


def _model(n_secs: int = 80, n_periods: int = 180, seed: int = 5, fit_market: bool = True):
    rng = np.random.default_rng(seed)
    dates = tuple(date(2000 + t // 12, t % 12 + 1, 28) for t in range(n_periods))
    securities = tuple(f"S{i:03d}" for i in range(n_secs))
    chars = CharacteristicPanel(
        dates, securities, ("Size", "Value", "Momentum"),
        rng.normal(size=(n_periods, n_secs, 3)), Frequency.MONTHLY,
    ).standardise()
    design = np.concatenate([np.ones((n_periods, n_secs, 1)), chars.values], axis=2)
    factors = rng.normal(size=(n_periods, 4)) * np.array([0.04, 0.02, 0.015, 0.01])
    returns = np.einsum("tsk,tk->ts", design, factors) + rng.normal(
        size=(n_periods, n_secs)
    ) * 0.05
    panel = ReturnPanel(dates, securities, returns, Frequency.MONTHLY)
    return estimate_cross_sectional_model(
        panel, chars, standardise=False, min_securities=20, fit_market=fit_market
    )


def _common_risk(model, weights: dict[str, float]) -> float:
    """A portfolio's common-factor risk: the quantity that must not move."""
    exposure = np.zeros(len(model.factor_ids))
    for code, weight in weights.items():
        vector = model.exposures[code]
        for i, fid in enumerate(model.factor_ids):
            exposure[i] += weight * vector.exposures.get(fid, 0.0)
    return float(np.sqrt(exposure @ model.covariance.matrix @ exposure))


def _weights(model, seed: int = 3) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    codes = list(model.exposures)
    raw = rng.dirichlet(np.ones(len(codes)) * 3.0)
    return dict(zip(codes, raw))


# ------------------------------------------------------------------- the invariance


def test_common_factor_risk_is_identical_in_both_bases():
    """The claim, measured. If this moves, the rebasing is wrong and everything that
    depends on it is wrong with it."""
    model = _model()
    weights = _weights(model)
    assert _common_risk(to_unit(model), weights) == pytest.approx(
        _common_risk(model, weights), rel=1e-12
    )


def test_the_invariance_holds_for_many_different_portfolios():
    model = _model()
    unit = to_unit(model)
    for seed in range(12):
        weights = _weights(model, seed=seed)
        assert _common_risk(unit, weights) == pytest.approx(
            _common_risk(model, weights), rel=1e-10
        ), seed


def test_a_long_short_portfolio_is_also_preserved():
    """The hard case: net weights near zero, where the shift term dominates if the
    market factor is not absorbing it correctly."""
    model = _model()
    codes = list(model.exposures)
    half = len(codes) // 2
    weights = {c: (1.0 / half if i < half else -1.0 / half)
               for i, c in enumerate(codes[: 2 * half])}
    assert _common_risk(to_unit(model), weights) == pytest.approx(
        _common_risk(model, weights), rel=1e-9
    )


def test_the_round_trip_returns_the_original_model():
    model = _model()
    back = to_standard(to_unit(model))
    assert back.basis == STANDARD
    assert np.allclose(back.covariance.matrix, model.covariance.matrix)
    for code, vector in model.exposures.items():
        for fid in model.factor_ids:
            assert back.exposures[code].exposures.get(fid, 0.0) == pytest.approx(
                vector.exposures.get(fid, 0.0), abs=1e-12
            )


# -------------------------------------------------------------- what the user sees


def test_style_exposures_land_inside_the_unit_interval():
    model = _model()
    unit = to_unit(model)
    market = unit.factor_ids[0]
    values = [
        v
        for vector in unit.exposures.values()
        for fid, v in vector.exposures.items()
        if fid != market
    ]
    assert min(values) >= 0.0
    assert max(values) <= 1.0


def test_the_market_average_sits_at_a_half():
    """What makes it readable: 0.5 is the market, above is a tilt, below is a tilt the
    other way. A standardised zero maps there exactly."""
    model = _model()
    unit = to_unit(model)
    market = unit.factor_ids[0]
    style = unit.factor_ids[1]
    for code, vector in model.exposures.items():
        if abs(vector.exposures.get(style, 0.0)) < 1e-9:
            assert unit.exposures[code].exposures[style] == pytest.approx(0.5)
    assert all(
        v.exposures[market] == 1.0 for v in unit.exposures.values()
    ), "the market exposure must stay at one"


def test_specific_risk_is_untouched():
    """It is per security, not per factor, so no basis change can reach it."""
    model = _model()
    unit = to_unit(model)
    for code in model.exposures:
        assert (
            unit.exposures[code].specific.by_issuer[code]
            == model.exposures[code].specific.by_issuer[code]
        )


def test_the_basis_tag_moves_with_the_model():
    model = _model()
    assert model.basis == STANDARD
    assert to_unit(model).basis == UNIT
    assert "0.5 being the market average" in describe_basis(to_unit(model))
    assert "standard deviations" in describe_basis(model)


# ------------------------------------------------------------------ what it refuses


def test_a_model_without_a_market_factor_is_refused():
    """There is nowhere for the shift to go, so the fitted returns would change. Refusing
    beats rebasing something whose industry exposures might span the constant instead --
    that needs a different transform, not a guess."""
    model = _model(fit_market=False)
    with pytest.raises(BasisError, match="nothing to absorb the shift"):
        to_unit(model)


def test_rebasing_twice_is_refused_in_both_directions():
    model = _model()
    with pytest.raises(BasisError, match="already on the standard basis"):
        to_standard(model)
    with pytest.raises(BasisError, match="already on the unit basis"):
        to_unit(to_unit(model))


def test_a_non_positive_bound_is_refused():
    with pytest.raises(BasisError, match="bound must be positive"):
        to_unit(_model(), bound=0.0)


# ------------------------------------------------- the failure this exists to prevent


def test_mixing_unitized_exposures_with_a_standard_covariance_understates_active_risk():
    """Why the transform moves a whole model rather than exposing a function on exposures.

    The direction matters more than the magnitude. On a long-only book the error is about
    +7%, because the market exposure is 1.0 in both bases and its unchanged contribution
    masks the rest. Net the market out -- which is what an active risk calculation does,
    and what this kind of model is mostly used for -- and the 0.5 shift multiplies by a
    weight sum of zero and cancels, leaving exposures at exactly `scale` times the correct
    ones. Active risk then comes back **eight times too small**.

    Overstating risk is an annoyance. Understating it by eight lets someone take eight
    times the position they intended, and nothing in either half looks wrong on its own.
    """
    model = _model()
    unit = to_unit(model)
    codes = list(model.exposures)
    half = len(codes) // 2
    active = {
        c: (1.0 / half if i < half else -1.0 / half)
        for i, c in enumerate(codes[: 2 * half])
    }

    def mismatched(weights):
        exposure = np.zeros(len(model.factor_ids))
        for code, weight in weights.items():
            for i, fid in enumerate(model.factor_ids):
                exposure[i] += weight * unit.exposures[code].exposures.get(fid, 0.0)
        return float(np.sqrt(exposure @ model.covariance.matrix @ exposure))

    correct = _common_risk(model, active)
    wrong = mismatched(active)
    assert correct > 0.0
    # Exactly the scale factor: 1 / (2 * bound) = 1/8.
    assert wrong / correct == pytest.approx(1.0 / (2.0 * DEFAULT_BOUND), rel=0.02)
    assert wrong < 0.2 * correct

    # On a long-only book the same mistake is small and in the other direction, which is
    # precisely what makes it hard to catch by eye.
    long_only = _weights(model)
    ratio = mismatched(long_only) / _common_risk(model, long_only)
    assert 1.0 < ratio < 1.3

    # The tag is what makes either case detectable rather than silent.
    assert unit.basis != model.basis


def test_a_zero_exposure_survives_the_round_trip_rather_than_being_dropped():
    """Standardised zero becomes 0.5, and a sparsity filter that discarded it would read
    it back as zero -- which is a different company."""
    model = _model()
    unit = to_unit(model)
    style = unit.factor_ids[1]
    assert all(style in v.exposures for v in unit.exposures.values())


def test_the_default_bound_matches_the_winsorising_bound():
    """A different bound here would put exposures outside [0, 1], defeating the point."""
    from praxis.midoffice.model.characteristics import CharacteristicPanel as _CP
    import inspect

    signature = inspect.signature(_CP.standardise)
    assert signature.parameters["winsorise"].default == DEFAULT_BOUND
