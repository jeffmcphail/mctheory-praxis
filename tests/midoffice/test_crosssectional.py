"""Cross-sectional (Barra-style) model estimation, and characteristics as exposures.

The estimator is checked against synthetic worlds whose factor returns and specific risks are
known by construction, because that is the only way to tell recovery from plausibility. Two of
the tests here exist because a Monte Carlo found a real bias the eye would not have: the
degrees-of-freedom understatement of specific risk, and the estimation-noise inflation of the
factor covariance diagonal.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pytest

from praxis.midoffice.model.characteristics import CharacteristicPanel
from praxis.midoffice.model.covariance import ShrunkCovariance, SampleCovariance
from praxis.midoffice.model.crosssectional import (
    MARKET_FACTOR,
    MIN_SECURITIES_PER_PERIOD,
    estimate_cross_sectional_model,
    regression_design,
    regression_universe,
)
from praxis.midoffice.model.panel import Frequency, ReturnPanel


def _dates(n: int) -> tuple[date, ...]:
    return tuple(date(2000 + t // 12, t % 12 + 1, 28) for t in range(n))


def _world(
    n_secs: int = 120,
    n_periods: int = 240,
    seed: int = 7,
    spec_vol: float = 0.06,
    n_char: int = 3,
    frequency: Frequency = Frequency.MONTHLY,
):
    """A world whose factor returns and specific risk are known exactly.

    Returns the panel, the already-standardised characteristics the returns were built from,
    the true factor returns, and the true specific volatility.
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_periods)
    secs = tuple(f"S{i:04d}" for i in range(n_secs))
    names = tuple(f"C{i}" for i in range(n_char))
    standardised = CharacteristicPanel(
        dates, secs, names, rng.normal(size=(n_periods, n_secs, n_char)), frequency
    ).standardise()
    true_vol = np.linspace(0.04, 0.010, n_char + 1)
    f_true = rng.normal(size=(n_periods, n_char + 1)) * true_vol
    design = np.concatenate(
        [np.ones((n_periods, n_secs, 1)), standardised.values], axis=2
    )
    returns = np.einsum("tsk,tk->ts", design, f_true) + rng.normal(
        size=(n_periods, n_secs)
    ) * spec_vol
    return (
        ReturnPanel(dates, secs, returns, frequency),
        standardised,
        f_true,
        spec_vol,
    )


def _fit(returns, chars, **kw):
    kw.setdefault("standardise", False)
    kw.setdefault("annualise", False)
    kw.setdefault("min_securities", 10)
    # These worlds build day t's return from day t's characteristics, so they are read
    # same-day. A model on characteristics computed from prices never is: see the lag tests.
    kw.setdefault("lag", 0)
    return estimate_cross_sectional_model(returns, chars, **kw)


# --------------------------------------------------------------------------- panel structure


def test_a_panel_refuses_values_of_the_wrong_shape():
    with pytest.raises(ValueError, match="dates x securities x characteristics"):
        CharacteristicPanel(_dates(3), ("A", "B"), ("Size",), np.zeros((3, 2, 2)))


def test_a_panel_refuses_duplicate_securities():
    with pytest.raises(ValueError, match="duplicate securities"):
        CharacteristicPanel(_dates(2), ("A", "A"), ("Size",), np.zeros((2, 2, 1)))


def test_a_panel_refuses_duplicate_characteristic_names():
    with pytest.raises(ValueError, match="duplicate characteristics"):
        CharacteristicPanel(_dates(2), ("A",), ("Size", "Size"), np.zeros((2, 1, 2)))


def test_a_panel_refuses_descending_dates():
    backwards = tuple(reversed(_dates(3)))
    with pytest.raises(ValueError, match="ascending"):
        CharacteristicPanel(backwards, ("A",), ("Size",), np.zeros((3, 1, 1)))


def test_constant_holds_one_snapshot_across_every_date():
    panel = CharacteristicPanel.constant(
        _dates(4), {"A": {"Size": 1.0, "Value": 2.0}, "B": {"Size": 3.0, "Value": 4.0}}
    )
    assert panel.values.shape == (4, 2, 2)
    assert np.array_equal(panel.at(0), panel.at(3))
    assert panel.snapshot()["B"]["Value"] == 4.0


def test_constant_leaves_an_absent_characteristic_missing():
    panel = CharacteristicPanel.constant(
        _dates(2), {"A": {"Size": 1.0, "Value": 2.0}, "B": {"Size": 3.0}}
    )
    assert "Value" not in panel.snapshot()["B"]
    assert np.isnan(panel.values[0, 1, 1])


def test_a_snapshot_defaults_to_the_most_recent_period():
    values = np.zeros((3, 1, 1))
    values[:, 0, 0] = [10.0, 20.0, 30.0]
    panel = CharacteristicPanel(_dates(3), ("A",), ("Size",), values)
    assert panel.snapshot()["A"]["Size"] == 30.0
    assert panel.snapshot(0)["A"]["Size"] == 10.0


def test_a_snapshot_omits_a_security_with_nothing_at_all():
    values = np.full((2, 2, 1), np.nan)
    values[:, 0, 0] = 1.0
    panel = CharacteristicPanel(_dates(2), ("A", "B"), ("Size",), values)
    assert set(panel.snapshot()) == {"A"}


def test_coverage_counts_only_complete_characteristic_rows():
    values = np.ones((4, 2, 2))
    values[1, 1, 0] = np.nan
    values[2, 1, 1] = np.nan
    panel = CharacteristicPanel(_dates(4), ("A", "B"), ("Size", "Value"), values)
    assert panel.coverage() == {"A": 4, "B": 2}


# --------------------------------------------------------------------- standardisation


def test_standardising_gives_zero_mean_and_unit_dispersion():
    rng = np.random.default_rng(0)
    panel = CharacteristicPanel(
        _dates(3), tuple(f"S{i}" for i in range(50)), ("Size",),
        rng.normal(loc=17.0, scale=4.0, size=(3, 50, 1)),
    )
    out = panel.standardise()
    column = out.at(0)[:, 0]
    assert abs(float(np.mean(column))) < 1e-12
    assert float(np.std(column, ddof=1)) == pytest.approx(1.0, abs=0.05)
    assert out.standardised is True


def test_a_cap_weighted_mean_puts_the_market_at_zero_style_exposure():
    """The reason Barra weights the mean, and the reason this is not cosmetic.

    Equal-weighting leaves the market portfolio with a large exposure to the very factor the
    standardisation was supposed to centre, so every portfolio's tilt is measured against a
    benchmark nobody holds.
    """
    cap = np.array([10.0, 12.0, 14.0, 18.0, 22.0, 25.0])
    values = np.tile(cap.reshape(1, -1, 1), (2, 1, 1))
    secs = tuple(f"S{i}" for i in range(6))
    panel = CharacteristicPanel(_dates(2), secs, ("Size",), values)
    weights = np.tile(np.exp(cap), (2, 1))
    market = weights[0] / weights[0].sum()

    cap_weighted = panel.standardise(weights).at(0)[:, 0]
    equal_weighted = panel.standardise().at(0)[:, 0]

    assert abs(float(market @ cap_weighted)) < 1e-12
    assert float(market @ equal_weighted) > 1.0


def test_the_dispersion_stays_equal_weighted_even_when_the_mean_is_not():
    """A cap-weighted standard deviation would let a few names set the factor's scale."""
    cap = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    values = np.tile(cap.reshape(1, -1, 1), (1, 1, 1))
    panel = CharacteristicPanel(_dates(1), tuple(f"S{i}" for i in range(6)), ("X",), values)
    heavy = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1e6]])
    out = panel.standardise(heavy, winsorise=None).at(0)[:, 0]
    # Equal-weighted dispersion of the raw values is unchanged by the weighting, so the
    # spacing between adjacent securities stays one raw unit over that dispersion.
    spacing = float(np.diff(out)[0])
    assert spacing == pytest.approx(1.0 / float(np.std(cap, ddof=1)), rel=1e-12)


def test_winsorising_clips_the_raw_value_so_the_clip_actually_bites():
    """The defect this module was rewritten to fix.

    Clipping standardised values cannot catch an outlier, because by then the outlier has set
    the scale it would be measured against: a book-to-price of twenty inflates the standard
    deviation to about eight, standardises to +0.1, and sails through any sane clip while
    crushing every ordinary company to about -2.3.
    """
    raw = np.array([0.3, 0.6, 20.0, 1.2, 1.5, 1.8])
    values = raw.reshape(1, -1, 1)
    panel = CharacteristicPanel(_dates(1), tuple(f"S{i}" for i in range(6)), ("Value",), values)

    clipped = panel.standardise(winsorise=4.0).at(0)[:, 0]
    unclipped = panel.standardise(winsorise=None).at(0)[:, 0]

    ordinary = [0, 1, 3, 4, 5]
    assert np.ptp(clipped[ordinary]) > 3.0 * np.ptp(unclipped[ordinary])


def test_the_outlier_sets_the_scale_which_is_why_order_matters():
    """Pins the mechanism, not just the symptom: the raw standard deviation is inflated."""
    raw = np.array([0.3, 0.6, 20.0, 1.2, 1.5, 1.8])
    centre = float(np.median(raw))
    mad_scale = float(np.median(np.abs(raw - centre))) * 1.4826
    assert float(np.std(raw, ddof=1)) > 5.0 * mad_scale
    # And the outlier, measured against the inflated scale, is not even two sigma out.
    assert (raw.max() - raw.mean()) / np.std(raw, ddof=1) < 2.5


def test_the_winsorised_count_reports_how_many_raw_values_moved():
    raw = np.array([0.3, 0.6, 20.0, 1.2, 1.5, 1.8])
    values = np.tile(raw.reshape(1, -1, 1), (4, 1, 1))
    panel = CharacteristicPanel(_dates(4), tuple(f"S{i}" for i in range(6)), ("Value",), values)
    assert panel.standardise(winsorise=4.0).winsorised == 4
    assert panel.standardise(winsorise=None).winsorised == 0


def test_no_standardised_exposure_exceeds_the_stated_bound():
    rng = np.random.default_rng(5)
    raw = rng.standard_cauchy(size=(3, 60, 2)) * 3.0
    panel = CharacteristicPanel(_dates(3), tuple(f"S{i}" for i in range(60)), ("A", "B"), raw)
    out = panel.standardise(winsorise=3.0)
    assert float(np.nanmax(np.abs(out.values))) <= 3.0 + 1e-12


def test_a_characteristic_with_no_variation_standardises_to_zero():
    panel = CharacteristicPanel(_dates(2), ("A", "B", "C"), ("Flat",), np.full((2, 3, 1), 7.0))
    assert np.array_equal(panel.standardise().values, np.zeros((2, 3, 1)))


def test_a_zero_mad_falls_back_to_the_standard_deviation():
    """More than half the securities sharing one value makes the MAD zero."""
    raw = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 40.0])
    values = raw.reshape(1, -1, 1)
    panel = CharacteristicPanel(_dates(1), tuple(f"S{i}" for i in range(6)), ("X",), values)
    out = panel.standardise(winsorise=1.0).at(0)[:, 0]
    assert np.isfinite(out).all()
    assert float(np.nanmax(np.abs(out))) <= 1.0 + 1e-12


def test_standardising_preserves_what_was_missing():
    values = np.ones((2, 4, 1))
    values[0, 2, 0] = np.nan
    panel = CharacteristicPanel(_dates(2), ("A", "B", "C", "D"), ("X",), values)
    out = panel.standardise()
    assert np.isnan(out.values[0, 2, 0])
    assert not np.isnan(out.values[1, 2, 0])


def test_standardise_refuses_weights_of_the_wrong_shape():
    panel = CharacteristicPanel(_dates(3), ("A", "B"), ("X",), np.ones((3, 2, 1)))
    with pytest.raises(ValueError, match="dates x securities"):
        panel.standardise(np.ones((3, 5)))


# -------------------------------------------------------------- recovery of a known truth


def test_the_estimator_recovers_the_factor_returns_it_was_built_from():
    returns, chars, f_true, _ = _world(n_secs=200, n_periods=240, seed=7)
    model = _fit(returns, chars)
    observed = np.sqrt(np.array(model.notes.observed_variance))
    truth = np.std(f_true, axis=0, ddof=1)
    # Biased slightly high by estimation noise, which the next tests are about.
    assert np.all(observed > 0.9 * truth)
    assert np.all(observed < 1.2 * truth)


def test_the_market_factor_is_an_exposure_of_one_for_every_security():
    """Where the cross-sectional model inverts the time-series one.

    A time-series intercept is alpha and is deliberately not an exposure. The unit column here
    is the market factor and every security really does load 1.0 on it.
    """
    returns, chars, _, _ = _world(n_secs=60, n_periods=60)
    model = _fit(returns, chars)
    market_id = model.factor_ids[0]
    assert model.factor_names[market_id] == MARKET_FACTOR
    assert all(ev.exposures[market_id] == 1.0 for ev in model.exposures.values())


def test_omitting_the_market_factor_leaves_only_the_characteristics():
    returns, chars, _, _ = _world(n_secs=60, n_periods=60, n_char=3)
    model = _fit(returns, chars, fit_market=False)
    assert MARKET_FACTOR not in model.factor_names.values()
    assert len(model.factor_ids) == 3


def test_specific_risk_is_unbiased_on_a_narrow_cross_section():
    """The degrees-of-freedom correction, where it matters.

    Twenty-five securities on ten factors: the fit absorbs ten of twenty-five observations
    every period, so uncorrected residuals understate specific volatility by
    sqrt(15/25) = 0.775 -- a 22 percent understatement of the term that dominates a
    concentrated portfolio's risk.
    """
    returns, chars, _, spec_vol = _world(n_secs=25, n_periods=600, seed=3, n_char=9)
    model = _fit(returns, chars, min_securities=12)
    estimated = np.mean(
        [ev.specific.by_issuer[code] for code, ev in model.exposures.items()]
    )
    uncorrected = spec_vol * np.sqrt((25 - 10) / 25)
    assert estimated == pytest.approx(spec_vol, rel=0.03)
    assert uncorrected < 0.8 * spec_vol  # the bias the correction removes is material


def test_specific_risk_is_unbiased_on_a_wide_cross_section_too():
    returns, chars, _, spec_vol = _world(n_secs=200, n_periods=400, seed=11)
    model = _fit(returns, chars)
    estimated = np.mean(
        [ev.specific.by_issuer[code] for code, ev in model.exposures.items()]
    )
    assert estimated == pytest.approx(spec_vol, rel=0.02)


# ------------------------------------------------------------ the estimation-noise diagnostic


def test_noise_share_measures_the_bias_it_claims_to():
    """The factor covariance diagonal is inflated by the sampling variance of its own
    estimation, and `sampling_variance` predicts that inflation.

    Averaged over replications, (observed - realized) / predicted should be one. A single
    replication cannot show this: over 240 periods the sample variance of the true factor
    series has a relative standard error of about 9 percent, which swamps the bias.
    """
    ratios = []
    for seed in range(12):
        returns, chars, f_true, _ = _world(n_secs=60, n_periods=240, seed=500 + seed)
        model = _fit(returns, chars)
        realized = np.var(f_true, axis=0, ddof=1)
        observed = np.array(model.notes.observed_variance)
        predicted = np.array(model.notes.sampling_variance)
        ratios.append(float(np.mean((observed - realized) / predicted)))
    mean = float(np.mean(ratios))
    stderr = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
    assert abs(mean - 1.0) < max(3.0 * stderr, 0.25)


def test_noise_share_falls_as_the_cross_section_widens():
    shares = []
    for n_secs in (40, 100, 400):
        returns, chars, _, _ = _world(n_secs=n_secs, n_periods=180, seed=17)
        model = _fit(returns, chars)
        shares.append(float(np.mean(model.notes.noise_share)))
    assert shares[0] > shares[1] > shares[2]
    assert shares[2] < 0.5 * shares[0]


def test_noise_concentrates_on_the_lowest_variance_factor():
    """Sampling variance is roughly constant across standardised factors, so it is a larger
    fraction of a small factor's variance -- the same structure as the time-series intercept
    bias landing hardest on the lowest-variance factor."""
    returns, chars, _, _ = _world(n_secs=80, n_periods=240, seed=23, n_char=3)
    model = _fit(returns, chars)
    shares = model.notes.noise_share
    variances = model.notes.observed_variance
    assert variances[0] == max(variances)  # market, built with the largest vol
    assert shares[0] == min(shares)
    assert shares[-1] == max(shares)


def test_a_loud_noise_share_is_reported_as_a_qualification():
    returns, chars, _, _ = _world(n_secs=15, n_periods=120, seed=29, n_char=3)
    model = _fit(returns, chars, min_securities=12)
    assert any("sampling error of its own estimation" in q for q in model.qualifications())


def test_the_noise_share_of_an_unestimated_covariance_is_empty():
    returns, chars, _, _ = _world(n_secs=40, n_periods=40)
    model = _fit(returns, chars)
    assert len(model.notes.noise_share) == len(model.factor_ids)
    assert all(np.isfinite(model.notes.noise_share))


# ------------------------------------------------------------------------ what it refuses


def test_a_frequency_mismatch_is_refused():
    returns, chars, _, _ = _world(n_secs=40, n_periods=40)
    weekly = CharacteristicPanel(
        chars.dates, chars.securities, chars.names, chars.values, Frequency.WEEKLY
    )
    with pytest.raises(ValueError, match="align them before estimating"):
        _fit(returns, weekly)


def test_no_shared_dates_is_refused():
    returns, chars, _, _ = _world(n_secs=40, n_periods=40)
    shifted = CharacteristicPanel(
        tuple(date(d.year + 50, d.month, d.day) for d in chars.dates),
        chars.securities, chars.names, chars.values, chars.frequency,
    )
    with pytest.raises(ValueError, match="share no dates"):
        _fit(returns, shifted)


def test_no_shared_securities_is_refused():
    returns, chars, _, _ = _world(n_secs=40, n_periods=40)
    renamed = CharacteristicPanel(
        chars.dates, tuple(f"X{s}" for s in chars.securities),
        chars.names, chars.values, chars.frequency,
    )
    with pytest.raises(ValueError, match="share no securities"):
        _fit(returns, renamed)


def test_every_period_too_thin_is_refused_with_advice():
    returns, chars, _, _ = _world(n_secs=12, n_periods=60)
    with pytest.raises(ValueError, match="lower min_securities"):
        estimate_cross_sectional_model(
            returns, chars, standardise=False, min_securities=MIN_SECURITIES_PER_PERIOD
        )


def test_regression_weights_of_the_wrong_shape_are_refused():
    returns, chars, _, _ = _world(n_secs=40, n_periods=40)
    with pytest.raises(ValueError, match="to match the return panel"):
        _fit(returns, chars, regression_weights=np.ones((40, 9)))


# --------------------------------------------------------------------- what it reports


def test_a_thin_period_is_dropped_and_named():
    returns, chars, _, _ = _world(n_secs=40, n_periods=60, seed=31)
    blanked = returns.returns.copy()
    blanked[5, 5:] = np.nan  # five usable securities that period
    thinned = ReturnPanel(returns.dates, returns.securities, blanked, returns.frequency)
    model = _fit(thinned, chars, min_securities=20)
    assert chars.dates[5] in model.notes.thin_periods
    assert model.notes.n_periods == 59
    assert any("dropped for having fewer than" in q for q in model.qualifications())


def test_a_collinear_design_is_flagged_rather_than_hidden():
    """Industry dummies alongside a market factor: the classic rank deficiency.

    Three mutually exclusive industry indicators sum to the unit column, so the design has no
    unique solution. lstsq returns the minimum-norm one, which is a solution without an
    economic reading, and the notes say so rather than letting it pass.
    """
    n_secs, n_periods = 60, 36
    rng = np.random.default_rng(41)
    dates, secs = _dates(n_periods), tuple(f"S{i:03d}" for i in range(n_secs))
    industry = np.zeros((n_periods, n_secs, 3))
    for j in range(n_secs):
        industry[:, j, j % 3] = 1.0
    chars = CharacteristicPanel(dates, secs, ("Tech", "Fin", "Energy"), industry)
    returns = ReturnPanel(
        dates, secs, rng.normal(scale=0.05, size=(n_periods, n_secs)), Frequency.MONTHLY
    )
    model = _fit(returns, chars, standardise=False, fit_market=True)
    assert len(model.notes.rank_deficient_periods) == n_periods
    assert any("minimum-norm solution" in q for q in model.qualifications())
    # There is a market factor, so the usual-cause hint may mention it.
    assert any("alongside a market factor" in q for q in model.qualifications())


class _Capture:
    """A covariance estimator that keeps the factor returns it was handed."""

    def __init__(self):
        self.returns = None

    def estimate(self, returns):
        self.returns = np.array(returns, dtype=float, copy=True)
        return SampleCovariance().estimate(returns)


def _industry_world(seed, n_secs=60, n_periods=36):
    """Three mutually exclusive industries of twenty and one style, with no market factor
    needed: the indicators span the unit column themselves."""
    rng = np.random.default_rng(seed)
    dates, secs = _dates(n_periods), tuple(f"S{i:03d}" for i in range(n_secs))
    values = np.zeros((n_periods, n_secs, 4))
    for j in range(n_secs):
        values[:, j, j % 3] = 1.0
    values[:, :, 3] = rng.normal(size=(n_periods, n_secs))
    chars = CharacteristicPanel(dates, secs, ("Ind0", "Ind1", "Ind2", "Size"), values)
    return chars, rng.normal(scale=0.05, size=(n_periods, n_secs))


def test_a_factor_nobody_carries_in_a_period_is_left_out_of_that_period():
    """An industry with every member missing on one day is an all-zero column that day.

    Left in, it makes the design rank deficient for a reason unrelated to every other column,
    and the period takes a minimum-norm solution. Left out, the period is an ordinary
    full-rank fit and the factor's return that day is recorded as exactly zero. The counts
    are exact, not directions: the real run's 75 collinear periods were all this.
    """
    chars, raw = _industry_world(seed=61)
    members = [j for j in range(60) if j % 3 == 2]
    raw[np.ix_([4, 9, 17], members)] = np.nan
    returns = ReturnPanel(chars.dates, chars.securities, raw, Frequency.MONTHLY)
    capture = _Capture()
    model = _fit(returns, chars, fit_market=False, covariance_estimator=capture)
    notes = model.notes
    assert notes.rank_deficient_periods == ()
    assert notes.empty_periods == (0, 0, 3, 0)
    assert notes.present_periods == (36, 36, 33, 36)
    assert (capture.returns[[4, 9, 17], 2] == 0.0).all()
    assert any(
        q.startswith("3 factor-period(s) had no security in the regression") and "Ind2 (3)" in q
        for q in model.qualifications()
    )


def test_a_factor_carried_by_one_security_is_counted_where_noise_share_cannot_see_it():
    """One member: the regression fits it exactly and its residual is zero. noise_share takes
    its sampling variance from everyone else's residuals, so with a common component in the
    factor return it reads the factor as well estimated -- a real run's Other Industries was
    one stock in all 939 periods and showed 5.7 percent. The single-member count does not
    depend on any of that."""
    rng = np.random.default_rng(67)
    n_secs, n_periods = 60, 36
    dates, secs = _dates(n_periods), tuple(f"S{i:03d}" for i in range(n_secs))
    values = np.zeros((n_periods, n_secs, 3))
    values[:, 0, 2] = 1.0                                   # Lone: one member
    for j in range(1, n_secs):
        values[:, j, j % 2] = 1.0
    chars = CharacteristicPanel(dates, secs, ("Ind0", "Ind1", "Lone"), values)
    raw = rng.normal(scale=0.05, size=(n_periods, n_secs))
    raw += rng.normal(scale=0.2, size=(n_periods, 1))       # what the industries carry
    returns = ReturnPanel(dates, secs, raw, Frequency.MONTHLY)
    model = _fit(returns, chars, fit_market=False)
    notes = model.notes
    assert notes.noise_share[2] < 0.25                      # looks well estimated
    assert notes.single_member_periods == (0, 0, 36)        # and is one stock every period
    assert notes.present_periods == (36, 36, 36)
    assert notes.empty_periods == (0, 0, 0)
    assert any(
        "exactly one security exposed" in q and "Lone (36 of 36)" in q
        for q in model.qualifications()
    )


def test_the_collinearity_note_names_a_constant_column_and_no_absent_market_factor():
    """A style column identical across the cross-section duplicates whatever spans the unit
    column -- here the industry dummies -- and is named, because that was checked. Fitted
    without a market factor, the note must not blame one."""
    chars, raw = _industry_world(seed=71)
    values = chars.values.copy()
    values[5, :, 3] = 0.7                                   # Size constant, in one period
    flat = CharacteristicPanel(chars.dates, chars.securities, chars.names, values)
    returns = ReturnPanel(chars.dates, chars.securities, raw, Frequency.MONTHLY)
    model = _fit(returns, flat, fit_market=False)
    assert model.notes.rank_deficient_periods == (chars.dates[5],)
    assert model.notes.collinear_constant == (0, 0, 0, 1)
    note = next(q for q in model.qualifications() if "minimum-norm solution" in q)
    assert "Size (1)" in note
    assert "market factor" not in note


def test_a_collinearity_with_no_identifiable_cause_does_not_invent_one():
    chars, raw = _industry_world(seed=73)
    values = np.concatenate([chars.values, 2.0 * chars.values[:, :, 3:]], axis=2)
    doubled = CharacteristicPanel(
        chars.dates, chars.securities, chars.names + ("Size2",), values
    )
    returns = ReturnPanel(chars.dates, chars.securities, raw, Frequency.MONTHLY)
    model = _fit(returns, doubled, fit_market=False)
    assert len(model.notes.rank_deficient_periods) == 36
    note = next(q for q in model.qualifications() if "minimum-norm solution" in q)
    assert "market factor" not in note
    assert "no empty or constant column accounts for it" in note


def test_the_regression_universe_is_known_before_estimating_and_agrees_with_it():
    """The set an industry scheme must count over: securities in at least one estimated
    period's regression, by the estimator's own test, before anything is estimated."""
    returns, chars, _, _ = _world(n_secs=40, n_periods=60, seed=79)
    raw = returns.returns.copy()
    raw[:, 0] = np.nan                  # never a return
    raw[:10, 2:] = np.nan               # the first ten periods are thin...
    raw[10:, 1] = np.nan                # ...and the only ones security 1 has a return in
    weights = np.ones((60, 40))
    weights[:, 3] = 0.0                 # never any weight
    panel = ReturnPanel(returns.dates, returns.securities, raw, returns.frequency)
    universe = regression_universe(
        panel, chars, regression_weights=weights, min_securities=20, lag=0
    )
    model = _fit(panel, chars, regression_weights=weights, min_securities=20)
    assert len(model.notes.thin_periods) == 10
    assert universe == model.notes.regression_universe
    assert set(returns.securities) - set(universe) == {
        returns.securities[k] for k in (0, 1, 3)
    }


def test_a_security_without_characteristics_at_the_exposure_date_is_excluded():
    returns, chars, _, _ = _world(n_secs=40, n_periods=60, seed=43)
    values = chars.values.copy()
    values[-1, 3, 0] = np.nan
    holed = CharacteristicPanel(
        chars.dates, chars.securities, chars.names, values, chars.frequency
    )
    model = _fit(returns, holed)
    missing = chars.securities[3]
    assert missing in model.notes.no_exposures
    assert missing not in model.exposures
    assert any("incomplete characteristic row" in q for q in model.qualifications())


def test_a_security_with_too_little_history_is_excluded_and_named():
    returns, chars, _, _ = _world(n_secs=40, n_periods=60, seed=47)
    holed = returns.returns.copy()
    holed[:-5, 2] = np.nan
    thin = ReturnPanel(returns.dates, returns.securities, holed, returns.frequency)
    model = _fit(thin, chars, min_observations=24)
    short = returns.securities[2]
    assert short in model.notes.short_history
    assert short not in model.exposures
    assert any("residual observation" in q for q in model.qualifications())


def test_the_exposure_index_chooses_which_period_becomes_the_exposures():
    returns, chars, _, _ = _world(n_secs=40, n_periods=60, seed=53)
    last = _fit(returns, chars, exposure_index=-1)
    first = _fit(returns, chars, exposure_index=0)
    code = returns.securities[0]
    style_id = last.factor_ids[1]
    assert last.exposures[code].exposures[style_id] == pytest.approx(
        chars.values[-1, 0, 0]
    )
    assert first.exposures[code].exposures[style_id] == pytest.approx(
        chars.values[0, 0, 0]
    )


def test_notes_record_the_conventions_the_estimate_ran_under():
    returns, chars, _, _ = _world(n_secs=40, n_periods=60)
    model = _fit(returns, chars)
    assert model.notes.weighting == "equal"
    # True because the fixture panel is already standardised, not because this call did it.
    # The note records the state of the exposures, which is what downstream readers need.
    assert model.notes.standardised is True
    assert model.notes.market_factor is True
    assert 0.0 < model.notes.mean_r_squared < 1.0


def test_genuinely_unstandardised_exposures_are_reported_as_a_qualification():
    """The warning must key on the exposures, not on the parameter.

    This test previously passed an already-standardised panel with standardise=False and
    asserted the warning fired -- which pinned a false positive as if it were the contract.
    On the real S&P 500 run it told the reader that correctly-scaled exposures were on
    incomparable scales, which is exactly the kind of warning that teaches people to skim
    the rest.
    """
    returns, chars, _, _ = _world(n_secs=40, n_periods=60)
    raw = CharacteristicPanel(
        chars.dates, chars.securities, chars.names,
        chars.values * 1000.0 + 17.0, chars.frequency,
    )
    assert not raw.standardised
    model = _fit(returns, raw)
    assert model.notes.standardised is False
    assert any("incomparable scales" in q for q in model.qualifications())


def test_a_pre_standardised_panel_does_not_trigger_the_warning():
    """The other half: build_risk_indices returns a standardised panel and callers then
    pass standardise=False. Nothing is wrong and nothing should be said."""
    returns, chars, _, _ = _world(n_secs=40, n_periods=60)
    assert chars.standardised
    model = _fit(returns, chars)
    assert not any("incomparable scales" in q for q in model.qualifications())


def test_describe_works_for_cross_sectional_notes_too():
    """FactorModel.notes is a structural protocol, not the time-series class."""
    returns, chars, _, _ = _world(n_secs=40, n_periods=60)
    model = _fit(returns, chars)
    text = model.describe()
    assert "4-factor model" in text
    assert "monthly" in text


# ----------------------------------------------------------------------- the knobs


def test_regression_weights_change_the_estimate():
    returns, chars, _, _ = _world(n_secs=60, n_periods=120, seed=59)
    rng = np.random.default_rng(59)
    cap = rng.lognormal(mean=10.0, sigma=2.0, size=60)
    weights = np.tile(np.sqrt(cap), (120, 1))
    equal = _fit(returns, chars)
    weighted = _fit(returns, chars, regression_weights=weights)
    assert weighted.notes.weighting != equal.notes.weighting
    assert not np.allclose(
        equal.covariance.matrix, weighted.covariance.matrix, rtol=1e-6
    )


def test_a_zero_weight_removes_a_security_from_that_period():
    returns, chars, _, _ = _world(n_secs=40, n_periods=60, seed=61)
    weights = np.ones((60, 40))
    weights[0, 20:] = 0.0
    model = _fit(returns, chars, regression_weights=weights, min_securities=25)
    assert chars.dates[0] in model.notes.thin_periods


def test_the_covariance_estimator_is_pluggable():
    returns, chars, _, _ = _world(n_secs=60, n_periods=40, seed=67, n_char=8)
    plain = _fit(returns, chars)
    shrunk = _fit(
        returns, chars, covariance_estimator=ShrunkCovariance(SampleCovariance(), intensity=0.5)
    )
    assert not np.allclose(plain.covariance.matrix, shrunk.covariance.matrix)
    assert shrunk.notes.covariance is not None
    assert shrunk.notes.covariance.shrinkage_intensity == pytest.approx(0.5)


def test_annualising_scales_variance_by_periods_and_volatility_by_its_root():
    returns, chars, _, _ = _world(n_secs=60, n_periods=120, seed=71)
    monthly = _fit(returns, chars, annualise=False)
    annual = _fit(returns, chars, annualise=True)
    assert np.allclose(annual.covariance.matrix, monthly.covariance.matrix * 12.0)
    code = returns.securities[0]
    assert annual.exposures[code].specific.by_issuer[code] == pytest.approx(
        monthly.exposures[code].specific.by_issuer[code] * np.sqrt(12.0)
    )


def test_standardising_inside_the_estimator_matches_standardising_outside():
    returns, chars, _, _ = _world(n_secs=60, n_periods=120, seed=73)
    raw = CharacteristicPanel(
        chars.dates, chars.securities, chars.names,
        chars.values * 7.0 + 3.0, chars.frequency,
    )
    inside = _fit(returns, raw, standardise=True, winsorise=None)
    outside = _fit(returns, raw.standardise(winsorise=None), standardise=False)
    assert np.allclose(inside.covariance.matrix, outside.covariance.matrix)


def test_first_factor_id_shifts_the_whole_basis():
    returns, chars, _, _ = _world(n_secs=40, n_periods=60)
    model = _fit(returns, chars, first_factor_id=501)
    assert model.factor_ids == (501, 502, 503, 504)
    assert model.factor_names[501] == MARKET_FACTOR


# ------------------------------------------------- specific risk: measured, modelled, blended


def _persistent_world(
    n_secs: int = 80, n_periods: int = 60, seed: int = 5, sensitivity: float = 0.3
):
    """A world where the characteristic is persistent and really drives specific risk.

    Persistence matters: a characteristic redrawn every period cannot explain a security's
    specific risk however strong the relationship, which is a mistake easily made in a
    fixture and silently reported as a model that explains nothing.
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_periods)
    securities = tuple(f"S{i:03d}" for i in range(n_secs))
    level = rng.normal(size=n_secs)
    drift = rng.normal(scale=0.05, size=(n_periods, n_secs))
    raw = np.stack([level[None, :] + drift, rng.normal(size=(n_periods, n_secs))], axis=2)
    chars = CharacteristicPanel(
        dates, securities, ("Size", "Value"), raw, Frequency.MONTHLY
    ).standardise()
    design = np.concatenate([np.ones((n_periods, n_secs, 1)), chars.values], axis=2)
    factors = rng.normal(size=(n_periods, 3)) * np.array([0.04, 0.02, 0.015])
    specific = 0.06 * (1.0 + sensitivity * chars.values[:, :, 0])
    returns = np.einsum("tsk,tk->ts", design, factors) + rng.normal(
        size=(n_periods, n_secs)
    ) * specific
    caps = np.exp(10.0 + 2.0 * chars.values[:, :, 1])
    return ReturnPanel(dates, securities, returns, Frequency.MONTHLY), chars, caps


def test_historical_specific_risk_is_still_the_default():
    returns, chars, _ = _persistent_world()
    default = _fit(returns, chars, min_securities=20)
    explicit = _fit(returns, chars, min_securities=20, specific_risk="historical")
    assert default.notes.specific_risk_method == "historical"
    assert default.notes.specific_risk_model is None
    for code in default.exposures:
        assert default.exposures[code].specific.by_issuer[code] == pytest.approx(
            explicit.exposures[code].specific.by_issuer[code]
        )


def test_modelling_specific_risk_keeps_securities_that_measuring_it_drops():
    """The point of the whole module: a new issue has characteristics but no history."""
    returns, chars, _ = _persistent_world()
    blanked = returns.returns.copy()
    blanked[:-4, :3] = np.nan          # three securities with four months of history
    thin = ReturnPanel(returns.dates, returns.securities, blanked, returns.frequency)

    measured = _fit(thin, chars, min_securities=20, specific_risk="historical")
    modelled = _fit(thin, chars, min_securities=20, specific_risk="model")

    new_issues = set(returns.securities[:3])
    assert not (new_issues & set(measured.exposures))
    assert new_issues <= set(modelled.exposures)
    assert all(
        modelled.exposures[c].specific.by_issuer[c] > 0.0 for c in new_issues
    )


def test_the_blend_leans_on_the_model_for_a_new_issue_and_on_history_for_an_old_one():
    returns, chars, _ = _persistent_world()
    blanked = returns.returns.copy()
    blanked[:-4, :3] = np.nan
    thin = ReturnPanel(returns.dates, returns.securities, blanked, returns.frequency)
    model = _fit(thin, chars, min_securities=20, specific_risk="blend")
    risk_model = model.notes.specific_risk_model
    assert risk_model is not None
    # Thresholds would be wrong here: the weight depends on how much these particular
    # characteristics explain, and in this fixture they explain a great deal, so the model
    # keeps real weight even at sixty periods. The claim is the ordering and its size.
    assert risk_model.model_weight(4) > 0.75
    assert risk_model.model_weight(4) > 2.0 * risk_model.model_weight(60)
    assert risk_model.model_weight(60) < 0.5


def test_a_useless_characteristic_degrades_the_blend_to_the_historical_estimate():
    """Graceful degradation, and the reason the blend is safe to prefer.

    With characteristics that explain nothing, the inverse-variance weighting puts almost
    everything on the sample estimate by itself -- nobody has to notice and intervene.
    """
    returns, chars, _ = _persistent_world(sensitivity=0.0)
    measured = _fit(returns, chars, min_securities=20, specific_risk="historical")
    blended = _fit(returns, chars, min_securities=20, specific_risk="blend")
    shared = set(measured.exposures) & set(blended.exposures)
    difference = np.array(
        [
            blended.exposures[c].specific.by_issuer[c]
            / measured.exposures[c].specific.by_issuer[c]
            - 1.0
            for c in shared
        ]
    )
    assert float(np.mean(np.abs(difference))) < 0.20


def test_capitalisations_calibrate_kappa_through_the_estimator():
    returns, chars, caps = _persistent_world()
    plain = _fit(returns, chars, min_securities=20, specific_risk="model")
    sized = _fit(
        returns, chars, min_securities=20, specific_risk="model", capitalisations=caps
    )
    assert not plain.notes.specific_risk_model.notes.kappa_calibrated
    assert sized.notes.specific_risk_model.notes.kappa_calibrated
    assert len(sized.notes.specific_risk_model.kappa_knots) == 10


def test_the_specific_risk_model_notes_reach_the_models_qualifications():
    returns, chars, _ = _persistent_world()
    model = _fit(returns, chars, min_securities=20, specific_risk="model")
    assert any("falls back to the normal constant" in q for q in model.qualifications())


def test_an_unknown_specific_risk_method_is_refused():
    returns, chars, _ = _persistent_world()
    with pytest.raises(ValueError, match="'historical', 'model' or 'blend'"):
        _fit(returns, chars, min_securities=20, specific_risk="bayesian")


def test_capitalisations_of_the_wrong_shape_are_refused_by_the_estimator():
    returns, chars, _ = _persistent_world()
    with pytest.raises(ValueError, match="to match the return panel"):
        _fit(
            returns,
            chars,
            min_securities=20,
            specific_risk="model",
            capitalisations=np.ones((60, 9)),
        )


def test_short_history_is_still_named_even_when_it_no_longer_excludes():
    """Reported rather than silent: a modelled forecast for a security with four months of
    history is a different kind of number from one with sixty, and the notes should say so."""
    returns, chars, _ = _persistent_world()
    blanked = returns.returns.copy()
    blanked[:-4, :3] = np.nan
    thin = ReturnPanel(returns.dates, returns.securities, blanked, returns.frequency)
    model = _fit(thin, chars, min_securities=20, specific_risk="model")
    assert set(returns.securities[:3]) <= set(model.notes.short_history)
    assert set(returns.securities[:3]) <= set(model.exposures)


# ------------------------------- specific risk: estimated over the regression, forecast beyond it


def _never_in_a_regression(n_periods: int = 60):
    """The persistent world with its first two securities never in a regression: a complete
    characteristic row in every period, and not one return."""
    returns, chars, caps = _persistent_world(n_periods=n_periods)
    raw = returns.returns.copy()
    raw[:, :2] = np.nan
    return ReturnPanel(returns.dates, returns.securities, raw, returns.frequency), chars, caps


def test_specific_risk_is_estimated_over_the_regression_universe_and_nothing_else():
    """Two securities never reach a regression, so they have no residual to learn from. The
    specific risk model must come out exactly as it does from a panel that never held them,
    and its notes must speak for the securities it was fitted on, not the panel's. The S&P
    500 run reported 13 such securities as getting "the average-level forecast" when they got
    no forecast at all."""
    returns, chars, caps = _never_in_a_regression()
    values = chars.values.copy()
    values[:, 1, :] = np.nan            # and this one has no characteristic row either
    chars = dataclasses.replace(chars, values=values)
    model = _fit(
        returns, chars, min_securities=20, specific_risk="blend", capitalisations=caps
    )
    kept = returns.securities[2:]
    without = _fit(
        ReturnPanel(returns.dates, kept, returns.returns[:, 2:], returns.frequency),
        dataclasses.replace(chars, securities=kept, values=chars.values[:, 2:, :]),
        min_securities=20,
        specific_risk="blend",
        capitalisations=caps[:, 2:],
    )
    fitted = model.notes.specific_risk_model
    reference = without.notes.specific_risk_model
    assert set(returns.securities) - set(model.notes.regression_universe) == set(
        returns.securities[:2]
    )
    assert fitted.notes.n_securities == 78
    assert np.array_equal(fitted.gamma, reference.gamma)
    assert fitted.kappa_constant == reference.kappa_constant
    assert fitted.kappa_knots == reference.kappa_knots
    assert fitted.forecast_level == reference.forecast_level
    assert fitted.relative_residual_sd == reference.relative_residual_sd
    assert not any("no characteristic row" in q for q in model.qualifications())


def test_a_security_outside_the_universe_gets_a_fallback_and_never_the_floor_list():
    """It still needs a number -- a holding outside the estimation universe is held all the
    same -- and gets the characteristic forecast extrapolated to it, labelled as a fallback.
    Pushed far enough into the tail that the linear V runs out, that number is the floor, and
    the security is still not on the floor list: hitting the floor is a statement about an
    estimate, and nothing was estimated on it."""
    returns, chars, _ = _never_in_a_regression()
    values = chars.values.copy()
    values[-1, 0, 0] = -6.0
    model = _fit(
        returns,
        dataclasses.replace(chars, values=values),
        min_securities=20,
        specific_risk="blend",
    )
    outside = returns.securities[0]
    risk_model = model.notes.specific_risk_model
    row = dict(zip(chars.names, values[-1, 0, :].tolist()))
    assert 1.0 + risk_model.relative_level(row) < 0.1
    assert model.notes.fallback_specific_risk == returns.securities[:2]
    assert outside not in model.notes.floored_specific_risk
    assert outside not in model.notes.delivered_floor
    forecast = model.exposures[outside].specific.by_issuer[outside]
    assert forecast == risk_model.forecast({outside: row})[outside]
    assert forecast == risk_model.floor_level(0.1)
    lines = model.qualifications()
    assert any(q.startswith("2 security(ies) are in no estimated period's regression") and
               "fallback" in q and outside in q for q in lines)
    assert not any("floor" in q and outside in q for q in lines)


def test_a_blend_says_which_floor_was_hit_the_characteristic_forecasts_or_the_delivered_one():
    """`floored` tests the characteristic forecast alone, and under a blend that is not the
    number delivered. On the S&P 500 run JPM's characteristic forecast was below the floor and
    the blend delivered 12.8 percent, while the line said its forecast was the floor. Here both
    securities' characteristic forecasts are below it; the one with 120 periods of history is
    carried clear by it, the one with four is not."""
    returns, chars, _ = _persistent_world(n_periods=120)
    raw = returns.returns.copy()
    raw[:-4, 1] = np.nan
    panel = ReturnPanel(returns.dates, returns.securities, raw, returns.frequency)
    values = chars.values.copy()
    values[-1, :2, 0] = -6.0
    tail = dataclasses.replace(chars, values=values)
    long_history, short_history = returns.securities[:2]

    blended = _fit(panel, tail, min_securities=20, specific_risk="blend")
    notes = blended.notes
    floor = notes.specific_risk_model.floor_level(0.1)
    assert notes.floored_specific_risk == (long_history, short_history)
    assert notes.delivered_floor == (short_history,)
    assert blended.exposures[long_history].specific.by_issuer[long_history] > 3.0 * floor
    assert blended.exposures[short_history].specific.by_issuer[short_history] == floor
    lines = blended.qualifications()
    assert any(
        q.startswith("2 security(ies) have a characteristic forecast below the specific-risk "
                     "floor") and q.endswith(f": {long_history}, {short_history}")
        for q in lines
    )
    assert any(
        q.startswith("1 security(ies) are delivered the floor itself")
        and q.endswith(f": {short_history}")
        for q in lines
    )
    assert not any("rather than the model" in q for q in lines)

    modelled = _fit(panel, tail, min_securities=20, specific_risk="model")
    assert modelled.notes.delivered_floor == modelled.notes.floored_specific_risk
    assert any(
        "so their forecast is the floor rather than the model" in q
        for q in modelled.qualifications()
    )


def test_a_blend_that_carries_every_floored_forecast_clear_says_so():
    returns, chars, _ = _persistent_world(n_periods=120)
    values = chars.values.copy()
    values[-1, 0, 0] = -6.0
    model = _fit(
        returns,
        dataclasses.replace(chars, values=values),
        min_securities=20,
        specific_risk="blend",
    )
    assert model.notes.floored_specific_risk == (returns.securities[0],)
    assert model.notes.delivered_floor == ()
    assert any(
        q.endswith(
            "; blended with their own residual history, no forecast delivered is the "
            f"floor: {returns.securities[0]}"
        )
        for q in model.qualifications()
    )


# ------------------------------------------- point-in-time membership: who the index held


def _departures(n_periods: int = 60):
    """The persistent world inside an index: ten securities leave it halfway through, still
    trading, and a scattering of single days on which a security was not a member."""
    returns, chars, caps = _persistent_world(n_periods=n_periods)
    members = np.random.default_rng(83).random(returns.returns.shape) > 0.05
    members[n_periods // 2 :, :10] = False
    return returns, chars, caps, members


def test_membership_restricts_each_regression_exactly_as_blanking_the_non_members_would():
    """A security enters period t's regression only if it passes the existing test and is a
    member on t. That is the whole change: the model must come out exactly as it does from a
    panel whose non-member returns were never there -- factor returns, residuals, and the
    specific risk model fitted to them."""
    returns, chars, caps, members = _departures()
    blanked = ReturnPanel(
        returns.dates,
        returns.securities,
        np.where(members, returns.returns, np.nan),
        returns.frequency,
    )
    kw = dict(min_securities=20, specific_risk="blend", capitalisations=caps)
    masked = _fit(returns, chars, membership=members, **kw)
    reference = _fit(blanked, chars, **kw)
    assert np.array_equal(masked.covariance.matrix, reference.covariance.matrix)
    assert masked.notes.mean_r_squared == reference.notes.mean_r_squared
    assert masked.notes.sampling_variance == reference.notes.sampling_variance
    assert masked.notes.regression_universe == reference.notes.regression_universe
    assert masked.notes.cross_section_sizes == reference.notes.cross_section_sizes
    fitted = masked.notes.specific_risk_model
    expected = reference.notes.specific_risk_model
    assert np.array_equal(fitted.gamma, expected.gamma)
    assert fitted.relative_residual_sd == expected.relative_residual_sd
    assert masked.exposures == reference.exposures
    # And it is a restriction: the departed names lost half their periods.
    unmasked = _fit(returns, chars, **kw)
    assert not np.array_equal(masked.covariance.matrix, unmasked.covariance.matrix)


def test_a_membership_that_holds_everyone_changes_nothing():
    returns, chars, caps, _ = _departures()
    everyone = np.ones(returns.returns.shape, dtype=bool)
    kw = dict(min_securities=20, specific_risk="blend", capitalisations=caps)
    held = _fit(returns, chars, membership=everyone, **kw)
    default = _fit(returns, chars, **kw)
    assert np.array_equal(held.covariance.matrix, default.covariance.matrix)
    assert held.exposures == default.exposures
    assert held.notes.cross_section_sizes == default.notes.cross_section_sizes
    assert regression_universe(
        returns, chars, membership=everyone, min_securities=20, lag=0
    ) == regression_universe(returns, chars, min_securities=20, lag=0)


def test_the_regression_universe_counts_only_periods_in_which_a_security_was_a_member():
    """An industry scheme counts its members over this. A security that reached a regression
    only after leaving the index must not count, and nor may one whose member periods were
    all too thin to estimate."""
    returns, chars, _, _ = _world(n_secs=40, n_periods=60, seed=79)
    raw = returns.returns.copy()
    raw[:10, 3:] = np.nan               # the first ten periods are thin
    panel = ReturnPanel(returns.dates, returns.securities, raw, returns.frequency)
    members = np.ones((60, 40), dtype=bool)
    members[:, 0] = False               # a return in every period and never a member
    members[10:, 1] = False             # a member only while the periods were thin
    members[:59, 2] = False             # a member in the last period alone
    universe = regression_universe(
        panel, chars, membership=members, min_securities=20, lag=0
    )
    model = _fit(panel, chars, membership=members, min_securities=20)
    assert len(model.notes.thin_periods) == 10
    assert universe == model.notes.regression_universe
    assert set(returns.securities) - set(universe) == set(returns.securities[:2])
    assert regression_universe(panel, chars, min_securities=20, lag=0) == returns.securities


def test_the_notes_record_each_estimated_period_and_the_size_of_its_regression():
    """Restricting to members only removes, so the cross-sections shrink and sampling error
    rises. The sizes are what show it."""
    returns, chars, _, members = _departures()
    raw = returns.returns.copy()
    raw[3, 15:] = np.nan                # period 3 is thin
    panel = ReturnPanel(returns.dates, returns.securities, raw, returns.frequency)
    model = _fit(panel, chars, membership=members, min_securities=20)
    in_regression = members & np.isfinite(raw)
    counts = in_regression.sum(axis=1)
    estimated = counts >= 20
    assert returns.dates[3] in model.notes.thin_periods
    assert model.notes.estimated_dates == tuple(
        d for d, kept in zip(returns.dates, estimated) if kept
    )
    assert model.notes.cross_section_sizes == tuple(int(n) for n in counts[estimated])
    assert len(model.notes.cross_section_sizes) == model.notes.n_periods


def test_a_membership_of_the_wrong_shape_or_kind_is_refused():
    """A 0/1 float array is not a membership: NaN is truthy, and a weight array passed here
    by mistake would hold every security with a weight."""
    returns, chars, _, _ = _world(n_secs=40, n_periods=40)
    with pytest.raises(ValueError, match="to match the return panel"):
        _fit(returns, chars, membership=np.ones((40, 9), dtype=bool))
    with pytest.raises(ValueError, match="to match the return panel"):
        regression_universe(returns, chars, membership=np.ones((40, 9), dtype=bool))
    with pytest.raises(TypeError, match="boolean"):
        _fit(returns, chars, membership=np.ones((40, 40)))


# ------------------------------- the estimation universe: whose values set the scale


def _tails(n_dates: int = 3, n_secs: int = 60, seed: int = 89):
    """Fifty members, and ten securities outside the index sitting in the lower tail of
    both characteristics: the ex-members that traded on as they shrank."""
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=23.0, scale=1.2, size=(n_dates, n_secs, 2))
    values[:, :10, :] -= 4.0
    members = np.ones((n_dates, n_secs), dtype=bool)
    members[:, :10] = False
    caps = np.exp(values[:, :, 0])
    secs = tuple(f"S{i:02d}" for i in range(n_secs))
    return CharacteristicPanel(_dates(n_dates), secs, ("Size", "Other"), values), caps, members


def test_standardising_over_members_centres_and_scales_on_them_alone():
    """The members' cap-weighted mean is zero and their equal-weighted standard deviation one,
    to machine precision -- which is what puts the index itself at zero exposure."""
    panel, caps, members = _tails()
    out = panel.standardise(caps, winsorise=None, membership=members)
    for t in range(3):
        inside = out.values[t][members[t]]
        weights = caps[t][members[t]]
        for k in range(2):
            assert abs(float(np.average(inside[:, k], weights=weights))) < 1e-14
            assert float(np.std(inside[:, k], ddof=1)) == pytest.approx(1.0, rel=1e-14)
    everyone = panel.standardise(caps, winsorise=None).values[0][members[0]]
    assert float(np.std(everyone[:, 0], ddof=1)) < 0.8


def test_a_security_outside_the_index_is_put_on_the_members_scale_and_moves_none_of_it():
    panel, caps, members = _tails()
    out = panel.standardise(caps, membership=members)
    raw = panel.values[1, :, 0]
    sample = raw[members[1]]
    centre = float(np.median(sample))
    scale = float(np.median(np.abs(sample - centre))) * 1.4826
    low, high = centre - 4.0 * scale, centre + 4.0 * scale
    clipped = np.clip(sample, low, high)
    mean = float(np.average(clipped, weights=caps[1][members[1]]))
    spread = float(np.std(clipped, ddof=1))
    expected = np.clip((np.clip(raw[:10], low, high) - mean) / spread, -4.0, 4.0)
    assert out.values[1, :10, 0] == pytest.approx(expected, rel=1e-14)

    # Push the outsiders deeper into the tail: no member's exposure moves by a bit.
    deeper = panel.values.copy()
    deeper[:, :10, :] -= 3.0
    moved = dataclasses.replace(panel, values=deeper).standardise(caps, membership=members)
    assert np.array_equal(moved.values[:, 10:, :], out.values[:, 10:, :])
    # Without a membership they set the scale for everyone.
    assert not np.array_equal(
        dataclasses.replace(panel, values=deeper).standardise(caps).values[:, 10:, :],
        panel.standardise(caps).values[:, 10:, :],
    )


def test_the_members_bounds_bind_an_outsider_in_the_tail_and_are_counted():
    """Ex-members sit in the tails, so some will hit the members' bounds, and the panel
    counts how many apart from the members' own clipped values."""
    panel, caps, members = _tails()
    deep = panel.values.copy()
    deep[:, :10, 0] -= 8.0
    out = dataclasses.replace(panel, values=deep).standardise(caps, membership=members)
    raw_moved = outside = 0
    for t in range(3):
        # All ten were clipped to the same raw bound, so they share one exposure.
        assert np.ptp(out.values[t, :10, 0]) == 0.0
        for k in range(2):
            raw = deep[t, :, k]
            sample = raw[members[t]]
            centre = float(np.median(sample))
            scale = float(np.median(np.abs(sample - centre))) * 1.4826
            low, high = centre - 4.0 * scale, centre + 4.0 * scale
            moved = (raw < low) | (raw > high)
            raw_moved += int(moved.sum())
            clipped = np.clip(raw, low, high)
            mean = float(np.average(clipped[members[t]], weights=caps[t][members[t]]))
            spread = float(np.std(clipped[members[t]], ddof=1))
            moved |= np.abs((clipped - mean) / spread) > 4.0
            outside += int(moved[~members[t]].sum())
    assert outside >= 30
    assert out.clipped_outside == outside
    assert out.winsorised == raw_moved


def test_an_outsider_inside_the_raw_bound_can_still_hit_the_standardised_one():
    """LNCAP's case. A cap-weighted mean far above the median means a value the raw bound
    let through can still standardise past -4, and for an ex-member in the lower tail that
    is the bound that binds. It counts as hitting one."""
    rng = np.random.default_rng(97)
    values = np.append(np.clip(rng.normal(size=50), -2.5, 2.5), -3.0)
    values[0] = 2.5
    caps = np.ones((1, 51))
    caps[0, 0] = 1e6
    members = np.ones((1, 51), dtype=bool)
    members[0, 50] = False
    panel = CharacteristicPanel(
        _dates(1), tuple(f"S{i}" for i in range(51)), ("Size",), values.reshape(1, -1, 1)
    )
    out = panel.standardise(caps, membership=members)
    assert out.winsorised == 0
    assert out.clipped_outside == 1
    assert out.values[0, 50, 0] == -4.0


def test_a_membership_that_holds_everyone_standardises_exactly_as_none():
    panel, caps, _ = _tails()
    everyone = np.ones(caps.shape, dtype=bool)
    held = panel.standardise(caps, membership=everyone)
    default = panel.standardise(caps)
    assert np.array_equal(held.values, default.values)
    assert held.winsorised == default.winsorised
    assert held.clipped_outside == default.clipped_outside == 0


def test_a_day_with_no_member_value_is_left_missing_not_standardised_on_the_rest():
    panel, caps, members = _tails()
    members = members.copy()
    members[1, :] = False
    out = panel.standardise(caps, membership=members)
    assert np.isnan(out.values[1]).all()
    assert np.isfinite(out.values[[0, 2]]).all()


def test_standardise_refuses_a_membership_of_the_wrong_shape_or_kind():
    panel, caps, _ = _tails()
    with pytest.raises(ValueError, match="dates x securities"):
        panel.standardise(caps, membership=np.ones((3, 9), dtype=bool))
    with pytest.raises(TypeError, match="boolean"):
        panel.standardise(caps, membership=np.ones((3, 60)))


def test_the_estimator_standardises_over_the_membership_it_regresses_on():
    """One array for both, so the two cannot disagree about who is a member."""
    returns, chars, caps, members = _departures()
    raw = dataclasses.replace(chars, values=chars.values * 3.0 + 10.0, standardised=False)
    kw = dict(min_securities=20, specific_risk="blend", capitalisations=caps)
    inside = _fit(returns, raw, standardise=True, standardise_weights=caps,
                  membership=members, **kw)
    by_hand = _fit(returns, raw.standardise(caps, membership=members), membership=members, **kw)
    assert np.array_equal(inside.covariance.matrix, by_hand.covariance.matrix)
    assert inside.notes.mean_r_squared == by_hand.notes.mean_r_squared
    assert inside.exposures == by_hand.exposures
    # Not what standardising over everyone gives, which is what it used to do.
    everyone = _fit(returns, raw.standardise(caps), membership=members, **kw)
    assert everyone.exposures != inside.exposures


# --------------------------------- the lag: each regression fixed the day before its return


def _timed(n_periods: int = 40, n_secs: int = 50, seed: int = 97):
    """Characteristics, returns, weights and a membership, every value distinct, so a row read
    from the wrong date shows. S0000 joins on period 10, S0001 leaves on period 20, and S0002
    is a member on the last date alone."""
    rng = np.random.default_rng(seed)
    dates = _dates(n_periods)
    secs = tuple(f"S{i:04d}" for i in range(n_secs))
    chars = CharacteristicPanel(
        dates, secs, ("C0", "C1"), rng.normal(size=(n_periods, n_secs, 2)), Frequency.MONTHLY
    ).standardise()
    returns = ReturnPanel(
        dates, secs, rng.normal(scale=0.05, size=(n_periods, n_secs)), Frequency.MONTHLY
    )
    weights = rng.uniform(0.5, 2.0, size=(n_periods, n_secs))
    members = np.ones((n_periods, n_secs), dtype=bool)
    members[:10, 0] = False
    members[20:, 1] = False
    members[:-1, 2] = False
    return returns, chars, weights, members


def test_each_period_reads_its_exposures_weights_and_membership_from_the_date_before():
    returns, chars, weights, members = _timed()
    design = regression_design(returns, chars, regression_weights=weights, membership=members)
    assert design.lag == 1
    assert design.dates == returns.dates
    assert design.exposure_dates == (None,) + returns.dates[:-1]
    assert np.array_equal(design.returns, returns.returns)
    assert np.array_equal(design.exposures[1:], chars.values[:-1])
    assert np.array_equal(design.weights[1:], weights[:-1])
    assert np.array_equal(design.members[1:], members[:-1])
    # The first date has nothing earlier to read, so it is in no regression.
    assert np.isnan(design.exposures[0]).all()
    assert not design.members[0].any() and not design.in_regression[0].any()
    same_day = regression_design(
        returns, chars, regression_weights=weights, membership=members, lag=0
    )
    assert same_day.exposure_dates == returns.dates
    assert np.array_equal(same_day.exposures, chars.values)


def test_a_name_is_regressed_on_the_day_it_leaves_and_not_on_the_day_it_joins():
    """Membership is read with the exposures. On the day a name joins, its regression was fixed
    when it was not yet a member; on the day one leaves, it still was one."""
    returns, chars, weights, members = _timed()
    inside = regression_design(
        returns, chars, regression_weights=weights, membership=members
    ).in_regression
    assert not inside[10, 0] and inside[11, 0]
    assert inside[20, 1] and not inside[21, 1]
    assert not inside[:, 2].any()
    same_day = regression_design(
        returns, chars, regression_weights=weights, membership=members, lag=0
    ).in_regression
    assert same_day[10, 0] and not same_day[20, 1] and same_day[-1, 2]


def test_the_estimator_fits_the_lagged_design_and_forecasts_with_the_last_dates_exposures():
    """The lag reaches the fit: each period's factor returns are the weighted regression of
    that date's returns on the row before, redone here by hand. The exposures the model
    forecasts with are still the last date's -- as of today, forecast forward."""
    returns, chars, weights, members = _timed()
    model = estimate_cross_sectional_model(
        returns,
        chars,
        regression_weights=weights,
        membership=members,
        standardise=False,
        annualise=False,
        min_securities=20,
    )
    design = regression_design(returns, chars, regression_weights=weights, membership=members)
    notes = model.notes
    assert notes.lag == 1
    assert notes.estimated_dates == returns.dates[1:]
    assert returns.dates[0] not in notes.thin_periods
    for row, fitted in zip(range(1, len(returns.dates)), notes.factor_returns.returns):
        usable = design.in_regression[row]
        x = np.column_stack([np.ones(int(usable.sum())), design.exposures[row][usable]])
        w = design.weights[row, usable]
        root = np.sqrt(w * (usable.sum() / w.sum()))
        expected, *_ = np.linalg.lstsq(
            x * root[:, None], design.returns[row, usable] * root, rcond=None
        )
        assert fitted == pytest.approx(expected, rel=1e-12, abs=1e-15)
    market, *styles = model.factor_ids
    assert len(model.exposures) > 40
    for code, vector in model.exposures.items():
        j = returns.securities.index(code)
        assert vector.exposures[market] == 1.0
        assert [vector.exposures.get(fid, 0.0) for fid in styles] == list(chars.values[-1, j])


def test_the_regression_universe_agrees_with_the_estimator_under_the_lag():
    """S0002 is a member on the last date alone: same-day it reaches one regression, but read
    the day before it reaches none, and both counts must agree with their estimator."""
    returns, chars, weights, members = _timed()
    kw = dict(regression_weights=weights, membership=members, min_securities=20)
    for lag in (0, 1):
        universe = regression_universe(returns, chars, lag=lag, **kw)
        model = estimate_cross_sectional_model(
            returns, chars, lag=lag, standardise=False, **kw
        )
        assert universe == model.notes.regression_universe, lag
        assert ("S0002" in universe) == (lag == 0), lag


def test_the_specific_risk_model_learns_from_the_exposures_each_residual_was_regressed_on(
    monkeypatch,
):
    """Its characteristics and its capitalisations are read with the regression's, so no
    residual is explained by an exposure built from the price that produced it."""
    from praxis.midoffice.model import crosssectional

    captured = {}
    real = crosssectional.estimate_specific_risk_model

    def spy(residuals, securities, characteristics, names, **kw):
        captured.update(
            securities=securities,
            characteristics=characteristics,
            capitalisations=kw.get("capitalisations"),
        )
        return real(residuals, securities, characteristics, names, **kw)

    monkeypatch.setattr(crosssectional, "estimate_specific_risk_model", spy)
    returns, chars, weights, members = _timed()
    caps = weights**2
    estimate_cross_sectional_model(
        returns,
        chars,
        regression_weights=weights,
        membership=members,
        standardise=False,
        min_securities=20,
        specific_risk="model",
        capitalisations=caps,
    )
    design = regression_design(returns, chars, regression_weights=weights, membership=members)
    columns = [returns.securities.index(code) for code in captured["securities"]]
    assert np.array_equal(
        captured["characteristics"], design.exposures[:, columns], equal_nan=True
    )
    assert np.array_equal(captured["capitalisations"][1:], caps[:-1, columns])
    assert np.isnan(captured["capitalisations"][0]).all()


def test_a_lag_must_be_a_whole_number_of_periods_the_panel_can_supply():
    returns, chars, _, _ = _timed()
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="whole number"):
            regression_design(returns, chars, lag=bad)
    with pytest.raises(ValueError, match="earlier one"):
        estimate_cross_sectional_model(
            returns, chars, lag=len(returns.dates), standardise=False
        )


def test_the_notes_carry_the_factor_returns_the_covariance_was_estimated_from():
    returns, chars, _, _ = _world(n_secs=60, n_periods=48)
    model = _fit(returns, chars)
    series = model.notes.factor_returns
    assert series.dates == model.notes.estimated_dates
    assert series.names == tuple(model.factor_names[fid] for fid in model.factor_ids)
    assert series.returns.shape == (model.notes.n_periods, len(model.factor_ids))
