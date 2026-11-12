"""Tests for the covariance estimators.

Three claims carry the module and each is verified rather than asserted:

* equal weights reduce **exactly** to `np.cov(ddof=1)`, because an estimator whose general
  case does not contain its special case is usually wrong in both;
* exponential weighting really does discard information, measured by Kish's effective
  observation count rather than by the row count it leaves untouched;
* the split-horizon estimator really does get the long window's correlations and the short
  window's volatilities at the same time, which is the whole reason to estimate one matrix
  with two memories;
* shrinkage beats the sample covariance out of sample, by more the smaller the sample -- which
  is the entire justification for it and is checked by Monte Carlo against a known truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from praxis.midoffice.model.covariance import (
    ExponentialCovariance,
    HorizonCovariance,
    NeweyWestCovariance,
    SampleCovariance,
    ShrinkageTarget,
    ShrunkCovariance,
    ledoit_wolf_intensity,
)

P = 10


def true_covariance(seed=99):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(P, P))
    return a @ a.T / P + np.diag(rng.uniform(0.01, 0.05, P))


TRUE = true_covariance()


def draw(n, seed=0, truth=None):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(P), TRUE if truth is None else truth, size=n)


def distance(matrix, truth=TRUE):
    return float(np.sqrt(np.sum((matrix - truth) ** 2)))


# --- the special case is contained in the general case --------------------------------


def test_equal_weights_reduce_to_the_sample_covariance():
    """Exactly, not approximately. The reliability-weight denominator is what makes it so."""
    returns = draw(60, seed=3)
    estimate = SampleCovariance().estimate(returns)
    assert estimate.matrix == pytest.approx(
        np.cov(returns, rowvar=False, ddof=1), abs=1e-14
    )


def test_equal_weights_use_every_observation():
    assert SampleCovariance().estimate(draw(60, seed=3)).effective_observations == (
        pytest.approx(60.0)
    )


def test_every_estimate_is_symmetric():
    returns = draw(40, seed=5)
    for estimator in (
        SampleCovariance(),
        ExponentialCovariance(12),
        ShrunkCovariance(),
        ShrunkCovariance(ExponentialCovariance(12), ShrinkageTarget.CONSTANT_CORRELATION),
    ):
        matrix = estimator.estimate(returns).matrix
        assert matrix == pytest.approx(matrix.T, abs=1e-15)


# --- exponential weighting discards information ---------------------------------------


def test_the_decay_is_the_half_life_expressed_as_a_ratio():
    """Half-lives rather than lambdas, because 0.94 and 0.97 differ by a factor of two."""
    assert ExponentialCovariance(11).decay == pytest.approx(0.5 ** (1 / 11))
    assert ExponentialCovariance(1).decay == pytest.approx(0.5)


def test_a_shorter_half_life_leaves_fewer_effective_observations():
    returns = draw(120, seed=7)
    counts = [
        ExponentialCovariance(h).estimate(returns).effective_observations
        for h in (6, 12, 24, 60, 240)
    ]
    assert all(a < b for a, b in zip(counts, counts[1:])), counts
    assert all(c < 120.0 for c in counts)


def test_the_effective_count_approaches_its_asymptotic_limit():
    """(1 + lambda) / (1 - lambda), which is why it barely depends on history length."""
    half_life = 6.0
    decay = 0.5 ** (1 / half_life)
    limit = (1 + decay) / (1 - decay)
    for n in (60, 120, 480):
        count = ExponentialCovariance(half_life).estimate(draw(n, seed=7)).effective_observations
        assert count == pytest.approx(limit, rel=0.05), (n, count, limit)


def test_exponential_weighting_follows_a_change_in_volatility():
    """A quiet history then a loud recent stretch: the weighted estimate should notice."""
    rng = np.random.default_rng(13)
    quiet = rng.multivariate_normal(np.zeros(P), TRUE * 0.25, size=100)
    loud = rng.multivariate_normal(np.zeros(P), TRUE * 4.0, size=20)
    returns = np.vstack([quiet, loud])
    sample = np.trace(SampleCovariance().estimate(returns).matrix)
    weighted = np.trace(ExponentialCovariance(6).estimate(returns).matrix)
    assert weighted > sample * 1.5


def test_a_half_life_that_discards_most_of_the_history_says_so():
    estimate = ExponentialCovariance(3).estimate(draw(120, seed=7))
    assert any("more than half the history" in n for n in estimate.notes)


# --- shrinkage earns its place --------------------------------------------------------


@pytest.mark.parametrize("n,least_gain", [(12, 0.10), (20, 0.05), (30, 0.03)])
def test_shrinkage_beats_the_sample_covariance_on_a_small_sample(n, least_gain):
    """The whole justification, measured against a known truth over 60 draws."""
    sample, shrunk = [], []
    for seed in range(60):
        returns = draw(n, seed=seed)
        sample.append(distance(SampleCovariance().estimate(returns).matrix))
        shrunk.append(distance(ShrunkCovariance().estimate(returns).matrix))
    gain = (np.mean(sample) - np.mean(shrunk)) / np.mean(sample)
    assert gain > least_gain, f"n={n}: gain was only {gain:.1%}"


def test_the_gain_shrinks_as_the_sample_grows():
    """Converging on the sample covariance is correct: with enough data it is the right answer."""
    gains = []
    for n in (12, 30, 120):
        sample, shrunk = [], []
        for seed in range(60):
            returns = draw(n, seed=seed)
            sample.append(distance(SampleCovariance().estimate(returns).matrix))
            shrunk.append(distance(ShrunkCovariance().estimate(returns).matrix))
        gains.append((np.mean(sample) - np.mean(shrunk)) / np.mean(sample))
    assert all(a > b for a, b in zip(gains, gains[1:])), gains


def test_the_estimated_intensity_falls_as_the_sample_grows():
    intensities = [
        np.mean(
            [
                ShrunkCovariance().estimate(draw(n, seed=seed)).shrinkage_intensity
                for seed in range(40)
            ]
        )
        for n in (12, 30, 120, 480)
    ]
    assert all(a > b for a, b in zip(intensities, intensities[1:])), intensities
    assert intensities[-1] < 0.1


def test_shrinkage_repairs_a_singular_sample_covariance_at_source():
    """Which is the problem psd_relative_floor patches downstream.

    With six observations of ten factors the sample covariance is singular by construction.
    The shrunk estimate is positive definite, so the floor never has to fire for this reason.
    """
    returns = draw(6, seed=1)
    sample = SampleCovariance().estimate(returns)
    shrunk = ShrunkCovariance().estimate(returns)
    assert sample.is_rank_deficient
    assert np.linalg.eigvalsh(sample.matrix).min() < 1e-12
    assert np.linalg.eigvalsh(shrunk.matrix).min() > 1e-3
    assert shrunk.shrinkage_intensity > 0.3


def test_an_intensity_of_zero_is_the_raw_estimate_and_one_is_the_target():
    returns = draw(40, seed=5)
    raw = SampleCovariance().estimate(returns).matrix
    assert ShrunkCovariance(intensity=0.0).estimate(returns).matrix == pytest.approx(raw)
    pure = ShrunkCovariance(intensity=1.0).estimate(returns).matrix
    off_diagonal = pure[~np.eye(P, dtype=bool)]
    assert off_diagonal == pytest.approx(np.zeros_like(off_diagonal), abs=1e-15)


def test_the_diagonal_target_is_the_mean_variance_times_the_identity():
    returns = draw(40, seed=5)
    raw = SampleCovariance().estimate(returns).matrix
    pure = ShrunkCovariance(intensity=1.0).estimate(returns).matrix
    assert np.diag(pure) == pytest.approx(np.full(P, np.mean(np.diag(raw))))


def test_the_constant_correlation_target_keeps_each_variance():
    returns = draw(40, seed=5)
    raw = SampleCovariance().estimate(returns).matrix
    pure = ShrunkCovariance(
        intensity=1.0, target=ShrinkageTarget.CONSTANT_CORRELATION
    ).estimate(returns).matrix
    assert np.diag(pure) == pytest.approx(np.diag(raw))
    # And replaces every correlation with their average, so the off-diagonals are not raw.
    assert pure[0, 1] != pytest.approx(raw[0, 1], rel=1e-6)


def test_shrinkage_composes_with_exponential_weighting():
    """One decides which periods to believe, the other how much of the result to believe."""
    returns = draw(60, seed=5)
    estimate = ShrunkCovariance(ExponentialCovariance(12)).estimate(returns)
    assert estimate.half_life == 12
    assert estimate.shrinkage_intensity is not None
    assert estimate.effective_observations < 60.0
    assert "exponential" in estimate.method and "shrunk" in estimate.method


def test_a_target_the_estimate_already_equals_shrinks_completely():
    """Zero distance to the target means there is nothing to lose by taking it."""
    identity_returns = np.eye(4)[:, :4] * 0.0 + np.array([1.0, -1.0, 1.0, -1.0])[:, None]
    assert ledoit_wolf_intensity(
        identity_returns, np.zeros((4, 4)), np.zeros((4, 4))
    ) == 1.0


# --- validation -----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -12.0])
def test_a_non_positive_half_life_is_refused(bad):
    with pytest.raises(ValueError, match="half_life must be positive"):
        ExponentialCovariance(bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_an_intensity_outside_the_unit_interval_is_refused(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ShrunkCovariance(intensity=bad)


def test_one_period_is_refused():
    for estimator in (SampleCovariance(), ExponentialCovariance(12)):
        with pytest.raises(ValueError, match="at least two periods"):
            estimator.estimate(draw(1, seed=1))


def test_rank_deficiency_is_judged_on_effective_observations():
    """A long history with a short half-life has plenty of rows and little information."""
    estimate = ExponentialCovariance(2).estimate(draw(200, seed=7))
    assert estimate.n_periods == 200
    assert estimate.effective_observations < P
    assert estimate.is_rank_deficient, "200 rows, but fewer than P effective observations"


# --- the wiring into the model builder ------------------------------------------------


def test_the_builder_defaults_to_the_sample_covariance():
    """So adding the estimators changed no existing caller's numbers."""
    from datetime import date

    from praxis.midoffice.model.panel import FactorReturns, Frequency, ReturnPanel
    from praxis.midoffice.model.timeseries import estimate_timeseries_model

    n = 60
    dates = tuple(date(2015 + i // 12, i % 12 + 1, 28) for i in range(n))
    rng = np.random.default_rng(21)
    f = rng.normal(0, 0.04, (n, 3))
    factors = FactorReturns(dates, ("A", "B", "C"), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(dates, {"S": rng.normal(0, 0.05, n)}, Frequency.MONTHLY)

    default = estimate_timeseries_model(panel, factors, annualise=False)
    explicit = estimate_timeseries_model(
        panel, factors, annualise=False, covariance_estimator=SampleCovariance()
    )
    assert default.covariance.matrix == pytest.approx(explicit.covariance.matrix)
    assert default.notes.effective_observations == pytest.approx(float(n))


def test_the_thin_conditioning_warning_uses_effective_observations():
    """An exponentially weighted estimate can trip it where the row count would not."""
    from datetime import date

    from praxis.midoffice.model.panel import FactorReturns, Frequency, ReturnPanel
    from praxis.midoffice.model.timeseries import estimate_timeseries_model

    n, k = 36, 12
    dates = tuple(date(2015 + i // 12, i % 12 + 1, 28) for i in range(n))
    rng = np.random.default_rng(11)
    f = rng.normal(0, 0.04, (n, k))
    factors = FactorReturns(dates, tuple(f"F{i}" for i in range(k)), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(dates, {"S": rng.normal(0, 0.05, n)}, Frequency.MONTHLY)

    plain = estimate_timeseries_model(panel, factors, min_observations=24)
    assert plain.notes.periods_per_factor == pytest.approx(3.0)
    assert not any("per factor" in q for q in plain.qualifications())

    weighted = estimate_timeseries_model(
        panel, factors, min_observations=24,
        covariance_estimator=ExponentialCovariance(12),
    )
    assert weighted.notes.periods_per_factor < 3.0
    assert any("effective observation(s) per factor" in q for q in weighted.qualifications())


def test_shrinking_suppresses_the_conditioning_warning():
    """Because it has repaired the thing the warning is about."""
    from datetime import date

    from praxis.midoffice.model.panel import FactorReturns, Frequency, ReturnPanel
    from praxis.midoffice.model.timeseries import estimate_timeseries_model

    n, k = 36, 12
    dates = tuple(date(2015 + i // 12, i % 12 + 1, 28) for i in range(n))
    rng = np.random.default_rng(11)
    f = rng.normal(0, 0.04, (n, k))
    factors = FactorReturns(dates, tuple(f"F{i}" for i in range(k)), f, Frequency.MONTHLY)
    panel = ReturnPanel.of(dates, {"S": rng.normal(0, 0.05, n)}, Frequency.MONTHLY)

    shrunk = estimate_timeseries_model(
        panel, factors, min_observations=24,
        covariance_estimator=ShrunkCovariance(ExponentialCovariance(12)),
    )
    assert not any("per factor" in q for q in shrunk.qualifications())
    eigenvalues = np.linalg.eigvalsh(shrunk.covariance.matrix)
    assert eigenvalues.max() / eigenvalues.min() < 5.0, "shrinkage should condition it well"


# ------------------------------------------------------- split-horizon (BIM301) estimation


def _structured(n_periods: int = 600, seed: int = 0, spike: float = 3.0):
    """A stable correlation structure with a volatility regime change at the end.

    The hard case for a single half-life: correlations want a long window, the regime change
    wants a short one, and one parameter cannot serve both.
    """
    true = np.array([[1, .6, .3, 0], [.6, 1, .3, 0], [.3, .3, 1, 0], [0, 0, 0, 1]], float)
    chol = np.linalg.cholesky(true)
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=(n_periods, 4)) @ chol.T
    returns[-60:] *= spike
    return returns, true


def _correlation(matrix: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(matrix))
    return matrix / np.outer(sd, sd)


def test_equal_half_lives_reduce_to_the_single_half_life_estimator():
    """An estimator whose general case does not contain its special case is wrong in both."""
    returns, _ = _structured()
    split = HorizonCovariance(30.0, 30.0).estimate(returns)
    single = ExponentialCovariance(30.0).estimate(returns)
    assert np.allclose(split.matrix, single.matrix)
    assert split.effective_observations == pytest.approx(single.effective_observations)


def test_the_correlations_come_from_the_long_window_exactly():
    returns, _ = _structured()
    split = HorizonCovariance(volatility_half_life=12.0, correlation_half_life=72.0)
    long_only = ExponentialCovariance(72.0).estimate(returns)
    assert np.allclose(
        _correlation(split.estimate(returns).matrix), _correlation(long_only.matrix)
    )


def test_the_volatilities_come_from_the_short_window_exactly():
    returns, _ = _structured()
    split = HorizonCovariance(volatility_half_life=12.0, correlation_half_life=72.0)
    short_only = ExponentialCovariance(12.0).estimate(returns)
    assert np.allclose(
        np.sqrt(np.diag(split.estimate(returns).matrix)),
        np.sqrt(np.diag(short_only.matrix)),
    )


def test_with_newey_west_lags_equal_half_lives_reduce_to_newey_west():
    """The split horizon under Newey-West, as USE4 corrects both: its special case is the
    single-half-life Newey-West estimator."""
    returns, _ = _structured()
    split = HorizonCovariance(30.0, 30.0, newey_west_lags=4).estimate(returns)
    single = NeweyWestCovariance(lags=4, half_life=30.0).estimate(returns)
    assert np.allclose(split.matrix, single.matrix, rtol=1e-12, atol=0)
    assert split.effective_observations == pytest.approx(single.effective_observations)
    assert split.method == "Newey-West, 4 lag(s), split horizon"


def test_with_newey_west_lags_each_horizon_is_its_newey_west_estimate():
    returns, _ = _structured()
    split = HorizonCovariance(12.0, 72.0, newey_west_lags=3).estimate(returns).matrix
    long = NeweyWestCovariance(lags=3, half_life=72.0).estimate(returns).matrix
    short = NeweyWestCovariance(lags=3, half_life=12.0).estimate(returns).matrix
    assert np.allclose(np.diag(split), np.diag(short), rtol=1e-12, atol=0)
    assert np.allclose(_correlation(split), _correlation(long), rtol=1e-12, atol=1e-15)
    assert HorizonCovariance(12.0, 72.0, newey_west_lags=3).volatility_estimator() == (
        NeweyWestCovariance(lags=3, half_life=12.0)
    )
    with pytest.raises(ValueError, match="newey_west_lags"):
        HorizonCovariance(12.0, 72.0, newey_west_lags=-1)


def test_variance_ratio_volatilities_take_the_level_short_and_the_serial_correlation_long():
    """midoffice12's variant against the lags' noise: each variance its one-period variance on
    the volatility half-life times its Newey-West variance over its one-period variance on the
    ratio's half-life; the correlations the plain split's."""
    returns, _ = _structured()
    lags = 3
    plain = HorizonCovariance(12.0, 72.0, newey_west_lags=lags).estimate(returns).matrix
    level = np.diag(ExponentialCovariance(12.0).estimate(returns).matrix)
    for ratio_half_life in (72.0, 36.0):
        variant = HorizonCovariance(12.0, 72.0, newey_west_lags=lags,
                                    ratio_half_life=ratio_half_life)
        estimate = variant.estimate(returns)
        lagged = NeweyWestCovariance(lags=lags, half_life=ratio_half_life).estimate(returns)
        one_period = ExponentialCovariance(ratio_half_life).estimate(returns)
        assert np.allclose(np.diag(estimate.matrix),
                           level * np.diag(lagged.matrix) / np.diag(one_period.matrix),
                           rtol=1e-12, atol=0)
        assert np.allclose(_correlation(estimate.matrix), _correlation(plain), rtol=1e-12,
                           atol=1e-15)
        assert estimate.method == (f"Newey-West, 3 lag(s), split horizon, each volatility one "
                                   f"period's times its variance ratio on {ratio_half_life:g}")
    # On the volatility half-life it is the plain split; at no lags the ratio is one, exactly.
    same = HorizonCovariance(12.0, 72.0, newey_west_lags=lags, ratio_half_life=12.0)
    assert np.allclose(same.estimate(returns).matrix, plain, rtol=1e-12, atol=0)
    assert np.array_equal(
        HorizonCovariance(12.0, 72.0, newey_west_lags=0, ratio_half_life=72.0)
        .estimate(returns).matrix,
        HorizonCovariance(12.0, 72.0, newey_west_lags=0).estimate(returns).matrix,
    )
    with pytest.raises(ValueError, match="needs newey_west_lags"):
        HorizonCovariance(12.0, 72.0, ratio_half_life=72.0)
    with pytest.raises(ValueError, match="ratio_half_life must be positive"):
        HorizonCovariance(12.0, 72.0, newey_west_lags=2, ratio_half_life=0.0)
    with pytest.raises(ValueError, match="no quadratic form"):
        same.volatility_estimator()


def test_variance_ratio_volatilities_carry_less_of_the_lags_noise():
    """Why the variant: twenty lags of autocovariance on a 90-day half-life are mostly noise.
    Over AR(1) factors its variances scatter about the Bartlett target at under seven tenths
    of the plain split's spread, and sit no further from it on average."""
    lags, rho = 20, 0.2
    j = np.arange(1, lags + 1)
    target = (1.0 + 2.0 * np.sum((1.0 - j / (lags + 1.0)) * rho**j)) / (1.0 - rho**2)
    plain, ratio = [], []
    for seed in range(40):
        returns = _ar1(rho, 1500, 700 + seed)
        plain.append(np.diag(
            HorizonCovariance(90.0, 504.0, newey_west_lags=lags).estimate(returns).matrix))
        ratio.append(np.diag(
            HorizonCovariance(90.0, 504.0, newey_west_lags=lags, ratio_half_life=504.0)
            .estimate(returns).matrix))
    plain, ratio = np.array(plain) / target, np.array(ratio) / target
    assert ratio.std() < 0.7 * plain.std()
    assert abs(ratio.mean() - 1.0) <= abs(plain.mean() - 1.0)


def test_the_split_gets_both_optima_where_neither_single_half_life_can():
    """Barra's rationale, verified rather than quoted.

    Correlation accuracy improves monotonically with half-life because the structure is
    stationary; volatility accuracy is U-shaped because a regime change has to be tracked.
    The optima are therefore at different half-lives, and one parameter must compromise.
    """
    off = np.triu_indices(4, 1)

    def score(estimator, reps=60):
        corr_err, vol_err = [], []
        for seed in range(reps):
            returns, true = _structured(seed=4000 + seed)
            estimate = estimator.estimate(returns)
            corr_err.append(
                np.sqrt(np.mean((_correlation(estimate.matrix)[off] - true[off]) ** 2))
            )
            vol_err.append(
                np.sqrt(np.mean((np.sqrt(np.diag(estimate.matrix)) - 3.0) ** 2))
            )
        return float(np.mean(corr_err)), float(np.mean(vol_err))

    short_corr, short_vol = score(ExponentialCovariance(12.0))
    long_corr, long_vol = score(ExponentialCovariance(72.0))
    split_corr, split_vol = score(HorizonCovariance(12.0, 72.0))

    # Each single half-life is good at one thing and bad at the other.
    assert long_corr < short_corr
    assert short_vol < long_vol
    # The split matches the better of the two at both, simultaneously.
    assert split_corr == pytest.approx(long_corr, rel=1e-9)
    assert split_vol == pytest.approx(short_vol, rel=1e-9)


def test_the_split_stays_positive_semi_definite():
    """A PSD correlation matrix rescaled by a positive diagonal stays PSD, so the split
    cannot introduce the negative eigenvalues psd_relative_floor exists to repair."""
    returns, _ = _structured(n_periods=200, seed=5)
    estimate = HorizonCovariance.bim301_long(12.0).estimate(returns)
    assert float(np.linalg.eigvalsh(estimate.matrix).min()) > -1e-14


def test_effective_observations_report_the_correlation_window():
    """The binding constraint: the correlation matrix has n(n-1)/2 elements to the
    volatilities' n, so it is what decides whether the estimate is singular."""
    returns, _ = _structured()
    split = HorizonCovariance(volatility_half_life=12.0, correlation_half_life=72.0)
    estimate = split.estimate(returns)
    assert estimate.effective_observations == pytest.approx(
        ExponentialCovariance(72.0).estimate(returns).effective_observations
    )
    assert estimate.effective_observations > ExponentialCovariance(
        12.0
    ).estimate(returns).effective_observations


def test_kish_effective_observations_match_barras_three_times_rule():
    """BIM301 footnote: 'the effective observation window is three times the half life'.

    The closed form for an unbounded exponentially weighted window is (1+lambda)/(1-lambda),
    which tends to 2/ln(2) = 2.885 times the half-life. Barra rounds that to three; this is
    an independent check that the Kish measure here means what they mean.
    """
    rng = np.random.default_rng(0)
    for half_life in (12.0, 36.0, 48.0):
        returns = rng.normal(size=(int(20 * half_life), 3))
        estimate = ExponentialCovariance(half_life).estimate(returns)
        assert estimate.effective_observations / half_life == pytest.approx(
            2.0 / np.log(2.0), rel=1e-3
        )


def test_the_bim301_long_configuration_is_one_year_and_three_years():
    monthly = HorizonCovariance.bim301_long(12.0)
    assert monthly.volatility_half_life == pytest.approx(12.0)
    assert monthly.correlation_half_life == pytest.approx(36.0)
    weekly = HorizonCovariance.bim301_long(52.0)
    assert weekly.volatility_half_life == pytest.approx(52.0)


def test_the_bim301_short_configuration_is_ninety_trading_days():
    assert HorizonCovariance.bim301_short(252.0).volatility_half_life == pytest.approx(90.0)
    # The same configuration survives a change of panel frequency.
    assert HorizonCovariance.bim301_short(52.0).volatility_half_life == pytest.approx(
        90.0 / 252.0 * 52.0
    )
    assert HorizonCovariance.bim301_short(12.0).correlation_half_life == pytest.approx(24.0)


def test_the_short_horizon_configuration_is_more_responsive_than_the_long():
    """Responsiveness is about a transition, not a level.

    Once both windows sit inside the same volatility regime the two configurations agree,
    and the shorter one is merely noisier -- which is why this fixture spikes only the last
    eight periods, while the long window is still mostly looking at the old regime.
    """
    returns, _ = _structured()
    returns[-60:] /= 3.0        # undo the long regime change
    returns[-8:] *= 3.0         # a fresh one the long window has barely seen
    short = HorizonCovariance.bim301_short(12.0).estimate(returns)
    long = HorizonCovariance.bim301_long(12.0).estimate(returns)
    assert np.all(np.diag(short.matrix) > np.diag(long.matrix))
    assert short.effective_observations < long.effective_observations


def test_a_completed_regime_change_leaves_both_horizons_agreeing():
    """The other side of the same coin, pinned so the test above is not read as a claim
    that a short half-life simply forecasts more risk."""
    returns, _ = _structured()          # spike covers the last 60 periods: both windows see it
    short = np.sqrt(np.diag(HorizonCovariance.bim301_short(12.0).estimate(returns).matrix))
    long = np.sqrt(np.diag(HorizonCovariance.bim301_long(12.0).estimate(returns).matrix))
    assert np.allclose(short, long, rtol=0.35)
    assert not np.all(short > long)


def test_a_non_positive_half_life_is_refused_by_the_split_horizon():
    # Named apart from the exponential estimator's test of the same refusal, which this one's
    # old name shadowed, so that one never ran.
    with pytest.raises(ValueError, match="volatility_half_life must be positive"):
        HorizonCovariance(0.0, 36.0)
    with pytest.raises(ValueError, match="correlation_half_life must be positive"):
        HorizonCovariance(12.0, -1.0)


def test_a_factor_with_no_variation_is_left_uncorrelated_rather_than_invented():
    returns, _ = _structured()
    returns[:, 3] = 0.0
    estimate = HorizonCovariance(12.0, 72.0).estimate(returns)
    assert np.allclose(estimate.matrix[3, :], 0.0)
    assert np.allclose(estimate.matrix[:, 3], 0.0)
    assert any("no variation over the correlation window" in n for n in estimate.notes)


def test_describe_names_both_half_lives():
    returns, _ = _structured(n_periods=100)
    text = HorizonCovariance.bim301_long(12.0).estimate(returns).describe()
    assert "12 vol" in text and "36 corr" in text


def test_the_split_estimator_plugs_into_a_factor_model():
    from datetime import date

    from praxis.midoffice.model.characteristics import CharacteristicPanel
    from praxis.midoffice.model.crosssectional import estimate_cross_sectional_model
    from praxis.midoffice.model.panel import Frequency, ReturnPanel

    rng = np.random.default_rng(9)
    n_periods, n_secs = 180, 60
    dates = tuple(date(2000 + t // 12, t % 12 + 1, 28) for t in range(n_periods))
    secs = tuple(f"S{i:03d}" for i in range(n_secs))
    chars = CharacteristicPanel(
        dates, secs, ("A", "B"), rng.normal(size=(n_periods, n_secs, 2)), Frequency.MONTHLY
    ).standardise()
    returns = ReturnPanel(
        dates, secs, rng.normal(scale=0.05, size=(n_periods, n_secs)), Frequency.MONTHLY
    )
    model = estimate_cross_sectional_model(
        returns,
        chars,
        standardise=False,
        min_securities=20,
        covariance_estimator=HorizonCovariance.bim301_long(12.0),
    )
    assert model.notes.covariance is not None
    assert model.notes.covariance.volatility_half_life == pytest.approx(12.0)
    assert "split horizon" in model.describe()


# ------------------------------------------------------------- Newey-West autocorrelation


def _ar1(rho: float, n_periods: int, seed: int, n_factors: int = 2) -> np.ndarray:
    """An AR(1) factor series, whose long-run variance is known in closed form."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(n_periods, n_factors))
    series = np.zeros_like(innovations)
    for t in range(1, n_periods):
        series[t] = rho * series[t - 1] + innovations[t]
    return series


def _bartlett_expectation(rho: float, lags: int) -> float:
    """What a Bartlett-weighted estimator should recover, as a multiple of Gamma_0.

    Not the full (1+rho)/(1-rho): the Bartlett weights discount every lag, so the estimator
    is biased low by a predictable amount. Testing against the right target rather than
    against the long-run variance is what distinguishes a kernel working correctly from one
    that is broken.
    """
    j = np.arange(1, lags + 1)
    return float(1.0 + 2.0 * np.sum((1.0 - j / (lags + 1.0)) * rho**j))


def test_bartlett_weights_at_h_minus_one_lags_are_the_per_day_variance_of_h_day_returns():
    """The horizon (midoffice11's ruling 3), exactly. Two days k apart fall together in h - k
    of the h-day windows, so the windows' sum of squares counts lag k's autocovariance h - k
    times -- per day 1 - k/h, Bartlett's weight at h - 1 lags. Exact on a series padded with
    h - 1 zeros at each end, so every window that reaches a day holds it whole, and summing to
    zero, so the estimator's centring takes nothing off; normalised as the estimator normalises
    one day's, by n - 1."""
    rng = np.random.default_rng(41)
    for h in (1, 2, 5, 11, 21):
        draws = rng.integers(-9, 10, size=(299, 3)).astype(float)
        core = np.vstack([draws, -draws.sum(axis=0)])
        pad = np.zeros((h - 1, 3))
        series = np.vstack([pad, core, pad])
        n = series.shape[0]
        sums = np.array([series[t : t + h].sum(axis=0) for t in range(n - h + 1)])
        per_day = sums.T @ sums / (h * (n - 1))
        estimate = NeweyWestCovariance(lags=h - 1).estimate(series).matrix
        assert np.allclose(estimate, per_day, rtol=1e-12, atol=0)
    # One day is no lags: the one-day covariance itself.
    assert np.array_equal(NeweyWestCovariance(lags=0).estimate(core).matrix,
                          SampleCovariance().estimate(core).matrix)


def test_zero_lags_reduce_to_the_sample_covariance():
    returns = _ar1(0.4, 200, seed=0)
    assert np.allclose(
        NeweyWestCovariance(lags=0).estimate(returns).matrix,
        SampleCovariance().estimate(returns).matrix,
    )


def test_it_recovers_what_bartlett_weighting_says_it_should():
    for rho, lags in ((0.2, 5), (0.3, 9), (0.5, 9)):
        measured = []
        for seed in range(10):
            returns = _ar1(rho, 4000, seed=200 + seed)
            corrected = NeweyWestCovariance(lags=lags).estimate(returns)
            plain = NeweyWestCovariance(lags=0).estimate(returns)
            measured.append(
                float(np.mean(np.diag(corrected.matrix) / np.diag(plain.matrix)))
            )
        assert np.mean(measured) == pytest.approx(
            _bartlett_expectation(rho, lags), rel=0.05
        )


def test_the_correction_is_in_the_right_direction_and_grows_with_persistence():
    ratios = []
    for rho in (0.0, 0.2, 0.4, 0.6):
        values = []
        for seed in range(6):
            returns = _ar1(rho, 2000, seed=300 + seed)
            corrected = NeweyWestCovariance(lags=10).estimate(returns)
            plain = NeweyWestCovariance(lags=0).estimate(returns)
            values.append(
                float(np.mean(np.diag(corrected.matrix) / np.diag(plain.matrix)))
            )
        ratios.append(float(np.mean(values)))
    assert ratios[0] == pytest.approx(1.0, abs=0.05)
    assert all(a < b for a, b in zip(ratios, ratios[1:]))


def test_an_independent_series_is_left_alone_on_average():
    """Unbiased, not identical -- and the difference matters for how this is tested.

    With no true autocorrelation the estimated autocovariances are pure noise with a standard
    deviation of about Gamma_0/sqrt(T) each, so a single panel's correction is a few percent
    off in either direction and an off-diagonal element whose true value is zero can move by
    much more than any relative tolerance allows. The claim is that it does not move
    systematically.
    """
    ratios = []
    for seed in range(20):
        rng = np.random.default_rng(5 + seed)
        returns = rng.normal(size=(2000, 3))
        corrected = NeweyWestCovariance().estimate(returns)
        plain = SampleCovariance().estimate(returns)
        ratios.append(float(np.mean(np.diag(corrected.matrix) / np.diag(plain.matrix))))
    mean = float(np.mean(ratios))
    stderr = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
    assert abs(mean - 1.0) < max(3.0 * stderr, 0.02)


def test_bartlett_weighting_keeps_the_estimate_positive_semi_definite():
    """The reason for the weights, over enough panels to mean something."""
    worst = np.inf
    for seed in range(120):
        rng = np.random.default_rng(900 + seed)
        n_periods = int(rng.integers(30, 300))
        n_factors = int(rng.integers(2, 9))
        matrix = NeweyWestCovariance().estimate(
            _ar1(0.4, n_periods, 900 + seed, n_factors=n_factors)
        ).matrix
        worst = min(worst, float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0).min()))
    assert worst > 0.0


def test_combining_it_with_exponential_weighting_can_break_that_guarantee():
    """Pinned because the docstring claims it and because the combination is tempting.

    Short panel, many factors, heavy decay: the estimate goes indefinite most of the time,
    and by a wide margin rather than a rounding error.
    """
    negatives = 0
    worst = np.inf
    for seed in range(120):
        rng = np.random.default_rng(5000 + seed)
        n_periods = int(rng.integers(25, 60))
        n_factors = int(rng.integers(5, 12))
        matrix = NeweyWestCovariance(lags=8, half_life=6.0).estimate(
            _ar1(0.6, n_periods, 5000 + seed, n_factors=n_factors)
        ).matrix
        smallest = float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0).min())
        worst = min(worst, smallest)
        negatives += smallest < -1e-12
    assert negatives > 60
    assert worst < -0.1


def test_the_combination_warns_that_it_is_no_longer_guaranteed():
    estimate = NeweyWestCovariance(lags=4, half_life=12.0).estimate(_ar1(0.4, 80, 7))
    assert any("not guaranteed positive" in note for note in estimate.notes)


def test_an_indefinite_result_is_named_rather_than_left_to_be_discovered():
    estimate = NeweyWestCovariance(lags=8, half_life=6.0).estimate(
        _ar1(0.6, 30, 5000, n_factors=8)
    )
    smallest = float(np.linalg.eigvalsh(estimate.matrix).min())
    if smallest < 0.0:
        assert any("negative eigenvalue" in note for note in estimate.notes)


def test_the_automatic_bandwidth_follows_newey_and_wests_rule():
    estimator = NeweyWestCovariance()
    for n_periods in (100, 250, 1000, 4000):
        assert estimator.bandwidth(n_periods) == int(
            np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0))
        )
    # Never more lags than there are usable pairs.
    assert NeweyWestCovariance(lags=500).bandwidth(10) == 9


def test_a_bandwidth_of_zero_says_it_did_nothing():
    estimate = NeweyWestCovariance(lags=0).estimate(_ar1(0.4, 60, 3))
    assert any("no autocorrelation correction at all" in n for n in estimate.notes)


def test_the_method_names_the_lag_count():
    assert "9 lag(s)" in NeweyWestCovariance(lags=9).estimate(_ar1(0.3, 100, 1)).method


def test_negative_lags_and_non_positive_half_lives_are_refused():
    with pytest.raises(ValueError, match="lags must be non-negative"):
        NeweyWestCovariance(lags=-1)
    with pytest.raises(ValueError, match="half_life must be positive"):
        NeweyWestCovariance(half_life=0.0)


def test_newey_west_plugs_into_a_factor_model():
    from datetime import date

    from praxis.midoffice.model.panel import FactorReturns, Frequency, ReturnPanel
    from praxis.midoffice.model.timeseries import estimate_timeseries_model

    rng = np.random.default_rng(21)
    n_periods = 120
    dates = tuple(date(2000 + t // 12, t % 12 + 1, 28) for t in range(n_periods))
    factor_returns = FactorReturns(
        dates, ("A", "B"), _ar1(0.4, n_periods, 21) * 0.03, Frequency.MONTHLY
    )
    securities = tuple(f"S{i:02d}" for i in range(20))
    panel = ReturnPanel(
        dates, securities, rng.normal(scale=0.05, size=(n_periods, 20)), Frequency.MONTHLY
    )
    model = estimate_timeseries_model(
        panel, factor_returns, covariance_estimator=NeweyWestCovariance(lags=6)
    )
    assert "Newey-West" in model.describe()
    plain = estimate_timeseries_model(panel, factor_returns)
    assert np.trace(model.covariance.matrix) > np.trace(plain.covariance.matrix)
