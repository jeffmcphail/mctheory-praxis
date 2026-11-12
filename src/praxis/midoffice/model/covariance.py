"""Covariance estimators: how much of the sample covariance to believe.

For a risk *forecast* these matter more than the regression does.  A sample covariance over
120 monthly periods weights a return from 2016 exactly like last month's, and its smallest
eigenvalues are mostly estimation noise -- which is the same conditioning problem
`RiskConfig.psd_relative_floor` patches downstream rather than fixing at source.

Three ideas, and one that ties them together.

**Exponential weighting** gives recent periods more weight, with a half-life.  RiskMetrics
uses a daily decay of 0.94, about an 11-day half-life; a monthly model wants something far
longer.  The cost is information: down-weighting the past means fewer observations in effect.

**Shrinkage** pulls the sample covariance toward a structured target.  Ledoit and Wolf's
insight is that the optimal intensity can be estimated from the data rather than chosen, and
that the resulting matrix beats the sample covariance out of sample at exactly the sample
sizes risk models actually have.

**Effective observations** is the thread.  Equal weights over `n` periods give `n` effective
observations; exponential weights give fewer, by Kish's measure `(sum w)^2 / sum w^2`.  That
number, not the raw period count, is what determines how much of the matrix is signal -- so
it is what the conditioning warning should use.  A 120-period history with a 6-month half-life
carries about 17 effective observations, and a 40-factor covariance built from it is singular
however many rows the input had.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np


class ShrinkageTarget(Enum):
    """What a shrunk covariance is pulled toward.

    `DIAGONAL` scales the identity by the mean variance, which says "assume the factors are
    uncorrelated and equally volatile". `CONSTANT_CORRELATION` keeps each factor's own
    variance and replaces every correlation with their average, which is usually the better
    target for security returns. For *factor* returns the diagonal is often the more honest
    one, because a factor set is typically built to be close to orthogonal already.
    """

    DIAGONAL = "diagonal"
    CONSTANT_CORRELATION = "constant_correlation"


@dataclass(frozen=True)
class CovarianceEstimate:
    """A covariance matrix and how much of the sample it actually used."""

    matrix: np.ndarray
    effective_observations: float
    method: str
    n_periods: int = 0
    shrinkage_intensity: float | None = None
    half_life: float | None = None
    volatility_half_life: float | None = None
    correlation_half_life: float | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        object.__setattr__(self, "matrix", (matrix + matrix.T) / 2.0)

    @property
    def n_factors(self) -> int:
        return self.matrix.shape[0]

    @property
    def observations_per_factor(self) -> float:
        return (
            self.effective_observations / self.n_factors
            if self.n_factors
            else float("inf")
        )

    @property
    def is_rank_deficient(self) -> bool:
        """By effective observations, not by row count.

        A long history with a short half-life has plenty of rows and very little information,
        and the row count would say the matrix is fine.
        """
        return self.effective_observations <= self.n_factors

    def describe(self) -> str:
        parts = [self.method]
        if self.volatility_half_life is not None:
            parts.append(
                f"half-life {self.volatility_half_life:g} vol / "
                f"{self.correlation_half_life:g} corr"
            )
        elif self.half_life is not None:
            parts.append(f"half-life {self.half_life:g}")
        if self.shrinkage_intensity is not None:
            parts.append(f"shrinkage {self.shrinkage_intensity:.3f}")
        parts.append(
            f"{self.effective_observations:.1f} effective observation(s) of "
            f"{self.n_periods}"
        )
        return ", ".join(parts)


class CovarianceEstimator(Protocol):
    """Anything that turns a period-by-factor return matrix into a covariance estimate."""

    def estimate(self, returns: np.ndarray) -> CovarianceEstimate: ...


def _weighted_covariance(
    returns: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    """Weighted covariance with the reliability-weight correction, and Kish's effective n.

    The denominator `1 - sum(w^2)` on normalised weights is what makes this reduce exactly to
    the `ddof=1` sample covariance when the weights are equal -- `test_equal_weights_reduce_to
    _the_sample_covariance` pins that, because an estimator whose general case does not
    contain its special case is usually wrong in both.
    """
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("covariance weights must sum to something positive")
    w = weights / total
    sum_squares = float(w @ w)
    effective = 1.0 / sum_squares if sum_squares > 0.0 else 0.0

    mean = w @ returns
    centred = returns - mean
    denominator = 1.0 - sum_squares
    if denominator <= 0.0:
        raise ValueError(
            "the weights put all their mass on effectively one period, so there is no "
            "dispersion left to measure"
        )
    matrix = (centred * w[:, None]).T @ centred / denominator
    return matrix, effective


@dataclass(frozen=True)
class SampleCovariance:
    """Equal weights on every period. Honest, and the worst of the estimators here."""

    def estimate(self, returns: np.ndarray) -> CovarianceEstimate:
        returns = np.atleast_2d(np.asarray(returns, dtype=float))
        n = returns.shape[0]
        if n < 2:
            raise ValueError(f"a covariance needs at least two periods, got {n}")
        matrix, effective = _weighted_covariance(returns, np.ones(n))
        return CovarianceEstimate(
            matrix=matrix,
            effective_observations=effective,
            method="sample",
            n_periods=n,
        )


@dataclass(frozen=True)
class ExponentialCovariance:
    """Exponentially weighted, by half-life in periods.

    `half_life` is in the panel's own periods, so 24 means two years on monthly data and five
    weeks on daily. Expressing the decay as a half-life rather than as RiskMetrics' lambda is
    deliberate: 0.94 and 0.97 are indistinguishable to read and differ by a factor of two in
    memory, 11 days against 23.
    """

    half_life: float

    def __post_init__(self) -> None:
        if self.half_life <= 0.0:
            raise ValueError(f"half_life must be positive, got {self.half_life!r}")

    @property
    def decay(self) -> float:
        """RiskMetrics' lambda, for comparison with published figures."""
        return float(0.5 ** (1.0 / self.half_life))

    def estimate(self, returns: np.ndarray) -> CovarianceEstimate:
        returns = np.atleast_2d(np.asarray(returns, dtype=float))
        n = returns.shape[0]
        if n < 2:
            raise ValueError(f"a covariance needs at least two periods, got {n}")
        # Most recent period is the last row and carries the most weight.
        ages = np.arange(n - 1, -1, -1, dtype=float)
        weights = self.decay**ages
        matrix, effective = _weighted_covariance(returns, weights)
        notes: list[str] = []
        if effective < 0.5 * n:
            notes.append(
                f"a half-life of {self.half_life:g} period(s) over {n} period(s) leaves "
                f"{effective:.1f} effective observation(s), so more than half the history is "
                f"being discarded by the weighting"
            )
        return CovarianceEstimate(
            matrix=matrix,
            effective_observations=effective,
            method="exponential",
            n_periods=n,
            half_life=self.half_life,
            notes=tuple(notes),
        )


@dataclass(frozen=True)
class HorizonCovariance:
    """Correlations on a long half-life, volatilities on a short one, recombined.

    Barra's BIM301 design, and their rationale is worth stating because it is not obvious that
    one matrix should be estimated with two memories.

    Correlation structure is **more stationary than volatility**.  Volatility moves fast and
    clusters, so a forecast wants recent observations weighted heavily; correlations drift
    slowly, so weighting recent observations heavily throws away information for nothing.

    More decisively, the correlation matrix has far more elements to estimate -- n(n-1)/2
    against n -- so the robustness problem binds much harder there.  A short half-life on a
    correlation matrix is how you arrive at the spurious structure that makes a minimum-risk
    portfolio forecast zero risk and then realise plenty of it.  Estimating correlations long
    and volatilities short is a way of spending the available history where it buys the most:
    responsiveness on the few parameters that need it, robustness on the many that do not.

    This is also why `effective_observations` here reports the **correlation** half-life's
    figure.  That is the one that decides whether the matrix is singular, and it is the
    constraint Barra's own reasoning is organised around.

    The recombination is safe: a positive semi-definite correlation matrix rescaled by a
    positive diagonal stays positive semi-definite, so the split cannot introduce the negative
    eigenvalues `psd_relative_floor` exists to repair.

    Equal half-lives reduce this exactly to `ExponentialCovariance`, which
    `test_equal_half_lives_reduce_to_the_single_half_life_estimator` pins -- an estimator
    whose general case does not contain its special case is usually wrong in both.

    `newey_west_lags`, given, estimates both matrices by `NeweyWestCovariance` with that many
    lags at their half-lives, as USE4 corrects both horizons for serial correlation; equal
    half-lives then reduce it to `NeweyWestCovariance`. None, the default, is BIM301's plain
    exponential weighting. Newey-West under exponential weights is not guaranteed positive
    semi-definite, and neither then is the recombination: the notes say when it is not.
    """

    volatility_half_life: float
    correlation_half_life: float
    newey_west_lags: int | None = None

    def __post_init__(self) -> None:
        for name in ("volatility_half_life", "correlation_half_life"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if self.newey_west_lags is not None and self.newey_west_lags < 0:
            raise ValueError(
                f"newey_west_lags must be non-negative, or None for no Newey-West; got "
                f"{self.newey_west_lags!r}"
            )

    def _estimator(self, half_life: float):
        if self.newey_west_lags is None:
            return ExponentialCovariance(half_life)
        return NeweyWestCovariance(lags=self.newey_west_lags, half_life=half_life)

    def volatility_estimator(self) -> "ExponentialCovariance | NeweyWestCovariance":
        """The estimator the diagonal comes from: each variance is its estimate, the
        correlations aside. So the diagonal is a quadratic form in the returns even though the
        matrix is not."""
        return self._estimator(self.volatility_half_life)

    @classmethod
    def bim301_long(cls, periods_per_year: float = 12.0) -> "HorizonCovariance":
        """BIM301L: volatility half-life one year, correlation half-life three years.

        The long-horizon model, for an investor rebalancing on a monthly-ish cadence.  Barra
        calls these figures typical rather than universal -- component models vary -- so this
        is a starting configuration, not a constant.
        """
        return cls(periods_per_year, 3.0 * periods_per_year)

    @classmethod
    def bim301_short(cls, periods_per_year: float = 252.0) -> "HorizonCovariance":
        """BIM301S: volatility half-life 90 trading days, correlation half-life two years.

        The short-horizon model, built for a one-month forecast and a trading cadence to
        match.  90 trading days is expressed here as a fraction of a year so the
        configuration survives a change of panel frequency: on weekly data it is 18 periods,
        on monthly about 4.3.
        """
        return cls((90.0 / 252.0) * periods_per_year, 2.0 * periods_per_year)

    def estimate(self, returns: np.ndarray) -> CovarianceEstimate:
        returns = np.atleast_2d(np.asarray(returns, dtype=float))
        n = returns.shape[0]
        if n < 2:
            raise ValueError(f"a covariance needs at least two periods, got {n}")
        ages = np.arange(n - 1, -1, -1, dtype=float)

        if self.newey_west_lags is None:
            corr_decay = 0.5 ** (1.0 / self.correlation_half_life)
            vol_decay = 0.5 ** (1.0 / self.volatility_half_life)
            corr_matrix, corr_effective = _weighted_covariance(returns, corr_decay**ages)
            vol_matrix, vol_effective = _weighted_covariance(returns, vol_decay**ages)
        else:
            long = self._estimator(self.correlation_half_life).estimate(returns)
            short = self._estimator(self.volatility_half_life).estimate(returns)
            corr_matrix, corr_effective = long.matrix, long.effective_observations
            vol_matrix = short.matrix

        # Correlations from the long estimate, volatilities from the short one.
        long_sd = np.sqrt(np.clip(np.diag(corr_matrix), 0.0, None))
        short_sd = np.sqrt(np.clip(np.diag(vol_matrix), 0.0, None))
        safe = np.where(long_sd > 0.0, long_sd, 1.0)
        correlation = corr_matrix / np.outer(safe, safe)
        # A factor with no movement over the long window has no correlation to anyone; leave
        # it uncorrelated rather than inventing one, and let its own variance stand.
        flat = long_sd <= 0.0
        if flat.any():
            correlation[flat, :] = 0.0
            correlation[:, flat] = 0.0
            correlation[flat, flat] = 1.0
        matrix = correlation * np.outer(short_sd, short_sd)

        notes: list[str] = []
        if flat.any():
            notes.append(
                f"{int(flat.sum())} factor(s) had no variation over the correlation window, "
                f"so they are treated as uncorrelated with everything"
            )
        if corr_effective < 0.5 * n:
            notes.append(
                f"a correlation half-life of {self.correlation_half_life:g} period(s) over "
                f"{n} period(s) leaves {corr_effective:.1f} effective observation(s), so "
                f"more than half the history is being discarded by the weighting"
            )
        method = "exponential, split horizon"
        if self.newey_west_lags is not None:
            method = f"Newey-West, {self.newey_west_lags} lag(s), split horizon"
            smallest = float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0).min())
            if smallest < 0.0:
                notes.append(
                    f"the split-horizon Newey-West matrix has a negative eigenvalue "
                    f"({smallest:.2e}), so some portfolio has a negative variance forecast until "
                    f"psd_relative_floor repairs it"
                )
        return CovarianceEstimate(
            matrix=matrix,
            effective_observations=corr_effective,
            method=method,
            n_periods=n,
            half_life=self.correlation_half_life,
            volatility_half_life=self.volatility_half_life,
            correlation_half_life=self.correlation_half_life,
            notes=tuple(notes),
        )


@dataclass(frozen=True)
class NeweyWestCovariance:
    """Autocorrelation-corrected covariance, with Bartlett weights.

    The other two corrections in this module -- exponential weighting and shrinkage -- both
    assume each period is an independent draw.  Returns are not.  BIM301 applies Newey-West
    throughout and its footnote is blunt about why: serial correlation is "significant enough
    to affect model performance, even out to two weeks for highly liquid developed market
    stocks."

    **What goes wrong without it.**  A risk number is almost always quoted for a horizon
    longer than one period -- annualised, or monthly from daily data -- and that conversion
    multiplies the one-period variance by the number of periods.  That step is only valid
    under independence.  With positive autocorrelation the variance of a sum grows faster
    than the number of terms, so annualising a one-period variance *understates* the risk,
    and understates it by more the more persistent the series.  For an AR(1) with coefficient
    rho the understatement is exactly `(1 + rho) / (1 - rho)`: at rho = 0.3 the true
    annualised variance is 1.86 times what independence would give.

    What this estimates instead is the **long-run** covariance:

        S = Gamma_0 + sum_j  (1 - j/(L+1)) (Gamma_j + Gamma_j')

    where `Gamma_j` is the lag-j autocovariance and `1 - j/(L+1)` are the Bartlett weights.
    Multiplying *that* by the number of periods is valid, which is the whole point.

    The Bartlett weights are not a smoothing convenience.  A truncated sum of raw
    autocovariances can easily produce a matrix with negative eigenvalues -- a negative
    variance forecast for some portfolio -- and Bartlett weights make the estimator positive
    semi-definite by construction.  Over 400 random panels the equal-weighted estimator's
    smallest eigenvalue never fell below +0.08.

    **Combined with exponential weighting that guarantee is gone, and not academically so.**
    On short panels with many factors and a heavy decay -- 25 to 60 periods, 5 to 12 factors,
    a six-period half-life -- 310 of 400 panels came out with a negative eigenvalue, the worst
    at -0.487.  That is a real negative variance, not a rounding artifact.  The combination is
    still worth having, since both corrections address real effects, but `psd_relative_floor`
    is doing load-bearing work underneath it and the notes say when the combination was used.

    `lags` defaults to Newey and West's own rule, `floor(4 (T/100)^(2/9))`, which is
    deliberately conservative and, measured, close to the best available.  Bartlett weighting
    leaves some of the adjustment on the table -- the expected recovery is
    `1 + 2 sum_j (1 - j/(L+1)) rho^j` rather than the full `(1+rho)/(1-rho)` -- and raising L
    does *not* close the gap: more lags cut the truncation bias but add estimation noise from
    high-lag autocovariances that should be zero, and past about twenty lags the measured
    recovery stops improving and starts falling back.  At rho = 0.5 and T = 4000, L = 9
    recovers 2.57 of the true 3.00, L = 20 recovers 2.67, and L = 50 recovers 2.67.  At daily
    frequency BIM301's two-week observation argues for setting it explicitly.
    """

    lags: int | None = None
    half_life: float | None = None

    def __post_init__(self) -> None:
        if self.lags is not None and self.lags < 0:
            raise ValueError(f"lags must be non-negative, got {self.lags!r}")
        if self.half_life is not None and self.half_life <= 0.0:
            raise ValueError(f"half_life must be positive, got {self.half_life!r}")

    def bandwidth(self, n_periods: int) -> int:
        """How many lags this estimator will use on a panel of this length."""
        if self.lags is not None:
            return min(self.lags, max(n_periods - 1, 0))
        automatic = int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0)))
        return max(0, min(automatic, max(n_periods - 1, 0)))

    def estimate(self, returns: np.ndarray) -> CovarianceEstimate:
        returns = np.atleast_2d(np.asarray(returns, dtype=float))
        n = returns.shape[0]
        if n < 2:
            raise ValueError(f"a covariance needs at least two periods, got {n}")

        if self.half_life is None:
            weights = np.ones(n)
        else:
            decay = 0.5 ** (1.0 / self.half_life)
            weights = decay ** np.arange(n - 1, -1, -1, dtype=float)
        w = weights / weights.sum()
        sum_squares = float(w @ w)
        effective = 1.0 / sum_squares if sum_squares > 0.0 else 0.0
        denominator = 1.0 - sum_squares
        if denominator <= 0.0:
            raise ValueError(
                "the weights put all their mass on effectively one period, so there is no "
                "dispersion left to measure"
            )

        mean = w @ returns
        centred = returns - mean
        matrix = (centred * w[:, None]).T @ centred / denominator

        bandwidth = self.bandwidth(n)
        for lag in range(1, bandwidth + 1):
            recent = centred[lag:]
            earlier = centred[:-lag]
            pair_weights = w[lag:]
            gamma = (recent * pair_weights[:, None]).T @ earlier / denominator
            bartlett = 1.0 - lag / (bandwidth + 1.0)
            matrix = matrix + bartlett * (gamma + gamma.T)

        notes: list[str] = []
        if bandwidth == 0:
            notes.append(
                "the lag truncation came out at zero, so this is the sample covariance with "
                "no autocorrelation correction at all"
            )
        if self.half_life is not None:
            notes.append(
                "Newey-West combined with exponential weighting is not guaranteed positive "
                "semi-definite the way the equal-weighted estimator is; psd_relative_floor "
                "is what repairs it if it is not"
            )
        smallest = float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0).min())
        if smallest < 0.0:
            notes.append(
                f"the autocorrelation-corrected matrix has a negative eigenvalue "
                f"({smallest:.2e}), so some portfolio has a negative variance forecast until "
                f"psd_relative_floor repairs it"
            )
        return CovarianceEstimate(
            matrix=matrix,
            effective_observations=effective,
            method=f"Newey-West, {bandwidth} lag(s)",
            n_periods=n,
            half_life=self.half_life,
            notes=tuple(notes),
        )


def _target(matrix: np.ndarray, target: ShrinkageTarget) -> np.ndarray:
    variances = np.diag(matrix).copy()
    if target is ShrinkageTarget.DIAGONAL:
        return np.eye(len(variances)) * variances.mean()
    # Constant correlation: keep each variance, average every off-diagonal correlation.
    volatilities = np.sqrt(np.maximum(variances, 0.0))
    outer = np.outer(volatilities, volatilities)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(outer > 0.0, matrix / outer, 0.0)
    n = len(variances)
    if n < 2:
        return matrix.copy()
    off_diagonal = correlation[~np.eye(n, dtype=bool)]
    average = float(np.mean(off_diagonal)) if off_diagonal.size else 0.0
    structured = np.full((n, n), average) * outer
    np.fill_diagonal(structured, variances)
    return structured


def ledoit_wolf_intensity(returns: np.ndarray, matrix: np.ndarray, target: np.ndarray) -> float:
    """The shrinkage intensity Ledoit and Wolf's estimator implies, clipped to [0, 1].

    The ratio of the sample covariance's own estimation variance to its distance from the
    target: shrink hard when the sample is noisy relative to how far it sits from the
    structure, and barely at all when it is not. Estimating this rather than choosing it is
    the whole point -- a hand-picked intensity is a free parameter nobody can defend.
    """
    returns = np.atleast_2d(np.asarray(returns, dtype=float))
    n = returns.shape[0]
    centred = returns - returns.mean(axis=0)
    distance = float(np.sum((matrix - target) ** 2))
    if distance <= 0.0:
        return 1.0
    # Mean squared Frobenius distance between each period's outer product and the estimate:
    # how much the sample covariance moves when one observation changes.
    scatter = 0.0
    for row in centred:
        outer = np.outer(row, row)
        scatter += float(np.sum((outer - matrix) ** 2))
    variance = scatter / (n * n)
    return float(np.clip(min(variance, distance) / distance, 0.0, 1.0))


@dataclass(frozen=True)
class ShrunkCovariance:
    """Any estimator, pulled toward a structured target.

    Wrapping rather than replacing, because shrinking an exponentially weighted covariance is
    a thing practitioners actually do and the two concerns are independent: one decides which
    periods to believe, the other how much of the result to believe.
    """

    inner: CovarianceEstimator = field(default_factory=SampleCovariance)
    target: ShrinkageTarget = ShrinkageTarget.DIAGONAL
    intensity: float | None = None

    def __post_init__(self) -> None:
        if self.intensity is not None and not 0.0 <= self.intensity <= 1.0:
            raise ValueError(
                f"shrinkage intensity must be in [0, 1], got {self.intensity!r}; it is the "
                f"weight on the target, so 0 is the raw estimate and 1 discards it entirely"
            )

    def estimate(self, returns: np.ndarray) -> CovarianceEstimate:
        base = self.inner.estimate(returns)
        structured = _target(base.matrix, self.target)
        intensity = (
            ledoit_wolf_intensity(returns, base.matrix, structured)
            if self.intensity is None
            else self.intensity
        )
        matrix = intensity * structured + (1.0 - intensity) * base.matrix
        notes = list(base.notes)
        if intensity > 0.9:
            notes.append(
                f"shrinkage intensity came out at {intensity:.3f}, so the estimate is almost "
                f"entirely the {self.target.value} target; the sample covariance is carrying "
                f"very little information at this sample size"
            )
        return CovarianceEstimate(
            matrix=matrix,
            # Shrinkage does not add observations, but it does make the matrix usable at
            # sample sizes where the raw estimate is singular. The effective count stays
            # honest about the data; conditioning should be judged on the shrunk matrix.
            effective_observations=base.effective_observations,
            method=f"{base.method}+shrunk({self.target.value})",
            n_periods=base.n_periods,
            shrinkage_intensity=intensity,
            half_life=base.half_life,
            notes=tuple(notes),
        )
