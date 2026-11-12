"""Out-of-sample bias statistics: whether a risk model's forecasts match the risk that followed.

The test Barra runs on its own models. Each forecast is made at a date's close from data through
that date alone, taken to the horizon's scale, and set against the return the portfolio then
realised: z = realised / forecast. Over T forecasts the bias statistic is the standard deviation
of z:

    B = sd(z)

one when the forecasts are right, below one where they overpredict risk and above where they
underpredict it. For normal z its own standard error is about 1/sqrt(2T), so the 95 percent band
is about 1 +/- sqrt(2/T): +/-0.054 over the 686 daily forecasts of the S&P 500 run, +/-0.25 over
its 32 non-overlapping 21-day ones.

**Fat tails widen the band.** The variance of a sample variance is (kurtosis - 1)/T, not 2/T,
and a single stock's daily returns are far from normal, so a member alone can fall outside the
band from one earnings day. `BiasStatistics.kurtosis` is reported beside the statistic, and the
band it implies, so the two readings can be told apart.

**What a portfolio is held for.** Each is formed at the forecast date from what was known then --
that date's members, capitalisations and forecast -- and held to the next. A member with no
return the next day (an acquired company, its last day as a member) is left out of every
portfolio formed that day, as the index itself leaves it out; how many such member-days there
were is counted. Over a longer horizon a portfolio is bought and held, a name that stops trading
held as cash from then.

**At a horizon of h periods** every forecast, at h times its one-period variance, is set against
the h periods' return after it (midoffice12). The windows overlap, each starting a period after
the last, so the outcomes are serially correlated and the statistic varies more than 2/T says:
its band is found by simulation, right forecasts of normal returns (`overlapping_band`). Every
h-th forecast from the first, the windows apart, is the statistic beside it, its band the plain
one. At h = 1 both are the one-period statistic.

**The eigenportfolios realise against estimated factor returns,** which carry each period's
estimation noise: they test whether the covariance's eigenvalues are biased -- the case for
Barra's eigenfactor adjustment -- and cannot judge a correction for that noise, which real
portfolios can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import nnls

from ..risk.basis import PSDRepair, nearest_psd

ROLLING_WINDOW = 252
"""Twelve months of daily forecasts: the rolling windows' length."""

OVERLAP_DRAWS = 10_000
"""How many simulated tests the overlapping windows' band is read from: its quantiles are then
good to about 0.001."""

OVERLAP_SEED = 12
"""The band's simulation is seeded, so a test reads the same band every time it is run."""


def bias_statistic(z: np.ndarray) -> float:
    """The standard deviation of the standardised outcomes, ddof 1, over the finite ones."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return float("nan")
    return float(np.std(z, ddof=1))


def bias_band(n: int) -> tuple[float, float]:
    """The statistic's 95 percent band for n normal outcomes: about 1 +/- sqrt(2/n)."""
    if n < 1:
        return float("nan"), float("nan")
    half = float(np.sqrt(2.0 / n))
    return 1.0 - half, 1.0 + half


@lru_cache(maxsize=None)
def _simulated_bands(size: int, horizon: int, draws: int, seed: int) -> tuple[np.ndarray, ...]:
    """The 2.5 and 97.5 percentiles of the statistic over the first n of `size` overlapping
    windows, for every n from 2: `draws` tests of right forecasts of standard normal returns,
    each window the sum of `horizon` consecutive returns over sqrt(horizon), each starting a
    period after the last. One simulation serves every n, each test's statistic read off its
    prefix sums."""
    k = np.arange(2, size + 1, dtype=float)
    stats = np.empty((draws, size - 1))
    rng = np.random.default_rng(seed)
    chunk = 1_000
    for start in range(0, draws, chunk):
        m = min(chunk, draws - start)
        days = rng.standard_normal((m, size + horizon - 1))
        running = np.concatenate([np.zeros((m, 1)), np.cumsum(days, axis=1)], axis=1)
        z = (running[:, horizon:] - running[:, :-horizon]) / np.sqrt(horizon)
        first = np.cumsum(z, axis=1)[:, 1:]
        second = np.cumsum(z * z, axis=1)[:, 1:]
        variance = (second - first * first / k) / (k - 1.0)
        stats[start : start + m] = np.sqrt(np.clip(variance, 0.0, None))
    low, high = np.quantile(stats, [0.025, 0.975], axis=0)
    return low, high


def overlapping_band(
    n: int, horizon: int, *, draws: int = OVERLAP_DRAWS, seed: int = OVERLAP_SEED
) -> tuple[float, float]:
    """The statistic's 95 percent band for n outcomes over `horizon` periods each whose windows
    overlap, each starting a period after the last: for right forecasts of normal returns, by
    simulation. The overlap makes the outcomes serially correlated, and the band wider than
    1 +/- sqrt(2/n) -- about sqrt((2h^2 + 1) / 3h) times as wide, for large n. At horizon 1 the
    windows are apart, and it is `bias_band`'s."""
    if horizon <= 1:
        return bias_band(n)
    if n < 2:
        return float("nan"), float("nan")
    # Simulated once per size class, so a band depends on n and the horizon alone.
    size = max(1024, 1 << (n - 1).bit_length())
    low, high = _simulated_bands(size, int(horizon), int(draws), int(seed))
    return float(low[n - 2]), float(high[n - 2])


def rolling_bias(z: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """The statistic over every run of `window` consecutive outcomes, stepped one at a time;
    a window with fewer than two finite outcomes is nan. Empty if there is no full window."""
    z = np.asarray(z, dtype=float)
    if z.size < window:
        return np.empty(0)
    return np.array([bias_statistic(z[i : i + window]) for i in range(z.size - window + 1)])


@dataclass(frozen=True)
class BiasStatistics:
    """One portfolio's bias statistic over a test, its band, its rolling windows."""

    n: int
    statistic: float
    low: float
    high: float
    # The rolling windows' statistics, and their mean absolute deviation from one.
    rolling: np.ndarray = field(repr=False, compare=False)
    mrad: float = float("nan")
    # The standardised outcomes' kurtosis (normal 3), and the band it implies:
    # 1 +/- 1.96 sqrt((kurtosis - 1) / (4 n)).
    kurtosis: float = float("nan")
    mean: float = float("nan")
    # How many periods each outcome's window spans, each starting a period after the last: 1
    # for outcomes apart, whose band is 1 +/- sqrt(2/n); more, and the band is simulated.
    overlap: int = 1

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.statistic):
            return "-"
        if self.statistic < self.low:
            return "over"
        if self.statistic > self.high:
            return "under"
        return "in"

    @property
    def fat_band(self) -> tuple[float, float]:
        """The band the outcomes' own kurtosis implies; for independent outcomes alone, so nan
        where the windows overlap."""
        if not np.isfinite(self.kurtosis) or self.n < 1 or self.overlap > 1:
            return float("nan"), float("nan")
        half = 1.96 * float(np.sqrt(max(self.kurtosis - 1.0, 0.0) / (4.0 * self.n)))
        return 1.0 - half, 1.0 + half


def bias_statistics(
    forecast: np.ndarray,
    realised: np.ndarray,
    *,
    window: int = ROLLING_WINDOW,
    overlap: int = 1,
) -> BiasStatistics:
    """The statistic of realised / forecast over the outcomes where both are finite and the
    forecast is positive, in date order. `overlap` is how many periods each outcome's window
    spans, each starting a period after the last: past 1 the band is `overlapping_band`'s."""
    forecast = np.asarray(forecast, dtype=float)
    realised = np.asarray(realised, dtype=float)
    ok = np.isfinite(forecast) & np.isfinite(realised) & (forecast > 0.0)
    z = np.where(ok, realised / np.where(ok, forecast, 1.0), np.nan)
    finite = z[ok]
    n = int(finite.size)
    low, high = bias_band(n) if overlap <= 1 else overlapping_band(n, overlap)
    rolling = rolling_bias(z[ok], window)
    kurtosis = mean = float("nan")
    if n >= 4:
        centred = finite - finite.mean()
        kurtosis = float(np.mean(centred**4) / np.mean(centred**2) ** 2)
        mean = float(finite.mean())
    return BiasStatistics(
        n=n,
        statistic=bias_statistic(finite),
        low=low,
        high=high,
        rolling=rolling,
        mrad=float(np.mean(np.abs(rolling - 1.0))) if rolling.size else float("nan"),
        kurtosis=kurtosis,
        mean=mean,
        overlap=max(int(overlap), 1),
    )


@dataclass(frozen=True)
class RiskForecast:
    """One date's forecast in arrays, on a fixed list of securities.

    `covariance` is the factor covariance as the risk engine uses it -- through
    `risk.basis.nearest_psd` at the configured floor, `repair` saying whether that bound --
    annualised; `exposures` securities by factors, a row of nan where a security has no
    forecast; `specific` its annualised specific volatility, nan likewise. `periods_per_year`
    takes a variance to a single period's scale.
    """

    covariance: np.ndarray
    exposures: np.ndarray
    specific: np.ndarray
    periods_per_year: float
    repair: PSDRepair = field(default_factory=lambda: PSDRepair(False, 0, 0.0, 0.0))
    # The eigenportfolios to test, annualised eigenvalues and eigenvectors as columns, where they
    # are not the covariance's own sorted afresh: under the eigenfactor risk adjustment the
    # unadjusted covariance's, in its order, so rank k is one eigenfactor at every scale.
    eigenbasis: tuple[np.ndarray, np.ndarray] | None = field(default=None, compare=False)

    @classmethod
    def from_model(
        cls, model, securities: Sequence[str], *, eigen_scale: float | None = None
    ) -> "RiskForecast":
        """A `FactorModel`'s forecast on `securities`, in its factor order. `eigen_scale` reads
        it at another eigenfactor adjustment scale a than the model's own, from the
        simulation its notes carry (`eigen.EigenBias`), one simulation serving every a; at 0,
        the covariance unadjusted."""
        notes = model.notes
        periods = notes.frequency.periods_per_year if notes.annualised else 1.0
        scale = notes.frequency.variance_scale if notes.annualised else 1.0
        eigen = getattr(notes, "eigen", None)
        own = getattr(notes, "eigen_adjustment", None)
        if eigen_scale is None or eigen_scale == (own or 0.0):
            matrix = model.covariance.matrix
            a = own
        else:
            if eigen is None:
                raise ValueError("the model carries no eigenfactor simulation to read another "
                                 "scale from; fit it with eigen_adjustment set")
            matrix = eigen.adjusted(eigen_scale) * scale
            a = eigen_scale
        eigenbasis = None
        if eigen is not None and a:
            values, vectors = eigen.basis(a)
            eigenbasis = (values * scale, vectors)
        covariance, repair = nearest_psd(np.asarray(matrix, dtype=float))
        ids = list(model.factor_ids)
        exposures = np.full((len(securities), len(ids)), np.nan)
        specific = np.full(len(securities), np.nan)
        for j, code in enumerate(securities):
            vector = model.exposures.get(code)
            if vector is None:
                continue
            exposures[j] = [vector.exposures.get(fid, 0.0) for fid in ids]
            specific[j] = vector.specific.by_issuer[code]
        return cls(covariance, exposures, specific, float(periods), repair, eigenbasis)

    @property
    def has_forecast(self) -> np.ndarray:
        return np.isfinite(self.specific) & np.isfinite(self.exposures).all(axis=1)

    def variances(self, weights: np.ndarray) -> np.ndarray:
        """Each portfolio's annual forecast variance, one row of `weights` over the securities
        per portfolio. Every security a portfolio holds must have a forecast."""
        weights = np.atleast_2d(np.asarray(weights, dtype=float))
        held = (weights != 0.0).any(axis=0)
        if (held & ~self.has_forecast).any():
            raise ValueError("a portfolio holds a security with no forecast")
        w = weights[:, held]
        x = w @ self.exposures[held]
        factor = np.einsum("pk,kl,pl->p", x, self.covariance, x)
        return factor + (w**2) @ (self.specific[held] ** 2)

    def security_variances(self) -> np.ndarray:
        """Each security's own annual forecast variance; nan where it has none."""
        x = self.exposures
        return np.einsum("jk,kl,jl->j", x, self.covariance, x) + self.specific**2


def _inverse_times_ones(x: np.ndarray, s2: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """V^-1 1 for V = X F X' + diag(s2), by the push-through identity
    V^-1 = D^-1 - D^-1 X (I + F X'D^-1 X)^-1 F X' D^-1, which needs no inverse of F: the
    covariance may be singular, as a repaired one nearly is."""
    d_inv = 1.0 / s2
    k = covariance.shape[0]
    middle = np.eye(k) + covariance @ (x.T @ (x * d_inv[:, None]))
    core = np.linalg.solve(middle, covariance @ (x.T @ d_inv))
    return d_inv - d_inv * (x @ core)


@dataclass(frozen=True)
class MinimumVariance:
    """A minimum-variance portfolio and how well its solver did."""

    weights: np.ndarray
    held: int
    # The KKT conditions' worst relative violation: on the portfolio's support every
    # security's marginal variance (V w)_i equals w'Vw, and off it is no smaller.
    violation: float


def minimum_variance(
    forecast: RiskForecast, eligible: np.ndarray, *, long_only: bool = True
) -> MinimumVariance:
    """The fully invested portfolio of the eligible securities with the least forecast risk.

    Unconstrained it is V^-1 1 / 1'V^-1 1, closed form. Long-only it is the non-negative
    least-squares problem min ||A w - b|| over w >= 0, with A'A = V -- A stacks the covariance's
    square root times the exposures over the specific volatilities -- and A'b = 1, whose
    solution, rescaled to sum to one, is the long-only minimum-variance portfolio: both have
    (V w)_i equal on the support and no smaller off it. Lawson and Hanson's active-set method,
    `scipy.optimize.nnls`, solves it exactly.
    """
    eligible = np.asarray(eligible, dtype=bool) & forecast.has_forecast
    idx = np.flatnonzero(eligible)
    x = forecast.exposures[idx]
    s2 = forecast.specific[idx] ** 2
    covariance = forecast.covariance
    ones = _inverse_times_ones(x, s2, covariance)
    if long_only:
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
        root = vectors * np.sqrt(np.clip(values, 0.0, None))
        a = np.vstack([(x @ root).T, np.diag(np.sqrt(s2))])
        b = a @ ones
        solution, _ = nnls(a, b, maxiter=50 * idx.size)
        w = solution / solution.sum()
    else:
        w = ones / ones.sum()
    full = np.zeros(forecast.specific.size)
    full[idx] = w
    marginal = x @ (covariance @ (x.T @ w)) + s2 * w
    level = float(w @ marginal)
    support = w > 0.0 if long_only else np.ones(w.size, dtype=bool)
    gaps = np.abs(marginal[support] - level)
    if long_only and (~support).any():
        gaps = np.concatenate([gaps, np.clip(level - marginal[~support], 0.0, None)])
    return MinimumVariance(
        weights=full,
        held=int(np.count_nonzero(w > 0.0) if long_only else w.size),
        violation=float(gaps.max() / level) if level > 0.0 else float("nan"),
    )


def eigenportfolios(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The covariance's eigenvalues, largest first, and its eigenvectors as columns, each
    signed so its largest component is positive."""
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order].copy()
    lead = vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])]
    vectors[:, lead < 0.0] *= -1.0
    return values, vectors


def buy_and_hold(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each portfolio's return over a block of periods, bought at its start and held: each
    security's returns compounded, a missing one held as cash, then weighted."""
    growth = np.prod(1.0 + np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0), axis=0)
    return np.atleast_2d(weights) @ (growth - 1.0)


@dataclass(frozen=True)
class BiasRun:
    """Every forecast and outcome a test produced, by date."""

    dates: tuple[date, ...]
    # The named portfolios, each a row: daily forecast volatility and next-day return.
    names: tuple[str, ...]
    forecast: np.ndarray
    realised: np.ndarray
    # Every member singly: dates by securities, nan where not held that day.
    member_forecast: np.ndarray
    member_realised: np.ndarray
    # The eigenportfolios by rank, largest eigenvalue first: factors by dates.
    eigen_forecast: np.ndarray
    eigen_realised: np.ndarray
    # Each horizon h evaluated past one period: every forecast date's outcome over the h
    # periods after it, bought and held, the windows overlapping -- nan where the test has
    # fewer than h periods left -- for the portfolios, the members and the eigenportfolios, in
    # the shapes above. The forecast at h is the one-period one times sqrt(h).
    horizons: tuple[int, ...]
    horizon_realised: Mapping[int, np.ndarray]
    member_horizon_realised: Mapping[int, np.ndarray]
    eigen_horizon_realised: Mapping[int, np.ndarray]
    # Per date: whether the engine's PSD repair bound, and the minimum-variance portfolios'
    # sizes and KKT violations.
    repaired: np.ndarray
    min_variance_held: Mapping[str, np.ndarray]
    min_variance_violation: Mapping[str, np.ndarray]
    # Member-days with a forecast and no return the next day, left out of every portfolio.
    dropped: int
    securities: tuple[str, ...]
    random_seed: int

    def _at(self, forecast, realised, by_horizon, horizon, overlapping, axis):
        if horizon == 1:
            return forecast, realised
        if horizon not in self.horizons:
            raise KeyError(f"horizon {horizon} was not evaluated; this run has "
                           f"{(1,) + self.horizons}")
        forecast, realised = forecast * np.sqrt(horizon), by_horizon[horizon]
        if not overlapping:
            # Every h-th forecast date from the first: the windows apart.
            apart = np.arange(0, len(self.dates), horizon)
            forecast = np.take(forecast, apart, axis=axis)
            realised = np.take(realised, apart, axis=axis)
        return forecast, realised

    def portfolio_outcomes(self, horizon: int = 1, *, overlapping: bool = True):
        """The portfolios' forecasts and outcomes at a horizon, names by dates; apart, every
        h-th forecast date's from the first."""
        return self._at(self.forecast, self.realised, self.horizon_realised, horizon,
                        overlapping, 1)

    def member_outcomes(self, horizon: int = 1, *, overlapping: bool = True):
        """Every member's forecasts and outcomes at a horizon, dates by securities."""
        return self._at(self.member_forecast, self.member_realised,
                        self.member_horizon_realised, horizon, overlapping, 0)

    def eigen_outcomes(self, horizon: int = 1, *, overlapping: bool = True):
        """The eigenportfolios' forecasts and outcomes at a horizon, factors by dates."""
        return self._at(self.eigen_forecast, self.eigen_realised, self.eigen_horizon_realised,
                        horizon, overlapping, 1)

    def statistics(
        self, name: str, *, horizon: int = 1, overlapping: bool = True
    ) -> BiasStatistics:
        """A named portfolio's statistics at a horizon: at one period the daily statistic;
        past it the windows overlapping, their band simulated, or apart."""
        row = self.names.index(name)
        forecast, realised = self.portfolio_outcomes(horizon, overlapping=overlapping)
        return bias_statistics(forecast[row], realised[row],
                               overlap=horizon if overlapping else 1)


def _random_weights(keys, sizes, eligible_idx, count, long_short):
    """Each random portfolio's weights over the eligible securities: the `count` with the
    smallest keys, weighted by their sizes; long-short, the first `count` long and the next
    `count` short, each side summing to one."""
    order = np.argsort(keys[:, eligible_idx], axis=1)
    weights = np.zeros((keys.shape[0], keys.shape[1]))
    rows = np.arange(keys.shape[0])[:, None]
    longs = eligible_idx[order[:, :count]]
    w = sizes[rows, longs]
    weights[rows, longs] = w / w.sum(axis=1, keepdims=True)
    if long_short:
        shorts = eligible_idx[order[:, count : 2 * count]]
        w = sizes[rows, shorts]
        weights[rows, shorts] = -w / w.sum(axis=1, keepdims=True)
    return weights


class BiasTest:
    """The out-of-sample test, one forecast at a time (`evaluate` runs it over an iterable).

    `returns`, `membership` and `capitalisations` are dates by securities on the panel's rows;
    `factor_returns` dates by factors on the same rows, nan where the period was not estimated.
    `groups` names boolean masks over the securities, each a portfolio of its members,
    cap-weighted -- the industries. Each date's portfolios hold its members with a forecast, a
    positive capitalisation and a return the next day:

    - the index, cap-weighted and equally weighted;
    - the `smallest` and the `largest` by capitalisation, equally weighted -- the size pair;
    - each group, cap-weighted;
    - the minimum-variance portfolio, long-only and unconstrained;
    - `random_portfolios` long-only portfolios of `random_names` members and as many long-short
      ones of `random_names` a side, each member's key and weight drawn once from `seed`, so a
      portfolio keeps its names from day to day but for the index's changes;
    - every member singly, and the covariance's eigenportfolios against the factor returns --
      or the forecast's own `eigenbasis` where it carries one.

    Every forecast date is also tested over each horizon h past one period -- every portfolio
    and member bought at its forecast date and held h periods, the eigenportfolios against the
    sum of the h periods' factor returns -- the windows overlapping; every h-th from the first
    is the windows apart (`BiasRun.statistics`). Several tests built alike draw the same random
    portfolios, so forecasts of the same dates -- one simulation's eigenfactor adjustments at
    several scales, say -- can be tested side by side in one pass.
    """

    def __init__(
        self,
        *,
        dates: Sequence[date],
        securities: Sequence[str],
        returns: np.ndarray,
        membership: np.ndarray,
        capitalisations: np.ndarray,
        groups: Mapping[str, np.ndarray],
        factor_returns: np.ndarray,
        smallest: int = 85,
        largest: int = 50,
        random_portfolios: int = 100,
        random_names: int = 30,
        seed: int = 11,
        horizons: Sequence[int] = (21,),
    ) -> None:
        self.dates = dates
        self.securities = tuple(securities)
        self.returns = returns
        self.membership = membership
        self.capitalisations = capitalisations
        self.groups = groups
        self.factor_returns = factor_returns
        self.smallest = smallest
        self.largest = largest
        self.random_portfolios = random_portfolios
        self.random_names = random_names
        self.seed = seed
        n_secs = len(self.securities)
        self.n_secs = n_secs
        self.horizons = tuple(sorted({int(h) for h in horizons if int(h) > 1}))
        self.n_factors = factor_returns.shape[1]
        rng = np.random.default_rng(seed)
        self.long_keys, self.long_sizes = rng.random((random_portfolios, n_secs)), rng.random(
            (random_portfolios, n_secs)
        )
        self.short_keys, self.short_sizes = rng.random((random_portfolios, n_secs)), rng.random(
            (random_portfolios, n_secs)
        )
        self.group_names = list(groups)
        self.names = (
            ["the index, cap-weighted", "the index, equally weighted",
             f"smallest {smallest}, equally weighted", f"largest {largest}, equally weighted"]
            + [f"industry: {g}" for g in self.group_names]
            + ["minimum variance, long-only", "minimum variance, unconstrained"]
            + [f"random long-only {i}" for i in range(random_portfolios)]
            + [f"random long-short {i}" for i in range(random_portfolios)]
        )
        self.rows_seen: list[int] = []
        self.out_forecast: list[np.ndarray] = []
        self.out_realised: list[np.ndarray] = []
        self.member_forecast: list[np.ndarray] = []
        self.member_realised: list[np.ndarray] = []
        self.eigen_forecast: list[np.ndarray] = []
        self.eigen_realised: list[np.ndarray] = []
        self.h_realised: dict[int, list[np.ndarray]] = {h: [] for h in self.horizons}
        self.mh_realised: dict[int, list[np.ndarray]] = {h: [] for h in self.horizons}
        self.eh_realised: dict[int, list[np.ndarray]] = {h: [] for h in self.horizons}
        self.repaired: list[bool] = []
        self.held = {"long-only": [], "unconstrained": []}
        self.violation = {"long-only": [], "unconstrained": []}
        self.dropped = 0
        self.first_row: int | None = None
        self.last_row = len(dates) - 1

    def add(self, t: int, forecast: RiskForecast) -> None:
        """The forecast as of row t against what followed it."""
        names, n_secs, returns = self.names, self.n_secs, self.returns
        if self.first_row is None:
            self.first_row = t
        if t >= self.last_row:
            raise ValueError(f"the forecast as of row {t} has no next period to realise in")
        nxt = returns[t + 1]
        caps = self.capitalisations[t]
        known = self.membership[t] & forecast.has_forecast
        self.dropped += int(np.count_nonzero(known & ~np.isfinite(nxt)))
        eligible = known & np.isfinite(nxt) & np.isfinite(caps) & (caps > 0.0)
        idx = np.flatnonzero(eligible)
        weights = np.zeros((len(names), n_secs))
        cap_w = np.where(eligible, caps, 0.0)
        weights[0] = cap_w / cap_w.sum()
        weights[1, idx] = 1.0 / idx.size
        by_cap = idx[np.argsort(caps[idx], kind="stable")]
        weights[2, by_cap[: self.smallest]] = 1.0 / min(self.smallest, by_cap.size)
        weights[3, by_cap[-self.largest :]] = 1.0 / min(self.largest, by_cap.size)
        for g, group in enumerate(self.group_names):
            mask = eligible & self.groups[group]
            if mask.any():
                weights[4 + g] = np.where(mask, caps, 0.0) / caps[mask].sum()
        base = 4 + len(self.group_names)
        for k, label in enumerate(("long-only", "unconstrained")):
            portfolio = minimum_variance(forecast, eligible, long_only=label == "long-only")
            weights[base + k] = portfolio.weights
            self.held[label].append(portfolio.held)
            self.violation[label].append(portfolio.violation)
        start = base + 2
        weights[start : start + self.random_portfolios] = _random_weights(
            self.long_keys, self.long_sizes, idx, self.random_names, False
        )
        weights[start + self.random_portfolios :] = _random_weights(
            self.short_keys, self.short_sizes, idx, self.random_names, True
        )
        live = np.abs(weights).sum(axis=1) > 0.0
        variance = np.full(len(names), np.nan)
        variance[live] = forecast.variances(weights[live])
        scale = forecast.periods_per_year
        self.out_forecast.append(np.sqrt(variance / scale))
        self.out_realised.append(np.where(live, weights @ np.nan_to_num(nxt, nan=0.0), np.nan))
        own = np.sqrt(forecast.security_variances() / scale)
        self.member_forecast.append(np.where(eligible, own, np.nan))
        self.member_realised.append(np.where(eligible, nxt, np.nan))
        if forecast.eigenbasis is None:
            values, vectors = eigenportfolios(forecast.covariance)
        else:
            values, vectors = forecast.eigenbasis
        self.eigen_forecast.append(np.sqrt(np.clip(values, 0.0, None) / scale))
        self.eigen_realised.append(vectors.T @ self.factor_returns[t + 1])
        self.repaired.append(forecast.repair.repaired)
        for h in self.horizons:
            if t + h > self.last_row:
                self.h_realised[h].append(np.full(len(names), np.nan))
                self.mh_realised[h].append(np.full(n_secs, np.nan))
                self.eh_realised[h].append(np.full(self.n_factors, np.nan))
                continue
            block = returns[t + 1 : t + 1 + h]
            self.h_realised[h].append(np.where(live, buy_and_hold(block, weights), np.nan))
            growth = np.prod(1.0 + np.nan_to_num(block, nan=0.0), axis=0) - 1.0
            self.mh_realised[h].append(np.where(eligible, growth, np.nan))
            self.eh_realised[h].append(
                vectors.T @ np.nansum(self.factor_returns[t + 1 : t + 1 + h], axis=0)
            )
        self.rows_seen.append(t)

    def result(self) -> BiasRun:
        """Every forecast and outcome added so far."""
        names, n_factors, n_secs, horizons = self.names, self.n_factors, self.n_secs, self.horizons

        def stack(items, width):
            return np.array(items).T if items else np.empty((width, 0))

        return BiasRun(
            dates=tuple(self.dates[t] for t in self.rows_seen),
            names=tuple(names),
            forecast=stack(self.out_forecast, len(names)),
            realised=stack(self.out_realised, len(names)),
            member_forecast=np.array(self.member_forecast),
            member_realised=np.array(self.member_realised),
            eigen_forecast=stack(self.eigen_forecast, n_factors),
            eigen_realised=stack(self.eigen_realised, n_factors),
            horizons=horizons,
            horizon_realised={h: stack(self.h_realised[h], len(names)) for h in horizons},
            member_horizon_realised={h: np.array(self.mh_realised[h]).reshape(-1, n_secs)
                                     for h in horizons},
            eigen_horizon_realised={h: stack(self.eh_realised[h], n_factors) for h in horizons},
            repaired=np.array(self.repaired, dtype=bool),
            min_variance_held={k: np.array(v) for k, v in self.held.items()},
            min_variance_violation={k: np.array(v) for k, v in self.violation.items()},
            dropped=self.dropped,
            securities=self.securities,
            random_seed=self.seed,
        )


def evaluate(
    forecasts: Iterable[tuple[int, RiskForecast]],
    *,
    progress: Callable[[int, int], None] | None = None,
    total: int | None = None,
    **test,
) -> BiasRun:
    """Every forecast in `forecasts` -- (row, forecast as of that row) on the panel's rows --
    against the next row's outcome, and over each of `horizons` (in periods), for a standard
    set of portfolios: a `BiasTest` built from the rest of the arguments, run over them."""
    run = BiasTest(**test)
    for count, (t, forecast) in enumerate(forecasts):
        run.add(t, forecast)
        if progress is not None:
            progress(count + 1, total or 0)
    return run.result()
