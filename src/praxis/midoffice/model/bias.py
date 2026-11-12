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

**The eigenportfolios realise against estimated factor returns,** which carry each period's
estimation noise: they test whether the covariance's eigenvalues are biased -- the case for
Barra's eigenfactor adjustment -- and cannot judge a correction for that noise, which real
portfolios can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import nnls

from ..risk.basis import PSDRepair, nearest_psd

ROLLING_WINDOW = 252
"""Twelve months of daily forecasts: the rolling windows' length."""


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
        if not np.isfinite(self.kurtosis) or self.n < 1:
            return float("nan"), float("nan")
        half = 1.96 * float(np.sqrt(max(self.kurtosis - 1.0, 0.0) / (4.0 * self.n)))
        return 1.0 - half, 1.0 + half


def bias_statistics(
    forecast: np.ndarray, realised: np.ndarray, *, window: int = ROLLING_WINDOW
) -> BiasStatistics:
    """The statistic of realised / forecast over the outcomes where both are finite and the
    forecast is positive, in date order."""
    forecast = np.asarray(forecast, dtype=float)
    realised = np.asarray(realised, dtype=float)
    ok = np.isfinite(forecast) & np.isfinite(realised) & (forecast > 0.0)
    z = np.where(ok, realised / np.where(ok, forecast, 1.0), np.nan)
    finite = z[ok]
    n = int(finite.size)
    low, high = bias_band(n)
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

    @classmethod
    def from_model(cls, model, securities: Sequence[str]) -> "RiskForecast":
        """A `FactorModel`'s forecast on `securities`, in its factor order."""
        notes = model.notes
        periods = notes.frequency.periods_per_year if notes.annualised else 1.0
        covariance, repair = nearest_psd(np.asarray(model.covariance.matrix, dtype=float))
        ids = list(model.factor_ids)
        exposures = np.full((len(securities), len(ids)), np.nan)
        specific = np.full(len(securities), np.nan)
        for j, code in enumerate(securities):
            vector = model.exposures.get(code)
            if vector is None:
                continue
            exposures[j] = [vector.exposures.get(fid, 0.0) for fid in ids]
            specific[j] = vector.specific.by_issuer[code]
        return cls(covariance, exposures, specific, float(periods), repair)

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
    # The non-overlapping horizon: its forecast dates, and the same three at its scale.
    horizon: int
    horizon_dates: tuple[date, ...]
    horizon_forecast: np.ndarray
    horizon_realised: np.ndarray
    member_horizon_forecast: np.ndarray
    member_horizon_realised: np.ndarray
    eigen_horizon_forecast: np.ndarray
    eigen_horizon_realised: np.ndarray
    # Per date: whether the engine's PSD repair bound, and the minimum-variance portfolios'
    # sizes and KKT violations.
    repaired: np.ndarray
    min_variance_held: Mapping[str, np.ndarray]
    min_variance_violation: Mapping[str, np.ndarray]
    # Member-days with a forecast and no return the next day, left out of every portfolio.
    dropped: int
    securities: tuple[str, ...]
    random_seed: int

    def statistics(self, name: str, *, horizon: bool = False) -> BiasStatistics:
        row = self.names.index(name)
        if horizon:
            return bias_statistics(self.horizon_forecast[row], self.horizon_realised[row])
        return bias_statistics(self.forecast[row], self.realised[row])


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


def evaluate(
    forecasts: Iterable[tuple[int, RiskForecast]],
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
    horizon: int = 21,
    progress: Callable[[int, int], None] | None = None,
    total: int | None = None,
) -> BiasRun:
    """Every forecast in `forecasts` -- (row, forecast as of that row) on the panel's rows --
    against the next row's outcome, for a standard set of portfolios.

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
    - every member singly, and the covariance's eigenportfolios against the factor returns.

    Every `horizon`-th forecast date, from the first, is also tested over the `horizon` periods
    after it, bought and held: the non-overlapping check.
    """
    securities = tuple(securities)
    n_secs = len(securities)
    rng = np.random.default_rng(seed)
    long_keys, long_sizes = rng.random((random_portfolios, n_secs)), rng.random(
        (random_portfolios, n_secs)
    )
    short_keys, short_sizes = rng.random((random_portfolios, n_secs)), rng.random(
        (random_portfolios, n_secs)
    )
    group_names = list(groups)
    names = (
        ["the index, cap-weighted", "the index, equally weighted",
         f"smallest {smallest}, equally weighted", f"largest {largest}, equally weighted"]
        + [f"industry: {g}" for g in group_names]
        + ["minimum variance, long-only", "minimum variance, unconstrained"]
        + [f"random long-only {i}" for i in range(random_portfolios)]
        + [f"random long-short {i}" for i in range(random_portfolios)]
    )
    rows_seen: list[int] = []
    out_forecast: list[np.ndarray] = []
    out_realised: list[np.ndarray] = []
    member_forecast: list[np.ndarray] = []
    member_realised: list[np.ndarray] = []
    eigen_forecast: list[np.ndarray] = []
    eigen_realised: list[np.ndarray] = []
    horizon_rows: list[int] = []
    h_forecast: list[np.ndarray] = []
    h_realised: list[np.ndarray] = []
    mh_forecast: list[np.ndarray] = []
    mh_realised: list[np.ndarray] = []
    eh_forecast: list[np.ndarray] = []
    eh_realised: list[np.ndarray] = []
    repaired: list[bool] = []
    held = {"long-only": [], "unconstrained": []}
    violation = {"long-only": [], "unconstrained": []}
    dropped = 0
    first_row: int | None = None
    last_row = len(dates) - 1
    for count, (t, forecast) in enumerate(forecasts):
        if first_row is None:
            first_row = t
        if t >= last_row:
            raise ValueError(f"the forecast as of row {t} has no next period to realise in")
        nxt = returns[t + 1]
        caps = capitalisations[t]
        known = membership[t] & forecast.has_forecast
        dropped += int(np.count_nonzero(known & ~np.isfinite(nxt)))
        eligible = known & np.isfinite(nxt) & np.isfinite(caps) & (caps > 0.0)
        idx = np.flatnonzero(eligible)
        weights = np.zeros((len(names), n_secs))
        cap_w = np.where(eligible, caps, 0.0)
        weights[0] = cap_w / cap_w.sum()
        weights[1, idx] = 1.0 / idx.size
        by_cap = idx[np.argsort(caps[idx], kind="stable")]
        weights[2, by_cap[:smallest]] = 1.0 / min(smallest, by_cap.size)
        weights[3, by_cap[-largest:]] = 1.0 / min(largest, by_cap.size)
        for g, group in enumerate(group_names):
            mask = eligible & groups[group]
            if mask.any():
                weights[4 + g] = np.where(mask, caps, 0.0) / caps[mask].sum()
        base = 4 + len(group_names)
        for k, label in enumerate(("long-only", "unconstrained")):
            portfolio = minimum_variance(forecast, eligible, long_only=label == "long-only")
            weights[base + k] = portfolio.weights
            held[label].append(portfolio.held)
            violation[label].append(portfolio.violation)
        start = base + 2
        weights[start : start + random_portfolios] = _random_weights(
            long_keys, long_sizes, idx, random_names, False
        )
        weights[start + random_portfolios :] = _random_weights(
            short_keys, short_sizes, idx, random_names, True
        )
        live = np.abs(weights).sum(axis=1) > 0.0
        variance = np.full(len(names), np.nan)
        variance[live] = forecast.variances(weights[live])
        scale = forecast.periods_per_year
        out_forecast.append(np.sqrt(variance / scale))
        out_realised.append(np.where(live, weights @ np.nan_to_num(nxt, nan=0.0), np.nan))
        own = np.sqrt(forecast.security_variances() / scale)
        member_forecast.append(np.where(eligible, own, np.nan))
        member_realised.append(np.where(eligible, nxt, np.nan))
        values, vectors = eigenportfolios(forecast.covariance)
        eigen_forecast.append(np.sqrt(np.clip(values, 0.0, None) / scale))
        eigen_realised.append(vectors.T @ factor_returns[t + 1])
        repaired.append(forecast.repair.repaired)
        if (t - first_row) % horizon == 0 and t + horizon <= last_row:
            block = returns[t + 1 : t + 1 + horizon]
            horizon_rows.append(t)
            h_forecast.append(out_forecast[-1] * np.sqrt(horizon))
            h_realised.append(np.where(live, buy_and_hold(block, weights), np.nan))
            growth = np.prod(1.0 + np.nan_to_num(block, nan=0.0), axis=0) - 1.0
            mh_forecast.append(member_forecast[-1] * np.sqrt(horizon))
            mh_realised.append(np.where(eligible, growth, np.nan))
            eh_forecast.append(eigen_forecast[-1] * np.sqrt(horizon))
            eh_realised.append(vectors.T @ np.nansum(factor_returns[t + 1 : t + 1 + horizon],
                                                     axis=0))
        rows_seen.append(t)
        if progress is not None:
            progress(count + 1, total or 0)

    def stack(items, width):
        return np.array(items).T if items else np.empty((width, 0))

    return BiasRun(
        dates=tuple(dates[t] for t in rows_seen),
        names=tuple(names),
        forecast=stack(out_forecast, len(names)),
        realised=stack(out_realised, len(names)),
        member_forecast=np.array(member_forecast),
        member_realised=np.array(member_realised),
        eigen_forecast=stack(eigen_forecast, factor_returns.shape[1]),
        eigen_realised=stack(eigen_realised, factor_returns.shape[1]),
        horizon=horizon,
        horizon_dates=tuple(dates[t] for t in horizon_rows),
        horizon_forecast=stack(h_forecast, len(names)),
        horizon_realised=stack(h_realised, len(names)),
        member_horizon_forecast=np.array(mh_forecast),
        member_horizon_realised=np.array(mh_realised),
        eigen_horizon_forecast=stack(eh_forecast, factor_returns.shape[1]),
        eigen_horizon_realised=stack(eh_realised, factor_returns.shape[1]),
        repaired=np.array(repaired, dtype=bool),
        min_variance_held={k: np.array(v) for k, v in held.items()},
        min_variance_violation={k: np.array(v) for k, v in violation.items()},
        dropped=dropped,
        securities=securities,
        random_seed=seed,
    )
