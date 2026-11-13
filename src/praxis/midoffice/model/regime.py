"""The volatility regime adjustment (Menchero, Orr and Wang 2011; Barra USE4).

A covariance estimated over a long window forecasts the average volatility of the window behind
it, so it lags a change of regime: after a calm spell it forecasts too little when the market
turns, and after a storm too much once it calms -- every portfolio with it, the broad ones most.
The S&P 500 model's bias statistics show it by halves (midoffice13 and 14): the cap-weighted
index 1.22 in the test's first half and 0.75 in its second, the minimum-variance portfolios 1.25
and 1.30 then 0.83 and 0.86. A shape adjustment cannot move a level. This one reads the model's
own out-of-sample record, day by day, and scales its forecast by it.

**The bias statistics.** On each day t, the factors' cross-sectional bias statistic is the root
mean square, over the K factors, of each factor's return that day over its volatility as
forecast at the close of the day before:

    B_F(t) = sqrt( (1/K) sum_k ( f_k(t) / sigma_k(t-1) )^2 ),

and the specific returns', over the members, cap-weighted:

    B_S(t) = sqrt( sum_n w_n ( u_n(t) / s_n(t-1) )^2 ),    sum_n w_n = 1,

u_n(t) = r_n(t) - x_n(t-1)' f(t) the member's specific return on its exposures as forecast, the
weights its capitalisation at t - 1 over the members with a forecast, a capitalisation and a
return -- the cap-weighted index's own. Both are one on average when the forecasts are right;
above one, they were too low.

**The multiplier** as of a day T is the square root of the exponentially weighted mean of the
squared statistic over the days through T, each side on its own half-life h:

    lambda(T)^2 = sum_{t <= T} 0.5^((T - t) / h) B(t)^2 / sum_{t <= T} 0.5^((T - t) / h),

and the adjusted forecast is the factor covariance times lambda_F(T)^2 and each specific variance
times lambda_S(T)^2 (`bias.RiskForecast.regime_adjusted`). Out of sample throughout: B(t) is read
from the forecast made at t - 1's close, unadjusted, and the returns of day t, so lambda(T) uses
nothing after T's close -- the factor returns of day T are estimated at it.

**Before the history is long enough the multiplier is one.** The history starts the day after
the first forecast the model makes, and a side's multiplier waits for `REGIME_MIN_DAYS` days of
it: before then the forecast is the model's own, unadjusted. The exponential weights are
normalised over the days there are, so from then on lambda is a weighted mean however short the
history; a half-life long against the history is near equal weights over it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np

from .bias import RiskForecast
from .crosssectional import cross_sectional_models_as_of
from .panel import ReturnPanel

REGIME_MIN_DAYS = 21
"""Days of history a multiplier waits for, a month: before then it is one, the forecast
unadjusted. The shortest half-life midoffice15 tests, so every half-life starts on the same day."""

REGIME_HALF_LIVES = (21.0, 42.0, 84.0)
"""The half-lives midoffice15's brief tests on each side, in trading days."""

SIDES = ("factor", "specific")


def factor_bias(factor_returns: np.ndarray, volatilities: np.ndarray) -> tuple[float, int]:
    """B_F^2 for one day, and how many factors it is over: the mean, over the factors with a
    finite return and a positive forecast volatility, of (f_k / sigma_k)^2, both at the period's
    own scale. nan, 0 where there is none."""
    f = np.asarray(factor_returns, dtype=float)
    sigma = np.asarray(volatilities, dtype=float)
    if f.shape != sigma.shape:
        raise ValueError(f"{f.size} factor returns against {sigma.size} volatilities")
    ok = np.isfinite(f) & np.isfinite(sigma) & (sigma > 0.0)
    if not ok.any():
        return float("nan"), 0
    z = f[ok] / sigma[ok]
    return float(np.mean(z * z)), int(ok.sum())


def specific_bias(
    specific_returns: np.ndarray, volatilities: np.ndarray, weights: np.ndarray
) -> tuple[float, int, float]:
    """B_S^2 for one day, how many securities it is over, and their effective number: the
    weighted mean, over the securities with a finite specific return, a positive forecast
    volatility and a positive weight, of (u_n / s_n)^2, the weights normalised over them; the
    effective number (sum w)^2 / sum w^2. nan, 0, 0 where there is none."""
    u = np.asarray(specific_returns, dtype=float)
    s = np.asarray(volatilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not u.shape == s.shape == w.shape:
        raise ValueError("specific returns, volatilities and weights must be one per security")
    ok = np.isfinite(u) & np.isfinite(s) & (s > 0.0) & np.isfinite(w) & (w > 0.0)
    if not ok.any():
        return float("nan"), 0, 0.0
    weight = w[ok] / w[ok].sum()
    z = u[ok] / s[ok]
    return float(weight @ (z * z)), int(ok.sum()), float(1.0 / (weight @ weight))


def regime_multiplier(
    squared: Sequence[float], half_life: float | None, min_days: int = REGIME_MIN_DAYS
) -> float:
    """lambda as of the last of `squared` -- the daily squared statistics, oldest first: the square
    root of their mean weighted 0.5 ** (age / half_life), age 0 the last. One for a side that is
    off (`half_life` None) and while the history is shorter than `min_days`."""
    if half_life is None:
        return 1.0
    if not half_life > 0.0:
        raise ValueError(f"a half-life is positive, or None for the side off; got {half_life!r}")
    values = np.asarray(squared, dtype=float)
    if values.size < max(int(min_days), 1):
        return 1.0
    ages = np.arange(values.size - 1, -1, -1, dtype=float)
    weights = 0.5 ** (ages / half_life)
    return float(math.sqrt(float(weights @ values) / float(weights.sum())))


@dataclass
class RegimeHistory:
    """The volatility regime adjustment's history: each day's two bias statistics, squared, out of
    sample, and the multipliers they give as of the last day added (`regime_multiplier`).

    `add` takes day t's statistics from the forecast made at the close of t - 1 -- unadjusted --
    the factor returns estimated on day t, the securities' returns on day t, and t - 1's
    capitalisations and membership, which set the specific side's weights."""

    min_days: int = REGIME_MIN_DAYS
    dates: list[date] = field(default_factory=list)
    factor_squared: list[float] = field(default_factory=list)
    specific_squared: list[float] = field(default_factory=list)
    factors: list[int] = field(default_factory=list)
    members: list[int] = field(default_factory=list)
    effective: list[float] = field(default_factory=list)

    def add(
        self,
        day: date,
        previous: RiskForecast,
        factor_returns: np.ndarray,
        returns: np.ndarray,
        capitalisations: np.ndarray,
        membership: np.ndarray | None = None,
    ) -> None:
        """Day t's statistics. `previous` is the forecast as of t - 1 on the securities `returns`
        are over; `factor_returns` its factors' returns on day t, in its order; `capitalisations`
        and `membership` t - 1's, one per security (membership None: every security)."""
        if self.dates and day <= self.dates[-1]:
            raise ValueError(f"the history runs forward: {day} after {self.dates[-1]}")
        f = np.asarray(factor_returns, dtype=float)
        if f.shape != (previous.covariance.shape[0],):
            raise ValueError(f"{f.size} factor returns for a forecast of "
                             f"{previous.covariance.shape[0]} factors")
        periods = previous.periods_per_year
        sigma = np.sqrt(np.clip(np.diag(previous.covariance), 0.0, None) / periods)
        factor, k = factor_bias(f, sigma)
        returns = np.asarray(returns, dtype=float)
        caps = np.asarray(capitalisations, dtype=float)
        held = previous.has_forecast & np.isfinite(returns) & np.isfinite(caps) & (caps > 0.0)
        if membership is not None:
            held &= np.asarray(membership, dtype=bool)
        x = np.where(held[:, None], previous.exposures, 0.0)
        u = np.where(held, returns - x @ f, np.nan)
        s = previous.specific / math.sqrt(periods)
        specific, n, effective = specific_bias(u, s, np.where(held, caps, 0.0))
        if not (np.isfinite(factor) and np.isfinite(specific)):
            raise ValueError(f"{day}: no factor or no member to read a bias statistic from")
        self.dates.append(day)
        self.factor_squared.append(factor)
        self.specific_squared.append(specific)
        self.factors.append(k)
        self.members.append(n)
        self.effective.append(effective)

    def add_refit(
        self,
        day: date,
        previous: RiskForecast,
        refit,
        returns: np.ndarray,
        capitalisations: np.ndarray,
        membership: np.ndarray | None = None,
    ) -> bool:
        """Day t's statistics as the bias test reads them: `refit` the model refitted as of t, its
        last estimated period day t's own -- the factor returns that day -- and `previous` the
        forecast as of t - 1. False, and nothing added, where day t was not estimated."""
        factors = refit.notes.factor_returns
        if factors is None or not factors.dates or factors.dates[-1] != day:
            return False
        self.add(day, previous, factors.returns[-1], returns, capitalisations, membership)
        return True

    def __len__(self) -> int:
        return len(self.dates)

    def squared(self, side: str) -> list[float]:
        if side not in SIDES:
            raise ValueError(f"a side is one of {SIDES}, not {side!r}")
        return self.factor_squared if side == "factor" else self.specific_squared

    def multiplier(self, side: str, half_life: float | None) -> float:
        """lambda on `side` as of the last day added, on `half_life` (None: the side off, one)."""
        return regime_multiplier(self.squared(side), half_life, self.min_days)

    def path(self, side: str, half_life: float | None) -> np.ndarray:
        """lambda on `side` as of each day added, in order: the multiplier each day's forecast
        would carry had it been made at that day's close."""
        values = self.squared(side)
        return np.array([regime_multiplier(values[: i + 1], half_life, self.min_days)
                         for i in range(len(values))])


def regime_history(
    returns: ReturnPanel,
    characteristics,
    as_of: Sequence[date],
    *,
    capitalisations: np.ndarray,
    membership: np.ndarray | None = None,
    min_days: int = REGIME_MIN_DAYS,
    **options,
) -> RegimeHistory:
    """The history over `as_of`, consecutive dates of the return panel, as the bias test reads it:
    the model refitted as of each (`crosssectional.cross_sectional_models_as_of`, with
    `capitalisations`, `membership` and `options` its options), each day's statistics read from
    the forecast as of the date before and the refit as of the day. The first date has none. The
    multiplier as of a date is the history through it, which nothing after it moves."""
    at = {d: i for i, d in enumerate(returns.dates)}
    history = RegimeHistory(min_days=min_days)
    previous: tuple[int, RiskForecast] | None = None
    models = cross_sectional_models_as_of(returns, characteristics, as_of,
                                          capitalisations=capitalisations,
                                          membership=membership, **options)
    for day, model in models:
        t = at[day]
        if previous is not None:
            row, forecast = previous
            if row != t - 1:
                raise ValueError(f"the dates must be consecutive: {day} follows "
                                 f"{returns.dates[row]}")
            history.add_refit(day, forecast, model, returns.returns[t], capitalisations[t - 1],
                              None if membership is None else membership[t - 1])
        previous = (t, RiskForecast.from_model(model, returns.securities))
    return history
