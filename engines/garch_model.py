"""
engines/garch_model.py
========================
GARCH family models for realized volatility forecasting.

Used by the volatility strategy to produce E[RV] forecasts at
multiple horizons (7d, 14d, 30d) which are compared to implied
vol (from Deribit DVOL) to compute the Volatility Risk Premium (VRP).

Models:
    GARCH(1,1)  — baseline, fast, handles vol clustering
    EGARCH(1,1) — exponential, captures leverage effect (vol spikes more on drops)
    GJR-GARCH   — Glosten-Jagannathan-Runkle, asymmetric response

Input: daily returns (percent) from hourly spot bars, last N days.
Output: annualized vol forecasts at 7d, 14d, 30d horizons (as decimal fractions).

Usage:
    from engines.garch_model import fit_garch_ensemble, GARCHForecast
    fc = fit_garch_ensemble(daily_rets_pct, horizons=[7, 14, 30])
    print(fc.vol_7d, fc.vol_30d)  # e.g. 0.75, 0.72 (75%, 72% annualized)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


@dataclass
class GARCHForecast:
    """Multi-horizon GARCH volatility forecast."""
    model_name:  str
    vol_7d:      float     # annualized vol forecast, 7-day horizon (decimal, e.g. 0.75)
    vol_14d:     float
    vol_30d:     float
    current_vol: float     # current conditional vol (1-step)
    persistence: float     # alpha + beta (closeness to IGARCH)
    aic:         float
    converged:   bool
    params:      dict = field(default_factory=dict)

    def vol_at(self, days: int) -> float:
        """Interpolated vol forecast for arbitrary horizon."""
        if days <= 7:
            return self.vol_7d
        elif days <= 14:
            t = (days - 7) / 7
            return self.vol_7d + t * (self.vol_14d - self.vol_7d)
        elif days <= 30:
            t = (days - 14) / 16
            return self.vol_14d + t * (self.vol_30d - self.vol_14d)
        else:
            return self.vol_30d


@dataclass
class EnsembleForecast:
    """Ensemble of GARCH model forecasts with weighted average."""
    forecasts:   list[GARCHForecast]
    vol_7d:      float    # AIC-weighted ensemble
    vol_14d:     float
    vol_30d:     float
    current_vol: float
    best_model:  str
    vrp_7d:      float = 0.0   # IV_7d - E[RV_7d] (set externally)
    vrp_30d:     float = 0.0   # IV_30d - E[RV_30d]

    def vol_at(self, days: int) -> float:
        if days <= 7:
            return self.vol_7d
        elif days <= 14:
            t = (days - 7) / 7
            return self.vol_7d + t * (self.vol_14d - self.vol_7d)
        elif days <= 30:
            t = (days - 14) / 16
            return self.vol_14d + t * (self.vol_30d - self.vol_14d)
        return self.vol_30d


def _garch_multi_step_forecast(
    omega: float, alpha: float, beta: float,
    h_t: float, eps_t: float,
    horizon: int = 30,
) -> np.ndarray:
    """
    Analytic multi-step GARCH variance forecast.

    h_{t+1} = omega + alpha*eps_t^2 + beta*h_t
    h_{t+k} = omega + (alpha+beta)*h_{t+k-1}  for k >= 2
              (since E[eps^2_{t+k-1}] = h_{t+k-1})

    Returns array of daily variance forecasts h_{t+1}...h_{t+horizon}
    in the same units as h_t (percent-squared if rets are in percent).
    """
    h_fc = np.empty(horizon)
    h = omega + alpha * eps_t**2 + beta * h_t
    h_fc[0] = h
    ab = alpha + beta
    for i in range(1, horizon):
        h = omega + ab * h
        h_fc[i] = h
    return h_fc


def _gjr_multi_step_forecast(
    omega: float, alpha: float, gamma: float, beta: float,
    h_t: float, eps_t: float, horizon: int = 30,
) -> np.ndarray:
    """
    Multi-step GJR-GARCH forecast.
    h_{t+1} = omega + (alpha + gamma/2)*eps_t^2*I(eps_t<0)*2 + beta*h_t
    For k >= 2, E[I(eps<0)] = 0.5:
    h_{t+k} = omega + (alpha + gamma/2)*h_{t+k-1} + beta*h_{t+k-1}
    """
    h_fc = np.empty(horizon)
    indicator = 1.0 if eps_t < 0 else 0.0
    h = omega + alpha * eps_t**2 + gamma * eps_t**2 * indicator + beta * h_t
    h_fc[0] = h
    ab = alpha + gamma * 0.5 + beta  # expected persistence with 50% neg days
    for i in range(1, horizon):
        h = omega + ab * h
        h_fc[i] = h
    return h_fc


def _fit_garch(rets_pct: np.ndarray, model_type: str = "GARCH") -> GARCHForecast | None:
    """
    Fit a single GARCH-family model. Returns None on convergence failure.

    rets_pct: daily percent returns (e.g. [-2.1, 1.3, 0.8, ...])
    """
    try:
        from arch import arch_model

        if model_type == "GARCH":
            am = arch_model(rets_pct, vol="Garch", p=1, q=1,
                           dist="Normal", rescale=True)
        elif model_type == "EGARCH":
            am = arch_model(rets_pct, vol="EGARCH", p=1, q=1,
                           dist="Normal", rescale=True)
        elif model_type == "GJR":
            am = arch_model(rets_pct, vol="GARCH", p=1, o=1, q=1,
                           dist="Normal", rescale=True)
        else:
            return None

        res = am.fit(disp="off", show_warning=False)

        if not np.isfinite(res.aic):
            return None

        # Extract params and current state
        p       = dict(res.params)
        mu      = p.get("mu", 0.0)
        omega   = p["omega"]
        alpha   = p.get("alpha[1]", 0.0)
        beta    = p.get("beta[1]", 0.0)
        gamma   = p.get("gamma[1]", 0.0)   # GJR asymmetry

        scale   = getattr(res, "scale", 1.0)
        cv_last = float(res.conditional_volatility[-1]) / scale
        h_t     = cv_last**2
        eps_t   = float(rets_pct[-1] - mu)

        if model_type == "GJR":
            h_fc = _gjr_multi_step_forecast(
                omega, alpha, gamma, beta, h_t, eps_t, horizon=30
            )
        else:
            h_fc = _garch_multi_step_forecast(
                omega, alpha, beta, h_t, eps_t, horizon=30
            )

        # Convert daily variance forecasts to annualized vol (decimal fraction)
        # Floor forecast at a minimum sensible level
        # (prevents near-zero forecasts when alpha≈0 and omega tiny)
        h_floor = max(h_t * 0.25, 1e-4)  # at least 25% of current vol variance

        def to_ann(h_arr):
            return float(np.sqrt(max(h_arr.mean(), h_floor) * 365)) / 100.0

        return GARCHForecast(
            model_name  = model_type,
            vol_7d      = to_ann(h_fc[:7]),
            vol_14d     = to_ann(h_fc[:14]),
            vol_30d     = to_ann(h_fc[:30]),
            current_vol = float(np.sqrt(h_t * 365)) / 100.0,
            persistence = float(alpha + beta + gamma * 0.5),
            aic         = float(res.aic),
            converged   = True,
            params      = {"omega": omega, "alpha": alpha,
                          "beta": beta, "gamma": gamma},
        )

    except Exception as e:
        logger.debug(f"  GARCH {model_type} failed: {e}")
        return None


def fit_garch_ensemble(
    daily_rets_pct: np.ndarray | pd.Series,
    min_obs: int = 90,
) -> EnsembleForecast | None:
    """
    Fit GARCH(1,1), EGARCH, and GJR-GARCH. Return AIC-weighted ensemble.

    Args:
        daily_rets_pct: Daily returns in PERCENT (e.g. BTC daily ret of -2.3%)
        min_obs: Minimum observations required

    Returns:
        EnsembleForecast or None if all models fail
    """
    if hasattr(daily_rets_pct, "values"):
        rets = daily_rets_pct.dropna().values.astype(float)
    else:
        rets = np.asarray(daily_rets_pct, dtype=float)
        rets = rets[np.isfinite(rets)]

    if len(rets) < min_obs:
        return None

    # Fit all three models
    models = []
    for mtype in ["GARCH", "EGARCH", "GJR"]:
        fc = _fit_garch(rets, mtype)
        if fc is not None:
            models.append(fc)

    if not models:
        return None

    # AIC-weighted ensemble (lower AIC = better)
    # Weight ∝ exp(-0.5 * delta_AIC), Akaike weights
    aics   = np.array([m.aic for m in models])
    d_aic  = aics - aics.min()
    weights = np.exp(-0.5 * d_aic)
    weights /= weights.sum()

    best = models[int(np.argmin(aics))]

    def wt_avg(attr):
        return float(sum(w * getattr(m, attr) for w, m in zip(weights, models)))

    return EnsembleForecast(
        forecasts   = models,
        vol_7d      = wt_avg("vol_7d"),
        vol_14d     = wt_avg("vol_14d"),
        vol_30d     = wt_avg("vol_30d"),
        current_vol = wt_avg("current_vol"),
        best_model  = best.model_name,
    )


def compute_realized_vol(
    hourly_prices: pd.Series,
    start: pd.Timestamp,
    end:   pd.Timestamp,
    annualize: bool = True,
) -> float:
    """
    Compute realized vol from hourly prices over [start, end].
    Returns annualized vol as decimal fraction (e.g. 0.75 = 75%).
    """
    window = hourly_prices[(hourly_prices.index >= start) &
                           (hourly_prices.index <  end)]
    if len(window) < 4:
        return np.nan
    log_rets = np.log(window / window.shift(1)).dropna()
    rv_hourly = float(log_rets.std())
    if annualize:
        return rv_hourly * np.sqrt(24 * 365)
    return rv_hourly


def rolling_garch_forecasts(
    hourly_prices: pd.Series,
    lookback_days: int = 180,
    refit_freq:    int = 7,       # refit every N days
    horizons:      list[int] = None,
) -> pd.DataFrame:
    """
    Produce a daily time series of GARCH forecasts using a rolling window.

    Args:
        hourly_prices: Hourly close prices with UTC timestamp index
        lookback_days: Training window in days
        refit_freq:    Days between model refits (refit is expensive)
        horizons:      Forecast horizons in days (default [7, 14, 30])

    Returns:
        DataFrame with columns: vol_7d, vol_14d, vol_30d, current_vol,
                                 best_model, persistence
        Indexed by date (one row per trading day)
    """
    if horizons is None:
        horizons = [7, 14, 30]

    # Convert to daily close prices then daily returns
    daily = hourly_prices.resample("D").last().dropna()
    daily_rets_pct = daily.pct_change().dropna() * 100

    rows    = []
    last_fc = None
    days    = daily_rets_pct.index

    for i, day in enumerate(days):
        if i < lookback_days:
            continue  # need full lookback window

        # Refit model every refit_freq days or on first day
        if (i - lookback_days) % refit_freq == 0 or last_fc is None:
            window_rets = daily_rets_pct.iloc[i - lookback_days:i].values
            fc = fit_garch_ensemble(window_rets)
            if fc is not None:
                last_fc = fc

        if last_fc is None:
            continue

        rows.append({
            "date":        day,
            "vol_7d":      last_fc.vol_7d,
            "vol_14d":     last_fc.vol_14d,
            "vol_30d":     last_fc.vol_30d,
            "current_vol": last_fc.current_vol,
            "best_model":  last_fc.best_model,
            "persistence": last_fc.forecasts[0].persistence if last_fc.forecasts else np.nan,
        })

        if (i % 30 == 0):
            logger.info(f"  GARCH rolling: day {i}/{len(days)}, "
                       f"vol_7d={last_fc.vol_7d*100:.1f}%")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")
    return df
