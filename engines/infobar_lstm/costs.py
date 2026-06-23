"""
engines/infobar_lstm/costs.py -- long/flat strategy P&L, turnover, spot cost.

Execution model (protocol P1 section 1.3): a signal is decided at bar i's CLOSE
and executed at bar i+1's OPEN (strictly after the signal -> causal, not
same-instant-optimistic). Position held during bar j is the decision made at
bar j-1. Cost is charged per position CHANGE (each one-way leg) at a spot
one-way rate; a full enter+exit cycle is two legs = one round trip.

The atlas pass/fail line (Exp 1): gross_alpha > TC * turnover. Turnover is
reported loudly so this is always visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Spot round-trip baselines (protocol P1 section 1.2), maker-biased. Reported as
# a sensitivity sweep; one-way = RT / 2.
RT_BPS_SWEEP = (20.0, 40.0, 60.0)
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def bars_per_year(bars: pd.DataFrame) -> float:
    """Estimate bars/year from median bar duration (irregular event clock)."""
    dur_ms = (bars["end_timestamp"] - bars["start_timestamp"]).astype(float)
    med = float(np.median(dur_ms[dur_ms > 0])) if (dur_ms > 0).any() else np.nan
    if not np.isfinite(med) or med <= 0:
        return float("nan")
    return SECONDS_PER_YEAR / (med / 1000.0)


def daily_sharpe(per_bar_ret: np.ndarray, end_ts_ms: np.ndarray,
                 ann_days: float = 365.25) -> float:
    """Sharpe of the DAILY-aggregated return series, annualized by sqrt(365).

    Info bars close on an irregular event clock (~tens of seconds), so a
    per-bar Sharpe annualized by sqrt(bars/year) is uninterpretable. The
    standard, comparable frame is to sum per-bar returns into UTC calendar
    days and take Sharpe on the daily series. NaN-safe.
    """
    r = np.asarray(per_bar_ret, dtype=float)
    ts = np.asarray(end_ts_ms, dtype=float)
    m = np.isfinite(r) & np.isfinite(ts)
    r, ts = r[m], ts[m]
    if r.size < 2:
        return float("nan")
    day = np.floor(ts / 86_400_000.0).astype(np.int64)
    _, inv = np.unique(day, return_inverse=True)
    daily = np.zeros(inv.max() + 1)
    np.add.at(daily, inv, r)
    if daily.size < 2:
        return float("nan")
    sd = daily.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(daily.mean() / sd * np.sqrt(ann_days))


@dataclass
class StratResult:
    net_ret: np.ndarray        # per-bar net return series
    gross_ret: np.ndarray
    n_bars: int
    n_changes: int             # one-way position changes (legs)
    long_rate: float           # fraction of bars held long
    turnover_per_100: float    # legs per 100 bars
    turnover_per_day: float
    gross_cum: float
    net_cum: float
    sharpe_net: float
    sharpe_gross: float
    rt_bps: float


def evaluate_strategy(bars: pd.DataFrame, signal: np.ndarray,
                      rt_bps: float) -> StratResult:
    """Score a long/flat signal on `bars`.

    signal[i] in {0,1} is the position decided at bar i's close (held during
    bar i+1). open[i] is the executable price at the start of bar i. Sharpe is
    computed on the DAILY-aggregated return series (daily_sharpe).
    """
    opn = bars["open"].to_numpy(dtype=float)
    end_ts = bars["end_timestamp"].to_numpy(dtype=float)
    sig = np.asarray(signal, dtype=float)
    n = len(bars)

    # Position held during bar j = decision at j-1 (shift forward by 1).
    held = np.zeros(n)
    held[1:] = sig[:-1]

    # Bar j return from open[j] -> open[j+1]; last bar has no forward open.
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:-1] = opn[1:] / opn[:-1] - 1.0

    gross = held * fwd_ret
    # Position change at bar j (a leg executed at open[j]).
    change = np.abs(np.diff(np.concatenate([[0.0], held])))
    tc_oneway = (rt_bps / 2.0) / 1e4
    cost = change * tc_oneway
    net = gross - cost

    valid = np.isfinite(fwd_ret)
    n_eff = int(valid.sum())
    n_changes = int(change.sum())
    day_ms = 86_400_000.0
    span_days = max(
        (float(bars["end_timestamp"].iloc[-1] - bars["start_timestamp"].iloc[0])
         / day_ms), 1e-9)

    return StratResult(
        net_ret=net,
        gross_ret=gross,
        n_bars=n_eff,
        n_changes=n_changes,
        long_rate=float(np.nanmean(held[valid])) if n_eff else float("nan"),
        turnover_per_100=100.0 * n_changes / max(n_eff, 1),
        turnover_per_day=n_changes / span_days,
        gross_cum=float(np.nansum(gross)),
        net_cum=float(np.nansum(net)),
        sharpe_net=daily_sharpe(net, end_ts),
        sharpe_gross=daily_sharpe(gross, end_ts),
        rt_bps=rt_bps,
    )
