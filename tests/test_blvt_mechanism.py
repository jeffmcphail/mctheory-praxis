"""
Tests for engines.forced_trade.blvt_mechanism -- synthetic data only.

The T1 mechanism verdict rests on this detector, so it must show it can:
  * recover planted TRIGGERED rebalances (time to the minute, leverage before
    and after) through premium noise and stale token prints,
  * find NOTHING when q never changes,
  * find a planted DAILY reset at its hour, with the slope the mechanics
    predict, via the schedule regression -- and stay silent when rebalances
    are triggered or absent. (The mean-|d| boundary profile is descriptive
    only: it lacks power for small daily resets under premium noise.)

Run:  pytest tests/test_blvt_mechanism.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engines.forced_trade.blvt_mechanism import DetectParams, detect


def simulate(days=30, sigma=0.0012, sign=1, band=(1.25, 4.0), target=2.5,
             daily_reset_minute=None, rebalance=True, premium_sd=0.0015,
             stale_frac=0.2, seed=0):
    """1m perp path + leveraged-token market price.

    Triggered mode: q resets to `target` leverage when |L| leaves `band`.
    Scheduled mode (daily_reset_minute set): resets at that minute of each UTC
    day, and never otherwise. Market price = NAV * (1 + AR(1) premium), with a
    share of minutes left stale (no token print).
    """
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2021-01-01", periods=n, freq="1min", tz="UTC")
    F = 30000 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    V = np.empty(n)
    V[0] = 10.0
    q = sign * target * V[0] / F[0]
    events = []
    for t in range(1, n):
        V[t] = V[t - 1] + q * (F[t] - F[t - 1])
        L = q * F[t] / V[t]
        reset = False
        if rebalance and daily_reset_minute is None:
            reset = abs(L) < band[0] or abs(L) > band[1]
        elif rebalance:
            reset = (t % 1440) == daily_reset_minute
        if reset:
            q_new = sign * target * V[t] / F[t]
            events.append({"time": idx[t], "L_pre": L, "L_post": sign * target})
            q = q_new
    e = np.zeros(n)
    eps = rng.normal(0, premium_sd * np.sqrt(1 - 0.9 ** 2), n)
    for t in range(1, n):
        e[t] = 0.9 * e[t - 1] + eps[t]
    P = V * (1 + e)
    stale = rng.random(n) < stale_frac
    stale[0] = False
    P = pd.Series(np.where(stale, np.nan, P), index=idx).ffill()
    df = pd.DataFrame({"P": P, "tok_trades": 1, "F": F}, index=idx)
    return df, pd.DataFrame(events)


def _detectable(df, truth, p=DetectParams()):
    """Events need window_h hours of data on each side (a stated edge limit)."""
    w = pd.Timedelta(hours=p.window_h)
    return truth[(truth["time"] >= df.index[0] + w)
                 & (truth["time"] <= df.index[-1] - w)].reset_index(drop=True)


def test_recovers_planted_triggered_rebalances():
    # A narrower band than BLVT's 1.25-4x so a 60-day fixture triggers often;
    # the detector does not know the band.
    df, truth = simulate(days=60, sigma=0.0015, band=(2.0, 3.2), seed=3)
    truth = _detectable(df, truth)
    assert len(truth) >= 4, "fixture should trigger several rebalances"
    _, _, ev, _ = detect(df, +1)
    ev = ev[ev["significant"]].reset_index(drop=True)
    assert len(ev) == len(truth), (len(ev), len(truth))
    for _, t in truth.iterrows():
        dt = (ev["time"] - t["time"]).abs().min()
        assert dt <= pd.Timedelta(minutes=30), f"event at {t['time']} missed by {dt}"
        k = (ev["time"] - t["time"]).abs().idxmin()
        assert ev.loc[k, "L_pre"] == pytest.approx(t["L_pre"], rel=0.10)
        assert ev.loc[k, "L_post"] == pytest.approx(t["L_post"], rel=0.10)


def test_short_token_triggered_rebalances_found():
    df, truth = simulate(days=60, sigma=0.0015, sign=-1, seed=5)
    truth = _detectable(df, truth)
    assert len(truth) >= 4
    _, _, ev, _ = detect(df, -1)
    ev = ev[ev["significant"]].reset_index(drop=True)
    assert len(ev) == len(truth), (len(ev), len(truth))
    assert (ev["L_pre"] < 0).all() and (ev["L_post"] < 0).all()


def test_null_no_rebalance_no_events():
    df, truth = simulate(days=30, sigma=0.0006, rebalance=False, seed=11)
    assert truth.empty
    _, _, ev, _ = detect(df, +1)
    assert len(ev) == 0 or not ev["significant"].any(), ev


@pytest.mark.parametrize("sign,beta_theory", [(+1, 2.0), (-1, -4.0)])
def test_schedule_regression_finds_daily_reset_ftx_scale(sign, beta_theory):
    """Positive-control scale: 75 days, 3x, daily reset at 00:02 UTC. The
    reset must show at hour 0 with the slope the mechanics predict:
    L - 1 = 2 for the long token, -(L + 1) = -4 for the short one."""
    df, _ = simulate(days=75, sigma=0.0012, target=3.0, sign=sign,
                     daily_reset_minute=2, seed=7)
    prof = detect(df, sign)[3]
    z = prof["sched_z"]
    assert z.abs().idxmax() == 0
    assert np.sign(z.loc[0]) == np.sign(beta_theory) and abs(z.loc[0]) >= 4
    assert prof.loc[0, "sched_beta"] == pytest.approx(beta_theory, rel=0.3)


def test_schedule_regression_power_at_blvt_scale():
    """A year of a 2x SHORT token with a daily reset is found. (A 2x LONG
    token resets by only (L - 1) * r, and a single year gives |z| ~2-5 --
    a stated power limit, which is why real tokens are run over their full
    multi-year lives and UP/DOWN twins are read together.)"""
    df, _ = simulate(days=365, sigma=0.0007, target=2.0, sign=-1,
                     daily_reset_minute=2, seed=8)
    z = detect(df, -1)[3]["sched_z"]
    assert z.abs().idxmax() == 0 and z.loc[0] <= -4


@pytest.mark.parametrize("kw", [
    dict(days=365, sigma=0.0009, target=2.5, band=(2.0, 3.2)),   # triggered
    dict(days=365, sigma=0.0007, rebalance=False),                # null
])
def test_schedule_regression_silent_without_schedule(kw):
    df, _ = simulate(seed=8, **kw)
    z = detect(df, +1)[3]["sched_z"]
    assert (z.abs() < 4).all(), z.round(2).to_dict()


def test_boundary_profile_flat_when_triggered():
    df, truth = simulate(days=60, sigma=0.0015, band=(2.0, 3.2), seed=3)
    assert len(truth) >= 4
    prof = detect(df, +1)[3]
    assert (prof["z_vs_rest"] < 4).all(), prof["z_vs_rest"].round(2).to_dict()
