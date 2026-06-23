"""
engines/infobar_lstm/benchmarks.py -- the controls a directional result must beat.

Per protocol section 6.2, every model result is judged OOS, net of cost, against:
  1. buy-and-hold (and its OWN Sharpe -- the beta control)
  2. a naive momentum/MA baseline at comparable turnover
  3. a null at the same trade rate (random long/flat, distribution over seeds)
  4. beta isolation: up-trend vs down-trend sub-periods evaluated separately

A long/flat model in an up-window looks brilliant from beta alone; these
controls isolate skill from latent long beta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import StratResult, evaluate_strategy, daily_sharpe


def buy_hold(bars: pd.DataFrame, rt_bps: float = 0.0) -> StratResult:
    """Always-long benchmark (one entry; ~zero turnover)."""
    sig = np.ones(len(bars))
    return evaluate_strategy(bars, sig, rt_bps=rt_bps)


def momentum_signal(bars: pd.DataFrame, window: int = 20) -> np.ndarray:
    """Causal MA baseline: long when close_i > MA(close, window)_i, else flat.

    Uses a left-closed rolling mean (no future leakage). Decision at bar i close.
    """
    close = bars["close"].astype(float)
    ma = close.rolling(window).mean()
    sig = (close > ma).astype(float)
    sig[ma.isna()] = 0.0
    return sig.to_numpy()


def null_at_rate(bars: pd.DataFrame, target_changes: int, rt_bps: float,
                 n_seeds: int = 200, base_seed: int = 0) -> dict:
    """Distribution of net Sharpe for random long/flat signals matched to the
    reference strategy's TURNOVER (same number of position changes).

    Per-bar-independent randomness would churn far more than a position-holding
    strategy and so would not be a fair "null at the same trade rate" (protocol
    section 6.2 #3). Here we place exactly `target_changes` random change-points
    and alternate long/flat between them, so the null pays the same cost the
    strategy does and only the signal's INFORMATION is nulled. Deterministic
    per draw; returns mean / 5th / 95th pct of the null Sharpe.
    """
    n = len(bars)
    n_cp = int(min(max(target_changes, 0), n - 1))
    sharpes = np.empty(n_seeds)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        sig = np.zeros(n)
        if n_cp > 0:
            cps = np.sort(rng.choice(np.arange(1, n), size=n_cp, replace=False))
            state = int(rng.random() < 0.5)
            prev = 0
            for cp in cps:
                sig[prev:cp] = state
                state ^= 1
                prev = cp
            sig[prev:] = state
        res = evaluate_strategy(bars, sig, rt_bps=rt_bps)
        sharpes[s] = res.sharpe_net
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return {"mean": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(sharpes)),
        "p05": float(np.percentile(sharpes, 5)),
        "p95": float(np.percentile(sharpes, 95)),
        "n": int(sharpes.size),
    }


def trend_subperiods(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for up-trend vs down-trend bars (by buy-hold drift).

    A bar is 'up' if the forward open-to-open return is positive. Used to
    evaluate a long/flat strategy in up AND down regimes separately so latent
    long beta cannot masquerade as skill.
    """
    opn = bars["open"].to_numpy(dtype=float)
    fwd = np.full(len(bars), np.nan)
    fwd[:-1] = opn[1:] / opn[:-1] - 1.0
    up = fwd > 0
    down = fwd < 0
    return up, down


def subperiod_sharpe(strat: StratResult, mask: np.ndarray,
                     end_ts_ms: np.ndarray) -> float:
    """Daily-aggregated net Sharpe restricted to a sub-period mask."""
    r = np.asarray(strat.net_ret, dtype=float)[: len(mask)].copy()
    ts = np.asarray(end_ts_ms, dtype=float)[: len(mask)]
    r = np.where(mask, r, np.nan)          # keep only masked bars
    return daily_sharpe(r, ts)
