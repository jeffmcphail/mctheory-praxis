"""
engines/triple_barrier.py
==========================
Unified Triple Barrier exit framework (Lopez de Prado, Advances in Financial
Machine Learning, Chapter 3).

The generalized version implemented here unifies mean-reversion and momentum
exit strategies through a single parameterization:

    sl_pct    : hard stop loss below entry (e.g. 0.01 = 1%)
    tp_pct    : take-profit trigger (e.g. 0.02 = 2%)
    trail_pct : trailing stop width from high-water mark after TP is triggered
                0.0 → standard triple barrier (TP exits immediately)
                >0  → momentum mode: TP arms trailing stop, trail exits
    t_bars    : vertical barrier — max bars to hold regardless

Exit logic:
    trail_active = False
    hwm = entry_price

    for each bar:
        update hwm = max(hwm, current_price)   [only after trail_active]

        if not trail_active:
            if price <= entry * (1 - sl_pct):
                EXIT — stop loss
            if price >= entry * (1 + tp_pct):
                if trail_pct == 0:
                    EXIT — take profit (standard)
                else:
                    trail_active = True; hwm = price  [momentum: arm trailing]

        if trail_active:
            hwm = max(hwm, price)
            if price <= hwm * (1 - trail_pct):
                EXIT — trailing stop

        if bars_held >= t_bars:
            EXIT — vertical barrier (opportunity cost)

CPO grid design (targeting 2:1 to 3:1 tp/sl ratios):
    sl_pct:    [0.010, 0.020, 0.030]
    tp_pct:    2× or 3× sl  →  [0.020, 0.030, 0.040, 0.060, 0.090]
    trail_pct: [0.000, 0.003, 0.005, 0.010]  (0 = mean-reversion, >0 = momentum)
    t_bars:    strategy-dependent

The RF/CPO procedure discovers which combination is best per model per day —
mean-reversion models will naturally prefer trail=0, momentum models trail>0,
without those assumptions being hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ── Exit parameter dataclass ──────────────────────────────────────────────────

@dataclass
class BarrierConfig:
    """
    Complete exit specification for one trade.
    Combine with any entry signal to form a full strategy config.
    """
    sl_pct:    float = 0.020     # hard stop loss (fraction, e.g. 0.02 = 2%)
    tp_pct:    float = 0.040     # take-profit trigger (fraction)
    trail_pct: float = 0.000     # trailing stop width after TP (0 = direct TP exit)
    t_bars:    int   = 48        # vertical barrier in bars

    @property
    def tp_sl_ratio(self) -> float:
        return self.tp_pct / (self.sl_pct + 1e-10)

    def to_feature_vector(self) -> list[float]:
        """Normalized feature vector for RF input."""
        return [
            self.sl_pct    / 0.05,   # normalize to [0,1] assuming max 5%
            self.tp_pct    / 0.10,   # normalize to [0,1] assuming max 10%
            self.trail_pct / 0.02,   # normalize to [0,1] assuming max 2%
            self.t_bars    / 168.0,  # normalize to [0,1] assuming max 1 week
        ]

    @staticmethod
    def param_names() -> list[str]:
        return ["barrier_sl_norm", "barrier_tp_norm",
                "barrier_trail_norm", "barrier_t_bars_norm"]


# ── Standard exit grid ────────────────────────────────────────────────────────

def standard_barrier_grid(
    sl_values: list[float] | None = None,
    tp_ratios: list[float] | None = None,
    trail_values: list[float] | None = None,
    t_bars_values: list[int] | None = None,
) -> list[BarrierConfig]:
    """
    Generate the standard CPO exit parameter grid.

    Defaults target 2:1 to 3:1 TP/SL ratios with optional trailing stop.

    Default grid size: 3 sl × 2 tp_ratios × 4 trail × 3 t_bars = 72 configs
    """
    if sl_values is None:
        sl_values = [0.010, 0.020, 0.030]
    if tp_ratios is None:
        tp_ratios = [2.0, 3.0]
    if trail_values is None:
        trail_values = [0.000, 0.003, 0.005, 0.010]
    if t_bars_values is None:
        t_bars_values = [24, 48, 96]

    configs = []
    for sl in sl_values:
        for ratio in tp_ratios:
            tp = round(sl * ratio, 4)
            for trail in trail_values:
                # Trail must be <= sl (otherwise trail fires before any meaningful move)
                if trail > sl:
                    continue
                for t in t_bars_values:
                    configs.append(BarrierConfig(
                        sl_pct=sl, tp_pct=tp,
                        trail_pct=trail, t_bars=t,
                    ))
    return configs


# ── Core triple barrier simulation ───────────────────────────────────────────

ExitReason = Literal["stop_loss", "take_profit", "trailing_stop", "vertical_barrier"]


def simulate_trade(
    prices: np.ndarray,         # price series starting at (and including) entry bar
    direction: int,              # +1 = long, -1 = short
    barrier: BarrierConfig,
    tc_pct: float = 0.0002,     # one-way transaction cost (fraction)
) -> dict:
    """
    Simulate a single trade under the triple barrier framework.

    Parameters
    ----------
    prices    : 1-D price array. prices[0] is the entry price.
    direction : +1 = long, -1 = short
    barrier   : BarrierConfig (sl, tp, trail, t_bars)
    tc_pct    : one-way TC as fraction (e.g. 0.0002 = 2bps)

    Returns
    -------
    dict with keys:
        exit_idx     : bar index of exit (0-based from entry)
        exit_price   : price at exit
        exit_reason  : one of ExitReason literals
        gross_return : return before TC (signed, long-positive)
        net_return   : return after round-trip TC
        bars_held    : number of bars held
        label        : +1 if profitable, -1 if not (RF training target)
    """
    if len(prices) == 0:
        return _null_result()

    entry_price = prices[0]
    sl  = barrier.sl_pct
    tp  = barrier.tp_pct
    tr  = barrier.trail_pct
    max_bars = min(barrier.t_bars, len(prices) - 1)

    trail_active = False
    hwm = entry_price   # high-water mark (for long) or low-water mark (for short)

    for i in range(1, max_bars + 1):
        price = prices[i]

        # Price relative to entry
        if direction == 1:
            rel = (price - entry_price) / entry_price
            above_tp = rel >= tp
            below_sl = rel <= -sl
        else:  # short
            rel = (entry_price - price) / entry_price
            above_tp = rel >= tp
            below_sl = rel <= -sl

        # Update HWM once trailing is active
        if trail_active:
            if direction == 1:
                hwm = max(hwm, price)
                drawback = (hwm - price) / hwm
            else:
                hwm = min(hwm, price)
                drawback = (price - hwm) / hwm

            if drawback >= tr:
                return _make_result(
                    i, price, entry_price, direction,
                    "trailing_stop", tc_pct,
                )

        else:
            # Hard stop loss
            if below_sl:
                return _make_result(
                    i, price, entry_price, direction,
                    "stop_loss", tc_pct,
                )

            # Take profit
            if above_tp:
                if tr == 0.0:
                    # Standard triple barrier: exit at TP
                    return _make_result(
                        i, price, entry_price, direction,
                        "take_profit", tc_pct,
                    )
                else:
                    # Momentum mode: arm trailing stop
                    trail_active = True
                    hwm = price

        # Vertical barrier (last bar in the loop)
        if i == max_bars:
            return _make_result(
                i, price, entry_price, direction,
                "vertical_barrier", tc_pct,
            )

    # Fallback (should not reach here normally)
    return _make_result(
        len(prices) - 1, prices[-1], entry_price, direction,
        "vertical_barrier", tc_pct,
    )


def _make_result(
    exit_idx: int,
    exit_price: float,
    entry_price: float,
    direction: int,
    reason: ExitReason,
    tc_pct: float,
) -> dict:
    if direction == 1:
        gross = (exit_price - entry_price) / entry_price
    else:
        gross = (entry_price - exit_price) / entry_price

    net = gross - 2 * tc_pct  # round-trip TC

    return {
        "exit_idx":     exit_idx,
        "exit_price":   exit_price,
        "exit_reason":  reason,
        "gross_return": float(gross),
        "net_return":   float(net),
        "bars_held":    exit_idx,
        "label":        1 if net > 0 else -1,
    }


def _null_result() -> dict:
    return {
        "exit_idx":     0,
        "exit_price":   0.0,
        "exit_reason":  "vertical_barrier",
        "gross_return": 0.0,
        "net_return":   0.0,
        "bars_held":    0,
        "label":        -1,
    }


# ── Day-level simulation (multiple entries per day) ───────────────────────────

def run_day_with_barriers(
    prices: np.ndarray,
    signals: np.ndarray,
    barrier: BarrierConfig,
    direction: int = 1,
    tc_pct: float = 0.0002,
    one_position_at_a_time: bool = True,
) -> dict:
    """
    Run the triple barrier framework on a full day of prices + signals.

    Parameters
    ----------
    prices    : hourly (or sub-hourly) price array for the day
    signals   : entry signal array (+1=enter, 0=no signal) same length as prices
    barrier   : BarrierConfig
    direction : +1 = long-only, -1 = short-only, 0 = follow signal sign
    tc_pct    : one-way TC
    one_position_at_a_time : if True, skip new entries while in a position

    Returns
    -------
    dict with daily_return, gross_return, n_trades, win_trades, n_tp, n_sl, n_trail, n_vert
    """
    n = len(prices)
    equity = 1.0
    gross_total = 0.0
    n_trades = 0
    n_tp = n_sl = n_trail = n_vert = 0
    pos_end = -1  # bar index where current position ends

    for i in range(n):
        if one_position_at_a_time and i <= pos_end:
            continue

        sig = signals[i]
        if sig == 0 or np.isnan(sig):
            continue

        d = direction if direction != 0 else int(np.sign(sig))
        if d == 0:
            continue

        # Simulate trade from bar i onward
        remaining = prices[i:]
        result = simulate_trade(remaining, d, barrier, tc_pct)

        # Compound equity
        equity *= (1 + result["net_return"])
        gross_total += result["gross_return"]
        n_trades += 1
        pos_end = i + result["exit_idx"]

        r = result["exit_reason"]
        if r == "take_profit":     n_tp    += 1
        elif r == "stop_loss":     n_sl    += 1
        elif r == "trailing_stop": n_trail += 1
        else:                      n_vert  += 1

    win_trades = sum(
        1 for _ in range(n_trades)
    )  # approximation — will refine in strategy layers

    return {
        "daily_return": float(equity - 1.0),
        "gross_return": float(gross_total),
        "n_trades":     n_trades,
        "n_tp":         n_tp,
        "n_sl":         n_sl,
        "n_trail":      n_trail,
        "n_vert":       n_vert,
    }
