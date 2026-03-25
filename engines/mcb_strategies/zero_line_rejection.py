"""
Zero Line Rejection — Trend Continuation Setup

Methodology:
    In a bullish trend (MFI green, WT came from below zero recently),
    waves pull back toward the zero line but do NOT reach OS extremes.
    A WT cross-up near zero = fresh momentum signal in trend direction.

This is a higher-frequency setup than Anchor & Trigger — it fires more
often and works well when there's clear directional momentum.

Entry: wt1 crosses wt2 upward, wt2 is in the zero-rejection zone
       (between os_band and zero_ceiling), MFI is green.
Exit: wt2 exceeds overbought level OR cross-down near zero.
"""

import pandas as pd
from .base import MCBStrategy, ParamSpec


class ZeroLineRejectionStrategy(MCBStrategy):
    id = "zero_line_rejection"
    name = "Zero Line Rejection"
    description = (
        "Trend continuation setup. Waves retrace toward zero (but not to OS extremes) "
        "and cross back up with green MFI — catching the resumption of a bullish trend "
        "rather than a full reversal from oversold. Higher frequency than Anchor & Trigger."
    )
    param_specs = [
        ParamSpec(
            "entry_floor", "Min WT2 for Entry", "float", -30.0,
            min=-55.0, max=-5.0, step=1.0,
            help="WT2 must be above this (not too deep in OS zone — that's for Anchor & Trigger)"
        ),
        ParamSpec(
            "entry_ceiling", "Max WT2 for Entry", "float", 5.0,
            min=-15.0, max=20.0, step=1.0,
            help="WT2 must be below this (true zero-line rejection, not a mid-range cross)"
        ),
        ParamSpec(
            "require_green_mfi", "Require Green MFI", "bool", True,
            help="Only enter when money flow is bullish (rsi_mfi > 0)"
        ),
        ParamSpec(
            "ob_exit", "Overbought Exit Level", "float", 53.0,
            min=30.0, max=80.0, step=1.0,
            help="Exit when WT2 reaches this overbought level"
        ),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        floor   = self.params["entry_floor"]
        ceiling = self.params["entry_ceiling"]
        req_mfi = self.params["require_green_mfi"]
        ob_exit = self.params["ob_exit"]

        entry = (
            df["wt_cross_up"] &
            (df["wt2"] >= floor) &
            (df["wt2"] <= ceiling) &
            ((~req_mfi) | (df["rsi_mfi"] > 0))
        )

        exit_s = (df["wt2"] > ob_exit) | df["sell_dot"] | df["wt_cross_down"]

        labels = pd.Series("", index=df.index)
        labels[entry]  = "ZERO REJECTION ENTRY"
        labels[exit_s & ~entry] = "EXIT"

        df["entry"]        = entry
        df["exit_signal"]  = exit_s
        df["signal_label"] = labels
        return df
