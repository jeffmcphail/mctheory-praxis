"""
Bullish Divergence — Reversal Setup

Methodology:
    Classic momentum divergence: price makes a lower low while WaveTrend
    makes a higher low. This indicates weakening selling pressure even as
    price extends lower — a high-conviction bottom signal.

    Entry: bull_div detected within the last N bars AND a wt_cross_up
    fires in the OS zone. The lookahead window handles the common case
    where the divergence prints 1-3 bars before the confirming cross.

Exit: WT2 crosses above zero line (take the reversion target).
"""

import pandas as pd
from .base import MCBStrategy, ParamSpec


class BullishDivergenceStrategy(MCBStrategy):
    id = "bullish_divergence"
    name = "Bullish Divergence"
    description = (
        "High-conviction reversal setup. Price makes lower low, WaveTrend makes "
        "higher low — classic divergence. Enters on the next WT cross-up in the OS "
        "zone within a lookback window after the divergence is detected."
    )
    param_specs = [
        ParamSpec(
            "os_level", "Min OS Level for Entry", "float", -40.0,
            min=-80.0, max=-20.0, step=1.0,
            help="WT2 must be below this for the cross to count"
        ),
        ParamSpec(
            "div_lookback", "Divergence Lookback Bars", "int", 5,
            min=1, max=20, step=1,
            help="Enter if a bull_div occurred within this many bars before the cross-up"
        ),
        ParamSpec(
            "exit_level", "Exit WT2 Level", "float", 0.0,
            min=-20.0, max=40.0, step=1.0,
            help="Exit when WT2 rises above this level"
        ),
        ParamSpec(
            "mfi_filter", "MFI Must Not Be Deeply Red", "bool", True,
            help="Skip entries when rsi_mfi < -50 (capitulation still ongoing)"
        ),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        os_lvl     = self.params["os_level"]
        lookback   = int(self.params["div_lookback"])
        exit_lvl   = self.params["exit_level"]
        mfi_filter = self.params["mfi_filter"]

        # Rolling window: was there a bull_div in the last `lookback` bars?
        recent_div = df["bull_div"].rolling(lookback, min_periods=1).max().astype(bool)

        entry = (
            df["wt_cross_up"] &
            (df["wt2"] < os_lvl) &
            recent_div
        )
        if mfi_filter:
            entry &= (df["rsi_mfi"] > -50)

        exit_s = (df["wt2"] > exit_lvl) | df["sell_dot"]

        labels = pd.Series("", index=df.index)
        labels[entry]              = "BULL DIV ENTRY"
        labels[exit_s & ~entry]    = "DIV TARGET EXIT"

        df["entry"]        = entry
        df["exit_signal"]  = exit_s
        df["signal_label"] = labels
        return df
