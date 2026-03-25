"""
Bullish Divergence — Reversal Setup

Methodology:
    Classic momentum divergence: price makes a lower low while WaveTrend
    makes a higher low. This indicates weakening selling pressure even as
    price extends lower — a high-conviction bottom signal.

    MCb plots a purple triangle when this condition is detected. We add
    the requirement that a WT cross-up actually fires in the OS zone to
    confirm the divergence (not just a divergence in progress).

Entry: bull_div flag is True AND wt_cross_up AND wt2 < os_level
       (The cross confirms the divergence is resolving, not just forming)
Exit: WT2 crosses above zero line (take the reversion target)
"""

import pandas as pd
from .base import MCBStrategy, ParamSpec


class BullishDivergenceStrategy(MCBStrategy):
    id = "bullish_divergence"
    name = "Bullish Divergence"
    description = (
        "High-conviction reversal setup. Price makes lower low, WaveTrend makes "
        "higher low — classic divergence. Enters only when the WT cross-up confirms "
        "the divergence is resolving. Targets the zero line as the reversion destination."
    )
    param_specs = [
        ParamSpec(
            "os_level", "Min OS Level for Entry", "float", -40.0,
            min=-80.0, max=-20.0, step=1.0,
            help="WT2 must be below this for the divergence cross to count"
        ),
        ParamSpec(
            "confirm_with_buy_dot", "Require Green Buy Dot", "bool", False,
            help="Also require the standard green buy dot to fire (tighter filter)"
        ),
        ParamSpec(
            "exit_level", "Exit WT2 Level", "float", 0.0,
            min=-20.0, max=40.0, step=1.0,
            help="Exit when WT2 rises above this level (0 = zero line)"
        ),
        ParamSpec(
            "mfi_filter", "MFI Must Not Be Deeply Red", "bool", True,
            help="Skip divergence entries when rsi_mfi < -50 (capitulation is still ongoing)"
        ),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        os_lvl      = self.params["os_level"]
        req_dot     = self.params["confirm_with_buy_dot"]
        exit_lvl    = self.params["exit_level"]
        mfi_filter  = self.params["mfi_filter"]

        div_signal = df["bull_div"] & df["wt_cross_up"] & (df["wt2"] < os_lvl)
        if req_dot:
            div_signal &= df["buy_dot"]
        if mfi_filter:
            div_signal &= (df["rsi_mfi"] > -50)

        exit_s = (df["wt2"] > exit_lvl) | df["sell_dot"]

        labels = pd.Series("", index=df.index)
        labels[div_signal] = "BULL DIV ENTRY"
        labels[exit_s & ~div_signal] = "DIV TARGET EXIT"

        df["entry"]        = div_signal
        df["exit_signal"]  = exit_s
        df["signal_label"] = labels
        return df
