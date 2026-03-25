"""
MFI Momentum — Capital Flow Following

Methodology:
    The RSI+MFI indicator shows net capital flow direction. When money
    crosses from net outflow (red) to net inflow (green), and WaveTrend
    is in agreement (wt1 > wt2), it signals a regime shift worth riding.

    This is a macro-bias strategy — it tries to stay positioned in the
    direction of institutional capital flow rather than picking exact
    turning points. Works best on daily and 4H timeframes.

Entry: rsi_mfi crosses above 0 AND wt1 > wt2 (trend agreement)
Exit: rsi_mfi crosses below 0 OR wt_cross_down above zero
"""

import pandas as pd
from .base import MCBStrategy, ParamSpec


class MFIMomentumStrategy(MCBStrategy):
    id = "mfi_momentum"
    name = "MFI Momentum"
    description = (
        "Capital flow following strategy. Enters when the RSI+MFI indicator crosses "
        "from red (outflow) to green (inflow) with WaveTrend confirming bullish direction. "
        "Exits when money flow turns negative again. Best on 4H and daily charts."
    )
    param_specs = [
        ParamSpec(
            "mfi_threshold", "MFI Entry Threshold", "float", 0.0,
            min=-20.0, max=20.0, step=1.0,
            help="rsi_mfi must cross above this level to trigger entry"
        ),
        ParamSpec(
            "require_wt_bull", "Require WT1 > WT2", "bool", True,
            help="Only enter when WaveTrend is also in bullish alignment (wt1 above wt2)"
        ),
        ParamSpec(
            "mfi_exit_threshold", "MFI Exit Threshold", "float", 0.0,
            min=-20.0, max=20.0, step=1.0,
            help="Exit when rsi_mfi drops below this level"
        ),
        ParamSpec(
            "ob_stop", "OB Stop Level", "float", 70.0,
            min=50.0, max=100.0, step=1.0,
            help="Also exit if WT2 reaches extreme overbought (take profit)"
        ),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mfi_thresh      = self.params["mfi_threshold"]
        req_wt          = self.params["require_wt_bull"]
        mfi_exit_thresh = self.params["mfi_exit_threshold"]
        ob_stop         = self.params["ob_stop"]

        # MFI cross: previous bar below threshold, current bar above
        mfi_cross_up = (
            (df["rsi_mfi"].shift(1) <= mfi_thresh) &
            (df["rsi_mfi"] > mfi_thresh)
        )
        mfi_cross_down = (
            (df["rsi_mfi"].shift(1) >= mfi_exit_thresh) &
            (df["rsi_mfi"] < mfi_exit_thresh)
        )

        wt_bull = df["wt1"] > df["wt2"] if req_wt else pd.Series(True, index=df.index)

        entry  = mfi_cross_up & wt_bull
        exit_s = mfi_cross_down | (df["wt2"] > ob_stop)

        labels = pd.Series("", index=df.index)
        labels[entry]              = "MFI CROSS UP ENTRY"
        labels[exit_s & ~entry]    = "MFI EXIT"

        df["entry"]        = entry
        df["exit_signal"]  = exit_s
        df["signal_label"] = labels
        return df
