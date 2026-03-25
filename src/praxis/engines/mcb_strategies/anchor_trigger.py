"""
Anchor & Trigger — Classic MCb Long Setup

Methodology (from official MCb documentation):
    1. First WT cross-up in OS zone = ANCHOR (absorbs worst selling pressure)
       → Do NOT enter here
    2. Waves reset slightly above zero
    3. Second WT cross-up back into OS zone = TRIGGER
       → Enter LONG on trigger, especially if MFI is green

The key insight: the anchor establishes the low; the trigger enters after
confirmation that buyers have returned. This avoids catching falling knives.

Exit: WT2 rises above zero OR a sell dot fires.
"""

import pandas as pd
from .base import MCBStrategy, ParamSpec


class AnchorTriggerStrategy(MCBStrategy):
    id = "anchor_trigger"
    name = "Anchor & Trigger"
    description = (
        "Classic MCb two-wave long setup. Waits for the anchor wave (first OS cross) "
        "then enters on the trigger wave (second OS cross with MFI confirmation). "
        "Avoids catching falling knives — the most conservative MCb long entry."
    )
    param_specs = [
        ParamSpec(
            "os_level", "Oversold Level", "float", -53.0,
            min=-80.0, max=-40.0, step=1.0,
            help="WT2 must be below this level for both anchor and trigger crosses"
        ),
        ParamSpec(
            "require_green_mfi", "Require Green MFI on Trigger", "bool", True,
            help="If enabled, trigger only fires when rsi_mfi > 0 (money flowing in)"
        ),
        ParamSpec(
            "reset_above", "Anchor Reset Threshold", "float", -20.0,
            min=-40.0, max=10.0, step=1.0,
            help="WT2 must climb above this level between anchor and trigger"
        ),
        ParamSpec(
            "exit_above", "Exit Zero Level", "float", 0.0,
            min=-20.0, max=30.0, step=1.0,
            help="Exit long when WT2 rises above this level"
        ),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        os_lvl       = self.params["os_level"]
        req_mfi      = self.params["require_green_mfi"]
        reset_above  = self.params["reset_above"]
        exit_above   = self.params["exit_above"]

        entry = pd.Series(False, index=df.index)
        exit_s = pd.Series(False, index=df.index)
        labels = pd.Series("", index=df.index)

        # State machine: track anchor
        anchor_seen = False
        anchor_reset = False   # did WT2 rise above reset_above after anchor?

        for i in range(1, len(df)):
            row = df.iloc[i]

            # State transition: once anchor seen, watch for reset
            if anchor_seen and not anchor_reset:
                if row["wt2"] > reset_above:
                    anchor_reset = True

            # --- Anchor detection ---
            if (not anchor_seen and
                    row["wt_cross_up"] and
                    row["wt2"] < os_lvl):
                anchor_seen = True
                anchor_reset = False
                labels.iloc[i] = "ANCHOR"
                continue

            # --- Trigger detection ---
            if (anchor_seen and anchor_reset and
                    row["wt_cross_up"] and
                    row["wt2"] < os_lvl):
                mfi_ok = (not req_mfi) or (row["rsi_mfi"] > 0)
                if mfi_ok:
                    entry.iloc[i] = True
                    labels.iloc[i] = "TRIGGER ENTRY"
                # Reset state regardless
                anchor_seen = False
                anchor_reset = False

            # --- Exit ---
            if row["wt2"] > exit_above or row["sell_dot"]:
                exit_s.iloc[i] = True
                if not labels.iloc[i]:
                    labels.iloc[i] = "WT ZERO EXIT" if row["wt2"] > exit_above else "SELL DOT EXIT"

        df["entry"]        = entry
        df["exit_signal"]  = exit_s
        df["signal_label"] = labels
        return df
