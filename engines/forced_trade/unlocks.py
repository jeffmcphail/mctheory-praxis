"""
engines/forced_trade/unlocks.py

T2 -- Scenario F1 (token unlock cliffs): do `circulating_supply` jumps mark
unlock events?

THE LOAD-BEARING QUESTION
-------------------------
Not "are there jumps" but "are the jumps CLIFFS or DRIFT".

    CLIFF  -- a vesting contract releases a tranche on a contractual date. The
              recipients did not choose the timing. That is compulsion, and it
              is what scenario F1 claims to trade.
    DRIFT  -- block rewards / staking emissions add supply continuously. Nobody
              is compelled to do anything on any particular day. There is no
              event to trade.

Both show up as `circulating_supply` increasing. Only the first is a
forced-trade scenario, so this module separates them explicitly rather than
counting "supply increase events" and calling them unlocks.

THE SEPARATION TEST
-------------------
For each asset, per-period fractional supply change is computed, and each
increase above `jump_pct` is scored against the asset OWN recent change
distribution:

    cliff_score = (jump - median_recent_change) / MAD(recent_change)

A tranche release is a large multiple of the background emission rate; smooth
emission is not. The distribution of cliff scores, and the share of total supply
growth arriving in cliff-scored periods versus arriving smoothly, is the answer.

DATING PRECISION
----------------
`market_data` is a DAILY snapshot table (Rule 35 conformant: `timestamp` in ms,
`date` as the cached rendering). An unlock can therefore be dated no more
precisely than to the sampling interval, and if the collector skips days the
true resolution is worse than daily. `sampling_report` measures the ACTUAL
inter-observation gaps rather than assuming the nominal cadence, because a
detector that dates events to the day on a series with 3-day gaps is reporting
false precision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engines.forced_trade.common import (
    DEFAULT_DB, MS_PER_DAY, days_between, ms_to_str, per_year, read_only_db,
)

logger = logging.getLogger("forced_trade.unlocks")


@dataclass(frozen=True)
class UnlockParams:
    """All thresholds. Defaults chosen before inspecting output."""
    jump_pct: float = 0.01        # min single-period fractional supply increase
    cliff_lookback: int = 30      # periods for the background-change reference
    cliff_mad_mult: float = 5.0   # cliff if jump exceeds median + this many MADs
    min_periods: int = 10         # asset needs this many observations to score

    def label(self) -> str:
        return (f"jump>{self.jump_pct:.2%} cliff>median+{self.cliff_mad_mult:g}MAD "
                f"(lookback={self.cliff_lookback})")


DEFAULT_SWEEP = (
    ("loose",  UnlockParams(jump_pct=0.002, cliff_mad_mult=3.0)),
    ("base",   UnlockParams(jump_pct=0.010, cliff_mad_mult=5.0)),
    ("strict", UnlockParams(jump_pct=0.030, cliff_mad_mult=8.0)),
)


# ================================================================= loading ==

def load_supply(db_path=DEFAULT_DB) -> pd.DataFrame:
    """Full `market_data` supply series, one row per (asset, timestamp)."""
    with read_only_db(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT asset, timestamp, date, circulating_supply, total_supply, "
            "market_cap FROM market_data ORDER BY asset, timestamp", conn)
    logger.info("loaded market_data: %d rows, %d assets",
                len(df), df["asset"].nunique())
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-asset usable span, sampling cadence and NULL rate.

    This is T2 question one, and it gates everything else: a supply series with
    100 daily points cannot support an event study of a scenario whose events
    are quarterly.
    """
    rows = []
    for asset, g in df.groupby("asset"):
        g = g.sort_values("timestamp")
        ts = g["timestamp"].values.astype("int64")
        gaps = np.diff(ts) / MS_PER_DAY if len(ts) > 1 else np.array([np.nan])
        cs = g["circulating_supply"]
        rows.append({
            "asset": asset,
            "rows": len(g),
            "first": ms_to_str(int(ts[0]), date_only=True),
            "last": ms_to_str(int(ts[-1]), date_only=True),
            "span_days": round(days_between(int(ts[0]), int(ts[-1])), 1),
            "circ_nonnull": int(cs.notna().sum()),
            "circ_null_pct": round(100.0 * cs.isna().mean(), 1),
            "median_gap_days": round(float(np.nanmedian(gaps)), 2),
            "max_gap_days": round(float(np.nanmax(gaps)), 2),
            "n_gaps_over_2d": int(np.nansum(gaps > 2.0)),
            "distinct_values": int(cs.nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values("asset").reset_index(drop=True)


# ================================================================ detector ==

def detect_jumps(df: pd.DataFrame, p: UnlockParams) -> pd.DataFrame:
    """Per-asset supply increases above `jump_pct`, each scored cliff-vs-drift."""
    out = []
    for asset, g in df.groupby("asset"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        cs = g["circulating_supply"].astype(float)
        if cs.notna().sum() < p.min_periods:
            logger.warning("[%s] only %d non-null supply observations -- skipped "
                           "(min_periods=%d)", asset, int(cs.notna().sum()),
                           p.min_periods)
            continue

        chg = cs.pct_change()
        # Background change: trailing, excluding the observation being scored.
        med = chg.rolling(p.cliff_lookback, min_periods=3).median().shift(1)
        mad = (chg - med).abs().rolling(p.cliff_lookback, min_periods=3).median().shift(1)

        hit = chg > p.jump_pct
        for i in np.flatnonzero(hit.fillna(False).values):
            m, a = med.iloc[i], mad.iloc[i]
            # THREE CASES, and conflating them gets the answer backwards.
            #  a > 0   -- ordinary scoring against a noisy background.
            #  a == 0  -- the background is PERFECTLY FLAT. A discrete step out
            #             of a flat series is the purest possible cliff, yet
            #             dividing by zero MAD would score it n/a and classify
            #             it as drift. It is scored +inf instead. This is
            #             exactly the XRP escrow-release pattern.
            #  a is NaN -- not enough trailing history to judge. UNKNOWN, and
            #             kept distinct from "measured and rejected".
            if pd.notna(a) and a > 0:
                score = float((chg.iloc[i] - m) / a)
                enough = True
            elif pd.notna(a) and a == 0:
                score = float("inf") if chg.iloc[i] > (m if pd.notna(m) else 0.0) else 0.0
                enough = True
            else:
                score = np.nan
                enough = False
            out.append({
                "asset": asset,
                "timestamp": int(g["timestamp"].iloc[i]),
                "date": g["date"].iloc[i],
                "prev_date": g["date"].iloc[i - 1] if i > 0 else None,
                "gap_days": round((int(g["timestamp"].iloc[i])
                                   - int(g["timestamp"].iloc[i - 1])) / MS_PER_DAY, 2)
                if i > 0 else np.nan,
                "supply_before": float(cs.iloc[i - 1]) if i > 0 else np.nan,
                "supply_after": float(cs.iloc[i]),
                "jump_frac": float(chg.iloc[i]),
                "background_median": float(m) if pd.notna(m) else np.nan,
                "background_mad": float(a) if pd.notna(a) else np.nan,
                "cliff_score": score,
                "scorable": enough,
                "is_cliff": bool(enough and pd.notna(score)
                                 and score > p.cliff_mad_mult),
            })
    cols = ["asset", "timestamp", "date", "prev_date", "gap_days", "supply_before",
            "supply_after", "jump_frac", "background_median", "background_mad",
            "cliff_score", "scorable", "is_cliff"]
    return pd.DataFrame(out, columns=cols)


def growth_decomposition(df: pd.DataFrame, jumps: pd.DataFrame) -> pd.DataFrame:
    """Cliff vs drift, as a share of TOTAL observed supply growth per asset.

    This is the answer to "which pattern dominates". Counting jump EVENTS is not
    enough: three cliffs among 200 smooth days could still be where all the
    supply arrives, or could be rounding noise on a smooth emission curve.
    """
    rows = []
    for asset, g in df.groupby("asset"):
        g = g.sort_values("timestamp")
        cs = g["circulating_supply"].astype(float)
        if cs.notna().sum() < 2:
            continue
        chg = cs.pct_change()
        total_growth = float(cs.iloc[-1] / cs.iloc[0] - 1) if cs.iloc[0] else np.nan
        pos = chg[chg > 0]
        aj = jumps[jumps["asset"] == asset]
        cliff_growth = float(aj.loc[aj["is_cliff"], "jump_frac"].sum())
        jump_growth = float(aj["jump_frac"].sum())
        rows.append({
            "asset": asset,
            "n_obs": int(cs.notna().sum()),
            "total_growth_pct": round(100.0 * total_growth, 4)
            if pd.notna(total_growth) else np.nan,
            "sum_positive_chg_pct": round(100.0 * float(pos.sum()), 4),
            "n_periods_increase": int((chg > 0).sum()),
            "n_periods_flat": int((chg == 0).sum()),
            "n_periods_decrease": int((chg < 0).sum()),
            "median_daily_chg_pct": round(100.0 * float(chg.median()), 6)
            if chg.notna().any() else np.nan,
            "max_daily_chg_pct": round(100.0 * float(chg.max()), 4)
            if chg.notna().any() else np.nan,
            "n_jumps": int(len(aj)),
            "n_cliffs": int(aj["is_cliff"].sum()) if len(aj) else 0,
            "jump_share_of_growth": round(jump_growth / float(pos.sum()), 4)
            if float(pos.sum()) > 0 else np.nan,
            "cliff_share_of_growth": round(cliff_growth / float(pos.sum()), 4)
            if float(pos.sum()) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def verdict(cov: pd.DataFrame, jumps_by_setting: dict, decomp: pd.DataFrame,
            pattern_setting: str = "loose") -> dict:
    """Plain read on F1 from what was measured.

    `pattern_setting` names WHICH jump set the cliff-vs-drift call is made on,
    because at the base and strict thresholds the jump set is EMPTY and a
    verdict derived from an empty set is degenerate (0/x = 0 reads as DRIFT for
    the wrong reason). The honest statement is: no jump clears the base
    threshold at all, and the pattern call is therefore made on the loose
    candidate set with the threshold named.
    """
    n_assets = int(len(cov))
    max_span = float(cov["span_days"].max()) if len(cov) else 0.0
    counts = {k: {"jumps": int(len(v)),
                  "cliffs": int(v["is_cliff"].sum()) if len(v) else 0}
              for k, v in jumps_by_setting.items()}
    j = jumps_by_setting.get(pattern_setting, pd.DataFrame())
    n_cliffs = int(j["is_cliff"].sum()) if len(j) else 0

    dominant, basis = "n/a", "no jumps detected at any setting"
    if len(decomp):
        share = decomp["cliff_share_of_growth"].dropna()
        if len(share) and len(j):
            dominant = "CLIFF" if float(share.median()) > 0.5 else "DRIFT"
            basis = (f"median cliff share of positive supply growth = "
                     f"{float(share.median()):.3f} at setting '{pattern_setting}'")
    return {
        "assets_with_supply": n_assets,
        "max_span_days": max_span,
        "counts_by_setting": counts,
        "pattern_setting": pattern_setting,
        "n_cliffs_at_pattern_setting": n_cliffs,
        "cliffs_per_year_all_assets": round(per_year(n_cliffs, max_span), 1)
        if max_span > 0 else float("nan"),
        "dominant_pattern": dominant,
        "basis": basis,
    }
